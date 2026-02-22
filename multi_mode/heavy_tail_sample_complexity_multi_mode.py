#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.lines import Line2D
from scipy.stats import wasserstein_distance


TARGET_CENTERS = np.array([0.0, 25.0], dtype=np.float64)
TARGET_WEIGHTS = np.array([0.6, 0.4], dtype=np.float64)
TARGET_MEAN = float(np.sum(TARGET_CENTERS * TARGET_WEIGHTS))
SIGMA_TARGET = 1.0
NU_TARGET = 3.0
K_MONTE_CARLO = 200

SOURCE_CONFIGS = [
    ("gaussian", None, r"$\mathcal{N}(0,1)$"),
    ("t1", 1.0, r"$\mathcal{T}_{1}(0,1)$"),
    ("t3", 3.0, r"$\mathcal{T}_{3}(0,1)$"),
    ("t5", 5.0, r"$\mathcal{T}_{5}(0,1)$"),
    ("t10", 10.0, r"$\mathcal{T}_{10}(0,1)$"),
]

DATASET_SIZES = [1024, 4096, 16384]
T_PLOT = [0.01, 0.05] + [round(v, 2) for v in np.arange(0.1, 1.0, 0.05)] + [0.99]
N_DIAGNOSTIC = 4096
PATH_START_POINTS = np.array([-10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40], dtype=np.float64)
PATH_DT = 0.01
ABLATION_N_SAMPLES = 10_000
EULERIAN_MC_SAMPLES = 4000
EULERIAN_TAIL_CANDIDATE_MULT = 80


@dataclass
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 256
    max_steps: int = 6000
    min_steps: int = 600
    patience: int = 600
    improve_tol: float = 1e-5


MODEL_HIDDEN = 64
MODEL_N_HIDDEN = 4
MODEL_TIME_EMBED_DIM = 8


class VelocityMLP(nn.Module):
    def __init__(self, hidden: int = 64, n_hidden: int = 4, time_embed_dim: int = 8) -> None:
        super().__init__()
        self.time_embed_dim = int(time_embed_dim)
        if self.time_embed_dim < 1:
            raise ValueError("time_embed_dim must be >= 1")

        layers: list[nn.Module] = []
        in_dim = 1 + self.time_embed_dim  # (x, time_embedding)
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.SiLU())
            in_dim = hidden
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        half = self.time_embed_dim // 2
        if half > 0:
            freqs = torch.arange(1, half + 1, device=t.device, dtype=t.dtype)[None, :]
            angles = 2.0 * math.pi * t * freqs
            emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        else:
            emb = torch.empty((t.shape[0], 0), device=t.device, dtype=t.dtype)
        if self.time_embed_dim % 2 == 1:
            emb = torch.cat([emb, t], dim=1)
        return emb[:, : self.time_embed_dim]

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        if x.ndim == 1:
            x = x[:, None]
        te = self._time_embedding(t.squeeze(1))
        return self.net(torch.cat([x, te], dim=1)).squeeze(1)


def build_model(device: torch.device, hidden: int = MODEL_HIDDEN, n_hidden: int = MODEL_N_HIDDEN, time_embed_dim: int = MODEL_TIME_EMBED_DIM) -> VelocityMLP:
    return VelocityMLP(hidden=hidden, n_hidden=n_hidden, time_embed_dim=time_embed_dim).to(device)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def student_t_pdf(x: np.ndarray, mu: float, sigma: float, nu: float) -> np.ndarray:
    z = (x - mu) / sigma
    log_coef = math.lgamma((nu + 1.0) / 2.0) - math.lgamma(nu / 2.0)
    log_coef -= 0.5 * math.log(nu * math.pi) + math.log(sigma)
    return np.exp(log_coef - 0.5 * (nu + 1.0) * np.log1p((z * z) / nu))


def mixture_student_t_pdf(
    x: np.ndarray,
    centers: np.ndarray = TARGET_CENTERS,
    weights: np.ndarray = TARGET_WEIGHTS,
    sigma: float = SIGMA_TARGET,
    nu: float = NU_TARGET,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    weights = weights / np.sum(weights)
    out = np.zeros_like(x, dtype=np.float64)
    for c, w in zip(centers, weights):
        out += float(w) * student_t_pdf(x, float(c), sigma, nu)
    return out


def sample_source(n: int, source_nu: float | None, rng: np.random.Generator) -> np.ndarray:
    if source_nu is None:
        return rng.normal(0.0, 1.0, size=n)
    return rng.standard_t(df=source_nu, size=n)


def sample_target(n: int, rng: np.random.Generator) -> np.ndarray:
    comp = rng.choice(len(TARGET_CENTERS), size=n, p=TARGET_WEIGHTS / np.sum(TARGET_WEIGHTS))
    return TARGET_CENTERS[comp] + SIGMA_TARGET * rng.standard_t(df=NU_TARGET, size=n)


def _normalize_logweights(logw: np.ndarray) -> np.ndarray:
    logw = logw - np.max(logw, axis=0, keepdims=True)
    w = np.exp(logw)
    denom = np.sum(w, axis=0, keepdims=True) + 1e-12
    return w / denom


def get_ground_truth_velocity(
    t: float,
    x: np.ndarray,
    rng: np.random.Generator,
    source_nu: float | None = None,
    centers: np.ndarray = TARGET_CENTERS,
    weights: np.ndarray = TARGET_WEIGHTS,
    nu_target: float = NU_TARGET,
    K: int = K_MONTE_CARLO,
) -> np.ndarray:
    """Double-latent Monte Carlo estimate of u*(t, x) for mixture Student-t target."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    centers = np.asarray(centers, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    weights = weights / np.sum(weights)
    t = float(np.clip(t, 1e-4, 1.0 - 1e-4))

    # Target latent: s1 ~ Gamma(nu_target/2, rate=nu_target/2)
    s1 = rng.gamma(shape=nu_target / 2.0, scale=2.0 / nu_target, size=K)
    k_idx = rng.choice(len(centers), size=K, p=weights)
    mu_k = centers[k_idx]

    # Source latent: Gaussian source => s0 = 1; Student-t source => Gamma latent.
    if source_nu is None:
        s0 = np.ones(K, dtype=np.float64)
    else:
        s0 = rng.gamma(shape=source_nu / 2.0, scale=2.0 / source_nu, size=K)

    xm = x[None, :] - t * mu_k[:, None]

    # D_k = (1-t)^2/s0_k + t^2/s1_k
    D = ((1.0 - t) ** 2) / s0[:, None] + (t**2) / s1[:, None]

    # Unnormalized importance weights:
    # w_k = 1/sqrt(D_k) * exp(-(x-t*mu)^2 / (2*D_k))
    logw = -0.5 * np.log(D) - (xm * xm) / (2.0 * D)
    w = _normalize_logweights(logw)

    # G_k = t/(s1_k * D_k)
    G = t / (s1[:, None] * D)

    # E_k = mu_k + G_k (x - t mu_k)
    E = mu_k[:, None] + G * xm

    E_hat = np.sum(w * E, axis=0)
    return (E_hat - x) / (1.0 - t + 1e-8)


def train_one_model(
    x0: np.ndarray,
    x1: np.ndarray,
    config: TrainConfig,
    device: torch.device,
    seed: int,
    time_sampling: str = "uniform",
    beta_alpha: float = 2.0,
    beta_beta: float = 1.0,
) -> tuple[VelocityMLP, list[float]]:
    torch.manual_seed(seed)
    model = build_model(device=device)
    optim = torch.optim.Adam(model.parameters(), lr=config.lr)

    x0_t = torch.tensor(x0, dtype=torch.float32, device=device)
    x1_t = torch.tensor(x1, dtype=torch.float32, device=device)

    n = x0_t.shape[0]
    bs = min(config.batch_size, n)
    indices = np.arange(n)

    loss_hist: list[float] = []
    best = float("inf")
    best_step = 0
    beta_dist = None
    if time_sampling == "beta":
        beta_dist = torch.distributions.Beta(
            torch.tensor(float(beta_alpha), device=device),
            torch.tensor(float(beta_beta), device=device),
        )

    for step in range(1, config.max_steps + 1):
        batch_idx = np.random.choice(indices, size=bs, replace=bs > n)
        b0 = x0_t[batch_idx]
        b1 = x1_t[batch_idx]

        if time_sampling == "uniform":
            t = torch.rand(bs, device=device)
        elif time_sampling == "beta":
            t = beta_dist.sample((bs,))  # type: ignore[union-attr]
        else:
            raise ValueError(f"Unknown time_sampling: {time_sampling}")
        x_t = (1.0 - t) * b0 + t * b1
        target_v = b1 - b0

        pred = model(t, x_t)
        loss = torch.mean((pred - target_v) ** 2)

        optim.zero_grad()
        loss.backward()
        optim.step()

        lv = float(loss.item())
        loss_hist.append(lv)

        if lv + config.improve_tol < best:
            best = lv
            best_step = step

        if step >= config.min_steps and (step - best_step) >= config.patience:
            break

    return model, loss_hist


def load_checkpoint_if_compatible(ckpt_path: str, device: torch.device) -> VelocityMLP | None:
    if not os.path.exists(ckpt_path):
        return None

    ckpt = torch.load(ckpt_path, map_location=device)
    arch = ckpt.get("model_arch", {})
    hidden = int(arch.get("hidden", MODEL_HIDDEN))
    n_hidden = int(arch.get("n_hidden", MODEL_N_HIDDEN))
    time_embed_dim = int(arch.get("time_embed_dim", MODEL_TIME_EMBED_DIM))
    model = build_model(device=device, hidden=hidden, n_hidden=n_hidden, time_embed_dim=time_embed_dim)

    try:
        model.load_state_dict(ckpt["model_state"], strict=True)
    except Exception:
        return None

    model.eval()
    return model


def euler_integrate(
    x0: np.ndarray,
    velocity_fn: Callable[[float, np.ndarray], np.ndarray],
    dt: float = 0.05,
) -> np.ndarray:
    x = x0.astype(np.float64).copy()
    n_steps = int(round(1.0 / dt))
    for k in range(n_steps):
        t = k * dt
        v = velocity_fn(t, x)
        x = x + dt * v
    return x


@torch.no_grad()
def model_velocity_fn(model: VelocityMLP, device: torch.device) -> Callable[[float, np.ndarray], np.ndarray]:
    model.eval()

    def _fn(t: float, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32, device=device)
            tt = torch.full((xt.shape[0],), float(t), dtype=torch.float32, device=device)
            return model(tt, xt).detach().cpu().numpy().astype(np.float64)

    return _fn


def make_ground_truth_velocity_fn(source_nu: float | None, seed: int) -> Callable[[float, np.ndarray], np.ndarray]:
    def _fn(t: float, x: np.ndarray) -> np.ndarray:
        # Deterministic per-time seeding keeps repeated calls consistent.
        step_seed = int((t + 1e-8) * 1_000_000)
        rng_local = np.random.default_rng(seed + 37 * step_seed + 19)
        return get_ground_truth_velocity(t, x, rng=rng_local, source_nu=source_nu)

    return _fn


def integrate_characteristic_paths(
    x_start: np.ndarray,
    velocity_fn: Callable[[float, np.ndarray], np.ndarray],
    dt: float = PATH_DT,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.arange(0.0, 1.0 + 1e-12, dt, dtype=np.float64)
    traj = np.zeros((len(x_start), len(times)), dtype=np.float64)
    x = np.asarray(x_start, dtype=np.float64).copy()
    traj[:, 0] = x
    for i, t in enumerate(times[:-1]):
        x = x + dt * velocity_fn(float(t), x)
        traj[:, i + 1] = x
    return times, traj


def estimate_weighted_l2_error(
    t: float,
    source_nu: float | None,
    model_fn: Callable[[float, np.ndarray], np.ndarray],
    n_samples: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    cand_n = max(EULERIAN_TAIL_CANDIDATE_MULT * n_samples, 100_000)
    x1_cand = sample_target(cand_n, rng=rng)
    q99 = np.quantile(x1_cand, 0.99)
    x1_tail = x1_cand[x1_cand > q99]
    if x1_tail.size >= n_samples:
        x1 = rng.choice(x1_tail, size=n_samples, replace=False)
    else:
        # Fallback: if tail count is too small, use the largest available samples.
        x1 = np.sort(x1_cand)[-n_samples:]
    x0 = sample_source(n_samples, source_nu=source_nu, rng=rng)
    x_t = (1.0 - t) * x0 + t * x1
    pred = model_fn(t, x_t)
    gt = get_ground_truth_velocity(
        t,
        x_t,
        rng=np.random.default_rng(seed + 97),
        source_nu=source_nu,
    )
    return float(np.mean((pred - gt) ** 2))


def hill_estimator(samples: np.ndarray, frac: float = 0.05, center: float = TARGET_MEAN) -> float:
    x = np.asarray(samples, dtype=np.float64) - center
    x = np.sort(x[x > 0.0])
    if len(x) < 20:
        return float("nan")

    k = max(10, int(len(x) * frac))
    if k >= len(x):
        k = len(x) - 1
    if k < 1:
        return float("nan")

    tail = x[-k:]
    xk = tail[0]
    if xk <= 0:
        return float("nan")
    logs = np.log((tail + 1e-12) / (xk + 1e-12))
    return float(k / (np.sum(logs) + 1e-12))


def right_tail_ratio(samples: np.ndarray) -> float:
    q999 = np.quantile(samples, 0.999)
    q99 = np.quantile(samples, 0.99)
    return float(q999 / (q99 + 1e-12))


def tail_wasserstein(samples: np.ndarray, ref: np.ndarray) -> float:
    thr = np.quantile(ref, 0.99)
    s_tail = samples[samples > thr]
    r_tail = ref[ref > thr]
    if len(s_tail) < 2:
        return 100.0
    return float(wasserstein_distance(s_tail, r_tail))


def plot_velocity_timestep_grid_for_dof(
    x_grid: np.ndarray,
    t_values: list[float],
    n: int,
    dof: int,
    models: dict[tuple[str, int], VelocityMLP],
    device: torch.device,
    out_path: str,
    seed: int,
) -> None:
    n_plots = len(t_values)
    ncols = 7
    nrows = int(math.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.8 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    mv_g = model_velocity_fn(models[("gaussian", n)], device)
    mv_t = model_velocity_fn(models[(f"t{dof}", n)], device)

    for i, t in enumerate(t_values):
        ax = axes[i]
        rng_g = np.random.default_rng(seed + int(round(10_000 * t)) + 11)
        rng_t = np.random.default_rng(seed + int(round(10_000 * t)) + 29 + 100 * dof)

        gt_g = get_ground_truth_velocity(t, x_grid, rng=rng_g, source_nu=None)
        gt_t = get_ground_truth_velocity(t, x_grid, rng=rng_t, source_nu=float(dof))
        pred_g = mv_g(t, x_grid)
        pred_t = mv_t(t, x_grid)
        l2_g = estimate_weighted_l2_error(
            t=t,
            source_nu=None,
            model_fn=mv_g,
            n_samples=EULERIAN_MC_SAMPLES,
            seed=seed + int(round(1_000_000 * t)) + 101,
        )
        l2_t = estimate_weighted_l2_error(
            t=t,
            source_nu=float(dof),
            model_fn=mv_t,
            n_samples=EULERIAN_MC_SAMPLES,
            seed=seed + int(round(1_000_000 * t)) + 313 + 1000 * dof,
        )

        # Draw learned velocities first, then ground truth on top for visibility.
        ax.plot(x_grid, pred_g, color="#1f77b4", lw=1.6, linestyle="--", alpha=0.95, label="Gaussian learned", zorder=2)
        ax.plot(
            x_grid,
            pred_t,
            color="#d62728",
            lw=1.6,
            linestyle="--",
            alpha=0.95,
            label=fr"t$_{{{dof}}}$ learned",
            zorder=2,
        )
        ax.plot(
            x_grid,
            gt_g,
            color="#1f77b4",
            lw=2.2,
            linestyle="-",
            alpha=0.95,
            label="Gaussian GT",
            zorder=3,
        )
        ax.plot(
            x_grid,
            gt_t,
            color="#d62728",
            lw=2.2,
            linestyle="-",
            alpha=0.95,
            label=fr"t$_{{{dof}}}$ GT",
            zorder=3,
        )
        # Reference lines for source mean and target component means.
        ax.axvline(0.0, color="gray", lw=0.8, alpha=0.35, linestyle="-", zorder=1, label="p0 mean")
        for j, mu in enumerate(TARGET_CENTERS):
            ax.axvline(
                float(mu),
                color="black",
                lw=0.8,
                alpha=0.45,
                linestyle="-",
                zorder=1,
                label="p1 means" if j == 0 else None,
            )
        ax.text(0.03, 0.92, f"t={t:.2f}", transform=ax.transAxes, fontsize=9)
        ax.text(
            0.03,
            0.75,
            f"Tail L2 G: {l2_g:.2e}\nTail L2 t{dof}: {l2_t:.2e}",
            transform=ax.transAxes,
            fontsize=7,
            bbox=dict(facecolor="white", alpha=0.60, edgecolor="none"),
        )
        ax.set_xlim(float(np.min(x_grid)), float(np.max(x_grid)))
        ax.grid(alpha=0.25)

    for j in range(n_plots, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        fontsize=12,
        frameon=False,
    )
    fig.supxlabel("x")
    fig.supylabel("velocity")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_characteristic_paths(
    times: np.ndarray,
    gt_traj: np.ndarray,
    learned_traj: np.ndarray,
    x_start: np.ndarray,
    source_name: str,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(float(np.min(x_start)), float(np.max(x_start)))

    for i, x0 in enumerate(x_start):
        color = cmap(norm(float(x0)))
        ax.plot(times, gt_traj[i], color=color, lw=1.6, linestyle="-", alpha=0.95)
        ax.plot(times, learned_traj[i], color=color, lw=1.6, linestyle="--", alpha=0.95)

    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.set_title(f"Characteristic Particle Paths ({source_name})")
    ax.grid(alpha=0.25)

    style_legend = [
        Line2D([0], [0], color="black", lw=1.8, linestyle="-", label="Ground Truth"),
        Line2D([0], [0], color="black", lw=1.8, linestyle="--", label="Learned"),
    ]
    ax.legend(handles=style_legend, loc="upper left")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("start point $x_0$")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_hybrid_velocity_fn(
    model_fn: Callable[[float, np.ndarray], np.ndarray],
    gt_fn: Callable[[float, np.ndarray], np.ndarray],
    mode: str,
) -> Callable[[float, np.ndarray], np.ndarray]:
    def _use_gt(t: float) -> bool:
        if mode == "pure_gt":
            return True
        if mode == "pure_model":
            return False
        if mode == "fix_early":
            return 0.0 <= t <= 0.5
        if mode == "fix_early_middle":
            return 0.1 <= t <= 0.6
        if mode == "fix_late":
            return 0.5 <= t <= 1.0
        raise ValueError(f"Unknown hybrid mode: {mode}")

    def _fn(t: float, x: np.ndarray) -> np.ndarray:
        return gt_fn(t, x) if _use_gt(float(t)) else model_fn(t, x)

    return _fn


def plot_ablation_metric_bars(
    metric_name: str,
    metric_title: str,
    source_labels: list[str],
    config_order: list[tuple[str, str]],
    ablation_metrics: dict[str, dict[str, dict[str, float]]],
    out_path: str,
    baseline_value: float | None = None,
    baseline_label: str = "Ground Truth Baseline",
    ylim: tuple[float, float] | None = None,
) -> None:
    xs = np.arange(len(source_labels))
    n_cfg = len(config_order)
    width = 0.15
    offsets = (np.arange(n_cfg) - (n_cfg - 1) / 2.0) * width
    colors = ["#7f7f7f", "#f4a261", "#2a9d8f", "#264653", "#e76f51"]

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    for i, (cfg_key, cfg_label) in enumerate(config_order):
        vals = [ablation_metrics[s][cfg_key][metric_name] for s in source_labels]
        ax.bar(xs + offsets[i], vals, width=width, color=colors[i % len(colors)], alpha=0.90, label=cfg_label)

    if baseline_value is not None:
        ax.axhline(
            float(baseline_value),
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.9,
            label=baseline_label,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(source_labels)
    ax.set_ylabel(metric_name)
    ax.set_title(metric_title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_histogram_comparison(
    ref_samples: np.ndarray,
    gt_samples: np.ndarray,
    model_samples: np.ndarray,
    out_path: str,
    x_min: float,
    x_max: float,
) -> None:
    x = np.linspace(x_min, x_max, 1200)
    pdf = mixture_student_t_pdf(x, centers=TARGET_CENTERS, weights=TARGET_WEIGHTS, sigma=SIGMA_TARGET, nu=NU_TARGET)

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.plot(
        x,
        pdf,
        color="black",
        lw=0.8,
        label=r"Analytical mixture $0.6\,\mathcal{T}_{\nu=3}(0)+0.4\,\mathcal{T}_{\nu=3}(25)$",
    )

    bins = np.linspace(x_min, x_max, 90)
    ax.hist(gt_samples, bins=bins, density=True, alpha=0.5, color="orange", label="Ground Truth Flow")
    ax.hist(model_samples, bins=bins, density=True, alpha=0.45, color="purple", label="Model Flow")

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_title("Density comparison")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_metric_bars(
    metric_name: str,
    metric_title: str,
    source_labels: list[str],
    size_items: list[int],
    metrics_gt: dict[str, dict[str, float]],
    metrics_model: dict[str, dict[int, dict[str, float]]],
    ref_value: float,
    out_path: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    xs = np.arange(len(source_labels))
    width = 0.18 if len(size_items) >= 3 else 0.24

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    gt_vals = [metrics_gt[s][metric_name] for s in source_labels]
    n_bars = 1 + len(size_items)
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2.0) * width
    ax.bar(xs + offsets[0], gt_vals, width=width, color="orange", alpha=0.85, label="Ground Truth Flow")

    model_colors = {1024: "#d5b3ff", 4096: "#9f78d1", 16384: "#5a2a83"}
    for i, n in enumerate(size_items):
        vals = [metrics_model[s][n][metric_name] for s in source_labels]
        ax.bar(
            xs + offsets[i + 1],
            vals,
            width=width,
            color=model_colors.get(n, None),
            alpha=0.9,
            label=f"Model n={n}",
        )

    ax.axhline(ref_value, linestyle="--", color="black", lw=1.0, label="Reference baseline")

    ax.set_xticks(xs)
    ax.set_xticklabels(source_labels)
    ax.set_title(metric_title)
    ax.set_ylabel(metric_name)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.2, axis="y")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def compute_metrics(samples: np.ndarray, ref_samples: np.ndarray) -> dict[str, float]:
    return {
        "hill": hill_estimator(samples, frac=0.05, center=TARGET_MEAN),
        "right_tail_ratio": right_tail_ratio(samples),
        "tail_wasserstein": tail_wasserstein(samples, ref_samples),
    }


def stable_name_seed(name: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(name)) % 100_000


def _format_hparam(v: float) -> str:
    s = f"{v:.6g}"
    return s.replace("-", "m").replace(".", "p")


def run_experiment(args: argparse.Namespace) -> None:
    if args.time_sampling == "uniform":
        time_tag = "uniform"
    elif args.time_sampling == "beta":
        if args.beta_alpha <= 0 or args.beta_beta <= 0:
            raise ValueError("--beta_alpha and --beta_beta must be > 0 when --time_sampling=beta")
        time_tag = f"beta_a{_format_hparam(args.beta_alpha)}_b{_format_hparam(args.beta_beta)}"
    else:
        raise ValueError(f"Unknown --time_sampling: {args.time_sampling}")

    centers_tag = "c" + "_".join(_format_hparam(float(c)) for c in TARGET_CENTERS.tolist())
    mean_tag = f"mu{_format_hparam(TARGET_MEAN)}"
    run_tag = f"multi_{mean_tag}_{centers_tag}_t{time_tag}"
    args.output_dir = os.path.join(args.output_root, run_tag)

    ensure_dir(args.output_dir)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    vel_dir = os.path.join(args.output_dir, "velocity_plots")
    lag_dir = os.path.join(args.output_dir, "lagrangian_paths")
    ablation_dir = os.path.join(args.output_dir, "ablation")
    sample_dir = os.path.join(args.output_dir, "samples")
    hist_dir = os.path.join(args.output_dir, "histograms")
    metric_dir = os.path.join(args.output_dir, "metrics")
    for d in [ckpt_dir, vel_dir, lag_dir, ablation_dir, sample_dir, hist_dir, metric_dir]:
        ensure_dir(d)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    base_rng = np.random.default_rng(args.seed)

    train_cfg = TrainConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        min_steps=args.min_steps,
        patience=args.patience,
        improve_tol=args.improve_tol,
    )

    source_items = SOURCE_CONFIGS
    size_items = DATASET_SIZES
    if args.quick:
        source_items = SOURCE_CONFIGS[:2]
        size_items = [1024, 4096]

    models: dict[tuple[str, int], VelocityMLP] = {}

    # Part 1: training + checkpoints
    print("[Part 1] Training models")
    for src_name, src_nu, _ in source_items:
        for n in size_items:
            ckpt_path = os.path.join(ckpt_dir, f"model_{src_name}_n{n}.pt")
            if args.reuse_checkpoints:
                cached = load_checkpoint_if_compatible(ckpt_path, device=device)
                if cached is not None:
                    models[(src_name, n)] = cached
                    print(f"[Part 1] Reusing checkpoint: {ckpt_path}")
                    continue

            rng = np.random.default_rng(base_rng.integers(1, 10**9))
            x0 = sample_source(n, src_nu, rng)
            x1 = sample_target(n, rng)

            model, losses = train_one_model(
                x0,
                x1,
                train_cfg,
                device=device,
                seed=args.seed + n,
                time_sampling=args.time_sampling,
                beta_alpha=args.beta_alpha,
                beta_beta=args.beta_beta,
            )
            models[(src_name, n)] = model

            payload = {
                "model_state": model.state_dict(),
                "source": src_name,
                "source_nu": src_nu,
                "n": n,
                "model_arch": {
                    "hidden": MODEL_HIDDEN,
                    "n_hidden": MODEL_N_HIDDEN,
                    "time_embed_dim": MODEL_TIME_EMBED_DIM,
                },
                "train_cfg": train_cfg.__dict__,
                "time_sampling": args.time_sampling,
                "beta_alpha": args.beta_alpha,
                "beta_beta": args.beta_beta,
                "loss_tail": losses[-50:],
            }
            torch.save(payload, ckpt_path)

    n_diag = N_DIAGNOSTIC if N_DIAGNOSTIC in size_items else max(size_items)

    # Task 1: Velocity visualization + weighted Eulerian L2 annotations at n=4096.
    print(f"[Task 1] Rendering velocity grids with weighted L2 error (n={n_diag})")
    x_grid = np.linspace(args.x_min, args.x_max, args.velocity_grid_points)
    for dof in [1, 3, 5, 10]:
        src_key = f"t{dof}"
        if ("gaussian", n_diag) not in models or (src_key, n_diag) not in models:
            continue
        out = os.path.join(vel_dir, f"velocity_timestep_grid_dof{dof}_n{n_diag}.pdf")
        plot_velocity_timestep_grid_for_dof(
            x_grid=x_grid,
            t_values=T_PLOT,
            n=n_diag,
            dof=dof,
            models=models,
            device=device,
            out_path=out,
            seed=args.seed,
        )

    # Task 2: Lagrangian characteristic paths at n=4096.
    print(f"[Task 2] Rendering characteristic paths (n={n_diag}, dt={PATH_DT})")
    for src_name, src_nu, _ in source_items:
        if (src_name, n_diag) not in models:
            continue
        gt_fn = make_ground_truth_velocity_fn(src_nu, seed=args.seed + 1000 + stable_name_seed(src_name))
        learned_fn = model_velocity_fn(models[(src_name, n_diag)], device)
        times, gt_traj = integrate_characteristic_paths(PATH_START_POINTS, gt_fn, dt=PATH_DT)
        _, learned_traj = integrate_characteristic_paths(PATH_START_POINTS, learned_fn, dt=PATH_DT)
        out = os.path.join(lag_dir, f"characteristic_paths_{src_name}_n{n_diag}.pdf")
        plot_characteristic_paths(
            times=times,
            gt_traj=gt_traj,
            learned_traj=learned_traj,
            x_start=PATH_START_POINTS,
            source_name=src_name,
            out_path=out,
        )

    # Part 3a: mass generation
    print("[Part 3a] Generating samples")
    n_eval = args.n_eval_samples
    ref_samples = sample_target(n_eval, np.random.default_rng(args.seed + 9001))
    np.save(os.path.join(sample_dir, "reference_target_samples.npy"), ref_samples)

    gt_samples_by_source: dict[str, np.ndarray] = {}
    model_samples_by_source_size: dict[str, dict[int, np.ndarray]] = {}

    for src_name, src_nu, _ in source_items:
        x0_eval = sample_source(
            n_eval,
            src_nu,
            np.random.default_rng(args.seed + stable_name_seed(src_name)),
        )

        def gt_v(t: float, x: np.ndarray) -> np.ndarray:
            step_seed = int((t + 1e-8) * 10_000)
            rng_local = np.random.default_rng(args.seed + 37 * step_seed + 19)
            return get_ground_truth_velocity(t, x, rng=rng_local, source_nu=src_nu)

        gt_final = euler_integrate(x0_eval, gt_v, dt=args.dt)
        gt_samples_by_source[src_name] = gt_final
        np.save(os.path.join(sample_dir, f"ground_truth_flow_{src_name}.npy"), gt_final)

        model_samples_by_source_size[src_name] = {}
        for n in size_items:
            mv = model_velocity_fn(models[(src_name, n)], device)
            model_final = euler_integrate(x0_eval, mv, dt=args.dt)
            model_samples_by_source_size[src_name][n] = model_final
            np.save(os.path.join(sample_dir, f"model_flow_{src_name}_n{n}.npy"), model_final)

    # Task 3: Temporal substitution ablation at n=4096.
    print(f"[Task 3] Running temporal substitution ablation (n={n_diag}, M={ABLATION_N_SAMPLES})")
    ablation_config_order: list[tuple[str, str]] = [
        ("pure_model", "Pure Model"),
        ("pure_gt", "Pure GT"),
        ("fix_early", "Fix Early [0, 0.5]"),
        ("fix_early_middle", "Fix Early-Mid [0.1, 0.6]"),
        ("fix_late", "Fix Late [0.5, 1]"),
    ]
    ablation_metrics: dict[str, dict[str, dict[str, float]]] = {}
    ref_ablation = sample_target(ABLATION_N_SAMPLES, np.random.default_rng(args.seed + 7777))
    ref_ablation_metrics = compute_metrics(ref_ablation, ref_ablation)
    np.save(os.path.join(ablation_dir, "reference_target_samples.npy"), ref_ablation)

    for src_name, src_nu, _ in source_items:
        if (src_name, n_diag) not in models:
            continue
        ablation_metrics[src_name] = {}
        x0_abl = sample_source(
            ABLATION_N_SAMPLES,
            src_nu,
            np.random.default_rng(args.seed + 999 + stable_name_seed(src_name)),
        )
        gt_fn = make_ground_truth_velocity_fn(src_nu, seed=args.seed + 5000 + stable_name_seed(src_name))
        model_fn = model_velocity_fn(models[(src_name, n_diag)], device)

        for cfg_key, _ in ablation_config_order:
            hybrid_fn = make_hybrid_velocity_fn(model_fn=model_fn, gt_fn=gt_fn, mode=cfg_key)
            final_samples = euler_integrate(x0_abl, hybrid_fn, dt=args.dt)
            np.save(os.path.join(ablation_dir, f"samples_{src_name}_{cfg_key}_n{n_diag}.npy"), final_samples)
            ablation_metrics[src_name][cfg_key] = compute_metrics(final_samples, ref_ablation)

    ablation_source_labels = [s[0] for s in source_items if s[0] in ablation_metrics]
    if len(ablation_source_labels) > 0:
        plot_ablation_metric_bars(
            metric_name="hill",
            metric_title=f"Temporal Substitution Ablation: Hill Estimator (n={n_diag})",
            source_labels=ablation_source_labels,
            config_order=ablation_config_order,
            ablation_metrics=ablation_metrics,
            out_path=os.path.join(ablation_dir, "ablation_bar_hill.pdf"),
            baseline_value=ref_ablation_metrics["hill"],
            baseline_label="GT Sample Metric",
        )
        plot_ablation_metric_bars(
            metric_name="tail_wasserstein",
            metric_title=f"Temporal Substitution Ablation: Tail Wasserstein (n={n_diag})",
            source_labels=ablation_source_labels,
            config_order=ablation_config_order,
            ablation_metrics=ablation_metrics,
            out_path=os.path.join(ablation_dir, "ablation_bar_tail_wasserstein.pdf"),
            ylim=(0.0, 2.0),
        )
        plot_ablation_metric_bars(
            metric_name="right_tail_ratio",
            metric_title=f"Temporal Substitution Ablation: Right Tail Ratio (n={n_diag})",
            source_labels=ablation_source_labels,
            config_order=ablation_config_order,
            ablation_metrics=ablation_metrics,
            out_path=os.path.join(ablation_dir, "ablation_bar_right_tail_ratio.pdf"),
            baseline_value=ref_ablation_metrics["right_tail_ratio"],
            baseline_label="GT Sample Metric",
            ylim=(0.0, 3.0),
        )

    # Part 3b: histogram figures
    print("[Part 3b] Rendering histogram comparisons")
    for src_name, _, _ in source_items:
        for n in size_items:
            out = os.path.join(hist_dir, f"hist_{src_name}_n{n}.pdf")
            plot_histogram_comparison(
                ref_samples=ref_samples,
                gt_samples=gt_samples_by_source[src_name],
                model_samples=model_samples_by_source_size[src_name][n],
                out_path=out,
                x_min=args.x_min,
                x_max=args.x_max,
            )

    # Part 3c: metrics + grouped bars
    print("[Part 3c] Computing metrics and rendering bars")
    metrics_gt: dict[str, dict[str, float]] = {}
    metrics_model: dict[str, dict[int, dict[str, float]]] = {}

    for src_name, _, _ in source_items:
        metrics_gt[src_name] = compute_metrics(gt_samples_by_source[src_name], ref_samples)
        metrics_model[src_name] = {}
        for n in size_items:
            metrics_model[src_name][n] = compute_metrics(model_samples_by_source_size[src_name][n], ref_samples)

    metrics_ref = compute_metrics(ref_samples, ref_samples)

    source_labels = [x[0] for x in source_items]
    plot_metric_bars(
        metric_name="hill",
        metric_title="Hill Estimator (Top 5%)",
        source_labels=source_labels,
        size_items=size_items,
        metrics_gt=metrics_gt,
        metrics_model=metrics_model,
        ref_value=metrics_ref["hill"],
        out_path=os.path.join(metric_dir, "bar_hill.pdf"),
    )
    plot_metric_bars(
        metric_name="right_tail_ratio",
        metric_title="Right Tail Ratio Q(0.999)/Q(0.99)",
        source_labels=source_labels,
        size_items=size_items,
        metrics_gt=metrics_gt,
        metrics_model=metrics_model,
        ref_value=metrics_ref["right_tail_ratio"],
        out_path=os.path.join(metric_dir, "bar_right_tail_ratio.pdf"),
        ylim=(0.0, 3.0),
    )
    plot_metric_bars(
        metric_name="tail_wasserstein",
        metric_title="Tail Wasserstein (x > Q_ref(0.99))",
        source_labels=source_labels,
        size_items=size_items,
        metrics_gt=metrics_gt,
        metrics_model=metrics_model,
        ref_value=metrics_ref["tail_wasserstein"],
        out_path=os.path.join(metric_dir, "bar_tail_wasserstein.pdf"),
        ylim=(0.0, 2.0),
    )

    summary = {
        "target_centers": TARGET_CENTERS.tolist(),
        "target_weights": TARGET_WEIGHTS.tolist(),
        "target_mean": TARGET_MEAN,
        "source_items": [s[0] for s in source_items],
        "dataset_sizes": size_items,
        "diagnostic_n": n_diag,
        "n_eval_samples": n_eval,
        "dt": args.dt,
        "time_sampling": args.time_sampling,
        "beta_alpha": args.beta_alpha,
        "beta_beta": args.beta_beta,
        "device": str(device),
        "metrics_gt": metrics_gt,
        "metrics_model": metrics_model,
        "metrics_reference": metrics_ref,
        "ablation_metrics": ablation_metrics,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        

    print(f"Finished. Outputs saved to: {args.output_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Heavy-tail flow matching sample-complexity experiment (multi-mode target)")
    p.add_argument("--output_root", type=str, default="./output")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--seed", type=int, default=66)

    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--min_steps", type=int, default=2000)
    p.add_argument("--patience", type=int, default=600)
    p.add_argument("--improve_tol", type=float, default=1e-5)

    p.add_argument("--n_eval_samples", type=int, default=20000)
    p.add_argument("--dt", type=float, default=0.025)
    p.add_argument("--time_sampling", type=str, choices=["uniform", "beta"], default="uniform")
    p.add_argument("--beta_alpha", type=float, default=2.0)
    p.add_argument("--beta_beta", type=float, default=1.0)
    p.add_argument("--x_min", type=float, default=-10.0)
    p.add_argument("--x_max", type=float, default=40.0)
    p.add_argument("--velocity_grid_points", type=int, default=500)
    p.add_argument("--reuse_checkpoints", action="store_true", default=True, help="reuse existing checkpoints when compatible")
    p.add_argument("--no_reuse_checkpoints", action="store_false", dest="reuse_checkpoints", help="always retrain models")
    p.add_argument("--quick", action="store_true", help="small smoke run")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
