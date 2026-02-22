# Single-Mode Heavy-Tail Experiment

Script:

- `single_mode/heavy_tail_sample_complexity_single_mode.py`

This script trains flow-matching velocity models for a **single-mode** Student-t target, compares them against a Monte Carlo ground-truth velocity field, and generates diagnostics/ablation plots.

## 1) Experiment definition

### Target distribution

Configured by constants in the script:

- `MU_TARGET`: target center
- `SIGMA_TARGET`: target scale
- `NU_TARGET`: target Student-t dof

Sampling form:

- `X1 = MU_TARGET + SIGMA_TARGET * T_nu`

### Source distributions

`SOURCE_CONFIGS`:

- Gaussian: `N(0,1)`
- Student-t: `nu in {1, 3, 5, 10}`

### Dataset sizes

`DATASET_SIZES` controls train sizes (current defaults: `1024, 4096, 16384`).

## 2) Core methods implemented

### Ground-truth velocity

`get_ground_truth_velocity(t, x, ...)` uses a double-latent Monte Carlo approximation with:

- target latent `s1 ~ Gamma(nu_target/2, nu_target/2)`
- source latent `s0 = 1` for Gaussian source, or `Gamma(nu_source/2, nu_source/2)` for Student-t source

Then for each latent sample:

- `D = (1-t)^2 / s0 + t^2 / s1`
- weight `w ∝ D^{-1/2} exp(-(x - t*mu)^2/(2D))`
- gain `G = t / (s1 * D)`
- conditional estimate `E = mu + G(x - t*mu)`
- final velocity `u*(t,x) = (E_hat - x)/(1-t)` where `E_hat` is weighted average over latent samples.

### Learned model

- MLP with time embedding (`VelocityMLP`)
- Conditional flow matching loss on interpolants `x_t = (1-t)x0 + tx1`
- Predicted velocity trained to match `x1 - x0`

### Time sampling

Training `t` sampling mode:

- `uniform`: `t ~ U(0,1)`
- `beta`: `t ~ Beta(alpha, beta)`

Use:

- `--time_sampling {uniform,beta}`
- `--beta_alpha`, `--beta_beta` when beta mode is selected.

## 3) Output directory naming

Pass `--output_root` (default `./output`).

Run folder is auto-generated as:

- `single_mu<mu>_t<sampler_tag>`

Examples:

- `single_mu4_tuniform`
- `single_mu4_tbeta_a5_b7`

## 4) CLI parameters

### I/O + runtime

- `--output_root`: root folder for generated run outputs
- `--device`: torch device string (default currently `cuda:0`)
- `--seed`: base random seed
- `--quick`: reduced run for sanity checks

### Training hyperparameters

- `--lr`
- `--batch_size`
- `--max_steps`
- `--min_steps`
- `--patience`
- `--improve_tol`

### Simulation / sampling

- `--n_eval_samples`: number of generated samples for evaluation
- `--dt`: Euler integration step size

### Time sampling

- `--time_sampling {uniform,beta}`
- `--beta_alpha`
- `--beta_beta`

### Visualization range

- `--x_min`
- `--x_max`
- `--velocity_grid_points`

### Checkpoint reuse

- `--reuse_checkpoints`
- `--no_reuse_checkpoints`

If reuse is enabled, matching checkpoints are loaded instead of retraining.

## 5) What outputs are generated

Inside one run folder (`./output/<run_tag>/`):

- `checkpoints/`
- `velocity_plots/`
- `lagrangian_paths/`
- `ablation/`
- `samples/`
- `histograms/`
- `metrics/`
- `summary.json`

### `checkpoints/`

- `model_<source>_n<n>.pt`

Contains:

- model state
- architecture metadata
- source info
- selected train config and time-sampling metadata

### `velocity_plots/`

- `velocity_timestep_grid_dof<dof>_n<n_diag>.pdf`

Each subplot is one `t` in `T_PLOT`.

Shown per subplot:

- Gaussian GT (solid blue)
- Gaussian learned (dashed blue)
- Student-t(dof) GT (solid red)
- Student-t(dof) learned (dashed red)
- vertical mean lines: `p0` mean (gray, at 0) and `p1` mean (black, at `MU_TARGET`)
- annotation with tail-region weighted L2 errors

### `lagrangian_paths/`

- `characteristic_paths_<source>_n<n_diag>.pdf`

Shows particle trajectories under GT vs learned velocity for fixed start points.

### `samples/`

- `reference_target_samples.npy`
- `ground_truth_flow_<source>.npy`
- `model_flow_<source>_n<n>.npy`

### `histograms/`

- `hist_<source>_n<n>.pdf`

Compares:

- analytical target PDF
- histogram of GT-flow samples
- histogram of model-flow samples

### `metrics/`

- `bar_hill.pdf`
- `bar_right_tail_ratio.pdf`
- `bar_tail_wasserstein.pdf`

Grouped bars across source distributions and sample sizes.

### `ablation/`

Contains temporal substitution ablation samples and bar plots, including:

- `ablation_bar_hill.pdf`
- `ablation_bar_tail_wasserstein.pdf`
- `ablation_bar_right_tail_ratio.pdf`
- `samples_<source>_<config>_n<n_diag>.npy`
- `reference_target_samples.npy`

Ablation configs used:

- `pure_model`
- `pure_gt`
- `fix_early` (`t in [0, 0.5]` uses GT)
- `fix_early_middle` (`t in [0.1, 0.6]` uses GT)
- `fix_late` (`t in [0.5, 1]` uses GT)

## 6) Tail metrics definitions (exactly as implemented)

### Hill estimator (`hill`)

- Uses top `5%` positive tail of centered samples (`x - MU_TARGET`)
- Computes inverse average log-spacing ratio estimator on sorted tail

### Right tail ratio (`right_tail_ratio`)

- `Q(0.999) / Q(0.99)`

### Tail Wasserstein (`tail_wasserstein`)

- Let `thr = Q_ref(0.99)` from reference samples
- Compare `samples[samples > thr]` vs `ref[ref > thr]`
- Distance: `scipy.stats.wasserstein_distance`
- If model tail has fewer than 2 samples, returns penalty value `100.0`

## 7) Typical run commands

### Default run

```bash
python single_mode/heavy_tail_sample_complexity_single_mode.py
```

### Beta time sampling run

```bash
python single_mode/heavy_tail_sample_complexity_single_mode.py \
  --time_sampling beta --beta_alpha 2 --beta_beta 5
```

### Quick sanity run

```bash
python single_mode/heavy_tail_sample_complexity_single_mode.py --quick --max_steps 3000 --min_steps 500
```
