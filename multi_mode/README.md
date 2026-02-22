# Multi-Mode Heavy-Tail Experiment

Scripts:

- `multi_mode/heavy_tail_sample_complexity_multi_mode.py`
- `multi_mode/heavy_tail_sample_complexity.py` (synced copy)

This script extends the single-mode pipeline to a **mixture** Student-t target (multi-modal), and generates the same classes of diagnostics.

## 1) Experiment definition

### Target distribution (mixture)

Configured by constants in the script:

- `TARGET_CENTERS`
- `TARGET_WEIGHTS`
- `SIGMA_TARGET`
- `NU_TARGET`

Current default setup:

- centers: `[0.0, 25.0]`
- weights: `[0.6, 0.4]`
- target mean is computed as weighted mean of centers (`TARGET_MEAN`)

Sampling form:

- pick component `k ~ Categorical(TARGET_WEIGHTS)`
- sample `X1 = TARGET_CENTERS[k] + SIGMA_TARGET * T_nu`

### Source distributions

`SOURCE_CONFIGS`:

- Gaussian: `N(0,1)`
- Student-t: `nu in {1, 3, 5, 10}`

### Dataset sizes

`DATASET_SIZES` controls train sizes (current defaults: `1024, 4096, 16384`).

## 2) Core methods implemented

### Ground-truth velocity (mixture-aware)

`get_ground_truth_velocity(t, x, ...)` samples:

- target latent scale `s1`
- source latent scale `s0` (or fixed 1 for Gaussian source)
- target component index `k`

For each latent sample, uses that sample’s component center `mu_k` and computes:

- `D = (1-t)^2 / s0 + t^2 / s1`
- weight `w ∝ D^{-1/2} exp(-(x - t*mu_k)^2/(2D))`
- gain `G = t / (s1 * D)`
- conditional estimate `E = mu_k + G(x - t*mu_k)`

Then average over latent samples and return:

- `u*(t,x) = (E_hat - x)/(1-t)`

### Learned model

Same model/training style as single-mode:

- `VelocityMLP` with time embedding
- CFM loss on interpolant samples

### Time sampling

Training `t` sampling mode:

- `uniform`
- `beta` with `--beta_alpha`, `--beta_beta`

## 3) Output directory naming

Use `--output_root` (default `./output`).

Run folder is auto-generated:

- `multi_mu<target_mean>_c<center1>_<center2>_..._t<sampler_tag>`

Example:

- `multi_mu10_c0_25_tuniform`

## 4) CLI parameters

### I/O + runtime

- `--output_root`
- `--device`
- `--seed`
- `--quick`

### Training hyperparameters

- `--lr`
- `--batch_size`
- `--max_steps`
- `--min_steps`
- `--patience`
- `--improve_tol`

### Simulation / sampling

- `--n_eval_samples`
- `--dt`

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

## 5) Generated outputs

Inside one run folder (`./output/<run_tag>/`):

- `checkpoints/`
- `velocity_plots/`
- `lagrangian_paths/`
- `ablation/`
- `samples/`
- `histograms/`
- `metrics/`
- `summary.json`

The directory structure and file naming are parallel to single-mode.

### Notable multi-mode differences

- Histogram analytical curve is mixture PDF (`mixture_student_t_pdf`).
- Velocity plots include black vertical lines for each target mode center (`TARGET_CENTERS`).
- Summary includes:
  - `target_centers`
  - `target_weights`
  - `target_mean`

## 6) Tail metrics definitions (exactly as implemented)

### Hill estimator (`hill`)

- Applies to positive tail of centered data (`x - TARGET_MEAN`)
- Uses top `5%` tail log-ratio estimator

### Right tail ratio (`right_tail_ratio`)

- `Q(0.999) / Q(0.99)`

### Tail Wasserstein (`tail_wasserstein`)

- threshold from reference: `thr = Q_ref(0.99)`
- compare `samples > thr` to `reference > thr`
- distance via `scipy.stats.wasserstein_distance`
- returns `100.0` if tail sample count is too small (`<2`)

## 7) Ablation details

Temporal substitution configs are identical to single-mode:

- `pure_model`
- `pure_gt`
- `fix_early` (`[0, 0.5]`)
- `fix_early_middle` (`[0.1, 0.6]`)
- `fix_late` (`[0.5, 1]`)

Ablation outputs are saved under `ablation/`.

## 8) Typical run commands

### Default two-mode run

```bash
python multi_mode/heavy_tail_sample_complexity_multi_mode.py
```

### Beta time sampling run

```bash
python multi_mode/heavy_tail_sample_complexity_multi_mode.py \
  --time_sampling beta --beta_alpha 2 --beta_beta 5
```

### Change visualization range

```bash
python multi_mode/heavy_tail_sample_complexity_multi_mode.py \
  --x_min -15 --x_max 45
```
