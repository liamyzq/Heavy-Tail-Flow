# Heavy-Tail Diffusion Experiments

This repo contains two main experiment drivers:

- `single_mode/heavy_tail_sample_complexity_single_mode.py`
- `multi_mode/heavy_tail_sample_complexity_multi_mode.py`

`multi_mode/heavy_tail_sample_complexity_single_mode.py` is kept as a synced copy for convenience.

## UV Setup (Saved For This Repo)

Project files already saved:

- `pyproject.toml`
- `setup_uv.sh`

From `/home/mlw0719/heavy_tail_diffusion`:

```bash
# If uv is not available in PATH:
python -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"

# One-step setup
./setup_uv.sh
```

`setup_uv.sh` does:

1. Locate `uv` (`uv` in PATH or `~/.local/bin/uv`).
2. Create `.venv`.
3. Run `uv sync` from `pyproject.toml`.

Run scripts with uv-managed environment:

```bash
source .venv/bin/activate
python single_mode/heavy_tail_sample_complexity_single_mode.py
python multi_mode/heavy_tail_sample_complexity_multi_mode.py
```

## Output Structure

Both scripts use:

- `--output_root` (default: `./output`)

Each run writes into an auto-generated subdirectory based on key hyperparameters.

### Single-mode run folder name

- `single_mu<target-mean>_t<time-sampling-tag>`

Examples:

- `single_mu4_tuniform`
- `single_mu4_tbeta_a5_b7`

### Multi-mode run folder name

- `multi_mu<target-mean>_c<center1>_<center2>_..._t<time-sampling-tag>`

Examples:

- `multi_mu10_c0_25_tuniform`
- `multi_mu10_c0_25_tbeta_a2_b5`

### Files produced inside each run folder

- `checkpoints/`: saved model weights per `(source, n)`
- `velocity_plots/`: velocity field figures (PDF)
- `lagrangian_paths/`: characteristic trajectory figures (PDF)
- `ablation/`: hybrid-velocity ablation figures (PDF)
- `samples/`: `.npy` arrays of reference/GT/model generated samples
- `histograms/`: density comparison figures (PDF)
- `metrics/`: grouped bar plots for tail metrics (PDF)
- `summary.json`: run metadata + computed metrics

## Script Details

### 1) `single_mode/heavy_tail_sample_complexity_single_mode.py`

Target:

- single Student-t target with constants:
  - `MU_TARGET`
  - `SIGMA_TARGET`
  - `NU_TARGET`

Source family:

- Gaussian + Student-t (`nu in {1,3,5,10}`).

Important built-in constants:

- `DATASET_SIZES` (training sizes)
- `T_PLOT` (velocity plot time grid)
- `N_DIAGNOSTIC` (model size used for diagnostic visualizations)
- `K_MONTE_CARLO` (ground-truth MC latent samples)

CLI hyperparameters (major):

- `--output_root`
- `--device`, `--seed`
- `--lr`, `--batch_size`, `--max_steps`, `--min_steps`, `--patience`, `--improve_tol`
- `--n_eval_samples`, `--dt`
- `--time_sampling {uniform,beta}`, `--beta_alpha`, `--beta_beta`
- `--x_min`, `--x_max`, `--velocity_grid_points`
- `--reuse_checkpoints` / `--no_reuse_checkpoints`
- `--quick`

Example:

```bash
python single_mode/heavy_tail_sample_complexity_single_mode.py \
  --time_sampling beta --beta_alpha 2 --beta_beta 5 \
  --x_min -5 --x_max 35
```

### 2) `multi_mode/heavy_tail_sample_complexity_multi_mode.py`

Target:

- mixture Student-t target with constants:
  - `TARGET_CENTERS`
  - `TARGET_WEIGHTS`
  - `SIGMA_TARGET`
  - `NU_TARGET`

Current default two-mode setup:

- centers: `[0.0, 25.0]`
- weights: `[0.6, 0.4]`

Source family and most training/plot hyperparameters are the same as single-mode.

Additional/important behavior:

- Ground-truth velocity integrates over target mode indicator + latent scales.
- Output run directory name includes target centers (and implied mean).

Example:

```bash
python multi_mode/heavy_tail_sample_complexity_multi_mode.py \
  --time_sampling uniform \
  --x_min -10 --x_max 40
```

## How To Change Target Settings

### Single mode

Edit constants in:

- `single_mode/heavy_tail_sample_complexity_single_mode.py`

Typical edits:

- `MU_TARGET`
- `SIGMA_TARGET`
- `NU_TARGET`

### Multi mode

Edit constants in:

- `multi_mode/heavy_tail_sample_complexity_multi_mode.py`

Typical edits:

- `TARGET_CENTERS`
- `TARGET_WEIGHTS`
- `SIGMA_TARGET`
- `NU_TARGET`

After changing constants, rerun the corresponding script.