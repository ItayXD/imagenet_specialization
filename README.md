# ImageNet Exchangeability Pipeline

Cluster-ready pipeline for studying first-layer exchangeability in width-scaled ImageNet ensembles as a function of images seen (`P`).

## What This Implements

1. Online ImageNet-1k training up to `1e7` images seen.
2. Manuscript-style Adam schedule shape:
- warmup `8e-5 -> 8e-3` over first `1%` of run,
- cosine decay `8e-3 -> 8e-5` over remaining `99%`.
3. Grouped ensemble execution:
- widths: `N = [32, 64, 128, 256, 512]`
- per-width ensemble size `M=16`
- run as `4` jobs per width, each job training `4` vectorized members.
4. Exact target checkpoint points:
- 15 log-spaced `P` values from `1e5` to `1e7`.
5. Exchangeability analysis on:
- first-layer weights,
- first-layer activations at `conv_init` (pre-BN/pre-ReLU).
6. Two analyses:
- `across_real vs across_shuffled` (primary),
- `within_vs_across` diagnostic + shuffled diagnostic.
7. Metrics and figures:
- KS distance, raw p-value, two-sided sigma, W1 distance,
- train/val loss and error vs images seen,
- per-width curves.
8. W&B tracking (online with fallback behavior configured by mode).

## Repository Entry Points

- Training entrypoint: `main.py`
- Core training loop: `src/experiment/training/online_momentum.py`
- Shared analysis utils: `src/experiment/exchangeability_utils.py`
- Config generator: `scripts/build_imagenet_sweep.py`
- Manifest builder: `scripts/build_exchangeability_manifest.py`
- SLURM submit helper: `scripts/submit_exchangeability_slurm.sh`
- Analysis script: `scripts/analyze_exchangeability.py`
- Plot script: `scripts/plot_exchangeability.py`
- Notebooks:
  - `notebooks/exchangeability_analysis.ipynb`
  - `notebooks/exchangeability_plots.ipynb`
- Largest-setting smoke harness: `scripts/run_largest_smoke.py`

## Required Environment

Set paths via environment variables:

```bash
export IMAGENET_FOLDER=/path/to/imagenet
export BASE_SAVE_DIR=/path/to/run_outputs
export REMOTE_RESULTS_FOLDER=/path/to/permanent_results   # optional
```

`IMAGENET_FOLDER` should contain torchvision-compatible ImageNet layout.

## Install

```bash
uv python install 3.11
uv sync --extra local
```

This creates `.venv` using Python 3.11 and installs local CPU dependencies plus `pytest`.

For cluster GPU environments:

```bash
uv python install 3.11
uv sync --extra cluster
```

## Generate Configs (20 jobs)

```bash
python scripts/build_imagenet_sweep.py
```

This creates:
- `conf/experiment/exchangeability_w{width}_g{group}.yaml`
- widths: 5
- groups per width: 4

Default Hydra config is set to `experiment=exchangeability_w32_g0`.

## Build Manifest

```bash
python scripts/build_exchangeability_manifest.py \
  --config-dir conf/experiment \
  --output conf/exchangeability_manifest.csv \
  --base-save-dir "$BASE_SAVE_DIR"
```

Manifest fields:
- `job_id,width,group_id,member_seed_list,data_seed,target_images_seen,p_targets_images_seen,base_dir,save_dir,wandb_group,experiment_name`

## Launch on SLURM

```bash
bash scripts/submit_exchangeability_slurm.sh conf/exchangeability_manifest.csv
```

Optional runtime overrides:

```bash
export SBATCH_GPUS=1
export SBATCH_CPUS=24
export SBATCH_MEM=128G
export SBATCH_TIME=72:00:00
```

To compute a better `SBATCH_TIME`, run the largest-setting smoke timing first (below).

## W&B Tracking

Enabled in generated configs:
- `wandb_enabled: true`
- `wandb_mode: online`

Set credentials on cluster nodes as usual (`WANDB_API_KEY` or login).

## Largest-Setting Smoke Test

Run one short memory/progression smoke job locally or on a single cluster node:

```bash
uv run python scripts/run_largest_smoke.py \
  --experiment exchangeability_w512_g0 \
  --max-tranches 50
```

This forces `max_tranches=50`, verifies large-width grouped training progresses, and prints:
- measured images/sec,
- estimated full-run hours to `target_images_seen`,
- suggested `SBATCH_TIME` (with configurable safety factor).

## Analyze Exchangeability

```bash
uv run python scripts/analyze_exchangeability.py \
  --base-save-dir "$BASE_SAVE_DIR" \
  --run-id exchangeability \
  --output-csv outputs/exchangeability_metrics.csv \
  --shuffle-repeats 2000 \
  --probe-batch-size 1024 \
  --probe-loader-batch-size 1
```

Output schema includes:
- `width,images_seen,representation,analysis_type,shuffle_id,ks_distance,ks_p_raw,ks_sigma_two_sided,w1_distance,train_loss,val_loss,train_error,val_error`

## Plot

```bash
uv run python scripts/plot_exchangeability.py \
  --input-csv outputs/exchangeability_metrics.csv \
  --output-dir outputs/plots_exchangeability
```

Generated figures include:
- `ks_distance_vs_images_seen.png`
- `w1_distance_vs_images_seen.png`
- `train_loss_vs_images_seen.png`
- `val_loss_vs_images_seen.png`
- `train_error_vs_images_seen.png`
- `val_error_vs_images_seen.png`

## Notebook Workflow

1. Open `notebooks/exchangeability_analysis.ipynb` to inspect/aggregate CSV outputs.
2. Open `notebooks/exchangeability_plots.ipynb` to regenerate plot assets interactively.

Both notebooks consume the same CSV generated by `scripts/analyze_exchangeability.py`.

## Tests

New test coverage is in:
- `test/test_exchangeability_utils.py`
- `test/test_manifest_builder.py`
- `test/test_largest_smoke_harness.py`

### Local tests (run now)

```bash
uv run pytest -q test/test_exchangeability_utils.py test/test_manifest_builder.py test/test_largest_smoke_harness.py
```

Expected locally:
- utility + manifest tests pass,
- largest-smoke harness test is skipped unless explicitly enabled.

### Cluster tests to run

1. Fast sanity subset:

```bash
uv run pytest -q test/test_exchangeability_utils.py test/test_manifest_builder.py
```

2. Largest-setting smoke harness (opt-in):

```bash
export RUN_LARGEST_SMOKE=1
export IMAGENET_FOLDER=/path/to/imagenet
export BASE_SAVE_DIR=/path/to/run_outputs
uv run pytest -q test/test_largest_smoke_harness.py
```

3. Recommended timing run for job-time extrapolation:

```bash
uv run python scripts/run_largest_smoke.py \
  --experiment exchangeability_w512_g0 \
  --max-tranches 50 \
  --safety-factor 1.35
```

Use the printed `suggested_sbatch_time` to set:

```bash
export SBATCH_TIME=HH:MM:SS
```

## Required User Inputs

Before full cluster execution, set/provide:

1. `IMAGENET_FOLDER`: absolute path to ImageNet-1k data.
2. `BASE_SAVE_DIR`: absolute path for checkpoints/artifacts.
3. `REMOTE_RESULTS_FOLDER` (optional): permanent copy destination.
4. W&B:
- `WANDB_API_KEY`,
- optional `wandb_entity` value (if using team/org account).
5. SLURM details:
- partition/account/QoS flags if your cluster requires them,
- preferred `SBATCH_GPUS`, `SBATCH_CPUS`, `SBATCH_MEM`, `SBATCH_TIME`.

## GitHub Sync Prep

Project is prepared for GitHub sync with `.gitignore`, `pyproject.toml`, `.python-version`, and `uv.lock`.

If this directory is not yet a git repo:

```bash
git init
git add .
git commit -m "Migrate to uv + exchangeability pipeline"
```

Then connect and push:

```bash
git remote add origin <YOUR_GITHUB_REPO_URL>
git branch -M main
git push -u origin main
```

## Notes

- This codebase still contains older legacy files from previous experiments; the exchangeability pipeline uses the files listed above.
- Analysis with full pooled distributions and large shuffle counts is compute-heavy by design.
