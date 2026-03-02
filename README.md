# ImageNet Exchangeability Pipeline

Cluster-ready pipeline to measure first-layer exchangeability in ImageNet ensembles as a function of images seen (`P`).

## Scope

This repository now implements:

1. Online ImageNet-1k training to `target_images_seen=10_000_000`.
2. Manuscript LR shape: warmup `8e-5 -> 8e-3` then cosine decay `8e-3 -> 8e-5`.
3. Width sweep with grouped ensembles:
1. widths `N in [32, 64, 128, 256, 512]`
2. 4 groups per width
3. 4 members per group
4. total `M=16` members per width
4. Target checkpoint points: 15 deterministic log-spaced `P` values (`1e5` to `1e7`).
5. Exchangeability analysis for:
1. first-layer weights
2. first-layer activations (`conv_init`, pre-BN/pre-ReLU)
6. Statistical outputs:
1. KS distance
2. KS raw p-value
3. two-sided sigma equivalent
4. Wasserstein-1 (W1)
7. Analyses:
1. `across_real_vs_across_shuffled` (primary)
2. `within_vs_across` diagnostic (with shuffle null)
8. Training metrics logging: train/val loss and error.
9. W&B integration with online/offline fallback behavior.

## Key Files

- `main.py`
- `config_structs.py`
- `src/experiment/training/online_momentum.py`
- `src/experiment/exchangeability_utils.py`
- `scripts/build_imagenet_sweep.py`
- `scripts/build_exchangeability_manifest.py`
- `scripts/bootstrap_cluster_dirs.sh`
- `scripts/cluster_env.sh`
- `scripts/download_imagenet.sh`
- `scripts/prepare_imagenet_archives.py`
- `scripts/submit_exchangeability_slurm.sh`
- `scripts/run_largest_smoke.py`
- `scripts/submit_largest_smoke_slurm.sh`
- `scripts/analyze_exchangeability.py`
- `scripts/plot_exchangeability.py`
- `notebooks/exchangeability_analysis.ipynb`
- `notebooks/exchangeability_plots.ipynb`

## Cluster Defaults (Already Set)

Default root is:

`/n/netscratch/kempner_pehlevan_lab/Lab/ilavie`

The code defaults to:

1. `EXCHANGEABILITY_ROOT=/n/netscratch/kempner_pehlevan_lab/Lab/ilavie`
2. `IMAGENET_FOLDER=$EXCHANGEABILITY_ROOT/imagenet`
3. `BASE_SAVE_DIR=$EXCHANGEABILITY_ROOT/exchangeability_outputs`
4. `REMOTE_RESULTS_FOLDER=$EXCHANGEABILITY_ROOT`
5. `SBATCH_ACCOUNT=kempner_pehlevan_lab`
6. `SBATCH_PARTITION=kempner`
7. `WANDB_PROJECT=imagenet_specialization`

Load these defaults in each shell with:

```bash
source scripts/cluster_env.sh
```

## `wandb_entity` meaning

`wandb_entity` is optional. Keep it empty (`''`) to log under the user account authenticated by your `WANDB_API_KEY`. Set it only if you want to log to a team/org workspace.

## Install with `uv` (Python 3.11)

### Local (CPU)

```bash
uv python install 3.11
uv sync --extra local
```

### Cluster (GPU)

```bash
uv python install 3.11
uv sync --extra cluster
```

## ImageNet Data Setup

`torchvision.datasets.ImageNet` does not provide a built-in public download for ImageNet due licensing.
This repo now includes a download script that pulls the three official archives from URLs you provide and places them in your fixed directory.

Use this directory (already defaulted):

```bash
source scripts/cluster_env.sh
bash scripts/bootstrap_cluster_dirs.sh
bash scripts/download_imagenet.sh
```

`scripts/download_imagenet.sh` expects these secrets to be set (for example in `~/.secrets`):

1. `IMAGENET_TRAIN_URL`
2. `IMAGENET_VAL_URL`
3. `IMAGENET_DEVKIT_URL`

The script downloads:

1. `ILSVRC2012_img_train.tar`
2. `ILSVRC2012_img_val.tar`
3. `ILSVRC2012_devkit_t12.tar.gz`

and then runs archive preparation so torchvision-ready `train/` and `val/` folders are materialized under `$IMAGENET_FOLDER`.

## Generate Experiment Configs (20 Jobs)

```bash
source scripts/cluster_env.sh
uv run python scripts/build_imagenet_sweep.py
```

This writes:

- `conf/experiment/exchangeability_w{width}_g{group_id}.yaml`

## Build Manifest

```bash
source scripts/cluster_env.sh
uv run python scripts/build_exchangeability_manifest.py \
  --config-dir conf/experiment \
  --output conf/exchangeability_manifest.csv
```

## Full Training Submission (SLURM Array)

```bash
source scripts/cluster_env.sh
bash scripts/submit_exchangeability_slurm.sh conf/exchangeability_manifest.csv
```

Supported overrides:

```bash
export SBATCH_GPUS=1
export SBATCH_CPUS=24
export SBATCH_MEM=128G
export SBATCH_TIME=72:00:00
```

Logs default to:

`$BASE_SAVE_DIR/slurm_logs`

## Largest-Setting Smoke Timing (Submit with `sbatch`)

Use this first to estimate full runtime and tune `SBATCH_TIME`.

```bash
source scripts/cluster_env.sh
bash scripts/submit_largest_smoke_slurm.sh exchangeability_w512_g0 50 10000000 1.35
```

This runs 50 tranches at the largest setting and prints:

1. `images_per_second`
2. `estimated_full_hours`
3. `suggested_sbatch_time`

Apply that suggestion before full array submission.

## Analysis

```bash
source scripts/cluster_env.sh
uv run python scripts/analyze_exchangeability.py \
  --base-save-dir "$BASE_SAVE_DIR" \
  --run-id exchangeability \
  --output-csv "$BASE_SAVE_DIR/exchangeability_metrics.csv" \
  --shuffle-repeats 2000 \
  --probe-batch-size 1024 \
  --probe-loader-batch-size 1
```

## Plotting

```bash
uv run python scripts/plot_exchangeability.py \
  --input-csv "$BASE_SAVE_DIR/exchangeability_metrics.csv" \
  --output-dir "$BASE_SAVE_DIR/plots_exchangeability"
```

## Notebook Analysis

- `notebooks/exchangeability_analysis.ipynb`
- `notebooks/exchangeability_plots.ipynb`

Both notebooks consume the same CSV from `scripts/analyze_exchangeability.py`.

## Tests

### Local tests already intended to run on CPU

```bash
uv run pytest -q test/test_exchangeability_utils.py test/test_manifest_builder.py test/test_largest_smoke_harness.py
```

Expected locally:

1. utility + manifest tests pass
2. largest smoke harness test is skipped unless `RUN_LARGEST_SMOKE=1`

### Cluster tests (submit via `sbatch`)

Fast non-GPU sanity:

```bash
source scripts/cluster_env.sh
bash scripts/submit_fast_tests_slurm.sh
```

Largest smoke pytest harness (GPU, opt-in):

```bash
source scripts/cluster_env.sh
bash scripts/submit_largest_smoke_pytest_slurm.sh
```

## Time Extrapolation

Yes. Runtime extrapolation is implemented in `scripts/run_largest_smoke.py`.

It measures elapsed wall time for the smoke run (`max_tranches`, default 50) and computes an estimated full runtime to `target_images_seen` plus a safety-adjusted `suggested_sbatch_time`.

## W&B Setup

On cluster:

```bash
source ~/.secrets
```

Defaults already set:

1. project: `imagenet_specialization`
2. entity: empty string (uses your authenticated user account)

If you later want a team entity:

```bash
export WANDB_ENTITY=<team_or_org_name>
uv run python scripts/build_imagenet_sweep.py
```

## GitHub Sync

Current branch/worktree is ready for push after your review.

If remote is not set:

```bash
git remote add origin <YOUR_GITHUB_REPO_URL>
git branch -M main
git push -u origin main
```

## What I Still Need From You

1. `IMAGENET_TRAIN_URL`, `IMAGENET_VAL_URL`, `IMAGENET_DEVKIT_URL` in `~/.secrets`.
