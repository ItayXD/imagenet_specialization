# ImageNet Exchangeability Pipeline

Cluster-ready pipeline to measure first-layer exchangeability in ImageNet ensembles as a function of images seen (`P`).

## Scope

This repository now implements:

1. Online ImageNet-1k training to `target_images_seen=10_000_000`.
2. Manuscript LR shape: warmup `8e-5 -> 8e-3` then cosine decay `8e-3 -> 8e-5`.
3. Width sweep with grouped ensembles:
1. widths `N in [32, 64, 128, 256, 512]`
2. widths 32/64/128/256: 4 groups x 4 members
3. width 512: 16 groups x 1 member (memory-safe on 40GB A100)
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
- `scripts/download_imagenet_hf.py`
- `scripts/submit_exchangeability_slurm.sh`
- `scripts/submit_timing_sweep_slurm.sh`
- `scripts/run_largest_smoke.py`
- `scripts/run_timing_manifest_row.py`
- `scripts/submit_largest_smoke_slurm.sh`
- `scripts/summarize_timing_sweep.py`
- `scripts/build_width_slurm_jobs.py`
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
8. `HF_IMAGENET_REPO_ID=ILSVRC/imagenet-1k` (override if needed)
9. `HF_HOME=$EXCHANGEABILITY_ROOT/hf_cache`
10. `HF_DATASETS_CACHE=$HF_HOME/datasets`
11. `HUGGINGFACE_HUB_CACHE=$HF_HOME/hub`
12. `UV_CACHE_DIR=$EXCHANGEABILITY_ROOT/uv_cache`

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

Run `uv sync --extra cluster` before submitting SLURM jobs; submit scripts execute with `${UV_PROJECT_ENVIRONMENT}/bin/python` and do not resolve/install dependencies at runtime.
`source scripts/cluster_env.sh` pins `UV_PROJECT_ENVIRONMENT` to `/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/uv_envs/imagenet_specialization-py311` so jobs do not touch home `.venv`.

## ImageNet Data Setup

This repo includes a Hugging Face ImageNet downloader that exports torchvision-style `train/` and `val/` folders directly under your fixed directory.
The training loader automatically uses `ImageFolder` when those folders exist.

Use this directory (already defaulted):

```bash
source scripts/cluster_env.sh
bash scripts/bootstrap_cluster_dirs.sh
bash scripts/download_imagenet.sh
```

`scripts/download_imagenet.sh` expects these secrets to be set (for example in `~/.secrets`):

1. `HF_TOKEN`
2. optional `HF_IMAGENET_REPO_ID` (default `ILSVRC/imagenet-1k`)

The script downloads from HF and materializes:

1. `$IMAGENET_FOLDER/train/<class>/...jpg`
2. `$IMAGENET_FOLDER/val/<class>/...jpg`

Safety behavior:

1. the downloader fails if HF/uv cache paths point inside `$HOME`
2. caches are forced under `/n/netscratch/kempner_pehlevan_lab/Lab/ilavie` defaults

Optional debug caps:

1. `IMAGENET_MAX_TRAIN=<num>`
2. `IMAGENET_MAX_VAL=<num>`

## Generate Experiment Configs (32 Jobs)

```bash
source scripts/cluster_env.sh
uv run python scripts/build_imagenet_sweep.py
```

This writes:

- `conf/experiment/exchangeability_w{width}_g{group_id}.yaml`
- width-512 configs default to `ensemble_size=1`, `ensemble_subsets=1`.
- all widths default to `minibatch_size=1024`, `microbatch_size=128`.
- other widths keep grouped `ensemble_size=4`.

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
sbatch scripts/submit_exchangeability_slurm.sh conf/exchangeability_manifest.csv
```

Do not run submit scripts with `bash`; they are SLURM payload scripts and must be launched with `sbatch`.
Submit from the repository root so `SLURM_SUBMIT_DIR` points to this project (or pass `PROJECT_ROOT=/abs/path/to/repo`).

Supported overrides (via `sbatch` flags):

```bash
sbatch --array=0-31 --time=72:00:00 --cpus-per-task=24 --mem=128G --gpus=1 \
  scripts/submit_exchangeability_slurm.sh conf/exchangeability_manifest.csv
```

Logs default to:

`$BASE_SAVE_DIR/slurm_logs`

## Largest-Setting Smoke Timing (Submit with `sbatch`)

Use this first to estimate full runtime and tune `SBATCH_TIME`.

```bash
source scripts/cluster_env.sh
sbatch scripts/submit_largest_smoke_slurm.sh exchangeability_w512_g0 50 10000000 1.35
```

This runs 50 tranches at the largest setting and prints:

1. `images_per_second`
2. `estimated_full_hours`
3. `suggested_sbatch_time`

Smoke runs now use experiment config values by default.
You can still override with `--minibatch-size`, `--microbatch-size`, and `--num-workers`.

Apply that suggestion before full array submission.

## Timing Sweep For All Jobs (Submit with `sbatch`)

Run a short pilot for each manifest row to estimate per-job runtime:

```bash
source scripts/cluster_env.sh
sbatch scripts/submit_timing_sweep_slurm.sh conf/exchangeability_manifest.csv 20 10000000 1.35
```

After the sweep completes, summarize recommendations:

```bash
source scripts/cluster_env.sh
uv run python scripts/summarize_timing_sweep.py \
  --summary-dir "$BASE_SAVE_DIR/timing_sweep"
```

Outputs:

1. `$BASE_SAVE_DIR/timing_sweep/timing_estimates.csv` (per job)
2. `$BASE_SAVE_DIR/timing_sweep/timing_by_width.csv` (recommended `--time` per width)

## Bake Per-Width Times Into Saved Submit Jobs

If each width has a different walltime, generate width-specific manifests and submit scripts with baked `#SBATCH --time`.

```bash
source scripts/cluster_env.sh
uv run python scripts/build_width_slurm_jobs.py \
  --manifest conf/exchangeability_manifest.csv \
  --timing-width-csv "$BASE_SAVE_DIR/timing_sweep/timing_by_width.csv"
```

This writes:

1. `conf/manifests_by_width/exchangeability_manifest_w{width}.csv`
2. `conf/slurm_jobs/submit_exchangeability_w{width}.sbatch` (each with fixed `#SBATCH --time` and width-specific array size)
3. `conf/slurm_jobs/submit_exchangeability_all_widths.sh` (helper that submits all width jobs)

Submit all widths:

```bash
bash conf/slurm_jobs/submit_exchangeability_all_widths.sh
```

Or submit one width:

```bash
sbatch conf/slurm_jobs/submit_exchangeability_w512.sbatch
```

### Pinned Per-Width Times (2026-03-04 timing sweep)

These files are checked in and already pinned to the measured `selected` times:

1. `conf/slurm_jobs/submit_exchangeability_w32.sbatch` -> `04:17:58`
2. `conf/slurm_jobs/submit_exchangeability_w64.sbatch` -> `05:57:25`
3. `conf/slurm_jobs/submit_exchangeability_w128.sbatch` -> `09:26:25`
4. `conf/slurm_jobs/submit_exchangeability_w256.sbatch` -> `11:56:37`
5. `conf/slurm_jobs/submit_exchangeability_w512.sbatch` -> `13:17:00`

Associated manifests are also checked in:

1. `conf/manifests_by_width/exchangeability_manifest_w32.csv` (4 jobs)
2. `conf/manifests_by_width/exchangeability_manifest_w64.csv` (4 jobs)
3. `conf/manifests_by_width/exchangeability_manifest_w128.csv` (4 jobs)
4. `conf/manifests_by_width/exchangeability_manifest_w256.csv` (8 jobs)
5. `conf/manifests_by_width/exchangeability_manifest_w512.csv` (16 jobs)

Run one width at a time:

```bash
source scripts/cluster_env.sh
sbatch conf/slurm_jobs/submit_exchangeability_w32.sbatch
sbatch conf/slurm_jobs/submit_exchangeability_w64.sbatch
sbatch conf/slurm_jobs/submit_exchangeability_w128.sbatch
sbatch conf/slurm_jobs/submit_exchangeability_w256.sbatch
sbatch conf/slurm_jobs/submit_exchangeability_w512.sbatch
```

Expected terminal output per submit:

`Submitted batch job <JOBID>`

Run all widths with one command:

```bash
source scripts/cluster_env.sh
bash conf/slurm_jobs/submit_exchangeability_all_widths.sh
```

Expected terminal output:

1. five lines, each `Submitted batch job <JOBID>`
2. one array job per width

## Analysis

```bash
source scripts/cluster_env.sh
uv run python scripts/analyze_exchangeability.py \
  --base-save-dir "$BASE_SAVE_DIR" \
  --run-id exchangeability \
  --run-id-resolution latest_prefix \
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
sbatch scripts/submit_fast_tests_slurm.sh
```

Largest smoke pytest harness (GPU, opt-in):

```bash
source scripts/cluster_env.sh
sbatch scripts/submit_largest_smoke_pytest_slurm.sh
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

1. Nothing else for infra. Just ensure `HF_TOKEN` and `WANDB_API_KEY` exist in `~/.secrets`.
