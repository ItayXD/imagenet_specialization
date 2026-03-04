Run this exact sequence on cluster.

1. Initialize env and directories.
```bash
source scripts/cluster_env.sh
bash scripts/bootstrap_cluster_dirs.sh
```
Does: loads `~/.secrets`, sets all paths/caches to `/n/netscratch/kempner_pehlevan_lab/Lab/ilavie`, creates required dirs.  
Expect: printed paths (`EXCHANGEABILITY_ROOT`, `IMAGENET_FOLDER`, `HF_HOME`, `UV_CACHE_DIR`, etc.).

2. Download/export ImageNet from HF.
```bash
bash scripts/download_imagenet.sh
```
Does: uses `HF_TOKEN`, downloads from `HF_IMAGENET_REPO_ID` (default `ILSVRC/imagenet-1k`), exports to:
- `/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/imagenet/train/...`
- `/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/imagenet/val/...`
Expect: progress bars, then `Finished HF ImageNet export...`.  
If token/access is wrong: auth error from HF.

3. Install env (once per node/env).
```bash
uv python install 3.11
uv sync --extra cluster
```
Does: installs Python + deps.  
Expect: package resolve/install output, no errors.

4. Submit fast sanity tests (must be `sbatch`).
```bash
sbatch scripts/submit_fast_tests_slurm.sh
```
Does: runs `pytest` for fast utility/manifest tests on 1 GPU partition job.  
Expect submit output: `Submitted batch job <JOBID>`.  
Log: `/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/exchangeability_outputs/slurm_logs/fast_tests_<JOBID>.out`  
Expected result in log: `6 passed`.

5. Submit largest smoke pytest harness.
```bash
sbatch scripts/submit_largest_smoke_pytest_slurm.sh
```
Does: runs smoke harness test on GPU.  
Expect: `Submitted batch job <JOBID>`.  
Log: `.../slurm_logs/largest_smoke_test_<JOBID>.out`.

6. Submit largest timing smoke (for runtime extrapolation).
```bash
sbatch scripts/submit_largest_smoke_slurm.sh exchangeability_w512_g0 50 10000000 1.35
```
Does: runs 50 tranches, estimates full runtime.  
Expect in log (`.../largest_smoke_<JOBID>.out`):  
- `images_per_second=...`  
- `estimated_full_hours=...`  
- `suggested_sbatch_time=HH:MM:SS`

7. Build configs + manifest.
```bash
source scripts/cluster_env.sh
uv run python scripts/build_imagenet_sweep.py
uv run python scripts/build_exchangeability_manifest.py --output conf/exchangeability_manifest.csv
```
Does: generates 20 jobs (5 widths × 4 groups), manifest rows.  
Expect: `Wrote 20 rows to conf/exchangeability_manifest.csv`.

8. Submit full training array.
```bash
sbatch scripts/submit_exchangeability_slurm.sh conf/exchangeability_manifest.csv
```
Does: launches array job (1 GPU per task), each task trains one manifest row.  
Expect: `Submitted batch job <ARRAY_JOBID>`.  
Logs per task: `.../slurm_logs/exchangeability_<ARRAY_JOBID>_<TASK_ID>.out`.

9. After training, run analysis + plots.
```bash
source scripts/cluster_env.sh
uv run python scripts/analyze_exchangeability.py --base-save-dir "$BASE_SAVE_DIR" --run-id exchangeability --output-csv "$BASE_SAVE_DIR/exchangeability_metrics.csv" --shuffle-repeats 2000 --probe-batch-size 1024 --probe-loader-batch-size 1
uv run python scripts/plot_exchangeability.py --input-csv "$BASE_SAVE_DIR/exchangeability_metrics.csv" --output-dir "$BASE_SAVE_DIR/plots_exchangeability"
```
Expect:
- CSV: `/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/exchangeability_outputs/exchangeability_metrics.csv`
- plots in `/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/exchangeability_outputs/plots_exchangeability/`.

Monitor jobs:
```bash
squeue -u "$USER"
```