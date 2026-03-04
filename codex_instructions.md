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

________

Perfect. This is exactly where we wanted to land.

I’ve now locked defaults to what passed:
- width-512 runs as true single-member jobs: `ensemble_size=1`, `ensemble_subsets=1`
- width-512 batch defaults: `minibatch=128`, `microbatch=128`
- total grid now: 32 jobs (`4+4+4+4+16`)

### What to do next (back to main plan)

1. Sync this latest code to cluster (pull latest commit).
2. Run fast sanity tests:
```bash
source scripts/cluster_env.sh
sbatch scripts/submit_fast_tests_slurm.sh
```
3. Submit full training array:
```bash
source scripts/cluster_env.sh
sbatch scripts/submit_exchangeability_slurm.sh conf/exchangeability_manifest.csv
```
4. Monitor:
- logs in `/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/exchangeability_outputs/slurm_logs`
- W&B project `imagenet_specialization`

5. After enough checkpoints finish, run analysis:
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

6. Generate plots:
```bash
uv run python scripts/plot_exchangeability.py \
  --input-csv "$BASE_SAVE_DIR/exchangeability_metrics.csv" \
  --output-dir "$BASE_SAVE_DIR/plots_exchangeability"
```

If you want, I can now prepare a single “production submit” command set with recommended `--time/--mem/--cpus` overrides per width bucket (small widths vs width-512) to reduce queue failures and wasted allocation.



___

# Implementation Plan: ImageNet Exchangeability Pipeline (Full Scope, Cluster-Ready)

## Brief Summary
Implement the full study pipeline you approved, including all items from both plans:
1. Online ImageNet-1k training to `1e7` images seen with warmup+cosine LR schedule shape from manuscript.
2. Ensemble execution as 4-member grouped jobs (`4 groups x 5 widths = 20 jobs`) with deterministic manifest/seeds.
3. Exchangeability analysis for first-layer weights and activations.
4. Two analyses:
1. `Across-real vs across-shuffled` (primary).
2. `Within-vs-across` diagnostic with shuffle null.
5. KS + W1 metrics, per-width KS-vs-`P` plots, train/val logging, W&B tracking.
6. Script and notebook versions for analysis and plotting.
7. Largest-setting memory/training smoke test (`N=512`, grouped run, 50 tranches).

## Planned File Touches
1. [main.py](/Users/itay/Downloads/Supplement/ImageNet/main.py)
2. [config_structs.py](/Users/itay/Downloads/Supplement/ImageNet/config_structs.py)
3. [conf/config.yaml](/Users/itay/Downloads/Supplement/ImageNet/conf/config.yaml)
4. [conf/experiment](/Users/itay/Downloads/Supplement/ImageNet/conf/experiment)
5. [src/run/constants.py](/Users/itay/Downloads/Supplement/ImageNet/src/run/constants.py)
6. [src/run/OnlinePreprocessDevice.py](/Users/itay/Downloads/Supplement/ImageNet/src/run/OnlinePreprocessDevice.py)
7. [src/run/OnlineTaskRunner.py](/Users/itay/Downloads/Supplement/ImageNet/src/run/OnlineTaskRunner.py)
8. [src/experiment/imagenet_resnet.py](/Users/itay/Downloads/Supplement/ImageNet/src/experiment/imagenet_resnet.py)
9. [src/experiment/training/online_momentum.py](/Users/itay/Downloads/Supplement/ImageNet/src/experiment/training/online_momentum.py)
10. [scripts/build_imagenet_sweep.py](/Users/itay/Downloads/Supplement/ImageNet/scripts/build_imagenet_sweep.py)
11. New: [scripts/build_exchangeability_manifest.py](/Users/itay/Downloads/Supplement/ImageNet/scripts/build_exchangeability_manifest.py)
12. New: [scripts/submit_exchangeability_slurm.sh](/Users/itay/Downloads/Supplement/ImageNet/scripts/submit_exchangeability_slurm.sh)
13. New: [scripts/analyze_exchangeability.py](/Users/itay/Downloads/Supplement/ImageNet/scripts/analyze_exchangeability.py)
14. New: [scripts/plot_exchangeability.py](/Users/itay/Downloads/Supplement/ImageNet/scripts/plot_exchangeability.py)
15. New: [notebooks/exchangeability_analysis.ipynb](/Users/itay/Downloads/Supplement/ImageNet/notebooks/exchangeability_analysis.ipynb)
16. New: [notebooks/exchangeability_plots.ipynb](/Users/itay/Downloads/Supplement/ImageNet/notebooks/exchangeability_plots.ipynb)
17. New/updated tests under [test](/Users/itay/Downloads/Supplement/ImageNet/test)
18. [README.md](/Users/itay/Downloads/Supplement/ImageNet/README.md)

## Public Interfaces / Config Additions
1. Extend `TrainingParams` with:
`target_images_seen`, `p_targets_images_seen`, `wandb_enabled`, `wandb_project`, `wandb_entity`, `wandb_mode`, `run_id`, `width`, `group_id`, `member_group_size`, `probe_batch_size`.
2. Keep existing schedule params but enforce manuscript-style shape:
`warmup_epochs=0.5`, `init_lr=8e-5`, `eta_0=8e-3`, `min_lr=8e-5`.
3. Add manifest schema fields:
`job_id,width,group_id,member_seed_list,data_seed,target_images_seen,p_targets_images_seen,base_dir,save_dir,wandb_group`.
4. Add analysis table schema fields:
`width,images_seen,representation,analysis_type,ks_distance,ks_p_raw,ks_sigma_two_sided,w1_distance,shuffle_id,train_loss,val_loss,train_error,val_error`.

## Exact Training Protocol
1. Dataset remains ImageNet-1k path currently used.
2. Online regime metric is cumulative `images_seen`.
3. `P` targets are 15 deterministic log-spaced integers from `1e5` to `1e7`.
4. Stop criterion is `images_seen >= 1e7`.
5. LR schedule is scaled to actual training horizon:
warmup on first 1% of total steps, then cosine decay for remaining 99%, endpoints fixed at `8e-5 -> 8e-3 -> 8e-5`.
6. Preserve heavy augmentation (`AutoAugmentPolicy.IMAGENET` already in loader).
7. Save at each target `P`:
full model state + first-layer weight artifact + metric snapshot.

## Ensemble/Cluster Execution
1. Use SLURM array with 20 tasks total.
2. Each task trains 4 vectorized members of same width.
3. Four groups per width reconstruct full `M=16`.
4. Shared data-order seed across widths and groups.
5. Distinct member seeds per group from manifest.
6. W&B online logging with fallback behavior configured via mode/env.

## Exchangeability Analysis Definitions
### Representation construction
1. Weights:
extract first-layer kernels per member/filter, flatten per filter vector.
2. Activations:
tap `conv_init` pre-BN/pre-ReLU on fixed probe batch of 1024 val images, flatten each filter over `B*H*W`.

### Similarity
1. Absolute cosine similarity for all comparisons.

### Analysis A (Primary baseline test)
1. Build `A_real` from all undirected cross-member pairs `e<f`, all neuron pairs `(i,j)`.
2. Build shuffled `X'` via flatten-permute-reshape over `(E*N)` vectors.
3. Build `A_shuf` with same cross-member rule from `X'`.
4. Compute KS and W1 between `A_real` and `A_shuf`.
5. Repeat shuffles to produce null summaries and uncertainty.

### Analysis B (Secondary diagnostic)
1. Build `W_real` from within-member neuron pairs `i<j`, pooled over members.
2. Use `A_real` as across reference.
3. Compute KS and W1 between `W_real` and `A_real`.
4. Compute shuffle-based null diagnostic values.

### Pair convention
1. Across uses undirected member pairs only (`e<f`), not directed duplicates.

## Statistical Outputs
1. KS distance and KS raw p-value.
2. Two-sided sigma conversion from p.
3. W1 distance.
4. Shuffle-repeat summaries for uncertainty bands.
5. No BH-FDR correction in primary outputs (raw-only as chosen).

## Plot/Notebook Deliverables
1. Script plots:
per-width KS-vs-images-seen curves for each representation and analysis type.
2. Script plots:
per-width W1-vs-images-seen curves.
3. Script plots:
per-width train/val error and loss vs images seen.
4. Notebook `exchangeability_analysis.ipynb`:
interactive loading, filtering, sanity checks, table exploration.
5. Notebook `exchangeability_plots.ipynb`:
publication-style figure regeneration from saved analysis tables.
6. Notebooks call shared analysis utilities, not reimplemented logic.

## W&B Tracking Deliverables
1. Log per-step and per-target-`P` training metrics.
2. Log LR and images_seen.
3. Log analysis artifacts and output table paths.
4. Group runs by `study_id`, width, and group id.

## Tests And Validation
1. Unit test for target-`P` scheduler exactness and monotonic trigger behavior.
2. Unit test for similarity kernels on deterministic tensors.
3. Unit test for shuffle transform correctness (flatten-permute-reshape invariants).
4. Unit test for pair-count/index correctness under undirected rules.
5. Integration test for end-to-end miniature run producing checkpoints and analysis rows.
6. Largest-setting smoke test:
`N=512`, grouped job (4 members), 50 tranches, assert no OOM and confirm train loop progresses.
7. Acceptance check:
complete `(width,P)` coverage for all five widths and all 15 targets.

## Execution Sequence
1. Stabilize runtime/config fields and checkpoint triggers.
2. Add manifest builder and SLURM array launcher.
3. Update training loop for images-seen stop and scaled warmup+cosine schedule.
4. Add W&B logging hooks.
5. Implement exchangeability analysis core and table writer.
6. Implement plotting script.
7. Add notebook analysis and notebook plotting equivalents.
8. Add/adjust tests including largest-setting smoke test harness.
9. Update README with full runbook and reproducibility steps.

## Explicit Assumptions
1. Cluster has SLURM and supports job arrays.
2. ImageNet-1k path and credentials are available in runtime environment.
3. W&B credentials may be absent on some nodes, so fallback mode must be supported.
4. Storage is sufficient for full states at 15 targets x 20 jobs.
5. Existing augmentation stack remains valid for online-regime extension.

_______

Implemented end-to-end pipeline changes across training, orchestration, analysis, plotting, notebooks, tests, and docs.

**What I changed**
- Runtime/config stabilization:
  - Fixed task validation execution and microbatch/minibatch validation in [read_tasks.py](/Users/itay/Downloads/Supplement/ImageNet/src/tasks/read_tasks.py).
  - Fixed callback wiring to pass actual callable in [imagenet_resnet.py](/Users/itay/Downloads/Supplement/ImageNet/src/experiment/imagenet_resnet.py).
  - Added preprocess directory creation in [OnlinePreprocessDevice.py](/Users/itay/Downloads/Supplement/ImageNet/src/run/OnlinePreprocessDevice.py).
  - Hardened runner DataLoader settings in [OnlineTaskRunner.py](/Users/itay/Downloads/Supplement/ImageNet/src/run/OnlineTaskRunner.py).
  - Made result copy optional when `REMOTE_RESULTS_FOLDER` is unset in [main.py](/Users/itay/Downloads/Supplement/ImageNet/main.py).
  - Updated env-driven constants in [constants.py](/Users/itay/Downloads/Supplement/ImageNet/src/run/constants.py).

- New config interface:
  - Added training fields (`target_images_seen`, `p_targets_images_seen`, W&B, run/group metadata, probe size, logging cadence, smoke override) in [config_structs.py](/Users/itay/Downloads/Supplement/ImageNet/config_structs.py).
  - Switched default Hydra experiment to exchangeability config in [config.yaml](/Users/itay/Downloads/Supplement/ImageNet/conf/config.yaml).
  - Added 20 grouped configs (`exchangeability_w{32,64,128,256,512}_g{0..3}.yaml`) under [conf/experiment](/Users/itay/Downloads/Supplement/ImageNet/conf/experiment).

- Training protocol implementation:
  - Rewrote [online_momentum.py](/Users/itay/Downloads/Supplement/ImageNet/src/experiment/training/online_momentum.py) to:
    - Train to cumulative `images_seen`.
    - Trigger exact target-`P` checkpoints.
    - Use scaled manuscript warmup+cosine schedule.
    - Save full state checkpoints and first-layer weight artifacts per target.
    - Log train/val loss+error and LR.
    - Integrate W&B (online + fallback).
    - Support `max_tranches` smoke runs.

- Cluster execution assets:
  - Regenerated config builder for exchangeability runs in [build_imagenet_sweep.py](/Users/itay/Downloads/Supplement/ImageNet/scripts/build_imagenet_sweep.py).
  - Added manifest builder in [build_exchangeability_manifest.py](/Users/itay/Downloads/Supplement/ImageNet/scripts/build_exchangeability_manifest.py).
  - Added robust SLURM array submit helper in [submit_exchangeability_slurm.sh](/Users/itay/Downloads/Supplement/ImageNet/scripts/submit_exchangeability_slurm.sh).
  - Added CSV-safe row runner in [run_manifest_row.py](/Users/itay/Downloads/Supplement/ImageNet/scripts/run_manifest_row.py).
  - Added largest-setting smoke harness in [run_largest_smoke.py](/Users/itay/Downloads/Supplement/ImageNet/scripts/run_largest_smoke.py).

- Exchangeability analysis + plotting:
  - Added shared math/stats utilities in [exchangeability_utils.py](/Users/itay/Downloads/Supplement/ImageNet/src/experiment/exchangeability_utils.py).
  - Added full analysis pipeline in [analyze_exchangeability.py](/Users/itay/Downloads/Supplement/ImageNet/scripts/analyze_exchangeability.py):
    - Weights + activations.
    - Across-real vs across-shuffled (flatten-permute-reshape).
    - Within-vs-across diagnostic + shuffle diagnostic.
    - KS/W1/sigma outputs with per-shuffle rows.
  - Added plotting script in [plot_exchangeability.py](/Users/itay/Downloads/Supplement/ImageNet/scripts/plot_exchangeability.py).

- Notebook deliverables:
  - [exchangeability_analysis.ipynb](/Users/itay/Downloads/Supplement/ImageNet/notebooks/exchangeability_analysis.ipynb)
  - [exchangeability_plots.ipynb](/Users/itay/Downloads/Supplement/ImageNet/notebooks/exchangeability_plots.ipynb)
  - Added [scripts/__init__.py](/Users/itay/Downloads/Supplement/ImageNet/scripts/__init__.py) for notebook imports.

- Tests:
  - Added [test_exchangeability_utils.py](/Users/itay/Downloads/Supplement/ImageNet/test/test_exchangeability_utils.py)
  - Added [test_manifest_builder.py](/Users/itay/Downloads/Supplement/ImageNet/test/test_manifest_builder.py)
  - Added [test_largest_smoke_harness.py](/Users/itay/Downloads/Supplement/ImageNet/test/test_largest_smoke_harness.py)

- Docs:
  - Replaced README with full runbook in [README.md](/Users/itay/Downloads/Supplement/ImageNet/README.md).
  - Updated dependencies in [requirements.txt](/Users/itay/Downloads/Supplement/ImageNet/requirements.txt).
  - Removed remaining `math`-module imports in legacy scripts to satisfy AGENTS rule.

**Validation I ran**
- Syntax checks (passed): `python -m py_compile` on all new/changed Python files.
- Shell syntax check (passed): `bash -n` on [submit_exchangeability_slurm.sh](/Users/itay/Downloads/Supplement/ImageNet/scripts/submit_exchangeability_slurm.sh).

**What I could not run here**
- Runtime script/test execution is blocked in this environment:
  - `pytest` is not installed (`command not found`).
  - `omegaconf` is missing for script execution.
  - Local Python/Numpy runtime is partially broken (shared library issue), so full execution checks were not possible.

**Next steps**
1. Install runtime deps (`pip install -r requirements.txt`, plus `pytest` if needed).
2. Generate manifest and launch array jobs (`build_exchangeability_manifest.py`, `submit_exchangeability_slurm.sh`).
3. Run analysis + plotting scripts, then inspect notebook outputs.