# CLAUDE.md

Exchangeability / weight-spectra experiments: ResNet width sweeps on
ImageNet-1k and CIFAR-5M, plus toy models for width-consistency of weight
spectra under SGD vs Muon. JAX/Flax + Hydra configs + SLURM on Cannon.

**Read `AGENTS.md` first — it is the authoritative guide** for cluster
workflow, paths, SLURM submission/monitoring, manifest and run-id
conventions, and common mistakes. Do not duplicate or contradict it.

## Commands

```bash
uv run pytest -q                      # tests (use uv run, never .venv binaries)
uv run python scripts/<script>.py     # local scripts (manifests, plotting)
sbatch scripts/submit_*.sh ...        # SLURM payloads: always sbatch, never bash
```

Known-broken legacy tests (pre-existing, unrelated to current work):
`test/test_momentum.py`, `test/test_pd.py`, `test/test_sh.py` (import errors),
`test/test_runner.py::test_tcs_yaml`.

## Key paths

- Local repo: `/Users/itay/HarvardDocs/ImageNet`
- Remote repo (Cannon): `/n/home13/ilavie/imagenet_specialization`
- Cluster scratch: `/n/netscratch/kempner_pehlevan_lab/Lab/ilavie`
  (`$EXCHANGEABILITY_ROOT`; results under `exchangeability_imagenet/`,
  `exchangeability_cifar5m/`, `toy_spectra/`)
- Always `source scripts/cluster_env.sh` on Cannon before submitting.
- Local and remote repos are NOT auto-synced: `scp`/`rsync` changed files
  to Cannon before submitting.

## toy_spectra pipeline

Toy models (3-layer linear, 2/3-layer relu with committee-teacher target)
testing width-consistency of weight spectra under muP SGD vs idealized Muon
(momentum-free, exact orthogonalization). Science context:
`muon_spectra_toy/HANDOFF.md`; conventions: `scripts/toy_spectra/core.py`
docstring.

```bash
# 1. build manifest (sweep: 3 models x {sgd,muon} x widths x seeds)
uv run python scripts/toy_spectra/build_toy_spectra_manifest.py \
  --output conf/toy_spectra_manifest.csv
# 2. sync repo to Cannon, then submit (array size = manifest rows)
sbatch scripts/submit_toy_spectra_slurm.sh conf/toy_spectra_manifest.csv
# 3. rsync results locally, then plot
uv run python scripts/plot_toy_spectra.py \
  --results-root <local copy of $EXCHANGEABILITY_ROOT/toy_spectra> \
  --output-dir artifacts/toy_spectra_plots --job-suffix job<ARRAY_JOB_ID>
```

- run_id scheme: `toy_{model}_{opt}_N{N}_s{seed}` (+ `_job<id>` suffix on
  cluster); save root `$TOY_SPECTRA_BASE_SAVE_DIR` (default
  `$EXCHANGEABILITY_ROOT/toy_spectra`).
- Each run dir: `metadata.json`, `metrics.jsonl`, `spectra_{P}.npz` at
  log-spaced sample counts P (including P=0 init reference).
- The key figure is outlier-fraction-vs-P per width
  (`*_outlier_fraction_sgd_vs_muon.*`): width-collapse under Muon vs
  width-separated curves under SGD.
