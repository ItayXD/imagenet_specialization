#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${PROJECT_ROOT:-$(pwd)}"
if [[ ! -f "${ROOT_DIR}/pyproject.toml" ]]; then
  echo "Could not locate project root at ${ROOT_DIR}." >&2
  echo "Run from repo root or set PROJECT_ROOT explicitly." >&2
  exit 2
fi
cd "${ROOT_DIR}"

sbatch "conf/slurm_jobs/submit_exchangeability_w32.sbatch"
sbatch "conf/slurm_jobs/submit_exchangeability_w64.sbatch"
sbatch "conf/slurm_jobs/submit_exchangeability_w128.sbatch"
sbatch "conf/slurm_jobs/submit_exchangeability_w256.sbatch"
sbatch "conf/slurm_jobs/submit_exchangeability_w512.sbatch"
