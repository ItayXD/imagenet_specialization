#!/usr/bin/env bash
#SBATCH --job-name=cifar5m-timing
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --partition=kempner
#SBATCH --array=0-4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This script must be submitted with sbatch." >&2
  echo "Usage: sbatch conf/slurm_jobs/submit_exchangeability_cifar5m_timing_sparse.sbatch" >&2
  exit 2
fi

ROOT_DIR="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -f "${ROOT_DIR}/pyproject.toml" ]]; then
  echo "Could not locate project root at ${ROOT_DIR}." >&2
  echo "Submit from repo root or set PROJECT_ROOT explicitly." >&2
  exit 2
fi

if [[ -f "${ROOT_DIR}/scripts/cluster_env.sh" ]]; then
  source "${ROOT_DIR}/scripts/cluster_env.sh"
fi

bash "${ROOT_DIR}/scripts/submit_timing_sweep_slurm.sh" \
  "${ROOT_DIR}/conf/manifests/cifar5m/exchangeability_manifest_timing_sparse.csv" \
  20 \
  5000000 \
  1.2 \
  "" \
  "${BASE_SAVE_DIR}/timing_sweep_cifar5m"
