#!/usr/bin/env bash
#SBATCH --job-name=cifar5m-exchg-sgd4e-1
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --partition=kempner
#SBATCH --array=0-35
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=72:00:00

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This script must be submitted with sbatch." >&2
  echo "Usage: sbatch conf/slurm_jobs/submit_exchangeability_cifar5m_sgd_lr04.sbatch" >&2
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

export BASE_SAVE_DIR="${BASE_SAVE_DIR:-${CIFAR5M_BASE_SAVE_DIR:-/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/exchangeability_cifar5m}}"

bash "${ROOT_DIR}/scripts/submit_exchangeability_slurm.sh" "${ROOT_DIR}/conf/manifests/cifar5m/exchangeability_manifest_sgd_lr04.csv"
