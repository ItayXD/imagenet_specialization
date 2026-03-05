#!/usr/bin/env bash
#SBATCH --job-name=imgnet-exchg-w256
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --partition=kempner
#SBATCH --array=0-7
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=22:01:47

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This script must be submitted with sbatch." >&2
  echo "Usage: sbatch conf/slurm_jobs/submit_exchangeability_w256.sbatch" >&2
  exit 2
fi

ROOT_DIR="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -f "${ROOT_DIR}/pyproject.toml" ]]; then
  echo "Could not locate project root at ${ROOT_DIR}." >&2
  echo "Submit from repo root or set PROJECT_ROOT explicitly." >&2
  exit 2
fi

bash "${ROOT_DIR}/scripts/submit_exchangeability_slurm.sh" "${ROOT_DIR}/conf/manifests_by_width/exchangeability_manifest_w256.csv"
