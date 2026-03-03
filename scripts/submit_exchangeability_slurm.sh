#!/usr/bin/env bash
#SBATCH --job-name=imgnet-exchg
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --partition=kempner
#SBATCH --array=0-19
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=72:00:00

set -euo pipefail

MANIFEST_PATH="${1:-conf/exchangeability_manifest.csv}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This script must be submitted with sbatch." >&2
  echo "Usage: sbatch scripts/submit_exchangeability_slurm.sh [manifest_path]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT_DIR}/scripts/cluster_env.sh" ]]; then
  source "${ROOT_DIR}/scripts/cluster_env.sh"
fi

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "Manifest not found: ${MANIFEST_PATH}" >&2
  exit 1
fi

TOTAL_ROWS=$(( $(wc -l < "${MANIFEST_PATH}") - 1 ))
if [[ ${TOTAL_ROWS} -le 0 ]]; then
  echo "Manifest has no rows: ${MANIFEST_PATH}" >&2
  exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ ${TASK_ID} -ge ${TOTAL_ROWS} ]]; then
  echo "Skipping task_id=${TASK_ID}; manifest only has ${TOTAL_ROWS} rows."
  exit 0
fi

LOG_DIR="${SLURM_LOG_DIR:-${BASE_SAVE_DIR:-/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/exchangeability_outputs}/slurm_logs}"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/exchangeability_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.out") 2>&1

echo "Running exchangeability row ${TASK_ID} / ${TOTAL_ROWS} in job ${SLURM_ARRAY_JOB_ID}"
cd "${ROOT_DIR}"
uv run --extra cluster python scripts/run_manifest_row.py --manifest "${MANIFEST_PATH}" --index "${TASK_ID}"
