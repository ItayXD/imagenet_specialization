#!/usr/bin/env bash
#SBATCH --job-name=imgnet-exchg
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --partition=kempner
#SBATCH --array=0-31
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

ROOT_DIR="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -f "${ROOT_DIR}/pyproject.toml" ]]; then
  echo "Could not locate project root at ${ROOT_DIR}." >&2
  echo "Submit from repo root or set PROJECT_ROOT explicitly." >&2
  exit 2
fi
if [[ -f "${ROOT_DIR}/scripts/cluster_env.sh" ]]; then
  source "${ROOT_DIR}/scripts/cluster_env.sh"
fi

if [[ "${MANIFEST_PATH}" != /* ]]; then
  MANIFEST_PATH="${ROOT_DIR}/${MANIFEST_PATH}"
fi

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "Manifest not found: ${MANIFEST_PATH}" >&2
  exit 1
fi

default_base_save_dir_for_manifest() {
  local manifest_path="$1"
  if [[ "${manifest_path}" == *cifar5m* ]]; then
    printf '%s\n' "${CIFAR5M_BASE_SAVE_DIR:-/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/exchangeability_cifar5m}"
  else
    printf '%s\n' "${IMAGENET_BASE_SAVE_DIR:-/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/exchangeability_imagenet}"
  fi
}

BASE_SAVE_DIR="${BASE_SAVE_DIR:-$(default_base_save_dir_for_manifest "${MANIFEST_PATH}")}"
if [[ "${MANIFEST_PATH}" == *cifar5m* ]]; then
  expected_imagenet_base="${IMAGENET_BASE_SAVE_DIR:-/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/exchangeability_imagenet}"
  if [[ "${BASE_SAVE_DIR}" == "${expected_imagenet_base}" ]]; then
    BASE_SAVE_DIR="$(default_base_save_dir_for_manifest "${MANIFEST_PATH}")"
  fi
fi
export BASE_SAVE_DIR

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

LOG_DIR="${SLURM_LOG_DIR:-${BASE_SAVE_DIR}/slurm_logs}"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/exchangeability_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.out") 2>&1

if [[ -z "${UV_PROJECT_ENVIRONMENT:-}" ]]; then
  echo "UV_PROJECT_ENVIRONMENT is not set." >&2
  echo "Run once before submitting jobs:" >&2
  echo "  source scripts/cluster_env.sh && uv sync --extra cluster" >&2
  exit 2
fi

if [[ ! -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]]; then
  echo "Missing Python env at ${UV_PROJECT_ENVIRONMENT}" >&2
  echo "Run once before submitting jobs:" >&2
  echo "  source scripts/cluster_env.sh && uv sync --extra cluster" >&2
  exit 2
fi

echo "Running exchangeability row ${TASK_ID} / ${TOTAL_ROWS} in job ${SLURM_ARRAY_JOB_ID}"
echo "Using UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
cd "${ROOT_DIR}"

RUN_ID_SUFFIX="${RUN_ID_SUFFIX-job${SLURM_ARRAY_JOB_ID}}"
CMD=(uv run python scripts/run_manifest_row.py --manifest "${MANIFEST_PATH}" --index "${TASK_ID}")
if [[ -n "${RUN_ID_SUFFIX}" ]]; then
  echo "Using run_id suffix: ${RUN_ID_SUFFIX}"
  CMD+=(--run-id-suffix "${RUN_ID_SUFFIX}")
else
  echo "RUN_ID_SUFFIX empty; using run_id from experiment config."
fi
"${CMD[@]}"
