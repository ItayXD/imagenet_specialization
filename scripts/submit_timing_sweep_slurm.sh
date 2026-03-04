#!/usr/bin/env bash
#SBATCH --job-name=imgnet-timing
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --partition=kempner
#SBATCH --array=0-31
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00

set -euo pipefail

MANIFEST_PATH="${1:-conf/exchangeability_manifest.csv}"
MAX_TRANCHES="${2:-20}"
TARGET_IMAGES_SEEN="${3:-10000000}"
SAFETY_FACTOR="${4:-1.35}"
NUM_WORKERS="${5:-}"
SUMMARY_DIR_ARG="${6:-}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This script must be submitted with sbatch." >&2
  echo "Usage: sbatch scripts/submit_timing_sweep_slurm.sh [manifest_path] [max_tranches] [target_images_seen] [safety_factor] [num_workers_override] [summary_dir]" >&2
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
exec > >(tee -a "${LOG_DIR}/timing_sweep_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.out") 2>&1

if [[ -n "${SUMMARY_DIR_ARG}" ]]; then
  if [[ "${SUMMARY_DIR_ARG}" != /* ]]; then
    SUMMARY_DIR="${ROOT_DIR}/${SUMMARY_DIR_ARG}"
  else
    SUMMARY_DIR="${SUMMARY_DIR_ARG}"
  fi
else
  SUMMARY_DIR="${BASE_SAVE_DIR:-/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/exchangeability_outputs}/timing_sweep"
fi
mkdir -p "${SUMMARY_DIR}"

PY_BIN="${UV_PROJECT_ENVIRONMENT}/bin/python"
if [[ ! -x "${PY_BIN}" ]]; then
  echo "Missing Python env at ${UV_PROJECT_ENVIRONMENT}" >&2
  echo "Run once before submitting jobs:" >&2
  echo "  source scripts/cluster_env.sh && uv sync --extra cluster" >&2
  exit 2
fi

echo "Running timing sweep row ${TASK_ID}/${TOTAL_ROWS} on job ${SLURM_ARRAY_JOB_ID}"
echo "Using UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
echo "Using max_tranches=${MAX_TRANCHES}, target_images_seen=${TARGET_IMAGES_SEEN}, safety_factor=${SAFETY_FACTOR}, num_workers_override=${NUM_WORKERS:-<config>}"
echo "Timing summary directory: ${SUMMARY_DIR}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
fi

cd "${ROOT_DIR}"
CMD=(
  "${PY_BIN}" scripts/run_timing_manifest_row.py
  --manifest "${MANIFEST_PATH}"
  --index "${TASK_ID}"
  --max-tranches "${MAX_TRANCHES}"
  --target-images-seen "${TARGET_IMAGES_SEEN}"
  --safety-factor "${SAFETY_FACTOR}"
  --summary-dir "${SUMMARY_DIR}"
)
if [[ -n "${NUM_WORKERS}" ]]; then
  CMD+=(--num-workers "${NUM_WORKERS}")
fi
"${CMD[@]}"
