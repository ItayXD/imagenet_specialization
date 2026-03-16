#!/usr/bin/env bash
#SBATCH --job-name=imgnet-layerwise-w1
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --partition=kempner
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=6:00:00

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This script must be submitted with sbatch." >&2
  echo "Usage: sbatch scripts/submit_layerwise_weight_w1_slurm.sh <width>" >&2
  exit 2
fi

WIDTH="${1:-}"
if [[ -z "${WIDTH}" ]]; then
  echo "Missing width argument." >&2
  echo "Usage: sbatch scripts/submit_layerwise_weight_w1_slurm.sh <width>" >&2
  exit 2
fi
if ! [[ "${WIDTH}" =~ ^[0-9]+$ ]]; then
  echo "Width must be a positive integer; got: ${WIDTH}" >&2
  exit 2
fi

ROOT_DIR="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -f "${ROOT_DIR}/pyproject.toml" ]]; then
  echo "Could not locate project root at ${ROOT_DIR}." >&2
  echo "Submit from repo root or set PROJECT_ROOT explicitly." >&2
  exit 2
fi
if [[ -f "${ROOT_DIR}/scripts/cluster_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/scripts/cluster_env.sh"
fi

BASE_SAVE_DIR="${BASE_SAVE_DIR:-/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/exchangeability_outputs}"
CANONICAL_OUTPUT_CSV="${BASE_SAVE_DIR}/layerwise_weight_w1.csv"

resolve_path() {
  local raw_path="$1"
  local base_dir="$2"
  if [[ "${raw_path}" == /* ]]; then
    printf '%s\n' "${raw_path}"
  else
    printf '%s\n' "${base_dir}/${raw_path}"
  fi
}

OUTPUT_CSV_OVERRIDE="${LAYERWISE_WEIGHT_W1_OUTPUT_CSV:-}"
if [[ -n "${OUTPUT_CSV_OVERRIDE}" ]]; then
  OUTPUT_CSV="$(resolve_path "${OUTPUT_CSV_OVERRIDE}" "${BASE_SAVE_DIR}")"
else
  OUTPUT_CSV="${BASE_SAVE_DIR}/layerwise_weight_w1_w${WIDTH}.csv"
fi

LOG_DIR="${SLURM_LOG_DIR:-${BASE_SAVE_DIR}/slurm_logs}"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/layerwise_weight_w1_w${WIDTH}_${SLURM_JOB_ID}.out") 2>&1

PY_BIN="${UV_PROJECT_ENVIRONMENT}/bin/python"
if [[ ! -x "${PY_BIN}" ]]; then
  echo "Missing Python env at ${UV_PROJECT_ENVIRONMENT}" >&2
  echo "Run once before submitting jobs:" >&2
  echo "  source scripts/cluster_env.sh && uv sync --extra cluster" >&2
  exit 2
fi

RUN_ID="${LAYERWISE_WEIGHT_W1_RUN_ID:-exchangeability}"
RUN_ID_RESOLUTION="${LAYERWISE_WEIGHT_W1_RUN_ID_RESOLUTION:-latest_prefix}"
GPU_BLOCK_ROWS="${LAYERWISE_WEIGHT_W1_GPU_BLOCK_ROWS:-0}"

RESUME_SETTING="$(printf '%s' "${LAYERWISE_WEIGHT_W1_RESUME:-true}" | tr '[:upper:]' '[:lower:]')"
case "${RESUME_SETTING}" in
  true|1|yes|resume)
    RESUME_FLAG="--resume"
    ;;
  false|0|no|no-resume)
    RESUME_FLAG="--no-resume"
    ;;
  *)
    echo "Invalid LAYERWISE_WEIGHT_W1_RESUME value: ${LAYERWISE_WEIGHT_W1_RESUME:-}" >&2
    echo "Expected one of: true,false,1,0,yes,no,resume,no-resume" >&2
    exit 2
    ;;
esac

if [[ "${RESUME_FLAG}" == "--no-resume" && -f "${OUTPUT_CSV}" ]]; then
  ALLOW_OVERWRITE="$(printf '%s' "${LAYERWISE_WEIGHT_W1_ALLOW_OVERWRITE:-false}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${ALLOW_OVERWRITE}" != "1" && "${ALLOW_OVERWRITE}" != "true" && "${ALLOW_OVERWRITE}" != "yes" ]]; then
    echo "Refusing to run with --no-resume because output CSV already exists: ${OUTPUT_CSV}" >&2
    echo "Set LAYERWISE_WEIGHT_W1_ALLOW_OVERWRITE=true only if you want to recompute from scratch." >&2
    exit 2
  fi
fi

echo "Running layerwise weight W1 analysis for width=${WIDTH} in job ${SLURM_JOB_ID}"
echo "Using UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
echo "Using BASE_SAVE_DIR=${BASE_SAVE_DIR}"
echo "Using output_csv=${OUTPUT_CSV}"
echo "Using resume flag=${RESUME_FLAG}"
echo "Using gpu_block_rows=${GPU_BLOCK_ROWS}"

cd "${ROOT_DIR}"
CMD=(
  "${PY_BIN}" scripts/analyze_layerwise_weight_w1.py
  --base-save-dir "${BASE_SAVE_DIR}"
  --run-id "${RUN_ID}"
  --run-id-resolution "${RUN_ID_RESOLUTION}"
  --output-csv "${OUTPUT_CSV}"
  --gpu-block-rows "${GPU_BLOCK_ROWS}"
  --widths "${WIDTH}"
  "${RESUME_FLAG}"
)
echo "Running: ${CMD[*]}"
"${CMD[@]}"

MERGE_AFTER_RUN="$(printf '%s' "${LAYERWISE_WEIGHT_W1_MERGE_AFTER_RUN:-true}" | tr '[:upper:]' '[:lower:]')"
if [[ "${MERGE_AFTER_RUN}" == "1" || "${MERGE_AFTER_RUN}" == "true" || "${MERGE_AFTER_RUN}" == "yes" ]]; then
  MERGED_OUTPUT_OVERRIDE="${LAYERWISE_WEIGHT_W1_MERGED_OUTPUT_CSV:-}"
  if [[ -n "${MERGED_OUTPUT_OVERRIDE}" ]]; then
    MERGED_OUTPUT_CSV="$(resolve_path "${MERGED_OUTPUT_OVERRIDE}" "${BASE_SAVE_DIR}")"
  else
    MERGED_OUTPUT_CSV="${CANONICAL_OUTPUT_CSV}"
  fi
  MERGE_CMD=(
    "${PY_BIN}" scripts/merge_layerwise_weight_w1_csvs.py
    --inputs-glob "${BASE_SAVE_DIR}/layerwise_weight_w1_w*.csv"
    --inputs-glob "${MERGED_OUTPUT_CSV}"
    --output "${MERGED_OUTPUT_CSV}"
  )
  echo "Merging per-width layerwise CSVs into ${MERGED_OUTPUT_CSV}"
  echo "Running: ${MERGE_CMD[*]}"
  "${MERGE_CMD[@]}"
fi
