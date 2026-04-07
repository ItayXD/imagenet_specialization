#!/usr/bin/env bash
#SBATCH --job-name=single-net-subpart-w1
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --partition=kempner
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This script must be submitted with sbatch." >&2
  echo "Usage: sbatch scripts/submit_single_network_subpart_w1_slurm.sh <width>" >&2
  exit 2
fi

WIDTH="${1:-}"
if [[ -z "${WIDTH}" ]]; then
  echo "Missing width argument." >&2
  echo "Usage: sbatch scripts/submit_single_network_subpart_w1_slurm.sh <width>" >&2
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

DATASET="${EXCHANGEABILITY_DATASET:-imagenet}"
case "${DATASET}" in
  imagenet|cifar5m)
    ;;
  *)
    echo "Unsupported EXCHANGEABILITY_DATASET=${DATASET}. Expected imagenet or cifar5m." >&2
    exit 2
    ;;
esac

default_base_save_dir_for_dataset() {
  if [[ "${DATASET}" == "cifar5m" ]]; then
    printf '%s\n' "${CIFAR5M_BASE_SAVE_DIR:-/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/exchangeability_cifar5m}"
  else
    printf '%s\n' "${IMAGENET_BASE_SAVE_DIR:-/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/exchangeability_imagenet}"
  fi
}

default_run_id_for_dataset() {
  if [[ "${DATASET}" == "cifar5m" ]]; then
    printf '%s\n' "exchangeability_cifar5m"
  else
    printf '%s\n' "exchangeability"
  fi
}

resolve_path() {
  local raw_path="$1"
  local base_dir="$2"
  if [[ "${raw_path}" == /* ]]; then
    printf '%s\n' "${raw_path}"
  else
    printf '%s\n' "${base_dir}/${raw_path}"
  fi
}

BASE_SAVE_DIR="${BASE_SAVE_DIR:-$(default_base_save_dir_for_dataset)}"
export BASE_SAVE_DIR
RUN_ID="${SINGLE_NETWORK_SUBPART_W1_RUN_ID:-$(default_run_id_for_dataset)}"
RUN_ID_RESOLUTION="${SINGLE_NETWORK_SUBPART_W1_RUN_ID_RESOLUTION:-latest_prefix}"
CANONICAL_OUTPUT_CSV="${BASE_SAVE_DIR}/single_network_subpart_w1.csv"
CANONICAL_SUMMARY_CSV="${BASE_SAVE_DIR}/single_network_subpart_w1_summary.csv"

OUTPUT_CSV_OVERRIDE="${SINGLE_NETWORK_SUBPART_W1_OUTPUT_CSV:-}"
if [[ -n "${OUTPUT_CSV_OVERRIDE}" ]]; then
  OUTPUT_CSV="$(resolve_path "${OUTPUT_CSV_OVERRIDE}" "${BASE_SAVE_DIR}")"
else
  OUTPUT_CSV="${BASE_SAVE_DIR}/single_network_subpart_w1_w${WIDTH}.csv"
fi

SUMMARY_CSV_OVERRIDE="${SINGLE_NETWORK_SUBPART_W1_SUMMARY_CSV:-}"
if [[ -n "${SUMMARY_CSV_OVERRIDE}" ]]; then
  SUMMARY_CSV="$(resolve_path "${SUMMARY_CSV_OVERRIDE}" "${BASE_SAVE_DIR}")"
else
  SUMMARY_CSV="${BASE_SAVE_DIR}/single_network_subpart_w1_summary_w${WIDTH}.csv"
fi

LOG_DIR="${SLURM_LOG_DIR:-${BASE_SAVE_DIR}/slurm_logs}"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/single_network_subpart_w1_${DATASET}_w${WIDTH}_${SLURM_JOB_ID}.out") 2>&1

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not available on PATH." >&2
  echo "Run once before submitting jobs:" >&2
  echo "  source scripts/cluster_env.sh && uv sync --extra cluster" >&2
  exit 2
fi

if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" && ! -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]]; then
  echo "Missing Python env at ${UV_PROJECT_ENVIRONMENT}" >&2
  echo "Run once before submitting jobs:" >&2
  echo "  source scripts/cluster_env.sh && uv sync --extra cluster" >&2
  exit 2
fi

FRACTIONS="${SINGLE_NETWORK_SUBPART_W1_FRACTIONS:-0.25 0.5}"
REPEATS="${SINGLE_NETWORK_SUBPART_W1_REPEATS:-32}"
SEED="${SINGLE_NETWORK_SUBPART_W1_SEED:-20260311}"
STEPS="${SINGLE_NETWORK_SUBPART_W1_STEPS:-}"
MEMBER_INDICES="${SINGLE_NETWORK_SUBPART_W1_MEMBER_INDICES:-}"

echo "Running single-network subpart W1 diagnostics for dataset=${DATASET} width=${WIDTH} in job ${SLURM_JOB_ID}"
echo "Using BASE_SAVE_DIR=${BASE_SAVE_DIR}"
echo "Using run_id=${RUN_ID}"
echo "Using run_id_resolution=${RUN_ID_RESOLUTION}"
echo "Using output_csv=${OUTPUT_CSV}"
echo "Using summary_csv=${SUMMARY_CSV}"
echo "Using fractions=${FRACTIONS}"
echo "Using repeats=${REPEATS}"
echo "Using seed=${SEED}"

cd "${ROOT_DIR}"
CMD=(
  uv run python scripts/diagnose_single_network_subpart_w1.py
  --base-save-dir "${BASE_SAVE_DIR}"
  --run-id "${RUN_ID}"
  --resolution-mode "${RUN_ID_RESOLUTION}"
  --widths "${WIDTH}"
  --fractions
)
for frac in ${FRACTIONS}; do
  CMD+=("${frac}")
done
CMD+=(
  --repeats "${REPEATS}"
  --seed "${SEED}"
  --output-csv "${OUTPUT_CSV}"
  --summary-csv "${SUMMARY_CSV}"
)
if [[ -n "${STEPS}" ]]; then
  CMD+=(--steps)
  for step in ${STEPS}; do
    CMD+=("${step}")
  done
fi
if [[ -n "${MEMBER_INDICES}" ]]; then
  CMD+=(--member-indices)
  for member_index in ${MEMBER_INDICES}; do
    CMD+=("${member_index}")
  done
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

MERGE_AFTER_RUN="$(printf '%s' "${SINGLE_NETWORK_SUBPART_W1_MERGE_AFTER_RUN:-true}" | tr '[:upper:]' '[:lower:]')"
if [[ "${MERGE_AFTER_RUN}" == "1" || "${MERGE_AFTER_RUN}" == "true" || "${MERGE_AFTER_RUN}" == "yes" ]]; then
  MERGED_OUTPUT_OVERRIDE="${SINGLE_NETWORK_SUBPART_W1_MERGED_OUTPUT_CSV:-}"
  if [[ -n "${MERGED_OUTPUT_OVERRIDE}" ]]; then
    MERGED_OUTPUT_CSV="$(resolve_path "${MERGED_OUTPUT_OVERRIDE}" "${BASE_SAVE_DIR}")"
  else
    MERGED_OUTPUT_CSV="${CANONICAL_OUTPUT_CSV}"
  fi

  MERGED_SUMMARY_OVERRIDE="${SINGLE_NETWORK_SUBPART_W1_MERGED_SUMMARY_CSV:-}"
  if [[ -n "${MERGED_SUMMARY_OVERRIDE}" ]]; then
    MERGED_SUMMARY_CSV="$(resolve_path "${MERGED_SUMMARY_OVERRIDE}" "${BASE_SAVE_DIR}")"
  else
    MERGED_SUMMARY_CSV="${CANONICAL_SUMMARY_CSV}"
  fi

  MERGE_RAW_CMD=(
    uv run python scripts/merge_single_network_subpart_w1_csvs.py
    --mode raw
    --inputs-glob "${BASE_SAVE_DIR}/single_network_subpart_w1_w*.csv"
    --inputs-glob "${MERGED_OUTPUT_CSV}"
    --output "${MERGED_OUTPUT_CSV}"
  )
  echo "Merging per-width diagnostic CSVs into ${MERGED_OUTPUT_CSV}"
  echo "Running: ${MERGE_RAW_CMD[*]}"
  "${MERGE_RAW_CMD[@]}"

  MERGE_SUMMARY_CMD=(
    uv run python scripts/merge_single_network_subpart_w1_csvs.py
    --mode summary
    --inputs-glob "${BASE_SAVE_DIR}/single_network_subpart_w1_summary_w*.csv"
    --inputs-glob "${MERGED_SUMMARY_CSV}"
    --output "${MERGED_SUMMARY_CSV}"
  )
  echo "Merging per-width summary CSVs into ${MERGED_SUMMARY_CSV}"
  echo "Running: ${MERGE_SUMMARY_CMD[*]}"
  "${MERGE_SUMMARY_CMD[@]}"
fi
