#!/usr/bin/env bash
#SBATCH --job-name=imgnet-exchg-analysis
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --partition=kempner
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=6:00:00

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This script must be submitted with sbatch." >&2
  echo "Usage: sbatch scripts/submit_exchangeability_analysis_slurm.sh <width>" >&2
  exit 2
fi

WIDTH="${1:-}"
if [[ -z "${WIDTH}" ]]; then
  echo "Missing width argument." >&2
  echo "Usage: sbatch scripts/submit_exchangeability_analysis_slurm.sh <width>" >&2
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
CANONICAL_OUTPUT_CSV="${BASE_SAVE_DIR}/exchangeability_metrics.csv"

resolve_path() {
  local raw_path="$1"
  local base_dir="$2"
  if [[ "${raw_path}" == /* ]]; then
    printf '%s\n' "${raw_path}"
  else
    printf '%s\n' "${base_dir}/${raw_path}"
  fi
}

OUTPUT_CSV_OVERRIDE="${EXCHANGEABILITY_OUTPUT_CSV:-}"
if [[ -n "${OUTPUT_CSV_OVERRIDE}" ]]; then
  OUTPUT_CSV="$(resolve_path "${OUTPUT_CSV_OVERRIDE}" "${BASE_SAVE_DIR}")"
else
  OUTPUT_CSV="${BASE_SAVE_DIR}/exchangeability_metrics_w${WIDTH}.csv"
fi

SIMILARITY_OUTPUT_DIR_OVERRIDE="${EXCHANGEABILITY_SIMILARITY_OUTPUT_DIR:-}"
if [[ -n "${SIMILARITY_OUTPUT_DIR_OVERRIDE}" ]]; then
  SIMILARITY_OUTPUT_DIR="$(resolve_path "${SIMILARITY_OUTPUT_DIR_OVERRIDE}" "${BASE_SAVE_DIR}")"
else
  # Keep default compatibility with notebook ECDF caching:
  # notebooks/exchangeability_plots.ipynb expects exchangeability_metrics_similarity.
  SIMILARITY_OUTPUT_DIR="${BASE_SAVE_DIR}/exchangeability_metrics_similarity"
fi

LOG_DIR="${SLURM_LOG_DIR:-${BASE_SAVE_DIR}/slurm_logs}"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/exchangeability_analysis_w${WIDTH}_${SLURM_JOB_ID}.out") 2>&1

PY_BIN="${UV_PROJECT_ENVIRONMENT}/bin/python"
if [[ ! -x "${PY_BIN}" ]]; then
  echo "Missing Python env at ${UV_PROJECT_ENVIRONMENT}" >&2
  echo "Run once before submitting jobs:" >&2
  echo "  source scripts/cluster_env.sh && uv sync --extra cluster" >&2
  exit 2
fi

RUN_ID="${EXCHANGEABILITY_RUN_ID:-exchangeability}"
RUN_ID_RESOLUTION="${EXCHANGEABILITY_RUN_ID_RESOLUTION:-latest_prefix}"
SHUFFLE_REPEATS="${EXCHANGEABILITY_SHUFFLE_REPEATS:-2000}"
SHUFFLE_STATS_WORKERS="${EXCHANGEABILITY_SHUFFLE_STATS_WORKERS:-0}"
PROBE_BATCH_SIZE="${EXCHANGEABILITY_PROBE_BATCH_SIZE:-1024}"
LOG_EVERY_SHUFFLES="${EXCHANGEABILITY_LOG_EVERY_SHUFFLES:-50}"
WRITE_EVERY_SHUFFLES="${EXCHANGEABILITY_WRITE_EVERY_SHUFFLES:-50}"
PROBE_SEED="${EXCHANGEABILITY_PROBE_SEED:-1234}"

if [[ -n "${EXCHANGEABILITY_SHUFFLE_BATCH_SIZE:-}" ]]; then
  SHUFFLE_BATCH_SIZE="${EXCHANGEABILITY_SHUFFLE_BATCH_SIZE}"
elif [[ "${WIDTH}" -ge 512 ]]; then
  SHUFFLE_BATCH_SIZE=16
else
  SHUFFLE_BATCH_SIZE=32
fi

if [[ -n "${EXCHANGEABILITY_PROBE_LOADER_BATCH_SIZE:-}" ]]; then
  PROBE_LOADER_BATCH_SIZE="${EXCHANGEABILITY_PROBE_LOADER_BATCH_SIZE}"
elif [[ "${WIDTH}" -ge 512 ]]; then
  PROBE_LOADER_BATCH_SIZE=32
else
  PROBE_LOADER_BATCH_SIZE=128
fi

if [[ -n "${EXCHANGEABILITY_ACTIVATION_CHUNK_SIZE:-}" ]]; then
  ACTIVATION_CHUNK_SIZE="${EXCHANGEABILITY_ACTIVATION_CHUNK_SIZE}"
elif [[ "${WIDTH}" -ge 512 ]]; then
  ACTIVATION_CHUNK_SIZE=8
else
  ACTIVATION_CHUNK_SIZE=0
fi

RESUME_SETTING="$(printf '%s' "${EXCHANGEABILITY_RESUME:-true}" | tr '[:upper:]' '[:lower:]')"
case "${RESUME_SETTING}" in
  true|1|yes|resume)
    RESUME_FLAG="--resume"
    ;;
  false|0|no|no-resume)
    RESUME_FLAG="--no-resume"
    ;;
  *)
    echo "Invalid EXCHANGEABILITY_RESUME value: ${EXCHANGEABILITY_RESUME:-}" >&2
    echo "Expected one of: true,false,1,0,yes,no,resume,no-resume" >&2
    exit 2
    ;;
esac

if [[ "${RESUME_FLAG}" == "--no-resume" && -f "${OUTPUT_CSV}" ]]; then
  ALLOW_OVERWRITE="$(printf '%s' "${EXCHANGEABILITY_ALLOW_OVERWRITE:-false}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${ALLOW_OVERWRITE}" != "1" && "${ALLOW_OVERWRITE}" != "true" && "${ALLOW_OVERWRITE}" != "yes" ]]; then
    echo "Refusing to run with --no-resume because output CSV already exists: ${OUTPUT_CSV}" >&2
    echo "Set EXCHANGEABILITY_ALLOW_OVERWRITE=true only if you want to recompute from scratch." >&2
    exit 2
  fi
fi

echo "Running exchangeability analysis for width=${WIDTH} in job ${SLURM_JOB_ID}"
echo "Using UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
echo "Using BASE_SAVE_DIR=${BASE_SAVE_DIR}"
echo "Using output_csv=${OUTPUT_CSV}"
echo "Using similarity_output_dir=${SIMILARITY_OUTPUT_DIR}"
echo "Using resume flag=${RESUME_FLAG}"
echo "Using shuffle_batch_size=${SHUFFLE_BATCH_SIZE}"
echo "Using probe_loader_batch_size=${PROBE_LOADER_BATCH_SIZE}"
echo "Using activation_chunk_size=${ACTIVATION_CHUNK_SIZE}"

cd "${ROOT_DIR}"
CMD=(
  "${PY_BIN}" scripts/analyze_exchangeability.py
  --base-save-dir "${BASE_SAVE_DIR}"
  --run-id "${RUN_ID}"
  --run-id-resolution "${RUN_ID_RESOLUTION}"
  --output-csv "${OUTPUT_CSV}"
  --similarity-output-dir "${SIMILARITY_OUTPUT_DIR}"
  --shuffle-repeats "${SHUFFLE_REPEATS}"
  --shuffle-batch-size "${SHUFFLE_BATCH_SIZE}"
  --shuffle-stats-workers "${SHUFFLE_STATS_WORKERS}"
  --probe-batch-size "${PROBE_BATCH_SIZE}"
  --probe-loader-batch-size "${PROBE_LOADER_BATCH_SIZE}"
  --activation-chunk-size "${ACTIVATION_CHUNK_SIZE}"
  --probe-seed "${PROBE_SEED}"
  --log-every-shuffles "${LOG_EVERY_SHUFFLES}"
  --write-every-shuffles "${WRITE_EVERY_SHUFFLES}"
  --widths "${WIDTH}"
  "${RESUME_FLAG}"
)
echo "Running: ${CMD[*]}"
"${CMD[@]}"

MERGE_AFTER_RUN="$(printf '%s' "${EXCHANGEABILITY_MERGE_AFTER_RUN:-true}" | tr '[:upper:]' '[:lower:]')"
if [[ "${MERGE_AFTER_RUN}" == "1" || "${MERGE_AFTER_RUN}" == "true" || "${MERGE_AFTER_RUN}" == "yes" ]]; then
  MERGED_OUTPUT_OVERRIDE="${EXCHANGEABILITY_MERGED_OUTPUT_CSV:-}"
  if [[ -n "${MERGED_OUTPUT_OVERRIDE}" ]]; then
    MERGED_OUTPUT_CSV="$(resolve_path "${MERGED_OUTPUT_OVERRIDE}" "${BASE_SAVE_DIR}")"
  else
    MERGED_OUTPUT_CSV="${CANONICAL_OUTPUT_CSV}"
  fi
  MERGE_CMD=(
    "${PY_BIN}" scripts/merge_exchangeability_analysis_csvs.py
    --inputs-glob "${BASE_SAVE_DIR}/exchangeability_metrics_w*.csv"
    --inputs-glob "${MERGED_OUTPUT_CSV}"
    --output "${MERGED_OUTPUT_CSV}"
  )
  echo "Merging per-width analysis CSVs into ${MERGED_OUTPUT_CSV}"
  echo "Running: ${MERGE_CMD[*]}"
  "${MERGE_CMD[@]}"
fi
