#!/usr/bin/env bash
#SBATCH --job-name=toy-spectra
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --partition=kempner
#SBATCH --array=0-13
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00

# Resource notes: toy_spectra rows are tiny relative to the ImageNet jobs.
# `--gpus=1` / `--partition=kempner` / `--account=kempner_pehlevan_lab` follow
# the repo convention (see submit_resnet_block_spectra_slurm.sh etc.). The user
# runs at most 14 jobs in parallel, so each array task processes a CHUNK of
# manifest rows (ROWS_PER_TASK, default 1): pick ROWS_PER_TASK and
# --array=0-(ceil(TOTAL/ROWS_PER_TASK)-1) so ~14 (or fewer) tasks cover the
# whole manifest, each doing minutes of real work instead of seconds. mem/time
# are trimmed well below the ImageNet template since these MLP runs use <1G.

set -euo pipefail

MANIFEST_PATH="${1:-conf/toy_spectra_manifest.csv}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This script must be submitted with sbatch." >&2
  echo "Usage: sbatch scripts/submit_toy_spectra_slurm.sh [manifest_path]" >&2
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

BASE_SAVE_DIR="${TOY_SPECTRA_BASE_SAVE_DIR:-${EXCHANGEABILITY_ROOT:-/n/netscratch/kempner_pehlevan_lab/Lab/ilavie}/toy_spectra}"
export BASE_SAVE_DIR

TOTAL_ROWS=$(( $(wc -l < "${MANIFEST_PATH}") - 1 ))
if [[ ${TOTAL_ROWS} -le 0 ]]; then
  echo "Manifest has no rows: ${MANIFEST_PATH}" >&2
  exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ROWS_PER_TASK="${ROWS_PER_TASK:-1}"
ROW_START=$(( TASK_ID * ROWS_PER_TASK ))
ROW_END=$(( ROW_START + ROWS_PER_TASK - 1 ))
if [[ ${ROW_END} -ge ${TOTAL_ROWS} ]]; then ROW_END=$(( TOTAL_ROWS - 1 )); fi
if [[ ${ROW_START} -ge ${TOTAL_ROWS} ]]; then
  echo "Skipping task_id=${TASK_ID}; row_start=${ROW_START} >= ${TOTAL_ROWS} rows."
  exit 0
fi

LOG_DIR="${SLURM_LOG_DIR:-${BASE_SAVE_DIR}/slurm_logs}"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/toy_spectra_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.out") 2>&1
echo "Task ${TASK_ID} handling rows ${ROW_START}..${ROW_END} (ROWS_PER_TASK=${ROWS_PER_TASK})"

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

echo "Running toy-spectra row ${TASK_ID} / ${TOTAL_ROWS} in job ${SLURM_ARRAY_JOB_ID}"
echo "Using UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
echo "Saving under BASE_SAVE_DIR=${BASE_SAVE_DIR}"
cd "${ROOT_DIR}"

RUN_ID_SUFFIX="${RUN_ID_SUFFIX-job${SLURM_ARRAY_JOB_ID}}"
for (( idx=ROW_START; idx<=ROW_END; idx++ )); do
  echo "=== row ${idx} ==="
  CMD=(uv run python scripts/toy_spectra/run_toy_spectra.py
       --manifest "${MANIFEST_PATH}" --index "${idx}"
       --output-dir "${BASE_SAVE_DIR}")
  if [[ -n "${RUN_ID_SUFFIX}" ]]; then
    CMD+=(--run-id-suffix "${RUN_ID_SUFFIX}")
  fi
  "${CMD[@]}"
done
