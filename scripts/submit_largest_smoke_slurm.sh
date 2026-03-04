#!/usr/bin/env bash
#SBATCH --job-name=imgnet-smoke
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --partition=kempner
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This script must be submitted with sbatch." >&2
  echo "Usage: sbatch scripts/submit_largest_smoke_slurm.sh [experiment] [max_tranches] [target_images_seen] [safety_factor] [minibatch_size] [microbatch_size] [num_workers]" >&2
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

EXPERIMENT="${1:-exchangeability_w512_g0}"
MAX_TRANCHES="${2:-50}"
TARGET_IMAGES_SEEN="${3:-10000000}"
SAFETY_FACTOR="${4:-1.35}"
MINIBATCH_SIZE="${5:-128}"
MICROBATCH_SIZE="${6:-16}"
NUM_WORKERS="${7:-4}"

LOG_DIR="${SLURM_LOG_DIR:-${BASE_SAVE_DIR:-/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/exchangeability_outputs}/slurm_logs}"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/largest_smoke_${SLURM_JOB_ID}.out") 2>&1

PY_BIN="${UV_PROJECT_ENVIRONMENT}/bin/python"
if [[ ! -x "${PY_BIN}" ]]; then
  echo "Missing Python env at ${UV_PROJECT_ENVIRONMENT}" >&2
  echo "Run once before submitting jobs:" >&2
  echo "  source scripts/cluster_env.sh && uv sync --extra cluster" >&2
  exit 2
fi

echo "Running largest smoke timing job ${SLURM_JOB_ID}"
echo "Using UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
echo "Using smoke minibatch/microbatch/workers: ${MINIBATCH_SIZE}/${MICROBATCH_SIZE}/${NUM_WORKERS}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
fi
cd "${ROOT_DIR}"
"${PY_BIN}" scripts/run_largest_smoke.py \
  --experiment "${EXPERIMENT}" \
  --max-tranches "${MAX_TRANCHES}" \
  --target-images-seen "${TARGET_IMAGES_SEEN}" \
  --safety-factor "${SAFETY_FACTOR}" \
  --minibatch-size "${MINIBATCH_SIZE}" \
  --microbatch-size "${MICROBATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}"
