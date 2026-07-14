#!/usr/bin/env bash
#SBATCH --job-name=classifier-powerlaw
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --partition=kempner
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This script must be submitted with sbatch." >&2
  echo "Usage: sbatch scripts/submit_classifier_powerlaw_slurm.sh <dataset> <optimizer> [widths...]" >&2
  exit 2
fi

DATASET="${1:-imagenet}"
OPTIMIZER_KEY="${2:-sgd}"
shift 2 2>/dev/null || true
WIDTHS=("$@")
if [[ ${#WIDTHS[@]} -eq 0 ]]; then
  # No widths given: env override, else the full sweep in one job.
  read -r -a WIDTHS <<< "${CLASSIFIER_POWERLAW_WIDTHS:-32 64 128 256 512}"
fi

case "${DATASET}" in
  cifar5m|imagenet) ;;
  *) echo "Unsupported dataset=${DATASET}; expected cifar5m or imagenet." >&2; exit 2 ;;
esac
case "${OPTIMIZER_KEY}" in
  sgd|adam|muon) ;;
  *) echo "Unsupported optimizer=${OPTIMIZER_KEY}; expected sgd, adam, or muon." >&2; exit 2 ;;
esac
for W in "${WIDTHS[@]}"; do
  if ! [[ "${W}" =~ ^[0-9]+$ ]]; then
    echo "Widths must be positive integers; got '${W}'." >&2
    exit 2
  fi
done

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

default_base_save_dir_for_dataset() {
  if [[ "${DATASET}" == "cifar5m" ]]; then
    printf '%s\n' "${CIFAR5M_BASE_SAVE_DIR:-/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/exchangeability_cifar5m}"
  else
    printf '%s\n' "${IMAGENET_BASE_SAVE_DIR:-/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/exchangeability_imagenet}"
  fi
}

BASE_SAVE_DIR="${CLASSIFIER_POWERLAW_BASE_SAVE_DIR:-$(default_base_save_dir_for_dataset)}"
RUN_ID_OVERRIDE="${CLASSIFIER_POWERLAW_RUN_ID:-}"
RUN_ID_RESOLUTION="${CLASSIFIER_POWERLAW_RUN_ID_RESOLUTION:-exact}"
IMAGES_SEEN="${CLASSIFIER_POWERLAW_IMAGES_SEEN:-0}"
NUM_IMAGES="${CLASSIFIER_POWERLAW_NUM_IMAGES:-50000}"
SEED="${CLASSIFIER_POWERLAW_SEED:-2423}"
EVAL_BATCH_SIZE="${CLASSIFIER_POWERLAW_EVAL_BATCH_SIZE:-250}"
NUM_WORKERS="${CLASSIFIER_POWERLAW_NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
NUM_BINS="${CLASSIFIER_POWERLAW_NUM_BINS:-24}"

# Each width writes to ${OUTPUT_ROOT}/${DATASET}/${OPTIMIZER_KEY}_w<width> (the python
# script creates the per-width dirs).
OUTPUT_ROOT="${CLASSIFIER_POWERLAW_OUTPUT_ROOT:-${BASE_SAVE_DIR}/classifier_powerlaw}"

LOG_DIR="${SLURM_LOG_DIR:-${OUTPUT_ROOT}/logs}"
mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"
WIDTHS_TAG="$(IFS=-; echo "${WIDTHS[*]}")"
exec > >(tee -a "${LOG_DIR}/classifier_powerlaw_${DATASET}_${OPTIMIZER_KEY}_w${WIDTHS_TAG}_${SLURM_JOB_ID}.out") 2>&1

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not available on PATH." >&2
  echo "Run once before submitting jobs: source scripts/cluster_env.sh && uv sync --extra cluster" >&2
  exit 2
fi
if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" && ! -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]]; then
  echo "Missing Python env at ${UV_PROJECT_ENVIRONMENT}" >&2
  echo "Run once before submitting jobs: source scripts/cluster_env.sh && uv sync --extra cluster" >&2
  exit 2
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

cd "${ROOT_DIR}"

CMD=(
  uv run python -u scripts/analyze_classifier_powerlaw.py
  --dataset "${DATASET}"
  --optimizer-key "${OPTIMIZER_KEY}"
  --widths "${WIDTHS[@]}"
  --base-save-dir "${BASE_SAVE_DIR}"
  --run-id-resolution "${RUN_ID_RESOLUTION}"
  --images-seen "${IMAGES_SEEN}"
  --num-images "${NUM_IMAGES}"
  --seed "${SEED}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --num-bins "${NUM_BINS}"
  --output-root "${OUTPUT_ROOT}"
)
if [[ -n "${RUN_ID_OVERRIDE}" ]]; then
  CMD+=(--run-id "${RUN_ID_OVERRIDE}")
fi

echo "Running classifier power-law analysis in job ${SLURM_JOB_ID}"
echo "dataset=${DATASET} optimizer=${OPTIMIZER_KEY} widths=${WIDTHS[*]}"
echo "base_save_dir=${BASE_SAVE_DIR}"
echo "output_root=${OUTPUT_ROOT}"
echo "num_images=${NUM_IMAGES} eval_batch_size=${EVAL_BATCH_SIZE} num_workers=${NUM_WORKERS}"
echo "command=${CMD[*]}"
"${CMD[@]}"
