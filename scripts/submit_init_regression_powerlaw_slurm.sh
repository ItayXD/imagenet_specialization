#!/usr/bin/env bash
#SBATCH --job-name=init-reg-powerlaw
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --partition=kempner
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
# Keep SLURM's default logs off $HOME (netscratch dir must exist at submit time).
#SBATCH --output=/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/slurm_logs/init_reg_powerlaw_%j.out
#SBATCH --error=/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/slurm_logs/init_reg_powerlaw_%j.err

# ResNet18-at-init + ridge-regression power-law / SVD analysis. Untrained counterpart of
# submit_classifier_powerlaw_slurm.sh. One output dir per (width, ridge lambda).
#
# Usage: sbatch scripts/submit_init_regression_powerlaw_slurm.sh <dataset> [widths...]
#   e.g. sbatch scripts/submit_init_regression_powerlaw_slurm.sh imagenet 32 64 128 256 512

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This script must be submitted with sbatch." >&2
  echo "Usage: sbatch scripts/submit_init_regression_powerlaw_slurm.sh <dataset> [widths...]" >&2
  exit 2
fi

DATASET="${1:-imagenet}"
shift 1 2>/dev/null || true
WIDTHS=("$@")
if [[ ${#WIDTHS[@]} -eq 0 ]]; then
  read -r -a WIDTHS <<< "${INIT_REG_WIDTHS:-32 64 128 256 512}"
fi

case "${DATASET}" in
  cifar5m|imagenet) ;;
  *) echo "Unsupported dataset=${DATASET}; expected cifar5m or imagenet." >&2; exit 2 ;;
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

BASE_SAVE_DIR="${INIT_REG_BASE_SAVE_DIR:-$(default_base_save_dir_for_dataset)}"
RESIDUAL_SCALE_INIT="${INIT_REG_RESIDUAL_SCALE_INIT:-ones}"
NUM_TRAIN_IMAGES="${INIT_REG_NUM_TRAIN_IMAGES:-150000}"
NUM_VAL_IMAGES="${INIT_REG_NUM_VAL_IMAGES:-50000}"
NUM_CALIB_BATCHES="${INIT_REG_NUM_CALIB_BATCHES:-100}"
RIDGE_REL_LAMBDAS="${INIT_REG_RIDGE_REL_LAMBDAS:-0.0 1e-3 1e-2 1e-1 1.0}"
INIT_SEED="${INIT_REG_INIT_SEED:-0}"
TRAIN_SEED="${INIT_REG_TRAIN_SEED:-1234}"
SEED="${INIT_REG_SEED:-2423}"
EVAL_BATCH_SIZE="${INIT_REG_EVAL_BATCH_SIZE:-250}"
NUM_WORKERS="${INIT_REG_NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
NUM_BINS="${INIT_REG_NUM_BINS:-24}"

OUTPUT_ROOT="${INIT_REG_OUTPUT_ROOT:-${BASE_SAVE_DIR}/init_regression_powerlaw}"

LOG_DIR="${SLURM_LOG_DIR:-${OUTPUT_ROOT}/logs}"
mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"
WIDTHS_TAG="$(IFS=-; echo "${WIDTHS[*]}")"
exec > >(tee -a "${LOG_DIR}/init_reg_powerlaw_${DATASET}_w${WIDTHS_TAG}_${SLURM_JOB_ID}.out") 2>&1

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
  uv run python -u scripts/analyze_init_regression_powerlaw.py
  --dataset "${DATASET}"
  --widths "${WIDTHS[@]}"
  --base-save-dir "${BASE_SAVE_DIR}"
  --residual-scale-init "${RESIDUAL_SCALE_INIT}"
  --num-train-images "${NUM_TRAIN_IMAGES}"
  --num-val-images "${NUM_VAL_IMAGES}"
  --num-calib-batches "${NUM_CALIB_BATCHES}"
  --ridge-rel-lambdas ${RIDGE_REL_LAMBDAS}
  --init-seed "${INIT_SEED}"
  --train-seed "${TRAIN_SEED}"
  --seed "${SEED}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --num-bins "${NUM_BINS}"
  --output-root "${OUTPUT_ROOT}"
)

echo "Running at-init regression power-law analysis in job ${SLURM_JOB_ID}"
echo "dataset=${DATASET} widths=${WIDTHS[*]} residual_scale_init=${RESIDUAL_SCALE_INIT}"
echo "ridge_rel_lambdas=${RIDGE_REL_LAMBDAS}"
echo "base_save_dir=${BASE_SAVE_DIR}"
echo "output_root=${OUTPUT_ROOT}"
echo "num_train_images=${NUM_TRAIN_IMAGES} num_val_images=${NUM_VAL_IMAGES} calib_batches=${NUM_CALIB_BATCHES}"
echo "command=${CMD[*]}"
"${CMD[@]}"
