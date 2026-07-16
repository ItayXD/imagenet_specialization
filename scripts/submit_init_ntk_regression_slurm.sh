#!/usr/bin/env bash
#SBATCH --job-name=init-ntk-reg
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --partition=kempner
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/slurm_logs/init_ntk_reg_%j.out
#SBATCH --error=/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/slurm_logs/init_ntk_reg_%j.err

# Linearized (tangent/NTK) random-feature readout of ResNet18 at init.
# Usage: sbatch scripts/submit_init_ntk_regression_slurm.sh <dataset> <width> [m]

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Submit with sbatch." >&2; exit 2
fi
DATASET="${1:-cifar5m}"
WIDTH="${2:-32}"
M="${3:-4000}"

ROOT_DIR="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
[[ -f "${ROOT_DIR}/scripts/cluster_env.sh" ]] && source "${ROOT_DIR}/scripts/cluster_env.sh"

NUM_TRAIN="${INIT_NTK_NUM_TRAIN:-20000}"
NUM_VAL="${INIT_NTK_NUM_VAL:-10000}"
CALIB="${INIT_NTK_CALIB:-50}"
CHUNK="${INIT_NTK_CHUNK:-16}"
BATCH="${INIT_NTK_BATCH:-200}"
LAMBDAS="${INIT_NTK_LAMBDAS:-0.0 1e-3 1e-2 1e-1 1.0}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
cd "${ROOT_DIR}"

CMD=(
  uv run python -u scripts/analyze_init_ntk_regression.py
  --dataset "${DATASET}" --width "${WIDTH}" --num-features "${M}"
  --num-train-images "${NUM_TRAIN}" --num-val-images "${NUM_VAL}"
  --num-calib-batches "${CALIB}" --jvp-chunk "${CHUNK}"
  --eval-batch-size "${BATCH}" --num-workers "${SLURM_CPUS_PER_TASK:-8}"
  --ridge-rel-lambdas ${LAMBDAS}
)
echo "job ${SLURM_JOB_ID}: dataset=${DATASET} width=${WIDTH} m=${M} train=${NUM_TRAIN} val=${NUM_VAL}"
echo "command=${CMD[*]}"
"${CMD[@]}"
