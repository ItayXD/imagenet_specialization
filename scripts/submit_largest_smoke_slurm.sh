#!/usr/bin/env bash
#SBATCH --job-name=imgnet-smoke
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --partition=kempner
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=04:00:00

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT_DIR}/scripts/cluster_env.sh" ]]; then
  source "${ROOT_DIR}/scripts/cluster_env.sh"
fi

EXPERIMENT="${1:-exchangeability_w512_g0}"
MAX_TRANCHES="${2:-50}"
TARGET_IMAGES_SEEN="${3:-10000000}"
SAFETY_FACTOR="${4:-1.35}"

LOG_DIR="${SLURM_LOG_DIR:-${BASE_SAVE_DIR:-/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/exchangeability_outputs}/slurm_logs}"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/largest_smoke_${SLURM_JOB_ID}.out") 2>&1

echo "Running largest smoke timing job ${SLURM_JOB_ID}"
cd "${ROOT_DIR}"
uv run python scripts/run_largest_smoke.py \
  --experiment "${EXPERIMENT}" \
  --max-tranches "${MAX_TRANCHES}" \
  --target-images-seen "${TARGET_IMAGES_SEEN}" \
  --safety-factor "${SAFETY_FACTOR}"
