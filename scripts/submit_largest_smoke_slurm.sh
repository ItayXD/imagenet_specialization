#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT_DIR}/scripts/cluster_env.sh" ]]; then
  source "${ROOT_DIR}/scripts/cluster_env.sh"
fi
EXPERIMENT="${1:-exchangeability_w512_g0}"
MAX_TRANCHES="${2:-50}"
TARGET_IMAGES_SEEN="${3:-10000000}"
SAFETY_FACTOR="${4:-1.35}"

: "${SBATCH_ACCOUNT:=kempner_pehlevan_lab}"
: "${SBATCH_PARTITION:=kempner}"
: "${SBATCH_GPUS:=1}"
: "${SBATCH_CPUS:=24}"
: "${SBATCH_MEM:=128G}"
: "${SBATCH_TIME:=04:00:00}"
: "${PY_LAUNCH:=uv run python}"

LOG_DIR="${SLURM_LOG_DIR:-${BASE_SAVE_DIR:-/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/exchangeability_outputs}/slurm_logs}"
mkdir -p "${LOG_DIR}"

sbatch \
  --job-name=imgnet-smoke \
  --account="${SBATCH_ACCOUNT}" \
  --partition="${SBATCH_PARTITION}" \
  --gpus="${SBATCH_GPUS}" \
  --cpus-per-task="${SBATCH_CPUS}" \
  --mem="${SBATCH_MEM}" \
  --time="${SBATCH_TIME}" \
  --output="${LOG_DIR}/largest_smoke_%j.out" \
  --wrap="set -euo pipefail; cd \"${ROOT_DIR}\"; ${PY_LAUNCH} scripts/run_largest_smoke.py --experiment \"${EXPERIMENT}\" --max-tranches \"${MAX_TRANCHES}\" --target-images-seen \"${TARGET_IMAGES_SEEN}\" --safety-factor \"${SAFETY_FACTOR}\""
