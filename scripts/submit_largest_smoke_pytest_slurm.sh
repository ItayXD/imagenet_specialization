#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT_DIR}/scripts/cluster_env.sh" ]]; then
  source "${ROOT_DIR}/scripts/cluster_env.sh"
fi

: "${SBATCH_ACCOUNT:=kempner_pehlevan_lab}"
: "${SBATCH_PARTITION:=kempner}"
: "${SBATCH_GPUS:=1}"
: "${SBATCH_CPUS:=24}"
: "${SBATCH_MEM:=128G}"
: "${SBATCH_TIME:=04:00:00}"
: "${PY_LAUNCH:=uv run}"

LOG_DIR="${SLURM_LOG_DIR:-${BASE_SAVE_DIR:-/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/exchangeability_outputs}/slurm_logs}"
mkdir -p "${LOG_DIR}"

sbatch \
  --job-name=imgnet-smoke-test \
  --account="${SBATCH_ACCOUNT}" \
  --partition="${SBATCH_PARTITION}" \
  --gpus="${SBATCH_GPUS}" \
  --cpus-per-task="${SBATCH_CPUS}" \
  --mem="${SBATCH_MEM}" \
  --time="${SBATCH_TIME}" \
  --output="${LOG_DIR}/largest_smoke_test_%j.out" \
  --wrap="set -euo pipefail; cd \"${ROOT_DIR}\"; RUN_LARGEST_SMOKE=1 ${PY_LAUNCH} pytest -q test/test_largest_smoke_harness.py"
