#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT_DIR}/scripts/cluster_env.sh" ]]; then
  source "${ROOT_DIR}/scripts/cluster_env.sh"
fi

: "${SBATCH_ACCOUNT:=kempner_pehlevan_lab}"
: "${SBATCH_PARTITION:=kempner}"
: "${SBATCH_CPUS:=4}"
: "${SBATCH_MEM:=8G}"
: "${SBATCH_TIME:=00:20:00}"
: "${PY_LAUNCH:=uv run}"

LOG_DIR="${SLURM_LOG_DIR:-${BASE_SAVE_DIR:-/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/exchangeability_outputs}/slurm_logs}"
mkdir -p "${LOG_DIR}"

sbatch \
  --job-name=imgnet-fast-tests \
  --account="${SBATCH_ACCOUNT}" \
  --partition="${SBATCH_PARTITION}" \
  --cpus-per-task="${SBATCH_CPUS}" \
  --mem="${SBATCH_MEM}" \
  --time="${SBATCH_TIME}" \
  --output="${LOG_DIR}/fast_tests_%j.out" \
  --wrap="set -euo pipefail; cd \"${ROOT_DIR}\"; ${PY_LAUNCH} pytest -q test/test_exchangeability_utils.py test/test_manifest_builder.py"
