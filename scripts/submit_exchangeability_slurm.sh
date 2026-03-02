#!/usr/bin/env bash
set -euo pipefail

MANIFEST_PATH="${1:-conf/exchangeability_manifest.csv}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT_DIR}/scripts/cluster_env.sh" ]]; then
  source "${ROOT_DIR}/scripts/cluster_env.sh"
fi

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "Manifest not found: ${MANIFEST_PATH}" >&2
  exit 1
fi

TOTAL_ROWS=$(( $(wc -l < "${MANIFEST_PATH}") - 1 ))
if [[ ${TOTAL_ROWS} -le 0 ]]; then
  echo "Manifest has no rows: ${MANIFEST_PATH}" >&2
  exit 1
fi

: "${SBATCH_GPUS:=1}"
: "${SBATCH_CPUS:=24}"
: "${SBATCH_MEM:=128G}"
: "${SBATCH_TIME:=72:00:00}"
: "${SBATCH_ACCOUNT:=kempner_pehlevan_lab}"
: "${SBATCH_PARTITION:=kempner}"
: "${PY_LAUNCH:=uv run python}"

LOG_DIR="${SLURM_LOG_DIR:-${BASE_SAVE_DIR:-/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/exchangeability_outputs}/slurm_logs}"
mkdir -p "${LOG_DIR}"

sbatch \
  --job-name=imgnet-exchg \
  --account="${SBATCH_ACCOUNT}" \
  --partition="${SBATCH_PARTITION}" \
  --array=0-$((TOTAL_ROWS - 1)) \
  --gpus=${SBATCH_GPUS} \
  --cpus-per-task=${SBATCH_CPUS} \
  --mem=${SBATCH_MEM} \
  --time=${SBATCH_TIME} \
  --output="${LOG_DIR}/exchangeability_%A_%a.out" \
  --export=ALL,MANIFEST_PATH="${MANIFEST_PATH}",ROOT_DIR="${ROOT_DIR}" \
  --wrap="set -euo pipefail; cd \"\$ROOT_DIR\"; ${PY_LAUNCH} scripts/run_manifest_row.py --manifest \"\$MANIFEST_PATH\" --index \"\$SLURM_ARRAY_TASK_ID\""
