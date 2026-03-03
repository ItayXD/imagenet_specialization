#!/usr/bin/env bash
#SBATCH --job-name=imgnet-smoke-test
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --partition=kempner
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=04:00:00

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This script must be submitted with sbatch." >&2
  echo "Usage: sbatch scripts/submit_largest_smoke_pytest_slurm.sh" >&2
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

LOG_DIR="${SLURM_LOG_DIR:-${BASE_SAVE_DIR:-/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/exchangeability_outputs}/slurm_logs}"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/largest_smoke_test_${SLURM_JOB_ID}.out") 2>&1

PY_BIN="${UV_PROJECT_ENVIRONMENT}/bin/python"
if [[ ! -x "${PY_BIN}" ]]; then
  echo "Missing Python env at ${UV_PROJECT_ENVIRONMENT}" >&2
  echo "Run once before submitting jobs:" >&2
  echo "  source scripts/cluster_env.sh && uv sync --extra cluster" >&2
  exit 2
fi

echo "Running largest smoke pytest harness on job ${SLURM_JOB_ID}"
echo "Using UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
echo "Using smoke minibatch/microbatch: ${LARGEST_SMOKE_MINIBATCH_SIZE:-256}/${LARGEST_SMOKE_MICROBATCH_SIZE:-32}"
cd "${ROOT_DIR}"
RUN_LARGEST_SMOKE=1 \
LARGEST_SMOKE_MINIBATCH_SIZE="${LARGEST_SMOKE_MINIBATCH_SIZE:-256}" \
LARGEST_SMOKE_MICROBATCH_SIZE="${LARGEST_SMOKE_MICROBATCH_SIZE:-32}" \
"${PY_BIN}" -m pytest -q test/test_largest_smoke_harness.py
