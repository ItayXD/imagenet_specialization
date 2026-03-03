#!/usr/bin/env bash
#SBATCH --job-name=imgnet-fast-tests
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --partition=kempner
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This script must be submitted with sbatch." >&2
  echo "Usage: sbatch scripts/submit_fast_tests_slurm.sh" >&2
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
exec > >(tee -a "${LOG_DIR}/fast_tests_${SLURM_JOB_ID}.out") 2>&1

PY_BIN="${UV_PROJECT_ENVIRONMENT}/bin/python"
if [[ ! -x "${PY_BIN}" ]]; then
  echo "Missing Python env at ${UV_PROJECT_ENVIRONMENT}" >&2
  echo "Run once before submitting jobs:" >&2
  echo "  source scripts/cluster_env.sh && uv sync --extra cluster" >&2
  exit 2
fi

echo "Running fast tests on job ${SLURM_JOB_ID}"
echo "Using UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
cd "${ROOT_DIR}"
"${PY_BIN}" -m pytest -q test/test_exchangeability_utils.py test/test_manifest_builder.py
