#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT_DIR}/scripts/cluster_env.sh" ]]; then
  source "${ROOT_DIR}/scripts/cluster_env.sh"
fi

mkdir -p "${EXCHANGEABILITY_ROOT}"
mkdir -p "${IMAGENET_FOLDER}"
mkdir -p "${BASE_SAVE_DIR}"
mkdir -p "${BASE_SAVE_DIR}/slurm_logs"

echo "Created/verified:"
echo "  EXCHANGEABILITY_ROOT=${EXCHANGEABILITY_ROOT}"
echo "  IMAGENET_FOLDER=${IMAGENET_FOLDER}"
echo "  BASE_SAVE_DIR=${BASE_SAVE_DIR}"
echo
echo "Next step: download/export ImageNet into ${IMAGENET_FOLDER}:"
echo "  bash scripts/download_imagenet.sh"
echo "This expects HF_TOKEN in ~/.secrets (and optional HF_IMAGENET_REPO_ID)."
