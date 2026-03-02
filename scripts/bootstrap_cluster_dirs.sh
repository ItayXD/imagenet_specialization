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
echo "Next step: place ImageNet data in ${IMAGENET_FOLDER}."
echo "Accepted layouts:"
echo "  1) extracted train/ and val/ folders"
echo "  2) raw official archives (train/val/devkit) for torchvision parsing"
