#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT_DIR}/scripts/cluster_env.sh" ]]; then
  source "${ROOT_DIR}/scripts/cluster_env.sh"
fi

mkdir -p "${EXCHANGEABILITY_ROOT}"
mkdir -p "${IMAGENET_FOLDER}"
mkdir -p "${CIFAR5M_FOLDER}/raw"
mkdir -p "${CIFAR5M_FOLDER}/splits"
mkdir -p "${BASE_SAVE_DIR}"
mkdir -p "${BASE_SAVE_DIR}/slurm_logs"
mkdir -p "${HF_HOME}"
mkdir -p "${HF_DATASETS_CACHE}"
mkdir -p "${HUGGINGFACE_HUB_CACHE}"
mkdir -p "${UV_CACHE_DIR}"
mkdir -p "$(dirname "${UV_PROJECT_ENVIRONMENT}")"

echo "Created/verified:"
echo "  EXCHANGEABILITY_ROOT=${EXCHANGEABILITY_ROOT}"
echo "  IMAGENET_FOLDER=${IMAGENET_FOLDER}"
echo "  CIFAR5M_FOLDER=${CIFAR5M_FOLDER}"
echo "  BASE_SAVE_DIR=${BASE_SAVE_DIR}"
echo "  HF_HOME=${HF_HOME}"
echo "  HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
echo "  HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}"
echo "  UV_CACHE_DIR=${UV_CACHE_DIR}"
echo "  UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
echo
echo "Next step: download/export ImageNet into ${IMAGENET_FOLDER}:"
echo "  bash scripts/download_imagenet.sh"
echo "This expects HF_TOKEN in ~/.secrets (and optional HF_IMAGENET_REPO_ID)."
echo
echo "Next step: download CIFAR-5M into ${CIFAR5M_FOLDER}:"
echo "  bash scripts/download_cifar5m.sh"
