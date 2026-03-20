#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT_DIR}/scripts/cluster_env.sh" ]]; then
  source "${ROOT_DIR}/scripts/cluster_env.sh"
fi

mkdir -p "${CIFAR5M_FOLDER}/raw"
mkdir -p "${CIFAR5M_FOLDER}/splits"

FORCE_FLAG=""
if [[ "${CIFAR5M_FORCE:-0}" == "1" ]]; then
  FORCE_FLAG="--force"
fi

uv run python scripts/download_cifar5m.py \
  --root "${CIFAR5M_FOLDER}" \
  ${FORCE_FLAG}

echo "CIFAR-5M ready under ${CIFAR5M_FOLDER}"
