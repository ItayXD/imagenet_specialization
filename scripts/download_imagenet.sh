#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT_DIR}/scripts/cluster_env.sh" ]]; then
  source "${ROOT_DIR}/scripts/cluster_env.sh"
fi

mkdir -p "${IMAGENET_FOLDER}"

: "${HF_IMAGENET_REPO_ID:=ILSVRC/imagenet-1k}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is missing. Put it in ~/.secrets and source scripts/cluster_env.sh first." >&2
  exit 1
fi

MAX_TRAIN="${IMAGENET_MAX_TRAIN:-0}"
MAX_VAL="${IMAGENET_MAX_VAL:-0}"
FORCE_FLAG=""
if [[ "${IMAGENET_FORCE:-0}" == "1" ]]; then
  FORCE_FLAG="--force"
fi

echo "Downloading from HF repo: ${HF_IMAGENET_REPO_ID}"
uv run python scripts/download_imagenet_hf.py \
  --repo-id "${HF_IMAGENET_REPO_ID}" \
  --root "${IMAGENET_FOLDER}" \
  --max-train "${MAX_TRAIN}" \
  --max-val "${MAX_VAL}" \
  ${FORCE_FLAG}

echo "ImageNet HF export ready under ${IMAGENET_FOLDER}"
