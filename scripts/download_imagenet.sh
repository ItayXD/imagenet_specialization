#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT_DIR}/scripts/cluster_env.sh" ]]; then
  source "${ROOT_DIR}/scripts/cluster_env.sh"
fi

mkdir -p "${IMAGENET_FOLDER}"

_download() {
  local url="$1"
  local dest="$2"
  local tmp="${dest}.part"

  if [[ -f "${dest}" ]]; then
    echo "Exists, skipping: ${dest}"
    return
  fi

  echo "Downloading -> ${dest}"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c --allow-overwrite=true --continue=true --max-tries=10 --retry-wait=5 --out "$(basename "${dest}")" --dir "$(dirname "${dest}")" "${url}"
  else
    curl --fail --location --retry 10 --retry-delay 5 --output "${tmp}" "${url}"
    mv "${tmp}" "${dest}"
  fi
}

_require_url() {
  local name="$1"
  local value="${!name:-}"
  if [[ -z "${value}" ]]; then
    echo "Missing ${name}. Set it in ~/.secrets or the shell before running." >&2
    echo "Expected variables: IMAGENET_TRAIN_URL IMAGENET_VAL_URL IMAGENET_DEVKIT_URL" >&2
    exit 1
  fi
}

_require_url IMAGENET_TRAIN_URL
_require_url IMAGENET_VAL_URL
_require_url IMAGENET_DEVKIT_URL

TRAIN_TAR="${IMAGENET_FOLDER}/ILSVRC2012_img_train.tar"
VAL_TAR="${IMAGENET_FOLDER}/ILSVRC2012_img_val.tar"
DEVKIT_TAR="${IMAGENET_FOLDER}/ILSVRC2012_devkit_t12.tar.gz"

_download "${IMAGENET_TRAIN_URL}" "${TRAIN_TAR}"
_download "${IMAGENET_VAL_URL}" "${VAL_TAR}"
_download "${IMAGENET_DEVKIT_URL}" "${DEVKIT_TAR}"

if [[ "${IMAGENET_PREPARE_ARCHIVES:-1}" == "1" ]]; then
  echo "Preparing torchvision train/val folders from archives..."
  uv run python scripts/prepare_imagenet_archives.py --imagenet-root "${IMAGENET_FOLDER}"
fi

echo "ImageNet assets ready under ${IMAGENET_FOLDER}"
