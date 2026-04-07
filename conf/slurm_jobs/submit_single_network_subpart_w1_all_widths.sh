#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${PROJECT_ROOT:-$(pwd)}"
if [[ ! -f "${ROOT_DIR}/pyproject.toml" ]]; then
  echo "Could not locate project root at ${ROOT_DIR}." >&2
  echo "Run from repo root or set PROJECT_ROOT explicitly." >&2
  exit 2
fi
cd "${ROOT_DIR}"

DATASET="${EXCHANGEABILITY_DATASET:-imagenet}"
case "${DATASET}" in
  imagenet)
    widths=(32 64 128 256 512)
    ;;
  cifar5m)
    widths=(32 64 128 256)
    ;;
  *)
    echo "Unsupported EXCHANGEABILITY_DATASET=${DATASET}. Expected imagenet or cifar5m." >&2
    exit 2
    ;;
esac

for width in "${widths[@]}"; do
  sbatch "conf/slurm_jobs/submit_single_network_subpart_w1_w${width}.sbatch"
done
