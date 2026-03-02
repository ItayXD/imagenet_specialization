#!/usr/bin/env bash
set -euo pipefail

if [[ -f "${HOME}/.secrets" ]]; then
  # shellcheck disable=SC1090
  source "${HOME}/.secrets"
fi

export EXCHANGEABILITY_ROOT="${EXCHANGEABILITY_ROOT:-/n/netscratch/kempner_pehlevan_lab/Lab/ilavie}"
export IMAGENET_FOLDER="${IMAGENET_FOLDER:-${EXCHANGEABILITY_ROOT}/imagenet}"
export BASE_SAVE_DIR="${BASE_SAVE_DIR:-${EXCHANGEABILITY_ROOT}/exchangeability_outputs}"
export REMOTE_RESULTS_FOLDER="${REMOTE_RESULTS_FOLDER:-${EXCHANGEABILITY_ROOT}}"

export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-kempner_pehlevan_lab}"
export SBATCH_PARTITION="${SBATCH_PARTITION:-kempner}"

export WANDB_PROJECT="${WANDB_PROJECT:-imagenet_specialization}"
export HF_IMAGENET_REPO_ID="${HF_IMAGENET_REPO_ID:-ILSVRC/imagenet-1k}"
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}"
