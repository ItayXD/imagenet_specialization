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
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${EXCHANGEABILITY_ROOT}/.cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${EXCHANGEABILITY_ROOT}/uv_cache}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${EXCHANGEABILITY_ROOT}/uv_envs/imagenet_specialization-py311}"

export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-kempner_pehlevan_lab}"
export SBATCH_PARTITION="${SBATCH_PARTITION:-kempner}"

export WANDB_PROJECT="${WANDB_PROJECT:-imagenet_specialization}"
export HF_IMAGENET_REPO_ID="${HF_IMAGENET_REPO_ID:-ILSVRC/imagenet-1k}"
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}"
export HF_HOME="${HF_HOME:-${EXCHANGEABILITY_ROOT}/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
