#!/usr/bin/env bash
#SBATCH --job-name=imgnet-spectrum
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --partition=kempner
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This script must be submitted with sbatch." >&2
  echo "Usage: sbatch scripts/submit_resnet_block_spectra_slurm.sh" >&2
  exit 2
fi

ROOT_DIR="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -f "${ROOT_DIR}/pyproject.toml" ]]; then
  echo "Could not locate project root at ${ROOT_DIR}." >&2
  echo "Submit from repo root or set PROJECT_ROOT explicitly." >&2
  exit 2
fi
if [[ -f "${ROOT_DIR}/scripts/cluster_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/scripts/cluster_env.sh"
fi

INPUT_BASE_SAVE_DIR="${RESNET_BLOCK_SPECTRA_INPUT_BASE_SAVE_DIR:-/n/pehlevan_lab/Users/ilavie/imagenet_specialization_results}"
RUN_ID="${RESNET_BLOCK_SPECTRA_RUN_ID:-exchangeability}"
RUN_ID_RESOLUTION="${RESNET_BLOCK_SPECTRA_RUN_ID_RESOLUTION:-latest_prefix}"
LAYER_SELECTION="${RESNET_BLOCK_SPECTRA_LAYER_SELECTION:-first_block_second_conv}"
ARTIFACT_STEM="${RESNET_BLOCK_SPECTRA_ARTIFACT_STEM:-resnet_block1_conv1_spectra}"

default_base_save_dir_for_analysis() {
  local dataset_hint="${EXCHANGEABILITY_DATASET:-}"
  if [[ -z "${dataset_hint}" ]]; then
    if [[ "${INPUT_BASE_SAVE_DIR}" == *cifar5m* || "${RUN_ID}" == *cifar5m* ]]; then
      dataset_hint="cifar5m"
    else
      dataset_hint="imagenet"
    fi
  fi

  if [[ "${dataset_hint}" == "cifar5m" ]]; then
    printf '%s\n' "${CIFAR5M_BASE_SAVE_DIR:-/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/exchangeability_cifar5m}"
  else
    printf '%s\n' "${IMAGENET_BASE_SAVE_DIR:-/n/netscratch/kempner_pehlevan_lab/Lab/ilavie/exchangeability_imagenet}"
  fi
}

DEFAULT_OUTPUT_BASE_SAVE_DIR="$(default_base_save_dir_for_analysis)"
SPECTRA_DIR="${RESNET_BLOCK_SPECTRA_OUTPUT_DIR:-${DEFAULT_OUTPUT_BASE_SAVE_DIR}/${ARTIFACT_STEM}}"
PLOT_DIR="${RESNET_BLOCK_SPECTRA_PLOT_DIR:-${DEFAULT_OUTPUT_BASE_SAVE_DIR}/plots_${ARTIFACT_STEM}}"
LOG_DIR="${SLURM_LOG_DIR:-${DEFAULT_OUTPUT_BASE_SAVE_DIR}/slurm_logs}"

mkdir -p "${LOG_DIR}" "${SPECTRA_DIR}" "${PLOT_DIR}"
exec > >(tee -a "${LOG_DIR}/resnet_block_spectra_${SLURM_JOB_ID}.out") 2>&1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

echo "Running spectrum extraction in job ${SLURM_JOB_ID}"
echo "Input base save dir: ${INPUT_BASE_SAVE_DIR}"
echo "Run resolution: run_id=${RUN_ID} mode=${RUN_ID_RESOLUTION}"
echo "Layer selection: ${LAYER_SELECTION}"
echo "Artifact stem: ${ARTIFACT_STEM}"
echo "Default output base save dir: ${DEFAULT_OUTPUT_BASE_SAVE_DIR}"
echo "Spectra dir: ${SPECTRA_DIR}"
echo "Plot dir: ${PLOT_DIR}"
echo "Resources: gpus=${SLURM_GPUS:-1} cpus=${SLURM_CPUS_PER_TASK:-4} mem=48G time=2:00:00"

cd "${ROOT_DIR}"
CMD=(
  uv run python scripts/plot_resnet_block_spectra.py
  --base-save-dir "${INPUT_BASE_SAVE_DIR}"
  --run-id "${RUN_ID}"
  --run-id-resolution "${RUN_ID_RESOLUTION}"
  --layer-selection "${LAYER_SELECTION}"
  --artifact-stem "${ARTIFACT_STEM}"
  --spectra-dir "${SPECTRA_DIR}"
  --output-dir "${PLOT_DIR}"
)

WIDTHS_RAW="${RESNET_BLOCK_SPECTRA_WIDTHS:-}"
if [[ -n "${WIDTHS_RAW}" ]]; then
  read -r -a WIDTHS <<< "${WIDTHS_RAW}"
  CMD+=(--widths "${WIDTHS[@]}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
