#!/usr/bin/env bash

# Run the four gradient-clipped 30-epoch layer-reduction experiments sequentially.
# Usage: ./run_top_layer_dropping.sh

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
clustering_dir="$(cd "${script_dir}/../../.." && pwd)"
repository_dir="$(cd "${clustering_dir}/../.." && pwd)"
venv_activate="${repository_dir}/.venv/bin/activate"
log_dir="${LOG_DIR:-/tmp}"

mkdir -p "${log_dir}"

if [[ ! -f "${venv_activate}" ]]; then
  echo "Virtual environment activation script not found: ${venv_activate}" >&2
  exit 1
fi

# Activate wavCSE's project environment even when this script is launched by
# nohup from a shell that currently has Conda's base environment enabled.
source "${venv_activate}"
python_bin="${PYTHON_BIN:-python}"

configs=(
  "configs/layer_reduction/top_layer_dropping/soft_top_layers_all_seed42_clip1.yml"
  "configs/layer_reduction/top_layer_dropping/soft_top_layers_0_22_seed42_clip1.yml"
  "configs/layer_reduction/top_layer_dropping/soft_top_layers_0_20_seed42_clip1.yml"
  "configs/layer_reduction/top_layer_dropping/soft_top_layers_0_18_seed42_clip1.yml"
)

run_names=(
  "ncmtl_soft_top_layers_all_seed42_clip1"
  "ncmtl_soft_top_layers_0_22_seed42_clip1"
  "ncmtl_soft_top_layers_0_20_seed42_clip1"
  "ncmtl_soft_top_layers_0_18_seed42_clip1"
)

cd "${clustering_dir}"

echo "Activated virtual environment: ${VIRTUAL_ENV}"
echo "Python executable: $(command -v "${python_bin}")"
echo "Starting sequential gradient-clipped layer-reduction experiments."
echo "Each run will use device.type and device.index from its YAML config."
echo "Each run uses training.gradient_clip_norm=1.0."

for index in "${!configs[@]}"; do
  config="${configs[$index]}"
  run_name="${run_names[$index]}"
  run_log="${log_dir}/${run_name}.log"

  echo "[$((index + 1))/${#configs[@]}] Starting ${run_name}."
  echo "Config: ${config}"
  echo "Log: ${run_log}"

  if "${python_bin}" -u run.py \
      --task_type ks_si_er \
      --config "${config}" \
      > "${run_log}" 2>&1; then
    echo "Completed ${run_name}."
  else
    status=$?
    echo "${run_name} failed with exit status ${status}. Stopping sequence."
    echo "Inspect ${run_log} for details."
    exit "${status}"
  fi
done

echo "All top-layer-dropping experiments completed successfully."
