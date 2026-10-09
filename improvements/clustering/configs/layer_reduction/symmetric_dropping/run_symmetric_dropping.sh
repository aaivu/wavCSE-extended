#!/usr/bin/env bash

# Run four unclipped 30-epoch symmetric layer-dropping experiments sequentially.
# Usage: ./run_symmetric_dropping.sh

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

source "${venv_activate}"
python_bin="${PYTHON_BIN:-python}"

configs=(
  "configs/layer_reduction/symmetric_dropping/soft_symmetric_k2_seed42.yml"
  "configs/layer_reduction/symmetric_dropping/soft_symmetric_k4_seed42.yml"
  "configs/layer_reduction/symmetric_dropping/soft_symmetric_k6_seed42.yml"
  "configs/layer_reduction/symmetric_dropping/soft_symmetric_k8_seed42.yml"
)

run_names=(
  "ncmtl_soft_symmetric_k2_seed42"
  "ncmtl_soft_symmetric_k4_seed42"
  "ncmtl_soft_symmetric_k6_seed42"
  "ncmtl_soft_symmetric_k8_seed42"
)

cd "${clustering_dir}"

echo "Activated virtual environment: ${VIRTUAL_ENV}"
echo "Python executable: $(command -v "${python_bin}")"
echo "Starting sequential symmetric layer-dropping experiments."
echo "Each run uses the YAML device configuration and gradient clipping is disabled."

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

echo "All symmetric layer-dropping experiments completed successfully."

