#!/usr/bin/env bash

# Run the two selected 100-epoch soft-row-sharing experiments sequentially.
# Usage: bash improvements/clustering/configs/layer_reduction/best_100ep/run_best_100ep.sh

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
clustering_dir="$(cd "${script_dir}/../../.." && pwd)"
repository_dir="$(cd "${clustering_dir}/../.." && pwd)"
venv_activate="${repository_dir}/.venv/bin/activate"
log_dir="${LOG_DIR:-${clustering_dir}/logs/epoch100/layer_reduction/best_100ep}"

if [[ ! -f "${venv_activate}" ]]; then
  echo "Virtual environment activation script not found: ${venv_activate}" >&2
  exit 1
fi

source "${venv_activate}"
python_bin="${PYTHON_BIN:-python}"
mkdir -p "${log_dir}"

configs=(
  "configs/layer_reduction/best_100ep/soft_even_alternate_k4_seed42_100ep.yml"
  "configs/layer_reduction/best_100ep/soft_odd_alternate_k8_seed42_100ep.yml"
)
run_names=(
  "ncmtl_soft_even_alternate_k4_seed42_100ep"
  "ncmtl_soft_odd_alternate_k8_seed42_100ep"
)

cd "${clustering_dir}"
echo "Python executable: $(command -v "${python_bin}")"
echo "GPU selection comes from each YAML file (cuda:1)."
echo "Running the two 100-epoch experiments sequentially."

for index in "${!configs[@]}"; do
  config="${configs[$index]}"
  run_name="${run_names[$index]}"
  run_log="${log_dir}/${run_name}.log"

  echo "[$((index + 1))/${#configs[@]}] Starting ${run_name}"
  echo "Config: ${config}"
  echo "Console log: ${run_log}"

  if "${python_bin}" -u run.py \
      --task_type ks_si_er \
      --config "${config}" \
      > "${run_log}" 2>&1; then
    echo "Completed ${run_name}"
  else
    status=$?
    echo "${run_name} failed with exit status ${status}; stopping sequence." >&2
    echo "Inspect ${run_log} for details." >&2
    exit "${status}"
  fi
done

echo "Both 100-epoch experiments completed successfully."
