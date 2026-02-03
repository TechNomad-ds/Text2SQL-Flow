#!/usr/bin/env bash
set -euo pipefail

export DF_API_KEY="" # your api key

run_name="" # your run name
original_db_root_dir="" 
original_json_path="" # your original json path
num_variations=2
model_name="gpt-4o" # your model name
home_dir="./" # your current home directory
run_dir="${home_dir}/${run_name}"
log_dir="${run_dir}/logs"

mkdir -p "${run_dir}" "${log_dir}"

run_step() {
  local log_file="$1"
  shift
  echo "[cmd] $*"
  PYTHONUNBUFFERED=1 "$@" > "${log_file}" 2>&1
}

echo "Building augmentation data on original DBs..."
run_step "${log_dir}/aug_original.log" \
  python ${home_dir}/pipeline_augmented_data_batch.py \
    --db_root_path "${original_db_root_dir}" \
    --entry_file_name "${original_json_path}" \
    --cache_path "${run_dir}/aug_cache" \
    --num_variations "${num_variations}" \
    --model_name "${model_name}" \
    --data_source "spider" \
    --batch_size 1000

echo "Done! Outputs in: ${run_dir}"