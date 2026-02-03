#!/bin/bash
set -e

tmp_cache_dir=""
export HF_HOME="${tmp_cache_dir}/hf_cache"
export HF_DATASETS_CACHE="${tmp_cache_dir}/dataset_cache"
export TRITON_CACHE_DIR="${tmp_cache_dir}/triton_cache"
export TMPDIR="${tmp_cache_dir}/tmp"
export NLTK_DATA=""

mkdir -p $HF_HOME $HF_DATASETS_CACHE $TRITON_CACHE_DIR $TMPDIR


home_dir=""
model_path="" # path to your model, here we use Qwen2.5-Coder-7B-Instruct
dataset_dir="data"
num_train_epochs=3 # number of training epochs

visible_devices="0,1,2,3,4,5,6,7"
tensor_parallel_size=4

# ========== Model Configuration ==========
trust_remote_code=true

# ========== Method Configuration ==========
stage="sft"
do_train=true
report_to="tensorboard"
finetuning_type="full"
flash_attn="fa2"

# ========== Dataset Configuration ==========
template="llama3"
cutoff_len=4096
overwrite_cache=true
preprocessing_num_workers=8
dataloader_num_workers=8

# ========== Output Configuration ==========
logging_steps=50
save_steps=1000
plot_loss=true
overwrite_output_dir=true
save_only_model=false

# ========== Training Configuration ==========
deepspeed_config="" # path to your deepspeed config file, here we use the ds_z2_config.json provided by LLaMA-Factory
per_device_train_batch_size=1
gradient_accumulation_steps=8
learning_rate="2.0e-5"
max_grad_norm=0.5
lr_scheduler_type="cosine"
warmup_ratio=0.15
bf16=true
ddp_timeout=180000000

# ========== Evaluation Configuration ==========
val_size="0.1"
per_device_eval_batch_size=4
eval_strategy="steps"
eval_steps=1000



# list of datasets to train
dataset_name_list=(
  "dataset_1"
  "dataset_2"
  "dataset_3"
  )

for dataset_name in ${dataset_name_list[@]}; do
  echo "=========================================="
  echo "Processing dataset: $dataset_name"
  echo "=========================================="
  
  output_dir="${home_dir}/${dataset_name}/ckpt"
  logging_dir="${home_dir}/${dataset_name}/log"
  benchmark_result_dir="${home_dir}/${dataset_name}/result"
  output_model_dir="${output_dir}"

  # Create directories
  mkdir -p "${output_dir}" "${logging_dir}" "${benchmark_result_dir}"

  # Training
  echo "Starting training for $dataset_name..."
  llamafactory-cli train \
    --model_name_or_path "$model_path" \
    --trust_remote_code "$trust_remote_code" \
    --stage "$stage" \
    --do_train "$do_train" \
    --report_to "$report_to" \
    --finetuning_type "$finetuning_type" \
    --flash_attn "$flash_attn" \
    --dataset_dir "$dataset_dir" \
    --dataset "$dataset_name" \
    --template "$template" \
    --cutoff_len "$cutoff_len" \
    --overwrite_cache "$overwrite_cache" \
    --preprocessing_num_workers "$preprocessing_num_workers" \
    --dataloader_num_workers "$dataloader_num_workers" \
    --output_dir "$output_dir" \
    --logging_dir "$logging_dir" \
    --logging_steps "$logging_steps" \
    --save_steps "$save_steps" \
    --plot_loss "$plot_loss" \
    --overwrite_output_dir "$overwrite_output_dir" \
    --save_only_model "$save_only_model" \
    --deepspeed "$deepspeed_config" \
    --per_device_train_batch_size "$per_device_train_batch_size" \
    --gradient_accumulation_steps "$gradient_accumulation_steps" \
    --learning_rate "$learning_rate" \
    --num_train_epochs "$num_train_epochs" \
    --max_grad_norm "$max_grad_norm" \
    --lr_scheduler_type "$lr_scheduler_type" \
    --warmup_ratio "$warmup_ratio" \
    --bf16 "$bf16" \
    --ddp_timeout "$ddp_timeout" \
    --val_size "$val_size" \
    --per_device_eval_batch_size "$per_device_eval_batch_size" \
    --eval_strategy "$eval_strategy" \
    --eval_steps "$eval_steps"

  # check if training was successful
  if [ ! -d "$output_model_dir" ] || [ -z "$(ls -A $output_model_dir 2>/dev/null)" ]; then
    echo "Warning: Training may have failed for $dataset_name. Checkpoint directory is empty or missing."
    echo "Skipping evaluation for $dataset_name."
    continue
  fi

  echo "Completed training for $dataset_name"
  echo ""
done

echo "=========================================="
echo "All datasets trained!"
echo "=========================================="
