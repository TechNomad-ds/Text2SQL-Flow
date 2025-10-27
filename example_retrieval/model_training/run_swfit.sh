#!/bin/bash

nproc_per_node=8
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model /hf-models/Qwen3-Embedding-0.6B \
    --task_type embedding \
    --model_type qwen3_emb \
    --train_type full \
    --dataset /dataset_path.jsonl \
    --eval_strategy no \
    --output_dir /output_dir \
    --num_train_epochs 6 \
    --save_steps 1339 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 6e-6 \
    --loss_type infonce \
    --label_names labels \
    --dataloader_drop_last true \
    --deepspeed zero3
