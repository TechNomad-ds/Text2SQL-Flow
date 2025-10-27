# LLM Training for Text2SQL-Flow

This directory contains the training scripts and configuration for fine-tuning LLMs on Text-to-SQL tasks.

## Overview

This module provides training utilities for fine-tuning LLMs (specifically Qwen2.5-Coder-7B-Instruct) on Text-to-SQL datasets using [LLaMAFactory](https://github.com/hiyouga/LLaMA-Factory).

## Files

- `train_7b.sh`: Training script for 7B model
- `train_7b.yaml`: Training configuration file

## Prerequisites

Before running the training, ensure you have:

1. **LLaMAFactory installed**:
   ```bash
   pip install llamafactory
   ```

2. **Set up environment variables** in `train_7b.sh`:
   - `HF_HOME`: HuggingFace cache directory
   - `HF_DATASETS_CACHE`: Datasets cache directory
   - `TRITON_CACHE_DIR`: Triton cache directory
   - `TMPDIR`: Temporary files directory

3. **Configure `train_7b.yaml`**:
   - Set `model_name_or_path` to your model path (e.g., Qwen2.5-Coder-7B-Instruct)
   - Set `dataset_dir` to your dataset directory
   - Configure `deepspeed` path if using DeepSpeed

## Usage

### Basic Training

Run the training script:

```bash
bash train_7b.sh
```

### Configuration

Edit `train_7b.yaml` to customize training parameters:

- **Model**: Configure model path and settings
- **Dataset**: Specify dataset directory and preprocessing options
- **Training**: Adjust batch size, learning rate, epochs, etc.
- **Output**: Configure logging and checkpointing
- **Evaluation**: Set validation and evaluation strategy

Key parameters:
- `stage: sft`: Supervised fine-tuning
- `finetuning_type: full`: Full parameter fine-tuning
- `cutoff_len: 4096`: Maximum sequence length
- `learning_rate: 2.0e-5`: Learning rate
- `num_train_epochs: 2`: Number of training epochs

### Advanced Options

For multi-GPU training with DeepSpeed, configure the `deepspeed` parameter with your DeepSpeed config file path.

## Output

- **Checkpoints**: Saved to `./ckpts/train-7b-synsql-1`
- **Logs**: TensorBoard logs in `./log_output/train-7b`
- **Evaluation**: Validation results reported during training

## Notes

- The training uses Flash Attention 2 (`flash_attn: fa2`) for efficient training
- Gradient checkpointing and mixed precision training (bf16) are enabled
- The configuration includes a validation split (10%) for evaluation during training

## Related Documentation

For more details about Text2SQL-Flow, see the main [README](../Text2SQL/README.md).
