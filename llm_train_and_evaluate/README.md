This directory provides **training scripts**, **benchmark evaluation pipelines**, and **dataset containment filtering tools** for fine-tuning and evaluating large language models (LLMs) on **Text-to-SQL** tasks.

It is designed to support **multi-dataset sequential training**, **standardized benchmark evaluation**, and **leakage-aware data filtering** in a reproducible workflow.

---

## Overview

| File                         | Description                                                                                                                                                |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `train_llm.sh`               | Batch fine-tuning script for sequentially training on multiple datasets using [LLaMAFactory](https://github.com/hiyouga/LLaMA-Factory).                    |
| `eval_bench.sh`              | Entry script for running benchmark evaluations over a list of trained model checkpoints.                                                                   |
| `eval_open_source_models.py` | Orchestrates evaluation of one or more models on Spider, BIRD, Spider-DK, Spider-Realistic, Spider-Syn, and EHRSQL.                                        |
| `auto_evaluation.py`         | Single-benchmark evaluation runner supporting greedy decoding and optional sampling with pass@k / majority voting, followed by benchmark-specific scoring. |
| `filter_containment.py`      | Dataset containment filter that detects and reports potential train–test leakage using exact SQL matching and n-gram Jaccard similarity.                   |

---

## Prerequisites

### LLaMAFactory (for training and inference)

```bash
pip install llamafactory
```

Ensure that your CUDA, DeepSpeed, and PyTorch environments are properly configured according to LLaMAFactory’s documentation.

---

## 1. Training (`train_llm.sh`)

This script fine-tunes a base LLM on **multiple Text-to-SQL datasets sequentially** using LLaMAFactory.

### Configuration (edit before running)

| Variable               | Description                                                     | Example                                 |
| ---------------------- | --------------------------------------------------------------- | --------------------------------------- |
| `home_dir`             | Root directory for all outputs (checkpoints, logs, results).    | `"/path/to/outputs"`                    |
| `model_path`           | Path to the base model (e.g., Qwen2.5-Coder-7B-Instruct).       | `"/path/to/Qwen2.5-Coder-7B-Instruct"`  |
| `dataset_dir`          | Directory containing LLaMAFactory dataset configs and data.     | `"data"`                                |
| `dataset_name_list`    | Ordered list of dataset names (must exist under `dataset_dir`). | `("dataset_1" "dataset_2" "dataset_3")` |
| `num_train_epochs`     | Number of training epochs per dataset.                          | `3`                                     |
| `visible_devices`      | GPU device IDs.                                                 | `"0,1,2,3,4,5,6,7"`                     |
| `tensor_parallel_size` | Tensor parallelism degree (typically number of GPUs used).      | `4`                                     |
| `deepspeed_config`     | Path to DeepSpeed config file (e.g., `ds_z2_config.json`).      | `"/path/to/ds_z2_config.json"`          |

Additional LLaMAFactory options (e.g., training stage, prompt template, cutoff length, batch size, learning rate) are defined inside the script and can be adjusted as needed.

---

### Output Layout

For each dataset in `dataset_name_list`, the script creates:

* `{home_dir}/{dataset_name}/ckpt` — model checkpoints
* `{home_dir}/{dataset_name}/log` — TensorBoard logs
* `{home_dir}/{dataset_name}/result` — placeholder directory for evaluation outputs

---

### Usage

```bash
# 1. Configure paths and variables in train_llm.sh
# 2. Run training
bash train_llm.sh
```

Datasets are processed **one after another**.
If training fails for a dataset, its checkpoint directory may be empty; downstream evaluation will be skipped automatically for that dataset.

---

## 2. Evaluation

`eval_bench.sh` is a wrapper script that calls `eval_open_source_models.py` with a predefined list of model checkpoints.Configure the following variables in the script:

* `MODELS` — array of model checkpoint paths (e.g., `"model_1/ckpt" "model_2/ckpt"`).
* `TMPDIR`, `NLTK_DATA` — environment variables if required by your system.

#### Usage

```bash
# Edit MODELS and paths in eval_bench.sh, then run:
bash eval_bench.sh
```

## References

For the complete pipeline (data augmentation → training → evaluation), please refer to:

* Main pipeline documentation: [Text2SQL-Flow README](../README.md)
* Data augmentation module: [data_augmentation README](../data_augmentation/README.md)
