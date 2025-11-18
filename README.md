# Text2SQL-Flow: A SQL-Aware Data Augmentation Framework

## Overview

**Text2SQL-Flow** is a comprehensive SQL-aware data augmentation framework that systematically generates large-scale, semantically valid, and structurally diverse Text-to-SQL pairs from limited seed data. This framework addresses the critical challenges in Text-to-SQL tasks where model performance is constrained by the scarcity, limited diversity, and structural simplicity of existing datasets.

### Key Features

- **Six-Dimensional Augmentation**: Operates along six augmentation dimensions for comprehensive data generation
- **End-to-End Pipeline**: Features SQL execution verification, natural language question generation, chain-of-thought reasoning trace generation, and data classification
- **Modular Database Manager**: Ensures cross-database compatibility and scalability
- **SQLFlow Dataset**: Constructs a high-quality dataset comprising 89,544 annotated examples
- **Dual Application**: Supports both fine-tuning and prompt-based settings with masked alignment retrieval

## Framework Architecture

```
Text2SQL-Flow/
├── data_augmentation/          # Core augmentation framework
│   ├── operators/             # Augmentation operators
│   │   ├── generate/         # Generation operators
│   │   ├── filter/           # Filtering operators
│   │   └── eval/             # Evaluation operators
│   ├── pipelines/            # End-to-end pipelines
│   └── data/                 # Sample data
├── example_retrieval/         # Masked alignment retrieval
│   ├── mask/                 # SQL masking utilities
│   ├── data_process/         # Data processing
│   ├── retrieval_method/     # Retrieval implementations
│   └── model_training/       # Retrieval model training
├── llm_train/                # LLM fine-tuning
│   ├── train_7b.sh          # Training script
│   └── train_7b.yaml        # Training configuration
└── scripts/                  # Utility scripts
    ├── process_dataset.py   # Dataset processing
    └── generate_ddls.py     # DDL generation
```

## Installation

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/OpenDCAI/DataFlow
cd DataFlow/Text2SQL-Flow
```

2. **Install dependencies**:
```bash
pip install -e .
```

3. **Install additional requirements**:
```bash
# For LLM training
pip install llamafactory

# For example retrieval
pip install vllm transformers
```

## Quick Start

### 1. Data Augmentation Pipeline

Run the complete Text2SQL-Flow augmentation pipeline:

```bash
cd data_augmentation/pipelines
python text2sql_pipeline_refine.py
```

This will:
- Generate SQL variations from seed data
- Create natural language questions
- Generate chain-of-thought reasoning traces
- Apply SQL execution verification
- Classify and filter generated data

### 2. LLM Fine-tuning

Fine-tune open-source LLMs on the generated SQLFlow dataset:

```bash
cd llm_train
# Configure your environment variables in train_7b.sh
bash train_7b.sh
```

### 3. Example Retrieval

Train and use the masked alignment retrieval method:

```bash
cd example_retrieval/model_training
bash run_swfit.sh
```

## Dataset and Model

- **Dataset**: The SQLFlow dataset is publicly available on Hugging Face Hub: [debugger123/SQLFlow](https://huggingface.co/datasets/debugger123/SQLFlow).
- **Embedding Model**: The embedding model used for few-shot example retrieval is also hosted on Hugging Face Hub: [xccr/SQLFlow-Retrieval-0.6B](https://huggingface.co/xccr/SQLFlow-Retrieval-0.6B).
