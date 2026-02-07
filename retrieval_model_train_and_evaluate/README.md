# Retrieval Model Training & Evaluation

## Overview

This directory includes building training data for a retrieval embedding model, fine-tuning the model with SWIFT SFT, and then using the trained model to retrieve few-shot in-context examples to construct prompts for downstream Text-to-SQL generation.


## Prerequisites

Ensure the DAIL-SQL and SWIFT environments are set up.
```bash
apt install default-jre
apt install default-jdk

cd third_party/stanford-corenlp-full-2018-10-05
nohup java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer &
cd ../../

conda create -n DAIL-SQL python=3.8
conda activate DAIL-SQL

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install sqlglot zss
python nltk_downloader.py

pip isntall ms-swift==3.6.3
```


## Step 1: Build Training Data

Go to `training_data_process/` and follow the README steps to generate training data for the retrieval model.
Keep the final training data path for Step 2.


## Step 2: Train Embedding Model with SWIFT SFT

Training is done with SWIFT SFT. Refer to:

- `model_training/run_swift.sh`


## Step 3: Retrieve Few-shot Examples & Build Prompts

Go to `retrieval_strategy/` and follow its README to build few-shot prompts using the trained embedding model.

### Example command

```bash
python generate_question.py \
  --data_type spider \
  --split test \
  --tokenizer gpt-4o \ --max_seq_len 4096 \
  --prompt_repr SQL \
  --k_shot 5 \
  --example_type QA \
  --selector_type EMBED \
  --select_model_path /path/to/your/swift_output_embedding_model \
  --if_mask True
```

