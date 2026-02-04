# Retrieval Strategy

## Overview

This module retrieves **few-shot in-context examples** using the *Question–SQL Similarity Selection* strategy and then builds prompts for downstream Text-to-SQL generation.


## Environmental Setup

This pipeline depends on environment of DAIL-SQLs.
Download `stanford-corenlp` and unzip it into `./third_party`, then start the CoreNLP server:

```bash
apt install default-jre
apt install default-jdk

cd third_party/stanford-corenlp-full-2018-10-05
nohup java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer &
cd ../../
```

Set up the Python environment:

```bash
conda create -n DAIL-SQL python=3.8
conda activate DAIL-SQL

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install sqlglot zss
python nltk_downloader.py
```

## Replace required components

Before running retrieval, replace the following components in your DAIL-SQL workspace:

* `prompt/` 
* `utils/` 
* `generate_question.py`

> If you run the *retrieval baseline*, you may also need to update the retrieval model path in:
> `prompt/ExampleSelectorTemplate.py`


## Build few-shot prompts with Question–SQL Similarity Selection

Run `generate_question.py` to construct few-shot prompts for a given dataset split. 

Example command:

```bash
python generate_question.py \
  --data_type bird \
  --split test \
  --tokenizer gpt-4o \
  --max_seq_len 4096 \
  --prompt_repr SQL \
  --k_shot 5 \
  --example_type QA \
  --selector_type EMBED \
  --select_model_path /path/sft-models/Qwen3-Embedding-0.6B-sft \
  --if_mask True
```
