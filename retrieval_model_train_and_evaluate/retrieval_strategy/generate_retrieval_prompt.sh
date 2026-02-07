# Run parameters can refer to DAIL-SQL settings
# Examples
python generate_question.py \
  --data_type spider-syn \
  --split test \
  --tokenizer gpt-4o \
  --max_seq_len 4096 \
  --prompt_repr SQL \
  --k_shot 5 \
  --example_type QA \
  --selector_type COSMASKSQLSIMILAR \
  --pre_test_result ./dataset/process/BIRD-TEST_SQL_0-SHOT_CTX-1000_ANS-4096/RESULTS_MODEL-gpt-4o.txt

python generate_question.py \
  --data_type spider-syn \
  --split test \
  --tokenizer gpt-4o \
  --max_seq_len 4096 \
  --prompt_repr SQL \
  --k_shot 5 \
  --example_type QA \
  --selector_type COSSQLSIMILAR \
  --pre_test_result ./dataset/process/BIRD-TEST_SQL_0-SHOT_CTX-1000_ANS-4096/RESULTS_MODEL-gpt-4o.txt


# Question-SQL Similarity Selection
python generate_question.py \
  --data_type bird \
  --split test \
  --tokenizer gpt-4o \
  --max_seq_len 4096 \
  --prompt_repr SQL \
  --k_shot 5 \
  --example_type QA \
  --selector_type EMBED \
  --select_model_path /path/hf-models/Qwen3-Embedding-0.6B \
  --if_mask True

python generate_question.py \
  --data_type bird \
  --split test \
  --tokenizer gpt-4o \
  --max_seq_len 4096 \
  --prompt_repr SQL \
  --k_shot 5 \
  --example_type QA \
  --selector_type EMBED \
  --select_model_path /path/hf-models/Qwen3-Embedding-0.6B


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

python generate_question.py \
  --data_type bird \
  --split test \
  --tokenizer gpt-4o \
  --max_seq_len 4096 \
  --prompt_repr SQL \
  --k_shot 5 \
  --example_type QA \
  --selector_type EMBED \
  --select_model_path /path/sft-models/Qwen3-Embedding-0.6B-sft
