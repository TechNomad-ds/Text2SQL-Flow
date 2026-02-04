#!/usr/bin/env bash
# 对data_augmentation合成的数据进行如下处理，生成NL-SQL对齐模型的训练数据


python process_augmented_data.py \
  --input_file augmented_data.json \
  --output_file processed_augmented_data.json

# 使用DAIL-SQL对processed_augmented_data.json及其数据库内容进行数据预处理
python get_masked_sql.py \
  --dataset spider \
  --train_json /path/DAIL-SQL/dataset/dataset_name/processed_augmented_data.json \
  --path_data /path/DAIL-SQL/dataset/ \
  --out augmented_data_masked.json

# 不使用mask策略
python get_training_data.py \
  --input_files augmented_data_masked.json \
  --output_file augmented_training_data.json \
# 使用mask策略
python get_training_data.py \
  --input_files augmented_data_masked.json \
  --output_file augmented_training_data_masked.json \
  --mask