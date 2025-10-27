set -e

input_data_file=""
output_data_file=""
db_path=""
tables_file=""
db_content_index_path=""

python process_dataset.py \
    --input_data_file $input_data_file \
    --output_data_file $output_data_file \
    --db_path $db_path \
    --tables $tables_file \
    --source synthetic \
    --mode train \
    --value_limit_num 2 \
    --db_content_index_path $db_content_index_path