import json
import random
from tqdm import tqdm
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--mask", action="store_true", help="Whether to mask schema elements in SQL")
    args = parser.parse_args()

    # Define four input file paths
    input_file = args.input_file

    # Define the final output file path
    output_file = args.output_file

    # Number of negative SQL samples to randomly draw for each item
    NUM_RANDOM_SAMPLES = 30


    def load_data(file_path):
        """Load a JSON file from the given path."""
        print(f"Loading file: {file_path} ...")
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)


    total_items_processed = 0

    if args.mask:
        QUESTION_KEY = "masked_question"
        SQL_KEY = "masked_sql"
    else:
        QUESTION_KEY = "question"
        SQL_KEY = "query"

    # Open the output file in 'w' mode to overwrite previous results
    with open(output_file, 'w', encoding='utf-8') as f_out:
        current_datas = load_data(input_file)
        
        # Build an independent SQL pool for the current file
        # Ensures negative sampling only happens within the same file
        current_sql_pool = [item[SQL_KEY] for item in current_datas]
        
        print(f"Processing {len(current_datas)} items in {os.path.basename(input_file)} ...")
        
        # Iterate over each data item
        for item in tqdm(current_datas, desc=f"Processing {os.path.basename(input_file)}"):
            exclusion_set = {item[SQL_KEY]}
            random_pool = [sql for sql in current_sql_pool if sql not in exclusion_set]

            # Sample safely from the pool
            if len(random_pool) < NUM_RANDOM_SAMPLES:
                rejected_response = random_pool
            else:
                rejected_response = random.sample(random_pool, NUM_RANDOM_SAMPLES)

            filtered_item = {
                "query": item[QUESTION_KEY],
                "response": item[SQL_KEY],
                "rejected_response": rejected_response
            }

            json_line = json.dumps(filtered_item, ensure_ascii=False)
            f_out.write(json_line + '\n')
        
        total_items_processed += len(current_datas)

    print(f"\nProcessing complete! {total_items_processed} items written to {output_file}")


    # --- Validation ---
    print("\n--- Verifying output file ---")
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Total lines in file: {len(lines)}")
    if len(lines) > 3:
        # Randomly sample 3 lines for verification
        sampled_lines = random.sample(lines, 3)

        for i, line in enumerate(sampled_lines):
            item = json.loads(line)
            print(f"\n--- Sample {i+1} ---")
            print("Query:", item["query"])
            print("Response:", item["response"])
            print(f"Rejected Responses (total {len(item['rejected_response'])}, showing first 5):")
            for j, r in enumerate(item["rejected_response"][:5]):
                print(f"  {j+1}. {r}")
    else:
        print("Less than 3 lines in file, skipping sample verification.")
