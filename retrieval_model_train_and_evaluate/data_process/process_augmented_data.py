import json
import re
# import spacy
from tqdm import tqdm

TOKENIZER_PATTERN = re.compile(r"-?\d+\.?\d*|\w+|[^\w\s]")

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def process_and_filter_data(data):
    """
    Recursively process a data structure:
    """
    new_list = []
    for item in tqdm(data, desc="Processing data"):
        if 'query' not in item and 'sql' in item:
            item['query'] = item['sql']
        
        if 'query' in item and isinstance(item['query'], str):
            if 'WITH' in item['query']:
                continue
        
        if 'question' in item and 'question_toks' not in item:
            item['question_toks'] = TOKENIZER_PATTERN.findall(item['question'])
        
        if item is not None:
            new_list.append(item)
    return new_list

def remove_sql_comments(sql_string: str) -> str:
    """
    Remove all single-line and multi-line comments from a SQL string.
    """
    # Remove block comments: /* ... */ (DOTALL makes '.' match newlines)
    cleaned_sql = re.sub(r'/\*.*?\*/', '', sql_string, flags=re.DOTALL)
    # Remove single-line comments: -- 
    cleaned_sql = re.sub(r'--.*', '', cleaned_sql)
    return cleaned_sql

def main():
    data = load_data("train_augmented.json")

    for item in data:
        if 'query' in item:
            item['query'] = remove_sql_comments(item['query'])
            # item['query'] = item['query'].replace('\n', ' ').strip()

    with open("train_augmented.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print(f"{len(data)} data for alignment model training.")

if __name__ == "__main__":
    main()
