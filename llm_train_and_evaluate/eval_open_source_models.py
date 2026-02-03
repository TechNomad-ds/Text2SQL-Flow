import os
import argparse
import shlex
import sys

os.environ["NLTK_DATA"] = "" # path to your nltk data

parser = argparse.ArgumentParser(description="Evaluate multiple models on various benchmarks")
parser.add_argument("--models", type=str, default="",
                    help="Comma-separated list of model names or paths")
parser.add_argument("--output_dir", type=str, default="./eval_output",
                    help="Base directory to save all outputs. Each eval_name will have its own subdirectory (default: ./eval_output)")
parser.add_argument("--visible_devices", type=str, default="0,1,2,3,4,5,6,7",
                    help="Visible GPU devices for vLLM (default: 0,1,2,3,4,5,6,7)")
parser.add_argument("--tensor_parallel_size", type=int, default=4,
                    help="Tensor parallel size / number of GPUs (default: 4)")
parser.add_argument("--data_root_dir", type=str, default="data",
                    help="Root directory for evaluation datasets")

args = parser.parse_args()

models = [m.strip() for m in args.models.split(",") if m.strip()]
python_cmd = shlex.quote(sys.executable)
visible_devices = args.visible_devices
tensor_parallel_size = args.tensor_parallel_size

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

print(f"Models to evaluate: {models}")
print(f"All outputs will be saved to: {args.output_dir}/<eval_name>/")
print(f"Using GPUs: {visible_devices}, Tensor parallel size: {tensor_parallel_size}")

# Convert relative paths to absolute paths
base_output_dir = os.path.abspath(args.output_dir)

script_dir = "./"
original_cwd = os.getcwd()

# change to script directory to ensure relative paths in auto_evaluation.py work correctly
os.chdir(script_dir)

def _missing_paths(paths):
    return [p for p in paths if p and not os.path.exists(p)]

def _run_if_ready(cmd, required_paths, label):
    missing = _missing_paths(required_paths)
    if missing:
        print(f"[WARN] Skip {label}, missing paths:")
        for p in missing:
            print(f"  - {p}")
        return
    os.system(cmd)

data_root_dir = os.path.abspath(args.data_root_dir)

def _model_name_from_path(model_path: str) -> str:
    normalized = model_path.rstrip("/").strip()
    if not normalized:
        return "model"
    base = os.path.basename(normalized)
    if base in {"ckpt", "ckpts", "checkpoint", "checkpoints"}:
        parent = os.path.basename(os.path.dirname(normalized))
        return parent or base
    return base

try:
    for model in models:
        model_name = _model_name_from_path(model)
        output_dir = os.path.join(base_output_dir, model_name)
        os.makedirs(output_dir, exist_ok=True)

        dev_spider_eval_name = "dev_spider"
        spider_ts_db = os.path.join(script_dir, "test_suite_sql_eval", "test_suite_database")
        if not os.path.exists(spider_ts_db):
            print("[WARN] test_suite_sql_eval not found, TS evaluation will be skipped for Spider.")
            spider_ts_db = ""
        dev_spider_input = f"{data_root_dir}/dev_spider.json"
        dev_spider_gold = f"{data_root_dir}/spider/dev_gold.sql"
        dev_spider_db = f"{data_root_dir}/spider/database"
        dev_spider_evaluation_cmd = (
            f"{python_cmd} auto_evaluation.py --output_ckpt_dir {model} --source spider "
            f"--visible_devices {visible_devices} --input_file {dev_spider_input} "
            f"--eval_name {dev_spider_eval_name} --tensor_parallel_size {tensor_parallel_size} --n 8 "
            f"--gold_file {dev_spider_gold} --db_path {dev_spider_db} "
            f"--ts_db_path {spider_ts_db} --output_dir {output_dir}"
        )
        _run_if_ready(dev_spider_evaluation_cmd, [dev_spider_input, dev_spider_gold, dev_spider_db], dev_spider_eval_name)

        test_spider_eval_name = "test_spider"
        test_spider_input = f"{data_root_dir}/test_spider.json"
        test_spider_gold = f"{data_root_dir}/spider/test_gold.sql"
        test_spider_db = f"{data_root_dir}/spider/test_database"
        test_spider_evaluation_cmd = (
            f"{python_cmd} auto_evaluation.py --output_ckpt_dir {model} --source spider "
            f"--visible_devices {visible_devices} --input_file {test_spider_input} "
            f"--eval_name {test_spider_eval_name} --tensor_parallel_size {tensor_parallel_size} --n 8 "
            f"--gold_file {test_spider_gold} --db_path {test_spider_db} "
            f"--ts_db_path {spider_ts_db} --output_dir {output_dir}"
        )
        _run_if_ready(test_spider_evaluation_cmd, [test_spider_input, test_spider_gold, test_spider_db], test_spider_eval_name)

        dev_bird_eval_name = "dev_bird"
        dev_bird_input = f"{data_root_dir}/dev_bird.json"
        dev_bird_gold = f"{data_root_dir}/bird/dev_20240627/dev.json"
        dev_bird_db = f"{data_root_dir}/bird/dev_20240627/dev_databases"
        dev_bird_evaluation_cmd = (
            f"{python_cmd} auto_evaluation.py --output_ckpt_dir {model} --source bird "
            f"--visible_devices {visible_devices} --input_file {dev_bird_input} "
            f"--eval_name {dev_bird_eval_name} --tensor_parallel_size {tensor_parallel_size} --n 8 "
            f"--gold_file {dev_bird_gold} --db_path {dev_bird_db} --output_dir {output_dir}"
        )
        _run_if_ready(dev_bird_evaluation_cmd, [dev_bird_input, dev_bird_gold, dev_bird_db], dev_bird_eval_name)

        spider_dk_eval_name = "dev_spider_dk"
        spider_dk_input = f"{data_root_dir}/dev_spider_dk.json"
        spider_dk_gold = f"{data_root_dir}/Spider-DK/spider_dk_gold.sql"
        spider_dk_db = f"{data_root_dir}/Spider-DK/database"
        spider_dk_evaluation_cmd = (
            f"{python_cmd} auto_evaluation.py --output_ckpt_dir {model} --source spider "
            f"--visible_devices {visible_devices} --input_file {spider_dk_input} "
            f"--eval_name {spider_dk_eval_name} --tensor_parallel_size {tensor_parallel_size} --n 8 "
            f"--gold_file {spider_dk_gold} --db_path {spider_dk_db} "
            f"--ts_db_path {spider_ts_db} --output_dir {output_dir}"
        )
        _run_if_ready(spider_dk_evaluation_cmd, [spider_dk_input, spider_dk_gold, spider_dk_db], spider_dk_eval_name)

        spider_realistic_eval_name = "dev_spider_realistic"
        spider_realistic_input = f"{data_root_dir}/dev_spider_realistic.json"
        spider_realistic_gold = f"{data_root_dir}/spider-realistic/spider_realistic_gold.sql"
        spider_realistic_db = f"{data_root_dir}/spider/database"
        spider_realistic_evaluation_cmd = (
            f"{python_cmd} auto_evaluation.py --output_ckpt_dir {model} --source spider "
            f"--visible_devices {visible_devices} --input_file {spider_realistic_input} "
            f"--eval_name {spider_realistic_eval_name} --tensor_parallel_size {tensor_parallel_size} --n 8 "
            f"--gold_file {spider_realistic_gold} --db_path {spider_realistic_db} "
            f"--ts_db_path {spider_ts_db} --output_dir {output_dir}"
        )
        _run_if_ready(
            spider_realistic_evaluation_cmd,
            [spider_realistic_input, spider_realistic_gold, spider_realistic_db],
            spider_realistic_eval_name,
        )

        spider_syn_eval_name = "dev_spider_syn"
        spider_syn_input = f"{data_root_dir}/dev_spider_syn.json"
        spider_syn_gold = f"{data_root_dir}/Spider-Syn/spider_syn_gold.sql"
        spider_syn_db = f"{data_root_dir}/spider/database"
        spider_syn_evaluation_cmd = (
            f"{python_cmd} auto_evaluation.py --output_ckpt_dir {model} --source spider "
            f"--visible_devices {visible_devices} --input_file {spider_syn_input} "
            f"--eval_name {spider_syn_eval_name} --tensor_parallel_size {tensor_parallel_size} --n 8 "
            f"--gold_file {spider_syn_gold} --db_path {spider_syn_db} "
            f"--ts_db_path {spider_ts_db} --output_dir {output_dir}"
        )
        _run_if_ready(spider_syn_evaluation_cmd, [spider_syn_input, spider_syn_gold, spider_syn_db], spider_syn_eval_name)

        dev_ehrsql_eval_name = "dev_ehrsql"
        dev_ehrsql_input = f"{data_root_dir}/dev_ehrsql.json"
        dev_ehrsql_gold = f"{data_root_dir}/EHRSQL/dev.json"
        dev_ehrsql_db = f"{data_root_dir}/EHRSQL/database"
        dev_ehrsql_evaluation_cmd = (
            f"{python_cmd} auto_evaluation.py --output_ckpt_dir {model} --source bird "
            f"--visible_devices {visible_devices} --input_file {dev_ehrsql_input} "
            f"--eval_name {dev_ehrsql_eval_name} --tensor_parallel_size {tensor_parallel_size} --n 8 "
            f"--gold_file {dev_ehrsql_gold} --db_path {dev_ehrsql_db} --output_dir {output_dir}"
        )
        _run_if_ready(dev_ehrsql_evaluation_cmd, [dev_ehrsql_input, dev_ehrsql_gold, dev_ehrsql_db], dev_ehrsql_eval_name)

finally:
    os.chdir(original_cwd)

print(f"\n{'='*60}")
print(f"Evaluation completed!")
print(f"{'='*60}")
print(f"All outputs saved to: {base_output_dir}/<model_name>/<eval_name>/")
