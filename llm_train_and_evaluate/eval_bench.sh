set -e

export TMPDIR=""
export NLTK_DATA=""

# list of models to evaluate
MODELS=(
    "model_1/ckpt"
    "model_2/ckpt"
    "model_3/ckpt"
    )

# join array into a comma-separated list for --models
MODELS_CSV=$(IFS=,; echo "${MODELS[*]}")

python eval_open_source_models.py \
    --models "$MODELS_CSV" \
    --output_dir eval_results