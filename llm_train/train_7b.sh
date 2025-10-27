#!/bin/bash
set -e

export HF_HOME=""

export HF_DATASETS_CACHE=""

export TRITON_CACHE_DIR=""

export TMPDIR=""

mkdir -p $HF_HOME $HF_DATASETS_CACHE $TRITON_CACHE_DIR $TMPDIR

llamafactory-cli train train_7b.yaml