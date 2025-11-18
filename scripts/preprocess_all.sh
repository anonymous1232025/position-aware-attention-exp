#!/bin/bash

# Data preprocessing script for Position-Aware Attention experiments
# This script preprocesses all datasets

set -e

echo "Preprocessing datasets for Position-Aware Attention experiments..."

# Set Python hash seed for reproducibility
export PYTHONHASHSEED=42

# Create preprocessed data directory
mkdir -p data/preprocessed

# Preprocess WikiText-103
echo "Preprocessing WikiText-103..."
python scripts/preprocess/preprocess_lm.py \
    --dataset_path data/wikitext103 \
    --output_path data/preprocessed/wikitext103.pt \
    --tokenizer_name gpt2 \
    --max_length 512 || echo "WikiText-103 preprocessing failed"

# Preprocess SQuAD 2.0
echo "Preprocessing SQuAD 2.0..."
python scripts/preprocess/preprocess_qa.py \
    --dataset_path data/squad2 \
    --output_path data/preprocessed/squad2.pt \
    --tokenizer_name bert-base-uncased \
    --max_length 512 || echo "SQuAD 2.0 preprocessing failed"

# Preprocess GLUE datasets
echo "Preprocessing GLUE datasets..."
for task in cola sst2 mrpc qqp mnli qnli rte wnli; do
    echo "Preprocessing GLUE ${task}..."
    python scripts/preprocess/preprocess_glue.py \
        --dataset_path data/glue_${task} \
        --output_path data/preprocessed/glue_${task}.pt \
        --tokenizer_name bert-base-uncased \
        --max_length 128 || echo "GLUE ${task} preprocessing failed"
done

# Preprocess ArXiv
echo "Preprocessing ArXiv..."
python scripts/preprocess/preprocess_longdoc.py \
    --dataset_path data/arxiv \
    --output_path data/preprocessed/arxiv.pt \
    --tokenizer_name gpt2 \
    --max_length 2048 || echo "ArXiv preprocessing failed"

echo "Data preprocessing complete!"

