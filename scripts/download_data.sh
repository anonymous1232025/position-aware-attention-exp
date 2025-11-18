#!/bin/bash

# Data download script for Position-Aware Attention experiments
# This script downloads all datasets used in the experiments

set -e

echo "Downloading datasets for Position-Aware Attention experiments..."

# Create data directory
mkdir -p data

# Set Python hash seed for reproducibility
export PYTHONHASHSEED=42

# Download WikiText-103
echo "Downloading WikiText-103..."
python -c "from datasets import load_dataset; load_dataset('wikitext', 'wikitext-103-raw-v1').save_to_disk('data/wikitext103')" || echo "WikiText-103 download failed"

# Download WMT'14 En-De (if available)
echo "Downloading WMT'14 En-De..."
echo "Note: WMT'14 En-De requires manual download. Please refer to:"
echo "https://github.com/pytorch/fairseq/tree/main/examples/translation#wmt14-english-to-german-non-segmented"

# Download SQuAD 2.0
echo "Downloading SQuAD 2.0..."
python -c "from datasets import load_dataset; load_dataset('squad_v2').save_to_disk('data/squad2')" || echo "SQuAD 2.0 download failed"

# Download GLUE
echo "Downloading GLUE datasets..."
python -c "from datasets import load_dataset; 
for task in ['cola', 'sst2', 'mrpc', 'qqp', 'mnli', 'qnli', 'rte', 'wnli']:
    try:
        load_dataset('glue', task).save_to_disk(f'data/glue_{task}')
        print(f'Downloaded GLUE {task}')
    except Exception as e:
        print(f'Failed to download GLUE {task}: {e}')" || echo "GLUE download failed"

# Download ArXiv (long documents)
echo "Downloading ArXiv dataset..."
python -c "from datasets import load_dataset; load_dataset('scientific_papers', 'arxiv').save_to_disk('data/arxiv')" || echo "ArXiv download failed"

echo "Data download complete!"
echo "Note: Some datasets may require manual download. Please check individual dataset documentation."

