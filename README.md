# Position-Aware Attention Mechanism - Experimental Reproducibility

This repository contains code for reproducing all experiments in the paper "Position-Aware Attention Mechanism: A Mathematical Framework for Enhanced Spatial Information Processing in Transformer Architectures".

## Repository Structure

```
supplement-exp/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── setup.sh                   # Environment setup script
├── src/                       # Main source code
│   └── position_aware_attention/
│       ├── __init__.py
│       ├── models.py          # Model architectures
│       ├── attention.py       # Position-aware attention implementation
│       ├── position_effect.py # Position effect function implementation
│       └── triple_attention.py # Triple-attention architecture
├── scripts/                   # Scripts for experiments
│   ├── download_data.sh       # Download all datasets
│   ├── preprocess_all.sh      # Preprocess all datasets
│   ├── train/                 # Training scripts
│   │   ├── train_lm.py
│   │   ├── train_mt.py
│   │   ├── train_qa.py
│   │   ├── train_glue.py
│   │   └── train_longdoc.py
│   ├── eval/                  # Evaluation scripts
│   │   ├── eval_lm.py
│   │   ├── eval_mt.py
│   │   ├── eval_qa.py
│   │   ├── eval_glue.py
│   │   └── eval_longdoc.py
│   └── analysis/              # Analysis scripts
│       ├── analyze_attention.py
│       └── analyze_parameters.py
└── configs/                   # Configuration files
    ├── wikitext103.yaml
    ├── wmt14_ende.yaml
    ├── squad2.yaml
    ├── glue_*.yaml
    └── arxiv.yaml
```

## Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Or use the setup script
bash setup.sh
```

### 2. Download Data

```bash
bash scripts/download_data.sh
```

### 3. Preprocess Data

```bash
bash scripts/preprocess_all.sh
```

### 4. Run Training

#### Language Modeling (WikiText-103)
```bash
python scripts/train/train_lm.py \
    --config configs/wikitext103.yaml \
    --seed 42 \
    --output_dir outputs/wikitext103_seed42 \
    --wandb_project position-aware-attention \
    --wandb_run_name wikitext103_seed42
```

#### Machine Translation (WMT'14 En-De)
```bash
python scripts/train/train_mt.py \
    --config configs/wmt14_ende.yaml \
    --seed 42 \
    --output_dir outputs/wmt14_ende_seed42 \
    --wandb_project position-aware-attention \
    --wandb_run_name wmt14_ende_seed42
```

#### Question Answering (SQuAD 2.0)
```bash
python scripts/train/train_qa.py \
    --config configs/squad2.yaml \
    --seed 42 \
    --output_dir outputs/squad2_seed42 \
    --wandb_project position-aware-attention \
    --wandb_run_name squad2_seed42
```

#### GLUE Benchmark
```bash
for task in cola sst2 mrpc qqp mnli qnli rte wnli; do
    python scripts/train/train_glue.py \
        --config configs/glue_${task}.yaml \
        --seed 42 \
        --output_dir outputs/glue_${task}_seed42 \
        --wandb_project position-aware-attention \
        --wandb_run_name glue_${task}_seed42
done
```

#### Long Documents (ArXiv)
```bash
python scripts/train/train_longdoc.py \
    --config configs/arxiv.yaml \
    --seed 42 \
    --output_dir outputs/arxiv_seed42 \
    --wandb_project position-aware-attention \
    --wandb_run_name arxiv_seed42
```

### 5. Run Evaluation

```bash
# Example: Evaluate language modeling
python scripts/eval/eval_lm.py \
    --checkpoint outputs/wikitext103_seed42/checkpoint-best.pt \
    --test_data data/wikitext103/test.pt \
    --output_file results/wikitext103_seed42_test.json
```

## Reproducibility Guarantees

### Random Seed Settings

We use fixed random seeds across 7 levels of randomness:
- Python random: seed 42
- NumPy random: seed 42
- PyTorch random: seed 42
- PyTorch CUDA: seed 42
- cuDNN: deterministic mode enabled
- DataLoader: worker_init_fn with seed 42
- Environment variable: PYTHONHASHSEED=42

### Expected Results

- All results should match within 0.1% for deterministic operations
- Statistical variations (mean ± std) should match within 5% across runs
- Coefficient of variation < 3% for all reported metrics

## Hardware Requirements

- **GPUs:** 4 × NVIDIA A100 40GB (or equivalent)
- **Memory:** ~35GB per GPU for training
- **Storage:** ~500GB for all datasets and checkpoints
- **Time:** See individual task documentation for training times

