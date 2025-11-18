#!/usr/bin/env python3
"""
Long Document Training Script
Train position-aware attention model on ArXiv documents
"""

import argparse
import sys
from pathlib import Path

# Similar structure to train_lm.py but for long documents
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Position-Aware Attention on Long Documents')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--wandb_project', type=str, default='position-aware-attention')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    
    args = parser.parse_args()
    print("Long Document training template")
    print(f"Config: {args.config}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output_dir}")

