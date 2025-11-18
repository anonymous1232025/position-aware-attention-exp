#!/usr/bin/env python3
"""
GLUE Data Preprocessing Script
Preprocess GLUE datasets for classification tasks
"""

import argparse
import torch
from pathlib import Path

def preprocess_glue(dataset_path: str, output_path: str, tokenizer_name: str = "bert-base-uncased", max_length: int = 128):
    """
    Preprocess GLUE dataset.
    
    Args:
        dataset_path: Path to downloaded GLUE dataset
        output_path: Path to save preprocessed data
        tokenizer_name: Tokenizer to use (default: bert-base-uncased)
        max_length: Maximum sequence length
    """
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    
    print(f"Loading tokenizer: {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # GLUE tasks have different input formats
    # Handle sentence pairs (MNLI, QQP, etc.) and single sentences (SST-2, etc.)
    def tokenize_function(examples):
        """Tokenize GLUE examples."""
        # Check if dataset has sentence1 and sentence2
        if 'sentence1' in examples and 'sentence2' in examples:
            # Sentence pair task
            return tokenizer(
                examples['sentence1'],
                examples['sentence2'],
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt'
            )
        elif 'sentence' in examples:
            # Single sentence task
            return tokenizer(
                examples['sentence'],
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt'
            )
        else:
            raise ValueError("Unknown GLUE dataset format")
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing"
    )
    
    print(f"Saving preprocessed data to {output_path}...")
    torch.save(tokenized_dataset, output_path)
    print("Preprocessing complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess GLUE dataset')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to GLUE dataset')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save preprocessed data')
    parser.add_argument('--tokenizer_name', type=str, default='bert-base-uncased', help='Tokenizer name')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    
    args = parser.parse_args()
    preprocess_glue(args.dataset_path, args.output_path, args.tokenizer_name, args.max_length)

