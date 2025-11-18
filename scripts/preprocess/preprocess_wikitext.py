#!/usr/bin/env python3
"""
WikiText-103 Data Preprocessing Script
Preprocess WikiText-103 dataset for language modeling tasks
"""

import argparse
import torch
from pathlib import Path
import sys

def preprocess_wikitext(dataset_path: str, output_path: str, tokenizer_name: str = "gpt2", max_length: int = 512):
    """
    Preprocess WikiText-103 dataset.
    
    Args:
        dataset_path: Path to downloaded WikiText-103 dataset
        output_path: Path to save preprocessed data
        tokenizer_name: Tokenizer to use (default: gpt2)
        max_length: Maximum sequence length
    """
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    
    print(f"Loading tokenizer: {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        """Tokenize text examples."""
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    print(f"Saving preprocessed data to {output_path}...")
    torch.save(tokenized_dataset, output_path)
    print("Preprocessing complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess WikiText-103 for language modeling')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to WikiText-103 dataset')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save preprocessed data')
    parser.add_argument('--tokenizer_name', type=str, default='gpt2', help='Tokenizer name')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    
    args = parser.parse_args()
    preprocess_wikitext(args.dataset_path, args.output_path, args.tokenizer_name, args.max_length)

