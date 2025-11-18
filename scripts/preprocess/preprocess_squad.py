#!/usr/bin/env python3
"""
SQuAD 2.0 Data Preprocessing Script
Preprocess SQuAD 2.0 dataset for question answering tasks
"""

import argparse
import torch
from pathlib import Path

def preprocess_squad(dataset_path: str, output_path: str, tokenizer_name: str = "bert-base-uncased", max_length: int = 512):
    """
    Preprocess SQuAD 2.0 dataset.
    
    Args:
        dataset_path: Path to downloaded SQuAD 2.0 dataset
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
    
    def tokenize_function(examples):
        """Tokenize question-answer pairs."""
        questions = examples['question']
        contexts = examples['context']
        
        return tokenizer(
            questions,
            contexts,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )
    
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
    parser = argparse.ArgumentParser(description='Preprocess SQuAD 2.0 for question answering')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to SQuAD 2.0 dataset')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save preprocessed data')
    parser.add_argument('--tokenizer_name', type=str, default='bert-base-uncased', help='Tokenizer name')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    
    args = parser.parse_args()
    preprocess_squad(args.dataset_path, args.output_path, args.tokenizer_name, args.max_length)

