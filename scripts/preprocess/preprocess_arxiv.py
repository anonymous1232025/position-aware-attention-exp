#!/usr/bin/env python3
"""
ArXiv Papers Data Preprocessing Script
Preprocess ArXiv dataset for long document tasks
"""

import argparse
import torch
from pathlib import Path

def preprocess_arxiv(dataset_path: str, output_path: str, tokenizer_name: str = "gpt2", max_length: int = 2048):
    """
    Preprocess ArXiv dataset for long document processing.
    
    Args:
        dataset_path: Path to downloaded ArXiv dataset
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
        """Tokenize long documents."""
        # Combine article and abstract if available
        texts = []
        for i in range(len(examples.get('article', []))):
            article = examples.get('article', [''])[i]
            abstract = examples.get('abstract', [''])[i]
            # Combine article and abstract
            combined_text = f"{abstract}\n\n{article}" if abstract else article
            texts.append(combined_text)
        
        return tokenizer(
            texts,
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
    parser = argparse.ArgumentParser(description='Preprocess ArXiv dataset for long documents')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to ArXiv dataset')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save preprocessed data')
    parser.add_argument('--tokenizer_name', type=str, default='gpt2', help='Tokenizer name')
    parser.add_argument('--max_length', type=int, default=2048, help='Maximum sequence length')
    
    args = parser.parse_args()
    preprocess_arxiv(args.dataset_path, args.output_path, args.tokenizer_name, args.max_length)

