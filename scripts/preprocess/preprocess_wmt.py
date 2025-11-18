#!/usr/bin/env python3
"""
WMT'14 En-De Data Preprocessing Script
Preprocess WMT'14 English-German dataset for machine translation tasks
"""

import argparse
import torch
from pathlib import Path

def preprocess_wmt(dataset_path: str, output_path: str, tokenizer_name: str = "bert-base-uncased", max_length: int = 512):
    """
    Preprocess WMT'14 En-De dataset.
    
    Args:
        dataset_path: Path to downloaded WMT'14 En-De dataset
        output_path: Path to save preprocessed data
        tokenizer_name: Tokenizer to use (default: bert-base-uncased)
        max_length: Maximum sequence length
    """
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    
    print(f"Loading tokenizer: {tokenizer_name}...")
    # For machine translation, we typically need separate tokenizers for source and target
    # This is a simplified version - actual implementation may need sentencepiece or other tokenizers
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def tokenize_function(examples):
        """Tokenize translation pairs."""
        # Handle both source and target languages
        if 'translation' in examples:
            # Format: {'translation': [{'en': '...', 'de': '...'}, ...]}
            sources = [t.get('en', '') for t in examples['translation']]
            targets = [t.get('de', '') for t in examples['translation']]
        elif 'en' in examples and 'de' in examples:
            sources = examples['en']
            targets = examples['de']
        else:
            raise ValueError("Unknown WMT dataset format")
        
        # Tokenize source and target separately
        source_tokenized = tokenizer(
            sources,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        target_tokenized = tokenizer(
            targets,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': source_tokenized['input_ids'],
            'attention_mask': source_tokenized['attention_mask'],
            'labels': target_tokenized['input_ids']
        }
    
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
    parser = argparse.ArgumentParser(description='Preprocess WMT\'14 En-De for machine translation')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to WMT\'14 En-De dataset')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save preprocessed data')
    parser.add_argument('--tokenizer_name', type=str, default='bert-base-uncased', help='Tokenizer name')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    
    args = parser.parse_args()
    preprocess_wmt(args.dataset_path, args.output_path, args.tokenizer_name, args.max_length)

