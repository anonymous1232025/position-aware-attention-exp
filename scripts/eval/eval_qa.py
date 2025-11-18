#!/usr/bin/env python3
"""Question Answering Evaluation Script"""

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Position-Aware Attention on Question Answering')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    
    args = parser.parse_args()
    print("Question Answering evaluation template")

