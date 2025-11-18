#!/usr/bin/env python3
"""Long Document Evaluation Script"""

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Position-Aware Attention on Long Documents')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    
    args = parser.parse_args()
    print("Long Document evaluation template")

