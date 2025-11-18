#!/usr/bin/env python3
"""
Parameter Sensitivity Analysis Script
Analyze parameter sensitivity for alpha, beta, gamma
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze parameter sensitivity')
    parser.add_argument('--alpha_range', type=float, nargs=3, default=[0.5, 2.0, 0.1], 
                       help='Alpha range: start, end, step')
    parser.add_argument('--beta_range', type=float, nargs=3, default=[0.5, 3.0, 0.1],
                       help='Beta range: start, end, step')
    parser.add_argument('--gamma_range', type=float, nargs=3, default=[0.5, 3.0, 0.1],
                       help='Gamma range: start, end, step')
    parser.add_argument('--output_dir', type=str, default='parameter_analysis', help='Output directory')
    
    args = parser.parse_args()
    print("Parameter sensitivity analysis template")
    print(f"Alpha range: {args.alpha_range}")
    print(f"Beta range: {args.beta_range}")
    print(f"Gamma range: {args.gamma_range}")
    print(f"Output: {args.output_dir}")

