#!/bin/bash

# Setup script for Position-Aware Attention experiments
# This script sets up the environment and installs dependencies

set -e

echo "Setting up environment for Position-Aware Attention experiments..."

# Create necessary directories
echo "Creating directories..."
mkdir -p data
mkdir -p outputs
mkdir -p results
mkdir -p logs

# Set Python hash seed for reproducibility
export PYTHONHASHSEED=42

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data (if needed)
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')" || echo "NLTK data download skipped"

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Download datasets: bash scripts/download_data.sh"
echo "2. Preprocess data: bash scripts/preprocess_all.sh"
echo "3. Run training: See README.md for commands"

