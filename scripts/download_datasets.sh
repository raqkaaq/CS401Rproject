#!/bin/bash
# Script to pre-download datasets before submitting jobs
# Run this on the LOGIN NODE (which has internet access)

# Get the directory where this script is located and change to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT" || {
    echo "Error: Could not change to project root: $PROJECT_ROOT"
    exit 1
}

echo "=== Pre-downloading Datasets ==="
echo "This script downloads datasets to ~/.cache/huggingface/datasets/"
echo "Run this on the LOGIN NODE before submitting jobs"
echo ""

# Load Python module
module load python/3.12
source ~/venv/bin/activate

# Set cache directory
export HF_DATASETS_CACHE="$HOME/.cache/huggingface/datasets"
mkdir -p "$HF_DATASETS_CACHE"

echo "Cache directory: $HF_DATASETS_CACHE"
echo ""

# Function to download a dataset
download_dataset() {
    local dataset_name=$1
    local split=${2:-"train"}
    
    echo "Downloading $dataset_name (split: $split)..."
    
    python -c "
from datasets import load_dataset
import os
import sys

dataset_name = sys.argv[1]
split = sys.argv[2]
cache_dir = os.environ.get('HF_DATASETS_CACHE', os.path.expanduser('~/.cache/huggingface/datasets'))

print(f'  Loading {dataset_name}...')
try:
    # GSM8K requires a config name ('main' or 'socratic')
    if dataset_name == 'openai/gsm8k':
        dataset = load_dataset(dataset_name, 'main', split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    print(f'  ✓ Downloaded {dataset_name}')
    print(f'  Samples: {len(dataset)}')
    print(f'  Cached in: {cache_dir}')
except Exception as e:
    print(f'  ✗ Failed to download {dataset_name}: {e}')
    raise
" "$dataset_name" "$split"
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Successfully downloaded $dataset_name"
    else
        echo "  ✗ Failed to download $dataset_name"
        return 1
    fi
}

# Download datasets used by parsers
echo "Downloading datasets..."
echo ""

download_dataset "trl-lib/DeepMath-103K" "train"
download_dataset "checkai/instruction-poems" "train"
download_dataset "sh0416/ag_news" "train"
download_dataset "openai/gsm8k" "train"

echo ""
echo "=== Download Complete ==="
echo ""
echo "Datasets are now cached in: $HF_DATASETS_CACHE"
echo ""
echo "You can now submit your job - datasets will be loaded from cache"

