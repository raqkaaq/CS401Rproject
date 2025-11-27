#!/bin/bash
# Diagnostic script to check if dataset cache exists and is accessible

echo "=== Dataset Cache Diagnostic ==="
echo ""

# Check cache directory
CACHE_DIR="$HOME/.cache/huggingface/datasets"
echo "Cache directory: $CACHE_DIR"
echo ""

if [ -d "$CACHE_DIR" ]; then
    echo "✓ Cache directory exists"
    echo "Contents:"
    ls -lh "$CACHE_DIR" | head -20
    echo ""
    
    # Check for DeepMath dataset
    DATASET_DIRS=$(find "$CACHE_DIR" -type d -name "*DeepMath*" -o -name "*trl-lib*" 2>/dev/null)
    if [ -n "$DATASET_DIRS" ]; then
        echo "✓ Found DeepMath-related directories:"
        echo "$DATASET_DIRS"
        echo ""
        for dir in $DATASET_DIRS; do
            echo "  Directory: $dir"
            echo "  Size: $(du -sh "$dir" 2>/dev/null | cut -f1)"
            echo "  Files: $(find "$dir" -type f | wc -l)"
        done
    else
        echo "⚠ No DeepMath dataset directories found"
    fi
else
    echo "✗ Cache directory does not exist"
fi

echo ""
echo "=== Python Environment Check ==="
module load python/3.12 2>/dev/null || module load python/3.11 2>/dev/null
source ~/venv/bin/activate 2>/dev/null

python -c "
import os
from datasets import load_dataset

cache_dir = os.path.expanduser('~/.cache/huggingface/datasets')
print(f'Python cache directory: {cache_dir}')
print(f'Exists: {os.path.exists(cache_dir)}')

# Check environment variables
import os
hf_cache = os.environ.get('HF_DATASETS_CACHE')
print(f'HF_DATASETS_CACHE env var: {hf_cache}')

# Try to load dataset
print('')
print('Attempting to load dataset...')
try:
    dataset = load_dataset('trl-lib/DeepMath-103K', split='train', download_mode='reuse_cache_if_exists')
    print(f'✓ Successfully loaded dataset: {len(dataset)} samples')
except Exception as e:
    print(f'✗ Failed to load: {e}')
    print('')
    print('This means the dataset is not in cache or cache is not accessible')
"

