#!/bin/bash
# Script to pre-download models before submitting jobs
# Run this on the LOGIN NODE (which has internet access)

# Get the directory where this script is located and change to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT" || {
    echo "Error: Could not change to project root: $PROJECT_ROOT"
    exit 1
}

echo "=== Pre-downloading Models ==="
echo "This script downloads models to models/ directory"
echo "Run this on the LOGIN NODE before submitting jobs"
echo ""

# Load Python module
module load python/3.12
source ~/venv/bin/activate

# Create models directory
mkdir -p models

# Function to download a model
download_model() {
    local model_id=$1
    echo ""
    echo "Downloading $model_id..."
    
    # Extract owner and model name
    if [[ "$model_id" == *"/"* ]]; then
        owner=$(echo "$model_id" | cut -d'/' -f1)
        model_name=$(echo "$model_id" | cut -d'/' -f2)
        local_dir="models/$owner/$model_name"
    else
        local_dir="models/$model_id"
    fi
    
    # Check if already downloaded
    if [ -d "$local_dir" ] && [ -f "$local_dir/config.json" ]; then
        echo "  ✓ $model_id already downloaded at $local_dir"
        return 0
    fi
    
    # Download using Python
    python -c "
from huggingface_hub import snapshot_download
import os
model_id = '$model_id'
local_dir = '$local_dir'
os.makedirs(local_dir, exist_ok=True)
print(f'  Downloading to {local_dir}...')
snapshot_download(repo_id=model_id, local_dir=local_dir)
print(f'  ✓ Downloaded {model_id}')
"
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Successfully downloaded $model_id"
    else
        echo "  ✗ Failed to download $model_id"
        return 1
    fi
}

# Download models (update these to match your needs)
echo "Downloading training and evaluation models..."
download_model "Qwen/Qwen2.5-0.5B-Instruct"
download_model "Qwen/Qwen2.5-7B-Instruct"
download_model "Qwen/Qwen2.5-14B-Instruct"

# For testing with smaller models
# download_model "Qwen/Qwen2.5-0.5B-Instruct"

echo ""
echo "=== Download Complete ==="
echo ""
echo "Models are now in the models/ directory:"
ls -lh models/*/*/config.json 2>/dev/null | awk '{print "  " $9}' || echo "  (check models/ directory)"

echo ""
echo "You can now submit your job - models will be loaded from local directory"

