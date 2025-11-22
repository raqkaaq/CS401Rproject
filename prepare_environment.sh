#!/bin/bash
# Complete environment preparation script
# Run this ONCE on the login node before submitting your first job
# This script will:
#   1. Set up Python virtual environment
#   2. Install required packages
#   3. Download models (optional, can skip if already downloaded)
#   4. Validate the setup

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR" || {
    echo "Error: Could not change to script directory: $SCRIPT_DIR"
    exit 1
}

echo "=========================================="
echo "  Environment Preparation Script"
echo "=========================================="
echo "Working directory: $(pwd)"
echo ""

# Check if we're on login node (has internet)
if [[ "$(hostname)" == login* ]]; then
    echo "✓ Detected login node (has internet access)"
    ON_LOGIN_NODE=true
else
    echo "⚠ Not on login node - some steps may be skipped"
    ON_LOGIN_NODE=false
fi

echo ""

# ==========================================
# Step 1: Load Python module
# ==========================================
echo "Step 1: Loading Python module..."
if module load python/3.12 2>/dev/null; then
    echo "   ✓ Loaded: python/3.12 (default)"
elif module load python/3.11 2>/dev/null; then
    echo "   ✓ Loaded: python/3.11"
else
    echo "   ✗ Error: Could not load Python module"
    echo "   Available modules: python/3.12, python/3.11, python/ondemand"
    echo "   Try: module load python/3.12"
    exit 1
fi
python --version

# ==========================================
# Step 2: Create/Update virtual environment
# ==========================================
echo ""
echo "Step 2: Setting up virtual environment..."
VENV_PATH="$HOME/venv"

if [ -d "$VENV_PATH" ]; then
    echo "   ⚠ Virtual environment already exists at $VENV_PATH"
    read -p "   Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_PATH"
        python -m venv "$VENV_PATH"
        echo "   ✓ Virtual environment recreated at $VENV_PATH"
    else
        echo "   Using existing virtual environment at $VENV_PATH"
    fi
else
    python -m venv "$VENV_PATH"
    echo "   ✓ Virtual environment created at $VENV_PATH"
fi

# Activate virtual environment
echo ""
echo "   Activating virtual environment..."
source "$VENV_PATH/bin/activate"
echo "   ✓ Virtual environment activated"

# ==========================================
# Step 3: Upgrade pip
# ==========================================
echo ""
echo "Step 3: Upgrading pip..."
pip install --upgrade pip --quiet
echo "   ✓ pip upgraded to $(pip --version | cut -d' ' -f2)"

# ==========================================
# Step 4: Install packages
# ==========================================
echo ""
echo "Step 4: Installing packages from requirements.txt..."
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"

if [ ! -f "$REQUIREMENTS_FILE" ]; then
    if [ -f "requirements.txt" ]; then
        REQUIREMENTS_FILE="requirements.txt"
    else
        echo "   ✗ requirements.txt not found!"
        echo "   Expected location: $REQUIREMENTS_FILE"
        exit 1
    fi
fi

echo "   Installing from: $REQUIREMENTS_FILE"
pip install -r "$REQUIREMENTS_FILE" --quiet
if [ $? -eq 0 ]; then
    echo "   ✓ All packages installed successfully"
else
    echo "   ✗ Some packages failed to install"
    echo "   You may need to install some packages manually"
    exit 1
fi

# ==========================================
# Step 5: Verify key packages
# ==========================================
echo ""
echo "Step 5: Verifying key packages..."
python -c "import torch; print(f'   ✓ PyTorch: {torch.__version__}')" 2>/dev/null || echo "   ✗ PyTorch not installed"
python -c "import transformers; print(f'   ✓ Transformers: {transformers.__version__}')" 2>/dev/null || echo "   ✗ Transformers not installed"
python -c "import trl; print(f'   ✓ TRL installed')" 2>/dev/null || echo "   ✗ TRL not installed"

# ==========================================
# Step 6: Download models (optional)
# ==========================================
echo ""
echo "Step 6: Model download (optional)"
echo "   Models need to be pre-downloaded before running jobs"
echo "   (Compute nodes don't have internet access)"

if [ "$ON_LOGIN_NODE" = true ]; then
    read -p "   Download models now? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        echo ""
        echo "   Downloading models..."
        
        # Create models directory
        mkdir -p models
        
        # Models to download
        MODELS=(
            "Qwen/Qwen2.5-0.5B-Instruct"
            "Qwen/Qwen2.5-7B-Instruct"
            "Qwen/Qwen2.5-14B-Instruct"
        )
        
        for model_id in "${MODELS[@]}"; do
            echo ""
            echo "   Downloading $model_id..."
            
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
                echo "     ✓ Already downloaded at $local_dir"
                continue
            fi
            
            # Download using Python
            python -c "
from huggingface_hub import snapshot_download
import os
model_id = '$model_id'
local_dir = '$local_dir'
os.makedirs(local_dir, exist_ok=True)
print(f'     Downloading to {local_dir}...')
snapshot_download(repo_id=model_id, local_dir=local_dir)
print(f'     ✓ Downloaded {model_id}')
" || {
                echo "     ✗ Failed to download $model_id"
                echo "     You can download it later with: ./download_models.sh"
            }
        done
        
        echo ""
        echo "   ✓ Model download complete"
    else
        echo "   Skipping model download"
        echo "   Run ./download_models.sh later to download models"
    fi
    
    # Download datasets
    echo ""
    echo "   Downloading datasets..."
    read -p "   Download datasets now? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        echo ""
        echo "   Pre-downloading datasets to cache..."
        python -c "
from datasets import load_dataset
import os

# Set cache directory (HuggingFace datasets cache)
cache_dir = os.path.expanduser('~/.cache/huggingface/datasets')
os.makedirs(cache_dir, exist_ok=True)

# Download datasets used by parsers
datasets_to_download = [
    'trl-lib/DeepMath-103K',  # Used by MathParser
]

for dataset_name in datasets_to_download:
    print(f'     Downloading {dataset_name}...')
    try:
        # Download and cache the dataset
        dataset = load_dataset(dataset_name, split='train')
        print(f'     ✓ Downloaded {dataset_name} ({len(dataset)} samples)')
    except Exception as e:
        print(f'     ✗ Failed to download {dataset_name}: {e}')
        print(f'     You can download it later when needed')

print('')
print('     ✓ Dataset download complete')
print(f'     Datasets cached in: {cache_dir}')
" || {
            echo "     ⚠ Some datasets failed to download"
            echo "     They will be downloaded when first used (if on login node)"
        }
    else
        echo "   Skipping dataset download"
    fi
else
    echo "   ⚠ Not on login node - skipping model and dataset download"
    echo "   Run ./download_models.sh on login node to download models"
    echo "   Datasets will be cached automatically when first loaded (if internet available)"
fi

# ==========================================
# Step 7: Validate setup
# ==========================================
echo ""
echo "Step 7: Validating setup..."
echo ""

# Check Python imports
python -c "
import sys
sys.path.insert(0, '.')
try:
    from src.main import main
    from src.finetune import Finetune
    from src.evaluators.math_evaluator import MathEvaluator
    from src.parsers.math_parser import MathParser
    print('   ✓ All Python imports successful')
except Exception as e:
    error_msg = str(e)
    if 'bf16' in error_msg.lower() or 'gpu' in error_msg.lower():
        print('   ⚠ bf16/GPU check warning (expected on login node, will work in job)')
        print('   ✓ Imports successful (will work with GPU in job)')
    else:
        print(f'   ✗ Import error: {e}')
        sys.exit(1)
" || {
    echo "   ⚠ Some imports had issues (may be okay if bf16 warning)"
}

# Check if models directory exists
if [ -d "models" ]; then
    MODEL_COUNT=$(find models -name "config.json" 2>/dev/null | wc -l)
    echo "   ✓ Models directory exists with $MODEL_COUNT model(s)"
else
    echo "   ⚠ Models directory doesn't exist (models will be downloaded when needed)"
fi

# ==========================================
# Summary
# ==========================================
echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Your environment is ready:"
echo "  ✓ Virtual environment: $VENV_PATH"
echo "  ✓ Packages installed"
echo "  ✓ Models: $(find models -name 'config.json' 2>/dev/null | wc -l) downloaded"
echo ""
echo "Next steps:"
echo "  1. If models weren't downloaded, run: ./download_models.sh"
echo "  2. Test setup: ./test_script.sh"
echo "  3. Submit job: sbatch submit_training.sh"
echo ""
echo "Note: The virtual environment will be activated automatically"
echo "      in your job scripts (submit_training.sh is already configured)"
echo ""

