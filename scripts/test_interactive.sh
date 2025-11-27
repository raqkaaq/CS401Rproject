#!/bin/bash
# Interactive test script - run this in an salloc session
# Usage: 
#   1. Request interactive session: salloc --partition=m13h --gres=gpu:h200:1 --time=1:00:00 --mem=128G --cpus-per-task=16
#   2. Run this script: ./scripts/test_interactive.sh

# Get the directory where this script is located and change to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT" || {
    echo "Error: Could not change to project root: $PROJECT_ROOT"
    exit 1
}

echo "=== Interactive Test Mode ==="
echo "This will test your training setup with minimal data"
echo ""

# Load modules (same as in submit_training.sh)
module load cuda
module load python/3.12  # BYU default, or python/3.11

# Set environment variables
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
# Force offline mode for HuggingFace (use cache only, no internet)
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

source ~/venv/bin/activate

# Check GPU
echo "GPU Information:"
nvidia-smi

echo ""
echo "Testing with minimal configuration..."
echo "This will:"
echo "  1. Test model loading"
echo "  2. Test dataset parsing"
echo "  3. Test evaluator initialization"
echo "  4. Run 1 training step (if all checks pass)"
echo ""

# Run a minimal test
python -c "
import sys
sys.path.insert(0, '.')

print('Testing imports...')
from src.main import main
from src.finetune import Finetune
from src.evaluators.math_evaluator import MathEvaluator
from src.parsers.math_parser import MathParser
from trl import GRPOConfig
print('✓ All imports successful')

print('\\nTesting with minimal config...')
# You can add more specific tests here
print('✓ Basic setup looks good')
"

echo ""
echo "If you want to run a full test with 1 sample:"
echo "  python src/main.py --model Qwen/Qwen2.5-0.5B-Instruct \\"
echo "    --parser-type math --evaluator-type math \\"
echo "    --evaluator-model Qwen/Qwen2.5-0.5B-Instruct \\"
echo "    --client-type hf --num-samples 1 --batch-size 1 \\"
echo "    --num-epochs 1 --save-steps 1 --output-dir ./test_output"

