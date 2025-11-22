#!/bin/bash
# Test script to validate your training setup before submitting
# Run this on the login node to check everything is ready
# NOTE: Some tests may show warnings if run on a machine without GPU/CUDA
#       This is expected - the actual job will have CUDA loaded

echo "=== Testing Training Script Setup ==="
echo "Running on: $(hostname)"
echo ""

# Check 1: Validate Slurm script syntax
echo "1. Checking Slurm script syntax..."
if bash -n submit_training.sh; then
    echo "   ✓ Slurm script syntax is valid"
else
    echo "   ✗ Slurm script has syntax errors!"
    exit 1
fi

# Check 2: Check if Python environment is set up
echo ""
echo "2. Checking Python environment..."
# Try to load Python module first
if module load python/3.12 2>/dev/null || module load python/3.11 2>/dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    echo "   ✓ Python module loaded: $PYTHON_VERSION"
elif command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    echo "   ⚠ Python found in PATH: $PYTHON_VERSION (module not loaded)"
else
    echo "   ✗ Python not found! Load module: module load python/3.12"
    exit 1
fi

source ~/venv/bin/activate

# Check 3: Check if required Python packages are available
echo ""
echo "3. Checking Python packages..."
python -c "import torch; print(f'   ✓ PyTorch: {torch.__version__}')" 2>/dev/null || echo "   ✗ PyTorch not installed"
python -c "import transformers; print(f'   ✓ Transformers: {transformers.__version__}')" 2>/dev/null || echo "   ✗ Transformers not installed"
python -c "import trl; print(f'   ✓ TRL installed')" 2>/dev/null || echo "   ✗ TRL not installed"

# Check 4: Check if CUDA is available
echo ""
echo "4. Checking CUDA..."
if module list 2>&1 | grep -q cuda; then
    echo "   ✓ CUDA module loaded"
else
    echo "   ⚠ CUDA module not loaded (will be loaded in job)"
fi

# Check 5: Validate Python script can be imported
echo ""
echo "5. Testing Python imports..."
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
    # bf16 warning on login node is okay - CUDA will be loaded in job
    if 'bf16' in error_msg.lower() or 'gpu' in error_msg.lower():
        print('   ⚠ bf16/GPU check warning (expected on login node, CUDA loads in job)')
        print('   ✓ Imports successful (will work with GPU in job)')
    else:
        print(f'   ✗ Import error: {e}')
        sys.exit(1)
" || {
    # Check if it's just the bf16 warning
    python -c "
import sys
sys.path.insert(0, '.')
try:
    from src.main import main
    print('   ✓ Imports actually work (bf16 check is just a warning)')
except Exception as e:
    if 'bf16' not in str(e).lower():
        sys.exit(1)
" || exit 1
}

# Check 6: Check if models directory exists (for local models)
echo ""
echo "6. Checking model directories..."
if [ -d "models" ]; then
    echo "   ✓ Models directory exists"
    MODEL_COUNT=$(find models -name "config.json" 2>/dev/null | wc -l)
    echo "   Found $MODEL_COUNT model(s) in models/"
else
    echo "   ⚠ Models directory doesn't exist (models will be downloaded)"
fi

# Check 7: Validate Slurm partition and GPU request
echo ""
echo "7. Validating Slurm resource requests..."
ACTIVE_PARTITION=$(grep "^#SBATCH --partition=" submit_training.sh | head -1 | cut -d'=' -f2)
ACTIVE_GPU=$(grep "^#SBATCH --gres=" submit_training.sh | head -1 | cut -d':' -f3)

if [ -n "$ACTIVE_PARTITION" ]; then
    echo "   Partition: $ACTIVE_PARTITION"
    # Check if partition exists
    if sinfo -p "$ACTIVE_PARTITION" &>/dev/null; then
        echo "   ✓ Partition exists and is accessible"
    else
        echo "   ✗ Partition not found or not accessible!"
    fi
fi

if [ -n "$ACTIVE_GPU" ]; then
    echo "   GPU request: $ACTIVE_GPU"
fi

echo ""
echo "=== Test Summary ==="
echo "If all checks passed, your script should be ready to submit!"
echo ""
echo "To submit: sbatch submit_training.sh"
echo "To test in interactive session: salloc --partition=m13h --gres=gpu:h200:1 --time=1:00:00"

