#!/bin/bash
# Setup script to install packages before submitting jobs
# Run this ONCE on the login node before submitting your first job

echo "=== Setting up Python Environment ==="
echo ""

# Load Python module
echo "1. Loading Python module..."
# BYU has: python/3.12 (default), python/3.11, python/ondemand
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

# Create virtual environment
echo ""
echo "2. Creating virtual environment..."
echo "   Location options:"
echo "   - ~/venv (home directory) - RECOMMENDED: shared across all nodes"
echo "   - ~/scratch/venv (scratch space) - Faster, but may not be shared"
echo "   - Project directory - If you have project-specific space"
echo ""

VENV_PATH="$HOME/venv"
# Uncomment to use scratch space instead:
# VENV_PATH="$HOME/scratch/venv"
# mkdir -p "$HOME/scratch"

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
echo "3. Activating virtual environment..."
source "$VENV_PATH/bin/activate"
echo "   ✓ Virtual environment activated from $VENV_PATH"

# Upgrade pip
echo ""
echo "4. Upgrading pip..."
pip install --upgrade pip
echo "   ✓ pip upgraded"

# Install packages
echo ""
echo "5. Installing packages from requirements.txt..."
if [ ! -f "requirements.txt" ]; then
    echo "   ✗ requirements.txt not found!"
    exit 1
fi

pip install -r requirements.txt
if [ $? -eq 0 ]; then
    echo "   ✓ All packages installed successfully"
else
    echo "   ✗ Some packages failed to install"
    echo "   You may need to install some packages manually"
    exit 1
fi

# Verify key packages
echo ""
echo "6. Verifying key packages..."
python -c "import torch; print(f'   ✓ PyTorch: {torch.__version__}')" || echo "   ✗ PyTorch not installed"
python -c "import transformers; print(f'   ✓ Transformers: {transformers.__version__}')" || echo "   ✗ Transformers not installed"
python -c "import trl; print(f'   ✓ TRL installed')" || echo "   ✗ TRL not installed"

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Your environment is ready. You can now:"
echo "  1. Submit jobs: sbatch submit_training.sh"
echo "  2. Test setup: ./test_script.sh"
echo ""
echo "Note: The virtual environment will be activated automatically"
echo "      in your job scripts (submit_training.sh is already configured)"
echo ""
echo "VENV_PATH used: $VENV_PATH"
echo "Update submit_training.sh if you used a different location!"

