#!/bin/bash
# Setup script to install packages before submitting jobs
# Run this ONCE on the login node before submitting your first job

echo "=== Setting up Python Environment ==="
echo ""

# Load Python module
echo "1. Loading Python module..."
module load python/3.9  # or python/3.10, python/3.11 - check what's available
if [ $? -ne 0 ]; then
    echo "   Error: Could not load Python module"
    echo "   Try: module avail python"
    exit 1
fi
echo "   ✓ Python module loaded"
python --version

# Create virtual environment
echo ""
echo "2. Creating virtual environment..."
if [ -d "$HOME/venv" ]; then
    echo "   ⚠ Virtual environment already exists at ~/venv"
    read -p "   Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf ~/venv
        python -m venv ~/venv
        echo "   ✓ Virtual environment recreated"
    else
        echo "   Using existing virtual environment"
    fi
else
    python -m venv ~/venv
    echo "   ✓ Virtual environment created at ~/venv"
fi

# Activate virtual environment
echo ""
echo "3. Activating virtual environment..."
source ~/venv/bin/activate
echo "   ✓ Virtual environment activated"

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

