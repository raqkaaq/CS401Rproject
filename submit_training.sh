#!/bin/bash
#SBATCH --job-name=grpo_training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=grpo_%j.out
#SBATCH --error=grpo_%j.err

# GPU and Partition Configuration
# Based on BYU's actual partition names (from sinfo output):
# - m13h: H200 GPUs (141GB) - 4 nodes, 8 GPUs per node
# - m13l: L40S GPUs (48GB) - 4 nodes, 4 GPUs per node  
# - mgh: GH200 (Grace Hopper) - 2 nodes, 1 GPU per node
# - cs2: H100 GPUs (96GB) - 2 nodes, 8 GPUs per node
# - cs/cssp1/dw: A100 GPUs (80GB) - various nodes, 8 GPUs per node
# - m9g: P100 GPUs (16GB) - 40 nodes, 4 GPUs per node

# OPTION 1: H200 (RECOMMENDED for 7B+14B models)
#SBATCH --partition=m13h
#SBATCH --gres=gpu:h200:1

# OPTION 2: H100 (Alternative for 7B+14B models)
# Uncomment these and comment out OPTION 1 if H200 is busy:
# #SBATCH --partition=cs2
# #SBATCH --gres=gpu:h100:1

# OPTION 3: A100 (Good for 7B+14B, but may be preemptable)
# #SBATCH --partition=cs
# #SBATCH --gres=gpu:a100:1

# OPTION 4: L40S (Might be tight for both 7B+14B models)
# #SBATCH --partition=m13l
# #SBATCH --gres=gpu:l40s:1

# OPTION 5: Grace Hopper GH200
# #SBATCH --partition=mgh
# #SBATCH --gres=gpu:gh200:1

# NOTE: Only ONE partition/GPU combination should be active!
# Comment out the others. Start with OPTION 1 (m13h with H200).

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Load required modules
# Adjust module names based on what's available on BYU's cluster
# You can check available modules with: module avail
module load cuda

# Load Python module
# BYU has: python/3.12 (default), python/3.11, python/ondemand
module load python/3.12  # or python/3.11 if preferred

# Set up Python environment
# IMPORTANT: You must install packages BEFORE submitting the job!
# 
# On the login node, run:
#   module load python/3.9
#   python -m venv ~/venv
#   source ~/venv/bin/activate
#   pip install --upgrade pip
#   pip install -r requirements.txt
#
# Then uncomment ONE of the options below:

# Option 1: If using venv (RECOMMENDED)
source ~/venv/bin/activate

# Option 2: If using conda
# source ~/.bashrc
# conda activate your_env_name

# Option 3: If Python packages are installed system-wide (unlikely)
# (No activation needed, but packages must be installed)

# Set environment variables
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface

# Memory and performance settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Enable better error reporting
set -e  # Exit on error
set -u  # Exit on undefined variable

# Print GPU information
echo "GPU Information:"
nvidia-smi
echo ""
echo "Requested Resources:"
echo "  GPUs: $SLURM_GPUS"
echo "  CPUs: $SLURM_CPUS_PER_TASK"
echo "  Memory: $SLURM_MEM_PER_NODE"
echo "  Partition: $SLURM_JOB_PARTITION"

# Print Python and package versions
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Change to project directory (adjust path as needed)
# If running from project root, this may not be necessary
# The script will automatically find the project root, but being in the right directory helps
cd "$SLURM_SUBMIT_DIR" || cd ~/CS401Rproject || pwd
echo "Working directory: $(pwd)"

# Run training
# Note: Using --client-type hf to force HuggingFace client (Ollama won't be available on compute nodes)
# When using HF client, --evaluator-model should use HuggingFace format (e.g., "Qwen/Qwen2.5-0.5B-Instruct")
# When using Ollama client, use Ollama format (e.g., "qwen2.5:0.5b-instruct")
# 
# The finetuned model will be saved to the --output-dir directory (default: ./trainer_output)
# Within that directory, you'll find:
#   - checkpoint-{step}/ directories (if save_steps is configured)
#   - Final model files (config.json, model files, tokenizer files)
#   - Training logs and metrics
python src/main.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --parser-type math \
  --evaluator-type math \
  --evaluator-model Qwen/Qwen2.5-14B-Instruct \
  --client-type hf \
  --output-dir ./trainer_output \
  --num-epochs 1 \
  --learning-rate 5e-6 \
  --batch-size 8 \
  --save-steps 500 \
  --save-strategy steps \
  --logging-steps 10
# Save options:
#   --save-strategy steps --save-steps 500    # Save every 500 steps (current)
#   --save-strategy epoch                     # Save at end of each epoch
#   --save-strategy no                        # Only save final model
# Note: To save every N epochs, calculate steps: (dataset_size / batch_size) * N
#       Then use: --save-strategy steps --save-steps <calculated_steps>

# Print completion time
echo "End Time: $(date)"
echo "Job completed successfully!"

# Final GPU status
echo ""
echo "Final GPU Status:"
nvidia-smi

