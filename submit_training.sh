#!/bin/bash
#SBATCH --job-name=grpo_training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus=h200:1
#SBATCH --time=24:00:00
#SBATCH --output=grpo_%j.out
#SBATCH --error=grpo_%j.err
#SBATCH --partition=marylou13h  # H200 GPUs (141GB each) - best for 7B+14B models
# Alternative partitions:
# --partition=marylouGH --gpus=h100:1  # H100 (96GB) - Grace Hopper nodes
# --partition=marylou13l --gpus=l40s:1  # L40S (48GB) - might be tight for both models
# For preemption nodes (A100 80GB): remove --partition and add --qos=preempt

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
module load python/3.9  # or python/3.10, python/3.11 - check what's available

# Set up Python environment
# Option 1: If using conda
# source ~/.bashrc
# conda activate your_env_name

# Option 2: If using venv (create it in your home directory first)
# source ~/venv/bin/activate

# Option 3: If Python packages are installed via module
# (Some clusters have pre-installed packages)

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
# cd /path/to/CS401Rproject

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
  --batch-size 8

# Print completion time
echo "End Time: $(date)"
echo "Job completed successfully!"

# Final GPU status
echo ""
echo "Final GPU Status:"
nvidia-smi

