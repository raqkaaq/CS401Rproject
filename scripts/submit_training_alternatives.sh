#!/bin/bash
# Alternative Slurm configurations for different GPU types
# Copy the relevant section to submit_training.sh

# ============================================================================
# OPTION 1: H200 (141GB) - RECOMMENDED for 7B+14B models
# ============================================================================
#SBATCH --partition=marylou13h
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
# Best for: Large models (7B+ training, 14B+ evaluation)
# Availability: 4 nodes, 8 GPUs per node

# ============================================================================
# OPTION 2: H100 (96GB) - Good for 7B+14B models
# ============================================================================
#SBATCH --partition=marylouGH
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
# Best for: Large models with Grace Hopper architecture
# Availability: 2 nodes, 1 GPU per node

# ============================================================================
# OPTION 3: A100 (80GB) - Preemption only, good for 7B+14B
# ============================================================================
#SBATCH --qos=preempt
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
# Best for: Large models, but jobs can be preempted
# Availability: Preemption nodes, 8 GPUs per node

# ============================================================================
# OPTION 4: L40S (48GB) - Might be tight for both 7B+14B
# ============================================================================
#SBATCH --partition=marylou13l
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
# Best for: Smaller models or single model training
# Availability: 4 nodes, 4 GPUs per node
# Note: May need to reduce batch size or use quantization

# ============================================================================
# OPTION 5: P100 (16GB) - Only for very small models
# ============================================================================
#SBATCH --partition=marylou9g
#SBATCH --gpus=p100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Best for: Models < 1B parameters
# Availability: 40 nodes, 4 GPUs per node

# ============================================================================
# Resource Recommendations by Model Size:
# ============================================================================
# Training Model | Evaluator Model | Recommended GPU | Memory
# --------------|-----------------|-----------------|--------
# < 1B          | < 1B            | P100 (16GB)     | 32GB
# 1-3B          | 1-3B            | L40S (48GB)     | 64GB
# 3-7B          | 3-7B            | A100 (80GB)     | 128GB
# 7B            | 7B              | H100 (96GB)     | 128GB
# 7B            | 14B             | H200 (141GB)    | 128GB
# 14B+          | 14B+            | H200 (141GB)    | 256GB+

