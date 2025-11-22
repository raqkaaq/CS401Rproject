# Running on BYU Supercomputer

This guide explains how to run the GRPO fine-tuning on BYU's supercomputer cluster.

## Prerequisites

1. **Account Setup**: Ensure you have a BYU Research Computing account
2. **Environment Setup**: Set up your Python environment with required packages
3. **Model Pre-download**: Pre-download models to avoid downloading during job execution

## Model Storage Location

The finetuned model is saved to the directory specified by `--output-dir` (default: `./trainer_output`).

Within the output directory, you'll find:
- **Checkpoint directories**: `checkpoint-{step}/` (if `save_steps` is configured in GRPOConfig)
- **Final model files**: After training completes, the final model will be saved with:
  - `config.json` - Model configuration
  - `pytorch_model.bin` or `model.safetensors` - Model weights
  - `tokenizer.json`, `tokenizer_config.json` - Tokenizer files
  - `generation_config.json` - Generation configuration
- **Training logs**: TensorBoard logs and training metrics

### Example Output Structure
```
trainer_output/
├── checkpoint-100/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── ...
├── checkpoint-200/
│   └── ...
├── final_model/  (or checkpoint-final)
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── ...
└── runs/
    └── [timestamp]/
        └── events.out.tfevents.*  (TensorBoard logs)
```

## Client Type Selection

The code now automatically detects and uses the appropriate client:

- **`--client-type auto`** (default): Tries Ollama first, falls back to HFClient if Ollama is unavailable
- **`--client-type hf`**: Forces HuggingFace client (recommended for supercomputer)
- **`--client-type ollama`**: Forces Ollama client (will fail if Ollama isn't running)

### Model Format for Different Clients

- **HFClient**: Use HuggingFace format (e.g., `Qwen/Qwen2.5-0.5B-Instruct`)
- **OllamaClient**: Use Ollama format (e.g., `qwen2.5:0.5b-instruct`)

## Slurm Submission

### 1. Prepare Your Environment

On the login node, set up your environment:

```bash
# Load Python module
module load python/3.9  # or 3.10, 3.11 - check with: module avail python

# Create and activate virtual environment
python -m venv ~/venv
source ~/venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Pre-download Models

Before submitting your job, download models to avoid network issues on compute nodes:

```bash
# In an interactive session or on login node (if allowed)
python -c "from src.inference import download_model; download_model('Qwen/Qwen2.5-0.5B-Instruct', 'models/Qwen/Qwen2.5-0.5B-Instruct')"
```

Or use an interactive job:
```bash
salloc --gpus=1 --time=1:00:00
# Then download models
```

### 3. Customize the Slurm Script

Edit `submit_training.sh` and adjust:
- Partition name (check with `sinfo`)
- Module names (check with `module avail`)
- Python environment activation method
- Resource requirements (GPUs, memory, time)
- Training parameters

### 4. Submit the Job

```bash
sbatch submit_training.sh
```

### 5. Monitor Your Job

```bash
# Check job status
squeue --me

# View output (while running)
tail -f grpo_<job_id>.out

# View errors
tail -f grpo_<job_id>.err
```

### 6. Access Results

After training completes:
- Check the output directory: `./trainer_output/`
- Load the model: `AutoModelForCausalLM.from_pretrained("./trainer_output/checkpoint-<step>")`

## Resource Recommendations

Based on model size:
- **0.5B models**: 1 GPU, 16-32GB RAM, 8 CPUs
- **1-3B models**: 1-2 GPUs, 32-64GB RAM, 8-16 CPUs
- **7B+ models**: 2-4 GPUs, 64GB+ RAM, 16+ CPUs

Adjust `--batch-size` and `--num-epochs` based on your dataset size and time limits.

## Troubleshooting

1. **Ollama connection errors**: Use `--client-type hf` to force HuggingFace client
2. **Out of memory**: Reduce `--batch-size` or request more memory/GPUs
3. **Model download fails**: Pre-download models before submitting job
4. **CUDA errors**: Ensure CUDA module is loaded and GPU is requested correctly

## Additional Notes

- Use `tmux` or `screen` for long-running interactive sessions
- Check BYU's GPU restrictions: https://rc.byu.edu/wiki/?page=Getting+Started+with+GPUs
- For checkpointing, ensure your code saves periodically (configure `save_steps` in GRPOConfig)

