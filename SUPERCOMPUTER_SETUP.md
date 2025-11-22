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

### 2. Pre-download Models (REQUIRED)

**IMPORTANT**: Compute nodes don't have internet access. You MUST pre-download models on the login node before submitting jobs.

**Option 1: Use the download script (RECOMMENDED)**
```bash
# On login node (has internet access)
./download_models.sh
```

**Option 2: Manual download**
```bash
# On login node
module load python/3.12
source ~/venv/bin/activate

# Download models
python -c "
from huggingface_hub import snapshot_download
import os

models = [
    'Qwen/Qwen2.5-7B-Instruct',
    'Qwen/Qwen2.5-14B-Instruct'
]

for model_id in models:
    local_dir = f'models/{model_id}'
    os.makedirs(local_dir, exist_ok=True)
    print(f'Downloading {model_id}...')
    snapshot_download(repo_id=model_id, local_dir=local_dir)
    print(f'✓ Downloaded {model_id}')
"
```

**Option 3: Interactive session (if login node doesn't allow downloads)**
```bash
salloc --partition=m13h --gres=gpu:h200:1 --time=1:00:00 --mem=128G --cpus-per-task=16
# Then run download_models.sh or manual download
exit  # Release resources after download
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

### BYU GPU Options

| GPU Type | Memory | Cluster | Nodes | GPUs/Node | Best For |
|----------|--------|---------|-------|-----------|----------|
| **H200** | 141GB | marylou13h | 4 | 8 | **7B+14B models (RECOMMENDED)** |
| **H100** | 96GB | marylouGH | 2 | 1 | 7B+14B models |
| **A100** | 80GB | Preemption | 10 | 8 | 7B+14B models (preemptable) |
| **L40S** | 48GB | marylou13l | 4 | 4 | 1-3B models |
| **P100** | 16GB | marylou9g | 40 | 4 | < 1B models |

### Model Size Recommendations

| Training Model | Evaluator Model | Recommended GPU | CPU | RAM | Partition |
|----------------|-----------------|-----------------|-----|-----|-----------|
| < 1B | < 1B | P100 (16GB) | 8 | 32GB | marylou9g |
| 1-3B | 1-3B | L40S (48GB) | 16 | 64GB | marylou13l |
| 3-7B | 3-7B | A100 (80GB) | 16 | 128GB | Preemption |
| 7B | 7B | H100 (96GB) | 16 | 128GB | marylouGH |
| **7B** | **14B** | **H200 (141GB)** | **16** | **128GB** | **marylou13h** |
| 14B+ | 14B+ | H200 (141GB) | 32 | 256GB | marylou13h |

### Current Configuration (7B training + 14B evaluation)

The default script uses:
- **GPU**: H200 (141GB) - `--gpus=h200:1`
- **Partition**: marylou13h
- **CPUs**: 16
- **Memory**: 128GB

This provides plenty of headroom for both models. Adjust `--batch-size` based on available memory and training speed.

## Troubleshooting

1. **Ollama connection errors**: Use `--client-type hf` to force HuggingFace client
2. **Out of memory**: Reduce `--batch-size` or request more memory/GPUs
3. **Model download fails**: Pre-download models before submitting job
4. **CUDA errors**: Ensure CUDA module is loaded and GPU is requested correctly

## Additional Notes

- Use `tmux` or `screen` for long-running interactive sessions
- Check BYU's GPU restrictions: https://rc.byu.edu/wiki/?page=Getting+Started+with+GPUs
- For checkpointing, ensure your code saves periodically (configure `save_steps` in GRPOConfig)

