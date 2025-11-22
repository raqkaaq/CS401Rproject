# Accelerate Guide

This guide explains how to use Hugging Face Accelerate to accelerate your training.

## What is Accelerate?

Hugging Face Accelerate is a library that simplifies distributed training and provides optimizations for:
- **Better GPU memory management** - More efficient memory usage
- **Mixed precision training** - Automatic bf16/fp16 support
- **Multi-GPU training** - Easy scaling to multiple GPUs
- **Gradient accumulation** - Optimized gradient handling
- **CPU offloading** - Move unused parameters to CPU when needed

## Current Setup

Your training script (`submit_training.sh`) now uses `accelerate launch` instead of `python`:

```bash
accelerate launch src/main.py [arguments...]
```

This automatically detects your GPU configuration and optimizes training accordingly.

## Single GPU (Current Default)

For single GPU training (what you're currently using), accelerate will:
- Automatically detect the GPU
- Enable mixed precision (bf16) if supported
- Optimize memory usage
- No configuration needed!

## Multi-GPU Setup

If you want to use multiple GPUs, you have two options:

### Option 1: Configure Accelerate (Recommended)

1. **On the login node**, run:
   ```bash
   accelerate config
   ```

2. Answer the prompts:
   - **In which compute environment are you running?** → `This machine`
   - **Which type of machine are you using?** → `multi-GPU`
   - **How many different machines will you use?** → `1`
   - **What is the rank of this machine?** → `0`
   - **What is the IP address of the machine that will host the main process?** → `localhost` (or the node hostname)
   - **What is the main port to be used?** → `29500` (default)
   - **Which GPUs should be used?** → `all` (or specify like `0,1,2,3`)
   - **Do you want to use DeepSpeed?** → `no` (unless you want DeepSpeed)
   - **Do you want to use FullyShardedDataParallel?** → `no` (unless you want FSDP)
   - **Do you want to use Megatron-LM?** → `no`
   - **Do you want to enable mixed precision training?** → `bf16` (for H200/H100/A100)

3. This creates a config file at `~/.cache/huggingface/accelerate/default_config.yaml`

4. **Update your SLURM script** to request multiple GPUs:
   ```bash
   #SBATCH --gres=gpu:h200:4  # Request 4 GPUs instead of 1
   ```

5. **Update environment variables** in `submit_training.sh`:
   ```bash
   # Remove or comment out this line (accelerate handles it):
   # export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
   ```

### Option 2: Command-Line Arguments (Quick Test)

You can also specify multi-GPU directly in the command:

```bash
accelerate launch --num_processes=4 --num_machines=1 --mixed_precision=bf16 src/main.py [arguments...]
```

But Option 1 (config file) is recommended for consistency.

## Benefits You'll See

With accelerate, you should notice:
- **Faster training** - Better GPU utilization
- **Lower memory usage** - More efficient memory management
- **Easier scaling** - Simple switch to multi-GPU when needed
- **Better stability** - Optimized gradient handling

## Checking Accelerate Configuration

To see your current accelerate config:

```bash
accelerate env
```

This shows:
- Number of GPUs detected
- Mixed precision settings
- Distributed training setup
- CUDA version

## Example: Multi-GPU Training

If you want to use 4 GPUs on an H200 node:

1. **Update SLURM script** (`submit_training.sh`):
   ```bash
   #SBATCH --gres=gpu:h200:4  # Request 4 GPUs
   ```

2. **Configure accelerate** (on login node):
   ```bash
   accelerate config
   # Select: multi-GPU, 4 GPUs, bf16
   ```

3. **Update batch size** (optional, but recommended):
   ```bash
   # In submit_training.sh, increase batch size:
   --batch-size 32  # 4x the single-GPU batch size
   ```

4. **Submit job**:
   ```bash
   sbatch submit_training.sh
   ```

## Troubleshooting

### "No GPU detected"
- Make sure you're on a compute node with GPUs
- Check: `nvidia-smi`
- Verify: `python -c "import torch; print(torch.cuda.is_available())"`

### "CUDA out of memory"
- Reduce batch size: `--batch-size 4` (or lower)
- Enable gradient checkpointing in GRPOConfig (if supported)
- Use CPU offloading (advanced, requires accelerate config)

### "Accelerate not found"
- Make sure accelerate is installed: `pip install accelerate`
- Check: `pip list | grep accelerate`
- Reinstall if needed: `pip install --upgrade accelerate`

## Advanced: DeepSpeed Integration

For very large models, you can use DeepSpeed with accelerate:

1. Install DeepSpeed: `pip install deepspeed`
2. Configure: `accelerate config` → Select DeepSpeed
3. Accelerate will handle DeepSpeed automatically

## More Information

- [Accelerate Documentation](https://huggingface.co/docs/accelerate)
- [Accelerate GitHub](https://github.com/huggingface/accelerate)
- [TRL with Accelerate](https://huggingface.co/docs/trl/main/en/accelerate)

