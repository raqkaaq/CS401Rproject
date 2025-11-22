# Training Guide - Step-by-Step Instructions

This guide walks you through running GRPO training on the BYU supercomputer.

## Prerequisites Checklist

Before starting, ensure you have:

- [ ] SSH access to the BYU supercomputer login node
- [ ] Project code in `~/CS401Rproject`
- [ ] Virtual environment set up
- [ ] Models pre-downloaded
- [ ] Datasets pre-downloaded

---

## Step 1: Initial Setup (One-Time, on Login Node)

If you haven't set up your environment yet:

```bash
cd ~/CS401Rproject
./prepare_environment.sh
```

This will:
- Set up Python virtual environment
- Install required packages
- Download models (when prompted)
- Download datasets (when prompted)

**Time:** ~10-30 minutes (depending on download speeds)

---

## Step 2: Verify Setup (Login Node)

Check that everything is ready:

```bash
cd ~/CS401Rproject

# Check dataset cache
./check_dataset_cache.sh

# Quick syntax check
./test_script.sh
```

**Expected output:**
- ✓ Dataset cache exists and is accessible
- ✓ All tests pass (warnings about CUDA on login node are OK)

---

## Step 3: Quick Test (Optional - Interactive Session)

Test with a minimal run to verify everything works:

```bash
# Request interactive GPU session (1 hour)
salloc --partition=m13h --gres=gpu:h200:1 --time=1:00:00 --mem=128G --cpus-per-task=16

# Once allocated, activate environment
cd ~/CS401Rproject
module load cuda
module load python/3.12
source ~/venv/bin/activate

# Run minimal test (1 sample, 1 epoch)
python src/main.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --parser-type math \
  --evaluator-type math \
  --evaluator-model Qwen/Qwen2.5-0.5B-Instruct \
  --client-type hf \
  --num-samples 1 \
  --batch-size 1 \
  --num-epochs 1 \
  --save-steps 1 \
  --output-dir ./test_output

# Exit when done
exit
```

**Time:** ~5-10 minutes

**If this works**, proceed to full training. If it fails, check error messages and fix issues.

---

## Step 4: Submit Full Training Job (Login Node)

Once testing is successful, submit the full training job:

```bash
cd ~/CS401Rproject

# Submit the job
sbatch submit_training.sh
```

**Expected output:**
```
Submitted batch job 8688046
```

**Note the job ID** - you'll use it to monitor the job.

---

## Step 5: Monitor Your Job

### Option A: Use the monitoring script

```bash
# Replace JOBID with your actual job ID
./monitor_job.sh 8688046
```

### Option B: Manual monitoring

```bash
# Check job status
squeue -u $USER

# View output (replace JOBID)
tail -f grpo_JOBID.out

# View errors (replace JOBID)
tail -f grpo_JOBID.err
```

### Option C: Check job details

```bash
# Detailed job information
scontrol show job JOBID

# Job statistics (after completion)
seff JOBID
```

---

## Step 6: Check Results

After training completes:

```bash
# Check output directory
ls -lh trainer_output/

# Structure will be:
# trainer_output/
#   ├── checkpoint-500/    # Checkpoints (if save-strategy=steps)
#   ├── checkpoint-1000/
#   ├── ...
#   └── checkpoint-final/   # Final model
#       ├── config.json
#       ├── model.safetensors
#       ├── tokenizer.json
#       └── ...
```

---

## Training Configuration

The default configuration in `submit_training.sh`:

- **Model:** `Qwen/Qwen2.5-7B-Instruct` (training)
- **Evaluator Model:** `Qwen/Qwen2.5-14B-Instruct` (evaluation)
- **Dataset:** `trl-lib/DeepMath-103K` (math problems)
- **Epochs:** 1
- **Batch Size:** 8
- **Learning Rate:** 5e-6
- **Checkpoints:** Every 500 steps
- **Output:** `./trainer_output/`

### Customizing Training

Edit `submit_training.sh` and modify the `python src/main.py` command:

```bash
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
  --logging-steps 10 \
  --num-samples 10000  # Add this to limit dataset size
```

**Available arguments:**
- `--num-samples N`: Limit dataset to N samples (useful for testing)
- `--num-epochs N`: Number of training epochs
- `--batch-size N`: Batch size (adjust based on GPU memory)
- `--learning-rate FLOAT`: Learning rate (e.g., 5e-6)
- `--save-steps N`: Save checkpoint every N steps
- `--save-strategy {no,epoch,steps}`: When to save checkpoints
- `--logging-steps N`: Log metrics every N steps

---

## Troubleshooting

### Job stuck in PENDING

```bash
# Check why
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

# Common reasons:
# - (Resources): Waiting for GPU availability
# - (Priority): Lower priority than other jobs
# - (PartitionDown): Partition is down
```

**Solution:** Wait, or try a different partition/GPU type.

### "Network is unreachable" error

**Cause:** Dataset or model not cached.

**Solution:**
```bash
# On login node
cd ~/CS401Rproject
./download_datasets.sh
./download_models.sh
```

### Out of memory (OOM)

**Solution:** Reduce batch size in `submit_training.sh`:
```bash
--batch-size 4  # Instead of 8
```

### Job cancelled/timeout

**Solution:** Increase time limit in `submit_training.sh`:
```bash
#SBATCH --time=48:00:00  # Instead of 24:00:00
```

### Checkpoint not found

Checkpoints are saved to `--output-dir` (default: `./trainer_output/`). If using `--save-strategy steps`, checkpoints appear every `--save-steps` steps.

---

## Quick Reference

### Essential Commands

```bash
# Submit job
sbatch submit_training.sh

# Monitor job
squeue -u $USER
./monitor_job.sh JOBID

# Cancel job
scancel JOBID

# View output
tail -f grpo_JOBID.out

# Check results
ls -lh trainer_output/
```

### Resource Limits (Current Config)

- **GPUs:** 1x H200
- **CPUs:** 16
- **Memory:** 128GB
- **Time:** 24 hours
- **Partition:** m13h

---

## Next Steps After Training

1. **Evaluate the model:** Test on validation set
2. **Compare checkpoints:** Check which checkpoint performs best
3. **Fine-tune hyperparameters:** Adjust learning rate, batch size, etc.
4. **Scale up:** Increase dataset size or number of epochs

---

## Support

If you encounter issues:

1. Check error logs: `grpo_JOBID.err`
2. Check output logs: `grpo_JOBID.out`
3. Verify setup: `./check_dataset_cache.sh`
4. Review this guide and `SUPERCOMPUTER_SETUP.md`

