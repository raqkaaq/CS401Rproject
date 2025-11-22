# Quick Start - All Commands Before Training

Run these commands **on the login node** before submitting your training job.

## Step 1: Get Latest Code

```bash
cd ~/CS401Rproject
git pull
```

## Step 2: Setup Environment (One-Time Only)

If you haven't set up your environment yet:

```bash
cd ~/CS401Rproject
./prepare_environment.sh
```

When prompted:
- **Recreate venv?** → `N` (unless you want to start fresh)
- **Download models?** → `Y` (if not already downloaded)
- **Download datasets?** → `Y` (if not already downloaded)

**Skip this step if you've already run it before.**

## Step 3: Verify Models Are Downloaded

```bash
# Check if models exist
ls -la models/Qwen/Qwen2.5-7B-Instruct/config.json
ls -la models/Qwen/Qwen2.5-14B-Instruct/config.json

# If missing, download them:
./download_models.sh
```

## Step 4: Verify Datasets Are Cached

```bash
# Check dataset cache
./check_dataset_cache.sh

# If missing, download them:
./download_datasets.sh
```

## Step 5: Quick Test (Optional but Recommended)

Test with a minimal run to catch errors early:

```bash
# Request interactive GPU session
salloc --partition=m13h --gres=gpu:h200:1 --time=1:00:00 --mem=128G --cpus-per-task=16

# Once allocated, set environment variables
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Load modules and activate venv
module load cuda python/3.12
source ~/venv/bin/activate

# Run minimal test
cd ~/CS401Rproject
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

## Step 6: Submit Full Training Job

Once everything is verified:

```bash
cd ~/CS401Rproject
sbatch submit_training.sh
```

**Note:** The training script now uses `accelerate launch` for better performance. See `ACCELERATE_GUIDE.md` for details on multi-GPU setup.

You'll see: `Submitted batch job 8688046` (note the job ID)

## Step 7: Monitor Your Job

```bash
# Replace JOBID with your actual job ID
./monitor_job.sh JOBID

# Or manually:
squeue -u $USER
tail -f grpo_JOBID.out
```

---

## Complete Command Sequence (Copy-Paste Ready)

If you've already set up your environment before, here's the minimal sequence:

```bash
# 1. Get latest code
cd ~/CS401Rproject
git pull

# 2. Verify models (download if missing)
ls models/Qwen/Qwen2.5-7B-Instruct/config.json || ./download_models.sh

# 3. Verify datasets (download if missing)
./check_dataset_cache.sh || ./download_datasets.sh

# 4. Submit job
sbatch submit_training.sh

# 5. Monitor (replace JOBID)
./monitor_job.sh JOBID
```

---

## Troubleshooting

### Models not found?
```bash
./download_models.sh
```

### Datasets not cached?
```bash
./download_datasets.sh
```

### Environment not set up?
```bash
./prepare_environment.sh
```

### Check what's installed?
```bash
source ~/venv/bin/activate
pip list | grep -E "(torch|transformers|trl|datasets)"
```

---

## What Each Script Does

- **`prepare_environment.sh`** - Sets up Python venv, installs packages, downloads models/datasets
- **`download_models.sh`** - Downloads models to `models/` directory
- **`download_datasets.sh`** - Downloads datasets to `~/.cache/huggingface/datasets/`
- **`check_dataset_cache.sh`** - Verifies dataset cache exists and is accessible
- **`submit_training.sh`** - Submits the training job to Slurm
- **`monitor_job.sh`** - Monitors a running job

