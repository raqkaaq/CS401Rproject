# Monitoring Your Training Job

This guide explains how to monitor your GRPO training job on BYU's supercomputer.

## Quick Status Check

### 1. Check Job Status
```bash
# See all your jobs
squeue --me

# See detailed info about your job
squeue -j <JOB_ID> -l

# See just your job ID and status
squeue --me --format="%.10i %.9P %.20j %.8u %.2t %.10M %.6D %R"
```

### 2. View Output Logs (While Running)
```bash
# View the output log (stdout)
tail -f grpo_<JOB_ID>.out

# View the error log (stderr)
tail -f grpo_<JOB_ID>.err

# View last 50 lines
tail -n 50 grpo_<JOB_ID>.out

# View with line numbers
tail -f grpo_<JOB_ID>.out | cat -n
```

### 3. Check GPU Usage (If Job is Running)
```bash
# Find which node your job is on
squeue --me --format="%.10i %.20j %.8u %.2t %.10M %.6D %N"

# SSH to the compute node (replace NODENAME with actual node)
ssh NODENAME

# Check GPU usage
nvidia-smi

# Monitor GPU usage in real-time (if nvtop is available)
module load nvtop
nvtop
```

## Detailed Monitoring

### Check Resource Usage

```bash
# Check CPU and memory usage on the compute node
ssh <NODE_NAME> "top -u $USER"

# Or use htop if available
ssh <NODE_NAME> "htop -u $USER"
```

### Monitor Training Progress

The training script outputs progress to `grpo_<JOB_ID>.out`. Look for:
- Model loading messages
- Dataset parsing progress
- Training step information
- Loss/reward metrics
- Checkpoint saves

### Check Disk Space

```bash
# Check your home directory space
df -h $HOME

# Check scratch space (if using)
df -h /scratch

# Check project directory
du -sh ./trainer_output
```

## Common Issues and Solutions

### Job is Pending (PD)

```bash
# Check why job is pending
squeue -j <JOB_ID> -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

# Common reasons:
# - Priority: Job is waiting for higher priority jobs
# - Resources: Requested resources not available
# - Dependency: Waiting for another job
# - Partition: Partition is full
```

### Job Failed

```bash
# Check error log
cat grpo_<JOB_ID>.err

# Check output log for clues
tail -n 100 grpo_<JOB_ID>.out

# Common errors:
# - Out of memory: Reduce batch-size
# - CUDA errors: Check GPU availability
# - Module errors: Check module loading
# - Import errors: Check Python environment
```

### Job is Running but Slow

```bash
# Check GPU utilization
ssh <NODE_NAME> "nvidia-smi"

# If GPU utilization is low:
# - May be CPU-bound (data loading)
# - May be waiting for evaluator model
# - May need to increase batch size
```

## Real-Time Monitoring Script

Create a monitoring script `monitor_job.sh`:

```bash
#!/bin/bash
# Usage: ./monitor_job.sh <JOB_ID>

JOB_ID=$1

if [ -z "$JOB_ID" ]; then
    echo "Usage: ./monitor_job.sh <JOB_ID>"
    echo "Or get your latest job: ./monitor_job.sh \$(squeue --me --format='%.10i' --noheader | head -1)"
    exit 1
fi

echo "Monitoring job $JOB_ID"
echo "===================="

while squeue -j $JOB_ID &>/dev/null; do
    clear
    echo "Job Status: $(date)"
    echo "===================="
    squeue -j $JOB_ID
    echo ""
    echo "Last 20 lines of output:"
    echo "------------------------"
    if [ -f "grpo_${JOB_ID}.out" ]; then
        tail -n 20 grpo_${JOB_ID}.out
    else
        echo "Output file not found yet..."
    fi
    echo ""
    echo "Last 10 lines of errors:"
    echo "------------------------"
    if [ -f "grpo_${JOB_ID}.err" ]; then
        tail -n 10 grpo_${JOB_ID}.err
    else
        echo "No errors yet..."
    fi
    sleep 5
done

echo "Job $JOB_ID has completed!"
```

## Check Training Results

### After Job Completes

```bash
# Check if training completed successfully
tail -n 50 grpo_<JOB_ID>.out | grep -i "completed\|error\|failed"

# Check output directory
ls -lh ./trainer_output/

# Check for saved checkpoints
find ./trainer_output -name "checkpoint-*" -type d

# Check TensorBoard logs
ls -lh ./trainer_output/runs/
```

### View TensorBoard Logs

```bash
# On login node (if X11 forwarding enabled)
tensorboard --logdir=./trainer_output/runs --port=6006

# Or use Open OnDemand if available
# Access via web interface
```

## Useful Slurm Commands

```bash
# Cancel a job
scancel <JOB_ID>

# Cancel all your jobs
scancel -u $USER

# Get detailed job information
scontrol show job <JOB_ID>

# Get job accounting info (after completion)
sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,MaxVMSize

# Check partition information
sinfo -p marylou13h

# Check GPU availability
sinfo -p marylou13h -o "%P %G %F"
```

## Email Notifications

Add to your Slurm script to get email notifications:

```bash
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@byu.edu
```

## Quick Reference

| Task | Command |
|------|---------|
| Check job status | `squeue --me` |
| View output | `tail -f grpo_<JOB_ID>.out` |
| View errors | `tail -f grpo_<JOB_ID>.err` |
| Cancel job | `scancel <JOB_ID>` |
| Job details | `scontrol show job <JOB_ID>` |
| GPU usage | `ssh <NODE> nvidia-smi` |
| Check disk | `df -h $HOME` |

## Tips

1. **Use `tmux` or `screen`** for persistent sessions when monitoring
2. **Check logs regularly** to catch issues early
3. **Monitor GPU utilization** to ensure efficient resource use
4. **Save checkpoints frequently** to avoid losing progress
5. **Check disk space** before long runs

