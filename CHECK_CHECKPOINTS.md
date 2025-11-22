# How to Check if Checkpoints Worked

## Quick Check

```bash
# Check if checkpoint directories exist
ls -lh trainer_output/

# You should see directories like:
# checkpoint-500/
# checkpoint-1000/
# checkpoint-1500/
# ...
```

## Detailed Check

### 1. List All Checkpoints

```bash
cd ~/CS401Rproject
ls -d trainer_output/checkpoint-* 2>/dev/null | sort -V
```

**Expected output:**
```
trainer_output/checkpoint-500
trainer_output/checkpoint-1000
trainer_output/checkpoint-1500
...
```

### 2. Verify Checkpoint Contents

Each checkpoint should contain essential files:

```bash
# Check a specific checkpoint (replace 500 with your checkpoint number)
CHECKPOINT=checkpoint-500
ls -lh trainer_output/$CHECKPOINT/
```

**Required files:**
- `config.json` - Model configuration
- `model.safetensors` or `pytorch_model.bin` - Model weights
- `tokenizer.json` - Tokenizer
- `tokenizer_config.json` - Tokenizer config
- `generation_config.json` - Generation settings
- `training_args.bin` - Training arguments

### 3. Check Checkpoint Size

```bash
# Check size of each checkpoint
du -sh trainer_output/checkpoint-*

# Or detailed:
du -h trainer_output/checkpoint-*/model.safetensors 2>/dev/null
```

**Expected sizes:**
- 0.5B model: ~1-2 GB per checkpoint
- 7B model: ~14-28 GB per checkpoint
- 14B model: ~28-56 GB per checkpoint

### 4. Verify Checkpoint is Loadable

Test if you can load the checkpoint:

```bash
cd ~/CS401Rproject
module load python/3.12
source ~/venv/bin/activate

python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

checkpoint = 'trainer_output/checkpoint-500'
if os.path.exists(checkpoint):
    print(f'Loading checkpoint from {checkpoint}...')
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        print(f'✓ Tokenizer loaded: {len(tokenizer)} tokens')
        
        # For large models, you might want to skip loading the full model
        # Just check if files exist
        if os.path.exists(f'{checkpoint}/model.safetensors'):
            size = os.path.getsize(f'{checkpoint}/model.safetensors') / (1024**3)
            print(f'✓ Model weights file exists: {size:.2f} GB')
        elif os.path.exists(f'{checkpoint}/pytorch_model.bin'):
            size = os.path.getsize(f'{checkpoint}/pytorch_model.bin') / (1024**3)
            print(f'✓ Model weights file exists: {size:.2f} GB')
        
        print('✓ Checkpoint appears valid')
    except Exception as e:
        print(f'✗ Error loading checkpoint: {e}')
else:
    print(f'✗ Checkpoint directory not found: {checkpoint}')
"
```

### 5. Check Training Logs

```bash
# Check if checkpoints were saved (from job output)
grep -i "checkpoint" grpo_JOBID.out

# Or check for save messages
grep -i "saving" grpo_JOBID.out | tail -20
```

**Expected output:**
```
Saving checkpoint checkpoint-500 to trainer_output/checkpoint-500
Saving checkpoint checkpoint-1000 to trainer_output/checkpoint-1000
...
```

### 6. Count Checkpoints

```bash
# Count how many checkpoints were saved
ls -d trainer_output/checkpoint-* 2>/dev/null | wc -l

# Or with details
ls -d trainer_output/checkpoint-* 2>/dev/null | while read dir; do
    echo "$(basename $dir): $(du -sh $dir | cut -f1)"
done
```

## Complete Checkpoint Verification Script

```bash
#!/bin/bash
# checkpoint_check.sh - Verify all checkpoints

OUTPUT_DIR="trainer_output"

echo "=== Checkpoint Verification ==="
echo ""

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "✗ Output directory not found: $OUTPUT_DIR"
    exit 1
fi

# Find all checkpoints
CHECKPOINTS=$(ls -d $OUTPUT_DIR/checkpoint-* 2>/dev/null | sort -V)

if [ -z "$CHECKPOINTS" ]; then
    echo "⚠ No checkpoints found in $OUTPUT_DIR"
    echo "Check training logs to see if checkpoints were saved"
    exit 1
fi

echo "Found $(echo "$CHECKPOINTS" | wc -l) checkpoints:"
echo ""

# Check each checkpoint
for checkpoint in $CHECKPOINTS; do
    name=$(basename $checkpoint)
    echo "Checking $name..."
    
    # Required files
    required_files=("config.json" "tokenizer.json")
    missing_files=()
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$checkpoint/$file" ]; then
            missing_files+=("$file")
        fi
    done
    
    # Check for model weights
    has_weights=false
    if [ -f "$checkpoint/model.safetensors" ] || [ -f "$checkpoint/pytorch_model.bin" ]; then
        has_weights=true
    fi
    
    if [ ${#missing_files[@]} -eq 0 ] && [ "$has_weights" = true ]; then
        size=$(du -sh "$checkpoint" | cut -f1)
        echo "  ✓ Valid ($size)"
    else
        echo "  ✗ Invalid - missing: ${missing_files[*]}"
        if [ "$has_weights" = false ]; then
            echo "    Missing model weights"
        fi
    fi
done

echo ""
echo "=== Summary ==="
echo "Total checkpoints: $(echo "$CHECKPOINTS" | wc -l)"
echo "Total size: $(du -sh $OUTPUT_DIR | cut -f1)"
```

Save this as `checkpoint_check.sh` and run:
```bash
chmod +x checkpoint_check.sh
./checkpoint_check.sh
```

## Common Issues

### No checkpoints found?

**Check training logs:**
```bash
tail -100 grpo_JOBID.out | grep -i checkpoint
```

**Possible causes:**
- Training didn't reach `--save-steps` threshold
- `--save-strategy` is set to `"no"`
- Training failed before first checkpoint
- Output directory path issue

### Checkpoint incomplete?

**Check if training was interrupted:**
```bash
# Check if job completed
squeue -u $USER
seff JOBID
```

**Check last checkpoint:**
```bash
# Find latest checkpoint
LATEST=$(ls -td trainer_output/checkpoint-* | head -1)
ls -lh $LATEST/
```

### Checkpoint too small?

**Verify model files:**
```bash
# Check model file size
ls -lh trainer_output/checkpoint-500/model.safetensors
```

If it's suspiciously small (< 100MB for 7B model), the checkpoint might be corrupted.

## Quick Reference

```bash
# List checkpoints
ls -d trainer_output/checkpoint-*

# Check latest checkpoint
ls -lh trainer_output/checkpoint-$(ls -td trainer_output/checkpoint-* | head -1 | xargs basename)/

# Count checkpoints
ls -d trainer_output/checkpoint-* | wc -l

# Total size
du -sh trainer_output/

# Check training logs
grep -i "saving checkpoint" grpo_JOBID.out
```

