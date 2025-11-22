# GPU Memory Guide

## Your Current Configuration

- **Training Model**: Qwen/Qwen2.5-7B-Instruct (7B parameters)
- **Evaluator Model**: Qwen/Qwen2.5-14B-Instruct (14B parameters)
- **Batch Size**: 8 (recommended), 128 (too large - will OOM)
- **Estimated Memory**: ~94-103 GB (batch_size=8), ~250-350 GB (batch_size=128)

## Batch Size vs Memory

| Batch Size | Activation Memory | Total Memory | GPU Needed |
|------------|-------------------|--------------|------------|
| 1 | ~1-2 GB | ~85-90 GB | H100/A100 |
| 4 | ~5-8 GB | ~90-95 GB | H100 |
| 8 | ~10-15 GB | ~94-103 GB | **H200 (recommended)** |
| 16 | ~20-30 GB | ~110-130 GB | H200 (tight) |
| 32 | ~40-60 GB | ~130-160 GB | H200 (may OOM) |
| 64 | ~80-120 GB | ~170-220 GB | ❌ Too large |
| 128 | ~160-240 GB | ~250-350 GB | ❌ **Will definitely OOM** |

## GPU Recommendations

| GPU | Memory | Will It Fit? | Recommendation |
|-----|--------|---------------|----------------|
| **H200** | 141 GB | ✅ Yes (tight) | **RECOMMENDED** - Best option |
| **H100** | 96 GB | ⚠️ Maybe | Reduce batch size to 4, or use quantization |
| **A100** | 80 GB | ❌ No | Won't fit - need smaller models or quantization |
| **L40S** | 48 GB | ❌ No | Too small |

## Memory Breakdown

- **7B Training Model (bf16)**: ~14 GB
- **Training Overhead** (gradients, optimizer, activations): ~42-50 GB
- **14B Evaluator Model (bf16, inference only)**: ~28 GB
- **Batch Activations** (batch_size=8): ~10-15 GB
- **Total**: ~94-103 GB

## If Memory is Tight

### Option 1: Reduce Batch Size (Easiest)
```bash
--batch-size 4  # Reduces activation memory by ~50%
```

### Option 2: Use 8-bit Quantization for Evaluator
Modify `HFClient` initialization to load evaluator in 8-bit:
```python
evaluator = MathEvaluator(
    model="Qwen/Qwen2.5-14B-Instruct",
    client=HFClient(load_in_8bit=True),  # Add this
    ...
)
```
This reduces 14B model from ~28GB to ~14GB.

### Option 3: Use Smaller Evaluator Model
```bash
--evaluator-model Qwen/Qwen2.5-7B-Instruct  # Use same size as training model
```

### Option 4: Gradient Checkpointing
If GRPOConfig supports it, enable gradient checkpointing to trade compute for memory.

## Testing Memory Usage

Before running the full training, test with a small run:

```bash
accelerate launch src/main.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --parser-type math \
  --evaluator-type math \
  --evaluator-model Qwen/Qwen2.5-14B-Instruct \
  --client-type hf \
  --num-samples 10 \  # Small test
  --batch-size 8 \
  --num-epochs 1 \
  --output-dir ./test_output
```

Monitor memory with:
```bash
watch -n 1 nvidia-smi
```

## Expected Behavior

- **H200**: Should work, but monitor for OOM errors
- **H100**: May OOM with batch_size=8, try batch_size=4
- **A100/L40S**: Will definitely OOM - need optimizations

## Memory Optimization Tips

1. **Use bf16** (already enabled in your config) - saves ~50% vs fp32
2. **Reduce batch size** - linear reduction in activation memory
3. **Use quantization** - 8-bit reduces model memory by ~50%
4. **Clear cache** between runs: `torch.cuda.empty_cache()`
5. **Use gradient accumulation** - simulate larger batches with less memory

---

## Using 0.5B Models (Much Lower Memory!)

If you use **Qwen/Qwen2.5-0.5B-Instruct** for both training and evaluation:

### Memory Requirements for 0.5B Models

| Batch Size | Activation Memory | Total Memory | GPU Needed |
|------------|-------------------|--------------|------------|
| 8 | ~0.5-1 GB | ~6-8 GB | ✅ **Any GPU** |
| 16 | ~1-2 GB | ~7-9 GB | ✅ **Any GPU** |
| 32 | ~2-4 GB | ~8-11 GB | ✅ **Any GPU** |
| 64 | ~4-8 GB | ~10-15 GB | ✅ **Any GPU** |
| **128** | **~8-15 GB** | **~14-22 GB** | ✅ **Any GPU (even L40S!)** |
| 256 | ~16-30 GB | ~22-37 GB | ✅ A100/H100/H200 |

### Memory Breakdown (0.5B Models)

- **0.5B Training Model (bf16)**: ~1 GB
- **Training Overhead** (gradients, optimizer, activations): ~3-4 GB
- **0.5B Evaluator Model (bf16, inference only)**: ~1 GB
- **Batch Activations** (batch_size=128): ~8-15 GB
- **Total**: ~14-22 GB

### GPU Recommendations for 0.5B Models

| GPU | Memory | Batch Size 128? | Recommendation |
|-----|--------|-----------------|----------------|
| **L40S** | 48 GB | ✅ **Yes!** | Perfect for 0.5B models |
| **A100** | 80 GB | ✅ **Yes!** | Plenty of room |
| **H100** | 96 GB | ✅ **Yes!** | Can go even larger |
| **H200** | 141 GB | ✅ **Yes!** | Can use batch_size=256+ |

### Example Command for 0.5B Models

```bash
accelerate launch src/main.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --parser-type math \
  --evaluator-type math \
  --evaluator-model Qwen/Qwen2.5-0.5B-Instruct \
  --client-type hf \
  --num-samples 500 \
  --batch-size 128 \  # ✅ Totally fine with 0.5B models!
  --num-epochs 3 \
  --output-dir ./test_output
```

### Benefits of 0.5B Models

1. **Much faster training** - Smaller models train faster
2. **Larger batch sizes** - Can use batch_size=128+ easily
3. **Works on smaller GPUs** - Even L40S (48GB) can handle it
4. **Faster inference** - Evaluator runs much faster
5. **Lower cost** - Less GPU time needed

### Trade-offs

- **Lower model capacity** - May not perform as well on complex tasks
- **Less capable evaluator** - 0.5B evaluator may be less accurate
- **Still good for testing** - Perfect for development and testing workflows

### Recommendation

For **testing and development**, 0.5B models with batch_size=128 is perfect! You'll use ~15-20 GB, which fits on almost any GPU.

