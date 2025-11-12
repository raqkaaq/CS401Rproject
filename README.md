# PRewrite PPO Rewriter (TRL v0.25.0)

Minimal reinforcement learning setup to train an instruction rewriter (PRewrite) with PPO using Hugging Face TRL.

Paper: PRewrite — https://arxiv.org/abs/2401.08189  
TRL Docs: https://huggingface.co/docs/trl/v0.25.0/en/ppo_trainer

## Components

- `load_datasets.py` – Defines `PRewriteDataset`, `RewriteExample`, meta prompt helpers (`build_meta_prompt`, `format_rewriter_query`). The dataset reads a JSONL or Parquet source and yields examples via `iter()`. Each example must have at least: `instruction`, `input`, `output`. Optional: `task` ("qa", "math", "classification"), `template`.
- `finetune.py` – New wrapper `PRewritePPOTrainer` builds a `PPOConfig` internally from kwargs; performs rollouts, evaluates with an external task model, and calls `ppo.step`.
- `inference.py` – Utilities for running inference with Ollama (warming models, generation) and can be extended to perform greedy or search-based rewrite evaluation.
- `run.ipynb` – Example notebook showing dataset loading, model loading, evaluator creation, and a demo training loop.

## Data Format

Each training example (JSONL line or Parquet row):
```jsonc
{
  "instruction": "Rewrite to focus on reasoning",
  "input": "If John has 3 apples and buys 2 more, how many apples does he have?",
  "output": "5",
  "task": "math"
}
```
Fields:
- `instruction`: original (possibly suboptimal) instruction.
- `input`: task-specific input text.
- `output`: ground-truth answer or label.
- `task` (optional): drives reward metric selection (math→numeric, qa→F1 token, else exact).
- `template` (optional): overrides meta prompt formatting per example.

## Training (Windows PowerShell)

1. Environment & dependencies:
```powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

2. Ensure a task model is available:
   - Ollama: install, run `ollama serve`, pull a model (e.g. `ollama pull gemma3:4b`).
   - Or use HF local pipeline with `--task_backend hf`.

3. Run PPO training (toy example):
```powershell
python finetune.py ^
  --dataset data/train.jsonl ^
  --output_dir runs/rewriter-ppo ^
  --base_model gpt2 ^
  --task_backend ollama ^
  --task_model gemma3:4b ^
  --max_steps 50 ^
  --learning_rate 5e-6 ^
  --kl_coef 0.02
```

Internally this constructs `PPOConfig(learning_rate=5e-6, kl_coef=0.02, ...)` and performs rollouts+updates.

## Using the Notebook (`run.ipynb`)

The notebook cell now instantiates:
```python
trainer = PRewritePPOTrainer(
    base_model=base_model_local,
    evaluator=evaluator,
    meta_family="generic",
    log_dir="runs/PRrewrite",
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_ppo_epochs=1,
    kl_coef=0.02,
    seed=42,
)
```
No `PPOConfig` object is passed—arguments are folded into one automatically.

## Inference (basic generation)

```powershell
python inference.py --model_dir runs/rewriter-ppo --instruction "Improve reasoning for addition" --strategy I
```

Extend `inference.py` to implement PRewrite-S (search over K rewrites) scoring them via dev set examples and picking the highest-reward one.

## Reward Metrics

Selected automatically by `compute_reward`:
- math: numeric match (last number extraction)
- qa: token-level F1
- classification / other: exact match
- optional BLEU via `--reward_mode bleu`

## PPO Hyperparameters (key)
- `learning_rate`: optimizer LR
- `kl_coef`: KL penalty between policy and reference
- `num_ppo_epochs`: PPO epochs per batch
- `gradient_accumulation_steps`: accumulation for memory-saving
- `per_device_train_batch_size`: effective batch per device

Tune these for larger models; defaults are for quick smoke tests.

## Troubleshooting
- Missing pad token → wrapper sets pad to EOS.
- Out of memory → lower batch size or increase accumulation steps.
- Sparse rewards → consider `whiten_rewards=True`.
- Slow task model responses → cache evaluator outputs.

## Next Extensions
- Integrate learned reward model (replace ZeroRewardAdapter).
- Add PEFT/LoRA (`peft_config`) for efficient fine-tuning.
- Implement dev-set stopping criteria.

## License
Example code under permissive terms; consult original paper licensing for content reuse.