# PRewrite-style PPO Training with TRL

This repo sketches a minimal implementation of the PRewrite algorithm (Prompt Rewriting with Reinforcement Learning) using Hugging Face TRL's PPOTrainer.

- Paper: PRewrite — https://arxiv.org/abs/2401.08189 (PDF: https://arxiv.org/pdf/2401.08189)
- TRL docs (v0.25.0): https://huggingface.co/docs/trl/v0.25.0/en/ppo_trainer#trl.PPOTrainer

## What’s included

- `datasets.py` — JSONL-backed dataset utilities and meta-prompt helpers
- `finetune.py` — PPO training loop for a rewriter policy (LLM) that outputs rewritten instructions
- `inference.py` — Run PRewrite-I (greedy) or PRewrite-S (search) against the finetuned rewriter
- `requirements.txt` — Python dependencies

## How it maps to the paper

- Rewriter LLM (policy): any HF CausalLM you choose via `--base_model` (PaLM2 in paper; use your preferred open model)
- Task LLM (frozen, black-box):
  - `--task_backend ollama --task_model <name>` — uses local Ollama HTTP API to query a frozen model at temperature 0
  - `--task_backend hf --task_model <hf-id>` — uses a HF pipeline locally (optional fallback)
- Reward: compares the task LLM’s output to ground-truth using a simple metric (F1 for QA/Math; EM for classification)
- RL: PPO with KL constraint (via TRL PPOTrainer)
- Inference: PRewrite-I (greedy) and PRewrite-S (search over K candidates with optional dev-set scoring via Ollama)

## Quickstart (Windows PowerShell)

1) Create and activate a Python environment (recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies

```powershell
pip install -r requirements.txt
```

3) Prepare a small training JSONL (one object per line):

```json
{"instruction": "Answer the question", "input": "Who is the president of the United States?", "output": "Joe Biden", "task": "qa"}
{"instruction": "Classify the sentiment", "input": "This was awesome!", "output": "positive", "task": "classification"}
```

Save it as `data/train.jsonl`.

4) Ensure a frozen task LLM is available
- Ollama: install and run `ollama serve` (https://ollama.com/), pull a model (e.g., `ollama pull llama3`) and note its name.

5) Train the rewriter with PPO (toy demo config)

```powershell
python finetune.py --dataset data/train.jsonl --output_dir runs/rewriter-ppo --base_model gpt2 --task_backend ollama --task_model llama3:latest --max_steps 50
```

Notes:
- `gpt2` is a tiny demo model. For quality, use an instruction-tuned model that fits your hardware.
- PPO settings are intentionally small; increase for real runs.

6) Run inference (PRewrite-I)

```powershell
python inference.py --model_dir runs/rewriter-ppo --instruction "Answer the question" --strategy I
```

7) Run search (PRewrite-S) with optional dev-set scoring via Ollama

```powershell
python inference.py --model_dir runs/rewriter-ppo --instruction "Answer the question" --strategy S --K 10 --dev_path data/dev.jsonl --task_model llama3:latest
```

## Data format

`datasets.py` expects JSONL with fields:
- `instruction`: the under-optimized instruction t
- `input`: the task input x (text)
- `output`: the ground truth y
- `task` (optional): one of `qa`, `classification`, `math` (default `qa`)
- `template` (optional): overrides default template for the task

## Implementation notes & tips

- Rewards: paper finds the final task metric works; F1 tends to be more stable; perplexity can be harmful unless combined with F1 (see Appendix D).
- Temperatures: during RL, policy/value use temperature=1.0 for exploration; task LLM is queried at temperature=0.
- Dev-set convergence: stop training based on dev set performance (not implemented here; you can extend `finetune.py`).
- Cost/latency: PPO with a black-box task LLM can be API-expensive. Cache task LLM outputs if possible.
- Ollama deploy: If you want to serve the finetuned rewriter via Ollama, you’ll need to convert and package the HF model into an Ollama model (GGUF + Modelfile). That conversion is outside the scope of this minimal example.

## Troubleshooting

- Transformers / Torch versions: use `pip install --upgrade pip` if you hit resolution issues.
- GPU not used: add `accelerate config` and launch with `accelerate launch finetune.py ...` for multi-GPU.
- Long generation: lower `max_new_tokens`.

## License

This code is provided as a minimal example. Paper content is CC BY 4.0 (see arXiv page).