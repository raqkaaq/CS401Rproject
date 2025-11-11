"""finetune.py — Class-based PPO finetuning for PRewrite rewriter using TRL.

References:
- PRewrite: https://arxiv.org/pdf/2401.08189
- TRL PPOTrainer v0.25.0: https://huggingface.co/docs/trl/v0.25.0/en/ppo_trainer#trl.PPOTrainer
"""
from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List

from transformers import AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

from datasets import (
    JsonlRewriteDataset,
    RewriteExample,
    build_meta_prompt,
    format_rewriter_query,
)


def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if _norm(pred) == _norm(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    p = _norm(pred).split()
    g = _norm(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = 0
    g_counts = {}
    for t in g:
        g_counts[t] = g_counts.get(t, 0) + 1
    for t in p:
        if g_counts.get(t, 0) > 0:
            common += 1
            g_counts[t] -= 1
    precision = common / max(len(p), 1)
    recall = common / max(len(g), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class BaseTaskEvaluator(ABC):
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 256) -> str:
        ...

    def score(self, example: RewriteExample, rewritten_instruction: str) -> float:
        prompt = example.build_task_prompt(rewritten_instruction)
        output = self.generate(prompt, temperature=0.0)
        if example.task in {"qa", "math"}:
            return token_f1(output, example.output)
        return exact_match(output, example.output)


class OllamaTaskEvaluator(BaseTaskEvaluator):
    def __init__(self, model: str, host: str = "http://localhost:11434"):
        self.model = model
        self.base_url = host.rstrip("/")
        try:
            import requests  # noqa: F401
        except Exception as e:
            raise RuntimeError("Please 'pip install requests' to use the Ollama evaluator.") from e

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 256) -> str:
        import requests
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": float(temperature), "num_predict": int(max_tokens)},
        }
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")


class HFTaskEvaluator(BaseTaskEvaluator):
    def __init__(self, model_name: str, device: Optional[str] = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipe = pipeline(
            "text-generation",
            model=AutoModelForCausalLM.from_pretrained(model_name),
            tokenizer=self.tokenizer,
            device_map="auto" if device is None else None,
        )

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 256) -> str:
        out = self.pipe(prompt, do_sample=temperature > 0.0, max_new_tokens=max_tokens, temperature=temperature)
        return out[0]["generated_text"][len(prompt) :]


class PRewritePPOTrainer:
    def __init__(
        self,
        base_model: str,
        evaluator: BaseTaskEvaluator,
        ppo_config: PPOConfig,
        meta_family: str = "generic",
        log_dir: Optional[str] = None,
    ) -> None:
        self.evaluator = evaluator
        self.meta_prompt = build_meta_prompt(meta_family)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
        self.ppo = PPOTrainer(config=ppo_config, model=self.model, tokenizer=self.tokenizer)
        # tensorboard writer
        try:
            self.writer = SummaryWriter(log_dir)
        except Exception:
            self.writer = None
        self.global_step = 0

    def _policy_rollout(self, initial_instruction: str, max_new_tokens: int = 64) -> tuple[str, list[int], list[int]]:
        query = format_rewriter_query(self.meta_prompt, initial_instruction)
        q = self.tokenizer(query, return_tensors="pt").to(self.ppo.accelerator.device)
        gen = self.ppo.generate(
            q["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        response_ids = gen[:, q["input_ids"].shape[1]:]
        rewritten = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0].strip()
        return rewritten, q["input_ids"][0].tolist(), response_ids[0].tolist()

    def step_on_example(self, ex: RewriteExample) -> float:
        rewritten, query_ids, response_ids = self._policy_rollout(ex.instruction)
        reward = self.evaluator.score(ex, rewritten)
        import torch
        q = torch.tensor(query_ids, device=self.ppo.accelerator.device)
        r = torch.tensor(response_ids, device=self.ppo.accelerator.device)
        self.ppo.step([q], [r], [float(reward)])
        # log to tensorboard if available
        self.global_step += 1
        if getattr(self, "writer", None) is not None:
            try:
                self.writer.add_scalar("train/reward", float(reward), self.global_step)
            except Exception:
                pass
        return float(reward)

    def train(self, dataset: JsonlRewriteDataset, max_steps: int = 200, log_every: int = 10) -> None:
        step = 0
        while step < max_steps:
            for ex in dataset.iter():
                if step >= max_steps:
                    break
                reward = self.step_on_example(ex)
                step += 1
                if step % log_every == 0:
                    self.ppo.accelerator.print(f"Step {step}/{max_steps} | reward={reward:.3f}")

    def save(self, output_dir: str) -> None:
        self.ppo.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.ppo.accelerator.print(f"Saved rewriter policy to: {output_dir}")
        if getattr(self, "writer", None) is not None:
            try:
                self.writer.flush()
                self.writer.close()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(description="PRewrite PPO finetuning (class-based)")
    ap.add_argument("--dataset", required=True, help="Path to JSONL training data")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--base_model", default="gpt2")
    ap.add_argument("--task_backend", choices=["ollama", "hf"], default="ollama")
    ap.add_argument("--task_model", required=True, help="Ollama name (e.g., llama3:latest) or HF id for --task_backend hf")
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--learning_rate", type=float, default=5e-6)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--mini_batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--target_kl", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--meta_family", default="generic")
    ap.add_argument("--log_dir", default="runs/PRrewrite", help="TensorBoard log directory")
    args = ap.parse_args()

    evaluator: BaseTaskEvaluator
    if args.task_backend == "ollama":
        evaluator = OllamaTaskEvaluator(model=args.task_model)
    else:
        evaluator = HFTaskEvaluator(model_name=args.task_model)

    ppo_cfg = PPOConfig(
        model_name=args.base_model,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        target_kl=args.target_kl,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
    )

    trainer = PRewritePPOTrainer(
        base_model=args.base_model,
        evaluator=evaluator,
        ppo_config=ppo_cfg,
        meta_family=args.meta_family,
        log_dir=args.log_dir,
    )

    ds = JsonlRewriteDataset(args.dataset)
    trainer.train(ds, max_steps=args.max_steps)
    trainer.save(args.output_dir)


if __name__ == "__main__":
    main()
