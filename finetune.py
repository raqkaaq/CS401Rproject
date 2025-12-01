"""finetune.py

PPO-based finetuning for PRewrite instruction rewriting using TRL v0.25.0.

Paper: PRewrite (arXiv:2401.08189)
This file provides a fresh, minimal wrapper that builds PPOConfig internally from kwargs.
"""
from __future__ import annotations
import os
import json
from transformers import AutoTokenizer
from typing import Optional, List, Tuple, Any
from abc import ABC, abstractmethod
from torch import nn
import torch
from transformers import GenerationConfig, AutoModelForCausalLM
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead, create_reference_model

from inference import OllamaClient, HFClient, download_model
from dataset_parsers import get_parser  # NEW
import re

# ---------------- Reward helpers -----------------
# Calculates the reward for each prediction based on the example and the chosen reward mode.
# Supports exact match, token-level F1, BLEU score, and math answer matching.
class RewardComputer:
    def __init__(self, mode: str = "exact") -> None:
        self.mode = mode.lower() if mode else "exact"

    @staticmethod
    def _norm(s: str) -> str:
        return " ".join(s.strip().lower().split())

    # Exact match reward: 1 if normalized strings match, else 0, probably not useful
    def exact_match(self, pred: str, gold: str) -> float:
        return 1.0 if self._norm(pred) == self._norm(gold) else 0.0

    #f1 is always normalized
    def token_f1(self, pred: str, gold: str) -> float:
        p = self._norm(pred).split(); g = self._norm(gold).split()
        if not p and not g: return 1.0
        if not p or not g: return 0.0
        common = 0; g_counts = {}
        for t in g: g_counts[t] = g_counts.get(t, 0) + 1
        for t in p:
            if g_counts.get(t, 0) > 0:
                common += 1; g_counts[t] -= 1
        precision = common / max(len(p),1); recall = common / max(len(g),1)
        if precision + recall == 0: return 0.0
        return 2*precision*recall/(precision+recall)

    @staticmethod
    def _extract_last_number(s: str) -> Optional[str]:
        if not isinstance(s,str): return None
        # Check for "#### number" pattern first, all the GSM8K answers are in this format
        m = re.search(r"####\s*([-+]?[0-9]+\.?[0-9]*)", s)
        if m: return m.group(1)
        nums = re.findall(r"[-+]?[0-9]+\.?[0-9]*", s)
        return nums[-1] if nums  else None

    #reward is 1 if the last number matches, else 0
    def math_reward(self, pred: str, gold: str) -> float:
        pnum = self._extract_last_number(pred); gnum = self._extract_last_number(gold)
        if pnum is not None and gnum is not None:
            try:
                if "." in pnum or "." in gnum: return 1.0 if float(pnum)==float(gnum) else 0.0
                return 1.0 if int(pnum)==int(gnum) else 0.0
            except Exception:
                return 1.0 if pnum==gnum else 0.0
        return self.exact_match(pred, gold)

    @staticmethod
    def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    #Vibe Coded bleu_score, already normalized
    def bleu_score(self, pred: str, gold: str, max_n: int = 4, smooth: bool = True) -> float:
        from collections import Counter
        import math
        if not isinstance(pred, str) or not isinstance(gold, str):
            return 0.0
        ref = gold.strip().split()
        hyp = pred.strip().split()
        if not ref or not hyp:
            return 0.0
        precisions = []
        for n in range(1, max_n+1):
            r_ngrams = Counter(self.ngrams(ref, n))
            h_ngrams = Counter(self.ngrams(hyp, n))
            if not h_ngrams:
                precisions.append(0.0)
                continue
            overlap = sum(min(cnt, r_ngrams.get(ng, 0)) for ng, cnt in h_ngrams.items())
            if smooth and overlap == 0:
                overlap += 1
                denom = sum(h_ngrams.values()) + 1
            else:
                denom = sum(h_ngrams.values())
            precisions.append(overlap / denom)
        geo = 0.0 if min(precisions) == 0 else math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        ref_len = len(ref)
        hyp_len = len(hyp)
        if hyp_len == 0:
            bp = 0.0
        elif hyp_len > ref_len:
            bp = 1.0
        else:
            bp = math.exp(1 - float(ref_len) / float(hyp_len))
        return float(bp * geo)

    #this is the main reward function to calculate the value of a particular prediction
    def compute_reward(self, example: dict, pred: str) -> float:
        m = self.mode.lower() if self.mode else "exact"
        # Handle both dict (Hugging Face dataset) and dict
        if isinstance(example, dict):
            gold = example.get("answer", example.get("output", ""))
            task = example.get("task", None)
        else:
            gold = example.output
            task = getattr(example, "task", None)
        if m=="f1": return self.token_f1(pred, gold)
        if m=="bleu": return self.bleu_score(pred, gold)
        if m=="math": return self.math_reward(pred, gold)
        if m=="exact": return self.exact_match(pred, gold)
        if task=="math": return self.math_reward(pred, gold)
        if task=="qa": return self.token_f1(pred, gold)
        return self.exact_match(pred, gold)


# ---------------- Evaluators -----------------
# this is how the Trainer should interact with the reward computer, it calls the evaluator, then calculates the reward
class BaseTaskEvaluator:
    def __init__(self, reward_mode: str = "auto") -> None: 
        self.reward_mode = reward_mode
        self.reward_computer = RewardComputer(reward_mode)
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 1024) -> str: raise NotImplementedError
    def score(self, example: dict, rewritten_instruction: str) -> float:
        if isinstance(example, dict):
            question = example.get("question") or example.get("instruction", "")
            inp = example.get("input", "")
            if inp and len(str(inp).strip()) > 0:
                task_prompt = f"{rewritten_instruction}\n\nQuestion:\n{question}\n\nInput:\n{inp}"
            else:
                task_prompt = f"{rewritten_instruction}\n\nQuestion:\n{question}"
        else:
            task_prompt = example.build_task_prompt(rewritten_instruction)
        out = self.generate(task_prompt, max_tokens=1024)
        return self.reward_computer.compute_reward(example, out)

class OllamaTaskEvaluator(BaseTaskEvaluator):
    def __init__(self, model: str, client: OllamaClient, reward_mode: str = "auto"):
        super().__init__(reward_mode)
        self.model=model
        self.client=client
        self.client.warmup_model(model)
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        return self.client.generate(self.model, prompt, temperature=temperature, max_tokens=max_tokens)

class HFTaskEvaluator(BaseTaskEvaluator):
    def __init__(self, model: str, client: HFClient, reward_mode: str = "auto"):
        super().__init__(reward_mode)
        self.model=model
        self.client=client
        self.client.warmup_model(model)
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        return self.client.generate(self.model, prompt, max_new_tokens=max_tokens)

# ---------------- Reward Model -----------------
# Reward model that uses the evaluator to compute rewards for PPO training
class RewardModel(nn.Module):
    def __init__(self, evaluator: BaseTaskEvaluator, reward_mode: str = "auto", data: Optional[List[dict]] = None):
        super().__init__()
        self.evaluator = evaluator
        self.reward_mode = reward_mode
        self.data = data if data is not None else None

    def forward(self, preds: List[str]) -> torch.Tensor:
        rewards = []
        for pred, example in zip(preds, self.data):
            reward = self.evaluator.score(example, pred)
            rewards.append(reward)
        return torch.tensor(rewards, dtype=torch.float32).to("cpu")

# ---------------- PPO Wrapper -----------------
# I think this is buggy, havent gotten around to fix it yet.
class PRewriteTrainer:
    def __init__(self, model_id:  str, dataset: str, evaluator_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
                 task: str = "exact", ollama: bool = True):
        if ollama:
            print("Using Ollama evaluator")
            self.evaluator = OllamaTaskEvaluator(evaluator_model, OllamaClient(), reward_mode=task)
        else:
            print("Using HF evaluator")
            self.evaluator = HFTaskEvaluator(evaluator_model, HFClient(), reward_mode=task)

        tokenizer, model, ref_model = self.load_from_hf(model_id)
        if model is None or ref_model is None:
            raise RuntimeError("Failed to load model or reference model")
        self.tokenizer = tokenizer

        # Parse & tokenize dataset via parser factory
        parser = get_parser(dataset, self.tokenizer)
        tokenized_train_ds, tokenized_eval_ds, raw_train_examples = parser.prepare()

        # Reward model uses raw examples list
        self.reward_model = RewardModel(self.evaluator, data=raw_train_examples)

        ppo_config = PPOConfig(
            exp_name="prewrite_finetune",
            num_ppo_epochs=1,
            gamma=1.0,
            lam=0.95,
            whiten_rewards=True,
            cliprange_value=0.2,
            vf_coef=0.1,
        )
        optim = torch.optim.AdamW(model.parameters(), lr=1e-5)
        sched = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0.0, total_iters=1000)

        self.ppo_trainer = PPOTrainer(
            args=ppo_config,
            ref_model=None,
            value_model=model,
            processing_class=self.tokenizer,
            model=model,
            reward_model=self.reward_model,
            train_dataset=tokenized_train_ds,
            eval_dataset=tokenized_eval_ds,
            optimizers=(optim, sched)
        )

    def train(self):
        print("Starting PPO training...")
        self.ppo_trainer.train()

    @staticmethod
    def load_from_hf(model_id: str) -> Tuple[Any, Any, Any]:
        """Load model & tokenizer from Hugging Face hub or local path; optionally persist locally.

        Returns (tokenizer, model, ref_model) where model is an AutoModelForCausalLMWithValueHead instance.
        Ref_model is a frozen copy of the base model without value head.
        This is useful for baseline SFT loading prior to PPO wrapping.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        #check if the model is downloaded locally already, recall models are saved as models/{owner}/{model_name}
        local_dir = "models/" + model_id
        if not os.path.exists(local_dir):
            print(f"Downloading model {model_id} from Hugging Face hub...")
            if "/" in model_id:
                owner, model_name = model_id.split("/")
                local_dir = download_model(model_id, f"models/{owner}/{model_name}")
            else:
                local_dir = download_model(model_id, f"models/{model_id}")
        print(f"Loading model from local directory: {local_dir}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(local_dir, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            # Set left padding for decoder-only models (required for correct generation)
            tokenizer.padding_side = 'left'
            base_config = GenerationConfig.from_pretrained(local_dir)
            
            # Load the policy model (with value head for PPO)
            model = AutoModelForCausalLMWithValueHead.from_pretrained(local_dir)
            ref_model = create_reference_model(model)
            
            # Move both models to target device
            model = model.to(device)
            ref_model = ref_model.to(device)
            return tokenizer, model, ref_model
        except Exception as e:
            raise RuntimeError(f"Error loading model {model_id} from {local_dir}: {e}") from e
