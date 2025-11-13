"""finetune.py

PPO-based finetuning for PRewrite instruction rewriting using TRL v0.25.0.

Paper: PRewrite (arXiv:2401.08189)
This file provides a fresh, minimal wrapper that builds PPOConfig internally from kwargs.
"""
from __future__ import annotations

import argparse
import os
from typing import Optional, Any, List, Tuple

from transformers import AutoTokenizer
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from copy import deepcopy

from load_datasets import (
    PRewriteDataset,
    RewriteExample
)

from inference import OllamaClient, HFClient
import re

# ---------------- Reward helpers -----------------
# Calculates the reward for each prediction based on the example and the chosen reward mode.
# Supports exact match, token-level F1, BLEU score, and math answer matching.
class RewardComputer:
    def __init__(self, mode: str = "auto") -> None:
        self.mode = mode.lower() if mode else "auto"

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
    def compute_reward(self, example: RewriteExample, pred: str, mode: str = "auto") -> float:
        m = (mode or "auto").lower()
        gold = example.output
        if m=="f1": return self.token_f1(pred, gold)
        if m=="bleu": return self.bleu_score(pred, gold)
        if m=="math": return self.math_reward(pred, gold)
        if m=="exact": return self.exact_match(pred, gold)
        if getattr(example,"task",None)=="math": return self.math_reward(pred, gold)
        if getattr(example,"task",None)=="qa": return self.token_f1(pred, gold)
        return self.exact_match(pred, gold)


# ---------------- Evaluators -----------------
# this is how the Trainer should interact with the reward computer, it calls the evaluator, then calculates the reward
class BaseTaskEvaluator:
    def __init__(self, reward_mode: str = "auto") -> None: 
        self.reward_mode = reward_mode
        self.reward_computer = RewardComputer(reward_mode)
    def generate(self, prompt: str, max_tokens: int = 256) -> str: raise NotImplementedError
    def score(self, example: RewriteExample, rewritten_instruction: str) -> float:
        task_prompt = example.build_task_prompt(rewritten_instruction)
        out = self.generate(task_prompt, max_tokens=256)
        return self.reward_computer.compute_reward(example, out, self.reward_mode)

class OllamaTaskEvaluator(BaseTaskEvaluator):
    def __init__(self, model: str, client: OllamaClient, reward_mode: str = "auto"):
        super().__init__(reward_mode)
        self.model=model
        self.client=client
        self.client.warmup_model(model)
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 256) -> str:
        return self.client.generate(self.model, prompt, temperature=temperature, max_tokens=max_tokens)

class HFTaskEvaluator(BaseTaskEvaluator):
    def __init__(self, model: str, client: HFClient, reward_mode: str = "auto"):
        super().__init__(reward_mode)
        self.model=model
        self.client=client
        self.client.warmup_model(model)
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        return self.client.generate(self.model, prompt, max_new_tokens=max_tokens)

# ---------------- Reward Model -----------------
# Reward model that uses the evaluator to compute rewards for PPO training
class RewardModel(nn.Module):
    def __init__(self, evaluator: BaseTaskEvaluator, reward_mode: str = "auto", data: List[RewriteExample] = [], device: str = "cuda"):
        super().__init__()
        self.evaluator = evaluator
        self.reward_mode = reward_mode
        self.data = data
        self.device = device

    def forward(self, preds: List[str]) -> torch.Tensor:
        rewards = []
        for pred, example in zip(preds, self.data):
            reward = self.evaluator.score(example, pred, self.reward_mode)
            rewards.append(reward)
        return torch.tensor(rewards, dtype=torch.float32).to(self.device)

# ---------------- PPO Wrapper -----------------
#I think this is buggy, havent gotten around to fix it yet.
class PRewriteTrainer:
    def __init__(self, tokenizer, model, ref_model, dataset, task: str = "auto"):
        self.reward_model = RewardModel(RewardComputer(mode=task), data=dataset)
        ppo_config = PPOConfig(
            exp_name="prewrite_finetune",
            num_ppo_epochs=1,
            gamma=1.0,
            lam=0.95,
            whiten_rewards=True,
            cliprange_value=0.2,
            vf_coef=0.1,
        )
        optim = torch.optim.AdamW(
            model.parameters(),
            lr=1e-5
        )
        sched = torch.optim.lr_scheduler.LinearLR(
            optim, start_factor=1.0,
            end_factor=0.0,
            total_iters=1000
        )
        self.ppo_trainer = PPOTrainer(
            args=ppo_config,
            model=model,
            ref_model=ref_model,
            train_dataset=dataset.train,
            eval_dataset=dataset.val,
            optimizers=(optim, sched)
        )
    def train(self):
        self.ppo_trainer.train()
    