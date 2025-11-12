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
from torch.utils.tensorboard import SummaryWriter
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

from load_datasets import (
    PRewriteDataset,
    RewriteExample,
    build_meta_prompt,
    format_rewriter_query,
)

from inference import OllamaClient
import re


# ---------------- HF loading helper -----------------
# This loads model_id from Hugging face and saves it locally.
def load_from_hf(model_id: str, save_to: Optional[str] = None, trust_remote_code: bool = True):
    """Load model & tokenizer from Hugging Face hub or local path; optionally persist locally.

    Returns (tokenizer, model) where model is an AutoModelForCausalLM (no value head).
    This is useful for baseline SFT loading prior to PPO wrapping.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if save_to and os.path.isdir(save_to):
        try:
            tok = AutoTokenizer.from_pretrained(save_to, trust_remote_code=trust_remote_code)
            mdl = AutoModelForCausalLM.from_pretrained(save_to, trust_remote_code=trust_remote_code)
            return tok, mdl
        except Exception:
            pass  # fallback to remote
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    mdl = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if save_to:
        os.makedirs(save_to, exist_ok=True)
        tok.save_pretrained(save_to)
        mdl.save_pretrained(save_to)
    return tok, mdl

# ---------------- Reward helpers -----------------
# Calculates the reward for each prediction based on the example and the chosen reward mode.
# Supports exact match, token-level F1, BLEU score, and math answer matching.
class RewardComputer:
    def __init__(self, mode: str = "auto"):
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
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 256) -> str: raise NotImplementedError
    def score(self, example: RewriteExample, rewritten_instruction: str) -> float:
        task_prompt = example.build_task_prompt(rewritten_instruction)
        out = self.generate(task_prompt, temperature=0.0)
        return self.reward_computer.compute_reward(example, out, self.reward_mode)

class OllamaTaskEvaluator(BaseTaskEvaluator):
    def __init__(self, model: str, client: OllamaClient, reward_mode: str = "auto"):
        def __init__(self):
            super().__init__(reward_mode)
            self.model=model
            self.client=client
            self.client.warmup_model(model)
        def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 256) -> str:
            return self.client.generate(self.model, prompt, temperature=temperature, max_tokens=max_tokens)

# ---------------- PPO Wrapper -----------------
#I think this is buggy, havent gotten around to fix it yet.
class PRewritePPOTrainer:
    DEFAULT_PPO_ARGS = dict(
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_ppo_epochs=1,
        kl_coef=0.02,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=0.1,
        whiten_rewards=False,
        seed=42,
    )

    def __init__(self, base_model: str, dataset: PRewriteDataset, evaluator: BaseTaskEvaluator, meta_family: str = "generic", log_dir: Optional[str] = None,
                 load_dtype: Optional[str] = None, use_8bit: bool = False, **ppo_kwargs: Any) -> None:
        """Construct the PPO trainer.

        Parameters:
        - base_model: path or model id for the policy/value model with value head
        - dataset: PRewriteDataset (train split chosen by its constructor). Required up-front so we can
                   pass a real dataset length to TRL (no lazy init surprises).
        - evaluator: task evaluator instance providing reward signals
        - meta_family: which meta prompt family to use
        - log_dir: optional TensorBoard log directory
        - **ppo_kwargs: overrides for PPOConfig (kl_coef, learning_rate, etc.)
        """
        self.evaluator=evaluator
        self.dataset=dataset
        self.meta_prompt=build_meta_prompt(meta_family)
        cfg_dict={**self.DEFAULT_PPO_ARGS, **ppo_kwargs}
        self.ppo_config=PPOConfig(**cfg_dict)
        self.tokenizer=AutoTokenizer.from_pretrained(base_model, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token=self.tokenizer.eos_token
        # Configure memory-saving load options
        model_kwargs = {}
        if load_dtype:
            import torch
            dtype_map = {
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32,
            }
            model_kwargs["torch_dtype"] = dtype_map.get(load_dtype.lower(), torch.float16)
        if use_8bit:
            model_kwargs["load_in_8bit"] = True
        self.model=AutoModelForCausalLMWithValueHead.from_pretrained(base_model, **model_kwargs)
        if not hasattr(self.model, "generation_config"):
            self.model.generation_config=getattr(self.model,"config",None)
        # Some underlying architectures or value-head wrappers may not expose base_model_prefix,
        # but TRL/PPOTrainer relies on it for reference model creation and weight handling.
        if not hasattr(self.model, "base_model_prefix"):
            for cand in ["transformer", "model", "base_model"]:
                if hasattr(self.model, cand):
                    self.model.base_model_prefix = cand
                    break
            else:
                # Fallback: use generic name; TRL will treat entire model as base
                self.model.base_model_prefix = "model"
        # Ensure attributes PPOTrainer probes (model/pretrained_model) exist.
        # Try to discover an underlying submodule; if absent, reuse the wrapper itself.
        # Try to identify an underlying module to expose, but avoid loading a second full model.
        underlying = None
        for cand in ["transformer", "base_model", "model"]:
            if hasattr(self.model, cand):
                underlying = getattr(self.model, cand)
                break
        # Avoid creating self-referential attributes that cause recursion.
        if underlying is not None and underlying is not self.model:
            if not hasattr(self.model, "model"):
                self.model.model = underlying
            if not hasattr(self.model, "pretrained_model"):
                self.model.pretrained_model = underlying
        # Provide gradient checkpointing interface expected by TRL if missing.
        if not hasattr(self.model, "is_gradient_checkpointing"):
            # Reuse underlying flag if present, else default False
            base_flag = getattr(underlying, "is_gradient_checkpointing", False)
            self.model.is_gradient_checkpointing = base_flag
        if not hasattr(self.model, "gradient_checkpointing_enable"):
            def _gc_enable(gradient_checkpointing_kwargs=None):
                setattr(self.model, "is_gradient_checkpointing", True)
                # forward to underlying if it implements the method
                if hasattr(underlying, "gradient_checkpointing_enable"):
                    try:
                        underlying.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
                    except TypeError:
                        underlying.gradient_checkpointing_enable()
            self.model.gradient_checkpointing_enable = _gc_enable
        if not hasattr(self.model, "gradient_checkpointing_disable"):
            def _gc_disable():
                setattr(self.model, "is_gradient_checkpointing", False)
                if hasattr(underlying, "gradient_checkpointing_disable"):
                    underlying.gradient_checkpointing_disable()
            self.model.gradient_checkpointing_disable = _gc_disable
        import torch, torch.nn as nn
        class ZeroRewardAdapter(nn.Module):
            def forward(self, input_ids=None, **kwargs):
                # Minimal stub: TRL expects a value for reward model forward shape alignment.
                if input_ids is None:
                    return torch.zeros((1,1), device=self.model.device)
                return torch.zeros((input_ids.shape[0],1), device=self.model.device)
        self._reward_model=ZeroRewardAdapter()

        # Build a minimal dataset object that satisfies length; TRL still expects samples
        from torch.utils.data import Dataset
        df = self.dataset.get_split_df(self.dataset._requested_split)
        length = len(df) if df is not None else 0
        class _LengthOnlyDataset(Dataset):
            def __init__(self, length: int): self.length=length
            def __len__(self): return self.length
            def __getitem__(self, idx):
                import torch
                return {"input_ids": torch.tensor([], dtype=torch.long)}
        train_ds=_LengthOnlyDataset(length)
        # Let TRL create the reference model automatically; explicitly pass value_model
        # Provide a separate reference model to avoid internal cloning & recursion.
        from transformers import AutoModelForCausalLM
        ref_model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        self.ppo=PPOTrainer(
            args=self.ppo_config,
            processing_class=self.tokenizer,
            model=self.model,
            ref_model=ref_model,
            reward_model=self._reward_model,
            value_model=self.model,
            train_dataset=train_ds,
        )
        try:
            self.writer=SummaryWriter(log_dir) if log_dir else None
        except Exception:
            self.writer=None
        self.global_step=0

    # _ensure_ppo removed: initialization now always happens in __init__ with dataset

    def _generate_rewrite(self, instruction: str, max_new_tokens: int = 64) -> Tuple[str,List[int],List[int]]:
        query=format_rewriter_query(self.meta_prompt, instruction)
        import torch
        q=self.tokenizer(query, return_tensors="pt").to(self.ppo.accelerator.device)
        gen=self.ppo.generate(q["input_ids"], max_new_tokens=max_new_tokens, do_sample=True, temperature=1.0, pad_token_id=self.tokenizer.pad_token_id)
        resp_ids=gen[:, q["input_ids"].shape[1]:]
        rewritten=self.tokenizer.batch_decode(resp_ids, skip_special_tokens=True)[0].strip()
        return rewritten, q["input_ids"][0].tolist(), resp_ids[0].tolist()

    def step_on_example(self, ex: RewriteExample) -> float:
        rewritten, query_ids, response_ids = self._generate_rewrite(ex.instruction)
        reward=self.evaluator.score(ex, rewritten)
        import torch
        q=torch.tensor(query_ids, device=self.ppo.accelerator.device)
        r=torch.tensor(response_ids, device=self.ppo.accelerator.device)
        self.ppo.step([q],[r],[float(reward)])
        self.global_step+=1
        if self.writer:
            try: self.writer.add_scalar("train/reward", float(reward), self.global_step)
            except Exception: pass
        return float(reward)

    def train(self, max_steps: int = 200, log_every: int = 10) -> None:
        step=0
        while step < max_steps:
            for ex in self.dataset.iter():
                if step>=max_steps: break
                rew=self.step_on_example(ex); step+=1
                if step % log_every == 0:
                    try: self.ppo.accelerator.print(f"step {step}/{max_steps} reward={rew:.3f}")
                    except Exception: print(f"step {step}/{max_steps} reward={rew:.3f}")

    def save(self, output_dir: str) -> None:
        self.ppo.save_pretrained(output_dir); self.tokenizer.save_pretrained(output_dir)
        msg=f"Saved PPO policy to {output_dir}"
        try: self.ppo.accelerator.print(msg)
        except Exception: print(msg)
        if self.writer:
            try: self.writer.flush(); self.writer.close()
            except Exception: pass