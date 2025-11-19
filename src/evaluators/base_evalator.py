from abc import ABC, abstractmethod
from typing import List, Tuple
from collections import Counter
import math

class BaseEvaluator(ABC):
    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def evaluate(self, base_llm_output: str, **kwargs) -> float:
        raise NotImplementedError
    
    def pass_to_inference(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError
    
    def reward_function(self, completions: List[str], **kwargs) -> List[float]:
        """
        Reward function to be used by the GRPOTrainer.

        The kwargs are generally the other information in the dataset excluding the "prompt" key.

        Args:
            completions: List of completions, where each completion is a string.
            **kwargs: Additional keyword arguments passed to the reward function.

        Returns:
            List of reward scores (floats).
        """
        
        rewards = []
        for completion in completions:
            base_llm_output = self.pass_to_inference(completion, **kwargs)
            reward = self.evaluate(base_llm_output, **kwargs)
            rewards.append(reward)
        return rewards
    
    def math_reward(self, pred: str, gold: str) -> float:
        pass

    def bleu_score(self, pred: str, gold: str, max_n: int = 4, smooth: bool = True) -> float:
        def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
            return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

        if not isinstance(pred, str) or not isinstance(gold, str):
            return 0.0
        ref = gold.strip().split()
        hyp = pred.strip().split()
        if not ref or not hyp:
            return 0.0
        precisions = []
        for n in range(1, max_n+1):
            r_ngrams = Counter(ngrams(ref, n))
            h_ngrams = Counter(ngrams(hyp, n))
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

    def token_f1(self, pred: str, gold: str) -> float:
        def _norm(s: str) -> str:
            return " ".join(s.strip().lower().split())

        p = _norm(pred).split(); g = _norm(gold).split()
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