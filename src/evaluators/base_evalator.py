from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from collections import Counter
import math
import torch

class BaseEvaluator(ABC):
    def __init__(self, model: str, client=None, temperature: float = 0.0, max_tokens: int = 512, 
                 prefer_client: str = "auto", evaluator_8bit: bool = False):
        """
        Initialize the BaseEvaluator.
        
        Args:
            model: Model identifier (Ollama format for OllamaClient, HF format for HFClient)
            client: Optional client instance. If None, will auto-detect based on prefer_client.
            temperature: Temperature for generation
            max_tokens: Maximum tokens for generation
            prefer_client: Preferred client type. Options:
                - "auto": Try Ollama first, fall back to HFClient if unavailable
                - "ollama": Force OllamaClient (will raise error if unavailable)
                - "hf": Force HFClient
            evaluator_8bit: If True, use 8-bit quantization for HFClient (reduces memory usage)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if client is None:
            from ..inference import OllamaClient, HFClient
            
            if prefer_client == "hf":
                # Force HFClient
                print("Using HFClient (forced)")
                # Use bf16 for GPU inference if available (faster and more memory efficient)
                torch_dtype = "bf16" if torch.cuda.is_available() else None
                if torch_dtype:
                    print(f"  Using {torch_dtype} precision for GPU inference")
                if evaluator_8bit:
                    print("  Using 8-bit quantization for evaluator model (reduces memory usage)")
                self.client = HFClient(torch_dtype=torch_dtype, load_in_8bit=evaluator_8bit)
                self.client.warmup_model(self.model)
            elif prefer_client == "ollama":
                # Force OllamaClient
                print("Using OllamaClient (forced)")
                self.client = OllamaClient()
                if not self.client.is_healthy():
                    raise RuntimeError("Ollama is not available but 'ollama' was specified as prefer_client")
                self.client.warmup_model(self.model)
            else:
                # Auto-detect: try Ollama first, fall back to HFClient
                print("Auto-detecting client...")
                ollama_client = OllamaClient()
                if ollama_client.is_healthy():
                    print("Ollama is available, using OllamaClient")
                    self.client = ollama_client
                    self.client.warmup_model(self.model)
                else:
                    print("Ollama is not available, falling back to HFClient")
                    # Use bf16 for GPU inference if available (faster and more memory efficient)
                    torch_dtype = "bf16" if torch.cuda.is_available() else None
                    if torch_dtype:
                        print(f"  Using {torch_dtype} precision for GPU inference")
                    if evaluator_8bit:
                        print("  Using 8-bit quantization for evaluator model (reduces memory usage)")
                    self.client = HFClient(torch_dtype=torch_dtype, load_in_8bit=evaluator_8bit)
                    self.client.warmup_model(self.model)
        else:
            self.client = client

    @abstractmethod
    def evaluate(self, base_llm_output: str, **kwargs) -> float:
        """
        Evaluate the base LLM output.

        Args:
            base_llm_output: The output from the base LLM.
            **kwargs: Additional keyword arguments passed to the evaluate function.

        Returns:
            The reward score (float).
        """
        raise NotImplementedError
    
    def pass_to_inference(self, prompt: str, **kwargs) -> str:
        """
        Pass a prompt to the inference client to get a response from the base LLM.

        Args:
            prompt: The prompt string to send to the model
            **kwargs: Additional keyword arguments (e.g., temperature, max_tokens, max_new_tokens)

        Returns:
            The generated response string from the base LLM
        """
        # Extract common generation parameters from kwargs
        # Check if client is OllamaClient or HFClient and call appropriate method
        client_type = type(self.client).__name__
        
        if client_type == 'OllamaClient':
            return self.client.generate(
                model=self.model,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        elif client_type == 'HFClient':
            return self.client.generate(
                model_id=self.model,
                prompt=prompt,
                num_beams=1,
                do_sample=True,
                repetition_penalty=1.15,
                max_new_tokens=self.max_tokens
            )
        else:
            # Fallback: try the base LLMClient interface
            return self.client.generate(model=self.model, prompt=prompt, temperature=self.temperature, max_tokens=self.max_tokens)
    
    def pass_to_inference_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Pass multiple prompts to the inference client in batch for faster processing.

        Args:
            prompts: List of prompt strings to send to the model
            **kwargs: Additional keyword arguments (e.g., temperature, max_tokens, max_new_tokens)

        Returns:
            List of generated response strings from the base LLM
        """
        # Check if client is OllamaClient or HFClient and call appropriate batch method
        client_type = type(self.client).__name__
        
        if client_type == 'OllamaClient':
            return self.client.generate_batch(
                model=self.model,
                prompts=prompts,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        elif client_type == 'HFClient':
            return self.client.generate_batch(
                model_id=self.model,
                prompts=prompts,
                num_beams=1,
                do_sample=True,
                repetition_penalty=1.15,
                max_new_tokens=self.max_tokens
            )
        else:
            # Fallback: call generate() multiple times (not batched, but works)
            return [self.pass_to_inference(prompt, **kwargs) for prompt in prompts]
    
    def reward_function(self, completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
        """
        Reward function to be used by the GRPOTrainer.

        The kwargs are generally the other information in the dataset excluding the "prompt" key.

        Args:
            completions: List of completions, where each completion is a list containing a dict with 'role' and 'content' keys.
                        Format: [[{'role': 'assistant', 'content': '...'}], ...]
            **kwargs: Additional keyword arguments passed to the reward function.

        Returns:
            List of reward scores (floats).
        """
        # Extract completion strings (handle both string and list-of-dicts formats)
        completion_strings = []
        for completion in completions:
            if isinstance(completion, str):
                completion_strings.append(completion)
            elif isinstance(completion, list) and len(completion) > 0:
                if isinstance(completion[0], dict) and "content" in completion[0]:
                    completion_strings.append(completion[0]["content"])
                else:
                    completion_strings.append(str(completion[0]))
            else:
                completion_strings.append(str(completion))
        
        rewards = []
        for i, completion_string in enumerate(completion_strings):
            base_llm_output = self.pass_to_inference(completion_string, **kwargs)
            
            # Log first base LLM output
            if i == 0:
                print("\n" + "="*80)
                print("FIRST BASE LLM OUTPUT (from inference model):")
                print("="*80)
                print(base_llm_output)
                print("="*80 + "\n")
            
            # Extract individual items from kwargs when they're lists (batched data)
            # This handles the case where GRPOTrainer passes batched kwargs
            individual_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, list) and len(value) > i:
                    individual_kwargs[key] = value[i]
                else:
                    individual_kwargs[key] = value
            
            reward = self.evaluate(base_llm_output, **individual_kwargs)
            rewards.append(reward)
        print("Rewards: ", rewards)
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