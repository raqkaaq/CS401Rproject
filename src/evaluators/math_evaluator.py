from typing import Optional, Dict, Any, List
from trl.rewards import accuracy_reward
from .base_evalator import BaseEvaluator


class MathEvaluator(BaseEvaluator):
    """
    A simple math evaluator for math problems.
    """
    
    def __init__(self, model: str, client=None, temperature: float = 0.0, max_tokens: int = 256,
                 prefer_client: str = "auto"):
        """
        Initialize the math evaluator.
        
        Args:
            model: Model name to use
            client: Optional client (defaults to auto-detect if None)
            temperature: Temperature for generation
            max_tokens: Maximum tokens for generation
            prefer_client: Client preference ("auto", "ollama", or "hf")
        """
        super().__init__(model, client, temperature, max_tokens, prefer_client=prefer_client)

    def evaluate(self, base_llm_output: str, **kwargs) -> float:
        """
            Unused as it this is a special case for the math evaluator.
        """
        pass

    def reward_function(self, rewritten_prompts: List[List[Dict[str, str]]], **kwargs) -> List[float]:
        """
        Reward function to be used by the GRPOTrainer.

        The kwargs are generally the other information in the dataset excluding the "prompt" key.

        Args:
            rewritten_prompts: List of rewritten prompts. Can be either:
                              - List of strings (from finetune.py wrapper)
                              - List of lists containing dicts with 'role' and 'content' keys.
            **kwargs: Additional keyword arguments passed to the reward function.

        Returns:
            List of reward scores (floats).
        """
        solution = kwargs.get("solution", None)

        # Extract all prompt strings for batch processing
        prompt_strings = []
        for rewritten_prompt in rewritten_prompts:
            # Handle both string format and list-of-dicts format
            if isinstance(rewritten_prompt, str):
                prompt_strings.append(rewritten_prompt)
            elif isinstance(rewritten_prompt, list) and len(rewritten_prompt) > 0:
                # Extract content from dict format
                if isinstance(rewritten_prompt[0], dict) and "content" in rewritten_prompt[0]:
                    prompt_strings.append(rewritten_prompt[0]["content"])
                else:
                    prompt_strings.append(str(rewritten_prompt[0]))
            else:
                prompt_strings.append(str(rewritten_prompt))

        # Batch inference - much faster than sequential calls
        # Filter out 'prompts' from kwargs to avoid conflict with the prompts parameter
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'prompts'}
        base_llm_outputs = self.pass_to_inference_batch(prompt_strings, **filtered_kwargs)
        
        # Format completions for accuracy_reward: it expects [[{"content": "..."}], ...]
        formatted_completions = [[{"content": output}] for output in base_llm_outputs]
        
        rewards = accuracy_reward(formatted_completions, solution)
        print("Rewards: ", rewards)
        return rewards
