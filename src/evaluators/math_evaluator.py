from typing import Optional, Dict, Any, List
from trl.rewards import accuracy_reward
from .base_evalator import BaseEvaluator


class MathEvaluator(BaseEvaluator):
    """
    A simple math evaluator for math problems.
    """
    
    def __init__(self, model: str, client=None, temperature: float = 0.0, max_tokens: int = 256):
        """
        Initialize the math evaluator.
        
        Args:
            model: Model name to use
            client: Optional client (defaults to OllamaClient if None)
            temperature: Temperature for generation
            max_tokens: Maximum tokens for generation
        """
        super().__init__(model, client, temperature, max_tokens)

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
            completions: List of completions, where each completion is a list containing a dict with 'role' and 'content' keys.
                        Format: [[{'role': 'assistant', 'content': '...'}], ...]
            **kwargs: Additional keyword arguments passed to the reward function.

        Returns:
            List of reward scores (floats).
        """
        solution = kwargs.get("solution", None)


        # Format completions as a list of dictionaries with the role and content
        formatted_completions = []
        rewritten_prompt_strings = [rewritten_prompt for rewritten_prompt in rewritten_prompts]
        for rewritten_prompt_string in rewritten_prompt_strings:
            base_llm_output = self.pass_to_inference(rewritten_prompt_string, **kwargs)
            print("base_llm_output: ", base_llm_output)
            formatted_completions.append({"role": "assistant", "content": base_llm_output})
        rewards = accuracy_reward(formatted_completions, solution)
        return rewards
