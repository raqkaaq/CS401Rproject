from typing import Optional, Dict, Any, List
from trl.rewards import accuracy_reward
from .base_evalator import BaseEvaluator


class EasyMathEvaluator(BaseEvaluator):
    """
    An evaluator for GSM8K math problems.
    Uses accuracy_reward to compare model outputs with ground truth answers.
    """
    
    def __init__(self, model: str, client=None, temperature: float = 0.0, max_tokens: int = 256,
                 prefer_client: str = "auto"):
        """
        Initialize the EasyMath evaluator for GSM8K.
        
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
        Evaluate the base LLM output (not used in reward_function, but required by base class).
        
        Args:
            base_llm_output: The output from the base LLM
            **kwargs: Additional keyword arguments
            
        Returns:
            Reward score (float)
        """
        # This method is not used directly for GSM8K evaluation
        # The reward_function method handles the evaluation instead
        pass

    def reward_function(self, rewritten_prompts: List[List[Dict[str, str]]], **kwargs) -> List[float]:
        """
        Reward function to be used by the GRPOTrainer.
        
        For GSM8K, this function:
        1. Takes rewritten prompts and passes them to the inference model
        2. Compares the generated answers with the ground truth using accuracy_reward
        3. Returns a list of reward scores

        The kwargs are generally the other information in the dataset excluding the "prompt" key.
        For GSM8K, we expect "solution" or "answer" field containing the ground truth.

        Args:
            rewritten_prompts: List of rewritten prompts. Can be either:
                              - List of strings (from finetune.py wrapper)
                              - List of lists containing dicts with 'role' and 'content' keys.
            **kwargs: Additional keyword arguments passed to the reward function.
                     Should contain "solution" or "answer" field with ground truth.

        Returns:
            List of reward scores (floats).
        """
        # Get the solution/answer from kwargs (parser provides "solution" field)
        solution = kwargs.get("solution", kwargs.get("answer", None))
        return 0.0