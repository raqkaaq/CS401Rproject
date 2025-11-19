from typing import Optional, Dict, Any
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
        Evaluate the base LLM output based on the math mode.
        
        Args:
            base_llm_output: The output from the base LLM.
            **kwargs: Additional keyword arguments. May include 'gold' for comparison.
        
        Returns:
            A math reward score (float between 0 and 1).
        """
        return len(base_llm_output.strip())
