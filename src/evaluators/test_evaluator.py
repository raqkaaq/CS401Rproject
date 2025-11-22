from typing import Optional, Dict, Any
from .base_evalator import BaseEvaluator


class TestEvaluator(BaseEvaluator):
    """
    A simple test evaluator for testing purposes.
    This evaluator can be used to test the evaluation pipeline without
    implementing complex evaluation logic.
    """
    
    def __init__(self, model: str, client=None, temperature: float = 0.0, max_tokens: int = 256, 
                 test_mode: str = "length", prefer_client: str = "auto"):
        """
        Initialize the test evaluator.
        
        Args:
            model: Model name to use
            client: Optional client (defaults to auto-detect if None)
            temperature: Temperature for generation
            max_tokens: Maximum tokens for generation
            test_mode: Test mode to use. Options:
                - "length": Returns normalized length score (0-1)
                - "constant": Returns a constant score (default 0.5)
                - "gold_match": Compares with gold answer if provided in kwargs
            prefer_client: Client preference ("auto", "ollama", or "hf")
        """
        super().__init__(model, client, temperature, max_tokens, prefer_client=prefer_client)
        self.test_mode = test_mode
    
    def evaluate(self, base_llm_output: str, **kwargs) -> float:
        """
        Evaluate the base LLM output based on the test mode.
        
        Args:
            base_llm_output: The output from the base LLM.
            **kwargs: Additional keyword arguments. May include 'gold' for comparison.
        
        Returns:
            A test reward score (float between 0 and 1).
        """
        if self.test_mode == "length":
            # Return a score based on output length (normalized to 0-1)
            # Cap at reasonable length for normalization
            normalized_length = min(len(base_llm_output.strip()) / 100.0, 1.0)
            return float(normalized_length)
        
        elif self.test_mode == "constant":
            # Return a constant test score
            return 0.5
        
        elif self.test_mode == "gold_match":
            # Compare with gold answer if provided
            gold = kwargs.get("gold", None)
            if gold is None:
                return 0.0
            
            # Simple exact match check
            if base_llm_output.strip() == gold.strip():
                return 1.0
            else:
                # Use token F1 for partial match
                return self.token_f1(base_llm_output, gold)
        
        else:
            # Default: return 0.5
            return 0.5
