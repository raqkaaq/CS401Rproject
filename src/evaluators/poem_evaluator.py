import poetrytools
from collections import Counter
import re
from .base_evalator import BaseEvaluator

class PoemEvaluator(BaseEvaluator):
    def __init__(self, model: str, client=None, temperature: float = 0.0, max_tokens: int = 512,
                 prefer_client: str = "auto"):
        """
        Initialize the poem evaluator.
        
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
        Evaluates the poem output based on multiple factors to prevent reward hacking.
        Combines rhyme, diversity, repetition penalty, length, and semantic coherence.

        Args:
            base_llm_output: The output from the base LLM (poem as a string).
            **kwargs: Additional keyword arguments. May include 'gold' for comparison.

        Returns:
            A test reward score (float between 0 and 1).
        """
        # Clean the input
        base_llm_output = base_llm_output.replace("—", " ")
        
        
        return float(max(0.0, min(1.0, final_score)))
