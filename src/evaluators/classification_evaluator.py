from typing import Optional, Dict, Any, List
from trl.rewards import accuracy_reward
from .base_evalator import BaseEvaluator

class ClassificationEvaluator(BaseEvaluator):
    """
    A simple classification evaluator for classification tasks.
    """
    
    def __init__(self, model: str, client=None, temperature: float = 0.0, max_tokens: int = 256,
                 prefer_client: str = "auto"):
        """
        Initialize the classification evaluator.
        """
        super().__init__(model, client, temperature, max_tokens, prefer_client=prefer_client)
    def evaluate(self, base_llm_output: str, **kwargs) -> float:
        """
        Evaluate the classification output.
        """

        gold_label = int(kwargs.get("label", None))
        base_llm_output = base_llm_output.strip().lower()
        matching_case = {
            1: "World",
            2: "Sports",
            3: "Business",
            4: "Science/Technology",
        }

        if gold_label is not None:
            gold_label = matching_case[gold_label]

        if base_llm_output.strip().lower() == gold_label.strip().lower():
            return 1.0
        else:
            return 0.0