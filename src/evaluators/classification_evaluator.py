from typing import Optional, Dict, Any, List
from trl.rewards import accuracy_reward
from .base_evalator import BaseEvaluator
import re

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
        label = kwargs.get("label", None)
        
        # Handle both single value and list cases (defensive programming)
        if isinstance(label, list):
            if len(label) > 0:
                gold_label = int(label[0])
            else:
                return 0.0
        elif label is not None:
            gold_label = int(label)
        else:
            return 0.0
        
        base_llm_output = base_llm_output.strip()
        matching_case = {
            1: "World",
            2: "Sports",
            3: "Business",
            4: "Science/Technology",
        }

        if gold_label in matching_case:
            gold_label_str = matching_case[gold_label]
        else:
            return 0.0

        # Check if the answer is in the last 30 characters
        last_30_chars = base_llm_output[-30:].lower()
        gold_label_lower = gold_label_str.strip().lower()

        brevity_score = max(0, 1 - len(base_llm_output) / 128)
        correctness_score = 0

        # Also attempt to grab text after Classificaiton:
        classification_text = re.search(r"Classification:\s*(.+)", base_llm_output)
        if classification_text:
            classification_text = classification_text.group(1).strip().lower()
            if gold_label_lower in classification_text:
                correctness_score = 1
                return (correctness_score * 0.7) + (brevity_score * 0.3)
            else:
                return 0.0
        else:
            return 0.0
        if gold_label_lower in last_30_chars:
            correctness_score = 1
            return (correctness_score * 0.7) + (brevity_score * 0.3)
        else:
            return 0.0