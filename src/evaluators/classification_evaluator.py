from typing import Optional, Dict, Any, List
from trl.rewards import accuracy_reward
from .base_evalator import BaseEvaluator
import re

class ClassificationEvaluator(BaseEvaluator):
    """
    A simple classification evaluator for classification tasks.
    """
    
    def __init__(self, model: str, client=None, temperature: float = 0.0, max_tokens: int = 256,
                 prefer_client: str = "auto", evaluator_8bit: bool = False):
        """
        Initialize the classification evaluator.
        
        Args:
            model: Model name to use
            client: Optional client (defaults to auto-detect if None)
            temperature: Temperature for generation
            max_tokens: Maximum tokens for generation
            prefer_client: Client preference ("auto", "ollama", or "hf")
            evaluator_8bit: If True, use 8-bit quantization for HFClient (reduces memory usage)
        """
        super().__init__(model, client, temperature, max_tokens, prefer_client=prefer_client, evaluator_8bit=evaluator_8bit)
    
    def length_evaluation(self, base_llm_output: str) -> float:
        """
        Evaluate the length of the classification output.
        """
        brevity_score = max(0.0, 1.0 - len(base_llm_output) / 128)
        return brevity_score
    
    def evaluate(self, base_llm_output: str, **kwargs) -> float:
        """
        Evaluate the classification output.
        Returns a single combined score (correctness * 0.7 + length * 0.3).
        """
        label = kwargs.get("label", None)
        correctness_score = 0.0
        
        # Handle both single value and list cases (defensive programming)
        if isinstance(label, list):
            if len(label) > 0:
                gold_label = int(label[0])
            else:
                correctness_score = 0.0
                length_score = self.length_evaluation(base_llm_output)
                return (correctness_score * 0.7) + (length_score * 0.3)
        elif label is not None:
            gold_label = int(label)
        else:
            correctness_score = 0.0
            length_score = self.length_evaluation(base_llm_output)
            return (correctness_score * 0.7) + (length_score * 0.3)
        
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
            correctness_score = 0.0
            length_score = self.length_evaluation(base_llm_output)
            return (correctness_score * 0.7) + (length_score * 0.3)

        # Check if the answer is in the last 30 characters
        last_30_chars = base_llm_output[-30:].lower()
        gold_label_lower = gold_label_str.strip().lower()

        # Also attempt to grab text after Classification:
        classification_text = re.search(r"Classification:\s*(.+)", base_llm_output)
        if classification_text:
            classification_text = classification_text.group(1).strip().lower()
            if gold_label_lower in classification_text:
                correctness_score = 1.0
            else:
                correctness_score = 0.0
        else:
            correctness_score = 0.0
        
        # Also check last 30 characters
        if gold_label_lower in last_30_chars:
            correctness_score = 1.0
        
        length_score = self.length_evaluation(base_llm_output)
        # Return a single combined score (not a list)
        return (correctness_score * 0.7) + (length_score * 0.3)