from typing import Optional, Dict, Any, List, Tuple
from trl.rewards import accuracy_reward
from .base_evalator import BaseEvaluator
import re

class ClassificationEvaluator(BaseEvaluator):
    """
    A simple classification evaluator for classification tasks.
    Returns two separate rewards using one-hot encoding:
    - Class reward: one-hot encoded based on predicted class
    - Length reward: one-hot encoded based on output length bucket
    """
    
    def __init__(self, model: str, client=None, temperature: float = 0.0, max_tokens: int = 256,
                 prefer_client: str = "auto"):
        """
        Initialize the classification evaluator.
        """
        super().__init__(model, client, temperature, max_tokens, prefer_client=prefer_client)
        
        # Define length buckets for one-hot encoding
        # [very_short, short, medium, long, very_long]
        self.length_buckets = [0, 50, 100, 150, 200]  # Character thresholds
    
    def _get_class_one_hot(self, base_llm_output: str, gold_label: int) -> List[float]:
        """
        Get one-hot encoded class reward.
        Returns a vector of size 4 (one for each class) where:
        - The predicted class position gets 1.0 if correct, 0.0 if incorrect
        - This allows the model to learn which class was predicted and whether it's correct
        """
        base_llm_output = base_llm_output.strip().lower()
        matching_case = {
            1: "world",
            2: "sports",
            3: "business",
            4: "science/technology",
        }
        
        if gold_label not in matching_case:
            return [0.0, 0.0, 0.0, 0.0]
        
        gold_label_str = matching_case[gold_label]
        
        # One-hot encoding: [World, Sports, Business, Science/Technology]
        class_reward = [0.0, 0.0, 0.0, 0.0]
        predicted_class_idx = None
        
        # Check for classification in the output
        classification_text = re.search(r"classification:\s*(.+)", base_llm_output)
        other_classification_text = re.search(r"category:\s*(.+)", base_llm_output)
        if classification_text:
            classification_text = classification_text.group(1).strip()
            
            # Check which class name appears in the classification text
            for idx, class_name in matching_case.items():
                if class_name in classification_text:
                    predicted_class_idx = idx - 1  # Convert to 0-based index
                    break
        elif other_classification_text:
            other_classification_text = other_classification_text.group(1).strip()
            for idx, class_name in matching_case.items():
                if class_name in other_classification_text:
                    predicted_class_idx = idx - 1  # Convert to 0-based index
                    break
        
        # Fallback: check last 30 characters for any class name
        if predicted_class_idx is None:
            last_30_chars = base_llm_output[-30:]
            for idx, class_name in matching_case.items():
                if class_name in last_30_chars:
                    predicted_class_idx = idx - 1  # Convert to 0-based index
                    break
        
        # If we found a predicted class, set the reward
        if predicted_class_idx is not None:
            # Reward is 1.0 if predicted class matches gold label, else 0.0
            # This encodes both which class was predicted and correctness
            if predicted_class_idx == (gold_label - 1):
                class_reward[predicted_class_idx] = 1.0
            else:
                class_reward[predicted_class_idx] = 0.0
        
        return class_reward
    
    def _get_length_one_hot(self, base_llm_output: str) -> List[float]:
        """
        Get one-hot encoded length reward.
        Returns a vector of size 5 representing length buckets:
        [very_short, short, medium, long, very_long]
        """
        length = len(base_llm_output.strip())
        
        # One-hot encoding: [very_short, short, medium, long, very_long]
        length_reward = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Determine which bucket the length falls into
        if length < self.length_buckets[1]:  # < 50
            length_reward[0] = 1.0  # very_short
        elif length < self.length_buckets[2]:  # 50-99
            length_reward[1] = 1.0  # short
        elif length < self.length_buckets[3]:  # 100-149
            length_reward[2] = 1.0  # medium
        elif length < self.length_buckets[4]:  # 150-199
            length_reward[3] = 1.0  # long
        else:  # >= 200
            length_reward[4] = 1.0  # very_long
        
        return length_reward
    
    def evaluate(self, base_llm_output: str, **kwargs) -> Tuple[List[float], List[float]]:
        """
        Evaluate the classification output.
        Returns a tuple of (class_reward, length_reward) where each is a one-hot encoded vector.
        
        Returns:
            Tuple[List[float], List[float]]: (class_one_hot, length_one_hot)
            - class_one_hot: 4-element vector [World, Sports, Business, Science/Technology]
            - length_one_hot: 5-element vector [very_short, short, medium, long, very_long]
        """
        label = kwargs.get("label", None)
        
        # Handle both single value and list cases (defensive programming)
        if isinstance(label, list):
            if len(label) > 0:
                gold_label = int(label[0])
            else:
                return ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0])
        elif label is not None:
            gold_label = int(label)
        else:
            return ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0])
        
        base_llm_output = base_llm_output.strip()
        
        # Get one-hot encoded rewards
        class_reward = self._get_class_one_hot(base_llm_output, gold_label)
        length_reward = self._get_length_one_hot(base_llm_output)
        
        return (class_reward, length_reward)
    
    def reward_function(self, completions: List[List[Dict[str, str]]], **kwargs) -> List[List[float]]:
        """
        Reward function to be used by the GRPOTrainer.
        Returns a list of reward vectors, where each vector is the concatenation of
        class one-hot encoding (4 elements) and length one-hot encoding (5 elements).
        
        Returns:
            List[List[float]]: List of 9-element reward vectors [class_4_elements + length_5_elements]
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
        
        # Log first rewritten prompt (completion from rewriter)
        if completion_strings:
            print("\n" + "="*80)
            print("FIRST REWRITTEN PROMPT (from rewriter):")
            print("="*80)
            print(completion_strings[0])
            print("="*80 + "\n")
        
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
            individual_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, list) and len(value) > i:
                    individual_kwargs[key] = value[i]
                else:
                    individual_kwargs[key] = value
            
            # Get tuple of (class_reward, length_reward)
            class_reward, length_reward = self.evaluate(base_llm_output, **individual_kwargs)
            
            # Concatenate the two one-hot encoded rewards into a single vector
            # Result: [class_4_elements] + [length_5_elements] = 9 elements total
            combined_reward = class_reward + length_reward
            rewards.append(combined_reward)
        
        print("Rewards (class + length one-hot encoded): ", rewards)
        return rewards