from typing import Optional, Dict, Any, List
from trl.rewards import accuracy_reward
from .base_evalator import BaseEvaluator


class EasyMathEvaluator(BaseEvaluator):
    """
    An evaluator for GSM8K math problems.
    Uses accuracy_reward to compare model outputs with ground truth answers.
    """
    
    def __init__(self, model: str, client=None, temperature: float = 0.0, max_tokens: int = 256,
                 prefer_client: str = "auto", evaluator_8bit: bool = False):
        """
        Initialize the EasyMath evaluator for GSM8K.
        
        Args:
            model: Model name to use
            client: Optional client (defaults to auto-detect if None)
            temperature: Temperature for generation
            max_tokens: Maximum tokens for generation
            prefer_client: Client preference ("auto", "ollama", or "hf")
            evaluator_8bit: If True, use 8-bit quantization for HFClient (reduces memory usage)
        """
        super().__init__(model, client, temperature, max_tokens, prefer_client=prefer_client, evaluator_8bit=evaluator_8bit)

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
        
        rewards = []
        for i in range(len(base_llm_outputs)):
            # Handle batched solution data - get solution for this specific item
            if isinstance(solution, list):
                current_solution = solution[i] if i < len(solution) else None
            else:
                current_solution = solution
            
            # Check if solution is in the output (check last 100 characters)
            if current_solution is not None:
                output_trimmed = base_llm_outputs[i][-50:]
                # Ensure both are strings for the 'in' operator
                if isinstance(current_solution, str) and current_solution in output_trimmed:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            else:
                rewards.append(0.0)
        print("Rewards: ", rewards)
        return rewards