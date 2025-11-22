"""finetune.py

GRPO-based finetuning using TRL's GRPOTrainer.

This module provides a Finetune class that uses parsers and evaluators
to download, parse datasets, and train models with reward functions.
"""
from __future__ import annotations
from typing import List, Optional
import os
from trl import GRPOTrainer, GRPOConfig
from src.parsers.base_parser import BaseParser
from src.evaluators.base_evalator import BaseEvaluator


class Finetune:
    """Fine-tuning class that uses parsers and evaluators with GRPOTrainer."""
    
    def __init__(
        self,
        model: str,
        parser: BaseParser,
        evaluator: BaseEvaluator,
        training_args: Optional[GRPOConfig] = GRPOConfig(output_dir="./trainer_output"),
    ):
        """
        Initialize the Finetune class.
        
        Args:
            model: Model identifier (e.g., "Qwen/Qwen2-0.5B-Instruct")
            parser: Parser instance that implements BaseParser interface
            evaluator: Evaluator instance that implements BaseEvaluator interface
            training_args: Optional GRPOConfig for training. If None, uses default config.
        """
        self.model = model
        self.parser = parser
        self.evaluator = evaluator
        
        # Download the dataset using the parser
        print("Downloading dataset...")
        self.parser.download_dataset()
        
        # Parse the dataset using the parser
        print("Parsing dataset...")
        self.dataset = self.parser.parse()
        
        # Create a wrapper function that adapts GRPOTrainer's format to evaluator's format
        def reward_func_wrapper(completions: List[List[dict]], **kwargs) -> List[float]:
            """
            Wrapper function that adapts GRPOTrainer's completion format to evaluator's format.
            
            GRPOTrainer passes completions as: [[{"content": "..."}], ...]
            BaseEvaluator.reward_function expects: List[str]
            
            Args:
                completions: List of completions, where each completion is [{"content": "..."}]
                **kwargs: Additional keyword arguments passed to the reward function
                
            Returns:
                List of reward scores (floats)
            """
            # Extract content strings from GRPOTrainer's format
            completion_contents = [completion[0]["content"] for completion in completions]
            
            # Call the evaluator's reward_function
            return self.evaluator.reward_function(completion_contents, **kwargs)
        
        self.reward_funcs = reward_func_wrapper
        
        # Set default training args if not provided
        if training_args is None:
            training_args = GRPOConfig(output_dir=f"{model.replace('/', '-')}-GRPO")
        
        # Check if model exists locally and use local path if available
        # Models are stored in models/{owner}/{model_name} format
        model_path = self.model
        local_model_dir = f"models/{self.model}"
        if os.path.exists(local_model_dir) and os.path.isfile(os.path.join(local_model_dir, "config.json")):
            print(f"Using local model from: {local_model_dir}")
            model_path = local_model_dir
        else:
            print(f"Model not found locally at {local_model_dir}, will try HuggingFace Hub")
            print("Note: On compute nodes without internet, models must be pre-downloaded to models/ directory")
        
        # Initialize GRPOTrainer
        print("Initializing GRPOTrainer...")
        # GRPOTrainer will use the model_path (local directory if found, or HuggingFace ID)
        # With HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 set in environment,
        # transformers library will use local cache only
        self.trainer = GRPOTrainer(
            model=model_path,
            reward_funcs=self.reward_funcs,
            args=training_args,
            train_dataset=self.dataset,
        )
    
    def train(self):
        """Start training the model."""
        print("Starting GRPO training...")
        self.trainer.train()
        print("Training completed!")

