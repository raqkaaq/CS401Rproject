"""finetune.py

GRPO-based finetuning using TRL's GRPOTrainer.

This module provides a Finetune class that uses parsers and evaluators
to download, parse datasets, and train models with reward functions.
"""
from __future__ import annotations
from typing import List, Optional
import os
import torch
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
        
        # Check GPU availability
        if torch.cuda.is_available():
            print(f"CUDA is available. GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("Warning: CUDA is not available. Training will run on CPU (very slow).")
            print("Make sure you're running on a GPU node and CUDA module is loaded.")
        
        # Initialize GRPOTrainer
        print("Initializing GRPOTrainer...")
        # GRPOTrainer will use the model_path (local directory if found, or HuggingFace ID)
        # With HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 set in environment,
        # transformers library will use local cache only
        # GRPOTrainer will automatically use GPU via accelerate if available
        
        # Load tokenizer separately to set padding_side before GRPOTrainer uses it
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Set left padding for decoder-only models (required for correct generation)
        tokenizer.padding_side = 'left'
        print(f"✓ Tokenizer configured with padding_side='{tokenizer.padding_side}'")
        
        self.trainer = GRPOTrainer(
            model=model_path,
            tokenizer=tokenizer,  # Pass the configured tokenizer
            reward_funcs=self.reward_funcs,
            args=training_args,
            train_dataset=self.dataset,
        )
        
        # Ensure tokenizer padding_side is still set after GRPOTrainer initialization
        if hasattr(self.trainer, 'tokenizer') and self.trainer.tokenizer is not None:
            self.trainer.tokenizer.padding_side = 'left'
            print(f"✓ Verified tokenizer padding_side='{self.trainer.tokenizer.padding_side}'")
        
        # Verify model is on GPU (if available)
        if torch.cuda.is_available() and hasattr(self.trainer, 'model'):
            try:
                model_device = next(self.trainer.model.parameters()).device
                if model_device.type == "cuda":
                    print(f"✓ Model is on GPU: {model_device}")
                else:
                    print(f"⚠ Warning: Model is on {model_device}, expected CUDA")
            except Exception as e:
                print(f"Could not verify model device: {e}")
    
    def train(self):
        """Start training the model."""
        print("Starting GRPO training...")
        self.trainer.train()
        print("Training completed!")

