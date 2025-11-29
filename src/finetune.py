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
                          To limit token generation during training, set:
                          - max_completion_length: Maximum tokens for generated completions (e.g., 256)
                          - max_prompt_length: Maximum tokens for input prompts (e.g., 512)
                          Example: GRPOConfig(max_completion_length=256, max_prompt_length=512, ...)
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
            
            # Print the finetuned prompt outputs (completions)
            print("\n" + "="*80)
            print("FINETUNED PROMPT OUTPUTS (Training Step)")
            print("="*80)
            for i, completion_content in enumerate(completion_contents):
                if i == 0:
                    print(f"\n[Completion {i+1}/{len(completion_contents)}]")
                    print("-" * 80)
                    print("Rewritten prompt (what inference model will receive):")
                    print(completion_content)
                    print("-" * 80)
                    # Also print kwargs to see what other data is available
                    if i == 0:  # Only print once to avoid spam
                        print(f"Available kwargs keys: {list(kwargs.keys())}")
                        if "solution" in kwargs:
                            print(f"Sample solution: {kwargs['solution']}")
            print("="*80 + "\n")
            
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
        
        # Initialize GRPOTrainer (it will load the tokenizer automatically from the model)
        self.trainer = GRPOTrainer(
            model=model_path,
            reward_funcs=self.reward_funcs,
            args=training_args,
            train_dataset=self.dataset,
        )
        
        # Configure tokenizer after GRPOTrainer initialization
        if hasattr(self.trainer, 'tokenizer') and self.trainer.tokenizer is not None:
            tokenizer = self.trainer.tokenizer
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            # Set left padding for decoder-only models (required for correct generation)
            tokenizer.padding_side = 'left'
            print(f"✓ Tokenizer configured with padding_side='{tokenizer.padding_side}'")
        
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
    
    def train(self, resume_from_checkpoint: Optional[str] = None):
        """
        Start training the model.
        
        Args:
            resume_from_checkpoint: Optional path to checkpoint directory to resume from.
                                   If None and checkpoints exist in output_dir, will auto-detect
                                   the latest checkpoint.
        """
        # Auto-detect latest checkpoint if not specified
        if resume_from_checkpoint is None:
            output_dir = self.trainer.args.output_dir
            if os.path.exists(output_dir):
                # Find all checkpoint directories
                checkpoints = []
                for item in os.listdir(output_dir):
                    checkpoint_path = os.path.join(output_dir, item)
                    if os.path.isdir(checkpoint_path) and item.startswith("checkpoint-"):
                        # Extract step number
                        try:
                            step_num = int(item.split("-")[1])
                            checkpoints.append((step_num, checkpoint_path))
                        except (ValueError, IndexError):
                            continue
                
                if checkpoints:
                    # Sort by step number and get the latest
                    checkpoints.sort(key=lambda x: x[0])
                    latest_checkpoint = checkpoints[-1][1]
                    resume_from_checkpoint = latest_checkpoint
                    print(f"✓ Auto-detected latest checkpoint: {resume_from_checkpoint}")
                    print(f"  Resuming from step {checkpoints[-1][0]}")
        
        if resume_from_checkpoint:
            if not os.path.exists(resume_from_checkpoint):
                print(f"⚠ Warning: Checkpoint path does not exist: {resume_from_checkpoint}")
                print("  Starting training from scratch instead.")
                resume_from_checkpoint = None
            else:
                print(f"✓ Resuming training from checkpoint: {resume_from_checkpoint}")
        
        print("Starting GRPO training...")
        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        print("Training completed!")

