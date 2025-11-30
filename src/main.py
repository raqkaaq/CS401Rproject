"""main.py

Main entry point for GRPO fine-tuning using parsers and evaluators.
"""
import argparse
import sys
from pathlib import Path
import torch

# Add project root to path so imports work
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trl import GRPOConfig

from src.finetune import Finetune
from src.parsers.test_parser import TestParser
from src.parsers.math_parser import MathParser
from src.parsers.poem_parser import PoemParser
from src.parsers.classification_parser import ClassificationParser
from src.evaluators.test_evaluator import TestEvaluator
from src.evaluators.math_evaluator import MathEvaluator
from src.evaluators.poem_evaluator import PoemEvaluator
from src.evaluators.classification_evaluator import ClassificationEvaluator

def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="GRPO Fine-tuning with Parsers and Evaluators")
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model identifier for training (HuggingFace format, e.g., 'Qwen/Qwen2.5-0.5B-Instruct')"
    )
    
    # Parser arguments
    parser.add_argument(
        "--parser-type",
        type=str,
        default="test",
        choices=["test", "math", "poem", "classification"],
        help="Type of parser to use"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="test_dataset",
        help="Name of the dataset"
    )
    parser.add_argument(
        "--meta-prompt",
        type=str,
        default="",
        help="Meta prompt to prepend to each prompt"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to use (for test parser)"
    )
    
    # Evaluator arguments
    parser.add_argument(
        "--evaluator-type",
        type=str,
        default="test",
        choices=["test", "math", "poem", "classification"],
        help="Type of evaluator to use"
    )
    parser.add_argument(
        "--evaluator-model",
        type=str,
        default="qwen2.5:0.5b-instruct",
        help="Model to use for evaluation (Ollama format for Ollama, e.g., 'qwen2.5:0.5b-instruct', or HF format for HF, e.g., 'Qwen/Qwen2.5-0.5B-Instruct')"
    )
    parser.add_argument(
        "--client-type",
        type=str,
        default="auto",
        choices=["auto", "ollama", "hf"],
        help="Client type preference: 'auto' (try Ollama, fallback to HF), 'ollama' (force Ollama), 'hf' (force HF)"
    )
    parser.add_argument(
        "--test-mode",
        type=str,
        default="length",
        choices=["length", "constant", "gold_match"],
        help="Test mode for test evaluator"
    )
    
    # Training arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./trainer_output",
        help="Output directory for training"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size"
    )
    parser.add_argument(
        "--save-steps",
        type=float,
        default=500,
        help="Save checkpoint every N steps (only used if save-strategy is 'steps')"
    )
    parser.add_argument(
        "--save-strategy",
        type=str,
        default="steps",
        choices=["no", "epoch", "steps"],
        help="Save strategy: 'no' (only final), 'epoch' (end of each epoch), 'steps' (every N steps via save-steps)"
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Log training metrics every N steps"
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=None,
        help="Maximum number of tokens for generated completions (default: model default, typically 256)"
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=None,
        help="Maximum number of tokens for input prompts (default: model default, typically 512)"
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from (e.g., './trainer_output/checkpoint-450'). If not specified, will auto-detect latest checkpoint in output_dir if training was interrupted."
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.05,
        help="KL penalty coefficient (beta) for controlling divergence from reference model. Default: 0.0 (no penalty). Typical values: 0.01-0.1. Higher values prevent policy from deviating too far from reference."
    )
    
    args = parser.parse_args()
    
    # Verify GPU availability
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("⚠ Warning: CUDA is not available!")
        print("  Training will run on CPU (very slow).")
        print("  Make sure you're running on a GPU node with CUDA module loaded.")
        # In non-interactive mode (SLURM jobs), continue anyway but warn
        import sys
        if sys.stdin.isatty():
            # Interactive mode - ask user
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
        else:
            # Non-interactive mode (SLURM) - warn and continue
            print("  Continuing in non-interactive mode (SLURM job)...")
    
    # Create parser
    print(f"Creating {args.parser_type} parser...")
    if args.parser_type == "test":
        parser_instance = TestParser(
            dataset_name=args.dataset_name,
            meta_prompt=args.meta_prompt,
            num_samples=args.num_samples
        )
    elif args.parser_type == "math":
        parser_instance = MathParser(
            dataset_name=args.dataset_name,
            meta_prompt=args.meta_prompt,
            num_samples=args.num_samples
        )
    elif args.parser_type == "poem":
        parser_instance = PoemParser(
            dataset_name=args.dataset_name,
            meta_prompt=args.meta_prompt,
            num_samples=args.num_samples
        )
    elif args.parser_type == "classification":
        parser_instance = ClassificationParser(
            dataset_name=args.dataset_name,
            meta_prompt=args.meta_prompt,
            num_samples=args.num_samples
        )
    else:
        raise ValueError(f"Unknown parser type: {args.parser_type}")
    
    # Create evaluator
    print(f"Creating {args.evaluator_type} evaluator...")
    if args.evaluator_type == "test":
        evaluator_instance = TestEvaluator(
            model=args.evaluator_model,
            client=None,  # Will auto-detect based on prefer_client
            temperature=0.0,
            max_tokens=256,
            test_mode=args.test_mode,
            prefer_client=args.client_type
        )
    elif args.evaluator_type == "math": 
        evaluator_instance = MathEvaluator(
            model=args.evaluator_model,
            client=None,  # Will auto-detect based on prefer_client
            temperature=0.0,
            max_tokens=512,  # Reduced from 5012 for faster evaluation (math problems rarely need more)
            prefer_client=args.client_type
        )
    elif args.evaluator_type == "poem":
        evaluator_instance = PoemEvaluator(
            model=args.evaluator_model,
            client=None,  # Will auto-detect based on prefer_client
            temperature=0.0,
            max_tokens=512,
            prefer_client=args.client_type
        )
    elif args.evaluator_type == "classification":
        evaluator_instance = ClassificationEvaluator(
            model=args.evaluator_model,
            client=None,
            temperature=0.0,
            max_tokens=512,
            prefer_client=args.client_type
        )
    else:
        raise ValueError(f"Unknown evaluator type: {args.evaluator_type}")
    
    # Create training config
    # GRPOConfig save_strategy options: 'no', 'epoch', 'steps'
    # - 'epoch': saves at end of each epoch (save_steps is ignored)
    # - 'steps': saves every save_steps steps
    # - 'no': only saves final model
    
    # Configure training with GPU optimizations
    training_config_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "logging_steps": args.logging_steps,
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,  # Required parameter, but only used when save_strategy='steps'
        "bf16": torch.cuda.is_available(),  # Enable bf16 only if GPU is available (H200/H100/A100 support bf16)
        # fp16=False,  # bf16 is preferred for newer GPUs
        "dataloader_pin_memory": torch.cuda.is_available(),  # Pin memory for faster GPU data transfer
        "beta": args.beta,  # KL penalty coefficient
    }
    
    # Add token length limits if specified
    if args.max_completion_length is not None:
        training_config_kwargs["max_completion_length"] = args.max_completion_length
        print(f"✓ Setting max_completion_length={args.max_completion_length} tokens")
    if args.max_prompt_length is not None:
        training_config_kwargs["max_prompt_length"] = args.max_prompt_length
        print(f"✓ Setting max_prompt_length={args.max_prompt_length} tokens")
    if args.beta > 0.0:
        print(f"✓ KL penalty (beta) enabled: {args.beta}")
    else:
        print(f"  KL penalty (beta) disabled: {args.beta} (no constraint on policy divergence)")
    
    training_config = GRPOConfig(**training_config_kwargs)
    
    # Create finetune instance
    print("Initializing Finetune...")
    finetune = Finetune(
        model=args.model,
        parser=parser_instance,
        evaluator=evaluator_instance,
        training_args=training_config
    )
    
    # Start training
    print("Starting training...")
    finetune.train(resume_from_checkpoint=args.resume_from_checkpoint)
    print("Training completed!")


if __name__ == "__main__":
    main()

