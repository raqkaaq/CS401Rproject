"""main.py

Main entry point for GRPO fine-tuning using parsers and evaluators.
"""
import argparse
from trl import GRPOConfig

from src.finetune import Finetune
from src.parsers.test_parser import TestParser
from src.parsers.math_parser import MathParser
from src.evaluators.test_evaluator import TestEvaluator
from src.evaluators.math_evaluator import MathEvaluator


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
        choices=["test", "math"],
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
        choices=["test", "math"],
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
    
    args = parser.parse_args()
    
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
            max_tokens=5012,
            prefer_client=args.client_type
        )
    else:
        raise ValueError(f"Unknown evaluator type: {args.evaluator_type}")
    
    # Create training config
    # GRPOConfig save_strategy options: 'no', 'epoch', 'steps'
    # - 'epoch': saves at end of each epoch (save_steps is ignored)
    # - 'steps': saves every save_steps steps
    # - 'no': only saves final model
    
    training_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,  # Required parameter, but only used when save_strategy='steps'
        bf16=True,  # Enable bf16 for H200/H100 GPUs (Ampere+ architecture)
        # fp16=False,  # bf16 is preferred for newer GPUs
    )
    
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
    finetune.train()
    print("Training completed!")


if __name__ == "__main__":
    main()

