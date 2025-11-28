#!/usr/bin/env python3
"""
Evaluation script for prompt rewriting models.

This script evaluates if fine-tuning improves prompt rewriting by:
1. Loading a fine-tuned model for prompt rewriting
2. Loading test data using a parser
3. For each test sample:
   - Uses the fine-tuned model to rewrite the prompt
   - Uses an inference model to attempt the rewritten prompt
   - Evaluates the output with a reward score
4. Optionally compares with baseline (original prompt without rewriting)
5. Collects statistics and saves results

Example usage:
    # Evaluate a fine-tuned model on math problems
    python evaluate_prompt_rewriting.py \\
        --rewriter-model-dir ./trainer_output/checkpoint-450 \\
        --base-rewriter-model Qwen/Qwen2.5-0.5B-Instruct \\
        --inference-model Qwen/Qwen2.5-0.5B-Instruct \\
        --parser-type math \\
        --evaluator-type math \\
        --num-test-samples 100 \\
        --output-file results.json

    # Evaluate on poem dataset
    python evaluate_prompt_rewriting.py \\
        --rewriter-model-dir ./trainer_output/checkpoint-450 \\
        --base-rewriter-model Qwen/Qwen2.5-0.5B-Instruct \\
        --inference-model Qwen/Qwen2.5-0.5B-Instruct \\
        --parser-type poem \\
        --evaluator-type poem \\
        --num-test-samples 50 \\
        --output-file poem_results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.inference import HFClient
from src.parsers.test_parser import TestParser
from src.parsers.math_parser import MathParser
from src.parsers.poem_parser import PoemParser
from src.evaluators.test_evaluator import TestEvaluator
from src.evaluators.math_evaluator import MathEvaluator
from src.evaluators.poem_evaluator import PoemEvaluator


def load_rewriter_model(model_path: str) -> HFClient:
    """
    Load the prompt rewriting model using HuggingFace format.
    
    Args:
        model_path: Directory path or HF model identifier for the model
    
    Returns:
        HFClient instance with the model loaded
    """
    print(f"Loading rewriter model from: {model_path}")
    
    # Check if it's a directory path
    if os.path.exists(model_path) and os.path.isdir(model_path):
        if not os.path.exists(os.path.join(model_path, "config.json")):
            raise FileNotFoundError(f"Model directory exists but config.json not found: {model_path}")
        print(f"  Using local model directory")
    else:
        print(f"  Will download from HuggingFace: {model_path}")
    
    # Always use HFClient
    torch_dtype = "bf16" if torch.cuda.is_available() else None
    if torch_dtype:
        print(f"  Using {torch_dtype} precision for GPU inference")
    client = HFClient(torch_dtype=torch_dtype)
    
    # Load the model
    client.warmup_model(model_path)
    client.model_id = model_path  # Store the model path/identifier
    
    print("✓ Rewriter model loaded successfully")
    return client


def load_inference_model(model: str) -> HFClient:
    """
    Load the inference model for attempting the rewritten prompts.
    
    Args:
        model: HuggingFace model identifier or local directory path
    
    Returns:
        HFClient instance with the model loaded
    """
    print(f"Loading inference model: {model}")
    
    # Check if it's a local directory
    if os.path.exists(model) and os.path.isdir(model):
        print(f"  Using local model directory")
    else:
        print(f"  Will download from HuggingFace if needed")
    
    # Always use HFClient
    torch_dtype = "bf16" if torch.cuda.is_available() else None
    if torch_dtype:
        print(f"  Using {torch_dtype} precision for GPU inference")
    client = HFClient(torch_dtype=torch_dtype)
    client.warmup_model(model)
    
    print("✓ Inference model loaded successfully")
    return client


def generate_rewritten_prompt(rewriter_client: HFClient, prompt_messages: List[Dict[str, str]], 
                              max_tokens: int = 512) -> str:
    """
    Use the model to rewrite a prompt.
    
    Args:
        rewriter_client: HFClient instance with the rewriter model loaded
        prompt_messages: List of message dicts with 'role' and 'content' keys
        max_tokens: Maximum tokens for rewriting
    
    Returns:
        Rewritten prompt string (only the newly generated part)
    """
    # Construct input prompt from messages
    # Format: system message + user message (same as training)
    system_msg = ""
    user_msg = ""
    
    for msg in prompt_messages:
        if msg.get("role") == "system":
            system_msg = msg.get("content", "")
        elif msg.get("role") == "user":
            user_msg = msg.get("content", "")
    
    # Compose rewriter input exactly how training used it
    if system_msg:
        rewriter_input = f"{system_msg}\n\n{user_msg}"
    else:
        rewriter_input = user_msg
    
    # Generate rewritten prompt using HFClient
    tokenizer = rewriter_client.tokenizer
    toks = tokenizer(rewriter_input, return_tensors="pt", padding=True, truncation=True)
    toks = {k: v.to(rewriter_client.device) for k, v in toks.items()}
    input_len = (toks.get("attention_mask") == 1).sum().item()
    
    # Generate using the model directly
    with torch.inference_mode():
        outputs = rewriter_client.model.generate(
            input_ids=toks.get("input_ids"),
            attention_mask=toks.get("attention_mask"),
            max_new_tokens=max_tokens,
            do_sample=False,
            use_cache=True,
        )
    
    # Extract only the newly generated tokens (after input length)
    generated_tokens = outputs[0][input_len:]
    rewritten = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return rewritten


def run_evaluation(
    rewriter_model_dir: str,
    base_rewriter_model: str,
    inference_model: str,
    parser_type: str,
    evaluator_type: str,
    num_test_samples: Optional[int] = None,
    meta_prompt: str = "",
    dataset_name: Optional[str] = None,
    rewriter_max_tokens: int = 1024,
    inference_max_tokens: int = 1024,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the full evaluation pipeline comparing fine-tuned vs base rewriter models.
    
    Flow:
    1. Fine-tuned Rewriter LLM -> rewritten prompt -> Inference LLM -> output -> reward
    2. Base Rewriter LLM -> rewritten prompt -> Inference LLM -> output -> reward
    3. Compare rewards to see if fine-tuning improved rewriting
    
    Args:
        rewriter_model_dir: Directory containing the fine-tuned rewriter model
        base_rewriter_model: Model identifier for base (pre-finetuned) rewriter model
        inference_model: Model identifier for inference (base LLM that solves the task)
        parser_type: Type of parser ("test", "math", "poem")
        evaluator_type: Type of evaluator ("test", "math", "poem")
        num_test_samples: Number of test samples to evaluate (None = all)
        meta_prompt: Meta prompt to use for parsing
        dataset_name: Dataset name for parser (optional)
        rewriter_max_tokens: Max tokens for prompt rewriting
        inference_max_tokens: Max tokens for inference
        output_file: Optional file path to save results (JSON)
    
    Returns:
        Dictionary containing evaluation results and statistics
    """
    print("=" * 80)
    print("Starting Prompt Rewriting Evaluation")
    print("=" * 80)
    print("Comparing Fine-tuned vs Base Rewriter Models")
    print("=" * 80)
    
    # Load test data using parser
    print(f"\n1. Loading test data using {parser_type} parser...")
    if parser_type == "test":
        parser = TestParser(
            dataset_name=dataset_name or "test_dataset",
            meta_prompt=meta_prompt,
            num_samples=num_test_samples or 10
        )
    elif parser_type == "math":
        parser = MathParser(
            dataset_name=dataset_name or "math_dataset",
            meta_prompt=meta_prompt,
            num_samples=num_test_samples or 100
        )
    elif parser_type == "poem":
        parser = PoemParser(
            dataset_name=dataset_name or "poem_dataset",
            meta_prompt=meta_prompt,
            num_samples=num_test_samples or 100
        )
    else:
        raise ValueError(f"Unknown parser type: {parser_type}")
    
    parser.download_dataset()
    test_data = parser.parse()
    
    if num_test_samples:
        test_data = test_data[:num_test_samples]
    
    print(f"✓ Loaded {len(test_data)} test samples")
    
    # Load fine-tuned rewriter model
    print(f"\n2. Loading fine-tuned rewriter model from: {rewriter_model_dir}")
    finetuned_rewriter_client = load_rewriter_model(rewriter_model_dir)
    
    # Load base rewriter model
    print(f"\n3. Loading base rewriter model: {base_rewriter_model}")
    base_rewriter_client = load_rewriter_model(base_rewriter_model)
    
    # Load inference model
    print(f"\n4. Loading inference model: {inference_model}")
    inference_client = load_inference_model(inference_model)
    
    # Create evaluator
    print(f"\n5. Creating {evaluator_type} evaluator...")
    if evaluator_type == "test":
            evaluator = TestEvaluator(
            model=inference_model,
            client=inference_client,
            temperature=0.0,
            max_tokens=inference_max_tokens,
            prefer_client="hf"
        )
    elif evaluator_type == "math":
        evaluator = MathEvaluator(
            model=inference_model,
            client=inference_client,
            temperature=0.0,
            max_tokens=inference_max_tokens,
            prefer_client="hf"
        )
    elif evaluator_type == "poem":
        evaluator = PoemEvaluator(
            model=inference_model,
            client=inference_client,
            temperature=0.0,
            max_tokens=inference_max_tokens,
            prefer_client="hf"
        )
    else:
        raise ValueError(f"Unknown evaluator type: {evaluator_type}")
    
    print("✓ Evaluator created")
    
    # Run evaluation on each test sample
    print(f"\n6. Running evaluation on {len(test_data)} samples...")
    print("   Flow: Rewriter -> Rewritten Prompt -> Inference Model -> Output -> Reward")
    results = []
    
    for i, sample in enumerate(tqdm(test_data, desc="Evaluating")):
        # Extract prompt messages
        prompt_messages = sample.get("prompt", [])
        if not prompt_messages:
            print(f"Warning: Sample {i} has no prompt, skipping")
            continue
        
        # Get original user prompt
        original_user_prompt = None
        for msg in prompt_messages:
            if msg.get("role") == "user":
                original_user_prompt = msg.get("content", "")
                break
        
        if not original_user_prompt:
            print(f"Warning: Sample {i} has no user prompt, skipping")
            continue
        
        # ===== FINE-TUNED REWRITER PATH =====
        # Generate rewritten prompt using fine-tuned model
        try:
            rewritten_prompt_finetuned = generate_rewritten_prompt(
                finetuned_rewriter_client,
                prompt_messages,
                max_tokens=rewriter_max_tokens
            )
        except Exception as e:
            print(f"Error rewriting prompt with fine-tuned model for sample {i}: {e}")
            rewritten_prompt_finetuned = ""
        
        # Attempt task with rewritten prompt (fine-tuned)
        try:
            eval_output_finetuned = _generate_with_inference_model(
                inference_client, inference_model, rewritten_prompt_finetuned, inference_max_tokens
            )
        except Exception as e:
            print(f"Error generating with fine-tuned rewritten prompt for sample {i}: {e}")
            eval_output_finetuned = ""
        
        # Evaluate fine-tuned output
        try:
            eval_kwargs = {k: v for k, v in sample.items() if k != "prompt"}
            reward_finetuned = evaluator.reward_function([eval_output_finetuned], **eval_kwargs)[0]
        except Exception as e:
            print(f"Error evaluating fine-tuned output for sample {i}: {e}")
            reward_finetuned = 0.0
        
        # ===== BASE REWRITER PATH =====
        # Generate rewritten prompt using base model
        try:
            rewritten_prompt_base = generate_rewritten_prompt(
                base_rewriter_client,
                prompt_messages,
                max_tokens=rewriter_max_tokens
            )
        except Exception as e:
            print(f"Error rewriting prompt with base model for sample {i}: {e}")
            rewritten_prompt_base = ""
        
        # Attempt task with rewritten prompt (base)
        try:
            eval_output_base = _generate_with_inference_model(
                inference_client, inference_model, rewritten_prompt_base, inference_max_tokens
            )
        except Exception as e:
            print(f"Error generating with base rewritten prompt for sample {i}: {e}")
            eval_output_base = ""
        
        # Evaluate base output
        try:
            eval_kwargs = {k: v for k, v in sample.items() if k != "prompt"}
            reward_base = evaluator.reward_function([eval_output_base], **eval_kwargs)[0]
        except Exception as e:
            print(f"Error evaluating base output for sample {i}: {e}")
            reward_base = 0.0
        
        # Store results
        result = {
            "sample_index": i,
            "original_prompt": original_user_prompt,
            "rewritten_prompt_finetuned": rewritten_prompt_finetuned,
            "rewritten_prompt_base": rewritten_prompt_base,
            "eval_output_finetuned": eval_output_finetuned,
            "eval_output_base": eval_output_base,
            "reward_finetuned": reward_finetuned,
            "reward_base": reward_base,
            "improvement": reward_finetuned - reward_base,
        }
        
        # Include any additional fields from the sample
        for key, value in sample.items():
            if key != "prompt" and key not in result:
                result[key] = value
        
        results.append(result)
    
    # Calculate statistics
    print(f"\n7. Calculating statistics...")
    rewards_finetuned = [r["reward_finetuned"] for r in results]
    rewards_base = [r["reward_base"] for r in results]
    improvements = [r["improvement"] for r in results]
    
    stats = {
        "num_samples": len(results),
        "mean_reward_finetuned": sum(rewards_finetuned) / len(rewards_finetuned) if rewards_finetuned else 0.0,
        "mean_reward_base": sum(rewards_base) / len(rewards_base) if rewards_base else 0.0,
        "mean_improvement": sum(improvements) / len(improvements) if improvements else 0.0,
        "max_reward_finetuned": max(rewards_finetuned) if rewards_finetuned else 0.0,
        "max_reward_base": max(rewards_base) if rewards_base else 0.0,
        "min_reward_finetuned": min(rewards_finetuned) if rewards_finetuned else 0.0,
        "min_reward_base": min(rewards_base) if rewards_base else 0.0,
        "num_improved": sum(1 for imp in improvements if imp > 0),
        "num_worse": sum(1 for imp in improvements if imp < 0),
        "num_same": sum(1 for imp in improvements if imp == 0),
    }
    
    # Print summary
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    print(f"Total samples evaluated: {stats['num_samples']}")
    print(f"\nFine-tuned Rewriter:")
    print(f"  Mean reward: {stats['mean_reward_finetuned']:.4f}")
    print(f"  Max reward:  {stats['max_reward_finetuned']:.4f}")
    print(f"  Min reward:  {stats['min_reward_finetuned']:.4f}")
    print(f"\nBase Rewriter:")
    print(f"  Mean reward: {stats['mean_reward_base']:.4f}")
    print(f"  Max reward:  {stats['max_reward_base']:.4f}")
    print(f"  Min reward:  {stats['min_reward_base']:.4f}")
    print(f"\nComparison:")
    print(f"  Mean improvement: {stats['mean_improvement']:.4f}")
    print(f"  Samples improved: {stats['num_improved']} ({100*stats['num_improved']/stats['num_samples']:.1f}%)")
    print(f"  Samples worse:    {stats['num_worse']} ({100*stats['num_worse']/stats['num_samples']:.1f}%)")
    print(f"  Samples same:     {stats['num_same']} ({100*stats['num_same']/stats['num_samples']:.1f}%)")
    print("=" * 80)
    
    # Prepare final results
    final_results = {
        "config": {
            "rewriter_model_dir": rewriter_model_dir,
            "base_rewriter_model": base_rewriter_model,
            "inference_model": inference_model,
            "parser_type": parser_type,
            "evaluator_type": evaluator_type,
            "num_test_samples": num_test_samples,
        },
        "statistics": stats,
        "results": results,
    }
    
    # Save results if output file specified
    if output_file:
        print(f"\n8. Saving results to: {output_file}")
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        print("✓ Results saved")
    
    return final_results


def _generate_with_inference_model(
    inference_client: HFClient,
    inference_model: str,
    prompt: str,
    max_tokens: int
) -> str:
    """
    Helper function to generate output using inference model.
    
    Args:
        inference_client: HFClient instance with inference model loaded
        inference_model: Model identifier
        prompt: Prompt to generate from
        max_tokens: Maximum tokens to generate
    
    Returns:
        Generated output string (only newly generated tokens)
    """
    # Extract only newly generated tokens
    tokenizer = inference_client.tokenizer
    toks = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    toks = {k: v.to(inference_client.device) for k, v in toks.items()}
    input_len = (toks.get("attention_mask") == 1).sum().item()
    
    with torch.inference_mode():
        outputs = inference_client.model.generate(
            input_ids=toks.get("input_ids"),
            attention_mask=toks.get("attention_mask"),
            max_new_tokens=max_tokens,
            do_sample=False,
            use_cache=True,
        )
    
    # Extract only the newly generated tokens
    generated_tokens = outputs[0][input_len:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate prompt rewriting models on test data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--rewriter-model-dir",
        type=str,
        required=True,
        help="Directory containing the fine-tuned prompt rewriting model"
    )
    parser.add_argument(
        "--base-rewriter-model",
        type=str,
        required=True,
        help="Model identifier for base (pre-finetuned) rewriter model. "
             "HF format: 'Qwen/Qwen2.5-0.5B-Instruct' or local directory path"
    )
    parser.add_argument(
        "--inference-model",
        type=str,
        required=True,
        help="Model identifier for inference (base LLM that attempts the task). "
             "HF format: 'Qwen/Qwen2.5-0.5B-Instruct' or local directory path"
    )
    
    # Parser arguments
    parser.add_argument(
        "--parser-type",
        type=str,
        default="test",
        choices=["test", "math", "poem"],
        help="Type of parser to use for loading test data"
    )
    parser.add_argument(
        "--num-test-samples",
        type=int,
        default=None,
        help="Number of test samples to evaluate (None = all available)"
    )
    parser.add_argument(
        "--meta-prompt",
        type=str,
        default="",
        help="Meta prompt to use for parsing test data"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset name for parser (optional, uses default if not specified)"
    )
    
    # Evaluator arguments
    parser.add_argument(
        "--evaluator-type",
        type=str,
        default="test",
        choices=["test", "math", "poem"],
        help="Type of evaluator to use for scoring"
    )
    
    # Client type arguments
    # Generation arguments
    parser.add_argument(
        "--rewriter-max-tokens",
        type=int,
        default=512,
        help="Maximum tokens for prompt rewriting"
    )
    parser.add_argument(
        "--inference-max-tokens",
        type=int,
        default=256,
        help="Maximum tokens for inference (task completion)"
    )
    
    # Evaluation options
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file path to save results (JSON format)"
    )
    
    args = parser.parse_args()
    
    # Verify GPU availability   
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠ Warning: CUDA is not available!")
    
    # Run evaluation
    results = run_evaluation(
        rewriter_model_dir=args.rewriter_model_dir,
        base_rewriter_model=args.base_rewriter_model,
        inference_model=args.inference_model,
        parser_type=args.parser_type,
        evaluator_type=args.evaluator_type,
        num_test_samples=args.num_test_samples,
        meta_prompt=args.meta_prompt,
        dataset_name=args.dataset_name,
        rewriter_max_tokens=args.rewriter_max_tokens,
        inference_max_tokens=args.inference_max_tokens,
        output_file=args.output_file,
    )
    
    print("\n✓ Evaluation completed successfully!")


if __name__ == "__main__":
    main()

