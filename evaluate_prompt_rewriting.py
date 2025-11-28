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
        --inference-model qwen2.5:0.5b-instruct \\
        --parser-type math \\
        --evaluator-type math \\
        --num-test-samples 100 \\
        --output-file results.json

    # Evaluate on poem dataset
    python evaluate_prompt_rewriting.py \\
        --rewriter-model-dir ./trainer_output/checkpoint-450 \\
        --inference-model Qwen/Qwen2.5-0.5B-Instruct \\
        --parser-type poem \\
        --evaluator-type poem \\
        --num-test-samples 50 \\
        --inference-client-type hf \\
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

from src.inference import HFClient, OllamaClient
from src.parsers.test_parser import TestParser
from src.parsers.math_parser import MathParser
from src.parsers.poem_parser import PoemParser
from src.evaluators.test_evaluator import TestEvaluator
from src.evaluators.math_evaluator import MathEvaluator
from src.evaluators.poem_evaluator import PoemEvaluator


def load_rewriter_model(model_dir: str, client_type: str = "auto") -> Any:
    """
    Load the fine-tuned prompt rewriting model.
    
    Args:
        model_dir: Directory containing the fine-tuned model
        client_type: Client type preference ("auto", "ollama", or "hf")
    
    Returns:
        Client instance with the model loaded
    """
    print(f"Loading rewriter model from: {model_dir}")
    
    # Check if model directory exists
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Determine client type
    if client_type == "hf":
        print("Using HFClient for rewriter model")
        torch_dtype = "bf16" if torch.cuda.is_available() else None
        if torch_dtype:
            print(f"  Using {torch_dtype} precision for GPU inference")
        client = HFClient(torch_dtype=torch_dtype)
    elif client_type == "ollama":
        print("Using OllamaClient for rewriter model")
        client = OllamaClient()
        if not client.is_healthy():
            raise RuntimeError("Ollama is not available but 'ollama' was specified")
    else:  # auto
        print("Auto-detecting client for rewriter model...")
        ollama_client = OllamaClient()
        if ollama_client.is_healthy():
            print("Ollama is available, using OllamaClient")
            client = ollama_client
        else:
            print("Ollama is not available, using HFClient")
            torch_dtype = "bf16" if torch.cuda.is_available() else None
            if torch_dtype:
                print(f"  Using {torch_dtype} precision for GPU inference")
            client = HFClient(torch_dtype=torch_dtype)
    
    # Load the model
    # For HF models, use the model directory path
    # For Ollama, we'd need to convert/load the model differently
    if isinstance(client, HFClient):
        # Use the model directory directly
        client.warmup_model(model_dir)
        client.model_id = model_dir  # Store the model path
    else:
        # For Ollama, we need the model name, not the directory
        # This assumes the model has been imported to Ollama
        model_name = os.path.basename(model_dir.rstrip("/"))
        client.warmup_model(model_name)
        client.model_id = model_name
    
    print("✓ Rewriter model loaded successfully")
    return client


def load_inference_model(model: str, client_type: str = "auto") -> Any:
    """
    Load the inference model for attempting the rewritten prompts.
    
    Args:
        model: Model identifier (HF format or Ollama format)
        client_type: Client type preference ("auto", "ollama", or "hf")
    
    Returns:
        Client instance with the model loaded
    """
    print(f"Loading inference model: {model}")
    
    if client_type == "hf":
        print("Using HFClient for inference model")
        torch_dtype = "bf16" if torch.cuda.is_available() else None
        if torch_dtype:
            print(f"  Using {torch_dtype} precision for GPU inference")
        client = HFClient(torch_dtype=torch_dtype)
        client.warmup_model(model)
    elif client_type == "ollama":
        print("Using OllamaClient for inference model")
        client = OllamaClient()
        if not client.is_healthy():
            raise RuntimeError("Ollama is not available but 'ollama' was specified")
        client.warmup_model(model)
    else:  # auto
        print("Auto-detecting client for inference model...")
        ollama_client = OllamaClient()
        if ollama_client.is_healthy():
            print("Ollama is available, using OllamaClient")
            client = ollama_client
            client.warmup_model(model)
        else:
            print("Ollama is not available, using HFClient")
            torch_dtype = "bf16" if torch.cuda.is_available() else None
            if torch_dtype:
                print(f"  Using {torch_dtype} precision for GPU inference")
            client = HFClient(torch_dtype=torch_dtype)
            client.warmup_model(model)
    
    print("✓ Inference model loaded successfully")
    return client


def generate_rewritten_prompt(rewriter_client: Any, prompt_messages: List[Dict[str, str]], 
                              max_tokens: int = 512) -> str:
    """
    Use the fine-tuned model to rewrite a prompt.
    
    Args:
        rewriter_client: Client instance with the rewriter model loaded
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
    
    # Generate rewritten prompt
    if isinstance(rewriter_client, HFClient):
        # Use batch generation with single prompt to get proper token extraction
        # This ensures we only get the newly generated tokens
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
    else:  # OllamaClient
        rewritten = rewriter_client.generate(
            model=rewriter_client.model_id,
            prompt=rewriter_input,
            temperature=0.0,
            max_tokens=max_tokens
        )
        return rewritten


def run_evaluation(
    rewriter_model_dir: str,
    inference_model: str,
    parser_type: str,
    evaluator_type: str,
    num_test_samples: Optional[int] = None,
    meta_prompt: str = "",
    dataset_name: Optional[str] = None,
    rewriter_client_type: str = "auto",
    inference_client_type: str = "auto",
    rewriter_max_tokens: int = 512,
    inference_max_tokens: int = 256,
    compare_baseline: bool = True,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the full evaluation pipeline.
    
    Args:
        rewriter_model_dir: Directory containing the fine-tuned rewriter model
        inference_model: Model identifier for inference (base LLM)
        parser_type: Type of parser ("test", "math", "poem")
        evaluator_type: Type of evaluator ("test", "math", "poem")
        num_test_samples: Number of test samples to evaluate (None = all)
        meta_prompt: Meta prompt to use for parsing
        dataset_name: Dataset name for parser (optional)
        rewriter_client_type: Client type for rewriter ("auto", "ollama", "hf")
        inference_client_type: Client type for inference ("auto", "ollama", "hf")
        rewriter_max_tokens: Max tokens for prompt rewriting
        inference_max_tokens: Max tokens for inference
        compare_baseline: Whether to compare with baseline (original prompt)
        output_file: Optional file path to save results (JSON)
    
    Returns:
        Dictionary containing evaluation results and statistics
    """
    print("=" * 80)
    print("Starting Prompt Rewriting Evaluation")
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
    
    # Load rewriter model
    print(f"\n2. Loading rewriter model from: {rewriter_model_dir}")
    rewriter_client = load_rewriter_model(rewriter_model_dir, rewriter_client_type)
    
    # Load inference model
    print(f"\n3. Loading inference model: {inference_model}")
    inference_client = load_inference_model(inference_model, inference_client_type)
    
    # Create evaluator
    print(f"\n4. Creating {evaluator_type} evaluator...")
    if evaluator_type == "test":
        evaluator = TestEvaluator(
            model=inference_model,
            client=inference_client,
            temperature=0.0,
            max_tokens=inference_max_tokens,
            prefer_client=inference_client_type
        )
    elif evaluator_type == "math":
        evaluator = MathEvaluator(
            model=inference_model,
            client=inference_client,
            temperature=0.0,
            max_tokens=inference_max_tokens,
            prefer_client=inference_client_type
        )
    elif evaluator_type == "poem":
        evaluator = PoemEvaluator(
            model=inference_model,
            client=inference_client,
            temperature=0.0,
            max_tokens=inference_max_tokens,
            prefer_client=inference_client_type
        )
    else:
        raise ValueError(f"Unknown evaluator type: {evaluator_type}")
    
    print("✓ Evaluator created")
    
    # Run evaluation on each test sample
    print(f"\n5. Running evaluation on {len(test_data)} samples...")
    results = []
    
    for i, sample in enumerate(tqdm(test_data, desc="Evaluating")):
        # Extract prompt messages
        prompt_messages = sample.get("prompt", [])
        if not prompt_messages:
            print(f"Warning: Sample {i} has no prompt, skipping")
            continue
        
        # Get original user prompt (for baseline comparison)
        original_user_prompt = None
        for msg in prompt_messages:
            if msg.get("role") == "user":
                original_user_prompt = msg.get("content", "")
                break
        
        if not original_user_prompt:
            print(f"Warning: Sample {i} has no user prompt, skipping")
            continue
        
        # Generate rewritten prompt using fine-tuned model
        try:
            rewritten_prompt = generate_rewritten_prompt(
                rewriter_client,
                prompt_messages,
                max_tokens=rewriter_max_tokens
            )
        except Exception as e:
            print(f"Error rewriting prompt for sample {i}: {e}")
            rewritten_prompt = ""
        
        # Attempt task with rewritten prompt
        # Note: For instruction-tuned models, the output typically contains only
        # the generated continuation (answer), not the input prompt
        try:
            if isinstance(inference_client, HFClient):
                # Extract only newly generated tokens for consistency
                tokenizer = inference_client.tokenizer
                toks = tokenizer(rewritten_prompt, return_tensors="pt", padding=True, truncation=True)
                toks = {k: v.to(inference_client.device) for k, v in toks.items()}
                input_len = (toks.get("attention_mask") == 1).sum().item()
                
                with torch.inference_mode():
                    outputs = inference_client.model.generate(
                        input_ids=toks.get("input_ids"),
                        attention_mask=toks.get("attention_mask"),
                        max_new_tokens=inference_max_tokens,
                        do_sample=False,
                        use_cache=True,
                    )
                
                # Extract only the newly generated tokens
                generated_tokens = outputs[0][input_len:]
                eval_rewritten = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            else:  # OllamaClient
                eval_rewritten = inference_client.generate(
                    model=inference_model,
                    prompt=rewritten_prompt,
                    temperature=0.0,
                    max_tokens=inference_max_tokens
                )
        except Exception as e:
            print(f"Error generating with rewritten prompt for sample {i}: {e}")
            eval_rewritten = ""
        
        # Evaluate rewritten output
        try:
            # Prepare kwargs for evaluator (include any additional fields from sample)
            eval_kwargs = {k: v for k, v in sample.items() if k != "prompt"}
            
            # Call evaluator's reward function
            reward_rewritten = evaluator.reward_function(
                [eval_rewritten],
                **eval_kwargs
            )[0]
        except Exception as e:
            print(f"Error evaluating rewritten output for sample {i}: {e}")
            reward_rewritten = 0.0
        
        # Baseline comparison (original prompt without rewriting)
        reward_baseline = None
        eval_baseline = None
        
        if compare_baseline:
            try:
                if isinstance(inference_client, HFClient):
                    # Extract only newly generated tokens for consistency
                    tokenizer = inference_client.tokenizer
                    toks = tokenizer(original_user_prompt, return_tensors="pt", padding=True, truncation=True)
                    toks = {k: v.to(inference_client.device) for k, v in toks.items()}
                    input_len = (toks.get("attention_mask") == 1).sum().item()
                    
                    with torch.inference_mode():
                        outputs = inference_client.model.generate(
                            input_ids=toks.get("input_ids"),
                            attention_mask=toks.get("attention_mask"),
                            max_new_tokens=inference_max_tokens,
                            do_sample=False,
                            use_cache=True,
                        )
                    
                    # Extract only the newly generated tokens
                    generated_tokens = outputs[0][input_len:]
                    eval_baseline = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                else:  # OllamaClient
                    eval_baseline = inference_client.generate(
                        model=inference_model,
                        prompt=original_user_prompt,
                        temperature=0.0,
                        max_tokens=inference_max_tokens
                    )
                
                eval_kwargs = {k: v for k, v in sample.items() if k != "prompt"}
                reward_baseline = evaluator.reward_function(
                    [eval_baseline],
                    **eval_kwargs
                )[0]
            except Exception as e:
                print(f"Error evaluating baseline for sample {i}: {e}")
                reward_baseline = 0.0
        
        # Store results
        result = {
            "sample_index": i,
            "original_prompt": original_user_prompt,
            "rewritten_prompt": rewritten_prompt,
            "eval_rewritten": eval_rewritten,
            "reward_rewritten": reward_rewritten,
        }
        
        if compare_baseline:
            result.update({
                "eval_baseline": eval_baseline,
                "reward_baseline": reward_baseline,
                "improvement": reward_rewritten - reward_baseline if reward_baseline is not None else None
            })
        
        # Include any additional fields from the sample
        for key, value in sample.items():
            if key != "prompt" and key not in result:
                result[key] = value
        
        results.append(result)
    
    # Calculate statistics
    print(f"\n6. Calculating statistics...")
    rewards_rewritten = [r["reward_rewritten"] for r in results]
    
    stats = {
        "num_samples": len(results),
        "mean_reward_rewritten": sum(rewards_rewritten) / len(rewards_rewritten) if rewards_rewritten else 0.0,
        "max_reward_rewritten": max(rewards_rewritten) if rewards_rewritten else 0.0,
        "min_reward_rewritten": min(rewards_rewritten) if rewards_rewritten else 0.0,
    }
    
    if compare_baseline:
        rewards_baseline = [r.get("reward_baseline", 0.0) for r in results if r.get("reward_baseline") is not None]
        improvements = [r.get("improvement", 0.0) for r in results if r.get("improvement") is not None]
        
        stats.update({
            "mean_reward_baseline": sum(rewards_baseline) / len(rewards_baseline) if rewards_baseline else 0.0,
            "mean_improvement": sum(improvements) / len(improvements) if improvements else 0.0,
            "num_improved": sum(1 for imp in improvements if imp > 0),
            "num_worse": sum(1 for imp in improvements if imp < 0),
            "num_same": sum(1 for imp in improvements if imp == 0),
        })
    
    # Print summary
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    print(f"Total samples evaluated: {stats['num_samples']}")
    print(f"Mean reward (rewritten): {stats['mean_reward_rewritten']:.4f}")
    
    if compare_baseline:
        print(f"Mean reward (baseline): {stats['mean_reward_baseline']:.4f}")
        print(f"Mean improvement: {stats['mean_improvement']:.4f}")
        print(f"Samples improved: {stats['num_improved']}")
        print(f"Samples worse: {stats['num_worse']}")
        print(f"Samples same: {stats['num_same']}")
    
    print("=" * 80)
    
    # Prepare final results
    final_results = {
        "config": {
            "rewriter_model_dir": rewriter_model_dir,
            "inference_model": inference_model,
            "parser_type": parser_type,
            "evaluator_type": evaluator_type,
            "num_test_samples": num_test_samples,
            "compare_baseline": compare_baseline,
        },
        "statistics": stats,
        "results": results,
    }
    
    # Save results if output file specified
    if output_file:
        print(f"\n7. Saving results to: {output_file}")
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        print("✓ Results saved")
    
    return final_results


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
        "--inference-model",
        type=str,
        required=True,
        help="Model identifier for inference (base LLM that attempts the task). "
             "HF format: 'Qwen/Qwen2.5-0.5B-Instruct', Ollama format: 'qwen2.5:0.5b-instruct'"
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
    parser.add_argument(
        "--rewriter-client-type",
        type=str,
        default="auto",
        choices=["auto", "ollama", "hf"],
        help="Client type for rewriter model"
    )
    parser.add_argument(
        "--inference-client-type",
        type=str,
        default="auto",
        choices=["auto", "ollama", "hf"],
        help="Client type for inference model"
    )
    
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
        "--no-baseline",
        action="store_true",
        help="Don't compare with baseline (original prompt)"
    )
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
        inference_model=args.inference_model,
        parser_type=args.parser_type,
        evaluator_type=args.evaluator_type,
        num_test_samples=args.num_test_samples,
        meta_prompt=args.meta_prompt,
        dataset_name=args.dataset_name,
        rewriter_client_type=args.rewriter_client_type,
        inference_client_type=args.inference_client_type,
        rewriter_max_tokens=args.rewriter_max_tokens,
        inference_max_tokens=args.inference_max_tokens,
        compare_baseline=not args.no_baseline,
        output_file=args.output_file,
    )
    
    print("\n✓ Evaluation completed successfully!")


if __name__ == "__main__":
    main()

