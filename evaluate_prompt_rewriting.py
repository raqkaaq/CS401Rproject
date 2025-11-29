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


For running on GPU no internet:

export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
# Force offline mode for HuggingFace (use cache only, no internet)
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

 python evaluate_prompt_rewriting.py \
        --rewriter-model-dir ./trainer_output/checkpoint-450 \
        --base-rewriter-model Qwen/Qwen2.5-0.5B-Instruct \
        --inference-model Qwen/Qwen2.5-0.5B-Instruct \
        --parser-type math \
        --evaluator-type math \
        --num-test-samples 100 \
        --output-file results.json


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
from trl.rewards import accuracy_reward


def load_rewriter_model(model_path: str) -> HFClient:
    """
    Load the prompt rewriting model using HuggingFace format.
    
    Args:
        model_path: Directory path (absolute or relative) or HF model identifier for the model
    
    Returns:
        HFClient instance with the model loaded
    """
    print(f"Loading rewriter model from: {model_path}")
    
    # Check if it's a local directory path (absolute or relative)
    # Resolve relative paths to absolute for easier handling
    if os.path.exists(model_path) and os.path.isdir(model_path):
        model_path = os.path.abspath(model_path)  # Convert to absolute path
        if not os.path.exists(os.path.join(model_path, "config.json")):
            raise FileNotFoundError(f"Model directory exists but config.json not found: {model_path}")
        print(f"  Using local model directory: {model_path}")
        is_local_dir = True
    else:
        print(f"  Will use HuggingFace model identifier: {model_path}")
        is_local_dir = False
    
    # Always use HFClient
    torch_dtype = "bf16" if torch.cuda.is_available() else None
    if torch_dtype:
        print(f"  Using {torch_dtype} precision for GPU inference")
    client = HFClient(torch_dtype=torch_dtype)
    
    # Load the model
    # For local directories, load directly. For HF model IDs, use warmup_model logic
    if is_local_dir:
        # Load directly from the local directory path
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        client.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if client.tokenizer.pad_token is None:
            client.tokenizer.pad_token = client.tokenizer.eos_token
        client.tokenizer.padding_side = 'left'
        
        # Build model kwargs
        model_kwargs = {}
        if isinstance(torch_dtype, str):
            dtype_map = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32,
            }
            model_kwargs["torch_dtype"] = dtype_map.get(torch_dtype.lower(), None)
        
        # Load model
        try:
            load_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
            if load_kwargs:
                client.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
            else:
                client.model = AutoModelForCausalLM.from_pretrained(model_path)
        except Exception as e:
            print(f"Model load with kwargs {model_kwargs} failed: {e}. Falling back to default load.")
            client.model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Move to device
        if torch.cuda.is_available():
            try:
                client.model.to(client.device)
                print(f"  Model moved to {client.device}")
            except Exception as e:
                print(f"Warning: Failed to move model to {client.device}: {e}")
        
        client.model.eval()
        
        # Verify model is on correct device
        try:
            first_param = next(client.model.parameters())
            print(f"  Model loaded on device: {first_param.device}")
        except StopIteration:
            pass
    else:
        # Use the existing warmup_model logic for HF model IDs
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
    
    # Check if it's a local directory path (absolute or relative)
    if os.path.exists(model) and os.path.isdir(model):
        model = os.path.abspath(model)  # Convert to absolute path
        if not os.path.exists(os.path.join(model, "config.json")):
            raise FileNotFoundError(f"Model directory exists but config.json not found: {model}")
        print(f"  Using local model directory: {model}")
        is_local_dir = True
    else:
        print(f"  Will use HuggingFace model identifier: {model}")
        is_local_dir = False
    
    # Always use HFClient
    torch_dtype = "bf16" if torch.cuda.is_available() else None
    if torch_dtype:
        print(f"  Using {torch_dtype} precision for GPU inference")
    client = HFClient(torch_dtype=torch_dtype)
    
    # Load the model
    # For local directories, load directly. For HF model IDs, use warmup_model logic
    if is_local_dir:
        # Load directly from the local directory path
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        client.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        if client.tokenizer.pad_token is None:
            client.tokenizer.pad_token = client.tokenizer.eos_token
        client.tokenizer.padding_side = 'left'
        
        # Build model kwargs
        model_kwargs = {}
        if isinstance(torch_dtype, str):
            dtype_map = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32,
            }
            model_kwargs["torch_dtype"] = dtype_map.get(torch_dtype.lower(), None)
        
        # Load model
        try:
            load_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
            if load_kwargs:
                client.model = AutoModelForCausalLM.from_pretrained(model, **load_kwargs)
            else:
                client.model = AutoModelForCausalLM.from_pretrained(model)
        except Exception as e:
            print(f"Model load with kwargs {model_kwargs} failed: {e}. Falling back to default load.")
            client.model = AutoModelForCausalLM.from_pretrained(model)
        
        # Move to device
        if torch.cuda.is_available():
            try:
                client.model.to(client.device)
                print(f"  Model moved to {client.device}")
            except Exception as e:
                print(f"Warning: Failed to move model to {client.device}: {e}")
        
        client.model.eval()
        
        # Verify model is on correct device
        try:
            first_param = next(client.model.parameters())
            print(f"  Model loaded on device: {first_param.device}")
        except StopIteration:
            pass
    else:
        # Use the existing warmup_model logic for HF model IDs
        client.warmup_model(model)
    
    print("✓ Inference model loaded successfully")
    return client


def _get_model(model_or_wrapper):
    """Get the actual model from either a model or DataParallel wrapper."""
    if isinstance(model_or_wrapper, torch.nn.DataParallel):
        return model_or_wrapper.module
    return model_or_wrapper


def generate_rewritten_prompt_batch(rewriter_client: HFClient, prompt_messages_list: List[List[Dict[str, str]]], 
                                     max_tokens: int = 512) -> List[str]:
    """
    Generate rewritten prompts for multiple inputs in batch (much faster than sequential).
    
    Args:
        rewriter_client: HFClient instance with the rewriter model loaded
        prompt_messages_list: List of prompt message lists (one per sample)
        max_tokens: Maximum tokens for rewriting
    
    Returns:
        List of rewritten prompt strings
    """
    # Construct input prompts from all messages
    rewriter_inputs = []
    for prompt_messages in prompt_messages_list:
        system_msg = ""
        user_msg = ""
        
        for msg in prompt_messages:
            if msg.get("role") == "system":
                system_msg = msg.get("content", "")
            elif msg.get("role") == "user":
                user_msg = msg.get("content", "")
        
        if system_msg:
            rewriter_input = f"{system_msg}\n\n{user_msg}"
        else:
            rewriter_input = user_msg
        
        rewriter_inputs.append(rewriter_input)
    
    # Batch tokenize and generate
    tokenizer = rewriter_client.tokenizer
    toks = tokenizer(rewriter_inputs, return_tensors="pt", padding=True, truncation=True)
    toks = {k: v.to(rewriter_client.device) for k, v in toks.items()}
    
    # Get input lengths for each prompt
    input_lengths = [(toks.get("attention_mask")[i] == 1).sum().item() for i in range(len(rewriter_inputs))]
    
    # Batch generate - handle DataParallel wrapper
    model = _get_model(rewriter_client.model)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=toks.get("input_ids"),
            attention_mask=toks.get("attention_mask"),
            max_new_tokens=max_tokens,
            do_sample=False,
            use_cache=True,
        )
    
    # Extract only the newly generated tokens for each output
    rewritten_prompts = []
    for i, output in enumerate(outputs):
        input_len = input_lengths[i]
        generated_tokens = output
        rewritten = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        rewritten_prompts.append(rewritten)
    
    return rewritten_prompts


def generate_rewritten_prompt(rewriter_client: HFClient, prompt_messages: List[Dict[str, str]], 
                              max_tokens: int = 512, max_retries: int = 3) -> str:
    """
    Use the model to rewrite a prompt with retry logic if output is empty.
    
    Args:
        rewriter_client: HFClient instance with the rewriter model loaded
        prompt_messages: List of message dicts with 'role' and 'content' keys
        max_tokens: Maximum tokens for rewriting
        max_retries: Maximum number of retries if output is empty
    
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
    
    # Generate rewritten prompt using HFClient with retry logic
    tokenizer = rewriter_client.tokenizer
    toks = tokenizer(rewriter_input, return_tensors="pt", padding=True, truncation=True)
    toks = {k: v.to(rewriter_client.device) for k, v in toks.items()}
    input_len = (toks.get("attention_mask") == 1).sum().item()
    
    # Retry up to max_retries times if output is empty
    model = _get_model(rewriter_client.model)
    for attempt in range(max_retries):
        # Generate using the model directly - handle DataParallel wrapper
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=toks.get("input_ids"),
                attention_mask=toks.get("attention_mask"),
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=True,
            )
        
        # Extract only the newly generated tokens (after input length)
        generated_tokens = outputs[0]
        rewritten = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # If we got a non-empty result, return it
        if rewritten and rewritten.strip():
            return rewritten
        
        # If this isn't the last attempt, continue to retry
        if attempt < max_retries - 1:
            continue
    
    # If all retries failed, return empty string
    return ""


def _evaluate_solution(
    evaluator: Any,
    generated_solution: str,
    sample: Dict[str, Any],
    evaluator_type: str
) -> float:
    """
    Evaluate a generated solution against the gold solution.
    
    Args:
        evaluator: Evaluator instance
        generated_solution: The generated solution string
        sample: Sample dictionary containing gold solution
        evaluator_type: Type of evaluator ("test", "math", "poem")
    
    Returns:
        Reward score (float)
    """
    if evaluator_type == "math":
        # For math evaluator, use accuracy_reward directly
        solution = sample.get("solution", None)
        if solution is None:
            return 0.0
        
        # Handle empty generated solution
        if not generated_solution or not isinstance(generated_solution, str) or not generated_solution.strip():
            return 0.0
        
        # Format for accuracy_reward: expects [[{"content": "..."}], ...]
        formatted_completions = [[{"content": generated_solution}]]
        
        # accuracy_reward expects solution as a list (same length as completions)
        # Looking at math_evaluator.py line 71, it passes solution directly which can be a list or single value
        if isinstance(solution, str):
            solution_list = [solution]
        elif isinstance(solution, list):
            # Use first element if list has items, otherwise empty string
            solution_list = [solution[0]] if len(solution) > 0 else [""]
        else:
            solution_list = [str(solution)]
        
        try:
            rewards = accuracy_reward(formatted_completions, solution_list)
            # accuracy_reward may return a tensor or list
            if hasattr(rewards, '__len__') and len(rewards) > 0:
                reward_val = rewards[0] if isinstance(rewards, (list, tuple)) else rewards.item() if hasattr(rewards, 'item') else float(rewards)
                return float(reward_val)
            elif isinstance(rewards, (int, float)):
                return float(rewards)
            else:
                return 0.0
        except Exception as e:
            print(f"Error in accuracy_reward: {e}")
            print(f"  formatted_completions: {formatted_completions}")
            print(f"  solution_list: {solution_list}")
            print(f"  generated_solution type: {type(generated_solution)}, value: {repr(generated_solution[:100]) if generated_solution else 'None'}")
            import traceback
            traceback.print_exc()
            return 0.0
    elif evaluator_type == "poem":
        # For poem evaluator, use the evaluate method directly
        return evaluator.evaluate(generated_solution, **{k: v for k, v in sample.items() if k != "prompt"})
    elif evaluator_type == "test":
        # For test evaluator, use the evaluate method directly
        return evaluator.evaluate(generated_solution, **{k: v for k, v in sample.items() if k != "prompt"})
    else:
        # Default: return 0.0
        return 0.0


def run_evaluation(
    rewriter_model_dir: str,
    base_rewriter_model: str,
    inference_model: str,
    parser_type: str,
    evaluator_type: str,
    num_test_samples: Optional[int] = None,
    meta_prompt: str = "",
    dataset_name: Optional[str] = None,
    rewriter_max_tokens: int = 512,
    inference_max_tokens: int = 512,
    batch_size: int = 1,
    num_gpus: int = 1,
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
        batch_size: Batch size for processing (1 = sequential, >1 = batched, much faster)
        num_gpus: Number of GPUs to use (1 = single GPU, >1 = multi-GPU via DataParallel)
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
    
    # Shuffle the data with a random seed for reproducibility (or use None for truly random)
    # This ensures different runs can use different samples if desired
    
    if num_test_samples:
        test_data = test_data[:num_test_samples]
    
    print(f"✓ Loaded {len(test_data)} test samples")
    
    # Setup multi-GPU if requested
    if num_gpus > 1 and torch.cuda.is_available():
        if torch.cuda.device_count() < num_gpus:
            print(f"Warning: Requested {num_gpus} GPUs but only {torch.cuda.device_count()} available. Using {torch.cuda.device_count()} GPUs.")
            num_gpus = torch.cuda.device_count()
        print(f"\n2. Setting up multi-GPU support ({num_gpus} GPUs)")
    else:
        num_gpus = 1
    
    # Load fine-tuned rewriter model
    print(f"\n{3 if num_gpus == 1 else 3}. Loading fine-tuned rewriter model from: {rewriter_model_dir}")
    finetuned_rewriter_client = load_rewriter_model(rewriter_model_dir)
    
    # Setup multi-GPU for rewriter models if requested
    if num_gpus > 1 and torch.cuda.is_available():
        if hasattr(finetuned_rewriter_client, 'model') and finetuned_rewriter_client.model is not None:
            finetuned_rewriter_client.model = torch.nn.DataParallel(finetuned_rewriter_client.model)
            print(f"  Fine-tuned rewriter model wrapped with DataParallel for {num_gpus} GPUs")
    
    # Load base rewriter model
    print(f"\n{4 if num_gpus == 1 else 4}. Loading base rewriter model: {base_rewriter_model}")
    base_rewriter_client = load_rewriter_model(base_rewriter_model)
    
    # Setup multi-GPU for base rewriter if requested
    if num_gpus > 1 and torch.cuda.is_available():
        if hasattr(base_rewriter_client, 'model') and base_rewriter_client.model is not None:
            base_rewriter_client.model = torch.nn.DataParallel(base_rewriter_client.model)
            print(f"  Base rewriter model wrapped with DataParallel for {num_gpus} GPUs")
    
    # Load inference model
    print(f"\n{5 if num_gpus == 1 else 5}. Loading inference model: {inference_model}")
    inference_client = load_inference_model(inference_model)
    
    # Setup multi-GPU for inference model if requested
    if num_gpus > 1 and torch.cuda.is_available():
        if hasattr(inference_client, 'model') and inference_client.model is not None:
            inference_client.model = torch.nn.DataParallel(inference_client.model)
            print(f"  Inference model wrapped with DataParallel for {num_gpus} GPUs")
    
    # Create evaluator
    print(f"\n{6 if num_gpus == 1 else 6}. Creating {evaluator_type} evaluator...")
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
    print(f"\n{7 if num_gpus == 1 else 7}. Running evaluation on {len(test_data)} samples...")
    if batch_size > 1:
        print(f"   Using batch size: {batch_size} (much faster!)")
    if num_gpus > 1:
        print(f"   Using {num_gpus} GPUs with DataParallel")
    print("   Flow: Rewriter -> Rewritten Prompt -> Inference Model -> Output -> Reward")
    results = []
    
    # Pre-process all samples to extract valid prompt messages
    valid_samples = []
    valid_indices = []
    for i, sample in enumerate(test_data):
        prompt_messages = sample.get("prompt", [])
        
        # Handle case where prompt_messages might be a string representation
        if isinstance(prompt_messages, str):
            try:
                import ast
                prompt_messages = ast.literal_eval(prompt_messages)
            except:
                continue
        
        if not prompt_messages or not isinstance(prompt_messages, list):
            continue
        
        # Get original user prompt
        original_user_prompt = None
        for msg in prompt_messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                original_user_prompt = msg.get("content", "")
                break
        
        if original_user_prompt:
            valid_samples.append((i, sample, prompt_messages))
            valid_indices.append(i)
    
    print(f"   Processing {len(valid_samples)} valid samples...")
    
    # Process in batches if batch_size > 1
    if batch_size > 1:
        # Process in batches
        for batch_start in tqdm(range(0, len(valid_samples), batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, len(valid_samples))
            batch_samples = valid_samples[batch_start:batch_end]
            
            # Extract prompt messages for batch
            batch_prompt_messages = [pm for _, _, pm in batch_samples]
            
            # Batch rewrite with fine-tuned model
            try:
                # Debug: Check if models have different generation configs
                if batch_start == 0:
                    finetuned_model = _get_model(finetuned_rewriter_client.model)
                    base_model = _get_model(base_rewriter_client.model)
                    if hasattr(finetuned_model, 'generation_config') and hasattr(base_model, 'generation_config'):
                        finetuned_max_new_tokens = getattr(finetuned_model.generation_config, 'max_new_tokens', None)
                        base_max_new_tokens = getattr(base_model.generation_config, 'max_new_tokens', None)
                        if finetuned_max_new_tokens != base_max_new_tokens:
                            print(f"Warning: Fine-tuned model generation_config.max_new_tokens={finetuned_max_new_tokens}, "
                                  f"Base model generation_config.max_new_tokens={base_max_new_tokens}")
                            print(f"  Using explicit max_tokens={rewriter_max_tokens} (this should override config)")
                
                rewritten_prompts_finetuned = generate_rewritten_prompt_batch(
                    finetuned_rewriter_client,
                    batch_prompt_messages,
                    max_tokens=rewriter_max_tokens
                )
                # Debug: Count empty outputs
                empty_count = sum(1 for rp in rewritten_prompts_finetuned if not rp or not rp.strip())
                if empty_count > 0:
                    # Track which samples have empty outputs
                    empty_indices = [batch_start + i for i, rp in enumerate(rewritten_prompts_finetuned) if not rp or not rp.strip()]
                    if batch_start == 0 or len(results) % 50 == 0:
                        print(f"Warning: Fine-tuned rewriter produced {empty_count}/{len(rewritten_prompts_finetuned)} empty outputs in batch {batch_start}-{batch_end}")
                        if len(empty_indices) <= 5:
                            print(f"  Empty output sample indices: {empty_indices}")
                        else:
                            print(f"  Empty output sample indices (first 5): {empty_indices[:5]}...")
                    if empty_count == len(rewritten_prompts_finetuned):
                        print(f"  All fine-tuned rewritten prompts are empty! Checking first sample...")
                        print(f"  First rewritten prompt (length): {len(rewritten_prompts_finetuned[0]) if rewritten_prompts_finetuned else 0}")
            except Exception as e:
                print(f"Error in batch rewriting (fine-tuned) for batch {batch_start}-{batch_end}: {e}")
                import traceback
                traceback.print_exc()
                rewritten_prompts_finetuned = [""] * len(batch_samples)
            
            # Batch rewrite with base model
            try:
                rewritten_prompts_base = generate_rewritten_prompt_batch(
                    base_rewriter_client,
                    batch_prompt_messages,
                    max_tokens=rewriter_max_tokens
                )
                # Debug: Count empty outputs
                empty_count = sum(1 for rp in rewritten_prompts_base if not rp or not rp.strip())
                if empty_count > 0 and batch_start == 0:
                    print(f"Warning: Base rewriter produced {empty_count}/{len(rewritten_prompts_base)} empty outputs in first batch")
            except Exception as e:
                print(f"Error in batch rewriting (base) for batch {batch_start}-{batch_end}: {e}")
                import traceback
                traceback.print_exc()
                rewritten_prompts_base = [""] * len(batch_samples)
            
            # Batch inference for fine-tuned rewritten prompts
            # Filter out empty prompts
            valid_finetuned_prompts = []
            valid_finetuned_indices = []
            for idx, rp in enumerate(rewritten_prompts_finetuned):
                if rp and rp.strip():
                    valid_finetuned_prompts.append(rp)
                    valid_finetuned_indices.append(idx)
            
            eval_outputs_finetuned = [""] * len(batch_samples)
            if valid_finetuned_prompts:
                try:
                    # Use batch generation
                    batch_outputs = inference_client.generate_batch(
                        model_id=inference_model,
                        prompts=valid_finetuned_prompts,
                        max_new_tokens=inference_max_tokens,
                        do_sample=False
                    )
                    for local_idx, global_idx in enumerate(valid_finetuned_indices):
                        eval_outputs_finetuned[global_idx] = batch_outputs[local_idx]
                except Exception as e:
                    print(f"Error in batch inference (fine-tuned) for batch {batch_start}-{batch_end}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                # All fine-tuned prompts were empty - log this for debugging
                if batch_start == 0:
                    print(f"Warning: All fine-tuned rewritten prompts were empty in first batch ({len(rewritten_prompts_finetuned)} samples)")
            
            # Batch inference for base rewritten prompts
            valid_base_prompts = []
            valid_base_indices = []
            for idx, rp in enumerate(rewritten_prompts_base):
                if rp and rp.strip():
                    valid_base_prompts.append(rp)
                    valid_base_indices.append(idx)
            
            eval_outputs_base = [""] * len(batch_samples)
            if valid_base_prompts:
                try:
                    # Use batch generation
                    batch_outputs = inference_client.generate_batch(
                        model_id=inference_model,
                        prompts=valid_base_prompts,
                        max_new_tokens=inference_max_tokens,
                        do_sample=False
                    )
                    for local_idx, global_idx in enumerate(valid_base_indices):
                        eval_outputs_base[global_idx] = batch_outputs[local_idx]
                except Exception as e:
                    print(f"Error in batch inference (base) for batch {batch_start}-{batch_end}: {e}")
            
            # Evaluate each sample in the batch
            for batch_idx, (orig_idx, sample, prompt_messages) in enumerate(batch_samples):
                original_user_prompt = None
                for msg in prompt_messages:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        original_user_prompt = msg.get("content", "")
                        break
                
                # Get results from batch processing
                rewritten_prompt_finetuned = rewritten_prompts_finetuned[batch_idx]
                rewritten_prompt_base = rewritten_prompts_base[batch_idx]
                eval_output_finetuned = eval_outputs_finetuned[batch_idx]
                eval_output_base = eval_outputs_base[batch_idx]
                
                # Evaluate outputs
                try:
                    reward_finetuned = _evaluate_solution(
                        evaluator, eval_output_finetuned, sample, evaluator_type
                    )
                except Exception as e:
                    if batch_start == 0 and batch_idx == 0:
                        print(f"Warning: Error evaluating fine-tuned output: {e}")
                    reward_finetuned = 0.0
                
                try:
                    reward_base = _evaluate_solution(
                        evaluator, eval_output_base, sample, evaluator_type
                    )
                except Exception as e:
                    if batch_start == 0 and batch_idx == 0:
                        print(f"Warning: Error evaluating base output: {e}")
                    reward_base = 0.0
                
                # Track samples where fine-tuned fails but base succeeds (the "worse" cases)
                if reward_finetuned < reward_base:
                    # This is a "worse" case - fine-tuned got lower reward than base
                    print(f"  [Sample {orig_idx}] Fine-tuned reward: {reward_finetuned:.4f} < Base reward: {reward_base:.4f}")
                    print(f"    Fine-tuned rewritten prompt empty: {not rewritten_prompt_finetuned or not rewritten_prompt_finetuned.strip()}")
                    print(f"    Base rewritten prompt empty: {not rewritten_prompt_base or not rewritten_prompt_base.strip()}")
                    print(f"    Fine-tuned eval output length: {len(eval_output_finetuned)}")
                    print(f"    Base eval output length: {len(eval_output_base)}")
                
                # Debug first sample in first batch
                if batch_start == 0 and batch_idx == 0:
                    print(f"\n=== DEBUG: First Sample ===")
                    print(f"Fine-tuned rewritten prompt length: {len(rewritten_prompt_finetuned)}")
                    print(f"Fine-tuned eval output length: {len(eval_output_finetuned)}")
                    print(f"Fine-tuned reward: {reward_finetuned}")
                    print(f"Base rewritten prompt length: {len(rewritten_prompt_base)}")
                    print(f"Base eval output length: {len(eval_output_base)}")
                    print(f"Base reward: {reward_base}")
                    if rewritten_prompt_finetuned:
                        print(f"Fine-tuned rewritten prompt (first 200 chars):\n {rewritten_prompt_finetuned[:200]}")
                    if eval_output_finetuned:
                        print(f"Fine-tuned eval output (first 200 chars):\n {eval_output_finetuned[:200]}")
                    if rewritten_prompt_base:
                        print(f"Base rewritten prompt (first 200 chars):\n {rewritten_prompt_base[:200]}")
                    if eval_output_base:
                        print(f"Base eval output (first 200 chars):\n {eval_output_base[:200]}")
                    print(f"===========================\n")
                
                # Store results
                result = {
                    "sample_index": orig_idx,
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
                
                # Print summary every 50 samples
                if len(results) % 50 == 0:
                    _print_progress_summary(results)
                    # Additional diagnostics for the "2 per 50" pattern
                    _print_diagnostic_summary(results[-50:])  # Last 50 samples
    else:
        # Sequential processing (original code)
        for i, sample in enumerate(tqdm(test_data, desc="Evaluating")):
            # Extract prompt messages
            prompt_messages = sample.get("prompt", [])
            
            # Handle case where prompt_messages might be a string representation
            if isinstance(prompt_messages, str):
                try:
                    import ast
                    prompt_messages = ast.literal_eval(prompt_messages)
                except:
                    print(f"Warning: Sample {i} has invalid prompt format (string that can't be parsed), skipping")
                    continue
            
            if not prompt_messages:
                print(f"Warning: Sample {i} has no prompt, skipping")
                continue
            
            if not isinstance(prompt_messages, list):
                print(f"Warning: Sample {i} prompt is not a list (type: {type(prompt_messages)}), skipping")
                continue
            
            # Get original user prompt
            original_user_prompt = None
            system_prompt = ""
            for msg in prompt_messages:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "")
                if role == "user":
                    original_user_prompt = msg.get("content", "")
                elif role == "system":
                    system_prompt = msg.get("content", "")
            
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
                if not rewritten_prompt_finetuned or rewritten_prompt_finetuned.strip() == "":
                    print(f"Warning: Sample {i} - fine-tuned rewriter produced empty output")
            except Exception as e:
                print(f"Error rewriting prompt with fine-tuned model for sample {i}: {e}")
                import traceback
                traceback.print_exc()
                rewritten_prompt_finetuned = ""
            
            # Attempt task with rewritten prompt (fine-tuned)
            # Skip if rewritten prompt is empty
            if not rewritten_prompt_finetuned or not rewritten_prompt_finetuned.strip():
                eval_output_finetuned = ""
            else:
                try:
                    eval_output_finetuned = _generate_with_inference_model(
                        inference_client, inference_model, rewritten_prompt_finetuned, inference_max_tokens
                    )
                except Exception as e:
                    print(f"Error generating with fine-tuned rewritten prompt for sample {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    eval_output_finetuned = ""
            
            # Evaluate fine-tuned output
            try:
                reward_finetuned = _evaluate_solution(
                    evaluator,
                    eval_output_finetuned,
                    sample,
                    evaluator_type
                )
            except Exception as e:
                print(f"Error evaluating fine-tuned output for sample {i}: {e}")
                import traceback
                traceback.print_exc()
                reward_finetuned = 0.0
            
            # ===== BASE REWRITER PATH =====
            # Generate rewritten prompt using base model
            try:
                rewritten_prompt_base = generate_rewritten_prompt(
                    base_rewriter_client,
                    prompt_messages,
                    max_tokens=rewriter_max_tokens
                )
                if not rewritten_prompt_base or rewritten_prompt_base.strip() == "":
                    print(f"Warning: Sample {i} - base rewriter produced empty output")
            except Exception as e:
                print(f"Error rewriting prompt with base model for sample {i}: {e}")
                import traceback
                traceback.print_exc()
                rewritten_prompt_base = ""
            
            # Attempt task with rewritten prompt (base)
            # Skip if rewritten prompt is empty
            if not rewritten_prompt_base or not rewritten_prompt_base.strip():
                eval_output_base = ""
            else:
                try:
                    eval_output_base = _generate_with_inference_model(
                        inference_client, inference_model, rewritten_prompt_base, inference_max_tokens
                    )
                except Exception as e:
                    print(f"Error generating with base rewritten prompt for sample {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    eval_output_base = ""
            
            # Evaluate base output
            try:
                reward_base = _evaluate_solution(
                    evaluator,
                    eval_output_base,
                    sample,
                    evaluator_type
                )
            except Exception as e:
                print(f"Error evaluating base output for sample {i}: {e}")
                import traceback
                traceback.print_exc()
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
            
            # Print summary every 50 samples
            if (i + 1) % 50 == 0:
                _print_progress_summary(results)
    
    # Calculate statistics
    print(f"\n7. Calculating final statistics...")
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


def _evaluate_solution(
    evaluator: Any,
    generated_solution: str,
    sample: Dict[str, Any],
    evaluator_type: str
) -> float:
    """
    Evaluate a generated solution against the gold solution.
    
    Args:
        evaluator: Evaluator instance
        generated_solution: The generated solution string
        sample: Sample dictionary containing gold solution
        evaluator_type: Type of evaluator ("test", "math", "poem")
    
    Returns:
        Reward score (float)
    """
    if evaluator_type == "math":
        # For math evaluator, use accuracy_reward directly
        solution = sample.get("solution", None)
        if solution is None:
            return 0.0
        
        # Handle empty generated solution
        if not generated_solution or not isinstance(generated_solution, str) or not generated_solution.strip():
            return 0.0
        
        # Format for accuracy_reward: expects [[{"content": "..."}], ...]
        formatted_completions = [[{"content": generated_solution}]]
        
        # accuracy_reward expects solution as a list (same length as completions)
        # Looking at math_evaluator.py line 71, it passes solution directly which can be a list or single value
        if isinstance(solution, str):
            solution_list = [solution]
        elif isinstance(solution, list):
            # Use first element if list has items, otherwise empty string
            solution_list = [solution[0]] if len(solution) > 0 else [""]
        else:
            solution_list = [str(solution)]
        
        try:
            rewards = accuracy_reward(formatted_completions, solution_list)
            # accuracy_reward may return a tensor or list
            if hasattr(rewards, '__len__') and len(rewards) > 0:
                reward_val = rewards[0] if isinstance(rewards, (list, tuple)) else rewards.item() if hasattr(rewards, 'item') else float(rewards)
                return float(reward_val)
            elif isinstance(rewards, (int, float)):
                return float(rewards)
            else:
                return 0.0
        except Exception as e:
            print(f"Error in accuracy_reward: {e}")
            print(f"  formatted_completions: {formatted_completions}")
            print(f"  solution_list: {solution_list}")
            print(f"  generated_solution type: {type(generated_solution)}, value: {repr(generated_solution[:100]) if generated_solution else 'None'}")
            import traceback
            traceback.print_exc()
            return 0.0
    elif evaluator_type == "poem":
        # For poem evaluator, use the evaluate method directly
        return evaluator.evaluate(generated_solution, **{k: v for k, v in sample.items() if k != "prompt"})
    elif evaluator_type == "test":
        # For test evaluator, use the evaluate method directly
        return evaluator.evaluate(generated_solution, **{k: v for k, v in sample.items() if k != "prompt"})
    else:
        # Default: return 0.0
        return 0.0


def _print_progress_summary(results: List[Dict[str, Any]]):
    """Print a progress summary of current results."""
    current_rewards_finetuned = [r["reward_finetuned"] for r in results]
    current_rewards_base = [r["reward_base"] for r in results]
    current_improvements = [r["improvement"] for r in results]
    
    current_stats = {
        "num_samples": len(results),
        "mean_reward_finetuned": sum(current_rewards_finetuned) / len(current_rewards_finetuned) if current_rewards_finetuned else 0.0,
        "mean_reward_base": sum(current_rewards_base) / len(current_rewards_base) if current_rewards_base else 0.0,
        "mean_improvement": sum(current_improvements) / len(current_improvements) if current_improvements else 0.0,
        "num_improved": sum(1 for imp in current_improvements if imp > 0),
        "num_worse": sum(1 for imp in current_improvements if imp < 0),
        "num_same": sum(1 for imp in current_improvements if imp == 0),
    }
    
    print(f"\n{'='*60}")
    print(f"Progress Update: {current_stats['num_samples']} samples completed")
    print(f"{'='*60}")
    print(f"Fine-tuned Rewriter - Mean reward: {current_stats['mean_reward_finetuned']:.4f}")
    print(f"Base Rewriter        - Mean reward: {current_stats['mean_reward_base']:.4f}")
    print(f"Mean improvement: {current_stats['mean_improvement']:.4f}")
    print(f"Improved: {current_stats['num_improved']} ({100*current_stats['num_improved']/current_stats['num_samples']:.1f}%) | "
          f"Worse: {current_stats['num_worse']} ({100*current_stats['num_worse']/current_stats['num_samples']:.1f}%) | "
          f"Same: {current_stats['num_same']} ({100*current_stats['num_same']/current_stats['num_samples']:.1f}%)")
    print(f"{'='*60}\n")


def _print_diagnostic_summary(recent_results: List[Dict[str, Any]]):
    """Print diagnostic information about the recent batch to identify patterns."""
    if not recent_results:
        return
    
    # Count empty outputs
    empty_finetuned_prompts = sum(1 for r in recent_results if not r.get("rewritten_prompt_finetuned", "").strip())
    empty_base_prompts = sum(1 for r in recent_results if not r.get("rewritten_prompt_base", "").strip())
    empty_finetuned_outputs = sum(1 for r in recent_results if not r.get("eval_output_finetuned", "").strip())
    empty_base_outputs = sum(1 for r in recent_results if not r.get("eval_output_base", "").strip())
    
    # Find "worse" cases (fine-tuned < base)
    worse_cases = [r for r in recent_results if r.get("improvement", 0) < 0]
    
    print(f"\n--- Diagnostic Summary (Last {len(recent_results)} samples) ---")
    print(f"Empty fine-tuned rewritten prompts: {empty_finetuned_prompts}/{len(recent_results)} ({100*empty_finetuned_prompts/len(recent_results):.1f}%)")
    print(f"Empty base rewritten prompts: {empty_base_prompts}/{len(recent_results)} ({100*empty_base_prompts/len(recent_results):.1f}%)")
    print(f"Empty fine-tuned eval outputs: {empty_finetuned_outputs}/{len(recent_results)} ({100*empty_finetuned_outputs/len(recent_results):.1f}%)")
    print(f"Empty base eval outputs: {empty_base_outputs}/{len(recent_results)} ({100*empty_base_outputs/len(recent_results):.1f}%)")
    print(f"Worse cases (fine-tuned < base): {len(worse_cases)}")
    
    if worse_cases:
        print(f"\nSample indices with worse performance:")
        for r in worse_cases[:5]:  # Show first 5
            idx = r.get("sample_index", "?")
            ft_reward = r.get("reward_finetuned", 0.0)
            base_reward = r.get("reward_base", 0.0)
            ft_prompt_empty = not r.get("rewritten_prompt_finetuned", "").strip()
            base_prompt_empty = not r.get("rewritten_prompt_base", "").strip()
            print(f"  Sample {idx}: FT={ft_reward:.4f}, Base={base_reward:.4f} | "
                  f"FT_prompt_empty={ft_prompt_empty}, Base_prompt_empty={base_prompt_empty}")
    
    # Check if there's a pattern in sample indices
    if len(worse_cases) >= 2:
        worse_indices = [r.get("sample_index", -1) for r in worse_cases]
        worse_indices.sort()
        if len(worse_indices) >= 2:
            gaps = [worse_indices[i+1] - worse_indices[i] for i in range(len(worse_indices)-1)]
            print(f"\nGaps between worse-case sample indices: {gaps}")
            if len(set(gaps)) == 1:
                print(f"  → Consistent gap of {gaps[0]} detected! This suggests a pattern.")
    print("--- End Diagnostic Summary ---\n")


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
    
    # Handle DataParallel wrapper
    model = _get_model(inference_client.model)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=toks.get("input_ids"),
            attention_mask=toks.get("attention_mask"),
            max_new_tokens=max_tokens,
            do_sample=False,
            use_cache=True,
        )
    
    # Extract only the newly generated tokens
    generated_tokens = outputs[0]
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
        default=512,
        help="Maximum tokens for inference (task completion)"
    )
    
    # Performance options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing (1 = sequential, >1 = batched, much faster). Recommended: 4-16"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (1 = single GPU, >1 = multi-GPU via DataParallel)"
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
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        output_file=args.output_file,
    )
    
    print("\n✓ Evaluation completed successfully!")


if __name__ == "__main__":
    main()

