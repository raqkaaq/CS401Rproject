from typing import List, Dict, Any, Optional
import re
from .base_parser import BaseParser
from datasets import load_dataset


class EasyMathParser(BaseParser):
    """
    Parser for the GSM8K dataset from OpenAI.
    The dataset contains "question" and "answer" fields.
    """
    
    def __init__(self, dataset_name: str = "openai/gsm8k", meta_prompt: str = "", 
                 num_samples: int = None):
        """
        Initialize the EasyMath parser for GSM8K dataset.
        
        Args:
            dataset_name: Name of the dataset (default: "openai/gsm8k")
            meta_prompt: Meta prompt to use (optional)
            num_samples: Number of samples to parse (None for all)
        """
        super().__init__(dataset_name, meta_prompt, num_samples)
        self.dataset = None
    
    def download_dataset(self):
        """
        Download the GSM8K dataset from HuggingFace.
        The dataset will be cached in ~/.cache/huggingface/datasets/ for future use.
        """
        import os
        dataset_name = "openai/gsm8k"
        
        # Set cache directory via environment variable (datasets library uses this)
        cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Set environment variable for datasets library
        os.environ["HF_DATASETS_CACHE"] = cache_dir
        
        print(f"Loading dataset: {dataset_name}")
        print(f"Cache directory: {cache_dir}")
        print(f"HF_DATASETS_CACHE env: {os.environ.get('HF_DATASETS_CACHE', 'not set')}")
        
        # Check if cache directory exists and has content
        if os.path.exists(cache_dir):
            cache_contents = os.listdir(cache_dir)
            print(f"Cache directory exists with {len(cache_contents)} items")
            # Look for GSM8K-related directories
            gsm8k_dirs = [d for d in cache_contents if "gsm8k" in d.lower() or "openai" in d.lower()]
            if gsm8k_dirs:
                print(f"Found GSM8K cache directories: {gsm8k_dirs}")
        else:
            print(f"Warning: Cache directory {cache_dir} does not exist")
        
        try:
            # Load the dataset - GSM8K has "train" and "test" splits
            # We'll use "train" by default, but the split can be specified in dataset_name
            # GSM8K requires a config name ('main' or 'socratic')
            if ":" in self.dataset_name:
                parts = self.dataset_name.split(":", 1)
                if len(parts) == 2:
                    base_name, split = parts
                    # Check if base_name is gsm8k and needs config
                    if base_name == "openai/gsm8k":
                        self.dataset = load_dataset(
                            base_name,
                            "main",
                            split=split,
                            download_mode="reuse_cache_if_exists"
                        )
                    else:
                        self.dataset = load_dataset(
                            base_name,
                            split=split,
                            download_mode="reuse_cache_if_exists"
                        )
                else:
                    # Format: dataset:config:split or just dataset:config
                    raise ValueError(f"Invalid dataset_name format: {self.dataset_name}")
            else:
                # Default to train split
                # Check if it's gsm8k and needs config
                if dataset_name == "openai/gsm8k":
                    self.dataset = load_dataset(
                        dataset_name,
                        "main",
                        split="train",
                        download_mode="reuse_cache_if_exists"
                    )
                else:
                    self.dataset = load_dataset(
                        dataset_name,
                        split="train",
                        download_mode="reuse_cache_if_exists"
                    )
            print(f"✓ Loaded dataset: {len(self.dataset)} samples")
        except Exception as e:
            error_str = str(e)
            print(f"Error loading dataset: {error_str}")
            
            # Check if it's a network error - if so, try offline mode
            if "Network" in error_str or "unreachable" in error_str or "Connection" in error_str:
                print("Network error detected. Attempting to load from cache in offline mode...")
                try:
                    from datasets import load_from_disk
                    import glob
                    
                    # Look for cached dataset files
                    cache_pattern = os.path.join(cache_dir, "*", "*gsm8k*", "*")
                    cache_files = glob.glob(cache_pattern)
                    if cache_files:
                        print(f"Found cache files: {cache_files[:3]}...")
                    
                    # Try loading with trust_local_files
                    try:
                        if ":" in self.dataset_name:
                            parts = self.dataset_name.split(":", 1)
                            if len(parts) == 2:
                                base_name, split = parts
                                # Check if base_name is gsm8k and needs config
                                if base_name == "openai/gsm8k":
                                    self.dataset = load_dataset(
                                        base_name,
                                        "main",
                                        split=split,
                                        trust_remote_code=True,
                                        download_mode="reuse_cache_if_exists"
                                    )
                                else:
                                    self.dataset = load_dataset(
                                        base_name,
                                        split=split,
                                        trust_remote_code=True,
                                        download_mode="reuse_cache_if_exists"
                                    )
                        else:
                            # Check if it's gsm8k and needs config
                            if dataset_name == "openai/gsm8k":
                                self.dataset = load_dataset(
                                    dataset_name,
                                    "main",
                                    split="train",
                                    trust_remote_code=True,
                                    download_mode="reuse_cache_if_exists"
                                )
                            else:
                                self.dataset = load_dataset(
                                    dataset_name,
                                    split="train",
                                    trust_remote_code=True,
                                    download_mode="reuse_cache_if_exists"
                                )
                        print(f"✓ Loaded from cache: {len(self.dataset)} samples")
                        return
                    except:
                        pass
                    
                    # If that fails, the cache might not be complete
                    raise RuntimeError(
                        f"Dataset '{dataset_name}' not found in cache and cannot download (no internet on compute node).\n"
                        f"Cache directory checked: {cache_dir}\n"
                        f"Cache contents: {os.listdir(cache_dir) if os.path.exists(cache_dir) else 'N/A'}\n"
                        f"\n"
                        f"Please verify the dataset was downloaded correctly on the login node:\n"
                        f"  python -c \"from datasets import load_dataset; d=load_dataset('{dataset_name}', split='train'); print(f'Loaded {{len(d)}} samples')\"\n"
                        f"Or run: ./download_datasets.sh\n"
                        f"\n"
                        f"Then verify cache exists: ./check_dataset_cache.sh"
                    ) from e
                except RuntimeError:
                    raise
                except Exception as e2:
                    raise RuntimeError(
                        f"Failed to load dataset from cache: {e2}\n"
                        f"Please download on login node: ./download_datasets.sh"
                    ) from e2
            # Re-raise other errors
            raise

    
    def parse(self) -> List[Dict[str, Any]]:
        """
        Parse the GSM8K dataset into a compatible format for the GRPOTrainer.
        
        The dataset format will be:
        [
            {
                "prompt": [
                    {
                        "content": "Question text...",
                        "role": "user"
                    }
                ],
                "answer": "Step-by-step solution...",
                "question": "Question text..."
            },
            ...
        ]

        Returns:
            The parsed dataset in GRPOTrainer format.
        """
        if self.dataset is None:
            self.download_dataset()
        
        # Determine how many samples to use and also randomize the samples
        # Note: self.dataset is already a Dataset (not DatasetDict) because split was used
        self.dataset = self.dataset.shuffle(seed=42)
        num_to_parse = self.num_samples if self.num_samples is not None else len(self.dataset)
        
        parsed_data = []
        for i in range(min(num_to_parse, len(self.dataset))):
            sample = self.dataset[i]
            
            # Extract question and answer from GSM8K format
            question = str(sample.get("question", ""))
            answer = str(sample.get("answer", ""))

            # Extract solution number using regex to find "####" followed by a number
            solution_number = None
            solution_match = re.search(r'####\s*(\d+(?:\.\d+)?)', answer)
            if solution_match:
                # Wrap numeric solution in \boxed{} for proper LaTeX parsing by accuracy_reward
                # This ensures math_verify can parse and verify the solution correctly
                solution_number = solution_match.group(1)

            # Build prompt messages with optional meta prompt
            if self.meta_prompt:
                prompt_messages = [
                    {
                        "content": self.meta_prompt + "\n\n" + question,
                        "role": "user"
                    }
                ]
            else:
                prompt_messages = [
                    {
                        "content": question,
                        "role": "user"
                    }
                ]
            
            parsed_sample = {
                "prompt": prompt_messages,
                "original_question": question,
                "answer": answer,
                "solution": solution_number if solution_number else answer
            }
            
            # Include other fields from the original sample if they exist
            for key, value in sample.items():
                if key not in ["prompt", "question", "answer"]:
                    parsed_sample[key] = value
            
            parsed_data.append(parsed_sample)
        
        print(f"Parsed {len(parsed_data)} samples from GSM8K dataset")
        if parsed_data:
            print(f"Sample parsed structure (first item keys): {list(parsed_data[0].keys())}")
            print(f"Sample question: {parsed_data[0].get('question', '')[:100]}...")
        return parsed_data

