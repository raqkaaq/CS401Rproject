from typing import List, Dict, Any, Optional
import os
from .base_parser import BaseParser
from datasets import load_dataset


class PoemParser(BaseParser):
    """
    A parser that loads poem prompts from the checkai/instruction-poems dataset.
    Uses the INSTRUCTION field as the prompt with the system prompt attached.
    """
    
    def __init__(self, dataset_name: str = "poem_dataset", meta_prompt: str = "", 
                 num_samples: int = 50000):
        """
        Initialize the poem parser.
        
        Args:
            dataset_name: Name of the poem dataset (not used, kept for compatibility)
            meta_prompt: Meta prompt to use as system prompt
            num_samples: Number of poem samples to use (default: 50000)
        """
        super().__init__(dataset_name, meta_prompt, num_samples)
        self.num_samples = num_samples
        self.dataset = None
    
    def download_dataset(self):
        """
        Download the instruction-poems dataset from HuggingFace.
        The dataset will be cached in ~/.cache/huggingface/datasets/ for future use.
        """
        dataset_name = "checkai/instruction-poems"
        
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
        else:
            print(f"Warning: Cache directory {cache_dir} does not exist")
        
        try:
            # Try to load from cache first (works even without internet)
            # download_mode="reuse_cache_if_exists" will use cache if available, 
            # but will still try to connect to internet to check for updates
            # We need to catch network errors and retry with offline mode
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
                    # Try loading with download_mode="reuse_cache_if_exists"
                    try:
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
        Parse the poem dataset into a compatible format for the GRPOTrainer.
        
        The dataset format will be:
        [
            {
                "prompt": [
                    {
                        "content": "<meta_prompt>",
                        "role": "system"
                    },
                    {
                        "content": "<INSTRUCTION from dataset>",
                        "role": "user"
                    }
                ],
                # Additional fields from original dataset
            },
            ...
        ]
        
        Returns:
            The parsed dataset in GRPOTrainer format.
        """
        if self.dataset is None:
            self.download_dataset()
        
        # Determine how many samples to use and also randomize the samples
        # Note: self.dataset is already a Dataset (not DatasetDict) because split="train" was used
        self.dataset = self.dataset.shuffle(seed=42)
        num_to_parse = self.num_samples if self.num_samples is not None else len(self.dataset)
        num_to_parse = min(num_to_parse, len(self.dataset))
        
        parsed_data = []
        for i in range(num_to_parse):
            sample = self.dataset[i]
            
            # Use the INSTRUCTION field as the prompt content
            # Try different possible field names (case-insensitive)
            instruction = None
            for field_name in ["INSTRUCTION", "instruction", "Instruction", "prompt", "Prompt"]:
                if field_name in sample:
                    instruction = str(sample[field_name])
                    break
            
            if instruction is None:
                # If no instruction field found, try to get the first string field
                for key, value in sample.items():
                    if isinstance(value, str) and value.strip():
                        instruction = value
                        print(f"Warning: No INSTRUCTION field found, using '{key}' field instead")
                        break
            
            if instruction is None:
                print(f"Warning: Sample {i} has no usable instruction field. Skipping.")
                continue
            
            # Create prompt_messages with system prompt and instruction
            prompt_messages = [
                {
                    "content": self.meta_prompt,
                    "role": "system"
                },
                {
                    "content": instruction,
                    "role": "user"
                }
            ]
            
            parsed_sample = {
                "prompt": prompt_messages
            }
            
            # Include other fields from the original sample if they exist
            for key, value in sample.items():
                if key not in ["INSTRUCTION", "instruction", "Instruction", "prompt", "Prompt"]:
                    parsed_sample[key] = value
            
            parsed_data.append(parsed_sample)
        
        print(f"Parsed {len(parsed_data)} samples from poem dataset")
        if parsed_data:
            print(f"Sample parsed structure (first item): {parsed_data[0]}")
        return parsed_data

