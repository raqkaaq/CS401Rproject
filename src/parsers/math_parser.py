from typing import List, Dict, Any, Optional
from .base_parser import BaseParser
from datasets import load_dataset


class MathParser(BaseParser):
    """
    A simple test parser for testing purposes.
    This parser can be used to test the parsing pipeline without
    implementing complex dataset parsing logic.
    """
    
    def __init__(self, dataset_name: str = "test_dataset", meta_prompt: str = "", 
                 num_samples: int = 10):
        """
        Initialize the test parser.
        
        Args:
            dataset_name: Name of the test dataset
            meta_prompt: Meta prompt to use (optional)
            num_samples: Number of test samples to generate
        """
        super().__init__(dataset_name, meta_prompt)
        self.num_samples = num_samples
        self.dataset = None
    
    def download_dataset(self):
        """
        Download the dataset from HuggingFace.
        The dataset will be cached in ~/.cache/huggingface/datasets/ for future use.
        """
        import os
        dataset_name = "trl-lib/DeepMath-103K"
        
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
            # Look for DeepMath-related directories
            deepmath_dirs = [d for d in cache_contents if "DeepMath" in d or "trl-lib" in d]
            if deepmath_dirs:
                print(f"Found DeepMath cache directories: {deepmath_dirs}")
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
                    # Try with offline mode - this should only use cache
                    # Note: This might not work if the cache metadata needs internet
                    # But it's worth trying
                    from datasets import load_from_disk
                    import glob
                    
                    # Look for cached dataset files
                    cache_pattern = os.path.join(cache_dir, "*", "*DeepMath*", "*")
                    cache_files = glob.glob(cache_pattern)
                    if cache_files:
                        print(f"Found cache files: {cache_files[:3]}...")
                    
                    # Try loading with download_mode=None and trust_local_files=True
                    # This is a newer parameter that might help
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
        Parse the test dataset into a compatible format for the GRPOTrainer.
        
        The dataset format will be:
        [
            {
                "prompt": [
                    {
                        "content": "Test question 1: What is 1 + 2?",
                        "role": "user"
                    }
                ],
                # Additional fields from original dataset (answer, gold, etc.)
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
        self.dataset = self.dataset.shuffle(seed=42)  # TODO: Make this a random seed
        num_to_parse = self.num_samples if self.num_samples is not None else len(self.dataset)
        
        parsed_data = []
        for i in range(num_to_parse):
            sample = self.dataset[i]
            
            prompt_content = str(sample["prompt"])

            prompt_messages = [
                {
                    "content": self.meta_prompt,
                    "role": "system"
                },
                {
                    "content": prompt_content,
                    "role": "user"
                }
            ]
            
            parsed_sample = {
                "prompt": prompt_messages
            }
            
            # Include other fields from the original sample if they exist
            for key, value in sample.items():
                if key != "prompt":
                    parsed_sample[key] = value
            
            parsed_data.append(parsed_sample)
        
        print(f"Parsed {len(parsed_data)} samples from math dataset")
        if parsed_data:
            print(f"Sample parsed structure (first item): {parsed_data[0]}")
        return parsed_data

