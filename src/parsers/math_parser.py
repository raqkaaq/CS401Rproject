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
        try:
            self.dataset = load_dataset("trl-lib/DeepMath-103K", split="train")
            print(f"Downloaded dataset: {len(self.dataset)} samples")
        except Exception as e:
            if "Network" in str(e) or "unreachable" in str(e) or "Connection" in str(e):
                raise RuntimeError(
                    "Dataset not found in cache and cannot download (no internet on compute node). "
                    "Please pre-download datasets on login node. The prepare_environment.sh script "
                    "can do this, or run: python -c \"from datasets import load_dataset; load_dataset('trl-lib/DeepMath-103K', split='train')\""
                ) from e
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

