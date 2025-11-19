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
        Create synthetic test data instead of downloading.
        In a real implementation, this would download the dataset.
        """
        self.dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

    
    def parse(self, datapoint_size: Optional[int] = None) -> List[Dict[str, Any]]:
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
        
        Args:
            datapoint_size: The number of datapoints to parse. If None, parse all samples.
        
        Returns:
            The parsed dataset in GRPOTrainer format.
        """
        if self.dataset is None:
            self.download_dataset()
        
        # Determine how many samples to use and also randomize the samples
        # Note: self.dataset is already a Dataset (not DatasetDict) because split="train" was used
        self.dataset = self.dataset.shuffle(seed=42)  # TODO: Make this a random seed
        num_to_parse = datapoint_size if datapoint_size is not None else len(self.dataset)
        num_to_parse = min(num_to_parse, len(self.dataset))
        
        parsed_data = []
        for i in range(num_to_parse):
            sample = self.dataset[i]
            
            # Handle the prompt - it may already be in the correct format (list of message dicts)
            # or it might be a string. Check the structure.
            if isinstance(sample["prompt"], list):
                # If prompt is already a list of message dicts, use it directly
                prompt_messages = sample["prompt"]
                # If it's a list of lists (nested), take the first inner list
                if prompt_messages and isinstance(prompt_messages[0], list):
                    prompt_messages = prompt_messages[0]
            else:
                # If prompt is a string, convert it to the expected format
                prompt_content = str(sample["prompt"])
                if self.meta_prompt:
                    prompt_content = f"{self.meta_prompt}\n\n{prompt_content}"
                prompt_messages = [
                    {
                        "content": prompt_content,
                        "role": "user"
                    }
                ]
            
            # Apply meta_prompt if provided and prompt is in message format
            if self.meta_prompt and isinstance(sample["prompt"], list):
                # Prepend meta_prompt to the first message's content
                if prompt_messages and len(prompt_messages) > 0:
                    original_content = prompt_messages[0].get("content", "")
                    prompt_messages[0]["content"] = f"{self.meta_prompt}\n\n{original_content}"
            
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

