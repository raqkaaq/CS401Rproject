from typing import List, Dict, Any, Optional
from .base_parser import BaseParser


class TestParser(BaseParser):
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
        # Create synthetic test data
        self.dataset = {
            "train": [
                {
                    "prompt": f"Test question {i}: What is {i} + {i+1}?",
                    "answer": f"{i + (i+1)}",
                    "gold": f"{i + (i+1)}"
                }
                for i in range(1, self.num_samples + 1)
            ]
        }
        print(f"Created synthetic test dataset with {self.num_samples} samples")
    
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
        
        # Determine how many samples to use
        num_to_parse = datapoint_size if datapoint_size is not None else len(self.dataset["train"])
        num_to_parse = min(num_to_parse, len(self.dataset["train"]))
        
        parsed_data = []
        for i in range(num_to_parse):
            sample = self.dataset["train"][i]
            
            # Create the prompt in the expected format
            prompt_content = sample["prompt"]
            if self.meta_prompt:
                prompt_content = f"{self.meta_prompt}\n\n{prompt_content}"
            
            parsed_sample = {
                "prompt": [
                    {
                        "content": prompt_content,
                        "role": "user"
                    }
                ]
            }
            
            # Include other fields from the original sample if they exist
            for key, value in sample.items():
                if key != "prompt":
                    parsed_sample[key] = value
            
            parsed_data.append(parsed_sample)
        
        print(f"Parsed {len(parsed_data)} samples from test dataset")
        return parsed_data

