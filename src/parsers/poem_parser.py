from typing import List, Dict, Any, Optional
import random
import os
from .base_parser import BaseParser


class PoemParser(BaseParser):
    """
    A parser that generates poem prompts in the form "Write me a poem about X and Y".
    Uses nouns from noun_dict.txt to generate combinations.
    """
    
    def __init__(self, dataset_name: str = "poem_dataset", meta_prompt: str = "", 
                 num_samples: int = 50000):
        """
        Initialize the poem parser.
        
        Args:
            dataset_name: Name of the poem dataset
            meta_prompt: Meta prompt to use (optional)
            num_samples: Number of poem samples to generate (default: 50000)
        """
        super().__init__(dataset_name, meta_prompt, num_samples)
        self.num_samples = num_samples
        self.dataset = None
        self.nouns = []
        self._load_nouns()
    
    def _load_nouns(self):
        """Load nouns from noun_dict.txt file."""
        # Get the directory where this file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        noun_file = os.path.join(current_dir, "noun_dict.txt")
        
        try:
            with open(noun_file, 'r', encoding='utf-8') as f:
                self.nouns = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(self.nouns)} nouns from {noun_file}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find noun_dict.txt at {noun_file}. "
                "Please ensure the file exists in the parsers directory."
            )
    
    def download_dataset(self):
        """
        Generate synthetic poem prompt data.
        Creates prompts in the form "Write me a poem about X and Y" where X and Y are nouns.
        """
        if not self.nouns:
            self._load_nouns()
        
        # Generate unique combinations
        # Use a set to track combinations we've already used
        used_combinations = set()
        dataset_samples = []
        
        # Set random seed for reproducibility
        random.seed(42)
        
        # Generate samples
        max_attempts = self.num_samples * 10  # Prevent infinite loop
        attempts = 0
        
        while len(dataset_samples) < self.num_samples and attempts < max_attempts:
            attempts += 1
            
            # Randomly select two nouns (allowing same noun twice for variety)
            noun1 = random.choice(self.nouns)
            noun2 = random.choice(self.nouns)
            
            # Create a canonical representation of the pair (order-independent)
            # This ensures "X and Y" is the same as "Y and X"
            pair = tuple(sorted([noun1, noun2]))
            
            # Skip if we've already used this combination
            if pair in used_combinations:
                continue
            
            used_combinations.add(pair)
            
            # Create the prompt
            prompt = f"Write me a haiku about {noun1} and {noun2}"
            
            dataset_samples.append({
                "prompt": prompt,
                "noun1": noun1,
                "noun2": noun2
            })
        
        if len(dataset_samples) < self.num_samples:
            print(f"Warning: Only generated {len(dataset_samples)} unique combinations "
                  f"out of {self.num_samples} requested. "
                  f"Total possible unique pairs: {len(self.nouns) * (len(self.nouns) + 1) // 2}")
        
        self.dataset = {
            "train": dataset_samples
        }
        print(f"Created synthetic poem dataset with {len(dataset_samples)} samples")
    
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
                        "content": "Write me a poem about X and Y",
                        "role": "user"
                    }
                ],
                # Additional fields from original dataset (noun1, noun2, etc.)
            },
            ...
        ]
        
        Returns:
            The parsed dataset in GRPOTrainer format.
        """
        if self.dataset is None:
            self.download_dataset()
        
        # Determine how many samples to use
        num_to_parse = self.num_samples if self.num_samples is not None else len(self.dataset["train"])
        num_to_parse = min(num_to_parse, len(self.dataset["train"]))
        
        parsed_data = []
        for i in range(num_to_parse):
            sample = self.dataset["train"][i]
            
            prompt_content = sample["prompt"]
            
            # Create prompt_messages in the format shown in math_parser.py
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
        
        print(f"Parsed {len(parsed_data)} samples from poem dataset")
        if parsed_data:
            print(f"Sample parsed structure (first item): {parsed_data[0]}")
        return parsed_data

