from abc import ABC, abstractmethod

class BaseParser(ABC):
    def __init__(self, dataset_name: str, meta_prompt: str):
        self.dataset_name = dataset_name
        self.meta_prompt = meta_prompt
        self.dataset = None
    
    @abstractmethod
    def download_dataset(self):
        raise NotImplementedError

    @abstractmethod
    def parse(self):
        """
        Parse the dataset into a compatible format for the GRPOTrainer.

        The GRPOTrainer expects a dataset in the following format at a minimum:
            - prompt: The prompt to be passed to the model.

        """
        raise NotImplementedError