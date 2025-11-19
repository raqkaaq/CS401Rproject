from abc import ABC, abstractmethod

class BaseParser(ABC):
    def __init__(self, dataset_name: str, meta_prompt: str):
        self.dataset_name = dataset_name
        self.meta_prompt = meta_prompt
        self.dataset = None
    
    def download_dataset(self):
        raise NotImplementedError

    def parse(self):
        raise NotImplementedError