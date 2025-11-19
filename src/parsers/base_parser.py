from abc import ABC, abstractmethod

class BaseParser(ABC):
    def __init__(self, dataset_name: str, meta_prompt: str, num_samples: int = None):
        self.dataset_name = dataset_name
        self.meta_prompt = meta_prompt
        self.dataset = None
        self.num_samples = num_samples

    @abstractmethod
    def download_dataset(self):
        raise NotImplementedError

    @abstractmethod
    def parse(self, datapoint_size: int = None):
        """
        Parse the dataset into a compatible format for the GRPOTrainer.

        The GRPOTrainer expects a dataset in the following format at a minimum:
            - prompt: The prompt to be passed to the model.

            Here is an example:

            https://huggingface.co/docs/trl/main/en/grpo_trainer

            [
            {
            "content": "Let \\( X_l = \\{f \\in C([0,l], \\mathbf{R}), 0 \\leq f(x) \\leq 2, \\forall x \\in [0,l]\\}\\) and let \\( T: X \\rightarrow C([0,l], \\mathbf{R})\\) be defined by \\((T(f))(x) = \\int_0^{x} f(t)dt\\). Determine the largest number \\( l_0 \\) such that \\( T \\) maps \\( X_{l_0} \\) into \\( X_{l_0} \\).",
            "role": "user"
            }
            ]

        Args:
            datapoint_size: The size of the datapoint to parse. If None, parse the entire dataset.

        Returns:
            The parsed dataset.
        """
        raise NotImplementedError