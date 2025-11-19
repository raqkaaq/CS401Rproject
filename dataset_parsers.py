from __future__ import annotations
from typing import Optional, List, Tuple, Dict
from datasets import load_dataset, Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

class BaseDatasetParser:
    def __init__(
        self,
        dataset_name: str,
        tokenizer: PreTrainedTokenizerBase,
        eval_split: Optional[str] = None,
        max_seq_len: int = 512,
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.eval_split = eval_split
        self.max_seq_len = min(max_seq_len, tokenizer.model_max_length or max_seq_len)
        self.raw: Optional[DatasetDict] = None
        self.tokenized_train: Optional[Dataset] = None
        self.tokenized_eval: Optional[Dataset] = None
        self.raw_train_examples: List[Dict] = []

    def load(self) -> DatasetDict:
        # Allow optional config after a colon: dataset_name:config (e.g., "openai/gsm8k:main")
        if ":" in self.dataset_name and "/" in self.dataset_name:
            base, config = self.dataset_name.split(":")
            self.raw = load_dataset(base, config)
        else:
            self.raw = load_dataset(self.dataset_name)
        return self.raw

    # ----- override points -----
    def normalize_row(self, row: Dict) -> Dict:
        # Provide "question", "answer", "task"
        q = (row.get("question") or row.get("prompt") or row.get("input") or "").strip()
        a = (row.get("answer") or row.get("final_answer") or row.get("output") or "").strip()
        return {"question": q, "answer": a, "task": row.get("task", "task")}

    def build_query_text(self, row: Dict) -> str:
        # The text the policy sees to produce a rewritten instruction
        return row.get("question", "")

    # ----- internal helpers -----
    def _tokenize_batch(self, batch: Dict) -> Dict:
        enc = self.tokenizer(
            batch["query_text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_len,
        )
        return enc

    def prepare(self) -> Tuple[Dataset, Optional[Dataset], List[Dict]]:
        if self.raw is None:
            self.load()
        assert isinstance(self.raw, DatasetDict)

        # split selection
        eval_split = self.eval_split
        if eval_split is None:
            for cand in ("validation", "test", "eval"):
                if cand in self.raw:
                    eval_split = cand
                    break

        train_ds: Dataset = self.raw["train"]
        eval_ds: Optional[Dataset] = self.raw.get(eval_split) if eval_split else None

        # 1) normalize to required keys
        train_ds = train_ds.map(self.normalize_row)
        if eval_ds is not None:
            eval_ds = eval_ds.map(self.normalize_row)

        # 2) build query_text (what PPO model will read)
        def add_query_text(b):
            return {
                "query_text": [
                    self.build_query_text({"question": q, "answer": a})
                    for q, a in zip(b["question"], b["answer"])
                ]
            }
        train_ds = train_ds.map(add_query_text, batched=True)
        if eval_ds is not None:
            eval_ds = eval_ds.map(add_query_text, batched=True)

        # 3) stash raw training examples for reward computation (keep text fields)
        self.raw_train_examples = train_ds.select(range(len(train_ds))).to_list()

        # 4) tokenize queries (pad to fixed length to avoid collate errors)
        train_ds = train_ds.map(self._tokenize_batch, batched=True, remove_columns=[])
        if eval_ds is not None:
            eval_ds = eval_ds.map(self._tokenize_batch, batched=True, remove_columns=[])

        # 5) DROP all non-tensor columns so the collator doesn't see strings
        keep_cols = {"input_ids", "attention_mask"}
        drop_train = [c for c in train_ds.column_names if c not in keep_cols]
        train_ds = train_ds.remove_columns(drop_train)
        if eval_ds is not None:
            drop_eval = [c for c in eval_ds.column_names if c not in keep_cols]
            eval_ds = eval_ds.remove_columns(drop_eval)

        # 6) set format for PyTorch
        train_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
        if eval_ds is not None:
            eval_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

        self.tokenized_train = train_ds
        self.tokenized_eval = eval_ds
        return self.tokenized_train, self.tokenized_eval, self.raw_train_examples


class GSM8KDatasetParser(BaseDatasetParser):
    def normalize_row(self, row: Dict) -> Dict:
        q = (row.get("question") or "").strip()
        a = (row.get("answer") or "").strip()
        return {"question": q, "answer": a, "task": "math"}

    def build_query_text(self, row: Dict) -> str:
        q = row.get("question", "")
        return (
            "Rewrite an instruction that will guide a small language model to solve the following math word problem. "
            "Emphasize concise step-by-step reasoning, avoid giving the final answer, and discourage unnecessary verbosity.\n\n"
            f"Problem:\n{q}\n\nRewritten instruction:"
        )


def get_parser(dataset_name: str, tokenizer: PreTrainedTokenizerBase) -> BaseDatasetParser:
    base = dataset_name.split(":")[0]
    if base in ("openai/gsm8k", "gsm8k"):
        return GSM8KDatasetParser(dataset_name, tokenizer, eval_split="test", max_seq_len=512)
    return BaseDatasetParser(dataset_name, tokenizer, max_seq_len=512)