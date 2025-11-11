"""datasets.py

Helpers to prepare datasets for PRewrite finetuning.

This module provides:
- `prepare_hf_dataset_to_jsonl(hf_id, split_map, output_path)` to download a HuggingFace
  dataset and write a JSONL file with the fields required by the trainer.
- `JsonlRewriteDataset` and `RewriteExample` classes used by `finetune.py` and `inference.py`.
- `build_meta_prompt` and `format_rewriter_query` helpers used to construct rewriter queries.

The JSONL format expected (one JSON object per line) contains at least:
  - instruction: the initial instruction string (t)
  - input: optional task input (x)
  - output: the ground-truth output (y)
  - task: a short task type identifier like "classification", "qa", "math"

This file intentionally keeps utilities small and dependency-light. It uses `datasets`
from HuggingFace when asked to prepare a dataset.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Dict, Any

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover - optional dependency
    load_dataset = None


@dataclass
class RewriteExample:
    instruction: str
    input: Optional[str]
    output: str
    task: str = "classification"

    def build_task_prompt(self, rewritten_instruction: str) -> str:
        """Construct the task prompt passed to the task LLM.

        Default template: "{rewritten_instruction}\nInput: {input}" if input exists,
        otherwise just the rewritten_instruction.
        """
        if self.input and len(self.input.strip()) > 0:
            return f"{rewritten_instruction}\nInput: {self.input}"
        return rewritten_instruction


class JsonlRewriteDataset:
    """Simple JSONL-backed dataset wrapper.

    Each line in the JSONL should be a dict with keys: instruction, input (opt), output, task (opt).
    """

    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset JSONL not found: {self.path}")

    def iter(self) -> Iterator[RewriteExample]:
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                yield RewriteExample(
                    instruction=record.get("instruction", ""),
                    input=record.get("input", ""),
                    output=record.get("output", ""),
                    task=record.get("task", "classification"),
                )


def build_meta_prompt(family: str = "generic") -> str:
    """Return a small meta prompt used to instruct the rewriter LLM.

    The paper uses a generic meta prompt. We provide a minimal default and keep
    this function so other meta prompts can be added centrally.
    """
    fam = (family or "generic").lower()
    if fam == "generic":
        return (
            "Rewrite the following instruction via rephrasing and/or adding specific requirements. "
            "Add instructions which would be helpful to solve the problem correctly. Output the new instruction only."
        )
    # fallback
    return (
        "Rewrite the following instruction via rephrasing and/or adding specific requirements. "
        "Output the new instruction only."
    )


def format_rewriter_query(meta_prompt: str, instruction: str) -> str:
    """Format the text sent to the rewriter LLM.

    Matches Equation (1) from the paper: LLM_R("{m}\nInstruction: {p}")
    """
    return f"{meta_prompt}\nInstruction: {instruction}"


def prepare_hf_dataset_to_jsonl(hf_id: str, split: str = "train", limit: Optional[int] = None, output_path: str = "dataset.jsonl") -> str:
    """Download a HuggingFace dataset and write a simple JSONL for finetuning.

    This function attempts to map common dataset columns into the minimal
    instruction/input/output fields. For many datasets a simple mapping is used;
    for custom datasets you should create a JSONL manually.

    Returns the path to the written JSONL file.
    """
    if load_dataset is None:
        raise RuntimeError("`datasets` library not available. Install `datasets` to use this helper.")
    ds = load_dataset(hf_id, split=split)
    out_path = Path(output_path)
    with out_path.open("w", encoding="utf-8") as fout:
        cnt = 0
        for row in ds:
            # best-effort mapping heuristics
            record: Dict[str, Any] = {}
            # instruction: try common keys
            if "instruction" in row:
                record["instruction"] = row["instruction"]
            elif "prompt" in row:
                record["instruction"] = row["prompt"]
            else:
                # fallback: create a generic instruction depending on dataset
                record["instruction"] = "Perform the task described."

            # input: try common inputs
            record["input"] = row.get("input", row.get("text", row.get("article", "")))

            # output/label
            if "label" in row:
                record["output"] = str(row["label"])
            elif "answer" in row:
                record["output"] = row["answer"]
            elif "label_text" in row:
                record["output"] = row["label_text"]
            else:
                # some datasets use different naming; try to take first string field not used above
                out_val = None
                for k, v in row.items():
                    if k in {"instruction", "prompt", "input", "text", "article"}:
                        continue
                    if isinstance(v, str) and v:
                        out_val = v
                        break
                record["output"] = out_val or ""

            # task: best-effort
            record["task"] = "classification" if isinstance(record["output"], str) and record["output"].isdigit() is False else "classification"

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            cnt += 1
            if limit is not None and cnt >= limit:
                break
    return str(out_path)


__all__ = [
    "RewriteExample",
    "JsonlRewriteDataset",
    "build_meta_prompt",
    "format_rewriter_query",
    "prepare_hf_dataset_to_jsonl",
]
