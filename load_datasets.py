"""datasets.py

Minimal dataset helpers for PRewrite that use pandas DataFrame as the
primary in-memory representation. This keeps things simple: you can load
CSV/JSONL into pandas, or pass a DataFrame directly to the dataset wrapper.

Exports kept compatible with `finetune.py`:
- `RewriteExample` dataclass
- `PRewriteDataset` which can be created from a file path or a DataFrame
- `build_meta_prompt` and `format_rewriter_query`
- `prepare_df_to_jsonl(df, output_path)` helper to write a DataFrame to JSONL
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Any

try:
    import pandas as pd
except Exception as e:  # pragma: no cover - user must install pandas
    raise RuntimeError("Please 'pip install pandas' to use the dataset helpers") from e


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
        if self.input and len(str(self.input).strip()) > 0:
            return f"{rewritten_instruction}\nInput: {self.input}"
        return rewritten_instruction


# Note: JsonlRewriteDataset removed by user request. Use PRewriteDataset instead.


class PRewriteDataset:
    """Dataset wrapper that organizes data like the PRewrite paper.

    Accepts a pandas DataFrame or a file path. If the source contains an
    explicit 'split' column it will use those splits. Otherwise, you can
    provide `dev_fraction` and the class will reserve that fraction of the
    provided data as a dev set (random sampling with `seed`).

    Use by creating with `PRewriteDataset(df, split='train', dev_fraction=0.1)`
    and calling `.iter()` to iterate over examples for the chosen split.
    """

    def __init__(self, source: Any, split: str = "train", dev_fraction: float = 0.0, seed: int = 42):
        if isinstance(source, pd.DataFrame):
            df = source.copy()
        else:
            p = Path(source)
            if not p.exists():
                raise FileNotFoundError(f"Dataset file not found: {p}")
            if p.suffix.lower() in {".parquet", ".pq"}:
                df = pd.read_parquet(p)
            elif p.suffix.lower() in {".jsonl", ".ndjson", ".json"}:
                df = pd.read_json(p, lines=True)
            else:
                df = pd.read_csv(p)

        df.columns = [c.lower() for c in df.columns]
        self._seed = seed
        self._requested_split = split

        # Determine splits
        if "split" in df.columns:
            # Expect values like 'train','dev','test'
            self.train_df = df[df["split"] == "train"].reset_index(drop=True)
            self.dev_df = df[df["split"] == "dev"].reset_index(drop=True)
            self.test_df = df[df["split"] == "test"].reset_index(drop=True)
        else:
            # treat provided DataFrame as train; optionally carve out a dev set
            if dev_fraction and dev_fraction > 0.0:
                dev_df = df.sample(frac=dev_fraction, random_state=seed)
                train_df = df.drop(dev_df.index).reset_index(drop=True)
                self.train_df = train_df
                self.dev_df = dev_df.reset_index(drop=True)
            else:
                self.train_df = df.reset_index(drop=True)
                self.dev_df = None
            self.test_df = None

        # map split name to DataFrame
        self._split_map = {
            "train": self.train_df,
            "dev": self.dev_df,
            "test": self.test_df,
        }

    def iter(self) -> Iterator[RewriteExample]:
        df = self._split_map.get(self._requested_split)
        if df is None:
            raise ValueError(f"No data for split: {self._requested_split}")
        for _, row in df.iterrows():
            instruction = row.get("instruction") or row.get("prompt") or row.get("question") or ""
            inp = row.get("input") or row.get("text") or row.get("article") or ""
            output = row.get("output") or row.get("answer") or row.get("label_text") or row.get("label") or ""
            try:
                instruction = str(instruction) if instruction is not None else ""
            except Exception:
                instruction = ""
            try:
                inp = str(inp) if inp is not None else ""
            except Exception:
                inp = ""
            try:
                output = str(output) if output is not None else ""
            except Exception:
                output = ""
            task = row.get("task") or "classification"
            yield RewriteExample(instruction=instruction, input=inp, output=output, task=task)

    def get_split_df(self, split: str = "train") -> Optional[pd.DataFrame]:
        return self._split_map.get(split)



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


def prepare_df_to_jsonl(df: pd.DataFrame, output_path: str = "dataset.jsonl") -> str:
    """Write a pandas DataFrame to JSONL using the fields expected by the trainer.

    This performs a best-effort mapping of common column names to the
    instruction/input/output/task fields.
    """
    out_path = Path(output_path)
    cols = {c.lower(): c for c in df.columns}
    with out_path.open("w", encoding="utf-8") as fout:
        for _, row in df.iterrows():
            record = {
                "instruction": row.get(cols.get("instruction", ""), "") or row.get(cols.get("prompt", ""), ""),
                "input": row.get(cols.get("input", ""), "") or row.get(cols.get("text", ""), ""),
                "output": row.get(cols.get("output", ""), "") or row.get(cols.get("answer", ""), "") or row.get(cols.get("label_text", ""), "") or row.get(cols.get("label", ""), ""),
                "task": row.get(cols.get("task", ""), "classification") or "classification",
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(out_path)


__all__ = [
    "RewriteExample",
    "build_meta_prompt",
    "format_rewriter_query",
    "prepare_df_to_jsonl",
    "PRewriteDataset",
]
