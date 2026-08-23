"""Turn title strings into padded token ids."""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.config import MAX_LEN
from src.data import Splits, load_titles, make_splits  # re-exported for train.py

__all__ = ["TitleDataset", "build_tokenizer", "load_titles", "make_splits", "Splits"]


class TitleDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer: PreTrainedTokenizerBase):
        self.labels = frame["label"].tolist()
        self.encodings = tokenizer(
            frame["title"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def build_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)
