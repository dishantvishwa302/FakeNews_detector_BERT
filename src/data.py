"""CSV loading and stratified splits. No PyTorch — the TF-IDF baseline can import this."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_PATH, SEED, TEST_SIZE, VAL_SIZE


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_titles(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["title", "label"]).copy()
    df["title"] = df["title"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    df = df[df["title"].str.len() > 0]
    return df.reset_index(drop=True)


def make_splits(df: pd.DataFrame) -> Splits:
    train_val, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=SEED
    )
    train, val = train_test_split(
        train_val,
        test_size=VAL_SIZE,
        stratify=train_val["label"],
        random_state=SEED,
    )
    return Splits(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )
