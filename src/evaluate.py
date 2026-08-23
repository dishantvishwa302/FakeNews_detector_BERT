"""Print accuracy / precision / recall / F1 and a confusion matrix.

    python -m src.evaluate
"""

from __future__ import annotations

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import BATCH_SIZE, BEST_MODEL_DIR, LABEL_NAMES, TEST_SPLIT_PATH, get_device
from src.data import load_titles, make_splits
from src.dataset import TitleDataset


def load_test_frame():
    if TEST_SPLIT_PATH.exists():
        return load_titles(TEST_SPLIT_PATH)
    return make_splits(load_titles()).test


def main() -> None:
    if not BEST_MODEL_DIR.exists():
        raise SystemExit(f"No checkpoint at {BEST_MODEL_DIR}. Run: python -m src.train")

    dev = get_device()
    tokenizer = AutoTokenizer.from_pretrained(BEST_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BEST_MODEL_DIR)
    model.to(dev)
    model.eval()

    test_df = load_test_frame()
    loader = DataLoader(TitleDataset(test_df, tokenizer), batch_size=BATCH_SIZE)

    preds, gold = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(dev) for k, v in batch.items()}
            logits = model(**batch).logits
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            gold.extend(labels.tolist())

    target_names = [LABEL_NAMES[0], LABEL_NAMES[1]]
    print("n_test =", len(gold))
    print(classification_report(gold, preds, target_names=target_names, digits=4))
    print("confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(gold, preds))


if __name__ == "__main__":
    main()
