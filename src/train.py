"""Fine-tune DistilBERT for fake vs real titles.

Run from the repo root:

    python -m src.train
"""

from __future__ import annotations

import random

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    BEST_MODEL_DIR,
    CHECKPOINT_DIR,
    EPOCHS,
    LR,
    MODEL_NAME,
    SEED,
    TEST_SPLIT_PATH,
    WEIGHT_DECAY,
    get_device,
)
from src.dataset import TitleDataset, build_tokenizer, load_titles, make_splits


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def run_eval(model, loader, dev) -> tuple[float, float]:
    model.eval()
    losses, preds, gold = [], [], []
    for batch in loader:
        batch = {k: v.to(dev) for k, v in batch.items()}
        out = model(**batch)
        losses.append(out.loss.item())
        preds.extend(out.logits.argmax(dim=-1).cpu().tolist())
        gold.extend(batch["labels"].cpu().tolist())
    return float(np.mean(losses)), float(f1_score(gold, preds, average="binary"))


def main() -> None:
    set_seed()
    dev = get_device()
    print(f"device: {dev}")
    print(f"model:  {MODEL_NAME}")

    df = load_titles()
    splits = make_splits(df)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    splits.test.to_csv(TEST_SPLIT_PATH, index=False)

    tokenizer = build_tokenizer(MODEL_NAME)
    train_ds = TitleDataset(splits.train, tokenizer)
    val_ds = TitleDataset(splits.val, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(dev)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps
    )

    best_f1 = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{EPOCHS}"):
            batch = {k: v.to(dev) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running.append(out.loss.item())

        val_loss, val_f1 = run_eval(model, val_loader, dev)
        print(
            f"epoch {epoch}: train_loss={np.mean(running):.4f}  "
            f"val_loss={val_loss:.4f}  val_f1={val_f1:.4f}"
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(BEST_MODEL_DIR)
            tokenizer.save_pretrained(BEST_MODEL_DIR)
            print(f"  saved best checkpoint to {BEST_MODEL_DIR} (val_f1={best_f1:.4f})")

    print("done. next: python -m src.evaluate")


if __name__ == "__main__":
    main()
