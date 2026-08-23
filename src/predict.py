"""Classify one headline.

    python -m src.predict "Fed holds interest rates steady after a two-day meeting"
"""

from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import BEST_MODEL_DIR, LABEL_NAMES, MAX_LEN, get_device


def load_model():
    if not BEST_MODEL_DIR.exists():
        raise SystemExit(f"No checkpoint at {BEST_MODEL_DIR}. Run: python -m src.train")
    tok = AutoTokenizer.from_pretrained(BEST_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BEST_MODEL_DIR)
    model.to(get_device())
    model.eval()
    return tok, model


@torch.no_grad()
def predict_one(text: str, tokenizer, model) -> tuple[str, float, float]:
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )
    enc = {k: v.to(get_device()) for k, v in enc.items()}
    probs = torch.softmax(model(**enc).logits, dim=-1)[0]
    pred = int(probs.argmax().item())
    return LABEL_NAMES[pred], float(probs[0]), float(probs[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Fake vs real title classifier")
    parser.add_argument("text", nargs="?", help="headline to classify")
    args = parser.parse_args()
    text = args.text or (sys.stdin.read().strip() if not sys.stdin.isatty() else "")
    if not text:
        parser.error("pass a headline, e.g. python -m src.predict \"WHO reports decline in measles cases\"")

    tokenizer, model = load_model()
    label, p_real, p_fake = predict_one(text, tokenizer, model)
    print(f"title:  {text}")
    print(f"pred:   {label}")
    print(f"P(real)={p_real:.3f}  P(fake)={p_fake:.3f}")


if __name__ == "__main__":
    main()
