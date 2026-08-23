# DistilBERT fake-news titles (interview walkthrough)

Repository: FakeNews_detector_BERT

Say this in order. Stop after each section if they want to go deeper.

## 1. What it is (30 seconds)

A binary classifier on **news titles**: real vs fake. DistilBERT encodes the title; a linear head on the `[CLS]` vector outputs two logits. Softmax gives `P(real)` and `P(fake)`.

One path:

```
title  -->  WordPiece tokenizer  -->  DistilBERT  -->  [CLS]  -->  linear(2)
                                                              |
                                                         argmax = label
```

## 2. Data

`data/sample_titles.csv`: 800 real-style + 800 fake-style titles, stratified 70 / 10 / 20 train / val / test (`src/config.py`).

This file is synthetic so the demo runs offline. The original assignment used Fakeddit (`clean_title`, `2_way_label`), 2,500 per class, **79.6%** test accuracy with `bert-base-uncased`.

Honest line: high accuracy on the bundled CSV mostly means "news desk vs clickbait", not fact-checking.

## 3. Model

`distilbert-base-uncased` + `AutoModelForSequenceClassification(num_labels=2)`.

- DistilBERT is a smaller BERT trained to match BERT's hidden states (distillation).
- Max length **64** because titles are short. The assignment used 128; that wasted padding.
- We do not train a Transformer from scratch. That code in the old notebook did not run.

If they ask for BERT-base: change `MODEL_NAME` in `src/config.py`. Same training script.

## 4. Training

`python -m src.train`

- AdamW, `lr=2e-5`, weight decay `0.01`, 2 epochs, batch 16
- Linear warmup for 10% of steps, then linear decay
- Grad clip at 1.0
- Save the epoch with the best **validation F1**, not the last train loss

The old notebook printed only the last batch loss and had no validation split.

## 5. Baseline

`python -m src.baseline`

TF-IDF (unigrams + bigrams) + logistic regression on the same split.

Measured on the bundled 320-title test set: **98.4% accuracy** (5 real titles tagged fake; no fake title tagged real). That is the style leak. DistilBERT should match it; it is not the number to brag about. On Fakeddit, bag-of-words is weaker because both classes are informal Reddit titles.

## 6. Inference

`python -m src.predict "…"`

Tokenize the same way as training (`max_length=64`, padding). Read probabilities, not only the argmax — a 0.51 vs 0.99 gap is a talking point.

## 7. What I would not claim

- This is not a production fact-checker. It never looks up a source.
- The bundled labels are **stylistic**. A dry lie would look "real"; a sarcastic true headline could look "fake".
- No Integrated Gradients in this repo. Captum was in the assignment; convergence deltas were large on short titles, so I would not present those plots as solid explanations.
- Checkpoints are not committed (hundreds of MB). Train locally, then demo `predict`.
