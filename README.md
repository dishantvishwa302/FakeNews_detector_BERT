# Fake vs real news titles with DistilBERT

Fine-tune DistilBERT on short headlines. Binary labels: `0` real, `1` fake. Built to walk through in an interview, not to hide behind a 200-line custom Transformer.

## 5-minute demo

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# no GPU needed
python -m src.baseline

# DistilBERT (downloads the model once; CPU is slow but works)
python -m src.train
python -m src.evaluate
python -m src.predict "Fed holds interest rates steady after a two-day meeting"
python -m src.predict "You won't believe this spice reverses aging overnight"
```

Open `notebooks/demo.ipynb` for the same story with printed examples.

## What to say (30 seconds)

I classify news **titles** as fake or real. The model is DistilBERT plus a 2-class head. I tokenize, fine-tune for two epochs, and pick the checkpoint with the best validation F1. A TF-IDF + logistic regression baseline sits next to it so I can say whether the transformer is actually helping.

## Layout

```
data/sample_titles.csv   1,600 bundled titles (offline)
src/config.py            hyperparameters
src/data.py              load + stratified split
src/dataset.py           tokenizer + PyTorch Dataset
src/baseline.py          TF-IDF + logistic regression
src/train.py             DistilBERT fine-tune
src/evaluate.py          test metrics
src/predict.py           one headline → real/fake + probabilities
```

## Original assignment vs this repo

| | Assignment notebook | This repo |
|---|---|---|
| Data | Fakeddit TSVs on Colab | 1,600 titles in `data/` |
| Model | `bert-base-uncased` (GPU) | DistilBERT (laptop CPU) |
| Extra | custom BERT from scratch + Captum | dropped — buggy and hard to explain live |
| Result | 79.6% accuracy on 1,000 Fakeddit titles | run `evaluate` on the bundled split |

The custom encoder had real bugs (`self.out` vs `self.fc`, tokenizer never created, train and test read the same file). HuggingFace DistilBERT is the part I can defend.

## Interview extras

- Why titles only: Fakeddit's `clean_title` is the signal; full article text is a different task.
- Why DistilBERT: same WordPiece tokenizer family as BERT, ~40% fewer parameters, same fine-tuning story.
- Why a baseline: on this demo CSV, TF-IDF already gets **98.4%** test accuracy. DistilBERT is the recipe to show; the hard number is still 79.6% on Fakeddit.
- What I would not claim: this does not "detect truth". It detects patterns in how the title is written.

Walkthrough: [DESIGN_DOC.md](DESIGN_DOC.md). Dataset caveats: [data/DATA.md](data/DATA.md).
