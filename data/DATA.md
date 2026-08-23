# sample_titles.csv

1,600 synthetic English headlines. 800 labeled `0` (real style), 800 labeled `1` (fake style).

## Why this file exists

The original assignment used [Fakeddit](https://github.com/entitize/Fakeddit) TSV files that are not in this repo. This CSV is bundled so `python -m src.train` runs on a laptop with no extra download.

## How it was made

`scripts/build_sample.py` builds two styles:

- **Real:** named institution + dry verb + measurable claim ("The Federal Reserve holds interest rates after a two-day meeting").
- **Fake:** clickbait / conspiracy framing ("You won't believe this spice reverses aging overnight").

Some real templates mix an agency with a loosely related event. That is fine for a **style** classifier; do not treat a title as a factual report.

## What this dataset is not

It is not Fakeddit, ISOT, or LIAR. High accuracy here mostly means the model learned "news desk vs clickbait", not "true vs false". Say that in the interview.

On the original Fakeddit 5k-title subsample with `bert-base-uncased`, test accuracy was **79.6%** (see DESIGN_DOC.md). Those two classes look much more similar (both are Reddit titles).
