"""Bag-of-words baseline: TF-IDF + logistic regression.

Interview point: if DistilBERT cannot beat this on a tiny title-only set,
the transformer is not buying you anything.

    python -m src.baseline
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from src.config import LABEL_NAMES, SEED
from src.data import load_titles, make_splits


def main() -> None:
    splits = make_splits(load_titles())
    clf = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20_000)),
            ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]
    )
    clf.fit(splits.train["title"], splits.train["label"])
    preds = clf.predict(splits.test["title"])
    gold = splits.test["label"]
    names = [LABEL_NAMES[0], LABEL_NAMES[1]]
    print("TF-IDF + logistic regression (titles only)")
    print(classification_report(gold, preds, target_names=names, digits=4))
    print("confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(gold, preds))


if __name__ == "__main__":
    main()
