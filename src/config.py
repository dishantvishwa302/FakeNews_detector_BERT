"""Single place for paths and training knobs.

Interview talking point: if someone asks 'what did you tune?', point here.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sample_titles.csv"
CHECKPOINT_DIR = ROOT / "checkpoints"
BEST_MODEL_DIR = CHECKPOINT_DIR / "best_model"
TEST_SPLIT_PATH = CHECKPOINT_DIR / "test_split.csv"

# DistilBERT: same tokenizer family as BERT, ~40% fewer parameters.
# Swap to "bert-base-uncased" if you have a GPU and want the original paper model.
MODEL_NAME = "distilbert-base-uncased"

MAX_LEN = 64          # titles are short; 128 was overkill in the original notebook
BATCH_SIZE = 16
EPOCHS = 2
LR = 2e-5
WEIGHT_DECAY = 0.01
SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1        # 10% of the training split, used only to pick the best epoch

LABEL_NAMES = {0: "real", 1: "fake"}


def get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
