"""
Shared data loading utilities for the health claim checker.

Used by test_claim.py, training/train_model.py, and notebooks.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_TRAIN = ROOT / "data" / "train"

# Original 4-class → binary: 0=RELIABLE, 1=MISINFORMATION
LABEL_MAP: dict[int, int | None] = {
    0: 1,   # false / misleading  → MISINFORMATION
    1: 0,   # true                → RELIABLE
    2: 1,   # mixture             → MISINFORMATION
    3: 1,   # unproven            → MISINFORMATION
    -1: None,  # invalid          → drop
}

BINARY_NAMES: dict[int, str] = {0: "RELIABLE", 1: "MISINFORMATION"}


def load_parquet_folder(folder: str | Path = DATA_TRAIN) -> pd.DataFrame:
    """Load all *.parquet files from a folder and concatenate them."""
    files = sorted(glob.glob(str(Path(folder) / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {folder}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def prepare_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map original labels to binary and drop invalid rows."""
    out = df.copy()
    out["label_bin"] = out["label"].map(LABEL_MAP)
    out = out.dropna(subset=["label_bin"])
    out["label_bin"] = out["label_bin"].astype(int)
    return out


def load_training_data(folder: str | Path = DATA_TRAIN) -> pd.DataFrame:
    """Convenience: load parquet folder and apply label mapping."""
    return prepare_labels(load_parquet_folder(folder))


if __name__ == "__main__":
    df = load_training_data()
    print(f"Dataset size: {len(df)}")
    print("Columns:", df.columns.tolist())
    print("\nLabel distribution:")
    print(df["label_bin"].value_counts())
    print("\nExample claims:")
    for i in range(3):
        row = df.iloc[i]
        print(f"  Claim: {row['claim']}")
        print(f"  Label: {BINARY_NAMES[row['label_bin']]}\n")
