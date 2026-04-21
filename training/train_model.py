"""
Fine-tune DistilBERT for binary health claim classification.

Labels: 0 = RELIABLE, 1 = MISINFORMATION
Saves model + tokenizer to models/bert_model/

Usage:
    python3 training/train_model.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_TRAIN = ROOT / "data" / "train"
MODEL_OUT = ROOT / "models" / "bert_model"

BASE_MODEL = "distilbert-base-uncased"

LABEL_MAP = {0: 1, 1: 0, 2: 1, 3: 1, -1: None}
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5
SEED = 42


def load_data() -> pd.DataFrame:
    files = sorted(glob.glob(str(DATA_TRAIN / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {DATA_TRAIN}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["label_bin"] = df["label"].map(LABEL_MAP)
    df = df.dropna(subset=["label_bin"])
    df["label_bin"] = df["label_bin"].astype(int)
    return df


def make_text(row: pd.Series) -> str:
    claim = str(row.get("claim") or "").strip()
    mt = row.get("main_text")
    if pd.isna(mt) or not str(mt).strip():
        return claim
    return f"{claim}\n\n{str(mt)[:1500]}"


def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary"),
    }


def main() -> None:
    print("Loading data…")
    df = load_data()
    print(f"  {len(df)} examples | label dist: {df['label_bin'].value_counts().to_dict()}")

    df["text"] = df.apply(make_text, axis=1)
    train_df, val_df = train_test_split(
        df[["text", "label_bin"]], test_size=0.15, random_state=SEED, stratify=df["label_bin"]
    )

    print(f"  Train: {len(train_df)} | Val: {len(val_df)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )

    train_ds = Dataset.from_pandas(train_df.rename(columns={"label_bin": "labels"}))
    val_ds = Dataset.from_pandas(val_df.rename(columns={"label_bin": "labels"}))

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])

    print(f"Loading base model: {BASE_MODEL}")
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

    args = TrainingArguments(
        output_dir=str(MODEL_OUT / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=SEED,
        logging_steps=20,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Training…")
    trainer.train()

    print(f"Saving model to {MODEL_OUT}")
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_OUT))
    tokenizer.save_pretrained(str(MODEL_OUT))

    metrics = trainer.evaluate()
    print("Final val metrics:", metrics)
    print("Done.")


if __name__ == "__main__":
    main()
