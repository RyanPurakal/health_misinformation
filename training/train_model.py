"""
Fine-tunes DistilBERT for binary health-claim classification. Reads all parquet
files from data/train/, applies class-weighted loss to counter the 4.7:1
misinformation-to-reliable imbalance, and saves the best checkpoint (by macro
F1) to models/bert_model/. Must be run from the repo root.
"""
import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from torch.utils.data import Dataset

BASE_MODEL = "distilbert-base-uncased"
MODEL_OUT = "models/bert_model"
MAX_LEN = 256
BATCH_SIZE = 16
GRAD_ACCUM = 1
EPOCHS = 8
LR = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
SEED = 42

LABEL_MAP = {
    0: 1,   # false → misinformation
    1: 0,   # true  → reliable
    2: 1,   # mixture → misinformation
    3: 1,   # unproven → misinformation
    -1: None,
}


def load_data(folder="data/train"):
    files = glob.glob(f"{folder}/*.parquet")
    if not files:
        raise FileNotFoundError(f"No parquet files found in {folder}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["label"] = df["label"].map(LABEL_MAP)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["main_text"] = df["main_text"].fillna("").astype(str).str.strip()
    return df


class ClaimDataset(Dataset):
    def __init__(self, claims, main_texts, labels, tokenizer):
        pairs = list(zip(claims, main_texts))
        # Concatenate claim + article; tokenizer truncates from the right
        sep = tokenizer.sep_token or "[SEP]"
        combined = [
            f"{str(c)} {sep} {str(t)}" if str(t).strip() else str(c)
            for c, t in pairs
        ]
        self.encodings = tokenizer(
            combined,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    per_class = f1_score(labels, preds, average=None, zero_division=0)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_reliable": float(per_class[0]) if len(per_class) > 0 else 0.0,
        "f1_misinfo": float(per_class[1]) if len(per_class) > 1 else 0.0,
    }


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weights = self.class_weights.to(outputs.logits.device)
        loss = nn.CrossEntropyLoss(weight=weights)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def main():
    df = load_data()
    print(f"Total samples: {len(df)}")
    print(df["label"].value_counts())

    has_text = df["main_text"].str.len() > 50
    print(f"Rows with main_text: {has_text.sum()} ({has_text.mean():.1%})")

    train_df, val_df = train_test_split(
        df, test_size=0.15, random_state=SEED, stratify=df["label"]
    )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_df["label"].values,
    )
    class_weights = torch.tensor(weights, dtype=torch.float)
    print(f"Class weights — RELIABLE: {weights[0]:.3f}, MISINFO: {weights[1]:.3f}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    train_ds = ClaimDataset(train_df["claim"], train_df["main_text"], train_df["label"], tokenizer)
    val_ds = ClaimDataset(val_df["claim"], val_df["main_text"], val_df["label"], tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

    args = TrainingArguments(
        output_dir=f"{MODEL_OUT}/checkpoints",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        dataloader_drop_last=False,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=20,
        seed=SEED,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    os.makedirs(MODEL_OUT, exist_ok=True)
    model.save_pretrained(MODEL_OUT)
    tokenizer.save_pretrained(MODEL_OUT)
    print(f"\nModel saved to {MODEL_OUT}")


if __name__ == "__main__":
    main()
