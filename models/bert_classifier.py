"""
BERT-based health claim classifier for inference.

Loads a fine-tuned DistilBERT model saved by training/train_model.py.

Usage:
    from models.bert_classifier import BertHealthClassifier

    clf = BertHealthClassifier()          # loads models/bert_model/
    label, conf = clf.predict("Vaccines cause autism")
    print(label, conf)                    # "MISINFORMATION", 0.93
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = ROOT / "models" / "bert_model"

BINARY_NAMES = {0: "RELIABLE", 1: "MISINFORMATION"}
MAX_LEN = 256


class BertHealthClassifier:
    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR) -> None:
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model not found at {model_dir}. "
                "Run `python3 training/train_model.py` first."
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> tuple[str, float]:
        """
        Returns (label_name, confidence) for a single text.
        """
        labels, confs = self.predict_batch([text])
        return labels[0], confs[0]

    def predict_batch(
        self, texts: list[str]
    ) -> tuple[list[str], list[float]]:
        """
        Returns (label_names, confidences) for a list of texts.
        """
        enc = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        preds = np.argmax(probs, axis=-1)
        labels = [BINARY_NAMES[int(p)] for p in preds]
        confs = [float(probs[i, preds[i]]) for i in range(len(preds))]
        return labels, confs


if __name__ == "__main__":
    clf = BertHealthClassifier()
    examples = [
        "Vaccines cause autism.",
        "Regular hand-washing reduces the spread of infection.",
        "Drinking bleach cures COVID-19.",
    ]
    for ex in examples:
        label, conf = clf.predict(ex)
        print(f"[{label} | {conf:.3f}] {ex}")
