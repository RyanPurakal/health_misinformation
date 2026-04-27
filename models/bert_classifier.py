"""
Inference wrapper for the fine-tuned DistilBERT model. Loads weights from
models/bert_model/ (or the checkpoint-68 fallback) once at construction and
exposes a single predict(claim, article_text) method; read-only after init.
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = os.path.join("models", "bert_model")
CHECKPOINT_FALLBACK = os.path.join(MODEL_DIR, "checkpoints", "checkpoint-68")
FALLBACK_BASE = "distilbert-base-uncased"
MAX_LEN = 512

LABEL_NAMES = {0: "RELIABLE", 1: "MISINFORMATION"}


def _resolve_model_dir() -> str:
    if os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        return MODEL_DIR
    if os.path.isdir(CHECKPOINT_FALLBACK):
        return CHECKPOINT_FALLBACK
    raise FileNotFoundError(
        f"No model found at {MODEL_DIR} or {CHECKPOINT_FALLBACK}.\n"
        "Run training/train_model.py first."
    )


class HealthClaimClassifier:
    def __init__(self, model_dir: str | None = None):
        model_dir = model_dir or _resolve_model_dir()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir if os.path.exists(os.path.join(model_dir, "tokenizer_config.json"))
            else FALLBACK_BASE
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, claim: str, article_text: str = "") -> dict:
        sep = self.tokenizer.sep_token or "[SEP]"
        text = f"{claim} {sep} {article_text.strip()}" if article_text.strip() else claim

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).squeeze()
        label_id = int(torch.argmax(probs).item())
        confidence = float(probs[label_id].item())

        return {
            "label_id": label_id,
            "label": LABEL_NAMES[label_id],
            "confidence": confidence,
            "probs": {LABEL_NAMES[i]: float(probs[i]) for i in range(len(probs))},
        }
