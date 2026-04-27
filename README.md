# Health Claim Checker

A tool that takes a health-related claim and returns **RELIABLE** or **MISINFORMATION**, a confidence score, and a short explanation. Built for learning and prototyping — not medical advice.

## Architecture

The project has two independent layers that can be used separately or together:

```
┌─────────────────────────────────────────────────────────┐
│  Web app layer  (app.py + frontend/)                    │
│                                                         │
│  Browser → POST /api/predict → FastAPI (app.py)         │
│                                    │                    │
│                              TF-IDF + LogReg            │
│                           (model/claim_model.joblib)    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Research / fine-tuning layer  (training/ + data/)      │
│                                                         │
│  data/train/*.parquet                                   │
│       │                                                 │
│       ▼                                                 │
│  training/train_model.py  (fine-tunes DistilBERT)       │
│       │                                                 │
│       ▼                                                 │
│  models/bert_model/  (saved weights + tokenizer)        │
│       │                                                 │
│       ▼                                                 │
│  models/bert_classifier.py  → {label, confidence}       │
│       │                                                 │
│  llm/explanation_generator.py  → explanation text       │
└─────────────────────────────────────────────────────────┘
```

## Quick start — web app

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 train_model.py        # trains and saves model/claim_model.joblib
uvicorn app:app --reload --port 8080
# open http://localhost:8080
```

## Quick start — DistilBERT research stack

```bash
source .venv/bin/activate
python3 training/train_model.py   # fine-tunes DistilBERT; saves to models/bert_model/
```

## Key design decisions

- **Two classifiers, different trade-offs**: the web app uses TF-IDF + LogisticRegression for fast, dependency-light serving. The research stack fine-tunes DistilBERT with article context for higher accuracy (78% overall, 67% macro F1).
- **Article context matters**: DistilBERT is trained on `claim + article_text` concatenated. Claim-only accuracy is lower — confidence scores below ~65% signal missing context.
- **Class-weighted loss**: the training dataset is ~4.7:1 misinformation-to-reliable. Without weighting the model ignores reliable claims; weighting raised reliable-class recall from 3% to 61%.
- **TF-IDF explanation lookup**: returns verbatim explanations from the training set rather than generating text; falls back to a generic string when no close match is found.

## Limits

- Paywalled, JS-heavy, and PDF-only URLs often fail extraction.
- High confidence does not mean correct — treat output as a signal, not a verdict.

## Directory map

| Path | Responsibility |
|---|---|
| `app.py` | FastAPI server — loads model at startup, serves `/api/predict` and static frontend |
| `train_model.py` | Trains TF-IDF + LogReg pipeline; saves `model/claim_model.joblib` |
| `frontend/` | Browser UI — calls the API; includes a JS-only fallback classifier |
| `model/` | Saved sklearn model artifact |
| `data/` | Parquet dataset loading and label remapping utilities |
| `models/` | DistilBERT inference wrapper and saved fine-tuned weights |
| `training/` | DistilBERT fine-tuning script |
| `llm/` | TF-IDF nearest-neighbour explanation retrieval |
| `notebooks/` | One-off script to download the HuggingFace dataset |
