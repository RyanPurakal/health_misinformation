# models/

DistilBERT inference wrapper and saved model weights.

## Files

| Path | Role |
|---|---|
| `bert_classifier.py` | `HealthClaimClassifier` — loads the model once, exposes `predict(claim, article_text)` |
| `bert_model/` | Saved weights + tokenizer written by `training/train_model.py` |
| `bert_model/checkpoints/` | Per-epoch checkpoints; `checkpoint-68` is the fallback if the root model is missing |

## How the model directory is resolved

`bert_classifier.py` prefers `models/bert_model/` (has tokenizer). If that directory has no `config.json` it falls back to `models/bert_model/checkpoints/checkpoint-68/`. If neither exists, it raises `FileNotFoundError` immediately — run `training/train_model.py` first.

## Input format

`predict()` concatenates claim and article text with the tokenizer's separator token before encoding: `claim [SEP] article_text`. If `article_text` is empty, only the claim is passed. Accuracy is higher when article text is available.
