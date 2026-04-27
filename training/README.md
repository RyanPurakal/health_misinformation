# training/

DistilBERT fine-tuning script. Run once from the repo root to produce the model weights used by `models/bert_classifier.py`.

## Usage

```bash
python3 training/train_model.py
```

Saves the best checkpoint (by macro F1) to `models/bert_model/`.

## Key decisions in the training script

| Decision | Reason |
|---|---|
| Class-weighted loss | Dataset is ~4.7:1 misinformation-to-reliable; without weighting the model achieves near-zero recall on reliable claims |
| claim + article_text input | Claim text alone (~11 words median) carries little signal; the article body is the evidence |
| Early stopping (patience 3) | Prevents overfitting on the small dataset (1,265 samples); best checkpoint is loaded automatically |
| Macro F1 as best-model metric | Weighted F1 and accuracy are dominated by the majority class; macro F1 gives equal weight to both classes |

## Outputs

| Path | Contents |
|---|---|
| `models/bert_model/model.safetensors` | Best model weights |
| `models/bert_model/tokenizer*` | Tokenizer files (required for inference) |
| `models/bert_model/checkpoints/` | All per-epoch checkpoints |
