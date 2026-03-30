# Health claim checker

CLI + training scripts for a simple health-claim classifier (reliable vs misinformation) with optional article-context scraping.

## Setup

```bash
cd /path/to/healthinfo
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run (interactive)

Run from the project root:

```bash
source venv/bin/activate
python3 test_claim.py
```

## Train

```bash
source venv/bin/activate
python3 training/train_model.py
```

## Notes / limits

- Scraping won’t work for every URL (paywalls, logins, CAPTCHAs, PDF-only pages, JS-heavy sites).
- Avoid committing large generated artifacts (model checkpoints, caches, datasets). Use `.gitignore` to keep pushes small.

