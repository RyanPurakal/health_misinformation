# Health claim checker

This project is a small “health claim checker”: you give it a **health-related claim** (or a **link to an article**) and it returns a label like **RELIABLE** or **MISINFORMATION**, plus a short explanation.

It’s meant for learning and prototyping—not as medical advice.

## What it does

- **Checks a claim** and returns a label + confidence score.
- **Optionally uses article text**: if you paste a URL, it tries to extract the main article text and use that as extra context.
- **Shows an explanation**: it tries to reuse explanations from the dataset; if it can’t find a close match, it shows a generic explanation based on the predicted label.

## How to use it

### 1) Install

```bash
cd /path/to/healthinfo
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2) Run (interactive)

```bash
source venv/bin/activate
python3 test_claim.py
```

Then type either:
- a **plain claim** (example: `Vaccines cause autism`), or
- a **URL** to an article (if extraction fails, you may need to paste the article text directly).

## Training (optional)

If you want to retrain the model on your data:

```bash
source venv/bin/activate
python3 training/train_model.py
```

## Tech stack

- **Language**: Python
- **ML**: PyTorch + Hugging Face Transformers (`Trainer`)
- **Data**: pandas (dataset prep)
- **Explanations**: scikit-learn (TF‑IDF similarity lookup)
- **Scraping**: trafilatura + readability-lxml + BeautifulSoup (optional Playwright)

## What to expect (limits)

- **Not every URL will work**: paywalls, logins, CAPTCHAs, PDF-only pages, and JS-heavy sites often block text extraction.
- **Claim-only is harder**: the model can do worse when you provide only a short claim with no article context.
- **Confidence isn’t truth**: a high confidence score can still be wrong.

## What’s in this repo

- `test_claim.py`: interactive program you run
- `training/train_model.py`: training script
- `scraper/`: article text extraction helpers
- `data/`: dataset loaders + explanation lookup


