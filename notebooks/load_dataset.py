"""
Download the health_fact dataset from HuggingFace and save it as parquet
under data/train/ for local use.

Run this once to populate data/train/ if the parquet files are not present.
"""

from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "train"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading ImperialCollegeLondon/health_fact from HuggingFace…")
dataset = load_dataset("ImperialCollegeLondon/health_fact")
print(dataset)

train_df = dataset["train"].to_pandas()
out_path = OUT_DIR / "Health Fact 0000.parquet"
train_df.to_parquet(out_path, index=False)
print(f"Saved {len(train_df)} rows to {out_path}")
