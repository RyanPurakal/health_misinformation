"""
Exploratory script: loads all parquet files from data/train/, applies the
4-class → binary label remap, and prints size, distribution, and sample rows.
"""
import pandas as pd
import glob

def load_parquet_folder(path):
    files = glob.glob(f"{path}/*.parquet")
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)

df = load_parquet_folder("data/train")  # or whichever folder

print(f"Dataset size: {len(df)}")
print("Columns:", df.columns)

# --- NEW: Map labels to binary ---
label_map = {
    0: 1,   # false / misleading → misinformation
    1: 0,   # true → reliable
    2: 1,   # mixture → misinformation
    3: 1,   # unproven → misinformation
    -1: None  # drop invalid
}

df['label'] = df['label'].map(label_map)
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

print("Label distribution:")
print(df['label'].value_counts())

# Optional: check a few examples
for i in range(5):
    print(f"Claim: {df['claim'].iloc[i]}")
    print(f"Label: {df['label'].iloc[i]}")
    print()