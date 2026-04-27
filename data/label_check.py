"""
Sanity-check script: loads the training parquet files, remaps labels to binary
(0 = reliable, 1 = misinformation), and prints the resulting class distribution.
"""
import pandas as pd
import glob

# --- Function to load all parquet files in a folder ---
def load_parquet_folder(path):
    files = glob.glob(f"{path}/*.parquet")
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)

# --- Load the training dataset ---
df = load_parquet_folder("data/train")  # <-- make sure this matches your folder

print(f"Dataset size: {len(df)}")
print("Columns:", df.columns)

# --- Map original labels to binary ---
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

print("\nLabel distribution:")
print(df['label'].value_counts())

# --- Optional: print a few example claims ---
print("\nExample claims:\n")
for i in range(5):
    print(f"Claim: {df['claim'].iloc[i]}")
    print(f"Label: {df['label'].iloc[i]}\n")