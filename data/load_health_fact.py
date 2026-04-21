from data.utils import load_training_data, BINARY_NAMES

df = load_training_data()

print(f"Dataset size: {len(df)}")
print("Columns:", df.columns.tolist())
print("Label distribution:")
print(df["label_bin"].value_counts())
print()
for i in range(5):
    row = df.iloc[i]
    print(f"Claim: {row['claim']}")
    print(f"Label: {BINARY_NAMES[row['label_bin']]}")
    print()