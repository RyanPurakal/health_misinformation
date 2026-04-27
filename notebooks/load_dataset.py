"""
One-off download script: pulls the ImperialCollegeLondon/health_fact dataset
from HuggingFace Hub and prints a summary; run once to inspect the raw schema.
"""
from datasets import load_dataset

dataset = load_dataset("ImperialCollegeLondon/health_fact")

print(dataset)
