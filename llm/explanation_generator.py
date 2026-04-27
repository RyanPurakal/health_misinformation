"""
TF-IDF nearest-neighbour explanation retrieval. Builds a TF-IDF index over
training claims at construction (cached to .cache/explanations/ by dataset
hash), then returns the closest matching explanation for the predicted label;
falls back to a generic string when similarity is below SIMILARITY_THRESHOLD.
"""
import os
import glob
import pickle
import hashlib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CACHE_DIR = os.path.join(".cache", "explanations")
SIMILARITY_THRESHOLD = 0.25

GENERIC_EXPLANATIONS = {
    0: (
        "This claim appears to be supported by established medical and scientific evidence. "
        "Reliable health claims typically align with peer-reviewed research and consensus guidelines."
    ),
    1: (
        "This claim shows characteristics of health misinformation. "
        "It may exaggerate, misrepresent, or lack support from peer-reviewed evidence. "
        "Always verify health claims with trusted medical sources."
    ),
}

LABEL_MAP = {
    0: 1,
    1: 0,
    2: 1,
    3: 1,
    -1: None,
}


def _dataset_hash(df: pd.DataFrame) -> str:
    key = str(len(df)) + str(list(df.columns))
    return hashlib.sha1(key.encode()).hexdigest()


class ExplanationGenerator:
    def __init__(self, data_folder: str = "data/train"):
        self._vectorizer = None
        self._matrix = None
        self._df = None
        self._load(data_folder)

    def _load(self, folder: str):
        files = glob.glob(f"{folder}/*.parquet")
        if not files:
            return

        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df["binary_label"] = df["label"].map(LABEL_MAP)
        df = df.dropna(subset=["binary_label", "explanation", "claim"])
        df = df[df["explanation"].str.strip() != ""]
        df["binary_label"] = df["binary_label"].astype(int)
        self._df = df.reset_index(drop=True)

        cache_key = _dataset_hash(self._df)
        cache_path = os.path.join(CACHE_DIR, f"expgen_{cache_key}.pkl")

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self._vectorizer, self._matrix = pickle.load(f)
        else:
            self._vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
            self._matrix = self._vectorizer.fit_transform(self._df["claim"])
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump((self._vectorizer, self._matrix), f)

    def get_explanation(self, claim: str, predicted_label: int) -> str:
        if self._vectorizer is None or self._df is None:
            return GENERIC_EXPLANATIONS.get(predicted_label, "")

        vec = self._vectorizer.transform([claim])
        sims = cosine_similarity(vec, self._matrix).flatten()

        # Only look for matches with the same predicted label
        mask = self._df["binary_label"] == predicted_label
        masked_sims = np.where(mask, sims, -1)
        best_idx = int(np.argmax(masked_sims))
        best_score = float(masked_sims[best_idx])

        if best_score >= SIMILARITY_THRESHOLD:
            return self._df["explanation"].iloc[best_idx]

        return GENERIC_EXPLANATIONS.get(predicted_label, "")
