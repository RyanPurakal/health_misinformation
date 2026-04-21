"""
Explanation generator for health claim classification.

Uses semantic nearest-neighbor search over training claim embeddings
(sentence-transformers/all-MiniLM-L6-v2) to retrieve the most relevant
fact-check explanation from the training dataset.

If no training claim is similar enough, returns a generic fallback
based on the predicted label.

Usage:
    from llm.explanation_generator import ExplanationGenerator

    gen = ExplanationGenerator()
    explanation = gen.explain("Vaccines cause autism", predicted_label=1)
    print(explanation)
"""

from __future__ import annotations

import glob
import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA_TRAIN = ROOT / "data" / "train"
EMBED_CACHE_DIR = ROOT / ".cache" / "explanations"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LABEL_MAP = {0: 1, 1: 0, 2: 1, 3: 1, -1: None}

GENERIC = {
    0: (
        "No closely matching fact-check was found in the training data. "
        "The classifier leans toward RELIABLE based on patterns in the training data."
    ),
    1: (
        "No closely matching fact-check was found in the training data. "
        "The classifier leans toward MISINFORMATION based on patterns in the training data."
    ),
}

# Similarity thresholds
MIN_SIM = 0.55
MIN_MARGIN = 0.02


def _fingerprint(folder: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(glob.glob(str(folder / "*.parquet"))):
        p = Path(path)
        st = p.stat()
        h.update(str(p.resolve()).encode())
        h.update(str(st.st_size).encode())
        h.update(str(st.st_mtime_ns).encode())
    return h.hexdigest()


def _load_training_data() -> pd.DataFrame:
    files = sorted(glob.glob(str(DATA_TRAIN / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files in {DATA_TRAIN}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["label_bin"] = df["label"].map(LABEL_MAP)
    df = df.dropna(subset=["label_bin"])
    df["label_bin"] = df["label_bin"].astype(int)
    return df


class ExplanationGenerator:
    def __init__(self, model_name: str = EMBED_MODEL_NAME) -> None:
        print(f"Loading embedding model ({model_name})…", flush=True)
        self._model = SentenceTransformer(model_name)

        print("Loading training data for explanation lookup…", flush=True)
        df = _load_training_data()
        self._claims = df["claim"].astype(str).tolist()
        self._explanations = df["explanation"].astype(str).tolist()
        self._labels = df["label_bin"].tolist()
        self._embeddings = self._load_or_build_embeddings(df)

    def _load_or_build_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        fp = _fingerprint(DATA_TRAIN)
        cache_path = EMBED_CACHE_DIR / f"expgen_{fp[:40]}.pkl"

        if cache_path.is_file():
            try:
                with open(cache_path, "rb") as f:
                    payload = pickle.load(f)
                if (
                    payload.get("fingerprint") == fp
                    and payload.get("model") == EMBED_MODEL_NAME
                    and payload.get("n") == len(self._claims)
                ):
                    mat = payload["embeddings"]
                    if isinstance(mat, np.ndarray) and mat.shape[0] == len(self._claims):
                        return mat
            except Exception:
                pass

        print("Computing embeddings for training claims (cached for later runs)…", flush=True)
        EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        mat = self._model.encode(
            self._claims,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        mat = np.asarray(mat, dtype=np.float32)

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {
                        "fingerprint": fp,
                        "model": EMBED_MODEL_NAME,
                        "n": len(self._claims),
                        "embeddings": mat,
                    },
                    f,
                )
        except OSError:
            pass

        return mat

    def explain(
        self,
        query: str,
        predicted_label: int,
        min_sim: float = MIN_SIM,
        min_margin: float = MIN_MARGIN,
    ) -> str:
        """
        Return an explanation string for the given query and predicted label.

        Tries to find a semantically similar training claim and reuse its
        fact-check explanation. Falls back to a generic message if no
        sufficiently similar claim is found.
        """
        query = (query or "").strip()
        if not query:
            return GENERIC[predicted_label]

        q_emb = self._model.encode(
            [query], show_progress_bar=False, normalize_embeddings=True
        )[0].astype(np.float32)

        sims = self._embeddings @ q_emb
        n = int(sims.shape[0])
        if n == 0:
            return GENERIC[predicted_label]

        best_i = int(np.argmax(sims))
        best = float(sims[best_i])

        if n == 1:
            if best < min_sim:
                return GENERIC[predicted_label]
            return self._explanations[best_i]

        sims_tmp = sims.copy()
        sims_tmp[best_i] = -np.inf
        second = float(sims[int(np.argmax(sims_tmp))])
        margin = best - second

        if best < min_sim or margin < min_margin:
            return GENERIC[predicted_label]

        return self._explanations[best_i]

    def explain_with_metadata(
        self,
        query: str,
        predicted_label: int,
        min_sim: float = MIN_SIM,
        min_margin: float = MIN_MARGIN,
    ) -> dict:
        """
        Same as explain() but also returns similarity scores and matched claim.
        Returns a dict with keys: explanation, matched_claim, similarity, margin, is_generic.
        """
        query = (query or "").strip()
        if not query:
            return {
                "explanation": GENERIC[predicted_label],
                "matched_claim": None,
                "similarity": 0.0,
                "margin": 0.0,
                "is_generic": True,
            }

        q_emb = self._model.encode(
            [query], show_progress_bar=False, normalize_embeddings=True
        )[0].astype(np.float32)

        sims = self._embeddings @ q_emb
        n = int(sims.shape[0])

        best_i = int(np.argmax(sims))
        best = float(sims[best_i])
        margin = 0.0

        if n > 1:
            sims_tmp = sims.copy()
            sims_tmp[best_i] = -np.inf
            second = float(sims[int(np.argmax(sims_tmp))])
            margin = best - second

        passed = best >= min_sim and (n == 1 or margin >= min_margin)
        if passed:
            return {
                "explanation": self._explanations[best_i],
                "matched_claim": self._claims[best_i],
                "similarity": best,
                "margin": margin,
                "is_generic": False,
            }

        return {
            "explanation": GENERIC[predicted_label],
            "matched_claim": None,
            "similarity": best,
            "margin": margin,
            "is_generic": True,
        }


if __name__ == "__main__":
    gen = ExplanationGenerator()
    tests = [
        ("Vaccines cause autism.", 1),
        ("Drinking bleach cures COVID-19.", 1),
        ("Handwashing prevents the spread of germs.", 0),
    ]
    for claim, label in tests:
        result = gen.explain_with_metadata(claim, label)
        print(f"Claim   : {claim}")
        print(f"Explanation: {result['explanation'][:200]}")
        print(f"Matched : {result['matched_claim']}")
        print(f"Sim={result['similarity']:.3f}  Margin={result['margin']:.3f}  Generic={result['is_generic']}")
        print()
