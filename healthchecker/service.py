"""Load models once, run `ClaimCheckerService.analyze(text)` for CLI or HTTP."""

from __future__ import annotations

import glob
import hashlib
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import trafilatura
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_TRAIN = ROOT / "data" / "train"
EMBED_CACHE_DIR = ROOT / ".cache" / "claim_embeddings"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

LABEL_MAP = {0: 1, 1: 0, 2: 1, 3: 1, -1: None}
BINARY_NAMES = {0: "RELIABLE", 1: "MISINFORMATION"}
MIN_CONFIDENCE = 0.65

GENERIC_EXPLANATION = {
    0: (
        "No training claim was similar enough to reuse its fact-check text. "
        "The classifier leans toward RELIABLE for this input based on patterns in the training data."
    ),
    1: (
        "No training claim was similar enough to reuse its fact-check text. "
        "The classifier leans toward MISINFORMATION for this input based on patterns in the training data."
    ),
    -1: (
        "The classifier is not confident enough to make a reliable judgment on this claim. "
        "Try providing more context or a URL to an article for a better result."
    ),
}

MIN_SEM_URL = 0.58
MARGIN_URL = 0.035
MIN_SEM_DIRECT_LONG = 0.58
MIN_SEM_DIRECT_SHORT = 0.62
SHORT_CLAIM_MAX_WORDS = 5
MARGIN_DIRECT_LONG = 0.018
MARGIN_DIRECT_SHORT = 0.022


def dataset_fingerprint(folder: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(glob.glob(str(folder / "*.parquet"))):
        p = Path(path)
        st = p.stat()
        h.update(str(p.resolve()).encode())
        h.update(str(st.st_size).encode())
        h.update(str(st.st_mtime_ns).encode())
    return h.hexdigest()


def load_parquet_folder(folder: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(folder / "*.parquet")))
    if not files:
        raise FileNotFoundError(
            f"No parquet files in {folder}. Add *.parquet under data/train/ or adjust paths."
        )
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def prepare_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["label_bin"] = out["label"].map(LABEL_MAP)
    out = out.dropna(subset=["label_bin"])
    out["label_bin"] = out["label_bin"].astype(int)
    return out


def row_model_text(row: pd.Series) -> str:
    claim = str(row.get("claim") or "").strip()
    mt = row.get("main_text")
    if pd.isna(mt) or not str(mt).strip():
        return claim
    snippet = str(mt)[:4000]
    return f"{claim}\n\n{snippet}"


def train_classifier(df: pd.DataFrame) -> Pipeline:
    texts = [row_model_text(row) for _, row in df.iterrows()]
    y = df["label_bin"].to_numpy()
    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2),
            ),
            (
                "clf",
                LogisticRegression(max_iter=2000, class_weight="balanced"),
            ),
        ]
    )
    pipe.fit(texts, y)
    return pipe


def encode_claims_matrix(
    model: SentenceTransformer, claims: list[str]
) -> np.ndarray:
    emb = model.encode(
        claims,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)


def load_or_build_claim_embeddings(
    df: pd.DataFrame, model: SentenceTransformer, *, verbose: bool = False
) -> np.ndarray:
    fp = dataset_fingerprint(DATA_TRAIN)
    cache_path = EMBED_CACHE_DIR / f"v2_{fp[:40]}.pkl"
    claims = df["claim"].astype(str).tolist()

    if cache_path.is_file():
        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
            if (
                payload.get("fingerprint") == fp
                and payload.get("model") == EMBED_MODEL_NAME
                and payload.get("n") == len(claims)
            ):
                mat = payload["embeddings"]
                if isinstance(mat, np.ndarray) and mat.shape[0] == len(claims):
                    return mat
        except Exception:
            pass

    if verbose:
        print(
            "Computing semantic embeddings for training claims (cached for later runs)…",
            flush=True,
        )
    EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    mat = encode_claims_matrix(model, claims)
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(
                {
                    "fingerprint": fp,
                    "model": EMBED_MODEL_NAME,
                    "n": len(claims),
                    "embeddings": mat,
                },
                f,
            )
    except OSError:
        pass
    return mat


def semantic_nearest_explanation(
    model: SentenceTransformer,
    claim_embeddings: np.ndarray,
    explanations: list[str],
    query_text: str,
    min_sim: float,
    min_margin: float,
) -> tuple[str | None, float, float]:
    q = (query_text or "").strip()
    if not q:
        return None, 0.0, 0.0

    q_emb = model.encode(
        [q], show_progress_bar=False, normalize_embeddings=True
    )[0]
    sims = claim_embeddings @ q_emb.astype(np.float32)
    n = int(sims.shape[0])
    if n == 0:
        return None, 0.0, 0.0

    best_i = int(np.argmax(sims))
    best = float(sims[best_i])

    if n == 1:
        if best < min_sim:
            return None, best, 0.0
        return explanations[best_i], best, 0.0

    sims_copy = sims.copy()
    sims_copy[best_i] = -np.inf
    second_i = int(np.argmax(sims_copy))
    second = float(sims[second_i])
    margin = best - second

    if best < min_sim or margin < min_margin:
        return None, best, margin
    return explanations[best_i], best, margin


def leading_excerpt(
    text: str, *, max_sentences: int = 4, max_chars: int = 900
) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
        if len(out) >= max_sentences:
            break
    joined = " ".join(out)
    if len(joined) <= max_chars:
        return joined
    return joined[: max_chars - 1].rsplit(" ", 1)[0] + "…"


def text_for_explanation_from_article(article: str, title: str | None) -> str:
    lead = leading_excerpt(article, max_sentences=4, max_chars=800)
    bits: list[str] = []
    if title and title.strip():
        bits.append(title.strip())
    if lead:
        bits.append(lead)
    s = " ".join(bits).strip()
    if not s:
        s = article[:500].strip()
    return s[:1200]


def looks_like_url(s: str) -> bool:
    t = s.strip()
    return bool(re.match(r"^https?://", t, re.I))


def fetch_article_content(url: str) -> tuple[str | None, str | None]:
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None, None
        from trafilatura.metadata import extract_metadata

        meta = extract_metadata(downloaded)
        title = (meta.title or "").strip() if meta else None
        if title and len(title) > 240:
            title = title[:240] + "…"
        text = trafilatura.extract(downloaded)
        return text, title
    except Exception:
        return None, None


def direct_threshold_for_query(raw: str) -> tuple[float, float]:
    n = len(raw.split())
    if n <= SHORT_CLAIM_MAX_WORDS:
        return MIN_SEM_DIRECT_SHORT, MARGIN_DIRECT_SHORT
    return MIN_SEM_DIRECT_LONG, MARGIN_DIRECT_LONG


def build_query_text(user_input: str) -> tuple[str, str, float, float]:
    raw = user_input.strip()
    if looks_like_url(raw):
        article, title = fetch_article_content(raw)
        if not article or not article.strip():
            return "", "", MIN_SEM_URL, MARGIN_URL
        combined = f"Article extracted from URL:\n\n{article[:12000]}"
        match_text = text_for_explanation_from_article(article, title)
        return combined, match_text, MIN_SEM_URL, MARGIN_URL
    min_s, min_m = direct_threshold_for_query(raw)
    return raw, raw[:2000].strip(), min_s, min_m


@dataclass
class ClaimCheckerService:
    """Holds trained classifier + embedding model; thread-safe for read-only `analyze`."""

    df: pd.DataFrame
    clf: Pipeline
    embed_model: SentenceTransformer
    claim_embeddings: np.ndarray
    explanations: list[str]

    @classmethod
    def load(cls, *, verbose: bool = True) -> ClaimCheckerService:
        if verbose:
            print("Loading training data…", flush=True)
        df = prepare_labels(load_parquet_folder(DATA_TRAIN))
        if verbose:
            print(f"Training classifier on {len(df)} examples…", flush=True)
        clf = train_classifier(df)
        if verbose:
            print(f"Loading embedding model ({EMBED_MODEL_NAME})…", flush=True)
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        claim_emb = load_or_build_claim_embeddings(
            df, embed_model, verbose=verbose
        )
        explanations = df["explanation"].astype(str).tolist()
        return cls(
            df=df,
            clf=clf,
            embed_model=embed_model,
            claim_embeddings=claim_emb,
            explanations=explanations,
        )

    def analyze(self, user_input: str) -> dict[str, Any]:
        """
        Returns JSON-serializable dict. On URL fetch failure, includes `error` key.
        """
        line = user_input.strip()
        if not line:
            return {"error": "empty", "message": "Enter a claim or URL."}

        q_full, q_match, min_sim, min_margin = build_query_text(line)
        if looks_like_url(line) and not q_full:
            return {
                "error": "fetch_failed",
                "message": (
                    "Could not extract article text (blocked, paywall, or non-HTML). "
                    "Try another URL or paste the article text."
                ),
            }

        pred = int(self.clf.predict([q_full])[0])
        proba = self.clf.predict_proba([q_full])[0]
        conf = float(proba[pred])
        is_uncertain = conf < MIN_CONFIDENCE
        display_label = -1 if is_uncertain else pred

        expl, sim, margin = semantic_nearest_explanation(
            self.embed_model,
            self.claim_embeddings,
            self.explanations,
            q_match,
            min_sim=min_sim,
            min_margin=min_margin,
        )
        used_dataset = expl is not None
        if expl is None:
            expl = GENERIC_EXPLANATION[display_label]

        label_str = "UNCERTAIN" if is_uncertain else BINARY_NAMES[pred]
        return {
            "label": label_str,
            "confidence": conf,
            "explanation": expl,
            "meta": {
                "used_dataset_explanation": used_dataset,
                "semantic_similarity": sim,
                "semantic_margin": margin,
                "min_similarity": min_sim,
                "min_margin": min_margin,
                "uncertain": is_uncertain,
                "is_url": looks_like_url(line),
            },
        }
