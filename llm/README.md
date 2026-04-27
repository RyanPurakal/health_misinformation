# llm/

Explanation retrieval for the research stack. Returns a human-written explanation paired with the predicted label rather than generating text.

## Files

| File | Role |
|---|---|
| `explanation_generator.py` | `ExplanationGenerator` — builds a TF-IDF index over training claims at startup, then finds the nearest matching explanation at query time |

## How it works

1. At construction, all training rows with non-empty `explanation` fields are loaded (1,265 rows after filtering).
2. A TF-IDF matrix is built over the `claim` column and cached to `.cache/explanations/` keyed by a dataset hash — subsequent startups skip rebuilding.
3. At query time, the input claim is vectorised and cosine similarity is computed against the index. Only candidates with the same predicted label are considered.
4. If the best similarity score is ≥ 0.25, the matching row's `explanation` is returned. Otherwise a generic fallback string is used.

## Key design decision

Retrieval over generation avoids hallucinating medical explanations. The 0.25 similarity threshold was chosen so that clearly unrelated claims (similarity near 0) always fall back to the generic text.
