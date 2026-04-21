#!/usr/bin/env python3
"""
Interactive health claim checker (CLI).

For a browser UI: `uvicorn web.app:app --reload` from the project root, then open http://127.0.0.1:8000
"""

from __future__ import annotations

from healthchecker.service import ClaimCheckerService


def main() -> None:
    svc = ClaimCheckerService.load(verbose=True)
    print(
        "Ready. Enter a health claim, a URL to an article, or 'quit'.\n",
        flush=True,
    )

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line or line.lower() in ("quit", "exit", "q"):
            break

        result = svc.analyze(line)
        if result.get("error") == "fetch_failed":
            print(result["message"] + "\n")
            continue
        if result.get("error") == "empty":
            print(result["message"] + "\n")
            continue

        pred_label = result["label"]
        conf = result["confidence"]
        expl = result["explanation"]
        meta = result["meta"]
        used_dataset = meta["used_dataset_explanation"]
        is_uncertain = meta["uncertain"]
        min_sim = meta["min_similarity"]
        min_margin = meta["min_margin"]
        sim = meta["semantic_similarity"]
        margin = meta["semantic_margin"]

        print(f"Label: {pred_label}")
        print(
            f"Confidence: {conf:.3f}"
            + (" (below threshold — result is uncertain)" if is_uncertain else "")
        )
        if used_dataset:
            print(
                f"(Dataset fact-check text — semantic similarity {sim:.3f}, "
                f"need ≥{min_sim:.2f}; margin vs 2nd {margin:.3f}, need ≥{min_margin:.3f})"
            )
        else:
            print(
                f"(No claim passed semantic gates — best {sim:.3f} vs need ≥{min_sim:.2f}, "
                f"margin {margin:.3f} vs need ≥{min_margin:.3f}; generic explanation.)"
            )
        print(f"Explanation: {expl}\n")


if __name__ == "__main__":
    main()
