"""Build Leave-One-Out fold definitions over the 13 cases.

Each fold uses one case as the held-out test set; the remaining cases are used
for training (with a small slice of the training set reserved for in-fold
validation / early stopping).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from config import MANIFEST_FILE, FOLDS_FILE


def main() -> None:
    if not MANIFEST_FILE.exists():
        raise SystemExit(f"manifest.jsonl not found; run prepare_segments.py first.")

    cases = []
    seen = set()
    with MANIFEST_FILE.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["case"] not in seen:
                seen.add(rec["case"])
                cases.append(rec["case"])

    cases.sort()
    print(f"Found {len(cases)} cases.")
    for c in cases:
        print(f"  - {c}")

    folds = []
    for i, test_case in enumerate(cases):
        train_cases = [c for c in cases if c != test_case]
        # Use the *first* training case as in-fold validation set
        # (stable, reproducible — same case is val whenever it's not held out)
        val_case = train_cases[0]
        folds.append({
            "fold": i,
            "test_case": test_case,
            "val_case": val_case,
            "train_cases": [c for c in train_cases if c != val_case],
        })

    payload = {
        "n_folds": len(folds),
        "all_cases": cases,
        "folds": folds,
    }
    FOLDS_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {len(folds)} folds -> {FOLDS_FILE}")


if __name__ == "__main__":
    main()
