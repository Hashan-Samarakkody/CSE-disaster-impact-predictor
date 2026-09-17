"""Phase 0 baseline freeze (Y1/Y2/Y3 improvement protocol, 2026-09-17).

Writes ONE json, `artifacts/frozen_baseline.json`, holding everything the improvement
work must be able to prove it did not disturb:

  * the git commit the freeze was taken at,
  * event ids (dataset row index) and event dates,
  * every target's values, in row order,
  * the walk-forward fold definitions (train/test index arrays),
  * every model's pooled out-of-fold predictions and metrics,
  * the bootstrap verdict rows.

`tests/test_y2_frozen.py` reads it back and fails if ANY Y2 number moves. Y1/Y3 are
recorded too, but only as a research record -- they are allowed to change, that is the
point of the work.

    python scripts/freeze_baseline.py
"""
from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ART = ROOT / "artifacts"
OUT = ART / "frozen_baseline.json"


def _f(x):
    """numpy -> json-safe, NaN -> None (json can't round-trip NaN portably)."""
    a = np.asarray(x, dtype=float)
    return [None if not np.isfinite(v) else float(v) for v in a.ravel()]


def main() -> None:
    dataset = pd.read_parquet(ART / "dataset.parquet")
    results = pickle.loads((ART / "results_regression.pkl").read_bytes())
    splits = pickle.loads((ART / "splits.pkl").read_bytes())
    verdict = pd.read_parquet(ART / "verdict_table.parquet")
    classification = pd.read_parquet(ART / "classification_summary.parquet")

    targets = [t for t in dataset.columns if t.split("_")[0] in {"Y1", "Y2", "Y3"}
               and pd.api.types.is_numeric_dtype(dataset[t])]

    payload = {
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                 text=True).stdout.strip(),
        "n_events": int(len(dataset)),
        "event_ids": [int(i) for i in dataset.index],
        "event_dates": [str(d.date()) for d in pd.to_datetime(dataset["event_date"])],
        "targets": {t: _f(dataset[t]) for t in targets},
        "folds": [{"train": [int(i) for i in np.asarray(s.train_index)],
                   "test": [int(i) for i in np.asarray(s.test_index)]} for s in splits],
        "models": {},
        "verdict": verdict.to_dict(orient="records"),
        "classification": classification.to_dict(orient="records"),
    }

    for model, per_target in results.items():
        payload["models"][model] = {}
        for target, store in per_target.items():
            payload["models"][model][target] = {
                "y_true": _f(np.concatenate(store["y_true"])),
                "y_pred": _f(np.concatenate(store["y_pred"])),
                "index": [int(i) for i in np.concatenate(store["index"])] if store.get("index") else None,
                "rmse": _f(store["rmse"]),
                "mae": _f(store["mae"]),
                "r2": _f(store["r2"]),
                "pooled_r2": float(store["pooled_r2"]),
            }

    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size / 1024:.0f} KB, "
          f"commit {payload['commit'][:8]}, {payload['n_events']} events, "
          f"{len(payload['folds'])} folds, {len(payload['models'])} models)")


if __name__ == "__main__":
    main()
