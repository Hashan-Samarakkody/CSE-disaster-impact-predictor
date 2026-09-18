"""Phase 0 baseline freeze (Y1/Y2/Y3 improvement protocol, 2026-09-17)."""
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
from src.utils.artifact_store import artifact_file
ART = ROOT / "artifacts"
OUT = artifact_file("frozen_baseline.json")


def _f(x):
    """numpy -> json-safe, NaN -> None (json can't round-trip NaN portably)."""
    a = np.asarray(x, dtype=float)
    return [None if not np.isfinite(v) else float(v) for v in a.ravel()]


def main() -> None:
    dataset = pd.read_parquet(artifact_file("dataset.parquet"))
    results = pickle.loads((artifact_file("results_regression.pkl")).read_bytes())
    splits = pickle.loads((artifact_file("splits.pkl")).read_bytes())
    verdict = pd.read_parquet(artifact_file("verdict_table.parquet"))
    classification = pd.read_parquet(artifact_file("classification_summary.parquet"))

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
