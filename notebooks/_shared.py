"""Shared plumbing for the split notebook pipeline.

Every notebook in this directory starts with `from _shared import *`. This module owns
the three things that would otherwise be re-declared (and drift) in nine places: where
things live, how cached artifacts are read and written, and the figure style.

The artifact cache is the reason the pipeline is split at all. Stage 01 hits live APIs
(World Bank via wbgapi, S&P 500 via yfinance) and parses 24 Excel workbooks; stages 03,
06 and 09 need none of that. Caching between stages means a figure tweak costs seconds
instead of a full re-run, and it closes the reproducibility gap flagged as audit item
P2-8: every cached file is written alongside a provenance record naming its retrieval
date and the library versions that produced it.
"""

from __future__ import annotations

import json
import os
import pickle
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Windows + torch + the Intel OpenMP runtime that ships with numpy will abort the
# process on a duplicate libiomp5md.dll unless this is set before torch is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---------------------------------------------------------------- paths

NOTEBOOK_DIR = Path(__file__).resolve().parent
REPO_ROOT = NOTEBOOK_DIR.parent
DATA_DIR = REPO_ROOT / "data"
ARTIFACT_DIR = REPO_ROOT / "artifacts"
FIGURE_DIR = REPO_ROOT / "docs" / "figures"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# `src` is imported as a package from the repo root, matching how the tests import it.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RANDOM_STATE = 42

# Y1_EventWindow_0_5_LogReturn_Pct / Y1_EventWindow_0_10_LogReturn_Pct are cumulative event-window returns, pre-declared in
# docs/EXTERNAL_DATA_PRE_DECLARATION.md Sec. 7.2. They are regression targets and
# drive the C4 label, so they must appear here -- stage 05 slices dataset[TARGET_COLS].
TARGET_COLS = ["Y1_ASPI_5D_Forward_LogReturn_Pct", "Y2_abnormal_volume", "Y3_recovery_days",
               "Y1_EventWindow_0_5_LogReturn_Pct", "Y1_EventWindow_0_10_LogReturn_Pct"]

# Definitional bounds from thesis Sec. 3.2.2. Clipping to them is projection onto the
# target's support: every true value already lies inside, so absolute error cannot
# increase on any point.
TARGET_BOUNDS = {
    "Y1_ASPI_5D_Forward_LogReturn_Pct": (None, None),
    "Y2_abnormal_volume": (-1.0, None),
    "Y3_recovery_days": (0.0, 90.0),
    # Cumulative log returns are unbounded in both directions, exactly like Y1.
    "Y1_EventWindow_0_5_LogReturn_Pct": (None, None),
    "Y1_EventWindow_0_10_LogReturn_Pct": (None, None),
}


def clip_to_bounds(target: str, values):
    lo, hi = TARGET_BOUNDS[target]
    return np.clip(np.asarray(values, dtype=float), lo, hi)


# One column per target naming the trading session on which that target's label is
# actually "known" -- used ONLY by `walk_forward.purge_horizon_overlap` to drop
# training rows whose label depends on prices/events dated on/after a fold's first
# test event (fold-boundary embargo). Per 2026-09-16 methodology-audit review
# (finding #5): a single shared Y1-5-day purge under-purged Y1_EventWindow_0_10 (a
# 10-trading-day horizon) and Y3 (up to 90 trading days), while over-purging Y2 (which
# has no forward horizon at all -- its label is known the same session). Each target
# now gets its own column, built in `feature_eng.build_targets`.
TARGET_LABEL_END_DATE_COL = {
    "Y1_ASPI_5D_Forward_LogReturn_Pct": "Y1_horizon_end_date",
    "Y1_EventWindow_0_5_LogReturn_Pct": "Y1_EventWindow_0_5_horizon_end_date",
    "Y1_EventWindow_0_10_LogReturn_Pct": "Y1_EventWindow_0_10_horizon_end_date",
    "Y2_abnormal_volume": "Y2_label_end_date",
    "Y3_recovery_days": "Y3_label_end_date",
}

# Same mapping, keyed by the stage-05 CLASSIFICATION label name instead of the
# regression target it is derived from (src/models/classifiers.py's LABELS dict) --
# notebook 05 had NO fold-boundary purge at all before the 2026-09-16 methodology-audit
# review (finding #5), even though every label depends on a future-looking target with
# its own horizon (up to 90 trading days for the two Y3-derived labels).
LABEL_END_DATE_COL = {
    "C1_negative_return": TARGET_LABEL_END_DATE_COL["Y1_ASPI_5D_Forward_LogReturn_Pct"],
    "C1b_adverse_move": TARGET_LABEL_END_DATE_COL["Y1_ASPI_5D_Forward_LogReturn_Pct"],
    "C2_volume_spike": TARGET_LABEL_END_DATE_COL["Y2_abnormal_volume"],
    "C3_recovers_in_90": TARGET_LABEL_END_DATE_COL["Y3_recovery_days"],
    "C3b_slow_recovery": TARGET_LABEL_END_DATE_COL["Y3_recovery_days"],
    "C4_car5_negative": TARGET_LABEL_END_DATE_COL["Y1_EventWindow_0_5_LogReturn_Pct"],
}


# ---------------------------------------------------------------- artifact cache


def _provenance_record() -> dict:
    """Library versions and a UTC timestamp, recorded beside every artifact.

    Without this the results are not reproducible: wbgapi and yfinance are pulled live
    and unpinned, so a re-run months later can silently return revised macro figures.
    """
    versions = {}
    for name in ("numpy", "pandas", "sklearn", "xgboost", "torch", "statsmodels",
                 "shap", "matplotlib", "wbgapi", "yfinance"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = "not installed"
    return {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "versions": versions,
    }


def save_frame(df: pd.DataFrame, name: str, note: str = "") -> Path:
    """Write a DataFrame to the artifact cache with a provenance sidecar."""
    path = ARTIFACT_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    meta = _provenance_record()
    meta.update({"artifact": name, "rows": int(len(df)),
                 "columns": list(map(str, df.columns)), "note": note})
    (ARTIFACT_DIR / f"{name}.provenance.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"cached {name}.parquet  ({len(df)} rows x {len(df.columns)} cols)")
    return path


def load_frame(name: str) -> pd.DataFrame:
    path = ARTIFACT_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact '{name}' is missing. Run the notebook stage that produces it "
            f"before this one -- see the stage table in architecture.md.")
    return pd.read_parquet(path)


def save_object(obj, name: str, note: str = "") -> Path:
    """Pickle a Python object (the nested `results` dicts, fitted models, fold splits).

    Pickle rather than parquet because `results` is a nested dict of ragged per-fold
    numpy arrays, which has no flat tabular form.
    """
    path = ARTIFACT_DIR / f"{name}.pkl"
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    meta = _provenance_record()
    meta.update({"artifact": name, "note": note, "type": type(obj).__name__})
    (ARTIFACT_DIR / f"{name}.provenance.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"cached {name}.pkl")
    return path


def load_object(name: str):
    path = ARTIFACT_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact '{name}' is missing. Run the notebook stage that produces it "
            f"before this one -- see the stage table in architecture.md.")
    with open(path, "rb") as fh:
        return pickle.load(fh)


def save_json(payload: dict, name: str) -> Path:
    path = ARTIFACT_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"cached {name}.json")
    return path


def load_json(name: str) -> dict:
    path = ARTIFACT_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Artifact '{name}' is missing. Run the stage that writes it.")
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_status() -> pd.DataFrame:
    """One row per cached artifact -- printed at the top of every stage so a reader can
    see which upstream stage produced the inputs, and when."""
    rows = []
    for meta_path in sorted(ARTIFACT_DIR.glob("*.provenance.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rows.append({"artifact": meta.get("artifact"),
                     "written_utc": meta.get("written_utc"),
                     "rows": meta.get("rows", ""),
                     "note": meta.get("note", "")})
    if not rows:
        return pd.DataFrame(columns=["artifact", "written_utc", "rows", "note"])
    return pd.DataFrame(rows)


__all__ = [
    "np", "pd", "Path", "json",
    "NOTEBOOK_DIR", "REPO_ROOT", "DATA_DIR", "ARTIFACT_DIR", "FIGURE_DIR",
    "RANDOM_STATE", "TARGET_COLS", "TARGET_BOUNDS", "clip_to_bounds",
    "TARGET_LABEL_END_DATE_COL", "LABEL_END_DATE_COL",
    "save_frame", "load_frame", "save_object", "load_object", "save_json", "load_json",
    "artifact_status",
]
