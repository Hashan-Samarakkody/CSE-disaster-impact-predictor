"""Central paths, seeds and target definitions used across the whole project.

Every module and notebook imports these instead of hard coding a path or a
constant, so the repository can be cloned anywhere and still run.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Windows plus torch plus the Intel OpenMP runtime shipped with numpy will abort
# the process on a duplicate libiomp5md.dll unless this is set before torch loads.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
TABLE_DIR = ARTIFACT_DIR / "tables"
MODEL_DIR = ARTIFACT_DIR / "models"
RESULT_DIR = ARTIFACT_DIR / "results"
ARTIFACT_FIGURE_DIR = ARTIFACT_DIR / "figures"
EXTERNAL_CACHE_DIR = ARTIFACT_DIR / "external"

DOCS_DIR = PROJECT_ROOT / "docs"
FIGURE_DIR = DOCS_DIR / "figures"
THESIS_MATERIALS_DIR = DOCS_DIR / "thesis_materials"

# Raw input filenames. The yearly per security workbooks are discovered by pattern
# in src/data/cse_market_data.py, so only the single file inputs are named here.
CSE_MARKET_INDICES_FILE = RAW_DATA_DIR / "cse_market_indices_daily.xls"
EMDAT_DISASTERS_FILE = RAW_DATA_DIR / "emdat_sri_lanka_disasters.xlsx"
MARKET_CAPITALIZATION_FILE = EXTERNAL_DATA_DIR / "cse_market_capitalization.csv"

RANDOM_STATE = 42

# The three research targets, defined in docs/TARGET_DEFINITION_PROTOCOL.md and frozen
# there on 2026-09-19 at commit 928a255, before any performance under these definitions
# was observed. P0 is the last close before the prediction origin.
ASPI_PERCENTAGE_CHANGE = "Y1_ASPI_5D_Forward_LogReturn_Pct"
VOLUME_CRASH_MAGNITUDE = "Y2_5D_Forward_AbnormalVolume_LogRatio"
MARKET_RECOVERY_DAYS = "Y3_ASPI_Recovery_Time"

# Pre registered horizon sensitivity analyses of target one. Never promoted to primary.
ASPI_PERCENTAGE_CHANGE_10D = "Y1_ASPI_10D_Forward_LogReturn_Pct"
ASPI_SENSITIVITY_COLS = [f"Y1_ASPI_{h}D_Forward_LogReturn_Pct" for h in (10, 15, 20)]

TARGET_COLS = [ASPI_PERCENTAGE_CHANGE, VOLUME_CRASH_MAGNITUDE, MARKET_RECOVERY_DAYS,
               ASPI_PERCENTAGE_CHANGE_10D]

# Columns kept so every target can be recomputed by hand. Never predictors.
TARGET_AUDIT_COLS = ["prediction_origin_session", "P0", "Y2_V_base", "Y2_V_future5"]

# Definitional support. Y2 is a log ratio, so it is unbounded below as well as above.
TARGET_BOUNDS = {
    ASPI_PERCENTAGE_CHANGE: (None, None),
    VOLUME_CRASH_MAGNITUDE: (None, None),
    MARKET_RECOVERY_DAYS: (0.0, 90.0),
    ASPI_PERCENTAGE_CHANGE_10D: (None, None),
}

# The session on which each target's label becomes known. The walk forward purge reads
# these; Y2's is the session of V5, not the prediction origin.
TARGET_LABEL_END_DATE_COL = {
    ASPI_PERCENTAGE_CHANGE: "Y1_horizon_end_date",
    ASPI_PERCENTAGE_CHANGE_10D: "Y1_10D_horizon_end_date",
    VOLUME_CRASH_MAGNITUDE: "Y2_horizon_end_date",
    MARKET_RECOVERY_DAYS: "Y3_label_end_date",
}

LABEL_END_DATE_COL = {
    # C0 is the protocol's stage-1 label: D is decided by P1..P5, so its label closes on
    # the same session as Y1's five-session horizon.
    "C0_drawdown_occurs": TARGET_LABEL_END_DATE_COL[ASPI_PERCENTAGE_CHANGE],
    "C1_negative_return": TARGET_LABEL_END_DATE_COL[ASPI_PERCENTAGE_CHANGE],
    "C1b_adverse_move": TARGET_LABEL_END_DATE_COL[ASPI_PERCENTAGE_CHANGE],
    "C2_volume_spike": TARGET_LABEL_END_DATE_COL[VOLUME_CRASH_MAGNITUDE],
    "C3_recovers_in_90": TARGET_LABEL_END_DATE_COL[MARKET_RECOVERY_DAYS],
    "C3b_slow_recovery": TARGET_LABEL_END_DATE_COL[MARKET_RECOVERY_DAYS],
}


# Every outcome column, every constituent of one, and every bookkeeping date: what a
# predictor must never be built from. Feature construction is a denylist, so a
# target-family column missing here is silently admitted as a feature.
NON_FEATURE_COLS = (
    set(TARGET_COLS)
    | set(ASPI_SENSITIVITY_COLS)        # 10/15/20D forward returns share P0 with Y1
    | set(TARGET_AUDIT_COLS)            # P0, V_base, V_future5, prediction_origin_session
    | set(TARGET_LABEL_END_DATE_COL.values())
    | {f"Y1_{h}D_horizon_end_date" for h in (10, 15, 20)}
    | {"Y3_event_observed", "Y3_censored", "Y3_censor_reason", "Y3_drawdown_occurred"}
)


def assert_no_target_leakage(feature_cols) -> None:
    """Raise if any outcome-derived column reached the feature set."""
    leaked = sorted(set(feature_cols) & NON_FEATURE_COLS)
    if leaked:
        raise AssertionError(f"outcome columns leaked into FEATURE_COLS: {leaked}")


def clip_to_bounds(target: str, values):
    """Clip predictions to the target's definitional support."""
    low, high = TARGET_BOUNDS[target]
    return np.clip(np.asarray(values, dtype=float), low, high)


def ensure_output_directories() -> None:
    """Create the generated output directories if they do not exist yet."""
    for directory in (TABLE_DIR, MODEL_DIR, RESULT_DIR, ARTIFACT_FIGURE_DIR,
                      EXTERNAL_CACHE_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
