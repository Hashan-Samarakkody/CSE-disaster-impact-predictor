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

# The three research targets. Column names are frozen: they appear inside cached
# artifacts and inside the regression test that holds the volume target fixed.
ASPI_PERCENTAGE_CHANGE = "Y1_ASPI_5D_Forward_LogReturn_Pct"
VOLUME_CRASH_MAGNITUDE = "Y2_abnormal_volume"
MARKET_RECOVERY_DAYS = "Y3_recovery_days"

# Pre registered ten session horizon variant of the ASPI target. It is a sensitivity
# analysis of target one, not a fourth research target.
ASPI_PERCENTAGE_CHANGE_10D = "Y1_EventWindow_0_10_LogReturn_Pct"

TARGET_COLS = [ASPI_PERCENTAGE_CHANGE, VOLUME_CRASH_MAGNITUDE, MARKET_RECOVERY_DAYS,
               ASPI_PERCENTAGE_CHANGE_10D]

# Definitional support of each target, from thesis section 3.2.2. Clipping a
# prediction to these bounds can never increase its absolute error.
TARGET_BOUNDS = {
    ASPI_PERCENTAGE_CHANGE: (None, None),
    VOLUME_CRASH_MAGNITUDE: (-1.0, None),
    MARKET_RECOVERY_DAYS: (0.0, 90.0),
    ASPI_PERCENTAGE_CHANGE_10D: (None, None),
}

# The session on which each target's label becomes known. Used only to purge
# training rows whose label reaches into a test period.
TARGET_LABEL_END_DATE_COL = {
    ASPI_PERCENTAGE_CHANGE: "Y1_horizon_end_date",
    ASPI_PERCENTAGE_CHANGE_10D: "Y1_EventWindow_0_10_horizon_end_date",
    VOLUME_CRASH_MAGNITUDE: "Y2_label_end_date",
    MARKET_RECOVERY_DAYS: "Y3_label_end_date",
}

# Same mapping keyed by the classification label derived from each target.
LABEL_END_DATE_COL = {
    "C1_negative_return": TARGET_LABEL_END_DATE_COL[ASPI_PERCENTAGE_CHANGE],
    "C1b_adverse_move": TARGET_LABEL_END_DATE_COL[ASPI_PERCENTAGE_CHANGE],
    "C2_volume_spike": TARGET_LABEL_END_DATE_COL[VOLUME_CRASH_MAGNITUDE],
    "C3_recovers_in_90": TARGET_LABEL_END_DATE_COL[MARKET_RECOVERY_DAYS],
    "C3b_slow_recovery": TARGET_LABEL_END_DATE_COL[MARKET_RECOVERY_DAYS],
}


def clip_to_bounds(target: str, values):
    """Clip predictions to the target's definitional support."""
    low, high = TARGET_BOUNDS[target]
    return np.clip(np.asarray(values, dtype=float), low, high)


def ensure_output_directories() -> None:
    """Create the generated output directories if they do not exist yet."""
    for directory in (TABLE_DIR, MODEL_DIR, RESULT_DIR, ARTIFACT_FIGURE_DIR,
                      EXTERNAL_CACHE_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
