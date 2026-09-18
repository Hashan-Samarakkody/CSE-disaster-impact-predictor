"""One import line for every notebook stage.

This is a thin facade over the real modules in src/. It exists so a notebook can
open with a single import instead of fifteen, and so that the paths, seed and
target definitions every stage needs come from one place.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NOTEBOOK_DIR = Path(__file__).resolve().parent
REPO_ROOT = NOTEBOOK_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.settings import (  # noqa: E402
    ARTIFACT_DIR, CSE_MARKET_INDICES_FILE, DATA_DIR, EMDAT_DISASTERS_FILE,
    EXTERNAL_DATA_DIR, FIGURE_DIR, LABEL_END_DATE_COL, PROJECT_ROOT, RANDOM_STATE,
    RAW_DATA_DIR, TARGET_BOUNDS, TARGET_COLS, TARGET_LABEL_END_DATE_COL,
    clip_to_bounds, ensure_output_directories)
from src.training.inner_cv import purged_inner_cv  # noqa: E402
from src.training.walk_forward import (  # noqa: E402
    MEDIAN_IMPUTE_COLS, median_impute_from_train)
from src.utils.artifact_store import (  # noqa: E402
    artifact_file, artifact_status, load_frame, load_json, load_object, save_frame,
    save_json, save_object)

ensure_output_directories()

__all__ = [
    "np", "pd", "Path", "json",
    "NOTEBOOK_DIR", "REPO_ROOT", "PROJECT_ROOT", "DATA_DIR", "RAW_DATA_DIR",
    "EXTERNAL_DATA_DIR", "ARTIFACT_DIR", "FIGURE_DIR",
    "CSE_MARKET_INDICES_FILE", "EMDAT_DISASTERS_FILE",
    "RANDOM_STATE", "TARGET_COLS", "TARGET_BOUNDS", "clip_to_bounds",
    "TARGET_LABEL_END_DATE_COL", "LABEL_END_DATE_COL", "purged_inner_cv",
    "MEDIAN_IMPUTE_COLS", "median_impute_from_train",
    "artifact_file", "save_frame", "load_frame", "save_object", "load_object",
    "save_json", "load_json", "artifact_status",
]
