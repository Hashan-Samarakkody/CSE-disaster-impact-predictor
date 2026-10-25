"""Build recovery_state_descriptives.parquet — descriptives by Y3 state."""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.utils.artifact_store import artifact_file, save_frame
from src.targets.event_targets import (
    DRAWDOWN_WINDOW, MAX_RECOVERY_SESSIONS,
)


# Recovery threshold: ASPI must return to baseline (P0), i.e. ratio >= 1.0
RECOVERY_THRESHOLD = 1.0  # price >= P0


def _descriptives(series: pd.Series) -> dict:
    vals = series.dropna()
    if len(vals) == 0:
        return {
            "n": 0, "mean": float("nan"), "median": float("nan"),
            "q25": float("nan"), "q75": float("nan"),
            "min": float("nan"), "max": float("nan"),
        }
    return {
        "n": len(vals),
        "mean": float(vals.mean()),
        "median": float(vals.median()),
        "q25": float(vals.quantile(0.25)),
        "q75": float(vals.quantile(0.75)),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }


def main():
    dataset = pd.read_parquet(artifact_file("dataset.parquet"))

    y3_col = "Y3_ASPI_Recovery_Time"
    obs_col = "Y3_event_observed"
    censored_col = "Y3_censored"
    drawdown_col = "Y3_drawdown_occurred"
    censor_reason_col = "Y3_censor_reason"

    # Define states
    recovered_mask = (dataset[obs_col] == 1) & (dataset[drawdown_col] == True)
    no_drawdown_mask = dataset[drawdown_col] == False
    # censored: event not observed but drawdown occurred
    censored_mask = (dataset[obs_col] == 0) & (dataset[drawdown_col] == True)

    # Try to identify truncated_by_next from censor_reason
    if censor_reason_col in dataset.columns:
        trunc_mask = (
            censored_mask &
            dataset[censor_reason_col].str.contains("next_disaster", na=False)
        )
        cap_mask = (
            censored_mask &
            dataset[censor_reason_col].str.contains("90_day_cap", na=False)
        )
    else:
        trunc_mask = pd.Series(False, index=dataset.index)
        cap_mask = censored_mask.copy()

    rows = []

    for state, mask in [
        ("recovered", recovered_mask),
        ("no_drawdown", no_drawdown_mask),
        ("censored_by_cap", cap_mask),
        ("censored_by_next_disaster", trunc_mask),
    ]:
        sub = dataset.loc[mask, y3_col]
        desc = _descriptives(sub)
        desc.update({
            "state": state,
            "drawdown_threshold": f"min(P1..P{DRAWDOWN_WINDOW}) < P0",
            "detection_gate_length": int(DRAWDOWN_WINDOW),
            "recovery_threshold": RECOVERY_THRESHOLD,
            "horizon_cap": int(MAX_RECOVERY_SESSIONS),
        })
        rows.append(desc)

    df = pd.DataFrame(rows)
    # Reorder columns
    df = df[[
        "state", "n", "mean", "median", "q25", "q75", "min", "max",
        "drawdown_threshold", "detection_gate_length",
        "recovery_threshold", "horizon_cap",
    ]]

    save_frame(df, "recovery_state_descriptives",
               note="Descriptives by Y3 recovery state. Thresholds read from "
                    "src/targets/event_targets.py constants: DRAWDOWN_WINDOW, "
                    "MAX_RECOVERY_SESSIONS. Recovery requires ASPI >= P0 after prior dip.")


if __name__ == "__main__":
    main()
