"""Event-study inference on the realised market response (Revision 2, T11).

Tests whether the CSE actually reacted to a qualifying disaster, which is a different
question from whether the reaction can be forecast. Writes the cumulative average abnormal
return and volume by event-time session, with every published test statistic, and the
event-time figures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.event_study import (EVENT_WINDOW, build_abnormal_panel,
                                        event_study_table)
from src.targets.abnormal_returns import align_market_factor
from src.utils.artifact_store import artifact_file, save_frame
from src.visualization import event_study_figures as figures

# Cumulation starts on the first complete session after the origin, matching the targets.
CUMULATION_START = 1


def load_daily() -> pd.DataFrame:
    """Daily ASPI return, log volume and the market factor, on the CSE calendar."""
    market_feats = pd.read_parquet(artifact_file("market_feats.parquet"))
    sp500 = pd.read_parquet(artifact_file("sp500.parquet"))

    daily = (market_feats[["date", "aspi_close", "trading_volume", "log_return"]]
             .sort_values("date").reset_index(drop=True))
    daily["sp500_log_return"] = align_market_factor(
        daily["date"], sp500["date"], sp500["sp500_log_return"])
    volume = daily["trading_volume"].where(daily["trading_volume"] > 0)
    daily["log_volume"] = np.log(volume)
    return daily


def main() -> None:
    daily = load_daily()
    dataset = pd.read_parquet(artifact_file("dataset.parquet"))
    event_dates = pd.to_datetime(dataset["event_date"])
    print(f"{len(event_dates)} events | event window k = {EVENT_WINDOW[0]} to "
          f"{EVENT_WINDOW[1]} | cumulating from k = {CUMULATION_START}")

    tables = []
    for label, kwargs in (
        ("return", dict(model="market", value_col="log_return")),
        ("volume", dict(model="mean", value_col="log_volume")),
    ):
        panel = build_abnormal_panel(daily, event_dates, **kwargs)
        table = event_study_table(panel, start=CUMULATION_START, label=label)
        tables.append(table)
        figures.plot_event_time_caar(table, label)

        settled = table[table.event_time == 5].iloc[0]
        print(f"\n{label}: {panel.abnormal.shape[0]} events with a usable estimation window"
              f" | mean cross correlation {settled.mean_cross_correlation:+.4f}")
        print(f"  CAAR(1,5)  = {settled.caar:+.4f} "
              f"[{settled.ci_low:+.4f}, {settled.ci_high:+.4f}]")
        print(f"  t cross    = {settled.t_cross_sectional:+.3f}  p = {settled.p_cross_sectional:.4f}")
        print(f"  t BMP      = {settled.t_bmp:+.3f}  p = {settled.p_bmp:.4f}")
        print(f"  t Corrado  = {settled.t_corrado:+.3f}  p = {settled.p_corrado:.4f}")
        print(f"  t KP       = {settled.t_kolari_pynnonen:+.3f}  "
              f"p = {settled.p_kolari_pynnonen:.4f}")

    combined = pd.concat(tables, ignore_index=True)
    save_frame(combined, "event_study_caar",
               "Cumulative average abnormal return and volume by event-time session, "
               "with the cross-sectional, BMP, Corrado and Kolari-Pynnonen statistics.")
    figures.plot_event_time_panel(combined)
    print(f"\nwrote event_study_caar.parquet ({len(combined)} rows)")


if __name__ == "__main__":
    main()
