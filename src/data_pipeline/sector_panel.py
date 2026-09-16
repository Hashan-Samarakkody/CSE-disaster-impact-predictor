"""Sector-level event panel.

One row per (event, sector) so the index-level aggregation can be tested directly. The
20 sectors of one event move together, so folds are cut between events
(`grouped_walk_forward`) and intervals resample events, not rows (`event_block_bootstrap`)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .feature_eng import FeatureEngineer, FeatureEngineeringConfig


def build_sector_panel(sector_long: pd.DataFrame, events: pd.DataFrame,
                       min_history: int = 60) -> pd.DataFrame:
    """One row per (event, sector), with that sector's own pre-event market features.

    Each sector's close series is run through the existing `FeatureEngineer` by presenting
    it as the price column, so targets and features are computed by exactly the same code
    that produced the index-level table -- no parallel implementation to drift out of sync.

    `sector_long` needs columns date/sector/close; `events` needs event_date plus whatever
    disaster covariates should travel with each row.
    """
    cfg = FeatureEngineeringConfig(price_col="close", volume_col="__absent__")
    fe = FeatureEngineer(cfg)

    events = events.copy()
    events["event_date"] = pd.to_datetime(events["event_date"])
    disaster_cols = [c for c in events.columns if c != "event_date"]

    rows = []
    for sector, block in sector_long.groupby("sector", sort=True):
        block = block[["date", "close"]].sort_values("date").reset_index(drop=True)
        if len(block) < min_history:
            continue

        feats = fe.engineer_market_features(block)
        targets = fe.build_targets(block, events[["event_date"]])
        if targets.empty:
            continue

        # Same as the index-level table: market state is read as of the day BEFORE the
        # event, so no feature can contain the event-day move it is used to predict.
        snap = events[["event_date"]].copy()
        snap["asof_date"] = snap["event_date"] - pd.Timedelta(days=1)
        snap = pd.merge_asof(snap.sort_values("asof_date"), feats.sort_values("date"),
                             left_on="asof_date", right_on="date", direction="backward")

        merged = snap.merge(targets, on="event_date", how="inner")
        merged["sector"] = sector
        rows.append(merged)

    if not rows:
        return pd.DataFrame()

    panel = pd.concat(rows, ignore_index=True)
    panel = panel.merge(events[["event_date"] + disaster_cols], on="event_date", how="left")
    panel = panel.rename(columns={"Y1_ASPI_5D_Forward_LogReturn_Pct": "Y1_sector_log_return",
                                  "Y3_recovery_days": "Y3_sector_recovery_days",
                                  "Y1_EventWindow_0_5_LogReturn_Pct": "Y1_sector_car_5",
                                  "Y1_EventWindow_0_10_LogReturn_Pct": "Y1_sector_car_10"})
    panel = panel.drop(columns=[c for c in ("Y2_abnormal_volume",) if c in panel.columns])

    # event_id groups the rows that must never be split across a fold boundary.
    codes = {d: i for i, d in enumerate(sorted(panel["event_date"].unique()))}
    panel["event_id"] = panel["event_date"].map(codes)
    return panel.sort_values(["event_id", "sector"]).reset_index(drop=True)


def grouped_walk_forward(panel: pd.DataFrame, train_events: int, test_events: int,
                         step: int):
    """Chronological folds whose boundaries fall between events, never inside one.

    Splitting on rows would put some of an event's sectors in train and the rest in test.
    Since sectors move together on the day of a shock, that would leak the answer across
    the boundary and inflate every score in the panel.
    """
    ids = np.sort(panel["event_id"].unique())
    start = 0
    while start + train_events + test_events <= len(ids):
        tr_ids = ids[start:start + train_events]
        te_ids = ids[start + train_events:start + train_events + test_events]
        yield (np.where(panel["event_id"].isin(tr_ids))[0],
               np.where(panel["event_id"].isin(te_ids))[0])
        start += step


def event_block_bootstrap(event_ids, y_true, y_pred_a, y_pred_b, n_boot: int = 5000,
                          random_state: int = 42):
    """Paired bootstrap that resamples EVENTS, carrying all of an event's sectors together.

    Resampling rows independently would treat 20 correlated sector observations as 20
    independent ones and shrink the interval by roughly sqrt(20) -- manufacturing
    significance out of the clustering. Returns the RMSE delta (positive favours model a)
    with its 95% interval.
    """
    event_ids = np.asarray(event_ids)
    yt, pa, pb = (np.asarray(v, dtype=float) for v in (y_true, y_pred_a, y_pred_b))
    uniq = np.unique(event_ids)
    index_by_event = {e: np.where(event_ids == e)[0] for e in uniq}

    def rmse(idx, p):
        return float(np.sqrt(np.mean((p[idx] - yt[idx]) ** 2)))

    all_idx = np.arange(len(yt))
    point = rmse(all_idx, pb) - rmse(all_idx, pa)

    rng = np.random.default_rng(random_state)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index_by_event[e] for e in picked])
        draws[b] = rmse(idx, pb) - rmse(idx, pa)

    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"delta_rmse": point, "ci_low": float(lo), "ci_high": float(hi),
            "n_events": len(uniq), "n_rows": len(yt), "significant": bool(lo > 0)}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2010-01-01", periods=400)
    sectors = ["Alpha", "Beta", "Gamma"]
    long = pd.concat([
        pd.DataFrame({"date": dates, "sector": s,
                      "close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))})
        for s in sectors])
    events = pd.DataFrame({"event_date": dates[[80, 150, 220, 300]],
                           "disaster_Flood": [1.0, 0.0, 1.0, 1.0]})

    panel = build_sector_panel(long, events)
    assert len(panel) == len(sectors) * len(events), len(panel)
    assert {"Y1_sector_log_return", "Y3_sector_recovery_days", "sector", "event_id"} <= set(panel.columns)
    assert "Y2_abnormal_volume" not in panel.columns

    folds = list(grouped_walk_forward(panel, train_events=2, test_events=1, step=1))
    assert folds, "expected at least one fold"
    for tr, te in folds:
        # The defining property: no event may appear on both sides of a boundary.
        assert not (set(panel["event_id"].iloc[tr]) & set(panel["event_id"].iloc[te]))

    ids = panel["event_id"].to_numpy()
    yt = panel["Y1_sector_log_return"].to_numpy()
    good, bad = yt + rng.normal(0, 1e-4, len(yt)), yt + rng.normal(0, 0.05, len(yt))
    res = event_block_bootstrap(ids, yt, good, bad, n_boot=500)
    assert res["delta_rmse"] > 0 and res["n_events"] == 4

    print("sector_panel.py self-check passed")
