"""Time-aware SMOGN-style oversampling for regression events.

Interpolates each minority row with its nearest same-type neighbour inside a bounded
time window, in raw units, blending X and all targets together. Categorical one-hots are
copied from a parent and derived columns are recomputed, so no synthetic row breaks an
identity that holds in the real data. Rationale and known limitations:
`architecture/feature_engineering.md`."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Deterministic functions of OTHER feature columns: recomputed after interpolation
# rather than interpolated, so the identity that holds in the real table also holds on
# every synthetic row. Definitions mirror feature_eng.py and notebook 02.
DERIVED_COLS = (
    "log_financial_damage",
    "log_population_affected",
    "squared_return",
    "damage_to_gdp",
    "log_damage_x_flood",
)


def _recompute_derived(df: pd.DataFrame, gdp: np.ndarray | None) -> None:
    """Recompute derived columns in place. Order matters: the log-damage term is
    rebuilt before the interaction term that multiplies it."""
    cols = set(df.columns)
    if {"log_financial_damage", "financial_damage"} <= cols:
        df["log_financial_damage"] = np.log1p(df["financial_damage"].clip(lower=0))
    if {"log_population_affected", "population_affected"} <= cols:
        df["log_population_affected"] = np.log1p(df["population_affected"].clip(lower=0))
    if {"squared_return", "log_return"} <= cols:
        df["squared_return"] = df["log_return"] ** 2
    if {"damage_to_gdp", "financial_damage"} <= cols and gdp is not None:
        # GDP is NaN for some years; NaN > 0 is False, so those rows get 0.0 -- the
        # same value the real feature table gives them via its own fillna.
        df["damage_to_gdp"] = np.where(gdp > 0, df["financial_damage"].to_numpy() / gdp, 0.0)
    if {"log_damage_x_flood", "log_financial_damage", "disaster_Flood"} <= cols:
        df["log_damage_x_flood"] = df["log_financial_damage"] * df["disaster_Flood"]


def time_aware_smogn(
    X: pd.DataFrame,
    y: pd.DataFrame,
    minority_mask: pd.Series,
    event_dates: pd.Series | None = None,
    max_year_gap: float = 5.0,
    random_state: int = 42,
    n_synthetic_per_row: int = 1,
    type_cols: list[str] | None = None,
    gdp_current_usd: pd.Series | None = None,
    max_synthetic_share: float = 0.25,
    return_report: bool = False,
):
    """Generate synthetic minority samples via neighbour-pair interpolation.

    `event_dates` enables the temporal window; `type_cols` names the one-hot block that
    must match between parents and is copied, not blended; `gdp_current_usd` is carried
    (never a feature) so `damage_to_gdp` can be rebuilt as a real ratio.
    `max_synthetic_share` is a pre-registered ceiling and RAISES rather than truncating."""
    rng = np.random.default_rng(random_state)
    type_cols = [c for c in (type_cols or []) if c in X.columns]
    minority_idx = list(X.index[minority_mask])

    report = {
        "n_minority": len(minority_idx),
        "n_synthetic": 0,
        "pairs_in_window": 0,
        "pairs_window_relaxed": 0,
        "skipped_no_partner": 0,
    }

    def _out(xs, ys):
        return (xs, ys, report) if return_report else (xs, ys)

    if len(minority_idx) < 2:
        return _out(X.copy(), y.copy())

    # Type key = name of the active one-hot column, used both to constrain pairing
    # and to copy the block verbatim.
    if type_cols:
        type_key = X.loc[minority_idx, type_cols].idxmax(axis=1)
    else:
        type_key = pd.Series("", index=minority_idx)

    # Distance columns exclude the one-hot block (constant once the type constraint
    # applies, and a near-zero std blows up the scaling) and the derived columns, which
    # would otherwise let damage similarity dominate neighbour choice four times over.
    dist_cols = [c for c in X.columns if c not in type_cols and c not in DERIVED_COLS]
    X_min = X.loc[minority_idx, dist_cols].astype(float)
    col_std = X_min.std().replace(0.0, 1.0).clip(lower=1e-12)
    X_min_scaled = (X_min - X_min.mean()) / col_std

    gaps = None
    if event_dates is not None:
        days = (
            pd.to_datetime(event_dates.loc[minority_idx])
            .to_numpy()
            .astype("datetime64[D]")
            .astype(np.int64)
        )
        gaps = pd.DataFrame(
            np.abs(days[:, None] - days[None, :]) / 365.25,
            index=minority_idx,
            columns=minority_idx,
        )

    rows_X: list[pd.Series] = []
    rows_y: list[pd.Series] = []
    rows_gdp: list[float] = []

    for idx in minority_idx:
        same_type = [j for j in minority_idx if j != idx and type_key[j] == type_key[idx]]
        if not same_type:
            # Singleton type in this fold: nothing physically comparable to interpolate against, so
            # no synthetic row. Deliberately no noise fallback -- at ~30 training rows that is a
            # near-duplicate, and scale-blind noise can flip Y1's sign while leaving Y3 at 90.
            report["skipped_no_partner"] += 1
            continue

        in_window = same_type if gaps is None else [j for j in same_type if gaps.at[idx, j] <= max_year_gap]
        # Relax the temporal window before relaxing the type: +/-5 years is a soft
        # market-regime argument, shared disaster type is a hard physical one.
        candidates = in_window or same_type
        report["pairs_in_window" if in_window else "pairs_window_relaxed"] += 1

        dists = ((X_min_scaled.loc[candidates] - X_min_scaled.loc[idx]) ** 2).sum(axis=1)
        neighbour_idx = dists.idxmin()

        x_a, x_b = X.loc[idx], X.loc[neighbour_idx]
        y_a, y_b = y.loc[idx], y.loc[neighbour_idx]
        for _ in range(n_synthetic_per_row):
            r = float(rng.uniform(0.0, 1.0))
            x_new = x_a + r * (x_b - x_a)
            if type_cols:
                # Copy the categorical block from the nearer parent, the k=1 analogue of SMOTE-NC's
                # majority vote. Under the same-type rule both parents share it, so this is exact
                # either way and also guards against float drift.
                donor = x_a if r < 0.5 else x_b
                x_new.loc[type_cols] = donor.loc[type_cols].to_numpy()
            rows_X.append(x_new)
            rows_y.append(y_a + r * (y_b - y_a))
            if gdp_current_usd is not None:
                g_a, g_b = gdp_current_usd.loc[idx], gdp_current_usd.loc[neighbour_idx]
                rows_gdp.append(g_a + r * (g_b - g_a))

    if not rows_X:
        return _out(X.copy(), y.copy())

    syn_X = pd.DataFrame(rows_X, columns=X.columns).reset_index(drop=True)
    syn_y = pd.DataFrame(rows_y, columns=y.columns).reset_index(drop=True)
    _recompute_derived(syn_X, np.asarray(rows_gdp, dtype=float) if rows_gdp else None)

    if max_synthetic_share < 1.0:  # share of 1.0 disables the cap (used by the unit test)
        cap = int(np.floor(max_synthetic_share / (1.0 - max_synthetic_share) * len(X)))
        if len(syn_X) > cap:
            raise ValueError(
                f"{len(syn_X)} synthetic rows on {len(X)} real rows exceeds the pre-registered "
                f"{max_synthetic_share:.0%} synthetic-share cap (max {cap}). Lower "
                f"n_synthetic_per_row or narrow minority_mask."
            )

    report["n_synthetic"] = len(syn_X)
    # Note: ignore_index discards the original labels, so the returned frames can no
    # longer be re-joined against event_dates by index.
    return _out(
        pd.concat([X, syn_X], ignore_index=True),
        pd.concat([y, syn_y], ignore_index=True),
    )
