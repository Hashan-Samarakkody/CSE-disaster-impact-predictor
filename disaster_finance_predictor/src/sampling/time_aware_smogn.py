"""Time-aware SMOGN-style oversampling for regression events.

Real SmoteR-style neighbor-pair interpolation, not Gaussian noise on a single
point: for each minority row, find its nearest same-minority-class neighbor
in feature space, restricted to a bounded temporal window (real disasters
decades apart reflect different market/liquidity regimes and shouldn't be
interpolated together), then synthesize a new row on the line between the
two real points -- X and all target columns together, so a synthetic
long-recovery event doesn't get a random unrelated Y1/Y2.

The real `nickkunz/smogn` package was considered instead of this rewrite, but
its API (`smogn.smoter(data, y=<one column>)`) is single-response-variable
only and doesn't fit this project's joint 3-target setup without breaking
that joint consistency -- this implementation keeps the "time-aware" name
and intent while actually interpolating between real neighbors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def time_aware_smogn(
    X: pd.DataFrame,
    y: pd.DataFrame,
    minority_mask: pd.Series,
    event_dates: pd.Series | None = None,
    max_year_gap: float = 5.0,
    noise_scale: float = 0.01,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic minority samples via neighbor-pair interpolation.

    `event_dates` (aligned to X/y's index) enables the temporal-window
    constraint. If omitted, falls back to feature-space nearest-neighbor only
    (no temporal restriction) -- documented, not silent.
    """
    rng = np.random.default_rng(random_state)
    X_syn = X.copy()
    y_syn = y.copy()

    minority_idx = X.index[minority_mask]
    if len(minority_idx) < 2:
        return X_syn, y_syn

    # Standardize for feature-space distance so no single large-scale column
    # (e.g. financial_damage) dominates neighbor selection.
    X_min = X.loc[minority_idx]
    col_std = X_min.std().replace(0, 1.0)
    X_min_scaled = (X_min - X_min.mean()) / col_std

    synthetic_rows_X = []
    synthetic_rows_y = []

    for idx in minority_idx:
        candidates = [j for j in minority_idx if j != idx]
        if event_dates is not None:
            gap_years = (event_dates.loc[candidates] - event_dates.loc[idx]).abs().dt.days / 365.25
            candidates = [j for j, gap in zip(candidates, gap_years) if gap <= max_year_gap]

        if not candidates:
            # No neighbor inside the temporal window -- fall back to noise
            # on this single point rather than skip it silently.
            row_x = X.loc[idx]
            row_y = y.loc[idx]
            synthetic_rows_X.append(row_x + rng.normal(0.0, noise_scale, size=len(row_x)))
            synthetic_rows_y.append(row_y + rng.normal(0.0, noise_scale, size=len(row_y)))
            continue

        dists = ((X_min_scaled.loc[candidates] - X_min_scaled.loc[idx]) ** 2).sum(axis=1)
        neighbor_idx = dists.idxmin()

        r = rng.uniform(0.0, 1.0)
        x_a, x_b = X.loc[idx], X.loc[neighbor_idx]
        y_a, y_b = y.loc[idx], y.loc[neighbor_idx]
        synthetic_rows_X.append(x_a + r * (x_b - x_a))
        synthetic_rows_y.append(y_a + r * (y_b - y_a))

    syn_x_df = pd.DataFrame(synthetic_rows_X, columns=X.columns)
    syn_y_df = pd.DataFrame(synthetic_rows_y, columns=y.columns)

    X_syn = pd.concat([X_syn, syn_x_df], ignore_index=True)
    y_syn = pd.concat([y_syn, syn_y_df], ignore_index=True)

    return X_syn, y_syn
