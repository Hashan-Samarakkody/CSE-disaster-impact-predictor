"""Time-aware SMOGN-style oversampling for regression events.

Real SmoteR-style neighbour-pair interpolation, not Gaussian noise on a single
point: for each minority row, find its nearest same-minority-class neighbour
in feature space, restricted to a bounded temporal window (real disasters
decades apart reflect different market/liquidity regimes and shouldn't be
interpolated together), then synthesize a new row on the line between the
two real points -- X and all target columns together, so a synthetic
long-recovery event doesn't get a random unrelated Y1/Y2.

Three properties beyond plain interpolation, each fixing a way the earlier
version produced rows that could not exist in the real data:

1. **Categorical integrity.** One-hot disaster-type columns are copied
   verbatim from a parent, never blended -- the earlier version produced rows
   that were 60% Flood and 40% Storm.
2. **Derived-feature consistency.** Columns that are FUNCTIONS of other
   columns (the log-damage terms, the damage/GDP ratio, the squared return,
   the damage x flood interaction) are recomputed from the interpolated
   parents rather than interpolated themselves. Interpolating a child
   independently of its parent breaks the identity between them on every
   synthetic row, teaching the model a relationship that holds nowhere in
   the real data.
3. **Same-type pairing.** A flood is interpolated only with another flood.
   A flood interpolated with a drought is not a market-regime approximation,
   it is a physical impossibility.

The real `nickkunz/smogn` package was considered instead of this rewrite, but
its API (`smogn.smoter(data, y=<one column>)`) is single-response-variable
only and doesn't fit this project's joint 3-target setup without breaking
that joint consistency. Canonical SMOGN's majority under-sampling is
deliberately DISABLED (a supported configuration of the reference
implementation, not a deviation from it): at ~30 training rows per fold,
discarding real observations to balance a class ratio is the most expensive
action available.

Interpolation happens in RAW units (e.g. US dollars of damage), which is what
EM-DAT measures and what the damage/GDP ratio needs as a numerator; the log
terms are then derived from the interpolated raw value. Interpolating in log
space instead would produce a geometric mean, a different and also defensible
choice.

Known residual limitation: `rolling_std_*` columns are convex functionals of
the price path, so a convex combination of two rows over-states volatility
relative to the same functional applied to the combined path. This biases
synthetic volatility slightly HIGH and cannot be corrected without the
underlying price series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Columns that are deterministic functions of OTHER feature columns. These are
# recomputed after interpolation instead of being interpolated. Definitions mirror
# exactly where the real feature table builds them:
#   feature_eng.py  log_financial_damage    = log1p(clip(financial_damage, 0))
#   feature_eng.py  log_population_affected = log1p(clip(population_affected, 0))
#   feature_eng.py  squared_return          = log_return ** 2
#   notebook §7     damage_to_gdp           = financial_damage / gdp_current_usd
#   notebook §7     log_damage_x_flood      = log_financial_damage * disaster_Flood
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

    `event_dates` (aligned to X/y's index) enables the temporal-window
    constraint. If omitted, falls back to feature-space nearest-neighbour only
    (no temporal restriction) -- documented, not silent.

    `type_cols` names a mutually-exclusive one-hot block (the disaster types).
    When given, parents must share the same active type, and the block is copied
    verbatim rather than blended. When empty, behaviour is unconstrained as
    before. Not auto-detected: a prefix or looks-binary heuristic would silently
    misclassify some future 0/1 feature.

    `gdp_current_usd` is a per-row carrier value interpolated alongside the
    features so `damage_to_gdp` can be rebuilt as a genuine ratio of two
    interpolated real quantities. It is deliberately not a model feature and
    never enters X.

    `n_synthetic_per_row` is the standard SMOTE oversampling ratio: that many
    interpolation ratios drawn per minority row along the same real neighbour
    pair. Still real-neighbour interpolation, just a denser sample of the
    segment between two real events.

    `max_synthetic_share` is a pre-registered ceiling on the synthetic fraction
    of the returned table. Exceeding it raises rather than truncating: over-
    augmentation must fail loudly, not degrade quality quietly.
    """
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

    # Distance columns exclude the one-hot block (constant within a candidate set once
    # the type constraint applies, and a near-zero std would blow up the scaling) and
    # the derived columns (financial_damage plus its three children would otherwise
    # make damage similarity dominate neighbour choice four times over).
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
            # Singleton type in this fold (Earthquake and Mass movement have one event
            # each in the full in-scope set). Nothing physically comparable exists to
            # interpolate against, so no synthetic row is produced. Deliberately not
            # falling back to noise on the single point: at ~30 training rows that is a
            # near-duplicate which silently doubles one event's weight, and scale-blind
            # additive noise on a target like Y1 (sigma ~0.014) can flip its sign while
            # leaving Y3 at 90 -- a combination that cannot occur, since Y3 = 0 exactly
            # when Y1 >= 0.
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
                # Copy the categorical block from the nearer parent -- the k=1 analogue
                # of SMOTE-NC's majority vote over neighbours. Under the same-type rule
                # both parents share the block, so this is exact either way; it also
                # guards against float drift from the arithmetic above.
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
