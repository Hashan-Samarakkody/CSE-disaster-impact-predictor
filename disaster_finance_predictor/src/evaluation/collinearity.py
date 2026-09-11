"""Collinearity diagnostics, and the pre-declared rule for acting on them.

Audit finding P2-3 says collinearity is "severe and largely unaddressed": 31 features on
30-row folds is p > n before selection, and several columns are exact monotone
transforms of each other. This module measures it and applies a rule that is fixed in
advance, so the feature set is never chosen by looking at held-out scores.

Why Spearman is the default. `financial_damage` is zero for 47 of 64 events with a long
right tail on the remaining 17, so a Pearson correlation on that column is driven by two
or three observations. Spearman is also invariant to monotone transforms, which means
`financial_damage` against `log_financial_damage` reads exactly 1.00 -- not a defect but
the cleanest possible statement that the two columns carry one variable. Pearson would
report roughly 0.7 and understate the redundancy. Pearson is still available, and is the
right matrix for VIF and for reasoning about the linear solve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The six thematic blocks of FEATURE_COLS. Declared by hand because the grouping is a
# statement about what the variables mean, not something to infer from a name prefix.
FEATURE_GROUPS = {
    "damage/severity": ["financial_damage", "population_affected", "log_financial_damage",
                        "log_population_affected", "damage_to_gdp", "log_damage_x_flood"],
    "disaster type": ["disaster_Drought", "disaster_Flood", "disaster_Other", "disaster_Storm"],
    "recency": ["days_since_last_disaster", "disasters_trailing_365d"],
    "returns/momentum": ["log_return", "lag_return_t-1", "lag_return_t-2", "lag_return_t-3",
                         "lag_return_t-5", "price_to_sma_5", "price_to_ema_5",
                         "price_to_sma_10", "price_to_ema_10", "price_to_sma_20",
                         "price_to_ema_20"],
    "volatility": ["rolling_std_5", "rolling_std_10", "rolling_std_20", "rolling_std_30",
                   "squared_return"],
    "volume": ["vol_ratio_1_30", "vol_ratio_5_30", "vol_ratio_10_30", "vol_cv_30",
               "log_vol_change_1"],
    "macro/global": ["gdp_growth_pct", "inflation_cpi_pct", "sp500_log_return",
                     "fx_logret_1", "fx_logret_5", "fx_vol_30"],
    # External blocks A, B and D (docs/EXTERNAL_DATA_PRE_DECLARATION.md). Hazard is kept
    # separate from damage/severity deliberately: it is instrument-measured intensity,
    # whereas the damage columns are post-hoc assessments that are absent for most
    # events, and grouping them would hide exactly that distinction in the heatmap.
    "hazard (measured)": ["hz_precip_max3d", "hz_precip_mean3d", "hz_precip_spread3d",
                          "hz_districts_wet", "hz_wind_max3d", "hz_precip_anom"],
    "physical severity": ["di_districts_hit", "di_affected_log", "di_houses_destroyed_log",
                          "di_houses_damaged_log", "di_deaths_log", "di_records",
                          "di_available"],
    "confounder": ["days_to_election", "election_within_5d"],
}

# How many derivation steps separate a column from a raw measurement. The drop rule
# below keeps the most primitive member of a redundant group, so this ordering is what
# makes the rule deterministic rather than a judgement call at the time of use.
_DERIVATION_DEPTH = {
    "financial_damage": 0, "population_affected": 0,
    "log_financial_damage": 1, "log_population_affected": 1,
    "damage_to_gdp": 1, "log_damage_x_flood": 2,
    "log_return": 0, "squared_return": 1,
    # External blocks. The raw measurements are the per-district accumulations and the
    # DesInventar counts; everything else is one or two transforms off them, so the
    # "keep the least-derived member" rule stays deterministic across the new columns.
    "hz_precip_max3d": 1, "hz_precip_mean3d": 1, "hz_wind_max3d": 1,
    "hz_precip_spread3d": 2, "hz_districts_wet": 2, "hz_precip_anom": 2,
    "di_records": 0, "di_available": 0, "di_districts_hit": 1,
    "di_affected_log": 1, "di_deaths_log": 1,
    "di_houses_destroyed_log": 1, "di_houses_damaged_log": 1,
    "fx_logret_1": 1, "fx_logret_5": 1, "fx_vol_30": 2,
    "days_to_election": 0, "election_within_5d": 1,
}


def assign_group(name: str) -> str:
    for group, members in FEATURE_GROUPS.items():
        if name in members:
            return group
    return "other"


def derivation_depth(name: str) -> int:
    if name in _DERIVATION_DEPTH:
        return _DERIVATION_DEPTH[name]
    # Lags and rolling windows are one step off the raw series; ratios of two rolling
    # quantities are two.
    if name.startswith("price_to_"):
        return 2
    if name.startswith(("lag_return", "rolling_std")):
        return 1
    return 0


def correlation_matrix(X: pd.DataFrame, method: str = "spearman") -> pd.DataFrame:
    return X.corr(method=method)


def critical_r(n: int, alpha: float = 0.05) -> float:
    """Smallest |r| distinguishable from zero at this n, via the Fisher z transform.

    At n=64 this is about 0.246. Correlations below it are not evidence of anything and
    should be masked in any heatmap, or a reader will point at a 0.3 cell and ask what
    it means.
    """
    from scipy import stats
    if n < 4:
        return float("nan")
    z = stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z / np.sqrt(n - 3)))


def cluster_order(R: pd.DataFrame):
    """Hierarchical leaf order on distance 1 - |rho|, with optimal leaf ordering.

    Optimal leaf ordering matters at this size: plain `leaves_list` leaves arbitrary
    within-cluster flips that make the blocks look noisier than they are.
    """
    from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
    from scipy.spatial.distance import squareform

    d = 1.0 - R.abs().to_numpy()
    np.fill_diagonal(d, 0.0)
    d = (d + d.T) / 2.0  # enforce exact symmetry; float drift upsets squareform
    condensed = squareform(d, checks=False)
    link = linkage(condensed, method="average")
    link = optimal_leaf_ordering(link, condensed)
    return list(leaves_list(link)), link


def top_correlated_pairs(X: pd.DataFrame, method="spearman", top_n=15) -> pd.DataFrame:
    R = correlation_matrix(X, method=method)
    cols = list(R.columns)
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rows.append({"feature_a": cols[i], "feature_b": cols[j],
                         "rho": float(R.iloc[i, j]), "abs_rho": abs(float(R.iloc[i, j])),
                         "group_a": assign_group(cols[i]), "group_b": assign_group(cols[j])})
    return (pd.DataFrame(rows).sort_values("abs_rho", ascending=False)
            .head(top_n).reset_index(drop=True))


def compute_vif(X: pd.DataFrame, drop_reference: str = "disaster_Other") -> pd.DataFrame:
    """Variance inflation factors.

    One disaster one-hot must be dropped: the four indicators sum to 1, so together with
    an intercept the design is exactly singular and every VIF in the block comes back as
    inf or NaN. Each call is guarded so a singular column records `inf` rather than
    aborting the whole table.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant

    cols = [c for c in X.columns if c != drop_reference]
    # Zero-variance columns cannot have a VIF and would poison the whole design matrix.
    keep = [c for c in cols if X[c].std(ddof=0) > 0]
    dropped = sorted(set(cols) - set(keep))

    design = add_constant(X[keep].astype(float), has_constant="add")
    rows = []
    for i, name in enumerate(design.columns):
        if name == "const":
            continue
        try:
            v = float(variance_inflation_factor(design.to_numpy(), i))
        except Exception:
            v = float("inf")
        rows.append({"feature": name, "vif": v, "group": assign_group(name)})
    for name in dropped:
        rows.append({"feature": name, "vif": float("nan"), "group": assign_group(name)})
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def condition_indices(X: pd.DataFrame) -> pd.Series:
    """sqrt(lambda_max / lambda_i) of the standardised design.

    Never singular-fragile the way individual VIFs are, and the >30 rule of thumb gives
    one headline number instead of 31 separate ones.
    """
    Z = X.astype(float).to_numpy()
    Z = (Z - Z.mean(axis=0)) / np.where(Z.std(axis=0) == 0, 1.0, Z.std(axis=0))
    eig = np.linalg.svd(Z, compute_uv=False) ** 2
    eig = np.where(eig <= 0, np.finfo(float).tiny, eig)
    return pd.Series(np.sqrt(eig.max() / eig), name="condition_index")


# --------------------------------------------------------------- the drop rule

REDUNDANCY_THRESHOLD = 0.95


def redundant_drop_set(X: pd.DataFrame, threshold: float = REDUNDANCY_THRESHOLD,
                       method: str = "spearman"):
    """Apply the pre-declared redundancy rule and report exactly what it removes.

    **The rule, fixed before any model was scored under it:** within any group of
    features whose pairwise |rho| exceeds `threshold`, keep the single most primitive
    member (lowest derivation depth; ties broken alphabetically for determinism) and
    drop the rest.

    It never consults a target, a fold or a score, so applying it cannot be selection on
    the test set. Whatever it does to the metrics is reported either way.

    Returns (keep, drop, detail_frame).
    """
    R = correlation_matrix(X, method=method).abs()
    cols = list(R.columns)

    # Connected components of the "correlated above threshold" graph.
    adjacency = {c: set() for c in cols}
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if R.loc[a, b] >= threshold:
                adjacency[a].add(b)
                adjacency[b].add(a)

    seen, drop, detail = set(), [], []
    for col in cols:
        if col in seen or not adjacency[col]:
            seen.add(col)
            continue
        stack, component = [col], set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency[node] - component)
        seen |= component

        survivor = sorted(component, key=lambda c: (derivation_depth(c), c))[0]
        for member in sorted(component - {survivor}):
            drop.append(member)
            detail.append({"dropped": member, "kept": survivor,
                           "rho_with_kept": float(R.loc[member, survivor]),
                           "group": assign_group(member)})

    keep = [c for c in cols if c not in set(drop)]
    return keep, sorted(drop), pd.DataFrame(detail)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 64
    base = rng.normal(size=n)
    frame = pd.DataFrame({
        "financial_damage": np.abs(base) * 1e6,
        "log_financial_damage": np.log1p(np.abs(base) * 1e6),   # monotone => rho == 1
        "log_return": rng.normal(size=n),
        "disaster_Flood": rng.integers(0, 2, n).astype(float),
        "disaster_Other": rng.integers(0, 2, n).astype(float),
    })

    R = correlation_matrix(frame)
    assert np.allclose(R.to_numpy(), R.to_numpy().T)
    assert np.allclose(np.diag(R.to_numpy()), 1.0)
    assert R.loc["financial_damage", "log_financial_damage"] == 1.0, "monotone pair must be 1.0"
    assert abs(critical_r(64) - 0.246) < 0.01, critical_r(64)

    keep, drop, detail = redundant_drop_set(frame)
    # The log transform is one derivation step deeper, so the raw column survives.
    assert drop == ["log_financial_damage"], drop
    assert "financial_damage" in keep

    vif = compute_vif(frame)
    assert len(vif) == len(frame.columns) - 1  # the reference dummy is excluded
    assert vif["vif"].notna().any()

    order, _ = cluster_order(R)
    assert sorted(order) == list(range(len(frame.columns)))
    assert condition_indices(frame).max() >= 1.0

    print("collinearity.py self-check passed")
