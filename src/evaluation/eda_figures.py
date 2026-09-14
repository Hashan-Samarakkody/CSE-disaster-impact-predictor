"""Rigorous exploratory data analysis figures.

`figures.py` covers targets, collinearity and model evaluation. This module covers what a
thesis EDA chapter is actually examined on: where the data came from, what is missing,
what is extreme, how each variable is distributed, and how the candidate predictors relate
to the targets.

Three conventions carry over from `figures.py`, and one is added:

- every figure states its sample size on its face (`stamp`);
- nothing is encoded by hue alone;
- honest reference lines are drawn in rather than left implicit.
- **added here:** a missing value and a zero-filled value are drawn DIFFERENTLY. The
  study's central data problem is that `financial_damage` is zero-filled for most events,
  which is invisible in a conventional missingness plot because the column has no NaNs at
  all. A plot that showed it as complete would misrepresent the dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .figures import (CATEGORICAL, MARKERS, NEUTRAL, PALETTE, SEQUENTIAL, WIDTH_FULL,
                      H_MED, H_TALL, save_figure, stamp)


def _pretty(name: str) -> str:
    return (str(name).replace("_", " ").replace("log ", "log·")
            .replace("disaster ", "").strip())


# --------------------------------------------------------------- missingness


def plot_missingness_matrix(dataset, cols=None, damage_source_col="damage_source",
                            name="eda_01_missingness_matrix"):
    """Event x variable map of what is observed, what is absent, and what is zero-filled.

    The three states are drawn distinctly on purpose. A conventional missingness plot
    reports `financial_damage` as 100% complete, because the loader already replaced its
    absent values with 0.0 -- and that zero then reads to a model as "no damage" rather
    than "unknown". Collapsing those two states is the single most consequential thing a
    missingness figure for this dataset could get wrong.
    """
    import matplotlib.pyplot as plt

    df = dataset.copy()
    if cols is None:
        cols = [c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c]) and not c.startswith("sec_")]
    cols = [c for c in cols if c in df.columns]

    # 0 = observed, 1 = zero-filled (present but not measured), 2 = missing
    state = pd.DataFrame(0, index=df.index, columns=cols, dtype=float)
    state[df[cols].isna()] = 2.0

    zero_filled = {}
    if damage_source_col in df.columns:
        imputed = df[damage_source_col].astype(str).str.contains("missing", case=False,
                                                                 na=False)
        for c in cols:
            if "damage" in c:
                zero_filled[c] = imputed
    # Availability flags name their own partner column explicitly.
    for flag, partner in (("di_available", "di_"), ("mag_area_available", "mag_area_km2"),
                          ("mag_wind_available", "mag_wind_kph"),
                          ("deaths_available", "total_deaths"),
                          ("homeless_available", "no_homeless")):
        if flag not in df.columns:
            continue
        absent = df[flag] == 0
        for c in cols:
            if c == partner or (partner.endswith("_") and c.startswith(partner)
                                and not c.endswith("available")):
                zero_filled[c] = absent
    for c, mask in zero_filled.items():
        state.loc[mask.reindex(state.index, fill_value=False), c] = 1.0

    fig, ax = plt.subplots(figsize=(WIDTH_FULL, max(H_MED, 0.16 * len(cols))))
    cmap = matplotlib_listed(["#F2F2F2", PALETTE["orange"], PALETTE["vermil"]])
    ax.imshow(state.T.to_numpy(), aspect="auto", interpolation="nearest", cmap=cmap,
              vmin=0, vmax=2)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels([_pretty(c) for c in cols], fontsize=5.5)
    ax.set_xlabel("event (chronological)")
    ax.set_title("Data completeness by event and variable")
    ax.grid(False)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in
               ("#F2F2F2", PALETTE["orange"], PALETTE["vermil"])]
    ax.legend(handles, ["observed", "zero-filled (present, not measured)", "missing (NaN)"],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=6.5)

    n_zero = int(sum(int(m.sum()) for m in zero_filled.values()))
    stamp(fig, f"n = {len(df)} events x {len(cols)} variables · "
               f"{int(df[cols].isna().sum().sum())} NaN cells · {n_zero} zero-filled cells")
    save_figure(fig, name)
    return fig, state


def matplotlib_listed(colors):
    from matplotlib.colors import ListedColormap
    return ListedColormap(colors)


def plot_missingness_ranked(dataset, cols=None, top_n=25,
                            name="eda_02_missingness_ranked"):
    """Per-variable completeness, worst first, with the count printed on each bar."""
    import matplotlib.pyplot as plt

    df = dataset
    if cols is None:
        cols = [c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c]) and not c.startswith("sec_")]
    frac = df[cols].notna().mean().sort_values()
    frac = frac.head(top_n) if len(frac) > top_n else frac

    fig, ax = plt.subplots(figsize=(WIDTH_FULL, max(H_MED, 0.18 * len(frac))))
    colours = [PALETTE["vermil"] if v < 0.5 else
               PALETTE["orange"] if v < 1.0 else PALETTE["green"] for v in frac]
    ax.barh([_pretty(c) for c in frac.index], frac.to_numpy(), color=colours)
    ax.axvline(1.0, color=PALETTE["grey"], lw=0.8, ls="--")
    for y, (c, v) in enumerate(frac.items()):
        ax.text(min(v + 0.01, 0.99), y, f"{int(df[c].notna().sum())}/{len(df)}",
                va="center", fontsize=6)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("fraction of events with an observed value")
    ax.set_title("Completeness by variable (worst first)")
    ax.tick_params(axis="y", labelsize=6)
    stamp(fig, f"n = {len(df)} events")
    save_figure(fig, name)
    return fig, frac


# --------------------------------------------------------------- outliers


def outlier_table(series, iqr_k: float = 1.5, mad_z: float = 3.5):
    """Flag outliers two ways and return the fences.

    The IQR rule is the conventional one. The modified z-score (median absolute
    deviation) is reported beside it because IQR fences are themselves computed from
    quantiles that a handful of extreme events can move -- and this dataset has exactly
    that shape, with the 2004 tsunami and the 2005 flood sitting far from the rest.
    """
    s = pd.Series(series).astype(float).dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr

    med = s.median()
    mad = (s - med).abs().median()
    # 0.6745 makes the MAD a consistent estimator of sigma under normality.
    z = 0.6745 * (s - med) / mad if mad else pd.Series(0.0, index=s.index)

    return {"q1": float(q1), "q3": float(q3), "iqr_lo": float(lo), "iqr_hi": float(hi),
            "iqr_outliers": s[(s < lo) | (s > hi)],
            "mad_outliers": s[z.abs() > mad_z],
            "n": int(len(s))}


def plot_outlier_panel(dataset, cols, date_col="event_date", label_top=3,
                       name="eda_03_outliers"):
    """Box plot plus jittered points per variable, with the extreme events named.

    Naming the events matters more than counting them: an outlier that turns out to be
    the 2004 Indian Ocean tsunami is a real observation to be discussed, not a data error
    to be trimmed.
    """
    import matplotlib.pyplot as plt

    cols = [c for c in cols if c in dataset.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(WIDTH_FULL, H_MED))
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)
    summary = []

    for ax, c in zip(axes, cols):
        s = dataset[c].astype(float)
        info = outlier_table(s)
        ok = s.dropna()
        ax.boxplot(ok, widths=0.5, showfliers=False,
                   medianprops=dict(color=PALETTE["ink"], lw=1.2),
                   boxprops=dict(color=PALETTE["grey"]),
                   whiskerprops=dict(color=PALETTE["grey"]),
                   capprops=dict(color=PALETTE["grey"]))
        x = 1 + rng.uniform(-0.14, 0.14, len(ok))
        flagged = ok.index.isin(info["iqr_outliers"].index)
        ax.scatter(x[~flagged], ok[~flagged], s=9, color=PALETTE["blue"], alpha=0.55,
                   zorder=3)
        ax.scatter(x[flagged], ok[flagged], s=22, color=PALETTE["vermil"], marker="D",
                   zorder=4, label=f"IQR outlier ({int(flagged.sum())})")
        ax.axhline(info["iqr_hi"], color=PALETTE["orange"], lw=0.8, ls=":")
        ax.axhline(info["iqr_lo"], color=PALETTE["orange"], lw=0.8, ls=":")

        # Name the most extreme events, so they can be discussed rather than trimmed.
        if date_col in dataset.columns and len(info["iqr_outliers"]):
            worst = (ok - ok.median()).abs().sort_values(ascending=False).head(label_top)
            for idx in worst.index:
                d = pd.Timestamp(dataset[date_col].loc[idx]).date()
                ax.annotate(str(d), (1.18, ok.loc[idx]), fontsize=5.5,
                            color=PALETTE["vermil"], va="center")
        ax.set_xticks([])
        ax.set_title(_pretty(c), fontsize=8)
        ax.legend(fontsize=5.5, loc="lower right")
        summary.append({"variable": c, "n": info["n"],
                        "iqr_outliers": int(len(info["iqr_outliers"])),
                        "mad_outliers": int(len(info["mad_outliers"])),
                        "iqr_lo": info["iqr_lo"], "iqr_hi": info["iqr_hi"]})

    fig.suptitle("Outlier screen: IQR fences (1.5x) with extreme events named", y=1.02,
                 x=0.02, ha="left", fontweight="bold", fontsize=9)
    stamp(fig, f"n = {len(dataset)} events · outliers are RETAINED, never trimmed")
    save_figure(fig, name)
    return fig, pd.DataFrame(summary)


# --------------------------------------------------------------- distributions


def plot_feature_distributions(dataset, cols, ncols=4, bins=18,
                               name="eda_04_feature_distributions"):
    """Histogram grid with skew and excess kurtosis annotated on each panel.

    Reported because several models here are linear and the walk-forward split is
    chronological: a heavily skewed predictor whose tail sits entirely in one fold is a
    specific, checkable risk rather than a generic caveat.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    cols = [c for c in cols if c in dataset.columns]
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(WIDTH_FULL, 1.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    rows = []

    for ax, c in zip(axes, cols):
        s = dataset[c].astype(float).dropna()
        ax.hist(s, bins=bins, color=PALETTE["blue"], alpha=0.75, edgecolor="white",
                linewidth=0.3)
        ax.axvline(s.mean(), color=PALETTE["vermil"], lw=1.0, ls="-")
        ax.axvline(s.median(), color=PALETTE["orange"], lw=1.0, ls="--")
        sk = float(stats.skew(s)) if len(s) > 2 else np.nan
        ku = float(stats.kurtosis(s)) if len(s) > 3 else np.nan
        ax.set_title(f"{_pretty(c)}\nskew {sk:+.2f} · kurt {ku:+.2f}", fontsize=6)
        ax.tick_params(labelsize=5)
        ax.set_yticks([])
        rows.append({"variable": c, "n": len(s), "mean": s.mean(), "median": s.median(),
                     "std": s.std(), "skew": sk, "excess_kurtosis": ku,
                     "min": s.min(), "max": s.max()})

    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle("Predictor distributions — mean (solid) vs median (dashed)", y=1.005,
                 x=0.02, ha="left", fontweight="bold", fontsize=9)
    fig.tight_layout()
    stamp(fig, f"n = {len(dataset)} events")
    save_figure(fig, name)
    return fig, pd.DataFrame(rows)


def plot_qq_grid(dataset, target_cols, name="eda_05_qq_targets"):
    """Normal Q-Q plot per target, with the Shapiro-Wilk p-value annotated."""
    import matplotlib.pyplot as plt
    from scipy import stats

    cols = [c for c in target_cols if c in dataset.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(WIDTH_FULL, H_MED))
    axes = np.atleast_1d(axes)
    rows = []
    for ax, c in zip(axes, cols):
        s = dataset[c].astype(float).dropna()
        stats.probplot(s, dist="norm", plot=ax)
        ax.get_lines()[0].set(marker="o", markersize=3, color=PALETTE["blue"],
                              linestyle="none")
        ax.get_lines()[1].set(color=PALETTE["vermil"], lw=1.0)
        p = float(stats.shapiro(s).pvalue) if 3 <= len(s) <= 5000 else np.nan
        ax.set_title(f"{_pretty(c)}\nShapiro p = {p:.4f}", fontsize=7)
        ax.set_xlabel("theoretical quantiles", fontsize=6)
        ax.set_ylabel("observed", fontsize=6)
        ax.tick_params(labelsize=5)
        rows.append({"target": c, "n": len(s), "shapiro_p": p,
                     "normal_at_5pct": bool(p > 0.05) if np.isfinite(p) else None})
    fig.suptitle("Target normality — Q-Q against a normal reference", y=1.03, x=0.02,
                 ha="left", fontweight="bold", fontsize=9)
    fig.tight_layout()
    stamp(fig, "a low p rejects normality; RMSE and R² remain valid, Gaussian intervals do not")
    save_figure(fig, name)
    return fig, pd.DataFrame(rows)


# --------------------------------------------------------------- relationships


def plot_target_scatter_matrix(dataset, targets, features, name="eda_06_scatter_matrix"):
    """Scatter grid of each target against each candidate severity predictor.

    Spearman rho is annotated rather than Pearson: the severity variables are heavily
    skewed and several relationships would be monotone but not linear. The critical value
    at this n is drawn into the annotation so a reader can see immediately whether a
    correlation is distinguishable from zero.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    from .collinearity import critical_r

    targets = [t for t in targets if t in dataset.columns]
    features = [f for f in features if f in dataset.columns]
    fig, axes = plt.subplots(len(targets), len(features),
                             figsize=(WIDTH_FULL, 1.65 * len(targets)), squeeze=False)
    rows = []
    for i, t in enumerate(targets):
        for j, f in enumerate(features):
            ax = axes[i][j]
            sub = dataset[[t, f]].dropna()
            ax.scatter(sub[f], sub[t], s=10, color=PALETTE["blue"], alpha=0.6)
            if len(sub) > 3:
                rho, p = stats.spearmanr(sub[f], sub[t])
                rcrit = critical_r(len(sub))
                sig = abs(rho) >= rcrit
                # Only draw a trend where the correlation clears the critical value --
                # a fitted line through noise is the most misleading thing on a scatter.
                if sig:
                    b, a = np.polyfit(sub[f], sub[t], 1)
                    xs = np.linspace(sub[f].min(), sub[f].max(), 50)
                    ax.plot(xs, a + b * xs, color=PALETTE["vermil"], lw=1.1)
                ax.set_title(f"ρ={rho:+.2f}{'*' if sig else ''} (|ρ|crit={rcrit:.2f})",
                             fontsize=5.5, loc="right",
                             color=PALETTE["vermil"] if sig else PALETTE["grey"])
                rows.append({"target": t, "feature": f, "spearman_rho": float(rho),
                             "p": float(p), "n": len(sub), "critical_r": float(rcrit),
                             "distinguishable": bool(sig)})
            if j == 0:
                ax.set_ylabel(_pretty(t), fontsize=6)
            if i == len(targets) - 1:
                ax.set_xlabel(_pretty(f), fontsize=6)
            ax.tick_params(labelsize=5)
    fig.suptitle("Targets vs severity predictors (* = |ρ| above the critical value)",
                 y=1.005, x=0.02, ha="left", fontweight="bold", fontsize=9)
    fig.tight_layout()
    stamp(fig, f"n = {len(dataset)} events · Spearman, trend drawn only where significant")
    save_figure(fig, name)
    return fig, pd.DataFrame(rows)


def plot_feature_target_correlation(dataset, target, feature_cols, top_n=20,
                                    name=None):
    """Ranked |Spearman| of every predictor against one target, with the critical line."""
    import matplotlib.pyplot as plt
    from scipy import stats

    from .collinearity import critical_r

    name = name or f"eda_07_corr_{target}"
    feature_cols = [c for c in feature_cols if c in dataset.columns]
    rows = []
    for c in feature_cols:
        sub = dataset[[target, c]].dropna()
        if len(sub) < 4 or sub[c].nunique() < 3:
            continue
        rho, p = stats.spearmanr(sub[c], sub[target])
        rows.append({"feature": c, "rho": float(rho), "p": float(p), "n": len(sub)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return None, frame
    frame["abs_rho"] = frame.rho.abs()
    frame = frame.sort_values("abs_rho", ascending=False).head(top_n)
    rcrit = critical_r(int(frame.n.median()))

    fig, ax = plt.subplots(figsize=(WIDTH_FULL, max(H_MED, 0.2 * len(frame))))
    colours = [PALETTE["blue"] if r > 0 else PALETTE["orange"] for r in frame.rho]
    ax.barh([_pretty(f) for f in frame.feature][::-1], frame.rho.to_numpy()[::-1],
            color=colours[::-1])
    for v, lbl in ((rcrit, f"critical |ρ| = {rcrit:.2f}"), (-rcrit, None)):
        ax.axvline(v, color=PALETTE["vermil"], lw=0.9, ls="--", label=lbl)
    ax.axvline(0, color=PALETTE["ink"], lw=0.8)
    ax.set_xlabel("Spearman ρ with " + _pretty(target))
    ax.set_title(f"Predictor association with {_pretty(target)}")
    ax.tick_params(axis="y", labelsize=6)
    ax.legend(fontsize=6)
    n_sig = int((frame.abs_rho >= rcrit).sum())
    stamp(fig, f"n = {int(frame.n.median())} events · {n_sig} of {len(frame)} shown clear "
               f"the critical value")
    save_figure(fig, name)
    return fig, frame


# --------------------------------------------------------------- acquisition / context


def plot_event_timeline(market, dataset, price_col="aspi_close", date_col="date",
                        event_col="event_date", size_col="log_population_affected",
                        name="eda_08_event_timeline"):
    """The ASPI series with every modelled event marked and scaled by severity.

    This is the acquisition figure: it shows at a glance what period the market series
    covers, where the events fall inside it, and where coverage begins and ends.
    """
    import matplotlib.pyplot as plt

    m = market.sort_values(date_col)
    fig, ax = plt.subplots(figsize=(WIDTH_FULL, H_MED))
    ax.plot(m[date_col], m[price_col], color=PALETTE["ink"], lw=0.7)
    ax.set_yscale("log")

    ev = dataset.copy()
    ev[event_col] = pd.to_datetime(ev[event_col])
    y = np.interp(ev[event_col].map(pd.Timestamp.toordinal),
                  m[date_col].map(pd.Timestamp.toordinal), m[price_col])
    if size_col in ev.columns:
        s = ev[size_col].to_numpy(dtype=float)
        spread = float(np.nanmax(s) - np.nanmin(s)) or 1.0
        s = 8 + 60 * (s - np.nanmin(s)) / spread
    else:
        s = 18
    ax.scatter(ev[event_col], y, s=s, color=PALETTE["vermil"], alpha=0.65,
               edgecolor="white", linewidth=0.3, zorder=3, label="modelled event")
    ax.set_ylabel(f"{_pretty(price_col)} (log scale)")
    ax.set_title("Market series and modelled events")
    ax.legend(fontsize=6.5, loc="upper left")
    stamp(fig, f"{len(m)} trading days {m[date_col].min().date()} → {m[date_col].max().date()} · "
               f"{len(ev)} events · marker area ∝ population affected")
    save_figure(fig, name)
    return fig


def plot_class_balance(label_frame, name="eda_09_class_balance"):
    """Prevalence per classification label, against the 0.5 line.

    A label far from 0.5 is one where raw accuracy is gameable by the majority rule, and
    that is exactly the trap `C3_recovers_in_90` falls into at prevalence 0.90. Drawing
    the distance from 0.5 makes it visible before any model is fitted.
    """
    import matplotlib.pyplot as plt

    frame = label_frame.sort_values("prevalence")
    fig, ax = plt.subplots(figsize=(WIDTH_FULL, max(H_MED, 0.35 * len(frame))))
    colours = [PALETTE["vermil"] if abs(p - 0.5) > 0.25 else
               PALETTE["orange"] if abs(p - 0.5) > 0.15 else PALETTE["green"]
               for p in frame.prevalence]
    ax.barh(frame.label, frame.prevalence, color=colours)
    ax.axvline(0.5, color=PALETTE["ink"], lw=1.0, ls="--", label="balanced (0.5)")
    for y, (_, r) in enumerate(frame.iterrows()):
        ax.text(r.prevalence + 0.01, y, f"{r.prevalence:.2f}  (maj. rule {max(r.prevalence, 1 - r.prevalence):.2f})",
                va="center", fontsize=6)
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("prevalence of the positive class")
    ax.set_title("Class balance per label — red labels make raw accuracy meaningless")
    ax.legend(fontsize=6.5)
    stamp(fig, f"{len(frame)} labels · the majority rule scores the value in brackets")
    save_figure(fig, name)
    return fig


def plot_sector_coverage(sector_long, date_col="date", sector_col="sector",
                        value_col="close", name="eda_10_sector_coverage"):
    """First and last observation per sector index, so the unbalanced panel is visible."""
    import matplotlib.pyplot as plt

    g = (sector_long.dropna(subset=[value_col])
         .groupby(sector_col)[date_col].agg(["min", "max", "count"])
         .sort_values("min"))
    fig, ax = plt.subplots(figsize=(WIDTH_FULL, max(H_MED, 0.22 * len(g))))
    full_start, full_end = g["min"].min(), g["max"].max()
    for y, (sec, r) in enumerate(g.iterrows()):
        short = r["min"] > full_start or r["max"] < full_end
        ax.hlines(y, r["min"], r["max"], lw=4,
                  color=PALETTE["orange"] if short else PALETTE["blue"])
        ax.text(r["max"], y, f"  {int(r['count'])}", va="center", fontsize=5.5)
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels(g.index, fontsize=6)
    ax.set_title("Sector index coverage — orange bars do not span the window")
    stamp(fig, f"{len(g)} sector indices · {full_start.date()} → {full_end.date()}")
    save_figure(fig, name)
    return fig, g.reset_index()


def eda_manifest(figure_dir=None) -> pd.DataFrame:
    """Every EDA figure on disk, with size, for the thesis figure list."""
    from .figures import FIGURE_DIR

    d = figure_dir or FIGURE_DIR
    rows = [{"filename": p.name, "kb": p.stat().st_size // 1024}
            for p in sorted(d.glob("eda_*.png"))]
    return pd.DataFrame(rows)
