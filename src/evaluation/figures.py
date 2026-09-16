"""Thesis figure suite.

The pipeline previously produced two images across 59 notebook cells, both SHAP, neither
exported to disk. This module is the single place where figures are defined, styled and
saved, so that thirty figures cost thirty three-line notebook cells rather than thirty
bespoke ones.

Two conventions run through everything here, and both exist because the underlying
results are weak:

- **Every evaluation figure carries its sample size on its face** (see `stamp`). A ROC
  curve drawn from 30 points looks exactly like one drawn from 30,000. Putting "n = 30
  pooled out-of-fold points" in the corner is what stops a reader over-reading it.
- **Nothing is encoded by hue alone.** Every categorical series carries colour *and*
  marker *and* linestyle, and magnitude heatmaps use a monotonic-lightness colormap, so
  the figures survive greyscale printing.

Where a figure could imply performance the numbers do not support, the honest reference
is drawn in: chance diagonals on ROC, prevalence lines on precision-recall, naive-baseline
rules on every metric bar, marginal-interval comparisons on every conformal plot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from . import collinearity as col
from .metrics import bootstrap_metric_ci, hanley_mcneil_ci, wilson_ci

# Default output location: <repo root>/docs/figures.
# figures.py -> evaluation[0] -> src[1] -> repo root[2]
FIGURE_DIR = Path(__file__).resolve().parents[2] / "docs" / "figures"

# Okabe-Ito, colourblind-safe, ordered so the leading entries also separate in luminance.
PALETTE = {
    "ink": "#000000", "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "vermil": "#D55E00", "sky": "#56B4E9", "pink": "#CC79A7", "yellow": "#F0E442",
    "grey": "#666666",
}
CATEGORICAL = [PALETTE[k] for k in
               ("blue", "orange", "green", "vermil", "sky", "pink", "yellow", "grey")]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1)),
              (0, (3, 5, 1, 5))]

DIVERGING = "RdBu_r"    # signed quantities (correlation)
SEQUENTIAL = "cividis"  # magnitude only; monotonic lightness, so greyscale-safe
NEUTRAL = "0.92"        # masked / not-distinguishable-from-zero cells

# A4 with 2.5 cm margins gives a 16 cm text block = 6.3 in.
WIDTH_FULL, WIDTH_HALF = 6.3, 3.1
H_SHORT, H_MED, H_TALL = 2.4, 3.6, 5.0

# Y1 rebaselined onto the pre-event close (methodology-audit finding #8, 2026-09-16),
# absorbing EventWindow_0_5's formula -- there is no separate CAR[0,+5] entry below.
TARGET_LABELS = {
    "Y1_ASPI_5D_Forward_LogReturn_Pct": "Y1 · 5-day cumulative return from pre-event close",
    "Y2_abnormal_volume": "Y2 · abnormal volume (V / 30-day mean − 1)",
    "Y3_recovery_days": "Y3 · recovery time (trading days, capped 90)",
    "Y1_EventWindow_0_10_LogReturn_Pct": "CAR[0,+10] · cumulative return, 10 trading days",
}
# Short forms for multi-panel figures, where the full labels collide.
SHORT_TARGET_LABELS = {
    "Y1_ASPI_5D_Forward_LogReturn_Pct": "Y1 · CAR[0,+5]",
    "Y2_abnormal_volume": "Y2 · abnormal volume",
    "Y3_recovery_days": "Y3 · recovery days",
    "Y1_sector_log_return": "Y1 · sector log return",
    "Y3_sector_recovery_days": "Y3 · sector recovery",
    "Y1_EventWindow_0_10_LogReturn_Pct": "CAR[0,+10]",
}

MODEL_LABELS = {
    "ridge": "Ridge", "random_forest": "Random Forest", "xgboost": "XGBoost",
    "mlp": "MLP", "ensemble": "Ensemble", "stacked": "Stacked",
    "naive_zero": "Naive: zero", "naive_train_mean": "Naive: train mean",
}


def _label(name: str, mapping: dict) -> str:
    return mapping.get(name, name.replace("_", " "))


def _pretty(name: str) -> str:
    """Compact feature names so a 31-tick axis stays readable."""
    return (str(name)
            .replace("price_to_sma_", "P/SMA ").replace("price_to_ema_", "P/EMA ")
            .replace("rolling_std_", "vol ").replace("lag_return_t-", "ret lag")
            .replace("log_financial_damage", "log damage")
            .replace("log_population_affected", "log affected")
            .replace("financial_damage", "damage").replace("population_affected", "affected")
            .replace("disasters_trailing_365d", "events 365d")
            .replace("days_since_last_disaster", "days since last")
            .replace("log_damage_x_flood", "log damage × flood")
            .replace("damage_to_gdp", "damage/GDP").replace("sp500_log_return", "S&P 500 ret")
            .replace("gdp_growth_pct", "GDP growth").replace("inflation_cpi_pct", "inflation")
            # log_return and lag_return_t-1 both rendered as "ret t-1", giving the heatmap two
            # identically labelled rows. At an event's as-of row log_return is the previous
            # session's return and lag_return_t-1 the one before that.
            .replace("squared_return", "ret²").replace("log_return", "ret asof")
            .replace("disaster_", ""))


# --------------------------------------------------------------- style and export


def apply_thesis_style(base: int = 9, serif: bool = False) -> None:
    """Set rcParams for print-quality thesis figures. Idempotent; call once per notebook."""
    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.rcParams.update({
        "figure.figsize": (WIDTH_FULL, H_MED), "figure.dpi": 110,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "font.size": base, "axes.labelsize": base, "axes.titlesize": base + 1,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "xtick.labelsize": base - 1, "ytick.labelsize": base - 1,
        "legend.fontsize": base - 1, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "lines.linewidth": 1.4, "lines.markersize": 4.5,
        # colour AND marker AND linestyle, so nothing depends on hue alone
        "axes.prop_cycle": (cycler(color=CATEGORICAL) + cycler(marker=MARKERS)
                            + cycler(linestyle=LINESTYLES)),
    })
    if serif:
        plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix"})


def set_figure_dir(path) -> Path:
    """Redirect exports, used by the tests to write into a tmp_path."""
    global FIGURE_DIR
    FIGURE_DIR = Path(path)
    return FIGURE_DIR


def save_figure(fig, name: str, formats=("png",), dpi: int = 300, close: bool = False,
                verbose: bool = True) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / f"{name}.png"
    for ext in formats:
        p = FIGURE_DIR / f"{name}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        if ext == "png":
            out = p
    if verbose:
        print(f"saved -> {out.relative_to(out.parents[2]) if len(out.parents) > 2 else out} "
              f"({out.stat().st_size // 1024} KB)")
    if close:
        import matplotlib.pyplot as plt
        plt.close(fig)
    return out


def stamp(fig, text: str) -> None:
    """Small grey footer naming the sample size. Applied to every evaluation figure.

    This is the honesty mechanism: making n unavoidable on the face of the figure is what
    prevents a 30-point curve being read as a strong result.
    """
    fig.text(0.995, -0.02, text, ha="right", va="top", fontsize=6.5,
             color=PALETTE["grey"], transform=fig.transFigure)


# --------------------------------------------------------------- descriptive figures


def plot_target_distributions(dataset, target_cols, bins: int = 20):
    """Histogram + ECDF per target.

    The ECDF row is the point: a histogram of Y3 buries its point masses inside a bin,
    while the ECDF shows the mass at 0 and the mass at the 90 cap as two unmistakable
    vertical jumps.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(target_cols), figsize=(WIDTH_FULL, H_TALL))
    for j, t in enumerate(target_cols):
        v = pd.to_numeric(dataset[t], errors="coerce").dropna().to_numpy()
        ax = axes[0, j]
        ax.hist(v, bins=bins, color=PALETTE["blue"], edgecolor="white", linewidth=0.4)
        ax.plot(v, np.full_like(v, -0.5), "|", color=PALETTE["ink"], markersize=4, alpha=0.5)
        ax.set_title(_label(t, TARGET_LABELS), fontsize=8)
        ax.set_ylabel("events" if j == 0 else "")

        ax2 = axes[1, j]
        ax2.ecdf(v, color=PALETTE["vermil"], marker=None, linestyle="-")
        ax2.set_ylabel("ECDF" if j == 0 else "")
        ax2.set_ylim(0, 1)
        if t == "Y3_recovery_days":
            p0, p90 = float(np.mean(v == 0)), float(np.mean(v >= 90))
            ax2.annotate(f"P(Y3=0) = {p0:.2f}\nP(Y3=90) = {p90:.2f}", xy=(0.45, 0.25),
                         xycoords="axes fraction", fontsize=7)
        ax2.annotate(f"median {np.median(v):.4g}", xy=(0.45, 0.06),
                     xycoords="axes fraction", fontsize=7)
    stamp(fig, f"N = {len(dataset)} events")
    fig.tight_layout()
    return fig


def plot_target_dependence_y1_y3(dataset, y1="Y1_ASPI_5D_Forward_LogReturn_Pct", y3="Y3_recovery_days"):
    """Make the `Y3 = 0 <=> Y1 >= 0` identity visible rather than only asserted in prose.

    The recovery window includes the event day and the baseline is the prior close, so a
    non-negative event-day return recovers on day 0 by construction. Over half of Y3 is
    therefore a mechanical restatement of the sign of Y1, and the off-diagonal of the 2x2
    contingency is exactly zero.
    """
    import matplotlib.pyplot as plt

    d = dataset[[y1, y3]].dropna()
    a, b = d[y1].to_numpy(), d[y3].to_numpy()
    zero = b == 0

    fig, (ax, axt) = plt.subplots(1, 2, figsize=(WIDTH_FULL, H_MED),
                                  gridspec_kw={"width_ratios": [2.2, 1]})
    ax.scatter(a[zero], b[zero], s=26, marker="o", facecolor="none",
               edgecolor=PALETTE["blue"], linewidth=1.0, label="Y3 = 0")
    ax.scatter(a[~zero], b[~zero], s=26, marker="^", color=PALETTE["vermil"], label="Y3 > 0")
    ax.axvline(0.0, color=PALETTE["ink"], linewidth=1.2)
    ax.set_xlabel(_label(y1, TARGET_LABELS))
    ax.set_ylabel(_label(y3, TARGET_LABELS))
    ax.set_title("Y3 is mechanically determined by the sign of Y1")
    ax.legend(loc="upper left")

    tab = np.array([[int(np.sum((a >= 0) & zero)), int(np.sum((a >= 0) & ~zero))],
                    [int(np.sum((a < 0) & zero)), int(np.sum((a < 0) & ~zero))]])
    axt.imshow(tab, cmap=SEQUENTIAL)
    axt.set_xticks([0, 1], ["Y3 = 0", "Y3 > 0"], fontsize=7)
    axt.set_yticks([0, 1], ["Y1 ≥ 0", "Y1 < 0"], fontsize=7)
    for i in range(2):
        for j in range(2):
            axt.text(j, i, str(tab[i, j]), ha="center", va="center", fontsize=10,
                     color="white" if tab[i, j] < tab.max() * 0.6 else "black")
    axt.set_title("contingency", fontsize=8)
    axt.grid(False)
    stamp(fig, f"N = {len(d)} events · off-diagonal cell (Y1 ≥ 0 and Y3 > 0) should be 0")
    fig.tight_layout()
    return fig


def plot_feature_correlation_heatmap(X, method: str = "spearman", cluster: bool = True,
                                     annot_threshold: float = 0.70, mask_insignificant=True):
    """Clustered feature correlation heatmap -- the collinearity evidence for P2-3.

    Spearman by default: damage is zero-heavy with a long tail, so Pearson there rides on
    two or three points, and monotone-invariance makes a column and its own log read 1.00.
    Cells below the n-dependent significance threshold are greyed out."""
    import matplotlib.pyplot as plt

    R = col.correlation_matrix(X, method=method)
    names = list(R.columns)
    order = col.cluster_order(R)[0] if cluster and len(names) > 2 else list(range(len(names)))
    R = R.iloc[order, order]
    names = [names[i] for i in order]

    M = R.to_numpy(dtype=float).copy()
    r_crit = col.critical_r(len(X))
    if mask_insignificant and np.isfinite(r_crit):
        M = np.ma.masked_where(np.abs(M) < r_crit, M)

    cmap = matplotlib.colormaps[DIVERGING].copy()
    cmap.set_bad(NEUTRAL)

    size = max(WIDTH_FULL, 0.20 * len(names) + 1.6)
    fig, ax = plt.subplots(figsize=(size, size * 0.92))
    im = ax.imshow(M, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), [_pretty(n) for n in names], rotation=90, fontsize=6)
    ax.set_yticks(range(len(names)), [_pretty(n) for n in names], fontsize=6)
    ax.grid(False)

    # Colour the tick labels by thematic block, so the reader gets the data-driven
    # ordering and the thematic reading from one figure instead of two.
    groups = sorted({col.assign_group(n) for n in names})
    gcolor = {g: CATEGORICAL[i % len(CATEGORICAL)] for i, g in enumerate(groups)}
    for lab, n in zip(ax.get_xticklabels(), names):
        lab.set_color(gcolor[col.assign_group(n)])
    for lab, n in zip(ax.get_yticklabels(), names):
        lab.set_color(gcolor[col.assign_group(n)])

    for i in range(len(names)):
        for j in range(len(names)):
            v = R.to_numpy()[i, j]
            if i != j and abs(v) >= annot_threshold:
                ax.text(j, i, f"{v:.2f}".replace("0.", "."), ha="center", va="center",
                        fontsize=5.0, color="white" if abs(v) > 0.85 else "black")

    fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02, label=f"{method} rho")
    ax.set_title(f"Feature correlation ({method}, hierarchically clustered)")
    n_pairs = len(names) * (len(names) - 1) // 2
    stamp(fig, f"n = {len(X)} events · |rho| < {r_crit:.2f} greyed (not distinguishable from 0 "
               f"at alpha=.05) · {n_pairs} pairwise tests, ~{0.05 * n_pairs:.0f} spurious expected")
    fig.tight_layout()
    return fig


def plot_vif(X, drop_reference: str = "disaster_Other", thresholds=(5.0, 10.0)):
    """VIF bars on a log axis, plus the condition-index panel.

    One disaster one-hot is dropped: the four indicators sum to 1, so with an intercept
    the design is exactly singular and the whole block returns inf. The condition index is
    the companion that never blows up and gives one headline number.
    """
    import matplotlib.pyplot as plt

    vif = col.compute_vif(X, drop_reference=drop_reference).dropna(subset=["vif"])
    ci = col.condition_indices(X)

    fig, (ax, axc) = plt.subplots(1, 2, figsize=(WIDTH_FULL, H_TALL),
                                  gridspec_kw={"width_ratios": [3, 1]})
    finite = np.isfinite(vif["vif"].to_numpy())
    plot_v = np.where(finite, vif["vif"].to_numpy(), np.nanmax(vif["vif"][finite]) * 2 if finite.any() else 1e3)
    groups = sorted(vif["group"].unique())
    gcolor = {g: CATEGORICAL[i % len(CATEGORICAL)] for i, g in enumerate(groups)}
    ax.barh([_pretty(f) for f in vif["feature"]], plot_v,
            color=[gcolor[g] for g in vif["group"]])
    for i, ok in enumerate(finite):
        if not ok:
            ax.text(plot_v[i], i, " ∞", va="center", fontsize=8)
    ax.set_xscale("log")
    for th, ls in zip(thresholds, ("--", ":")):
        ax.axvline(th, color=PALETTE["ink"], linestyle=ls, linewidth=1.0, label=f"VIF = {th:g}")
    ax.invert_yaxis()
    ax.set_xlabel("variance inflation factor (log scale)")
    ax.set_title("Collinearity: VIF")
    ax.legend(loc="lower right")
    ax.tick_params(axis="y", labelsize=6)

    axc.plot(range(len(ci)), np.sort(ci)[::-1], marker="o", linestyle="-",
             color=PALETTE["vermil"])
    axc.axhline(30, color=PALETTE["ink"], linestyle="--", linewidth=1.0)
    axc.set_yscale("log")
    axc.set_title(f"condition index\nmax = {ci.max():.0f}", fontsize=8)
    axc.set_xlabel("component")
    stamp(fig, f"n = {len(X)} events, p = {X.shape[1]} features · "
               f"'{drop_reference}' dropped as the one-hot reference · index > 30 is the usual concern")
    fig.tight_layout()
    return fig


def plot_walk_forward_folds(splits, dataset, y, target="Y1_ASPI_5D_Forward_LogReturn_Pct",
                            date_col="event_date", mark=("2004-12-26", "2005-11-21")):
    """Fold geometry over the event sequence, with the target stemmed above it.

    Shows at a glance that the two largest shocks sit permanently inside fold 0's
    *training* window and appear in no test fold -- which is why pooled test variance is a
    fraction of full-sample variance, and therefore why pooled R2 is measured against a
    compressed denominator.
    """
    import matplotlib.pyplot as plt

    n = len(dataset)
    vals = pd.to_numeric(y[target], errors="coerce").to_numpy()
    fig, (axt, ax) = plt.subplots(2, 1, figsize=(WIDTH_FULL, H_TALL), sharex=True,
                                  gridspec_kw={"height_ratios": [1, 1.3]})

    axt.stem(range(n), vals, basefmt=" ", markerfmt="o", linefmt="-")
    for line in axt.get_lines():
        line.set_color(PALETTE["grey"])
        line.set_markersize(2.5)
    axt.axhline(0, color=PALETTE["ink"], linewidth=0.8)
    axt.set_ylabel(_label(target, TARGET_LABELS), fontsize=7)

    marked = []
    if date_col in dataset.columns:
        dates = pd.to_datetime(dataset[date_col])
        for m in mark:
            hits = np.where(dates.dt.strftime("%Y-%m-%d") == m)[0]
            for h in hits:
                marked.append((int(h), m))
    for pos, lab in marked:
        for a in (axt, ax):
            a.axvline(pos, color=PALETTE["vermil"], linestyle="--", linewidth=1.0)
        axt.annotate(lab, xy=(pos, vals[pos]), fontsize=6, rotation=90,
                     textcoords="offset points", xytext=(3, 4))

    for i, s in enumerate(splits):
        tr, te = np.asarray(s.train_index), np.asarray(s.test_index)
        ax.barh(i, tr.max() - tr.min() + 1, left=tr.min(), height=0.6,
                color=PALETTE["grey"], alpha=0.55, hatch="///", edgecolor="white")
        ax.barh(i, te.max() - te.min() + 1, left=te.min(), height=0.6,
                color=PALETTE["blue"], edgecolor="white")
        sd = np.nanstd(vals[te])
        ax.text(te.max() + 0.6, i, f"n={len(te)}, σ={sd:.4g}", va="center", fontsize=6)
    ax.set_yticks(range(len(splits)), [f"fold {i}" for i in range(len(splits))])
    ax.invert_yaxis()
    ax.set_xlabel("event index (chronological)")
    ax.set_title("Walk-forward geometry: hatched = train, solid = test")

    full_sd, test_sd = np.nanstd(vals), np.nanstd(
        vals[np.concatenate([np.asarray(s.test_index) for s in splits])])
    stamp(fig, f"{len(splits)} folds · full-sample σ = {full_sd:.4g}, pooled-test σ = "
               f"{test_sd:.4g} (ratio {test_sd / full_sd:.2f})")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------- evaluation figures


def plot_model_vs_baseline(results, target_cols, metric="rmse",
                           baselines=("naive_zero", "naive_train_mean")):
    """Pooled metric per model, with both naive baselines drawn as reference RULES.

    A threshold is drawn as a line, not as another bar: the question is whether a model
    clears the null, not how the null ranks among models. Bars that fail to clear it are
    hatched rather than recoloured, so the distinction survives greyscale.
    """
    import matplotlib.pyplot as plt
    from .metrics import pooled_arrays

    def _score(yt, yp):
        return (float(np.sqrt(np.mean((yp - yt) ** 2))) if metric == "rmse"
                else float(np.mean(np.abs(yp - yt))))

    models = [m for m in results if m not in baselines]
    fig, axes = plt.subplots(1, len(target_cols), figsize=(WIDTH_FULL, H_MED))
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, target_cols):
        vals, labels = [], []
        for m in models:
            if t not in results[m]:
                continue
            yt, yp = pooled_arrays(results, m, t)
            vals.append(_score(yt, yp))
            labels.append(_label(m, MODEL_LABELS))
        order = np.argsort(vals)
        vals = list(np.array(vals)[order])
        labels = list(np.array(labels)[order])

        base_vals = {}
        for b in baselines:
            if b in results and t in results[b]:
                ytb, ypb = pooled_arrays(results, b, t)
                base_vals[b] = _score(ytb, ypb)
        worst_null = max(base_vals.values()) if base_vals else np.inf

        bars = ax.barh(labels, vals, color=PALETTE["blue"])
        for bar, v in zip(bars, vals):
            if v >= worst_null:
                bar.set_hatch("///")
                bar.set_facecolor(PALETTE["grey"])
        for b, ls in zip(base_vals, ("--", ":")):
            ax.axvline(base_vals[b], color=PALETTE["ink"], linestyle=ls, linewidth=1.1,
                       label=_label(b, MODEL_LABELS))
        ax.invert_yaxis()
        ax.set_title(_label(t, SHORT_TARGET_LABELS), fontsize=8)
        ax.set_xlabel(metric.upper())
        ax.tick_params(axis="y", labelsize=7)
    axes[0].legend(loc="lower right", fontsize=6)
    stamp(fig, "hatched = does not beat the weaker naive baseline · pooled out-of-fold")
    fig.tight_layout()
    return fig


def plot_skill_forest(results, target_cols, reference="naive_zero", n_boot=2000,
                      random_state=42):
    """Forest plot of skill score 1 - RMSE_model/RMSE_reference, with bootstrap CIs.

    This is the figure that converts a `beats_null` boolean into "beats the null *within
    noise*". Where no model beats the null, every whisker crosses zero, and the figure
    says so without needing a caption to explain it away.
    """
    import matplotlib.pyplot as plt
    from .metrics import pooled_arrays

    models = [m for m in results if not m.startswith("naive_")]
    fig, axes = plt.subplots(1, len(target_cols), figsize=(WIDTH_FULL, H_MED), sharex=False)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(random_state)

    for ax, t in zip(axes, target_cols):
        if reference not in results or t not in results[reference]:
            continue
        ytr, ypr = pooled_arrays(results, reference, t)
        ys, los, his, labels = [], [], [], []
        for m in models:
            if t not in results[m]:
                continue
            yt, yp = pooled_arrays(results, m, t)
            k = min(len(yt), len(ytr))
            e_m, e_r, = (yp - yt)[-k:], (ypr - ytr)[-k:]
            point = 1.0 - np.sqrt(np.mean(e_m ** 2)) / np.sqrt(np.mean(e_r ** 2))
            idx = rng.integers(0, k, size=(n_boot, k))
            draws = 1.0 - (np.sqrt(np.mean(e_m[idx] ** 2, axis=1))
                           / np.sqrt(np.mean(e_r[idx] ** 2, axis=1)))
            lo, hi = np.quantile(draws, [0.025, 0.975])
            ys.append(point); los.append(lo); his.append(hi)
            labels.append(_label(m, MODEL_LABELS))
        pos = np.arange(len(ys))
        ax.errorbar(ys, pos, xerr=[np.array(ys) - np.array(los), np.array(his) - np.array(ys)],
                    fmt="o", color=PALETTE["blue"], ecolor=PALETTE["grey"], capsize=2,
                    linestyle="none")
        ax.axvline(0.0, color=PALETTE["ink"], linewidth=1.3)
        ax.set_yticks(pos, labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(_label(t, SHORT_TARGET_LABELS), fontsize=8)
        ax.set_xlabel(f"skill vs {_label(reference, MODEL_LABELS)}")
    stamp(fig, f"positive = better than the null · {n_boot} bootstrap resamples of the "
               f"pooled out-of-fold events · a whisker crossing 0 is not a win")
    fig.tight_layout()
    return fig


def plot_pred_vs_actual(results, models, target, splits=None, dataset=None):
    """Predicted against actual, with the identity line and the null prediction.

    The characteristic small-sample failure -- predictions collapsing into a narrow
    horizontal band far from the identity line -- is what a negative R2 looks like, and
    seeing it is more convincing than reading the number.
    """
    import matplotlib.pyplot as plt
    from .metrics import pooled_arrays

    models = [m for m in models if target in results.get(m, {})]
    if not models:
        raise ValueError(f"no model in `models` has results for {target}")
    fig, axes = plt.subplots(1, len(models), figsize=(WIDTH_FULL, H_MED + 0.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        yt, yp = pooled_arrays(results, m, target)
        lim = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        ax.plot(lim, lim, linestyle="--", color=PALETTE["grey"], marker=None, linewidth=1.0)
        ax.axhline(float(np.mean(yt)), color=PALETTE["orange"], linestyle=":", marker=None,
                   linewidth=1.0)
        ax.scatter(yt, yp, s=24, color=PALETTE["blue"], marker="o", alpha=0.85)
        ss = 1 - np.sum((yp - yt) ** 2) / np.sum((yt - yt.mean()) ** 2)
        ax.set_title(f"{_label(m, MODEL_LABELS)}\npooled R² = {ss:.3f}, n = {len(yt)}",
                     fontsize=8)
        ax.set_xlabel("actual")
    axes[0].set_ylabel("predicted")
    stamp(fig, f"{_label(target, TARGET_LABELS)} · dashed = identity, dotted = test mean")
    fig.tight_layout()
    return fig


def plot_overfitting_gap(train_fit_r2: dict, results, target_cols, model="random_forest"):
    """Slope chart from in-sample R2 down to held-out pooled R2, per target."""
    import matplotlib.pyplot as plt
    from .metrics import pooled_arrays

    fig, ax = plt.subplots(figsize=(WIDTH_FULL, H_MED))
    for i, t in enumerate(target_cols):
        if t not in train_fit_r2 or t not in results.get(model, {}):
            continue
        yt, yp = pooled_arrays(results, model, t)
        held = 1 - np.sum((yp - yt) ** 2) / np.sum((yt - yt.mean()) ** 2)
        tr = float(train_fit_r2[t])
        ax.plot([0, 1], [tr, held], marker="o", color=CATEGORICAL[i % len(CATEGORICAL)],
                linestyle="-", label=f"{_label(t, TARGET_LABELS)} (gap {tr - held:.2f})")
        ax.annotate(f"{tr:.2f}", (0, tr), textcoords="offset points", xytext=(-22, -3),
                    fontsize=7)
        ax.annotate(f"{held:.2f}", (1, held), textcoords="offset points", xytext=(6, -3),
                    fontsize=7)
    ax.axhline(0, color=PALETTE["ink"], linewidth=1.2)
    ax.set_xticks([0, 1], ["in-sample refit", "held-out (pooled)"])
    ax.set_xlim(-0.25, 1.35)
    ax.set_ylabel("R²")
    ax.set_title(f"Memorisation gap — {_label(model, MODEL_LABELS)}")
    ax.legend(loc="lower left", fontsize=6)
    stamp(fig, "in-sample R² is not predictive performance; the gap is the overfitting")
    fig.tight_layout()
    return fig


def plot_roc_with_ci(results, models, target="Y1_ASPI_5D_Forward_LogReturn_Pct", n_boot=2000,
                     random_state=42, positive="negative_return"):
    """ROC with a bootstrap band, the chance diagonal, and an auto-flag when the CI covers 0.5.

    The auto-annotation is the safeguard: the figure states its own inconclusiveness, so it
    cannot be over-read by someone skimming the legend.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve
    from .metrics import pooled_arrays

    fig, ax = plt.subplots(figsize=(WIDTH_HALF * 1.7, H_MED + 0.3))
    ax.plot([0, 1], [0, 1], color=PALETTE["ink"], linestyle="--", marker=None,
            linewidth=1.3, label="chance (AUC 0.5)")
    grid = np.linspace(0, 1, 101)
    rng = np.random.default_rng(random_state)

    for i, m in enumerate(models):
        if target not in results.get(m, {}):
            continue
        yt, yp = pooled_arrays(results, m, target)
        lab = (yt < 0).astype(int)          # 'crash' is the positive class
        score = -yp                          # a bigger predicted drop scores more crash-like
        if lab.sum() == 0 or lab.sum() == len(lab):
            continue
        fpr, tpr, _ = roc_curve(lab, score)
        auc = roc_auc_score(lab, score)
        lo, hi = hanley_mcneil_ci(auc, int(lab.sum()), int((1 - lab).sum()))

        curves, discarded = [], 0
        for _ in range(n_boot):
            idx = rng.integers(0, len(lab), len(lab))
            if lab[idx].sum() in (0, len(idx)):
                discarded += 1
                continue
            f, t_, _ = roc_curve(lab[idx], score[idx])
            curves.append(np.interp(grid, f, t_))
        if curves:
            band = np.array(curves)
            ax.fill_between(grid, np.quantile(band, 0.025, axis=0),
                            np.quantile(band, 0.975, axis=0),
                            color=CATEGORICAL[i % len(CATEGORICAL)], alpha=0.12, linewidth=0)
        flag = " — includes 0.5" if lo <= 0.5 else ""
        ax.plot(fpr, tpr, marker=None, color=CATEGORICAL[i % len(CATEGORICAL)],
                linestyle=LINESTYLES[i % len(LINESTYLES)],
                label=f"{_label(m, MODEL_LABELS)}: {auc:.3f} [{lo:.2f}, {hi:.2f}]{flag}")

    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC — Y1 direction (positive class = negative return)")
    ax.legend(loc="lower right", fontsize=6)
    stamp(fig, "AUC with Hanley–McNeil 95% CI · shaded = bootstrap band")
    fig.tight_layout()
    return fig


def plot_conformal_coverage(coverage_frame):
    """Achieved coverage with Wilson bars, beside mean half-width.

    Coverage without width is meaningless -- an infinitely wide interval covers
    everything -- so the two panels are always drawn together, and the marginal
    constant-predictor interval is always shown alongside the model's.
    """
    import matplotlib.pyplot as plt

    fig, (ax, axw) = plt.subplots(1, 2, figsize=(WIDTH_FULL, H_MED))
    targets = list(dict.fromkeys(coverage_frame["target"]))
    width = 0.36
    for k, kind in enumerate(("model", "marginal")):
        sub = coverage_frame[coverage_frame["kind"] == kind]
        pos = np.arange(len(targets)) + (k - 0.5) * width
        cov = [float(sub[sub["target"] == t]["coverage"].iloc[0]) for t in targets]
        lo, hi = [], []
        for t in targets:
            row = sub[sub["target"] == t].iloc[0]
            a, b = wilson_ci(int(row["covered"]), int(row["n"]))
            lo.append(a); hi.append(b)
        ax.errorbar(pos, cov, yerr=[np.array(cov) - np.array(lo), np.array(hi) - np.array(cov)],
                    fmt=MARKERS[k], color=CATEGORICAL[k], capsize=3, linestyle="none",
                    label=kind)
        axw.bar(pos, [float(sub[sub["target"] == t]["half_width"].iloc[0]) for t in targets],
                width=width, color=CATEGORICAL[k], label=kind)

    nominal = float(coverage_frame["nominal"].iloc[0]) if "nominal" in coverage_frame else 0.8
    ax.axhline(nominal, color=PALETTE["ink"], linestyle="--", linewidth=1.1,
               label=f"nominal {nominal:.0%}")
    for a in (ax, axw):
        a.set_xticks(range(len(targets)), [t.split("_")[0] for t in targets])
        a.legend(fontsize=6)
    ax.set_ylabel("empirical coverage")
    ax.set_title("Conformal coverage (Wilson 95% CI)")
    axw.set_ylabel("mean half-width")
    axw.set_yscale("log")
    axw.set_title("Interval width — lower is better")
    stamp(fig, "a model interval must be NARROWER than the marginal one at comparable "
               "coverage to be worth anything")
    fig.tight_layout()
    return fig


def figure_manifest(rows=None) -> pd.DataFrame:
    """Inventory of exported figures, for the thesis figure list."""
    files = sorted(FIGURE_DIR.glob("fig_*.png"))
    return pd.DataFrame([{"filename": f.name, "kb": f.stat().st_size // 1024} for f in files])
