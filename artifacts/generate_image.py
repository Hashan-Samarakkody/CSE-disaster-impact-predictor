"""Manuscript Figures 16 and 20, drawn only from committed artifacts (Round-3 revision).

Figure 16: predicted versus realised five-session return, real-time information set,
           confirmatory capacity k = 10 (ridge, random forest, MLP).
Figure 20: paired real-time versus ex-post RMSE for the five-session return, k = 10.
Usage: python scripts/build_manuscript_figures.py   (from the repository root)
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path(".")
T = ROOT / "artifacts" / "tables"
OUT = ROOT / "docs" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
pred = pd.read_parquet(T / "aspi_grid_predictions.parquet")
met = pd.read_parquet(T / "aspi_grid_metrics.parquet")
NAMES = {"ridge": "Ridge", "random_forest": "Random forest", "mlp": "MLP", "elastic_net": "Elastic net", "xgboost": "XGBoost"}

# ---- Figure 16
sel = pred[(pred.horizon == 5) & (pred.k == 10) & (pred.info_set == "real_time")]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharex=True, sharey=True)
lim = np.ceil(max(abs(sel.y_true).max(), abs(sel.y_pred).max()) + 0.5)
for ax, m in zip(axes, ["ridge", "random_forest", "mlp"]):
    g = sel[sel.model == m]
    rmse = np.sqrt(np.mean((g.y_pred - g.y_true) ** 2))
    ax.scatter(g.y_true, g.y_pred, s=18, alpha=0.8)
    ax.plot([-lim, lim], [-lim, lim], "--", color="grey", lw=1, label="Identity")
    ax.axhline(0, color="black", lw=0.6, label="Zero benchmark")
    ax.set_title(f"{NAMES[m]} (RMSE = {rmse:.3f}, n = {len(g)})", fontsize=10)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_xlabel("Realised return (percentage points)")
axes[0].set_ylabel("Predicted return (percentage points)"); axes[0].legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig_16_pred_vs_realised_realtime_k10.png", dpi=200); plt.close(fig)

# ---- Figure 20
m5 = met[(met.horizon == 5) & (met.k == 10)]
zero = met[(met.horizon == 5) & (met.model == "naive_zero")].rmse.iloc[0]
fig, ax = plt.subplots(figsize=(5.2, 4.8))
for m in ["ridge", "random_forest", "mlp", "elastic_net", "xgboost"]:
    rt = m5[(m5.info_set == "real_time") & (m5.model == m)].rmse.iloc[0]
    ep = m5[(m5.info_set == "ex_post") & (m5.model == m)].rmse.iloc[0]
    conf = m in ("ridge", "random_forest", "mlp")
    ax.scatter(rt, ep, s=45, marker="o" if conf else "^", color="tab:blue" if conf else "tab:orange")
    off = {"elastic_net": (6, -12), "ridge": (-34, 6)}.get(m, (5, 4))
    ax.annotate(NAMES[m] + ("" if conf else " (expl.)"), (rt, ep), textcoords="offset points", xytext=off, fontsize=8)
lo, hi = 2.6, 3.1
ax.plot([lo, hi], [lo, hi], "--", color="grey", lw=1)
ax.axvline(zero, color="black", lw=0.6, ls=":"); ax.axhline(zero, color="black", lw=0.6, ls=":")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("Real-time RMSE"); ax.set_ylabel("Ex-post RMSE")
ax.set_title("Five-session return, k = 10 (points below the diagonal favour ex post)", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig_20_information_sets_return_k10.png", dpi=200); plt.close(fig)
print("written:", OUT / "fig_16_pred_vs_realised_realtime_k10.png", OUT / "fig_20_information_sets_return_k10.png")