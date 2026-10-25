"""Emit results_to_artifact_map.csv — every number intended for the manuscript,
with its source artifact, column, and row filter.  On completion requirement step 3."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.artifact_store import artifact_file

OUT = ROOT / "results_to_artifact_map.csv"
rows = []


def row(section, label, value, artifact, column, row_filter=""):
    rows.append({
        "thesis_section": section,
        "label": label,
        "value": value,
        "source_artifact": artifact,
        "source_column": column,
        "row_filter": row_filter,
    })


# ── Event study ──────────────────────────────────────────────────────────────
es = pd.read_parquet(artifact_file("event_study_caar.parquet"))
ret5 = es[(es.series == "return") & (es.event_time == 5)].iloc[0]
vol20 = es[(es.series == "volume") & (es.event_time == 20)].iloc[0]

row("EventStudy", "return_CAAR_1_5", f"{ret5.caar:+.4f}", "event_study_caar.parquet", "caar",
    "series=return, event_time=5")
row("EventStudy", "return_p_bmp_k5", f"{ret5.p_bmp:.4f}", "event_study_caar.parquet", "p_bmp",
    "series=return, event_time=5")
row("EventStudy", "return_p_corrado_k5", f"{ret5.p_corrado:.4f}", "event_study_caar.parquet",
    "p_corrado", "series=return, event_time=5")
row("EventStudy", "return_p_kp_k5", f"{ret5.p_kolari_pynnonen:.4f}", "event_study_caar.parquet",
    "p_kolari_pynnonen", "series=return, event_time=5")
row("EventStudy", "volume_CAAR_1_20", f"{vol20.caar:+.4f}", "event_study_caar.parquet", "caar",
    "series=volume, event_time=20")
row("EventStudy", "volume_p_corrado_k20", f"{vol20.p_corrado:.4f}", "event_study_caar.parquet",
    "p_corrado", "series=volume, event_time=20")
row("EventStudy", "volume_p_kp_k20", f"{vol20.p_kolari_pynnonen:.4f}", "event_study_caar.parquet",
    "p_kolari_pynnonen", "series=volume, event_time=20")
row("EventStudy", "n_events", str(int(ret5.n_events)), "event_study_caar.parquet", "n_events",
    "series=return, event_time=5")

# ── Placebo ───────────────────────────────────────────────────────────────────
import json
prov = json.loads(artifact_file("placebo_event_study_provenance.json").read_text())
rej = prov["empirical_rejection_rate_5pct"]
row("Placebo", "reject_bmp_5pct", f"{rej['bmp']:.3f}",
    "placebo_event_study_provenance.json", "empirical_rejection_rate_5pct.bmp", "")
row("Placebo", "reject_corrado_5pct", f"{rej['corrado']:.3f}",
    "placebo_event_study_provenance.json", "empirical_rejection_rate_5pct.corrado", "")
row("Placebo", "reject_kp_5pct", f"{rej['kolari_pynnonen']:.3f}",
    "placebo_event_study_provenance.json", "empirical_rejection_rate_5pct.kolari_pynnonen", "")

# ── Y1 magnitude ──────────────────────────────────────────────────────────────
vt = pd.read_parquet(artifact_file("verdict_table.parquet"))

# Primary confirmatory baseline row (try aspi_grid_metrics if available with new cols)
try:
    gm = pd.read_parquet(artifact_file("aspi_grid_metrics.parquet"))
    has_new_cols = "rmse_improvement" in gm.columns
except Exception:
    gm = None
    has_new_cols = False

# Naive baselines (from aspi_grid_metrics or verdict_table)
for bname in ("naive_zero", "naive_train_mean"):
    vt_sub = vt[(vt.target == "Y1_ASPI_5D_Forward_LogReturn_Pct") &
                (vt.model == bname)].head(1)
    if len(vt_sub):
        r = vt_sub.iloc[0]
        row("Y1_magnitude", f"baseline_rmse_{bname}_h5",
            f"{r.rmse_model:.4f}", "verdict_table.parquet", "rmse_model",
            f"target=Y1_ASPI_5D_Forward_LogReturn_Pct, model={bname}")

# Best model from aspi_grid_metrics (confirmatory, h=5, k=10)
if gm is not None:
    conf = gm[(gm.horizon == 5) & (gm.info_set == "real_time") & (gm.k == 10) & gm.confirmatory]
    best = conf.sort_values("rmse").head(1)
    if len(best):
        b = best.iloc[0]
        row("Y1_magnitude", f"best_conf_rmse_h5_rt_{b['model']}",
            f"{b['rmse']:.4f}", "aspi_grid_metrics.parquet", "rmse",
            f"horizon=5, info_set=real_time, k=10, model={b['model']}, confirmatory=True")
        row("Y1_magnitude", f"best_conf_pooled_r2_h5_rt_{b['model']}",
            f"{b['pooled_r2']:.4f}", "aspi_grid_metrics.parquet", "pooled_r2",
            f"horizon=5, info_set=real_time, k=10, model={b['model']}, confirmatory=True")

# Holm significance count
conf_y1 = vt[(vt.target == "Y1_ASPI_5D_Forward_LogReturn_Pct") & vt.confirmatory]
n_sig = int(conf_y1.holm_significant.sum())
row("Y1_magnitude", "n_holm_significant_confirmatory_h5",
    str(n_sig), "verdict_table.parquet", "holm_significant",
    "target=Y1_ASPI_5D_Forward_LogReturn_Pct, confirmatory=True")

# ── Y1 direction ──────────────────────────────────────────────────────────────
dm = pd.read_parquet(artifact_file("aspi_direction_metrics.parquet"))
for h in (5, 10):
    for m in ("logistic", "random_forest"):
        sub = dm[(dm.horizon == h) & (dm.model == m)]
        if not len(sub):
            continue
        r = sub.iloc[0]
        row("Y1_direction", f"auc_h{h}_{m}", f"{r.roc_auc:.4f}",
            "aspi_direction_metrics.parquet", "roc_auc", f"horizon={h}, model={m}")
        row("Y1_direction", f"auc_ci_low_h{h}_{m}", f"{r.auc_ci_low:.4f}",
            "aspi_direction_metrics.parquet", "auc_ci_low", f"horizon={h}, model={m}")
        row("Y1_direction", f"auc_ci_high_h{h}_{m}", f"{r.auc_ci_high:.4f}",
            "aspi_direction_metrics.parquet", "auc_ci_high", f"horizon={h}, model={m}")
        row("Y1_direction", f"p_holm_h{h}_{m}", f"{r.p_holm:.4f}",
            "aspi_direction_metrics.parquet", "p_holm", f"horizon={h}, model={m}")
        row("Y1_direction", f"holm_significant_h{h}_{m}", str(bool(r.holm_significant)),
            "aspi_direction_metrics.parquet", "holm_significant", f"horizon={h}, model={m}")
        if m == "logistic":
            row("Y1_direction", f"balanced_accuracy_h{h}_{m}",
                f"{r.balanced_accuracy:.4f}", "aspi_direction_metrics.parquet",
                "balanced_accuracy", f"horizon={h}, model={m}")
            row("Y1_direction", f"mcc_h{h}_{m}", f"{r.mcc:.4f}",
                "aspi_direction_metrics.parquet", "mcc", f"horizon={h}, model={m}")

# ── Y2 volume ─────────────────────────────────────────────────────────────────
y2_vt = vt[(vt.target == "Y2_5D_Forward_AbnormalVolume_LogRatio")]
rf_zero = y2_vt[(y2_vt.model == "random_forest") & (y2_vt.baseline == "naive_zero")]
if len(rf_zero):
    r = rf_zero.iloc[0]
    row("Y2_volume", "rf_rmse", f"{r.rmse_model:.4f}", "verdict_table.parquet", "rmse_model",
        "target=Y2, model=random_forest, baseline=naive_zero")
    row("Y2_volume", "rf_delta_rmse_vs_zero", f"{r.delta_rmse:.4f}",
        "verdict_table.parquet", "delta_rmse",
        "target=Y2, model=random_forest, baseline=naive_zero")
    row("Y2_volume", "rf_ci_low_vs_zero", f"{r.ci_low:.4f}",
        "verdict_table.parquet", "ci_low", "target=Y2, model=random_forest, baseline=naive_zero")
    row("Y2_volume", "rf_ci_high_vs_zero", f"{r.ci_high:.4f}",
        "verdict_table.parquet", "ci_high", "target=Y2, model=random_forest, baseline=naive_zero")
    row("Y2_volume", "rf_p_holm_vs_zero", f"{r.p_holm:.4f}",
        "verdict_table.parquet", "p_holm", "target=Y2, model=random_forest, baseline=naive_zero")
    row("Y2_volume", "rf_holm_significant", str(bool(r.holm_significant)),
        "verdict_table.parquet", "holm_significant",
        "target=Y2, model=random_forest, baseline=naive_zero")

# pooled R²
y2_models = vt[(vt.target == "Y2_5D_Forward_AbnormalVolume_LogRatio") &
               (vt.model.isin(["random_forest", "ensemble"]))].drop_duplicates("model")
for _, r in y2_models.iterrows():
    # pooled_r2 not in verdict_table directly; note source
    row("Y2_volume", f"pooled_r2_note_{r['model']}",
        "see aspi_grid_metrics or results_regression.pkl",
        "verdict_table.parquet / aspi_grid_metrics.parquet", "pooled_r2",
        f"target=Y2, model={r['model']}")

# ── Y3 recovery ───────────────────────────────────────────────────────────────
rg = pd.read_parquet(artifact_file("recovery_grid_metrics.parquet"))
ds = pd.read_parquet(artifact_file("dataset.parquet"))
n_total = len(ds)
n_observed = int(ds["Y3_event_observed"].sum()) if "Y3_event_observed" in ds.columns else None
n_censored = int(ds["Y3_censored"].sum()) if "Y3_censored" in ds.columns else None

row("Y3_recovery", "n_total_events", str(n_total), "dataset.parquet", "Y3_ASPI_Recovery_Time",
    "all rows")
if n_observed:
    row("Y3_recovery", "n_observed_recoveries", str(n_observed),
        "dataset.parquet", "Y3_event_observed", "Y3_event_observed=True")
if n_censored:
    row("Y3_recovery", "n_censored", str(n_censored),
        "dataset.parquet", "Y3_censored", "Y3_censored=True")

# Best survival model
best_surv = rg.sort_values("c_index", ascending=False).head(1).iloc[0]
row("Y3_recovery", f"best_c_index_{best_surv['model']}",
    f"{best_surv.c_index:.4f}", "recovery_grid_metrics.parquet", "c_index",
    f"model={best_surv['model']}")
row("Y3_recovery", f"best_c_ci_low_{best_surv['model']}",
    f"{best_surv.c_index_ci_low:.4f}", "recovery_grid_metrics.parquet", "c_index_ci_low",
    f"model={best_surv['model']}")
row("Y3_recovery", f"best_c_ci_high_{best_surv['model']}",
    f"{best_surv.c_index_ci_high:.4f}", "recovery_grid_metrics.parquet", "c_index_ci_high",
    f"model={best_surv['model']}")
row("Y3_recovery", "n_holm_significant_survival",
    str(int(rg.holm_significant.sum())), "recovery_grid_metrics.parquet", "holm_significant", "")
row("Y3_recovery", "baseline_naive_zero_excluded",
    "True — zero sessions is not a valid recovery duration",
    "src/evaluation/verification.py", "DEFAULT_BASELINES", "Y3_ASPI_Recovery_Time")

# KM baseline c-index
km_row = rg[rg.model == "km_train_baseline"]
if len(km_row):
    k = km_row.iloc[0]
    row("Y3_recovery", "km_baseline_c_index", f"{k.c_index:.4f}",
        "recovery_grid_metrics.parquet", "c_index", "model=km_train_baseline")

# ── CIF baseline ─────────────────────────────────────────────────────────────
cif = pd.read_parquet(artifact_file("recovery_cif_baseline.parquet"))
for _, r in cif.iterrows():
    row("Y3_recovery", f"cif_at_t{r.horizon}", f"{r.cif_estimate:.4f}",
        "recovery_cif_baseline.parquet", "cif_estimate", f"horizon={r.horizon}")
    row("Y3_recovery", f"cif_ci_low_t{r.horizon}", f"{r.ci_low:.4f}",
        "recovery_cif_baseline.parquet", "ci_low", f"horizon={r.horizon}")

# ── Equivalence test ──────────────────────────────────────────────────────────
eq = pd.read_parquet(artifact_file("equivalence_test.parquet"))
for _, r in eq.iterrows():
    tname = str(r.get("target", ""))
    row("InfoSetEquivalence", f"delta_rmse_{tname}", f"{r.delta_rmse:.4f}",
        "equivalence_test.parquet", "delta_rmse", f"target={tname}")
    row("InfoSetEquivalence", f"equiv_supported_{tname}", str(bool(r.equivalence_supported)),
        "equivalence_test.parquet", "equivalence_supported", f"target={tname}")

# ── MDE ───────────────────────────────────────────────────────────────────────
mde = pd.read_parquet(artifact_file("minimum_detectable_effect.parquet"))
for _, r in mde.iterrows():
    tname = str(r.get("target", ""))
    row("PowerMDE", f"mde_rmse_{tname}", f"{r.get('mde_rmse', 'N/A'):.4f}",
        "minimum_detectable_effect.parquet", "mde_rmse", f"target={tname}")

# ── Sample counts ─────────────────────────────────────────────────────────────
sa = pd.read_parquet(artifact_file("sample_accounting.parquet"))
for _, r in sa.iterrows():
    tname = str(r.get("target", ""))
    var = str(r.get("variant", ""))
    row("Sampling", f"n_labelled_{tname}_{var}", str(int(r.get("n_labelled", 0))),
        "sample_accounting.parquet", "n_labelled", f"target={tname}, variant={var}")
    row("Sampling", f"n_held_out_{tname}_{var}", str(int(r.get("n_held_out", 0))),
        "sample_accounting.parquet", "n_held_out", f"target={tname}, variant={var}")

# ── Holm family sizes ─────────────────────────────────────────────────────────
am = pd.read_parquet(artifact_file("analysis_map.parquet"))
for target, grp in am[am.tier == "confirmatory"].groupby("target"):
    row("HolmCorrection", f"family_size_{target}",
        str(int(grp.family_size.max() if "family_size" in grp.columns else len(grp))),
        "analysis_map.parquet", "family_size", f"target={target}, tier=confirmatory")

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print(f"wrote {OUT.relative_to(ROOT)}  ({len(out)} rows)")
