"""Build docs/manuscript_numbers.json — every number quoted in manuscript
tables/text that depends on artifacts.

Each entry has: value, artifact, column (or note), table/location reference.
Run from repo root:
    python scripts/build_manuscript_numbers.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.artifact_store import artifact_file

OUT = ROOT / "docs" / "manuscript_numbers.json"


def _r(value, artifact: str, column: str, location: str) -> dict:
    """One number record."""
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    return {"value": value, "artifact": artifact, "column": column,
            "location": location}


def main() -> None:
    sa = pd.read_parquet(artifact_file("sample_accounting.parquet"))
    vt = pd.read_parquet(artifact_file("verdict_table.parquet"))
    dm = pd.read_parquet(artifact_file("aspi_direction_metrics.parquet"))
    dp = pd.read_parquet(artifact_file("aspi_direction_permutation.parquet"))
    es = pd.read_parquet(artifact_file("event_study_caar.parquet"))
    rg = pd.read_parquet(artifact_file("recovery_grid_metrics.parquet"))
    mde = pd.read_parquet(artifact_file("minimum_detectable_effect.parquet"))
    rs = pd.read_parquet(artifact_file("robustness_suite.parquet"))

    nums: list[dict] = []

    # ── Sample accounting (Tables 4, 7, 8, 8a) ──────────────────────────────
    sa_art = "sample_accounting.parquet"
    y1_5d = sa[sa.target == "Y1_ASPI_5D"].iloc[0]
    y1_10d = sa[sa.target == "Y1_ASPI_10D"].iloc[0]
    y2 = sa[sa.target == "Y2_5D"].iloc[0]
    y3 = sa[(sa.target == "Y3_recovery_time") & (sa.variant == "main")].iloc[0]

    nums += [
        _r(74, sa_art, "n_events_total", "abstract / Table 4: total qualifying disasters"),
        _r(int(y1_5d.n_labelled), sa_art, "n_labelled[Y1_ASPI_5D]", "Table 4: Y1 labelled events"),
        _r(int(y1_5d.n_held_out), sa_art, "n_held_out[Y1_ASPI_5D]", "Table 4: Y1 held-out events"),
        _r(int(y1_5d.n_purged), sa_art, "n_purged[Y1_ASPI_5D]", "Table 4: Y1 purged"),
        _r(int(y1_5d.n_unevaluated_tail), sa_art, "n_unevaluated_tail[Y1_ASPI_5D]", "Table 4: Y1 tail"),
        _r(int(y2.n_labelled), sa_art, "n_labelled[Y2_5D]", "Table 4: Y2 labelled events"),
        _r(int(y2.n_held_out), sa_art, "n_held_out[Y2_5D]", "Table 4: Y2 held-out events"),
        _r(int(y2.n_purged), sa_art, "n_purged[Y2_5D]", "Table 4: Y2 purged"),
        _r(int(y3.n_labelled), sa_art, "n_labelled[Y3_recovery_time]", "Table 4: Y3 labelled events"),
        _r(int(y3.n_held_out), sa_art, "n_held_out[Y3_recovery_time]", "Table 4: Y3 held-out events"),
        _r(int(y3.n_purged), sa_art, "n_purged[Y3_recovery_time]", "Table 4: Y3 purged"),
        _r(int(y3.n_unevaluated_tail), sa_art, "n_unevaluated_tail[Y3_recovery_time]", "Table 4: Y3 tail"),
    ]

    # ── Confirmatory family (Table 9) ────────────────────────────────────────
    vt_art = "verdict_table.parquet"
    conf = vt[vt.confirmatory == True]
    nums += [
        _r(len(conf), vt_art, "count(confirmatory==True)", "Table 9: confirmatory family size (24)"),
        _r(0.05, vt_art, "Holm alpha (protocol)", "Table 9 / stats guide: Holm α"),
    ]

    # Smallest Holm p in Y1 family
    y1_conf = conf[conf.target.str.contains("Y1")]
    y1_min_holm = float(y1_conf.p_holm.min())
    nums.append(_r(round(y1_min_holm, 3), vt_art, "p_holm.min()[Y1 confirmatory]",
                   "Table 9: Y1 smallest Holm p"))

    # Y2 primary result
    y2_rf_zero = conf[(conf.target.str.contains("Y2")) &
                      (conf.model == "random_forest") &
                      (conf.baseline == "naive_zero")].iloc[0]
    nums += [
        _r(round(float(y2_rf_zero.delta_rmse), 3), vt_art,
           "delta_rmse[Y2,RF,naive_zero,confirmatory]", "Table 9 / abstract: Y2 RF ΔRMSE"),
        _r(round(float(y2_rf_zero.ci_low), 3), vt_art,
           "ci_low[Y2,RF,naive_zero,confirmatory]", "Table 9: Y2 RF CI lower"),
        _r(round(float(y2_rf_zero.ci_high), 3), vt_art,
           "ci_high[Y2,RF,naive_zero,confirmatory]", "Table 9: Y2 RF CI upper"),
        _r(round(float(y2_rf_zero.p_holm), 4), vt_art,
           "p_holm[Y2,RF,naive_zero,confirmatory]", "Table 9 / stats guide: Y2 RF Holm p"),
        _r(bool(y2_rf_zero.holm_significant), vt_art,
           "holm_significant[Y2,RF,naive_zero,confirmatory]", "Table 9: Y2 Holm significant"),
    ]

    # ── Direction classifier (Table 10) ────────────────────────────────────
    dm_art = "aspi_direction_metrics.parquet"
    h10_log = dm[(dm.horizon == 10) & (dm.model == "logistic")].iloc[0]
    nums += [
        _r(round(float(h10_log.roc_auc), 4), dm_art,
           "roc_auc[h=10,logistic]", "Table 10: logistic h=10 AUC"),
        _r(round(float(h10_log.auc_ci_low), 3), dm_art,
           "auc_ci_low[h=10,logistic]", "Table 10: logistic h=10 AUC CI lower"),
        _r(round(float(h10_log.auc_ci_high), 3), dm_art,
           "auc_ci_high[h=10,logistic]", "Table 10: logistic h=10 AUC CI upper"),
        _r(round(float(h10_log.balanced_accuracy), 4), dm_art,
           "balanced_accuracy[h=10,logistic]", "Table 10: logistic h=10 balanced accuracy"),
        _r(round(float(h10_log.mcc), 4), dm_art,
           "mcc[h=10,logistic]", "Table 10: logistic h=10 MCC"),
        _r(int(h10_log.n), dm_art, "n[h=10,logistic]", "Table 10: test n"),
    ]

    # ── Permutation test (Tables 16a, 17, text) ────────────────────────────
    dp_art = "aspi_direction_permutation.parquet"
    prim = dp[dp.primary == True].iloc[0]
    nums += [
        _r(round(float(prim.observed_auc), 4), dp_art,
           "observed_auc[primary]", "Tables 16a/17: primary permutation AUC"),
        _r(round(float(prim.p_value), 4), dp_art,
           "p_value[primary]", "Tables 16a/17: primary permutation p"),
        _r(int(prim.n_permutations), dp_art,
           "n_permutations[primary]", "Tables 16a/17: B"),
        _r(round(float(prim.perm_auc_mean), 4), dp_art,
           "perm_auc_mean[primary]", "Table 16a: permutation AUC mean"),
        _r(round(float(prim.perm_auc_p95), 4), dp_art,
           "perm_auc_p95[primary]", "Table 16a: permutation AUC 95th pct"),
    ]

    # All 10 permutation rows (K1-K11 or similar appendix tables)
    for _, row in dp.iterrows():
        nums.append(_r(round(float(row.p_value), 4), dp_art,
                       f"p_value[h={row.horizon},{row.model}]",
                       f"Appendix K: permutation p h={row.horizon} {row.model}"))
        nums.append(_r(round(float(row.observed_auc), 4), dp_art,
                       f"observed_auc[h={row.horizon},{row.model}]",
                       f"Appendix K: AUC h={row.horizon} {row.model}"))

    # ── Event study / correlations (Table 7) ───────────────────────────────
    es_art = "event_study_caar.parquet"
    ret5 = es[(es.series == "return") & (es.event_time == 5)].iloc[0]
    vol5 = es[(es.series == "volume") & (es.event_time == 5)].iloc[0]
    nums += [
        _r(round(float(ret5.caar), 4), es_art,
           "caar[return,k=5]", "Table 7: CAAR return k=5"),
        _r(round(float(ret5.ci_low), 4), es_art,
           "ci_low[return,k=5]", "Table 7: CAAR return CI lower"),
        _r(round(float(ret5.ci_high), 4), es_art,
           "ci_high[return,k=5]", "Table 7: CAAR return CI upper"),
        _r(round(float(ret5.t_cross_sectional), 3), es_art,
           "t_cross_sectional[return,k=5]", "Table 7: return t_cross"),
        _r(round(float(ret5.p_cross_sectional), 4), es_art,
           "p_cross_sectional[return,k=5]", "Table 7: return p_cross"),
        _r(round(float(ret5.t_bmp), 3), es_art,
           "t_bmp[return,k=5]", "Table 7: return t_BMP"),
        _r(round(float(ret5.t_corrado), 3), es_art,
           "t_corrado[return,k=5]", "Table 7: return t_Corrado"),
        _r(round(float(ret5.mean_cross_correlation), 4), es_art,
           "mean_cross_correlation[return]", "Table 7: return mean pairwise r"),
        _r(round(float(vol5.caar), 4), es_art,
           "caar[volume,k=5]", "Table 7: CAAR volume k=5"),
        _r(round(float(vol5.ci_low), 4), es_art,
           "ci_low[volume,k=5]", "Table 7: CAAR volume CI lower"),
        _r(round(float(vol5.ci_high), 4), es_art,
           "ci_high[volume,k=5]", "Table 7: CAAR volume CI upper"),
        _r(round(float(vol5.t_corrado), 3), es_art,
           "t_corrado[volume,k=5]", "Table 7: volume t_Corrado"),
        _r(round(float(vol5.p_corrado), 4), es_art,
           "p_corrado[volume,k=5]", "Table 7: volume p_Corrado"),
        _r(round(float(vol5.mean_cross_correlation), 4), es_art,
           "mean_cross_correlation[volume]", "Table 7: volume mean pairwise r"),
    ]

    # ── Recovery (Table 8 / 8a) ─────────────────────────────────────────────
    rg_art = "recovery_grid_metrics.parquet"
    best = rg.sort_values("c_index", ascending=False).iloc[0]
    km = rg[rg.model == "km_train_baseline"].iloc[0]
    nums += [
        _r(round(float(best.c_index), 3), rg_art,
           "c_index[best model]", "Table 8: best model C-index"),
        _r(round(float(best.c_index_ci_low), 3), rg_art,
           "c_index_ci_low[best model]", "Table 8: best model C-index CI lower"),
        _r(round(float(best.c_index_ci_high), 3), rg_art,
           "c_index_ci_high[best model]", "Table 8: best model C-index CI upper"),
        _r(str(best.model), rg_art, "model[best c_index]", "Table 8: best model name"),
        _r(round(float(km.c_index), 3), rg_art,
           "c_index[km_train_baseline]", "Table 8: KM baseline C-index"),
        _r(round(float(best.integrated_brier_score), 3), rg_art,
           "integrated_brier_score[best model]", "Table 8a: best model IBS"),
        _r(round(float(km.integrated_brier_score), 3), rg_art,
           "integrated_brier_score[km_train_baseline]", "Table 8a: KM baseline IBS"),
    ]

    # Individual Brier scores at each horizon
    for t_col in ["brier_t10", "brier_t20", "brier_t30", "brier_t60", "brier_t90"]:
        h = t_col.split("t")[1]
        nums.append(_r(round(float(best[t_col]), 3), rg_art,
                       f"{t_col}[best model]", f"Table 8a: Brier score t={h}"))
        nums.append(_r(round(float(km[t_col]), 3), rg_art,
                       f"{t_col}[km_train_baseline]", f"Table 8a: KM Brier t={h}"))

    # ── MDE (Table 17 / appendix) ───────────────────────────────────────────
    mde_art = "minimum_detectable_effect.parquet"
    for _, row in mde.iterrows():
        nums.append(_r(round(float(row.mde_rmse), 3), mde_art,
                       f"mde_rmse[{row.target}]", f"Table 17 / appendix: MDE {row.target}"))

    # ── RT vs EP comparison (from docs/audit.md narrative; no artifact row) ─
    nums.append({
        "value": 0.133,
        "artifact": "docs/audit.md (narrative, section 3771)",
        "column": "ex_post - real_time RMSE delta, h=5, k=10",
        "location": "text: RT vs EP equivalence test ΔRMSE",
        "note": ("CURRENT_NUMBERS.md cites robustness_suite.parquet (RT_vs_EP row) "
                 "but no such row exists. Value sourced from docs/audit.md. "
                 "robustness_suite.parquet has separate real_time and ex_post rows."),
    })
    nums.append({
        "value": [-0.074, 0.309],
        "artifact": "docs/audit.md (narrative, section 3771)",
        "column": "CI[ex_post - real_time RMSE delta]",
        "location": "text: RT vs EP equivalence test CI",
        "note": "Same provenance caveat as ΔRMSE above.",
    })

    OUT.write_text(json.dumps({"generated_by": "scripts/build_manuscript_numbers.py",
                               "numbers": nums}, indent=2))
    print(f"wrote {len(nums)} entries to {OUT.name}")


if __name__ == "__main__":
    main()
