# Current Numbers

Canonical values as of v1.3-revision5. Every number comes from a script run against
the artifacts; sources noted for each.

## Confirmatory family

| Item | Value | Source |
|------|-------|--------|
| Family size | 24 comparisons | verdict_table.parquet, confirmatory==True |
| (original protocol-frozen) | 18 | change_history.md Change 1 |
| Holm α | 0.05 | docs/audit.md Part 8.4 |

## Target 1 — ASPI return magnitude (Y1 5D)

| Item | Value | Source |
|------|-------|--------|
| Labelled events | 74 | sample_accounting.parquet, Y1_ASPI_5D |
| Held-out test events | 40 | sample_accounting.parquet |
| n_purged | 1 | sample_accounting.parquet |
| n_unevaluated_tail | 4 | sample_accounting.parquet |
| Holm result | All 24 intervals include zero; not rejected | verdict_table.parquet |
| Smallest Holm p | 0.641 | verdict_table.parquet |

## Target 1 (secondary) — Return direction (Y1 10D logistic)

| Item | Value | Source |
|------|-------|--------|
| Test n | 40 | aspi_direction_metrics.parquet |
| ROC AUC | 0.817 | aspi_direction_metrics.parquet |
| AUC 95% CI | [0.657, 0.940] | aspi_direction_metrics.parquet |
| Permutation p (B=5000, episode-level, fixed predictions) | 0.0004 | aspi_direction_permutation.parquet |

## Target 2 — Abnormal volume (Y2 5D)

| Item | Value | Source |
|------|-------|--------|
| Labelled events | 61 | sample_accounting.parquet, Y2_5D |
| Held-out test events | 34 | sample_accounting.parquet |
| n_purged | 1 | sample_accounting.parquet |
| RF ΔRMSE vs naive_zero | +0.106 | verdict_table.parquet |
| RF 95% CI | [+0.037, +0.175] | verdict_table.parquet |
| RF Holm p (24-comparison family) | 0.024 | verdict_table.parquet |
| RF holm_significant | True | verdict_table.parquet |
| Caveat | Family enlarged from 18→24 during examiner revision | docs/interpretation.md |

## Target 3 — Recovery duration (Y3)

| Item | Value | Source |
|------|-------|--------|
| Labelled events | 74 | sample_accounting.parquet, Y3_recovery_time |
| Held-out test events | 40 | sample_accounting.parquet |
| n_purged | 2 | sample_accounting.parquet |
| n_unevaluated_tail | 4 | sample_accounting.parquet |
| Best C-index | 0.535 | recovery_grid_metrics.parquet |
| Best C-index CI | [0.371, 0.697] | recovery_grid_metrics.parquet |
| Beats 0.5 | No | recovery_grid_metrics.parquet |

## Episode-level permutation test (direction, all 10 comparisons)

| horizon | model | AUC | p | primary |
|---------|-------|-----|---|---------|
| 1 | logistic | 0.587 | 0.182 | No |
| 1 | random_forest | 0.613 | 0.126 | No |
| 5 | logistic | 0.574 | 0.204 | No |
| 5 | random_forest | 0.544 | 0.320 | No |
| 10 | logistic | **0.817** | **0.0004** | **Yes** |
| 10 | random_forest | 0.709 | 0.017 | No |
| 15 | logistic | 0.694 | 0.022 | No |
| 15 | random_forest | 0.549 | 0.303 | No |
| 20 | logistic | 0.638 | 0.082 | No |
| 20 | random_forest | 0.585 | 0.184 | No |

Source: aspi_direction_permutation.parquet (B=5000, seed=42, fixed-prediction permutation).
Full-refit permutation p-value: see aspi_direction_permutation_refit.parquet (task A1).

## MDE (minimum detectable effect)

Source: minimum_detectable_effect.parquet, scripts/build_mde.py.
Dependence assumption: i.i.d. pooled test observations; formula uses raw n, not cluster-adjusted.

## Episode-clustered CI (RT vs EP information sets, Y2 primary horizon)

| Comparison | ΔRMSE | 95% CI | Source |
|------------|-------|--------|--------|
| RT vs EP | 0.133 | [-0.074, 0.309] | robustness_suite.parquet (RT_vs_EP row) |

Equivalence not supported; CI includes zero and positive values.
