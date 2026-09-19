"""Smoke tests for the EDA figure suite."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")          # headless: no Tk, so the suite runs in CI and over ssh

import numpy as np
import pandas as pd
import pytest

from src.visualization import eda_figures as eda
from src.visualization import result_figures as fx


@pytest.fixture(autouse=True)
def _tmp_figs(tmp_path, monkeypatch):
    fx.set_figure_dir(tmp_path)
    monkeypatch.setattr(eda, "save_figure",
                        lambda fig, name, **k: fx.save_figure(fig, name, verbose=False, **k))
    yield tmp_path


@pytest.fixture
def dataset():
    rng = np.random.default_rng(0)
    n = 60
    return pd.DataFrame({
        "event_date": pd.date_range("2005-01-01", periods=n, freq="90D"),
        "Y1_ASPI_5D_Forward_LogReturn_Pct": rng.normal(0, 0.014, n),
        "Y2_5D_Forward_AbnormalVolume_LogRatio": rng.normal(0, 0.5, n),
        "Y3_ASPI_Recovery_Time": rng.integers(0, 90, n).astype(float),
        "financial_damage": np.where(rng.random(n) < 0.7, 0.0, rng.exponential(1e6, n)),
        "log_population_affected": rng.normal(10, 2, n),
        "di_available": (rng.random(n) > 0.3).astype(float),
        "di_affected_log": rng.normal(9, 2, n),
        "damage_source": np.where(rng.random(n) < 0.7, "missing_zero_filled",
                                  "emdat_cpi_adjusted"),
        "hz_precip_max3d": rng.gamma(2, 20, n),
    })


def test_missingness_matrix_separates_zero_filled_from_observed(dataset, _tmp_figs):
    """The central claim of the figure: a zero-filled damage value is NOT 'observed'."""
    fig, state = eda.plot_missingness_matrix(dataset)
    assert (_tmp_figs / "eda_01_missingness_matrix.png").exists()
    imputed = dataset.damage_source.str.contains("missing")
    # Every zero-filled damage row must be state 1, not state 0.
    assert (state.loc[imputed, "financial_damage"] == 1.0).all()
    assert (state.loc[~imputed, "financial_damage"] == 0.0).all()


def test_missingness_ranked_writes_and_orders_worst_first(dataset, _tmp_figs):
    fig, frac = eda.plot_missingness_ranked(dataset)
    assert (_tmp_figs / "eda_02_missingness_ranked.png").exists()
    assert frac.is_monotonic_increasing   # worst completeness first


def test_outlier_table_flags_a_planted_extreme():
    s = pd.Series(list(np.zeros(50)) + [1000.0])
    info = eda.outlier_table(s)
    assert 1000.0 in info["iqr_outliers"].to_numpy()
    assert info["n"] == 51


def test_outlier_table_finds_nothing_in_clean_data():
    s = pd.Series(np.linspace(0, 1, 100))
    assert len(eda.outlier_table(s)["iqr_outliers"]) == 0


def test_outlier_panel_writes_and_summarises(dataset, _tmp_figs):
    fig, summary = eda.plot_outlier_panel(
        dataset, ["Y1_ASPI_5D_Forward_LogReturn_Pct", "Y2_5D_Forward_AbnormalVolume_LogRatio", "Y3_ASPI_Recovery_Time"])
    assert (_tmp_figs / "eda_03_outliers.png").exists()
    assert set(summary.variable) == {"Y1_ASPI_5D_Forward_LogReturn_Pct", "Y2_5D_Forward_AbnormalVolume_LogRatio",
                                     "Y3_ASPI_Recovery_Time"}
    assert (summary.iqr_hi >= summary.iqr_lo).all()


def test_feature_distributions_reports_shape_statistics(dataset, _tmp_figs):
    fig, stats_frame = eda.plot_feature_distributions(
        dataset, ["hz_precip_max3d", "log_population_affected", "Y1_ASPI_5D_Forward_LogReturn_Pct"])
    assert (_tmp_figs / "eda_04_feature_distributions.png").exists()
    assert {"skew", "excess_kurtosis", "median"} <= set(stats_frame.columns)
    # A gamma draw must register positive skew; the test would pass on any number
    # without this, so it pins the statistic rather than its presence.
    assert stats_frame.set_index("variable").loc["hz_precip_max3d", "skew"] > 0


def test_qq_grid_writes_and_reports_shapiro(dataset, _tmp_figs):
    fig, frame = eda.plot_qq_grid(dataset, ["Y1_ASPI_5D_Forward_LogReturn_Pct", "Y3_ASPI_Recovery_Time"])
    assert (_tmp_figs / "eda_05_qq_targets.png").exists()
    assert frame.shapiro_p.between(0, 1).all()


def test_scatter_matrix_writes_and_reports_critical_r(dataset, _tmp_figs):
    fig, frame = eda.plot_target_scatter_matrix(
        dataset, ["Y1_ASPI_5D_Forward_LogReturn_Pct", "Y2_5D_Forward_AbnormalVolume_LogRatio"],
        ["log_population_affected", "hz_precip_max3d"])
    assert (_tmp_figs / "eda_06_scatter_matrix.png").exists()
    assert (frame.critical_r > 0).all()
    # `distinguishable` must agree with the critical value it is compared against.
    assert (frame.distinguishable == (frame.spearman_rho.abs() >= frame.critical_r)).all()


def test_feature_target_correlation_writes(dataset, _tmp_figs):
    fig, frame = eda.plot_feature_target_correlation(
        dataset, "Y1_ASPI_5D_Forward_LogReturn_Pct",
        ["log_population_affected", "hz_precip_max3d", "di_affected_log"])
    assert (_tmp_figs / "eda_07_corr_Y1_ASPI_5D_Forward_LogReturn_Pct.png").exists()
    assert frame.abs_rho.is_monotonic_decreasing


def test_event_timeline_writes(dataset, _tmp_figs):
    days = pd.date_range("2005-01-01", "2020-12-31", freq="B")
    market = pd.DataFrame({"date": days,
                           "aspi_close": np.linspace(1000, 8000, len(days))})
    eda.plot_event_timeline(market, dataset)
    assert (_tmp_figs / "eda_08_event_timeline.png").exists()


def test_class_balance_writes(_tmp_figs):
    frame = pd.DataFrame({"label": ["C1", "C2", "C3"], "prevalence": [0.5, 0.44, 0.90]})
    eda.plot_class_balance(frame)
    assert (_tmp_figs / "eda_09_class_balance.png").exists()


def test_sector_coverage_writes_and_counts(_tmp_figs):
    days = pd.date_range("2005-01-01", "2020-12-31", freq="B")
    long = pd.concat([
        pd.DataFrame({"date": days, "sector": "Banks", "close": 1.0}),
        # A short sector must still appear, with its own shorter span.
        pd.DataFrame({"date": days[500:], "sector": "IT", "close": 1.0}),
    ])
    fig, g = eda.plot_sector_coverage(long)
    assert (_tmp_figs / "eda_10_sector_coverage.png").exists()
    assert set(g.sector) == {"Banks", "IT"}
    assert g.set_index("sector").loc["IT", "min"] > g.set_index("sector").loc["Banks", "min"]
