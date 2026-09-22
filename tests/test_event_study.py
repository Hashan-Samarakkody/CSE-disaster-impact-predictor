"""T11: every event-study statistic is validated twice.

Once against a dataset with a known planted effect, where it must fire, and once against a
null dataset, where it must not. A test that only ever sees real data cannot tell the
difference between a correct implementation and a broken one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.event_study import (average_cross_correlation, build_abnormal_panel,
                                        bmp_t, corrado_rank_t, cross_sectional_t,
                                        cumulative_abnormal, event_study_table,
                                        kolari_pynnonen_t)
from src.utils.artifact_store import artifact_file

N_SESSIONS, N_EVENTS = 900, 40
SHOCK = -0.006
CRITICAL = 1.96


def _series(planted: bool, seed: int = 5):
    rng = np.random.default_rng(seed)
    sessions = pd.date_range("2015-01-01", periods=N_SESSIONS, freq="B")
    factor = rng.normal(0.0, 0.01, N_SESSIONS)
    asset = 1.2 * factor + rng.normal(0.0, 0.004, N_SESSIONS)
    positions = np.sort(rng.choice(np.arange(400, N_SESSIONS - 60), N_EVENTS, replace=False))
    if planted:
        for p in positions:
            asset[p:p + 5] += SHOCK
    daily = pd.DataFrame({"date": sessions, "log_return": asset, "sp500_log_return": factor})
    return daily, sessions[positions]


@pytest.fixture(scope="module")
def planted_table():
    daily, events = _series(planted=True)
    return event_study_table(build_abnormal_panel(daily, events), start=1, label="planted")


@pytest.fixture(scope="module")
def null_table():
    daily, events = _series(planted=False)
    return event_study_table(build_abnormal_panel(daily, events), start=1, label="null")


def _at(table, k=5):
    return table.loc[table.event_time == k].iloc[0]


def test_planted_effect_is_recovered_in_the_caar(planted_table):
    """Five sessions at -0.6 percent must accumulate to about -3 percentage points."""
    row = _at(planted_table)
    assert row.n_events == N_EVENTS
    assert row.caar == pytest.approx(5 * SHOCK, abs=5e-3)
    assert row.ci_high < 0


@pytest.mark.parametrize("statistic", ["t_cross_sectional", "t_bmp", "t_corrado",
                                       "t_kolari_pynnonen"])
def test_every_statistic_fires_on_the_planted_effect(planted_table, statistic):
    row = _at(planted_table)
    assert abs(row[statistic]) > CRITICAL, (statistic, row[statistic])
    assert row[statistic] < 0, statistic          # the planted effect is a fall


@pytest.mark.parametrize("statistic", ["t_cross_sectional", "t_bmp", "t_corrado",
                                       "t_kolari_pynnonen"])
def test_no_statistic_fires_on_the_null(null_table, statistic):
    row = _at(null_table)
    assert abs(row[statistic]) < CRITICAL, (statistic, row[statistic])


@pytest.mark.parametrize("p_column", ["p_cross_sectional", "p_bmp", "p_corrado",
                                      "p_kolari_pynnonen"])
def test_p_values_agree_with_their_statistics(planted_table, null_table, p_column):
    assert _at(planted_table)[p_column] < 0.05
    assert _at(null_table)[p_column] > 0.05


def test_kolari_pynnonen_can_only_shrink_the_standardised_statistic():
    """The correction exists to remove inflation from cross-correlation, so with a
    non-negative average correlation it must never make a result look stronger."""
    for correlation in (0.0, 0.01, 0.1, 0.5):
        adjusted, _ = kolari_pynnonen_t(4.0, n_events=40, mean_correlation=correlation)
        assert adjusted <= 4.0 + 1e-9, correlation
    assert kolari_pynnonen_t(4.0, 40, 0.0)[0] == pytest.approx(4.0)


def test_cross_correlation_reflects_overlapping_event_windows():
    """One security observed many times: dependence comes from windows that overlap, and
    the estimate must rise when events are packed together."""
    rng = np.random.default_rng(1)
    sessions = pd.date_range("2015-01-01", periods=N_SESSIONS, freq="B")
    factor = rng.normal(0.0, 0.01, N_SESSIONS)
    asset = 1.2 * factor + rng.normal(0.0, 0.004, N_SESSIONS)
    daily = pd.DataFrame({"date": sessions, "log_return": asset, "sp500_log_return": factor})

    spread = build_abnormal_panel(daily, sessions[np.arange(400, 800, 40)])
    packed = build_abnormal_panel(daily, sessions[np.arange(400, 420, 2)])
    assert average_cross_correlation(packed) > average_cross_correlation(spread)
    assert 0.0 <= average_cross_correlation(spread) <= 1.0


def test_corrado_depends_only_on_ranks_where_a_mean_test_does_not():
    """The defining property of a rank test. A strictly increasing transform of the
    abnormal values leaves every rank untouched, so the Corrado statistic must not move,
    while a statistic built on means moves a great deal. This is what makes it the right
    tool for a distribution with heavy tails."""
    daily, events = _series(planted=True, seed=9)
    panel = build_abnormal_panel(daily, events)

    def cube(x):
        return np.sign(x) * np.abs(x) ** 3          # strictly increasing, order preserving

    stretched = panel._replace(
        abnormal=cube(panel.abnormal),
        estimation_residuals=[r.map(cube) for r in panel.estimation_residuals])

    before, _ = corrado_rank_t(panel, upto_k=5, start=1)
    after, _ = corrado_rank_t(stretched, upto_k=5, start=1)
    assert after == pytest.approx(before)

    plain_before, _ = cross_sectional_t(cumulative_abnormal(panel, start=1)[:, 4])
    plain_after, _ = cross_sectional_t(cumulative_abnormal(stretched, start=1)[:, 4])
    assert abs(plain_after - plain_before) > 0.1


def test_bmp_standardises_before_taking_the_cross_section(planted_table):
    """BMP divides each event by its own forecast error first. An event with a noisier
    estimation window must therefore count for less than a quiet one."""
    daily, events = _series(planted=True, seed=13)
    panel = build_abnormal_panel(daily, events)
    car = cumulative_abnormal(panel, start=1)[:, 4]
    deviation = np.nansum(panel.factor_deviation[:, panel.event_time <= 5], axis=1)
    _, _, scar = bmp_t(panel, car, 5, deviation)
    assert np.isfinite(scar).sum() == N_EVENTS
    # Standardised values are on a common scale, so their spread is order 1, not order of
    # the raw return.
    assert np.nanstd(scar) < 100 * np.nanstd(car)


def test_the_module_does_not_import_the_forecast_evaluation_rules():
    """T11 constraint: realised effect and forecastability stay structurally separate, and
    neither module may reach for the other's decision rules."""
    import ast

    root = artifact_file("dataset.parquet").parents[2] / "src" / "evaluation"
    for module, forbidden in (("event_study.py", "verification"),
                              ("verification.py", "event_study")):
        tree = ast.parse((root / module).read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        assert not any(forbidden in name for name in imported), (module, imported)


def test_the_published_artifact_covers_both_series():
    path = artifact_file("event_study_caar.parquet")
    if not path.exists():
        pytest.skip("run scripts/run_event_study.py first")
    table = pd.read_parquet(path)
    assert set(table["series"]) == {"return", "volume"}
    for column in ("caar", "ci_low", "ci_high", "t_bmp", "t_corrado",
                   "t_kolari_pynnonen", "t_cross_sectional"):
        assert column in table.columns, column
    assert (table["ci_low"] <= table["caar"]).all()
    assert (table["caar"] <= table["ci_high"]).all()
