import matplotlib

matplotlib.use("Agg")  # must precede any pyplot import

import numpy as np
import pandas as pd
import pytest

from src.evaluation import collinearity as col
from src.visualization import result_figures as fx
from src.evaluation.metrics import pooled_frame
from src.training.walk_forward import generate_walk_forward_splits

TARGETS = ["Y1_ASPI_5D_Forward_LogReturn_Pct", "Y2_5D_Forward_AbnormalVolume_LogRatio", "Y3_ASPI_Recovery_Time"]


def _bundle(n_events=40, n_folds_cfg=(20, 10, 10)):
    """A miniature stand-in for the real pipeline state: a dataset, a feature matrix with
    one deliberately redundant pair, walk-forward splits, and a `results` dict shaped
    exactly like the one the modelling stage produces."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2005-01-01", periods=n_events, freq="90D")
    base = rng.normal(size=n_events)

    dataset = pd.DataFrame({
        "event_date": dates,
        "disaster_type": rng.choice(["Flood", "Storm", "Drought"], n_events),
        "financial_damage": np.abs(base) * 1e6,
        "log_financial_damage": np.log1p(np.abs(base) * 1e6),  # monotone -> spearman 1.0
        "log_return": rng.normal(0, 0.01, n_events),
        "rolling_std_30": np.abs(rng.normal(0.01, 0.002, n_events)),
        "disaster_Flood": rng.integers(0, 2, n_events).astype(float),
        "disaster_Other": rng.integers(0, 2, n_events).astype(float),
    })
    y = pd.DataFrame({
        "Y1_ASPI_5D_Forward_LogReturn_Pct": rng.normal(0, 0.014, n_events),
        "Y2_5D_Forward_AbnormalVolume_LogRatio": rng.normal(0, 0.58, n_events),
        "Y3_ASPI_Recovery_Time": np.clip(rng.exponential(12, n_events).round(), 0, 90),
    })
    y.loc[y.index[:3], "Y2_5D_Forward_AbnormalVolume_LogRatio"] = np.nan  # mirrors the 3 real missing events
    dataset = pd.concat([dataset, y], axis=1)

    feature_cols = ["financial_damage", "log_financial_damage", "log_return",
                    "rolling_std_30", "disaster_Flood", "disaster_Other"]
    X = dataset[feature_cols]

    tw, te, st = n_folds_cfg
    splits = list(generate_walk_forward_splits(n_events, tw, te, st))

    results = {}
    for name, noise in (("ridge", 0.9), ("random_forest", 1.1), ("naive_zero", None),
                        ("naive_train_mean", None)):
        results[name] = {}
        for t in TARGETS:
            store = {"rmse": [], "mae": [], "r2": [], "y_true": [], "y_pred": []}
            for s in splits:
                idx = np.asarray(s.test_index)
                idx = idx[y[t].iloc[idx].notna().to_numpy()]
                yt = y[t].iloc[idx].to_numpy()
                if name == "naive_zero":
                    yp = np.zeros_like(yt)
                elif name == "naive_train_mean":
                    yp = np.full_like(yt, float(np.nanmean(y[t].iloc[s.train_index])))
                else:
                    yp = yt * 0.3 + rng.normal(0, np.nanstd(yt) * noise, len(yt))
                store["y_true"].append(yt)
                store["y_pred"].append(yp)
                store["rmse"].append(float(np.sqrt(np.mean((yp - yt) ** 2))))
                store["mae"].append(float(np.mean(np.abs(yp - yt))))
                store["r2"].append(0.0)
            store["pooled_r2"] = 0.0
            results[name][t] = store
    return dataset, X, y, splits, results


def test_pooled_frame_maps_predictions_back_to_the_right_events():
    """The highest-value assertion here. A silent off-by-one in this mapping would corrupt
    every scatter, residual and severity plot in a way that still looks plausible."""
    dataset, X, y, splits, results = _bundle()

    # Overwrite predictions with the event row index itself, so the mapping is checkable.
    for t in TARGETS:
        store = results["ridge"][t]
        store["y_pred"] = []
        for s in splits:
            idx = np.asarray(s.test_index)
            idx = idx[y[t].iloc[idx].notna().to_numpy()]
            store["y_pred"].append(idx.astype(float))

    for t in TARGETS:
        frame = pooled_frame(results, "ridge", t, splits, dataset, y=y)
        assert np.array_equal(frame["row_index"].to_numpy(), frame["y_pred"].to_numpy())
        # Y2 drops the first three events, so its frame must be shorter than Y1's.
        assert len(frame) == sum(len(a) for a in results["ridge"][t]["y_true"])


def test_pooled_frame_handles_a_model_that_skips_fold_zero():
    """The stacked model forfeits fold 0 to its meta-learner and stores one fewer fold.
    The offset must be inferred, not assumed."""
    dataset, X, y, splits, results = _bundle()
    t = "Y1_ASPI_5D_Forward_LogReturn_Pct"
    results["stacked"] = {t: {k: v[1:] if isinstance(v, list) else v
                              for k, v in results["ridge"][t].items()}}
    frame = pooled_frame(results, "stacked", t, splits, dataset, y=y)
    assert frame["fold"].min() == 1, "fold numbering must account for the skipped fold"


def test_pooled_frame_raises_on_a_mask_mismatch():
    dataset, X, y, splits, results = _bundle()
    results["ridge"]["Y1_ASPI_5D_Forward_LogReturn_Pct"]["y_true"][0] = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        pooled_frame(results, "ridge", "Y1_ASPI_5D_Forward_LogReturn_Pct", splits, dataset, y=y)


def test_redundant_drop_rule_keeps_the_more_primitive_column():
    _, X, _, _, _ = _bundle()
    keep, drop, detail = col.redundant_drop_set(X)
    assert "log_financial_damage" in drop
    assert "financial_damage" in keep
    assert detail.loc[detail["dropped"] == "log_financial_damage", "kept"].iloc[0] == \
        "financial_damage"


def test_vif_survives_the_singular_one_hot_block():
    _, X, _, _, _ = _bundle()
    vif = col.compute_vif(X, drop_reference="disaster_Other")
    assert len(vif) == X.shape[1] - 1
    assert not vif["vif"].isna().all()


def test_critical_r_matches_the_fisher_z_value_at_n64():
    assert col.critical_r(64) == pytest.approx(0.246, abs=0.01)


@pytest.mark.parametrize("name", [
    "target_distributions", "target_dependence", "correlation_heatmap", "vif",
    "walk_forward", "model_vs_baseline", "skill_forest", "pred_vs_actual",
    "overfitting_gap", "roc",
])
def test_every_figure_renders_and_exports(name, tmp_path):
    import matplotlib.pyplot as plt

    dataset, X, y, splits, results = _bundle()
    fx.apply_thesis_style()
    fx.set_figure_dir(tmp_path)

    builders = {
        "target_distributions": lambda: fx.plot_target_distributions(dataset, TARGETS),
        "target_dependence": lambda: fx.plot_target_dependence_y1_y3(dataset),
        "correlation_heatmap": lambda: fx.plot_feature_correlation_heatmap(X),
        "vif": lambda: fx.plot_vif(X),
        "walk_forward": lambda: fx.plot_walk_forward_folds(splits, dataset, y),
        "model_vs_baseline": lambda: fx.plot_model_vs_baseline(results, TARGETS),
        "skill_forest": lambda: fx.plot_skill_forest(results, TARGETS, n_boot=200),
        "pred_vs_actual": lambda: fx.plot_pred_vs_actual(
            results, ["ridge", "random_forest"], TARGETS[0]),
        "overfitting_gap": lambda: fx.plot_overfitting_gap(
            {t: 0.7 for t in TARGETS}, results, TARGETS, model="random_forest"),
        "roc": lambda: fx.plot_roc_with_ci(
            results, ["ridge", "random_forest"], n_boot=100),
    }

    fig = builders[name]()
    assert isinstance(fig, matplotlib.figure.Figure)
    assert fig.axes and any(ax.has_data() for ax in fig.axes), f"{name} drew nothing"

    path = fx.save_figure(fig, f"fig_test_{name}", close=True, verbose=False)
    assert path.exists() and path.stat().st_size > 3_000
    assert path.parent == tmp_path, "figure escaped the configured output directory"
    plt.close("all")


def test_style_uses_more_than_hue_so_figures_survive_greyscale():
    import matplotlib.pyplot as plt

    fx.apply_thesis_style()
    cyc = plt.rcParams["axes.prop_cycle"].by_key()
    assert "marker" in cyc and "linestyle" in cyc and "color" in cyc
