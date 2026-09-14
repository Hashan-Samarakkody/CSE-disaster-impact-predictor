import numpy as np
import pandas as pd

from src.sampling.time_aware_smogn import time_aware_smogn


def test_smogn_keeps_onehot_types_derived_identities_and_stays_between_real_parents():
    # Two Floods two years apart (interpolatable) plus one Storm with no same-type partner.
    idx = [0, 1, 2]
    damage = np.array([1_000_000.0, 5_000_000.0, 2_000_000.0])
    flood = np.array([1.0, 1.0, 0.0])
    X = pd.DataFrame(
        {
            "financial_damage": damage,
            "log_financial_damage": np.log1p(damage),
            "log_return": [-0.01, -0.03, -0.02],
            "squared_return": [0.0001, 0.0009, 0.0004],
            "damage_to_gdp": damage / 1e11,
            "disaster_Flood": flood,
            "disaster_Storm": 1.0 - flood,
            "log_damage_x_flood": np.log1p(damage) * flood,
        },
        index=idx,
    )
    y = pd.DataFrame(
        {"Y1": [-0.01, -0.03, -0.02], "Y2": [0.5, 1.5, 1.0], "Y3": [40.0, 80.0, 60.0]},
        index=idx,
    )
    dates = pd.Series(pd.to_datetime(["2010-01-01", "2012-01-01", "2011-01-01"]), index=idx)
    type_cols = ["disaster_Flood", "disaster_Storm"]

    X_aug, y_aug = time_aware_smogn(
        X,
        y,
        pd.Series(True, index=idx),
        event_dates=dates,
        type_cols=type_cols,
        gdp_current_usd=pd.Series(1e11, index=idx),
        max_synthetic_share=1.0,  # the cap is a separate guard-rail, not what this test targets
    )
    syn = X_aug.iloc[len(X):]

    # Real rows are returned first, unmutated and in order.
    pd.testing.assert_frame_equal(X_aug.iloc[: len(X)], X.reset_index(drop=True))
    # The lone Storm has no same-type partner, so it yields no synthetic row.
    assert len(syn) == 2
    assert (syn["disaster_Flood"] == 1.0).all()
    # One-hot block is copied verbatim, never blended into a fractional type.
    assert (syn[type_cols].sum(axis=1) == 1.0).all()
    # Derived features are recomputed from the interpolated parents, not interpolated.
    assert np.allclose(syn["log_financial_damage"], np.log1p(syn["financial_damage"]))
    assert np.allclose(syn["squared_return"], syn["log_return"] ** 2)
    assert np.allclose(syn["damage_to_gdp"], syn["financial_damage"] / 1e11)
    # Recompute order: the interaction term uses the rebuilt log, not the interpolated one.
    assert np.allclose(syn["log_damage_x_flood"], syn["log_financial_damage"])
    # Interpolation, never extrapolation: synthetic rows sit inside their parents' range.
    assert syn["financial_damage"].between(damage[0], damage[1]).all()
    assert y_aug.iloc[len(X):]["Y3"].between(40.0, 80.0).all()


def test_smogn_raises_rather_than_silently_exceeding_the_synthetic_share_cap():
    idx = list(range(4))
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "disaster_Flood": 1.0}, index=idx)
    y = pd.DataFrame({"Y3": [40.0, 50.0, 60.0, 70.0]}, index=idx)

    # 4 minority rows x 2 draws = 8 synthetic on 4 real = 67% synthetic, far above the
    # pre-registered 25% ceiling.
    try:
        time_aware_smogn(
            X, y, pd.Series(True, index=idx),
            type_cols=["disaster_Flood"], n_synthetic_per_row=2, max_synthetic_share=0.25,
        )
    except ValueError as exc:
        assert "cap" in str(exc)
    else:
        raise AssertionError("expected the synthetic-share cap to raise")
