"""Right-censored survival model for Y3 (recovery time)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

CAP = 90.0
# lifelines' AFT fitters require strictly positive durations. A same-day recovery (Y3=0)
DURATION_EPS = 0.5


class AFTRecoveryModel:
    """Weibull/LogNormal AFT model with right-censoring at `cap`, chosen in-fold by AIC.

    Mirrors `HurdleRecoveryModel`'s interface (`fit`, `predict`) and degenerate-fold
    handling so it can be dropped into the same walk-forward loop and results table.
    """

    def __init__(self, cap: float = CAP, penalizer: float = 0.1):
        self.cap = cap
        self.penalizer = penalizer
        self.model_ = None
        self.feature_cols_ = None
        self.fallback_ = None
        self.degenerate_ = False

    def fit(self, X, y, censored=None):
        """`censored=None` infers censoring from the 90-day cap alone (`y >= cap`), the
        original behaviour. Pass the real censoring indicator explicitly (e.g.
        """
        from lifelines import LogNormalAFTFitter, WeibullAFTFitter
        from lifelines.exceptions import ConvergenceError

        X = pd.DataFrame(np.asarray(X))
        y = np.asarray(y, dtype=float)
        self.feature_cols_ = list(X.columns)
        self.fallback_ = float(np.mean(y)) if len(y) else self.cap

        censored = (y >= self.cap) if censored is None else np.asarray(censored, dtype=bool)
        # A fold needs both event types (recovered and censored) and enough rows per
        if len(y) < 8 or censored.all() or (~censored).sum() < 4:
            self.degenerate_ = True
            return self

        df = X.copy()
        df["_duration"] = y + DURATION_EPS
        df["_observed"] = ~censored

        candidates = [WeibullAFTFitter(penalizer=self.penalizer),
                      LogNormalAFTFitter(penalizer=self.penalizer)]
        fitted, aics = [], []
        for model in candidates:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(df, duration_col="_duration", event_col="_observed")
                fitted.append(model)
                aics.append(model.AIC_)
            except (ConvergenceError, np.linalg.LinAlgError, ValueError):
                continue

        if not fitted:
            self.degenerate_ = True
            return self

        self.model_ = fitted[int(np.argmin(aics))]
        return self

    def predict(self, X):
        X = pd.DataFrame(np.asarray(X), columns=self.feature_cols_)
        if self.degenerate_ or self.model_ is None:
            return np.full(len(X), self.fallback_)

        # Median survival time: robust point summary for a right-skewed AFT distribution,
        # unlike the mean, which can be numerically unstable (Weibull/LogNormal tails).
        median = self.model_.predict_median(X).to_numpy()
        median = np.nan_to_num(median, nan=self.cap, posinf=self.cap) - DURATION_EPS
        return np.clip(median, 0.0, self.cap)

    def concordance_index(self, X, y, censored=None):
        """Harrell's C-index: the metric right-censored survival predictions should be
        judged on (RMSE/MAE are still reported for comparability with the other models,
        but they are not the metric this model was fit to optimise). Same `censored`
        override as `fit` -- pass the real indicator, not just `y >= cap`, once some
        rows are censored by a competing event before the 90-day cap."""
        from lifelines.utils import concordance_index

        if self.degenerate_ or self.model_ is None:
            return float("nan")
        X = pd.DataFrame(np.asarray(X), columns=self.feature_cols_)
        y = np.asarray(y, dtype=float)
        median = self.model_.predict_median(X).to_numpy()
        median = np.nan_to_num(median, nan=self.cap * 10, posinf=self.cap * 10)
        censored = (y >= self.cap) if censored is None else np.asarray(censored, dtype=bool)
        # lifelines' convention: predicted_scores should be concordant with the duration
        return float(concordance_index(y, median, event_observed=~censored))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 60
    X = rng.normal(size=(n, 3))
    true_duration = np.clip(np.abs(X[:, 0]) * 15 + rng.normal(0, 3, n), 0, None)
    y = np.minimum(true_duration, CAP)

    model = AFTRecoveryModel().fit(X, y)
    assert not model.degenerate_, "should fit on a well-posed 60-row synthetic fold"
    pred = model.predict(X)
    assert pred.min() >= 0.0 and pred.max() <= CAP, (pred.min(), pred.max())
    cidx = model.concordance_index(X, y)
    assert 0.0 <= cidx <= 1.0, cidx
    assert cidx > 0.55, f"expected the model to beat chance concordance (0.5), got {cidx}"

    tiny = AFTRecoveryModel().fit(X[:5], y[:5])
    assert tiny.degenerate_
    assert np.allclose(tiny.predict(X[:5]), tiny.fallback_)

    print(f"survival_recovery.py self-check passed (c-index={cidx:.3f})")
