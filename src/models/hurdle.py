"""Two-stage hurdle model for Y3 (recovery time).

Y3 is censored at 90 days by construction, so ~10% of events sit exactly on the cap. A
single regressor smears that point mass; this model classifies recovery-within-window
first, then regresses `log1p(duration)` on the uncensored rows only."""

from __future__ import annotations

import numpy as np

CAP = 90.0


class HurdleRecoveryModel:
    """Stage 1 classifies recovery-within-window; stage 2 regresses log1p(duration)."""

    def __init__(self, classifier, regressor, cap: float = CAP):
        self.classifier = classifier
        self.regressor = regressor
        self.cap = cap
        self.fallback_ = None      # used when a fold has no variation to learn from
        self.degenerate_ = False

    def fit(self, X, y, recovered=None):
        """`recovered=None` infers stage-1 recovery from `y < cap` alone, the original
        behaviour. Pass the real indicator explicitly (methodology-audit finding #7:
        """
        y = np.asarray(y, dtype=float)
        recovered = (y < self.cap) if recovered is None else np.asarray(recovered, dtype=bool)
        self.fallback_ = float(np.mean(y)) if len(y) else self.cap

        # A fold in which every event recovered (or none did) gives stage 1 one class and
        # nothing to fit. Fall back to the training mean rather than raising: at this N a
        # degenerate fold is a real possibility, not a programming error.
        if recovered.all() or (~recovered).any() and recovered.sum() < 2:
            self.degenerate_ = True
            return self

        self.classifier.fit(X, recovered.astype(int))
        self.regressor.fit(np.asarray(X)[recovered], np.log1p(y[recovered]))
        return self

    def predict(self, X):
        X = np.asarray(X)
        if self.degenerate_:
            return np.full(len(X), self.fallback_)

        p_recover = self.classifier.predict_proba(X)[:, 1]
        duration = np.expm1(self.regressor.predict(X))
        # A stage-2 prediction only describes recovered events, so it cannot exceed the
        # window it was measured inside.
        duration = np.clip(duration, 0.0, self.cap)
        return np.clip(p_recover * duration + (1.0 - p_recover) * self.cap, 0.0, self.cap)

    def predict_recovery_probability(self, X):
        if self.degenerate_:
            return np.full(len(np.asarray(X)), float("nan"))
        return self.classifier.predict_proba(np.asarray(X))[:, 1]


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 3))
    # Recovery is driven by the first feature; duration by the second.
    recovers = X[:, 0] + rng.normal(0, 0.3, n) > 0
    y = np.where(recovers, np.clip(np.abs(X[:, 1]) * 8, 0, 89), CAP)

    model = HurdleRecoveryModel(
        LogisticRegression(max_iter=1000),
        RandomForestRegressor(n_estimators=100, max_depth=3, random_state=0),
    ).fit(X, y)

    pred = model.predict(X)
    assert pred.min() >= 0.0 and pred.max() <= CAP, (pred.min(), pred.max())
    # It must beat the training mean on data where the structure is genuinely present.
    assert np.mean(np.abs(pred - y)) < np.mean(np.abs(np.mean(y) - y))

    # A fold where nothing is censored must not raise; it degrades to the mean.
    allrec = HurdleRecoveryModel(LogisticRegression(max_iter=1000),
                                 RandomForestRegressor(n_estimators=10, random_state=0))
    allrec.fit(X, np.full(n, 5.0))
    assert allrec.degenerate_ and np.allclose(allrec.predict(X), 5.0)

    print("hurdle.py self-check passed")
