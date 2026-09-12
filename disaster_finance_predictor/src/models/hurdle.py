"""Two-stage hurdle model for Y3 (recovery time).

Y3 is not a continuous duration. It is a point mass at 0 (every event whose event-day
return was non-negative "recovers" immediately, by construction), a short right tail, and
a second point mass at the 90-day cap. Fitting one continuous regressor across that shape
is the wrong model, which is the most likely reason the training-mean predictor currently
beats every fitted model on this target.

The 90 at the cap is also not an observation. For a non-recovering event the data says
`T > 90`, not `T = 90`; assigning 90 is a biased point estimate. The hurdle handles that
by predicting the *probability* of recovering inside the window and only regressing
duration on the events that actually did:

    E[Y3] = P(recover) * E[duration | recovered] + (1 - P(recover)) * 90

A survival model would be the textbook answer to censoring, but with roughly 9 censored
events in 64 there is nothing to estimate a hazard from, and the censoring point here is a
design constant rather than an end of observation. The hurdle is the version this sample
can support.

Disclose alongside any result from this model: stage 1 is close to a sign classifier for
Y1, because `Y3 = 0` if and only if `Y1 >= 0`.
"""

from __future__ import annotations

import numpy as np

CAP = 90.0


class HurdleRecoveryModel:
    """Stage 1 classifies recovery-within-window; stage 2 regresses log1p(duration).

    Stage 2 trains only on rows that recovered, so it never sees the censored 90s. The
    log1p transform is applied because the recovered durations are strongly right-skewed
    (median 0, upper quartile ~5 trading days) and squared error on the raw scale would be
    dominated by a handful of slow recoveries.
    """

    def __init__(self, classifier, regressor, cap: float = CAP):
        self.classifier = classifier
        self.regressor = regressor
        self.cap = cap
        self.fallback_ = None      # used when a fold has no variation to learn from
        self.degenerate_ = False

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        recovered = y < self.cap
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
