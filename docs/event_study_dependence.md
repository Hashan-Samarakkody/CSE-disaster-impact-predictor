# Why Kolari–Pynnönen Does Not Apply Here, and What We Use Instead

## The Single-Index Problem

Kolari and Pynnönen (2010) correct the Boehmer–Musumeci–Poulsen (BMP) standardised
test for cross-event correlation.  The correction scales the BMP statistic by

    sqrt( (1 − r̄) / (1 + (N − 1) r̄) )

where r̄ is the mean pairwise sample correlation of estimation-window market-model
residuals across all N event pairs.

The formula assumes that r̄ is a small positive number reflecting genuine but weak
dependence among otherwise distinct securities.  In a *single-security* (or here, a
*single-index*) event study, this assumption collapses: every event's market-model
residuals come from the **same** ASPI return series minus the same SP 500 factor.
Any two events whose 250-session estimation windows share calendar sessions will
therefore have residual series that are virtually identical over the shared region.

## Empirical Evidence

For the 74 CSE disaster events analysed here:

| Series   | Mean pairwise residual correlation |
|----------|-----------------------------------|
| Returns  | 0.9977                            |
| Volume   | 1.0000                            |

The return correlation of 0.9977 arises because most event-pairs overlap substantially
in their estimation windows.  Volume correlation reaches the numerical ceiling because
estimation-window log-volume changes are dominated by long-run market trends shared by
all events.

Plugging r̄ = 0.9977 into the K–P formula with N = 74 gives a denominator of

    1 + 73 × 0.9977 ≈ 73.8

so the correction factor is approximately sqrt(0.0023 / 73.8) ≈ 0.0056.  The K–P
statistic becomes virtually zero regardless of the underlying BMP value; it carries no
information about whether an effect is present.  The resulting p-values are uniformly
near 1.0, as reported in the event-study table.

## Correct Inference: Randomization Test

We replace K–P with a pure randomization p-value
(`scripts/placebo_randomization_test.py`):

1. For each of **B = 5 000** replications, draw one placebo origin for each of the 74
   real events uniformly at random from that event's pool.  Each pool consists of
   trading sessions satisfying:
   - same calendar month (1–12) as the real event;
   - different calendar year;
   - outside every real event's declared window (sessions [−5, +20]);
   - 20-session pre-event realised volatility within a factor of 3 of the real event's.

2. Compute the CAAR (market-model, returns) and CAAV (mean-adjusted, log-volume) for
   each of the declared horizons k = 1, 5, 10, 20 trading sessions.

3. Report the two-sided randomization p-value
   p = (1 + #{|CAAR_b| ≥ |CAAR_observed|}) / (1 + B).

This provides a valid reference distribution under the null of no event effect,
regardless of the cross-event correlation structure, because the placebo draws respect
the same temporal constraints as the real events without conditioning on them.

## Results

| Series  | Horizon | Observed CAAR | Placebo mean | Placebo SD | p (rand.) |
|---------|---------|--------------|-------------|-----------|----------|
| Return  | 1       | −0.00059     | +0.00027    | 0.00151   | 0.670    |
| Return  | 5       | +0.00055     | +0.00211    | 0.00372   | 0.898    |
| Return  | 10      | +0.00394     | +0.00233    | 0.00533   | 0.496    |
| Return  | 20      | +0.00402     | +0.00261    | 0.00797   | 0.636    |
| Volume  | 1       | +0.00416     | +0.10954    | 0.10708   | 0.981    |
| Volume  | 5       | +0.62963     | +0.53010    | 0.43411   | 0.410    |
| Volume  | 10      | +1.29795     | +1.00771    | 0.82241   | 0.362    |
| Volume  | 20      | +2.89096     | +2.28813    | 1.57317   | 0.348    |

No horizon for either series reaches conventional significance (all p > 0.30), consistent
with the BMP and Corrado statistics and consistent with the study's primary conclusion
that disaster impacts on the CSE are not reliably detected in aggregate event-study
analysis.

## References

- Kolari, J. W. and Pynnönen, S. (2010). Event study testing with cross-sectional
  correlation of abnormal returns. *Review of Financial Studies*, 23(11), 3996–4025.
- Boehmer, E., Musumeci, J. and Poulsen, A. B. (1991). Event-study methodology under
  conditions of event-induced variance. *Journal of Financial Economics*, 30(2), 253–272.
- Corrado, C. J. (1989). A nonparametric test for abnormal security-price performance
  in event studies. *Journal of Financial Economics*, 23(2), 385–395.
