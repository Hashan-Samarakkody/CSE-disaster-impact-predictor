# Repository metadata

The GitHub About field and the repository description are settings on github.com, not
files in this repository, so they cannot be corrected by a commit. This page carries the
text they should be set to, so that the correction is recorded, reviewable and easy to
apply (Revision 2, D1).

## What is wrong with the current description

The About field currently describes the project as "creating an explainable **early
warning system**" for "abnormal volume **crashes**". Both halves contradict the study's own
findings.

**"Early warning system" is not supportable.** Several predictors are finalised after the
event, not before it: EM-DAT damage assessments and casualty counts settle over days to
months, DesInventar compiles local loss records afterwards, NASA POWER reanalysis publishes
with a lag, and the World Bank series are annual. Those are 31 of the 68 features, and they
are classed `ex_post` in `artifacts/results/feature_spec.json` for exactly this reason. A
model that needs them cannot be run the day before a disaster. The study is an **ex post
impact attribution** exercise.

**"Volume crashes" presupposes a fall.** Target two is a two sided log ratio, and 23 of its
61 observed values are positive: turnover rose. The neutral name, used throughout the code
and the frozen protocol, is **forward abnormal trading volume**.

## The text to set

Paste this into the GitHub About field, replacing what is there:

> Ex post impact attribution study of how natural disasters affected the Colombo Stock
> Exchange, 2000 to 2025. Measures the realised response in returns, trading volume and
> recovery duration across 74 events, and tests out of sample whether any of it can be
> forecast. Reports its negative results.

If the field is too short for that, the one line version:

> Ex post study of how natural disasters affected the Colombo Stock Exchange, and whether
> the response can be forecast out of sample.

Suggested topics: `event-study`, `financial-econometrics`, `survival-analysis`,
`reproducible-research`, `sri-lanka`, `natural-disasters`.

## How to apply it

On github.com, open the repository, click the gear icon beside **About** on the right of
the Code tab, replace the Description field and save. There is no CLI step in this
environment: the GitHub CLI is not installed, so this is the one item in the revision that
has to be done by hand.

## Status

**Outstanding, requires the repository owner.** Everything inside the repository that
carried the same two claims has already been corrected: `README.md` states the ex post
framing in its Research objective section, and the neutral target name is used in
`README.md`, `docs/architecture.md`, `docs/target_definitions.md` and
`src/config/settings.py`, where the identifier is now `FORWARD_ABNORMAL_VOLUME`.
