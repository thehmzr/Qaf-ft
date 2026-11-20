# Assessment protocol

How the model is judged, with the statistical reasoning behind each rule.
Section numbers refer to *An Introduction to Statistical Learning (Python edition)*.

## 1. A flag is a hypothesis test (ch. 13)

For every word the null hypothesis is "recited correctly". A flag rejects it.
A false flag is a Type I error; a missed mistake is a Type II error (Table 13.1).
Lowering one raises the other (§13.1.2), so the operating point is a choice, not a fact.

## 2. Per-word rates compound (§13.2, §13.3.1)

With m independent checks at per-word false-flag rate a, the chance of at least one
false flag is 1 - (1 - a)^m (eq. 13.5).

| per-word rate | 10-word ayah | 30-word ayah | 300-word session |
|---|---|---|---|
| 5%   | 40% | 79% | ~100% |
| 1%   | 10% | 26% | 95% |
| 0.1% | 1%  | 3%  | 26% |
| 0.03% | 0.3% | 0.9% | 9% |

So "5% false positives" is unusable, and the per-word target has to be near 0.1% or below.

## 3. Which error rate to control (§13.3, §13.4)

- Family-wise rate (Bonferroni: per-word level = a/m, §13.3.2) guarantees "no false flag
  in this ayah" with probability 1 - a, but costs power as m grows (§13.3.3).
- False discovery rate (Benjamini-Hochberg, Algorithm 13.2) controls the share of shown
  flags that are wrong. This matches what a learner experiences: "when it flags, it is right".

Decision: report both. Thresholds are set for a target share of wrong flags (5% then 1%),
and the per-ayah family-wise rate is reported alongside.

## 4. p-values from an empirical null (§13.5)

The score of a word has no known distribution, so the null is estimated from data:
scores of words known to be correct, from speakers not used for training. A word's p-value
is the fraction of that null with a score at least as extreme. The null set must cover
phone audio, women, children and non-Arab reciters, or the p-values are wrong for them.

## 5. Thresholds and ROC (§4.4.2)

A 0.5 cut-off minimises total error but ignores that the two errors cost differently
(the Default example: 3.73% total error is accepted to catch more defaulters). Here the
costly error is the false flag, so the cut-off moves the other way. Every result is given
as a full ROC curve plus the catch rate at fixed false-flag rates (5%, 1%, 0.1%).
Accuracy alone is never reported: mistakes are rare, so "never flag" scores well.

## 6. Honest error estimates (ch. 5)

- Training error understates test error (§5.1); nothing is reported on data used to fit
  or to choose a threshold.
- Clips from one speaker are not independent. Folds and held-out sets are split by
  SPEAKER, 5-fold (§5.1.3: k = 5 or 10 balances bias and variance, §5.1.4).
- Anything chosen from data (features, thresholds, calibration) is chosen inside the fold.
- Intervals come from the bootstrap (§5.2), resampling speakers, 1,000 or more draws.
- Two models are compared on the same clips with a paired bootstrap.

## 7. Keep the second stage simple (§10.6, ch. 6)

The verifier that turns scores into a flag decision has a handful of inputs and little
labelled data. A regularised logistic regression is the default; a larger model has to
beat it under cross-validation to be used (the Hitters comparison, Table 10.2).

## 8. Fitting the acoustic model (§10.7)

Early stopping on a held-out speaker set, dropout, weight decay and data augmentation are
the regularisers. Model selection uses the catch rate at a fixed false-flag rate on
held-out learners, not the training loss.

## 9. Label noise sets the ceiling

Where annotators disagree with each other, no model can be measured below that noise.
Agreement between annotators is reported next to every result on human-labelled data.
