# Discussion section (draft)

**Scope.** This discusses the results in [results_section.md](results_section.md),
§4.1–§4.8, for the [S3] study (climate-horizon gradient weighting). Numbers
appear only where they are needed for an argument. The tables are in the
Results.

**Labels used throughout.** Every paragraph keeps three kinds of statement
apart:
- **Observed:** what the experiments measured.
- **Interpretation:** what the evidence supports.
- **Possible explanation:** a mechanism that was *not* tested. It is worded
  cautiously and should stay that way in the final paper.

---

## 5.1 The answer to the research question

**Observed.**
- Weighting the climate encoder's gradients by measured per-horizon climate
  reliance (D) did not improve on equal weighting (B).
- D's error was slightly higher at every horizon, and no difference was
  significant. By the frozen criteria, D is non-inferior at h=1–2 and there is
  no measurable difference at h=3–4 (§4.2).
- The result is the same for:
  - the ensemble estimand;
  - all nine folds;
  - other bootstrap block lengths.
- The 2026 check has the same sign (§4.8).

**Interpretation.** The research question has a negative answer for this model
and data. At weight ratios of 1.7–4.0 between horizons (§4.3), out-of-sample
horizon weighting of the climate encoder's loss is not an effective way to
improve forecast accuracy. The result is informative because two simpler
explanations can be ruled out:
- **The intervention was not trivial.** The weights were far from uniform
  (§4.3).
- **The baseline was not weak.** B beats persistence from h=2 and is close to
  the stored reference model at h=3–4 (§4.1).

**What the result does not answer.** It does not show that controlling which
horizons train the encoder is ineffective in general. Arm F points the other
way (§5.3).

## 5.2 Why the weighting had so little effect

**Observed.** At identical model and optimiser states:
- the measured weights rotated each climate-encoder update by about 7° and
  changed its size by about 13% (§4.6);
- the raw gradient norm grew by 14.7%, but the Adam update norm by only 2.6%;
- the trained B and D models differ less from each other than two seeds of B
  do, both in predictions and in the learned kernels;
- gates differ by at most 0.02 (§4.6, §4.7).

In the same analysis, each horizon's encoder gradient was partly aligned with
the h=4 gradient (cosine 0.48–0.68; `climate_horizon_results.md` §6).

**Interpretation.** The measured weighting was a small perturbation to training
compared with the variation from random initialisation and data order. A
change smaller than seed-to-seed variation cannot be expected to produce an
accuracy difference that seven test years could detect. This part follows
directly from the measurements.

**Possible explanations.** None of these was tested.

1. **Partly shared gradient direction.** The horizons' encoder gradients
   already point partly in the same direction. Re-weighting mean-one
   combinations of partly aligned vectors can only rotate the sum by a limited
   amount, which fits the small angles observed.
2. **Adam's per-parameter normalisation.** Adam rescales each parameter's step
   by a running estimate of its gradient magnitude. That could absorb much of
   a change in gradient scale. The gap between the +14.7% raw-gradient change
   and the +2.6% update change is consistent with this, but it was not
   isolated experimentally. For example, no arm used plain SGD.
3. **Compensation by the rest of the network.** The gate and the prediction
   heads are trained on the unweighted loss. They may adapt to whatever the
   encoder produces and damp any difference that reaches the forecast.
4. **Rare gradient clipping and model selection.** Clipping was active on up
   to 2% of a run's steps. Early stopping uses the unweighted validation loss.
   Either may reduce differences further. There is no direct evidence for or
   against this.

The diagnostics describe the size of the change. They do not identify which of
these mechanisms, if any, is responsible.

## 5.3 The longest-horizon routing arm (F)

**Observed.** Training the climate encoder from the h=4 loss alone (F) gave
the lowest error of any arm, at every horizon, including h=1 (§4.4).
- The confidence intervals exclude zero at h=1–2 and include it at h=3–4.
- The p-values are unadjusted.
- F was run with 3 seeds.
- Its corrected form was first seen after the routing correction.
- It was not scored on 2026.

**Interpretation.** F is the only evidence in this study that controlling which
horizons train the climate encoder can change accuracy. It is a lead that
needs confirmation, not a result. Two features of the evidence argue for
caution:
- the gain is almost constant across horizons (−0.64 to −0.69);
- it is not tied to the horizons where climate reliance is largest.

**Possible explanations.** None of these was tested.

- F is a much stronger intervention than D. It removes the short-horizon
  gradients from the encoder entirely, rather than down-weighting them by a
  factor of 2–4. That may move the encoder further than the seed noise that
  swamped D.
- **F also improved h=1, which was not expected.** h=1 relies little on
  weather (§4.3). Possible reasons:
  - An encoder trained only on the h=4 loss might learn a smoother or more
    general weather summary, which the separately trained h=1 climate head
    can still use.
  - Removing three of the four loss terms from the encoder may simply act as
    regularisation.
- Neither possibility was examined. With 3 seeds, a gain concentrated in a few
  years (§4.5 shows this pattern for other contrasts) cannot be excluded.

## 5.4 Lag alignment and the source of the weights

**Observed.**
- **Lag alignment.** Measured weighting reduced error when the delay kernels
  were defined relative to the forecast origin (C vs A). It did not when they
  were defined relative to the target week (D vs B).
- **The interaction.** The interaction is positive at both horizons reported
  (+0.52 at h=1, +0.58 at h=4). It does not survive correction over 20 tests
  and is concentrated in 2017 and 2019 (§4.5).
- **Starting point.** With equal weights, origin-relative lags were *worse*
  than target-relative lags (A vs B), so part of C's gain may only recover A's
  deficit.
- **Other weight sources.** Weights built from the incremental-benefit utility
  (G) were worse than B at every horizon. So was GradNorm applied to the
  climate group (H) (§4.4).

**Interpretation.**
- Whether measured weighting helps appears to depend on how the delay kernel is
  parameterised. This is a post-hoc, descriptive finding.
- The results give no support to measuring *incremental benefit* as the
  weighting signal. That utility was small and changed sign (§4.3), and the
  weights built from it varied widely between folds.
- GradNorm-style balancing of training rates did not help this encoder either.

**Possible explanation (untested) for the alignment effect.**
- With origin-relative kernels, all four horizons read the same lag positions.
  Their losses then compete directly over the same kernel parameters, so the
  balance between them could matter.
- With target-relative kernels, each horizon uses only delays of at least h
  weeks, renormalised over its own usable range. This partly separates what
  the horizons ask of the kernel. It may leave less conflict for the weighting
  to resolve.

This would be consistent with the interaction, but no experiment isolated it.

## 5.5 Relation to previous work

**Multi-task loss balancing.**
- GradNorm (Chen et al. 2018) and uncertainty weighting (Kendall et al. 2018)
  set task weights from quantities measured during training.
- Our design instead set them from an out-of-sample measurement of input
  reliance and applied them to one encoder.
- In this setting, neither our measured weights nor GradNorm improved on equal
  weighting. This is one comparison, on one small model and dataset. It does
  not bear on how these methods perform in the settings where they were
  proposed.

**Gradient routing.**
- F restricts the encoder to gradients from one horizon's loss. That is a
  simple instance of the broader idea in Gradient Routing (Cloud et al. 2024),
  which masks gradients so that chosen data update chosen parameters.
- Cloud et al. used routing to localise capabilities, not to improve forecast
  accuracy.
- Our lead is at most loosely consistent with the view that *which* signal
  trains a module can matter more than *how much* it is weighted. It does not
  test that view.

**Climate delays in dengue.**
- The learned rainfall kernels centre at 8–10 weeks. That is in the range of
  our own training-only scan (5–10 weeks) and of the 7–9-week lags that Liu
  et al. (2025) selected for Sri Lanka.
- Agreement with the per-district scan is weak, and the scan's own signal is
  below its resolvability threshold.
- The kernels are therefore compatible with the reported lags without being
  evidence that the model recovered a biological delay.
- Lowe et al. (2018) found that delays differ by climate variable. We report
  only the rainfall kernel, so the results do not address that finding.

**Persistence and horizon.**
- B loses to persistence at one week and beats it from two weeks (§4.1). This
  agrees with our earlier one-week study, on different weather data.
- It illustrates why a persistence baseline is needed at short horizons. That
  baseline is absent from the comparisons in DengueGNN (GulMohamed et al.
  2026), which was verified from the paper.
- We do not draw a comparison with published error values, because datasets,
  periods and targets differ.

## 5.6 Assumptions and limitations that affect interpretation

1. **Statistical power.** There are only seven headline test years, and they
   share training histories.
   - The bootstrap does not resolve dependence between folds.
   - The smallest attainable Wilcoxon p is 0.016.
   - Differences under about 0.3 MAE, the size of the frozen criteria, are
     hard to distinguish from zero. The primary null should be read as "no
     effect of useful size", not "exactly no effect".
2. **Seed sensitivity.** The sign of D − B changes with the seed subset
   (+0.07 with 5 seeds; about −0.04 with seeds 0–2). All exploratory
   contrasts use 3 seeds, so their magnitudes are uncertain.
3. **Post-inspection correction.** The protocol was frozen before scoring. A
   gradient-routing error was found and fixed after the results had been seen,
   and the affected arms were rerun. The primary null was the same before and
   after. However, the corrected results, and especially F, are evaluations on
   data that had already been inspected. 2026 is not a blind test either.
4. **What the utility measures.** Shuffle reliance is the degradation of a
   frozen equal-weight pilot when weather timing is broken. The method assumes
   this indicates where encoder gradients are useful. That assumption was not
   tested, and the reliance measure is not a causal effect of climate.
5. **Arbitrary design constants.** The weight rule (a 0.05 floor, mean-one
   normalisation) and the encoder learning rate were fixed, not tuned. The
   planned hyperparameter-sensitivity analysis was not run. A stronger weight
   rule might behave differently. F suggests this, but it was not tested.
6. **Data.** Weather is a reconstruction at district centroids, retrieved in
   2026. It is not what a forecaster would have had at each forecast origin.
   Reporting delays are not modelled, and the data cover one country.
7. **Evidence availability.** The corrected result files cited for most tables
   are not committed to the repository. Until they are, those numbers rest on
   the project documentation.

## 5.7 Unexpected findings

- **The null despite 2–4× weights.** Before the study, weights this far from
  uniform were expected to change the encoder noticeably. They changed each
  update by about 7° and 13%, and the trained models by less than seeds do.
- **Arm F improved every horizon, including h=1.** The expected pattern, if F
  worked at all, was gains mainly at long horizons.
- **GradNorm made the model slightly worse.** So did weights from the
  incremental-benefit utility.
- **The interaction with lag alignment.** Measured weighting behaved
  differently under origin-relative and target-relative lags. It helped only
  with origin-relative lags, which were themselves worse under equal weights.
- **The weighting moved the kernels without moving the forecasts.** Rainfall
  kernels shifted by about 0.7 weeks under D, yet predictions changed less
  than between seeds.

## 5.8 What can and cannot be concluded

| Can be concluded (within this model, data and protocol) | Cannot be concluded |
| --- | --- |
| Out-of-sample horizon weighting of the climate encoder's loss, at 1.7–4.0× weight ratios, gave no improvement of useful size over equal weighting, and was non-inferior at h=1–2 | That horizon-aware training of a climate encoder cannot help. F is an open lead. |
| The weighting changed each optimiser update only slightly, and trained models less than random seeds do | *Why* it had little effect. Gradient alignment, Adam normalisation, compensation by the heads and clipping are hypotheses. |
| Shuffle reliance rises with horizon on this data, so the premise of the method holds descriptively | That climate *causes* the forecast gains, or that the utility measures incremental benefit |
| Weights from incremental-benefit utilities, and GradNorm on the climate group, did not help in this setting | That GradNorm or loss balancing is ineffective in general |
| The effect of measured weighting appears to depend on lag alignment (descriptive, post-hoc) | That origin-relative weighting is a better method. The effect does not survive correction and is concentrated in 2017 and 2019. |
| The learned rainfall kernels centre at 8–10 weeks, in the range of reported lags | That the encoder recovered the biological delay |
| F is the best configuration observed (3 seeds, unadjusted) | That F improves forecasts. It needs a pre-declared test on untouched data (weeks after 2026-05-17 or an external dataset). |

**Summary for the Conclusion.**
- The proposed weighting did not improve forecasts. The null is well-defined:
  a competent baseline, weights far from uniform, frozen criteria.
- Its most useful product is a measurement of how weakly loss re-weighting
  alters training in this model.
- A more direct routing rule is a lead for future work. Its evidence is
  exploratory, and it should be presented that way.
