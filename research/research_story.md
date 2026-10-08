# Research story: AEGIS-Dengue

**What this is.** The logical storyline for the paper. It is not the paper,
and not polished prose. Each step follows from the one before it:

> Problem → Gap → Research question → Method → Evaluation → Findings → Contribution

**Basis.** Every claim comes from [research_evidence.md](research_evidence.md),
which cites the repository files. Experiment numbers (#1, #2, …) refer to its
evidence table (§16). The study tags are the ones used there:
- **[S1]** the earlier one-week studies;
- **[S2]** the long-horizon benchmark;
- **[S3]** the climate-horizon extension.

**Assumption.** The paper is built around [S3], the climate-horizon training-strategy
study, as recommended in research_evidence.md §0. That choice is not yet
confirmed by the team. If the team chooses the [S2] benchmark instead, this
story does not apply.

---

## The story in one paragraph

A small multi-horizon neural forecaster for weekly dengue cases in 25 Sri Lankan
districts includes a learnable encoder that decides how far back to look at
weather. Weather only helps forecasts several weeks ahead. At one week ahead,
the latest case count carries almost all the signal. Yet the climate encoder is
trained by the losses of all horizons together. When a similar encoder was
trained only at one week ahead, it learned delays unrelated to the measured
ones. We therefore asked whether weighting the encoder's gradients by each
horizon's measured reliance on climate helps. Every other part of the model
keeps its ordinary loss. The weights are measured without touching test data.
Under a protocol frozen before scoring, it did not help: the weighted model
matched the equal-weight model at every horizon. We measured why the
intervention was so weak: it changes each update only slightly, less than
changing the random seed does. A simpler routing rule was the best
configuration, but only in exploratory analysis. The contribution is a
controlled, well-instrumented negative result about a training strategy. It is
not a new algorithm that improves forecasts.

---

## 1. Research problem

- **The setting.** We forecast weekly dengue cases for 25 Sri Lankan districts
  1–4 weeks ahead. The model is a small neural network that combines past case
  counts with past weather.
- **The difficulty.** Weather affects dengue with a delay of several weeks.
  But the value of weather differs sharply by horizon. At one week ahead the
  latest case count carries almost all the signal. At four weeks ahead weather
  timing matters.
- **The problem this creates for training.** A shared network component that
  learns how to delay weather is trained by the losses of all four horizons at
  once. Only some of those losses carry information about weather timing.

## 2. Motivation

**Why dengue lead time matters**
- District-level warnings several weeks ahead would give the NDCU and Public
  Health Inspectors time for vector control. They would also give hospitals
  time to prepare staff and ward capacity. (Source: `Research_Paper_New.md` §3.)
- One week of lead time is too short for either purpose.

**Why the training question matters, from our own earlier results**

1. **At one week ahead, the ceiling is information, not the model.** [S1]
   - No architecture, loss, optimiser or hyperparameter change reliably beat
     persistence (repeating the latest count). Each apparent gain came from the
     2017 epidemic year alone (#14–#19).
   - Removing all climate inputs at one week ahead cost only +0.01 MAE.
2. **Weather timing becomes useful further ahead.**
   - At 3–4 weeks ahead the model beats persistence: +6.9% and +10.8% skill (#12).
   - We shuffled the weather in time while keeping the model's size fixed.
     That cost +0.02 MAE at one week but +1.55 MAE at four weeks (p = 0.016),
     removing half of the four-week skill (#11).
   - The same pattern appears in three model families in the benchmark (#22).
3. **A learnable delay encoder trained at one week ahead learned nothing useful.** [S1]
   - Its learned delays were unrelated to the delays measured directly from
     the data (r = −0.16).
   - It was less accurate than fixed hand-chosen windows: 20.67 vs 18.36 MAE (#13).

Taken together, these results suggest the encoder may fail because most of its
training signal comes from horizons where weather timing doesn't matter. This
suggestion is what motivates the study. It is a hypothesis, not a measured
fact.

## 3. What existing approaches do

| Approach | What it does | Relevance here | Citation status |
| --- | --- | --- | --- |
| Sri Lankan dengue deep learning (Weng et al. 2024: spatio-temporal graph networks; DengueGNN, GulMohamed et al. 2026; Gogovi 2026: LSTM vs graph networks) | District-level neural forecasters, mostly at one horizon | Some report gains without a persistence baseline or an identity-graph control | In the team bibliography; Gogovi is only in the `full paper.tex` bibliography. All must be checked. |
| Climate-lag modelling (fixed rolling windows; distributed-lag non-linear models; Liu et al. 2025, SEIR–LSTM with 7–9-week lags) | Delays are fixed by hand, or fitted once under a single objective | No mechanism for making the delay depend on the forecast horizon | Liu is in the team bibliography. Distributed-lag models are marked "do not cite until verified". |
| Multi-task loss balancing (**GradNorm**, Chen et al. 2018; also Kendall et al. 2018 and PCGrad) | Reweights task losses for **all shared parameters**, using statistics computed **during training** | The closest family of methods | Only GradNorm is **verified**. The others are marked "do not cite until verified". |
| Count likelihoods on recurrent networks (DeepAR) | A Negative Binomial output on a recurrent network | Our Negative Binomial head is not new in itself | Marked "do not cite until verified" |

## 4. The gap

The gap is narrow and specific. It is not a claim that the area is unstudied.

1. **Loss weighting targets the whole network, not one input.** Methods such as
   GradNorm decide how much each task's loss counts for all shared parameters,
   using signals gathered during training. We have not verified any method that
   (a) weights horizon losses for *one input modality's encoder only*, and
   (b) sets those weights from an *out-of-sample* measurement of how much each
   horizon relies on that input.
2. **Climate delays in dengue models are fixed, or learned under one objective.**
   Our own attempt to learn them end-to-end at one week ahead failed (§2).
3. **Dengue deep-learning evaluations are often weak.** They often omit a
   persistence baseline, matched controls and frozen decision criteria. A
   training-strategy claim therefore needs a stricter test than is usual in
   this area.

**Wording limit.** The brief states that the literature has not been checked
thoroughly enough to claim this question has never been studied
(`Research_Paper_New.md` §4.2). Present it as the question this paper
investigates.

## 5. Research question

> In a multi-horizon dengue forecaster, does weighting the gradients that
> reach **only** a learnable climate-delay encoder help? The weights come from
> each horizon's measured, out-of-sample reliance on climate. The comparison is
> an otherwise identical model trained with equal horizon weights.

**Hypothesis.** Weighting the encoder toward the horizons that rely on climate
should help it learn useful delays and lower forecast error at 3–4 weeks. At
1–2 weeks it should not make forecasts worse.

**Frozen success criteria**, fixed before scoring:
- **Improvement:** the weighted model's MAE is at least 0.30 lower at h = 3
  or 4, with Holm-adjusted p < 0.05 and a 95% confidence interval below 0.
- **Non-inferiority at h = 1–2:** the upper end of the 95% confidence interval
  is at most +0.30 MAE.
- Otherwise the result is "no measurable difference".

## 6. Proposed approach

**What kind of contribution this is.** A modification of the training
strategy, built on a standard architecture. It is not a new model family and
not a new optimiser.

### 6.1 Architecture

A two-branch Negative Binomial forecaster with 10,028 parameters.

- **Case branch.** A GRU that reads 12 weeks of case history. It has no graph:
  the graph variants we tested never beat this control (#18).
- **Climate branch.** A learnable delay encoder, `HorizonLagEncoder`, followed
  by a small climate GRU.
  - The encoder learns a separate delay kernel for each district and weather
    variable, covering delays of 0–29 weeks.
  - Each kernel is a softmax mixture of 6 Gaussian bumps. Softmax is used
    because sparser alternatives froze 58–74% of kernels (#19).
  - The kernel is "target-relative": for horizon h it only uses weather
    already observed at the forecast origin, i.e. delays of at least h weeks.
- **Output.** For each horizon, the climate correction passes through a
  learned gate and is added to the case correction. The sum shifts the forecast
  away from the latest observed count, so a zero correction means "no change"
  (y_t + 1). The output distribution is Negative Binomial. A Negative Binomial
  head matched the quantile head in the benchmark (#20).

### 6.2 Training strategy: the intervention

- **The weighted loss** (weights w_h) trains **only** the climate encoder and
  the climate GRU.
- **The ordinary equal-weight loss** trains **everything else**, including
  every prediction head.
- Both gradients come from a single forward pass. They are followed by one
  gradient clip at norm 1.0 and one Adam step.
- The weights have mean one. With all weights equal to 1, the update is
  exactly the baseline update; this was checked numerically.

### 6.3 Measuring the weights without touching test data

1. Within each fold's training years, train frozen equal-weight "pilot" models
   on 3 inner blocks.
2. Score each pilot twice: once with real weather, and once with each
   district's weather swapped for its window from another year in the same
   season.
3. A horizon's utility is the resulting increase in MAE ("shuffle reliance").
4. The weights are computed from these utilities as
   w_h = 0.05 + 0.95 · 4u_h / Σu.

## 7. Experimental comparison

**Primary comparison (declared in advance)**
- D (measured weights) vs B (equal weights).
- The two models are identical in every other respect.
- 5 seeds and 7 headline test years (2017–19 and 2022–25), at h = 1–4.
- p-values are Holm-adjusted over the 4 horizons.
- 95% confidence intervals come from a 4-week block bootstrap.

**Planned exploratory arms** (3 seeds each; compared with B on matched seeds 0–2)

| Arm | What changes | What it isolates |
| --- | --- | --- |
| A | Lags measured from the forecast origin, not the target; equal weights | Lag alignment |
| C | Origin-relative lags with measured weights | Lag alignment |
| E | Fixed weights [1, 2, 3, 4] | Whether measuring the weights matters |
| F | Only the 4-week loss trains the climate encoder | Extreme routing |
| G | Weights from a different utility measure (gain over a case-only model) | The source of the utility |
| H | GradNorm applied to the climate group | The nearest existing method |

**Post-hoc analyses** (designed after the results were seen, and labelled as such)
- A 2×2 factorial of lag alignment × weighting.
- Gradient and update diagnostics on 60 pairs of B and D in identical states.
- Inspection of the learned kernels and gates.
- A matched-seed reanalysis.
- A block-length sensitivity check.

**Context rows** (not controlled ablations)
- Persistence, rescored at each horizon.
- Three reference forecasts stored from the benchmark.
- Chronos-2, run zero-shot.

**Protocol disclosure**
- The protocol was frozen before scoring.
- After inspection, a gradient-routing bug was found: the climate correction
  head was wrongly in the weighted group. It was fixed and the affected arms
  were rerun.
- 2026 is a frozen-specification check, not a blind test.

## 8. Most important findings

1. **Primary result: no benefit.**
   - D − B = +0.06 to +0.08 MAE at every horizon (Holm p ≥ 0.44).
   - The criteria verdict: non-inferior at 1–2 weeks, no measurable difference
     at 3–4 weeks.
   - The weights were far from uniform (1.7–4.0× between horizons), so the
     intervention was not trivially small.
   - The same holds for the ensemble estimand, for all 9 folds, and in the
     pre-correction run. (#1, #3, #5)
2. **The weighting changes training only slightly.** In 60 comparisons at
   identical model states:
   - the encoder's update direction rotated by about 7–8°;
   - its size differed by about 13%;
   - trained B and D differ less than two seeds of B do. (#4)
3. **Exploratory, unconfirmed: how gradients are routed mattered more than
   weighting them.**
   - Training the climate encoder only from the 4-week loss (F) gave −0.64 to
     −0.69 MAE vs B, better in 5–6 of 7 test years. This is unadjusted, uses 3
     seeds, and the confidence intervals at 3–4 weeks include 0.
   - Measured weighting helped only with origin-relative lags: C vs A is −0.64
     at h=4 (a post-hoc interaction).
   - GradNorm on the climate group (H) and the alternative utility (G) made the
     model worse. (#2, #6)
4. **The premise holds: weather timing matters only further ahead.**
   - The shuffle cost is +0.02 MAE at one week and +1.55 at four weeks.
   - The pattern replicates across model families.
   - The learned rainfall kernels centre at 8–10 weeks but agree only weakly
     with the measured delays (Spearman 0.15–0.31). (#7, #11, #22)
5. **Supporting finding: at one week ahead, the ceiling is information, not
   optimisation.** None of the following gave a reliable gain; each apparent
   gain was the 2017 epidemic year alone:
   - hyperparameter search: the response surface was flat;
   - AdamW and plateau scheduling: the scheduler cut the learning rate only
     after the best epoch had been saved, in 23 of 27 runs;
   - gradient clipping: the largest gradient norm was 2.87, so the clip never
     mattered;
   - LayerNorm and alternative activations. (#14–#17)

## 9. What the findings actually demonstrate

**Demonstrated, within this data, model and protocol**
- At these weight magnitudes (up to 4×), per-horizon loss weighting of a
  climate-only encoder gives no measurable accuracy change against equal
  weighting.
- The weighting's effect on each update is small compared with seed-to-seed
  variation. The measured angles, update sizes and the gap between B and D
  quantify this.
- Weather's predictive value depends on the horizon. It is near zero at one
  week and substantial at four. This is shown with a control that holds model
  capacity fixed.
- At one week ahead, training-procedure choices are not what limits accuracy.

**Not demonstrated. Do not claim.**
- That horizon-selective encoder training is useless. F points the other way,
  but it is unconfirmed.
- *Why* the weighting failed. Adam's per-parameter normalisation, rare
  clipping and seed noise are hypotheses that were not tested.
- That weather *causes* the gain. The shuffle measures reliance on correctly
  timed weather, not a causal effect.
- That the encoder recovered the biological delay. The correct wording is
  "learned delay kernel".
- Generalisation beyond Sri Lanka, or to truly unseen data. All data up to
  2026-05-17 had been inspected.
- Calibration. Say "empirical 95% coverage (0.93–0.96)".

## 10. Realistic contribution

**This work does not introduce a fundamentally new algorithm.** Weighting task
losses is established (GradNorm), and a Negative Binomial head on a recurrent
network is known. The contributions are methodological, experimental and
specific to this application:

1. **Methodological/experimental: a controlled test of a specific training
   strategy, with a negative primary result.**
   - The strategy routes horizon-weighted gradients to one input encoder, with
     weights measured out-of-sample inside the training years.
   - It is tested against an identical control, under criteria frozen before
     scoring.
   - The routing correction is fully disclosed.
   - The result: non-inferior, but no improvement.
2. **Diagnostic: a measured account of why the intervention was weak.** It
   reports the size of the change in updates and in trained models, compared
   with seed variance. This is descriptive, not causal.
3. **Architectural and application evidence for small dengue forecasters,
   within this setting:**
   - an identity backbone matches or beats the graph variants tested;
   - a softmax delay-kernel mixture avoids the dead-kernel failure of sparse
     alternatives;
   - weather's value rises with the forecast horizon;
   - at one week ahead, optimisation choices are not the constraint.
4. **An exploratory lead, stated as one:** restricting climate-encoder learning
   to the longest horizon. It needs confirmation on untouched data, either
   Sri Lankan weeks published after 2026-05-17 or an external dataset.

**How to phrase the contribution in one sentence.** *"We test whether
out-of-sample, horizon-specific gradient weighting helps a learnable
climate-delay encoder in multi-horizon dengue forecasting. Under a frozen
protocol it does not, and we measure why the intervention is weaker than the
nominal weights suggest."*
