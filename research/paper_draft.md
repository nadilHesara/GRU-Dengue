# Does Horizon-Informed Gradient Weighting Help a Learnable Climate-Delay Encoder? A Leakage-Controlled Study of District-Level Dengue Forecasting in Sri Lanka

Kulathunge K. A. N. H., Angeesa R. P. T., Gunaweera N., Mallawarachchi H. S., Wijesekara M. S. T.
Department of Computer Science and Engineering, University of Moratuwa, Sri Lanka
{nadilk.23, thilokyaa.23, nethsithg.23, hesandim.23, senilkam.23}@cse.mrt.ac.lk
*[TA and supervisor names to be added]*

---

## Abstract

Weather affects dengue transmission weeks after it happens. So a forecaster
that learns weather delays should draw its training signal from horizons long
enough for those delays to matter. In a standard multi-horizon network,
however, the learnable delay encoder is trained by the losses of every
horizon, including those that barely use weather.

We test a training strategy that targets this mismatch. First, we measure how
much each forecast horizon relies on correctly timed weather, using pilot
models inside the training years. We then use those measurements to weight
the gradients that reach only the climate encoder. Every other part of the
network keeps its ordinary loss.

We evaluate the strategy on weekly dengue counts for 25 Sri Lankan districts,
at horizons of 1–4 weeks. The protocol is walk-forward, and the decision
criteria were frozen before scoring. The weighted model did not improve on an
identical equal-weight model: mean absolute error changed by +0.06 to +0.08
cases per district-week, within noise at every horizon. This held although the
weights differed by up to four times between horizons. Diagnostics show that
the weighting changes each update by only about 7° and 13%, and changes the
trained model less than a new random seed does.

In exploratory arms, training the climate encoder from the 4-week loss alone
was the best configuration. GradNorm-style balancing was worse than equal
weighting. We report the negative primary result and treat the routing
result as a lead that still needs confirmation.

---

## 1. Introduction

Dengue control in Sri Lanka depends on lead time. Vector control teams need
weeks to act, and hospitals need time to plan staff and ward capacity. The
2017 epidemic, the largest on record [2], showed the cost of reacting late. A
useful forecaster must therefore work several weeks ahead, not just one.

Weather is the obvious source of that lead time. Rainfall and temperature
affect mosquito breeding and viral development with a delay. In Sri Lanka,
climate lags of 7–9 weeks have been used to model district-level dengue [4].
But at one week ahead, the latest case count already carries almost all the
signal. In preliminary experiments on the same case data, removing every
climate input changed one-week error by only 0.01 cases per district-week. At
four weeks, scrambling the timing of the weather removed about half of the
model's advantage over persistence. The value of weather depends on the
horizon.

This creates a training problem. A multi-horizon network usually shares one
encoder across horizons and trains it on a combined loss. If that encoder is
meant to learn weather delays, most of its training signal comes from horizons
where weather timing hardly matters. Our preliminary results show the effect.
A learnable delay encoder trained at one week ahead learned delays unrelated
to those measured from the data, and it was less accurate than fixed
hand-chosen windows.

Existing tools do not target this case directly. Multi-task balancing methods
such as GradNorm [11] and uncertainty weighting [12] reweight task losses for
all shared parameters, using statistics gathered during training. Gradient
routing [14] restricts which data update which parameters, but it was proposed
for localising capabilities, not for forecasting. We did not find a study that
sets horizon weights from an out-of-sample measurement of input reliance and
applies them to one input encoder.

We ask a narrow question. Does weighting the gradients that reach only a
learnable climate encoder, by each horizon's measured reliance on climate,
improve multi-horizon dengue forecasts over an otherwise identical model
trained with equal weights?

To answer it, we build a two-branch forecaster for 1–4 weeks ahead. One branch
reads case history. The other reads weather through learnable per-district
delay kernels. We measure each horizon's climate reliance inside the training
years and route the weighted loss to the climate encoder only. We compare the
weighted model with an equal-weight twin under a walk-forward protocol, using
decision criteria fixed before any test year was scored.

Our contributions are:

1. **A controlled test of a training strategy, with a negative result.** We
   compare out-of-sample, horizon-specific gradient weighting of a single input
   encoder with an identical equal-weight model. The weighting gave no
   improvement of useful size, even with weights up to four times apart.
2. **Measurements of what the weighting changes.** We quantify its effect on
   each optimiser update and on the trained model, and compare both with
   seed-to-seed variation.
3. **Exploratory evidence about gradient routing.** Training the climate
   encoder from the longest horizon alone was the best configuration. GradNorm
   applied to the same encoder was not. These results are unconfirmed.

---

## 2. Related Work

**Neural dengue forecasting.** Recent dengue forecasters pair a recurrent
temporal encoder with spatial structure and climate covariates.
- Graph neural networks have been applied to Sri Lanka's 25 districts and
  reported to outperform ARIMA, random forests and LSTMs [2].
- DengueGNN combines dynamic graph convolution with attention-augmented LSTMs
  across several countries [3].
- A hybrid SEIR–LSTM model links climate and socio-economic drivers to dengue
  in 16 aggregated Sri Lankan regions [4].

These studies establish recurrent encoders as a natural backbone, and our
model uses one. They differ in how carefully they separate what each component
adds. DengueGNN, for example, reports no persistence baseline, and none of its
ablations removes the graph entirely [3]. We therefore compare against
persistence and against a matched control that differs from our model in the
training strategy only.

**Delayed climate effects.** Statistical epidemiology models delayed weather
effects with distributed lag non-linear models (DLNMs) [6]. These estimate how
risk varies jointly with exposure and lag. A DLNM study of dengue in Barbados
found delays that differ by variable: about five months for drought and one to
two months for heavy rainfall [7]. Neural forecasters usually fix lags before
training. One study picks lagged climate inputs for 27 Brazilian states and
explains them with SHAP [5]; another selects 7–9-week lags for Sri Lanka [4].
We take two ideas from this work. Delays can be long, and they vary by
variable. Our encoder therefore learns a separate smooth delay kernel for each
district and variable, much as a DLNM uses a smooth lag basis. Unlike these
approaches, it learns the delays inside the network, and that raises the
question of which horizon's loss should train them.

**Multi-horizon forecasting and count likelihoods.** Direct multi-horizon
models predict every horizon from one encoder, as in MQ-RNN [8] and the
Temporal Fusion Transformer [9]. DeepAR pairs a recurrent network with a
Negative Binomial likelihood for count data [10]. Our model follows both
choices: direct heads per horizon and a Negative Binomial output. Neither is a
contribution of this paper. What these models share is one combined loss over
horizons. We did not find work on *which* horizons should train a particular
input encoder.

**Loss balancing and gradient routing.** When tasks share parameters, their
losses can dominate one another or conflict. There are three common responses:
- *Adapt the loss weights during training.* GradNorm tunes gradient magnitudes
  so that tasks train at similar rates [11]. Uncertainty weighting learns
  weights from homoscedastic task uncertainty [12].
- *Alter the task gradients.* PCGrad projects away conflicting gradient
  components [13].
- *Restrict which gradients reach which parameters.* Gradient routing masks
  gradients during backpropagation, so that chosen data update chosen parts
  of the network [14].

Our method sits between the first and third families. Like GradNorm, it
weights task losses. Like gradient routing, it sends the weighted loss to a
chosen part of the network. It differs in where the weights come from: a
measurement made on held-out years inside the training period, rather than
statistics from the current training run. We include GradNorm as a comparison
arm, because it is the closest established method.

---

## 3. Methodology

### 3.1 Task

For forecast origin *t* and horizon *h* ∈ {1, 2, 3, 4} weeks, the model
predicts the case count *y*<sub>*t+h*</sub> for every district, using data up
to week *t*. All four horizons are predicted directly from the same origin.

### 3.2 Two-branch forecaster

Figure 1 shows the architecture. It has 10,028 parameters.

> **Figure 1 (to be created).** Two-branch architecture. Case history →
> per-district layers → case GRU. Weather → climate encoder (delay kernels →
> climate GRU). Per-horizon gated fusion → Negative Binomial head. A second
> panel should show the gradient routing: weighted loss → climate encoder
> only; equal-weight loss → all other parameters.

**Case branch.** Each district's last 12 weeks enter as nine channels:
- log cases;
- seasonal position (sine and cosine of day of year);
- observation flags;
- the district's coordinates;
- the district's weekly rank among all districts;
- its trailing 52-week case total.

Two per-district layers feed a GRU [15] with 32 units. The branch has no
spatial message passing. In our preliminary experiments, contiguity-based,
distance-based and learned graphs never beat this per-district control.

**Climate encoder.** Seven weather variables enter over 37 weeks: rainfall,
rainy-day fraction, temperature, diurnal temperature range, dew point,
relative humidity and wind speed.
- A delay kernel for each district and variable weights the weather history
  over delays of 0–29 weeks.
- Each kernel is a softmax mixture of six Gaussian bumps. The mixture weights
  come from a learned district embedding combined with a per-variable
  projection.
- We use softmax rather than a sparse alternative. In preliminary tests,
  sparsemax and entmax left most kernels with zero gradient.
- The smoothed weather streams pass through a shared climate GRU with 16 units.

We call the delay kernels and the climate GRU together the **climate encoder**.

**Target-relative delays.** Delays are measured from the target week *t+h*,
not the origin. For horizon *h*, the kernel is restricted to delays of at
least *h* weeks, so it uses only weather already observed at the origin, and it
is renormalised over that range. A learned delay of eight weeks therefore
means the same thing at every horizon. As a comparison, we also test
**origin-relative** delays: one kernel over lags before the origin, shared by
all horizons, with the same number of parameters.

**Fusion and likelihood.** For each horizon, the two branches produce
corrections δ<sub>case,*h*</sub> and δ<sub>clim,*h*</sub>. A gate
*g*<sub>*h*</sub> ∈ (0, 1) reads the case state, a district bias and a horizon
bias. The two corrections are combined as

  log μ<sub>*h*</sub> = log(1 + *y*<sub>*t*</sub>) + δ<sub>case,*h*</sub> + *g*<sub>*h*</sub> · δ<sub>clim,*h*</sub>,

where the sum of the two corrections is clipped to [−10, 10]. μ<sub>*h*</sub>
is the mean of a Negative Binomial (NB2) distribution. Its dispersion comes
from the case branch. A zero correction reproduces the latest count, plus one.
The point forecast is the NB median.

### 3.3 Climate-only gradient weighting

Let *L*<sub>*h*</sub> be the masked NB negative log-likelihood at horizon *h*,
and *a*<sub>*h*</sub> the share of observed cells at that horizon. We define
two losses:

  *L*<sub>equal</sub> = Σ<sub>*h*</sub> *a*<sub>*h*</sub> *L*<sub>*h*</sub>,
  *L*<sub>clim</sub> = Σ<sub>*h*</sub> *a*<sub>*h*</sub> *w*<sub>*h*</sub> *L*<sub>*h*</sub> / Σ<sub>*h*</sub> *a*<sub>*h*</sub> *w*<sub>*h*</sub>.

After one forward pass, the gradients are assigned by parameter group:
- **Climate encoder:** receives the gradient of *L*<sub>clim</sub>.
- **Everything else:** the case branch, the gate and all prediction heads
  receive the gradient of *L*<sub>equal</sub>. Every head is therefore trained
  by its own unweighted loss.

One global gradient-norm clip at 1.0 and one Adam step [16] follow. With all
*w*<sub>*h*</sub> = 1, the update is exactly the baseline update. We checked
this numerically.

### 3.4 Measuring horizon utility without test data

The weights must not use test data, so we measure them inside each fold's
training years:
1. We split the training period into three inner blocks. Each block has its
   own training span, early-stopping year and scoring year.
2. In each block, we train an equal-weight pilot model with two seeds.
3. We score each pilot twice on the scoring year. The first time uses the real
   weather. The second time, each district's weather window is replaced by
   the same district's window from another year, in the same season (±2
   weeks). We draw five such replacements.
4. The **shuffle reliance** of horizon *h* is the increase in error:
   *g*<sub>*h*</sub> = MAE<sub>shuffled</sub> − MAE<sub>real</sub>. We
   average it over seeds and blocks, then clip it at zero to get
   *u*<sub>*h*</sub>.

The weights are

  *w*<sub>*h*</sub> = 0.05 + 0.95 · 4*u*<sub>*h*</sub> / Σ<sub>*k*</sub> *u*<sub>*k*</sub>.

They have mean one, and each horizon keeps at least weight 0.05. Shuffle
reliance measures how much a trained model depends on correctly timed weather.
It is not a causal effect of climate.

### 3.5 Baselines and comparison arms

The **equal-weight model (B)** is the main baseline. It is identical to the
**utility-weighted model (D)** in every respect except the weights. This
isolates the training strategy. Table I lists the other arms. They separate
four questions:
- Does lag alignment matter?
- Do the weights need to be measured at all?
- Does a harder routing rule do better?
- How does an established balancing method compare?

**Table I.** Comparison arms. All arms train the prediction heads and the case
branch with the equal-weight loss.

| Arm | Delay alignment | What trains the climate encoder |
| --- | --- | --- |
| B (baseline) | target-relative | equal weights |
| D (proposed) | target-relative | shuffle-reliance weights |
| A | origin-relative | equal weights |
| C | origin-relative | shuffle-reliance weights |
| E | target-relative | fixed weights [1, 2, 3, 4] (normalised) |
| F | target-relative | the 4-week loss only |
| G | target-relative | weights from gain over a case-only pilot |
| H | target-relative | GradNorm [11], applied to the climate encoder |

We also report two external references, which are not controlled ablations:
- **persistence**, the forecast that repeats the latest count; at short
  horizons, this is the baseline any model must beat;
- **Chronos-2** [20], a pretrained time-series model applied zero-shot with no
  training on our data, to place our small models' accuracy in context.

---

## 4. Experimental Setup

**Data.** Weekly confirmed dengue counts come from the denguedatahub package
[1]. We merge 26 reporting areas into Sri Lanka's 25 districts, giving 1,012
weekly periods from December 2006 to May 2026. Daily weather comes from ERA5
reanalysis [17], retrieved through the Open-Meteo archive [18]. It is sampled
at each district's centroid and aggregated to reporting weeks.
- **Missing targets** are masked, never imputed.
- **Missing inputs** are filled from a training-period district × month
  climatology and flagged.

**Splits.** We use nine expanding walk-forward folds, with test years
2017–2025.
- Training data end two years before each test year. The following year is
  used for validation, and each cell is assigned to a split by its target
  week.
- Results average the seven **headline** years (2017–2019 and 2022–2025).
  The COVID years 2020–2021 are shown only in a robustness check.
- All methods are scored on the same 11,725 cells per horizon.
- The first 19 weeks of 2026 serve as a hold-out check.

**Training.** We use Adam with batch size 64 and at most 150 epochs.
- **Learning rates:** 2×10⁻² for the delay kernels, without weight decay.
  3×10⁻³ with weight decay 10⁻⁴ for all other parameters.
- **Early stopping:** patience 15, on the *unweighted* validation likelihood,
  so model selection does not depend on the weights under test.
- **Seeds:** the primary comparison uses five seeds per arm. The exploratory
  arms use three, and each is compared with B on the same three seeds.
- **Compute:** all 583 training fits took 93.9 GPU-minutes on one laptop GPU
  (RTX 4070 Laptop). One complete forecast for 25 districts and four horizons
  takes 2.4 ms.

**Metrics.** Mean absolute error (MAE, in cases per district-week) is the
primary metric. **Peak MAE** is MAE restricted to district-weeks at or above
the district's 90th percentile of pre-test history. It measures accuracy in
high-incidence weeks. A model's score is the mean of its per-seed scores.

**Statistics and decision criteria.** We test paired differences across the
seven years with t-tests and Holm correction across the four horizons [19].
Confidence intervals come from a block bootstrap over 4-week blocks of
origins, resampling all districts together. Before any test year was scored,
we fixed these criteria:
- **Improvement:** D − B ≤ −0.30 MAE at 3 or 4 weeks, with Holm-adjusted
  *p* < 0.05 and a 95% CI below zero.
- **Non-inferiority at 1–2 weeks:** the upper CI bound is at most +0.30.
- Otherwise: **no measurable difference**.

The other arms are exploratory, and their tests are not adjusted.

**Protocol history.** We froze the arms, seeds, folds, weights and criteria
before scoring any test year. After the first results had been inspected, we
found an implementation error: the climate correction head had received
weighted gradients instead of unweighted ones. We fixed it and reran the
affected arms under the same criteria.
- The primary conclusion was the same before and after the fix.
- All results below are from the corrected runs.
- Because the data had already been seen, these are corrected retrospective
  evaluations, not a blind test. The same applies to 2026.

---

## 5. Results

### 5.1 The baseline is competitive

Table II places the equal-weight model among the reference forecasts. B has a
higher MAE than persistence at one week. From two weeks on, its MAE is lower,
and at four weeks it is 3.2 cases per district-week below persistence. B is
within 0.25 MAE of a separately trained Negative Binomial GRU at 3–4 weeks.
It is within 0.7 of zero-shot Chronos-2 at every horizon. The primary
comparison is therefore made against a competent model.

**Table II.** Headline MAE (cases per district-week). Lower is better.

| Method | 1 wk | 2 wk | 3 wk | 4 wk | Peak MAE, 4 wk |
| --- | --- | --- | --- | --- | --- |
| Persistence | 16.42 | 20.16 | 24.58 | 28.47 | 47.98 |
| NB GRU, separate models (reference) | 15.94 | 19.26 | 22.46 | 25.07 | 42.60 |
| Chronos-2, zero-shot (reference) | 15.96 | 19.07 | 22.29 | 25.11 | 43.74 |
| Equal-weight model (B) | 16.64 | 19.67 | 22.71 | 25.22 | 42.99 |
| Utility-weighted model (D) | 16.70 | 19.75 | 22.78 | 25.28 | 42.81 |

### 5.2 The weights were far from uniform

Shuffle reliance was largest at four weeks in all ten folds. It increased
strictly with horizon in nine of them (Fig. 2). As a result, the weights
favoured the longest horizon by a factor of 1.7–4.0 over the shortest.
- *Example (a 2022 test fold):* the weights were 0.46, 0.81, 1.18 and 1.54
  for horizons 1–4.
- *The alternative utility (arm G)* measured the gain over a pilot without
  weather. It was small and changed sign across folds, from −0.03 to +0.12.

> **Figure 2.** Frozen per-horizon weights for each fold, compared with
> uniform weights. Path: `results/climate_horizon/analysis/figures/weights_vs_uniform.pdf`

### 5.3 Utility weighting did not improve the forecasts

Table III gives the primary result. D's error was slightly higher than B's
at every horizon. No difference was significant. By the frozen criteria:
- at 1–2 weeks, D is **non-inferior** to B;
- at 3–4 weeks, there is **no measurable difference**;
- the improvement criterion was not met.

The conclusion is the same whether we score the seed-averaged forecast
(+0.06), include the COVID years (+0.05 to +0.07), or lengthen the bootstrap
blocks. In the 2026 hold-out, D was again worse than B at every horizon
(+0.18 to +0.50). That check is consistent in sign but is not blind.

**Table III.** Primary comparison: D − B, headline MAE, five seeds. Negative
values favour D.

| Horizon | D − B | Holm *p* | 95% CI | Verdict |
| --- | --- | --- | --- | --- |
| 1 wk | +0.06 | 0.72 | [−0.03, +0.17] | non-inferior |
| 2 wk | +0.08 | 0.44 | [−0.03, +0.21] | non-inferior |
| 3 wk | +0.07 | 0.72 | [−0.07, +0.23] | no difference |
| 4 wk | +0.06 | 0.72 | [−0.10, +0.22] | no difference |

### 5.4 Exploratory arms

Table IV compares each exploratory arm with B on the same three seeds.
Three patterns stand out:
- **Arm F had the lowest error at every horizon.** In F, only the 4-week loss
  trains the climate encoder. Its advantage over B was 0.64–0.69 MAE, and F
  was better in five or six of the seven years. The CIs exclude zero at 1–2
  weeks ([−1.26, −0.20] at 1 week) but include it at 3–4 weeks ([−1.70,
  +0.41] at 4 weeks).
- **Weights from gain over a case-only pilot (G) and GradNorm (H) were both
  worse than B** at every horizon.
- **Lag alignment changed the effect of weighting.** With origin-relative
  delays, measured weighting (C) had lower MAE than B, while equal weighting
  (A) had higher MAE. With target-relative delays, D and B were equal.

In a post-hoc 2×2 analysis over all nine folds, the interaction between
weighting and alignment was +0.58 MAE at 4 weeks (95% CI [+0.26, +0.91]).
None of the fold-level tests survived Holm correction, and most of the effect
came from 2017 and 2019.

**Table IV.** Exploratory arms: change in headline MAE relative to B on the
same seeds. Negative values favour the arm. Tests are unadjusted.

| Arm | 1 wk | 2 wk | 3 wk | 4 wk |
| --- | --- | --- | --- | --- |
| A: origin delays, equal weights | +0.24 | +0.27 | +0.44 | +0.50 |
| C: origin delays, measured weights | −0.40 | −0.42 | −0.31 | −0.27 |
| E: fixed weights [1,2,3,4] | −0.06 | −0.11 | −0.09 | −0.12 |
| **F: 4-week loss only** | **−0.65** | **−0.69** | **−0.66** | **−0.64** |
| G: case-only-gain weights | +0.62 | +0.58 | +0.70 | +0.80 |
| H: GradNorm | +0.12 | +0.11 | +0.18 | +0.11 |

### 5.5 How much the weighting changes training

To see why D and B behave alike, we compared their gradients at 60 matched
training states: the same model, optimiser state and batch. The weighting
changed each update very little (Table V).
- The update to the climate encoder turned by about 7° on average, and its
  size changed by 13%.
- The raw gradient norm grew by 14.7%, but the Adam update norm by only 2.6%.
- In the trained models, B and D differ less from each other than two seeds
  of B do. This holds both for their forecasts and for their learned kernels.

**Table V.** Size of the change caused by utility weighting.

| Quantity | Value |
| --- | --- |
| Angle between D and B updates (mean per pair) | 7.1° (range 2.8–13.9°) |
| Relative update difference ‖*u*<sub>D</sub> − *u*<sub>B</sub>‖ / ‖*u*<sub>B</sub>‖ | 13.1% (range 5–24%) |
| Forecast difference: B vs D / two seeds of B | 1.7–2.9 / 3.5–5.2 cases |
| Rainfall-kernel L1 distance: B vs D / two seeds of B | 0.41 / 0.72 |

**What the encoder learned.** The learned rainfall kernels centre at 8–10
weeks. D's kernels sit about 0.7 weeks later than B's. Agreement with a
training-only cross-correlation scan of each district's rainfall and cases is
weak (Spearman 0.15–0.31). The climate gate rises with horizon, from 0.43 at
one week to 0.56 at four, and B and D differ by at most 0.02.

---

## 6. Discussion

**What the result means.** For this model and data, weighting the climate
encoder by measured horizon reliance does not improve forecasts. We can rule
out the two simplest explanations for the null. First, the intervention was
not trivial: the weights were up to four times apart (§5.2). Second, the
baseline was not weak: B beats persistence from two weeks and sits close to
the stronger references (§5.1). The null therefore says something about the
strategy itself, at least at this strength.

**Why the effect was small.** The diagnostics give a direct answer at one
level. The weighting changed each update less than a change of random seed
changes the whole run (Table V). A perturbation that small is unlikely to
produce an accuracy gain that seven test years could detect.

Why the change was so small is a separate question. Our diagnostics do not
settle it. Three explanations fit the evidence, and none was tested:
- **Partly aligned gradients.** In the same diagnostics, each horizon's encoder
  gradient already pointed partly in the direction of the 4-week gradient
  (cosine 0.48–0.68). Re-weighting a sum of partly aligned vectors can only
  rotate it a little.
- **Adam's per-parameter normalisation** may absorb much of any change in
  gradient scale. The gap between the +14.7% gradient change and the +2.6%
  update change fits this, but we did not isolate it, for example by
  training with plain SGD.
- **Compensation by the rest of the network.** The gate and the heads, trained
  on the unweighted loss, may adapt to whatever the encoder produces.

**Routing versus weighting.** Arm F removes the short-horizon gradients from
the climate encoder instead of down-weighting them. It was the best
configuration. This suggests that *which* losses reach the encoder may matter
more than *how much* each is weighted, which is in the spirit of gradient
routing [14]. But the evidence is weak:
- F had three seeds and unadjusted tests.
- Its intervals at 3–4 weeks include zero.
- We first saw its corrected form after the fix.

One feature is unexpected. F improved one-week forecasts by as much as
four-week forecasts, although one-week forecasts barely rely on weather. One
possible explanation is that an encoder trained on the long horizon learns a
smoother weather summary, which the one-week head, trained separately, can
still use. Another is that removing three loss terms simply regularises the
encoder. We tested neither.

**Lag alignment.** Measured weighting helped only with origin-relative
delays. With those delays, all horizons read the same lag positions and
compete over the same kernel parameters. Balancing their losses could then
matter. With target-relative delays, each horizon uses its own part of the
kernel, which may leave little conflict to resolve. This is a hypothesis. The
interaction does not survive correction and depends mostly on two years.
Origin-relative delays were also *worse* under equal weights, so part of the
gain from weighting may only recover that deficit.

**Relation to previous work.** Loss-balancing methods set weights from
statistics gathered during training [11], [12]. Our weights came from
held-out reliance instead, and neither approach helped this encoder: GradNorm
(H) was slightly worse than equal weighting. This is one comparison on one
small model. It says nothing about these methods in the settings they were
designed for.

The learned kernels centre at 8–10 weeks. That is close to the 7–9-week lags
chosen for Sri Lanka in [4]. However, the kernels agree only weakly with our
per-district scan, so we treat them as *learned delay kernels*, not as a
recovered biological delay. We report only rainfall, so the variable-specific
delays found in [7] remain untested here.

Finally, B loses to persistence at one week and wins from two weeks. Like our
preliminary results, this supports routinely including persistence in dengue
forecast evaluation, which [3] does not.

---

## 7. Limitations

- **Low power.** Seven headline years, which share training histories, cannot
  resolve differences much smaller than 0.3 MAE. The primary null means "no
  effect of useful size", not "no effect".
- **Seed sensitivity.** With only the first three seeds, the sign of D − B
  reverses (about −0.04). All exploratory arms rest on three seeds.
- **Not a blind test.** The protocol was frozen before scoring. However, the
  implementation error was fixed after the results had been inspected, and all
  data through May 2026 had been seen. Arm F, in particular, needs a
  pre-declared test on untouched data, such as weeks reported after May 2026
  or another country's surveillance series.
- **Untested assumptions.** We assume that shuffle reliance indicates where
  encoder gradients are useful. The weight formula and learning rates were
  fixed rather than tuned, and a stronger weighting rule might behave
  differently.
- **Data.** Weather is reanalysis sampled at district centroids and retrieved
  in 2026. It is not what a forecaster would have had at each origin.
  Reporting delays are not modelled. The study covers one country.

---

## 8. Conclusion

We tested whether weighting a learnable climate encoder's gradients by each
horizon's measured reliance on weather improves multi-horizon dengue
forecasts. Under decision criteria fixed before scoring, it did not. The
weighted model matched an identical equal-weight model at every horizon, even
with weights up to four times apart. Our diagnostics show why the
intervention was weak: it changes each update less than a new random seed
changes the whole run.

Two leads came from the exploratory arms. Training the climate encoder from
the longest horizon alone was the best configuration, and measured weighting
helped only with origin-relative delays. Both need confirmation on data that
nobody has yet seen. For researchers training small multi-horizon models, the
practical lesson is to measure how much a training intervention changes the
updates, and compare that change with seed-to-seed variation, before
expecting it to change accuracy.

---

## References

[1] T. S. Talagala, "denguedatahub: A tidy format datasets of dengue by country," R package version 4.1.0, 2026, doi: 10.32614/CRAN.package.denguedatahub.

[2] J. Weng *et al.*, "Graph representation learning for dengue forecasting," in *Proc. IEEE Int. Conf. Big Data*, 2024, pp. 4448–4456, doi: 10.1109/BigData62323.2024.10825335.

[3] R. B. GulMohamed, W. Hetany, H. A. Almaimani, and F. A. S. Khiery, "DengueGNN: Graph-based deep learning for modeling disease spread dynamics and prediction," *Sci. Rep.*, vol. 16, art. 10584, 2026, doi: 10.1038/s41598-026-43073-y.

[4] Y. Liu, P. Fransson, J. Heidecke, P. Liyanage, J. Wallin, and J. Rocklöv, "An explainable covariate compartmental model for predicting the spatio-temporal patterns of dengue in Sri Lanka," *PLoS Comput. Biol.*, vol. 21, no. 9, e1013540, 2025.

[5] X. Chen and P. Moraga, "Forecasting dengue across Brazil with LSTM neural networks and SHAP-driven lagged climate and spatial effects," *BMC Public Health*, vol. 25, art. 973, 2025.

[6] A. Gasparrini, B. Armstrong, and M. G. Kenward, "Distributed lag non-linear models," *Stat. Med.*, vol. 29, no. 21, pp. 2224–2234, 2010.

[7] R. Lowe *et al.*, "Nonlinear and delayed impacts of climate on dengue risk in Barbados: A modelling study," *PLoS Med.*, 2018, doi: 10.1371/journal.pmed.1002613.

[8] R. Wen, K. Torkkola, B. Narayanaswamy, and D. Madeka, "A multi-horizon quantile recurrent forecaster," NIPS Time Series Workshop, 2017, arXiv:1711.11053.

[9] B. Lim, S. O. Arik, N. Loeff, and T. Pfister, "Temporal fusion transformers for interpretable multi-horizon time series forecasting," arXiv:1912.09363, 2019.

[10] D. Salinas, V. Flunkert, J. Gasthaus, and T. Januschowski, "DeepAR: Probabilistic forecasting with autoregressive recurrent networks," *Int. J. Forecast.*, vol. 36, no. 3, pp. 1181–1191, 2020.

[11] Z. Chen, V. Badrinarayanan, C.-Y. Lee, and A. Rabinovich, "GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks," in *Proc. ICML*, PMLR 80, 2018.

[12] A. Kendall, Y. Gal, and R. Cipolla, "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics," in *Proc. IEEE CVPR*, 2018.

[13] T. Yu, S. Kumar, A. Gupta, S. Levine, K. Hausman, and C. Finn, "Gradient surgery for multi-task learning," in *Proc. NeurIPS*, 2020.

[14] A. Cloud, J. Goldman-Wetzler, E. Wybitul, J. Miller, and A. M. Turner, "Gradient routing: Masking gradients to localize computation in neural networks," arXiv:2410.04332, 2024.

[15] K. Cho *et al.*, "Learning phrase representations using RNN encoder–decoder for statistical machine translation," in *Proc. EMNLP*, 2014, pp. 1724–1734.

[16] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. ICLR*, 2015.

[17] J. Muñoz-Sabater *et al.*, "ERA5-Land: A state-of-the-art global reanalysis dataset for land applications," *Earth Syst. Sci. Data*, 2021, doi: 10.5194/essd-13-4349-2021.

[18] Open-Meteo, "Historical weather API documentation," 2026. [Online]. Available: https://open-meteo.com/en/docs/historical-weather-api

[19] S. Holm, "A simple sequentially rejective multiple test procedure," *Scand. J. Stat.*, vol. 6, no. 2, pp. 65–70, 1979.

[20] A. F. Ansari *et al.*, "Chronos-2: From univariate to universal forecasting," arXiv:2510.15821, 2025.
