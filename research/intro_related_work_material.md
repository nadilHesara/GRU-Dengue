# Material for the Introduction and Related Work

**What this is.** The facts, citations and limits to use when writing the
Introduction and Related Work. It is **not** the prose. The paragraph plan
follows the requested structure:
- Introduction: problem → existing approach → gap → what we do → contributions.
- Related Work: grouped by research direction.

**Basis.**
- Project facts come from [research_evidence.md](research_evidence.md) and
  [research_story.md](research_story.md). Numbers are cited by experiment (#n)
  from the evidence table.
- Citations were checked against their sources on **2026-09-30** (Part C).
- Study tags: **[S1]** the earlier one-week study; **[S2]** the long-horizon
  benchmark; **[S3]** the climate-horizon extension.
- The framing assumes the paper is about the [S3] training-strategy study.

---

## Part A. Introduction (target 500–700 words)

### Paragraph 1. The problem and why it matters

**Purpose:** establish the forecasting task and why forecasts several weeks
ahead, rather than one, are the useful target.

Facts to use:
- **Task:** weekly confirmed dengue cases for 25 Sri Lankan districts, 2006–2026,
  from `denguedatahub` (Talagala 2026).
- **Operational need:** vector control by the NDCU and Public Health Inspectors,
  and hospital staffing and ward capacity, both need lead times of several weeks
  (`Research_Paper_New.md` §3).
- **Sri Lanka's 2017 outbreak** is the largest on record. The Weng et al. (2024)
  abstract states this directly. That abstract is verified; see Part C.
- **Weather acts on dengue with a delay:**
  - Liu et al. (2025) use 7–9-week lags for precipitation and temperature in
    Sri Lanka (verified).
  - Our own training-only scan finds a 5–10-week lag association. It is weak:
    below the scan's resolvability threshold.
- **Horizon matters.** At one week ahead the latest count carries almost all
  the signal: removing all climate inputs costs +0.01 MAE [S1]. Skill over
  persistence appears only at 3–4 weeks (#12).

Wording: do not open with a generic "dengue is a global threat" sentence. Open
with the operational need for weeks of lead time.

### Paragraph 2. The existing approach

**Purpose:** describe how neural dengue forecasters are usually built and how
they handle climate.

Facts to use:
- **Neural forecasters** for Sri Lanka and elsewhere combine a recurrent
  temporal encoder with spatial components. Examples: GNNs for 25 Sri Lankan
  districts (Weng et al. 2024); dynamic graph convolution with attention-LSTMs
  (DengueGNN, GulMohamed et al. 2026); LSTMs with lagged climate and
  neighbouring-state inputs (Chen & Moraga 2025).
- **Climate delay is usually handled in one of two ways:**
  - lags are fixed or selected before training, as inputs (Chen & Moraga 2025;
    Liu et al. 2025, which selects 7–9 weeks);
  - lags are estimated with distributed-lag models in statistical frameworks
    (Gasparrini et al. 2010; Lowe et al. 2018).
- **Multi-horizon neural forecasters** predict several horizons directly from
  one encoder (MQ-RNN, Wen et al. 2017; TFT, Lim et al.). They are trained on
  a combined loss over horizons.

### Paragraph 3. The gap

**Purpose:** show that a learnable delay encoder, shared across horizons, has a
training-signal problem, and that current loss-balancing tools do not target it.

The argument, with evidence:
1. **The signal comes from the wrong horizons.** A learnable climate-delay
   encoder in a multi-horizon model is trained by all horizons' losses.
   However, weather timing is informative only at longer horizons. The
   capacity-matched shuffle cost is +0.02 MAE at h=1 and +1.55 at h=4,
   p = 0.016 (#11).
2. **Our own evidence of the failure.** Trained end-to-end at h=1, a learnable
   delay encoder learned delays unrelated to measured ones (r = −0.16) and lost
   to fixed windows, 20.67 vs 18.36 MAE (#13).
3. **Existing loss-balancing methods weight differently.** GradNorm (Chen et
   al. 2018) and uncertainty weighting (Kendall et al. 2018) adjust loss
   weights using signals from *during training*. PCGrad (Yu et al. 2020)
   projects away conflicting task gradients. None of them uses an
   *out-of-sample measurement of how much each task relies on one input*.
4. **Gradient routing exists, but not for this use.** Gradient Routing (Cloud
   et al. 2024) already restricts which parameters a given part of the data
   updates. It is used for localising capabilities and for unlearning, not for
   horizon weighting in forecasting.

**Honest framing.** The gap is a combination of existing ideas (per-parameter
gradient routing, task weighting, an out-of-sample utility) applied to one
input encoder. We have not shown that the combination is unstudied. Say "we
investigate whether…", not "no prior work…".

### Paragraph 4. What this study does

**Purpose:** state the question, method and test in two or three sentences,
without implementation detail.

Facts to use:
- **Question:** does weighting the gradients reaching *only* the climate-delay
  encoder help? The weights come from each horizon's measured, out-of-sample
  reliance on climate. The comparison is an identical model trained with equal
  weights.
- **Model:** a two-branch Negative Binomial forecaster for h = 1–4. It has a
  case GRU and a learnable, target-relative climate-delay encoder. The
  prediction heads keep their ordinary, unweighted loss.
- **Weights:** measured inside the training years, from how much a frozen
  pilot model degrades when its weather is swapped for the same season in
  another year.
- **Test:**
  - walk-forward evaluation over 7 headline test years, 5 seeds;
  - decision criteria frozen before scoring;
  - exploratory arms, including GradNorm on the climate group.
- **Disclosure:** a routing error was corrected after the results had been
  inspected. Mention it briefly here; it is detailed in the Experimental Setup.

### Closing: contributions (from `research_story.md` §10)

1. **A controlled test of a training strategy, with a negative primary result.**
   Horizon-specific, out-of-sample gradient weighting routed to a single input
   encoder, tested against an identical equal-weight control under frozen
   criteria. Result: non-inferior at h=1–2 and no measurable difference at
   h=3–4 (D − B = +0.06 to +0.08 MAE). This held even though the weights were
   far from uniform (1.7–4.0×).
2. **Measurements of how little the weighting changes training.**
   - About 7–8° of update rotation and a 13% relative difference in update
     size.
   - Trained models differ less than two seeds do.
   - This describes the effect; it does not explain it.
3. **Exploratory evidence about routing.** Restricting climate-encoder learning
   to the longest horizon was the best configuration (−0.64 to −0.69 MAE,
   unadjusted, 3 seeds). GradNorm-style balancing was worse. This lead needs
   confirmation.

### Do not put in the Introduction

- State-of-the-art claims. Nothing here is verified as state of the art.
  Chronos-2 zero-shot and the stored references are roughly level with B.
- The [S2] benchmark results (h up to 12, onset detection, conformal intervals).
- The old Negative Binomial "beats persistence 7/7" numbers, which are
  superseded.
- "Recovers the biological delay", "causal effect of climate", "calibrated".

---

## Part B. Related Work: four directions

Each direction gives:
1. the problem it addresses;
2. the methodological idea;
3. representative studies;
4. what we take from it;
5. the gap, **only where the source supports it**.

### B1. Neural and spatio-temporal dengue forecasting

1. **Problem.** District or regional forecasting of dengue case counts.
2. **Idea.** Recurrent temporal encoders (LSTM/GRU), often combined with graph
   convolution or attention over regions, plus climate and other covariates.
3. **Studies (all verified):**
   - Weng et al. 2024: GNNs for 25 Sri Lankan districts, 2013–2022; reported
     to outperform ARIMA, random forest and LSTM.
   - DengueGNN (GulMohamed et al. 2026): dynamic graph convolution with
     attention-augmented LSTMs on OpenDengue data.
   - Liu et al. 2025: SEIR–LSTM hybrid over 16 aggregated Sri Lankan regions.
   - Karasinghe et al. 2024: a statistical ARIMA reference for Sri Lanka
     (title and DOI in the bibliography; content not re-checked).
4. **What we take.** A small recurrent encoder is the standard temporal
   backbone, and our GRU follows it. Our own evidence justifies **dropping
   graph convolution**: identity beats contiguity, Gaussian and learned graphs
   (#18).
5. **Supportable gaps. Stay within these.**
   - DengueGNN compares 10 baselines but includes **no persistence or naive
     baseline**. Its ablations remove the dynamic graph, the attention and the
     mobility features, but **do not remove the graph entirely**. Verified
     from the paper (PMC).
   - The Weng et al. **abstract** does not mention a persistence baseline or
     state a horizon. **Do not** say the paper lacks persistence unless someone
     checks the full text. `full paper.tex` claims its baseline table omits
     persistence; that claim is unverified here.
   - Liu et al. reset immunity to zero at the start of 2017 for a new strain
     (verified). The earlier draft calls this "retrospective knowledge". That
     is our interpretation; label it as ours if used.
   - **Gogovi 2026** ("LSTM beats graph variants") **could not be found**. Do
     not cite it until someone obtains the paper.

### B2. Modelling delayed climate effects

1. **Problem.** Weather affects dengue transmission after a delay, and the
   response can be nonlinear.
2. **Idea.**
   - Statistical approach: distributed-lag non-linear models (DLNMs) estimate
     a joint exposure–lag–response surface.
   - Machine-learning approach: lags are chosen before training and supplied as
     features.
3. **Studies (all verified):**
   - Gasparrini et al. 2010: the DLNM framework (illustrated on temperature and
     mortality).
   - Lowe et al. 2018: DLNMs with a hierarchical mixed model for dengue in
     Barbados. Drought raised risk at about 5 months, excess rainfall at 1–2
     months.
   - Liu et al. 2025: 7–9-week precipitation and temperature lags for Sri Lanka.
   - Chen & Moraga 2025: an LSTM for 27 Brazilian states with optimal lagged
     climate variables and neighbouring-state influence, interpreted with SHAP.
4. **What we take.**
   - Delays of several weeks are well supported, which justifies a 0–29-week
     kernel range.
   - Lowe et al. show delays differ by climate variable. That supports our
     **per-variable, per-district** kernels.
   - The smooth-basis idea behind DLNMs is similar to our **Gaussian-bump
     mixture**. This is our analogy; don't present it as their claim.
5. **Gap.** In these studies the lag structure is either fixed before training
   or estimated once, in a statistical model. In our model the delay is learned
   inside a neural network, and the question is which horizon's loss should
   train it. Keep this descriptive ("these approaches fix or estimate lags
   before/outside training"); don't call it a flaw.

### B3. Multi-horizon neural forecasting and count likelihoods

1. **Problem.** Forecasting several future steps with a single model, and
   producing distributions for count data.
2. **Idea.**
   - Direct multi-horizon forecasting: one encoder with a head per horizon,
     trained on a joint loss.
   - Parametric likelihood heads, such as the Negative Binomial for counts.
3. **Studies (all verified):**
   - MQ-RNN (Wen et al. 2017): direct multi-horizon quantile forecasting with
     sequence-to-sequence networks.
   - TFT (Lim et al.): multi-horizon forecasting with recurrent layers,
     attention and variable selection.
   - DeepAR (Salinas et al. 2020): an autoregressive RNN with a Negative
     Binomial likelihood for count data.
4. **What we take.**
   - Our model predicts all horizons **directly from one origin**, as MQ-RNN
     does, rather than autoregressively as DeepAR does.
   - A **Negative Binomial head on a recurrent network is established**
     (DeepAR), so it is not our contribution. We use it because it matched
     the pinball loss in our benchmark (#20).
5. **Gap.** In these models, horizons share an encoder and a combined loss.
   We did not find a treatment of *which horizons should train a particular
   input encoder*. State this as what we did not find, not as a proven absence.

### B4. Loss balancing and gradient manipulation in multi-task learning

This is the direction closest to our method.

1. **Problem.** Tasks sharing parameters can train at different rates or
   interfere with one another.
2. **Idea.** There are three families:
   - adapt the loss weights during training;
   - alter the task gradients themselves;
   - restrict which parameters receive which gradients.
3. **Studies (all verified):**
   - GradNorm (Chen et al. 2018): dynamically tunes gradient magnitudes to
     balance training rates across tasks.
   - Uncertainty weighting (Kendall et al. 2018): weights losses by learned
     homoscedastic uncertainty.
   - PCGrad (Yu et al. 2020): projects a task's gradient onto the normal plane
     of any conflicting task gradient.
   - Gradient Routing (Cloud et al. 2024): applies data-dependent, weighted
     masks during backpropagation, so that chosen data update chosen parts of
     the network.
4. **What we take.**
   - Horizons are treated as tasks.
   - GradNorm is run as a comparison arm (H), applied to the climate group.
   - Our routing, where the weighted loss updates only the climate encoder, is
     conceptually a **form of gradient routing**.
5. **Where we differ. Keep this factual:**
   - GradNorm and uncertainty weighting derive weights from quantities during
     training.
   - Our weights come from an **out-of-sample pilot measurement** made inside
     the training years.
   - Our weights are applied to **one input modality's encoder only**.
   - This is a design difference, not a demonstrated advantage. In our results,
     neither the measured weights (D) nor GradNorm (H) improved on equal
     weighting.

### Optional B5 (one or two sentences). Evaluation practice

Use this only if space allows, to justify persistence, WIS and walk-forward
evaluation:
- Johansson et al. 2019: an open dengue forecasting challenge.
- Bracher et al. 2021: WIS for interval forecasts.

These entries are in the team bibliography but were not re-checked today.

---

## Part C. Citation verification (checked 2026-09-30)

| Key | Paper | Verified against | Status and what it supports |
| --- | --- | --- | --- |
| `chen2018gradnorm` (new key) | GradNorm, ICML 2018 (PMLR 80) | arXiv 1711.02257 | ✅ Balances training by tuning gradient magnitudes |
| `kendall2018multi` (new) | Kendall, Gal, Cipolla, CVPR 2018 | arXiv 1705.07115 | ✅ Learned homoscedastic-uncertainty loss weighting |
| `yu2020pcgrad` (new) | "Gradient Surgery for Multi-Task Learning", NeurIPS 2020 | arXiv 2001.06782 | ✅ Projects conflicting gradients |
| `cloud2024gradient` (new) | "Gradient Routing: Masking Gradients to Localize Computation in Neural Networks", 2024 | arXiv 2410.04332 | ✅ Weighted gradient masks localise computation. Treat as closest prior art for the routing idea. |
| `salinas2020deepar` | DeepAR, Int. J. Forecasting 36(3) 2020 | arXiv 1704.04110 + search | ✅ Autoregressive RNN with an NB likelihood for counts |
| `wen2017mqrnn` (new) | "A Multi-Horizon Quantile Recurrent Forecaster", NIPS 2017 Time Series Workshop | arXiv 1711.11053 | ✅ Direct multi-horizon forecasting |
| `lim2021tft` (new) | "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" | arXiv 1912.09363 | ✅ Content verified. ⚠ The final venue (believed to be Int. J. Forecasting 2021) was **not** confirmed; cite the arXiv version or check. |
| `weng2024graph` | IEEE BigData 2024 | NASA NTRS abstract | ✅ 25 districts, 2013–2022; GNNs beat ARIMA/RF/LSTM. ⚠ NTRS lists an extra author (Mahi Pasarkar) missing from the bib. Check the author list. Horizon and persistence are **not** confirmed. |
| `gulmohamed2026denguegnn` | DengueGNN, Sci. Rep. 2026 | PMC13039421 | ✅ 10 baselines, no naive/persistence baseline; ablations do not remove the graph entirely |
| `liu2025explainable` | PLoS Comput. Biol. 2025 | PLOS page | ✅ SEIR–LSTM; 16 regions; 7–9-week lags; immunity reset in 2017 |
| `chen2025climatespatial` | BMC Public Health 25:973, 2025 | Search (article page) | ✅ LSTM, 27 states, lagged climate, neighbouring-state influence, SHAP. ⚠ The "4 and 12 weeks" horizons in `full paper.tex` were **not** confirmed. |
| `lowe2018barbados` | PLoS Med 2018 | PLOS page | ✅ DLNM with a hierarchical model; drought at ~5 months, rainfall at 1–2 months |
| `gasparrini2010dlnm` | Stat. Med. 29(21) 2010 | Wiley listing | ✅ The DLNM framework |
| `gogovi2026spatial` | "When Spatial Message Passing Fails to Help…" | Two searches | ❌ **Not found.** Do not cite unless the paper is obtained. |
| `cho2014gru`, `kingma2015adam`, `talagala2026denguedatahub` | GRU, Adam, data source | Bibliography only | Standard references. Not re-checked, and low risk. |
| `johansson2019open`, `bracher2021evaluating`, `karasinghe2024weekly` | Evaluation and statistical references | Bibliography only | Not re-checked. Optional. |

**Not relevant to the [S3] research question.** Leave these out of the
Related Work unless a result section needs them: Chronos/Chronos-2 (outside
comparator only; cite where it appears in the results), LoRA, LightGBM,
conformal methods (Vovk; Gibbs & Candès), the Mosqlimate sprint, the global
ensemble study, Mordecai, Graph WaveNet and Kipf & Welling.

## Part D. Claims about prior work we must not make

- That Weng et al. lack a persistence baseline. Only the abstract was checked.
- Anything attributed to Gogovi 2026. The paper was not found.
- That no prior work routes weighted gradients to specific parameters.
  Gradient Routing does.
- That GradNorm or uncertainty weighting fail on this kind of problem. We have
  one comparison arm (H), and it is exploratory.
- That any prior dengue model is "worse" than ours. Cross-paper errors are not
  comparable across datasets, periods and targets (`full paper.tex`, Related
  Work).
