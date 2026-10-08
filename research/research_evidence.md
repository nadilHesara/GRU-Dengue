# Research evidence map: AEGIS-Dengue (DNN module paper)

**Purpose.** This file organises the evidence for the paper before any section
is drafted. It is not the paper. Every number is copied from a repository file,
and the source is named next to it. Nothing here is new analysis.

**Compiled:** 2026-09-30, from branch `full-paper` (commit `a6029b3`).

---

## 0. Read this first: the three bodies of work, and three evidence hazards

The repository contains three studies. They use different weather data and
different scoring cells, so their numbers must never be mixed in one table.

| Tag | Study | Weather input | Horizons | Main source |
| --- | --- | --- | --- | --- |
| **[S1]** | One-week lineage (short paper) and its follow-ups: baseline, losses, activations, tuning, optimiser, graphs, LayerNorm, climate ablation, clipping | GEE ERA5-Land polygon means | h = 1 (some at 1–4) | `README.md` §6–§8j, `docs/short paper.pdf` |
| **[S2]** | Long-horizon benchmark on common cells: 17+ forecasters, NB ablations, climate controls, intervals, onset, 2026 hold-out | Open-Meteo `era5_seamless`, district centroids | h = 1–12 | `docs/long_horizon_benchmark.md`, `docs/revision_2026_09_30.md`, `docs/full paper.tex` |
| **[S3]** | Climate-horizon extension: horizon-utility gradient weighting routed only to a learnable climate-delay encoder | Open-Meteo `era5_seamless`, district centroids | h = 1–4 | `docs/Research_Paper_New.md`, `docs/climate_horizon_results.md`, `docs/climate_horizon_correction_audit.md` |

**Recommended framing (a recommendation, not a fact from the repo).** The
paper's brief is DNN architecture, learning strategy, training strategy and
optimisation. **[S3] fits that brief.** Its novelty is a training strategy:
climate-only gradient routing from out-of-sample per-horizon utilities.
`docs/Research_Paper_New.md` is also declared "the single source document for
drafting the CS3631 full research paper". It says `docs/full paper.tex` (the
[S2] benchmark paper) is "**not** to be reused". [S1] supplies the motivation
and the optimisation ablations. [S2] supplies the reference forecasts and the
likelihood and sharing ablations.

**Evidence hazards. Check these before quoting any number.**

1. **Early NB "breakthrough" numbers are superseded.** `docs/negative_binomial.md`,
   `docs/new_findings_report.md`, `results/reports/significance_report.md` and
   README §8i report NB beating persistence 7/7 at h=1 (15.69, p = 0.0002).
   These were measured on different cells. The multi-horizon version also had
   a split leak: windows were split by the h=1 target, so validation origins
   carried test-year targets (`docs/revision_2026_09_30.md`). On common cells
   the separate NB model gets 15.94 (p = 0.25) and the shared NB model 16.63
   (`docs/full paper.tex`, "One week in the benchmark"). **Do not cite the old numbers.**
2. **The corrected [S3] artefacts are not in the repository.** The docs cite
   `results/climate_horizon/correction_2026-09-30/…`. That folder is gitignored
   (`.gitignore` line 73) and exists on no branch. The committed tables in
   `results/climate_horizon/analysis/tables/` hold the **original (routing v1)**
   numbers. For example, `primary_D_vs_B.tex` shows D − B = +0.05 at h=4, while
   the corrected figure in the docs is +0.060. The corrected numbers in §11 below
   are therefore taken from the docs only. Before submission, someone must
   supply the corrected artefacts or confirm that they exist.
3. **The [S3] results are not blind.** The frozen manifest (16:44, SHA-256
   `f4acdf1c…`) predates scoring. A gradient-routing bug was found after the
   results had been inspected, and the affected arms were rerun under a
   correction manifest (18:58, `5434845f…`). The 2026 outcomes had already been
   seen. This history must be disclosed (`docs/climate_horizon_results.md` §9).

Smaller inconsistencies:
- `docs/baseline.md` and `docs/improvements.md` show CPU-run numbers. The
  README §6/§8 GPU run is current, and both runs reach the same conclusions.
- README §1 still calls the multi-horizon NB model the "current model". That
  predates the revision.

---

## 1. Research problem

- District-level weekly dengue forecasting for Sri Lanka, 25 districts. The
  goal is lead time for vector control (NDCU, PHIs) and hospital preparation.
  (`docs/short paper.pdf` §1; `docs/Research_Paper_New.md` §3)
- **At one week ahead, the latest case count carries almost all the signal.**
  Dropping every climate channel costs +0.01 MAE (README §7). No
  architecture, loss or optimisation change reliably beat persistence at h=1
  [S1] (README §1 summary paragraph).
- Weather acts on dengue with a delay. A training-only cross-correlation scan
  puts the rainfall peak at 5–10 weeks. However, the deseasonalised correlation
  is weak (+0.036 to +0.04), below the scan's own 0.10 resolvability threshold
  (README §8b; `Research_Paper_New.md` §3).
- **DNN-specific problem [S3].** A learnable climate-delay encoder shared
  across horizons is trained mostly by short-horizon losses, which say little
  about weather. At h=1, the learnable encoder learned delays unrelated to the
  measured ones (r = −0.16) and lost to fixed windows (20.67 vs 18.36 MAE)
  (README §7, [S1]).

## 2. Research objective / question

**[S3] question, as the repository states it** (`Research_Paper_New.md` §1, §3, §4):
> Does weighting the gradients that reach *only* a learnable climate-delay
> encoder by each horizon's measured climate reliance improve multi-horizon
> dengue forecasts over an identical equal-weight model?

**Hypothesis:** a delay encoder shared across horizons is dominated by short
horizons. Weighting its gradients by each horizon's out-of-sample climate
reliance should help it learn useful delays.

**Frozen decision criteria (primary D vs B):**
- *Improvement:* D − B ≤ −0.30 MAE at h=3 or h=4, with Holm p < 0.05 and a
  CI below 0.
- *Non-inferiority at h=1–2:* CI upper bound ≤ +0.30.
- Otherwise, "no measurable difference".

(`Research_Paper_New.md` §7)

The brief itself says the novelty claim must be worded as "the question this
paper investigates", not "never studied before" (§4.2).

## 3. Dataset

| Item | Detail | Source |
| --- | --- | --- |
| Cases | Weekly confirmed cases, `denguedatahub` v4.1.0 (Talagala 2026). 26 areas become 25 districts (Kalmunai merged into Ampara). 1,012 weekly periods, 2006-12-23 to 2026-05-17. | `full paper.tex` §Data |
| Calendar | Sorted by true period dates, not source week labels. Irregular 2009 periods and a 2026 week-convention change are documented. | `docs/reporting_calendar.md` |
| Weather [S1] | GEE ERA5-Land polygon means, with CHIRPS used for validation only. **No longer recoverable.** | `short paper.pdf` §2 |
| Weather [S2, S3] | Open-Meteo `era5_seamless` at GADM centroids, fetched 29 Sep 2026. Hashes are in `results/benchmark/data_manifest.json`. | `revision_2026_09_30.md` |
| [S3] inputs | Case branch: 9 channels × 12 weeks. Climate branch: 7 weather channels × 37 weeks. | `Research_Paper_New.md` §5.1 |
| Missing data | Targets are masked, never imputed. Inputs are imputed from training-year district × month climatology and flagged. | `Research_Paper_New.md` §6 |
| Scoring cells [S3] | 11,725 identical cells per horizon, headline folds | `climate_horizon_results.md` §3 |

## 4. Existing / related approaches

Taken from `full paper.tex` (Related Work, Table 1) and `Research_Paper_New.md` §12.

| Work | Relation | Citation status |
| --- | --- | --- |
| Weng et al. 2024 (5 ST-GNNs, Sri Lanka, 3 weeks) | Reports no persistence baseline | In team bib; to be checked |
| GulMohamed et al. 2026 (DengueGNN) | No identity-graph control | In team bib; to be checked |
| Gogovi 2026 | LSTM beats graph variants; agrees with our graph result | In `full paper.tex` bib only |
| Liu et al. 2025 (SEIR–LSTM, 7–9-week lags) | Climate-delay evidence for Sri Lanka | In team bib; to be checked |
| **GradNorm (Chen et al., ICML 2018)** | The closest multi-task balancing method. [S3] differs: it routes weighted gradients to *one input encoder* using *out-of-sample* utilities. | **Verified** in the project |
| DeepAR (NB likelihood on an RNN) | NB on an RNN is not new in itself | **"Do not cite until verified"** |
| Kendall 2018, PCGrad, gradient routing (Cloud 2024), DLNM, TFT | Candidate related work | **"Do not cite until verified"** |
| Chronos / Chronos-2 | Outside comparator only | In team bib |

## 5. Baseline model(s)

- **Naive:** persistence, rescored at every horizon (16.42 / 20.16 / 24.58 /
  28.47 at h=1–4). Persistence uses case data only, so it is identical under
  both weather sources. Other naive baselines: seasonal naive, climatology and
  seasonal persistence [S2].
- **Mandatory GRU baseline (course requirement):** the identity-graph GRU. The
  GCN+GRU with its adjacency replaced by the identity becomes `gru_only`
  (README §6).
- **[S3] controlled baseline: arm B.** The identical two-branch model trained
  with equal horizon weights. This is the only controlled comparator for the
  primary claim.
- **Stored reference forecasts [S2], scored on the same cells:** `gru_v2_nb`,
  `nb_shared_v2` and `lgbm_v2`. They were trained under a different protocol,
  so they are "not controlled ablations" (`climate_horizon_results.md` §3).
- **Outside comparator:** Chronos-2 joint, zero-shot only. Fine-tuned
  variants are excluded from course-facing tables (`Research_Paper_New.md` §2).

## 6. Proposed approach [S3]

Source: `Research_Paper_New.md` §5 and `climate_horizon_results.md` §1.

1. **Two-branch multi-horizon NB2 forecaster, 10,028 parameters.**
   - *Case branch:* two per-node layers, then a GRU of 32 units. The
     adjacency is the identity, so there is no message passing.
   - *Climate branch:* `HorizonLagEncoder` builds a per-district, per-variable
     delay kernel over d = 0…29 as a softmax mixture of 6 Gaussian bumps, from
     a district embedding × feature projection. It is target-relative: for
     horizon h it reads only delays with d ≥ h, renormalised. It is followed by
     a shared climate GRU (16 units) and a climate head.
   - *Fusion per horizon:* δ_h = δ_case,h + σ(gate_h)·δ_climate,h, and
     log μ_h = log(1 + y_t) + clamp(δ_h, ±10). α_h is softplus + 1e-4, and the
     point forecast is the NB median.
2. **Climate-only gradient routing (routing v2).** A single forward pass
   produces both gradients. ∇L_climate goes to the encoder and climate GRU
   only. ∇L_equal goes to every other parameter, including all heads. Then one
   global-norm clip at 1.0 and one Adam step follow.
   - L_equal = Σ a_h L_h.
   - L_climate = Σ a_h w_h L_h / Σ a_h w_h.
   - Encoder: lr 2e-2, no weight decay. Everything else: lr 3e-3, weight decay 1e-4.
3. **Leakage-controlled utility measurement.** Each fold uses 3 inner blocks
   inside the training years. A frozen equal-weight pilot (2 seeds) scores its
   utility year twice: once with real weather and once with 5 donor
   perturbations (the same district's window from another year, same season
   ±2 weeks). This gives g_h = MAE_shuffled − MAE_real and u_h = max(g_h, 0).
   The weights are w_h = 0.05 + 0.95·4u_h/Σu.
4. **Numerical checks passed:**
   - w = 1 reproduces the baseline update exactly.
   - Non-uniform w changes only the climate group's raw gradients.
   - The routed gradients equal the explicit per-horizon combination.

## 7. Important architectural decisions (and the evidence behind each)

| Decision | Evidence | Source |
| --- | --- | --- |
| **Identity adjacency (no GCN)** | `gru_only` beats `gcn_gru` on all 9 folds [S1]. Contiguity, Gaussian and learned graphs all fail to beat the identity (16.68 vs 18.98 / 19.25 / 16.61, p = 0.85 for the learned graph). Graphs still fail at h up to 12 [S2]. A per-district gate between graph and identity learned nothing (stayed near 0.5). | README §6, §8e; `long_horizon_benchmark.md` §3; `gated_fusion.md` §6 |
| **Anchored residual target / anchored NB mean** | A zero prediction reproduces persistence (NB: y_t + 1), so training starts from the strongest naive forecast. | `full paper.tex` §Methods |
| **NB2 likelihood** | On common cells, NB matches the pinball loss within 0.04 MAE up to h=4 but does not beat it [S2]. It gives a closed-form distribution, and the fraction of zeros it predicts (0.042–0.048) is close to the observed 0.038. | `full paper.tex` Table "tab:nb" |
| **Log-MSE objective rejected** | A log-space loss mis-weights errors relative to count MAE (a 0.20 log error is 1.3 cases at 5 but 332 at 1,500). Level weighting cut MAE from 18.98 to 16.59, but 100% of that gain was the 2017 fold [S1]. | README §8 |
| **Softmax (not sparsemax/entmax) in the lag mixture** | Sparse maps create an absorbing state: 74% (sparsemax) and 58% (entmax15) of kernels end with dead gradients, versus 2% for softmax. No accuracy gain from any map [S1]. | README §8b |
| **Separate LR for the lag encoder** | The encoder gradient is far smaller than the GRU's. Script 16 already patched this with a 10× LR [S1]. In [S3] the encoder uses 2e-2 vs 3e-3. | README §8b; audit parameter table |
| **Target-relative vs origin-relative lags** | Tested as arms A/B vs C/D. The post-hoc factorial shows an interaction (see §11). | `climate_horizon_results.md` §4 |
| **Gradient clip 1.0 retained** | No exploding gradients (max pre-clip norm 2.87; the clip fires on 0.3% of steps; no threshold moves MAE by more than 0.07) [S1-era model]. In [S3] clipping occurred on at most 2.0% of a run's steps. | README §8j; `climate_horizon_results.md` §6 |
| **Early stopping on unweighted validation NLL (patience 15)** | Keeps model selection independent of the weighting under test. | `Research_Paper_New.md` §5.4 |
| **Separate vs shared trunk across horizons** | [S1]: identity separate vs shared is a wash; the shared trunk costs ~¼ of the time. [S2] NB: the shared trunk is 0.6–0.8 MAE worse up to h=4. | README §8f; `full paper.tex` Table "tab:nb" |

## 8. Experimental design

- **Walk-forward folds:** 9 expanding folds with test years 2017–2025. The
  headline uses 7 folds; 2020–21 (COVID) are reported separately. The year
  before each test year is used for validation. Splits are by each target's
  own date.
- **Seeds:** [S3] primary uses 5 seeds (0–4); exploratory arms use 3 (0–2). [S1] and [S2] use 3.
- **Primary family [S3]:** D vs B at h=1–4, Holm over the 4 horizons.
- **Planned exploratory arms:** A, C, E, F, G, H. **Post-hoc analyses:**
  matched-seed reanalysis, 2×2 factorial, block-length sensitivity, gradient diagnostics.
- **Uncertainty:** a 4-week block bootstrap of contiguous origins, all
  districts jointly. Paired t and Wilcoxon tests at fold level. With 7 folds,
  the minimum attainable Wilcoxon p is 0.016.
- **Two estimands, kept separate:** seed-mean and ensemble.
- **2026 (Jan–May, 19 weeks):** a frozen-specification check, *not* a blind
  test.
- **Compute [S3]:** 583 fits, 93.9 GPU-min on an RTX 4070 Laptop. Inference
  takes 2.38 ms per complete forecast (GPU).
- **Harness neutrality checks [S1]:** every modified training script
  reproduces the committed control number exactly (e.g., 18.98 and fold-1
  54.81), asserted by tests.

## 9. Evaluation metrics

- MAE on raw counts (primary).
- RMSE.
- Peak MAE: cells at or above the district's 90th percentile of pre-test
  history, a statistical threshold rather than an official one.
- WIS.
- Empirical 95% coverage. Wording rule: "empirical coverage", not "calibrated".
- Skill: 1 − MAE / persistence MAE on the same cells.
- Seed standard deviation, used as the noise yardstick throughout [S1].
- [S2] only: onset ROC-AUC, PR-AUC, Brier, detection and false alarms.

## 10. Main experiments

The full list, with classifications, is in the evidence table in §16. The
experiments classed **A** are:
- [S3] primary D vs B;
- the [S3] exploratory arms, especially F, G and H;
- the [S3] weights/utilities and gradient diagnostics;
- the routing-correction history, which must be disclosed.

## 11. Most important results

**Primary, [S3], corrected.** D − B on headline folds, seed-mean, 5 seeds
(`climate_horizon_results.md` §2):

| h | B | D | D − B | Holm p | 95% block CI | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 16.64 | 16.70 | +0.061 | 0.72 | [−0.030, +0.170] | non-inferior |
| 2 | 19.67 | 19.75 | +0.079 | 0.44 | [−0.034, +0.208] | non-inferior |
| 3 | 22.71 | 22.78 | +0.071 | 0.72 | [−0.067, +0.231] | no measurable difference |
| 4 | 25.22 | 25.28 | +0.060 | 0.72 | [−0.096, +0.223] | no measurable difference |

- The same holds for the ensemble estimand (+0.057 to +0.061) and for all 9
  folds (+0.049 to +0.072).
- **The weights were far from uniform:** max/min ratio 1.7–4.0 per fold,
  rising with horizon.
- B and D both beat persistence at h=3–4 (h=4: −3.2 MAE). Because both do,
  this is not evidence for the weighting.

**Exploratory arms vs B, matched seeds 0–2, unadjusted** (Δ MAE at h=1 / 2 / 3 / 4):

| Arm | Δ vs B |
| --- | --- |
| A: origin-relative lags, equal weights | +0.24 / +0.27 / +0.44 / +0.50 |
| C: origin-relative lags, measured weights | −0.40 / −0.42 / −0.31 / −0.27 |
| E: fixed weights [1, 2, 3, 4] | −0.06 / −0.11 / −0.09 / −0.12 |
| **F: climate-encoder gradients from h=4 only** | **−0.65 / −0.69 / −0.66 / −0.64** (h=3–4 CIs include 0; Wilcoxon p 0.03–0.16) |
| G: case-only-utility weights | +0.62 / +0.58 / +0.70 / +0.80 |
| H: GradNorm on the climate group | +0.12 / +0.11 / +0.18 / +0.11 |

**Training-dynamics diagnostics [S3], post-hoc, 60 identical-state pairs:**
- Encoder update rotation: about 7–8°.
- Relative update difference: 13.1% (range 5–24%).
- Raw gradient norm: +14.7%, even though the weights have mean one.
- Trained B and D differ less than two seeds of B do (predictions: 1.7–2.9 vs
  3.5–5.2 cases; kernel L1: 0.41 vs 0.72).
- Gates rise with horizon (0.43 → 0.56).

**Motivating results [S1]:**
- Multi-horizon skill over same-horizon persistence is +6.9% at h=3 and +10.8%
  at h=4 (h=4: 7/7 folds, p = 0.0068), and it is not the 2017 artefact
  (README §8f).
- In the capacity-matched climate shuffle, the cost rises monotonically with
  horizon: +0.02 at h=1 and +1.55 MAE at h=4 (p = 0.016). This removes about
  half of the h=4 skill (README §8h). It replicates at +1.91 in [S2].

## 12. Important negative or unexpected results

1. **The primary hypothesis failed [S3].** Measured weighting gave no reliable
   improvement, and the weights were not near-uniform.
2. **Utility choice matters in the "wrong" direction [S3].**
   - Incremental-benefit utilities (G) did worst (+0.58 to +0.80).
   - GradNorm on the climate group (H) was also worse (+0.11 to +0.18).
3. **The simplest routing did best, unexpectedly [S3].** F (encoder learns
   from h=4 only) was the best configuration. This result is exploratory.
4. **Weighting helped only with origin-relative lags [S3, post-hoc].** The
   interaction (D−B)−(C−A) at h=4 is +0.58, CI [+0.26, +0.91]. No test
   survives Holm, and the effect is concentrated in 2017 and 2019.
5. **Every h=1 intervention was a 2017 artefact [S1].** This covers the loss
   (100.1% of the gain from fold 1), simplex maps (96%), tuning, AdamW (91%)
   and LayerNorm (ex-fold-1 −0.02). The general lesson: always decompose by
   fold.
6. **The optimiser schedule is mechanically inert [S1].** ReduceLROnPlateau
   fired after the best epoch had already been saved in 23 of 27 runs, because
   a plateau scheduler and early stopping on the same metric are nearly
   mutually exclusive.
7. **The hyperparameter surface is flat [S1].** Across 50 trials, validation
   MAE moves by about 0.1–0.2. The tuned model's gain of −0.75 is under 2 seed
   standard deviations.
8. **There are no exploding gradients [S1].** The clip is inert.
   - Tight clipping (0.25) and LayerNorm reduce only seed variance (12–30% and
     24–51%).
   - LayerNorm improved conditioning, but the score did not move.
9. **Learnable lags do not recover biological delay.**
   - [S1] h=1: r = −0.16 with measured delays.
   - [S3]: kernels centre at 8–10 weeks, but Spearman with the scan is only
     0.15–0.31. Required wording: "learned delay kernel", not "recovered delay".
10. **Plain channel removal gives the opposite sign to the shuffle control
    [S1].** Without the shuffle control (`no_climate` is negative at h=1–3),
    one would conclude climate doesn't matter.
11. **Earlier NB wins disappear under the fixed protocol [S2].** They were an
    artefact of the split leak and different cells.

## 13. Limitations

These are listed in `Research_Paper_New.md` §11 and `climate_horizon_results.md` §11:
- **Low power.** There are only 7 headline folds, and they share training
  histories. The bootstrap does not resolve that dependence.
- **Weather quality.** Weather is a centroid-based reconstruction, final
  values retrieved in 2026. It is not what was available in real time.
  Reporting delays are not modelled.
- **No blind test.** All data up to 2026-05-17 had been inspected, and the
  corrected results came after a post-inspection bug fix.
- **Exploratory effects are fragile.** F, C and the factorial depend on a few
  years (2017, 2019) and on 3 seeds. Under seeds 0–2, D−B is about −0.04,
  against +0.07 under the declared 5 seeds.
- **Arms not run:** common-support lags, gate-only-on-encoder, measured-lag
  initialisation, hyperparameter sensitivity.
- **Scope:** one country, 25 districts.
- **Evidence gap:** the corrected artefacts are not committed (§0 hazard 2).

## 14. Potential research contribution

Worded to match the evidence, following `climate_horizon_results.md` §10:

1. **A leakage-controlled training strategy, with a null primary result.**
   Per-horizon utilities are measured out of sample, inside the training
   years, and routed only to a learnable climate-delay encoder. Under frozen
   criteria it was non-inferior at h=1–2 and showed no improvement at h=3–4.
   The weights were far from uniform.
2. **A measured account of what the weighting changes in training.**
   - Update rotation is about 7–8°, and update size differs by 13%.
   - Trained models differ less than seeds do.
   - This is descriptive, not causal.
3. **Exploratory leads for confirmation:**
   - longest-horizon-only encoder training (F);
   - weighting × lag-alignment interaction.
4. **Supporting DNN-design evidence [S1, S2]:**
   - At h=1, architecture and optimisation changes are bounded by input
     information, not optimisation: flat HP surface, inert scheduler, no
     exploding gradients, no graph gain.
   - Climate's temporal content is load-bearing only at longer horizons. This
     motivates horizon-aware encoder training.

## 15. Evidence supporting each claim

| Claim | Evidence | Status |
| --- | --- | --- |
| D does not improve on B | `climate_horizon_results.md` §2; the corrected CSV is not committed. The committed v1 table `results/climate_horizon/analysis/tables/primary_D_vs_B.tex` agrees in sign and size. | Strong within the protocol; the corrected artefact is missing |
| The null is not due to near-uniform weights | `weights_final/*.json` (committed); max/min 1.7–4.0 | Verifiable in the repo |
| F beats B | `climate_horizon_results.md` §4 (matched seeds) | Exploratory, unadjusted, 3 seeds; artefact not committed |
| Weighting changes updates only slightly | `climate_horizon_results.md` §6 | Post-hoc; artefact not committed |
| Climate matters more with horizon | README §8h ([S1], p = 0.016 at h=4); `long_horizon_benchmark.md` §2 ([S2], replicated, 3 model families) | Strong at h=4; individual tests at other horizons are weak |
| Identity beats graphs | README §6, §8e; `long_horizon_benchmark.md` §3; `gated_fusion.md` | Strong (9/9 folds for contiguity at h=1) |
| Optimisation is not the bottleneck at h=1 | README §8c, §8d, §8g, §8j | Consistent across 4 experiments |
| NB ≈ pinball; neither is clearly better | `full paper.tex` "tab:nb" | Supported on common cells |
| Sparse simplex maps kill kernel gradients | README §8b | Mechanism measured on folds 8 and 1 |
| The routing bug did not change the primary null | `climate_horizon_results.md` §9: v1 +0.049 to +0.082 vs v2 +0.060 to +0.079 | Documented |

---

## 16. Research evidence table

Legend:
- **A** = essential for the paper.
- **B** = useful supporting evidence.
- **C** = not necessary.

"Should appear in paper?" reflects the recommended [S3]-centred framing.

| # | Experiment | Purpose | Compared methods | Key result | Should appear in paper? | Reason |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | [S3] Primary D vs B | Test horizon-utility climate-only gradient weighting | D (measured w) vs B (equal w), 5 seeds | D−B +0.06 to +0.08; Holm p ≥ 0.44; non-inferior h1–2; no difference h3–4 | **A** | The paper's primary hypothesis test |
| 2 | [S3] Exploratory arms A, C, E, F, G, H | Isolate lag alignment, weight source and routing | Each vs B, matched seeds 0–2 | F −0.64 to −0.69; C −0.27 to −0.42; G +0.58 to +0.80; H (GradNorm) +0.11 to +0.18; E ≈ −0.1 | **A** | Ablation of the training strategy; includes GradNorm, the nearest prior method |
| 3 | [S3] Weights and utilities analysis | Rule out "weights ≈ 1" as the cause of the null | Shuffle-reliance vs incremental-benefit utilities | Weights rise with h, max/min 1.7–4.0; incremental utility is small and changes sign | **A** | Needed to interpret the null |
| 4 | [S3] Gradient/update diagnostics | How weighting changes optimisation | 60 identical-state B/D pairs | ~7–8° rotation; 13% update difference; B–D gap smaller than seed gap; clipping ≤ 2% of steps | **A** | Optimisation-level evidence, central to the DNN focus |
| 5 | [S3] Routing correction (v1 → v2) | Disclose the implementation error | v1 vs v2 arms | Primary null unchanged; F changed substantially | **A** (as disclosure) | Required for honesty; affects how F is read |
| 6 | [S3] 2×2 factorial (alignment × weighting) | Explain why C helps and D doesn't | A, B, C, D, seeds 0–2, 9 folds | Interaction +0.58 at h4, CI [+0.26, +0.91]; none survives Holm | **B** | Post-hoc; supports exploratory lead 3 |
| 7 | [S3] Kernels and gates | Inspect what the encoder learned | B, D kernels vs training-only scan | Rainfall kernels centre at 8–10 weeks; Spearman 0.15–0.31; gates 0.43 → 0.56 | **B** | Interpretability; guards against "delay recovery" claims |
| 8 | [S3] Accuracy vs references | Place B/D against other forecasters | Persistence, gru_v2_nb, nb_shared_v2, lgbm_v2, Chronos-2 zero-shot | B/D beat persistence at h3–4; roughly level with the references | **B** | Context; the references are not controlled ablations |
| 9 | [S3] 2026 check | Frozen-specification sanity check | D vs B, 19 weeks | D−B +0.18 to +0.50, h4 CI [+0.03, +1.00] (pointwise, unadjusted) | **B** | Consistent in sign only; not blind |
| 10 | [S3] Compute and latency | Course compute requirement | — | 583 fits, 93.9 GPU-min; 2.38 ms per forecast | **B** | Required section content |
| 11 | [S1] Climate ablation h=1–4 with shuffle | Is h≥3 skill due to climate content? | full / no_climate / shuffled | Shuffle cost +0.02 (h1) → +1.55 (h4, p = 0.016); removes half the h4 skill | **B** (strong) | Direct motivation for horizon-dependent climate utility, and the origin of the shuffle-reliance idea |
| 12 | [S1] Multi-horizon h=1–4 | Does skill appear beyond h=1? | Separate vs shared, identity vs contiguity | +6.9% / +10.8% skill at h3/h4, p < 0.01; not the 2017 artefact | **B** (strong) | Motivation: horizon is where model capacity matters |
| 13 | [S1] Learnable climate lags at h=1 | Can a delay encoder learn delays end-to-end? | Learnable kernel vs fixed windows vs none | 20.67 vs 18.36 vs 18.07; learned vs measured r = −0.16 | **B** (strong) | The failure the [S3] method is designed to fix |
| 14 | [S1] Loss/objective arms | Learning strategy: objective mismatch | Baseline MSE, Huber, level-weighted, both, quantile | Level-weighted 16.59 vs 18.98, but 100% from 2017 | **B** | Learning-strategy evidence; justifies count/quantile likelihoods |
| 15 | [S1] Optimiser and LR schedule | Optimisation choice | Adam, AdamW, AdamW + ReduceLROnPlateau | Not significant (p = 0.31); scheduler fires after the best epoch in 23/27 runs | **B** | Optimisation mechanism finding; justifies keeping Adam |
| 16 | [S1] Hyperparameter search | Training strategy: is the config under-tuned? | 50-trial random search over 9 axes | −0.75 MAE, under 2 seed sd; flat surface | **B** | Justifies inherited hyperparameters; answers the course's tuning section |
| 17 | [S1] Gradient clipping | Optimisation stability | No clip, 0.25, 1.0, 5.0 | Max norm 2.87; no MAE change > 0.07; 0.25 lowers seed sd by 12–30% | **B** | Justifies clip = 1.0 in [S3] |
| 18 | [S1] Graph representations | Architecture: is spatial message passing useful? | Identity, contiguity, Gaussian, learned | No graph beats identity (learned −0.07, p = 0.85) | **B** | Justifies the identity backbone in one sentence plus a table |
| 19 | [S1] Simplex activations in the lag mixture | Architecture of the kernel mixture | Softmax vs temp-softmax, sparsemax, entmax15, floored entmax, gumbel | No accuracy gain; sparse maps kill 58–74% of kernels | **B** | Justifies the softmax mixture used in `HorizonLagEncoder` |
| 20 | [S2] Likelihood ablation (MSE / pinball / NB) | Learning strategy on common cells | Separate GRU v2 heads | NB ≈ pinball (±0.04 up to h4); both beat MSE at h1 by ~0.7 | **B** | Justifies the NB head without over-claiming |
| 21 | [S2] Shared vs separate NB trunk | Parameter sharing across horizons | Shared vs separate | Shared +0.6–0.8 worse up to h4; better at h10–12 | **B** | Relevant to the shared multi-horizon design |
| 22 | [S2] Climate controls across families | Weather vs seasonality | Climatology / removal / shuffle / 1-week delay; GRU, NB, LightGBM, Chronos-2 | Observed weather beats climatology in all 3 families at every h | **B** | Supports the premise that climate utility is horizon-dependent |
| 23 | [S1] LayerNorm before the head | Architecture/conditioning | With vs without LayerNorm | Accuracy null (ex-2017 ≈ 0); seed sd −24–51% | C | Minor; could be one line under "stability" |
| 24 | [S1] Gated spatial/temporal fusion | Per-district graph gate | Gated vs gated_uniform vs gru_only | Gate stays near 0.5; 18.36 vs 16.88 | C | Redundant with #18 |
| 25 | [S1] Outbreak-history features (v2) | Feature engineering | v1 vs v2 | −0.5 to −0.9 at h2–4; no help at h1 | C | Feature work, not DNN design (though the case branch uses these channels) |
| 26 | [S1] Residual diagnostic | Is the error structured? | — | Residual autocorrelation is high at h4 (lag-1 ACF 0.665) | C | Diagnostic only |
| 27 | [S1] NB head at h=1 and multi-horizon NB (scripts 31/32) | Count likelihood | NB vs persistence | Reported 15.69, 7/7 folds — **superseded** (leak/different cells) | C (do not cite) | Invalidated by the revision |
| 28 | [S2] Full h=1–12 benchmark (17 forecasters + ensembles) | Benchmark | Naive, GRU, tabular, Chronos, ensembles | Skill 18–25% at h12; Chronos-2 ≈ GRU | C | A benchmark paper, not a DNN-design paper; use only the rows in #8 |
| 29 | [S2] Probabilistic / conformal intervals | Calibration | Native, previous-year, rolling and adaptive conformal | WIS 20–33% better than persistence; adaptive conformal holds 0.95 coverage | C | Out of scope |
| 30 | [S2] Onset / early warning | Operational warning | Onset classifier vs forecasters | Onset ROC-AUC 0.74–0.77 | C | Out of scope |
| 31 | [S2] Graphs at long horizons | Spatial structure at h ≤ 12 | Contiguity and learned graph vs GRU | Neither helps at any h | C | Covered by #18; one clause at most |
| 32 | [S2] Chronos fine-tuning and ensembles | Foundation models | LoRA fine-tuning, stacking, top-3, NB–Chronos mixture | Fine-tuned models best at h10–12 | C | The brief excludes fine-tuned foundation models from course tables |

---

## 17. The 3–5 findings the paper should communicate

1. **Horizon-utility gradient weighting of a climate-delay encoder did not
   help.** Under a frozen, leakage-controlled protocol, measured per-horizon
   weights routed only to the climate encoder (D) produced no reliable change
   against equal weights (B): D − B = +0.06 to +0.08 MAE. It was non-inferior
   at h=1–2, with no measurable difference at h=3–4. The weights were far from
   uniform (1.7–4.0×), so the null is not a trivially small intervention.
   (#1, #3)
2. **The weighting barely changes optimisation, and seed noise swamps what it
   does change.** On identical states, the encoder update rotates by only about
   7–8° and changes by about 13%. Trained B and D differ less than two seeds of
   B. Adam's per-element normalisation and rare clipping are untested possible
   contributors. This is descriptive, not a proven cause. (#4)
3. **How the encoder's gradients are routed may matter more than fine
   weighting (exploratory).** Training the climate encoder only from the
   longest horizon (F) was the best configuration: −0.64 to −0.69 MAE vs B on
   matched seeds, better in 5–6 of 7 folds. Weighting helped only with
   origin-relative lags (interaction +0.58 at h=4). GradNorm-style balancing
   (H) and incremental-benefit utilities (G) were worse. None of this is
   confirmed: the results are unadjusted, use 3 seeds, and come after the
   correction. (#2, #5, #6)
4. **Climate's temporal content only becomes useful at longer horizons, which
   is the premise of the design.** At h=1, the capacity-matched shuffle costs
   nothing (+0.02). At h=4 it costs +1.55 MAE (p = 0.016) and removes half the
   skill over persistence. The effect replicates across three model families.
   A learnable delay encoder trained at h=1 learned delays unrelated to the
   measured ones. (#11, #12, #13, #22)
5. **At one week, the ceiling is information, not optimisation or
   architecture.** Several interventions gave no reliable gain: a 9-axis
   hyperparameter search (flat surface), AdamW and plateau scheduling (the
   scheduler fires after the kept epoch in 23/27 runs), gradient clipping
   (max norm 2.87, inert), LayerNorm, alternative simplex activations and every
   graph variant. Each apparent gain was the 2017 epidemic fold. (#14–#19)

---

## 18. Open items the team must resolve before drafting

- **Confirm the framing.** Is the paper the [S3] training-strategy study (as
  recommended here and in `Research_Paper_New.md`) or the [S2] benchmark
  (`full paper.tex`)?
- **Supply or commit the corrected artefacts** in
  `results/climate_horizon/correction_2026-09-30/`, or state that the paper's
  [S3] numbers rest on the docs.
- Still **[PENDING]** per `Research_Paper_New.md`:
  - TA and supervisor names;
  - the contribution and colour map;
  - an architecture diagram;
  - confirmation on untouched data (post-2026-05-17 weeks or an external
    dataset).
- **Verify citations** marked "do not cite until verified", especially
  DeepAR, PCGrad and Kendall. Only GradNorm is verified.
