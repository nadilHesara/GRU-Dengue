# Results section (draft) and selection rationale

**Scope.** This is the Results section for the [S3] study (climate-horizon
gradient weighting). Its research question, from [research_story.md](research_story.md) §5:

> Does weighting the gradients that reach **only** a learnable climate-delay
> encoder help? The weights come from each horizon's measured, out-of-sample
> reliance on climate. The comparison is an otherwise identical model trained
> with equal horizon weights.

**Provenance of the numbers.** Every number is copied from a repository file.
The source is under each table.
- **"committed"** means the file is in the repo and can be re-checked there.
- **"docs only"** means the number is taken from
  `docs/climate_horizon_results.md`. Those corrected (routing v2) artefacts
  are not committed (see research_evidence.md §0, hazard 2).

**Arm names used below**

| Arm | Lag alignment | Climate-encoder weighting |
| --- | --- | --- |
| **B** (baseline) | target-relative | equal weights |
| **D** (proposed) | target-relative | measured shuffle-reliance weights |
| A | origin-relative | equal |
| C | origin-relative | measured |
| E | target-relative | fixed [1, 2, 3, 4] |
| F | target-relative | the h=4 loss only trains the encoder |
| G | target-relative | case-only-utility weights |
| H | target-relative | GradNorm applied to the climate group |

In every arm, the prediction heads and the case branch are trained with the
ordinary unweighted loss.

---

## Part 1. Results (draft text)

Unless stated otherwise:
- MAE is in cases per district-week.
- Scores are averaged over the seven headline test years (2017–2019 and
  2022–2025).
- All methods are scored on the same 11,725 cells per horizon.

### 4.1 Reference accuracy of the baseline

**Purpose.** Establish how accurate the equal-weight baseline B is, before
testing the change to its training.

**Table 1.** Headline MAE by horizon (seed-mean). B and D use seeds 0–4; the
stored references use seeds 0–2.

| Method | h=1 | h=2 | h=3 | h=4 | Peak MAE, h=4 |
| --- | --- | --- | --- | --- | --- |
| Persistence | 16.42 | 20.16 | 24.58 | 28.47 | 47.98 |
| **B: equal weights (baseline)** | 16.64 | 19.67 | 22.71 | 25.22 | 42.99 |
| **D: measured weights (proposed)** | 16.70 | 19.75 | 22.78 | 25.28 | 42.81 |
| GRU-NB, separate models (stored reference) | 15.94 | 19.26 | 22.46 | 25.07 | 42.60 |
| Chronos-2 joint, zero-shot (outside comparator) | 15.96 | 19.07 | 22.29 | 25.11 | 43.74 |

*Source: `climate_horizon_results.md` §3 (docs only). Persistence and B are
unchanged by the correction.*

**Observed.**
- B has a higher MAE than persistence at h=1 (16.64 vs 16.42).
- B has a lower MAE than persistence from h=2 on; at h=4, 25.22 vs 28.47.
- B is within 0.25 MAE of a separately trained NB GRU (a stored reference) at
  h=3–4. The gap is 0.41 at h=2 and 0.70 at h=1.
- B is within 0.7 MAE of zero-shot Chronos-2 at every horizon. The gap is 0.11
  at h=4 and 0.68 at h=1.
- D follows B at every horizon.
- The stored reference was trained under a different protocol, so it is a
  point of comparison, not a controlled ablation.

### 4.2 Primary comparison: measured weights (D) vs equal weights (B)

**Purpose.** Test the research question under the decision criteria frozen
before scoring:
- **improvement:** D − B ≤ −0.30 at h=3 or 4, with Holm p < 0.05 and a CI
  below 0;
- **non-inferiority at h=1–2:** CI upper bound ≤ +0.30.

**Table 2.** D − B, headline MAE, seed-mean over 5 seeds. p-values are
Holm-adjusted over the four horizons. CIs are 95% from a 4-week block
bootstrap. Negative favours D.

| h | B | D | D − B | Holm p | 95% CI | Verdict (frozen criteria) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 16.64 | 16.70 | +0.061 | 0.72 | [−0.030, +0.170] | non-inferior |
| 2 | 19.67 | 19.75 | +0.079 | 0.44 | [−0.034, +0.208] | non-inferior |
| 3 | 22.71 | 22.78 | +0.071 | 0.72 | [−0.067, +0.231] | no measurable difference |
| 4 | 25.22 | 25.28 | +0.060 | 0.72 | [−0.096, +0.223] | no measurable difference |

*Source: `climate_horizon_results.md` §2 (docs only; the corrected
`primary_D_vs_B_headline7.csv` is not committed).*

**Observed.**
- **D had a slightly higher MAE than B at every horizon (+0.06 to +0.08).**
- No difference was significant.
- The improvement criterion was not met at h=3 or h=4.
- D was non-inferior to B at h=1–2.

**Table 3.** Robustness of the primary contrast (D − B, MAE).

| Variant | h=1 to h=4 | Status |
| --- | --- | --- |
| Ensemble estimand (MAE of seed-averaged forecast) | +0.057 to +0.061 | Declared |
| All 9 folds (including 2020–21) | +0.049 to +0.072 (Holm p ≥ 0.37) | Declared |
| Bootstrap block length 8 or 13 weeks | All CIs include 0 | Post-hoc |
| Seeds 0–2 only (instead of 0–4) | about −0.04 | Post-hoc |

*Source: `climate_horizon_results.md` §2, §4 (docs only).*

**Observed.** No variant gives an improvement that meets the frozen criteria.
The sign depends on which seeds are used, and the magnitude stays under 0.1
MAE in every variant.

### 4.3 The measured weights and utilities

**Purpose.** Check two things:
- whether the intervention was substantial, i.e. whether the weights actually
  departed from uniform;
- whether, on the data used here, the measured climate reliance increases with
  horizon (the premise of the method).

**Table 4.** Shuffle-reliance utility g_h and the frozen weights w_h, per outer
fold. g_h is the pilot MAE increase when weather windows are swapped. The
weights have mean 1; uniform weights would all equal 1.

| Fold (test year) | g₁ | g₂ | g₃ | g₄ | w₁ | w₂ | w₃ | w₄ | max/min w |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 (2017) | 0.11 | 0.12 | 0.12 | 0.35 | 0.64 | 0.72 | 0.70 | 1.94 | 3.02 |
| 2 (2018) | 0.18 | 0.22 | 0.27 | 0.44 | 0.66 | 0.79 | 0.97 | 1.57 | 2.39 |
| 3 (2019) | 0.37 | 0.46 | 0.50 | 0.67 | 0.76 | 0.93 | 1.00 | 1.31 | 1.73 |
| 4 (2020) | 0.51 | 0.75 | 0.93 | 1.17 | 0.63 | 0.89 | 1.10 | 1.38 | 2.18 |
| 5 (2021) | 0.89 | 1.38 | 1.83 | 2.44 | 0.57 | 0.85 | 1.11 | 1.47 | 2.59 |
| 6 (2022) | 0.66 | 1.22 | 1.80 | 2.38 | 0.46 | 0.81 | 1.18 | 1.54 | 3.33 |
| 7 (2023) | 0.53 | 0.92 | 1.33 | 1.74 | 0.50 | 0.82 | 1.17 | 1.51 | 3.05 |
| 8 (2024) | 0.13 | 0.26 | 0.41 | 0.51 | 0.43 | 0.81 | 1.24 | 1.53 | 3.59 |
| 9 (2025) | 0.35 | 0.57 | 0.81 | 1.02 | 0.53 | 0.84 | 1.17 | 1.46 | 2.73 |
| 10 (2026) | 0.36 | 0.71 | 1.15 | 1.61 | 0.41 | 0.75 | 1.19 | 1.65 | 4.03 |

*Source: `results/climate_horizon/weights_final/fold*_shuffle.json`
(committed, reused unchanged after the correction).*

**Observed.**
- The measured reliance on weather is largest at h=4 in every fold. It rises
  strictly with horizon in 9 of 10 folds; fold 1 has a small dip from h=2 to
  h=3.
- The weights depart from uniform by a factor of 1.7–4.0 (max/min).
- **The alternative utility is small and changes sign.** It measures the gain
  over a size-matched model with no weather input, and ranges from −0.03 to
  +0.12 across folds and horizons. It is the basis of arm G.
  (*Source: `weights_final/fold*_case_only.json`, committed.*)

### 4.4 Exploratory arms

**Purpose.** Isolate what matters, all against B on the same seeds:
- **lag alignment:** arms A and C;
- **whether the weights need to be measured:** arm E;
- **extreme routing:** arm F;
- **utility source:** arm G;
- **the nearest existing balancing method:** arm H.

These arms were declared in advance as exploratory. The p-values are
unadjusted, and all arms use seeds 0–2.

**Table 5.** Δ headline MAE vs B, matched seeds 0–2. Negative means the arm is
better than B.

| Arm | h=1 | h=2 | h=3 | h=4 |
| --- | --- | --- | --- | --- |
| A: origin lags, equal weights | +0.24 | +0.27 | +0.44 | +0.50 |
| C: origin lags, measured weights | −0.40 | −0.42 | −0.31 | −0.27 |
| D: target lags, measured weights (seeds 0–2 only) | −0.01 | −0.04 | −0.01 | −0.05 |
| E: fixed weights [1, 2, 3, 4] | −0.06 | −0.11 | −0.09 | −0.12 |
| **F: encoder trained by the h=4 loss only** | **−0.65** | **−0.69** | **−0.66** | **−0.64** |
| G: case-only-utility weights | +0.62 | +0.58 | +0.70 | +0.80 |
| H: GradNorm on the climate group | +0.12 | +0.11 | +0.18 | +0.11 |

*Source: `climate_horizon_results.md` §4 (docs only).*

**Table 6.** F vs B in detail, matched seeds 0–2.

| h | F | B | F − B | 95% CI | Years F better | Wilcoxon p (unadj.) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 15.95 | 16.60 | −0.65 | [−1.26, −0.20] | 5/7 | 0.16 |
| 2 | 19.01 | 19.70 | −0.69 | [−1.47, −0.08] | 6/7 | 0.08 |
| 3 | 22.04 | 22.70 | −0.66 | [−1.51, +0.13] | 6/7 | 0.08 |
| 4 | 24.59 | 25.23 | −0.64 | [−1.70, +0.41] | 6/7 | 0.03 |

*Source: `climate_horizon_results.md` §4 (docs only). Over all 9 folds:
−0.55 / −0.55 / −0.53 / −0.60, with F better in 6–8 of 9 years.*

**Observed.**
- **Arm F had the lowest MAE of any arm at every horizon.** The CIs at h=3–4
  include 0. F was not scored on 2026. F's corrected form was first seen after
  the routing correction.
- **Arm G was worse than B at every horizon** (+0.58 to +0.80).
- **Arm H (GradNorm) was worse than B at every horizon** (+0.11 to +0.18).
- **Measured weighting behaved differently depending on lag alignment.**
  - With origin-relative lags, C was better than B, and C vs D (matched seeds)
    was −0.39 / −0.38 / −0.30 / −0.22.
  - With target-relative lags, D was about equal to B.
- Fixed ramp weights (E) were within 0.12 of B.

### 4.5 Alignment × weighting (post-hoc factorial)

**Purpose.** Test whether the measured weighting's effect depends on lag
alignment. This analysis was designed after the results had been seen.

**Table 7.** Contrasts on common seeds 0–2 over all 9 folds. Δ MAE; negative
favours the first-named arm.

| Contrast | h=1 | h=4 | 95% CI at h=4 | Years negative | Holm-20 p |
| --- | --- | --- | --- | --- | --- |
| C − A: weighting, origin lags | −0.55 | −0.64 | [−1.02, −0.29] | 6–7/9 | 1.00 |
| D − B: weighting, target lags | −0.02 | −0.06 | [−0.24, +0.11] | 5–6/9 | 1.00 |
| B − A: alignment, equal weights | −0.19 | −0.38 | [−0.74, −0.03] | 3–4/9 | 1.00 |
| (D − B) − (C − A): interaction | +0.52 | +0.58 | [+0.26, +0.91] | 2–3/9 | 1.00 |

*Source: `climate_horizon_results.md` §4 (docs only).*

**Observed.**
- Descriptively, weighting reduced error with origin-relative lags and did not
  with target-relative lags.
- **No contrast survives Holm correction** over the 20 fold-level tests.
- The effects are concentrated in 2017 and 2019. For example, B − A at h=4 is
  −2.63 in 2017 and between −0.24 and +0.21 in six other years.

### 4.6 How the weighting changes training

**Purpose.** Measure how large the intervention is at the level of the
optimiser. These post-hoc diagnostics used 60 paired comparisons of B-weighted
and D-weighted steps from identical model and optimiser states:
- folds 1, 6 and 9;
- seeds 0 and 1;
- B and D final states;
- 5 batches.

**Table 8.** Effect of the measured weights on the climate-encoder update.

| Quantity | Mean | Median | Range |
| --- | --- | --- | --- |
| Angle between raw gradients (per pair) | 8.4° | 7.4° | 2.9–20.6° |
| Angle between Adam updates (per pair) | 7.1° | 6.7° | 2.8–13.9° |
| Relative update difference ‖u_D − u_B‖ / ‖u_B‖ | 13.1% | 12.6% | 5.1–24.5% |
| Change in raw-gradient norm | +14.7% | — | — |
| Change in update norm | +2.6% | — | — |
| Sampled pairs where clipping was active | 0 of 60 | — | — |

**Table 9.** Trained-model differences: B vs D, compared with two seeds of B.

| Quantity | B vs D | Two seeds of B |
| --- | --- | --- |
| Mean prediction difference (cases) | 1.7–2.9 | 3.5–5.2 |
| Rainfall kernel L1 distance | 0.41 | 0.72 |

*Source: `climate_horizon_results.md` §6 (docs only).*

**Observed.**
- The measured weights rotated each encoder update by about 7° on average and
  changed its size by about 13%.
- Over complete training runs, clipping was active on at most 2.0% of any
  run's steps.
- **The trained B and D models differ less from each other than two seeds of B
  differ from each other.**

### 4.7 What the climate encoder learned

**Purpose.** Report the learned delay structure and gate values, the two
places where D could have diverged from B.

**Table 10.** Learned rainfall-kernel centre (a target-relative delay in weeks)
and mean climate gate.

| Arm | Kernel centre, h=1 | h=4 | Gate, h=1 | Gate, h=4 |
| --- | --- | --- | --- | --- |
| B: equal weights | 8.02 | 8.91 | 0.43 | 0.56 |
| D: measured weights | 8.7 | 9.6 | B ± 0.02 | B ± 0.02 |

*Source: B rows from the committed `analysis/tables/rainfall_kernel_summary.tex`
and `gates_and_corrections.tex` (B was unchanged by the correction). D values
are from `climate_horizon_results.md` §5 (docs only; the committed D rows are
pre-correction).*

**Observed.**
- **Kernels.** The rainfall kernels centre at 8–10 weeks. The measured
  weights shift them later by about 0.7 weeks. Agreement with the training-only
  cross-correlation scan is weak (Spearman 0.15–0.31).
- **Gates.** The climate gate rises with horizon, from 0.43 to 0.56. The gated
  climate correction rises from 0.13 to 0.30 on the log-mean scale. B and D
  differ by at most 0.02 on both.

### 4.8 2026 hold-out check

**Purpose.** Check whether the primary contrast carries over to the only data
after the last test year. This is not a blind test: 2026 outcomes had been
seen before the correction.

**Table 11.** D − B, January–May 2026 (19 weeks).

| h | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| D − B (MAE) | +0.18 | +0.28 | +0.28 | +0.50 |

*Source: `climate_horizon_results.md` §2 (docs only).*

**Observed.**
- D had a higher MAE than B at every horizon in 2026.
- At h=4 the 95% interval is [+0.03, +1.00]. It is pointwise and unadjusted,
  from a 4-week block bootstrap with only 5 blocks.
- This agrees in sign with §4.2 and is not a confirmatory result.

---

## Part 2. Why each included result is necessary

| § | Result | Selection criterion | Why the research question needs it |
| --- | --- | --- | --- |
| 4.1 | Reference accuracy (Table 1) | Establishes the baseline | Without it, a null D − B could reflect a weak or broken baseline. B beats persistence at h=2–4 and is within 0.25 MAE of the stored NB reference at h=3–4, so the null is measured on a competent model. |
| 4.2 | D vs B plus robustness (Tables 2–3) | Evaluates the proposed method | This is the direct answer to the question, under the frozen criteria. Table 3 shows the answer does not depend on the estimand, the fold set or the block length. |
| 4.3 | Weights and utilities (Table 4) | Reveals a limitation and supports a finding | It rules out the explanation "the weights were close to 1". It also shows, on the same data, that reliance on weather rises with horizon, which is the premise of the method. |
| 4.4 | Exploratory arms (Tables 5–6) | Meaningful comparison | It separates weighting from routing, lag alignment and utility source. It includes GradNorm (H), the nearest existing method. F is the one configuration that did better than B, so reporting it (with its caveats) is necessary. |
| 4.5 | Factorial (Table 7) | Supports a finding (post-hoc) | It shows the weighting effect depends on lag alignment, which explains why C improved while D did not. It is labelled post-hoc because none of it survives Holm. |
| 4.6 | Training diagnostics (Tables 8–9) | Supports a finding | It quantifies how large the intervention is at the optimiser level. This is the evidence the Discussion needs to interpret the null without speculating. |
| 4.7 | Kernels and gates (Table 10) | Reveals a limitation | It shows the weighting barely changed what was learned. It also prevents a "recovered biological delay" claim. |
| 4.8 | 2026 check (Table 11) | Reveals a limitation | It is the only data after 2025. It is consistent in sign with the retrospective result but is not blind, and that must be stated. |

## Part 3. What was left out, and where it belongs

| Excluded result | Reason | Where it belongs, if anywhere |
| --- | --- | --- |
| Earlier one-week study: losses, activations, tuning, AdamW, clipping, LayerNorm, graphs | Different weather source (GEE); does not test the research question | One or two sentences in the Introduction or Background as motivation; the graph and simplex results can justify design choices in Methods |
| Earlier climate-shuffle ablation at h=1–4 (+1.55 MAE at h=4) | Motivation, on a different weather source | Introduction. Table 4 already shows the premise on this study's data, so repeating it would duplicate the finding. |
| Long-horizon benchmark (h up to 12, onset, conformal intervals, Chronos fine-tuning) | Does not test the research question | Not in this paper |
| RMSE, WIS and 95% coverage for B and D | Add nothing to the comparison: WIS at h=4 is 15.24 (B) vs 15.31 (D); coverage is 0.93–0.96 | Optional single sentence in 4.1 |
| COVID folds (2020–21) separately | Already covered by the all-9-fold row in Table 3 | — |
| Original (routing v1) results | Superseded; the primary null is the same (+0.049 to +0.082) | Experimental Setup or Discussion, as the disclosure of the correction |
| Compute and latency (583 fits, 93.9 GPU-min, 2.38 ms per forecast) | Not a result about the research question | Experimental Setup (required by the course) |
| Per-year D vs B table | The committed `per_fold_B_D.tex` holds pre-correction D values; no corrected per-year table is committed | Add only if the corrected artefact is supplied |

## Part 4. Notes for whoever writes the final version

- **Missing artefacts.** Every "docs only" number depends on
  `results/climate_horizon/correction_2026-09-30/`, which is not committed.
  Commit it, or have a team member confirm the values, before submission.
- **Wording to keep:**
  - "shuffle reliance", not "climate benefit";
  - "learned delay kernel", not "recovered delay";
  - "empirical coverage", not "calibrated".
- **Correction to the docs.** The docs say the weights "rise with horizon in
  all 10 folds". The committed weights show fold 1 is not strictly monotone
  (0.72 → 0.70). Table 4 states this accurately; use its wording.
