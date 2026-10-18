# AEGIS-Dengue

District-level dengue early warning for Sri Lanka. AEGIS-Dengue forecasts
weekly dengue case counts for all 25 districts, 1–4 weeks ahead (up to 12 in
the long-horizon benchmark), from case history and daily ERA5-Land / CHIRPS
weather aggregated to the reporting week. The data covers 1,012 weekly
reporting periods, 2006-12-23 to 2026-05-17.

Every model is scored on a walk-forward protocol: one test year per fold,
preprocessing fit on the training years only, and the forecasts compared with
same-horizon persistence ("next week looks like this week"). Persistence is the
baseline that matters, because it is hard to beat at short horizons.

**Team:** Nadil Hesara, Thilokya Angeesa, Hesandi Mallawarachchi,
Nethsith Gunaweera, Senilka Thisangi.

---

## Main findings

![Skill over same-horizon persistence, by forecast horizon](figures/fig_skill_by_horizon.png)

1. **One week ahead, persistence is hard to beat.** Five separate changes to
   the GCN+GRU baseline (loss, activations, hyperparameters, optimiser, graph)
   each moved the headline MAE by less than the seed noise.
   ([research log §6–§8e](docs/research_log.md))
2. **Spatial graphs did not help.** No adjacency (queen contiguity, a Gaussian
   distance kernel or a learned graph) beat the identity "no graph" control.
   ([graph_representation.md](docs/graph_representation.md))
3. **Three to four weeks ahead the model has real skill.** It beats
   same-horizon persistence on MAE and peak MAE at h=3–4, on 6–7 of 7 headline
   folds (p < 0.01). A shuffle control shows that about half of that skill at
   h=4 comes from the weather's timing.
   ([multi_horizon.md](docs/multi_horizon.md),
   [climate_ablation.md](docs/climate_ablation.md))
4. **Long-horizon benchmark (h = 1–12).** From h=3 on, almost every learned
   model beats persistence, and the margin grows with the horizon (+14% at
   h=4, +19–25% at h=12, 7/7 folds). A zero-shot foundation model (Chronos-2)
   matches the trained GRU, and LightGBM and ridge baselines trail it. Observed
   weather beats a training-years climatology in all three model families. On
   the 2026 hold-out, all five declared models beat persistence at every
   horizon, although most bootstrap intervals include zero. ([long_horizon_benchmark.md](docs/long_horizon_benchmark.md),
   [archive/revision_2026_09_30.md](docs/archive/revision_2026_09_30.md))
5. **Climate-horizon extension: a negative primary result.** Weighting the
   climate encoder's gradients by measured per-horizon climate reliance did
   not improve on an otherwise identical equal-weight model (D − B = +0.06
   to +0.08 MAE, Holm p ≥ 0.44). It was non-inferior at h=1–2.
   ([climate_horizon_results.md](docs/climate_horizon_results.md))

Headline MAE at h=1 / 2 / 3 / 4 on the seven headline test years (2017–19,
2022–25):

| Model | h=1 | h=2 | h=3 | h=4 |
| --- | --- | --- | --- | --- |
| Persistence | 16.42 | 20.16 | 24.58 | 28.47 |
| GRU, Negative Binomial head | 15.94 | 19.26 | 22.46 | 25.07 |
| Two-branch climate model (B) | 16.64 | 19.67 | 22.71 | 25.22 |
| Chronos-2, zero-shot | 15.96 | 19.07 | 22.29 | 25.11 |

Source: `docs/climate_horizon_results.md` §3.

---

## Repository layout

```text
src/            reusable code: data, models, training, evaluation, inference
scripts/        numbered entry points, grouped by stage
  data/           0–11   download, validate, calendar, climate, master panel
  features/       12, 14 model tensors (v0–v4), walk-forward folds
  graph/          13     district adjacency
  training/       16–59  every model and experiment
  evaluation/     15–61  baselines, diagnostics, statistics, figures
  inference/             predict_dengue.py
tests/          pytest suite
configs/        data, model and training defaults
data/           raw sources (committed where small) and generated files
results/        committed result tables, manifests and figures
figures/        paper figures (PDF + PNG)
notebooks/      data loading, ERA5 extraction and EDA notebooks
docs/           one design or results document per topic
```

The script numbers record the order the experiments were added. Within a
subfolder a few numbers are shared by two scripts (for example
`training/31.train_negbin.py` and `training/31.long_horizon_neural.py`); the
names, not the numbers, identify a script.

---

## Setup

Python 3.11 or 3.12.

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install -r requirements-model.txt
```

`requirements-model.txt` pins PyTorch 2.13.0. For an NVIDIA GPU, install the
matching CUDA wheel instead, for example
`pip install torch --index-url https://download.pytorch.org/whl/cu128`.
The models are small (about 10k parameters) and train on a CPU.

### Data

The dengue counts (`data/raw/srilanka_weekly_data.csv`), CHIRPS rainfall and
GADM district boundaries are committed. The daily climate file is not
(52 MB). Rebuild it from Open-Meteo's ERA5 archive (no account needed):

```powershell
.venv\Scripts\python.exe scripts\data\5c.fetch_open_meteo_climate.py
```

The original Google Earth Engine extraction (`scripts/data/5*.py`, and
`notebooks/1.era5_extraction.ipynb`, written for Google Colab) needs an Earth
Engine account; see [running_from_zero.md](docs/running_from_zero.md).

---

## Reproducing the results

```powershell
$py = ".venv\Scripts\python.exe"

# Data pipeline
& $py scripts\data\2.create_calendar.py
& $py scripts\data\4.create_canonical_dengue.py
& $py scripts\data\9.aggregate_climate_to_periods.py
& $py scripts\data\10.create_master_panel.py
& $py scripts\features\12.build_model_tensors.py
& $py scripts\graph\13.build_adjacency.py
& $py scripts\features\14.build_folds.py

# Baselines and the original model
& $py scripts\evaluation\15.evaluate_naive_baselines.py
& $py scripts\training\16.train_gcn_gru.py --seeds 3
& $py scripts\training\27.train_multi_horizon.py
& $py scripts\training\29.train_climate_ablation.py

# Long-horizon benchmark, run in this order (bash, e.g. Git Bash)
bash scripts/run_revision.sh data
bash scripts/run_revision.sh gpu
bash scripts/run_revision.sh cpu
bash scripts/run_revision.sh analysis

# Paper figures
& $py scripts\evaluation\61.make_paper_figures.py
```

Most training scripts accept `--folds 8 --seeds 1` for a quick smoke run.
The full command list, run times and the values to expect at each step are in
[running_from_zero.md](docs/running_from_zero.md). The climate-horizon
extension has its own frozen run protocol, described in
[climate_horizon_implementation.md](docs/climate_horizon_implementation.md).

### Tests

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

CI runs the suite on Python 3.11 and 3.12 (`.github/workflows/ci.yml`).

---

## Documentation

| Start here | |
| --- | --- |
| [research_log.md](docs/research_log.md) | The full experiment record up to 2026-09-28 (the previous README) |
| [running_from_zero.md](docs/running_from_zero.md) | Every command to rebuild the project, in order |
| [climate_horizon_results.md](docs/climate_horizon_results.md) | Latest results: the climate-horizon extension |
| [long_horizon_benchmark.md](docs/long_horizon_benchmark.md) | h = 1–12 benchmark, strong baselines, intervals, early warning |

Topic documents: data ([reporting_calendar](docs/reporting_calendar.md),
[climate_dataset_schema](docs/climate_dataset_schema.md),
[model_tensors](docs/model_tensors.md)); models and experiments
([baseline](docs/baseline.md), [improvements](docs/improvements.md),
[learnable_lags_results](docs/learnable_lags_results.md),
[simplex_activations](docs/simplex_activations.md),
[hyperparameter_tuning](docs/hyperparameter_tuning.md),
[optimiser_scheduling](docs/optimiser_scheduling.md),
[graph_representation](docs/graph_representation.md),
[gated_fusion](docs/gated_fusion.md),
[multi_horizon](docs/multi_horizon.md), [layer_norm](docs/layer_norm.md),
[climate_ablation](docs/climate_ablation.md),
[negative_binomial](docs/negative_binomial.md),
[new_findings_report](docs/new_findings_report.md),
[gradient_clipping](docs/gradient_clipping.md),
[outbreak_history_features](docs/outbreak_history_features.md),
[residual_analysis](docs/residual_analysis.md)).
Superseded plans and audit trails are in [docs/archive/](docs/archive/).

---

## Data sources

| Data | Source | Terms |
| --- | --- | --- |
| Weekly district dengue counts | Sri Lanka Epidemiology Unit Weekly Epidemiological Reports, via [denguedatahub](https://github.com/thiyangt/denguedatahub) | See the source repository |
| Daily weather | ERA5-Land / ERA5 (Copernicus Climate Change Service), via Google Earth Engine or Open-Meteo | Copernicus licence; attribution required |
| Daily rainfall | CHIRPS (UCSB Climate Hazards Center) | Public domain |
| District boundaries | GADM 4.1, level 1 | Free for academic use; redistribution not permitted |

## Licence

The code is released under the [MIT Licence](LICENSE). The datasets above
remain under their providers' terms.
