# Grid event or data artefact? Code, data and results

Code, data and results for the paper *Grid event or data artefact? Hierarchical weak supervision for anomaly
attribution in multi-city feeder telemetry* by Saad Farah, Lahcen Hassine, Hatim Lechgar and Anass Rghioui
(Computing and Smart Systems (CSS), Hassania School of Public Works (EHTP), Casablanca, Morocco).

A zero, a frozen value or a perfectly straight stretch in feeder telemetry can be a real outage or a failure of the
recording chain. The method types every anomaly as an **acquisition artefact** or a **grid event**, with a calibrated
probability and without labelled data:

- 19 labelling rules read the level at which an anomaly appears (feeder, city, region, shared recording chain) and the
  traces it leaves (held values, straight-line fills, exact zeros, noise floors);
- a constrained Dawid–Skene label model turns their sparse votes into probabilistic labels;
- a multi-scale residual detector (a TCN-GAT forecast residual split into region, city and zone parts) adds anomalies
  that no rule covers;
- an end model and a fusion step produce the typed catalogue, which is evaluated on injected events, on outage
  statistics and forecasts, and by transfer to a second city.

## Contents

| Folder | Content |
|---|---|
| `data/raw/` | `Data Morocco.xlsx` (UCI 1158) and `tetouan/Tetuan City power consumption.csv` (UCI 849) |
| `data/external/natural_earth/` | Natural Earth 1:50m land (public domain), for the map figure |
| `src/` | the method: rule engines (`rules_v0.py`, `rules_v1.py`), label model, detector, TCN-GAT predictor, end model and fusion, injection benchmark, downstream test, transfer test, GDN and USAD baselines, figure and table modules |
| `notebooks/` | NB1–NB7, executed, with their outputs |
| `tools/` | scripts that produce the results not made by the notebooks (development selection, ablations, deep baselines, outage and forecasting tests on the 30 test datasets, Snorkel baseline, paired tests, robustness and calibration checks, the cleaning control S1b) and the generators of the notebooks (`make_nb*.py`) |
| `results/` | every output, from rule items and labels to the benchmark runs, the downstream and transfer results, the released catalogue and the paper's tables and figures |

## The typed catalogue

`results/catalogue_final/catalogue.csv` holds 404 entries for the Moroccan data: 327 rule items and 77 detector-only
candidates, each with its type, confidence, probability per type and episode. Its data card is
`results/catalogue_final/README.md`.

| Code | Type | Code | Type |
|---|---|---|---|
| A1 | zero-fill gap (several cities) | G1 | regional outage |
| A2 | simultaneous fill (hold or straight line) | G2 | city outage |
| A3 | single-zone fill | G3 | unplanned feeder outage |
| A4 | clock error | G4 | planned work |
| N | normal or untyped | G5 | load transfer |

Zone codes: L1–L5 (Laâyoune), B1–B3 (Boujdour), F1–F7 (Foum El Oued), M1–M2 (Marrakech). Time stamps are used
**as recorded**, without clock correction.

## Data

| Dataset | Source | Licence |
|---|---|---|
| High-Resolution Load Dataset from Smart Meters Across Various Cities in Morocco | UCI Machine Learning Repository, id 1158, [doi:10.24432/C5FC8R](https://doi.org/10.24432/C5FC8R); data article: Bensalah et al., *Data in Brief* 62 (2025) 112067, [doi:10.1016/j.dib.2025.112067](https://doi.org/10.1016/j.dib.2025.112067) | CC BY 4.0 |
| Power Consumption of Tetouan City | UCI Machine Learning Repository, id 849, [doi:10.24432/C5B034](https://doi.org/10.24432/C5B034) | CC BY 4.0 |
| Natural Earth 1:50m land | naturalearthdata.com | public domain |

`Data Morocco.xlsx` is the file served by the UCI repository (22,434,854 bytes; SHA-256
`507134d67ecbbddbf17640e1937da132d0db8000fbfd32a88dda9a4cbf53c6fc`).

## How to run

The local runs used Python 3.12; NB4 ran on Google Colab with a GPU. `lightgbm` is pinned at 4.6.0 (4.7.0 crashed on the authors' Windows machine); every
`summary.json` records the versions used.

```bash
pip install -r requirements.txt
python src/rules_v0.py "data/raw/Data Morocco.xlsx" results/event_catalogue_v0     # ~20 s
python src/rules_v1.py "data/raw/Data Morocco.xlsx" results/catalogue_v1           # ~30 s
cd notebooks
jupyter nbconvert --to notebook --execute --inplace NB1_evidence_figures.ipynb     # ~2 min
jupyter nbconvert --to notebook --execute --inplace NB1b_rules_v1.ipynb           # ~1 min
jupyter nbconvert --to notebook --execute --inplace NB2_label_model.ipynb         # ~1.5 min
jupyter nbconvert --to notebook --execute --inplace NB3_detector_end_model.ipynb  # ~13 min
cd ..
python tools/dev_select_detector.py      # development selection (seeds 100-104), before NB4, ~35 min
# NB4 (injection benchmark): 2-3 h on a GPU (Colab), ~2.5 h on 8 CPU cores; resumable
python tools/graph_ablation.py           # no-graph ablation, ~30 min
python tools/deep_baselines.py           # GDN and USAD, ~50 min; then tools/gdn_forecast_check.py
python tools/ablation_summary.py         # ablations of the main detector (seconds)
python tools/attribution_wilcoxon.py     # paired tests of the attribution table (seconds)
# NB5 (downstream test) and NB6 (Tetouan transfer, ~13 min): execute like the others
python tools/outage_all_datasets.py      # outage statistics on the 30 test datasets, ~1 h
python tools/forecast_all_datasets.py    # forecasting on the 30 test datasets, ~2.5 h
python tools/marrakech_beta_sensitivity.py
python tools/review_robustness.py        # real-data DM tests with longer bandwidths and block bootstraps; the A1 items (~1 min)
python tools/calibration_by_class.py     # calibration of the attribution per class (seconds)
python tools/s1b_control.py              # cleaning control S1b on the real data and the 30 test datasets, ~70 min; resumable
# NB7 (final catalogue, figures and tables, ~2 min)
```

Notes:

- NB7 needs the LightGBM predictors saved by NB4 (`results/benchmark_v1/models/lightgbm_models.pkl`). The file is not
  included because of its size; re-running NB4 creates it.
- `tools/review_robustness.py` and `tools/s1b_control.py` read NB5's per-step losses
  (`results/downstream_v1/loss_real.pkl`), also left out because of its size; executing NB5 creates it.
- The Snorkel baseline (`tools/snorkel_baseline.py`) runs in a separate environment with `snorkel==0.10.0`; the
  commands are in its docstring.
- Notebooks resolve the repository root by themselves and also run on Google Colab from a copy of the repository on
  Google Drive (`MyDrive/grid-event-or-data-artefact/`).
- Codes such as D60 or F42 in comments refer to the authors' internal research log.

## Licence

Code (`src/`, `tools/`, `notebooks/`): MIT, see `LICENSE`. Results and the typed catalogue (`results/`): CC BY 4.0,
see `LICENSE-DATA.md`. The datasets in `data/` keep their own licences (table above).

## Citation

Please cite the paper (reference to be added on publication) and the datasets above.
