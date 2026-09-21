# Tideglass Causal Renewal Forecaster — Topic 2 Phase 1

> 🚧 **ONGOING** — This project is currently under development.

Team **LongCat**'s entry to the Huawei 2026 Nuremberg Tech Arena, Topic 2 (Phase 1). It forecasts the EAGLE-I county outage ratio `predicted_x = customers_out / Customers` for five US counties (Cook, Bergen, Kings, Queens, Suffolk) at two horizons: Task A (48 hourly leads) and Task B (24 quarter-hour leads). The main goal is a leakage-free, reproducible forecaster whose inputs are auditable against each issue time.

**Author:** Sude Duygun (sudeduygun11@gmail.com)

## Overview

- **Motivation:** [TODO]
- **Problem context:** Rolling forecasts of county-level electricity outage ratio from EAGLE-I outage data (v4), with strict issue-time causality.
- **Main objective:** Predict a continuous, finite outage ratio clipped to `[0, 1]` for every declared county, issue time, and lead.
- **Expected outcome:** A hash-bound release consisting of the model, frozen issue-time features, `predictions.csv`, a technical report, and validation evidence.

**Model in brief** (details in [MODEL_CARD.md](MODEL_CARD.md)): a deterministic, regularized log-count renewal model (weighted Ridge on `log1p(customers_out)`), fitted separately per task. Task B adds a fixed 20% last-state renewal path. A final groupwise scale minimizes outage-ratio MAE.

| Item | Value |
|---|---|
| Counties | Cook, Bergen, Kings, Queens, Suffolk |
| Task A | 48 hourly leads (22,080 prediction rows) |
| Task B | 24 quarter-hour leads (43,800 prediction rows) |
| Total prediction rows | 65,880 |
| Data | EAGLE-I v4 (DOI `10.6084/m9.figshare.24237376.v4`, CC BY 4.0) |
| Seed | 41729 |
| Target cutoff | 31 Aug 2025, 23:59:59 UTC |

## Project Status

- [x] Data acquisition/extraction script (`scripts/download_data.py`)
- [x] Causal issue-time feature construction and frozen features
- [x] Model training (Task A and Task B) and calibration
- [x] Validation on Sep–Nov 2024 expanding-window replay
- [x] Combined Task A/B `predictions.csv` and submission validator
- [x] Technical report (`LongCat_Challenge2.pdf`) and release manifest
- [ ] Work in progress: [TODO]
- [ ] Planned experiment/feature: [TODO]
- [ ] Final evaluation: [TODO]
- [ ] Documentation: [TODO]

## Objectives

1. Forecast county outage ratio for Task A (hourly) and Task B (quarter-hourly) across five counties.
2. Guarantee causality: every feature row satisfies `max_source_timestamp <= issue_time`.
3. Provide exact, network-free reproduction of the submitted predictions from frozen artifacts.
4. Handle outage-tail imbalance without sacrificing MAE on routine intervals.

## Methodology / Approach

1. **Input:** EAGLE-I 15-minute county outage counts and `MCC.csv` customer denominators (2023–2025).
2. **Preprocessing:** Extract the five counties; treat reporting gaps as missing, never as zero.
3. **Feature engineering:** County identity, issue/lead geometry, cyclic target time, current and trailing issue-time states, 1/7/14-day recurrence, data staleness, and cross-county contemporaneous median.
4. **Model:** Weighted Ridge on `log1p(customers_out)`, one per task. Sample weight is `1 + min(log1p(y)/10, 1.5)`.
5. **Training:** Final fit on 206,818 (A) and 407,603 (B) rows. Groupwise MAE calibration: Task A by FIPS and lead, Task B by lead with 500-row shrinkage.
6. **Evaluation:** 2023 Sep–Nov selects the calibration form; untouched Sep–Nov 2024 replay evaluates it. Metrics include MAE, top-decile MAE, and p95 absolute error.
7. **Analysis:** See the technical report and `artifacts/validation_summary.json`.

```text
EAGLE-I data → Causal preprocessing → Ridge renewal model + calibration → Replay validation → predictions.csv
```

## Repository Structure

```text
LongCat_Challenge2/
├── README.md
├── MODEL_CARD.md                  # model, training, limitations
├── DATA_AND_LICENSES.md           # data sources and licenses
├── SUBMISSION_SCHEMA.md           # CSV interface contract
├── FINAL_CHECKLIST.md
├── LongCat_Challenge2.pdf         # technical report
├── predictions.csv                # Task A/B predictions
├── MANIFEST.sha256
├── requirements.lock
├── src/longcat/                   # library code (io.py, pipeline.py)
├── scripts/                       # download, train, predict, evaluate, report, package
├── config/config.json             # counties, denominators, schedules, constants
├── data/                          # frozen_issue_features.csv.gz
├── artifacts/                     # model.joblib, metrics, hashes, provenance
├── report/                        # report template, HTML, figures
└── tests/                         # causality, predictions, risk tests
```

## Getting Started

Requires Python 3 (CPU and 8 GB RAM are sufficient).

```bash
git clone https://github.com/SudeDuygun11/LongCat-nuremberg.git
cd LongCat-nuremberg
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.lock
```

### Exact reproduction (no retraining)

```bash
PYTHONPATH=src python scripts/predict.py --output reproduced_predictions.csv
sha256sum predictions.csv reproduced_predictions.csv
```

Both files must have SHA-256 `ddb7129d0123d5e765a26c78ca0f453b91b749bbd12ceef43edec5b0eb75d1b2`.

### Full acquisition and retraining

EAGLE-I v4 is multi-gigabyte. Use a local copy or let the script download it.

```bash
PYTHONPATH=src python scripts/download_data.py --source-dir /path/to/eaglei_figshare_24237376_v4/data
# or: PYTHONPATH=src python scripts/download_data.py --download-full
PYTHONPATH=src python scripts/train.py
PYTHONPATH=src python scripts/predict.py
PYTHONPATH=src python scripts/build_report.py
PYTHONPATH=src python scripts/package_release.py
PYTHONPATH=src python scripts/release_preflight.py
```

### Verification

```bash
PYTHONPATH=src python scripts/evaluate.py
PYTHONPATH=src python scripts/validate_submission.py
python -m unittest discover -s tests -v
```

## Results

Validation metrics are in `artifacts/validation_metrics.csv` and `artifacts/validation_summary.json`.

| Task | Metric | Value |
|---|---|---|
| A | MAE | [TODO] |
| B | MAE | [TODO] |

Public-replay diagnostics are local validation evidence, not an organizer or leaderboard score.

## Limitations

- Cannot anticipate an unseen storm onset without an issue-time weather forecast.
- Utility reporting gaps are treated as missing, not zero; sudden coverage changes can shift the state distribution.
- No weather, cause labels, or external forecasts are used.

## Data and Licenses

EAGLE-I v4 is licensed CC BY 4.0. Software dependencies are pinned in `requirements.lock`. See [DATA_AND_LICENSES.md](DATA_AND_LICENSES.md). Repository license: [TODO]

## Roadmap

- [ ] [TODO]

## Contact

Sude Duygun — sudeduygun11@gmail.com
