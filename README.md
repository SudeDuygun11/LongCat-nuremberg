# Tideglass Causal Renewal Forecaster — Topic 2 Phase 1

This directory contains team **LongCat**'s complete Huawei 2026 Nuremberg Tech Arena Topic 2 Phase-1 release, prepared by **Sude Duygun** (`sudeduygun11@gmail.com`). It predicts the EAGLE-I county outage ratio `predicted_x = customers_out / Customers` for Task A and Task B. The model, frozen issue-time features, prediction file, report, validation evidence, source declarations, and package manifests describe one hash-bound release.

## Final deliverables

- `LongCat_Challenge2.pdf` — 3–8 page technical report
- `predictions.csv` — combined organizer-format Task A/B predictions
- `LongCat_Challenge2.zip` — complete code/model/reproduction package
- `LongCat_Challenge2.zip.sha256` and `MANIFEST.sha256` — release bindings

The CSV contains 65,880 rows: 22,080 Task-A rows and 43,800 Task-B rows across Cook, Bergen, Kings, Queens, and Suffolk counties.

## Fast exact reproduction (no retraining)

Working directory: the unpacked package root (`LongCat_Challenge2`).

```bash
cd /path/to/LongCat_Challenge2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
PYTHONPATH=src python scripts/predict.py --output reproduced_predictions.csv
sha256sum predictions.csv reproduced_predictions.csv
```

Both files must have SHA-256 `ddb7129d0123d5e765a26c78ca0f453b91b749bbd12ceef43edec5b0eb75d1b2`. Typical frozen inference is under 10 seconds on a laptop.

## Full acquisition and retraining

EAGLE-I v4 is public but multi-gigabyte. Either point the extractor at an existing Figshare data directory or allow it to download the three annual CSVs and `MCC.csv`.

```bash
cd /path/to/LongCat_Challenge2
source .venv/bin/activate
PYTHONPATH=src python scripts/download_data.py --source-dir /path/to/eaglei_figshare_24237376_v4/data
# Or download the declared public source: PYTHONPATH=src python scripts/download_data.py --download-full
PYTHONPATH=src python scripts/train.py
PYTHONPATH=src python scripts/predict.py
PYTHONPATH=src python scripts/build_report.py
PYTHONPATH=src python scripts/package_release.py
PYTHONPATH=src python scripts/release_preflight.py
```

Expected runtimes: extraction 2–8 minutes from local files (download time excluded); feature construction/training 3–6 minutes; frozen inference under 10 seconds; report/package under 30 seconds. CPU and 8 GB RAM are sufficient.

## What is frozen

- `artifacts/model.joblib`: exact fitted Task-A and Task-B preprocessing/Ridge pipelines.
- The same model artifact contains the pre-declared groupwise MAE calibration learned only from pre-2025 forward replays.
- `data/frozen_issue_features.csv.gz`: one causal feature row per submitted prediction, including an auditable `max_source_timestamp` that never exceeds `issue_time`.
- `config/config.json`: counties, MCC denominators, schedules, leads, seed, and blending constants.
- `artifacts/artifact_hashes.json`: bindings for the model, frozen inputs, and prediction bytes.

The exact reproduction path never reads Sep–Nov target labels or a network service.

## Verification

```bash
cd /path/to/LongCat_Challenge2
source .venv/bin/activate
PYTHONPATH=src python scripts/evaluate.py
PYTHONPATH=src python scripts/validate_submission.py
python -m unittest discover -s tests -v
```

See `FINAL_CHECKLIST.md`, `DATA_AND_LICENSES.md`, `MODEL_CARD.md`, and `SUBMISSION_SCHEMA.md` for the final checks, evidence, and interface contracts.
