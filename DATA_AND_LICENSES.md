# Data and licenses

## EAGLE-I county outage data and MCC denominator

- **Dataset:** *The Environment for Analysis of Geo-Located Energy Information’s Recorded Electricity Outages 2014–2025*, version 4.
- **Publisher:** EAGLE-I / Oak Ridge National Laboratory authors via Figshare.
- **DOI:** `10.6084/m9.figshare.24237376.v4`.
- **Files used:** `eaglei_outages_2023.csv`, `eaglei_outages_2024.csv`, `eaglei_outages_2025.csv`, and `MCC.csv`.
- **License:** Creative Commons Attribution 4.0 (CC BY 4.0).
- **Role:** 15-minute county outage counts, the historical/rolling issue-time state, validation targets, and fixed `Customers` denominators.

The packaged release excludes the reconstructible multi-gigabyte raw files and the temporary five-county extracts. `scripts/download_data.py` reacquires/extracts them. Exact submitted bytes instead use the small, target-free `data/frozen_issue_features.csv.gz`.

## Organizer material

The implementation follows the official Phase-1 submission guideline and CSV example acquired from the challenge Drive on 23 August 2026. Those materials define the county-ratio target, five-county choice, rolling issuance, Task-A/Task-B horizons, test window, and seven-column CSV. They are challenge specification material, not model training data, and are not redistributed here.

## Software

Dependency versions are frozen in `requirements.lock`; license identifiers are recorded in `artifacts/dependency_licenses.json`. Core licenses are BSD-3-Clause (NumPy, pandas, scikit-learn, SciPy, joblib, Jinja2, pypdf), Apache-2.0 (Requests), and MIT (Plotly, Kaleido).

## Excluded sources

No weather observation, weather reanalysis, external forecast API, outage-cause label, private organizer target, or post-cutoff target label is used. Excluding weather avoids substituting target-time observations for genuine issue-time forecasts and makes the frozen information boundary directly auditable.
