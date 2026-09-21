# Model card — Tideglass Causal Renewal Forecaster

## Intended use

Phase-1 rolling forecasts of EAGLE-I county outage ratio for exactly five declared counties. Task A issues 48 hourly leads; Task B issues 24 quarter-hour leads. Predictions are continuous, finite, and clipped to `[0,1]`.

## Architecture

Tideglass is a deterministic, regularized log-count renewal model, fitted separately by task. It combines county identity, issue/lead geometry, cyclic target time, current and trailing issue-time states, 1/7/14-day recurrence, data staleness, and the cross-county contemporaneous median. A weighted Ridge objective on `log1p(customers_out)` limits tail leverage while retaining positive-event influence. Task B adds a fixed 20% last-state renewal path; Task A uses the structural model alone. A final groupwise scale minimizes outage-ratio MAE: Task A groups by FIPS and lead, while Task B groups by lead with 500-row shrinkage toward its global scale.

## Training and validation

- Seed: 41729.
- Final target cutoff: 31 August 2025, 23:59:59 UTC.
- Final fitted rows: 206,818 (A) and 407,603 (B).
- Honest evaluation: 2023 Sep–Nov selects the calibration form and scales; an untouched expanding-window Sep–Nov 2024 replay evaluates that decision. Final release scales are then refitted on the historical 2024 replay.
- Task-A validation truth: mean of available quarter-hour observations in the hour ending at target time.
- Task-B validation truth: native 15-minute value.
- Missing target observations are excluded from metric denominators; they are never imputed as zero.

## Imbalance treatment

Targets are log-transformed; each sample weight is `1 + min(log1p(y)/10, 1.5)`. This keeps routine intervals dominant for MAE while giving large outages up to 2.5× training weight. Reported validation includes top-decile MAE and p95 absolute error.

## Causality and leakage

Every row carries `max_source_timestamp`. Training and release assertions require `max_source_timestamp <= issue_time`. Recurrence values are only used when their timestamps are already observable at issue time; otherwise a trailing-week median plus a missing indicator is used. The packaged inference path contains no target label column.

## Limitations

- The model cannot anticipate an unseen storm onset without an issue-time weather forecast.
- Utility reporting gaps are treated as missing, not zero; sudden coverage changes can shift the state distribution.
- A county centroid is not used because the model consumes county outage states only.
- Public-replay diagnostics are local validation evidence, not an organizer or leaderboard score.
