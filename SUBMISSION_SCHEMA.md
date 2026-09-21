# Submission schema

`predictions.csv` uses this exact ordered schema:

1. `task_id` — `A` or `B`
2. `fips_code` — five-character zero-padded county FIPS
3. `county` — human-readable county name
4. `state` — human-readable state name
5. `issue_time` — ISO-8601 UTC (`YYYY-MM-DDTHH:MM:SSZ`)
6. `target_time` — ISO-8601 UTC
7. `predicted_x` — finite continuous outage ratio in `[0,1]`

The primary key is `(task_id, fips_code, issue_time, target_time)`.

- Task A: 92 daily issues × 5 counties × 48 hourly leads = **22,080 rows**.
- Task B: 365 six-hourly issues × 5 counties × 24 quarter-hour leads = **43,800 rows**.
- Combined: **65,880 rows**.

Every batch contains all counties and the exact full lead set. Pre-window issues cover the first test-day targets; complete horizons are retained even when a terminal target falls immediately outside the half-open scoring window.
