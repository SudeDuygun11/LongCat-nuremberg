#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn

from longcat.io import ROOT, load_config, sha256_file, write_json
from longcat.pipeline import MODEL_FEATURES, fit_calibration, fit_models, load_selected_files, make_feature_rows, metric_bundle, predict_counts, release_schedule, task_schedule


def schedule_between(start: str, end: str) -> list[tuple[str, pd.Timestamp]]:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    return [("A", x) for x in task_schedule("A", start_ts, end_ts)] + [("B", x) for x in task_schedule("B", start_ts, end_ts)]


def clean_targets(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["observed_customers_out"].notna()].reset_index(drop=True)


def evaluate_methods(model: dict, frame: pd.DataFrame, fold: str) -> list[dict]:
    rows: list[dict] = []
    for task_id in ("A", "B"):
        task = frame[frame["task_id"] == task_id].copy()
        structural = dict(model)
        structural["blend"] = dict(model["blend"])
        structural["blend"][task_id] = 0.0
        structural.pop("calibration", None)
        uncalibrated = dict(model)
        uncalibrated.pop("calibration", None)
        methods = {
            "weekly_routine": task["median_168h"].to_numpy(dtype=float),
            "last_observation": task["last"].to_numpy(dtype=float),
            "renewal_structure": predict_counts(structural, task),
            "uncalibrated_renewal": predict_counts(uncalibrated, task),
            "tideglass_full": predict_counts(model, task),
        }
        for name, prediction in methods.items():
            rows.append({"fold": fold, "task_id": task_id, "method": name, **metric_bundle(task, prediction)})
    return rows


def main() -> None:
    started = time.time()
    cfg = load_config()
    np.random.seed(int(cfg["seed"]))
    selected = ROOT / "data" / "selected_raw"
    paths = [selected / f"eaglei_selected_{year}.csv" for year in (2023, 2024, 2025)]
    if not all(path.exists() for path in paths):
        raise FileNotFoundError("Run scripts/download_data.py first; selected EAGLE-I extracts are missing")
    panel = load_selected_files(paths)
    expected_fips = {item["fips_code"] for item in cfg["counties"]}
    if set(panel.series) != expected_fips:
        raise RuntimeError(f"county coverage mismatch: {set(panel.series)}")

    work = ROOT / "data" / "work"
    work.mkdir(exist_ok=True)
    cached = [work / "features_2023.pkl.gz", work / "features_2024.pkl.gz", work / "features_2025_pre.pkl.gz"]
    if all(path.exists() for path in cached):
        print("Loading cached causal training rows...", flush=True)
        y2023, y2024, y2025_pre = (pd.read_pickle(path) for path in cached)
    else:
        print("Building causal training rows...", flush=True)
        y2023 = clean_targets(make_feature_rows(panel, schedule_between("2023-01-01", "2023-12-31 18:00"), True))
        y2024 = clean_targets(make_feature_rows(panel, schedule_between("2024-01-01", "2024-12-31 18:00"), True))
        y2025_pre = clean_targets(make_feature_rows(panel, schedule_between("2025-01-01", "2025-08-31 18:00"), True))
        for frame, path in zip((y2023, y2024, y2025_pre), cached):
            frame.to_pickle(path)

    train_2023 = y2023[y2023["issue_time"] < pd.Timestamp("2023-09-01", tz="UTC")].reset_index(drop=True)
    validation_2023 = y2023[y2023["issue_time"].between(pd.Timestamp("2023-09-01", tz="UTC"), pd.Timestamp("2023-11-30 18:00", tz="UTC"))].reset_index(drop=True)
    train_2024 = pd.concat([y2023, y2024[y2024["issue_time"] < pd.Timestamp("2024-09-01", tz="UTC")]], ignore_index=True)
    validation_2024 = y2024[y2024["issue_time"].between(pd.Timestamp("2024-09-01", tz="UTC"), pd.Timestamp("2024-11-30 18:00", tz="UTC"))].reset_index(drop=True)

    validation_rows: list[dict] = []
    print(f"Fitting 2023_forward on {len(train_2023):,} rows...", flush=True)
    model_2023 = fit_models(train_2023, cfg)
    validation_rows.extend(evaluate_methods(model_2023, validation_2023, "2023_forward"))
    historical_calibration = fit_calibration(model_2023, validation_2023, cfg)["calibration"]

    print(f"Fitting 2024_forward on {len(train_2024):,} rows...", flush=True)
    model_2024 = fit_models(train_2024, cfg)
    model_2024["calibration"] = historical_calibration
    validation_rows.extend(evaluate_methods(model_2024, validation_2024, "2024_forward"))

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    metrics = pd.DataFrame(validation_rows)
    metrics.to_csv(artifacts / "validation_metrics.csv", index=False, float_format="%.12g", lineterminator="\n")
    write_json(artifacts / "validation_summary.json", {"design": "2023 Sep-Nov selects the declared groupwise calibration; the untouched 2024 Sep-Nov expanding-window replay evaluates it; all features obey source timestamp <= issue_time", "official_score_claimed": False, "rows": validation_rows})

    final_training = pd.concat([y2023, y2024, y2025_pre], ignore_index=True)
    final_training = final_training[pd.to_datetime(final_training["target_time"], utc=True) <= pd.Timestamp(cfg["model"]["training_cutoff_utc"])].reset_index(drop=True)
    print(f"Fitting final frozen model on {len(final_training):,} rows...", flush=True)
    model = fit_models(final_training, cfg)
    final_calibration_model = fit_models(train_2024, cfg)
    final_calibration_model = fit_calibration(final_calibration_model, validation_2024, cfg)
    model["calibration"] = final_calibration_model["calibration"]
    model.update({"method": cfg["project"]["method_name"], "seed": int(cfg["seed"]), "training_cutoff_utc": cfg["model"]["training_cutoff_utc"]})
    joblib.dump(model, artifacts / "model.joblib", compress=3)

    print("Building 65,880 causal issue-time feature rows...", flush=True)
    frozen = make_feature_rows(panel, release_schedule(cfg), include_target=False, config=cfg)
    for column in ("issue_time", "target_time", "max_source_timestamp"):
        frozen[column] = pd.to_datetime(frozen[column], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    frozen_path = ROOT / "data" / "frozen_issue_features.csv.gz"
    frozen.to_csv(frozen_path, index=False, float_format="%.12g", lineterminator="\n", compression={"method": "gzip", "compresslevel": 9, "mtime": 0})

    write_json(artifacts / "feature_schema.json", {"model_features": MODEL_FEATURES, "target": "log1p(customers_out), converted to predicted_x with the organizer MCC denominator", "causal_invariant": "max_source_timestamp <= issue_time for every frozen row", "task_a_truth_for_validation": "mean of available EAGLE-I quarter-hour values in the hour ending at target_time", "task_b_truth_for_validation": "native 15-minute EAGLE-I value at target_time"})
    write_json(artifacts / "runtime.json", {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "joblib": importlib.metadata.version("joblib"), "seed": int(cfg["seed"]), "training_seconds": round(time.time() - started, 2), "final_training_rows": int(len(final_training)), "frozen_inference_rows": int(len(frozen))})
    write_json(artifacts / "artifact_hashes.json", {"model.joblib": sha256_file(artifacts / "model.joblib"), "frozen_issue_features.csv.gz": sha256_file(frozen_path)})
    print(json.dumps({"training_rows": model["training_rows"], "frozen_rows": len(frozen)}, indent=2))


if __name__ == "__main__":
    main()
