#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader

from longcat.io import ROOT, load_config, sha256_file
from longcat.pipeline import PREDICTION_COLUMNS

ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def validate_predictions(path: Path = ROOT / "predictions.csv") -> list[str]:
    cfg = load_config()
    frame = pd.read_csv(path, dtype={"fips_code": str})
    assert frame.columns.tolist() == PREDICTION_COLUMNS
    assert len(frame) == cfg["contract"]["expected_rows"]
    assert frame.groupby("task_id").size().to_dict() == {"A": 22080, "B": 43800}
    assert frame["fips_code"].nunique() == 5 and set(frame["fips_code"]) == {x["fips_code"] for x in cfg["counties"]}
    assert frame[PREDICTION_COLUMNS[:-1]].notna().all().all()
    assert np.isfinite(frame["predicted_x"]).all() and frame["predicted_x"].between(0, 1).all()
    assert frame[["task_id", "fips_code", "issue_time", "target_time"]].duplicated().sum() == 0
    assert frame["issue_time"].map(lambda value: bool(ISO.match(value))).all()
    assert frame["target_time"].map(lambda value: bool(ISO.match(value))).all()
    issue = pd.to_datetime(frame["issue_time"], utc=True)
    target = pd.to_datetime(frame["target_time"], utc=True)
    lead = (target - issue).dt.total_seconds().astype(int) // 60
    for task_id, key in (("A", "task_a"), ("B", "task_b")):
        task = frame["task_id"] == task_id
        assert sorted(lead[task].unique()) == cfg["contract"][key]["lead_minutes"]
        expected_batch = 5 * len(cfg["contract"][key]["lead_minutes"])
        assert frame[task].groupby("issue_time").size().eq(expected_batch).all()
        assert frame[task].groupby(["issue_time", "fips_code"]).size().eq(len(cfg["contract"][key]["lead_minutes"])).all()
    return ["exact schema and 65,880-row Task A/B Cartesian contract", "canonical UTC timing, complete horizons, unique keys, finite ratios"]


def validate_frozen() -> list[str]:
    frame = pd.read_csv(ROOT / "data" / "frozen_issue_features.csv.gz", dtype={"fips_code": str})
    assert len(frame) == 65880
    assert (pd.to_datetime(frame["max_source_timestamp"], utc=True) <= pd.to_datetime(frame["issue_time"], utc=True)).all()
    stored = json.loads((ROOT / "artifacts" / "artifact_hashes.json").read_text())
    assert sha256_file(ROOT / "artifacts" / "model.joblib") == stored["model.joblib"]
    assert sha256_file(ROOT / "data" / "frozen_issue_features.csv.gz") == stored["frozen_issue_features.csv.gz"]
    if "predictions.csv" in stored:
        assert sha256_file(ROOT / "predictions.csv") == stored["predictions.csv"]
    return ["frozen causal inputs obey source-time <= issue-time", "model, frozen-input, and prediction hashes"]


def validate_pdf(path: Path = ROOT / "LongCat_Challenge2.pdf") -> list[str]:
    reader = PdfReader(path)
    assert 3 <= len(reader.pages) <= 8
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    for required in ["Tideglass", "LongCat", "Sude Duygun", "sudeduygun11@gmail.com", "Task A", "Task B", "acquisition", "missing", "timestamp", "feature", "training", "imbalance", "ablation", "Sources and licenses"]:
        assert required.lower() in text.lower(), required
    assert len(text) > 7000
    return [f"technical report length ({len(reader.pages)} pages) and required content"]


def validate_zip(path: Path = ROOT / "LongCat_Challenge2.zip") -> list[str]:
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())
        required = {"LongCat_Challenge2.pdf", "predictions.csv", "README.md", "FINAL_CHECKLIST.md", "MANIFEST.sha256", "artifacts/model.joblib", "data/frozen_issue_features.csv.gz", "scripts/predict.py", "requirements.lock"}
        assert required <= names, sorted(required - names)
        assert not any("selected_raw" in name or "/work/" in name or "/raw/" in name for name in names)
    return ["ZIP CRC, required release members, and raw-data exclusion"]


def main(include_zip: bool = True) -> list[str]:
    checks = validate_predictions() + validate_frozen() + validate_pdf()
    if include_zip:
        checks += validate_zip()
    for check in checks:
        print(f"PASS: {check}")
    return checks


if __name__ == "__main__":
    main()
