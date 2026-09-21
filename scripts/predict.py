#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from longcat.io import ROOT
from longcat.pipeline import submission_from_features


def generate(feature_path: Path, output_path: Path) -> None:
    model = joblib.load(ROOT / "artifacts" / "model.joblib")
    features = pd.read_csv(feature_path, dtype={"fips_code": str, "issue_hour": str, "lead_band": str})
    predictions = submission_from_features(model, features)
    predictions.to_csv(output_path, index=False, float_format="%.10f", lineterminator="\n")
    print(f"Wrote {len(predictions):,} rows to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce the frozen Topic-2 prediction CSV")
    parser.add_argument("--features", type=Path, default=ROOT / "data" / "frozen_issue_features.csv.gz")
    parser.add_argument("--output", type=Path, default=ROOT / "predictions.csv")
    args = parser.parse_args()
    generate(args.features, args.output)
