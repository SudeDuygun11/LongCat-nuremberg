#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path

from longcat.io import ROOT, sha256_file
from predict import generate


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        reproduced = Path(directory) / "predictions.csv"
        generate(ROOT / "data" / "frozen_issue_features.csv.gz", reproduced)
        expected_hash = sha256_file(ROOT / "predictions.csv")
        actual_hash = sha256_file(reproduced)
        if actual_hash != expected_hash:
            raise SystemExit(f"byte reproduction failed: {actual_hash} != {expected_hash}")
        print(f"PASS: byte-identical prediction reproduction ({actual_hash})")
