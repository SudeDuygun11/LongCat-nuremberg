#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import requests

from longcat.io import ROOT, load_config, write_json

ARTICLE_API = "https://api.figshare.com/v2/articles/24237376"
YEARS = (2023, 2024, 2025)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def download(url: str, path: Path) -> None:
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with path.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire the public EAGLE-I source and extract the five declared counties")
    parser.add_argument("--source-dir", type=Path, help="Existing directory containing EAGLE-I annual CSVs and MCC.csv")
    parser.add_argument("--download-full", action="store_true", help="Download the multi-GB public annual files from Figshare")
    args = parser.parse_args()
    cfg = load_config()
    fips = {item["fips_code"] for item in cfg["counties"]}
    raw = ROOT / "data" / "raw"
    selected = ROOT / "data" / "selected_raw"
    raw.mkdir(parents=True, exist_ok=True)
    selected.mkdir(parents=True, exist_ok=True)

    if args.source_dir:
        source = args.source_dir
        metadata = requests.get(ARTICLE_API, timeout=60).json()
    elif args.download_full:
        metadata = requests.get(ARTICLE_API, timeout=60).json()
        source = raw
        by_name = {item["name"]: item for item in metadata["files"]}
        for name in [f"eaglei_outages_{year}.csv" for year in YEARS] + ["MCC.csv"]:
            path = source / name
            if not path.exists():
                print(f"Downloading {name} ({by_name[name]['size']:,} bytes)...", flush=True)
                download(by_name[name]["download_url"], path)
    else:
        raise SystemExit("Provide --source-dir /path/to/eaglei/data or use --download-full")

    records = []
    for year in YEARS:
        source_path = source / f"eaglei_outages_{year}.csv"
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        pieces = []
        for chunk in pd.read_csv(source_path, dtype={"fips_code": str}, chunksize=1_000_000):
            chunk["fips_code"] = chunk["fips_code"].str.zfill(5)
            match = chunk[chunk["fips_code"].isin(fips)]
            if len(match):
                pieces.append(match)
        extract = pd.concat(pieces, ignore_index=True)
        output = selected / f"eaglei_selected_{year}.csv"
        extract.to_csv(output, index=False, lineterminator="\n")
        records.append({"year": year, "source_file": source_path.name, "selected_rows": len(extract), "selected_sha256": digest(output)})
        print(f"{year}: {len(extract):,} selected rows")

    write_json(ROOT / "artifacts" / "source_extract_manifest.json", {
        "article": metadata.get("title"), "doi": metadata.get("doi"), "license": metadata.get("license"),
        "article_api": ARTICLE_API, "counties": sorted(fips), "extracts": records,
    })


if __name__ == "__main__":
    main()
