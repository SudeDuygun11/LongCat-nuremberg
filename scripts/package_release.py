#!/usr/bin/env python3
from __future__ import annotations

import zipfile
from pathlib import Path

from longcat.io import ROOT, sha256_file

EXCLUDED_PARTS = {"raw", "selected_raw", "work", "cache", "build", "tmp", "__pycache__", ".venv"}
EXCLUDED_NAMES = {
    "LongCat_Challenge2.zip",
    "LongCat_Challenge2.zip.sha256",
    "MANIFEST.sha256",
}


def release_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.name in EXCLUDED_NAMES or path.suffix == ".pyc" or path.name == ".DS_Store":
            continue
        files.append(path)
    return sorted(files, key=lambda value: value.relative_to(ROOT).as_posix())


def main() -> None:
    files = release_files()
    manifest = ROOT / "MANIFEST.sha256"
    manifest.write_text("".join(f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in files), encoding="utf-8")
    files.append(manifest)
    target = ROOT / "LongCat_Challenge2.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(files, key=lambda value: value.relative_to(ROOT).as_posix()):
            archive.write(path, path.relative_to(ROOT).as_posix())
    (ROOT / "LongCat_Challenge2.zip.sha256").write_text(
        f"{sha256_file(target)}  LongCat_Challenge2.zip\n", encoding="utf-8"
    )
    print(f"Packaged {len(files)} files into {target.name} ({target.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
