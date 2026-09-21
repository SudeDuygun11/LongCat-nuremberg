#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import zipfile

from pypdf import PdfReader

from longcat.io import ROOT, load_config
from validate_submission import main as validate_submission


PUBLIC_TEXT_FILES = [
    "README.md",
    "FINAL_CHECKLIST.md",
    "MODEL_CARD.md",
    "DATA_AND_LICENSES.md",
    "SUBMISSION_SCHEMA.md",
    "config/config.json",
    "report/report.html",
    "report/report_template.html.j2",
]
def verify_public_identity_and_wording() -> None:
    cfg = load_config()
    assert cfg["project"]["team_name"] == "LongCat"
    assert cfg["project"]["participant_name"] == "Sude Duygun"
    assert cfg["project"]["contact"] == "sudeduygun11@gmail.com"

    public_text = "\n".join(
        (ROOT / name).read_text(encoding="utf-8") for name in PUBLIC_TEXT_FILES
    )
    report_text = "\n".join(
        page.extract_text() or ""
        for page in PdfReader(ROOT / "LongCat_Challenge2.pdf").pages
    )
    for required in ["LongCat", "Sude Duygun", "sudeduygun11@gmail.com"]:
        assert required in public_text, required
        assert required in report_text, required


def verify_manifest_zip() -> None:
    expected = {}
    for line in (ROOT / "MANIFEST.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        expected[name] = digest
    with zipfile.ZipFile(ROOT / "LongCat_Challenge2.zip") as archive:
        assert archive.testzip() is None
        assert set(expected) == set(archive.namelist()) - {"MANIFEST.sha256"}
        for name, digest in expected.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name


def main() -> int:
    validate_submission(include_zip=True)
    verify_manifest_zip()
    verify_public_identity_and_wording()
    confirmation = load_config()["release_confirmation"]
    assert all(confirmation.values()), [key for key, value in confirmation.items() if not value]
    print("PASS: team, participant, and contact identity are complete")
    print("PASS: public identity and release wording verified")
    print("PASS: every release gate is closed and hash-bound")
    print("FINAL RELEASE PREFLIGHT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
