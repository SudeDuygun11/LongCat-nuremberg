# Final release checklist

Release identity:

- Team: **LongCat**
- Participant: **Sude Duygun**
- Contact: **sudeduygun11@gmail.com**
- Report: `LongCat_Challenge2.pdf`
- Prediction file: `predictions.csv`
- Reproduction archive: `LongCat_Challenge2.zip`

Verification:

- Run `PYTHONPATH=src python scripts/evaluate.py` for byte-exact prediction reproduction.
- Run `PYTHONPATH=src python scripts/validate_submission.py` for the schema, timing, causality, report, and archive checks.
- Run `PYTHONPATH=src python scripts/release_preflight.py` for the complete hash-bound release check.
- Independently confirm the official portal filename and cutoff.

The release scripts build and validate local artifacts; they do not transmit files.
