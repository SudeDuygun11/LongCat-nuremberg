import tempfile,unittest
from pathlib import Path
from longcat.io import ROOT,sha256_file
import sys
sys.path.insert(0,str(ROOT/'scripts'))
from predict import generate

class ReproductionTest(unittest.TestCase):
    def test_prediction_bytes_are_reproduced(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'predictions.csv';generate(ROOT/'data'/'frozen_issue_features.csv.gz',path)
            self.assertEqual(sha256_file(path),sha256_file(ROOT/'predictions.csv'))

if __name__=='__main__':unittest.main()
