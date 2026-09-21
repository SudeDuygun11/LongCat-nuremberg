import sys,unittest
from longcat.io import ROOT
sys.path.insert(0,str(ROOT/'scripts'))
from validate_submission import validate_predictions

class PredictionTest(unittest.TestCase):
    def test_exact_release_contract(self):self.assertEqual(len(validate_predictions()),2)

if __name__=='__main__':unittest.main()
