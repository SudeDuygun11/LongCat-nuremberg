import unittest
import pandas as pd
from longcat.io import ROOT

class CausalityTest(unittest.TestCase):
    def test_every_frozen_source_time_is_available_at_issue(self):
        frame=pd.read_csv(ROOT/'data'/'frozen_issue_features.csv.gz')
        source=pd.to_datetime(frame.max_source_timestamp,utc=True);issue=pd.to_datetime(frame.issue_time,utc=True)
        self.assertTrue((source<=issue).all())
        self.assertNotIn('observed_customers_out',frame.columns)

if __name__=='__main__':unittest.main()
