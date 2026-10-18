from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '02_code'))
from stabilizer_core import AffineCorrector


class DegenerateEstimationTests(unittest.TestCase):
    def test_collapsed_two_points_are_estimation_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'Matched.csv'
            pd.DataFrame([[2,0,0,10,10],[2,30,30,10,10]],
                         columns=['Frame_No','x','y','mx','my']).to_csv(path,index=False)
            corrector=AffineCorrector('unused',str(path))
            with self.assertRaises(RuntimeError):
                corrector._M(2)
            self.assertFalse(corrector.estimate_stats[0]['ok'])
            self.assertIsNone(corrector.estimate_stats[0]['M'])
            self.assertEqual(corrector._mcache,{})


if __name__=='__main__':
    unittest.main()
