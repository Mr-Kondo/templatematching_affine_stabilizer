from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '02_code'))
import affine_template_matching as atm


class FrameIndexTests(unittest.TestCase):
    def test_shuffled_legacy_csv_estimates_each_frame_once(self):
        rows = [[frame, x, y, x + frame, y - frame]
                for frame in [4, 2, 3] for x, y in [(0, 0), (30, 10), (10, 40)]]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'Matched.csv'
            pd.DataFrame(rows, columns=['Frame_No', 'x', 'y', 'mx', 'my']).to_csv(path, index=False)
            corrector = atm.AffineCorrector('unused', str(path), str(Path(folder) / 'output.mp4'))
            corrector.precompute()
            for frame in [4, 2, 3, 2]:
                np.testing.assert_allclose(corrector._M(frame), [[1, 0, -frame], [0, 1, frame]], atol=1e-6)
            self.assertEqual([st['frame'] for st in corrector.estimate_stats], [2, 3, 4])
            with self.assertRaisesRegex(ValueError, 'フレーム5'):
                corrector._M(5)

    def test_nonfinite_or_fractional_frames_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'Matched.csv'
            for frame, mx in [(float('nan'), 0), (2.5, 0), (1, 0), (2, float('inf'))]:
                pd.DataFrame([[frame, 0, 0, mx, 0]], columns=['Frame_No', 'x', 'y', 'mx', 'my']).to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    atm.AffineCorrector('unused', str(path))


if __name__ == '__main__':
    unittest.main()
