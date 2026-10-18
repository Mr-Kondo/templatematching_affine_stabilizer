from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
import search_candidates as search


class CandidateChecks(unittest.TestCase):
    def test_local_and_pyramid_coordinate_origins(self):
        base = np.random.default_rng(42).integers(0, 255, (128, 128), dtype=np.uint8)
        template = dict(id=1, cx=64, cy=64, th=8, l=48, r=48, t=48, b=48,
                        patch=base[56:72, 56:72])
        translated = cv2.warpAffine(base, np.float32([[1, 0, 3], [0, 1, 5]]), (128,128))
        for method in [search.predicted_match, search.pyramid_match]:
            position = method(translated, template, (67,69))[:2]
            np.testing.assert_allclose(position, [67,69], atol=.15)

    def test_lk_detects_textureless_point_shortage(self):
        from lk_candidate import lk_track
        frames, templates, motion = search.scene('low_texture',42,count=3)
        positions,counters=lk_track(frames,templates)
        self.assertEqual(counters['lk_point_shortage_frames'],2)
        self.assertEqual(counters['lk_rejected_points'],10)
        self.assertEqual(len(counters['fallback_indices']),10)
        self.assertTrue(np.isfinite(positions).all())

    def test_phase_and_ecc_map_in_the_documented_direction(self):
        from registration_candidates import phase_shift, refine_ecc
        from synth import build_base_scene
        base=cv2.cvtColor(build_base_scene(640,360,42),cv2.COLOR_BGR2GRAY)
        current=cv2.warpAffine(base,np.float32([[1,0,4],[0,1,6]]),(640,360))
        displacement,response=phase_shift(base,current)
        np.testing.assert_allclose(displacement,[4,6],atol=.3)
        self.assertGreater(response,.2)
        initial=np.float32([[1,0,-3.8],[0,1,-6.2]])
        mask=np.full(base.shape,255,np.uint8)
        refined,ok=refine_ecc(base,current,initial,mask)
        self.assertTrue(ok)
        np.testing.assert_allclose(refined,[[1,0,-4],[0,1,-6]],atol=.12)
        with patch.object(cv2,'findTransformECC',side_effect=cv2.error('nonconvergence')):
            retained,ok=refine_ecc(base,current,initial,mask)
        self.assertFalse(ok)
        np.testing.assert_array_equal(retained,initial)

    def test_truth_positions_give_zero_tracking_and_background_error(self):
        frames, templates, motion = search.scene('normal', 42, count=5)
        centres = np.array([[t['cx'],t['cy']] for t in templates],float)
        positions = np.array([(m[:,:2]@centres.T+m[:,2:3]).T for m in motion[1:]])
        report = search.evaluate(positions, templates, motion,
            dict(track_s=0, fallback_indices=[], fallback_s=0))
        self.assertEqual(report['mean_error_px'],0)
        self.assertLess(report['background_mean_px'],.001)
        self.assertEqual(report['estimation_failures'],0)


if __name__ == '__main__':
    unittest.main()
