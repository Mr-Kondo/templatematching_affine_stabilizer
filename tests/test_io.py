from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '02_code'))
import affine_template_matching as atm


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.video = self.folder / 'input.mp4'
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*'mp4v'), 30, (64, 64))
        self.assertTrue(writer.isOpened())
        frame = np.random.default_rng(42).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        for _ in range(3):
            writer.write(frame)
        writer.release()
        self.csv = self.folder / 'Input.csv'
        pd.DataFrame([[1, 16, 16, 8, 8, 8, 8, 8], [2, 48, 48, 8, 8, 8, 8, 8],
                      [3, 16, 48, 8, 8, 8, 8, 8]], columns=atm.TEMPLATE_FIELDS).to_csv(self.csv, index=False)

    def test_complete_pipeline_publishes_four_outputs(self):
        outdir = self.folder / 'new'
        matcher, corrector = atm.run_pipeline(str(self.video), str(self.csv), outdir)
        self.assertFalse(matcher.cap.isOpened())
        self.assertEqual(set(p.name for p in outdir.iterdir()),
                         {'Matched.csv', 'Matched_diagnostics.csv', 'Affined.mp4', 'Affined_diagnostics.csv'})
        atm.verify_video(outdir / 'Affined.mp4', 3)

    def test_existing_output_is_never_changed(self):
        output = self.folder / 'Matched.csv'
        output.write_bytes(b'existing')
        before = {p.name: p.read_bytes() for p in self.folder.iterdir()}
        with self.assertRaises(FileExistsError):
            atm.run_pipeline(str(self.video), str(self.csv), self.folder)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.folder.iterdir()})

    def test_partial_failure_removes_only_owned_outputs(self):
        unrelated = self.folder / 'temp'
        unrelated.mkdir()
        (unrelated / 'keep.txt').write_text('keep')
        with patch.object(atm.AffineCorrector, '_M', side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                atm.run_pipeline(str(self.video), str(self.csv), self.folder)
        self.assertEqual((unrelated / 'keep.txt').read_text(), 'keep')
        self.assertFalse((self.folder / 'Matched.csv').exists())
        self.assertFalse(list(self.folder.glob('.stabilizer-*')))

    def test_publish_race_preserves_foreign_file(self):
        first, second = self.folder / 'first.csv', self.folder / 'second.csv'
        with self.assertRaises(FileExistsError):
            with atm.output_files([first, second]) as staged:
                for p in staged:
                    p.write_text('own')
                second.write_text('foreign')
        self.assertFalse(first.exists())
        self.assertEqual(second.read_text(), 'foreign')

    def test_match_exception_releases_capture(self):
        matcher = atm.TemplateMatcher(str(self.video), str(self.csv))
        with patch.object(matcher, '_match', side_effect=RuntimeError('match failed')):
            with self.assertRaises(RuntimeError):
                matcher.run(str(self.folder / 'Matched.csv'))
        self.assertFalse(matcher.cap.isOpened())

    def test_writer_failure_and_warp_failure_release_resources(self):
        matched = self.folder / 'matches.csv'
        pd.DataFrame(columns=['Frame_No', 'x', 'y', 'mx', 'my']).to_csv(matched, index=False)
        corrector = atm.AffineCorrector(str(self.video), str(matched), str(self.folder / 'out.mp4'))
        for opened in [False, True]:
            writer = Mock()
            writer.isOpened.return_value = opened
            with patch.object(atm.cv2, 'VideoWriter', return_value=writer), \
                 patch.object(corrector, '_warp_frames', side_effect=RuntimeError('warp')):
                with self.assertRaises(RuntimeError):
                    corrector.correct()
            writer.release.assert_called_once()
            self.assertFalse(corrector.out.exists())
            self.assertFalse(list(self.folder.glob('.stabilizer-*')))

    def test_input_output_collision_and_bad_video(self):
        with self.assertRaises(ValueError):
            atm.run_pipeline(str(self.folder / 'Affined.mp4'), str(self.csv), self.folder)
        self.video.write_bytes(b'bad video')
        with self.assertRaises(ValueError):
            atm.run_pipeline(str(self.video), str(self.csv), self.folder / 'out')
        self.assertFalse((self.folder / 'out').exists())

    def test_silent_writer_failure_is_detected(self):
        with self.assertRaises((RuntimeError, ValueError)):
            atm.verify_video(self.folder / 'absent.mp4', 3)
        with self.assertRaises(RuntimeError):
            atm.verify_video(self.video, 4)


if __name__ == '__main__':
    unittest.main()
