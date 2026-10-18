import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '02_code'))
import affine_template_matching as atm


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(id=1, cx=20, cy=20, tpl=8, l=8, r=8, t=8, b=8),
                     dict(id=2, cx=40, cy=40, tpl=8, l=8, r=8, t=8, b=8)]

    def write_csv(self, path, rows=None):
        with open(path, 'w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(atm.TEMPLATE_FIELDS)
            for row in rows if rows is not None else self.rows:
                writer.writerow([row[key] for key in atm.TEMPLATE_FIELDS.values()])

    def test_existing_input(self):
        cap = cv2.VideoCapture(str(ROOT / '01_input/input.mp4'))
        ok, first = cap.read()
        cap.release()
        self.assertTrue(ok)
        self.assertEqual(len(atm.load_templates(ROOT / '01_input/Input.csv', first.shape)), 5)

    def test_invalid_fields_identify_row_and_column(self):
        for key, invalid in [('id', 0), ('tpl', 0), ('tpl', -2), ('tpl', 7),
                             ('tpl', 102), ('cx', -1), ('cx', 100), ('cy', 100),
                             ('l', 0), ('l', -1), ('r', 3), ('b', 101),
                             ('cy', 2.5), ('id', 'nan'), ('cx', 'inf'),
                             ('cx', ''), ('cx', 'text'), ('id', '1e99999')]:
            with self.subTest(key=key, invalid=invalid):
                self.rows[0][key] = invalid
                column = next(c for c, k in atm.TEMPLATE_FIELDS.items() if k == key)
                with self.assertRaisesRegex(ValueError, f'行2 {column}'):
                    atm.validate_templates(self.rows, (100, 100))
                self.setUp()

    def test_empty_one_duplicate_ids_and_centres(self):
        for rows in [[], self.rows[:1], [self.rows[0], self.rows[0]],
                     [self.rows[0], dict(self.rows[0], id=2)]]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                atm.validate_templates(rows, (100, 100))

    def test_two_points_are_accepted_but_fail_quality(self):
        templates = atm.validate_templates(self.rows, (100, 100))
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'matched.csv'
            pd.DataFrame([[2, t['id'], t['cx'], t['cy'], t['cx'], t['cy']]
                          for t in templates], columns=['Frame_No', 'Template_No',
                          'x', 'y', 'mx', 'my']).to_csv(path, index=False)
            corrector = atm.AffineCorrector('unused', str(path))
            np.testing.assert_allclose(corrector._M(2), [[1, 0, 0], [0, 1, 0]], atol=1e-6)
            self.assertTrue(corrector.estimate_stats[0]['ok'])
            self.assertFalse(corrector.estimate_stats[0]['quality_ok'])

    def test_csv_schema_and_ragged_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.csv'
            for body in ['', 'No.,x座標\n1,2\n',
                         ','.join(atm.TEMPLATE_FIELDS) + ',No.\n',
                         ','.join(atm.TEMPLATE_FIELDS) + '\n1,2\n',
                         ','.join(atm.TEMPLATE_FIELDS) + '\n1,2,3,8,8,8,8,8,9\n']:
                with self.subTest(body=body):
                    path.write_text(body)
                    with self.assertRaises(ValueError):
                        atm.load_templates(path, (100, 100))

    def test_integral_strings_and_bom(self):
        self.rows[0]['cx'] = '20.0'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.csv'
            self.write_csv(path)
            path.write_text('\ufeff' + path.read_text())
            self.assertEqual(atm.load_templates(path, (100, 100))[0]['cx'], 20)

    def test_edges_padding_and_research_do_not_shrink(self):
        frame = np.random.default_rng(42).integers(0, 255, (100, 100), dtype=np.uint8)
        for cx, cy in [(0, 0), (99, 99), (0, 99), (99, 0), (50, 50)]:
            with self.subTest(cx=cx, cy=cy):
                row = dict(self.rows[0], cx=cx, cy=cy)
                atm.validate_template(row, frame.shape, 2)
                patch = atm.crop(frame, cx, cy, 4)
                self.assertEqual(patch.shape, (8, 8))
                matcher = atm.TemplateMatcher.__new__(atm.TemplateMatcher)
                template = dict(id=1, cx=cx, cy=cy, th=4, l=4, r=4, t=4, b=4, patch=patch)
                mx, my, diagnostics = matcher._match(frame, template)
                self.assertTrue(np.isfinite([mx, my]).all())
                self.assertIn('low_conf', diagnostics)
        for cx, cy in [(-20, 10), (120, 10), (10, -20), (10, 120)]:
            with self.assertRaisesRegex(ValueError, '完全に画像外'):
                atm.crop(frame, cx, cy, 4)

    def test_gui_invalid_add_does_not_mutate(self):
        gui = atm.TemplateGUI.__new__(atm.TemplateGUI)
        gui.tpls, gui.hist, gui.frame_shape, gui.tree = [], [], (100, 100), Mock()
        with self.assertRaises(ValueError):
            gui._add_tpl(20, 20, 7, 8, 8, 8, 8)
        self.assertEqual(gui.tpls, [])
        gui.tree.insert.assert_not_called()

    def test_gui_edit_cancel_and_invalid_are_atomic(self):
        for answers in [[25, None], [25, 25, 7, 8, 8, 8, 8]]:
            gui = atm.TemplateGUI.__new__(atm.TemplateGUI)
            gui.tpls = [self.rows[0].copy()]
            gui.frame_shape, gui.root = (100, 100), None
            gui.tree, gui.simpledialog, gui.messagebox = Mock(), Mock(), Mock()
            gui.tree.focus.return_value = '1'
            gui.simpledialog.askinteger.side_effect = answers
            gui._edit(None)
            self.assertEqual(gui.tpls, self.rows[:1])
            gui.tree.item.assert_not_called()

    def test_invalid_cli_does_not_create_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.csv'
            self.rows[0]['tpl'] = 7
            self.write_csv(path)
            run = subprocess.run([sys.executable, str(ROOT / '02_code/affine_template_matching.py'),
                '--nogui', '--video', str(ROOT / '01_input/input.mp4'), '--csv', str(path)],
                cwd=tmp, capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            self.assertIn('行2 テンプレート', run.stderr)
            self.assertNotIn('Traceback', run.stderr)
            self.assertFalse((Path(tmp) / '03_output').exists())


if __name__ == '__main__':
    unittest.main()
