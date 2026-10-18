from pathlib import Path
import queue
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '02_code'))
import stabilizer_gui as atm


class GUITests(unittest.TestCase):
    def gui(self):
        gui = atm.TemplateGUI.__new__(atm.TemplateGUI)
        gui.tpls, gui.hist, gui.frame_shape = [], [], (100, 100)
        gui.root, gui.tree, gui.canvas, gui.scale = Mock(), Mock(), Mock(), Mock()
        gui.messagebox = Mock()
        gui.pb = {}
        gui.btn_delete, gui.btn_undo, gui.btn_execute = Mock(), Mock(), Mock()
        gui.outdir = Path('unused')
        return gui

    def test_large_progress_queue_finishes_without_100ms_per_item(self):
        gui, notifications = self.gui(), queue.Queue()
        notifications.put(('setmax', 20000))
        for i in range(20000):
            notifications.put(('prog', i + 1))
        notifications.put(('done', None))
        while not notifications.empty():
            gui._poll(notifications)
        self.assertEqual(gui.pb, {'maximum': 20000, 'value': 20000})
        gui.messagebox.showinfo.assert_called_once()
        gui.root.destroy.assert_called_once()
        self.assertTrue(all(call.args[0] == 0 for call in gui.root.after.call_args_list))

    def test_error_after_progress_is_not_lost(self):
        gui, notifications = self.gui(), queue.Queue()
        for event in [('setmax', 50), ('prog', 40), ('setmax', 100), ('prog', 3),
                      ('err', RuntimeError('failure'))]:
            notifications.put(event)
        gui._poll(notifications)
        self.assertEqual(gui.pb, {'maximum': 100, 'value': 3})
        gui.messagebox.showerror.assert_called_once_with('Error', 'failure')
        gui.root.after.assert_not_called()

    def test_delete_add_undo_keeps_ids_unique(self):
        gui = self.gui()
        gui._show = Mock()
        gui.scale.get.return_value = 1
        for x in [20, 40]:
            gui._add_tpl(x, 20, 8, 8, 8, 8, 8)
        gui.tree.focus.return_value = '2'
        gui._del()
        gui._add_tpl(60, 20, 8, 8, 8, 8, 8)
        gui._undo()
        self.assertEqual([t['id'] for t in gui.tpls], [1, 3, 2])
        self.assertEqual(len({t['id'] for t in gui.tpls}), 3)

    def test_redraw_reuses_canvas_item(self):
        gui = self.gui()
        gui.cap, gui.tk, gui.zoom = Mock(), Mock(), 1
        gui.cap.read.return_value = (True, np.zeros((100, 100, 3), np.uint8))
        gui.canvas.create_image.return_value = 7
        photo = Mock()
        photo.width.return_value = photo.height.return_value = 100
        with patch.object(atm, '_pil_to_tk_photo', return_value=photo):
            for _ in range(20):
                gui._show(1)
        gui.canvas.create_image.assert_called_once()
        self.assertEqual(gui.canvas.itemconfigure.call_count, 19)


if __name__ == '__main__':
    unittest.main()
