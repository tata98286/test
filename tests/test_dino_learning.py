import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

import app
import dino_learning


class ValidatingCursor:
    def __init__(self):
        self.lastrowid = 101
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, query, args=None):
        if args is not None:
            self.assert_placeholders(query, args)
        self._rows = []

    def executemany(self, query, rows):
        for args in rows:
            self.assert_placeholders(query, args)

    @staticmethod
    def assert_placeholders(query, args):
        if query.count('%s') != len(args):
            raise AssertionError(
                f'placeholder={query.count("%s")} args={len(args)}')

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class FakeConnection:
    def __init__(self, cursor=None):
        self.cursor_value = cursor or ValidatingCursor()

    def cursor(self):
        return self.cursor_value

    def close(self):
        pass

    def begin(self):
        pass

    def commit(self):
        pass

    def rollback(self):
        pass


class DinoLearningTest(unittest.TestCase):
    def test_event_insert_has_matching_arguments_and_passes_event_id_to_crop(self):
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        candidates = []
        for index in range(3):
            candidates.append({
                'frame_index': index,
                'frame': frame,
                'confidence': .8,
                'video_second': float(index),
                'dino': {'fire': .7, 'smoke': .2, 'lights': .1, 'clouds': .05,
                         'threshold': .15, 'crops': []},
                'boxes': [(1, 1, 30, 30)],
                'box_labels': ['fire'],
                'box_confidences': [.8],
            })
        job = {'token': 'job', 'original_name': 'sample.mp4', 'event_ids': [],
               'vlm_events': {}, 'evidence_names': [], 'pending_vlm': 0}
        saved = MagicMock()
        with patch.object(app, 'open_db_connection', return_value=FakeConnection()), \
             patch.object(app, 'perceptual_hash', return_value='hash'), \
             patch.object(app.cv2, 'imwrite', return_value=True), \
             patch.object(dino_learning, 'save_event_crops', saved), \
             patch.object(app, 'enqueue_vlm'):
            event_id = app.start_event_verification(job, candidates, (.5, .5))
        self.assertEqual(event_id, 101)
        self.assertEqual(saved.call_args.args[-1], 101)

    def test_crop_collection_keeps_event_label_and_yolo_confidence(self):
        cursor = ValidatingCursor()
        with tempfile.TemporaryDirectory() as temp:
            old_crop_dir = dino_learning.CROP_DIR
            old_factory = dino_learning._conn_factory
            dino_learning.CROP_DIR = Path(temp)
            dino_learning._conn_factory = lambda: FakeConnection(cursor)
            try:
                candidate = {
                    'frame': np.full((50, 50, 3), 100, dtype=np.uint8),
                    'video_second': 2.5,
                    'box_labels': ['smoke'],
                    'box_confidences': [.73],
                    'dino': {'crops': [{
                        'box': (5, 5, 40, 40), 'source_index': 0,
                        'scores': {'fire': .1, 'smoke': .8,
                                   'lights': .2, 'clouds': .3},
                    }]},
                }
                captured = []
                cursor.executemany = lambda query, rows: captured.extend(rows)
                dino_learning.save_event_crops('token', 1, [candidate], event_id=77)
            finally:
                dino_learning.CROP_DIR = old_crop_dir
                dino_learning._conn_factory = old_factory
        self.assertEqual(captured[0][0], 77)
        self.assertEqual(captured[0][4], 'smoke')
        self.assertEqual(captured[0][5], .73)

    def test_init_restores_latest_deployed_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            deployed = Path(temp) / 'deployed.pt'
            deployed.write_bytes(b'checkpoint')
            cursor = ValidatingCursor()
            cursor._rows = [{'checkpoint_path': str(deployed)}]

            def execute(query, args=None):
                cursor._rows = [{'checkpoint_path': str(deployed)}]
            cursor.execute = execute
            reload_hook = MagicMock()
            dino_learning.init(lambda: FakeConnection(cursor), Path(temp) / 'base.pt',
                               reload_hook, lambda fn: None)
            reload_hook.assert_called_once_with(deployed)


if __name__ == '__main__':
    unittest.main()
