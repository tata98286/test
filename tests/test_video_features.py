import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import cv2
import numpy as np
import app as module
from video_sources import youtube_url, download_youtube
from vlm_display import overlay, STATES


class VideoFeaturesTest(unittest.TestCase):
    def test_fire_or_smoke_can_enter_event_pipeline(self):
        self.assertTrue(module.has_fire_or_smoke({'fire'}))
        self.assertTrue(module.has_fire_or_smoke({'smoke'}))
        self.assertTrue(module.has_fire_or_smoke({'fire', 'smoke'}))
        self.assertFalse(module.has_fire_or_smoke(set()))
        self.assertFalse(module.has_fire_or_smoke({'light', 'cloud'}))

    def test_image_is_converted_to_five_frame_test_video(self):
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / 'sample.png'
            video_path = Path(temp) / 'sample.mp4'
            cv2.imwrite(str(image_path), np.full((120, 160, 3), 127, dtype=np.uint8))
            module.prepare_image_test_video(image_path, video_path)
            capture = cv2.VideoCapture(str(video_path))
            self.assertTrue(capture.isOpened())
            self.assertEqual(round(capture.get(cv2.CAP_PROP_FPS)), 5)
            self.assertEqual(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 5)
            ok, frame = capture.read()
            capture.release()
            self.assertTrue(ok)
            self.assertEqual(frame.shape[:2], (120, 160))

    def test_vlm_parser_rejects_prompt_echo_and_accepts_korean_evidence(self):
        malformed = 'YES, One short Korean sentence describing the visible evidence. For UNCERTAIN, explain what specifically needs human visual review.'
        self.assertIsNone(module.parse_vlm_response(malformed))
        self.assertEqual(module.extract_vlm_verdict(malformed), 'YES')
        self.assertEqual(module.extract_vlm_verdict('YES, NO, or UNCERTAIN\nYES'), 'YES')
        self.assertEqual(
            module.parse_vlm_response('YES\n화면 중앙에 주황색 불꽃과 짙은 연기가 보입니다.'),
            ('confirmed', 'YES\n화면 중앙에 주황색 불꽃과 짙은 연기가 보입니다.'),
        )
        self.assertIsNone(module.parse_vlm_response('YES\nAt least one image clearly shows flames or smoke consistent with combustion.'))
        self.assertIsNone(module.parse_vlm_response('UNCERTAIN\n화재인지 확실하지 않습니다.'))
        self.assertEqual(
            module.parse_vlm_response('UNCERTAIN\n대상이 너무 작고 흐려서 불꽃인지 조명인지 구분하기 어렵습니다.'),
            ('uncertain', 'UNCERTAIN\n대상이 너무 작고 흐려서 불꽃인지 조명인지 구분하기 어렵습니다.'),
        )

    def test_url_validation(self):
        expected = 'https://www.youtube.com/watch?v=BaW_jenozKc'
        for url in ['https://youtu.be/BaW_jenozKc?t=2', expected, 'https://youtube.com/shorts/BaW_jenozKc']:
            self.assertEqual(youtube_url(url), expected)
        for url in ['http://127.0.0.1/a', 'file:///etc/passwd', 'https://youtube.com.evil.org/watch?v=BaW_jenozKc', 'https://youtube.com/playlist?list=x']:
            with self.assertRaises(ValueError):
                youtube_url(url)

    def test_all_verdict_overlays_and_flash(self):
        frame = np.zeros((240, 640, 3), dtype=np.uint8)
        for status in STATES:
            job = {'vlm_events': {1: {'event_id': 1, 'status': status, 'finished_at': time.monotonic()}}}
            rendered = overlay(frame.copy(), job)
            self.assertGreater(int(rendered.sum()), 0)
        job = {'vlm_events': {1: {'event_id': 1, 'status': 'confirmed', 'finished_at': 100}}}
        with patch('vlm_display.time.monotonic', return_value=100.1):
            self.assertEqual(overlay(frame.copy(), job)[100, 0].tolist(), [0, 0, 255])
        with patch('vlm_display.time.monotonic', return_value=102):
            self.assertEqual(overlay(frame.copy(), job)[100, 0].tolist(), [0, 0, 0])

    def test_download_error_and_cancellation(self):
        with tempfile.TemporaryDirectory() as temp:
            for stopped in (True, False):
                job = {'token': 'abc', 'source_url': 'https://www.youtube.com/watch?v=BaW_jenozKc', 'stop_requested': stopped}
                partial = Path(temp) / 'abc_source.mp4.part'
                partial.write_bytes(b'partial')
                with patch('yt_dlp.YoutubeDL', side_effect=RuntimeError('unavailable')):
                    download_youtube(job, Path(temp))
                self.assertEqual(job['status'], 'cancelled' if stopped else 'error')
                self.assertFalse(partial.exists())

    def test_original_alarm_colors_text_geometry_and_blink(self):
        for status, label, color in [('confirmed', 'LLM: FIRE DETECTED!', (0, 0, 255)),
                                     ('false_alarm', 'LLM: FALSE ALARM', (0, 255, 0))]:
            job = {'vlm_events': {1: {'event_id': 1, 'status': status, 'finished_at': 100}}}
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            with patch('vlm_display.time.monotonic', return_value=100.1):
                actual = overlay(frame.copy(), job)
            expected = frame.copy()
            cv2.rectangle(expected, (0, 0), (1280, 720), color, 20)
            width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 5)[0][0]
            cv2.putText(expected, label, ((1280-width)//2, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)
            # Exclude the web metadata header; alarm pixels must match the notebook.
            np.testing.assert_array_equal(actual[43:], expected[43:])
            for now in (100.3, 101.1):
                with patch('vlm_display.time.monotonic', return_value=now):
                    hidden = overlay(frame.copy(), job)
                self.assertEqual(int(hidden[43:].sum()), 0)

    def test_vlm_database_failure_does_not_hang(self):
        job = {'pending_vlm': 1, 'vlm_events': {7: {'event_id': 7, 'status': 'queued'}}}
        with patch.object(module, 'classify_with_vlm', return_value=('confirmed', 'YES\n불꽃 확인')), patch.object(module, 'open_db_connection', side_effect=RuntimeError('DB test error')):
            module.run_vlm_for_event(job, 7, [])
        self.assertEqual(job['pending_vlm'], 0)
        self.assertEqual(job['vlm_events'][7]['status'], 'error')

    def test_video_pipeline_saves_overlay_h264_and_partial_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'source.mp4'
            target = Path(temp) / 'result.mp4'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 10, (320, 240))
            for _ in range(6):
                writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
            writer.release()
            fake_result = MagicMock(boxes=None, orig_img=np.zeros((240, 320, 3), dtype=np.uint8))
            model = MagicMock(names={0:'fire', 1:'smoke'})
            model.predict.return_value = [fake_result]
            job = {'source_path':source, 'result_path':target, 'original_name':'test', 'event_ids':[],
                   'pending_vlm':0, 'vlm_events':{1:{'event_id':1,'status':'confirmed','finished_at':time.monotonic()}},
                   'latest_vlm_result':'confirmed'}
            with patch.object(module, 'get_model', return_value=model), patch.object(module, 'start_metric_run', return_value=0), patch.object(module, 'finish_metric_run'):
                stream = module.generate_inspection_stream(job)
                self.assertIn(b'Content-Type: image/jpeg', next(stream))
                job['stop_requested'] = True
                list(stream)
            self.assertEqual(job['status'], 'complete')
            self.assertTrue(job['report']['stopped_early'])
            self.assertEqual(job['report']['processed_frames'], 1)
            capture = cv2.VideoCapture(str(target))
            ok, image = capture.read()
            self.assertTrue(ok)
            self.assertGreater(int(image[:42].sum()), 0)
            self.assertEqual(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 11)
            capture.release()


if __name__ == '__main__':
    unittest.main()
