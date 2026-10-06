import json
import hashlib
import os
import re
import secrets
import threading
import time
import uuid
import queue
import subprocess
from video_sources import youtube_url, download_youtube
from vlm_display import overlay, feedback, STATES
from datetime import timedelta
from functools import wraps
from pathlib import Path

import cv2
import pymysql
from dotenv import load_dotenv
from flask import Flask, Response, abort, flash, g, jsonify, redirect, render_template, request, send_from_directory, session, url_for
from ultralytics import YOLO
from werkzeug.security import check_password_hash, generate_password_hash

load_dotenv(Path(__file__).with_name('.env'))
app = Flask(__name__)
app.logger.setLevel('INFO')
app.config.update(
    SECRET_KEY=os.environ['SECRET_KEY'],
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=os.getenv('COOKIE_SECURE', 'false').lower() == 'true',
    PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
    MAX_CONTENT_LENGTH=500 * 1024 * 1024,
)
DUMMY_HASH = generate_password_hash(secrets.token_urlsafe(32), method='scrypt')
MODEL_PATH = Path(os.getenv('MODEL_PATH', r'C:\its\model\best.2026.09.17.pt'))
VLM_MODEL_PATH = Path(os.getenv('VLM_MODEL_PATH', r'C:\its\model\Qwen2-VL-2B-Instruct'))
RESULT_DIR = Path(app.instance_path) / 'inspection_results'
RESULT_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
IMAGE_TEST_FPS = 5.0
IMAGE_TEST_FRAMES = 5
MAX_IMAGE_PIXELS = 40_000_000
_model = None
_model_lock = threading.Lock()
_vlm_model = None
_vlm_processor = None
_vlm_lock = threading.Lock()
_dino_model = None
_dino_heads = None
_dino_lock = threading.Lock()
_jobs_lock = threading.Lock()
inspection_jobs = {}
_vlm_queue = queue.Queue(maxsize=32)
_vlm_worker_started = False
VLM_PROMPT = Path(__file__).with_name('vlm_prompt.txt').read_text(encoding='utf-8')
VLM_PROMPT_VERSION = hashlib.sha256(VLM_PROMPT.encode()).hexdigest()[:16]
EVENT_START_WINDOW_SECONDS = 3.0
EVENT_END_QUIET_SECONDS = 30.0
EVENT_LOCATION_DISTANCE = 0.20
DINO_MODEL_PATH = Path(os.getenv('DINO_MODEL_PATH', r'C:\its\model\full_crop_gate_multilabel.pt'))
HUMAN_VERDICTS = {'FIRE', 'FALSE_ALARM', 'UNCERTAIN'}
YOLO_THRESHOLDS = {40, 70, 90}


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if not session.get('user_id'):
            return redirect(url_for('login'))
        return view(*args, **kwargs)
    return wrapped_view


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f'모델 파일을 찾을 수 없습니다: {MODEL_PATH}')
        _model = YOLO(str(MODEL_PATH))
    return _model


def alert_class_ids(model):
    return [class_id for class_id, name in model.names.items() if name in {'fire', 'smoke'}]


def matches_yolo_condition(detected_labels, detection_mode='or'):
    if detection_mode == 'and':
        return {'fire', 'smoke'}.issubset(detected_labels)
    return bool({'fire', 'smoke'}.intersection(detected_labels))


def parse_yolo_threshold(form):
    try:
        threshold = int(form.get('yolo_threshold', '40'))
    except (TypeError, ValueError):
        raise ValueError('YOLO 신뢰도는 40%, 70%, 90% 중에서 선택해 주세요.')
    if threshold not in YOLO_THRESHOLDS:
        raise ValueError('YOLO 신뢰도는 40%, 70%, 90% 중에서 선택해 주세요.')
    return threshold


def parse_human_review(form):
    verdict = form.get('human_verdict', '').strip().upper()
    notes = form.get('human_notes', '').strip()
    if verdict not in HUMAN_VERDICTS:
        raise ValueError('사람 최종 판정을 선택해 주세요.')
    if len(notes) > 500:
        raise ValueError('사람 의견은 500자 이하로 입력해 주세요.')
    return verdict, notes or None


def get_dino():
    global _dino_model, _dino_heads
    if _dino_model is None:
        import torch
        from transformers.models.dinov3_vit import DINOv3ViTConfig, DINOv3ViTModel
        checkpoint = torch.load(DINO_MODEL_PATH, map_location='cpu', weights_only=False)
        config = DINOv3ViTConfig(
            hidden_size=1024, intermediate_size=4096, num_hidden_layers=24,
            num_attention_heads=16, num_register_tokens=4, image_size=518,
            patch_size=16, use_gated_mlp=False,
        )
        model = DINOv3ViTModel(config)
        backbone = {
            (key[6:] if key.startswith('model.') else key): value
            for key, value in checkpoint['backbone']['state_dict'].items()
        }
        model.load_state_dict(backbone, strict=True)
        spec = checkpoint['head']['spec']
        heads = torch.nn.ModuleDict({
            'gate': torch.nn.Sequential(torch.nn.LayerNorm(spec['dim']), torch.nn.Linear(spec['dim'], 2)),
            'aux': torch.nn.Sequential(torch.nn.LayerNorm(spec['dim']), torch.nn.Linear(spec['dim'], spec['aux'])),
        })
        heads.load_state_dict(checkpoint['head']['state_dict'], strict=True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # DINOv3의 RoPE 계산은 이 체크포인트에서 FP16 오버플로가 발생할 수 있다.
        # RTX 30 계열은 BF16을 지원하며 메모리 사용량은 FP16과 동일하다.
        dtype = torch.bfloat16 if device == 'cuda' else torch.float32
        _dino_model = model.to(device=device, dtype=dtype).eval()
        _dino_heads = heads.to(device=device, dtype=dtype).eval()
        _dino_heads.threshold = float(checkpoint['deploy']['threshold'])
        _dino_heads.classes = checkpoint['deploy']['classes']
        _dino_heads.pad = float(checkpoint['deploy']['pad'])
    return _dino_model, _dino_heads


def classify_yolo_crops_with_dino(frame, boxes):
    import torch
    model, heads = get_dino()
    height, width = frame.shape[:2]
    crops = []
    for x1, y1, x2, y2 in boxes:
        pad_x, pad_y = (x2 - x1) * heads.pad, (y2 - y1) * heads.pad
        left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
        right, bottom = min(width, int(x2 + pad_x)), min(height, int(y2 + pad_y))
        if right > left and bottom > top:
            crop = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2RGB)
            crop = cv2.resize(crop, (518, 518), interpolation=cv2.INTER_AREA)
            crops.append(crop)
    if not crops:
        return {'fire': 0.0, 'smoke': 0.0, 'lights': 0.0, 'clouds': 0.0, 'passed': False, 'threshold': heads.threshold}
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    pixels = torch.from_numpy(__import__('numpy').stack(crops)).to(device=device, dtype=dtype).permute(0, 3, 1, 2) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype)[None, :, None, None]
    with torch.inference_mode():
        cls = model(pixel_values=(pixels - mean) / std).last_hidden_state[:, 0]
        scores = torch.sigmoid(heads['aux'](cls)).amax(dim=0).float().cpu().tolist()
    values = dict(zip(heads.classes, scores))
    values.update(passed=max(values['fire'], values['smoke']) >= heads.threshold, threshold=heads.threshold)
    return values


def plot_alerts(result):
    frame = result.orig_img.copy()
    if result.boxes is None:
        return frame
    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        label = result.names[class_id]
        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
        color = (0, 0, 255) if label == 'fire' else (0, 165, 255)
        caption = f'{label.upper()} {confidence * 100:.0f}%'
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        (text_width, text_height), baseline = cv2.getTextSize(
            caption, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
        )
        text_top = max(0, y1 - text_height - baseline - 8)
        cv2.rectangle(frame, (x1, text_top), (x1 + text_width + 10, y1), color, -1)
        cv2.putText(
            frame, caption, (x1 + 5, y1 - baseline - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
        )
    return frame


def open_db_connection():
    return pymysql.connect(
        host=os.getenv('DB_HOST', '127.0.0.1'),
        port=int(os.getenv('DB_PORT', '3306')),
        user=os.environ['DB_USER'], password=os.environ['DB_PASSWORD'],
        database=os.getenv('DB_NAME', 'its'), charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor, autocommit=True,
        connect_timeout=5,
    )


def start_metric_run(job):
    connection = open_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                'INSERT INTO inspection_run_metric (job_token, source_filename) VALUES (%s, %s)',
                (job['token'], job['original_name']),
            )
            return cursor.lastrowid
    finally:
        connection.close()


def finish_metric_run(run_id, job, buckets, processed_frames, fps, elapsed_seconds):
    if not run_id:
        return
    status = 'ERROR' if job.get('status') in {'error', 'cancelled'} else ('STOPPED' if job.get('stop_requested') else 'COMPLETE')
    yolo_count = sum(bucket['yolo'] for bucket in buckets.values())
    pass_count = sum(bucket['passed'] for bucket in buckets.values())
    block_count = sum(bucket['blocked'] for bucket in buckets.values())
    connection = open_db_connection()
    try:
        with connection.cursor() as cursor:
            if buckets:
                cursor.executemany(
                    '''INSERT INTO inspection_metric_second
                       (run_id, video_second, yolo_candidate_count, dino_pass_count, dino_block_count,
                        max_yolo_score, max_dino_fire_score, max_dino_smoke_score,
                        max_dino_light_score, max_dino_cloud_score)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)''',
                    [(run_id, second, value['yolo'], value['passed'], value['blocked'],
                      value['yolo_score'], value['fire'], value['smoke'], value['lights'], value['clouds'])
                     for second, value in sorted(buckets.items())],
                )
            cursor.execute(
                '''UPDATE inspection_run_metric SET completed_at=CURRENT_TIMESTAMP, status=%s,
                   video_duration_seconds=%s, processing_seconds=%s, processed_frames=%s,
                   yolo_candidate_count=%s, dino_pass_count=%s, dino_block_count=%s, event_count=%s
                   WHERE run_id=%s''',
                (status, round(processed_frames / fps, 3), round(elapsed_seconds, 3), processed_frames,
                 yolo_count, pass_count, block_count, len(job.get('event_ids', [])), run_id),
            )
    finally:
        connection.close()


def perceptual_hash(frame):
    gray = cv2.cvtColor(cv2.resize(frame, (32, 32)), cv2.COLOR_BGR2GRAY)
    low = cv2.dct(gray.astype('float32'))[:8, :8]
    bits = low > float(low[1:, 1:].mean())
    return f'{sum(int(bit) << index for index, bit in enumerate(bits.flatten())):016x}'


def hashes_are_similar(current, previous):
    matches = 0
    for value in current:
        if any((int(value, 16) ^ int(old, 16)).bit_count() <= 9 for old in previous):
            matches += 1
    return matches >= 2


def get_vlm():
    global _vlm_model, _vlm_processor
    if _vlm_model is None:
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        _vlm_processor = AutoProcessor.from_pretrained(
            str(VLM_MODEL_PATH), local_files_only=True, use_fast=False,
        )
        _vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
            str(VLM_MODEL_PATH), dtype=torch.float16,
            device_map='cuda', local_files_only=True,
        )
        _vlm_model.eval()
    return _vlm_model, _vlm_processor


VLM_VERDICTS = {'YES': 'confirmed', 'NO': 'false_alarm', 'UNCERTAIN': 'uncertain'}
VLM_BOILERPLATE = (
    'one short korean sentence', 'at least one image clearly shows',
    'the relevant scene is sufficiently visible', 'use only when a specific',
    'yes, no, or uncertain', 'output exactly two lines', 'line 1:', 'line 2:',
    '판단 근거가 누락', '사람의 육안 확인이 필요합니다',
)


def extract_vlm_verdict(answer):
    for line in (value.strip() for value in (answer or '').splitlines() if value.strip()):
        upper = line.upper()
        match = re.match(r'^(?:VERDICT\s*[:：]\s*)?(YES|NO|UNCERTAIN)\b', upper)
        if match and not re.match(r'^YES\s*[,/]\s*NO\b', upper):
            return match.group(1)
    return None


def extract_korean_evidence(answer):
    candidates = []
    for line in (value.strip() for value in (answer or '').splitlines() if value.strip()):
        cleaned = re.sub(r'^(?:YES|NO|UNCERTAIN)\s*[,.:：-]?\s*', '', line, flags=re.I).strip()
        lowered = cleaned.lower()
        if not cleaned or any(fragment in lowered for fragment in VLM_BOILERPLATE):
            continue
        if len(re.findall(r'[가-힣]', cleaned)) >= 4:
            candidates.append(cleaned)
    return candidates[0][:430] if candidates else None


def parse_vlm_response(answer):
    verdict = extract_vlm_verdict(answer)
    evidence = extract_korean_evidence(answer)
    if verdict == 'UNCERTAIN' and evidence and not any(
        word in evidence for word in ('흐리', '어두', '가림', '가려', '작아', '작게', '식별', '구분', '가려져', '해상도', '충돌', '초점')
    ):
        evidence = None
    if verdict and evidence:
        return VLM_VERDICTS[verdict], f'{verdict}\n{evidence}'
    return None


def classify_with_vlm(frames, details=None):
    import torch
    from PIL import Image
    model, processor = get_vlm()
    center_frame = frames[len(frames) // 2]
    images = [Image.fromarray(cv2.cvtColor(cv2.resize(center_frame, (480, 480)), cv2.COLOR_BGR2RGB))]
    def generate_answer(prompt, use_images=True):
        messages = [{'role': 'user', 'content': [
            *[{'type': 'image'} for _ in (images if use_images else [])], {'type': 'text', 'text': prompt},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images if use_images else None, padding=True, return_tensors='pt')
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=50 if prompt == VLM_PROMPT else 160)
        generated = [output[len(source):] for source, output in zip(inputs['input_ids'], output_ids)]
        return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    from independent_description import judge_with_description
    verdict, answer, bilingual = judge_with_description(generate_answer, VLM_PROMPT)
    if details is not None:
        details.update(bilingual)
    return verdict, answer[:500]


def run_vlm_for_event(job, event_id, frames):
    started = time.monotonic()
    bilingual = {}
    record = job['vlm_events'][event_id]
    record.update(status='analyzing')
    app.logger.info('VLM start event=%s frames=%s', event_id, len(frames))
    try:
        with _vlm_lock:
            verdict, answer = classify_with_vlm(frames, details=bilingual)
    except Exception:
        app.logger.exception('VLM verification failed event=%s', event_id)
        verdict, answer = 'error', 'VLM 분석 중 오류가 발생했습니다. 서버 로그를 확인해 주세요.'
    try:
        connection = open_db_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    'UPDATE fire_event SET vlm_result=%s, vlm_answer=%s, vlm_prompt_version=%s, vlm_prompt=%s WHERE event_id=%s',
                    (verdict, answer, VLM_PROMPT_VERSION, VLM_PROMPT, event_id))
                cursor.execute('UPDATE model_comparison_event SET vlm_result=%s, vlm_answer=%s WHERE event_id=%s',
                               (verdict, answer, event_id))
                cursor.execute('''INSERT INTO vlm_language_result
                    (event_id, english_answer, korean_translation, translation_status, request_history)
                    VALUES (%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE
                    english_answer=VALUES(english_answer), korean_translation=VALUES(korean_translation),
                    translation_status=VALUES(translation_status), request_history=VALUES(request_history)''',
                    (event_id, bilingual.get('english'), bilingual.get('korean'),
                     bilingual.get('translation_status', 'not_requested'), json.dumps(bilingual.get('attempts', []), ensure_ascii=False)))
        finally:
            connection.close()
    except Exception:
        app.logger.exception('VLM result persistence failed event=%s', event_id)
        verdict, answer = 'error', '판정 결과 DB 저장에 실패했습니다. 서버 로그를 확인해 주세요.'
    finally:
        with _jobs_lock:
            record.update(status=verdict, answer=answer, finished_at=time.monotonic())
            job['latest_vlm_result'] = verdict
            job['latest_vlm_answer'] = answer
            job['latest_vlm_korean'] = bilingual.get('korean')
            job['translation_status'] = bilingual.get('translation_status', 'not_requested')
            job['pending_vlm'] -= 1
        app.logger.info('VLM complete event=%s result=%s elapsed=%.2fs answer=%s',
                        event_id, verdict, time.monotonic()-started, answer)


def vlm_worker():
    while True:
        job, event_id, frames = _vlm_queue.get()
        try:
            run_vlm_for_event(job, event_id, frames)
        except Exception:
            app.logger.exception('Unexpected VLM worker failure event=%s', event_id)
            with _jobs_lock:
                job['vlm_events'][event_id].update(status='error', finished_at=time.monotonic())
                job['pending_vlm'] = max(0, job['pending_vlm'] - 1)
        finally:
            del frames
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _vlm_queue.task_done()


def enqueue_vlm(job, event_id, frames):
    global _vlm_worker_started
    with _jobs_lock:
        if not _vlm_worker_started:
            threading.Thread(target=vlm_worker, daemon=True, name='vlm-worker').start()
            _vlm_worker_started = True
        job['pending_vlm'] += 1
    _vlm_queue.put((job, event_id, frames))
    app.logger.info('VLM queued event=%s waiting=%s', event_id, _vlm_queue.qsize())


def locations_are_similar(first, second, maximum_distance=EVENT_LOCATION_DISTANCE):
    if first is None or second is None:
        return False
    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5 <= maximum_distance


def update_event_detection(event, video_second, confidence, center):
    connection = open_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                '''UPDATE fire_event
                   SET last_detected_at=CURRENT_TIMESTAMP, video_last_seen_seconds=%s,
                       yolo_confidence=GREATEST(yolo_confidence, %s), detection_count=%s,
                       location_x=%s, location_y=%s
                   WHERE event_id=%s AND event_state='ACTIVE' ''',
                (round(video_second, 3), round(confidence * 100, 1), event['detection_count'],
                 round(center[0], 6), round(center[1], 6), event['event_id']),
            )
    finally:
        connection.close()


def end_event(event_id):
    connection = open_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE fire_event SET event_state='ENDED', ended_at=CURRENT_TIMESTAMP WHERE event_id=%s AND event_state='ACTIVE'",
                (event_id,),
            )
    finally:
        connection.close()


def start_event_verification(job, candidates, center):
    selected = sorted(candidates, key=lambda item: item['frame_index'])
    frames = [item['frame'] for item in selected]
    hashes = [perceptual_hash(frame) for frame in frames]
    evidence_names = []
    for index, frame in enumerate(frames, 1):
        name = f"{job['token']}_event_{len(job['event_ids']) + 1}_{index}.jpg"
        cv2.imwrite(str(RESULT_DIR / name), frame)
        evidence_names.append(name)

    cached = None
    dino_scores = {name: max(item['dino'][name] for item in candidates) for name in ('fire', 'smoke', 'lights', 'clouds')}
    connection = open_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                '''SELECT vlm_result, vlm_answer, event_hashes FROM fire_event
                   WHERE 1=0 AND event_hashes IS NOT NULL AND vlm_result IN ('confirmed','false_alarm','uncertain')
                   ORDER BY detected_at DESC LIMIT 50'''
            )
            for old in cursor.fetchall():
                cached_data = json.loads(old['event_hashes'])
                if not isinstance(cached_data, dict) or cached_data.get('prompt') != VLM_PROMPT_VERSION:
                    continue
                if len([line for line in (old['vlm_answer'] or '').splitlines() if line.strip()]) < 2:
                    continue
                old_hashes = cached_data['hashes']
                if hashes_are_similar(hashes, old_hashes):
                    cached = old
                    break
            cursor.execute(
                '''INSERT INTO fire_event
                   (source_filename, yolo_confidence, vlm_result, vlm_answer,
                     vlm_prompt_version, vlm_prompt, event_hashes, evidence_names,
                     event_state, last_detected_at, detection_count,
                     video_started_seconds, video_last_seen_seconds, location_x, location_y,
                     dino_fire_score, dino_smoke_score, dino_light_score, dino_cloud_score,
                     dino_threshold, dino_passed, yolo_candidate_count, dino_pass_count)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s,
                           'ACTIVE', CURRENT_TIMESTAMP, %s, %s, %s, %s, %s,
                           %s, %s, %s, %s, %s, TRUE, %s, %s)''',
                (job['original_name'], round(max(item['confidence'] for item in selected) * 100, 1),
                 cached['vlm_result'] if cached else None,
                 cached['vlm_answer'] if cached else None,
                  VLM_PROMPT_VERSION, VLM_PROMPT,
                  json.dumps({'prompt': VLM_PROMPT_VERSION, 'hashes': hashes,
                              'yolo_condition': job.get('detection_mode', 'or').upper(),
                              'yolo_threshold': job.get('yolo_threshold', 40)}), json.dumps(evidence_names),
                  len(candidates), round(candidates[0]['video_second'], 3),
                  round(candidates[-1]['video_second'], 3), round(center[0], 6), round(center[1], 6),
                  round(dino_scores['fire'] * 100, 3), round(dino_scores['smoke'] * 100, 3),
                  round(dino_scores['lights'] * 100, 3), round(dino_scores['clouds'] * 100, 3),
                  round(candidates[0]['dino']['threshold'], 3), len(candidates), len(candidates)),
            )
            event_id = cursor.lastrowid
            photo_urls = [f'/event-evidence/{event_id}/{name}' for name in evidence_names]
            photo_urls += [None] * (3 - len(photo_urls))
            cursor.execute(
                '''INSERT INTO model_comparison_event
                   (event_id, source_filename, yolo_score, dino_fire_score, dino_smoke_score,
                    dino_light_score, dino_cloud_score, dino_threshold, dino_passed,
                    vlm_result, vlm_answer, photo_1_url, photo_2_url, photo_3_url)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,TRUE,%s,%s,%s,%s,%s)''',
                (event_id, job['original_name'], round(max(item['confidence'] for item in selected) * 100, 1),
                 round(dino_scores['fire'] * 100, 3), round(dino_scores['smoke'] * 100, 3),
                 round(dino_scores['lights'] * 100, 3), round(dino_scores['clouds'] * 100, 3),
                 round(candidates[0]['dino']['threshold'], 3),
                 cached['vlm_result'] if cached else None, cached['vlm_answer'] if cached else None,
                 photo_urls[0], photo_urls[1], photo_urls[2]),
            )
    finally:
        connection.close()
    job.setdefault('vlm_events', {})[event_id] = {'event_id': event_id, 'status': 'queued'}
    job['event_ids'].append(event_id)
    job['evidence_names'] = evidence_names
    if cached:
        job['vlm_events'][event_id].update(status=cached['vlm_result'], answer=cached['vlm_answer'], cached=True, finished_at=time.monotonic())
        job['latest_vlm_result'] = cached['vlm_result']
        job['latest_vlm_answer'] = cached['vlm_answer']
        return event_id
    enqueue_vlm(job, event_id, frames)
    return event_id


def clean_old_results():
    cutoff = time.time() - 24 * 60 * 60
    protected = set()
    try:
        connection = open_db_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute('SELECT evidence_names FROM fire_event WHERE evidence_names IS NOT NULL')
                for row in cursor.fetchall():
                    try:
                        protected.update(json.loads(row['evidence_names'] or '[]'))
                    except (TypeError, json.JSONDecodeError):
                        continue
        finally:
            connection.close()
    except pymysql.MySQLError:
        app.logger.exception('Could not load protected event evidence list')
    for path in RESULT_DIR.glob('*'):
        if path.is_file() and path.name not in protected and path.stat().st_mtime < cutoff:
            path.unlink(missing_ok=True)


def prepare_image_test_video(image_path, video_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('이미지를 열 수 없습니다. 지원되는 이미지 파일인지 확인하세요.')
    height, width = image.shape[:2]
    if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
        raise ValueError('이미지 크기가 너무 큽니다. 4천만 픽셀 이하 이미지를 선택해 주세요.')
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), IMAGE_TEST_FPS, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError('이미지 테스트용 영상을 만들 수 없습니다.')
    try:
        for _ in range(IMAGE_TEST_FRAMES):
            writer.write(image)
    finally:
        writer.release()


def inspect_video(source_path, result_path, confidence_threshold=0.40):
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise ValueError('영상을 열 수 없습니다. 지원되는 영상 파일인지 확인하세요.')
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError('영상 크기를 확인할 수 없습니다.')
    writer = cv2.VideoWriter(str(result_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError('결과 영상 파일을 생성하지 못했습니다.')

    class_counts = {'cloud': 0, 'fire': 0, 'light': 0, 'smoke': 0}
    alert_frames = 0
    max_alert_confidence = 0.0
    processed_frames = 0
    model = get_model()
    try:
        with _model_lock:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                result = model.predict(
                    frame, conf=confidence_threshold, imgsz=640,
                    classes=alert_class_ids(model), verbose=False,
                )[0]
                frame_has_alert = False
                if result.boxes is not None:
                    for class_id, confidence in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
                        label = result.names[int(class_id)]
                        class_counts[label] = class_counts.get(label, 0) + 1
                        if label in {'fire', 'smoke'}:
                            frame_has_alert = True
                            max_alert_confidence = max(max_alert_confidence, float(confidence))
                if frame_has_alert:
                    alert_frames += 1
                writer.write(plot_alerts(result))
                processed_frames += 1
    finally:
        capture.release()
        writer.release()

    if processed_frames == 0:
        raise ValueError('영상에서 처리할 프레임을 찾지 못했습니다.')
    return {
        'filename': source_path.name,
        'processed_frames': processed_frames,
        'total_frames': total_frames,
        'duration_seconds': round(processed_frames / fps, 1),
        'alert_frames': alert_frames,
        'max_alert_confidence': round(max_alert_confidence * 100, 1),
        'class_counts': class_counts,
        'has_alert': alert_frames > 0,
    }


def generate_inspection_stream(job):
    source_path = job['source_path']
    result_path = job['result_path']
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        job.update(status='error', error='영상을 열 수 없습니다.')
        source_path.unlink(missing_ok=True)
        return
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(str(result_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    class_counts = {'cloud': 0, 'fire': 0, 'light': 0, 'smoke': 0}
    alert_frames = 0
    max_alert_confidence = 0.0
    processed_frames = 0
    event_candidates = []
    active_events = []
    metric_buckets = {}
    metric_started = time.perf_counter()
    metric_run_id = start_metric_run(job)
    confidence_threshold = job.get('yolo_threshold', 40) / 100
    job['status'] = 'running'
    try:
        with _model_lock:
            model = get_model()
            while True:
                if job.get('stop_requested'):
                    break
                ok, frame = capture.read()
                if not ok:
                    break
                result = model.predict(
                    frame, conf=confidence_threshold, imgsz=640,
                    classes=alert_class_ids(model), verbose=False,
                )[0]
                frame_has_alert = False
                frame_confidence = 0.0
                detected_labels = set()
                alert_centers = []
                alert_boxes = []
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls.item())
                        confidence = float(box.conf.item())
                        label = result.names[int(class_id)]
                        detected_labels.add(label)
                        class_counts[label] = class_counts.get(label, 0) + 1
                        if label in {'fire', 'smoke'}:
                            frame_has_alert = True
                            frame_confidence = max(frame_confidence, confidence)
                            max_alert_confidence = max(max_alert_confidence, confidence)
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            alert_centers.append(((x1 + x2) / (2 * width), (y1 + y2) / (2 * height)))
                            alert_boxes.append((x1, y1, x2, y2))
                if frame_has_alert:
                    alert_frames += 1
                video_second = processed_frames / fps
                center = None
                if alert_centers:
                    center = (sum(p[0] for p in alert_centers) / len(alert_centers), sum(p[1] for p in alert_centers) / len(alert_centers))
                dino = None
                if matches_yolo_condition(detected_labels, job.get('detection_mode', 'or')) and center is not None:
                    with _dino_lock:
                        dino = classify_yolo_crops_with_dino(frame, alert_boxes)
                    second = int(video_second)
                    bucket = metric_buckets.setdefault(second, {
                        'yolo': 0, 'passed': 0, 'blocked': 0, 'yolo_score': 0.0,
                        'fire': 0.0, 'smoke': 0.0, 'lights': 0.0, 'clouds': 0.0,
                    })
                    bucket['yolo'] += 1
                    bucket['passed' if dino['passed'] else 'blocked'] += 1
                    bucket['yolo_score'] = max(bucket['yolo_score'], frame_confidence * 100)
                    for name in ('fire', 'smoke', 'lights', 'clouds'):
                        bucket[name] = max(bucket[name], dino[name] * 100)
                for event in list(active_events):
                    if video_second - event['last_seen_second'] >= EVENT_END_QUIET_SECONDS:
                        end_event(event['event_id'])
                        job['vlm_events'][event['event_id']]['monitoring'] = False
                        active_events.remove(event)
                    else:
                        job['vlm_events'][event['event_id']]['quiet_remaining'] = max(0, round(EVENT_END_QUIET_SECONDS - (video_second - event['last_seen_second'])))
                matched_event = None
                if frame_has_alert and center is not None:
                    nearby = [event for event in active_events if locations_are_similar(center, event['center'])]
                    if nearby:
                        matched_event = min(nearby, key=lambda event: (event['center'][0] - center[0]) ** 2 + (event['center'][1] - center[1]) ** 2)
                        matched_event['last_seen_second'] = video_second
                        matched_event['detection_count'] += 1
                        matched_event['max_confidence'] = max(matched_event['max_confidence'], frame_confidence)
                        matched_event['center'] = (matched_event['center'][0] * .8 + center[0] * .2, matched_event['center'][1] * .8 + center[1] * .2)
                        if video_second - matched_event['last_db_update_second'] >= 1.0:
                            update_event_detection(matched_event, video_second, frame_confidence, matched_event['center'])
                            matched_event['last_db_update_second'] = video_second
                if matches_yolo_condition(detected_labels, job.get('detection_mode', 'or')) and center is not None and matched_event is None:
                    if not dino['passed']:
                        event_candidates = []
                        annotated = overlay(plot_alerts(result), job)
                        writer.write(annotated)
                        processed_frames += 1
                        job['progress'] = round(processed_frames / total_frames * 100) if total_frames else 0
                        encoded, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
                        if encoded:
                            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
                        continue
                    if event_candidates and not locations_are_similar(center, event_candidates[-1]['center']):
                        event_candidates = []
                    event_candidates.append({
                        'frame_index': processed_frames,
                        'video_second': video_second,
                        'confidence': frame_confidence,
                        'frame': frame.copy(),
                        'center': center,
                        'dino': dino,
                    })
                    event_candidates = [item for item in event_candidates if video_second - item['video_second'] <= EVENT_START_WINDOW_SECONDS]
                    if len(event_candidates) >= 5:
                        representative_frames = [
                            event_candidates[0], event_candidates[2], event_candidates[4],
                        ]
                        event_center = (sum(item['center'][0] for item in event_candidates[:5]) / 5, sum(item['center'][1] for item in event_candidates[:5]) / 5)
                        event_id = start_event_verification(job, representative_frames, event_center)
                        active_events.append({'event_id': event_id, 'center': event_center, 'last_seen_second': video_second,
                                              'last_db_update_second': video_second, 'detection_count': 5,
                                              'max_confidence': frame_confidence})
                        event_candidates = []
                elif matched_event is not None:
                    event_candidates = []
                annotated = overlay(plot_alerts(result), job)
                writer.write(annotated)
                processed_frames += 1
                job['progress'] = round(processed_frames / total_frames * 100) if total_frames else 0
                encoded, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if encoded:
                    yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
        if processed_frames == 0:
            raise ValueError('영상에서 처리할 프레임을 찾지 못했습니다.')
        job['status'] = 'vlm'
        last_frame = result.orig_img.copy()
        while job['pending_vlm']:
            encoded, jpeg = cv2.imencode('.jpg', overlay(last_frame.copy(), job))
            if encoded:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
            time.sleep(.25)
        # Final hold explicitly shows the last-frame verdict if it arrived after EOF.
        if job.get('vlm_events'):
            for _ in range(max(1, round(fps))):
                final_frame = overlay(last_frame.copy(), job)
                writer.write(final_frame)
            encoded, jpeg = cv2.imencode('.jpg', final_frame)
            if encoded:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
        writer.release()
        job['status'] = 'finalizing'
        import imageio_ffmpeg
        converted = result_path.with_name(result_path.stem + '_web.mp4')
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-i', str(result_path),
                        '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', str(converted)], check=True,
                       capture_output=True, timeout=600)
        converted.replace(result_path)
        job['report'] = {
            'filename': job['original_name'],
            'detection_mode': job.get('detection_mode', 'or'),
            'yolo_threshold': job.get('yolo_threshold', 40),
            'source_url': job.get('source_url'),
            'processed_frames': processed_frames,
            'total_frames': total_frames,
            'duration_seconds': round(processed_frames / fps, 1),
            'alert_frames': alert_frames,
            'max_alert_confidence': round(max_alert_confidence * 100, 1),
            'class_counts': class_counts,
            'has_alert': alert_frames > 0,
            'result_name': result_path.name,
            'event_ids': job['event_ids'],
            'evidence_names': job.get('evidence_names', []),
            'vlm_result': job.get('latest_vlm_result'),
            'vlm_answer': job.get('latest_vlm_answer'),
            'vlm_korean': job.get('latest_vlm_korean'),
            'translation_status': job.get('translation_status'),
            'stopped_early': job.get('stop_requested', False),
        }
        job['video_complete'] = True
        job.update(status='vlm' if job['pending_vlm'] else 'complete', progress=100)
    except GeneratorExit:
        job.update(status='cancelled', error='검사 화면 연결이 종료되었습니다.')
        raise
    except Exception as exc:
        app.logger.exception('Video inspection failed')
        job.update(status='error', error=f'검사 중 오류가 발생했습니다: {exc}')
        result_path.unlink(missing_ok=True)
    finally:
        for event in active_events:
            try:
                end_event(event['event_id'])
            except Exception:
                app.logger.exception('Failed to close event %s', event['event_id'])
        capture.release()
        writer.release()
        source_path.unlink(missing_ok=True)
        try:
            finish_metric_run(metric_run_id, job, metric_buckets, processed_frames, fps, time.perf_counter() - metric_started)
        except Exception:
            app.logger.exception('Failed to save inspection metrics')


def get_db():
    if 'db' not in g:
        g.db = open_db_connection()
    return g.db


def load_recent_events():
    with get_db().cursor() as cursor:
        cursor.execute(
            '''SELECT e.event_id, e.detected_at, e.status, e.accepted_at,
                      e.yolo_confidence, e.vlm_result, e.vlm_answer,
                      e.vlm_prompt_version, e.vlm_prompt,
                      e.event_state, e.last_detected_at, e.ended_at,
                      e.detection_count, e.video_started_seconds, e.video_last_seen_seconds,
                      e.dino_fire_score, e.dino_smoke_score, e.dino_light_score, e.dino_cloud_score,
                      e.dino_threshold, e.dino_passed, e.yolo_candidate_count, e.dino_pass_count,
                      e.evidence_names,
                      l.english_answer, l.korean_translation, l.translation_status, l.request_history,
                      u.username AS accepted_by_username,
                      m.human_verdict, m.human_notes, m.labeled_at,
                      reviewer.username AS labeled_by_username
               FROM fire_event e
               LEFT JOIN `user` u ON u.user_id = e.accepted_by
               LEFT JOIN vlm_language_result l ON l.event_id = e.event_id
               LEFT JOIN model_comparison_event m ON m.event_id = e.event_id
               LEFT JOIN `user` reviewer ON reviewer.user_id = m.labeled_by
               ORDER BY e.detected_at DESC LIMIT 20'''
        )
        events = cursor.fetchall()
        for event in events:
            try:
                names = json.loads(event['evidence_names'] or '[]')
            except (TypeError, json.JSONDecodeError):
                names = []
            event['evidence_urls'] = [url_for('event_evidence', event_id=event['event_id'], filename=name) for name in names]
        return events


@app.teardown_appcontext
def close_db(error=None):
    db = g.pop('db', None)
    if db:
        db.close()


@app.before_request
def protect_forms():
    if request.method == 'POST':
        expected = session.get('csrf_token', '')
        if not expected or not secrets.compare_digest(expected, request.form.get('csrf_token', '')):
            abort(400, description='요청이 만료되었습니다. 페이지를 새로고침한 뒤 다시 시도하세요.')
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_urlsafe(32)


@app.after_request
def security_headers(response):
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    return response


@app.errorhandler(pymysql.MySQLError)
def database_error(error):
    app.logger.error('Database operation failed (%s)', type(error).__name__)
    return render_template('auth.html', mode='login', error='DB 연결을 처리하지 못했습니다. 잠시 후 다시 시도하세요.'), 503


@app.route('/')
@login_required
def home():
    with get_db().cursor() as cursor:
        cursor.execute('SELECT user_id, username, role, created_at FROM `user` WHERE user_id=%s', (session['user_id'],))
        user = cursor.fetchone()
    events = load_recent_events()
    if not user:
        session.clear()
        return redirect(url_for('login'))
    job_token = session.get('active_job')
    if job_token and job_token not in inspection_jobs:
        session.pop('active_job', None)
        job_token = None
    return render_template(
        'home.html', user=user, report=session.get('last_report'),
        job_token=job_token, events=events,
        current_job=inspection_jobs.get(job_token, {}),
    )


@app.post('/inspect')
@login_required
def inspect():
    media_upload = request.files.get('media')
    source_url = request.form.get('youtube_url', '').strip()
    detection_mode = request.form.get('detection_mode', 'or').lower()
    if detection_mode not in {'or', 'and'}:
        abort(400)
    try:
        yolo_threshold = parse_yolo_threshold(request.form)
    except ValueError as exc:
        flash(str(exc))
        return redirect(url_for('home'))
    upload = None
    source_type = None
    if source_url:
        try:
            source_url = youtube_url(source_url)
        except ValueError as exc:
            flash(str(exc))
            return redirect(url_for('home'))
        suffix = '.mp4'
        source_type = 'youtube'
    elif media_upload and media_upload.filename:
        upload = media_upload
        suffix = Path(upload.filename).suffix.lower()
        if suffix in ALLOWED_VIDEO_EXTENSIONS:
            source_type = 'video'
        elif suffix in ALLOWED_IMAGE_EXTENSIONS:
            source_type = 'image'
        else:
            flash('지원되는 영상 또는 이미지 파일을 선택해 주세요.')
            return redirect(url_for('home'))
    else:
        flash('영상, 이미지 또는 YouTube 주소를 입력해 주세요.')
        return redirect(url_for('home'))

    clean_old_results()
    token = uuid.uuid4().hex
    source_path = RESULT_DIR / f'{token}_source{suffix if source_type != "image" else ".mp4"}'
    result_name = f'{token}_result.mp4'
    result_path = RESULT_DIR / result_name
    if source_type == 'image':
        image_path = RESULT_DIR / f'{token}_upload{suffix}'
        upload.save(image_path)
        try:
            prepare_image_test_video(image_path, source_path)
        except (ValueError, RuntimeError) as exc:
            image_path.unlink(missing_ok=True)
            source_path.unlink(missing_ok=True)
            flash(str(exc))
            return redirect(url_for('home'))
    elif not source_url:
        upload.save(source_path)
    inspection_jobs[token] = {
        'token': token,
        'user_id': session['user_id'],
        'source_path': source_path,
        'result_path': result_path,
        'original_name': 'YouTube 영상' if source_url else Path(upload.filename).name,
        'source_type': source_type,
        'detection_mode': detection_mode,
        'yolo_threshold': yolo_threshold,
        'source_url': source_url or None,
        'vlm_events': {},
        'status': 'downloading' if source_url else 'pending',
        'progress': 0,
        'error': None,
        'report': None,
        'event_ids': [],
        'evidence_names': [],
        'pending_vlm': 0,
        'video_complete': False,
        'latest_vlm_result': None,
        'latest_vlm_answer': None,
        'stop_requested': False,
    }
    if source_url:
        (RESULT_DIR / f'{token}_source.json').write_text(
            json.dumps({'source_url': source_url, 'job_token': token,
                        'yolo_condition': detection_mode.upper(),
                        'yolo_threshold': yolo_threshold}, ensure_ascii=False), encoding='utf-8')
        threading.Thread(target=download_youtube, args=(inspection_jobs[token], RESULT_DIR), daemon=True).start()
    session.pop('last_report', None)
    session['active_job'] = token
    return redirect(url_for('home'))


@app.get('/inspection-stream/<token>')
@login_required
def inspection_stream(token):
    job = inspection_jobs.get(token)
    if not job or job['user_id'] != session['user_id']:
        abort(404)
    with _jobs_lock:
        if job['status'] != 'pending':
            abort(409)
        job['status'] = 'starting'
    return Response(generate_inspection_stream(job), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.get('/inspection-status/<token>')
@login_required
def inspection_status(token):
    job = inspection_jobs.get(token)
    if not job or job['user_id'] != session['user_id']:
        return jsonify(status='error', error='검사 작업을 찾을 수 없습니다.'), 404
    if job['status'] == 'complete':
        report = job['report']
        session['last_report'] = job['report']
        session.pop('active_job', None)
    elif job['status'] in {'error', 'cancelled'}:
        session.pop('active_job', None)
    return jsonify(
        status=job['status'], progress=job['progress'], error=job['error'],
        stop_requested=job.get('stop_requested', False),
        vlm_feedback=feedback(job), pending_vlm=job['pending_vlm'],
    )


@app.post('/inspection-stop/<token>')
@login_required
def inspection_stop(token):
    job = inspection_jobs.get(token)
    if not job or job['user_id'] != session['user_id']:
        return jsonify(ok=False, error='검사 작업을 찾을 수 없습니다.'), 404
    if job['status'] in {'complete', 'error', 'cancelled'}:
        return jsonify(ok=False, error='이미 종료된 검사입니다.'), 409
    job['stop_requested'] = True
    return jsonify(ok=True)


@app.post('/events/<int:event_id>/accept')
@login_required
def accept_event(event_id):
    try:
        verdict, notes = parse_human_review(request.form)
    except ValueError as exc:
        if request.headers.get('Accept') == 'application/json':
            return jsonify(ok=False, error=str(exc)), 400
        flash(str(exc))
        return redirect(url_for('home') + '#event-board')

    connection = get_db()
    try:
        connection.begin()
        with connection.cursor() as cursor:
            cursor.execute(
                '''UPDATE fire_event
                   SET status='RECEIVED', accepted_at=CURRENT_TIMESTAMP, accepted_by=%s
                   WHERE event_id=%s AND status='UNRECEIVED' ''',
                (session['user_id'], event_id),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                if request.headers.get('Accept') == 'application/json':
                    return jsonify(ok=False, error='이미 접수됐거나 존재하지 않는 이벤트입니다.'), 409
                flash('이미 접수됐거나 존재하지 않는 이벤트입니다.')
                return redirect(url_for('home') + '#event-board')
            cursor.execute(
                '''UPDATE model_comparison_event
                   SET human_verdict=%s, human_notes=%s,
                       labeled_at=CURRENT_TIMESTAMP, labeled_by=%s
                   WHERE event_id=%s''',
                (verdict, notes, session['user_id'], event_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError('비교 이벤트 기록을 찾을 수 없습니다.')
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    if request.headers.get('Accept') == 'application/json':
        return jsonify(ok=True)
    return redirect(url_for('home') + '#event-board')


@app.get('/event-board/rows')
@login_required
def event_board_rows():
    return render_template('_event_rows.html', events=load_recent_events())


@app.get('/inspection-result/<filename>')
@login_required
def inspection_result(filename):
    report = session.get('last_report') or {}
    allowed = [report.get('result_name', ''), *report.get('evidence_names', [])]
    if not any(secrets.compare_digest(filename, allowed_name) for allowed_name in allowed):
        abort(404)
    mimetype = 'image/jpeg' if filename.lower().endswith('.jpg') else 'video/mp4'
    return send_from_directory(RESULT_DIR, filename, mimetype=mimetype, as_attachment=False)


@app.get('/event-evidence/<int:event_id>/<filename>')
@login_required
def event_evidence(event_id, filename):
    with get_db().cursor() as cursor:
        cursor.execute('SELECT evidence_names FROM fire_event WHERE event_id=%s', (event_id,))
        event = cursor.fetchone()
    if not event:
        abort(404)
    try:
        allowed = json.loads(event['evidence_names'] or '[]')
    except (TypeError, json.JSONDecodeError):
        allowed = []
    if not any(secrets.compare_digest(filename, name) for name in allowed):
        abort(404)
    return send_from_directory(RESULT_DIR, filename, mimetype='image/jpeg', as_attachment=False)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not re.fullmatch(r'[A-Za-z0-9_]{3,100}', username):
            error = '아이디는 영문, 숫자, 밑줄로 3~100자 입력해 주세요.'
        elif not 8 <= len(password) <= 128:
            error = '비밀번호는 8~128자로 입력해 주세요.'
        elif password != request.form.get('password_confirm', ''):
            error = '비밀번호 확인이 일치하지 않습니다.'
        else:
            hashed = generate_password_hash(password, method='scrypt')
            try:
                with get_db().cursor() as cursor:
                    cursor.execute('INSERT INTO `user` (username, password_hash, role) VALUES (%s, %s, %s)', (username, hashed, 'USER'))
            except pymysql.err.IntegrityError as exc:
                if exc.args[0] != 1062:
                    raise
                error = '이미 사용 중인 아이디입니다.'
            else:
                flash('회원가입이 완료되었습니다. 로그인해 주세요.')
                return redirect(url_for('login'))
    return render_template('auth.html', mode='signup', error=error), 400 if error else 200


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = None
        if len(username) <= 100 and len(password) <= 128:
            with get_db().cursor() as cursor:
                cursor.execute('SELECT user_id, password_hash FROM `user` WHERE username=%s', (username,))
                user = cursor.fetchone()
        valid = check_password_hash(user['password_hash'] if user else DUMMY_HASH, password)
        if user and valid:
            session.clear()
            session['user_id'] = user['user_id']
            session['csrf_token'] = secrets.token_urlsafe(32)
            session.permanent = True
            return redirect(url_for('home'))
        error = '아이디 또는 비밀번호가 올바르지 않습니다.'
    return render_template('auth.html', mode='login', error=error), 401 if error else 200


@app.post('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.errorhandler(413)
def file_too_large(error):
    flash('영상 파일은 최대 500MB까지 업로드할 수 있습니다.')
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(
        host=os.getenv('APP_HOST', '0.0.0.0'),
        port=int(os.getenv('APP_PORT', '5000')),
        debug=False,
    )
