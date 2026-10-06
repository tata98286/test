"""dino_learning.py — 실전 crop 자동 수집·라벨링·헤드 재학습 폐루프
전제:
  - classify_yolo_crops_with_dino 가 후보별 점수 반환(amax 수정본)일 것
  - bundle/ 디렉토리에 anchor + eval 세트가 반출돼 있을 것 (Colab 반출 스크립트 참조)
활성화: 환경변수 DINO_AUTO_TRAIN=true
"""
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

TRAINING_DIR = Path(os.getenv('DINO_LEARNING_DIR', r'C:\its\dino_learning'))
CROP_DIR = TRAINING_DIR / 'crops'
CKPT_DIR = TRAINING_DIR / 'checkpoints'
BUNDLE_DIR = TRAINING_DIR / 'bundle'      # anchor/, eval/ 각각 labels.json 포함
for d in (CROP_DIR, CKPT_DIR, BUNDLE_DIR):
    d.mkdir(parents=True, exist_ok=True)

CLASSES = ['fire', 'smoke', 'lights', 'clouds']
CFG = {
    'enabled': os.getenv('DINO_AUTO_TRAIN', 'false').lower() == 'true',
    'min_new': int(os.getenv('DINO_TRAIN_MIN_NEW', '40')),
    'epochs': 3,
    'lr': 2e-4,                 # 원 학습(1e-3)의 1/5 — 미세조정
    'batch': 16,
    'pos_weight_cap': 5.0,
    'vlm_positive': False,      # VLM YES를 양성으로 쓸지(기본 False: 사람 판정만)
    'pass_recall_tol': 0.005,   # 회귀 게이트 허용 열화
    'block_recall_tol': 0.02,
}

_state = {'running': False, 'last_result': None, 'last_run_at': None}
_lock = threading.Lock()
_conn_factory = None
_base_ckpt = None
_reload_hook = None          # app이 등록: 새 체크포인트로 모델 재로드
_model_accessor = None       # app이 등록: 락 보호 하에 (model, heads) 사용


def init(connection_factory, base_checkpoint_path, reload_hook, model_accessor):
    global _conn_factory, _base_ckpt, _reload_hook, _model_accessor
    _conn_factory = connection_factory
    _base_ckpt = Path(base_checkpoint_path)
    _reload_hook = reload_hook
    _model_accessor = model_accessor
    # 배포 버전은 프로세스 재시작 뒤에도 유지한다. 테이블이 아직 준비되지
    # 않은 설치에서는 base 체크포인트로 정상 기동하고 상태에 경고만 남긴다.
    try:
        conn = _db()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    'SELECT checkpoint_path FROM dino_model_version '
                    'WHERE deployed=1 ORDER BY version_id DESC LIMIT 1')
                active = cursor.fetchone()
        finally:
            conn.close()
        if active:
            checkpoint = Path(active['checkpoint_path'])
            if checkpoint.is_file():
                _reload_hook(checkpoint)
            else:
                _state['startup_warning'] = f'배포 체크포인트 없음: {checkpoint}'
    except Exception as exc:
        _state['startup_warning'] = f'배포 버전 조회 실패: {type(exc).__name__}'


def _db():
    return _conn_factory()


# ============================================================
# 1) crop 자동 저장 (이벤트 생성 시)
# ============================================================
def save_event_crops(job_token, event_seq, candidates, event_id=None):
    """이벤트 대표 후보들의 각 bbox crop을 저장하고 PENDING 행 적립.
    candidates: start_event_verification의 selected — 각 항목은
    'frame', 'boxes', 'box_labels', 'video_second', 'dino' 보유."""
    if event_id is None:
        raise ValueError('crop 저장에는 확정된 event_id가 필요합니다.')
    rows = []
    written_paths = []
    for cand_idx, item in enumerate(candidates):
        dino = item.get('dino') or {}
        per_crop = dino.get('crops') or []
        frame = item['frame']
        for crop_idx, info in enumerate(per_crop):
            x1, y1, x2, y2 = info['box']
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue
            name = f"{job_token}_e{event_seq}_c{cand_idx}_{crop_idx}.jpg"
            path = CROP_DIR / name
            if not cv2.imwrite(str(path), frame[y1:y2, x1:x2]):
                continue
            written_paths.append(path)
            source_idx = int(info.get('source_index', crop_idx))
            labels_list = item.get('box_labels') or []
            confidence_list = item.get('box_confidences') or []
            yolo_label = labels_list[source_idx] if source_idx < len(labels_list) else None
            yolo_conf = (round(float(confidence_list[source_idx]), 5)
                         if source_idx < len(confidence_list) else None)
            rows.append((
                event_id, name, round(float(item.get('video_second', 0)), 3),
                str(path), yolo_label, yolo_conf,
                json.dumps(info['scores']),
            ))
    if not rows:
        return
    conn = _db()
    try:
        with conn.cursor() as cursor:
            cursor.executemany(
                '''INSERT INTO dino_training_sample
                   (event_id, source_filename, video_second, crop_path,
                    yolo_label, yolo_conf, dino_scores, status)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,'PENDING')''', rows)
    except Exception:
        for path in written_paths:
            path.unlink(missing_ok=True)
        raise
    finally:
        conn.close()


# ============================================================
# 2) 자동 라벨링 (판정 도착 시)
# ============================================================
def label_event_crops(event_id, source, verdict):
    """vlm: false_alarm → 음성 / confirmed → (CFG가 허용 시) 약한 양성
       human: FALSE_ALARM → 음성, FIRE → 박스의 YOLO 클래스 양성 (vlm 라벨 덮어쓰기)"""
    conn = _db()
    try:
        with conn.cursor() as cursor:
            if source == 'vlm':
                if verdict == 'false_alarm':
                    label, keep_status = '[0,0,0,0]', 'LABELED'
                elif verdict == 'confirmed' and CFG['vlm_positive']:
                    label, keep_status = None, 'LABELED'   # 클래스는 yolo_label로
                else:
                    return
            elif source == 'human':
                if verdict == 'FALSE_ALARM':
                    label, keep_status = '[0,0,0,0]', 'LABELED'
                elif verdict == 'FIRE':
                    label, keep_status = None, 'LABELED'
                else:
                    return
            else:
                return

            if label is None:  # 양성 — yolo_label 기반 multi-hot
                cursor.execute(
                    'SELECT sample_id, yolo_label FROM dino_training_sample '
                    'WHERE event_id=%s AND status IN (%s,%s)',
                    (event_id, 'PENDING', 'LABELED'))
                for row in cursor.fetchall():
                    sid = row['sample_id']
                    yl = row['yolo_label']
                    onehot = [0, 0, 0, 0]
                    if yl in ('fire', 'smoke'):
                        onehot[CLASSES.index(yl)] = 1
                    cursor.execute(
                        'UPDATE dino_training_sample SET label=%s, '
                        'label_source=%s, status=%s, labeled_at=NOW() '
                        'WHERE sample_id=%s',
                        (json.dumps(onehot), source, keep_status, sid))
                return

            where = "event_id=%s AND status IN ('PENDING','LABELED')"
            if source == 'vlm':          # 사람 라벨이 있으면 건드리지 않음
                where += " AND (label_source IS NULL OR label_source='vlm')"
            cursor.execute(
                f'UPDATE dino_training_sample SET label=%s, label_source=%s, '
                f'status=%s, labeled_at=NOW() WHERE {where}',
                (label, source, keep_status, event_id))
    finally:
        conn.close()


# ============================================================
# 3) 학습 스케줄링
# ============================================================
def maybe_schedule_training(force=False):
    if not CFG['enabled'] and not force:
        return
    with _lock:
        if _state['running']:
            return
        conn = _db()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) n FROM dino_training_sample "
                               "WHERE status='LABELED'")
                n = cursor.fetchone()['n']
        finally:
            conn.close()
        if not force and n < CFG['min_new']:
            return
        _state['running'] = True
    threading.Thread(target=_training_run, daemon=True,
                     name='dino-training').start()


# ============================================================
# 4) 학습 파이프라인
# ============================================================
def _embed(paths, use_accessor):
    """crop 경로들 → CLS 임베딩 (전처리는 classify와 완전 동일)"""
    import torch
    def run(model, heads):
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        out = []
        for i in range(0, len(paths), CFG['batch']):
            imgs = []
            for p in paths[i:i + CFG['batch']]:
                crop = cv2.imread(str(p))
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop = cv2.resize(crop, (518, 518), interpolation=cv2.INTER_AREA)
                imgs.append(crop)
            pixels = torch.from_numpy(np.stack(imgs)).to(
                device=device, dtype=dtype).permute(0, 3, 1, 2) / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype)[None, :, None, None]
            std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype)[None, :, None, None]
            with __import__('torch').inference_mode():
                cls = model(pixel_values=(pixels - mean) / std).last_hidden_state[:, 0]
            out.append(cls.float().cpu())
        return torch.cat(out)
    return use_accessor(run)


def _load_bundle_labels(kind):
    """bundle/{kind}/labels.json → [(경로, multi-hot)] — 없으면 빈 리스트"""
    meta = BUNDLE_DIR / kind / 'labels.json'
    if not meta.exists():
        return []
    data = json.loads(meta.read_text(encoding='utf-8'))
    root = BUNDLE_DIR / kind
    return [(root / rel, lab) for rel, lab in data.items() if (root / rel).exists()]


def _training_run():
    import torch
    import torch.nn as nn
    try:
        t0 = time.time()
        # ---- 1. 학습 데이터 수집: LABELED(마이닝) + anchor(재생) ----
        conn = _db()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT sample_id, crop_path, label FROM dino_training_sample "
                    "WHERE status='LABELED'")
                mined = cursor.fetchall()
        finally:
            conn.close()
        anchor = _load_bundle_labels('anchor')
        eval_set = _load_bundle_labels('eval')
        if not mined:
            _state['last_result'] = {'ok': False, 'reason': '확정된 신규 라벨 없음'}
            _state['last_run_at'] = str(datetime.now())
            return
        if not anchor:
            _state['last_result'] = {'ok': False,
                                     'reason': 'anchor 번들 없음 — Colab 반출 필요'}
            _state['last_run_at'] = str(datetime.now())
            return
        if not eval_set:
            _state['last_result'] = {'ok': False,
                                     'reason': 'eval 번들 없음 — 회귀 게이트 불가'}
            _state['last_run_at'] = str(datetime.now())
            return

        pairs = ([(Path(r['crop_path']), json.loads(r['label'])) for r in mined]
                 + anchor)
        paths = [p for p, _ in pairs]
        labels = torch.tensor([l for _, l in pairs], dtype=torch.float32)

        # ---- 2. 임베딩 1회 계산 (추론 락 점유 — 수십 초) ----
        emb = _embed(paths, _model_accessor)
        ev_paths = [p for p, _ in eval_set]
        ev_labels = torch.tensor([l for _, l in eval_set], dtype=torch.float32)
        eval_positive = (ev_labels[:, 0] + ev_labels[:, 1]) > 0
        if not bool(eval_positive.any()) or not bool((~eval_positive).any()):
            _state['last_result'] = {
                'ok': False, 'reason': 'eval 번들에는 양성과 음성이 모두 필요'}
            _state['last_run_at'] = str(datetime.now())
            return
        ev_emb = _embed(ev_paths, _model_accessor)

        # ---- 3. base 헤드 로드 → 여기서부터 미세조정 ----
        base_ckpt = torch.load(_base_ckpt, map_location='cpu', weights_only=False)
        spec = base_ckpt['head']['spec']
        threshold = float(base_ckpt['deploy']['threshold'])

        def make_heads():
            return nn.ModuleDict({
                'gate': nn.Sequential(nn.LayerNorm(spec['dim']),
                                      nn.Linear(spec['dim'], 2)),
                'aux': nn.Sequential(nn.LayerNorm(spec['dim']),
                                     nn.Linear(spec['dim'], spec['aux'])),
            })

        base_heads = make_heads()
        base_heads.load_state_dict(base_ckpt['head']['state_dict'], strict=True)
        base_heads.eval()

        new_heads = make_heads()
        new_heads.load_state_dict(base_ckpt['head']['state_dict'], strict=True)

        pos = labels.sum(0)
        neg = len(labels) - pos
        pos_w = torch.clamp(neg / pos.clamp(min=1), max=CFG['pos_weight_cap'])
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        ce = nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(new_heads.parameters(), lr=CFG['lr'],
                                weight_decay=1e-4)

        gate_y = (labels[:, 0] + labels[:, 1] > 0).long()
        n = len(emb)
        for epoch in range(CFG['epochs']):
            new_heads.train()
            perm = torch.randperm(n)
            for i in range(0, n, 64):
                idx = perm[i:i + 64]
                g_logits = new_heads['gate'](emb[idx])
                a_logits = new_heads['aux'](emb[idx])
                loss = bce(a_logits, labels[idx]) + 0.3 * ce(g_logits, gate_y[idx])
                loss.backward(); opt.step(); opt.zero_grad()

        # ---- 4. 회귀 게이트: base 대비 열화 없어야 배포 ----
        def metrics(heads):
            heads.eval()
            with torch.no_grad():
                probs = torch.sigmoid(heads['aux'](ev_emb))
            ps = torch.maximum(probs[:, 0], probs[:, 1])
            is_pass = (ev_labels[:, 0] + ev_labels[:, 1]) > 0
            pred = ps >= threshold
            return {
                'pass_recall': float(pred[is_pass].float().mean()),
                'block_recall': float((~pred[~is_pass]).float().mean()),
            }

        m_base, m_new = metrics(base_heads), metrics(new_heads)
        passed = (m_new['pass_recall'] >= m_base['pass_recall'] - CFG['pass_recall_tol']
                  and m_new['block_recall'] >= m_base['block_recall'] - CFG['block_recall_tol'])

        result = {'mined': len(mined), 'anchor': len(anchor),
                  'base': m_base, 'new': m_new, 'gate_passed': passed,
                  'elapsed': round(time.time() - t0, 1)}

        if passed:
            # ---- 5. 버전 체크포인트 저장 (형식은 get_dino 로더와 동일) ----
            ckpt = {
                'format': 'fire_dinov3_full_v2_auto',
                'created': str(datetime.now()),
                'backbone': base_ckpt['backbone'],          # 백본은 그대로
                'head': {'spec': spec,
                         'state_dict': {k: v for k, v in new_heads.state_dict().items()}},
                'deploy': dict(base_ckpt['deploy']),
            }
            out = CKPT_DIR / f"full_auto_{datetime.now():%Y%m%d_%H%M%S}.pt"
            torch.save(ckpt, out)
            result['checkpoint'] = str(out)

            conn = _db()
            try:
                conn.begin()
                with conn.cursor() as cursor:
                    cursor.execute('UPDATE dino_model_version SET deployed=0 WHERE deployed=1')
                    cursor.execute(
                        'INSERT INTO dino_model_version '
                        '(checkpoint_path, num_mined, num_anchor, metrics, deployed) '
                        'VALUES (%s,%s,%s,%s,1)',
                        (str(out), len(mined), len(anchor), json.dumps(result)))
                    cursor.execute(
                        "UPDATE dino_training_sample SET status='USED' "
                        "WHERE status='LABELED'")
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
            _reload_hook(out)     # app: _dino_model 리셋 → 다음 호출부터 새 버전
        _state['last_result'] = result
        _state['last_run_at'] = str(datetime.now())
    except Exception:
        import traceback
        _state['last_result'] = {'ok': False, 'error': traceback.format_exc()[-800:]}
    finally:
        with _lock:
            _state['running'] = False


def status():
    def bundle_count(kind):
        try:
            return len(_load_bundle_labels(kind))
        except Exception:
            return 0

    anchor_count = bundle_count('anchor')
    eval_count = bundle_count('eval')
    return {
        'config': CFG,
        'readiness': {
            'base_checkpoint': bool(_base_ckpt and _base_ckpt.is_file()),
            'anchor_count': anchor_count,
            'eval_count': eval_count,
            'auto_training_ready': bool(CFG['enabled'] and anchor_count and eval_count),
        },
        'state': _state,
    }
