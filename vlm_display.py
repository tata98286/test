"""Render the same VLM feedback into live JPEGs and saved video frames."""
import time
import cv2

STATES = {
    'queued': ('VLM: QUEUED', (0, 180, 255), 'VLM 검증 대기'),
    'analyzing': ('VLM: ANALYZING', (0, 180, 255), 'VLM 분석 중'),
    'confirmed': ('LLM: FIRE DETECTED!', (0, 0, 255), 'VLM: 실제 화재/연기'),
    'false_alarm': ('LLM: FALSE ALARM', (0, 255, 0), 'VLM: 오탐'),
    'uncertain': ('VLM: HUMAN REVIEW REQUIRED', (0, 165, 255), '판단 불확실 · 육안 확인 필요'),
    'error': ('VLM: ANALYSIS ERROR', (150, 150, 150), 'VLM 분석 오류'),
}


def feedback(job):
    records = list(job.get('vlm_events', {}).values())
    if not records:
        return None
    # An arrival is briefly emphasized even when another event is queued.
    recent = [r for r in records if time.monotonic() - r.get('finished_at', -1e20) < 1]
    chosen = max(recent, key=lambda r: r['finished_at']) if recent else records[-1]
    return dict(chosen)


def overlay(frame, job):
    state = feedback(job)
    if not state:
        return frame
    label, color, _ = STATES[state['status']]
    height, width = frame.shape[:2]
    # Keep web metadata separate from the original notebook's verdict alarm.
    text = f"EVT-{state['event_id']}"
    if state['status'] not in {'confirmed', 'false_alarm'}:
        text += ' | ' + label
    if state.get('cached'):
        text += ' (CACHED)'
    scale = min(0.8, max(0.25, (width - 24) / max(1, cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0])))
    cv2.rectangle(frame, (0, 0), (width, 42), (25, 25, 25), -1)
    cv2.putText(frame, text, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
    if 'quiet_remaining' in state:
        monitor = ('SAME EVENT | quiet timeout: ' + str(state['quiet_remaining']) + 's') if state.get('monitoring', True) else 'EVENT ENDED'
        cv2.putText(frame, monitor, (10, max(55, height - 18)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
    elapsed = time.monotonic() - state.get('finished_at', -1e20)
    if 0 <= elapsed < 1 and ((1.0 + elapsed) % .5) < .25:
        # Original: 20px border, centered text at y=150, font scale=2,
        # thickness=5; both border and text flash together for one second.
        cv2.rectangle(frame, (0, 0), (width, height), color, 20)
        text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 5)[0][0]
        cv2.putText(frame, label, ((width - text_width) // 2, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)
    return frame
