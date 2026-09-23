const form = document.getElementById('inspection-form');
const videoInput = document.getElementById('video');
const button = document.getElementById('inspect-button');

if (form) {
  form.addEventListener('submit', (event) => {
    if (!videoInput.files.length && !document.getElementById('youtube-url').value.trim()) {
      event.preventDefault();
      videoInput.focus();
      return;
    }
    button.disabled = true;
    button.textContent = '영상 업로드 중...';
  });
}

const streamConfig = document.getElementById('stream-config');
if (streamConfig) {
  const progress = document.getElementById('live-progress');
  const image = document.getElementById('live-image');
  let streamStarted = image.hasAttribute('src');
  const checkStatus = async () => {
    try {
      const response = await fetch(streamConfig.dataset.statusUrl, {cache: 'no-store'});
      const job = await response.json();
      const statusNames = {downloading: 'YouTube 가져오는 중', vlm: 'VLM 판정 대기', finalizing: '결과 영상 저장 중'};
      progress.textContent = statusNames[job.status] || `${job.progress || 0}%`;
      if (job.status === 'downloading') progress.textContent += ` ${job.progress || 0}%`;
      if (job.status === 'pending' && !streamStarted) {
        streamStarted = true;
        image.src = streamConfig.dataset.streamUrl;
      }
      const state = job.vlm_feedback;
      const labels = {queued:'검증 대기', analyzing:'AI 분석 중', confirmed:'실제 화재/연기', false_alarm:'오탐', uncertain:'육안 확인 필요', error:'분석 오류'};
      if (state) document.getElementById('vlm-live-status').textContent = `EVT-${state.event_id} · VLM ${labels[state.status]}${state.cached ? ' (기존 판정 재사용)' : ''}`;

      if (job.status === 'complete') {
        window.location.reload();
        return;
      }
      if (job.status === 'error' || job.status === 'cancelled') {
        window.alert(job.error || '검사가 중단되었습니다.');
        window.location.reload();
        return;
      }
    } catch (error) {
      progress.textContent = '연결 확인 중';
    }
    window.setTimeout(checkStatus, 800);
  };
  checkStatus();
}

const stopForm = document.getElementById('stop-inspection-form');
if (stopForm) {
  stopForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const stopButton = document.getElementById('stop-inspection-button');
    if (stopButton.disabled) return;
    stopButton.disabled = true;
    stopButton.textContent = '종료 처리 중...';
    try {
      const response = await fetch(stopForm.action, {
        method: 'POST', body: new FormData(stopForm),
        headers: {'Accept': 'application/json'},
      });
      const result = await response.json();
      if (!response.ok || !result.ok) throw new Error(result.error || 'stop failed');
      stopButton.textContent = '현재 지점에서 종료 중';
    } catch (error) {
      stopButton.disabled = false;
      stopButton.textContent = '여기까지 검사 종료';
      window.alert('검사를 종료하지 못했습니다. 다시 시도해 주세요.');
    }
  });
}

const eventBoardConfig = document.getElementById('event-board-config');
const eventRows = document.getElementById('event-rows');
if (eventBoardConfig && eventRows) {
  let accepting = false;
  const refreshEvents = async () => {
    try {
      const response = await fetch(eventBoardConfig.dataset.rowsUrl, {cache: 'no-store'});
      const detailsAreOpen = eventRows.querySelector('details[open]') !== null;
      if (response.ok && !response.redirected && !accepting && !detailsAreOpen) {
        eventRows.innerHTML = await response.text();
      }
    } catch (error) {
      // Preserve existing rows during a temporary network failure.
    } finally {
      window.setTimeout(refreshEvents, 1500);
    }
  };
  eventRows.addEventListener('submit', async (event) => {
    const acceptForm = event.target;
    if (!(acceptForm instanceof HTMLFormElement)) return;
    event.preventDefault();
    if (accepting) return;
    accepting = true;
    const acceptButton = acceptForm.querySelector('button');
    acceptButton.disabled = true;
    try {
      const response = await fetch(acceptForm.action, {
        method: 'POST', body: new FormData(acceptForm),
        headers: {'Accept': 'application/json'},
      });
      if (!response.ok || response.redirected || !(await response.json()).ok) throw new Error('accept failed');
      acceptButton.textContent = '접수 완료';
    } catch (error) {
      acceptButton.disabled = false;
      window.alert('접수하지 못했습니다. 연결 상태를 확인하고 다시 시도해 주세요.');
    } finally {
      accepting = false;
    }
  });
  window.setTimeout(refreshEvents, 1500);
}
