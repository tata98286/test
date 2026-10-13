const promptDialog = document.getElementById('vlm-prompt-dialog');
if (promptDialog) {
  const fixed = document.getElementById('vlm-fixed-prompt');
  const draft = document.getElementById('vlm-test-prompt');
  const mode = document.getElementById('vlm-prompt-mode');
  const value = document.getElementById('vlm-test-prompt-value');
  const label = document.getElementById('vlm-prompt-mode-label');
  const message = document.getElementById('prompt-dialog-message');
  const apply = document.getElementById('apply-vlm-prompt');
  const running = promptDialog.dataset.running === 'true';
  const storageKey = 'its.vlm.prompt.test-draft.v1';
  try { draft.value = localStorage.getItem(storageKey) || ''; } catch (_) {}
  document.getElementById('open-vlm-prompt').addEventListener('click', () => {
    promptDialog.querySelectorAll('input[name="prompt-choice"]').forEach(input => { input.checked = input.value === mode.value; });
    promptDialog.showModal();
  });
  document.getElementById('close-vlm-prompt').addEventListener('click', () => promptDialog.close());
  document.getElementById('copy-fixed-prompt').addEventListener('click', () => { draft.value = fixed.value; draft.focus(); });
  apply.addEventListener('click', () => {
    const choice = promptDialog.querySelector('input[name="prompt-choice"]:checked').value;
    if (choice === 'test' && !draft.value.trim()) { message.textContent = '테스트 프롬프트를 입력해 주세요.'; draft.focus(); return; }
    mode.value = choice;
    value.value = choice === 'test' ? draft.value.trim() : '';
    try { localStorage.setItem(storageKey, draft.value); } catch (_) {}
    label.textContent = `VLM · ${choice === 'test' ? '테스트 프롬프트 (다음 검사에 적용)' : '고정 프롬프트'}`;
    promptDialog.close();
  });
  if (running) {
    apply.disabled = true;
    promptDialog.querySelectorAll('input[name="prompt-choice"]').forEach(input => { input.disabled = true; });
    message.textContent = '검사 중에는 적용 설정을 바꿀 수 없습니다. 검사가 끝난 뒤 선택해 주세요.';
  }
}
