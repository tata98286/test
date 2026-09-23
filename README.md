# ITS 회원가입 / 로그인

Flask + MySQL 기반 한국어 회원가입, 로그인, 내 공간, 로그아웃 페이지입니다.
비밀번호는 Werkzeug scrypt(개별 salt) 해시로 저장하고 검증합니다.
회원가입으로 생성되는 계정의 역할은 항상 USER입니다.
로그인 후 `C:\its\model\best.2026.09.17.pt` YOLO 모델로 영상 또는 이미지를 검사하고
화재·연기 탐지 결과를 확인할 수 있습니다. 이미지는 동일한 장면 5프레임으로 구성한
짧은 내부 테스트 클립으로 변환해 기존 이벤트 조건과 같은 흐름으로 검사합니다.
YOLO가 40% 이상으로 불 또는 연기를 감지한 장면 5장을 3초 안에 모으면
그중 첫 장, 가운데 장, 마지막 장을 `C:\its\model\Qwen2-VL-2B-Instruct`
로컬 VLM에 전달해 즉시 2차 판정합니다. 호출 뒤 10초 동안은 다시 호출하지
않습니다. 이벤트는 DB에 기록되고 화면에서 접수할 수 있습니다.
검사 화면의 영상/이미지 공용 파일 입력과 함께 `불 또는 연기(OR)`와
`불과 연기 모두(AND)` 조건을 선택해
같은 자료의 민감 탐지와 엄격 탐지 결과를 비교할 수 있습니다.

## 실행 (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --force-reinstall torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
Copy-Item .env.example .env
# .env에 DB 계정과 무작위 SECRET_KEY를 설정합니다.
.\.venv\Scripts\python.exe app.py
```

현재 PC에는 `.env`가 이미 설정되어 있으므로 덮어쓰지 마세요.
이 PC에서는 http://127.0.0.1:5000/signup, 같은 네트워크의 다른 PC에서는
`http://이-PC의-IP:5000/signup`에서 가입하고 로그인할 수 있습니다.
다른 환경에서는 먼저 `sql/001_create_its_user.sql`을 MySQL에서 실행하세요.
DB 계정에는 its.user의 SELECT, INSERT 권한이 필요합니다.
`.env`는 Git에서 제외되며 실제 비밀번호를 커밋하지 않습니다.

## 확인

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

검증은 설정된 DB에 고유한 테스트 계정을 생성하고 종료 시 삭제하므로
테스트 실행 계정에는 its.user의 DELETE 권한도 필요합니다.

로컬 개발용 서버입니다. 외부 배포 시 HTTPS, COOKIE_SECURE=true,
운영용 WSGI 서버와 로그인 시도 제한을 설정하세요.
# 2026-09-21 업데이트: VLM 영상 알람 / YouTube 검사

## 2026-09-22 VLM 원본 기준 복원

현재 실행 경로는 원본 노트북의 한 장 입력과 yes/no 질문을 사용합니다.
후보는 시간순으로 유지하며, 5개 중 첫/중간/끝 3장을 보관하고 가운데 장면 하나를 VLM에 전달합니다.
입력 크기 480×480, 원본 질문, max_new_tokens=50을 적용했습니다.
yes/no 응답에 설명이 없더라도 유효한 판정입니다. 빈 응답이나 다른 응답은 오탐으로 간주하지 않습니다.
번역 호출과 언어별 표시를 제거했습니다. 기존 DB 기록은 보존합니다.
DINO 필터, 3초 내 5개 후보, 30초 무감지 종료와 이벤트 통합은 웹 기능으로 유지하며, 원본의 10초/5초 반복 호출 쿨다운으로 되돌리지 않았습니다.
기존 기록 39/40 장면의 재추론은 yes, 비화재라고 제공된 38번 장면도 Yes였습니다. 이 확인은 정확도 평가가 아니며 기존 판정 행을 덮어쓰지 않았습니다.


- 파일 업로드 또는 YouTube 개별 영상 주소로 검사합니다. 주소를 입력하면 주소가 우선합니다.
- 공개 녹화 영상만 가져옵니다. 최대 30분, 500MB, 720p이며 개인 브라우저 쿠키를 사용하지 않습니다.
- 다운로드 중 진행률 및 중지를 지원합니다. 가져오기가 끝나면 기존 YOLO → DINO → VLM 검사로 연결합니다.
- VLM 요청을 순차 대기열로 처리하고 이벤트 ID, 검증 대기/분석 중/결과를 표시합니다.
- 실시간 영상과 저장 영상에 FIRE DETECTED / FALSE ALARM / HUMAN REVIEW REQUIRED / ANALYSIS ERROR를 표시합니다.
- 결과 도착 직후 1초 동안 테두리가 점멸합니다. 기존 사건의 감시/무감지 종료 대기 상태도 표시합니다.
- 영상이 먼저 끝나면 마지막 프레임을 유지하며 요청된 VLM 처리를 기다립니다. 마지막 1초는 최종 결과 확인용 정지 화면이며 과거 프레임을 소급 수정하지 않습니다.
- 결과 파일은 브라우저 재생용 H.264 MP4로 변환합니다. 기존 분석 결과와 같이 음성은 포함하지 않습니다.
- 원본 프롬프트 대신 사용자가 승인한 현재 프롬프트와 3장 입력을 유지합니다. DINO, 이벤트 통합, 접수, 근거/사진/비교 DB 기능도 유지합니다.
- 검증 작업 로그는 이벤트 ID·소요시간·판정·오류를 포함합니다. 실패해도 대기 상태가 영구 유지되지 않도록 처리합니다.
- YouTube 출처는 결과 요약과 `instance/inspection_results/<job_token>_source.json`에 기록합니다.
- YouTube 지원은 [yt-dlp 공식 Python API](https://github.com/yt-dlp/yt-dlp#embedding-yt-dlp)를 사용합니다. 서비스 접근 제한 시 파일 업로드로 검사할 수 있습니다.
