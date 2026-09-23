"""YouTube single-video import; never use the browser's personal cookies."""
import re
import shutil
import time
from pathlib import Path
from urllib.parse import urlsplit, parse_qs

MAX_BYTES = 500 * 1024 * 1024
MAX_SECONDS = 1800


def youtube_url(value):
    url = urlsplit(value.strip())
    if url.scheme not in {'https', 'http'} or url.username or url.password or url.port:
        raise ValueError('올바른 YouTube 영상 주소를 입력해 주세요.')
    host = (url.hostname or '').lower()
    parts = url.path.strip('/').split('/')
    if host == 'youtu.be':
        video_id = parts[0]
    elif host in {'youtube.com', 'www.youtube.com', 'm.youtube.com'}:
        if url.path == '/watch':
            video_id = parse_qs(url.query).get('v', [''])[0]
        elif len(parts) == 2 and parts[0] in {'shorts', 'embed', 'live'}:
            video_id = parts[1]
        else:
            video_id = ''
    else:
        video_id = ''
    if not re.fullmatch(r'[A-Za-z0-9_-]{11}', video_id):
        raise ValueError('YouTube 개별 영상 주소만 사용할 수 있습니다.')
    return 'https://www.youtube.com/watch?v=' + video_id


def download_youtube(job, directory):
    from yt_dlp import YoutubeDL
    started = time.monotonic()

    def check():
        if job.get('stop_requested'):
            raise ValueError('영상 가져오기를 중지했습니다.')
        if time.monotonic() - started > 600:
            raise ValueError('영상 가져오기 제한 시간(10분)을 초과했습니다.')

    def progress(data):
        check()
        size = data.get('downloaded_bytes', 0)
        if size > MAX_BYTES:
            raise ValueError('영상 크기는 최대 500MB입니다.')
        total = data.get('total_bytes') or data.get('total_bytes_estimate') or 0
        job['progress'] = min(99, round(size / total * 100)) if total else 0

    def video_filter(info, *, incomplete=False):
        check()
        if info.get('is_live') or info.get('live_status') == 'is_upcoming':
            return '실시간 방송은 지원하지 않습니다. 녹화된 영상을 사용해 주세요.'
        if (info.get('duration') or 0) > MAX_SECONDS:
            return '테스트 영상은 최대 30분입니다.'
        if (info.get('filesize') or 0) > MAX_BYTES:
            return '영상 크기는 최대 500MB입니다.'

    options = {
        'format': 'best[ext=mp4][height<=720]/bestvideo[ext=mp4][height<=720]/best[height<=720]',
        'outtmpl': str(directory / (job['token'] + '_source.%(ext)s')),
        'noplaylist': True, 'quiet': True, 'no_warnings': True, 'noprogress': True,
        'socket_timeout': 20, 'retries': 2, 'fragment_retries': 2,
        'max_filesize': MAX_BYTES, 'match_filter': video_filter,
        'progress_hooks': [progress], 'cachedir': False,
        'js_runtimes': {'node': {'path': shutil.which('node')}} if shutil.which('node') else {},
    }
    try:
        with YoutubeDL(options) as downloader:
            info = downloader.extract_info(job['source_url'], download=True)
            check()
            if not info:
                raise ValueError('영상을 가져오지 못했습니다. 길이·크기·공개 상태를 확인해 주세요.')
            path = directory / Path(downloader.prepare_filename(info)).name
            if not path.is_file() or path.stat().st_size > MAX_BYTES:
                raise ValueError('영상 다운로드가 완료되지 않았거나 크기 제한을 초과했습니다.')
            job.update(source_path=path, original_name=(info.get('title') or 'YouTube')[:240],
                       status='pending', progress=0)
    except Exception as exc:
        for path in directory.glob(job['token'] + '_source.*'):
            path.unlink(missing_ok=True)
        job.update(status='cancelled' if job.get('stop_requested') else 'error',
                   error='YouTube 영상 가져오기 실패: ' + str(exc)[:350])
