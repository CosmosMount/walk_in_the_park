import io
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import numpy as np
from PIL import Image


class _FrameStore:

    def __init__(self, jpeg_quality: int = 80):
        self._jpeg_quality = jpeg_quality
        self._condition = threading.Condition()
        self._frame_jpeg: Optional[bytes] = None
        self._frame_id = 0
        self._last_step: Optional[int] = None
        self._last_source: Optional[str] = None
        self._last_update_time: Optional[float] = None

    def update(self,
               frame: np.ndarray,
               step: Optional[int] = None,
               source: str = 'train'):
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            raise ValueError(f'Expected HxWx3/4 frame, got {frame.shape}.')

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]

        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer,
                                    format='JPEG',
                                    quality=self._jpeg_quality,
                                    optimize=True)

        with self._condition:
            self._frame_jpeg = buffer.getvalue()
            self._frame_id += 1
            self._last_step = int(step) if step is not None else None
            self._last_source = source
            self._last_update_time = time.time()
            self._condition.notify_all()

    def get_latest(self):
        with self._condition:
            return self._frame_id, self._frame_jpeg

    def wait_for_next(self, last_frame_id: int, timeout: float):
        with self._condition:
            if self._frame_id == last_frame_id:
                self._condition.wait(timeout=timeout)
            return self._frame_id, self._frame_jpeg

    def metadata(self):
        with self._condition:
            return {
                'frame_id': self._frame_id,
                'step': self._last_step,
                'source': self._last_source,
                'updated_at': self._last_update_time,
                'has_frame': self._frame_jpeg is not None,
            }


class RemotePreviewServer:

    def __init__(self,
                 host: str = '0.0.0.0',
                 port: int = 8080,
                 title: str = 'Walk In The Park Preview',
                 stream_fps: float = 10.0,
                 jpeg_quality: int = 80):
        self.host = host
        self.port = port
        self.title = title
        self.stream_fps = stream_fps
        self._frame_store = _FrameStore(jpeg_quality=jpeg_quality)
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        preview_server = self

        class Handler(BaseHTTPRequestHandler):

            def do_GET(self):
                if self.path in ('/', '/index.html'):
                    self._serve_index()
                    return
                if self.path == '/meta':
                    self._serve_meta()
                    return
                if self.path == '/latest.jpg':
                    self._serve_latest_frame()
                    return
                if self.path == '/stream.mjpg':
                    self._serve_stream()
                    return

                self.send_error(HTTPStatus.NOT_FOUND, 'Unknown endpoint.')

            def log_message(self, fmt, *args):
                return

            def _serve_index(self):
                html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{preview_server.title}</title>
  <style>
    body {{
      margin: 0;
      padding: 24px;
      background: #0f172a;
      color: #e2e8f0;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 22px;
    }}
    p {{
      margin: 0 0 12px;
      color: #94a3b8;
    }}
    img {{
      width: min(100%, 960px);
      display: block;
      border: 1px solid #334155;
      border-radius: 12px;
      background: #020617;
    }}
    code {{
      color: #f8fafc;
    }}
  </style>
</head>
<body>
  <h1>{preview_server.title}</h1>
  <p id="meta">Waiting for the first frame.</p>
  <img src="/stream.mjpg" alt="live preview">
  <p>Still image: <code>/latest.jpg</code></p>
  <script>
    async function refreshMeta() {{
      const response = await fetch('/meta');
      const meta = await response.json();
      const status = meta.has_frame
        ? `frame=${{meta.frame_id}} step=${{meta.step}} source=${{meta.source}}`
        : 'Waiting for the first frame.';
      document.getElementById('meta').textContent = status;
    }}
    refreshMeta();
    setInterval(refreshMeta, 1000);
  </script>
</body>
</html>
"""
                data = html.encode('utf-8')
                self.send_response(HTTPStatus.OK)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _serve_meta(self):
                data = json.dumps(preview_server._frame_store.metadata()).encode(
                    'utf-8')
                self.send_response(HTTPStatus.OK)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _serve_latest_frame(self):
                _, frame = preview_server._frame_store.get_latest()
                if frame is None:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE,
                                    'No frame available yet.')
                    return

                self.send_response(HTTPStatus.OK)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(frame)))
                self.end_headers()
                self.wfile.write(frame)

            def _serve_stream(self):
                boundary = b'frame'
                self.send_response(HTTPStatus.OK)
                self.send_header('Age', '0')
                self.send_header('Cache-Control', 'no-cache, private')
                self.send_header('Pragma', 'no-cache')
                self.send_header(
                    'Content-Type',
                    f'multipart/x-mixed-replace; boundary={boundary.decode()}')
                self.end_headers()

                frame_id = -1
                while True:
                    frame_id, frame = preview_server._frame_store.wait_for_next(
                        frame_id, timeout=max(0.05, 1.0 / preview_server.stream_fps))
                    if frame is None:
                        continue

                    try:
                        self.wfile.write(b'--' + boundary + b'\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n')
                        self.wfile.write(
                            f'Content-Length: {len(frame)}\r\n\r\n'.encode(
                                'utf-8'))
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
                        self.wfile.flush()
                    except BrokenPipeError:
                        return
                    except ConnectionResetError:
                        return

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        daemon=True)
        self._thread.start()
        return self

    def update_frame(self,
                     frame: np.ndarray,
                     step: Optional[int] = None,
                     source: str = 'train'):
        self._frame_store.update(frame, step=step, source=source)

    def close(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

