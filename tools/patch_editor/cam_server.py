"""
cam_server.py  —  Headless-friendly HTTP camera-frame server.

Spawns a background ThreadingHTTPServer.  The editor calls
``push_frame(t, path)`` whenever the timeline scrubs; the browser page
auto-updates via SSE (Server-Sent Events) — no polling, no page reload.

SSH port-forward on your local machine
---------------------------------------
  ssh -L 8765:localhost:8765 <user>@<server>

Then open in any browser
------------------------
  http://localhost:8765/

Endpoints
---------
  GET /          — auto-updating viewer page (HTML + EventSource)
  GET /frame     — current cam1 JPEG (204 when none)
  GET /events    — SSE stream  (data: <seq_int>)
  GET /meta      — JSON  {"t": float, "seq": int, "filename": str|null}
"""

from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Shared state (written by editor thread, read by HTTP threads)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_state: dict = {"t": 0.0, "path": None, "seq": 0}

_clients: List[queue.Queue] = []
_clients_lock = threading.Lock()

_server: Optional[ThreadingHTTPServer] = None
_server_thread: Optional[threading.Thread] = None
_port: int = 8765


def push_frame(t: float, frame_path: Optional[Path]) -> None:
    """Call from the editor whenever the current timestamp changes."""
    with _lock:
        _state["t"] = t
        _state["path"] = frame_path
        _state["seq"] += 1
        seq = _state["seq"]
    _broadcast(seq)


def _broadcast(seq: int) -> None:
    msg = f"data: {seq}\n\n"
    encoded = msg.encode()
    dead: List[queue.Queue] = []
    with _clients_lock:
        for q in _clients:
            try:
                q.put_nowait(encoded)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _clients.remove(q)


# ---------------------------------------------------------------------------
# Viewer HTML (single-page, no external deps)
# ---------------------------------------------------------------------------

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Cam1 Feed</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0a0a0a;
      color: #888;
      font: 12px/1.4 monospace;
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }
    #bar {
      flex-shrink: 0;
      padding: 4px 8px;
      background: #111;
      border-bottom: 1px solid #222;
      display: flex;
      gap: 12px;
      align-items: center;
    }
    #status { color: #4a9; }
    #info   { color: #666; }
    #wrap {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }
    #img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      display: block;
    }
    #placeholder {
      color: #333;
      font-size: 14px;
    }
  </style>
</head>
<body>
  <div id="bar">
    <span id="status">connecting\u2026</span>
    <span id="info"></span>
  </div>
  <div id="wrap">
    <img id="img" src="" alt="" style="display:none">
    <div id="placeholder">no frame</div>
  </div>
  <script>
    var seq = -1;
    var img = document.getElementById('img');
    var placeholder = document.getElementById('placeholder');
    var status  = document.getElementById('status');
    var info    = document.getElementById('info');

    var es = new EventSource('/events');

    es.onopen = function() {
      status.textContent = 'connected';
    };

    es.onmessage = function(e) {
      var s = parseInt(e.data, 10);
      if (s === seq) return;
      seq = s;
      // Fetch metadata, then update image
      fetch('/meta').then(function(r) { return r.json(); }).then(function(m) {
        var src = '/frame?s=' + seq;
        img.onload = function() {
          placeholder.style.display = 'none';
          img.style.display = 'block';
        };
        img.onerror = function() {
          img.style.display = 'none';
          placeholder.style.display = '';
          placeholder.textContent = 'frame error';
        };
        img.src = src;
        info.textContent = 't=' + m.t.toFixed(3) + 's' + (m.filename ? '  ' + m.filename : '');
        status.textContent = 'live';
      }).catch(function() {
        status.textContent = 'meta error';
      });
    };

    es.onerror = function() {
      status.textContent = 'disconnected \u2014 reload page';
    };
  </script>
</body>
</html>
""".encode("utf-8")


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass  # suppress per-request console noise

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]
        if path == "/":
            self._respond(200, "text/html; charset=utf-8", _HTML)
        elif path == "/frame":
            self._serve_frame()
        elif path == "/events":
            self._serve_sse()
        elif path == "/meta":
            self._serve_meta()
        else:
            self._respond(404, "text/plain", b"not found")

    # -- helpers -------------------------------------------------------------

    def _respond(self, code: int, ct: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _serve_frame(self):
        with _lock:
            p = _state["path"]
        if p is None:
            self.send_response(204)
            self.end_headers()
            return
        fp = Path(p)
        if not fp.is_file():
            self._respond(404, "text/plain", b"file not found")
            return
        data = fp.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _serve_meta(self):
        with _lock:
            t   = _state["t"]
            p   = _state["path"]
            seq = _state["seq"]
        fn = Path(p).name if p else None
        body = json.dumps({"t": t, "seq": seq, "filename": fn}).encode()
        self._respond(200, "application/json", body)

    def _serve_sse(self):
        q: queue.Queue = queue.Queue(maxsize=16)
        with _clients_lock:
            _clients.append(q)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")   # disable nginx buffering
        self.end_headers()

        # Initial heartbeat so EventSource knows the connection is alive
        try:
            self.wfile.write(b": ping\n\n")
            self.wfile.flush()
        except OSError:
            with _clients_lock:
                if q in _clients:
                    _clients.remove(q)
            return

        while True:
            try:
                msg: bytes = q.get(timeout=20)
                self.wfile.write(msg)
                self.wfile.flush()
            except queue.Empty:
                # Send a keep-alive comment so the connection doesn't time out
                try:
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
                except OSError:
                    break
            except OSError:
                break

        with _clients_lock:
            if q in _clients:
                _clients.remove(q)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def start(port: int = 8765) -> int:
    """
    Start the HTTP server in a background daemon thread.
    Returns the actual port bound (useful if port=0 for OS-assigned).
    Safe to call multiple times; subsequent calls are no-ops.
    """
    global _server, _server_thread, _port
    if _server is not None:
        return _port
    srv = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    _port = srv.server_address[1]
    _server = srv
    _server_thread = threading.Thread(target=srv.serve_forever, daemon=True, name="cam-server")
    _server_thread.start()
    return _port


def stop() -> None:
    """Shut down the server (called on editor exit, optional)."""
    global _server
    if _server is not None:
        _server.shutdown()
        _server = None


def url() -> str:
    """Return the viewer URL."""
    return f"http://localhost:{_port}/"
