"""
web_main.py  —  Browser-based patch editor (headless-server edition).

Single HTTP port serves everything: interactive map editor + camera feed.
No Qt, no VNC, no X11 required.

Usage
-----
  # Single scenario (existing):
  python3 -m tools.patch_editor.web_main  <html_file>  [--port 8800]

  # Batch mode (new):
  python3 -m tools.patch_editor.web_main  <scenario_folder>  --port 8800 \\
      --carla-host localhost --carla-port 2000 \\
      --routes-out /path/to/xml_output

On your local machine, SSH-forward the single port:
  ssh -L 8800:localhost:8800 <user>@<server>

Then open:
  http://localhost:8800/
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from .loader import SceneData, load_from_html
from .patch_model import PatchModel   # only used to reuse patch_path_for()


# ---------------------------------------------------------------------------
# Global state (written at startup, read by HTTP threads)
# ---------------------------------------------------------------------------

_scene_data: Optional[SceneData] = None
_patch: dict = {"overrides": []}
_patch_lock = threading.Lock()
_patch_path: Optional[Path] = None

# Camera: list of (subdir_name, sorted_frame_nums, frame_paths)
_cam_subdirs: List[Tuple[str, List[int], Dict[int, Path]]] = []
_cam_t_min: float = 0.0
_cam_dt: float = 0.1


# ---------------------------------------------------------------------------
# Batch mode state
# ---------------------------------------------------------------------------

class BatchState:
    """Manages multi-scenario editing workflow."""

    def __init__(self) -> None:
        self.enabled: bool = False
        self.scenario_dirs: List[Path] = []
        self.current_idx: int = 0
        self.preloaded: Dict[int, dict] = {}   # idx -> {scene_data, cam_subdirs, ...}
        self.lock = threading.Lock()
        self.preload_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="preload")
        self.pipeline_args: Optional[argparse.Namespace] = None
        self.routes_out_dir: Optional[Path] = None
        # CARLA connection info
        self.carla_host: str = "localhost"
        self.carla_port: int = 2000
        # Auto-export pipeline stage toggles
        self.skip_eval: bool = False
        self.skip_grp_simplify: bool = False
        # Serial CARLA export queue
        self.carla_queue: List[Dict[str, Any]] = []   # ordered queue items
        self.carla_queue_lock = threading.Lock()
        self._carla_worker_thread: Optional[threading.Thread] = None

    def scenario_name(self, idx: int) -> str:
        if 0 <= idx < len(self.scenario_dirs):
            return self.scenario_dirs[idx].name
        return "?"

    def status_dict(self) -> dict:
        # Build a fast lookup: scenario name → queue item
        with self.carla_queue_lock:
            queue_by_name = {q["scenario"]: q for q in self.carla_queue}
            queue_snapshot = list(self.carla_queue)

        def _scenario_entry(i: int, d: Path) -> dict:
            html = d / "trajectory_plot.html"
            has_html = html.exists()
            has_patch = False
            if has_html:
                try:
                    pp = PatchModel.patch_path_for(html)
                    has_patch = pp.exists() and pp.stat().st_size > 10
                except Exception:
                    pass
            qi = queue_by_name.get(d.name)
            return {
                "idx": i,
                "name": d.name,
                "has_html": has_html,
                "has_patch": has_patch,
                "queue_status": qi["status"] if qi else None,
                "queue_stage":  qi["stage"]  if qi else None,
                "queue_error":  qi.get("error") if qi else None,
            }

        return {
            "enabled": self.enabled,
            "total": len(self.scenario_dirs),
            "current": self.current_idx,
            "current_name": self.scenario_name(self.current_idx),
            "scenario_list": [_scenario_entry(i, d)
                              for i, d in enumerate(self.scenario_dirs)],
            "carla_queue": queue_snapshot,
        }


_batch = BatchState()


# ---------------------------------------------------------------------------
# CARLA health check
# ---------------------------------------------------------------------------

_carla_status: Dict[str, Any] = {
    "connected": False,
    "host": "localhost",
    "port": 2000,
    "last_check": 0,
    "error": None,
}
_carla_status_lock = threading.Lock()


def _ensure_carla_egg() -> None:
    """Add the CARLA .egg to sys.path so ``import carla`` works."""
    # Try well-known locations relative to the workspace root
    workspace = Path(__file__).resolve().parents[2]
    candidates = [
        workspace / "carla912" / "PythonAPI" / "carla" / "dist",
        workspace / "carla" / "PythonAPI" / "carla" / "dist",
        Path("/opt/carla/PythonAPI/carla/dist"),
    ]
    # Also honour CARLA_ROOT env var
    cr = os.environ.get("CARLA_ROOT")
    if cr:
        candidates.insert(0, Path(cr) / "PythonAPI" / "carla" / "dist")

    for dist in candidates:
        if not dist.is_dir():
            continue
        # Prefer py3 eggs over py2
        eggs = sorted(dist.glob("carla-*.egg"), reverse=True) + \
               sorted(dist.glob("carla-*.whl"), reverse=True)
        py3 = [e for e in eggs if "py3" in e.name or "py3" in str(e)]
        for egg in (py3 or eggs):
            if str(egg) not in sys.path:
                sys.path.append(str(egg))
                print(f"[CARLA] Added {egg} to sys.path", flush=True)
            return
    print("[CARLA] Warning: could not find carla .egg — set CARLA_ROOT", flush=True)


def _find_carla_python() -> Optional[str]:
    """Find a Python interpreter that can ``import carla`` successfully."""
    workspace = Path(__file__).resolve().parents[2]
    # Candidates: conda envs known to have py3.7 + egg support, then generic
    candidates = [
        "/data/miniconda3/envs/colmdrivermarco2/bin/python",
        "/data/miniconda3/envs/run_custom_eval_baseline/bin/python",
        "/data/miniconda3/envs/carla_dataset_utils/bin/python",
    ]
    cr = os.environ.get("CARLA_ROOT")
    egg_path = str(workspace / "carla912" / "PythonAPI" / "carla" / "dist" /
                    "carla-0.9.12-py3.7-linux-x86_64.egg")

    for py in candidates:
        if not Path(py).exists():
            continue
        try:
            proc = subprocess.run(
                [py, "-c",
                 f"import sys; sys.path.append('{egg_path}'); import carla; "
                 f"print('ok')"],
                capture_output=True, text=True, timeout=10,
            )
            if proc.returncode == 0 and "ok" in proc.stdout:
                print(f"[CARLA] Using {py} for carla health checks", flush=True)
                return py
        except Exception:
            pass
    return None


# Cached Python interpreter path for carla subprocess calls
_carla_python: Optional[str] = None
_carla_python_searched = False


def _check_carla_health(host: str, port: int) -> Dict[str, Any]:
    """Probe CARLA server connectivity.  Returns status dict."""
    global _carla_python, _carla_python_searched

    # The CARLA 0.9.12 egg is built for CPython 3.7.  Loading its native
    # extension under a newer interpreter (3.8+) causes a hard segfault —
    # not a catchable exception.  So we ALWAYS use the subprocess path
    # unless ``import carla`` already works in-process *without* adding
    # any egg to sys.path (i.e. pip-installed or env already set up).
    carla_available_inproc = False
    try:
        import carla  # type: ignore
        carla_available_inproc = True
    except ModuleNotFoundError:
        pass

    if carla_available_inproc:
        try:
            client = carla.Client(host, port)
            client.set_timeout(5.0)
            world = client.get_world()
            map_name = world.get_map().name if world else "?"
            return {
                "connected": True,
                "host": host,
                "port": port,
                "map": map_name,
                "error": None,
                "last_check": time.time(),
            }
        except Exception as exc:
            return {
                "connected": False,
                "host": host,
                "port": port,
                "map": None,
                "error": str(exc),
                "last_check": time.time(),
            }

    # Fall back to subprocess with a compatible Python interpreter.
    if not _carla_python_searched:
        _carla_python = _find_carla_python()
        _carla_python_searched = True

    if _carla_python:
        return _check_carla_health_subprocess(host, port)

    return {
            "connected": False,
            "host": host,
            "port": port,
            "map": None,
            "error": str(in_proc_exc),
            "last_check": time.time(),
        }


def _check_carla_health_subprocess(host: str, port: int) -> Dict[str, Any]:
    """Probe CARLA via a subprocess using a compatible Python interpreter."""
    workspace = Path(__file__).resolve().parents[2]
    egg_path = str(workspace / "carla912" / "PythonAPI" / "carla" / "dist" /
                    "carla-0.9.12-py3.7-linux-x86_64.egg")
    script = (
        f"import sys; sys.path.insert(0,'{egg_path}'); import carla,json; "
        f"c=carla.Client('{host}',{port}); c.set_timeout(5.0); "
        f"w=c.get_world(); m=w.get_map().name if w else '?'; "
        f"print(json.dumps({{'connected':True,'map':m}}))"
    )
    try:
        proc = subprocess.run(
            [_carla_python, "-c", script],   # type: ignore[arg-type]
            capture_output=True, text=True, timeout=15,
        )
        if proc.returncode == 0:
            result = json.loads(proc.stdout.strip().split("\n")[-1])
            return {
                "connected": True,
                "host": host,
                "port": port,
                "map": result.get("map", "?"),
                "error": None,
                "last_check": time.time(),
            }
        else:
            err = proc.stderr.strip().split("\n")[-1] if proc.stderr else "unknown"
            return {
                "connected": False,
                "host": host,
                "port": port,
                "map": None,
                "error": err,
                "last_check": time.time(),
            }
    except subprocess.TimeoutExpired:
        return {
            "connected": False,
            "host": host,
            "port": port,
            "map": None,
            "error": "timeout connecting to CARLA",
            "last_check": time.time(),
        }
    except Exception as exc:
        return {
            "connected": False,
            "host": host,
            "port": port,
            "map": None,
            "error": str(exc),
            "last_check": time.time(),
        }


def _carla_health_loop() -> None:
    """Background thread: periodically probe CARLA."""
    while True:
        host = _batch.carla_host
        port = _batch.carla_port
        result = _check_carla_health(host, port)
        with _carla_status_lock:
            _carla_status.update(result)
        time.sleep(8)   # check every 8 seconds


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

def _run_pipeline_for_scenario(scenario_dir: Path) -> Optional[Path]:
    """
    Run the V2XPNP pipeline on a scenario directory to produce trajectory_plot.html.
    Returns the path to the generated HTML, or None on failure.
    """
    html_path = scenario_dir / "trajectory_plot.html"

    # If HTML already exists, use it
    if html_path.exists():
        print(f"[BATCH] Using existing HTML for {scenario_dir.name}", flush=True)
        return html_path

    # Run the pipeline as a subprocess to isolate state
    print(f"[BATCH] Running pipeline for {scenario_dir.name} ...", flush=True)
    cmd = [
        sys.executable, "-m", "v2xpnp.pipeline.entrypoint",
        str(scenario_dir),
    ]
    # Forward pipeline args if available
    if _batch.pipeline_args:
        pa = _batch.pipeline_args
        if pa.map_pkl:
            for mp in pa.map_pkl:
                cmd.extend(["--map-pkl", mp])
        if pa.carla_map_cache:
            cmd.extend(["--carla-map-cache", pa.carla_map_cache])
        if pa.carla_map_offset_json:
            cmd.extend(["--carla-map-offset-json", pa.carla_map_offset_json])

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,   # 5-minute timeout
            cwd=str(Path(__file__).resolve().parents[2]),  # workspace root
        )
        if proc.returncode != 0:
            print(f"[BATCH] Pipeline failed for {scenario_dir.name}:\n{proc.stderr[-500:]}", flush=True)
            return None
        if html_path.exists():
            print(f"[BATCH] Pipeline produced HTML for {scenario_dir.name}", flush=True)
            return html_path
        else:
            print(f"[BATCH] Pipeline ran but no HTML at {html_path}", flush=True)
            return None
    except subprocess.TimeoutExpired:
        print(f"[BATCH] Pipeline timed out for {scenario_dir.name}", flush=True)
        return None
    except Exception as exc:
        print(f"[BATCH] Pipeline error for {scenario_dir.name}: {exc}", flush=True)
        return None


def _load_scenario_data(scenario_dir: Path) -> Optional[dict]:
    """
    Load (or generate then load) a scenario's data for the editor.
    Returns dict with scene_data, cam info, patch data, etc.
    """
    html_path = scenario_dir / "trajectory_plot.html"

    # Generate HTML if needed
    if not html_path.exists():
        html_path = _run_pipeline_for_scenario(scenario_dir)
        if html_path is None:
            return None

    try:
        scene_data = load_from_html(html_path)
    except Exception as exc:
        print(f"[BATCH] Failed to load HTML for {scenario_dir.name}: {exc}", flush=True)
        return None

    # Discover camera frames
    cam_subdirs: List[Tuple[str, List[int], Dict[int, Path]]] = []
    cam_t_min = 0.0
    cam_dt = 0.1

    try:
        for child in sorted(scenario_dir.iterdir()):
            if not child.is_dir():
                continue
            fps: Dict[int, Path] = {}
            for img in child.glob("*_cam1.jpeg"):
                try:
                    fps[int(img.stem.split("_")[0])] = img
                except ValueError:
                    pass
            if not fps:
                for img in child.glob("*_cam1.jpg"):
                    try:
                        fps[int(img.stem.split("_")[0])] = img
                    except ValueError:
                        pass
            if fps:
                try:
                    v = int(child.name)
                    if v > 0:
                        nums = sorted(fps.keys())
                        cam_subdirs.append((child.name, nums, fps))
                except ValueError:
                    pass
        cam_subdirs.sort(key=lambda x: int(x[0]))
    except Exception:
        pass

    ego = next((t for t in scene_data.tracks if t.role == "ego"), None)
    if ego and len(ego.frames) >= 2:
        dt = ego.frames[1].t - ego.frames[0].t
        if 0.01 < dt < 10.0:
            cam_dt = dt
        cam_t_min = ego.frames[0].t

    # Load existing patch
    patch_path = PatchModel.patch_path_for(html_path)
    patch_data: dict = {"overrides": []}
    if patch_path.exists():
        try:
            patch_data = json.loads(patch_path.read_text())
        except Exception:
            pass

    return {
        "scene_data": scene_data,
        "html_path": html_path,
        "patch_path": patch_path,
        "patch_data": patch_data,
        "cam_subdirs": cam_subdirs,
        "cam_t_min": cam_t_min,
        "cam_dt": cam_dt,
        "scenario_name": scenario_dir.name,
    }


def _preload_next(idx: int) -> None:
    """Preload scenario at `idx` in a background thread."""
    if idx < 0 or idx >= len(_batch.scenario_dirs):
        return
    if idx in _batch.preloaded:
        return  # already loaded
    scenario_dir = _batch.scenario_dirs[idx]
    print(f"[BATCH] Preloading scenario {idx}: {scenario_dir.name} ...", flush=True)
    result = _load_scenario_data(scenario_dir)
    if result:
        with _batch.lock:
            _batch.preloaded[idx] = result
        print(f"[BATCH] Preloaded scenario {idx}: {scenario_dir.name}", flush=True)
    else:
        print(f"[BATCH] Failed to preload scenario {idx}: {scenario_dir.name}", flush=True)


def _activate_scenario(idx: int) -> bool:
    """Switch the active scenario to index `idx`.  Returns True on success."""
    global _scene_data, _patch, _patch_path, _cam_subdirs, _cam_t_min, _cam_dt

    with _batch.lock:
        data = _batch.preloaded.get(idx)

    if data is None:
        # Not preloaded — load synchronously
        data = _load_scenario_data(_batch.scenario_dirs[idx])
        if data is None:
            return False

    _scene_data = data["scene_data"]
    _cam_subdirs = data["cam_subdirs"]
    _cam_t_min = data["cam_t_min"]
    _cam_dt = data["cam_dt"]

    with _patch_lock:
        _patch = data["patch_data"]
    _patch_path = data["patch_path"]

    _batch.current_idx = idx

    # Kick off preload of next scenario
    next_idx = idx + 1
    if next_idx < len(_batch.scenario_dirs) and next_idx not in _batch.preloaded:
        _batch.preload_executor.submit(_preload_next, next_idx)

    print(f"[BATCH] Activated scenario {idx}: {data['scenario_name']}", flush=True)
    return True


# ---------------------------------------------------------------------------
# Serial CARLA export queue
# ---------------------------------------------------------------------------

def _carla_queue_worker() -> None:
    """Daemon thread: process CARLA export queue one item at a time."""
    while True:
        # Grab the next pending item under the lock, mark it running
        with _batch.carla_queue_lock:
            item = next((q for q in _batch.carla_queue if q["status"] == "pending"), None)
            if item is None:
                _batch._carla_worker_thread = None
                return
            item["status"] = "running"
            item["stage"] = "starting"

        scenario_dir = Path(item["scenario_dir"])
        patch_data   = item["patch_data"]
        print(f"[QUEUE] Starting CARLA export for {item['scenario']}", flush=True)
        try:
            _run_background_export_and_eval(
                scenario_dir, patch_data, _batch.routes_out_dir, _progress_item=item)
            item["status"] = "done"
            item["stage"]  = "complete"
            print(f"[QUEUE] Done: {item['scenario']}", flush=True)
        except Exception as exc:
            item["status"] = "error"
            item["error"]  = str(exc)
            print(f"[QUEUE] Error: {item['scenario']}: {exc}", flush=True)


def _enqueue_for_carla(scenario_dir: Path, patch_data: dict) -> dict:
    """
    Add scenario to the serial CARLA queue.
    - If already pending: update patch_data in-place (latest save wins).
    - If running: leave running, note that patch was re-saved separately.
    - If done/error: re-enqueue for re-processing.
    Returns a status dict with 'action': 'updated'|'running'|'queued'.
    """
    if not _batch.routes_out_dir:
        return {"action": "skipped", "reason": "no routes_out_dir"}

    name = scenario_dir.name
    with _batch.carla_queue_lock:
        existing = next((q for q in _batch.carla_queue if q["scenario"] == name), None)

        if existing is not None:
            if existing["status"] == "running":
                # Can't modify a running job; patch file is already saved to disk
                return {"action": "running"}
            elif existing["status"] == "pending":
                # Update patch data with latest save
                existing["patch_data"] = patch_data
                existing["queued_at"]  = time.time()
                print(f"[QUEUE] Updated pending item: {name}", flush=True)
                return {"action": "updated"}
            else:
                # done or error — remove and re-add below
                _batch.carla_queue.remove(existing)

        item: Dict[str, Any] = {
            "scenario":     name,
            "scenario_dir": str(scenario_dir),
            "patch_data":   patch_data,
            "status":       "pending",
            "stage":        "queued",
            "error":        None,
            "queued_at":    time.time(),
        }
        _batch.carla_queue.append(item)
        print(f"[QUEUE] Enqueued: {name}", flush=True)

        # Start the worker thread if it isn't alive
        if (_batch._carla_worker_thread is None
                or not _batch._carla_worker_thread.is_alive()):
            t = threading.Thread(target=_carla_queue_worker, daemon=True,
                                 name="carla-queue-worker")
            _batch._carla_worker_thread = t
            t.start()

    return {"action": "queued"}


def _dequeue_for_carla(scenario_name: str) -> dict:
    """Remove a pending item from the CARLA queue. Running/done items are left."""
    with _batch.carla_queue_lock:
        item = next((q for q in _batch.carla_queue
                     if q["scenario"] == scenario_name and q["status"] == "pending"), None)
        if item is None:
            return {"removed": False, "reason": "not found or not pending"}
        _batch.carla_queue.remove(item)
    print(f"[QUEUE] Removed: {scenario_name}", flush=True)
    return {"removed": True}


# ---------------------------------------------------------------------------
# Background tasks: XML route export + eval
# ---------------------------------------------------------------------------

def _load_raw_dataset_from_html(html_path: Path) -> dict:
    """Load the full raw dataset dict from the HTML (not just SceneData)."""
    import re as _re
    text = html_path.read_text(encoding="utf-8", errors="replace")
    m = _re.search(
        r'<script[^>]+id="dataset"[^>]*type="application/json"[^>]*>(.*?)</script>',
        text, _re.DOTALL | _re.IGNORECASE,
    )
    if not m:
        m = _re.search(r'<script[^>]*>(.*?"scenario_name".*?)</script>', text, _re.DOTALL)
    if not m:
        raise ValueError(f"No embedded dataset found in {html_path}")
    return json.loads(m.group(1))


def _run_background_export_and_eval(scenario_dir: Path, patch_data: dict,
                                     routes_out: Path,
                                     _progress_item: Optional[Dict[str, Any]] = None) -> None:
    """
    Background task: export CARLA routes directly from the already-processed
    dataset (no redundant pipeline re-run), then ground-align + eval.

    _progress_item: if provided (a carla_queue entry dict), sub-stage updates
    are written to it so the UI can show fine-grained progress.
    """
    scenario_name = scenario_dir.name
    t0_total = time.time()

    def _set_stage(stage: str):
        if _progress_item is not None:
            _progress_item["stage"] = stage

    def _elapsed(t0):
        return f"{time.time()-t0:.1f}s"

    def _run_subprocess(cmd, label, timeout=600, env=None):
        """Run a subprocess, log output, raise on failure."""
        t0 = time.time()
        print(f"[BG] {label} for {scenario_name} ...", flush=True)
        print(f"[BG]   cmd: {' '.join(cmd)}", flush=True)
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
                cwd=str(Path(__file__).resolve().parents[2]),
                env=env,
            )
        except subprocess.TimeoutExpired:
            elapsed = _elapsed(t0)
            print(f"[BG]   TIMED OUT after {timeout}s ({elapsed})", flush=True)
            raise RuntimeError(f"{label} timed out after {timeout}s")
        elapsed = _elapsed(t0)
        stdout_tail = proc.stdout.strip().split('\n')
        stderr_tail = proc.stderr.strip().split('\n')
        if stdout_tail and stdout_tail[-1]:
            for line in stdout_tail[-5:]:
                if line.strip():
                    print(f"[BG]   {line}", flush=True)
        if proc.returncode != 0:
            print(f"[BG]   FAILED ({elapsed}, exit {proc.returncode})", flush=True)
            for line in stderr_tail[-8:]:
                if line.strip():
                    print(f"[BG]   stderr: {line}", flush=True)
            raise RuntimeError(f"{label} failed (exit {proc.returncode}):\n"
                              + '\n'.join(stderr_tail[-5:]))
        print(f"[BG]   OK ({elapsed})", flush=True)
        return proc

    try:
        # 1. Save the patch
        patch_path = PatchModel.patch_path_for(scenario_dir / "trajectory_plot.html")
        patch_path.write_text(json.dumps(patch_data, indent=2))
        n_overrides = sum(1 for ov in (patch_data.get("overrides") or [])
                         if any(ov.get(k) for k in ("delete","snap_to_outermost",
                                "phase_override","lane_segment_overrides","waypoint_overrides")))
        print(f"[BG] === Starting export for {scenario_name} "
              f"({n_overrides} patches) ===", flush=True)

        routes_dir = routes_out / scenario_name
        routes_dir.mkdir(parents=True, exist_ok=True)

        # 2. Load dataset from existing HTML + apply patch (no pipeline re-run)
        _set_stage("load_and_patch")
        t0 = time.time()

        html_path = scenario_dir / "trajectory_plot.html"
        if not html_path.exists():
            raise FileNotFoundError(f"HTML not found: {html_path}")

        dataset = _load_raw_dataset_from_html(html_path)
        print(f"[BG]   loaded dataset: {len(dataset.get('tracks', []))} tracks", flush=True)

        # Apply patch to dataset in-process
        from tools.patch_editor.patch_apply import apply_patch_to_dataset
        n_applied = apply_patch_to_dataset(dataset, patch_data, verbose=True)
        print(f"[BG]   applied {n_applied} patch override(s) ({_elapsed(t0)})", flush=True)

        # 3. Export CARLA routes directly (no subprocess)
        _set_stage("route_export")
        t0 = time.time()

        from v2xpnp.pipeline.route_export_stage_04_orchestration import export_carla_routes
        from v2xpnp.pipeline.runtime_orchestration import _split_tracks_for_carla_export

        # Split tracks into ego/actor (handles model defaults, parked detection, etc.)
        ego_tracks, actor_tracks = _split_tracks_for_carla_export(dataset)

        # Get align_cfg and town from dataset
        carla_map = dataset.get("carla_map") or {}
        align_cfg = dict(carla_map.get("align_cfg") or {})
        lane_corr_meta = dataset.get("lane_correspondence")
        if not isinstance(lane_corr_meta, dict):
            lane_corr_meta = ((dataset.get("metadata") or {}).get("lane_correspondence")
                              if isinstance(dataset.get("metadata"), dict) else None)
        if isinstance(lane_corr_meta, dict):
            icp_refine = lane_corr_meta.get("icp_refine")
            if isinstance(icp_refine, dict):
                align_cfg["lane_corr_icp_refine"] = dict(icp_refine)
        # Town name from carla_map.name (e.g. "Carla/Maps/ucla_v2/ucla_v2" → "ucla_v2")
        map_name = str(carla_map.get("name") or "unknown")
        town = map_name.rsplit("/", 1)[-1] if "/" in map_name else map_name

        print(f"[BG]   exporting routes: {len(ego_tracks)} ego, {len(actor_tracks)} actors, "
              f"town={town}, align_cfg={'identity' if not align_cfg else 'custom'}", flush=True)

        report = export_carla_routes(
            out_dir=routes_dir,
            town=town,
            route_id=scenario_name,
            ego_tracks=ego_tracks,
            actor_tracks=actor_tracks,
            align_cfg=align_cfg,
            ego_path_source="auto",
            actor_control_mode="replay",
            walker_control_mode="replay",
            encode_timing=True,
            snap_to_road=False,
            static_spawn_only=False,
            default_dt=0.1,
        )

        xml_files = sorted(routes_dir.glob("*.xml"))
        print(f"[BG]   exported {len(xml_files)} XML file(s): "
              f"{', '.join(f.name for f in xml_files[:5])}"
              f"{'...' if len(xml_files)>5 else ''} ({_elapsed(t0)})", flush=True)

        # Setup CARLA subprocess environment (shared by steps 3a, 3.5, 4, 5)
        carla_py = _find_carla_python() or sys.executable
        workspace = Path(__file__).resolve().parents[2]
        egg = workspace / "carla912" / "PythonAPI" / "carla" / "dist" / \
            "carla-0.9.12-py3.7-linux-x86_64.egg"
        # Include PythonAPI/carla so `agents.navigation.*` is importable for GRP.
        carla_pkg = workspace / "carla912" / "PythonAPI" / "carla"
        py_paths = [str(egg), str(carla_pkg), str(workspace), str(workspace / "tools")]
        existing_py = os.environ.get("PYTHONPATH", "")
        if existing_py:
            py_paths.append(existing_py)
        carla_env = {**os.environ, "PYTHONPATH": os.pathsep.join(py_paths)}

        # 3a. Rename ego XMLs to _REPLAY so GRP-simplified version takes the standard name
        replay_ego_files = []
        for f in sorted(routes_dir.glob("*_ego_vehicle_*.xml")):
            if "_REPLAY" not in f.stem:
                replay_path = f.with_name(f.stem + "_REPLAY" + f.suffix)
                f.rename(replay_path)
                replay_ego_files.append(replay_path)
        print(f"[BG]   renamed {len(replay_ego_files)} ego XML(s) to _REPLAY", flush=True)

        # 3.5. GRP ego simplification — reads *_REPLAY.xml, writes simplified ego XML (no time/speed)
        if _batch.skip_grp_simplify:
            print("[BG]   GRP simplification SKIPPED (--no-grp-simplify)", flush=True)
        else:
            _set_stage("grp_simplify")
            cmd_grp = [
                carla_py,
                str(workspace / "tools" / "ego_grp_simplify.py"),
                str(routes_dir),
                "--carla-host", _batch.carla_host,
                "--carla-port", str(_batch.carla_port),
            ]
            try:
                _run_subprocess(cmd_grp, "GRP ego simplification", timeout=120, env=carla_env)
            except Exception as e:
                print(f"[BG]   GRP simplification non-fatal: {e}", flush=True)

        # 3.6. GRP fallback — if GRP didn't create the non-REPLAY ego XML, copy
        #      from _REPLAY so downstream steps (ground alignment, etc.) that
        #      reference the manifest filename still work.
        import shutil as _shutil
        for replay_f in replay_ego_files:
            non_replay_name = replay_f.stem.replace("_REPLAY", "") + replay_f.suffix
            non_replay_path = replay_f.with_name(non_replay_name)
            if not non_replay_path.exists():
                _shutil.copy2(replay_f, non_replay_path)
                print(f"[BG]   GRP fallback: copied {replay_f.name} → {non_replay_path.name}", flush=True)

        # 4. CARLA ground alignment (ray casting) — subprocess (needs CARLA server)
        _set_stage("ground_align")
        ground_report_path = routes_dir / "ground_alignment_report.json"
        cmd_align = [
            carla_py, "-m", "v2xpnp.pipeline.carla_ground_align",
            str(routes_dir),
            "--carla-host", _batch.carla_host,
            "--carla-port", str(_batch.carla_port),
            "--report-json", str(ground_report_path),
            "--verbose",
        ]
        try:
            _run_subprocess(cmd_align, "CARLA ground alignment", timeout=900, env=carla_env)
            if ground_report_path.exists():
                try:
                    ga_report = json.loads(ground_report_path.read_text(encoding="utf-8"))
                    nz_wp = int(ga_report.get("total_nonzero_tilt_waypoints", 0))
                    nz_files = int(ga_report.get("files_with_nonzero_tilt", 0))
                    total_files = int(ga_report.get("files_processed", 0))
                    print(
                        f"[BG]   ground-align tilt summary: {nz_wp} nonzero-tilt waypoints "
                        f"across {nz_files}/{total_files} files",
                        flush=True,
                    )
                    if nz_wp == 0:
                        print(
                            "[BG]   WARNING: ground-align produced zero nonzero-tilt waypoints; "
                            "verify CARLA map surface sampling for this scenario",
                            flush=True,
                        )
                except Exception as exc:
                    print(f"[BG]   WARNING: failed to parse ground alignment report: {exc}", flush=True)
        except Exception as e:
            print(f"[BG]   Ground alignment non-fatal: {e}", flush=True)

        # 5. Custom eval (subprocess — separate process with CARLA connection)
        if _batch.skip_eval:
            print("[BG]   Custom eval SKIPPED (--no-eval)", flush=True)
        else:
            _set_stage("custom_eval")
            cmd_eval = [
                carla_py, "tools/run_custom_eval.py",
                "--routes-dir", str(routes_dir),
                "--port", str(_batch.carla_port),
                "--planner", "log-replay",
                "--npc-only-fake-ego",
                "--custom-actor-control-mode", "replay",
                "--log-replay-actors",
                "--capture-logreplay-images",
            ]
            _run_subprocess(cmd_eval, "Custom eval", timeout=600, env=carla_env)

        print(f"[BG] === Export COMPLETE for {scenario_name} "
              f"(total {_elapsed(t0_total)}) ===", flush=True)

    except Exception as exc:
        print(f"[BG] === Export FAILED for {scenario_name} "
              f"(total {_elapsed(t0_total)}): {exc} ===", flush=True)
        traceback.print_exc()
        raise


# ---------------------------------------------------------------------------
# Data serialisation
# ---------------------------------------------------------------------------

def _jf(v) -> Optional[float]:
    """Return float v, or None if NaN/Inf (JSON can't encode those)."""
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _serialise_scene(scene: SceneData) -> dict:
    tracks_out = []
    for t in scene.tracks:
        frames_out = [
            {
                "t":    _jf(f.t),
                "x":    _jf(f.x),    "y":    _jf(f.y),
                "cx":   _jf(f.cx),   "cy":   _jf(f.cy),   "cyaw": _jf(f.cyaw),
                "ccli": int(f.ccli) if f.ccli is not None else -1,
            }
            for f in t.frames
        ]
        tracks_out.append({
            "id":      t.track_id,
            "role":    t.role,
            "obj_type": t.obj_type,
            "t_start": t.t_start,
            "t_end":   t.t_end,
            "frames":  frames_out,
            "ccli_changes": t.ccli_change_times(),
        })

    lines_out = [
        {
            "idx":   l.idx,
            "label": l.label,
            "pts":   [[float(p[0]), float(p[1])] for p in l.pts],
            "road_id": l.road_id,
            "lane_id": l.lane_id,
            "dir_sign": l.dir_sign,
        }
        for l in scene.carla_lines
    ]

    out: Dict[str, Any] = {
        "scenario_name": scene.scenario_name,
        "tracks":        tracks_out,
        "carla_lines":   lines_out,
    }
    if scene.bg_image_b64 and scene.bg_image_bounds:
        out["bg_image"] = {
            "b64":    scene.bg_image_b64,
            "bounds": scene.bg_image_bounds,
        }
    return out


# ---------------------------------------------------------------------------
# Camera frame discovery
# ---------------------------------------------------------------------------

def _discover_cam(html_path: Path, scene: SceneData) -> None:
    global _cam_subdirs, _cam_t_min, _cam_dt

    scenario_dir = html_path.parent
    candidates: List[Tuple[int, str, List[int], Dict[int, Path]]] = []

    try:
        for child in sorted(scenario_dir.iterdir()):
            if not child.is_dir():
                continue
            fps: Dict[int, Path] = {}
            for img in child.glob("*_cam1.jpeg"):
                try:
                    fps[int(img.stem.split("_")[0])] = img
                except ValueError:
                    pass
            if not fps:
                for img in child.glob("*_cam1.jpg"):
                    try:
                        fps[int(img.stem.split("_")[0])] = img
                    except ValueError:
                        pass
            if fps:
                nums = sorted(fps.keys())
                candidates.append((len(nums), child.name, nums, fps))
    except Exception:
        pass

    # Keep only positive-integer-named subdirs (1, 2, 3…) sorted by number.
    # Negative folders (-1, -2…) are infrastructure cameras, not ego cameras.
    def _pos_int(name: str) -> Optional[int]:
        try:
            v = int(name)
            return v if v > 0 else None
        except ValueError:
            return None

    candidates = [(name, nums, fp) for _, name, nums, fp in candidates if _pos_int(name) is not None]
    candidates.sort(key=lambda x: int(x[0]))
    _cam_subdirs = candidates

    # Infer dt / t_min from ego track
    ego = next((t for t in scene.tracks if t.role == "ego"), None)
    if ego and len(ego.frames) >= 2:
        dt = ego.frames[1].t - ego.frames[0].t
        if 0.01 < dt < 10.0:
            _cam_dt = dt
        _cam_t_min = ego.frames[0].t


def _frame_for_t(t: float, cam_idx: int) -> Optional[Path]:
    if cam_idx < 0 or cam_idx >= len(_cam_subdirs):
        return None
    _, nums, fps = _cam_subdirs[cam_idx]
    if not nums:
        return None
    target = int(round((t - _cam_t_min) / _cam_dt))
    pos = bisect.bisect_left(nums, target)
    cands = []
    if pos < len(nums):
        cands.append(nums[pos])
    if pos > 0:
        cands.append(nums[pos - 1])
    nearest = min(cands, key=lambda n: abs(n - target))
    if abs(nearest - target) > 3:
        return None
    return fps[nearest]


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    # ── routing ────────────────────────────────────────────────────────────

    def do_GET(self):   # noqa: N802
        parsed = urlparse(self.path)
        path   = parsed.path
        qs     = parse_qs(parsed.query)

        if path == "/":
            self._send(200, "text/html; charset=utf-8", _HTML)
        elif path == "/data":
            if _scene_data is None:
                self._send(503, "application/json", b'{"error":"not loaded"}')
                return
            body = json.dumps(_serialise_scene(_scene_data), separators=(",", ":")).encode()
            self._send(200, "application/json", body)
        elif path == "/patch":
            with _patch_lock:
                body = json.dumps(_patch).encode()
            self._send(200, "application/json", body)
        elif path == "/frame":
            t   = float(qs.get("t",   ["0"])[0])
            cam = int(qs.get("cam",   ["0"])[0])
            self._serve_frame(t, cam)
        elif path == "/cam_subdirs":
            body = json.dumps(
                [{"name": n, "frames": len(nums)} for n, nums, _ in _cam_subdirs]
            ).encode()
            self._send(200, "application/json", body)
        elif path == "/batch_status":
            body = json.dumps(_batch.status_dict()).encode()
            self._send(200, "application/json", body)
        elif path == "/carla_status":
            with _carla_status_lock:
                body = json.dumps(_carla_status).encode()
            self._send(200, "application/json", body)
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):   # noqa: N802
        parsed = urlparse(self.path)
        path   = parsed.path

        if path == "/patch":
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            try:
                data = json.loads(body)
                with _patch_lock:
                    global _patch
                    _patch = data
                if _patch_path:
                    _patch_path.write_text(json.dumps(data, indent=2))
                # Auto-enqueue for CARLA when a patch is saved — only if the
                # scenario's HTML already exists (no HTML = pipeline not yet run,
                # export would fail immediately).
                enqueue_result = {}
                if _batch.enabled and _batch.routes_out_dir and _batch.scenario_dirs:
                    cur_dir = _batch.scenario_dirs[_batch.current_idx]
                    if (cur_dir / "trajectory_plot.html").exists():
                        enqueue_result = _enqueue_for_carla(cur_dir, data)
                    else:
                        enqueue_result = {"action": "skipped", "reason": "no_html"}
                self._send(200, "application/json",
                           json.dumps({"ok": True, "queue": enqueue_result}).encode())
            except Exception as exc:
                self._send(400, "application/json",
                           json.dumps({"error": str(exc)}).encode())
        elif path == "/next":
            self._handle_next()
        elif path == "/queue_remove":
            self._handle_queue_remove()
        elif path == "/goto":
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            try:
                req = json.loads(body)
                idx = int(req.get("idx", -1))
                self._handle_goto(idx)
            except Exception as exc:
                self._send(400, "application/json",
                           json.dumps({"error": str(exc)}).encode())
        elif path == "/skip":
            self._handle_skip()
        elif path == "/carla_reconnect":
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            try:
                req = json.loads(body)
                host = str(req.get("host", _batch.carla_host))
                port = int(req.get("port", _batch.carla_port))
                _batch.carla_host = host
                _batch.carla_port = port
                result = _check_carla_health(host, port)
                with _carla_status_lock:
                    _carla_status.update(result)
                self._send(200, "application/json", json.dumps(result).encode())
            except Exception as exc:
                self._send(400, "application/json",
                           json.dumps({"error": str(exc)}).encode())
        else:
            self._send(404, "text/plain", b"not found")

    # ── batch handlers ─────────────────────────────────────────────────────

    def _handle_next(self) -> None:
        """Save patch and advance to next scenario (CARLA export handled via queue)."""
        if not _batch.enabled:
            self._send(400, "application/json", b'{"error":"batch mode not active"}')
            return

        cur_idx = _batch.current_idx

        # Save current patch to disk
        with _patch_lock:
            saved_patch = dict(_patch)
        if _patch_path:
            _patch_path.write_text(json.dumps(saved_patch, indent=2))

        # Move to next scenario
        next_idx = cur_idx + 1
        if next_idx >= len(_batch.scenario_dirs):
            self._send(200, "application/json",
                       json.dumps({"done": True, "message": "All scenarios complete!"}).encode())
            return

        ok = _activate_scenario(next_idx)
        if not ok:
            self._send(500, "application/json",
                       json.dumps({"error": f"Failed to load scenario {next_idx}"}).encode())
            return

        self._send(200, "application/json",
                   json.dumps({"done": False, **_batch.status_dict()}).encode())

    def _handle_queue_remove(self) -> None:
        """Remove a pending scenario from the CARLA queue."""
        if not _batch.enabled:
            self._send(400, "application/json", b'{"error":"batch mode not active"}')
            return
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)
        try:
            req = json.loads(body)
            name = str(req.get("scenario", ""))
            result = _dequeue_for_carla(name)
            self._send(200, "application/json",
                       json.dumps({**result, **_batch.status_dict()}).encode())
        except Exception as exc:
            self._send(400, "application/json",
                       json.dumps({"error": str(exc)}).encode())

    def _handle_goto(self, idx: int) -> None:
        """Jump to a specific scenario index."""
        if not _batch.enabled:
            self._send(400, "application/json", b'{"error":"batch mode not active"}')
            return
        if idx < 0 or idx >= len(_batch.scenario_dirs):
            self._send(400, "application/json", b'{"error":"invalid index"}')
            return
        ok = _activate_scenario(idx)
        if not ok:
            self._send(500, "application/json",
                       json.dumps({"error": f"Failed to load scenario {idx}"}).encode())
            return
        self._send(200, "application/json",
                   json.dumps({"done": False, **_batch.status_dict()}).encode())

    def _handle_skip(self) -> None:
        """Skip current scenario without saving / exporting."""
        if not _batch.enabled:
            self._send(400, "application/json", b'{"error":"batch mode not active"}')
            return
        next_idx = _batch.current_idx + 1
        if next_idx >= len(_batch.scenario_dirs):
            self._send(200, "application/json",
                       json.dumps({"done": True, "message": "All scenarios complete!"}).encode())
            return
        ok = _activate_scenario(next_idx)
        if not ok:
            self._send(500, "application/json",
                       json.dumps({"error": f"Failed to load scenario {next_idx}"}).encode())
            return
        self._send(200, "application/json",
                   json.dumps({"done": False, **_batch.status_dict()}).encode())

    # ── helpers ────────────────────────────────────────────────────────────

    def _send(self, code: int, ct: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _serve_frame(self, t: float, cam: int) -> None:
        p = _frame_for_t(t, cam)
        if p is None:
            self.send_response(204)
            self.end_headers()
            return
        data = p.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


# ---------------------------------------------------------------------------
# Embedded HTML page  (everything inline — zero external dependencies)
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Patch Editor</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden;background:#111;color:#ccc;font:12px/1.4 'Segoe UI',system-ui,sans-serif}
body{display:flex;flex-direction:column}

/* ── toolbar ── */
#tb{flex-shrink:0;background:#1a1a2e;border-bottom:1px solid #333;padding:4px 8px;
    display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.tg{display:flex;gap:3px;align-items:center}
.tg-label{color:#888;font-size:10px;text-transform:uppercase;letter-spacing:.5px;margin-right:2px}
button{background:#2a2a3e;border:1px solid #444;color:#ccc;padding:4px 12px;
       cursor:pointer;font:12px 'Segoe UI',system-ui,sans-serif;border-radius:3px;transition:all .15s}
button:hover{background:#3a3a4e;border-color:#666}
button.on{background:#1e3a5a;border-color:#4488cc;color:#eef}
button:disabled{opacity:.35;cursor:default}
.vsep{width:1px;background:#333;align-self:stretch;margin:0 4px}

/* Play button - big and prominent */
#btn-play{background:#1a5a1a;border-color:#2a8a2a;color:#8f8;font-size:14px;
           padding:5px 18px;font-weight:bold;min-width:90px}
#btn-play:hover{background:#2a6a2a}
#btn-play.on{background:#8a2a1a;border-color:#cc4444;color:#faa}

/* Speed slider */
#speed-wrap{display:flex;align-items:center;gap:4px}
#speed-val{color:#888;font-size:10px;min-width:28px}
#speed{width:60px;height:14px;accent-color:#4488cc}

#lane-snap-opts{display:flex;align-items:center;gap:5px;margin-left:4px}
#lane-snap-opts label{display:flex;align-items:center;gap:3px;color:#8aa;font-size:10px}
#lane-snap-opts input[type="checkbox"]{accent-color:#4488cc}
#lane-blend{width:52px;background:#222;border:1px solid #555;color:#ccd;
            font:11px monospace;padding:2px 4px;border-radius:3px}
#lane-break-lbl{color:#678;font:10px monospace;min-width:72px}

/* Ego selector */
#ego-sel{background:#222;border:1px solid #4488cc;color:#4af;font:bold 12px monospace;
         padding:3px 8px;border-radius:3px;cursor:pointer;min-width:90px}
#ego-sel option{background:#222;color:#ccc}

#save-ok{color:#4a9;margin-left:6px;font-size:11px}

/* ── main area ── */
#main{flex:1;display:flex;min-height:0;position:relative}
#map-wrap{flex:1;position:relative;overflow:hidden;background:#0e0e0e}
#map{display:block;width:100%;height:100%}

/* Map overlay info */
#map-overlay{position:absolute;top:8px;left:8px;background:rgba(0,0,0,.75);
             border:1px solid #333;border-radius:4px;padding:6px 10px;
             font:bold 13px monospace;color:#4af;pointer-events:none;z-index:10}
#map-overlay .ego-name{color:#ffcc44;font-size:14px}
#map-overlay .time-info{color:#888;font-size:11px;margin-top:2px}

/* ── right panel with tabs ── */
#right{width:240px;flex-shrink:0;border-left:1px solid #333;display:flex;
       flex-direction:column;overflow:hidden;background:#131318}
#right-tabs{display:flex;flex-shrink:0;border-bottom:1px solid #2a2a2a}
.rtab{flex:1;padding:5px 4px;text-align:center;cursor:pointer;font-size:10px;
      font-weight:bold;text-transform:uppercase;letter-spacing:.5px;color:#666;
      background:#131318;border:none;transition:all .12s}
.rtab:hover{color:#aaa;background:#1c1c28}
.rtab.on{color:#4af;background:#16202e;border-bottom:2px solid #4af}
#actors-panel{display:flex;flex-direction:column;flex:1;overflow:hidden;min-height:0}
#scenarios-panel{display:none;flex-direction:column;flex:1;overflow:hidden;min-height:0}
#scenarios-panel.active{display:flex}
#actors-panel.active{display:flex}

/* ── floating camera panel ── */
#cam-float{position:absolute;z-index:50;top:8px;right:228px;
           width:480px;min-width:200px;min-height:150px;
           background:#0a0a0e;border:1px solid #444;border-radius:6px;
           box-shadow:0 4px 24px rgba(0,0,0,.7);display:flex;flex-direction:column;
           overflow:hidden;resize:both}
#cam-float.collapsed{height:auto!important;min-height:0;resize:none}
#cam-float.collapsed #cam-inner{display:none}
#cam-hdr{display:flex;align-items:center;gap:6px;padding:5px 8px;
          background:#181820;cursor:move;user-select:none;flex-shrink:0;
          border-bottom:1px solid #2a2a2a;border-radius:6px 6px 0 0}
#cam-hdr span{color:#4af;font-size:11px;font-weight:bold;flex:1}
#cam-hdr button{background:none;border:none;color:#888;font-size:14px;
                cursor:pointer;padding:0 4px;line-height:1}
#cam-hdr button:hover{color:#eee}
#cam-inner{flex:1;display:flex;flex-direction:column;min-height:0}
#cam-box{flex:1;background:#0a0a0a;position:relative;overflow:hidden;min-height:100px}
#cam-img{width:100%;height:100%;object-fit:contain;display:block}
#cam-info{position:absolute;bottom:0;left:0;right:0;padding:2px 6px;
           font-size:10px;color:#666;background:rgba(0,0,0,.6)}
/* resize grip visual hint */
#cam-float::after{content:'';position:absolute;bottom:2px;right:2px;
                   width:12px;height:12px;border-right:2px solid #555;
                   border-bottom:2px solid #555;pointer-events:none;opacity:.6}
#actor-hdr{padding:5px 8px;color:#888;font-size:11px;font-weight:bold;
           border-bottom:1px solid #222;text-transform:uppercase;letter-spacing:.5px}
#actor-list{flex:1;overflow-y:auto;padding:2px 0}
.ai{padding:4px 10px;cursor:pointer;white-space:nowrap;overflow:hidden;
    text-overflow:ellipsis;font-size:12px;border-left:3px solid transparent;transition:all .1s}
.ai:hover{background:#1c1c2c}
.ai.sel{background:#1e3050;color:#eee;border-left-color:#4af}
.ai.ego-item{font-weight:bold}
.ai.del{color:#b55;text-decoration:line-through}
.ai.pat{color:#fc0}
#patch-hdr{padding:5px 8px;color:#888;font-size:10px;font-weight:bold;
           border-top:1px solid #222;text-transform:uppercase;letter-spacing:.5px}
#patch-sum{padding:4px 8px;font-size:11px;color:#aaa;
            min-height:48px;white-space:pre-wrap;word-break:break-all;
            font-family:monospace}
/* ── scenario browser ── */
#scenario-list{flex:1;overflow-y:auto;padding:2px 0}
.si{padding:4px 8px;cursor:pointer;font-size:11px;border-left:3px solid transparent;
    transition:all .1s;display:flex;align-items:center;gap:5px;min-width:0}
.si:hover{background:#1c1c2c}
.si.cur{background:#1a2a1a;border-left-color:#4f4;color:#eee}
.si-name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#aaa}
.si.cur .si-name{color:#eee}
.si-badge{flex-shrink:0;font-size:9px;font-weight:bold;padding:1px 5px;border-radius:8px;
          letter-spacing:.3px;white-space:nowrap}
.si-badge.nohtml{background:#222;color:#555;border:1px solid #333}
.si-badge.ready{background:#222;color:#888;border:1px solid #444}
.si-badge.patched{background:#332200;color:#fc0;border:1px solid #664400}
.si-badge.queued{background:#0a1a2e;color:#8af;border:1px solid #2a4a6e}
.si-badge.running{background:#2a1a00;color:#fa0;border:1px solid #664400;
                   animation:si-pulse .9s infinite}
.si-badge.done{background:#0a2a0a;color:#4f4;border:1px solid #1a5a1a}
.si-badge.error{background:#2a0a0a;color:#f44;border:1px solid #5a1a1a}
@keyframes si-pulse{0%,100%{opacity:1}50%{opacity:.4}}
.si-rm{flex-shrink:0;background:none;border:none;color:#555;cursor:pointer;
        font-size:11px;padding:0 2px;line-height:1;display:none}
.si:hover .si-rm.visible{display:inline}
#queue-summary{flex-shrink:0;padding:5px 8px;font-size:10px;color:#888;
               border-top:1px solid #1a1a1a;background:#0e0e14;
               cursor:pointer;transition:background .12s}
#queue-summary:hover{background:#14141e}
#queue-summary.has-items{color:#8af}

/* ── timeline ── */
#tl-wrap{flex-shrink:0;background:#161620;border-top:1px solid #333;padding:4px 8px 5px}
#tl-lbl{color:#888;font-size:11px;margin-bottom:3px;height:15px;overflow:hidden;
        display:flex;align-items:center;gap:8px}
#tl-lbl .actor-name{color:#4af;font-weight:bold}
#tl{display:block;width:100%;height:36px;cursor:col-resize;border-radius:3px}

/* ── status ── */
#status{flex-shrink:0;background:#111;border-top:1px solid #1e1e1e;padding:3px 10px;
        color:#888;font-size:11px}

/* Tool hints */
.tool-hint{position:absolute;bottom:50px;left:50%;transform:translateX(-50%);
           background:rgba(30,58,90,.9);color:#adf;padding:8px 16px;border-radius:6px;
           font-size:12px;pointer-events:none;z-index:20;white-space:nowrap;
           border:1px solid #4488cc;display:none}
#map-wrap:has(.tool-hint.show) .tool-hint.show{display:block}
.tool-hint.show{display:block!important}

/* ── batch bar ── */
#batch-bar{display:none;flex-shrink:0;background:#1a2a1a;border-bottom:1px solid #2a5a2a;
           padding:5px 10px;align-items:center;gap:8px;flex-wrap:wrap}
#batch-bar.active{display:flex}
#batch-counter{color:#8f8;font:bold 14px monospace;min-width:80px}
#batch-name{color:#ff0;font:bold 12px monospace;flex:1;overflow:hidden;
             text-overflow:ellipsis;white-space:nowrap;max-width:260px}
#btn-next{background:#1a4a6a;border-color:#2a7aaa;color:#8cf;font-size:13px;
          padding:4px 18px;font-weight:bold}
#btn-next:hover{background:#1a5a8a}
#btn-skip{background:#4a3a1a;border-color:#886622;color:#fc8;font-size:12px;padding:4px 12px}
#btn-skip:hover{background:#5a4a2a}
#btn-prev{background:#2a2a3e;color:#aaf;padding:4px 12px}
#btn-prev:hover{background:#3a3a5e}
#batch-queue-info{font-size:10px;color:#888;cursor:pointer;padding:2px 8px;
                   border:1px solid #333;border-radius:3px;white-space:nowrap}
#batch-queue-info:hover{border-color:#555;color:#aaa}
#batch-queue-info.active{color:#8af;border-color:#4466aa}

/* ── CARLA status ── */
#carla-bar{display:none;flex-shrink:0;background:#111;border-bottom:1px solid #333;
           padding:3px 10px;align-items:center;gap:8px;font-size:11px}
#carla-bar.active{display:flex}
#carla-dot{width:10px;height:10px;border-radius:50%;background:#555;flex-shrink:0}
#carla-dot.ok{background:#4f4}
#carla-dot.err{background:#f44;animation:carla-pulse 1.5s infinite}
@keyframes carla-pulse{0%,100%{opacity:1}50%{opacity:.3}}
#carla-text{color:#aaa;flex:1}
#carla-reconnect{color:#4af;background:none;border:1px solid #4af;padding:2px 10px;
                 font-size:10px;cursor:pointer}
#carla-reconnect:hover{background:#1a2a4a}

/* ── reconnect modal ── */
#reconnect-modal{display:none;position:fixed;top:0;left:0;right:0;bottom:0;
                  background:rgba(0,0,0,.7);z-index:100;align-items:center;justify-content:center}
#reconnect-modal.show{display:flex}
#reconnect-box{background:#1a1a2e;border:2px solid #4488cc;border-radius:8px;padding:20px;
               min-width:320px;color:#ccc}
#reconnect-box h3{color:#4af;margin-bottom:10px}
#reconnect-box label{display:block;margin:6px 0 2px;color:#aaa;font-size:11px}
#reconnect-box input{background:#222;border:1px solid #555;color:#eee;padding:5px 8px;
                      width:100%;font:13px monospace;border-radius:3px}
#reconnect-box .btn-row{display:flex;gap:8px;margin-top:14px;justify-content:flex-end}
#reconnect-box button{padding:6px 16px}

/* ── done overlay ── */
#done-overlay{display:none;position:fixed;top:0;left:0;right:0;bottom:0;
               background:rgba(0,0,0,.85);z-index:100;align-items:center;justify-content:center}
#done-overlay.show{display:flex}
#done-box{text-align:center;color:#8f8;font-size:24px;font-weight:bold}
#done-box p{color:#aaa;font-size:14px;margin-top:10px}
</style>
</head>
<body>

<!-- CARLA status bar -->
<div id="carla-bar">
  <div id="carla-dot"></div>
  <span id="carla-text">CARLA: checking...</span>
  <button id="carla-reconnect" onclick="showReconnect()">Reconnect</button>
</div>

<!-- Batch progress bar -->
<div id="batch-bar">
  <span id="batch-counter">1 / 1</span>
  <span id="batch-name">scenario</span>
  <div style="flex:1"></div>
  <span id="batch-queue-info" onclick="switchRightTab('scenarios')" title="Show scenario list">Queue: –</span>
  <button id="btn-prev" onclick="batchPrev()" title="Go to previous scenario">&#9664; Prev</button>
  <button id="btn-skip" onclick="batchSkip()" title="Skip without saving">Skip</button>
  <button id="btn-next" onclick="batchNext()" title="Save patch and go to next scenario">Next &#9654;</button>
</div>

<!-- Reconnect modal -->
<div id="reconnect-modal">
  <div id="reconnect-box">
    <h3>CARLA Connection</h3>
    <label>Host</label>
    <input type="text" id="rc-host" value="localhost">
    <label>Port</label>
    <input type="number" id="rc-port" value="2000">
    <div class="btn-row">
      <button onclick="hideReconnect()">Cancel</button>
      <button onclick="doReconnect()" style="background:#1a4a6a;color:#4af">Connect</button>
    </div>
  </div>
</div>

<!-- Done overlay -->
<div id="done-overlay">
  <div id="done-box">
    &#10003; All Scenarios Complete!
    <p>Background tasks may still be running — check the terminal.</p>
  </div>
</div>

<div id="tb">
  <!-- Ego selector -->
  <div class="tg">
    <span class="tg-label">Ego:</span>
    <select id="ego-sel" onchange="onEgoChange(this.value)"></select>
  </div>
  <div class="vsep"></div>

  <!-- Playback -->
  <div class="tg">
    <button id="btn-play" onclick="playPause()" title="Space">&#9654; Play</button>
    <button id="btn-follow" onclick="toggleFollow()" title="F">Follow</button>
    <div id="speed-wrap">
      <input type="range" id="speed" min="1" max="30" value="10"
             oninput="setPlaySpeed(this.value)" title="Playback speed">
      <span id="speed-val">1.0x</span>
    </div>
  </div>
  <div class="vsep"></div>

  <!-- Tools -->
  <div class="tg">
    <span class="tg-label">Tool:</span>
    <button id="btn-sel" class="on" onclick="setTool('sel')" title="S">&#9750; Select</button>
    <button id="btn-lane" onclick="setTool('lane')" title="L">&#8644; Lane Snap</button>
    <button id="btn-wp" onclick="setTool('wp')" title="W">&#9679; Waypoint</button>
    <div id="lane-snap-opts" title="Lane snap options">
      <label><input id="lane-after-cur" type="checkbox" onchange="updateLaneSnapUi()">After t</label>
      <input id="lane-blend" type="number" min="0" max="10" step="0.1" value="0.8" title="Blend-in seconds after breakpoint">
      <span id="lane-break-lbl">full</span>
    </div>
  </div>
  <div class="vsep"></div>

  <!-- Actions -->
  <div class="tg">
    <button onclick="doOuter()" title="O">Outermost</button>
    <button onclick="doPhase()" title="P">Phase</button>
    <button onclick="doDelete()" title="D" style="color:#f88">&#10005; Delete</button>
  </div>
  <div class="vsep"></div>

  <!-- Undo/Redo/Save -->
  <div class="tg">
    <button id="btn-undo" onclick="doUndo()" title="Ctrl+Z" disabled>&#8630; Undo</button>
    <button id="btn-redo" onclick="doRedo()" title="Ctrl+Y" disabled>&#8631; Redo</button>
    <button onclick="fitAll()" title="A">Fit</button>
    <button onclick="doSave()" title="Ctrl+S" style="color:#4a9">&#128190; Save</button>
    <span id="save-ok"></span>
  </div>
</div>

<div id="main">
  <div id="map-wrap">
    <canvas id="map"></canvas>
    <div id="map-overlay">
      <div class="ego-name" id="ov-ego">No ego selected</div>
      <div class="time-info" id="ov-time">t = 0.00s</div>
    </div>
    <div class="tool-hint" id="tool-hint"></div>
  </div>

  <!-- Floating camera panel -->
  <div id="cam-float">
    <div id="cam-hdr">
      <span id="cam-label">Camera — Ego 0</span>
      <button onclick="toggleCamCollapse()" title="Collapse" id="cam-collapse-btn">&#9660;</button>
    </div>
    <div id="cam-inner">
      <div id="cam-box">
        <img id="cam-img" src="" alt="">
        <div id="cam-info">no image</div>
      </div>
    </div>
  </div>

  <div id="right">
    <div id="right-tabs">
      <button class="rtab on" id="rtab-actors"  onclick="switchRightTab('actors')">Actors</button>
      <button class="rtab"    id="rtab-scenarios" onclick="switchRightTab('scenarios')" style="display:none">Scenarios</button>
    </div>
    <div id="actors-panel" class="active">
      <div id="actor-hdr">Actors</div>
      <div id="actor-list"></div>
      <div id="patch-hdr">Patch Info</div>
      <div id="patch-sum">&lt;no patches&gt;</div>
    </div>
    <div id="scenarios-panel">
      <div id="scenario-list"></div>
      <div id="queue-summary" onclick="switchRightTab('scenarios')">Queue: no batch data</div>
    </div>
  </div>
</div>

<div id="tl-wrap">
  <div id="tl-lbl"><span>No actor selected</span></div>
  <canvas id="tl"></canvas>
</div>
<div id="status">Loading&#8230;</div>

<script>
// ============================================================
// Constants
// ============================================================
const ROLE_COL  = {ego:'#4488ff', vehicle:'#ffa040', walker:'#40cc60', cyclist:'#bb80ff'};
const ROLE_ICON = {ego:'\u2605', vehicle:'\u25b6', walker:'\u265f', cyclist:'\u2699'};
const PHASE_CYCLE = ['approach','turn','exit',null];
const HIT_PX = 14;
const WP_HIT_PX = 10;
const LANE_SNAP_BLEND_DEFAULT = 0.8;
const LANE_SNAP_BLEND_MAX = 10.0;
const STATIC_PATH_THRESH_M = 1.8;
const STATIC_NET_THRESH_M = 1.2;
const STATIC_AVG_SPEED_THRESH = 0.8;
const STATIC_MAX_SEG_SPEED_THRESH = 1.8;

// ============================================================
// State
// ============================================================
let tracks=[], carlaLines=[], trackById={}, lineByIdx={};
let patch={overrides:[]}, ovById={};
let selId=null, curT=0, tMin=0, tMax=10;
let selIds=new Set();   // multi-actor selection (always includes selId when non-null)
let tool='sel';
let camIdx=0;

// Optional CARLA top-down underlay (set when scene data carries one)
let bgImg=null;            // HTMLImageElement, ready when bgImg.complete && naturalWidth>0
let bgImgBounds=null;      // {min_x,max_x,min_y,max_y} in V2XPNP world frame
let bgImgVisible=true;     // toggle
let bgImgOpacity=0.85;     // 0..1
function _loadBgImageFromData(data){
  bgImg=null; bgImgBounds=null;
  const meta=data && data.bg_image;
  if(!meta||!meta.b64||!meta.bounds) return;
  const b=meta.bounds;
  if(![b.min_x,b.max_x,b.min_y,b.max_y].every(v=>Number.isFinite(v))) return;
  if(b.max_x<=b.min_x||b.max_y<=b.min_y) return;
  const im=new Image();
  im.onload=()=>{ if(typeof redrawMap==='function') redrawMap(); };
  im.src='data:image/jpeg;base64,'+meta.b64;
  bgImg=im; bgImgBounds=b;
}
let egoTracks=[];
let _camSubdirCount=0;
let followEgoIdx=0;   // which ego we're following (index into egoTracks)

// Play / follow
let playing=false, playInterval=null;
let followMode=false;
let playFPS=10, playDT=0.1;
let viewRotation=0;   // radians, for follow-mode rotation

// Waypoint state
let wpDragging=false, wpDragIdx=-1, wpDragStart=null, wpDragOrig=null;
let wpSelected=new Set();    // set of frame indices that are selected
let wpBoxSel=false;         // rubber-band selection active
let wpBoxStart=null;        // {x,y} in world coords
let wpBoxEnd=null;          // {x,y} in world coords
let wpDragUndoPending=false;
let wpDragChanged=false;

// Canvas + view
const mapEl  = document.getElementById('map');
const mapCtx = mapEl.getContext('2d');
const tlEl   = document.getElementById('tl');
const tlCtx  = tlEl.getContext('2d');
let cW=800, cH=600, dpr=1;
let vt={tx:400, ty:300, s:1};

// Undo/redo (JSON snapshots)
let undoStack=[], redoStack=[];

// Mouse state
let mDown=false, mDragging=false, mStart={x:0,y:0}, vtSnap={tx:0,ty:0};
let hovLine=null;
let camDebounce=null;
let camLoading=false;
let camPendingT=null;  // queued sim time for the next request, if any
let camLastT=-999;     // last requested time
let camReqId=0;        // monotonic request id for staleness check
let camLastBlobUrl=null;  // current blob URL on cam-img, for revoke on swap

// ============================================================
// Camera panel drag + collapse
// ============================================================
(function initCamDrag(){
  const panel=document.getElementById('cam-float');
  const hdr=document.getElementById('cam-hdr');
  let dragging=false, ox=0, oy=0;
  hdr.addEventListener('mousedown',e=>{
    if(e.target.tagName==='BUTTON') return;
    dragging=true; ox=e.clientX-panel.offsetLeft; oy=e.clientY-panel.offsetTop;
    e.preventDefault();
  });
  document.addEventListener('mousemove',e=>{
    if(!dragging) return;
    let nx=e.clientX-ox, ny=e.clientY-oy;
    nx=Math.max(0,Math.min(window.innerWidth-60,nx));
    ny=Math.max(0,Math.min(window.innerHeight-40,ny));
    panel.style.left=nx+'px'; panel.style.right='auto';
    panel.style.top=ny+'px';
  });
  document.addEventListener('mouseup',()=>{ dragging=false; });
})();

let camCollapsed=false;
function toggleCamCollapse(){
  const panel=document.getElementById('cam-float');
  const btn=document.getElementById('cam-collapse-btn');
  camCollapsed=!camCollapsed;
  if(camCollapsed){
    panel.classList.add('collapsed');
    btn.innerHTML='&#9650;'; btn.title='Expand';
  }else{
    panel.classList.remove('collapsed');
    btn.innerHTML='&#9660;'; btn.title='Collapse';
  }
}

// ============================================================
// View helpers (with rotation support for follow mode)
// ============================================================
function w2c(wx, wy){
  // Apply rotation around center
  let dx = wx*vt.s, dy = -wy*vt.s;
  if(viewRotation!==0){
    const c=Math.cos(viewRotation), s=Math.sin(viewRotation);
    const rx = dx*c - dy*s, ry = dx*s + dy*c;
    dx=rx; dy=ry;
  }
  return [vt.tx + dx, vt.ty + dy];
}
function c2w(cx, cy){
  let dx = cx-vt.tx, dy = cy-vt.ty;
  if(viewRotation!==0){
    const c=Math.cos(-viewRotation), s=Math.sin(-viewRotation);
    const rx = dx*c - dy*s, ry = dx*s + dy*c;
    dx=rx; dy=ry;
  }
  return [dx/vt.s, -dy/vt.s];
}

function fx(f){ return (f.cx != null) ? f.cx : f.x; }
function fy(f){ return (f.cy != null) ? f.cy : f.y; }
function clamp(v, lo, hi){ return Math.max(lo, Math.min(hi, v)); }
function smoothstep01(u){
  const x=clamp(u,0,1);
  return x*x*(3-2*x);
}
function trackMotionStats(track){
  const fs=(track&&Array.isArray(track.frames))?track.frames:[];
  if(fs.length<2){
    return {path:0, net:0, duration:0, avgSpeed:0, maxSegSpeed:0};
  }
  const x0=fx(fs[0]), y0=fy(fs[0]);
  const xn=fx(fs[fs.length-1]), yn=fy(fs[fs.length-1]);
  let path=0, maxSegSpeed=0;
  let px=x0, py=y0;
  let pt=Number(fs[0].t)||0;
  for(let i=1;i<fs.length;i++){
    const x=fx(fs[i]), y=fy(fs[i]);
    const t=Number(fs[i].t);
    const tt=Number.isFinite(t)?t:pt;
    if(x!=null&&y!=null&&px!=null&&py!=null){
      const seg=Math.hypot(x-px,y-py);
      path+=seg;
      maxSegSpeed=Math.max(maxSegSpeed, seg/Math.max(1e-3, tt-pt));
    }
    px=x; py=y; pt=tt;
  }
  const net=(x0!=null&&y0!=null&&xn!=null&&yn!=null)?Math.hypot(xn-x0,yn-y0):0;
  const t0=Number(fs[0].t)||0;
  const t1=Number(fs[fs.length-1].t)||t0;
  const duration=Math.max(0,t1-t0);
  const avgSpeed=duration>1e-3?path/duration:0;
  return {path, net, duration, avgSpeed, maxSegSpeed};
}
function isStaticVehicleTrack(track){
  if(!track||track.role!=='vehicle') return false;
  const typ=String(track.obj_type||'').toLowerCase();
  if(typ.includes('static')||typ.includes('parked')||typ.includes('parking')) return true;
  const st=trackMotionStats(track);
  if(st.path<=STATIC_PATH_THRESH_M && st.net<=STATIC_NET_THRESH_M) return true;
  return st.net<=STATIC_NET_THRESH_M &&
         st.avgSpeed<=STATIC_AVG_SPEED_THRESH &&
         st.maxSegSpeed<=STATIC_MAX_SEG_SPEED_THRESH;
}

// ============================================================
// Frame interpolation
// ============================================================
function frameAt(track, t){
  const fs=track.frames;
  if(!fs.length) return null;
  let lo=0, hi=fs.length-1;
  while(lo<hi){ const m=(lo+hi)>>1; if(fs[m].t<t) lo=m+1; else hi=m; }
  if(lo>0 && Math.abs(fs[lo-1].t-t)<Math.abs(fs[lo].t-t)) lo--;
  return fs[lo];
}
function frameIdxAt(track, t){
  const fs=track.frames;
  if(!fs.length) return -1;
  let lo=0, hi=fs.length-1;
  while(lo<hi){ const m=(lo+hi)>>1; if(fs[m].t<t) lo=m+1; else hi=m; }
  if(lo>0 && Math.abs(fs[lo-1].t-t)<Math.abs(fs[lo].t-t)) lo--;
  return lo;
}

// ============================================================
// Patch helpers
// ============================================================
function getOv(id){
  if(!ovById[id]){
    const ov={actor_id:id};
    ovById[id]=ov;
    patch.overrides.push(ov);
  }
  return ovById[id];
}
function isDeleted(id){ return !!(ovById[id]&&ovById[id].delete); }
function isPatched(id){
  const ov=ovById[id]; if(!ov) return false;
  return !!(ov.delete||ov.snap_to_outermost||ov.phase_override||
            (ov.lane_segment_overrides&&ov.lane_segment_overrides.length)||
            (ov.waypoint_overrides&&ov.waypoint_overrides.length));
}
function applyPatch(p){
  patch=p; ovById={};
  for(const ov of (patch.overrides||[])) ovById[ov.actor_id]=ov;
}
function pushUndo(){
  undoStack.push(JSON.stringify(patch));
  if(undoStack.length>80) undoStack.shift();
  redoStack=[];
  syncUndoBtns();
}
function syncUndoBtns(){
  document.getElementById('btn-undo').disabled=!undoStack.length;
  document.getElementById('btn-redo').disabled=!redoStack.length;
}
function laneSnapUseBreakpoint(){
  const el=document.getElementById('lane-after-cur');
  return !!(el&&el.checked);
}
function laneSnapBlendSec(){
  const el=document.getElementById('lane-blend');
  const raw=el?parseFloat(el.value):LANE_SNAP_BLEND_DEFAULT;
  if(!isFinite(raw)||raw<0) return 0;
  return Math.min(LANE_SNAP_BLEND_MAX, raw);
}
function updateLaneSnapUi(){
  const lbl=document.getElementById('lane-break-lbl');
  if(!lbl) return;
  if(laneSnapUseBreakpoint()){
    lbl.textContent='t='+curT.toFixed(2)+'s';
  } else {
    lbl.textContent='full';
  }
}

// ============================================================
// Play / Follow
// ============================================================
function setPlaySpeed(v){
  const spd = v/10;
  playDT = 0.1*spd;
  document.getElementById('speed-val').textContent=spd.toFixed(1)+'x';
  if(playing){
    clearInterval(playInterval);
    playInterval=setInterval(playTick, 1000/playFPS);
  }
}
function playTick(){
  let t=curT+playDT;
  if(t>tMax){ t=tMin; }
  setCurT(t);
}
function playPause(){
  playing=!playing;
  const btn=document.getElementById('btn-play');
  if(playing){
    btn.innerHTML='&#9208; Pause'; btn.classList.add('on');
    // Clear any stale loading state from scrubbing so the first play tick fires immediately
    camLoading=false; camPendingT=null;
    playInterval=setInterval(playTick, 1000/playFPS);
  } else {
    btn.innerHTML='&#9654; Play'; btn.classList.remove('on');
    clearInterval(playInterval); playInterval=null;
  }
}
function stopPlay(){
  if(playing){ playing=false; clearInterval(playInterval); playInterval=null;
    const btn=document.getElementById('btn-play');
    btn.innerHTML='&#9654; Play'; btn.classList.remove('on'); }
}
function toggleFollow(){
  followMode=!followMode;
  document.getElementById('btn-follow').classList.toggle('on',followMode);
  if(followMode){
    applyFollow();
  } else {
    viewRotation=0;
    redrawAll();
  }
}
function applyFollow(){
  if(!followMode) return;
  // Follow the currently selected ego, or the first ego
  let followTrack=null;
  if(selId && trackById[selId] && trackById[selId].role==='ego'){
    followTrack=trackById[selId];
  } else if(egoTracks.length>0){
    followTrack=egoTracks[followEgoIdx]||egoTracks[0];
  }
  if(!followTrack) return;

  const fi=frameIdxAt(followTrack,curT);
  if(fi<0) return;
  const f=followTrack.frames[fi];
  const x=fx(f), y=fy(f);
  if(x==null||y==null) return;

  // Compute heading from consecutive frames
  let heading=0;
  if(fi<followTrack.frames.length-1){
    const f2=followTrack.frames[fi+1];
    const dx2=fx(f2)-x, dy2=fy(f2)-y;
    if(Math.hypot(dx2,dy2)>0.01) heading=Math.atan2(dy2,dx2);
  } else if(fi>0){
    const f0=followTrack.frames[fi-1];
    const dx2=x-fx(f0), dy2=y-fy(f0);
    if(Math.hypot(dx2,dy2)>0.01) heading=Math.atan2(dy2,dx2);
  }

  // Rotate so heading points up (top of screen)
  viewRotation = -(heading - Math.PI/2);

  // Center ego slightly below center so more forward view is visible
  const offY = cH * 0.2;  // push ego 20% below center
  vt.tx = cW/2;
  vt.ty = cH/2 + offY;

  // Now we need vt.tx/ty such that the ego position maps to (cW/2, cH/2+offY)
  // w2c(x,y) should = (cW/2, cH/2+offY)
  // vt.tx + (x*vt.s*cos(r) - (-y*vt.s)*sin(r)) = cW/2
  // vt.ty + (x*vt.s*sin(r) + (-y*vt.s)*cos(r)) = cH/2+offY
  const c=Math.cos(viewRotation), s=Math.sin(viewRotation);
  const dx = x*vt.s, dy = -y*vt.s;
  const rx = dx*c - dy*s, ry = dx*s + dy*c;
  vt.tx = cW/2 - rx;
  vt.ty = cH/2 + offY - ry;
}

// ============================================================
// Ego selector
// ============================================================
function onEgoChange(v){
  followEgoIdx=parseInt(v)||0;
  const ego=egoTracks[followEgoIdx];
  if(ego){
    selectActor(ego.id);
    followMode=true;
    document.getElementById('btn-follow').classList.add('on');
    applyFollow();
    redrawAll();
  }
}
function buildEgoSelector(){
  const sel=document.getElementById('ego-sel');
  sel.innerHTML='';
  egoTracks.forEach((tr,i)=>{
    const o=document.createElement('option');
    o.value=i;
    o.textContent='Ego '+i+' ('+tr.id+')';
    sel.appendChild(o);
  });
  if(egoTracks.length>0){
    sel.value=0;
    followEgoIdx=0;
  }
}

// ============================================================
// Lane projection helpers
// ============================================================
function projectOntoPolyline(pts, x, y){
  let bestX=pts[0][0], bestY=pts[0][1], bestD=Infinity;
  for(let i=0;i<pts.length-1;i++){
    const ax=pts[i][0],ay=pts[i][1],bx=pts[i+1][0],by=pts[i+1][1];
    const abx=bx-ax,aby=by-ay,ab2=abx*abx+aby*aby;
    if(ab2<1e-12) continue;
    const t=Math.max(0,Math.min(1,((x-ax)*abx+(y-ay)*aby)/ab2));
    const px=ax+t*abx,py=ay+t*aby,d=Math.hypot(x-px,y-py);
    if(d<bestD){bestD=d;bestX=px;bestY=py;}
  }
  return [bestX,bestY];
}

// -- Lane chain: find continuation lanes at endpoints --
let laneChainCache={};   // startIdx -> [idx, idx, ...]

function _laneHeading(pts, end){
  // end=false → heading at start, end=true → heading at end
  if(pts.length<2) return 0;
  if(!end){
    return Math.atan2(pts[1][1]-pts[0][1], pts[1][0]-pts[0][0]);
  } else {
    const n=pts.length;
    return Math.atan2(pts[n-1][1]-pts[n-2][1], pts[n-1][0]-pts[n-2][0]);
  }
}
function _angleDiff(a,b){
  let d=a-b;
  while(d>Math.PI) d-=2*Math.PI;
  while(d<-Math.PI) d+=2*Math.PI;
  return Math.abs(d);
}

// Build the full chain of connected lanes starting from startIdx.
// Walks backward (predecessors) and forward (successors) by matching endpoints.
function buildLaneChain(startIdx){
  if(laneChainCache[startIdx]) return laneChainCache[startIdx];
  const DIST_THR=5.0;   // max gap between lane endpoints (metres)
  const ANG_THR=Math.PI/5;  // ~36 deg heading tolerance

  function findSuccessor(curIdx, visited){
    const cur=lineByIdx[curIdx];
    if(!cur||cur.pts.length<2) return null;
    const endPt=cur.pts[cur.pts.length-1];
    const endH=_laneHeading(cur.pts, true);
    let bestIdx=null, bestD=Infinity;
    for(const l of carlaLines){
      if(l.idx===curIdx||visited.has(l.idx)) continue;
      if(l.pts.length<2) continue;
      const startPt=l.pts[0];
      const d=Math.hypot(endPt[0]-startPt[0], endPt[1]-startPt[1]);
      if(d>DIST_THR) continue;
      const startH=_laneHeading(l.pts, false);
      if(_angleDiff(endH,startH)>ANG_THR) continue;
      if(d<bestD){ bestD=d; bestIdx=l.idx; }
    }
    return bestIdx;
  }
  function findPredecessor(curIdx, visited){
    const cur=lineByIdx[curIdx];
    if(!cur||cur.pts.length<2) return null;
    const startPt=cur.pts[0];
    const startH=_laneHeading(cur.pts, false);
    let bestIdx=null, bestD=Infinity;
    for(const l of carlaLines){
      if(l.idx===curIdx||visited.has(l.idx)) continue;
      if(l.pts.length<2) continue;
      const endPt=l.pts[l.pts.length-1];
      const d=Math.hypot(startPt[0]-endPt[0], startPt[1]-endPt[1]);
      if(d>DIST_THR) continue;
      const endH=_laneHeading(l.pts, true);
      if(_angleDiff(startH,endH)>ANG_THR) continue;
      if(d<bestD){ bestD=d; bestIdx=l.idx; }
    }
    return bestIdx;
  }

  // Walk backward
  const preds=[];
  const visited=new Set([startIdx]);
  let cur=startIdx;
  for(let i=0;i<50;i++){  // safety limit
    const p=findPredecessor(cur, visited);
    if(p===null) break;
    preds.unshift(p);
    visited.add(p);
    cur=p;
  }
  // Walk forward
  const succs=[];
  cur=startIdx;
  for(let i=0;i<50;i++){
    const s=findSuccessor(cur, visited);
    if(s===null) break;
    succs.push(s);
    visited.add(s);
    cur=s;
  }
  const chain=[...preds, startIdx, ...succs];
  laneChainCache[startIdx]=chain;
  return chain;
}

// Build concatenated polyline from a lane chain
function getLaneChainPts(chain){
  const allPts=[];
  for(const idx of chain){
    const l=lineByIdx[idx];
    if(!l||l.pts.length<2) continue;
    // Skip first point if it overlaps with previous segment's last point
    const startI=(allPts.length>0&&l.pts.length>0&&
      Math.hypot(allPts[allPts.length-1][0]-l.pts[0][0],
                 allPts[allPts.length-1][1]-l.pts[0][1])<1.0)?1:0;
    for(let i=startI;i<l.pts.length;i++) allPts.push(l.pts[i]);
  }
  return allPts;
}

function laneOverrideInRange(ls, t){
  const hasStart=ls.start_t!=null && isFinite(Number(ls.start_t));
  const hasEnd=ls.end_t!=null && isFinite(Number(ls.end_t));
  if(hasStart && t < Number(ls.start_t)-1e-9) return false;
  if(hasEnd && t > Number(ls.end_t)+1e-9) return false;
  return true;
}

function laneOverrideBlendWeight(ls, t){
  if(ls.start_t==null || !isFinite(Number(ls.start_t))) return 1;
  const blend=Number(ls.blend_in_s);
  if(!isFinite(blend) || blend<=1e-6) return 1;
  const u=(t-Number(ls.start_t))/blend;
  if(u<=0) return 0;
  if(u>=1) return 1;
  // Smoothstep blend for C1 continuity at both ends.
  return u*u*(3-2*u);
}

function applyLaneOverridesAtFrame(track, frameIdx, x, y, ov){
  const lsos=ov&&ov.lane_segment_overrides;
  if(!lsos||!lsos.length) return [x,y];
  const f=track.frames[frameIdx];
  if(!f) return [x,y];
  const t=Number(f.t||0);
  for(const ls of lsos){
    if(!laneOverrideInRange(ls,t)) continue;
    const laneIdx=parseInt(ls.lane_id,10);
    if(!isFinite(laneIdx)) continue;
    const chainPts=getLaneChainPts(buildLaneChain(laneIdx));
    if(chainPts.length<2) continue;
    const snapped=projectOntoPolyline(chainPts,x,y);
    const w=laneOverrideBlendWeight(ls,t);
    if(w>=0.999){
      x=snapped[0]; y=snapped[1];
    } else if(w>0.001){
      x=x+(snapped[0]-x)*w;
      y=y+(snapped[1]-y)*w;
    }
  }
  return [x,y];
}

function getLaneBasePos(track, frameIdx, ov){
  const f=track.frames[frameIdx];
  if(!f) return null;
  let x=fx(f), y=fy(f);
  if(x==null||y==null) return null;
  if(ov) [x,y]=applyLaneOverridesAtFrame(track, frameIdx, x, y, ov);
  return [x,y];
}

function setWaypointOverrideDelta(ov, frameIdx, dx, dy){
  if(!ov) return;
  if(!ov.waypoint_overrides) ov.waypoint_overrides=[];
  const eps=1e-6;
  const idx=ov.waypoint_overrides.findIndex(w=>w.frame_idx===frameIdx);
  if(Math.abs(dx)<=eps && Math.abs(dy)<=eps){
    if(idx>=0) ov.waypoint_overrides.splice(idx,1);
    return;
  }
  if(idx>=0){
    ov.waypoint_overrides[idx].dx=dx;
    ov.waypoint_overrides[idx].dy=dy;
  } else {
    ov.waypoint_overrides.push({frame_idx:frameIdx, dx:dx, dy:dy});
  }
}

function samplePolylineByArc(points, cumulative, dist){
  if(!points.length) return null;
  if(dist<=0) return [points[0][0], points[0][1]];
  const last=cumulative.length-1;
  if(last<=0) return [points[0][0], points[0][1]];
  const total=cumulative[last];
  if(dist>=total) return [points[last][0], points[last][1]];
  let lo=0, hi=last;
  while(lo+1<hi){
    const mid=(lo+hi)>>1;
    if(cumulative[mid]<dist) lo=mid;
    else hi=mid;
  }
  const s0=cumulative[lo], s1=cumulative[hi];
  const den=Math.max(1e-6, s1-s0);
  const u=(dist-s0)/den;
  const x=points[lo][0]+(points[hi][0]-points[lo][0])*u;
  const y=points[lo][1]+(points[hi][1]-points[lo][1])*u;
  return [x,y];
}

function smoothWaypointEdits(actorId){
  const tr=trackById[actorId];
  const ov=ovById[actorId];
  if(!tr||!ov||!Array.isArray(ov.waypoint_overrides)||ov.waypoint_overrides.length===0){
    return null;
  }

  const changed=[...ov.waypoint_overrides]
    .map(w=>({
      frameIdx:parseInt(w.frame_idx,10),
      mag:Math.hypot(Number(w.dx)||0, Number(w.dy)||0),
    }))
    .filter(w=>Number.isFinite(w.frameIdx)&&w.frameIdx>=0&&w.frameIdx<tr.frames.length&&w.mag>1e-6)
    .sort((a,b)=>a.frameIdx-b.frameIdx);

  if(!changed.length) return null;

  const firstIdx=changed[0].frameIdx;
  const lastIdx=changed[changed.length-1].frameIdx;
  const fullTrackEdited=
    firstIdx===0 &&
    lastIdx===tr.frames.length-1 &&
    changed.length===tr.frames.length;
  if(fullTrackEdited){
    let dist=0;
    let prev=getEffectivePos(tr,0);
    for(let i=1;i<tr.frames.length;i++){
      const cur=getEffectivePos(tr,i);
      if(prev&&cur){
        dist+=Math.hypot(cur[0]-prev[0], cur[1]-prev[1]);
      }
      if(cur) prev=cur;
    }
    const t0=Number(tr.frames[firstIdx]?.t)||0;
    const t1=Number(tr.frames[lastIdx]?.t)||t0;
    const dur=Math.max(1e-6, t1-t0);
    return {
      firstIdx:firstIdx, lastIdx:lastIdx, winStart:firstIdx, winEnd:lastIdx,
      duration:dur, distance:dist, avgSpeed:dist/dur,
    };
  }

  const winStart=Math.max(0, firstIdx-1);
  const winEnd=Math.min(tr.frames.length-1, lastIdx+1);
  if(winEnd<=winStart){
    return {
      firstIdx:firstIdx, lastIdx:lastIdx, winStart:winStart, winEnd:winEnd,
      duration:0, distance:0, avgSpeed:0,
    };
  }

  const desired=[];
  const times=[];
  for(let i=winStart;i<=winEnd;i++){
    const ti=Number(tr.frames[i]?.t);
    if(Number.isFinite(ti)) times.push(ti);
    else if(times.length) times.push(times[times.length-1]+playDT);
    else times.push(0);

    const ep=getEffectivePos(tr,i);
    if(ep) desired.push([ep[0],ep[1]]);
    else{
      const bp=getLaneBasePos(tr,i,ov);
      desired.push(bp||[0,0]);
    }
  }

  const cumulative=[0];
  for(let i=1;i<desired.length;i++){
    const seg=Math.hypot(desired[i][0]-desired[i-1][0], desired[i][1]-desired[i-1][1]);
    cumulative.push(cumulative[i-1]+seg);
  }

  const totalDist=cumulative[cumulative.length-1]||0;
  const t0=times[0], t1=times[times.length-1];
  const duration=Math.max(1e-6, t1-t0);
  const avgSpeed=totalDist/duration;

  for(let i=winStart;i<=winEnd;i++){
    const local=i-winStart;
    const u=(times[local]-t0)/duration;
    const targetDist=totalDist<=1e-6 ? 0 : smoothstep01(u)*totalDist;
    const targetPos=samplePolylineByArc(desired, cumulative, targetDist) || desired[local];
    const base=getLaneBasePos(tr,i,ov);
    if(!base) continue;
    setWaypointOverrideDelta(ov, i, targetPos[0]-base[0], targetPos[1]-base[1]);
  }

  ov.waypoint_overrides.sort((a,b)=>a.frame_idx-b.frame_idx);
  return {
    firstIdx:firstIdx, lastIdx:lastIdx, winStart:winStart, winEnd:winEnd,
    duration:duration, distance:totalDist, avgSpeed:avgSpeed,
  };
}

// Get the effective position for a frame, considering all overrides
function getEffectivePos(track, frameIdx){
  const f=track.frames[frameIdx];
  let x=fx(f), y=fy(f);
  if(x==null||y==null) return null;

  const ov=ovById[track.id];
  if(!ov||isDeleted(track.id)) return [x,y];

  // Apply lane snap (with lane chain continuation)
  [x,y]=applyLaneOverridesAtFrame(track, frameIdx, x, y, ov);

  // Apply waypoint overrides
  const wos=ov.waypoint_overrides;
  if(wos&&wos.length){
    const wo=wos.find(w=>w.frame_idx===frameIdx);
    if(wo){ x+=wo.dx; y+=wo.dy; }
  }

  return [x,y];
}

// Get full effective trajectory for a track
function getEffectivePath(track){
  return track.frames.map((_,i)=>getEffectivePos(track,i));
}

function doUndo(){
  if(!undoStack.length) return;
  redoStack.push(JSON.stringify(patch));
  applyPatch(JSON.parse(undoStack.pop()));
  syncUndoBtns(); redrawAll(); rebuildActorList(); updatePatchSummary();
  status('Undo');
}
function doRedo(){
  if(!redoStack.length) return;
  undoStack.push(JSON.stringify(patch));
  applyPatch(JSON.parse(redoStack.pop()));
  syncUndoBtns(); redrawAll(); rebuildActorList(); updatePatchSummary();
  status('Redo');
}

// ============================================================
// Tools
// ============================================================
function setTool(t){
  tool=t;
  document.getElementById('btn-sel' ).classList.toggle('on',t==='sel');
  document.getElementById('btn-lane').classList.toggle('on',t==='lane');
  document.getElementById('btn-wp'  ).classList.toggle('on',t==='wp');
  mapEl.style.cursor = t==='lane' ? 'cell' : t==='wp' ? 'grab' : 'crosshair';
  const hint=document.getElementById('tool-hint');
  if(t==='lane'){
    hint.textContent='Set timeline t if needed, toggle "After t", then click a CARLA lane line';
    hint.classList.add('show');
    setTimeout(()=>hint.classList.remove('show'),4000);
  } else if(t==='wp'){
    hint.textContent='Click/Shift-click/drag-box to select waypoints. Drag to move. Static cars drag as a whole. Delete/Backspace removes with smoothing.';
    hint.classList.add('show');
    setTimeout(()=>hint.classList.remove('show'),5000);
  } else {
    hint.classList.remove('show');
  }
  redrawAll();
  status(t==='lane'
    ? 'Lane Snap \u2014 toggle "After t" to keep pre-breakpoint trajectory unchanged'
    : t==='wp'
    ? 'Waypoint \u2014 click/shift/box-select, drag to move (static cars move fully), Delete to remove'
    : 'Select \u2014 click an actor');
  wpSelected.clear();
}
function doDelete(){
  if(selIds.size===0){ status('Select an actor first'); return; }
  pushUndo();
  for(const id of selIds) getOv(id).delete=true;
  rebuildActorList(); redrawAll(); updatePatchSummary();
  status('Deleted '+selIds.size+' actor(s) \u2014 Ctrl+Z to undo');
}
function doOuter(){
  if(selIds.size===0){ status('Select an actor first'); return; }
  pushUndo();
  for(const id of selIds) getOv(id).snap_to_outermost=true;
  updatePatchSummary(); status('Outermost snap: '+selIds.size+' actor(s)');
}
function doPhase(){
  if(selIds.size===0){ status('Select an actor first'); return; }
  pushUndo();
  let lbl='';
  for(const id of selIds){
    const ov=getOv(id);
    const cur=ov.phase_override||null;
    const idx=PHASE_CYCLE.indexOf(cur);
    const next=PHASE_CYCLE[(idx+1)%PHASE_CYCLE.length];
    if(next===null) delete ov.phase_override; else ov.phase_override=next;
    lbl=next||'(cleared)';
  }
  updatePatchSummary(); status('Phase \u2192 '+lbl+' ('+selIds.size+' actor(s))');
}
function doLaneSnap(lineIdx){
  if(selIds.size===0){ status('Select an actor first, then switch to Lane Snap'); return; }
  const useBreakpoint=laneSnapUseBreakpoint();
  const startT=useBreakpoint ? Number(curT.toFixed(4)) : null;
  const blendSec=useBreakpoint ? laneSnapBlendSec() : 0;
  pushUndo();
  for(const id of selIds){
    const ov=getOv(id);
    const lso={lane_id:String(lineIdx),start_t:startT,end_t:null};
    if(useBreakpoint && blendSec>0) lso.blend_in_s=Number(blendSec.toFixed(3));
    ov.lane_segment_overrides=[lso];
  }
  rebuildActorList(); redrawAll(); updatePatchSummary();
  let msg='';
  if(useBreakpoint){
    msg='Lane snap \u2192 line '+lineIdx+' from t='+startT.toFixed(2)+
        's (blend '+blendSec.toFixed(2)+'s, '+selIds.size+' actor(s))';
  } else {
    msg='Lane snap \u2192 line '+lineIdx+' full trajectory ('+selIds.size+' actor(s))';
  }
  setTool('sel');
  status(msg);
}
async function doSave(){
  try{
    await fetch('/patch',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify(patch)});
    const n=(patch.overrides||[]).filter(ov=>isPatched(ov.actor_id)).length;
    const ok=document.getElementById('save-ok');
    ok.textContent='\u2713 Saved ('+n+' patches)';
    setTimeout(()=>ok.textContent='',3000);
  }catch(e){ status('Save failed: '+e.message); }
}

// ============================================================
// Hit testing
// ============================================================
function hitActor(wx,wy){
  let best=null, bestD=Infinity;
  const thr=HIT_PX/vt.s;
  for(const tr of tracks){
    if(isDeleted(tr.id)) continue;
    const f=frameAt(tr,curT); if(!f) continue;
    const ep=getEffectivePos(tr, tr.frames.indexOf(f));
    if(!ep) continue;
    const d=Math.hypot(ep[0]-wx, ep[1]-wy);
    if(d<bestD&&d<thr){ bestD=d; best=tr.id; }
  }
  return best;
}
// Hit-test against an actor's full trajectory polyline
function hitTrajectory(wx,wy){
  let best=null, bestD=Infinity;
  const thr=6/vt.s;   // 6 screen pixels tolerance
  for(const tr of tracks){
    if(isDeleted(tr.id)) continue;
    const path=isPatched(tr.id)?getEffectivePath(tr):null;
    // Build polyline from frames or effective path
    const pts=[];
    for(let i=0;i<tr.frames.length;i++){
      if(path&&path[i]){
        pts.push(path[i]);
      } else {
        const x=fx(tr.frames[i]), y=fy(tr.frames[i]);
        if(x!=null&&y!=null) pts.push([x,y]);
      }
    }
    if(pts.length<2) continue;
    const d=polyDist(pts,wx,wy);
    if(d<bestD&&d<thr){ bestD=d; best=tr.id; }
  }
  return best;
}
function hitWaypoint(wx,wy){
  // Returns {trackId, frameIdx} or null
  if(!selId || !trackById[selId]) return null;
  const tr=trackById[selId];
  const thr=WP_HIT_PX/vt.s;
  let best=null, bestD=Infinity;
  for(let i=0;i<tr.frames.length;i++){
    const ep=getEffectivePos(tr,i);
    if(!ep) continue;
    const d=Math.hypot(ep[0]-wx, ep[1]-wy);
    if(d<bestD&&d<thr){ bestD=d; best={trackId:tr.id, frameIdx:i}; }
  }
  return best;
}
function hitLine(wx,wy){
  let best=null, bestD=Infinity;
  const thr=8/vt.s;
  for(const l of carlaLines){
    const d=polyDist(l.pts,wx,wy);
    if(d<bestD&&d<thr){ bestD=d; best=l.idx; }
  }
  return best;
}
function polyDist(pts,px,py){
  let best=Infinity;
  for(let i=0;i<pts.length-1;i++){
    const ax=pts[i][0],ay=pts[i][1],bx=pts[i+1][0],by=pts[i+1][1];
    const abx=bx-ax,aby=by-ay,ab2=abx*abx+aby*aby;
    if(ab2<1e-12) continue;
    const t=Math.max(0,Math.min(1,((px-ax)*abx+(py-ay)*aby)/ab2));
    const d=Math.hypot(px-ax-t*abx,py-ay-t*aby);
    if(d<best) best=d;
  }
  return best;
}

// ============================================================
// Overlap detection
// ============================================================
const OVERLAP_DIST=2.5; // metres — actors closer than this are overlapping
function computeOverlaps(){
  const ids=new Set();
  const pos=[];
  for(const tr of tracks){
    if(isDeleted(tr.id)) continue;
    const fi=frameIdxAt(tr,curT);
    if(fi<0) continue;
    const ep=getEffectivePos(tr,fi);
    if(!ep) continue;
    pos.push({id:tr.id, x:ep[0], y:ep[1]});
  }
  for(let a=0;a<pos.length;a++){
    for(let b=a+1;b<pos.length;b++){
      const dx=pos[a].x-pos[b].x, dy=pos[a].y-pos[b].y;
      if(Math.sqrt(dx*dx+dy*dy)<OVERLAP_DIST){
        ids.add(pos[a].id); ids.add(pos[b].id);
      }
    }
  }
  return ids;
}

// ============================================================
// Rendering
// ============================================================
function redrawMap(){
  mapCtx.save();
  mapCtx.scale(dpr,dpr);
  mapCtx.fillStyle='#0e0e0e';
  mapCtx.fillRect(0,0,cW,cH);

  // CARLA top-down underlay (drawn first, beneath everything).
  // Image bounds are an axis-aligned bbox in V2XPNP frame; we apply the same
  // w2c projection to the bounds corners as we do to every polyline point —
  // so they share screen-space and stay aligned.
  if(bgImg && bgImg.complete && bgImg.naturalWidth>0 && bgImgBounds && bgImgVisible){
    const b=bgImgBounds;
    const [tlx,tly]=w2c(b.min_x,b.min_y);
    const [brx,bry]=w2c(b.max_x,b.max_y);
    const x=Math.min(tlx,brx), y=Math.min(tly,bry);
    const w=Math.abs(brx-tlx), h=Math.abs(bry-tly);
    if(w>0 && h>0){
      const prevAlpha=mapCtx.globalAlpha;
      mapCtx.globalAlpha=Math.max(0,Math.min(1,bgImgOpacity));
      mapCtx.drawImage(bgImg,x,y,w,h);
      mapCtx.globalAlpha=prevAlpha;
    }
  }

  // CARLA lines
  for(const l of carlaLines){
    if(l.pts.length<2) continue;
    const hi=(tool==='lane'&&l.idx===hovLine);
    mapCtx.strokeStyle=hi?'#4d4':'#3a3a3a';
    mapCtx.lineWidth=hi?2.5:0.8;
    mapCtx.beginPath();
    let [sx,sy]=w2c(l.pts[0][0],l.pts[0][1]); mapCtx.moveTo(sx,sy);
    for(let i=1;i<l.pts.length;i++){
      [sx,sy]=w2c(l.pts[i][0],l.pts[i][1]); mapCtx.lineTo(sx,sy);
    }
    mapCtx.stroke();
  }

  // Trajectories
  for(const tr of tracks){
    const sel=selIds.has(tr.id), del=isDeleted(tr.id), pat=isPatched(tr.id);
    const col=ROLE_COL[tr.role]||'#aaa';
    const isEgo=tr.role==='ego';
    const epath=pat?getEffectivePath(tr):null;
    const hasModifiedPath=epath&&epath.some((p,i)=>{
      if(!p) return false;
      const f=tr.frames[i];
      return Math.abs(p[0]-fx(f))>0.01||Math.abs(p[1]-fy(f))>0.01;
    });

    // Original trajectory
    mapCtx.globalAlpha=del?0.12:(hasModifiedPath?0.2:1);
    mapCtx.strokeStyle=del?'#933':sel?'#88ccff':col;
    mapCtx.lineWidth=sel?2.5:(isEgo?2:1.2);
    if(hasModifiedPath) mapCtx.setLineDash([3,3]);
    if(tr.frames.length>=2){
      mapCtx.beginPath();
      let first=true;
      for(let i=0;i<tr.frames.length;i++){
        const x=fx(tr.frames[i]),y=fy(tr.frames[i]);
        if(x==null||y==null){first=true;continue;}
        const [sx,sy]=w2c(x,y);
        if(first){mapCtx.moveTo(sx,sy);first=false;} else mapCtx.lineTo(sx,sy);
      }
      mapCtx.stroke();
    }
    mapCtx.setLineDash([]);
    mapCtx.globalAlpha=1;

    // Modified trajectory - solid bright orange/yellow
    if(hasModifiedPath){
      mapCtx.strokeStyle=sel?'#ffe066':'#ffa500';
      mapCtx.lineWidth=sel?3.5:2.5;
      mapCtx.beginPath();
      let first=true;
      for(let i=0;i<epath.length;i++){
        if(!epath[i]){first=true;continue;}
        const [sx,sy]=w2c(epath[i][0],epath[i][1]);
        if(first){mapCtx.moveTo(sx,sy);first=true;} else mapCtx.lineTo(sx,sy);
        first=false;
      }
      mapCtx.stroke();

      // Direction arrows on modified path
      mapCtx.fillStyle=sel?'#ffe066':'#ffa500';
      for(let i=0;i<epath.length-1;i+=Math.max(1,Math.floor(epath.length/8))){
        if(!epath[i]||!epath[i+1]) continue;
        const [sx,sy]=w2c(epath[i][0],epath[i][1]);
        const [sx2,sy2]=w2c(epath[i+1][0],epath[i+1][1]);
        const ang=Math.atan2(sy2-sy,sx2-sx);
        mapCtx.save();
        mapCtx.translate(sx,sy);
        mapCtx.rotate(ang);
        mapCtx.beginPath();
        mapCtx.moveTo(6,0); mapCtx.lineTo(-2,-3); mapCtx.lineTo(-2,3);
        mapCtx.fill();
        mapCtx.restore();
      }
    }

    // Waypoint dots (when waypoint tool is active and this actor is selected)
    if(tool==='wp'&&sel&&!del){
      for(let i=0;i<tr.frames.length;i++){
        const ep=getEffectivePos(tr,i);
        if(!ep) continue;
        const [sx,sy]=w2c(ep[0],ep[1]);
        const isNear=Math.abs(tr.frames[i].t-curT)<(playDT*2);
        const isSel=wpSelected.has(i);
        const isDrag=wpDragIdx===i||(wpDragging&&isSel);
        mapCtx.fillStyle=isDrag?'#ff0':isSel?'#ff8844':isNear?'#fff':'#88ccff';
        const r=isSel?5:isNear?5:3;
        mapCtx.beginPath();
        mapCtx.arc(sx,sy,r,0,Math.PI*2);
        mapCtx.fill();
        if(isSel||isNear){
          mapCtx.strokeStyle=isSel?'#ff8844':'#fff';
          mapCtx.lineWidth=isSel?2:1;
          mapCtx.beginPath();
          mapCtx.arc(sx,sy,r+2,0,Math.PI*2);
          mapCtx.stroke();
        }
      }
      // Rubber-band selection rectangle
      if(wpBoxSel&&wpBoxStart&&wpBoxEnd){
        const [s1x,s1y]=w2c(wpBoxStart.x,wpBoxStart.y);
        const [s2x,s2y]=w2c(wpBoxEnd.x,wpBoxEnd.y);
        mapCtx.save();
        mapCtx.strokeStyle='#ff8844';
        mapCtx.lineWidth=1;
        mapCtx.setLineDash([4,3]);
        mapCtx.strokeRect(Math.min(s1x,s2x),Math.min(s1y,s2y),
                          Math.abs(s2x-s1x),Math.abs(s2y-s1y));
        mapCtx.fillStyle='rgba(255,136,68,0.08)';
        mapCtx.fillRect(Math.min(s1x,s2x),Math.min(s1y,s2y),
                        Math.abs(s2x-s1x),Math.abs(s2y-s1y));
        mapCtx.restore();
      }
    }
  }

  // ── Overlap detection at current time ──
  const overlapIds=computeOverlaps();

  // Current-time markers + vehicle annotations
  for(const tr of tracks){
    if(isDeleted(tr.id)) continue;
    const fi=frameIdxAt(tr,curT);
    if(fi<0) continue;
    const f=tr.frames[fi];
    const ep=getEffectivePos(tr,fi);
    if(!ep) continue;
    const [px,py]=ep;
    const [sx,sy]=w2c(px,py);
    const sel=selIds.has(tr.id);
    const col=ROLE_COL[tr.role]||'#aaa';
    const isEgo=tr.role==='ego';
    const egoIdx=isEgo?egoTracks.indexOf(tr):-1;

    // Direction indicator (triangle showing heading)
    if(isEgo || sel){
      let heading=0;
      if(fi<tr.frames.length-1){
        const ep2=getEffectivePos(tr,fi+1);
        if(ep2) heading=Math.atan2(ep2[1]-py,ep2[0]-px);
      } else if(fi>0){
        const ep0=getEffectivePos(tr,fi-1);
        if(ep0) heading=Math.atan2(py-ep0[1],px-ep0[0]);
      }
      // Draw direction triangle
      mapCtx.save();
      mapCtx.translate(sx,sy);
      // Convert world heading to screen heading
      const screenAng = -heading + viewRotation;
      mapCtx.rotate(-screenAng);
      const sz=sel?14:(isEgo?11:8);
      mapCtx.fillStyle=isEgo?'rgba(68,136,255,0.4)':'rgba(255,160,64,0.3)';
      mapCtx.beginPath();
      mapCtx.moveTo(sz,0); mapCtx.lineTo(-sz*0.6,-sz*0.7); mapCtx.lineTo(-sz*0.6,sz*0.7);
      mapCtx.closePath(); mapCtx.fill();
      mapCtx.restore();
    }

    // Dot
    const dotR=sel?8:(isEgo?6:3.5);
    mapCtx.fillStyle=sel?'#fff':col;
    mapCtx.beginPath(); mapCtx.arc(sx,sy,dotR,0,Math.PI*2); mapCtx.fill();
    if(isEgo){
      mapCtx.strokeStyle=sel?'#fff':'#4488ff';
      mapCtx.lineWidth=2;
      mapCtx.beginPath(); mapCtx.arc(sx,sy,dotR+3,0,Math.PI*2); mapCtx.stroke();
    }
    if(sel){
      mapCtx.strokeStyle='#5af'; mapCtx.lineWidth=2;
      mapCtx.beginPath(); mapCtx.arc(sx,sy,14,0,Math.PI*2); mapCtx.stroke();
    }

    // Overlap warning ring
    if(overlapIds.has(tr.id)){
      mapCtx.strokeStyle='#ff2222';
      mapCtx.lineWidth=2.5;
      mapCtx.setLineDash([4,3]);
      mapCtx.beginPath(); mapCtx.arc(sx,sy,18,0,Math.PI*2); mapCtx.stroke();
      mapCtx.setLineDash([]);
      // Small warning badge
      mapCtx.fillStyle='#ff2222';
      mapCtx.font='bold 10px sans-serif';
      mapCtx.textAlign='center'; mapCtx.textBaseline='middle';
      mapCtx.beginPath(); mapCtx.arc(sx+12,sy-12,7,0,Math.PI*2); mapCtx.fill();
      mapCtx.fillStyle='#fff';
      mapCtx.fillText('!',sx+12,sy-12);
    }

    // Label - always show for ego, show for others when selected or nearby
    const showLabel = isEgo || sel || (vt.s > 2);
    if(showLabel){
      const labelSize=isEgo?12:(sel?11:9);
      mapCtx.font=`bold ${labelSize}px monospace`;
      mapCtx.textAlign='center'; mapCtx.textBaseline='bottom';

      let label;
      if(isEgo){
        label='EGO '+egoIdx;
      } else {
        label=(ROLE_ICON[tr.role]||'')+' '+tr.id;
      }
      const tw=mapCtx.measureText(label).width;
      const lx=sx, ly=sy-(sel?18:(isEgo?16:10));

      // Background pill
      mapCtx.fillStyle=isEgo?'rgba(0,0,80,0.8)':'rgba(0,0,0,0.7)';
      const pillH=labelSize+4, pillW=tw+10;
      mapCtx.beginPath();
      mapCtx.roundRect(lx-pillW/2, ly-pillH+2, pillW, pillH, 3);
      mapCtx.fill();
      if(isEgo){
        mapCtx.strokeStyle='#4488ff';
        mapCtx.lineWidth=1;
        mapCtx.beginPath();
        mapCtx.roundRect(lx-pillW/2, ly-pillH+2, pillW, pillH, 3);
        mapCtx.stroke();
      }

      mapCtx.fillStyle=isEgo?'#ffcc44':(sel?'#fff':col);
      mapCtx.fillText(label,lx,ly);

      // Show type/role subtitle for non-ego
      if(!isEgo && (sel || vt.s > 3)){
        const sub=tr.obj_type&&tr.obj_type!==tr.role?tr.obj_type:tr.role;
        mapCtx.font='9px monospace';
        mapCtx.fillStyle='#888';
        mapCtx.fillText(sub, lx, ly+11);
      }
    }
  }

  mapCtx.restore();

  // Update overlay
  const egoName = egoTracks[followEgoIdx];
  document.getElementById('ov-ego').textContent =
    egoName ? 'EGO '+followEgoIdx+' ('+egoName.id+')' : 'No ego selected';
  document.getElementById('ov-time').textContent =
    't = '+curT.toFixed(2)+'s' + (followMode?' [FOLLOWING]':'');
}

function redrawTimeline(){
  const W=tlEl.offsetWidth, H=36;
  tlEl.width=W*dpr; tlEl.height=H*dpr;
  tlEl.style.width=W+'px'; tlEl.style.height=H+'px';
  tlCtx.save(); tlCtx.scale(dpr,dpr);
  tlCtx.fillStyle='#161620'; tlCtx.fillRect(0,0,W,H);

  const rng=tMax-tMin||1;
  const t2x=t=>(t-tMin)/rng*W;

  // Track bar background
  tlCtx.fillStyle='rgba(68,136,255,0.08)';
  tlCtx.fillRect(0,0,W,H);

  // CCLI change ticks
  if(selId&&trackById[selId]){
    tlCtx.fillStyle='#f80';
    for(const ct of (trackById[selId].ccli_changes||[])){
      tlCtx.fillRect(t2x(ct)-1,0,2,H);
    }
  }

  // Time grid
  tlCtx.strokeStyle='rgba(255,255,255,0.05)';
  tlCtx.lineWidth=1;
  const step=rng>20?5:rng>5?1:0.5;
  for(let t=Math.ceil(tMin/step)*step;t<=tMax;t+=step){
    const x=t2x(t);
    tlCtx.beginPath(); tlCtx.moveTo(x,0); tlCtx.lineTo(x,H); tlCtx.stroke();
    tlCtx.fillStyle='#444'; tlCtx.font='9px monospace';
    tlCtx.textAlign='center'; tlCtx.fillText(t.toFixed(1),x,H-2);
  }

  // Progress fill
  const cx=t2x(curT);
  tlCtx.fillStyle='rgba(68,136,255,.12)'; tlCtx.fillRect(0,0,cx,H);

  // Cursor line
  tlCtx.fillStyle='#5af'; tlCtx.fillRect(cx-1.5,0,3,H);

  // Cursor time label
  tlCtx.fillStyle='#fff'; tlCtx.font='bold 11px monospace';
  tlCtx.textAlign='center';
  tlCtx.fillText('t='+curT.toFixed(2)+'s',cx,12);

  tlCtx.restore();
}

function redrawAll(){ redrawMap(); redrawTimeline(); }

// ============================================================
// Mouse: map canvas
// ============================================================
mapEl.addEventListener('mousedown',e=>{
  const rect=mapEl.getBoundingClientRect();
  const cx=e.clientX-rect.left, cy=e.clientY-rect.top;
  const [wx,wy]=c2w(cx,cy);

  // Waypoint tool: click/shift-click/box-select/drag
  if(tool==='wp' && e.button===0){
    const wp=hitWaypoint(wx,wy);
    if(wp){
      const tr=trackById[wp.trackId];
      const dragWholeStatic=isStaticVehicleTrack(tr);
      // Shift-click toggles selection; plain click sets single selection.
      // For static vehicles, dragging any waypoint moves the whole actor.
      if(dragWholeStatic){
        wpSelected.clear();
        for(let i=0;i<tr.frames.length;i++) wpSelected.add(i);
      } else if(e.shiftKey){
        if(wpSelected.has(wp.frameIdx)) wpSelected.delete(wp.frameIdx);
        else wpSelected.add(wp.frameIdx);
      } else if(!wpSelected.has(wp.frameIdx)){
        wpSelected.clear();
        wpSelected.add(wp.frameIdx);
      }
      // Start dragging all selected waypoints
      wpDragging=true;
      wpDragIdx=wp.frameIdx;
      wpDragStart={x:wx,y:wy};
      wpDragUndoPending=true;
      wpDragChanged=false;
      // Store original positions for all selected waypoints
      wpDragOrig={};
      for(const idx of wpSelected){
        const ep=getEffectivePos(tr,idx);
        if(ep) wpDragOrig[idx]={x:ep[0],y:ep[1]};
      }
      mapEl.style.cursor='grabbing';
      e.preventDefault();
      redrawAll();
      return;
    }
    // No waypoint hit — start rubber-band box selection
    if(!e.shiftKey) wpSelected.clear();
    wpBoxSel=true;
    wpBoxStart={x:wx,y:wy};
    wpBoxEnd={x:wx,y:wy};
    e.preventDefault();
    redrawAll();
    return;
  }

  mDown=true; mDragging=false;
  mStart={x:e.clientX,y:e.clientY};
  vtSnap={tx:vt.tx,ty:vt.ty};
  e.preventDefault();
});
mapEl.addEventListener('mousemove',e=>{
  const rect=mapEl.getBoundingClientRect();
  const cx=e.clientX-rect.left, cy=e.clientY-rect.top;
  const [wx,wy]=c2w(cx,cy);

  // Rubber-band box selection
  if(wpBoxSel){
    wpBoxEnd={x:wx,y:wy};
    redrawMap();
    return;
  }

  // Waypoint dragging (multi-waypoint)
  if(wpDragging && selId && trackById[selId]){
    const tr=trackById[selId];
    // Compute world-space delta from drag start
    const ddx=wx-wpDragStart.x;
    const ddy=wy-wpDragStart.y;
    const dNorm=Math.hypot(ddx,ddy);
    if(dNorm<=1e-6){
      redrawMap();
      return;
    }

    const curOv=getOv(selId);
    if(wpDragUndoPending){
      pushUndo();
      wpDragUndoPending=false;
    }
    wpDragChanged=true;

    for(const idx of wpSelected){
      const orig=wpDragOrig[idx];
      if(!orig) continue;

      // New world position
      const newWx=orig.x+ddx;
      const newWy=orig.y+ddy;

      // Get base position (with lane snap but without waypoint override)
      const base=getLaneBasePos(tr, idx, curOv);
      if(!base) continue;
      setWaypointOverrideDelta(curOv, idx, newWx-base[0], newWy-base[1]);
    }
    redrawMap();
    return;
  }

  if(mDown){
    const dx=e.clientX-mStart.x, dy=e.clientY-mStart.y;
    if(!mDragging&&(Math.abs(dx)>3||Math.abs(dy)>3)) mDragging=true;
    if(mDragging){
      // Pan with any button if alt is held, or middle/right button
      if(e.buttons===4||e.altKey||e.buttons===2){
        vt.tx=vtSnap.tx+dx; vt.ty=vtSnap.ty+dy;
        if(followMode){ followMode=false; document.getElementById('btn-follow').classList.remove('on'); viewRotation=0; }
        redrawAll();
      }
    }
  }
  if(tool==='lane'){
    const h=hitLine(wx,wy);
    if(h!==hovLine){ hovLine=h; redrawMap(); }
  }
  if(tool==='wp'&&!wpBoxSel){
    const wp=hitWaypoint(wx,wy);
    mapEl.style.cursor=wp?'grab':'crosshair';
  }
});
mapEl.addEventListener('mouseup',e=>{
  // Rubber-band box selection end
  if(wpBoxSel){
    wpBoxSel=false;
    if(selId&&trackById[selId]&&wpBoxStart&&wpBoxEnd){
      const tr=trackById[selId];
      const bx1=Math.min(wpBoxStart.x,wpBoxEnd.x), bx2=Math.max(wpBoxStart.x,wpBoxEnd.x);
      const by1=Math.min(wpBoxStart.y,wpBoxEnd.y), by2=Math.max(wpBoxStart.y,wpBoxEnd.y);
      // Only count as box if it's bigger than a few pixels
      if(Math.abs(bx2-bx1)*vt.s>5 || Math.abs(by2-by1)*vt.s>5){
        for(let i=0;i<tr.frames.length;i++){
          const ep=getEffectivePos(tr,i);
          if(!ep) continue;
          if(ep[0]>=bx1&&ep[0]<=bx2&&ep[1]>=by1&&ep[1]<=by2){
            wpSelected.add(i);
          }
        }
      }
    }
    wpBoxStart=null; wpBoxEnd=null;
    if(wpSelected.size>0) status(wpSelected.size+' waypoints selected \u2014 drag to move, Delete to remove');
    redrawAll();
    return;
  }

  // Waypoint drag end
  if(wpDragging){
    wpDragging=false;
    wpDragIdx=-1;
    wpDragOrig=null;
    wpDragUndoPending=false;
    mapEl.style.cursor=tool==='wp'?'grab':'crosshair';
    if(wpDragChanged && selId){
      const smoothInfo=smoothWaypointEdits(selId);
      wpDragChanged=false;
      updatePatchSummary();
      redrawAll();
      if(smoothInfo){
        status(
          wpSelected.size+' waypoint(s) adjusted \u2014 smoothed t='+
          smoothInfo.duration.toFixed(2)+'s @ '+smoothInfo.avgSpeed.toFixed(2)+
          ' m/s (window '+smoothInfo.firstIdx+'\u2192'+smoothInfo.lastIdx+')'
        );
      } else {
        status(wpSelected.size+' waypoint(s) adjusted \u2014 Ctrl+Z to undo');
      }
    } else {
      wpDragChanged=false;
      redrawAll();
    }
    return;
  }

  if(!mDragging&&e.button===0){
    const rect=mapEl.getBoundingClientRect();
    const [wx,wy]=c2w(e.clientX-rect.left,e.clientY-rect.top);
    if(tool==='sel'){
      let hit=hitActor(wx,wy);
      if(!hit) hit=hitTrajectory(wx,wy);
      selectActor(hit, e.shiftKey);
    } else if(tool==='lane'){
      const li=hitLine(wx,wy);
      if(li!=null) doLaneSnap(li);
      else status('No CARLA line at cursor \u2014 move closer and try again');
    } else if(tool==='wp'){
      // Click without drag on non-waypoint = select actor
      let hit=hitActor(wx,wy);
      if(!hit) hit=hitTrajectory(wx,wy);
      if(hit) selectActor(hit, e.shiftKey);
    }
  }
  mDown=false; mDragging=false;
});
mapEl.addEventListener('wheel',e=>{
  e.preventDefault();
  const f=e.deltaY<0?1.15:1/1.15;
  const rect=mapEl.getBoundingClientRect();
  const cx=e.clientX-rect.left, cy=e.clientY-rect.top;
  const [wx,wy]=c2w(cx,cy);
  vt.s*=f;
  // Recompute so the point under cursor stays fixed
  const c=Math.cos(viewRotation), s=Math.sin(viewRotation);
  const dx=wx*vt.s, dy=-wy*vt.s;
  const rx=dx*c-dy*s, ry=dx*s+dy*c;
  vt.tx=cx-rx; vt.ty=cy-ry;
  redrawAll();
},{passive:false});
mapEl.addEventListener('contextmenu',e=>e.preventDefault());

// ============================================================
// Timeline mouse
// ============================================================
function tlT(e){
  const r=tlEl.getBoundingClientRect();
  return tMin+Math.max(0,Math.min(1,(e.clientX-r.left)/r.width))*(tMax-tMin);
}
tlEl.addEventListener('mousedown',e=>{
  stopPlay();
  setCurT(tlT(e));
});
tlEl.addEventListener('mousemove',e=>{if(e.buttons===1) setCurT(tlT(e));});

function setCurT(t){
  curT=Math.max(tMin,Math.min(tMax,t));
  if(followMode) applyFollow();
  updateLaneSnapUi();
  updateCam(); redrawTimeline(); redrawMap();
}

// ============================================================
// Camera
// ============================================================
function _drainCamPending(){
  if(camPendingT === null) return;
  const t = camPendingT;
  camPendingT = null;
  // Only fetch if the queued time is still meaningful (still close to curT)
  if(Math.abs(t - curT) > 0.01){ /* user moved on; latest curT will be served */ }
  updateCam();
}
function _setCamImageFromBlob(blob, loadT){
  const img = document.getElementById('cam-img');
  const url = URL.createObjectURL(blob);
  // Revoke previous blob URL once the new one renders, to avoid memory leak
  const prev = camLastBlobUrl;
  img.onload = ()=>{
    if(prev && prev !== url) URL.revokeObjectURL(prev);
    img.onload = null;
  };
  img.src = url;
  camLastBlobUrl = url;
  document.getElementById('cam-info').textContent =
    't='+loadT.toFixed(3)+'s  |  Ego '+camIdx;
}

function updateCam(){
  document.getElementById('cam-label').textContent='Camera \u2014 Ego '+camIdx;

  // Serialise requests: queue the latest target time and bail if one is in flight.
  if(camLoading){ camPendingT = curT; return; }
  // Playback throttling: ~8 cam fps in sim time
  if(playing && Math.abs(curT-camLastT)<0.12) return;

  const loadT = curT;
  camLastT = loadT;
  camLoading = true;
  const reqId = ++camReqId;
  const src = '/frame?t='+loadT.toFixed(3)+'&cam='+camIdx;

  // Watchdog: if a fetch hangs (browser/network glitch), force-reset after 4 s
  // so future updates aren't blocked forever.
  const watchdog = setTimeout(()=>{
    if(reqId === camReqId && camLoading){
      camLoading = false;
      console.warn('[cam] watchdog fired after 4s for t='+loadT);
      _drainCamPending();
    }
  }, 4000);

  fetch(src)
    .then(r=>{
      if(r.status === 204) return null;       // no frame at this t
      if(!r.ok) throw new Error('HTTP '+r.status);
      return r.blob();
    })
    .then(blob=>{
      if(reqId !== camReqId) return;          // a newer request superseded us
      if(blob){
        _setCamImageFromBlob(blob, loadT);
      }else{
        document.getElementById('cam-info').textContent =
          'no frame for ego '+camIdx+' at t='+loadT.toFixed(3)+'s';
      }
    })
    .catch(err=>{
      if(reqId !== camReqId) return;
      document.getElementById('cam-info').textContent =
        'cam error: '+(err && err.message || err);
    })
    .finally(()=>{
      clearTimeout(watchdog);
      if(reqId === camReqId){
        camLoading = false;
        _drainCamPending();
      }
    });
}

async function loadCamSubdirs(){
  const r=await fetch('/cam_subdirs');
  const list=await r.json();
  _camSubdirCount=list.length;
}

// ============================================================
// Actor list + selection
// ============================================================
function selectActor(id, shift=false){
  if(shift && id){
    // Multi-select: toggle the actor in the selection set
    if(selIds.has(id)){
      selIds.delete(id);
      // If we removed the primary, pick another or null
      if(selId===id){
        selId=selIds.size>0?[...selIds][selIds.size-1]:null;
      }
    } else {
      selIds.add(id);
      selId=id;   // make the newly-added actor the primary
    }
  } else {
    // Single select: replace entire selection
    selIds.clear();
    selId=id;
    if(id) selIds.add(id);
  }

  if(selId&&trackById[selId]){
    const tr=trackById[selId];
    tMin=tr.t_start; tMax=tr.t_end;
    const tlLbl=document.getElementById('tl-lbl');
    const multiTag=selIds.size>1?' <span style="color:#ff8844">[+' +(selIds.size-1)+' more]</span>':'';
    if(tr.role==='ego'){
      const egoIdx=egoTracks.indexOf(tr);
      tlLbl.innerHTML='<span class="actor-name">EGO '+egoIdx+'</span> '+
        tr.id+' | '+tr.frames.length+' frames | '+
        tr.t_start.toFixed(2)+'\u2013'+tr.t_end.toFixed(2)+'s'+multiTag;
    } else {
      tlLbl.innerHTML='<span class="actor-name">'+tr.id+'</span> ['+tr.role+'] | '+
        tr.frames.length+' frames | '+
        tr.t_start.toFixed(2)+'\u2013'+tr.t_end.toFixed(2)+'s'+multiTag;
    }
    curT=Math.max(tMin,Math.min(tMax,curT));
    // Auto-switch camera for ego vehicles
    if(tr.role==='ego'){
      const egoIdx=egoTracks.indexOf(tr);
      if(egoIdx>=0&&egoIdx<_camSubdirCount){
        camIdx=egoIdx;
        // Update ego selector too
        const egoSel=document.getElementById('ego-sel');
        if(egoSel) egoSel.value=egoIdx;
        followEgoIdx=egoIdx;
        // Force camera refresh: reset threshold so updateCam() fires immediately
        // regardless of when the last frame was shown (prevents ego-switch being dropped)
        camLastT=-999;
      }
    }
    if(followMode) applyFollow();
  } else {
    document.getElementById('tl-lbl').innerHTML='<span>No actor selected</span>';
  }
  updateLaneSnapUi();
  rebuildActorList(); updatePatchSummary(); redrawAll(); updateCam();
}

function rebuildActorList(){
  const list=document.getElementById('actor-list');
  list.innerHTML='';
  const olap=computeOverlaps();
  for(const tr of tracks){
    const div=document.createElement('div');
    const del=isDeleted(tr.id), pat=isPatched(tr.id);
    const isEgo=tr.role==='ego';
    const egoIdx=isEgo?egoTracks.indexOf(tr):-1;
    const isSel=selIds.has(tr.id);
    const isOlap=olap.has(tr.id);
    div.className='ai'+(isSel?' sel':'')+(del?' del':pat?' pat':'')+(isEgo?' ego-item':'');
    if(isOlap&&!del) div.style.borderLeftColor='#ff2222';

    if(isEgo){
      div.textContent='\u2605 EGO '+egoIdx+' ('+tr.id+')'+(del?' \u2715':pat?' \u25c6':'')+(isOlap?' \u26a0':'');
    } else {
      const typeTag=tr.obj_type&&tr.obj_type!==tr.role?' ('+tr.obj_type+')':'';
      div.textContent=(ROLE_ICON[tr.role]||'\u25cf')+' '+tr.id+
                      ' ['+tr.role+']'+typeTag+(del?' \u2715':pat?' \u25c6':'')+(isOlap?' \u26a0':'');
    }
    div.addEventListener('click',(ev)=>selectActor(tr.id, ev.shiftKey));
    list.appendChild(div);
  }
}

function updatePatchSummary(){
  const el=document.getElementById('patch-sum');
  if(!selId){ el.textContent='Select an actor to see patches'; return; }
  const ov=ovById[selId];
  if(!ov||!isPatched(selId)){ el.textContent='No patches on this actor\nUse tools above to edit'; return; }
  const lines=[];
  if(ov.delete) lines.push('\u2715 DELETED');
  if(ov.snap_to_outermost) lines.push('\u2192 Snap to outermost lane');
  if(ov.phase_override) lines.push('\u21bb Phase: '+ov.phase_override);
  for(const ls of (ov.lane_segment_overrides||[])){
    const st=Number(ls.start_t), et=Number(ls.end_t);
    const t0=ls.start_t!=null&&isFinite(st)?st.toFixed(2):'start';
    const t1=ls.end_t  !=null&&isFinite(et)?et.toFixed(2):'end';
    const blend=isFinite(Number(ls.blend_in_s))&&Number(ls.blend_in_s)>0
      ? ' blend='+Number(ls.blend_in_s).toFixed(2)+'s' : '';
    lines.push('\u229e Lane snap \u2192 line '+ls.lane_id+' ['+t0+'\u2013'+t1+']'+blend);
  }
  const wps=(ov.waypoint_overrides||[]);
  if(wps.length){
    lines.push('\u2022 '+wps.length+' waypoint adjustment(s)');
  }
  el.textContent=lines.join('\n');
}

// ============================================================
// Fit all
// ============================================================
function fitAll(){
  viewRotation=0;
  let xMn=Infinity,xMx=-Infinity,yMn=Infinity,yMx=-Infinity;
  for(const tr of tracks)
    for(const f of tr.frames){
      const x=fx(f),y=fy(f);
      if(x==null||y==null||!isFinite(x)||!isFinite(y)) continue;
      if(x<xMn)xMn=x; if(x>xMx)xMx=x;
      if(y<yMn)yMn=y; if(y>yMx)yMx=y;
    }
  if(!isFinite(xMn)) return;
  const rX=xMx-xMn||100, rY=yMx-yMn||100, mg=50;
  vt.s=Math.min((cW-2*mg)/rX,(cH-2*mg)/rY);
  vt.tx=cW/2-((xMn+xMx)/2)*vt.s;
  vt.ty=cH/2+((yMn+yMx)/2)*vt.s;
  if(followMode){ followMode=false; document.getElementById('btn-follow').classList.remove('on'); }
  redrawAll();
}

// ============================================================
// Waypoint multi-select & delete with smooth fill
// ============================================================

function wpSelectAll(){
  if(!selId||!trackById[selId]) return;
  const tr=trackById[selId];
  wpSelected.clear();
  for(let i=0;i<tr.frames.length;i++) wpSelected.add(i);
  status(wpSelected.size+' waypoints selected');
  redrawAll();
}

// Catmull-Rom spline interpolation between control points
function catmullRomInterp(points, numOut){
  // points: array of {x,y}  (at least 2)
  // numOut: number of output points
  if(points.length<2) return points.map(p=>({x:p.x,y:p.y}));
  if(points.length===2){
    // Linear interpolation
    const out=[];
    for(let i=0;i<numOut;i++){
      const t=numOut>1?i/(numOut-1):0;
      out.push({x:points[0].x+(points[1].x-points[0].x)*t,
                y:points[0].y+(points[1].y-points[0].y)*t});
    }
    return out;
  }

  // Build extended control points (duplicate endpoints)
  const cp=[points[0], ...points, points[points.length-1]];
  const nSeg=cp.length-3;   // number of segments
  const totalLen=[];
  let cumLen=0;
  // Compute approximate arc length for each segment
  for(let s=0;s<nSeg;s++){
    const p0=cp[s],p1=cp[s+1],p2=cp[s+2],p3=cp[s+3];
    const dx=p2.x-p1.x, dy=p2.y-p1.y;
    const len=Math.sqrt(dx*dx+dy*dy)||1e-6;
    cumLen+=len;
    totalLen.push(cumLen);
  }

  const out=[];
  for(let i=0;i<numOut;i++){
    const u=numOut>1?(i/(numOut-1))*cumLen:0;
    // Find which segment
    let seg=0;
    for(let s=0;s<nSeg;s++){
      if(u<=(totalLen[s]||0)+1e-9){ seg=s; break; }
      if(s===nSeg-1) seg=s;
    }
    const segStart=seg>0?totalLen[seg-1]:0;
    const segLen=(totalLen[seg]||1)-segStart;
    const t=Math.max(0,Math.min(1,(u-segStart)/segLen));

    const p0=cp[seg], p1=cp[seg+1], p2=cp[seg+2], p3=cp[seg+3];
    const t2=t*t, t3=t2*t;
    // Catmull-Rom basis (tension=0.5)
    const x=0.5*((2*p1.x)+(-p0.x+p2.x)*t+(2*p0.x-5*p1.x+4*p2.x-p3.x)*t2+
                 (-p0.x+3*p1.x-3*p2.x+p3.x)*t3);
    const y=0.5*((2*p1.y)+(-p0.y+p2.y)*t+(2*p0.y-5*p1.y+4*p2.y-p3.y)*t2+
                 (-p0.y+3*p1.y-3*p2.y+p3.y)*t3);
    out.push({x,y});
  }
  return out;
}

function wpDeleteSelected(){
  if(!selId||!trackById[selId]||wpSelected.size===0) return;
  const tr=trackById[selId];
  const totalFrames=tr.frames.length;
  const selIndices=[...wpSelected].sort((a,b)=>a-b);

  // Can't delete ALL waypoints
  if(selIndices.length>=totalFrames){
    status('Cannot delete all waypoints \u2014 keep at least 2');
    return;
  }

  pushUndo();
  const curOv=getOv(selId);
  if(!curOv.waypoint_overrides) curOv.waypoint_overrides=[];

  // Find contiguous runs of deleted indices
  const runs=[];
  let runStart=selIndices[0], runEnd=selIndices[0];
  for(let i=1;i<selIndices.length;i++){
    if(selIndices[i]===runEnd+1){ runEnd=selIndices[i]; }
    else { runs.push([runStart,runEnd]); runStart=selIndices[i]; runEnd=selIndices[i]; }
  }
  runs.push([runStart,runEnd]);

  // For each run, find anchor points (the non-deleted neighbours) and
  // generate smooth interpolated waypoint overrides for the gap
  for(const [rStart,rEnd] of runs){
    // Collect anchor control points: 2 before gap + 2 after gap (for spline quality)
    const anchors=[];
    // Points before the gap
    for(let i=Math.max(0,rStart-2);i<rStart;i++){
      if(!wpSelected.has(i)){
        const ep=getEffectivePos(tr,i);
        if(ep){ anchors.push({x:ep[0],y:ep[1]}); }
      }
    }
    // Points after the gap
    for(let i=rEnd+1;i<=Math.min(totalFrames-1,rEnd+3);i++){
      if(!wpSelected.has(i)){
        const ep=getEffectivePos(tr,i);
        if(ep){ anchors.push({x:ep[0],y:ep[1]}); }
      }
    }

    if(anchors.length<2){
      // Not enough anchors — just set deleted waypoints to match nearest anchor
      for(let i=rStart;i<=rEnd;i++){
        const ep=anchors.length>0?anchors[0]:getEffectivePos(tr,i);
        if(!ep) continue;
        // Set as zero override (effectively remove any existing override)
        _setWpOverride(curOv,tr,i,ep[0]??ep.x,ep[1]??ep.y);
      }
      continue;
    }

    // Interpolate smooth path for the gap indices
    const gapCount=rEnd-rStart+1;
    // We want to interpolate between first anchor and last anchor, with the gap in order
    const interpPts=catmullRomInterp(anchors, gapCount+2);
    // interpPts[0] = first anchor, interpPts[last] = last anchor
    // Take the middle points for the gap
    for(let g=0;g<gapCount;g++){
      const idx=rStart+g;
      const pt=interpPts[g+1];   // skip first anchor point
      _setWpOverride(curOv,tr,idx,pt.x,pt.y);
    }
  }

  wpSelected.clear();
  updatePatchSummary();
  redrawAll();
  status(selIndices.length+' waypoint(s) deleted and smoothly filled \u2014 Ctrl+Z to undo');
}

function _setWpOverride(ov, tr, frameIdx, worldX, worldY){
  // Compute the override delta relative to base position
  const base=getLaneBasePos(tr, frameIdx, ov);
  if(!base) return;
  setWaypointOverrideDelta(ov, frameIdx, worldX-base[0], worldY-base[1]);
}

// ============================================================
// Keyboard
// ============================================================
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='SELECT'||e.target.tagName==='INPUT') return;
  const k=e.key.toLowerCase();
  if(k===' '){  e.preventDefault(); playPause(); }
  else if(k==='f'){ e.preventDefault(); toggleFollow(); }
  else if(k==='s'&&!e.ctrlKey){ e.preventDefault(); setTool('sel'); }
  else if(k==='l'){ e.preventDefault(); setTool('lane'); }
  else if(k==='w'){ e.preventDefault(); setTool('wp'); }
  else if(k==='d'&&tool!=='wp'){ e.preventDefault(); doDelete(); }
  else if(k==='o'){ e.preventDefault(); doOuter(); }
  else if(k==='p'){ e.preventDefault(); doPhase(); }
  else if(k==='a'&&!e.ctrlKey){ e.preventDefault(); 
    if(tool==='wp'&&selId) wpSelectAll(); else fitAll(); }
  else if(e.ctrlKey&&k==='a'&&tool==='wp'){ e.preventDefault(); wpSelectAll(); }
  else if(e.ctrlKey&&k==='s'){ e.preventDefault(); doSave(); }
  else if(e.ctrlKey&&k==='z'){ e.preventDefault(); doUndo(); }
  else if(e.ctrlKey&&k==='y'){ e.preventDefault(); doRedo(); }
  else if(k==='arrowright'){ e.preventDefault(); setCurT(curT+playDT); }
  else if(k==='arrowleft'){ e.preventDefault(); setCurT(curT-playDT); }
  else if((k==='delete'||k==='backspace')&&tool==='wp'){
    e.preventDefault(); wpDeleteSelected();
  }
  else if(k==='tab'){
    e.preventDefault();
    const idx=tracks.findIndex(t=>t.id===selId);
    const nx=e.shiftKey?idx-1:idx+1;
    if(nx>=0&&nx<tracks.length) selectActor(tracks[nx].id);
  }
  // Number keys 0-9 to quickly select ego
  else if(k>='0'&&k<='9'&&!e.ctrlKey){
    const n=parseInt(k);
    if(n<egoTracks.length){
      e.preventDefault();
      document.getElementById('ego-sel').value=n;
      onEgoChange(n);
    }
  }
});

// ============================================================
// Resize
// ============================================================
function onResize(){
  dpr=window.devicePixelRatio||1;
  const wrap=document.getElementById('map-wrap');
  cW=wrap.clientWidth; cH=wrap.clientHeight;
  mapEl.style.width=cW+'px'; mapEl.style.height=cH+'px';
  mapEl.width=cW*dpr; mapEl.height=cH*dpr;
  redrawAll();
}
new ResizeObserver(onResize).observe(document.getElementById('map-wrap'));

// ============================================================
// Status
// ============================================================
function status(msg){ document.getElementById('status').textContent=msg; }

// ============================================================
// Init
// ============================================================
async function init(){
  status('Loading dataset\u2026');
  let data;
  try{
    const dr=await fetch('/data');
    if(!dr.ok){ status('Failed to load /data: HTTP '+dr.status); return; }
    data=await dr.json();
    console.log('[init] data loaded:',data.tracks.length,'tracks',data.carla_lines.length,'lines');
  }catch(e){
    status('JSON parse error \u2014 check browser console: '+e.message);
    console.error('[init] /data parse failed:',e);
    return;
  }

  tracks=data.tracks; carlaLines=data.carla_lines;
  trackById={}; for(const t of tracks) trackById[t.id]=t;
  lineByIdx={}; for(const l of carlaLines) lineByIdx[l.idx]=l;
  laneChainCache={};  // clear lane chain cache on reload
  _loadBgImageFromData(data);

  // Sort: ego first, then vehicle/cyclist/walker
  const ORD={ego:0,vehicle:1,cyclist:2,walker:3};
  tracks.sort((a,b)=>(ORD[a.role]??9)-(ORD[b.role]??9)||a.id.localeCompare(b.id));

  // Identify ego tracks
  egoTracks=tracks.filter(t=>t.role==='ego');
  buildEgoSelector();

  // Set initial time range from all tracks
  tMin=Infinity; tMax=-Infinity;
  for(const tr of tracks){
    if(tr.t_start<tMin) tMin=tr.t_start;
    if(tr.t_end>tMax) tMax=tr.t_end;
  }
  if(!isFinite(tMin)){ tMin=0; tMax=10; }

  // Load existing patch
  try{
    const pr=await fetch('/patch');
    if(pr.ok) applyPatch(await pr.json());
  }catch(_){}

  await loadCamSubdirs();
  onResize();
  fitAll();
  rebuildActorList();
  syncUndoBtns();
  const blendEl=document.getElementById('lane-blend');
  if(blendEl&&!isFinite(parseFloat(blendEl.value))){
    blendEl.value=String(LANE_SNAP_BLEND_DEFAULT);
  }
  updateLaneSnapUi();

  // Auto-select first ego
  if(egoTracks.length>0){
    selectActor(egoTracks[0].id);
  }

  status(data.scenario_name+'  |  '+tracks.length+' actors  |  '+
         egoTracks.length+' ego vehicles  |  '+carlaLines.length+' CARLA lines  |  '+
         'Space=Play  F=Follow  W=Waypoint  L=LaneSnap');
  updateCam();

  // Init batch mode UI
  await initBatch();
}

// ============================================================
// Batch Mode
// ============================================================
let batchEnabled=false, batchTotal=0, batchCurrent=0, batchName='';
let _batchScenarioList=[];   // latest scenario_list from server

async function initBatch(){
  try{
    const r=await fetch('/batch_status');
    if(!r.ok) return;
    const bs=await r.json();
    batchEnabled=bs.enabled;
    if(!batchEnabled) return;

    batchTotal=bs.total;
    batchCurrent=bs.current;
    batchName=bs.current_name;

    document.getElementById('batch-bar').classList.add('active');
    // Show Scenarios tab when batch is active
    document.getElementById('rtab-scenarios').style.display='';
    updateBatchUI(bs);

    // Start CARLA health polling
    document.getElementById('carla-bar').classList.add('active');
    pollCarla();
    setInterval(pollCarla, 5000);

    // Start batch status polling (replaces old bg task polling)
    setInterval(pollBatchStatus, 3000);
  }catch(e){
    console.warn('[batch] init failed:',e);
  }
}

function switchRightTab(tab){
  const actorsPanel=document.getElementById('actors-panel');
  const scenariosPanel=document.getElementById('scenarios-panel');
  const tabActors=document.getElementById('rtab-actors');
  const tabScenarios=document.getElementById('rtab-scenarios');
  if(tab==='scenarios'){
    actorsPanel.classList.remove('active');
    scenariosPanel.classList.add('active');
    tabActors.classList.remove('on');
    tabScenarios.classList.add('on');
  }else{
    scenariosPanel.classList.remove('active');
    actorsPanel.classList.add('active');
    tabScenarios.classList.remove('on');
    tabActors.classList.add('on');
  }
}

function _scenarioBadge(sc){
  // Returns {cls, label} for the scenario status badge
  const qs=sc.queue_status;
  if(qs==='running') return {cls:'running', label:'\u21bb running'};
  if(qs==='pending') return {cls:'queued',  label:'\u25b6 queued'};
  if(qs==='done')    return {cls:'done',    label:'\u2713 done'};
  if(qs==='error')   return {cls:'error',   label:'\u2717 error'};
  if(!sc.has_html)   return {cls:'nohtml',  label:'no html'};
  if(sc.has_patch)   return {cls:'patched', label:'\u25c6 patched'};
  return {cls:'ready', label:'\u25cb ready'};
}

function updateScenarioList(bs){
  _batchScenarioList=bs.scenario_list||[];
  const list=document.getElementById('scenario-list');
  if(!list) return;
  list.innerHTML='';
  for(const sc of _batchScenarioList){
    const div=document.createElement('div');
    div.className='si'+(sc.idx===batchCurrent?' cur':'');
    div.title=sc.name;

    const badge=_scenarioBadge(sc);
    const badgeEl=document.createElement('span');
    badgeEl.className='si-badge '+badge.cls;
    badgeEl.textContent=badge.label;

    const nameEl=document.createElement('span');
    nameEl.className='si-name';
    nameEl.textContent=sc.name;

    // Remove-from-queue button (only shown for pending items)
    const rmBtn=document.createElement('button');
    rmBtn.className='si-rm'+(sc.queue_status==='pending'?' visible':'');
    rmBtn.innerHTML='&#10005;';
    rmBtn.title='Remove from queue';
    rmBtn.onclick=(e)=>{ e.stopPropagation(); removeFromQueue(sc.name); };

    div.appendChild(badgeEl);
    div.appendChild(nameEl);
    div.appendChild(rmBtn);
    div.addEventListener('click',()=>batchGoto(sc.idx));
    list.appendChild(div);
  }
  // Scroll current item into view
  const curEl=list.querySelector('.si.cur');
  if(curEl) curEl.scrollIntoView({block:'nearest'});
}

function updateQueueSummary(bs){
  const queue=bs.carla_queue||[];
  const pending=queue.filter(q=>q.status==='pending').length;
  const running=queue.filter(q=>q.status==='running').length;
  const done   =queue.filter(q=>q.status==='done').length;
  const errors =queue.filter(q=>q.status==='error').length;

  const hasItems=queue.length>0;
  const parts=[];
  if(running) parts.push('\u21bb '+running+' running');
  if(pending) parts.push(pending+' pending');
  if(done)    parts.push('\u2713 '+done+' done');
  if(errors)  parts.push('\u2717 '+errors+' err');
  const txt=hasItems?('Queue: '+parts.join(' \u00b7 ')):'Queue: empty';

  // Update batch bar queue info button
  const qinfo=document.getElementById('batch-queue-info');
  if(qinfo){ qinfo.textContent=txt; qinfo.className=hasItems?'active':''; }

  // Update queue-summary in scenarios panel
  const qs=document.getElementById('queue-summary');
  if(qs){ qs.textContent=txt; qs.className=hasItems?'has-items':''; }
}

function updateBatchUI(bs){
  batchTotal=bs.total||batchTotal;
  batchCurrent=bs.current??batchCurrent;
  batchName=bs.current_name||batchName;

  document.getElementById('batch-counter').textContent=
    (batchCurrent+1)+' / '+batchTotal;
  document.getElementById('batch-name').textContent=batchName;

  // Disable prev on first
  document.getElementById('btn-prev').disabled=(batchCurrent<=0);

  updateScenarioList(bs);
  updateQueueSummary(bs);
}

async function batchNext(){
  if(!batchEnabled) return;
  if(playing) playPause();
  await doSave();
  status('Moving to next scenario...');
  document.getElementById('btn-next').disabled=true;
  try{
    const r=await fetch('/next',{method:'POST'});
    const data=await r.json();
    if(data.done){
      document.getElementById('done-overlay').classList.add('show');
      status('All scenarios complete!');
      return;
    }
    if(data.error){ status('Error: '+data.error); return; }
    updateBatchUI(data);
    await reloadEditorData();
  }catch(e){
    status('Error advancing scenario: '+e.message);
  }finally{
    document.getElementById('btn-next').disabled=false;
  }
}

async function batchSkip(){
  if(!batchEnabled) return;
  if(playing) playPause();
  status('Skipping scenario...');
  try{
    const r=await fetch('/skip',{method:'POST'});
    const data=await r.json();
    if(data.done){
      document.getElementById('done-overlay').classList.add('show');
      status('All scenarios complete!');
      return;
    }
    if(data.error){ status('Error: '+data.error); return; }
    updateBatchUI(data);
    await reloadEditorData();
  }catch(e){
    status('Error skipping: '+e.message);
  }
}

async function batchPrev(){
  if(!batchEnabled || batchCurrent<=0) return;
  if(playing) playPause();
  status('Going to previous scenario...');
  try{
    const r=await fetch('/goto',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({idx:batchCurrent-1})});
    const data=await r.json();
    if(data.error){ status('Error: '+data.error); return; }
    updateBatchUI(data);
    await reloadEditorData();
  }catch(e){
    status('Error: '+e.message);
  }
}

async function batchGoto(idx){
  if(!batchEnabled) return;
  if(idx===batchCurrent) return;
  if(playing) playPause();
  status('Switching to scenario '+idx+'...');
  try{
    const r=await fetch('/goto',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({idx})});
    const data=await r.json();
    if(data.error){ status('Error: '+data.error); return; }
    updateBatchUI(data);
    await reloadEditorData();
  }catch(e){
    status('Error: '+e.message);
  }
}

async function removeFromQueue(scenarioName){
  try{
    const r=await fetch('/queue_remove',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({scenario:scenarioName})});
    const data=await r.json();
    if(data.error){ status('Remove failed: '+data.error); return; }
    updateBatchUI(data);
    status('Removed from queue: '+scenarioName);
  }catch(e){
    status('Error: '+e.message);
  }
}

async function reloadEditorData(){
  // Reload scene data from server after scenario switch
  status('Loading new scenario...');
  try{
    const dr=await fetch('/data');
    if(!dr.ok){ status('Failed to load /data'); return; }
    const data=await dr.json();

    tracks=data.tracks; carlaLines=data.carla_lines;
    trackById={}; for(const t of tracks) trackById[t.id]=t;
    lineByIdx={}; for(const l of carlaLines) lineByIdx[l.idx]=l;
    laneChainCache={};  // clear lane chain cache on reload
    _loadBgImageFromData(data);

    const ORD={ego:0,vehicle:1,cyclist:2,walker:3};
    tracks.sort((a,b)=>(ORD[a.role]??9)-(ORD[b.role]??9)||a.id.localeCompare(b.id));
    egoTracks=tracks.filter(t=>t.role==='ego');
    buildEgoSelector();

    tMin=Infinity; tMax=-Infinity;
    for(const tr of tracks){
      if(tr.t_start<tMin) tMin=tr.t_start;
      if(tr.t_end>tMax) tMax=tr.t_end;
    }
    if(!isFinite(tMin)){ tMin=0; tMax=10; }
    curT=tMin;

    // Reset patch state
    patch={overrides:[]}; ovById={};
    undoStack=[]; redoStack=[];
    syncUndoBtns();

    // Load new patch
    try{
      const pr=await fetch('/patch');
      if(pr.ok) applyPatch(await pr.json());
    }catch(_){}

    await loadCamSubdirs();
    // Scenario switch: invalidate any in-flight or queued frame requests so the
    // new scenario's first updateCam() isn't dropped or merged with stale state.
    camLoading = false;
    camPendingT = null;
    camLastT = -999;
    camReqId++;
    fitAll();
    rebuildActorList();

    if(egoTracks.length>0) selectActor(egoTracks[0].id);

    onResize();
    updateCam();

    status(data.scenario_name+'  |  '+tracks.length+' actors  |  '+
           egoTracks.length+' ego  |  '+carlaLines.length+' lines  |  '+
           'Space=Play  F=Follow  W=Waypoint  L=LaneSnap');
  }catch(e){
    status('Reload error: '+e.message);
    console.error('[reload]',e);
  }
}

// ============================================================
// CARLA Health Monitor
// ============================================================
async function pollCarla(){
  try{
    const r=await fetch('/carla_status');
    if(!r.ok) return;
    const s=await r.json();
    const dot=document.getElementById('carla-dot');
    const txt=document.getElementById('carla-text');
    if(s.connected){
      dot.className='ok';
      txt.textContent='CARLA: '+s.host+':'+s.port+(s.map?' ('+s.map+')':'');
      txt.style.color='#8f8';
    }else{
      dot.className='err';
      txt.textContent='CARLA disconnected: '+(s.error||'unknown');
      txt.style.color='#f88';
    }
  }catch(_){}
}

async function pollBatchStatus(){
  if(!batchEnabled) return;
  try{
    const r=await fetch('/batch_status');
    if(!r.ok) return;
    const bs=await r.json();
    // Update scenario list badges and queue summary without overwriting batchCurrent
    updateScenarioList(bs);
    updateQueueSummary(bs);
  }catch(_){}
}

function showReconnect(){
  const modal=document.getElementById('reconnect-modal');
  modal.classList.add('show');
  // Pre-fill current values
  fetch('/carla_status').then(r=>r.json()).then(s=>{
    document.getElementById('rc-host').value=s.host||'localhost';
    document.getElementById('rc-port').value=s.port||2000;
  }).catch(_=>{});
}

function hideReconnect(){
  document.getElementById('reconnect-modal').classList.remove('show');
}

async function doReconnect(){
  const host=document.getElementById('rc-host').value.trim()||'localhost';
  const port=parseInt(document.getElementById('rc-port').value)||2000;
  try{
    const r=await fetch('/carla_reconnect',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({host,port})});
    const s=await r.json();
    if(s.connected){
      hideReconnect();
      pollCarla();
      status('CARLA reconnected to '+host+':'+port);
    }else{
      status('CARLA connection failed: '+(s.error||'unknown'));
    }
  }catch(e){
    status('Reconnect error: '+e.message);
  }
}

init().catch(e=>{ status('Init error: '+e.message); console.error('[init] uncaught:',e); });
</script>
</body>
</html>
""".encode("utf-8")


# ---------------------------------------------------------------------------
# Scenario discovery
# ---------------------------------------------------------------------------

def _is_scenario_directory(d: Path) -> bool:
    """
    A scenario directory contains numbered subdirs (e.g. 1/, 2/) with YAML files.
    Matches the pipeline convention.
    """
    if not d.is_dir():
        return False
    # Check for trajectory_plot.html or numbered subdirectories with YAML
    if (d / "trajectory_plot.html").exists():
        return True
    for child in d.iterdir():
        if child.is_dir():
            try:
                int(child.name)
                # Check if it has YAML / yaml files
                if list(child.glob("*.yaml")) or list(child.glob("*.yml")):
                    return True
            except ValueError:
                pass
    return False


def _find_scenario_directories(parent: Path) -> List[Path]:
    """Discover scenario subdirectories under `parent`."""
    results: List[Path] = []
    for child in sorted(parent.iterdir()):
        if child.is_dir() and _is_scenario_directory(child):
            results.append(child)
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Browser-based patch editor (single or batch mode)"
    )
    parser.add_argument(
        "input",
        help="Pipeline HTML file (single mode) OR scenario folder (batch mode)",
    )
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--patch", help="Patch JSON path (single mode only)")

    # Batch / pipeline options
    parser.add_argument(
        "--routes-out",
        help="Output directory for exported XML routes (batch mode)",
    )
    parser.add_argument(
        "--carla-host", default="localhost",
        help="CARLA server hostname (default: localhost)",
    )
    parser.add_argument(
        "--carla-port", type=int, default=2000,
        help="CARLA server port (default: 2000)",
    )

    # Pipeline forwarding args (optional)
    parser.add_argument("--map-pkl", action="append",
                        help="Map pickle file(s) for pipeline")
    parser.add_argument("--carla-map-cache",
                        help="CARLA map cache dir for pipeline")
    parser.add_argument("--carla-map-offset-json",
                        help="CARLA map offset JSON for pipeline")

    # Auto-export pipeline stage toggles (batch mode)
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help=(
            "Skip the final 'Custom eval' (in-CARLA log-replay smoke test) on save/auto-export. "
            "Use for visual-review-only batches where ego spawn issues block the eval."
        ),
    )
    parser.add_argument(
        "--no-grp-simplify",
        action="store_true",
        help=(
            "Skip the GRP ego XML simplification step on save/auto-export. "
            "Saves 30-120s per scenario; the _REPLAY xml is used by log-replay regardless."
        ),
    )

    args = parser.parse_args()

    global _scene_data, _patch, _patch_path

    input_path = Path(args.input).expanduser().resolve()

    # ---------- Determine mode ----------
    is_batch = input_path.is_dir()

    if is_batch:
        # ── Batch mode ──
        scenario_dirs = _find_scenario_directories(input_path)
        if not scenario_dirs:
            print(f"ERROR: No scenario directories found in {input_path}", file=sys.stderr)
            sys.exit(1)

        _batch.enabled = True
        _batch.scenario_dirs = scenario_dirs
        _batch.carla_host = args.carla_host
        _batch.carla_port = args.carla_port
        _batch.skip_eval = bool(args.no_eval)
        _batch.skip_grp_simplify = bool(args.no_grp_simplify)
        if _batch.skip_eval:
            print("Auto-export: Custom eval will be SKIPPED (--no-eval)")
        if _batch.skip_grp_simplify:
            print("Auto-export: GRP simplification will be SKIPPED (--no-grp-simplify)")

        if args.routes_out:
            _batch.routes_out_dir = Path(args.routes_out).expanduser().resolve()
            _batch.routes_out_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Default: <input_dir>/carla_routes_patched/
            _batch.routes_out_dir = input_path / "carla_routes_patched"
            _batch.routes_out_dir.mkdir(parents=True, exist_ok=True)
            print(f"Routes output (auto): {_batch.routes_out_dir}")

        # Store pipeline args for later use
        _batch.pipeline_args = args

        print(f"BATCH MODE: {len(scenario_dirs)} scenarios in {input_path.name}")
        for i, d in enumerate(scenario_dirs):
            has_html = (d / "trajectory_plot.html").exists()
            marker = " (HTML ready)" if has_html else " (needs pipeline)"
            print(f"  [{i}] {d.name}{marker}")

        # Load first scenario
        print(f"\nLoading first scenario: {scenario_dirs[0].name} ...", flush=True)
        ok = _activate_scenario(0)
        if not ok:
            print("ERROR: Failed to load first scenario", file=sys.stderr)
            sys.exit(1)

        # Start CARLA health monitor thread
        carla_thread = threading.Thread(
            target=_carla_health_loop, daemon=True, name="carla-health",
        )
        carla_thread.start()
        print(f"CARLA health monitor started ({args.carla_host}:{args.carla_port})")

    else:
        # ── Single file mode (original behavior) ──
        html_path = input_path
        if not html_path.exists():
            print(f"ERROR: File not found: {html_path}", file=sys.stderr)
            sys.exit(1)

        patch_path = (
            Path(args.patch).expanduser().resolve()
            if args.patch
            else PatchModel.patch_path_for(html_path)
        )
        _patch_path = patch_path

        print(f"Loading {html_path.name} …", flush=True)
        _scene_data = load_from_html(html_path)
        _discover_cam(html_path, _scene_data)

        if patch_path.exists():
            try:
                _patch = json.loads(patch_path.read_text())
                n = sum(1 for ov in _patch.get("overrides", []) if ov.get("delete") or
                        ov.get("snap_to_outermost") or ov.get("phase_override") or
                        ov.get("lane_segment_overrides"))
                print(f"Loaded patch: {n} overrides from {patch_path.name}", flush=True)
            except Exception as exc:
                print(f"[WARN] Could not load patch {patch_path}: {exc}", flush=True)

        cam_info = f"  Camera: {len(_cam_subdirs)} subdirs found" if _cam_subdirs else ""
        print(f"  {len(_scene_data.tracks)} actors, {len(_scene_data.carla_lines)} CARLA lines{cam_info}")

    # ── Start server ──
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), _Handler)
    actual_port = srv.server_address[1]

    print(f"\nEditor ready at:  http://localhost:{actual_port}/")
    print(f"SSH tunnel:       ssh -L {actual_port}:localhost:{actual_port} <user>@<server>")
    if is_batch:
        print(f"Batch:            {len(scenario_dirs)} scenarios, starting with {scenario_dirs[0].name}")
    print("Press Ctrl+C to stop.\n", flush=True)

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
