"""
loader.py  —  Load pipeline HTML output into SceneData + MapData.

Input: the HTML file written by runtime_orchestration.py which embeds
a <script id="dataset" type="application/json"> block.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CarlaLine:
    idx: int
    label: str
    pts: np.ndarray          # (N, 2) float64  world XY
    length: float            # arc length metres
    mid_x: float
    mid_y: float
    road_id: Optional[int]
    lane_id: Optional[int]
    dir_sign: Optional[int]  # +1 / -1 / None


@dataclass
class ActorFrame:
    t: float
    x: float
    y: float
    yaw: float               # raw heading degrees
    cx: float                # CARLA-snapped X
    cy: float                # CARLA-snapped Y
    cyaw: float              # CARLA-snapped heading
    ccli: int                # CARLA line index (-1 = unknown)
    csource: str


@dataclass
class ActorTrack:
    track_id: str            # "ego_0", "ego_1", "0", "1", ...
    role: str                # "ego" | "vehicle" | "walker" | "cyclist"
    obj_type: str
    frames: List[ActorFrame]

    @property
    def t_start(self) -> float:
        return self.frames[0].t if self.frames else 0.0

    @property
    def t_end(self) -> float:
        return self.frames[-1].t if self.frames else 0.0

    def ccli_change_times(self) -> List[float]:
        """Return timestamps where ccli changes (excluding -1 transitions)."""
        changes = []
        prev = None
        for f in self.frames:
            if prev is not None and f.ccli != prev and f.ccli >= 0 and prev >= 0:
                changes.append(f.t)
            prev = f.ccli
        return changes

    def interp(self, t: float) -> ActorFrame:
        """Linear interpolation of position/heading at time t."""
        frames = self.frames
        if not frames:
            return ActorFrame(t, 0, 0, 0, 0, 0, 0, -1, "")
        if t <= frames[0].t:
            return frames[0]
        if t >= frames[-1].t:
            return frames[-1]

        # Binary search
        lo, hi = 0, len(frames) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if frames[mid].t <= t:
                lo = mid
            else:
                hi = mid

        f0, f1 = frames[lo], frames[hi]
        dt = f1.t - f0.t
        if dt < 1e-9:
            return f0
        alpha = (t - f0.t) / dt

        def lerp(a, b):
            return a + alpha * (b - a)

        def lerp_angle(a, b):
            d = (b - a + 180) % 360 - 180
            return a + alpha * d

        return ActorFrame(
            t=t,
            x=lerp(f0.x, f1.x),
            y=lerp(f0.y, f1.y),
            yaw=lerp_angle(f0.yaw, f1.yaw),
            cx=lerp(f0.cx, f1.cx),
            cy=lerp(f0.cy, f1.cy),
            cyaw=lerp_angle(f0.cyaw, f1.cyaw),
            ccli=f0.ccli,
            csource=f0.csource,
        )


@dataclass
class SceneData:
    scenario_name: str
    map_bbox: Dict[str, float]   # min_x, max_x, min_y, max_y
    tracks: List[ActorTrack]
    carla_lines: List[CarlaLine]
    line_by_idx: Dict[int, CarlaLine] = field(default_factory=dict)
    track_by_id: Dict[str, ActorTrack] = field(default_factory=dict)
    # Optional CARLA top-down underlay (only present if pipeline produced it)
    bg_image_b64: Optional[str] = None        # raw base64 JPEG bytes
    bg_image_bounds: Optional[Dict[str, float]] = None  # {min_x,max_x,min_y,max_y} in V2XPNP frame

    def __post_init__(self):
        self.line_by_idx = {cl.idx: cl for cl in self.carla_lines}
        self.track_by_id = {tr.track_id: tr for tr in self.tracks}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _safe_float(v, default: float = 0.0) -> float:
    try:
        r = float(v)
        return r if math.isfinite(r) else default
    except Exception:
        return default


def _safe_int(v, default: int = -1) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _polyline_length(pts: np.ndarray) -> float:
    if pts.shape[0] < 2:
        return 0.0
    diffs = np.diff(pts, axis=0)
    return float(np.sum(np.sqrt((diffs ** 2).sum(axis=1))))


def _parse_frame(raw: dict) -> ActorFrame:
    cx = _safe_float(raw.get("cx"), _safe_float(raw.get("sx"), _safe_float(raw.get("x"), 0.0)))
    cy = _safe_float(raw.get("cy"), _safe_float(raw.get("sy"), _safe_float(raw.get("y"), 0.0)))
    cyaw = _safe_float(raw.get("cyaw"), _safe_float(raw.get("syaw"), _safe_float(raw.get("yaw"), 0.0)))
    return ActorFrame(
        t=_safe_float(raw.get("t"), 0.0),
        x=_safe_float(raw.get("x"), cx),
        y=_safe_float(raw.get("y"), cy),
        yaw=_safe_float(raw.get("yaw"), 0.0),
        cx=cx,
        cy=cy,
        cyaw=cyaw,
        ccli=_safe_int(raw.get("ccli"), -1),
        csource=str(raw.get("csource") or ""),
    )


def _parse_track(raw: dict) -> ActorTrack:
    tid = str(raw.get("id", "unknown"))
    role = str(raw.get("role", "vehicle"))
    obj_type = str(raw.get("obj_type", role))
    frames = [_parse_frame(f) for f in (raw.get("frames") or [])]
    return ActorTrack(track_id=tid, role=role, obj_type=obj_type, frames=frames)


def _parse_carla_line(raw: dict) -> Optional[CarlaLine]:
    pts_raw = raw.get("polyline") or []
    if len(pts_raw) < 2:
        return None
    pts = np.array([[float(p[0]), float(p[1])] for p in pts_raw], dtype=np.float64)
    length = _polyline_length(pts)
    mid_idx = len(pts_raw) // 2
    return CarlaLine(
        idx=_safe_int(raw.get("index"), -1),
        label=str(raw.get("label") or f"c{raw.get('index', '?')}"),
        pts=pts,
        length=length,
        mid_x=_safe_float(raw.get("mid_x"), float(pts[mid_idx, 0])),
        mid_y=_safe_float(raw.get("mid_y"), float(pts[mid_idx, 1])),
        road_id=_safe_int(raw.get("road_id"), -1) if raw.get("road_id") is not None else None,
        lane_id=_safe_int(raw.get("lane_id"), 0) if raw.get("lane_id") is not None else None,
        dir_sign=_safe_int(raw.get("dir_sign"), 1) if raw.get("dir_sign") is not None else None,
    )


def load_from_html(html_path: Path) -> SceneData:
    """Extract embedded dataset JSON from pipeline HTML output."""
    text = html_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(
        r'<script[^>]+id=["\']dataset["\'][^>]*type=["\']application/json["\'][^>]*>(.*?)</script>',
        text, re.DOTALL | re.IGNORECASE,
    )
    if not m:
        # Fallback: try any script tag containing "scenario_name"
        m = re.search(r'<script[^>]*>(.*?"scenario_name".*?)</script>', text, re.DOTALL)
    if not m:
        raise ValueError(f"No embedded dataset found in {html_path}")
    return load_from_dict(json.loads(m.group(1)))


def load_from_json(json_path: Path) -> SceneData:
    return load_from_dict(json.loads(json_path.read_text(encoding="utf-8")))


def load_from_dict(d: dict) -> SceneData:
    scenario_name = str(d.get("scenario_name") or d.get("map_name") or "unknown")

    bbox = d.get("map_bbox") or {}
    map_bbox = {
        "min_x": _safe_float(bbox.get("min_x"), 0.0),
        "max_x": _safe_float(bbox.get("max_x"), 100.0),
        "min_y": _safe_float(bbox.get("min_y"), 0.0),
        "max_y": _safe_float(bbox.get("max_y"), 100.0),
    }

    tracks = [_parse_track(t) for t in (d.get("tracks") or [])]
    tracks = [tr for tr in tracks if tr.frames]  # drop empty

    carla_map_dict = d.get("carla_map") or {}
    carla_lines_raw = carla_map_dict.get("lines") or []
    carla_lines = [cl for cl in (_parse_carla_line(raw) for raw in carla_lines_raw) if cl is not None]

    # CARLA top-down underlay: inline (single-shot) or via image_ref → carla_images[ref]
    bg_b64: Optional[str] = None
    bg_bounds: Optional[Dict[str, float]] = None
    inline_b64 = carla_map_dict.get("image_b64")
    inline_bounds = carla_map_dict.get("image_bounds")
    if isinstance(inline_b64, str) and inline_b64 and isinstance(inline_bounds, dict):
        bg_b64 = inline_b64
        bg_bounds = {k: _safe_float(inline_bounds.get(k), 0.0) for k in ("min_x", "max_x", "min_y", "max_y")}
    else:
        ref = carla_map_dict.get("image_ref")
        global_imgs = d.get("carla_images") or {}
        row = global_imgs.get(ref) if isinstance(ref, str) else None
        if isinstance(row, dict):
            row_b64 = row.get("image_b64")
            row_bounds = row.get("image_bounds") or inline_bounds
            if isinstance(row_b64, str) and row_b64 and isinstance(row_bounds, dict):
                bg_b64 = row_b64
                bg_bounds = {k: _safe_float(row_bounds.get(k), 0.0) for k in ("min_x", "max_x", "min_y", "max_y")}

    return SceneData(
        scenario_name=scenario_name,
        map_bbox=map_bbox,
        tracks=tracks,
        carla_lines=carla_lines,
        bg_image_b64=bg_b64,
        bg_image_bounds=bg_bounds,
    )


def find_nearest_line(
    carla_lines: List[CarlaLine],
    px: float,
    py: float,
    max_dist: float = 30.0,
    heading_deg: Optional[float] = None,
    heading_tolerance_deg: float = 45.0,
) -> Optional[CarlaLine]:
    """Find closest CARLA line to (px, py), optionally filtered by heading."""
    best_line = None
    best_dist = max_dist

    for cl in carla_lines:
        pts = cl.pts
        # Quick mid-point rejection
        if abs(cl.mid_x - px) > max_dist * 2 or abs(cl.mid_y - py) > max_dist * 2:
            continue

        # Closest point on polyline
        d = _min_dist_to_polyline(pts, px, py)
        if d >= best_dist:
            continue

        if heading_deg is not None:
            # Check heading compatibility using the line's end-to-end heading
            dx = float(pts[-1, 0] - pts[0, 0])
            dy = float(pts[-1, 1] - pts[0, 1])
            line_heading = math.degrees(math.atan2(dy, dx))
            diff = abs((heading_deg - line_heading + 180) % 360 - 180)
            if diff > heading_tolerance_deg and diff < (360 - heading_tolerance_deg):
                continue

        best_dist = d
        best_line = cl

    return best_line


def find_rightmost_parallel_line(
    carla_lines: List[CarlaLine],
    current_line: CarlaLine,
    max_lateral: float = 15.0,
    heading_tolerance_deg: float = 25.0,
) -> Optional[CarlaLine]:
    """Find the rightmost line parallel to current_line (outermost lane snap)."""
    # Compute current line heading
    pts = current_line.pts
    dx = float(pts[-1, 0] - pts[0, 0])
    dy = float(pts[-1, 1] - pts[0, 1])
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None
    heading = math.atan2(dy, dx)  # radians
    right_vec = (math.sin(heading), -math.cos(heading))  # perpendicular right

    cx, cy = current_line.mid_x, current_line.mid_y

    best_line = None
    best_right_offset = 0.0  # current line is at 0

    for cl in carla_lines:
        if cl.idx == current_line.idx:
            continue
        # Check heading
        cdx = float(cl.pts[-1, 0] - cl.pts[0, 0])
        cdy = float(cl.pts[-1, 1] - cl.pts[0, 1])
        cl_heading = math.atan2(cdy, cdx)
        hdiff = abs(math.degrees(cl_heading - heading))
        hdiff = min(hdiff, 360 - hdiff)
        if hdiff > heading_tolerance_deg:
            continue

        # Lateral offset (positive = right of current line)
        lx = cl.mid_x - cx
        ly = cl.mid_y - cy
        lateral = lx * right_vec[0] + ly * right_vec[1]

        # Longitudinal must be small relative to lateral
        fwd_vec = (math.cos(heading), math.sin(heading))
        longitudinal = abs(lx * fwd_vec[0] + ly * fwd_vec[1])
        if abs(lateral) > max_lateral or longitudinal > max_lateral * 3:
            continue

        if lateral > best_right_offset:
            best_right_offset = lateral
            best_line = cl

    return best_line


def _min_dist_to_polyline(pts: np.ndarray, px: float, py: float) -> float:
    """Minimum distance from (px,py) to a polyline."""
    min_d2 = float("inf")
    p = np.array([px, py], dtype=np.float64)
    for i in range(len(pts) - 1):
        a = pts[i]
        b = pts[i + 1]
        ab = b - a
        ab2 = float(np.dot(ab, ab))
        if ab2 < 1e-12:
            d2 = float(np.dot(p - a, p - a))
        else:
            t = max(0.0, min(1.0, float(np.dot(p - a, ab)) / ab2))
            proj = a + t * ab
            diff = p - proj
            d2 = float(np.dot(diff, diff))
        if d2 < min_d2:
            min_d2 = d2
    return math.sqrt(max(0.0, min_d2))
