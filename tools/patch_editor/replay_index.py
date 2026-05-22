"""
replay_index.py  —  Discover and index TCP-vid closed-loop planner runs
for the patch editor's replay overlay.

Given a tcp-vid root (the directory produced by running TCP in video mode,
e.g. ``results/results_driving_custom/tcp-videos/tcp-vid/<TAG>/``), this
module:

  1. Walks ``<root>/**/image/<RUNSTAMP>/`` looking for runs that contain
     ``meta_<ego>/`` + ``cine_top_wide_<ego>/`` (the cinematic top-down
     cameras the patch editor will overlay on its map view).
  2. Parses each run's spawn point and pose timeline so the editor can
     match a v2xpnp scenarioset directory to its closed-loop replay
     deterministically (spatial match) without relying on filename
     conventions.
  3. Exposes lazy accessors for per-frame world pose, per-frame cine
     image paths, and per-run scorecard diagnostics.

No knowledge of CARLA or the editor's HTTP layer here — pure data.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# cine_top_wide default extrinsics from tcp_agent.py::_cinematic_sensors().
# Kept here so the editor can compute the BEV footprint independently —
# tcp_agent reads these from env vars at run time but they almost always
# stay at the defaults.
CINE_TOP_WIDE_Z = 120.0     # metres above ego
CINE_TOP_WIDE_FOV_H = 50.0  # horizontal FOV in degrees
CINE_TOP_WIDE_W = 1920
CINE_TOP_WIDE_H = 1080

# CARLA GNSS conversion constants — used only when the agent's saved
# meta JSON lacks the explicit ``world_*`` fields (i.e. runs produced
# before that logging was added). The default CARLA map georeference
# has lat/lon origin at (0, 0); the route planner scales lat→m by
# 111324.6 and lon→m by 111319.5.
GNSS_LAT_SCALE = 111324.60662786
GNSS_LON_SCALE = 111319.490945


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FramePose:
    """One frame of an ego's pose timeline (CARLA world frame)."""
    frame: int            # tick index (matches the meta_*.json basename)
    sim_time: float       # seconds from the run start (frame * dt)
    x: float              # world X (meters)
    y: float              # world Y (meters)
    yaw_deg: float        # world heading, CARLA convention (degrees)
    source: str           # "logged" (world_* in meta) or "approximate" (inverted gps)
    speed: Optional[float] = None
    throttle: Optional[float] = None
    brake: Optional[float] = None
    steer: Optional[float] = None


@dataclass
class EgoReplay:
    """All replay assets for one ego in one closed-loop run."""
    ego_id: int
    cine_top_wide_dir: Optional[Path]   # may be None if camera missing
    cine_front_dir: Optional[Path]
    cine_chase_dir: Optional[Path]
    meta_dir: Path
    frame_numbers: List[int] = field(default_factory=list)   # sorted, unique
    # Lazily-populated pose timeline (frame → FramePose)
    _poses: Optional[Dict[int, FramePose]] = field(default=None, repr=False)
    spawn_xy: Optional[Tuple[float, float]] = None
    spawn_yaw_deg: Optional[float] = None
    alignment_source: str = "unknown"  # "logged" / "approximate" / "spawn_only"

    def frame_count(self) -> int:
        return len(self.frame_numbers)

    def image_path(self, kind: str, frame: int) -> Optional[Path]:
        base = {
            "cine_top_wide": self.cine_top_wide_dir,
            "cine_front":    self.cine_front_dir,
            "cine_chase":    self.cine_chase_dir,
        }.get(kind)
        if base is None:
            return None
        candidate = base / f"{frame:04d}.png"
        return candidate if candidate.exists() else None

    def poses(self, dt: float = 0.05) -> Dict[int, FramePose]:
        if self._poses is None:
            self._poses = _load_pose_timeline(self.meta_dir, self.frame_numbers, dt)
            if self._poses:
                # Update alignment_source from the first valid pose so the
                # manifest reports the actual source (logged vs approximate).
                first = next(iter(self._poses.values()))
                self.alignment_source = first.source
            elif self.spawn_xy is not None:
                self.alignment_source = "spawn_only"
        return self._poses


@dataclass
class ReplayRun:
    """One closed-loop scenario run."""
    run_dir: Path                       # .../image/<RUNSTAMP>/
    parent_dir: Path                    # .../<TAG>_partial_<DATESTAMP>/
    run_tag: str                        # parent_dir.name
    runstamp: str                       # run_dir.name
    egos: List[EgoReplay]
    spawn_xys: List[Optional[Tuple[float, float]]]   # parallel to egos; from point_coordinates.json
    spawn_yaws_deg: List[Optional[float]]            # parallel to egos
    results_json: Optional[Path]        # .../ego_vehicle_<i>/results.json
    point_coords_json: Optional[Path]   # .../log/point_coordinates.json

    def diagnostics(self) -> Dict[int, dict]:
        """Per-ego summary derived from ego_vehicle_<i>/results.json."""
        out: Dict[int, dict] = {}
        for ego in self.egos:
            rj = self.parent_dir / f"ego_vehicle_{ego.ego_id}" / "results.json"
            if not rj.exists():
                out[ego.ego_id] = {}
                continue
            try:
                blob = json.loads(rj.read_text())
            except Exception:
                out[ego.ego_id] = {}
                continue
            rec = (blob.get("_checkpoint", {})
                       .get("records") or [{}])[0]
            hug = rec.get("hugsim", {}) or {}
            pdm = rec.get("pdm", {}) or {}
            inf = rec.get("infractions", {}) or {}
            meta = rec.get("meta", {}) or {}
            scores = rec.get("scores", {}) or {}
            out[ego.ego_id] = {
                "route_completion":     hug.get("route_completion"),
                "driving_score":        scores.get("score_composed"),
                "score_penalty":        scores.get("score_penalty"),
                "score_route":          scores.get("score_route"),
                "pdm_score":            pdm.get("pdm_score"),
                "hug_score":            hug.get("hug_score"),
                "ego_distance_travelled": hug.get("ego_distance_travelled"),
                "expert_route_distance": hug.get("expert_route_distance"),
                "frame_count":          hug.get("frame_count"),
                "route_length":         meta.get("route_length"),
                "spawn_status":         meta.get("ego_spawn_status"),
                "partial_spawn":        meta.get("partial_spawn_accepted"),
                "terminal_status":      meta.get("route_terminal_status"),
                "status":               rec.get("status"),
                "collisions_vehicle":   len(inf.get("collisions_vehicle") or []),
                "collisions_pedestrian": len(inf.get("collisions_pedestrian") or []),
                "collisions_layout":    len(inf.get("collisions_layout") or []),
                "red_light":            len(inf.get("red_light") or []),
                "outside_route_lanes":  len(inf.get("outside_route_lanes") or []),
                "route_dev":            len(inf.get("route_dev") or []),
                "vehicle_blocked":      len(inf.get("vehicle_blocked") or []),
                "route_timeout":        len(inf.get("route_timeout") or []),
            }
        return out


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

_RE_EGO_DIR = re.compile(r"^(?P<kind>cine_top_wide|cine_front|cine_chase|meta)_(?P<ego>\d+)$")


def _list_run_dirs(root: Path) -> List[Path]:
    """Find all ``<root>/.../image/<RUNSTAMP>/`` directories that contain
    cinematic camera output. Returns absolute Paths.
    """
    if not root.is_dir():
        return []
    out: List[Path] = []
    # We expect: <root>/<TAG>/<scenario_partial_dir>/image/<RUNSTAMP>
    # but we tolerate variation (e.g. ``<root>/<scenario_partial_dir>/image/<RUNSTAMP>``).
    for image_dir in root.rglob("image"):
        if not image_dir.is_dir():
            continue
        for run_dir in image_dir.iterdir():
            if not run_dir.is_dir():
                continue
            # Quick filter: must contain at least one cine_top_wide_<n> or meta_<n> subdir
            ok = False
            for child in run_dir.iterdir():
                if child.is_dir() and _RE_EGO_DIR.match(child.name):
                    ok = True
                    break
            if ok:
                out.append(run_dir.resolve())
    return out


def _build_ego_replay(run_dir: Path, ego_id: int) -> Optional[EgoReplay]:
    meta_dir = run_dir / f"meta_{ego_id}"
    top_wide = run_dir / f"cine_top_wide_{ego_id}"
    cine_front = run_dir / f"cine_front_{ego_id}"
    cine_chase = run_dir / f"cine_chase_{ego_id}"
    if not meta_dir.is_dir():
        return None

    frame_numbers: List[int] = []
    seen: set = set()
    for f in meta_dir.glob("*.json"):
        try:
            n = int(f.stem)
        except ValueError:
            continue
        if n not in seen:
            seen.add(n)
            frame_numbers.append(n)
    frame_numbers.sort()
    if not frame_numbers:
        # Fall back to the cine dir if meta is empty
        if top_wide.is_dir():
            for f in top_wide.glob("*.png"):
                try:
                    n = int(f.stem)
                except ValueError:
                    continue
                if n not in seen:
                    seen.add(n)
                    frame_numbers.append(n)
            frame_numbers.sort()

    return EgoReplay(
        ego_id=ego_id,
        cine_top_wide_dir=top_wide if top_wide.is_dir() else None,
        cine_front_dir=cine_front if cine_front.is_dir() else None,
        cine_chase_dir=cine_chase if cine_chase.is_dir() else None,
        meta_dir=meta_dir,
        frame_numbers=frame_numbers,
    )


def _parse_point_coords(pc_json: Path) -> Tuple[
    List[Optional[Tuple[float, float]]], List[Optional[float]]
]:
    """Return parallel lists of (spawn_xy, spawn_yaw_deg) per ego from
    ``log/point_coordinates.json``. Lists are sparse — missing entries
    return as ``None``.
    """
    try:
        blob = json.loads(pc_json.read_text())
    except Exception:
        return [], []
    ego_routes = blob.get("ego_routes") or []
    spawn_xys: List[Optional[Tuple[float, float]]] = []
    spawn_yaws: List[Optional[float]] = []
    for entry in ego_routes:
        if not isinstance(entry, dict):
            spawn_xys.append(None)
            spawn_yaws.append(None)
            continue
        pts = entry.get("points") or []
        first = pts[0] if pts else None
        if not isinstance(first, dict):
            spawn_xys.append(None)
            spawn_yaws.append(None)
            continue
        try:
            spawn_xys.append((float(first["x"]), float(first["y"])))
        except Exception:
            spawn_xys.append(None)
        try:
            spawn_yaws.append(float(first.get("yaw", 0.0)))
        except Exception:
            spawn_yaws.append(None)
    return spawn_xys, spawn_yaws


def _build_run(run_dir: Path) -> Optional[ReplayRun]:
    # Collect ego ids by scanning child dir names
    ego_ids: set = set()
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        m = _RE_EGO_DIR.match(child.name)
        if m:
            try:
                ego_ids.add(int(m.group("ego")))
            except ValueError:
                pass
    if not ego_ids:
        return None
    egos: List[EgoReplay] = []
    for eid in sorted(ego_ids):
        e = _build_ego_replay(run_dir, eid)
        if e is not None and e.frame_count() > 0:
            egos.append(e)
    if not egos:
        return None

    parent_dir = run_dir.parent.parent  # .../<scenario_partial_dir>/

    # For each EgoReplay, derive the authoritative spawn from the
    # first non-empty meta_<ego> frame. ``point_coordinates.json`` lives
    # at the batch-parent level and lists routes in the scenario's
    # original ego order — which does NOT necessarily correspond to
    # the meta_X directory order (the agent applies its own
    # original→active remapping). Using the first-frame pose is the
    # only reliable cross-reference.
    for ego in egos:
        first_pose = _first_valid_meta_pose(ego.meta_dir, ego.frame_numbers)
        if first_pose is not None:
            ego.spawn_xy = (first_pose[0], first_pose[1])
            ego.spawn_yaw_deg = first_pose[2]
            ego.alignment_source = first_pose[3]
    spawn_xys = [ego.spawn_xy for ego in egos]
    spawn_yaws = [ego.spawn_yaw_deg for ego in egos]

    results_json = parent_dir / "results.json"
    pc_json = parent_dir / "log" / "point_coordinates.json"

    return ReplayRun(
        run_dir=run_dir,
        parent_dir=parent_dir,
        run_tag=parent_dir.name,
        runstamp=run_dir.name,
        egos=egos,
        spawn_xys=spawn_xys,
        spawn_yaws_deg=spawn_yaws,
        results_json=results_json if results_json.exists() else None,
        point_coords_json=pc_json if pc_json.exists() else None,
    )


def scan_replay_root(root: Path) -> List[ReplayRun]:
    """Discover every closed-loop run under ``root``."""
    runs: List[ReplayRun] = []
    for run_dir in _list_run_dirs(root):
        run = _build_run(run_dir)
        if run is not None:
            runs.append(run)
    return runs


# ---------------------------------------------------------------------------
# Pose timeline loading
# ---------------------------------------------------------------------------

def _gps_to_world(gps_x: float, gps_y: float) -> Tuple[float, float]:
    """Best-effort inverse of TCP's route-planner gps→local mapping.

    The TCP agent stores ``gps_x = lat * GNSS_LAT_SCALE`` and
    ``gps_y = lon * GNSS_LON_SCALE``. CARLA's default GNSS sensor
    produces ``lat = -world_y / GNSS_LON_SCALE`` and
    ``lon = world_x / GNSS_LAT_SCALE`` for the zero-georeference map.

    Inverting end-to-end: ``world_x ≈ gps_y``, ``world_y ≈ -gps_x``.
    This is approximate; it ignores latitude-dependent conversion
    error and any non-zero map georeference. Use it only when meta
    lacks explicit ``world_*`` fields.
    """
    return gps_y, -gps_x


def _theta_to_yaw_deg(theta_rad: float) -> float:
    """``theta = compass + π/2`` (radians). Empirically, CARLA's
    ``Rotation.yaw`` (degrees) matches ``theta_rad - π`` (in degrees)
    on the example tcp-vid runs we tested. This conversion is only
    used when the meta JSON lacks explicit ``world_yaw_deg`` — the
    logged-pose path is exact.
    """
    return math.degrees(theta_rad - math.pi)


def _first_valid_meta_pose(meta_dir: Path, frame_numbers: Iterable[int]
                           ) -> Optional[Tuple[float, float, float, str]]:
    """Return ``(world_x, world_y, world_yaw_deg, source)`` for the
    earliest meta_*.json that contains a usable pose, or ``None``."""
    for n in frame_numbers:
        meta_path = meta_dir / f"{n:04d}.json"
        if not meta_path.exists():
            continue
        try:
            blob = json.loads(meta_path.read_text())
        except Exception:
            continue
        if not isinstance(blob, dict) or not blob:
            continue
        wx = blob.get("world_x")
        wy = blob.get("world_y")
        wyaw = blob.get("world_yaw_deg")
        if wx is not None and wy is not None and wyaw is not None:
            try:
                return float(wx), float(wy), float(wyaw), "logged"
            except Exception:
                pass
        gx = blob.get("gps_x")
        gy = blob.get("gps_y")
        th = blob.get("theta")
        if gx is None or gy is None or th is None:
            continue
        try:
            fx, fy = _gps_to_world(float(gx), float(gy))
            return fx, fy, _theta_to_yaw_deg(float(th)), "approximate"
        except Exception:
            continue
    return None


def _load_pose_timeline(meta_dir: Path, frame_numbers: Iterable[int], dt: float
                        ) -> Dict[int, FramePose]:
    out: Dict[int, FramePose] = {}
    for n in frame_numbers:
        meta_path = meta_dir / f"{n:04d}.json"
        if not meta_path.exists():
            continue
        try:
            blob = json.loads(meta_path.read_text())
        except Exception:
            continue
        if not isinstance(blob, dict) or not blob:
            continue
        # Preferred: explicit logged world pose
        wx = blob.get("world_x")
        wy = blob.get("world_y")
        wyaw = blob.get("world_yaw_deg")
        if wx is not None and wy is not None and wyaw is not None:
            try:
                fx, fy, fyaw = float(wx), float(wy), float(wyaw)
                source = "logged"
            except Exception:
                fx = fy = fyaw = None  # type: ignore[assignment]
                source = ""
        else:
            fx = fy = fyaw = None  # type: ignore[assignment]
            source = ""
        # Fallback: invert route-planner-local gps + theta
        if fx is None:
            gx = blob.get("gps_x")
            gy = blob.get("gps_y")
            th = blob.get("theta")
            if gx is None or gy is None or th is None:
                continue
            try:
                fx, fy = _gps_to_world(float(gx), float(gy))
                fyaw = _theta_to_yaw_deg(float(th))
                source = "approximate"
            except Exception:
                continue
        out[n] = FramePose(
            frame=n,
            sim_time=float(n) * dt,
            x=float(fx),
            y=float(fy),
            yaw_deg=float(fyaw),
            source=source,
            speed=_maybe_float(blob.get("speed")),
            throttle=_maybe_float(blob.get("throttle")),
            brake=_maybe_float(blob.get("brake")),
            steer=_maybe_float(blob.get("steer")),
        )
    return out


def _maybe_float(v) -> Optional[float]:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Scenario ↔ run matching
# ---------------------------------------------------------------------------

def match_scenario(scenario_egos: List[Tuple[float, float]],
                   runs: List[ReplayRun],
                   tol_m: float = 2.0) -> Optional[ReplayRun]:
    """Find the run whose ego spawn positions best match
    ``scenario_egos`` (a list of (x, y) per ego, CARLA world).

    Matching strategy: a run is a *perfect* match if every scenario
    ego has a corresponding run ego whose spawn (from
    ``point_coordinates.json``) is within ``tol_m`` metres. Returns
    the best match (lowest mean distance), or None.
    """
    if not scenario_egos:
        return None
    best: Optional[Tuple[float, ReplayRun]] = None
    for run in runs:
        if not run.spawn_xys:
            continue
        # Greedy 1-to-1 assignment by nearest-neighbor (small N, ego count <= 10)
        remaining = list(range(len(run.spawn_xys)))
        total_d = 0.0
        matched = 0
        for sx, sy in scenario_egos:
            best_i, best_d = -1, math.inf
            for i in remaining:
                rsxy = run.spawn_xys[i]
                if rsxy is None:
                    continue
                d = math.hypot(sx - rsxy[0], sy - rsxy[1])
                if d < best_d:
                    best_d, best_i = d, i
            if best_i < 0 or best_d > tol_m:
                continue
            total_d += best_d
            matched += 1
            remaining.remove(best_i)
        if matched == len(scenario_egos) and matched > 0:
            mean_d = total_d / matched
            if best is None or mean_d < best[0]:
                best = (mean_d, run)
    return best[1] if best else None


# ---------------------------------------------------------------------------
# Public conveniences
# ---------------------------------------------------------------------------

def cine_top_wide_footprint_m(
    z: float = CINE_TOP_WIDE_Z,
    fov_h_deg: float = CINE_TOP_WIDE_FOV_H,
    width_px: int = CINE_TOP_WIDE_W,
    height_px: int = CINE_TOP_WIDE_H,
) -> Tuple[float, float, float]:
    """Return ``(footprint_w_m, footprint_h_m, m_per_px)`` for a
    pinhole CARLA camera at altitude ``z`` looking straight down with
    horizontal FOV ``fov_h_deg`` on an image of size ``width_px × height_px``.
    """
    half_w = z * math.tan(math.radians(fov_h_deg) / 2.0)
    m_per_px = (2.0 * half_w) / float(width_px)
    return (m_per_px * width_px, m_per_px * height_px, m_per_px)


__all__ = [
    "CINE_TOP_WIDE_Z", "CINE_TOP_WIDE_FOV_H", "CINE_TOP_WIDE_W", "CINE_TOP_WIDE_H",
    "FramePose", "EgoReplay", "ReplayRun",
    "scan_replay_root", "match_scenario", "cine_top_wide_footprint_m",
]
