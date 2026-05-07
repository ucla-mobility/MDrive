#!/usr/bin/env python3
"""
Convert V2XPNP YAML trajectories into map-anchored replay data and interactive HTML.

Compared with yaml_to_carla_log:
- Uses the raw YAML coordinates (no global alignment offset / yaw transform).
- Selects the best source vector map automatically between:
  - v2v_corridors_vector_map.pkl
  - v2x_intersection_vector_map.pkl
- Map-matches trajectories onto the selected map and preserves timing.
- Exports a single interactive HTML replay with a time slider.
"""

from __future__ import annotations

import argparse
import atexit
import base64
import json
import math
import os
import pickle
import re
import signal
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------- CARLA Process Management ---------------------- #

def _trajectory_path_length(traj: Sequence["Waypoint"]) -> float:
    """Compute XY path length for a trajectory."""
    if len(traj) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(traj, traj[1:]):
        total += math.hypot(float(b.x) - float(a.x), float(b.y) - float(a.y))
    return float(total)

CARLA_STARTUP_TIMEOUT = 60.0
CARLA_PORT_TRIES = 8
CARLA_PORT_STEP = 3  # CARLA uses 3 consecutive ports (port, port+1, port+2)


def _is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a TCP port is open (i.e., something is listening)."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _is_carla_port_range_free(host: str, port: int, timeout: float = 0.5) -> bool:
    """
    Check if a CARLA port range is free.
    
    CARLA uses 3 consecutive ports:
      - port: RPC port (main port)
      - port+1: streaming port
      - port+2: secondary streaming port
    
    All 3 must be free for CARLA to start successfully.
    """
    for offset in range(3):
        check_port = port + offset
        if check_port > 65535:
            return False
        if _is_port_open(host, check_port, timeout=timeout):
            return False
    return True


def _wait_for_port(host: str, port: int, timeout: float) -> bool:
    """Wait until port is open or timeout expires. Returns True if port opened."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _is_port_open(host, port, timeout=1.0):
            return True
        time.sleep(0.5)
    return False


def _wait_for_port_close(host: str, port: int, timeout: float) -> bool:
    """Wait until port is closed or timeout expires. Returns True if port closed."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _is_port_open(host, port, timeout=1.0):
            return True
        time.sleep(0.5)
    return False


def _find_available_port(host: str, preferred_port: int, tries: int, step: int) -> int:
    """
    Find an available port range for CARLA starting from preferred_port.
    
    CARLA needs 3 consecutive ports (port, port+1, port+2), so we check all 3.
    We also use a step of at least 3 to avoid port range overlaps.
    """
    attempts = max(1, tries)
    # CARLA uses 3 ports, so step must be at least 3 to avoid overlaps
    effective_step = max(3, step)

    # Phase 1: respect local search around preferred_port
    checked: List[int] = []
    for idx in range(attempts):
        port = preferred_port + idx * effective_step
        if port + 2 > 65535:
            break
        checked.append(port)
        if _is_carla_port_range_free(host, port, timeout=0.2):
            return port

    # Phase 2: broad forward scan (high ports are often free on shared servers)
    # Cap scan to keep startup responsive.
    max_scan_ranges = 10000
    scanned = 0
    forward_start = max(10000, preferred_port + attempts * effective_step)
    forward_end = 65533  # needs port, port+1, port+2
    for port in range(forward_start, forward_end + 1, effective_step):
        checked.append(port)
        scanned += 1
        if _is_carla_port_range_free(host, port, timeout=0.05):
            return port
        if scanned >= max_scan_ranges:
            break

    # Phase 3: wrap-around scan below preferred port
    for port in range(1024, max(1024, preferred_port), effective_step):
        checked.append(port)
        scanned += 1
        if _is_carla_port_range_free(host, port, timeout=0.05):
            return port
        if scanned >= (max_scan_ranges * 2):
            break

    checked_preview = checked[:10]
    raise RuntimeError(
        f"No free CARLA port range found. preferred={preferred_port}, "
        f"local_tries={attempts}, step={effective_step}, scanned={len(checked)}. "
        f"Checked sample={checked_preview}. "
        f"CARLA needs 3 consecutive free ports (port, port+1, port+2)."
    )


def _should_add_offscreen(extra_args: List[str]) -> bool:
    """Determine if -RenderOffScreen should be added to CARLA args."""
    if os.environ.get("DISPLAY"):
        return False
    lowered = {arg.lower() for arg in extra_args}
    if any("renderoffscreen" in arg for arg in lowered):
        return False
    if any("windowed" in arg for arg in lowered):
        return False
    return True


class CarlaProcessManager:
    """Manages CARLA server lifecycle with automatic port selection and clean shutdown."""
    
    def __init__(
        self,
        carla_root: Path,
        host: str,
        port: int,
        extra_args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        port_tries: int = CARLA_PORT_TRIES,
        port_step: int = CARLA_PORT_STEP,
    ) -> None:
        self.carla_root = Path(carla_root)
        self.host = host
        self.port = port
        self.extra_args = list(extra_args or [])
        self.env = dict(env or os.environ.copy())
        self.port_tries = port_tries
        self.port_step = port_step
        self.process: Optional[subprocess.Popen] = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self) -> int:
        """Start CARLA server, auto-selecting port if needed. Returns the actual port."""
        if self.is_running():
            return self.port
        carla_script = self.carla_root / "CarlaUE4.sh"
        if not carla_script.exists():
            raise FileNotFoundError(f"CARLA script not found: {carla_script}")
        
        # Find available port range if preferred is busy (CARLA uses port, port+1, port+2)
        if not _is_carla_port_range_free(self.host, self.port):
            busy_ports = [
                self.port + i for i in range(3) 
                if _is_port_open(self.host, self.port + i, timeout=0.5)
            ]
            print(f"[CARLA] Port range {self.port}-{self.port+2} has busy ports: {busy_ports}")
            new_port = _find_available_port(
                self.host,
                self.port,
                tries=self.port_tries,
                step=self.port_step,
            )
            print(f"[CARLA] Switching to port range {new_port}-{new_port+2}.")
            self.port = new_port
        
        cmd = [str(carla_script), f"--world-port={self.port}"]
        if _should_add_offscreen(self.extra_args):
            cmd.append("-RenderOffScreen")
        cmd.extend(self.extra_args)
        
        print(f"[CARLA] Starting: {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd,
            cwd=str(self.carla_root),
            env=self.env,
        )
        
        if not _wait_for_port(self.host, self.port, CARLA_STARTUP_TIMEOUT):
            print(
                f"[CARLA] WARNING: Port {self.port} did not open within "
                f"{CARLA_STARTUP_TIMEOUT:.0f}s."
            )
        else:
            print(f"[CARLA] Server ready on {self.host}:{self.port}")
        
        return self.port

    def stop(self, timeout: float = 15.0) -> None:
        """Stop CARLA server gracefully, with forced kill as fallback."""
        if self.process is None:
            return
        if self.process.poll() is None:
            print("[CARLA] Stopping server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout)
                print("[CARLA] Server stopped.")
            except subprocess.TimeoutExpired:
                print("[CARLA] Server did not stop gracefully, killing...")
                self.process.kill()
                self.process.wait(timeout=timeout)
        self.process = None

    def restart(self, wait_seconds: float = 2.0) -> int:
        """Restart CARLA server. Returns the new port."""
        self.stop()
        _wait_for_port_close(self.host, self.port, timeout=10.0)
        time.sleep(max(0.0, wait_seconds))
        return self.start()


_active_carla_manager: Optional[CarlaProcessManager] = None
_signal_handlers_installed = False


def _install_carla_signal_handlers() -> None:
    """Install signal handlers for clean CARLA shutdown on SIGINT/SIGTERM."""
    global _signal_handlers_installed
    if _signal_handlers_installed:
        return

    def _handler(signum: int, frame: object) -> None:
        if _active_carla_manager is not None:
            _active_carla_manager.stop()
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        sys.exit(0)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    _signal_handlers_installed = True


def _cleanup_carla_atexit() -> None:
    """Atexit handler to ensure CARLA is stopped on script exit."""
    global _active_carla_manager
    if _active_carla_manager is not None:
        _active_carla_manager.stop()
        _active_carla_manager = None


atexit.register(_cleanup_carla_atexit)


def _sanitize_for_json(obj):
    """Recursively replace non-finite floats (inf, -inf, NaN) with None for JSON compatibility."""
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:  # pragma: no cover
    cKDTree = None  # type: ignore

try:
    from scipy.optimize import linear_sum_assignment as _scipy_lsa  # type: ignore
except Exception:  # pragma: no cover
    _scipy_lsa = None  # type: ignore

import hashlib

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v2xpnp.pipeline.trajectory_ingest import (  # noqa: E402
    Waypoint,
    _apply_early_spawn_time_overrides,
    _apply_late_despawn_time_overrides,
    _maximize_safe_early_spawn_actors,
    _maximize_safe_late_despawn_actors,
    apply_se2,
    invert_se2,
    build_trajectories,
    is_vehicle_type,
    map_obj_type,
    pick_yaml_dirs,
)

try:
    from tqdm import tqdm as _tqdm  # type: ignore
except ImportError:
    _tqdm = None  # type: ignore


def _progress_bar(iterable, total=None, desc=None, disable=False):
    """Wrap iterable in tqdm if available, otherwise return as-is."""
    if _tqdm is not None and not disable:
        return _tqdm(iterable, total=total, desc=desc, ncols=100, leave=True)
    return iterable


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _normalize_yaw_deg(yaw: float) -> float:
    y = float(yaw)
    while y > 180.0:
        y -= 360.0
    while y <= -180.0:
        y += 360.0
    return y


def _lerp_angle_deg(a: float, b: float, alpha: float) -> float:
    aa = _normalize_yaw_deg(a)
    bb = _normalize_yaw_deg(b)
    delta = _normalize_yaw_deg(bb - aa)
    return _normalize_yaw_deg(aa + float(alpha) * delta)


def _is_negative_subdir(path: Path) -> bool:
    try:
        return int(path.name) < 0
    except Exception:
        return False


# =============================================================================
# Sidewalk Compression and Walker Stabilization
# =============================================================================
# This module implements implicit sidewalk compression and walker spawn
# stabilization to improve CARLA replay fidelity for pedestrian trajectories
# from real-world LiDAR data.
#
# Key features:
# - Trajectory-aware walker classification (sidewalk-consistent vs crossing)
# - Nonlinear lateral compression for distant sidewalk pedestrians
# - Spawn separation to prevent walker-walker overlap
# - Auto-calibration from map geometry where possible
# =============================================================================


@dataclass
class WalkerClassification:
    """Classification result for a single walker trajectory."""
    vid: int
    is_sidewalk_consistent: bool
    is_crossing: bool
    is_jaywalking: bool
    is_road_walking: bool
    road_occupancy_ratio: float
    crossing_signature_strength: float
    median_lane_distance: float
    min_lane_distance: float
    max_lane_distance: float
    time_in_road_region: float
    lateral_traversal_distance: float
    classification_reason: str


@dataclass
class WalkerCompressionResult:
    """Result of applying sidewalk compression to a walker trajectory."""
    vid: int
    applied: bool
    original_traj: List[Waypoint]
    compressed_traj: List[Waypoint]
    avg_lateral_offset: float
    max_lateral_offset: float
    frames_modified: int
    compression_reason: str


@dataclass
class WalkerStabilizationResult:
    """Result of spawn stabilization for walkers."""
    adjusted_count: int
    conflict_pairs_resolved: int
    avg_separation_offset: float
    max_separation_offset: float
    details: Dict[int, Dict[str, object]]


class WalkerSidewalkProcessor:
    """
    Processor for sidewalk compression and walker stabilization.
    
    Uses CARLA map polylines (road geometry) to determine distance to drivable lanes.
    
    Implements:
    - Trajectory-aware classification of walkers
    - Lateral compression for sidewalk-consistent walkers
    - Spawn separation to prevent overlap
    """
    
    def __init__(
        self,
        carla_map_lines: List[List[List[float]]],
        lane_spacing_m: Optional[float] = None,
        sidewalk_start_factor: float = 0.5,
        sidewalk_outer_factor: float = 3.0,
        compression_target_band_m: float = 2.5,
        compression_power: float = 1.5,
        min_spawn_separation_m: float = 0.8,
        walker_radius_m: float = 0.35,
        crossing_road_ratio_thresh: float = 0.15,
        crossing_lateral_thresh_m: float = 4.0,
        road_presence_min_frames: int = 5,
        max_lateral_offset_m: float = 3.0,
        dt: float = 0.1,
        freeze_stationary_jitter: bool = False,
    ):
        """
        Initialize the WalkerSidewalkProcessor.
        
        Parameters
        ----------
        carla_map_lines : List[List[List[float]]]
            CARLA map polylines in V2XPNP coordinate space. Each line is a list
            of [x, y] points representing road centerlines.
        lane_spacing_m : float, optional
            Average spacing between parallel lane centerlines. If None,
            auto-calibrated from CARLA map geometry.
        sidewalk_start_factor : float
            Factor of lane_spacing for sidewalk start distance (x = factor * spacing).
        sidewalk_outer_factor : float
            Factor k such that sidewalk outer band y = k * x.
        compression_target_band_m : float
            Target sidewalk width after compression.
        compression_power : float
            Power for nonlinear compression falloff (>1 = stronger compression at distance).
        min_spawn_separation_m : float
            Minimum separation between walker spawn positions.
        walker_radius_m : float
            Approximate walker collision radius for separation calculations.
        crossing_road_ratio_thresh : float
            Ratio of trajectory time in road region above which walker is classified crossing.
        crossing_lateral_thresh_m : float
            Lateral traversal distance indicating crossing behavior.
        road_presence_min_frames : int
            Minimum sustained frames in road region to trigger crossing classification.
        max_lateral_offset_m : float
            Maximum allowed lateral offset from compression.
        dt : float
            Timestep for trajectory.
        freeze_stationary_jitter : bool
            If True, stationary/jitter walkers are kept and frozen at spawn pose
            instead of being removed. Useful for visualization.
        """
        self.dt = float(dt)
        self.sidewalk_start_factor = float(sidewalk_start_factor)
        self.sidewalk_outer_factor = float(sidewalk_outer_factor)
        self.compression_target_band_m = float(compression_target_band_m)
        self.compression_power = float(compression_power)
        self.min_spawn_separation_m = float(min_spawn_separation_m)
        self.walker_radius_m = float(walker_radius_m)
        self.crossing_road_ratio_thresh = float(crossing_road_ratio_thresh)
        self.crossing_lateral_thresh_m = float(crossing_lateral_thresh_m)
        self.road_presence_min_frames = int(road_presence_min_frames)
        self.max_lateral_offset_m = float(max_lateral_offset_m)
        self.freeze_stationary_jitter = bool(freeze_stationary_jitter)
        
        # Build spatial index from CARLA map polylines
        self._carla_lines = carla_map_lines
        self._build_carla_road_index()
        
        # Auto-calibrate lane spacing if not provided
        if lane_spacing_m is not None:
            self.lane_spacing_m = float(lane_spacing_m)
        else:
            self.lane_spacing_m = self._estimate_lane_spacing_from_carla()
        
        # Derived geometry parameters
        self.sidewalk_start_distance = self.sidewalk_start_factor * self.lane_spacing_m
        self.sidewalk_outer_distance = self.sidewalk_outer_factor * self.sidewalk_start_distance
        
        # Cache for lane distance computations
        self._lane_dist_cache: Dict[Tuple[float, float], Tuple[float, float, float]] = {}
    
    def _build_carla_road_index(self) -> None:
        """
        Build spatial index from CARLA map polylines for fast distance queries.
        
        Samples points along polylines and builds a KD-tree for nearest neighbor queries.
        Also stores segment information for computing lateral direction.
        """
        # Sample points from all CARLA road polylines
        sample_points: List[Tuple[float, float]] = []
        # Store segment info: for each sampled point, store (line_idx, seg_idx, t)
        self._point_segment_info: List[Tuple[int, int, float]] = []
        
        SAMPLE_SPACING = 1.0  # Sample every ~1m along polylines
        
        for line_idx, line in enumerate(self._carla_lines):
            if len(line) < 2:
                continue
            
            for seg_idx in range(len(line) - 1):
                p0 = line[seg_idx]
                p1 = line[seg_idx + 1]
                if len(p0) < 2 or len(p1) < 2:
                    continue
                
                x0, y0 = float(p0[0]), float(p0[1])
                x1, y1 = float(p1[0]), float(p1[1])
                seg_len = math.hypot(x1 - x0, y1 - y0)
                
                if seg_len < 1e-6:
                    continue
                
                # Sample along segment
                n_samples = max(1, int(seg_len / SAMPLE_SPACING))
                for i in range(n_samples + 1):
                    t = i / max(1, n_samples)
                    px = x0 + t * (x1 - x0)
                    py = y0 + t * (y1 - y0)
                    sample_points.append((px, py))
                    self._point_segment_info.append((line_idx, seg_idx, t))
        
        if sample_points:
            self._road_points = np.asarray(sample_points, dtype=np.float64)
            if cKDTree is not None:
                self._road_tree = cKDTree(self._road_points)
            else:
                self._road_tree = None
        else:
            self._road_points = np.zeros((0, 2), dtype=np.float64)
            self._road_tree = None
        
        print(f"[WALKER] Built CARLA road index: {len(self._carla_lines)} lines, {len(sample_points)} sample points")
    
    def _estimate_lane_spacing_from_carla(self) -> float:
        """
        Estimate average lane spacing from CARLA road polylines.
        
        Analyzes spacing between parallel road lines to determine typical lane width.
        Falls back to standard lane width if estimation fails.
        """
        DEFAULT_LANE_SPACING = 3.5  # Standard CARLA lane width
        
        if not self._carla_lines or len(self._carla_lines) < 2:
            return DEFAULT_LANE_SPACING
        
        # Collect midpoints of each polyline
        line_mids: List[Tuple[float, float]] = []
        for line in self._carla_lines:
            if len(line) < 2:
                continue
            mid_idx = len(line) // 2
            if len(line[mid_idx]) >= 2:
                line_mids.append((float(line[mid_idx][0]), float(line[mid_idx][1])))
        
        if len(line_mids) < 2:
            return DEFAULT_LANE_SPACING
        
        # Find pairwise distances in plausible range
        spacing_samples: List[float] = []
        for i, (x1, y1) in enumerate(line_mids):
            for j, (x2, y2) in enumerate(line_mids):
                if i >= j:
                    continue
                dist = math.hypot(x2 - x1, y2 - y1)
                # Plausible lane spacing range: 2.5m to 5m
                if 2.5 < dist < 5.0:
                    spacing_samples.append(dist)
        
        if spacing_samples:
            spacing_samples.sort()
            return float(spacing_samples[len(spacing_samples) // 2])
        
        return DEFAULT_LANE_SPACING
    
    def _compute_carla_road_distance_and_direction(
        self,
        x: float,
        y: float,
    ) -> Tuple[float, float, float]:
        """
        Compute distance from point to nearest CARLA road polyline.
        
        Returns (distance, dir_x, dir_y) where (dir_x, dir_y) is unit vector
        pointing from nearest road point toward the query point.
        """
        # Check cache
        key = (round(x, 2), round(y, 2))
        if key in self._lane_dist_cache:
            return self._lane_dist_cache[key]
        
        if self._road_points.shape[0] == 0:
            return (float("inf"), 0.0, 0.0)
        
        query_pt = np.asarray([x, y], dtype=np.float64)
        
        if self._road_tree is not None:
            dist, idx = self._road_tree.query(query_pt, k=1)
            nearest_pt = self._road_points[idx]
        else:
            # Fallback: brute force
            diff = self._road_points - query_pt[None, :]
            dists_sq = np.sum(diff * diff, axis=1)
            idx = int(np.argmin(dists_sq))
            dist = float(np.sqrt(dists_sq[idx]))
            nearest_pt = self._road_points[idx]
        
        # Compute direction from road toward point
        dx = x - float(nearest_pt[0])
        dy = y - float(nearest_pt[1])
        mag = math.hypot(dx, dy)
        
        if mag < 1e-6:
            dir_x, dir_y = 0.0, 0.0
        else:
            dir_x, dir_y = dx / mag, dy / mag
        
        result = (float(dist), dir_x, dir_y)
        self._lane_dist_cache[key] = result
        return result
    
    def _compute_trajectory_road_distances(
        self,
        traj: Sequence[Waypoint],
    ) -> np.ndarray:
        """Compute CARLA road distances for all trajectory points."""
        if not traj:
            return np.array([], dtype=np.float64)
        
        if self._road_points.shape[0] == 0:
            return np.full(len(traj), float("inf"), dtype=np.float64)
        
        pts = np.asarray([[float(wp.x), float(wp.y)] for wp in traj], dtype=np.float64)
        
        if self._road_tree is not None:
            dists, _ = self._road_tree.query(pts, k=1, workers=-1)
            return np.asarray(dists, dtype=np.float64)
        else:
            # Brute force fallback
            out = np.empty(pts.shape[0], dtype=np.float64)
            for i, pt in enumerate(pts):
                diff = self._road_points - pt[None, :]
                out[i] = float(np.min(np.sqrt(np.sum(diff * diff, axis=1))))
            return out
    
    def is_walker_stationary_jitter(
        self,
        vid: int,
        traj: Sequence[Waypoint],
        times: Optional[Sequence[float]] = None,
    ) -> Tuple[bool, str, Dict[str, float]]:
        """
        Detect stationary or jittery walkers that should be removed.
        
        Returns (is_stationary, reason, stats).
        
        A walker is considered stationary/jitter if:
        - Very low path length (essentially not moving)
        - High path length but extremely low net displacement (jitter in place)
        - Very low displacement per frame over many frames
        """
        stats: Dict[str, float] = {}
        n = len(traj)
        stats["num_frames"] = float(n)
        
        if n < 2:
            stats["stationary_jitter"] = 1.0
            return True, "single_frame", stats
        
        xs = np.array([float(wp.x) for wp in traj], dtype=np.float64)
        ys = np.array([float(wp.y) for wp in traj], dtype=np.float64)
        
        # Compute motion metrics
        diffs = np.sqrt((xs[1:] - xs[:-1]) ** 2 + (ys[1:] - ys[:-1]) ** 2)
        path_len = float(np.sum(diffs))
        net_disp = float(math.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
        
        stats["path_len_m"] = path_len
        stats["net_disp_m"] = net_disp
        
        # Variance
        var_x = float(np.var(xs))
        var_y = float(np.var(ys))
        total_var = var_x + var_y
        stats["pos_variance_m2"] = total_var
        
        # Path efficiency (1 = straight line, 0 = going nowhere)
        path_efficiency = net_disp / max(0.01, path_len)
        stats["path_efficiency"] = path_efficiency
        
        # Per-frame metrics
        path_per_frame = path_len / max(1, n - 1)
        stats["path_per_frame_m"] = path_per_frame
        
        # ---- Stationary checks ----
        
        # 1. Essentially zero movement
        if path_len < 0.5 and net_disp < 0.3:
            stats["stationary_jitter"] = 1.0
            return True, "no_movement", stats
        
        # 2. Very low variance (clustered around one point)
        if total_var < 0.2:
            stats["stationary_jitter"] = 1.0
            return True, "low_variance", stats
        
        # 3. Jitter: high path but very low net displacement and variance
        # This catches walkers vibrating in place
        if net_disp < 2.0 and path_efficiency < 0.15 and total_var < 1.5:
            stats["stationary_jitter"] = 1.0
            return True, "jitter_detected", stats
        
        # 4. Many frames with tiny per-frame movement
        if n >= 30 and path_per_frame < 0.03 and net_disp < 2.0:
            stats["stationary_jitter"] = 1.0
            return True, "low_path_per_frame", stats
        
        # 5. Low net displacement for long trajectories
        if n >= 50 and net_disp < 3.0 and path_efficiency < 0.25:
            stats["stationary_jitter"] = 1.0
            return True, "low_net_disp_long_traj", stats
        
        stats["stationary_jitter"] = 0.0
        return False, "moving", stats

    def _should_preserve_stationary_path_with_smoothing(
        self,
        reason: str,
        stats: Dict[str, float],
    ) -> bool:
        """
        Keep some stationary/jitter-tagged walkers on their original path (with
        very light smoothing) when motion is still path-like.
        """
        reason_s = str(reason or "")
        if reason_s in {"single_frame", "no_movement"}:
            return False

        path_len = float(stats.get("path_len_m", 0.0))
        net_disp = float(stats.get("net_disp_m", 0.0))
        path_eff = float(stats.get("path_efficiency", 0.0))
        num_frames = int(stats.get("num_frames", 0.0))

        if path_len < 1.0:
            return False
        if net_disp < 0.35:
            return False
        if path_eff < 0.30:
            return False
        if num_frames < 3:
            return False
        return True

    @staticmethod
    def _light_smooth_walker_trajectory(
        traj: Sequence[Waypoint],
        passes: int = 2,
        max_shift_m: float = 0.25,
    ) -> List[Waypoint]:
        """
        Tiny low-pass smoothing for near-static paths; preserves timing and intent.
        """
        n = len(traj)
        if n < 3:
            return [
                Waypoint(
                    x=float(wp.x),
                    y=float(wp.y),
                    z=float(wp.z),
                    yaw=float(wp.yaw),
                    pitch=float(getattr(wp, "pitch", 0.0)),
                    roll=float(getattr(wp, "roll", 0.0)),
                )
                for wp in traj
            ]

        xs = np.asarray([float(wp.x) for wp in traj], dtype=np.float64)
        ys = np.asarray([float(wp.y) for wp in traj], dtype=np.float64)
        sx = xs.copy()
        sy = ys.copy()

        for _ in range(max(1, int(passes))):
            sx_new = sx.copy()
            sy_new = sy.copy()
            sx_new[1:-1] = 0.25 * sx[:-2] + 0.50 * sx[1:-1] + 0.25 * sx[2:]
            sy_new[1:-1] = 0.25 * sy[:-2] + 0.50 * sy[1:-1] + 0.25 * sy[2:]
            dx = sx_new - xs
            dy = sy_new - ys
            mag = np.hypot(dx, dy)
            scale = np.ones_like(mag)
            mask = mag > float(max_shift_m)
            if np.any(mask):
                scale[mask] = float(max_shift_m) / np.maximum(mag[mask], 1e-9)
            sx = xs + dx * scale
            sy = ys + dy * scale

        out: List[Waypoint] = []
        for i, wp in enumerate(traj):
            out.append(
                Waypoint(
                    x=float(sx[i]),
                    y=float(sy[i]),
                    z=float(wp.z),
                    yaw=float(wp.yaw),
                    pitch=float(getattr(wp, "pitch", 0.0)),
                    roll=float(getattr(wp, "roll", 0.0)),
                )
            )
        return out
    
    def classify_walker(
        self,
        vid: int,
        traj: Sequence[Waypoint],
        times: Optional[Sequence[float]] = None,
    ) -> WalkerClassification:
        """
        Classify walker trajectory as sidewalk-consistent or crossing/jaywalking.
        
        Classification is trajectory-aware (not frame-based).
        Uses CARLA road polylines to determine distance to drivable area.
        """
        if not traj:
            return WalkerClassification(
                vid=vid,
                is_sidewalk_consistent=False,
                is_crossing=False,
                is_jaywalking=False,
                is_road_walking=False,
                road_occupancy_ratio=0.0,
                crossing_signature_strength=0.0,
                median_lane_distance=float("inf"),
                min_lane_distance=float("inf"),
                max_lane_distance=0.0,
                time_in_road_region=0.0,
                lateral_traversal_distance=0.0,
                classification_reason="empty_trajectory",
            )
        
        # Compute CARLA road distances for entire trajectory
        road_dists = self._compute_trajectory_road_distances(traj)
        
        if road_dists.size == 0:
            return WalkerClassification(
                vid=vid,
                is_sidewalk_consistent=False,
                is_crossing=False,
                is_jaywalking=False,
                is_road_walking=False,
                road_occupancy_ratio=0.0,
                crossing_signature_strength=0.0,
                median_lane_distance=float("inf"),
                min_lane_distance=float("inf"),
                max_lane_distance=0.0,
                time_in_road_region=0.0,
                lateral_traversal_distance=0.0,
                classification_reason="no_road_distances",
            )
        
        # Statistics
        median_dist = float(np.median(road_dists))
        min_dist = float(np.min(road_dists))
        max_dist = float(np.max(road_dists))
        
        x = self.sidewalk_start_distance
        
        # Road occupancy: fraction of trajectory with d <= x
        in_road = road_dists <= x
        road_occupancy_ratio = float(np.mean(in_road.astype(np.float64)))
        
        # Compute sustained presence in road region
        max_road_run = 0
        current_run = 0
        for in_r in in_road:
            if in_r:
                current_run += 1
                max_road_run = max(max_road_run, current_run)
            else:
                current_run = 0
        
        time_in_road_s = float(max_road_run) * self.dt
        
        # Lateral traversal: total lateral movement relative to lane direction
        lateral_traversal = 0.0
        for i in range(1, len(traj)):
            dx = float(traj[i].x) - float(traj[i - 1].x)
            dy = float(traj[i].y) - float(traj[i - 1].y)
            # Approximate lateral as perpendicular to heading
            yaw_rad = math.radians(float(traj[i - 1].yaw))
            # Lateral = component perpendicular to heading
            lat_component = abs(-dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad))
            lateral_traversal += lat_component
        
        # Crossing signature: combination of road presence and lateral movement
        crossing_signature = 0.0
        if road_occupancy_ratio > 0.05:
            # Significant road presence indicates potential crossing
            crossing_signature += road_occupancy_ratio * 0.5
        if lateral_traversal > self.crossing_lateral_thresh_m:
            crossing_signature += min(1.0, lateral_traversal / (2 * self.crossing_lateral_thresh_m)) * 0.5
        
        # Classification logic
        is_crossing = False
        is_jaywalking = False
        is_road_walking = False
        is_sidewalk_consistent = False
        reason = ""
        
        # Clear crossing: sustained road presence + lateral traversal
        if (max_road_run >= self.road_presence_min_frames and
            lateral_traversal > self.crossing_lateral_thresh_m):
            is_crossing = True
            reason = f"crossing_detected: road_frames={max_road_run}, lateral={lateral_traversal:.1f}m"
        
        # Road walking: high road occupancy without clear crossing pattern
        elif road_occupancy_ratio > self.crossing_road_ratio_thresh * 2:
            is_road_walking = True
            reason = f"road_walking: occupancy_ratio={road_occupancy_ratio:.2f}"
        
        # Jaywalking: brief road intrusion
        elif (road_occupancy_ratio > self.crossing_road_ratio_thresh and
              max_road_run >= 3):
            is_jaywalking = True
            reason = f"jaywalking: road_ratio={road_occupancy_ratio:.2f}, max_run={max_road_run}"
        
        # Sidewalk-consistent: majority of trajectory is far from road
        elif road_occupancy_ratio <= self.crossing_road_ratio_thresh:
            # Check majority condition
            sidewalk_ratio = float(np.mean((road_dists > x).astype(np.float64)))
            if sidewalk_ratio >= 0.7:
                is_sidewalk_consistent = True
                reason = f"sidewalk_consistent: sidewalk_ratio={sidewalk_ratio:.2f}, median_dist={median_dist:.1f}m"
            else:
                # Borderline case
                if max_road_run <= 2 and min_dist > x * 0.7:
                    is_sidewalk_consistent = True
                    reason = f"sidewalk_borderline: short_road_intrusion, min_dist={min_dist:.1f}m"
                else:
                    reason = f"unclassified: sidewalk_ratio={sidewalk_ratio:.2f}, min_dist={min_dist:.1f}m"
        else:
            reason = f"unclassified_default: road_ratio={road_occupancy_ratio:.2f}"
        
        return WalkerClassification(
            vid=vid,
            is_sidewalk_consistent=is_sidewalk_consistent,
            is_crossing=is_crossing,
            is_jaywalking=is_jaywalking,
            is_road_walking=is_road_walking,
            road_occupancy_ratio=road_occupancy_ratio,
            crossing_signature_strength=crossing_signature,
            median_lane_distance=median_dist,
            min_lane_distance=min_dist,
            max_lane_distance=max_dist,
            time_in_road_region=time_in_road_s,
            lateral_traversal_distance=lateral_traversal,
            classification_reason=reason,
        )
    
    def _compute_compression_offset(
        self,
        lane_distance: float,
    ) -> float:
        """
        Compute lateral offset for sidewalk compression.
        
        Uses nonlinear falloff: walkers further from road receive stronger compression.
        
        Geometry:
        - x = sidewalk_start_distance: no compression
        - y = sidewalk_outer_distance: maximum compression
        - Region [x, y] compressed into target band
        """
        x = self.sidewalk_start_distance
        y = self.sidewalk_outer_distance
        
        if lane_distance <= x:
            # Inside road/curb region: push OUTWARD onto the sidewalk.
            # Target is sidewalk_start_distance + a small margin (0.4 m)
            # so the walker is clearly on the sidewalk, not on the curb.
            margin = 0.4
            target = x + margin
            # Negative offset = move away from road in the application code
            return -(target - lane_distance)
        
        if lane_distance >= y:
            # Beyond outer sidewalk: strong compression toward x + target_band
            target_max = x + self.compression_target_band_m
            offset = lane_distance - target_max
            return min(offset, self.max_lateral_offset_m)
        
        # Within sidewalk band [x, y]: nonlinear compression
        # Normalize position in band: 0 at x, 1 at y
        band_width = y - x
        normalized_pos = (lane_distance - x) / band_width
        
        # Nonlinear falloff: stronger compression at outer edge
        compressed_pos = math.pow(normalized_pos, 1.0 / self.compression_power)
        
        # Map to target band
        target_dist = x + compressed_pos * self.compression_target_band_m
        
        # Compute offset (positive = move toward lane)
        offset = lane_distance - target_dist
        
        return max(0.0, min(offset, self.max_lateral_offset_m))
    
    def compress_walker_trajectory(
        self,
        vid: int,
        traj: Sequence[Waypoint],
        classification: WalkerClassification,
    ) -> WalkerCompressionResult:
        """
        Apply sidewalk compression to a walker trajectory.
        
        Only applies to sidewalk-consistent walkers.
        Uses CARLA road polylines for distance computation.
        """
        if not traj:
            return WalkerCompressionResult(
                vid=vid,
                applied=False,
                original_traj=list(traj),
                compressed_traj=list(traj),
                avg_lateral_offset=0.0,
                max_lateral_offset=0.0,
                frames_modified=0,
                compression_reason="empty_trajectory",
            )
        
        # Process sidewalk-consistent walkers AND non-crossing road-walking
        # walkers.  Road-walking walkers that aren't crossing the road are
        # just walking slightly on the road edge and should be nudged onto
        # the sidewalk.
        eligible = (
            classification.is_sidewalk_consistent
            or (classification.is_road_walking and not classification.is_crossing)
        )
        if not eligible:
            return WalkerCompressionResult(
                vid=vid,
                applied=False,
                original_traj=list(traj),
                compressed_traj=list(traj),
                avg_lateral_offset=0.0,
                max_lateral_offset=0.0,
                frames_modified=0,
                compression_reason=f"not_eligible: {classification.classification_reason}",
            )
        
        # Check if compression is needed (any points beyond target band OR
        # any points inside the road region that need pushing outward)
        road_dists = self._compute_trajectory_road_distances(traj)
        x = self.sidewalk_start_distance
        target_outer = x + self.compression_target_band_m
        
        needs_compression = np.any(road_dists > target_outer) or np.any(road_dists < x)
        if not needs_compression:
            return WalkerCompressionResult(
                vid=vid,
                applied=False,
                original_traj=list(traj),
                compressed_traj=list(traj),
                avg_lateral_offset=0.0,
                max_lateral_offset=0.0,
                frames_modified=0,
                compression_reason="already_within_target_band",
            )
        
        # Apply compression
        compressed: List[Waypoint] = []
        offsets: List[float] = []
        frames_modified = 0
        
        for i, wp in enumerate(traj):
            d = float(road_dists[i])
            offset = self._compute_compression_offset(d)
            
            if abs(offset) > 1e-4:
                # Compute lateral direction (toward road) using CARLA road geometry
                # dir_x, dir_y points FROM road TOWARD the walker
                _, dir_x, dir_y = self._compute_carla_road_distance_and_direction(float(wp.x), float(wp.y))
                
                if abs(dir_x) + abs(dir_y) > 1e-6:
                    # Positive offset: move toward road (subtract dir)
                    # Negative offset: move away from road onto sidewalk (subtract negative = add dir)
                    new_x = float(wp.x) - offset * dir_x
                    new_y = float(wp.y) - offset * dir_y
                    compressed.append(Waypoint(x=new_x, y=new_y, z=float(wp.z), yaw=float(wp.yaw)))
                    offsets.append(abs(offset))
                    frames_modified += 1
                else:
                    compressed.append(wp)
                    offsets.append(0.0)
            else:
                compressed.append(wp)
                offsets.append(0.0)
        
        avg_offset = float(np.mean(offsets)) if offsets else 0.0
        max_offset = float(np.max(offsets)) if offsets else 0.0
        
        return WalkerCompressionResult(
            vid=vid,
            applied=True,
            original_traj=list(traj),
            compressed_traj=compressed,
            avg_lateral_offset=avg_offset,
            max_lateral_offset=max_offset,
            frames_modified=frames_modified,
            compression_reason=f"compressed: {frames_modified}/{len(traj)} frames, avg={avg_offset:.2f}m, max={max_offset:.2f}m",
        )
    
    def stabilize_walker_spawns(
        self,
        walker_trajectories: Dict[int, List[Waypoint]],
        walker_times: Dict[int, List[float]],
    ) -> Tuple[Dict[int, List[Waypoint]], WalkerStabilizationResult]:
        """
        Stabilize walker spawn positions to prevent overlap.
        
        Ensures minimum separation between walkers spawning at similar times.
        """
        if not walker_trajectories:
            return walker_trajectories, WalkerStabilizationResult(
                adjusted_count=0,
                conflict_pairs_resolved=0,
                avg_separation_offset=0.0,
                max_separation_offset=0.0,
                details={},
            )
        
        # Collect spawn positions and times
        spawn_info: List[Tuple[int, float, float, float, float]] = []  # (vid, t, x, y, z)
        for vid, traj in walker_trajectories.items():
            if not traj:
                continue
            times = walker_times.get(vid, [])
            t0 = float(times[0]) if times else 0.0
            wp0 = traj[0]
            spawn_info.append((vid, t0, float(wp0.x), float(wp0.y), float(wp0.z)))
        
        if len(spawn_info) < 2:
            return walker_trajectories, WalkerStabilizationResult(
                adjusted_count=0,
                conflict_pairs_resolved=0,
                avg_separation_offset=0.0,
                max_separation_offset=0.0,
                details={},
            )
        
        # Sort by spawn time
        spawn_info.sort(key=lambda s: s[1])
        
        # Find conflicts: walkers spawning too close together
        min_sep = self.min_spawn_separation_m + 2 * self.walker_radius_m
        time_window = 2.0  # Consider walkers within 2s as potentially conflicting
        
        # Track adjustments
        adjustments: Dict[int, Tuple[float, float]] = {}  # vid -> (dx, dy)
        conflict_pairs = 0
        
        for i, (vid_i, t_i, x_i, y_i, z_i) in enumerate(spawn_info):
            for j in range(i + 1, len(spawn_info)):
                vid_j, t_j, x_j, y_j, z_j = spawn_info[j]
                
                # Check time proximity
                if t_j - t_i > time_window:
                    break
                
                # Check spatial proximity
                dist = math.hypot(x_j - x_i, y_j - y_i)
                if dist >= min_sep:
                    continue
                
                conflict_pairs += 1
                
                # Compute separation direction
                if dist < 1e-4:
                    # Overlapping: use random-ish direction
                    sep_dx, sep_dy = 1.0, 0.0
                else:
                    sep_dx = (x_j - x_i) / dist
                    sep_dy = (y_j - y_i) / dist
                
                # Required offset to achieve separation
                needed_offset = (min_sep - dist) / 2.0 + 0.1
                needed_offset = min(needed_offset, self.max_lateral_offset_m / 2)
                
                # Apply symmetric offset (push both apart)
                adj_i = adjustments.get(vid_i, (0.0, 0.0))
                adj_j = adjustments.get(vid_j, (0.0, 0.0))
                
                adjustments[vid_i] = (
                    adj_i[0] - sep_dx * needed_offset,
                    adj_i[1] - sep_dy * needed_offset,
                )
                adjustments[vid_j] = (
                    adj_j[0] + sep_dx * needed_offset,
                    adj_j[1] + sep_dy * needed_offset,
                )
        
        if not adjustments:
            return walker_trajectories, WalkerStabilizationResult(
                adjusted_count=0,
                conflict_pairs_resolved=conflict_pairs,
                avg_separation_offset=0.0,
                max_separation_offset=0.0,
                details={},
            )
        
        # Apply adjustments to trajectories
        adjusted_trajectories = dict(walker_trajectories)
        details: Dict[int, Dict[str, object]] = {}
        offset_magnitudes: List[float] = []
        
        for vid, (dx, dy) in adjustments.items():
            if vid not in adjusted_trajectories:
                continue
            
            offset_mag = math.hypot(dx, dy)
            if offset_mag < 1e-4:
                continue
            
            # Cap the offset
            if offset_mag > self.max_lateral_offset_m:
                scale = self.max_lateral_offset_m / offset_mag
                dx *= scale
                dy *= scale
                offset_mag = self.max_lateral_offset_m
            
            offset_magnitudes.append(offset_mag)
            
            # Apply constant offset to entire trajectory (preserves motion)
            original_traj = adjusted_trajectories[vid]
            new_traj = [
                Waypoint(
                    x=float(wp.x) + dx,
                    y=float(wp.y) + dy,
                    z=float(wp.z),
                    yaw=float(wp.yaw),
                )
                for wp in original_traj
            ]
            adjusted_trajectories[vid] = new_traj
            
            details[vid] = {
                "offset_x": float(dx),
                "offset_y": float(dy),
                "offset_magnitude": float(offset_mag),
            }
        
        avg_offset = float(np.mean(offset_magnitudes)) if offset_magnitudes else 0.0
        max_offset = float(np.max(offset_magnitudes)) if offset_magnitudes else 0.0
        
        return adjusted_trajectories, WalkerStabilizationResult(
            adjusted_count=len(offset_magnitudes),
            conflict_pairs_resolved=conflict_pairs,
            avg_separation_offset=avg_offset,
            max_separation_offset=max_offset,
            details=details,
        )
    
    def process_walkers(
        self,
        vehicles: Dict[int, List[Waypoint]],
        vehicle_times: Dict[int, List[float]],
        obj_info: Dict[int, Dict[str, object]],
    ) -> Tuple[Dict[int, List[Waypoint]], Dict[str, object]]:
        """
        Main entry point: process all walker trajectories.
        
        Returns updated vehicles dict and detailed report.
        """
        # Identify walkers
        walker_vids: List[int] = []
        for vid, traj in vehicles.items():
            if not traj:
                continue
            meta = obj_info.get(vid, {})
            obj_type = str(meta.get("obj_type") or "")
            if _is_pedestrian_type(obj_type):
                walker_vids.append(vid)
        
        if not walker_vids:
            return vehicles, {
                "enabled": True,
                "walker_count": 0,
                "classifications": {},
                "compressions": {},
                "stabilization": {},
                "stationary_removed": {},
                "summary": "no_walkers_found",
            }
        
        # ---- Pass 0: Remove stationary/jittery walkers ----
        # These are walkers with essentially no meaningful movement
        stationary_jitter_removed: Dict[int, Dict[str, object]] = {}
        stationary_jitter_frozen: Dict[int, Dict[str, object]] = {}
        stationary_jitter_preserved: Dict[int, Dict[str, object]] = {}
        non_stationary_vids: List[int] = []
        
        for vid in walker_vids:
            traj = vehicles[vid]
            times = vehicle_times.get(vid, [])
            is_stat, reason, stats = self.is_walker_stationary_jitter(vid, traj, times)
            
            if is_stat:
                row = {
                    "reason": reason,
                    "stats": stats,
                }
                if self.freeze_stationary_jitter:
                    if self._should_preserve_stationary_path_with_smoothing(reason, stats):
                        stationary_jitter_preserved[vid] = row
                        print(f"[WALKER] PRESERVED walker_{vid}: stationary/jitter ({reason}) -> light_smooth, "
                              f"path={stats.get('path_len_m', 0):.1f}m, net_disp={stats.get('net_disp_m', 0):.1f}m, "
                              f"frames={int(stats.get('num_frames', 0))}")
                    else:
                        stationary_jitter_frozen[vid] = row
                        print(f"[WALKER] FROZEN walker_{vid}: stationary/jitter ({reason}), "
                              f"path={stats.get('path_len_m', 0):.1f}m, net_disp={stats.get('net_disp_m', 0):.1f}m, "
                              f"frames={int(stats.get('num_frames', 0))}")
                else:
                    stationary_jitter_removed[vid] = row
                    print(f"[WALKER] REMOVED walker_{vid}: stationary/jitter ({reason}), "
                          f"path={stats.get('path_len_m', 0):.1f}m, net_disp={stats.get('net_disp_m', 0):.1f}m, "
                          f"frames={int(stats.get('num_frames', 0))}")
            else:
                non_stationary_vids.append(vid)
        
        # Update vehicle dict based on stationary policy.
        updated_vehicles = dict(vehicles)
        if self.freeze_stationary_jitter:
            for vid in stationary_jitter_preserved:
                traj = updated_vehicles.get(vid, [])
                if not traj:
                    continue
                updated_vehicles[vid] = self._light_smooth_walker_trajectory(traj)
            for vid in stationary_jitter_frozen:
                traj = updated_vehicles.get(vid, [])
                if not traj:
                    continue
                anchor = traj[0]
                updated_vehicles[vid] = [
                    Waypoint(
                        x=float(anchor.x),
                        y=float(anchor.y),
                        z=float(anchor.z),
                        yaw=float(anchor.yaw),
                        pitch=float(getattr(anchor, "pitch", 0.0)),
                        roll=float(getattr(anchor, "roll", 0.0)),
                    )
                    for _ in traj
                ]
        else:
            for vid in stationary_jitter_removed:
                if vid in updated_vehicles:
                    del updated_vehicles[vid]
                if vid in vehicle_times:
                    del vehicle_times[vid]

        if stationary_jitter_removed:
            print(f"[WALKER] Removed {len(stationary_jitter_removed)} stationary/jitter walkers")
        if stationary_jitter_frozen:
            print(f"[WALKER] Frozen {len(stationary_jitter_frozen)} stationary/jitter walkers as static")
        if stationary_jitter_preserved:
            print(f"[WALKER] Preserved {len(stationary_jitter_preserved)} stationary/jitter walkers with light smoothing")
        
        report: Dict[str, object] = {
            "enabled": True,
            "walker_count": len(walker_vids),
            "walker_count_after_stationary_removal": len(non_stationary_vids),
            "stationary_removed_count": len(stationary_jitter_removed),
            "stationary_removed": {
                vid: info for vid, info in stationary_jitter_removed.items()
            },
            "stationary_frozen_count": len(stationary_jitter_frozen),
            "stationary_frozen": {
                vid: info for vid, info in stationary_jitter_frozen.items()
            },
            "stationary_preserved_count": len(stationary_jitter_preserved),
            "stationary_preserved": {
                vid: info for vid, info in stationary_jitter_preserved.items()
            },
            "stationary_policy": "freeze" if self.freeze_stationary_jitter else "remove",
            "carla_road_lines": len(self._carla_lines),
            "carla_road_sample_points": int(self._road_points.shape[0]),
            "lane_spacing_m": self.lane_spacing_m,
            "sidewalk_start_distance_m": self.sidewalk_start_distance,
            "sidewalk_outer_distance_m": self.sidewalk_outer_distance,
            "compression_target_band_m": self.compression_target_band_m,
        }
        
        # If all walkers removed, return early
        if not non_stationary_vids:
            report["classifications"] = {}
            report["compressions"] = {}
            report["stabilization"] = {}
            report["classification_summary"] = {
                "sidewalk_consistent": 0, "crossing": 0, "jaywalking": 0, "road_walking": 0, "unclassified": 0
            }
            report["compression_summary"] = {
                "compressed": 0, "skipped": 0, "total_frames_modified": 0, "avg_lateral_offset": 0.0, "max_lateral_offset": 0.0
            }
            if self.freeze_stationary_jitter and (stationary_jitter_frozen or stationary_jitter_preserved):
                report["summary"] = (
                    f"all {len(walker_vids)} walkers detected stationary/jitter "
                    f"(frozen={len(stationary_jitter_frozen)}, preserved={len(stationary_jitter_preserved)})"
                )
            else:
                report["summary"] = f"all {len(walker_vids)} walkers removed as stationary/jitter"
            return updated_vehicles, report
        
        # ---- Continue with non-stationary walkers ----
        # Classify remaining walkers
        classifications: Dict[int, WalkerClassification] = {}
        for vid in non_stationary_vids:
            traj = updated_vehicles[vid]
            times = vehicle_times.get(vid, [])
            classifications[vid] = self.classify_walker(vid, traj, times)
        
        classification_summary = {
            "sidewalk_consistent": sum(1 for c in classifications.values() if c.is_sidewalk_consistent),
            "crossing": sum(1 for c in classifications.values() if c.is_crossing),
            "jaywalking": sum(1 for c in classifications.values() if c.is_jaywalking),
            "road_walking": sum(1 for c in classifications.values() if c.is_road_walking),
            "unclassified": sum(1 for c in classifications.values() if not (
                c.is_sidewalk_consistent or c.is_crossing or c.is_jaywalking or c.is_road_walking
            )),
        }
        report["classification_summary"] = classification_summary
        report["classifications"] = {
            vid: {
                "is_sidewalk_consistent": c.is_sidewalk_consistent,
                "is_crossing": c.is_crossing,
                "is_jaywalking": c.is_jaywalking,
                "is_road_walking": c.is_road_walking,
                "road_occupancy_ratio": c.road_occupancy_ratio,
                "median_lane_distance": c.median_lane_distance,
                "lateral_traversal_distance": c.lateral_traversal_distance,
                "classification_reason": c.classification_reason,
            }
            for vid, c in classifications.items()
        }
        
        # Apply compression to sidewalk-consistent walkers (non-stationary only)
        compression_results: Dict[int, WalkerCompressionResult] = {}
        
        for vid in non_stationary_vids:
            classification = classifications[vid]
            result = self.compress_walker_trajectory(vid, updated_vehicles[vid], classification)
            compression_results[vid] = result
            
            if result.applied:
                updated_vehicles[vid] = result.compressed_traj
        
        compression_summary = {
            "compressed": sum(1 for r in compression_results.values() if r.applied),
            "skipped": sum(1 for r in compression_results.values() if not r.applied),
            "total_frames_modified": sum(r.frames_modified for r in compression_results.values()),
            "avg_lateral_offset": float(np.mean([r.avg_lateral_offset for r in compression_results.values() if r.applied])) if any(r.applied for r in compression_results.values()) else 0.0,
            "max_lateral_offset": float(np.max([r.max_lateral_offset for r in compression_results.values() if r.applied])) if any(r.applied for r in compression_results.values()) else 0.0,
        }
        report["compression_summary"] = compression_summary
        report["compressions"] = {
            vid: {
                "applied": r.applied,
                "avg_lateral_offset": r.avg_lateral_offset,
                "max_lateral_offset": r.max_lateral_offset,
                "frames_modified": r.frames_modified,
                "reason": r.compression_reason,
            }
            for vid, r in compression_results.items()
        }
        
        # Stabilize walker spawns (non-stationary only)
        walker_trajs = {vid: updated_vehicles[vid] for vid in non_stationary_vids if vid in updated_vehicles}
        walker_times_subset = {vid: vehicle_times.get(vid, []) for vid in non_stationary_vids}
        
        stabilized_trajs, stab_result = self.stabilize_walker_spawns(walker_trajs, walker_times_subset)
        
        # Update trajectories with stabilization results
        for vid, traj in stabilized_trajs.items():
            updated_vehicles[vid] = traj
        
        report["stabilization"] = {
            "adjusted_count": stab_result.adjusted_count,
            "conflict_pairs_resolved": stab_result.conflict_pairs_resolved,
            "avg_separation_offset": stab_result.avg_separation_offset,
            "max_separation_offset": stab_result.max_separation_offset,
            "details": {str(k): v for k, v in stab_result.details.items()},
        }
        
        report["summary"] = (
            f"processed {len(walker_vids)} walkers: "
            f"{len(stationary_jitter_removed)} stationary/jitter removed, "
            f"{classification_summary['sidewalk_consistent']} sidewalk, "
            f"{classification_summary['crossing']} crossing, "
            f"{compression_summary['compressed']} compressed, "
            f"{stab_result.adjusted_count} spawn-stabilized"
        )
        
        return updated_vehicles, report


def _is_pedestrian_type(obj_type: str | None) -> bool:
    if not obj_type:
        return False
    ot = str(obj_type).lower()
    return any(token in ot for token in ("pedestrian", "walker", "person", "people"))


def _is_cyclist_type(obj_type: str | None) -> bool:
    if not obj_type:
        return False
    ot = str(obj_type).lower()
    return any(token in ot for token in ("bicycle", "cyclist", "bike", "scooter"))


def _infer_actor_role(obj_type: str | None, traj: Sequence[Waypoint]) -> str:
    if _is_pedestrian_type(obj_type):
        return "walker"
    if _is_cyclist_type(obj_type):
        return "cyclist"
    if len(traj) <= 1:
        return "static"
    return "vehicle"


def _compute_bbox_xy(points: np.ndarray) -> Tuple[float, float, float, float]:
    if points.size == 0:
        return (0.0, 1.0, 0.0, 1.0)
    min_x = float(np.min(points[:, 0]))
    max_x = float(np.max(points[:, 0]))
    min_y = float(np.min(points[:, 1]))
    max_y = float(np.max(points[:, 1]))
    if not math.isfinite(min_x) or not math.isfinite(max_x) or not math.isfinite(min_y) or not math.isfinite(max_y):
        return (0.0, 1.0, 0.0, 1.0)
    if max_x <= min_x:
        max_x = min_x + 1.0
    if max_y <= min_y:
        max_y = min_y + 1.0
    return (min_x, max_x, min_y, max_y)


class _MapStubBase:
    def __init__(self, *args: object, **kwargs: object):
        self.__dict__.update(kwargs)

    def __setstate__(self, state: object) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["state"] = state


class _MapStub(_MapStubBase):
    pass


class _LaneStub(_MapStubBase):
    pass


class _MapPointStub(_MapStubBase):
    pass


class _MapUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "opencood.data_utils.datasets.map.map_types":
            if name == "Map":
                return _MapStub
            if name == "Lane":
                return _LaneStub
            if name == "MapPoint":
                return _MapPointStub
        try:
            return super().find_class(module, name)
        except Exception:
            return _MapStubBase


def _extract_xyz_points(items: object) -> List[Tuple[float, float, float]]:
    out: List[Tuple[float, float, float]] = []
    if not isinstance(items, (list, tuple)):
        return out
    for item in items:
        if hasattr(item, "x") and hasattr(item, "y"):
            x = _safe_float(getattr(item, "x", 0.0))
            y = _safe_float(getattr(item, "y", 0.0))
            z = _safe_float(getattr(item, "z", 0.0))
            if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                out.append((x, y, z))
            continue
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            x = _safe_float(item[0], 0.0)
            y = _safe_float(item[1], 0.0)
            z = _safe_float(item[2], 0.0) if len(item) >= 3 else 0.0
            if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                out.append((x, y, z))
    return out


@dataclass
class LaneFeature:
    index: int
    uid: str
    road_id: int
    lane_id: int
    lane_type: str
    polyline: np.ndarray  # shape (N, 3)
    boundary: np.ndarray  # shape (M, 3)
    entry_lanes: List[str]
    exit_lanes: List[str]

    @property
    def label_xy(self) -> Tuple[float, float]:
        if self.polyline.shape[0] == 0:
            return (0.0, 0.0)
        mid = self.polyline.shape[0] // 2
        return (float(self.polyline[mid, 0]), float(self.polyline[mid, 1]))


@dataclass
class VectorMapData:
    name: str
    source_path: str
    lanes: List[LaneFeature]
    bbox: Tuple[float, float, float, float]

    @property
    def lane_count(self) -> int:
        return len(self.lanes)


@dataclass
class TrackCandidate:
    orig_vid: int
    source_subdir: str
    traj: List[Waypoint]
    times: List[float]
    meta: Dict[str, object]


def _load_vector_map(path: Path) -> VectorMapData:
    with path.open("rb") as f:
        obj = _MapUnpickler(f).load()

    map_features = getattr(obj, "map_features", None)
    if not isinstance(map_features, list):
        raise RuntimeError(f"Map pickle {path} does not expose map_features list.")

    lanes: List[LaneFeature] = []
    all_xy: List[Tuple[float, float]] = []
    for idx, feat in enumerate(map_features):
        road_id = _safe_int(getattr(feat, "road_id", idx), idx)
        lane_id = _safe_int(getattr(feat, "lane_id", idx), idx)
        lane_type = str(getattr(feat, "type", "unknown"))
        uid = f"{road_id}_{lane_id}"

        polyline_pts = _extract_xyz_points(getattr(feat, "polyline", []))
        boundary_pts = _extract_xyz_points(getattr(feat, "boundary", []))
        if len(polyline_pts) < 2:
            continue

        polyline = np.asarray(polyline_pts, dtype=np.float64)
        boundary = np.asarray(boundary_pts, dtype=np.float64) if boundary_pts else np.zeros((0, 3), dtype=np.float64)
        entry_lanes = [str(v) for v in (getattr(feat, "entry_lanes", []) or [])]
        exit_lanes = [str(v) for v in (getattr(feat, "exit_lanes", []) or [])]

        all_xy.extend((float(p[0]), float(p[1])) for p in polyline_pts)
        all_xy.extend((float(p[0]), float(p[1])) for p in boundary_pts)
        lanes.append(
            LaneFeature(
                index=len(lanes),
                uid=uid,
                road_id=road_id,
                lane_id=lane_id,
                lane_type=lane_type,
                polyline=polyline,
                boundary=boundary,
                entry_lanes=entry_lanes,
                exit_lanes=exit_lanes,
            )
        )

    if not lanes:
        raise RuntimeError(f"No lane features found in map pickle: {path}")

    xy_arr = np.asarray(all_xy, dtype=np.float64) if all_xy else np.zeros((0, 2), dtype=np.float64)
    bbox = _compute_bbox_xy(xy_arr)
    return VectorMapData(name=path.stem, source_path=str(path), lanes=lanes, bbox=bbox)


def _as_finite_xy_pair(item: object) -> Optional[Tuple[float, float]]:
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        x = _safe_float(item[0], float("nan"))
        y = _safe_float(item[1], float("nan"))
        if math.isfinite(x) and math.isfinite(y):
            return (float(x), float(y))
    if hasattr(item, "x") and hasattr(item, "y"):
        x = _safe_float(getattr(item, "x", float("nan")), float("nan"))
        y = _safe_float(getattr(item, "y", float("nan")), float("nan"))
        if math.isfinite(x) and math.isfinite(y):
            return (float(x), float(y))
    return None


def _extract_xy_lines_recursive(obj: object, out_lines: List[List[Tuple[float, float]]], depth: int = 0) -> None:
    if obj is None or depth > 10:
        return

    if isinstance(obj, dict):
        if "lines" in obj and isinstance(obj["lines"], (list, tuple)):
            for line in obj["lines"]:
                if not isinstance(line, (list, tuple)):
                    continue
                pts: List[Tuple[float, float]] = []
                for p in line:
                    xy = _as_finite_xy_pair(p)
                    if xy is not None:
                        pts.append(xy)
                if len(pts) >= 2:
                    out_lines.append(pts)
            if out_lines:
                return
        for v in obj.values():
            _extract_xy_lines_recursive(v, out_lines, depth + 1)
        return

    if isinstance(obj, (list, tuple)):
        pts: List[Tuple[float, float]] = []
        for p in obj:
            xy = _as_finite_xy_pair(p)
            if xy is not None:
                pts.append(xy)
            else:
                pts = []
                break
        if len(pts) >= 2:
            out_lines.append(pts)
            return
        for v in obj:
            _extract_xy_lines_recursive(v, out_lines, depth + 1)
        return

    if hasattr(obj, "__dict__"):
        _extract_xy_lines_recursive(getattr(obj, "__dict__", {}), out_lines, depth + 1)


def _normalize_carla_line_records(raw: object) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    if not isinstance(raw, list):
        return records
    for rec in raw:
        if not isinstance(rec, dict):
            continue
        pts_raw = rec.get("points")
        if not isinstance(pts_raw, (list, tuple)):
            continue
        pts: List[Tuple[float, float]] = []
        for p in pts_raw:
            xy = _as_finite_xy_pair(p)
            if xy is None:
                continue
            pts.append((float(xy[0]), float(xy[1])))
        if len(pts) < 2:
            continue
        road_id = rec.get("road_id")
        lane_id = rec.get("lane_id")
        dir_sign = rec.get("dir_sign")
        try:
            road_id = int(road_id) if road_id is not None else None
        except Exception:
            road_id = None
        try:
            lane_id = int(lane_id) if lane_id is not None else None
        except Exception:
            lane_id = None
        try:
            dir_sign = int(dir_sign) if dir_sign is not None else None
        except Exception:
            dir_sign = None
        if dir_sign not in (-1, 1):
            dir_sign = None
        point_yaws = rec.get("point_yaws")
        point_s = rec.get("point_s")
        if not isinstance(point_yaws, (list, tuple)):
            point_yaws = None
        if not isinstance(point_s, (list, tuple)):
            point_s = None
        records.append(
            {
                "points": pts,
                "road_id": road_id,
                "lane_id": lane_id,
                "dir_sign": dir_sign,
                "point_yaws": [float(v) for v in point_yaws] if point_yaws is not None else None,
                "point_s": [float(v) for v in point_s] if point_s is not None else None,
                "direction_source": str(rec.get("direction_source", "")),
                "travel_ordered": bool(rec.get("travel_ordered", False)),
                "order_flipped_vs_s_asc": bool(rec.get("order_flipped_vs_s_asc", False)),
                "yaw_alignment_score": _safe_float(rec.get("yaw_alignment_score"), float("nan")),
            }
        )
    return records


def _load_carla_map_cache(
    path: Path,
) -> Tuple[
    List[List[Tuple[float, float]]],
    Optional[Tuple[float, float, float, float]],
    str,
    List[Dict[str, object]],
]:
    with path.open("rb") as f:
        raw = pickle.load(f)

    map_name = ""
    bounds: Optional[Tuple[float, float, float, float]] = None
    lines: List[List[Tuple[float, float]]] = []
    line_records: List[Dict[str, object]] = []
    if isinstance(raw, dict):
        map_name = str(raw.get("map_name") or "")
        b = raw.get("bounds")
        if isinstance(b, (list, tuple)) and len(b) >= 4:
            bx0 = _safe_float(b[0], float("nan"))
            bx1 = _safe_float(b[1], float("nan"))
            by0 = _safe_float(b[2], float("nan"))
            by1 = _safe_float(b[3], float("nan"))
            if all(math.isfinite(v) for v in (bx0, bx1, by0, by1)):
                bounds = (float(bx0), float(bx1), float(by0), float(by1))
        line_records = _normalize_carla_line_records(raw.get("line_records"))

    if line_records:
        for rec in line_records:
            pts = rec.get("points")
            if isinstance(pts, list) and len(pts) >= 2:
                lines.append([(float(x), float(y)) for x, y in pts])
    else:
        _extract_xy_lines_recursive(raw, lines)
        # Old cache fallback: preserve shape, metadata unknown.
        line_records = [
            {"points": [(float(x), float(y)) for x, y in ln], "road_id": None, "lane_id": None, "dir_sign": None}
            for ln in lines
            if len(ln) >= 2
        ]

    deduped: List[List[Tuple[float, float]]] = []
    deduped_records: List[Dict[str, object]] = []
    for idx, ln in enumerate(lines):
        if len(ln) < 2:
            continue
        pts = [(float(x), float(y)) for x, y in ln]
        deduped.append(pts)
        if idx < len(line_records) and isinstance(line_records[idx], dict):
            rec = dict(line_records[idx])
            rec["points"] = pts
            if rec.get("dir_sign") not in (-1, 1):
                rec["dir_sign"] = None
            deduped_records.append(rec)
        else:
            deduped_records.append(
                {"points": pts, "road_id": None, "lane_id": None, "dir_sign": None}
            )
    return deduped, bounds, map_name, deduped_records


def _load_carla_map_cache_lines(path: Path) -> Tuple[List[List[Tuple[float, float]]], Optional[Tuple[float, float, float, float]], str]:
    lines, bounds, map_name, _ = _load_carla_map_cache(path)
    return lines, bounds, map_name


def _load_carla_alignment_cfg(path: Optional[Path]) -> Dict[str, object]:
    cfg: Dict[str, object] = {
        "scale": 1.0,
        "theta_deg": 0.0,
        "tx": 0.0,
        "ty": 0.0,
        "flip_y": False,
        "source_path": "",
    }
    if path is None or not path.exists():
        return cfg
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return cfg
    theta_deg = (
        float(raw.get("theta_deg", 0.0))
        if "theta_deg" in raw
        else float(raw.get("theta_rad", 0.0)) * 180.0 / math.pi if "theta_rad" in raw else 0.0
    )
    cfg["scale"] = float(raw.get("scale", 1.0))
    cfg["theta_deg"] = float(theta_deg)
    cfg["tx"] = float(raw.get("tx", 0.0))
    cfg["ty"] = float(raw.get("ty", 0.0))
    cfg["flip_y"] = bool(raw.get("flip_y", False) or raw.get("y_flip", False))
    cfg["source_path"] = str(path)
    return cfg


def _transform_carla_lines(
    lines: Sequence[Sequence[Tuple[float, float]]],
    align_cfg: Dict[str, object],
) -> Tuple[List[List[Tuple[float, float]]], Tuple[float, float, float, float]]:
    scale = float(align_cfg.get("scale", 1.0))
    theta_deg = float(align_cfg.get("theta_deg", 0.0))
    tx = float(align_cfg.get("tx", 0.0))
    ty = float(align_cfg.get("ty", 0.0))
    flip_y = bool(align_cfg.get("flip_y", False))

    out: List[List[Tuple[float, float]]] = []
    all_xy: List[Tuple[float, float]] = []
    for ln in lines:
        pts_out: List[Tuple[float, float]] = []
        for x0, y0 in ln:
            x = float(x0) * scale
            y = float(y0) * scale
            xt, yt = apply_se2((x, y), theta_deg, tx, ty, flip_y=flip_y)
            if not (math.isfinite(xt) and math.isfinite(yt)):
                continue
            pts_out.append((float(xt), float(yt)))
        if len(pts_out) >= 2:
            out.append(pts_out)
            all_xy.extend(pts_out)

    arr = np.asarray(all_xy, dtype=np.float64) if all_xy else np.zeros((0, 2), dtype=np.float64)
    bbox = _compute_bbox_xy(arr)
    return out, bbox


def _transform_carla_line_records(
    line_records: Sequence[Dict[str, object]],
    align_cfg: Dict[str, object],
) -> List[Dict[str, object]]:
    scale = float(align_cfg.get("scale", 1.0))
    theta_deg = float(align_cfg.get("theta_deg", 0.0))
    tx = float(align_cfg.get("tx", 0.0))
    ty = float(align_cfg.get("ty", 0.0))
    flip_y = bool(align_cfg.get("flip_y", False))

    out: List[Dict[str, object]] = []
    for rec in line_records:
        if not isinstance(rec, dict):
            continue
        pts_raw = rec.get("points")
        if not isinstance(pts_raw, (list, tuple)):
            continue
        pts_out: List[Tuple[float, float]] = []
        for p in pts_raw:
            xy = _as_finite_xy_pair(p)
            if xy is None:
                continue
            x = float(xy[0]) * scale
            y = float(xy[1]) * scale
            xt, yt = apply_se2((x, y), theta_deg, tx, ty, flip_y=flip_y)
            if not (math.isfinite(xt) and math.isfinite(yt)):
                continue
            pts_out.append((float(xt), float(yt)))
        if len(pts_out) < 2:
            continue
        dir_sign = rec.get("dir_sign")
        try:
            dir_sign = int(dir_sign) if dir_sign is not None else None
        except Exception:
            dir_sign = None
        if dir_sign not in (-1, 1):
            dir_sign = None
        road_id = rec.get("road_id")
        lane_id = rec.get("lane_id")
        try:
            road_id = int(road_id) if road_id is not None else None
        except Exception:
            road_id = None
        try:
            lane_id = int(lane_id) if lane_id is not None else None
        except Exception:
            lane_id = None
        out.append(
            {
                "points": pts_out,
                "road_id": road_id,
                "lane_id": lane_id,
                "dir_sign": dir_sign,
                "point_yaws": (
                    [float(v) for v in rec.get("point_yaws", [])]
                    if isinstance(rec.get("point_yaws"), (list, tuple))
                    else None
                ),
                "point_s": (
                    [float(v) for v in rec.get("point_s", [])]
                    if isinstance(rec.get("point_s"), (list, tuple))
                    else None
                ),
                "direction_source": str(rec.get("direction_source", "")),
                "travel_ordered": bool(rec.get("travel_ordered", False)),
                "order_flipped_vs_s_asc": bool(rec.get("order_flipped_vs_s_asc", False)),
                "yaw_alignment_score": _safe_float(rec.get("yaw_alignment_score"), float("nan")),
            }
        )
    return out


def _downsample_xy_line(points_xy: Sequence[Tuple[float, float]], max_points: int) -> List[List[float]]:
    if not points_xy:
        return []
    arr = np.asarray([[float(x), float(y)] for (x, y) in points_xy], dtype=np.float64)
    if arr.shape[0] <= 0:
        return []
    if max_points <= 0 or arr.shape[0] <= max_points:
        return [[float(p[0]), float(p[1])] for p in arr]
    idx = np.linspace(0, arr.shape[0] - 1, max_points, dtype=np.int32)
    sampled = arr[idx]
    if idx[-1] != arr.shape[0] - 1:
        sampled = np.vstack([sampled, arr[-1:]])
    return [[float(p[0]), float(p[1])] for p in sampled]


def _capture_carla_topdown_image(
    carla_host: str,
    carla_port: int,
    raw_bounds: Tuple[float, float, float, float],
    image_width: int = 8192,
    fov_deg: float = 10.0,
    margin_factor: float = 1.08,
    carla_map_name: str = "ucla_v2",
    tile_fov_deg: float = 15.0,
    tile_px: int = 2048,
    overscan: float = 2.0,
    max_pixel_error: float = 0.5,
) -> Tuple[bytes, Tuple[float, float, float, float]]:
    """Capture a seamless top-down image via tiled rendering.

    Uses many small tiles with a narrow FOV so each tile is nearly
    orthographic.  No overscan/crop — just small enough tiles that
    perspective distortion is negligible.

    Returns (jpeg_bytes, actual_bounds) where actual_bounds is
    (min_x, max_x, min_y, max_y) in raw CARLA world coordinates.
    """
    _ensure_carla_egg_on_path()
    import carla  # type: ignore
    import time as _time

    min_x, max_x, min_y, max_y = raw_bounds
    map_w = (max_x - min_x) * margin_factor
    map_h = (max_y - min_y) * margin_factor
    map_cx = (min_x + max_x) / 2.0
    map_cy = (min_y + max_y) / 2.0

    # Many small tiles → each tile covers a small world area → small
    # off-axis angles → negligible perspective distortion.
    # ~150 m per tile → 8×6 grid for the UCLA map.
    tile_size_m = 150.0
    n_tiles_x = max(2, math.ceil(map_w / tile_size_m))
    n_tiles_y = max(2, math.ceil(map_h / tile_size_m))
    tile_world_w = map_w / n_tiles_x
    tile_world_h = map_h / n_tiles_y

    tile_half_fov = math.radians(tile_fov_deg / 2.0)
    # Altitude so horizontal FOV spans exactly tile_world_w
    tile_altitude = tile_world_w / (2.0 * math.tan(tile_half_fov))

    tile_img_w = tile_px
    tile_img_h = max(2, int(round(tile_px * tile_world_h / tile_world_w)))

    out_w = tile_img_w * n_tiles_x
    out_h = tile_img_h * n_tiles_y

    actual_bounds = (
        map_cx - map_w / 2.0,
        map_cx + map_w / 2.0,
        map_cy - map_h / 2.0,
        map_cy + map_h / 2.0,
    )

    print(
        f"[INFO] Tiled capture: {n_tiles_x}x{n_tiles_y} tiles, "
        f"tile={tile_img_w}x{tile_img_h}px, "
        f"tile_world={tile_world_w:.1f}x{tile_world_h:.1f}m, "
        f"fov={tile_fov_deg}°, altitude={tile_altitude:.1f}m, "
        f"output={out_w}x{out_h}px"
    )

    client = carla.Client(carla_host, carla_port)
    client.set_timeout(30.0)

    # --- Ensure the correct map is loaded ---
    world = client.get_world()
    current_map_name = world.get_map().name
    want_map = str(carla_map_name).strip()
    if want_map.lower() not in current_map_name.lower():
        print(f"[INFO] Current CARLA map is '{current_map_name}', loading '{want_map}'...")
        available_maps = client.get_available_maps()
        target_path = None
        for m in available_maps:
            if want_map.lower() in m.lower():
                target_path = m
                break
        if target_path is None:
            raise RuntimeError(
                f"Map '{want_map}' not found among available CARLA maps: {available_maps}"
            )
        world = client.load_world(target_path)
        _time.sleep(3.0)
        for _ in range(20):
            if world.get_settings().synchronous_mode:
                world.tick()
            else:
                _time.sleep(0.2)
        print(f"[INFO] Loaded CARLA map: {world.get_map().name}")
    else:
        print(f"[INFO] CARLA already has correct map loaded: {current_map_name}")

    # --- Temporarily switch to async mode ---
    original_settings = world.get_settings()
    was_sync = original_settings.synchronous_mode
    if was_sync:
        print("[INFO] CARLA is in synchronous mode; temporarily switching to async for capture...")
        async_settings = world.get_settings()
        async_settings.synchronous_mode = False
        async_settings.fixed_delta_seconds = 0.0
        world.apply_settings(async_settings)
        _time.sleep(1.0)

    # Flat even lighting: sun at zenith, overcast, no fog
    weather = carla.WeatherParameters(
        cloudiness=80.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=0.0,
        sun_azimuth_angle=0.0,
        sun_altitude_angle=90.0,
        fog_density=0.0,
        fog_distance=0.0,
        fog_falloff=0.0,
        wetness=0.0,
        scattering_intensity=1.0,
        mie_scattering_scale=0.0,
        rayleigh_scattering_scale=0.0331,
    )
    world.set_weather(weather)
    print("[INFO] Waiting for CARLA weather/lighting to settle...")
    _time.sleep(5.0)

    bp_lib = world.get_blueprint_library()

    try:
        from PIL import Image as _PILImage  # type: ignore
    except ImportError:
        _PILImage = None

    # --- Capture each tile (no overscan, just small tiles) ---
    tile_arrays: List[List[Optional[np.ndarray]]] = [
        [None] * n_tiles_x for _ in range(n_tiles_y)
    ]

    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            tcx = actual_bounds[0] + (tx + 0.5) * tile_world_w
            tcy = actual_bounds[2] + (ty + 0.5) * tile_world_h

            cam_bp = bp_lib.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", str(tile_img_w))
            cam_bp.set_attribute("image_size_y", str(tile_img_h))
            cam_bp.set_attribute("fov", str(tile_fov_deg))

            transform = carla.Transform(
                carla.Location(x=float(tcx), y=float(tcy), z=float(tile_altitude)),
                carla.Rotation(pitch=-90.0, yaw=-90.0, roll=0.0),
            )
            camera = world.spawn_actor(cam_bp, transform)
            _time.sleep(0.8)

            WARMUP = 15
            frame_count = [0]
            latest_img = [None]

            def _make_cb(fc, li):
                def _on_image(image):
                    fc[0] += 1
                    li[0] = image
                return _on_image

            camera.listen(_make_cb(frame_count, latest_img))
            for _ in range(400):
                _time.sleep(0.05)
                if frame_count[0] >= WARMUP:
                    break
            _time.sleep(0.3)
            camera.stop()
            camera.destroy()

            if latest_img[0] is None:
                print(f"  tile ({tx},{ty})/{n_tiles_x}x{n_tiles_y} FAILED — filling black")
                tile_arrays[ty][tx] = np.zeros((tile_img_h, tile_img_w, 3), dtype=np.uint8)
            else:
                raw_img = latest_img[0]
                arr = np.frombuffer(raw_img.raw_data, dtype=np.uint8).reshape(
                    (raw_img.height, raw_img.width, 4)
                )
                tile_arrays[ty][tx] = arr[:, :, :3].copy()
                print(f"  tile ({tx},{ty})/{n_tiles_x}x{n_tiles_y} OK  frames={frame_count[0]}")

    # --- Restore original settings ---
    if was_sync:
        print("[INFO] Restoring CARLA synchronous mode.")
        world.apply_settings(original_settings)
        _time.sleep(0.5)

    # --- Stitch tiles ---
    rows: List[np.ndarray] = []
    for ty in range(n_tiles_y):
        row_tiles = [tile_arrays[ty][tx] for tx in range(n_tiles_x)]
        rows.append(np.concatenate(row_tiles, axis=1))
    full_rgb = np.concatenate(rows, axis=0)

    # Encode to JPEG
    jpeg_bytes: bytes
    if _PILImage is not None:
        import io as _io
        pil_img = _PILImage.fromarray(full_rgb)
        buf = _io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=92)
        jpeg_bytes = buf.getvalue()
    else:
        import cv2  # type: ignore
        _, enc = cv2.imencode(".jpg", full_rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        jpeg_bytes = bytes(enc)

    print(
        f"[INFO] Captured CARLA top-down image: {full_rgb.shape[1]}x{full_rgb.shape[0]} "
        f"({len(jpeg_bytes)} bytes) tiles={n_tiles_x}x{n_tiles_y} "
        f"altitude={tile_altitude:.1f}m fov={tile_fov_deg}°"
    )
    return jpeg_bytes, actual_bounds


def _ensure_carla_egg_on_path() -> None:
    """If `import carla` fails, locate the CARLA 0.9.12 egg in this workspace and
    add it to sys.path. Idempotent and safe to call before any capture function.
    """
    try:
        import carla  # type: ignore  # noqa: F401
        return
    except ImportError:
        pass
    candidates = [
        Path("/data2/marco/CoLMDriver/carla912/PythonAPI/carla/dist"),
        Path(__file__).resolve().parents[2] / "carla912" / "PythonAPI" / "carla" / "dist",
        Path(__file__).resolve().parents[2] / "carla" / "PythonAPI" / "carla" / "dist",
    ]
    cr = os.environ.get("CARLA_ROOT")
    if cr:
        candidates.insert(0, Path(cr) / "PythonAPI" / "carla" / "dist")
    py_major_minor = f"py{sys.version_info[0]}.{sys.version_info[1]}"
    for d in candidates:
        if not d.is_dir():
            continue
        eggs = sorted(d.glob("carla-*.egg"))
        for egg in eggs:
            if py_major_minor in egg.name:
                if str(egg) not in sys.path:
                    sys.path.insert(0, str(egg))
                    print(f"[INFO] Added CARLA egg to sys.path: {egg}")
                return
    raise ImportError(
        f"CARLA egg for {py_major_minor} not found in known locations. "
        f"Set CARLA_ROOT or install the carla wheel for this Python version."
    )


def _capture_carla_topdown_orthorectified(
    carla_host: str,
    carla_port: int,
    raw_bounds: Tuple[float, float, float, float],
    carla_map_name: str = "ucla_v2",
    altitude: float = 1500.0,
    fov_deg: float = 60.0,
    image_px: int = 16384,
    waypoint_spacing_m: float = 1.0,
    output_px_per_meter: float = 10.0,
    margin_factor: float = 1.05,
    gamma: float = 1.25,
) -> Tuple[bytes, Tuple[float, float, float, float]]:
    """Capture a single perspective top-down image from CARLA and orthorectify it
    using lane waypoints as 3D ground control points.

    Each GCP's true (x, y, z) is queried from `world.get_map().generate_waypoints`,
    so elevation is handled exactly per-lane via a piecewise-linear remap
    (no flat-plane assumption, no tile seams).

    Returns ``(jpeg_bytes, output_bounds)`` where ``output_bounds`` is in raw
    CARLA coordinates and pixels are linearly proportional to ``(x, y)`` world.
    """
    _ensure_carla_egg_on_path()
    import carla  # type: ignore
    import time as _time
    import cv2  # type: ignore
    from scipy.interpolate import LinearNDInterpolator  # type: ignore

    min_x, max_x, min_y, max_y = raw_bounds
    map_w = (max_x - min_x) * margin_factor
    map_h = (max_y - min_y) * margin_factor
    map_cx = (min_x + max_x) / 2.0
    map_cy = (min_y + max_y) / 2.0
    out_bounds = (
        map_cx - map_w / 2.0,
        map_cx + map_w / 2.0,
        map_cy - map_h / 2.0,
        map_cy + map_h / 2.0,
    )
    out_w = max(2, int(round(map_w * output_px_per_meter)))
    out_h = max(2, int(round(map_h * output_px_per_meter)))

    half_fov_rad = math.radians(fov_deg) / 2.0
    ground_span = 2.0 * altitude * math.tan(half_fov_rad)
    if ground_span < max(map_w, map_h) * 0.95:
        # Bump altitude so the FOV covers the whole map with some margin.
        altitude = max(map_w, map_h) / (2.0 * math.tan(half_fov_rad)) * 1.1
        ground_span = 2.0 * altitude * math.tan(half_fov_rad)
        print(f"[INFO] Ortho capture: bumped altitude to {altitude:.0f}m to cover map")

    print(
        f"[INFO] Ortho capture: altitude={altitude:.0f}m fov={fov_deg}° "
        f"capture={image_px}x{image_px}px ground_span≈{ground_span:.0f}m → "
        f"output={out_w}x{out_h}px ({output_px_per_meter} px/m), "
        f"map={map_w:.0f}x{map_h:.0f}m"
    )

    client = carla.Client(carla_host, carla_port)
    client.set_timeout(30.0)
    world = client.get_world()
    current_map_name = world.get_map().name
    want_map = str(carla_map_name).strip()
    if want_map.lower() not in current_map_name.lower():
        print(f"[INFO] Loading CARLA map '{want_map}' ...")
        target_path = None
        for m in client.get_available_maps():
            if want_map.lower() in m.lower():
                target_path = m
                break
        if target_path is None:
            raise RuntimeError(
                f"Map '{want_map}' not found in CARLA available maps"
            )
        world = client.load_world(target_path)
        _time.sleep(3.0)

    # Async mode for sensor capture
    original_settings = world.get_settings()
    was_sync = original_settings.synchronous_mode
    if was_sync:
        async_settings = world.get_settings()
        async_settings.synchronous_mode = False
        async_settings.fixed_delta_seconds = 0.0
        world.apply_settings(async_settings)
        _time.sleep(1.0)

    # Clear afternoon — sun lower in sky for shadow contrast that makes
    # lane markings stand out against asphalt.
    world.set_weather(carla.WeatherParameters(
        cloudiness=10.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=0.0,
        sun_azimuth_angle=140.0,
        sun_altitude_angle=40.0,
        fog_density=0.0,
        fog_distance=0.0,
        fog_falloff=0.0,
        wetness=0.0,
        scattering_intensity=1.0,
        mie_scattering_scale=0.0,
        rayleigh_scattering_scale=0.0331,
    ))
    _time.sleep(4.0)

    # === Capture one frame ===
    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(image_px))
    cam_bp.set_attribute("image_size_y", str(image_px))
    cam_bp.set_attribute("fov", str(fov_deg))
    # Use CARLA's default histogram-based auto exposure (what driving cameras
    # use). Just disable a few cosmetic post-processes that hurt readability.
    for attr, val in [
        ("motion_blur_intensity", "0"),
        ("lens_flare_intensity", "0"),
        ("bloom_intensity", "0"),
    ]:
        if cam_bp.has_attribute(attr):
            try:
                cam_bp.set_attribute(attr, val)
            except Exception:
                pass

    cam_transform = carla.Transform(
        carla.Location(x=float(map_cx), y=float(map_cy), z=float(altitude)),
        carla.Rotation(pitch=-90.0, yaw=-90.0, roll=0.0),
    )
    camera = world.spawn_actor(cam_bp, cam_transform)
    _time.sleep(1.0)

    WARMUP = 15
    state = {"count": 0, "img": None}

    def _on_image(image):
        state["count"] += 1
        state["img"] = image

    camera.listen(_on_image)
    for _ in range(400):
        _time.sleep(0.05)
        if state["count"] >= WARMUP:
            break
    _time.sleep(0.3)
    camera.stop()
    cam_w2c = np.array(
        camera.get_transform().get_inverse_matrix(), dtype=np.float64
    )
    camera.destroy()

    if state["img"] is None:
        raise RuntimeError("Camera failed to deliver any frame for ortho capture")

    raw_img = state["img"]
    captured_rgb = np.frombuffer(raw_img.raw_data, dtype=np.uint8).reshape(
        (raw_img.height, raw_img.width, 4)
    )[:, :, 2::-1].copy()  # BGRA → RGB

    # === Sample 3D waypoints as ground control points ===
    cmap = world.get_map()
    waypoints = cmap.generate_waypoints(float(waypoint_spacing_m))
    if not waypoints:
        raise RuntimeError("CARLA map produced zero waypoints")
    gcps_world = np.array(
        [[w.transform.location.x, w.transform.location.y, w.transform.location.z]
         for w in waypoints],
        dtype=np.float64,
    )

    if was_sync:
        world.apply_settings(original_settings)
        _time.sleep(0.3)

    print(
        f"[INFO] Sampled {len(gcps_world)} GCPs from generate_waypoints "
        f"(z range {gcps_world[:, 2].min():.1f}..{gcps_world[:, 2].max():.1f}m)"
    )

    # === Project each GCP through the captured camera ===
    focal = image_px / (2.0 * math.tan(half_fov_rad))
    K = np.array(
        [[focal, 0.0, image_px / 2.0],
         [0.0, focal, image_px / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    pts_h = np.hstack([gcps_world, np.ones((len(gcps_world), 1))])
    pts_cam_ue = (cam_w2c @ pts_h.T).T[:, :3]
    # UE camera frame (x=fwd, y=right, z=up) → standard pinhole (x=right, y=down, z=fwd)
    pts_std = np.column_stack(
        [pts_cam_ue[:, 1], -pts_cam_ue[:, 2], pts_cam_ue[:, 0]]
    )

    in_front = pts_std[:, 2] > 0.5
    pts_std = pts_std[in_front]
    gcps_world = gcps_world[in_front]
    if len(pts_std) < 4:
        raise RuntimeError(f"Too few GCPs in front of camera: {len(pts_std)}")

    proj = (K @ pts_std.T).T
    src_uv = proj[:, :2] / proj[:, 2:3]

    inside = (
        (src_uv[:, 0] >= 0)
        & (src_uv[:, 0] < image_px)
        & (src_uv[:, 1] >= 0)
        & (src_uv[:, 1] < image_px)
    )
    src_uv = src_uv[inside]
    gcps_world = gcps_world[inside]
    print(f"[INFO] {len(src_uv)} GCPs project inside the captured image")
    if len(src_uv) < 8:
        raise RuntimeError(f"Too few GCPs landed inside the image: {len(src_uv)}")

    # === DIAGNOSTIC: dump source image with projected GCPs as red dots ===
    # Inspect this PNG to confirm projection math: each red dot should land on
    # a lane visible in the captured image. If dots are systematically offset
    # from lanes → projection bug. If dots land on lanes but the warped output
    # is still misaligned → bounds/warp bug.
    debug_dir = os.environ.get("V2X_ORTHO_DEBUG_DIR")
    if debug_dir:
        try:
            import cv2 as _cv2_dbg
            dbg = captured_rgb.copy()
            for u, v in src_uv:
                iu, iv = int(round(u)), int(round(v))
                if 0 <= iu < image_px and 0 <= iv < image_px:
                    _cv2_dbg.circle(dbg, (iu, iv), 4, (255, 0, 0), -1)
            dbg_path = Path(debug_dir).expanduser().resolve()
            dbg_path.mkdir(parents=True, exist_ok=True)
            out_png = dbg_path / "ortho_source_with_gcps.jpg"
            _cv2_dbg.imwrite(str(out_png), dbg[:, :, ::-1],
                             [_cv2_dbg.IMWRITE_JPEG_QUALITY, 88])
            print(f"[DEBUG] Wrote source+GCP overlay: {out_png}")
        except Exception as exc:
            print(f"[DEBUG] Could not write GCP overlay: {exc}")

    # === Map GCPs to ortho target pixels ===
    # Convention matches existing tiled output: pixel (0, 0) = (min_x, min_y) world.
    tgt_u = (gcps_world[:, 0] - out_bounds[0]) * output_px_per_meter
    tgt_v = (gcps_world[:, 1] - out_bounds[2]) * output_px_per_meter

    # === Add 4 boundary GCPs at the output rectangle corners ===
    # Without these, areas off the road network fall outside the convex hull of
    # waypoint GCPs and get black-filled / wildly extrapolated. We project each
    # output-rectangle corner through the captured camera assuming a flat plane
    # at the median lane elevation — that's "right enough" for visual context.
    median_z = float(np.median(gcps_world[:, 2]))
    corners_world = np.array([
        [out_bounds[0], out_bounds[2], median_z],
        [out_bounds[1], out_bounds[2], median_z],
        [out_bounds[0], out_bounds[3], median_z],
        [out_bounds[1], out_bounds[3], median_z],
    ], dtype=np.float64)
    corners_h = np.hstack([corners_world, np.ones((4, 1))])
    corners_cam_ue = (cam_w2c @ corners_h.T).T[:, :3]
    corners_std = np.column_stack(
        [corners_cam_ue[:, 1], -corners_cam_ue[:, 2], corners_cam_ue[:, 0]]
    )
    corners_proj = (K @ corners_std.T).T
    corners_src_uv = corners_proj[:, :2] / corners_proj[:, 2:3]
    # Clamp to image bounds so the warp samples valid pixels even if the
    # corner projects slightly outside (camera FOV margin).
    corners_src_uv[:, 0] = np.clip(corners_src_uv[:, 0], 0.0, image_px - 1.0)
    corners_src_uv[:, 1] = np.clip(corners_src_uv[:, 1], 0.0, image_px - 1.0)
    corners_tgt_u = (corners_world[:, 0] - out_bounds[0]) * output_px_per_meter
    corners_tgt_v = (corners_world[:, 1] - out_bounds[2]) * output_px_per_meter

    src_uv = np.vstack([src_uv, corners_src_uv])
    tgt_u = np.concatenate([tgt_u, corners_tgt_u])
    tgt_v = np.concatenate([tgt_v, corners_tgt_v])
    print(f"[INFO] Added 4 boundary GCPs at z={median_z:.1f}m for full-image coverage")

    # Build forward remap: for each output pixel, which source pixel?
    interp_u = LinearNDInterpolator(
        np.column_stack([tgt_u, tgt_v]), src_uv[:, 0], fill_value=-1.0
    )
    interp_v = LinearNDInterpolator(
        np.column_stack([tgt_u, tgt_v]), src_uv[:, 1], fill_value=-1.0
    )

    yy, xx = np.indices((out_h, out_w))
    map_x = interp_u(xx, yy).astype(np.float32)
    map_y = interp_v(xx, yy).astype(np.float32)

    warped = cv2.remap(
        captured_rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # Gamma correction: gamma > 1 darkens midtones (good for sun-blown captures).
    # Done as a 256-entry LUT so it's basically free.
    if abs(gamma - 1.0) > 0.01:
        inv_g = 1.0 / float(gamma)
        lut = np.array(
            [((i / 255.0) ** inv_g) * 255.0 for i in range(256)],
            dtype=np.uint8,
        )
        warped = cv2.LUT(warped, lut)

    try:
        from PIL import Image as _PILImage  # type: ignore
        import io as _io
        pil_img = _PILImage.fromarray(warped)
        buf = _io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=92)
        jpeg_bytes = buf.getvalue()
    except ImportError:
        _, enc = cv2.imencode(
            ".jpg", warped[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92]
        )
        jpeg_bytes = bytes(enc)

    print(
        f"[INFO] Orthorectified image: {warped.shape[1]}x{warped.shape[0]} "
        f"({len(jpeg_bytes)} bytes)"
    )
    return jpeg_bytes, out_bounds


def _load_or_capture_carla_topdown(
    image_cache_path: Path,
    meta_cache_path: Path,
    carla_host: str = "localhost",
    carla_port: int = 2005,
    raw_bounds: Optional[Tuple[float, float, float, float]] = None,
    capture_enabled: bool = True,
    carla_map_name: str = "ucla_v2",
    tile_fov_deg: float = 30.0,
    tile_px: int = 2048,
    method: str = "ortho",
    ortho_altitude: float = 1500.0,
    ortho_fov_deg: float = 60.0,
    ortho_image_px: int = 16384,
    ortho_waypoint_spacing_m: float = 1.0,
    ortho_px_per_meter: float = 10.0,
    ortho_gamma: float = 1.25,
) -> Optional[Tuple[bytes, Tuple[float, float, float, float]]]:
    """Load a cached CARLA top-down JPEG, or capture one from a running server.

    Returns ``(jpeg_bytes, actual_raw_bounds)`` or *None* if unavailable.
    ``actual_raw_bounds`` is (min_x, max_x, min_y, max_y) in raw CARLA coords.
    """
    # --- try cache first ---
    if image_cache_path.exists() and meta_cache_path.exists():
        try:
            jpeg_bytes = image_cache_path.read_bytes()
            meta = json.loads(meta_cache_path.read_text(encoding="utf-8"))
            b = meta["bounds"]
            cached_bounds = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            print(
                f"[INFO] Loaded cached CARLA top-down image: {image_cache_path} "
                f"({len(jpeg_bytes)} bytes)"
            )
            return jpeg_bytes, cached_bounds
        except Exception as exc:
            print(f"[WARN] Failed to load cached CARLA top-down image: {exc}")

    if not capture_enabled:
        print("[INFO] CARLA top-down image capture disabled and no cache found.")
        return None

    if raw_bounds is None:
        print("[WARN] No raw CARLA map bounds available; cannot capture top-down image.")
        return None

    method_norm = (method or "ortho").strip().lower()
    try:
        if method_norm == "ortho":
            jpeg_bytes, actual_bounds = _capture_carla_topdown_orthorectified(
                carla_host=carla_host,
                carla_port=carla_port,
                raw_bounds=raw_bounds,
                carla_map_name=carla_map_name,
                altitude=ortho_altitude,
                fov_deg=ortho_fov_deg,
                image_px=ortho_image_px,
                waypoint_spacing_m=ortho_waypoint_spacing_m,
                output_px_per_meter=ortho_px_per_meter,
                gamma=ortho_gamma,
            )
            meta_extra = {
                "method": "ortho",
                "altitude": ortho_altitude,
                "fov_deg": ortho_fov_deg,
                "image_px": ortho_image_px,
                "waypoint_spacing_m": ortho_waypoint_spacing_m,
                "px_per_meter": ortho_px_per_meter,
                "gamma": ortho_gamma,
            }
        elif method_norm == "tiled":
            jpeg_bytes, actual_bounds = _capture_carla_topdown_image(
                carla_host=carla_host,
                carla_port=carla_port,
                raw_bounds=raw_bounds,
                carla_map_name=carla_map_name,
                tile_fov_deg=tile_fov_deg,
                tile_px=tile_px,
            )
            meta_extra = {
                "method": "tiled",
                "tile_fov_deg": tile_fov_deg,
                "tile_px": tile_px,
            }
        else:
            raise ValueError(
                f"Unknown carla-topdown method: '{method}' (expected 'ortho' or 'tiled')"
            )
    except Exception as exc:
        print(f"[WARN] Failed to capture CARLA top-down image ({method_norm}): {exc}")
        return None

    # --- save cache ---
    try:
        image_cache_path.parent.mkdir(parents=True, exist_ok=True)
        image_cache_path.write_bytes(jpeg_bytes)
        meta_payload = {"bounds": list(actual_bounds)}
        meta_payload.update(meta_extra)
        meta_cache_path.write_text(
            json.dumps(meta_payload, indent=2),
            encoding="utf-8",
        )
        print(f"[INFO] Cached CARLA top-down image to: {image_cache_path}")
    except Exception as exc:
        print(f"[WARN] Could not write CARLA top-down image cache: {exc}")

    return jpeg_bytes, actual_bounds


def _transform_image_bounds_to_v2xpnp(
    raw_bounds: Tuple[float, float, float, float],
    align_cfg: Dict[str, object],
) -> Dict[str, float]:
    """Transform the raw CARLA image bounds through the same SE2 used for polylines.

    Returns a dict with min_x, max_x, min_y, max_y in V2XPNP coordinate space.
    """
    min_x_raw, max_x_raw, min_y_raw, max_y_raw = raw_bounds
    corners_raw = [
        (min_x_raw, min_y_raw),
        (max_x_raw, min_y_raw),
        (min_x_raw, max_y_raw),
        (max_x_raw, max_y_raw),
    ]
    scale = float(align_cfg.get("scale", 1.0))
    theta_deg = float(align_cfg.get("theta_deg", 0.0))
    tx = float(align_cfg.get("tx", 0.0))
    ty = float(align_cfg.get("ty", 0.0))
    flip_y = bool(align_cfg.get("flip_y", False))

    xs: List[float] = []
    ys: List[float] = []
    for cx_raw, cy_raw in corners_raw:
        x0 = cx_raw * scale
        y0 = cy_raw * scale
        xt, yt = apply_se2((x0, y0), theta_deg, tx, ty, flip_y=flip_y)
        xs.append(xt)
        ys.append(yt)
    return {
        "min_x": float(min(xs)),
        "max_x": float(max(xs)),
        "min_y": float(min(ys)),
        "max_y": float(max(ys)),
    }


class LaneMatcher:
    def __init__(self, map_data: VectorMapData):
        self.map_data = map_data
        self._lane_polys = [lane.polyline for lane in map_data.lanes]

        vertex_xy: List[Tuple[float, float]] = []
        vertex_lane_idx: List[int] = []
        for lane_idx, poly in enumerate(self._lane_polys):
            if poly.shape[0] == 0:
                continue
            for p in poly:
                vertex_xy.append((float(p[0]), float(p[1])))
                vertex_lane_idx.append(lane_idx)
        self.vertex_xy = np.asarray(vertex_xy, dtype=np.float64)
        self.vertex_lane_idx = np.asarray(vertex_lane_idx, dtype=np.int32)
        self.tree = cKDTree(self.vertex_xy) if (cKDTree is not None and self.vertex_xy.shape[0] > 0) else None

    def nearest_vertex_distance(self, points_xy: np.ndarray) -> np.ndarray:
        if points_xy.size == 0:
            return np.zeros((0,), dtype=np.float64)
        if self.tree is not None:
            dists, _ = self.tree.query(points_xy, k=1, workers=-1)
            return np.asarray(dists, dtype=np.float64)
        out = np.empty((points_xy.shape[0],), dtype=np.float64)
        for i, pt in enumerate(points_xy):
            diff = self.vertex_xy - pt[None, :]
            out[i] = float(np.min(np.sqrt(np.sum(diff * diff, axis=1))))
        return out

    def _nearest_vertex_candidates(self, x: float, y: float, top_k: int) -> List[int]:
        if self.vertex_xy.shape[0] == 0:
            return []
        k = max(1, min(int(top_k), int(self.vertex_xy.shape[0])))
        if self.tree is not None:
            dists, idxs = self.tree.query(np.asarray([x, y], dtype=np.float64), k=k)
            if np.isscalar(idxs):
                idx_list = [int(idxs)]
            else:
                idx_list = [int(v) for v in np.asarray(idxs).reshape(-1)]
        else:
            diff = self.vertex_xy - np.asarray([x, y], dtype=np.float64)[None, :]
            d2 = np.sum(diff * diff, axis=1)
            order = np.argsort(d2)[:k]
            idx_list = [int(v) for v in order]

        lane_candidates: List[int] = []
        seen: set[int] = set()
        for idx in idx_list:
            lane_idx = int(self.vertex_lane_idx[idx])
            if lane_idx in seen:
                continue
            seen.add(lane_idx)
            lane_candidates.append(lane_idx)
        return lane_candidates

    @staticmethod
    def _project_point_to_polyline(polyline: np.ndarray, x: float, y: float) -> Optional[Dict[str, float]]:
        if polyline.shape[0] == 0:
            return None
        if polyline.shape[0] == 1:
            px = float(polyline[0, 0])
            py = float(polyline[0, 1])
            pz = float(polyline[0, 2])
            dist = math.hypot(px - x, py - y)
            return {
                "x": px,
                "y": py,
                "z": pz,
                "yaw": 0.0,
                "dist": float(dist),
                "segment_idx": 0.0,
            }

        p0 = polyline[:-1, :2]
        p1 = polyline[1:, :2]
        seg = p1 - p0
        seg_len2 = np.sum(seg * seg, axis=1)
        valid = seg_len2 > 1e-12
        if not np.any(valid):
            px = float(polyline[0, 0])
            py = float(polyline[0, 1])
            pz = float(polyline[0, 2])
            dist = math.hypot(px - x, py - y)
            return {
                "x": px,
                "y": py,
                "z": pz,
                "yaw": 0.0,
                "dist": float(dist),
                "segment_idx": 0.0,
            }

        xy = np.asarray([x, y], dtype=np.float64)
        t = np.zeros((seg.shape[0],), dtype=np.float64)
        t[valid] = np.sum((xy - p0[valid]) * seg[valid], axis=1) / seg_len2[valid]
        t = np.clip(t, 0.0, 1.0)
        proj = p0 + seg * t[:, None]
        diff = proj - xy[None, :]
        dist2 = np.sum(diff * diff, axis=1)
        best = int(np.argmin(dist2))
        best_t = float(t[best])

        snap_x = float(proj[best, 0])
        snap_y = float(proj[best, 1])
        z0 = float(polyline[best, 2])
        z1 = float(polyline[best + 1, 2])
        snap_z = z0 + best_t * (z1 - z0)
        seg_dx = float(seg[best, 0])
        seg_dy = float(seg[best, 1])
        yaw = _normalize_yaw_deg(math.degrees(math.atan2(seg_dy, seg_dx))) if (abs(seg_dx) + abs(seg_dy)) > 1e-9 else 0.0
        return {
            "x": snap_x,
            "y": snap_y,
            "z": float(snap_z),
            "yaw": float(yaw),
            "dist": float(math.sqrt(max(0.0, float(dist2[best])))),
            "segment_idx": float(best + best_t),
        }

    def match(self, x: float, y: float, z: float, lane_top_k: int = 8) -> Optional[Dict[str, object]]:
        lane_candidates = self._nearest_vertex_candidates(float(x), float(y), top_k=max(6, lane_top_k * 3))
        if not lane_candidates:
            return None
        lane_candidates = lane_candidates[: max(1, lane_top_k)]

        best_match: Optional[Dict[str, object]] = None
        for lane_idx in lane_candidates:
            lane = self.map_data.lanes[lane_idx]
            proj = self._project_point_to_polyline(lane.polyline, float(x), float(y))
            if proj is None:
                continue
            match = {
                "lane_index": int(lane_idx),
                "lane_uid": lane.uid,
                "road_id": int(lane.road_id),
                "lane_id": int(lane.lane_id),
                "lane_type": lane.lane_type,
                "x": float(proj["x"]),
                "y": float(proj["y"]),
                "z": float(proj["z"]),
                "yaw": float(proj["yaw"]),
                "dist": float(proj["dist"]),
                "segment_idx": float(proj["segment_idx"]),
                "raw_z": float(z),
            }
            if best_match is None or float(match["dist"]) < float(best_match["dist"]):
                best_match = match
        return best_match

    def project_to_lane(self, lane_idx: int, x: float, y: float, z: float) -> Optional[Dict[str, object]]:
        lane_i = int(lane_idx)
        if lane_i < 0 or lane_i >= len(self.map_data.lanes):
            return None
        lane = self.map_data.lanes[lane_i]
        proj = self._project_point_to_polyline(lane.polyline, float(x), float(y))
        if proj is None:
            return None
        return {
            "lane_index": int(lane_i),
            "lane_uid": lane.uid,
            "road_id": int(lane.road_id),
            "lane_id": int(lane.lane_id),
            "lane_type": lane.lane_type,
            "x": float(proj["x"]),
            "y": float(proj["y"]),
            "z": float(proj["z"]),
            "yaw": float(proj["yaw"]),
            "dist": float(proj["dist"]),
            "segment_idx": float(proj["segment_idx"]),
            "raw_z": float(z),
        }

    def match_candidates(self, x: float, y: float, z: float, lane_top_k: int = 8) -> List[Dict[str, object]]:
        lane_candidates = self._nearest_vertex_candidates(float(x), float(y), top_k=max(6, lane_top_k * 3))
        if not lane_candidates:
            return []
        lane_candidates = lane_candidates[: max(1, lane_top_k)]
        out: List[Dict[str, object]] = []
        for lane_idx in lane_candidates:
            m = self.project_to_lane(int(lane_idx), float(x), float(y), float(z))
            if m is not None:
                out.append(m)
        out.sort(key=lambda m: float(m.get("dist", float("inf"))))
        return out


def _sample_points(points_xy: np.ndarray, max_count: int) -> np.ndarray:
    if points_xy.shape[0] <= max_count:
        return points_xy
    if max_count <= 0:
        return points_xy
    idx = np.linspace(0, points_xy.shape[0] - 1, max_count, dtype=np.int32)
    return points_xy[idx]


def _collect_reference_points(ego_trajs: Sequence[Sequence[Waypoint]], vehicles: Dict[int, Sequence[Waypoint]]) -> np.ndarray:
    pts: List[Tuple[float, float]] = []
    for traj in ego_trajs:
        for wp in traj:
            pts.append((float(wp.x), float(wp.y)))
    if pts:
        return np.asarray(pts, dtype=np.float64)
    for traj in vehicles.values():
        for wp in traj:
            pts.append((float(wp.x), float(wp.y)))
    if not pts:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)


def _outside_bbox_ratio(points_xy: np.ndarray, bbox: Tuple[float, float, float, float], margin: float) -> float:
    if points_xy.size == 0:
        return 1.0
    min_x, max_x, min_y, max_y = bbox
    margin = max(0.0, float(margin))
    outside = (
        (points_xy[:, 0] < (min_x - margin))
        | (points_xy[:, 0] > (max_x + margin))
        | (points_xy[:, 1] < (min_y - margin))
        | (points_xy[:, 1] > (max_y + margin))
    )
    return float(np.mean(outside.astype(np.float64)))


def _select_best_map(
    maps: Sequence[VectorMapData],
    ego_trajs: Sequence[Sequence[Waypoint]],
    vehicles: Dict[int, Sequence[Waypoint]],
    sample_count: int,
    bbox_margin: float,
) -> Tuple[VectorMapData, List[Dict[str, object]]]:
    points_xy = _collect_reference_points(ego_trajs, vehicles)
    points_xy = _sample_points(points_xy, max(32, int(sample_count)))
    if points_xy.size == 0:
        raise RuntimeError("No trajectory points available for map selection.")

    details: List[Dict[str, object]] = []
    for map_data in maps:
        matcher = LaneMatcher(map_data)
        dists = matcher.nearest_vertex_distance(points_xy)
        median_dist = float(np.median(dists)) if dists.size else 1e9
        mean_dist = float(np.mean(dists)) if dists.size else 1e9
        p90_dist = float(np.quantile(dists, 0.9)) if dists.size else 1e9
        outside_ratio = _outside_bbox_ratio(points_xy, map_data.bbox, margin=float(bbox_margin))
        score = median_dist + 0.25 * mean_dist + 60.0 * outside_ratio
        details.append(
            {
                "name": map_data.name,
                "source_path": map_data.source_path,
                "sample_points": int(points_xy.shape[0]),
                "median_nearest_m": float(median_dist),
                "mean_nearest_m": float(mean_dist),
                "p90_nearest_m": float(p90_dist),
                "outside_bbox_ratio": float(outside_ratio),
                "score": float(score),
            }
        )

    details.sort(key=lambda x: float(x["score"]))
    chosen_name = str(details[0]["name"])
    chosen_map = next(m for m in maps if m.name == chosen_name)
    return chosen_map, details


def _sample_traj_xy(traj: Sequence[Waypoint], alpha: float) -> Tuple[float, float]:
    if not traj:
        return (0.0, 0.0)
    if len(traj) == 1:
        return (float(traj[0].x), float(traj[0].y))
    clamped = min(1.0, max(0.0, float(alpha)))
    idx = int(round(clamped * float(len(traj) - 1)))
    idx = min(len(traj) - 1, max(0, idx))
    wp = traj[idx]
    return (float(wp.x), float(wp.y))


def _trajectory_similarity_distance(a: Sequence[Waypoint], b: Sequence[Waypoint]) -> float:
    """
    Distance between two trajectories using start/mid/end correspondence.
    Handles opposite direction by checking both forward and reversed pairing.
    """
    if not a or not b:
        return float("inf")
    alphas = (0.0, 0.5, 1.0)
    a_sig = [_sample_traj_xy(a, alpha) for alpha in alphas]
    b_sig = [_sample_traj_xy(b, alpha) for alpha in alphas]

    forward = 0.0
    reverse = 0.0
    for i in range(len(alphas)):
        ax, ay = a_sig[i]
        bfx, bfy = b_sig[i]
        brx, bry = b_sig[len(alphas) - 1 - i]
        forward = max(forward, math.hypot(ax - bfx, ay - bfy))
        reverse = max(reverse, math.hypot(ax - brx, ay - bry))
    return float(min(forward, reverse))


def _merge_actor_meta(acc: Dict[str, object], incoming: Dict[str, object]) -> Dict[str, object]:
    out = dict(acc)
    if not out.get("obj_type") and incoming.get("obj_type"):
        out["obj_type"] = incoming.get("obj_type")
    if not out.get("model") and incoming.get("model"):
        out["model"] = incoming.get("model")
    if out.get("length") is None and incoming.get("length") is not None:
        out["length"] = incoming.get("length")
    if out.get("width") is None and incoming.get("width") is not None:
        out["width"] = incoming.get("width")
    return out


def _build_actor_meta_for_timing_optimization(
    vehicles: Dict[int, List[Waypoint]],
    obj_info: Dict[int, Dict[str, object]],
) -> Dict[int, Dict[str, object]]:
    """
    Build actor metadata in the format expected by yaml_to_carla_log timing optimizers.
    """
    actor_meta: Dict[int, Dict[str, object]] = {}
    for vid, traj in vehicles.items():
        if not traj:
            continue
        meta = dict(obj_info.get(int(vid), {}))
        obj_type = str(meta.get("obj_type") or "npc")
        if not is_vehicle_type(obj_type):
            continue
        role = _infer_actor_role(obj_type, traj)
        if role.startswith("walker"):
            kind = "walker"
        elif role == "static":
            kind = "static"
        else:
            kind = "npc"
        actor_meta[int(vid)] = {
            "kind": str(kind),
            "length": meta.get("length"),
            "width": meta.get("width"),
            "model": str(meta.get("model") or map_obj_type(obj_type)),
        }
    return actor_meta


def _augment_timing_inputs_with_ego_blockers(
    vehicles: Dict[int, List[Waypoint]],
    vehicle_times: Dict[int, List[float]],
    actor_meta: Dict[int, Dict[str, object]],
    ego_trajs: Sequence[Sequence[Waypoint]],
    ego_times: Sequence[Sequence[float]],
    dt: float,
) -> Tuple[
    Dict[int, List[Waypoint]],
    Dict[int, List[float]],
    Dict[int, Dict[str, object]],
    Dict[int, str],
]:
    """
    Add ego trajectories as non-adjustable timing blockers for spawn/despawn optimization.

    These blockers are included in conflict checks, but tagged with ``timing_blocker=True``
    so they are never selected for spawn/despawn adjustment.
    """
    def _copy_wp(wp: Waypoint) -> Waypoint:
        return Waypoint(
            x=float(wp.x),
            y=float(wp.y),
            z=float(wp.z),
            yaw=float(wp.yaw),
            pitch=float(getattr(wp, "pitch", 0.0)),
            roll=float(getattr(wp, "roll", 0.0)),
        )

    out_vehicles: Dict[int, List[Waypoint]] = {
        int(vid): [_copy_wp(wp) for wp in traj]
        for vid, traj in vehicles.items()
    }
    out_times: Dict[int, List[float]] = {
        int(vid): [float(t) for t in (vehicle_times.get(int(vid)) or [])]
        for vid in vehicles.keys()
    }
    out_meta: Dict[int, Dict[str, object]] = {
        int(vid): dict(meta) for vid, meta in actor_meta.items()
    }
    blocker_labels: Dict[int, str] = {}

    used_ids = set(out_vehicles.keys()) | set(out_meta.keys())
    next_blocker_id = -1
    while next_blocker_id in used_ids:
        next_blocker_id -= 1

    for ego_idx, traj in enumerate(ego_trajs):
        if not traj:
            continue
        blocker_id = int(next_blocker_id)
        while blocker_id in used_ids:
            blocker_id -= 1
        next_blocker_id = blocker_id - 1
        used_ids.add(blocker_id)

        copied_traj = [_copy_wp(wp) for wp in traj]
        src_times = ego_times[ego_idx] if ego_idx < len(ego_times) else []
        if src_times and len(src_times) == len(copied_traj):
            copied_times = [float(t) for t in src_times]
        else:
            copied_times = [float(i) * float(dt) for i in range(len(copied_traj))]

        out_vehicles[blocker_id] = copied_traj
        out_times[blocker_id] = copied_times
        out_meta[blocker_id] = {
            "kind": "npc",
            "length": 4.9,
            "width": 2.0,
            "model": "vehicle.lincoln.mkz_2020",
            "timing_blocker": True,
            "source": "ego",
        }
        blocker_labels[blocker_id] = f"ego_{ego_idx}"

    return out_vehicles, out_times, out_meta, blocker_labels


def _merge_subdir_trajectories(
    yaml_dirs: Sequence[Path],
    dt: float,
    id_merge_distance_m: float,
) -> Tuple[
    Dict[int, List[Waypoint]],
    Dict[int, List[float]],
    List[List[Waypoint]],
    List[List[float]],
    Dict[int, Dict[str, object]],
    Dict[int, str],
    Dict[int, int],
    Dict[str, object],
]:
    vehicles: Dict[int, List[Waypoint]] = {}
    vehicle_times: Dict[int, List[float]] = {}
    ego_trajs: List[List[Waypoint]] = []
    ego_times: List[List[float]] = []
    obj_info: Dict[int, Dict[str, object]] = {}
    actor_source_subdir: Dict[int, str] = {}
    actor_orig_vid: Dict[int, int] = {}
    candidates_by_vid: Dict[int, List[TrackCandidate]] = {}

    for yd in yaml_dirs:
        is_negative = _is_negative_subdir(yd)
        v_map, v_times, ego_traj, ego_time, v_info = build_trajectories(
            yaml_dir=yd,
            dt=float(dt),
            tx=0.0,
            ty=0.0,
            tz=0.0,
            yaw_deg=0.0,
            flip_y=False,
        )

        if ego_traj and not is_negative:
            ego_trajs.append(ego_traj)
            ego_times.append(ego_time)

        for vid, traj in v_map.items():
            if not traj:
                continue
            candidates_by_vid.setdefault(int(vid), []).append(
                TrackCandidate(
                    orig_vid=int(vid),
                    source_subdir=yd.name,
                    traj=list(traj),
                    times=list(v_times.get(vid, [])),
                    meta=dict(v_info.get(vid, {})),
                )
            )

    max_vid = max(candidates_by_vid.keys(), default=-1)
    next_vid = max_vid + 1
    merge_distance = max(0.0, float(id_merge_distance_m))

    groups_with_collisions = 0
    merged_duplicates = 0
    split_tracks_created = 0
    total_input_tracks = 0

    for orig_vid in sorted(candidates_by_vid.keys()):
        candidates = sorted(
            candidates_by_vid[orig_vid],
            key=lambda c: (-len(c.traj), c.source_subdir),
        )
        total_input_tracks += len(candidates)

        if len(candidates) > 1:
            groups_with_collisions += 1

        # Cluster tracks with same raw vid by trajectory similarity.
        clusters: List[Dict[str, object]] = []
        for cand in candidates:
            best_idx = -1
            best_dist = float("inf")
            for idx, cluster in enumerate(clusters):
                rep = cluster["representative"]
                dist = _trajectory_similarity_distance(cand.traj, rep.traj)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx >= 0 and best_dist <= merge_distance:
                cluster = clusters[best_idx]
                members = cluster["members"]
                members.append(cand)
                rep = cluster["representative"]
                if len(cand.traj) > len(rep.traj):
                    cluster["representative"] = cand
            elif best_idx >= 0 and merge_distance <= 0.0:
                cluster = clusters[best_idx]
                members = cluster["members"]
                members.append(cand)
                rep = cluster["representative"]
                if len(cand.traj) > len(rep.traj):
                    cluster["representative"] = cand
            else:
                clusters.append({"representative": cand, "members": [cand]})

        merged_duplicates += max(0, len(candidates) - len(clusters))
        split_tracks_created += max(0, len(clusters) - 1)

        clusters = sorted(
            clusters,
            key=lambda c: (
                -len(c["representative"].traj),
                -len(c["members"]),
                c["representative"].source_subdir,
            ),
        )
        for ci, cluster in enumerate(clusters):
            representative = cluster["representative"]
            members = cluster["members"]
            assigned_vid = int(orig_vid) if ci == 0 else int(next_vid)
            if ci > 0:
                next_vid += 1

            merged_meta: Dict[str, object] = {}
            for member in members:
                merged_meta = _merge_actor_meta(merged_meta, member.meta)

            vehicles[assigned_vid] = list(representative.traj)
            vehicle_times[assigned_vid] = list(representative.times)
            actor_source_subdir[assigned_vid] = representative.source_subdir
            obj_info[assigned_vid] = merged_meta
            actor_orig_vid[assigned_vid] = int(orig_vid)

    for vid, meta in obj_info.items():
        if not meta.get("obj_type"):
            meta["obj_type"] = "npc"
        if not meta.get("model"):
            meta["model"] = map_obj_type(str(meta.get("obj_type") or "npc"))

    merge_stats = {
        "input_tracks": int(total_input_tracks),
        "output_tracks": int(len(vehicles)),
        "ids_with_collisions": int(groups_with_collisions),
        "merged_duplicates": int(merged_duplicates),
        "split_tracks_created": int(split_tracks_created),
        "id_merge_distance_m": float(merge_distance),
    }
    return vehicles, vehicle_times, ego_trajs, ego_times, obj_info, actor_source_subdir, actor_orig_vid, merge_stats


def _actor_role_for_dedup(obj_type: str | None, traj: Sequence[Waypoint]) -> str:
    role = _infer_actor_role(obj_type, traj)
    role = str(role).lower()
    if role.startswith("walker"):
        return "walker"
    if role in ("vehicle", "npc"):
        return "vehicle"
    if role == "cyclist":
        return "cyclist"
    if role == "static":
        return "static"
    return role


def _track_ticks(times: Sequence[float] | None, n: int, dt: float) -> List[int]:
    if times and len(times) == n:
        base_dt = max(1e-6, float(dt))
        ticks: List[int] = []
        for t in times:
            ticks.append(int(round(float(t) / base_dt)))
        return ticks
    return list(range(int(n)))


def _pair_overlap_metrics(
    traj_a: Sequence[Waypoint],
    times_a: Sequence[float] | None,
    traj_b: Sequence[Waypoint],
    times_b: Sequence[float] | None,
    dt: float,
) -> Optional[Dict[str, float]]:
    if not traj_a or not traj_b:
        return None

    ticks_a = _track_ticks(times_a, len(traj_a), dt=float(dt))
    ticks_b = _track_ticks(times_b, len(traj_b), dt=float(dt))
    idx_a: Dict[int, int] = {}
    idx_b: Dict[int, int] = {}
    for i, tk in enumerate(ticks_a):
        if tk not in idx_a:
            idx_a[tk] = i
    for i, tk in enumerate(ticks_b):
        if tk not in idx_b:
            idx_b[tk] = i

    common_ticks = sorted(set(idx_a.keys()).intersection(idx_b.keys()))
    if not common_ticks:
        return None

    dists: List[float] = []
    yaw_diffs: List[float] = []
    for tk in common_ticks:
        wa = traj_a[idx_a[tk]]
        wb = traj_b[idx_b[tk]]
        dists.append(math.hypot(float(wa.x) - float(wb.x), float(wa.y) - float(wb.y)))
        yaw_diffs.append(abs(_normalize_yaw_deg(float(wa.yaw) - float(wb.yaw))))

    arr_d = np.asarray(dists, dtype=np.float64)
    arr_yaw = np.asarray(yaw_diffs, dtype=np.float64)
    common_n = int(len(common_ticks))
    overlap_a = float(common_n) / max(1, len(idx_a))
    overlap_b = float(common_n) / max(1, len(idx_b))
    return {
        "common_points": float(common_n),
        "overlap_ratio_a": float(overlap_a),
        "overlap_ratio_b": float(overlap_b),
        "median_dist_m": float(np.median(arr_d)),
        "p90_dist_m": float(np.quantile(arr_d, 0.9)),
        "max_dist_m": float(np.max(arr_d)),
        "median_yaw_diff_deg": float(np.median(arr_yaw)),
        "p90_yaw_diff_deg": float(np.quantile(arr_yaw, 0.9)),
    }


def _deduplicate_cross_id_tracks(
    vehicles: Dict[int, List[Waypoint]],
    vehicle_times: Dict[int, List[float]],
    obj_info: Dict[int, Dict[str, object]],
    actor_source_subdir: Dict[int, str],
    actor_orig_vid: Dict[int, int],
    dt: float,
    max_median_dist_m: float,
    max_p90_dist_m: float,
    max_median_yaw_diff_deg: float,
    min_common_points: int,
    min_overlap_ratio_each: float,
    min_overlap_ratio_any: float,
) -> Tuple[
    Dict[int, List[Waypoint]],
    Dict[int, List[float]],
    Dict[int, Dict[str, object]],
    Dict[int, str],
    Dict[int, int],
    Dict[int, List[int]],
    Dict[str, object],
]:
    if not vehicles:
        return vehicles, vehicle_times, obj_info, actor_source_subdir, actor_orig_vid, {}, {
            "cross_id_dedup_enabled": True,
            "cross_id_pair_checks": 0,
            "cross_id_candidate_pairs": 0,
            "cross_id_clusters": 0,
            "cross_id_removed": 0,
            "cross_id_removed_ids": [],
            "cross_id_max_median_dist_m": float(max_median_dist_m),
            "cross_id_max_p90_dist_m": float(max_p90_dist_m),
            "cross_id_max_median_yaw_diff_deg": float(max_median_yaw_diff_deg),
            "cross_id_min_common_points": int(min_common_points),
            "cross_id_min_overlap_ratio_each": float(min_overlap_ratio_each),
            "cross_id_min_overlap_ratio_any": float(min_overlap_ratio_any),
        }

    # Copy so caller keeps deterministic ownership.
    vehicles = {int(k): list(v) for k, v in vehicles.items()}
    vehicle_times = {int(k): list(v) for k, v in vehicle_times.items()}
    obj_info = {int(k): dict(v) for k, v in obj_info.items()}
    actor_source_subdir = {int(k): str(v) for k, v in actor_source_subdir.items()}
    actor_orig_vid = {int(k): int(v) for k, v in actor_orig_vid.items()}

    actor_alias_vids: Dict[int, List[int]] = {int(vid): [int(vid)] for vid in vehicles.keys()}
    ids = sorted(int(v) for v in vehicles.keys())
    id_to_pos = {vid: i for i, vid in enumerate(ids)}
    parents = list(range(len(ids)))

    def _find(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def _union(i: int, j: int) -> None:
        ri = _find(i)
        rj = _find(j)
        if ri == rj:
            return
        if ri < rj:
            parents[rj] = ri
        else:
            parents[ri] = rj

    role_by_id: Dict[int, str] = {}
    for vid in ids:
        meta = obj_info.get(int(vid), {})
        role_by_id[int(vid)] = _actor_role_for_dedup(str(meta.get("obj_type") or ""), vehicles[int(vid)])

    ids_by_role: Dict[str, List[int]] = {}
    for vid in ids:
        ids_by_role.setdefault(role_by_id.get(vid, "unknown"), []).append(int(vid))

    pair_checks = 0
    candidate_pairs = 0
    for role, role_ids in ids_by_role.items():
        if len(role_ids) < 2:
            continue
        role_ids = sorted(role_ids)
        for i in range(len(role_ids)):
            va = role_ids[i]
            for j in range(i + 1, len(role_ids)):
                vb = role_ids[j]
                pair_checks += 1
                metrics = _pair_overlap_metrics(
                    traj_a=vehicles[va],
                    times_a=vehicle_times.get(va),
                    traj_b=vehicles[vb],
                    times_b=vehicle_times.get(vb),
                    dt=float(dt),
                )
                if metrics is None:
                    continue

                common_points = int(round(float(metrics["common_points"])))
                overlap_a = float(metrics["overlap_ratio_a"])
                overlap_b = float(metrics["overlap_ratio_b"])
                median_dist = float(metrics["median_dist_m"])
                p90_dist = float(metrics["p90_dist_m"])
                median_yaw = float(metrics["median_yaw_diff_deg"])

                if common_points < int(min_common_points):
                    continue
                if max(overlap_a, overlap_b) < float(min_overlap_ratio_any):
                    continue
                if median_dist > float(max_median_dist_m):
                    continue
                if p90_dist > float(max_p90_dist_m):
                    continue
                overlap_each_ok = min(overlap_a, overlap_b) >= float(min_overlap_ratio_each)
                if not overlap_each_ok:
                    # Allow subset-style duplicates (one id track is mostly contained in another)
                    # when geometry/yaw similarity is very strong over a sufficiently long overlap.
                    strict_subset_match = (
                        common_points >= max(int(min_common_points) + 4, 12)
                        and max(overlap_a, overlap_b) >= max(float(min_overlap_ratio_any), 0.85)
                        and median_dist <= min(float(max_median_dist_m) * 0.65, 0.85)
                        and p90_dist <= min(float(max_p90_dist_m) * 0.70, 1.30)
                        and (role == "walker" or median_yaw <= min(float(max_median_yaw_diff_deg), 12.0))
                    )
                    if not strict_subset_match:
                        continue
                # Pedestrian yaw from labels is often noisy; skip yaw-gating for walkers.
                if role != "walker" and median_yaw > float(max_median_yaw_diff_deg):
                    continue

                candidate_pairs += 1
                _union(id_to_pos[va], id_to_pos[vb])

    clusters: Dict[int, List[int]] = {}
    for vid in ids:
        root = _find(id_to_pos[vid])
        clusters.setdefault(root, []).append(int(vid))

    removed_ids: List[int] = []
    merged_clusters = 0
    for members in sorted(clusters.values(), key=lambda arr: (len(arr), arr), reverse=True):
        if len(members) <= 1:
            continue
        merged_clusters += 1

        def _quality_key(vid: int) -> Tuple[float, int, int]:
            traj = vehicles.get(int(vid), [])
            times = vehicle_times.get(int(vid), [])
            n_t = len(times) if times else len(traj)
            path_len = _trajectory_path_length(traj)
            return (float(path_len), int(n_t), int(len(traj)))

        representative = sorted(members, key=lambda vid: (_quality_key(int(vid)), -int(vid)), reverse=True)[0]
        rep = int(representative)

        merged_meta = dict(obj_info.get(rep, {}))
        merged_aliases: List[int] = list(actor_alias_vids.get(rep, [rep]))
        sources = [actor_source_subdir.get(rep, "")]
        for vid in members:
            v = int(vid)
            if v == rep:
                continue
            merged_meta = _merge_actor_meta(merged_meta, obj_info.get(v, {}))
            merged_aliases.extend(actor_alias_vids.get(v, [v]))
            if actor_source_subdir.get(v):
                sources.append(actor_source_subdir.get(v, ""))
            removed_ids.append(v)

        obj_info[rep] = merged_meta
        actor_alias_vids[rep] = sorted(set(int(v) for v in merged_aliases))
        # Preserve primary source while keeping provenance in metadata.
        rep_source = actor_source_subdir.get(rep, "")
        source_unique = sorted(set(s for s in sources if s))
        if rep_source:
            actor_source_subdir[rep] = rep_source
        elif source_unique:
            actor_source_subdir[rep] = source_unique[0]
        if source_unique:
            obj_info[rep]["_merged_sources"] = source_unique

        for vid in members:
            v = int(vid)
            if v == rep:
                continue
            vehicles.pop(v, None)
            vehicle_times.pop(v, None)
            obj_info.pop(v, None)
            actor_source_subdir.pop(v, None)
            actor_orig_vid.pop(v, None)
            actor_alias_vids.pop(v, None)

    stats = {
        "cross_id_dedup_enabled": True,
        "cross_id_pair_checks": int(pair_checks),
        "cross_id_candidate_pairs": int(candidate_pairs),
        "cross_id_clusters": int(merged_clusters),
        "cross_id_removed": int(len(removed_ids)),
        "cross_id_removed_ids": [int(v) for v in sorted(set(removed_ids))],
        "cross_id_max_median_dist_m": float(max_median_dist_m),
        "cross_id_max_p90_dist_m": float(max_p90_dist_m),
        "cross_id_max_median_yaw_diff_deg": float(max_median_yaw_diff_deg),
        "cross_id_min_common_points": int(min_common_points),
        "cross_id_min_overlap_ratio_each": float(min_overlap_ratio_each),
        "cross_id_min_overlap_ratio_any": float(min_overlap_ratio_any),
    }
    return vehicles, vehicle_times, obj_info, actor_source_subdir, actor_orig_vid, actor_alias_vids, stats


def _dominant_lane(lanes: Sequence[int]) -> int:
    counts: Dict[int, int] = {}
    for lane in lanes:
        lid = int(lane)
        if lid < 0:
            continue
        counts[lid] = counts.get(lid, 0) + 1
    if not counts:
        return -1
    return max(counts.items(), key=lambda kv: (int(kv[1]), -int(kv[0])))[0]


def _parse_lane_type_set(value: object, fallback: Sequence[str]) -> set[str]:
    if value is None:
        return {str(v).strip() for v in fallback if str(v).strip()}
    if isinstance(value, (list, tuple, set)):
        items = [str(v).strip() for v in value]
    else:
        items = [tok.strip() for tok in re.split(r"[,\s]+", str(value)) if tok.strip()]
    out = {tok for tok in items if tok}
    if not out:
        out = {str(v).strip() for v in fallback if str(v).strip()}
    return out


def _compute_motion_stats_xy(
    traj: Sequence[Waypoint],
    times: Sequence[float] | None,
    default_dt: float,
) -> Dict[str, float]:
    n = int(len(traj))
    if n <= 0:
        return {
            "num_points": 0.0,
            "duration_s": 0.0,
            "net_disp_m": 0.0,
            "path_len_m": 0.0,
            "radius_p90_m": 0.0,
            "radius_max_m": 0.0,
            "step_p95_m": 0.0,
            "large_step_ratio": 0.0,
            "max_from_start_m": 0.0,
        }

    xs = np.asarray([float(wp.x) for wp in traj], dtype=np.float64)
    ys = np.asarray([float(wp.y) for wp in traj], dtype=np.float64)
    pts = np.stack([xs, ys], axis=1)
    if times and len(times) == n:
        duration_s = max(0.0, float(times[-1]) - float(times[0]))
    else:
        duration_s = max(0.0, float(default_dt) * max(0, n - 1))

    if n >= 2:
        diffs = pts[1:] - pts[:-1]
        steps = np.sqrt(np.sum(diffs * diffs, axis=1))
        path_len = float(np.sum(steps))
    else:
        steps = np.zeros((0,), dtype=np.float64)
        path_len = 0.0

    net_disp = float(math.hypot(float(xs[-1] - xs[0]), float(ys[-1] - ys[0]))) if n >= 2 else 0.0
    center = np.median(pts, axis=0)
    radii = np.sqrt(np.sum((pts - center[None, :]) ** 2, axis=1))
    radius_p90 = float(np.quantile(radii, 0.9)) if radii.size else 0.0
    radius_max = float(np.max(radii)) if radii.size else 0.0
    step_p95 = float(np.quantile(steps, 0.95)) if steps.size else 0.0
    max_from_start = float(np.max(np.sqrt(np.sum((pts - pts[0:1]) ** 2, axis=1)))) if n >= 1 else 0.0

    return {
        "num_points": float(n),
        "duration_s": float(duration_s),
        "net_disp_m": float(net_disp),
        "path_len_m": float(path_len),
        "radius_p90_m": float(radius_p90),
        "radius_max_m": float(radius_max),
        "step_p95_m": float(step_p95),
        "large_step_ratio": 0.0,  # filled by caller based on configured threshold
        "max_from_start_m": float(max_from_start),
    }


def _max_contiguous_true(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    best = 0
    run = 0
    for v in mask.astype(bool).tolist():
        if v:
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return int(best)


def _compute_robust_stationary_cluster_stats(
    traj: Sequence[Waypoint],
    eps_m: float,
    large_step_threshold_m: float,
) -> Dict[str, float]:
    n = int(len(traj))
    if n <= 0:
        return {
            "valid": 0.0,
            "inlier_ratio": 0.0,
            "outlier_ratio": 0.0,
            "max_outlier_run": 0.0,
            "inlier_net_disp_m": 0.0,
            "inlier_path_len_m": 0.0,
            "inlier_radius_p90_m": 0.0,
            "inlier_radius_max_m": 0.0,
            "inlier_step_p95_m": 0.0,
            "inlier_large_step_ratio": 0.0,
            "inlier_max_from_start_m": 0.0,
            "inlier_count": 0.0,
            "outlier_count": 0.0,
        }

    pts = np.asarray([[float(wp.x), float(wp.y)] for wp in traj], dtype=np.float64)
    eps = max(0.05, float(eps_m))
    if n == 1:
        return {
            "valid": 1.0,
            "inlier_ratio": 1.0,
            "outlier_ratio": 0.0,
            "max_outlier_run": 0.0,
            "inlier_net_disp_m": 0.0,
            "inlier_path_len_m": 0.0,
            "inlier_radius_p90_m": 0.0,
            "inlier_radius_max_m": 0.0,
            "inlier_step_p95_m": 0.0,
            "inlier_large_step_ratio": 0.0,
            "inlier_max_from_start_m": 0.0,
            "inlier_count": 1.0,
            "outlier_count": 0.0,
        }

    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    neighbor_count = np.sum(dist <= eps, axis=1)
    if neighbor_count.size <= 0:
        return {
            "valid": 0.0,
            "inlier_ratio": 0.0,
            "outlier_ratio": 1.0,
            "max_outlier_run": float(n),
            "inlier_net_disp_m": 0.0,
            "inlier_path_len_m": 0.0,
            "inlier_radius_p90_m": 0.0,
            "inlier_radius_max_m": 0.0,
            "inlier_step_p95_m": 0.0,
            "inlier_large_step_ratio": 0.0,
            "inlier_max_from_start_m": 0.0,
            "inlier_count": 0.0,
            "outlier_count": float(n),
        }
    candidate_idx = np.where(neighbor_count == np.max(neighbor_count))[0]
    if candidate_idx.size > 1:
        median_d = np.median(dist[candidate_idx], axis=1)
        seed_i = int(candidate_idx[int(np.argmin(median_d))])
    else:
        seed_i = int(candidate_idx[0])

    center = pts[seed_i]
    inlier = np.sqrt(np.sum((pts - center[None, :]) ** 2, axis=1)) <= eps
    if np.any(inlier):
        center = np.median(pts[inlier], axis=0)
        inlier = np.sqrt(np.sum((pts - center[None, :]) ** 2, axis=1)) <= eps

    inlier_count = int(np.sum(inlier))
    outlier_count = int(n - inlier_count)
    inlier_ratio = float(inlier_count / max(1, n))
    outlier_ratio = float(outlier_count / max(1, n))
    max_outlier_run = _max_contiguous_true(np.logical_not(inlier))

    inlier_pts = pts[inlier]
    if inlier_pts.shape[0] >= 1:
        center_in = np.median(inlier_pts, axis=0)
        in_r = np.sqrt(np.sum((inlier_pts - center_in[None, :]) ** 2, axis=1))
        in_radius_p90 = float(np.quantile(in_r, 0.9))
        in_radius_max = float(np.max(in_r))
        in_max_from_start = float(np.max(np.sqrt(np.sum((inlier_pts - inlier_pts[0:1]) ** 2, axis=1))))
    else:
        in_radius_p90 = 0.0
        in_radius_max = 0.0
        in_max_from_start = 0.0

    if inlier_pts.shape[0] >= 2:
        in_step = np.sqrt(np.sum((inlier_pts[1:] - inlier_pts[:-1]) ** 2, axis=1))
        in_net = float(np.linalg.norm(inlier_pts[-1] - inlier_pts[0]))
        in_path = float(np.sum(in_step))
        in_step_p95 = float(np.quantile(in_step, 0.95))
        in_large_ratio = float(np.mean((in_step > max(0.05, float(large_step_threshold_m))).astype(np.float64)))
    else:
        in_net = 0.0
        in_path = 0.0
        in_step_p95 = 0.0
        in_large_ratio = 0.0

    return {
        "valid": 1.0,
        "inlier_ratio": float(inlier_ratio),
        "outlier_ratio": float(outlier_ratio),
        "max_outlier_run": float(max_outlier_run),
        "inlier_net_disp_m": float(in_net),
        "inlier_path_len_m": float(in_path),
        "inlier_radius_p90_m": float(in_radius_p90),
        "inlier_radius_max_m": float(in_radius_max),
        "inlier_step_p95_m": float(in_step_p95),
        "inlier_large_step_ratio": float(in_large_ratio),
        "inlier_max_from_start_m": float(in_max_from_start),
        "inlier_count": float(inlier_count),
        "outlier_count": float(outlier_count),
    }


def _is_parked_vehicle(
    traj: Sequence[Waypoint],
    times: Sequence[float] | None,
    default_dt: float,
    cfg: Dict[str, float],
) -> Tuple[bool, Dict[str, float]]:
    stats = _compute_motion_stats_xy(traj, times, default_dt)
    n = int(round(float(stats.get("num_points", 0.0))))
    if n <= 1:
        stats["large_step_ratio"] = 0.0
        stats["parked_vehicle"] = 1.0
        return True, stats

    large_step_threshold_m = max(0.05, float(cfg.get("large_step_threshold_m", 0.6)))
    net_disp_max_m = max(0.0, float(cfg.get("net_disp_max_m", 1.0)))
    radius_p90_max_m = max(0.0, float(cfg.get("radius_p90_max_m", 1.1)))
    radius_max_m = max(0.0, float(cfg.get("radius_max_m", 2.0)))
    p95_step_max_m = max(0.0, float(cfg.get("p95_step_max_m", 0.55)))
    max_from_start_m = max(0.0, float(cfg.get("max_from_start_m", 1.8)))
    large_step_ratio_max = max(0.0, float(cfg.get("large_step_ratio_max", 0.08)))
    robust_cluster_enabled = bool(cfg.get("robust_cluster_enabled", True))
    robust_cluster_eps_m = max(0.05, float(cfg.get("robust_cluster_eps_m", 0.8)))
    robust_min_inlier_ratio = max(0.0, min(1.0, float(cfg.get("robust_min_inlier_ratio", 0.8))))
    robust_max_outlier_run = max(0, int(cfg.get("robust_max_outlier_run", 3)))
    robust_min_points = max(3, int(cfg.get("robust_min_points", 6)))

    xs = np.asarray([float(wp.x) for wp in traj], dtype=np.float64)
    ys = np.asarray([float(wp.y) for wp in traj], dtype=np.float64)
    pts = np.stack([xs, ys], axis=1)
    diffs = pts[1:] - pts[:-1]
    step = np.sqrt(np.sum(diffs * diffs, axis=1))
    large_step_ratio = float(np.mean((step > large_step_threshold_m).astype(np.float64))) if step.size else 0.0
    stats["large_step_ratio"] = float(large_step_ratio)

    parked = bool(
        float(stats.get("net_disp_m", 0.0)) <= net_disp_max_m
        and float(stats.get("radius_p90_m", 0.0)) <= radius_p90_max_m
        and float(stats.get("radius_max_m", 0.0)) <= radius_max_m
        and float(stats.get("step_p95_m", 0.0)) <= p95_step_max_m
        and float(stats.get("max_from_start_m", 0.0)) <= max_from_start_m
        and float(large_step_ratio) <= large_step_ratio_max
    )
    stats["parked_by_robust_cluster"] = 0.0

    if (not parked) and robust_cluster_enabled and n >= robust_min_points:
        robust = _compute_robust_stationary_cluster_stats(
            traj=traj,
            eps_m=robust_cluster_eps_m,
            large_step_threshold_m=large_step_threshold_m,
        )
        stats["robust_valid"] = float(robust.get("valid", 0.0))
        stats["robust_inlier_ratio"] = float(robust.get("inlier_ratio", 0.0))
        stats["robust_outlier_ratio"] = float(robust.get("outlier_ratio", 1.0))
        stats["robust_max_outlier_run"] = float(robust.get("max_outlier_run", float(n)))
        stats["robust_inlier_net_disp_m"] = float(robust.get("inlier_net_disp_m", 0.0))
        stats["robust_inlier_path_len_m"] = float(robust.get("inlier_path_len_m", 0.0))
        stats["robust_inlier_radius_p90_m"] = float(robust.get("inlier_radius_p90_m", 0.0))
        stats["robust_inlier_radius_max_m"] = float(robust.get("inlier_radius_max_m", 0.0))
        stats["robust_inlier_step_p95_m"] = float(robust.get("inlier_step_p95_m", 0.0))
        stats["robust_inlier_large_step_ratio"] = float(robust.get("inlier_large_step_ratio", 0.0))
        stats["robust_inlier_max_from_start_m"] = float(robust.get("inlier_max_from_start_m", 0.0))
        stats["robust_inlier_count"] = float(robust.get("inlier_count", 0.0))
        stats["robust_outlier_count"] = float(robust.get("outlier_count", 0.0))

        robust_parked = bool(
            float(robust.get("valid", 0.0)) >= 0.5
            and float(robust.get("inlier_ratio", 0.0)) >= robust_min_inlier_ratio
            and int(round(float(robust.get("max_outlier_run", float(n))))) <= robust_max_outlier_run
            and float(robust.get("inlier_net_disp_m", 0.0)) <= net_disp_max_m
            and float(robust.get("inlier_radius_p90_m", 0.0)) <= radius_p90_max_m
            and float(robust.get("inlier_radius_max_m", 0.0)) <= radius_max_m
            and float(robust.get("inlier_step_p95_m", 0.0)) <= p95_step_max_m
            and float(robust.get("inlier_max_from_start_m", 0.0)) <= max_from_start_m
            and float(robust.get("inlier_large_step_ratio", 0.0)) <= large_step_ratio_max
        )
        if robust_parked:
            parked = True
            stats["parked_by_robust_cluster"] = 1.0
    else:
        stats["robust_valid"] = 0.0

    stats["parked_vehicle"] = 1.0 if parked else 0.0
    return parked, stats


# =============================================================================
# GRP-aware trajectory alignment for CARLA log replay
# =============================================================================
# This module connects to a running CARLA server, uses the GlobalRoutePlanner to
# validate and correct trajectories so they follow legal drivable routes.  It
# reuses the DP-based snapping approach from
# scenario_generator/pipeline/step_07_route_alignment but modifies the
# optimisation objective: timing-critical waypoints are NEVER removed, waypoint
# minimisation is secondary, and lane-change suppression is prioritised.

_grp_carla_module = None
_grp_class = None
_grpdao_class = None


def _setup_carla_for_grp() -> bool:
    """Set up CARLA Python API paths, reusing the same logic as step_07."""
    project_root = Path(__file__).resolve().parents[2]
    carla_candidates = [
        project_root / "carla" / "PythonAPI" / "carla",
        project_root / "carla912" / "PythonAPI" / "carla",
        project_root / "external_paths" / "carla_root" / "PythonAPI" / "carla",
        Path("/opt/carla/PythonAPI/carla"),
        Path.home() / "carla" / "PythonAPI" / "carla",
    ]
    paths_to_remove = []
    for p in list(sys.path):
        if not p:
            continue
        try:
            path_obj = Path(p).resolve()
        except Exception:
            continue
        if "site-packages" in path_obj.as_posix().split("/"):
            continue
        if not (path_obj == project_root or project_root in path_obj.parents):
            continue
        potential_shadow = path_obj / "carla"
        if potential_shadow.exists() and potential_shadow.is_dir() and not (potential_shadow / "dist").exists():
            paths_to_remove.append(p)
    for p in paths_to_remove:
        if p in sys.path:
            sys.path.remove(p)
    for carla_path in carla_candidates:
        if carla_path.exists():
            dist_dir = carla_path / "dist"
            if dist_dir.exists():
                egg_files = sorted(dist_dir.glob("carla-*-py3*.egg"), reverse=True)
                if not egg_files:
                    egg_files = list(dist_dir.glob("carla-*.egg"))
                for egg in egg_files:
                    if str(egg) not in sys.path:
                        sys.path.append(str(egg))
                        break
            if str(carla_path) not in sys.path:
                sys.path.append(str(carla_path))
            return True
    return False


def _ensure_carla_grp():
    """Import CARLA and GRP classes, caching the result."""
    global _grp_carla_module, _grp_class, _grpdao_class
    if _grp_carla_module is not None:
        return _grp_carla_module, _grp_class, _grpdao_class
    _setup_carla_for_grp()
    sys.modules.pop("carla", None)
    sys.modules.pop("agents", None)
    import carla as _carla_mod  # type: ignore
    if not all(hasattr(_carla_mod, a) for a in ("Client", "Location", "Transform")):
        raise ImportError("Imported carla module does not look like CARLA PythonAPI.")
    from agents.navigation.global_route_planner import GlobalRoutePlanner  # type: ignore
    try:
        from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO  # type: ignore
    except ImportError:
        GlobalRoutePlannerDAO = None
    _grp_carla_module = _carla_mod
    _grp_class = GlobalRoutePlanner
    _grpdao_class = GlobalRoutePlannerDAO
    return _grp_carla_module, _grp_class, _grpdao_class


def _grp_yaw_diff_deg(a: float, b: float) -> float:
    """Smallest signed-angle difference in degrees."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _grp_route_length(route: list) -> float:
    """Total length of a GRP route."""
    if not route or len(route) < 2:
        return 0.0
    total = 0.0
    prev = route[0][0].transform.location
    for wp, _ in route[1:]:
        loc = wp.transform.location
        total += prev.distance(loc)
        prev = loc
    return float(total)


def _grp_route_has_turn(route: list) -> bool:
    """True if route contains non-LANEFOLLOW options (i.e. a turn/lane-change)."""
    for _, opt in route:
        name = getattr(opt, "name", str(opt)).upper()
        if name not in ("LANEFOLLOW", "VOID"):
            return True
    return False
