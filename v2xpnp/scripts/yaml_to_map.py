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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------- CARLA Process Management ---------------------- #

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

from v2xpnp.scripts.yaml_to_carla_log import (  # noqa: E402
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
        non_stationary_vids: List[int] = []
        
        for vid in walker_vids:
            traj = vehicles[vid]
            times = vehicle_times.get(vid, [])
            is_stat, reason, stats = self.is_walker_stationary_jitter(vid, traj, times)
            
            if is_stat:
                stationary_jitter_removed[vid] = {
                    "reason": reason,
                    "stats": stats,
                }
                print(f"[WALKER] REMOVED walker_{vid}: stationary/jitter ({reason}), "
                      f"path={stats.get('path_len_m', 0):.1f}m, net_disp={stats.get('net_disp_m', 0):.1f}m, "
                      f"frames={int(stats.get('num_frames', 0))}")
            else:
                non_stationary_vids.append(vid)
        
        # Update vehicle dict to remove stationary walkers
        updated_vehicles = dict(vehicles)
        for vid in stationary_jitter_removed:
            if vid in updated_vehicles:
                del updated_vehicles[vid]
            if vid in vehicle_times:
                del vehicle_times[vid]
        
        if stationary_jitter_removed:
            print(f"[WALKER] Removed {len(stationary_jitter_removed)} stationary/jitter walkers")
        
        report: Dict[str, object] = {
            "enabled": True,
            "walker_count": len(walker_vids),
            "walker_count_after_stationary_removal": len(non_stationary_vids),
            "stationary_removed_count": len(stationary_jitter_removed),
            "stationary_removed": {
                vid: info for vid, info in stationary_jitter_removed.items()
            },
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
    return any(token in ot for token in ("bicycle", "cyclist", "bike"))


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


def _load_carla_map_cache_lines(path: Path) -> Tuple[List[List[Tuple[float, float]]], Optional[Tuple[float, float, float, float]], str]:
    with path.open("rb") as f:
        raw = pickle.load(f)

    map_name = ""
    bounds: Optional[Tuple[float, float, float, float]] = None
    lines: List[List[Tuple[float, float]]] = []
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

    _extract_xy_lines_recursive(raw, lines)
    deduped: List[List[Tuple[float, float]]] = []
    for ln in lines:
        if len(ln) >= 2:
            deduped.append([(float(x), float(y)) for x, y in ln])
    return deduped, bounds, map_name


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

    try:
        jpeg_bytes, actual_bounds = _capture_carla_topdown_image(
            carla_host=carla_host,
            carla_port=carla_port,
            raw_bounds=raw_bounds,
            carla_map_name=carla_map_name,
            tile_fov_deg=tile_fov_deg,
            tile_px=tile_px,
        )
    except Exception as exc:
        print(f"[WARN] Failed to capture CARLA top-down image: {exc}")
        return None

    # --- save cache ---
    try:
        image_cache_path.parent.mkdir(parents=True, exist_ok=True)
        image_cache_path.write_bytes(jpeg_bytes)
        meta_cache_path.write_text(
            json.dumps(
                {
                    "bounds": list(actual_bounds),
                    "tile_fov_deg": tile_fov_deg,
                    "tile_px": tile_px,
                },
                indent=2,
            ),
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


def _grp_refine_trajectory_timing_preserving(
    carla_map,
    grp,
    waypoints_carla: List[Dict],
    timestamps: List[float],
    radius: float = 2.5,
    k: int = 6,
    heading_thresh: float = 40.0,
    lane_change_penalty: float = 50.0,
    turn_penalty: float = 8.0,
    deviation_weight: float = 1.0,
    route_weight: float = 0.8,
    waypoint_step: float = 1.0,
) -> List[Dict]:
    """GRP-aware trajectory alignment that preserves ALL timing waypoints.

    Unlike step_07's refine_waypoints_dp which may compress/skip waypoints,
    this variant keeps every input waypoint to maintain temporal fidelity.
    It snaps each waypoint to the nearest valid CARLA lane while penalising
    lane changes and GRP-illegal transitions.

    Parameters
    ----------
    carla_map : carla.Map
    grp : GlobalRoutePlanner
    waypoints_carla : list of dict with 'x', 'y', 'z', 'yaw'
    timestamps : parallel list of float timestamps
    radius : search radius for CARLA waypoint candidates
    k : max candidates per input waypoint
    heading_thresh : yaw tolerance for candidate filtering (degrees)
    lane_change_penalty : DP cost for switching lanes
    turn_penalty : DP cost for GRP-indicated turns
    deviation_weight : weight for spatial deviation from original position
    route_weight : weight for GRP route-length cost
    waypoint_step : CARLA map generate_waypoints resolution

    Returns
    -------
    list of dict : aligned waypoints with 'x', 'y', 'z', 'yaw', 't' fields
    """
    global _grp_carla_module  # Need carla module for Location creation
    
    n = len(waypoints_carla)
    if n == 0:
        return []
    if n == 1:
        wp = waypoints_carla[0]
        return [{"x": wp["x"], "y": wp["y"], "z": wp["z"], "yaw": wp["yaw"],
                 "t": timestamps[0] if timestamps else 0.0, "grp_aligned": True}]

    # Generate all map waypoints as snap candidates
    all_wps = carla_map.generate_waypoints(waypoint_step)
    if not all_wps:
        # Fallback: return unmodified
        return [
            {"x": wp["x"], "y": wp["y"], "z": wp["z"], "yaw": wp["yaw"],
             "t": timestamps[i] if i < len(timestamps) else 0.0, "grp_aligned": False}
            for i, wp in enumerate(waypoints_carla)
        ]

    coords = np.array([[w.transform.location.x, w.transform.location.y] for w in all_wps])

    # Recompute headings from the input trajectory for direction matching.
    # Use a wider forward-looking window (up to HEADING_WINDOW frames) so that
    # duplicate-position frames or micro-jitter don't yield garbage headings.
    # For the FIRST FEW frames, use an even wider window to capture overall
    # trajectory direction — critical for correct snapping in intersections.
    HEADING_WINDOW = 5
    FIRST_FRAME_LOOKAHEAD = 20  # Wider lookahead for first few frames
    FIRST_FRAME_COUNT = 5  # Apply wider lookahead to these many initial frames
    
    # First compute a "global" trajectory heading from overall displacement
    # This helps detect the intended direction for vehicles starting in intersections
    global_heading: Optional[float] = None
    if n >= 3:
        # Find a point with meaningful displacement from start
        for end_idx in range(min(FIRST_FRAME_LOOKAHEAD, n - 1), 0, -1):
            gdx = waypoints_carla[end_idx]["x"] - waypoints_carla[0]["x"]
            gdy = waypoints_carla[end_idx]["y"] - waypoints_carla[0]["y"]
            if abs(gdx) > 1.0 or abs(gdy) > 1.0:  # Need substantial displacement
                global_heading = math.degrees(math.atan2(gdy, gdx))
                break
    
    headings = [0.0] * n
    for i in range(n):
        dx, dy = 0.0, 0.0
        # For first few frames, use wider lookahead to get overall direction
        effective_window = FIRST_FRAME_LOOKAHEAD if i < FIRST_FRAME_COUNT else HEADING_WINDOW
        # Try forward window first
        for j in range(1, effective_window + 1):
            if i + j < n:
                dx = waypoints_carla[i + j]["x"] - waypoints_carla[i]["x"]
                dy = waypoints_carla[i + j]["y"] - waypoints_carla[i]["y"]
                if abs(dx) > 0.3 or abs(dy) > 0.3:  # need meaningful displacement
                    break
        if abs(dx) <= 0.3 and abs(dy) <= 0.3:
            # Forward didn't help — try backward window
            for j in range(1, effective_window + 1):
                if i - j >= 0:
                    dx = waypoints_carla[i]["x"] - waypoints_carla[i - j]["x"]
                    dy = waypoints_carla[i]["y"] - waypoints_carla[i - j]["y"]
                    if abs(dx) > 0.3 or abs(dy) > 0.3:
                        break
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            headings[i] = math.degrees(math.atan2(dy, dx))
        elif global_heading is not None and i < FIRST_FRAME_COUNT:
            # For first few frames with no displacement, use global heading
            headings[i] = global_heading
        elif i > 0:
            headings[i] = headings[i - 1]

    rad2 = radius * radius

    def _get_candidates(idx: int) -> List[int]:
        wp = waypoints_carla[idx]
        xy = np.array([wp["x"], wp["y"]])
        diff = coords - xy[None, :]
        dist2 = np.einsum("ij,ij->i", diff, diff)
        idxs = np.where(dist2 <= rad2)[0]
        if idxs.size == 0:
            # Expand search to nearest k
            nearest_k = min(k * 3, len(dist2))
            idxs = np.argsort(dist2)[:nearest_k]
        # Filter by heading — graduated fallback, never accept opposite direction
        traj_yaw = headings[idx]
        filtered = []
        for ci in idxs:
            wp_yaw = all_wps[ci].transform.rotation.yaw
            if _grp_yaw_diff_deg(wp_yaw, traj_yaw) <= heading_thresh:
                filtered.append((ci, dist2[ci]))
        if not filtered:
            # Relaxed filter: accept up to 90° but never opposite direction
            for ci in idxs:
                wp_yaw = all_wps[ci].transform.rotation.yaw
                if _grp_yaw_diff_deg(wp_yaw, traj_yaw) <= 90.0:
                    filtered.append((ci, dist2[ci]))
        if not filtered:
            # Last resort: accept up to 120° — still reject clearly opposite (>120°)
            for ci in idxs:
                wp_yaw = all_wps[ci].transform.rotation.yaw
                if _grp_yaw_diff_deg(wp_yaw, traj_yaw) <= 120.0:
                    filtered.append((ci, dist2[ci]))
        if not filtered:
            # Expanded-radius search: widen to 3× radius, still enforce heading
            expanded_rad2 = rad2 * 9.0  # 3× radius
            exp_idxs = np.where(dist2 <= expanded_rad2)[0]
            if exp_idxs.size == 0:
                exp_idxs = np.argsort(dist2)[:min(k * 6, len(dist2))]
            for ci in exp_idxs:
                wp_yaw = all_wps[ci].transform.rotation.yaw
                if _grp_yaw_diff_deg(wp_yaw, traj_yaw) <= 120.0:
                    filtered.append((ci, dist2[ci]))
        if not filtered:
            # Still empty — try 150° (reject only truly opposite > 150°)
            all_sorted = np.argsort(dist2)
            for ci in all_sorted[:min(k * 10, len(dist2))]:
                wp_yaw = all_wps[ci].transform.rotation.yaw
                if _grp_yaw_diff_deg(wp_yaw, traj_yaw) <= 150.0:
                    filtered.append((ci, dist2[ci]))
                    if len(filtered) >= k:
                        break
        if not filtered:
            # Absolute last resort: nearest single candidate (will be penalised
            # heavily by heading cost in the DP, but keeps DP from having empty
            # candidate sets which would break the algorithm).
            filtered = [(int(np.argmin(dist2)), float(np.min(dist2)))]
        filtered.sort(key=lambda x: x[1])
        return [ci for ci, _ in filtered[:k]]

    # Build candidates per waypoint
    cands_per_wp = [_get_candidates(i) for i in range(n)]

    # DP: for each waypoint i and each candidate ci, compute min cost
    INF = float("inf")
    best_cost = [dict() for _ in range(n)]
    backref = [dict() for _ in range(n)]

    # Initialize first waypoint — include heading penalty so wrong-direction
    # lanes don't seed the DP path at near-zero cost.
    # ENHANCED: Use much stronger heading penalty and forward-looking route check
    # to prevent wrong-direction snaps in intersections.
    
    # Find a forward reference point for route compatibility check
    forward_ref_idx = min(10, n - 1)  # Look ~10 frames ahead
    forward_ref_wp = waypoints_carla[forward_ref_idx]
    forward_ref_loc = None
    if forward_ref_idx > 0:
        try:
            forward_ref_loc = _grp_carla_module.Location(
                x=forward_ref_wp["x"],
                y=forward_ref_wp["y"],
                z=forward_ref_wp.get("z", 0.0),
            )
        except Exception:
            forward_ref_loc = None
    
    for ci in cands_per_wp[0]:
        dev = float(np.linalg.norm(np.array([waypoints_carla[0]["x"], waypoints_carla[0]["y"]]) - coords[ci]))
        heading_diff_0 = _grp_yaw_diff_deg(
            all_wps[ci].transform.rotation.yaw, headings[0]
        )
        
        # ENHANCED heading cost for first frame - be very strict about direction
        heading_cost_0 = 0.0
        if heading_diff_0 > 120.0:
            # Opposite direction - nearly impossible to recover
            heading_cost_0 = lane_change_penalty * 10.0
        elif heading_diff_0 > 90.0:
            # Wrong quadrant - strongly disfavour
            heading_cost_0 = lane_change_penalty * 6.0
        elif heading_diff_0 > 60.0:
            # Significant mismatch - moderate penalty
            heading_cost_0 = lane_change_penalty * 2.0
        elif heading_diff_0 > heading_thresh:
            # Minor mismatch - light penalty
            heading_cost_0 = deviation_weight * heading_diff_0 * 0.3
        
        # Forward-looking route compatibility check for first frame
        # Penalize candidates that can't route to where the trajectory goes
        route_compat_cost = 0.0
        if forward_ref_loc is not None and forward_ref_idx > 0:
            try:
                first_wp_loc = all_wps[ci].transform.location
                route_to_future = grp.trace_route(first_wp_loc, forward_ref_loc)
                if not route_to_future:
                    # No route found - heavily penalize
                    route_compat_cost = lane_change_penalty * 3.0
                else:
                    # Check if route involves excessive backtracking
                    route_len = _grp_route_length(route_to_future)
                    direct_dist = first_wp_loc.distance(forward_ref_loc)
                    if direct_dist > 1.0 and route_len > direct_dist * 3.0:
                        # Route is much longer than direct path - might be wrong direction
                        route_compat_cost = lane_change_penalty * 1.5
            except Exception:
                # GRP route check failed - small penalty
                route_compat_cost = lane_change_penalty * 0.5
        
        best_cost[0][ci] = deviation_weight * dev + heading_cost_0 + route_compat_cost
        backref[0][ci] = None

    # Forward pass — every waypoint MUST be kept (no skipping)
    for i in range(1, n):
        obs = np.array([waypoints_carla[i]["x"], waypoints_carla[i]["y"]])
        for ci in cands_per_wp[i]:
            dev = float(np.linalg.norm(obs - coords[ci]))
            base_cost = deviation_weight * dev
            best_val = INF
            best_prev_ci = None

            curr_wp = all_wps[ci]
            for prev_ci, prev_cost in best_cost[i - 1].items():
                prev_wp = all_wps[prev_ci]

                # Same lane bonus / lane change penalty
                same_lane = (
                    prev_wp.road_id == curr_wp.road_id
                    and prev_wp.lane_id == curr_wp.lane_id
                )

                # GRP route check — only compute for non-trivially-close waypoints
                dist_between = prev_wp.transform.location.distance(curr_wp.transform.location)
                route_cost = 0.0
                has_turn = False
                if dist_between > 0.5:
                    try:
                        route = grp.trace_route(prev_wp.transform.location, curr_wp.transform.location)
                        if route:
                            route_len = _grp_route_length(route)
                            has_turn = _grp_route_has_turn(route)
                            route_cost = route_weight * route_len
                        else:
                            route_cost = route_weight * dist_between * 3.0  # penalty for no-route
                    except Exception:
                        route_cost = route_weight * dist_between * 2.0
                else:
                    route_cost = route_weight * dist_between

                # Heading mismatch penalty: penalise candidates whose
                # lane direction disagrees with actual travel direction
                heading_diff = _grp_yaw_diff_deg(
                    curr_wp.transform.rotation.yaw, headings[i]
                )
                heading_cost = 0.0
                if heading_diff > 90.0:
                    heading_cost = lane_change_penalty * 4.0  # strongly disfavour opposite direction
                elif heading_diff > heading_thresh:
                    heading_cost = deviation_weight * heading_diff * 0.15

                total = (
                    prev_cost
                    + base_cost
                    + route_cost
                    + heading_cost
                    + (0.0 if same_lane else lane_change_penalty)
                    + (turn_penalty if has_turn else 0.0)
                )

                if total < best_val:
                    best_val = total
                    best_prev_ci = prev_ci

            if best_prev_ci is not None:
                best_cost[i][ci] = best_val
                backref[i][ci] = best_prev_ci

    # Backtrack
    if not best_cost[-1]:
        # Fallback: return unmodified
        return [
            {"x": wp["x"], "y": wp["y"], "z": wp["z"], "yaw": wp["yaw"],
             "t": timestamps[i] if i < len(timestamps) else 0.0, "grp_aligned": False}
            for i, wp in enumerate(waypoints_carla)
        ]

    end_ci = min(best_cost[-1], key=lambda ci: best_cost[-1][ci])
    chosen_cis = [0] * n
    ci = end_ci
    for i in range(n - 1, -1, -1):
        chosen_cis[i] = ci
        prev = backref[i].get(ci)
        if prev is not None:
            ci = prev

    # Build output — preserve ALL timestamps
    result = []
    for i in range(n):
        cw = all_wps[chosen_cis[i]]
        loc = cw.transform.location
        yaw = cw.transform.rotation.yaw
        result.append({
            "x": float(loc.x),
            "y": float(loc.y),
            "z": float(loc.z),
            "yaw": float(yaw),
            "t": float(timestamps[i]) if i < len(timestamps) else 0.0,
            "road_id": int(cw.road_id),
            "lane_id": int(cw.lane_id),
            "grp_aligned": True,
        })

    # Post-process: suppress micro lane-switches (A→B→A patterns)
    if len(result) >= 3:
        lane_ids = [(r["road_id"], r["lane_id"]) for r in result]
        # Build runs
        runs: List[Tuple[Tuple[int, int], int, int]] = []
        cur_lane = lane_ids[0]
        run_start = 0
        for ri in range(1, len(lane_ids)):
            if lane_ids[ri] != cur_lane:
                runs.append((cur_lane, run_start, ri - 1))
                cur_lane = lane_ids[ri]
                run_start = ri
        runs.append((cur_lane, run_start, len(lane_ids) - 1))

        # Suppress short B in A→B→A
        MAX_SHORT_RUN = 8
        changed = True
        while changed:
            changed = False
            for ri in range(1, len(runs) - 1):
                a_lane = runs[ri - 1][0]
                b_lane = runs[ri][0]
                c_lane = runs[ri + 1][0]
                b_len = runs[ri][2] - runs[ri][1] + 1
                if a_lane == c_lane and b_lane != a_lane and b_len <= MAX_SHORT_RUN:
                    # Re-snap B frames to A lane
                    for fi in range(runs[ri][1], runs[ri][2] + 1):
                        nearest_a = carla_map.get_waypoint(
                            _grp_carla_module.Location(
                                x=waypoints_carla[fi]["x"],
                                y=waypoints_carla[fi]["y"],
                                z=waypoints_carla[fi].get("z", 0.0),
                            ),
                            project_to_road=True,
                            lane_type=_grp_carla_module.LaneType.Driving,
                        )
                        if nearest_a is not None:
                            loc = nearest_a.transform.location
                            result[fi]["x"] = float(loc.x)
                            result[fi]["y"] = float(loc.y)
                            result[fi]["z"] = float(loc.z)
                            result[fi]["yaw"] = float(nearest_a.transform.rotation.yaw)
                            result[fi]["road_id"] = int(nearest_a.road_id)
                            result[fi]["lane_id"] = int(nearest_a.lane_id)
                    # Merge runs
                    runs[ri - 1] = (a_lane, runs[ri - 1][1], runs[ri + 1][2])
                    runs.pop(ri + 1)
                    runs.pop(ri)
                    changed = True
                    break

    return result


def _is_actor_confidently_stationary(
    traj: Sequence[Waypoint],
    times: Sequence[float] | None,
    default_dt: float,
    parked_cfg: Dict[str, float],
) -> Tuple[bool, str, Dict[str, float]]:
    """Enhanced stationary classification that refines (not replaces) _is_parked_vehicle.

    Adds:
    - Velocity-based sustained motion check
    - Positional variance check
    - Sustained displacement window check
    - Jitter detection (high path length but low net displacement)
    - Per-frame path efficiency check

    Returns (is_stationary, reason, stats).
    """
    is_parked, stats = _is_parked_vehicle(traj, times, default_dt, parked_cfg)
    if is_parked:
        return True, "parked_basic", stats

    n = len(traj)
    if n < 3:
        return False, "too_few_points", stats

    # Additional checks for borderline cases
    xs = np.array([float(wp.x) for wp in traj], dtype=np.float64)
    ys = np.array([float(wp.y) for wp in traj], dtype=np.float64)

    # Compute path metrics
    diffs = np.sqrt((xs[1:] - xs[:-1]) ** 2 + (ys[1:] - ys[:-1]) ** 2)
    path_len = float(np.sum(diffs)) if diffs.size else 0.0
    net_disp = float(math.hypot(xs[-1] - xs[0], ys[-1] - ys[0])) if n >= 2 else 0.0
    stats["jitter_path_len_m"] = float(path_len)
    stats["jitter_net_disp_m"] = float(net_disp)

    # 1. Positional variance check: if the variance is extremely low, it's stationary
    var_x = float(np.var(xs))
    var_y = float(np.var(ys))
    total_var = var_x + var_y
    stats["pos_variance_m2"] = float(total_var)
    if total_var < 0.15:
        stats["parked_vehicle"] = 1.0
        return True, "low_variance", stats

    # 2. Velocity-based check: compute per-step velocities
    dt_val = float(default_dt) if default_dt > 0 else 0.1
    velocities = diffs / dt_val
    stats["velocity_median_mps"] = float(np.median(velocities)) if velocities.size else 0.0
    stats["velocity_p95_mps"] = float(np.quantile(velocities, 0.95)) if velocities.size else 0.0

    # If median velocity is walking speed and p95 is still very low → stationary with jitter
    if float(np.median(velocities)) < 0.3 and float(np.quantile(velocities, 0.95)) < 0.8:
        stats["parked_vehicle"] = 1.0
        return True, "low_velocity", stats

    # 3. Sustained displacement: check if max displacement over any 2-second window is tiny
    window_frames = max(3, int(2.0 / dt_val))
    max_window_disp = 0.0
    for start in range(0, n - window_frames + 1):
        end = start + window_frames
        window_xs = xs[start:end]
        window_ys = ys[start:end]
        disp = math.hypot(float(window_xs[-1] - window_xs[0]), float(window_ys[-1] - window_ys[0]))
        max_window_disp = max(max_window_disp, disp)
    stats["max_2s_window_disp_m"] = float(max_window_disp)
    if max_window_disp < 0.5:
        stats["parked_vehicle"] = 1.0
        return True, "low_sustained_disp", stats

    # 4. Jitter detection: high path length with very low net displacement
    # This catches actors that vibrate in place (sensor noise, tracking jitter)
    jitter_net_disp_max_m = float(parked_cfg.get("jitter_net_disp_max_m", 5.0))
    jitter_path_ratio_min = float(parked_cfg.get("jitter_path_ratio_min", 3.0))
    jitter_variance_max_m2 = float(parked_cfg.get("jitter_variance_max_m2", 2.0))

    if net_disp < jitter_net_disp_max_m:
        # Low net displacement - check if it's jitter
        path_efficiency = net_disp / max(0.01, path_len)  # 0 = pure jitter, 1 = straight line
        stats["jitter_path_efficiency"] = float(path_efficiency)

        # Jitter: lots of movement but going nowhere (low efficiency) + low variance
        if path_len > 0.5 and path_efficiency < (1.0 / jitter_path_ratio_min) and total_var < jitter_variance_max_m2:
            stats["parked_vehicle"] = 1.0
            return True, "jitter_detected", stats

    # 5. Per-frame path check: if path per frame is extremely low for many frames, it's stationary
    path_per_frame = path_len / max(1, n - 1)
    stats["path_per_frame_m"] = float(path_per_frame)
    # Vehicles typically move at least 0.05m per frame (0.5 m/s at 10 Hz)
    # For many-frame trajectories with tiny per-frame movement, mark as jitter
    if n >= 50 and path_per_frame < 0.03 and net_disp < 3.0:
        stats["parked_vehicle"] = 1.0
        return True, "low_path_per_frame", stats

    return False, "moving", stats


# ---------------------------------------------------------------------------
# Outer-lane bias for parked vehicles
# ---------------------------------------------------------------------------


def _find_outermost_driving_lane_wp(carla_wp, carla_mod):
    """Walk from *carla_wp* toward the road edge and return the outermost
    **Driving** lane waypoint on the same road.

    CARLA convention: negative lane_id → right-hand side of the road (where
    parked cars typically sit).  We walk via ``get_right_lane()`` for
    negative lane_ids and ``get_left_lane()`` for positive ones.

    Returns *carla_wp* itself if it is already the outermost, or if the
    outermost lane is not a Driving lane.
    """
    if carla_wp is None:
        return carla_wp

    road_id = carla_wp.road_id
    best = carla_wp

    # Determine walk direction based on lane_id sign
    lid = carla_wp.lane_id
    if lid < 0:
        step_fn = lambda w: w.get_right_lane()
    elif lid > 0:
        step_fn = lambda w: w.get_left_lane()
    else:
        return carla_wp  # lane_id == 0 is center, nothing to do

    MAX_STEPS = 10  # safety limit
    cur = carla_wp
    for _ in range(MAX_STEPS):
        nxt = step_fn(cur)
        if nxt is None:
            break
        # Must stay on the same road
        if nxt.road_id != road_id:
            break
        # Must be a Driving lane
        if nxt.lane_type != carla_mod.LaneType.Driving:
            break
        best = nxt
        cur = nxt

    return best


def _build_moving_vehicle_spatial_set(
    vehicles: Dict[int, List],
    stationary_vids: set,
    to_carla_fn,
    cell_size: float = 2.0,
) -> set:
    """Build a set of (grid_col, grid_row) cells occupied by *moving*
    vehicles' trajectories (in CARLA coords).  Used to quickly check
    whether a candidate parked-vehicle position would interfere.

    *to_carla_fn* converts (v2x_x, v2x_y) → (carla_x, carla_y).
    """
    cells: set = set()
    inv = 1.0 / cell_size
    for vid, traj in vehicles.items():
        if vid in stationary_vids:
            continue
        for wp in traj:
            cx, cy = to_carla_fn(float(wp.x), float(wp.y))
            cells.add((int(math.floor(cx * inv)), int(math.floor(cy * inv))))
    return cells


def _point_in_moving_cells(
    cx: float, cy: float, cells: set, cell_size: float = 2.0, radius: float = 3.0,
) -> bool:
    """Return True if the CARLA point (cx, cy) is within *radius* of any
    cell in *cells* (approximated by checking the surrounding grid cells)."""
    inv = 1.0 / cell_size
    r_cells = int(math.ceil(radius / cell_size))
    gx = int(math.floor(cx * inv))
    gy = int(math.floor(cy * inv))
    for dx in range(-r_cells, r_cells + 1):
        for dy in range(-r_cells, r_cells + 1):
            if (gx + dx, gy + dy) in cells:
                return True
    return False


def _try_bias_parked_to_outer_lane(
    snapped_wp,           # CARLA waypoint from get_waypoint()
    carla_map,
    carla_mod,            # the carla module itself
    moving_cells: set,    # grid cells of moving vehicles
    cell_size: float = 2.0,
    max_lateral_m: float = 4.0,
    interference_radius: float = 3.0,
):
    """Attempt to move a parked vehicle to the outermost driving lane.

    Returns ``(outer_wp, did_move, reason)`` where *outer_wp* is the
    chosen CARLA waypoint (either the outer lane or the original if we
    decided not to move), *did_move* is a bool, and *reason* is a short
    string for the report.

    Rules:
      1. Find the outermost driving lane on the same road.
      2. If it's the same lane → no-op.
      3. If the lateral distance > *max_lateral_m* → too far, skip.
      4. If moving there would place the vehicle inside a moving-vehicle
         path cell that wasn't already interfered → revert.
    """
    if snapped_wp is None:
        return snapped_wp, False, "no_snap"

    outer_wp = _find_outermost_driving_lane_wp(snapped_wp, carla_mod)

    # Same lane already?
    if (outer_wp.road_id == snapped_wp.road_id
            and outer_wp.lane_id == snapped_wp.lane_id):
        return snapped_wp, False, "already_outermost"

    # Check lateral distance
    oloc = outer_wp.transform.location
    sloc = snapped_wp.transform.location
    lateral_dist = math.hypot(
        float(oloc.x) - float(sloc.x),
        float(oloc.y) - float(sloc.y),
    )
    if lateral_dist > max_lateral_m:
        return snapped_wp, False, f"too_far_{lateral_dist:.1f}m"

    # Interference check — does the new position create a new collision
    # with a moving vehicle path that the original position did not have?
    orig_interferes = _point_in_moving_cells(
        float(sloc.x), float(sloc.y), moving_cells,
        cell_size=cell_size, radius=interference_radius,
    )
    new_interferes = _point_in_moving_cells(
        float(oloc.x), float(oloc.y), moving_cells,
        cell_size=cell_size, radius=interference_radius,
    )

    if new_interferes and not orig_interferes:
        # Moving would create new interference — revert
        return snapped_wp, False, "would_add_interference"

    return outer_wp, True, f"biased_outer_lane_{lateral_dist:.1f}m"


def _check_parked_obstructs_moving_path(
    parked_x: float,
    parked_y: float,
    moving_cells: set,
    cell_size: float = 2.0,
    obstruction_radius: float = 4.0,
) -> bool:
    """Check if a parked vehicle at (parked_x, parked_y) in CARLA coords
    obstructs any moving vehicle path.
    
    Returns True if the parked vehicle is within obstruction_radius of any
    moving vehicle path cell.
    """
    return _point_in_moving_cells(
        parked_x, parked_y, moving_cells,
        cell_size=cell_size, radius=obstruction_radius,
    )


def _find_obstructing_parked_vehicles(
    parked_positions: Dict[int, Tuple[float, float]],  # vid -> (carla_x, carla_y)
    moving_cells: set,
    ego_cells: set,
    cell_size: float = 2.0,
    obstruction_radius: float = 4.0,
) -> Tuple[List[int], List[int]]:
    """Identify parked vehicles that obstruct moving actors or ego.
    
    Returns (obstructs_ego_vids, obstructs_actor_vids).
    Vehicles obstructing ego are higher priority for removal.
    """
    obstructs_ego: List[int] = []
    obstructs_actor: List[int] = []
    
    for vid, (cx, cy) in parked_positions.items():
        # Check if it obstructs ego path
        if _point_in_moving_cells(cx, cy, ego_cells, cell_size, obstruction_radius):
            obstructs_ego.append(vid)
        # Check if it obstructs any moving actor path
        elif _point_in_moving_cells(cx, cy, moving_cells, cell_size, obstruction_radius):
            obstructs_actor.append(vid)
    
    return obstructs_ego, obstructs_actor


def _grp_align_trajectories(
    vehicles: Dict[int, List[Waypoint]],
    vehicle_times: Dict[int, List[float]],
    ego_trajs: List[List[Waypoint]],
    ego_times: List[List[float]],
    obj_info: Dict[int, Dict[str, object]],
    parked_vehicle_cfg: Dict[str, float],
    align_cfg: Dict[str, object],
    carla_host: str = "localhost",
    carla_port: int = 2005,
    carla_map_name: str = "ucla_v2",
    sampling_resolution: float = 2.0,
    snap_radius: float = 2.5,
    snap_k: int = 6,
    heading_thresh: float = 40.0,
    lane_change_penalty: float = 50.0,
    default_dt: float = 0.1,
    enabled: bool = True,
) -> Tuple[
    Dict[int, List[Waypoint]],
    Dict[int, List[float]],
    List[List[Waypoint]],
    List[List[float]],
    Dict[str, object],
]:
    """Apply GRP-aware trajectory alignment for all non-stationary actors and egos.

    Connects to CARLA, transforms V2XPNP coordinates into CARLA frame,
    runs DP-based GRP alignment, then transforms back.

    Returns updated (vehicles, vehicle_times, ego_trajs, ego_times, report).
    """
    import inspect as _inspect

    report: Dict[str, object] = {
        "enabled": bool(enabled),
        "actors_processed": 0,
        "actors_skipped_stationary": 0,
        "actors_skipped_short": 0,
        "actors_aligned": 0,
        "actors_failed": 0,
        "egos_processed": 0,
        "egos_aligned": 0,
        "egos_failed": 0,
        "actor_details": {},
    }

    if not enabled:
        report["reason"] = "disabled_by_flag"
        print("[GRP-ALIGN] Disabled by flag — skipping trajectory alignment.")
        return vehicles, vehicle_times, ego_trajs, ego_times, report

    # Coordinate transform parameters (V2XPNP → CARLA is the inverse)
    scale = float(align_cfg.get("scale", 1.0))
    theta_deg = float(align_cfg.get("theta_deg", 0.0))
    tx = float(align_cfg.get("tx", 0.0))
    ty = float(align_cfg.get("ty", 0.0))
    flip_y = bool(align_cfg.get("flip_y", False))
    inv_scale = 1.0 / scale if abs(scale) > 1e-12 else 1.0

    def v2x_to_carla(x: float, y: float) -> Tuple[float, float]:
        cx, cy = invert_se2((x, y), theta_deg, tx, ty, flip_y=flip_y)
        return cx * inv_scale, cy * inv_scale

    def carla_to_v2x(x: float, y: float) -> Tuple[float, float]:
        sx, sy = x * scale, y * scale
        return apply_se2((sx, sy), theta_deg, tx, ty, flip_y=flip_y)

    def yaw_v2x_to_carla(yaw_v2x: float) -> float:
        """Transform yaw from V2XPNP frame to CARLA frame."""
        adjusted = float(yaw_v2x) - float(theta_deg)
        if flip_y:
            adjusted = -adjusted
        return _normalize_yaw_deg(adjusted)

    def yaw_carla_to_v2x(yaw_carla: float) -> float:
        """Transform yaw from CARLA frame to V2XPNP frame."""
        adjusted = float(yaw_carla)
        if flip_y:
            adjusted = -adjusted
        return _normalize_yaw_deg(adjusted + float(theta_deg))

    # --- Connect to CARLA ---
    print(f"[GRP-ALIGN] Connecting to CARLA at {carla_host}:{carla_port} ...")
    try:
        carla, GlobalRoutePlanner, GlobalRoutePlannerDAO = _ensure_carla_grp()
        client = carla.Client(carla_host, int(carla_port))
        client.set_timeout(30.0)
        world = client.get_world()

        current_map = world.get_map().name
        current_map_base = current_map.split("/")[-1] if "/" in current_map else current_map
        if current_map_base != carla_map_name:
            print(f"[GRP-ALIGN] Loading map '{carla_map_name}' (current: {current_map_base}) ...")
            world = client.load_world(carla_map_name)

        carla_map = world.get_map()
        print(f"[GRP-ALIGN] Connected — map: {carla_map.name}")
    except Exception as exc:
        print(f"[GRP-ALIGN] CARLA connection failed: {exc}")
        print("[GRP-ALIGN] Continuing without GRP alignment.")
        report["reason"] = f"carla_connection_failed: {exc}"
        return vehicles, vehicle_times, ego_trajs, ego_times, report

    # --- Set up GRP ---
    try:
        grp_init_params = list(_inspect.signature(GlobalRoutePlanner.__init__).parameters.values())[1:]
        if len(grp_init_params) >= 2 and grp_init_params[0].name != "dao":
            grp = GlobalRoutePlanner(carla_map, sampling_resolution)
        elif GlobalRoutePlannerDAO is not None:
            grp = GlobalRoutePlanner(GlobalRoutePlannerDAO(carla_map, sampling_resolution))
        else:
            grp = GlobalRoutePlanner(carla_map, sampling_resolution)
        if hasattr(grp, "setup"):
            grp.setup()
        print(f"[GRP-ALIGN] GlobalRoutePlanner ready (resolution={sampling_resolution}m).")
    except Exception as exc:
        print(f"[GRP-ALIGN] GRP setup failed: {exc}")
        report["reason"] = f"grp_setup_failed: {exc}"
        return vehicles, vehicle_times, ego_trajs, ego_times, report

    # --- Helper: align a single trajectory ---
    def _align_one(
        traj: List[Waypoint],
        times: List[float],
        label: str,
    ) -> Tuple[Optional[List[Waypoint]], Optional[List[float]], Dict[str, object]]:
        """Align one trajectory. Returns (new_traj, new_times, detail) or (None, None, detail)."""
        detail: Dict[str, object] = {"label": label, "original_len": len(traj)}

        if len(traj) < 2:
            detail["action"] = "skipped_short"
            return None, None, detail

        # Transform to CARLA coords
        wps_carla = []
        for wp in traj:
            cx, cy = v2x_to_carla(float(wp.x), float(wp.y))
            cyaw = yaw_v2x_to_carla(float(wp.yaw))
            wps_carla.append({"x": cx, "y": cy, "z": float(wp.z), "yaw": cyaw})

        ts = list(times) if times else [float(i) * default_dt for i in range(len(traj))]

        try:
            aligned = _grp_refine_trajectory_timing_preserving(
                carla_map=carla_map,
                grp=grp,
                waypoints_carla=wps_carla,
                timestamps=ts,
                radius=snap_radius,
                k=snap_k,
                heading_thresh=heading_thresh,
                lane_change_penalty=lane_change_penalty,
            )
        except Exception as exc:
            detail["action"] = f"alignment_failed: {exc}"
            return None, None, detail

        if not aligned or not any(r.get("grp_aligned", False) for r in aligned):
            detail["action"] = "alignment_produced_no_result"
            return None, None, detail

        # Transform back to V2XPNP coords
        new_traj = []
        new_times = []
        for r in aligned:
            vx, vy = carla_to_v2x(float(r["x"]), float(r["y"]))
            vyaw = yaw_carla_to_v2x(float(r["yaw"]))
            new_traj.append(Waypoint(x=vx, y=vy, z=float(r["z"]), yaw=vyaw))
            new_times.append(float(r["t"]))

        detail["action"] = "aligned"
        detail["aligned_len"] = len(new_traj)

        # Compute displacement stats
        orig_xy = np.array([[float(wp.x), float(wp.y)] for wp in traj[:len(new_traj)]])
        new_xy = np.array([[float(wp.x), float(wp.y)] for wp in new_traj])
        min_len = min(orig_xy.shape[0], new_xy.shape[0])
        if min_len > 0:
            disps = np.sqrt(np.sum((orig_xy[:min_len] - new_xy[:min_len]) ** 2, axis=1))
            detail["median_displacement_m"] = float(np.median(disps))
            detail["p90_displacement_m"] = float(np.quantile(disps, 0.9))
            detail["max_displacement_m"] = float(np.max(disps))

        return new_traj, new_times, detail

    # --- Process actor vehicles (two-pass) ---
    #
    # Pass 1: classify every actor as skip / stationary / moving so we can
    #         build a spatial index of moving-vehicle paths *before* we
    #         decide where to freeze parked vehicles.
    # Pass 2: freeze stationary vehicles (with outer-lane bias) and GRP-
    #         align moving ones.
    # ------------------------------------------------------------------
    actor_vids = sorted(vehicles.keys())
    n_actors = len(actor_vids)
    print(f"[GRP-ALIGN] Processing {n_actors} actor trajectories ...")

    # -- Pass 1: classify ------------------------------------------------
    _cls_skip: list = []       # (vid, reason_str)
    _cls_stationary: list = [] # (vid, stat_reason, stat_stats)
    _cls_moving: list = []     # vid

    for vid in actor_vids:
        traj = vehicles[vid]
        times = vehicle_times.get(vid, [])
        meta = dict(obj_info.get(vid, {}))
        obj_type = str(meta.get("obj_type") or "npc")

        if not is_vehicle_type(obj_type):
            _cls_skip.append((vid, "not_vehicle"))
            continue
        if _is_pedestrian_type(obj_type) or _is_cyclist_type(obj_type):
            _cls_skip.append((vid, f"non_motor ({obj_type})"))
            continue

        is_stat, stat_reason, stat_stats = _is_actor_confidently_stationary(
            traj, times, default_dt, parked_vehicle_cfg,
        )
        if is_stat:
            _cls_stationary.append((vid, stat_reason, stat_stats))
        else:
            _cls_moving.append(vid)

    stationary_vids = {v for v, _, _ in _cls_stationary}
    print(
        f"[GRP-ALIGN]   Classification: {len(_cls_skip)} skipped, "
        f"{len(_cls_stationary)} stationary, {len(_cls_moving)} moving"
    )

    # -- Build moving-vehicle spatial index (CARLA coords) ----------------
    # We include ego trajectories as well — a parked car should not be
    # pushed into the ego's path either.
    OUTER_LANE_CELL_SIZE = 2.0
    OBSTRUCTION_CHECK_RADIUS = 4.0  # Distance to check for parked vehicle obstructions
    
    moving_cells = _build_moving_vehicle_spatial_set(
        vehicles, stationary_vids, v2x_to_carla,
        cell_size=OUTER_LANE_CELL_SIZE,
    )
    
    # Build separate ego-specific cell set for priority obstruction detection
    ego_cells: set = set()
    inv_cs = 1.0 / OUTER_LANE_CELL_SIZE
    for etraj in ego_trajs:
        for wp in etraj:
            cx_e, cy_e = v2x_to_carla(float(wp.x), float(wp.y))
            ego_cells.add((
                int(math.floor(cx_e * inv_cs)),
                int(math.floor(cy_e * inv_cs)),
            ))
    # Also add ego cells to moving_cells
    moving_cells.update(ego_cells)

    # -- Pass 2a: handle skipped actors ----------------------------------
    for vid, reason in _cls_skip:
        report["actors_skipped_short"] = int(report["actors_skipped_short"]) + 1
        if "non_motor" in reason:
            report["actor_details"][int(vid)] = {
                "action": f"skipped_{reason}",
                "original_len": len(vehicles[vid]),
            }

    # -- Pass 2b: freeze stationary actors (with outer-lane bias) ---------
    # Track parked vehicle positions for obstruction removal
    parked_positions_carla: Dict[int, Tuple[float, float]] = {}  # vid -> (carla_x, carla_y)
    
    n_outer_biased = 0
    for vid, stat_reason, stat_stats in _progress_bar(
        _cls_stationary,
        total=len(_cls_stationary),
        desc="[GRP-ALIGN] Freezing parked",
    ):
        traj = vehicles[vid]
        report["actors_processed"] = int(report["actors_processed"]) + 1

        med_x = float(np.median([float(wp.x) for wp in traj]))
        med_y = float(np.median([float(wp.y) for wp in traj]))
        med_z = float(np.median([float(wp.z) for wp in traj]))
        med_yaw = float(traj[len(traj) // 2].yaw)
        cx, cy = v2x_to_carla(med_x, med_y)

        outer_bias_reason = "no_snap"
        try:
            carla_loc = carla.Location(x=float(cx), y=float(cy), z=float(med_z))
            snapped_wp = carla_map.get_waypoint(
                carla_loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if snapped_wp is not None:
                # --- Outer-lane bias ---
                final_wp, did_bias, outer_bias_reason = _try_bias_parked_to_outer_lane(
                    snapped_wp,
                    carla_map,
                    carla,
                    moving_cells,
                    cell_size=OUTER_LANE_CELL_SIZE,
                    max_lateral_m=4.0,
                    interference_radius=3.0,
                )
                if did_bias:
                    n_outer_biased += 1

                floc = final_wp.transform.location
                sx, sy = carla_to_v2x(float(floc.x), float(floc.y))
                syaw = yaw_carla_to_v2x(float(final_wp.transform.rotation.yaw))
                frozen = Waypoint(x=sx, y=sy, z=float(floc.z), yaw=syaw)
                # Track CARLA position for obstruction checking
                parked_positions_carla[vid] = (float(floc.x), float(floc.y))
            else:
                frozen = Waypoint(x=med_x, y=med_y, z=med_z, yaw=med_yaw)
                # Track CARLA position from original coords
                parked_positions_carla[vid] = (cx, cy)
        except Exception:
            frozen = Waypoint(x=med_x, y=med_y, z=med_z, yaw=med_yaw)
            parked_positions_carla[vid] = (cx, cy)

        vehicles[vid] = [frozen] * len(traj)
        report["actors_skipped_stationary"] = int(report["actors_skipped_stationary"]) + 1
        report["actor_details"][int(vid)] = {
            "action": f"frozen_stationary ({stat_reason})",
            "original_len": len(traj),
            "frozen_x": float(frozen.x),
            "frozen_y": float(frozen.y),
            "outer_lane_bias": outer_bias_reason,
        }

    if n_outer_biased:
        print(f"[GRP-ALIGN]   Outer-lane bias applied to {n_outer_biased}/{len(_cls_stationary)} parked vehicles")

    # -- Pass 2b-ii: Remove parked vehicles that obstruct moving actors or ego --
    obstructs_ego, obstructs_actor = _find_obstructing_parked_vehicles(
        parked_positions_carla,
        moving_cells,
        ego_cells,
        cell_size=OUTER_LANE_CELL_SIZE,
        obstruction_radius=OBSTRUCTION_CHECK_RADIUS,
    )
    
    # Initialize removal tracking in report
    report["parked_removed_obstructing_ego"] = []
    report["parked_removed_obstructing_actor"] = []
    
    # Remove parked vehicles that obstruct ego (highest priority)
    for vid in obstructs_ego:
        if vid in vehicles:
            print(f"[GRP-ALIGN]   REMOVED actor_{vid}: parked vehicle obstructing EGO path")
            del vehicles[vid]
            if vid in vehicle_times:
                del vehicle_times[vid]
            # Update report
            if int(vid) in report["actor_details"]:
                report["actor_details"][int(vid)]["action"] = "REMOVED_obstruct_ego"
            report["parked_removed_obstructing_ego"].append(int(vid))
    
    # Remove parked vehicles that obstruct other moving actors
    for vid in obstructs_actor:
        if vid in vehicles:
            print(f"[GRP-ALIGN]   REMOVED actor_{vid}: parked vehicle obstructing moving actor path")
            del vehicles[vid]
            if vid in vehicle_times:
                del vehicle_times[vid]
            # Update report
            if int(vid) in report["actor_details"]:
                report["actor_details"][int(vid)]["action"] = "REMOVED_obstruct_actor"
            report["parked_removed_obstructing_actor"].append(int(vid))
    
    total_removed = len(obstructs_ego) + len(obstructs_actor)
    if total_removed > 0:
        print(f"[GRP-ALIGN]   Removed {total_removed} obstructing parked vehicles "
              f"({len(obstructs_ego)} blocking ego, {len(obstructs_actor)} blocking actors)")

    # -- Pass 2c: GRP-align moving actors ---------------------------------
    for vid in _progress_bar(_cls_moving, total=len(_cls_moving), desc="[GRP-ALIGN] Aligning"):
        traj = vehicles[vid]
        times = vehicle_times.get(vid, [])
        report["actors_processed"] = int(report["actors_processed"]) + 1

        new_traj, new_times, detail = _align_one(traj, times, f"actor_{vid}")
        report["actor_details"][int(vid)] = detail

        if new_traj is not None and new_times is not None:
            vehicles[vid] = new_traj
            vehicle_times[vid] = new_times
            report["actors_aligned"] = int(report["actors_aligned"]) + 1
        else:
            report["actors_failed"] = int(report["actors_failed"]) + 1

    # --- Process ego trajectories ---
    n_egos = len(ego_trajs)
    print(f"[GRP-ALIGN] Processing {n_egos} ego trajectories ...")

    for ego_idx in range(n_egos):
        traj = ego_trajs[ego_idx]
        times = ego_times[ego_idx] if ego_idx < len(ego_times) else []

        report["egos_processed"] = int(report["egos_processed"]) + 1

        # Check if ego is stationary (unlikely but possible)
        is_stat, stat_reason, _ = _is_actor_confidently_stationary(
            traj, times, default_dt, parked_vehicle_cfg,
        )
        if is_stat:
            report["actor_details"][f"ego_{ego_idx}"] = {  # type: ignore
                "action": f"skipped_stationary ({stat_reason})",
                "original_len": len(traj),
            }
            continue

        new_traj, new_times, detail = _align_one(traj, times, f"ego_{ego_idx}")
        report["actor_details"][f"ego_{ego_idx}"] = detail  # type: ignore

        if new_traj is not None and new_times is not None:
            ego_trajs[ego_idx] = new_traj
            if ego_idx < len(ego_times):
                ego_times[ego_idx] = new_times
            report["egos_aligned"] = int(report["egos_aligned"]) + 1
        else:
            report["egos_failed"] = int(report["egos_failed"]) + 1

    # Summary
    print(
        f"[GRP-ALIGN] Done — "
        f"actors: {report['actors_aligned']} aligned, "
        f"{report['actors_skipped_stationary']} stationary, "
        f"{report['actors_failed']} failed | "
        f"egos: {report['egos_aligned']} aligned, "
        f"{report['egos_failed']} failed"
    )

    return vehicles, vehicle_times, ego_trajs, ego_times, report


def _suppress_short_lane_runs(
    lanes: List[int],
    max_short_run: int,
    endpoint_short_run: int,
) -> List[int]:
    if not lanes:
        return []
    out = [int(v) for v in lanes]

    def _build_runs(vals: List[int]) -> List[Tuple[int, int, int]]:
        runs: List[Tuple[int, int, int]] = []
        s = 0
        cur = int(vals[0])
        for i in range(1, len(vals)):
            if int(vals[i]) != cur:
                runs.append((cur, s, i - 1))
                s = i
                cur = int(vals[i])
        runs.append((cur, s, len(vals) - 1))
        return runs

    changed = True
    guard_iters = 0
    while changed and guard_iters < 6:
        guard_iters += 1
        changed = False
        runs = _build_runs(out)
        for i, (lane_id, s, e) in enumerate(runs):
            run_len = int(e - s + 1)
            if lane_id < 0:
                continue
            if 0 < i < len(runs) - 1:
                prev_lane = int(runs[i - 1][0])
                next_lane = int(runs[i + 1][0])
                if prev_lane == next_lane and prev_lane >= 0 and run_len <= int(max_short_run):
                    for j in range(s, e + 1):
                        out[j] = prev_lane
                    changed = True
            elif i == 0 and len(runs) > 1 and run_len <= int(endpoint_short_run):
                next_lane = int(runs[i + 1][0])
                if next_lane >= 0:
                    for j in range(s, e + 1):
                        out[j] = next_lane
                    changed = True
            elif i == len(runs) - 1 and len(runs) > 1 and run_len <= int(endpoint_short_run):
                prev_lane = int(runs[i - 1][0])
                if prev_lane >= 0:
                    for j in range(s, e + 1):
                        out[j] = prev_lane
                    changed = True
    return out


def _stabilize_lane_sequence(
    traj: Sequence[Waypoint],
    raw_candidates: Sequence[Sequence[Dict[str, object]]],
    matcher: LaneMatcher,
    cfg: Dict[str, float],
) -> List[int]:
    n = int(len(traj))
    if n <= 0:
        return []

    raw_lane: List[int] = []
    raw_dist: List[float] = []
    for cands in raw_candidates:
        if cands:
            raw_lane.append(int(cands[0]["lane_index"]))
            raw_dist.append(float(cands[0]["dist"]))
        else:
            raw_lane.append(-1)
            raw_dist.append(float("inf"))

    confirm_window = max(2, int(cfg.get("confirm_window", 5)))
    confirm_votes = max(2, int(cfg.get("confirm_votes", 3)))
    cooldown_frames = max(0, int(cfg.get("cooldown_frames", 3)))
    endpoint_guard = max(0, int(cfg.get("endpoint_guard_frames", 4)))
    endpoint_extra_votes = max(0, int(cfg.get("endpoint_extra_votes", 1)))
    min_improve_m = max(0.0, float(cfg.get("min_improvement_m", 0.2)))
    keep_lane_max_dist = max(0.0, float(cfg.get("keep_lane_max_dist", 3.0)))
    short_run_max = max(0, int(cfg.get("short_run_max", 2)))
    endpoint_short_run = max(0, int(cfg.get("endpoint_short_run", 2)))

    anchor_window = min(n, max(confirm_window + endpoint_guard, 6))
    start_anchor = _dominant_lane(raw_lane[:anchor_window])
    end_anchor = _dominant_lane(raw_lane[max(0, n - anchor_window):])

    current_lane = start_anchor if start_anchor >= 0 else next((lid for lid in raw_lane if lid >= 0), -1)
    cooldown = 0
    stable_lane: List[int] = []
    for i in range(n):
        raw_lid = int(raw_lane[i])
        chosen = int(current_lane)

        if chosen < 0:
            if raw_lid >= 0:
                chosen = raw_lid
            elif i < endpoint_guard and start_anchor >= 0:
                chosen = int(start_anchor)

        if chosen >= 0 and raw_lid >= 0 and raw_lid != chosen:
            required_votes = int(confirm_votes)
            if i < endpoint_guard or i >= max(0, n - endpoint_guard):
                required_votes += int(endpoint_extra_votes)

            w0 = i
            w1 = min(n, i + confirm_window)
            support_raw = sum(1 for j in range(w0, w1) if int(raw_lane[j]) == raw_lid)
            if i >= max(0, n - endpoint_guard):
                # Near tail, incorporate short look-back support too.
                wb0 = max(0, i - confirm_window + 1)
                support_raw = max(support_raw, sum(1 for j in range(wb0, i + 1) if int(raw_lane[j]) == raw_lid))

            cur_proj = matcher.project_to_lane(chosen, float(traj[i].x), float(traj[i].y), float(traj[i].z)) if chosen >= 0 else None
            cur_dist = float(cur_proj["dist"]) if cur_proj is not None else float("inf")
            raw_d = float(raw_dist[i])
            dist_improve = float(cur_dist - raw_d)

            if support_raw >= required_votes:
                support_strong = support_raw >= (required_votes + 1)
                if cooldown > 0 and not support_strong:
                    pass
                else:
                    if support_strong or dist_improve >= min_improve_m or cur_dist > keep_lane_max_dist:
                        chosen = raw_lid
                        cooldown = int(cooldown_frames)

        if i < endpoint_guard and start_anchor >= 0 and chosen != int(start_anchor):
            # Suppress startup lane jumps unless strongly supported.
            w1 = min(n, i + confirm_window + 1)
            support_chosen = sum(1 for j in range(i, w1) if int(raw_lane[j]) == int(chosen))
            if support_chosen < (confirm_votes + endpoint_extra_votes + 1):
                chosen = int(start_anchor)
                cooldown = max(cooldown, int(cooldown_frames))

        if i >= max(0, n - endpoint_guard) and end_anchor >= 0 and chosen != int(end_anchor):
            # Suppress tail-end flips unless they dominate tail support.
            w0 = max(0, n - max(confirm_window + endpoint_guard, 6))
            support_chosen = sum(1 for j in range(w0, n) if int(raw_lane[j]) == int(chosen))
            support_end = sum(1 for j in range(w0, n) if int(raw_lane[j]) == int(end_anchor))
            if support_chosen <= support_end:
                chosen = int(end_anchor)
                cooldown = max(cooldown, int(cooldown_frames))

        stable_lane.append(int(chosen))
        current_lane = int(chosen)
        if cooldown > 0:
            cooldown -= 1

    stable_lane = _suppress_short_lane_runs(
        stable_lane,
        max_short_run=int(short_run_max),
        endpoint_short_run=int(endpoint_short_run),
    )
    return stable_lane


def _build_track_frames(
    traj: Sequence[Waypoint],
    times: Sequence[float] | None,
    matcher: LaneMatcher,
    snap_to_map: bool,
    lane_change_cfg: Optional[Dict[str, object]] = None,
    lane_policy: Optional[Dict[str, object]] = None,
    skip_snap: bool = False,
) -> List[Dict[str, object]]:
    # --- Pedestrians / actors that should never be lane-snapped ---
    if skip_snap:
        frames: List[Dict[str, object]] = []
        for i, wp in enumerate(traj):
            t = float(times[i]) if (times is not None and i < len(times)) else float(i)
            frames.append(
                {
                    "t": float(round(t, 6)),
                    "rx": float(wp.x),
                    "ry": float(wp.y),
                    "rz": float(wp.z),
                    "ryaw": float(wp.yaw),
                    "mx": float(wp.x),
                    "my": float(wp.y),
                    "mz": float(wp.z),
                    "myaw": float(wp.yaw),
                    "x": float(wp.x),
                    "y": float(wp.y),
                    "z": float(wp.z),
                    "yaw": float(wp.yaw),
                    "li": -1,
                    "ld": float("inf"),
                    "li_raw": -1,
                    "ld_raw": float("inf"),
                }
            )
        return frames

    cfg = dict(lane_change_cfg or {})
    lane_filter_enabled = bool(cfg.get("enabled", True))
    lane_top_k = max(2, int(cfg.get("lane_top_k", 8)))
    policy = dict(lane_policy or {})
    allowed_lane_types_raw = policy.get("allowed_lane_types")
    allowed_lane_types: Optional[set[str]] = None
    if allowed_lane_types_raw is not None:
        allowed_lane_types = _parse_lane_type_set(allowed_lane_types_raw, fallback=[])
    stationary_lane_types = _parse_lane_type_set(policy.get("stationary_when_lane_types"), fallback=[])

    raw_candidates: List[List[Dict[str, object]]] = []
    raw_best_unfiltered: List[Optional[Dict[str, object]]] = []
    for wp in traj:
        nearest = matcher.match_candidates(float(wp.x), float(wp.y), float(wp.z), lane_top_k=lane_top_k)
        raw_best_unfiltered.append(nearest[0] if nearest else None)
        filtered = list(nearest)
        if allowed_lane_types is not None:
            filtered = [c for c in nearest if str(c.get("lane_type", "")) in allowed_lane_types]
            if not filtered:
                wide_k = max(int(lane_top_k) * 8, 48)
                if wide_k > int(lane_top_k):
                    wider = matcher.match_candidates(float(wp.x), float(wp.y), float(wp.z), lane_top_k=wide_k)
                    filtered = [c for c in wider if str(c.get("lane_type", "")) in allowed_lane_types]
        raw_candidates.append(filtered)

    stable_lane_idx: List[int]
    if lane_filter_enabled:
        stable_lane_idx = _stabilize_lane_sequence(
            traj=traj,
            raw_candidates=raw_candidates,
            matcher=matcher,
            cfg=cfg,
        )
    else:
        stable_lane_idx = [int(cands[0]["lane_index"]) if cands else -1 for cands in raw_candidates]

    force_stationary = False
    stationary_anchor: Optional[Dict[str, object]] = None
    stationary_lane_idx = -1
    if stationary_lane_types:
        lane_type_seq: List[str] = []
        for lane_idx in stable_lane_idx:
            if int(lane_idx) < 0:
                lane_type_seq.append("")
            else:
                lane_type_seq.append(str(matcher.map_data.lanes[int(lane_idx)].lane_type))
        if any(lt in stationary_lane_types for lt in lane_type_seq):
            preferred = [
                int(li)
                for li in stable_lane_idx
                if int(li) >= 0 and str(matcher.map_data.lanes[int(li)].lane_type) in stationary_lane_types
            ]
            if not preferred:
                preferred = [int(li) for li in stable_lane_idx if int(li) >= 0]
            stationary_lane_idx = _dominant_lane(preferred)
            if stationary_lane_idx >= 0:
                med_x = float(np.median([float(wp.x) for wp in traj])) if traj else 0.0
                med_y = float(np.median([float(wp.y) for wp in traj])) if traj else 0.0
                med_z = float(np.median([float(wp.z) for wp in traj])) if traj else 0.0
                stationary_anchor = matcher.project_to_lane(stationary_lane_idx, med_x, med_y, med_z)
                if stationary_anchor is None and traj:
                    wp0 = traj[0]
                    stationary_anchor = matcher.project_to_lane(
                        stationary_lane_idx,
                        float(wp0.x),
                        float(wp0.y),
                        float(wp0.z),
                    )
                if stationary_anchor is not None:
                    force_stationary = True
                    stable_lane_idx = [int(stationary_lane_idx) for _ in stable_lane_idx]

    frames: List[Dict[str, object]] = []
    for i, wp in enumerate(traj):
        t = float(times[i]) if (times is not None and i < len(times)) else float(i)
        raw_match = raw_best_unfiltered[i] if i < len(raw_best_unfiltered) else None
        chosen_lane = int(stable_lane_idx[i]) if i < len(stable_lane_idx) else -1

        if force_stationary and stationary_anchor is not None and int(stationary_lane_idx) >= 0:
            map_x = float(stationary_anchor["x"])
            map_y = float(stationary_anchor["y"])
            map_z = float(stationary_anchor["z"])
            map_yaw = float(stationary_anchor["yaw"])
            lane_idx = int(stationary_lane_idx)
            cur_proj = matcher.project_to_lane(int(stationary_lane_idx), float(wp.x), float(wp.y), float(wp.z))
            lane_dist = float(cur_proj["dist"]) if cur_proj is not None else float(stationary_anchor.get("dist", 0.0))
        else:
            match: Optional[Dict[str, object]] = None
            if chosen_lane >= 0:
                for cand in raw_candidates[i]:
                    if int(cand.get("lane_index", -1)) == chosen_lane:
                        match = cand
                        break
                if match is None:
                    match = matcher.project_to_lane(chosen_lane, float(wp.x), float(wp.y), float(wp.z))
                    if (
                        match is not None
                        and allowed_lane_types is not None
                        and str(match.get("lane_type", "")) not in allowed_lane_types
                    ):
                        match = None
            if match is None:
                if (
                    raw_match is not None
                    and (
                        allowed_lane_types is None
                        or str(raw_match.get("lane_type", "")) in allowed_lane_types
                    )
                ):
                    match = raw_match

            if match is None:
                map_x, map_y, map_z, map_yaw = float(wp.x), float(wp.y), float(wp.z), float(wp.yaw)
                lane_idx, lane_dist = -1, float("inf")
            else:
                map_x = float(match["x"])
                map_y = float(match["y"])
                map_z = float(match["z"])
                map_yaw = float(match["yaw"])
                lane_idx = int(match["lane_index"])
                lane_dist = float(match["dist"])

        if lane_idx < 0 and raw_match is None:
            map_x, map_y, map_z, map_yaw = float(wp.x), float(wp.y), float(wp.z), float(wp.yaw)
            lane_idx, lane_dist = -1, float("inf")

        display_x = map_x if snap_to_map else float(wp.x)
        display_y = map_y if snap_to_map else float(wp.y)
        display_z = map_z if snap_to_map else float(wp.z)
        display_yaw = map_yaw if snap_to_map else float(wp.yaw)
        frames.append(
            {
                "t": float(round(t, 6)),
                "rx": float(wp.x),
                "ry": float(wp.y),
                "rz": float(wp.z),
                "ryaw": float(wp.yaw),
                "mx": float(map_x),
                "my": float(map_y),
                "mz": float(map_z),
                "myaw": float(map_yaw),
                "x": float(display_x),
                "y": float(display_y),
                "z": float(display_z),
                "yaw": float(display_yaw),
                "li": int(lane_idx),
                "ld": float(lane_dist),
                "li_raw": int(raw_match["lane_index"]) if raw_match is not None else -1,
                "ld_raw": float(raw_match["dist"]) if raw_match is not None else float("inf"),
            }
        )
    return frames


def _trajectory_path_length(traj: Sequence[Waypoint]) -> float:
    if len(traj) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(traj, traj[1:]):
        total += math.hypot(float(b.x) - float(a.x), float(b.y) - float(a.y))
    return float(total)


def _downsample_line(points: np.ndarray, max_points: int) -> List[List[float]]:
    if points.shape[0] == 0:
        return []
    if max_points <= 0 or points.shape[0] <= max_points:
        return [[float(p[0]), float(p[1]), float(p[2])] for p in points]
    idx = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int32)
    sampled = points[idx]
    if idx[-1] != points.shape[0] - 1:
        sampled = np.vstack([sampled, points[-1:]])
    return [[float(p[0]), float(p[1]), float(p[2])] for p in sampled]


def _collect_timeline(ego_tracks: Sequence[Dict[str, object]], actor_tracks: Sequence[Dict[str, object]]) -> List[float]:
    times: set[float] = set()
    for track in list(ego_tracks) + list(actor_tracks):
        for fr in track.get("frames", []):
            try:
                times.add(float(fr["t"]))
            except Exception:
                continue
    if not times:
        return [0.0]
    return sorted(times)


def _polyline_xy_from_points(points: object) -> np.ndarray:
    if not isinstance(points, (list, tuple)):
        return np.zeros((0, 2), dtype=np.float64)
    out: List[Tuple[float, float]] = []
    for p in points:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            x = _safe_float(p[0], float("nan"))
            y = _safe_float(p[1], float("nan"))
            if math.isfinite(x) and math.isfinite(y):
                out.append((float(x), float(y)))
    if len(out) < 2:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(out, dtype=np.float64)


def _polyline_cumlen_xy(poly_xy: np.ndarray) -> np.ndarray:
    n = int(poly_xy.shape[0])
    if n <= 0:
        return np.zeros((0,), dtype=np.float64)
    if n == 1:
        return np.zeros((1,), dtype=np.float64)
    seg = poly_xy[1:] - poly_xy[:-1]
    seg_len = np.sqrt(np.sum(seg * seg, axis=1))
    cum = np.zeros((n,), dtype=np.float64)
    cum[1:] = np.cumsum(seg_len)
    return cum


def _polyline_total_len(cumlen: np.ndarray) -> float:
    if cumlen.size <= 0:
        return 0.0
    return float(cumlen[-1])


def _yaw_abs_diff_deg(a: float, b: float) -> float:
    return abs(_normalize_yaw_deg(float(a) - float(b)))


def _project_point_to_polyline_xy(poly_xy: np.ndarray, cumlen: np.ndarray, x: float, y: float) -> Optional[Dict[str, float]]:
    if poly_xy.shape[0] <= 0:
        return None
    if poly_xy.shape[0] == 1:
        px = float(poly_xy[0, 0])
        py = float(poly_xy[0, 1])
        return {
            "x": px,
            "y": py,
            "yaw": 0.0,
            "dist": float(math.hypot(px - x, py - y)),
            "s": 0.0,
            "s_norm": 0.0,
            "segment_idx": 0.0,
        }

    p0 = poly_xy[:-1]
    p1 = poly_xy[1:]
    seg = p1 - p0
    seg_len2 = np.sum(seg * seg, axis=1)
    valid = seg_len2 > 1e-12
    if not np.any(valid):
        px = float(poly_xy[0, 0])
        py = float(poly_xy[0, 1])
        return {
            "x": px,
            "y": py,
            "yaw": 0.0,
            "dist": float(math.hypot(px - x, py - y)),
            "s": 0.0,
            "s_norm": 0.0,
            "segment_idx": 0.0,
        }

    xy = np.asarray([float(x), float(y)], dtype=np.float64)
    t = np.zeros((seg.shape[0],), dtype=np.float64)
    t[valid] = np.sum((xy - p0[valid]) * seg[valid], axis=1) / seg_len2[valid]
    t = np.clip(t, 0.0, 1.0)
    proj = p0 + seg * t[:, None]
    d2 = np.sum((proj - xy[None, :]) ** 2, axis=1)
    best = int(np.argmin(d2))
    best_t = float(t[best])
    seg_dx = float(seg[best, 0])
    seg_dy = float(seg[best, 1])
    seg_len = math.hypot(seg_dx, seg_dy)
    yaw = _normalize_yaw_deg(math.degrees(math.atan2(seg_dy, seg_dx))) if seg_len > 1e-9 else 0.0
    s_base = float(cumlen[best]) if cumlen.size > best else 0.0
    s = s_base + float(best_t) * float(seg_len)
    total_len = _polyline_total_len(cumlen)
    s_norm = (s / total_len) if total_len > 1e-9 else 0.0
    return {
        "x": float(proj[best, 0]),
        "y": float(proj[best, 1]),
        "yaw": float(yaw),
        "dist": float(math.sqrt(max(0.0, float(d2[best])))),
        "s": float(s),
        "s_norm": float(max(0.0, min(1.0, float(s_norm)))),
        "segment_idx": float(best + best_t),
    }


def _sample_polyline_at_s_xy(poly_xy: np.ndarray, cumlen: np.ndarray, s: float) -> Optional[Dict[str, float]]:
    n = int(poly_xy.shape[0])
    if n <= 0:
        return None
    if n == 1:
        return {
            "x": float(poly_xy[0, 0]),
            "y": float(poly_xy[0, 1]),
            "yaw": 0.0,
            "s": 0.0,
            "s_norm": 0.0,
            "segment_idx": 0.0,
        }
    total_len = _polyline_total_len(cumlen)
    if total_len <= 1e-9:
        return {
            "x": float(poly_xy[0, 0]),
            "y": float(poly_xy[0, 1]),
            "yaw": 0.0,
            "s": 0.0,
            "s_norm": 0.0,
            "segment_idx": 0.0,
        }
    ss = max(0.0, min(float(total_len), float(s)))
    idx = int(np.searchsorted(cumlen, ss, side="right") - 1)
    idx = max(0, min(idx, n - 2))
    s0 = float(cumlen[idx])
    s1 = float(cumlen[idx + 1])
    denom = max(1e-9, s1 - s0)
    t = max(0.0, min(1.0, (ss - s0) / denom))
    p0 = poly_xy[idx]
    p1 = poly_xy[idx + 1]
    x = float(p0[0] + t * (p1[0] - p0[0]))
    y = float(p0[1] + t * (p1[1] - p0[1]))
    yaw = _normalize_yaw_deg(math.degrees(math.atan2(float(p1[1] - p0[1]), float(p1[0] - p0[0]))))
    return {
        "x": x,
        "y": y,
        "yaw": float(yaw),
        "s": float(ss),
        "s_norm": float(ss / total_len),
        "segment_idx": float(idx + t),
    }


def _sample_polyline_at_ratio_xy(poly_xy: np.ndarray, cumlen: np.ndarray, ratio: float) -> Optional[Dict[str, float]]:
    total_len = _polyline_total_len(cumlen)
    r = max(0.0, min(1.0, float(ratio)))
    return _sample_polyline_at_s_xy(poly_xy, cumlen, r * total_len)


def _evaluate_lane_pair_metrics(v2_xy: np.ndarray, v2_cum: np.ndarray, carla_xy: np.ndarray, carla_cum: np.ndarray) -> Dict[str, float]:
    """Evaluate geometric similarity between a V2XPNP lane and a CARLA line.

    Returns a dict with individual metric values plus a composite *score*
    (lower = better).  A hard direction filter rejects anti-parallel
    candidates by setting score = inf when monotonic_ratio < 0.45.
    """
    _INF_METRICS: Dict[str, float] = {
        "score": float("inf"),
        "median_dist_m": float("inf"),
        "p90_dist_m": float("inf"),
        "mean_dist_m": float("inf"),
        "coverage_1m": 0.0,
        "coverage_2m": 0.0,
        "angle_median_deg": 180.0,
        "angle_p90_deg": 180.0,
        "monotonic_ratio": 0.0,
        "length_ratio": 0.0,
        "end_dist_m": float("inf"),
        "n_samples": 0.0,
    }
    v2_len = _polyline_total_len(v2_cum)
    carla_len = _polyline_total_len(carla_cum)
    if v2_xy.shape[0] < 2 or carla_xy.shape[0] < 2 or v2_len <= 1e-6 or carla_len <= 1e-6:
        return dict(_INF_METRICS)

    n_samples = int(max(14, min(84, round(v2_len / 1.6))))
    s_vals = np.linspace(0.0, float(v2_len), n_samples, dtype=np.float64)
    dists: List[float] = []
    angs: List[float] = []
    uvals: List[float] = []
    for sv in s_vals.tolist():
        sp = _sample_polyline_at_s_xy(v2_xy, v2_cum, float(sv))
        if sp is None:
            continue
        proj = _project_point_to_polyline_xy(carla_xy, carla_cum, float(sp["x"]), float(sp["y"]))
        if proj is None:
            continue
        dists.append(float(proj["dist"]))
        angs.append(float(_yaw_abs_diff_deg(float(sp["yaw"]), float(proj["yaw"]))))
        uvals.append(float(proj["s_norm"]))

    if not dists:
        return dict(_INF_METRICS)

    d_arr = np.asarray(dists, dtype=np.float64)
    a_arr = np.asarray(angs, dtype=np.float64)
    u_arr = np.asarray(uvals, dtype=np.float64)
    median_dist = float(np.quantile(d_arr, 0.5))
    p90_dist = float(np.quantile(d_arr, 0.9))
    mean_dist = float(np.mean(d_arr))
    coverage_1m = float(np.mean((d_arr <= 1.0).astype(np.float64)))
    coverage_2m = float(np.mean((d_arr <= 2.0).astype(np.float64)))
    angle_median = float(np.quantile(a_arr, 0.5))
    angle_p90 = float(np.quantile(a_arr, 0.9))
    mono = 1.0
    if u_arr.size >= 2:
        du = np.diff(u_arr)
        mono = float(np.mean((du >= -0.035).astype(np.float64)))
    length_ratio = float(min(v2_len, carla_len) / max(v2_len, carla_len))
    p_start = _project_point_to_polyline_xy(carla_xy, carla_cum, float(v2_xy[0, 0]), float(v2_xy[0, 1]))
    p_end = _project_point_to_polyline_xy(carla_xy, carla_cum, float(v2_xy[-1, 0]), float(v2_xy[-1, 1]))
    end_dist = float(0.5 * ((float(p_start["dist"]) if p_start else 4.0) + (float(p_end["dist"]) if p_end else 4.0)))

    # --- Hard direction filter ---
    # If the monotonic ratio is below threshold, the CARLA line is pointing
    # the wrong way or wraps around oddly.  Reject with infinite score.
    if mono < 0.45:
        out = dict(_INF_METRICS)
        out["monotonic_ratio"] = float(mono)
        out["angle_median_deg"] = float(angle_median)
        out["n_samples"] = float(len(dists))
        return out

    score = (
        1.00 * median_dist
        + 0.40 * p90_dist
        + 0.010 * angle_median
        + 0.90 * (1.0 - coverage_2m)
        + 0.55 * (1.0 - mono)
        + 0.30 * (1.0 - length_ratio)
        + 0.12 * end_dist
    )
    return {
        "score": float(score),
        "median_dist_m": float(median_dist),
        "p90_dist_m": float(p90_dist),
        "mean_dist_m": float(mean_dist),
        "coverage_1m": float(coverage_1m),
        "coverage_2m": float(coverage_2m),
        "angle_median_deg": float(angle_median),
        "angle_p90_deg": float(angle_p90),
        "monotonic_ratio": float(mono),
        "length_ratio": float(length_ratio),
        "end_dist_m": float(end_dist),
        "n_samples": float(len(dists)),
    }


def _quality_from_metrics(metrics: Dict[str, float]) -> str:
    med = float(metrics.get("median_dist_m", float("inf")))
    cov2 = float(metrics.get("coverage_2m", 0.0))
    ang = float(metrics.get("angle_median_deg", 180.0))
    mono = float(metrics.get("monotonic_ratio", 0.0))
    score = float(metrics.get("score", float("inf")))
    if med <= 1.25 and cov2 >= 0.85 and ang <= 25.0 and mono >= 0.85 and score <= 2.4:
        return "high"
    if med <= 2.2 and cov2 >= 0.55 and ang <= 55.0 and mono >= 0.55 and score <= 4.8:
        return "medium"
    if med <= 3.4 and cov2 >= 0.35 and score <= 7.0:
        return "low"
    return "poor"


# ---------------------------------------------------------------------------
# Connectivity / adjacency helpers for global lane correspondence
# ---------------------------------------------------------------------------

def _build_v2_connectivity_graph(
    v2_feats: Dict[int, Dict[str, object]],
    v2_lanes_raw: list,
) -> Dict[str, object]:
    """Build maps from V2XPNP lane index → connected lane indices.

    Uses *entry_lanes* / *exit_lanes* (lists of UID strings like "road_lane")
    stored in the raw lane dicts.

    Returns dict with:
      uid_to_index: {uid_str → lane_index}
      successors:   {lane_index → set of successor lane indices}
      predecessors: {lane_index → set of predecessor lane indices}
      adjacency:    {lane_index → set of adjacent lane indices (same road_id, |Δlane_id|=1)}
    """
    uid_to_index: Dict[str, int] = {}
    for lane in v2_lanes_raw:
        if not isinstance(lane, dict):
            continue
        li = _safe_int(lane.get("index"), -1)
        if li < 0:
            continue
        uid = f"{_safe_int(lane.get('road_id'), 0)}_{_safe_int(lane.get('lane_id'), 0)}"
        uid_to_index[uid] = int(li)

    successors: Dict[int, set] = {}
    predecessors: Dict[int, set] = {}
    for lane in v2_lanes_raw:
        if not isinstance(lane, dict):
            continue
        li = _safe_int(lane.get("index"), -1)
        if li < 0:
            continue
        for uid in (lane.get("exit_lanes") or []):
            si = uid_to_index.get(str(uid))
            if si is not None and si != li:
                successors.setdefault(li, set()).add(si)
                predecessors.setdefault(si, set()).add(li)
        for uid in (lane.get("entry_lanes") or []):
            pi = uid_to_index.get(str(uid))
            if pi is not None and pi != li:
                predecessors.setdefault(li, set()).add(pi)
                successors.setdefault(pi, set()).add(li)

    # Build spatial adjacency: same road_id, |lane_id difference| == 1
    road_lanes: Dict[int, List[Tuple[int, int]]] = {}  # road_id → [(lane_id, lane_index)]
    for li, lf in v2_feats.items():
        rid = int(lf.get("road_id", 0))
        lid = int(lf.get("lane_id", 0))
        road_lanes.setdefault(rid, []).append((lid, li))
    adjacency: Dict[int, set] = {}
    for rid, lanes in road_lanes.items():
        lid_to_idx = {lid: li for lid, li in lanes}
        for lid, li in lanes:
            for delta in (-1, 1):
                neighbor = lid_to_idx.get(lid + delta)
                if neighbor is not None and neighbor != li:
                    adjacency.setdefault(li, set()).add(neighbor)
                    adjacency.setdefault(neighbor, set()).add(li)

    return {
        "uid_to_index": uid_to_index,
        "successors": successors,
        "predecessors": predecessors,
        "adjacency": adjacency,
    }


def _carla_endpoint_proximity(
    carla_feats: Dict[int, Dict[str, object]],
    threshold_m: float = 3.0,
) -> Dict[int, set]:
    """For each CARLA line, find other CARLA lines whose start is close to
    this line's end (within *threshold_m* metres).  This approximates
    successor connectivity for unlabeled CARLA polylines."""
    endpoints: List[Tuple[int, float, float]] = []  # (line_index, end_x, end_y)
    startpoints: List[Tuple[int, float, float]] = []  # (line_index, start_x, start_y)
    for ci, cf in carla_feats.items():
        poly = cf["poly_xy"]
        endpoints.append((ci, float(poly[-1, 0]), float(poly[-1, 1])))
        startpoints.append((ci, float(poly[0, 0]), float(poly[0, 1])))

    if not startpoints:
        return {}

    start_xy = np.asarray([(s[1], s[2]) for s in startpoints], dtype=np.float64)
    start_idx = [s[0] for s in startpoints]
    start_tree = cKDTree(start_xy) if cKDTree is not None else None

    carla_successors: Dict[int, set] = {}
    thresh2 = threshold_m * threshold_m
    for ci, ex, ey in endpoints:
        if start_tree is not None:
            near = start_tree.query_ball_point(np.asarray([ex, ey], dtype=np.float64), r=threshold_m)
            for ni in near:
                cj = start_idx[ni]
                if cj != ci:
                    carla_successors.setdefault(ci, set()).add(cj)
        else:
            for cj, sx, sy in startpoints:
                if cj != ci and (ex - sx) ** 2 + (ey - sy) ** 2 <= thresh2:
                    carla_successors.setdefault(ci, set()).add(cj)
    return carla_successors


def _carla_lateral_adjacency(
    carla_feats: Dict[int, Dict[str, object]],
    threshold_m: float = 5.0,
) -> Dict[int, set]:
    """Detect pairs of CARLA lines that are roughly parallel and nearby
    (lateral adjacency).  Uses mid-point proximity + angle check."""
    mids: List[Tuple[int, float, float, float]] = []  # (ci, mx, my, heading_deg)
    for ci, cf in carla_feats.items():
        poly = cf["poly_xy"]
        mid = poly[poly.shape[0] // 2]
        dx = float(poly[-1, 0] - poly[0, 0])
        dy = float(poly[-1, 1] - poly[0, 1])
        heading = math.degrees(math.atan2(dy, dx)) if math.hypot(dx, dy) > 1e-6 else 0.0
        mids.append((ci, float(mid[0]), float(mid[1]), heading))

    if len(mids) < 2:
        return {}

    mid_xy = np.asarray([(m[1], m[2]) for m in mids], dtype=np.float64)
    mid_tree = cKDTree(mid_xy) if cKDTree is not None else None

    adjacency: Dict[int, set] = {}
    for i, (ci, mx, my, hi) in enumerate(mids):
        if mid_tree is not None:
            near = mid_tree.query_ball_point(np.asarray([mx, my], dtype=np.float64), r=threshold_m)
        else:
            near = [j for j in range(len(mids)) if math.hypot(mids[j][1] - mx, mids[j][2] - my) <= threshold_m]
        for j in near:
            if j == i:
                continue
            cj, _, _, hj = mids[j]
            angle_diff = abs(_normalize_yaw_deg(hi - hj))
            if angle_diff <= 30.0:  # roughly parallel (same direction)
                adjacency.setdefault(ci, set()).add(cj)
                adjacency.setdefault(cj, set()).add(ci)
    return adjacency


def _connectivity_consistency_penalty(
    assignment: Dict[int, int],  # v2_lane_index → carla_line_index
    v2_graph: Dict[str, object],
    carla_successors: Dict[int, set],
    weight: float = 0.5,
) -> float:
    """Compute a penalty for connectivity violations in the assignment.

    For each V2XPNP lane pair (A→B) that are connected by entry/exit_lanes,
    check whether their assigned CARLA lines are also connected (end-to-start
    proximity).  Returns a total penalty value.
    """
    v2_succs: Dict[int, set] = v2_graph.get("successors", {})
    penalty = 0.0
    n_checked = 0
    for li_a, ci_a in assignment.items():
        for li_b in v2_succs.get(li_a, set()):
            ci_b = assignment.get(li_b)
            if ci_b is None:
                continue
            n_checked += 1
            if ci_a == ci_b:
                continue  # same CARLA line — no penalty
            if ci_b in carla_successors.get(ci_a, set()):
                continue  # CARLA lines are connected — no penalty
            penalty += weight
    return penalty


def _adjacency_consistency_penalty(
    assignment: Dict[int, int],  # v2_lane_index → carla_line_index
    v2_graph: Dict[str, object],
    carla_adjacency: Dict[int, set],
    weight: float = 0.3,
) -> float:
    """Compute a penalty for adjacency violations.

    For V2XPNP lanes that share the same road_id with |Δlane_id|=1,
    their assigned CARLA lines should be laterally adjacent.
    """
    v2_adj: Dict[int, set] = v2_graph.get("adjacency", {})
    penalty = 0.0
    seen: set = set()
    for li_a, ci_a in assignment.items():
        for li_b in v2_adj.get(li_a, set()):
            pair = (min(li_a, li_b), max(li_a, li_b))
            if pair in seen:
                continue
            seen.add(pair)
            ci_b = assignment.get(li_b)
            if ci_b is None:
                continue
            if ci_a == ci_b:
                continue  # same line — could be a split, mild penalty
            if ci_b in carla_adjacency.get(ci_a, set()):
                continue  # properly adjacent
            penalty += weight
    return penalty


# ---------------------------------------------------------------------------
# Lane correspondence cache helpers
# ---------------------------------------------------------------------------

def _correspondence_cache_key(payload: Dict[str, object]) -> str:
    """Compute a deterministic hash key from the map lane polylines and
    CARLA line polylines so we can cache the correspondence result.
    Includes the map name so intersection and corridor sections get
    distinct caches."""
    h = hashlib.sha256()
    # Hash map identity (name + source_path) for intersection vs corridor distinction
    map_meta = payload.get("map", {})
    map_name = str(map_meta.get("name", "unknown"))
    map_src = str(map_meta.get("source_path", ""))
    h.update(f"map_name:{map_name}|src:{map_src}|".encode())
    # Hash V2XPNP lanes
    lanes = payload.get("map", {}).get("lanes", [])
    for lane in (lanes if isinstance(lanes, list) else []):
        if not isinstance(lane, dict):
            continue
        li = _safe_int(lane.get("index"), -1)
        h.update(f"v2:{li}:".encode())
        for p in (lane.get("polyline") or []):
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                h.update(f"{float(p[0]):.4f},{float(p[1]):.4f};".encode())
    # Hash CARLA lines
    carla_lines = (payload.get("carla_map") or {}).get("lines", [])
    for i, ln in enumerate(carla_lines if isinstance(carla_lines, list) else []):
        h.update(f"c:{i}:".encode())
        pts = ln.get("polyline") if isinstance(ln, dict) else ln
        for p in (pts if isinstance(pts, (list, tuple)) else []):
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                h.update(f"{float(p[0]):.4f},{float(p[1]):.4f};".encode())
    return h.hexdigest()[:24]


def _load_cached_correspondence(cache_dir: Optional[Path], cache_key: str, map_name: str = "") -> Optional[Dict[str, object]]:
    if cache_dir is None:
        return None
    # Sanitise map name for use in filename
    safe_name = "".join(c if (c.isalnum() or c in "_-") else "_" for c in map_name)[:60]
    # Try pickle first (new format with features), then JSON (legacy)
    pkl_fname = f"lane_corr_{safe_name}_{cache_key}.pkl" if safe_name else f"lane_corr_{cache_key}.pkl"
    pkl_file = cache_dir / pkl_fname
    if pkl_file.exists():
        try:
            with pkl_file.open("rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict) and data.get("cache_key") == cache_key:
                print(f"[INFO] Loaded cached lane correspondence from {pkl_file}")
                return data
        except Exception:
            pass
    # Fall back to JSON (legacy format, no features)
    json_fname = f"lane_corr_{safe_name}_{cache_key}.json" if safe_name else f"lane_corr_{cache_key}.json"
    cache_file = cache_dir / json_fname
    if not cache_file.exists():
        # Also check legacy filename without map name
        legacy = cache_dir / f"lane_corr_{cache_key}.json"
        if legacy.exists():
            cache_file = legacy
        else:
            return None
    if not cache_file.exists():
        return None
    try:
        with cache_file.open("r") as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("cache_key") == cache_key:
            print(f"[INFO] Loaded cached lane correspondence from {cache_file} (legacy JSON, no features)")
            return data
    except Exception:
        pass
    return None


def _save_cached_correspondence(
    cache_dir: Optional[Path],
    cache_key: str,
    lane_to_carla: Dict[int, Dict[str, object]],
    carla_to_lanes: Dict[int, List[int]],
    driving_lane_types: list,
    map_name: str = "",
    v2_feats: Optional[Dict[int, Dict[str, object]]] = None,
    carla_feats: Optional[Dict[int, Dict[str, object]]] = None,
) -> None:
    if cache_dir is None:
        return
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(c if (c.isalnum() or c in "_-") else "_" for c in map_name)[:60]
        # Save as pickle (includes features with numpy arrays)
        pkl_fname = f"lane_corr_{safe_name}_{cache_key}.pkl" if safe_name else f"lane_corr_{cache_key}.pkl"
        pkl_file = cache_dir / pkl_fname
        serialisable = {
            "cache_key": cache_key,
            "lane_to_carla": {int(k): v for k, v in lane_to_carla.items()},
            "carla_to_lanes": {int(k): v for k, v in carla_to_lanes.items()},
            "driving_lane_types": list(driving_lane_types),
            "v2_feats": v2_feats,
            "carla_feats": carla_feats,
        }
        with pkl_file.open("wb") as f:
            pickle.dump(serialisable, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[INFO] Saved lane correspondence cache to {pkl_file}")
    except Exception as e:
        print(f"[WARN] Failed to save lane correspondence cache: {e}")


# ---------------------------------------------------------------------------
# Main lane correspondence builder — global optimization
# ---------------------------------------------------------------------------

def _build_lane_correspondence(
    payload: Dict[str, object],
    candidate_top_k: int = 28,
    driving_lane_types: Sequence[str] = ("1",),
    cache_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Build V2XPNP-lane → CARLA-line correspondence using global
    optimisation (Hungarian / linear_sum_assignment) with multi-signal
    scoring: geometric distance, heading, connectivity consistency,
    adjacency consistency.

    Improvements over the previous greedy matcher:
    1. **Direction hard-filter** in _evaluate_lane_pair_metrics (mono < 0.45 → inf)
    2. **Global one-to-one assignment** via scipy linear_sum_assignment instead
       of greedy per-lane picking.
    3. **Connectivity consistency** — penalises assignments where connected
       V2XPNP lanes map to disconnected CARLA lines.
    4. **Adjacency consistency** — penalises assignments where spatially
       adjacent V2XPNP lanes map to non-adjacent CARLA lines.
    5. **Split/merge detection** — after global assignment, attempts to find
       secondary CARLA segments that cover gaps in partially-matched lanes.
    6. **Disk caching** — stores correspondence results keyed by content hash.
    """
    v2_lanes_raw = payload.get("map", {}).get("lanes", [])
    carla_map = payload.get("carla_map")
    if not isinstance(v2_lanes_raw, list) or not isinstance(carla_map, dict):
        return {"enabled": False, "reason": "missing_map_payload"}
    carla_lines_raw = carla_map.get("lines")
    if not isinstance(carla_lines_raw, list) or not carla_lines_raw:
        return {"enabled": False, "reason": "missing_carla_lines"}

    # --- Cache check ---
    _v2_map_name = str(payload.get("map", {}).get("name", ""))
    cache_key = _correspondence_cache_key(payload)
    cached = _load_cached_correspondence(cache_dir, cache_key, map_name=_v2_map_name)

    # --- If cache has features, use them directly (skip all phases) ---
    if cached is not None:
        cached_v2_feats = cached.get("v2_feats")
        cached_carla_feats = cached.get("carla_feats")
        if cached_v2_feats and cached_carla_feats:
            try:
                cached_l2c = cached.get("lane_to_carla", {})
                cached_c2l = cached.get("carla_to_lanes", {})
                lane_to_carla: Dict[int, Dict[str, object]] = {int(k): v for k, v in cached_l2c.items()}
                carla_to_lanes: Dict[int, List[int]] = {int(k): v for k, v in cached_c2l.items()}
                drive_types = {str(v) for v in driving_lane_types}
                print(f"  [CORR] Using fully cached correspondence (skipping all phases).")
                return {
                    "enabled": True,
                    "v2_feats": cached_v2_feats,
                    "carla_feats": cached_carla_feats,
                    "lane_candidates": {},
                    "lane_to_carla": lane_to_carla,
                    "carla_to_lanes": carla_to_lanes,
                    "driving_lane_types": sorted(drive_types),
                }
            except Exception:
                print("  [CORR] Cache corrupted, recomputing...")
                cached = None  # force recompute

    # ===== Phase 1: Feature extraction =====
    if cached is None:
        print(f"  [CORR] Phase 1/9: Feature extraction ({len(v2_lanes_raw)} v2 lanes, {len(carla_lines_raw)} CARLA lines)...")
    else:
        print(f"  [CORR] Phase 1/9: Feature extraction (legacy cache, rebuilding features)...")
    v2_feats: Dict[int, Dict[str, object]] = {}
    for lane in v2_lanes_raw:
        if not isinstance(lane, dict):
            continue
        li = _safe_int(lane.get("index"), -1)
        if li < 0:
            continue
        poly_xy = _polyline_xy_from_points(lane.get("polyline"))
        if poly_xy.shape[0] < 2:
            continue
        cum = _polyline_cumlen_xy(poly_xy)
        total_len = _polyline_total_len(cum)
        if total_len <= 1e-6:
            continue
        v2_feats[li] = {
            "lane_index": int(li),
            "road_id": _safe_int(lane.get("road_id"), 0),
            "lane_id": _safe_int(lane.get("lane_id"), 0),
            "lane_type": str(lane.get("lane_type", "")),
            "poly_xy": poly_xy,
            "cum": cum,
            "total_len": float(total_len),
            "label_x": _safe_float(lane.get("label_x"), float(poly_xy[poly_xy.shape[0] // 2, 0])),
            "label_y": _safe_float(lane.get("label_y"), float(poly_xy[poly_xy.shape[0] // 2, 1])),
            "entry_lanes": list(lane.get("entry_lanes") or []),
            "exit_lanes": list(lane.get("exit_lanes") or []),
        }

    carla_feats: Dict[int, Dict[str, object]] = {}
    for i, ln in enumerate(carla_lines_raw):
        if isinstance(ln, dict):
            poly_src = ln.get("polyline")
        else:
            poly_src = ln
        poly_xy = _polyline_xy_from_points(poly_src)
        if poly_xy.shape[0] < 2:
            continue
        cum = _polyline_cumlen_xy(poly_xy)
        total_len = _polyline_total_len(cum)
        if total_len <= 1e-6:
            continue
        poly_rev = poly_xy[::-1].copy()
        cum_rev = _polyline_cumlen_xy(poly_rev)
        label_mid = poly_xy[poly_xy.shape[0] // 2]
        carla_feats[i] = {
            "line_index": int(i),
            "poly_xy": poly_xy,
            "cum": cum,
            "total_len": float(total_len),
            "poly_xy_rev": poly_rev,
            "cum_rev": cum_rev,
            "label_x": float(label_mid[0]),
            "label_y": float(label_mid[1]),
        }

    if not v2_feats or not carla_feats:
        return {
            "enabled": False,
            "reason": "insufficient_lanes",
            "num_v2_lanes": int(len(v2_feats)),
            "num_carla_lines": int(len(carla_feats)),
        }

    # --- If we have a legacy cache (no features), use correspondence but we already rebuilt features ---
    if cached is not None:
        try:
            cached_l2c = cached.get("lane_to_carla", {})
            cached_c2l = cached.get("carla_to_lanes", {})
            lane_to_carla_cached: Dict[int, Dict[str, object]] = {int(k): v for k, v in cached_l2c.items()}
            carla_to_lanes_cached: Dict[int, List[int]] = {int(k): v for k, v in cached_c2l.items()}
            drive_types = {str(v) for v in driving_lane_types}
            print(f"  [CORR] Using cached correspondence (skipping phases 2-9).")
            # Re-save with features for next time
            _save_cached_correspondence(
                cache_dir, cache_key, lane_to_carla_cached, carla_to_lanes_cached,
                list(drive_types), _v2_map_name, v2_feats, carla_feats
            )
            return {
                "enabled": True,
                "v2_feats": v2_feats,
                "carla_feats": carla_feats,
                "lane_candidates": {},
                "lane_to_carla": lane_to_carla_cached,
                "carla_to_lanes": carla_to_lanes_cached,
                "driving_lane_types": sorted(drive_types),
            }
        except Exception:
            print("  [CORR] Cache corrupted, recomputing...")
            pass  # cache corrupted, recompute

    # ===== Phase 2: Spatial indexing (KD-tree of all CARLA vertices) =====
    print(f"  [CORR] Phase 2/9: Spatial indexing (v2_feats={len(v2_feats)}, carla_feats={len(carla_feats)})...")
    vx: List[Tuple[float, float]] = []
    vl: List[int] = []
    for ci, cf in carla_feats.items():
        poly_xy = cf["poly_xy"]
        for p in poly_xy:
            vx.append((float(p[0]), float(p[1])))
            vl.append(int(ci))
    vertex_xy = np.asarray(vx, dtype=np.float64) if vx else np.zeros((0, 2), dtype=np.float64)
    vertex_line = np.asarray(vl, dtype=np.int32) if vl else np.zeros((0,), dtype=np.int32)
    tree = cKDTree(vertex_xy) if (cKDTree is not None and vertex_xy.shape[0] > 0) else None

    def candidate_carla_lines(x: float, y: float, k: int) -> List[int]:
        if vertex_xy.shape[0] <= 0:
            return []
        kk = max(1, min(int(k), int(vertex_xy.shape[0])))
        if tree is not None:
            _, idxs = tree.query(np.asarray([float(x), float(y)], dtype=np.float64), k=kk)
            if np.isscalar(idxs):
                idx_list = [int(idxs)]
            else:
                idx_list = [int(v) for v in np.asarray(idxs).reshape(-1)]
        else:
            d2 = np.sum((vertex_xy - np.asarray([float(x), float(y)], dtype=np.float64)[None, :]) ** 2, axis=1)
            idx_list = [int(v) for v in np.argsort(d2)[:kk]]
        out: List[int] = []
        seen: set = set()
        for vi in idx_list:
            li = int(vertex_line[vi])
            if li in seen:
                continue
            seen.add(li)
            out.append(li)
        return out

    # ===== Phase 3: Candidate scoring =====
    print(f"  [CORR] Phase 3/9: Candidate scoring ({len(v2_feats)} lanes × top-{candidate_top_k})...")
    lane_candidates: Dict[int, List[Dict[str, object]]] = {}
    drive_types = {str(v) for v in driving_lane_types}
    for li, lf in v2_feats.items():
        poly = lf["poly_xy"]
        n = int(poly.shape[0])
        # Sample more points along the lane for better candidate discovery
        sample_indices = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
        sample_pts = [poly[min(idx, n - 1)] for idx in sample_indices]
        cand_idx: set = set()
        for p in sample_pts:
            cands = candidate_carla_lines(float(p[0]), float(p[1]), k=max(8, int(candidate_top_k)))
            for ci in cands:
                cand_idx.add(int(ci))
        if not cand_idx:
            continue

        cand_rows: List[Dict[str, object]] = []
        for ci in sorted(cand_idx):
            cf = carla_feats.get(int(ci))
            if cf is None:
                continue
            m_fwd = _evaluate_lane_pair_metrics(lf["poly_xy"], lf["cum"], cf["poly_xy"], cf["cum"])
            m_rev = _evaluate_lane_pair_metrics(lf["poly_xy"], lf["cum"], cf["poly_xy_rev"], cf["cum_rev"])
            reversed_used = bool(float(m_rev.get("score", float("inf"))) < float(m_fwd.get("score", float("inf"))))
            met = m_rev if reversed_used else m_fwd
            q = _quality_from_metrics(met)
            row = {
                "lane_index": int(li),
                "carla_line_index": int(ci),
                "reversed": bool(reversed_used),
                "quality": str(q),
            }
            row.update({k: float(v) for k, v in met.items()})
            cand_rows.append(row)
        cand_rows.sort(key=lambda r: float(r.get("score", float("inf"))))
        lane_candidates[int(li)] = cand_rows

    # ===== Phase 4: Build connectivity/adjacency graphs =====
    print(f"  [CORR] Phase 4/9: Connectivity & adjacency graphs...")
    v2_graph = _build_v2_connectivity_graph(v2_feats, v2_lanes_raw)
    carla_successors = _carla_endpoint_proximity(carla_feats, threshold_m=3.5)
    carla_adj = _carla_lateral_adjacency(carla_feats, threshold_m=5.0)

    # ===== Phase 5: Global one-to-one assignment (Hungarian algorithm) =====
    # We solve the assignment for driving lanes first (one-to-one exclusive),
    # then assign non-driving lanes (allow shared CARLA lines).
    driving_lanes_list = sorted(
        [li for li, lf in v2_feats.items() if str(lf.get("lane_type", "")) in drive_types],
    )
    non_driving_lanes_list = sorted(
        [li for li in v2_feats if li not in set(driving_lanes_list)],
    )
    print(f"  [CORR] Phase 5/9: Hungarian assignment ({len(driving_lanes_list)} driving + {len(non_driving_lanes_list)} non-driving lanes)...")

    # Collect all CARLA line candidates that appear for any driving lane
    all_carla_candidates: set = set()
    for li in driving_lanes_list:
        for row in lane_candidates.get(li, []):
            all_carla_candidates.add(int(row["carla_line_index"]))
    carla_cand_list = sorted(all_carla_candidates)

    lane_to_carla: Dict[int, Dict[str, object]] = {}

    if driving_lanes_list and carla_cand_list and _scipy_lsa is not None:
        # Build cost matrix: rows = driving V2 lanes, cols = CARLA candidates
        n_v2 = len(driving_lanes_list)
        n_carla = len(carla_cand_list)
        v2_idx_map = {li: i for i, li in enumerate(driving_lanes_list)}
        carla_idx_map = {ci: j for j, ci in enumerate(carla_cand_list)}

        # Large penalty for unmatched or impossible pairs
        BIG_COST = 100.0
        cost_matrix = np.full((n_v2, n_carla), BIG_COST, dtype=np.float64)

        # Best candidate row for each (v2_lane, carla_line) pair
        best_row_lookup: Dict[Tuple[int, int], Dict[str, object]] = {}

        for li in driving_lanes_list:
            i = v2_idx_map[li]
            for row in lane_candidates.get(li, []):
                ci = int(row["carla_line_index"])
                j = carla_idx_map.get(ci)
                if j is None:
                    continue
                sc = float(row.get("score", float("inf")))
                if sc < cost_matrix[i, j]:
                    cost_matrix[i, j] = sc
                    best_row_lookup[(li, ci)] = row

        # Run Hungarian algorithm
        # Handle rectangular matrices: more V2 lanes or more CARLA lines
        if n_v2 <= n_carla:
            row_ind, col_ind = _scipy_lsa(cost_matrix)
        else:
            # Transpose so we have fewer rows
            col_ind_t, row_ind_t = _scipy_lsa(cost_matrix.T)
            row_ind = row_ind_t
            col_ind = col_ind_t

        # Build initial assignment
        initial_assignment: Dict[int, int] = {}
        for r, c in zip(row_ind, col_ind):
            if r < n_v2 and c < n_carla:
                li = driving_lanes_list[r]
                ci = carla_cand_list[c]
                if cost_matrix[r, c] < BIG_COST:
                    initial_assignment[li] = ci

        # ===== Phase 5b: Connectivity/adjacency refinement =====
        print(f"  [CORR] Phase 5b/9: Connectivity refinement (initial={len(initial_assignment)} assignments)...")
        # Evaluate current assignment quality including structural consistency
        base_penalty = (
            _connectivity_consistency_penalty(initial_assignment, v2_graph, carla_successors, weight=0.5)
            + _adjacency_consistency_penalty(initial_assignment, v2_graph, carla_adj, weight=0.3)
        )

        # Attempt local swaps to reduce structural penalties
        improved = True
        assignment = dict(initial_assignment)
        max_swap_iters = min(50, n_v2 * n_v2)
        swap_iter = 0
        while improved and swap_iter < max_swap_iters:
            improved = False
            swap_iter += 1
            v2_assigned = list(assignment.keys())
            for idx_a in range(len(v2_assigned)):
                if improved:
                    break
                li_a = v2_assigned[idx_a]
                ci_a = assignment[li_a]
                # Try swapping with another assigned lane
                for idx_b in range(idx_a + 1, len(v2_assigned)):
                    li_b = v2_assigned[idx_b]
                    ci_b = assignment[li_b]
                    if ci_a == ci_b:
                        continue
                    # Check if swapped costs are not much worse
                    i_a, j_a = v2_idx_map[li_a], carla_idx_map[ci_a]
                    i_b, j_b = v2_idx_map[li_b], carla_idx_map[ci_b]
                    j_a_new, j_b_new = carla_idx_map[ci_b], carla_idx_map[ci_a]
                    old_cost = cost_matrix[i_a, j_a] + cost_matrix[i_b, j_b]
                    new_cost = cost_matrix[i_a, j_a_new] + cost_matrix[i_b, j_b_new]
                    # Only consider swaps where geometric cost increase is small
                    if new_cost > old_cost + 2.0:
                        continue
                    # Evaluate structural improvement
                    test = dict(assignment)
                    test[li_a] = ci_b
                    test[li_b] = ci_a
                    new_penalty = (
                        _connectivity_consistency_penalty(test, v2_graph, carla_successors, weight=0.5)
                        + _adjacency_consistency_penalty(test, v2_graph, carla_adj, weight=0.3)
                    )
                    cur_penalty = (
                        _connectivity_consistency_penalty(assignment, v2_graph, carla_successors, weight=0.5)
                        + _adjacency_consistency_penalty(assignment, v2_graph, carla_adj, weight=0.3)
                    )
                    total_improvement = (cur_penalty - new_penalty) - (new_cost - old_cost)
                    if total_improvement > 0.1:
                        assignment[li_a] = ci_b
                        assignment[li_b] = ci_a
                        improved = True
                        break
                # Also try swapping with an unassigned CARLA line
                if not improved:
                    used_carla_set = set(assignment.values())
                    for row in lane_candidates.get(li_a, []):
                        ci_new = int(row["carla_line_index"])
                        if ci_new in used_carla_set:
                            continue
                        j_new = carla_idx_map.get(ci_new)
                        if j_new is None:
                            continue
                        old_cost_a = cost_matrix[i_a, j_a]
                        new_cost_a = cost_matrix[i_a, j_new]
                        if new_cost_a >= BIG_COST:
                            continue
                        test = dict(assignment)
                        test[li_a] = ci_new
                        new_penalty = (
                            _connectivity_consistency_penalty(test, v2_graph, carla_successors, weight=0.5)
                            + _adjacency_consistency_penalty(test, v2_graph, carla_adj, weight=0.3)
                        )
                        cur_penalty = (
                            _connectivity_consistency_penalty(assignment, v2_graph, carla_successors, weight=0.5)
                            + _adjacency_consistency_penalty(assignment, v2_graph, carla_adj, weight=0.3)
                        )
                        total_improvement = (cur_penalty - new_penalty) - (new_cost_a - old_cost_a)
                        if total_improvement > 0.1:
                            assignment[li_a] = ci_new
                            improved = True
                            break

        # Convert assignment to lane_to_carla entries
        for li, ci in assignment.items():
            row = best_row_lookup.get((li, ci))
            if row is None:
                # Build metrics on the fly
                cf = carla_feats.get(ci)
                lf = v2_feats.get(li)
                if cf is None or lf is None:
                    continue
                m_fwd = _evaluate_lane_pair_metrics(lf["poly_xy"], lf["cum"], cf["poly_xy"], cf["cum"])
                m_rev = _evaluate_lane_pair_metrics(lf["poly_xy"], lf["cum"], cf["poly_xy_rev"], cf["cum_rev"])
                reversed_used = bool(float(m_rev.get("score", float("inf"))) < float(m_fwd.get("score", float("inf"))))
                met = m_rev if reversed_used else m_fwd
                q = _quality_from_metrics(met)
                row = {
                    "lane_index": int(li),
                    "carla_line_index": int(ci),
                    "reversed": bool(reversed_used),
                    "quality": str(q),
                }
                row.update({k: float(v) for k, v in met.items()})
            lane_to_carla[int(li)] = dict(row)

    elif driving_lanes_list:
        # Fallback: greedy assignment if scipy not available
        driving_lanes_sorted = sorted(
            driving_lanes_list,
            key=lambda li: float(lane_candidates.get(li, [{}])[0].get("score", float("inf"))),
        )
        used_carla: set = set()
        for li in driving_lanes_sorted:
            rows = lane_candidates.get(li, [])
            picked: Optional[Dict[str, object]] = None
            for r in rows:
                if int(r["carla_line_index"]) in used_carla:
                    continue
                if str(r.get("quality", "poor")) == "poor":
                    continue
                picked = r
                break
            if picked is None:
                for r in rows:
                    if int(r["carla_line_index"]) in used_carla:
                        continue
                    picked = r
                    break
            if picked is None and rows:
                picked = rows[0]
            if picked is None:
                continue
            lane_to_carla[int(li)] = dict(picked)
            used_carla.add(int(picked["carla_line_index"]))

    # ===== Phase 6: Non-driving lanes (allow shared CARLA lines) =====
    print(f"  [CORR] Phase 6/9: Non-driving lane assignment...")
    for li in non_driving_lanes_list:
        if li in lane_to_carla:
            continue
        rows = lane_candidates.get(li, [])
        if not rows:
            continue
        if str(rows[0].get("quality", "poor")) != "poor":
            lane_to_carla[int(li)] = dict(rows[0])

    # ===== Phase 7: Split detection =====
    print(f"  [CORR] Phase 7/9: Split detection ({len(lane_to_carla)} assigned so far)...")
    # For driving lanes with low coverage, try to find secondary CARLA
    # segments that cover the uncovered portion of the V2XPNP lane.
    used_primary_carla = set()
    for li, info in lane_to_carla.items():
        used_primary_carla.add(int(info.get("carla_line_index", -1)))

    split_merges: Dict[int, List[int]] = {}  # v2_lane → list of extra CARLA indices
    for li in driving_lanes_list:
        if li not in lane_to_carla:
            continue
        info = lane_to_carla[li]
        cov = float(info.get("coverage_2m", 1.0))
        if cov >= 0.75:
            continue  # good enough coverage, no split needed
        lf = v2_feats.get(li)
        if lf is None:
            continue
        primary_ci = int(info.get("carla_line_index", -1))
        primary_reversed = bool(info.get("reversed", False))
        pcf = carla_feats.get(primary_ci)
        if pcf is None:
            continue

        # Find which portion of the V2XPNP lane is uncovered
        v2_poly = lf["poly_xy"]
        v2_cum = lf["cum"]
        v2_len = float(lf["total_len"])
        n_probe = int(max(10, min(40, round(v2_len / 2.0))))
        s_probes = np.linspace(0.0, v2_len, n_probe, dtype=np.float64)

        c_poly = pcf["poly_xy_rev"] if primary_reversed else pcf["poly_xy"]
        c_cum = pcf["cum_rev"] if primary_reversed else pcf["cum"]

        uncovered_pts: List[Tuple[float, float]] = []
        for sv in s_probes.tolist():
            sp = _sample_polyline_at_s_xy(v2_poly, v2_cum, sv)
            if sp is None:
                continue
            proj = _project_point_to_polyline_xy(c_poly, c_cum, sp["x"], sp["y"])
            if proj is not None and proj["dist"] <= 2.0:
                continue
            uncovered_pts.append((sp["x"], sp["y"]))

        if not uncovered_pts:
            continue

        # Find candidate CARLA lines near uncovered points
        extra_cands: set = set()
        for px, py in uncovered_pts[:5]:  # sample a few
            cands = candidate_carla_lines(px, py, k=max(4, int(candidate_top_k) // 2))
            for ci in cands:
                if ci != primary_ci:
                    extra_cands.add(ci)

        best_extra_ci = -1
        best_extra_row: Optional[Dict[str, object]] = None
        for ci in sorted(extra_cands):
            cf = carla_feats.get(ci)
            if cf is None:
                continue
            # Evaluate how well this extra CARLA line covers the uncovered portion
            m_fwd = _evaluate_lane_pair_metrics(lf["poly_xy"], lf["cum"], cf["poly_xy"], cf["cum"])
            m_rev = _evaluate_lane_pair_metrics(lf["poly_xy"], lf["cum"], cf["poly_xy_rev"], cf["cum_rev"])
            rev_used = bool(float(m_rev.get("score", float("inf"))) < float(m_fwd.get("score", float("inf"))))
            met = m_rev if rev_used else m_fwd
            q = _quality_from_metrics(met)
            if q == "poor":
                continue
            if best_extra_row is None or float(met["score"]) < float(best_extra_row.get("score", float("inf"))):
                best_extra_ci = ci
                best_extra_row = {
                    "lane_index": int(li),
                    "carla_line_index": int(ci),
                    "reversed": bool(rev_used),
                    "quality": str(q),
                }
                best_extra_row.update({k: float(v) for k, v in met.items()})

        if best_extra_ci >= 0:
            split_merges.setdefault(li, []).append(best_extra_ci)

    # ===== Phase 8: Build reverse mapping =====
    print(f"  [CORR] Phase 8/9: Reverse mapping...")
    carla_to_lanes: Dict[int, List[int]] = {}
    for li, info in lane_to_carla.items():
        ci = int(info.get("carla_line_index", -1))
        if ci < 0:
            continue
        carla_to_lanes.setdefault(ci, []).append(int(li))
    # Include split/merge secondary mappings in the reverse map
    for li, extra_cis in split_merges.items():
        for ci in extra_cis:
            carla_to_lanes.setdefault(ci, []).append(int(li))

    for ci, lset in carla_to_lanes.items():
        lset_sorted = sorted(set(int(v) for v in lset))
        carla_to_lanes[ci] = lset_sorted
        for li in lset_sorted:
            if li in lane_to_carla:
                lane_to_carla[li]["shared_carla_line"] = bool(len(lset_sorted) > 1)

    # Add split_merges info to lane_to_carla entries
    for li, extra_cis in split_merges.items():
        if li in lane_to_carla:
            lane_to_carla[li]["split_extra_carla_lines"] = [int(c) for c in extra_cis]

    # ===== Phase 9: Cache result =====
    print(f"  [CORR] Phase 9/9: Caching result...")
    _save_cached_correspondence(
        cache_dir, cache_key, lane_to_carla, carla_to_lanes,
        sorted(drive_types), map_name=_v2_map_name,
        v2_feats=v2_feats, carla_feats=carla_feats
    )

    return {
        "enabled": True,
        "v2_feats": v2_feats,
        "carla_feats": carla_feats,
        "lane_candidates": lane_candidates,
        "lane_to_carla": lane_to_carla,
        "carla_to_lanes": carla_to_lanes,
        "driving_lane_types": sorted(drive_types),
        "split_merges": split_merges,
        "v2_graph": v2_graph,
        "carla_successors": carla_successors,
    }


def _carla_lines_connected(
    ci_a: int,
    ci_b: int,
    carla_feats: Dict[int, Dict[str, object]],
    carla_succs: Dict[int, set],
    threshold_m: float = 6.0,
) -> bool:
    """Check whether two CARLA lines are connected (endpoints within threshold)."""
    if ci_a == ci_b:
        return True
    if ci_b in carla_succs.get(ci_a, set()):
        return True
    if ci_a in carla_succs.get(ci_b, set()):
        return True
    cf_a = carla_feats.get(ci_a)
    cf_b = carla_feats.get(ci_b)
    if cf_a is None or cf_b is None:
        return False
    # Check all 4 endpoint combinations (handles reversed segments)
    pa_s, pa_e = cf_a["poly_xy"][0], cf_a["poly_xy"][-1]
    pb_s, pb_e = cf_b["poly_xy"][0], cf_b["poly_xy"][-1]
    for pa, pb in [(pa_e, pb_s), (pa_s, pb_s), (pa_e, pb_e), (pa_s, pb_e)]:
        if math.hypot(float(pa[0] - pb[0]), float(pa[1] - pb[1])) < threshold_m:
            return True
    return False


def _apply_lane_correspondence_to_payload(payload: Dict[str, object], correspondence: Dict[str, object]) -> None:
    if not bool(correspondence.get("enabled", False)):
        payload.setdefault("metadata", {})["lane_correspondence"] = {
            "enabled": False,
            "reason": str(correspondence.get("reason", "unknown")),
        }
        return

    v2_feats: Dict[int, Dict[str, object]] = dict(correspondence.get("v2_feats", {}))
    carla_feats: Dict[int, Dict[str, object]] = dict(correspondence.get("carla_feats", {}))
    lane_to_carla: Dict[int, Dict[str, object]] = dict(correspondence.get("lane_to_carla", {}))
    carla_to_lanes: Dict[int, List[int]] = dict(correspondence.get("carla_to_lanes", {}))

    lanes_raw = payload.get("map", {}).get("lanes", [])
    if isinstance(lanes_raw, list):
        for lane in lanes_raw:
            if not isinstance(lane, dict):
                continue
            li = _safe_int(lane.get("index"), -1)
            if li < 0:
                continue
            v2_label = f"r{_safe_int(lane.get('road_id'), 0)}_l{_safe_int(lane.get('lane_id'), 0)}_t{str(lane.get('lane_type', ''))}"
            lane["v2_label"] = v2_label
            m = lane_to_carla.get(int(li))
            if m is None:
                lane["carla_match"] = None
            else:
                q = str(m.get("quality", "poor"))
                split_extras = m.get("split_extra_carla_lines", [])
                lane["carla_match"] = {
                    "carla_line_index": int(m.get("carla_line_index", -1)),
                    "carla_line_label": f"c{int(m.get('carla_line_index', -1))}",
                    "reversed": bool(m.get("reversed", False)),
                    "quality": q,
                    "usable": bool(q != "poor"),
                    "score": float(m.get("score", float("inf"))),
                    "median_dist_m": float(m.get("median_dist_m", float("inf"))),
                    "p90_dist_m": float(m.get("p90_dist_m", float("inf"))),
                    "coverage_2m": float(m.get("coverage_2m", 0.0)),
                    "angle_median_deg": float(m.get("angle_median_deg", 180.0)),
                    "monotonic_ratio": float(m.get("monotonic_ratio", 0.0)),
                    "length_ratio": float(m.get("length_ratio", 0.0)),
                    "shared_carla_line": bool(m.get("shared_carla_line", False)),
                    "split_extra_carla_lines": [int(c) for c in split_extras] if split_extras else [],
                }

    carla_map = payload.get("carla_map")
    if isinstance(carla_map, dict):
        lines_raw = carla_map.get("lines", [])
        out_lines: List[Dict[str, object]] = []
        for i, ln in enumerate(lines_raw if isinstance(lines_raw, list) else []):
            poly_src = ln.get("polyline") if isinstance(ln, dict) else ln
            poly_xy = _polyline_xy_from_points(poly_src)
            if poly_xy.shape[0] < 2:
                continue
            mid = poly_xy[poly_xy.shape[0] // 2]
            matched_lanes = list(carla_to_lanes.get(int(i), []))
            matched_labels: List[str] = []
            for li in matched_lanes:
                lf = v2_feats.get(int(li))
                if lf is None:
                    matched_labels.append(f"lane_{int(li)}")
                else:
                    matched_labels.append(
                        f"r{int(lf.get('road_id', 0))}_l{int(lf.get('lane_id', 0))}_t{str(lf.get('lane_type', ''))}"
                    )
            best_quality = "none"
            for li in matched_lanes:
                m = lane_to_carla.get(int(li), {})
                q = str(m.get("quality", "poor"))
                if q == "high":
                    best_quality = "high"
                    break
                if q == "medium" and best_quality not in ("high",):
                    best_quality = "medium"
                if q == "low" and best_quality not in ("high", "medium"):
                    best_quality = "low"
                if q == "poor" and best_quality == "none":
                    best_quality = "poor"
            out_lines.append(
                {
                    "index": int(i),
                    "label": f"c{int(i)}",
                    "label_x": float(mid[0]),
                    "label_y": float(mid[1]),
                    "matched_v2_lane_indices": [int(v) for v in matched_lanes],
                    "matched_v2_labels": matched_labels,
                    "match_quality": best_quality,
                    "polyline": [[float(p[0]), float(p[1])] for p in poly_xy],
                }
            )
        carla_map["lines"] = out_lines
        carla_map["line_count"] = int(len(out_lines))

    all_tracks: List[Dict[str, object]] = []
    for key in ("ego_tracks", "actor_tracks"):
        tv = payload.get(key, [])
        if isinstance(tv, list):
            for tr in tv:
                if isinstance(tr, dict):
                    all_tracks.append(tr)

    print(f"[INFO] Applying lane correspondence projections to {len(all_tracks)} tracks...")
    _corr_apply_t0 = time.monotonic()

    # --- CARLA line connectivity for gap handling ---
    carla_succs: Dict[int, set] = correspondence.get("carla_successors", {})
    # Widen connectivity threshold for gap bridging (allow slightly larger gaps)
    _GAP_CONNECT_M = 6.0

    for tr in _progress_bar(
        all_tracks,
        total=len(all_tracks),
        desc="Lane corr projection",
        disable=len(all_tracks) < 8,
    ):
        frames = tr.get("frames", [])
        if not isinstance(frames, list):
            continue

        # =====================================================================
        # Pass 1: Raw per-frame CARLA mapping
        # =====================================================================
        raw_mappings: List[Dict[str, object]] = []
        for fr in frames:
            if not isinstance(fr, dict):
                raw_mappings.append({})
                continue
            mx = _safe_float(fr.get("mx"), _safe_float(fr.get("x"), 0.0))
            my = _safe_float(fr.get("my"), _safe_float(fr.get("y"), 0.0))
            myaw = _safe_float(fr.get("myaw"), _safe_float(fr.get("yaw"), 0.0))
            li = _safe_int(fr.get("li"), -1)

            cx = float(mx)
            cy = float(my)
            cyaw = float(myaw)
            cli = -1
            cld = float("inf")
            mapping_quality = "none"
            source_mode = "fallback_raw"

            mapped = lane_to_carla.get(int(li))
            if mapped is not None:
                mapped_quality = str(mapped.get("quality", "poor"))
                # Poor global matches can still be locally correct for short turn
                # segments; accept only when local projection is close.
                poor_local_max_dist = 3.0
                cidx = _safe_int(mapped.get("carla_line_index"), -1)
                cf = carla_feats.get(cidx)
                lf = v2_feats.get(int(li))
                if cf is not None:
                    cxy = cf["poly_xy_rev"] if bool(mapped.get("reversed", False)) else cf["poly_xy"]
                    ccum = cf["cum_rev"] if bool(mapped.get("reversed", False)) else cf["cum"]
                    ratio_accept_dist = 5.0 if mapped_quality != "poor" else poor_local_max_dist
                    ratio = None
                    if lf is not None:
                        proj_v2 = _project_point_to_polyline_xy(lf["poly_xy"], lf["cum"], float(mx), float(my))
                        if proj_v2 is not None:
                            ratio = float(proj_v2.get("s_norm", 0.0))
                    if ratio is not None:
                        samp = _sample_polyline_at_ratio_xy(cxy, ccum, float(ratio))
                        if samp is not None:
                            ratio_d = float(math.hypot(float(samp["x"]) - mx, float(samp["y"]) - my))
                            if ratio_d <= ratio_accept_dist:
                                cx = float(samp["x"])
                                cy = float(samp["y"])
                                cyaw = float(samp["yaw"])
                                cli = int(cidx)
                                cld = ratio_d
                                mapping_quality = mapped_quality
                                source_mode = "lane_correspondence"
                    # Fallback: direct projection onto the CARLA polyline
                    if cli < 0:
                        proj_c = _project_point_to_polyline_xy(cxy, ccum, float(mx), float(my))
                        if proj_c is not None:
                            proj_dist = float(proj_c["dist"])
                            if mapped_quality != "poor" or proj_dist <= poor_local_max_dist:
                                cx = float(proj_c["x"])
                                cy = float(proj_c["y"])
                                cyaw = float(proj_c["yaw"])
                                cli = int(cidx)
                                cld = proj_dist
                                mapping_quality = mapped_quality
                                source_mode = "lane_projection"

            if cli < 0:
                best_ci = -1
                best_proj = None
                for cidx, cf in carla_feats.items():
                    proj = _project_point_to_polyline_xy(cf["poly_xy"], cf["cum"], float(mx), float(my))
                    if proj is None:
                        continue
                    if best_proj is None or float(proj["dist"]) < float(best_proj["dist"]):
                        best_proj = proj
                        best_ci = int(cidx)
                if best_proj is not None and best_ci >= 0:
                    cx = float(best_proj["x"])
                    cy = float(best_proj["y"])
                    cyaw = float(best_proj["yaw"])
                    cli = int(best_ci)
                    cld = float(best_proj["dist"])
                    mapping_quality = "nearest"
                    source_mode = "nearest_carla_line"

            raw_mappings.append({
                "cx": float(cx),
                "cy": float(cy),
                "cyaw": float(cyaw),
                "cli": int(cli),
                "cld": float(cld),
                "csource": str(source_mode),
                "cquality": str(mapping_quality),
            })

        if not raw_mappings:
            continue

        # =====================================================================
        # Pass 2: Gap bridging — trajectory-level handling of unmatched regions
        # =====================================================================
        # Identify "anchored" runs (lane_correspondence / lane_projection)
        # and "gap" runs (nearest_carla_line / fallback_raw).
        # For gap runs:
        #   - If the raw cli sequence forms a CONNECTED CHAIN of CARLA segments,
        #     treat it as a natural road traversal (keep raw cli, use actor's
        #     own trajectory for cx,cy to avoid snapping to segment endpoints).
        #   - If the cli sequence is DISCONNECTED (intersection bouncing),
        #     use the actor's own (mx,my) and assign cli from nearest anchor.

        def _frame_is_anchored(fi: int) -> bool:
            if fi < 0 or fi >= len(raw_mappings):
                return False
            rm = raw_mappings[fi]
            return bool(rm) and str(rm.get("csource", "")) in ("lane_correspondence", "lane_projection")

        # Build runs of anchored / gap frames
        frame_runs: List[Tuple[int, int, bool]] = []  # (start, end, is_anchored)
        if raw_mappings:
            cur_anch = _frame_is_anchored(0)
            run_start = 0
            for fi in range(1, len(raw_mappings)):
                anch = _frame_is_anchored(fi)
                if anch != cur_anch:
                    frame_runs.append((run_start, fi - 1, cur_anch))
                    run_start = fi
                    cur_anch = anch
            frame_runs.append((run_start, len(raw_mappings) - 1, cur_anch))

        for run_idx, (rs, re, is_anch) in enumerate(frame_runs):
            if is_anch:
                continue  # already well-mapped
            gap_len = re - rs + 1
            if gap_len < 1:
                continue

            # --- Build RLE of raw cli values in this gap ---
            gap_cli_rle: List[Tuple[int, int, int]] = []  # (cli, start_fi, end_fi)
            prev_gap_cli = -999
            for fi in range(rs, re + 1):
                rm = raw_mappings[fi]
                if not rm:
                    continue
                cli_val = int(rm.get("cli", -1))
                if cli_val != prev_gap_cli:
                    gap_cli_rle.append((cli_val, fi, fi))
                    prev_gap_cli = cli_val
                else:
                    gap_cli_rle[-1] = (gap_cli_rle[-1][0], gap_cli_rle[-1][1], fi)

            # --- Check if consecutive cli values form a connected chain ---
            is_connected_chain = True
            if len(gap_cli_rle) > 1:
                for ri in range(1, len(gap_cli_rle)):
                    ci_prev = gap_cli_rle[ri - 1][0]
                    ci_curr = gap_cli_rle[ri][0]
                    if ci_prev < 0 or ci_curr < 0:
                        is_connected_chain = False
                        break
                    if not _carla_lines_connected(ci_prev, ci_curr, carla_feats, carla_succs, _GAP_CONNECT_M):
                        is_connected_chain = False
                        break
            elif len(gap_cli_rle) == 1:
                is_connected_chain = True  # single segment, trivially connected

            # --- Find nearest boundary anchors ---
            prev_anchor_fi = -1
            next_anchor_fi = -1
            for ri in range(run_idx - 1, -1, -1):
                if frame_runs[ri][2]:  # is_anchored
                    prev_anchor_fi = frame_runs[ri][1]  # last frame of prev anchor
                    break
            for ri in range(run_idx + 1, len(frame_runs)):
                if frame_runs[ri][2]:
                    next_anchor_fi = frame_runs[ri][0]  # first frame of next anchor
                    break

            prev_anchor_cli = int(raw_mappings[prev_anchor_fi].get("cli", -1)) if prev_anchor_fi >= 0 else -1
            next_anchor_cli = int(raw_mappings[next_anchor_fi].get("cli", -1)) if next_anchor_fi >= 0 else -1

            if is_connected_chain and len(gap_cli_rle) <= 8:
                # ----- CONNECTED CHAIN: natural road traversal -----
                # Keep the raw cli but use actor's own (mx,my) — no snap.
                for fi in range(rs, re + 1):
                    fr = frames[fi]
                    if not isinstance(fr, dict) or not raw_mappings[fi]:
                        continue
                    fmx = _safe_float(fr.get("mx"), _safe_float(fr.get("x"), 0.0))
                    fmy = _safe_float(fr.get("my"), _safe_float(fr.get("y"), 0.0))
                    fmyaw = _safe_float(fr.get("myaw"), _safe_float(fr.get("yaw"), 0.0))
                    raw_mappings[fi]["cx"] = float(fmx)
                    raw_mappings[fi]["cy"] = float(fmy)
                    raw_mappings[fi]["cyaw"] = float(fmyaw)
                    raw_mappings[fi]["cld"] = 0.0
                    raw_mappings[fi]["csource"] = "gap_chain"
                    raw_mappings[fi]["cquality"] = "chain"
            else:
                # ----- DISCONNECTED GAP: intersection or scattered segments -----
                # Use actor's own (mx,my) — no snap.  Assign cli from nearest
                # boundary anchor so the lane-id is reasonable but cx,cy tracks
                # the actor's actual position.
                anchor_cli = prev_anchor_cli if prev_anchor_cli >= 0 else next_anchor_cli
                for fi in range(rs, re + 1):
                    fr = frames[fi]
                    if not isinstance(fr, dict) or not raw_mappings[fi]:
                        continue
                    fmx = _safe_float(fr.get("mx"), _safe_float(fr.get("x"), 0.0))
                    fmy = _safe_float(fr.get("my"), _safe_float(fr.get("y"), 0.0))
                    fmyaw = _safe_float(fr.get("myaw"), _safe_float(fr.get("yaw"), 0.0))
                    raw_mappings[fi]["cx"] = float(fmx)
                    raw_mappings[fi]["cy"] = float(fmy)
                    raw_mappings[fi]["cyaw"] = float(fmyaw)
                    raw_mappings[fi]["cli"] = int(anchor_cli) if anchor_cli >= 0 else int(raw_mappings[fi].get("cli", -1))
                    raw_mappings[fi]["cld"] = 0.0
                    raw_mappings[fi]["csource"] = "gap_bridge"
                    raw_mappings[fi]["cquality"] = "bridge"

        # =====================================================================
        # Pass 2.5: Snap gap frames to nearest CARLA line
        # =====================================================================
        # Gap frames currently use raw (mx,my).  In many cases they are
        # inside an intersection where a CARLA *junction lane* (turn
        # polyline) exists nearby.  Projecting onto the nearest CARLA
        # line produces a much smoother visual result.
        _GAP_SNAP_DIST = 8.0  # max allowed projection distance for gap snap
        for fi in range(len(raw_mappings)):
            rm = raw_mappings[fi]
            if not rm or fi >= len(frames) or not isinstance(frames[fi], dict):
                continue
            src = str(rm.get("csource", ""))
            if src not in ("gap_chain", "gap_bridge"):
                continue
            fmx = _safe_float(frames[fi].get("mx"), _safe_float(frames[fi].get("x"), 0.0))
            fmy = _safe_float(frames[fi].get("my"), _safe_float(frames[fi].get("y"), 0.0))
            fmyaw = _safe_float(frames[fi].get("myaw"), _safe_float(frames[fi].get("yaw"), 0.0))
            # Find nearest CARLA line
            best_ci = -1
            best_proj = None
            for cidx_s, cf_s in carla_feats.items():
                proj_s = _project_point_to_polyline_xy(
                    cf_s["poly_xy"], cf_s["cum"], float(fmx), float(fmy)
                )
                if proj_s is None:
                    continue
                if best_proj is None or float(proj_s["dist"]) < float(best_proj["dist"]):
                    best_proj = proj_s
                    best_ci = int(cidx_s)
            if best_proj is not None and float(best_proj["dist"]) <= _GAP_SNAP_DIST:
                rm["cx"] = float(best_proj["x"])
                rm["cy"] = float(best_proj["y"])
                rm["cyaw"] = float(best_proj["yaw"])
                rm["cli"] = int(best_ci)
                rm["cld"] = float(best_proj["dist"])
                rm["csource"] = "gap_snap"
            # else: leave as raw (mx,my)

        # =====================================================================
        # Pass 3: Trajectory-level cli sequence smoothing
        # =====================================================================
        # Build run-length encoding of cli across ALL frames and remove very
        # short runs by merging into the adjacent longer run.
        MIN_CLI_RUN = 5  # minimum frames for a cli run to survive
        cli_seq = [int(rm.get("cli", -1)) if rm else -1 for rm in raw_mappings]
        cli_rle: List[List] = []  # [[cli, start, end]]
        for fi, cli_val in enumerate(cli_seq):
            if not cli_rle or cli_val != cli_rle[-1][0]:
                cli_rle.append([cli_val, fi, fi])
            else:
                cli_rle[-1][2] = fi

        # Remove short runs by merging into neighbours
        changed = True
        while changed:
            changed = False
            new_rle: List[List] = []
            for entry in cli_rle:
                cli_val, es, ee = entry
                run_len = ee - es + 1
                if run_len < MIN_CLI_RUN and len(new_rle) > 0:
                    # Merge into previous run
                    new_rle[-1][2] = ee
                    changed = True
                else:
                    new_rle.append(entry)
            cli_rle = new_rle
            # Also merge trailing short runs
            if len(cli_rle) >= 2:
                last = cli_rle[-1]
                if last[2] - last[1] + 1 < MIN_CLI_RUN:
                    cli_rle[-2][2] = last[2]
                    cli_rle.pop()
                    changed = True

        # Apply smoothed cli and re-project affected frames
        MAX_SNAP_DIST = 8.0  # safety cap: if projection > this, use mx,my
        for entry in cli_rle:
            cli_val, es, ee = entry
            for fi in range(es, ee + 1):
                rm = raw_mappings[fi]
                if not rm:
                    continue
                old_cli = int(rm.get("cli", -1))
                if old_cli != cli_val and cli_val >= 0:
                    rm["cli"] = int(cli_val)
                    # Skip re-projection for raw gap frames (they use mx,my)
                    src = str(rm.get("csource", ""))
                    if src in ("gap_chain", "gap_bridge"):
                        continue
                    # Re-project onto the new cli polyline
                    cf = carla_feats.get(cli_val)
                    if cf is not None and fi < len(frames) and isinstance(frames[fi], dict):
                        fmx = _safe_float(frames[fi].get("mx"), _safe_float(frames[fi].get("x"), 0.0))
                        fmy = _safe_float(frames[fi].get("my"), _safe_float(frames[fi].get("y"), 0.0))
                        proj = _project_point_to_polyline_xy(cf["poly_xy"], cf["cum"], fmx, fmy)
                        if proj is not None and float(proj["dist"]) <= MAX_SNAP_DIST:
                            rm["cx"] = float(proj["x"])
                            rm["cy"] = float(proj["y"])
                            rm["cyaw"] = float(proj["yaw"])
                            rm["cld"] = float(proj["dist"])
                            rm["csource"] = "smoothed" if src != "gap_snap" else "gap_snap"
                        else:
                            # Past end of polyline — fall back to mx,my
                            fmyaw = _safe_float(frames[fi].get("myaw"), _safe_float(frames[fi].get("yaw"), 0.0))
                            rm["cx"] = float(fmx)
                            rm["cy"] = float(fmy)
                            rm["cyaw"] = float(fmyaw)
                            rm["cld"] = 0.0
                            rm["csource"] = "smoothed_fallback"

        # =====================================================================
        # Pass 4: A→B→A lane-change suppression
        # =====================================================================
        # Detect patterns where cli goes A→B→A and B is short.  These are
        # false lane changes — collapse B into A.
        # Rebuild cli_rle from the (possibly smoothed) raw_mappings.
        cli_seq2 = [int(rm.get("cli", -1)) if rm else -1 for rm in raw_mappings]
        cli_rle2: List[List] = []
        for fi, cv in enumerate(cli_seq2):
            if not cli_rle2 or cv != cli_rle2[-1][0]:
                cli_rle2.append([cv, fi, fi])
            else:
                cli_rle2[-1][2] = fi

        MAX_ABA_FRAMES = 15  # B run must be <= this to be collapsed
        aba_changed = True
        while aba_changed:
            aba_changed = False
            for ri in range(1, len(cli_rle2) - 1):
                a_cli = cli_rle2[ri - 1][0]
                b_cli = cli_rle2[ri][0]
                c_cli = cli_rle2[ri + 1][0]
                b_len = cli_rle2[ri][2] - cli_rle2[ri][1] + 1
                if a_cli == c_cli and a_cli >= 0 and b_cli != a_cli and b_len <= MAX_ABA_FRAMES:
                    # Collapse B into A
                    cli_rle2[ri - 1][2] = cli_rle2[ri + 1][2]
                    cli_rle2.pop(ri + 1)
                    cli_rle2.pop(ri)
                    aba_changed = True
                    break  # restart scan after modification

        # Apply A→B→A suppression and re-project collapsed frames
        for entry in cli_rle2:
            cli_val, es, ee = entry
            for fi in range(es, ee + 1):
                rm = raw_mappings[fi]
                if not rm:
                    continue
                old_cli = int(rm.get("cli", -1))
                if old_cli != cli_val and cli_val >= 0:
                    rm["cli"] = int(cli_val)
                    src = str(rm.get("csource", ""))
                    if src in ("gap_chain", "gap_bridge"):
                        continue
                    cf = carla_feats.get(cli_val)
                    if cf is not None and fi < len(frames) and isinstance(frames[fi], dict):
                        fmx = _safe_float(frames[fi].get("mx"), _safe_float(frames[fi].get("x"), 0.0))
                        fmy = _safe_float(frames[fi].get("my"), _safe_float(frames[fi].get("y"), 0.0))
                        proj = _project_point_to_polyline_xy(cf["poly_xy"], cf["cum"], fmx, fmy)
                        if proj is not None and float(proj["dist"]) <= MAX_SNAP_DIST:
                            rm["cx"] = float(proj["x"])
                            rm["cy"] = float(proj["y"])
                            rm["cyaw"] = float(proj["yaw"])
                            rm["cld"] = float(proj["dist"])
                            rm["csource"] = "aba_suppress" if src != "gap_snap" else "gap_snap"
                        else:
                            fmyaw = _safe_float(frames[fi].get("myaw"), _safe_float(frames[fi].get("yaw"), 0.0))
                            rm["cx"] = float(fmx)
                            rm["cy"] = float(fmy)
                            rm["cyaw"] = float(fmyaw)
                            rm["cld"] = 0.0
                            rm["csource"] = "aba_fallback"

        # =====================================================================
        # Pass 5: Final safety cap — prevent any remaining huge gaps
        # =====================================================================
        for fi in range(len(raw_mappings)):
            rm = raw_mappings[fi]
            if not rm or fi >= len(frames) or not isinstance(frames[fi], dict):
                continue
            cld_val = float(rm.get("cld", 0.0))
            if cld_val > MAX_SNAP_DIST:
                fmx = _safe_float(frames[fi].get("mx"), _safe_float(frames[fi].get("x"), 0.0))
                fmy = _safe_float(frames[fi].get("my"), _safe_float(frames[fi].get("y"), 0.0))
                fmyaw = _safe_float(frames[fi].get("myaw"), _safe_float(frames[fi].get("yaw"), 0.0))
                # Try nearest CARLA line as last resort
                best_ci = -1
                best_proj = None
                for cidx_s, cf_s in carla_feats.items():
                    proj_s = _project_point_to_polyline_xy(cf_s["poly_xy"], cf_s["cum"], float(fmx), float(fmy))
                    if proj_s is None:
                        continue
                    if best_proj is None or float(proj_s["dist"]) < float(best_proj["dist"]):
                        best_proj = proj_s
                        best_ci = int(cidx_s)
                if best_proj is not None and float(best_proj["dist"]) <= MAX_SNAP_DIST:
                    rm["cx"] = float(best_proj["x"])
                    rm["cy"] = float(best_proj["y"])
                    rm["cyaw"] = float(best_proj["yaw"])
                    rm["cli"] = int(best_ci)
                    rm["cld"] = float(best_proj["dist"])
                    rm["csource"] = "safety_nearest"
                else:
                    rm["cx"] = float(fmx)
                    rm["cy"] = float(fmy)
                    rm["cyaw"] = float(fmyaw)
                    rm["cld"] = 0.0
                    rm["csource"] = "safety_fallback"

        # =====================================================================
        # Write final mappings to frames and count lane changes
        # =====================================================================
        phantom_changes = 0
        prev_cli = -1
        for fi, fr in enumerate(frames):
            if not isinstance(fr, dict) or fi >= len(raw_mappings):
                continue
            rm = raw_mappings[fi]
            if not rm:
                continue
            fr["cx"] = float(rm.get("cx", 0.0))
            fr["cy"] = float(rm.get("cy", 0.0))
            fr["cyaw"] = float(rm.get("cyaw", 0.0))
            fr["cli"] = int(rm.get("cli", -1))
            fr["cld"] = float(rm.get("cld", float("inf")))
            fr["csource"] = str(rm.get("csource", "fallback_raw"))
            fr["cquality"] = str(rm.get("cquality", "none"))
            cur_cli = int(rm.get("cli", -1))
            # Only count transitions between DIFFERENT cli values
            # in anchored regions as phantom changes
            src = str(rm.get("csource", ""))
            if prev_cli >= 0 and cur_cli >= 0 and cur_cli != prev_cli:
                if src in ("lane_correspondence", "lane_projection", "smoothed"):
                    phantom_changes += 1
            prev_cli = cur_cli

        tr["carla_lane_changes"] = int(phantom_changes)

    qual_counts: Dict[str, int] = {}
    for m in lane_to_carla.values():
        q = str(m.get("quality", "poor"))
        qual_counts[q] = qual_counts.get(q, 0) + 1
    usable_lane_count = int(sum(int(v) for q, v in qual_counts.items() if str(q) != "poor"))

    # Compute validation diagnostics
    total_phantom_changes = sum(
        int(tr.get("carla_lane_changes", 0))
        for tr in all_tracks
        if isinstance(tr, dict)
    )
    split_merge_count = len(correspondence.get("split_merges", {}))

    # Connectivity validation: count how many V2XPNP connectivity edges
    # are preserved (matched lanes → connected CARLA lines)
    v2_graph = correspondence.get("v2_graph", {})
    v2_succs: Dict[int, set] = v2_graph.get("successors", {}) if isinstance(v2_graph, dict) else {}
    conn_total = 0
    conn_preserved = 0
    for li_a, succs in v2_succs.items():
        for li_b in succs:
            ci_a_info = lane_to_carla.get(int(li_a))
            ci_b_info = lane_to_carla.get(int(li_b))
            if ci_a_info is None or ci_b_info is None:
                continue
            conn_total += 1
            ci_a = int(ci_a_info.get("carla_line_index", -1))
            ci_b = int(ci_b_info.get("carla_line_index", -1))
            if ci_a == ci_b:
                conn_preserved += 1
            elif ci_b in (correspondence.get("v2_graph", {}) or {}).get("successors", {}).get(ci_a, set()):
                conn_preserved += 1

    payload.setdefault("metadata", {})["lane_correspondence"] = {
        "enabled": True,
        "driving_lane_types": list(correspondence.get("driving_lane_types", [])),
        "mapped_lane_count": int(len(lane_to_carla)),
        "usable_lane_count": int(usable_lane_count),
        "mapped_carla_line_count": int(len(carla_to_lanes)),
        "quality_counts": qual_counts,
        "split_merge_count": int(split_merge_count),
        "total_phantom_lane_changes": int(total_phantom_changes),
        "connectivity_edges_total": int(conn_total),
        "connectivity_edges_preserved": int(conn_preserved),
    }
    _corr_apply_dt = time.monotonic() - _corr_apply_t0
    print(f"[INFO] Lane correspondence projection pass complete in {_corr_apply_dt:.2f}s")

def _build_export_payload(
    scenario_dir: Path,
    selected_map: VectorMapData,
    selection_details: Sequence[Dict[str, object]],
    ego_trajs: Sequence[Sequence[Waypoint]],
    ego_times: Sequence[Sequence[float]],
    vehicles: Dict[int, Sequence[Waypoint]],
    vehicle_times: Dict[int, Sequence[float]],
    obj_info: Dict[int, Dict[str, object]],
    actor_source_subdir: Dict[int, str],
    actor_orig_vid: Dict[int, int],
    actor_alias_vids: Dict[int, List[int]],
    merge_stats: Dict[str, object],
    timing_optimization: Dict[str, object],
    matcher: LaneMatcher,
    snap_to_map: bool,
    map_max_points_per_line: int,
    lane_change_cfg: Optional[Dict[str, object]] = None,
    vehicle_lane_policy_cfg: Optional[Dict[str, object]] = None,
    parked_vehicle_cfg: Optional[Dict[str, float]] = None,
    carla_map_layer: Optional[Dict[str, object]] = None,
    default_dt: float = 0.1,
) -> Dict[str, object]:
    print(f"[INFO] Building ego track frames ({len(ego_trajs)} ego trajectories)...")
    ego_tracks: List[Dict[str, object]] = []
    for ego_idx, traj in enumerate(ego_trajs):
        times = ego_times[ego_idx] if ego_idx < len(ego_times) else []
        frames = _build_track_frames(
            traj,
            times,
            matcher,
            snap_to_map=snap_to_map,
            lane_change_cfg=lane_change_cfg,  # apply same map-lane stabilization to ego display
        )
        ego_tracks.append(
            {
                "id": f"ego_{ego_idx}",
                "name": f"ego_{ego_idx}",
                "role": "ego",
                "obj_type": "ego",
                "model": "ego",
                "source_subdir": str(ego_idx),
                "path_length_m": float(_trajectory_path_length(traj)),
                "frames": frames,
            }
        )

    print(f"[INFO] Building actor track frames ({len(vehicles)} actors)...")
    actor_tracks: List[Dict[str, object]] = []
    skipped_non_actors = 0
    map_lane_type_universe = sorted({str(l.lane_type) for l in selected_map.lanes})
    veh_policy_cfg = dict(vehicle_lane_policy_cfg or {})
    
    # Extract early spawn timing info for visualization
    early_spawn_info = timing_optimization.get("early_spawn", {})
    adjusted_spawn_times: Dict[str, float] = {}
    if early_spawn_info.get("applied"):
        raw_spawn_times = early_spawn_info.get("adjusted_spawn_times", {})
        for vid_str, spawn_t in raw_spawn_times.items():
            adjusted_spawn_times[str(vid_str)] = float(spawn_t)
    parked_cfg = dict(parked_vehicle_cfg or {})
    forbidden_lane_types = _parse_lane_type_set(veh_policy_cfg.get("forbidden_lane_types"), fallback=["2"])
    parked_only_lane_types = _parse_lane_type_set(veh_policy_cfg.get("parked_only_lane_types"), fallback=["3"])
    _actor_vid_list = sorted(vehicles.keys())
    for _actor_i, vid in enumerate(_progress_bar(_actor_vid_list, desc="Actor track frames", disable=len(_actor_vid_list) < 10)):
        traj = vehicles[vid]
        if not traj:
            continue
        meta = dict(obj_info.get(vid, {}))
        obj_type = str(meta.get("obj_type") or "npc")
        if not is_vehicle_type(obj_type):
            skipped_non_actors += 1
            continue

        model = str(meta.get("model") or map_obj_type(obj_type))
        role = _infer_actor_role(obj_type, traj)
        times = vehicle_times.get(vid, [])
        is_ped_or_cyclist = (str(role).lower() in {"walker", "cyclist"})
        is_motor_vehicle = str(role).lower() in {"vehicle", "npc", "static"}
        is_parked_vehicle = False
        parked_stats: Dict[str, float] = {}
        lane_policy: Optional[Dict[str, object]] = None
        if is_motor_vehicle:
            is_parked_vehicle, parked_stats = _is_parked_vehicle(
                traj=traj,
                times=times,
                default_dt=float(default_dt),
                cfg=parked_cfg,
            )
            allowed_types = set(map_lane_type_universe)
            allowed_types -= forbidden_lane_types
            if not is_parked_vehicle:
                allowed_types -= parked_only_lane_types
            stationary_types = parked_only_lane_types.intersection(allowed_types) if is_parked_vehicle else set()
            lane_policy = {
                "allowed_lane_types": sorted(str(v) for v in allowed_types),
                "stationary_when_lane_types": sorted(str(v) for v in stationary_types),
            }
        frames = _build_track_frames(
            traj,
            times,
            matcher,
            snap_to_map=snap_to_map,
            lane_change_cfg=lane_change_cfg,
            lane_policy=lane_policy,
            skip_snap=is_ped_or_cyclist,
        )
        # Compute spawn timing info
        first_observed_time = float(times[0]) if times else 0.0
        early_spawn_time = adjusted_spawn_times.get(str(vid), None)
        has_early_spawn = early_spawn_time is not None and early_spawn_time < first_observed_time - 1e-6
        
        actor_tracks.append(
            {
                "id": f"actor_{vid}",
                "vid": int(vid),
                "orig_vid": int(actor_orig_vid.get(vid, vid)),
                "merged_vids": [int(v) for v in actor_alias_vids.get(vid, [vid])],
                "name": f"actor_{vid}",
                "role": role,
                "obj_type": obj_type,
                "model": model,
                "length": _safe_float(meta.get("length"), 0.0),
                "width": _safe_float(meta.get("width"), 0.0),
                "source_subdir": str(actor_source_subdir.get(vid, "")),
                "path_length_m": float(_trajectory_path_length(traj)),
                "parked_vehicle": bool(is_parked_vehicle),
                "parked_motion_stats": parked_stats,
                "lane_snap_policy": lane_policy or {},
                "frames": frames,
                # Spawn timing visualization fields
                "first_observed_time": float(first_observed_time),
                "early_spawn_time": float(early_spawn_time) if early_spawn_time is not None else None,
                "has_early_spawn": bool(has_early_spawn),
            }
        )

    # --- Ego-vs-actor deduplication ---
    # Remove actor tracks that are duplicates of ego tracks (same physical
    # vehicle seen by both ego sensor and actor sensor).
    ego_actor_dedup_removed: List[str] = []
    for ego_tr in ego_tracks:
        ego_frames = ego_tr.get("frames", [])
        if not ego_frames:
            continue
        ego_xy = [(float(f.get("mx", f.get("x", 0))), float(f.get("my", f.get("y", 0)))) for f in ego_frames]
        remaining_actors: List[Dict[str, object]] = []
        for act_tr in actor_tracks:
            act_frames = act_tr.get("frames", [])
            if not act_frames:
                remaining_actors.append(act_tr)
                continue
            # Compare common frame range
            min_len = min(len(ego_xy), len(act_frames))
            if min_len < 5:
                remaining_actors.append(act_tr)
                continue
            dists: List[float] = []
            for fi in range(min_len):
                ex, ey = ego_xy[fi]
                ax = float(act_frames[fi].get("mx", act_frames[fi].get("x", 0)))
                ay = float(act_frames[fi].get("my", act_frames[fi].get("y", 0)))
                dists.append(math.hypot(ex - ax, ey - ay))
            dists.sort()
            median_d = dists[len(dists) // 2]
            p90_d = dists[int(len(dists) * 0.9)]
            if median_d < 2.0 and p90_d < 4.0:
                ego_actor_dedup_removed.append(str(act_tr.get("id", "")))
            else:
                remaining_actors.append(act_tr)
        actor_tracks = remaining_actors
    if ego_actor_dedup_removed:
        print(f"[INFO] Ego-actor dedup: removed {len(ego_actor_dedup_removed)} actor tracks: {ego_actor_dedup_removed}")

    print(f"[INFO] Preparing lane output ({len(selected_map.lanes)} lanes)...")
    lane_type_counts: Dict[str, int] = {}
    lanes_out: List[Dict[str, object]] = []
    for lane in selected_map.lanes:
        lane_type_counts[lane.lane_type] = lane_type_counts.get(lane.lane_type, 0) + 1
        lx, ly = lane.label_xy
        lanes_out.append(
            {
                "index": int(lane.index),
                "uid": lane.uid,
                "road_id": int(lane.road_id),
                "lane_id": int(lane.lane_id),
                "lane_type": lane.lane_type,
                "entry_lanes": list(lane.entry_lanes),
                "exit_lanes": list(lane.exit_lanes),
                "label_x": float(lx),
                "label_y": float(ly),
                "polyline": _downsample_line(lane.polyline, max_points=map_max_points_per_line),
                "boundary": _downsample_line(lane.boundary, max_points=map_max_points_per_line),
            }
        )

    print(f"[INFO] Preparing CARLA map layer output...")
    carla_map_out: Optional[Dict[str, object]] = None
    if isinstance(carla_map_layer, dict) and carla_map_layer:
        raw_lines = carla_map_layer.get("lines")
        out_lines: List[List[List[float]]] = []
        if isinstance(raw_lines, list):
            for ln in raw_lines:
                if not isinstance(ln, (list, tuple)):
                    continue
                pts: List[Tuple[float, float]] = []
                for p in ln:
                    if not isinstance(p, (list, tuple)) or len(p) < 2:
                        continue
                    x = _safe_float(p[0], float("nan"))
                    y = _safe_float(p[1], float("nan"))
                    if math.isfinite(x) and math.isfinite(y):
                        pts.append((float(x), float(y)))
                ds = _downsample_xy_line(pts, max_points=map_max_points_per_line)
                if len(ds) >= 2:
                    out_lines.append(ds)
        if out_lines:
            bbox_raw = carla_map_layer.get("bbox")
            bbox_dict = {"min_x": 0.0, "max_x": 1.0, "min_y": 0.0, "max_y": 1.0}
            if isinstance(bbox_raw, dict):
                bbox_dict = {
                    "min_x": _safe_float(bbox_raw.get("min_x"), 0.0),
                    "max_x": _safe_float(bbox_raw.get("max_x"), 1.0),
                    "min_y": _safe_float(bbox_raw.get("min_y"), 0.0),
                    "max_y": _safe_float(bbox_raw.get("max_y"), 1.0),
                }
            carla_map_out = {
                "name": str(carla_map_layer.get("name") or "carla_map"),
                "source_path": str(carla_map_layer.get("source_path") or ""),
                "bbox": bbox_dict,
                "line_count": int(len(out_lines)),
                "lines": out_lines,
                "transform": dict(carla_map_layer.get("transform") or {}),
            }
            # Attach the top-down image underlay if provided
            img_b64 = carla_map_layer.get("image_b64")
            img_bounds = carla_map_layer.get("image_bounds")
            if img_b64 and isinstance(img_bounds, dict):
                carla_map_out["image_b64"] = str(img_b64)
                carla_map_out["image_bounds"] = dict(img_bounds)

    print(f"[INFO] Collecting timeline & computing display bbox...")
    timeline = _collect_timeline(ego_tracks, actor_tracks)
    all_display_points: List[Tuple[float, float]] = []
    for lane in lanes_out:
        for p in lane["polyline"]:
            all_display_points.append((float(p[0]), float(p[1])))
    if carla_map_out is not None:
        for ln in carla_map_out.get("lines", []):
            for p in ln:
                all_display_points.append((float(p[0]), float(p[1])))
    for track in ego_tracks + actor_tracks:
        for fr in track["frames"]:
            all_display_points.append((float(fr["x"]), float(fr["y"])))
    points_arr = np.asarray(all_display_points, dtype=np.float64) if all_display_points else np.zeros((0, 2), dtype=np.float64)
    bbox = _compute_bbox_xy(points_arr)
    print(f"[INFO] Export payload assembled: ego_tracks={len(ego_tracks)} actor_tracks={len(actor_tracks)} lanes={len(lanes_out)} timeline_steps={len(timeline)}")

    return {
        "metadata": {
            "scenario_dir": str(scenario_dir),
            "selected_map": selected_map.name,
            "selected_map_path": selected_map.source_path,
            "map_selection_scores": list(selection_details),
            "snap_to_map": bool(snap_to_map),
            "num_ego_tracks": int(len(ego_tracks)),
            "num_actor_tracks": int(len(actor_tracks)),
            "skipped_non_actor_objects": int(skipped_non_actors),
            "timeline_steps": int(len(timeline)),
            "id_merge_stats": merge_stats,
            "timing_policy": dict(timing_optimization.get("timing_policy", {})),
            "timing_optimization": dict(timing_optimization),
            "lane_change_filter": dict(lane_change_cfg or {}),
            "vehicle_lane_policy": {
                "forbidden_lane_types": sorted(forbidden_lane_types),
                "parked_only_lane_types": sorted(parked_only_lane_types),
                "parked_vehicle_cfg": parked_cfg,
            },
            "map_layers": {
                "v2xpnp": True,
                "carla": bool(carla_map_out is not None),
            },
        },
        "map": {
            "name": selected_map.name,
            "source_path": selected_map.source_path,
            "bbox": {
                "min_x": float(selected_map.bbox[0]),
                "max_x": float(selected_map.bbox[1]),
                "min_y": float(selected_map.bbox[2]),
                "max_y": float(selected_map.bbox[3]),
            },
            "lane_type_counts": lane_type_counts,
            "lanes": lanes_out,
        },
        "carla_map": carla_map_out,
        "view_bbox": {
            "min_x": float(bbox[0]),
            "max_x": float(bbox[1]),
            "min_y": float(bbox[2]),
            "max_y": float(bbox[3]),
        },
        "timeline": [float(t) for t in timeline],
        "ego_tracks": ego_tracks,
        "actor_tracks": actor_tracks,
    }


def _build_html(dataset: Dict[str, object]) -> str:
    dataset_json = json.dumps(_sanitize_for_json(dataset), ensure_ascii=True, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>YAML to Map Replay</title>
  <style>
    :root {{
      --bg: #0d1a24;
      --panel: #142635;
      --panel2: #193346;
      --border: #2a4a60;
      --text: #e4edf3;
      --muted: #a8bccb;
      --accent: #4ec4ff;
      --ok: #74f4b3;
      --warn: #ffc26f;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; height: 100%; background: radial-gradient(circle at 15% 10%, #24455d 0%, #0f1f2a 42%, #09131b 100%); color: var(--text); font-family: "IBM Plex Sans", "Segoe UI", sans-serif; }}
    #app {{ height: 100%; display: grid; grid-template-columns: minmax(760px, 1fr) 390px; gap: 10px; padding: 10px; }}
    #main {{ position: relative; border: 1px solid var(--border); border-radius: 12px; overflow: hidden; background: #0b1822; }}
    #canvas {{ width: 100%; height: 100%; display: block; }}
    #hud {{ position: absolute; left: 12px; top: 12px; background: rgba(7, 16, 24, 0.78); border: 1px solid #365a72; border-radius: 10px; padding: 8px 10px; min-width: 290px; font-size: 12px; line-height: 1.35; }}
    #hud .line {{ margin: 2px 0; }}
    #sidebar {{ border: 1px solid var(--border); border-radius: 12px; background: linear-gradient(180deg, var(--panel) 0%, var(--panel2) 100%); display: flex; flex-direction: column; min-height: 0; }}
    #sidebarInner {{ padding: 12px; overflow: auto; }}
    h2 {{ margin: 0 0 10px 0; font-size: 18px; }}
    h3 {{ margin: 12px 0 8px 0; font-size: 13px; color: var(--ok); letter-spacing: 0.2px; }}
    .section {{ border: 1px solid #29465c; border-radius: 10px; padding: 10px; margin-bottom: 10px; background: rgba(9, 18, 26, 0.34); }}
    .small {{ font-size: 12px; color: var(--muted); }}
    .row {{ display: flex; gap: 8px; align-items: center; margin-bottom: 8px; }}
    .row.wrap {{ flex-wrap: wrap; }}
    button, input[type="range"], select {{ width: 100%; }}
    button {{ min-height: 34px; border: 1px solid #365d75; border-radius: 8px; background: linear-gradient(180deg, #2e5068 0%, #253f53 100%); color: var(--text); cursor: pointer; }}
    select {{ min-height: 34px; border: 1px solid #365d75; border-radius: 8px; background: #0f2433; color: var(--text); padding: 4px 8px; }}
    button:hover {{ border-color: #4e89a9; }}
    button:disabled {{ opacity: 0.45; cursor: default; }}
    .btnPrimary {{ background: linear-gradient(180deg, #3ca9df 0%, #2f83b0 100%); color: #08131a; font-weight: 700; }}
    .badge {{ display: inline-block; padding: 2px 6px; border-radius: 999px; border: 1px solid #50758d; font-size: 10px; margin-right: 6px; }}
    .legendItem {{ display: flex; align-items: center; gap: 8px; font-size: 12px; margin-bottom: 5px; }}
    .legendSwatch {{ width: 16px; height: 10px; border-radius: 3px; border: 1px solid #203645; }}
    .mono {{ font-family: "IBM Plex Mono", "Consolas", monospace; font-size: 11px; }}
    #actorList {{ max-height: 240px; overflow: auto; border: 1px solid #2f4e63; border-radius: 8px; padding: 6px; background: rgba(7, 14, 21, 0.45); }}
    .actorRow {{ display: grid; grid-template-columns: 16px 1fr; gap: 8px; align-items: center; margin-bottom: 5px; }}
    .actorDot {{ width: 10px; height: 10px; border-radius: 50%; }}
    #hoverInfo {{ white-space: pre-wrap; font-size: 12px; line-height: 1.35; }}
    label.cb {{ display: flex; align-items: center; gap: 6px; font-size: 12px; color: #d6e5ef; }}
  </style>
</head>
<body>
  <div id="app">
    <div id="main">
      <canvas id="canvas"></canvas>
      <div id="hud">
        <div class="line" id="hudScenario">Scenario: -</div>
        <div class="line" id="hudMap">Map: -</div>
        <div class="line" id="hudTime">Time: -</div>
        <div class="line" id="hudCounts">Actors: -</div>
        <div class="line" id="hudView">View: -</div>
      </div>
    </div>
    <aside id="sidebar">
      <div id="sidebarInner">
        <h2>YAML to Map Replay</h2>

        <div class="section">
          <h3>Timeline</h3>
          <div class="row">
            <button id="prevBtn">Prev</button>
            <button id="playBtn" class="btnPrimary">Play</button>
            <button id="nextBtn">Next</button>
          </div>
          <div class="row">
            <input id="timeSlider" type="range" min="0" max="0" value="0" step="1" />
          </div>
          <div class="small mono" id="timelineLabel">t=0.000</div>
        </div>

        <div class="section">
          <h3>Display</h3>
          <div class="row wrap">
            <label class="cb"><input id="snapToggle" type="checkbox" checked />Use map-snapped poses</label>
            <label class="cb"><input id="showLabelsToggle" type="checkbox" checked />Show lane labels</label>
            <label class="cb"><input id="showBoundaryToggle" type="checkbox" />Show lane boundaries</label>
            <label class="cb"><input id="showTrajToggle" type="checkbox" checked />Show full trajectories</label>
            <label class="cb"><input id="showImageToggle" type="checkbox" checked />Show map image underlay</label>
          </div>
          <div class="row">
            <label class="small" style="min-width: 92px;">Image opacity</label>
            <input id="imageOpacitySlider" type="range" min="0" max="100" value="55" step="1" />
          </div>
          <div class="row">
            <label class="small" style="min-width: 92px;">Map Layer</label>
            <select id="mapLayerSelect">
              <option value="v2xpnp">V2XPNP Map</option>
              <option value="carla">CARLA Westwood Map</option>
            </select>
          </div>
          <div class="row">
            <button id="fitBtn">Fit View</button>
          </div>
        </div>

        <div class="section">
          <h3>Map Selection</h3>
          <div class="small mono" id="mapScoreText"></div>
        </div>

        <div class="section">
          <h3>Lane Types (V2XPNP)</h3>
          <div id="laneLegend"></div>
        </div>

        <div class="section">
          <h3>Actors</h3>
          <div id="actorLegend"></div>
          <div id="actorList"></div>
        </div>

        <div class="section">
          <h3>Hover</h3>
          <div id="hoverInfo" class="mono">Move cursor over an actor marker.</div>
        </div>
      </div>
    </aside>
  </div>

  <script id="dataset" type="application/json">{dataset_json}</script>
  <script>
  (() => {{
    'use strict';

    const DATA = JSON.parse(document.getElementById('dataset').textContent);

    const laneTypePalette = {{
      '1': '#4e79a7',
      '2': '#f28e2b',
      '3': '#59a14f',
      '4': '#e15759',
      'unknown': '#9aa6af',
    }};
    const actorRolePalette = {{
      ego: '#f8c65f',
      vehicle: '#6bc6ff',
      walker: '#7df0a8',
      cyclist: '#c3a4ff',
      static: '#c9ced2',
      npc: '#6bc6ff',
    }};

    const canvas = document.getElementById('canvas');
    const hudScenario = document.getElementById('hudScenario');
    const hudMap = document.getElementById('hudMap');
    const hudTime = document.getElementById('hudTime');
    const hudCounts = document.getElementById('hudCounts');
    const hudView = document.getElementById('hudView');
    const slider = document.getElementById('timeSlider');
    const playBtn = document.getElementById('playBtn');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const fitBtn = document.getElementById('fitBtn');
    const snapToggle = document.getElementById('snapToggle');
    const showLabelsToggle = document.getElementById('showLabelsToggle');
    const showBoundaryToggle = document.getElementById('showBoundaryToggle');
    const showTrajToggle = document.getElementById('showTrajToggle');
    const showImageToggle = document.getElementById('showImageToggle');
    const imageOpacitySlider = document.getElementById('imageOpacitySlider');
    const mapLayerSelect = document.getElementById('mapLayerSelect');
    const timelineLabel = document.getElementById('timelineLabel');
    const mapScoreText = document.getElementById('mapScoreText');
    const laneLegend = document.getElementById('laneLegend');
    const actorLegend = document.getElementById('actorLegend');
    const actorList = document.getElementById('actorList');
    const hoverInfo = document.getElementById('hoverInfo');

    const timeline = Array.isArray(DATA.timeline) ? DATA.timeline : [0.0];
    slider.max = String(Math.max(0, timeline.length - 1));
    slider.value = '0';
    const hasCarlaLayer = !!(DATA.carla_map && Array.isArray(DATA.carla_map.lines) && DATA.carla_map.lines.length > 0);
    if (!hasCarlaLayer && mapLayerSelect) {{
      mapLayerSelect.value = 'v2xpnp';
      mapLayerSelect.disabled = true;
    }}

    // --- Preload CARLA top-down image underlay ---
    let carlaMapImage = null;
    const hasCarlaImage = !!(DATA.carla_map && DATA.carla_map.image_b64 && DATA.carla_map.image_bounds);
    if (hasCarlaImage) {{
      carlaMapImage = new Image();
      carlaMapImage.src = 'data:image/jpeg;base64,' + DATA.carla_map.image_b64;
      carlaMapImage.onload = () => {{ render(); }};
    }}
    if (!hasCarlaImage && showImageToggle) {{
      showImageToggle.checked = false;
      showImageToggle.disabled = true;
    }}

    const state = {{
      tIndex: 0,
      playing: false,
      playHandle: null,
      lastLayer: 'v2xpnp',
      view: {{
        cx: 0.0,
        cy: 0.0,
        scale: 3.0,
        minScale: 0.1,
        maxScale: 260.0,
      }},
      drag: null,
      hovered: null,
    }};

    function clamp(v, lo, hi) {{
      return Math.max(lo, Math.min(hi, v));
    }}

    function normalizeDeg(x) {{
      let a = Number(x) || 0;
      while (a > 180) a -= 360;
      while (a <= -180) a += 360;
      return a;
    }}

    function lerpAngleDeg(a, b, t) {{
      const aa = normalizeDeg(a);
      const bb = normalizeDeg(b);
      const d = normalizeDeg(bb - aa);
      return normalizeDeg(aa + d * t);
    }}

    function getCanvasCssSize() {{
      const rect = canvas.getBoundingClientRect();
      return {{ width: rect.width, height: rect.height }};
    }}

    function resizeCanvas() {{
      const dpr = window.devicePixelRatio || 1;
      const size = getCanvasCssSize();
      const w = Math.max(2, Math.round(size.width * dpr));
      const h = Math.max(2, Math.round(size.height * dpr));
      if (canvas.width !== w || canvas.height !== h) {{
        canvas.width = w;
        canvas.height = h;
      }}
    }}

    function worldToScreen(x, y) {{
      const size = getCanvasCssSize();
      return {{
        x: size.width * 0.5 + (x - state.view.cx) * state.view.scale,
        y: size.height * 0.5 - (y - state.view.cy) * state.view.scale,
      }};
    }}

    function screenToWorld(sx, sy) {{
      const size = getCanvasCssSize();
      return {{
        x: state.view.cx + (sx - size.width * 0.5) / state.view.scale,
        y: state.view.cy - (sy - size.height * 0.5) / state.view.scale,
      }};
    }}

    function eventScreenPos(evt) {{
      const rect = canvas.getBoundingClientRect();
      return {{ x: evt.clientX - rect.left, y: evt.clientY - rect.top }};
    }}

    function timeNow() {{
      return timeline[clamp(state.tIndex, 0, timeline.length - 1)] || 0.0;
    }}

    function findFramePair(frames, t) {{
      if (!Array.isArray(frames) || frames.length === 0) return null;
      if (t < frames[0].t || t > frames[frames.length - 1].t) return null;
      let lo = 0;
      let hi = frames.length - 1;
      while (lo <= hi) {{
        const mid = (lo + hi) >> 1;
        const mt = frames[mid].t;
        if (mt < t) {{
          lo = mid + 1;
        }} else if (mt > t) {{
          hi = mid - 1;
        }} else {{
          return {{ i0: mid, i1: mid, alpha: 0.0 }};
        }}
      }}
      const i1 = clamp(lo, 0, frames.length - 1);
      const i0 = clamp(i1 - 1, 0, frames.length - 1);
      const t0 = frames[i0].t;
      const t1 = frames[i1].t;
      const alpha = (t1 > t0) ? ((t - t0) / (t1 - t0)) : 0.0;
      return {{ i0, i1, alpha }};
    }}

    function sampleTrackPose(track, t, useMapSnap, activeLayer) {{
      const frames = track.frames || [];
      const pair = findFramePair(frames, t);
      if (!pair) return null;
      const f0 = frames[pair.i0];
      const f1 = frames[pair.i1];
      const a = pair.alpha;
      const useCarlaSnap = !!(useMapSnap && activeLayer === 'carla');
      const px0 = useCarlaSnap ? (Number.isFinite(Number(f0.cx)) ? Number(f0.cx) : Number(f0.mx)) : (useMapSnap ? Number(f0.mx) : Number(f0.rx));
      const py0 = useCarlaSnap ? (Number.isFinite(Number(f0.cy)) ? Number(f0.cy) : Number(f0.my)) : (useMapSnap ? Number(f0.my) : Number(f0.ry));
      const pz0 = useMapSnap ? Number(f0.mz) : Number(f0.rz);
      const pyaw0 = useCarlaSnap ? (Number.isFinite(Number(f0.cyaw)) ? Number(f0.cyaw) : Number(f0.myaw)) : (useMapSnap ? Number(f0.myaw) : Number(f0.ryaw));
      if (pair.i0 === pair.i1) {{
        return {{
          x: px0, y: py0, z: pz0, yaw: pyaw0,
          laneIndex: useCarlaSnap ? Number(f0.cli) : Number(f0.li),
          laneDist: useCarlaSnap ? Number(f0.cld) : Number(f0.ld),
          laneSource: useCarlaSnap ? String(f0.csource || '') : 'v2_lane',
          laneQuality: useCarlaSnap ? String(f0.cquality || '') : '',
          v2LaneIndex: Number(f0.li),
          carlaLineIndex: Number(f0.cli),
          frame: f0,
        }};
      }}
      const px1 = useCarlaSnap ? (Number.isFinite(Number(f1.cx)) ? Number(f1.cx) : Number(f1.mx)) : (useMapSnap ? Number(f1.mx) : Number(f1.rx));
      const py1 = useCarlaSnap ? (Number.isFinite(Number(f1.cy)) ? Number(f1.cy) : Number(f1.my)) : (useMapSnap ? Number(f1.my) : Number(f1.ry));
      const pz1 = useMapSnap ? Number(f1.mz) : Number(f1.rz);
      const pyaw1 = useCarlaSnap ? (Number.isFinite(Number(f1.cyaw)) ? Number(f1.cyaw) : Number(f1.myaw)) : (useMapSnap ? Number(f1.myaw) : Number(f1.ryaw));
      const laneFrame = (a < 0.5) ? f0 : f1;
      return {{
        x: px0 + (px1 - px0) * a,
        y: py0 + (py1 - py0) * a,
        z: pz0 + (pz1 - pz0) * a,
        yaw: lerpAngleDeg(pyaw0, pyaw1, a),
        laneIndex: useCarlaSnap ? Number(laneFrame.cli) : Number(laneFrame.li),
        laneDist: useCarlaSnap ? Number(laneFrame.cld) : Number(laneFrame.ld),
        laneSource: useCarlaSnap ? String(laneFrame.csource || '') : 'v2_lane',
        laneQuality: useCarlaSnap ? String(laneFrame.cquality || '') : '',
        v2LaneIndex: Number(laneFrame.li),
        carlaLineIndex: Number(laneFrame.cli),
        frame: laneFrame,
      }};
    }}

    function laneColor(laneType) {{
      const key = String(laneType);
      return laneTypePalette[key] || '#8f9aa2';
    }}

    function actorColor(role) {{
      const key = String(role || 'npc').toLowerCase();
      return actorRolePalette[key] || actorRolePalette.npc;
    }}

    function getActiveMapLayer() {{
      if (!mapLayerSelect) return 'v2xpnp';
      const v = String(mapLayerSelect.value || 'v2xpnp').toLowerCase();
      if (v === 'carla' && hasCarlaLayer) return 'carla';
      return 'v2xpnp';
    }}

    function getLayerBBox(layer) {{
      if (layer === 'carla') {{
        const b = DATA.carla_map?.bbox;
        if (b) return b;
      }}
      const b0 = DATA.map?.bbox;
      if (b0) return b0;
      return DATA.view_bbox || null;
    }}

    function getActiveLayerBBox() {{
      return getLayerBBox(getActiveMapLayer());
    }}

    function fitScaleForBBox(b) {{
      if (!b) return state.view.scale;
      const minX = Number(b.min_x);
      const maxX = Number(b.max_x);
      const minY = Number(b.min_y);
      const maxY = Number(b.max_y);
      if (!Number.isFinite(minX) || !Number.isFinite(maxX) || !Number.isFinite(minY) || !Number.isFinite(maxY)) {{
        return state.view.scale;
      }}
      const size = getCanvasCssSize();
      const rangeX = Math.max(1.0, maxX - minX);
      const rangeY = Math.max(1.0, maxY - minY);
      const pad = 0.90;
      const sx = (size.width * pad) / rangeX;
      const sy = (size.height * pad) / rangeY;
      return clamp(Math.min(sx, sy), state.view.minScale, state.view.maxScale);
    }}

    function fitView() {{
      const b = getActiveLayerBBox();
      if (!b) return;
      const minX = Number(b.min_x);
      const maxX = Number(b.max_x);
      const minY = Number(b.min_y);
      const maxY = Number(b.max_y);
      if (!Number.isFinite(minX) || !Number.isFinite(maxX) || !Number.isFinite(minY) || !Number.isFinite(maxY)) return;
      state.view.scale = fitScaleForBBox(b);
      state.view.cx = 0.5 * (minX + maxX);
      state.view.cy = 0.5 * (minY + maxY);
    }}

    function remapViewForLayerSwitch(prevLayer, nextLayer) {{
      const oldB = getLayerBBox(prevLayer);
      const newB = getLayerBBox(nextLayer);
      if (!oldB || !newB) return;
      const oldMinX = Number(oldB.min_x);
      const oldMaxX = Number(oldB.max_x);
      const oldMinY = Number(oldB.min_y);
      const oldMaxY = Number(oldB.max_y);
      const newMinX = Number(newB.min_x);
      const newMaxX = Number(newB.max_x);
      const newMinY = Number(newB.min_y);
      const newMaxY = Number(newB.max_y);
      if (
        !Number.isFinite(oldMinX) || !Number.isFinite(oldMaxX) || !Number.isFinite(oldMinY) || !Number.isFinite(oldMaxY) ||
        !Number.isFinite(newMinX) || !Number.isFinite(newMaxX) || !Number.isFinite(newMinY) || !Number.isFinite(newMaxY)
      ) {{
        return;
      }}
      const oldRangeX = Math.max(1.0, oldMaxX - oldMinX);
      const oldRangeY = Math.max(1.0, oldMaxY - oldMinY);
      const newRangeX = Math.max(1.0, newMaxX - newMinX);
      const newRangeY = Math.max(1.0, newMaxY - newMinY);

      const oldFit = fitScaleForBBox(oldB);
      const newFit = fitScaleForBBox(newB);
      const zoomFactor = (oldFit > 1e-9) ? (state.view.scale / oldFit) : 1.0;

      const nx = clamp((state.view.cx - oldMinX) / oldRangeX, 0.0, 1.0);
      const ny = clamp((state.view.cy - oldMinY) / oldRangeY, 0.0, 1.0);
      state.view.cx = newMinX + nx * newRangeX;
      state.view.cy = newMinY + ny * newRangeY;
      state.view.scale = clamp(newFit * zoomFactor, state.view.minScale, state.view.maxScale);
    }}

    function drawLaneLine(ctx, points, color, widthPx, alpha) {{
      if (!Array.isArray(points) || points.length < 2) return;
      ctx.beginPath();
      for (let i = 0; i < points.length; i += 1) {{
        const p = points[i];
        const px = Array.isArray(p) ? Number(p[0]) : Number(p?.x);
        const py = Array.isArray(p) ? Number(p[1]) : Number(p?.y);
        const s = worldToScreen(px, py);
        if (i === 0) ctx.moveTo(s.x, s.y);
        else ctx.lineTo(s.x, s.y);
      }}
      ctx.strokeStyle = color;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = widthPx;
      ctx.stroke();
      ctx.globalAlpha = 1.0;
    }}

    function drawActorShape(ctx, pose, role, color) {{
      const s = worldToScreen(pose.x, pose.y);
      const yaw = (Number(pose.yaw) || 0.0) * Math.PI / 180.0;
      const base = clamp(state.view.scale * 0.9, 4.0, 15.0);
      ctx.save();
      ctx.translate(s.x, s.y);
      ctx.rotate(-yaw);
      ctx.fillStyle = color;
      ctx.strokeStyle = '#09131a';
      ctx.lineWidth = 1.2;
      if (role === 'ego') {{
        ctx.beginPath();
        ctx.moveTo(base * 1.3, 0);
        ctx.lineTo(-base * 0.8, base * 0.85);
        ctx.lineTo(-base * 0.8, -base * 0.85);
        ctx.closePath();
      }} else if (role === 'vehicle' || role === 'npc') {{
        const w = base * 2.0;
        const h = base * 1.0;
        ctx.beginPath();
        ctx.rect(-w * 0.5, -h * 0.5, w, h);
      }} else if (role === 'cyclist') {{
        ctx.beginPath();
        ctx.moveTo(base * 1.0, 0);
        ctx.lineTo(0, base * 1.0);
        ctx.lineTo(-base * 1.0, 0);
        ctx.lineTo(0, -base * 1.0);
        ctx.closePath();
      }} else if (role === 'walker') {{
        ctx.beginPath();
        ctx.arc(0, 0, base * 0.75, 0, Math.PI * 2);
      }} else {{
        ctx.beginPath();
        ctx.rect(-base * 0.75, -base * 0.75, base * 1.5, base * 1.5);
      }}
      ctx.fill();
      ctx.stroke();
      ctx.restore();
      return s;
    }}

    function render() {{
      resizeCanvas();
      const size = getCanvasCssSize();
      const dpr = window.devicePixelRatio || 1;
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, size.width, size.height);
      ctx.fillStyle = '#0b1822';
      ctx.fillRect(0, 0, size.width, size.height);

      const useMapSnap = !!snapToggle.checked;
      const showLabels = !!showLabelsToggle.checked;
      const showBoundary = !!showBoundaryToggle.checked;
      const showTraj = !!showTrajToggle.checked;
      const t = timeNow();
      const activeLayer = getActiveMapLayer();
      const lanes = (DATA.map && Array.isArray(DATA.map.lanes)) ? DATA.map.lanes : [];
      // --- Draw CARLA map image underlay ---
      const showImage = hasCarlaImage && carlaMapImage && carlaMapImage.complete && carlaMapImage.naturalWidth > 0 && !!showImageToggle.checked;
      if (showImage && (activeLayer === 'carla')) {{
        const ib = DATA.carla_map.image_bounds;
        // image_bounds are in V2XPNP coords: min_x, max_x, min_y, max_y
        // Due to flip_y, the raw image top (small CARLA-Y) maps to max_y in V2XPNP
        // worldToScreen maps max_y to screen-top and min_y to screen-bottom
        const topLeft = worldToScreen(Number(ib.min_x), Number(ib.max_y));
        const bottomRight = worldToScreen(Number(ib.max_x), Number(ib.min_y));
        const imgOpacity = Number(imageOpacitySlider.value) / 100.0;
        ctx.globalAlpha = imgOpacity;
        ctx.drawImage(carlaMapImage, topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);
        ctx.globalAlpha = 1.0;
      }}
      if (activeLayer === 'carla' && hasCarlaLayer) {{
        const carlaLines = Array.isArray(DATA.carla_map?.lines) ? DATA.carla_map.lines : [];
        for (const ln of carlaLines) {{
          const poly = Array.isArray(ln?.polyline) ? ln.polyline : ln;
          drawLaneLine(ctx, poly, '#9fb2c4', 1.2, 0.78);
        }}
        if (showLabels) {{
          ctx.font = '10px "IBM Plex Mono", Consolas, monospace';
          ctx.fillStyle = '#d0dde8';
          ctx.globalAlpha = 0.86;
          for (const ln of carlaLines) {{
            const lx = Number(ln?.label_x);
            const ly = Number(ln?.label_y);
            if (!Number.isFinite(lx) || !Number.isFinite(ly)) continue;
            const s = worldToScreen(lx, ly);
            const lbl = String(ln?.label || `c${{Number(ln?.index ?? -1)}}`);
            let txt = lbl;
            if (Array.isArray(ln?.matched_v2_labels) && ln.matched_v2_labels.length > 0) {{
              txt += ` <- ${{ln.matched_v2_labels[0]}}`;
            }}
            ctx.fillText(txt, s.x + 3, s.y - 3);
          }}
          ctx.globalAlpha = 1.0;
        }}
      }} else {{
        for (const lane of lanes) {{
          drawLaneLine(ctx, lane.polyline, laneColor(lane.lane_type), 1.4, 0.78);
        }}
        if (showBoundary) {{
          ctx.setLineDash([4, 4]);
          for (const lane of lanes) {{
            drawLaneLine(ctx, lane.boundary, '#8a8a8a', 1.0, 0.48);
          }}
          ctx.setLineDash([]);
        }}
        if (showLabels) {{
          ctx.font = '10px "IBM Plex Mono", Consolas, monospace';
          ctx.fillStyle = '#d0dde8';
          ctx.globalAlpha = 0.85;
          for (const lane of lanes) {{
            const s = worldToScreen(Number(lane.label_x), Number(lane.label_y));
            let txt = `r${{lane.road_id}} l${{lane.lane_id}} t${{lane.lane_type}}`;
            const cm = lane.carla_match;
            if (cm && Number.isInteger(Number(cm.carla_line_index)) && Number(cm.carla_line_index) >= 0) {{
              txt += ` -> c${{Number(cm.carla_line_index)}} (${{String(cm.quality || '').slice(0,1)}})`;
            }}
            ctx.fillText(txt, s.x + 3, s.y - 3);
          }}
          ctx.globalAlpha = 1.0;
        }}
      }}

      const tracks = []
        .concat(Array.isArray(DATA.ego_tracks) ? DATA.ego_tracks : [])
        .concat(Array.isArray(DATA.actor_tracks) ? DATA.actor_tracks : []);
      if (showTraj) {{
        for (const tr of tracks) {{
          const frames = tr.frames || [];
          if (frames.length < 2) continue;
          ctx.beginPath();
          for (let i = 0; i < frames.length; i += 1) {{
            const fr = frames[i];
            const x = (useMapSnap && activeLayer === 'carla')
              ? (Number.isFinite(Number(fr.cx)) ? Number(fr.cx) : Number(fr.mx))
              : (useMapSnap ? Number(fr.mx) : Number(fr.rx));
            const y = (useMapSnap && activeLayer === 'carla')
              ? (Number.isFinite(Number(fr.cy)) ? Number(fr.cy) : Number(fr.my))
              : (useMapSnap ? Number(fr.my) : Number(fr.ry));
            const s = worldToScreen(x, y);
            if (i === 0) ctx.moveTo(s.x, s.y);
            else ctx.lineTo(s.x, s.y);
          }}
          ctx.strokeStyle = actorColor(tr.role);
          ctx.globalAlpha = 0.24;
          ctx.lineWidth = (tr.role === 'ego') ? 2.2 : 1.0;
          ctx.stroke();
          ctx.globalAlpha = 1.0;
        }}
      }}

      const markers = [];
      for (const tr of tracks) {{
        const pose = sampleTrackPose(tr, t, useMapSnap, activeLayer);
        if (!pose) continue;
        const color = actorColor(tr.role);
        const screen = drawActorShape(ctx, pose, String(tr.role || 'npc'), color);
        markers.push({{
          track: tr,
          pose,
          sx: screen.x,
          sy: screen.y,
        }});

        ctx.font = '10px "IBM Plex Sans", sans-serif';
        ctx.fillStyle = '#e6eef5';
        ctx.globalAlpha = 0.9;
        ctx.fillText(String(tr.name || tr.id), screen.x + 6, screen.y - 6);
        ctx.globalAlpha = 1.0;
      }}

      const hover = state.hovered;
      if (hover && hover.track && hover.pose) {{
        const s = worldToScreen(hover.pose.x, hover.pose.y);
        ctx.beginPath();
        ctx.arc(s.x, s.y, 9, 0, Math.PI * 2);
        ctx.strokeStyle = '#ffe08a';
        ctx.lineWidth = 2.0;
        ctx.stroke();
      }}

      hudScenario.textContent = `Scenario: ${{DATA.metadata?.scenario_dir || '-'}}`;
      const activeMapName = (activeLayer === 'carla' && hasCarlaLayer)
        ? (DATA.carla_map?.name || 'carla_map')
        : (DATA.map?.name || '-');
      hudMap.textContent = `Map: ${{activeMapName}} (layer=${{activeLayer}})`;
      state.lastLayer = activeLayer;
      hudTime.textContent = `Time: ${{t.toFixed(3)}} s (step ${{state.tIndex + 1}}/${{timeline.length}})`;
      hudCounts.textContent = `Actors: ego=${{(DATA.ego_tracks || []).length}} custom=${{(DATA.actor_tracks || []).length}}`;
      hudView.textContent = `View: scale=${{state.view.scale.toFixed(3)}} px/m center=(${{state.view.cx.toFixed(2)}}, ${{state.view.cy.toFixed(2)}})`;
      timelineLabel.textContent = `t=${{t.toFixed(3)}} sec`;

      // hover nearest marker
      if (state.mouse) {{
        let best = null;
        let bestD2 = 14 * 14;
        for (const m of markers) {{
          const dx = m.sx - state.mouse.x;
          const dy = m.sy - state.mouse.y;
          const d2 = dx * dx + dy * dy;
          if (d2 < bestD2) {{
            bestD2 = d2;
            best = m;
          }}
        }}
        state.hovered = best;
      }} else {{
        state.hovered = null;
      }}

      if (state.hovered && state.hovered.track) {{
        const tr = state.hovered.track;
        const pose = state.hovered.pose;
        let laneTxt = 'lane: none';
        if (activeLayer === 'carla') {{
          const carlaLines = Array.isArray(DATA.carla_map?.lines) ? DATA.carla_map.lines : [];
          const ci = Number(pose.carlaLineIndex);
          if (Number.isInteger(ci) && ci >= 0) {{
            const cl = carlaLines.find((x) => Number(x?.index) === ci) || null;
            if (cl) {{
              const mappedLbl = (Array.isArray(cl.matched_v2_labels) && cl.matched_v2_labels.length > 0)
                ? cl.matched_v2_labels.join(', ')
                : '-';
              laneTxt =
                `carla_lane: ${{String(cl.label || `c${{ci}}`)}} dist=${{Number(pose.laneDist).toFixed(2)}}m src=${{String(pose.laneSource || '-')}} q=${{String(pose.laneQuality || '-')}}\\n` +
                `mapped_v2: ${{mappedLbl}}\\n` +
                `pose_v2_lane_index=${{Number.isInteger(pose.v2LaneIndex) ? pose.v2LaneIndex : -1}}`;
            }} else {{
              laneTxt = `carla_lane: c${{ci}} dist=${{Number(pose.laneDist).toFixed(2)}}m`;
            }}
          }}
        }} else if (Number.isInteger(pose.laneIndex) && pose.laneIndex >= 0) {{
          const lane = lanes.find((ln) => Number(ln?.index) === Number(pose.laneIndex)) || null;
          if (lane) {{
          const cm = lane.carla_match;
          const mapTxt = (cm && Number.isInteger(Number(cm.carla_line_index)) && Number(cm.carla_line_index) >= 0)
            ? ` -> c${{Number(cm.carla_line_index)}} q=${{String(cm.quality || '-')}}`
            : '';
          laneTxt = `lane: road=${{lane.road_id}} lane=${{lane.lane_id}} type=${{lane.lane_type}} dist=${{Number(pose.laneDist).toFixed(2)}}m${{mapTxt}}`;
          }}
        }}
        hoverInfo.textContent =
          `name: ${{tr.name}}\\n` +
          `id: ${{tr.id}}\\n` +
          `vid: ${{tr.vid ?? '-'}} orig_vid: ${{tr.orig_vid ?? '-'}}\\n` +
          `merged_vids: ${{Array.isArray(tr.merged_vids) ? tr.merged_vids.join(',') : '-'}}\\n` +
          `role: ${{tr.role}}\\n` +
          `obj_type: ${{tr.obj_type || '-'}}\\n` +
          `model: ${{tr.model || '-'}}\\n` +
          `x=${{Number(pose.x).toFixed(2)}} y=${{Number(pose.y).toFixed(2)}} z=${{Number(pose.z).toFixed(2)}} yaw=${{Number(pose.yaw).toFixed(1)}}\\n` +
          laneTxt;
      }} else {{
        hoverInfo.textContent = 'Move cursor over an actor marker.';
      }}
    }}

    function setIndex(i) {{
      state.tIndex = clamp(i, 0, timeline.length - 1);
      slider.value = String(state.tIndex);
      render();
    }}

    function togglePlay() {{
      state.playing = !state.playing;
      playBtn.textContent = state.playing ? 'Pause' : 'Play';
      if (state.playing) {{
        const tick = () => {{
          if (!state.playing) return;
          if (state.tIndex >= timeline.length - 1) {{
            state.playing = false;
            playBtn.textContent = 'Play';
            return;
          }}
          setIndex(state.tIndex + 1);
          state.playHandle = window.setTimeout(tick, 60);
        }};
        tick();
      }} else if (state.playHandle) {{
        window.clearTimeout(state.playHandle);
        state.playHandle = null;
      }}
    }}

    slider.addEventListener('input', (evt) => {{
      const next = Number.parseInt(evt.target.value, 10);
      setIndex(Number.isFinite(next) ? next : 0);
    }});
    prevBtn.addEventListener('click', () => setIndex(state.tIndex - 1));
    nextBtn.addEventListener('click', () => setIndex(state.tIndex + 1));
    playBtn.addEventListener('click', togglePlay);
    fitBtn.addEventListener('click', () => {{
      fitView();
      render();
    }});
    snapToggle.addEventListener('change', render);
    showLabelsToggle.addEventListener('change', render);
    showBoundaryToggle.addEventListener('change', render);
    showTrajToggle.addEventListener('change', render);
    if (showImageToggle) showImageToggle.addEventListener('change', render);
    if (imageOpacitySlider) imageOpacitySlider.addEventListener('input', render);
    if (mapLayerSelect) {{
      mapLayerSelect.addEventListener('change', () => {{
        const prevLayer = String(state.lastLayer || 'v2xpnp');
        const nextLayer = getActiveMapLayer();
        if (prevLayer !== nextLayer) {{
          remapViewForLayerSwitch(prevLayer, nextLayer);
        }}
        state.lastLayer = nextLayer;
        render();
      }});
    }}

    canvas.addEventListener('mousedown', (evt) => {{
      const p = eventScreenPos(evt);
      state.drag = {{ sx: p.x, sy: p.y, cx: state.view.cx, cy: state.view.cy }};
    }});
    window.addEventListener('mouseup', () => {{
      state.drag = null;
    }});
    window.addEventListener('mousemove', (evt) => {{
      const rect = canvas.getBoundingClientRect();
      if (evt.clientX < rect.left || evt.clientX > rect.right || evt.clientY < rect.top || evt.clientY > rect.bottom) {{
        state.mouse = null;
      }} else {{
        state.mouse = {{ x: evt.clientX - rect.left, y: evt.clientY - rect.top }};
      }}
      if (!state.drag) {{
        render();
        return;
      }}
      const p = eventScreenPos(evt);
      const dx = p.x - state.drag.sx;
      const dy = p.y - state.drag.sy;
      state.view.cx = state.drag.cx - dx / state.view.scale;
      state.view.cy = state.drag.cy + dy / state.view.scale;
      render();
    }});
    canvas.addEventListener('wheel', (evt) => {{
      evt.preventDefault();
      const p = eventScreenPos(evt);
      const before = screenToWorld(p.x, p.y);
      const zoom = Math.exp(-evt.deltaY * 0.0012);
      state.view.scale = clamp(state.view.scale * zoom, state.view.minScale, state.view.maxScale);
      const after = screenToWorld(p.x, p.y);
      state.view.cx += before.x - after.x;
      state.view.cy += before.y - after.y;
      render();
    }}, {{ passive: false }});

    window.addEventListener('keydown', (evt) => {{
      if (evt.key === 'ArrowRight') {{
        setIndex(state.tIndex + 1);
      }} else if (evt.key === 'ArrowLeft') {{
        setIndex(state.tIndex - 1);
      }} else if (evt.key === ' ') {{
        evt.preventDefault();
        togglePlay();
      }}
    }});

    function buildSidebar() {{
      const scores = DATA.metadata?.map_selection_scores || [];
      const scoreLines = [];
      for (const s of scores) {{
        scoreLines.push(
          `${{s.name}}\\n` +
          `  score=${{Number(s.score).toFixed(3)}} ` +
          `median=${{Number(s.median_nearest_m).toFixed(2)}}m ` +
          `p90=${{Number(s.p90_nearest_m).toFixed(2)}}m ` +
          `outside=${{(100 * Number(s.outside_bbox_ratio)).toFixed(1)}}%`
        );
      }}
      const mergeStats = DATA.metadata?.id_merge_stats || {{}};
      scoreLines.push(
        'ID merge stats\\n' +
        `  input=${{Number(mergeStats.input_tracks || 0)}} output=${{Number(mergeStats.output_tracks || 0)}} ` +
        `collisions=${{Number(mergeStats.ids_with_collisions || 0)}} ` +
        `merged=${{Number(mergeStats.merged_duplicates || 0)}} split=${{Number(mergeStats.split_tracks_created || 0)}} ` +
        `thr=${{Number(mergeStats.id_merge_distance_m || 0).toFixed(2)}}m`
      );
      scoreLines.push(
        'Cross-ID dedup\\n' +
        `  enabled=${{Boolean(mergeStats.cross_id_dedup_enabled)}} ` +
        `pairs_checked=${{Number(mergeStats.cross_id_pair_checks || 0)}} ` +
        `candidate_pairs=${{Number(mergeStats.cross_id_candidate_pairs || 0)}}\\n` +
        `  clusters=${{Number(mergeStats.cross_id_clusters || 0)}} removed=${{Number(mergeStats.cross_id_removed || 0)}}`
      );
      const tOpt = DATA.metadata?.timing_optimization || {{}};
      const tEarly = tOpt.early_spawn || {{}};
      const tLate = tOpt.late_despawn || {{}};
      scoreLines.push(
        'Timing optimization\\n' +
        `  early: enabled=${{Boolean(tEarly.enabled)}} applied=${{Boolean(tEarly.applied)}} ` +
        `adjusted=${{Array.isArray(tEarly.adjusted_actor_ids) ? tEarly.adjusted_actor_ids.length : 0}}\\n` +
        `  late: enabled=${{Boolean(tLate.enabled)}} applied=${{Boolean(tLate.applied)}} ` +
        `adjusted=${{Array.isArray(tLate.adjusted_actor_ids) ? tLate.adjusted_actor_ids.length : 0}} ` +
        `horizon=${{Number(tLate.hold_until_time || 0).toFixed(2)}}s`
      );
      const laneCfg = DATA.metadata?.lane_change_filter || {{}};
      scoreLines.push(
        'Lane-change filter\\n' +
        `  enabled=${{Boolean(laneCfg.enabled)}} top_k=${{Number(laneCfg.lane_top_k || 0)}} ` +
        `window=${{Number(laneCfg.confirm_window || 0)}} votes=${{Number(laneCfg.confirm_votes || 0)}}\\n` +
        `  cooldown=${{Number(laneCfg.cooldown_frames || 0)}} endpoint_guard=${{Number(laneCfg.endpoint_guard_frames || 0)}}`
      );
      const lc = DATA.metadata?.lane_correspondence || {{}};
      if (Boolean(lc.enabled)) {{
        scoreLines.push(
          'Lane correspondence\\n' +
          `  mapped_lanes=${{Number(lc.mapped_lane_count || 0)}} usable=${{Number(lc.usable_lane_count || 0)}} mapped_carla_lines=${{Number(lc.mapped_carla_line_count || 0)}}\\n` +
          `  quality=${{JSON.stringify(lc.quality_counts || {{}})}}\\n` +
          `  splits=${{Number(lc.split_merge_count || 0)}} phantom_changes=${{Number(lc.total_phantom_lane_changes || 0)}}\\n` +
          `  connectivity=${{Number(lc.connectivity_edges_preserved || 0)}}/${{Number(lc.connectivity_edges_total || 0)}}`
        );
      }}
      if (DATA.carla_map && DATA.carla_map.transform) {{
        const ctf = DATA.carla_map.transform;
        scoreLines.push(
          'CARLA layer alignment\\n' +
          `  lines=${{Number(DATA.carla_map.line_count || 0)}} scale=${{Number(ctf.scale || 1).toFixed(3)}} ` +
          `theta=${{Number(ctf.theta_deg || 0).toFixed(3)}}deg\\n` +
          `  tx=${{Number(ctf.tx || 0).toFixed(3)}} ty=${{Number(ctf.ty || 0).toFixed(3)}} flip_y=${{Boolean(ctf.flip_y)}}`
        );
      }}
      mapScoreText.textContent = scoreLines.join('\\n\\n');

      laneLegend.innerHTML = '';
      const laneCounts = (DATA.map && DATA.map.lane_type_counts) ? DATA.map.lane_type_counts : {{}};
      Object.keys(laneCounts).sort((a, b) => Number(a) - Number(b)).forEach((k) => {{
        const row = document.createElement('div');
        row.className = 'legendItem';
        const sw = document.createElement('div');
        sw.className = 'legendSwatch';
        sw.style.background = laneColor(k);
        const txt = document.createElement('div');
        txt.textContent = `type=${{k}} lanes=${{laneCounts[k]}}`;
        row.appendChild(sw);
        row.appendChild(txt);
        laneLegend.appendChild(row);
      }});

      const actorCounts = {{}};
      const tracks = []
        .concat(Array.isArray(DATA.ego_tracks) ? DATA.ego_tracks : [])
        .concat(Array.isArray(DATA.actor_tracks) ? DATA.actor_tracks : []);
      for (const tr of tracks) {{
        const key = String(tr.role || 'npc');
        actorCounts[key] = (actorCounts[key] || 0) + 1;
      }}

      actorLegend.innerHTML = '';
      Object.keys(actorCounts).sort().forEach((k) => {{
        const row = document.createElement('div');
        row.className = 'legendItem';
        const sw = document.createElement('div');
        sw.className = 'legendSwatch';
        sw.style.background = actorColor(k);
        const txt = document.createElement('div');
        txt.textContent = `${{k}}: ${{actorCounts[k]}}`;
        row.appendChild(sw);
        row.appendChild(txt);
        actorLegend.appendChild(row);
      }});

      actorList.innerHTML = '';
      tracks.forEach((tr) => {{
        const row = document.createElement('div');
        row.className = 'actorRow';
        const dot = document.createElement('div');
        dot.className = 'actorDot';
        dot.style.background = actorColor(tr.role);
        const txt = document.createElement('div');
        const hasOrig = Number.isInteger(tr.orig_vid) && Number(tr.orig_vid) !== Number(tr.vid);
        const idPart = hasOrig ? `id=${{tr.vid}} (orig=${{tr.orig_vid}})` : `id=${{tr.vid || tr.id}}`;
        const mergedCount = Array.isArray(tr.merged_vids) ? tr.merged_vids.length : 1;
        const mergePart = mergedCount > 1 ? ` merged=${{mergedCount}}` : '';
        // Early spawn indicator
        let spawnPart = '';
        if (tr.has_early_spawn && tr.early_spawn_time != null) {{
          const advanceSec = (tr.first_observed_time - tr.early_spawn_time).toFixed(1);
          spawnPart = ` 🚀${{advanceSec}}s`;
        }}
        txt.textContent = `${{tr.name}} [${{tr.role}}] ${{idPart}} obj=${{tr.obj_type || '-'}}${{mergePart}}${{spawnPart}}`;
        // Add tooltip with timing details
        if (tr.first_observed_time != null) {{
          let tooltip = `First observed: ${{tr.first_observed_time.toFixed(2)}}s`;
          if (tr.early_spawn_time != null) {{
            tooltip += `\\nEarly spawn: ${{tr.early_spawn_time.toFixed(2)}}s`;
          }}
          row.title = tooltip;
        }}
        row.appendChild(dot);
        row.appendChild(txt);
        actorList.appendChild(row);
      }});
    }}

    window.addEventListener('resize', () => {{
      render();
    }});

    buildSidebar();
    fitView();
    render();
  }})();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert YAML scenarios to map-anchored replay + interactive HTML.")
    parser.add_argument("--scenario-dir", required=False, default=None, help="Scenario folder containing YAML subfolders.")
    parser.add_argument(
        "--scenario-dirs",
        nargs="+",
        default=None,
        help="Multiple scenario directories to process in batch. Each directory will be processed, simulated, aligned, and have results/videos generated. Results are named after each scenario."
    )
    parser.add_argument(
        "--subdir",
        default="all",
        help="Subfolder selector like yaml_to_carla_log (--subdir -1 / 0 / all).",
    )
    parser.add_argument("--out-dir", default=None, help="Output directory (default: <scenario-dir>/yaml_map_export).")
    parser.add_argument(
        "--batch-results-root",
        default=None,
        help="Root directory for batch results. Each scenario gets a subdirectory named after the scenario folder."
    )
    parser.add_argument(
        "--generate-videos",
        action="store_true",
        help="Generate videos from captured images after each scenario simulation."
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=10,
        help="Frames per second for generated videos (default: 10)."
    )
    parser.add_argument(
        "--video-resize-factor",
        type=int,
        default=2,
        help="Resize factor for generated videos (default: 2). Passed to gen_video.py --resize-factor."
    )
    parser.add_argument("--dt", type=float, default=0.1, help="Timestep spacing in seconds.")
    parser.add_argument(
        "--maximize-safe-early-spawn",
        dest="maximize_safe_early_spawn",
        action="store_true",
        help="Advance late-detected actors to the earliest safe spawn times (default: on).",
    )
    parser.add_argument(
        "--no-maximize-safe-early-spawn",
        dest="maximize_safe_early_spawn",
        action="store_false",
        help="Disable early-spawn timing optimization.",
    )
    parser.set_defaults(maximize_safe_early_spawn=True)
    parser.add_argument(
        "--early-spawn-safety-margin",
        type=float,
        default=0.25,
        help="Extra safety margin (m) for early-spawn interference checks (default: 0.25).",
    )
    parser.add_argument(
        "--maximize-safe-late-despawn",
        dest="maximize_safe_late_despawn",
        action="store_true",
        help="Extend actor lifetimes toward scenario horizon when safe (default: on).",
    )
    parser.add_argument(
        "--no-maximize-safe-late-despawn",
        dest="maximize_safe_late_despawn",
        action="store_false",
        help="Disable late-despawn timing optimization.",
    )
    parser.set_defaults(maximize_safe_late_despawn=True)
    parser.add_argument(
        "--late-despawn-safety-margin",
        type=float,
        default=0.25,
        help="Extra safety margin (m) for late-despawn hold checks (default: 0.25).",
    )
    parser.add_argument(
        "--timing-optimization-report",
        default=None,
        help="Optional JSON path for detailed timing optimization report.",
    )
    parser.add_argument(
        "--lane-change-filter",
        dest="lane_change_filter",
        action="store_true",
        help="Stabilize lane snapping with hysteresis to suppress noise-driven rapid lane flips (default: on).",
    )
    parser.add_argument(
        "--no-lane-change-filter",
        dest="lane_change_filter",
        action="store_false",
        help="Disable lane-change stabilization filter.",
    )
    parser.set_defaults(lane_change_filter=True)
    parser.add_argument(
        "--lane-change-confirm-window",
        type=int,
        default=5,
        help="Look-ahead window (frames) for confirming lane changes (default: 5).",
    )
    parser.add_argument(
        "--lane-change-confirm-votes",
        type=int,
        default=3,
        help="Minimum supporting observations in confirm window to accept a lane change (default: 3).",
    )
    parser.add_argument(
        "--lane-change-cooldown-frames",
        type=int,
        default=3,
        help="Minimum cooldown frames after a lane change before accepting another weak change (default: 3).",
    )
    parser.add_argument(
        "--lane-change-endpoint-guard-frames",
        type=int,
        default=4,
        help="Extra-guard frames near trajectory start/end against spurious lane changes (default: 4).",
    )
    parser.add_argument(
        "--lane-change-endpoint-extra-votes",
        type=int,
        default=1,
        help="Additional votes required for lane changes near endpoints (default: 1).",
    )
    parser.add_argument(
        "--lane-change-min-improvement-m",
        type=float,
        default=0.2,
        help="Minimum lane-distance improvement (m) that can justify a change with moderate evidence (default: 0.2).",
    )
    parser.add_argument(
        "--lane-change-keep-lane-max-dist",
        type=float,
        default=3.0,
        help="If current lane projection distance exceeds this (m), allow switching more readily (default: 3.0).",
    )
    parser.add_argument(
        "--lane-change-short-run-max",
        type=int,
        default=2,
        help="Max internal run length (frames) considered jitter and collapsed in post-filter pass (default: 2).",
    )
    parser.add_argument(
        "--lane-change-endpoint-short-run",
        type=int,
        default=2,
        help="Max start/end run length (frames) considered jitter and collapsed (default: 2).",
    )
    parser.add_argument(
        "--lane-snap-top-k",
        type=int,
        default=8,
        help="Number of candidate lanes considered per point for stabilized snapping (default: 8).",
    )
    parser.add_argument(
        "--vehicle-forbidden-lane-types",
        default="2",
        help=(
            "Comma/space-separated lane types that motor vehicles can never snap to "
            "(default: 2)."
        ),
    )
    parser.add_argument(
        "--vehicle-parked-only-lane-types",
        default="3",
        help=(
            "Comma/space-separated lane types only parked motor vehicles may snap to "
            "(default: 3)."
        ),
    )
    parser.add_argument(
        "--parked-net-disp-max-m",
        type=float,
        default=1.0,
        help="Parked detection threshold: max start-to-end displacement (m) (default: 1.0).",
    )
    parser.add_argument(
        "--parked-radius-p90-max-m",
        type=float,
        default=1.1,
        help="Parked detection threshold: max p90 radius around median pose (m) (default: 1.1).",
    )
    parser.add_argument(
        "--parked-radius-max-m",
        type=float,
        default=2.0,
        help="Parked detection threshold: max radius around median pose (m) (default: 2.0).",
    )
    parser.add_argument(
        "--parked-p95-step-max-m",
        type=float,
        default=0.55,
        help="Parked detection threshold: max p95 frame-to-frame step (m) (default: 0.55).",
    )
    parser.add_argument(
        "--parked-max-from-start-m",
        type=float,
        default=1.8,
        help="Parked detection threshold: max distance from first pose (m) (default: 1.8).",
    )
    parser.add_argument(
        "--parked-large-step-threshold-m",
        type=float,
        default=0.6,
        help="Parked detection: step size considered a large jump (m) (default: 0.6).",
    )
    parser.add_argument(
        "--parked-large-step-max-ratio",
        type=float,
        default=0.08,
        help="Parked detection: max fraction of large-jump steps (default: 0.08).",
    )
    parser.add_argument(
        "--parked-robust-cluster",
        dest="parked_robust_cluster",
        action="store_true",
        help="Enable robust parked detection using dominant inlier cluster (default: on).",
    )
    parser.add_argument(
        "--no-parked-robust-cluster",
        dest="parked_robust_cluster",
        action="store_false",
        help="Disable robust inlier-cluster parked detection.",
    )
    parser.set_defaults(parked_robust_cluster=True)
    parser.add_argument(
        "--parked-robust-cluster-eps-m",
        type=float,
        default=0.8,
        help="Robust parked detection: inlier cluster radius (m) (default: 0.8).",
    )
    parser.add_argument(
        "--parked-robust-min-inlier-ratio",
        type=float,
        default=0.8,
        help="Robust parked detection: minimum inlier ratio (default: 0.8).",
    )
    parser.add_argument(
        "--parked-robust-max-outlier-run",
        type=int,
        default=3,
        help="Robust parked detection: max contiguous outlier frames (default: 3).",
    )
    parser.add_argument(
        "--parked-robust-min-points",
        type=int,
        default=6,
        help="Robust parked detection: minimum trajectory points (default: 6).",
    )
    parser.add_argument(
        "--id-merge-distance-m",
        type=float,
        default=8.0,
        help=(
            "When same numeric actor id appears in multiple subdirs, trajectories farther than this "
            "distance are split into separate actors (default: 8.0). Set <=0 to preserve legacy merging."
        ),
    )
    parser.add_argument(
        "--cross-id-dedup",
        dest="cross_id_dedup",
        action="store_true",
        help="Merge near-identical tracks even if their actor IDs differ (default: on).",
    )
    parser.add_argument(
        "--no-cross-id-dedup",
        dest="cross_id_dedup",
        action="store_false",
        help="Disable cross-ID overlap deduplication.",
    )
    parser.set_defaults(cross_id_dedup=True)
    parser.add_argument(
        "--cross-id-dedup-max-median-dist-m",
        type=float,
        default=1.2,
        help="Max median XY distance for cross-ID dedup (default: 1.2m).",
    )
    parser.add_argument(
        "--cross-id-dedup-max-p90-dist-m",
        type=float,
        default=2.0,
        help="Max p90 XY distance for cross-ID dedup (default: 2.0m).",
    )
    parser.add_argument(
        "--cross-id-dedup-max-median-yaw-diff-deg",
        type=float,
        default=35.0,
        help="Max median yaw difference for non-walker cross-ID dedup (default: 35 deg).",
    )
    parser.add_argument(
        "--cross-id-dedup-min-common-points",
        type=int,
        default=8,
        help="Minimum overlapping timesteps required for cross-ID dedup (default: 8).",
    )
    parser.add_argument(
        "--cross-id-dedup-min-overlap-each",
        type=float,
        default=0.30,
        help="Minimum overlap ratio for each track in a dedup pair (default: 0.30).",
    )
    parser.add_argument(
        "--cross-id-dedup-min-overlap-any",
        type=float,
        default=0.75,
        help="Minimum overlap ratio for at least one track in a dedup pair (default: 0.75).",
    )
    parser.add_argument(
        "--map-pkl",
        action="append",
        default=[],
        help="Map pickle path (repeatable). If omitted, uses corridors + intersection defaults.",
    )
    parser.add_argument(
        "--map-selection-sample-count",
        type=int,
        default=1200,
        help="Max sampled trajectory points for selecting best map (default: 1200).",
    )
    parser.add_argument(
        "--map-selection-bbox-margin",
        type=float,
        default=20.0,
        help="BBox margin (m) for map selection outside-penalty (default: 20).",
    )
    parser.add_argument(
        "--carla-map-layer",
        dest="carla_map_layer",
        action="store_true",
        help="Include aligned CARLA map polylines as an optional HTML map layer (default: on).",
    )
    parser.add_argument(
        "--no-carla-map-layer",
        dest="carla_map_layer",
        action="store_false",
        help="Disable CARLA map layer in output payload/HTML.",
    )
    parser.add_argument(
        "--skip-map-snap-compute",
        action="store_true",
        help="Skip per-frame CARLA projection (slow step). CARLA layer and lane correspondence cache still load. 'Use map snapped poses' won't work.",
    )
    parser.set_defaults(carla_map_layer=True)
    parser.add_argument(
        "--carla-map-cache",
        default="/data2/marco/CoLMDriver/v2xpnp/map/carla_map_cache.pkl",
        help="Path to cached CARLA map polylines pickle (default: v2xpnp/map/carla_map_cache.pkl).",
    )
    parser.add_argument(
        "--carla-map-offset-json",
        default="/data2/marco/CoLMDriver/v2xpnp/map/ucla_map_offset_carla.json",
        help="CARLA->V2XPNP alignment JSON (tx/ty/theta/flip_y/scale) for CARLA map layer.",
    )
    parser.add_argument(
        "--carla-map-image-cache",
        default="/data2/marco/CoLMDriver/v2xpnp/map/carla_topdown_cache.jpg",
        help="Path to cached CARLA top-down JPEG image (default: v2xpnp/map/carla_topdown_cache.jpg).",
    )
    parser.add_argument(
        "--carla-host",
        default="localhost",
        help="CARLA server hostname for top-down image capture (default: localhost).",
    )
    parser.add_argument(
        "--carla-port",
        type=int,
        default=2005,
        help="CARLA server RPC port for top-down image capture (default: 2005).",
    )
    parser.add_argument(
        "--start-carla",
        default=True,
        action="store_true",
        help="Automatically launch a local CARLA server before GRP alignment/CARLA operations.",
    )
    parser.add_argument(
        "--carla-root",
        type=str,
        default=None,
        help=(
            "Path to CARLA installation directory containing CarlaUE4.sh. "
            "Defaults to CARLA_ROOT env var or ./carla912."
        ),
    )
    parser.add_argument(
        "--carla-arg",
        action="append",
        default=[],
        help="Extra arguments to pass to CarlaUE4.sh (repeatable).",
    )
    parser.add_argument(
        "--carla-port-tries",
        type=int,
        default=CARLA_PORT_TRIES,
        help=f"How many ports to try if the desired CARLA port is already in use (default: {CARLA_PORT_TRIES}).",
    )
    parser.add_argument(
        "--carla-port-step",
        type=int,
        default=CARLA_PORT_STEP,
        help=f"Port increment when searching for a free CARLA port (default: {CARLA_PORT_STEP}).",
    )
    parser.add_argument(
        "--carla-map-name",
        default="ucla_v2",
        help="CARLA map name to load before capturing top-down image (default: ucla_v2).",
    )
    parser.add_argument(
        "--capture-carla-image",
        dest="capture_carla_image",
        action="store_true",
        help="Attempt to capture a top-down image from a running CARLA server if no cache exists (default: on).",
    )
    parser.add_argument(
        "--no-capture-carla-image",
        dest="capture_carla_image",
        action="store_false",
        help="Disable CARLA image capture; only use cached image.",
    )
    parser.set_defaults(capture_carla_image=True)
    parser.add_argument(
        "--lane-correspondence",
        dest="lane_correspondence",
        action="store_true",
        help="Build robust V2XPNP-lane to CARLA-line correspondence and snap actors accordingly (default: on).",
    )
    parser.add_argument(
        "--no-lane-correspondence",
        dest="lane_correspondence",
        action="store_false",
        help="Disable lane correspondence and CARLA-lane actor snapping.",
    )
    parser.set_defaults(lane_correspondence=True)
    parser.add_argument(
        "--lane-correspondence-driving-types",
        default="1",
        help="Comma/space lane types treated as driving lanes for one-to-one lane correspondence (default: 1).",
    )
    parser.add_argument(
        "--lane-correspondence-top-k",
        type=int,
        default=28,
        help="Candidate CARLA lines examined per V2 lane during correspondence (default: 28).",
    )
    parser.add_argument(
        "--lane-correspondence-cache-dir",
        type=str,
        default="__script_dir__",
        help="Directory for caching lane correspondence results (default: alongside this script). Use '__output_dir__' to stash next to scenario output.",
    )
    parser.add_argument(
        "--snap-to-map",
        dest="snap_to_map",
        action="store_true",
        help="Use map-matched coordinates as rendered/exported pose (default: on).",
    )
    parser.add_argument(
        "--no-snap-to-map",
        dest="snap_to_map",
        action="store_false",
        help="Keep raw YAML coordinates as rendered pose while still reporting matched lanes.",
    )
    parser.set_defaults(snap_to_map=True)
    parser.add_argument(
        "--map-max-points-per-line",
        type=int,
        default=600,
        help="Max points per lane polyline in HTML payload (default: 600).",
    )

    # --- GRP trajectory alignment ---
    parser.add_argument(
        "--grp-align",
        dest="grp_align",
        action="store_true",
        help="Enable GRP-aware trajectory alignment via CARLA GlobalRoutePlanner (default: off).",
    )
    parser.add_argument(
        "--no-grp-align",
        dest="grp_align",
        action="store_false",
        help="Disable GRP trajectory alignment.",
    )
    parser.set_defaults(grp_align=True)
    parser.add_argument(
        "--grp-snap-radius",
        type=float,
        default=2.5,
        help="GRP alignment: search radius for CARLA waypoint candidates (default: 2.5m).",
    )
    parser.add_argument(
        "--grp-snap-k",
        type=int,
        default=6,
        help="GRP alignment: max candidates per input waypoint (default: 6).",
    )
    parser.add_argument(
        "--grp-heading-thresh",
        type=float,
        default=40.0,
        help="GRP alignment: yaw tolerance for candidate filtering in degrees (default: 40).",
    )
    parser.add_argument(
        "--grp-lane-change-penalty",
        type=float,
        default=50.0,
        help="GRP alignment: DP cost penalty for switching lanes (default: 50).",
    )
    parser.add_argument(
        "--grp-sampling-resolution",
        type=float,
        default=2.0,
        help="GRP alignment: GlobalRoutePlanner sampling resolution in meters (default: 2.0).",
    )

    # --- Walker sidewalk compression and stabilization ---
    parser.add_argument(
        "--walker-sidewalk-compression",
        dest="walker_sidewalk_compression",
        action="store_true",
        help="Enable walker sidewalk compression and spawn stabilization (default: on).",
    )
    parser.add_argument(
        "--no-walker-sidewalk-compression",
        dest="walker_sidewalk_compression",
        action="store_false",
        help="Disable walker sidewalk compression.",
    )
    parser.set_defaults(walker_sidewalk_compression=True)
    parser.add_argument(
        "--walker-lane-spacing-m",
        type=float,
        default=None,
        help="Lane spacing for sidewalk geometry (default: auto-calibrate from map).",
    )
    parser.add_argument(
        "--walker-sidewalk-start-factor",
        type=float,
        default=0.5,
        help="Sidewalk start distance = factor * lane_spacing (default: 0.5).",
    )
    parser.add_argument(
        "--walker-sidewalk-outer-factor",
        type=float,
        default=3.0,
        help="Sidewalk outer band factor k: y = k * sidewalk_start_distance (default: 3.0, range 2-4).",
    )
    parser.add_argument(
        "--walker-compression-target-band-m",
        type=float,
        default=2.5,
        help="Target sidewalk width after compression in meters (default: 2.5).",
    )
    parser.add_argument(
        "--walker-compression-power",
        type=float,
        default=1.5,
        help="Nonlinear compression power (>1 = stronger compression at distance) (default: 1.5).",
    )
    parser.add_argument(
        "--walker-min-spawn-separation-m",
        type=float,
        default=0.8,
        help="Minimum separation between walker spawn positions (default: 0.8).",
    )
    parser.add_argument(
        "--walker-radius-m",
        type=float,
        default=0.35,
        help="Approximate walker collision radius (default: 0.35).",
    )
    parser.add_argument(
        "--walker-crossing-road-ratio-thresh",
        type=float,
        default=0.15,
        help="Road occupancy ratio threshold for crossing classification (default: 0.15).",
    )
    parser.add_argument(
        "--walker-crossing-lateral-thresh-m",
        type=float,
        default=4.0,
        help="Lateral traversal distance indicating crossing behavior (default: 4.0).",
    )
    parser.add_argument(
        "--walker-road-presence-min-frames",
        type=int,
        default=5,
        help="Min sustained frames in road region for crossing classification (default: 5).",
    )
    parser.add_argument(
        "--walker-max-lateral-offset-m",
        type=float,
        default=3.0,
        help="Maximum allowed lateral offset from compression (default: 3.0).",
    )
    # --- CARLA Route Export Options ---
    parser.add_argument(
        "--export-carla-routes",
        default=True,
        action="store_true",
        help="Export CARLA-compatible XML route files for use with run_custom_eval.py",
    )
    parser.add_argument(
        "--carla-routes-dir",
        type=str,
        default=None,
        help="Output directory for CARLA routes (default: <output-dir>/carla_routes/)",
    )
    parser.add_argument(
        "--carla-town",
        type=str,
        default="ucla_v2",
        help="CARLA town name for route XML files (default: ucla_v2)",
    )
    parser.add_argument(
        "--carla-route-id",
        type=str,
        default="0",
        help="Route ID to use in CARLA route XML files (default: 0)",
    )
    parser.add_argument(
        "--carla-actor-control-mode",
        choices=("policy", "replay"),
        default="policy",
        help=(
            "Control mode for NPC vehicles. 'policy' uses CARLA's AI planners (realistic driving). "
            "'replay' uses exact logged transform replay. (default: policy)"
        ),
    )
    parser.add_argument(
        "--carla-walker-control-mode",
        choices=("policy", "replay"),
        default="policy",
        help=(
            "Control mode for walkers. 'policy' uses CARLA's walker AI. "
            "'replay' uses exact logged transform replay. (default: policy)"
        ),
    )
    parser.add_argument(
        "--carla-encode-timing",
        action="store_true",
        default=True,
        help="Include timing information in CARLA route waypoints (default: enabled)",
    )
    parser.add_argument(
        "--no-carla-encode-timing",
        dest="carla_encode_timing",
        action="store_false",
        help="Disable timing information in CARLA route waypoints",
    )
    parser.add_argument(
        "--carla-snap-to-road",
        action="store_true",
        default=False,
        help="Enable snap_to_road in CARLA route files (default: disabled for accuracy)",
    )
    parser.add_argument(
        "--carla-static-spawn-only",
        action="store_true",
        default=False,
        help="For parked vehicles, only output spawn position (no trajectory) for efficiency",
    )
    # --- Run CARLA Scenario After Export ---
    parser.add_argument(
        "--run-custom-eval",
        default=True,
        action="store_true",
        help="After exporting CARLA routes, automatically run the scenario using tools/run_custom_eval.py",
    )
    parser.add_argument(
        "--eval-planner",
        type=str,
        default="log-replay",
        help="Planner for run_custom_eval (default: 'log-replay' for exact trajectory replay).",
    )
    parser.add_argument(
        "--eval-port",
        type=int,
        default=None,
        help="CARLA port for run_custom_eval (default: same as --carla-port)",
    )
    parser.add_argument(
        "--eval-overwrite",
        action="store_true",
        default=True,
        help="Overwrite existing evaluation results (default: enabled)",
    )
    parser.add_argument(
        "--eval-timeout-factor",
        type=float,
        default=2.0,
        help="Timeout multiplier for scenario duration (default: 2.0)",
    )
    parser.add_argument(
        "--capture-logreplay-images",
        default=True,
        action="store_true",
        help="Save camera images during log-replay evaluation (passed to run_custom_eval.py)",
    )
    parser.add_argument(
        "--capture-every-sensor-frame",
        action="store_true",
        help="Save RGB/BEV images for every sensor frame during evaluation (passed to run_custom_eval.py)",
    )
    return parser.parse_args()


# =============================================================================
# CARLA Route Export Module
# =============================================================================
# This module exports trajectories to CARLA-compatible XML route files
# that work with run_custom_eval.py and the CARLA leaderboard.
# =============================================================================

def _write_carla_route_xml(
    path: Path,
    route_id: str,
    role: str,
    town: str,
    waypoints: List[Waypoint],
    times: Optional[List[float]] = None,
    snap_to_road: bool = False,
    control_mode: str = "policy",
    target_speed_mps: Optional[float] = None,
    model: Optional[str] = None,
    speeds: Optional[List[float]] = None,
) -> None:
    """Write a CARLA-compatible route XML file with timing and control mode."""
    root = ET.Element("routes")
    route_attrs = {
        "id": str(route_id),
        "town": town,
        "role": role,
        "snap_to_road": "true" if snap_to_road else "false",
    }
    if control_mode:
        route_attrs["control_mode"] = control_mode
    if target_speed_mps is not None and target_speed_mps > 0:
        route_attrs["target_speed"] = f"{target_speed_mps:.2f}"
    if model:
        route_attrs["model"] = model
    
    route_elem = ET.SubElement(root, "route", route_attrs)
    
    for idx, wp in enumerate(waypoints):
        attrs = {
            "x": f"{float(wp.x):.6f}",
            "y": f"{float(wp.y):.6f}",
            "z": f"{float(wp.z):.6f}",
            "yaw": f"{float(wp.yaw):.6f}",
            "pitch": "360.000000",
            "roll": "0.000000",
        }
        if times is not None and idx < len(times):
            try:
                attrs["time"] = f"{float(times[idx]):.6f}"
            except (TypeError, ValueError):
                pass
        if speeds is not None and idx < len(speeds):
            try:
                attrs["speed"] = f"{float(speeds[idx]):.4f}"
            except (TypeError, ValueError):
                pass
        ET.SubElement(route_elem, "waypoint", attrs)
    
    tree = ET.ElementTree(root)
    # ET.indent() requires Python 3.9+; use fallback for older versions
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    else:
        _indent_xml(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    """Add indentation to XML tree for pretty printing (Python <3.9 fallback)."""
    indent_str = "\n" + "  " * level
    if len(elem):  # has children
        if not elem.text or not elem.text.strip():
            elem.text = indent_str + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent_str
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent_str
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent_str


def _compute_target_speed(
    waypoints: List[Waypoint],
    times: Optional[List[float]],
    default_dt: float = 0.1,
    window_seconds: float = 2.0,
    moving_threshold_mps: float = 0.3,
) -> float:
    """Compute a robust representative cruising speed from a trajectory.

    The old approach summed consecutive-point distances, which inflated speed
    dramatically due to GPS/lidar tracking noise (e.g. 18 m real displacement
    computed as 280 m cumulative path → 27 m/s instead of 1.8 m/s).

    This version uses **displacement-based sliding windows**:
      1. For each point, look ahead by *window_seconds* and compute the
         straight-line displacement divided by elapsed time.  This naturally
         cancels out high-frequency jitter.
      2. Discard windows where the vehicle is essentially stationary
         (speed < *moving_threshold_mps*).
      3. Return the **75th-percentile** of the remaining window speeds,
         which represents the vehicle's typical cruising speed while
         filtering out acceleration/deceleration transients.

    Falls back to total displacement / total duration when no moving
    windows are found (very slow vehicles).
    """
    n = len(waypoints)
    if n < 2:
        return 0.0

    # Build time array
    if times and len(times) >= n:
        t = [float(ti) for ti in times[:n]]
    else:
        t = [float(default_dt) * i for i in range(n)]

    total_dt = t[-1] - t[0]
    if total_dt < 1e-6:
        return 0.0

    # --- Sliding-window displacement speeds ---
    window_speeds: List[float] = []
    j = 0  # leading pointer
    for i in range(n):
        # Advance j so that t[j] - t[i] >= window_seconds
        while j < n - 1 and (t[j] - t[i]) < window_seconds:
            j += 1
        dt_w = t[j] - t[i]
        if dt_w < 1e-6:
            continue
        dx = float(waypoints[j].x) - float(waypoints[i].x)
        dy = float(waypoints[j].y) - float(waypoints[i].y)
        disp = math.hypot(dx, dy)
        window_speeds.append(disp / dt_w)

    # Keep only windows where the vehicle is actually moving
    moving_speeds = [s for s in window_speeds if s >= moving_threshold_mps]

    if moving_speeds:
        moving_speeds.sort()
        # 75th percentile — robust cruising speed
        idx_75 = int(len(moving_speeds) * 0.75)
        idx_75 = min(idx_75, len(moving_speeds) - 1)
        return moving_speeds[idx_75]

    # Fallback: total displacement / total time (vehicle barely moved)
    dx_total = float(waypoints[-1].x) - float(waypoints[0].x)
    dy_total = float(waypoints[-1].y) - float(waypoints[0].y)
    return math.hypot(dx_total, dy_total) / total_dt


def _sanitize_trajectory(
    waypoints: List[Waypoint],
    times: List[float],
    default_dt: float = 0.1,
    max_plausible_speed_mps: float = 40.0,
) -> Tuple[List[Waypoint], List[float]]:
    """Remove tracking-noise artefacts (zigzag / ID-swap / teleportation).

    The tracker sometimes assigns the same ID to two nearby vehicles on
    alternating frames, producing impossible ~10 m jumps at 10 Hz (≡ 100 m/s).
    This function detects such frames and replaces them by holding the last
    good position, then does a light median-filter pass to smooth residual
    jitter.

    Parameters
    ----------
    waypoints : list of Waypoint
        Raw trajectory in CARLA coordinates.
    times : list of float
        Corresponding timestamps.
    max_plausible_speed_mps : float
        Any instantaneous displacement/dt above this is treated as a glitch.
        Default 40 m/s ≈ 144 km/h — generous enough for highway traffic.

    Returns
    -------
    (cleaned_wps, cleaned_times) with the same length as the input.
    """
    n = len(waypoints)
    if n < 3:
        return list(waypoints), list(times)

    dt = default_dt

    # ---- Pass 1: flag impossible jumps ----
    xs = [float(wp.x) for wp in waypoints]
    ys = [float(wp.y) for wp in waypoints]
    yaws = [float(wp.yaw) for wp in waypoints]
    zs = [float(wp.z) for wp in waypoints]

    bad = [False] * n  # True → this frame is an artefact
    for i in range(1, n):
        dti = (times[i] - times[i - 1]) if i < len(times) else dt
        if dti < 1e-6:
            dti = dt
        dx = xs[i] - xs[i - 1]
        dy = ys[i] - ys[i - 1]
        dist = math.hypot(dx, dy)
        speed = dist / dti
        if speed > max_plausible_speed_mps:
            bad[i] = True

    # Also flag zigzag patterns: if i-1 is fine, i is bad, i+1 returns to
    # roughly the same position as i-1 → classic ID-swap.  Mark i as bad.
    for i in range(1, n - 1):
        if bad[i]:
            continue
        if not bad[i - 1] and not bad[i + 1]:
            # Check if i is a single-frame outlier
            d_prev = math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])
            d_next = math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
            d_skip = math.hypot(xs[i + 1] - xs[i - 1], ys[i + 1] - ys[i - 1])
            # If skipping this frame produces a much shorter path → outlier
            if d_prev + d_next > 3 * (d_skip + 0.1):
                dti = (times[i] - times[i - 1]) if i < len(times) else dt
                if dti < 1e-6:
                    dti = dt
                if d_prev / dti > max_plausible_speed_mps * 0.5:
                    bad[i] = True

    n_bad = sum(bad)
    if n_bad > 0:
        # Replace bad frames by holding the last good position
        for i in range(1, n):
            if bad[i]:
                # Find the previous good frame
                j = i - 1
                while j > 0 and bad[j]:
                    j -= 1
                xs[i] = xs[j]
                ys[i] = ys[j]
                zs[i] = zs[j]
                yaws[i] = yaws[j]

    # ---- Pass 2: Lightweight median filter (window=3) on x,y ----
    # Suppresses residual 1-frame jitter without distorting turns.
    fxs = list(xs)
    fys = list(ys)
    for i in range(1, n - 1):
        vals_x = sorted([xs[i - 1], xs[i], xs[i + 1]])
        vals_y = sorted([ys[i - 1], ys[i], ys[i + 1]])
        fxs[i] = vals_x[1]
        fys[i] = vals_y[1]

    # ---- Pass 3: Remove consecutive duplicate positions (dedup) ----
    # After fixing artefacts, many consecutive frames have identical (x,y).
    # Keep only the first and last of each constant run to avoid WaypointFollower
    # getting stuck trying to reach an already-reached point.
    clean_wps: List[Waypoint] = []
    clean_times: List[float] = []
    i = 0
    while i < n:
        # Start of a run of identical positions
        j = i + 1
        while j < n and abs(fxs[j] - fxs[i]) < 0.05 and abs(fys[j] - fys[i]) < 0.05:
            j += 1
        # Keep first frame of the run
        clean_wps.append(Waypoint(x=fxs[i], y=fys[i], z=zs[i], yaw=yaws[i]))
        clean_times.append(times[i])
        # If the run spans >2 frames, also keep the last (to preserve timing)
        if j - i > 2:
            last = j - 1
            clean_wps.append(Waypoint(x=fxs[last], y=fys[last], z=zs[last], yaw=yaws[last]))
            clean_times.append(times[last])
        elif j - i == 2:
            # Keep both frames (they're not truly duplicates, just close)
            clean_wps.append(Waypoint(x=fxs[i + 1], y=fys[i + 1], z=zs[i + 1], yaw=yaws[i + 1]))
            clean_times.append(times[i + 1])
        i = j

    if len(clean_wps) < 2 and n >= 2:
        # Degenerate: entire trajectory was one position; keep first and last
        clean_wps = [Waypoint(x=fxs[0], y=fys[0], z=zs[0], yaw=yaws[0]),
                     Waypoint(x=fxs[-1], y=fys[-1], z=zs[-1], yaw=yaws[-1])]
        clean_times = [times[0], times[-1]]

    return clean_wps, clean_times


def _trajectory_instability_score(
    waypoints: Sequence[Waypoint],
    jump_threshold_m: float = 2.8,
    yaw_flip_threshold_deg: float = 120.0,
) -> float:
    """Heuristic instability score for export-track source selection."""
    if len(waypoints) < 2:
        return 0.0
    score = 0.0
    for i in range(1, len(waypoints)):
        dx = float(waypoints[i].x) - float(waypoints[i - 1].x)
        dy = float(waypoints[i].y) - float(waypoints[i - 1].y)
        if math.hypot(dx, dy) > float(jump_threshold_m):
            score += 1.0
        if _grp_yaw_diff_deg(float(waypoints[i].yaw), float(waypoints[i - 1].yaw)) > float(yaw_flip_threshold_deg):
            score += 2.0
    return float(score)


def _compute_per_waypoint_speeds(
    waypoints: List[Waypoint],
    times: Optional[List[float]],
    default_dt: float = 0.1,
    window_seconds: float = 2.0,
    min_speed: float = 0.0,
    max_speed: float = 30.0,
    smooth_passes: int = 3,
) -> List[float]:
    """Compute a smoothed per-waypoint speed profile from trajectory + timing.

    For each waypoint the speed is estimated via forward *and* backward
    displacement windows of *window_seconds*, then averaged.  This cancels
    high-frequency tracking noise while preserving real acceleration and
    deceleration.  A lightweight rolling-average smoother is applied
    afterwards to remove any residual spikes.

    Returns a list of the same length as *waypoints* with speed in m/s.
    """
    n = len(waypoints)
    if n == 0:
        return []
    if n == 1:
        return [0.0]

    # Build time array
    if times and len(times) >= n:
        t = [float(ti) for ti in times[:n]]
    else:
        t = [float(default_dt) * i for i in range(n)]

    # --- Forward-window displacement speed ---
    fwd: List[float] = [0.0] * n
    j = 0
    for i in range(n):
        while j < n - 1 and (t[j] - t[i]) < window_seconds:
            j += 1
        dt_w = t[j] - t[i]
        if dt_w > 1e-6:
            dx = float(waypoints[j].x) - float(waypoints[i].x)
            dy = float(waypoints[j].y) - float(waypoints[i].y)
            fwd[i] = math.hypot(dx, dy) / dt_w

    # --- Backward-window displacement speed ---
    bwd: List[float] = [0.0] * n
    k = n - 1
    for i in range(n - 1, -1, -1):
        while k > 0 and (t[i] - t[k]) < window_seconds:
            k -= 1
        dt_w = t[i] - t[k]
        if dt_w > 1e-6:
            dx = float(waypoints[i].x) - float(waypoints[k].x)
            dy = float(waypoints[i].y) - float(waypoints[k].y)
            bwd[i] = math.hypot(dx, dy) / dt_w

    # Average forward and backward estimates
    raw: List[float] = []
    for i in range(n):
        if fwd[i] > 0 and bwd[i] > 0:
            raw.append(0.5 * (fwd[i] + bwd[i]))
        elif fwd[i] > 0:
            raw.append(fwd[i])
        elif bwd[i] > 0:
            raw.append(bwd[i])
        else:
            raw.append(0.0)

    # Clamp to [min_speed, max_speed]
    raw = [max(min_speed, min(max_speed, s)) for s in raw]

    # --- Rolling average smoother (kernel size 5) ---
    speeds = list(raw)
    k_half = 2  # kernel radius → window of 5
    for _ in range(smooth_passes):
        smoothed = list(speeds)
        for i in range(n):
            lo = max(0, i - k_half)
            hi = min(n, i + k_half + 1)
            smoothed[i] = sum(speeds[lo:hi]) / (hi - lo)
        speeds = smoothed

    # Final clamp
    speeds = [max(min_speed, min(max_speed, s)) for s in speeds]
    return speeds


def _stabilize_initial_route_yaw_for_export(
    waypoints: List[Waypoint],
    max_lookahead: int = 50,
    min_displacement_m: float = 1.0,
    max_prefix_frames: int = 50,
    stationary_prefix_tol_m: float = 0.8,
    apply_if_diff_deg: float = 35.0,
) -> List[Waypoint]:
    """Adjust yaws during stationary/slow-start period to avoid U-turns.

    This fixes cases where:
    1. Initial route waypoints have yaw pointing opposite to movement direction
    2. Tracking jitter causes yaw to flip 180° during stationary periods
    
    The function computes the actual movement direction from the first significant
    displacement, then corrects all yaws in the stationary prefix that differ
    by more than apply_if_diff_deg from that direction.
    """
    if len(waypoints) < 2:
        return waypoints

    out = [Waypoint(x=float(w.x), y=float(w.y), z=float(w.z), yaw=float(w.yaw)) for w in waypoints]
    w0 = out[0]

    # Find the first waypoint with significant displacement to determine movement direction
    end_idx = min(max_lookahead, len(out) - 1)
    ref_idx = -1
    ref_dx = 0.0
    ref_dy = 0.0
    for j in range(1, end_idx + 1):
        dx = float(out[j].x) - float(w0.x)
        dy = float(out[j].y) - float(w0.y)
        if math.hypot(dx, dy) >= min_displacement_m:
            ref_idx = j
            ref_dx = dx
            ref_dy = dy
            break

    if ref_idx < 0:
        return out

    spawn_heading = _normalize_yaw_deg(math.degrees(math.atan2(ref_dy, ref_dx)))

    # Fix all waypoints in the stationary prefix (not just first few)
    # A waypoint is "stationary" if it hasn't moved far from the spawn point
    prefix_end = min(max_prefix_frames, len(out))
    for i in range(prefix_end):
        di = math.hypot(float(out[i].x) - float(w0.x), float(out[i].y) - float(w0.y))
        if di > stationary_prefix_tol_m:
            # Once we've moved significantly, stop fixing yaws
            break
        yaw_diff = _grp_yaw_diff_deg(float(out[i].yaw), spawn_heading)
        if yaw_diff >= apply_if_diff_deg:
            out[i] = Waypoint(
                x=float(out[i].x),
                y=float(out[i].y),
                z=float(out[i].z),
                yaw=float(spawn_heading),
            )

    return out

def _spread_parked_vehicles(
    actor_tracks: List[Dict[str, object]],
    min_gap_m: float = 3.0,
) -> List[Dict[str, object]]:
    """
    Spread out parked/static vehicles that are too close together (bumper-to-bumper).
    
    For vehicles that are stationary (parked) and share similar headings (same lane),
    this function adds spacing along their heading axis to avoid collisions.
    
    Args:
        actor_tracks: List of actor track dictionaries
        min_gap_m: Minimum gap (in meters) between parked vehicles
    
    Returns:
        Modified actor_tracks with adjusted positions
    """
    if not actor_tracks:
        return actor_tracks
    
    # Identify parked/static vehicles with their positions
    parked_info = []
    for idx, track in enumerate(actor_tracks):
        is_parked = bool(track.get("parked_vehicle", False))
        role = str(track.get("role", "")).lower()
        model = str(track.get("model", "")).lower()
        frames = track.get("frames", [])
        
        # Check if it's a vehicle (not walker)
        is_vehicle = model.startswith("vehicle.") or role in ("npc", "static", "parked")
        is_walker = role in ("walker", "cyclist", "pedestrian")
        
        if is_walker:
            continue
        
        # Check if vehicle is stationary based on movement
        if not is_parked and len(frames) >= 2:
            first_f = frames[0]
            last_f = frames[-1]
            fx = float(first_f.get("rx", first_f.get("x", 0)))
            fy = float(first_f.get("ry", first_f.get("y", 0)))
            lx = float(last_f.get("rx", last_f.get("x", 0)))
            ly = float(last_f.get("ry", last_f.get("y", 0)))
            total_movement = math.sqrt((lx - fx) ** 2 + (ly - fy) ** 2)
            if total_movement < 1.5:  # Less than 1.5m total = essentially stationary
                is_parked = True
        
        if is_parked and is_vehicle and frames:
            f = frames[0]
            x = float(f.get("rx", f.get("x", 0)))
            y = float(f.get("ry", f.get("y", 0)))
            yaw = float(f.get("ryaw", f.get("yaw", 0)))
            parked_info.append({
                "idx": idx,
                "x": x,
                "y": y,
                "yaw": yaw,
            })
    
    if len(parked_info) < 2:
        return actor_tracks
    
    # Group by similar heading (within 30 degrees = same lane direction)
    heading_groups: Dict[int, List[dict]] = {}
    for pi in parked_info:
        # Normalize yaw to 0-180 range (treat opposite directions as same lane)
        norm_yaw = pi["yaw"] % 180
        group_key = int(norm_yaw / 30)  # Group by 30-degree buckets
        if group_key not in heading_groups:
            heading_groups[group_key] = []
        heading_groups[group_key].append(pi)
    
    # Process each heading group
    spread_count = 0
    for group_key, group in heading_groups.items():
        if len(group) < 2:
            continue
        
        # Calculate average heading for projection
        avg_yaw = sum(p["yaw"] for p in group) / len(group)
        cos_yaw = math.cos(math.radians(avg_yaw))
        sin_yaw = math.sin(math.radians(avg_yaw))
        
        # Project positions onto heading axis
        for p in group:
            p["proj"] = p["x"] * cos_yaw + p["y"] * sin_yaw
        
        # Sort by projection (position along heading direction)
        group.sort(key=lambda p: p["proj"])
        
        # Spread vehicles that are too close
        for i in range(1, len(group)):
            prev = group[i - 1]
            curr = group[i]
            
            gap = curr["proj"] - prev["proj"]
            
            if gap < min_gap_m:
                shift_needed = min_gap_m - gap + 0.5  # Extra buffer
                
                # Shift current vehicle forward along heading
                new_x = curr["x"] + shift_needed * cos_yaw
                new_y = curr["y"] + shift_needed * sin_yaw
                
                # Update all frames in this track
                orig_idx = curr["idx"]
                frames = actor_tracks[orig_idx].get("frames", [])
                for f in frames:
                    if "rx" in f:
                        f["rx"] = float(f["rx"]) + shift_needed * cos_yaw
                    if "ry" in f:
                        f["ry"] = float(f["ry"]) + shift_needed * sin_yaw
                    if "x" in f:
                        f["x"] = float(f["x"]) + shift_needed * cos_yaw
                    if "y" in f:
                        f["y"] = float(f["y"]) + shift_needed * sin_yaw
                
                # Update curr for next iteration
                curr["proj"] = prev["proj"] + min_gap_m
                curr["x"] = new_x
                curr["y"] = new_y
                spread_count += 1
                
                vid = actor_tracks[orig_idx].get("vid", orig_idx)
                print(f"[CARLA_EXPORT] Spread parked vehicle vid={vid} by {shift_needed:.2f}m to avoid bumper-to-bumper")
    
    if spread_count > 0:
        print(f"[CARLA_EXPORT] Spread {spread_count} parked vehicles to maintain {min_gap_m}m minimum gap")
    
    return actor_tracks


def export_carla_routes(
    out_dir: Path,
    town: str,
    route_id: str,
    ego_tracks: List[Dict[str, object]],
    actor_tracks: List[Dict[str, object]],
    align_cfg: Dict[str, object],
    actor_control_mode: str = "policy",
    walker_control_mode: str = "policy",
    encode_timing: bool = True,
    snap_to_road: bool = False,
    static_spawn_only: bool = False,
    default_dt: float = 0.1,
) -> Dict[str, object]:
    """
    Export ego and actor trajectories to CARLA-compatible XML route files.
    
    Args:
        out_dir: Output directory for CARLA routes
        town: CARLA town name
        route_id: Route ID for XML files
        ego_tracks: List of ego track dictionaries from payload
        actor_tracks: List of actor track dictionaries from payload
        align_cfg: Alignment configuration (for V2XPNP -> CARLA transform)
        actor_control_mode: 'policy' for AI planners or 'replay' for exact replay
        walker_control_mode: 'policy' for AI walkers or 'replay' for exact replay
        encode_timing: Whether to include timing in waypoints
        snap_to_road: Whether to snap actors to road in CARLA
        static_spawn_only: For parked vehicles, only output spawn point
        default_dt: Default time step
    
    Returns:
        Report dictionary with export statistics
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    actors_dir = out_dir / "actors"
    actors_dir.mkdir(parents=True, exist_ok=True)
    
    # Spread out parked vehicles that are too close together (before coordinate transform)
    actor_tracks = _spread_parked_vehicles(actor_tracks, min_gap_m=3.0)
    
    # Coordinate transform parameters (V2XPNP -> CARLA)
    scale = float(align_cfg.get("scale", 1.0))
    theta_deg = float(align_cfg.get("theta_deg", 0.0))
    tx = float(align_cfg.get("tx", 0.0))
    ty = float(align_cfg.get("ty", 0.0))
    flip_y = bool(align_cfg.get("flip_y", False))
    inv_scale = 1.0 / scale if abs(scale) > 1e-12 else 1.0
    
    def v2x_to_carla(x: float, y: float) -> Tuple[float, float]:
        cx, cy = invert_se2((x, y), theta_deg, tx, ty, flip_y=flip_y)
        return cx * inv_scale, cy * inv_scale
    
    def yaw_v2x_to_carla(yaw_v2x: float) -> float:
        adjusted = float(yaw_v2x) - float(theta_deg)
        if flip_y:
            adjusted = -adjusted
        return _normalize_yaw_deg(adjusted)
    
    report: Dict[str, object] = {
        "enabled": True,
        "output_dir": str(out_dir),
        "town": town,
        "route_id": route_id,
        "actor_control_mode": actor_control_mode,
        "walker_control_mode": walker_control_mode,
        "ego_count": 0,
        "npc_count": 0,
        "walker_count": 0,
        "static_count": 0,
        "total_actors": 0,
        "ego_files": [],
        "actor_files": [],
    }
    
    manifest: Dict[str, List[Dict[str, object]]] = {}
    
    # --- Export Ego Routes ---
    ego_entries: List[Dict[str, object]] = []
    for ego_idx, ego_track in enumerate(ego_tracks):
        frames = ego_track.get("frames", [])
        if not frames:
            continue
        
        # Transform to CARLA coordinates using RAW (rx/ry) positions before lane snapping
        # This corresponds to visualization with "Use map-snapped poses" UNCHECKED
        carla_wps: List[Waypoint] = []
        carla_times: List[float] = []
        for f in frames:
            # Use raw rx/ry coordinates (before lane snapping)
            rx = float(f.get("rx", f.get("x", 0)))
            ry = float(f.get("ry", f.get("y", 0)))
            rz = float(f.get("rz", f.get("z", 0)))
            ryaw = float(f.get("ryaw", f.get("yaw", 0)))
            t = float(f.get("t", 0))
            
            cx, cy = v2x_to_carla(rx, ry)
            cyaw = yaw_v2x_to_carla(ryaw)
            carla_wps.append(Waypoint(x=cx, y=cy, z=rz, yaw=cyaw))
            carla_times.append(t)
        
        if not carla_wps:
            continue
        
        ego_xml_name = f"{town.lower()}_custom_ego_vehicle_{ego_idx}.xml"
        ego_xml_path = out_dir / ego_xml_name
        
        _write_carla_route_xml(
            path=ego_xml_path,
            route_id=route_id,
            role="ego",
            town=town,
            waypoints=carla_wps,
            times=carla_times if encode_timing else None,
            snap_to_road=False,  # Ego should follow exact route
            control_mode="",  # Ego controlled by agent
        )
        
        ego_entry = {
            "file": ego_xml_name,
            "route_id": route_id,
            "town": town,
            "name": f"ego_{ego_idx}",
            "kind": "ego",
            "model": str(ego_track.get("model", "vehicle.lincoln.mkz_2020")),
        }
        ego_entries.append(ego_entry)
        report["ego_files"].append(ego_xml_name)
        report["ego_count"] = int(report["ego_count"]) + 1
    
    manifest["ego"] = ego_entries
    
    # --- Export Actor Routes (grouped by kind) ---
    actors_by_kind: Dict[str, List[Dict[str, object]]] = {}
    
    for actor_track in actor_tracks:
        actor_id = actor_track.get("id", "unknown")
        vid = actor_track.get("vid", 0)
        role = str(actor_track.get("role", "npc")).lower()
        model = str(actor_track.get("model", ""))
        is_parked = bool(actor_track.get("parked_vehicle", False))
        frames = actor_track.get("frames", [])
        
        if not frames:
            continue
        
        # Determine kind and control mode
        if role in ("walker", "cyclist"):
            kind = "walker"
            control = walker_control_mode
        elif is_parked:
            kind = "static"
            control = "replay"  # Static actors always use replay (stationary)
        else:
            kind = "npc"
            control = actor_control_mode
        
        # Build candidate source tracks in V2X coordinates.
        # RAW keeps highest geometric fidelity, while correspondence-projected
        # poses can suppress A<->B lane jitter for policy NPCs.
        raw_wps: List[Waypoint] = []
        raw_times: List[float] = []
        corr_wps: List[Waypoint] = []
        corr_times: List[float] = []
        corr_valid = True

        for f in frames:
            rz = float(f.get("rz", f.get("z", 0)))
            t = float(f.get("t", 0))

            # RAW source (always available)
            rx = float(f.get("rx", f.get("x", 0)))
            ry = float(f.get("ry", f.get("y", 0)))
            ryaw = float(f.get("ryaw", f.get("yaw", 0)))
            raw_cx, raw_cy = v2x_to_carla(rx, ry)
            raw_cyaw = yaw_v2x_to_carla(ryaw)
            raw_wps.append(Waypoint(x=raw_cx, y=raw_cy, z=rz, yaw=raw_cyaw))
            raw_times.append(t)

            # Correspondence source (optional)
            if corr_valid:
                if ("cx" not in f) or ("cy" not in f) or ("cyaw" not in f):
                    corr_valid = False
                else:
                    sx = _safe_float(f.get("cx"), float("nan"))
                    sy = _safe_float(f.get("cy"), float("nan"))
                    syaw = _safe_float(f.get("cyaw"), float("nan"))
                    if not (math.isfinite(sx) and math.isfinite(sy) and math.isfinite(syaw)):
                        corr_valid = False
                    else:
                        corr_cx, corr_cy = v2x_to_carla(float(sx), float(sy))
                        corr_cyaw = yaw_v2x_to_carla(float(syaw))
                        corr_wps.append(Waypoint(x=corr_cx, y=corr_cy, z=rz, yaw=corr_cyaw))
                        corr_times.append(t)

        if not raw_wps:
            continue

        # Choose source trajectory:
        # - default RAW
        # - for policy NPCs, compare RAW vs correspondence and switch only when
        #   correspondence is materially more stable (less jumpy / fewer 180 flips).
        selected_source = "raw"
        selected_wps: List[Waypoint] = raw_wps
        selected_times: List[float] = raw_times
        selected_orig_len = len(raw_wps)

        def _postprocess_export_track(src_wps: List[Waypoint], src_times: List[float]) -> Tuple[List[Waypoint], List[float], float]:
            wps_local = [Waypoint(x=float(w.x), y=float(w.y), z=float(w.z), yaw=float(w.yaw)) for w in src_wps]
            times_local = [float(ti) for ti in src_times]
            if kind in ("npc", "walker") and len(wps_local) >= 3:
                # Use a lower speed threshold for walkers (no walker goes >10 m/s)
                san_speed = 40.0 if kind == "npc" else 10.0
                wps_local, times_local = _sanitize_trajectory(
                    wps_local,
                    times_local,
                    default_dt,
                    max_plausible_speed_mps=san_speed,
                )
            # Run stabilization AFTER sanitization so one noisy frame does not
            # bias spawn heading and force a wrong-way initialization.
            if kind in ("npc", "walker"):
                wps_local = _stabilize_initial_route_yaw_for_export(wps_local)
            score_local = _trajectory_instability_score(wps_local)
            return wps_local, times_local, float(score_local)

        raw_proc_wps, raw_proc_times, raw_score = _postprocess_export_track(raw_wps, raw_times)
        selected_wps, selected_times = raw_proc_wps, raw_proc_times

        if kind == "npc" and str(control).lower() == "policy" and corr_valid and len(corr_wps) == len(raw_wps):
            corr_proc_wps, corr_proc_times, corr_score = _postprocess_export_track(corr_wps, corr_times)
            if corr_score + 2.0 < raw_score:
                selected_source = "corr"
                selected_wps, selected_times = corr_proc_wps, corr_proc_times
                selected_orig_len = len(corr_wps)
                print(
                    f"[EXPORT] {actor_id}: using correspondence poses "
                    f"(instability {raw_score:.1f} -> {corr_score:.1f})"
                )

        carla_wps = selected_wps
        carla_times = selected_times

        if kind in ("npc", "walker") and len(carla_wps) != selected_orig_len:
            print(
                f"[SANITIZE] {actor_id} ({selected_source}): "
                f"{selected_orig_len} -> {len(carla_wps)} waypoints after cleaning"
            )

        # For static actors with spawn_only mode, keep only the first waypoint
        if kind == "static" and static_spawn_only and carla_wps:
            carla_wps = [carla_wps[0]]
            carla_times = [carla_times[0]] if carla_times else []
        
        # Compute per-waypoint speeds and global fallback target speed
        per_wp_speeds: Optional[List[float]] = None
        target_speed: Optional[float] = None
        if control == "policy" and len(carla_wps) >= 2:
            per_wp_speeds = _compute_per_waypoint_speeds(carla_wps, carla_times, default_dt)
            target_speed = _compute_target_speed(carla_wps, carla_times, default_dt)
        
        # Create kind subdirectory
        kind_dir = actors_dir / kind
        kind_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        actor_type = role.replace(" ", "_").replace("-", "_").title()
        if not actor_type or actor_type.lower() == "npc":
            actor_type = "Vehicle"
        xml_name = f"{town.lower()}_custom_{actor_type}_{vid}_{kind}.xml"
        xml_path = kind_dir / xml_name
        
        _write_carla_route_xml(
            path=xml_path,
            route_id=route_id,
            role=kind,
            town=town,
            waypoints=carla_wps,
            times=carla_times if encode_timing else None,
            snap_to_road=snap_to_road and kind == "npc",
            control_mode=control,
            target_speed_mps=target_speed,
            model=model,
            speeds=per_wp_speeds,
        )
        
        # Build manifest entry
        actor_entry: Dict[str, object] = {
            "file": f"actors/{kind}/{xml_name}",
            "route_id": route_id,
            "town": town,
            "name": str(actor_id),
            "kind": kind,
            "model": model,
            "control_mode": control,
        }
        if target_speed is not None and target_speed > 0:
            actor_entry["target_speed"] = round(target_speed, 2)
            # route_parser.py reads "speed" (not "target_speed"), so emit both.
            actor_entry["speed"] = round(target_speed, 2)
        
        # Add to manifest by kind
        if kind not in actors_by_kind:
            actors_by_kind[kind] = []
        actors_by_kind[kind].append(actor_entry)
        report["actor_files"].append(f"actors/{kind}/{xml_name}")
        
        # Update counts
        if kind == "npc":
            report["npc_count"] = int(report["npc_count"]) + 1
        elif kind == "walker":
            report["walker_count"] = int(report["walker_count"]) + 1
        elif kind == "static":
            report["static_count"] = int(report["static_count"]) + 1
    
    # Merge actor manifest entries
    for kind, entries in actors_by_kind.items():
        manifest[kind] = entries
    
    report["total_actors"] = (
        int(report["npc_count"]) + 
        int(report["walker_count"]) + 
        int(report["static_count"])
    )
    
    # Write manifest
    manifest_path = out_dir / "actors_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    
    # Write control mode configuration for run_custom_eval.py
    control_cfg = {
        "actor_control_mode": actor_control_mode,
        "walker_control_mode": walker_control_mode,
        "encode_timing": encode_timing,
        "snap_to_road": snap_to_road,
        "static_spawn_only": static_spawn_only,
        "town": town,
        "route_id": route_id,
        "manifest_path": "actors_manifest.json",
        "ego_count": report["ego_count"],
        "npc_count": report["npc_count"],
        "walker_count": report["walker_count"],
        "static_count": report["static_count"],
    }
    control_cfg_path = out_dir / "carla_control_config.json"
    control_cfg_path.write_text(json.dumps(control_cfg, indent=2), encoding="utf-8")
    
    print(f"[CARLA-EXPORT] Exported CARLA routes to: {out_dir}")
    print(
        f"[CARLA-EXPORT]   Ego: {report['ego_count']}, "
        f"NPC (control={actor_control_mode}): {report['npc_count']}, "
        f"Walker (control={walker_control_mode}): {report['walker_count']}, "
        f"Static: {report['static_count']}"
    )
    
    return report


def main() -> None:
    global _active_carla_manager
    args = parse_args()
    
    # --- Auto-launch CARLA server if requested ---
    carla_manager: Optional[CarlaProcessManager] = None
    if bool(args.start_carla):
        # Determine CARLA root directory
        carla_root_str = args.carla_root
        if not carla_root_str:
            carla_root_str = os.environ.get("CARLA_ROOT")
        if not carla_root_str:
            # Default: look for carla912 relative to workspace root
            workspace_root = Path(__file__).resolve().parent.parent.parent
            carla_root_str = str(workspace_root / "carla912")
        carla_root = Path(carla_root_str).expanduser().resolve()
        
        if not carla_root.exists():
            raise SystemExit(f"CARLA root not found: {carla_root}")
        
        carla_manager = CarlaProcessManager(
            carla_root=carla_root,
            host=str(args.carla_host),
            port=int(args.carla_port),
            extra_args=list(args.carla_arg),
            port_tries=int(args.carla_port_tries),
            port_step=int(args.carla_port_step),
        )
        
        # Install signal handlers for clean shutdown
        _install_carla_signal_handlers()
        _active_carla_manager = carla_manager
        
        # Start CARLA and update port if it changed
        actual_port = carla_manager.start()
        if actual_port != int(args.carla_port):
            args.carla_port = actual_port
            print(f"[INFO] CARLA port updated to: {actual_port}")
    
    try:
        # Check if batch mode or single mode
        if args.scenario_dirs:
            _run_batch_processing(args, carla_manager)
        elif args.scenario_dir:
            _run_main_logic(args)
        else:
            raise SystemExit("Either --scenario-dir or --scenario-dirs must be provided.")
    finally:
        # Ensure CARLA is stopped on exit
        if carla_manager is not None:
            carla_manager.stop()
            _active_carla_manager = None


def _is_scenario_directory(path: Path) -> bool:
    """
    Check if a directory is a valid scenario directory.
    
    A scenario directory typically contains YAML subdirectories with vehicle/actor data.
    """
    if not path.is_dir():
        return False
    
    # Check for common scenario indicators
    yaml_indicators = ['yaml_to_carla_log', 'yaml_data', 'vehicle_data', 'actor_data']
    
    # Check if any subdirectory matches scenario patterns
    for subdir in path.iterdir():
        if subdir.is_dir():
            # Check for numbered subdirectories (common in scenarios)
            if subdir.name.isdigit():
                return True
            # Check for yaml-related subdirectories
            if any(ind in subdir.name.lower() for ind in yaml_indicators):
                return True
            # Check if subdir contains .yaml files
            try:
                yaml_files = list(subdir.glob('*.yaml')) + list(subdir.glob('*.yml'))
                if yaml_files:
                    return True
            except PermissionError:
                pass
    
    # Check for direct yaml files
    try:
        yaml_files = list(path.glob('*.yaml')) + list(path.glob('*.yml'))
        if yaml_files:
            return True
    except PermissionError:
        pass
    
    return False


def _expand_scenario_directories(paths: List[Path]) -> List[Path]:
    """
    Expand a list of paths to find all valid scenario directories.
    
    If a path is a parent directory containing multiple scenarios, expand it.
    If a path is a scenario directory itself, keep it.
    
    Returns:
        List of scenario directories (deduplicated)
    """
    expanded = []
    seen = set()
    
    for path in paths:
        if not path.exists():
            continue
            
        if _is_scenario_directory(path):
            # This is a scenario directory
            if path not in seen:
                expanded.append(path)
                seen.add(path)
        else:
            # Check if this is a parent directory containing scenarios
            try:
                for subdir in sorted(path.iterdir()):
                    if subdir.is_dir() and _is_scenario_directory(subdir):
                        if subdir not in seen:
                            expanded.append(subdir)
                            seen.add(subdir)
            except PermissionError:
                pass
    
    return expanded


def _run_batch_processing(args: argparse.Namespace, carla_manager: Optional[CarlaProcessManager] = None) -> None:
    """
    Process multiple scenario directories in batch mode.
    
    Each scenario directory is processed, simulated, aligned, and has results/videos generated.
    Results are named after each scenario folder for easy identification.
    
    Supports:
    - Multiple explicit scenario directories
    - Parent directories containing multiple scenarios (auto-expanded)
    - Handles duplicate scenario names by adding unique suffixes
    """
    from datetime import datetime
    
    input_paths = [Path(d).expanduser().resolve() for d in args.scenario_dirs]
    
    # Validate and report missing directories
    for path in input_paths:
        if not path.exists():
            print(f"[WARN] Path not found, skipping: {path}")
    
    existing_paths = [p for p in input_paths if p.exists()]
    if not existing_paths:
        raise SystemExit("No valid paths found.")
    
    # Expand to find all scenario directories
    scenario_dirs = _expand_scenario_directories(existing_paths)
    if not scenario_dirs:
        print("[INFO] No scenario directories found directly. Checking if inputs are parent directories...")
        # Try treating each path as a parent and find scenarios inside
        for path in existing_paths:
            if path.is_dir():
                print(f"[INFO] Scanning: {path}")
                for subdir in sorted(path.iterdir()):
                    if subdir.is_dir():
                        print(f"  - {subdir.name}: {'scenario' if _is_scenario_directory(subdir) else 'not a scenario'}")
        raise SystemExit("No valid scenario directories found. Check that directories contain YAML data.")
    
    print(f"[INFO] Found {len(scenario_dirs)} scenario directories to process")
    
    # Determine batch results root
    batch_results_root = None
    if args.batch_results_root:
        batch_results_root = Path(args.batch_results_root).expanduser().resolve()
    else:
        # Default: create a batch_results folder in the parent of the first scenario
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_results_root = scenario_dirs[0].parent / f"batch_results_{timestamp}"
    batch_results_root.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING MODE")
    print(f"{'='*60}")
    print(f"Total scenarios: {len(scenario_dirs)}")
    print(f"Results root: {batch_results_root}")
    print(f"{'='*60}\n")
    
    # Track scenario names to handle duplicates
    seen_names: Dict[str, int] = {}
    
    batch_report = {
        "start_time": datetime.now().isoformat(),
        "scenarios": [],
        "success_count": 0,
        "failure_count": 0,
        "total_count": len(scenario_dirs),
    }
    
    for idx, scenario_dir in enumerate(scenario_dirs, 1):
        base_scenario_name = scenario_dir.name
        
        # Handle duplicate scenario names by adding a suffix
        if base_scenario_name in seen_names:
            seen_names[base_scenario_name] += 1
            scenario_name = f"{base_scenario_name}_{seen_names[base_scenario_name]}"
        else:
            seen_names[base_scenario_name] = 0
            scenario_name = base_scenario_name
        
        print(f"\n{'='*60}")
        print(f"PROCESSING SCENARIO {idx}/{len(scenario_dirs)}: {scenario_name}")
        print(f"  Source: {scenario_dir}")
        print(f"{'='*60}\n")
        
        scenario_result = {
            "name": scenario_name,
            "base_name": base_scenario_name,
            "path": str(scenario_dir),
            "success": False,
            "error": None,
            "output_dir": None,
            "video_paths": [],
        }
        
        try:
            # Create scenario-specific output directory
            scenario_out_dir = batch_results_root / scenario_name
            scenario_out_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a copy of args for this scenario
            scenario_args = argparse.Namespace(**vars(args))
            scenario_args.scenario_dir = str(scenario_dir)
            scenario_args.out_dir = str(scenario_out_dir)
            
            # Run the main logic for this scenario
            _run_main_logic(scenario_args)
            
            scenario_result["success"] = True
            scenario_result["output_dir"] = str(scenario_out_dir)
            batch_report["success_count"] += 1
            
            # Generate videos if requested
            if args.generate_videos:
                video_paths = _generate_scenario_videos(
                    scenario_dir,
                    scenario_name,
                    fps=float(args.video_fps),
                    resize_factor=int(getattr(args, 'video_resize_factor', 2)),
                )
                scenario_result["video_paths"] = video_paths
            
            print(f"\n[OK] Scenario {scenario_name} completed successfully.")
            
        except Exception as exc:
            scenario_result["error"] = str(exc)
            batch_report["failure_count"] += 1
            print(f"\n[ERROR] Scenario {scenario_name} failed: {exc}")
            import traceback
            traceback.print_exc()
        
        batch_report["scenarios"].append(scenario_result)
    
    batch_report["end_time"] = datetime.now().isoformat()
    
    # Write batch report
    report_path = batch_results_root / "batch_report.json"
    report_path.write_text(json.dumps(batch_report, indent=2), encoding="utf-8")
    
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Success: {batch_report['success_count']}/{batch_report['total_count']}")
    print(f"Failures: {batch_report['failure_count']}/{batch_report['total_count']}")
    print(f"Results: {batch_results_root}")
    print(f"Report: {report_path}")
    print(f"{'='*60}\n")


def _ego_indices(scenario_dir: Path) -> List[int]:
    """Return sorted positive-integer ego indices found as subdirectories of *scenario_dir*.

    The dataset convention uses 1-indexed subdirectories (``1/``, ``2/``, …) for each ego
    vehicle's camera images.  Negative folders (e.g. ``-1/``) are infrastructure and are skipped.
    """
    indices: List[int] = []
    try:
        for child in scenario_dir.iterdir():
            if not child.is_dir():
                continue
            name = child.name.strip()
            if not name.isdigit():
                continue
            idx = int(name)
            if idx > 0:
                indices.append(idx)
    except OSError:
        pass
    return sorted(set(indices))


def _pick_run_image_dir(image_root: Path) -> Optional[Path]:
    """Pick the most-recently-modified image run subfolder under *image_root*.

    ``run_custom_eval.py`` stores captured images in a timestamped subdirectory
    under ``<results>/<scenario>/image/<timestamp>/``.  This helper picks the newest one.
    """
    if not image_root.exists():
        return None
    dirs = [p for p in image_root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)


def _generate_scenario_videos(
    scenario_dir: Path,
    scenario_name: str,
    fps: float = 10.0,
    resize_factor: int = 2,
    fullvideos_dir: Optional[Path] = None,
    results_root: Optional[Path] = None,
    video_conda_env: Optional[str] = None,
) -> List[str]:
    """Generate per-ego side-by-side, CARLA-only and real-only videos.

    Mirrors the video generation stage of ``run_train1_logreplay_batch.py``:
    all videos are written into a central *fullvideos_dir* named
    ``{scenario_name}_{variant}_{ego_id}.mp4``.

    It invokes ``visualization/gen_video.py`` exactly as the batch pipeline does.

    Args:
        scenario_dir: Dataset scenario directory (contains ``1/``, ``2/``, … ego camera folders).
        scenario_name: Scenario identifier used for video filenames.
        fps: Frames per second for the output videos.
        resize_factor: Down-scale factor passed to ``gen_video.py``.
        fullvideos_dir: Output directory for all videos.
            Defaults to ``results/results_driving_custom/fullvideos``.
        results_root: Root of the evaluation results tree.
            Defaults to ``results/results_driving_custom``.
        video_conda_env: Optional conda environment name to run gen_video.py under.

    Returns:
        List of paths to the generated mp4 files.
    """
    repo_root = Path(__file__).resolve().parents[2]
    gen_video_script = repo_root / "visualization" / "gen_video.py"
    if not gen_video_script.exists():
        print(f"[WARN] gen_video.py not found at {gen_video_script}; skipping video generation.")
        return []

    if results_root is None:
        results_root = repo_root / "results" / "results_driving_custom"
    if fullvideos_dir is None:
        fullvideos_dir = results_root / "fullvideos"
    fullvideos_dir.mkdir(parents=True, exist_ok=True)

    # Discover the image run directory created by run_custom_eval.
    image_root = results_root / scenario_name / "image"
    run_image_dir = _pick_run_image_dir(image_root)
    if run_image_dir is None:
        print(f"[WARN] No image run directory found under {image_root}; skipping video generation.")
        return []
    print(f"[INFO] image run dir: {run_image_dir}")

    # Discover ego indices from the dataset scenario directory.
    ego_ids = _ego_indices(scenario_dir)
    if not ego_ids:
        print(f"[WARN] No positive ego-index folders found in {scenario_dir}; skipping video step.")
        return []

    # Build the python command prefix.
    python_bin = sys.executable

    def _build_cmd(script_args: List[str]) -> List[str]:
        if video_conda_env:
            cmd = ["conda", "run", "-n", video_conda_env, "python", "-u", str(gen_video_script)]
        else:
            cmd = [python_bin, "-u", str(gen_video_script)]
        cmd.extend(script_args)
        return cmd

    generated: List[str] = []

    for ego_id in ego_ids:
        real_cam_dir = scenario_dir / str(ego_id)
        # Logreplay images use 0-indexed ego IDs; dataset folders use 1-indexed.
        # Try both naming conventions: rgb_front_N (logreplay_agent) and logreplay_rgb_N (tcp_agent).
        carla_img_dir: Optional[Path] = None
        logreplay_base = run_image_dir / "logreplayimages"
        for candidate_name in [
            f"rgb_front_{ego_id - 1}",
            f"logreplay_rgb_{ego_id - 1}",
        ]:
            candidate = logreplay_base / candidate_name
            if candidate.exists():
                carla_img_dir = candidate
                break

        out_side = fullvideos_dir / f"{scenario_name}_sidebyside_{ego_id}.mp4"
        out_carla = fullvideos_dir / f"{scenario_name}_carla_{ego_id}.mp4"
        out_real = fullvideos_dir / f"{scenario_name}_real_{ego_id}.mp4"

        real_exists = real_cam_dir.exists()
        carla_exists = carla_img_dir is not None and carla_img_dir.exists()

        if not real_exists:
            print(f"[WARN] Missing real cam folder for ego {ego_id}: {real_cam_dir}")
        if not carla_exists:
            print(f"[WARN] Missing logreplay folder for ego {ego_id}")

        jobs: List[tuple] = []

        # Side-by-side: real (cam1) + CARLA
        if real_exists and carla_exists and carla_img_dir is not None:
            jobs.append((
                "sidebyside",
                out_side,
                _build_cmd([
                    str(real_cam_dir),
                    "--only-suffix", "cam1",
                    "--side-by-side-dir", str(carla_img_dir),
                    "--fps", str(fps),
                    "--resize-factor", str(resize_factor),
                    "--output", str(out_side),
                ]),
            ))

        # CARLA-only
        if carla_exists and carla_img_dir is not None:
            jobs.append((
                "carla",
                out_carla,
                _build_cmd([
                    str(carla_img_dir),
                    "--fps", str(fps),
                    "--resize-factor", str(resize_factor),
                    "--output", str(out_carla),
                ]),
            ))

        # Real-only (cam1)
        if real_exists:
            jobs.append((
                "real",
                out_real,
                _build_cmd([
                    str(real_cam_dir),
                    "--only-suffix", "cam1",
                    "--fps", str(fps),
                    "--resize-factor", str(resize_factor),
                    "--output", str(out_real),
                ]),
            ))

        if not jobs:
            print(f"[WARN] No video inputs available for ego {ego_id}; skipping.")
            continue

        for tag, out_mp4, cmd in jobs:
            print(f"[INFO] Generating {tag} video for ego {ego_id}: {out_mp4.name}")
            try:
                subprocess.run(cmd, check=True)
                if out_mp4.exists() and out_mp4.stat().st_size > 0:
                    generated.append(str(out_mp4))
                    print(f"[OK] {tag} video: {out_mp4}")
                else:
                    print(f"[WARN] {tag} video was not created or is empty.")
            except subprocess.CalledProcessError as exc:
                print(f"[WARN] {tag} video generation failed (exit {exc.returncode})")
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[WARN] {tag} video generation error: {exc}")

    if generated:
        print(f"[INFO] All videos written to: {fullvideos_dir}")

    return generated


def _run_main_logic(args: argparse.Namespace) -> None:
    """Main processing logic, separated for clean CARLA lifecycle management."""
    # skip_map_snap_compute only affects _apply_lane_correspondence_to_payload; CARLA layer still loads
    scenario_dir = Path(args.scenario_dir).expanduser().resolve()
    if not scenario_dir.exists():
        raise SystemExit(f"Scenario directory not found: {scenario_dir}")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (scenario_dir / "yaml_map_export")
    out_dir.mkdir(parents=True, exist_ok=True)

    yaml_dirs = pick_yaml_dirs(scenario_dir, args.subdir)
    if not yaml_dirs:
        raise SystemExit(f"No YAML dirs found under: {scenario_dir}")

    print(f"[INFO] Selected YAML dirs ({len(yaml_dirs)}):")
    for yd in yaml_dirs:
        print(f"  - {yd}")

    vehicles, vehicle_times, ego_trajs, ego_times, obj_info, actor_source_subdir, actor_orig_vid, merge_stats = _merge_subdir_trajectories(
        yaml_dirs=yaml_dirs,
        dt=float(args.dt),
        id_merge_distance_m=float(args.id_merge_distance_m),
    )
    actor_alias_vids: Dict[int, List[int]] = {int(vid): [int(vid)] for vid in vehicles.keys()}
    if bool(args.cross_id_dedup):
        (
            vehicles,
            vehicle_times,
            obj_info,
            actor_source_subdir,
            actor_orig_vid,
            actor_alias_vids,
            cross_id_stats,
        ) = _deduplicate_cross_id_tracks(
            vehicles=vehicles,
            vehicle_times=vehicle_times,
            obj_info=obj_info,
            actor_source_subdir=actor_source_subdir,
            actor_orig_vid=actor_orig_vid,
            dt=float(args.dt),
            max_median_dist_m=float(args.cross_id_dedup_max_median_dist_m),
            max_p90_dist_m=float(args.cross_id_dedup_max_p90_dist_m),
            max_median_yaw_diff_deg=float(args.cross_id_dedup_max_median_yaw_diff_deg),
            min_common_points=int(args.cross_id_dedup_min_common_points),
            min_overlap_ratio_each=float(args.cross_id_dedup_min_overlap_each),
            min_overlap_ratio_any=float(args.cross_id_dedup_min_overlap_any),
        )
    else:
        cross_id_stats = {
            "cross_id_dedup_enabled": False,
            "cross_id_pair_checks": 0,
            "cross_id_candidate_pairs": 0,
            "cross_id_clusters": 0,
            "cross_id_removed": 0,
            "cross_id_removed_ids": [],
        }
    merge_stats.update(cross_id_stats)
    merge_stats["output_tracks"] = int(len(vehicles))

    timing_optimization: Dict[str, object] = {
        "timing_policy": {
            "spawn": "first_observed_frame",
            "despawn": "last_observed_frame",
            "global_early_spawn_optimization": False,
            "global_late_despawn_optimization": False,
        },
        "early_spawn": {
            "enabled": bool(args.maximize_safe_early_spawn),
            "applied": False,
            "reason": "disabled_by_flag" if not bool(args.maximize_safe_early_spawn) else "not_run",
            "adjusted_actor_ids": [],
            "adjusted_spawn_times": {},
        },
        "late_despawn": {
            "enabled": bool(args.maximize_safe_late_despawn),
            "applied": False,
            "reason": "disabled_by_flag" if not bool(args.maximize_safe_late_despawn) else "not_run",
            "adjusted_actor_ids": [],
            "hold_until_time": 0.0,
        },
    }

    actor_meta_for_timing = _build_actor_meta_for_timing_optimization(vehicles, obj_info)
    if bool(args.maximize_safe_early_spawn):
        if actor_meta_for_timing:
            selected_spawn_times, early_report = _maximize_safe_early_spawn_actors(
                vehicles=vehicles,
                vehicle_times=vehicle_times,
                actor_meta=actor_meta_for_timing,
                dt=float(args.dt),
                safety_margin=float(args.early_spawn_safety_margin),
            )
            vehicles, vehicle_times, adjusted_ids, applied_spawn_times = _apply_early_spawn_time_overrides(
                vehicles=vehicles,
                vehicle_times=vehicle_times,
                early_spawn_times=selected_spawn_times,
                dt=float(args.dt),
            )
            early_report = dict(early_report)
            early_report["enabled"] = True
            early_report["applied"] = True
            early_report["reason"] = "applied"
            early_report["adjusted_actor_ids"] = [int(v) for v in adjusted_ids]
            early_report["adjusted_spawn_times"] = {
                str(int(vid)): float(t) for vid, t in sorted(applied_spawn_times.items())
            }
            timing_optimization["early_spawn"] = early_report
            timing_optimization["timing_policy"]["global_early_spawn_optimization"] = True
        else:
            timing_optimization["early_spawn"] = {
                "enabled": True,
                "applied": False,
                "reason": "no_actor_metadata",
                "adjusted_actor_ids": [],
                "adjusted_spawn_times": {},
            }

    if bool(args.maximize_safe_late_despawn):
        if actor_meta_for_timing:
            hold_until_time = 0.0
            for times in vehicle_times.values():
                if times:
                    hold_until_time = max(float(hold_until_time), float(times[-1]))
            late_report: Dict[str, object] = {
                "enabled": True,
                "applied": False,
                "hold_until_time": float(hold_until_time),
                "adjusted_actor_ids": [],
            }
            if hold_until_time > 0.0:
                selected_ids, select_report = _maximize_safe_late_despawn_actors(
                    vehicles=vehicles,
                    vehicle_times=vehicle_times,
                    actor_meta=actor_meta_for_timing,
                    dt=float(args.dt),
                    safety_margin=float(args.late_despawn_safety_margin),
                    hold_until_time=float(hold_until_time),
                )
                vehicles, vehicle_times, adjusted_ids = _apply_late_despawn_time_overrides(
                    vehicles=vehicles,
                    vehicle_times=vehicle_times,
                    selected_late_hold_ids=selected_ids,
                    dt=float(args.dt),
                    hold_until_time=float(hold_until_time),
                )
                late_report.update(dict(select_report))
                late_report["applied"] = True
                late_report["reason"] = "applied"
                late_report["adjusted_actor_ids"] = [int(v) for v in adjusted_ids]
                timing_optimization["timing_policy"]["global_late_despawn_optimization"] = True
            else:
                late_report["reason"] = "non_positive_horizon"
            timing_optimization["late_despawn"] = late_report
        else:
            timing_optimization["late_despawn"] = {
                "enabled": True,
                "applied": False,
                "reason": "no_actor_metadata",
                "adjusted_actor_ids": [],
                "hold_until_time": 0.0,
            }

    print(
        f"[INFO] Parsed trajectories: egos={len(ego_trajs)} actors={len(vehicles)} "
        f"(timed={sum(1 for v in vehicle_times.values() if v)}) "
        f"id_collisions={merge_stats['ids_with_collisions']} "
        f"merged={merge_stats['merged_duplicates']} "
        f"split={merge_stats['split_tracks_created']} "
        f"cross_id_removed={merge_stats.get('cross_id_removed', 0)} "
        f"cross_id_clusters={merge_stats.get('cross_id_clusters', 0)} "
        f"early_adjusted={len(timing_optimization.get('early_spawn', {}).get('adjusted_actor_ids', []))} "
        f"late_adjusted={len(timing_optimization.get('late_despawn', {}).get('adjusted_actor_ids', []))}"
    )

    map_paths: List[Path] = []
    if args.map_pkl:
        map_paths = [Path(p).expanduser().resolve() for p in args.map_pkl]
    else:
        map_paths = [
            Path("/data2/marco/CoLMDriver/v2xpnp/map/v2v_corridors_vector_map.pkl"),
            Path("/data2/marco/CoLMDriver/v2xpnp/map/v2x_intersection_vector_map.pkl"),
        ]
    for p in map_paths:
        if not p.exists():
            raise SystemExit(f"Map pickle not found: {p}")

    map_data_list = [_load_vector_map(p) for p in map_paths]
    chosen_map, selection_scores = _select_best_map(
        maps=map_data_list,
        ego_trajs=ego_trajs,
        vehicles=vehicles,
        sample_count=int(args.map_selection_sample_count),
        bbox_margin=float(args.map_selection_bbox_margin),
    )
    print(f"[INFO] Selected map: {chosen_map.name} ({chosen_map.source_path})")
    for score in selection_scores:
        print(
            "  - {name}: score={score:.3f} median={median_nearest_m:.2f}m "
            "p90={p90_nearest_m:.2f}m outside={outside_bbox_ratio:.3f}".format(**score)
        )

    carla_map_layer: Optional[Dict[str, object]] = None
    if bool(args.carla_map_layer):
        carla_cache_path = Path(args.carla_map_cache).expanduser().resolve()
        carla_align_path = Path(args.carla_map_offset_json).expanduser().resolve() if args.carla_map_offset_json else None
        if not carla_cache_path.exists():
            print(f"[WARN] CARLA map cache not found; skipping CARLA layer: {carla_cache_path}")
        else:
            try:
                raw_lines, cache_bounds, cache_map_name = _load_carla_map_cache_lines(carla_cache_path)
                align_cfg = _load_carla_alignment_cfg(carla_align_path)
                transformed_lines, transformed_bbox = _transform_carla_lines(raw_lines, align_cfg)
                if transformed_lines:
                    carla_map_layer = {
                        "name": str(cache_map_name or "carla_westwood_map"),
                        "source_path": str(carla_cache_path),
                        "alignment_path": str(carla_align_path) if carla_align_path is not None else "",
                        "raw_bounds": {
                            "min_x": float(cache_bounds[0]),
                            "max_x": float(cache_bounds[1]),
                            "min_y": float(cache_bounds[2]),
                            "max_y": float(cache_bounds[3]),
                        }
                        if cache_bounds is not None
                        else None,
                        "bbox": {
                            "min_x": float(transformed_bbox[0]),
                            "max_x": float(transformed_bbox[1]),
                            "min_y": float(transformed_bbox[2]),
                            "max_y": float(transformed_bbox[3]),
                        },
                        "lines": transformed_lines,
                        "transform": {
                            "scale": float(align_cfg.get("scale", 1.0)),
                            "theta_deg": float(align_cfg.get("theta_deg", 0.0)),
                            "tx": float(align_cfg.get("tx", 0.0)),
                            "ty": float(align_cfg.get("ty", 0.0)),
                            "flip_y": bool(align_cfg.get("flip_y", False)),
                            "source_path": str(align_cfg.get("source_path", "")),
                        },
                    }
                    print(
                        f"[INFO] Loaded CARLA map layer: lines={len(transformed_lines)} "
                        f"source={carla_cache_path} align={carla_align_path if carla_align_path else '-'}"
                    )
                else:
                    print(f"[WARN] CARLA map cache had no valid polylines after transform: {carla_cache_path}")
            except Exception as exc:
                print(f"[WARN] Failed to build CARLA map layer from cache: {exc}")

    # --- Load or capture CARLA top-down image underlay ---
    if carla_map_layer is not None:
        img_cache_path = Path(args.carla_map_image_cache).expanduser().resolve()
        img_meta_path = img_cache_path.with_suffix(".json")
        # raw_bounds from the CARLA map cache
        raw_b = carla_map_layer.get("raw_bounds")
        raw_bounds_tuple: Optional[Tuple[float, float, float, float]] = None
        if isinstance(raw_b, dict):
            try:
                raw_bounds_tuple = (
                    float(raw_b["min_x"]),
                    float(raw_b["max_x"]),
                    float(raw_b["min_y"]),
                    float(raw_b["max_y"]),
                )
            except Exception:
                pass
        result = _load_or_capture_carla_topdown(
            image_cache_path=img_cache_path,
            meta_cache_path=img_meta_path,
            carla_host=str(args.carla_host),
            carla_port=int(args.carla_port),
            raw_bounds=raw_bounds_tuple,
            capture_enabled=bool(args.capture_carla_image),
            carla_map_name=str(args.carla_map_name),
        )
        if result is not None:
            jpeg_bytes, img_raw_bounds = result
            img_b64_str = base64.b64encode(jpeg_bytes).decode("ascii")
            # Transform image bounds into V2XPNP coordinate space
            img_transform = carla_map_layer.get("transform", {})
            img_bounds_v2 = _transform_image_bounds_to_v2xpnp(img_raw_bounds, img_transform)
            carla_map_layer["image_b64"] = img_b64_str
            carla_map_layer["image_bounds"] = img_bounds_v2
            print(
                f"[INFO] CARLA top-down image attached: "
                f"{len(jpeg_bytes)} bytes, b64={len(img_b64_str)} chars, "
                f"bounds={img_bounds_v2}"
            )
        else:
            print("[INFO] No CARLA top-down image available (skipping underlay).")

    # --- GRP-aware trajectory alignment (before map-matching / export) ---
    grp_align_report: Dict[str, object] = {"enabled": False}
    if bool(args.grp_align):
        grp_align_cfg_path = (
            Path(args.carla_map_offset_json).expanduser().resolve()
            if args.carla_map_offset_json
            else None
        )
        grp_align_cfg = _load_carla_alignment_cfg(grp_align_cfg_path)
        vehicles, vehicle_times, ego_trajs, ego_times, grp_align_report = _grp_align_trajectories(
            vehicles=vehicles,
            vehicle_times=vehicle_times,
            ego_trajs=ego_trajs,
            ego_times=ego_times,
            obj_info=obj_info,
            parked_vehicle_cfg={
                "net_disp_max_m": float(args.parked_net_disp_max_m),
                "radius_p90_max_m": float(args.parked_radius_p90_max_m),
                "radius_max_m": float(args.parked_radius_max_m),
                "p95_step_max_m": float(args.parked_p95_step_max_m),
                "max_from_start_m": float(args.parked_max_from_start_m),
                "large_step_threshold_m": float(args.parked_large_step_threshold_m),
                "large_step_ratio_max": float(args.parked_large_step_max_ratio),
                "robust_cluster_enabled": float(1.0 if bool(args.parked_robust_cluster) else 0.0),
                "robust_cluster_eps_m": float(args.parked_robust_cluster_eps_m),
                "robust_min_inlier_ratio": float(args.parked_robust_min_inlier_ratio),
                "robust_max_outlier_run": float(args.parked_robust_max_outlier_run),
                "robust_min_points": float(args.parked_robust_min_points),
            },
            align_cfg=grp_align_cfg,
            carla_host=str(args.carla_host),
            carla_port=int(args.carla_port),
            carla_map_name=str(args.carla_map_name),
            sampling_resolution=float(args.grp_sampling_resolution),
            snap_radius=float(args.grp_snap_radius),
            snap_k=int(args.grp_snap_k),
            heading_thresh=float(args.grp_heading_thresh),
            lane_change_penalty=float(args.grp_lane_change_penalty),
            default_dt=float(args.dt),
            enabled=True,
        )

    matcher = LaneMatcher(chosen_map)
    lane_change_cfg: Dict[str, object] = {
        "enabled": bool(args.lane_change_filter),
        "lane_top_k": int(args.lane_snap_top_k),
        "confirm_window": int(args.lane_change_confirm_window),
        "confirm_votes": int(args.lane_change_confirm_votes),
        "cooldown_frames": int(args.lane_change_cooldown_frames),
        "endpoint_guard_frames": int(args.lane_change_endpoint_guard_frames),
        "endpoint_extra_votes": int(args.lane_change_endpoint_extra_votes),
        "min_improvement_m": float(args.lane_change_min_improvement_m),
        "keep_lane_max_dist": float(args.lane_change_keep_lane_max_dist),
        "short_run_max": int(args.lane_change_short_run_max),
        "endpoint_short_run": int(args.lane_change_endpoint_short_run),
    }
    vehicle_lane_policy_cfg: Dict[str, object] = {
        "forbidden_lane_types": str(args.vehicle_forbidden_lane_types),
        "parked_only_lane_types": str(args.vehicle_parked_only_lane_types),
    }
    parked_vehicle_cfg: Dict[str, float] = {
        "net_disp_max_m": float(args.parked_net_disp_max_m),
        "radius_p90_max_m": float(args.parked_radius_p90_max_m),
        "radius_max_m": float(args.parked_radius_max_m),
        "p95_step_max_m": float(args.parked_p95_step_max_m),
        "max_from_start_m": float(args.parked_max_from_start_m),
        "large_step_threshold_m": float(args.parked_large_step_threshold_m),
        "large_step_ratio_max": float(args.parked_large_step_max_ratio),
        "robust_cluster_enabled": float(1.0 if bool(args.parked_robust_cluster) else 0.0),
        "robust_cluster_eps_m": float(args.parked_robust_cluster_eps_m),
        "robust_min_inlier_ratio": float(args.parked_robust_min_inlier_ratio),
        "robust_max_outlier_run": float(args.parked_robust_max_outlier_run),
        "robust_min_points": float(args.parked_robust_min_points),
    }

    # --- Walker sidewalk compression and spawn stabilization ---
    walker_processing_report: Dict[str, object] = {"enabled": False}
    if bool(args.walker_sidewalk_compression):
        # Get CARLA map lines (road polylines in V2XPNP coordinate space)
        carla_lines_for_walker: List[List[List[float]]] = []
        if carla_map_layer is not None:
            carla_lines_for_walker = carla_map_layer.get("lines", [])
        
        if not carla_lines_for_walker:
            print("[WARN] Walker sidewalk compression: no CARLA map lines available, skipping.")
            walker_processing_report = {
                "enabled": False,
                "reason": "no_carla_map_lines",
            }
        else:
            print(f"[INFO] Processing walker sidewalk compression using {len(carla_lines_for_walker)} CARLA road polylines...")
            walker_processor = WalkerSidewalkProcessor(
                carla_map_lines=carla_lines_for_walker,
                lane_spacing_m=args.walker_lane_spacing_m,  # None = auto-calibrate
                sidewalk_start_factor=float(args.walker_sidewalk_start_factor),
                sidewalk_outer_factor=float(args.walker_sidewalk_outer_factor),
                compression_target_band_m=float(args.walker_compression_target_band_m),
                compression_power=float(args.walker_compression_power),
                min_spawn_separation_m=float(args.walker_min_spawn_separation_m),
                walker_radius_m=float(args.walker_radius_m),
                crossing_road_ratio_thresh=float(args.walker_crossing_road_ratio_thresh),
                crossing_lateral_thresh_m=float(args.walker_crossing_lateral_thresh_m),
                road_presence_min_frames=int(args.walker_road_presence_min_frames),
                max_lateral_offset_m=float(args.walker_max_lateral_offset_m),
                dt=float(args.dt),
            )
            vehicles, walker_processing_report = walker_processor.process_walkers(
                vehicles=vehicles,
                vehicle_times=vehicle_times,
                obj_info=obj_info,
            )
            # Print summary
            if walker_processing_report.get("walker_count", 0) > 0:
                stationary_removed = walker_processing_report.get("stationary_removed_count", 0)
                cls_summary = walker_processing_report.get("classification_summary", {})
                comp_summary = walker_processing_report.get("compression_summary", {})
                stab = walker_processing_report.get("stabilization", {})
                print(
                    f"[INFO] Walker processing: {walker_processing_report.get('walker_count', 0)} walkers | "
                    f"stationary/jitter removed: {stationary_removed} | "
                    f"classified: sidewalk={cls_summary.get('sidewalk_consistent', 0)}, "
                    f"crossing={cls_summary.get('crossing', 0)}, "
                    f"jaywalking={cls_summary.get('jaywalking', 0)}, "
                    f"road_walking={cls_summary.get('road_walking', 0)} | "
                    f"compressed: {comp_summary.get('compressed', 0)}, "
                    f"spawn-stabilized: {stab.get('adjusted_count', 0)}"
                )
                if float(comp_summary.get("max_lateral_offset", 0)) > 0:
                    print(
                        f"  [COMPRESS] avg_offset={comp_summary.get('avg_lateral_offset', 0):.2f}m, "
                        f"max_offset={comp_summary.get('max_lateral_offset', 0):.2f}m, "
                        f"lane_spacing={walker_processing_report.get('lane_spacing_m', 0):.1f}m"
                    )
            else:
                print("[INFO] Walker processing: no walkers found in trajectory data.")

    payload = _build_export_payload(
        scenario_dir=scenario_dir,
        selected_map=chosen_map,
        selection_details=selection_scores,
        ego_trajs=ego_trajs,
        ego_times=ego_times,
        vehicles=vehicles,
        vehicle_times=vehicle_times,
        obj_info=obj_info,
        actor_source_subdir=actor_source_subdir,
        actor_orig_vid=actor_orig_vid,
        actor_alias_vids=actor_alias_vids,
        merge_stats=merge_stats,
        timing_optimization=timing_optimization,
        matcher=matcher,
        snap_to_map=bool(args.snap_to_map),
        map_max_points_per_line=int(args.map_max_points_per_line),
        lane_change_cfg=lane_change_cfg,
        vehicle_lane_policy_cfg=vehicle_lane_policy_cfg,
        parked_vehicle_cfg=parked_vehicle_cfg,
        carla_map_layer=carla_map_layer,
        default_dt=float(args.dt),
    )
    if bool(args.lane_correspondence) and bool(payload.get("carla_map")):
        print("[INFO] Building lane correspondence (Hungarian assignment)...")
        driving_types = sorted(
            _parse_lane_type_set(
                args.lane_correspondence_driving_types,
                fallback=["1"],
            )
        )
        # Resolve cache directory — default is alongside this script
        _corr_cache_dir_raw = args.lane_correspondence_cache_dir
        if _corr_cache_dir_raw == "__script_dir__":
            _corr_cache_dir = Path(__file__).resolve().parent / "lane_corr_cache"
        elif _corr_cache_dir_raw == "__output_dir__":
            _corr_cache_dir = out_dir / "lane_corr_cache"
        elif _corr_cache_dir_raw:
            _corr_cache_dir = Path(_corr_cache_dir_raw).expanduser().resolve()
        else:
            _corr_cache_dir = None
        correspondence = _build_lane_correspondence(
            payload=payload,
            candidate_top_k=int(args.lane_correspondence_top_k),
            driving_lane_types=driving_types,
            cache_dir=_corr_cache_dir,
        )
        if bool(getattr(args, "skip_map_snap_compute", False)):
            print("[INFO] Skipping per-frame CARLA projection (--skip-map-snap-compute). 'Use map snapped poses' disabled.")
        else:
            print("[INFO] Starting per-frame CARLA projection from lane correspondence...")
            _corr_proj_t0 = time.monotonic()
            _apply_lane_correspondence_to_payload(payload, correspondence)
            print(f"[INFO] Per-frame CARLA projection done in {time.monotonic() - _corr_proj_t0:.2f}s")
        lc_meta = payload.get("metadata", {}).get("lane_correspondence", {})
        if bool(lc_meta.get("enabled", False)):
            print(
                "[INFO] Lane correspondence: "
                f"mapped_lanes={int(lc_meta.get('mapped_lane_count', 0))} "
                f"usable={int(lc_meta.get('usable_lane_count', 0))} "
                f"mapped_carla_lines={int(lc_meta.get('mapped_carla_line_count', 0))} "
                f"quality={lc_meta.get('quality_counts', {})} "
                f"splits={int(lc_meta.get('split_merge_count', 0))} "
                f"phantom_changes={int(lc_meta.get('total_phantom_lane_changes', 0))} "
                f"conn_preserved={int(lc_meta.get('connectivity_edges_preserved', 0))}/{int(lc_meta.get('connectivity_edges_total', 0))}"
            )
        else:
            print(f"[WARN] Lane correspondence disabled or failed: {lc_meta}")

    print("[INFO] Serializing replay data JSON...")
    _json_t0 = time.monotonic()
    data_json_path = out_dir / "yaml_map_replay_data.json"
    data_json_path.write_text(json.dumps(_sanitize_for_json(payload), indent=2), encoding="utf-8")
    print(f"[OK] Wrote replay data JSON: {data_json_path} ({time.monotonic() - _json_t0:.2f}s)")

    print("[INFO] Building interactive HTML viewer...")
    _html_t0 = time.monotonic()
    html_path = out_dir / "yaml_map_replay_viewer.html"
    html_path.write_text(_build_html(payload), encoding="utf-8")
    print(f"[OK] Wrote interactive HTML: {html_path} ({time.monotonic() - _html_t0:.2f}s)")

    # --- CARLA Route XML Export (optional) ---
    carla_export_report: Dict[str, object] = {"enabled": False}
    if bool(args.export_carla_routes):
        _export_t0 = time.monotonic()
        # Retrieve alignment config for coordinate transform
        grp_align_cfg_path = (
            Path(args.carla_map_offset_json).expanduser().resolve()
            if args.carla_map_offset_json
            else None
        )
        grp_align_cfg = _load_carla_alignment_cfg(grp_align_cfg_path)
        
        # Determine output directory for CARLA routes
        carla_routes_out = (
            Path(args.carla_routes_dir).expanduser().resolve()
            if args.carla_routes_dir
            else out_dir / "carla_routes"
        )
        
        carla_export_report = export_carla_routes(
            ego_tracks=payload.get("ego_tracks", []),
            actor_tracks=payload.get("actor_tracks", []),
            align_cfg=grp_align_cfg,
            out_dir=carla_routes_out,
            town=str(args.carla_town),
            route_id=str(args.carla_route_id),
            actor_control_mode=str(args.carla_actor_control_mode),
            walker_control_mode=str(args.carla_walker_control_mode),
            encode_timing=bool(args.carla_encode_timing),
            snap_to_road=bool(args.carla_snap_to_road),
            static_spawn_only=bool(args.carla_static_spawn_only),
            default_dt=float(args.dt),
        )
        print(f"[INFO] CARLA route export stage done in {time.monotonic() - _export_t0:.2f}s")

    summary_path = out_dir / "yaml_map_selection_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "scenario_dir": str(scenario_dir),
                "selected_map": chosen_map.name,
                "selected_map_path": chosen_map.source_path,
                "map_selection_scores": selection_scores,
                "snap_to_map": bool(args.snap_to_map),
                "id_merge_stats": merge_stats,
                "timing_optimization": timing_optimization,
                "grp_alignment": {
                    k: v for k, v in grp_align_report.items() if k != "actor_details"
                },
                "walker_sidewalk_processing": {
                    "enabled": bool(walker_processing_report.get("enabled", False)),
                    "walker_count": int(walker_processing_report.get("walker_count", 0)),
                    "lane_spacing_m": float(walker_processing_report.get("lane_spacing_m", 0.0)),
                    "classification_summary": walker_processing_report.get("classification_summary", {}),
                    "compression_summary": walker_processing_report.get("compression_summary", {}),
                    "stabilization_summary": {
                        k: v for k, v in walker_processing_report.get("stabilization", {}).items()
                        if k != "details"
                    },
                },
                "vehicle_lane_policy": vehicle_lane_policy_cfg,
                "parked_vehicle_cfg": parked_vehicle_cfg,
                "lane_correspondence": payload.get("metadata", {}).get("lane_correspondence", {}),
                "carla_map_layer": {
                    "enabled": bool(carla_map_layer is not None),
                    "source_path": str(carla_map_layer.get("source_path")) if carla_map_layer else "",
                    "alignment_path": str(carla_map_layer.get("alignment_path")) if carla_map_layer else "",
                    "line_count": int(len(carla_map_layer.get("lines", []))) if carla_map_layer else 0,
                },
                "carla_route_export": {
                    "enabled": bool(carla_export_report.get("enabled", False)),
                    "output_dir": str(carla_export_report.get("output_dir", "")),
                    "town": str(carla_export_report.get("town", "")),
                    "route_id": str(carla_export_report.get("route_id", "")),
                    "actor_control_mode": str(carla_export_report.get("actor_control_mode", "")),
                    "walker_control_mode": str(carla_export_report.get("walker_control_mode", "")),
                    "ego_count": int(carla_export_report.get("ego_count", 0)),
                    "npc_count": int(carla_export_report.get("npc_count", 0)),
                    "walker_count": int(carla_export_report.get("walker_count", 0)),
                    "static_count": int(carla_export_report.get("static_count", 0)),
                    "total_actors": int(carla_export_report.get("total_actors", 0)),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[OK] Wrote selection summary: {summary_path}")
    if bool(grp_align_report.get("enabled", False)):
        grp_report_path = out_dir / "grp_alignment_report.json"
        grp_report_path.write_text(json.dumps(grp_align_report, indent=2, default=str), encoding="utf-8")
        print(f"[OK] Wrote GRP alignment report: {grp_report_path}")
    if bool(walker_processing_report.get("enabled", False)) and walker_processing_report.get("walker_count", 0) > 0:
        walker_report_path = out_dir / "walker_sidewalk_report.json"
        walker_report_path.write_text(json.dumps(walker_processing_report, indent=2, default=str), encoding="utf-8")
        print(f"[OK] Wrote walker sidewalk report: {walker_report_path}")
    if args.timing_optimization_report:
        timing_report_path = Path(args.timing_optimization_report).expanduser().resolve()
        timing_report_path.parent.mkdir(parents=True, exist_ok=True)
        timing_report_path.write_text(json.dumps(timing_optimization, indent=2), encoding="utf-8")
        print(f"[OK] Wrote timing optimization report: {timing_report_path}")
    print("[DONE] yaml_to_map export complete.")
    
    # --- Automatically run CARLA scenario with exported routes ---
    if bool(args.run_custom_eval) and bool(carla_export_report.get("enabled", False)):
        carla_routes_dir = Path(carla_export_report.get("output_dir", ""))
        if carla_routes_dir.exists():
            # Derive scenario name from scenario_dir for results folder naming
            scenario_name = scenario_dir.name
            _run_carla_scenario(
                routes_dir=carla_routes_dir,
                port=int(args.eval_port) if args.eval_port else int(args.carla_port),
                planner=str(args.eval_planner) if args.eval_planner else None,
                overwrite=bool(args.eval_overwrite),
                actor_control_mode=str(args.carla_actor_control_mode),
                walker_control_mode=str(args.carla_walker_control_mode),
                capture_logreplay_images=bool(getattr(args, 'capture_logreplay_images', False)),
                capture_every_sensor_frame=bool(getattr(args, 'capture_every_sensor_frame', False)),
                scenario_name=scenario_name,
            )
            
            # Generate videos if requested (single scenario mode)
            if bool(getattr(args, 'generate_videos', False)):
                print(f"\n[INFO] Generating videos for scenario: {scenario_name}")
                video_paths = _generate_scenario_videos(
                    scenario_dir,
                    scenario_name,
                    fps=float(getattr(args, 'video_fps', 10)),
                    resize_factor=int(getattr(args, 'video_resize_factor', 2)),
                )
                if video_paths:
                    print(f"[OK] Generated {len(video_paths)} videos")
                else:
                    print("[WARN] No videos generated (no image directories found)")
        else:
            print(f"[WARN] CARLA routes directory not found: {carla_routes_dir}")
            print("[WARN] Skipping run_custom_eval.")


def _run_carla_scenario(
    routes_dir: Path,
    port: int,
    planner: Optional[str] = None,
    overwrite: bool = True,
    actor_control_mode: str = "policy",
    walker_control_mode: str = "policy",
    capture_logreplay_images: bool = False,
    capture_every_sensor_frame: bool = False,
    scenario_name: Optional[str] = None,
) -> None:
    """Run the CARLA scenario using tools/run_custom_eval.py.
    
    When planner='log-replay', ego follows exact trajectory with timing.
    Actor control mode determines how NPCs behave:
      - 'policy': Use CARLA's AI planners (WaypointFollower)
      - 'replay': Use transform log replay for NPCs
    
    Args:
        routes_dir: Path to the routes directory
        port: CARLA server port
        planner: Planner type ('log-replay', 'autopilot', etc.)
        overwrite: Overwrite existing results
        actor_control_mode: 'policy' or 'replay' for NPCs
        walker_control_mode: 'policy' or 'replay' for walkers
        capture_logreplay_images: Save log-replay images
        capture_every_sensor_frame: Save all sensor frames
        scenario_name: Name for results folder (defaults to routes_dir name)
    """
    repo_root = Path(__file__).resolve().parents[2]
    run_custom_eval_script = repo_root / "tools" / "run_custom_eval.py"
    
    if not run_custom_eval_script.exists():
        print(f"[ERROR] run_custom_eval.py not found: {run_custom_eval_script}")
        return
    
    # Use scenario_name for results folder, fall back to routes_dir name
    results_tag = scenario_name if scenario_name else routes_dir.name
    
    python_bin = sys.executable
    cmd = [
        python_bin,
        str(run_custom_eval_script),
        "--routes-dir", str(routes_dir),
        "--port", str(port),
        "--custom-actor-control-mode", actor_control_mode,
        "--results-tag", results_tag,
    ]
    
    if overwrite:
        cmd.append("--overwrite")
    
    # Always normalize actor z so walkers/pedestrians aren't spawned underground
    cmd.append("--normalize-actor-z")
    
    if planner:
        cmd.extend(["--planner", planner])
    
    # For replay mode on actors, enable log replay actors flag
    if actor_control_mode == "replay":
        cmd.append("--log-replay-actors")
    
    # Image capture flags - enable both for proper dense capture
    if capture_logreplay_images:
        cmd.append("--capture-logreplay-images")
        cmd.append("--capture-every-sensor-frame")  # Required for dense image capture
    elif capture_every_sensor_frame:
        cmd.append("--capture-every-sensor-frame")
    
    print(f"[INFO] Running CARLA scenario: {' '.join(cmd)}")
    print(f"[INFO]   Scenario name: {results_tag}")
    print(f"[INFO]   Planner: {planner or 'default'}")
    print(f"[INFO]   Actor control: {actor_control_mode}")
    print(f"[INFO]   Walker control: {walker_control_mode}")
    if capture_logreplay_images or capture_every_sensor_frame:
        print(f"[INFO]   Image capture: logreplay={capture_logreplay_images}, every_frame={capture_logreplay_images or capture_every_sensor_frame}")
    
    try:
        subprocess.run(cmd, check=True)
        print("[OK] CARLA scenario completed successfully.")
    except subprocess.CalledProcessError as exc:
        print(f"[WARN] run_custom_eval.py failed with exit code {exc.returncode}")
    except FileNotFoundError as exc:
        print(f"[ERROR] Could not run scenario: {exc}")


if __name__ == "__main__":
    main()
