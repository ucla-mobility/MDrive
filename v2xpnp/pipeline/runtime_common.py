#!/usr/bin/env python3
"""
Plot raw YAML trajectories on the V2XPNP vector map.

A simplified version of yaml_to_map that:
- Loads raw trajectories from YAML scenario directories
- Optionally applies a coordinate offset (tx, ty, tz) and rotation (yaw)
- Selects the best matching V2XPNP map (corridors or intersection)
- Outputs an interactive HTML visualization
- Supports multiple scenarios with a dropdown selector

Usage:
    # Single scenario
    python -m v2xpnp.pipeline.entrypoint /path/to/scenario

    # Multiple scenarios (directory of scenarios)
    python -m v2xpnp.pipeline.entrypoint /path/to/train4 --multi
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import math
import os
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:  # pragma: no cover
    cKDTree = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v2xpnp.pipeline.trajectory_ingest import (
    Waypoint,
    apply_se2,
    build_trajectories,
    pick_yaml_dirs,
    list_yaml_timesteps,
)
from v2xpnp.pipeline import route_export as ytm

_LANE_CORR_DIAG_MOD: object = None

# Default knob profile from optimizer trial 23 (run:
# results/planner_loop_train4_20260228_012216). Env vars still take priority.
_TRIAL23_DEFAULT_OVERRIDES: Dict[str, float] = {
    "V2X_ALIGN_DIRECT_TURN_MIN_YAW_CHANGE_DEG": 16.0,
    "V2X_ALIGN_EARLY_LANE_CHANGE_EXTRA_PENALTY": 140.0,
    "V2X_ALIGN_INTERSECTION_BLEND_MIN_LATERAL_EVIDENCE_M": 0.8,
    "V2X_ALIGN_INTERSECTION_BLEND_MIN_YAW_EVIDENCE_DEG": 12.0,
    "V2X_ALIGN_LANE_CHANGE_HORIZON_PENALTY": 120.0,
    "V2X_ALIGN_LANE_CHANGE_JUMP_ABS_M": 1.8,
    "V2X_ALIGN_LANE_CHANGE_JUMP_GUARD_MIN_GAIN_M": 0.55,
    "V2X_ALIGN_LANE_CHANGE_JUMP_GUARD_PENALTY": 420.0,
    "V2X_ALIGN_LANE_CHANGE_JUMP_RATIO": 2.4,
    "V2X_ALIGN_LANE_CHANGE_MIN_GAIN_M": 0.9,
    "V2X_ALIGN_LANE_CHANGE_WEAK_EVIDENCE_PENALTY": 120.0,
    "V2X_ALIGN_OPPOSITE_SWITCH_MIN_GAIN_M": 1.2,
    "V2X_ALIGN_OPPOSITE_SWITCH_PENALTY": 700.0,
    "V2X_ALIGN_RETIME_TRANSITION_MIN_IMPROVEMENT_M": 0.45,
    "V2X_ALIGN_SIGN_FLIP_STRICT_MIN_GAIN_M": 1.0,
    "V2X_ALIGN_SIGN_FLIP_STRICT_PENALTY": 520.0,
    "V2X_ALIGN_WEAK_JUMP_MIN_GAIN_M": 0.9,
    "V2X_ALIGN_WEAK_JUMP_RATIO": 2.4,
    "V2X_CARLA_CORR_SCORE_MARGIN_GOOD": 0.85,
    "V2X_CARLA_CORR_SCORE_MARGIN_WEAK": 0.55,
    "V2X_CARLA_ENABLE_MICRO_JITTER_SMOOTH": 0.0,
    "V2X_CARLA_FAR_MAX_NEAREST": 8.0,
    "V2X_CARLA_INTERSECTION_SHAPE_MAX_QUERY_OFFSET_M": 2.8,
    "V2X_CARLA_INTERSECTION_SHAPE_MAX_RAW_OFFSET_M": 2.8,
    "V2X_CARLA_INTERSECTION_SHAPE_MAX_RAW_WORSEN_PEAK_M": 0.8,
    "V2X_CARLA_INTERSECTION_SHAPE_MAX_RAW_WORSEN_PER_FRAME_M": 0.35,
    "V2X_CARLA_INTERSECTION_SHAPE_MAX_WORSEN_PER_FRAME_M": 0.55,
    "V2X_CARLA_NEAREST_CONT_DIST_SLACK": 0.8,
    "V2X_CARLA_NEAREST_CONT_SCORE_SLACK": 0.6,
    "V2X_CARLA_OPPOSITE_REJECT_DEG": 170.0,
    "V2X_CARLA_SMOOTH_COST_SLACK_BASE": 0.5,
    "V2X_CARLA_SMOOTH_COST_SLACK_PER_FRAME": 0.35,
    "V2X_CARLA_SMOOTH_MAX_MID_RUN": 12.0,
    "V2X_CARLA_SMOOTH_MIN_STABLE_NEIGHBOR": 30.0,
    "V2X_CARLA_TRANSITION_COST_SLACK_BASE": 1.2,
    "V2X_CARLA_TRANSITION_COST_SLACK_PER_FRAME": 0.45,
    "V2X_CARLA_TRANSITION_MIN_NEIGHBOR": 10.0,
    "V2X_CARLA_TRANSITION_SPIKE_MAX_FRAMES": 3.0,
    "V2X_CARLA_WEAK_SWITCH_GUARD_MIN_DIST_GAIN_M": 1.1,
    "V2X_CARLA_WEAK_SWITCH_GUARD_MIN_SCORE_GAIN": 0.9,
    "V2X_CARLA_WEAK_SWITCH_GUARD_RAW_STEP_MAX_M": 0.9,
    "V2X_CARLA_WRONG_WAY_REJECT_DEG": 178.0,
}


# =============================================================================
# Utility Functions
# =============================================================================


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    default_value = _TRIAL23_DEFAULT_OVERRIDES.get(str(name), float(default))
    raw = os.environ.get(str(name), None)
    if raw is None:
        return float(default_value)
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return float(default_value)
    if not math.isfinite(val):
        return float(default_value)
    return float(val)


def _env_int(
    name: str,
    default: int,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    default_value = _TRIAL23_DEFAULT_OVERRIDES.get(str(name), int(default))
    raw = os.environ.get(str(name), None)
    if raw is None:
        out = int(float(default_value))
    else:
        try:
            out = int(float(raw))
        except (TypeError, ValueError):
            out = int(float(default_value))
    if minimum is not None:
        out = max(int(minimum), int(out))
    if maximum is not None:
        out = min(int(maximum), int(out))
    return int(out)


def _load_lane_corr_diag_module():
    """Lazy-load diagnostics helpers to avoid hard dependency/circular imports."""
    global _LANE_CORR_DIAG_MOD
    if _LANE_CORR_DIAG_MOD is None:
        try:
            from v2xpnp.pipeline import correspondence_diagnostics as lcd  # type: ignore
            _LANE_CORR_DIAG_MOD = lcd
        except Exception:
            _LANE_CORR_DIAG_MOD = False
    if _LANE_CORR_DIAG_MOD is False:
        return None
    return _LANE_CORR_DIAG_MOD


def _compute_bbox_xy(points: np.ndarray) -> Tuple[float, float, float, float]:
    if points.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    xy = points.reshape(-1, 2) if points.ndim == 1 else points[:, :2]
    min_x = float(np.min(xy[:, 0]))
    max_x = float(np.max(xy[:, 0]))
    min_y = float(np.min(xy[:, 1]))
    max_y = float(np.max(xy[:, 1]))
    return (min_x, max_x, min_y, max_y)


def _sanitize_for_json(obj):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _is_negative_subdir(path: Path) -> bool:
    """Check if the subdir name starts with a minus sign (negative scenario)."""
    name = path.name or ""
    return name.startswith("-")


# =============================================================================
# Map Loading
# =============================================================================


class _MapStubBase:
    """Stub for unpickling map objects without needing full dependencies."""
    pass


class _MapUnpickler(pickle.Unpickler):
    """Custom unpickler that maps unknown classes to stubs."""

    def find_class(self, module: str, name: str):
        # Return stub for vector map types
        if "vector_map" in module.lower() or "map" in module.lower():
            return type(name, (_MapStubBase,), {})
        return super().find_class(module, name)


@dataclass
class LaneFeature:
    index: int
    uid: str
    road_id: int
    lane_id: int
    lane_type: str
    polyline: np.ndarray  # shape (N, 3)
    boundary: np.ndarray  # shape (M, 3) or empty
    entry_lanes: List[str]
    exit_lanes: List[str]


@dataclass
class VectorMapData:
    name: str
    source_path: str
    lanes: List[LaneFeature]
    bbox: Tuple[float, float, float, float]  # (min_x, max_x, min_y, max_y)


def _extract_xyz_points(items: object) -> List[Tuple[float, float, float]]:
    """Extract (x, y, z) points from various map data structures."""
    out: List[Tuple[float, float, float]] = []
    if items is None:
        return out
    if isinstance(items, np.ndarray):
        arr = items.reshape(-1, 3) if items.ndim == 1 and items.size % 3 == 0 else items
        for row in arr:
            if len(row) >= 3:
                out.append((float(row[0]), float(row[1]), float(row[2])))
        return out
    for item in items:
        if hasattr(item, "x") and hasattr(item, "y") and hasattr(item, "z"):
            out.append((float(item.x), float(item.y), float(item.z)))
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            out.append((float(item[0]), float(item[1]), float(item[2])))
    return out


def load_vector_map(path: Path) -> VectorMapData:
    """Load a V2XPNP vector map from pickle file."""
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


# =============================================================================
# Map Selection
# =============================================================================


def _collect_reference_points(
    ego_trajs: Sequence[Sequence[Waypoint]],
    vehicles: Dict[int, Sequence[Waypoint]],
) -> np.ndarray:
    """Collect sample points from ego and vehicle trajectories."""
    pts: List[Tuple[float, float]] = []
    for traj in ego_trajs:
        for wp in traj:
            pts.append((float(wp.x), float(wp.y)))
    for traj in vehicles.values():
        for wp in traj:
            pts.append((float(wp.x), float(wp.y)))
    if not pts:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)


def _sample_points(points_xy: np.ndarray, max_count: int) -> np.ndarray:
    if points_xy.shape[0] <= max_count:
        return points_xy
    indices = np.linspace(0, len(points_xy) - 1, max_count, dtype=int)
    return points_xy[indices]


def _outside_bbox_ratio(points_xy: np.ndarray, bbox: Tuple[float, float, float, float], margin: float) -> float:
    if points_xy.size == 0:
        return 0.0
    min_x, max_x, min_y, max_y = bbox
    outside = (
        (points_xy[:, 0] < (min_x - margin))
        | (points_xy[:, 0] > (max_x + margin))
        | (points_xy[:, 1] < (min_y - margin))
        | (points_xy[:, 1] > (max_y + margin))
    )
    return float(np.mean(outside.astype(np.float64)))


class LaneMatcher:
    """Lane matcher for computing nearest distances and projecting points onto lanes."""

    def __init__(self, map_data: VectorMapData):
        self.map_data = map_data
        self._lane_polys = [lane.polyline for lane in map_data.lanes]
        
        # Build vertex index for fast lookup
        vertex_xy: List[Tuple[float, float]] = []
        vertex_lane_idx: List[int] = []
        for lane_idx, lane in enumerate(map_data.lanes):
            for pt in lane.polyline:
                vertex_xy.append((float(pt[0]), float(pt[1])))
                vertex_lane_idx.append(lane_idx)
        
        self.vertex_xy = np.asarray(vertex_xy, dtype=np.float64) if vertex_xy else np.zeros((0, 2), dtype=np.float64)
        self.vertex_lane_idx = np.asarray(vertex_lane_idx, dtype=np.int32) if vertex_lane_idx else np.zeros((0,), dtype=np.int32)
        
        # Build lane connectivity graph
        self._build_lane_graph()

    def _build_lane_graph(self):
        """Build lane connectivity graph for smooth transitions."""
        self.lane_successors: Dict[int, List[int]] = {}
        self.lane_predecessors: Dict[int, List[int]] = {}
        
        uid_to_idx: Dict[str, int] = {}
        succ_sets: Dict[int, set] = {}
        pred_sets: Dict[int, set] = {}
        for idx, lane in enumerate(self.map_data.lanes):
            uid_to_idx[lane.uid] = idx
            succ_sets[idx] = set()
            pred_sets[idx] = set()
        
        for idx, lane in enumerate(self.map_data.lanes):
            for exit_uid in lane.exit_lanes:
                if exit_uid in uid_to_idx:
                    succ_idx = uid_to_idx[exit_uid]
                    succ_sets[idx].add(succ_idx)
                    pred_sets[succ_idx].add(idx)
            # Some maps only encode entry_lanes. Mirror them into forward edges.
            for entry_uid in lane.entry_lanes:
                if entry_uid in uid_to_idx:
                    pred_idx = uid_to_idx[entry_uid]
                    succ_sets[pred_idx].add(idx)
                    pred_sets[idx].add(pred_idx)

        self.lane_successors = {idx: sorted(vals) for idx, vals in succ_sets.items()}
        self.lane_predecessors = {idx: sorted(vals) for idx, vals in pred_sets.items()}

    def nearest_vertex_distance(self, points_xy: np.ndarray) -> np.ndarray:
        if points_xy.size == 0 or self.vertex_xy.size == 0:
            return np.array([], dtype=np.float64)
        # Use broadcasting to compute distances
        dists = np.zeros(len(points_xy), dtype=np.float64)
        for i, (px, py) in enumerate(points_xy):
            d = np.sqrt((self.vertex_xy[:, 0] - px) ** 2 + (self.vertex_xy[:, 1] - py) ** 2)
            dists[i] = float(np.min(d))
        return dists

    def _nearest_vertex_candidates(self, x: float, y: float, top_k: int) -> List[int]:
        """Find top_k nearest lane candidates based on vertex proximity."""
        if self.vertex_xy.shape[0] == 0:
            return []
        k = max(1, min(int(top_k), int(self.vertex_xy.shape[0])))
        
        # Compute distances to all vertices
        diff = self.vertex_xy - np.asarray([x, y], dtype=np.float64)[None, :]
        d2 = np.sum(diff * diff, axis=1)
        order = np.argsort(d2)[:k * 3]  # Get more candidates to account for lane overlap
        
        # Extract unique lane indices
        lane_candidates: List[int] = []
        seen: set = set()
        for idx in order:
            lane_idx = int(self.vertex_lane_idx[idx])
            if lane_idx in seen:
                continue
            seen.add(lane_idx)
            lane_candidates.append(lane_idx)
            if len(lane_candidates) >= k:
                break
        return lane_candidates

    @staticmethod
    def _project_point_to_polyline(polyline: np.ndarray, x: float, y: float) -> Optional[Dict[str, float]]:
        """Project a point onto a polyline, returning the nearest point on the line."""
        if polyline.shape[0] == 0:
            return None
        if polyline.shape[0] == 1:
            px = float(polyline[0, 0])
            py = float(polyline[0, 1])
            pz = float(polyline[0, 2]) if polyline.shape[1] > 2 else 0.0
            dist = math.hypot(px - x, py - y)
            return {"x": px, "y": py, "z": pz, "yaw": 0.0, "dist": dist, "segment_idx": 0.0}

        p0 = polyline[:-1, :2]
        p1 = polyline[1:, :2]
        seg = p1 - p0
        seg_len2 = np.sum(seg * seg, axis=1)
        valid = seg_len2 > 1e-12
        
        if not np.any(valid):
            px = float(polyline[0, 0])
            py = float(polyline[0, 1])
            pz = float(polyline[0, 2]) if polyline.shape[1] > 2 else 0.0
            dist = math.hypot(px - x, py - y)
            return {"x": px, "y": py, "z": pz, "yaw": 0.0, "dist": dist, "segment_idx": 0.0}

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
        z0 = float(polyline[best, 2]) if polyline.shape[1] > 2 else 0.0
        z1 = float(polyline[best + 1, 2]) if polyline.shape[1] > 2 else 0.0
        snap_z = z0 + best_t * (z1 - z0)
        seg_dx = float(seg[best, 0])
        seg_dy = float(seg[best, 1])
        
        # Compute yaw from segment direction
        if abs(seg_dx) + abs(seg_dy) > 1e-9:
            yaw = math.degrees(math.atan2(seg_dy, seg_dx))
            # Normalize to [-180, 180]
            while yaw > 180:
                yaw -= 360
            while yaw <= -180:
                yaw += 360
        else:
            yaw = 0.0
            
        return {
            "x": snap_x,
            "y": snap_y,
            "z": snap_z,
            "yaw": yaw,
            "dist": math.sqrt(max(0.0, float(dist2[best]))),
            "segment_idx": float(best + best_t),
        }

    def match(self, x: float, y: float, z: float = 0.0, lane_top_k: int = 8) -> Optional[Dict[str, object]]:
        """Find the nearest lane and project the point onto it."""
        lane_candidates = self._nearest_vertex_candidates(float(x), float(y), top_k=max(6, lane_top_k))
        if not lane_candidates:
            return None

        best_match: Optional[Dict[str, object]] = None
        for lane_idx in lane_candidates:
            lane = self.map_data.lanes[lane_idx]
            proj = self._project_point_to_polyline(lane.polyline, float(x), float(y))
            if proj is None:
                continue
            match = {
                "lane_index": int(lane_idx),
                "lane_uid": lane.uid,
                "lane_type": lane.lane_type,
                "x": float(proj["x"]),
                "y": float(proj["y"]),
                "z": float(proj["z"]),
                "yaw": float(proj["yaw"]),
                "dist": float(proj["dist"]),
            }
            if best_match is None or float(match["dist"]) < float(best_match["dist"]):
                best_match = match
        return best_match
    
    def get_all_candidates(self, x: float, y: float, top_k: int = 12) -> List[Dict[str, object]]:
        """Get all lane candidates with their projections."""
        lane_candidates = self._nearest_vertex_candidates(float(x), float(y), top_k=top_k)
        results: List[Dict[str, object]] = []
        for lane_idx in lane_candidates:
            lane = self.map_data.lanes[lane_idx]
            proj = self._project_point_to_polyline(lane.polyline, float(x), float(y))
            if proj is None:
                continue
            results.append({
                "lane_index": int(lane_idx),
                "lane_uid": lane.uid,
                "lane_type": lane.lane_type,
                "road_id": lane.road_id,
                "lane_id": lane.lane_id,
                "x": float(proj["x"]),
                "y": float(proj["y"]),
                "z": float(proj["z"]),
                "yaw": float(proj["yaw"]),
                "dist": float(proj["dist"]),
                "segment_idx": float(proj["segment_idx"]),
            })
        return results


class TrajectoryAligner:
    """
    Trajectory-level lane alignment using Viterbi-style dynamic programming.
    
    Key principles:
    1. Heavily penalize lane changes (lane_id change) - not road segment transitions
    2. Use high-confidence future moments to inform past lane assignments
    3. Minimize total displacement while maintaining lane continuity
    4. Road segment transitions (different road_id, same lane_id) are FREE
    5. Heavily penalize quick lane changes (ping-pong: lane A→B→A)
    """
    
    # Tuning parameters
    LANE_CHANGE_PENALTY = 80.0      # Base penalty for changing lanes (increased)
    QUICK_RETURN_PENALTY = 200.0    # Extra penalty for A→B→A pattern
    MIN_DWELL_FRAMES = 15           # Minimum frames to stay in a lane before changing (was 10)
    RECENT_HISTORY_FRAMES = 15      # How far back to look for ping-pong detection
    YAW_CHANGE_THRESHOLD = 20.0     # Degrees: significant yaw change that might indicate lane change (increased)
    DIST_WEIGHT = 1.0               # Weight for projection distance
    YAW_MISMATCH_WEIGHT = 0.3       # Weight for yaw mismatch with lane
    CONFIDENCE_THRESHOLD = 2.0      # Distance threshold for high confidence
    DISPLACEMENT_WEIGHT = 0.5       # Weight for displacement consistency
    CONNECTIVITY_BONUS = -5.0       # Bonus for using connected lanes (negative = reward)
    DISCONNECTED_PENALTY = 180.0    # Penalty for disconnected transitions when connected options exist
    BIDIRECTIONAL_YAW_MATCH = True  # Treat opposite lane direction as valid when geometry is reversed
    
    # Confidence window parameters (new)
    CONFIDENCE_DISTANCE_GAP = 1.5   # Min gap (m) to 2nd-best for high confidence
    CONFIDENT_FRAME_WINDOW = 10     # Frames to look forward/backward for context
    MIN_CONFIDENT_FRAMES = 5        # Minimum confident frames to establish anchor
    PHASE_TRANSITION_FRAMES = 5     # Frames for phase transition smoothing
    SPURIOUS_DIP_FRAMES = 8         # Max frames for a "spurious" lane dip
    
    # Lateral continuity parameters (new) - penalize skipping lanes
    LANE_SKIP_PENALTY = 150.0       # Penalty per lane skipped (e.g., L1->L-3 skips L-2)
    SIGN_CHANGE_BASE = 1            # Base "distance" for crossing center (L1 <-> L-1)
    SHORT_CONNECTOR_FRAMES = 10     # Max duration for removable A->B->C connector road B
    CONNECTOR_COST_SLACK = 4.0      # Allowed absolute distance increase when removing connector road

    # Intersection multi-lane refinement (new)
    REFINE_MULTILANE_RUNS = True
    RUN_LANE_CHANGE_PENALTY = 12.0  # Penalty for lane changes within one road run
    RUN_START_CONN_WEIGHT = 0.6     # Weight for matching predecessor road at run start
    RUN_END_CONN_WEIGHT = 1.2       # Weight for matching successor road at run end
    LANE_CHANGE_BLEND_FRAMES = 6    # Half-window for smoothing lane changes in intersections
    ROAD_FLICKER_FRAMES = 16        # Max run length for A->B->A road flicker removal
    ROAD_FLICKER_COST_SLACK = 1.0   # Base allowed distance increase for flicker smoothing
    ROAD_FLICKER_PER_FRAME_SLACK = 0.30  # Extra allowed increase per smoothed frame
    ROAD_FLICKER_LONG_MAX_FRAMES = 120   # Max run length for strong-evidence long A->B->A compression
    ROAD_FLICKER_LONG_GAIN_PER_FRAME = 0.50  # Required per-frame distance gain for long compression
    LANE_FLICKER_FRAMES = 12         # Max lane-run length for A->B->A lane flicker smoothing (was 4; real lane changes take >1.2s)
    LANE_FLICKER_COST_SLACK = 0.60   # Base allowed increase when removing a short lane flicker
    LANE_FLICKER_PER_FRAME_SLACK = 0.30  # Extra allowed increase per smoothed lane-flicker frame
    LAST_SECOND_LANE_FLICKER_FRAMES = 10  # Max run length for transition-bridge lane flickers (was 6)
    LAST_SECOND_LANE_FLICKER_COST_SLACK = 0.50  # Base allowed increase for bridge lane flickers
    LAST_SECOND_LANE_FLICKER_PER_FRAME_SLACK = 1.10  # Extra allowed increase per bridge frame
    PREMATURE_LANE_CHANGE_LOOKAHEAD_FRAMES = 26  # Max lookahead when delaying early lane switches
    PREMATURE_LANE_CHANGE_CONFIRM_FRAMES = 3     # Sustained evidence frames required before switching
    PREMATURE_LANE_CHANGE_GAIN_M = 0.70          # Required lane-B distance advantage over lane-A
    PREMATURE_LANE_CHANGE_TRIGGER_MARGIN_M = 0.50  # Trigger delay only if current switch is this much worse than lane-A
    PREMATURE_LANE_CHANGE_COST_SLACK = 1.20      # Allowed total fit increase when delaying switch
    PREMATURE_LANE_CHANGE_PER_FRAME_SLACK = 0.20 # Extra allowed increase per delayed frame
    EDGE_LANE_SPIKE_FRAMES = 6       # Max head/tail run length for segment-edge lane spike smoothing
    EDGE_LANE_SPIKE_MIN_NEIGHBOR_FRAMES = 8  # Min stable neighbor run to lock edge lane
    EDGE_LANE_SPIKE_COST_SLACK = 0.50  # Base allowed increase for edge lane spike smoothing
    EDGE_LANE_SPIKE_PER_FRAME_SLACK = 1.10  # Extra allowed increase per edge frame
    TERMINAL_ROAD_TAIL_FRAMES = 4    # Max end-of-segment same-lane road tail to collapse
    TERMINAL_ROAD_TAIL_COST_SLACK = 0.50  # Allowed increase when collapsing terminal road tail
    APPROACH_LANE_LOCK_MIN_TAIL_FRAMES = 8   # Min tail frames on final lane before road exit
    APPROACH_LANE_LOCK_MAX_RUN_FRAMES = 45   # Max combined frames for two-lane approach run
    APPROACH_LANE_LOCK_PER_FRAME_SLACK = 1.30  # Allowed added fit error per locked frame
    APPROACH_LANE_LOCK_MAX_POINT_DIST = 6.0    # Hard cap for worst locked-frame mismatch
    APPROACH_LANE_LOCK_MAX_POINT_DELTA = 3.0   # Allowed increase over original worst mismatch
    TURN_CHAIN_MAX_FRAMES = 64      # Max frames to compress same-lane connector road chains
    TURN_CHAIN_COST_SLACK = 5.0     # Allowed distance increase for turn-chain compression
    TURN_CONNECTOR_MAX_FRAMES = 20   # Max run length treated as turn connector road
    TURN_MIN_FRAMES = 8              # Min frames for synthetic turn segment
    TURN_MIN_YAW_CHANGE = 20.0       # Min heading change (deg) to trigger turn synthesis
    TURN_BEZIER_SCALE = 0.35         # Tangent scale factor for synthetic turn arc
    TURN_END_BLEND_FRAMES = 10       # Frames to include from stable destination run
    TURN_SYNTH_COST_SLACK = 2.0      # Base allowed total distance increase for synthetic arc
    TURN_SYNTH_PER_FRAME_SLACK = 1.0   # Extra allowed distance increase per synthesized frame
    TURN_SYNTH_MAX_POINT_DIST = 10.0   # Hard cap for worst-frame synthetic mismatch (meters)
    JUMP_STALL_RAW_MIN_STEP = 0.25   # Min raw frame displacement for jump-stall detection
    JUMP_STALL_RAW_MAX_STEP = 2.50   # Ignore true raw teleports larger than this
    JUMP_STALL_MIN_JUMP_M = 2.20     # Absolute snapped jump threshold
    JUMP_STALL_JUMP_RATIO = 2.80     # Snapped jump must exceed ratio * raw_step
    JUMP_STALL_STALL_STEP_M = 0.12   # Near-zero snapped step treated as stall
    JUMP_STALL_RESUME_STEP_M = 0.35  # Snapped step indicating movement resumed
    JUMP_STALL_MIN_STALL_FRAMES = 2  # Require at least this many stalled frames
    JUMP_STALL_LOOKAHEAD_FRAMES = 14 # Max frames to search for resume anchor
    JUMP_STALL_COST_SLACK = 0.80     # Base allowed distance increase for continuity patch
    JUMP_STALL_PER_FRAME_SLACK = 0.35  # Additional allowed increase per patched frame
    JUMP_STALL_MAX_POINT_DELTA = 1.80  # Max per-point mismatch increase vs original
    
    def __init__(
        self,
        matcher: LaneMatcher,
        verbose: bool = False,
        allowed_lane_types: Optional[set] = None,
    ):
        self.matcher = matcher
        self.verbose = verbose
        # Make intersection-turn synthesis tunable without changing defaults.
        self.TURN_CHAIN_MAX_FRAMES = _env_int(
            "V2X_ALIGN_TURN_CHAIN_MAX_FRAMES",
            int(self.TURN_CHAIN_MAX_FRAMES),
            minimum=1,
            maximum=240,
        )
        self.TURN_CHAIN_COST_SLACK = _env_float(
            "V2X_ALIGN_TURN_CHAIN_COST_SLACK",
            float(self.TURN_CHAIN_COST_SLACK),
        )
        self.TURN_CONNECTOR_MAX_FRAMES = _env_int(
            "V2X_ALIGN_TURN_CONNECTOR_MAX_FRAMES",
            int(self.TURN_CONNECTOR_MAX_FRAMES),
            minimum=1,
            maximum=120,
        )
        self.TURN_MIN_FRAMES = _env_int(
            "V2X_ALIGN_TURN_MIN_FRAMES",
            int(self.TURN_MIN_FRAMES),
            minimum=3,
            maximum=120,
        )
        self.TURN_MIN_YAW_CHANGE = _env_float(
            "V2X_ALIGN_TURN_MIN_YAW_CHANGE_DEG",
            float(self.TURN_MIN_YAW_CHANGE),
        )
        self.TURN_BEZIER_SCALE = _env_float(
            "V2X_ALIGN_TURN_BEZIER_SCALE",
            float(self.TURN_BEZIER_SCALE),
        )
        self.TURN_END_BLEND_FRAMES = _env_int(
            "V2X_ALIGN_TURN_END_BLEND_FRAMES",
            int(self.TURN_END_BLEND_FRAMES),
            minimum=1,
            maximum=120,
        )
        self.TURN_SYNTH_COST_SLACK = _env_float(
            "V2X_ALIGN_TURN_SYNTH_COST_SLACK",
            float(self.TURN_SYNTH_COST_SLACK),
        )
        self.TURN_SYNTH_PER_FRAME_SLACK = _env_float(
            "V2X_ALIGN_TURN_SYNTH_PER_FRAME_SLACK",
            float(self.TURN_SYNTH_PER_FRAME_SLACK),
        )
        self.TURN_SYNTH_MAX_POINT_DIST = _env_float(
            "V2X_ALIGN_TURN_SYNTH_MAX_POINT_DIST",
            float(self.TURN_SYNTH_MAX_POINT_DIST),
        )
        self.allowed_lane_types: Optional[set] = (
            {str(v) for v in allowed_lane_types} if allowed_lane_types is not None else None
        )
        self._road_lane_features: Dict[Tuple[int, int], List[LaneFeature]] = {}
        self._road_lane_ids: Dict[int, List[int]] = {}
        self._build_road_lane_index()
        self.road_successors = self._build_road_successors()

    def _filter_candidates_by_lane_type(
        self,
        candidates: List[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        """Apply optional lane_type whitelist to candidate lanes."""
        if self.allowed_lane_types is None:
            return candidates
        filtered = [
            c for c in candidates
            if str(c.get("lane_type", "")) in self.allowed_lane_types
        ]
        return filtered

    def _build_road_lane_index(self) -> None:
        """Index map lanes by (road_id, lane_id) for geometry-based refinement."""
        road_to_ids: Dict[int, set] = {}
        for lane in self.matcher.map_data.lanes:
            key = (int(lane.road_id), int(lane.lane_id))
            if key not in self._road_lane_features:
                self._road_lane_features[key] = []
            self._road_lane_features[key].append(lane)
            rid = int(lane.road_id)
            if rid not in road_to_ids:
                road_to_ids[rid] = set()
            road_to_ids[rid].add(int(lane.lane_id))
        self._road_lane_ids = {rid: sorted(list(ids)) for rid, ids in road_to_ids.items()}

    def _build_road_successors(self) -> Dict[int, set]:
        """Build road-level connectivity from lane-level successors."""
        road_succ: Dict[int, set] = {}
        lanes = self.matcher.map_data.lanes
        for from_lane_idx, succ_indices in self.matcher.lane_successors.items():
            if from_lane_idx < 0 or from_lane_idx >= len(lanes):
                continue
            from_road = int(lanes[from_lane_idx].road_id)
            if from_road not in road_succ:
                road_succ[from_road] = set()
            for to_lane_idx in succ_indices:
                if to_lane_idx < 0 or to_lane_idx >= len(lanes):
                    continue
                to_road = int(lanes[to_lane_idx].road_id)
                road_succ[from_road].add(to_road)
        return road_succ
    
    def _lane_lateral_distance(self, from_lid: int, to_lid: int) -> int:
        """
        Compute the lateral distance (number of lane crossings) between two lane_ids.
        
        Lane numbering: ..., -3, -2, -1, [center], 1, 2, 3, ...
        There is no lane 0, so L1 and L-1 are adjacent across the center.
        
        Returns the number of lane boundaries crossed.
        """
        if from_lid == to_lid:
            return 0
        
        # Convert to linear index (accounting for no lane 0)
        def to_index(lid):
            return lid - 1 if lid > 0 else lid  # L1->0, L2->1, L-1->-1, L-2->-2
        
        return abs(to_index(from_lid) - to_index(to_lid))
    
    def _compute_lane_confidence(
        self, 
        frame_candidates: List[List[Dict[str, object]]]
    ) -> List[Tuple[int, float]]:
        """
        Compute per-frame confidence for lane assignment.
        
        Returns list of (best_lane_id, confidence) where:
        - confidence > 1.0: high confidence (clear winner)
        - confidence ~= 1.0: ambiguous (close alternatives)
        - confidence < 1.0: uncertain
        """
        confidences = []
        for candidates in frame_candidates:
            if not candidates:
                confidences.append((None, 0.0))
                continue
            
            # Sort by distance
            sorted_cands = sorted(candidates, key=lambda c: c["dist"])
            best = sorted_cands[0]
            best_dist = best["dist"]
            best_lid = best["lane_id"]
            
            if len(sorted_cands) == 1:
                confidences.append((best_lid, 2.0))  # Only one option = high confidence
                continue
            
            # Find second-best with DIFFERENT lane_id
            second_best_dist = None
            for c in sorted_cands[1:]:
                if c["lane_id"] != best_lid:
                    second_best_dist = c["dist"]
                    break
            
            if second_best_dist is None:
                confidences.append((best_lid, 2.0))  # All same lane_id
                continue
            
            # Confidence = distance gap ratio
            gap = second_best_dist - best_dist
            confidence = gap / self.CONFIDENCE_DISTANCE_GAP
            confidences.append((best_lid, confidence))
        
        return confidences
    
    def _detect_trajectory_phases(
        self,
        frame_candidates: List[List[Dict[str, object]]],
        confidences: List[Tuple[int, float]]
    ) -> List[Dict[str, object]]:
        """
        Detect trajectory phases based on road_id transitions and confident lane assignments.
        
        Returns list of phases, each with:
        - start_frame, end_frame
        - dominant_lane_id (most confident lane in this phase)
        - confidence_score
        """
        if not frame_candidates:
            return []
        
        # Find phase boundaries based on road_id transitions
        phases = []
        current_phase_start = 0
        current_road_ids = set()
        
        for i, candidates in enumerate(frame_candidates):
            if not candidates:
                continue
            
            # Get closest candidate's road_id
            best_cand = min(candidates, key=lambda c: c["dist"])
            road_id = best_cand["road_id"]
            
            # Detect phase boundary (new road that's not in recent set)
            if road_id not in current_road_ids and len(current_road_ids) > 0:
                # Check if this is a real transition (sustained)
                next_road_ids = set()
                for j in range(i, min(i + 5, len(frame_candidates))):
                    if frame_candidates[j]:
                        next_road_ids.add(min(frame_candidates[j], key=lambda c: c["dist"])["road_id"])
                
                if road_id in next_road_ids and not (next_road_ids & current_road_ids):
                    # Real transition - save current phase
                    if i > current_phase_start:
                        phases.append({
                            "start": current_phase_start,
                            "end": i - 1,
                            "road_ids": current_road_ids.copy()
                        })
                    current_phase_start = i
                    current_road_ids = {road_id}
                    continue
            
            current_road_ids.add(road_id)
        
        # Add final phase
        if current_phase_start < len(frame_candidates):
            phases.append({
                "start": current_phase_start,
                "end": len(frame_candidates) - 1,
                "road_ids": current_road_ids
            })
        
        # For each phase, find dominant lane_id from confident frames
        for phase in phases:
            lane_votes = {}
            for i in range(phase["start"], phase["end"] + 1):
                if i < len(confidences):
                    lid, conf = confidences[i]
                    if lid is not None and conf > 1.0:  # High confidence only
                        lane_votes[lid] = lane_votes.get(lid, 0) + conf
            
            if lane_votes:
                phase["dominant_lane_id"] = max(lane_votes, key=lambda k: lane_votes[k])
                phase["confidence_score"] = max(lane_votes.values())
            else:
                phase["dominant_lane_id"] = None
                phase["confidence_score"] = 0.0
        
        return phases
    
    def _smooth_spurious_dips(
        self,
        results: List[Optional[Dict[str, object]]],
        frame_candidates: List[List[Dict[str, object]]]
    ) -> List[Optional[Dict[str, object]]]:
        """
        Post-process results to smooth out spurious lane dips.
        
        A "spurious dip" is when the assigned lane_id briefly changes for < SPURIOUS_DIP_FRAMES
        then returns to the original lane_id.
        """
        if len(results) < 3:
            return results
        
        smoothed = list(results)
        n = len(results)
        
        # Find dips: sequences of different lane_id surrounded by same lane_id
        i = 0
        while i < n:
            if results[i] is None:
                i += 1
                continue
            
            base_lid = results[i]["lane_id"]
            
            # Look for dip start
            j = i + 1
            while j < n and results[j] is not None and results[j]["lane_id"] == base_lid:
                j += 1
            
            if j >= n:
                break
            
            # Found potential dip start at j
            dip_start = j
            dip_lid = results[j]["lane_id"] if results[j] else None
            
            # Find dip end
            while j < n and (results[j] is None or results[j]["lane_id"] == dip_lid):
                j += 1
            
            dip_end = j - 1
            dip_length = dip_end - dip_start + 1
            
            # Check if it returns to base_lid
            if j < n and results[j] is not None and results[j]["lane_id"] == base_lid:
                if dip_length <= self.SPURIOUS_DIP_FRAMES:
                    # This is a spurious dip - smooth it out
                    if self.verbose:
                        print(f"    [SMOOTH] Removing spurious dip: frames {dip_start}-{dip_end}, "
                              f"lane_id {dip_lid}->{base_lid}")
                    
                    for k in range(dip_start, dip_end + 1):
                        if smoothed[k] is not None:
                            # Find best candidate with base_lid
                            if frame_candidates[k]:
                                matching = [c for c in frame_candidates[k] if c["lane_id"] == base_lid]
                                if matching:
                                    best = min(matching, key=lambda c: c["dist"])
                                    smoothed[k] = dict(best)
                                    smoothed[k]["sdist"] = best["dist"]
                                    smoothed[k]["assigned_lane_id"] = base_lid
            
            i = j if j > i else i + 1
        
        return smoothed

    def _smooth_short_lane_flicker_runs(
        self,
        results: List[Optional[Dict[str, object]]],
        frame_candidates: List[List[Dict[str, object]]],
    ) -> List[Optional[Dict[str, object]]]:
        """
        Remove short lane_id flickers of the form A->B->A using run-level detection.

        Compared to frame-wise dip detection, this catches nested dips that occur
        inside a larger lane transition block.
        """
        if len(results) < 3:
            return results

        smoothed = list(results)
        n = len(smoothed)

        while True:
            changed = False
            lane_runs: List[Tuple[int, int, int]] = []  # (lane_id, start, end)
            i = 0
            while i < n:
                if smoothed[i] is None:
                    i += 1
                    continue
                lane_id = int(smoothed[i]["lane_id"])
                start = i
                while (
                    i + 1 < n
                    and smoothed[i + 1] is not None
                    and int(smoothed[i + 1]["lane_id"]) == lane_id
                ):
                    i += 1
                lane_runs.append((lane_id, start, i))
                i += 1

            for run_idx in range(1, len(lane_runs) - 1):
                lane_a, _, _ = lane_runs[run_idx - 1]
                lane_b, b_start, b_end = lane_runs[run_idx]
                lane_c, _, _ = lane_runs[run_idx + 1]
                b_len = b_end - b_start + 1

                if lane_a != lane_c:
                    continue
                if lane_b == lane_a:
                    continue

                prev_frame = smoothed[b_start - 1] if b_start - 1 >= 0 else None
                next_frame = smoothed[b_end + 1] if b_end + 1 < n else None
                prev_road = int(prev_frame["road_id"]) if prev_frame is not None else None
                next_road = int(next_frame["road_id"]) if next_frame is not None else None
                is_transition_bridge = (
                    prev_road is not None
                    and next_road is not None
                    and int(prev_road) != int(next_road)
                    and b_len <= int(self.LAST_SECOND_LANE_FLICKER_FRAMES)
                )
                if b_len > self.LANE_FLICKER_FRAMES and not is_transition_bridge:
                    continue

                original_cost = 0.0
                replacement_cost = 0.0
                replacement: List[Tuple[int, Dict[str, object]]] = []
                feasible = True

                for t in range(b_start, b_end + 1):
                    curr = smoothed[t]
                    if curr is None:
                        feasible = False
                        break
                    original_cost += float(curr.get("dist", curr.get("sdist", 0.0)))
                    matching = [c for c in frame_candidates[t] if int(c["lane_id"]) == int(lane_a)]
                    if not matching:
                        feasible = False
                        break
                    best = min(matching, key=lambda c: float(c["dist"]))
                    replacement_cost += float(best["dist"])
                    replacement.append((t, best))

                if not feasible:
                    continue

                local_slack = float(self.LANE_FLICKER_COST_SLACK) + float(self.LANE_FLICKER_PER_FRAME_SLACK) * float(b_len)

                # "Last-second" lane flicker bridge: A->B->A around a road transition.
                if is_transition_bridge:
                    bridge_slack = float(self.LAST_SECOND_LANE_FLICKER_COST_SLACK) + float(self.LAST_SECOND_LANE_FLICKER_PER_FRAME_SLACK) * float(b_len)
                    local_slack = max(local_slack, bridge_slack)

                if replacement_cost > original_cost + local_slack:
                    continue

                for t, best in replacement:
                    repl = dict(best)
                    repl["sdist"] = float(best["dist"])
                    repl["assigned_lane_id"] = int(lane_a)
                    smoothed[t] = repl

                if self.verbose:
                    mode = "bridge" if is_transition_bridge else "regular"
                    print(
                        f"    [SMOOTH] Lane flicker: lane {lane_a}->{lane_b}->{lane_c}, "
                        f"frames {b_start}-{b_end}, cost {original_cost:.2f}->{replacement_cost:.2f}, mode={mode}"
                    )

                changed = True
                break

            if not changed:
                break

        return smoothed

    def _delay_premature_lane_changes(
        self,
        results: List[Optional[Dict[str, object]]],
        frame_candidates: List[List[Dict[str, object]]],
    ) -> List[Optional[Dict[str, object]]]:
        """
        Delay lane_id switches that happen before geometric evidence supports them.

        This addresses early jump-to-other-lane artifacts where the selected lane
        at change onset is significantly farther than staying on the previous lane.
        """
        if len(results) < 3:
            return results
        out = list(results)
        n = len(out)
        max_lookahead = max(2, int(self.PREMATURE_LANE_CHANGE_LOOKAHEAD_FRAMES))
        confirm_frames = max(1, int(self.PREMATURE_LANE_CHANGE_CONFIRM_FRAMES))
        gain_m = float(self.PREMATURE_LANE_CHANGE_GAIN_M)
        trigger_margin = float(self.PREMATURE_LANE_CHANGE_TRIGGER_MARGIN_M)
        min_lateral_distance = _env_int("V2X_ALIGN_PREMATURE_MIN_LATERAL_DISTANCE", 1, minimum=1, maximum=4)
        hold_if_unconfirmed = (
            _env_int("V2X_ALIGN_PREMATURE_HOLD_IF_UNCONFIRMED", 1, minimum=0, maximum=1) == 1
        )
        hold_max_from_over_to = _env_float("V2X_ALIGN_PREMATURE_HOLD_MAX_FROM_OVER_TO_M", 1.0)

        def _best_for_lane(ti: int, lane_id: int) -> Optional[Dict[str, object]]:
            if ti < 0 or ti >= len(frame_candidates):
                return None
            matching = [c for c in frame_candidates[ti] if int(c.get("lane_id", 0)) == int(lane_id)]
            if not matching:
                return None
            return min(matching, key=lambda c: float(c.get("dist", float("inf"))))

        t = 1
        while t < n:
            prev = out[t - 1]
            cur = out[t]
            if prev is None or cur is None:
                t += 1
                continue
            from_lane_id = int(prev.get("assigned_lane_id", prev.get("lane_id", 0)))
            to_lane_id = int(cur.get("assigned_lane_id", cur.get("lane_id", 0)))
            if from_lane_id == 0 or to_lane_id == 0 or from_lane_id == to_lane_id:
                t += 1
                continue

            lateral_dist = self._lane_lateral_distance(int(from_lane_id), int(to_lane_id))
            sign_flip = int(from_lane_id) * int(to_lane_id) < 0
            # Focus on suspicious lane jumps; include sign flips even when
            # numerically adjacent because those are often semantic artifacts.
            if int(lateral_dist) < int(min_lateral_distance) and not bool(sign_flip):
                t += 1
                continue

            best_from_now = _best_for_lane(int(t), int(from_lane_id))
            if best_from_now is None:
                t += 1
                continue
            cur_dist = float(cur.get("dist", cur.get("sdist", float("inf"))))
            from_now_dist = float(best_from_now.get("dist", float("inf")))
            if not math.isfinite(cur_dist) or not math.isfinite(from_now_dist):
                t += 1
                continue
            # Only delay when the current switched lane is clearly worse right now.
            if cur_dist <= from_now_dist + float(trigger_margin):
                t += 1
                continue

            end = min(n - 1, t + max_lookahead)
            switch_at: Optional[int] = None
            for j in range(t, end + 1):
                ok = True
                for jj in range(j, min(end + 1, j + confirm_frames)):
                    cand_from = _best_for_lane(int(jj), int(from_lane_id))
                    cand_to = _best_for_lane(int(jj), int(to_lane_id))
                    if cand_from is None or cand_to is None:
                        ok = False
                        break
                    if float(cand_to.get("dist", float("inf"))) > float(cand_from.get("dist", float("inf"))) - float(gain_m):
                        ok = False
                        break
                if ok:
                    switch_at = int(j)
                    break

            # If no sustained evidence for switching appears, optionally hold the
            # current lane (from_lane) for as long as it remains reasonably close.
            if switch_at is None and bool(hold_if_unconfirmed):
                switch_at = int(end) + 1

            if switch_at is None or switch_at <= t:
                t += 1
                continue

            original_cost = 0.0
            replacement_cost = 0.0
            replacement_rows: List[Tuple[int, Dict[str, object]]] = []
            feasible = True
            for k in range(int(t), int(switch_at)):
                curr_k = out[k]
                if curr_k is None:
                    feasible = False
                    break
                cand_from = _best_for_lane(int(k), int(from_lane_id))
                if cand_from is None:
                    feasible = False
                    break
                if bool(hold_if_unconfirmed):
                    cand_to = _best_for_lane(int(k), int(to_lane_id))
                    if cand_to is not None:
                        from_dist = float(cand_from.get("dist", float("inf")))
                        to_dist = float(cand_to.get("dist", float("inf")))
                        if math.isfinite(from_dist) and math.isfinite(to_dist):
                            # Stop extending hold once "from" lane is clearly worse.
                            if from_dist > to_dist + float(hold_max_from_over_to):
                                break
                original_cost += float(curr_k.get("dist", curr_k.get("sdist", 0.0)))
                replacement_cost += float(cand_from.get("dist", 0.0))
                replacement_rows.append((int(k), cand_from))

            if not feasible or not replacement_rows:
                t += 1
                continue

            run_len = len(replacement_rows)
            local_slack = float(self.PREMATURE_LANE_CHANGE_COST_SLACK) + float(self.PREMATURE_LANE_CHANGE_PER_FRAME_SLACK) * float(run_len)
            if replacement_cost > original_cost + float(local_slack):
                t += 1
                continue

            for k, cand_from in replacement_rows:
                repl = dict(cand_from)
                repl["sdist"] = float(cand_from.get("dist", 0.0))
                repl["assigned_lane_id"] = int(from_lane_id)
                out[k] = repl

            if self.verbose:
                print(
                    f"    [SMOOTH] Delayed premature lane change: lane {from_lane_id}->{to_lane_id}, "
                    f"frames {t}-{switch_at - 1}, cost {original_cost:.2f}->{replacement_cost:.2f}"
                )
            t = max(t + 1, switch_at)

        return out

    def _suppress_weak_jump_lane_changes(
        self,
        waypoints: List[Waypoint],
        results: List[Optional[Dict[str, object]]],
        frame_candidates: List[List[Dict[str, object]]],
    ) -> List[Optional[Dict[str, object]]]:
        """
        Suppress weakly-supported lane switches that introduce abrupt XY jumps.

        This is intentionally conservative: only apply when lane change evidence
        is weak and snapped motion is much larger than observed raw motion.
        """
        if len(results) < 3 or len(waypoints) != len(results):
            return results

        out = list(results)
        n = len(out)
        max_hold_frames = _env_int("V2X_ALIGN_WEAK_JUMP_HOLD_MAX_FRAMES", 12, minimum=1, maximum=60)
        min_gain_m = _env_float("V2X_ALIGN_WEAK_JUMP_MIN_GAIN_M", 0.9)
        jump_abs_m = _env_float("V2X_ALIGN_WEAK_JUMP_ABS_M", 1.8)
        jump_ratio = _env_float("V2X_ALIGN_WEAK_JUMP_RATIO", 2.4)
        raw_floor_m = _env_float("V2X_ALIGN_WEAK_JUMP_RAW_FLOOR_M", 0.25)
        max_yaw_change_deg = _env_float("V2X_ALIGN_WEAK_JUMP_MAX_YAW_CHANGE_DEG", 24.0)
        max_per_frame_cost_delta = _env_float("V2X_ALIGN_WEAK_JUMP_MAX_PER_FRAME_COST_DELTA", 0.45)
        cost_slack_base = _env_float("V2X_ALIGN_WEAK_JUMP_COST_SLACK_BASE", 1.2)
        cost_slack_per_frame = _env_float("V2X_ALIGN_WEAK_JUMP_COST_SLACK_PER_FRAME", 0.25)

        def _best_for_lane(ti: int, lane_id: int) -> Optional[Dict[str, object]]:
            if ti < 0 or ti >= len(frame_candidates):
                return None
            matching = [c for c in frame_candidates[ti] if int(c.get("lane_id", 0)) == int(lane_id)]
            if not matching:
                return None
            return min(matching, key=lambda c: float(c.get("dist", float("inf"))))

        i = 1
        while i < n:
            prev = out[i - 1]
            cur = out[i]
            if prev is None or cur is None:
                i += 1
                continue

            from_lane_id = int(prev.get("assigned_lane_id", prev.get("lane_id", 0)))
            to_lane_id = int(cur.get("assigned_lane_id", cur.get("lane_id", 0)))
            if from_lane_id == 0 or to_lane_id == 0 or from_lane_id == to_lane_id:
                i += 1
                continue

            raw_dx = float(waypoints[i].x) - float(waypoints[i - 1].x)
            raw_dy = float(waypoints[i].y) - float(waypoints[i - 1].y)
            raw_step = float(math.hypot(float(raw_dx), float(raw_dy)))
            snap_step = float(
                math.hypot(
                    float(cur.get("x", 0.0)) - float(prev.get("x", 0.0)),
                    float(cur.get("y", 0.0)) - float(prev.get("y", 0.0)),
                )
            )
            jump_thr = max(
                float(jump_abs_m),
                float(jump_ratio) * max(float(raw_floor_m), float(raw_step)),
            )
            if snap_step <= float(jump_thr):
                i += 1
                continue

            yaw_change = self._normalize_yaw_diff(float(waypoints[i - 1].yaw), float(waypoints[i].yaw))
            if float(yaw_change) > float(max_yaw_change_deg):
                i += 1
                continue

            best_from_now = _best_for_lane(int(i), int(from_lane_id))
            if best_from_now is None:
                i += 1
                continue
            curr_dist = float(cur.get("dist", cur.get("sdist", float("inf"))))
            from_now_dist = float(best_from_now.get("dist", float("inf")))
            if not math.isfinite(curr_dist) or not math.isfinite(from_now_dist):
                i += 1
                continue
            gain_m = float(from_now_dist) - float(curr_dist)
            if gain_m >= float(min_gain_m):
                i += 1
                continue

            hold_rows: List[Tuple[int, Dict[str, object]]] = []
            original_cost = 0.0
            replacement_cost = 0.0
            j = i
            while j < n and (j - i) < int(max_hold_frames):
                curr_j = out[j]
                if curr_j is None:
                    break
                lane_j = int(curr_j.get("assigned_lane_id", curr_j.get("lane_id", 0)))
                if lane_j != int(to_lane_id):
                    break
                cand_from = _best_for_lane(int(j), int(from_lane_id))
                if cand_from is None:
                    break
                old_d = float(curr_j.get("dist", curr_j.get("sdist", float("inf"))))
                new_d = float(cand_from.get("dist", float("inf")))
                if (not math.isfinite(old_d)) or (not math.isfinite(new_d)):
                    break
                if new_d > old_d + float(max_per_frame_cost_delta):
                    break
                original_cost += float(old_d)
                replacement_cost += float(new_d)
                hold_rows.append((int(j), cand_from))
                j += 1

            if not hold_rows:
                i += 1
                continue

            local_slack = float(cost_slack_base) + float(cost_slack_per_frame) * float(len(hold_rows))
            if replacement_cost > original_cost + float(local_slack):
                i += 1
                continue

            for j_idx, cand_from in hold_rows:
                repl = dict(cand_from)
                repl["sdist"] = float(cand_from.get("dist", 0.0))
                repl["assigned_lane_id"] = int(from_lane_id)
                out[j_idx] = repl

            if self.verbose:
                print(
                    f"    [SMOOTH] Suppressed weak jump lane change: lane {from_lane_id}->{to_lane_id}, "
                    f"frames {i}-{hold_rows[-1][0]}, cost {original_cost:.2f}->{replacement_cost:.2f}"
                )
            i = max(i + 1, hold_rows[-1][0] + 1)

        return out

    def _retime_large_road_transition_jumps(
        self,
        waypoints: List[Waypoint],
        results: List[Optional[Dict[str, object]]],
        frame_candidates: List[List[Dict[str, object]]],
    ) -> List[Optional[Dict[str, object]]]:
        """
        Reduce large one-step XY snaps at road/lane transitions by moving the
        transition boundary slightly earlier when the target road/lane was already
        geometrically plausible in preceding frames.
        """
        if len(results) < 3 or len(waypoints) != len(results):
            return results

        out = list(results)
        n = len(out)
        max_back_frames = _env_int("V2X_ALIGN_RETIME_TRANSITION_MAX_BACK_FRAMES", 12, minimum=1, maximum=60)
        jump_abs_m = _env_float("V2X_ALIGN_RETIME_TRANSITION_JUMP_ABS_M", 2.0)
        jump_ratio = _env_float("V2X_ALIGN_RETIME_TRANSITION_JUMP_RATIO", 2.6)
        raw_floor_m = _env_float("V2X_ALIGN_RETIME_TRANSITION_RAW_FLOOR_M", 0.25)
        max_yaw_change_deg = _env_float("V2X_ALIGN_RETIME_TRANSITION_MAX_YAW_CHANGE_DEG", 20.0)
        max_per_frame_cost_delta = _env_float("V2X_ALIGN_RETIME_TRANSITION_MAX_PER_FRAME_COST_DELTA", 1.2)
        cost_slack_base = _env_float("V2X_ALIGN_RETIME_TRANSITION_COST_SLACK_BASE", 1.1)
        cost_slack_per_frame = _env_float("V2X_ALIGN_RETIME_TRANSITION_COST_SLACK_PER_FRAME", 0.30)
        min_improvement_m = _env_float("V2X_ALIGN_RETIME_TRANSITION_MIN_IMPROVEMENT_M", 0.45)
        teleport_raw_step_m = _env_float("V2X_ALIGN_RETIME_TELEPORT_RAW_STEP_M", 0.6)
        teleport_snap_step_m = _env_float("V2X_ALIGN_RETIME_TELEPORT_SNAP_STEP_M", 1.8)

        def _best_on_road_lane(ti: int, road_id: int, lane_id: int) -> Optional[Dict[str, object]]:
            if ti < 0 or ti >= len(frame_candidates):
                return None
            return self._best_candidate_on_road_lane(frame_candidates[ti], int(road_id), int(lane_id))

        i = 1
        while i < n:
            prev = out[i - 1]
            cur = out[i]
            if prev is None or cur is None:
                i += 1
                continue
            if bool(prev.get("synthetic_turn", False)) or bool(cur.get("synthetic_turn", False)):
                i += 1
                continue

            prev_road = int(prev.get("road_id", -1))
            prev_lane = int(prev.get("assigned_lane_id", prev.get("lane_id", 0)))
            cur_road = int(cur.get("road_id", -1))
            cur_lane = int(cur.get("assigned_lane_id", cur.get("lane_id", 0)))
            if prev_road < 0 or cur_road < 0 or prev_lane == 0 or cur_lane == 0:
                i += 1
                continue
            if prev_road == cur_road and prev_lane == cur_lane:
                i += 1
                continue

            raw_step = float(
                math.hypot(
                    float(waypoints[i].x) - float(waypoints[i - 1].x),
                    float(waypoints[i].y) - float(waypoints[i - 1].y),
                )
            )
            snap_step = float(
                math.hypot(
                    float(cur.get("x", 0.0)) - float(prev.get("x", 0.0)),
                    float(cur.get("y", 0.0)) - float(prev.get("y", 0.0)),
                )
            )
            jump_thr = max(float(jump_abs_m), float(jump_ratio) * max(float(raw_floor_m), float(raw_step)))
            if snap_step <= float(jump_thr):
                i += 1
                continue

            yaw_change = self._normalize_yaw_diff(float(waypoints[i - 1].yaw), float(waypoints[i].yaw))
            if float(yaw_change) > float(max_yaw_change_deg):
                i += 1
                continue

            # Candidate retiming plans: move the switch boundary backward.
            best_plan: Optional[Tuple[Tuple[float, float, int], List[Tuple[int, Dict[str, object]]]]] = None
            min_k = max(1, int(i) - int(max_back_frames))
            for k in range(int(i) - 1, int(min_k) - 1, -1):
                before = out[k - 1]
                if before is None or bool(before.get("synthetic_turn", False)):
                    break

                old_cost = 0.0
                new_cost = 0.0
                replacement_rows: List[Tuple[int, Dict[str, object]]] = []
                feasible = True
                for t in range(int(k), int(i)):
                    curr_t = out[t]
                    if curr_t is None or bool(curr_t.get("synthetic_turn", False)):
                        feasible = False
                        break
                    cand = _best_on_road_lane(int(t), int(cur_road), int(cur_lane))
                    if cand is None:
                        feasible = False
                        break
                    old_d = float(curr_t.get("dist", curr_t.get("sdist", float("inf"))))
                    new_d = float(cand.get("dist", float("inf")))
                    if (not math.isfinite(old_d)) or (not math.isfinite(new_d)):
                        feasible = False
                        break
                    if new_d > old_d + float(max_per_frame_cost_delta):
                        feasible = False
                        break
                    old_cost += float(old_d)
                    new_cost += float(new_d)
                    replacement_rows.append((int(t), cand))

                if not feasible or not replacement_rows:
                    continue

                local_slack = float(cost_slack_base) + float(cost_slack_per_frame) * float(len(replacement_rows))
                if new_cost > old_cost + float(local_slack):
                    continue

                # Evaluate resulting max step over [k-1, i] boundary window.
                replacement_map = {int(t): cand for t, cand in replacement_rows}

                def _xy_at(ti: int) -> Optional[Tuple[float, float]]:
                    if ti < 0 or ti >= n:
                        return None
                    if ti in replacement_map:
                        cand = replacement_map[ti]
                        return (float(cand["x"]), float(cand["y"]))
                    row = out[ti]
                    if row is None:
                        return None
                    return (float(row["x"]), float(row["y"]))

                def _orig_xy_at(ti: int) -> Optional[Tuple[float, float]]:
                    if ti < 0 or ti >= n:
                        return None
                    row = out[ti]
                    if row is None:
                        return None
                    return (float(row["x"]), float(row["y"]))

                max_step = 0.0
                old_max_step = 0.0
                ok_window = True
                creates_teleport = False
                for t in range(int(k) - 1, int(i)):
                    old_p0 = _orig_xy_at(int(t))
                    old_p1 = _orig_xy_at(int(t + 1))
                    if old_p0 is None or old_p1 is None:
                        ok_window = False
                        break
                    old_step = float(
                        math.hypot(
                            float(old_p1[0]) - float(old_p0[0]),
                            float(old_p1[1]) - float(old_p0[1]),
                        )
                    )
                    old_max_step = max(float(old_max_step), float(old_step))
                    p0 = _xy_at(int(t))
                    p1 = _xy_at(int(t + 1))
                    if p0 is None or p1 is None:
                        ok_window = False
                        break
                    step = float(math.hypot(float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1])))
                    max_step = max(float(max_step), float(step))
                    raw_step_local = float(
                        math.hypot(
                            float(waypoints[t + 1].x) - float(waypoints[t].x),
                            float(waypoints[t + 1].y) - float(waypoints[t].y),
                        )
                    )
                    if float(raw_step_local) < float(teleport_raw_step_m) and float(step) > float(teleport_snap_step_m):
                        creates_teleport = True
                        break
                if not ok_window:
                    continue
                if bool(creates_teleport):
                    continue
                if float(max_step) >= float(old_max_step) - 1e-6:
                    continue

                improvement = float(snap_step) - float(max_step)
                if improvement < float(min_improvement_m):
                    continue

                score = (float(max_step), float(new_cost - old_cost), int(k))
                if best_plan is None or score < best_plan[0]:
                    best_plan = (score, replacement_rows)

            if best_plan is None:
                i += 1
                continue

            _, rows = best_plan
            if rows:
                start_t = int(rows[0][0])
                end_t = int(rows[-1][0])
                for t, cand in rows:
                    repl = dict(cand)
                    repl["sdist"] = float(cand.get("dist", 0.0))
                    repl["assigned_lane_id"] = int(cur_lane)
                    out[int(t)] = repl
                if self.verbose:
                    print(
                        f"    [SMOOTH] Retimed large transition jump: "
                        f"{prev_road}_{prev_lane}->{cur_road}_{cur_lane}, frames {start_t}-{end_t}"
                    )
                i = max(i + 1, end_t + 1)
                continue

            i += 1

        return out

    def _soften_alignment_transition_jumps(
        self,
        waypoints: List[Waypoint],
        results: List[Optional[Dict[str, object]]],
        jump_threshold_m: float = 1.6,
        jump_ratio_vs_raw: float = 2.2,
        max_shift_m: float = 1.2,
        max_query_cost_delta: float = 1.0,
    ) -> List[Optional[Dict[str, object]]]:
        """
        Soften residual one-step XY jumps at road/lane boundaries by nudging both
        boundary points toward a bounded midpoint.
        """
        if len(results) < 2 or len(waypoints) != len(results):
            return results

        smoothed = list(results)
        n = len(smoothed)
        synth_max_shift_m = _env_float("V2X_ALIGN_SOFTEN_TRANSITION_SYN_MAX_SHIFT_M", 4.0)
        synth_max_query_cost_delta = _env_float("V2X_ALIGN_SOFTEN_TRANSITION_SYN_MAX_QUERY_DELTA", 3.0)
        teleport_raw_step_m = _env_float("V2X_ALIGN_SOFTEN_TRANSITION_TELEPORT_RAW_STEP_M", 0.6)
        teleport_snap_step_m = _env_float("V2X_ALIGN_SOFTEN_TRANSITION_TELEPORT_SNAP_STEP_M", 1.8)

        def _snap_step_with_overrides(
            step_idx: int,
            overrides: Dict[int, Tuple[float, float]],
        ) -> Optional[float]:
            if step_idx <= 0 or step_idx >= n:
                return None
            a = smoothed[step_idx - 1]
            b = smoothed[step_idx]
            if a is None or b is None:
                return None
            ax, ay = (
                overrides.get(step_idx - 1, (float(a.get("x", 0.0)), float(a.get("y", 0.0))))
            )
            bx, by = overrides.get(step_idx, (float(b.get("x", 0.0)), float(b.get("y", 0.0))))
            return float(math.hypot(float(bx) - float(ax), float(by) - float(ay)))

        for i in range(1, n):
            a = smoothed[i - 1]
            b = smoothed[i]
            if a is None or b is None:
                continue

            a_key = (int(a.get("road_id", -1)), int(a.get("assigned_lane_id", a.get("lane_id", 0))))
            b_key = (int(b.get("road_id", -1)), int(b.get("assigned_lane_id", b.get("lane_id", 0))))
            if a_key == b_key:
                continue

            raw_step = float(
                math.hypot(
                    float(waypoints[i].x) - float(waypoints[i - 1].x),
                    float(waypoints[i].y) - float(waypoints[i - 1].y),
                )
            )
            ax = float(a.get("x", 0.0))
            ay = float(a.get("y", 0.0))
            bx = float(b.get("x", 0.0))
            by = float(b.get("y", 0.0))
            snap_step = float(math.hypot(float(bx) - float(ax), float(by) - float(ay)))
            if snap_step <= max(float(jump_threshold_m), float(jump_ratio_vs_raw) * max(0.2, float(raw_step))):
                continue

            is_synthetic_boundary = bool(a.get("synthetic_turn", False)) or bool(b.get("synthetic_turn", False))
            local_max_shift = float(synth_max_shift_m if is_synthetic_boundary else max_shift_m)
            local_max_query_delta = float(
                synth_max_query_cost_delta if is_synthetic_boundary else max_query_cost_delta
            )

            mx = 0.5 * (float(ax) + float(bx))
            my = 0.5 * (float(ay) + float(by))
            target = max(0.9, 1.8 * max(0.2, float(raw_step)))
            if snap_step <= float(target):
                continue

            scale = float(target) / max(1e-6, float(snap_step))
            nax = float(mx) + (float(ax) - float(mx)) * float(scale)
            nay = float(my) + (float(ay) - float(my)) * float(scale)
            nbx = float(mx) + (float(bx) - float(mx)) * float(scale)
            nby = float(my) + (float(by) - float(my)) * float(scale)

            shift_a = float(math.hypot(float(nax) - float(ax), float(nay) - float(ay)))
            shift_b = float(math.hypot(float(nbx) - float(bx), float(nby) - float(by)))
            if shift_a > float(local_max_shift):
                sa = float(local_max_shift) / max(1e-6, float(shift_a))
                nax = float(ax) + (float(nax) - float(ax)) * float(sa)
                nay = float(ay) + (float(nay) - float(ay)) * float(sa)
            if shift_b > float(local_max_shift):
                sb = float(local_max_shift) / max(1e-6, float(shift_b))
                nbx = float(bx) + (float(nbx) - float(bx)) * float(sb)
                nby = float(by) + (float(nby) - float(by)) * float(sb)

            old_da = float(math.hypot(float(ax) - float(waypoints[i - 1].x), float(ay) - float(waypoints[i - 1].y)))
            old_db = float(math.hypot(float(bx) - float(waypoints[i].x), float(by) - float(waypoints[i].y)))
            new_da = float(math.hypot(float(nax) - float(waypoints[i - 1].x), float(nay) - float(waypoints[i - 1].y)))
            new_db = float(math.hypot(float(nbx) - float(waypoints[i].x), float(nby) - float(waypoints[i].y)))
            if new_da > old_da + float(local_max_query_delta):
                continue
            if new_db > old_db + float(local_max_query_delta):
                continue

            overrides = {
                int(i - 1): (float(nax), float(nay)),
                int(i): (float(nbx), float(nby)),
            }
            affected_steps = [idx for idx in (i - 1, i, i + 1) if 0 < idx < n]
            old_local_max = 0.0
            new_local_max = 0.0
            has_step = False
            creates_teleport = False
            for step_idx in affected_steps:
                old_step = _snap_step_with_overrides(int(step_idx), overrides={})
                new_step = _snap_step_with_overrides(int(step_idx), overrides=overrides)
                if old_step is None or new_step is None:
                    continue
                has_step = True
                old_local_max = max(float(old_local_max), float(old_step))
                new_local_max = max(float(new_local_max), float(new_step))
                raw_step_local = float(
                    math.hypot(
                        float(waypoints[step_idx].x) - float(waypoints[step_idx - 1].x),
                        float(waypoints[step_idx].y) - float(waypoints[step_idx - 1].y),
                    )
                )
                if float(raw_step_local) < float(teleport_raw_step_m) and float(new_step) > float(teleport_snap_step_m):
                    creates_teleport = True
                    break
            if not bool(has_step):
                continue
            if bool(creates_teleport):
                continue
            if float(new_local_max) >= float(old_local_max) - 1e-6:
                continue

            a["x"] = float(nax)
            a["y"] = float(nay)
            b["x"] = float(nbx)
            b["y"] = float(nby)
            a["dist"] = float(new_da)
            a["sdist"] = float(new_da)
            b["dist"] = float(new_db)
            b["sdist"] = float(new_db)
            a["transition_jump_soften"] = True
            b["transition_jump_soften"] = True

            # Recompute local yaws from softened segment direction.
            seg_yaw = float(math.degrees(math.atan2(float(nby) - float(nay), float(nbx) - float(nax))))
            a["yaw"] = float(seg_yaw)
            b["yaw"] = float(seg_yaw)

        return smoothed

    def _smooth_edge_lane_spikes(
        self,
        results: List[Optional[Dict[str, object]]],
        frame_candidates: List[List[Dict[str, object]]],
    ) -> List[Optional[Dict[str, object]]]:
        """
        Smooth short lane runs at the beginning/end of a segment.

        This targets edge artifacts where a segment starts (or ends) with a
        brief lane assignment that is immediately replaced by a long stable lane.
        """
        if len(results) < 2:
            return results

        smoothed = list(results)
        n = len(smoothed)

        def build_lane_runs(seq: List[Optional[Dict[str, object]]]) -> List[Tuple[int, int, int]]:
            runs: List[Tuple[int, int, int]] = []  # (lane_id, start, end)
            i = 0
            while i < len(seq):
                if seq[i] is None:
                    i += 1
                    continue
                lane_id = int(seq[i]["lane_id"])
                start = i
                while (
                    i + 1 < len(seq)
                    and seq[i + 1] is not None
                    and int(seq[i + 1]["lane_id"]) == lane_id
                ):
                    i += 1
                runs.append((lane_id, start, i))
                i += 1
            return runs

        while True:
            changed = False
            lane_runs = build_lane_runs(smoothed)
            if len(lane_runs) < 2:
                break

            # Head spike
            head_lane, head_start, head_end = lane_runs[0]
            next_lane, next_start, next_end = lane_runs[1]
            head_len = head_end - head_start + 1
            next_len = next_end - next_start + 1
            if (
                head_start == 0
                and head_lane != next_lane
                and head_len <= int(self.EDGE_LANE_SPIKE_FRAMES)
                and next_len >= int(self.EDGE_LANE_SPIKE_MIN_NEIGHBOR_FRAMES)
            ):
                original_cost = 0.0
                replacement_cost = 0.0
                replacement: List[Tuple[int, Dict[str, object]]] = []
                feasible = True
                for t in range(head_start, head_end + 1):
                    curr = smoothed[t]
                    if curr is None:
                        feasible = False
                        break
                    original_cost += float(curr.get("dist", curr.get("sdist", 0.0)))
                    matching = [c for c in frame_candidates[t] if int(c["lane_id"]) == int(next_lane)]
                    if not matching:
                        feasible = False
                        break
                    best = min(matching, key=lambda c: float(c["dist"]))
                    replacement_cost += float(best["dist"])
                    replacement.append((t, best))

                if feasible:
                    local_slack = float(self.EDGE_LANE_SPIKE_COST_SLACK) + float(self.EDGE_LANE_SPIKE_PER_FRAME_SLACK) * float(head_len)
                    if replacement_cost <= original_cost + local_slack:
                        for t, best in replacement:
                            repl = dict(best)
                            repl["sdist"] = float(best["dist"])
                            repl["assigned_lane_id"] = int(next_lane)
                            smoothed[t] = repl
                        if self.verbose:
                            print(
                                f"    [SMOOTH] Edge lane spike(head): lane {head_lane}->{next_lane}, "
                                f"frames {head_start}-{head_end}, cost {original_cost:.2f}->{replacement_cost:.2f}"
                            )
                        changed = True
                        continue

            # Tail spike
            prev_lane, prev_start, prev_end = lane_runs[-2]
            tail_lane, tail_start, tail_end = lane_runs[-1]
            tail_len = tail_end - tail_start + 1
            prev_len = prev_end - prev_start + 1
            if (
                tail_end == (n - 1)
                and tail_lane != prev_lane
                and tail_len <= int(self.EDGE_LANE_SPIKE_FRAMES)
                and prev_len >= int(self.EDGE_LANE_SPIKE_MIN_NEIGHBOR_FRAMES)
            ):
                original_cost = 0.0
                replacement_cost = 0.0
                replacement: List[Tuple[int, Dict[str, object]]] = []
                feasible = True
                for t in range(tail_start, tail_end + 1):
                    curr = smoothed[t]
                    if curr is None:
                        feasible = False
                        break
                    original_cost += float(curr.get("dist", curr.get("sdist", 0.0)))
                    matching = [c for c in frame_candidates[t] if int(c["lane_id"]) == int(prev_lane)]
                    if not matching:
                        feasible = False
                        break
                    best = min(matching, key=lambda c: float(c["dist"]))
                    replacement_cost += float(best["dist"])
                    replacement.append((t, best))

                if feasible:
                    local_slack = float(self.EDGE_LANE_SPIKE_COST_SLACK) + float(self.EDGE_LANE_SPIKE_PER_FRAME_SLACK) * float(tail_len)
                    if replacement_cost <= original_cost + local_slack:
                        for t, best in replacement:
                            repl = dict(best)
                            repl["sdist"] = float(best["dist"])
                            repl["assigned_lane_id"] = int(prev_lane)
                            smoothed[t] = repl
                        if self.verbose:
                            print(
                                f"    [SMOOTH] Edge lane spike(tail): lane {prev_lane}->{tail_lane}, "
                                f"frames {tail_start}-{tail_end}, cost {original_cost:.2f}->{replacement_cost:.2f}"
                            )
                        changed = True
                        continue

            if not changed:
                break

        return smoothed

    def _smooth_short_connector_roads(
        self,
        results: List[Optional[Dict[str, object]]],
        frame_candidates: List[List[Dict[str, object]]],
    ) -> List[Optional[Dict[str, object]]]:
        """
        Remove short A->B->C road detours when A already connects directly to C.

        This targets brief connector-road snippets that inflate path complexity
        without improving fit quality.
        """
        if len(results) < 3:
            return results

        smoothed = list(results)

        while True:
            changed = False
            runs: List[Tuple[int, int, int]] = []
            i = 0
            n = len(smoothed)

            while i < n:
                if smoothed[i] is None:
                    i += 1
                    continue
                road_id = int(smoothed[i]["road_id"])
                start = i
                while (
                    i + 1 < n
                    and smoothed[i + 1] is not None
                    and int(smoothed[i + 1]["road_id"]) == road_id
                ):
                    i += 1
                runs.append((road_id, start, i))
                i += 1

            for run_idx in range(1, len(runs) - 1):
                road_a, _, _ = runs[run_idx - 1]
                road_b, b_start, b_end = runs[run_idx]
                road_c, _, _ = runs[run_idx + 1]
                b_len = b_end - b_start + 1

                if b_len > self.SHORT_CONNECTOR_FRAMES:
                    continue
                if road_a == road_b or road_b == road_c or road_a == road_c:
                    continue
                if road_c not in self.road_successors.get(road_a, set()):
                    continue
                if road_b in self.road_successors.get(road_a, set()):
                    # Keep explicit A->B connectivity; only smooth disconnected detours.
                    continue

                best_cost = float("inf")
                best_assign: Optional[List[Tuple[int, Dict[str, object]]]] = None
                best_switch = None

                # One-switch model: road_a before switch, road_c after switch.
                for switch_idx in range(b_start, b_end + 2):
                    assign: List[Tuple[int, Dict[str, object]]] = []
                    total_cost = 0.0
                    feasible = True

                    for t in range(b_start, b_end + 1):
                        target_road = road_a if t < switch_idx else road_c
                        candidates = [c for c in frame_candidates[t] if int(c["road_id"]) == int(target_road)]
                        if not candidates:
                            feasible = False
                            break
                        best = min(candidates, key=lambda c: c["dist"])
                        total_cost += float(best["dist"])
                        assign.append((t, best))

                    if feasible and total_cost < best_cost:
                        best_cost = total_cost
                        best_assign = assign
                        best_switch = switch_idx

                if best_assign is None:
                    continue

                original_cost = 0.0
                original_feasible = True
                for t in range(b_start, b_end + 1):
                    curr = smoothed[t]
                    if curr is None:
                        original_feasible = False
                        break
                    original_cost += float(curr.get("dist", curr.get("sdist", 0.0)))
                if not original_feasible:
                    continue
                if best_cost > original_cost + self.CONNECTOR_COST_SLACK:
                    continue

                if self.verbose:
                    print(
                        f"    [SMOOTH] Connector road removal: road {road_a}->{road_b}->{road_c}, "
                        f"frames {b_start}-{b_end}, switch={best_switch}, "
                        f"cost {original_cost:.2f}->{best_cost:.2f}"
                    )

                for t, best in best_assign:
                    repl = dict(best)
                    repl["sdist"] = float(best["dist"])
                    repl["assigned_lane_id"] = int(best["lane_id"])
                    smoothed[t] = repl

                changed = True
                break

            if not changed:
                break

        return smoothed

    def _smooth_same_lane_road_flicker(
        self,
        results: List[Optional[Dict[str, object]]],
        frame_candidates: List[List[Dict[str, object]]],
    ) -> List[Optional[Dict[str, object]]]:
        """Remove short A->B->A road flickers when lane_id is unchanged."""
        if len(results) < 3:
            return results

        smoothed = list(results)
        n = len(smoothed)

        while True:
            changed = False
            runs: List[Tuple[int, int, int, int]] = []  # (road_id, lane_id, start, end)
            i = 0
            while i < n:
                if smoothed[i] is None:
                    i += 1
                    continue
                road_id = int(smoothed[i]["road_id"])
                lane_id = int(smoothed[i]["lane_id"])
                start = i
                while (
                    i + 1 < n
                    and smoothed[i + 1] is not None
                    and int(smoothed[i + 1]["road_id"]) == road_id
                    and int(smoothed[i + 1]["lane_id"]) == lane_id
                ):
                    i += 1
                runs.append((road_id, lane_id, start, i))
                i += 1

            for run_idx in range(1, len(runs) - 1):
                road_a, lane_a, _, _ = runs[run_idx - 1]
                road_b, lane_b, b_start, b_end = runs[run_idx]
                road_c, lane_c, _, _ = runs[run_idx + 1]
                b_len = b_end - b_start + 1

                if lane_a != lane_b or lane_b != lane_c:
                    continue
                if road_a != road_c or road_a == road_b:
                    continue
                is_long_flicker = b_len > self.ROAD_FLICKER_FRAMES
                if is_long_flicker and b_len > self.ROAD_FLICKER_LONG_MAX_FRAMES:
                    continue

                original_cost = 0.0
                replacement_cost = 0.0
                replacement: List[Tuple[int, Dict[str, object]]] = []
                feasible = True
                for t in range(b_start, b_end + 1):
                    curr = smoothed[t]
                    if curr is None:
                        feasible = False
                        break
                    original_cost += float(curr.get("dist", curr.get("sdist", 0.0)))
                    best = self._best_candidate_on_road_lane(frame_candidates[t], road_a, lane_a)
                    if best is None:
                        feasible = False
                        break
                    replacement_cost += float(best["dist"])
                    replacement.append((t, best))

                if not feasible:
                    continue
                if is_long_flicker:
                    # For long A->B->A runs, only compress when geometric evidence
                    # is strong: replacement must reduce total distance notably.
                    required_gain = float(self.ROAD_FLICKER_LONG_GAIN_PER_FRAME) * float(b_len)
                    if replacement_cost > (original_cost - required_gain):
                        continue
                else:
                    local_slack = float(self.ROAD_FLICKER_COST_SLACK) + float(self.ROAD_FLICKER_PER_FRAME_SLACK) * float(b_len)
                    if replacement_cost > original_cost + local_slack:
                        continue

                for t, best in replacement:
                    repl = dict(best)
                    repl["sdist"] = float(best["dist"])
                    repl["assigned_lane_id"] = int(lane_a)
                    smoothed[t] = repl

                if self.verbose:
                    mode = "long" if is_long_flicker else "short"
                    print(
                        f"    [SMOOTH] Same-lane road flicker: road {road_a}->{road_b}->{road_c}, "
                        f"frames {b_start}-{b_end}, cost {original_cost:.2f}->{replacement_cost:.2f}, mode={mode}"
                    )

                changed = True
                break

            if not changed:
                break

        return smoothed

    def _smooth_terminal_same_lane_road_tail(
        self,
        results: List[Optional[Dict[str, object]]],
        frame_candidates: List[List[Dict[str, object]]],
    ) -> List[Optional[Dict[str, object]]]:
        """
        Collapse short terminal road switches when lane_id is unchanged.

        This discourages end-of-segment last-second road hops (A->B at segment end)
        that do not materially improve geometric fit.
        """
        if len(results) < 2:
            return results

        smoothed = list(results)
        n = len(smoothed)

        while True:
            runs: List[Tuple[int, int, int, int]] = []  # (road_id, lane_id, start, end)
            i = 0
            while i < n:
                if smoothed[i] is None:
                    i += 1
                    continue
                road_id = int(smoothed[i]["road_id"])
                lane_id = int(smoothed[i]["lane_id"])
                start = i
                while (
                    i + 1 < n
                    and smoothed[i + 1] is not None
                    and int(smoothed[i + 1]["road_id"]) == road_id
                    and int(smoothed[i + 1]["lane_id"]) == lane_id
                ):
                    i += 1
                runs.append((road_id, lane_id, start, i))
                i += 1

            if len(runs) < 2:
                break

            prev_road, prev_lane, _, _ = runs[-2]
            tail_road, tail_lane, tail_start, tail_end = runs[-1]
            tail_len = tail_end - tail_start + 1

            if int(prev_lane) != int(tail_lane):
                break
            if int(prev_road) == int(tail_road):
                break
            if tail_len > int(self.TERMINAL_ROAD_TAIL_FRAMES):
                break

            original_cost = 0.0
            replacement_cost = 0.0
            replacement: List[Tuple[int, Dict[str, object]]] = []
            feasible = True
            for t in range(tail_start, tail_end + 1):
                curr = smoothed[t]
                if curr is None:
                    feasible = False
                    break
                original_cost += float(curr.get("dist", curr.get("sdist", 0.0)))
                best = self._best_candidate_on_road_lane(frame_candidates[t], prev_road, prev_lane)
                if best is None:
                    feasible = False
                    break
                replacement_cost += float(best["dist"])
                replacement.append((t, best))

            if not feasible:
                break

            if replacement_cost > original_cost + float(self.TERMINAL_ROAD_TAIL_COST_SLACK):
                break

            for t, best in replacement:
                repl = dict(best)
                repl["sdist"] = float(best["dist"])
                repl["assigned_lane_id"] = int(prev_lane)
                smoothed[t] = repl

            if self.verbose:
                print(
                    f"    [SMOOTH] Terminal road tail: road {prev_road}->{tail_road}, "
                    f"lane {prev_lane}, frames {tail_start}-{tail_end}, "
                    f"cost {original_cost:.2f}->{replacement_cost:.2f}"
                )

        return smoothed

    def _smooth_same_lane_turn_connector_chains(
        self,
        results: List[Optional[Dict[str, object]]],
        frame_candidates: List[List[Dict[str, object]]],
    ) -> List[Optional[Dict[str, object]]]:
        """
        Compress long same-lane connector road chains into a single start->end transition.

        This reduces turn-time snapping across many short connector roads when the
        intended lane is unchanged.
        """
        if len(results) < 4:
            return results
        if "intersection" not in self.matcher.map_data.name.lower():
            return results

        smoothed = list(results)
        n = len(smoothed)
        min_chain_runs = _env_int("V2X_ALIGN_TURN_CHAIN_MIN_RUNS", 3, minimum=3, maximum=8)

        while True:
            changed = False
            runs: List[Tuple[int, int, int, int]] = []  # (road_id, lane_id, start, end)
            i = 0
            while i < n:
                if smoothed[i] is None:
                    i += 1
                    continue
                road_id = int(smoothed[i]["road_id"])
                lane_id = int(smoothed[i]["lane_id"])
                start = i
                while (
                    i + 1 < n
                    and smoothed[i + 1] is not None
                    and int(smoothed[i + 1]["road_id"]) == road_id
                    and int(smoothed[i + 1]["lane_id"]) == lane_id
                ):
                    i += 1
                runs.append((road_id, lane_id, start, i))
                i += 1

            if len(runs) < 3:
                break

            for start_idx in range(0, len(runs) - 2):
                road_a, lane_a, _, _ = runs[start_idx]

                # Grow a block of same-lane runs.
                end_idx = start_idx + 1
                total_frames = 0
                while end_idx < len(runs):
                    road_k, lane_k, s_k, e_k = runs[end_idx]
                    if lane_k != lane_a:
                        break
                    total_frames += (e_k - s_k + 1)
                    if total_frames > self.TURN_CHAIN_MAX_FRAMES:
                        break
                    end_idx += 1

                # end_idx is exclusive.
                if end_idx - start_idx < int(min_chain_runs):
                    continue
                if total_frames > self.TURN_CHAIN_MAX_FRAMES:
                    continue

                # Use start anchor run and last run as end anchor.
                road_c, lane_c, _, _ = runs[end_idx - 1]
                if lane_c != lane_a:
                    continue
                if road_a == road_c:
                    continue

                # Middle connector runs must be short.
                middle_ok = True
                for k in range(start_idx + 1, end_idx - 1):
                    _, _, s_k, e_k = runs[k]
                    if (e_k - s_k + 1) > (self.ROAD_FLICKER_FRAMES * 2):
                        middle_ok = False
                        break
                if not middle_ok:
                    continue

                interval_start = runs[start_idx + 1][2]
                interval_end = runs[end_idx - 1][3]
                if interval_end < interval_start:
                    continue

                original_cost = 0.0
                original_feasible = True
                for t in range(interval_start, interval_end + 1):
                    curr = smoothed[t]
                    if curr is None:
                        original_feasible = False
                        break
                    original_cost += float(curr.get("dist", curr.get("sdist", 0.0)))
                if not original_feasible:
                    continue

                best_cost = float("inf")
                best_switch = None
                best_assign: Optional[List[Tuple[int, Dict[str, object]]]] = None
                for switch_idx in range(interval_start, interval_end + 2):
                    assign: List[Tuple[int, Dict[str, object]]] = []
                    total_cost = 0.0
                    feasible = True
                    for t in range(interval_start, interval_end + 1):
                        target_road = road_a if t < switch_idx else road_c
                        best = self._best_candidate_on_road_lane(frame_candidates[t], target_road, lane_a)
                        if best is None:
                            feasible = False
                            break
                        total_cost += float(best["dist"])
                        assign.append((t, best))
                    if feasible and total_cost < best_cost:
                        best_cost = total_cost
                        best_switch = switch_idx
                        best_assign = assign

                if best_assign is None:
                    continue
                if best_cost > original_cost + self.TURN_CHAIN_COST_SLACK:
                    continue

                for t, best in best_assign:
                    repl = dict(best)
                    repl["sdist"] = float(best["dist"])
                    repl["assigned_lane_id"] = int(lane_a)
                    smoothed[t] = repl

                if self.verbose:
                    print(
                        f"    [SMOOTH] Same-lane turn chain: road {road_a}->...->{road_c}, "
                        f"frames {interval_start}-{interval_end}, switch={best_switch}, "
                        f"cost {original_cost:.2f}->{best_cost:.2f}"
                    )

                changed = True
                break

            if not changed:
                break

        return smoothed

    def _inject_intersection_turn_segments(
        self,
        waypoints: List[Waypoint],
        results: List[Optional[Dict[str, object]]],
        frame_candidates: List[List[Dict[str, object]]],
    ) -> List[Optional[Dict[str, object]]]:
        """
        Replace noisy connector-road turn windows with a synthetic smooth turn arc.

        Start/end lane anchors are inferred from snapped context, then a cubic
        Bezier is fit between them for intersection-only turn windows.
        """
        if len(results) < 4:
            return results
        if "intersection" not in self.matcher.map_data.name.lower():
            return results

        smoothed = list(results)
        n = len(smoothed)

        def run_len(run: Tuple[int, int, int, int]) -> int:
            return int(run[3] - run[2] + 1)

        def build_runs(seq: List[Optional[Dict[str, object]]]) -> List[Tuple[int, int, int, int]]:
            runs: List[Tuple[int, int, int, int]] = []  # (road_id, lane_id, start, end)
            i = 0
            while i < len(seq):
                if seq[i] is None:
                    i += 1
                    continue
                road_id = int(seq[i]["road_id"])
                lane_id = int(seq[i]["lane_id"])
                start = i
                while (
                    i + 1 < len(seq)
                    and seq[i + 1] is not None
                    and int(seq[i + 1]["road_id"]) == road_id
                    and int(seq[i + 1]["lane_id"]) == lane_id
                ):
                    i += 1
                runs.append((road_id, lane_id, start, i))
                i += 1
            return runs

        def bezier(
            p0: Tuple[float, float],
            p1: Tuple[float, float],
            p2: Tuple[float, float],
            p3: Tuple[float, float],
            u: float,
        ) -> Tuple[float, float]:
            om = 1.0 - u
            b0 = om * om * om
            b1 = 3.0 * om * om * u
            b2 = 3.0 * om * u * u
            b3 = u * u * u
            x = b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0]
            y = b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1]
            return (x, y)

        def bezier_derivative(
            p0: Tuple[float, float],
            p1: Tuple[float, float],
            p2: Tuple[float, float],
            p3: Tuple[float, float],
            u: float,
        ) -> Tuple[float, float]:
            om = 1.0 - u
            dx = (
                3.0 * om * om * (p1[0] - p0[0])
                + 6.0 * om * u * (p2[0] - p1[0])
                + 3.0 * u * u * (p3[0] - p2[0])
            )
            dy = (
                3.0 * om * om * (p1[1] - p0[1])
                + 6.0 * om * u * (p2[1] - p1[1])
                + 3.0 * u * u * (p3[1] - p2[1])
            )
            return (dx, dy)

        while True:
            changed = False
            runs = build_runs(smoothed)
            if len(runs) < 4:
                break

            run_idx = 1  # needs one run before connector streak
            while run_idx < len(runs) - 1:
                start_road, start_lane, _, _ = runs[run_idx - 1]

                # Grow connector streak from run_idx forward while runs are short.
                end_idx = run_idx
                streak_frames = 0
                while end_idx < len(runs):
                    curr_len = run_len(runs[end_idx])
                    if curr_len > self.TURN_CONNECTOR_MAX_FRAMES:
                        break
                    if streak_frames + curr_len > self.TURN_CHAIN_MAX_FRAMES:
                        break
                    streak_frames += curr_len
                    end_idx += 1

                # end_idx is exclusive of short-run streak.
                connector_count = end_idx - run_idx
                if connector_count < 1:
                    run_idx += 1
                    continue

                # Prefer a stable post-turn run when present so destination
                # lane intent controls the synthetic arc endpoint.
                has_stable_post = end_idx < len(runs)
                if has_stable_post:
                    end_anchor_idx = end_idx
                    end_road, end_lane, end_anchor_start, end_anchor_end = runs[end_anchor_idx]
                    interval_end = min(
                        int(end_anchor_end),
                        int(end_anchor_start) + max(1, int(self.TURN_END_BLEND_FRAMES)) - 1,
                    )
                else:
                    # Truncated tail: use last connector run as endpoint.
                    if connector_count < 2:
                        run_idx += 1
                        continue
                    end_anchor_idx = end_idx - 1
                    end_road, end_lane, _, end_anchor_end = runs[end_anchor_idx]
                    interval_end = int(end_anchor_end)

                if int(start_road) == int(end_road) and int(start_lane) == int(end_lane):
                    run_idx += 1
                    continue

                interval_start = runs[run_idx][2]
                interval_len = interval_end - interval_start + 1
                if interval_len < self.TURN_MIN_FRAMES:
                    run_idx += 1
                    continue

                yaw_change = self._normalize_yaw_diff(
                    float(waypoints[interval_start].yaw),
                    float(waypoints[interval_end].yaw),
                )
                if yaw_change < self.TURN_MIN_YAW_CHANGE:
                    run_idx += 1
                    continue

                start_anchor = self._best_candidate_on_road_lane(
                    frame_candidates[interval_start], start_road, start_lane
                )
                end_anchor = self._best_candidate_on_road_lane(
                    frame_candidates[interval_end], end_road, end_lane
                )
                if start_anchor is None or end_anchor is None:
                    run_idx += 1
                    continue

                p0 = (float(start_anchor["x"]), float(start_anchor["y"]))
                p3 = (float(end_anchor["x"]), float(end_anchor["y"]))
                span = math.hypot(p3[0] - p0[0], p3[1] - p0[1])
                if span < 1e-3:
                    run_idx += 1
                    continue

                y0 = math.radians(float(start_anchor["yaw"]))
                y3 = math.radians(float(end_anchor["yaw"]))
                base_d = max(2.0, min(20.0, self.TURN_BEZIER_SCALE * span))

                # Parametrize by raw cumulative displacement to preserve timing.
                cum: List[float] = [0.0]
                for t in range(interval_start + 1, interval_end + 1):
                    dx = float(waypoints[t].x) - float(waypoints[t - 1].x)
                    dy = float(waypoints[t].y) - float(waypoints[t - 1].y)
                    cum.append(cum[-1] + math.hypot(dx, dy))
                total = max(1e-6, cum[-1])

                original_cost = 0.0
                original_max = 0.0
                for t in range(interval_start, interval_end + 1):
                    curr = smoothed[t]
                    if curr is not None:
                        curr_dist = float(curr.get("dist", curr.get("sdist", 0.0)))
                    elif frame_candidates[t]:
                        curr_dist = float(min(float(c["dist"]) for c in frame_candidates[t]))
                    else:
                        curr_dist = 0.0
                    original_cost += curr_dist
                    original_max = max(original_max, curr_dist)

                synthetic_rows: Optional[List[Tuple[int, Dict[str, object]]]] = None
                synthetic_cost = float("inf")
                synthetic_max = float("inf")
                best_d = base_d
                tested_d: set = set()
                for d_mult in (0.35, 0.5, 0.75, 1.0, 1.25, 1.5):
                    d = max(2.0, min(20.0, base_d * float(d_mult)))
                    d_key = round(d, 6)
                    if d_key in tested_d:
                        continue
                    tested_d.add(d_key)

                    p1 = (p0[0] + d * math.cos(y0), p0[1] + d * math.sin(y0))
                    p2 = (p3[0] - d * math.cos(y3), p3[1] - d * math.sin(y3))

                    candidate_rows: List[Tuple[int, Dict[str, object]]] = []
                    candidate_cost = 0.0
                    candidate_max = 0.0
                    for k, t in enumerate(range(interval_start, interval_end + 1)):
                        u = max(0.0, min(1.0, cum[k] / total))
                        x, y = bezier(p0, p1, p2, p3, u)
                        dx, dy = bezier_derivative(p0, p1, p2, p3, u)
                        if abs(dx) + abs(dy) > 1e-9:
                            yaw = math.degrees(math.atan2(dy, dx))
                        else:
                            yaw = self._interp_yaw_deg(float(start_anchor["yaw"]), float(end_anchor["yaw"]), u)
                        z = (1.0 - u) * float(start_anchor["z"]) + u * float(end_anchor["z"])
                        sdist = math.hypot(float(waypoints[t].x) - x, float(waypoints[t].y) - y)
                        use_start_meta = u < 0.5
                        meta = start_anchor if use_start_meta else end_anchor
                        repl = dict(meta)
                        repl["x"] = float(x)
                        repl["y"] = float(y)
                        repl["z"] = float(z)
                        repl["yaw"] = float(yaw)
                        repl["dist"] = float(sdist)
                        repl["sdist"] = float(sdist)
                        repl["road_id"] = int(start_road if use_start_meta else end_road)
                        repl["lane_id"] = int(start_lane if use_start_meta else end_lane)
                        repl["assigned_lane_id"] = int(start_lane if use_start_meta else end_lane)
                        repl["synthetic_turn"] = True
                        repl["turn_u"] = float(u)
                        candidate_rows.append((t, repl))
                        candidate_cost += float(sdist)
                        candidate_max = max(candidate_max, float(sdist))

                    if (candidate_cost + 1e-6) < synthetic_cost or (
                        abs(candidate_cost - synthetic_cost) <= 1e-6 and candidate_max < synthetic_max
                    ):
                        synthetic_rows = candidate_rows
                        synthetic_cost = float(candidate_cost)
                        synthetic_max = float(candidate_max)
                        best_d = float(d)

                if synthetic_rows is None:
                    run_idx += 1
                    continue

                local_slack = max(
                    10.0,
                    float(self.TURN_SYNTH_COST_SLACK) + float(self.TURN_SYNTH_PER_FRAME_SLACK) * float(interval_len),
                )
                if synthetic_cost > original_cost + local_slack:
                    run_idx += 1
                    continue
                if synthetic_max > max(float(self.TURN_SYNTH_MAX_POINT_DIST), original_max + 4.0):
                    run_idx += 1
                    continue

                for t, repl in synthetic_rows:
                    smoothed[t] = repl

                if self.verbose:
                    print(
                        f"    [TURN] Synthetic intersection arc: "
                        f"{start_road}_{start_lane} -> {end_road}_{end_lane}, "
                        f"frames {interval_start}-{interval_end}, yaw_change={yaw_change:.1f}, "
                        f"cost {original_cost:.2f}->{synthetic_cost:.2f}, "
                        f"max {original_max:.2f}->{synthetic_max:.2f}, d={best_d:.2f}"
                    )

                changed = True
                break

            if not changed:
                break

        return smoothed

    def _inject_direct_intersection_turn_boundaries(
        self,
        waypoints: List[Waypoint],
        results: List[Optional[Dict[str, object]]],
        frame_candidates: List[List[Dict[str, object]]],
    ) -> List[Optional[Dict[str, object]]]:
        """
        Inject synthetic turn arcs for direct road-boundary transitions.

        This complements connector-chain turn synthesis when trajectories switch
        directly from one stable road run to another at intersections.
        """
        if len(results) < 6:
            return results
        if "intersection" not in self.matcher.map_data.name.lower():
            return results

        before_frames = _env_int("V2X_ALIGN_DIRECT_TURN_BEFORE_FRAMES", 8, minimum=2, maximum=30)
        after_frames = _env_int("V2X_ALIGN_DIRECT_TURN_AFTER_FRAMES", 10, minimum=2, maximum=30)
        min_neighbor_frames = _env_int("V2X_ALIGN_DIRECT_TURN_MIN_NEIGHBOR_FRAMES", 4, minimum=3, maximum=50)
        min_frames = _env_int("V2X_ALIGN_DIRECT_TURN_MIN_FRAMES", 8, minimum=4, maximum=40)
        min_yaw_change = _env_float("V2X_ALIGN_DIRECT_TURN_MIN_YAW_CHANGE_DEG", 16.0)
        enable_low_yaw_jump_bridge = (
            _env_int("V2X_ALIGN_DIRECT_TURN_ENABLE_LOW_YAW_JUMP_BRIDGE", 1, minimum=0, maximum=1) == 1
        )
        low_yaw_jump_abs_m = _env_float("V2X_ALIGN_DIRECT_TURN_LOW_YAW_JUMP_ABS_M", 2.2)
        low_yaw_jump_ratio = _env_float("V2X_ALIGN_DIRECT_TURN_LOW_YAW_JUMP_RATIO", 2.8)
        low_yaw_jump_raw_floor_m = _env_float("V2X_ALIGN_DIRECT_TURN_LOW_YAW_JUMP_RAW_FLOOR_M", 0.25)
        cost_slack_base = _env_float("V2X_ALIGN_DIRECT_TURN_COST_SLACK_BASE", 2.0)
        cost_slack_per_frame = _env_float("V2X_ALIGN_DIRECT_TURN_COST_SLACK_PER_FRAME", 1.0)
        max_point_dist = _env_float("V2X_ALIGN_DIRECT_TURN_MAX_POINT_DIST", 9.0)

        smoothed = list(results)
        n = len(smoothed)

        def _build_runs(seq: List[Optional[Dict[str, object]]]) -> List[Tuple[int, int, int, int]]:
            runs: List[Tuple[int, int, int, int]] = []  # (road_id, lane_id, start, end)
            i = 0
            while i < len(seq):
                if seq[i] is None:
                    i += 1
                    continue
                road_id = int(seq[i]["road_id"])
                lane_id = int(seq[i]["lane_id"])
                s = i
                while (
                    i + 1 < len(seq)
                    and seq[i + 1] is not None
                    and int(seq[i + 1]["road_id"]) == int(road_id)
                    and int(seq[i + 1]["lane_id"]) == int(lane_id)
                ):
                    i += 1
                runs.append((int(road_id), int(lane_id), int(s), int(i)))
                i += 1
            return runs

        def _bezier(
            p0: Tuple[float, float],
            p1: Tuple[float, float],
            p2: Tuple[float, float],
            p3: Tuple[float, float],
            u: float,
        ) -> Tuple[float, float]:
            om = 1.0 - float(u)
            b0 = om * om * om
            b1 = 3.0 * om * om * float(u)
            b2 = 3.0 * om * float(u) * float(u)
            b3 = float(u) * float(u) * float(u)
            return (
                b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0],
                b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1],
            )

        def _bezier_deriv(
            p0: Tuple[float, float],
            p1: Tuple[float, float],
            p2: Tuple[float, float],
            p3: Tuple[float, float],
            u: float,
        ) -> Tuple[float, float]:
            om = 1.0 - float(u)
            return (
                3.0 * om * om * (p1[0] - p0[0]) + 6.0 * om * float(u) * (p2[0] - p1[0]) + 3.0 * float(u) * float(u) * (p3[0] - p2[0]),
                3.0 * om * om * (p1[1] - p0[1]) + 6.0 * om * float(u) * (p2[1] - p1[1]) + 3.0 * float(u) * float(u) * (p3[1] - p2[1]),
            )

        while True:
            changed = False
            runs = _build_runs(smoothed)
            if len(runs) < 2:
                break

            for ridx in range(0, len(runs) - 1):
                road_a, lane_a, a_start, a_end = runs[ridx]
                road_b, lane_b, b_start, b_end = runs[ridx + 1]
                len_a = int(a_end - a_start + 1)
                len_b = int(b_end - b_start + 1)
                if len_a < int(min_neighbor_frames) or len_b < int(min_neighbor_frames):
                    continue
                if int(road_a) == int(road_b) and int(lane_a) == int(lane_b):
                    continue
                if int(road_a) == int(road_b):
                    continue

                interval_start = max(int(a_start), int(a_end) - int(before_frames) + 1)
                interval_end = min(int(b_end), int(b_start) + int(after_frames) - 1)
                interval_len = int(interval_end - interval_start + 1)
                if interval_len < int(min_frames):
                    continue
                if any(bool(smoothed[t].get("synthetic_turn", False)) for t in range(interval_start, interval_end + 1) if smoothed[t] is not None):
                    continue

                yaw_change = self._normalize_yaw_diff(
                    float(waypoints[interval_start].yaw),
                    float(waypoints[interval_end].yaw),
                )
                if float(yaw_change) < float(min_yaw_change):
                    if not bool(enable_low_yaw_jump_bridge):
                        continue
                    if int(a_end) + 1 != int(b_start):
                        continue
                    if smoothed[a_end] is None or smoothed[b_start] is None:
                        continue
                    raw_step = float(
                        math.hypot(
                            float(waypoints[b_start].x) - float(waypoints[a_end].x),
                            float(waypoints[b_start].y) - float(waypoints[a_end].y),
                        )
                    )
                    snap_step = float(
                        math.hypot(
                            float(smoothed[b_start]["x"]) - float(smoothed[a_end]["x"]),
                            float(smoothed[b_start]["y"]) - float(smoothed[a_end]["y"]),
                        )
                    )
                    jump_thr = max(
                        float(low_yaw_jump_abs_m),
                        float(low_yaw_jump_ratio) * max(float(low_yaw_jump_raw_floor_m), float(raw_step)),
                    )
                    if snap_step <= float(jump_thr):
                        continue

                start_anchor = self._best_candidate_on_road_lane(frame_candidates[interval_start], int(road_a), int(lane_a))
                end_anchor = self._best_candidate_on_road_lane(frame_candidates[interval_end], int(road_b), int(lane_b))
                if start_anchor is None or end_anchor is None:
                    continue

                p0 = (float(start_anchor["x"]), float(start_anchor["y"]))
                p3 = (float(end_anchor["x"]), float(end_anchor["y"]))
                span = math.hypot(float(p3[0]) - float(p0[0]), float(p3[1]) - float(p0[1]))
                if span < 1e-3:
                    continue
                y0 = math.radians(float(start_anchor["yaw"]))
                y3 = math.radians(float(end_anchor["yaw"]))
                base_d = max(2.0, min(20.0, float(self.TURN_BEZIER_SCALE) * float(span)))

                cum: List[float] = [0.0]
                for t in range(interval_start + 1, interval_end + 1):
                    dx = float(waypoints[t].x) - float(waypoints[t - 1].x)
                    dy = float(waypoints[t].y) - float(waypoints[t - 1].y)
                    cum.append(float(cum[-1]) + float(math.hypot(float(dx), float(dy))))
                total = max(1e-6, float(cum[-1]))

                original_cost = 0.0
                original_max = 0.0
                for t in range(interval_start, interval_end + 1):
                    curr = smoothed[t]
                    if curr is not None:
                        curr_dist = float(curr.get("dist", curr.get("sdist", 0.0)))
                    elif frame_candidates[t]:
                        curr_dist = float(min(float(c["dist"]) for c in frame_candidates[t]))
                    else:
                        curr_dist = 0.0
                    original_cost += float(curr_dist)
                    original_max = max(float(original_max), float(curr_dist))

                synthetic_rows: Optional[List[Tuple[int, Dict[str, object]]]] = None
                synthetic_cost = float("inf")
                synthetic_max = float("inf")
                for d_mult in (0.35, 0.5, 0.75, 1.0, 1.25):
                    d = max(2.0, min(20.0, float(base_d) * float(d_mult)))
                    p1 = (float(p0[0]) + float(d) * math.cos(y0), float(p0[1]) + float(d) * math.sin(y0))
                    p2 = (float(p3[0]) - float(d) * math.cos(y3), float(p3[1]) - float(d) * math.sin(y3))
                    cand_rows: List[Tuple[int, Dict[str, object]]] = []
                    cand_cost = 0.0
                    cand_max = 0.0
                    for k, t in enumerate(range(interval_start, interval_end + 1)):
                        u = max(0.0, min(1.0, float(cum[k]) / float(total)))
                        bx, by = _bezier(p0, p1, p2, p3, float(u))
                        dx, dy = _bezier_deriv(p0, p1, p2, p3, float(u))
                        if abs(float(dx)) + abs(float(dy)) > 1e-9:
                            yaw = math.degrees(math.atan2(float(dy), float(dx)))
                        else:
                            yaw = self._interp_yaw_deg(float(start_anchor["yaw"]), float(end_anchor["yaw"]), float(u))
                        z = (1.0 - float(u)) * float(start_anchor["z"]) + float(u) * float(end_anchor["z"])
                        sdist = math.hypot(float(waypoints[t].x) - float(bx), float(waypoints[t].y) - float(by))
                        use_start_meta = float(u) < 0.5
                        meta = start_anchor if bool(use_start_meta) else end_anchor
                        repl = dict(meta)
                        repl["x"] = float(bx)
                        repl["y"] = float(by)
                        repl["z"] = float(z)
                        repl["yaw"] = float(yaw)
                        repl["dist"] = float(sdist)
                        repl["sdist"] = float(sdist)
                        repl["road_id"] = int(road_a if bool(use_start_meta) else road_b)
                        repl["lane_id"] = int(lane_a if bool(use_start_meta) else lane_b)
                        repl["assigned_lane_id"] = int(lane_a if bool(use_start_meta) else lane_b)
                        repl["synthetic_turn"] = True
                        repl["turn_u"] = float(u)
                        cand_rows.append((int(t), repl))
                        cand_cost += float(sdist)
                        cand_max = max(float(cand_max), float(sdist))
                    if (float(cand_cost) + 1e-6) < float(synthetic_cost) or (
                        abs(float(cand_cost) - float(synthetic_cost)) <= 1e-6 and float(cand_max) < float(synthetic_max)
                    ):
                        synthetic_rows = cand_rows
                        synthetic_cost = float(cand_cost)
                        synthetic_max = float(cand_max)

                if synthetic_rows is None:
                    continue

                local_slack = max(
                    8.0,
                    float(cost_slack_base) + float(cost_slack_per_frame) * float(interval_len),
                )
                if float(synthetic_cost) > float(original_cost) + float(local_slack):
                    continue
                if float(synthetic_max) > max(float(max_point_dist), float(original_max) + 3.5):
                    continue

                for t, repl in synthetic_rows:
                    smoothed[t] = repl
                if self.verbose:
                    print(
                        f"    [TURN] Direct boundary arc: {road_a}_{lane_a}->{road_b}_{lane_b}, "
                        f"frames {interval_start}-{interval_end}, yaw_change={yaw_change:.1f}, "
                        f"cost {original_cost:.2f}->{synthetic_cost:.2f}"
                    )
                changed = True
                break

            if not changed:
                break

        return smoothed

    def _lane_transition_geom_cost(
        self,
        from_road_id: int,
        from_lane_id: int,
        to_road_id: int,
        to_lane_id: int,
    ) -> float:
        """
        Geometry fallback for road-to-road continuity when explicit graph edges are sparse.
        Uses minimum endpoint distance between lane centerlines.
        """
        from_feats = self._road_lane_features.get((int(from_road_id), int(from_lane_id)), [])
        to_feats = self._road_lane_features.get((int(to_road_id), int(to_lane_id)), [])
        if not from_feats or not to_feats:
            return 25.0

        best = float("inf")
        for fa in from_feats:
            if fa.polyline.shape[0] < 2:
                continue
            a_pts = [fa.polyline[0, :2], fa.polyline[-1, :2]]
            for fb in to_feats:
                if fb.polyline.shape[0] < 2:
                    continue
                b_pts = [fb.polyline[0, :2], fb.polyline[-1, :2]]
                for pa in a_pts:
                    for pb in b_pts:
                        d = float(math.hypot(float(pa[0]) - float(pb[0]), float(pa[1]) - float(pb[1])))
                        if d < best:
                            best = d
        if not math.isfinite(best):
            return 25.0
        return best

    @staticmethod
    def _best_candidate_on_road_lane(
        candidates: List[Dict[str, object]],
        road_id: int,
        lane_id: int,
    ) -> Optional[Dict[str, object]]:
        matching = [
            c for c in candidates
            if int(c.get("road_id", -1)) == int(road_id) and int(c.get("lane_id", 0)) == int(lane_id)
        ]
        if not matching:
            return None
        return min(matching, key=lambda c: float(c["dist"]))

    def _refine_multilane_road_runs(
        self,
        results: List[Optional[Dict[str, object]]],
        frame_candidates: List[List[Dict[str, object]]],
    ) -> List[Optional[Dict[str, object]]]:
        """
        Re-optimize lane choice within each multi-lane road run.

        This addresses edge cases where global lane_id continuity keeps a vehicle
        in the wrong approach lane before an intersection turn.
        """
        if len(results) < 2:
            return results
        if "intersection" not in self.matcher.map_data.name.lower():
            return results

        smoothed = list(results)
        n = len(smoothed)

        # Build contiguous runs by road_id.
        runs: List[Tuple[int, int, int]] = []
        i = 0
        while i < n:
            if smoothed[i] is None:
                i += 1
                continue
            road_id = int(smoothed[i]["road_id"])
            start = i
            while (
                i + 1 < n
                and smoothed[i + 1] is not None
                and int(smoothed[i + 1]["road_id"]) == road_id
            ):
                i += 1
            runs.append((road_id, start, i))
            i += 1

        for run_idx, (road_id, start, end) in enumerate(runs):
            lane_ids = self._road_lane_ids.get(int(road_id), [])
            if len(lane_ids) <= 1:
                continue

            # Keep only lane_ids that are represented in this run's candidates.
            active_lane_ids: List[int] = []
            for lid in lane_ids:
                present = any(
                    self._best_candidate_on_road_lane(frame_candidates[t], road_id, lid) is not None
                    for t in range(start, end + 1)
                )
                if present:
                    active_lane_ids.append(int(lid))
            if len(active_lane_ids) <= 1:
                continue

            prev_ctx = smoothed[start - 1] if start - 1 >= 0 else None
            next_ctx = smoothed[end + 1] if end + 1 < n else None

            dp_costs: List[Dict[int, float]] = []
            dp_back: List[Dict[int, int]] = []

            # Initialize
            init_costs: Dict[int, float] = {}
            init_back: Dict[int, int] = {}
            for lid in active_lane_ids:
                cand = self._best_candidate_on_road_lane(frame_candidates[start], road_id, lid)
                if cand is None:
                    continue
                cost = float(cand["dist"])
                if prev_ctx is not None and int(prev_ctx.get("road_id", road_id)) != int(road_id):
                    cost += self.RUN_START_CONN_WEIGHT * self._lane_transition_geom_cost(
                        int(prev_ctx.get("road_id", road_id)),
                        int(prev_ctx.get("lane_id", lid)),
                        int(road_id),
                        int(lid),
                    )
                init_costs[int(lid)] = cost
                init_back[int(lid)] = int(lid)
            if not init_costs:
                continue
            dp_costs.append(init_costs)
            dp_back.append(init_back)

            # Forward DP
            for t in range(start + 1, end + 1):
                curr_costs: Dict[int, float] = {}
                curr_back: Dict[int, int] = {}
                for lid in active_lane_ids:
                    cand = self._best_candidate_on_road_lane(frame_candidates[t], road_id, lid)
                    if cand is None:
                        continue
                    emission = float(cand["dist"])
                    best_total = float("inf")
                    best_prev = None
                    for prev_lid, prev_total in dp_costs[-1].items():
                        trans = 0.0
                        if int(prev_lid) != int(lid):
                            trans += self.RUN_LANE_CHANGE_PENALTY * float(
                                max(1, self._lane_lateral_distance(int(prev_lid), int(lid)))
                            )
                        total = float(prev_total) + trans + emission
                        if total < best_total:
                            best_total = total
                            best_prev = int(prev_lid)
                    if best_prev is None:
                        continue
                    curr_costs[int(lid)] = best_total
                    curr_back[int(lid)] = best_prev
                if not curr_costs:
                    break
                dp_costs.append(curr_costs)
                dp_back.append(curr_back)

            if len(dp_costs) != (end - start + 1):
                continue

            # End preference toward successor road.
            final_costs = dict(dp_costs[-1])
            if next_ctx is not None and int(next_ctx.get("road_id", road_id)) != int(road_id):
                next_road = int(next_ctx.get("road_id", road_id))
                next_lane = int(next_ctx.get("lane_id", 0))
                for lid in list(final_costs.keys()):
                    final_costs[lid] += self.RUN_END_CONN_WEIGHT * self._lane_transition_geom_cost(
                        int(road_id),
                        int(lid),
                        next_road,
                        next_lane,
                    )

            if not final_costs:
                continue
            best_last_lid = min(final_costs.keys(), key=lambda lid: float(final_costs[lid]))

            # Backtrack lane_id path for this run.
            path_lids: List[int] = [int(best_last_lid)]
            for k in range(len(dp_back) - 1, 0, -1):
                prev_lid = dp_back[k].get(path_lids[-1])
                if prev_lid is None:
                    break
                path_lids.append(int(prev_lid))
            path_lids.reverse()
            if len(path_lids) != (end - start + 1):
                continue

            # Apply refined path.
            changed = False
            for offset, lid in enumerate(path_lids):
                t = start + offset
                best = self._best_candidate_on_road_lane(frame_candidates[t], road_id, lid)
                if best is None:
                    continue
                curr = smoothed[t]
                if curr is None:
                    continue
                if int(curr.get("lane_id", 0)) != int(lid):
                    changed = True
                repl = dict(best)
                repl["sdist"] = float(best["dist"])
                repl["assigned_lane_id"] = int(lid)
                smoothed[t] = repl

            if changed and self.verbose:
                old_lids = {
                    int(results[t]["lane_id"])
                    for t in range(start, end + 1)
                    if results[t] is not None and int(results[t]["road_id"]) == int(road_id)
                }
                new_lids = {
                    int(smoothed[t]["lane_id"])
                    for t in range(start, end + 1)
                    if smoothed[t] is not None and int(smoothed[t]["road_id"]) == int(road_id)
                }
                print(
                    f"    [REFINE] Multi-lane run road {road_id} frames {start}-{end}: "
                    f"lanes {sorted(old_lids)} -> {sorted(new_lids)}"
                )

        return smoothed

    def _lock_intersection_approach_lane(
        self,
        waypoints: List[Waypoint],
        results: List[Optional[Dict[str, object]]],
        frame_candidates: List[List[Dict[str, object]]],
    ) -> List[Optional[Dict[str, object]]]:
        """
        Simplify two-step approach-lane cascades on one road before a road exit.

        Pattern targeted:
          same road: lane A -> lane B, then exit to different road.
        If lane B is the committed tail lane before the exit, relabel the whole
        same-road approach window to lane B when fit degradation stays bounded.
        """
        if len(results) < 3 or len(results) != len(waypoints):
            return results
        if "intersection" not in self.matcher.map_data.name.lower():
            return results

        smoothed = list(results)
        n = len(smoothed)
        max_cost_ratio = _env_float("V2X_ALIGN_APPROACH_LOCK_MAX_COST_RATIO", 1.65)
        max_mean_delta_m = _env_float("V2X_ALIGN_APPROACH_LOCK_MAX_MEAN_DELTA_M", 0.45)
        sign_flip_max_mean_delta_m = _env_float("V2X_ALIGN_APPROACH_LOCK_SIGN_FLIP_MAX_MEAN_DELTA_M", 0.20)
        approach_lock_step_guard_floor_m = _env_float("V2X_ALIGN_APPROACH_LOCK_STEP_GUARD_FLOOR_M", 1.2)
        approach_lock_step_guard_ratio = _env_float("V2X_ALIGN_APPROACH_LOCK_STEP_GUARD_RATIO", 2.2)

        def _raw_step(i0: int, i1: int) -> float:
            if i0 < 0 or i1 < 0 or i0 >= len(waypoints) or i1 >= len(waypoints):
                return 0.0
            return float(
                math.hypot(
                    float(waypoints[i1].x) - float(waypoints[i0].x),
                    float(waypoints[i1].y) - float(waypoints[i0].y),
                )
            )

        def _max_snap_step_over_transitions(
            step_start: int,
            step_end: int,
            xy_override: Optional[Dict[int, Tuple[float, float]]] = None,
        ) -> float:
            if step_start > step_end:
                return 0.0
            best = 0.0
            for step_idx in range(int(step_start), int(step_end) + 1):
                if step_idx <= 0 or step_idx >= n:
                    continue
                row_a = smoothed[step_idx - 1]
                row_b = smoothed[step_idx]
                if row_a is None or row_b is None:
                    return float("inf")
                ax, ay = (
                    xy_override.get(step_idx - 1, (float(row_a.get("x", 0.0)), float(row_a.get("y", 0.0))))
                    if isinstance(xy_override, dict)
                    else (float(row_a.get("x", 0.0)), float(row_a.get("y", 0.0)))
                )
                bx, by = (
                    xy_override.get(step_idx, (float(row_b.get("x", 0.0)), float(row_b.get("y", 0.0))))
                    if isinstance(xy_override, dict)
                    else (float(row_b.get("x", 0.0)), float(row_b.get("y", 0.0)))
                )
                step = float(math.hypot(float(bx) - float(ax), float(by) - float(ay)))
                best = max(float(best), float(step))
            return float(best)

        while True:
            changed = False
            runs: List[Tuple[int, int, int, int]] = []  # (road_id, lane_id, start, end)
            i = 0
            while i < n:
                if smoothed[i] is None:
                    i += 1
                    continue
                road_id = int(smoothed[i]["road_id"])
                lane_id = int(smoothed[i]["lane_id"])
                start = i
                while (
                    i + 1 < n
                    and smoothed[i + 1] is not None
                    and int(smoothed[i + 1]["road_id"]) == road_id
                    and int(smoothed[i + 1]["lane_id"]) == lane_id
                ):
                    i += 1
                runs.append((road_id, lane_id, start, i))
                i += 1

            if len(runs) < 2:
                break

            for run_idx in range(0, len(runs) - 1):
                road_a, lane_a, a_start, a_end = runs[run_idx]
                road_b, lane_b, b_start, b_end = runs[run_idx + 1]
                has_next = (run_idx + 2) < len(runs)
                next_road = None
                next_lane = None
                if has_next:
                    next_road, next_lane, _, _ = runs[run_idx + 2]

                if road_a != road_b:
                    continue
                if lane_a == lane_b:
                    continue
                if has_next and next_road == road_a:
                    continue

                # Keep only adjacent one-step lane shifts (e.g., 3->2) within one road.
                if self._lane_lateral_distance(int(lane_a), int(lane_b)) != 1:
                    continue

                approach_len = int(b_end - a_start + 1)
                tail_len = int(b_end - b_start + 1)
                if approach_len > int(self.APPROACH_LANE_LOCK_MAX_RUN_FRAMES):
                    continue
                if tail_len < int(self.APPROACH_LANE_LOCK_MIN_TAIL_FRAMES):
                    continue

                # Only lock when final approach lane is at least as compatible with
                # the upcoming lane as the earlier lane.
                if has_next and next_lane is not None:
                    dist_a_next = self._lane_lateral_distance(int(lane_a), int(next_lane))
                    dist_b_next = self._lane_lateral_distance(int(lane_b), int(next_lane))
                    if dist_b_next > dist_a_next:
                        continue

                original_cost = 0.0
                replacement_cost = 0.0
                original_max = 0.0
                replacement_max = 0.0
                replacement: List[Tuple[int, Dict[str, object]]] = []
                feasible = True

                for t in range(a_start, b_end + 1):
                    curr = smoothed[t]
                    if curr is None:
                        feasible = False
                        break
                    curr_dist = float(curr.get("dist", curr.get("sdist", 0.0)))
                    original_cost += curr_dist
                    original_max = max(original_max, curr_dist)

                    best = self._best_candidate_on_road_lane(frame_candidates[t], road_a, lane_b)
                    if best is None:
                        feasible = False
                        break
                    best_dist = float(best["dist"])
                    replacement_cost += best_dist
                    replacement_max = max(replacement_max, best_dist)
                    replacement.append((t, best))

                if not feasible:
                    continue

                local_slack = float(self.APPROACH_LANE_LOCK_PER_FRAME_SLACK) * float(approach_len)
                if replacement_cost > original_cost + local_slack:
                    continue
                # Guardrail: do not force a lane relabel when geometric fit
                # degrades too much in aggregate or per-frame average.
                if original_cost > 1e-6 and replacement_cost > float(original_cost) * float(max_cost_ratio):
                    continue
                mean_delta = (float(replacement_cost) - float(original_cost)) / max(1.0, float(approach_len))
                if mean_delta > float(max_mean_delta_m):
                    continue
                if int(lane_a) * int(lane_b) < 0 and mean_delta > float(sign_flip_max_mean_delta_m):
                    continue
                if replacement_max > max(
                    float(self.APPROACH_LANE_LOCK_MAX_POINT_DIST),
                    original_max + float(self.APPROACH_LANE_LOCK_MAX_POINT_DELTA),
                ):
                    continue

                replacement_map = {
                    int(t): (float(best["x"]), float(best["y"]))
                    for t, best in replacement
                }
                step_eval_start = max(1, int(a_start) - 1)
                step_eval_end = min(int(n) - 1, int(b_end) + 1)
                raw_step_local_max = 0.0
                for step_idx in range(int(step_eval_start), int(step_eval_end) + 1):
                    raw_step_local_max = max(
                        float(raw_step_local_max),
                        float(_raw_step(int(step_idx - 1), int(step_idx))),
                    )
                original_snap_step_max = _max_snap_step_over_transitions(
                    int(step_eval_start),
                    int(step_eval_end),
                    xy_override=None,
                )
                replacement_snap_step_max = _max_snap_step_over_transitions(
                    int(step_eval_start),
                    int(step_eval_end),
                    xy_override=replacement_map,
                )
                step_guard_limit = max(
                    float(approach_lock_step_guard_floor_m),
                    float(approach_lock_step_guard_ratio) * float(raw_step_local_max),
                )
                if (
                    float(replacement_snap_step_max) > float(original_snap_step_max) + 1e-6
                    and float(replacement_snap_step_max) > float(step_guard_limit)
                ):
                    continue

                for t, best in replacement:
                    repl = dict(best)
                    repl["sdist"] = float(best["dist"])
                    repl["assigned_lane_id"] = int(lane_b)
                    smoothed[t] = repl

                if self.verbose:
                    next_label = f"{next_road}_{next_lane}" if has_next and next_road is not None and next_lane is not None else "segment_end"
                    print(
                        f"    [SMOOTH] Approach lane lock: road {road_a}, "
                        f"lane {lane_a}->{lane_b}, frames {a_start}-{b_end}, "
                        f"next={next_label}, "
                        f"cost {original_cost:.2f}->{replacement_cost:.2f}"
                    )

                changed = True
                break

            if not changed:
                break

        return smoothed

    @staticmethod
    def _interp_yaw_deg(yaw_a: float, yaw_b: float, alpha: float) -> float:
        """Interpolate yaw via shortest angular path."""
        a = float(yaw_a)
        b = float(yaw_b)
        diff = (b - a + 180.0) % 360.0 - 180.0
        y = a + float(alpha) * diff
        while y > 180.0:
            y -= 360.0
        while y <= -180.0:
            y += 360.0
        return y

    def _smooth_intersection_lane_changes(
        self,
        waypoints: List[Waypoint],
        results: List[Optional[Dict[str, object]]],
        frame_candidates: List[List[Dict[str, object]]],
    ) -> List[Optional[Dict[str, object]]]:
        """
        For intersection scenarios, blend lane-change windows on the same road.

        This produces a smooth transition between source/target lanes instead of
        snapping abruptly from one lane centerline to the other.
        """
        if len(results) < 3 or len(results) != len(waypoints):
            return results
        if "intersection" not in self.matcher.map_data.name.lower():
            return results

        smoothed = list(results)
        n = len(smoothed)
        min_yaw_evidence_deg = _env_float("V2X_ALIGN_INTERSECTION_BLEND_MIN_YAW_EVIDENCE_DEG", 12.0)
        min_lateral_evidence_m = _env_float("V2X_ALIGN_INTERSECTION_BLEND_MIN_LATERAL_EVIDENCE_M", 0.8)
        max_low_motion_mean_step_m = _env_float("V2X_ALIGN_INTERSECTION_BLEND_MAX_LOW_MOTION_MEAN_STEP_M", 0.4)
        i = 1
        while i < n:
            prev = smoothed[i - 1]
            curr = smoothed[i]
            if prev is None or curr is None:
                i += 1
                continue

            prev_road = int(prev["road_id"])
            curr_road = int(curr["road_id"])
            prev_lid = int(prev["lane_id"])
            curr_lid = int(curr["lane_id"])

            # Blend only lane changes that happen on the same road.
            if prev_road != curr_road or prev_lid == curr_lid:
                i += 1
                continue

            run_start = i - 1
            while (
                run_start - 1 >= 0
                and smoothed[run_start - 1] is not None
                and int(smoothed[run_start - 1]["road_id"]) == curr_road
            ):
                run_start -= 1
            run_end = i
            while (
                run_end + 1 < n
                and smoothed[run_end + 1] is not None
                and int(smoothed[run_end + 1]["road_id"]) == curr_road
            ):
                run_end += 1

            blend_start = max(run_start, i - self.LANE_CHANGE_BLEND_FRAMES)
            blend_end = min(run_end, i + self.LANE_CHANGE_BLEND_FRAMES)
            denom = max(1, blend_end - blend_start)
            if blend_end <= blend_start:
                i += 1
                continue

            raw_steps: List[float] = []
            for t in range(int(blend_start) + 1, int(blend_end) + 1):
                raw_steps.append(
                    float(
                        math.hypot(
                            float(waypoints[t].x) - float(waypoints[t - 1].x),
                            float(waypoints[t].y) - float(waypoints[t - 1].y),
                        )
                    )
                )
            if not raw_steps:
                i += 1
                continue
            mean_raw_step = float(sum(raw_steps) / max(1, len(raw_steps)))
            if float(mean_raw_step) < float(max_low_motion_mean_step_m):
                i += 1
                continue
            yaw_span = float(
                self._normalize_yaw_diff(
                    float(waypoints[blend_start].yaw),
                    float(waypoints[blend_end].yaw),
                )
            )
            heading_rad = math.radians(float(waypoints[blend_start].yaw))
            raw_dx = float(waypoints[blend_end].x) - float(waypoints[blend_start].x)
            raw_dy = float(waypoints[blend_end].y) - float(waypoints[blend_start].y)
            nx = -math.sin(float(heading_rad))
            ny = math.cos(float(heading_rad))
            lateral_disp = abs(float(raw_dx) * float(nx) + float(raw_dy) * float(ny))
            if float(yaw_span) < float(min_yaw_evidence_deg) and float(lateral_disp) < float(min_lateral_evidence_m):
                i += 1
                continue

            for t in range(blend_start, blend_end + 1):
                cand_a = self._best_candidate_on_road_lane(frame_candidates[t], curr_road, prev_lid)
                cand_b = self._best_candidate_on_road_lane(frame_candidates[t], curr_road, curr_lid)
                if cand_a is None or cand_b is None:
                    continue
                u = float(t - blend_start) / float(denom)
                # Smoothstep easing for a gradual lane-crossing arc.
                alpha = u * u * (3.0 - 2.0 * u)
                blended = dict(cand_a if alpha < 0.5 else cand_b)
                blended["x"] = (1.0 - alpha) * float(cand_a["x"]) + alpha * float(cand_b["x"])
                blended["y"] = (1.0 - alpha) * float(cand_a["y"]) + alpha * float(cand_b["y"])
                blended["z"] = (1.0 - alpha) * float(cand_a["z"]) + alpha * float(cand_b["z"])
                blended["yaw"] = self._interp_yaw_deg(float(cand_a["yaw"]), float(cand_b["yaw"]), alpha)
                blended["dist"] = (1.0 - alpha) * float(cand_a["dist"]) + alpha * float(cand_b["dist"])
                blended["sdist"] = float(blended["dist"])
                blended["assigned_lane_id"] = int(curr_lid if alpha >= 0.5 else prev_lid)
                smoothed[t] = blended

            i = blend_end + 1

        return smoothed

    def _smooth_jump_stall_artifacts(
        self,
        waypoints: List[Waypoint],
        results: List[Optional[Dict[str, object]]],
    ) -> List[Optional[Dict[str, object]]]:
        """
        Remove jump-then-wait artifacts where snapped poses teleport into a lane
        and then remain nearly stationary while raw points keep moving.
        """
        if len(results) < 4 or len(waypoints) != len(results):
            return results

        smoothed = list(results)
        n = len(smoothed)

        def _raw_step(i: int) -> float:
            if i <= 0 or i >= n:
                return 0.0
            dx = float(waypoints[i].x) - float(waypoints[i - 1].x)
            dy = float(waypoints[i].y) - float(waypoints[i - 1].y)
            return float(math.hypot(dx, dy))

        def _snap_step(i: int) -> float:
            if i <= 0 or i >= n:
                return 0.0
            a = smoothed[i - 1]
            b = smoothed[i]
            if a is None or b is None:
                return 0.0
            return float(math.hypot(float(b["x"]) - float(a["x"]), float(b["y"]) - float(a["y"])))

        i = 1
        while i < n - 2:
            prev = smoothed[i - 1]
            curr = smoothed[i]
            if prev is None or curr is None:
                i += 1
                continue

            raw_jump = _raw_step(i)
            snap_jump = _snap_step(i)
            if raw_jump < float(self.JUMP_STALL_RAW_MIN_STEP) or raw_jump > float(self.JUMP_STALL_RAW_MAX_STEP):
                i += 1
                continue
            if snap_jump < max(
                float(self.JUMP_STALL_MIN_JUMP_M),
                float(self.JUMP_STALL_JUMP_RATIO) * float(raw_jump),
            ):
                i += 1
                continue

            scan_end = min(n - 1, i + int(self.JUMP_STALL_LOOKAHEAD_FRAMES))
            stall_last = i
            stall_count = 0
            j = i + 1
            while j <= scan_end:
                if smoothed[j - 1] is None or smoothed[j] is None:
                    break
                raw_step = _raw_step(j)
                snap_step = _snap_step(j)
                if raw_step > (2.0 * float(self.JUMP_STALL_RAW_MAX_STEP)):
                    break
                if raw_step >= float(self.JUMP_STALL_RAW_MIN_STEP) and snap_step <= float(self.JUMP_STALL_STALL_STEP_M):
                    stall_last = j
                    stall_count += 1
                    j += 1
                    continue
                if snap_step >= float(self.JUMP_STALL_RESUME_STEP_M):
                    break
                if raw_step >= float(self.JUMP_STALL_RAW_MIN_STEP) and snap_step <= (1.8 * float(self.JUMP_STALL_STALL_STEP_M)):
                    stall_last = j
                    stall_count += 1
                    j += 1
                    continue
                break

            if stall_count < int(self.JUMP_STALL_MIN_STALL_FRAMES):
                i += 1
                continue

            start_x = float(prev["x"])
            start_y = float(prev["y"])
            end_idx = -1
            for k in range(stall_last + 1, scan_end + 1):
                row = smoothed[k]
                if row is None:
                    break
                d = math.hypot(float(row["x"]) - start_x, float(row["y"]) - start_y)
                if d >= max(0.8, 1.5 * float(raw_jump)):
                    end_idx = k
                    break
            if end_idx < 0:
                fallback = stall_last + 1
                if fallback <= scan_end and smoothed[fallback] is not None:
                    end_idx = int(fallback)
                else:
                    i += 1
                    continue
            if end_idx <= i or smoothed[end_idx] is None:
                i += 1
                continue

            end_row = smoothed[end_idx]
            assert end_row is not None
            end_x = float(end_row["x"])
            end_y = float(end_row["y"])
            if math.hypot(end_x - start_x, end_y - start_y) < 0.5:
                i = end_idx + 1
                continue

            cum: List[float] = [0.0]
            total = 0.0
            for t in range(i, end_idx + 1):
                step = min(_raw_step(t), 1.25 * float(self.JUMP_STALL_RAW_MAX_STEP))
                total += max(0.0, float(step))
                cum.append(float(total))
            if total <= 1e-6:
                total = float(max(1, end_idx - (i - 1)))
                cum = [float(v) for v in range(0, end_idx - (i - 1) + 1)]

            yaw_start = float(prev.get("yaw", waypoints[i - 1].yaw))
            yaw_end = float(end_row.get("yaw", waypoints[end_idx].yaw))
            z_start = float(prev.get("z", waypoints[i - 1].z))
            z_end = float(end_row.get("z", waypoints[end_idx].z))

            old_cost = 0.0
            new_cost = 0.0
            old_max = 0.0
            new_max = 0.0
            patch_rows: List[Tuple[int, Dict[str, object]]] = []
            denom = max(1e-6, float(total))
            feasible = True

            for off, t in enumerate(range(i, end_idx + 1), start=1):
                old = smoothed[t]
                if old is None:
                    feasible = False
                    break
                alpha = max(0.0, min(1.0, float(cum[off]) / denom))
                x = (1.0 - alpha) * start_x + alpha * end_x
                y = (1.0 - alpha) * start_y + alpha * end_y
                z = (1.0 - alpha) * z_start + alpha * z_end
                yaw = self._interp_yaw_deg(yaw_start, yaw_end, alpha)
                raw_x = float(waypoints[t].x)
                raw_y = float(waypoints[t].y)
                old_dist = math.hypot(raw_x - float(old["x"]), raw_y - float(old["y"]))
                new_dist = math.hypot(raw_x - x, raw_y - y)
                old_cost += float(old_dist)
                new_cost += float(new_dist)
                old_max = max(old_max, float(old_dist))
                new_max = max(new_max, float(new_dist))

                meta = prev if alpha < 0.5 else end_row
                repl = dict(meta)
                repl["x"] = float(x)
                repl["y"] = float(y)
                repl["z"] = float(z)
                repl["yaw"] = float(yaw)
                repl["dist"] = float(new_dist)
                repl["sdist"] = float(new_dist)
                repl["assigned_lane_id"] = int(repl.get("assigned_lane_id", repl.get("lane_id", 0)))
                repl["continuity_bridge"] = True
                patch_rows.append((t, repl))

            if not feasible or len(patch_rows) < 2:
                i += 1
                continue

            local_slack = float(self.JUMP_STALL_COST_SLACK) + float(self.JUMP_STALL_PER_FRAME_SLACK) * float(len(patch_rows))
            if new_cost > old_cost + local_slack:
                i += 1
                continue
            if new_max > max(float(self.TURN_SYNTH_MAX_POINT_DIST), old_max + float(self.JUMP_STALL_MAX_POINT_DELTA)):
                i += 1
                continue

            for t, repl in patch_rows:
                smoothed[t] = repl

            if self.verbose:
                print(
                    f"    [SMOOTH] Jump-stall continuity bridge: "
                    f"frames {i}-{end_idx}, stall_frames={stall_count}, "
                    f"cost {old_cost:.2f}->{new_cost:.2f}, max {old_max:.2f}->{new_max:.2f}"
                )

            i = end_idx + 1

        return smoothed

        
    def _compute_yaw_changes(self, waypoints: List[Waypoint]) -> List[float]:
        """Compute yaw change magnitude between consecutive frames."""
        changes = [0.0]  # First frame has no change
        for i in range(1, len(waypoints)):
            yaw_diff = abs(waypoints[i].yaw - waypoints[i-1].yaw)
            # Normalize to [0, 180]
            if yaw_diff > 180:
                yaw_diff = 360 - yaw_diff
            changes.append(yaw_diff)
        return changes
    
    def _compute_raw_displacements(self, waypoints: List[Waypoint]) -> List[float]:
        """Compute cumulative displacement along raw trajectory."""
        displacements = [0.0]
        for i in range(1, len(waypoints)):
            dx = waypoints[i].x - waypoints[i-1].x
            dy = waypoints[i].y - waypoints[i-1].y
            displacements.append(displacements[-1] + math.hypot(dx, dy))
        return displacements
    
    def _normalize_yaw_diff(self, yaw1: float, yaw2: float) -> float:
        """Compute absolute yaw difference, normalized to [0, 180]."""
        diff = abs(yaw1 - yaw2)
        if diff > 180:
            diff = 360 - diff
        if self.BIDIRECTIONAL_YAW_MATCH:
            diff = min(diff, abs(180.0 - diff))
        return diff
    
    def _detect_trajectory_segments(self, waypoints: List[Waypoint], jump_threshold: float = 5.0) -> List[Tuple[int, int]]:
        """
        Detect continuous segments within a trajectory by finding teleportation jumps.
        
        Args:
            waypoints: List of waypoints
            jump_threshold: Distance threshold (meters) to consider as a jump/teleportation
            
        Returns:
            List of (start_idx, end_idx) for each continuous segment
        """
        if len(waypoints) < 2:
            return [(0, len(waypoints) - 1)] if waypoints else []
        
        segments: List[Tuple[int, int]] = []
        segment_start = 0
        
        for i in range(1, len(waypoints)):
            dx = waypoints[i].x - waypoints[i-1].x
            dy = waypoints[i].y - waypoints[i-1].y
            dist = math.hypot(dx, dy)
            
            if dist > jump_threshold:
                # Jump detected - end current segment
                if i - 1 >= segment_start:
                    segments.append((segment_start, i - 1))
                segment_start = i
                if self.verbose:
                    print(f"    [SEGMENT] Jump detected at frame {i}: {dist:.1f}m")
        
        # Add final segment
        if segment_start < len(waypoints):
            segments.append((segment_start, len(waypoints) - 1))
        
        return segments
    
    def align_trajectory(self, waypoints: List[Waypoint]) -> List[Optional[Dict[str, object]]]:
        """
        Align an entire trajectory to lanes using dynamic programming.
        
        Uses lane_id (actual lane number like 1, 2, -1, -2) to determine lane changes,
        NOT lane_index (map feature index). Road segment transitions (different road_id
        but same lane_id) are FREE.
        
        Handles trajectory jumps/teleportations by segmenting and aligning independently.
        
        Returns list of snap results (one per waypoint), or None for frames that can't be snapped.
        """
        n = len(waypoints)
        if n == 0:
            return []
        
        # Detect trajectory segments (handles teleportation jumps)
        segments = self._detect_trajectory_segments(waypoints)
        
        if self.verbose and len(segments) > 1:
            print(f"    [SEGMENT] Trajectory split into {len(segments)} segments")
        
        # Align each segment independently
        all_results: List[Optional[Dict[str, object]]] = [None] * n
        
        for seg_idx, (seg_start, seg_end) in enumerate(segments):
            seg_waypoints = waypoints[seg_start:seg_end + 1]
            if not seg_waypoints:
                continue
            
            if self.verbose and len(segments) > 1:
                print(f"    [SEGMENT] Aligning segment {seg_idx + 1}: frames {seg_start}-{seg_end} ({len(seg_waypoints)} pts)")
            
            # Align this segment
            seg_results = self._align_segment(seg_waypoints)
            
            # Copy results to correct positions
            for i, result in enumerate(seg_results):
                all_results[seg_start + i] = result
        
        return all_results
    
    def _align_segment(self, waypoints: List[Waypoint]) -> List[Optional[Dict[str, object]]]:
        """
        Align a single continuous trajectory segment to lanes using dynamic programming.
        
        This is the core DP logic, operating on a guaranteed-continuous segment.
        """
        n = len(waypoints)
        if n == 0:
            return []

        # Lane-change robustness knobs (env-tunable).
        lane_jump_ratio = _env_float("V2X_ALIGN_LANE_CHANGE_JUMP_RATIO", 2.4)
        lane_jump_abs_m = _env_float("V2X_ALIGN_LANE_CHANGE_JUMP_ABS_M", 1.8)
        lane_jump_raw_floor_m = _env_float("V2X_ALIGN_LANE_CHANGE_JUMP_RAW_FLOOR_M", 0.25)
        lane_jump_base_penalty = _env_float("V2X_ALIGN_LANE_CHANGE_JUMP_BASE_PENALTY", 170.0)
        lane_jump_per_m_penalty = _env_float("V2X_ALIGN_LANE_CHANGE_JUMP_PER_M_PENALTY", 90.0)
        weak_change_penalty = _env_float("V2X_ALIGN_LANE_CHANGE_WEAK_EVIDENCE_PENALTY", 120.0)
        weak_change_min_gain_m = _env_float("V2X_ALIGN_LANE_CHANGE_MIN_GAIN_M", 0.9)
        worse_change_per_m_penalty = _env_float("V2X_ALIGN_LANE_CHANGE_WORSE_DIST_PER_M_PENALTY", 110.0)
        worse_change_max_penalty = _env_float("V2X_ALIGN_LANE_CHANGE_WORSE_DIST_MAX_PENALTY", 650.0)
        early_change_frames = _env_int("V2X_ALIGN_EARLY_LANE_CHANGE_FRAMES", 18, minimum=0, maximum=80)
        early_change_penalty = _env_float("V2X_ALIGN_EARLY_LANE_CHANGE_EXTRA_PENALTY", 140.0)
        early_change_min_yaw_deg = _env_float("V2X_ALIGN_EARLY_LANE_CHANGE_MIN_YAW_DEG", 24.0)
        sign_flip_small_step_penalty = _env_float("V2X_ALIGN_SIGN_FLIP_SMALL_STEP_PENALTY", 260.0)
        sign_flip_max_raw_step_m = _env_float("V2X_ALIGN_SIGN_FLIP_MAX_RAW_STEP_M", 1.2)
        sign_flip_max_yaw_change_deg = _env_float("V2X_ALIGN_SIGN_FLIP_MAX_YAW_CHANGE_DEG", 18.0)
        lane_change_vec_err_ratio = _env_float("V2X_ALIGN_LANE_CHANGE_VEC_ERR_RATIO", 2.0)
        lane_change_vec_err_abs_m = _env_float("V2X_ALIGN_LANE_CHANGE_VEC_ERR_ABS_M", 1.0)
        lane_change_vec_err_raw_floor_m = _env_float("V2X_ALIGN_LANE_CHANGE_VEC_ERR_RAW_FLOOR_M", 0.20)
        lane_change_vec_err_base_penalty = _env_float("V2X_ALIGN_LANE_CHANGE_VEC_ERR_BASE_PENALTY", 140.0)
        lane_change_vec_err_per_m_penalty = _env_float("V2X_ALIGN_LANE_CHANGE_VEC_ERR_PER_M_PENALTY", 90.0)
        lane_change_horizon_frames = _env_int("V2X_ALIGN_LANE_CHANGE_HORIZON_FRAMES", 4, minimum=0, maximum=12)
        lane_change_horizon_min_shared = _env_int("V2X_ALIGN_LANE_CHANGE_HORIZON_MIN_SHARED", 2, minimum=1, maximum=8)
        lane_change_horizon_min_gain_m = _env_float("V2X_ALIGN_LANE_CHANGE_HORIZON_MIN_GAIN_M", 1.0)
        lane_change_horizon_penalty = _env_float("V2X_ALIGN_LANE_CHANGE_HORIZON_PENALTY", 120.0)
        same_lane_disconnected_margin_m = _env_float("V2X_ALIGN_SAME_LANE_DISCONNECTED_MARGIN_M", 0.80)
        same_lane_disconnected_scale = _env_float("V2X_ALIGN_SAME_LANE_DISCONNECTED_SCALE", 0.18)
        lane_change_jump_guard_min_gain_m = _env_float("V2X_ALIGN_LANE_CHANGE_JUMP_GUARD_MIN_GAIN_M", 0.55)
        lane_change_jump_guard_max_yaw_deg = _env_float("V2X_ALIGN_LANE_CHANGE_JUMP_GUARD_MAX_YAW_DEG", 24.0)
        lane_change_jump_guard_penalty = _env_float("V2X_ALIGN_LANE_CHANGE_JUMP_GUARD_PENALTY", 420.0)
        lane_change_jump_guard_hard_reject = (
            _env_int("V2X_ALIGN_LANE_CHANGE_JUMP_GUARD_HARD_REJECT", 1, minimum=0, maximum=1) == 1
        )
        opposite_switch_max_raw_yaw_change_deg = _env_float("V2X_ALIGN_OPPOSITE_SWITCH_MAX_RAW_YAW_DEG", 20.0)
        opposite_switch_min_lane_yaw_delta_deg = _env_float("V2X_ALIGN_OPPOSITE_SWITCH_MIN_LANE_YAW_DELTA_DEG", 150.0)
        opposite_switch_min_gain_m = _env_float("V2X_ALIGN_OPPOSITE_SWITCH_MIN_GAIN_M", 1.2)
        opposite_switch_penalty = _env_float("V2X_ALIGN_OPPOSITE_SWITCH_PENALTY", 700.0)
        opposite_switch_guard_enabled = (
            _env_int("V2X_ALIGN_OPPOSITE_SWITCH_GUARD_ENABLED", 0, minimum=0, maximum=1) == 1
        )
        opposite_switch_hard_reject = (
            _env_int("V2X_ALIGN_OPPOSITE_SWITCH_HARD_REJECT", 1, minimum=0, maximum=1) == 1
        )
        sign_flip_strict_min_gain_m = _env_float("V2X_ALIGN_SIGN_FLIP_STRICT_MIN_GAIN_M", 1.00)
        sign_flip_strict_max_yaw_deg = _env_float("V2X_ALIGN_SIGN_FLIP_STRICT_MAX_YAW_DEG", 26.0)
        sign_flip_strict_penalty = _env_float("V2X_ALIGN_SIGN_FLIP_STRICT_PENALTY", 520.0)
        disconnected_keep_gain_m = _env_float("V2X_ALIGN_DISCONNECTED_KEEP_GAIN_M", 0.8)
        disconnected_keep_full_gain_m = _env_float("V2X_ALIGN_DISCONNECTED_KEEP_FULL_GAIN_M", 1.8)
        disconnected_keep_scale = _env_float("V2X_ALIGN_DISCONNECTED_REDUCED_SCALE", 0.25)
        same_lane_road_switch_penalty = _env_float("V2X_ALIGN_SAME_LANE_ROAD_SWITCH_PENALTY", 18.0)
        same_lane_road_switch_weak_penalty = _env_float("V2X_ALIGN_SAME_LANE_ROAD_SWITCH_WEAK_PENALTY", 130.0)
        same_lane_road_switch_min_gain_m = _env_float("V2X_ALIGN_SAME_LANE_ROAD_SWITCH_MIN_GAIN_M", 0.9)
        same_lane_road_switch_small_step_penalty = _env_float("V2X_ALIGN_SAME_LANE_ROAD_SWITCH_SMALL_STEP_PENALTY", 90.0)
        same_lane_road_switch_small_step_m = _env_float("V2X_ALIGN_SAME_LANE_ROAD_SWITCH_SMALL_STEP_M", 1.20)
        same_lane_road_switch_small_yaw_deg = _env_float("V2X_ALIGN_SAME_LANE_ROAD_SWITCH_SMALL_YAW_DEG", 15.0)
        
        # Pre-compute per-frame data
        yaw_changes = self._compute_yaw_changes(waypoints)
        
        # Get lane candidates for each frame
        frame_candidates: List[List[Dict[str, object]]] = []
        for wp in waypoints:
            candidates = self.matcher.get_all_candidates(wp.x, wp.y, top_k=12)
            candidates = self._filter_candidates_by_lane_type(candidates)
            frame_candidates.append(candidates)

        # Best distance per lane_id per frame (for lane-change evidence checks).
        best_dist_by_lane_per_frame: List[Dict[int, float]] = []
        for candidates in frame_candidates:
            row: Dict[int, float] = {}
            for cand in candidates:
                lid = int(cand["lane_id"])
                d = float(cand["dist"])
                if lid not in row or d < float(row[lid]):
                    row[lid] = d
            best_dist_by_lane_per_frame.append(row)
        
        # Compute per-frame confidence for lane assignment
        confidences = self._compute_lane_confidence(frame_candidates)
        
        # Detect trajectory phases with dominant lane_ids
        phases = self._detect_trajectory_phases(frame_candidates, confidences)
        
        # Build frame->phase mapping and get dominant lane_id hints
        phase_lane_hints: Dict[int, int] = {}  # frame -> suggested lane_id from phase
        for phase in phases:
            if phase["dominant_lane_id"] is not None:
                for t in range(phase["start"], phase["end"] + 1):
                    phase_lane_hints[t] = phase["dominant_lane_id"]
        
        if n == 1:
            # Single frame - just return nearest
            if frame_candidates[0]:
                best = min(frame_candidates[0], key=lambda c: c["dist"])
                best["sdist"] = best["dist"]
                return [best]
            return [None]
        
        # Build lane_id set (the actual lane numbers, not indices)
        all_lane_ids: set = set()
        for candidates in frame_candidates:
            for c in candidates:
                all_lane_ids.add(c["lane_id"])
        lane_id_list = sorted(all_lane_ids)
        lane_id_to_idx = {lid: i for i, lid in enumerate(lane_id_list)}
        num_lane_ids = len(lane_id_list)
        
        if num_lane_ids == 0:
            return [None] * n
        
        # DP tables using lane_id as state
        # State: (current_lane_id_idx, prev_lane_id_idx) -> cost
        INF = float('inf')
        current_costs: Dict[Tuple[int, int], float] = {}
        backtrack: Dict[Tuple[int, Tuple[int, int]], Tuple[int, int]] = {}  # (t, state) -> prev_state
        # Also track which lane_index was used for each state
        lane_index_choice: Dict[Tuple[int, Tuple[int, int]], int] = {}  # (t, state) -> best lane_index
        # Track snapped XY chosen by DP state for transition jump penalties.
        snap_xy_choice: Dict[Tuple[int, Tuple[int, int]], Tuple[float, float]] = {}
        # Track snapped lane yaw for opposite-direction switch rejection.
        snap_yaw_choice: Dict[Tuple[int, Tuple[int, int]], float] = {}
        
        # Build a quick lookup from lane_index to its successors
        lane_successors = self.matcher.lane_successors
        
        # Initialize first frame (no previous lane)
        for c in frame_candidates[0]:
            lid = c["lane_id"]
            if lid not in lane_id_to_idx:
                continue
            lid_idx = lane_id_to_idx[lid]
            yaw_mismatch = self._normalize_yaw_diff(waypoints[0].yaw, c["yaw"])
            frame_cost = self.DIST_WEIGHT * c["dist"] + self.YAW_MISMATCH_WEIGHT * yaw_mismatch
            state = (lid_idx, -1)  # -1 = no previous
            if state not in current_costs or frame_cost < current_costs[state]:
                current_costs[state] = frame_cost
                lane_index_choice[(0, state)] = c["lane_index"]
                snap_xy_choice[(0, state)] = (float(c["x"]), float(c["y"]))
                snap_yaw_choice[(0, state)] = float(c["yaw"])
        
        # Forward pass
        for t in range(1, n):
            wp = waypoints[t]
            yaw_change = yaw_changes[t]
            raw_dx = float(waypoints[t].x) - float(waypoints[t - 1].x)
            raw_dy = float(waypoints[t].y) - float(waypoints[t - 1].y)
            raw_step = float(
                math.hypot(
                    float(raw_dx),
                    float(raw_dy),
                )
            )
            
            # Determine if this frame shows significant yaw change (potential lane change)
            allow_easy_lane_change = yaw_change > self.YAW_CHANGE_THRESHOLD
            
            next_costs: Dict[Tuple[int, int], float] = {}
            
            for c in frame_candidates[t]:
                to_lane_id = c["lane_id"]
                if to_lane_id not in lane_id_to_idx:
                    continue
                to_lid_idx = lane_id_to_idx[to_lane_id]
                yaw_mismatch = self._normalize_yaw_diff(wp.yaw, c["yaw"])
                
                # Base cost for this frame
                frame_cost = self.DIST_WEIGHT * c["dist"] + self.YAW_MISMATCH_WEIGHT * yaw_mismatch
                
                # Phase hint penalty: if this frame's phase has a dominant lane_id,
                # penalize other lane_ids proportionally to the confidence gap
                if t in phase_lane_hints:
                    hint_lane_id = phase_lane_hints[t]
                    if to_lane_id != hint_lane_id:
                        # Get confidence at this frame
                        _, confidence = confidences[t]
                        if confidence < 1.0:  # Low confidence frame
                            # Penalize deviation from phase hint
                            # Penalty is higher when surrounding confident frames strongly suggest another lane
                            frame_cost += self.LANE_CHANGE_PENALTY * 0.5
                
                # Find best previous state
                for (from_lid_idx, prev_prev_lid_idx), prev_cost in current_costs.items():
                    if prev_cost >= INF:
                        continue
                    
                    from_lane_id = lane_id_list[from_lid_idx]
                    
                    # Get the lane_index used at the previous frame for this state
                    prev_state = (from_lid_idx, prev_prev_lid_idx)
                    prev_lane_index = lane_index_choice.get((t-1, prev_state))
                    prev_snap_xy = snap_xy_choice.get((t - 1, prev_state))
                    prev_snap_yaw = snap_yaw_choice.get((t - 1, prev_state))
                    to_lane_index = int(c["lane_index"])
                    
                    # Compute transition cost based on lane_id change
                    transition_cost = 0.0

                    # Connectivity check: if this transition is disconnected while connected
                    # alternatives exist for the previous lane, penalize it heavily.
                    if prev_lane_index is not None:
                        successors = lane_successors.get(prev_lane_index, [])
                        is_connected = to_lane_index in successors
                        connected_dists = [
                            float(other_cand["dist"])
                            for other_cand in frame_candidates[t]
                            if other_cand["lane_index"] in successors
                        ]
                        has_connected_alternative = len(connected_dists) > 0
                        if has_connected_alternative and not is_connected:
                            best_connected_dist = min(connected_dists)
                            gain_vs_connected = float(best_connected_dist) - float(c["dist"])
                            # Keep same semantic lane through map/connectivity ambiguity when
                            # connected alternatives are only marginally better in fit.
                            if int(from_lane_id) == int(to_lane_id):
                                if gain_vs_connected >= -float(same_lane_disconnected_margin_m):
                                    transition_cost += self.DISCONNECTED_PENALTY * float(same_lane_disconnected_scale)
                                else:
                                    transition_cost += self.DISCONNECTED_PENALTY
                            else:
                                # If the disconnected option is much closer to raw geometry,
                                # soften the connectivity penalty so we do not force a far snap
                                # simply to preserve topological continuity.
                                if gain_vs_connected >= float(disconnected_keep_full_gain_m):
                                    transition_cost += self.DISCONNECTED_PENALTY * 0.05
                                elif gain_vs_connected >= float(disconnected_keep_gain_m):
                                    transition_cost += self.DISCONNECTED_PENALTY * float(disconnected_keep_scale)
                                else:
                                    transition_cost += self.DISCONNECTED_PENALTY
                        elif is_connected and from_lane_id != to_lane_id:
                            transition_cost += self.CONNECTIVITY_BONUS

                    if (
                        int(from_lane_id) == int(to_lane_id)
                        and prev_lane_index is not None
                        and 0 <= int(prev_lane_index) < len(self.matcher.map_data.lanes)
                    ):
                        from_road_id = int(self.matcher.map_data.lanes[int(prev_lane_index)].road_id)
                        to_road_id = int(c.get("road_id", from_road_id))
                        if int(from_road_id) != int(to_road_id):
                            transition_cost += float(same_lane_road_switch_penalty)
                            same_road_best_dist: Optional[float] = None
                            for other_cand in frame_candidates[t]:
                                if (
                                    int(other_cand.get("lane_id", 0)) == int(from_lane_id)
                                    and int(other_cand.get("road_id", -10**9)) == int(from_road_id)
                                ):
                                    od = float(other_cand.get("dist", float("inf")))
                                    if not math.isfinite(od):
                                        continue
                                    if same_road_best_dist is None or od < float(same_road_best_dist):
                                        same_road_best_dist = float(od)
                            if same_road_best_dist is not None:
                                switch_gain = float(same_road_best_dist) - float(c["dist"])
                                if switch_gain < float(same_lane_road_switch_min_gain_m):
                                    weak_scale = 1.0 - max(0.0, float(switch_gain)) / max(
                                        1e-3,
                                        float(same_lane_road_switch_min_gain_m),
                                    )
                                    transition_cost += float(same_lane_road_switch_weak_penalty) * float(weak_scale)
                                    if (
                                        float(raw_step) <= float(same_lane_road_switch_small_step_m)
                                        and float(yaw_change) <= float(same_lane_road_switch_small_yaw_deg)
                                    ):
                                        transition_cost += float(same_lane_road_switch_small_step_penalty) * float(weak_scale)

                    if from_lane_id != to_lane_id:
                        # Actual lane change (different lane_id)
                        if allow_easy_lane_change:
                            transition_cost += self.LANE_CHANGE_PENALTY * 0.3
                        else:
                            transition_cost += self.LANE_CHANGE_PENALTY

                        # Require geometric evidence for lane changes: if staying on the
                        # previous lane_id is nearly as good, heavily penalize switching.
                        lane_gain_m = float("nan")
                        same_lane_best_dist = best_dist_by_lane_per_frame[t].get(int(from_lane_id))
                        if same_lane_best_dist is not None:
                            lane_gain_m = float(same_lane_best_dist) - float(c["dist"])
                            if lane_gain_m < float(weak_change_min_gain_m):
                                weak_scale = 1.0 - max(0.0, float(lane_gain_m)) / max(1e-3, float(weak_change_min_gain_m))
                                transition_cost += float(weak_change_penalty) * float(weak_scale)
                                if int(t) <= int(early_change_frames) and float(yaw_change) < float(early_change_min_yaw_deg):
                                    transition_cost += float(early_change_penalty) * float(weak_scale)
                            # If switching is currently much farther than staying on
                            # the previous lane, penalize proportionally to the
                            # immediate distance deficit.
                            if float(lane_gain_m) < 0.0 and float(worse_change_per_m_penalty) > 0.0:
                                deficit = abs(float(lane_gain_m))
                                extra = float(worse_change_per_m_penalty) * float(deficit)
                                transition_cost += min(float(worse_change_max_penalty), float(extra))

                        # Require short-horizon evidence for lane changes: if lane B
                        # does not stay better than lane A over the next few frames,
                        # penalize this switch as likely flicker/noise.
                        if int(lane_change_horizon_frames) > 0:
                            horizon_end = min(n, int(t) + int(lane_change_horizon_frames) + 1)
                            horizon_gain = 0.0
                            horizon_shared = 0
                            for jj in range(int(t), int(horizon_end)):
                                row = best_dist_by_lane_per_frame[jj]
                                d_from = row.get(int(from_lane_id))
                                d_to = row.get(int(to_lane_id))
                                if d_from is None or d_to is None:
                                    continue
                                horizon_shared += 1
                                horizon_gain += float(d_from) - float(d_to)
                            if (
                                int(horizon_shared) >= int(lane_change_horizon_min_shared)
                                and float(horizon_gain) < float(lane_change_horizon_min_gain_m)
                            ):
                                deficit = float(lane_change_horizon_min_gain_m) - float(horizon_gain)
                                scale = min(
                                    2.0,
                                    float(deficit) / max(1e-3, float(lane_change_horizon_min_gain_m)),
                                )
                                transition_cost += float(lane_change_horizon_penalty) * float(scale)

                        # PING-PONG PENALTY: Check if we're returning to a lane_id we just left
                        if prev_prev_lid_idx >= 0:
                            prev_prev_lane_id = lane_id_list[prev_prev_lid_idx]
                            if to_lane_id == prev_prev_lane_id:
                                # A→B→A pattern detected - heavy penalty
                                transition_cost += self.QUICK_RETURN_PENALTY
                        
                        # LANE SKIP PENALTY: Penalize skipping lanes (e.g., L1 -> L-3 instead of L1 -> L-2)
                        lateral_dist = self._lane_lateral_distance(from_lane_id, to_lane_id)
                        if lateral_dist > 1:
                            # Skipping lanes - heavy penalty proportional to distance
                            skip_penalty = (lateral_dist - 1) * self.LANE_SKIP_PENALTY
                            transition_cost += skip_penalty
                            if self.verbose and skip_penalty > 0:
                                pass  # Logged at end if chosen

                        # Sign-flip lane changes on tiny motion are usually semantic
                        # artifacts (e.g., +1 -> -2 at intersections). Penalize strongly.
                        if (
                            int(from_lane_id) * int(to_lane_id) < 0
                            and float(raw_step) <= float(sign_flip_max_raw_step_m)
                            and float(yaw_change) <= float(sign_flip_max_yaw_change_deg)
                        ):
                            transition_cost += float(sign_flip_small_step_penalty)
                            if (
                                math.isfinite(float(lane_gain_m))
                                and float(lane_gain_m) < float(sign_flip_strict_min_gain_m)
                                and float(yaw_change) <= float(sign_flip_strict_max_yaw_deg)
                            ):
                                strict_scale = 1.0 - max(0.0, float(lane_gain_m)) / max(1e-3, float(sign_flip_strict_min_gain_m))
                                transition_cost += float(sign_flip_strict_penalty) * float(strict_scale)
                        if bool(opposite_switch_guard_enabled) and prev_snap_yaw is not None:
                            lane_yaw_delta = float(_yaw_abs_diff_deg(float(prev_snap_yaw), float(c["yaw"])))
                            if (
                                lane_yaw_delta >= float(opposite_switch_min_lane_yaw_delta_deg)
                                and float(yaw_change) <= float(opposite_switch_max_raw_yaw_change_deg)
                                and (not math.isfinite(float(lane_gain_m)) or float(lane_gain_m) < float(opposite_switch_min_gain_m))
                            ):
                                if bool(opposite_switch_hard_reject):
                                    continue
                                gain_scale = 1.0
                                if math.isfinite(float(lane_gain_m)):
                                    gain_scale = 1.0 - max(0.0, float(lane_gain_m)) / max(1e-3, float(opposite_switch_min_gain_m))
                                transition_cost += float(opposite_switch_penalty) * float(max(0.2, gain_scale))

                        # Penalize lane switches whose snapped motion vector diverges
                        # strongly from observed raw per-frame motion.
                        lane_snap_step = float("nan")
                        if prev_snap_xy is not None:
                            sx_prev, sy_prev = prev_snap_xy
                            snap_dx = float(c["x"]) - float(sx_prev)
                            snap_dy = float(c["y"]) - float(sy_prev)
                            lane_snap_step = float(math.hypot(float(snap_dx), float(snap_dy)))
                            vec_err = float(math.hypot(float(snap_dx) - float(raw_dx), float(snap_dy) - float(raw_dy)))
                            vec_thr = max(
                                float(lane_change_vec_err_abs_m),
                                float(lane_change_vec_err_ratio) * max(float(lane_change_vec_err_raw_floor_m), float(raw_step)),
                            )
                            if vec_err > float(vec_thr):
                                excess = float(vec_err) - float(vec_thr)
                                transition_cost += float(lane_change_vec_err_base_penalty) + float(lane_change_vec_err_per_m_penalty) * float(excess)
                        if math.isfinite(float(lane_snap_step)):
                            step_thr_guard = max(
                                float(lane_jump_abs_m),
                                float(lane_jump_ratio) * max(float(lane_jump_raw_floor_m), float(raw_step)),
                            )
                            if (
                                float(lane_snap_step) > float(step_thr_guard)
                                and float(yaw_change) <= float(lane_change_jump_guard_max_yaw_deg)
                                and (not math.isfinite(float(lane_gain_m)) or float(lane_gain_m) < float(lane_change_jump_guard_min_gain_m))
                            ):
                                if bool(lane_change_jump_guard_hard_reject):
                                    continue
                                gain_scale = 1.0
                                if math.isfinite(float(lane_gain_m)):
                                    gain_scale = 1.0 - max(0.0, float(lane_gain_m)) / max(1e-3, float(lane_change_jump_guard_min_gain_m))
                                transition_cost += float(lane_change_jump_guard_penalty) * float(max(0.25, gain_scale))
                    # else: same lane_id, different road_id = FREE (no penalty)

                    # Strongly penalize lane/segment switches that induce a snapped XY jump
                    # far larger than observed raw motion.
                    if (
                        prev_snap_xy is not None
                        and prev_lane_index is not None
                        and int(to_lane_index) != int(prev_lane_index)
                    ):
                        sx_prev, sy_prev = prev_snap_xy
                        snap_step = float(
                            math.hypot(float(c["x"]) - float(sx_prev), float(c["y"]) - float(sy_prev))
                        )
                        step_thr = max(
                            float(lane_jump_abs_m),
                            float(lane_jump_ratio) * max(float(lane_jump_raw_floor_m), float(raw_step)),
                        )
                        if snap_step > float(step_thr):
                            excess = float(snap_step) - float(step_thr)
                            transition_cost += float(lane_jump_base_penalty) + float(lane_jump_per_m_penalty) * float(excess)
                    
                    total = prev_cost + transition_cost + frame_cost
                    new_state = (to_lid_idx, from_lid_idx)
                    
                    if new_state not in next_costs or total < next_costs[new_state]:
                        next_costs[new_state] = total
                        backtrack[(t, new_state)] = (from_lid_idx, prev_prev_lid_idx)
                        lane_index_choice[(t, new_state)] = c["lane_index"]
                        snap_xy_choice[(t, new_state)] = (float(c["x"]), float(c["y"]))
                        snap_yaw_choice[(t, new_state)] = float(c["yaw"])
            
            current_costs = next_costs
        
        # Backward pass: find best ending state
        if not current_costs:
            return [None] * n
        
        best_state = min(current_costs.keys(), key=lambda s: current_costs[s])
        if current_costs[best_state] >= INF:
            return [None] * n
        
        # Reconstruct path of lane_id indices and lane_indices
        path_states: List[Tuple[int, int]] = [best_state]
        state = best_state
        for t in range(n-1, 0, -1):
            if (t, state) not in backtrack:
                break
            prev_state = backtrack[(t, state)]
            path_states.append(prev_state)
            state = prev_state
        path_states.reverse()
        
        # Fill gaps if path is incomplete
        while len(path_states) < n:
            path_states.insert(0, path_states[0] if path_states else (0, -1))
        
        # Log lane changes if verbose
        if self.verbose:
            lane_changes = []
            for t in range(1, len(path_states)):
                prev_lid = lane_id_list[path_states[t-1][0]]
                curr_lid = lane_id_list[path_states[t][0]]
                if prev_lid != curr_lid:
                    lane_changes.append((t, prev_lid, curr_lid))
            if lane_changes:
                print(f"    [ALIGN] {len(lane_changes)} lane changes detected:")
                for t, from_lid, to_lid in lane_changes[:10]:  # Limit output
                    print(f"      t={t}: lane_id {from_lid} -> {to_lid}")
                if len(lane_changes) > 10:
                    print(f"      ... and {len(lane_changes) - 10} more")
        
        # Build results
        results: List[Optional[Dict[str, object]]] = []
        
        for t in range(n):
            state_t = path_states[t]
            assigned_lane_id_idx = state_t[0]
            assigned_lane_id = lane_id_list[assigned_lane_id_idx]
            
            # Get the best lane_index that was chosen for this state, or find nearest with matching lane_id
            best_proj: Optional[Dict[str, object]] = None
            
            # First try to get the exact choice from DP
            if (t, state_t) in lane_index_choice:
                chosen_idx = lane_index_choice[(t, state_t)]
                for c in frame_candidates[t]:
                    if c["lane_index"] == chosen_idx:
                        best_proj = c
                        break
            
            # Fallback: find nearest candidate with matching lane_id
            if best_proj is None:
                matching = [c for c in frame_candidates[t] if c["lane_id"] == assigned_lane_id]
                if matching:
                    best_proj = min(matching, key=lambda c: c["dist"])
                elif frame_candidates[t]:
                    # No matching lane_id - use nearest overall
                    best_proj = min(frame_candidates[t], key=lambda c: c["dist"])
            
            if best_proj is not None:
                result = dict(best_proj)
                result["sdist"] = result["dist"]
                result["assigned_lane_id"] = assigned_lane_id
                results.append(result)
            else:
                results.append(None)
        
        # Post-processing: smooth out spurious lane dips
        results = self._smooth_spurious_dips(results, frame_candidates)
        # Keep this off by default: it can over-optimize approach lanes and
        # introduce large intersection snaps in noisy map-connectivity regions.
        if self.REFINE_MULTILANE_RUNS and _env_int("V2X_ALIGN_ENABLE_REFINE_MULTILANE_RUNS", 0, minimum=0, maximum=1) == 1:
            results = self._refine_multilane_road_runs(results, frame_candidates)
        if _env_int("V2X_ALIGN_ENABLE_LOCK_INTERSECTION_APPROACH", 1, minimum=0, maximum=1) == 1:
            results = self._lock_intersection_approach_lane(waypoints, results, frame_candidates)
        results = self._smooth_short_lane_flicker_runs(results, frame_candidates)
        results = self._smooth_edge_lane_spikes(results, frame_candidates)
        results = self._smooth_same_lane_road_flicker(results, frame_candidates)
        results = self._smooth_terminal_same_lane_road_tail(results, frame_candidates)
        results = self._smooth_same_lane_turn_connector_chains(results, frame_candidates)
        if _env_int("V2X_ALIGN_ENABLE_SMOOTH_SHORT_CONNECTOR_ROADS", 1, minimum=0, maximum=1) == 1:
            results = self._smooth_short_connector_roads(results, frame_candidates)
        if _env_int("V2X_ALIGN_ENABLE_SMOOTH_INTERSECTION_LANE_CHANGES", 1, minimum=0, maximum=1) == 1:
            results = self._smooth_intersection_lane_changes(waypoints, results, frame_candidates)
        if _env_int("V2X_ALIGN_ENABLE_DELAY_PREMATURE_LANE_CHANGES", 1, minimum=0, maximum=1) == 1:
            results = self._delay_premature_lane_changes(results, frame_candidates)
        if _env_int("V2X_ALIGN_ENABLE_SUPPRESS_WEAK_JUMP_LANE_CHANGES", 1, minimum=0, maximum=1) == 1:
            results = self._suppress_weak_jump_lane_changes(waypoints, results, frame_candidates)
        if _env_int("V2X_ALIGN_ENABLE_RETIME_TRANSITION_JUMPS", 1, minimum=0, maximum=1) == 1:
            results = self._retime_large_road_transition_jumps(waypoints, results, frame_candidates)
        if _env_int("V2X_ALIGN_ENABLE_INJECT_TURN_SEGMENTS", 1, minimum=0, maximum=1) == 1:
            results = self._inject_intersection_turn_segments(waypoints, results, frame_candidates)
        if _env_int("V2X_ALIGN_ENABLE_DIRECT_TURN_BOUNDARIES", 1, minimum=0, maximum=1) == 1:
            results = self._inject_direct_intersection_turn_boundaries(waypoints, results, frame_candidates)
        if _env_int("V2X_ALIGN_ENABLE_SOFTEN_TRANSITION_JUMPS", 1, minimum=0, maximum=1) == 1:
            results = self._soften_alignment_transition_jumps(
                waypoints,
                results,
                jump_threshold_m=_env_float("V2X_ALIGN_SOFTEN_TRANSITION_JUMP_M", 1.6),
                jump_ratio_vs_raw=_env_float("V2X_ALIGN_SOFTEN_TRANSITION_JUMP_RATIO", 2.2),
                max_shift_m=_env_float("V2X_ALIGN_SOFTEN_TRANSITION_MAX_SHIFT_M", 1.2),
                max_query_cost_delta=_env_float("V2X_ALIGN_SOFTEN_TRANSITION_MAX_QUERY_DELTA", 1.0),
            )
        results = self._smooth_jump_stall_artifacts(waypoints, results)

        return results


def select_best_map(
    maps: Sequence[VectorMapData],
    ego_trajs: Sequence[Sequence[Waypoint]],
    vehicles: Dict[int, Sequence[Waypoint]],
    sample_count: int = 200,
    bbox_margin: float = 100.0,
) -> Tuple[VectorMapData, List[Dict[str, object]]]:
    """Select the best matching map based on trajectory coverage."""
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
        details.append({
            "name": map_data.name,
            "source_path": map_data.source_path,
            "sample_points": int(points_xy.shape[0]),
            "median_nearest_m": float(median_dist),
            "mean_nearest_m": float(mean_dist),
            "p90_nearest_m": float(p90_dist),
            "outside_bbox_ratio": float(outside_ratio),
            "score": float(score),
        })

    details.sort(key=lambda x: float(x["score"]))
    chosen_name = str(details[0]["name"])
    chosen_map = next(m for m in maps if m.name == chosen_name)
    return chosen_map, details
