"""
HUGSIM closed-loop metric helpers.

The HUGSIM driving score is:
    hug_score = route_completion * mean_epdm_score

where mean_epdm_score is the average over frame-level EPDM scores produced by
EPDMSScorer.score_frame_no_ep(...).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

try:
    from shapely.geometry import LineString, Point
except Exception:  # pragma: no cover - optional dependency guard
    LineString = None
    Point = None

try:
    from simulation.data_collection.epdms_scorer_md import EPDMSScorer
except Exception:  # pragma: no cover - optional dependency guard
    EPDMSScorer = None


def _empty_hugsim(reason: str) -> Dict[str, float | int | bool | str | None]:
    return {
        "available": False,
        "reason": str(reason),
        "hug_score": 0.0,
        "route_completion": 0.0,
        "mean_epdm_score": 0.0,
        "ego_distance_travelled": 0.0,
        "expert_route_distance": 0.0,
        "frame_count": 0,
    }


def _path_distance(path_xy: np.ndarray) -> float:
    if path_xy is None or len(path_xy) < 2:
        return 0.0
    deltas = np.diff(path_xy, axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


def _extract_expert_route_xy(config, ego_id: int) -> Optional[np.ndarray]:
    trajectory = None
    try:
        trajectory = config.trajectory[ego_id]
    except Exception:
        try:
            trajectory = config.trajectory
        except Exception:
            trajectory = None
    if trajectory is None:
        return None

    route_xy: List[List[float]] = []
    for location in trajectory:
        try:
            route_xy.append([float(location.x), float(location.y)])
        except Exception:
            continue
    if len(route_xy) < 2:
        return None
    return np.asarray(route_xy, dtype=np.float64)


class _RouteLane:
    """Minimal lane shim compatible with EPDMSScorer's lane queries."""

    def __init__(self, centerline_xy: np.ndarray, lane_width_m: float = 4.0):
        if LineString is None:
            raise RuntimeError("shapely is required for HUGSIM lane geometry.")
        self._line = LineString(centerline_xy.tolist())
        self.length = float(self._line.length)
        self.index = "hugsim_route_lane_0"
        self.shapely_polygon = self._line.buffer(
            lane_width_m / 2.0,
            cap_style=2,
            join_style=2,
        )

    def _heading_vec(self, s: float) -> np.ndarray:
        if self.length <= 1e-6:
            return np.array([1.0, 0.0], dtype=np.float64)
        eps = min(1.0, max(0.05, self.length * 1e-3))
        s0 = max(0.0, float(s) - eps)
        s1 = min(self.length, float(s) + eps)
        p0 = self._line.interpolate(s0)
        p1 = self._line.interpolate(s1)
        dx = float(p1.x - p0.x)
        dy = float(p1.y - p0.y)
        norm = math.hypot(dx, dy)
        if norm <= 1e-6:
            return np.array([1.0, 0.0], dtype=np.float64)
        return np.array([dx / norm, dy / norm], dtype=np.float64)

    def local_coordinates(self, point_xy: np.ndarray) -> tuple[float, float]:
        x = float(point_xy[0])
        y = float(point_xy[1])
        pt = Point(x, y)
        s = float(self._line.project(pt))
        proj = self._line.interpolate(s)
        heading_vec = self._heading_vec(s)
        dx = x - float(proj.x)
        dy = y - float(proj.y)
        # signed lateral offset against left-normal of heading
        lateral = dx * (-heading_vec[1]) + dy * heading_vec[0]
        return s, float(lateral)

    def heading_at(self, s: float) -> np.ndarray:
        return self._heading_vec(s)

    def distance(self, point_xy: np.ndarray) -> float:
        return float(self._line.distance(Point(float(point_xy[0]), float(point_xy[1]))))


class _RoadNetworkShim:
    def __init__(self, lanes: List[_RouteLane]):
        self._lanes = lanes

    def get_all_lanes(self) -> List[_RouteLane]:
        return list(self._lanes)


class _CurrentMapShim:
    def __init__(self, lanes: List[_RouteLane]):
        self.road_network = _RoadNetworkShim(lanes)


class _MapManagerShim:
    def __init__(self, current_map: _CurrentMapShim):
        self.current_map = current_map


class _EngineShim:
    def __init__(self, current_map: _CurrentMapShim):
        self.map_manager = _MapManagerShim(current_map)


class _AgentShim:
    def __init__(self, position_xy: np.ndarray):
        self.position = np.asarray(position_xy, dtype=np.float64)


class _EnvShim:
    def __init__(self, ego_start_xy: np.ndarray, lanes: List[_RouteLane]):
        current_map = _CurrentMapShim(lanes)
        self.engine = _EngineShim(current_map)
        self.agent = _AgentShim(ego_start_xy)


def _build_scenario_tracks(
    pdm_trace: List[Dict],
    pdm_world_trace: Optional[List[Dict]],
    sdc_id: str,
) -> Dict[str, Dict]:
    n = len(pdm_trace)
    tracks: Dict[str, Dict] = {}

    ego_pos = np.zeros((n, 3), dtype=np.float64)
    ego_heading = np.zeros(n, dtype=np.float64)
    ego_valid = np.ones(n, dtype=np.bool_)
    for idx, sample in enumerate(pdm_trace):
        ego_pos[idx, 0] = float(sample.get("x", 0.0))
        ego_pos[idx, 1] = float(sample.get("y", 0.0))
        ego_heading[idx] = float(sample.get("yaw", 0.0))

    tracks[sdc_id] = {
        "type": "VEHICLE",
        "state": {
            "position": ego_pos,
            "heading": ego_heading,
            "valid": ego_valid,
        },
    }

    if not pdm_world_trace:
        return tracks

    ego_actor_id = pdm_trace[0].get("actor_id") if pdm_trace else None
    actor_buffers: Dict[str, Dict] = {}

    for frame_idx in range(min(n, len(pdm_world_trace))):
        world_sample = pdm_world_trace[frame_idx] or {}
        for actor in world_sample.get("actors", []):
            actor_id = actor.get("id")
            if actor_id is None or actor_id == ego_actor_id:
                continue
            actor_key = str(actor_id)
            if actor_key not in actor_buffers:
                actor_type_raw = str(actor.get("type", "")).lower()
                if actor_type_raw.startswith("vehicle"):
                    actor_type = "VEHICLE"
                elif "cycl" in actor_type_raw:
                    actor_type = "CYCLIST"
                else:
                    actor_type = "PEDESTRIAN"
                actor_buffers[actor_key] = {
                    "type": actor_type,
                    "position": np.zeros((n, 3), dtype=np.float64),
                    "heading": np.zeros(n, dtype=np.float64),
                    "valid": np.zeros(n, dtype=np.bool_),
                }
            actor_buf = actor_buffers[actor_key]
            actor_buf["position"][frame_idx, 0] = float(actor.get("x", 0.0))
            actor_buf["position"][frame_idx, 1] = float(actor.get("y", 0.0))
            actor_buf["heading"][frame_idx] = float(actor.get("yaw", 0.0))
            actor_buf["valid"][frame_idx] = True

    for actor_key, actor_buf in actor_buffers.items():
        tracks[actor_key] = {
            "type": actor_buf["type"],
            "state": {
                "position": actor_buf["position"],
                "heading": actor_buf["heading"],
                "valid": actor_buf["valid"],
            },
        }
    return tracks


def compute_hugsim_route_metrics(
    pdm_trace: Optional[List[Dict]],
    pdm_world_trace: Optional[List[Dict]] = None,
    config=None,
    ego_id: int = 0,
    plan_horizon_steps: int = 8,
) -> Dict[str, float | int | bool | str | None]:
    """
    Compute per-route HUGSIM closed-loop driving score.
    """
    if EPDMSScorer is None:
        return _empty_hugsim("EPDMSScorer import failed.")
    if LineString is None or Point is None:
        return _empty_hugsim("shapely is unavailable.")
    if not pdm_trace or len(pdm_trace) < 2:
        return _empty_hugsim("Missing ego trace.")

    n = len(pdm_trace)
    if pdm_world_trace:
        n = min(n, len(pdm_world_trace))
    if n < 2:
        return _empty_hugsim("Trace too short.")

    pdm_trace = pdm_trace[:n]
    pdm_world_trace = pdm_world_trace[:n] if pdm_world_trace else None

    expert_route_xy = _extract_expert_route_xy(config, ego_id=ego_id)
    if expert_route_xy is None:
        return _empty_hugsim("Missing expert route trajectory.")

    ego_xy = np.asarray([[float(s.get("x", 0.0)), float(s.get("y", 0.0))] for s in pdm_trace], dtype=np.float64)
    if len(ego_xy) < 2:
        return _empty_hugsim("Ego trajectory too short.")

    times = np.asarray([float(s.get("t", 0.0)) for s in pdm_trace], dtype=np.float64)
    if len(times) > 1:
        dt = float(np.median(np.diff(times)))
    else:
        dt = 0.1
    if not np.isfinite(dt) or dt <= 1e-6:
        dt = 0.1

    try:
        lane = _RouteLane(expert_route_xy)
        env = _EnvShim(ego_start_xy=ego_xy[0], lanes=[lane])
        scenario_data = {
            "metadata": {
                "sdc_id": "ego",
                "timestep": dt,
            },
            "tracks": _build_scenario_tracks(
                pdm_trace=pdm_trace,
                pdm_world_trace=pdm_world_trace,
                sdc_id="ego",
            ),
            # Keep empty map states when lane-level red-light mapping is unavailable.
            "dynamic_map_states": {},
            "length": int(n),
        }
        scorer = EPDMSScorer(scenario_data, env)
    except Exception as exc:
        return _empty_hugsim(f"Failed to initialize EPDMSScorer: {exc}")

    gt_stride = max(1, int(getattr(scorer, "gt_stride", max(1, round(0.5 / dt)))))
    horizon_steps = max(1, int(plan_horizon_steps))

    frame_scores: List[float] = []
    for frame_idx in range(n):
        future_indices: List[int] = []
        for step in range(1, horizon_steps + 1):
            future_idx = frame_idx + step * gt_stride
            if future_idx >= n:
                break
            future_indices.append(future_idx)

        if not future_indices:
            frame_scores.append(0.0)
            continue

        plan_traj = ego_xy[future_indices]
        try:
            frame_result = scorer.score_frame_no_ep(plan_traj=plan_traj, frame_idx=frame_idx)
            if bool(frame_result.get("valid", True)):
                frame_scores.append(float(frame_result.get("score", 0.0)))
            else:
                frame_scores.append(0.0)
        except Exception:
            frame_scores.append(0.0)

    mean_epdm = float(np.mean(frame_scores)) if frame_scores else 0.0
    ego_distance = _path_distance(ego_xy)
    expert_distance = _path_distance(expert_route_xy)
    if expert_distance <= 1e-6:
        route_completion = 0.0
    else:
        route_completion = float(np.clip(ego_distance / expert_distance, 0.0, 1.0))

    hug_score = float(route_completion * mean_epdm)
    return {
        "available": True,
        "reason": None,
        "hug_score": hug_score,
        "route_completion": route_completion,
        "mean_epdm_score": mean_epdm,
        "ego_distance_travelled": ego_distance,
        "expert_route_distance": expert_distance,
        "frame_count": len(frame_scores),
    }

