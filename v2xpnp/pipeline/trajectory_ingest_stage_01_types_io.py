#!/usr/bin/env python3
"""
Convert a V2XPnP-style YAML sequence into CARLA leaderboard-ready XML routes.

Outputs:
  - ego_route.xml (optional) with the ego trajectory as role="ego".
  - actors/<name>.xml for every non-ego object, role="npc", snap_to_road="false".
  - actors_manifest.json describing all actors (route_id, model, speed, etc.).
  - Optional GIF visualizing the replay frames.

Usage (typical):
  python -m v2xpnp.pipeline.trajectory_ingest \\
      --scenario-dir /data2/marco/CoLMDriver/v2xpnp/Sample_Dataset/2023-03-17-16-12-12_3_0 \\
      --subdir -1 \\
      --town ucla_v2 \\
      --out-dir /data2/marco/CoLMDriver/v2xpnp/out_log_replay \\
      --gif

Notes:
  - All custom actors are marked snap_to_road="false" by default; override with --snap-to-road
    if you want snapping.
  - Apply global transforms with --tx/--ty/--tz and --yaw-deg to align to the CARLA map.
  - If --subdir is omitted (or set to "all") and multiple subfolders exist, all YAML subfolders
    are used for actor locations. Only non-negative subfolders are treated as ego vehicles;
    negative subfolders contribute actors only.
  - Use --spawn-viz with --xodr and/or --map-pkl/--use-carla-map to visualize spawn vs aligned points.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import random
import re
import socket
import subprocess
import sys
import queue
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import xml.etree.ElementTree as ET
import pickle  # optional; used for map caches
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required: pip install pyyaml") from exc

try:
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib import patches, transforms
    import imageio.v2 as imageio  # type: ignore
except Exception:  # pragma: no cover
    plt = None
    patches = None
    transforms = None
    imageio = None

# Optional CARLA client (only needed when --use-carla-map)
try:  # pragma: no cover
    import carla  # type: ignore
except Exception:
    carla = None


# ---------------------- Helpers ---------------------- #


def _lane_type_value(name: str):
    """Map a lane-type name (e.g. 'Driving') to a carla.LaneType enum value."""
    return getattr(carla.LaneType, name, None) if carla is not None else None


def _assert_carla_endpoint_reachable(host: str, port: int, timeout_s: float = 2.0) -> None:
    """
    Fail fast before invoking CARLA Python API calls that may segfault when no simulator is reachable.
    """
    try:
        with socket.create_connection((str(host), int(port)), timeout=float(timeout_s)):
            return
    except OSError as exc:
        raise RuntimeError(
            f"CARLA endpoint {host}:{int(port)} is not reachable: {exc}"
        ) from exc

def list_yaml_timesteps(folder: Path) -> List[Path]:
    """Return sorted YAML files in a folder (by stem as int if possible)."""
    files = [p for p in folder.iterdir() if p.suffix.lower() == ".yaml"]
    def sort_key(p: Path):
        try:
            return int(p.stem)
        except Exception:
            return p.stem
    return sorted(files, key=sort_key)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def yaw_from_angle(angle: Sequence[float] | None) -> float:
    """Dataset provides [roll?, yaw?, pitch?]; middle component works for provided samples."""
    if isinstance(angle, Sequence) and len(angle) >= 2:
        return float(angle[1])
    if isinstance(angle, Sequence) and angle:
        return float(angle[-1])
    return 0.0


def yaw_from_pose(pose: Sequence[float] | None) -> float:
    """true_ego_pose is [x, y, z, roll, yaw, pitch]."""
    if isinstance(pose, Sequence) and len(pose) >= 5:
        return float(pose[4])
    return 0.0


def apply_se2(point: Tuple[float, float], yaw_deg: float, tx: float, ty: float, flip_y: bool = False) -> Tuple[float, float]:
    """Rotate then translate (yaw in degrees), with optional Y flip."""
    rad = math.radians(yaw_deg)
    c, s = math.cos(rad), math.sin(rad)
    x, y = point
    if flip_y:
        y = -y
    xr = c * x - s * y + tx
    yr = s * x + c * y + ty
    return xr, yr


def invert_se2(point: Tuple[float, float], yaw_deg: float, tx: float, ty: float, flip_y: bool = False) -> Tuple[float, float]:
    """Inverse of apply_se2 (undo translation, rotation, and optional Y flip)."""
    x = point[0] - tx
    y = point[1] - ty
    rad = math.radians(-yaw_deg)
    c, s = math.cos(rad), math.sin(rad)
    xr = c * x - s * y
    yr = s * x + c * y
    if flip_y:
        yr = -yr
    return xr, yr


def euclid3(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    """3D euclidean distance (Python 3.7-compatible; math.dist is 3.8+)."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def is_vehicle_type(obj_type: str | None) -> bool:
    """Check if obj_type represents a vehicle (not trash, signs, etc.)."""
    if not obj_type:
        return True  # Default to vehicle
    ot = obj_type.lower()
    
    # Exclude non-vehicle, non-pedestrian types (static props)
    excluded_keywords = [
        "trash", "can", "barrel", "cone", "barrier",
        "sign", "pole", "light", "bench", "tree", "plant"
    ]
    
    for keyword in excluded_keywords:
        if keyword in ot:
            return False
    
    # Allow vehicles and pedestrians
    return True


def is_pedestrian_type(obj_type: str | None) -> bool:
    """Check if obj_type represents a pedestrian/walker."""
    if not obj_type:
        return False
    ot = obj_type.lower()
    
    pedestrian_keywords = ["pedestrian", "walker", "person", "people", "child"]

    for keyword in pedestrian_keywords:
        if keyword in ot:
            return True

    return False


# Walker blueprint policy: exclude child-sized models by default.
ADULT_WALKER_BLUEPRINTS = [
    "walker.pedestrian.0002",
    "walker.pedestrian.0003",
    "walker.pedestrian.0004",
    "walker.pedestrian.0005",
    "walker.pedestrian.0006",
    "walker.pedestrian.0007",
    "walker.pedestrian.0008",
]
CHILD_WALKER_BLUEPRINTS = {
    "walker.pedestrian.0001",
}
WALKER_MODEL_NEAR_DISTANCE_M = 1.8


def _is_child_walker_blueprint(bp_id: str) -> bool:
    return str(bp_id) in CHILD_WALKER_BLUEPRINTS


def _is_adult_walker_blueprint(bp_id: str) -> bool:
    return str(bp_id) in ADULT_WALKER_BLUEPRINTS


def _walker_model_index_seed(actor_id: int) -> int:
    """
    Stable integer seed for deterministic model fallback selection per actor id.
    """
    return (int(actor_id) * 1103515245 + 12345) & 0x7FFFFFFF


def _diversify_nearby_walker_models(
    actor_meta_by_id: Dict[int, Dict[str, object]],
    vehicles: Dict[int, List[Waypoint]],
    near_distance_m: float = WALKER_MODEL_NEAR_DISTANCE_M,
) -> Dict[str, int]:
    """
    Reduce same-model adjacency for walkers by reassigning nearby walkers to different adult models.
    Uses a deterministic greedy coloring-like pass over a proximity graph.
    """
    walker_ids: List[int] = []
    walker_xy: Dict[int, Tuple[float, float]] = {}
    raw_models: Dict[int, str] = {}

    for vid, meta in actor_meta_by_id.items():
        kind = str(meta.get("kind") or "")
        if not kind.startswith("walker"):
            continue
        traj = vehicles.get(vid) or []
        if not traj:
            continue
        first_wp = traj[0]
        walker_ids.append(int(vid))
        walker_xy[int(vid)] = (float(first_wp.x), float(first_wp.y))
        raw_models[int(vid)] = str(meta.get("model") or "").strip()

    if len(walker_ids) < 2:
        return {
            "walkers": len(walker_ids),
            "near_pairs": 0,
            "same_pairs_before": 0,
            "same_pairs_after": 0,
            "models_changed": 0,
        }

    neighbors: Dict[int, List[int]] = {vid: [] for vid in walker_ids}
    near_pairs = 0
    for idx, a in enumerate(walker_ids):
        ax, ay = walker_xy[a]
        for b in walker_ids[idx + 1 :]:
            bx, by = walker_xy[b]
            if math.hypot(ax - bx, ay - by) <= float(near_distance_m):
                neighbors[a].append(b)
                neighbors[b].append(a)
                near_pairs += 1
    neighbor_sets: Dict[int, set] = {vid: set(ids) for vid, ids in neighbors.items()}

    initial_models: Dict[int, str] = {}
    for vid in walker_ids:
        current = raw_models[vid]
        # Preserve child-sized blueprints — don't replace them with adult models.
        if _is_child_walker_blueprint(current):
            initial_models[vid] = current
            continue
        if _is_adult_walker_blueprint(current):
            initial_models[vid] = current
            continue
        fallback_idx = _walker_model_index_seed(vid) % len(ADULT_WALKER_BLUEPRINTS)
        initial_models[vid] = ADULT_WALKER_BLUEPRINTS[fallback_idx]

    def _same_model_pairs(models: Dict[int, str]) -> int:
        total = 0
        for i, a in enumerate(walker_ids):
            ma = models.get(a, "")
            for b in walker_ids[i + 1 :]:
                if models.get(b, "") != ma:
                    continue
                if b in neighbor_sets[a]:
                    total += 1
        return total

    same_pairs_before = _same_model_pairs(initial_models)

    order = sorted(walker_ids, key=lambda vid: (-len(neighbors[vid]), vid))
    assigned: Dict[int, str] = {}
    usage: Dict[str, int] = {model: 0 for model in ADULT_WALKER_BLUEPRINTS}

    for vid in order:
        original_model = initial_models[vid]
        # Never reassign child walker blueprints — keep them as-is.
        if _is_child_walker_blueprint(original_model):
            assigned[vid] = original_model
            continue
        neighbor_models = {assigned[nid] for nid in neighbors[vid] if nid in assigned}
        candidates = [m for m in ADULT_WALKER_BLUEPRINTS if m not in neighbor_models]
        if not candidates:
            candidates = list(ADULT_WALKER_BLUEPRINTS)

        def _score(model_id: str) -> Tuple[int, int, int, int]:
            same_neighbor = sum(1 for nid in neighbors[vid] if assigned.get(nid) == model_id)
            model_usage = usage.get(model_id, 0)
            change_penalty = 0 if model_id == original_model else 1
            stable_tie = (_walker_model_index_seed(vid) + ADULT_WALKER_BLUEPRINTS.index(model_id)) % 997
            return (same_neighbor, model_usage, change_penalty, stable_tie)

        best_model = min(candidates, key=_score)
        assigned[vid] = best_model
        usage[best_model] = usage.get(best_model, 0) + 1

    same_pairs_after = _same_model_pairs(assigned)
    models_changed = 0
    for vid in walker_ids:
        new_model = assigned.get(vid, initial_models[vid])
        prev_model = raw_models[vid]
        if prev_model != new_model:
            models_changed += 1
        actor_meta_by_id[vid]["model"] = new_model

    return {
        "walkers": len(walker_ids),
        "near_pairs": near_pairs,
        "same_pairs_before": same_pairs_before,
        "same_pairs_after": same_pairs_after,
        "models_changed": models_changed,
    }


def map_obj_type(obj_type: str | None, rng=None) -> str:
    """Map dataset obj_type to a CARLA 0.9.12 blueprint (vehicle or walker).

    Args:
        obj_type: The raw obj_type string from the dataset YAML.
        rng: Optional ``random.Random`` instance for deterministic selection.
             When None (default) the module-level ``random`` is used.
    """
    _pick = rng.choice if rng is not None else random.choice

    bus_blueprints = [
        "vehicle.volkswagen.t2",
        "vehicle.mitsubishi.fusorosa",
    ]

    truck_blueprints = [
        "vehicle.carlamotors.carlacola",
    ]

    firetruck_blueprints = [
        "vehicle.carlamotors.firetruck",
    ]

    van_blueprints = [
        "vehicle.mercedes.sprinter",
        "vehicle.volkswagen.t2",
    ]

    ambulance_blueprints = [
        "vehicle.ford.ambulance",
    ]

    police_blueprints = [
        "vehicle.dodge.charger_police",
        "vehicle.dodge.charger_police_2020",
    ]

    motorcycle_blueprints = [
        "vehicle.harley-davidson.low_rider",
        "vehicle.kawasaki.ninja",
        "vehicle.yamaha.yzf",
        "vehicle.vespa.zx125",
    ]

    bicycle_blueprints = [
        "vehicle.diamondback.century",
        "vehicle.gazelle.omafiets",
        "vehicle.bh.crossbike",
    ]

    suv_blueprints = [
        "vehicle.jeep.wrangler_rubicon",
        "vehicle.nissan.patrol",
        "vehicle.nissan.patrol_2021",
        "vehicle.lincoln.mkz_2020",
    ]

    # General car/sedan blueprints (default pool)
    car_blueprints = [
        "vehicle.tesla.model3",
        "vehicle.audi.a2",
        "vehicle.audi.tt",
        "vehicle.audi.etron",
        "vehicle.bmw.grandtourer",
        "vehicle.chevrolet.impala",
        "vehicle.citroen.c3",
        "vehicle.dodge.charger_2020",
        "vehicle.ford.crown",
        "vehicle.ford.mustang",
        "vehicle.lincoln.mkz_2017",
        "vehicle.mercedes.coupe",
        "vehicle.mercedes.coupe_2020",
        "vehicle.mini.cooper_s",
        "vehicle.mini.cooper_s_2021",
        "vehicle.nissan.micra",
        "vehicle.seat.leon",
        "vehicle.toyota.prius",
        "vehicle.volkswagen.t2",
    ]

    if not obj_type:
        return _pick(car_blueprints)
    ot = obj_type.lower()

    # Children — use child-sized walker blueprints
    if "child" in ot:
        return _pick(list(CHILD_WALKER_BLUEPRINTS))

    # Pedestrians/Walkers (adult)
    if is_pedestrian_type(obj_type):
        return _pick(ADULT_WALKER_BLUEPRINTS)

    # Buses and large vehicles
    if "bus" in ot:
        return _pick(bus_blueprints)

    # Trucks and vans
    if "truck" in ot:
        if "fire" in ot:
            return _pick(firetruck_blueprints)
        return _pick(truck_blueprints)
    if "van" in ot or "sprinter" in ot:
        return _pick(van_blueprints)

    # Long vehicles (articulated trucks, semis, etc.)
    if "long" in ot:
        return _pick(truck_blueprints + bus_blueprints)

    # Construction / utility carts — treat as vans
    if "construction" in ot or "cart" in ot:
        return _pick(van_blueprints)

    # Emergency vehicles
    if "ambulance" in ot:
        return _pick(ambulance_blueprints)
    if "police" in ot:
        return _pick(police_blueprints)

    # Motorcycles
    if "motor" in ot or "motorcycle" in ot:
        return _pick(motorcycle_blueprints)
    if "bike" in ot and "bicycle" not in ot:
        return _pick(motorcycle_blueprints)

    # Bicycles / scooter riders (with rider) — classified as cyclists
    if "bicycle" in ot or "cycl" in ot or "scooter" in ot:
        return _pick(bicycle_blueprints)

    # SUVs and larger cars
    if "suv" in ot or "jeep" in ot:
        return _pick(suv_blueprints)
    if "patrol" in ot:
        return _pick(suv_blueprints)

    # Sedans and cars (default category)
    return _pick(car_blueprints)


@dataclass
class Waypoint:
    x: float
    y: float
    z: float
    yaw: float
    pitch: float = 0.0
    roll: float = 0.0


# ---------------------- Spawn Preprocess ---------------------- #

@dataclass
class SpawnCandidate:
    dx: float
    dy: float
    source: str
    base_cost: float
    valid: bool = False
    reason: str | None = None
    spawn_loc: Optional[Tuple[float, float, float]] = None
    dz: float = 0.0
    z_source: Optional[str] = None
    align_stats: Optional[Dict[str, float]] = None


@dataclass
class BoxSample2D:
    cx: float
    cy: float
    corners: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]


def _path_distance(traj: List[Waypoint]) -> float:
    dist = 0.0
    for a, b in zip(traj, traj[1:]):
        dist += euclid3((a.x, a.y, a.z), (b.x, b.y, b.z))
    return dist


def _trajectory_motion_stats(
    traj: List[Waypoint],
    times: Optional[List[float]] = None,
    *,
    default_dt: float = 0.1,
) -> Dict[str, float]:
    if not traj:
        return {
            "path_dist": 0.0,
            "net_disp_xy": 0.0,
            "bbox_extent_xy": 0.0,
            "duration_s": 0.0,
            "avg_speed_mps": 0.0,
        }

    path_dist = 0.0
    for a, b in zip(traj, traj[1:]):
        path_dist += math.hypot(float(b.x) - float(a.x), float(b.y) - float(a.y))

    if len(traj) >= 2:
        net_disp_xy = math.hypot(
            float(traj[-1].x) - float(traj[0].x),
            float(traj[-1].y) - float(traj[0].y),
        )
    else:
        net_disp_xy = 0.0

    xs = [float(wp.x) for wp in traj]
    ys = [float(wp.y) for wp in traj]
    bbox_extent_xy = max(max(xs) - min(xs), max(ys) - min(ys)) if xs and ys else 0.0

    if times and len(times) == len(traj):
        try:
            t0 = float(times[0])
            t1 = float(times[-1])
            duration_s = max(0.0, t1 - t0)
        except Exception:
            duration_s = max(0.0, float(default_dt) * max(0, len(traj) - 1))
    else:
        duration_s = max(0.0, float(default_dt) * max(0, len(traj) - 1))

    avg_speed_mps = float(path_dist) / max(1e-6, float(duration_s))
    return {
        "path_dist": float(path_dist),
        "net_disp_xy": float(net_disp_xy),
        "bbox_extent_xy": float(bbox_extent_xy),
        "duration_s": float(duration_s),
        "avg_speed_mps": float(avg_speed_mps),
    }


def _is_heavy_vehicle_obj_type(obj_type_raw: str) -> bool:
    obj_lower = str(obj_type_raw or "").lower()
    heavy_keywords = ("truck", "bus", "trailer", "tractor")
    return any(token in obj_lower for token in heavy_keywords)


def _classify_actor_kind(
    traj: List[Waypoint],
    obj_type_raw: str,
    *,
    times: Optional[List[float]] = None,
    default_dt: float = 0.1,
    static_path_threshold: float = 1.2,
    static_net_disp_threshold: float = 0.8,
    static_bbox_extent_threshold: float = 0.9,
    static_avg_speed_threshold: float = 0.8,
    static_heavy_path_threshold: float = 8.0,
    static_heavy_bbox_extent_threshold: float = 1.2,
    static_heavy_avg_speed_threshold: float = 0.8,
) -> Tuple[str, bool, Dict[str, float]]:
    is_pedestrian = is_pedestrian_type(obj_type_raw)
    is_heavy_vehicle = _is_heavy_vehicle_obj_type(obj_type_raw)
    motion_stats = _trajectory_motion_stats(traj, times, default_dt=default_dt)
    if is_pedestrian:
        kind = "walker"
        if len(traj) <= 1:
            kind = "walker_static"
        elif motion_stats["path_dist"] < 0.5:
            kind = "walker_static"
        return kind, True, motion_stats

    kind = "npc"
    if len(traj) <= 1:
        kind = "static"
    elif (
        motion_stats["path_dist"] <= float(static_path_threshold)
        and motion_stats["net_disp_xy"] <= float(static_net_disp_threshold)
    ):
        kind = "static"
    elif (
        motion_stats["net_disp_xy"] <= float(static_net_disp_threshold)
        and motion_stats["bbox_extent_xy"] <= float(static_bbox_extent_threshold)
        and motion_stats["avg_speed_mps"] <= float(static_avg_speed_threshold)
    ):
        # Robust fallback for noisy lidar trajectories that jitter in place.
        kind = "static"
    elif (
        is_heavy_vehicle
        and motion_stats["net_disp_xy"] <= float(static_net_disp_threshold)
        and motion_stats["bbox_extent_xy"] <= float(static_heavy_bbox_extent_threshold)
        and motion_stats["avg_speed_mps"] <= float(static_heavy_avg_speed_threshold)
        and motion_stats["path_dist"] <= float(static_heavy_path_threshold)
    ):
        # Trucks/buses often have larger parked jitter envelopes; keep them static
        # when motion stays compact and start/end displacement is small.
        kind = "static"
    return kind, False, motion_stats


def _actor_radius(kind: str, length: Optional[float], width: Optional[float], model: str) -> float:
    if length is not None or width is not None:
        dim = max(float(length or 0.0), float(width or 0.0))
        if dim > 0.0:
            return max(0.2, 0.5 * dim)
    model_lower = str(model or "").lower()
    if kind.startswith("walker"):
        return 0.4
    if "bicycle" in model_lower or "cycl" in model_lower:
        return 0.9
    if kind == "static":
        return 0.6
    return 1.5


def _ensure_times(traj: List[Waypoint], times: List[float] | None, default_dt: float) -> List[float]:
    if times and len(times) == len(traj):
        return [float(t) for t in times]
    return [float(i) * float(default_dt) for i in range(len(traj))]


def _copy_waypoint(wp: Waypoint) -> Waypoint:
    return Waypoint(
        x=float(wp.x),
        y=float(wp.y),
        z=float(wp.z),
        yaw=float(wp.yaw),
        pitch=float(wp.pitch),
        roll=float(wp.roll),
    )


def _popcount(mask: int) -> int:
    """Python 3.8-compatible population count for non-negative integer bitmasks."""
    bit_count = getattr(mask, "bit_count", None)
    if callable(bit_count):
        return int(bit_count())
    value = int(mask)
    count = 0
    while value:
        value &= value - 1
        count += 1
    return count


def _distance_point_to_segment_xy(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    vx = float(bx) - float(ax)
    vy = float(by) - float(ay)
    wx = float(px) - float(ax)
    wy = float(py) - float(ay)
    seg_len_sq = vx * vx + vy * vy
    if seg_len_sq <= 1e-9:
        return math.hypot(wx, wy)
    u = (wx * vx + wy * vy) / seg_len_sq
    u = max(0.0, min(1.0, u))
    qx = float(ax) + u * vx
    qy = float(ay) + u * vy
    return math.hypot(float(px) - qx, float(py) - qy)


def _min_distance_point_to_traj_interval_xy(
    px: float,
    py: float,
    traj: List[Waypoint],
    times: List[float],
    interval_start: float,
    interval_end: float,
) -> float:
    """
    Minimum XY distance from a fixed point to an actor's piecewise-linear trajectory
    over [interval_start, interval_end]. Returns +inf if there is no overlap.
    """
    if not traj or not times:
        return float("inf")
    if interval_end < interval_start:
        return float("inf")

    if len(traj) == 1 or len(times) == 1:
        t0 = float(times[0])
        if interval_start <= t0 <= interval_end:
            wp0 = traj[0]
            return math.hypot(float(px) - float(wp0.x), float(py) - float(wp0.y))
        return float("inf")

    best = float("inf")
    for i in range(min(len(traj), len(times)) - 1):
        t0 = float(times[i])
        t1 = float(times[i + 1])
        lo = max(float(interval_start), min(t0, t1))
        hi = min(float(interval_end), max(t0, t1))
        if hi < lo:
            continue

        a = traj[i]
        b = traj[i + 1]
        if abs(t1 - t0) <= 1e-9:
            ax = float(a.x)
            ay = float(a.y)
            bx = float(b.x)
            by = float(b.y)
        else:
            alpha0 = (lo - t0) / (t1 - t0)
            alpha1 = (hi - t0) / (t1 - t0)
            ax = float(a.x) + (float(b.x) - float(a.x)) * alpha0
            ay = float(a.y) + (float(b.y) - float(a.y)) * alpha0
            bx = float(a.x) + (float(b.x) - float(a.x)) * alpha1
            by = float(a.y) + (float(b.y) - float(a.y)) * alpha1

        dist = _distance_point_to_segment_xy(px, py, ax, ay, bx, by)
        if dist < best:
            best = dist
            if best <= 1e-6:
                return best

    return best


def _connected_components_from_neighbors(
    nodes: List[int],
    neighbors: Dict[int, set[int]],
) -> List[List[int]]:
    node_set = set(int(n) for n in nodes)
    visited: set[int] = set()
    components: List[List[int]] = []
    for node in sorted(node_set):
        if node in visited:
            continue
        stack = [node]
        visited.add(node)
        comp: List[int] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in neighbors.get(cur, set()):
                nxt_i = int(nxt)
                if nxt_i in node_set and nxt_i not in visited:
                    visited.add(nxt_i)
                    stack.append(nxt_i)
        components.append(sorted(comp))
    return components


def _max_independent_set_component(
    component_nodes: List[int],
    neighbors: Dict[int, set[int]],
    start_times: Dict[int, float],
    max_calls: int = 2500000,
) -> Tuple[set[int], bool]:
    """
    Solve maximum independent set on one conflict component.
    Returns (selected_nodes, timed_out_flag).
    """
    n = len(component_nodes)
    if n <= 1:
        return set(component_nodes), False

    idx_of = {node: i for i, node in enumerate(component_nodes)}
    node_of = {i: node for node, i in idx_of.items()}
    nbr_mask: List[int] = [0 for _ in range(n)]
    for node in component_nodes:
        i = idx_of[node]
        mask = 0
        for nei in neighbors.get(node, set()):
            j = idx_of.get(int(nei))
            if j is not None and j != i:
                mask |= (1 << j)
        nbr_mask[i] = mask

    best_mask = 0
    best_size = 0
    calls = 0
    timed_out = False

    def choose_vertex(cand_mask: int) -> int:
        best_i = -1
        best_deg = -1
        m = cand_mask
        while m:
            lb = m & -m
            i = lb.bit_length() - 1
            deg = _popcount(nbr_mask[i] & cand_mask)
            if deg > best_deg:
                best_deg = deg
                best_i = i
            m ^= lb
        return best_i

    def dfs(cand_mask: int, cur_mask: int, cur_size: int) -> None:
        nonlocal best_mask, best_size, calls, timed_out
        if timed_out:
            return
        calls += 1
        if calls > max_calls:
            timed_out = True
            return
        if cand_mask == 0:
            if cur_size > best_size:
                best_size = cur_size
                best_mask = cur_mask
            return
        if cur_size + _popcount(cand_mask) <= best_size:
            return

        v = choose_vertex(cand_mask)
        if v < 0:
            if cur_size > best_size:
                best_size = cur_size
                best_mask = cur_mask
            return

        # Include branch
        dfs(
            cand_mask & ~((1 << v) | nbr_mask[v]),
            cur_mask | (1 << v),
            cur_size + 1,
        )
        # Exclude branch
        dfs(
            cand_mask & ~(1 << v),
            cur_mask,
            cur_size,
        )

    full_mask = (1 << n) - 1
    dfs(full_mask, 0, 0)

    if timed_out:
        remaining = set(component_nodes)
        greedy_selected: set[int] = set()
        while remaining:
            pick = min(
                remaining,
                key=lambda vid: (
                    len(neighbors.get(vid, set()) & remaining),
                    float(start_times.get(int(vid), 0.0)),
                    int(vid),
                ),
            )
            greedy_selected.add(int(pick))
            remaining.remove(int(pick))
            remaining -= (neighbors.get(int(pick), set()) & remaining)
        return greedy_selected, True

    selected: set[int] = set()
    m = best_mask
    while m:
        lb = m & -m
        i = lb.bit_length() - 1
        selected.add(int(node_of[i]))
        m ^= lb
    return selected, False


def _early_spawn_interval_is_clear(
    actor_id: int,
    spawn_time: float,
    start_time: float,
    *,
    ids: List[int],
    vehicles: Dict[int, List[Waypoint]],
    times_cache: Dict[int, List[float]],
    start_times: Dict[int, float],
    spawn_xy: Dict[int, Tuple[float, float]],
    radii: Dict[int, float],
    safety_margin: float,
    scheduled_early_spawns: Optional[Dict[int, float]] = None,
) -> Tuple[bool, Optional[Dict[str, object]]]:
    px, py = spawn_xy[int(actor_id)]
    rad = float(radii.get(int(actor_id), 1.0))
    spawn_time = max(0.0, float(spawn_time))
    start_time = float(start_time)

    for oid in ids:
        if int(oid) == int(actor_id):
            continue
        o_times = times_cache.get(int(oid)) or []
        o_traj = vehicles.get(int(oid)) or []
        if not o_times or not o_traj:
            continue

        overlap_start = max(spawn_time, float(o_times[0]))
        overlap_end = min(start_time, float(o_times[-1]))
        if overlap_end <= overlap_start + 1e-9:
            continue

        min_dist = _min_distance_point_to_traj_interval_xy(
            px,
            py,
            o_traj,
            o_times,
            overlap_start,
            overlap_end,
        )
        if math.isinf(min_dist):
            continue
        clearance = rad + float(radii.get(int(oid), 1.0)) + float(safety_margin)
        if min_dist < clearance:
            return False, {
                "type": "trajectory_conflict",
                "blocked_by": int(oid),
                "min_dist": float(min_dist),
                "clearance": float(clearance),
                "window": [float(overlap_start), float(overlap_end)],
            }

    if scheduled_early_spawns:
        for oid, o_spawn_time in scheduled_early_spawns.items():
            if int(oid) == int(actor_id):
                continue
            o_start = float(start_times.get(int(oid), 0.0))
            if float(o_spawn_time) >= o_start - 1e-6:
                continue

            overlap_start = max(spawn_time, float(o_spawn_time))
            overlap_end = min(start_time, o_start)
            if overlap_end <= overlap_start + 1e-9:
                continue

            clearance = rad + float(radii.get(int(oid), 1.0)) + float(safety_margin)
            dist = math.hypot(
                float(px) - float(spawn_xy[int(oid)][0]),
                float(py) - float(spawn_xy[int(oid)][1]),
            )
            if dist < clearance:
                return False, {
                    "type": "early_spawn_overlap",
                    "blocked_by": int(oid),
                    "min_dist": float(dist),
                    "clearance": float(clearance),
                    "window": [float(overlap_start), float(overlap_end)],
                }

    return True, None


def _earliest_dynamic_safe_spawn_time(
    actor_id: int,
    *,
    ids: List[int],
    vehicles: Dict[int, List[Waypoint]],
    times_cache: Dict[int, List[float]],
    start_times: Dict[int, float],
    spawn_xy: Dict[int, Tuple[float, float]],
    radii: Dict[int, float],
    safety_margin: float,
) -> Tuple[float, Optional[Dict[str, object]]]:
    start_time = float(start_times.get(int(actor_id), 0.0))
    if start_time <= 1e-6:
        return 0.0, None

    clear_at_zero, block = _early_spawn_interval_is_clear(
        int(actor_id),
        0.0,
        start_time,
        ids=ids,
        vehicles=vehicles,
        times_cache=times_cache,
        start_times=start_times,
        spawn_xy=spawn_xy,
        radii=radii,
        safety_margin=safety_margin,
        scheduled_early_spawns=None,
    )
    if clear_at_zero:
        return 0.0, None

    lo = 0.0
    hi = start_time
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        ok_mid, _ = _early_spawn_interval_is_clear(
            int(actor_id),
            mid,
            start_time,
            ids=ids,
            vehicles=vehicles,
            times_cache=times_cache,
            start_times=start_times,
            spawn_xy=spawn_xy,
            radii=radii,
            safety_margin=safety_margin,
            scheduled_early_spawns=None,
        )
        if ok_mid:
            hi = mid
        else:
            lo = mid
    return round(min(start_time, max(0.0, float(hi))), 6), block


def _maximize_safe_early_spawn_actors(
    vehicles: Dict[int, List[Waypoint]],
    vehicle_times: Dict[int, List[float]],
    actor_meta: Dict[int, Dict[str, object]],
    dt: float,
    safety_margin: float,
) -> Tuple[Dict[int, float], Dict[str, object]]:
    """
    For each late-detected actor, choose the earliest spawn time in [0, first_timestamp]
    that keeps the actor collision-free versus all other trajectories and previously
    scheduled early-spawn actors.
    """
    ids = sorted(int(vid) for vid, traj in vehicles.items() if traj and int(vid) in actor_meta)
    if not ids:
        return {}, {
            "candidates": 0,
            "already_at_t0": 0,
            "individually_safe": 0,
            "selected": 0,
            "adjusted": 0,
            "spawn_at_t0": 0,
            "dynamic_limited": 0,
            "static_conflict_adjustments": 0,
            "pair_conflicts": 0,
            "timed_out_components": 0,
            "blocked_examples": [],
            "static_conflict_examples": [],
            "selected_actor_ids": [],
        }

    safety_margin = max(0.0, float(safety_margin))
    times_cache: Dict[int, List[float]] = {}
    start_times: Dict[int, float] = {}
    spawn_xy: Dict[int, Tuple[float, float]] = {}
    radii: Dict[int, float] = {}

    for vid in ids:
        traj = vehicles[int(vid)]
        times = _ensure_times(traj, vehicle_times.get(int(vid)), dt)
        times_cache[int(vid)] = times
        start_times[int(vid)] = float(times[0]) if times else 0.0
        spawn_xy[int(vid)] = (float(traj[0].x), float(traj[0].y))
        meta = actor_meta.get(int(vid), {})
        radii[int(vid)] = float(
            _actor_radius(
                str(meta.get("kind", "npc")),
                meta.get("length"),
                meta.get("width"),
                str(meta.get("model", "")),
            )
        )

    already_at_t0 = [vid for vid in ids if float(start_times.get(vid, 0.0)) <= 1e-6]
    timing_blocker_ids = [
        vid for vid in ids if bool(actor_meta.get(int(vid), {}).get("timing_blocker", False))
    ]
    timing_blocker_set = set(int(v) for v in timing_blocker_ids)
    candidate_ids = [
        vid
        for vid in ids
        if float(start_times.get(vid, 0.0)) > 1e-6
        and int(vid) not in timing_blocker_set
    ]

    dynamic_earliest: Dict[int, float] = {}
    blocked_examples: List[Dict[str, object]] = []
    for vid in candidate_ids:
        earliest_t, block_info = _earliest_dynamic_safe_spawn_time(
            int(vid),
            ids=ids,
            vehicles=vehicles,
            times_cache=times_cache,
            start_times=start_times,
            spawn_xy=spawn_xy,
            radii=radii,
            safety_margin=safety_margin,
        )
        dynamic_earliest[int(vid)] = float(earliest_t)
        if block_info is not None and len(blocked_examples) < 25:
            blocked_examples.append({"actor_id": int(vid), **block_info})

    individually_safe = [
        vid for vid in candidate_ids if float(dynamic_earliest.get(int(vid), 1.0)) <= 1e-6
    ]
    scheduled_spawn_times: Dict[int, float] = {}
    static_conflict_adjustments = 0
    static_conflict_examples: List[Dict[str, object]] = []
    for vid in sorted(candidate_ids, key=lambda v: (float(start_times.get(int(v), 0.0)), int(v))):
        start_t = float(start_times.get(int(vid), 0.0))
        chosen_t = min(start_t, max(0.0, float(dynamic_earliest.get(int(vid), start_t))))

        if chosen_t < start_t - 1e-6:
            for oid in sorted(scheduled_spawn_times.keys(), key=lambda v: (float(start_times.get(int(v), 0.0)), int(v))):
                o_spawn_t = float(scheduled_spawn_times[int(oid)])
                o_start_t = float(start_times.get(int(oid), 0.0))
                if o_spawn_t >= o_start_t - 1e-6:
                    continue
                clearance = float(radii.get(int(vid), 1.0)) + float(radii.get(int(oid), 1.0)) + safety_margin
                dist = math.hypot(
                    float(spawn_xy[int(vid)][0]) - float(spawn_xy[int(oid)][0]),
                    float(spawn_xy[int(vid)][1]) - float(spawn_xy[int(oid)][1]),
                )
                if dist >= clearance:
                    continue

                overlap_start = max(chosen_t, o_spawn_t)
                overlap_end = min(start_t, o_start_t)
                if overlap_end <= overlap_start + 1e-9:
                    continue

                required_t = min(start_t, o_start_t)
                if required_t > chosen_t + 1e-9:
                    prev_t = chosen_t
                    chosen_t = required_t
                    static_conflict_adjustments += 1
                    if len(static_conflict_examples) < 25:
                        static_conflict_examples.append(
                            {
                                "actor_id": int(vid),
                                "blocked_by": int(oid),
                                "reason": "spawn_overlap",
                                "min_dist": float(dist),
                                "clearance": float(clearance),
                                "old_spawn_time": float(prev_t),
                                "new_spawn_time": float(chosen_t),
                            }
                        )

            ok_with_prior, static_block = _early_spawn_interval_is_clear(
                int(vid),
                chosen_t,
                start_t,
                ids=ids,
                vehicles=vehicles,
                times_cache=times_cache,
                start_times=start_times,
                spawn_xy=spawn_xy,
                radii=radii,
                safety_margin=safety_margin,
                scheduled_early_spawns=scheduled_spawn_times,
            )
            if not ok_with_prior and chosen_t < start_t - 1e-6:
                if static_block is not None and len(static_conflict_examples) < 25:
                    static_conflict_examples.append(
                        {
                            "actor_id": int(vid),
                            "fallback_to_original_time": True,
                            **static_block,
                        }
                    )
                chosen_t = start_t

        scheduled_spawn_times[int(vid)] = round(float(chosen_t), 6)

    selected_spawn_times: Dict[int, float] = {}
    advance_seconds: List[float] = []
    for vid in candidate_ids:
        start_t = float(start_times.get(int(vid), 0.0))
        chosen_t = float(scheduled_spawn_times.get(int(vid), start_t))
        if chosen_t < start_t - 1e-6:
            chosen_t = round(max(0.0, chosen_t), 6)
            selected_spawn_times[int(vid)] = chosen_t
            advance_seconds.append(start_t - chosen_t)

    spawn_at_t0 = sum(
        1 for t in selected_spawn_times.values() if float(t) <= 1e-6
    )
    dynamic_limited = sum(
        1 for vid in candidate_ids if float(dynamic_earliest.get(int(vid), 0.0)) > 1e-6
    )

    report = {
        "candidates": len(candidate_ids),
        "timing_blockers": len(timing_blocker_ids),
        "already_at_t0": len(already_at_t0),
        "individually_safe": len(individually_safe),
        "selected": len(selected_spawn_times),
        "adjusted": len(selected_spawn_times),
        "spawn_at_t0": int(spawn_at_t0),
        "dynamic_limited": int(dynamic_limited),
        "static_conflict_adjustments": int(static_conflict_adjustments),
        "pair_conflicts": int(static_conflict_adjustments),  # Backwards-compatible field.
        "timed_out_components": 0,
        "avg_advance_seconds": float(sum(advance_seconds) / len(advance_seconds))
        if advance_seconds
        else 0.0,
        "max_advance_seconds": float(max(advance_seconds)) if advance_seconds else 0.0,
        "blocked_examples": blocked_examples,
        "static_conflict_examples": static_conflict_examples,
        "selected_actor_ids": sorted(int(v) for v in selected_spawn_times.keys()),
        "planned_spawn_times": {
            str(int(vid)): float(scheduled_spawn_times.get(int(vid), float(start_times.get(int(vid), 0.0))))
            for vid in candidate_ids
        },
        "timing_blocker_ids": sorted(int(v) for v in timing_blocker_ids),
    }
    return selected_spawn_times, report


def _apply_early_spawn_time_overrides(
    vehicles: Dict[int, List[Waypoint]],
    vehicle_times: Dict[int, List[float]],
    early_spawn_times: Dict[int, float],
    dt: float,
) -> Tuple[Dict[int, List[Waypoint]], Dict[int, List[float]], List[int], Dict[int, float]]:
    out_trajs: Dict[int, List[Waypoint]] = {}
    out_times: Dict[int, List[float]] = {}
    adjusted_ids: List[int] = []
    applied_spawn_times: Dict[int, float] = {}

    for vid, traj in vehicles.items():
        copied_traj = [_copy_waypoint(wp) for wp in traj]
        times = _ensure_times(copied_traj, vehicle_times.get(int(vid)), dt)
        copied_times = [float(t) for t in times]

        target_spawn_t = early_spawn_times.get(int(vid))
        if target_spawn_t is not None and copied_traj and copied_times:
            start_t = float(copied_times[0])
            clipped_t = round(min(start_t, max(0.0, float(target_spawn_t))), 6)
            if clipped_t < start_t - 1e-6:
                copied_traj = [_copy_waypoint(copied_traj[0])] + copied_traj
                copied_times = [float(clipped_t)] + copied_times
                adjusted_ids.append(int(vid))
                applied_spawn_times[int(vid)] = float(clipped_t)

        out_trajs[int(vid)] = copied_traj
        out_times[int(vid)] = copied_times

    adjusted_ids.sort()
    return out_trajs, out_times, adjusted_ids, applied_spawn_times


def _maximize_safe_late_despawn_actors(
    vehicles: Dict[int, List[Waypoint]],
    vehicle_times: Dict[int, List[float]],
    actor_meta: Dict[int, Dict[str, object]],
    dt: float,
    safety_margin: float,
    hold_until_time: float,
) -> Tuple[set[int], Dict[str, object]]:
    """
    Select the maximum-cardinality set of actors that can stay spawned after their
    last timestamp until hold_until_time without interfering with other trajectories.
    """
    ids = sorted(int(vid) for vid, traj in vehicles.items() if traj and int(vid) in actor_meta)
    hold_until_time = float(hold_until_time)
    if not ids or hold_until_time <= 0.0:
        return set(), {
            "candidates": 0,
            "already_at_horizon": 0,
            "individually_safe": 0,
            "selected": 0,
            "pair_conflicts": 0,
            "timed_out_components": 0,
            "blocked_examples": [],
            "hold_until_time": hold_until_time,
        }

    safety_margin = max(0.0, float(safety_margin))
    times_cache: Dict[int, List[float]] = {}
    end_times: Dict[int, float] = {}
    end_xy: Dict[int, Tuple[float, float]] = {}
    radii: Dict[int, float] = {}

    for vid in ids:
        traj = vehicles[int(vid)]
        times = _ensure_times(traj, vehicle_times.get(int(vid)), dt)
        times_cache[int(vid)] = times
        end_times[int(vid)] = float(times[-1]) if times else 0.0
        end_xy[int(vid)] = (float(traj[-1].x), float(traj[-1].y))
        meta = actor_meta.get(int(vid), {})
        radii[int(vid)] = float(
            _actor_radius(
                str(meta.get("kind", "npc")),
                meta.get("length"),
                meta.get("width"),
                str(meta.get("model", "")),
            )
        )

    already_at_horizon = [
        vid for vid in ids if float(end_times.get(vid, 0.0)) >= hold_until_time - 1e-6
    ]
    timing_blocker_ids = [
        vid for vid in ids if bool(actor_meta.get(int(vid), {}).get("timing_blocker", False))
    ]
    timing_blocker_set = set(int(v) for v in timing_blocker_ids)
    candidate_ids = [
        vid
        for vid in ids
        if float(end_times.get(vid, 0.0)) < hold_until_time - 1e-6
        and int(vid) not in timing_blocker_set
    ]

    individually_safe: List[int] = []
    blocked_examples: List[Dict[str, object]] = []
    for vid in candidate_ids:
        px, py = end_xy[vid]
        end_t = float(end_times[vid])
        rad = float(radii[vid])
        safe = True
        block_info: Dict[str, object] | None = None

        for oid in ids:
            if oid == vid:
                continue
            o_times = times_cache.get(oid) or []
            o_traj = vehicles.get(oid) or []
            if not o_times or not o_traj:
                continue

            overlap_start = max(end_t, float(o_times[0]))
            overlap_end = min(hold_until_time, float(o_times[-1]))
            if overlap_end <= overlap_start + 1e-9:
                continue

            min_dist = _min_distance_point_to_traj_interval_xy(
                px,
                py,
                o_traj,
                o_times,
                overlap_start,
                overlap_end,
            )
            if math.isinf(min_dist):
                continue
            clearance = rad + float(radii.get(oid, 1.0)) + safety_margin
            if min_dist < clearance:
                safe = False
                block_info = {
                    "blocked_by": int(oid),
                    "min_dist": float(min_dist),
                    "clearance": float(clearance),
                    "window": [float(overlap_start), float(overlap_end)],
                }
                break

        if safe:
            individually_safe.append(int(vid))
        elif block_info is not None and len(blocked_examples) < 25:
            blocked_examples.append({"actor_id": int(vid), **block_info})

    neighbors: Dict[int, set[int]] = {vid: set() for vid in individually_safe}
    pair_conflicts = 0
    for i in range(len(individually_safe)):
        a = int(individually_safe[i])
        ax, ay = end_xy[a]
        for j in range(i + 1, len(individually_safe)):
            b = int(individually_safe[j])
            bx, by = end_xy[b]
            overlap_start = max(float(end_times[a]), float(end_times[b]))
            if hold_until_time <= overlap_start + 1e-9:
                continue
            clearance = float(radii[a]) + float(radii[b]) + safety_margin
            if math.hypot(ax - bx, ay - by) < clearance:
                neighbors[a].add(b)
                neighbors[b].add(a)
                pair_conflicts += 1

    components = _connected_components_from_neighbors(individually_safe, neighbors)
    selected: set[int] = set()
    timed_out_components = 0
    for comp in components:
        if not comp:
            continue
        if len(comp) == 1 and not neighbors.get(comp[0]):
            selected.add(int(comp[0]))
            continue
        chosen, timed_out = _max_independent_set_component(
            component_nodes=comp,
            neighbors=neighbors,
            start_times=end_times,
        )
        if timed_out:
            timed_out_components += 1
        selected.update(int(v) for v in chosen)

    report = {
        "candidates": len(candidate_ids),
        "timing_blockers": len(timing_blocker_ids),
        "already_at_horizon": len(already_at_horizon),
        "individually_safe": len(individually_safe),
        "selected": len(selected),
        "pair_conflicts": int(pair_conflicts),
        "timed_out_components": int(timed_out_components),
        "blocked_examples": blocked_examples,
        "selected_actor_ids": sorted(int(v) for v in selected),
        "timing_blocker_ids": sorted(int(v) for v in timing_blocker_ids),
        "hold_until_time": float(hold_until_time),
    }
    return selected, report


def _apply_late_despawn_time_overrides(
    vehicles: Dict[int, List[Waypoint]],
    vehicle_times: Dict[int, List[float]],
    selected_late_hold_ids: set[int],
    dt: float,
    hold_until_time: float,
) -> Tuple[Dict[int, List[Waypoint]], Dict[int, List[float]], List[int]]:
    out_trajs: Dict[int, List[Waypoint]] = {}
    out_times: Dict[int, List[float]] = {}
    adjusted_ids: List[int] = []
    hold_until_time = float(hold_until_time)

    for vid, traj in vehicles.items():
        copied_traj = [_copy_waypoint(wp) for wp in traj]
        times = _ensure_times(copied_traj, vehicle_times.get(int(vid)), dt)
        copied_times = [float(t) for t in times]

        if (
            int(vid) in selected_late_hold_ids
            and copied_traj
            and copied_times
            and float(copied_times[-1]) < hold_until_time - 1e-6
        ):
            copied_traj = copied_traj + [_copy_waypoint(copied_traj[-1])]
            copied_times = copied_times + [float(hold_until_time)]
            adjusted_ids.append(int(vid))

        out_trajs[int(vid)] = copied_traj
        out_times[int(vid)] = copied_times

    adjusted_ids.sort()
    return out_trajs, out_times, adjusted_ids


def _build_time_grid(all_times: List[float], sample_dt: float) -> List[float]:
    if not all_times:
        return [0.0]
    max_time = max(all_times)
    if max_time <= 0.0:
        return [0.0]
    sample_dt = max(0.05, float(sample_dt))
    n = int(math.floor(max_time / sample_dt)) + 1
    return [i * sample_dt for i in range(n + 1)]


def _lerp_yaw_deg(yaw0: float, yaw1: float, alpha: float) -> float:
    delta = (float(yaw1) - float(yaw0) + 180.0) % 360.0 - 180.0
    return float(yaw0) + float(alpha) * delta


def _sample_pose_xyyaw(
    traj: List[Waypoint],
    times: List[float],
    sample_times: List[float],
    always_active: bool,
) -> List[Optional[Tuple[float, float, float]]]:
    if not traj:
        return [None for _ in sample_times]
    if not times:
        times = [float(i) for i in range(len(traj))]
    if len(traj) == 1:
        pose = (float(traj[0].x), float(traj[0].y), float(traj[0].yaw))
        if always_active:
            return [pose for _ in sample_times]
        t0 = float(times[0])
        out: List[Optional[Tuple[float, float, float]]] = []
        for t in sample_times:
            out.append(pose if float(t) >= t0 - 1e-6 else None)
        return out

    out: List[Optional[Tuple[float, float, float]]] = []
    idx = 0
    last = len(times) - 1
    t_first = float(times[0])
    t_last = float(times[-1])
    for t in sample_times:
        tt = float(t)
        if tt < t_first - 1e-9:
            if always_active:
                wp0 = traj[0]
                out.append((float(wp0.x), float(wp0.y), float(wp0.yaw)))
            else:
                out.append(None)
            continue
        if tt > t_last + 1e-9:
            if always_active:
                wpn = traj[-1]
                out.append((float(wpn.x), float(wpn.y), float(wpn.yaw)))
            else:
                out.append(None)
            continue
        while idx + 1 < last and float(times[idx + 1]) < tt:
            idx += 1
        if idx + 1 >= len(times):
            wpn = traj[-1]
            out.append((float(wpn.x), float(wpn.y), float(wpn.yaw)))
            continue
        t0 = float(times[idx])
        t1 = float(times[idx + 1])
        wp0 = traj[idx]
        wp1 = traj[idx + 1]
        if t1 <= t0:
            alpha = 0.0
        else:
            alpha = (tt - t0) / (t1 - t0)
        alpha = max(0.0, min(1.0, float(alpha)))
        x = float(wp0.x) + (float(wp1.x) - float(wp0.x)) * alpha
        y = float(wp0.y) + (float(wp1.y) - float(wp0.y)) * alpha
        yaw = _lerp_yaw_deg(float(wp0.yaw), float(wp1.yaw), alpha)
        out.append((float(x), float(y), float(yaw)))
    return out


def _actor_bbox_dims(kind: str, length: Optional[float], width: Optional[float], model: str) -> Tuple[float, float]:
    base_len: Optional[float] = None
    base_wid: Optional[float] = None
    if length is not None:
        try:
            val = float(length)
            if val > 0.05:
                base_len = val
        except Exception:
            base_len = None
    if width is not None:
        try:
            val = float(width)
            if val > 0.05:
                base_wid = val
        except Exception:
            base_wid = None

    model_lower = str(model or "").lower()
    if kind.startswith("walker"):
        default_len, default_wid = 0.6, 0.6
    elif ("bicycle" in model_lower) or ("cycl" in model_lower):
        default_len, default_wid = 1.8, 0.6
    elif ("motor" in model_lower) or ("bike" in model_lower):
        default_len, default_wid = 2.1, 0.75
    elif ("bus" in model_lower) or ("sprinter" in model_lower):
        default_len, default_wid = 10.5, 2.55
    elif ("truck" in model_lower) or ("firetruck" in model_lower):
        default_len, default_wid = 8.0, 2.45
    elif ("van" in model_lower) or ("t2" in model_lower):
        default_len, default_wid = 5.9, 2.1
    else:
        default_len, default_wid = 4.8, 2.0

    if base_len is None and base_wid is None:
        return float(default_len), float(default_wid)
    if base_len is None:
        base_wid = float(base_wid or default_wid)
        base_len = max(1.0, 2.3 * base_wid)
    if base_wid is None:
        base_len = float(base_len or default_len)
        base_wid = max(0.6, 0.42 * base_len)
    return float(max(0.2, base_len)), float(max(0.2, base_wid))


def _oriented_box_corners_xy(
    cx: float,
    cy: float,
    yaw_deg: float,
    length: float,
    width: float,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    yaw_rad = math.radians(float(yaw_deg))
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    hl = 0.5 * float(length)
    hw = 0.5 * float(width)
    local = [
        (hl, hw),
        (hl, -hw),
        (-hl, -hw),
        (-hl, hw),
    ]
    corners = []
    for lx, ly in local:
        wx = float(cx) + lx * c - ly * s
        wy = float(cy) + lx * s + ly * c
        corners.append((float(wx), float(wy)))
    return (corners[0], corners[1], corners[2], corners[3])


def _sat_project(poly: Sequence[Tuple[float, float]], axis_x: float, axis_y: float) -> Tuple[float, float]:
    lo = None
    hi = None
    for px, py in poly:
        val = float(px) * float(axis_x) + float(py) * float(axis_y)
        if lo is None or val < lo:
            lo = val
        if hi is None or val > hi:
            hi = val
    return float(lo if lo is not None else 0.0), float(hi if hi is not None else 0.0)


def _boxes_intersect_sat(
    poly_a: Sequence[Tuple[float, float]],
    poly_b: Sequence[Tuple[float, float]],
) -> bool:
    for poly in (poly_a, poly_b):
        n = len(poly)
        for i in range(n):
            x0, y0 = poly[i]
            x1, y1 = poly[(i + 1) % n]
            ex = float(x1) - float(x0)
            ey = float(y1) - float(y0)
            ax = -ey
            ay = ex
            norm = math.hypot(ax, ay)
            if norm <= 1e-9:
                continue
            ax /= norm
            ay /= norm
            a0, a1 = _sat_project(poly_a, ax, ay)
            b0, b1 = _sat_project(poly_b, ax, ay)
            if a1 < b0 - 1e-6 or b1 < a0 - 1e-6:
                return False
    return True


def _sample_actor_boxes(
    traj: List[Waypoint],
    times: List[float],
    sample_times: List[float],
    *,
    always_active: bool,
    length: float,
    width: float,
) -> List[Optional[BoxSample2D]]:
    poses = _sample_pose_xyyaw(
        traj=traj,
        times=times,
        sample_times=sample_times,
        always_active=always_active,
    )
    out: List[Optional[BoxSample2D]] = []
    for pose in poses:
        if pose is None:
            out.append(None)
            continue
        x, y, yaw = pose
        corners = _oriented_box_corners_xy(
            cx=float(x),
            cy=float(y),
            yaw_deg=float(yaw),
            length=float(length),
            width=float(width),
        )
        out.append(BoxSample2D(cx=float(x), cy=float(y), corners=corners))
    return out


def _series_has_overlap(
    series_a: List[Optional[BoxSample2D]],
    series_b: List[Optional[BoxSample2D]],
    *,
    radius_a: float,
    radius_b: float,
    shift_a_dx: float = 0.0,
    shift_a_dy: float = 0.0,
) -> Tuple[bool, Optional[int]]:
    n = min(len(series_a), len(series_b))
    if n <= 0:
        return False, None
    max_center_dist = float(radius_a) + float(radius_b) + 0.25
    max_center_dist_sq = max_center_dist * max_center_dist
    shift_dx = float(shift_a_dx)
    shift_dy = float(shift_a_dy)
    needs_shift = abs(shift_dx) > 1e-9 or abs(shift_dy) > 1e-9
    for idx in range(n):
        a = series_a[idx]
        b = series_b[idx]
        if a is None or b is None:
            continue
        dx = (float(a.cx) + shift_dx) - float(b.cx)
        dy = (float(a.cy) + shift_dy) - float(b.cy)
        if dx * dx + dy * dy > max_center_dist_sq:
            continue
        if needs_shift:
            a_poly = tuple((float(px) + shift_dx, float(py) + shift_dy) for px, py in a.corners)
        else:
            a_poly = a.corners
        if _boxes_intersect_sat(a_poly, b.corners):
            return True, idx
    return False, None


def _apply_parked_vehicle_path_clearance(
    vehicles_aligned: Dict[int, List[Waypoint]],
    vehicle_times_aligned: Dict[int, List[float]],
    vehicles_original: Dict[int, List[Waypoint]],
    vehicle_times_original: Dict[int, List[float]],
    actor_meta: Dict[int, Dict[str, object]],
    args: argparse.Namespace,
) -> Tuple[Dict[int, List[Waypoint]], Dict[str, object]]:
    report: Dict[str, object] = {
        "enabled": bool(getattr(args, "parked_clearance", True)),
        "applied": False,
        "parked_candidates": 0,
        "moving_actors": 0,
        "moved_actors": 0,
        "unresolved_actors": 0,
        "total_shift_m": 0.0,
        "max_shift_m": 0.0,
        "actors": {},
    }
    if not bool(getattr(args, "parked_clearance", True)):
        report["reason"] = "disabled_by_flag"
        return vehicles_aligned, report

    parked_ids: List[int] = []
    moving_ids: List[int] = []
    for vid, meta in actor_meta.items():
        kind = str(meta.get("kind") or "")
        if kind == "static":
            parked_ids.append(int(vid))
        elif kind == "npc":
            moving_ids.append(int(vid))
    parked_ids.sort()
    moving_ids.sort()
    report["parked_candidates"] = len(parked_ids)
    report["moving_actors"] = len(moving_ids)
    if not parked_ids or not moving_ids:
        report["reason"] = "insufficient_actors"
        return vehicles_aligned, report

    sample_dt = max(0.05, float(getattr(args, "parked_clearance_sample_dt", 0.2)))
    max_shift = max(0.0, float(getattr(args, "parked_clearance_max_shift", 1.0)))
    shift_step = max(0.05, float(getattr(args, "parked_clearance_shift_step", 0.15)))
    angle_count = max(8, int(getattr(args, "parked_clearance_angle_count", 16)))
    default_dt = float(getattr(args, "dt", 0.1))

    all_ids = sorted(set(parked_ids + moving_ids))
    all_times: List[float] = []
    for vid in all_ids:
        traj_cur = vehicles_aligned.get(int(vid)) or []
        if traj_cur:
            all_times.extend(
                _ensure_times(traj_cur, vehicle_times_aligned.get(int(vid)), default_dt)
            )
        traj_org = vehicles_original.get(int(vid)) or []
        if traj_org:
            all_times.extend(
                _ensure_times(traj_org, vehicle_times_original.get(int(vid)), default_dt)
            )
    sample_times = _build_time_grid(all_times, sample_dt)
    if not sample_times:
        sample_times = [0.0]

    dims_by_id: Dict[int, Tuple[float, float]] = {}
    radii_by_id: Dict[int, float] = {}
    for vid in all_ids:
        meta = actor_meta.get(int(vid), {})
        length, width = _actor_bbox_dims(
            kind=str(meta.get("kind") or "npc"),
            length=meta.get("length"),
            width=meta.get("width"),
            model=str(meta.get("model") or ""),
        )
        dims_by_id[int(vid)] = (float(length), float(width))
        radii_by_id[int(vid)] = 0.5 * math.hypot(float(length), float(width))

    boxes_current: Dict[int, List[Optional[BoxSample2D]]] = {}
    boxes_original: Dict[int, List[Optional[BoxSample2D]]] = {}
    for vid in all_ids:
        meta = actor_meta.get(int(vid), {})
        kind = str(meta.get("kind") or "")
        always_active = kind in ("static", "walker_static")
        length, width = dims_by_id[int(vid)]

        traj_cur = vehicles_aligned.get(int(vid)) or []
        times_cur = _ensure_times(traj_cur, vehicle_times_aligned.get(int(vid)), default_dt)
        boxes_current[int(vid)] = _sample_actor_boxes(
            traj=traj_cur,
            times=times_cur,
            sample_times=sample_times,
            always_active=always_active,
            length=float(length),
            width=float(width),
        )

        traj_org = vehicles_original.get(int(vid)) or []
        times_org = _ensure_times(traj_org, vehicle_times_original.get(int(vid)), default_dt)
        boxes_original[int(vid)] = _sample_actor_boxes(
            traj=traj_org,
            times=times_org,
            sample_times=sample_times,
            always_active=always_active,
            length=float(length),
            width=float(width),
        )

    total_shift = 0.0
    max_shift_seen = 0.0
    moved_count = 0
    unresolved_count = 0

    for pid in parked_ids:
        parked_series_cur = boxes_current.get(int(pid)) or []
        parked_series_org = boxes_original.get(int(pid)) or []
        traj_cur = vehicles_aligned.get(int(pid)) or []
        if not parked_series_cur or not traj_cur:
            continue

        orig_overlap_by_mid: Dict[int, bool] = {}
        cur_overlap_by_mid: Dict[int, bool] = {}
        first_new_overlap_idx: Dict[int, int] = {}
        blockers: List[int] = []
        for mid in moving_ids:
            moving_cur = boxes_current.get(int(mid)) or []
            moving_org = boxes_original.get(int(mid)) or []
            if not moving_cur:
                continue
            orig_overlap, _ = _series_has_overlap(
                parked_series_org,
                moving_org,
                radius_a=float(radii_by_id.get(int(pid), 2.5)),
                radius_b=float(radii_by_id.get(int(mid), 2.5)),
            )
            cur_overlap, cur_idx = _series_has_overlap(
                parked_series_cur,
                moving_cur,
                radius_a=float(radii_by_id.get(int(pid), 2.5)),
                radius_b=float(radii_by_id.get(int(mid), 2.5)),
            )
            orig_overlap_by_mid[int(mid)] = bool(orig_overlap)
            cur_overlap_by_mid[int(mid)] = bool(cur_overlap)
            if cur_overlap and not orig_overlap:
                blockers.append(int(mid))
                if cur_idx is not None:
                    first_new_overlap_idx[int(mid)] = int(cur_idx)

        baseline_introduced = len(blockers)
        actor_entry: Dict[str, object] = {
            "baseline_introduced": int(baseline_introduced),
            "blockers": [int(v) for v in sorted(blockers)],
            "applied": False,
            "shift": [0.0, 0.0],
            "shift_m": 0.0,
        }
        if baseline_introduced <= 0:
            report["actors"][str(pid)] = actor_entry
            continue

        candidate_set: set[Tuple[int, int]] = {(0, 0)}
        if max_shift > 0.0:
            r = shift_step
            while r <= max_shift + 1e-9:
                for ai in range(angle_count):
                    theta = 2.0 * math.pi * float(ai) / float(angle_count)
                    dx = r * math.cos(theta)
                    dy = r * math.sin(theta)
                    candidate_set.add((int(round(dx * 1000.0)), int(round(dy * 1000.0))))
                r += shift_step
            for mid in blockers:
                idx = first_new_overlap_idx.get(int(mid))
                if idx is None or idx < 0 or idx >= len(parked_series_cur):
                    continue
                p_sample = parked_series_cur[idx]
                m_sample = (boxes_current.get(int(mid)) or [None] * len(parked_series_cur))[idx]
                if p_sample is None or m_sample is None:
                    continue
                vx = float(p_sample.cx) - float(m_sample.cx)
                vy = float(p_sample.cy) - float(m_sample.cy)
                norm = math.hypot(vx, vy)
                if norm <= 1e-6:
                    yaw_ref = float(traj_cur[0].yaw) if traj_cur else 0.0
                    vx = math.cos(math.radians(yaw_ref + 90.0))
                    vy = math.sin(math.radians(yaw_ref + 90.0))
                    norm = math.hypot(vx, vy)
                if norm <= 1e-6:
                    continue
                ux = vx / norm
                uy = vy / norm
                px = -uy
                py = ux
                for dist in (shift_step, min(max_shift, shift_step * 2.0), max_shift):
                    if dist <= 1e-6:
                        continue
                    for sx, sy in (
                        (ux, uy),
                        (px, py),
                        (-px, -py),
                    ):
                        candidate_set.add(
                            (
                                int(round(float(sx) * float(dist) * 1000.0)),
                                int(round(float(sy) * float(dist) * 1000.0)),
                            )
                        )

        candidates = [
            (float(ix) / 1000.0, float(iy) / 1000.0)
            for ix, iy in sorted(
                candidate_set,
                key=lambda item: math.hypot(float(item[0]), float(item[1])),
            )
        ]

        best_dx = 0.0
        best_dy = 0.0
        best_score = (baseline_introduced, baseline_introduced, 0.0)
        for cand_dx, cand_dy in candidates:
            introduced = 0
            overlap_total = 0
            for mid in moving_ids:
                moving_cur = boxes_current.get(int(mid)) or []
                if not moving_cur:
                    continue
                has_overlap, _ = _series_has_overlap(
                    parked_series_cur,
                    moving_cur,
                    radius_a=float(radii_by_id.get(int(pid), 2.5)),
                    radius_b=float(radii_by_id.get(int(mid), 2.5)),
                    shift_a_dx=float(cand_dx),
                    shift_a_dy=float(cand_dy),
                )
                if has_overlap:
                    overlap_total += 1
                    if not bool(orig_overlap_by_mid.get(int(mid), False)):
                        introduced += 1
            cand_mag = math.hypot(float(cand_dx), float(cand_dy))
            cand_score = (int(introduced), int(overlap_total), float(cand_mag))
            if cand_score < best_score:
                best_score = cand_score
                best_dx = float(cand_dx)
                best_dy = float(cand_dy)
                if introduced == 0:
                    # Keep searching same-radius options, but prefer early exact clears.
                    if overlap_total <= 0 and cand_mag <= max(shift_step, 0.10) + 1e-9:
                        break

        introduced_after = int(best_score[0])
        if introduced_after < baseline_introduced and (abs(best_dx) > 1e-9 or abs(best_dy) > 1e-9):
            for wp in traj_cur:
                wp.x = float(wp.x) + float(best_dx)
                wp.y = float(wp.y) + float(best_dy)
            boxes_current[int(pid)] = [
                None
                if sample is None
                else BoxSample2D(
                    cx=float(sample.cx) + float(best_dx),
                    cy=float(sample.cy) + float(best_dy),
                    corners=tuple(
                        (float(px) + float(best_dx), float(py) + float(best_dy))
                        for px, py in sample.corners
                    ),
                )
                for sample in parked_series_cur
            ]
            shift_mag = math.hypot(float(best_dx), float(best_dy))
            total_shift += float(shift_mag)
            max_shift_seen = max(max_shift_seen, float(shift_mag))
            moved_count += 1
            actor_entry["applied"] = True
            actor_entry["shift"] = [float(best_dx), float(best_dy)]
            actor_entry["shift_m"] = float(shift_mag)
            actor_entry["introduced_after"] = int(introduced_after)
            actor_entry["remaining_overlap_total"] = int(best_score[1])
        else:
            unresolved_count += 1
            actor_entry["introduced_after"] = int(introduced_after)
            actor_entry["remaining_overlap_total"] = int(best_score[1])
            actor_entry["reason"] = "no_improving_shift_found"

        report["actors"][str(pid)] = actor_entry

    report["applied"] = moved_count > 0
    report["moved_actors"] = int(moved_count)
    report["unresolved_actors"] = int(unresolved_count)
    report["total_shift_m"] = float(total_shift)
    report["max_shift_m"] = float(max_shift_seen)

    if moved_count > 0 or unresolved_count > 0:
        avg_shift = float(total_shift) / float(max(1, moved_count))
        print(
            "[PARKED_CLEARANCE] parked={} moving={} moved={} unresolved={} "
            "avg_shift={:.3f}m max_shift={:.3f}m".format(
                len(parked_ids),
                len(moving_ids),
                int(moved_count),
                int(unresolved_count),
                float(avg_shift),
                float(max_shift_seen),
            )
        )
    return vehicles_aligned, report


def _sample_positions(
    traj: List[Waypoint],
    times: List[float],
    sample_times: List[float],
    always_active: bool,
) -> List[Optional[Tuple[float, float]]]:
    if not traj:
        return [None for _ in sample_times]

    if len(traj) == 1 or always_active:
        pos = (traj[0].x, traj[0].y)
        return [pos for _ in sample_times]

    positions: List[Optional[Tuple[float, float]]] = []
    idx = 0
    last = len(times) - 1
    for t in sample_times:
        if t < times[0] or t > times[-1]:
            positions.append(None)
            continue
        while idx + 1 < last and times[idx + 1] < t:
            idx += 1
        if idx + 1 >= len(times):
            positions.append((traj[-1].x, traj[-1].y))
            continue
        t0 = times[idx]
        t1 = times[idx + 1]
        if t1 <= t0:
            alpha = 0.0
        else:
            alpha = (t - t0) / (t1 - t0)
        x = traj[idx].x + (traj[idx + 1].x - traj[idx].x) * alpha
        y = traj[idx].y + (traj[idx + 1].y - traj[idx].y) * alpha
        positions.append((x, y))
    return positions


def _resolve_ground_z(world, location) -> Optional[float]:
    if world is None:
        return None
    ground_projection = getattr(world, "ground_projection", None)
    if callable(ground_projection):
        try:
            probe = carla.Location(
                x=location.x,
                y=location.y,
                z=location.z + 50.0,
            )
            result = ground_projection(probe, 100.0)
            if result is not None:
                if hasattr(result, "z"):
                    return float(result.z)
                if isinstance(result, (tuple, list)) and result:
                    first = result[0]
                    if hasattr(first, "z"):
                        return float(first.z)
        except Exception:
            pass
    cast_ray = getattr(world, "cast_ray", None)
    if callable(cast_ray):
        try:
            start = carla.Location(
                x=location.x,
                y=location.y,
                z=location.z + 50.0,
            )
            end = carla.Location(
                x=location.x,
                y=location.y,
                z=location.z - 50.0,
            )
            hits = cast_ray(start, end)
            if hits:
                best_z = None
                best_dist = None
                for hit in hits:
                    hit_loc = getattr(hit, "location", None) or getattr(hit, "point", None)
                    if hit_loc is None:
                        continue
                    dz = abs(float(hit_loc.z) - float(location.z))
                    if best_dist is None or dz < best_dist:
                        best_dist = dz
                        best_z = float(hit_loc.z)
                if best_z is not None:
                    return best_z
        except Exception:
            pass
    return None


def _candidate_z_offsets(
    world,
    world_map,
    base_loc,
    normalize_z: bool,
) -> List[Tuple[float, str]]:
    """Return (dz, source) candidates ordered by |dz|, starting with authored z."""
    offsets: List[Tuple[float, str]] = [(0.0, "authored")]
    if not normalize_z:
        return offsets

    candidates: List[Tuple[str, float]] = []
    ground_z = _resolve_ground_z(world, base_loc) if world is not None else None
    if ground_z is not None:
        candidates.append(("ground", float(ground_z)))
    if world_map is not None:
        try:
            wp_any = world_map.get_waypoint(
                base_loc,
                project_to_road=True,
                lane_type=carla.LaneType.Any,
            )
        except Exception:
            wp_any = None
        if wp_any is not None:
            candidates.append(("waypoint_any", float(wp_any.transform.location.z)))

    orig_z = float(base_loc.z)
    for label, z in candidates:
        offsets.append((float(z) - orig_z, label))

    uniq: List[Tuple[float, str]] = []
    seen = set()
    for dz, label in offsets:
        key = round(float(dz), 3)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((float(dz), label))

    uniq.sort(key=lambda it: abs(it[0]))
    return uniq


def _try_spawn_candidate(
    world,
    world_map,
    blueprint,
    base_wp: Waypoint,
    cand: SpawnCandidate,
    normalize_z: bool,
) -> None:
    base_loc = carla.Location(
        x=base_wp.x + cand.dx,
        y=base_wp.y + cand.dy,
        z=base_wp.z,
    )
    z_offsets = _candidate_z_offsets(world, world_map, base_loc, normalize_z)

    first_loc = None
    first_dz = 0.0
    first_src = None
    last_exc: Optional[Exception] = None

    for dz, z_src in z_offsets:
        spawn_loc = carla.Location(
            x=base_loc.x,
            y=base_loc.y,
            z=base_loc.z + dz,
        )
        if first_loc is None:
            first_loc = spawn_loc
            first_dz = float(dz)
            first_src = z_src
        spawn_tf = carla.Transform(
            spawn_loc,
            carla.Rotation(pitch=base_wp.pitch, yaw=base_wp.yaw, roll=base_wp.roll),
        )
        actor = None
        try:
            actor = world.try_spawn_actor(blueprint, spawn_tf)
        except Exception as exc:
            last_exc = exc
            continue
        if actor is not None:
            cand.valid = True
            cand.spawn_loc = (float(spawn_loc.x), float(spawn_loc.y), float(spawn_loc.z))
            cand.dz = float(dz)
            cand.z_source = z_src
            try:
                actor.destroy()
            except Exception:
                pass
            return

    cand.valid = False
    if first_loc is not None:
        cand.spawn_loc = (float(first_loc.x), float(first_loc.y), float(first_loc.z))
        cand.dz = float(first_dz)
        cand.z_source = first_src
    if last_exc is not None:
        cand.reason = f"spawn_exception: {last_exc}"
    else:
        cand.reason = "spawn_failed"


def _yaw_diff_deg(a: float, b: float) -> float:
    """Smallest absolute angle difference in degrees."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    mid = len(vals) // 2
    if len(vals) % 2 == 1:
        return float(vals[mid])
    return 0.5 * (float(vals[mid - 1]) + float(vals[mid]))


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0.0:
        return float(min(values))
    if q >= 1.0:
        return float(max(values))
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vals[lo])
    frac = pos - lo
    return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)


def _moving_average(values: List[float], window: int) -> List[float]:
    if not values:
        return []
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if window == 1:
        return [float(v) for v in values]
    half = window // 2
    out: List[float] = []
    n = len(values)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        seg = values[lo:hi]
        out.append(float(sum(seg) / max(1, len(seg))))
    return out


def _bridge_missing_or_transient_offsets(
    dxs: List[float],
    dys: List[float],
    valid_mask: List[bool],
    lane_keys: List[Optional[Tuple[int, int]]],
    yaws: List[float],
    max_gap_steps: int,
    straight_thresh_deg: float,
) -> Dict[str, int]:
    """
    Bridge short map-projection gaps and transient lane oscillations by interpolation.
    This avoids artificial lane hopping when CARLA map projections are locally sparse.
    """
    n = len(dxs)
    if n == 0:
        return {"bridged_missing": 0, "bridged_transient": 0}
    max_gap_steps = max(1, int(max_gap_steps))
    straight_thresh_deg = max(1.0, float(straight_thresh_deg))

    bridged_missing = 0
    bridged_transient = 0

    # 1) Bridge short missing spans.
    i = 0
    while i < n:
        if valid_mask[i]:
            i += 1
            continue
        start = i
        while i < n and not valid_mask[i]:
            i += 1
        end = i - 1
        gap_len = end - start + 1
        left = start - 1
        right = i
        if left < 0 or right >= n:
            continue
        if gap_len > max_gap_steps:
            continue
        if _yaw_diff_deg(float(yaws[left]), float(yaws[right])) > straight_thresh_deg:
            continue
        for idx in range(start, end + 1):
            alpha = float(idx - left) / float(right - left)
            dxs[idx] = float(dxs[left]) * (1.0 - alpha) + float(dxs[right]) * alpha
            dys[idx] = float(dys[left]) * (1.0 - alpha) + float(dys[right]) * alpha
            valid_mask[idx] = True
            if lane_keys[left] is not None and lane_keys[right] == lane_keys[left]:
                lane_keys[idx] = lane_keys[left]
        bridged_missing += gap_len

    # 2) Bridge short A->B->A transient lane runs on roughly straight headings.
    i = 1
    while i < n - 1:
        left_key = lane_keys[i - 1]
        run_key = lane_keys[i]
        if left_key is None or run_key is None or run_key == left_key:
            i += 1
            continue
        run_start = i
        j = i
        while j < n and lane_keys[j] == run_key:
            j += 1
        run_end = j - 1
        run_len = run_end - run_start + 1
        if j >= n:
            i = j
            continue
        right_key = lane_keys[j]
        if right_key is None or right_key != left_key or run_len > max_gap_steps:
            i = j
            continue
        if _yaw_diff_deg(float(yaws[run_start - 1]), float(yaws[j])) > straight_thresh_deg:
            i = j
            continue
        left = run_start - 1
        right = j
        for idx in range(run_start, run_end + 1):
            alpha = float(idx - left) / float(right - left)
            dxs[idx] = float(dxs[left]) * (1.0 - alpha) + float(dxs[right]) * alpha
            dys[idx] = float(dys[left]) * (1.0 - alpha) + float(dys[right]) * alpha
            lane_keys[idx] = left_key
            valid_mask[idx] = True
        bridged_transient += run_len
        i = j

    return {
        "bridged_missing": int(bridged_missing),
        "bridged_transient": int(bridged_transient),
    }


def _best_lane_projection_with_waypoint(
    world_map,
    loc,
    raw_yaw: float,
    candidate_lanes: List[str],
    prev_lane_key: Optional[Tuple[int, int]],
) -> Tuple[Optional[Dict[str, float]], Optional[Tuple[int, int]], Optional[str], object]:
    """Pick the best local map projection and return both projected info and CARLA waypoint."""
    best_info: Optional[Dict[str, float]] = None
    best_lane_key: Optional[Tuple[int, int]] = None
    best_lane_name: Optional[str] = None
    best_wp = None
    best_score: Optional[float] = None

    for lane_name in candidate_lanes:
        lane_val = _lane_type_value(lane_name)
        if lane_val is None:
            continue
        try:
            cand_wp = world_map.get_waypoint(loc, project_to_road=True, lane_type=lane_val)
        except Exception:
            cand_wp = None
        if cand_wp is None:
            continue
        try:
            wloc = cand_wp.transform.location
            wyaw = float(cand_wp.transform.rotation.yaw)
            road_id = int(getattr(cand_wp, "road_id", 0))
            lane_id = int(getattr(cand_wp, "lane_id", 0))
        except Exception:
            continue
        lane_key = (road_id, lane_id)
        dist = math.hypot(float(wloc.x) - float(loc.x), float(wloc.y) - float(loc.y))
        yaw_diff = _yaw_diff_deg(float(wyaw), float(raw_yaw))
        score = float(dist) + 0.35 * (float(yaw_diff) / 90.0)
        if prev_lane_key is not None and lane_key != prev_lane_key:
            score += 0.15
        if lane_name == "Shoulder":
            score += 0.08
        elif lane_name == "Parking":
            score += 0.20
        if best_score is None or score < best_score:
            best_score = score
            best_wp = cand_wp
            best_lane_key = lane_key
            best_lane_name = lane_name
            best_info = {
                "x": float(wloc.x),
                "y": float(wloc.y),
                "z": float(wloc.z),
                "yaw": float(wyaw),
                "dist": float(dist),
                "road_id": float(road_id),
                "lane_id": float(lane_id),
            }
    return best_info, best_lane_key, best_lane_name, best_wp
