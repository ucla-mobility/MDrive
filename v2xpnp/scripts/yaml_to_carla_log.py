#!/usr/bin/env python3
"""
Convert a V2XPnP-style YAML sequence into CARLA leaderboard-ready XML routes.

Outputs:
  - ego_route.xml (optional) with the ego trajectory as role="ego".
  - actors/<name>.xml for every non-ego object, role="npc", snap_to_road="false".
  - actors_manifest.json describing all actors (route_id, model, speed, etc.).
  - Optional GIF visualizing the replay frames.

Usage (typical):
  python v2xpnp/scripts/yaml_to_carla_log.py \\
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
    
    pedestrian_keywords = ["pedestrian", "walker", "person", "people"]
    
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
    "walker.pedestrian.0009",
    "walker.pedestrian.0011",
    "walker.pedestrian.0012",
    "walker.pedestrian.0013",
    "walker.pedestrian.0014",
]
CHILD_WALKER_BLUEPRINTS = {
    "walker.pedestrian.0001",
    "walker.pedestrian.0010",
}


def _is_child_walker_blueprint(bp_id: str) -> bool:
    return str(bp_id) in CHILD_WALKER_BLUEPRINTS


def map_obj_type(obj_type: str | None) -> str:
    """Map dataset obj_type to a CARLA 0.9.12 blueprint (vehicle or walker)."""
    # Define blueprint pools for random selection (excluding children)
    walker_blueprints = ADULT_WALKER_BLUEPRINTS
    
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
        return random.choice(car_blueprints)
    ot = obj_type.lower()
    
    # Pedestrians/Walkers
    if is_pedestrian_type(obj_type):
        return random.choice(walker_blueprints)
    
    # Buses and large vehicles
    if "bus" in ot:
        return random.choice(bus_blueprints)
    
    # Trucks and vans
    if "truck" in ot:
        if "fire" in ot:
            return random.choice(firetruck_blueprints)
        return random.choice(truck_blueprints)
    if "van" in ot or "sprinter" in ot:
        return random.choice(van_blueprints)
    
    # Emergency vehicles
    if "ambulance" in ot:
        return random.choice(ambulance_blueprints)
    if "police" in ot:
        return random.choice(police_blueprints)
    
    # Motorcycles
    if "motor" in ot or "motorcycle" in ot:
        return random.choice(motorcycle_blueprints)
    if "bike" in ot and "bicycle" not in ot:
        return random.choice(motorcycle_blueprints)
    
    # Bicycles (with rider)
    if "bicycle" in ot or "cycl" in ot:
        return random.choice(bicycle_blueprints)
    
    # SUVs and larger cars
    if "suv" in ot or "jeep" in ot:
        return random.choice(suv_blueprints)
    if "patrol" in ot:
        return random.choice(suv_blueprints)
    
    # Sedans and cars (default category)
    return random.choice(car_blueprints)


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


def _path_distance(traj: List[Waypoint]) -> float:
    dist = 0.0
    for a, b in zip(traj, traj[1:]):
        dist += euclid3((a.x, a.y, a.z), (b.x, b.y, b.z))
    return dist


def _classify_actor_kind(traj: List[Waypoint], obj_type_raw: str) -> Tuple[str, bool]:
    is_pedestrian = is_pedestrian_type(obj_type_raw)
    if is_pedestrian:
        kind = "walker"
        if len(traj) >= 2 and _path_distance(traj) < 0.5:
            kind = "walker_static"
        return kind, True

    kind = "npc"
    if len(traj) <= 1:
        kind = "static"
    elif len(traj) >= 2 and _path_distance(traj) < 0.5:
        kind = "static"
    return kind, False


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
    candidate_ids = [vid for vid in ids if float(start_times.get(vid, 0.0)) > 1e-6]

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
    candidate_ids = [
        vid for vid in ids if float(end_times.get(vid, 0.0)) < hold_until_time - 1e-6
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
        "already_at_horizon": len(already_at_horizon),
        "individually_safe": len(individually_safe),
        "selected": len(selected),
        "pair_conflicts": int(pair_conflicts),
        "timed_out_components": int(timed_out_components),
        "blocked_examples": blocked_examples,
        "selected_actor_ids": sorted(int(v) for v in selected),
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


def _candidate_waypoints_from_continuous_lane(
    prev_wp,
    step_dist: float,
) -> List[object]:
    """
    Return forward lane-follow candidates from the previous snapped waypoint.
    Uses multiple nearby step distances and deduplicates candidates.
    """
    if prev_wp is None:
        return []
    candidates: List[object] = []
    seen = set()

    def _add_wp(wp) -> None:
        if wp is None:
            return
        try:
            key = (
                int(getattr(wp, "road_id", 0)),
                int(getattr(wp, "lane_id", 0)),
                int(round(float(getattr(wp, "s", 0.0)) * 100.0)),
            )
        except Exception:
            key = (0, 0, id(wp))
        if key in seen:
            return
        seen.add(key)
        candidates.append(wp)

    # If ego barely moved, keep the same centerline point to avoid jitter.
    if step_dist <= 0.15:
        _add_wp(prev_wp)

    base = max(0.5, min(4.0, float(step_dist) if step_dist > 1e-6 else 1.0))
    probe_steps = sorted(
        set(
            [
                base,
                max(0.5, base * 0.7),
                min(4.0, max(0.8, base * 1.35)),
            ]
        )
    )
    for step in probe_steps:
        try:
            nxt = list(prev_wp.next(float(step)))
        except Exception:
            nxt = []
        for wp in nxt:
            _add_wp(wp)

    if not candidates:
        _add_wp(prev_wp)
    return candidates


def _select_lane_locked_waypoint(
    prev_wp,
    raw_wp: Waypoint,
    raw_step_dx: float,
    raw_step_dy: float,
    local_best_info: Optional[Dict[str, float]],
) -> object:
    """
    Follow the previous snapped lane centerline (via waypoint graph) and pick
    the branch that best matches current trajectory direction/position.
    """
    if prev_wp is None:
        return None
    step_dist = math.hypot(float(raw_step_dx), float(raw_step_dy))
    candidates = _candidate_waypoints_from_continuous_lane(prev_wp, step_dist)
    if not candidates:
        return None

    raw_dir_norm = math.hypot(float(raw_step_dx), float(raw_step_dy))
    if raw_dir_norm <= 1e-6:
        raw_dir_x = math.cos(math.radians(float(raw_wp.yaw)))
        raw_dir_y = math.sin(math.radians(float(raw_wp.yaw)))
    else:
        raw_dir_x = float(raw_step_dx) / raw_dir_norm
        raw_dir_y = float(raw_step_dy) / raw_dir_norm

    try:
        prev_loc = prev_wp.transform.location
        prev_x = float(prev_loc.x)
        prev_y = float(prev_loc.y)
    except Exception:
        prev_x = float(raw_wp.x)
        prev_y = float(raw_wp.y)

    best_wp = None
    best_score: Optional[float] = None
    for cand_wp in candidates:
        try:
            cloc = cand_wp.transform.location
            cyaw = float(cand_wp.transform.rotation.yaw)
            cx = float(cloc.x)
            cy = float(cloc.y)
        except Exception:
            continue
        dist_to_raw = math.hypot(cx - float(raw_wp.x), cy - float(raw_wp.y))
        yaw_diff = _yaw_diff_deg(cyaw, float(raw_wp.yaw))
        score = float(dist_to_raw) + 0.20 * (float(yaw_diff) / 90.0)

        # Keep forward progression aligned with trajectory segment direction.
        mvx = cx - prev_x
        mvy = cy - prev_y
        mv_norm = math.hypot(mvx, mvy)
        if mv_norm > 1e-6:
            cosang = (mvx * raw_dir_x + mvy * raw_dir_y) / mv_norm
            cosang = max(-1.0, min(1.0, float(cosang)))
            if cosang < 0.0:
                score += 2.5 + 2.0 * abs(cosang)
            else:
                score += 0.30 * (1.0 - cosang)

        # Keep branch close to the local nearest lane-center projection.
        if isinstance(local_best_info, dict):
            try:
                ax = float(local_best_info["x"])
                ay = float(local_best_info["y"])
                score += 0.70 * math.hypot(cx - ax, cy - ay)
            except Exception:
                pass

        if best_score is None or score < best_score:
            best_score = score
            best_wp = cand_wp
    return best_wp


def _compute_piecewise_ego_offsets(
    traj: List[Waypoint],
    times: List[float],
    world_map,
    max_shift: float,
    intent_margin: float,
    smooth_window: int,
    max_step_delta: float,
    bridge_max_gap_steps: int,
    bridge_straight_thresh_deg: float,
    snap_ego_to_lane: bool = False,
) -> Tuple[List[Tuple[float, float]], Dict[str, object]]:
    """Compute per-waypoint XY offsets for ego alignment while preserving trajectory timing."""
    if world_map is None or not traj:
        return [], {"status": "no_map_or_traj"}

    lane_names_all = ["Driving", "Shoulder", "Parking"]
    sample_count = max(8, min(len(traj), 24))
    samples = _select_alignment_samples(traj, times, sample_count)
    sample_projections: List[Dict[str, Dict[str, float]]] = []
    for s in samples:
        loc = carla.Location(x=float(s["x"]), y=float(s["y"]), z=float(s["z"]))
        sample_projections.append(_project_to_lane_types(world_map, loc, lane_names_all))
    intent_info = _infer_alignment_intent("npc", sample_projections, intent_margin)
    candidate_lanes = list(intent_info.get("candidate_lanes") or lane_names_all)
    if not candidate_lanes:
        candidate_lanes = lane_names_all

    raw_dx: List[float] = []
    raw_dy: List[float] = []
    valid_mask: List[bool] = []
    lane_keys: List[Optional[Tuple[int, int]]] = []
    yaws: List[float] = []
    lane_switches_raw = 0
    prev_lane_key: Optional[Tuple[int, int]] = None
    lane_dists: List[float] = []
    yaw_diffs: List[float] = []
    lane_lock_used = bool(snap_ego_to_lane)
    lane_lock_fallbacks = 0
    lane_lock_kept_prev = 0
    prev_snap_wp = None

    for idx, wp in enumerate(traj):
        loc = carla.Location(x=float(wp.x), y=float(wp.y), z=float(wp.z))
        best, best_lane_key, best_lane_name, best_wp = _best_lane_projection_with_waypoint(
            world_map=world_map,
            loc=loc,
            raw_yaw=float(wp.yaw),
            candidate_lanes=candidate_lanes,
            prev_lane_key=prev_lane_key,
        )

        if lane_lock_used and prev_snap_wp is not None:
            prev_raw = traj[idx - 1] if idx > 0 else wp
            raw_step_dx = float(wp.x) - float(prev_raw.x)
            raw_step_dy = float(wp.y) - float(prev_raw.y)
            lane_locked_wp = _select_lane_locked_waypoint(
                prev_wp=prev_snap_wp,
                raw_wp=wp,
                raw_step_dx=raw_step_dx,
                raw_step_dy=raw_step_dy,
                local_best_info=best,
            )
            if lane_locked_wp is None:
                lane_lock_fallbacks += 1
            else:
                try:
                    lwloc = lane_locked_wp.transform.location
                    lwyaw = float(lane_locked_wp.transform.rotation.yaw)
                    lroad = int(getattr(lane_locked_wp, "road_id", 0))
                    llane = int(getattr(lane_locked_wp, "lane_id", 0))
                    best = {
                        "x": float(lwloc.x),
                        "y": float(lwloc.y),
                        "z": float(lwloc.z),
                        "yaw": float(lwyaw),
                        "dist": float(math.hypot(float(lwloc.x) - float(wp.x), float(lwloc.y) - float(wp.y))),
                        "road_id": float(lroad),
                        "lane_id": float(llane),
                    }
                    best_lane_key = (lroad, llane)
                    best_wp = lane_locked_wp
                    if math.hypot(raw_step_dx, raw_step_dy) <= 0.15:
                        lane_lock_kept_prev += 1
                except Exception:
                    lane_lock_fallbacks += 1

        if best is None:
            raw_dx.append(0.0)
            raw_dy.append(0.0)
            valid_mask.append(False)
            lane_keys.append(None)
            yaws.append(float(wp.yaw))
            continue

        dx = float(best["x"]) - float(wp.x)
        dy = float(best["y"]) - float(wp.y)
        mag = math.hypot(dx, dy)
        if mag > max_shift > 0.0:
            scale = max_shift / max(mag, 1e-6)
            dx *= scale
            dy *= scale
        raw_dx.append(dx)
        raw_dy.append(dy)
        valid_mask.append(True)
        lane_keys.append(best_lane_key)
        yaws.append(float(wp.yaw))
        lane_dists.append(float(best["dist"]))
        yaw_diffs.append(_yaw_diff_deg(float(best["yaw"]), float(wp.yaw)))
        if prev_lane_key is not None and best_lane_key is not None and best_lane_key != prev_lane_key:
            lane_switches_raw += 1
        if best_lane_name is not None or best_lane_key is not None:
            prev_lane_key = best_lane_key
        if best_wp is not None:
            prev_snap_wp = best_wp

    bridge_info = {"bridged_missing": 0, "bridged_transient": 0}
    if int(bridge_max_gap_steps) > 0:
        bridge_info = _bridge_missing_or_transient_offsets(
            dxs=raw_dx,
            dys=raw_dy,
            valid_mask=valid_mask,
            lane_keys=lane_keys,
            yaws=yaws,
            max_gap_steps=int(bridge_max_gap_steps),
            straight_thresh_deg=float(bridge_straight_thresh_deg),
        )

    # Backfill any remaining missing offsets from nearest valid point.
    if any(valid_mask):
        valid_idx = [i for i, ok in enumerate(valid_mask) if ok]
        for i, ok in enumerate(valid_mask):
            if ok:
                continue
            nearest = min(valid_idx, key=lambda j: abs(j - i))
            raw_dx[i] = raw_dx[nearest]
            raw_dy[i] = raw_dy[nearest]
    else:
        return (
            [(0.0, 0.0) for _ in traj],
            {
                "status": "no_projection",
                "intent": intent_info,
                "candidate_lanes": candidate_lanes,
                "no_wp_ratio": 1.0,
            },
        )

    # Smooth offsets to avoid jagged lane oscillation.
    smooth_dx = _moving_average(raw_dx, smooth_window)
    smooth_dy = _moving_average(raw_dy, smooth_window)
    if smooth_dx:
        smooth_dx[0] = raw_dx[0]
        smooth_dy[0] = raw_dy[0]
        smooth_dx[-1] = raw_dx[-1]
        smooth_dy[-1] = raw_dy[-1]

    # Limit frame-to-frame offset delta to avoid sudden teleports.
    max_step_delta = max(0.05, float(max_step_delta))
    for i in range(1, len(smooth_dx)):
        ddx = smooth_dx[i] - smooth_dx[i - 1]
        ddy = smooth_dy[i] - smooth_dy[i - 1]
        step = math.hypot(ddx, ddy)
        if step > max_step_delta:
            scale = max_step_delta / max(step, 1e-6)
            smooth_dx[i] = smooth_dx[i - 1] + ddx * scale
            smooth_dy[i] = smooth_dy[i - 1] + ddy * scale

    # Re-clamp final magnitude.
    offsets: List[Tuple[float, float]] = []
    for dx, dy in zip(smooth_dx, smooth_dy):
        mag = math.hypot(dx, dy)
        if mag > max_shift > 0.0:
            scale = max_shift / max(mag, 1e-6)
            dx *= scale
            dy *= scale
        offsets.append((float(dx), float(dy)))

    shift_mags = [math.hypot(dx, dy) for dx, dy in offsets]
    lane_keys_valid = [lk for lk in lane_keys if lk is not None]
    lane_switches_post = 0
    prev_key: Optional[Tuple[int, int]] = None
    for lk in lane_keys:
        if lk is None:
            continue
        if prev_key is not None and lk != prev_key:
            lane_switches_post += 1
        prev_key = lk
    report = {
        "status": "ok",
        "mode": "piecewise",
        "intent": intent_info,
        "candidate_lanes": candidate_lanes,
        "samples": len(samples),
        "no_wp_ratio": float(sum(1 for ok in valid_mask if not ok) / max(1, len(valid_mask))),
        "lane_switches": int(lane_switches_post),
        "lane_switches_raw": int(lane_switches_raw),
        "bridge": bridge_info,
        "lane_lock": {
            "enabled": lane_lock_used,
            "fallbacks": int(lane_lock_fallbacks),
            "kept_prev_steps": int(lane_lock_kept_prev),
        },
        "lane_key_coverage": float(len(lane_keys_valid) / max(1, len(lane_keys))),
        "lane_dist_median": float(_median(lane_dists)) if lane_dists else 999.0,
        "yaw_diff_median": float(_median(yaw_diffs)) if yaw_diffs else 180.0,
        "shift_median": float(_median(shift_mags)) if shift_mags else 0.0,
        "shift_max": float(max(shift_mags)) if shift_mags else 0.0,
        "first_offset": [float(offsets[0][0]), float(offsets[0][1])] if offsets else [0.0, 0.0],
        "last_offset": [float(offsets[-1][0]), float(offsets[-1][1])] if offsets else [0.0, 0.0],
    }
    return offsets, report


def _compute_piecewise_actor_offsets(
    traj: List[Waypoint],
    times: List[float],
    world_map,
    role: str,
    base_dx: float,
    base_dy: float,
    max_shift: float,
    intent_margin: float,
    local_limit: float,
    smooth_window: int,
    max_step_delta: float,
    bridge_max_gap_steps: int,
    bridge_straight_thresh_deg: float,
) -> Tuple[List[Tuple[float, float]], Dict[str, object]]:
    """
    Compute per-waypoint XY offsets around a global base offset.
    The refinement is bounded so it cannot drift far from the globally-selected shift.
    """
    if world_map is None or not traj:
        return [], {"status": "no_map_or_traj"}

    lane_names_all = (
        ["Driving", "Shoulder", "Parking"]
        if role in ("npc", "static")
        else ["Sidewalk", "Shoulder", "Driving"]
    )
    sample_count = max(8, min(len(traj), 24))
    samples = _select_alignment_samples(traj, times, sample_count)
    sample_projections: List[Dict[str, Dict[str, float]]] = []
    for s in samples:
        loc = carla.Location(
            x=float(s["x"] + base_dx),
            y=float(s["y"] + base_dy),
            z=float(s["z"]),
        )
        sample_projections.append(_project_to_lane_types(world_map, loc, lane_names_all))
    intent_info = _infer_alignment_intent(role, sample_projections, intent_margin)
    candidate_lanes = list(intent_info.get("candidate_lanes") or lane_names_all)
    if not candidate_lanes:
        candidate_lanes = lane_names_all

    local_limit = max(0.0, float(local_limit))
    raw_dx: List[float] = []
    raw_dy: List[float] = []
    valid_mask: List[bool] = []
    lane_keys: List[Optional[Tuple[int, int]]] = []
    yaws: List[float] = []
    lane_switches_raw = 0
    prev_lane_key: Optional[Tuple[int, int]] = None
    lane_dists: List[float] = []
    yaw_diffs: List[float] = []

    for wp in traj:
        loc = carla.Location(
            x=float(wp.x + base_dx),
            y=float(wp.y + base_dy),
            z=float(wp.z),
        )
        proj = _project_to_lane_types(world_map, loc, candidate_lanes)
        best = None
        best_lane_key: Optional[Tuple[int, int]] = None
        best_score = None
        best_lane_name = None
        for lane_name in candidate_lanes:
            info = proj.get(lane_name)
            if info is None:
                continue
            lane_key = (int(info["road_id"]), int(info["lane_id"]))
            yaw_diff = _yaw_diff_deg(float(info["yaw"]), float(wp.yaw))
            score = float(info["dist"]) + 0.35 * (yaw_diff / 90.0)
            if prev_lane_key is not None and lane_key != prev_lane_key:
                score += 0.15
            if role in ("npc", "static"):
                if lane_name == "Shoulder":
                    score += 0.08
                elif lane_name == "Parking":
                    score += 0.20
            if best_score is None or score < best_score:
                best_score = score
                best = info
                best_lane_key = lane_key
                best_lane_name = lane_name

        if best is None:
            raw_dx.append(float(base_dx))
            raw_dy.append(float(base_dy))
            valid_mask.append(False)
            lane_keys.append(None)
            yaws.append(float(wp.yaw))
            continue

        dx = float(best["x"]) - float(wp.x)
        dy = float(best["y"]) - float(wp.y)

        # Bound local refinement around the global offset.
        ddx = dx - float(base_dx)
        ddy = dy - float(base_dy)
        dmag = math.hypot(ddx, ddy)
        if dmag > local_limit > 0.0:
            scale = local_limit / max(dmag, 1e-6)
            dx = float(base_dx) + ddx * scale
            dy = float(base_dy) + ddy * scale

        # Bound absolute magnitude.
        mag = math.hypot(dx, dy)
        if mag > max_shift > 0.0:
            scale = max_shift / max(mag, 1e-6)
            dx *= scale
            dy *= scale

        raw_dx.append(dx)
        raw_dy.append(dy)
        valid_mask.append(True)
        lane_keys.append(best_lane_key)
        yaws.append(float(wp.yaw))
        lane_dists.append(float(best["dist"]))
        yaw_diffs.append(_yaw_diff_deg(float(best["yaw"]), float(wp.yaw)))
        if prev_lane_key is not None and best_lane_key is not None and best_lane_key != prev_lane_key:
            lane_switches_raw += 1
        if best_lane_name is not None:
            prev_lane_key = best_lane_key

    bridge_info = {"bridged_missing": 0, "bridged_transient": 0}
    if int(bridge_max_gap_steps) > 0:
        bridge_info = _bridge_missing_or_transient_offsets(
            dxs=raw_dx,
            dys=raw_dy,
            valid_mask=valid_mask,
            lane_keys=lane_keys,
            yaws=yaws,
            max_gap_steps=int(bridge_max_gap_steps),
            straight_thresh_deg=float(bridge_straight_thresh_deg),
        )

    if any(valid_mask):
        valid_idx = [i for i, ok in enumerate(valid_mask) if ok]
        for i, ok in enumerate(valid_mask):
            if ok:
                continue
            nearest = min(valid_idx, key=lambda j: abs(j - i))
            raw_dx[i] = raw_dx[nearest]
            raw_dy[i] = raw_dy[nearest]
    else:
        return (
            [(float(base_dx), float(base_dy)) for _ in traj],
            {
                "status": "no_projection",
                "intent": intent_info,
                "candidate_lanes": candidate_lanes,
                "no_wp_ratio": 1.0,
            },
        )

    smooth_dx = _moving_average(raw_dx, smooth_window)
    smooth_dy = _moving_average(raw_dy, smooth_window)
    if smooth_dx:
        smooth_dx[0] = raw_dx[0]
        smooth_dy[0] = raw_dy[0]
        smooth_dx[-1] = raw_dx[-1]
        smooth_dy[-1] = raw_dy[-1]

    # Prevent sudden frame-to-frame offset jumps.
    max_step_delta = max(0.05, float(max_step_delta))
    for i in range(1, len(smooth_dx)):
        ddx = smooth_dx[i] - smooth_dx[i - 1]
        ddy = smooth_dy[i] - smooth_dy[i - 1]
        step = math.hypot(ddx, ddy)
        if step > max_step_delta:
            scale = max_step_delta / max(step, 1e-6)
            smooth_dx[i] = smooth_dx[i - 1] + ddx * scale
            smooth_dy[i] = smooth_dy[i - 1] + ddy * scale

    offsets: List[Tuple[float, float]] = []
    for dx, dy in zip(smooth_dx, smooth_dy):
        # Re-enforce bounds.
        ddx = dx - float(base_dx)
        ddy = dy - float(base_dy)
        dmag = math.hypot(ddx, ddy)
        if dmag > local_limit > 0.0:
            scale = local_limit / max(dmag, 1e-6)
            dx = float(base_dx) + ddx * scale
            dy = float(base_dy) + ddy * scale
        mag = math.hypot(dx, dy)
        if mag > max_shift > 0.0:
            scale = max_shift / max(mag, 1e-6)
            dx *= scale
            dy *= scale
        offsets.append((float(dx), float(dy)))

    shift_mags = [math.hypot(dx, dy) for dx, dy in offsets]
    lane_keys_valid = [lk for lk in lane_keys if lk is not None]
    lane_switches_post = 0
    prev_key: Optional[Tuple[int, int]] = None
    for lk in lane_keys:
        if lk is None:
            continue
        if prev_key is not None and lk != prev_key:
            lane_switches_post += 1
        prev_key = lk
    report = {
        "status": "ok",
        "mode": "piecewise_refine",
        "intent": intent_info,
        "candidate_lanes": candidate_lanes,
        "samples": len(samples),
        "no_wp_ratio": float(sum(1 for ok in valid_mask if not ok) / max(1, len(valid_mask))),
        "lane_switches": int(lane_switches_post),
        "lane_switches_raw": int(lane_switches_raw),
        "bridge": bridge_info,
        "lane_key_coverage": float(len(lane_keys_valid) / max(1, len(lane_keys))),
        "lane_dist_median": float(_median(lane_dists)) if lane_dists else 999.0,
        "yaw_diff_median": float(_median(yaw_diffs)) if yaw_diffs else 180.0,
        "base_shift": [float(base_dx), float(base_dy)],
        "shift_median": float(_median(shift_mags)) if shift_mags else 0.0,
        "shift_max": float(max(shift_mags)) if shift_mags else 0.0,
        "first_offset": [float(offsets[0][0]), float(offsets[0][1])] if offsets else [float(base_dx), float(base_dy)],
        "last_offset": [float(offsets[-1][0]), float(offsets[-1][1])] if offsets else [float(base_dx), float(base_dy)],
    }
    return offsets, report


def _sample_offset_profile(
    times: List[float],
    offsets_wp: List[Tuple[float, float]],
    sample_times: List[float],
    always_active: bool,
) -> List[Optional[Tuple[float, float]]]:
    if not offsets_wp:
        return [None for _ in sample_times]
    if len(offsets_wp) == 1 or always_active:
        val = (float(offsets_wp[0][0]), float(offsets_wp[0][1]))
        return [val for _ in sample_times]
    out: List[Optional[Tuple[float, float]]] = []
    idx = 0
    last = len(times) - 1
    for t in sample_times:
        if t < times[0] or t > times[-1]:
            out.append(None)
            continue
        while idx + 1 < last and times[idx + 1] < t:
            idx += 1
        if idx + 1 >= len(times):
            out.append((float(offsets_wp[-1][0]), float(offsets_wp[-1][1])))
            continue
        t0 = times[idx]
        t1 = times[idx + 1]
        if t1 <= t0:
            alpha = 0.0
        else:
            alpha = (t - t0) / (t1 - t0)
        dx = float(offsets_wp[idx][0]) + (float(offsets_wp[idx + 1][0]) - float(offsets_wp[idx][0])) * alpha
        dy = float(offsets_wp[idx][1]) + (float(offsets_wp[idx + 1][1]) - float(offsets_wp[idx][1])) * alpha
        out.append((dx, dy))
    return out


def _select_alignment_samples(
    traj: List[Waypoint],
    times: List[float],
    sample_count: int,
) -> List[Dict[str, float]]:
    """Pick evenly spaced samples along the trajectory distance."""
    if not traj:
        return []
    n = len(traj)
    sample_count = max(2, min(int(sample_count), n))
    # Build cumulative distance along path
    cum = [0.0]
    for i in range(1, n):
        dx = traj[i].x - traj[i - 1].x
        dy = traj[i].y - traj[i - 1].y
        cum.append(cum[-1] + math.hypot(dx, dy))
    total = cum[-1]
    if total < 1e-6:
        idxs = sorted(
            set(int(round(i * (n - 1) / max(1, sample_count - 1))) for i in range(sample_count))
        )
    else:
        targets = [total * (i / (sample_count - 1)) for i in range(sample_count)]
        idxs = []
        for d in targets:
            idx = bisect.bisect_left(cum, d)
            idx = max(0, min(n - 1, idx))
            idxs.append(idx)
        idxs = sorted(set(idxs))
    samples: List[Dict[str, float]] = []
    for idx in idxs:
        wp = traj[idx]
        t = float(times[idx]) if idx < len(times) else float(idx)
        samples.append(
            {
                "x": float(wp.x),
                "y": float(wp.y),
                "z": float(wp.z),
                "yaw": float(wp.yaw),
                "t": t,
                "idx": float(idx),
            }
        )
    return samples


def _project_to_lane_types(
    world_map,
    loc,
    lane_type_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """Project a location to multiple lane types, returning per-lane info."""
    results: Dict[str, Dict[str, float]] = {}
    if world_map is None:
        return results
    for name in lane_type_names:
        lane_val = _lane_type_value(name)
        if lane_val is None:
            continue
        try:
            wp = world_map.get_waypoint(loc, project_to_road=True, lane_type=lane_val)
        except Exception:
            wp = None
        if wp is None:
            continue
        wloc = wp.transform.location
        dist = math.hypot(float(wloc.x) - float(loc.x), float(wloc.y) - float(loc.y))
        yaw = float(wp.transform.rotation.yaw)
        results[name] = {
            "x": float(wloc.x),
            "y": float(wloc.y),
            "z": float(wloc.z),
            "yaw": yaw,
            "dist": float(dist),
            "road_id": float(getattr(wp, "road_id", 0)),
            "lane_id": float(getattr(wp, "lane_id", 0)),
        }
    return results


def _infer_alignment_intent(
    role: str,
    projections: List[Dict[str, Dict[str, float]]],
    intent_margin: float,
) -> Dict[str, object]:
    """Infer lane intent (vehicles vs walkers) from projection distances."""
    if role in ("npc", "static"):
        lane_names = [n for n in ("Driving", "Shoulder", "Parking") if any(n in p for p in projections)]
        counts = {n: 0 for n in lane_names}
        dists: Dict[str, List[float]] = {n: [] for n in lane_names}
        for proj in projections:
            best_lane = None
            best_dist = None
            for ln in lane_names:
                info = proj.get(ln)
                if info is None:
                    continue
                dists[ln].append(float(info["dist"]))
                if best_dist is None or info["dist"] < best_dist:
                    best_dist = info["dist"]
                    best_lane = ln
            if best_lane is not None:
                counts[best_lane] += 1
        total = sum(counts.values()) or 1
        intent_lane = max(counts.items(), key=lambda kv: kv[1])[0] if counts else "Driving"
        intent_ratio = counts.get(intent_lane, 0) / float(total)
        med_dist = _median(dists.get(intent_lane, []))
        intent = intent_lane.lower()
        if med_dist > 6.0 and intent_ratio < 0.6:
            intent = "unknown"
        return {
            "intent": intent,
            "intent_ratio": intent_ratio,
            "lane_counts": counts,
            "lane_median_dist": {k: _median(v) for k, v in dists.items()},
            "candidate_lanes": lane_names or ["Driving"],
            "score_lanes": lane_names or ["Driving"],
        }

    # Walker intent
    lane_names = [n for n in ("Sidewalk", "Shoulder", "Driving") if any(n in p for p in projections)]
    sidewalk_count = 0
    road_count = 0
    mixed = 0
    sidewalk_dists: List[float] = []
    road_dists: List[float] = []
    for proj in projections:
        ds = proj.get("Sidewalk", {}).get("dist") if proj.get("Sidewalk") else None
        dd = proj.get("Driving", {}).get("dist") if proj.get("Driving") else None
        if ds is not None:
            sidewalk_dists.append(float(ds))
        if dd is not None:
            road_dists.append(float(dd))
        if ds is None and dd is None:
            continue
        if ds is not None and dd is not None:
            if ds + intent_margin < dd:
                sidewalk_count += 1
            elif dd + intent_margin < ds:
                road_count += 1
            else:
                mixed += 1
        elif ds is not None:
            sidewalk_count += 1
        elif dd is not None:
            road_count += 1
    total = sidewalk_count + road_count + mixed
    total = max(1, total)
    sidewalk_ratio = sidewalk_count / float(total)
    road_ratio = road_count / float(total)
    sidewalk_med = _median(sidewalk_dists)
    road_med = _median(road_dists)
    if sidewalk_ratio >= 0.6 and sidewalk_med < 3.0:
        intent = "sidewalk"
    elif road_ratio >= 0.6 and road_med < 3.0:
        intent = "road"
    else:
        intent = "mixed"
    if intent == "sidewalk":
        candidate_lanes = [n for n in ("Sidewalk", "Shoulder") if n in lane_names]
    elif intent == "road":
        candidate_lanes = [n for n in ("Driving", "Shoulder") if n in lane_names]
    else:
        candidate_lanes = lane_names
    if not candidate_lanes:
        candidate_lanes = lane_names or ["Sidewalk"]
    return {
        "intent": intent,
        "intent_ratio": max(sidewalk_ratio, road_ratio),
        "lane_counts": {"sidewalk": sidewalk_count, "road": road_count, "mixed": mixed},
        "lane_median_dist": {"sidewalk": sidewalk_med, "road": road_med},
        "candidate_lanes": candidate_lanes,
        "score_lanes": lane_names or ["Sidewalk", "Driving"],
    }


def _score_alignment_candidate(
    world_map,
    samples: List[Dict[str, float]],
    dx: float,
    dy: float,
    role: str,
    intent_info: Dict[str, object],
    lane_change_ref: int,
) -> Tuple[float, Dict[str, float]]:
    """Score a constant XY offset against map geometry + intent."""
    if world_map is None or not samples:
        return 1e6, {
            "dist_median": 999.0,
            "dist_p95": 999.0,
            "yaw_median": 180.0,
            "intent_mismatch": 1.0,
            "lane_changes": 0.0,
            "no_wp_ratio": 1.0,
        }
    lane_names = list(intent_info.get("score_lanes") or [])
    yaw_weight = 0.4
    distances: List[float] = []
    yaw_diffs: List[float] = []
    lane_types: List[str] = []
    lane_ids: List[Tuple[int, int]] = []
    no_wp = 0
    for sample in samples:
        loc = carla.Location(
            x=float(sample["x"] + dx),
            y=float(sample["y"] + dy),
            z=float(sample["z"]),
        )
        proj = _project_to_lane_types(world_map, loc, lane_names)
        best = None
        best_score = None
        best_lane = None
        for lane_name, info in proj.items():
            dist = float(info["dist"])
            yaw_diff = _yaw_diff_deg(float(info["yaw"]), float(sample["yaw"]))
            score = dist + yaw_weight * (yaw_diff / 90.0)
            if best_score is None or score < best_score:
                best_score = score
                best = info
                best_lane = lane_name
        if best is None:
            no_wp += 1
            continue
        distances.append(float(best["dist"]))
        yaw_diffs.append(_yaw_diff_deg(float(best["yaw"]), float(sample["yaw"])))
        lane_types.append(str(best_lane))
        lane_ids.append((int(best["road_id"]), int(best["lane_id"])))
    total = max(1, len(samples))
    no_wp_ratio = no_wp / float(total)
    dist_med = _median(distances) if distances else 999.0
    dist_p95 = _quantile(distances, 0.95) if distances else 999.0
    yaw_med = _median(yaw_diffs) if yaw_diffs else 180.0
    # Lane change count
    lane_changes = 0
    for prev, nxt in zip(lane_ids, lane_ids[1:]):
        if prev != nxt:
            lane_changes += 1
    lane_change_penalty = abs(lane_changes - int(lane_change_ref))
    # Intent mismatch
    intent = str(intent_info.get("intent") or "")
    intent_mismatch = 0.0
    if role in ("npc", "static"):
        intent_lane = intent.upper() if intent else ""
        if intent_lane:
            mismatches = sum(1 for ln in lane_types if ln.upper() != intent_lane)
            intent_mismatch = mismatches / float(max(1, len(lane_types)))
    else:
        if intent == "sidewalk":
            mismatches = sum(1 for ln in lane_types if ln == "Driving")
            intent_mismatch = mismatches / float(max(1, len(lane_types)))
        elif intent == "road":
            mismatches = sum(1 for ln in lane_types if ln == "Sidewalk")
            intent_mismatch = mismatches / float(max(1, len(lane_types)))
        else:
            intent_mismatch = 0.0
    # Weighted cost
    w_dist = 1.0
    w_p95 = 0.5
    w_yaw = 0.25
    w_intent = 2.0
    w_lane_change = 0.4
    w_offset = 0.2
    w_no_wp = 3.0
    cost = (
        w_dist * dist_med
        + w_p95 * dist_p95
        + w_yaw * (yaw_med / 90.0)
        + w_intent * intent_mismatch
        + w_lane_change * lane_change_penalty
        + w_offset * math.hypot(dx, dy)
        + w_no_wp * no_wp_ratio
    )
    return cost, {
        "dist_median": float(dist_med),
        "dist_p95": float(dist_p95),
        "yaw_median": float(yaw_med),
        "intent_mismatch": float(intent_mismatch),
        "lane_changes": float(lane_changes),
        "no_wp_ratio": float(no_wp_ratio),
    }


def _build_alignment_candidates(
    traj: List[Waypoint],
    times: List[float],
    role: str,
    world_map,
    max_shift: float,
    sample_count: int,
    window_count: int,
    intent_margin: float,
) -> Tuple[List[SpawnCandidate], Dict[str, object]]:
    """Generate alignment-based candidates using multiple waypoints."""
    if world_map is None or not traj:
        return [], {"status": "no_map_or_traj"}
    samples = _select_alignment_samples(traj, times, sample_count)
    if not samples:
        return [], {"status": "no_samples"}
    lane_names_all = ["Driving", "Shoulder", "Parking"] if role in ("npc", "static") else ["Sidewalk", "Shoulder", "Driving"]
    projections: List[Dict[str, Dict[str, float]]] = []
    for s in samples:
        loc = carla.Location(x=float(s["x"]), y=float(s["y"]), z=float(s["z"]))
        projections.append(_project_to_lane_types(world_map, loc, lane_names_all))
    intent_info = _infer_alignment_intent(role, projections, intent_margin)
    candidate_lanes = list(intent_info.get("candidate_lanes") or [])
    if not candidate_lanes:
        candidate_lanes = lane_names_all

    # Reference lane change count at zero offset
    ref_cost, ref_stats = _score_alignment_candidate(
        world_map, samples, 0.0, 0.0, role, intent_info, lane_change_ref=0
    )
    lane_change_ref = int(ref_stats.get("lane_changes", 0))

    candidates: Dict[Tuple[int, int], SpawnCandidate] = {}

    def _add_offset(dx: float, dy: float, source: str) -> None:
        dist = math.hypot(dx, dy)
        if dist > max_shift + 1e-6:
            return
        key = (int(round(dx * 100)), int(round(dy * 100)))
        if key in candidates:
            return
        cost, stats = _score_alignment_candidate(
            world_map, samples, dx, dy, role, intent_info, lane_change_ref
        )
        cand = SpawnCandidate(dx=float(dx), dy=float(dy), source=source, base_cost=float(cost))
        cand.align_stats = stats
        candidates[key] = cand

    # Always keep authored (zero-offset) as a candidate.
    _add_offset(0.0, 0.0, "align_authored")

    # Global median offsets per lane type
    for lane in candidate_lanes:
        offsets = []
        for s, proj in zip(samples, projections):
            info = proj.get(lane)
            if info is None:
                continue
            offsets.append((float(info["x"]) - float(s["x"]), float(info["y"]) - float(s["y"])))
        if offsets:
            dx = _median([o[0] for o in offsets])
            dy = _median([o[1] for o in offsets])
            _add_offset(dx, dy, f"align_{lane.lower()}")

    # Best-per-sample lane median
    offsets_best = []
    for s, proj in zip(samples, projections):
        best = None
        best_dist = None
        for lane in candidate_lanes:
            info = proj.get(lane)
            if info is None:
                continue
            if best_dist is None or info["dist"] < best_dist:
                best_dist = info["dist"]
                best = info
        if best is None:
            continue
        offsets_best.append((float(best["x"]) - float(s["x"]), float(best["y"]) - float(s["y"])))
    if offsets_best:
        dx = _median([o[0] for o in offsets_best])
        dy = _median([o[1] for o in offsets_best])
        _add_offset(dx, dy, "align_best")

    # Windowed medians for local slices
    window_count = max(1, int(window_count))
    if window_count > 1 and len(samples) >= window_count:
        total = len(samples)
        indices = list(range(total))
        for wi in range(window_count):
            start = int(round(wi * total / window_count))
            end = int(round((wi + 1) * total / window_count))
            window = indices[start:end]
            if not window:
                continue
            for lane in candidate_lanes:
                offsets = []
                for idx in window:
                    proj = projections[int(idx)]
                    info = proj.get(lane)
                    if info is None:
                        continue
                    s = samples[int(idx)]
                    offsets.append((float(info["x"]) - float(s["x"]), float(info["y"]) - float(s["y"])))
                if offsets:
                    dx = _median([o[0] for o in offsets])
                    dy = _median([o[1] for o in offsets])
                    _add_offset(dx, dy, f"align_{lane.lower()}_w{wi}")

    report = {
        "status": "ok",
        "intent": intent_info,
        "sample_count": len(samples),
        "lane_change_ref": lane_change_ref,
        "candidates": [
            {
                "dx": c.dx,
                "dy": c.dy,
                "source": c.source,
                "score": c.base_cost,
                "stats": c.align_stats or {},
            }
            for c in candidates.values()
        ],
    }
    return list(candidates.values()), report


def _align_ego_trajectories(
    ego_trajs: Sequence[List[Waypoint]] | None,
    ego_times_list: Sequence[List[float]] | None,
    world,
    world_map,
    blueprint_lib,
    args: argparse.Namespace,
) -> Dict[str, object]:
    """Align ego trajectories to map geometry using the same multi-waypoint alignment logic."""
    out: Dict[str, object] = {
        "status": "ok",
        "egos": [],
        "summary": {},
    }
    if not ego_trajs:
        out["status"] = "no_egos"
        out["summary"] = {
            "egos_considered": 0,
            "egos_aligned": 0,
            "egos_z_shifted": 0,
            "egos_no_candidates": 0,
            "egos_spawn_valid": 0,
        }
        return out
    if world_map is None:
        out["status"] = "no_map"
        out["summary"] = {
            "egos_considered": len(ego_trajs),
            "egos_aligned": 0,
            "egos_z_shifted": 0,
            "egos_no_candidates": len(ego_trajs),
            "egos_spawn_valid": 0,
        }
        return out

    max_shift = max(0.0, float(args.spawn_preprocess_max_shift))
    sample_count = int(args.spawn_preprocess_align_samples)
    window_count = int(args.spawn_preprocess_align_windows)
    intent_margin = float(args.spawn_preprocess_align_intent_margin)
    normalize_z = bool(args.spawn_preprocess_normalize_z)
    piecewise_mode = bool(getattr(args, "spawn_preprocess_align_ego_piecewise", True))
    piecewise_smooth_window = int(getattr(args, "spawn_preprocess_align_ego_smooth_window", 9))
    piecewise_max_step_delta = float(getattr(args, "spawn_preprocess_align_ego_max_step_delta", 0.45))
    bridge_max_gap_steps = int(getattr(args, "spawn_preprocess_bridge_max_gap_steps", 6))
    bridge_straight_thresh_deg = float(getattr(args, "spawn_preprocess_bridge_straight_thresh_deg", 18.0))
    snap_ego_to_lane = bool(getattr(args, "snap_ego_to_lane", False))

    ego_bp = None
    ego_bp_reason = "no_blueprint_library"
    ego_bp_model = str(getattr(args, "ego_model", ""))
    if blueprint_lib is not None:
        ego_bp, ego_bp_model, ego_bp_reason = _select_blueprint(
            blueprint_lib=blueprint_lib,
            model=ego_bp_model,
            kind="npc",
            obj_type_raw="car",
        )
    out["ego_blueprint"] = {
        "requested": str(getattr(args, "ego_model", "")),
        "used": ego_bp_model,
        "reason": ego_bp_reason,
    }

    aligned = 0
    z_shifted = 0
    no_candidates = 0
    spawn_valid = 0

    for ego_idx, traj in enumerate(ego_trajs):
        entry: Dict[str, object] = {
            "ego_index": int(ego_idx),
            "status": "ok",
            "chosen": None,
        }
        out["egos"].append(entry)
        if not traj:
            entry["status"] = "empty_traj"
            no_candidates += 1
            continue

        times = _ensure_times(
            traj,
            list(ego_times_list[ego_idx]) if ego_times_list and ego_idx < len(ego_times_list) else None,
            float(args.dt),
        )
        if piecewise_mode:
            offsets, align_report = _compute_piecewise_ego_offsets(
                traj=traj,
                times=times,
                world_map=world_map,
                max_shift=max_shift,
                intent_margin=intent_margin,
                smooth_window=piecewise_smooth_window,
                max_step_delta=piecewise_max_step_delta,
                bridge_max_gap_steps=bridge_max_gap_steps,
                bridge_straight_thresh_deg=bridge_straight_thresh_deg,
                snap_ego_to_lane=snap_ego_to_lane,
            )
            entry["alignment"] = align_report
            entry["candidate_count"] = len(offsets)
            if not offsets:
                entry["status"] = "no_candidates"
                no_candidates += 1
                continue

            chosen = SpawnCandidate(
                dx=float(offsets[0][0]),
                dy=float(offsets[0][1]),
                source="align_ego_piecewise",
                base_cost=0.0,
                valid=(world is None or ego_bp is None),
                reason=None if (world is None or ego_bp is None) else "spawn_unchecked",
            )
            if world is not None and ego_bp is not None:
                _try_spawn_candidate(
                    world=world,
                    world_map=world_map,
                    blueprint=ego_bp,
                    base_wp=traj[0],
                    cand=chosen,
                    normalize_z=normalize_z,
                )

                # If spawn check failed at the first offset, try local XY retries and
                # apply the best retry as a constant adjustment over the full path.
                if not chosen.valid:
                    retry_steps = [0.0, 0.2, -0.2, 0.4, -0.4, 0.8, -0.8]
                    retry_pairs: List[Tuple[float, float]] = []
                    for jx in retry_steps:
                        for jy in retry_steps:
                            retry_pairs.append((float(jx), float(jy)))
                    retry_pairs.sort(key=lambda p: math.hypot(p[0], p[1]))
                    base_dx = float(offsets[0][0])
                    base_dy = float(offsets[0][1])
                    retry_hit = None
                    for jx, jy in retry_pairs:
                        if abs(jx) < 1e-9 and abs(jy) < 1e-9:
                            continue
                        cand = SpawnCandidate(
                            dx=base_dx + jx,
                            dy=base_dy + jy,
                            source="align_ego_piecewise_retry",
                            base_cost=math.hypot(jx, jy),
                        )
                        _try_spawn_candidate(
                            world=world,
                            world_map=world_map,
                            blueprint=ego_bp,
                            base_wp=traj[0],
                            cand=cand,
                            normalize_z=normalize_z,
                        )
                        if cand.valid:
                            retry_hit = (jx, jy, cand)
                            break
                    if retry_hit is not None:
                        jx, jy, cand_ok = retry_hit
                        updated_offsets: List[Tuple[float, float]] = []
                        for dx, dy in offsets:
                            ndx = float(dx) + float(jx)
                            ndy = float(dy) + float(jy)
                            mag = math.hypot(ndx, ndy)
                            if mag > max_shift > 0.0:
                                scale = max_shift / max(mag, 1e-6)
                                ndx *= scale
                                ndy *= scale
                            updated_offsets.append((ndx, ndy))
                        offsets = updated_offsets
                        chosen = cand_ok
                        entry["spawn_retry"] = {"dx": float(jx), "dy": float(jy)}

            shift_mags = [math.hypot(dx, dy) for dx, dy in offsets]
            for wp, (dx, dy) in zip(traj, offsets):
                wp.x += float(dx)
                wp.y += float(dy)
                wp.z += float(chosen.dz)

            if any(abs(dx) >= 1e-6 or abs(dy) >= 1e-6 for dx, dy in offsets) or abs(float(chosen.dz)) >= 1e-6:
                aligned += 1
            if abs(float(chosen.dz)) >= 1e-6:
                z_shifted += 1
            if chosen.valid:
                spawn_valid += 1

            entry["offset_stats"] = {
                "median": float(_median(shift_mags)) if shift_mags else 0.0,
                "max": float(max(shift_mags)) if shift_mags else 0.0,
            }
            entry["chosen"] = {
                "dx": float(offsets[0][0]),
                "dy": float(offsets[0][1]),
                "dz": float(chosen.dz),
                "z_source": chosen.z_source,
                "source": chosen.source,
                "base_cost": float(chosen.base_cost),
                "valid_spawn": bool(chosen.valid),
                "reason": chosen.reason,
                "mode": "piecewise",
            }
            entry["valid_candidates"] = len(offsets) if chosen.valid else 0
            if args.spawn_preprocess_verbose:
                print(
                    f"[SPAWN_PRE][EGO] ego{ego_idx} piecewise "
                    f"first_dx={offsets[0][0]:.3f} first_dy={offsets[0][1]:.3f} "
                    f"dz={chosen.dz:.3f} valid={chosen.valid} "
                    f"median_shift={(entry['offset_stats']['median']):.3f} "
                    f"max_shift={(entry['offset_stats']['max']):.3f}"
                )
        else:
            candidates, align_report = _build_alignment_candidates(
                traj=traj,
                times=times,
                role="npc",
                world_map=world_map,
                max_shift=max_shift,
                sample_count=sample_count,
                window_count=window_count,
                intent_margin=intent_margin,
            )
            entry["alignment"] = align_report
            entry["candidate_count"] = len(candidates)
            if not candidates:
                entry["status"] = "no_candidates"
                no_candidates += 1
                continue

            candidates.sort(key=lambda c: (float(c.base_cost), math.hypot(float(c.dx), float(c.dy))))
            if world is not None and ego_bp is not None:
                for cand in candidates:
                    _try_spawn_candidate(
                        world=world,
                        world_map=world_map,
                        blueprint=ego_bp,
                        base_wp=traj[0],
                        cand=cand,
                        normalize_z=normalize_z,
                    )
            valid = [c for c in candidates if c.valid]
            choose_from = valid if valid else candidates
            chosen = min(choose_from, key=lambda c: (float(c.base_cost), math.hypot(float(c.dx), float(c.dy))))

            if abs(float(chosen.dx)) >= 1e-6 or abs(float(chosen.dy)) >= 1e-6 or abs(float(chosen.dz)) >= 1e-6:
                aligned += 1
            if abs(float(chosen.dz)) >= 1e-6:
                z_shifted += 1
            if chosen.valid:
                spawn_valid += 1

            for wp in traj:
                wp.x += float(chosen.dx)
                wp.y += float(chosen.dy)
                wp.z += float(chosen.dz)

            entry["chosen"] = {
                "dx": float(chosen.dx),
                "dy": float(chosen.dy),
                "dz": float(chosen.dz),
                "z_source": chosen.z_source,
                "source": chosen.source,
                "base_cost": float(chosen.base_cost),
                "valid_spawn": bool(chosen.valid),
                "reason": chosen.reason,
                "mode": "global",
            }
            entry["valid_candidates"] = sum(1 for c in candidates if c.valid)
            if args.spawn_preprocess_verbose:
                print(
                    f"[SPAWN_PRE][EGO] ego{ego_idx} chosen "
                    f"dx={chosen.dx:.3f} dy={chosen.dy:.3f} dz={chosen.dz:.3f} "
                    f"src={chosen.source} valid={chosen.valid} cost={chosen.base_cost:.3f}"
                )

    out["summary"] = {
        "egos_considered": len(ego_trajs),
        "egos_aligned": aligned,
        "egos_z_shifted": z_shifted,
        "egos_no_candidates": no_candidates,
        "egos_spawn_valid": spawn_valid,
    }
    return out


def _build_alignment_neighbor_map(
    vehicles: Dict[int, List[Waypoint]],
    actor_meta: Dict[int, Dict[str, object]],
    radius: float,
    heading_tol_deg: float,
) -> Dict[int, List[Dict[str, object]]]:
    """Build neighbor lists for vehicles with similar heading near spawn time."""
    radius = max(0.0, float(radius))
    heading_tol_deg = max(0.0, float(heading_tol_deg))
    items: List[Tuple[int, float, float, float]] = []
    for vid, traj in vehicles.items():
        if not traj:
            continue
        meta = actor_meta.get(vid)
        if meta is None:
            continue
        kind = str(meta.get("kind") or "")
        if kind not in ("npc", "static"):
            continue
        wp = traj[0]
        items.append((vid, float(wp.x), float(wp.y), float(wp.yaw)))
    neighbors: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for i in range(len(items)):
        vid_i, xi, yi, yaw_i = items[i]
        hi = (math.cos(math.radians(yaw_i)), math.sin(math.radians(yaw_i)))
        for j in range(i + 1, len(items)):
            vid_j, xj, yj, yaw_j = items[j]
            dx = xj - xi
            dy = yj - yi
            dist = math.hypot(dx, dy)
            if dist > radius:
                continue
            yaw_diff = _yaw_diff_deg(yaw_i, yaw_j)
            if yaw_diff > heading_tol_deg:
                continue
            cross = hi[0] * dy - hi[1] * dx
            side_i = "left" if cross > 0 else "right"
            side_j = "left" if cross < 0 else "right"
            neighbors[vid_i].append({"id": vid_j, "dist": dist, "yaw_diff": yaw_diff, "side": side_i})
            neighbors[vid_j].append({"id": vid_i, "dist": dist, "yaw_diff": yaw_diff, "side": side_j})
    return neighbors


def _lane_type_value(name: str):
    return getattr(carla.LaneType, name, None) if carla is not None else None


def _generate_spawn_candidates(
    base_wp: Waypoint,
    role: str,
    world_map,
    max_shift: float,
    grid_steps: List[float],
    lateral_margin: float,
    random_samples: int = 0,
    rng: Optional[random.Random] = None,
) -> List[SpawnCandidate]:
    candidates: Dict[Tuple[int, int], SpawnCandidate] = {}

    def _add_candidate(dx: float, dy: float, source: str, bias: float) -> None:
        dist = math.hypot(dx, dy)
        if dist > max_shift:
            return
        key = (int(round(dx * 100)), int(round(dy * 100)))
        base_cost = dist + bias
        existing = candidates.get(key)
        if existing is None or base_cost < existing.base_cost:
            candidates[key] = SpawnCandidate(dx=dx, dy=dy, source=source, base_cost=base_cost)

    _add_candidate(0.0, 0.0, "authored", 0.0)

    # Micro-jitter grid around authored pose
    for dx in grid_steps:
        for dy in grid_steps:
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                continue
            _add_candidate(dx, dy, "grid", 0.05)

    # Radial rings to increase coverage without biasing too far
    if max_shift > 0.0:
        ring_radii = []
        step = 0.5
        r = step
        while r <= max_shift + 1e-6:
            ring_radii.append(round(r, 2))
            r += step
        angles = [i * 30 for i in range(12)]
        for r in ring_radii:
            for ang in angles:
                rad = math.radians(float(ang))
                dx = r * math.cos(rad)
                dy = r * math.sin(rad)
                _add_candidate(dx, dy, f"ring_{r:.1f}", 0.08 + 0.01 * r)

    if world_map is None:
        # Random offsets (if requested)
        if random_samples > 0:
            rng = rng or random.Random(0)
            for _ in range(random_samples):
                r = max_shift * math.sqrt(rng.random())
                theta = 2.0 * math.pi * rng.random()
                dx = r * math.cos(theta)
                dy = r * math.sin(theta)
                _add_candidate(dx, dy, "random", 0.2 + 0.01 * r)
        return list(candidates.values())

    loc = carla.Location(x=base_wp.x, y=base_wp.y, z=base_wp.z)

    lane_candidates: List[Tuple[str, object]] = []
    if role in ("npc", "static"):
        for lane_name in ("Driving", "Shoulder", "Parking"):
            lane_val = _lane_type_value(lane_name)
            if lane_val is not None:
                lane_candidates.append((lane_name, lane_val))
    else:
        for lane_name in ("Sidewalk", "Shoulder"):
            lane_val = _lane_type_value(lane_name)
            if lane_val is not None:
                lane_candidates.append((lane_name, lane_val))

    driving_wp = None
    for lane_name, lane_val in lane_candidates:
        try:
            wp = world_map.get_waypoint(loc, project_to_road=True, lane_type=lane_val)
        except Exception:
            wp = None
        if wp is None:
            continue
        dx = float(wp.transform.location.x) - base_wp.x
        dy = float(wp.transform.location.y) - base_wp.y
        _add_candidate(dx, dy, f"lane_{lane_name.lower()}", 0.1)
        if lane_name == "Driving":
            driving_wp = wp

    # Lateral offsets from driving lane (captures shoulder-like positions)
    if driving_wp is not None:
        try:
            yaw = math.radians(float(driving_wp.transform.rotation.yaw))
            right = (math.sin(yaw), -math.cos(yaw))
            lane_width = getattr(driving_wp, "lane_width", 3.5) or 3.5
            for mult in (0.5, 1.0, 1.5):
                offset = mult * float(lane_width) + float(lateral_margin)
                for sign in (-1.0, 1.0):
                    dx = float(driving_wp.transform.location.x) + sign * right[0] * offset - base_wp.x
                    dy = float(driving_wp.transform.location.y) + sign * right[1] * offset - base_wp.y
                    _add_candidate(dx, dy, f"lane_lateral_{mult:.1f}", 0.12 + 0.03 * mult)
            # along-lane offsets (forward/back)
            try:
                for dist in (0.5, 1.0, 2.0, 3.0, 4.0):
                    nxt = driving_wp.next(dist)
                    if nxt:
                        loc = nxt[0].transform.location
                        _add_candidate(loc.x - base_wp.x, loc.y - base_wp.y, "lane_forward", 0.15)
                    prev = driving_wp.previous(dist)
                    if prev:
                        loc = prev[0].transform.location
                        _add_candidate(loc.x - base_wp.x, loc.y - base_wp.y, "lane_backward", 0.15)
            except Exception:
                pass
        except Exception:
            pass

    # Random offsets (if requested)
    if random_samples > 0:
        rng = rng or random.Random(0)
        for _ in range(random_samples):
            r = max_shift * math.sqrt(rng.random())
            theta = 2.0 * math.pi * rng.random()
            dx = r * math.cos(theta)
            dy = r * math.sin(theta)
            _add_candidate(dx, dy, "random", 0.2 + 0.01 * r)

    return list(candidates.values())


def _connect_carla_for_spawn(
    host: str,
    port: int,
    expected_town: Optional[str],
):
    if carla is None:
        raise RuntimeError("carla module not available")
    _assert_carla_endpoint_reachable(host, port, timeout_s=2.0)
    client = carla.Client(host, port)
    client.set_timeout(30.0)
    world = client.get_world()
    cmap = world.get_map()
    if expected_town and expected_town not in (cmap.name or ""):
        available_maps = client.get_available_maps()
        candidates = [m for m in available_maps if expected_town in m]
        if candidates:
            target_map = candidates[0]
            print(f"[INFO] Loading map '{target_map}' for spawn preprocessing")
            world = client.load_world(target_map)
            cmap = world.get_map()
    return client, world, cmap


def _extend_grid_steps(base_steps: List[float], max_shift: float) -> List[float]:
    steps = list(base_steps)
    # Add finer steps up to 1.0
    for val in (0.6, 0.8, 1.0):
        steps.extend([val, -val])
    # Add coarser steps up to max_shift
    if max_shift > 1.0:
        step = 0.5
        v = 1.5
        while v <= max_shift + 1e-6:
            steps.extend([v, -v])
            v += step
    # Deduplicate and clamp
    uniq = []
    seen = set()
    for v in steps:
        if abs(v) > max_shift + 1e-6:
            continue
        key = round(float(v), 3)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(float(v))
    return uniq


def _select_blueprint(
    blueprint_lib,
    model: str,
    kind: str,
    obj_type_raw: str,
) -> Tuple[Optional[object], str, str]:
    """
    Return (blueprint, model_used, reason).
    Tries exact match, pattern match, then role-aware fallbacks.
    """
    if blueprint_lib is None:
        return None, model, "no_blueprint_lib"

    # Exact match
    try:
        bp = blueprint_lib.find(model)
        if bp is not None:
            if kind.startswith("walker") and _is_child_walker_blueprint(getattr(bp, "id", "")):
                bp = None
            else:
                return bp, model, "exact"
    except Exception:
        pass

    # Pattern match
    try:
        matches = blueprint_lib.filter(model)
        if matches:
            if kind.startswith("walker"):
                matches = [m for m in matches if not _is_child_walker_blueprint(getattr(m, "id", ""))]
            if matches:
                return matches[0], matches[0].id, "pattern"
    except Exception:
        pass

    obj_lower = str(obj_type_raw or "").lower()
    fallback_models: List[str] = []

    if kind.startswith("walker") or "pedestrian" in obj_lower or "walker" in obj_lower:
        fallback_models = list(ADULT_WALKER_BLUEPRINTS)[:3]
    elif "bicycle" in obj_lower or "cycl" in obj_lower:
        fallback_models = [
            "vehicle.diamondback.century",
            "vehicle.gazelle.omafiets",
            "vehicle.bh.crossbike",
        ]
    elif "motor" in obj_lower or "motorcycle" in obj_lower or ("bike" in obj_lower and "bicycle" not in obj_lower):
        fallback_models = [
            "vehicle.harley-davidson.low_rider",
            "vehicle.kawasaki.ninja",
            "vehicle.yamaha.yzf",
            "vehicle.vespa.zx125",
        ]
    elif "bus" in obj_lower:
        fallback_models = [
            "vehicle.volkswagen.t2",
            "vehicle.mitsubishi.fusorosa",
        ]
    elif "truck" in obj_lower:
        fallback_models = [
            "vehicle.carlamotors.carlacola",
        ]
    elif "van" in obj_lower or "sprinter" in obj_lower:
        fallback_models = [
            "vehicle.mercedes.sprinter",
            "vehicle.volkswagen.t2",
        ]
    elif "ambulance" in obj_lower:
        fallback_models = [
            "vehicle.ford.ambulance",
        ]
    elif "police" in obj_lower:
        fallback_models = [
            "vehicle.dodge.charger_police",
            "vehicle.dodge.charger_police_2020",
        ]
    else:
        fallback_models = [
            "vehicle.tesla.model3",
            "vehicle.audi.a2",
            "vehicle.lincoln.mkz_2017",
            "vehicle.nissan.micra",
        ]

    for fallback in fallback_models:
        try:
            bp = blueprint_lib.find(fallback)
            if bp is not None and not (kind.startswith("walker") and _is_child_walker_blueprint(getattr(bp, "id", ""))):
                return bp, fallback, "fallback"
        except Exception:
            pass
        try:
            matches = blueprint_lib.filter(fallback)
            if matches:
                if kind.startswith("walker"):
                    matches = [m for m in matches if not _is_child_walker_blueprint(getattr(m, "id", ""))]
                if matches:
                    return matches[0], matches[0].id, "fallback_pattern"
        except Exception:
            pass

    # Final generic fallback
    try:
        if kind.startswith("walker"):
            matches = blueprint_lib.filter("walker.pedestrian.*")
            if matches:
                matches = [m for m in matches if not _is_child_walker_blueprint(getattr(m, "id", ""))]
                if matches:
                    return matches[0], matches[0].id, "fallback_any_walker"
    except Exception:
        pass
    try:
        matches = blueprint_lib.filter("vehicle.*")
        if matches:
            return matches[0], matches[0].id, "fallback_any_vehicle"
    except Exception:
        pass

    return None, model, "missing_blueprint"


def _preprocess_spawn_positions(
    vehicles: Dict[int, List[Waypoint]],
    vehicle_times: Dict[int, List[float]],
    actor_meta: Dict[int, Dict[str, object]],
    args: argparse.Namespace,
    ego_trajs: Sequence[List[Waypoint]] | None = None,
    ego_times_list: Sequence[List[float]] | None = None,
) -> Dict[str, object]:
    report: Dict[str, object] = {
        "settings": {},
        "actors": {},
        "summary": {},
    }

    if carla is None:
        print("[WARN] spawn preprocess requested but CARLA Python module is unavailable; skipping.")
        report["summary"]["status"] = "skipped_no_carla"
        return report

    try:
        client, world, world_map = _connect_carla_for_spawn(
            host=args.carla_host,
            port=args.carla_port,
            expected_town=args.expected_town,
        )
    except Exception as exc:
        print(f"[WARN] spawn preprocess failed to connect to CARLA: {exc}")
        report["summary"]["status"] = "skipped_carla_connect"
        return report

    blueprint_lib = world.get_blueprint_library() if world else None

    existing_actors = 0
    cleared_actors = 0
    try:
        existing_actors = len(world.get_actors()) if world else 0
    except Exception:
        existing_actors = 0

    # Clear dynamic actors to reduce spawn-test interference.
    if world is not None and existing_actors:
        try:
            to_destroy = []
            for actor in world.get_actors():
                try:
                    tid = actor.type_id or ""
                except Exception:
                    tid = ""
                if (
                    tid.startswith("vehicle.")
                    or tid.startswith("walker.")
                    or tid.startswith("sensor.")
                    or tid.startswith("controller.ai.")
                ):
                    to_destroy.append(actor.id)
            if to_destroy:
                try:
                    if hasattr(carla, "command") and client is not None:
                        commands = [carla.command.DestroyActor(aid) for aid in to_destroy]
                        client.apply_batch_sync(commands, True)
                    else:
                        for actor in world.get_actors(to_destroy):
                            try:
                                actor.destroy()
                            except Exception:
                                pass
                except Exception:
                    for actor in world.get_actors(to_destroy):
                        try:
                            actor.destroy()
                        except Exception:
                            pass
                cleared_actors = len(to_destroy)
                try:
                    settings = world.get_settings()
                    if getattr(settings, "synchronous_mode", False):
                        world.tick()
                    else:
                        world.wait_for_tick()
                except Exception:
                    pass
        except Exception:
            pass

    if cleared_actors:
        print(f"[SPAWN_PRE] Cleared {cleared_actors} dynamic actors from CARLA world before spawn checks.")
    elif existing_actors > 10:
        print(f"[WARN] CARLA world has {existing_actors} existing actors; spawn tests may be affected.")

    max_shift = max(0.0, float(args.spawn_preprocess_max_shift))
    grid_steps = []
    for token in re.split(r"[,\s]+", str(args.spawn_preprocess_grid or "").strip()):
        if not token:
            continue
        try:
            grid_steps.append(float(token))
        except Exception:
            continue
    if not grid_steps:
        grid_steps = [0.0, 0.2, -0.2, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2]
    grid_steps = _extend_grid_steps(grid_steps, max_shift)
    lateral_margin = 0.6

    sample_dt = float(args.spawn_preprocess_sample_dt)
    grid_size = max(1.0, float(args.spawn_preprocess_grid_size))
    max_candidates = max(5, int(args.spawn_preprocess_max_candidates))
    collision_weight = float(args.spawn_preprocess_collision_weight)
    normalize_z = bool(args.spawn_preprocess_normalize_z)
    random_samples = max(0, int(args.spawn_preprocess_random_samples))
    debug_radius = float(args.spawn_preprocess_debug_radius)
    debug_max_items = int(args.spawn_preprocess_debug_max_items)
    align_enabled = bool(args.spawn_preprocess_align)
    align_samples = int(args.spawn_preprocess_align_samples)
    align_windows = int(args.spawn_preprocess_align_windows)
    align_intent_margin = float(args.spawn_preprocess_align_intent_margin)
    align_neighbor_radius = float(args.spawn_preprocess_align_neighbor_radius)
    align_neighbor_weight = float(args.spawn_preprocess_align_neighbor_weight)
    refine_piecewise = bool(getattr(args, "spawn_preprocess_refine_piecewise", True))
    refine_max_local = float(getattr(args, "spawn_preprocess_refine_max_local", 0.8))
    refine_smooth_window = int(getattr(args, "spawn_preprocess_refine_smooth_window", 7))
    refine_max_step_delta = float(getattr(args, "spawn_preprocess_refine_max_step_delta", 0.35))
    refine_collision_slack = float(getattr(args, "spawn_preprocess_refine_collision_slack", 0.0))
    bridge_max_gap_steps = int(getattr(args, "spawn_preprocess_bridge_max_gap_steps", 6))
    bridge_straight_thresh_deg = float(getattr(args, "spawn_preprocess_bridge_straight_thresh_deg", 18.0))

    all_times: List[float] = []
    for vid, traj in vehicles.items():
        meta = actor_meta.get(vid)
        if meta is None:
            continue
        times = _ensure_times(traj, vehicle_times.get(vid), args.dt)
        all_times.extend(times)
    sample_times = _build_time_grid(all_times, sample_dt)

    report["settings"] = {
        "max_shift": max_shift,
        "grid_steps": grid_steps,
        "sample_dt": sample_dt,
        "grid_size": grid_size,
        "max_candidates": max_candidates,
        "collision_weight": collision_weight,
        "normalize_z": normalize_z,
        "random_samples": random_samples,
        "debug_radius": debug_radius,
        "debug_max_items": debug_max_items,
        "cleared_dynamic_actors": cleared_actors,
        "sample_times": len(sample_times),
        "align_enabled": align_enabled,
        "align_samples": align_samples,
        "align_windows": align_windows,
        "align_intent_margin": align_intent_margin,
        "align_neighbor_radius": align_neighbor_radius,
        "align_neighbor_weight": align_neighbor_weight,
        "align_ego_enabled": bool(getattr(args, "spawn_preprocess_align_ego", True)),
        "align_ego_piecewise": bool(getattr(args, "spawn_preprocess_align_ego_piecewise", True)),
        "snap_ego_to_lane": bool(getattr(args, "snap_ego_to_lane", False)),
        "align_ego_smooth_window": int(getattr(args, "spawn_preprocess_align_ego_smooth_window", 9)),
        "align_ego_max_step_delta": float(getattr(args, "spawn_preprocess_align_ego_max_step_delta", 0.45)),
        "refine_piecewise": refine_piecewise,
        "refine_max_local": refine_max_local,
        "refine_smooth_window": refine_smooth_window,
        "refine_max_step_delta": refine_max_step_delta,
        "refine_collision_slack": refine_collision_slack,
        "bridge_max_gap_steps": bridge_max_gap_steps,
        "bridge_straight_thresh_deg": bridge_straight_thresh_deg,
    }

    # Precompute base positions and radii
    base_positions: Dict[int, List[Optional[Tuple[float, float]]]] = {}
    radii: Dict[int, float] = {}
    times_cache: Dict[int, List[float]] = {}
    for vid, traj in vehicles.items():
        meta = actor_meta.get(vid)
        if meta is None:
            continue
        times = _ensure_times(traj, vehicle_times.get(vid), args.dt)
        times_cache[vid] = times
        kind = str(meta.get("kind"))
        always_active = kind in ("static", "walker_static")
        base_positions[vid] = _sample_positions(traj, times, sample_times, always_active)
        radii[vid] = _actor_radius(
            kind,
            meta.get("length"),
            meta.get("width"),
            str(meta.get("model", "")),
        )

    # Cache world actors/env objects for debug
    actor_items: List[Dict[str, object]] = []
    env_items: List[Dict[str, object]] = []
    if world is not None:
        try:
            for actor in world.get_actors():
                try:
                    loc = actor.get_location()
                    tf = actor.get_transform()
                    bbox = actor.bounding_box
                except Exception:
                    continue
                actor_items.append(
                    {
                        "id": int(actor.id),
                        "type": getattr(actor, "type_id", "actor"),
                        "loc": loc,
                        "bbox": _bbox_corners_2d(bbox, tf),
                    }
                )
        except Exception:
            pass
        try:
            label_any = getattr(carla.CityObjectLabel, "Any", None)
            env_objs = world.get_environment_objects(label_any) if label_any is not None else world.get_environment_objects()
            for env in env_objs:
                try:
                    tf = env.transform
                    loc = tf.location
                    bbox = env.bounding_box
                except Exception:
                    continue
                env_items.append(
                    {
                        "id": int(getattr(env, "id", -1)),
                        "type": getattr(env, "type_id", getattr(env, "type", "env")),
                        "loc": loc,
                        "bbox": _bbox_corners_2d(bbox, tf),
                    }
                )
        except Exception:
            pass

    # Build candidate lists with spawn validity
    candidates_by_actor: Dict[int, List[SpawnCandidate]] = {}
    bp_by_actor: Dict[int, object] = {}
    for vid, traj in vehicles.items():
        meta = actor_meta.get(vid)
        if meta is None or not traj:
            continue
        kind = str(meta.get("kind"))
        role = "npc" if kind in ("npc", "static") else "walker"
        model = str(meta.get("model") or "")
        actor_report = {
            "kind": kind,
            "model": model,
            "model_used": model,
            "candidates": [],
            "chosen": None,
        }
        report["actors"][str(vid)] = actor_report

        base_wp = traj[0]
        candidates = _generate_spawn_candidates(
            base_wp=base_wp,
            role=role,
            world_map=world_map,
            max_shift=max_shift,
            grid_steps=grid_steps,
            lateral_margin=lateral_margin,
            random_samples=random_samples,
            rng=random.Random(vid),
        )
        # Alignment-based candidates (multi-waypoint intent-aware offsets)
        if align_enabled:
            times = times_cache.get(vid) or _ensure_times(traj, vehicle_times.get(vid), args.dt)
            align_candidates, align_report = _build_alignment_candidates(
                traj=traj,
                times=times,
                role=role,
                world_map=world_map,
                max_shift=max_shift,
                sample_count=align_samples,
                window_count=align_windows,
                intent_margin=align_intent_margin,
            )
            if align_report:
                actor_report["alignment"] = align_report
            if align_candidates:
                merged: Dict[Tuple[int, int], SpawnCandidate] = {}
                for cand in candidates:
                    key = (int(round(cand.dx * 100)), int(round(cand.dy * 100)))
                    merged[key] = cand
                for cand in align_candidates:
                    key = (int(round(cand.dx * 100)), int(round(cand.dy * 100)))
                    existing = merged.get(key)
                    if existing is None or cand.base_cost < existing.base_cost:
                        merged[key] = cand
                candidates = list(merged.values())

        # Sort by base_cost and keep the best candidates first, ensuring alignment candidates stay.
        if candidates:
            align_list = [c for c in candidates if str(c.source).startswith("align_")]
            other_list = [c for c in candidates if c not in align_list]
            other_list.sort(key=lambda c: c.base_cost)
            keep_count = max(0, max_candidates - len(align_list))
            candidates = align_list + other_list[:keep_count]
        else:
            candidates = []

        if blueprint_lib is None:
            print(f"[WARN] Blueprint library unavailable; skipping spawn validation for actor {vid}.")
            for cand in candidates:
                cand.valid = True
                cand.reason = "no_blueprint_lib"
                cand.spawn_loc = (float(base_wp.x + cand.dx), float(base_wp.y + cand.dy), float(base_wp.z))
                cand.dz = 0.0
                cand.z_source = "authored"
            candidates_by_actor[vid] = candidates
            src_counts = {}
            for c in candidates:
                src_counts[c.source] = src_counts.get(c.source, 0) + 1
            actor_report["candidates"] = [
                {
                    "dx": c.dx,
                    "dy": c.dy,
                    "source": c.source,
                    "valid": c.valid,
                    "reason": c.reason,
                    "base_cost": c.base_cost,
                    "spawn_loc": c.spawn_loc,
                    "dz": c.dz,
                    "z_source": c.z_source,
                    "align_stats": c.align_stats,
                }
                for c in candidates
            ]
            actor_report["candidate_stats"] = {
                "total": len(candidates),
                "valid": len(candidates),
                "invalid": 0,
                "source_counts": src_counts,
                "failure_reasons": {},
            }
            actor_report["spawn_base"] = {
                "x": float(base_wp.x),
                "y": float(base_wp.y),
                "z": float(base_wp.z),
                "yaw": float(base_wp.yaw),
            }
            continue

        bp, model_used, reason = _select_blueprint(blueprint_lib, model, kind, str(meta.get("obj_type") or ""))
        actor_report["model_used"] = model_used
        actor_report["blueprint_reason"] = reason
        if bp is None:
            print(f"[WARN] No blueprint found for actor {vid} model '{model}'; leaving trajectory unchanged.")
            candidates_by_actor[vid] = []
            src_counts = {}
            for c in candidates:
                src_counts[c.source] = src_counts.get(c.source, 0) + 1
            actor_report["candidates"] = [
                {
                    "dx": c.dx,
                    "dy": c.dy,
                    "source": c.source,
                    "valid": False,
                    "reason": "missing_blueprint",
                    "base_cost": c.base_cost,
                    "spawn_loc": (float(base_wp.x + c.dx), float(base_wp.y + c.dy), float(base_wp.z)),
                    "dz": 0.0,
                    "z_source": "authored",
                    "align_stats": c.align_stats,
                }
                for c in candidates
            ]
            actor_report["status"] = "missing_blueprint"
            actor_report["candidate_stats"] = {
                "total": len(candidates),
                "valid": 0,
                "invalid": len(candidates),
                "source_counts": src_counts,
                "failure_reasons": {"missing_blueprint": len(candidates)} if candidates else {},
            }
            actor_report["spawn_base"] = {
                "x": float(base_wp.x),
                "y": float(base_wp.y),
                "z": float(base_wp.z),
                "yaw": float(base_wp.yaw),
            }
            continue
        bp_by_actor[vid] = bp
        if model_used and model_used != model:
            print(f"[WARN] Blueprint '{model}' unavailable; using '{model_used}' for actor {vid}.")
            meta["model"] = model_used
            model = model_used
            actor_report["model"] = model_used

        for cand in candidates:
            _try_spawn_candidate(world, world_map, bp, base_wp, cand, normalize_z)

        # If no valid candidates, expand search once more (try harder)
        if not any(c.valid for c in candidates):
            hard_max_shift = min(max_shift * 2.0, max_shift + 4.0)
            hard_grid_steps = _extend_grid_steps(grid_steps, hard_max_shift)
            hard_candidates = _generate_spawn_candidates(
                base_wp=base_wp,
                role=role,
                world_map=world_map,
                max_shift=hard_max_shift,
                grid_steps=hard_grid_steps,
                lateral_margin=lateral_margin,
                random_samples=random_samples * 2,
                rng=random.Random(vid + 100000),
            )
            hard_candidates.sort(key=lambda c: c.base_cost)
            hard_candidates = hard_candidates[: max_candidates * 3]
            for cand in hard_candidates:
                _try_spawn_candidate(world, world_map, bp, base_wp, cand, normalize_z)
            # merge candidates (keep best cost per offset)
            merged: Dict[Tuple[int, int], SpawnCandidate] = {}
            for cand in candidates + hard_candidates:
                key = (int(round(cand.dx * 100)), int(round(cand.dy * 100)))
                existing = merged.get(key)
                if existing is None or cand.base_cost < existing.base_cost:
                    merged[key] = cand
            candidates = sorted(merged.values(), key=lambda c: c.base_cost)
            candidates = candidates[: max_candidates * 2]

        candidates_by_actor[vid] = candidates
        valid_count = sum(1 for c in candidates if c.valid)
        invalid_count = len(candidates) - valid_count
        reasons: Dict[str, int] = {}
        sources: Dict[str, int] = {}
        for c in candidates:
            sources[c.source] = sources.get(c.source, 0) + 1
            if not c.valid:
                key = c.reason or "spawn_failed"
                reasons[key] = reasons.get(key, 0) + 1
        actor_report["candidate_stats"] = {
            "total": len(candidates),
            "valid": valid_count,
            "invalid": invalid_count,
            "source_counts": sources,
            "failure_reasons": reasons,
        }
        actor_report["spawn_base"] = {
            "x": float(base_wp.x),
            "y": float(base_wp.y),
            "z": float(base_wp.z),
            "yaw": float(base_wp.yaw),
        }
        actor_report["candidates"] = [
            {
                "dx": c.dx,
                "dy": c.dy,
                "source": c.source,
                "valid": c.valid,
                "reason": c.reason,
                "base_cost": c.base_cost,
                "spawn_loc": c.spawn_loc,
                "dz": c.dz,
                "z_source": c.z_source,
                "align_stats": c.align_stats,
            }
            for c in candidates
        ]

    neighbor_map: Dict[int, List[Dict[str, object]]] = {}
    if align_enabled and align_neighbor_weight > 0.0 and align_neighbor_radius > 0.0:
        neighbor_map = _build_alignment_neighbor_map(
            vehicles=vehicles,
            actor_meta=actor_meta,
            radius=align_neighbor_radius,
            heading_tol_deg=25.0,
        )
        report["settings"]["align_neighbor_pairs"] = sum(len(v) for v in neighbor_map.values())
        for vid, infos in neighbor_map.items():
            entry = report["actors"].get(str(vid))
            if entry is not None and isinstance(entry, dict):
                align_entry = entry.get("alignment")
                if isinstance(align_entry, dict):
                    align_entry["neighbors"] = infos

    # Global assignment with spatiotemporal collision avoidance
    occupancy: List[Dict[Tuple[int, int], List[Tuple[float, float, float, int]]]] = [
        defaultdict(list) for _ in sample_times
    ]

    def _collision_score(vid: int, cand: SpawnCandidate) -> float:
        positions = base_positions.get(vid, [])
        radius = radii.get(vid, 1.0)
        score = 0.0
        for t_idx, pos in enumerate(positions):
            if pos is None:
                continue
            x = pos[0] + cand.dx
            y = pos[1] + cand.dy
            cell_x = int(math.floor(x / grid_size))
            cell_y = int(math.floor(y / grid_size))
            cell_map = occupancy[t_idx]
            for gx in range(cell_x - 1, cell_x + 2):
                for gy in range(cell_y - 1, cell_y + 2):
                    for ox, oy, orad, oid in cell_map.get((gx, gy), []):
                        dist = math.hypot(x - ox, y - oy)
                        if dist < (radius + orad):
                            score += 1.0
        return score

    chosen_offsets: Dict[int, SpawnCandidate] = {}
    actor_order = sorted(
        [vid for vid in vehicles.keys() if vid in candidates_by_actor],
        key=lambda vid: len([c for c in candidates_by_actor[vid] if c.valid]) or 9999,
    )

    for vid in actor_order:
        meta = actor_meta.get(vid)
        if meta is None:
            continue
        cands = [c for c in candidates_by_actor.get(vid, []) if c.valid]
        if not cands:
            # fallback to no shift
            fallback = SpawnCandidate(
                dx=0.0,
                dy=0.0,
                source="fallback",
                base_cost=0.0,
                valid=False,
                reason="no_valid_candidates",
                dz=0.0,
                z_source="fallback",
            )
            chosen_offsets[vid] = fallback
            report["actors"][str(vid)]["chosen"] = {
                "dx": 0.0,
                "dy": 0.0,
                "dz": 0.0,
                "z_source": "fallback",
                "source": "fallback",
                "collision_score": None,
                "status": "no_valid_candidates",
            }
            cand_all = candidates_by_actor.get(vid, [])
            if cand_all:
                best_invalid = min(cand_all, key=lambda c: c.base_cost)
                report["actors"][str(vid)]["best_invalid_candidate"] = {
                    "dx": best_invalid.dx,
                    "dy": best_invalid.dy,
                    "source": best_invalid.source,
                    "base_cost": best_invalid.base_cost,
                    "reason": best_invalid.reason,
                    "spawn_loc": best_invalid.spawn_loc,
                    "dz": best_invalid.dz,
                    "z_source": best_invalid.z_source,
                }
            # add debug info for failed spawns
            base_wp = vehicles.get(vid, [None])[0]
            entry = report["actors"].get(str(vid), {})
            if base_wp is not None:
                bp = bp_by_actor.get(vid)
                probe_yaw = str(entry.get("kind", "")).startswith("npc") or str(entry.get("kind", "")).startswith("static")
                entry["debug"] = _collect_spawn_debug(
                    actor_id=vid,
                    base_wp=base_wp,
                    entry=entry,
                    world=world,
                    world_map=world_map,
                    blueprint=bp,
                    actor_items=actor_items,
                    env_items=env_items,
                    max_dist=debug_radius,
                    max_items=debug_max_items,
                    probe_yaw=probe_yaw,
                )
            continue

        best = None
        best_score = None
        best_collision = None
        best_neighbor = None
        for cand in cands:
            collision = _collision_score(vid, cand)
            neighbor_penalty = 0.0
            if neighbor_map and align_neighbor_weight > 0.0:
                offsets = []
                for info in neighbor_map.get(vid, []):
                    nid = int(info.get("id", -1))
                    chosen = chosen_offsets.get(nid)
                    if chosen is None:
                        continue
                    offsets.append((float(chosen.dx), float(chosen.dy)))
                if offsets:
                    avg_dx = sum(o[0] for o in offsets) / float(len(offsets))
                    avg_dy = sum(o[1] for o in offsets) / float(len(offsets))
                    neighbor_penalty = align_neighbor_weight * math.hypot(cand.dx - avg_dx, cand.dy - avg_dy)
            total = cand.base_cost + collision_weight * collision + neighbor_penalty
            if best_score is None or total < best_score:
                best = cand
                best_score = total
                best_collision = collision
                best_neighbor = neighbor_penalty

        if best is None:
            continue

        chosen_offsets[vid] = best
        report["actors"][str(vid)]["chosen"] = {
            "dx": best.dx,
            "dy": best.dy,
            "dz": best.dz,
            "z_source": best.z_source,
            "source": best.source,
            "collision_score": best_collision,
            "neighbor_penalty": best_neighbor,
            "status": "ok",
        }

        # Update occupancy
        positions = base_positions.get(vid, [])
        radius = radii.get(vid, 1.0)
        for t_idx, pos in enumerate(positions):
            if pos is None:
                continue
            x = pos[0] + best.dx
            y = pos[1] + best.dy
            cell = (int(math.floor(x / grid_size)), int(math.floor(y / grid_size)))
            occupancy[t_idx][cell].append((x, y, radius, vid))

        if args.spawn_preprocess_verbose:
            print(
                f"[SPAWN_PRE] actor {vid} kind={meta.get('kind')} model={meta.get('model')} "
                f"chosen dx={best.dx:.3f} dy={best.dy:.3f} dz={best.dz:.3f} "
                f"z_src={best.z_source} source={best.source} "
                f"collision={best_collision} neighbor={best_neighbor}"
            )

    # Optional second-stage refinement:
    #   1) keep globally optimized shift (collision-aware) as the anchor
    #   2) compute bounded per-waypoint local refinement around that anchor
    #   3) accept only if collision score does not get worse beyond slack
    piecewise_profiles: Dict[int, List[Tuple[float, float]]] = {}
    refined_accepted = 0
    refined_rejected = 0
    if refine_piecewise and world_map is not None and chosen_offsets:
        occupancy_const_all: List[Dict[Tuple[int, int], List[Tuple[float, float, float, int]]]] = [
            defaultdict(list) for _ in sample_times
        ]
        for oid, ocand in chosen_offsets.items():
            opos = base_positions.get(oid, [])
            orad = radii.get(oid, 1.0)
            for t_idx, pos in enumerate(opos):
                if pos is None:
                    continue
                x = pos[0] + float(ocand.dx)
                y = pos[1] + float(ocand.dy)
                cell = (int(math.floor(x / grid_size)), int(math.floor(y / grid_size)))
                occupancy_const_all[t_idx][cell].append((x, y, orad, oid))

        def _profile_collision_score(
            actor_id: int,
            sample_offsets: List[Optional[Tuple[float, float]]],
        ) -> float:
            score = 0.0
            apos = base_positions.get(actor_id, [])
            arad = radii.get(actor_id, 1.0)
            for t_idx, pos in enumerate(apos):
                if pos is None:
                    continue
                off = sample_offsets[t_idx] if t_idx < len(sample_offsets) else None
                if off is None:
                    continue
                x = float(pos[0]) + float(off[0])
                y = float(pos[1]) + float(off[1])
                cell_x = int(math.floor(x / grid_size))
                cell_y = int(math.floor(y / grid_size))
                cell_map = occupancy_const_all[t_idx]
                for gx in range(cell_x - 1, cell_x + 2):
                    for gy in range(cell_y - 1, cell_y + 2):
                        for ox, oy, orad, oid in cell_map.get((gx, gy), []):
                            if int(oid) == int(actor_id):
                                continue
                            if math.hypot(x - ox, y - oy) < (arad + orad):
                                score += 1.0
            return score

        for vid, cand in chosen_offsets.items():
            meta = actor_meta.get(vid)
            if meta is None:
                continue
            if not cand.valid:
                continue
            traj = vehicles.get(vid)
            if not traj or len(traj) < 3:
                continue
            times = times_cache.get(vid) or _ensure_times(traj, vehicle_times.get(vid), args.dt)
            kind = str(meta.get("kind") or "")
            role = "npc" if kind in ("npc", "static") else "walker"
            always_active = kind in ("static", "walker_static")

            refined_offsets, refine_report = _compute_piecewise_actor_offsets(
                traj=traj,
                times=times,
                world_map=world_map,
                role=role,
                base_dx=float(cand.dx),
                base_dy=float(cand.dy),
                max_shift=max_shift,
                intent_margin=align_intent_margin,
                local_limit=refine_max_local,
                smooth_window=refine_smooth_window,
                max_step_delta=refine_max_step_delta,
                bridge_max_gap_steps=bridge_max_gap_steps,
                bridge_straight_thresh_deg=bridge_straight_thresh_deg,
            )
            entry = report["actors"].get(str(vid), {})
            entry["refinement"] = refine_report
            if not refined_offsets:
                entry["refinement"]["accepted"] = False
                entry["refinement"]["reason"] = "no_offsets"
                refined_rejected += 1
                continue

            base_profile = _sample_offset_profile(
                times=times,
                offsets_wp=[(float(cand.dx), float(cand.dy)) for _ in traj],
                sample_times=sample_times,
                always_active=always_active,
            )
            refined_profile = _sample_offset_profile(
                times=times,
                offsets_wp=refined_offsets,
                sample_times=sample_times,
                always_active=always_active,
            )
            base_collision = _profile_collision_score(vid, base_profile)
            refined_collision = _profile_collision_score(vid, refined_profile)
            entry["refinement"]["base_collision"] = float(base_collision)
            entry["refinement"]["refined_collision"] = float(refined_collision)
            entry["refinement"]["collision_slack"] = float(refine_collision_slack)

            # Ensure first spawn remains valid for refined profile.
            spawn_ok = True
            if world is not None and vid in bp_by_actor and refined_offsets:
                spawn_probe = SpawnCandidate(
                    dx=float(refined_offsets[0][0]),
                    dy=float(refined_offsets[0][1]),
                    source="refine_probe",
                    base_cost=0.0,
                )
                _try_spawn_candidate(
                    world=world,
                    world_map=world_map,
                    blueprint=bp_by_actor[vid],
                    base_wp=traj[0],
                    cand=spawn_probe,
                    normalize_z=normalize_z,
                )
                spawn_ok = bool(spawn_probe.valid)
                entry["refinement"]["spawn_valid"] = bool(spawn_ok)
                entry["refinement"]["spawn_probe_reason"] = spawn_probe.reason

            if (not spawn_ok) or (refined_collision > base_collision + float(refine_collision_slack)):
                entry["refinement"]["accepted"] = False
                entry["refinement"]["reason"] = "spawn_invalid" if not spawn_ok else "collision_regression"
                refined_rejected += 1
                continue

            piecewise_profiles[vid] = refined_offsets
            entry["refinement"]["accepted"] = True
            refined_accepted += 1
            if args.spawn_preprocess_verbose:
                print(
                    f"[SPAWN_PRE][REFINE] actor {vid} accepted "
                    f"base_col={base_collision:.1f} refined_col={refined_collision:.1f}"
                )

    # Apply offsets to trajectories
    total_shifted = 0
    total_z_shifted = 0
    for vid, cand in chosen_offsets.items():
        profile = piecewise_profiles.get(vid)
        if profile is None and abs(cand.dx) < 1e-6 and abs(cand.dy) < 1e-6 and abs(cand.dz) < 1e-6:
            continue
        traj = vehicles.get(vid)
        if not traj:
            continue
        if profile is None:
            for wp in traj:
                wp.x += cand.dx
                wp.y += cand.dy
                wp.z += cand.dz
        else:
            for idx, wp in enumerate(traj):
                dx, dy = profile[min(idx, len(profile) - 1)]
                wp.x += float(dx)
                wp.y += float(dy)
                wp.z += cand.dz
            entry = report["actors"].get(str(vid))
            if isinstance(entry, dict):
                chosen_entry = entry.get("chosen")
                if isinstance(chosen_entry, dict):
                    chosen_entry["mode"] = "global_plus_piecewise_refine"
                    chosen_entry["first_dx"] = float(profile[0][0])
                    chosen_entry["first_dy"] = float(profile[0][1])
                    chosen_entry["last_dx"] = float(profile[-1][0])
                    chosen_entry["last_dy"] = float(profile[-1][1])
        total_shifted += 1
        if abs(cand.dz) >= 1e-6:
            total_z_shifted += 1

    ego_report: Dict[str, object] | None = None
    ego_summary: Dict[str, object] = {
        "egos_considered": 0,
        "egos_aligned": 0,
        "egos_z_shifted": 0,
        "egos_no_candidates": 0,
        "egos_spawn_valid": 0,
    }
    if bool(getattr(args, "spawn_preprocess_align_ego", True)):
        ego_report = _align_ego_trajectories(
            ego_trajs=ego_trajs,
            ego_times_list=ego_times_list,
            world=world,
            world_map=world_map,
            blueprint_lib=blueprint_lib,
            args=args,
        )
        report["ego"] = ego_report
        if isinstance(ego_report.get("summary"), dict):
            ego_summary = dict(ego_report["summary"])

    missing_bp = 0
    no_valid = 0
    fallback_bp = 0
    valid_counts: List[int] = []
    reason_totals: Dict[str, int] = {}
    source_totals: Dict[str, int] = {}
    bridge_missing_total = 0
    bridge_transient_total = 0
    for entry in report.get("actors", {}).values():
        if entry.get("status") == "missing_blueprint":
            missing_bp += 1
        chosen = entry.get("chosen") or {}
        if chosen.get("status") == "no_valid_candidates":
            no_valid += 1
        reason = str(entry.get("blueprint_reason") or "")
        if reason.startswith("fallback"):
            fallback_bp += 1
        stats = entry.get("candidate_stats") or {}
        if stats:
            valid_counts.append(int(stats.get("valid", 0)))
            for src, cnt in (stats.get("source_counts") or {}).items():
                source_totals[src] = source_totals.get(src, 0) + int(cnt)
            for r, cnt in (stats.get("failure_reasons") or {}).items():
                reason_totals[r] = reason_totals.get(r, 0) + int(cnt)
        refinement = entry.get("refinement")
        if isinstance(refinement, dict):
            bridge = refinement.get("bridge")
            if isinstance(bridge, dict):
                bridge_missing_total += int(bridge.get("bridged_missing", 0))
                bridge_transient_total += int(bridge.get("bridged_transient", 0))

    ego_bridge_missing = 0
    ego_bridge_transient = 0
    if isinstance(ego_report, dict):
        for ego_entry in ego_report.get("egos", []):
            if not isinstance(ego_entry, dict):
                continue
            alignment = ego_entry.get("alignment")
            if not isinstance(alignment, dict):
                continue
            bridge = alignment.get("bridge")
            if not isinstance(bridge, dict):
                continue
            ego_bridge_missing += int(bridge.get("bridged_missing", 0))
            ego_bridge_transient += int(bridge.get("bridged_transient", 0))

    report["summary"] = {
        "status": "ok",
        "actors_considered": len(actor_meta),
        "actors_shifted": total_shifted,
        "actors_z_shifted": total_z_shifted,
        "actors_missing_blueprint": missing_bp,
        "actors_no_valid_candidates": no_valid,
        "actors_with_fallback_blueprint": fallback_bp,
        "candidate_valid_counts": {
            "min": min(valid_counts) if valid_counts else 0,
            "max": max(valid_counts) if valid_counts else 0,
            "avg": (sum(valid_counts) / max(1, len(valid_counts))) if valid_counts else 0.0,
        },
        "candidate_source_totals": source_totals,
        "candidate_failure_reasons": reason_totals,
        "actors_refined_piecewise_accepted": int(refined_accepted),
        "actors_refined_piecewise_rejected": int(refined_rejected),
        "actors_bridged_missing_steps": int(bridge_missing_total),
        "actors_bridged_transient_steps": int(bridge_transient_total),
        "egos_considered": int(ego_summary.get("egos_considered", 0)),
        "egos_aligned": int(ego_summary.get("egos_aligned", 0)),
        "egos_z_shifted": int(ego_summary.get("egos_z_shifted", 0)),
        "egos_no_candidates": int(ego_summary.get("egos_no_candidates", 0)),
        "egos_spawn_valid": int(ego_summary.get("egos_spawn_valid", 0)),
        "egos_bridged_missing_steps": int(ego_bridge_missing),
        "egos_bridged_transient_steps": int(ego_bridge_transient),
    }
    ok_count = max(0, len(actor_meta) - missing_bp - no_valid)
    print(
        "[SPAWN_PRE] Summary: "
        f"actors={len(actor_meta)} ok={ok_count} "
        f"no_valid={no_valid} missing_bp={missing_bp} "
        f"fallback_bp={fallback_bp} shifted={total_shifted} z_shifted={total_z_shifted} "
        f"refine_ok={int(refined_accepted)} refine_reject={int(refined_rejected)} "
        f"bridge_missing={int(bridge_missing_total)} bridge_transient={int(bridge_transient_total)}"
    )
    if int(ego_summary.get("egos_considered", 0)) > 0:
        print(
            "[SPAWN_PRE] Ego alignment: "
            f"egos={int(ego_summary.get('egos_considered', 0))} "
            f"aligned={int(ego_summary.get('egos_aligned', 0))} "
            f"spawn_valid={int(ego_summary.get('egos_spawn_valid', 0))} "
            f"no_candidates={int(ego_summary.get('egos_no_candidates', 0))} "
            f"z_shifted={int(ego_summary.get('egos_z_shifted', 0))}"
        )
    if no_valid:
        samples = []
        for actor_id, entry in report.get("actors", {}).items():
            chosen = entry.get("chosen") or {}
            if chosen.get("status") != "no_valid_candidates":
                continue
            best = entry.get("best_invalid_candidate") or {}
            model_used = entry.get("model_used") or entry.get("model")
            samples.append(
                f"id={actor_id} kind={entry.get('kind')} model={model_used} "
                f"best_src={best.get('source')} cost={best.get('base_cost')}"
            )
            if len(samples) >= 10:
                break
        if samples:
            print("[SPAWN_PRE] No-valid examples: " + "; ".join(samples))
    return report

# ---------------------- Core conversion ---------------------- #

def build_trajectories(
    yaml_dir: Path,
    dt: float,
    tx: float,
    ty: float,
    tz: float,
    yaw_deg: float,
    flip_y: bool = False,
) -> Tuple[
    Dict[int, List[Waypoint]],
    Dict[int, List[float]],
    List[Waypoint],
    List[float],
    Dict[int, Dict[str, object]],
]:
    """Parse YAML sequence into per-vehicle trajectories and ego path, plus per-waypoint times."""
    yaml_paths = list_yaml_timesteps(yaml_dir)
    if not yaml_paths:
        raise SystemExit(f"No YAML files found under {yaml_dir}")

    vehicles: Dict[int, List[Waypoint]] = {}
    vehicle_times: Dict[int, List[float]] = {}
    ego_traj: List[Waypoint] = []
    ego_times: List[float] = []
    obj_info: Dict[int, Dict[str, object]] = {}
    spawn_report: Dict[str, object] | None = None

    for idx, path in enumerate(yaml_paths):
        try:
            frame_idx = int(path.stem)
        except Exception:
            frame_idx = idx
        frame_time = float(frame_idx) * float(dt)
        data = load_yaml(path)
        ego_pose = data.get("true_ego_pose") or data.get("lidar_pose")
        if ego_pose:
            ex, ey, ez = float(ego_pose[0]), float(ego_pose[1]), float(ego_pose[2])
            ex, ey = apply_se2((ex, ey), yaw_deg, tx, ty, flip_y=flip_y)
            ego_yaw = yaw_from_pose(ego_pose)
            if flip_y:
                ego_yaw = -ego_yaw
            ego_traj.append(
                Waypoint(
                    x=ex,
                    y=ey,
                    z=ez + tz,
                    yaw=ego_yaw + yaw_deg,
                    pitch=float(ego_pose[3]) if len(ego_pose) > 3 else 0.0,
                    roll=float(ego_pose[5]) if len(ego_pose) > 5 else 0.0,
                )
            )
            ego_times.append(frame_time)

        vehs = data.get("vehicles", {}) or {}
        for vid_str, payload in vehs.items():
            try:
                vid = int(vid_str)
            except Exception:
                continue
            if isinstance(payload, dict):
                existing = obj_info.get(vid, {})
                obj_type = payload.get("obj_type")
                if obj_type and (not existing.get("obj_type")):
                    existing["obj_type"] = obj_type
                    existing["model"] = map_obj_type(obj_type)
                elif obj_type and existing.get("obj_type") and obj_type != existing.get("obj_type"):
                    # Keep first seen obj_type but note mismatch once
                    if not existing.get("_obj_type_conflict"):
                        print(f"[WARN] obj_type conflict for id {vid}: '{existing.get('obj_type')}' vs '{obj_type}' (keeping first)")
                        existing["_obj_type_conflict"] = True
                ext = payload.get("extent") or []
                if isinstance(ext, Sequence):
                    length = float(ext[0]) * 2 if len(ext) > 0 else None
                    width = float(ext[1]) * 2 if len(ext) > 1 else None
                    if length is not None and existing.get("length") is None:
                        existing["length"] = length
                    if width is not None and existing.get("width") is None:
                        existing["width"] = width
                if existing:
                    obj_info[vid] = existing
            loc = payload.get("location") or [0, 0, 0]
            ang = payload.get("angle") or [0, 0, 0]
            pitch = float(ang[0]) if len(ang) > 0 else 0.0
            yaw = yaw_from_angle(ang)
            if flip_y:
                yaw = -yaw
            yaw += yaw_deg
            roll = float(ang[2]) if len(ang) > 2 else 0.0
            x, y = apply_se2((float(loc[0]), float(loc[1])), yaw_deg, tx, ty, flip_y=flip_y)
            z = float(loc[2]) + tz if len(loc) > 2 else tz
            wp = Waypoint(x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=roll)
            vehicles.setdefault(vid, []).append(wp)
            vehicle_times.setdefault(vid, []).append(frame_time)

    # Compute simple average speed (m/s) per vehicle from path length
    speeds: Dict[int, float] = {}
    for vid, traj in vehicles.items():
        dist = 0.0
        for a, b in zip(traj, traj[1:]):
            dist += euclid3((a.x, a.y, a.z), (b.x, b.y, b.z))
        speeds[vid] = dist / max(dt * max(len(traj) - 1, 1), 1e-6)

    return vehicles, vehicle_times, ego_traj, ego_times, obj_info


def extract_obj_info(yaml_dir: Path) -> Dict[int, Dict[str, object]]:
    """Gather obj_type/model/size from the first timestep in a YAML directory."""
    obj_info: Dict[int, Dict[str, object]] = {}
    first_yaml = next(iter(list_yaml_timesteps(yaml_dir)), None)
    if not first_yaml:
        return obj_info
    data0 = load_yaml(first_yaml)
    vehs0 = data0.get("vehicles", {}) or {}
    for vid_str, payload in vehs0.items():
        try:
            vid = int(vid_str)
        except Exception:
            continue
        obj_type = payload.get("obj_type") or "npc"
        model = map_obj_type(obj_type)
        ext = payload.get("extent") or []
        length = float(ext[0]) * 2 if len(ext) > 0 else None
        width = float(ext[1]) * 2 if len(ext) > 1 else None
        obj_info[vid] = {
            "obj_type": obj_type,
            "model": model,
            "length": length,
            "width": width,
        }
    return obj_info


def write_route_xml(
    path: Path,
    route_id: str,
    role: str,
    town: str,
    waypoints: List[Waypoint],
    times: List[float] | None = None,
    snap_to_road: bool = False,
    xml_tx: float = 0.0,
    xml_ty: float = 0.0,
) -> None:
    root = ET.Element("routes")
    route = ET.SubElement(
        root,
        "route",
        {
            "id": str(route_id),
            "town": town,
            "role": role,
            "snap_to_road": "true" if snap_to_road else "false",
        },
    )
    for idx, wp in enumerate(waypoints):
        # Normalize pitch and roll: CARLA XML expects pitch=360.0 (or 0.0) and roll=0.0
        # Using 360.0 for pitch as seen in reference XML files
        attrs = {
            "x": f"{wp.x + xml_tx:.6f}",
            "y": f"{wp.y + xml_ty:.6f}",
            "z": f"{wp.z:.6f}",
            "yaw": f"{wp.yaw:.6f}",
            "pitch": "360.000000",  # Normalized for CARLA compatibility
            "roll": "0.000000",     # Normalized for CARLA compatibility
        }
        if times and idx < len(times):
            try:
                attrs["time"] = f"{float(times[idx]):.6f}"
            except (TypeError, ValueError):
                pass
        ET.SubElement(
            route,
            "waypoint",
            attrs,
        )
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def parse_route_xml(path: Path) -> List[Waypoint]:
    """Load waypoints (x,y,z,yaw) from a CARLA route XML."""
    tree = ET.parse(path)
    root = tree.getroot()
    wps: List[Waypoint] = []
    for node in root.findall(".//waypoint"):
        try:
            x = float(node.attrib.get("x", 0.0))
            y = float(node.attrib.get("y", 0.0))
            z = float(node.attrib.get("z", 0.0))
            yaw = float(node.attrib.get("yaw", 0.0))
            wps.append(Waypoint(x=x, y=y, z=z, yaw=yaw))
        except Exception:
            continue
    return wps


def save_manifest(
    manifest_path: Path,
    actors_by_kind: Dict[str, List[dict]],
    ego_entries: List[dict],
) -> None:
    """Save manifest with actors organized by kind (ego, npc, static, etc.)."""
    manifest: Dict[str, List[dict]] = {}
    if ego_entries:
        manifest["ego"] = ego_entries
    # Add all other actor kinds
    for kind, entries in sorted(actors_by_kind.items()):
        manifest[kind] = entries
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# ---------------------- Visualization ---------------------- #

def plot_frame(
    timestep: int,
    actors_by_id: Dict[int, List[Waypoint]],
    ego_trajs: Sequence[List[Waypoint]],
    out_path: Path,
    axes_limits: Tuple[float, float, float, float] | None = None,
    map_lines: List[List[Tuple[float, float]]] | None = None,
    invert_plot_y: bool = False,
):
    if plt is None or patches is None or transforms is None:
        raise RuntimeError("matplotlib is required for visualization; install matplotlib imageio")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Timestep {timestep:06d}")

    xs: List[float] = []
    ys: List[float] = []
    for vid, traj in actors_by_id.items():
        if timestep >= len(traj):
            continue
        wp = traj[timestep]
        width = 2.0
        height = 4.0
        rect = patches.Rectangle(
            (wp.x - width / 2, wp.y - height / 2),
            width,
            height,
            linewidth=1.0,
            edgecolor="C0",
            facecolor="C0",
            alpha=0.4,
        )
        rot = transforms.Affine2D().rotate_deg_around(wp.x, wp.y, wp.yaw) + ax.transData
        rect.set_transform(rot)
        ax.add_patch(rect)
        ax.text(wp.x, wp.y, f"{vid}", fontsize=7, ha="center", va="center")
        xs.append(wp.x)
        ys.append(wp.y)

    if ego_trajs:
        for ego_idx, ego_traj in enumerate(ego_trajs):
            if not ego_traj:
                continue
            idx = min(timestep, len(ego_traj) - 1)
            ego = ego_traj[idx]
            color = "orange" if ego_idx == 0 else f"C{(ego_idx + 1) % 10}"
            tri = patches.RegularPolygon(
                (ego.x, ego.y),
                numVertices=3,
                radius=2.5,
                orientation=math.radians(ego.yaw),
                color=color,
                alpha=0.6,
            )
            ax.add_patch(tri)
            ax.text(ego.x, ego.y, f"ego{ego_idx}", fontsize=7, ha="center", va="center")
            xs.append(ego.x)
            ys.append(ego.y)

    if map_lines:
        for line in map_lines:
            if len(line) < 2:
                continue
            lx = [p[0] for p in line]
            ly = [p[1] for p in line]
            ax.plot(lx, ly, color="gray", linewidth=1.0, alpha=0.5)
            xs.extend(lx)
            ys.extend(ly)

    if axes_limits:
        minx, maxx, miny, maxy = axes_limits
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
    elif xs and ys:
        pad = 10.0
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)

    if invert_plot_y:
        ax.invert_yaxis()

    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_gif(frames_dir: Path, gif_path: Path, fps: float = 10.0) -> None:
    if imageio is None:
        raise RuntimeError("imageio is required for GIF output; install imageio")
    imgs = []
    for png in sorted(frames_dir.glob("frame_*.png")):
        imgs.append(imageio.imread(png))
    if not imgs:
        raise RuntimeError("No frames produced for GIF")
    duration_ms = 1000.0 / float(fps)
    imageio.mimsave(gif_path, imgs, duration=duration_ms / 1000.0)


def write_paths_png(
    actors_by_id: Dict[int, List[Waypoint]],
    ego_trajs: Sequence[List[Waypoint]],
    map_lines: List[List[Tuple[float, float]]],
    out_path: Path,
    axis_pad: float = 10.0,
    invert_plot_y: bool = False,
) -> None:
    if plt is None or patches is None:
        raise RuntimeError("matplotlib is required for --paths-png; install matplotlib")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Actor Paths")

    xs: List[float] = []
    ys: List[float] = []

    # Map
    for line in map_lines:
        if len(line) < 2:
            continue
        lx = [p[0] for p in line]
        ly = [p[1] for p in line]
        ax.plot(lx, ly, color="gray", linewidth=0.8, alpha=0.5, zorder=0)
        xs.extend(lx)
        ys.extend(ly)

    # Actors
    for vid, traj in actors_by_id.items():
        if len(traj) < 2:
            continue
        lx = [wp.x for wp in traj]
        ly = [wp.y for wp in traj]
        ax.plot(lx, ly, linewidth=1.5, alpha=0.9, label=f"id {vid}")
        ax.scatter(lx[0], ly[0], s=15, marker="o")
        xs.extend(lx)
        ys.extend(ly)

    # Ego(s)
    if ego_trajs:
        for ego_idx, ego_traj in enumerate(ego_trajs):
            if not ego_traj:
                continue
            lx = [wp.x for wp in ego_traj]
            ly = [wp.y for wp in ego_traj]
            color = "black" if ego_idx == 0 else f"C{(ego_idx + 1) % 10}"
            label = "ego" if ego_idx == 0 else f"ego{ego_idx}"
            ax.plot(lx, ly, color=color, linewidth=2.0, alpha=0.8, label=label)
            ax.scatter(lx[0], ly[0], s=30, marker="*", color=color)
            xs.extend(lx)
            ys.extend(ly)

    if xs and ys:
        pad = max(0.0, axis_pad)
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)

    if invert_plot_y:
        ax.invert_yaxis()

    ax.grid(True, linestyle="--", alpha=0.4)
    if len(actors_by_id) <= 20:  # avoid huge legends
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_actor_yaw_viz(
    actor_id: int,
    gt_traj: List[Waypoint],
    xml_traj: List[Waypoint],
    map_lines: List[List[Tuple[float, float]]] | None,
    out_path: Path,
    arrow_step: int = 5,
    arrow_len: float = 0.8,
    pad: float = 5.0,
    invert_plot_y: bool = False,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for visualization; install matplotlib")

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Actor {actor_id} yaw: GT vs XML")

    # Map layer
    if map_lines:
        for line in map_lines:
            if len(line) < 2:
                continue
            xs, ys = zip(*line)
            ax.plot(xs, ys, color="#cccccc", linewidth=0.7, alpha=0.6, zorder=0)

    # Paths
    gt_x = [wp.x for wp in gt_traj]
    gt_y = [wp.y for wp in gt_traj]
    xml_x = [wp.x for wp in xml_traj]
    xml_y = [wp.y for wp in xml_traj]
    # Draw XML first, then GT on top with markers so overlap is visible
    ax.plot(xml_x, xml_y, color="#d95f0e", linewidth=2.0, alpha=0.9, label="XML path", zorder=2)
    ax.plot(
        gt_x,
        gt_y,
        color="#2c7fb8",
        linewidth=2.2,
        linestyle="--",
        marker="o",
        markersize=2.5,
        markevery=max(1, int(len(gt_x) / 20)),
        label="GT path",
        zorder=3,
    )

    # If GT and XML are effectively identical, annotate it
    min_len = min(len(gt_traj), len(xml_traj))
    if min_len > 0:
        max_diff = 0.0
        for i in range(min_len):
            dx = gt_traj[i].x - xml_traj[i].x
            dy = gt_traj[i].y - xml_traj[i].y
            max_diff = max(max_diff, math.hypot(dx, dy))
        if max_diff < 1e-3:
            ax.text(
                0.02,
                0.98,
                "GT == XML (overlapping paths)",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="#444444",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
                zorder=10,
            )

    # Yaw arrows
    def _quiver(traj: List[Waypoint], color: str, label: str) -> None:
        if not traj:
            return
        step = max(1, int(arrow_step))
        xs = [wp.x for i, wp in enumerate(traj) if i % step == 0]
        ys = [wp.y for i, wp in enumerate(traj) if i % step == 0]
        us = [math.cos(math.radians(wp.yaw)) * arrow_len for i, wp in enumerate(traj) if i % step == 0]
        vs = [math.sin(math.radians(wp.yaw)) * arrow_len for i, wp in enumerate(traj) if i % step == 0]
        ax.quiver(
            xs,
            ys,
            us,
            vs,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=color,
            alpha=0.7,
            width=0.002,
            label=label,
        )

    _quiver(gt_traj, "#2c7fb8", "GT yaw")
    _quiver(xml_traj, "#d95f0e", "XML yaw")

    # Bounds
    xs_all = gt_x + xml_x
    ys_all = gt_y + xml_y
    if xs_all and ys_all:
        pad_val = max(0.0, float(pad))
        ax.set_xlim(min(xs_all) - pad_val, max(xs_all) + pad_val)
        ax.set_ylim(min(ys_all) - pad_val, max(ys_all) + pad_val)

    if invert_plot_y:
        ax.invert_yaxis()

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _sample_map_nodes(
    map_line_records: List[Dict[str, object]] | None,
    map_lines: List[List[Tuple[float, float]]] | None,
    step: int,
) -> List[Tuple[float, float]]:
    step = max(1, int(step))
    nodes: List[Tuple[float, float]] = []
    seen: set[Tuple[int, int]] = set()

    recs = map_line_records or []
    if recs:
        for rec in recs:
            pts = rec.get("points") if isinstance(rec, dict) else None
            if not isinstance(pts, list) or len(pts) < 2:
                continue
            for i in range(0, len(pts), step):
                try:
                    x = float(pts[i][0])
                    y = float(pts[i][1])
                except Exception:
                    continue
                key = (int(round(x * 100.0)), int(round(y * 100.0)))
                if key in seen:
                    continue
                seen.add(key)
                nodes.append((x, y))
    elif map_lines:
        for line in map_lines:
            if len(line) < 2:
                continue
            for i in range(0, len(line), step):
                x, y = line[i]
                key = (int(round(float(x) * 100.0)), int(round(float(y) * 100.0)))
                if key in seen:
                    continue
                seen.add(key)
                nodes.append((float(x), float(y)))
    return nodes


def _nearest_node_indices(
    query_points: Sequence[Tuple[float, float]],
    nodes: Sequence[Tuple[float, float]],
    max_radius: float,
) -> set[int]:
    if not query_points or not nodes:
        return set()
    max_r2 = float(max_radius) * float(max_radius)
    touched: set[int] = set()
    for qx, qy in query_points:
        best_idx = -1
        best_d2 = float("inf")
        for idx, (nx, ny) in enumerate(nodes):
            dx = float(nx) - float(qx)
            dy = float(ny) - float(qy)
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_idx = idx
        if best_idx >= 0 and best_d2 <= max_r2:
            touched.add(best_idx)
    return touched


def _capture_carla_topdown_bev(
    host: str,
    port: int,
    bounds: Tuple[float, float, float, float],
    image_w: int,
    image_h: int,
    fov_deg: float,
    margin_scale: float,
    expected_town: str | None = None,
) -> Dict[str, object] | None:
    """
    Capture a top-down RGB image from CARLA centered around bounds.
    Returns dict with image and projection data for world->pixel overlays.
    """
    if carla is None:
        return None
    if np is None:
        return None
    _assert_carla_endpoint_reachable(host, port, timeout_s=2.0)
    image_w = max(256, int(image_w))
    image_h = max(256, int(image_h))
    fov_deg = min(120.0, max(15.0, float(fov_deg)))
    margin_scale = max(1.0, float(margin_scale))

    minx, maxx, miny, maxy = bounds
    cx = 0.5 * (float(minx) + float(maxx))
    cy = 0.5 * (float(miny) + float(maxy))
    span_x = max(5.0, float(maxx) - float(minx)) * margin_scale
    span_y = max(5.0, float(maxy) - float(miny)) * margin_scale

    hfov = math.radians(fov_deg)
    vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * (float(image_h) / float(image_w)))
    alt_x = span_x / (2.0 * math.tan(hfov / 2.0))
    alt_y = span_y / (2.0 * math.tan(vfov / 2.0))

    client = carla.Client(host, int(port))
    client.set_timeout(20.0)
    world = client.get_world()
    cmap = world.get_map()
    if expected_town and expected_town not in str(cmap.name or ""):
        candidates = [m for m in client.get_available_maps() if expected_town in m]
        if candidates:
            world = client.load_world(candidates[0])
            cmap = world.get_map()

    ground_z = 0.0
    try:
        wp = cmap.get_waypoint(
            carla.Location(x=float(cx), y=float(cy), z=0.0),
            project_to_road=True,
            lane_type=carla.LaneType.Any,
        )
        if wp is not None:
            ground_z = float(wp.transform.location.z)
    except Exception:
        ground_z = 0.0

    cam_z = float(ground_z) + max(float(alt_x), float(alt_y)) + 2.0
    cam_tf = carla.Transform(
        carla.Location(x=float(cx), y=float(cy), z=float(cam_z)),
        carla.Rotation(pitch=-90.0, yaw=-90.0, roll=0.0),
    )

    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(image_w))
    cam_bp.set_attribute("image_size_y", str(image_h))
    cam_bp.set_attribute("fov", str(float(fov_deg)))
    cam_bp.set_attribute("sensor_tick", "0.0")
    cam_bp.set_attribute("motion_blur_intensity", "0.0")

    cam_actor = None
    img = None
    img_q: "queue.Queue[object]" = queue.Queue()
    try:
        cam_actor = world.spawn_actor(cam_bp, cam_tf)
        cam_actor.listen(lambda data: img_q.put(data))

        sync = bool(getattr(world.get_settings(), "synchronous_mode", False))
        for _ in range(20):
            if sync:
                world.tick()
            else:
                world.wait_for_tick(seconds=1.0)
            try:
                img = img_q.get(timeout=0.5)
                break
            except Exception:
                continue
        if img is None:
            return None

        arr = np.frombuffer(img.raw_data, dtype=np.uint8)
        arr = arr.reshape((int(img.height), int(img.width), 4))
        rgb = arr[:, :, :3][:, :, ::-1].copy()  # BGRA -> RGB

        focal = float(image_w) / (2.0 * math.tan(hfov / 2.0))
        K = np.array(
            [
                [focal, 0.0, float(image_w) / 2.0],
                [0.0, focal, float(image_h) / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        w2c = np.array(cam_actor.get_transform().get_inverse_matrix(), dtype=np.float64)
        return {
            "image": rgb,
            "width": int(image_w),
            "height": int(image_h),
            "K": K,
            "world_to_camera": w2c,
            "ground_z": float(ground_z),
            "camera_z": float(cam_z),
        }
    finally:
        try:
            if cam_actor is not None:
                cam_actor.stop()
        except Exception:
            pass
        try:
            if cam_actor is not None:
                cam_actor.destroy()
        except Exception:
            pass


def _project_world_xy_to_image(
    points_xy: Sequence[Tuple[float, float]],
    world_to_camera,
    K,
    width: int,
    height: int,
    z_world: float,
) -> List[Tuple[float, float]]:
    if np is None:
        return []
    if not points_xy:
        return []
    out: List[Tuple[float, float]] = []
    w = float(max(1, int(width)))
    h = float(max(1, int(height)))
    for x, y in points_xy:
        world_pt = np.array([float(x), float(y), float(z_world), 1.0], dtype=np.float64)
        cam_pt = np.dot(world_to_camera, world_pt)
        # Convert CARLA camera coordinates to conventional pinhole coordinates.
        cam_cv = np.array([cam_pt[1], -cam_pt[2], cam_pt[0]], dtype=np.float64)
        depth = float(cam_cv[2])
        if depth <= 1e-4:
            continue
        u = float(K[0, 0] * cam_cv[0] / depth + K[0, 2])
        v = float(K[1, 1] * cam_cv[1] / depth + K[1, 2])
        if -2.0 <= u <= w + 2.0 and -2.0 <= v <= h + 2.0:
            out.append((u, v))
    return out


def write_ego_alignment_bev_viz(
    ego_idx: int,
    pre_align_traj: List[Waypoint],
    post_align_traj: List[Waypoint],
    map_lines: List[List[Tuple[float, float]]] | None,
    map_line_records: List[Dict[str, object]] | None,
    map_image,
    map_image_bounds: Tuple[float, float, float, float] | None,
    captured_bev: Dict[str, object] | None,
    out_path: Path,
    xml_tx: float = 0.0,
    xml_ty: float = 0.0,
    pad: float = 24.0,
    node_step: int = 2,
    match_radius: float = 1.8,
    invert_plot_y: bool = False,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for visualization; install matplotlib")

    pre_xy = [(wp.x + float(xml_tx), wp.y + float(xml_ty)) for wp in pre_align_traj]
    post_xy = [(wp.x + float(xml_tx), wp.y + float(xml_ty)) for wp in post_align_traj]
    if not pre_xy and not post_xy:
        return

    all_nodes = _sample_map_nodes(map_line_records, map_lines, step=node_step)
    touched = _nearest_node_indices(post_xy, all_nodes, max_radius=max(0.1, float(match_radius)))
    touched_nodes = [all_nodes[i] for i in sorted(touched)]

    map_bounds = map_image_bounds
    if map_bounds is None and map_lines:
        pts: List[Tuple[float, float]] = []
        for ln in map_lines:
            pts.extend(ln)
        map_bounds = _bounds_from_points(pts)
    path_bounds = _bounds_from_points(pre_xy + post_xy)
    merged = _merge_bounds([map_bounds, path_bounds])

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_title(f"Ego {ego_idx}: Alignment on CARLA BEV")
    used_carla_capture = False

    if captured_bev is not None and np is not None:
        try:
            image = captured_bev.get("image")
            w2c = captured_bev.get("world_to_camera")
            K = captured_bev.get("K")
            width = int(captured_bev.get("width", 0))
            height = int(captured_bev.get("height", 0))
            z_world = float(captured_bev.get("ground_z", 0.0))
            if image is not None and w2c is not None and K is not None and width > 0 and height > 0:
                used_carla_capture = True
                ax.imshow(image, origin="upper", zorder=0)
                ax.set_xlim(0, width)
                ax.set_ylim(height, 0)
                ax.set_aspect("equal", adjustable="box")

                nodes_px = _project_world_xy_to_image(
                    all_nodes, w2c, K, width=width, height=height, z_world=z_world
                )
                touched_px = _project_world_xy_to_image(
                    touched_nodes, w2c, K, width=width, height=height, z_world=z_world
                )
                pre_px = _project_world_xy_to_image(
                    pre_xy, w2c, K, width=width, height=height, z_world=z_world
                )
                post_px = _project_world_xy_to_image(
                    post_xy, w2c, K, width=width, height=height, z_world=z_world
                )

                if nodes_px:
                    ax.scatter(
                        [p[0] for p in nodes_px],
                        [p[1] for p in nodes_px],
                        s=8,
                        marker="s",
                        c="#1f1f1f",
                        alpha=0.24,
                        linewidths=0.0,
                        label="all map nodes",
                        zorder=1,
                    )
                if touched_px:
                    ax.scatter(
                        [p[0] for p in touched_px],
                        [p[1] for p in touched_px],
                        s=15,
                        marker="s",
                        c="#00e5ff",
                        alpha=0.82,
                        linewidths=0.0,
                        label="nodes touched by aligned ego",
                        zorder=2,
                    )

                if pre_px:
                    ax.plot(
                        [p[0] for p in pre_px],
                        [p[1] for p in pre_px],
                        color="#1f77b4",
                        linestyle="--",
                        linewidth=2.2,
                        alpha=0.95,
                        label="GT from YAML + offsets (pre-align)",
                        zorder=4,
                    )
                    ax.scatter([pre_px[0][0]], [pre_px[0][1]], s=55, c="#1f77b4", marker="o", zorder=5, label="pre start")
                    ax.scatter([pre_px[-1][0]], [pre_px[-1][1]], s=55, c="#1f77b4", marker="x", zorder=5, label="pre end")
                if post_px:
                    ax.plot(
                        [p[0] for p in post_px],
                        [p[1] for p in post_px],
                        color="#e53935",
                        linewidth=2.4,
                        alpha=0.98,
                        label="post alignment/refinement",
                        zorder=6,
                    )
                    ax.scatter([post_px[0][0]], [post_px[0][1]], s=65, c="#e53935", marker="o", zorder=7, label="post start")
                    ax.scatter([post_px[-1][0]], [post_px[-1][1]], s=65, c="#e53935", marker="x", zorder=7, label="post end")

                # Zoom to ego ROI inside captured image
                roi = pre_px + post_px
                if roi:
                    rx = [p[0] for p in roi]
                    ry = [p[1] for p in roi]
                    span = max(20.0, max(max(rx) - min(rx), max(ry) - min(ry)))
                    pad_px = span * 0.22
                    x0 = max(0.0, min(rx) - pad_px)
                    x1 = min(float(width), max(rx) + pad_px)
                    y0 = max(0.0, min(ry) - pad_px)
                    y1 = min(float(height), max(ry) + pad_px)
                    ax.set_xlim(x0, x1)
                    ax.set_ylim(y1, y0)
        except Exception:
            used_carla_capture = False

    if not used_carla_capture:
        ax.set_aspect("equal", adjustable="box")
        if map_image is not None:
            if merged:
                ext = map_bounds or merged
                ax.imshow(map_image, extent=(ext[0], ext[1], ext[2], ext[3]), origin="lower", alpha=0.97, zorder=0)
            else:
                ax.imshow(map_image, origin="lower", alpha=0.97, zorder=0)
        elif map_lines:
            _plot_background_lines(ax, map_lines, color="#b8b8b8", lw=0.8, alpha=0.7)
            ax.text(
                0.02,
                0.98,
                "No captured CARLA BEV; using vector/raster fallback",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#666666",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
                zorder=9,
            )

        if all_nodes:
            ax.scatter(
                [p[0] for p in all_nodes],
                [p[1] for p in all_nodes],
                s=9,
                marker="s",
                c="#3b3b3b",
                alpha=0.22,
                linewidths=0.0,
                label="all map nodes",
                zorder=1,
            )
        if touched_nodes:
            ax.scatter(
                [p[0] for p in touched_nodes],
                [p[1] for p in touched_nodes],
                s=16,
                marker="s",
                c="#00bcd4",
                alpha=0.85,
                linewidths=0.0,
                label="nodes touched by aligned ego",
                zorder=2,
            )

        if pre_xy:
            ax.plot(
                [p[0] for p in pre_xy],
                [p[1] for p in pre_xy],
                color="#1f77b4",
                linestyle="--",
                linewidth=2.2,
                alpha=0.95,
                label="GT from YAML + offsets (pre-align)",
                zorder=4,
            )
            ax.scatter([pre_xy[0][0]], [pre_xy[0][1]], s=55, c="#1f77b4", marker="o", zorder=5, label="pre start")
            ax.scatter([pre_xy[-1][0]], [pre_xy[-1][1]], s=55, c="#1f77b4", marker="x", zorder=5, label="pre end")
        if post_xy:
            ax.plot(
                [p[0] for p in post_xy],
                [p[1] for p in post_xy],
                color="#e53935",
                linewidth=2.4,
                alpha=0.98,
                label="post alignment/refinement",
                zorder=6,
            )
            ax.scatter([post_xy[0][0]], [post_xy[0][1]], s=65, c="#e53935", marker="o", zorder=7, label="post start")
            ax.scatter([post_xy[-1][0]], [post_xy[-1][1]], s=65, c="#e53935", marker="x", zorder=7, label="post end")

        if merged:
            pad_val = max(0.0, float(pad)) * 1.2
            ax.set_xlim(merged[0] - pad_val, merged[1] + pad_val)
            ax.set_ylim(merged[2] - pad_val, merged[3] + pad_val)

        if invert_plot_y:
            ax.invert_yaxis()

    info = [
        f"background: {'CARLA captured BEV' if used_carla_capture else 'fallback'}",
        f"all nodes: {len(all_nodes)}",
        f"touched nodes: {len(touched_nodes)}",
        f"touch radius: {float(match_radius):.2f} m",
    ]
    ax.text(
        0.02,
        0.02,
        "\n".join(info),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#222222",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.82),
        zorder=10,
    )

    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout(rect=[0.0, 0.0, 0.76, 1.0])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_actor_alignment_bev_viz(
    title: str,
    pre_trajs: Dict[int, List[Waypoint]],
    post_trajs: Dict[int, List[Waypoint]],
    map_lines: List[List[Tuple[float, float]]] | None,
    map_line_records: List[Dict[str, object]] | None,
    map_image,
    map_image_bounds: Tuple[float, float, float, float] | None,
    captured_bev: Dict[str, object] | None,
    out_path: Path,
    xml_tx: float = 0.0,
    xml_ty: float = 0.0,
    pad: float = 24.0,
    node_step: int = 2,
    match_radius: float = 1.8,
    invert_plot_y: bool = False,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for visualization; install matplotlib")

    actor_ids = sorted(set(int(k) for k in pre_trajs.keys()) | set(int(k) for k in post_trajs.keys()))
    if not actor_ids:
        return

    pre_xy_by_id: Dict[int, List[Tuple[float, float]]] = {}
    post_xy_by_id: Dict[int, List[Tuple[float, float]]] = {}
    for aid in actor_ids:
        pre = pre_trajs.get(aid) or []
        post = post_trajs.get(aid) or []
        p_pre = [(wp.x + float(xml_tx), wp.y + float(xml_ty)) for wp in pre]
        p_post = [(wp.x + float(xml_tx), wp.y + float(xml_ty)) for wp in post]
        if p_pre:
            pre_xy_by_id[aid] = p_pre
        if p_post:
            post_xy_by_id[aid] = p_post
    if not pre_xy_by_id and not post_xy_by_id:
        return

    all_pre = [pt for path in pre_xy_by_id.values() for pt in path]
    all_post = [pt for path in post_xy_by_id.values() for pt in path]
    all_nodes = _sample_map_nodes(map_line_records, map_lines, step=node_step)
    touched = _nearest_node_indices(all_post, all_nodes, max_radius=max(0.1, float(match_radius)))
    touched_nodes = [all_nodes[i] for i in sorted(touched)]

    map_bounds = map_image_bounds
    if map_bounds is None and map_lines:
        pts: List[Tuple[float, float]] = []
        for ln in map_lines:
            pts.extend(ln)
        map_bounds = _bounds_from_points(pts)
    path_bounds = _bounds_from_points(all_pre + all_post)
    merged = _merge_bounds([map_bounds, path_bounds])

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_title(title)
    used_carla_capture = False
    individual_mode = len(actor_ids) == 1

    if captured_bev is not None and np is not None:
        try:
            image = captured_bev.get("image")
            w2c = captured_bev.get("world_to_camera")
            K = captured_bev.get("K")
            width = int(captured_bev.get("width", 0))
            height = int(captured_bev.get("height", 0))
            z_world = float(captured_bev.get("ground_z", 0.0))
            if image is not None and w2c is not None and K is not None and width > 0 and height > 0:
                used_carla_capture = True
                ax.imshow(image, origin="upper", zorder=0)
                ax.set_xlim(0, width)
                ax.set_ylim(height, 0)
                ax.set_aspect("equal", adjustable="box")

                nodes_px = _project_world_xy_to_image(
                    all_nodes, w2c, K, width=width, height=height, z_world=z_world
                )
                touched_px = _project_world_xy_to_image(
                    touched_nodes, w2c, K, width=width, height=height, z_world=z_world
                )
                if nodes_px:
                    ax.scatter(
                        [p[0] for p in nodes_px],
                        [p[1] for p in nodes_px],
                        s=7,
                        marker="s",
                        c="#1a1a1a",
                        alpha=0.22,
                        linewidths=0.0,
                        label="all map nodes",
                        zorder=1,
                    )
                if touched_px:
                    ax.scatter(
                        [p[0] for p in touched_px],
                        [p[1] for p in touched_px],
                        s=14,
                        marker="s",
                        c="#00e5ff",
                        alpha=0.82,
                        linewidths=0.0,
                        label="nodes touched by aligned paths",
                        zorder=2,
                    )

                pre_label_drawn = False
                post_label_drawn = False
                roi_pts: List[Tuple[float, float]] = []
                for aid in actor_ids:
                    pre_px = _project_world_xy_to_image(
                        pre_xy_by_id.get(aid, []), w2c, K, width=width, height=height, z_world=z_world
                    )
                    post_px = _project_world_xy_to_image(
                        post_xy_by_id.get(aid, []), w2c, K, width=width, height=height, z_world=z_world
                    )
                    roi_pts.extend(pre_px)
                    roi_pts.extend(post_px)
                    if pre_px:
                        ax.plot(
                            [p[0] for p in pre_px],
                            [p[1] for p in pre_px],
                            color="#1f77b4",
                            linestyle="--",
                            linewidth=2.1 if individual_mode else 1.1,
                            alpha=0.95 if individual_mode else 0.38,
                            label=None if pre_label_drawn else "GT from YAML + offsets (pre-align)",
                            zorder=4,
                        )
                        pre_label_drawn = True
                        if individual_mode:
                            ax.scatter([pre_px[0][0]], [pre_px[0][1]], s=50, c="#1f77b4", marker="o", zorder=5, label="pre start")
                            ax.scatter([pre_px[-1][0]], [pre_px[-1][1]], s=50, c="#1f77b4", marker="x", zorder=5, label="pre end")
                    if post_px:
                        ax.plot(
                            [p[0] for p in post_px],
                            [p[1] for p in post_px],
                            color="#e53935",
                            linewidth=2.3 if individual_mode else 1.3,
                            alpha=0.98 if individual_mode else 0.45,
                            label=None if post_label_drawn else "post alignment/refinement",
                            zorder=6,
                        )
                        post_label_drawn = True
                        if individual_mode:
                            ax.scatter([post_px[0][0]], [post_px[0][1]], s=58, c="#e53935", marker="o", zorder=7, label="post start")
                            ax.scatter([post_px[-1][0]], [post_px[-1][1]], s=58, c="#e53935", marker="x", zorder=7, label="post end")

                if roi_pts:
                    rx = [p[0] for p in roi_pts]
                    ry = [p[1] for p in roi_pts]
                    span = max(24.0, max(max(rx) - min(rx), max(ry) - min(ry)))
                    pad_px = span * 0.24
                    x0 = max(0.0, min(rx) - pad_px)
                    x1 = min(float(width), max(rx) + pad_px)
                    y0 = max(0.0, min(ry) - pad_px)
                    y1 = min(float(height), max(ry) + pad_px)
                    ax.set_xlim(x0, x1)
                    ax.set_ylim(y1, y0)
        except Exception:
            used_carla_capture = False

    if not used_carla_capture:
        ax.set_aspect("equal", adjustable="box")
        if map_image is not None:
            if merged:
                ext = map_bounds or merged
                ax.imshow(map_image, extent=(ext[0], ext[1], ext[2], ext[3]), origin="lower", alpha=0.97, zorder=0)
            else:
                ax.imshow(map_image, origin="lower", alpha=0.97, zorder=0)
        elif map_lines:
            _plot_background_lines(ax, map_lines, color="#b8b8b8", lw=0.8, alpha=0.7)

        if all_nodes:
            ax.scatter(
                [p[0] for p in all_nodes],
                [p[1] for p in all_nodes],
                s=9,
                marker="s",
                c="#3b3b3b",
                alpha=0.22,
                linewidths=0.0,
                label="all map nodes",
                zorder=1,
            )
        if touched_nodes:
            ax.scatter(
                [p[0] for p in touched_nodes],
                [p[1] for p in touched_nodes],
                s=16,
                marker="s",
                c="#00bcd4",
                alpha=0.85,
                linewidths=0.0,
                label="nodes touched by aligned paths",
                zorder=2,
            )

        pre_label_drawn = False
        post_label_drawn = False
        for aid in actor_ids:
            p_pre = pre_xy_by_id.get(aid, [])
            p_post = post_xy_by_id.get(aid, [])
            if p_pre:
                ax.plot(
                    [p[0] for p in p_pre],
                    [p[1] for p in p_pre],
                    color="#1f77b4",
                    linestyle="--",
                    linewidth=2.1 if individual_mode else 1.1,
                    alpha=0.95 if individual_mode else 0.38,
                    label=None if pre_label_drawn else "GT from YAML + offsets (pre-align)",
                    zorder=4,
                )
                pre_label_drawn = True
                if individual_mode:
                    ax.scatter([p_pre[0][0]], [p_pre[0][1]], s=50, c="#1f77b4", marker="o", zorder=5, label="pre start")
                    ax.scatter([p_pre[-1][0]], [p_pre[-1][1]], s=50, c="#1f77b4", marker="x", zorder=5, label="pre end")
            if p_post:
                ax.plot(
                    [p[0] for p in p_post],
                    [p[1] for p in p_post],
                    color="#e53935",
                    linewidth=2.3 if individual_mode else 1.3,
                    alpha=0.98 if individual_mode else 0.45,
                    label=None if post_label_drawn else "post alignment/refinement",
                    zorder=6,
                )
                post_label_drawn = True
                if individual_mode:
                    ax.scatter([p_post[0][0]], [p_post[0][1]], s=58, c="#e53935", marker="o", zorder=7, label="post start")
                    ax.scatter([p_post[-1][0]], [p_post[-1][1]], s=58, c="#e53935", marker="x", zorder=7, label="post end")

        if merged:
            pad_val = max(0.0, float(pad)) * 1.2
            ax.set_xlim(merged[0] - pad_val, merged[1] + pad_val)
            ax.set_ylim(merged[2] - pad_val, merged[3] + pad_val)
        if invert_plot_y:
            ax.invert_yaxis()

    info = [
        f"background: {'CARLA captured BEV' if used_carla_capture else 'fallback'}",
        f"actors: {len(actor_ids)}",
        f"all nodes: {len(all_nodes)}",
        f"touched nodes: {len(touched_nodes)}",
        f"touch radius: {float(match_radius):.2f} m",
    ]
    ax.text(
        0.02,
        0.02,
        "\n".join(info),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#222222",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.82),
        zorder=10,
    )

    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout(rect=[0.0, 0.0, 0.76, 1.0])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_ego_alignment_viz(
    ego_idx: int,
    pre_align_traj: List[Waypoint],
    post_align_traj: List[Waypoint],
    map_lines: List[List[Tuple[float, float]]] | None,
    map_line_records: List[Dict[str, object]] | None,
    out_path: Path,
    xml_tx: float = 0.0,
    xml_ty: float = 0.0,
    pad: float = 24.0,
    invert_plot_y: bool = False,
) -> None:
    """
    Visualize ego trajectory before/after spawn preprocess alignment.
    pre_align_traj should be the original GT+global-transform path (before preprocess).
    """
    if plt is None:
        raise RuntimeError("matplotlib is required for visualization; install matplotlib")

    if not pre_align_traj and not post_align_traj:
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Ego {ego_idx}: GT+offset vs aligned/refined")

    pre_x = [wp.x + float(xml_tx) for wp in pre_align_traj]
    pre_y = [wp.y + float(xml_ty) for wp in pre_align_traj]
    post_x = [wp.x + float(xml_tx) for wp in post_align_traj]
    post_y = [wp.y + float(xml_ty) for wp in post_align_traj]
    xs_all = pre_x + post_x
    ys_all = pre_y + post_y

    if map_lines:
        for line in map_lines:
            if len(line) < 2:
                continue
            xs, ys = zip(*line)
            ax.plot(xs, ys, color="#c8c8c8", linewidth=0.7, alpha=0.6, zorder=0)

        # Add lane direction arrows near the ego path ROI.
        # Uses lane metadata when available (road_id/lane_id/dir_sign).
        qx: List[float] = []
        qy: List[float] = []
        qu: List[float] = []
        qv: List[float] = []
        if xs_all and ys_all:
            roi_pad = max(8.0, float(pad) * 1.1)
            minx = min(xs_all) - roi_pad
            maxx = max(xs_all) + roi_pad
            miny = min(ys_all) - roi_pad
            maxy = max(ys_all) + roi_pad
        else:
            minx = miny = -float("inf")
            maxx = maxy = float("inf")

        recs = map_line_records or []
        if not recs:
            recs = [{"points": ln, "dir_sign": None} for ln in map_lines]

        for rec in recs:
            line = rec.get("points") if isinstance(rec, dict) else None
            if not isinstance(line, list) or len(line) < 3:
                continue
            dir_sign = rec.get("dir_sign") if isinstance(rec, dict) else None
            try:
                dir_sign = int(dir_sign) if dir_sign is not None else None
            except Exception:
                dir_sign = None
            if dir_sign not in (-1, 1):
                # Old cache or unknown direction metadata -> skip to avoid misleading arrows.
                continue
            xs = [p[0] for p in line]
            ys = [p[1] for p in line]
            if max(xs) < minx or min(xs) > maxx or max(ys) < miny or min(ys) > maxy:
                continue

            stride = max(6, len(line) // 20)
            start = stride // 2
            for i in range(start, len(line) - 1, stride):
                if dir_sign > 0:
                    x0, y0 = line[i]
                    x1, y1 = line[i + 1]
                else:
                    x0, y0 = line[i + 1]
                    x1, y1 = line[i]
                dx = x1 - x0
                dy = y1 - y0
                seg = math.hypot(dx, dy)
                if seg < 0.2:
                    continue
                # Keep arrows readable and consistent.
                arr_len = min(1.8, max(0.9, 0.55 * seg))
                qx.append(x0)
                qy.append(y0)
                qu.append((dx / seg) * arr_len)
                qv.append((dy / seg) * arr_len)

        if qx:
            ax.quiver(
                qx,
                qy,
                qu,
                qv,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color="#9a9a9a",
                alpha=0.65,
                width=0.0015,
                headwidth=3.8,
                headlength=4.5,
                headaxislength=4.0,
                zorder=1,
                label="lane direction",
            )
        elif map_lines:
            ax.text(
                0.02,
                0.985,
                "lane-direction arrows skipped (cache lacks lane metadata)",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#666666",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
                zorder=9,
            )

    if pre_x and pre_y:
        ax.plot(
            pre_x,
            pre_y,
            color="#1f77b4",
            linewidth=2.0,
            linestyle="--",
            alpha=0.9,
            label="GT from YAML + offsets (pre-align)",
            zorder=2,
        )
        ax.scatter([pre_x[0]], [pre_y[0]], s=55, color="#1f77b4", marker="o", zorder=3, label="pre start")
        ax.scatter([pre_x[-1]], [pre_y[-1]], s=55, color="#1f77b4", marker="x", zorder=3, label="pre end")

    if post_x and post_y:
        ax.plot(
            post_x,
            post_y,
            color="#d62728",
            linewidth=2.2,
            alpha=0.95,
            label="Post alignment/refinement",
            zorder=4,
        )
        ax.scatter([post_x[0]], [post_y[0]], s=65, color="#d62728", marker="o", zorder=5, label="post start")
        ax.scatter([post_x[-1]], [post_y[-1]], s=65, color="#d62728", marker="x", zorder=5, label="post end")

    min_len = min(len(pre_align_traj), len(post_align_traj))
    shift_vals: List[float] = []
    if min_len > 0:
        stride = max(1, min_len // 120)
        for i in range(0, min_len, stride):
            x0 = pre_align_traj[i].x + float(xml_tx)
            y0 = pre_align_traj[i].y + float(xml_ty)
            x1 = post_align_traj[i].x + float(xml_tx)
            y1 = post_align_traj[i].y + float(xml_ty)
            shift_vals.append(math.hypot(x1 - x0, y1 - y0))
            ax.plot([x0, x1], [y0, y1], color="#444444", alpha=0.25, linewidth=0.7, zorder=1)
        if not shift_vals:
            shift_vals.append(0.0)

    if xs_all and ys_all:
        pad_val = max(0.0, float(pad)) * 1.15
        ax.set_xlim(min(xs_all) - pad_val, max(xs_all) + pad_val)
        ax.set_ylim(min(ys_all) - pad_val, max(ys_all) + pad_val)

    if shift_vals:
        txt = (
            f"paired points: {min_len}\n"
            f"shift median: {_median(shift_vals):.3f} m\n"
            f"shift max: {max(shift_vals):.3f} m"
        )
        ax.text(
            0.02,
            0.02,
            txt,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8),
            zorder=10,
        )

    if invert_plot_y:
        ax.invert_yaxis()

    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_actor_raw_yaml_viz(
    actor_id: int,
    points_by_subdir: Dict[str, List[Tuple[float, float, float]]],
    map_lines: List[List[Tuple[float, float]]] | None,
    out_path: Path,
    pad: float = 20.0,
    invert_plot_y: bool = False,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for visualization; install matplotlib")

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Actor {actor_id} raw YAML points by subfolder")

    # Map layer
    if map_lines:
        for line in map_lines:
            if len(line) < 2:
                continue
            xs, ys = zip(*line)
            ax.plot(xs, ys, color="#cccccc", linewidth=0.7, alpha=0.6, zorder=0)

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]

    xs_all: List[float] = []
    ys_all: List[float] = []
    for idx, (subdir, pts) in enumerate(sorted(points_by_subdir.items(), key=lambda kv: _yaml_dir_sort_key(Path(kv[0])))):
        if not pts:
            continue
        pts_sorted = sorted(pts, key=lambda p: p[2])
        xs = [p[0] for p in pts_sorted]
        ys = [p[1] for p in pts_sorted]
        xs_all.extend(xs)
        ys_all.extend(ys)
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.8, zorder=2)
        ax.scatter(xs, ys, s=14, color=color, marker=marker, alpha=0.85, zorder=3, label=f"{subdir} (n={len(xs)})")
        # Start/end annotations
        ax.scatter([xs[0]], [ys[0]], s=40, color=color, marker="o", zorder=4)
        ax.scatter([xs[-1]], [ys[-1]], s=40, color=color, marker="x", zorder=4)
        ax.annotate(f"t={pts_sorted[0][2]:.1f}", (xs[0], ys[0]), textcoords="offset points", xytext=(6, 6), fontsize=8, color=color)
        ax.annotate(f"t={pts_sorted[-1][2]:.1f}", (xs[-1], ys[-1]), textcoords="offset points", xytext=(6, -10), fontsize=8, color=color)

    if xs_all and ys_all:
        pad_val = max(0.0, float(pad))
        ax.set_xlim(min(xs_all) - pad_val, max(xs_all) + pad_val)
        ax.set_ylim(min(ys_all) - pad_val, max(ys_all) + pad_val)

    if invert_plot_y:
        ax.invert_yaxis()

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------- CLI ---------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert V2XPnP YAML logs to CARLA route XML + manifest")
    p.add_argument("--scenario-dir", required=True, help="Path to the scenario folder containing subfolders with YAML frames")
    p.add_argument(
        "--subdir",
        default="all",
        help=(
            "Specific subfolder inside scenario-dir to use (e.g., -1). "
            "Use 'all' to process all subfolders for actor locations. If omitted and multiple "
            "subfolders exist, behavior is the same. Non-negative subfolders produce ego routes; "
            "negative subfolders contribute actors only."
        ),
    )
    p.add_argument("--out-dir", default=None, help="Output directory (default: <scenario-dir>/carla_log_export)")
    p.add_argument("--route-id", default="0", help="Route id to assign to ego and actors (default: 0)")
    p.add_argument("--town", default="ucla_v2", help="CARLA town/map name to embed in XML (default: ucla_v2)")
    p.add_argument("--ego-name", default="ego", help="Name for ego vehicle")
    p.add_argument("--ego-model", default="vehicle.lincoln.mkz2017", help="Blueprint for ego vehicle")
    p.add_argument("--dt", type=float, default=0.1, help="Timestep spacing in seconds (for speed estimation)")
    p.add_argument(
        "--encode-timing",
        action="store_true",
        help="Embed per-waypoint timing in XML using frame index * dt (enables log replay).",
    )
    p.add_argument(
        "--maximize-safe-early-spawn",
        dest="maximize_safe_early_spawn",
        action="store_true",
        help=(
            "For each late-detected actor, choose the earliest safe spawn time between "
            "scenario start and first detection, while enforcing strict non-interference "
            "against other actor trajectories."
        ),
    )
    p.add_argument(
        "--no-maximize-safe-early-spawn",
        dest="maximize_safe_early_spawn",
        action="store_false",
        help="Disable safe early-spawn optimization.",
    )
    p.set_defaults(maximize_safe_early_spawn=True)
    p.add_argument(
        "--early-spawn-safety-margin",
        type=float,
        default=0.25,
        help="Extra safety margin (meters) added to actor radii for early-spawn interference checks.",
    )
    p.add_argument(
        "--early-spawn-report",
        default=None,
        help="Optional JSON path for early-spawn optimization diagnostics.",
    )
    p.add_argument(
        "--maximize-safe-late-despawn",
        dest="maximize_safe_late_despawn",
        action="store_true",
        help=(
            "Keep as many actors as possible alive after their last timestamp by holding "
            "their final waypoint until scenario horizon, with strict non-interference checks."
        ),
    )
    p.add_argument(
        "--no-maximize-safe-late-despawn",
        dest="maximize_safe_late_despawn",
        action="store_false",
        help="Disable safe late-despawn hold optimization.",
    )
    p.set_defaults(maximize_safe_late_despawn=True)
    p.add_argument(
        "--late-despawn-safety-margin",
        type=float,
        default=0.25,
        help="Extra safety margin (meters) added to actor radii for late-despawn hold checks.",
    )
    p.add_argument(
        "--late-despawn-report",
        default=None,
        help="Optional JSON path for late-despawn optimization diagnostics.",
    )
    p.add_argument("--tx", type=float, default=0.0, help="Translation X to apply to all coordinates")
    p.add_argument("--ty", type=float, default=0.0, help="Translation Y to apply to all coordinates")
    p.add_argument("--tz", type=float, default=0.0, help="Translation Z to apply to all coordinates")
    p.add_argument("--xml-tx", type=float, default=0.0, help="Additional X offset applied only when writing XML outputs")
    p.add_argument("--xml-ty", type=float, default=0.0, help="Additional Y offset applied only when writing XML outputs")
    p.add_argument(
        "--coord-json",
        default="/data2/marco/CoLMDriver/v2xpnp/map/ucla_map_offset_carla.json",
        help="Optional JSON file containing transform keys like tx, ty, theta_deg/rad, flip_y; applied to all coordinates",
    )
    p.add_argument("--yaw-deg", type=float, default=0.0, help="Global yaw rotation (degrees, applied before translation)")
    p.add_argument("--snap-to-road", action="store_true", default=True, help="Enable road snapping for actors (defaults to on)")
    p.add_argument("--no-ego", action="store_true", help="Skip writing ego_route.xml")
    p.add_argument(
        "--ego-only",
        action="store_true",
        help="Ignore all non-ego actors and export/process only ego routes.",
    )
    p.add_argument("--gif", action="store_true", help="Generate GIF visualization")
    p.add_argument("--gif-path", default=None, help="Path for GIF (default: <out-dir>/replay.gif)")
    p.add_argument("--paths-png", default=None, help="If set, render a single PNG with each actor's full path as a polyline")
    p.add_argument(
        "--actor-yaw-viz-ids",
        default="",
        help="Comma/space-separated actor ids to plot GT vs XML yaw over the CARLA map.",
    )
    p.add_argument(
        "--actor-yaw-viz-dir",
        default=None,
        help="Output directory for actor yaw visualizations (default: <out-dir>/actor_yaw_viz).",
    )
    p.add_argument(
        "--actor-yaw-viz-step",
        type=int,
        default=10,
        help="Stride for yaw arrows in actor visualizations (default: 10).",
    )
    p.add_argument(
        "--actor-yaw-viz-arrow-len",
        type=float,
        default=0.8,
        help="Arrow length (meters) for yaw visualizations (default: 0.8).",
    )
    p.add_argument(
        "--actor-yaw-viz-pad",
        type=float,
        default=5.0,
        help="Padding (meters) around GT/XML path extents for actor yaw visualizations.",
    )
    p.add_argument(
        "--actor-raw-yaml-viz-ids",
        default="",
        help="Comma/space-separated actor ids to plot raw YAML points by subfolder.",
    )
    p.add_argument(
        "--actor-raw-yaml-viz-dir",
        default=None,
        help="Output directory for raw YAML actor visualizations (default: <out-dir>/actor_raw_yaml_viz).",
    )
    p.add_argument(
        "--actor-raw-yaml-viz-pad",
        type=float,
        default=20.0,
        help="Padding (meters) around raw YAML points for actor visualizations.",
    )
    p.add_argument(
        "--ego-alignment-viz",
        action="store_true",
        help="Plot ego GT path (YAML + offsets) vs post-alignment/refinement path.",
    )
    p.add_argument(
        "--ego-alignment-viz-dir",
        default=None,
        help="Output directory for ego alignment visualizations (default: <out-dir>/ego_alignment_viz).",
    )
    p.add_argument(
        "--ego-alignment-viz-pad",
        type=float,
        default=24.0,
        help="Padding (meters) around ego pre/post alignment paths (default: 24).",
    )
    p.add_argument(
        "--ego-alignment-bev-viz",
        action="store_true",
        help="Write ego alignment overlay on a CARLA BEV/raster image with map-node highlights.",
    )
    p.add_argument(
        "--ego-alignment-bev-viz-dir",
        default=None,
        help="Output directory for ego BEV alignment visualizations (default: <out-dir>/ego_alignment_bev_viz).",
    )
    p.add_argument(
        "--actor-alignment-bev-viz",
        action="store_true",
        help="Write actor alignment BEV visualizations (all actors, subsets, and per-actor).",
    )
    p.add_argument(
        "--actor-alignment-bev-viz-dir",
        default=None,
        help="Output directory for actor BEV alignment visualizations (default: <out-dir>/actor_alignment_bev_viz).",
    )
    p.add_argument(
        "--ego-alignment-bev-node-step",
        type=int,
        default=2,
        help="Stride when sampling map nodes for ego BEV viz boxes (default: 2).",
    )
    p.add_argument(
        "--ego-alignment-bev-match-radius",
        type=float,
        default=1.8,
        help="Radius (m) used to mark map nodes touched by aligned ego path (default: 1.8).",
    )
    p.add_argument(
        "--ego-alignment-bev-capture-from-carla",
        action="store_true",
        help="Capture a real top-down CARLA RGB image for ego BEV alignment viz (default: enabled).",
    )
    p.add_argument(
        "--no-ego-alignment-bev-capture-from-carla",
        dest="ego_alignment_bev_capture_from_carla",
        action="store_false",
        help="Disable CARLA RGB capture and use map image/vector fallback only.",
    )
    p.set_defaults(ego_alignment_bev_capture_from_carla=True)
    p.add_argument(
        "--ego-alignment-bev-capture-width",
        type=int,
        default=2048,
        help="Captured CARLA BEV width in pixels (default: 2048).",
    )
    p.add_argument(
        "--ego-alignment-bev-capture-height",
        type=int,
        default=2048,
        help="Captured CARLA BEV height in pixels (default: 2048).",
    )
    p.add_argument(
        "--ego-alignment-bev-capture-fov",
        type=float,
        default=70.0,
        help="Camera FOV (deg) for CARLA BEV capture (default: 70).",
    )
    p.add_argument(
        "--ego-alignment-bev-capture-margin",
        type=float,
        default=1.18,
        help="Margin scale around ego bounds when capturing CARLA BEV image (default: 1.18).",
    )
    p.add_argument("--map-pkl", default=None, help="Optional pickle containing vector map polylines to overlay")
    p.add_argument("--use-carla-map", default=True, action="store_true", help="Connect to CARLA to fetch map polylines for overlay")
    p.add_argument("--carla-host", default="127.0.0.1", help="CARLA host (default: 127.0.0.1)")
    p.add_argument("--carla-port", type=int, default=2010, help="CARLA port (default: 2010)")
    p.add_argument("--carla-sample", type=float, default=2.0, help="Waypoint sampling distance in meters (default: 2.0)")
    p.add_argument("--carla-cache", default=None, help="Path to cache map polylines (default: <out-dir>/carla_map_cache.pkl)")
    p.add_argument("--expected-town", default="ucla_v2", help="Assert CARLA map name contains this string when using --use-carla-map")
    p.add_argument("--axis-pad", type=float, default=10.0, help="Padding (meters) around actor/ego extents for visualization axes")
    p.add_argument("--flip-y", action="store_true", help="Mirror dataset Y axis and negate yaw (useful if overlay appears upside-down)")
    p.add_argument("--invert-plot-y", action="store_true", help="Invert matplotlib Y axis for visualization only")
    p.add_argument(
        "--spawn-viz",
        action="store_true",
        help="Generate a spawn-vs-aligned visualization over CARLA map and XODR layers.",
    )
    p.add_argument(
        "--spawn-viz-path",
        default=None,
        help="Output path for spawn-vs-aligned visualization (default: <out-dir>/spawn_alignment_viz.png).",
    )
    p.add_argument(
        "--xodr",
        default=None,
        help="Path to the OpenDRIVE XODR file for spawn visualization overlay.",
    )
    p.add_argument(
        "--xodr-step",
        type=float,
        default=2.0,
        help="Sampling step size (meters) for XODR geometry (default: 2.0).",
    )
    p.add_argument(
        "--map-image",
        default=None,
        help="Optional raster map image to use as the CARLA background layer (PNG/JPG).",
    )
    p.add_argument(
        "--map-image-bounds",
        nargs=4,
        type=float,
        default=None,
        metavar=("MINX", "MAXX", "MINY", "MAXY"),
        help="World bounds for the map image (minx maxx miny maxy). If omitted, bounds are inferred.",
    )
    p.add_argument(
        "--spawn-preprocess",
        action="store_true",
        help="Run CARLA-in-the-loop spawn preprocessing to improve actor spawn success.",
    )
    p.add_argument(
        "--spawn-preprocess-report",
        default=None,
        help="Optional JSON report path for spawn preprocessing results.",
    )
    p.add_argument(
        "--spawn-preprocess-max-shift",
        type=float,
        default=4.0,
        help="Maximum XY shift (meters) when generating spawn candidates (default: 4.0).",
    )
    p.add_argument(
        "--spawn-preprocess-random-samples",
        type=int,
        default=80,
        help="Number of random candidate offsets per actor (default: 80).",
    )
    p.add_argument(
        "--spawn-preprocess-fail-viz",
        action="store_true",
        help="Generate visualization for actors that failed to spawn (over CARLA map).",
    )
    p.add_argument(
        "--spawn-preprocess-fail-viz-dir",
        default=None,
        help="Output directory for failed spawn visualizations (default: <out-dir>/spawn_preprocess_fail_viz).",
    )
    p.add_argument(
        "--spawn-preprocess-fail-viz-window",
        type=float,
        default=60.0,
        help="Window size (meters) for per-actor failed spawn plots (default: 60).",
    )
    p.add_argument(
        "--spawn-preprocess-fail-viz-dpi",
        type=int,
        default=220,
        help="DPI for failed spawn visualizations (default: 220).",
    )
    p.add_argument(
        "--spawn-preprocess-fail-viz-sample",
        type=float,
        default=1.0,
        help="CARLA map sampling distance for failed spawn visualizations (default: 1.0).",
    )
    p.add_argument(
        "--spawn-preprocess-debug-radius",
        type=float,
        default=30.0,
        help="Radius (meters) for collecting nearby actors/env objects in failed spawn debug (default: 30).",
    )
    p.add_argument(
        "--spawn-preprocess-debug-max-items",
        type=int,
        default=10,
        help="Max nearby actors/env objects to record per failed spawn (default: 10).",
    )
    p.add_argument(
        "--spawn-preprocess-grid",
        default="0.0,0.2,-0.2,0.4,-0.4,0.8,-0.8,1.2,-1.2",
        help="Comma/space-separated XY offsets (meters) for local candidate grid.",
    )
    p.add_argument(
        "--spawn-preprocess-sample-dt",
        type=float,
        default=0.5,
        help="Sampling timestep (seconds) for collision scoring (default: 0.5).",
    )
    p.add_argument(
        "--spawn-preprocess-grid-size",
        type=float,
        default=5.0,
        help="Spatial hash grid size (meters) for collision checks (default: 5.0).",
    )
    p.add_argument(
        "--spawn-preprocess-max-candidates",
        type=int,
        default=60,
        help="Maximum candidate offsets per actor (default: 60).",
    )
    p.add_argument(
        "--spawn-preprocess-collision-weight",
        type=float,
        default=50.0,
        help="Weight for collision penalty in candidate scoring (default: 50.0).",
    )
    p.add_argument(
        "--spawn-preprocess-verbose",
        action="store_true",
        help="Enable verbose spawn preprocessing logs.",
    )
    p.add_argument(
        "--spawn-preprocess-align",
        action="store_true",
        help="Enable multi-waypoint alignment candidates during spawn preprocess (default: enabled).",
    )
    p.add_argument(
        "--no-spawn-preprocess-align",
        dest="spawn_preprocess_align",
        action="store_false",
        help="Disable multi-waypoint alignment candidates during spawn preprocess.",
    )
    p.set_defaults(spawn_preprocess_align=True)
    p.add_argument(
        "--spawn-preprocess-align-samples",
        type=int,
        default=12,
        help="Number of trajectory samples for alignment candidate generation.",
    )
    p.add_argument(
        "--spawn-preprocess-align-windows",
        type=int,
        default=3,
        help="Number of trajectory windows for slice-based alignment candidates.",
    )
    p.add_argument(
        "--spawn-preprocess-align-intent-margin",
        type=float,
        default=0.8,
        help="Distance margin (m) used to infer sidewalk vs road intent for walkers.",
    )
    p.add_argument(
        "--spawn-preprocess-align-neighbor-radius",
        type=float,
        default=6.0,
        help="Radius (m) for neighbor-aware alignment bias among nearby vehicles.",
    )
    p.add_argument(
        "--spawn-preprocess-align-neighbor-weight",
        type=float,
        default=0.15,
        help="Weight for neighbor offset coherence during global candidate selection.",
    )
    p.add_argument(
        "--spawn-preprocess-refine-piecewise",
        action="store_true",
        help="After global actor shift selection, run bounded per-waypoint refinement (default: enabled).",
    )
    p.add_argument(
        "--no-spawn-preprocess-refine-piecewise",
        dest="spawn_preprocess_refine_piecewise",
        action="store_false",
        help="Disable second-stage per-waypoint actor refinement.",
    )
    p.set_defaults(spawn_preprocess_refine_piecewise=True)
    p.add_argument(
        "--spawn-preprocess-refine-max-local",
        type=float,
        default=0.8,
        help="Max local deviation (m) from global actor shift in piecewise refinement (default: 0.8).",
    )
    p.add_argument(
        "--spawn-preprocess-refine-smooth-window",
        type=int,
        default=7,
        help="Smoothing window (waypoints) for actor piecewise refinement (default: 7).",
    )
    p.add_argument(
        "--spawn-preprocess-refine-max-step-delta",
        type=float,
        default=0.35,
        help="Max change (m) between adjacent actor offsets in piecewise refinement (default: 0.35).",
    )
    p.add_argument(
        "--spawn-preprocess-refine-collision-slack",
        type=float,
        default=0.0,
        help="Allowable collision score increase when accepting actor refinement (default: 0.0).",
    )
    p.add_argument(
        "--spawn-preprocess-bridge-max-gap-steps",
        type=int,
        default=6,
        help="Max consecutive waypoints treated as a map-projection gap for lane-bridge interpolation (default: 6). Set 0 to disable.",
    )
    p.add_argument(
        "--spawn-preprocess-bridge-straight-thresh-deg",
        type=float,
        default=18.0,
        help="Max heading change (deg) across a bridged segment; larger turns will not be bridged (default: 18).",
    )
    p.add_argument(
        "--spawn-preprocess-align-ego",
        action="store_true",
        help="Align ego trajectories during spawn preprocess (default: enabled).",
    )
    p.add_argument(
        "--no-spawn-preprocess-align-ego",
        dest="spawn_preprocess_align_ego",
        action="store_false",
        help="Disable ego trajectory alignment during spawn preprocess.",
    )
    p.set_defaults(spawn_preprocess_align_ego=True)
    p.add_argument(
        "--spawn-preprocess-align-ego-piecewise",
        action="store_true",
        help="Use per-waypoint ego alignment (not a single global offset) during spawn preprocess (default: enabled).",
    )
    p.add_argument(
        "--no-spawn-preprocess-align-ego-piecewise",
        dest="spawn_preprocess_align_ego_piecewise",
        action="store_false",
        help="Use only a single global ego offset during spawn preprocess.",
    )
    p.set_defaults(spawn_preprocess_align_ego_piecewise=True)
    p.add_argument(
        "--spawn-preprocess-align-ego-smooth-window",
        type=int,
        default=9,
        help="Smoothing window (waypoints) for piecewise ego offset profile (default: 9).",
    )
    p.add_argument(
        "--spawn-preprocess-align-ego-max-step-delta",
        type=float,
        default=0.45,
        help="Max change (m) in ego offset between adjacent waypoints for piecewise mode (default: 0.45).",
    )
    p.add_argument(
        "--snap-ego-to-lane",
        action="store_true",
        default=False,
        help=(
            "When aligning ego trajectories, lock snapping to a continuous forward lane-center path "
            "(prevents lane changes while still allowing turns)."
        ),
    )
    p.add_argument(
        "--spawn-preprocess-normalize-z",
        action="store_true",
        default=True,
        help="Use ground projection when validating spawn candidates (default: on).",
    )
    p.add_argument(
        "--no-spawn-preprocess-normalize-z",
        dest="spawn_preprocess_normalize_z",
        action="store_false",
        help="Disable ground projection during spawn candidate validation.",
    )
    p.add_argument("--run-custom-eval", action="store_true", help="After export, call tools/run_custom_eval.py with the generated routes dir")
    p.add_argument(
        "--eval-planner",
        default="",
        help="Planner for run_custom_eval (empty string means no planner flag; e.g., pass 'tcp' or 'log_replay')",
    )
    p.add_argument("--eval-port", type=int, default=2014, help="CARLA port for run_custom_eval (default: 2014)")
    return p.parse_args()


def _yaml_dir_sort_key(path: Path) -> Tuple[int, object]:
    name = path.name
    try:
        return (0, int(name))
    except Exception:
        return (1, name)


def _is_negative_subdir(path: Path) -> bool:
    try:
        return int(path.name) < 0
    except Exception:
        return False


def _parse_id_list(raw: str) -> List[int]:
    ids: List[int] = []
    for token in re.split(r"[,\s]+", raw.strip()):
        if not token:
            continue
        try:
            ids.append(int(token))
        except Exception:
            continue
    return ids


def pick_yaml_dirs(scenario_dir: Path, chosen: str | None) -> List[Path]:
    subdirs = [d for d in scenario_dir.iterdir() if d.is_dir()]

    if chosen:
        if str(chosen).lower() == "all":
            yaml_subdirs = [d for d in subdirs if list_yaml_timesteps(d)]
            numeric_subdirs = [d for d in yaml_subdirs if re.fullmatch(r"-?\d+", d.name or "")]
            if numeric_subdirs:
                yaml_subdirs = numeric_subdirs
            if not yaml_subdirs:
                raise SystemExit(f"No YAML subfolders found under {scenario_dir}")
            return sorted(yaml_subdirs, key=_yaml_dir_sort_key)
        cand = scenario_dir / chosen
        if not cand.is_dir():
            raise SystemExit(f"--subdir {chosen} not found under {scenario_dir}")
        return [cand]

    if list_yaml_timesteps(scenario_dir):
        return [scenario_dir]

    yaml_subdirs = [d for d in subdirs if list_yaml_timesteps(d)]
    numeric_subdirs = [d for d in yaml_subdirs if re.fullmatch(r"-?\d+", d.name or "")]
    if numeric_subdirs:
        yaml_subdirs = numeric_subdirs
    if len(yaml_subdirs) == 1:
        return yaml_subdirs
    if len(yaml_subdirs) > 1:
        return sorted(yaml_subdirs, key=_yaml_dir_sort_key)

    raise SystemExit(f"No YAML files found under {scenario_dir}")


def _extract_map_lines(obj, depth=0, out: List[List[Tuple[float, float]]] | None = None):
    """Heuristic extractor for vector map polylines from arbitrary pickle structures."""
    if out is None:
        out = []
    if obj is None or depth > 10:
        return out

    if isinstance(obj, dict):
        if "x" in obj and "y" in obj:
            try:
                out.append([(float(obj["x"]), float(obj["y"]))])
            except Exception:
                pass
        for v in obj.values():
            _extract_map_lines(v, depth + 1, out)
        return out

    if isinstance(obj, (list, tuple)):
        if len(obj) >= 2 and all(hasattr(it, "__len__") and len(it) >= 2 for it in obj if it is not None):
            try:
                pts = [(float(p[0]), float(p[1])) for p in obj if p is not None and len(p) >= 2]
                if len(pts) >= 2:
                    out.append(pts)
                    return out
            except Exception:
                pass
        for v in obj:
            _extract_map_lines(v, depth + 1, out)
        return out

    if hasattr(obj, "x") and hasattr(obj, "y"):
        try:
            out.append([(float(obj.x), float(obj.y))])
        except Exception:
            pass
        return out

    if hasattr(obj, "__dict__"):
        _extract_map_lines(obj.__dict__, depth + 1, out)
    return out


def _extract_map_line_records(obj) -> List[Dict[str, object]]:
    """
    Extract map line records with optional lane metadata.
    Returns records shaped as:
      {"points": [(x,y), ...], "road_id": int|None, "lane_id": int|None, "dir_sign": -1|1|None}
    """
    records: List[Dict[str, object]] = []
    if isinstance(obj, dict):
        raw = obj.get("line_records")
        if isinstance(raw, list):
            for rec in raw:
                if not isinstance(rec, dict):
                    continue
                pts_raw = rec.get("points")
                if not isinstance(pts_raw, (list, tuple)):
                    continue
                pts: List[Tuple[float, float]] = []
                for p in pts_raw:
                    if not isinstance(p, (list, tuple)) or len(p) < 2:
                        continue
                    try:
                        pts.append((float(p[0]), float(p[1])))
                    except Exception:
                        continue
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
                records.append(
                    {
                        "points": pts,
                        "road_id": road_id,
                        "lane_id": lane_id,
                        "dir_sign": dir_sign,
                    }
                )
    if records:
        return records

    # Fallback for old caches / generic pickle inputs.
    lines = _extract_map_lines(obj, out=[])
    for line in lines:
        if len(line) >= 2:
            records.append({"points": line, "road_id": None, "lane_id": None, "dir_sign": None})
    return records


def _integrate_geometry(
    x0: float,
    y0: float,
    hdg: float,
    length: float,
    curv_fn,
    step: float,
) -> List[Tuple[float, float]]:
    if length <= 0.0:
        return []
    step = max(step, 0.1)
    n = max(1, int(math.ceil(length / step)))
    ds = length / n
    x = x0
    y = y0
    theta = hdg
    points = [(x, y)]
    for i in range(n):
        s_mid = (i + 0.5) * ds
        kappa = curv_fn(s_mid)
        theta_mid = theta + 0.5 * kappa * ds
        x += ds * math.cos(theta_mid)
        y += ds * math.sin(theta_mid)
        theta += kappa * ds
        points.append((x, y))
    return points


def _sample_geometry(geom: ET.Element, step: float) -> List[Tuple[float, float]]:
    x0 = float(geom.attrib.get("x", 0.0))
    y0 = float(geom.attrib.get("y", 0.0))
    hdg = float(geom.attrib.get("hdg", 0.0))
    length = float(geom.attrib.get("length", 0.0))

    child = next(iter(geom), None)
    if child is None:
        return [(x0, y0)]

    if child.tag == "line":
        curv_fn = lambda s: 0.0
    elif child.tag == "arc":
        curvature = float(child.attrib.get("curvature", 0.0))
        curv_fn = lambda s, k=curvature: k
    elif child.tag == "spiral":
        curv_start = float(child.attrib.get("curvStart", 0.0))
        curv_end = float(child.attrib.get("curvEnd", 0.0))

        def curv_fn(s: float, cs=curv_start, ce=curv_end, total=length) -> float:
            if total <= 0.0:
                return cs
            return cs + (ce - cs) * (s / total)
    else:
        curv_fn = lambda s: 0.0

    return _integrate_geometry(x0, y0, hdg, length, curv_fn, step)


def load_xodr_points(path: Path, step: float) -> List[Tuple[float, float]]:
    root = ET.parse(path).getroot()
    points: List[Tuple[float, float]] = []
    for geom in root.findall(".//planView/geometry"):
        points.extend(_sample_geometry(geom, step))
    return points


def _bounds_from_points(points: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float] | None:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), max(xs), min(ys), max(ys))


def _merge_bounds(bounds_list: Sequence[Tuple[float, float, float, float] | None]) -> Tuple[float, float, float, float] | None:
    mins = []
    maxs = []
    for b in bounds_list:
        if not b:
            continue
        mins.append((b[0], b[2]))
        maxs.append((b[1], b[3]))
    if not mins or not maxs:
        return None
    minx = min(m[0] for m in mins)
    miny = min(m[1] for m in mins)
    maxx = max(m[0] for m in maxs)
    maxy = max(m[1] for m in maxs)
    return (minx, maxx, miny, maxy)


def _plot_background_lines(ax, lines: List[List[Tuple[float, float]]], color: str, lw: float, alpha: float):
    for line in lines:
        if len(line) < 2:
            continue
        xs = [p[0] for p in line]
        ys = [p[1] for p in line]
        ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha, zorder=1)


def _crop_lines_to_bounds(
    lines: List[List[Tuple[float, float]]],
    bounds: Tuple[float, float, float, float],
) -> List[List[Tuple[float, float]]]:
    minx, maxx, miny, maxy = bounds
    cropped: List[List[Tuple[float, float]]] = []
    for line in lines:
        if len(line) < 2:
            continue
        keep = False
        for x, y in line:
            if minx <= x <= maxx and miny <= y <= maxy:
                keep = True
                break
        if keep:
            cropped.append(line)
    return cropped


def _bbox_corners_2d(bbox, tf) -> List[Tuple[float, float]]:
    corners = []
    try:
        ext = bbox.extent
        center = bbox.location
        for sx, sy in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
            loc = carla.Location(
                x=center.x + sx * ext.x,
                y=center.y + sy * ext.y,
                z=center.z,
            )
            world_loc = tf.transform(loc)
            corners.append((float(world_loc.x), float(world_loc.y)))
    except Exception:
        return []
    return corners


def _nearest_items(
    items: List[Dict[str, object]],
    center: carla.Location,
    max_dist: float,
    limit: int,
) -> List[Dict[str, object]]:
    out = []
    for item in items:
        loc = item.get("loc")
        if loc is None:
            continue
        try:
            dist = float(loc.distance(center))
        except Exception:
            continue
        if dist > max_dist:
            continue
        out.append((dist, item))
    out.sort(key=lambda x: x[0])
    results = []
    for dist, it in out[:limit]:
        payload = dict(it)
        payload["dist"] = float(dist)
        results.append(payload)
    return results


def _collect_spawn_debug(
    actor_id: int,
    base_wp: Waypoint,
    entry: Dict[str, object],
    world,
    world_map,
    blueprint,
    actor_items: List[Dict[str, object]],
    env_items: List[Dict[str, object]],
    max_dist: float,
    max_items: int,
    probe_yaw: bool,
) -> Dict[str, object]:
    debug: Dict[str, object] = {}
    loc = carla.Location(x=base_wp.x, y=base_wp.y, z=base_wp.z)
    ground_z = _resolve_ground_z(world, loc) if world is not None else None
    if ground_z is not None:
        debug["ground_z"] = float(ground_z)
        debug["z_delta"] = float(base_wp.z) - float(ground_z)
    try:
        wp_any = world_map.get_waypoint(loc, project_to_road=False, lane_type=carla.LaneType.Any) if world_map else None
        wp_drive = world_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving) if world_map else None
        wp_sidewalk = world_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Sidewalk) if world_map else None
    except Exception:
        wp_any = wp_drive = wp_sidewalk = None
    if wp_any is not None:
        debug["wp_any"] = {
            "road_id": int(getattr(wp_any, "road_id", -1)),
            "lane_id": int(getattr(wp_any, "lane_id", -1)),
            "lane_type": str(getattr(wp_any, "lane_type", "")),
            "is_junction": bool(getattr(wp_any, "is_junction", False)),
        }
    if wp_drive is not None:
        debug["wp_drive"] = {
            "road_id": int(getattr(wp_drive, "road_id", -1)),
            "lane_id": int(getattr(wp_drive, "lane_id", -1)),
            "lane_type": str(getattr(wp_drive, "lane_type", "")),
            "is_junction": bool(getattr(wp_drive, "is_junction", False)),
        }
    if wp_sidewalk is not None:
        debug["wp_sidewalk"] = {
            "road_id": int(getattr(wp_sidewalk, "road_id", -1)),
            "lane_id": int(getattr(wp_sidewalk, "lane_id", -1)),
            "lane_type": str(getattr(wp_sidewalk, "lane_type", "")),
            "is_junction": bool(getattr(wp_sidewalk, "is_junction", False)),
        }

    near_actors = _nearest_items(actor_items, loc, max_dist, limit=max_items)
    near_env = _nearest_items(env_items, loc, max_dist, limit=max_items)
    debug["nearest_actors"] = [
        {
            "id": it.get("id"),
            "type": it.get("type"),
            "dist": it.get("dist"),
            "bbox": it.get("bbox"),
        }
        for it in near_actors
    ]
    debug["nearest_env_objects"] = [
        {
            "id": it.get("id"),
            "type": it.get("type"),
            "dist": it.get("dist"),
            "bbox": it.get("bbox"),
        }
        for it in near_env
    ]

    probe_results = []
    if blueprint is not None and world is not None:
        for dz in (0.0, 0.2, 0.5, 1.0, 2.0):
            spawn_loc = carla.Location(x=loc.x, y=loc.y, z=loc.z + dz)
            spawn_tf = carla.Transform(
                spawn_loc,
                carla.Rotation(pitch=base_wp.pitch, yaw=base_wp.yaw, roll=base_wp.roll),
            )
            ok = False
            actor = None
            try:
                actor = world.try_spawn_actor(blueprint, spawn_tf)
                ok = actor is not None
            except Exception:
                ok = False
            if actor is not None:
                try:
                    actor.destroy()
                except Exception:
                    pass
            probe_results.append({"dz": float(dz), "ok": bool(ok)})

        if probe_yaw:
            yaw_results = []
            for dyaw in (-20.0, -10.0, -5.0, 5.0, 10.0, 20.0):
                spawn_tf = carla.Transform(
                    carla.Location(x=loc.x, y=loc.y, z=loc.z),
                    carla.Rotation(pitch=base_wp.pitch, yaw=base_wp.yaw + dyaw, roll=base_wp.roll),
                )
                ok = False
                actor = None
                try:
                    actor = world.try_spawn_actor(blueprint, spawn_tf)
                    ok = actor is not None
                except Exception:
                    ok = False
                if actor is not None:
                    try:
                        actor.destroy()
                    except Exception:
                        pass
                yaw_results.append({"dyaw": float(dyaw), "ok": bool(ok)})
            debug["probe_yaw"] = yaw_results

    debug["probe_z"] = probe_results
    return debug


def _plot_failed_spawn_visualizations(
    report: Dict[str, object],
    map_lines: List[List[Tuple[float, float]]],
    out_dir: Path,
    window_m: float,
    dpi: int,
) -> None:
    if plt is None:
        print("[WARN] matplotlib not available; skipping failed spawn visualization.")
        return

    actors = report.get("actors") or {}
    failed = []
    for actor_id, entry in actors.items():
        chosen = entry.get("chosen") or {}
        if chosen.get("status") != "no_valid_candidates":
            continue
        base = entry.get("spawn_base") or {}
        if not base:
            continue
        failed.append((actor_id, entry, base))

    if not failed:
        print("[INFO] No failed actors to visualize.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Overview plot
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect("equal", adjustable="box")
    if map_lines:
        _plot_background_lines(ax, map_lines, color="#9e9e9e", lw=0.6, alpha=0.5)
    xs = []
    ys = []
    for actor_id, entry, base in failed:
        x = float(base.get("x", 0.0))
        y = float(base.get("y", 0.0))
        xs.append(x)
        ys.append(y)
        ax.scatter([x], [y], c="#d62728", s=30, marker="x", zorder=5)
        ax.text(x, y, str(actor_id), fontsize=6, color="#111111", zorder=6)
    if xs and ys:
        pad = max(10.0, 0.5 * window_m)
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_title("Failed Spawns Overview")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "failed_spawn_overview.png", dpi=dpi)
    plt.close(fig)

    # Per-actor zoomed plots
    half = max(10.0, 0.5 * float(window_m))
    for actor_id, entry, base in failed:
        cx = float(base.get("x", 0.0))
        cy = float(base.get("y", 0.0))
        bounds = (cx - half, cx + half, cy - half, cy + half)
        local_lines = _crop_lines_to_bounds(map_lines, bounds) if map_lines else []

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_aspect("equal", adjustable="box")
        if local_lines:
            _plot_background_lines(ax, local_lines, color="#b0b0b0", lw=0.7, alpha=0.6)

        debug = entry.get("debug") or {}
        if patches is not None:
            for obj in debug.get("nearest_env_objects", []):
                poly = obj.get("bbox")
                if poly:
                    ax.add_patch(
                        patches.Polygon(
                            poly,
                            closed=True,
                            fill=False,
                            edgecolor="#ff9896",
                            linewidth=0.8,
                            alpha=0.8,
                            zorder=1,
                        )
                    )
            for obj in debug.get("nearest_actors", []):
                poly = obj.get("bbox")
                if poly:
                    ax.add_patch(
                        patches.Polygon(
                            poly,
                            closed=True,
                            fill=False,
                            edgecolor="#1f77b4",
                            linewidth=0.8,
                            alpha=0.8,
                            zorder=1,
                        )
                    )

        # Candidate points
        candidates = entry.get("candidates") or []
        invalid_x = []
        invalid_y = []
        valid_x = []
        valid_y = []
        for cand in candidates:
            loc = cand.get("spawn_loc")
            if not loc:
                continue
            if cand.get("valid"):
                valid_x.append(loc[0])
                valid_y.append(loc[1])
            else:
                invalid_x.append(loc[0])
                invalid_y.append(loc[1])
        if invalid_x:
            ax.scatter(invalid_x, invalid_y, s=8, c="#808080", alpha=0.45, label="invalid candidates", zorder=2)
        if valid_x:
            ax.scatter(valid_x, valid_y, s=12, c="#2ca02c", alpha=0.8, label="valid candidates", zorder=3)

        # Base spawn
        ax.scatter([cx], [cy], s=60, marker="x", c="#d62728", label="spawn base", zorder=5)

        # Best invalid
        best = entry.get("best_invalid_candidate") or {}
        if best:
            bl = best.get("spawn_loc")
            if bl:
                ax.scatter([bl[0]], [bl[1]], s=40, marker="o", c="#ff7f0e", label="best invalid", zorder=4)

        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_title(f"Failed Spawn Actor {actor_id}")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)

        meta = f"kind={entry.get('kind')} model={entry.get('model_used') or entry.get('model')}"
        stats = entry.get("candidate_stats") or {}
        detail = f"candidates={stats.get('total')} valid={stats.get('valid')} invalid={stats.get('invalid')}"
        ax.text(bounds[0], bounds[3], meta, fontsize=7, va="top")
        ax.text(bounds[0], bounds[3] - 0.05 * (bounds[3] - bounds[2]), detail, fontsize=7, va="top")
        if debug.get("probe_z"):
            z_ok = [str(r["dz"]) for r in debug.get("probe_z") if r.get("ok")]
            ax.text(
                bounds[0],
                bounds[3] - 0.10 * (bounds[3] - bounds[2]),
                f"probe_z_ok: {', '.join(z_ok) if z_ok else 'none'}",
                fontsize=7,
                va="top",
            )

        # Debug text: nearest actors/env objects
        lines = []
        near_actors = debug.get("nearest_actors") or []
        near_env = debug.get("nearest_env_objects") or []
        if near_actors:
            lines.append("nearest actors:")
            for item in near_actors[:5]:
                lines.append(
                    f"  id={item.get('id')} type={item.get('type')} d={item.get('dist'):.2f}"
                )
        if near_env:
            lines.append("nearest env:")
            for item in near_env[:5]:
                lines.append(
                    f"  id={item.get('id')} type={item.get('type')} d={item.get('dist'):.2f}"
                )
        if lines:
            ax.text(
                bounds[1],
                bounds[3],
                "\n".join(lines),
                fontsize=6,
                va="top",
                ha="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, linewidth=0.5),
            )

        out_path = out_dir / f"failed_actor_{actor_id}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)


def _plot_offset_annotation(
    ax,
    aligned_pt: Tuple[float, float],
    spawn_pt: Tuple[float, float],
    label: str | None = None,
):
    dx = spawn_pt[0] - aligned_pt[0]
    dy = spawn_pt[1] - aligned_pt[1]
    if dx == 0.0 and dy == 0.0:
        return

    x_step = (aligned_pt[0] + dx, aligned_pt[1])

    # Highlight the pre-alignment (spawn) reference point.
    ax.scatter(
        [spawn_pt[0]],
        [spawn_pt[1]],
        s=130,
        marker="X",
        c="#ff7f0e",
        edgecolors="#111111",
        linewidths=0.6,
        label=label,
        zorder=8,
    )

    # Draw axis-aligned offset components.
    ax.plot(
        [aligned_pt[0], x_step[0]],
        [aligned_pt[1], x_step[1]],
        color="#ff7f0e",
        linewidth=1.8,
        zorder=7,
    )
    ax.plot(
        [x_step[0], spawn_pt[0]],
        [x_step[1], spawn_pt[1]],
        color="#1f77b4",
        linewidth=1.8,
        zorder=7,
    )

    bbox = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
    ax.annotate(
        f"dx={dx:+.2f}m",
        xy=((aligned_pt[0] + x_step[0]) * 0.5, aligned_pt[1]),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        va="bottom",
        color="#ff7f0e",
        fontsize=9,
        bbox=bbox,
        zorder=9,
    )
    ax.annotate(
        f"dy={dy:+.2f}m",
        xy=(x_step[0], (x_step[1] + spawn_pt[1]) * 0.5),
        xytext=(6, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        color="#1f77b4",
        fontsize=9,
        bbox=bbox,
        zorder=9,
    )


def _plot_spawn_alignment(
    ax,
    aligned_points: Dict[int, Tuple[float, float]],
    spawn_points: Dict[int, Tuple[float, float]],
    actor_kind_by_id: Dict[int, str],
    ego_aligned: List[Tuple[float, float]],
    ego_spawn: List[Tuple[float, float]],
    title: str,
    show_offsets: bool = True,
    offset_pair: Tuple[Tuple[float, float], Tuple[float, float]] | None = None,
    offset_label: str | None = None,
):
    kind_markers = {
        "npc": "o",
        "static": "s",
        "walker": "^",
        "walker_static": "^",
    }
    aligned_color = "#2ca02c"
    spawn_color = "#d62728"

    for kind, marker in kind_markers.items():
        ids = [vid for vid, k in actor_kind_by_id.items() if k == kind and vid in aligned_points]
        if not ids:
            continue
        a_pts = [aligned_points[vid] for vid in ids]
        s_pts = [spawn_points[vid] for vid in ids if vid in spawn_points]
        ax.scatter(
            [p[0] for p in a_pts],
            [p[1] for p in a_pts],
            s=20,
            marker=marker,
            c=aligned_color,
            alpha=0.7,
            label=f"{kind} aligned",
            zorder=3,
        )
        if s_pts:
            ax.scatter(
                [p[0] for p in s_pts],
                [p[1] for p in s_pts],
                s=40,
                marker=marker,
                facecolors="none",
                edgecolors=spawn_color,
                linewidths=1.0,
                label=f"{kind} spawn",
                zorder=4,
            )

    if ego_aligned:
        ax.scatter(
            [p[0] for p in ego_aligned],
            [p[1] for p in ego_aligned],
            s=80,
            marker="*",
            c="#111111",
            label="ego aligned",
            zorder=5,
        )
    if ego_spawn:
        ax.scatter(
            [p[0] for p in ego_spawn],
            [p[1] for p in ego_spawn],
            s=110,
            marker="*",
            facecolors="none",
            edgecolors="#ff7f0e",
            linewidths=1.5,
            label="ego spawn",
            zorder=6,
        )

    if show_offsets:
        for vid, a_pt in aligned_points.items():
            s_pt = spawn_points.get(vid)
            if not s_pt:
                continue
            if a_pt == s_pt:
                continue
            ax.plot([a_pt[0], s_pt[0]], [a_pt[1], s_pt[1]], color="#555555", alpha=0.3, linewidth=0.8, zorder=2)

    if offset_pair is not None:
        _plot_offset_annotation(ax, offset_pair[0], offset_pair[1], label=offset_label)

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3)


def _pick_offset_reference(
    aligned_points: Dict[int, Tuple[float, float]],
    ref_points: Dict[int, Tuple[float, float]],
    ego_aligned: List[Tuple[float, float]],
    ego_ref: List[Tuple[float, float]],
) -> Tuple[Tuple[float, float], Tuple[float, float], str] | None:
    if ego_aligned and ego_ref:
        return ego_aligned[0], ego_ref[0], "ego"
    for vid in sorted(aligned_points.keys()):
        r_pt = ref_points.get(vid)
        if r_pt is None:
            continue
        return aligned_points[vid], r_pt, f"id {vid}"
    return None


def load_vector_map_from_pickle(path: Path) -> List[List[Tuple[float, float]]]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    return _extract_map_lines(obj, out=[])


def fetch_carla_map_lines(
    host: str,
    port: int,
    sample: float,
    cache_path: Path | None,
    expected_town: str | None = None,
) -> Tuple[List[List[Tuple[float, float]]], Tuple[float, float, float, float] | None, List[Dict[str, object]]]:
    """Connect to CARLA, ensure the desired map is loaded, sample waypoints, and optionally cache."""
    # Cache reuse only if map name matches expectation (if provided).
    # Load cache before touching CARLA to avoid crashes in environments without a running server.
    if cache_path and cache_path.exists():
        try:
            cached = pickle.load(cache_path.open("rb"))
            if (
                isinstance(cached, dict)
                and "lines" in cached
                and (expected_town is None or expected_town in str(cached.get("map_name", "")))
            ):
                print(f"[INFO] Using cached map polylines from {cache_path} (map={cached.get('map_name')})")
                records = _extract_map_line_records(cached)
                return cached["lines"], cached.get("bounds"), records
        except Exception:
            pass

    if carla is None:
        raise SystemExit("carla Python module not available; install CARLA egg/wheel or omit --use-carla-map")

    _assert_carla_endpoint_reachable(host, port, timeout_s=2.0)
    client = carla.Client(host, port)
    client.set_timeout(10.0)

    available_maps = client.get_available_maps()
    world = client.get_world()
    cmap = world.get_map()
    print(f"[INFO] CARLA current map: {cmap.name}")
    print(f"[INFO] CARLA available maps: {', '.join(available_maps)}")

    if expected_town and expected_town not in (cmap.name or ""):
        candidates = [m for m in available_maps if expected_town in m]
        if not candidates:
            raise RuntimeError(f"CARLA map '{cmap.name}' does not match expected '{expected_town}', and no available map matches")
        target_map = candidates[0]
        print(f"[INFO] Loading map '{target_map}' to satisfy expected substring '{expected_town}'")
        world = client.load_world(target_map)
        cmap = world.get_map()
        print(f"[INFO] Loaded map: {cmap.name}")

    wps = cmap.generate_waypoints(distance=sample)
    buckets: Dict[Tuple[int, int], List[carla.Waypoint]] = {}
    for wp in wps:
        key = (wp.road_id, wp.lane_id)
        buckets.setdefault(key, []).append(wp)

    lines: List[List[Tuple[float, float]]] = []
    line_records: List[Dict[str, object]] = []
    bounds = [float("inf"), -float("inf"), float("inf"), -float("inf")]  # minx, maxx, miny, maxy
    for (road_id, lane_id), seq in buckets.items():
        seq.sort(key=lambda w: w.s)  # along-lane distance
        line: List[Tuple[float, float]] = []
        for w in seq:
            x, y = float(w.transform.location.x), float(w.transform.location.y)
            line.append((x, y))
            bounds[0] = min(bounds[0], x)
            bounds[1] = max(bounds[1], x)
            bounds[2] = min(bounds[2], y)
            bounds[3] = max(bounds[3], y)
        if len(line) >= 2:
            lines.append(line)
            # OpenDRIVE convention: lane_id > 0 typically runs opposite to increasing road-s.
            dir_sign = -1 if int(lane_id) > 0 else 1
            line_records.append(
                {
                    "points": line,
                    "road_id": int(road_id),
                    "lane_id": int(lane_id),
                    "dir_sign": int(dir_sign),
                }
            )

    btuple = None if bounds[0] == float("inf") else tuple(bounds)  # type: ignore

    if cache_path:
        try:
            pickle.dump(
                {
                    "lines": lines,
                    "line_records": line_records,
                    "bounds": btuple,
                    "map_name": cmap.name,
                },
                cache_path.open("wb"),
            )
            print(f"[INFO] Cached map polylines to {cache_path} (map={cmap.name})")
        except Exception:
            pass

    return lines, btuple, line_records


def main() -> None:
    args = parse_args()
    scenario_dir = Path(args.scenario_dir).expanduser().resolve()
    yaml_dirs = pick_yaml_dirs(scenario_dir, args.subdir)
    out_dir = Path(args.out_dir or (scenario_dir / "carla_log_export")).resolve()
    actors_dir = out_dir / "actors"
    actors_dir.mkdir(parents=True, exist_ok=True)

    # Optional transform overrides from JSON
    if args.coord_json:
        try:
            cfg = json.loads(Path(args.coord_json).read_text(encoding="utf-8"))
            json_tx = float(cfg.get("tx", 0.0))
            json_ty = float(cfg.get("ty", 0.0))
            json_tz = float(cfg.get("tz", 0.0)) if "tz" in cfg else 0.0
            json_theta_deg = (
                float(cfg.get("theta_deg", 0.0))
                if "theta_deg" in cfg
                else float(cfg.get("theta_rad", 0.0)) * 180.0 / math.pi if "theta_rad" in cfg else 0.0
            )
            json_flip = bool(cfg.get("flip_y", False) or cfg.get("y_flip", False))

            # Inverse transform: JSON describes CARLA->PKL; we need PKL->CARLA for XML
            if json_flip:
                args.tx += -json_tx
                args.ty += json_ty
                args.flip_y = True
            else:
                args.tx += -json_tx
                args.ty += -json_ty
            args.tz += -json_tz
            args.yaw_deg += -json_theta_deg

            # Allow XML-only offsets from the same file if present
            args.xml_tx += float(cfg.get("xml_tx", 0.0))
            args.xml_ty += float(cfg.get("xml_ty", 0.0))
        except Exception as exc:
            raise SystemExit(f"Failed to read coord_json {args.coord_json}: {exc}") from exc

    if len(yaml_dirs) > 1:
        print("[INFO] Using multiple YAML subfolders for actor locations:")
        for yd in yaml_dirs:
            print(f"  - {yd}")
        pos_subdirs = [yd for yd in yaml_dirs if not _is_negative_subdir(yd)]
        neg_subdirs = [yd for yd in yaml_dirs if _is_negative_subdir(yd)]
        if pos_subdirs:
            print("[INFO] Ego subfolders (non-negative):")
            for yd in pos_subdirs:
                print(f"  - {yd}")
        if neg_subdirs:
            print("[INFO] Non-ego subfolders (negative):")
            for yd in neg_subdirs:
                print(f"  - {yd}")

    vehicles: Dict[int, List[Waypoint]] = {}
    vehicle_times: Dict[int, List[float]] = {}
    ego_trajs: List[List[Waypoint]] = []
    ego_times_list: List[List[float]] = []
    obj_info: Dict[int, Dict[str, object]] = {}

    for yd in yaml_dirs:
        is_negative_subdir = _is_negative_subdir(yd)
        v_map, v_times, ego_traj, ego_times, v_info = build_trajectories(
            yaml_dir=yd,
            dt=args.dt,
            tx=args.tx,
            ty=args.ty,
            tz=args.tz,
            yaw_deg=args.yaw_deg,
            flip_y=args.flip_y,
        )
        if ego_traj and not is_negative_subdir:
            ego_trajs.append(ego_traj)
            ego_times_list.append(ego_times)
        for vid, meta in v_info.items():
            existing = obj_info.get(vid, {})
            if not existing:
                obj_info[vid] = meta
                continue
            # Fill missing fields without overwriting existing obj_type/model
            if not existing.get("obj_type") and meta.get("obj_type"):
                existing["obj_type"] = meta.get("obj_type")
                if meta.get("model"):
                    existing["model"] = meta.get("model")
            if existing.get("length") is None and meta.get("length") is not None:
                existing["length"] = meta.get("length")
            if existing.get("width") is None and meta.get("width") is not None:
                existing["width"] = meta.get("width")
            obj_info[vid] = existing
        for vid, traj in v_map.items():
            if vid not in vehicles or len(traj) > len(vehicles[vid]):
                vehicles[vid] = traj
                vehicle_times[vid] = v_times.get(vid, [])

    if args.ego_only:
        ignored = len(vehicles)
        vehicles = {}
        vehicle_times = {}
        obj_info = {}
        print(f"[INFO] --ego-only enabled: ignoring {ignored} non-ego actors.")

    # Build actor metadata (used for preprocessing and export)
    actor_meta_by_id: Dict[int, Dict[str, object]] = {}
    skipped_non_vehicles = 0
    for vid, traj in vehicles.items():
        if not traj:
            continue
        info = obj_info.get(vid, {})
        obj_type_val = info.get("obj_type")
        if not obj_type_val:
            print(f"[WARN] Missing obj_type for actor id {vid}; defaulting to npc")
            obj_type_raw = "npc"
        else:
            obj_type_raw = str(obj_type_val)
        if not is_vehicle_type(obj_type_raw):
            skipped_non_vehicles += 1
            continue
        kind, is_ped = _classify_actor_kind(traj, obj_type_raw)
        model = info.get("model") or map_obj_type(obj_type_raw)
        actor_meta_by_id[vid] = {
            "kind": kind,
            "is_pedestrian": is_ped,
            "obj_type": obj_type_raw,
            "model": model,
            "length": info.get("length"),
            "width": info.get("width"),
        }

    if skipped_non_vehicles > 0:
        print(f"[INFO] Skipped {skipped_non_vehicles} non-actor objects (props, static objects, etc.)")

    # Keep a copy for diagnostics/visualization (pre-spawn-preprocess state).
    ego_trajs_pre_align = [
        [
            Waypoint(
                x=float(wp.x),
                y=float(wp.y),
                z=float(wp.z),
                yaw=float(wp.yaw),
                pitch=float(wp.pitch),
                roll=float(wp.roll),
            )
            for wp in traj
        ]
        for traj in ego_trajs
    ]
    vehicles_pre_align: Dict[int, List[Waypoint]] = {
        int(vid): [
            Waypoint(
                x=float(wp.x),
                y=float(wp.y),
                z=float(wp.z),
                yaw=float(wp.yaw),
                pitch=float(wp.pitch),
                roll=float(wp.roll),
            )
            for wp in traj
        ]
        for vid, traj in vehicles.items()
    }

    if args.spawn_preprocess:
        report = _preprocess_spawn_positions(
            vehicles,
            vehicle_times,
            actor_meta_by_id,
            args,
            ego_trajs=ego_trajs,
            ego_times_list=ego_times_list,
        )
        spawn_report = report
        if args.spawn_preprocess_report:
            report_path = Path(args.spawn_preprocess_report)
            if not report_path.is_absolute():
                report_path = out_dir / report_path
            try:
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(f"[INFO] Spawn preprocess report written to {report_path}")
            except Exception as exc:
                print(f"[WARN] Failed to write spawn preprocess report: {exc}")

    # Export buffers (can diverge from raw trajectories when applying timing overrides).
    vehicles_export: Dict[int, List[Waypoint]] = {
        int(vid): [_copy_waypoint(wp) for wp in traj]
        for vid, traj in vehicles.items()
    }
    vehicle_times_export: Dict[int, List[float]] = {
        int(vid): _ensure_times(vehicles_export[int(vid)], vehicle_times.get(int(vid)), args.dt)
        for vid in vehicles_export.keys()
    }

    early_spawn_report: Dict[str, object] = {
        "enabled": bool(args.maximize_safe_early_spawn),
        "applied": False,
        "reason": None,
        "safety_margin": float(args.early_spawn_safety_margin),
        "adjusted_actor_ids": [],
        "adjusted_spawn_times": {},
    }
    if args.maximize_safe_early_spawn:
        if not args.encode_timing:
            early_spawn_report["reason"] = "encode_timing_disabled"
            print(
                "[INFO] Early-spawn optimization skipped because --encode-timing is disabled."
            )
        elif not actor_meta_by_id:
            early_spawn_report["reason"] = "no_actor_metadata"
            print("[INFO] Early-spawn optimization skipped because there are no actor trajectories.")
        else:
            selected_spawn_times, selection_report = _maximize_safe_early_spawn_actors(
                vehicles=vehicles_export,
                vehicle_times=vehicle_times_export,
                actor_meta=actor_meta_by_id,
                dt=float(args.dt),
                safety_margin=float(args.early_spawn_safety_margin),
            )
            (
                vehicles_export,
                vehicle_times_export,
                adjusted_ids,
                applied_spawn_times,
            ) = _apply_early_spawn_time_overrides(
                vehicles=vehicles_export,
                vehicle_times=vehicle_times_export,
                early_spawn_times=selected_spawn_times,
                dt=float(args.dt),
            )
            early_spawn_report.update(selection_report)
            early_spawn_report["applied"] = True
            early_spawn_report["adjusted_actor_ids"] = adjusted_ids
            early_spawn_report["adjusted_spawn_times"] = {
                str(int(vid)): float(t) for vid, t in sorted(applied_spawn_times.items())
            }
            print(
                "[EARLY_SPAWN] candidates={} adjusted={} at_t0={} dynamic_limited={} "
                "static_conflict_adjustments={} avg_advance={:.3f}s".format(
                    int(selection_report.get("candidates", 0)),
                    len(adjusted_ids),
                    int(selection_report.get("spawn_at_t0", 0)),
                    int(selection_report.get("dynamic_limited", 0)),
                    int(selection_report.get("static_conflict_adjustments", 0)),
                    float(selection_report.get("avg_advance_seconds", 0.0)),
                )
            )
    else:
        early_spawn_report["reason"] = "disabled_by_flag"

    if args.early_spawn_report:
        report_path = Path(args.early_spawn_report)
        if not report_path.is_absolute():
            report_path = out_dir / report_path
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(early_spawn_report, indent=2), encoding="utf-8")
            print(f"[INFO] Early-spawn report written to {report_path}")
        except Exception as exc:
            print(f"[WARN] Failed to write early-spawn report: {exc}")

    late_despawn_report: Dict[str, object] = {
        "enabled": bool(args.maximize_safe_late_despawn),
        "applied": False,
        "reason": None,
        "safety_margin": float(args.late_despawn_safety_margin),
        "adjusted_actor_ids": [],
        "hold_until_time": None,
    }
    if args.maximize_safe_late_despawn:
        if not args.encode_timing:
            late_despawn_report["reason"] = "encode_timing_disabled"
            print(
                "[INFO] Late-despawn optimization skipped because --encode-timing is disabled."
            )
        elif not actor_meta_by_id:
            late_despawn_report["reason"] = "no_actor_metadata"
            print("[INFO] Late-despawn optimization skipped because there are no actor trajectories.")
        else:
            horizon_candidates: List[float] = []
            for times in vehicle_times_export.values():
                if times:
                    horizon_candidates.append(float(times[-1]))
            for ego_times in ego_times_list:
                if ego_times:
                    horizon_candidates.append(float(ego_times[-1]))
            hold_until_time = max(horizon_candidates) if horizon_candidates else 0.0
            late_despawn_report["hold_until_time"] = float(hold_until_time)
            if hold_until_time <= 0.0:
                late_despawn_report["reason"] = "non_positive_horizon"
            else:
                selected_ids, selection_report = _maximize_safe_late_despawn_actors(
                    vehicles=vehicles_export,
                    vehicle_times=vehicle_times_export,
                    actor_meta=actor_meta_by_id,
                    dt=float(args.dt),
                    safety_margin=float(args.late_despawn_safety_margin),
                    hold_until_time=float(hold_until_time),
                )
                vehicles_export, vehicle_times_export, adjusted_ids = _apply_late_despawn_time_overrides(
                    vehicles=vehicles_export,
                    vehicle_times=vehicle_times_export,
                    selected_late_hold_ids=selected_ids,
                    dt=float(args.dt),
                    hold_until_time=float(hold_until_time),
                )
                late_despawn_report.update(selection_report)
                late_despawn_report["applied"] = True
                late_despawn_report["adjusted_actor_ids"] = adjusted_ids
                print(
                    "[LATE_DESPAWN] candidates={} safe={} selected={} adjusted={} "
                    "pair_conflicts={} timeout_components={} horizon={:.3f}s".format(
                        int(selection_report.get("candidates", 0)),
                        int(selection_report.get("individually_safe", 0)),
                        int(selection_report.get("selected", 0)),
                        len(adjusted_ids),
                        int(selection_report.get("pair_conflicts", 0)),
                        int(selection_report.get("timed_out_components", 0)),
                        float(hold_until_time),
                    )
                )
    else:
        late_despawn_report["reason"] = "disabled_by_flag"

    if args.late_despawn_report:
        report_path = Path(args.late_despawn_report)
        if not report_path.is_absolute():
            report_path = out_dir / report_path
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(late_despawn_report, indent=2), encoding="utf-8")
            print(f"[INFO] Late-despawn report written to {report_path}")
        except Exception as exc:
            print(f"[WARN] Failed to write late-despawn report: {exc}")

    # Write ego route (optional)
    ego_entries: List[dict] = []
    if not args.no_ego and ego_trajs:
        # Remove legacy ego_route.xml to avoid double-counting egos
        legacy_ego = out_dir / "ego_route.xml"
        if legacy_ego.exists():
            try:
                legacy_ego.unlink()
                print(f"[INFO] Removed legacy ego file {legacy_ego}")
            except Exception:
                pass
        for ego_idx, ego_traj in enumerate(ego_trajs):
            ego_times = ego_times_list[ego_idx] if ego_idx < len(ego_times_list) else []
            # Follow CustomRoutes naming: {town}_custom_ego_vehicle_{i}.xml
            ego_xml = out_dir / f"{args.town.lower()}_custom_ego_vehicle_{ego_idx}.xml"
            write_route_xml(
                ego_xml,
                route_id=args.route_id,
                role="ego",
                town=args.town,
                waypoints=ego_traj,
                times=ego_times if args.encode_timing else None,
                snap_to_road=False,
                xml_tx=args.xml_tx,
                xml_ty=args.xml_ty,
            )
            ego_entries.append({
                "file": ego_xml.name,
                "route_id": str(args.route_id),
                "town": args.town,
                "name": ego_xml.stem,
                "kind": "ego",
                "model": args.ego_model,
            })

    # Build actor entries after we know obj_type/model
    # Group by kind (npc, static, etc.) for manifest
    actors_by_kind: Dict[str, List[dict]] = {}
    actor_kind_by_id: Dict[int, str] = {}
    actor_xml_by_id: Dict[int, Path] = {}

    if args.ego_only and actors_dir.exists():
        removed = 0
        for stale in actors_dir.rglob("*.xml"):
            try:
                stale.unlink()
                removed += 1
            except Exception:
                pass
        if removed > 0:
            print(f"[INFO] Removed {removed} stale actor XML files because --ego-only is enabled.")

    for vid, traj in vehicles_export.items():
        if not traj:
            continue
        meta = actor_meta_by_id.get(vid)
        if meta is None:
            continue
        obj_type_raw = str(meta.get("obj_type") or "npc")
        kind = str(meta.get("kind"))
        model = meta.get("model") or map_obj_type(obj_type_raw)
        length = meta.get("length")
        width = meta.get("width")
        
        # Use obj_type directly for actor type in filename
        # Clean it up to make it suitable for filenames
        actor_type = obj_type_raw.replace(" ", "_").replace("-", "_").title()
        if not actor_type or actor_type.lower() == "npc":
            actor_type = "Vehicle"
        
        # Follow CustomRoutes naming: {town}_custom_{ActorType}_{id}_{kind}.xml
        name = f"{args.town.lower()}_custom_{actor_type}_{vid}_{kind}"
        
        # Create subdirectory for actor kind
        kind_dir = actors_dir / kind
        kind_dir.mkdir(parents=True, exist_ok=True)
        actor_xml = kind_dir / f"{name}.xml"

        write_route_xml(
            actor_xml,
            route_id=args.route_id,
            role=kind,
            town=args.town,
            waypoints=traj,
            times=vehicle_times_export.get(vid) if args.encode_timing else None,
            snap_to_road=args.snap_to_road is True,
            xml_tx=args.xml_tx,
            xml_ty=args.xml_ty,
        )
        actor_xml_by_id[vid] = actor_xml
        speed = 0.0
        if len(traj) >= 2:
            dist = 0.0
            for a, b in zip(traj, traj[1:]):
                dist += euclid3((a.x, a.y, a.z), (b.x, b.y, b.z))
            if args.encode_timing:
                times = vehicle_times_export.get(vid)
                if times and len(times) == len(traj):
                    total_time = times[-1] - times[0]
                    if total_time > 1e-6:
                        speed = dist / total_time
                    else:
                        speed = dist / max(args.dt * (len(traj) - 1), 1e-6)
                else:
                    speed = dist / max(args.dt * (len(traj) - 1), 1e-6)
            else:
                speed = dist / max(args.dt * (len(traj) - 1), 1e-6)
        
        entry = {
            "file": str(actor_xml.relative_to(out_dir)),
            "route_id": str(args.route_id),
            "town": args.town,
            "name": name,
            "kind": kind,
            "model": model,
        }
        
        # Add optional fields
        if speed > 0:
            entry["speed"] = speed
        if length is not None:
            entry["length"] = str(length) if isinstance(length, (int, float)) else length
        if width is not None:
            entry["width"] = str(width) if isinstance(width, (int, float)) else width
        
        if kind not in actors_by_kind:
            actors_by_kind[kind] = []
        actors_by_kind[kind].append(entry)
        actor_kind_by_id[vid] = kind

    save_manifest(out_dir / "actors_manifest.json", actors_by_kind, ego_entries)

    # Optional visualization
    if (
        args.gif
        or args.paths_png
        or args.spawn_viz
        or args.actor_yaw_viz_ids
        or args.actor_raw_yaml_viz_ids
        or args.ego_alignment_viz
        or args.ego_alignment_bev_viz
        or args.actor_alignment_bev_viz
        or args.spawn_preprocess_fail_viz
    ):
        if plt is None or (args.gif and imageio is None):
            raise SystemExit("matplotlib (and imageio for GIF) are required for visualization")
        map_lines: List[List[Tuple[float, float]]] = []
        map_line_records: List[Dict[str, object]] = []
        map_bounds = None
        # Priority: explicit map pickle -> CARLA live map (with cache) -> none
        if args.map_pkl:
            try:
                map_obj = pickle.load(Path(args.map_pkl).expanduser().open("rb"))
                map_lines = _extract_map_lines(map_obj, out=[])
                map_line_records = _extract_map_line_records(map_obj)
                if isinstance(map_obj, dict):
                    map_bounds = map_obj.get("bounds")
                print(f"[INFO] Loaded {len(map_lines)} polylines from {args.map_pkl}")
            except Exception as exc:
                print(f"[WARN] Failed to load map pickle {args.map_pkl}: {exc}")
        elif args.use_carla_map:
            cache_path = Path(args.carla_cache or (out_dir / "carla_map_cache.pkl"))
            try:
                sample = float(args.carla_sample)
                if args.spawn_preprocess_fail_viz:
                    sample = min(sample, float(args.spawn_preprocess_fail_viz_sample))
                map_lines, map_bounds, map_line_records = fetch_carla_map_lines(
                    host=args.carla_host,
                    port=args.carla_port,
                    sample=sample,
                    cache_path=cache_path,
                    expected_town=args.expected_town,
                )
                if map_lines:
                    print(f"[INFO] Loaded {len(map_lines)} map polylines from CARLA ({args.carla_host}:{args.carla_port})")
            except Exception as exc:
                print(f"[WARN] Failed to fetch map from CARLA: {exc}")
                map_bounds = None
        else:
            map_bounds = None

        map_image = None
        map_image_bounds = None
        if args.map_image:
            try:
                map_image = plt.imread(args.map_image)
                if args.map_image_bounds:
                    map_image_bounds = tuple(float(v) for v in args.map_image_bounds)  # type: ignore
                elif map_bounds:
                    map_image_bounds = map_bounds
            except Exception as exc:
                print(f"[WARN] Failed to load map image {args.map_image}: {exc}")
                map_image = None
                map_image_bounds = None

        if args.gif:
            frames_dir = out_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            max_len = max((len(t) for t in vehicles.values()), default=0)
            for et in ego_trajs:
                max_len = max(max_len, len(et))
            axes_limits = None
            # Precompute global limits for stable camera
            xs: List[float] = []
            ys: List[float] = []
            for traj in vehicles.values():
                xs.extend([wp.x for wp in traj])
                ys.extend([wp.y for wp in traj])
            for et in ego_trajs:
                for wp in et:
                    xs.append(wp.x)
                    ys.append(wp.y)
            for line in map_lines:
                for x, y in line:
                    xs.append(x)
                    ys.append(y)
            if xs and ys:
                pad = max(0.0, float(args.axis_pad))
                axes_limits = (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)

            for i in range(max_len):
                plot_frame(
                    i,
                    vehicles,
                    ego_trajs,
                    frames_dir / f"frame_{i:06d}.png",
                    axes_limits,
                    map_lines=map_lines,
                    invert_plot_y=args.invert_plot_y,
                )
            gif_path = Path(args.gif_path or (out_dir / "replay.gif"))
            write_gif(frames_dir, gif_path)
            print(f"[OK] GIF written to {gif_path}")

        if args.paths_png:
            png_path = Path(args.paths_png).expanduser()
            write_paths_png(
                actors_by_id=vehicles,
                ego_trajs=ego_trajs,
                map_lines=map_lines,
                out_path=png_path,
                axis_pad=float(args.axis_pad),
                invert_plot_y=args.invert_plot_y,
            )
            print(f"[OK] Paths PNG written to {png_path}")

        if args.actor_yaw_viz_ids:
            actor_ids = _parse_id_list(args.actor_yaw_viz_ids)
            out_dir_yaw = Path(args.actor_yaw_viz_dir or (out_dir / "actor_yaw_viz")).expanduser()
            out_dir_yaw.mkdir(parents=True, exist_ok=True)
            for vid in actor_ids:
                gt_traj = vehicles.get(vid) or []
                xml_path = actor_xml_by_id.get(vid)
                if xml_path is None:
                    # best-effort fallback search
                    matches = list(actors_dir.rglob(f"*_{vid}_*.xml"))
                    xml_path = matches[0] if matches else None
                if not gt_traj:
                    print(f"[WARN] No GT trajectory found for actor id {vid}")
                    continue
                if xml_path is None or not xml_path.exists():
                    print(f"[WARN] No XML found for actor id {vid}")
                    continue
                xml_traj = parse_route_xml(xml_path)
                if not xml_traj:
                    print(f"[WARN] XML had no waypoints for actor id {vid}: {xml_path}")
                    continue
                out_path = out_dir_yaw / f"actor_{vid}_yaw_viz.png"
                write_actor_yaw_viz(
                    actor_id=vid,
                    gt_traj=gt_traj,
                    xml_traj=xml_traj,
                    map_lines=map_lines,
                    out_path=out_path,
                    arrow_step=max(1, int(args.actor_yaw_viz_step)),
                    arrow_len=float(args.actor_yaw_viz_arrow_len),
                    pad=float(args.actor_yaw_viz_pad),
                    invert_plot_y=args.invert_plot_y,
                )
                print(f"[OK] Actor yaw viz written to {out_path}")

        if args.actor_raw_yaml_viz_ids:
            actor_ids = _parse_id_list(args.actor_raw_yaml_viz_ids)
            out_dir_raw = Path(args.actor_raw_yaml_viz_dir or (out_dir / "actor_raw_yaml_viz")).expanduser()
            out_dir_raw.mkdir(parents=True, exist_ok=True)

            # Collect per-subdir points directly from YAML (with transform + XML offsets applied)
            points_by_actor: Dict[int, Dict[str, List[Tuple[float, float, float]]]] = {vid: {} for vid in actor_ids}
            for yd in yaml_dirs:
                sub_name = yd.name
                yaml_paths = list_yaml_timesteps(yd)
                for idx, path in enumerate(yaml_paths):
                    try:
                        frame_idx = int(path.stem)
                    except Exception:
                        frame_idx = idx
                    t = float(frame_idx) * float(args.dt)
                    data = load_yaml(path)
                    vehs = data.get("vehicles", {}) or {}
                    for vid in actor_ids:
                        payload = vehs.get(vid) if vid in vehs else vehs.get(str(vid))
                        if not payload:
                            continue
                        loc = payload.get("location") or [0, 0, 0]
                        x0 = float(loc[0]) if len(loc) > 0 else 0.0
                        y0 = float(loc[1]) if len(loc) > 1 else 0.0
                        x, y = apply_se2((x0, y0), args.yaw_deg, args.tx, args.ty, flip_y=args.flip_y)
                        x += float(args.xml_tx)
                        y += float(args.xml_ty)
                        points_by_actor.setdefault(vid, {}).setdefault(sub_name, []).append((x, y, t))

            for vid in actor_ids:
                points = points_by_actor.get(vid, {})
                if not points:
                    print(f"[WARN] No YAML points found for actor id {vid}")
                    continue
                out_path = out_dir_raw / f"actor_{vid}_raw_yaml_points.png"
                write_actor_raw_yaml_viz(
                    actor_id=vid,
                    points_by_subdir=points,
                    map_lines=map_lines,
                    out_path=out_path,
                    pad=float(args.actor_raw_yaml_viz_pad),
                    invert_plot_y=args.invert_plot_y,
                )
                print(f"[OK] Actor raw YAML viz written to {out_path}")

        if args.ego_alignment_viz:
            out_dir_ego = Path(args.ego_alignment_viz_dir or (out_dir / "ego_alignment_viz")).expanduser()
            out_dir_ego.mkdir(parents=True, exist_ok=True)
            n_pre = len(ego_trajs_pre_align)
            n_post = len(ego_trajs)
            if n_pre == 0 or n_post == 0:
                print("[WARN] Ego alignment viz requested but no ego trajectories are available.")
            else:
                if n_pre != n_post:
                    print(f"[WARN] Ego count mismatch for viz: pre={n_pre} post={n_post}; using min count.")
                for ego_idx in range(min(n_pre, n_post)):
                    out_path = out_dir_ego / f"ego_{ego_idx}_pre_vs_post_alignment.png"
                    write_ego_alignment_viz(
                        ego_idx=ego_idx,
                        pre_align_traj=ego_trajs_pre_align[ego_idx],
                        post_align_traj=ego_trajs[ego_idx],
                        map_lines=map_lines,
                        map_line_records=map_line_records,
                        out_path=out_path,
                        xml_tx=float(args.xml_tx),
                        xml_ty=float(args.xml_ty),
                        pad=float(args.ego_alignment_viz_pad),
                        invert_plot_y=bool(args.invert_plot_y),
                    )
                    print(f"[OK] Ego alignment viz written to {out_path}")

        if args.ego_alignment_bev_viz:
            out_dir_ego_bev = Path(args.ego_alignment_bev_viz_dir or (out_dir / "ego_alignment_bev_viz")).expanduser()
            out_dir_ego_bev.mkdir(parents=True, exist_ok=True)
            n_pre = len(ego_trajs_pre_align)
            n_post = len(ego_trajs)
            if n_pre == 0 or n_post == 0:
                print("[WARN] Ego BEV alignment viz requested but no ego trajectories are available.")
            else:
                if map_image is None and not bool(args.ego_alignment_bev_capture_from_carla):
                    print("[WARN] Ego BEV viz requested without --map-image; using vector map fallback instead of raster CARLA image.")
                if n_pre != n_post:
                    print(f"[WARN] Ego count mismatch for BEV viz: pre={n_pre} post={n_post}; using min count.")
                for ego_idx in range(min(n_pre, n_post)):
                    out_path = out_dir_ego_bev / f"ego_{ego_idx}_alignment_bev_nodes.png"
                    pre_xy = [
                        (wp.x + float(args.xml_tx), wp.y + float(args.xml_ty))
                        for wp in ego_trajs_pre_align[ego_idx]
                    ]
                    post_xy = [
                        (wp.x + float(args.xml_tx), wp.y + float(args.xml_ty))
                        for wp in ego_trajs[ego_idx]
                    ]
                    ego_bounds = _bounds_from_points(pre_xy + post_xy)
                    captured_bev = None
                    if (
                        bool(args.ego_alignment_bev_capture_from_carla)
                        and ego_bounds is not None
                        and carla is not None
                        and np is not None
                    ):
                        try:
                            captured_bev = _capture_carla_topdown_bev(
                                host=str(args.carla_host),
                                port=int(args.carla_port),
                                bounds=ego_bounds,
                                image_w=int(args.ego_alignment_bev_capture_width),
                                image_h=int(args.ego_alignment_bev_capture_height),
                                fov_deg=float(args.ego_alignment_bev_capture_fov),
                                margin_scale=float(args.ego_alignment_bev_capture_margin),
                                expected_town=str(args.expected_town),
                            )
                            if captured_bev is None:
                                print(f"[WARN] Ego {ego_idx} CARLA BEV capture returned no image; using fallback background.")
                        except Exception as exc:
                            print(f"[WARN] Ego {ego_idx} CARLA BEV capture failed: {exc}; using fallback background.")
                            captured_bev = None
                    write_ego_alignment_bev_viz(
                        ego_idx=ego_idx,
                        pre_align_traj=ego_trajs_pre_align[ego_idx],
                        post_align_traj=ego_trajs[ego_idx],
                        map_lines=map_lines,
                        map_line_records=map_line_records,
                        map_image=map_image,
                        map_image_bounds=map_image_bounds,
                        captured_bev=captured_bev,
                        out_path=out_path,
                        xml_tx=float(args.xml_tx),
                        xml_ty=float(args.xml_ty),
                        pad=float(args.ego_alignment_viz_pad),
                        node_step=int(args.ego_alignment_bev_node_step),
                        match_radius=float(args.ego_alignment_bev_match_radius),
                        invert_plot_y=bool(args.invert_plot_y),
                    )
                    print(f"[OK] Ego BEV alignment viz written to {out_path}")

        if args.actor_alignment_bev_viz:
            actor_ids_all = sorted(set(int(v) for v in vehicles_pre_align.keys()) | set(int(v) for v in vehicles.keys()))
            if not actor_ids_all:
                print("[WARN] Actor BEV alignment viz skipped: no actor trajectories available.")
            else:
                out_dir_actor_bev = Path(
                    args.actor_alignment_bev_viz_dir or (out_dir / "actor_alignment_bev_viz")
                ).expanduser()
                out_dir_actor_bev.mkdir(parents=True, exist_ok=True)
                out_dir_actor_indiv = out_dir_actor_bev / "individual"
                out_dir_actor_indiv.mkdir(parents=True, exist_ok=True)

                if map_image is None and not bool(args.ego_alignment_bev_capture_from_carla):
                    print("[WARN] Actor BEV viz requested without CARLA capture and without --map-image; using vector map fallback.")

                walker_ids = [
                    vid
                    for vid in actor_ids_all
                    if str((actor_meta_by_id.get(vid) or {}).get("kind", "")) in ("walker", "walker_static")
                ]
                npc_ids = [
                    vid
                    for vid in actor_ids_all
                    if str((actor_meta_by_id.get(vid) or {}).get("kind", "")) == "npc"
                ]
                subset_specs = [
                    ("all_actors", "Actors: all", actor_ids_all),
                    ("all_walkers", "Actors: all walkers", walker_ids),
                    ("all_npc", "Actors: all NPC vehicles", npc_ids),
                ]

                for key, title, ids in subset_specs:
                    if not ids:
                        continue
                    pre_subset = {vid: vehicles_pre_align.get(vid, []) for vid in ids}
                    post_subset = {vid: vehicles.get(vid, []) for vid in ids}
                    pre_pts = [
                        (wp.x + float(args.xml_tx), wp.y + float(args.xml_ty))
                        for vid in ids
                        for wp in pre_subset.get(vid, [])
                    ]
                    post_pts = [
                        (wp.x + float(args.xml_tx), wp.y + float(args.xml_ty))
                        for vid in ids
                        for wp in post_subset.get(vid, [])
                    ]
                    subset_bounds = _bounds_from_points(pre_pts + post_pts)
                    captured_bev = None
                    if (
                        bool(args.ego_alignment_bev_capture_from_carla)
                        and subset_bounds is not None
                        and carla is not None
                        and np is not None
                    ):
                        try:
                            captured_bev = _capture_carla_topdown_bev(
                                host=str(args.carla_host),
                                port=int(args.carla_port),
                                bounds=subset_bounds,
                                image_w=int(args.ego_alignment_bev_capture_width),
                                image_h=int(args.ego_alignment_bev_capture_height),
                                fov_deg=float(args.ego_alignment_bev_capture_fov),
                                margin_scale=float(args.ego_alignment_bev_capture_margin),
                                expected_town=str(args.expected_town),
                            )
                        except Exception as exc:
                            print(f"[WARN] Actor subset '{key}' CARLA BEV capture failed: {exc}; using fallback background.")
                            captured_bev = None
                    out_path = out_dir_actor_bev / f"{key}_alignment_bev_nodes.png"
                    write_actor_alignment_bev_viz(
                        title=title,
                        pre_trajs=pre_subset,
                        post_trajs=post_subset,
                        map_lines=map_lines,
                        map_line_records=map_line_records,
                        map_image=map_image,
                        map_image_bounds=map_image_bounds,
                        captured_bev=captured_bev,
                        out_path=out_path,
                        xml_tx=float(args.xml_tx),
                        xml_ty=float(args.xml_ty),
                        pad=float(args.ego_alignment_viz_pad),
                        node_step=int(args.ego_alignment_bev_node_step),
                        match_radius=float(args.ego_alignment_bev_match_radius),
                        invert_plot_y=bool(args.invert_plot_y),
                    )
                    print(f"[OK] Actor BEV alignment viz written to {out_path}")

                for idx, vid in enumerate(actor_ids_all):
                    pre_single = {vid: vehicles_pre_align.get(vid, [])}
                    post_single = {vid: vehicles.get(vid, [])}
                    pre_pts = [(wp.x + float(args.xml_tx), wp.y + float(args.xml_ty)) for wp in pre_single.get(vid, [])]
                    post_pts = [(wp.x + float(args.xml_tx), wp.y + float(args.xml_ty)) for wp in post_single.get(vid, [])]
                    single_bounds = _bounds_from_points(pre_pts + post_pts)
                    captured_bev = None
                    if (
                        bool(args.ego_alignment_bev_capture_from_carla)
                        and single_bounds is not None
                        and carla is not None
                        and np is not None
                    ):
                        try:
                            captured_bev = _capture_carla_topdown_bev(
                                host=str(args.carla_host),
                                port=int(args.carla_port),
                                bounds=single_bounds,
                                image_w=int(args.ego_alignment_bev_capture_width),
                                image_h=int(args.ego_alignment_bev_capture_height),
                                fov_deg=float(args.ego_alignment_bev_capture_fov),
                                margin_scale=float(args.ego_alignment_bev_capture_margin),
                                expected_town=str(args.expected_town),
                            )
                        except Exception:
                            captured_bev = None

                    kind = str((actor_meta_by_id.get(vid) or {}).get("kind", "actor"))
                    out_path = out_dir_actor_indiv / f"actor_{vid}_{kind}_alignment_bev_nodes.png"
                    write_actor_alignment_bev_viz(
                        title=f"Actor {vid} ({kind}) alignment",
                        pre_trajs=pre_single,
                        post_trajs=post_single,
                        map_lines=map_lines,
                        map_line_records=map_line_records,
                        map_image=map_image,
                        map_image_bounds=map_image_bounds,
                        captured_bev=captured_bev,
                        out_path=out_path,
                        xml_tx=float(args.xml_tx),
                        xml_ty=float(args.xml_ty),
                        pad=float(args.ego_alignment_viz_pad),
                        node_step=int(args.ego_alignment_bev_node_step),
                        match_radius=float(args.ego_alignment_bev_match_radius),
                        invert_plot_y=bool(args.invert_plot_y),
                    )
                    if idx % 10 == 0 or idx == len(actor_ids_all) - 1:
                        print(f"[OK] Actor individual BEV viz progress: {idx + 1}/{len(actor_ids_all)}")

        if args.spawn_viz:
            spawn_viz_path = Path(args.spawn_viz_path or (out_dir / "spawn_alignment_viz.png")).expanduser()

            aligned_points: Dict[int, Tuple[float, float]] = {}
            spawn_points: Dict[int, Tuple[float, float]] = {}
            pre_align_points: Dict[int, Tuple[float, float]] = {}
            for vid, traj in vehicles.items():
                if not traj:
                    continue
                wp0 = traj[0]
                aligned_points[vid] = (wp0.x, wp0.y)
                spawn_points[vid] = (wp0.x + args.xml_tx, wp0.y + args.xml_ty)
                pre_align_points[vid] = invert_se2((wp0.x, wp0.y), args.yaw_deg, args.tx, args.ty, flip_y=args.flip_y)

            ego_aligned: List[Tuple[float, float]] = []
            ego_spawn: List[Tuple[float, float]] = []
            ego_pre_align: List[Tuple[float, float]] = []
            for ego_traj in ego_trajs:
                if not ego_traj:
                    continue
                wp0 = ego_traj[0]
                ego_aligned.append((wp0.x, wp0.y))
                ego_spawn.append((wp0.x + args.xml_tx, wp0.y + args.xml_ty))
                ego_pre_align.append(invert_se2((wp0.x, wp0.y), args.yaw_deg, args.tx, args.ty, flip_y=args.flip_y))

            offset_pair = None
            offset_label = None
            offset_ref = _pick_offset_reference(aligned_points, pre_align_points, ego_aligned, ego_pre_align)
            if offset_ref:
                offset_pair = (offset_ref[0], offset_ref[1])
                offset_label = f"pre-align ref ({offset_ref[2]})"

            xodr_points: List[Tuple[float, float]] = []
            xodr_path = Path(args.xodr).expanduser() if args.xodr else None
            if xodr_path and xodr_path.exists():
                try:
                    xodr_points = load_xodr_points(xodr_path, args.xodr_step)
                    print(f"[INFO] Loaded {len(xodr_points)} XODR points from {xodr_path}")
                except Exception as exc:
                    print(f"[WARN] Failed to load XODR {xodr_path}: {exc}")
                    xodr_points = []
            else:
                # Best-effort default XODR next to repo root (if present)
                default_xodr = Path(__file__).resolve().parents[2] / "ucla_v2.xodr"
                if default_xodr.exists():
                    try:
                        xodr_points = load_xodr_points(default_xodr, args.xodr_step)
                        print(f"[INFO] Loaded {len(xodr_points)} XODR points from {default_xodr}")
                    except Exception as exc:
                        print(f"[WARN] Failed to load XODR {default_xodr}: {exc}")

            map_points: List[Tuple[float, float]] = []
            for line in map_lines:
                map_points.extend(line)

            kind_by_id = dict(actor_kind_by_id)
            for vid in aligned_points.keys():
                kind_by_id.setdefault(vid, "npc")

            bounds = _merge_bounds(
                [
                    map_bounds,
                    _bounds_from_points(map_points),
                    _bounds_from_points(xodr_points),
                    _bounds_from_points(
                        list(aligned_points.values())
                        + list(spawn_points.values())
                        + list(pre_align_points.values())
                        + ego_aligned
                        + ego_spawn
                        + ego_pre_align
                    ),
                    map_image_bounds,
                ]
            )
            if bounds:
                pad = max(0.0, float(args.axis_pad))
                minx, maxx, miny, maxy = bounds
                bounds = (minx - pad, maxx + pad, miny - pad, maxy + pad)

            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            ax_map, ax_xodr = axes

            # Left: CARLA map layer (image if provided, else polylines)
            if map_image is not None:
                if map_image_bounds:
                    minx, maxx, miny, maxy = map_image_bounds
                elif bounds:
                    minx, maxx, miny, maxy = bounds
                else:
                    minx, maxx, miny, maxy = 0.0, 1.0, 0.0, 1.0
                ax_map.imshow(map_image, extent=(minx, maxx, miny, maxy), origin="lower", alpha=0.8, zorder=0)
            elif map_lines:
                _plot_background_lines(ax_map, map_lines, color="#9e9e9e", lw=0.8, alpha=0.6)
            _plot_spawn_alignment(
                ax_map,
                aligned_points,
                spawn_points,
                kind_by_id,
                ego_aligned,
                ego_spawn,
                title="CARLA Map Layer",
                offset_pair=offset_pair,
                offset_label=offset_label,
            )

            # Right: XODR layer
            if xodr_points:
                ax_xodr.scatter(
                    [p[0] for p in xodr_points],
                    [p[1] for p in xodr_points],
                    s=1,
                    c="#1f77b4",
                    alpha=0.5,
                    label="XODR geometry",
                    zorder=1,
                )
            _plot_spawn_alignment(
                ax_xodr,
                aligned_points,
                spawn_points,
                kind_by_id,
                ego_aligned,
                ego_spawn,
                title="XODR Layer",
                offset_pair=offset_pair,
                offset_label=offset_label,
            )

            for ax in axes:
                if bounds:
                    minx, maxx, miny, maxy = bounds
                    ax.set_xlim(minx, maxx)
                    ax.set_ylim(miny, maxy)
                if args.invert_plot_y:
                    ax.invert_yaxis()

            # Global legend and info
            handles, labels = ax_map.get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper right", frameon=True)

            info_lines = [
                f"Actors: {len(aligned_points)} (npc={sum(1 for k in kind_by_id.values() if k == 'npc')}, "
                f"static={sum(1 for k in kind_by_id.values() if k == 'static')}, "
                f"walker={sum(1 for k in kind_by_id.values() if k in ('walker', 'walker_static'))})",
                f"Egos: {len(ego_aligned)}",
                f"Alignment tx/ty/yaw: {args.tx:.2f}, {args.ty:.2f}, {args.yaw_deg:.2f}",
                f"XML offset xml_tx/xml_ty: {args.xml_tx:.2f}, {args.xml_ty:.2f}",
                f"flip_y: {args.flip_y}",
            ]
            fig.text(0.01, 0.01, "\n".join(info_lines), fontsize=9, ha="left", va="bottom")
            fig.suptitle("Spawn vs Aligned Positions (CARLA vs XODR)", fontsize=14)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(spawn_viz_path, dpi=180)
            plt.close(fig)
            print(f"[OK] Spawn alignment visualization written to {spawn_viz_path}")

        if args.spawn_preprocess_fail_viz:
            if spawn_report is None:
                print("[WARN] Spawn preprocess visualization requested but no report available.")
            else:
                fail_dir = args.spawn_preprocess_fail_viz_dir
                if not fail_dir:
                    fail_dir = str(out_dir / "spawn_preprocess_fail_viz")
                out_path = Path(fail_dir).expanduser()
                _plot_failed_spawn_visualizations(
                    report=spawn_report,
                    map_lines=map_lines,
                    out_dir=out_path,
                    window_m=float(args.spawn_preprocess_fail_viz_window),
                    dpi=int(args.spawn_preprocess_fail_viz_dpi),
                )
                print(f"[OK] Failed spawn visualization written to {out_path}")

    # Optional: run custom eval with generated routes
    if args.run_custom_eval:
        repo_root = Path(__file__).resolve().parents[2]
        python_bin = sys.executable
        cmd = [
            python_bin,
            str(repo_root / "tools" / "run_custom_eval.py"),
            "--routes-dir",
            str(out_dir),
            "--port",
            str(args.eval_port),
            "--overwrite",
        ]
        if args.eval_planner:
            cmd.extend(["--planner", args.eval_planner])
        print("[INFO] Running:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[WARN] run_custom_eval failed with exit code {exc.returncode}")

    print(f"[OK] Export complete -> {out_dir}")
    print("Files:")
    if ego_entries:
        for entry in ego_entries:
            print(f"  - {entry['file']}")
    print(f"  - actors_manifest.json")
    total_actors = sum(len(entries) for entries in actors_by_kind.values())
    if args.ego_only:
        print("  - actors/*/*.xml (skipped by --ego-only)")
    else:
        print(f"  - actors/*/*.xml ({total_actors} actors across {len(actors_by_kind)} categories)")
        for kind, entries in sorted(actors_by_kind.items()):
            print(f"    - {kind}: {len(entries)} actors")
    if args.gif:
        print(f"  - replay.gif")


if __name__ == "__main__":
    main()
