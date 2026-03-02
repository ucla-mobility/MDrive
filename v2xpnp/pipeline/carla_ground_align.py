#!/usr/bin/env python3
"""CARLA ground-alignment postprocessor for exported route XML files.

Connects to a running CARLA instance and uses hybrid raycasting to compute
optimal z, pitch, and roll for every waypoint of every vehicle actor so that
all four wheels sit on the road surface.

This is a pure postprocessing step that does NOT alter trajectories (x, y, yaw).

Usage (standalone):
    python -m v2xpnp.pipeline.carla_ground_align /path/to/carla_routes_dir

Usage (from pipeline):
    Called automatically when --carla-ground-align is passed to the pipeline
    entrypoint after XML route export completes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# CARLA import (deferred so module can be imported for type-checking without
# the CARLA egg on PYTHONPATH)
# ---------------------------------------------------------------------------
_carla = None  # type: ignore


def _ensure_carla():
    global _carla
    if _carla is not None:
        return _carla
    try:
        import carla  # type: ignore
        _carla = carla
        return _carla
    except ImportError as exc:
        raise RuntimeError(
            "Could not import CARLA. Ensure the CARLA PythonAPI egg is on PYTHONPATH."
        ) from exc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAYCAST_Z_ABOVE_M = 60.0
RAYCAST_Z_BELOW_M = 60.0
# For hybrid raycasting: probe a small grid around each wheel position.
RAYCAST_GRID_OFFSETS_M = (-0.3, 0.0, 0.3)
RAYCAST_HIGH_PERCENTILE = 0.75
# Default vehicle half-dimensions (used when blueprint lookup fails).
DEFAULT_HALF_LEN = 2.2   # ~4.4m wheelbase
DEFAULT_HALF_WID = 0.9   # ~1.8m track width
# Pedestrians/walkers/cyclists use only z, not pitch/roll.
WALKER_ROLES = frozenset({"walker", "pedestrian", "bicycle", "cyclist", "bike"})
# Maximum allowed pitch/roll (degrees) to reject outliers.
MAX_PITCH_DEG = 25.0
MAX_ROLL_DEG = 25.0
# Small clearance above ground to avoid clipping.
GROUND_CLEARANCE_M = 0.02
# Temporal smoothing window (number of frames each side) for pitch/roll/z.
SMOOTH_WINDOW = 2
# Non-zero threshold for pitch/roll diagnostics.
TILT_EPS_DEG = 1e-4


# ---------------------------------------------------------------------------
# Raycast helpers  (adapted from crosswalk_experiment.py)
# ---------------------------------------------------------------------------

def _ray_hit_road_like(hit) -> bool:
    """Return True if the raycast hit is a road-like surface."""
    carla = _ensure_carla()
    label = getattr(hit, "label", None)
    if label is None:
        return False
    if not hasattr(carla, "CityObjectLabel"):
        return False
    for name in ("Roads", "RoadLines", "Sidewalks", "Ground", "Terrain"):
        if hasattr(carla.CityObjectLabel, name):
            if label == getattr(carla.CityObjectLabel, name):
                return True
    return False


def _raycast_z(world, x: float, y: float, ref_z: float) -> Optional[float]:
    """Cast a vertical ray and return the best road-surface z."""
    carla = _ensure_carla()
    start = carla.Location(x=x, y=y, z=ref_z + RAYCAST_Z_ABOVE_M)
    end = carla.Location(x=x, y=y, z=ref_z - RAYCAST_Z_BELOW_M)
    try:
        hits = world.cast_ray(start, end)
    except Exception:
        hits = []
    if not hits:
        return None

    preferred: list[float] = []
    fallback: list[float] = []
    for hit in hits:
        hit_loc = getattr(hit, "location", None)
        if hit_loc is None or not hasattr(hit_loc, "z"):
            continue
        z = float(hit_loc.z)
        fallback.append(z)
        if _ray_hit_road_like(hit):
            preferred.append(z)

    if preferred:
        return max(preferred)
    if fallback:
        # Use waypoint as reference to pick the closest fallback hit.
        wp_z = _waypoint_z(world, x, y, ref_z)
        if wp_z is not None:
            return min(fallback, key=lambda z: abs(z - wp_z))
        ordered = sorted(fallback)
        return ordered[len(ordered) // 2]
    return None


def _waypoint_z(world, x: float, y: float, ref_z: float) -> Optional[float]:
    """Get z from CARLA's road waypoint projection."""
    carla = _ensure_carla()
    loc = carla.Location(x=x, y=y, z=ref_z)
    try:
        cmap = world.get_map()
        wp = cmap.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            wp = cmap.get_waypoint(
                loc, project_to_road=True,
                lane_type=getattr(carla.LaneType, "Any", carla.LaneType.Driving),
            )
        if wp is not None:
            return float(wp.transform.location.z)
    except Exception:
        pass
    return None


def _hybrid_raycast_z(world, x: float, y: float, ref_z: float,
                      fwd_x: float, fwd_y: float) -> Optional[float]:
    """Robust multi-sample raycast around (x, y) using a small grid.

    Casts rays at a 3x3 grid of offsets along forward and lateral axes,
    then returns the 75th-percentile z to reject noise.
    """
    right_x = -fwd_y
    right_y = fwd_x
    z_samples: list[float] = []
    for fwd_off in RAYCAST_GRID_OFFSETS_M:
        for lat_off in RAYCAST_GRID_OFFSETS_M:
            px = x + fwd_x * fwd_off + right_x * lat_off
            py = y + fwd_y * fwd_off + right_y * lat_off
            z = _raycast_z(world, px, py, ref_z)
            if z is not None:
                z_samples.append(z)
    if not z_samples:
        return _waypoint_z(world, x, y, ref_z)
    z_samples.sort()
    idx = int(round(RAYCAST_HIGH_PERCENTILE * (len(z_samples) - 1)))
    return z_samples[idx]


def _ground_z_at(world, x: float, y: float, ref_z: float,
                 fwd_x: float, fwd_y: float,
                 cache: dict) -> Optional[float]:
    """Get ground z with spatial caching (quantized to 5cm grid).

    Uses a single direct raycast per position (fast) with waypoint-API
    fallback.  The old 3×3 hybrid grid (9 rays per position) was ~9×
    slower due to RPC overhead and is unnecessary on well-modelled maps.
    """
    key = (round(x * 20) / 20, round(y * 20) / 20)
    if key in cache:
        return cache[key]
    z = _raycast_z(world, x, y, ref_z)
    if z is None:
        z = _waypoint_z(world, x, y, ref_z)
    cache[key] = z
    return z


# ---------------------------------------------------------------------------
# Vehicle dimension lookup
# ---------------------------------------------------------------------------

_BLUEPRINT_BBOX_CACHE: Dict[str, Tuple[float, float, float]] = {}


def _get_vehicle_half_dims(world, model: str) -> Tuple[float, float, float]:
    """Return (half_len, half_wid, half_height) for a vehicle blueprint.

    Uses the blueprint library to query the bounding box. Falls back to
    reasonable defaults for sedans.
    """
    if model in _BLUEPRINT_BBOX_CACHE:
        return _BLUEPRINT_BBOX_CACHE[model]

    carla = _ensure_carla()
    half_len, half_wid, half_h = DEFAULT_HALF_LEN, DEFAULT_HALF_WID, 0.8
    try:
        bp_lib = world.get_blueprint_library()
        bp = bp_lib.find(model)
        if bp is not None:
            # Spawn a temporary actor to read its bounding box, then destroy.
            # Use a location high above the map to avoid collisions.
            spawn_tf = carla.Transform(
                carla.Location(x=0.0, y=0.0, z=500.0),
                carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
            )
            actor = world.try_spawn_actor(bp, spawn_tf)
            if actor is not None:
                try:
                    bbox = actor.bounding_box
                    half_len = max(0.8, float(bbox.extent.x) * 0.92)
                    half_wid = max(0.5, float(bbox.extent.y) * 0.92)
                    half_h = max(0.3, float(bbox.extent.z))
                finally:
                    actor.destroy()
    except Exception as exc:
        print(f"[GROUND-ALIGN] Could not query bbox for {model}: {exc}")

    _BLUEPRINT_BBOX_CACHE[model] = (half_len, half_wid, half_h)
    return (half_len, half_wid, half_h)


# ---------------------------------------------------------------------------
# Core: compute z / pitch / roll for one waypoint
# ---------------------------------------------------------------------------

def _compute_ground_pose(
    world,
    x: float, y: float, z: float, yaw_deg: float,
    half_len: float, half_wid: float,
    z_cache: dict,
) -> Optional[Tuple[float, float, float]]:
    """Compute (ground_z, pitch_deg, roll_deg) for a vehicle at (x, y, yaw).

    Raycasts at four wheel-corner positions and derives pitch/roll from the
    height differences.

    Returns None if raycasts fail.
    """
    yaw_rad = math.radians(yaw_deg)
    fwd_x = math.cos(yaw_rad)
    fwd_y = math.sin(yaw_rad)
    right_x = -fwd_y
    right_y = fwd_x

    # Four wheel positions: front-left, front-right, rear-left, rear-right
    corners = {
        "fl": (x + fwd_x * half_len + right_x * (-half_wid),
               y + fwd_y * half_len + right_y * (-half_wid)),
        "fr": (x + fwd_x * half_len + right_x * half_wid,
               y + fwd_y * half_len + right_y * half_wid),
        "rl": (x - fwd_x * half_len + right_x * (-half_wid),
               y - fwd_y * half_len + right_y * (-half_wid)),
        "rr": (x - fwd_x * half_len + right_x * half_wid,
               y - fwd_y * half_len + right_y * half_wid),
    }

    ground: Dict[str, float] = {}
    for label, (cx, cy) in corners.items():
        gz = _ground_z_at(world, cx, cy, z, fwd_x, fwd_y, z_cache)
        if gz is not None:
            ground[label] = gz

    # Need at least one front and one rear, one left and one right.
    front_vals = [ground[k] for k in ("fl", "fr") if k in ground]
    rear_vals = [ground[k] for k in ("rl", "rr") if k in ground]
    left_vals = [ground[k] for k in ("fl", "rl") if k in ground]
    right_vals = [ground[k] for k in ("fr", "rr") if k in ground]
    if not front_vals or not rear_vals or not left_vals or not right_vals:
        return None

    front_z = sum(front_vals) / len(front_vals)
    rear_z = sum(rear_vals) / len(rear_vals)
    left_z = sum(left_vals) / len(left_vals)
    right_z = sum(right_vals) / len(right_vals)

    wheelbase = 2.0 * half_len
    track_width = 2.0 * half_wid
    pitch_deg = math.degrees(math.atan2(front_z - rear_z, wheelbase))
    roll_deg = math.degrees(math.atan2(right_z - left_z, track_width))

    # Clamp to avoid extreme values from bad raycasts.
    pitch_deg = max(-MAX_PITCH_DEG, min(MAX_PITCH_DEG, pitch_deg))
    roll_deg = max(-MAX_ROLL_DEG, min(MAX_ROLL_DEG, roll_deg))

    # Center z = average of all four corners (or available ones).
    all_ground = list(ground.values())
    center_z = sum(all_ground) / len(all_ground) + GROUND_CLEARANCE_M

    return (center_z, pitch_deg, roll_deg)


# ---------------------------------------------------------------------------
# Temporal smoothing
# ---------------------------------------------------------------------------

def _smooth_series(values: List[Optional[float]], window: int) -> List[Optional[float]]:
    """Apply a moving-average filter, skipping None entries."""
    if window <= 0:
        return values
    n = len(values)
    out: List[Optional[float]] = [None] * n
    for i in range(n):
        if values[i] is None:
            continue
        accum = 0.0
        count = 0
        for j in range(max(0, i - window), min(n, i + window + 1)):
            if values[j] is not None:
                accum += values[j]
                count += 1
        out[i] = accum / count if count > 0 else values[i]
    return out


def _safe_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_angle_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _angle_delta_deg(a: float, b: float) -> float:
    return abs(_normalize_angle_deg(float(a) - float(b)))


def _tilt_is_nonzero(value: Optional[float], eps: float = TILT_EPS_DEG) -> bool:
    if value is None:
        return False
    return abs(_normalize_angle_deg(float(value))) > float(eps)


# ---------------------------------------------------------------------------
# XML read / write helpers
# ---------------------------------------------------------------------------

def _parse_route_xml(path: Path) -> ET.ElementTree:
    return ET.parse(path)


def _write_route_xml(tree: ET.ElementTree, path: Path) -> None:
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def align_route_file(
    world,
    xml_path: Path,
    model: str,
    is_walker: bool,
    verbose: bool = False,
) -> Dict[str, object]:
    """Ground-align a single route XML file.

    Modifies the XML in-place with corrected z, pitch, roll values.

    Returns a stats dict.
    """
    tree = _parse_route_xml(xml_path)
    root = tree.getroot()

    waypoints = list(root.iter("waypoint"))
    if not waypoints:
        return {"file": str(xml_path), "waypoints": 0, "aligned": 0, "skipped": "no_waypoints"}

    # Get vehicle dimensions (only matters for vehicles, not walkers).
    if is_walker:
        half_len, half_wid, half_h = 0.3, 0.2, 0.9
    else:
        half_len, half_wid, half_h = _get_vehicle_half_dims(world, model)

    if verbose:
        print(f"[GROUND-ALIGN]   {xml_path.name}: {len(waypoints)} waypoints, "
              f"model={model}, dims=({half_len:.2f}, {half_wid:.2f}, {half_h:.2f})")

    z_cache: dict = {}
    raw_z: List[Optional[float]] = []
    raw_pitch: List[Optional[float]] = []
    raw_roll: List[Optional[float]] = []

    # Pass 1: compute raw ground z / pitch / roll for each waypoint.
    for wp_elem in waypoints:
        x = float(wp_elem.attrib["x"])
        y = float(wp_elem.attrib["y"])
        z = float(wp_elem.attrib["z"])
        yaw = float(wp_elem.attrib.get("yaw", "0.0"))

        if is_walker:
            # Walkers: only correct z, no pitch/roll.
            yaw_rad = math.radians(yaw)
            fwd_x = math.cos(yaw_rad)
            fwd_y = math.sin(yaw_rad)
            gz = _ground_z_at(world, x, y, z, fwd_x, fwd_y, z_cache)
            if gz is not None:
                raw_z.append(gz + GROUND_CLEARANCE_M)
            else:
                raw_z.append(None)
            raw_pitch.append(0.0)
            raw_roll.append(0.0)
        else:
            result = _compute_ground_pose(world, x, y, z, yaw, half_len, half_wid, z_cache)
            if result is not None:
                raw_z.append(result[0])
                raw_pitch.append(result[1])
                raw_roll.append(result[2])
            else:
                raw_z.append(None)
                raw_pitch.append(None)
                raw_roll.append(None)

    # Pass 2: temporal smoothing.
    smooth_z = _smooth_series(raw_z, SMOOTH_WINDOW)
    smooth_pitch = _smooth_series(raw_pitch, SMOOTH_WINDOW)
    smooth_roll = _smooth_series(raw_roll, SMOOTH_WINDOW)

    # Pass 3: write back to XML.
    aligned_count = 0
    nonzero_pitch = 0
    nonzero_roll = 0
    nonzero_tilt_waypoints = 0
    changed_pitch = 0
    changed_roll = 0
    changed_tilt_waypoints = 0
    for i, wp_elem in enumerate(waypoints):
        sz = smooth_z[i]
        sp = smooth_pitch[i]
        sr = smooth_roll[i]
        if sz is None:
            # Could not raycast; leave original values.
            continue
        orig_pitch = _safe_float(wp_elem.attrib.get("pitch"))
        orig_roll = _safe_float(wp_elem.attrib.get("roll"))
        new_pitch = float(sp) if sp is not None else 0.0
        new_roll = float(sr) if sr is not None else 0.0
        wp_elem.attrib["z"] = f"{sz:.6f}"
        wp_elem.attrib["pitch"] = f"{new_pitch:.6f}"
        wp_elem.attrib["roll"] = f"{new_roll:.6f}"

        is_pitch_nonzero = _tilt_is_nonzero(new_pitch)
        is_roll_nonzero = _tilt_is_nonzero(new_roll)
        if is_pitch_nonzero:
            nonzero_pitch += 1
        if is_roll_nonzero:
            nonzero_roll += 1
        if is_pitch_nonzero or is_roll_nonzero:
            nonzero_tilt_waypoints += 1

        pitch_changed = (
            orig_pitch is None
            or _angle_delta_deg(orig_pitch, new_pitch) > TILT_EPS_DEG
        )
        roll_changed = (
            orig_roll is None
            or _angle_delta_deg(orig_roll, new_roll) > TILT_EPS_DEG
        )
        if pitch_changed:
            changed_pitch += 1
        if roll_changed:
            changed_roll += 1
        if pitch_changed or roll_changed:
            changed_tilt_waypoints += 1
        aligned_count += 1

    _write_route_xml(tree, xml_path)

    return {
        "file": str(xml_path),
        "waypoints": len(waypoints),
        "aligned": aligned_count,
        "failed": len(waypoints) - aligned_count,
        "nonzero_pitch": nonzero_pitch,
        "nonzero_roll": nonzero_roll,
        "nonzero_tilt_waypoints": nonzero_tilt_waypoints,
        "changed_pitch": changed_pitch,
        "changed_roll": changed_roll,
        "changed_tilt_waypoints": changed_tilt_waypoints,
    }


# ---------------------------------------------------------------------------
# Directory-level processing
# ---------------------------------------------------------------------------

def align_routes_dir(
    world,
    routes_dir: Path,
    verbose: bool = False,
) -> Dict[str, object]:
    """Ground-align all route XML files in a CARLA routes output directory.

    Reads actors_manifest.json to determine which files exist and what
    vehicle model each uses.

    Returns a summary report dict.
    """
    manifest_path = routes_dir / "actors_manifest.json"
    if not manifest_path.exists():
        return {"error": f"No actors_manifest.json in {routes_dir}"}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results: List[Dict[str, object]] = []
    total_wp = 0
    total_aligned = 0
    total_nonzero_pitch = 0
    total_nonzero_roll = 0
    total_nonzero_tilt_waypoints = 0
    total_changed_tilt_waypoints = 0
    files_with_nonzero_tilt = 0

    # Process ego routes.
    # We process BOTH the manifest-referenced file AND its _REPLAY variant
    # (if it exists) because:
    #   - The export writes ego XMLs as "ego_vehicle_N.xml"
    #   - The dashboard renames them to "ego_vehicle_N_REPLAY.xml"
    #   - GRP simplification creates a new "ego_vehicle_N.xml" (sparse waypoints)
    #   - If GRP fails, only the _REPLAY version exists
    #   - The runtime (log-replay mode) uses _REPLAY files
    # Both versions need ground alignment for correct pitch/roll/z.
    for ego_entry in manifest.get("ego", []):
        xml_rel = ego_entry.get("file", "")
        model = str(ego_entry.get("model", "vehicle.lincoln.mkz_2020"))

        # Determine both the manifest path and the _REPLAY variant
        xml_path = routes_dir / xml_rel
        stem = xml_path.stem
        replay_name = stem + "_REPLAY" + xml_path.suffix if "_REPLAY" not in stem else ""
        replay_path = xml_path.with_name(replay_name) if replay_name else None

        # Process the manifest-referenced file (GRP-simplified or original)
        if xml_path.exists():
            stats = align_route_file(world, xml_path, model, is_walker=False, verbose=verbose)
            results.append(stats)
            total_wp += int(stats.get("waypoints", 0))
            total_aligned += int(stats.get("aligned", 0))
            total_nonzero_pitch += int(stats.get("nonzero_pitch", 0))
            total_nonzero_roll += int(stats.get("nonzero_roll", 0))
            nonzero_tilt_in_file = int(stats.get("nonzero_tilt_waypoints", 0))
            total_nonzero_tilt_waypoints += nonzero_tilt_in_file
            total_changed_tilt_waypoints += int(stats.get("changed_tilt_waypoints", 0))
            if nonzero_tilt_in_file > 0:
                files_with_nonzero_tilt += 1
        else:
            if verbose:
                print(f"[GROUND-ALIGN] Skipping missing ego file: {xml_path}")

        # Also process the _REPLAY variant (full-resolution trajectory used by log-replay)
        if replay_path and replay_path.exists():
            stats = align_route_file(world, replay_path, model, is_walker=False, verbose=verbose)
            results.append(stats)
            total_wp += int(stats.get("waypoints", 0))
            total_aligned += int(stats.get("aligned", 0))
            total_nonzero_pitch += int(stats.get("nonzero_pitch", 0))
            total_nonzero_roll += int(stats.get("nonzero_roll", 0))
            nonzero_tilt_in_file = int(stats.get("nonzero_tilt_waypoints", 0))
            total_nonzero_tilt_waypoints += nonzero_tilt_in_file
            total_changed_tilt_waypoints += int(stats.get("changed_tilt_waypoints", 0))
            if nonzero_tilt_in_file > 0:
                files_with_nonzero_tilt += 1

    # Process actor routes by kind.
    for kind in ("npc", "walker", "static", "pedestrian", "bicycle", "cyclist"):
        for entry in manifest.get(kind, []):
            xml_rel = entry.get("file", "")
            model = str(entry.get("model", ""))
            xml_path = routes_dir / xml_rel
            if not xml_path.exists():
                if verbose:
                    print(f"[GROUND-ALIGN] Skipping missing file: {xml_path}")
                continue
            is_walker = kind in WALKER_ROLES
            stats = align_route_file(world, xml_path, model, is_walker=is_walker, verbose=verbose)
            results.append(stats)
            total_wp += int(stats.get("waypoints", 0))
            total_aligned += int(stats.get("aligned", 0))
            total_nonzero_pitch += int(stats.get("nonzero_pitch", 0))
            total_nonzero_roll += int(stats.get("nonzero_roll", 0))
            nonzero_tilt_in_file = int(stats.get("nonzero_tilt_waypoints", 0))
            total_nonzero_tilt_waypoints += nonzero_tilt_in_file
            total_changed_tilt_waypoints += int(stats.get("changed_tilt_waypoints", 0))
            if nonzero_tilt_in_file > 0:
                files_with_nonzero_tilt += 1

    report = {
        "routes_dir": str(routes_dir),
        "files_processed": len(results),
        "total_waypoints": total_wp,
        "total_aligned": total_aligned,
        "total_failed": total_wp - total_aligned,
        "total_nonzero_pitch": total_nonzero_pitch,
        "total_nonzero_roll": total_nonzero_roll,
        "total_nonzero_tilt_waypoints": total_nonzero_tilt_waypoints,
        "total_changed_tilt_waypoints": total_changed_tilt_waypoints,
        "files_with_nonzero_tilt": files_with_nonzero_tilt,
        "details": results,
    }
    if verbose:
        print(f"[GROUND-ALIGN] Done: {len(results)} files, "
              f"{total_aligned}/{total_wp} waypoints aligned, "
              f"{total_nonzero_tilt_waypoints} with nonzero tilt")
    return report


# ---------------------------------------------------------------------------
# CARLA connection
# ---------------------------------------------------------------------------

def connect_carla(
    host: str = "localhost",
    port: int = 2000,
    timeout: float = 10.0,
):
    """Connect to a running CARLA server and return (client, world)."""
    carla = _ensure_carla()
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    world = client.get_world()
    return client, world


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="CARLA ground-alignment postprocessor for route XML files.",
    )
    parser.add_argument(
        "routes_dir",
        type=str,
        help="Directory containing actors_manifest.json and route XML files.",
    )
    parser.add_argument(
        "--carla-host",
        type=str,
        default=os.environ.get("CARLA_HOST", "localhost"),
        help="CARLA server hostname (default: localhost).",
    )
    parser.add_argument(
        "--carla-port",
        type=int,
        default=int(os.environ.get("CARLA_PORT", "2000")),
        help="CARLA server port (default: 2000).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="CARLA client timeout in seconds.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-file details.",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default=None,
        help="Optional path to write the full alignment report JSON.",
    )
    parser.add_argument(
        "--require-nonzero-tilt",
        action="store_true",
        help="Exit non-zero if no waypoint receives nonzero pitch/roll.",
    )
    args = parser.parse_args(argv)

    routes_dir = Path(args.routes_dir).expanduser().resolve()
    if not routes_dir.is_dir():
        print(f"[GROUND-ALIGN] Error: {routes_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    print(f"[GROUND-ALIGN] Connecting to CARLA at {args.carla_host}:{args.carla_port} ...")
    client, world = connect_carla(args.carla_host, args.carla_port, args.timeout)
    town = world.get_map().name
    print(f"[GROUND-ALIGN] Connected. Map: {town}")

    report = align_routes_dir(world, routes_dir, verbose=args.verbose)

    if "error" in report:
        print(f"[GROUND-ALIGN] Error: {report['error']}", file=sys.stderr)
        sys.exit(1)

    if args.report_json:
        report_path = Path(args.report_json).expanduser()
        if not report_path.is_absolute():
            report_path = (routes_dir / report_path).resolve()
        else:
            report_path = report_path.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[GROUND-ALIGN] Report: {report_path}")

    print(f"[GROUND-ALIGN] Summary: {report['files_processed']} files, "
          f"{report['total_aligned']}/{report['total_waypoints']} waypoints aligned, "
          f"{report['total_failed']} failed, "
          f"{report['total_nonzero_tilt_waypoints']} nonzero-tilt waypoints "
          f"across {report['files_with_nonzero_tilt']} files")

    if args.require_nonzero_tilt and int(report.get("total_nonzero_tilt_waypoints", 0)) == 0:
        print(
            "[GROUND-ALIGN] Error: no nonzero pitch/roll detected after alignment.",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
