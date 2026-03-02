#!/usr/bin/env python3
"""
Visual crosswalk experiment on the currently loaded CARLA world.

The script draws the C-style crosswalk from explicit c### lane ids
(default c359..c364), places it near the lane ends closer to c333 than c728,
captures one overhead frame, and uses persistent CARLA debug drawing.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import queue
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

try:
    import carla  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError(
        "Could not import CARLA. Ensure the CARLA PythonAPI egg is on PYTHONPATH."
    ) from exc


# Tone down white to better match native lane paint and avoid bloom/glow.
WHITE = carla.Color(172, 172, 172)
# Colors for positive z-gradient debugging (small -> medium -> largest boost).
UP_GRADIENT_COLORS = (
    carla.Color(140, 220, 140),
    carla.Color(245, 210, 100),
    carla.Color(245, 120, 90),
)
IMAGE_W = 1280
IMAGE_H = 720
CAMERA_FOV_DEG = 112.0
CAMERA_HEIGHT_M = 22.0
CAMERA_BACK_OFFSET_M = 9.0
CAMERA_LOOKAHEAD_M = 2.0
CROSSWALK_CENTER_FORWARD_OFFSET_M = 0.0
# Small lift above surface to avoid z-fighting while still reading as paint.
CROSSWALK_DRAW_Z_OFFSET_M = 0.010
# Never draw below the detected road surface; keep a tiny positive clearance.
CROSSWALK_MIN_SURFACE_CLEARANCE_M = 0.002
CROSSWALK_GROUND_PROJECTION_MAX_DIST_M = 8.0
RAYCAST_Z_ABOVE_M = 60.0
RAYCAST_Z_BELOW_M = 60.0
RAYCAST_ROBUST_FORWARD_OFFSETS_M = (-0.6, 0.0, 0.6)
RAYCAST_ROBUST_LATERAL_OFFSETS_M = (-0.4, 0.0, 0.4)
RAYCAST_ROBUST_HIGH_PERCENTILE = 0.75
SURFACE_Z_CACHE_XY_STEP_M = 0.04
SURFACE_Z_CACHE_DIR_STEP = 0.05
BOX_LINE_THICKNESS_M = 0.03
CROSSWALK_C_LINE_THICKNESS_M = 0.44
CROSSWALK_C_BLOCK_LENGTH_M = 2.0
# Keep stripe count similar after thickening by slightly reducing auto gap.
CROSSWALK_C_GAP_LANE_RATIO = 0.14
CROSSWALK_C_SUBLINE_THICKNESS_M = 0.05
CROSSWALK_C_SUBLINE_SPACING_M = 0.05
STRIPE_TRIM_OUTERMOST_PER_SIDE_DEFAULT = 1
HIGH_Z_PRUNE_COUNT_DEFAULT = 4
CROSSWALK_C_MAX_LOCAL_Z_DROP_M = 0.0
# Per-cid max z adjustment (meters). Positive lifts, negative lowers.
# Disabled by default; per-line dynamic ground projection is the default behavior.
CROSSWALK_CID_Z_BOOSTS_DEFAULT = ""
# Applied to selected stripes in order: first, second, third.
# Requested behavior: first small, second lower, third highest.
CROSSWALK_CID_Z_GRADIENT_FACTORS = (0.30, 0.10, 1.00)
# Stripe selection direction by cid:
#   +1 -> choose stripes on positive-lateral side of lane center (outward)
#    0 -> choose absolute nearest around lane center
#   -1 -> choose stripes on negative-lateral side.
CROSSWALK_CID_Z_GRADIENT_DIR_BY_CID = {
    361: -1,
    359: 1,
}
# Gradient selection mode by cid:
#   side_outward: start near lane center on selected side, then move outward.
#   edge_outward: always use far edge stripes on selected side.
#   nearest: nearest-by-distance around lane center.
CROSSWALK_CID_Z_MODE_BY_CID = {
    361: "side_outward",
    359: "edge_outward",
}
# Optional per-cid index shift after stripe selection.
# Negative shifts selection toward smaller stripe indices (left in the current ordering).
CROSSWALK_CID_STRIPE_SHIFT_BY_CID = {
    361: -1,
}
CROSSWALK_SIDE_MARGIN_LANE_RATIO = 0.5
CROSSWALK_BOTTOM_INSET_M = 2.0
CROSSWALK_C_STRIPE_STEP_OVERRIDE_M = 0.0
TARGET_LANE_CIDS_DEFAULT = "359,360,361,362,363,364"
BOTTOM_ANCHOR_CID_DEFAULT = 333
TOP_ANCHOR_CID_DEFAULT = 728
LANE_SPEC_JSON_DEFAULT = "v2xpnp/map/lane_correspondence_diagnostics.json"
CARLA_MAP_CACHE_DEFAULT = "v2xpnp/map/carla_map_cache.pkl"
Z_APPROACHES_DEFAULT = "hybrid_raycast"


@dataclass
class ReferenceWaypoint:
    waypoint: carla.Waypoint
    lane_width: float
    transform: carla.Transform
    forward: carla.Vector3D
    perpendicular: carla.Vector3D


@dataclass(frozen=True)
class LaneSpec:
    cid: int
    road_id: int
    lane_id: int


@dataclass
class CrosswalkPlan:
    ref: ReferenceWaypoint
    center: carla.Location
    start_lateral: float
    end_lateral: float
    median_lane_width: float
    lane_count: int
    target_lane_specs: list[LaneSpec]
    lane_lateral_by_cid: dict[int, float]
    bottom_anchor: LaneSpec
    top_anchor: LaneSpec


@dataclass(frozen=True)
class CameraView:
    name: str
    transform: carla.Transform
    fov_deg: float


def _normalize_angle_deg(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def _vector_norm_xy(vec: carla.Vector3D) -> float:
    return math.sqrt(float(vec.x) ** 2 + float(vec.y) ** 2)


def _normalize_xy(vec: carla.Vector3D) -> carla.Vector3D:
    mag = _vector_norm_xy(vec)
    if mag <= 1e-8:
        return carla.Vector3D(1.0, 0.0, 0.0)
    return carla.Vector3D(float(vec.x) / mag, float(vec.y) / mag, 0.0)


def _offset_location(
    base: carla.Location,
    forward: carla.Vector3D,
    lateral: carla.Vector3D,
    forward_m: float = 0.0,
    lateral_m: float = 0.0,
    up_m: float = 0.0,
) -> carla.Location:
    return carla.Location(
        x=float(base.x) + float(forward.x) * forward_m + float(lateral.x) * lateral_m,
        y=float(base.y) + float(forward.y) * forward_m + float(lateral.y) * lateral_m,
        z=float(base.z) + up_m,
    )


def _quantize(value: float, step: float) -> int:
    if step <= 0.0:
        return int(round(float(value) * 1000.0))
    return int(round(float(value) / float(step)))


def _normalize_z_mode(mode: str) -> str:
    token = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "map": "waypoint",
        "map_waypoint": "waypoint",
        "wp": "waypoint",
        "ground": "ground_projection",
        "proj": "ground_projection",
        "ray": "raycast",
        "robust": "hybrid_raycast",
        "hybrid": "hybrid_raycast",
    }
    return aliases.get(token, token)


def _surface_z_from_waypoint(world: carla.World, loc: carla.Location) -> float | None:
    cmap = world.get_map()
    lane_any = getattr(carla.LaneType, "Any", carla.LaneType.Driving)
    wp = cmap.get_waypoint(
        loc,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if wp is None:
        wp = cmap.get_waypoint(
            loc,
            project_to_road=True,
            lane_type=lane_any,
        )
    if wp is None:
        return None
    return float(wp.transform.location.z)


def _surface_z_from_ground_projection(world: carla.World, loc: carla.Location) -> float | None:
    if not hasattr(world, "ground_projection"):
        return None
    try:
        projected = world.ground_projection(loc, float(CROSSWALK_GROUND_PROJECTION_MAX_DIST_M))
    except Exception:
        projected = None
    if projected is None:
        return None
    projected_loc = projected
    if hasattr(projected, "location"):
        projected_loc = getattr(projected, "location")
    if not hasattr(projected_loc, "z"):
        return None
    return float(getattr(projected_loc, "z"))


def _ray_hit_road_like(hit: object) -> bool:
    label = getattr(hit, "label", None)
    if label is None:
        return False
    if not hasattr(carla, "CityObjectLabel"):
        return False
    names = [
        "Roads",
        "RoadLines",
        "Sidewalks",
        "Ground",
        "Terrain",
    ]
    for name in names:
        if hasattr(carla.CityObjectLabel, name):
            if label == getattr(carla.CityObjectLabel, name):
                return True
    return False


def _surface_z_from_raycast(world: carla.World, loc: carla.Location) -> float | None:
    if not hasattr(world, "cast_ray"):
        return None
    start = carla.Location(
        x=float(loc.x),
        y=float(loc.y),
        z=float(loc.z) + float(RAYCAST_Z_ABOVE_M),
    )
    end = carla.Location(
        x=float(loc.x),
        y=float(loc.y),
        z=float(loc.z) - float(RAYCAST_Z_BELOW_M),
    )
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
        ref = _surface_z_from_ground_projection(world, loc)
        if ref is None:
            ref = _surface_z_from_waypoint(world, loc)
        if ref is not None:
            return min(fallback, key=lambda z: abs(float(z) - float(ref)))
        ordered = sorted(float(v) for v in fallback)
        return float(ordered[len(ordered) // 2])
    return None


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return float(values[0])
    vals = sorted(float(v) for v in values)
    qq = max(0.0, min(1.0, float(q)))
    idx = int(round(qq * float(len(vals) - 1)))
    return float(vals[idx])


def _surface_z_from_hybrid_raycast(
    world: carla.World,
    loc: carla.Location,
    fallback_forward: carla.Vector3D | None,
    fallback_lateral: carla.Vector3D | None,
) -> float | None:
    fwd = _normalize_xy(fallback_forward or carla.Vector3D(1.0, 0.0, 0.0))
    lat = _normalize_xy(
        fallback_lateral
        or carla.Vector3D(-float(fwd.y), float(fwd.x), 0.0)
    )

    samples: list[carla.Location] = []
    for forward_m in RAYCAST_ROBUST_FORWARD_OFFSETS_M:
        for lateral_m in RAYCAST_ROBUST_LATERAL_OFFSETS_M:
            samples.append(
                _offset_location(
                    base=loc,
                    forward=fwd,
                    lateral=lat,
                    forward_m=float(forward_m),
                    lateral_m=float(lateral_m),
                )
            )

    z_candidates: list[float] = []
    for sample in samples:
        z = _surface_z_from_raycast(world, sample)
        if z is not None:
            z_candidates.append(float(z))

    if z_candidates:
        return _percentile(z_candidates, float(RAYCAST_ROBUST_HIGH_PERCENTILE))
    return _surface_z_from_raycast(world, loc)


def _first_not_none(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return float(value)
    return None


def _surface_z_cache_key(
    loc: carla.Location,
    z_mode: str,
    fallback_forward: carla.Vector3D | None,
    fallback_lateral: carla.Vector3D | None,
) -> tuple:
    mode = _normalize_z_mode(z_mode)
    base = (
        mode,
        _quantize(float(loc.x), float(SURFACE_Z_CACHE_XY_STEP_M)),
        _quantize(float(loc.y), float(SURFACE_Z_CACHE_XY_STEP_M)),
    )
    if mode == "hybrid_raycast":
        fwd = _normalize_xy(fallback_forward or carla.Vector3D(1.0, 0.0, 0.0))
        lat = _normalize_xy(
            fallback_lateral
            or carla.Vector3D(-float(fwd.y), float(fwd.x), 0.0)
        )
        return base + (
            _quantize(float(fwd.x), float(SURFACE_Z_CACHE_DIR_STEP)),
            _quantize(float(fwd.y), float(SURFACE_Z_CACHE_DIR_STEP)),
            _quantize(float(lat.x), float(SURFACE_Z_CACHE_DIR_STEP)),
            _quantize(float(lat.y), float(SURFACE_Z_CACHE_DIR_STEP)),
        )
    return base


def _resolve_surface_z(
    world: carla.World,
    loc: carla.Location,
    z_mode: str,
    fallback_forward: carla.Vector3D | None = None,
    fallback_lateral: carla.Vector3D | None = None,
    z_cache: dict[tuple, float | None] | None = None,
    cache_only: bool = False,
) -> float | None:
    mode = _normalize_z_mode(z_mode)
    cache_key = None
    if z_cache is not None:
        cache_key = _surface_z_cache_key(loc, mode, fallback_forward, fallback_lateral)
        if cache_key in z_cache:
            return z_cache[cache_key]
        if bool(cache_only):
            return None
    elif bool(cache_only):
        return None

    z: float | None
    if mode == "waypoint":
        z = _surface_z_from_waypoint(world, loc)
    elif mode == "ground_projection":
        z = _first_not_none(
            _surface_z_from_ground_projection(world, loc),
            _surface_z_from_waypoint(world, loc),
        )
    elif mode == "raycast":
        z = _first_not_none(
            _surface_z_from_raycast(world, loc),
            _surface_z_from_ground_projection(world, loc),
            _surface_z_from_waypoint(world, loc),
        )
    else:
        # hybrid_raycast (default): robust multi-sample raycast first.
        z = _first_not_none(
            _surface_z_from_hybrid_raycast(world, loc, fallback_forward, fallback_lateral),
            _surface_z_from_ground_projection(world, loc),
            _surface_z_from_waypoint(world, loc),
        )

    if z_cache is not None and cache_key is not None:
        z_cache[cache_key] = z
    return z


def _snap_location_to_surface_z(
    world: carla.World,
    loc: carla.Location,
    z_offset: float,
    z_mode: str = "hybrid_raycast",
    fallback_forward: carla.Vector3D | None = None,
    fallback_lateral: carla.Vector3D | None = None,
    z_cache: dict[tuple, float | None] | None = None,
    cache_only: bool = False,
) -> carla.Location:
    z = _resolve_surface_z(
        world=world,
        loc=loc,
        z_mode=z_mode,
        fallback_forward=fallback_forward,
        fallback_lateral=fallback_lateral,
        z_cache=z_cache,
        cache_only=cache_only,
    )
    if z is None:
        if bool(cache_only):
            raise RuntimeError("surface-z cache miss in cache-only mode")
        z = float(loc.z)
    return carla.Location(
        x=float(loc.x),
        y=float(loc.y),
        z=float(z) + float(z_offset),
    )


def _surface_frame_at_location(
    world: carla.World,
    loc: carla.Location,
    fallback_forward: carla.Vector3D,
    z_offset: float,
    z_mode: str = "hybrid_raycast",
    z_cache: dict[tuple, float | None] | None = None,
    cache_only: bool = False,
) -> tuple[carla.Location, carla.Vector3D, carla.Vector3D]:
    cmap = world.get_map()
    lane_any = getattr(carla.LaneType, "Any", carla.LaneType.Driving)
    # Prefer driving lane orientation for crosswalk paint frame.
    wp = cmap.get_waypoint(
        loc,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if wp is None:
        wp = cmap.get_waypoint(
            loc,
            project_to_road=True,
            lane_type=lane_any,
        )
    if wp is None:
        fwd = _normalize_xy(fallback_forward)
        lat = _normalize_xy(carla.Vector3D(-float(fwd.y), float(fwd.x), 0.0))
        return (
            _snap_location_to_surface_z(
                world=world,
                loc=loc,
                z_offset=z_offset,
                z_mode=z_mode,
                fallback_forward=fwd,
                fallback_lateral=lat,
                z_cache=z_cache,
                cache_only=cache_only,
            ),
            fwd,
            lat,
        )

    fwd = _normalize_xy(wp.transform.get_forward_vector())
    if float(fwd.x) * float(fallback_forward.x) + float(fwd.y) * float(fallback_forward.y) < 0.0:
        fwd = carla.Vector3D(-float(fwd.x), -float(fwd.y), 0.0)
    lat = _normalize_xy(carla.Vector3D(-float(fwd.y), float(fwd.x), 0.0))
    surface_loc = _snap_location_to_surface_z(
        world=world,
        loc=loc,
        z_offset=z_offset,
        z_mode=z_mode,
        fallback_forward=fwd,
        fallback_lateral=lat,
        z_cache=z_cache,
        cache_only=cache_only,
    )
    return (surface_loc, fwd, lat)


def _distance_xy(a: carla.Location, b: carla.Location) -> float:
    return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))


def _mean_location(points: list[carla.Location]) -> carla.Location:
    if not points:
        return carla.Location()
    n = float(len(points))
    return carla.Location(
        x=sum(float(p.x) for p in points) / n,
        y=sum(float(p.y) for p in points) / n,
        z=sum(float(p.z) for p in points) / n,
    )


def _parse_csv_ints(value: str) -> list[int]:
    result: list[int] = []
    for part in str(value).split(","):
        token = part.strip()
        if not token:
            continue
        result.append(int(token))
    return result


def _parse_csv_tokens(value: str) -> list[str]:
    out: list[str] = []
    for part in str(value).split(","):
        token = part.strip()
        if token:
            out.append(token)
    return out


def _parse_cid_z_boosts(value: str) -> dict[int, float]:
    boosts: dict[int, float] = {}
    raw = str(value).strip()
    if not raw:
        return boosts
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if ":" not in token:
            continue
        cid_str, boost_str = token.split(":", 1)
        try:
            cid = int(cid_str.strip())
            boost = float(boost_str.strip())
        except Exception:
            continue
        boosts[int(cid)] = float(boost)
    return boosts


def load_surface_z_cache(path: Path | None) -> dict[tuple, float | None]:
    cache: dict[tuple, float | None] = {}
    if path is None:
        return cache
    if not Path(path).exists():
        return cache
    try:
        payload = json.loads(Path(path).read_text())
    except Exception:
        return cache
    entries = payload.get("entries", []) if isinstance(payload, dict) else []
    if not isinstance(entries, list):
        return cache
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        key_raw = entry.get("key")
        if not isinstance(key_raw, list):
            continue
        key = tuple(key_raw)
        z_raw = entry.get("z")
        if z_raw is None:
            cache[key] = None
            continue
        try:
            cache[key] = float(z_raw)
        except Exception:
            continue
    return cache


def save_surface_z_cache(path: Path | None, cache: dict[tuple, float | None]) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for key, z in sorted(cache.items(), key=lambda kv: repr(kv[0])):
        entries.append(
            {
                "key": list(key),
                "z": None if z is None else float(z),
            }
        )
    payload = {
        "version": 1,
        "entries": entries,
    }
    path.write_text(json.dumps(payload, separators=(",", ":")))


def _select_gradient_indices(
    stripe_positions: list[float],
    target_lateral: float,
    direction_pref: int,
    count: int,
    mode: str,
) -> tuple[list[int], int]:
    n = len(stripe_positions)
    if n <= 0 or count <= 0:
        return ([], int(direction_pref))

    count = min(int(count), int(n))
    positions = [float(v) for v in stripe_positions]
    direction_pref = 1 if int(direction_pref) > 0 else (-1 if int(direction_pref) < 0 else 0)
    mode = str(mode).strip().lower()
    resolved_dir = int(direction_pref)

    if mode in {"edge_outward", "side_outward"} and resolved_dir == 0:
        resolved_dir = 1 if float(target_lateral) >= 0.0 else -1

    if mode == "edge_outward":
        if resolved_dir >= 0:
            start = max(0, n - count)
            return (list(range(start, n)), int(resolved_dir))
        edge_block = list(range(0, count))
        edge_block.reverse()  # inner -> outer for negative side
        return (edge_block, int(resolved_dir))

    if mode == "side_outward":
        if resolved_dir > 0:
            side_candidates = [i for i, p in enumerate(positions) if p >= float(target_lateral) - 1e-6]
        elif resolved_dir < 0:
            side_candidates = [i for i, p in enumerate(positions) if p <= float(target_lateral) + 1e-6]
        else:
            side_candidates = list(range(n))
        if not side_candidates:
            side_candidates = list(range(n))
        base = min(side_candidates, key=lambda i: abs(float(positions[i]) - float(target_lateral)))
        out: list[int] = []
        if resolved_dir > 0:
            for k in range(count):
                idx = int(base) + k
                if idx < n:
                    out.append(idx)
        elif resolved_dir < 0:
            for k in range(count):
                idx = int(base) - k
                if idx >= 0:
                    out.append(idx)
        else:
            ordered = sorted(range(n), key=lambda i: abs(float(positions[i]) - float(target_lateral)))
            out = ordered[:count]
        if len(out) < count:
            remaining = [i for i in range(n) if i not in out]
            remaining = sorted(remaining, key=lambda i: abs(float(positions[i]) - float(target_lateral)))
            out.extend(remaining[: max(0, count - len(out))])
        return (out[:count], int(resolved_dir))

    # nearest (default)
    ordered = sorted(range(n), key=lambda i: abs(float(positions[i]) - float(target_lateral)))
    return (ordered[:count], int(resolved_dir))


def _as_finite_xy_pair(item: object) -> tuple[float, float] | None:
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        try:
            x = float(item[0])
            y = float(item[1])
        except Exception:
            return None
        if math.isfinite(x) and math.isfinite(y):
            return (x, y)
    if hasattr(item, "x") and hasattr(item, "y"):
        try:
            x = float(getattr(item, "x"))
            y = float(getattr(item, "y"))
        except Exception:
            return None
        if math.isfinite(x) and math.isfinite(y):
            return (x, y)
    return None


def _extract_xy_lines_recursive(obj: object, out_lines: list[list[tuple[float, float]]], depth: int = 0) -> None:
    if obj is None or depth > 12:
        return

    if isinstance(obj, dict):
        if "lines" in obj and isinstance(obj["lines"], (list, tuple)):
            for line in obj["lines"]:
                if not isinstance(line, (list, tuple)):
                    continue
                pts: list[tuple[float, float]] = []
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
        pts: list[tuple[float, float]] = []
        for p in obj:
            xy = _as_finite_xy_pair(p)
            if xy is None:
                pts = []
                break
            pts.append(xy)
        if len(pts) >= 2:
            out_lines.append(pts)
            return
        for v in obj:
            _extract_xy_lines_recursive(v, out_lines, depth + 1)
        return

    if hasattr(obj, "__dict__"):
        _extract_xy_lines_recursive(getattr(obj, "__dict__", {}), out_lines, depth + 1)


def _load_carla_map_cache_lines(path: Path) -> list[list[tuple[float, float]]]:
    if not path.exists():
        raise FileNotFoundError(f"CARLA map cache not found: {path}")
    with path.open("rb") as f:
        raw = pickle.load(f)

    lines: list[list[tuple[float, float]]] = []
    _extract_xy_lines_recursive(raw, lines)
    out: list[list[tuple[float, float]]] = []
    for ln in lines:
        if len(ln) >= 2:
            out.append([(float(x), float(y)) for x, y in ln])
    if not out:
        raise RuntimeError(f"No lane polylines found in CARLA map cache: {path}")
    return out


def _point_to_polyline_min_dist(
    point_xy: tuple[float, float],
    polyline_xy: list[tuple[float, float]],
) -> float:
    px, py = float(point_xy[0]), float(point_xy[1])
    best = float("inf")
    for x, y in polyline_xy:
        d = math.hypot(float(x) - px, float(y) - py)
        if d < best:
            best = d
    return best


def _endpoint_indices_toward_anchor(
    line_xy: list[tuple[float, float]],
    anchor_xy: list[tuple[float, float]],
) -> tuple[int, int]:
    d0 = _point_to_polyline_min_dist(line_xy[0], anchor_xy)
    d1 = _point_to_polyline_min_dist(line_xy[-1], anchor_xy)
    if d0 <= d1:
        return (0, len(line_xy) - 1)
    return (len(line_xy) - 1, 0)


def _unique_lane_specs(specs: list[LaneSpec]) -> list[LaneSpec]:
    unique: list[LaneSpec] = []
    seen: set[tuple[int, int, int]] = set()
    for spec in specs:
        key = (int(spec.cid), int(spec.road_id), int(spec.lane_id))
        if key in seen:
            continue
        seen.add(key)
        unique.append(spec)
    return unique


def _load_lane_spec_candidates_from_diagnostics(
    lane_spec_json: Path,
    cids: list[int],
) -> dict[int, list[LaneSpec]]:
    if not lane_spec_json.exists():
        raise FileNotFoundError(f"Lane spec json not found: {lane_spec_json}")

    payload = json.loads(lane_spec_json.read_text())
    v2_lanes = payload.get("v2_lanes")
    carla_lines = payload.get("carla_lines")
    if not isinstance(v2_lanes, list):
        raise RuntimeError(
            f"{lane_spec_json} is missing 'v2_lanes' array with road/lane metadata."
        )
    if not isinstance(carla_lines, list):
        carla_lines = []

    by_cid: dict[int, list[LaneSpec]] = {}
    for cid in cids:
        candidates: list[LaneSpec] = []

        if 0 <= cid < len(carla_lines):
            cl_entry = carla_lines[cid]
            if isinstance(cl_entry, dict):
                v2_indices: list[int] = []
                for key in (
                    "matched_v2_lane_indices",
                    "supplemental_v2_lane_indices",
                    "counterpart_v2_lane_indices",
                ):
                    values = cl_entry.get(key, [])
                    if isinstance(values, list):
                        for idx in values:
                            try:
                                v2_indices.append(int(idx))
                            except Exception:
                                continue

                for v2_idx in v2_indices:
                    if v2_idx < 0 or v2_idx >= len(v2_lanes):
                        continue
                    lane_entry = v2_lanes[v2_idx]
                    if not isinstance(lane_entry, dict):
                        continue
                    if "road_id" not in lane_entry or "lane_id" not in lane_entry:
                        continue
                    candidates.append(
                        LaneSpec(
                            cid=int(cid),
                            road_id=int(lane_entry["road_id"]),
                            lane_id=int(lane_entry["lane_id"]),
                        )
                    )

        # Fallback: interpret c### as direct index into v2_lanes.
        if not candidates and 0 <= cid < len(v2_lanes):
            lane_entry = v2_lanes[cid]
            if isinstance(lane_entry, dict) and "road_id" in lane_entry and "lane_id" in lane_entry:
                candidates.append(
                    LaneSpec(
                        cid=int(cid),
                        road_id=int(lane_entry["road_id"]),
                        lane_id=int(lane_entry["lane_id"]),
                    )
                )

        candidates = _unique_lane_specs(candidates)
        if not candidates:
            raise RuntimeError(f"No road_id/lane_id candidates found for c{cid} in {lane_spec_json}.")
        by_cid[int(cid)] = candidates

    return by_cid


def _cloud_to_cloud_min_dist(
    cloud_a: list[carla.Waypoint],
    cloud_b: list[carla.Waypoint],
) -> float:
    if not cloud_a or not cloud_b:
        return float("inf")
    best = float("inf")
    for wp_a in cloud_a:
        loc_a = wp_a.transform.location
        for wp_b in cloud_b:
            d = _distance_xy(loc_a, wp_b.transform.location)
            if d < best:
                best = d
    return best


def _choose_anchor_spec_with_cloud(
    lane_buckets: dict[tuple[int, int], list[carla.Waypoint]],
    candidates: list[LaneSpec],
) -> LaneSpec:
    for spec in candidates:
        if lane_buckets.get((int(spec.road_id), int(spec.lane_id))):
            return spec
    raise RuntimeError("None of the anchor lane candidates exist in current CARLA map.")


def _choose_target_spec_near_bottom_anchor(
    lane_buckets: dict[tuple[int, int], list[carla.Waypoint]],
    candidates: list[LaneSpec],
    bottom_anchor_cloud: list[carla.Waypoint],
) -> LaneSpec:
    best_spec: LaneSpec | None = None
    best_score = float("inf")
    for spec in candidates:
        cloud = lane_buckets.get((int(spec.road_id), int(spec.lane_id)), [])
        if not cloud:
            continue
        score = _cloud_to_cloud_min_dist(cloud, bottom_anchor_cloud)
        if score < best_score:
            best_score = score
            best_spec = spec
    if best_spec is None:
        raise RuntimeError("No target lane candidates exist in current CARLA map.")
    return best_spec


def _waypoints_by_road_lane(
    carla_map: carla.Map,
    sample_step_m: float,
) -> dict[tuple[int, int], list[carla.Waypoint]]:
    buckets: dict[tuple[int, int], list[carla.Waypoint]] = {}
    for wp in carla_map.generate_waypoints(float(sample_step_m)):
        key = (int(wp.road_id), int(wp.lane_id))
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(wp)
    return buckets


def _min_dist_to_cloud(
    loc: carla.Location,
    cloud: list[carla.Waypoint],
) -> float:
    if not cloud:
        return float("inf")
    return min(_distance_xy(loc, wp.transform.location) for wp in cloud)


def _build_crosswalk_plan_from_lane_cids(
    world: carla.World,
    carla_map_cache: Path,
    target_lane_cids: list[int],
    bottom_anchor_cid: int,
    top_anchor_cid: int,
    bottom_inset_m: float,
    side_margin_lane_ratio: float,
    waypoint_sample_step_m: float = 1.0,
) -> CrosswalkPlan:
    _ = waypoint_sample_step_m
    lines = _load_carla_map_cache_lines(carla_map_cache)
    required = sorted(set(target_lane_cids + [int(bottom_anchor_cid), int(top_anchor_cid)]))
    for cid in required:
        if cid < 0 or cid >= len(lines):
            raise RuntimeError(
                f"Requested c{cid} out of range for CARLA cache with {len(lines)} lines: {carla_map_cache}"
            )

    carla_map = world.get_map()
    bottom_anchor_line = lines[int(bottom_anchor_cid)]
    top_anchor_line = lines[int(top_anchor_cid)]

    lane_dirs: list[carla.Vector3D] = []
    lane_centers: list[carla.Location] = []
    lane_widths: list[float] = []
    target_specs: list[LaneSpec] = []

    for cid in target_lane_cids:
        line = lines[int(cid)]
        b_idx, t_idx = _endpoint_indices_toward_anchor(line, bottom_anchor_line)
        bottom_xy = line[b_idx]
        top_xy = line[t_idx]
        if _point_to_polyline_min_dist(bottom_xy, bottom_anchor_line) > _point_to_polyline_min_dist(bottom_xy, top_anchor_line):
            # If endpoint selection is inconsistent, flip it to stay near bottom anchor.
            b_idx, t_idx = t_idx, b_idx
            bottom_xy, top_xy = top_xy, bottom_xy

        lane_dir = _normalize_xy(
            carla.Vector3D(
                float(top_xy[0]) - float(bottom_xy[0]),
                float(top_xy[1]) - float(bottom_xy[1]),
                0.0,
            )
        )

        center_guess = _offset_location(
            base=carla.Location(x=float(bottom_xy[0]), y=float(bottom_xy[1]), z=0.0),
            forward=lane_dir,
            lateral=carla.Vector3D(-float(lane_dir.y), float(lane_dir.x), 0.0),
            forward_m=float(bottom_inset_m),
        )
        wp = carla_map.get_waypoint(
            center_guess,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if wp is not None:
            center_loc = carla.Location(
                x=float(center_guess.x),
                y=float(center_guess.y),
                z=float(wp.transform.location.z),
            )
            lane_width = float(wp.lane_width)
            target_specs.append(
                LaneSpec(
                    cid=int(cid),
                    road_id=int(wp.road_id),
                    lane_id=int(wp.lane_id),
                )
            )
        else:
            center_loc = center_guess
            lane_width = 3.5
            target_specs.append(
                LaneSpec(
                    cid=int(cid),
                    road_id=-1,
                    lane_id=0,
                )
            )

        lane_dirs.append(lane_dir)
        lane_centers.append(center_loc)
        lane_widths.append(float(lane_width))

    if not lane_centers:
        raise RuntimeError("No target lanes resolved for crosswalk generation.")

    base_dir = lane_dirs[0]
    avg_x = 0.0
    avg_y = 0.0
    for vec in lane_dirs:
        cur = vec
        if float(cur.x) * float(base_dir.x) + float(cur.y) * float(base_dir.y) < 0.0:
            cur = carla.Vector3D(-float(cur.x), -float(cur.y), 0.0)
        avg_x += float(cur.x)
        avg_y += float(cur.y)
    forward = _normalize_xy(carla.Vector3D(avg_x, avg_y, 0.0))
    perpendicular = _normalize_xy(carla.Vector3D(-float(forward.y), float(forward.x), 0.0))
    center = _mean_location(lane_centers)

    center_wp = carla_map.get_waypoint(
        center,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if center_wp is not None:
        center = carla.Location(
            x=float(center.x),
            y=float(center.y),
            z=float(center_wp.transform.location.z),
        )
    else:
        center_wp = carla_map.generate_waypoints(2.0)[0]

    median_lane_width = _median(lane_widths)
    if median_lane_width <= 0.1:
        median_lane_width = 3.5
    side_margin = float(side_margin_lane_ratio) * float(median_lane_width)

    offsets: list[float] = []
    for lane_center in lane_centers:
        dx = float(lane_center.x) - float(center.x)
        dy = float(lane_center.y) - float(center.y)
        offsets.append(dx * float(perpendicular.x) + dy * float(perpendicular.y))
    lane_lateral_by_cid = {
        int(target_specs[i].cid): float(offsets[i])
        for i in range(min(len(target_specs), len(offsets)))
    }

    start_lateral = min(offsets) - 0.5 * float(median_lane_width) - side_margin
    end_lateral = max(offsets) + 0.5 * float(median_lane_width) + side_margin
    if end_lateral < start_lateral:
        start_lateral, end_lateral = end_lateral, start_lateral

    yaw = math.degrees(math.atan2(float(forward.y), float(forward.x)))
    ref_transform = carla.Transform(center, carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0))
    ref = ReferenceWaypoint(
        waypoint=center_wp,
        lane_width=float(median_lane_width),
        transform=ref_transform,
        forward=forward,
        perpendicular=perpendicular,
    )
    return CrosswalkPlan(
        ref=ref,
        center=center,
        start_lateral=float(start_lateral),
        end_lateral=float(end_lateral),
        median_lane_width=float(median_lane_width),
        lane_count=len(target_specs),
        target_lane_specs=target_specs,
        lane_lateral_by_cid=lane_lateral_by_cid,
        bottom_anchor=LaneSpec(cid=int(bottom_anchor_cid), road_id=-1, lane_id=0),
        top_anchor=LaneSpec(cid=int(top_anchor_cid), road_id=-1, lane_id=0),
    )


def _curvature_score(waypoint: carla.Waypoint, step_m: float = 2.0, hops: int = 6) -> float | None:
    cur = waypoint
    yaws = [float(cur.transform.rotation.yaw)]
    for _ in range(max(1, int(hops))):
        nxt = cur.next(step_m)
        if not nxt:
            return None
        cur = nxt[0]
        if cur.is_junction:
            return None
        yaws.append(float(cur.transform.rotation.yaw))
    deltas = [abs(_normalize_angle_deg(yaws[i + 1] - yaws[i])) for i in range(len(yaws) - 1)]
    return max(deltas) if deltas else 0.0


def get_reference_waypoint(world: carla.World) -> ReferenceWaypoint:
    return _get_reference_waypoint_near(world, None)


def _get_reference_waypoint_near(
    world: carla.World,
    target_location: carla.Location | None,
    max_target_z_diff: float | None = None,
) -> ReferenceWaypoint:
    carla_map = world.get_map()
    candidates = carla_map.generate_waypoints(2.0)
    best_wp = None
    best_score = float("inf")

    for wp in candidates:
        if wp.is_junction:
            continue
        score = _curvature_score(wp, step_m=2.0, hops=6)
        if score is None:
            continue
        if target_location is None:
            ranking = score
        else:
            dx = float(wp.transform.location.x) - float(target_location.x)
            dy = float(wp.transform.location.y) - float(target_location.y)
            dz = abs(float(wp.transform.location.z) - float(target_location.z))
            if max_target_z_diff is not None and dz > float(max_target_z_diff):
                continue
            distance = math.hypot(dx, dy)
            # Prioritize proximity to requested location first; use straightness as tie-breaker.
            ranking = distance * 1000.0 + dz * 100.0 + score
        if ranking < best_score:
            best_score = ranking
            best_wp = wp

    if best_wp is None:
        if target_location is not None and max_target_z_diff is not None:
            return _get_reference_waypoint_near(world, target_location, None)
        raise RuntimeError("No non-junction straight waypoint found.")

    if target_location is None:
        straightness = _curvature_score(best_wp, step_m=2.0, hops=6)
        if straightness is None or straightness > 7.5:
            raise RuntimeError(
                f"Straight waypoint search failed curvature threshold (best={straightness})."
            )

    tf = best_wp.transform
    forward = _normalize_xy(tf.get_forward_vector())
    perpendicular = _normalize_xy(carla.Vector3D(-float(forward.y), float(forward.x), 0.0))
    return ReferenceWaypoint(
        waypoint=best_wp,
        lane_width=float(best_wp.lane_width),
        transform=tf,
        forward=forward,
        perpendicular=perpendicular,
    )


def _walker_target_location(
    scenario_export_dir: Path,
    walker_id: int,
    anchor: str = "mid",
) -> carla.Location:
    walker_file = (
        scenario_export_dir
        / "actors"
        / "walker"
        / f"ucla_v2_custom_Pedestrian_{int(walker_id)}_walker.xml"
    )
    if not walker_file.exists():
        raise FileNotFoundError(f"Walker trajectory file not found: {walker_file}")

    root = ET.parse(walker_file).getroot()
    points: list[tuple[float, float, float]] = []
    for wp in root.iter("waypoint"):
        points.append(
            (
                float(wp.attrib["x"]),
                float(wp.attrib["y"]),
                float(wp.attrib["z"]),
            )
        )
    if not points:
        raise RuntimeError(f"No waypoints in walker trajectory: {walker_file}")

    anchor = str(anchor).lower()
    if anchor == "start":
        sel = points[0]
    elif anchor == "end":
        sel = points[-1]
    elif anchor == "mean":
        sel = (
            sum(p[0] for p in points) / len(points),
            sum(p[1] for p in points) / len(points),
            sum(p[2] for p in points) / len(points),
        )
    else:
        sel = points[len(points) // 2]
    return carla.Location(x=sel[0], y=sel[1], z=sel[2])


def _look_at_rotation(origin: carla.Location, target: carla.Location) -> carla.Rotation:
    dx = float(target.x) - float(origin.x)
    dy = float(target.y) - float(origin.y)
    dz = float(target.z) - float(origin.z)
    yaw = math.degrees(math.atan2(dy, dx))
    dist_xy = max(1e-6, math.hypot(dx, dy))
    pitch = math.degrees(math.atan2(dz, dist_xy))
    return carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)


def spawn_camera(
    world: carla.World,
    ref: ReferenceWaypoint,
    look_at: carla.Location,
) -> carla.Sensor:
    cam_tf = _camera_transform_from_ref(ref, look_at)
    return spawn_camera_transform(world, cam_tf, fov_deg=CAMERA_FOV_DEG)


def spawn_camera_transform(
    world: carla.World,
    transform: carla.Transform,
    fov_deg: float = CAMERA_FOV_DEG,
) -> carla.Sensor:
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(IMAGE_W))
    bp.set_attribute("image_size_y", str(IMAGE_H))
    bp.set_attribute("fov", str(float(fov_deg)))
    bp.set_attribute("sensor_tick", "0.0")
    # Keep normal postprocess, but clamp bloom/lens artifacts where supported.
    if bp.has_attribute("enable_postprocess_effects"):
        bp.set_attribute("enable_postprocess_effects", "true")
    if bp.has_attribute("motion_blur_intensity"):
        bp.set_attribute("motion_blur_intensity", "0.0")
    if bp.has_attribute("lens_flare_intensity"):
        bp.set_attribute("lens_flare_intensity", "0.0")
    if bp.has_attribute("bloom_intensity"):
        bp.set_attribute("bloom_intensity", "0.0")
    if bp.has_attribute("exposure_compensation"):
        bp.set_attribute("exposure_compensation", "-0.8")
    return world.spawn_actor(bp, transform)


def _camera_transform_from_ref(
    ref: ReferenceWaypoint,
    look_at: carla.Location,
) -> carla.Transform:
    cam_loc = _offset_location(
        base=ref.transform.location,
        forward=ref.forward,
        lateral=ref.perpendicular,
        forward_m=-CAMERA_BACK_OFFSET_M,
        up_m=CAMERA_HEIGHT_M,
    )
    target = _offset_location(
        base=look_at,
        forward=ref.forward,
        lateral=ref.perpendicular,
        forward_m=CAMERA_LOOKAHEAD_M,
    )
    cam_rot = _look_at_rotation(cam_loc, target)
    return carla.Transform(cam_loc, cam_rot)


def _build_camera_views(
    world: carla.World,
    plan: CrosswalkPlan,
    crosswalk_center: carla.Location,
    include_approach_views: bool = True,
) -> list[CameraView]:
    ref = plan.ref
    views: list[CameraView] = [
        CameraView(
            name="overhead",
            transform=_camera_transform_from_ref(ref, crosswalk_center),
            fov_deg=CAMERA_FOV_DEG,
        )
    ]
    if not include_approach_views:
        return views

    # Driver-like approach views toward the crosswalk from different lanes/distances.
    approach_specs = [
        ("approach_far", 36.0, 0.0, 2.1, 0.0, 92.0),
        ("approach_mid", 24.0, 0.0, 2.1, 0.0, 96.0),
        ("approach_left", 24.0, -3.6, 2.1, -1.2, 96.0),
        ("approach_right", 24.0, 3.6, 2.1, 1.2, 96.0),
    ]
    for name, back_m, lateral_m, cam_height_m, look_lat_m, fov_deg in approach_specs:
        cam_ground = _offset_location(
            base=crosswalk_center,
            forward=ref.forward,
            lateral=ref.perpendicular,
            forward_m=-float(back_m),
            lateral_m=float(lateral_m),
        )
        cam_loc = _snap_location_to_surface_z(
            world=world,
            loc=cam_ground,
            z_offset=float(cam_height_m),
            z_mode="ground_projection",
        )
        look_loc = _offset_location(
            base=crosswalk_center,
            forward=ref.forward,
            lateral=ref.perpendicular,
            lateral_m=float(look_lat_m),
            up_m=0.2,
        )
        cam_rot = _look_at_rotation(cam_loc, look_loc)
        views.append(
            CameraView(
                name=name,
                transform=carla.Transform(cam_loc, cam_rot),
                fov_deg=float(fov_deg),
            )
        )
    return views


def _project_world_to_image(
    cam_tf: carla.Transform,
    world_loc: carla.Location,
    width: int = 1280,
    height: int = 720,
    fov_deg: float = 90.0,
) -> tuple[float, float, float] | None:
    w2c = cam_tf.get_inverse_matrix()
    x = float(world_loc.x)
    y = float(world_loc.y)
    z = float(world_loc.z)

    cam_x = w2c[0][0] * x + w2c[0][1] * y + w2c[0][2] * z + w2c[0][3]
    cam_y = w2c[1][0] * x + w2c[1][1] * y + w2c[1][2] * z + w2c[1][3]
    cam_z = w2c[2][0] * x + w2c[2][1] * y + w2c[2][2] * z + w2c[2][3]

    # CARLA camera to conventional CV frame.
    cv_x = cam_y
    cv_y = -cam_z
    cv_z = cam_x
    if cv_z <= 1e-4:
        return None

    hfov = math.radians(float(fov_deg))
    vfov = 2.0 * math.atan(math.tan(hfov * 0.5) * (float(height) / float(width)))
    fx = float(width) / (2.0 * math.tan(hfov * 0.5))
    fy = float(height) / (2.0 * math.tan(vfov * 0.5))
    cx = float(width) * 0.5
    cy = float(height) * 0.5

    u = fx * (cv_x / cv_z) + cx
    v = fy * (cv_y / cv_z) + cy
    return (u, v, cv_z)


def _stripe_offsets(count: int, stripe_length: float, total_depth: float) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [0.0]
    gap = (total_depth - count * stripe_length) / (count - 1)
    if gap < 0.0:
        raise ValueError("Invalid stripe geometry: negative spacing.")
    start = -0.5 * total_depth + 0.5 * stripe_length
    step = stripe_length + gap
    return [start + idx * step for idx in range(count)]


def _collect_driving_lanes_across_corridor(
    world: carla.World,
    ref: ReferenceWaypoint,
    scan_half_width_m: float = 120.0,
    lateral_step_m: float = 0.5,
    max_longitudinal_error_m: float = 20.0,
    min_heading_alignment_abs: float = 0.8,
) -> list[carla.Waypoint]:
    cmap = world.get_map()
    lanes: dict[tuple[int, int, int], tuple[float, carla.Waypoint]] = {}

    samples = int(round((2.0 * scan_half_width_m) / lateral_step_m))
    for idx in range(samples + 1):
        lateral = -scan_half_width_m + idx * lateral_step_m
        query = _offset_location(
            base=ref.transform.location,
            forward=ref.forward,
            lateral=ref.perpendicular,
            lateral_m=lateral,
        )
        wp = cmap.get_waypoint(
            query,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if wp is None:
            continue

        wp_fwd = _normalize_xy(wp.transform.get_forward_vector())
        align = abs(
            float(wp_fwd.x) * float(ref.forward.x)
            + float(wp_fwd.y) * float(ref.forward.y)
        )
        if align < float(min_heading_alignment_abs):
            continue

        dx = float(wp.transform.location.x) - float(ref.transform.location.x)
        dy = float(wp.transform.location.y) - float(ref.transform.location.y)
        along = abs(dx * float(ref.forward.x) + dy * float(ref.forward.y))
        if along > float(max_longitudinal_error_m):
            continue

        key = (int(wp.road_id), int(wp.section_id), int(wp.lane_id))
        prev = lanes.get(key)
        if prev is None or along < prev[0]:
            lanes[key] = (along, wp)

    return [entry[1] for entry in lanes.values()]


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    if len(s) % 2 == 1:
        return float(s[mid])
    return 0.5 * (float(s[mid - 1]) + float(s[mid]))


def _crosswalk_span_right_to_right_with_margin(
    world: carla.World,
    ref: ReferenceWaypoint,
) -> tuple[float, float, float, float, int]:
    lanes = _collect_driving_lanes_across_corridor(world, ref)
    lane_offsets: list[tuple[float, float]] = []

    for lane_wp in lanes:
        lane_loc = lane_wp.transform.location
        dx = float(lane_loc.x) - float(ref.transform.location.x)
        dy = float(lane_loc.y) - float(ref.transform.location.y)
        offset = dx * float(ref.perpendicular.x) + dy * float(ref.perpendicular.y)
        lane_offsets.append((offset, float(lane_wp.lane_width)))

    lane_offsets.sort(key=lambda item: item[0])
    if not lane_offsets:
        half = 0.5 * float(ref.lane_width)
        return (-half, half, float(ref.lane_width), 0.5 * float(ref.lane_width), 1)

    min_off, min_w = lane_offsets[0]
    max_off, max_w = lane_offsets[-1]
    center_steps = [
        abs(lane_offsets[i + 1][0] - lane_offsets[i][0])
        for i in range(len(lane_offsets) - 1)
        if abs(lane_offsets[i + 1][0] - lane_offsets[i][0]) > 1e-4
    ]
    valid_steps = [d for d in center_steps if d <= 5.5]
    lane_spacing = _median(valid_steps or center_steps) if center_steps else float(ref.lane_width)
    margin = 0.5 * lane_spacing

    start_lateral = float(min_off) - 0.5 * float(min_w) - margin
    end_lateral = float(max_off) + 0.5 * float(max_w) + margin
    return (start_lateral, end_lateral, lane_spacing, margin, len(lane_offsets))


def generate_crosswalk_A(world: carla.World, ref: ReferenceWaypoint) -> None:
    stripe_count = 8
    stripe_length = 0.4
    stripe_height = 0.02
    total_depth = 4.0
    z_offset = CROSSWALK_DRAW_Z_OFFSET_M
    center_forward_offset = CROSSWALK_CENTER_FORWARD_OFFSET_M

    offsets = _stripe_offsets(stripe_count, stripe_length, total_depth)
    for offset in offsets:
        center_forward = center_forward_offset + offset
        center = _offset_location(
            base=ref.transform.location,
            forward=ref.forward,
            lateral=ref.perpendicular,
            forward_m=center_forward,
            up_m=z_offset,
        )
        bbox = carla.BoundingBox(
            center,
            carla.Vector3D(stripe_length * 0.5, ref.lane_width * 0.5, stripe_height * 0.5),
        )
        world.debug.draw_box(
            bbox,
            ref.transform.rotation,
            thickness=BOX_LINE_THICKNESS_M,
            color=WHITE,
            life_time=0.0,
            persistent_lines=True,
        )


def generate_crosswalk_B(world: carla.World, ref: ReferenceWaypoint) -> None:
    total_depth = 4.0
    stripe_height = 0.02
    z_offset = CROSSWALK_DRAW_Z_OFFSET_M
    center_forward_offset = CROSSWALK_CENTER_FORWARD_OFFSET_M

    center = _offset_location(
        base=ref.transform.location,
        forward=ref.forward,
        lateral=ref.perpendicular,
        forward_m=center_forward_offset,
        up_m=z_offset,
    )
    bbox = carla.BoundingBox(
        center,
        carla.Vector3D(total_depth * 0.5, ref.lane_width * 0.5, stripe_height * 0.5),
    )
    world.debug.draw_box(
        bbox,
        ref.transform.rotation,
        thickness=BOX_LINE_THICKNESS_M,
        color=WHITE,
        life_time=0.0,
        persistent_lines=True,
    )


def generate_crosswalk_C(
    world: carla.World,
    plan: CrosswalkPlan,
    stripe_step_override: float | None = None,
    cid_z_boosts: dict[int, float] | None = None,
    z_mode: str = "hybrid_raycast",
    z_cache: dict[tuple, float | None] | None = None,
    trim_outermost_per_side: int = STRIPE_TRIM_OUTERMOST_PER_SIDE_DEFAULT,
    high_z_prune_count: int = HIGH_Z_PRUNE_COUNT_DEFAULT,
    surface_z_cache_only: bool = False,
    draw_life_time_s: float = 0.0,
    draw_persistent_lines: bool = True,
) -> None:
    ref = plan.ref
    cid_z_boosts = cid_z_boosts or {}
    z_offset = CROSSWALK_DRAW_Z_OFFSET_M
    stripe_width = CROSSWALK_C_LINE_THICKNESS_M
    block_length = CROSSWALK_C_BLOCK_LENGTH_M
    auto_gap = CROSSWALK_C_GAP_LANE_RATIO * float(plan.median_lane_width)
    if stripe_step_override is None or float(stripe_step_override) <= 0.0:
        stripe_step = max(0.2, float(stripe_width) + float(auto_gap))
    else:
        stripe_step = max(0.2, float(stripe_step_override))
    start_lateral = float(plan.start_lateral)
    end_lateral = float(plan.end_lateral)
    total_span = max(0.5, end_lateral - start_lateral)
    stripe_positions: list[float] = []
    cursor = float(start_lateral)
    while cursor <= float(end_lateral) + 1e-6:
        stripe_positions.append(cursor)
        cursor += float(stripe_step)
    if len(stripe_positions) < 2:
        stripe_positions = [float(start_lateral), float(end_lateral)]
    trim_n = max(0, int(trim_outermost_per_side))
    if trim_n > 0 and len(stripe_positions) > 2 * trim_n:
        stripe_positions = stripe_positions[trim_n : len(stripe_positions) - trim_n]
    elif trim_n > 0 and len(stripe_positions) > 2:
        stripe_positions = stripe_positions[1:-1]

    subline_thickness = min(
        float(CROSSWALK_C_SUBLINE_THICKNESS_M),
        max(0.01, 0.45 * float(stripe_width)),
    )
    subline_spacing = max(0.01, float(CROSSWALK_C_SUBLINE_SPACING_M))
    max_local_drop = max(0.0, float(CROSSWALK_C_MAX_LOCAL_Z_DROP_M))
    subline_count = max(1, int(math.ceil(float(stripe_width) / subline_spacing)))
    if subline_count == 1:
        sub_offsets = [0.0]
    else:
        sub_step = float(stripe_width) / float(subline_count - 1)
        sub_offsets = [(-0.5 * float(stripe_width)) + float(i) * sub_step for i in range(subline_count)]

    total_drawn = 0
    stripe_extra_z = [0.0 for _ in stripe_positions]
    stripe_draw_offsets = [float(z_offset) for _ in stripe_positions]
    stripe_colors = [WHITE for _ in stripe_positions]
    stripe_up_boost = [0.0 for _ in stripe_positions]
    gradient_hits: dict[int, list[int]] = {}
    gradient_dir_used: dict[int, int] = {}
    gradient_shift_used: dict[int, int] = {}
    gradient_scale_used: dict[int, float] = {}
    gradient_factors = [float(v) for v in CROSSWALK_CID_Z_GRADIENT_FACTORS if float(v) > 0.0]
    if not gradient_factors:
        gradient_factors = [1.0]

    for cid, max_boost in sorted(cid_z_boosts.items()):
        if abs(float(max_boost)) <= 1e-6:
            continue
        if cid not in plan.lane_lateral_by_cid:
            continue
        target_lateral = float(plan.lane_lateral_by_cid[int(cid)])
        direction_pref = int(CROSSWALK_CID_Z_GRADIENT_DIR_BY_CID.get(int(cid), 0))
        mode = str(CROSSWALK_CID_Z_MODE_BY_CID.get(int(cid), "nearest"))
        ordered, resolved_dir = _select_gradient_indices(
            stripe_positions=stripe_positions,
            target_lateral=float(target_lateral),
            direction_pref=int(direction_pref),
            count=int(len(gradient_factors)),
            mode=mode,
        )
        stripe_shift = int(CROSSWALK_CID_STRIPE_SHIFT_BY_CID.get(int(cid), 0))
        if stripe_shift != 0 and ordered:
            shifted: list[int] = []
            for idx in ordered:
                j = int(idx) + int(stripe_shift)
                if 0 <= j < len(stripe_positions) and j not in shifted:
                    shifted.append(j)
            if shifted:
                ordered = shifted
        if not ordered:
            continue
        use_n = min(len(gradient_factors), len(ordered))
        selected = ordered[:use_n]
        # Keep requested gradient shape but rescale amplitude into drawable range.
        # This prevents large negative boosts from being fully clipped (and becoming visually "no-op").
        scale = 1.0
        if float(max_boost) < 0.0:
            max_down = max(0.0, float(z_offset) - float(CROSSWALK_MIN_SURFACE_CLEARANCE_M))
            req_down = abs(float(max_boost))
            if req_down > 1e-6 and req_down > max_down and max_down > 0.0:
                scale = float(max_down / req_down)
        gradient_scale_used[int(cid)] = float(scale)
        # Assign boosts in requested order: first, second, third.
        for rank, stripe_idx in enumerate(selected):
            factor = float(gradient_factors[rank])
            boost = float(max_boost) * float(scale) * factor
            # Keep the strongest signed adjustment if multiple cid windows overlap.
            if abs(boost) > abs(float(stripe_extra_z[stripe_idx])):
                stripe_extra_z[stripe_idx] = boost
            if boost > float(stripe_up_boost[stripe_idx]):
                stripe_up_boost[stripe_idx] = float(boost)
                color_idx = min(int(rank), len(UP_GRADIENT_COLORS) - 1)
                stripe_colors[stripe_idx] = UP_GRADIENT_COLORS[color_idx]
        gradient_hits[int(cid)] = selected
        gradient_dir_used[int(cid)] = int(resolved_dir)
        gradient_shift_used[int(cid)] = int(stripe_shift)

    stripe_draw_offsets = [
        max(float(CROSSWALK_MIN_SURFACE_CLEARANCE_M), float(z_offset) + float(extra))
        for extra in stripe_extra_z
    ]
    removed_high_indices: set[int] = set()
    prune_n = max(0, int(high_z_prune_count))
    high_z_threshold: float | None = None
    if prune_n > 0 and len(stripe_positions) > prune_n:
        stripe_z_estimates: list[tuple[int, float]] = []
        for idx, lateral_m in enumerate(stripe_positions):
            stripe_mid = _offset_location(
                base=ref.transform.location,
                forward=ref.forward,
                lateral=ref.perpendicular,
                lateral_m=float(lateral_m),
            )
            z_surface = _resolve_surface_z(
                world=world,
                loc=stripe_mid,
                z_mode=z_mode,
                fallback_forward=ref.forward,
                fallback_lateral=ref.perpendicular,
                z_cache=z_cache,
                cache_only=surface_z_cache_only,
            )
            if z_surface is None:
                if bool(surface_z_cache_only):
                    raise RuntimeError("surface-z cache miss while ranking high-z stripes")
                z_surface = float(stripe_mid.z)
            z_world = float(z_surface) + float(stripe_draw_offsets[idx])
            stripe_z_estimates.append((idx, z_world))
        ranked = sorted(stripe_z_estimates, key=lambda item: float(item[1]), reverse=True)
        removed_high_indices = {int(idx) for idx, _ in ranked[:prune_n]}
        high_z_threshold = float(ranked[prune_n - 1][1])

    for stripe_idx, lateral_m in enumerate(stripe_positions):
        if stripe_idx in removed_high_indices:
            continue
        draw_offset = float(stripe_draw_offsets[stripe_idx])
        stripe_color = stripe_colors[stripe_idx]
        stripe_mid = _offset_location(
            base=ref.transform.location,
            forward=ref.forward,
            lateral=ref.perpendicular,
            lateral_m=lateral_m,
        )
        surface_mid, surface_forward, surface_lateral = _surface_frame_at_location(
            world=world,
            loc=stripe_mid,
            fallback_forward=ref.forward,
            z_offset=draw_offset,
            z_mode=z_mode,
            z_cache=z_cache,
            cache_only=surface_z_cache_only,
        )
        for sub_offset in sub_offsets:
            sub_mid = _offset_location(
                base=surface_mid,
                forward=surface_forward,
                lateral=surface_lateral,
                lateral_m=float(sub_offset),
            )
            sub_mid_surface = _snap_location_to_surface_z(
                world=world,
                loc=sub_mid,
                z_offset=draw_offset,
                z_mode=z_mode,
                fallback_forward=surface_forward,
                fallback_lateral=surface_lateral,
                z_cache=z_cache,
                cache_only=surface_z_cache_only,
            )
            start = _offset_location(
                base=sub_mid_surface,
                forward=surface_forward,
                lateral=surface_lateral,
                forward_m=-0.5 * block_length,
            )
            end = _offset_location(
                base=sub_mid_surface,
                forward=surface_forward,
                lateral=surface_lateral,
                forward_m=0.5 * block_length,
            )
            start = _snap_location_to_surface_z(
                world=world,
                loc=start,
                z_offset=draw_offset,
                z_mode=z_mode,
                fallback_forward=surface_forward,
                fallback_lateral=surface_lateral,
                z_cache=z_cache,
                cache_only=surface_z_cache_only,
            )
            end = _snap_location_to_surface_z(
                world=world,
                loc=end,
                z_offset=draw_offset,
                z_mode=z_mode,
                fallback_forward=surface_forward,
                fallback_lateral=surface_lateral,
                z_cache=z_cache,
                cache_only=surface_z_cache_only,
            )

            # Guard against occasional low outlier surface samples that make a stripe vanish.
            mid_z = float(sub_mid_surface.z)
            min_allowed_z = mid_z - max_local_drop
            if float(start.z) < min_allowed_z:
                start = carla.Location(x=float(start.x), y=float(start.y), z=mid_z)
            if float(end.z) < min_allowed_z:
                end = carla.Location(x=float(end.x), y=float(end.y), z=mid_z)

            world.debug.draw_line(
                start,
                end,
                thickness=subline_thickness,
                color=stripe_color,
                life_time=float(draw_life_time_s),
                persistent_lines=bool(draw_persistent_lines),
            )
            total_drawn += 1
    print(
        f"[C] span_lateral=({start_lateral:.2f},{end_lateral:.2f}) total_span={total_span:.2f} "
        f"lane_count={plan.lane_count} median_lane_width={plan.median_lane_width:.2f} "
        f"block_length={block_length:.2f} stripe_width={stripe_width:.2f} "
        f"subline_thickness={subline_thickness:.3f} sublines={subline_count} "
        f"gap={auto_gap:.2f} stripe_step={stripe_step:.2f} stripes={len(stripe_positions)} "
        f"z_mode={_normalize_z_mode(z_mode)} "
        f"z_offset={z_offset:.4f} draw_offset_range=({min(stripe_draw_offsets):.4f},{max(stripe_draw_offsets):.4f}) "
        f"max_drop={max_local_drop:.3f} drawn_segments={total_drawn}"
    )
    if trim_n > 0:
        print(f"[C] trimmed_outermost_per_side={trim_n}")
    if removed_high_indices:
        print(
            f"[C] pruned_high_z_stripes={len(removed_high_indices)} "
            f"threshold_z={high_z_threshold:.4f} indices={sorted(removed_high_indices)}"
        )
    if cid_z_boosts:
        parts: list[str] = []
        for cid in sorted(cid_z_boosts.keys()):
            if int(cid) not in plan.lane_lateral_by_cid:
                parts.append(f"c{cid}:not-mapped")
                continue
            idxs = gradient_hits.get(int(cid), [])
            boosts = [stripe_extra_z[i] for i in idxs]
            boost_txt = "/".join(f"{float(v):+.3f}" for v in boosts) if boosts else "-"
            direction_pref = int(gradient_dir_used.get(int(cid), CROSSWALK_CID_Z_GRADIENT_DIR_BY_CID.get(int(cid), 0)))
            mode = str(CROSSWALK_CID_Z_MODE_BY_CID.get(int(cid), "nearest"))
            shift_used = int(gradient_shift_used.get(int(cid), CROSSWALK_CID_STRIPE_SHIFT_BY_CID.get(int(cid), 0)))
            scale_used = float(gradient_scale_used.get(int(cid), 1.0))
            parts.append(
                f"c{cid}:max={float(cid_z_boosts[cid]):+.3f} scale={scale_used:.3f} dir={direction_pref:+d} shift={shift_used:+d} mode={mode} gradient={boost_txt}"
            )
        applied = ", ".join(parts)
        print(f"[C] cid_z_boosts: {applied}")
    colorized_count = sum(1 for v in stripe_up_boost if float(v) > 0.0)
    if colorized_count > 0:
        print(f"[C] colored_up_gradient_stripes={colorized_count}/{len(stripe_positions)}")


def apply_ucla_v2_crosswalk(
    world: carla.World,
    carla_map_cache: Path = Path(CARLA_MAP_CACHE_DEFAULT),
    target_lane_cids: list[int] | None = None,
    bottom_anchor_cid: int = BOTTOM_ANCHOR_CID_DEFAULT,
    top_anchor_cid: int = TOP_ANCHOR_CID_DEFAULT,
    bottom_inset_m: float = CROSSWALK_BOTTOM_INSET_M,
    side_margin_lane_ratio: float = CROSSWALK_SIDE_MARGIN_LANE_RATIO,
    waypoint_sample_step_m: float = 1.0,
    stripe_step_override: float = CROSSWALK_C_STRIPE_STEP_OVERRIDE_M,
    cid_z_boosts: dict[int, float] | None = None,
    z_mode: str = "hybrid_raycast",
    z_cache: dict[tuple, float | None] | None = None,
    trim_outermost_per_side: int = 2,
    high_z_prune_count: int = 5,
    surface_z_cache_only: bool = False,
    draw_life_time_s: float = 0.0,
    draw_persistent_lines: bool = True,
) -> CrosswalkPlan:
    lane_cids = target_lane_cids or _parse_csv_ints(TARGET_LANE_CIDS_DEFAULT)
    if not lane_cids:
        raise RuntimeError("No target lane ids provided for UCLA v2 crosswalk placement.")
    plan = _build_crosswalk_plan_from_lane_cids(
        world=world,
        carla_map_cache=Path(carla_map_cache),
        target_lane_cids=lane_cids,
        bottom_anchor_cid=int(bottom_anchor_cid),
        top_anchor_cid=int(top_anchor_cid),
        bottom_inset_m=float(bottom_inset_m),
        side_margin_lane_ratio=float(side_margin_lane_ratio),
        waypoint_sample_step_m=float(waypoint_sample_step_m),
    )
    generate_crosswalk_C(
        world=world,
        plan=plan,
        stripe_step_override=float(stripe_step_override),
        cid_z_boosts=cid_z_boosts or {},
        z_mode=str(z_mode),
        z_cache=z_cache,
        trim_outermost_per_side=int(trim_outermost_per_side),
        high_z_prune_count=int(high_z_prune_count),
        surface_z_cache_only=bool(surface_z_cache_only),
        draw_life_time_s=float(draw_life_time_s),
        draw_persistent_lines=bool(draw_persistent_lines),
    )
    return plan


def capture_image(
    world: carla.World,
    camera: carla.Sensor,
    output_path: Path,
    timeout_s: float = 8.0,
    settle_frames: int = 5,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img_queue: "queue.Queue[carla.Image]" = queue.Queue()
    camera.listen(img_queue.put)

    image = None
    is_sync = bool(getattr(world.get_settings(), "synchronous_mode", False))
    deadline = time.time() + float(timeout_s)

    for _ in range(max(1, int(settle_frames))):
        if is_sync:
            world.tick()
        else:
            world.wait_for_tick(seconds=1.0)

    while True:
        try:
            img_queue.get_nowait()
        except queue.Empty:
            break

    while time.time() < deadline:
        if is_sync:
            world.tick()
        else:
            world.wait_for_tick(seconds=1.0)
        try:
            image = img_queue.get(timeout=0.5)
            break
        except queue.Empty:
            continue

    if image is None:
        raise RuntimeError(f"Timed out waiting for camera image: {output_path}")

    image.save_to_disk(str(output_path))
    return output_path


def cleanup(*actors: carla.Actor | None) -> None:
    for actor in actors:
        if actor is None:
            continue
        try:
            actor.destroy()
        except Exception:
            pass


def _ensure_target_world(
    client: carla.Client,
    world: carla.World,
    target_map_substring: str | None,
) -> carla.World:
    if not target_map_substring:
        return world

    current_name = str(world.get_map().name)
    if target_map_substring.lower() in current_name.lower():
        return world

    candidates = [name for name in client.get_available_maps() if target_map_substring.lower() in name.lower()]
    if not candidates:
        raise RuntimeError(
            f"Current map is '{current_name}', and no installed map matches '{target_map_substring}'."
        )
    return client.load_world(candidates[0])


def _reload_for_clear(client: carla.Client) -> carla.World:
    world = client.reload_world(False)
    world.wait_for_tick()
    return world


def _ensure_rendering_enabled(world: carla.World) -> None:
    settings = world.get_settings()
    if bool(getattr(settings, "no_rendering_mode", False)):
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        world.wait_for_tick()
        print("[INFO] Disabled no_rendering_mode so camera captures rendered frames.")


def _weather_signature(weather: carla.WeatherParameters) -> str:
    return (
        f"cloud={weather.cloudiness:.1f} rain={weather.precipitation:.1f} "
        f"sun_alt={weather.sun_altitude_angle:.1f} sun_az={weather.sun_azimuth_angle:.1f} "
        f"fog={weather.fog_density:.1f} wet={weather.wetness:.1f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--timeout", type=float, default=20.0, help="CARLA client timeout (seconds)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for crosswalk_C output",
    )
    parser.add_argument(
        "--target-map",
        default="ucla_v2",
        help="Map substring to enforce (loads matching map only if current world does not match)",
    )
    parser.add_argument(
        "--carla-map-cache",
        type=Path,
        default=Path(CARLA_MAP_CACHE_DEFAULT),
        help="CARLA map cache pickle used by plot_trajectories_on_map (c### labels refer to this index space)",
    )
    parser.add_argument(
        "--target-lane-cids",
        default=TARGET_LANE_CIDS_DEFAULT,
        help="Comma-separated CARLA line ids (c###) for the crosswalk span (default: c359..c364)",
    )
    parser.add_argument(
        "--bottom-anchor-cid",
        type=int,
        default=BOTTOM_ANCHOR_CID_DEFAULT,
        help="Anchor c### id defining the 'bottom' side of target lanes (default: c333)",
    )
    parser.add_argument(
        "--top-anchor-cid",
        type=int,
        default=TOP_ANCHOR_CID_DEFAULT,
        help="Anchor c### id defining the opposite side of target lanes (default: c728)",
    )
    parser.add_argument(
        "--bottom-inset-m",
        type=float,
        default=CROSSWALK_BOTTOM_INSET_M,
        help="Meters to move from the c333-side lane ends toward c728 before drawing crosswalk",
    )
    parser.add_argument(
        "--side-margin-lane-ratio",
        type=float,
        default=CROSSWALK_SIDE_MARGIN_LANE_RATIO,
        help="Extra margin on both sides as a fraction of median lane width",
    )
    parser.add_argument(
        "--waypoint-sample-step-m",
        type=float,
        default=1.0,
        help="Sampling step for lane clouds when resolving road_id/lane_id geometry",
    )
    parser.add_argument(
        "--stripe-step",
        type=float,
        default=CROSSWALK_C_STRIPE_STEP_OVERRIDE_M,
        help="Optional manual lateral spacing between C stripes (<=0 uses auto gap rule)",
    )
    parser.add_argument(
        "--trim-outermost-per-side",
        type=int,
        default=STRIPE_TRIM_OUTERMOST_PER_SIDE_DEFAULT,
        help="Number of outermost stripes to remove from each side before drawing.",
    )
    parser.add_argument(
        "--high-z-prune-count",
        type=int,
        default=HIGH_Z_PRUNE_COUNT_DEFAULT,
        help="Remove this many highest-z stripes (typically median-overlap stripes).",
    )
    parser.add_argument(
        "--surface-z-cache-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON file for persistent surface-z cache (speeds up repeated raycast/hybrid runs)."
        ),
    )
    parser.add_argument(
        "--z-approaches",
        type=str,
        default=Z_APPROACHES_DEFAULT,
        help=(
            "Comma-separated z snapping modes for crosswalk C experiments. "
            "Supported: waypoint,ground_projection,raycast,hybrid_raycast"
        ),
    )
    parser.add_argument(
        "--raycast-only",
        action="store_true",
        help="Override --z-approaches and run only the fast single-raycast mode.",
    )
    parser.add_argument(
        "--hybrid-raycast-only",
        action="store_true",
        help="Override --z-approaches and run only hybrid_raycast mode.",
    )
    parser.add_argument(
        "--cid-z-boosts",
        type=str,
        default=CROSSWALK_CID_Z_BOOSTS_DEFAULT,
        help=(
            "Per-cid max z adjustment in meters, format '361:0.12,359:-0.02'. "
            "Each cid applies a 3-step gradient over its 3 nearest stripes."
        ),
    )
    parser.add_argument(
        "--enable-cid-z-gradients",
        action="store_true",
        help="Enable cid-based z-gradient adjustments. Disabled by default.",
    )
    parser.add_argument(
        "--save-approach-views",
        dest="save_approach_views",
        action="store_true",
        help="Also save driver-like approach view images around the crosswalk",
    )
    parser.add_argument(
        "--no-save-approach-views",
        dest="save_approach_views",
        action="store_false",
        help="Save only the overhead image",
    )
    parser.set_defaults(save_approach_views=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    client = carla.Client(args.host, int(args.port))
    client.set_timeout(float(args.timeout))

    world = client.get_world()
    world = _ensure_target_world(client, world, args.target_map)
    baseline_weather = world.get_weather()
    # Clear stale persistent debug drawings from prior runs.
    world = _reload_for_clear(client)
    world = _ensure_target_world(client, world, args.target_map)
    world.set_weather(baseline_weather)
    _ensure_rendering_enabled(world)
    world.wait_for_tick()
    ws = world.get_settings()
    print(
        f"[INFO] world_map={world.get_map().name} sync={bool(getattr(ws, 'synchronous_mode', False))} "
        f"no_render={bool(getattr(ws, 'no_rendering_mode', False))}"
    )
    print(f"[INFO] baseline_weather: {_weather_signature(baseline_weather)}")

    target_lane_cids = _parse_csv_ints(str(args.target_lane_cids))
    if not target_lane_cids:
        raise RuntimeError("No target lane cids provided via --target-lane-cids.")
    trim_outermost_per_side = max(0, int(args.trim_outermost_per_side))
    high_z_prune_count = max(0, int(args.high_z_prune_count))
    cid_z_boosts = _parse_cid_z_boosts(str(args.cid_z_boosts))
    if not bool(args.enable_cid_z_gradients):
        cid_z_boosts = {}
    surface_z_cache_file = Path(args.surface_z_cache_file) if args.surface_z_cache_file else None
    surface_z_cache: dict[tuple, float | None] = load_surface_z_cache(surface_z_cache_file)
    supported_z_modes = {"waypoint", "ground_projection", "raycast", "hybrid_raycast"}
    if bool(args.raycast_only) and bool(args.hybrid_raycast_only):
        raise RuntimeError("Use at most one of --raycast-only and --hybrid-raycast-only.")
    if bool(args.hybrid_raycast_only):
        z_modes = ["hybrid_raycast"]
    elif bool(args.raycast_only):
        z_modes = ["raycast"]
    else:
        z_approaches_raw = _parse_csv_tokens(str(args.z_approaches))
        if not z_approaches_raw:
            z_approaches_raw = _parse_csv_tokens(Z_APPROACHES_DEFAULT)
        z_modes = []
        seen_z_modes: set[str] = set()
        for token in z_approaches_raw:
            mode = _normalize_z_mode(token)
            if mode not in supported_z_modes:
                raise RuntimeError(
                    f"Unsupported z mode '{token}' -> '{mode}'. "
                    f"Supported modes: {sorted(supported_z_modes)}"
                )
            if mode in seen_z_modes:
                continue
            seen_z_modes.add(mode)
            z_modes.append(mode)

    plan = _build_crosswalk_plan_from_lane_cids(
        world=world,
        carla_map_cache=Path(args.carla_map_cache),
        target_lane_cids=target_lane_cids,
        bottom_anchor_cid=int(args.bottom_anchor_cid),
        top_anchor_cid=int(args.top_anchor_cid),
        bottom_inset_m=float(args.bottom_inset_m),
        side_margin_lane_ratio=float(args.side_margin_lane_ratio),
        waypoint_sample_step_m=float(args.waypoint_sample_step_m),
    )
    ref = plan.ref
    target_specs_msg = ", ".join(
        f"c{s.cid}(r{s.road_id},l{s.lane_id})" for s in plan.target_lane_specs
    )
    print(
        f"[INFO] carla_map_cache={Path(args.carla_map_cache)} "
        f"target_lanes={target_specs_msg} "
        f"bottom_anchor=c{plan.bottom_anchor.cid} "
        f"top_anchor=c{plan.top_anchor.cid}"
    )
    print(
        f"[INFO] crosswalk_center=({plan.center.x:.2f}, {plan.center.y:.2f}, {plan.center.z:.2f}) "
        f"yaw={ref.transform.rotation.yaw:.2f} lane_width_med={plan.median_lane_width:.2f} "
        f"span=({plan.start_lateral:.2f},{plan.end_lateral:.2f}) lane_count={plan.lane_count}"
    )
    print(f"[INFO] z_approaches={','.join(z_modes)}")
    print(
        f"[INFO] stripe_filter: trim_outermost_per_side={trim_outermost_per_side} "
        f"high_z_prune_count={high_z_prune_count}"
    )
    if surface_z_cache_file is not None:
        print(
            f"[INFO] surface_z_cache_file={surface_z_cache_file} "
            f"entries_loaded={len(surface_z_cache)}"
        )
    if bool(args.raycast_only):
        print("[INFO] raycast_only=true (single mode run)")
    if bool(args.hybrid_raycast_only):
        print("[INFO] hybrid_raycast_only=true (single mode run)")
    if cid_z_boosts:
        print(
            "[INFO] cid_z_boosts="
            + ", ".join(f"c{cid}:{boost:+.3f}m" for cid, boost in sorted(cid_z_boosts.items()))
        )
    else:
        print("[INFO] cid_z_gradients=disabled (using per-line dynamic ground z only)")

    experiments: list[tuple[str, str, Path]] = []
    multi_mode = len(z_modes) > 1
    for z_mode in z_modes:
        safe_mode = z_mode.replace("-", "_")
        filename = f"crosswalk_C_{safe_mode}.png" if multi_mode else "crosswalk_C.png"
        experiments.append((f"C/{safe_mode}", z_mode, Path(args.output_dir) / filename))

    for exp_idx, (label, z_mode, output_path) in enumerate(experiments):
        crosswalk_center = _offset_location(
            base=plan.center,
            forward=ref.forward,
            lateral=ref.perpendicular,
            up_m=CROSSWALK_DRAW_Z_OFFSET_M,
        )

        generate_crosswalk_C(
            world=world,
            plan=plan,
            stripe_step_override=float(args.stripe_step),
            cid_z_boosts=cid_z_boosts,
            z_mode=z_mode,
            z_cache=surface_z_cache,
            trim_outermost_per_side=trim_outermost_per_side,
            high_z_prune_count=high_z_prune_count,
        )
        views = _build_camera_views(
            world=world,
            plan=plan,
            crosswalk_center=crosswalk_center,
            include_approach_views=bool(args.save_approach_views),
        )
        for view in views:
            camera = None
            try:
                camera = spawn_camera_transform(
                    world,
                    transform=view.transform,
                    fov_deg=float(view.fov_deg),
                )
                cam_tf = view.transform
                out_path = (
                    output_path
                    if view.name == "overhead"
                    else output_path.with_name(f"{output_path.stem}_{view.name}{output_path.suffix}")
                )
                print(
                    f"[{label}] view={view.name} "
                    f"camera=({cam_tf.location.x:.2f}, {cam_tf.location.y:.2f}, {cam_tf.location.z:.2f}) "
                    f"yaw={cam_tf.rotation.yaw:.2f} pitch={cam_tf.rotation.pitch:.2f} fov={view.fov_deg:.1f}"
                )
                projected = _project_world_to_image(
                    cam_tf,
                    crosswalk_center,
                    width=IMAGE_W,
                    height=IMAGE_H,
                    fov_deg=float(view.fov_deg),
                )
                if projected is None:
                    print(f"[{label}] view={view.name} crosswalk_center is behind camera")
                else:
                    u, v, depth = projected
                    in_frame = 0.0 <= u < float(IMAGE_W) and 0.0 <= v < float(IMAGE_H)
                    print(
                        f"[{label}] view={view.name} "
                        f"crosswalk_center=({crosswalk_center.x:.2f}, {crosswalk_center.y:.2f}, {crosswalk_center.z:.2f}) "
                        f"proj=({u:.1f}, {v:.1f}) depth={depth:.2f} in_frame={in_frame}"
                    )
                saved = capture_image(world, camera, out_path)
                print(f"[{label}] saved {saved}")
            finally:
                cleanup(camera)

        if exp_idx < len(experiments) - 1:
            world = _reload_for_clear(client)
            world = _ensure_target_world(client, world, args.target_map)
            world.set_weather(baseline_weather)
            _ensure_rendering_enabled(world)
            world.wait_for_tick()

    if surface_z_cache_file is not None:
        save_surface_z_cache(surface_z_cache_file, surface_z_cache)
        print(
            f"[INFO] surface_z_cache_saved={surface_z_cache_file} "
            f"entries={len(surface_z_cache)}"
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
