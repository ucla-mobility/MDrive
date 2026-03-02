#!/usr/bin/env python3
"""
Shared CARLA validation helpers for route bundles.

This module is intentionally entrypoint-agnostic so both debug and audit tooling can
reuse the same validation contract and simulation checks.
"""

from __future__ import annotations

import inspect
import json
import logging
import math
import re
import shutil
import sys
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCENARIO_GENERATOR_ROOT = REPO_ROOT / "scenario_generator"
if str(SCENARIO_GENERATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(SCENARIO_GENERATOR_ROOT))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def parse_offset_csv(raw: Any, *, default: Sequence[float]) -> List[float]:
    if raw is None:
        return [float(v) for v in default]
    if isinstance(raw, (list, tuple)):
        vals = [_safe_float(x, float("nan")) for x in raw]
    else:
        vals = [_safe_float(tok.strip(), float("nan")) for tok in str(raw).split(",")]
    out = [float(v) for v in vals if math.isfinite(v)]
    if not out:
        return [float(v) for v in default]
    # Keep order but de-duplicate.
    dedup: List[float] = []
    seen = set()
    for v in out:
        key = round(float(v), 6)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(float(v))
    return dedup


def build_spawn_offset_candidates(
    xy_offsets: Sequence[float],
    z_offsets: Sequence[float],
    max_attempts: int,
) -> List[Tuple[float, float, float]]:
    candidates: List[Tuple[float, float, float]] = []
    for dx in xy_offsets:
        for dy in xy_offsets:
            for dz in z_offsets:
                candidates.append((float(dx), float(dy), float(dz)))
    candidates.sort(key=lambda t: (abs(t[0]) + abs(t[1]) + abs(t[2]), abs(t[2]), abs(t[0]) + abs(t[1]), t[0], t[1], t[2]))
    if max_attempts > 0:
        candidates = candidates[:max_attempts]
    if not candidates:
        candidates = [(0.0, 0.0, 0.0)]
    return candidates


def _load_actor_manifest(routes_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    manifest_path = routes_dir / "actors_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _all_manifest_entries(manifest: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []
    for role, entries in manifest.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict):
                out.append((str(role), entry))
    return out


def _parse_route_xml_waypoints(route_path: Path) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    root = ET.parse(route_path).getroot()
    route = root.find("route")
    if route is None:
        raise ValueError(f"Missing <route> in {route_path}")

    attrs = {str(k): str(v) for k, v in route.attrib.items()}
    waypoints: List[Dict[str, Any]] = []
    for wp in route.iter("waypoint"):
        item = {
            "x": _safe_float(wp.attrib.get("x")),
            "y": _safe_float(wp.attrib.get("y")),
            "z": _safe_float(wp.attrib.get("z")),
            "yaw": _safe_float(wp.attrib.get("yaw")),
            "pitch": _safe_float(wp.attrib.get("pitch")),
            "roll": _safe_float(wp.attrib.get("roll")),
        }
        if "time" in wp.attrib:
            item["time"] = _safe_float(wp.attrib.get("time"))
        elif "t" in wp.attrib:
            item["time"] = _safe_float(wp.attrib.get("t"))
        if "speed" in wp.attrib:
            item["speed"] = _safe_float(wp.attrib.get("speed"))
        waypoints.append(item)
    return attrs, waypoints


def _write_first_waypoint_offset(route_path: Path, dx: float, dy: float, dz: float) -> bool:
    try:
        tree = ET.parse(route_path)
    except Exception:
        return False
    root = tree.getroot()
    route = root.find("route")
    if route is None:
        return False
    waypoints = list(route.iter("waypoint"))
    if not waypoints:
        return False

    for waypoint in waypoints:
        x = _safe_float(waypoint.attrib.get("x")) + float(dx)
        y = _safe_float(waypoint.attrib.get("y")) + float(dy)
        z = _safe_float(waypoint.attrib.get("z")) + float(dz)
        waypoint.set("x", f"{x:.3f}")
        waypoint.set("y", f"{y:.3f}")
        waypoint.set("z", f"{z:.3f}")

    backup = route_path.with_suffix(".xml.orig")
    if not backup.exists():
        shutil.copy(route_path, backup)
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    tree.write(str(route_path), encoding="utf-8", xml_declaration=True)
    return True


def _advance_first_waypoint_along_route(route_path: Path, steps: int = 1) -> Tuple[bool, Optional[float], int]:
    """
    Move the first waypoint to a later waypoint along the same trajectory.
    Returns (applied, distance_m, used_index).
    """
    try:
        tree = ET.parse(route_path)
    except Exception:
        return False, None, 0
    root = tree.getroot()
    route = root.find("route")
    if route is None:
        return False, None, 0

    waypoints = list(route.findall("waypoint"))
    if len(waypoints) < 3:
        return False, None, 0

    src = waypoints[0]
    max_shift = len(waypoints) - 2
    dst_idx = max(1, min(max_shift, int(steps)))
    dst = waypoints[dst_idx]

    x0 = _safe_float(src.attrib.get("x"), float("nan"))
    y0 = _safe_float(src.attrib.get("y"), float("nan"))
    z0 = _safe_float(src.attrib.get("z"), float("nan"))
    x1 = _safe_float(dst.attrib.get("x"), float("nan"))
    y1 = _safe_float(dst.attrib.get("y"), float("nan"))
    z1 = _safe_float(dst.attrib.get("z"), float("nan"))
    if not (math.isfinite(x1) and math.isfinite(y1) and math.isfinite(z1)):
        return False, None, 0

    for idx in range(dst_idx):
        try:
            route.remove(waypoints[idx])
        except Exception:
            continue

    backup = route_path.with_suffix(".xml.orig")
    if not backup.exists():
        shutil.copy(route_path, backup)
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    tree.write(str(route_path), encoding="utf-8", xml_declaration=True)

    dist_m: Optional[float] = None
    if all(math.isfinite(v) for v in (x0, y0, z0)):
        dist_m = float(math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2))
    return True, dist_m, int(dst_idx)


def _route_has_expected_ego_suffix(name: str) -> bool:
    return bool(re.search(r"_(\d+)\.xml$", name))


def validate_xml_manifest_contract(routes_dir: Path) -> Dict[str, Any]:
    """
    Validate route bundle contract used by setup_scenario_from_zip + leaderboard parser.
    """
    routes_dir = Path(routes_dir)
    errors: List[str] = []
    warnings: List[str] = []
    checked_files: List[str] = []
    counts: Dict[str, int] = {}

    manifest_path = routes_dir / "actors_manifest.json"
    if not manifest_path.exists():
        errors.append("missing_actors_manifest")
        return {
            "ok": False,
            "errors": errors,
            "warnings": warnings,
            "manifest_path": str(manifest_path),
            "counts": counts,
            "checked_files": checked_files,
        }

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"actors_manifest_invalid_json:{exc}")
        return {
            "ok": False,
            "errors": errors,
            "warnings": warnings,
            "manifest_path": str(manifest_path),
            "counts": counts,
            "checked_files": checked_files,
        }

    if not isinstance(manifest, dict):
        errors.append("actors_manifest_not_object")
        return {
            "ok": False,
            "errors": errors,
            "warnings": warnings,
            "manifest_path": str(manifest_path),
            "counts": counts,
            "checked_files": checked_files,
        }

    from tools.setup_scenario_from_zip import parse_route_metadata

    all_entries = _all_manifest_entries(manifest)
    if not all_entries:
        errors.append("actors_manifest_no_entries")

    for role, entry in all_entries:
        counts[role] = counts.get(role, 0) + 1
        rel = str(entry.get("file", "")).strip()
        if not rel:
            errors.append(f"manifest_entry_missing_file:{role}")
            continue
        xml_path = (routes_dir / rel).resolve()
        checked_files.append(rel)

        if not xml_path.exists():
            errors.append(f"manifest_file_missing:{rel}")
            continue
        if xml_path.suffix.lower() != ".xml":
            errors.append(f"manifest_file_not_xml:{rel}")
            continue

        try:
            xml_bytes = xml_path.read_bytes()
            route_id, town, xml_role = parse_route_metadata(xml_bytes)
        except Exception as exc:
            errors.append(f"route_metadata_parse_failed:{rel}:{exc}")
            continue

        # schema consistency checks
        if not route_id:
            errors.append(f"route_id_missing:{rel}")
        if not town:
            errors.append(f"route_town_missing:{rel}")
        if not xml_role:
            errors.append(f"route_role_missing:{rel}")

        attrs, waypoints = _parse_route_xml_waypoints(xml_path)
        if not waypoints:
            errors.append(f"route_waypoints_missing:{rel}")
        if "id" not in attrs or "town" not in attrs or "role" not in attrs:
            errors.append(f"route_attrs_missing_required:{rel}")

        if role == "ego" and not _route_has_expected_ego_suffix(xml_path.name):
            errors.append(f"ego_filename_suffix_invalid:{xml_path.name}")

        manifest_role = str(entry.get("kind", role)).strip().lower() or role.lower()
        if manifest_role in {"walker", "cyclist", "bike"}:
            # accepted aliases used by route parser
            pass
        elif xml_role.lower() != manifest_role:
            warnings.append(
                f"role_mismatch:{rel}:manifest={manifest_role}:xml={xml_role.lower()}"
            )

    # sanity check: at least one ego route must exist
    if counts.get("ego", 0) <= 0:
        errors.append("manifest_missing_ego_entries")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "manifest_path": str(manifest_path),
        "counts": counts,
        "checked_files": checked_files,
    }


def _load_carla_module() -> Any:
    try:
        import carla  # type: ignore

        return carla
    except Exception:
        pass

    carla_root = REPO_ROOT / "carla912"
    dist_dir = carla_root / "PythonAPI" / "carla" / "dist"
    eggs = sorted(dist_dir.glob("carla-*.egg"))
    if not eggs:
        raise FileNotFoundError(f"No CARLA egg under {dist_dir}")
    py3 = [egg for egg in eggs if "-py3" in egg.name]
    egg_path = str(py3[0] if py3 else eggs[0])
    if egg_path not in sys.path:
        sys.path.append(egg_path)
    import carla  # type: ignore  # noqa: E401

    return carla


def _load_grp_classes() -> Tuple[Any, Any]:
    # Ensure the CARLA PythonAPI "agents" package is importable.
    agents_parent = REPO_ROOT / "carla912" / "PythonAPI" / "carla"
    agents_parent_str = str(agents_parent)
    if agents_parent.is_dir() and agents_parent_str not in sys.path:
        sys.path.insert(0, agents_parent_str)

    # Compatible import style used across the repo.
    from agents.navigation.global_route_planner import GlobalRoutePlanner  # type: ignore
    from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO  # type: ignore

    return GlobalRoutePlanner, GlobalRoutePlannerDAO


def _make_grp(carla_map: Any, sampling_resolution: float = 2.0) -> Any:
    GlobalRoutePlanner, GlobalRoutePlannerDAO = _load_grp_classes()
    init_params = list(inspect.signature(GlobalRoutePlanner.__init__).parameters.values())[1:]
    try:
        if len(init_params) >= 2 and init_params[0].name != "dao":
            grp = GlobalRoutePlanner(carla_map, sampling_resolution)
        else:
            grp = GlobalRoutePlanner(GlobalRoutePlannerDAO(carla_map, sampling_resolution))
    except TypeError:
        try:
            grp = GlobalRoutePlanner(carla_map, sampling_resolution)
        except Exception:
            grp = GlobalRoutePlanner(GlobalRoutePlannerDAO(carla_map, sampling_resolution))
    if hasattr(grp, "setup"):
        grp.setup()
    return grp


def _load_world_for_town(
    carla_module: Any,
    host: str,
    port: int,
    desired_town: Optional[str],
    timeout_s: float,
) -> Tuple[Any, Any]:
    client = carla_module.Client(host, int(port))
    client.set_timeout(float(timeout_s))
    world = client.get_world()
    current_map = world.get_map().name
    current_base = current_map.split("/")[-1] if "/" in current_map else current_map
    if desired_town and desired_town.lower() not in current_base.lower():
        world = client.load_world(desired_town)
    return client, world


def route_feasibility_grp(
    routes_dir: Path,
    carla_host: str,
    carla_port: int,
    timeout_s: float = 15.0,
    sampling_resolution: float = 2.0,
) -> Dict[str, Any]:
    routes_dir = Path(routes_dir)
    manifest = _load_actor_manifest(routes_dir)
    ego_entries = [entry for role, entry in _all_manifest_entries(manifest) if role == "ego"]
    if not ego_entries:
        return {
            "ok": False,
            "error": "no_ego_entries",
            "checked": 0,
            "reachable": 0,
            "unreachable": [],
        }

    desired_town = str(ego_entries[0].get("town", "")).strip() or None
    try:
        carla_module = _load_carla_module()
        client, world = _load_world_for_town(carla_module, carla_host, int(carla_port), desired_town, timeout_s)
        _ = client
    except Exception as exc:
        return {
            "ok": False,
            "error": f"carla_connect_failed:{exc}",
            "checked": len(ego_entries),
            "reachable": 0,
            "unreachable": [str(e.get("file")) for e in ego_entries],
        }

    carla_map = world.get_map()
    try:
        grp = _make_grp(carla_map, sampling_resolution=sampling_resolution)
    except Exception as exc:
        return {
            "ok": False,
            "error": f"grp_init_failed:{exc}",
            "checked": len(ego_entries),
            "reachable": 0,
            "unreachable": [str(e.get("file")) for e in ego_entries],
        }

    unreachable: List[str] = []
    reachable = 0
    start_end_deltas: List[float] = []
    for entry in ego_entries:
        rel = str(entry.get("file", "")).strip()
        path = routes_dir / rel
        try:
            _, waypoints = _parse_route_xml_waypoints(path)
        except Exception:
            unreachable.append(rel)
            continue
        if len(waypoints) < 2:
            unreachable.append(rel)
            continue
        s = waypoints[0]
        e = waypoints[-1]
        start_loc = carla_module.Location(x=float(s["x"]), y=float(s["y"]), z=float(s["z"]))
        end_loc = carla_module.Location(x=float(e["x"]), y=float(e["y"]), z=float(e["z"]))
        s_wp = carla_map.get_waypoint(start_loc, project_to_road=True, lane_type=carla_module.LaneType.Driving)
        e_wp = carla_map.get_waypoint(end_loc, project_to_road=True, lane_type=carla_module.LaneType.Driving)
        if s_wp is None or e_wp is None:
            unreachable.append(rel)
            continue
        try:
            route = grp.trace_route(s_wp.transform.location, e_wp.transform.location)
        except Exception:
            route = []
        if not route:
            unreachable.append(rel)
            continue
        reachable += 1
        start_end_deltas.append(float(s_wp.transform.location.distance(start_loc) + e_wp.transform.location.distance(end_loc)))

    return {
        "ok": len(unreachable) == 0,
        "error": "" if len(unreachable) == 0 else "unreachable_ego_routes",
        "checked": len(ego_entries),
        "reachable": reachable,
        "unreachable": unreachable,
        "avg_snap_error_m": round(sum(start_end_deltas) / len(start_end_deltas), 3) if start_end_deltas else None,
    }


def validate_spawn(world: Any, expected_actor_ids: List[int]) -> Tuple[bool, str]:
    """Compatibility helper used by audit/debug flows."""
    for aid in expected_actor_ids:
        actor = world.get_actor(aid)
        if actor is None:
            return False, f"missing_actor_{aid}"
        if not actor.is_alive:
            return False, f"dead_actor_{aid}"
    return True, ""


def _actor_radius(actor: Any, margin: float = 0.2) -> float:
    try:
        bb = actor.bounding_box
        return max(max(float(bb.extent.x), float(bb.extent.y)) + float(margin), 0.4)
    except Exception:
        return 1.5


def _actor_velocity(actor: Any) -> Any:
    try:
        return actor.get_velocity()
    except Exception:
        return None


def _project_location(loc: Any, vel: Any, dt: float, carla_module: Any) -> Any:
    if vel is None:
        return loc
    return carla_module.Location(
        x=loc.x + vel.x * dt,
        y=loc.y + vel.y * dt,
        z=loc.z + vel.z * dt,
    )


def _closing_speed(ego_loc: Any, ego_vel: Any, other_loc: Any, other_vel: Any) -> float:
    if ego_vel is None or other_vel is None:
        return 0.0
    dx = ego_loc.x - other_loc.x
    dy = ego_loc.y - other_loc.y
    dz = ego_loc.z - other_loc.z
    mag = math.sqrt(dx * dx + dy * dy + dz * dz)
    if mag < 1e-6:
        return 0.0
    ux, uy, uz = dx / mag, dy / mag, dz / mag
    relx = other_vel.x - ego_vel.x
    rely = other_vel.y - ego_vel.y
    relz = other_vel.z - ego_vel.z
    return relx * ux + rely * uy + relz * uz


def compute_near_miss(
    ego_actor: Any,
    other_actors: List[Any],
    carla_module: Any,
    horizon_s: float,
    step_s: float,
    ttc_thresh: float,
    ttc_severe: float,
    closing_min: float,
    role_lookup: Optional[Dict[int, str]] = None,
    debug_hits: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[bool, bool, float]:
    """Compatibility helper used by audit/debug flows."""
    try:
        ego_loc = ego_actor.get_location()
    except Exception:
        return False, False, float("inf")

    ego_vel = _actor_velocity(ego_actor)
    ego_r = _actor_radius(ego_actor)

    min_ttc = float("inf")
    hit = False
    severe = False

    for other in other_actors:
        try:
            other_id = other.id
        except Exception:
            continue
        if other_id == ego_actor.id:
            continue
        try:
            other_loc = other.get_location()
        except Exception:
            continue

        other_vel = _actor_velocity(other)
        closing = _closing_speed(ego_loc, ego_vel, other_loc, other_vel)
        if closing <= closing_min:
            continue

        other_r = _actor_radius(other)
        r_sum = ego_r + other_r
        steps = max(1, int(horizon_s / max(step_s, 1e-3)))
        for i in range(1, steps + 1):
            t = i * step_s
            p_ego = _project_location(ego_loc, ego_vel, t, carla_module)
            p_other = _project_location(other_loc, other_vel, t, carla_module)
            dist = p_ego.distance(p_other)
            if dist <= r_sum:
                min_ttc = min(min_ttc, t)
                if debug_hits is not None:
                    debug_hits.append(
                        {
                            "ego_id": getattr(ego_actor, "id", None),
                            "other_id": other_id,
                            "other_role": role_lookup.get(other_id) if role_lookup else "",
                            "t": t,
                            "dist": dist,
                            "r_sum": r_sum,
                            "closing": closing,
                            "hit": t <= ttc_thresh,
                            "severe": t <= ttc_severe,
                        }
                    )
                if t <= ttc_thresh:
                    hit = True
                if t <= ttc_severe:
                    severe = True
                break

    return hit, severe, min_ttc


def _parse_route_transforms(route_path: Path, carla_module: Any) -> List[Any]:
    root = ET.parse(route_path).getroot()
    route_nodes = list(root.iter("route"))
    if not route_nodes:
        return []
    out = []
    for wp in route_nodes[0].iter("waypoint"):
        x = _safe_float(wp.attrib.get("x"))
        y = _safe_float(wp.attrib.get("y"))
        z = _safe_float(wp.attrib.get("z"))
        yaw = _safe_float(wp.attrib.get("yaw"))
        pitch = _safe_float(wp.attrib.get("pitch"))
        roll = _safe_float(wp.attrib.get("roll"))
        out.append(
            carla_module.Transform(
                carla_module.Location(x=float(x), y=float(y), z=float(z)),
                carla_module.Rotation(yaw=float(yaw), pitch=float(pitch), roll=float(roll)),
            )
        )
    return out


def _destroy_actors(actors: Iterable[Any]) -> None:
    for actor in actors:
        try:
            if actor is None:
                continue
            if hasattr(actor, "is_alive") and not bool(actor.is_alive):
                # Avoid CARLA warning spam for already-dead actors.
                continue
            actor.destroy()
        except Exception:
            pass


def _cleanup_world_actors(world: Any) -> int:
    """Destroy all vehicles, walkers, and static props left over from previous
    runs.  Returns the number of actors destroyed.  This prevents zombie actors
    (which still occupy physics space) from blocking spawn points."""
    destroyed = 0
    try:
        actors = world.get_actors()
        # Filter to types that our validation spawns.  Avoid destroying
        # infrastructure actors like traffic lights / signs.
        for filt in ["vehicle.*", "walker.*", "static.prop.*"]:
            for actor in actors.filter(filt):
                try:
                    if hasattr(actor, "is_alive") and not bool(actor.is_alive):
                        continue  # Skip already-dead actors to avoid CARLA warning spam.
                    actor.destroy()
                    destroyed += 1
                except Exception:
                    pass
    except Exception:
        pass
    return destroyed


def _pick_blueprint(blueprint_library: Any, role: str, model: Optional[str] = None) -> Any:
    if model:
        try:
            return blueprint_library.find(str(model))
        except Exception:
            pass

    role_l = str(role).lower()
    if role_l in {"pedestrian", "walker"}:
        walkers = [bp for bp in blueprint_library.filter("walker.pedestrian.*")]
        if walkers:
            return walkers[0]
    if role_l in {"bicycle", "bike", "cyclist"}:
        bikes = [bp for bp in blueprint_library.filter("vehicle.diamondback.*")]
        if bikes:
            return bikes[0]
    if role_l in {"static", "static_prop"}:
        props = [bp for bp in blueprint_library.filter("static.prop.*")]
        if props:
            return props[0]

    vehicles = [bp for bp in blueprint_library.filter("vehicle.*")]
    if vehicles:
        return vehicles[0]
    raise RuntimeError("No usable blueprint found")


def _apply_offset_transform(base_tf: Any, dx: float, dy: float, dz: float, carla_module: Any) -> Any:
    return carla_module.Transform(
        carla_module.Location(
            x=float(base_tf.location.x) + float(dx),
            y=float(base_tf.location.y) + float(dy),
            z=float(base_tf.location.z) + float(dz),
        ),
        carla_module.Rotation(
            pitch=float(base_tf.rotation.pitch),
            yaw=float(base_tf.rotation.yaw),
            roll=float(base_tf.rotation.roll),
        ),
    )


def _resolve_ground_z_raycast(world: Any, location: Any) -> Optional[float]:
    """Use CARLA physics raycasts to find the actual terrain mesh height.
    Mirrors the approach in route_scenario.py ``_resolve_ground_z``."""
    if world is None:
        return None
    # Try world.ground_projection first (available in CARLA >=0.9.12)
    ground_projection = getattr(world, "ground_projection", None)
    if callable(ground_projection):
        try:
            probe = type(location)(
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
    # Fallback: cast_ray downward
    cast_ray = getattr(world, "cast_ray", None)
    if callable(cast_ray):
        try:
            start = type(location)(x=location.x, y=location.y, z=location.z + 50.0)
            end = type(location)(x=location.x, y=location.y, z=location.z - 50.0)
            hits = cast_ray(start, end)
            if hits:
                top_z = None
                for hit in hits:
                    hit_loc = getattr(hit, "location", None) or getattr(hit, "point", None)
                    if hit_loc is None:
                        continue
                    z_val = float(hit_loc.z)
                    if top_z is None or z_val > top_z:
                        top_z = z_val
                if top_z is not None:
                    return float(top_z)
        except Exception:
            pass
    return None


def _resolve_map_z(world_map: Any, location: Any, lane_type: Any) -> Optional[float]:
    """Get the road surface Z from the CARLA HD map waypoint projection."""
    if world_map is None:
        return None

    def _get_wp_z(lt: Any) -> Optional[float]:
        try:
            wp = world_map.get_waypoint(location, project_to_road=True, lane_type=lt)
        except Exception:
            wp = None
        if wp is None:
            return None
        try:
            return float(wp.transform.location.z)
        except Exception:
            return None

    z = _get_wp_z(lane_type)
    # Broaden to Any if the specific lane type yielded nothing.
    if z is None:
        try:
            any_type = type(lane_type).Any
            if any_type != lane_type:
                z = _get_wp_z(any_type)
        except Exception:
            pass
    return z


def _select_ground_z(
    world: Any,
    world_map: Any,
    location: Any,
    lane_type: Any,
    *,
    obstacle_clip_m: float = 0.35,
    vehicle_clip_m: float = 0.60,
    prefer_ray: bool = False,
) -> Optional[float]:
    """Pick a robust ground-height estimate.  Mirrors``_select_ground_z``
    in ``route_scenario.py`` so that the validation spawn matches the
    actual runtime spawn behaviour."""
    ground_z = _resolve_ground_z_raycast(world, location)
    map_z = _resolve_map_z(world_map, location, lane_type)

    if ground_z is None and map_z is None:
        return None

    if prefer_ray and ground_z is not None:
        if map_z is not None and float(ground_z) > float(map_z) + obstacle_clip_m:
            return float(map_z)
        return float(ground_z)

    # Conservative fusion for vehicles: avoid clipping into road mesh.
    if map_z is not None and ground_z is not None:
        if float(ground_z) > float(map_z) + vehicle_clip_m:
            return float(map_z)
        return float(max(float(map_z), float(ground_z)))
    if map_z is not None:
        return float(map_z)
    return float(ground_z) if ground_z is not None else None


def _normalize_actor_z(
    transform: Any,
    role: str,
    world: Any,
    world_map: Any,
    carla_module: Any,
) -> Any:
    """Adjust only the Z coordinate of *transform* to the CARLA ground height,
    preserving the authored X/Y position.  This matches the ``normalize_actor_z``
    behaviour in ``run_custom_eval`` / ``route_scenario.py``."""
    role_l = str(role).lower()
    is_walker_like = role_l in {
        "pedestrian", "walker", "bicycle", "bike", "cyclist",
    }
    if is_walker_like or role_l in {"static", "static_prop"}:
        lane_type = carla_module.LaneType.Any
        prefer_ray = is_walker_like
    else:
        lane_type = carla_module.LaneType.Driving
        prefer_ray = False

    target_z = _select_ground_z(
        world, world_map, transform.location, lane_type,
        prefer_ray=prefer_ray,
    )
    if target_z is None:
        return transform

    return carla_module.Transform(
        carla_module.Location(
            x=float(transform.location.x),
            y=float(transform.location.y),
            z=float(target_z),
        ),
        carla_module.Rotation(
            pitch=float(transform.rotation.pitch),
            yaw=float(transform.rotation.yaw),
            roll=float(transform.rotation.roll),
        ),
    )


def _snap_transform_to_road(
    transform: Any,
    role: str,
    world_map: Any,
    carla_module: Any,
) -> Any:
    """Legacy full-snap helper.  Kept for reference but no longer called by the
    spawn path (replaced by ``_normalize_actor_z`` which preserves X/Y)."""
    lane_type = carla_module.LaneType.Driving
    role_l = str(role).lower()
    if role_l in {"pedestrian", "walker", "bicycle", "bike", "cyclist", "static", "static_prop"}:
        lane_type = carla_module.LaneType.Any
    try:
        wp = world_map.get_waypoint(transform.location, project_to_road=True, lane_type=lane_type)
    except Exception:
        wp = None
    if wp is None:
        return transform
    snapped = carla_module.Transform(
        carla_module.Location(
            x=float(wp.transform.location.x),
            y=float(wp.transform.location.y),
            z=float(wp.transform.location.z),
        ),
        carla_module.Rotation(
            pitch=float(transform.rotation.pitch),
            yaw=float(transform.rotation.yaw),
            roll=float(transform.rotation.roll),
        ),
    )
    return snapped


def _role_for_spawn_policy(role_name: str) -> str:
    role_l = str(role_name or "").strip().lower()
    if role_l.startswith("hero_"):
        return "ego"
    if "pedestrian" in role_l or "walker" in role_l:
        return "pedestrian"
    if "bicycle" in role_l or "cyclist" in role_l or "bike" in role_l:
        return "bicycle"
    if "static" in role_l or "prop" in role_l:
        return "static"
    if "npc" in role_l:
        return "npc"
    return role_l or "unknown"


def _spawn_progress_indices(role_name: str, waypoint_count: int) -> List[int]:
    role = _role_for_spawn_policy(role_name)
    if waypoint_count <= 0:
        return [0]

    if role in {"ego", "npc"}:
        raw = [0, 1, 2, 3, 5, 8, 12, 16]
    elif role in {"pedestrian", "bicycle"}:
        raw = [0, 1, 2, 3, 4, 6]
    else:
        raw = [0]

    out: List[int] = []
    seen = set()
    for idx in raw:
        clamped = max(0, min(int(waypoint_count) - 1, int(idx)))
        if clamped in seen:
            continue
        seen.add(clamped)
        out.append(clamped)
    if 0 not in seen:
        out.insert(0, 0)
    return out


def _spawn_clearance_m(role_name: str) -> float:
    role = _role_for_spawn_policy(role_name)
    if role == "static":
        # Keep a conservative envelope for static props to reduce overlap
        # with already spawned dynamic actors.
        return 1.6
    # For dynamic actors, rely on CARLA's exact occupancy check.
    return 0.0


def _is_candidate_blocked_by_spawned(
    transform: Any,
    spawned: Sequence[Any],
    min_clearance_m: float,
) -> bool:
    """Check whether *transform* would overlap with an already-spawned actor.

    Uses each spawned actor's bounding-box extents (if available) to compute
    a per-actor clearance envelope instead of a single flat distance.  Falls
    back to *min_clearance_m* when the bounding box cannot be read."""
    for actor in spawned:
        if actor is None:
            continue
        try:
            if hasattr(actor, "is_alive") and not bool(actor.is_alive):
                continue
            loc = actor.get_location()
            if loc is None:
                continue
            # Derive clearance from the actor's actual bounding box so that
            # large vehicles get a bigger exclusion zone than small props.
            clearance = float(min_clearance_m)
            try:
                bb = actor.bounding_box
                clearance = max(
                    clearance,
                    float(max(bb.extent.x, bb.extent.y)) + 0.25,
                )
            except Exception:
                pass
            if clearance <= 0.0:
                continue
            if float(loc.distance(transform.location)) < clearance:
                return True
        except Exception:
            continue
    return False


def _spawn_actor_with_offsets(
    world: Any,
    world_map: Any,
    blueprint: Any,
    role_name: str,
    base_tf: Any,
    route_waypoints: Sequence[Any],
    spawned_actors: Sequence[Any],
    offsets: Sequence[Tuple[float, float, float]],
    carla_module: Any,
) -> Tuple[Optional[Any], Optional[Tuple[float, float, float]], int, Optional[str], int]:
    attempts = 0
    last_error: Optional[str] = None
    last_progress_idx = 0

    if hasattr(blueprint, "has_attribute") and blueprint.has_attribute("role_name"):
        try:
            blueprint.set_attribute("role_name", str(role_name))
        except Exception:
            pass

    progress_indices = _spawn_progress_indices(str(role_name), len(route_waypoints))
    min_clearance = _spawn_clearance_m(str(role_name))

    for wp_idx in progress_indices:
        base_try = base_tf
        if route_waypoints and 0 <= int(wp_idx) < len(route_waypoints):
            base_try = route_waypoints[int(wp_idx)]
        last_progress_idx = int(wp_idx)

        # Normalize the *base* waypoint Z to ground level once per
        # progress index.  The per-candidate dz offset is then added ON TOP
        # of the ground height so the offset grid actually explores
        # different elevations above the road surface.
        base_grounded = _normalize_actor_z(
            base_try, role_name, world, world_map, carla_module,
        )

        for dx, dy, dz in offsets:
            attempts += 1
            tf = _apply_offset_transform(base_grounded, dx, dy, dz, carla_module)
            if _is_candidate_blocked_by_spawned(tf, spawned_actors, min_clearance):
                last_error = f"candidate_blocked_within_{min_clearance:.2f}m"
                continue
            try:
                actor = world.try_spawn_actor(blueprint, tf)
            except Exception as exc:
                actor = None
                last_error = str(exc)
            if actor is not None:
                # --- Post-spawn grounding (mirrors route_scenario.py) ---
                # Now that we have a live actor we can read its bounding box
                # and place it so the bottom of the bbox sits on the ground.
                try:
                    ground_z = _select_ground_z(
                        world, world_map, actor.get_location(),
                        carla_module.LaneType.Any,
                    )
                    if ground_z is not None:
                        bbox = actor.bounding_box
                        # base_offset lifts the actor so its underside
                        # touches the ground rather than its origin.
                        base_offset = float(bbox.extent.z) - float(bbox.location.z)
                        grounded_tf = actor.get_transform()
                        grounded_tf.location.z = float(ground_z) + base_offset
                        actor.set_transform(grounded_tf)
                except Exception:
                    pass  # keep the actor at its spawn position
                return actor, (float(dx), float(dy), float(dz)), attempts, None, int(wp_idx)

    if last_error is None:
        last_error = "try_spawn_actor returned None"
    return None, None, attempts, last_error, int(last_progress_idx)


def _compute_manual_control(actor: Any, waypoints: List[Any], reached_idx: int, desired_speed: float, kp_speed: float = 0.4, kp_steer: float = 0.6, lookahead: int = 4) -> Any:
    idx = min(max(0, int(reached_idx)), max(0, len(waypoints) - 1))
    target_idx = min(len(waypoints) - 1, idx + max(1, int(lookahead)))
    target_tf = waypoints[target_idx]
    loc = actor.get_transform().location
    vec = target_tf.location - loc
    desired_yaw = math.atan2(vec.y, vec.x)
    current_yaw = math.radians(actor.get_transform().rotation.yaw)
    yaw_err = desired_yaw - current_yaw
    while yaw_err > math.pi:
        yaw_err -= 2 * math.pi
    while yaw_err < -math.pi:
        yaw_err += 2 * math.pi

    steer = max(-1.0, min(1.0, kp_steer * yaw_err))
    vel = _actor_velocity(actor)
    speed = 0.0
    if vel is not None:
        speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

    desired = max(0.0, float(desired_speed))
    if speed < desired:
        throttle = max(0.0, min(1.0, kp_speed * (desired - speed)))
        brake = 0.0
    else:
        throttle = 0.0
        brake = max(0.0, min(1.0, 0.2 * (speed - desired)))

    return steer, throttle, brake


def _run_constant_velocity_baseline(
    routes_dir: Path,
    carla_module: Any,
    client: Any,
    world: Any,
    spawn_offsets: Sequence[Tuple[float, float, float]],
    timeout_s: float,
    require_risk: bool,
) -> Dict[str, Any]:
    def _default_checks(
        *,
        spawn_ok: bool = False,
        risk_ok: bool = False,
        route_follow_ok: bool = False,
    ) -> Dict[str, bool]:
        return {
            "spawn_all_actors": bool(spawn_ok),
            "constant_trajectory_risk_check": bool(risk_ok),
            "baseline_route_follow": bool(route_follow_ok),
        }

    def _default_metrics(
        *,
        spawn_expected: int,
        spawn_actual: int,
    ) -> Dict[str, Any]:
        return {
            "spawn_expected": int(spawn_expected),
            "spawn_actual": int(spawn_actual),
            "min_ttc_s": None,
            "near_miss": False,
            "route_completion_min": 0.0,
            "driving_score_min": 0.0,
        }

    manifest = _load_actor_manifest(routes_dir)
    if not manifest:
        return {
            "ok": False,
            "reason": "missing_manifest",
            "checks": _default_checks(),
            "metrics": _default_metrics(spawn_expected=0, spawn_actual=0),
            "spawn_failed_entries": [],
        }

    # Destroy any leftover actors from previous validation runs that might
    # still occupy physics space and block spawn points (zombie actors).
    n_cleaned = _cleanup_world_actors(world)
    if n_cleaned > 0:
        logger.debug("Pre-spawn cleanup: destroyed %d leftover actors", n_cleaned)
        # Give CARLA a couple of ticks to fully remove the destroyed actors.
        try:
            for _ in range(3):
                world.tick()
        except Exception:
            pass

    blueprint_library = world.get_blueprint_library()
    world_map = world.get_map()

    spawned: List[Any] = []
    expected_actor_ids: List[int] = []
    role_lookup: Dict[int, str] = {}
    ego_actors: List[Any] = []
    ego_waypoints: Dict[int, List[Any]] = {}
    ego_entry_speeds: Dict[int, float] = {}
    reached: Dict[int, int] = {}

    spawn_failures: List[Dict[str, Any]] = []
    spawn_repairs: List[Dict[str, Any]] = []

    # Spawn dynamic participants first; static props are attempted last so
    # they don't unnecessarily block critical vehicle spawns.
    spawn_plan = [
        ("ego", manifest.get("ego", [])),
        ("npc", manifest.get("npc", [])),
        ("bicycle", manifest.get("bicycle", [])),
        ("pedestrian", manifest.get("pedestrian", [])),
        ("static", manifest.get("static", [])),
    ]

    try:
        for role_group, entries in spawn_plan:
            if not isinstance(entries, list):
                continue
            for idx, entry in enumerate(entries):
                rel = str(entry.get("file", "")).strip()
                if not rel:
                    spawn_failures.append({"role": role_group, "index": idx, "file": rel, "reason": "missing_manifest_file"})
                    continue
                route_path = (routes_dir / rel).resolve()
                if not route_path.exists():
                    spawn_failures.append({"role": role_group, "index": idx, "file": rel, "reason": "missing_route_xml"})
                    continue

                waypoints = _parse_route_transforms(route_path, carla_module)
                if not waypoints:
                    spawn_failures.append({"role": role_group, "index": idx, "file": rel, "reason": "empty_waypoints"})
                    continue

                base_tf = waypoints[0]
                model = entry.get("model")
                blueprint = _pick_blueprint(blueprint_library, role_group, model)
                role_name = str(entry.get("kind") or role_group)
                if role_group == "ego":
                    role_name = f"hero_{len(ego_actors)}"

                actor, used_offset, attempts, error_text, used_wp_idx = _spawn_actor_with_offsets(
                    world,
                    world_map,
                    blueprint,
                    role_name,
                    base_tf,
                    waypoints,
                    spawned,
                    spawn_offsets,
                    carla_module,
                )
                if actor is None:
                    spawn_failures.append(
                        {
                            "role": role_group,
                            "index": idx,
                            "file": rel,
                            "reason": f"spawn_failed:{error_text or 'unknown'}",
                            "attempts": attempts,
                            "last_route_progress_index": int(used_wp_idx),
                        }
                    )
                    continue

                spawned.append(actor)
                expected_actor_ids.append(actor.id)
                role_lookup[actor.id] = role_name

                # --- Post-spawn adjustments (mirrors route_scenario.py) ---
                if role_group in ("static", "static_prop"):
                    # 1) Bounding-box Z correction: place the bottom of the
                    #    bounding box flush with the ground surface.
                    try:
                        bbox = actor.bounding_box
                        ground_z = _select_ground_z(
                            world, world_map, actor.get_location(),
                            carla_module.LaneType.Any,
                        )
                        if ground_z is not None:
                            target_z = (
                                float(ground_z)
                                - float(bbox.location.z)
                                + float(bbox.extent.z)
                            )
                            tf_adj = actor.get_transform()
                            tf_adj.location.z = target_z
                            actor.set_transform(tf_adj)
                    except Exception:
                        pass
                    # 2) Disable physics so static props don't fall / slide.
                    try:
                        actor.set_simulate_physics(False)
                    except Exception:
                        pass

                if used_offset is not None and (used_offset != (0.0, 0.0, 0.0) or int(used_wp_idx) > 0):
                    repair_item: Dict[str, Any] = {
                        "file": rel,
                        "role": role_group,
                        "offset": [float(used_offset[0]), float(used_offset[1]), float(used_offset[2])],
                    }
                    if int(used_wp_idx) > 0:
                        repair_item["route_shift_wp_index"] = int(used_wp_idx)
                        try:
                            repair_item["route_shift_m"] = round(
                                float(waypoints[0].location.distance(waypoints[int(used_wp_idx)].location)),
                                3,
                            )
                        except Exception:
                            repair_item["route_shift_m"] = None
                    spawn_repairs.append(repair_item)

                if role_group == "ego":
                    ego_idx = len(ego_actors)
                    ego_actors.append(actor)
                    ego_waypoints[ego_idx] = waypoints
                    reached[ego_idx] = 0
                    entry_speed = entry.get("speed")
                    ego_entry_speeds[ego_idx] = max(1.0, _safe_float(entry_speed, 8.0))

        spawn_expected = len(_all_manifest_entries(manifest))
        spawn_actual = len(spawned)

        spawn_ok, spawn_reason = validate_spawn(world, expected_actor_ids)
        if not spawn_ok:
            return {
                "ok": False,
                "reason": f"spawn_validation:{spawn_reason}",
                "checks": _default_checks(),
                "metrics": _default_metrics(spawn_expected=spawn_expected, spawn_actual=spawn_actual),
                "spawn_failed_entries": spawn_failures,
                "spawn_repairs": spawn_repairs,
            }
        if spawn_actual < spawn_expected:
            return {
                "ok": False,
                "reason": "spawn_expected_actual_mismatch",
                "checks": _default_checks(),
                "metrics": _default_metrics(spawn_expected=spawn_expected, spawn_actual=spawn_actual),
                "spawn_failed_entries": spawn_failures,
                "spawn_repairs": spawn_repairs,
            }

        if not ego_actors:
            return {
                "ok": False,
                "reason": "no_ego_spawned",
                "checks": _default_checks(),
                "metrics": _default_metrics(spawn_expected=spawn_expected, spawn_actual=spawn_actual),
                "spawn_failed_entries": spawn_failures,
                "spawn_repairs": spawn_repairs,
            }

        warmup_ticks = 12
        for _ in range(warmup_ticks):
            for ego_idx, ego in enumerate(ego_actors):
                try:
                    steer, throttle, brake = _compute_manual_control(
                        ego,
                        ego_waypoints[ego_idx],
                        reached[ego_idx],
                        ego_entry_speeds[ego_idx],
                    )
                    ego.apply_control(carla_module.VehicleControl(throttle=throttle, steer=steer, brake=brake))
                except Exception:
                    pass
            world.tick()

        max_ticks = max(60, int(timeout_s / 0.05))
        waypoint_thresh = 2.5
        near_miss = False
        severe = False
        min_ttc = float("inf")
        hits_debug: List[Dict[str, Any]] = []

        for _tick in range(max_ticks):
            for ego_idx, ego in enumerate(ego_actors):
                if not getattr(ego, "is_alive", True):
                    continue
                try:
                    steer, throttle, brake = _compute_manual_control(
                        ego,
                        ego_waypoints[ego_idx],
                        reached[ego_idx],
                        ego_entry_speeds[ego_idx],
                    )
                    ego.apply_control(carla_module.VehicleControl(throttle=throttle, steer=steer, brake=brake))
                except Exception:
                    continue

            world.tick()

            for ego_idx, ego in enumerate(ego_actors):
                if not getattr(ego, "is_alive", True):
                    continue
                waypoints = ego_waypoints[ego_idx]
                if not waypoints:
                    continue
                idx = min(reached[ego_idx], len(waypoints) - 1)
                try:
                    if ego.get_location().distance(waypoints[idx].location) < waypoint_thresh:
                        reached[ego_idx] = min(len(waypoints) - 1, idx + 1)
                except Exception:
                    pass

                others = [a for a in spawned if getattr(a, "id", None) != getattr(ego, "id", None)]
                hit, sev, ttc = compute_near_miss(
                    ego,
                    others,
                    carla_module,
                    horizon_s=1.8,
                    step_s=0.15,
                    ttc_thresh=0.9,
                    ttc_severe=0.65,
                    closing_min=1.0,
                    role_lookup=role_lookup,
                    debug_hits=hits_debug,
                )
                if hit:
                    near_miss = True
                if sev:
                    severe = True
                min_ttc = min(min_ttc, ttc)

            if all(reached[i] >= max(0, len(ego_waypoints[i]) - 1) for i in reached):
                break

        rc_vals: List[float] = []
        ds_vals: List[float] = []
        for ego_idx in sorted(ego_waypoints.keys()):
            total = max(1, len(ego_waypoints[ego_idx]) - 1)
            rc = float(reached.get(ego_idx, 0)) / float(total)
            ds = rc
            rc_vals.append(rc)
            ds_vals.append(ds)

        route_completion_min = min(rc_vals) if rc_vals else 0.0
        driving_score_min = min(ds_vals) if ds_vals else 0.0

        risk_ok = bool(near_miss) if require_risk else True
        route_follow_ok = route_completion_min >= 0.95 and driving_score_min >= 0.95

        checks = {
            "spawn_all_actors": bool(spawn_ok and spawn_actual >= spawn_expected),
            "constant_trajectory_risk_check": bool(risk_ok),
            "baseline_route_follow": bool(route_follow_ok),
        }

        failed_reasons: List[str] = []
        if not checks["spawn_all_actors"]:
            failed_reasons.append("spawn_all_actors_failed")
        if not checks["constant_trajectory_risk_check"]:
            failed_reasons.append("risk_check_failed")
        if not checks["baseline_route_follow"]:
            failed_reasons.append("route_follow_failed")

        return {
            "ok": all(checks.values()),
            "reason": "|".join(failed_reasons) if failed_reasons else "",
            "checks": checks,
            "metrics": {
                "spawn_expected": spawn_expected,
                "spawn_actual": spawn_actual,
                "min_ttc_s": None if not math.isfinite(min_ttc) else round(float(min_ttc), 4),
                "near_miss": bool(near_miss or severe),
                "route_completion_min": round(float(route_completion_min), 4),
                "driving_score_min": round(float(driving_score_min), 4),
            },
            "spawn_failed_entries": spawn_failures,
            "spawn_repairs": spawn_repairs,
            "near_miss_hits": hits_debug,
        }
    finally:
        _destroy_actors(spawned)


@dataclass
class CarlaValidationConfig:
    routes_dir: Path
    carla_host: str = "127.0.0.1"
    carla_port: int = 3000
    carla_validation_timeout: float = 180.0
    carla_require_risk: bool = True
    carla_align_before_validate: bool = False
    carla_repair_max_attempts: int = 2
    carla_repair_xy_offsets: Sequence[float] = (0.0, 0.25, -0.25, 0.5, -0.5, 1.0, -1.0)
    carla_repair_z_offsets: Sequence[float] = (0.0, 0.2, -0.2, 0.5, -0.5, 1.0)


def _align_routes(routes_dir: Path, town: str, host: str, port: int) -> Tuple[bool, str]:
    try:
        try:
            from pipeline.step_07_route_alignment.main import align_routes_in_directory  # type: ignore
        except Exception:
            from scenario_generator.pipeline.step_07_route_alignment.main import align_routes_in_directory  # type: ignore
    except Exception as exc:
        return False, f"alignment_import_failed:{exc}"

    try:
        align_routes_in_directory(
            routes_dir=Path(routes_dir),
            town=str(town),
            carla_host=str(host),
            carla_port=int(port),
            backup=True,
            sampling_resolution=2.0,
        )
    except Exception as exc:
        return False, f"alignment_failed:{exc}"
    return True, ""


def _persistent_spawn_repair(
    routes_dir: Path,
    failed_entries: Sequence[Dict[str, Any]],
    chosen_offset: Tuple[float, float, float],
    attempt_index: int,
) -> List[Dict[str, Any]]:
    repairs: List[Dict[str, Any]] = []
    dx, dy, dz = chosen_offset
    for failure in failed_entries:
        rel = str(failure.get("file", "")).strip()
        if not rel:
            continue
        role = str(failure.get("role", "")).strip().lower()
        route_path = (routes_dir / rel).resolve()
        if not route_path.exists():
            continue
        moved_along_route = False
        move_dist_m: Optional[float] = None
        move_idx = 0
        if role in {"ego", "npc", "pedestrian", "walker", "bicycle", "bike", "cyclist"}:
            last_progress = _safe_int(failure.get("last_route_progress_index"), 0)
            progress_steps = max(1, int(last_progress) + 1)
            escalation_steps = max(0, int(attempt_index) - 1) * 2
            steps = progress_steps + escalation_steps
            moved_along_route, move_dist_m, move_idx = _advance_first_waypoint_along_route(route_path, steps=steps)

        offset_applied = _write_first_waypoint_offset(route_path, dx, dy, dz)
        if moved_along_route or offset_applied:
            detail_parts: List[str] = [f"actor {rel}"]
            if moved_along_route:
                if move_dist_m is not None and math.isfinite(move_dist_m):
                    detail_parts.append(f"trimmed_to_wp[{move_idx}] ({move_dist_m:.2f}m)")
                else:
                    detail_parts.append(f"trimmed_to_wp[{move_idx}]")
            if offset_applied:
                detail_parts.append(f"route_offset ({dx:+.2f},{dy:+.2f},{dz:+.2f})")
            repairs.append(
                {
                    "strategy": "spawn_offset",
                    "applied": True,
                    "details": " ".join(detail_parts),
                }
            )
    return repairs


def _choose_repair_offset(
    all_offsets: Sequence[Tuple[float, float, float]],
    attempt_index: int,
) -> Tuple[float, float, float]:
    # Skip zero offset where possible for persistent repair attempts.
    # Prefer planar nudges first because vertical-only lifts rarely resolve
    # actor-vs-actor overlap on drivable lanes.
    non_zero = [o for o in all_offsets if any(abs(v) > 1e-6 for v in o)]
    pool = non_zero if non_zero else list(all_offsets)
    if not pool:
        return (0.0, 0.0, 0.0)

    def _priority_key(o: Tuple[float, float, float]) -> Tuple[Any, ...]:
        dx, dy, dz = o
        planar = abs(float(dx)) + abs(float(dy))
        zabs = abs(float(dz))
        has_planar = planar > 1e-6
        # bucket 0: pure XY shifts, bucket 1: XY+Z shifts, bucket 2: Z-only
        if has_planar and zabs <= 1e-6:
            bucket = 0
        elif has_planar:
            bucket = 1
        else:
            bucket = 2
        return (
            bucket,
            round(planar, 6),
            round(zabs, 6),
            round(abs(float(dx)), 6),
            round(abs(float(dy)), 6),
            round(abs(float(dz)), 6),
            round(float(dx), 6),
            round(float(dy), 6),
            round(float(dz), 6),
        )

    ordered = sorted(pool, key=_priority_key)
    idx = max(0, int(attempt_index) - 1) % len(ordered)
    return ordered[idx]


def run_final_carla_validation(config: CarlaValidationConfig) -> Dict[str, Any]:
    routes_dir = Path(config.routes_dir)
    xy_offsets = list(config.carla_repair_xy_offsets)
    z_offsets = list(config.carla_repair_z_offsets)
    all_spawn_offsets = build_spawn_offset_candidates(
        xy_offsets,
        z_offsets,
        max_attempts=max(1, len(xy_offsets) * len(xy_offsets) * len(z_offsets)),
    )

    # Ensure we can infer town from manifest if available.
    manifest = _load_actor_manifest(routes_dir)
    desired_town = None
    for role in ("ego", "npc", "pedestrian", "bicycle", "static"):
        entries = manifest.get(role) if isinstance(manifest, dict) else None
        if isinstance(entries, list) and entries:
            desired_town = str(entries[0].get("town", "")).strip() or None
            if desired_town:
                break

    repairs: List[Dict[str, Any]] = []
    attempted_alignment = False

    if config.carla_align_before_validate and desired_town:
        ok, err = _align_routes(routes_dir, desired_town, config.carla_host, int(config.carla_port))
        attempted_alignment = attempted_alignment or ok
        repairs.append(
            {
                "attempt": 0,
                "strategy": "pre_align_routes",
                "applied": bool(ok),
                "details": "aligned routes before validation" if ok else err,
            }
        )

    last_contract: Dict[str, Any] = {}
    last_feasibility: Dict[str, Any] = {}
    last_baseline: Dict[str, Any] = {}
    failure_reason = ""

    for attempt in range(1, max(1, int(config.carla_repair_max_attempts)) + 2):
        last_contract = validate_xml_manifest_contract(routes_dir)
        if not last_contract.get("ok", False):
            failure_reason = "xml_manifest_contract_failed"
            if attempt <= int(config.carla_repair_max_attempts):
                # Mild global z-lift fallback for malformed/noisy spawns.
                chosen = _choose_repair_offset(all_spawn_offsets, attempt)
                repaired = _persistent_spawn_repair(
                    routes_dir,
                    [{"file": str(e.get("file"))} for _role, e in _all_manifest_entries(_load_actor_manifest(routes_dir))],
                    chosen,
                    attempt,
                )
                repairs.append(
                    {
                        "attempt": attempt,
                        "strategy": "global_spawn_offset_repair",
                        "applied": bool(repaired),
                        "details": f"applied to {len(repaired)} entries with offset ({chosen[0]:+.2f},{chosen[1]:+.2f},{chosen[2]:+.2f})",
                    }
                )
                continue
            break

        last_feasibility = route_feasibility_grp(
            routes_dir,
            carla_host=config.carla_host,
            carla_port=int(config.carla_port),
            timeout_s=min(20.0, max(5.0, float(config.carla_validation_timeout) * 0.2)),
            sampling_resolution=2.0,
        )

        try:
            carla_module = _load_carla_module()
            client, world = _load_world_for_town(
                carla_module,
                config.carla_host,
                int(config.carla_port),
                desired_town,
                timeout_s=max(5.0, float(config.carla_validation_timeout) * 0.35),
            )
        except Exception as exc:
            failure_reason = f"carla_connect_failed:{exc}"
            last_baseline = {
                "ok": False,
                "reason": failure_reason,
                "checks": {
                    "spawn_all_actors": False,
                    "constant_trajectory_risk_check": False,
                    "baseline_route_follow": False,
                },
                "metrics": {
                    "spawn_expected": 0,
                    "spawn_actual": 0,
                    "min_ttc_s": None,
                    "near_miss": False,
                    "route_completion_min": 0.0,
                    "driving_score_min": 0.0,
                },
            }
            break

        original_settings = world.get_settings()
        try:
            settings = world.get_settings()
            settings.synchronous_mode = True
            if getattr(settings, "fixed_delta_seconds", None) is None:
                settings.fixed_delta_seconds = 0.05
            else:
                settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)

            last_baseline = _run_constant_velocity_baseline(
                routes_dir,
                carla_module,
                client,
                world,
                spawn_offsets=all_spawn_offsets,
                timeout_s=float(config.carla_validation_timeout),
                require_risk=bool(config.carla_require_risk),
            )
        finally:
            try:
                world.apply_settings(original_settings)
            except Exception:
                pass
            # Reload the world to fully reset CARLA's internal state (actor
            # registry, physics, etc.).  Without this, zombie actor debris
            # accumulates across validation runs and crashes CARLA after
            # ~10-15 consecutive scenarios.
            try:
                current_map = world.get_map().name
                town_name = current_map.split("/")[-1] if "/" in current_map else current_map
                client.load_world(town_name)
            except Exception as exc:
                logger.warning("load_world reset failed: %s", exc)

        checks = {
            "xml_manifest_contract": bool(last_contract.get("ok", False)),
            "route_feasibility_grp": bool(last_feasibility.get("ok", False)),
            "spawn_all_actors": bool((last_baseline.get("checks") or {}).get("spawn_all_actors", False)),
            "constant_trajectory_risk_check": bool((last_baseline.get("checks") or {}).get("constant_trajectory_risk_check", False)),
            "baseline_route_follow": bool((last_baseline.get("checks") or {}).get("baseline_route_follow", False)),
        }
        hard_gate_checks = {
            "xml_manifest_contract": bool(checks["xml_manifest_contract"]),
            "route_feasibility_grp": bool(checks["route_feasibility_grp"]),
            "spawn_all_actors": bool(checks["spawn_all_actors"]),
        }
        soft_quality_signals = {
            "constant_trajectory_risk_check": bool(checks["constant_trajectory_risk_check"]),
            "baseline_route_follow": bool(checks["baseline_route_follow"]),
        }
        soft_failures = [name for name, ok in soft_quality_signals.items() if not ok]

        if all(hard_gate_checks.values()):
            metrics = dict(last_baseline.get("metrics") or {})
            metrics.setdefault("spawn_expected", _safe_int(metrics.get("spawn_expected"), 0))
            metrics.setdefault("spawn_actual", _safe_int(metrics.get("spawn_actual"), 0))
            payload: Dict[str, Any] = {
                "passed": True,
                "gate_mode": "hard",
                "checks": checks,
                "hard_gate_checks": hard_gate_checks,
                "soft_quality_signals": soft_quality_signals,
                "soft_failures": soft_failures,
                "metrics": metrics,
                "repairs": repairs,
                "spawn_failed_entries": list(last_baseline.get("spawn_failed_entries") or []),
                "spawn_repairs": list(last_baseline.get("spawn_repairs") or []),
                "near_miss_hits": list(last_baseline.get("near_miss_hits") or []),
                "final_routes_dir": str(routes_dir),
                "failure_reason": None,
                "contract": last_contract,
                "route_feasibility": last_feasibility,
            }
            if soft_failures:
                payload["quality_note"] = (
                    "Scenario passed hard CARLA validation but failed one or more soft quality signals."
                )
            return payload

        failure_bits: List[str] = []
        for key, ok in hard_gate_checks.items():
            if not ok:
                failure_bits.append(key)
        failure_reason = "|".join(failure_bits) if failure_bits else (str(last_baseline.get("reason")) or "carla_validation_failed")

        # attempt repairs if budget left
        if attempt <= int(config.carla_repair_max_attempts):
            spawn_failed = list(last_baseline.get("spawn_failed_entries") or [])
            if spawn_failed:
                chosen = _choose_repair_offset(all_spawn_offsets, attempt)
                applied = _persistent_spawn_repair(routes_dir, spawn_failed, chosen, attempt)
                repairs.append(
                    {
                        "attempt": attempt,
                        "strategy": "spawn_offset",
                        "applied": bool(applied),
                        "details": applied[0]["details"] if applied else f"spawn repair attempted with offset ({chosen[0]:+.2f},{chosen[1]:+.2f},{chosen[2]:+.2f})",
                    }
                )
                continue

            # Alignment repair for route-feasibility failures.
            if not checks["route_feasibility_grp"] and desired_town and not attempted_alignment:
                ok, err = _align_routes(routes_dir, desired_town, config.carla_host, int(config.carla_port))
                attempted_alignment = attempted_alignment or ok
                repairs.append(
                    {
                        "attempt": attempt,
                        "strategy": "align_routes",
                        "applied": bool(ok),
                        "details": "aligned routes for feasibility repair" if ok else err,
                    }
                )
                if ok:
                    continue

            # fallback z-lift on all entries
            chosen = _choose_repair_offset(all_spawn_offsets, attempt)
            all_entries = [{"file": str(e.get("file"))} for _role, e in _all_manifest_entries(_load_actor_manifest(routes_dir))]
            applied = _persistent_spawn_repair(routes_dir, all_entries, chosen)
            repairs.append(
                {
                    "attempt": attempt,
                    "strategy": "global_spawn_offset",
                    "applied": bool(applied),
                    "details": f"applied to {len(applied)} route(s) with offset ({chosen[0]:+.2f},{chosen[1]:+.2f},{chosen[2]:+.2f})",
                }
            )
            continue

        break

    checks = {
        "xml_manifest_contract": bool(last_contract.get("ok", False)),
        "route_feasibility_grp": bool(last_feasibility.get("ok", False)),
        "spawn_all_actors": bool((last_baseline.get("checks") or {}).get("spawn_all_actors", False)),
        "constant_trajectory_risk_check": bool((last_baseline.get("checks") or {}).get("constant_trajectory_risk_check", False)),
        "baseline_route_follow": bool((last_baseline.get("checks") or {}).get("baseline_route_follow", False)),
    }
    hard_gate_checks = {
        "xml_manifest_contract": bool(checks["xml_manifest_contract"]),
        "route_feasibility_grp": bool(checks["route_feasibility_grp"]),
        "spawn_all_actors": bool(checks["spawn_all_actors"]),
    }
    soft_quality_signals = {
        "constant_trajectory_risk_check": bool(checks["constant_trajectory_risk_check"]),
        "baseline_route_follow": bool(checks["baseline_route_follow"]),
    }
    soft_failures = [name for name, ok in soft_quality_signals.items() if not ok]
    metrics = dict(last_baseline.get("metrics") or {})
    metrics.setdefault("spawn_expected", _safe_int(metrics.get("spawn_expected"), 0))
    metrics.setdefault("spawn_actual", _safe_int(metrics.get("spawn_actual"), 0))

    payload = {
        "passed": False,
        "gate_mode": "hard",
        "checks": checks,
        "hard_gate_checks": hard_gate_checks,
        "soft_quality_signals": soft_quality_signals,
        "soft_failures": soft_failures,
        "metrics": metrics,
        "repairs": repairs,
        "spawn_failed_entries": list(last_baseline.get("spawn_failed_entries") or []),
        "spawn_repairs": list(last_baseline.get("spawn_repairs") or []),
        "near_miss_hits": list(last_baseline.get("near_miss_hits") or []),
        "final_routes_dir": str(routes_dir),
        "failure_reason": failure_reason or "carla_validation_failed",
        "contract": last_contract,
        "route_feasibility": last_feasibility,
    }
    if soft_failures:
        payload["quality_note"] = "Soft quality signals failed during baseline rollout."
    return payload


def write_carla_validation_report(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
