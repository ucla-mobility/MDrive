"""Generate one high-quality BEV figure per scenario across all 4 buckets.

Strategy:
  - Reuses ``tools.build_scenario_database`` for route resolution: prefers an
    existing ``point_coordinates.json`` under
    ``results/results_driving_custom/baseline/codriving/<bucket>/<scenario_id>/``,
    falls back to ``route_alignment.align_route`` for interdrive (offline).
  - Each ego is placed at ~30% arc-length along its dense route. If a candidate
    pose is within ``--min-spacing`` of any other ego or any obstacle (npc,
    walker, bicycle, static), we slide the ego forward/backward along its
    route in 5% steps within ``[5%, 75%]`` until clearance is found.
  - Obstacles are spawned at their XML's first waypoint (their start pose).
  - Scenarios are grouped by town so we ``client.load_world(town)`` only once
    per town. Per scenario we spawn obstacles+egos, snap a top-down camera,
    save the image, and destroy all spawned actors.

Run with the ``colmdrivermarco2`` env (carla + agents.navigation importable):

  /data/miniconda3/envs/colmdrivermarco2/bin/python \
      tools/generate_scenario_bev_figures.py \
      --port 2000 --out figures/scenario_bevs

Outputs ``<bucket>_<scenario_id_with_/-replaced-by-_>.png``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from PIL import Image

REPO = Path("/data2/marco/CoLMDriver")
SCENARIOSET = REPO / "scenarioset"
INTERDRIVE_FALLBACK_ROOTS = (
    SCENARIOSET / "interdrive",
    REPO / "v2xpnp" / "interdrive",
)

# CARLA egg + agents/navigation must be importable (colmdrivermarco2 env).
CARLA_PY = REPO / "carla912" / "PythonAPI" / "carla"
CARLA_EGG = CARLA_PY / "dist" / "carla-0.9.12-py3.7-linux-x86_64.egg"
sys.path.insert(0, str(CARLA_PY))
sys.path.insert(0, str(CARLA_EGG))
sys.path.insert(0, str(REPO / "tools"))

import carla  # noqa: E402

# Reuse the canonical route-resolution machinery.
import build_scenario_database as bsd  # noqa: E402


# ─────────────────────────────── Discovery ──────────────────────────────────

def discover_scenarios() -> list[dict]:
    """Discover scenarios across all 4 buckets from the live scenarioset/.

    Mirrors ``bsd.discover_scenarios`` but uses ``scenarioset/interdrive``
    (the canonical post-reorg path), which the upstream module's stale
    ``INTERDRIVE`` constant does not point at.
    """
    scenarios = []

    interaction_root = SCENARIOSET / "interaction"
    if interaction_root.exists():
        for cat_dir in sorted(p for p in interaction_root.iterdir() if p.is_dir()):
            for sc_dir in sorted(p for p in cat_dir.iterdir() if p.is_dir()):
                if "_partial_" in sc_dir.name:
                    continue
                scenarios.append({
                    "bucket": "interaction",
                    "scenario_id": f"{cat_dir.name}/{sc_dir.name}",
                    "scenario_dir": sc_dir,
                    "category": cat_dir.name,
                })

    od_root = SCENARIOSET / "precrash"
    if od_root.exists():
        for sc_dir in sorted(p for p in od_root.iterdir() if p.is_dir()):
            if "_partial_" in sc_dir.name:
                continue
            scenarios.append({
                "bucket": "precrash",
                "scenario_id": sc_dir.name,
                "scenario_dir": sc_dir,
                "category": None,
            })

    v2_root = SCENARIOSET / "v2xpnp"
    if v2_root.exists():
        for sc_dir in sorted(p for p in v2_root.iterdir() if p.is_dir()):
            scenarios.append({
                "bucket": "v2xpnp",
                "scenario_id": sc_dir.name,
                "scenario_dir": sc_dir,
                "category": None,
            })

    # Interdrive lives under either ``scenarioset/interdrive/`` or
    # ``v2xpnp/interdrive/`` depending on the repo state — accept either.
    for candidate in (SCENARIOSET / "interdrive", REPO / "v2xpnp" / "interdrive"):
        if candidate.exists():
            for sc_dir in sorted(p for p in candidate.iterdir() if p.is_dir()):
                scenarios.append({
                    "bucket": "interdrive",
                    "scenario_id": sc_dir.name,
                    "scenario_dir": sc_dir,
                    "category": bsd._interdrive_type_tag(sc_dir.name),
                })
            break
    return scenarios


# ───────────────── Ego routes (reuse baseline JSON or offline) ──────────────

def resolve_ego_routes(scen: dict, ego_xmls: list[tuple[int, Path]]) -> dict[int, list[dict]]:
    """Return ``{ego_index: [{x,y,z,yaw}, ...]}`` of dense route points."""
    routes: dict[int, list[dict]] = {}
    json_path = bsd.find_existing_point_coords_json(scen["bucket"], scen["scenario_id"])
    if json_path is not None:
        with open(json_path) as f:
            data = json.load(f)
        for r in data.get("ego_routes", []):
            idx = int(r["ego_index"])
            routes[idx] = [
                {
                    "x": float(p["x"]),
                    "y": float(p["y"]),
                    "z": float(p.get("z", 0.0)),
                    "yaw": float(p["yaw"]),
                }
                for p in r["points"]
            ]
        if routes:
            return routes

    for idx, xml in ego_xmls:
        try:
            dense, _town = bsd.reproduce_route_offline(xml)
        except Exception as exc:
            print(f"    [WARN] offline reproduction failed for {xml}: {exc}")
            continue
        if dense:
            routes[idx] = [
                {
                    "x": float(p["x"]),
                    "y": float(p["y"]),
                    "z": float(p.get("z", 0.0)),
                    "yaw": float(p.get("yaw", 0.0)),
                }
                for p in dense
            ]
    return routes


# ─────────────────────────────── Obstacle parsing ────────────────────────────

def _first_waypoint(xml_path: Path) -> Optional[dict]:
    try:
        root = ET.parse(str(xml_path)).getroot()
    except Exception:
        return None
    for wp in root.iter("waypoint"):
        try:
            return {
                "x": float(wp.get("x", "0")),
                "y": float(wp.get("y", "0")),
                "z": float(wp.get("z", "0")),
                "yaw": float(wp.get("yaw", "0")),
            }
        except (TypeError, ValueError):
            continue
    return None


def _all_waypoints(xml_path: Path) -> list[dict]:
    try:
        root = ET.parse(str(xml_path)).getroot()
    except Exception:
        return []
    out: list[dict] = []
    for wp in root.iter("waypoint"):
        try:
            out.append({
                "x": float(wp.get("x", "0")),
                "y": float(wp.get("y", "0")),
                "z": float(wp.get("z", "0")),
                "yaw": float(wp.get("yaw", "0")),
            })
        except (TypeError, ValueError):
            continue
    return out


def _route_attrs(xml_path: Path) -> dict:
    try:
        root = ET.parse(str(xml_path)).getroot()
    except Exception:
        return {}
    route = root.find("route")
    if route is None:
        return {}
    return dict(route.attrib)


def collect_obstacles(scen: dict) -> list[dict]:
    """Return list of ``{kind, x, y, z, yaw, model}`` for every non-ego actor.

    For interdrive there are no obstacles; for v2xpnp/interaction we read
    ``actors_manifest.json``; for precrash we walk the scenario dir.
    """
    bucket = scen["bucket"]
    sd: Path = scen["scenario_dir"]
    if bucket == "interdrive":
        return []

    out: list[dict] = []
    manifest_path = sd / "actors_manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}
        for kind_key, kind_norm in (
            ("npc", "npc"),
            ("pedestrian", "pedestrian"),
            ("walker", "pedestrian"),
            ("bicycle", "bicycle"),
            ("static", "static"),
        ):
            for entry in manifest.get(kind_key, []):
                xml_rel = entry.get("file")
                if not xml_rel:
                    continue
                xml = sd / xml_rel
                wp = _first_waypoint(xml)
                if wp is None:
                    continue
                wp.update({
                    "kind": kind_norm,
                    "model": entry.get("model", ""),
                    "polyline": _all_waypoints(xml),
                    "xml": xml.name,
                })
                out.append(wp)
        return out

    # precrash fallback: walk every XML, classify by role attribute.
    for xml in sorted(sd.glob("*.xml")):
        if "_REPLAY" in xml.name:
            continue
        attrs = _route_attrs(xml)
        role = (attrs.get("role") or "").lower()
        if role in ("", "ego"):
            continue
        wp = _first_waypoint(xml)
        if wp is None:
            continue
        kind = {
            "static": "static",
            "npc": "npc", "vehicle": "npc",
            "pedestrian": "pedestrian", "walker": "pedestrian",
            "bicycle": "bicycle",
        }.get(role, "static")
        wp.update({
            "kind": kind,
            "model": attrs.get("model", ""),
            "polyline": _all_waypoints(xml),
        })
        out.append(wp)
    return out


# ────────────────────────── Route arc-length placement ──────────────────────

def _polyline_pose_at_fraction(poly: list[dict], frac: float,
                               yaw_window_m: float = 6.0) -> Optional[dict]:
    """Linear-interpolate (x, y, z) at *frac* of cumulative arc length along
    a generic polyline (NPC / walker / bicycle path). Yaw is derived from a
    *windowed* travel direction (chord across ``yaw_window_m`` metres
    centred on *frac*) so brief detection jitter or one-frame backwards GT
    poses don't flip the spawned actor's heading.
    """
    if not poly or len(poly) < 2:
        return None
    frac = float(max(0.0, min(1.0, frac)))
    cum = [0.0]
    for a, b in zip(poly, poly[1:]):
        cum.append(cum[-1] + math.hypot(b["x"] - a["x"], b["y"] - a["y"]))
    total = cum[-1]
    if total <= 1e-3:
        return None
    target = frac * total

    # Linear interpolate xyz at any arc-length.
    def _xyz_at(arc: float) -> tuple[float, float, float]:
        for j in range(1, len(cum)):
            if cum[j] >= arc:
                seg_lo, seg_hi = cum[j - 1], cum[j]
                t = (arc - seg_lo) / max(seg_hi - seg_lo, 1e-6)
                a = poly[j - 1]
                b = poly[j]
                return (
                    a["x"] + t * (b["x"] - a["x"]),
                    a["y"] + t * (b["y"] - a["y"]),
                    a.get("z", 0.0) + t * (b.get("z", 0.0) - a.get("z", 0.0)),
                )
        p = poly[-1]
        return (p["x"], p["y"], p.get("z", 0.0))

    x, y, z = _xyz_at(target)

    # Windowed yaw: chord direction over ~yaw_window_m centred on target.
    lo = max(0.0, target - yaw_window_m * 0.5)
    hi = min(total, target + yaw_window_m * 0.5)
    if hi - lo < 0.5:
        # Trajectory shorter than window — use whole thing.
        lo, hi = 0.0, total
    lx, ly, _ = _xyz_at(lo)
    hx, hy, _ = _xyz_at(hi)
    if math.hypot(hx - lx, hy - ly) < 0.1:
        # Stationary — fall back to local segment direction at target.
        for j in range(1, len(cum)):
            if cum[j] >= target:
                a, b = poly[j - 1], poly[j]
                yaw = math.degrees(math.atan2(b["y"] - a["y"], b["x"] - a["x"]))
                return {"x": x, "y": y, "z": z, "yaw": yaw}
        a, b = poly[-2], poly[-1]
        yaw = math.degrees(math.atan2(b["y"] - a["y"], b["x"] - a["x"]))
        return {"x": x, "y": y, "z": z, "yaw": yaw}
    yaw = math.degrees(math.atan2(hy - ly, hx - lx))
    return {"x": x, "y": y, "z": z, "yaw": yaw}


def _route_pose_at_progress(route: list[dict], frac: float) -> Optional[dict]:
    """Linear-interpolate (x,y,z) at *frac* of cumulative arc length along route.

    Yaw is computed from local route direction so spawned vehicles face forward
    along their dense path (the XML yaw of point 0 is preserved at exactly 0%).
    """
    if not route:
        return None
    if len(route) == 1:
        p = route[0]
        return {**p}
    frac = float(max(0.0, min(1.0, frac)))
    cum = [0.0]
    for a, b in zip(route, route[1:]):
        cum.append(cum[-1] + math.hypot(b["x"] - a["x"], b["y"] - a["y"]))
    total = cum[-1]
    if total <= 1e-3:
        p = route[0]
        return {**p}
    target = frac * total
    # locate segment
    for i in range(1, len(cum)):
        if cum[i] >= target:
            seg_lo = cum[i - 1]
            seg_hi = cum[i]
            seg_len = max(seg_hi - seg_lo, 1e-6)
            t = (target - seg_lo) / seg_len
            a = route[i - 1]
            b = route[i]
            x = a["x"] + t * (b["x"] - a["x"])
            y = a["y"] + t * (b["y"] - a["y"])
            z = a["z"] + t * (b["z"] - a["z"])
            yaw = math.degrees(math.atan2(b["y"] - a["y"], b["x"] - a["x"]))
            return {"x": x, "y": y, "z": z, "yaw": yaw}
    p = route[-1]
    nxt = route[-2]
    yaw = math.degrees(math.atan2(p["y"] - nxt["y"], p["x"] - nxt["x"]))
    return {"x": p["x"], "y": p["y"], "z": p["z"], "yaw": yaw}


def _too_close(p: dict, others: Iterable[dict], min_dist: float) -> bool:
    for q in others:
        if math.hypot(p["x"] - q["x"], p["y"] - q["y"]) < min_dist:
            return True
    return False


def _route_arc_length(route: list[dict]) -> tuple[list[float], float]:
    cum = [0.0]
    for a, b in zip(route, route[1:]):
        cum.append(cum[-1] + math.hypot(b["x"] - a["x"], b["y"] - a["y"]))
    return cum, cum[-1]


def _interaction_arc(
    route_self: list[dict],
    other_routes: list[list[dict]],
    threshold_m: float = 8.0,
) -> Optional[float]:
    """Return arc-length along *route_self* where it's closest to any
    *other_routes* polyline, if that distance is < *threshold_m*. Else None.
    """
    if not route_self or not other_routes:
        return None
    cum_self, _ = _route_arc_length(route_self)
    best_arc: Optional[float] = None
    best_dist = float("inf")
    for i, p in enumerate(route_self):
        for other in other_routes:
            for q in other:
                d = math.hypot(p["x"] - q["x"], p["y"] - q["y"])
                if d < best_dist:
                    best_dist = d
                    best_arc = cum_self[i]
    if best_dist > threshold_m:
        return None
    return best_arc


def _arc_to_obstacle(route: list[dict], obstacles: list[dict],
                     threshold_m: float = 12.0) -> Optional[float]:
    """Arc-length along *route* where it's closest to any *obstacle* (within
    ``threshold_m``). Returns None if no obstacle is close enough."""
    if not route or not obstacles:
        return None
    cum, _ = _route_arc_length(route)
    best_arc: Optional[float] = None
    best_dist = float("inf")
    for i, p in enumerate(route):
        for q in obstacles:
            d = math.hypot(p["x"] - q["x"], p["y"] - q["y"])
            if d < best_dist:
                best_dist = d
                best_arc = cum[i]
    if best_dist > threshold_m:
        return None
    return best_arc


def place_egos(
    routes: dict[int, list[dict]],
    obstacles: list[dict],
    *,
    nominal_frac: float = 0.25,
    min_ego_dist: float = 10.0,
    min_obstacle_dist: float = 6.0,
    scan_step: float = 0.04,
    scan_lo: float = 0.04,
    scan_hi: float = 0.85,
    interaction_threshold_m: float = 8.0,
    pre_interaction_offset_m: float = 6.0,
    obstacle_proximity_m: float = 12.0,
    pre_obstacle_offset_m: float = 8.0,
) -> tuple[dict[int, dict], list[int]]:
    """Interaction- and obstacle-aware, collision-avoiding ego placement.

    Per-ego target arc is computed from (in priority order):
      1. If the ego's route passes within ``obstacle_proximity_m`` of an
         obstacle (npc/walker/bike/static), place the ego
         ``pre_obstacle_offset_m`` *before* that closest-approach point so
         the figure shows the ego *approaching* the obstacle.
      2. Else if the route passes within ``interaction_threshold_m`` of
         another ego's route, place ``pre_interaction_offset_m`` before that
         closest-approach point.
      3. Else stay at ``nominal_frac``.

    All targets then go through a greedy collision check against egos
    placed earlier in this call AND all obstacles, sliding the candidate
    along the route in ``scan_step`` increments until clear.
    """
    placed: dict[int, dict] = {}
    dropped: list[int] = []

    target_fracs: dict[int, float] = {}
    for idx, route in routes.items():
        if not route:
            target_fracs[idx] = nominal_frac
            continue

        # Priority 1: approach obstacle.
        obs_arc = _arc_to_obstacle(route, obstacles,
                                   threshold_m=obstacle_proximity_m)
        if obs_arc is not None:
            _, total = _route_arc_length(route)
            target = max(scan_lo, (obs_arc - pre_obstacle_offset_m) / max(total, 1e-3))
            target_fracs[idx] = float(min(target, scan_hi))
            continue

        # Priority 2: interaction with other egos.
        others = [r for j, r in routes.items() if j != idx and r]
        arc = _interaction_arc(route, others, threshold_m=interaction_threshold_m)
        if arc is not None:
            _, total = _route_arc_length(route)
            target = max(scan_lo, (arc - pre_interaction_offset_m) / max(total, 1e-3))
            target_fracs[idx] = float(min(target, scan_hi))
            continue

        # Priority 3: nominal.
        target_fracs[idx] = nominal_frac

    for idx in sorted(routes.keys()):
        route = routes[idx]
        if not route:
            dropped.append(idx)
            continue
        prior_egos = list(placed.values())
        nom = target_fracs[idx]

        candidate_fracs = [nom]
        n_steps = int(round(max(nom - scan_lo, scan_hi - nom) / scan_step)) + 1
        for k in range(1, n_steps + 1):
            for off in (+k * scan_step, -k * scan_step):
                f = nom + off
                if scan_lo - 1e-6 <= f <= scan_hi + 1e-6:
                    candidate_fracs.append(f)

        # Two-tier collision check: a relatively wide gap between egos
        # (so multiple egos on the same lane don't render bumper-to-bumper)
        # and a smaller gap to obstacles (cars *should* be close to the
        # obstacle they're approaching, just not on top of it).
        chosen: Optional[dict] = None
        for frac in candidate_fracs:
            pose = _route_pose_at_progress(route, frac)
            if pose is None:
                continue
            if _too_close(pose, prior_egos, min_ego_dist):
                continue
            if _too_close(pose, obstacles, min_obstacle_dist):
                continue
            chosen = {**pose, "frac": frac}
            break
        if chosen is None:
            dropped.append(idx)
            continue
        placed[idx] = chosen
    return placed, dropped


# ───────────────────── Per-ego colour palette ───────────────────────────────

# Vivid, paper-friendly hues. Each ego gets a distinct colour, matched between
# the spawned vehicle and its overlaid route polyline.
EGO_COLORS_RGB: list[tuple[int, int, int]] = [
    (231, 76, 60),    # red
    (52, 152, 219),   # blue
    (46, 204, 113),   # green
    (241, 196, 15),   # yellow
    (155, 89, 182),   # purple
    (230, 126, 34),   # orange
    (26, 188, 156),   # teal
    (236, 64, 122),   # pink
    (149, 165, 166),  # silver
]

PEDESTRIAN_OVERLAY_RGB = (255, 105, 180)   # hot pink
BICYCLE_OVERLAY_RGB = (102, 187, 106)      # bright green


def ego_color_for(idx: int) -> tuple[int, int, int]:
    return EGO_COLORS_RGB[idx % len(EGO_COLORS_RGB)]


# ────────────────────────────── CARLA spawning ──────────────────────────────

def _bp_for_kind(blueprint_lib, kind: str, requested_model: str = ""):
    """Pick a blueprint for an obstacle. Honours ``requested_model`` if found.

    Falls back to a kind-appropriate generic if the requested model is not
    available in the local CARLA build.
    """
    if requested_model:
        try:
            return blueprint_lib.find(requested_model)
        except IndexError:
            pass
    if kind == "pedestrian":
        peds = blueprint_lib.filter("walker.pedestrian.*")
        return peds[0] if peds else None
    if kind == "bicycle":
        bikes = blueprint_lib.filter("vehicle.bh.crossbike")
        if bikes:
            return bikes[0]
        bikes = blueprint_lib.filter("vehicle.diamondback.*")
        return bikes[0] if bikes else None
    if kind == "static":
        # Prefer a road-cone-like prop if available; else any static prop.
        for q in ("static.prop.constructioncone", "static.prop.trafficcone01",
                  "static.prop.streetbarrier", "static.prop.warningconstruction"):
            try:
                return blueprint_lib.find(q)
            except IndexError:
                continue
        statics = blueprint_lib.filter("static.prop.*")
        return statics[0] if statics else None
    # vehicle / npc default
    try:
        return blueprint_lib.find("vehicle.tesla.model3")
    except IndexError:
        cars = blueprint_lib.filter("vehicle.*")
        return cars[0] if cars else None


def _ego_blueprint(blueprint_lib, rgb: tuple[int, int, int]):
    """Tesla Model 3 with a custom colour matching this ego's overlay."""
    try:
        bp = blueprint_lib.find("vehicle.tesla.model3")
    except IndexError:
        cars = blueprint_lib.filter("vehicle.*")
        bp = cars[0] if cars else None
    if bp is not None and bp.has_attribute("color"):
        try:
            bp.set_attribute("color", f"{rgb[0]},{rgb[1]},{rgb[2]}")
        except Exception:
            pass
    if bp is not None and bp.has_attribute("role_name"):
        try:
            bp.set_attribute("role_name", "ego_bev")
        except Exception:
            pass
    return bp


def _snap_to_lane(world_map, x: float, y: float, z: float, yaw_hint: float,
                  *, max_dist_m: float = 4.0) -> tuple[float, float, float, float]:
    """Snap (x, y) to the nearest driving-lane waypoint, returning
    ``(x', y', z', yaw_deg)`` aligned with the lane direction.

    Picks the candidate (current, left-lane, right-lane) whose yaw is
    closest to ``yaw_hint`` so we keep the ego facing along its planned
    direction (not the opposite-flow lane).
    """
    loc = carla.Location(x=float(x), y=float(y), z=float(z))
    try:
        wp = world_map.get_waypoint(
            loc, project_to_road=True, lane_type=carla.LaneType.Driving,
        )
    except Exception:
        return x, y, z, yaw_hint
    if wp is None:
        return x, y, z, yaw_hint

    candidates = [wp]
    try:
        for nbr_call in (wp.get_left_lane, wp.get_right_lane):
            nbr = nbr_call()
            if nbr is not None and nbr.lane_type == carla.LaneType.Driving:
                candidates.append(nbr)
    except Exception:
        pass

    def _yaw_delta(a: float, b: float) -> float:
        return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)

    best = min(candidates, key=lambda c: _yaw_delta(c.transform.rotation.yaw, yaw_hint))
    if best.transform.location.distance(loc) > max_dist_m:
        return x, y, z, yaw_hint
    t = best.transform
    return float(t.location.x), float(t.location.y), float(t.location.z), float(t.rotation.yaw)


def force_all_traffic_lights_green(world) -> None:
    """Set every traffic light in the world to green and freeze it there.
    Avoids the random red/yellow CARLA picks for figures.
    """
    try:
        actors = world.get_actors().filter("traffic.traffic_light")
    except Exception:
        return
    for tl in actors:
        try:
            tl.set_state(carla.TrafficLightState.Green)
            # Freeze so synchronous-mode ticks don't cycle it back.
            tl.freeze(True)
        except Exception:
            continue


def _enable_vehicle_lights(actor) -> None:
    """Turn on headlights + position lights on a spawned vehicle so it reads
    correctly in night-tagged scenarios."""
    try:
        ls = (carla.VehicleLightState.Position
              | carla.VehicleLightState.LowBeam
              | carla.VehicleLightState.HighBeam)
        actor.set_light_state(carla.VehicleLightState(ls))
    except Exception:
        pass


_OBSTACLE_MIN_GAP_M = {
    "npc": 4.0,         # ~car length
    "bicycle": 1.5,
    "pedestrian": 0.6,
    "static": 0.8,
}


# Per-scenario per-NPC overrides for manually-spotted GT issues. Each entry
# matches an obstacle by ``xml_substring`` (substring of the actor's source
# XML filename, e.g. ``Vehicle_24``) and applies an action:
#   "drop"            — remove the NPC entirely
#   "flip_yaw"        — rotate yaw by 180°
#   {"nudge": (dx, dy)} — translate the spawn position by (dx, dy) metres
# Entries can be repeated. Matched obstacles are processed in this order.
SCENARIO_OBSTACLE_OVERRIDES: dict[str, list[dict]] = {
    # User flagged a vehicle in the right-turn lane, top-left of the BEV,
    # sitting at the lane edge. Diagnostic: Vehicle_5 (chevy impala) is
    # 1.8 m off the nearest driving lane at frac=0.30. Drop it.
    "2023-03-23-15-39-40_3_1": [
        {"xml_substring": "Vehicle_5_npc.xml", "action": "drop"},
    ],
    # User flagged a vehicle stacked on top of another. Diagnostic:
    # Vehicle_25 (dodge charger) sits 2.1 m off the lane near the egos.
    "2023-03-23-15-42-40_6_0": [
        {"xml_substring": "Vehicle_25_npc.xml", "action": "drop"},
    ],
    # User flagged a vehicle parked in the median. Diagnostic: Vehicle_16
    # (mercedes sprinter) is 4.3 m off any driving lane at frac=0.30.
    "2023-04-04-15-39-17_11_0": [
        {"xml_substring": "Vehicle_16_npc.xml", "action": "drop"},
    ],
    # 14_0:
    #   - Vehicle_5 (audi a2) sits sideways on a median at frac=0.30 →
    #     drop it everywhere.
    #   - Vehicle_8 + Vehicle_10 are STATIC parked Coca-Cola trucks (model
    #     vehicle.carlamotors.carlacola) facing south, but the user wants
    #     them facing the opposite direction (parking flush with oncoming
    #     traffic). Flip 180° everywhere — statics don't move so this
    #     applies the same correction to every variant.
    "2023-04-05-16-17-26_14_0": [
        {"xml_substring": "Vehicle_5_npc.xml", "action": "drop"},
        {"xml_substring": "Vehicle_8_static.xml", "action": "flip_yaw"},
        {"xml_substring": "Vehicle_10_static.xml", "action": "flip_yaw"},
    ],
    # User flagged a white vehicle in the middle-left facing the wrong
    # way. Diagnostic: Vehicle_39 (vw t2 microbus, light-coloured) has
    # yaw 77° off the nearest lane axis. Flip 180°.
    "2023-04-05-16-24-26_21_1": [
        {"xml_substring": "Vehicle_39_npc.xml", "action": "flip_yaw"},
    ],
    # User flagged a sideways vehicle near the middle. Diagnostic:
    # Vehicle_37 (mini cooper) at +157° yaw and 1.6 m off the lane.
    "2023-04-05-16-25-26_22_1": [
        {"xml_substring": "Vehicle_37_npc.xml", "action": "drop"},
    ],
}


def _apply_obstacle_overrides(scen_id: str, obstacles: list[dict],
                               variant: str = "") -> list[dict]:
    """Apply per-scenario overrides. Each entry may include
    ``"variants": [...]`` to limit it to specific variant suffixes
    (``approach``, ``interaction``, ``past``, ``bev``, ``routes``);
    omit to apply to all.
    """
    overrides = SCENARIO_OBSTACLE_OVERRIDES.get(scen_id, [])
    if not overrides:
        return obstacles
    out: list[dict] = []
    for ob in obstacles:
        xml_name = str(ob.get("xml", ""))
        action_for: list[dict] = []
        for ov in overrides:
            sub = ov.get("xml_substring", "")
            if not sub or sub not in xml_name:
                continue
            allowed = ov.get("variants")
            if allowed and variant and variant not in allowed:
                continue
            action_for.append(ov)
        if not action_for:
            out.append(ob)
            continue
        keep = True
        new_ob = dict(ob)
        for ov in action_for:
            act = ov.get("action")
            if act == "drop":
                keep = False
                break
            if act == "flip_yaw":
                new_ob["yaw"] = (float(new_ob.get("yaw", 0.0)) + 180.0) % 360.0
            elif act == "nudge":
                dx, dy = ov.get("delta", (0.0, 0.0))
                new_ob["x"] = float(new_ob["x"]) + float(dx)
                new_ob["y"] = float(new_ob["y"]) + float(dy)
        if keep:
            out.append(new_ob)
    return out


def _filter_obstacle_overlaps(obstacles: list[dict]) -> list[dict]:
    """Drop obstacles whose spawn position would overlap an earlier obstacle
    of the same kind. Keep order: earlier-listed actors win the spot.
    """
    kept_by_kind: dict[str, list[tuple[float, float]]] = {}
    out: list[dict] = []
    for ob in obstacles:
        kind = ob.get("kind", "")
        gap = _OBSTACLE_MIN_GAP_M.get(kind, 1.0)
        anchors = kept_by_kind.setdefault(kind, [])
        x, y = ob["x"], ob["y"]
        if any(math.hypot(x - ax, y - ay) < gap for ax, ay in anchors):
            continue
        out.append(ob)
        anchors.append((x, y))
    return out


def _obstacle_pose_at_fraction(ob: dict, frac: float) -> dict:
    """Where this obstacle is at *frac* into the scenario timeline.

    NPCs, walkers, and bicycles use their recorded GT polyline so they're
    in time-sync with whatever fraction the egos are placed at. Static
    props don't move, so they stay at the first-waypoint pose.
    """
    if ob.get("kind") == "static":
        return ob
    poly = ob.get("polyline") or []
    pose = _polyline_pose_at_fraction(poly, frac) if len(poly) >= 2 else None
    if pose is None:
        return ob
    return {**ob, **pose}


def _destroy_cycle(world, actors: list, *, client=None, ticks: int = 4) -> None:
    """Destroy *actors* reliably and tick the world enough times for the
    destroys to land before the next spawn.

    Uses ``client.apply_batch_sync(DestroyActor)`` when a *client* is
    provided — that's the only CARLA API that actually waits for the
    server to confirm the destroys, which matters for high-actor scenes
    like v2xpnp (90+ actors per scenario) where the per-actor async
    ``actor.destroy()`` can leak ghost actors into the next capture.
    """
    if not actors:
        return
    ids = [a.id for a in actors if a is not None]
    destroyed = False
    if client is not None and ids:
        try:
            cmds = [carla.command.DestroyActor(i) for i in ids]
            client.apply_batch_sync(cmds, True)
            destroyed = True
        except Exception:
            pass
    if not destroyed:
        for a in actors:
            try:
                a.destroy()
            except Exception:
                pass
    for _ in range(ticks):
        try:
            world.tick()
        except Exception:
            break


def _safe_spawn(world, bp, transform, *, max_retries: int = 4, lift_step: float = 0.4):
    """Try ``world.try_spawn_actor`` with progressively higher z to dodge
    occasional "spawn failed because of collision at spawn position"."""
    z0 = transform.location.z
    for k in range(max_retries):
        t = carla.Transform(
            carla.Location(x=transform.location.x, y=transform.location.y,
                           z=z0 + 0.5 + k * lift_step),
            transform.rotation,
        )
        actor = world.try_spawn_actor(bp, t)
        if actor is not None:
            return actor
    return None


# ───────────────────────────── BEV camera capture ───────────────────────────

@dataclass
class CameraParams:
    cx: float
    cy: float
    cz: float
    fov_deg: float
    width: int
    height: int


def _camera_bbox(positions: list[dict]) -> tuple[float, float, float]:
    xs = [p["x"] for p in positions]
    ys = [p["y"] for p in positions]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    half_extent = max(max(xs) - min(xs), max(ys) - min(ys)) / 2.0
    return cx, cy, half_extent


def _camera_height_for_margin(margin: float, fov_deg: float) -> float:
    fov_rad = math.radians(max(1.0, min(fov_deg, 175.0)))
    return max(margin / math.tan(fov_rad / 2.0) + 12.0, 60.0)


def _circular_mean_yaw(yaws_deg: list[float]) -> float:
    if not yaws_deg:
        return 0.0
    sx = sum(math.cos(math.radians(y)) for y in yaws_deg)
    sy = sum(math.sin(math.radians(y)) for y in yaws_deg)
    return math.degrees(math.atan2(sy, sx))


_OCCLUDING_LABELS = {
    "Buildings", "Walls", "Fences", "Bridge", "GuardRail", "Static",
    "Other", "Vegetation",
}


def _los_clear(world, cam_loc, target_loc, *, slack_m: float = 1.5) -> bool:
    """True iff a ray from *cam_loc* to *target_loc* is not blocked by a
    building/wall/etc. We ignore Vehicles/Dynamic/Pedestrian hits (those are
    spawned actors we *want* to see) and only fail on rigid-world geometry.
    """
    try:
        hits = world.cast_ray(cam_loc, target_loc)
    except Exception:
        return True
    if not hits:
        return True
    target_dist = cam_loc.distance(target_loc)
    for hit in hits:
        try:
            label_name = hit.label.name
        except Exception:
            label_name = ""
        if label_name not in _OCCLUDING_LABELS:
            continue
        try:
            d = cam_loc.distance(hit.location)
        except Exception:
            continue
        if d + slack_m < target_dist:
            return False
    return True


def _oblique_camera_candidates(
    cx: float, cy: float, cz: float, fwd_yaw_deg: float,
) -> list[tuple[float, float, float, float, float]]:
    """Return a prioritised list of candidate camera poses
    ``(cam_x, cam_y, cam_z, cam_yaw, cam_pitch)``. Order: best-looking first.
    """
    out: list[tuple[float, float, float, float, float]] = []
    # Forward unit vector for the scenario.
    yaw_rad0 = math.radians(fwd_yaw_deg)
    cosY, sinY = math.cos(yaw_rad0), math.sin(yaw_rad0)

    # (yaw_offset_deg, dist_behind_m, height_above_m, pitch_deg).
    spec = [
        (0.0,    16.0, 18.0, -32.0),   # behind, classic chase
        (0.0,    20.0, 28.0, -45.0),   # behind, elevated drone
        (45.0,   16.0, 22.0, -38.0),   # rear-right
        (-45.0,  16.0, 22.0, -38.0),   # rear-left
        (90.0,   14.0, 20.0, -38.0),   # right side
        (-90.0,  14.0, 20.0, -38.0),   # left side
        (135.0,  14.0, 22.0, -42.0),   # front-right (looking back)
        (-135.0, 14.0, 22.0, -42.0),   # front-left
        (0.0,    24.0, 50.0, -65.0),   # high birdseye-ish, last resort
    ]
    for yaw_off, dist_behind, height_above, pitch_deg in spec:
        ang = yaw_rad0 + math.radians(yaw_off + 180.0)
        # Direction from scene centroid to the camera position.
        ux, uy = math.cos(ang), math.sin(ang)
        cam_x = cx + dist_behind * ux
        cam_y = cy + dist_behind * uy
        cam_z = cz + height_above
        # Camera looks back at the centroid (yaw rotated 180° from
        # placement direction).
        look_dx = cx - cam_x
        look_dy = cy - cam_y
        cam_yaw = math.degrees(math.atan2(look_dy, look_dx))
        out.append((cam_x, cam_y, cam_z, cam_yaw, pitch_deg))
    return out


def _oblique_camera_pose(
    world,
    placed_egos: dict[int, dict],
    obstacles: list[dict],
) -> tuple[float, float, float, float, float]:
    """Pick the best camera pose for an oblique chase-cam view.

    Try a ranked list of candidate positions (behind/elevated/sides/etc.).
    For each candidate, raycast to every spawned ego and the nearest
    obstacle. The first candidate whose LOS is not occluded by buildings/
    vegetation/walls wins; if none are fully clear, return the one with
    the highest "visible-target" count.
    """
    xs = [p["x"] for p in placed_egos.values()]
    ys = [p["y"] for p in placed_egos.values()]
    zs = [p["z"] for p in placed_egos.values()]
    yaws = [p["yaw"] for p in placed_egos.values()]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    cz = sum(zs) / len(zs)

    nearby = [
        ob for ob in obstacles
        if math.hypot(ob["x"] - cx, ob["y"] - cy) <= 25.0
    ]
    if nearby:
        ox = sum(o["x"] for o in nearby) / len(nearby)
        oy = sum(o["y"] for o in nearby) / len(nearby)
        dx, dy = ox - cx, oy - cy
        if math.hypot(dx, dy) < 1.0:
            yaw = _circular_mean_yaw(yaws)
            dx, dy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    else:
        yaw = _circular_mean_yaw(yaws)
        dx, dy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))

    fwd_yaw_deg = math.degrees(math.atan2(dy, dx))

    # Targets to LOS-check: every spawned ego centre, plus closest obstacle.
    targets: list[tuple[float, float, float]] = []
    for p in placed_egos.values():
        targets.append((p["x"], p["y"], p.get("z", 0.0) + 1.0))
    if nearby:
        nearest = min(nearby, key=lambda o: math.hypot(o["x"] - cx, o["y"] - cy))
        targets.append((nearest["x"], nearest["y"], nearest.get("z", 0.0) + 1.0))

    candidates = _oblique_camera_candidates(cx, cy, cz, fwd_yaw_deg)
    best_pose = candidates[0]
    best_visible = -1
    for pose in candidates:
        cam_loc = carla.Location(x=pose[0], y=pose[1], z=pose[2])
        visible = 0
        for tx, ty, tz in targets:
            tgt = carla.Location(x=tx, y=ty, z=tz)
            if _los_clear(world, cam_loc, tgt):
                visible += 1
        if visible == len(targets):
            return pose
        if visible > best_visible:
            best_visible = visible
            best_pose = pose
    return best_pose


def capture_oblique(
    world,
    placed_egos: dict[int, dict],
    obstacles: list[dict],
    *,
    width: int,
    height: int,
    fov: float = 70.0,
) -> np.ndarray:
    """Chase-cam-style 45°-ish view from behind/above the ego cluster.

    Picks a non-occluded angle by raycasting candidate camera positions
    against the world geometry.
    """
    cam_x, cam_y, cam_z, cam_yaw, cam_pitch = _oblique_camera_pose(
        world, placed_egos, obstacles,
    )
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(width))
    bp.set_attribute("image_size_y", str(height))
    bp.set_attribute("fov", f"{fov:.2f}")
    transform = carla.Transform(
        carla.Location(x=cam_x, y=cam_y, z=cam_z),
        carla.Rotation(pitch=cam_pitch, yaw=cam_yaw, roll=0.0),
    )
    sensor = world.spawn_actor(bp, transform)
    img_q: queue.Queue = queue.Queue()
    sensor.listen(img_q.put)
    try:
        for _ in range(8):
            world.tick()
        img = img_q.get(timeout=10.0)
    finally:
        sensor.stop()
        sensor.destroy()
    arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(height, width, 4)
    return arr[:, :, :3][:, :, ::-1].copy()


def draw_route_debug(world, ego_routes: dict[int, list[dict]],
                     life_time: float = 1.0,
                     base_z_offset: float = 1.0,
                     thickness: float = 0.9) -> None:
    """Draw each ego route as a thick coloured polyline using CARLA's debug
    framework, so the lines render in the simulator and appear in the next
    captured frame.

    Each segment is drawn separately at its own colour (matched to the
    spawned ego's vehicle colour). ``life_time=0`` means persist forever in
    CARLA, which is bad — use a positive value > capture latency.
    """
    debug = world.debug
    for idx, route in sorted(ego_routes.items()):
        if not route:
            continue
        rgb = ego_color_for(idx)
        color = carla.Color(r=rgb[0], g=rgb[1], b=rgb[2], a=255)
        for a, b in zip(route, route[1:]):
            try:
                p0 = carla.Location(
                    x=float(a["x"]), y=float(a["y"]),
                    z=float(a.get("z", 0.0)) + base_z_offset,
                )
                p1 = carla.Location(
                    x=float(b["x"]), y=float(b["y"]),
                    z=float(b.get("z", 0.0)) + base_z_offset,
                )
                debug.draw_line(p0, p1, thickness=thickness,
                                color=color, life_time=life_time)
            except Exception:
                continue


def capture_overhead(
    world,
    cx: float,
    cy: float,
    margin: float,
    *,
    width: int,
    height: int,
    fov: float,
) -> tuple[np.ndarray, CameraParams]:
    cam_z = _camera_height_for_margin(margin, fov)

    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(width))
    bp.set_attribute("image_size_y", str(height))
    bp.set_attribute("fov", f"{fov:.2f}")

    transform = carla.Transform(
        carla.Location(x=cx, y=cy, z=cam_z),
        carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
    )
    sensor = world.spawn_actor(bp, transform)
    img_q: queue.Queue = queue.Queue()
    sensor.listen(img_q.put)
    try:
        for _ in range(8):
            world.tick()
        img = img_q.get(timeout=10.0)
    finally:
        sensor.stop()
        sensor.destroy()
    arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(height, width, 4)
    rgb = arr[:, :, :3][:, :, ::-1].copy()  # BGRA -> RGB
    return rgb, CameraParams(cx=cx, cy=cy, cz=cam_z, fov_deg=fov,
                             width=width, height=height)


# ─────────────────────── World → pixel projection + overlay ─────────────────

def _world_to_pixel(pts_world: np.ndarray, cam: CameraParams) -> np.ndarray:
    """Project (N, 3) world XYZ to (N, 2) pixel coords for a downward camera.

    Mirror of ``capture_overhead_with_centerlines.world_to_pixel``: camera at
    ``(cam.cx, cam.cy, cam.cz)`` with pitch=-90, yaw=0. World +X → image right,
    world +Y → image down.
    """
    f = cam.height / (2.0 * math.tan(math.radians(cam.fov_deg) / 2.0))
    rel = pts_world - np.array([cam.cx, cam.cy, cam.cz])
    cam_pts = np.stack([rel[:, 0], rel[:, 1], -rel[:, 2]], axis=-1)
    K = np.array([[f, 0, cam.width / 2.0],
                  [0, f, cam.height / 2.0],
                  [0, 0, 1.0]], dtype=np.float64)
    proj = (K @ cam_pts.T).T
    proj[:, :2] /= proj[:, 2:3]
    return proj[:, :2]


def _smooth_polyline(pts: np.ndarray, iters: int = 2) -> np.ndarray:
    """Chaikin corner-cutting for visual smoothness. *pts* shape (N, 2)."""
    if len(pts) < 3:
        return pts
    cur = pts.astype(np.float64)
    for _ in range(iters):
        if len(cur) < 3:
            break
        new = [cur[0]]
        for i in range(len(cur) - 1):
            p0 = cur[i]
            p1 = cur[i + 1]
            new.append(0.75 * p0 + 0.25 * p1)
            new.append(0.25 * p0 + 0.75 * p1)
        new.append(cur[-1])
        cur = np.asarray(new)
    return cur


def _draw_fading_polyline(
    canvas: np.ndarray,
    pts_world: list[tuple[float, float, float]],
    cam: CameraParams,
    color_rgb: tuple[int, int, int],
    *,
    halo_width: int = 18,
    line_width: int = 9,
    head_alpha: float = 1.0,
    tail_alpha: float = 0.18,
    halo_alpha: float = 0.32,
    arrow: bool = True,
) -> None:
    """Draw a smooth, head-bright / tail-faded polyline on *canvas* in place.

    Uses a per-pixel *maximum* alpha mask so overlapping segments of the same
    polyline don't darken each other (avoids the over-saturated "blob" you
    get from sequential alpha composites). Each segment paints its alpha into
    a float mask via ``np.maximum``; the final mask is composited onto the
    canvas in a single pass.
    """
    if len(pts_world) < 2:
        return
    arr = np.asarray(pts_world, dtype=np.float64)
    if arr.shape[1] == 2:
        arr = np.concatenate([arr, np.zeros((len(arr), 1))], axis=1)
    px = _world_to_pixel(arr, cam)
    px = _smooth_polyline(px, iters=2)
    px_int = np.round(px).astype(np.int32)

    H, W = canvas.shape[:2]
    # canvas is an RGB ndarray; cv2 writes the colour tuple verbatim, so we
    # pass (R, G, B) directly (not BGR-swapped) to keep hues correct.
    rgb_tup = (int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2]))
    rgb_arr = np.array(rgb_tup, dtype=np.float32)

    n = len(px_int) - 1
    if n <= 0:
        return

    # Per-segment alpha schedule (ease-out).
    t = np.linspace(0.0, 1.0, n) ** 1.4
    seg_alphas = head_alpha + t * (tail_alpha - head_alpha)

    # Build halo + core masks via per-segment max-alpha accumulation.
    halo_mask = np.zeros((H, W), dtype=np.float32)
    core_mask = np.zeros((H, W), dtype=np.float32)
    halo_seg = np.empty((H, W), dtype=np.uint8)
    core_seg = np.empty((H, W), dtype=np.uint8)
    for i in range(n):
        a = float(np.clip(seg_alphas[i], 0.0, 1.0))
        if a <= 0.01:
            continue
        halo_seg[:] = 0
        core_seg[:] = 0
        cv2.line(halo_seg, tuple(px_int[i]), tuple(px_int[i + 1]),
                 255, halo_width, cv2.LINE_AA)
        cv2.line(core_seg, tuple(px_int[i]), tuple(px_int[i + 1]),
                 255, line_width, cv2.LINE_AA)
        np.maximum(halo_mask, (halo_seg.astype(np.float32) / 255.0) * a * halo_alpha,
                   out=halo_mask)
        np.maximum(core_mask, (core_seg.astype(np.float32) / 255.0) * a, out=core_mask)

    # Composite halo (soft glow), then core (crisp line).
    if halo_mask.max() > 0:
        m = halo_mask[..., None]
        canvas[:] = (canvas.astype(np.float32) * (1.0 - m) + rgb_arr * m).astype(np.uint8)
    if core_mask.max() > 0:
        m = core_mask[..., None]
        canvas[:] = (canvas.astype(np.float32) * (1.0 - m) + rgb_arr * m).astype(np.uint8)

    # Arrowhead at the tail (more visible than the faded tail line itself).
    if arrow and n >= 1:
        tip = tuple(int(v) for v in px_int[-1])
        head_len = max(line_width * 2.5, 18.0)
        if not (0 <= tip[0] < W and 0 <= tip[1] < H):
            return
        cum = 0.0
        base_idx = n
        for i in range(n - 1, -1, -1):
            d = float(np.linalg.norm(px[i + 1] - px[i]))
            cum += d
            if cum >= head_len:
                base_idx = i
                break
        if base_idx == n:
            return
        base = tuple(int(v) for v in np.round(px[base_idx]))
        a = float(np.clip(tail_alpha + 0.25, 0.0, 1.0))
        layer = canvas.copy()
        cv2.arrowedLine(layer, base, tip, rgb_tup, line_width + 2,
                        line_type=cv2.LINE_AA, tipLength=0.45)
        cv2.addWeighted(layer, a, canvas, 1.0 - a, 0.0, dst=canvas)


def _polyline_in_view(pts_world: list[dict], cam: CameraParams,
                      pad_px: int = 40) -> bool:
    """True if at least one point of the polyline projects inside the camera
    frame (with a *pad_px* slack so near-frame routes still render)."""
    if not pts_world:
        return False
    arr = np.asarray([[p["x"], p["y"], p.get("z", 0.0)] for p in pts_world],
                     dtype=np.float64)
    px = _world_to_pixel(arr, cam)
    in_x = (px[:, 0] >= -pad_px) & (px[:, 0] <= cam.width + pad_px)
    in_y = (px[:, 1] >= -pad_px) & (px[:, 1] <= cam.height + pad_px)
    return bool(np.any(in_x & in_y))


def _polyline_min_dist_to(points: list[dict], targets: list[dict]) -> float:
    if not points or not targets:
        return float("inf")
    best = float("inf")
    for p in points:
        for q in targets:
            d = math.hypot(p["x"] - q["x"], p["y"] - q["y"])
            if d < best:
                best = d
    return best


def render_overlay(
    base_rgb: np.ndarray,
    cam: CameraParams,
    placed_egos: dict[int, dict],
    ego_routes: dict[int, list[dict]],
    obstacles: list[dict],
    *,
    draw_egos: bool = True,
    draw_pedestrians: bool = True,
    draw_bicycles: bool = True,
    ped_proximity_m: float = 25.0,
    bike_proximity_m: float = 35.0,
    max_ped_polylines: int = 6,
    max_bike_polylines: int = 6,
) -> np.ndarray:
    """Return a copy of *base_rgb* with fading route polylines for every ego
    (colour-matched), plus optional pedestrian / bicycle paths.
    """
    canvas = base_rgb.copy()

    # Only show non-ego polylines that are *close* to at least one placed ego.
    # In dense v2x scenes (50+ walkers), this limits overlay to the
    # interaction-relevant subset and prevents the background from drowning
    # the ego trajectories. Non-relevant pedestrians/bikes are still spawned
    # in the scene and visible as small markers; we just don't draw their
    # path lines.
    ego_anchors = list(placed_egos.values())

    def _candidate_subset(kind: str, proximity: float, cap: int) -> list[dict]:
        # Keep only obstacles of *kind* whose polyline is in-view AND within
        # *proximity* of an ego. Sort by closest-to-ego distance and cap to
        # *cap* — limits dense-scene clutter without dropping the closest /
        # most figure-relevant ones.
        scored: list[tuple[float, dict]] = []
        for ob in obstacles:
            if ob.get("kind") != kind:
                continue
            poly = ob.get("polyline") or [{"x": ob["x"], "y": ob["y"], "z": ob.get("z", 0.0)}]
            if len(poly) < 2 or not _polyline_in_view(poly, cam):
                continue
            d = _polyline_min_dist_to(poly, ego_anchors)
            if d > proximity:
                continue
            scored.append((d, ob))
        scored.sort(key=lambda t: t[0])
        return [ob for _, ob in scored[:cap]]

    if draw_pedestrians:
        for ob in _candidate_subset("pedestrian", ped_proximity_m, max_ped_polylines):
            poly = ob.get("polyline") or []
            pts = [(p["x"], p["y"], p.get("z", 0.0)) for p in poly]
            _draw_fading_polyline(canvas, pts, cam, PEDESTRIAN_OVERLAY_RGB,
                                  halo_width=12, line_width=5,
                                  head_alpha=0.92, tail_alpha=0.15,
                                  halo_alpha=0.22)

    if draw_bicycles:
        for ob in _candidate_subset("bicycle", bike_proximity_m, max_bike_polylines):
            poly = ob.get("polyline") or []
            pts = [(p["x"], p["y"], p.get("z", 0.0)) for p in poly]
            _draw_fading_polyline(canvas, pts, cam, BICYCLE_OVERLAY_RGB,
                                  halo_width=14, line_width=6,
                                  head_alpha=0.92, tail_alpha=0.15,
                                  halo_alpha=0.22)

    if draw_egos:
        # Draw each ego's route from its 30%-progress pose forward — head bright,
        # tail faded — so the figure communicates "where this ego is heading".
        for idx in sorted(placed_egos.keys()):
            route = ego_routes.get(idx) or []
            if len(route) < 2:
                continue
            pose = placed_egos[idx]
            frac = pose.get("frac", 0.30)
            cum, total = _route_arc_length(route)
            target = float(frac) * max(total, 1e-3)
            # Find first index past 'target'.
            start_idx = 0
            for i in range(1, len(cum)):
                if cum[i] >= target:
                    start_idx = i
                    break
            sub = [{"x": pose["x"], "y": pose["y"], "z": pose["z"]}] + route[start_idx:]
            pts = [(p["x"], p["y"], p.get("z", 0.0)) for p in sub]
            color = ego_color_for(idx)
            _draw_fading_polyline(canvas, pts, cam, color,
                                  halo_width=26, line_width=11,
                                  head_alpha=1.0, tail_alpha=0.20,
                                  halo_alpha=0.32)

    return canvas


# ───────────────────────── Placement variants ─────────────────────────────

# Each variant produces one figure per scenario. The three variants together
# capture: leading-up-to-the-interaction, the interaction moment, and just
# past it — so the paper figure communicates the full arc.
PLACEMENT_VARIANTS: list[dict] = [
    {
        "suffix": "_approach",
        "nominal_frac": 0.10,
        "pre_obstacle_offset_m": 18.0,
        "pre_interaction_offset_m": 18.0,
    },
    {
        "suffix": "_interaction",
        "nominal_frac": 0.30,
        "pre_obstacle_offset_m": 6.0,
        "pre_interaction_offset_m": 5.0,
    },
    {
        "suffix": "_past",
        "nominal_frac": 0.55,
        # Negative offset = place AFTER the interaction/obstacle along the route.
        "pre_obstacle_offset_m": -6.0,
        "pre_interaction_offset_m": -4.0,
    },
]


# ─────────────────────────── Weather (no shadows) ───────────────────────────

def _normalize_weather_id(name: str) -> str:
    """Map an XML weather id to the closest ``carla.WeatherParameters``
    preset attribute name.

    Examples: ``"cloudy"`` → ``"CloudyNoon"``, ``"night"`` → ``"ClearNight"``,
    ``"clear noon"`` → ``"ClearNoon"``.
    """
    raw = (name or "").strip()
    if not raw:
        return ""
    # Already in preset form (e.g. "ClearNoon").
    if hasattr(carla.WeatherParameters, raw):
        return raw
    # Drop punctuation, split into tokens.
    tokens = [t for t in re.split(r"[\s_\-]+", raw) if t]
    if not tokens:
        return ""
    # Heuristic: tokens describe weather + time-of-day.
    weather_map = {
        "default": "Clear", "clear": "Clear", "sunny": "Clear",
        "cloudy": "Cloudy", "cloud": "Cloudy", "overcast": "Cloudy",
        "wet": "Wet", "rain": "MidRainy", "rainy": "MidRainy",
        "midrain": "MidRainy", "midrainy": "MidRainy",
        "softrain": "SoftRainy", "softrainy": "SoftRainy",
        "hardrain": "HardRain", "hardrainy": "HardRain",
        "wetcloudy": "WetCloudy",
    }
    tod_map = {
        "noon": "Noon", "day": "Noon", "morning": "Noon",
        "sunset": "Sunset", "evening": "Sunset", "dusk": "Sunset",
        "night": "Night", "dawn": "Sunset",
    }
    weather_part = ""
    tod_part = ""
    for t in tokens:
        tl = t.lower()
        if not weather_part and tl in weather_map:
            weather_part = weather_map[tl]
        elif not tod_part and tl in tod_map:
            tod_part = tod_map[tl]
    if not weather_part:
        weather_part = "Clear"
    if not tod_part:
        tod_part = "Noon"
    cand = f"{weather_part}{tod_part}"
    if hasattr(carla.WeatherParameters, cand):
        return cand
    fallback = f"{weather_part}Noon"
    return fallback if hasattr(carla.WeatherParameters, fallback) else "ClearNoon"


def _is_night_preset(preset: str) -> bool:
    return preset.endswith("Night")


def apply_scenario_weather(world, weather_id: str) -> Optional[str]:
    """Apply the CARLA built-in weather preset matching the scenario's XML
    ``weather`` attribute, as-is. ``night`` stays night, ``cloudy`` stays
    cloudy. No exposure / sun-altitude tampering.
    """
    preset = _normalize_weather_id(weather_id)
    if not preset:
        return None
    try:
        params = getattr(carla.WeatherParameters, preset)
    except AttributeError:
        return None
    try:
        world.set_weather(params)
        return preset
    except Exception:
        return None


# ─────────────────────────────── Map loading ────────────────────────────────

def _resolve_town_map_name(client, town: str) -> Optional[str]:
    """Resolve a scenario town name (e.g. ``ucla_v2``) to a CARLA map id."""
    if not town:
        return None
    avail = client.get_available_maps()
    for m in avail:
        if m.split("/")[-1].lower() == town.lower():
            return m
    for m in avail:
        if town.lower() in m.lower():
            return m
    return None


def load_town_if_needed(client, world, town: str) -> tuple[object, bool]:
    """Load *town* via ``client.load_world`` if the world isn't already on it.

    Returns ``(world, loaded)``. ``loaded=False`` means we kept the existing
    world. On failure (map not available) returns ``(world, False)``.
    """
    if not town:
        return world, False
    cur = ""
    try:
        cur = world.get_map().name
    except RuntimeError:
        pass
    if cur and town.lower() in cur.lower():
        return world, False
    match = _resolve_town_map_name(client, town)
    if match is None:
        return world, False
    print(f"  [town] loading {match} ...")
    new_world = client.load_world(match)
    time.sleep(2.0)
    return new_world, True


# ─────────────────────────────── Main pipeline ──────────────────────────────

@dataclass
class ScenarioPlan:
    bucket: str
    scenario_id: str
    scenario_dir: Path
    town: str
    weather_id: str
    ego_routes: dict[int, list[dict]]
    obstacles: list[dict]
    placed_egos: dict[int, dict] = field(default_factory=dict)
    dropped_egos: list[int] = field(default_factory=list)


def safe_filename(bucket: str, scenario_id: str) -> str:
    cleaned = scenario_id.replace("/", "_").replace("\\", "_")
    cleaned = re.sub(r"\s+", "_", cleaned)
    return f"{bucket}_{cleaned}.png"


def plan_scenarios(filter_buckets: Optional[set[str]] = None,
                   limit: Optional[int] = None) -> list[ScenarioPlan]:
    plans: list[ScenarioPlan] = []
    discovered = discover_scenarios()
    if filter_buckets:
        discovered = [s for s in discovered if s["bucket"] in filter_buckets]
    if limit is not None:
        discovered = discovered[:limit]
    for i, scen in enumerate(discovered, 1):
        try:
            actors = bsd.collect_actor_counts(scen)
        except Exception as exc:
            print(f"[{i}/{len(discovered)}] {scen['bucket']}/{scen['scenario_id']}: "
                  f"actor parse failed: {exc}")
            continue
        town = actors.get("town")
        if not town:
            print(f"[{i}/{len(discovered)}] {scen['bucket']}/{scen['scenario_id']}: "
                  f"no town in XML; skipping")
            continue
        ego_xmls = actors.get("ego_xmls", [])
        if not ego_xmls:
            print(f"[{i}/{len(discovered)}] {scen['bucket']}/{scen['scenario_id']}: "
                  f"no ego XMLs; skipping")
            continue
        routes = resolve_ego_routes(scen, ego_xmls)
        if not routes:
            print(f"[{i}/{len(discovered)}] {scen['bucket']}/{scen['scenario_id']}: "
                  f"no ego routes resolved; skipping")
            continue
        obstacles = collect_obstacles(scen)
        weather_id = actors.get("weather_id") or "default"
        plans.append(ScenarioPlan(
            bucket=scen["bucket"],
            scenario_id=scen["scenario_id"],
            scenario_dir=scen["scenario_dir"],
            town=town,
            weather_id=weather_id,
            ego_routes=routes,
            obstacles=obstacles,
        ))
    return plans


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--out", type=Path,
                    default=REPO / "figures" / "scenario_bevs",
                    help="output directory for PNGs")
    ap.add_argument("--bucket", action="append",
                    choices=["interaction", "precrash", "v2xpnp", "interdrive"],
                    help="restrict to one or more buckets (repeatable)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of scenarios processed (smoke testing)")
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--height", type=int, default=2048)
    ap.add_argument("--fov", type=float, default=70.0)
    ap.add_argument("--ego-frac", type=float, default=0.30,
                    help="nominal arc-length fraction for ego placement")
    ap.add_argument("--min-spacing", type=float, default=10.0,
                    help="minimum metres between any pair of spawned egos")
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite existing output PNGs (default: skip)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print("Discovering scenarios ...")
    plans = plan_scenarios(
        filter_buckets=set(args.bucket) if args.bucket else None,
        limit=args.limit,
    )
    print(f"Planned {len(plans)} scenarios.")

    # Group by town to amortise load_world.
    by_town: dict[str, list[ScenarioPlan]] = defaultdict(list)
    for p in plans:
        by_town[p.town].append(p)

    print("Connecting to CARLA ...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.get_world()

    orig_settings = world.get_settings()

    success = 0
    skipped_existing = 0
    failures: list[tuple[str, str]] = []
    for town, town_plans in sorted(by_town.items()):
        print(f"\n=== Town: {town}  ({len(town_plans)} scenarios) ===")
        world, _ = load_town_if_needed(client, world, town)
        sync_settings = carla.WorldSettings()
        sync_settings.synchronous_mode = True
        sync_settings.fixed_delta_seconds = 0.05
        if hasattr(sync_settings, "no_rendering_mode"):
            sync_settings.no_rendering_mode = False
        world.apply_settings(sync_settings)

        bplib = world.get_blueprint_library()
        last_weather: Optional[str] = None
        # Force all traffic lights green for the whole town pass — paper figures
        # shouldn't show random red/yellow stops.
        force_all_traffic_lights_green(world)
        for _ in range(2):
            try:
                world.tick()
            except Exception:
                break

        for j, plan in enumerate(town_plans, 1):
            base_name = safe_filename(plan.bucket, plan.scenario_id)
            stem = base_name[:-len(".png")]
            variant_paths = [args.out / f"{stem}{v['suffix']}.png"
                             for v in PLACEMENT_VARIANTS]
            bev_path = args.out / f"{stem}_bev.png"
            routes_path = args.out / f"{stem}_routes.png"
            all_outputs = variant_paths + [bev_path, routes_path]
            if not args.overwrite and all(p.exists() for p in all_outputs):
                skipped_existing += 1
                print(f"  [{j}/{len(town_plans)}] {plan.bucket}/{plan.scenario_id}: "
                      f"all variants exist, skip", flush=True)
                continue

            # Apply scenario's encoded weather once per scenario.
            if plan.weather_id and plan.weather_id != last_weather:
                apply_scenario_weather(world, plan.weather_id)
                last_weather = plan.weather_id
                for _ in range(3):
                    world.tick()
            is_night = _is_night_preset(_normalize_weather_id(plan.weather_id))

            world_map = world.get_map()
            n_variants_done = 0
            n_variants_failed = 0

            try:
                # ── 3 oblique placement variants ──────────────────────────
                for v_idx, variant in enumerate(PLACEMENT_VARIANTS):
                    out_path = variant_paths[v_idx]
                    if out_path.exists() and not args.overwrite:
                        n_variants_done += 1
                        continue

                    # Advance NPCs / walkers / bicycles to the same arc-length
                    # fraction as the egos for this variant, so they're shown
                    # in their GT positions at the same simulated time. Static
                    # props don't move.
                    advanced_obstacles = _filter_obstacle_overlaps([
                        _obstacle_pose_at_fraction(ob, variant["nominal_frac"])
                        for ob in plan.obstacles
                    ])
                    variant_name = variant["suffix"].lstrip("_")
                    advanced_obstacles = _apply_obstacle_overrides(
                        plan.scenario_id, advanced_obstacles, variant_name,
                    )
                    placed, dropped = place_egos(
                        plan.ego_routes,
                        advanced_obstacles,
                        nominal_frac=variant["nominal_frac"],
                        min_ego_dist=args.min_spacing,
                        pre_obstacle_offset_m=variant["pre_obstacle_offset_m"],
                        pre_interaction_offset_m=variant["pre_interaction_offset_m"],
                    )
                    if not placed:
                        n_variants_failed += 1
                        continue

                    spawned: list = []
                    try:
                        # Lane-snap all egos *first* so we know their final
                        # positions before deciding which obstacles to spawn.
                        snapped_egos: dict[int, dict] = {}
                        for idx, pose in placed.items():
                            sx, sy, sz, syaw = _snap_to_lane(
                                world_map, pose["x"], pose["y"], pose["z"], pose["yaw"],
                            )
                            pose["x"], pose["y"], pose["z"], pose["yaw"] = sx, sy, sz, syaw
                            snapped_egos[idx] = pose

                        # Drop obstacles that — after the ego lane-snap moved
                        # the egos a metre or two — now sit on top of an ego.
                        ego_anchors = list(snapped_egos.values())
                        spawn_obstacles = []
                        for ob in advanced_obstacles:
                            kind = ob.get("kind", "")
                            min_to_ego = 4.0 if kind in ("npc", "bicycle") else 1.5
                            if any(
                                math.hypot(ob["x"] - p["x"], ob["y"] - p["y"]) < min_to_ego
                                for p in ego_anchors
                            ):
                                continue
                            spawn_obstacles.append(ob)

                        for ob in spawn_obstacles:
                            bp = _bp_for_kind(bplib, ob["kind"], ob.get("model", ""))
                            if bp is None:
                                continue
                            transform = carla.Transform(
                                carla.Location(x=ob["x"], y=ob["y"], z=ob["z"]),
                                carla.Rotation(pitch=0.0, yaw=ob["yaw"], roll=0.0),
                            )
                            actor = _safe_spawn(world, bp, transform)
                            if actor is not None:
                                spawned.append(actor)

                        for idx, pose in snapped_egos.items():
                            color = ego_color_for(idx)
                            ego_bp = _ego_blueprint(bplib, color)
                            if ego_bp is None:
                                break
                            transform = carla.Transform(
                                carla.Location(x=pose["x"], y=pose["y"], z=pose["z"]),
                                carla.Rotation(pitch=0.0, yaw=pose["yaw"], roll=0.0),
                            )
                            actor = _safe_spawn(world, ego_bp, transform)
                            if actor is not None:
                                spawned.append(actor)
                                if is_night:
                                    _enable_vehicle_lights(actor)

                        for _ in range(4):
                            world.tick()

                        rgb = capture_oblique(
                            world, placed, plan.obstacles,
                            width=args.width, height=args.height, fov=args.fov,
                        )
                        Image.fromarray(rgb).save(out_path)
                        n_variants_done += 1
                    finally:
                        _destroy_cycle(world, spawned, client=client, ticks=4)

                # ── Top-down BEV (clean and route-overlay) ────────────────
                if args.overwrite or not (bev_path.exists() and routes_path.exists()):
                    advanced_obstacles_r = _filter_obstacle_overlaps([
                        _obstacle_pose_at_fraction(ob, PLACEMENT_VARIANTS[1]["nominal_frac"])
                        for ob in plan.obstacles
                    ])
                    advanced_obstacles_r = _apply_obstacle_overrides(
                        plan.scenario_id, advanced_obstacles_r, "bev",
                    )
                    placed_routes, _ = place_egos(
                        plan.ego_routes, advanced_obstacles_r,
                        nominal_frac=PLACEMENT_VARIANTS[1]["nominal_frac"],
                        min_ego_dist=args.min_spacing,
                        pre_obstacle_offset_m=PLACEMENT_VARIANTS[1]["pre_obstacle_offset_m"],
                        pre_interaction_offset_m=PLACEMENT_VARIANTS[1]["pre_interaction_offset_m"],
                    )
                    spawned_r: list = []
                    try:
                        snapped_egos_r: dict[int, dict] = {}
                        for idx, pose in placed_routes.items():
                            sx, sy, sz, syaw = _snap_to_lane(
                                world_map, pose["x"], pose["y"], pose["z"], pose["yaw"],
                            )
                            pose["x"], pose["y"], pose["z"], pose["yaw"] = sx, sy, sz, syaw
                            snapped_egos_r[idx] = pose

                        ego_anchors_r = list(snapped_egos_r.values())
                        spawn_obstacles_r = []
                        for ob in advanced_obstacles_r:
                            kind = ob.get("kind", "")
                            min_to_ego = 4.0 if kind in ("npc", "bicycle") else 1.5
                            if any(
                                math.hypot(ob["x"] - p["x"], ob["y"] - p["y"]) < min_to_ego
                                for p in ego_anchors_r
                            ):
                                continue
                            spawn_obstacles_r.append(ob)

                        for ob in spawn_obstacles_r:
                            bp = _bp_for_kind(bplib, ob["kind"], ob.get("model", ""))
                            if bp is None:
                                continue
                            transform = carla.Transform(
                                carla.Location(x=ob["x"], y=ob["y"], z=ob["z"]),
                                carla.Rotation(pitch=0.0, yaw=ob["yaw"], roll=0.0),
                            )
                            actor = _safe_spawn(world, bp, transform)
                            if actor is not None:
                                spawned_r.append(actor)
                        for idx, pose in snapped_egos_r.items():
                            color = ego_color_for(idx)
                            ego_bp = _ego_blueprint(bplib, color)
                            if ego_bp is None:
                                break
                            transform = carla.Transform(
                                carla.Location(x=pose["x"], y=pose["y"], z=pose["z"]),
                                carla.Rotation(pitch=0.0, yaw=pose["yaw"], roll=0.0),
                            )
                            actor = _safe_spawn(world, ego_bp, transform)
                            if actor is not None:
                                spawned_r.append(actor)
                                if is_night:
                                    _enable_vehicle_lights(actor)

                        if placed_routes:
                            ego_positions = list(placed_routes.values())
                            near_obs = [
                                ob for ob in plan.obstacles
                                if any(
                                    math.hypot(ob["x"] - p["x"], ob["y"] - p["y"]) <= 10.0
                                    for p in ego_positions
                                )
                            ]
                            cx, cy, half = _camera_bbox(ego_positions + near_obs)
                            margin = max(half + 6.0, 12.0)

                            # Settle once before capturing the clean BEV
                            # so no leftover debug primitives remain.
                            for _ in range(3):
                                world.tick()
                            if args.overwrite or not bev_path.exists():
                                rgb_clean, _cam_c = capture_overhead(
                                    world, cx, cy, margin,
                                    width=args.width, height=args.height,
                                    fov=90.0,
                                )
                                Image.fromarray(rgb_clean).save(bev_path)

                            if args.overwrite or not routes_path.exists():
                                # Re-draw the debug polylines every tick so
                                # they are guaranteed present in the
                                # captured frame (CARLA's debug primitives
                                # are short-lived).
                                for _ in range(3):
                                    draw_route_debug(
                                        world, plan.ego_routes, life_time=0.5,
                                    )
                                    world.tick()
                                draw_route_debug(
                                    world, plan.ego_routes, life_time=0.5,
                                )
                                rgb_routes, _cam_r = capture_overhead(
                                    world, cx, cy, margin,
                                    width=args.width, height=args.height,
                                    fov=90.0,
                                )
                                Image.fromarray(rgb_routes).save(routes_path)
                    finally:
                        _destroy_cycle(world, spawned_r, client=client, ticks=4)

                if n_variants_done >= 1:
                    success += 1
                else:
                    failures.append((plan.bucket, plan.scenario_id))

                print(f"  [{j}/{len(town_plans)}] {plan.bucket}/{plan.scenario_id}: "
                      f"variants={n_variants_done}/{len(PLACEMENT_VARIANTS)} "
                      f"failed={n_variants_failed} -> {stem}*", flush=True)
            except RuntimeError as exc:
                # CARLA timeout — most likely the simulator died on us. Try
                # to reconnect; if successful, push on with the next
                # scenario. If not, record the failure and let the rest of
                # the sweep continue (it will keep retrying every loop).
                msg = str(exc)
                print(f"  [{j}/{len(town_plans)}] {plan.bucket}/{plan.scenario_id}: "
                      f"FAILED ({type(exc).__name__}: {msg[:120]})", flush=True)
                failures.append((plan.bucket, plan.scenario_id))
                if "time-out" in msg or "simulator" in msg:
                    print("  [reconnect] attempting to re-acquire CARLA world ...",
                          flush=True)
                    try:
                        client.set_timeout(120.0)
                        world = client.get_world()
                        sync_settings = carla.WorldSettings()
                        sync_settings.synchronous_mode = True
                        sync_settings.fixed_delta_seconds = 0.05
                        if hasattr(sync_settings, "no_rendering_mode"):
                            sync_settings.no_rendering_mode = False
                        world.apply_settings(sync_settings)
                        bplib = world.get_blueprint_library()
                        last_weather = None
                        print("  [reconnect] OK", flush=True)
                    except Exception as exc2:
                        print(f"  [reconnect] FAILED ({type(exc2).__name__}: "
                              f"{str(exc2)[:120]})", flush=True)
            except Exception as exc:
                print(f"  [{j}/{len(town_plans)}] {plan.bucket}/{plan.scenario_id}: "
                      f"FAILED ({type(exc).__name__}: {exc})", flush=True)
                failures.append((plan.bucket, plan.scenario_id))

    # Restore original world settings.
    try:
        world.apply_settings(orig_settings)
    except Exception:
        pass

    print(f"\nDone. success={success} skipped_existing={skipped_existing} "
          f"failed={len(failures)}")
    if failures:
        print("Failures:")
        for b, s in failures[:20]:
            print(f"  {b}/{s}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")


if __name__ == "__main__":
    main()
