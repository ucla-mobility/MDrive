"""GRP route-quality inspector with three pipelines.

Connects to a running CARLA instance, loads each scenario's town, and traces
each ego's waypoint sequence through one or more GRP pipelines so we can
distinguish "GRP is broken" vs "this specific pipeline broke the route" vs
"the runtime would actually drive this poorly".

Three pipelines (modes):

  raw             — `grp.trace_route` pairwise on the raw XML waypoints. No
                    preprocessing. Useful as a baseline to see how raw GRP
                    behaves on the input as written.

  builder_legacy  — exactly what the scenario_builder_legacy "Run GRP" UI
                    button computes: `refine_waypoints_dp` (DP-based snap-to-
                    map, scoring deviation + GRP route cost + turn / shape
                    penalties) followed by leaderboard `interpolate_trajectory`
                    on the original waypoints. The visualization shows BOTH
                    the dense GRP trace (line) and the DP-aligned waypoints
                    (squares) — the same two layers the UI overlays.

  runtime         — exactly what `point_coordinates.json` contains at eval
                    time: route_scenario._align_start_waypoints (snap only
                    the first waypoint, yaw-aware) followed by leaderboard
                    `interpolate_trajectory` on the resulting trajectory.
                    No DP snap; the loop-guard inside interpolate_trajectory
                    is the only correction.

By default UCLA v2 smoothing is FORCED OFF in modes 2 and 3 (it auto-skips on
non-UCLA towns anyway, and the user has flagged it as harmful when it does
fire). Use --ucla-smoothing to mirror the actual current runtime behavior.

Usage
-----
1. Start a dedicated CARLA on a free port (so it doesn't interfere with
   the running evaluation):

       cd /data2/marco/CoLMDriver/carla912
       ./CarlaUE4.sh -world-port=9000 -RenderOffScreen &

2. Run the inspector (b2d_zoo conda env has carla 0.9.12 + numpy):

       conda activate b2d_zoo
       python tools/grp_inspector.py \
           --carla-port 9000 \
           --scenario-glob 'interaction/*/*' \
           --mode all \
           --visualize \
           --output grp_viz/grp_inspection.json \
           --vis-dir grp_vis

3. Inspect the output JSON + per-mode PNGs.  With --mode all you get
   3 PNGs per scenario in grp_vis/{raw,builder_legacy,runtime}/.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

# CARLA + GRP imports are deferred so --help works without carla installed.
# The ``agents.navigation.*`` modules ship with CARLA's PythonAPI but aren't
# installed into site-packages — we have to add the PythonAPI/carla directory
# to sys.path before importing them.  CARLA_ROOT defaults to the repo's bundled
# carla912/.
def _import_carla():
    import sys as _sys
    candidates = []
    env_root = os.environ.get("CARLA_ROOT")
    if env_root:
        candidates.append(os.path.join(env_root, "PythonAPI", "carla"))
    candidates.append(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "carla912", "PythonAPI", "carla",
    ))
    for c in candidates:
        if os.path.isdir(os.path.join(c, "agents")) and c not in _sys.path:
            _sys.path.insert(0, c)
    import carla  # noqa: WPS433
    try:
        from agents.navigation.global_route_planner import GlobalRoutePlanner  # type: ignore
    except ImportError as exc:
        raise ImportError(
            f"Could not import agents.navigation.global_route_planner.  "
            f"Tried PYTHONPATH additions: {candidates}.  "
            f"Set CARLA_ROOT to your CARLA install (one with PythonAPI/carla/agents).  "
            f"Underlying error: {exc}"
        )
    return carla, GlobalRoutePlanner


# ---------------------------------------------------------------------------
# XML / waypoint parsing
# ---------------------------------------------------------------------------

import xml.etree.ElementTree as _ET


def _load_tree(xml_path: str):
    if not os.path.isfile(xml_path):
        return None
    try:
        return _ET.parse(xml_path)
    except Exception:
        return None


def parse_xml_waypoints(xml_path: str) -> list[dict]:
    """Parse <waypoint> elements regardless of attribute order."""
    tree = _load_tree(xml_path)
    if tree is None:
        return []
    wps: list[dict] = []
    for wp in tree.getroot().iter("waypoint"):
        try:
            wps.append({
                "x": float(wp.get("x", "0")),
                "y": float(wp.get("y", "0")),
                "z": float(wp.get("z", "0") or "0"),
                "yaw": float(wp.get("yaw", "0")),
            })
        except (TypeError, ValueError):
            continue
    return wps


def get_town(xml_path: str) -> str | None:
    tree = _load_tree(xml_path)
    if tree is None:
        return None
    for route in tree.getroot().iter("route"):
        t = route.get("town")
        if t:
            return t
    return None


def is_ego_xml(xml_path: str) -> bool:
    tree = _load_tree(xml_path)
    if tree is None:
        return False
    for route in tree.getroot().iter("route"):
        if route.get("role") == "ego":
            return True
    return False


# ---------------------------------------------------------------------------
# Repo-path helpers (so we can import scenario_generator + leaderboard modules)
# ---------------------------------------------------------------------------

def _ensure_repo_paths() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    leaderboard = repo_root / "simulation" / "leaderboard"
    for p in (str(repo_root), str(leaderboard)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _import_align_mod():
    _ensure_repo_paths()
    from scenario_generator.pipeline.step_07_route_alignment import main as align_mod
    return align_mod


def _import_interpolate_trajectory():
    _ensure_repo_paths()
    from leaderboard.utils.route_manipulation import interpolate_trajectory
    return interpolate_trajectory


# ---------------------------------------------------------------------------
# GRP-result extraction (Waypoint or Transform → dict)
# ---------------------------------------------------------------------------

def _entry_to_dict(entry, road_option=None) -> dict | None:
    """Convert a (Waypoint|Transform, RoadOption) entry to {x,y,z,yaw,road_option}.

    CARLA 0.9.12 has Waypoint.transform as a callable property, and Transform's
    own .location/.rotation are non-callable, so we probe both shapes.
    """
    obj = entry
    if isinstance(entry, tuple) and len(entry) >= 1:
        obj = entry[0]
        if road_option is None and len(entry) >= 2:
            road_option = entry[1]
    loc = getattr(obj, "location", None)
    rot = getattr(obj, "rotation", None)
    if loc is None or rot is None or callable(loc) or callable(rot):
        tf = getattr(obj, "transform", None)
        if tf is None:
            return None
        if callable(tf):
            try:
                tf = tf()
            except Exception:
                return None
        loc = getattr(tf, "location", None)
        rot = getattr(tf, "rotation", None)
        if loc is None or rot is None:
            return None
    return {
        "x": float(loc.x),
        "y": float(loc.y),
        "z": float(loc.z),
        "yaw": float(rot.yaw),
        "road_option": str(road_option).split(".")[-1] if road_option is not None else None,
    }


# ---------------------------------------------------------------------------
# UCLA-smoothing override (monkey-patches GlobalRoutePlanner class temporarily)
# ---------------------------------------------------------------------------

class _NoUclaSmoothing:
    """Context manager that forces enable_ucla_v2_smoothing=False on any
    GlobalRoutePlanner instance's postprocess_route_trace call. Active only
    while inside the with-block.
    """
    def __init__(self, grp_class):
        self._cls = grp_class
        self._orig = None

    def __enter__(self):
        if not hasattr(self._cls, "postprocess_route_trace"):
            return self
        self._orig = self._cls.postprocess_route_trace
        orig = self._orig

        def patched(self_grp, *args, **kwargs):
            kwargs["enable_ucla_v2_smoothing"] = False
            return orig(self_grp, *args, **kwargs)
        self._cls.postprocess_route_trace = patched
        return self

    def __exit__(self, *_a):
        if self._orig is not None:
            self._cls.postprocess_route_trace = self._orig


# ---------------------------------------------------------------------------
# Mode 1: raw GRP — pairwise grp.trace_route on the XML waypoints, no prep
# ---------------------------------------------------------------------------

def trace_raw(carla_mod, grp, waypoints: list[dict]) -> list[dict]:
    """Run GRP between consecutive XML waypoints, return the dense trace.

    Each output point: x, y, z, yaw, road_option, or {"error":..,"segment":i}
    for failed segments.
    """
    out: list[dict] = []
    if len(waypoints) < 2:
        return out
    for i in range(len(waypoints) - 1):
        a = carla_mod.Location(x=waypoints[i]["x"], y=waypoints[i]["y"], z=waypoints[i]["z"])
        b = carla_mod.Location(x=waypoints[i + 1]["x"], y=waypoints[i + 1]["y"], z=waypoints[i + 1]["z"])
        try:
            seg = grp.trace_route(a, b)
        except Exception as exc:  # pylint: disable=broad-except
            out.append({"error": str(exc), "segment": i})
            continue
        for wp_obj, road_option in seg:
            d = _entry_to_dict(wp_obj, road_option)
            if d is not None:
                out.append(d)
    return out


# ---------------------------------------------------------------------------
# Mode 2: builder_legacy — refine_waypoints_dp + interpolate_trajectory
# ---------------------------------------------------------------------------

def _grp_route_recursive(carla_module, carla_map, grp, a_loc, b_loc,
                         max_detour: float = 2.0,
                         depth: int = 0,
                         max_depth: int = 4) -> list:
    """grp.trace_route with recursive midpoint fallback.

    Some CARLA road graphs (e.g. Town02 Major_Minor intersections) return
    pathological 5-10× detours for sparse 2-waypoint chords because the lane
    connectivity at the junction isn't represented as a direct connector.
    When detour > max_detour, we snap the chord midpoint to a drivable lane
    and recursively trace each half. Splitting the chord into shorter pieces
    forces grp.trace_route to use local junction connectors instead of
    routing the long way around.

    Returns the stitched route (list of (Waypoint, RoadOption)). If
    recursion fails, returns whatever grp.trace_route gave us (which might
    be a detour) or an empty list.
    """
    import math as _math
    chord = _math.hypot(b_loc.x - a_loc.x, b_loc.y - a_loc.y)
    if chord < 1.0:
        return []

    try:
        route = grp.trace_route(a_loc, b_loc)
    except Exception:
        route = []

    # Compute trace length
    def _trace_len(rt):
        total = 0.0
        prev = None
        for wp, _ in rt:
            loc = wp.transform.location if hasattr(wp, "transform") else wp.location
            if callable(loc):
                continue
            if prev is not None:
                total += _math.hypot(loc.x - prev[0], loc.y - prev[1])
            prev = (loc.x, loc.y)
        return total

    trace_len = _trace_len(route) if route else 0.0
    if route and trace_len <= chord * max_detour:
        return route

    # Detour too big — recurse via midpoint snapped to nearest drivable lane
    if depth >= max_depth:
        return route  # Give up, return whatever we got
    mid_x = (a_loc.x + b_loc.x) / 2.0
    mid_y = (a_loc.y + b_loc.y) / 2.0
    mid_z = (a_loc.z + b_loc.z) / 2.0
    try:
        mid_wp = carla_map.get_waypoint(
            carla_module.Location(x=mid_x, y=mid_y, z=mid_z),
            project_to_road=True,
            lane_type=carla_module.LaneType.Driving,
        )
    except Exception:
        mid_wp = None
    if mid_wp is None:
        return route
    mid_loc = carla_module.Location(
        x=float(mid_wp.transform.location.x),
        y=float(mid_wp.transform.location.y),
        z=float(mid_wp.transform.location.z),
    )
    if _math.hypot(mid_loc.x - a_loc.x, mid_loc.y - a_loc.y) < 0.5 or \
       _math.hypot(mid_loc.x - b_loc.x, mid_loc.y - b_loc.y) < 0.5:
        # Midpoint is essentially the same as an endpoint — recursion can't
        # progress; return the original route.
        return route

    left = _grp_route_recursive(carla_module, carla_map, grp, a_loc, mid_loc,
                                 max_detour, depth + 1, max_depth)
    right = _grp_route_recursive(carla_module, carla_map, grp, mid_loc, b_loc,
                                  max_detour, depth + 1, max_depth)
    if not left:
        return right
    if not right:
        return left
    return left + right[1:]  # Drop duplicate boundary point


def _densify_dp_inputs(waypoints: list[dict], max_gap_m: float,
                       carla_module=None, carla_map=None, grp=None,
                       large_gap_m: float = 10.0,
                       max_grp_detour: float = 4.0) -> tuple[list[dict], int]:
    """Densify XML waypoints so no two consecutive points are >max_gap_m apart.

    Two strategies:
    - For small gaps (≤large_gap_m): linear interpolation along the chord.
    - For LARGE gaps (>large_gap_m): use grp.trace_route(a, b) to fill in a
      lane-following path. Critical for sparse XMLs where a single 30-50 m
      chord otherwise cuts diagonally through off-road space (e.g. across a
      roundabout interior). With the lane-following fill, the DP downstream
      gets dense on-lane inputs and produces a coherent route.

    Falls back to linear interpolation for the large-gap case if grp is None
    or if grp.trace_route returns a detour > max_grp_detour × chord length.

    Returns (densified, n_inserted).
    """
    if max_gap_m <= 0 or len(waypoints) < 2:
        return list(waypoints), 0
    out: list[dict] = [waypoints[0]]
    inserted = 0
    for i in range(1, len(waypoints)):
        p, q = waypoints[i - 1], waypoints[i]
        dx = q["x"] - p["x"]
        dy = q["y"] - p["y"]
        dz = q.get("z", 0.0) - p.get("z", 0.0)
        d = math.hypot(dx, dy)
        used_grp = False
        if d > large_gap_m and carla_module is not None and grp is not None and carla_map is not None:
            a_loc = carla_module.Location(x=p["x"], y=p["y"], z=p.get("z", 0.0))
            b_loc = carla_module.Location(x=q["x"], y=q["y"], z=q.get("z", 0.0))
            # Use recursive grp.trace_route that handles long-detour
            # pathologies by inserting lane-snapped midpoints. This fixes
            # the bee-line issue where Town02 / Town05 sparse-input chords
            # otherwise return 5-10× detours that get rejected.
            # max_detour=4.0 matches the outer max_grp_detour budget — a
            # legitimate full-roundabout traversal can be ~π× chord (3.14x)
            # and we want to accept it. Recursion via lane-snapped midpoint
            # only kicks in for truly pathological detours (>4× chord).
            route = _grp_route_recursive(
                carla_module, carla_map, grp, a_loc, b_loc,
                max_detour=4.0, max_depth=4,
            )
            if route and len(route) >= 3:
                pts = []
                trace_len = 0.0
                prev_xy = None
                for entry in route:
                    rd = _entry_to_dict(entry)
                    if rd is None:
                        continue
                    xy = (rd["x"], rd["y"])
                    if prev_xy is not None:
                        trace_len += math.hypot(xy[0] - prev_xy[0], xy[1] - prev_xy[1])
                    prev_xy = xy
                    pts.append(rd)
                if pts and trace_len < d * max_grp_detour:
                    # Use ALL grp waypoints, no subsampling. grp samples lane
                    # connector splines at hop_resolution (2 m) which is
                    # exactly the density needed to follow tight junction
                    # turn-lanes faithfully. Subsampling at coarser spacing
                    # (3 m) was dropping intermediate connector waypoints,
                    # producing the "bee-line" diagonal cut visible at left-
                    # turn corners (e.g. Major_Minor/10 ego_2 cutting from
                    # (124,241) directly to (130,236) instead of arcing
                    # through the (127,241) → (132,240) → (135,236) corner.
                    for j in range(1, len(pts) - 1):
                        out.append({
                            "x": pts[j]["x"],
                            "y": pts[j]["y"],
                            "z": pts[j].get("z", 0.0),
                            "yaw": pts[j].get("yaw", 0.0),
                        })
                        inserted += 1
                    used_grp = True

        if not used_grp and d > max_gap_m:
            n_splits = max(2, int(math.ceil(d / max_gap_m)))
            for j in range(1, n_splits):
                t = j / n_splits
                out.append({
                    "x": p["x"] + dx * t,
                    "y": p["y"] + dy * t,
                    "z": p.get("z", 0.0) + dz * t,
                    "yaw": q.get("yaw", 0.0),
                })
                inserted += 1
        out.append(q)
    return out, inserted


def _stitch_via_grp(carla_module, grp, aligned_wps: list[dict],
                    max_detour_factor: float = 1.8) -> tuple[list[dict], dict]:
    """Build a dense route by stitching grp.trace_route segments between
    adjacent DP-aligned states. With disable_compression=True the states are
    ~3 m apart and on the lane graph, so grp.trace_route returns a short,
    dense local trace that follows lane-connector splines through junctions
    (smooth) and stays on the drivable surface (on-road).

    Per-segment loop guard: if grp returns a route longer than
    max_detour_factor × chord, that segment fell into a wrong-direction
    detour — fall back to the direct chord for THAT segment only.

    Returns (dense_route, stats).
    """
    out: list[dict] = []
    stats = {"n_grp_used": 0, "n_chord_used": 0, "n_grp_empty": 0, "n_grp_detour": 0}
    if len(aligned_wps) < 2:
        return list(aligned_wps), stats

    out.append({**aligned_wps[0], "road_option": None})
    for i in range(len(aligned_wps) - 1):
        a = aligned_wps[i]
        b = aligned_wps[i + 1]
        chord_d = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
        if chord_d < 0.3:
            out.append({**b, "road_option": None})
            continue

        a_loc = carla_module.Location(x=a["x"], y=a["y"], z=a.get("z", 0.0))
        b_loc = carla_module.Location(x=b["x"], y=b["y"], z=b.get("z", 0.0))
        try:
            route = grp.trace_route(a_loc, b_loc)
        except Exception:
            route = None

        if not route or len(route) < 2:
            stats["n_grp_empty"] += 1
            out.append({**b, "road_option": None})
            stats["n_chord_used"] += 1
            continue

        # Extract dense points; track total trace length for the loop guard.
        trace_pts: list[dict] = []
        trace_len = 0.0
        prev_xy: tuple[float, float] | None = None
        for entry in route:
            d = _entry_to_dict(entry)
            if d is None:
                continue
            xy = (d["x"], d["y"])
            if prev_xy is not None:
                trace_len += math.hypot(xy[0] - prev_xy[0], xy[1] - prev_xy[1])
            prev_xy = xy
            trace_pts.append(d)

        if not trace_pts:
            stats["n_grp_empty"] += 1
            out.append({**b, "road_option": None})
            stats["n_chord_used"] += 1
            continue

        if trace_len > chord_d * max_detour_factor and chord_d > 1.0:
            stats["n_grp_detour"] += 1
            out.append({**b, "road_option": None})
            stats["n_chord_used"] += 1
            continue

        out.extend(trace_pts[1:])
        stats["n_grp_used"] += 1

    return out, stats


def _walk_lane_between_states(carla_module, carla_map, aligned_wps: list[dict],
                               step_m: float = 1.0,
                               max_walk_factor: float = 2.5) -> tuple[list[dict], int]:
    """Walk the CARLA lane network between consecutive DP states.

    From each state[i], snap to a Waypoint on the lane graph and step forward
    via wp.next(step_m), greedily picking the next-waypoint that's closest to
    state[i+1]. Continue until we're within step_m of state[i+1] (or hit the
    walk budget).

    This replaces the direct chord between DP states with an actual lane-
    centerline path. By construction every output point is on a drivable
    lane, and it follows the lane connector splines through junctions, so
    sharp lane-pinch corners get the smooth geometry CARLA's OpenDRIVE
    actually has.

    Falls back to a direct copy of state[i+1] if the lane walk can't reach it
    within max_walk_factor × chord. Returns (walked_path, n_pairs_walked).
    """
    if len(aligned_wps) < 2:
        return list(aligned_wps), 0

    out: list[dict] = []
    n_walked = 0

    for i in range(len(aligned_wps) - 1):
        a = aligned_wps[i]
        b = aligned_wps[i + 1]
        chord_d = math.hypot(b["x"] - a["x"], b["y"] - a["y"])

        # Snap a to a CARLA Waypoint on the driving lane network
        a_loc = carla_module.Location(x=a["x"], y=a["y"], z=a.get("z", 0.0))
        try:
            a_wp = carla_map.get_waypoint(a_loc, project_to_road=True,
                                           lane_type=carla_module.LaneType.Driving)
        except Exception:
            a_wp = None

        if i == 0:
            if a_wp is not None:
                out.append({
                    "x": float(a_wp.transform.location.x),
                    "y": float(a_wp.transform.location.y),
                    "z": float(a_wp.transform.location.z),
                    "yaw": float(a_wp.transform.rotation.yaw),
                    "road_option": None,
                })
            else:
                out.append(a)

        if a_wp is None or chord_d < 0.5:
            out.append(b)
            continue

        target_xy = (b["x"], b["y"])
        # Allow a generous walk budget — at junctions the lane connector is
        # often longer than the straight-line chord between DP states.
        max_walk = max(chord_d * max_walk_factor, 6.0)
        walked = 0.0
        cur_wp = a_wp
        # Visited set to avoid loops at complex junctions.
        visited: set = {(cur_wp.road_id, cur_wp.lane_id, round(cur_wp.s, 1))}
        reached = False

        while walked < max_walk:
            try:
                next_wps = cur_wp.next(step_m)
            except Exception:
                next_wps = []
            if not next_wps:
                break
            # Greedy: pick the next-wp closest to target b
            best = min(next_wps, key=lambda w: math.hypot(
                w.transform.location.x - target_xy[0],
                w.transform.location.y - target_xy[1]
            ))
            key = (best.road_id, best.lane_id, round(best.s, 1))
            if key in visited:
                # Pick any unvisited alternative; if none, give up.
                alts = [w for w in next_wps
                        if (w.road_id, w.lane_id, round(w.s, 1)) not in visited]
                if not alts:
                    break
                best = min(alts, key=lambda w: math.hypot(
                    w.transform.location.x - target_xy[0],
                    w.transform.location.y - target_xy[1]
                ))
                key = (best.road_id, best.lane_id, round(best.s, 1))
            visited.add(key)
            cur_wp = best
            walked += step_m

            cur_xy = (float(cur_wp.transform.location.x),
                      float(cur_wp.transform.location.y))
            d_to_target = math.hypot(cur_xy[0] - target_xy[0],
                                     cur_xy[1] - target_xy[1])
            out.append({
                "x": cur_xy[0], "y": cur_xy[1],
                "z": float(cur_wp.transform.location.z),
                "yaw": float(cur_wp.transform.rotation.yaw),
                "road_option": None,
            })
            if d_to_target < step_m * 1.5:
                reached = True
                break

        if not reached:
            # Walk timed out — append b as a final anchor (snapped to lane).
            try:
                b_wp = carla_map.get_waypoint(
                    carla_module.Location(x=b["x"], y=b["y"], z=b.get("z", 0.0)),
                    project_to_road=True, lane_type=carla_module.LaneType.Driving,
                )
                if b_wp is not None:
                    out.append({
                        "x": float(b_wp.transform.location.x),
                        "y": float(b_wp.transform.location.y),
                        "z": float(b_wp.transform.location.z),
                        "yaw": float(b_wp.transform.rotation.yaw),
                        "road_option": None,
                    })
                else:
                    out.append(b)
            except Exception:
                out.append(b)
        n_walked += 1
    return out, n_walked


def _clean_input_waypoints(carla_module, carla_map, waypoints: list[dict],
                           max_search_m: float = 25.0) -> tuple[list[dict], int]:
    """Re-snap input XML waypoints sitting on junction-interior connector
    lanes onto the nearest non-junction drivable lane (radial search).

    Some XML files contain waypoints placed inside roundabout / intersection
    interiors (e.g. a waypoint at radius 19 m on Town03's roundabout whose
    outer perimeter is at radius 22 m). `project_to_road=True` snaps to a
    junction connector centered right on the bad input, so the pipeline
    routes through the interior. Walking forward/backward from that
    connector via wp.next() typically lands on another short non-junction
    segment ALSO inside the interior — the wider main-road perimeter is
    only reachable by searching outward in 2D.

    Strategy: probe `carla_map.get_waypoint` at offset positions in a circle
    around the input, at increasing radii, and pick the closest projection
    that's NOT on a junction connector. This finds the nearest "real" road
    lane outside the junction interior.

    Returns (cleaned, n_resnapped).
    """
    out: list[dict] = []
    n_resnapped = 0
    radii = [3.0, 6.0, 10.0, 15.0, 20.0, 25.0]
    n_dirs = 16
    for w in waypoints:
        loc = carla_module.Location(x=w["x"], y=w["y"], z=w.get("z", 0.0))
        try:
            wp = carla_map.get_waypoint(loc, project_to_road=True,
                                         lane_type=carla_module.LaneType.Driving)
        except Exception:
            wp = None
        if wp is None or not getattr(wp, "is_junction", False):
            out.append(w)
            continue

        # Radial search for the nearest non-junction lane whose heading
        # ROUGHLY MATCHES the input's heading. The input yaw encodes the
        # intended driving direction; preferring heading-aligned lanes
        # filters out wrong-direction inner lanes that happen to be close
        # but go orthogonal to the route (e.g. Town03 roundabout's inner
        # through-lanes at radius 19m heading NE while the input describes
        # a WSW-going outer perimeter at radius 22m).
        input_yaw = float(w.get("yaw", 0.0))
        best = None
        best_score = float("inf")  # lower is better; combines distance and yaw mismatch
        for r in radii:
            if r > max_search_m:
                break
            candidates_at_r: list = []
            for k in range(n_dirs):
                theta = 2 * math.pi * k / n_dirs
                tx = w["x"] + r * math.cos(theta)
                ty = w["y"] + r * math.sin(theta)
                try:
                    cand = carla_map.get_waypoint(
                        carla_module.Location(x=tx, y=ty, z=w.get("z", 0.0)),
                        project_to_road=True,
                        lane_type=carla_module.LaneType.Driving,
                    )
                except Exception:
                    continue
                if cand is None or getattr(cand, "is_junction", False):
                    continue
                d = math.hypot(cand.transform.location.x - w["x"],
                               cand.transform.location.y - w["y"])
                cand_yaw = float(cand.transform.rotation.yaw)
                yaw_diff = abs((cand_yaw - input_yaw + 180.0) % 360.0 - 180.0)
                # Score = distance + 0.1·yaw_diff. A 30° heading mismatch
                # adds 3 m of effective penalty, enough to prefer the
                # heading-aligned outer lane over a wrong-direction inner
                # lane that's a few meters closer.
                score = d + 0.1 * yaw_diff
                candidates_at_r.append((score, d, cand))
            if candidates_at_r:
                candidates_at_r.sort(key=lambda t: t[0])
                top_score, top_dist, top_cand = candidates_at_r[0]
                if top_score < best_score:
                    best_score = top_score
                    best = top_cand
                    best_dist = top_dist
                # We continue widening to see if a farther but better-aligned
                # lane wins on score; stop once we exceed max_search_m or
                # if the best is already a strong match (<5° yaw diff).
                if best is not None and best_score < top_dist + 0.5:
                    break

        if best is not None and best_dist <= max_search_m:
            out.append({
                "x": float(best.transform.location.x),
                "y": float(best.transform.location.y),
                "z": float(best.transform.location.z),
                "yaw": float(best.transform.rotation.yaw),
            })
            n_resnapped += 1
        else:
            out.append(w)
    return out, n_resnapped


def _snap_to_drivable_lane(carla_module, carla_map, points: list[dict]) -> tuple[list[dict], int]:
    """Project each point onto the nearest drivable lane centerline.

    Used to repair geometric smoothing that pushed points off-road (e.g.
    spike relaxation cutting a 90° intersection corner across the sidewalk
    interior). Each point is replaced with carla_map.get_waypoint(loc,
    project_to_road=True, lane_type=Driving). Returns (snapped, n_moved)
    where n_moved counts how many points actually changed by ≥0.3 m.
    """
    out: list[dict] = []
    n_moved = 0
    for p in points:
        loc = carla_module.Location(x=p["x"], y=p["y"], z=p.get("z", 0.0))
        try:
            wp = carla_map.get_waypoint(loc, project_to_road=True,
                                         lane_type=carla_module.LaneType.Driving)
        except Exception:
            wp = None
        if wp is None:
            out.append(p)
            continue
        nx = float(wp.transform.location.x)
        ny = float(wp.transform.location.y)
        nz = float(wp.transform.location.z)
        if math.hypot(nx - p["x"], ny - p["y"]) >= 0.3:
            n_moved += 1
        out.append({
            "x": nx, "y": ny, "z": nz,
            "yaw": float(wp.transform.rotation.yaw),
            "road_option": p.get("road_option"),
        })
    return out, n_moved


def _on_road_stats(carla_module, carla_map, points: list[dict],
                   max_lane_dist_m: float = 1.75) -> dict:
    """Per-point distance to the nearest drivable-lane centerline.

    Returns dict with:
        on_road_pct       : % of points within max_lane_dist_m of a lane center
        max_off_road_m    : worst point's distance from the lane centerline
        n_off_road        : count of off-road points
        longest_off_road_run_pts : longest consecutive off-road stretch (in
                                    sample count; useful to detect "drove off
                                    road for an extended time" as the user
                                    described)
    """
    if not points:
        return {"on_road_pct": 100.0, "max_off_road_m": 0.0,
                "n_off_road": 0, "longest_off_road_run_pts": 0}
    on_road = 0
    max_off = 0.0
    n_off = 0
    longest_run = 0
    cur_run = 0
    for p in points:
        loc = carla_module.Location(x=p["x"], y=p["y"], z=p.get("z", 0.0))
        try:
            wp = carla_map.get_waypoint(loc, project_to_road=True,
                                         lane_type=carla_module.LaneType.Driving)
        except Exception:
            wp = None
        if wp is None:
            n_off += 1
            cur_run += 1
            longest_run = max(longest_run, cur_run)
            continue
        d = math.hypot(p["x"] - wp.transform.location.x,
                       p["y"] - wp.transform.location.y)
        if d > max_off:
            max_off = d
        if d <= max_lane_dist_m:
            on_road += 1
            cur_run = 0
        else:
            n_off += 1
            cur_run += 1
            longest_run = max(longest_run, cur_run)
    return {
        "on_road_pct": round(100.0 * on_road / len(points), 2),
        "max_off_road_m": round(max_off, 2),
        "n_off_road": n_off,
        "longest_off_road_run_pts": longest_run,
    }


def _relax_spikes(points: list[dict], max_yaw_jump_deg: float = 50.0,
                  max_passes: int = 10, relaxation: float = 0.4) -> tuple[list[dict], int]:
    """Iteratively pull spike points toward the chord between their neighbors.

    A "spike" is an interior point where the heading change between the two
    adjacent segments (i-1→i vs i→i+1) exceeds `max_yaw_jump_deg`. This is
    the geometric signature of either:
      • An unintended lane-change wiggle (typical pattern: +30° then -30°
        between segments = 60° change-of-change).
      • A sharp junction-transition spike where the DP picked states on
        non-adjacent lane segments at a junction.

    Each pass moves every spike point `relaxation` fraction of the way toward
    its neighbors' midpoint (0.0 = no move, 1.0 = full midpoint replacement).
    Multiple passes converge to a smooth path while preserving the overall
    route. Endpoints are never modified — the start/end positions are kept.

    Returns (relaxed_points, total_relaxations_applied).
    """
    if len(points) < 3 or relaxation <= 0:
        return list(points), 0
    pts = [dict(p) for p in points]
    total_relaxed = 0
    for _ in range(max_passes):
        changed = False
        for i in range(1, len(pts) - 1):
            ax, ay = pts[i - 1]["x"], pts[i - 1]["y"]
            bx, by = pts[i]["x"], pts[i]["y"]
            cx, cy = pts[i + 1]["x"], pts[i + 1]["y"]
            h1 = math.degrees(math.atan2(by - ay, bx - ax))
            h2 = math.degrees(math.atan2(cy - by, cx - bx))
            diff = abs((h2 - h1 + 180.0) % 360.0 - 180.0)
            if diff > max_yaw_jump_deg:
                mid_x = (ax + cx) / 2.0
                mid_y = (ay + cy) / 2.0
                az = pts[i - 1].get("z", 0.0)
                cz = pts[i + 1].get("z", 0.0)
                mid_z = (az + cz) / 2.0
                pts[i]["x"] = bx + relaxation * (mid_x - bx)
                pts[i]["y"] = by + relaxation * (mid_y - by)
                pts[i]["z"] = pts[i].get("z", 0.0) + relaxation * (mid_z - pts[i].get("z", 0.0))
                total_relaxed += 1
                changed = True
        if not changed:
            break
    return pts, total_relaxed


def _remove_hairpins(points: list[dict], max_yaw_jump_deg: float = 120.0,
                     max_passes: int = 5) -> tuple[list[dict], int]:
    """Iteratively drop polyline points whose surrounding heading change > threshold.

    A heading change > 120° between consecutive ~3m segments is physically
    impossible for a vehicle — it means the DP selected a wrong-direction
    lane waypoint at that index. Drop the offending point; the polyline
    closes up by connecting the neighbors directly. Repeat until no more
    hairpins (or max_passes reached, since one removal can expose another).
    Returns (cleaned, total_removed).
    """
    if len(points) < 3:
        return list(points), 0
    pts = list(points)
    total_removed = 0
    for _ in range(max_passes):
        if len(pts) < 3:
            break
        keep = [True] * len(pts)
        for i in range(1, len(pts) - 1):
            ax, ay = pts[i - 1]["x"], pts[i - 1]["y"]
            bx, by = pts[i]["x"], pts[i]["y"]
            cx, cy = pts[i + 1]["x"], pts[i + 1]["y"]
            h1 = math.degrees(math.atan2(by - ay, bx - ax))
            h2 = math.degrees(math.atan2(cy - by, cx - bx))
            diff = abs((h1 - h2 + 180.0) % 360.0 - 180.0)
            if diff > max_yaw_jump_deg:
                keep[i] = False
        n_removed = sum(1 for k in keep if not k)
        if n_removed == 0:
            break
        pts = [p for p, k in zip(pts, keep) if k]
        total_removed += n_removed
    return pts, total_removed


def _bounded_moving_avg_smooth(points: list[dict], window: int = 5,
                                max_displacement_m: float = 0.5) -> list[dict]:
    """Apply a windowed moving-average to (x, y), but cap each point's
    displacement at max_displacement_m to prevent the smoothing from
    pushing a point off-road. Used to flatten heading discontinuities at
    grp_segment boundaries without breaking on-road compliance.
    """
    if len(points) < window or window < 2:
        return points
    half = window // 2
    out: list[dict] = []
    for i in range(len(points)):
        lo = max(0, i - half)
        hi = min(len(points), i + half + 1)
        avg_x = sum(p["x"] for p in points[lo:hi]) / (hi - lo)
        avg_y = sum(p["y"] for p in points[lo:hi]) / (hi - lo)
        dx = avg_x - points[i]["x"]
        dy = avg_y - points[i]["y"]
        d = math.hypot(dx, dy)
        if d > max_displacement_m and d > 1e-6:
            scale = max_displacement_m / d
            avg_x = points[i]["x"] + dx * scale
            avg_y = points[i]["y"] + dy * scale
        out.append({**points[i], "x": avg_x, "y": avg_y})
    return out


def _chaikin_smooth(points: list[dict], iterations: int = 2) -> list[dict]:
    """Chaikin's corner-cutting: round sharp angles in a polyline.

    Each segment AB is replaced by two cut-points at 1/4 and 3/4 of the way.
    Endpoints are preserved. After N iterations the polyline has roughly
    2^N more points and is C1-smooth.

    Yaw and road_option are dropped on smoothed intermediate points (we only
    use this for visualization, not as actual route metadata).
    """
    if iterations <= 0 or len(points) < 3:
        return points
    pts = points
    for _ in range(iterations):
        if len(pts) < 3:
            break
        new_pts = [pts[0]]
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            ax, ay, az = a["x"], a["y"], a.get("z", 0.0)
            bx, by, bz = b["x"], b["y"], b.get("z", 0.0)
            p1 = {"x": 0.75 * ax + 0.25 * bx,
                  "y": 0.75 * ay + 0.25 * by,
                  "z": 0.75 * az + 0.25 * bz,
                  "yaw": 0.0, "road_option": None}
            p2 = {"x": 0.25 * ax + 0.75 * bx,
                  "y": 0.25 * ay + 0.75 * by,
                  "z": 0.25 * az + 0.75 * bz,
                  "yaw": 0.0, "road_option": None}
            # First iter: skip the cut closest to the endpoint to keep the
            # polyline anchored. Subsequent iters: emit both cuts.
            if i > 0:
                new_pts.append(p1)
            if i < len(pts) - 2:
                new_pts.append(p2)
        new_pts.append(pts[-1])
        pts = new_pts
    return pts


def _input_gap_stats(waypoints: list[dict]) -> dict:
    """Quick diagnostic: max / mean / >10m count of inter-waypoint distances."""
    if len(waypoints) < 2:
        return {"n": len(waypoints), "max_gap": 0.0, "mean_gap": 0.0, "gaps_gt_10m": 0}
    gaps = []
    for i in range(1, len(waypoints)):
        p, q = waypoints[i - 1], waypoints[i]
        gaps.append(math.hypot(q["x"] - p["x"], q["y"] - p["y"]))
    return {
        "n": len(waypoints),
        "max_gap": round(max(gaps), 2),
        "mean_gap": round(sum(gaps) / len(gaps), 2),
        "gaps_gt_10m": sum(1 for g in gaps if g > 10),
    }


def trace_builder_legacy(carla_mod, world, carla_map, grp, waypoints: list[dict],
                         hop_resolution_m: float, ucla_smoothing: bool,
                         GlobalRoutePlanner_cls,
                         dp_densify_m: float = 3.0,
                         dp_max_skip: int = 20,
                         dp_disable_compression: bool = False,
                         dp_line_from_states: bool = True,
                         dp_smooth_iters: int = 2,
                         dp_w_deviation: float = 0.5,
                         dp_lane_change_penalty: float = 100.0,
                         dp_radius: float = 3.0,
                         bypass_dp: bool = False) -> dict:
    """Reproduce what scenario_builder_legacy "Run GRP" actually shows.

    Empirically (verified against `.grp_cache.json` files: 17/17 entries with a
    populated `trace_source` use `aligned_plan_fallback`), the primary path
    `_build_followed_dense_route` always raises in CARLA 0.9.12 + leaderboard
    (`'Boost.Python.function' object has no attribute 'location'` — Waypoint.
    transform is callable, not a property). `_compute_grp_preview_locked`
    silently catches this and falls back to dense-points-from-DP-plan. The
    DP plan is `grp.trace_route` segments stitched between compressed,
    lane-snapped waypoint states selected by `refine_waypoints_dp` — clean
    by construction.

    So we skip the dead interpolate_trajectory branch and go straight to the
    DP plan. (The `world`, `hop_resolution_m`, `ucla_smoothing`, and
    `GlobalRoutePlanner_cls` parameters are kept for signature-compat in case
    we ever revive the interpolate path on a fixed CARLA.)

    Returns:
        {
            "dense_route":  [...]   # the polyline = stitched DP-plan segments
            "aligned_waypoints": [...]  # DP-refined waypoint squares
            "trace_source": "dp_plan"
            "error": "..." (only on refine_waypoints_dp failure)
        }
    """
    del world, hop_resolution_m, ucla_smoothing, GlobalRoutePlanner_cls  # see docstring
    import copy
    if len(waypoints) < 2:
        return {"dense_route": [], "aligned_waypoints": [],
                "trace_source": None, "error": "too_few_waypoints"}

    align_mod = _import_align_mod()

    # Pre-clean (DISABLED): re-snap any input waypoint that sits on a
    # junction-interior connector lane onto the nearest non-junction lane.
    # In testing this introduced more problems (hairpins, sharp turns)
    # than it solved — the heading-aware radial search couldn't reliably
    # distinguish "main road" from "junction connector" lanes. Keeping the
    # function around in case we revisit; for now we trust the input as-is
    # and rely on the DP + grp.trace_route densify to handle bad XMLs.
    n_input_resnapped = 0

    input_gap_stats = _input_gap_stats(waypoints)

    # ── Pre-densify inputs to ≤dp_densify_m. For small gaps use linear
    # interpolation; for >10 m gaps use grp.trace_route to fill in a
    # lane-following path so the DP doesn't end up routing a chord across
    # roundabout interiors / off-road areas.
    densified, n_inserted = _densify_dp_inputs(
        waypoints, dp_densify_m,
        carla_module=carla_mod, carla_map=carla_map, grp=grp,
        large_gap_m=5.0,
        max_grp_detour=4.0,
    )

    plan = []
    aligned_wps: list[dict] = []

    if bypass_dp:
        # Bypass: use the grp-densified inputs DIRECTLY as the aligned
        # waypoints. The densifier already returned a clean lane-following
        # path via grp.trace_route on every >5 m gap, and DP downstream was
        # corrupting that on left/right-turn corners (snapping to inner-lane
        # candidates and producing diagonal bee-lines through intersection
        # interiors). Bypassing it preserves the lane-network path.
        for w in densified:
            aligned_wps.append({
                "x": float(w["x"]),
                "y": float(w["y"]),
                "z": float(w.get("z", 0.0)),
                "yaw": float(w.get("yaw", 0.0)),
            })
    else:
        # ── DP snap-to-map → (aligned waypoints, dense plan) ──────────────
        try:
            wps_for_dp = copy.deepcopy(densified)
            for w in wps_for_dp:
                if "yaw" not in w:
                    w["yaw"] = 0.0
            aligned_wps_raw, plan = align_mod.refine_waypoints_dp(
                carla_map, wps_for_dp, grp,
                max_skip=dp_max_skip,
                disable_compression=dp_disable_compression,
                w_deviation=dp_w_deviation,
                lane_change_penalty=dp_lane_change_penalty,
                radius=dp_radius,
            )
        except Exception as exc:  # pylint: disable=broad-except
            return {"dense_route": [], "aligned_waypoints": [],
                    "trace_source": None,
                    "input_gap_stats": input_gap_stats,
                    "n_densified_inserted": n_inserted,
                    "error": f"refine_waypoints_dp_failed: {exc}"}

        for w in aligned_wps_raw:
            aligned_wps.append({
                "x": float(w["x"]),
                "y": float(w["y"]),
                "z": float(w.get("z", 0.0)),
                "yaw": float(w.get("yaw", 0.0)),
            })

    # When --bypass-dp is set, the densified input already came directly
    # from grp.trace_route on the lane network — moving it via relaxation
    # or lane-snapping would just push the path AWAY from where grp said
    # the lane center is. So we skip those steps and trust the input.
    n_states_relaxed = 0
    if not bypass_dp:
        # Spike relaxation: pull lane-jump / junction-spike states toward
        # the chord between their neighbors. Endpoints preserved. Two-pass:
        # 15° aggressive (kills short-lived lane-change wobbles), 20° for
        # general cleanup. Threshold preserves legitimate roundabout
        # curvature (~10-15° per 3 m chord on a r=15-20 m roundabout).
        aligned_wps, _n_relax_a = _relax_spikes(
            aligned_wps, max_yaw_jump_deg=15.0, max_passes=20, relaxation=0.4,
        )
        aligned_wps, _n_relax_b = _relax_spikes(
            aligned_wps, max_yaw_jump_deg=20.0, max_passes=10, relaxation=0.5,
        )
        n_states_relaxed = _n_relax_a + _n_relax_b

    # Snap and post-snap softening — only when DP was used. With
    # --bypass-dp the densified input is already on lane centerlines from
    # grp.trace_route, so no further snap/relax is needed and any movement
    # would just push the path away from the input waypoints.
    n_lane_snapped = 0
    if not bypass_dp:
        # Relaxation cuts corners, sometimes pushing states off the
        # drivable surface. Snap each back to the nearest drivable lane.
        aligned_wps, n_lane_snapped = _snap_to_drivable_lane(carla_mod, carla_map, aligned_wps)

        # Post-snap softening on the worst kinks (>50°), then re-snap.
        aligned_wps, _n_post_relax = _relax_spikes(
            aligned_wps, max_yaw_jump_deg=50.0, max_passes=3, relaxation=0.15,
        )
        n_states_relaxed += _n_post_relax
        aligned_wps, _n_post_snap = _snap_to_drivable_lane(carla_mod, carla_map, aligned_wps)
        n_lane_snapped += _n_post_snap

    # ── Build dense_route ─────────────────────────────────────────────────
    # Two options:
    # (a) `dp_line_from_states=True` (default): connect aligned_waypoints (DP
    #     states) directly. With --dp-densify-m=3.0 they're ~3m apart and form
    #     a clean polyline. AVOIDS the bug where grp.trace_route between
    #     adjacent DP states detours through junctions / wrong lanes,
    #     producing the 30-40m gaps and 100° yaw jumps the user observed.
    # (b) `dp_line_from_states=False`: use the original DP plan (stitched
    #     grp.trace_route segments). Matches scenario_builder_legacy "Run GRP"
    #     dashed line output exactly — including its bugs on roundabouts.
    dense_route: list[dict] = []
    grp_stitch_stats: dict = {}
    if dp_line_from_states:
        # Polyline directly through the (relaxed + lane-snapped) DP states.
        # Earlier I tried stitching via grp.trace_route between adjacent
        # states, but that caused detours and hairpins on dense well-formed
        # inputs (e.g. Major_Minor_9). The DP states themselves are already
        # lane-snapped truth — connecting them is the simplest robust thing.
        # Sparse-input cases (e.g. Major_Minor_6 with 2 waypoints) are now
        # handled upstream by `_densify_dp_inputs` using grp.trace_route on
        # any >10 m XML gap, so the DP receives lane-following inputs even
        # when the XML doesn't.
        for w in aligned_wps:
            dense_route.append({
                "x": w["x"], "y": w["y"], "z": w["z"], "yaw": w["yaw"],
                "road_option": None,
            })
        trace_source = "dp_states_polyline"
    else:
        # plan is list of (Waypoint, RoadOption); _entry_to_dict handles both
        # the property-style and callable-style .transform attribute.
        for entry in (plan or []):
            d = _entry_to_dict(entry)
            if d is not None:
                dense_route.append(d)
        trace_source = "dp_plan"

    # Hairpin filter at 120°: drop only physically-impossible reversals. We
    # used to use 80° here when the dense_route was grp-stitched (where
    # boundary kinks are sampling artifacts), but with the polyline-through-
    # DP-states approach a 60-80° heading change between samples can be a
    # legitimate sharp turn at a junction, and removing those points cuts
    # the corner.
    dense_route, n_hairpins = _remove_hairpins(dense_route, max_yaw_jump_deg=120.0)

    # Smooth the remaining (now-physical) corners with Chaikin so legitimate
    # sharp transitions (lane-change endpoints, junction tangent jumps) read
    # as gentle arcs without distorting the route's actual path.
    if dp_smooth_iters > 0 and len(dense_route) >= 3:
        dense_route = _chaikin_smooth(dense_route, iterations=dp_smooth_iters)

    # Final pass: TWO bounded-moving-average passes to flatten heading
    # discontinuities at junction lane-connector boundaries. Each pass is
    # capped at 0.5 m displacement (≈14% of a 3.5 m lane width), so two
    # passes stay within ~1 m of the lane center even in the worst case —
    # well inside the lane. The wider window (11) and double pass smooth
    # tighter junction kinks that a single window=7 pass left as visible
    # sharp angles in /3 (orange at (5,-22), blue at (15,12)).
    if len(dense_route) >= 5:
        dense_route = _bounded_moving_avg_smooth(
            dense_route, window=11, max_displacement_m=0.5,
        )
        dense_route = _bounded_moving_avg_smooth(
            dense_route, window=11, max_displacement_m=0.5,
        )

    on_road = _on_road_stats(carla_mod, carla_map, dense_route)

    return {
        "dense_route": dense_route,
        "aligned_waypoints": aligned_wps,
        "trace_source": trace_source,
        "input_gap_stats": input_gap_stats,
        "n_input_resnapped": n_input_resnapped,
        "n_densified_inserted": n_inserted,
        "n_smooth_iters": dp_smooth_iters,
        "n_hairpins_removed": n_hairpins,
        "n_states_relaxed": n_states_relaxed,
        "n_lane_snapped": n_lane_snapped,
        "grp_stitch_stats": grp_stitch_stats,
        **{f"on_road_{k}": v for k, v in on_road.items()},
    }


# ---------------------------------------------------------------------------
# Mode 3: runtime — _align_start_waypoints + interpolate_trajectory
# ---------------------------------------------------------------------------

def trace_runtime(carla_mod, world, carla_map, waypoints: list[dict],
                  hop_resolution_m: float, ucla_smoothing: bool,
                  GlobalRoutePlanner_cls) -> dict:
    """Match what `point_coordinates.json` contains at eval time.

    Replicates route_scenario._align_start_waypoints (yaw-aware first-wp snap
    only) followed by leaderboard.interpolate_trajectory.
    """
    if len(waypoints) < 2:
        return {"dense_route": [], "error": "too_few_waypoints"}

    interpolate_trajectory = _import_interpolate_trajectory()

    trajectory = [carla_mod.Location(x=w["x"], y=w["y"], z=w["z"]) for w in waypoints]
    _snap_start_waypoint_inplace(carla_mod, carla_map, trajectory, waypoints)

    dense_route: list[dict] = []
    ucla_ctx = _NoUclaSmoothing(GlobalRoutePlanner_cls) if not ucla_smoothing else _Noop()
    try:
        with ucla_ctx:
            _gps, route = interpolate_trajectory(world, trajectory, hop_resolution=hop_resolution_m)
        for entry in route:
            d = _entry_to_dict(entry)
            if d is not None:
                dense_route.append(d)
    except Exception as exc:  # pylint: disable=broad-except
        return {"dense_route": [], "error": f"interpolate_trajectory_failed: {exc}"}
    return {"dense_route": dense_route}


class _Noop:
    def __enter__(self):
        return self
    def __exit__(self, *_a):
        return False


# ---------------------------------------------------------------------------
# Start-waypoint snap (replicates _align_preview_start_waypoint /
# _align_start_waypoints — they are line-for-line equivalent)
# ---------------------------------------------------------------------------

def _snap_start_waypoint_inplace(carla_mod, carla_map, trajectory, waypoints: list[dict],
                                 max_snap_dist_m: float = 5.0) -> bool:
    """Snap trajectory[0] to nearest drivable lane whose yaw best matches the
    XML's heading. Mutates trajectory[0] in-place. Returns True on success.
    """
    if not trajectory or len(waypoints) < 2:
        return False

    xml_yaw = float(waypoints[0].get("yaw", 0.0)) if waypoints else None

    # XML "heading" inferred from first segment (in case yaw attr is missing/wrong)
    dx = float(waypoints[1]["x"]) - float(waypoints[0]["x"])
    dy = float(waypoints[1]["y"]) - float(waypoints[0]["y"])
    inferred_heading = math.degrees(math.atan2(dy, dx)) if (dx or dy) else None

    def _angle_delta(a, b):
        return abs((a - b + 180.0) % 360.0 - 180.0)

    desired_yaw = None
    if inferred_heading is not None:
        if xml_yaw is None or _angle_delta(inferred_heading, xml_yaw) > 45.0:
            desired_yaw = inferred_heading
        else:
            desired_yaw = xml_yaw
    elif xml_yaw is not None:
        desired_yaw = xml_yaw
    if desired_yaw is None:
        return False

    try:
        wp = carla_map.get_waypoint(
            trajectory[0],
            project_to_road=True,
            lane_type=carla_mod.LaneType.Driving,
        )
    except Exception:
        wp = None
    if wp is None:
        return False

    candidates = [wp]
    for fn in ("get_left_lane", "get_right_lane"):
        try:
            n = getattr(wp, fn)()
        except Exception:
            n = None
        if n is not None and getattr(n, "lane_type", None) == carla_mod.LaneType.Driving:
            candidates.append(n)

    best = min(candidates, key=lambda c: _angle_delta(desired_yaw, float(c.transform.rotation.yaw)))
    try:
        if best.transform.location.distance(trajectory[0]) > max_snap_dist_m:
            return False
    except Exception:
        return False
    trajectory[0] = carla_mod.Location(
        x=float(best.transform.location.x),
        y=float(best.transform.location.y),
        z=float(best.transform.location.z),
    )
    return True


def _resample_uniform(xs: list[float], ys: list[float], step_m: float) -> list[tuple[float, float]]:
    """Resample a polyline at uniform `step_m` arc-length intervals using linear
    interpolation. Returns list of (x, y). Helps decouple curvature analysis
    from the polyline's variable point spacing (e.g. after Chaikin smoothing).
    """
    if len(xs) < 2:
        return list(zip(xs, ys))
    sampled: list[tuple[float, float]] = [(xs[0], ys[0])]
    target = step_m
    seg_start_dist = 0.0
    for i in range(1, len(xs)):
        seg = math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])
        if seg <= 0:
            continue
        seg_end_dist = seg_start_dist + seg
        while target <= seg_end_dist:
            t = (target - seg_start_dist) / seg
            sampled.append((xs[i - 1] + t * (xs[i] - xs[i - 1]),
                            ys[i - 1] + t * (ys[i] - ys[i - 1])))
            target += step_m
        seg_start_dist = seg_end_dist
    if (sampled[-1][0], sampled[-1][1]) != (xs[-1], ys[-1]):
        sampled.append((xs[-1], ys[-1]))
    return sampled


def analyze_route(points: list[dict],
                  input_waypoints: list[dict] | None = None) -> dict | None:
    valid = [p for p in points if "error" not in p]
    if len(valid) < 2:
        return None
    xs = [p["x"] for p in valid]
    ys = [p["y"] for p in valid]
    dists = [math.hypot(xs[j] - xs[j - 1], ys[j] - ys[j - 1]) for j in range(1, len(xs))]
    # Compute yaw jumps from SEGMENT geometry (atan2 of dx/dy), not from the
    # `yaw` field. The yaw field is unreliable on smoothed/synthesized
    # polylines (e.g. Chaikin sets new yaws to 0), and what we actually care
    # about is how sharply the path turns physically.
    seg_headings = [math.degrees(math.atan2(ys[j] - ys[j - 1], xs[j] - xs[j - 1]))
                    for j in range(1, len(xs))]
    yaw_jumps: list[float] = []
    for j in range(1, len(seg_headings)):
        diff = seg_headings[j] - seg_headings[j - 1]
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        yaw_jumps.append(abs(diff))
    path_len = sum(dists)
    straight = math.hypot(xs[-1] - xs[0], ys[-1] - ys[0])

    # ── Physical curvature analysis on a uniformly-resampled (1 m) polyline.
    # This decouples the metric from variable point spacing produced by Chaikin
    # smoothing, and lets us answer two real questions:
    #   1. min_turn_radius_m — tightest curve along the path. <5 m means
    #      "physically impossible for a vehicle" (turn radius 5 m at low speed
    #      is at the edge of feasibility; <3 m would scrape the curb).
    #   2. wiggle metrics — count of curvature sign reversals per 100 m.
    #      A clean curve has ≤1 reversal (entry / exit). A lane change adds 2.
    #      A path that wanders side-to-side has many.
    sampled = _resample_uniform(xs, ys, step_m=1.0)
    min_turn_radius_m = float("inf")
    n_sign_changes = 0
    n_close_sign_changes = 0  # pairs within 5 m forward — typical lane-change shape
    if len(sampled) >= 3:
        # Per-meter signed heading change (radians/m == 1/radius).
        signed_kappa: list[float] = []
        for j in range(1, len(sampled) - 1):
            dx1 = sampled[j][0] - sampled[j - 1][0]
            dy1 = sampled[j][1] - sampled[j - 1][1]
            dx2 = sampled[j + 1][0] - sampled[j][0]
            dy2 = sampled[j + 1][1] - sampled[j][1]
            h1 = math.atan2(dy1, dx1)
            h2 = math.atan2(dy2, dx2)
            d = h2 - h1
            while d > math.pi:
                d -= 2 * math.pi
            while d < -math.pi:
                d += 2 * math.pi
            signed_kappa.append(d)  # over ~1 m so this is rad/m
        if signed_kappa:
            max_abs_kappa = max(abs(k) for k in signed_kappa)
            if max_abs_kappa > 1e-6:
                min_turn_radius_m = round(1.0 / max_abs_kappa, 2)
        # Count sign changes in signed_kappa — and pairs of sign changes that
        # occur within 5 forward meters of each other (the signature of a
        # lane-change wiggle: + then − within ~5 m). The 5°/m magnitude floor
        # is calibrated to ignore residual sub-degree jitter left over from
        # spike relaxation on otherwise-smooth paths; a real lane-change
        # wiggle peaks at 10-20°/m.
        last_sign = 0
        last_idx = -10**9
        sig_threshold = math.radians(5.0)
        for j, k in enumerate(signed_kappa):
            if abs(k) < sig_threshold:
                continue
            sign = 1 if k > 0 else -1
            if last_sign != 0 and sign != last_sign:
                n_sign_changes += 1
                if (j - last_idx) <= 5:
                    n_close_sign_changes += 1
            last_sign = sign
            last_idx = j
    sign_changes_per_100m = round(n_sign_changes / max(1.0, path_len / 100.0), 2)
    # Compare to input path length when available so roundabout / U-turn /
    # any-go-around route doesn't get falsely flagged as a detour: the
    # comparison answers "did the GRP add unnecessary distance vs. what the
    # XML already laid out?" (a value near 1.0 = clean; >>1.0 = real detour).
    input_path_len = None
    input_detour_ratio = None
    if input_waypoints and len(input_waypoints) >= 2:
        input_path_len = sum(
            math.hypot(input_waypoints[j]["x"] - input_waypoints[j - 1]["x"],
                       input_waypoints[j]["y"] - input_waypoints[j - 1]["y"])
            for j in range(1, len(input_waypoints))
        )
        if input_path_len > 0:
            input_detour_ratio = round(path_len / input_path_len, 3)
    return {
        "n_points": len(points),
        "n_valid": len(valid),
        "path_length": round(path_len, 2),
        "straight_distance": round(straight, 2),
        "detour_ratio": round(path_len / max(1.0, straight), 3),
        "input_path_length": round(input_path_len, 2) if input_path_len is not None else None,
        "input_detour_ratio": input_detour_ratio,
        "max_dist_jump": round(max(dists) if dists else 0, 2),
        "max_yaw_jump": round(max(yaw_jumps) if yaw_jumps else 0, 1),
        "big_jumps_5m": sum(1 for d in dists if d > 5),
        "big_jumps_10m": sum(1 for d in dists if d > 10),
        "big_yaw_jumps_30": sum(1 for y in yaw_jumps if y > 30),
        "big_yaw_jumps_90": sum(1 for y in yaw_jumps if y > 90),
        # New physical-feasibility metrics
        "min_turn_radius_m": min_turn_radius_m if min_turn_radius_m != float("inf") else None,
        "n_curvature_sign_changes": n_sign_changes,
        "n_close_sign_changes": n_close_sign_changes,
        "sign_changes_per_100m": sign_changes_per_100m,
    }


def classify_breakage(metrics: dict | None) -> dict:
    """Categorize what's wrong with a traced route. Empty / all-False = clean.

    Flags:
        unparseable         : metrics is None (couldn't trace at all)
        big_gap             : max_dist_jump > 10 m (consecutive samples leap)
        detour              : output is >30% longer than INPUT path length
                              (input_detour_ratio > 1.3). Compares to input
                              path length, not straight-line, so a legitimate
                              roundabout traversal isn't falsely flagged.
        too_tight_radius    : min_turn_radius_m < 5 m. 5 m is roughly the
                              minimum turning radius of a typical sedan;
                              below that, the curve can't be physically
                              executed.
        wiggle              : ≥2 curvature sign changes within 5 forward
                              meters at any point — the geometric signature of
                              an unintended lane change (left, then right).
        excess_wiggle       : sign_changes_per_100m > 4 — overall path
                              wanders side-to-side too often (e.g. multiple
                              lane changes during a single roundabout pass).
    """
    if not metrics:
        return {"unparseable": True}
    idr = metrics.get("input_detour_ratio")
    detour = (idr is not None and idr > 1.3) or (idr is None and metrics["detour_ratio"] > 1.8)
    min_r = metrics.get("min_turn_radius_m")
    too_tight = min_r is not None and min_r < 5.0
    n_close = metrics.get("n_close_sign_changes", 0)
    spc = metrics.get("sign_changes_per_100m", 0.0)
    max_yaw = metrics.get("max_yaw_jump", 0.0)
    flags = {
        "big_gap": metrics["max_dist_jump"] > 10.0,
        "detour": detour,
        "too_tight_radius": too_tight,
        "wiggle": n_close >= 1 and max_yaw > 20.0,
        "excess_wiggle": spc > 4.0 and max_yaw > 20.0,
    }
    return flags


def classify_breakage_with_road(metrics: dict | None,
                                on_road_pct: float | None,
                                max_off_road_m: float | None,
                                longest_off_road_run_pts: int | None) -> dict:
    """Add off-road flags on top of geometric breakage.

    off_road        : >5% of points >1.75 m from any drivable lane center, OR
                      worst point >5 m from a lane (drove fully off the road).
    sustained_off_road : ≥5 consecutive points off-road (since the dense_route
                      is sampled at 1m, this means an ≥5 m off-road stretch).
    """
    base = classify_breakage(metrics)
    if on_road_pct is None:
        return base
    base["off_road"] = (on_road_pct < 95.0) or (max_off_road_m is not None and max_off_road_m > 5.0)
    base["sustained_off_road"] = (longest_off_road_run_pts is not None
                                   and longest_off_road_run_pts >= 5)
    return base


def is_broken(metrics: dict | None) -> bool:
    flags = classify_breakage(metrics)
    return any(flags.values())


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def sample_lane_network(world_map, resolution_m: float = 2.0) -> tuple[list[float], list[float]]:
    """Sample every lane in the loaded map at fixed centerline spacing.

    Returns (xs, ys).  Used as a backdrop scatter so route deviations from the
    actual road graph are visible.
    """
    try:
        wps = world_map.generate_waypoints(resolution_m)
    except Exception:  # pylint: disable=broad-except
        return [], []
    xs: list[float] = []
    ys: list[float] = []
    for wp in wps:
        loc = wp.transform.location
        xs.append(loc.x)
        ys.append(loc.y)
    return xs, ys


def visualize_scenario(out_path: str, scenario: str, town: str,
                       egos: list[dict], any_orig: bool,
                       lane_xs: list[float] | None = None,
                       lane_ys: list[float] | None = None,
                       mode_label: str | None = None) -> None:
    """Plot every ego in a scenario on a shared axis, one color per ego.

    Each ego dict: {name, cur_wps, cur_route, orig_wps, orig_route, cur_metrics, orig_metrics}.
    Two panels (CURRENT XML / .ORIG) when any ego has orig data, otherwise one.

    Figure size adapts to the data's aspect ratio so wide-and-thin scenarios
    (e.g. straight roads) get a tall-enough panel to see lateral structure.
    Overlapping egos are differentiated by cycling linestyle + slight linewidth.
    """
    import matplotlib  # type: ignore
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    n_panels = 2 if any_orig else 1
    cmap = plt.get_cmap("tab10" if len(egos) <= 10 else "tab20")
    linestyles = ["-", "--", ":", "-."]
    linewidths = [2.6, 2.1, 1.7]

    # ------ collect every (x, y) we will draw, across both panels ------
    all_xs: list[float] = []
    all_ys: list[float] = []
    panels = ["cur"] + (["orig"] if any_orig else [])
    for which in panels:
        for ego in egos:
            for w in ego.get(f"{which}_wps") or []:
                all_xs.append(w["x"]); all_ys.append(w["y"])
            for p in ego.get(f"{which}_route") or []:
                if "error" not in p:
                    all_xs.append(p["x"]); all_ys.append(p["y"])

    # ------ derive panel size & limits from data extent ------
    # We deliberately do NOT enforce equal aspect: scenarios are usually
    # 50-200m long but only a few meters wide laterally, and we want the
    # lateral structure (e.g. parallel egos in different lanes) to be visible.
    # Instead we size the panel proportionally to data aspect with floors,
    # then stretch each axis independently to fill the panel.
    aspect_note = ""
    if all_xs and all_ys:
        xmin, xmax = min(all_xs), max(all_xs)
        ymin, ymax = min(all_ys), max(all_ys)
        dx_raw = max(xmax - xmin, 1.0)
        dy_raw = max(ymax - ymin, 1.0)
        # Per-axis padding (independent so a thin axis still gets breathing room)
        x_pad = max(0.05 * dx_raw, 2.0)
        y_pad = max(0.05 * dy_raw, 2.0)
        xmin -= x_pad; xmax += x_pad
        ymin -= y_pad; ymax += y_pad
        dx, dy = xmax - xmin, ymax - ymin
        # Panel dims: target 12 on the long axis; floor the short axis at 5
        # so very narrow scenarios still get readable height/width.
        target_long, min_short, max_short = 12.0, 5.0, 12.0
        if dx >= dy:
            panel_w = target_long
            panel_h = min(max_short, max(min_short, target_long * dy / dx))
        else:
            panel_h = target_long
            panel_w = min(max_short, max(min_short, target_long * dx / dy))
        fig_w = panel_w * n_panels + 0.6
        fig_h = panel_h + 2.6  # title + legend below
        xlim = (xmin, xmax)
        ylim = (ymin, ymax)
        # Note when axes are stretched non-uniformly so the viewer isn't misled
        scale_ratio = (dx / dy) if dx >= dy else (dy / dx)
        if scale_ratio > 2.0:
            aspect_note = f"   ⚠ axes stretched (1:{scale_ratio:.1f})"
    else:
        fig_w, fig_h = 9 * n_panels, 8
        xlim = ylim = None

    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h), squeeze=False)

    def _plot(ax, which: str, label: str):
        # Lane network underlay (cropped to the visible window for speed)
        if lane_xs and lane_ys and xlim is not None and ylim is not None:
            x0, x1 = xlim; y0, y1 = ylim
            in_view_x: list[float] = []
            in_view_y: list[float] = []
            for lx, ly in zip(lane_xs, lane_ys):
                if x0 <= lx <= x1 and y0 <= ly <= y1:
                    in_view_x.append(lx); in_view_y.append(ly)
            if in_view_x:
                ax.scatter(in_view_x, in_view_y, s=6, c="0.7",
                           alpha=0.7, zorder=0, marker=".",
                           label=f"lane network ({len(in_view_x)} pts)")
        for i, ego in enumerate(egos):
            wps = ego[f"{which}_wps"]
            route = ego[f"{which}_route"]
            if not wps:
                continue
            color = cmap(i % cmap.N)
            ls = linestyles[i % len(linestyles)]
            lw = linewidths[i % len(linewidths)]
            name = ego["name"]
            metrics = ego.get(f"{which}_metrics")
            if route:
                valid = [p for p in route if "error" not in p]
                rxs = [p["x"] for p in valid]
                rys = [p["y"] for p in valid]
                if metrics:
                    leg = (f"{name}: {metrics['path_length']:.0f}m, "
                           f"detour={metrics['detour_ratio']:.2f}x, "
                           f"max_jump={metrics['max_dist_jump']:.0f}m, "
                           f"max_yaw={metrics['max_yaw_jump']:.0f}°")
                else:
                    leg = name
                ax.plot(rxs, rys, linestyle=ls, color=color, alpha=0.85,
                        linewidth=lw, label=leg)
            wxs = [w["x"] for w in wps]
            wys = [w["y"] for w in wps]
            ax.plot(wxs, wys, "s", color=color, markersize=6, alpha=0.85,
                    markeredgecolor="black", markeredgewidth=0.5)
            ax.plot(wxs[0], wys[0], "o", color=color, markersize=10,
                    markeredgecolor="black", markeredgewidth=1.0)
            ax.plot(wxs[-1], wys[-1], "*", color=color, markersize=14,
                    markeredgecolor="black", markeredgewidth=0.8)
        ax.set_title(label, fontsize=11)
        # auto aspect — panel shape already encodes data aspect (see above)
        ax.set_aspect("auto", adjustable="box")
        if xlim is not None:
            ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.grid(alpha=0.3)
        # Legend below the plot so it doesn't occlude data
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
                  fontsize=7, ncol=1, frameon=True)

    panel_title = {
        "raw": "RAW GRP  (line=grp.trace_route, squares=XML waypoints)",
        "builder_legacy": "BUILDER LEGACY  (line=DP-plan, squares=DP-aligned waypoints — what 'Run GRP' shows)",
        "runtime": "RUNTIME (point_coordinates.json)  (line=interpolate_trajectory, squares=XML waypoints)",
    }.get(mode_label or "", "CURRENT XML")
    _plot(axes[0][0], "cur", panel_title)
    if any_orig:
        _plot(axes[0][1], "orig", ".ORIG (raw GRP)")

    mode_str = f"  [mode={mode_label}]" if mode_label else ""
    fig.suptitle(f"{scenario}{mode_str}  (town={town}, {len(egos)} egos)   "
                 f"○=start  □=waypoint  ★=end{aspect_note}", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(out_path, dpi=85, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios-root", default="/data2/marco/CoLMDriver/scenarioset")
    parser.add_argument("--scenario-glob", default="interaction/*/*",
                        help='glob under scenarios-root (e.g. "interaction/Intersection_Deadlock_Resolution/*")')
    parser.add_argument("--carla-host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--carla-timeout-s", type=float, default=30.0)
    parser.add_argument("--hop-resolution-m", type=float, default=2.0,
                        help="GRP hop resolution (matches default in lm_route_manipulation)")
    parser.add_argument("--output", default="/tmp/grp_inspection.json")
    parser.add_argument("--vis-dir", default="/tmp/grp_vis",
                        help="Directory for per-route PNG visualizations (with --visualize)")
    parser.add_argument("--visualize", action="store_true",
                        help="Also save per-route PNG comparisons (slow on large batches)")
    parser.add_argument("--lane-underlay", action="store_true", default=True,
                        help="Draw CARLA lane network as background scatter (default: on)")
    parser.add_argument("--no-lane-underlay", dest="lane_underlay", action="store_false",
                        help="Disable lane-network backdrop")
    parser.add_argument("--lane-sample-m", type=float, default=2.0,
                        help="Spacing for lane-network sampling (m) — smaller=denser=slower")
    parser.add_argument("--include-orig", action="store_true",
                        help="(raw mode only) also analyze .orig backup files where present")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of egos (0=all)")
    parser.add_argument("--bucket", default="all",
                        help="Filter to one bucket: interaction|precrash|v2xpnp|all")
    parser.add_argument("--mode", default="all",
                        choices=["raw", "builder_legacy", "runtime", "all"],
                        help="Which GRP pipeline(s) to run. 'all' runs all 3.")
    parser.add_argument("--ucla-smoothing", action="store_true", default=False,
                        help="Allow UCLA v2 postprocess smoothing in modes 2 & 3 "
                             "(default OFF — the smoothing is harmful when it fires).")
    parser.add_argument("--dp-densify-m", type=float, default=3.0,
                        help="Pre-densify XML inputs to ≤ this gap (m) before "
                             "refine_waypoints_dp. Set to 0 to disable. Default 3.0 "
                             "fixes roundabout / sparse-curve scenarios.")
    parser.add_argument("--dp-max-skip", type=int, default=20,
                        help="DP max_skip parameter (refine_waypoints_dp). "
                             "20 matches scenario_builder_legacy. With --dp-densify-m=3.0 "
                             "you can drop to 5 for ~4× speedup with no quality loss "
                             "(skipping 5 dense pts = same effective gap as legacy's 20).")
    parser.add_argument("--bypass-dp", action="store_true", default=False,
                        help="Skip refine_waypoints_dp entirely and use the grp-densified "
                             "input waypoints directly as the dense_route (with hairpin "
                             "filter, Chaikin, moving avg). Recommended when DP is "
                             "corrupting the lane-network paths returned by grp.trace_route "
                             "(e.g. on left-turn corners where DP picks inner-lane snaps "
                             "and produces a diagonal bee-line).")
    parser.add_argument("--dp-disable-compression", action="store_true", default=True,
                        help="Skip refine_waypoints_dp's compression step that prunes "
                             "intermediate LANEFOLLOW states. Default ON. Required for "
                             "roundabouts; otherwise compression collapses the curved arc "
                             "into 2-3 sparse states with broken GRP fills between them.")
    parser.add_argument("--dp-line-from-plan", dest="dp_line_from_states",
                        action="store_false", default=True,
                        help="Build the dense_route line from refine_waypoints_dp's plan "
                             "(grp.trace_route stitched between DP states). Default OFF — "
                             "we draw a polyline through the DP states themselves, which "
                             "is the only thing actually trustworthy on roundabouts. Use "
                             "this flag only to reproduce scenario_builder_legacy's "
                             "exact (and on roundabouts, broken) output.")
    parser.add_argument("--dp-smooth-iters", type=int, default=4,
                        help="Chaikin corner-cutting iterations on the dense_route line "
                             "(default 4). Each iteration ~doubles point count and rounds "
                             "sharp corners. 0 disables smoothing.")
    parser.add_argument("--dp-w-deviation", type=float, default=0.5,
                        help="DP cost weight on input-deviation (default 0.5; legacy 1.0). "
                             "Lower → DP picks more lane-coherent candidates instead of "
                             "slavishly following jittery XML inputs (helps roundabouts).")
    parser.add_argument("--dp-lane-change-penalty", type=float, default=100.0,
                        help="DP cost penalty when adjacent states are on different lanes "
                             "(default 100.0; legacy 30.0). Higher → DP avoids unintended "
                             "lane changes / wiggles.")
    parser.add_argument("--dp-radius", type=float, default=3.0,
                        help="DP candidate-search radius in meters (default 3.0; legacy 1.5). "
                             "Wider → more lane options at junctions, better roundabout "
                             "coverage; narrower → tighter input matching.")
    args = parser.parse_args()

    enabled_modes = ["raw", "builder_legacy", "runtime"] if args.mode == "all" else [args.mode]

    carla_mod, GlobalRoutePlanner = _import_carla()

    print(f"[grp_inspector] Connecting to CARLA at {args.carla_host}:{args.carla_port}...")
    client = carla_mod.Client(args.carla_host, args.carla_port)
    client.set_timeout(args.carla_timeout_s)
    try:
        srv_ver = client.get_server_version()
        cli_ver = client.get_client_version()
        print(f"[grp_inspector] Connected.  client={cli_ver}  server={srv_ver}")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[grp_inspector] Failed to connect: {exc}", file=sys.stderr)
        return 2

    # Discover ego XMLs
    pattern = os.path.join(args.scenarios_root, args.scenario_glob)
    scenarios = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
    print(f"[grp_inspector] Discovered {len(scenarios)} scenario directories from glob: {pattern}")

    by_town: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for sdir in scenarios:
        for xml in glob.glob(f"{sdir}/*.xml"):
            if "/actors/" in xml:
                continue
            if not is_ego_xml(xml):
                continue
            town = get_town(xml)
            if not town:
                continue
            by_town[town].append((sdir, xml))

    total_egos = sum(len(v) for v in by_town.values())
    if args.limit and total_egos > args.limit:
        # Subsample: keep first N egos in town order
        kept = 0
        new_by_town: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for t, items in by_town.items():
            for it in items:
                if kept >= args.limit:
                    break
                new_by_town[t].append(it)
                kept += 1
            if kept >= args.limit:
                break
        by_town = new_by_town
        total_egos = kept

    print(f"[grp_inspector] Total ego XMLs to inspect: {total_egos} across {len(by_town)} town(s)")
    print(f"[grp_inspector] Towns: {sorted(by_town.keys())}")

    if args.visualize:
        os.makedirs(args.vis_dir, exist_ok=True)

    results: list[dict] = []
    for town, items in sorted(by_town.items()):
        print(f"\n[grp_inspector] === Town: {town} ({len(items)} egos) ===")
        try:
            world = client.load_world(town, reset_settings=False)
            time.sleep(1.5)  # let world settle
            map_obj = world.get_map()
            try:
                grp = GlobalRoutePlanner(map_obj, args.hop_resolution_m)
            except TypeError:
                # Older CARLA API needs DAO
                from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO  # type: ignore
                grp = GlobalRoutePlanner(GlobalRoutePlannerDAO(map_obj, args.hop_resolution_m))
            lane_xs: list[float] = []
            lane_ys: list[float] = []
            if args.visualize and args.lane_underlay:
                t0 = time.time()
                lane_xs, lane_ys = sample_lane_network(map_obj, args.lane_sample_m)
                print(f"[grp_inspector] sampled {len(lane_xs)} lane pts "
                      f"@ {args.lane_sample_m}m in {time.time()-t0:.1f}s")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[grp_inspector] FAILED to load town {town}: {exc}")
            for sdir, xml in items:
                for mode in enabled_modes:
                    results.append({
                        "scenario": os.path.relpath(sdir, args.scenarios_root),
                        "xml": os.path.basename(xml),
                        "town": town,
                        "mode": mode,
                        "error": f"town_load_failed: {exc}",
                    })
            continue

        # ------------------------------------------------------------------
        # Per-mode buffer of egos per scenario (drives one PNG per mode/scenario)
        # ------------------------------------------------------------------
        scenario_buf: dict[tuple[str, str], list[dict]] = defaultdict(list)
        n_egos_in_town = len(items)
        for ego_idx, (sdir, xml) in enumerate(items, start=1):
            scenario_rel = os.path.relpath(sdir, args.scenarios_root)
            xml_basename = os.path.basename(xml)
            cur_wps = parse_xml_waypoints(xml)
            ego_t0 = time.time()
            print(f"[grp_inspector]   [{ego_idx}/{n_egos_in_town}] {scenario_rel}/{xml_basename}  "
                  f"({len(cur_wps)} wps)...", flush=True)

            # ----- mode: raw ---------------------------------------------------
            if "raw" in enabled_modes:
                raw_route: list[dict] = []
                if cur_wps:
                    raw_route = trace_raw(carla_mod, grp, cur_wps)
                raw_metrics = analyze_route(raw_route, input_waypoints=cur_wps) if raw_route else None
                entry_raw: dict = {
                    "scenario": scenario_rel,
                    "xml": xml_basename,
                    "town": town,
                    "mode": "raw",
                    "current_n_waypoints": len(cur_wps),
                    "current_metrics": raw_metrics,
                }
                # Optional .orig comparison (raw only)
                orig_wps: list[dict] = []
                orig_route: list[dict] = []
                if args.include_orig:
                    orig_xml = xml + ".orig"
                    if os.path.isfile(orig_xml):
                        orig_wps = parse_xml_waypoints(orig_xml)
                        entry_raw["orig_n_waypoints"] = len(orig_wps)
                        if orig_wps:
                            orig_route = trace_raw(carla_mod, grp, orig_wps)
                            entry_raw["orig_metrics"] = analyze_route(orig_route, input_waypoints=orig_wps)
                results.append(entry_raw)
                if args.visualize:
                    scenario_buf[(scenario_rel, "raw")].append({
                        "name": xml_basename.replace(".xml", ""),
                        "cur_wps": cur_wps,
                        "cur_route": raw_route,
                        "orig_wps": orig_wps,
                        "orig_route": orig_route,
                        "cur_metrics": raw_metrics,
                        "orig_metrics": entry_raw.get("orig_metrics"),
                        "entry": entry_raw,
                    })

            # ----- mode: builder_legacy ----------------------------------------
            if "builder_legacy" in enabled_modes:
                bl_metrics = None
                bl_dense: list[dict] = []
                bl_aligned: list[dict] = []
                bl_error = None
                bl_trace_source = None
                bl_followed_err = None
                if cur_wps:
                    bl_out = trace_builder_legacy(
                        carla_mod, world, map_obj, grp, cur_wps,
                        hop_resolution_m=args.hop_resolution_m,
                        ucla_smoothing=args.ucla_smoothing,
                        GlobalRoutePlanner_cls=GlobalRoutePlanner,
                        dp_densify_m=args.dp_densify_m,
                        dp_max_skip=args.dp_max_skip,
                        dp_disable_compression=args.dp_disable_compression,
                        dp_line_from_states=args.dp_line_from_states,
                        dp_smooth_iters=args.dp_smooth_iters,
                        dp_w_deviation=args.dp_w_deviation,
                        dp_lane_change_penalty=args.dp_lane_change_penalty,
                        dp_radius=args.dp_radius,
                        bypass_dp=args.bypass_dp,
                    )
                    bl_dense = bl_out.get("dense_route", [])
                    bl_aligned = bl_out.get("aligned_waypoints", [])
                    bl_error = bl_out.get("error")
                    bl_trace_source = bl_out.get("trace_source")
                    bl_followed_err = bl_out.get("followed_route_error")
                    bl_metrics = analyze_route(bl_dense, input_waypoints=cur_wps) if bl_dense else None
                entry_bl: dict = {
                    "scenario": scenario_rel,
                    "xml": xml_basename,
                    "town": town,
                    "mode": "builder_legacy",
                    "current_n_waypoints": len(cur_wps),
                    "current_metrics": bl_metrics,
                    "n_aligned": len(bl_aligned),
                    "trace_source": bl_trace_source,
                    "input_gap_stats": bl_out.get("input_gap_stats") if cur_wps else None,
                    "n_input_resnapped": bl_out.get("n_input_resnapped", 0) if cur_wps else 0,
                    "n_densified_inserted": bl_out.get("n_densified_inserted", 0) if cur_wps else 0,
                    "n_hairpins_removed": bl_out.get("n_hairpins_removed", 0) if cur_wps else 0,
                    "n_states_relaxed": bl_out.get("n_states_relaxed", 0) if cur_wps else 0,
                    "n_lane_snapped": bl_out.get("n_lane_snapped", 0) if cur_wps else 0,
                    "on_road_pct": bl_out.get("on_road_on_road_pct") if cur_wps else None,
                    "max_off_road_m": bl_out.get("on_road_max_off_road_m") if cur_wps else None,
                    "longest_off_road_run_pts": bl_out.get("on_road_longest_off_road_run_pts") if cur_wps else 0,
                }
                if bl_error:
                    entry_bl["error"] = bl_error
                if bl_followed_err:
                    entry_bl["followed_route_error"] = bl_followed_err
                # Also stash dense_route for diagnostic purposes (only when
                # the ego is flagged off-road, to keep JSON size sane).
                if cur_wps and (entry_bl.get("on_road_pct") or 100) < 95:
                    entry_bl["_debug_dense_route"] = bl_dense
                results.append(entry_bl)
                tag = "[bl]"
                gs = entry_bl.get("input_gap_stats") or {}
                m = bl_metrics or {}
                print(f"[grp_inspector]     {tag} +{time.time()-ego_t0:5.1f}s  "
                      f"in_max={gs.get('max_gap','?')}m  out_n={m.get('n_points','?')}  "
                      f"out_max_gap={m.get('max_dist_jump','?')}m  detour={m.get('detour_ratio','?')}x  "
                      f"max_yaw={m.get('max_yaw_jump','?')}",
                      flush=True)
                if args.visualize:
                    # Visualization: dashed line = dense_route, squares = DP-aligned waypoints
                    # (matches the UI's two-layer overlay).
                    scenario_buf[(scenario_rel, "builder_legacy")].append({
                        "name": xml_basename.replace(".xml", ""),
                        "cur_wps": bl_aligned or cur_wps,  # fall back to original if DP failed
                        "cur_route": bl_dense,
                        "orig_wps": [],
                        "orig_route": [],
                        "cur_metrics": bl_metrics,
                        "orig_metrics": None,
                        "entry": entry_bl,
                    })

            # ----- mode: runtime ----------------------------------------------
            if "runtime" in enabled_modes:
                rt_metrics = None
                rt_dense: list[dict] = []
                rt_error = None
                if cur_wps:
                    rt_out = trace_runtime(
                        carla_mod, world, map_obj, cur_wps,
                        hop_resolution_m=args.hop_resolution_m,
                        ucla_smoothing=args.ucla_smoothing,
                        GlobalRoutePlanner_cls=GlobalRoutePlanner,
                    )
                    rt_dense = rt_out.get("dense_route", [])
                    rt_error = rt_out.get("error")
                    rt_metrics = analyze_route(rt_dense, input_waypoints=cur_wps) if rt_dense else None
                entry_rt: dict = {
                    "scenario": scenario_rel,
                    "xml": xml_basename,
                    "town": town,
                    "mode": "runtime",
                    "current_n_waypoints": len(cur_wps),
                    "current_metrics": rt_metrics,
                }
                if rt_error:
                    entry_rt["error"] = rt_error
                results.append(entry_rt)
                if args.visualize:
                    scenario_buf[(scenario_rel, "runtime")].append({
                        "name": xml_basename.replace(".xml", ""),
                        "cur_wps": cur_wps,
                        "cur_route": rt_dense,
                        "orig_wps": [],
                        "orig_route": [],
                        "cur_metrics": rt_metrics,
                        "orig_metrics": None,
                        "entry": entry_rt,
                    })

        # ------------------------------------------------------------------
        # Render PNGs: one per (scenario, mode) into <vis-dir>/<mode>/<scen>.png
        # ------------------------------------------------------------------
        if args.visualize:
            for (scenario_rel, mode), egos in scenario_buf.items():
                mode_dir = os.path.join(args.vis_dir, mode)
                os.makedirs(mode_dir, exist_ok=True)
                vis_name = f"{scenario_rel.replace('/', '_')}.png"
                vis_path = os.path.join(mode_dir, vis_name)
                any_orig = any(e["orig_wps"] for e in egos)
                try:
                    visualize_scenario(
                        vis_path, scenario_rel, town, egos, any_orig,
                        lane_xs=lane_xs or None, lane_ys=lane_ys or None,
                        mode_label=mode,
                    )
                    for e in egos:
                        e["entry"]["visualization"] = vis_path
                except Exception as exc:  # pylint: disable=broad-except
                    for e in egos:
                        e["entry"]["visualization_error"] = str(exc)

    # ------------------------------------------------------------------
    # Per-(mode, scenario) aggregation
    # ------------------------------------------------------------------
    by_mode_scenario: dict[tuple[str, str], dict] = {}
    for r in results:
        key = (r["mode"], r["scenario"])
        agg = by_mode_scenario.setdefault(key, {
            "mode": r["mode"],
            "scenario": r["scenario"],
            "town": r.get("town"),
            "n_egos": 0,
            "n_unparseable": 0,
            "n_broken": 0,
            "n_big_gap": 0,
            "n_detour": 0,
            "n_too_tight": 0,
            "n_wiggle": 0,
            "n_excess_wiggle": 0,
            "n_off_road": 0,
            "n_sustained_off_road": 0,
            "worst_max_yaw_jump": 0.0,
            "worst_max_dist_jump": 0.0,
            "worst_detour_ratio": 0.0,
            "worst_min_turn_radius": float("inf"),
            "worst_max_off_road_m": 0.0,
            "worst_on_road_pct": 100.0,
            "ego_files": [],
        })
        agg["n_egos"] += 1
        cm = r.get("current_metrics")
        flags = classify_breakage_with_road(
            cm,
            r.get("on_road_pct"),
            r.get("max_off_road_m"),
            r.get("longest_off_road_run_pts"),
        )
        if flags.get("unparseable"):
            agg["n_unparseable"] += 1
        else:
            agg["worst_max_yaw_jump"] = max(agg["worst_max_yaw_jump"], cm["max_yaw_jump"])
            agg["worst_max_dist_jump"] = max(agg["worst_max_dist_jump"], cm["max_dist_jump"])
            agg["worst_detour_ratio"] = max(agg["worst_detour_ratio"], cm["detour_ratio"])
            mr = cm.get("min_turn_radius_m")
            if mr is not None:
                agg["worst_min_turn_radius"] = min(agg["worst_min_turn_radius"], mr)
            if flags.get("big_gap"):              agg["n_big_gap"] += 1
            if flags.get("detour"):               agg["n_detour"] += 1
            if flags.get("too_tight_radius"):     agg["n_too_tight"] += 1
            if flags.get("wiggle"):               agg["n_wiggle"] += 1
            if flags.get("excess_wiggle"):        agg["n_excess_wiggle"] += 1
            if flags.get("off_road"):             agg["n_off_road"] += 1
            if flags.get("sustained_off_road"):   agg["n_sustained_off_road"] += 1
            orp = r.get("on_road_pct")
            if orp is not None:
                agg["worst_on_road_pct"] = min(agg["worst_on_road_pct"], orp)
            mor = r.get("max_off_road_m")
            if mor is not None:
                agg["worst_max_off_road_m"] = max(agg["worst_max_off_road_m"], mor)
            if any(flags.values()):               agg["n_broken"] += 1
        agg["ego_files"].append({
            "xml": r["xml"],
            "flags": flags if any(flags.values()) else None,
        })

    # Print per-mode summary table
    print()
    print("=" * 100)
    print(f"[grp_inspector] === PER-MODE BREAKDOWN ===")
    print("=" * 100)
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for (mode, _scen), agg in by_mode_scenario.items():
        by_mode[mode].append(agg)

    for mode in enabled_modes:
        aggs = by_mode.get(mode, [])
        if not aggs:
            continue
        n_scen = len(aggs)
        n_egos_total = sum(a["n_egos"] for a in aggs)
        n_unparseable = sum(a["n_unparseable"] for a in aggs)
        n_broken = sum(a["n_broken"] for a in aggs)
        n_big_gap = sum(a["n_big_gap"] for a in aggs)
        n_detour = sum(a["n_detour"] for a in aggs)
        n_too_tight = sum(a["n_too_tight"] for a in aggs)
        n_wiggle = sum(a["n_wiggle"] for a in aggs)
        n_excess_wiggle = sum(a["n_excess_wiggle"] for a in aggs)
        n_off_road = sum(a["n_off_road"] for a in aggs)
        n_sustained_off = sum(a["n_sustained_off_road"] for a in aggs)
        worst_orp = min((a["worst_on_road_pct"] for a in aggs), default=100.0)
        worst_off_m = max((a["worst_max_off_road_m"] for a in aggs), default=0.0)
        n_scen_any_broken = sum(1 for a in aggs if a["n_broken"] > 0 or a["n_unparseable"] > 0)
        n_scen_all_broken = sum(1 for a in aggs
                                if a["n_egos"] > 0 and (a["n_broken"] + a["n_unparseable"]) == a["n_egos"])

        def pct(n, d):
            return f"{100*n/max(1,d):.1f}%"

        print()
        print(f"  ── mode={mode}".ljust(98) + "──")
        print(f"     scenarios:                 {n_scen}")
        print(f"       any ego broken:            {n_scen_any_broken:>4} ({pct(n_scen_any_broken, n_scen)})")
        print(f"       ALL egos broken:           {n_scen_all_broken:>4} ({pct(n_scen_all_broken, n_scen)})")
        print(f"     ego XMLs:                  {n_egos_total}")
        print(f"       broken (any reason):       {n_broken:>4} ({pct(n_broken, n_egos_total)})")
        print(f"       unparseable / no trace:    {n_unparseable:>4} ({pct(n_unparseable, n_egos_total)})")
        print(f"       big gap     (>10 m jump):  {n_big_gap:>4} ({pct(n_big_gap, n_egos_total)})")
        print(f"       detour      (input >1.3x): {n_detour:>4} ({pct(n_detour, n_egos_total)})")
        print(f"       too tight   (radius <5 m): {n_too_tight:>4} ({pct(n_too_tight, n_egos_total)})  "
              f"<- impossible curve geometry")
        print(f"       wiggle      (lane-change): {n_wiggle:>4} ({pct(n_wiggle, n_egos_total)})  "
              f"<- unintended lane shift, ≥2 sign-flips within 5 m")
        print(f"       excess wiggle (>4/100m):   {n_excess_wiggle:>4} ({pct(n_excess_wiggle, n_egos_total)})")
        print(f"       OFF-ROAD    (<95% on lane):{n_off_road:>4} ({pct(n_off_road, n_egos_total)})  "
              f"<- path leaves drivable surface")
        print(f"       sustained off-road (≥5 m): {n_sustained_off:>4} ({pct(n_sustained_off, n_egos_total)})")
        worst_min_r = min((a["worst_min_turn_radius"] for a in aggs if a["worst_min_turn_radius"] != float("inf")), default=None)
        if worst_min_r is not None:
            print(f"       worst min turn radius:     {worst_min_r:>5.2f} m")
        print(f"       worst on-road pct:         {worst_orp:>5.2f} %")
        print(f"       worst max off-road dist:   {worst_off_m:>5.2f} m")
        if mode == "builder_legacy":
            mode_egos = [r for r in results if r.get("mode") == "builder_legacy"]
            n_dp = sum(1 for r in mode_egos
                       if r.get("trace_source") in ("dp_plan", "dp_states_polyline"))
            n_dp_fail = sum(1 for r in mode_egos
                            if r.get("trace_source") is None
                            and "refine_waypoints_dp_failed" in str(r.get("error", "")))
            n_sparse_inputs = sum(1 for r in mode_egos
                                  if (r.get("input_gap_stats") or {}).get("gaps_gt_10m", 0) > 0)
            n_densified = sum(1 for r in mode_egos if r.get("n_densified_inserted", 0) > 0)
            total_inserted = sum(r.get("n_densified_inserted", 0) for r in mode_egos)
            print(f"       DP succeeded (dp_plan):    {n_dp:>4} ({pct(n_dp, n_egos_total)})  "
                  f"<- matches legacy 'Run GRP' UI")
            print(f"       DP refinement failed:      {n_dp_fail:>4} ({pct(n_dp_fail, n_egos_total)})")
            print(f"       sparse XML inputs (>10m):  {n_sparse_inputs:>4} ({pct(n_sparse_inputs, n_egos_total)})  "
                  f"<- prime suspects for bad DP output")
            print(f"       egos densified pre-DP:     {n_densified:>4} "
                  f"(+{total_inserted} interpolated points total, --dp-densify-m={args.dp_densify_m})")
            n_egos_with_hairpins = sum(1 for r in mode_egos if r.get("n_hairpins_removed", 0) > 0)
            total_hairpins = sum(r.get("n_hairpins_removed", 0) for r in mode_egos)
            print(f"       egos with hairpins fixed:  {n_egos_with_hairpins:>4} ({pct(n_egos_with_hairpins, n_egos_total)})  "
                  f"({total_hairpins} wrong-direction states removed total)")

    # Cross-mode comparison: which scenarios change verdict between modes?
    if len(enabled_modes) > 1:
        print()
        print("  ── CROSS-MODE COMPARISON (per-scenario any-ego-broken) ──")
        scenarios_set = sorted({a["scenario"] for aggs in by_mode.values() for a in aggs})
        change_count = 0
        for sc in scenarios_set:
            verdicts = {}
            for mode in enabled_modes:
                a = next((x for x in by_mode.get(mode, []) if x["scenario"] == sc), None)
                if a is None:
                    continue
                verdicts[mode] = (a["n_broken"] + a["n_unparseable"]) > 0
            if len(set(verdicts.values())) > 1:
                change_count += 1
                pretty = "  ".join(f"{m}={'BROKEN' if v else 'ok':<6}" for m, v in verdicts.items())
                print(f"     {sc[:55]:<55}  {pretty}")
        print(f"     ({change_count}/{len(scenarios_set)} scenarios disagree across modes)")

    # Write outputs
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "per_ego": results,
            "per_mode_scenario": list(by_mode_scenario.values()),
            "enabled_modes": enabled_modes,
            "ucla_smoothing": args.ucla_smoothing,
        }, f, indent=2)

    print(f"\n  Output JSON: {out_path}")
    if args.visualize:
        print(f"  Visualizations: {args.vis_dir}/{{{','.join(enabled_modes)}}}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
