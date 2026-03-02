#!/usr/bin/env python3
"""
ego_grp_simplify.py — GRP-based ego route simplification.

Reads *_ego_vehicle_*_REPLAY.xml files from a routes directory, runs each
through the GRP route-alignment DP algorithm (refine_waypoints_dp) to produce
a simplified, compressed waypoint set, then writes the result as the standard
ego XML (same name without _REPLAY, no time/speed attributes).

Usage:
    python tools/ego_grp_simplify.py <routes_dir> \\
        --carla-host localhost --carla-port 2005
"""

from __future__ import annotations

import argparse
import inspect
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.dom import minidom


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRP-based ego route simplification")
    p.add_argument("routes_dir", type=Path, help="Directory containing *_REPLAY.xml ego files")
    p.add_argument("--carla-host", default="localhost")
    p.add_argument("--carla-port", type=int, default=2000)
    p.add_argument("--sampling-resolution", type=float, default=2.0,
                   help="GRP sampling resolution in metres")
    return p.parse_args()


def _ensure_carla_pythonapi_paths() -> None:
    """Ensure CARLA PythonAPI paths are present for `carla` and `agents` imports."""
    workspace = Path(__file__).resolve().parents[1]
    roots: List[Path] = []
    carla_root_env = os.environ.get("CARLA_ROOT")
    if carla_root_env:
        roots.append(Path(carla_root_env))
    roots.extend([workspace / "carla912", workspace / "carla"])

    for root in roots:
        pyapi = root / "PythonAPI"
        carla_pkg = pyapi / "carla"
        if pyapi.is_dir() and str(pyapi) not in sys.path:
            sys.path.insert(0, str(pyapi))
        if carla_pkg.is_dir() and str(carla_pkg) not in sys.path:
            sys.path.insert(0, str(carla_pkg))


def _connect_carla(host: str, port: int):
    """Connect to CARLA and return (carla_map, grp). Raises on failure."""
    _ensure_carla_pythonapi_paths()
    import carla  # type: ignore
    from agents.navigation.global_route_planner import GlobalRoutePlanner  # type: ignore
    try:
        from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO  # type: ignore
    except ImportError:
        GlobalRoutePlannerDAO = None

    client = carla.Client(host, port)
    client.set_timeout(30.0)
    world = client.get_world()
    carla_map = world.get_map()

    # CARLA 0.9.10 uses DAO; 0.9.12+ takes (map, resolution) directly.
    grp_params = list(inspect.signature(GlobalRoutePlanner.__init__).parameters.values())[1:]
    try:
        if grp_params and grp_params[0].name == "dao":
            grp = GlobalRoutePlanner(GlobalRoutePlannerDAO(carla_map, 2.0))
        else:
            grp = GlobalRoutePlanner(carla_map, 2.0)
    except TypeError:
        try:
            grp = GlobalRoutePlanner(carla_map, 2.0)
        except Exception:
            grp = GlobalRoutePlanner(GlobalRoutePlannerDAO(carla_map, 2.0))
    if hasattr(grp, "setup"):
        grp.setup()

    return carla_map, grp


def _load_replay_xml(path: Path) -> Tuple[Dict[str, str], List[Dict]]:
    """
    Parse a REPLAY ego XML.  Returns (route_attrs, waypoints).
    route_attrs has the <route> attributes; waypoints is a list of dicts
    with x, y, z, yaw, pitch, roll (time/speed are intentionally dropped).
    """
    tree = ET.parse(str(path))
    root = tree.getroot()
    route_elem = root.find("route")
    if route_elem is None:
        raise ValueError(f"No <route> element in {path}")

    route_attrs = dict(route_elem.attrib)
    # Strip timing/replay-specific attrs from the simplified version
    for key in ("control_mode", "target_speed"):
        route_attrs.pop(key, None)

    waypoints: List[Dict] = []
    for wp in route_elem.findall("waypoint"):
        waypoints.append({
            "x":     float(wp.get("x", 0)),
            "y":     float(wp.get("y", 0)),
            "z":     float(wp.get("z", 0)),
            "yaw":   float(wp.get("yaw", 0)),
            "pitch": float(wp.get("pitch", 0)) if wp.get("pitch") else 0.0,
            "roll":  float(wp.get("roll", 0))  if wp.get("roll")  else 0.0,
        })
    return route_attrs, waypoints


def _write_simplified_xml(out_path: Path, route_attrs: Dict[str, str],
                           waypoints: List[Dict]) -> None:
    """Write simplified ego XML — positions only, no time/speed."""
    routes = ET.Element("routes")
    route = ET.SubElement(routes, "route")
    for k, v in route_attrs.items():
        route.set(k, str(v))

    for wp in waypoints:
        w = ET.SubElement(route, "waypoint")
        w.set("x",   f"{wp['x']:.6f}")
        w.set("y",   f"{wp['y']:.6f}")
        w.set("z",   f"{wp.get('z', 0.0):.6f}")
        w.set("yaw", f"{wp['yaw']:.6f}")
        pitch = wp.get("pitch", 0.0)
        roll  = wp.get("roll",  0.0)
        if pitch:
            w.set("pitch", f"{float(pitch):.6f}")
        if roll:
            w.set("roll",  f"{float(roll):.6f}")

    rough = ET.tostring(routes, encoding="utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")
    # Remove redundant blank lines added by toprettyxml
    lines = [ln for ln in pretty.splitlines() if ln.strip()]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _simplify(carla_map, grp, waypoints: List[Dict]) -> List[Dict]:
    """
    Run GRP route-alignment DP simplification (refine_waypoints_dp).
    Falls back to returning the original waypoints on any failure.
    """
    try:
        workspace = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(workspace))
        from scenario_generator.pipeline.step_07_route_alignment.main import (  # type: ignore
            refine_waypoints_dp,
        )
        simplified, _ = refine_waypoints_dp(carla_map, waypoints, grp)
        if len(simplified) >= 2:
            return simplified
        print("[GRP-SIMPLIFY]   result too short — using original waypoints", flush=True)
    except Exception as e:
        print(f"[GRP-SIMPLIFY]   refine_waypoints_dp failed: {e} — using original", flush=True)
    return waypoints


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()
    routes_dir = args.routes_dir.resolve()

    if not routes_dir.is_dir():
        print(f"[GRP-SIMPLIFY] ERROR: routes_dir not found: {routes_dir}", flush=True)
        return 1

    replay_files = sorted(routes_dir.glob("*_ego_vehicle_*_REPLAY.xml"))
    if not replay_files:
        print(f"[GRP-SIMPLIFY] No *_ego_vehicle_*_REPLAY.xml files found in {routes_dir}",
              flush=True)
        return 0

    print(f"[GRP-SIMPLIFY] Connecting to CARLA at {args.carla_host}:{args.carla_port} ...",
          flush=True)
    try:
        carla_map, grp = _connect_carla(args.carla_host, args.carla_port)
    except Exception as e:
        print(f"[GRP-SIMPLIFY] ERROR: could not connect to CARLA: {e}", flush=True)
        return 1

    print(f"[GRP-SIMPLIFY] CARLA map: {carla_map.name}", flush=True)

    ok = 0
    for replay_path in replay_files:
        # Derive output path: strip _REPLAY from stem
        stem_simplified = replay_path.stem.replace("_REPLAY", "")
        out_path = routes_dir / (stem_simplified + replay_path.suffix)

        print(f"[GRP-SIMPLIFY]   {replay_path.name} → {out_path.name}", flush=True)
        try:
            route_attrs, waypoints = _load_replay_xml(replay_path)
        except Exception as e:
            print(f"[GRP-SIMPLIFY]   SKIP load error: {e}", flush=True)
            continue

        n_before = len(waypoints)
        simplified = _simplify(carla_map, grp, waypoints)
        n_after = len(simplified)

        try:
            _write_simplified_xml(out_path, route_attrs, simplified)
        except Exception as e:
            print(f"[GRP-SIMPLIFY]   SKIP write error: {e}", flush=True)
            continue

        print(f"[GRP-SIMPLIFY]   {n_before} → {n_after} waypoints  ({out_path.name})",
              flush=True)
        ok += 1

    print(f"[GRP-SIMPLIFY] Done: {ok}/{len(replay_files)} ego files simplified", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
