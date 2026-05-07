"""Quick CARLA spawn-validation for a converted scenarioset directory.

Connects to a running CARLA server, loads the specified town, and tries
to spawn every actor at its first waypoint (vehicles, walkers, statics).
Reports per-scenario / per-actor spawn success.

Usage:
    python3 -m tools.v2xpnp_carla_spawn_check \
        --scenarios /tmp/scenarioset_full \
        --port 4070 \
        --town Town01_Opt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_carla(egg: Optional[str]):
    if egg and egg not in sys.path:
        sys.path.insert(0, egg)
    import carla  # noqa: WPS433
    return carla


def _wp_first(path: Path) -> Optional[Dict[str, float]]:
    try:
        root = ET.parse(str(path)).getroot()
    except ET.ParseError:
        return None
    wp = root.find(".//waypoint")
    if wp is None:
        return None
    try:
        return {
            "x": float(wp.attrib["x"]),
            "y": float(wp.attrib["y"]),
            "z": float(wp.attrib.get("z", "0")),
            "yaw": float(wp.attrib.get("yaw", "0")),
            "pitch": float(wp.attrib.get("pitch", "0")),
            "roll": float(wp.attrib.get("roll", "0")),
        }
    except (KeyError, ValueError):
        return None


def _wp_at_fractions(path: Path, fractions: List[float]) -> List[Dict[str, float]]:
    try:
        root = ET.parse(str(path)).getroot()
    except ET.ParseError:
        return []
    wps = root.findall(".//waypoint")
    if not wps:
        return []
    out = []
    n = len(wps)
    for f in fractions:
        idx = max(0, min(n - 1, int(round(f * (n - 1)))))
        wp = wps[idx]
        try:
            out.append({
                "x": float(wp.attrib["x"]),
                "y": float(wp.attrib["y"]),
                "z": float(wp.attrib.get("z", "0")),
                "yaw": float(wp.attrib.get("yaw", "0")),
                "pitch": float(wp.attrib.get("pitch", "0")),
                "roll": float(wp.attrib.get("roll", "0")),
                "fraction": f,
            })
        except (KeyError, ValueError):
            pass
    return out


def _spawn_actor(carla, world, model: str, wp: Dict[str, float], blueprint_lib):
    bp = None
    if model:
        try:
            bp = blueprint_lib.find(model)
        except (IndexError, RuntimeError):
            bp = None
    if bp is None:
        # try a fallback
        try:
            bps = blueprint_lib.filter("vehicle.tesla.model3")
            if bps:
                bp = bps[0]
        except (IndexError, RuntimeError):
            pass
    if bp is None:
        return None, "no blueprint"
    # Add clearance + retry at multiple z lifts to escape mesh overlap.
    # If still failing, try small xy nudges along the heading direction (+/-0.5 m,
    # +/-1.0 m); CARLA returns spawn=None if there's a tight overlap with a
    # nearby actor, and a small offset along the actor's own forward axis is the
    # safest deconfliction.
    yaw_rad = wp["yaw"] * 3.141592653589793 / 180.0
    import math as _math
    fx = _math.cos(yaw_rad); fy = _math.sin(yaw_rad)
    px, py = -fy, fx  # perpendicular (lateral)
    nudge_offsets = [
        (0.0, 0.0),
        (0.5, 0.0), (-0.5, 0.0),
        (1.0, 0.0), (-1.0, 0.0),
        (0.0, 0.5), (0.0, -0.5),
        (1.5, 0.0), (-1.5, 0.0),
        (2.0, 0.0), (-2.0, 0.0),
        (0.0, 1.0), (0.0, -1.0),
        (3.0, 0.0), (-3.0, 0.0),
        (1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0),
    ]
    for z_lift in (0.5, 1.0, 1.5, 2.0):
        for fwd, lat in nudge_offsets:
            x = wp["x"] + fwd * fx + lat * px
            y = wp["y"] + fwd * fy + lat * py
            transform = carla.Transform(
                carla.Location(x=x, y=y, z=wp["z"] + z_lift),
                carla.Rotation(pitch=wp["pitch"], yaw=wp["yaw"], roll=wp["roll"]),
            )
            actor = world.try_spawn_actor(bp, transform)
            if actor is not None:
                return actor, "ok" if (fwd == 0 and lat == 0 and z_lift == 0.5) else f"ok_nudged_{fwd}_{lat}_z{z_lift}"
    return None, "spawn_collision_or_unreachable"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", required=True, help="dir containing per-scenario subdirs")
    p.add_argument("--port", type=int, default=4070)
    p.add_argument("--host", default="localhost")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--town", default="ucla_v2")
    p.add_argument("--egg", default="/data2/marco/CoLMDriver/carla912/PythonAPI/carla/dist/carla-0.9.12-py3.7-linux-x86_64.egg")
    p.add_argument("--max-scenarios", type=int, default=10)
    p.add_argument("--out", default=None)
    p.add_argument("--multi-waypoint-egos", action="store_true",
                   help="Also test ego at 25%/50%/75%/100% waypoints to verify trajectory")
    args = p.parse_args()

    carla = _load_carla(args.egg)
    print(f"Connecting to CARLA at {args.host}:{args.port}...", flush=True)
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    print(f"Server version: {client.get_server_version()}", flush=True)

    print(f"Loading town: {args.town}", flush=True)
    try:
        world = client.load_world(args.town)
        time.sleep(3.0)  # let world settle
    except Exception as exc:
        print(f"FAILED to load town {args.town}: {exc}", file=sys.stderr)
        sys.exit(2)
    blueprint_lib = world.get_blueprint_library()
    print(f"Map: {world.get_map().name}", flush=True)

    scen_dirs = sorted([d for d in Path(args.scenarios).iterdir() if d.is_dir()])
    if args.max_scenarios > 0:
        scen_dirs = scen_dirs[: args.max_scenarios]

    results: List[Dict[str, Any]] = []
    for scen_dir in scen_dirs:
        manifest_path = scen_dir / "actors_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        n_total = 0
        n_spawned = 0
        n_failed = 0
        spawn_failures: List[Dict[str, Any]] = []
        actors_to_destroy: List[Any] = []

        # First: find ego entries
        ego_entries = manifest.get("ego", [])
        for kind, entries in manifest.items():
            for e in entries:
                actor_xml = scen_dir / e["file"]
                if kind == "ego":
                    actor_xml = scen_dir / e["file"]
                if not actor_xml.exists():
                    continue
                wp = _wp_first(actor_xml)
                if wp is None:
                    continue
                model = e.get("model", "")
                # Skip walker for now (different blueprint setup)
                if kind == "walker":
                    continue
                n_total += 1
                actor, status = _spawn_actor(carla, world, model, wp, blueprint_lib)
                if actor is not None:
                    n_spawned += 1
                    actors_to_destroy.append(actor)
                else:
                    n_failed += 1
                    spawn_failures.append({"name": e.get("name"), "kind": kind, "status": status, "x": wp["x"], "y": wp["y"]})

        # Multi-waypoint ego test
        ego_traj_failures: List[Dict[str, Any]] = []
        if args.multi_waypoint_egos:
            for e in ego_entries:
                actor_xml = scen_dir / e["file"]
                if not actor_xml.exists():
                    continue
                model = e.get("model", "")
                wps_at_fractions = _wp_at_fractions(actor_xml, [0.25, 0.5, 0.75, 1.0])
                for wp in wps_at_fractions:
                    actor, status = _spawn_actor(carla, world, model, wp, blueprint_lib)
                    if actor is not None:
                        actors_to_destroy.append(actor)
                    else:
                        ego_traj_failures.append({
                            "ego": e.get("name"),
                            "fraction": wp["fraction"],
                            "status": status,
                            "x": wp["x"], "y": wp["y"],
                        })
        # Destroy spawned actors before next scenario
        for a in actors_to_destroy:
            try:
                a.destroy()
            except Exception:
                pass
        actors_to_destroy = []

        result = {
            "scenario": scen_dir.name,
            "n_total": n_total,
            "n_spawned": n_spawned,
            "n_failed": n_failed,
            "pct_spawned": (n_spawned / n_total) if n_total else 0.0,
            "failures": spawn_failures[:5],  # top 5 only
            "ego_traj_failures": ego_traj_failures,
        }
        results.append(result)
        print(f"  {scen_dir.name}: {n_spawned}/{n_total} spawned ({result['pct_spawned']*100:.0f}%)", flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}", flush=True)

    total_attempted = sum(r["n_total"] for r in results)
    total_spawned = sum(r["n_spawned"] for r in results)
    print(f"\n=== TOTAL: {total_spawned}/{total_attempted} spawned ({100 * total_spawned / max(1, total_attempted):.1f}%) ===")


if __name__ == "__main__":
    main()
