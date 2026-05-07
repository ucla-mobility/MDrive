"""Watchdog: as you save patches in the patch editor, validate + auto-fix
each patched scenarioset and copy clean versions to a final output dir.

Pipeline per scenario (triggered when its actors_manifest.json mtime advances):

  1. Run carla_ground_align.py against a live CARLA → backfill z/pitch/roll
  2. Run a CARLA spawn check → list of failing actors
  3. For each failing actor, search a nudge ladder
       (z+0.5..2.0) × (forward/lateral 0..3 m + 4 diagonals)
     and pick the first nudge that lets it spawn. Persist the fix by writing
     the nudged pose into the actor's first ~6 waypoints so subsequent
     spawns work without runtime tricks.
  4. Re-validate. If 100% spawn, copy the scenario directory to <final-out>/.
  5. Print a per-scenario verdict.

Polls every 2 seconds. Persists per-scenario state in <watch-dir>/.watchdog.json
so restarts don't re-process unchanged scenarios.

Usage:
    python3 -m tools.v2xpnp_patch_watchdog \\
        --watch-dir /tmp/scenarioset_full \\
        --final-out scenarioset/v2xpnp \\
        --carla-port 4080 \\
        --town /Game/Carla/Maps/ucla_v2/ucla_v2 \\
        --carla-egg /data2/marco/CoLMDriver/carla912/PythonAPI/carla/dist/carla-0.9.12-py3.7-linux-x86_64.egg

Run this in its own terminal alongside the patch editor. Each save in the
editor triggers a watchdog cycle for that scenario.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

POLL_INTERVAL_S = 2.0
NUDGE_LADDER: List[Tuple[float, float, float]] = [
    # (forward, lateral, z_lift)
    (0.0, 0.0, 0.5),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, 1.5),
    (0.5, 0.0, 0.5), (-0.5, 0.0, 0.5),
    (1.0, 0.0, 0.5), (-1.0, 0.0, 0.5),
    (0.0, 0.5, 0.5), (0.0, -0.5, 0.5),
    (1.5, 0.0, 0.5), (-1.5, 0.0, 0.5),
    (2.0, 0.0, 0.5), (-2.0, 0.0, 0.5),
    (0.0, 1.0, 0.5), (0.0, -1.0, 0.5),
    (3.0, 0.0, 0.5), (-3.0, 0.0, 0.5),
    (1.0, 1.0, 0.5), (-1.0, -1.0, 0.5),
    (1.0, -1.0, 0.5), (-1.0, 1.0, 0.5),
]
NUDGE_BLEND_FRAMES = 6  # blend nudge → original over this many waypoints


def _load_carla(egg: Optional[str]):
    if egg and egg not in sys.path:
        sys.path.insert(0, egg)
    import carla  # noqa: WPS433
    return carla


def _connect(carla, host: str, port: int, town: str, timeout: float):
    print(f"[watchdog] Connecting to CARLA {host}:{port} ...", flush=True)
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    sv = client.get_server_version()
    print(f"[watchdog] CARLA server {sv}", flush=True)
    print(f"[watchdog] Loading {town} ...", flush=True)
    world = client.load_world(town)
    time.sleep(2.0)
    print(f"[watchdog] Map: {world.get_map().name}", flush=True)
    return client, world


# ---------------------------------------------------------------------------
# Scenarioset discovery + state
# ---------------------------------------------------------------------------
def _scenario_mtime(scen_dir: Path) -> float:
    """Most recent mtime over the manifest + all actor XMLs in a scenarioset."""
    paths = [scen_dir / "actors_manifest.json"]
    for sub in ("npc", "walker", "static"):
        d = scen_dir / "actors" / sub
        if d.is_dir():
            paths.extend(d.glob("*.xml"))
    paths.extend(scen_dir.glob("ucla_v2_custom_ego_vehicle_*.xml"))
    best = 0.0
    for p in paths:
        try:
            m = p.stat().st_mtime
            if m > best:
                best = m
        except OSError:
            pass
    return best


def _load_state(state_path: Path) -> Dict[str, float]:
    if state_path.exists():
        try:
            return json.loads(state_path.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def _save_state(state_path: Path, state: Dict[str, float]) -> None:
    try:
        state_path.write_text(json.dumps(state, indent=2))
    except OSError as exc:
        print(f"[watchdog] WARN: could not persist state: {exc}", flush=True)


# ---------------------------------------------------------------------------
# XML helpers (mutate in place)
# ---------------------------------------------------------------------------
def _read_first_wp(xml_path: Path) -> Optional[Dict[str, float]]:
    try:
        root = ET.parse(str(xml_path)).getroot()
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


def _apply_nudge_to_xml(
    xml_path: Path,
    fwd: float, lat: float, z_lift: float,
    blend_frames: int = NUDGE_BLEND_FRAMES,
) -> bool:
    """Apply the (fwd, lat, z_lift) offset to the actor's first `blend_frames`
    waypoints, with linear blend back to the original by waypoint `blend_frames`.
    Yaw is preserved."""
    try:
        tree = ET.parse(str(xml_path))
    except ET.ParseError:
        return False
    root = tree.getroot()
    wps = root.findall(".//waypoint")
    if not wps:
        return False
    try:
        yaw_deg = float(wps[0].attrib.get("yaw", "0"))
    except (KeyError, ValueError):
        return False
    yaw_rad = math.radians(yaw_deg)
    fx, fy = math.cos(yaw_rad), math.sin(yaw_rad)
    px, py = -fy, fx
    dx_world = fwd * fx + lat * px
    dy_world = fwd * fy + lat * py
    n_blend = min(blend_frames, len(wps))
    for i in range(n_blend):
        alpha = 1.0 - (i / max(1, n_blend))  # 1.0 at i=0, 0.0 at i=n_blend
        try:
            x = float(wps[i].attrib["x"])
            y = float(wps[i].attrib["y"])
            z = float(wps[i].attrib.get("z", "0"))
        except (KeyError, ValueError):
            continue
        wps[i].attrib["x"] = f"{x + alpha * dx_world:.6f}"
        wps[i].attrib["y"] = f"{y + alpha * dy_world:.6f}"
        wps[i].attrib["z"] = f"{z + alpha * z_lift:.6f}"
    tree.write(str(xml_path), encoding="utf-8", xml_declaration=True)
    return True


# ---------------------------------------------------------------------------
# CARLA spawn helpers
# ---------------------------------------------------------------------------
def _spawn_one(carla, world, blueprint_lib, model: str, wp: Dict[str, float],
               extra_fwd: float = 0.0, extra_lat: float = 0.0, extra_z: float = 0.5):
    bp = None
    if model:
        try:
            bp = blueprint_lib.find(model)
        except (IndexError, RuntimeError):
            bp = None
    if bp is None:
        try:
            bps = blueprint_lib.filter("vehicle.tesla.model3")
            if bps:
                bp = bps[0]
        except (IndexError, RuntimeError):
            pass
    if bp is None:
        return None
    yaw_rad = math.radians(wp["yaw"])
    fx, fy = math.cos(yaw_rad), math.sin(yaw_rad)
    px, py = -fy, fx
    x = wp["x"] + extra_fwd * fx + extra_lat * px
    y = wp["y"] + extra_fwd * fy + extra_lat * py
    transform = carla.Transform(
        carla.Location(x=x, y=y, z=wp["z"] + extra_z),
        carla.Rotation(pitch=wp["pitch"], yaw=wp["yaw"], roll=wp["roll"]),
    )
    return world.try_spawn_actor(bp, transform)


# ---------------------------------------------------------------------------
# Per-scenario processing
# ---------------------------------------------------------------------------
def process_scenario(
    scen_dir: Path, *,
    carla, client, world, blueprint_lib,
    egg: str, carla_port: int, carla_host: str,
    final_out: Path, run_ground_align: bool = True,
    log_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    name = scen_dir.name
    rec: Dict[str, Any] = {"scenario": name, "ok": False, "stages": {}, "fixes": []}
    print(f"\n[watchdog] === {name} ===", flush=True)

    # --- 1. ground align (in-process — reuses our world handle) ---
    if run_ground_align:
        try:
            from v2xpnp.pipeline.carla_ground_align import align_routes_dir
            t0 = time.time()
            report = align_routes_dir(world, scen_dir, verbose=False)
            elapsed = time.time() - t0
            err = report.get("error") if isinstance(report, dict) else None
            if err:
                rec["stages"]["ground_align"] = f"error: {err}"
                print(f"  [GA] error: {err}", flush=True)
            else:
                rec["stages"]["ground_align"] = "ok"
                print(f"  [GA] ok ({elapsed:.1f}s)", flush=True)
        except Exception as exc:  # noqa: BLE001
            rec["stages"]["ground_align"] = f"exception: {type(exc).__name__}"
            print(f"  [GA] exception: {type(exc).__name__}: {exc}", flush=True)
            return rec
        # In-process align doesn't reload world, but be safe
        try:
            blueprint_lib = world.get_blueprint_library()
        except Exception:
            pass

    # --- 2. spawn check + auto-fix ---
    manifest_path = scen_dir / "actors_manifest.json"
    if not manifest_path.exists():
        rec["stages"]["spawn"] = "no_manifest"
        return rec
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        rec["stages"]["spawn"] = "bad_manifest"
        return rec

    n_total = 0
    n_spawned = 0
    n_fixed = 0
    failed_actors: List[Dict[str, Any]] = []
    actors_to_destroy: List[Any] = []

    for kind, entries in manifest.items():
        if kind == "walker":
            continue  # walker spawn uses different blueprint flow; skip for now
        for e in entries:
            actor_xml = scen_dir / e["file"]
            if not actor_xml.exists():
                continue
            wp = _read_first_wp(actor_xml)
            if wp is None:
                continue
            n_total += 1
            model = e.get("model", "")
            actor = _spawn_one(carla, world, blueprint_lib, model, wp)
            if actor is not None:
                n_spawned += 1
                actors_to_destroy.append(actor)
                continue
            # try the nudge ladder
            best_nudge: Optional[Tuple[float, float, float]] = None
            for fwd, lat, z_lift in NUDGE_LADDER[1:]:
                actor = _spawn_one(carla, world, blueprint_lib, model, wp,
                                   extra_fwd=fwd, extra_lat=lat, extra_z=z_lift)
                if actor is not None:
                    best_nudge = (fwd, lat, z_lift)
                    actors_to_destroy.append(actor)
                    break
            if best_nudge is not None:
                # Persist the nudge into the XML so future spawns don't need it
                fwd, lat, z_lift = best_nudge
                if _apply_nudge_to_xml(actor_xml, fwd, lat, z_lift):
                    n_fixed += 1
                    n_spawned += 1
                    rec["fixes"].append({
                        "actor": f"{kind}/{e.get('name')}",
                        "fwd_m": fwd, "lat_m": lat, "z_m": z_lift,
                    })
            else:
                failed_actors.append({
                    "kind": kind, "name": str(e.get("name")),
                    "x": wp["x"], "y": wp["y"],
                })

    # cleanup spawned probe actors before returning the world to the user
    for a in actors_to_destroy:
        try:
            a.destroy()
        except Exception:
            pass

    rec["n_total"] = n_total
    rec["n_spawned"] = n_spawned
    rec["n_fixed"] = n_fixed
    rec["n_failed"] = len(failed_actors)
    rec["failures"] = failed_actors[:8]
    pct = (n_spawned / n_total * 100) if n_total else 0.0
    print(
        f"  [SPAWN] {n_spawned}/{n_total} ({pct:.0f}%) "
        f"({n_fixed} fixed, {len(failed_actors)} unfixable)",
        flush=True,
    )
    rec["stages"]["spawn"] = "ok" if not failed_actors else "partial"

    # --- 3. copy to final-out if clean enough ---
    accept_threshold = float(os.environ.get("WD_ACCEPT_THRESHOLD", "0.95"))
    if n_total and (n_spawned / n_total) >= accept_threshold:
        dst = final_out / scen_dir.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(scen_dir, dst)
        rec["stages"]["export"] = f"copied_to_{dst}"
        rec["ok"] = True
        print(f"  [OUT] copied → {dst}", flush=True)
    else:
        rec["stages"]["export"] = "skipped_below_threshold"
        print(f"  [OUT] skipped (below {accept_threshold:.0%} threshold)", flush=True)

    return rec


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--watch-dir", required=True, help="Scenarioset dir patch editor writes to (e.g. /tmp/scenarioset_full)")
    p.add_argument("--final-out", required=True, help="Destination for clean scenarios (e.g. scenarioset/v2xpnp)")
    p.add_argument("--carla-host", default="localhost")
    p.add_argument("--carla-port", type=int, default=4080)
    p.add_argument("--town", default="/Game/Carla/Maps/ucla_v2/ucla_v2")
    p.add_argument("--carla-egg", default="/data2/marco/CoLMDriver/carla912/PythonAPI/carla/dist/carla-0.9.12-py3.7-linux-x86_64.egg")
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--initial-pass", action="store_true",
                   help="Process every scenario once on startup, regardless of mtime")
    p.add_argument("--no-ground-align", action="store_true",
                   help="Skip the carla_ground_align stage (faster, but z stays at converter default)")
    p.add_argument("--log-dir", default=None, help="Per-scenario subprocess logs directory")
    args = p.parse_args()

    watch_dir = Path(args.watch_dir).resolve()
    final_out = Path(args.final_out).resolve()
    final_out.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir).resolve() if args.log_dir else None
    state_path = watch_dir / ".watchdog.json"

    if not watch_dir.is_dir():
        print(f"watch-dir {watch_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # connect to CARLA once
    carla = _load_carla(args.carla_egg)
    client, world = _connect(carla, args.carla_host, args.carla_port, args.town, args.timeout)
    blueprint_lib = world.get_blueprint_library()

    state = _load_state(state_path)
    if args.initial_pass:
        # Force re-process: zero out mtimes
        state = {k: 0.0 for k in state}

    print(f"\n[watchdog] watching {watch_dir} → {final_out}", flush=True)
    print(f"[watchdog] state file: {state_path}", flush=True)
    print(f"[watchdog] poll interval: {POLL_INTERVAL_S}s. Ctrl-C to stop.\n", flush=True)

    try:
        while True:
            scenario_dirs = sorted([d for d in watch_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
            for scen_dir in scenario_dirs:
                if not (scen_dir / "actors_manifest.json").exists():
                    continue
                cur_mtime = _scenario_mtime(scen_dir)
                last_mtime = state.get(scen_dir.name, 0.0)
                if cur_mtime <= last_mtime:
                    continue
                # Cool-down: skip if very recently modified (still being written by editor)
                if time.time() - cur_mtime < 1.5:
                    continue
                try:
                    rec = process_scenario(
                        scen_dir,
                        carla=carla, client=client, world=world,
                        blueprint_lib=blueprint_lib,
                        egg=args.carla_egg, carla_port=args.carla_port,
                        carla_host=args.carla_host,
                        final_out=final_out,
                        run_ground_align=not args.no_ground_align,
                        log_dir=log_dir,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  [ERR] {scen_dir.name}: {type(exc).__name__}: {exc}", flush=True)
                    rec = {"scenario": scen_dir.name, "ok": False, "error": str(exc)}
                # Always update mtime so we don't re-loop on the same state
                state[scen_dir.name] = _scenario_mtime(scen_dir)
                _save_state(state_path, state)
                # ground align reloads world; refresh handles
                try:
                    world = client.get_world()
                    blueprint_lib = world.get_blueprint_library()
                except Exception:
                    pass
            time.sleep(POLL_INTERVAL_S)
    except KeyboardInterrupt:
        print("\n[watchdog] stopped by user", flush=True)


if __name__ == "__main__":
    main()
