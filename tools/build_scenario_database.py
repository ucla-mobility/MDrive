"""Build a CSV database of every CoLMDriver scenario across all 4 buckets.

Buckets:
  scenarioset/interaction/<Category>/<N>/
  scenarioset/precrash/<Letter>/
  scenarioset/v2xpnp/<timestamp>/
  v2xpnp/interdrive/<rN_townXX_type>/

For each scenario the script extracts:
  - bucket, scenario_id (path relative to bucket root)
  - town, category (interdrive type tag / interaction category / else None)
  - num_egos, num_npc_vehicles, num_pedestrians, num_bicycles, num_static
  - weather_id (default if not encoded in XML)
  - ego_routes: dense (x, y, yaw) tuple list per ego, JSON-encoded

Ego-route source priority:
  1. existing point_coordinates.json under results/results_driving_custom/baseline/codriving/<bucket>/<scenario_id>
  2. offline reproduction via tools/route_alignment.align_route + start-snap
     (run_custom_eval --align-ego-routes pipeline, no live CARLA)

The offline reproducer is verified 1:1 (<1e-4 m float noise) against existing
JSONs across Town02/Town05/Town06/ucla_v2 in scenarioset.

Run with the colmdrivermarco2 conda env so CARLA + agents/navigation are importable:
    /data/miniconda3/envs/colmdrivermarco2/bin/python tools/build_scenario_database.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

REPO = Path("/data2/marco/CoLMDriver")
SCENARIOSET = REPO / "scenarioset"
INTERDRIVE = REPO / "v2xpnp" / "interdrive"
RESULTS_BASE = REPO / "results" / "results_driving_custom" / "baseline" / "codriving"
DEFAULT_OUT_CSV = REPO / "scenario_database.csv"

CARLA_PY = REPO / "carla912" / "PythonAPI" / "carla"
CARLA_EGG = CARLA_PY / "dist" / "carla-0.9.12-py3.7-linux-x86_64.egg"
sys.path.insert(0, str(CARLA_PY))
sys.path.insert(0, str(CARLA_EGG))
sys.path.insert(0, str(REPO / "tools"))

import carla  # noqa: E402
from agents.navigation.global_route_planner import GlobalRoutePlanner  # noqa: E402
import route_alignment  # noqa: E402


# ────────────────────────────── Town map cache ──────────────────────────────

TOWN_XODR = {
    "Town01": REPO / "carla912/CarlaUE4/Content/Carla/Maps/OpenDrive/Town01.xodr",
    "Town02": REPO / "carla912/CarlaUE4/Content/Carla/Maps/OpenDrive/Town02.xodr",
    "Town03": REPO / "carla912/CarlaUE4/Content/Carla/Maps/OpenDrive/Town03.xodr",
    "Town04": REPO / "carla912/CarlaUE4/Content/Carla/Maps/OpenDrive/Town04.xodr",
    "Town05": REPO / "carla912/CarlaUE4/Content/Carla/Maps/OpenDrive/Town05.xodr",
    "Town06": REPO / "carla912/CarlaUE4/Content/Carla/Maps/OpenDrive/Town06.xodr",
    "Town07": REPO / "carla912/CarlaUE4/Content/Carla/Maps/OpenDrive/Town07.xodr",
    "Town10HD": REPO / "carla912/CarlaUE4/Content/Carla/Maps/OpenDrive/Town10HD.xodr",
    "ucla_v2": REPO / "v2xpnp/map/ucla_v2.xodr",
}
_TOWN_CACHE: dict[str, tuple] = {}


def get_town_map_grp(town: str):
    if town not in _TOWN_CACHE:
        path = TOWN_XODR.get(town)
        if path is None or not path.exists():
            raise FileNotFoundError(f"no xodr for town {town!r}")
        cmap = carla.Map(town, path.read_text())
        grp = GlobalRoutePlanner(cmap, 2.0)
        _TOWN_CACHE[town] = (cmap, grp)
    return _TOWN_CACHE[town]


# ─────────────────────────────── XML parsing ────────────────────────────────

def parse_route_xml(xml_path: Path) -> dict:
    """Extract waypoints + town + role + weather from a single route XML."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    route = root.find("route")
    if route is None:
        raise ValueError(f"no <route> in {xml_path}")
    wps = []
    for wp in route.iter("waypoint"):
        wps.append({
            "x": float(wp.get("x", "0")),
            "y": float(wp.get("y", "0")),
            "z": float(wp.get("z", "0")),
            "yaw": float(wp.get("yaw", "0")),
        })
    weather_elem = route.find("weather")
    weather_id = weather_elem.get("id") if weather_elem is not None else None
    return {
        "town": route.get("town"),
        "role": route.get("role"),
        "model": route.get("model"),
        "waypoints": wps,
        "weather_id": weather_id,
    }


# ───────────────────────── Offline route reproducer ─────────────────────────

def _angle_delta(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _heading_from_pts(pts: list[dict]):
    if len(pts) < 2:
        return None
    dx = pts[1]["x"] - pts[0]["x"]
    dy = pts[1]["y"] - pts[0]["y"]
    if dx == 0.0 and dy == 0.0:
        return None
    return math.degrees(math.atan2(dy, dx))


def _subsample_dense_route(pts: list[dict], min_spacing_m: float = 2.0,
                           keep_yaw_change_deg: float = 8.0) -> list[dict]:
    """Copied from tools/run_custom_eval._subsample_dense_route."""
    if not pts or min_spacing_m <= 0 or len(pts) <= 2:
        return list(pts)
    out = [pts[0]]
    accum = 0.0
    for i in range(1, len(pts) - 1):
        p_prev = pts[i - 1]
        p = pts[i]
        accum += math.hypot(p["x"] - p_prev["x"], p["y"] - p_prev["y"])
        last = out[-1]
        yaw_change = _angle_delta(p.get("yaw", 0.0), last.get("yaw", 0.0))
        if accum >= min_spacing_m or yaw_change >= keep_yaw_change_deg:
            out.append(p)
            accum = 0.0
    out.append(pts[-1])
    return out


def _align_start(carla_map, dense_pts: list[dict], max_snap_dist: float = 5.0) -> None:
    """Replicates RouteScenario._align_start_waypoints. Mutates dense_pts[0].xyz."""
    if not dense_pts:
        return
    xml_yaw = float(dense_pts[0].get("yaw", 0.0))
    heading = _heading_from_pts(dense_pts)
    if heading is not None and (xml_yaw is None or _angle_delta(heading, xml_yaw) > 45.0):
        desired_yaw = heading
    else:
        desired_yaw = xml_yaw
    if desired_yaw is None:
        return
    p0 = dense_pts[0]
    loc = carla.Location(x=p0["x"], y=p0["y"], z=p0.get("z", 0.0))
    wp = carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp is None:
        return
    candidates = [wp]
    for nbr in (wp.get_left_lane(), wp.get_right_lane()):
        if nbr is not None and nbr.lane_type == carla.LaneType.Driving:
            candidates.append(nbr)
    best = min(candidates, key=lambda c: _angle_delta(desired_yaw, c.transform.rotation.yaw))
    if best.transform.location.distance(loc) <= max_snap_dist:
        dense_pts[0]["x"] = float(best.transform.location.x)
        dense_pts[0]["y"] = float(best.transform.location.y)
        dense_pts[0]["z"] = float(best.transform.location.z)


def reproduce_route_offline(xml_path: Path) -> tuple[list[dict] | None, str | None]:
    parsed = parse_route_xml(xml_path)
    raw_wps = parsed["waypoints"]
    town = parsed["town"]
    if len(raw_wps) < 2 or town is None:
        return None, town
    cmap, grp = get_town_map_grp(town)
    aligned = route_alignment.align_route(carla, cmap, grp, raw_wps)
    dense = aligned["dense_route"]
    dense = _subsample_dense_route(dense, min_spacing_m=2.0)
    _align_start(cmap, dense)
    return dense, town


# ───────────────── Result-tree route resolver (existing JSON) ────────────────

def find_existing_point_coords_json(bucket: str, scenario_id: str) -> Path | None:
    """Locate a non-partial point_coordinates.json under the codriving result tree
    matching this scenarioset path. Returns None if no canonical run was logged."""
    base = RESULTS_BASE / bucket / scenario_id
    if not base.exists():
        return None
    matches = sorted(base.rglob("point_coordinates.json"))
    matches = [m for m in matches if "_partial_" not in str(m)]
    return matches[0] if matches else None


def load_existing_routes(json_path: Path) -> dict[int, list[dict]]:
    """Return {ego_index: [{x, y, yaw}, ...]} from a codriving point_coordinates.json."""
    with open(json_path) as f:
        data = json.load(f)
    out = {}
    for r in data.get("ego_routes", []):
        idx = int(r["ego_index"])
        out[idx] = [{"x": float(p["x"]), "y": float(p["y"]), "yaw": float(p["yaw"])}
                    for p in r["points"]]
    return out


# ──────────────── Scenario discovery + per-bucket metadata ──────────────────

def _ego_index_from_filename(stem: str) -> int | None:
    """The trailing _N before .xml is the ego index. e.g.
    town05_vehicle_1_0.xml -> 0, ucla_v2_custom_ego_vehicle_1.xml -> 1,
    r1_town05_ins_c_0.xml -> 0."""
    m = re.search(r"_(\d+)$", stem)
    return int(m.group(1)) if m else None


def _interdrive_type_tag(scenario_dir_name: str) -> str | None:
    """e.g. r1_town05_ins_c -> ins_c; r27_town06_hw_merge -> hw_merge."""
    m = re.match(r"^r\d+_town\d+_(.+)$", scenario_dir_name)
    return m.group(1) if m else None


def discover_scenarios() -> list[dict]:
    """One dict per scenario, with at least bucket/scenario_id/scenario_dir/category."""
    out = []
    # interaction: <Category>/<N>
    interaction_root = SCENARIOSET / "interaction"
    if interaction_root.exists():
        for cat_dir in sorted(p for p in interaction_root.iterdir() if p.is_dir()):
            for sc_dir in sorted(p for p in cat_dir.iterdir() if p.is_dir()):
                if "_partial_" in sc_dir.name:
                    continue
                out.append({
                    "bucket": "interaction",
                    "scenario_id": f"{cat_dir.name}/{sc_dir.name}",
                    "scenario_dir": sc_dir,
                    "category": cat_dir.name,
                })
    # precrash: <Letter>
    od_root = SCENARIOSET / "precrash"
    if od_root.exists():
        for sc_dir in sorted(p for p in od_root.iterdir() if p.is_dir()):
            if "_partial_" in sc_dir.name:
                continue
            out.append({
                "bucket": "precrash",
                "scenario_id": sc_dir.name,
                "scenario_dir": sc_dir,
                "category": None,
            })
    # v2xpnp: <timestamp>
    v2_root = SCENARIOSET / "v2xpnp"
    if v2_root.exists():
        for sc_dir in sorted(p for p in v2_root.iterdir() if p.is_dir()):
            out.append({
                "bucket": "v2xpnp",
                "scenario_id": sc_dir.name,
                "scenario_dir": sc_dir,
                "category": None,
            })
    # interdrive: <rN_townXX_type>
    if INTERDRIVE.exists():
        for sc_dir in sorted(p for p in INTERDRIVE.iterdir() if p.is_dir()):
            out.append({
                "bucket": "interdrive",
                "scenario_id": sc_dir.name,
                "scenario_dir": sc_dir,
                "category": _interdrive_type_tag(sc_dir.name),
            })
    return out


def collect_actor_counts(scen: dict) -> dict:
    """Returns {town, weather_id, num_egos, num_npc_vehicles, num_pedestrians,
    num_bicycles, num_static, ego_xmls: [(idx, path)]}"""
    sd = scen["scenario_dir"]
    bucket = scen["bucket"]

    # interdrive: every XML is an ego; town from dir name; no weather
    if bucket == "interdrive":
        ego_xmls = []
        town = None
        weather = None
        for xml in sorted(sd.glob("*.xml")):
            parsed = parse_route_xml(xml)
            if town is None:
                town = parsed["town"]
            if weather is None and parsed["weather_id"]:
                weather = parsed["weather_id"]
            idx = _ego_index_from_filename(xml.stem)
            ego_xmls.append((idx if idx is not None else len(ego_xmls), xml))
        ego_xmls.sort(key=lambda t: t[0])
        return {
            "town": town,
            "weather_id": weather or "default",
            "num_egos": len(ego_xmls),
            "num_npc_vehicles": 0,
            "num_pedestrians": 0,
            "num_bicycles": 0,
            "num_static": 0,
            "ego_xmls": ego_xmls,
        }

    manifest_path = sd / "actors_manifest.json"
    town = None
    weather = None
    ego_xmls: list[tuple[int, Path]] = []
    n_npc = n_ped = n_bic = n_static = 0

    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        for entry in manifest.get("ego", []):
            xml = sd / entry["file"]
            if not xml.exists():
                continue
            try:
                parsed = parse_route_xml(xml)
            except Exception:
                parsed = {"town": entry.get("town"), "weather_id": None}
            if town is None:
                town = parsed["town"]
            if weather is None and parsed.get("weather_id"):
                weather = parsed["weather_id"]
            idx = _ego_index_from_filename(xml.stem)
            ego_xmls.append((idx if idx is not None else len(ego_xmls), xml))
        n_npc = len(manifest.get("npc", []))
        # interaction uses 'pedestrian' key; v2xpnp uses 'walker'
        n_ped = len(manifest.get("pedestrian", [])) + len(manifest.get("walker", []))
        n_bic = len(manifest.get("bicycle", []))
        n_static = len(manifest.get("static", []))
    else:
        # precrash: discover from filenames + role attr
        for xml in sorted(sd.glob("*.xml")):
            if "_REPLAY" in xml.name:
                continue
            try:
                parsed = parse_route_xml(xml)
            except Exception:
                continue
            if town is None:
                town = parsed["town"]
            if weather is None and parsed.get("weather_id"):
                weather = parsed["weather_id"]
            role = (parsed.get("role") or "").lower()
            if role == "ego":
                idx = _ego_index_from_filename(xml.stem)
                ego_xmls.append((idx if idx is not None else len(ego_xmls), xml))
            elif role == "static":
                n_static += 1
            elif role in ("npc", "vehicle"):
                n_npc += 1
            elif role in ("pedestrian", "walker"):
                n_ped += 1
            elif role == "bicycle":
                n_bic += 1
    ego_xmls.sort(key=lambda t: t[0])
    return {
        "town": town,
        "weather_id": weather or "default",
        "num_egos": len(ego_xmls),
        "num_npc_vehicles": n_npc,
        "num_pedestrians": n_ped,
        "num_bicycles": n_bic,
        "num_static": n_static,
        "ego_xmls": ego_xmls,
    }


# ─────────────────────────────── CSV emission ───────────────────────────────

CSV_FIELDS = [
    "bucket", "scenario_id", "category", "town", "weather_id",
    "num_egos", "num_npc_vehicles", "num_pedestrians", "num_bicycles",
    "num_static",
    "ego_routes_json",
]


def build_row(scen: dict, *, prefer_offline: bool = False) -> dict:
    actors = collect_actor_counts(scen)
    bucket = scen["bucket"]
    scenario_id = scen["scenario_id"]
    row = {
        "bucket": bucket,
        "scenario_id": scenario_id,
        "category": scen.get("category") or "",
        "town": actors.get("town") or "",
        "weather_id": actors.get("weather_id") or "default",
        "num_egos": actors["num_egos"],
        "num_npc_vehicles": actors["num_npc_vehicles"],
        "num_pedestrians": actors["num_pedestrians"],
        "num_bicycles": actors["num_bicycles"],
        "num_static": actors["num_static"],
        "ego_routes_json": "{}",
    }

    routes: dict[int, list[list[float]]] = {}

    if not prefer_offline:
        json_path = find_existing_point_coords_json(bucket, scenario_id)
        if json_path is not None:
            existing = load_existing_routes(json_path)
            for idx, pts in existing.items():
                routes[idx] = [[round(p["x"], 4), round(p["y"], 4), round(p["yaw"], 4)] for p in pts]

    if not routes and actors["ego_xmls"]:
        for idx, xml in actors["ego_xmls"]:
            try:
                dense, _town = reproduce_route_offline(xml)
            except Exception as exc:
                print(f"  [WARN] offline reproduction failed for {xml}: {exc}")
                continue
            if dense is None:
                continue
            routes[idx] = [[round(p["x"], 4), round(p["y"], 4), round(p["yaw"], 4)] for p in dense]

    row["ego_routes_json"] = json.dumps({str(k): v for k, v in sorted(routes.items())},
                                        separators=(",", ":"))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_CSV,
                    help=f"output CSV path (default {DEFAULT_OUT_CSV})")
    ap.add_argument("--bucket", action="append",
                    choices=["interaction", "precrash", "v2xpnp", "interdrive"],
                    help="restrict to one or more buckets (repeatable). Default: all.")
    ap.add_argument("--prefer-offline", action="store_true",
                    help="always reproduce routes offline, ignore existing JSONs")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of scenarios processed (for smoke testing)")
    args = ap.parse_args()

    scenarios = discover_scenarios()
    if args.bucket:
        keep = set(args.bucket)
        scenarios = [s for s in scenarios if s["bucket"] in keep]
    if args.limit is not None:
        scenarios = scenarios[:args.limit]

    print(f"Discovered {len(scenarios)} scenarios.")
    rows = []
    for i, scen in enumerate(scenarios, 1):
        try:
            row = build_row(scen, prefer_offline=args.prefer_offline)
        except Exception as exc:
            print(f"  [ERROR] {scen['bucket']}/{scen['scenario_id']}: {exc}")
            continue
        rows.append(row)
        others = (row["num_npc_vehicles"] + row["num_pedestrians"]
                  + row["num_bicycles"] + row["num_static"])
        print(f"  [{i:>3}/{len(scenarios)}] {row['bucket']}/{row['scenario_id']}: "
              f"egos={row['num_egos']}, others={others}, town={row['town']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nWrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
