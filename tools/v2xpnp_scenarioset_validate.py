"""Static validation of a scenarioset/v2xpnp/<scenario>/ directory.

Checks:
  - actors_manifest.json exists and is valid JSON
  - All file references in manifest resolve to existing route XMLs
  - All route XMLs parse, have a <route> element, and >=1 <waypoint>
  - At least one ego entry exists
  - Per-actor first waypoints are spaced apart (no spawn-overlap by simple xy-radius)
  - All waypoint xy values are finite

This is a fast offline check. The actual CARLA spawn validation requires a
running simulator.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _wp_xy_yaw(node: ET.Element) -> Optional[Tuple[float, float, float]]:
    try:
        x = float(node.attrib.get("x", "nan"))
        y = float(node.attrib.get("y", "nan"))
        yaw = float(node.attrib.get("yaw", "nan"))
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(yaw)):
            return None
        return (x, y, yaw)
    except (TypeError, ValueError):
        return None


def _parse_route(path: Path) -> Tuple[bool, str, List[Tuple[float, float, float]]]:
    try:
        tree = ET.parse(str(path))
    except ET.ParseError as exc:
        return False, f"XML parse error: {exc}", []
    root = tree.getroot()
    routes = root.findall(".//route")
    if not routes:
        return False, "no <route> element", []
    waypoints: List[Tuple[float, float, float]] = []
    for r in routes:
        for wp in r.findall("waypoint"):
            xyz = _wp_xy_yaw(wp)
            if xyz is None:
                return False, "non-finite waypoint", []
            waypoints.append(xyz)
    if not waypoints:
        return False, "no <waypoint>", []
    return True, "", waypoints


def validate_scenario(scen_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "scenario": scen_dir.name,
        "ok": True,
        "errors": [],
        "warnings": [],
        "n_actors": 0,
        "n_ego": 0,
        "n_npc": 0,
        "n_walker": 0,
        "n_static": 0,
        "first_wp_collisions": [],   # list of (id_a, id_b, dist)
    }
    manifest_path = scen_dir / "actors_manifest.json"
    if not manifest_path.exists():
        out["ok"] = False
        out["errors"].append(f"missing {manifest_path}")
        return out
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        out["ok"] = False
        out["errors"].append(f"manifest JSON error: {exc}")
        return out
    if "ego" not in manifest or not manifest["ego"]:
        out["ok"] = False
        out["errors"].append("manifest has no ego entries")
        return out

    config_path = scen_dir / "carla_control_config.json"
    if not config_path.exists():
        out["warnings"].append("missing carla_control_config.json")

    actor_first_wp: List[Tuple[str, str, float, float]] = []  # (kind, id, x, y)
    for kind, entries in manifest.items():
        if kind == "ego":
            out["n_ego"] = len(entries)
        elif kind == "npc":
            out["n_npc"] = len(entries)
        elif kind == "walker":
            out["n_walker"] = len(entries)
        elif kind == "static":
            out["n_static"] = len(entries)
        for e in entries:
            out["n_actors"] += 1
            file_rel = e.get("file")
            if not file_rel:
                out["errors"].append(f"{kind} entry without 'file': {e}")
                out["ok"] = False
                continue
            actor_xml = scen_dir / file_rel
            if not actor_xml.exists():
                out["errors"].append(f"{kind}/{e.get('name')}: missing {file_rel}")
                out["ok"] = False
                continue
            ok, err, wps = _parse_route(actor_xml)
            if not ok:
                out["errors"].append(f"{kind}/{e.get('name')}: {err}")
                out["ok"] = False
                continue
            x0, y0, _ = wps[0]
            actor_first_wp.append((kind, str(e.get("name", "?")), x0, y0))

    # First-waypoint pairwise collision check (simple 2 m radius)
    radius_m = 2.0
    n = len(actor_first_wp)
    for i in range(n):
        ka, na, ax, ay = actor_first_wp[i]
        for j in range(i + 1, n):
            kb, nb, bx, by = actor_first_wp[j]
            d = math.hypot(ax - bx, ay - by)
            if d < radius_m:
                # Both static collisions are fine (they don't interact)
                if ka == "static" and kb == "static":
                    continue
                # walker vs static usually fine
                if {ka, kb} == {"walker", "static"}:
                    continue
                if {ka, kb} == {"walker", "walker"}:
                    continue
                out["first_wp_collisions"].append({
                    "a": f"{ka}/{na}",
                    "b": f"{kb}/{nb}",
                    "dist_m": round(d, 3),
                })
    if out["first_wp_collisions"]:
        out["warnings"].append(
            f"{len(out['first_wp_collisions'])} pair(s) of NPC/EGO actors within {radius_m} m at spawn"
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scenario_dirs", nargs="+", help="Path(s) to scenarioset/v2xpnp/<scenario>/")
    p.add_argument("--out", default=None, help="Optional JSON output path")
    args = p.parse_args()

    rows = []
    for sd in args.scenario_dirs:
        scen_dir = Path(sd).resolve()
        report = validate_scenario(scen_dir)
        rows.append(report)
        status = "OK" if report["ok"] and not report["first_wp_collisions"] else (
            "OK (warns)" if report["ok"] else "FAIL"
        )
        print(f"{report['scenario']:<48s} [{status}] "
              f"actors={report['n_actors']} (ego={report['n_ego']} "
              f"npc={report['n_npc']} walker={report['n_walker']} "
              f"static={report['n_static']}) "
              f"spawn_collisions={len(report['first_wp_collisions'])}")
        for err in report["errors"][:5]:
            print(f"    ERR: {err}")
        for warn in report["warnings"][:5]:
            print(f"    WARN: {warn}")
        for c in report["first_wp_collisions"][:5]:
            print(f"    SPAWN_PAIR: {c['a']} ↔ {c['b']} = {c['dist_m']:.2f} m")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
