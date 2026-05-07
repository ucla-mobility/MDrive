#!/usr/bin/env python3
"""Print averages over a set of scenarios:
    avg #CAVs, avg route length (m), avg heading change (deg), avg #actors.

Two input modes:

A) CSV mode (default): reads results/artifacts/scenario_database.csv. Its
   ego_routes_json column is GRP-densified (~2m spacing). Filters: --bucket,
   --category, --town, --scenario-id, --scenarios-from-csv.

B) Point-coordinates mode: pass --pc-root DIR (repeatable). Each DIR is walked
   for files named point_coordinates.json. The first non-_partial_ result per
   scenario_id is used. Actor counts come from
       --manifest-root SCENARIOSET_DIR
   which holds <scenario_id>/actors_manifest.json.

NEVER fall back to sparse <waypoint> XML — those are sub-mm-spaced raw GPS
samples for v2xpnp and explode heading-change calculations.

In both modes:
    route length per ego = sum of math.hypot between consecutive (x, y)
    heading change per ego = Σ|Δθ| in degrees where θ = atan2(dy, dx)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable

DEFAULT_CSV = Path("/data2/marco/CoLMDriver/results/artifacts/scenario_database.csv")


def _route_length_m(pts: list[list[float]]) -> float:
    total = 0.0
    for (xa, ya, *_), (xb, yb, *_) in zip(pts, pts[1:]):
        total += math.hypot(xb - xa, yb - ya)
    return total


def _heading_change_deg(pts: list[list[float]]) -> float:
    """Sum |Δheading| over consecutive segments. Assumes input is already
    GRP-densified (~2m spacing) — never feed sparse XML waypoints."""
    total = 0.0
    headings: list[float] = []
    for (xa, ya, *_), (xb, yb, *_) in zip(pts, pts[1:]):
        dx, dy = xb - xa, yb - ya
        if dx * dx + dy * dy > 1e-12:
            headings.append(math.atan2(dy, dx))
    for h0, h1 in zip(headings, headings[1:]):
        dh = (h1 - h0 + math.pi) % (2 * math.pi) - math.pi
        total += abs(dh)
    return math.degrees(total)


def _read_scenario_ids(path: Path, col: str) -> set[str]:
    with open(path) as f:
        reader = csv.DictReader(f)
        if col not in (reader.fieldnames or []):
            sys.exit(f"--col {col!r} not in {path}; columns: {reader.fieldnames}")
        return {row[col] for row in reader if row.get(col)}


def _iter_rows(csv_path: Path, *, buckets: set[str], categories: set[str],
               towns: set[str], ids: set[str]) -> Iterable[dict]:
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if buckets and row["bucket"] not in buckets:
                continue
            if categories and row.get("category", "") not in categories:
                continue
            if towns and row.get("town", "") not in towns:
                continue
            if ids and row["scenario_id"] not in ids and \
               row["scenario_id"].split("/")[-1] not in ids:
                continue
            yield row


_PARTIAL_RE = re.compile(r"_partial_\d{8}_\d{6}$")


def _canonical_scenario_id(name: str) -> str:
    """Strip a `_partial_YYYYMMDD_HHMMSS` suffix from a scenario folder name."""
    return _PARTIAL_RE.sub("", name)


def _discover_point_coords(pc_root: Path) -> dict[str, Path]:
    """Walk `pc_root` for point_coordinates.json files. Layout assumed:
        <pc_root>/<...>/<scenario_id>/image/<run_stamp>/log/point_coordinates.json
    Returns {canonical_scenario_id: Path}, preferring non-_partial_ runs and
    breaking ties by mtime (newest)."""
    by_scen: dict[str, list[Path]] = {}
    for pc in pc_root.rglob("point_coordinates.json"):
        # walk up to find the scenario_id segment (parent of "image")
        for ancestor in pc.parents:
            if ancestor.name == "image":
                scen_dir = ancestor.parent
                scen_id = _canonical_scenario_id(scen_dir.name)
                by_scen.setdefault(scen_id, []).append(pc)
                break
    out: dict[str, Path] = {}
    for scen_id, paths in by_scen.items():
        paths.sort(key=lambda p: (
            "_partial_" in str(p),       # False (canonical) sorts first
            -p.stat().st_mtime,          # newest first
        ))
        out[scen_id] = paths[0]
    return out


def _read_point_coordinates(pc_path: Path) -> dict[int, list[list[float]]]:
    """Returns {ego_index: [[x, y, yaw], ...]} from a point_coordinates.json.
    Drops the leading duplicate point (index 0 typically duplicates index 1)."""
    with open(pc_path) as f:
        data = json.load(f)
    out: dict[int, list[list[float]]] = {}
    for r in data.get("ego_routes", []):
        idx = int(r["ego_index"])
        pts = [[float(p["x"]), float(p["y"]), float(p.get("yaw", 0.0))]
               for p in r.get("points", [])]
        # Drop a leading near-duplicate (zero-length first segment)
        if len(pts) >= 2:
            dx = pts[1][0] - pts[0][0]
            dy = pts[1][1] - pts[0][1]
            if dx * dx + dy * dy < 1e-6:
                pts = pts[1:]
        out[idx] = pts
    return out


def _read_actor_counts(manifest_path: Path) -> int:
    """Returns num_actors = npc + pedestrian/walker + bicycle + static."""
    with open(manifest_path) as f:
        m = json.load(f)
    return (len(m.get("npc", []))
            + len(m.get("pedestrian", [])) + len(m.get("walker", []))
            + len(m.get("bicycle", []))
            + len(m.get("static", [])))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                    help=f"scenario database CSV (default: {DEFAULT_CSV})")
    ap.add_argument("--bucket", action="append", default=[],
                    help="filter by bucket (repeatable): interaction, precrash, v2xpnp, interdrive")
    ap.add_argument("--category", action="append", default=[],
                    help="filter by category (repeatable)")
    ap.add_argument("--town", action="append", default=[],
                    help="filter by town (repeatable)")
    ap.add_argument("--scenario-id", action="append", default=[],
                    help="filter by scenario_id; matches full id or basename (repeatable)")
    ap.add_argument("--scenarios-from-csv", type=Path,
                    help="read scenario ids from this CSV's --col column")
    ap.add_argument("--col", default="scenario",
                    help="column to read scenario ids from (default: scenario)")
    ap.add_argument("--pc-root", action="append", default=[], type=Path,
                    help="point_coordinates.json source root (repeatable). "
                         "Switches to PC mode; --csv is ignored.")
    ap.add_argument("--manifest-root", action="append", default=[], type=Path,
                    help="scenarioset root holding <scenario_id>/actors_manifest.json "
                         "for actor counts in PC mode (repeatable; first match wins).")
    ap.add_argument("--label", default="",
                    help="label printed in the header (cosmetic)")
    args = ap.parse_args()

    n_scen = 0
    egos_per_scen: list[int] = []
    actors_per_scen: list[int] = []
    route_lens: list[float] = []
    heading_changes: list[float] = []
    pc_missing_manifest: list[str] = []

    if args.pc_root:
        if not args.manifest_root:
            sys.exit("--pc-root mode requires --manifest-root for actor counts.")
        scen_pc: dict[str, Path] = {}
        for root in args.pc_root:
            if not root.exists():
                sys.exit(f"--pc-root not found: {root}")
            for scen_id, pc_path in _discover_point_coords(root).items():
                scen_pc.setdefault(scen_id, pc_path)
        for scen_id, pc_path in sorted(scen_pc.items()):
            manifest_path: Path | None = None
            for mroot in args.manifest_root:
                cand = mroot / scen_id / "actors_manifest.json"
                if cand.exists():
                    manifest_path = cand
                    break
            if manifest_path is None:
                pc_missing_manifest.append(scen_id)
                continue
            try:
                routes = _read_point_coordinates(pc_path)
                num_actors = _read_actor_counts(manifest_path)
            except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"warn: skipping {scen_id}: {e}", file=sys.stderr)
                continue
            n_scen += 1
            egos_per_scen.append(len(routes))
            actors_per_scen.append(num_actors)
            for pts in routes.values():
                if len(pts) < 2:
                    continue
                route_lens.append(_route_length_m(pts))
                heading_changes.append(_heading_change_deg(pts))
        source_note = "point_coordinates.json (densified)"
    else:
        if not args.csv.exists():
            sys.exit(f"CSV not found: {args.csv}\n"
                     f"Build it via: python3 tools/build_scenario_database.py")
        ids = set(args.scenario_id)
        if args.scenarios_from_csv:
            ids |= _read_scenario_ids(args.scenarios_from_csv, args.col)
        for row in _iter_rows(args.csv, buckets=set(args.bucket),
                              categories=set(args.category),
                              towns=set(args.town), ids=ids):
            n_scen += 1
            egos_per_scen.append(int(row["num_egos"]))
            actors_per_scen.append(
                int(row["num_npc_vehicles"]) + int(row["num_pedestrians"])
                + int(row["num_bicycles"]) + int(row["num_static"])
            )
            try:
                routes = json.loads(row["ego_routes_json"]) if row["ego_routes_json"] else {}
            except json.JSONDecodeError:
                routes = {}
            for pts in routes.values():
                if len(pts) < 2:
                    continue
                route_lens.append(_route_length_m(pts))
                heading_changes.append(_heading_change_deg(pts))
        source_note = "DB-densified ego_routes_json"

    if n_scen == 0:
        sys.exit("No scenarios matched the filters.")

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    if args.label:
        print(f"=== {args.label} ({source_note}) ===")
    print(f"scenarios:          {n_scen}")
    print(f"avg #CAVs:          {_mean(egos_per_scen):.2f}")
    print(f"avg route len (m):  {_mean(route_lens):.2f}   "
          f"(over {len(route_lens)} ego routes)")
    print(f"avg Δheading (deg): {_mean(heading_changes):.2f}   "
          f"(over {len(heading_changes)} ego routes)")
    print(f"avg #actors:        {_mean(actors_per_scen):.2f}   "
          f"(npc+ped+bicycle+static)")
    if pc_missing_manifest:
        print(f"WARN: {len(pc_missing_manifest)} scenarios skipped (no manifest): "
              f"{pc_missing_manifest[:3]}{'…' if len(pc_missing_manifest) > 3 else ''}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
