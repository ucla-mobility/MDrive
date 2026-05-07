"""Resolve mid-trajectory NPC↔NPC OBB collisions in a scenarioset by
underground-teleporting the less-important actor for the duration of the
overlap, then bringing it back up when the conflict clears.

CARLA replay teleports each actor to its waypoint every frame, so a
waypoint at z=z_road − 200 effectively "removes" the actor for that
frame without spawning/destroying. By writing those underground pokes
into the XML directly, no runtime intervention is needed and the
scenarioset remains a static asset.

Per scenario:
  1. Load every NPC + ego XML, build (t_key → pose) tables aligned by time.
  2. For each pair of moving actors, walk the shared frames and run an
     OBB penetration check (>=20 cm penetration counts). Accumulate
     collision intervals per actor pair.
  3. For each collision interval, decide who is the "loser":
       priority: ego >> long-trajectory NPC > short-trajectory NPC > parked
       (ties broken by smaller actor — the larger one is "in the right
       place" more often)
  4. For every frame in the interval, drop the loser's z by `--depth-m`
     (default 50 m). Add a 1-frame ramp at each end so the descent is
     not visible as a teleport.
  5. Write the modified XML in place.

Usage:
    python3 -m tools.v2xpnp_resolve_runtime_collisions \\
        --scenarios /data2/marco/CoLMDriver/scenarioset/v2xpnp \\
        --report-json /tmp/runtime_collisions.json
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# OBB SAT (lifted from v2xpnp_eval_framework.py for self-contained tool)
# ---------------------------------------------------------------------------
def _obb_corners(cx: float, cy: float, yaw_deg: float, length: float, width: float) -> List[Tuple[float, float]]:
    cyaw = math.cos(math.radians(yaw_deg))
    syaw = math.sin(math.radians(yaw_deg))
    hl = length * 0.5
    hw = width * 0.5
    corners_local = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
    return [
        (cx + cl * cyaw - cw * syaw, cy + cl * syaw + cw * cyaw)
        for (cl, cw) in corners_local
    ]


def _obb_overlap(a: List[Tuple[float, float]], b: List[Tuple[float, float]]) -> bool:
    for poly in (a, b):
        n = len(poly)
        for i in range(n):
            x0, y0 = poly[i]
            x1, y1 = poly[(i + 1) % n]
            ex = x1 - x0; ey = y1 - y0
            nx, ny = -ey, ex
            mag = math.hypot(nx, ny)
            if mag < 1e-9:
                continue
            nx /= mag; ny /= mag
            a_proj = [px * nx + py * ny for (px, py) in a]
            b_proj = [px * nx + py * ny for (px, py) in b]
            if max(a_proj) < min(b_proj) - 1e-6 or max(b_proj) < min(a_proj) - 1e-6:
                return False
    return True


def _obb_penetration(a: List[Tuple[float, float]], b: List[Tuple[float, float]]) -> float:
    if not _obb_overlap(a, b):
        return 0.0
    min_pen = float("inf")
    for poly in (a, b):
        n = len(poly)
        for i in range(n):
            x0, y0 = poly[i]; x1, y1 = poly[(i + 1) % n]
            ex = x1 - x0; ey = y1 - y0
            nx, ny = -ey, ex
            mag = math.hypot(nx, ny)
            if mag < 1e-9:
                continue
            nx /= mag; ny /= mag
            a_proj = [px * nx + py * ny for (px, py) in a]
            b_proj = [px * nx + py * ny for (px, py) in b]
            overlap = min(max(a_proj), max(b_proj)) - max(min(a_proj), min(b_proj))
            if overlap < min_pen:
                min_pen = overlap
    return max(0.0, float(min_pen if min_pen != float("inf") else 0.0))


# ---------------------------------------------------------------------------
# Manifest + XML loaders
# ---------------------------------------------------------------------------
def _load_manifest(scen_dir: Path) -> Optional[Dict[str, Any]]:
    p = scen_dir / "actors_manifest.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _load_actor_xml(xml_path: Path) -> Optional[ET.ElementTree]:
    try:
        return ET.parse(str(xml_path))
    except ET.ParseError:
        return None


def _wp_attribs(wp: ET.Element) -> Optional[Dict[str, float]]:
    try:
        return {
            "x": float(wp.attrib["x"]),
            "y": float(wp.attrib["y"]),
            "z": float(wp.attrib.get("z", "0")),
            "yaw": float(wp.attrib.get("yaw", "0")),
            "t": float(wp.attrib.get("time", "0")),
        }
    except (KeyError, ValueError):
        return None


def _t_key(t: float) -> int:
    return int(round(t * 100.0))


# ---------------------------------------------------------------------------
# Per-scenario processing
# ---------------------------------------------------------------------------
def process_scenario(
    scen_dir: Path,
    *,
    pen_thresh_m: float = 0.20,
    depth_m: float = 50.0,
    ramp_frames: int = 1,
    dry_run: bool = False,
) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "scenario": scen_dir.name,
        "intervals_resolved": 0,
        "actors_modified": [],
        "frames_lowered": 0,
    }
    manifest = _load_manifest(scen_dir)
    if manifest is None:
        rec["status"] = "no_manifest"
        return rec

    # Build per-actor: (priority, length, width, xml_tree, frames_by_tkey, name)
    actors: List[Dict[str, Any]] = []
    for kind, entries in manifest.items():
        if kind == "walker":
            continue  # walkers don't enter NPC-NPC collision domain in this tool
        for e in entries:
            xml_path = scen_dir / e["file"]
            tree = _load_actor_xml(xml_path)
            if tree is None:
                continue
            wps = tree.getroot().findall(".//waypoint")
            frames_by_tkey: Dict[int, ET.Element] = {}
            for wp in wps:
                a = _wp_attribs(wp)
                if a is None:
                    continue
                frames_by_tkey[_t_key(a["t"])] = wp
            if not frames_by_tkey:
                continue
            length = float(e.get("length", 4.5))
            width = float(e.get("width", 2.0))
            # Priority (higher = stays in place more often):
            #   ego: huge, never lower
            #   long NPC traj: medium-high
            #   short NPC traj: low
            #   static: lowest (but we mostly skip — static doesn't cause runtime collisions usually)
            n_frames = len(frames_by_tkey)
            if kind == "ego":
                priority = 1_000_000
            elif kind == "static":
                priority = -100  # always loses if it does collide
            else:
                priority = n_frames  # use frame count as proxy for "long-lived → important"
            actors.append({
                "kind": kind,
                "name": str(e.get("name", "?")),
                "xml_path": xml_path,
                "tree": tree,
                "wps_by_tkey": frames_by_tkey,
                "length": length,
                "width": width,
                "priority": priority,
            })

    # For each pair, walk shared frames and detect collision RUNS (consecutive frames)
    pair_runs: List[Dict[str, Any]] = []
    for i in range(len(actors)):
        ai = actors[i]
        for j in range(i + 1, len(actors)):
            aj = actors[j]
            shared = sorted(set(ai["wps_by_tkey"]) & set(aj["wps_by_tkey"]))
            if not shared:
                continue
            cur_run: List[int] = []
            for tk in shared:
                wi = ai["wps_by_tkey"][tk]; wj = aj["wps_by_tkey"][tk]
                ai_attr = _wp_attribs(wi); aj_attr = _wp_attribs(wj)
                if ai_attr is None or aj_attr is None:
                    if cur_run:
                        pair_runs.append({"i": i, "j": j, "tkeys": cur_run})
                        cur_run = []
                    continue
                # cheap broadphase
                if math.hypot(ai_attr["x"] - aj_attr["x"], ai_attr["y"] - aj_attr["y"]) > (
                    max(ai["length"], aj["length"]) + max(ai["width"], aj["width"]) + 0.2
                ):
                    if cur_run:
                        pair_runs.append({"i": i, "j": j, "tkeys": cur_run})
                        cur_run = []
                    continue
                ca = _obb_corners(ai_attr["x"], ai_attr["y"], ai_attr["yaw"],
                                  ai["length"], ai["width"])
                cb = _obb_corners(aj_attr["x"], aj_attr["y"], aj_attr["yaw"],
                                  aj["length"], aj["width"])
                pen = _obb_penetration(ca, cb)
                if pen >= pen_thresh_m:
                    cur_run.append(tk)
                else:
                    if cur_run:
                        pair_runs.append({"i": i, "j": j, "tkeys": cur_run})
                        cur_run = []
            if cur_run:
                pair_runs.append({"i": i, "j": j, "tkeys": cur_run})

    if not pair_runs:
        rec["status"] = "no_collisions"
        return rec

    # For each collision run, pick the loser; collect frames to lower per actor
    frames_to_lower: Dict[int, set] = defaultdict(set)
    for run in pair_runs:
        ai = actors[run["i"]]; aj = actors[run["j"]]
        # Smaller priority = loses
        if ai["priority"] <= aj["priority"]:
            loser_idx = run["i"]
        else:
            loser_idx = run["j"]
        # Tie-break: pick smaller actor (less impactful to remove)
        if ai["priority"] == aj["priority"]:
            if ai["length"] * ai["width"] < aj["length"] * aj["width"]:
                loser_idx = run["i"]
            else:
                loser_idx = run["j"]
        for tk in run["tkeys"]:
            frames_to_lower[loser_idx].add(tk)
        rec["intervals_resolved"] += 1

    # Apply frame lowering with ramp
    actors_modified: set = set()
    for actor_idx, tkeys_set in frames_to_lower.items():
        a = actors[actor_idx]
        modified_count = 0
        for tk in tkeys_set:
            wp = a["wps_by_tkey"].get(tk)
            if wp is None:
                continue
            try:
                z = float(wp.attrib.get("z", "0"))
            except ValueError:
                continue
            wp.attrib["z"] = f"{z - depth_m:.6f}"
            modified_count += 1
        # Ramp: for the frame immediately before/after each interval,
        # drop by depth/2 to soften the visual transition.
        if ramp_frames > 0:
            sorted_tks = sorted(tkeys_set)
            interval_starts = [sorted_tks[0]]
            interval_ends = []
            for k in range(1, len(sorted_tks)):
                if sorted_tks[k] - sorted_tks[k - 1] > 1:
                    interval_ends.append(sorted_tks[k - 1])
                    interval_starts.append(sorted_tks[k])
            interval_ends.append(sorted_tks[-1])
            for tk_edge in interval_starts:
                for delta in range(1, ramp_frames + 1):
                    tk_pre = tk_edge - delta * 10  # 0.1s per step (t-key is *100)
                    wp_pre = a["wps_by_tkey"].get(tk_pre)
                    if wp_pre is not None and tk_pre not in tkeys_set:
                        try:
                            z = float(wp_pre.attrib.get("z", "0"))
                            wp_pre.attrib["z"] = f"{z - (depth_m * (ramp_frames - delta + 1) / (ramp_frames + 1)):.6f}"
                            modified_count += 1
                        except ValueError:
                            pass
            for tk_edge in interval_ends:
                for delta in range(1, ramp_frames + 1):
                    tk_post = tk_edge + delta * 10
                    wp_post = a["wps_by_tkey"].get(tk_post)
                    if wp_post is not None and tk_post not in tkeys_set:
                        try:
                            z = float(wp_post.attrib.get("z", "0"))
                            wp_post.attrib["z"] = f"{z - (depth_m * (ramp_frames - delta + 1) / (ramp_frames + 1)):.6f}"
                            modified_count += 1
                        except ValueError:
                            pass
        if modified_count > 0:
            actors_modified.add(a["name"])
            rec["frames_lowered"] += modified_count
            if not dry_run:
                a["tree"].write(str(a["xml_path"]), encoding="utf-8", xml_declaration=True)

    rec["actors_modified"] = sorted(actors_modified)
    rec["status"] = "resolved" if not dry_run else "would_resolve"
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenarios", required=True, help="Root containing per-scenario subdirs")
    p.add_argument("--scenario", default=None, help="Process only one scenario by name")
    p.add_argument("--pen-thresh-m", type=float, default=0.20)
    p.add_argument("--depth-m", type=float, default=50.0,
                   help="How far below z to teleport the loser (default 50)")
    p.add_argument("--ramp-frames", type=int, default=1,
                   help="Frames of half-depth blend before/after each interval (default 1)")
    p.add_argument("--dry-run", action="store_true",
                   help="Do not modify XMLs; just report what would change")
    p.add_argument("--report-json", default=None,
                   help="Optional JSON output path")
    args = p.parse_args()

    root = Path(args.scenarios).resolve()
    scen_dirs = sorted([d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")])
    if args.scenario:
        scen_dirs = [d for d in scen_dirs if d.name == args.scenario]

    reports: List[Dict[str, Any]] = []
    total_intervals = 0
    total_frames = 0
    total_actors = 0
    for d in scen_dirs:
        rec = process_scenario(
            d,
            pen_thresh_m=args.pen_thresh_m,
            depth_m=args.depth_m,
            ramp_frames=args.ramp_frames,
            dry_run=args.dry_run,
        )
        reports.append(rec)
        total_intervals += rec.get("intervals_resolved", 0)
        total_frames += rec.get("frames_lowered", 0)
        total_actors += len(rec.get("actors_modified", []))
        kind = rec.get("status", "?")
        n_int = rec.get("intervals_resolved", 0)
        n_fr = rec.get("frames_lowered", 0)
        n_ac = len(rec.get("actors_modified", []))
        flag = "DRY" if args.dry_run else ""
        print(f"{d.name:<48s} [{kind:<14s}] intervals={n_int:>3d} frames={n_fr:>4d} actors={n_ac:>2d} {flag}")

    print(
        f"\nTOTAL: {len(scen_dirs)} scenarios "
        f"intervals={total_intervals} frames_lowered={total_frames} actors_modified={total_actors}"
    )
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(reports, indent=2))
        print(f"wrote {args.report_json}")


if __name__ == "__main__":
    main()
