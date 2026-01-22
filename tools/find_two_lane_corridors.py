#!/usr/bin/env python3
"""
Quick inspector for two-lane corridors in a CARLA town nodes file.

Assumes the town_nodes JSON format used in scenario_generator/town_nodes/*.json
and reports road_ids that have exactly two lanes (typically one per direction).
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def _accumulate_lengths(points: List[Tuple[float, float]]) -> float:
    """Return total polyline length for a lane."""
    if len(points) < 2:
        return 0.0
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def find_two_lane_roads(town_path: Path) -> Dict[int, Dict[str, object]]:
    data = json.loads(town_path.read_text())
    payload = data.get("payload") or {}
    required = ["x", "y", "road_id", "lane_id", "lane_direction"]
    for key in required:
        if key not in payload:
            raise ValueError(f"Missing key '{key}' in payload")

    lane_points: Dict[Tuple[int, int], List[Tuple[float, float, str]]] = defaultdict(list)
    for x, y, rid, lid, direction in zip(
        payload["x"],
        payload["y"],
        payload["road_id"],
        payload["lane_id"],
        payload["lane_direction"],
    ):
        lane_points[(int(rid), int(lid))].append((float(x), float(y), str(direction)))

    road_summary: Dict[int, Dict[str, object]] = defaultdict(
        lambda: {"lanes": set(), "dirs": set(), "lengths": [], "bbox": [math.inf, -math.inf, math.inf, -math.inf]}
    )

    for (rid, lid), pts in lane_points.items():
        xy = [(p[0], p[1]) for p in pts]
        length = _accumulate_lengths(xy)
        dirs = {p[2] for p in pts}
        summary = road_summary[rid]
        summary["lanes"].add(lid)
        summary["dirs"].update(dirs)
        summary["lengths"].append(length)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        summary["bbox"][0] = min(summary["bbox"][0], min(xs))
        summary["bbox"][1] = max(summary["bbox"][1], max(xs))
        summary["bbox"][2] = min(summary["bbox"][2], min(ys))
        summary["bbox"][3] = max(summary["bbox"][3], max(ys))

    two_lane = {}
    for rid, summary in road_summary.items():
        if len(summary["lanes"]) != 2:
            continue
        dirs = summary["dirs"]
        # Typically expect one forward and one opposing lane
        if not dirs or dirs == {"forward"} or dirs == {"opposing"}:
            continue
        two_lane[rid] = summary

    return two_lane


def main() -> None:
    ap = argparse.ArgumentParser(description="List two-lane corridor road_ids in a town_nodes JSON.")
    ap.add_argument(
        "--town",
        default="scenario_generator/town_nodes/Town02.json",
        help="Path to town_nodes JSON (default: Town02).",
    )
    args = ap.parse_args()
    town_path = Path(args.town)
    if not town_path.exists():
        raise SystemExit(f"Town file not found: {town_path}")

    roads = find_two_lane_roads(town_path)
    print(f"Found {len(roads)} two-lane corridor roads in {town_path.name}")
    for rid in sorted(roads.keys()):
        info = roads[rid]
        lanes = sorted(info["lanes"])
        dirs = sorted(info["dirs"])
        lengths = info["lengths"]
        avg_len = sum(lengths) / len(lengths) if lengths else 0.0
        bbox = info["bbox"]
        print(
            f" road_id={rid:>4} lanes={lanes} dirs={dirs} "
            f"avg_len_m={avg_len:.1f} bbox=[{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f}]"
        )


if __name__ == "__main__":
    main()
