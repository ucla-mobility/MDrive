#!/usr/bin/env python3
"""
Select the most mutually diverse scenarios within each category.

For each of 11 categories, selects K=19 scenarios (total 209) that maximize
intra-category diversity across multiple feature dimensions:

  1. Ego vehicle count & maneuver mix
  2. Actor types and counts (walkers, static props, parked vehicles, NPCs)
  3. Topology & constraint structure
  4. Town (map)
  5. Ego trajectory geometry (curvature, heading change, path length, bounding box)

Algorithm: deterministic facility-location-style selection.
  - Start with the scenario farthest from the category centroid.
  - Greedily add the scenario that maximizes the minimum distance to the
    already-selected set (max-min diversification / remote-clique).
  - This produces a globally diverse subset, not just greedy pairwise.

All features are z-score normalised so no single dimension dominates.
"""

import json
import math
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────

SRC_DIR = Path("/data2/marco/CoLMDriver/debug_runs_passed")
DST_DIR = Path("/data2/marco/CoLMDriver/debug_runs_diverse_batch2")
K_PER_CATEGORY = 10

# Folders to exclude (already selected in previous batches)
_EXCLUDE_FILE = Path("/tmp/diverse209_names.txt")
EXCLUDE_NAMES: set = set()
if _EXCLUDE_FILE.exists():
    EXCLUDE_NAMES = {line.strip() for line in _EXCLUDE_FILE.read_text().splitlines() if line.strip()}

# Categories to skip entirely
SKIP_CATEGORIES = {"Major/Minor Unsignalized Entry"}

# ── Feature extraction ──────────────────────────────────────────────────────

# All known categorical values (used for one-hot encoding)
ALL_TOPOLOGIES = [
    "intersection", "highway", "corridor", "two_lane_corridor",
    "t_junction", "roundabout",
]
ALL_MANEUVERS = ["straight", "lane_change", "left", "right"]
ALL_ACTOR_KINDS = ["static_prop", "walker", "parked_vehicle", "npc_vehicle"]
ALL_CONSTRAINT_TYPES = [
    "merges_into_lane_of", "opposite_approach_of",
    "perpendicular_left_of", "right_lane_of", "left_lane_of",
    "follow_route_of", "perpendicular_right_of",
    "same_exit_as", "same_approach_as", "same_road_as", "same_lane_as",
]
ALL_TOWNS = ["Town01", "Town02", "Town03", "Town05", "Town06"]
ALL_ENTRY_ROADS = ["main", "side", "on_ramp", "off_ramp", "roundabout", "unknown"]
ALL_EXIT_ROADS = ["main", "side", "on_ramp", "off_ramp", "roundabout", "unknown"]


def _parse_waypoints(xml_path: Path) -> List[Tuple[float, float]]:
    """Parse (x, y) waypoints from a route XML file."""
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return []
    pts = []
    for wp in root.iter("waypoint"):
        try:
            x = float(wp.attrib.get("x", "nan"))
            y = float(wp.attrib.get("y", "nan"))
            if math.isfinite(x) and math.isfinite(y):
                pts.append((x, y))
        except (ValueError, TypeError):
            continue
    return pts


def _route_geometry_features(all_waypoints: List[List[Tuple[float, float]]]) -> List[float]:
    """
    Compute geometric features across all ego routes in a scenario.
    Returns a fixed-length vector capturing trajectory diversity.
    """
    if not all_waypoints or all(len(w) == 0 for w in all_waypoints):
        return [0.0] * 12

    total_path_length = 0.0
    total_curvature = 0.0
    total_heading_change = 0.0
    num_turns = 0
    all_xs, all_ys = [], []
    num_routes = len(all_waypoints)
    waypoint_counts = []
    max_segment_length = 0.0
    min_segment_length = float("inf")
    route_lengths = []

    for wps in all_waypoints:
        if len(wps) < 2:
            waypoint_counts.append(len(wps))
            route_lengths.append(0.0)
            continue

        waypoint_counts.append(len(wps))
        xs = [p[0] for p in wps]
        ys = [p[1] for p in wps]
        all_xs.extend(xs)
        all_ys.extend(ys)

        # Path length
        path_len = 0.0
        for i in range(1, len(wps)):
            dx = xs[i] - xs[i - 1]
            dy = ys[i] - ys[i - 1]
            seg = math.hypot(dx, dy)
            path_len += seg
            if seg > max_segment_length:
                max_segment_length = seg
            if seg > 0 and seg < min_segment_length:
                min_segment_length = seg
        total_path_length += path_len
        route_lengths.append(path_len)

        # Heading changes & curvature
        headings = []
        for i in range(1, len(wps)):
            dx = xs[i] - xs[i - 1]
            dy = ys[i] - ys[i - 1]
            if abs(dx) > 1e-9 or abs(dy) > 1e-9:
                headings.append(math.atan2(dy, dx))

        if len(headings) >= 2:
            for i in range(1, len(headings)):
                dh = headings[i] - headings[i - 1]
                # Normalise to [-pi, pi]
                dh = (dh + math.pi) % (2 * math.pi) - math.pi
                total_heading_change += abs(dh)
                total_curvature += abs(dh)
                if abs(dh) > math.radians(30):
                    num_turns += 1

    # Bounding box of all routes combined
    if all_xs and all_ys:
        bbox_w = max(all_xs) - min(all_xs)
        bbox_h = max(all_ys) - min(all_ys)
        bbox_aspect = bbox_w / max(bbox_h, 1e-6)
    else:
        bbox_w = bbox_h = bbox_aspect = 0.0

    if min_segment_length == float("inf"):
        min_segment_length = 0.0

    avg_waypoints = np.mean(waypoint_counts) if waypoint_counts else 0.0
    std_route_lengths = float(np.std(route_lengths)) if len(route_lengths) > 1 else 0.0

    return [
        total_path_length,          # 0: total path length across all ego routes
        total_heading_change,       # 1: total absolute heading change
        total_curvature,            # 2: total curvature (same as heading change here)
        float(num_turns),           # 3: number of sharp turns (>30°)
        bbox_w,                     # 4: bounding box width
        bbox_h,                     # 5: bounding box height
        bbox_aspect,                # 6: bounding box aspect ratio
        float(num_routes),          # 7: number of ego routes
        avg_waypoints,              # 8: average waypoints per route
        max_segment_length,         # 9: max segment length
        min_segment_length,         # 10: min segment length
        std_route_lengths,          # 11: std dev of route lengths (spread)
    ]


def _extract_features(run_dir: Path) -> Optional[np.ndarray]:
    """Extract a fixed-length feature vector for a scenario run."""

    # ── Load spec ──
    schema_path = run_dir / "01_schema" / "output.json"
    if not schema_path.exists():
        return None
    with open(schema_path) as f:
        schema_data = json.load(f)
    spec = schema_data.get("spec", {})

    # ── Load summary for town ──
    summary_path = run_dir / "summary.json"
    town = "Town05"  # default
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        town = summary.get("town", "Town05")

    # ── Topology (one-hot) ──
    topo = spec.get("topology", "intersection")
    topo_vec = [1.0 if t == topo else 0.0 for t in ALL_TOPOLOGIES]

    # ── Town (one-hot) ──
    town_vec = [1.0 if t == town else 0.0 for t in ALL_TOWNS]

    # ── Ego vehicles ──
    egos = spec.get("ego_vehicles", [])
    num_egos = len(egos)

    # Maneuver distribution (fraction of egos doing each maneuver)
    maneuver_counts = Counter(e.get("maneuver", "straight") for e in egos)
    denom = max(num_egos, 1)
    maneuver_vec = [maneuver_counts.get(m, 0) / denom for m in ALL_MANEUVERS]

    # Entry/exit road distribution
    entry_counts = Counter(e.get("entry_road", "unknown") for e in egos)
    exit_counts = Counter(e.get("exit_road", "unknown") for e in egos)
    entry_vec = [entry_counts.get(r, 0) / denom for r in ALL_ENTRY_ROADS]
    exit_vec = [exit_counts.get(r, 0) / denom for r in ALL_EXIT_ROADS]

    # ── Actors ──
    actors = spec.get("actors", [])
    # Count by kind
    actor_kind_counts = Counter()
    total_actor_quantity = 0
    for a in actors:
        kind = a.get("kind", "unknown")
        qty = a.get("quantity", 1)
        actor_kind_counts[kind] += qty
        total_actor_quantity += qty

    actor_kind_vec = [float(actor_kind_counts.get(k, 0)) for k in ALL_ACTOR_KINDS]
    # Additional actor features
    num_actor_groups = float(len(actors))

    # Has static props flag
    has_static_props = float(spec.get("allow_static_props", False))

    # ── Constraints ──
    constraints = spec.get("vehicle_constraints", [])
    constraint_counts = Counter(c.get("type", "unknown") for c in constraints)
    constraint_vec = [float(constraint_counts.get(ct, 0)) for ct in ALL_CONSTRAINT_TYPES]
    num_constraints = float(len(constraints))

    # ── Boolean flags ──
    needs_oncoming = float(spec.get("needs_oncoming", False))
    needs_multi_lane = float(spec.get("needs_multi_lane", False))
    needs_on_ramp = float(spec.get("needs_on_ramp", False))
    needs_merge = float(spec.get("needs_merge", False))

    # ── Route geometry ──
    routes_dir = run_dir / "09_routes" / "routes"
    all_waypoints = []
    if routes_dir.exists():
        for xml_file in sorted(routes_dir.glob("*.xml")):
            wps = _parse_waypoints(xml_file)
            if wps:
                all_waypoints.append(wps)
    geo_vec = _route_geometry_features(all_waypoints)

    # ── Assemble feature vector ──
    feature_vec = (
        topo_vec                     # 6
        + town_vec                   # 5
        + [float(num_egos)]          # 1
        + maneuver_vec               # 4
        + entry_vec                  # 6
        + exit_vec                   # 6
        + actor_kind_vec             # 4
        + [num_actor_groups,         # 1
           float(total_actor_quantity),  # 1
           has_static_props]         # 1
        + constraint_vec             # 11
        + [num_constraints]          # 1
        + [needs_oncoming,           # 1
           needs_multi_lane,         # 1
           needs_on_ramp,            # 1
           needs_merge]              # 1
        + geo_vec                    # 12
    )

    return np.array(feature_vec, dtype=np.float64)


# ── Diversity selection ──────────────────────────────────────────────────────

def _z_normalise(matrix: np.ndarray) -> np.ndarray:
    """Z-score normalise each column. Constant columns get zero."""
    mu = matrix.mean(axis=0)
    sigma = matrix.std(axis=0)
    sigma[sigma < 1e-12] = 1.0  # avoid division by zero for constant features
    return (matrix - mu) / sigma


def _select_diverse_maxmin(features: np.ndarray, k: int) -> List[int]:
    """
    Max-min diversity selection (remote-clique / p-dispersion).

    1. Start with the point farthest from centroid.
    2. Greedily pick the point maximising min-distance to selected set.

    This is globally diverse, not just greedy pairwise.
    """
    n = features.shape[0]
    if n <= k:
        return list(range(n))

    # Pairwise distance matrix
    # Using squared Euclidean for efficiency (monotonic with Euclidean)
    diff = features[:, np.newaxis, :] - features[np.newaxis, :, :]
    dist_sq = (diff ** 2).sum(axis=2)
    dist = np.sqrt(dist_sq)

    # Step 1: seed with point farthest from centroid
    centroid = features.mean(axis=0)
    dist_to_centroid = np.sqrt(((features - centroid) ** 2).sum(axis=1))
    first = int(np.argmax(dist_to_centroid))

    selected = [first]
    # min_dist_to_selected[i] = min distance from point i to any selected point
    min_dist_to_selected = dist[first].copy()

    # Step 2: greedy max-min
    for _ in range(k - 1):
        # Mask already selected
        min_dist_to_selected[selected] = -1.0
        next_idx = int(np.argmax(min_dist_to_selected))
        selected.append(next_idx)
        # Update min distances
        np.minimum(min_dist_to_selected, dist[next_idx], out=min_dist_to_selected)

    return selected


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    src = SRC_DIR
    dst = DST_DIR

    if not src.exists():
        print(f"Source directory not found: {src}", file=sys.stderr)
        sys.exit(1)

    # Discover all runs grouped by category
    print(f"Scanning {src} ...")
    runs_by_category: Dict[str, List[Path]] = {}
    for run_dir in sorted(src.iterdir()):
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        cat = summary.get("category", "unknown")
        runs_by_category.setdefault(cat, []).append(run_dir)

    print(f"Found {sum(len(v) for v in runs_by_category.values())} runs in {len(runs_by_category)} categories\n")

    # Process each category
    dst.mkdir(parents=True, exist_ok=True)
    total_selected = 0
    all_selections: Dict[str, List[str]] = {}

    for cat in sorted(runs_by_category.keys()):
        if cat in SKIP_CATEGORIES:
            print(f"Category: {cat} -- SKIPPED (excluded category)")
            continue
        # Filter out already-selected runs
        run_dirs = [rd for rd in runs_by_category[cat] if rd.name not in EXCLUDE_NAMES]
        print(f"Category: {cat} ({len(run_dirs)} eligible runs, {len(runs_by_category[cat])} total)")

        # Extract features
        valid_dirs = []
        feature_list = []
        for rd in run_dirs:
            fv = _extract_features(rd)
            if fv is not None:
                valid_dirs.append(rd)
                feature_list.append(fv)

        if not feature_list:
            print(f"  SKIP: no valid features extracted")
            continue

        feature_matrix = np.array(feature_list)
        print(f"  Feature matrix: {feature_matrix.shape[0]} x {feature_matrix.shape[1]}")

        # Z-normalise within category so diversity is relative
        normed = _z_normalise(feature_matrix)

        # Select diverse subset
        k = min(K_PER_CATEGORY, len(valid_dirs))
        selected_indices = _select_diverse_maxmin(normed, k)
        selected_dirs = [valid_dirs[i] for i in selected_indices]

        print(f"  Selected {len(selected_dirs)} diverse scenarios")
        all_selections[cat] = [d.name for d in selected_dirs]

        # Copy to destination
        for rd in selected_dirs:
            target = dst / rd.name
            if not target.exists():
                shutil.copytree(str(rd), str(target))
            total_selected += 1

    print(f"\n{'='*60}")
    print(f"Total selected: {total_selected}")
    print(f"Destination: {dst}")

    # Write selection manifest
    manifest_path = dst / "_selection_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "total_selected": total_selected,
            "k_per_category": K_PER_CATEGORY,
            "categories": {
                cat: {"count": len(names), "runs": names}
                for cat, names in sorted(all_selections.items())
            },
        }, f, indent=2)
    print(f"Manifest: {manifest_path}")

    # Print diversity summary
    print(f"\n{'='*60}")
    print("SELECTION SUMMARY")
    print(f"{'='*60}")
    for cat in sorted(all_selections.keys()):
        names = all_selections[cat]
        print(f"  {cat}: {len(names)} selected")


if __name__ == "__main__":
    main()
