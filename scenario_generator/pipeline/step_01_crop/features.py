import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import generate_legal_paths as glp

from .models import CropFeatures, CropKey


def _opposite_dir(d: str) -> str:
    return {"E": "W", "W": "E", "N": "S", "S": "N"}.get(d, d)


def _cluster_points(points: np.ndarray, eps: float = 12.0) -> List[np.ndarray]:
    if len(points) == 0:
        return []
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    parents = list(range(len(points)))

    def find(a: int) -> int:
        while parents[a] != a:
            parents[a] = parents[parents[a]]
            a = parents[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parents[rb] = ra

    for i in range(len(points)):
        nbrs = tree.query_ball_point(points[i], r=eps)
        for j in nbrs:
            if j != i:
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(len(points)):
        r = find(i)
        groups.setdefault(r, []).append(i)
    return [np.asarray(v, dtype=int) for v in groups.values()]


def _build_road_corridors_from_sigs(sigs: List[Dict[str, Any]]) -> Dict[int, set]:
    """Union-find road corridors using straight-through paths as connectors."""
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: int, b: int) -> None:
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    for s in sigs:
        if str(s.get("entry_to_exit_turn", "")).strip().lower() == "straight":
            try:
                ent = int(s["entry"]["road_id"])
                ex = int(s["exit"]["road_id"])
            except Exception:
                continue
            union(ent, ex)

    for s in sigs:
        try:
            find(int(s["entry"]["road_id"]))
            find(int(s["exit"]["road_id"]))
        except Exception:
            continue

    corridors: Dict[int, set] = {}
    for rid in parent:
        root = find(rid)
        corridors.setdefault(root, set()).add(rid)

    road_to_corridor: Dict[int, set] = {}
    for rid in parent:
        root = find(rid)
        road_to_corridor[rid] = corridors[root]
    return road_to_corridor


def _corridor_key(road_corridors: Dict[int, set], road_id: int) -> Tuple[int, ...]:
    return tuple(sorted(road_corridors.get(road_id, {road_id})))


def _lane_counts_by_road_from_segments(segments: List[Any]) -> Dict[int, set]:
    lane_ids_by_road: Dict[int, set] = {}
    for s in segments:
        try:
            rid = int(s.road_id)
            lid = int(s.lane_id)
        except Exception:
            continue
        lane_ids_by_road.setdefault(rid, set()).add(lid)
    return lane_ids_by_road


def detect_junction_centers(segments: List[Any], adj: List[List[int]]) -> List[np.ndarray]:
    n = len(segments)
    indeg = np.zeros(n, dtype=int)
    for i in range(n):
        for j in adj[i]:
            indeg[j] += 1

    pts = []
    for i, s in enumerate(segments):
        if len(adj[i]) >= 2 or indeg[i] >= 2:
            pts.append(s.points[-1])
        for j in adj[i]:
            if segments[j].road_id != s.road_id:
                pts.append(s.points[-1])
                break

    if not pts:
        return []
    pts = np.asarray(pts, dtype=float)
    clusters = _cluster_points(pts, eps=14.0)
    centers = [pts[idxs].mean(axis=0) for idxs in clusters if len(idxs) >= 3]
    return centers


def _crop_contains_point(c: CropKey, p: np.ndarray) -> bool:
    return (c.xmin <= float(p[0]) <= c.xmax) and (c.ymin <= float(p[1]) <= c.ymax)


def _point_xy(p: Any) -> Tuple[float, float]:
    if isinstance(p, dict):
        return float(p["x"]), float(p["y"])
    return float(p[0]), float(p[1])


def _estimate_lane_count(segments_crop: List[Any]) -> int:
    by: Dict[Tuple[int, int], set] = {}
    for s in segments_crop:
        key = (int(s.road_id), int(s.section_id))
        by.setdefault(key, set()).add(int(s.lane_id))
    if not by:
        return 1
    return max(1, max(len(v) for v in by.values()))


def compute_crop_features(
    town_name: str,
    segments_full: List[Any],
    junction_centers: List[np.ndarray],
    center_xy: Tuple[float, float],
    crop: CropKey,
    min_path_len: float,
    max_paths: int,
    max_depth: int,
    corridor_mode: bool = False,
    roundabout_mode: bool = False,
) -> Optional[CropFeatures]:
    """
    Compute features for a crop region.
    
    Args:
        corridor_mode: If True, use relaxed constraints suitable for corridor topologies:
                       - Minimum 2 segments (one per lane) instead of 6
                       - Minimum 2 paths instead of 6
                       - Enable corridor_mode in path generation to handle pass-through segments
        roundabout_mode: If True, apply roundabout-specific path filtering to remove
                         jittery paths and keep only smoothest paths per cardinal combo.
    """
    cb = glp.CropBox(crop.xmin, crop.xmax, crop.ymin, crop.ymax)
    segs_crop = glp.crop_segments(segments_full, cb)
    
    # Corridor mode: allow minimum 2 segments (one per lane for bidirectional road)
    # Standard mode: require at least 6 segments (typical for intersections)
    min_segments = 2 if corridor_mode else 6
    if len(segs_crop) < min_segments:
        return None

    adj_crop = glp.build_connectivity(segs_crop)
    paths = glp.generate_legal_paths(
        segs_crop, adj_crop, cb,
        min_path_length=min_path_len,
        max_paths=max_paths,
        max_depth=max_depth,
        allow_within_region_fallback=False,
        corridor_mode=corridor_mode,
        roundabout_mode=roundabout_mode,
    )
    
    # Corridor mode: allow minimum 2 paths (one per direction for bidirectional road)
    # Standard mode: require at least 6 paths
    min_paths = 2 if corridor_mode else 6
    if len(paths) < min_paths:
        return None

    sigs = [glp.build_path_signature(p) for p in paths]
    road_corridors = _build_road_corridors_from_sigs(sigs)
    lane_ids_by_road = _lane_counts_by_road_from_segments(segs_crop)
    lane_count_by_corridor: Dict[Tuple[int, ...], int] = {}
    for rid, lanes in lane_ids_by_road.items():
        ck = _corridor_key(road_corridors, rid)
        lane_count_by_corridor.setdefault(ck, 0)
        lane_count_by_corridor[ck] = max(lane_count_by_corridor[ck], len(lanes))
    entry_dirs = sorted(set(s["entry"]["cardinal4"] for s in sigs))
    exit_dirs = sorted(set(s["exit"]["cardinal4"] for s in sigs))
    dirs = sorted(set(entry_dirs) | set(exit_dirs))
    turns = sorted(set(s["entry_to_exit_turn"] for s in sigs))

    straights = [s for s in sigs if s["entry_to_exit_turn"] == "straight"]
    entry_set = set(s["entry"]["cardinal4"] for s in straights)
    has_oncoming = any((_opposite_dir(d) in entry_set) for d in entry_set)

    entry_corridors = set()
    exit_corridors = set()
    entry_roads = set()
    exit_roads = set()
    for s in sigs:
        try:
            ent_rid = int(s["entry"]["road_id"])
            ex_rid = int(s["exit"]["road_id"])
        except Exception:
            continue
        entry_roads.add(ent_rid)
        exit_roads.add(ex_rid)
        entry_corridors.add(_corridor_key(road_corridors, ent_rid))
        exit_corridors.add(_corridor_key(road_corridors, ex_rid))
    distinct_entry_corridors = len(entry_corridors)
    distinct_exit_corridors = len(exit_corridors)
    has_corridor_merge = distinct_entry_corridors > 1 and distinct_exit_corridors == 1

    main_exit_corridor = None
    if exit_corridors:
        corridor_counts: Dict[Tuple[int, ...], int] = {}
        for s in sigs:
            try:
                ex_rid = int(s["exit"]["road_id"])
            except Exception:
                continue
            ck = _corridor_key(road_corridors, ex_rid)
            corridor_counts[ck] = corridor_counts.get(ck, 0) + 1
        if corridor_counts:
            main_exit_corridor = max(corridor_counts.items(), key=lambda kv: kv[1])[0]

    # T-junction detection:
    # Prefer corridor-based count (roads unified by straight-through connectivity).
    # This is more robust than raw unique road count in maps where a single corridor
    # is split into multiple road IDs around a junction.
    all_roads = entry_roads | exit_roads
    corridor_keys = set()
    for rid in all_roads:
        corridor_keys.add(_corridor_key(road_corridors, rid))

    corridor_count = len(corridor_keys)
    # Corridor-based T if exactly 3 corridors meet and we have at least two distinct directions
    is_t_corridor = (corridor_count == 3) and (len(dirs) >= 2)

    # Fallbacks: exactly 3 directions OR exactly 3 unique roads meeting
    # (handles cases where curved roads cause 4 cardinal directions but still form a T)
    is_t_fallback = (len(dirs) == 3) or (len(all_roads) == 3 and len(dirs) <= 4)

    is_four = (len(dirs) >= 4 and len(all_roads) >= 4)
    # Keep topology flags mutually exclusive to avoid ambiguous downstream logic.
    is_t = (is_t_corridor or is_t_fallback) and not is_four

    by_exit: Dict[Tuple[int, int], set] = {}
    for s in sigs:
        key = (int(s["exit"]["road_id"]), int(s["exit"]["section_id"]))
        by_exit.setdefault(key, set()).add(s["entry_to_exit_turn"])
    has_merge = any(("straight" in v and ("left" in v or "right" in v)) for v in by_exit.values())

    # For highways, also detect merges via multiple entry roads into the same exit road.
    # On-ramps often merge from a different road while staying in the same cardinal direction.
    by_exit_entry_roads: Dict[Tuple[int, int], set] = {}
    for s in sigs:
        key = (int(s["exit"]["road_id"]), int(s["exit"]["section_id"]))
        entry_key = (int(s["entry"]["road_id"]), int(s["entry"]["section_id"]))
        by_exit_entry_roads.setdefault(key, set()).add(entry_key)
    has_highway_merge = any(len(v) >= 2 for v in by_exit_entry_roads.values())

    has_merge = has_merge or has_highway_merge or has_corridor_merge

    lane_count = _estimate_lane_count(segs_crop)
    has_ml = lane_count >= 2

    # On-ramp heuristic: require a smaller-lane entry corridor merging into a larger mainline corridor.
    ramp_candidate_paths = 0
    ramp_main_lanes = 0
    ramp_entry_min_lanes = 0
    ramp_entry_max_lanes = 0
    has_valid_ramp_path = False
    if has_corridor_merge and main_exit_corridor:
        main_lanes = lane_count_by_corridor.get(main_exit_corridor, 0)
        ramp_main_lanes = main_lanes
        entry_lanes = [
            lane_count_by_corridor.get(ck, 0)
            for ck in entry_corridors
            if ck != main_exit_corridor
        ]
        for s in sigs:
            try:
                ent_rid = int(s["entry"]["road_id"])
                ex_rid = int(s["exit"]["road_id"])
            except Exception:
                continue
            ent_ck = _corridor_key(road_corridors, ent_rid)
            ex_ck = _corridor_key(road_corridors, ex_rid)
            if ent_ck == main_exit_corridor or ex_ck != main_exit_corridor:
                continue
            if str(s.get("entry_to_exit_turn", "")).strip().lower() == "uturn":
                continue
            ramp_candidate_paths += 1
        has_valid_ramp_path = ramp_candidate_paths > 0
        if entry_lanes:
            ramp_entry_min_lanes = min(entry_lanes)
            ramp_entry_max_lanes = max(entry_lanes)
        has_small_entry = any(lc > 0 and lc <= 2 for lc in entry_lanes)
        has_large_main = main_lanes >= 3
        has_on_ramp = (
            has_small_entry
            and has_large_main
            and has_valid_ramp_path
            and not is_four
            and ramp_entry_min_lanes > 0
            and ramp_entry_min_lanes < main_lanes
        )
    else:
        has_on_ramp = False

    jct_count = sum(1 for jc in junction_centers if _crop_contains_point(crop, jc))

    # Highway detection:
    # A highway is characterized by:
    # 1. 3+ lanes (highways are multi-lane by definition)
    # 2. Predominantly straight paths (limited turns, high straight ratio)
    # 3. Few or no intersections (grade-separated)
    # 4. Often has merge/on-ramp geometry
    straight_ratio = len(straights) / max(1, len(sigs))
    is_highway = (
        lane_count >= 3
        and (straight_ratio >= 0.5 or has_on_ramp)  # Allow on-ramp geometry with more curvature
        and (jct_count <= 1 or (has_on_ramp and jct_count <= 3))  # Ramps add extra junctions
        and not is_four  # Not a four-way intersection
    )

    cx, cy = center_xy
    man_stats: Dict[str, Dict[str, float]] = {}
    for man in ["straight", "left", "right", "uturn"]:
        man_stats[man] = {"count": 0.0, "max_entry_dist": 0.0, "max_exit_dist": 0.0}

    for s in sigs:
        man = s["entry_to_exit_turn"]
        ent = s["entry"]["point"]
        ex = s["exit"]["point"]
        ent_x, ent_y = _point_xy(ent)
        ex_x, ex_y = _point_xy(ex)
        ent_d = float(math.hypot(ent_x - cx, ent_y - cy))
        ex_d = float(math.hypot(ex_x - cx, ex_y - cy))
        st = man_stats.setdefault(man, {"count": 0.0, "max_entry_dist": 0.0, "max_exit_dist": 0.0})
        st["count"] += 1.0
        st["max_entry_dist"] = max(st["max_entry_dist"], ent_d)
        st["max_exit_dist"] = max(st["max_exit_dist"], ex_d)

    area = (crop.xmax - crop.xmin) * (crop.ymax - crop.ymin)
    return CropFeatures(
        town=town_name,
        crop=crop,
        center_xy=center_xy,
        turns=turns,
        entry_dirs=entry_dirs,
        exit_dirs=exit_dirs,
        dirs=dirs,
        has_oncoming_pair=has_oncoming,
        is_t_junction=is_t,
        is_four_way=is_four,
        is_highway=is_highway,
        has_merge_onto_same_road=has_merge,
        has_on_ramp=has_on_ramp,
        lane_count_est=lane_count,
        has_multi_lane=has_ml,
        maneuver_stats=man_stats,
        n_paths=len(paths),
        junction_count=jct_count,
        area=float(area),
        ramp_main_lanes=int(ramp_main_lanes),
        ramp_entry_min_lanes=int(ramp_entry_min_lanes),
        ramp_entry_max_lanes=int(ramp_entry_max_lanes),
        ramp_candidate_paths=int(ramp_candidate_paths),
        _segments_full=segments_full,
        _junction_centers=junction_centers,
    )
