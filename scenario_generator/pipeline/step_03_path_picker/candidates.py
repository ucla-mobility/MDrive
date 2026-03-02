import math
from typing import Any, Dict, List, Optional, Tuple


def _candidate_entry_lane_id(cand: Dict[str, Any]) -> Optional[int]:
    sig = (cand or {}).get("signature", {})
    ent = (sig or {}).get("entry", {})
    lid = (ent or {}).get("lane_id", None)
    try:
        return int(lid) if lid is not None else None
    except Exception:
        return None


def _candidate_exit_lane_id(cand: Dict[str, Any]) -> Optional[int]:
    """Get the exit lane ID from the path signature."""
    sig = (cand or {}).get("signature", {})
    ext = (sig or {}).get("exit", {})
    lid = (ext or {}).get("lane_id", None)
    try:
        return int(lid) if lid is not None else None
    except Exception:
        return None


def _candidate_entry_point(cand: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    sig = (cand or {}).get("signature", {})
    ent = (sig or {}).get("entry", {})
    p = (ent or {}).get("point", None)
    if not isinstance(p, dict):
        return None
    try:
        return float(p.get("x")), float(p.get("y"))
    except Exception:
        return None


def _candidate_entry_heading_rad(cand: Dict[str, Any]) -> Optional[float]:
    sig = (cand or {}).get("signature", {})
    ent = (sig or {}).get("entry", {})
    hd = (ent or {}).get("heading_deg", None)
    try:
        return math.radians(float(hd)) if hd is not None else None
    except Exception:
        return None


def _candidate_exit_road_id(cand: Dict[str, Any]) -> Optional[int]:
    sig = (cand or {}).get("signature", {})
    ex = (sig or {}).get("exit", {})
    rid = (ex or {}).get("road_id", None)
    try:
        return int(rid) if rid is not None else None
    except Exception:
        return None


def _candidate_entry_road_id(cand: Dict[str, Any]) -> Optional[int]:
    sig = (cand or {}).get("signature", {})
    ent = (sig or {}).get("entry", {})
    rid = (ent or {}).get("road_id", None)
    try:
        return int(rid) if rid is not None else None
    except Exception:
        return None


def _candidate_all_road_ids(cand: Dict[str, Any]) -> set:
    """Return set of all road_ids that this path travels through (entry + exit)."""
    roads = set()
    ent_rid = _candidate_entry_road_id(cand)
    ex_rid = _candidate_exit_road_id(cand)
    if ent_rid is not None:
        roads.add(ent_rid)
    if ex_rid is not None:
        roads.add(ex_rid)
    return roads


def _build_road_corridors(candidates: List[Dict[str, Any]]) -> Dict[int, set]:
    """
    Build a mapping from each road_id to its 'corridor' - the set of road_ids
    that are connected by straight-through paths. In a T-junction, the main road
    might have road_id=10 on one side and road_id=11 on the other, but they're
    the same logical corridor.
    """
    # Find pairs of roads connected by straight paths
    corridor_pairs = []
    for c in candidates:
        sig = (c or {}).get("signature", {})
        if str((sig or {}).get("entry_to_exit_turn", "")).strip().lower() == "straight":
            ent_rid = _candidate_entry_road_id(c)
            ex_rid = _candidate_exit_road_id(c)
            if ent_rid is not None and ex_rid is not None:
                corridor_pairs.append((ent_rid, ex_rid))

    # Build union-find to merge corridors
    parent = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    # Merge all straight-through connected roads
    for a, b in corridor_pairs:
        union(a, b)

    # Also add all other roads as their own corridor
    for c in candidates:
        ent_rid = _candidate_entry_road_id(c)
        ex_rid = _candidate_exit_road_id(c)
        if ent_rid is not None:
            find(ent_rid)
        if ex_rid is not None:
            find(ex_rid)

    # Build corridor sets
    corridors = {}
    for rid in parent:
        root = find(rid)
        if root not in corridors:
            corridors[root] = set()
        corridors[root].add(rid)

    # Map each road to its full corridor set
    road_to_corridor = {}
    for rid in parent:
        root = find(rid)
        road_to_corridor[rid] = corridors[root]

    return road_to_corridor


def _candidate_entry_cardinal(cand: Dict[str, Any]) -> str:
    sig = (cand or {}).get("signature", {})
    ent = (sig or {}).get("entry", {})
    return str((ent or {}).get("cardinal4", "")).strip().upper()


def _candidate_exit_cardinal(cand: Dict[str, Any]) -> str:
    sig = (cand or {}).get("signature", {})
    ex = (sig or {}).get("exit", {})
    return str((ex or {}).get("cardinal4", "")).strip().upper()


def _candidate_length(cand: Dict[str, Any]) -> float:
    sig = (cand or {}).get("signature", {})
    try:
        return float((sig or {}).get("length_m", 0.0))
    except Exception:
        return 0.0


def _candidate_lane_change_count(cand: Dict[str, Any]) -> int:
    """Count lane-id transitions *within the same road/section corridor*.

    This avoids counting lane_id changes that happen simply because the vehicle turned
    onto a different road/section (which is not a 'lane change' in the behavioral sense).
    """
    sig = (cand or {}).get("signature", {})
    lanes = sig.get("lanes", [])
    if not isinstance(lanes, list) or len(lanes) < 2:
        return 0
    roads = sig.get("roads", [])
    sections = sig.get("sections", [])

    count = 0
    prev_lane = None
    prev_road = None
    prev_section = None

    for i, lid in enumerate(lanes):
        road = roads[i] if isinstance(roads, list) and i < len(roads) else None
        sec = sections[i] if isinstance(sections, list) and i < len(sections) else None

        try:
            lane_int = int(lid)
        except Exception:
            prev_lane = lid
            prev_road = road
            prev_section = sec
            continue

        same_corridor = True
        try:
            if prev_road is not None and road is not None and int(road) != int(prev_road):
                same_corridor = False
            if prev_section is not None and sec is not None and int(sec) != int(prev_section):
                same_corridor = False
        except Exception:
            pass

        if (
            same_corridor
            and prev_lane is not None
            and isinstance(prev_lane, int)
            and lane_int != prev_lane
        ):
            count += 1

        prev_lane = lane_int
        prev_road = road
        prev_section = sec

    return count


def _candidate_straight_lateral_drift(cand: Dict[str, Any]) -> int:
    """For straight maneuvers, count how many intermediate segments deviate from entry/exit lane.

    Even if road IDs change (junction segments), a straight path that enters lane -2
    and exits lane -2 but uses lane -1 in the middle is 'drifting' and should be penalized.
    Returns the number of segments that deviate from the entry lane.
    """
    sig = (cand or {}).get("signature", {})
    man = str(sig.get("entry_to_exit_turn", "")).strip().lower()
    if man != "straight":
        return 0

    lanes = sig.get("lanes", [])
    if not isinstance(lanes, list) or len(lanes) < 3:
        return 0

    try:
        entry_lane = int(lanes[0])
        exit_lane = int(lanes[-1])
    except Exception:
        return 0

    # Only penalize if entry and exit are in the same lane (true straight)
    if entry_lane != exit_lane:
        return 0

    drift_count = 0
    for lid in lanes[1:-1]:
        try:
            if int(lid) != entry_lane:
                drift_count += 1
        except Exception:
            pass

    return drift_count


def _candidate_lane_inconsistency(cand: Dict[str, Any]) -> int:
    """Detect lane inconsistency: entry lane != exit lane, or intermediate drift.
    
    This applies to ALL maneuvers (straight, left, right) and penalizes paths where:
    1. Entry and exit are in different lanes (unexpected lane change)
    2. Intermediate segments deviate from expected lane progression
    3. Any mid-path lane jump (e.g., [-2, -1, -2] jumps to different lane then back)
    
    Returns a penalty count (0 = clean, 1+ = has issues).
    """
    sig = (cand or {}).get("signature", {})
    lanes = sig.get("lanes", [])
    if not isinstance(lanes, list) or len(lanes) < 2:
        return 0

    try:
        int(lanes[0])
        int(lanes[-1])
    except Exception:
        return 0

    # For turning maneuvers, the "natural" exit lane depends on the turn:
    # - Right turn from lane -1 naturally exits to rightmost lane (-1 or -2 depending on road)
    # - Left turn naturally exits to leftmost lane
    # But for simplicity, we check if lanes are inconsistent within entry/exit segments
    man = str(sig.get("entry_to_exit_turn", "")).strip().lower()
    
    # Count how many unique lanes are used
    unique_lanes = set()
    for lid in lanes:
        try:
            unique_lanes.add(int(lid))
        except Exception:
            pass
    
    # If more than 2 unique lanes, definitely has drift
    if len(unique_lanes) > 2:
        return len(unique_lanes) - 1
    
    # Check for mid-path lane jumps: any lane change within the path is suspicious
    # e.g., [-2, -1, -2] has a jump even though entry == exit
    # Count the number of lane transitions
    lane_transitions = 0
    for i in range(1, len(lanes)):
        try:
            prev_lane = int(lanes[i - 1])
            curr_lane = int(lanes[i])
            if curr_lane != prev_lane:
                lane_transitions += 1
        except Exception:
            pass
    
    # For straight paths, any lane transition is bad
    if man == "straight" and lane_transitions > 0:
        return lane_transitions
    
    # For turns, we expect exactly 1 transition (at the turn itself where roads change)
    # More than 1 transition means there's an unexpected lane jump
    if man in ("left", "right") and lane_transitions > 1:
        return lane_transitions - 1  # Subtract the expected turn transition
    
    return 0


def _candidate_lane_change_phase(cand: Dict[str, Any]) -> str:
    """Detect WHERE in the path the lane change occurs.
    
    Returns:
    - "before_intersection": lane change happens in the first segment (approach)
    - "in_intersection": lane change happens in the middle segment(s) (junction)
    - "after_intersection": lane change happens in the last segment (exit)
    - "none": no lane change detected
    - "multiple": lane changes in multiple phases
    """
    sig = (cand or {}).get("signature", {})
    lanes = sig.get("lanes", [])
    if not isinstance(lanes, list) or len(lanes) < 2:
        return "none"
    
    # For a 3-segment path: [approach, junction, exit]
    # segment 0 = before intersection
    # segment 1 = in intersection (or junction connector)
    # segment 2 = after intersection
    
    change_phases = set()
    prev_lane = None
    
    for i, lid in enumerate(lanes):
        try:
            lane_int = int(lid)
        except Exception:
            prev_lane = lid
            continue
        
        if prev_lane is not None and isinstance(prev_lane, int) and lane_int != prev_lane:
            # Lane change detected between segment i-1 and i
            # This means the change "happens" as we enter segment i
            n_segs = len(lanes)
            if n_segs <= 2:
                # Can't determine phase with only 2 segments
                change_phases.add("unknown")
            elif i == 1:
                # Change entering segment 1 (from approach to junction) = before/in intersection
                change_phases.add("in_intersection")
            elif i == n_segs - 1:
                # Change entering the last segment = after intersection
                change_phases.add("after_intersection")
            else:
                # Change in middle segments = in intersection
                change_phases.add("in_intersection")
        
        prev_lane = lane_int
    
    if not change_phases:
        return "none"
    if len(change_phases) > 1:
        return "multiple"
    return change_phases.pop()


def _candidate_geometric_discontinuity(cand: Dict[str, Any], threshold_m: float = 2.0) -> float:
    """Detect geometric discontinuity between segments.
    
    Even if lane IDs are consistent, the actual geometry may have gaps
    where one segment ends and the next begins at a different position.
    This happens at junctions where lane mappings don't align geometrically.
    
    Returns total gap distance in meters (0 = smooth, >0 = has gaps).
    """
    sig = (cand or {}).get("signature", {})
    segs = sig.get("segments_detailed", [])
    if not isinstance(segs, list) or len(segs) < 2:
        return 0.0
    
    total_gap = 0.0
    for i in range(len(segs) - 1):
        seg_curr = segs[i]
        seg_next = segs[i + 1]
        
        # Get end point of current segment and start point of next
        end_pt = seg_curr.get("end", {}).get("point", {})
        start_pt = seg_next.get("start", {}).get("point", {})
        
        try:
            end_x = float(end_pt.get("x", 0))
            end_y = float(end_pt.get("y", 0))
            start_x = float(start_pt.get("x", 0))
            start_y = float(start_pt.get("y", 0))
            
            # Calculate Euclidean distance
            gap = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5
            if gap > threshold_m:
                total_gap += gap
        except Exception:
            pass
    
    return total_gap


def _candidate_entry_in_crop(
    cand: Dict[str, Any],
    crop_region: Optional[Dict[str, Any]],
    margin: float = 5.0,
) -> bool:
    """
    Check if a candidate's entry point is within the crop region (with optional margin).
    
    Args:
        cand: Candidate path dictionary with signature.entry.point
        crop_region: Dict with xmin, xmax, ymin, ymax keys
        margin: Extra margin (meters) to expand the crop bounds
        
    Returns:
        True if entry point is inside crop (or if crop is not defined), False otherwise.
    """
    if not crop_region:
        return True  # No crop region = accept all
    
    if not all(k in crop_region for k in ("xmin", "xmax", "ymin", "ymax")):
        return True  # Invalid crop region = accept all
    
    entry_pt = _candidate_entry_point(cand)
    if entry_pt is None:
        return True  # Can't determine entry point = accept (don't over-constrain)
    
    x, y = entry_pt
    try:
        xmin = float(crop_region["xmin"]) - margin
        xmax = float(crop_region["xmax"]) + margin
        ymin = float(crop_region["ymin"]) - margin
        ymax = float(crop_region["ymax"]) + margin
    except (TypeError, ValueError):
        return True  # Invalid crop values = accept all
    
    return xmin <= x <= xmax and ymin <= y <= ymax


__all__ = [
    "_build_road_corridors",
    "_candidate_all_road_ids",
    "_candidate_entry_cardinal",
    "_candidate_entry_heading_rad",
    "_candidate_entry_in_crop",
    "_candidate_entry_lane_id",
    "_candidate_entry_point",
    "_candidate_entry_road_id",
    "_candidate_exit_cardinal",
    "_candidate_exit_lane_id",
    "_candidate_exit_road_id",
    "_candidate_geometric_discontinuity",
    "_candidate_lane_change_count",
    "_candidate_lane_change_phase",
    "_candidate_lane_inconsistency",
    "_candidate_length",
    "_candidate_straight_lateral_drift",
]
