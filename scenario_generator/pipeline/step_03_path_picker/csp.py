import math
from typing import Any, Dict, List, Optional, Tuple

from .candidates import (
    _build_road_corridors,
    _candidate_all_road_ids,
    _candidate_entry_cardinal,
    _candidate_entry_heading_rad,
    _candidate_entry_in_crop,
    _candidate_entry_lane_id,
    _candidate_entry_point,
    _candidate_entry_road_id,
    _candidate_exit_cardinal,
    _candidate_exit_lane_id,
    _candidate_exit_road_id,
    _candidate_geometric_discontinuity,
    _candidate_lane_change_count,
    _candidate_lane_change_phase,
    _candidate_lane_inconsistency,
    _candidate_length,
    _candidate_straight_lateral_drift,
)
from .constraints import (
    _candidate_matches_unary,
    _filter_constraints_with_evidence,
    _infer_constraints_from_description,
    _infer_road_role_sets,
    _normalize_constraints_obj,
    _opposite_cardinal,
)


def _lane_relation_ok(a_c: Dict[str, Any], b_c: Dict[str, Any], relation: str) -> Optional[bool]:
    """Return True/False if checkable, else None (insufficient signal => don't over-constrain)."""
    # Must be the same approach direction; lane relations are defined on approach.
    if _candidate_entry_cardinal(a_c) != _candidate_entry_cardinal(b_c):
        return False

    # If entry road_id is known for both, require them to match.
    rid_a = _candidate_entry_road_id(a_c)
    rid_b = _candidate_entry_road_id(b_c)
    if rid_a is not None and rid_b is not None and rid_a != rid_b:
        return False

    # If lane_id is known for both, prefer adjacent lanes.
    lid_a = _candidate_entry_lane_id(a_c)
    lid_b = _candidate_entry_lane_id(b_c)
    if lid_a is not None and lid_b is not None and abs(lid_a - lid_b) != 1:
        return False

    pa = _candidate_entry_point(a_c)
    pb = _candidate_entry_point(b_c)
    th = _candidate_entry_heading_rad(b_c)
    if pa is None or pb is None or th is None:
        return None

    dx = pa[0] - pb[0]
    dy = pa[1] - pb[1]
    # Left normal for b's heading.
    left_x = -math.sin(th)
    left_y = math.cos(th)
    lat = dx * left_x + dy * left_y

    # Small deadband to avoid numerical flicker.
    eps = 0.5
    if relation == "left":
        return lat > eps
    if relation == "right":
        return lat < -eps
    return None


def _consistent_with_constraints(
    assignment: Dict[str, Dict[str, Any]],
    candidate_for_v: Dict[str, Any],
    v: str,
    constraints: List[Dict[str, str]],
    road_corridors: Optional[Dict[int, set]] = None,
) -> bool:
    ent_v = _candidate_entry_cardinal(candidate_for_v)
    exrid_v = _candidate_exit_road_id(candidate_for_v)
    all_roads_v = _candidate_all_road_ids(candidate_for_v)

    for c in constraints:
        t = str(c.get("type", "")).strip().lower()
        a = c.get("a")
        b = c.get("b")
        if not a or not b:
            continue

        other = None
        if a == v and b in assignment:
            other = (b, assignment[b])
        elif b == v and a in assignment:
            other = (a, assignment[a])
        else:
            continue

        _, o_cand = other
        ent_o = _candidate_entry_cardinal(o_cand)
        exit_o = _candidate_exit_cardinal(o_cand)
        exrid_o = _candidate_exit_road_id(o_cand)
        all_roads_o = _candidate_all_road_ids(o_cand)

        if t == "same_approach_as":
            if ent_v != ent_o:
                return False
        elif t == "opposite_approach_of":
            if ent_v != _opposite_cardinal(ent_o):
                return False
        elif t == "perpendicular_right_of":
            # "a" approaches from perpendicular right of "b"
            # If b is W (westbound), a should be S (southbound) - 90° clockwise
            # W->S, S->E, E->N, N->W
            clockwise = {"W": "S", "S": "E", "E": "N", "N": "W"}
            if a == v:
                # v is "a", we need v's entry to be 90° clockwise from other's entry
                expected = clockwise.get(ent_o, None)
                if expected and ent_v != expected:
                    return False
            else:
                # v is "b", we need other's entry to be 90° clockwise from v's entry
                expected = clockwise.get(ent_v, None)
                if expected and ent_o != expected:
                    return False
        elif t == "perpendicular_left_of":
            # "a" approaches from perpendicular left of "b"
            # If b is W (westbound), a should be N (northbound) - 90° counter-clockwise
            # W->N, N->E, E->S, S->W
            counterclockwise = {"W": "N", "N": "E", "E": "S", "S": "W"}
            if a == v:
                expected = counterclockwise.get(ent_o, None)
                if expected and ent_v != expected:
                    return False
            else:
                expected = counterclockwise.get(ent_v, None)
                if expected and ent_o != expected:
                    return False
        elif t == "same_exit_as":
            # Both vehicles should have the same exit direction
            exit_v = _candidate_exit_cardinal(cand)
            if exit_v != exit_o:
                return False
        elif t == "same_road_as":
            # "same_road_as" means vehicles end up on the same logical road corridor.
            # This is satisfied if:
            # 1. Their exit road_ids are in the same corridor (connected by straight paths), OR
            # 2. One vehicle's exit road is among the roads the other travels on
            if exrid_v is None or exrid_o is None:
                # Cannot determine, don't over-constrain
                continue

            # Check if exit roads are in the same corridor
            same_corridor = False
            if road_corridors:
                corridor_v = road_corridors.get(exrid_v, {exrid_v})
                corridor_o = road_corridors.get(exrid_o, {exrid_o})
                if corridor_v & corridor_o:  # corridors overlap
                    same_corridor = True

            # Check if one's exit is in the other's traversed roads
            exit_in_other_path = (exrid_v in all_roads_o) or (exrid_o in all_roads_v)

            if not same_corridor and not exit_in_other_path:
                return False

    return True


def _solve_paths_csp(
    constraints_obj: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    description: str = "",
    require_straight: bool = False,
    require_on_ramp: bool = False,
    lane_counts_by_road: Optional[Dict[int, int]] = None,
    skip_evidence_filter: bool = False,
    crop_region: Optional[Dict[str, Any]] = None,
    required_relations: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Soft CSP/backtracking over candidate paths.
    Returns list of {"vehicle": ..., "path_name": ..., "confidence": ...} for each vehicle.
    Soft penalties are applied for constraint violations; the best-scoring assignment is chosen.
    
    If require_straight=True, heavily penalizes paths with turns (for CORRIDOR topology).
    
    If crop_region is provided, ALL candidates are filtered to only those whose
    entry point is strictly inside the crop region. This ensures role inference
    and path selection only consider the intended junction/area.
    """
    # EARLY CROP FILTERING: Filter candidates to only those inside the crop region.
    # This must happen BEFORE role inference so we don't confuse side/main classification.
    if crop_region:
        # Intersection/topology (non-corridor/on-ramp) needs a margin so entries just outside
        # the crop aren’t dropped; corridor/on-ramp stays strict on first pass.
        base_margin = 10.0 if (not require_straight and not require_on_ramp) else 0.0
        original_candidates = list(candidates)
        original_count = len(original_candidates)

        def _clamp_point_to_box(pt: Optional[tuple], box: Dict[str, Any], m: float) -> Optional[tuple]:
            if pt is None or not box:
                return pt
            try:
                x, y = pt
                xmin = float(box["xmin"]) - m
                xmax = float(box["xmax"]) + m
                ymin = float(box["ymin"]) - m
                ymax = float(box["ymax"]) + m
                cx = min(max(x, xmin), xmax)
                cy = min(max(y, ymin), ymax)
                return (cx, cy)
            except Exception:
                return pt

        def _filter_with_margin(margin: float) -> Tuple[List[Dict[str, Any]], int]:
            inside = [c for c in original_candidates if _candidate_entry_in_crop(c, crop_region, margin=margin)]
            inside_road_ids = {_candidate_entry_road_id(c) for c in inside}
            inside_cardinals = {_candidate_entry_cardinal(c) for c in inside}
            extras: List[Dict[str, Any]] = []
            for c in original_candidates:
                if c in inside:
                    continue
                rid = _candidate_entry_road_id(c)
                ent_card = _candidate_entry_cardinal(c)
                is_same_road = rid is not None and rid in inside_road_ids
                opp_map = {"N": "S", "S": "N", "E": "W", "W": "E"}
                is_opposite_card = ent_card and opp_map.get(ent_card) in inside_cardinals
                if not (is_same_road or is_opposite_card):
                    continue
                sig = c.get("signature", {})
                ent = sig.get("entry", {})
                pt = ent.get("point")
                clamped = _clamp_point_to_box(
                    (pt.get("x"), pt.get("y")) if isinstance(pt, dict) else None,
                    crop_region,
                    margin,
                )
                if clamped:
                    ent["point"] = {"x": clamped[0], "y": clamped[1]}
                    sig["entry"] = ent
                    c["signature"] = sig
                extras.append(c)
            return inside + extras, len(extras)

        candidates, extras_count = _filter_with_margin(base_margin)
        print(
            f"[INFO] CSP: Crop region filter: {original_count} -> {len(candidates)} candidates "
            f"(margin={base_margin}, strict=True, kept_same_road_extras={extras_count})"
        )
        if not candidates:
            relaxed_margin = max(10.0, base_margin + 15.0)
            relaxed_candidates, relaxed_extras_count = _filter_with_margin(relaxed_margin)
            print(
                f"[WARN] CSP: strict crop filter produced zero candidates; retrying relaxed margin {relaxed_margin}m "
                f"-> {len(relaxed_candidates)} candidates (kept_same_road_extras={relaxed_extras_count})"
            )
            if relaxed_candidates:
                candidates = relaxed_candidates
            else:
                # Last-resort guardrail: keep search alive instead of cascading to LLM-only fallback.
                candidates = original_candidates
                print(
                    f"[WARN] CSP: relaxed crop filter still empty; using unfiltered candidates "
                    f"({len(candidates)}) as fallback."
                )
    
    if require_straight:
        print("[INFO] CSP: require_straight=True, filtering to straight paths only")
    norm = _normalize_constraints_obj(constraints_obj)
    if not norm:
        raise ValueError("Invalid constraints object (missing vehicles list).")

    if not skip_evidence_filter:
        norm = _filter_constraints_with_evidence(description, norm)
        norm = _infer_constraints_from_description(description, norm)
    vehicles = [v["vehicle"] for v in norm["vehicles"]]
    unary = {v["vehicle"]: v for v in norm["vehicles"]}
    constraints = norm.get("constraints", [])
    
    # DEBUG: print extracted constraints
    print(f"[DEBUG CSP] Extracted constraints: {constraints}")
    for v_obj in norm["vehicles"]:
        print(f"[DEBUG CSP] Vehicle {v_obj.get('vehicle')}: maneuver={v_obj.get('maneuver')}")

    # Identify which vehicles are explicitly performing lane changes
    # This is determined by the LLM's extracted maneuver field - no regex needed
    vehicles_with_lane_change: set = set()
    for v in vehicles:
        if unary.get(v, {}).get("maneuver", "").lower() == "lane_change":
            vehicles_with_lane_change.add(v)
    
    # Extract lane change phase requirements for vehicles doing lane changes
    lane_change_phase_req: Dict[str, str] = {}
    for v in vehicles_with_lane_change:
        phase = str(unary.get(v, {}).get("lane_change_phase", "unknown")).strip().lower()
        if phase in ("before_intersection", "after_intersection"):
            lane_change_phase_req[v] = phase
    
    if lane_change_phase_req:
        print(f"[INFO] Lane change phase requirements: {lane_change_phase_req}")

    def _lane_count_for_road(road_id: Optional[int]) -> Optional[int]:
        if not lane_counts_by_road or road_id is None:
            return None
        try:
            return int(lane_counts_by_road.get(road_id))
        except Exception:
            pass
        try:
            return int(lane_counts_by_road.get(str(road_id)))
        except Exception:
            return None

    def _candidate_lane_renumbering_ok(cand: Dict[str, Any], gap_threshold_m: float = 2.0) -> bool:
        if not lane_counts_by_road:
            return False
        sig = (cand or {}).get("signature", {})
        lanes = sig.get("lanes", [])
        roads = sig.get("roads", [])
        segs = sig.get("segments_detailed", [])
        if not isinstance(lanes, list) or not isinstance(roads, list):
            return False
        if len(lanes) < 2 or len(lanes) != len(roads):
            return False

        for i in range(1, len(lanes)):
            try:
                prev_lane = int(lanes[i - 1])
                lane = int(lanes[i])
                prev_road = int(roads[i - 1])
                road = int(roads[i])
            except Exception:
                return False

            if lane == prev_lane:
                continue

            if prev_road == road:
                return False

            prev_count = _lane_count_for_road(prev_road)
            cur_count = _lane_count_for_road(road)
            if prev_count is None or cur_count is None:
                return False
            if abs(cur_count - prev_count) != 1:
                return False
            if abs(lane - prev_lane) != 1:
                return False

            if isinstance(segs, list) and i - 1 < len(segs) and i < len(segs):
                end_pt = segs[i - 1].get("end", {}).get("point", {})
                start_pt = segs[i].get("start", {}).get("point", {})
                try:
                    end_x = float(end_pt.get("x", 0))
                    end_y = float(end_pt.get("y", 0))
                    start_x = float(start_pt.get("x", 0))
                    start_y = float(start_pt.get("y", 0))
                    gap = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5
                    if gap > gap_threshold_m:
                        return False
                except Exception:
                    return False

        return True

    def _candidate_lane_consistent(cand: Dict[str, Any]) -> bool:
        return _candidate_lane_inconsistency(cand) == 0 or _candidate_lane_renumbering_ok(cand)

    adjacent_lane_req: Dict[str, str] = {}
    for v in vehicles:
        req = str(unary.get(v, {}).get("adjacent_lane_req", "")).strip().lower()
        if req in ("left", "right"):
            adjacent_lane_req[v] = req

    def _candidate_has_adjacent_lane(cand: Dict[str, Any], relation: str) -> Optional[bool]:
        saw_unknown = False
        for other in candidates:
            if other is cand:
                continue
            ok = _lane_relation_ok(other, cand, relation)
            if ok is True:
                return True
            if ok is None:
                saw_unknown = True
        if saw_unknown:
            return None
        return False

    def _lane_cohesion_adjust(
        cand_v: Dict[str, Any],
        cand_o: Dict[str, Any],
        v_name: str,
        o_name: str,
    ) -> float:
        """
        Soft bias: if two vehicles share the same entry road and have compatible maneuvers
        (straight/right, no pre-intersection lane change), reward same-lane or adjacent-lane starts.
        Apply only when approach direction matches and lane ids are known.
        """
        u_v = unary.get(v_name, {})
        u_o = unary.get(o_name, {})
        entry_road_v = str(u_v.get("entry_road", "unknown")).lower()
        entry_road_o = str(u_o.get("entry_road", "unknown")).lower()
        if not entry_road_v or entry_road_v != entry_road_o:
            return 0.0

        # Require matching approach cardinals to avoid cross approaches.
        if _candidate_entry_cardinal(cand_v) != _candidate_entry_cardinal(cand_o):
            return 0.0

        # Skip if either vehicle needs a lane change before the intersection.
        if v_name in vehicles_with_lane_change or o_name in vehicles_with_lane_change:
            return 0.0

        allowed_maneuvers = {"straight", "right"}
        man_v = str(u_v.get("maneuver", "unknown")).lower()
        man_o = str(u_o.get("maneuver", "unknown")).lower()
        if man_v not in allowed_maneuvers or man_o not in allowed_maneuvers:
            return 0.0

        lid_v = _candidate_entry_lane_id(cand_v)
        lid_o = _candidate_entry_lane_id(cand_o)
        if lid_v is None or lid_o is None:
            return 0.0

        if lid_v == lid_o:
            return -0.2  # prefer same lane when safe
        diff = abs(lid_v - lid_o)
        if diff == 1:
            return -0.1  # prefer adjacent lane
        return 0.05  # mild penalty for widely separated lanes

    def _candidate_effective_length(cand: Dict[str, Any], vehicle: str) -> float:
        base = _candidate_length(cand)
        
        # If require_straight, heavily penalize turning paths
        if require_straight:
            turn = str((cand.get("signature") or {}).get("entry_to_exit_turn", "unknown")).lower()
            if turn in ("left", "right", "uturn"):
                return max(0.0, base - 1000.0)  # Heavy penalty for turns in corridor scenarios
        
        # Only skip penalty if THIS vehicle is doing a lane change
        if vehicle in vehicles_with_lane_change:
            # Check if there's a phase requirement
            required_phase = lane_change_phase_req.get(vehicle)
            if required_phase:
                actual_phase = _candidate_lane_change_phase(cand)
                # Heavy penalty if lane change happens at wrong phase
                if actual_phase not in ("none", "unknown", required_phase):
                    return 0.0  # Zero score for wrong phase
                # Bonus for matching phase
                if actual_phase == required_phase:
                    return base + 100.0  # Large bonus for correct phase
            return base
        
        # Lane inconsistency: penalize paths that jump between lanes unexpectedly.
        # The refiner handles intentional lane changes, so the path picker should
        # prefer lane-consistent paths. This prevents selecting paths that randomly
        # jump through junction connectors to different lanes.
        # ONLY apply for intersection scenarios (not corridor or on-ramp).
        is_intersection = not require_straight and not require_on_ramp
        inconsistency = _candidate_lane_inconsistency(cand)
        if inconsistency > 0:
            print(f"[DEBUG LANE] {cand.get('name','?')[:40]} incon={inconsistency} is_int={is_intersection} req_str={require_straight} req_ramp={require_on_ramp}")
        if is_intersection and inconsistency > 0:
            # Heavy penalty for lane jumps - prefer staying in same lane
            penalized = max(0.0, base - 300.0 * inconsistency)
            print(f"[DEBUG LANE] -> PENALIZED from {base:.1f} to {penalized:.1f}")
            return penalized
        
        # Geometric discontinuity (gaps between segments) as secondary check
        geo_gap = _candidate_geometric_discontinuity(cand, threshold_m=2.0)
        
        if geo_gap == 0:
            # Geometrically smooth and lane-consistent path
            # Only penalize if there's actual lane drift in straight paths
            drift_count = _candidate_straight_lateral_drift(cand)
            if drift_count > 0:
                return max(0.0, base - 200.0 * drift_count)
            return base
        
        # Path has geometric gaps - apply penalties
        lc_count = _candidate_lane_change_count(cand)
        drift_count = _candidate_straight_lateral_drift(cand)
        
        # Combine penalties
        lane_penalty = max(lc_count + drift_count, inconsistency) * 200.0
        geo_penalty = geo_gap * 50.0  # 50m penalty per meter of gap
        
        total_penalty = lane_penalty + geo_penalty
        
        return max(0.0, base - total_penalty)

    # For on-ramps, we need to identify ramp candidates FIRST, then use them to build role sets
    road_corridors = _build_road_corridors(candidates)

    def _corridor_key(road_id: Optional[int]) -> Optional[frozenset]:
        if road_id is None:
            return None
        return frozenset(road_corridors.get(road_id, {road_id}))

    main_exit_corridor: Optional[frozenset] = None
    ramp_candidates: set = set()
    mainline_candidates: set = set()
    clean_mainline_candidates: set = set()
    smooth_mainline_candidates: set = set()
    prefer_merge_lane: Optional[int] = None
    merge_targets: set = set()
    merge_sources: set = set()
    if require_on_ramp:
        lane_count_by_corridor: Dict[frozenset, int] = {}
        if lane_counts_by_road:
            for rid_raw, count_raw in lane_counts_by_road.items():
                try:
                    rid_i = int(rid_raw)
                    count_i = int(count_raw)
                except Exception:
                    continue
                ck = _corridor_key(rid_i)
                if ck is None:
                    continue
                lane_count_by_corridor[ck] = max(lane_count_by_corridor.get(ck, 0), count_i)
        else:
            lane_ids_by_road: Dict[int, set] = {}
            for c in candidates:
                sig = (c or {}).get("signature", {})
                roads = sig.get("roads", [])
                lanes = sig.get("lanes", [])
                if isinstance(roads, list) and isinstance(lanes, list):
                    for rid, lid in zip(roads, lanes):
                        try:
                            rid_i = int(rid)
                            lid_i = int(lid)
                        except Exception:
                            continue
                        lane_ids_by_road.setdefault(rid_i, set()).add(lid_i)
                try:
                    ent = int(sig["entry"]["road_id"])
                    ent_l = int(sig["entry"]["lane_id"])
                    lane_ids_by_road.setdefault(ent, set()).add(ent_l)
                except Exception:
                    pass
                try:
                    ex = int(sig["exit"]["road_id"])
                    ex_l = int(sig["exit"]["lane_id"])
                    lane_ids_by_road.setdefault(ex, set()).add(ex_l)
                except Exception:
                    pass

            for rid, lanes in lane_ids_by_road.items():
                ck = _corridor_key(rid)
                if ck is None:
                    continue
                lane_count_by_corridor[ck] = max(lane_count_by_corridor.get(ck, 0), len(lanes))

        exit_corridor_counts: Dict[frozenset, int] = {}
        for c in candidates:
            exrid = _candidate_exit_road_id(c)
            ck = _corridor_key(exrid)
            if ck is None:
                continue
            exit_corridor_counts[ck] = exit_corridor_counts.get(ck, 0) + 1
        if exit_corridor_counts:
            main_exit_corridor = max(exit_corridor_counts.items(), key=lambda kv: kv[1])[0]

        ramp_options: List[Tuple[int, str, int]] = []
        if main_exit_corridor:
            main_lanes = lane_count_by_corridor.get(main_exit_corridor, 0)
            for c in candidates:
                exrid = _candidate_exit_road_id(c)
                entrid = _candidate_entry_road_id(c)
                exit_ck = _corridor_key(exrid)
                entry_ck = _corridor_key(entrid)
                if exit_ck is None or entry_ck is None:
                    continue
                if exit_ck != main_exit_corridor:
                    continue
                if entry_ck == main_exit_corridor:
                    continue
                entry_lanes = lane_count_by_corridor.get(entry_ck, 0)
                if not (main_lanes >= 3 and entry_lanes > 0 and entry_lanes < main_lanes):
                    continue
                if entry_lanes > 2:
                    continue
                turn = str((c.get("signature") or {}).get("entry_to_exit_turn", "")).strip().lower()
                if turn == "uturn":
                    continue
                name = c.get("name")
                exit_lane = _candidate_exit_lane_id(c)
                if name and exit_lane is not None:
                    ramp_options.append((entry_lanes, name, exit_lane))

        if ramp_options:
            prefer_single = any(entry_lanes == 1 for entry_lanes, _, _ in ramp_options)
            max_entry_lanes = 1 if prefer_single else 2
            for entry_lanes, name, _ in ramp_options:
                if entry_lanes <= max_entry_lanes:
                    ramp_candidates.add(name)
            if prefer_single:
                print("[INFO] CSP: prefer single-lane ramp candidates")

        if ramp_candidates:
            print(f"[INFO] CSP: on-ramp candidates available: {len(ramp_candidates)}")
        else:
            print("[WARN] CSP: no on-ramp candidates found; skipping on-ramp enforcement")
    
    # Now infer role sets, using ramp candidates if available
    role_sets = _infer_road_role_sets(candidates, ramp_candidate_names=ramp_candidates if require_on_ramp else None)
    
    if require_on_ramp:
        if main_exit_corridor:
            for c in candidates:
                exrid = _candidate_exit_road_id(c)
                entrid = _candidate_entry_road_id(c)
                exit_ck = _corridor_key(exrid)
                entry_ck = _corridor_key(entrid)
                if exit_ck is None or entry_ck is None:
                    continue
                if exit_ck == main_exit_corridor and entry_ck == main_exit_corridor:
                    name = c.get("name")
                    if not name:
                        continue
                    mainline_candidates.add(name)
                    if _candidate_lane_consistent(c):
                        clean_mainline_candidates.add(name)
                        if _candidate_geometric_discontinuity(c, threshold_m=2.0) == 0:
                            smooth_mainline_candidates.add(name)

        if ramp_candidates and mainline_candidates:
            ramp_exit_counts: Dict[int, int] = {}
            main_exit_lanes: set = set()
            clean_main_exit_lanes: set = set()
            for c in candidates:
                name = c.get("name")
                if not name:
                    continue
                exit_lane = _candidate_exit_lane_id(c)
                if exit_lane is None:
                    continue
                if name in ramp_candidates:
                    ramp_exit_counts[exit_lane] = ramp_exit_counts.get(exit_lane, 0) + 1
                if name in mainline_candidates:
                    main_exit_lanes.add(exit_lane)
                if name in clean_mainline_candidates:
                    clean_main_exit_lanes.add(exit_lane)
            if ramp_exit_counts:
                shared_lanes = [lane for lane in ramp_exit_counts if lane in main_exit_lanes]
                if shared_lanes:
                    preferred_pool = [lane for lane in shared_lanes if lane in clean_main_exit_lanes] or shared_lanes
                    prefer_merge_lane = max(preferred_pool, key=lambda lane: (ramp_exit_counts.get(lane, 0)))
                    print(f"[INFO] CSP: prefer merge lane {prefer_merge_lane}")

    # Domains (filtered by unary constraints)
    domains: Dict[str, List[Dict[str, Any]]] = {}
    if require_on_ramp:
        for c in constraints:
            if str(c.get("type", "")).strip().lower() == "merges_into_lane_of":
                a = c.get("a")
                b = c.get("b")
                if a:
                    merge_sources.add(a)
                if b:
                    merge_targets.add(b)
        if not merge_targets:
            side_vs = [v for v in vehicles if unary.get(v, {}).get("entry_road") == "side"]
            main_vs = [v for v in vehicles if unary.get(v, {}).get("entry_road") == "main"]
            if side_vs and main_vs:
                preferred_target = None
                for v in main_vs:
                    if str(v).strip().lower() == "vehicle 1":
                        preferred_target = v
                        break
                if preferred_target is None:
                    preferred_target = sorted(main_vs)[0]
                merge_targets.add(preferred_target)
    for v in vehicles:
        u = unary.get(v, {})
        man = str(u.get("maneuver", "unknown")).strip().lower()
        # Treat "lane_change" as "straight" for path filtering (they go straight but change lanes)
        if man == "lane_change":
            man = "straight"
        er = str(u.get("entry_road", "unknown")).strip().lower()
        xr = str(u.get("exit_road", "unknown")).strip().lower()
        if require_on_ramp and er == "side" and man == "straight":
            man = "unknown"
        dom = [c for c in candidates if _candidate_matches_unary(c, man, er, xr, role_sets)]
        
        if require_on_ramp:
            if er == "side" and ramp_candidates:
                dom_side = [c for c in dom if c.get("name") in ramp_candidates]
                if prefer_merge_lane is not None:
                    dom_lane = [c for c in dom_side if _candidate_exit_lane_id(c) == prefer_merge_lane]
                    if dom_lane:
                        dom_side = dom_lane
                if dom_side:
                    dom = dom_side
            elif er == "main" and mainline_candidates:
                dom_main = [c for c in dom if c.get("name") in mainline_candidates]
                if v in merge_targets and prefer_merge_lane is not None:
                    dom_lane = [c for c in dom_main if _candidate_exit_lane_id(c) == prefer_merge_lane]
                    if dom_lane:
                        dom_main = dom_lane
                if dom_main:
                    dom = dom_main
                if smooth_mainline_candidates:
                    dom_smooth = [c for c in dom if c.get("name") in smooth_mainline_candidates]
                    if dom_smooth:
                        dom = dom_smooth
                if clean_mainline_candidates:
                    dom_clean = [c for c in dom if c.get("name") in clean_mainline_candidates]
                    if dom_clean:
                        dom = dom_clean
        adj_req = adjacent_lane_req.get(v)
        if adj_req:
            dom_adj = [c for c in dom if _candidate_has_adjacent_lane(c, adj_req) is True]
            if dom_adj:
                dom = dom_adj
            else:
                print(f"[WARN] No candidates with a {adj_req} adjacent lane for {v}; keeping all.")

        # For straight maneuvers without an explicit lane_change intent, drop
        # lane-inconsistent paths (e.g., [-2,-1,-2]) to avoid mid-intersection hops.
        # ONLY for INTERSECTION topology (not corridor or on-ramp)
        # Also apply to turn maneuvers to avoid weird lane jumps during turns
        is_intersection_topology = not require_straight and not require_on_ramp
        if is_intersection_topology and v not in vehicles_with_lane_change:
            dom_clean = [c for c in dom if _candidate_lane_inconsistency(c) == 0]
            # Write debug to file since prints are suppressed
            with open("csp_debug.log", "a") as f:
                f.write(f"[DEBUG FILTER] {v}: man={man}, dom={len(dom)}, clean={len(dom_clean)}, is_int={is_intersection_topology}\n")
            if dom_clean:
                with open("csp_debug.log", "a") as f:
                    f.write(f"[DEBUG FILTER] {v}: Using {len(dom_clean)} clean paths instead of {len(dom)}\n")
                dom = dom_clean
            else:
                with open("csp_debug.log", "a") as f:
                    f.write(f"[WARN] {v}: No lane-consistent paths available; keeping all {len(dom)} paths\n")

        dom.sort(key=lambda c: (-_candidate_effective_length(c, v), str(c.get("name", ""))))
        domains[v] = dom

    for v, dom in domains.items():
        if not dom:
            raise ValueError(f"No candidates satisfy unary constraints for {v}.")

    order = sorted(vehicles, key=lambda v: (len(domains[v]), v))

    # Soft scoring: maximize (total_length - penalty_weight * violations)
    penalty_weight = 100.0
    max_len_per_v = {v: max(_candidate_effective_length(c, v) for c in domains[v]) for v in vehicles}

    def _constraint_penalty(c_v: Dict[str, Any], c_o: Dict[str, Any], t: str, v_is_a: bool) -> float:
        ent_v = _candidate_entry_cardinal(c_v)
        ent_o = _candidate_entry_cardinal(c_o)
        exrid_v = _candidate_exit_road_id(c_v)
        exrid_o = _candidate_exit_road_id(c_o)
        all_roads_v = _candidate_all_road_ids(c_v)
        all_roads_o = _candidate_all_road_ids(c_o)

        if t == "same_approach_as":
            return 0.0 if ent_v == ent_o else 1.0
        if t == "opposite_approach_of":
            return 0.0 if ent_v == _opposite_cardinal(ent_o) else 1.0
        if t == "perpendicular_right_of":
            # "a" approaches from perpendicular right of "b"
            # If b is W (westbound), a should be S (southbound) - 90° clockwise
            clockwise = {"W": "S", "S": "E", "E": "N", "N": "W"}
            if v_is_a:
                expected = clockwise.get(ent_o, None)
                return 0.0 if expected and ent_v == expected else 1.0
            else:
                expected = clockwise.get(ent_v, None)
                return 0.0 if expected and ent_o == expected else 1.0
        if t == "perpendicular_left_of":
            # "a" approaches from perpendicular left of "b"
            counterclockwise = {"W": "N", "N": "E", "E": "S", "S": "W"}
            if v_is_a:
                expected = counterclockwise.get(ent_o, None)
                return 0.0 if expected and ent_v == expected else 1.0
            else:
                expected = counterclockwise.get(ent_v, None)
                return 0.0 if expected and ent_o == expected else 1.0
        if t == "same_exit_as":
            exit_v = _candidate_exit_cardinal(c_v)
            exit_o = _candidate_exit_cardinal(c_o)
            return 0.0 if exit_v == exit_o else 1.0
        if t == "same_road_as":
            if exrid_v is None or exrid_o is None:
                return 0.0
            same_corridor = False
            if road_corridors:
                corridor_v = road_corridors.get(exrid_v, {exrid_v})
                corridor_o = road_corridors.get(exrid_o, {exrid_o})
                if corridor_v & corridor_o:
                    same_corridor = True
            exit_in_other_path = (exrid_v in all_roads_o) or (exrid_o in all_roads_v)
            return 0.0 if (same_corridor or exit_in_other_path) else 1.0
        if t == "follow_route_of":
            # Strong preference that the follower takes the same *route* as the leader:
            # same approach + same maneuver + same exit corridor (when checkable).
            turn_v = str(((c_v.get("signature") or {}).get("entry_to_exit_turn", ""))).strip().lower()
            turn_o = str(((c_o.get("signature") or {}).get("entry_to_exit_turn", ""))).strip().lower()

            if ent_v != ent_o:
                return 1.0

            # If either turn is unknown, don't over-constrain on turn; but if both known, require match.
            if turn_v and turn_o and (turn_v != turn_o):
                return 1.0

            if exrid_v is None or exrid_o is None:
                return 0.0

            corridor_v = road_corridors.get(exrid_v, {exrid_v}) if road_corridors else {exrid_v}
            corridor_o = road_corridors.get(exrid_o, {exrid_o}) if road_corridors else {exrid_o}
            same_corridor = bool(corridor_v & corridor_o)
            exit_in_other_path = (exrid_v in all_roads_o) or (exrid_o in all_roads_v)
            return 0.0 if (same_corridor or exit_in_other_path) else 1.0
        if t == "same_lane_as":
            # Symmetric: both vehicles must be in the SAME physical lane.
            # Same approach direction required.
            if _candidate_entry_cardinal(c_v) != _candidate_entry_cardinal(c_o):
                return 1.0  # Different approach = can't be same lane
            lid_v = _candidate_entry_lane_id(c_v)
            lid_o = _candidate_entry_lane_id(c_o)
            if lid_v is None or lid_o is None:
                return 0.0  # Can't determine, don't over-constrain
            return 0.0 if lid_v == lid_o else 1.0
        if t == "left_lane_of":
            # Asymmetric: "a" is in the lane to the left of "b" (on approach).
            ok = _lane_relation_ok(c_v, c_o, "left") if v_is_a else _lane_relation_ok(c_o, c_v, "left")
            return 0.0 if (ok is None or ok is True) else 1.0
        if t == "right_lane_of":
            # Asymmetric: "a" is in the lane to the right of "b" (on approach).
            ok = _lane_relation_ok(c_v, c_o, "right") if v_is_a else _lane_relation_ok(c_o, c_v, "right")
            return 0.0 if (ok is None or ok is True) else 1.0
        if t == "adjacent_lane_of":
            # Symmetric: vehicles must be in different (adjacent) lanes on approach.
            # Same approach direction required.
            if _candidate_entry_cardinal(c_v) != _candidate_entry_cardinal(c_o):
                return 1.0  # Different approach = can't check lane adjacency
            lid_v = _candidate_entry_lane_id(c_v)
            lid_o = _candidate_entry_lane_id(c_o)
            if lid_v is None or lid_o is None:
                return 0.0  # Can't determine, don't over-constrain
            if lid_v == lid_o:
                return 1.0  # Same lane = violation
            if abs(lid_v - lid_o) == 1:
                return 0.0  # Adjacent lanes = satisfied
            return 0.5  # Not adjacent but different = partial penalty
        if t == "merges_into_lane_of":
            # Asymmetric: "a" starts in adjacent lane but ENDS in same lane as "b"
            # This is for lane change scenarios where vehicle a merges into vehicle b's lane
            # Same approach direction required
            if _candidate_entry_cardinal(c_v) != _candidate_entry_cardinal(c_o):
                return 1.0  # Different approach = can't merge
            
            if v_is_a:
                # c_v is the merger (a), c_o is the reference (b)
                entry_lid_v = _candidate_entry_lane_id(c_v)
                entry_lid_o = _candidate_entry_lane_id(c_o)
                exit_lid_v = _candidate_exit_lane_id(c_v)
                exit_lid_o = _candidate_exit_lane_id(c_o)
            else:
                # c_o is the merger (a), c_v is the reference (b)
                entry_lid_v = _candidate_entry_lane_id(c_o)
                entry_lid_o = _candidate_entry_lane_id(c_v)
                exit_lid_v = _candidate_exit_lane_id(c_o)
                exit_lid_o = _candidate_exit_lane_id(c_v)
            
            if entry_lid_v is None or entry_lid_o is None or exit_lid_v is None or exit_lid_o is None:
                return 0.0  # Can't determine, don't over-constrain
            
            penalty = 0.0
            # Entry lanes must be DIFFERENT (adjacent preferred)
            if entry_lid_v == entry_lid_o:
                penalty += 1.0  # Same entry lane = violation (should start in different lane)
            elif abs(entry_lid_v - entry_lid_o) != 1:
                penalty += 0.3  # Not adjacent = partial penalty
            
            # Exit lanes must be the SAME (merge into reference's lane)
            if exit_lid_v != exit_lid_o:
                penalty += 1.0  # Different exit lane = violation (should merge into same lane)

            if prefer_merge_lane is not None:
                if exit_lid_v != prefer_merge_lane or exit_lid_o != prefer_merge_lane:
                    penalty += 2.0

            return penalty
        return 0.0

    best_assignment: Optional[Dict[str, Dict[str, Any]]] = None
    best_score = -1e18

    # Required relations helpers (pair-agnostic, orientation-aware)
    def _cardinal_relation(a: str, b: str) -> str:
        if a == b and a != "unknown":
            return "same"
        opposite = {("N", "S"), ("S", "N"), ("E", "W"), ("W", "E")}
        if (a, b) in opposite:
            return "opposite"
        perpendicular = {
            ("N", "E"), ("N", "W"),
            ("S", "E"), ("S", "W"),
            ("E", "N"), ("W", "N"),
            ("E", "S"), ("W", "S"),
        }
        if (a, b) in perpendicular:
            return "perpendicular"
        return "any"

    def _lane_relation(lid_a: Optional[int], lid_b: Optional[int]) -> str:
        if lid_a is None or lid_b is None:
            return "any"
        if lid_a == lid_b:
            return "same_lane"
        if abs(lid_a - lid_b) == 1:
            return "adjacent_lane"
        return "any"

    def _relation_ok_oriented(rel: Dict[str, Any], cand_a: Dict[str, Any], cand_b: Dict[str, Any]) -> bool:
        sig_a = cand_a.get("signature", {})
        sig_b = cand_b.get("signature", {})
        man_a = str(sig_a.get("entry_to_exit_turn", "unknown")).lower()
        man_b = str(sig_b.get("entry_to_exit_turn", "unknown")).lower()
        rm1 = str(rel.get("first_maneuver", "unknown")).lower()
        rm2 = str(rel.get("second_maneuver", "unknown")).lower()
        if rm1 != "unknown" and man_a != rm1:
            return False
        if rm2 != "unknown" and man_b != rm2:
            return False

        ent_a = _candidate_entry_cardinal(cand_a)
        ent_b = _candidate_entry_cardinal(cand_b)
        if ent_a and ent_b:
            cr = _cardinal_relation(ent_a, ent_b)
            erel = rel.get("entry_relation", "any")
            if erel != "any" and cr != erel:
                return False

        ex_road_a = _candidate_exit_road_id(cand_a)
        ex_road_b = _candidate_exit_road_id(cand_b)
        ex_rel = rel.get("exit_relation", "any")
        if ex_rel == "same_exit":
            if ex_road_a is None or ex_road_b is None or ex_road_a != ex_road_b:
                return False
        elif ex_rel == "different_exit":
            if ex_road_a is not None and ex_road_b is not None and ex_road_a == ex_road_b:
                return False

        ent_lane_a = _candidate_entry_lane_id(cand_a)
        ent_lane_b = _candidate_entry_lane_id(cand_b)
        elr = rel.get("entry_lane_relation", "any")
        if elr != "any" and _lane_relation(ent_lane_a, ent_lane_b) != elr:
            return False

        exit_lane_a = _candidate_exit_lane_id(cand_a)
        exit_lane_b = _candidate_exit_lane_id(cand_b)
        xlr = rel.get("exit_lane_relation", "any")
        if xlr == "merge_into":
            if exit_lane_a is None or exit_lane_b is None or exit_lane_a != exit_lane_b:
                return False
        elif xlr != "any" and _lane_relation(exit_lane_a, exit_lane_b) != xlr:
            return False

        return True

    def _assignment_satisfies_required_relations(asn: Dict[str, Dict[str, Any]]) -> bool:
        if not required_relations:
            return True
        names = list(asn.keys())
        n = len(names)
        for rel in required_relations:
            satisfied = False
            for i in range(n):
                for j in range(i + 1, n):
                    a = asn[names[i]]
                    b = asn[names[j]]
                    if _relation_ok_oriented(rel, a, b) or _relation_ok_oriented(rel, b, a):
                        satisfied = True
                        break
                if satisfied:
                    break
            if not satisfied:
                return False
        return True

    def backtrack(i: int, asn: Dict[str, Dict[str, Any]], cur_len: float, cur_pen: float, has_ramp: bool):
        nonlocal best_assignment, best_score
        if i >= len(order):
            if require_on_ramp and ramp_candidates and not has_ramp:
                return
            if not _assignment_satisfies_required_relations(asn):
                return
            score = cur_len - penalty_weight * cur_pen
            if score > best_score:
                best_score = score
                best_assignment = dict(asn)
            return

        # optimistic bound for pruning
        remaining = sum(max_len_per_v[v] for v in order[i:])
        bound = cur_len + remaining - penalty_weight * cur_pen
        if bound < best_score:
            return

        vname = order[i]
        for cand in domains[vname]:
            add_pen = 0.0
            for c in constraints:
                t = str(c.get("type", "")).strip().lower()
                a = c.get("a")
                b = c.get("b")
                if not a or not b:
                    continue
                other = None
                if a == vname and b in asn:
                    other = asn[b]
                elif b == vname and a in asn:
                    other = asn[a]
                if other is not None:
                    add_pen += _constraint_penalty(cand, other, t, v_is_a=(a == vname))
                    # Soft cohesion bonus/penalty for same-entry vehicles with compatible maneuvers
                    add_pen += _lane_cohesion_adjust(cand, other, vname, a if b == vname else b)
            asn[vname] = cand
            new_has_ramp = has_ramp or (cand.get("name") in ramp_candidates)
            backtrack(
                i + 1,
                asn,
                cur_len + _candidate_effective_length(cand, vname),
                cur_pen + add_pen,
                new_has_ramp,
            )
            del asn[vname]

    backtrack(0, {}, 0.0, 0.0, False)

    if not best_assignment:
        raise ValueError("No feasible assignment found.")

    # Print final assignment summary
    print("[DEBUG CSP] Final assignment:")
    for v in vehicles:
        c = best_assignment[v]
        sig = c.get("signature", {})
        lanes = sig.get("lanes", [])
        entry_lane = sig.get("entry", {}).get("lane_id", "?")
        entry_card = sig.get("entry", {}).get("cardinal4", "?")
        is_lc = v in vehicles_with_lane_change
        eff_len = _candidate_effective_length(c, v)
        incon = _candidate_lane_inconsistency(c)
        print(f"  {v}: entry={entry_card} lane={entry_lane} lanes={lanes} eff_len={eff_len:.0f}m incon={incon} lc_vehicle={is_lc}")

    out = []
    for v in vehicles:
        c = best_assignment[v]
        # Get unary constraint info including entry_road
        u = unary.get(v, {})
        # Encode a soft confidence: 1.0 if no penalties, else lower
        confidence = 1.0
        # Include entry_road and maneuver from unary constraints for debugging and validation
        out.append({
            "vehicle": v, 
            "path_name": str(c.get("name", "")), 
            "confidence": confidence,
            "entry_road": u.get("entry_road", "unknown"),
            "exit_road": u.get("exit_road", "unknown"),
            "maneuver": u.get("maneuver", "unknown"),
        })
    return out


__all__ = [
    "_consistent_with_constraints",
    "_solve_paths_csp",
]
