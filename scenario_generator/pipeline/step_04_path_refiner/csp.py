import math
import time
from typing import Any, Dict, List, Optional, Tuple

from .geometry import (
    CropBox,
    _cross2,
    _dist,
    _dot,
    _first_last_idx_in_crop,
    _forward_axis_at,
    _polyline_cumdist,
    _polyline_slice,
    _segments_to_polyline_with_map,
    find_conflict_between_polylines,
)


def _speed_class_to_mps(speed_class: str) -> float:
    return {"slow": 5.0, "normal": 8.0, "fast": 12.0}.get(speed_class, 8.0)


def _candidate_speeds(base: float) -> List[float]:
    # Small discrete set around base
    opts = sorted(set([base, base - 2.0, base - 1.0, base + 1.0, base + 2.0]))
    out = [x for x in opts if 2.0 <= x <= 20.0]
    return out


def _default_start_end_in_crop(
    segments_detailed: List[Dict[str, Any]],
    crop: CropBox,
    start_crop: Optional[CropBox] = None,
) -> Optional[Tuple[int, int, List[Tuple[float, float]]]]:
    pts, _ = _segments_to_polyline_with_map(segments_detailed)
    if not pts:
        return None
    scrop = start_crop or crop
    se = _first_last_idx_in_crop(pts, scrop)
    if se is None:
        return None
    s, e = se
    cum = _polyline_cumdist(pts)
    inside_len = cum[e] - cum[s]
    total_len = cum[-1] if cum else 0.0
    crop_w = max(0.0, crop.xmax - crop.xmin)
    crop_h = max(0.0, crop.ymax - crop.ymin)
    crop_diag = math.hypot(crop_w, crop_h)
    min_inside = max(40.0, 0.6 * crop_diag)
    # If the crop only captures a tiny slice of a long corridor, keep the full tail
    # so routes are not truncated to a few dozen meters.
    if inside_len < min_inside and total_len > 2.0 * crop_diag:
        e = len(pts) - 1
    return s, e, pts


def _eval_spawn_relation(
    rel: Dict[str, Any],
    spawn_xy: Dict[str, Tuple[float, float]],
    forward_axis: Dict[str, Tuple[float, float]],
) -> float:
    """
    Penalty for violating a spawn relation.
    """
    a = rel["a"]
    b = rel["b"]
    typ = rel["type"]
    dist_m = float(rel.get("distance_m", 10.0))
    tol = float(rel.get("tolerance_m", 6.0))
    allow_ol = bool(rel.get("allow_other_lane", True))

    pa = spawn_xy[a]
    pb = spawn_xy[b]
    f = forward_axis[b]
    delta = (pa[0] - pb[0], pa[1] - pb[1])
    along = _dot(delta, f)
    lateral = abs(_cross2(delta, f))

    # If other lane not allowed, penalize lateral strongly beyond ~3.5m
    if not allow_ol and lateral > 3.5:
        return 1000.0 + (lateral - 3.5) ** 2

    desired = dist_m if typ == "ahead_of" else -dist_m
    lo = desired - tol
    hi = desired + tol
    if along < lo:
        return (lo - along) ** 2
    if along > hi:
        return (along - hi) ** 2
    return 0.0


def refine_spawn_and_speeds_soft_csp(
    per_vehicle: Dict[str, Dict[str, Any]],
    crop: CropBox,
    constraints: Dict[str, Any],
    conflict_dist_thresh_m: float = 3.0,
    min_spawn_sep_m: Optional[float] = None,
) -> Dict[str, Any]:
    """
    per_vehicle[veh] must include:
      - "segments_detailed": list
    Returns solution dict with chosen start/end indices and speeds.
    """
    vehicles = list(per_vehicle.keys())

    start_outside_margin_m = float((constraints.get("options", {}) or {}).get("start_outside_crop_margin_m", 0.0))
    start_crop = CropBox(
        xmin=crop.xmin - start_outside_margin_m,
        xmax=crop.xmax + start_outside_margin_m,
        ymin=crop.ymin - start_outside_margin_m,
        ymax=crop.ymax + start_outside_margin_m,
    )

    # Build base polylines and default start/end indices (in crop)
    base = {}
    for v in vehicles:
        segs = per_vehicle[v]["segments_detailed"]
        d = _default_start_end_in_crop(segs, crop, start_crop=start_crop)
        if d is None:
            # fallback: use entire polyline
            pts, _ = _segments_to_polyline_with_map(segs)
            if len(pts) < 2:
                raise SystemExit(f"[ERROR] Vehicle {v} has no polyline points.")
            base[v] = {"pts": pts, "start": 0, "end": len(pts) - 1}
        else:
            s, e, pts = d
            base[v] = {"pts": pts, "start": s, "end": e}

    # Speeds
    base_speed = {v: 8.0 for v in vehicles}
    for sp in constraints.get("vehicle_speeds", []):
        v = sp.get("vehicle")
        sc = sp.get("speed_class")
        if v in base_speed and sc in ("slow", "normal", "fast"):
            base_speed[v] = _speed_class_to_mps(sc)

    # Conflicts: compute for every pair
    conflicts = []
    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):
            va, vb = vehicles[i], vehicles[j]
            ca = _polyline_slice(base[va]["pts"], base[va]["start"], base[va]["end"])
            cb = _polyline_slice(base[vb]["pts"], base[vb]["start"], base[vb]["end"])
            conf = find_conflict_between_polylines(ca, cb, dist_thresh_m=conflict_dist_thresh_m)
            if conf is not None:
                conflicts.append({"a": va, "b": vb, "conf": conf})

    # Candidate start indices (bounded around default)
    candidates_start = {}
    n_vehicles = len(vehicles)
    # Expand the start window when many vehicles are present to improve feasibility
    start_window_m = 25.0
    if n_vehicles >= 5:
        start_window_m = 60.0
    elif n_vehicles >= 4:
        start_window_m = 45.0
    for v in vehicles:
        pts = base[v]["pts"]
        s0 = base[v]["start"]
        e0 = base[v]["end"]
        # allowable starts: inside crop, and keep at least 2 points before end
        inside = [i for i, (x, y) in enumerate(pts) if start_crop.contains(x, y)]
        if not inside:
            inside = list(range(len(pts)))
        # bound by window: within +/- start_window_m of default start in arc-length
        cum = _polyline_cumdist(pts)
        good = []
        for i in inside:
            if i >= e0:
                continue
            if abs(cum[i] - cum[s0]) <= start_window_m:
                good.append(i)
        if not good:
            good = [s0]
        # prune to <=12 candidates (spread)
        good = sorted(set(good))
        if len(good) > 12:
            step = max(1, len(good) // 12)
            good = good[::step][:12]
        candidates_start[v] = good

    # Candidate end indices (best effort).
    # We prefer to keep ends inside crop, but allow a small margin beyond the crop to avoid
    # truncating intended maneuvers right at the boundary (softly penalized in the objective).
    end_outside_margin_m = float(constraints.get("options", {}).get("end_outside_crop_margin_m", 0.0))
    candidates_end: Dict[str, List[int]] = {}

    def _point_outside_cost(x: float, y: float) -> float:
        # 0 inside crop; positive squared distance to the nearest crop edge otherwise.
        dx = 0.0
        if x < crop.xmin:
            dx = crop.xmin - x
        elif x > crop.xmax:
            dx = x - crop.xmax
        dy = 0.0
        if y < crop.ymin:
            dy = crop.ymin - y
        elif y > crop.ymax:
            dy = y - crop.ymax
        return dx * dx + dy * dy

    crop_end = CropBox(
        xmin=crop.xmin - end_outside_margin_m,
        xmax=crop.xmax + end_outside_margin_m,
        ymin=crop.ymin - end_outside_margin_m,
        ymax=crop.ymax + end_outside_margin_m,
    )

    for v in vehicles:
        pts = base[v]["pts"]
        s0 = base[v]["start"]
        e0 = base[v]["end"]
        cum = _polyline_cumdist(pts)

        inside_end = [i for i, (x, y) in enumerate(pts) if crop_end.contains(x, y)]
        if not inside_end:
            inside_end = list(range(len(pts)))

        window_m = 35.0
        good = []
        for i in inside_end:
            if i <= s0 + 1:
                continue
            if abs(cum[i] - cum[e0]) <= window_m and i >= e0:
                good.append(i)

        if not good:
            good = [e0]

        good = sorted(set(good))
        if len(good) > 8:
            step = max(1, len(good) // 8)
            good = good[::step][:8]
        candidates_end[v] = good

    speed_opts = {v: _candidate_speeds(base_speed[v]) for v in vehicles}

    # Adaptively reduce candidate counts if there are many vehicles to avoid combinatorial explosion
    # Target: keep total search space under ~500k
    if n_vehicles >= 4:
        # Reduce candidates for larger vehicle counts
        max_starts = 8 if n_vehicles >= 5 else 6
        max_ends = 3 if n_vehicles >= 5 else 4
        max_speeds = 2 if n_vehicles >= 5 else 3
        print(f"[DEBUG] refiner CSP: Reducing candidates due to {n_vehicles} vehicles (max_starts={max_starts}, max_ends={max_ends}, max_speeds={max_speeds})", flush=True)
        
        for v in vehicles:
            if len(candidates_start[v]) > max_starts:
                step = max(1, len(candidates_start[v]) // max_starts)
                candidates_start[v] = candidates_start[v][::step][:max_starts]
            if len(candidates_end[v]) > max_ends:
                step = max(1, len(candidates_end[v]) // max_ends)
                candidates_end[v] = candidates_end[v][::step][:max_ends]
            if len(speed_opts[v]) > max_speeds:
                # Keep base speed and closest alternatives
                base_spd = base_speed[v]
                sorted_by_dist = sorted(speed_opts[v], key=lambda s: abs(s - base_spd))
                speed_opts[v] = sorted_by_dist[:max_speeds]

    # Objective weights
    W_SYNC = 10.0
    W_SPAWN_REL = 4.0
    W_START_SHIFT = 0.2
    W_SPEED_SHIFT = 0.2
    MIN_SPAWN_SEP_M = 8.0  # Increased from 4.0 to account for vehicle length + CARLA safety margin
    if min_spawn_sep_m is not None and float(min_spawn_sep_m) > 0.0:
        MIN_SPAWN_SEP_M = float(min_spawn_sep_m)
    min_spawn_sep_m_used = float(MIN_SPAWN_SEP_M)

    spawn_rels = constraints.get("spawn_relations", []) if isinstance(constraints.get("spawn_relations", []), list) else []
    sync = bool(((constraints.get("options") or {}).get("synchronize_conflicts", True)))

    # Precompute distances to conflict points for each start choice (approx)
    # We'll compute t_to_conf = (s_conf - s_start)/speed, but s_conf from sliced polyline.
    # For consistency, we define conflicts on the default-sliced polylines and approximate with indices in those slices.
    # This is good enough for the heuristic.
    conflict_info = []
    if sync and conflicts:
        for c in conflicts:
            a, b = c["a"], c["b"]
            # conflict s along the sliced polylines (not full)
            sa = float(c["conf"]["s_along"]["p1_m"])
            sb = float(c["conf"]["s_along"]["p2_m"])
            conflict_info.append((a, b, sa, sb, c["conf"]["point"]))

    def score(assign_start: Dict[str, int], assign_speed: Dict[str, float], assign_end: Dict[str, int]) -> Tuple[float, Dict[str, Any]]:
        # spawn xy + forward axes
        spawn_xy = {}
        fwd = {}
        for v in vehicles:
            pts = base[v]["pts"]
            si = assign_start[v]
            spawn_xy[v] = pts[si]
            fwd[v] = _forward_axis_at(pts, si)

        total = 0.0

        # spawn relations
        for rel in spawn_rels:
            if rel.get("a") in spawn_xy and rel.get("b") in spawn_xy:
                total += W_SPAWN_REL * _eval_spawn_relation(rel, spawn_xy, fwd)

        # discourage overlapping spawns (hard block now: any pair closer than MIN_SPAWN_SEP_M is infeasible)
        spawn_sep_debug = []
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                va, vb = vehicles[i], vehicles[j]
                pa, pb = spawn_xy[va], spawn_xy[vb]
                d = _dist(pa, pb)
                if d < MIN_SPAWN_SEP_M:
                    spawn_sep_debug.append({"a": va, "b": vb, "dist_m": d})
                    total = float("inf")

        # conflict sync
        conflict_points = []
        if sync and conflict_info:
            for (a, b, sa, sb, pconf) in conflict_info:
                pa = base[a]["pts"]
                pb = base[b]["pts"]
                ca = _polyline_cumdist(pa)
                cb = _polyline_cumdist(pb)
                # translate sconf by shift from default start in full cumdist
                shift_a = ca[assign_start[a]] - ca[base[a]["start"]]
                shift_b = cb[assign_start[b]] - cb[base[b]["start"]]
                da = max(0.0, sa - shift_a)
                db = max(0.0, sb - shift_b)
                ta = da / max(0.5, assign_speed[a])
                tb = db / max(0.5, assign_speed[b])
                total += W_SYNC * (ta - tb) ** 2
                conflict_points.append({"a": a, "b": b, "point": pconf, "t": {a: ta, b: tb}})

        # prefer small shifts
        for v in vehicles:
            pts = base[v]["pts"]
            cum = _polyline_cumdist(pts)
            ds = abs(cum[assign_start[v]] - cum[base[v]["start"]])
            total += W_START_SHIFT * ds
            total += W_SPEED_SHIFT * (assign_speed[v] - base_speed[v]) ** 2

        # Prefer keeping each refined path close to the original picked length.
        # This avoids large truncations that can silently alter scenario semantics.
        W_LENGTH_SHRINK = 200.0 if n_vehicles >= 4 else 80.0
        MIN_KEEP_RATIO = 0.75 if n_vehicles >= 4 else 0.65
        length_shrink_debug = []
        for v in vehicles:
            pts = base[v]["pts"]
            cum = _polyline_cumdist(pts)
            base_len = max(1e-3, float(cum[base[v]["end"]] - cum[base[v]["start"]]))
            refined_len = max(0.0, float(cum[assign_end[v]] - cum[assign_start[v]]))
            keep_ratio = refined_len / base_len
            if keep_ratio < MIN_KEEP_RATIO:
                total += W_LENGTH_SHRINK * ((MIN_KEEP_RATIO - keep_ratio) ** 2) * base_len
                length_shrink_debug.append(
                    {
                        "vehicle": v,
                        "keep_ratio": keep_ratio,
                        "base_len_m": base_len,
                        "refined_len_m": refined_len,
                    }
                )

        dbg = {
            "spawn_xy": {v: {"x": spawn_xy[v][0], "y": spawn_xy[v][1]} for v in vehicles},
            "conflicts": conflict_points,
        }
        if length_shrink_debug:
            dbg["length_shrink"] = length_shrink_debug
        if spawn_sep_debug:
            dbg["spawn_separation"] = spawn_sep_debug
        # Softly penalize choosing an end point outside the strict crop.
        W_END_OUTSIDE = 500.0
        W_END_SHIFT = 0.2  # keep end near default unless needed
        end_outside_debug = []
        for v in vehicles:
            ei = assign_end[v]
            x, y = base[v]["pts"][ei]
            c = _point_outside_cost(x, y)
            if c > 1e-9:
                total += W_END_OUTSIDE * c
                end_outside_debug.append({"vehicle": v, "end_idx": int(ei), "outside_cost": float(c)})
        if end_outside_debug:
            dbg["end_outside"] = end_outside_debug
        # Penalize moving end away from the default end to avoid degenerate early termination.
        for v in vehicles:
            ei = int(assign_end[v])
            base_e = int(base[v]["end"])
            if ei != base_e:
                total += W_END_SHIFT * float(abs(ei - base_e))
        return total, dbg

    # Brute-force search (vehicles small)
    # IMPORTANT: Cap iterations to prevent combinatorial explosion with many vehicles
    MAX_ITERATIONS = 500000  # Should complete in <10s on typical hardware
    iterations_count = 0
    
    best = None
    best_dbg = None
    best_assign_start: Optional[Dict[str, int]] = None
    best_assign_speed: Optional[Dict[str, float]] = None
    best_assign_end: Optional[Dict[str, int]] = None

    # recursive enumeration
    vehs = vehicles
    
    # Log search space size
    search_space = 1
    for v in vehs:
        n_starts = len(candidates_start[v])
        n_ends = len(candidates_end[v])
        n_speeds = len(speed_opts[v])
        combos = n_starts * n_ends * n_speeds
        search_space *= combos
        print(f"[DEBUG] refiner CSP: vehicle {v} has {n_starts} starts × {n_ends} ends × {n_speeds} speeds = {combos} combos", flush=True)
    print(f"[DEBUG] refiner CSP: total search space = {search_space:,} (capped at {MAX_ITERATIONS:,})", flush=True)

    def rec(i: int, cur_start: Dict[str, int], cur_speed: Dict[str, float], cur_end: Dict[str, int]) -> bool:
        nonlocal best, best_dbg, best_assign_start, best_assign_speed, best_assign_end, iterations_count
        if iterations_count >= MAX_ITERATIONS:
            return True  # Signal early stop
        if i == len(vehs):
            iterations_count += 1
            sc, dbg = score(cur_start, cur_speed, cur_end)
            if math.isinf(sc):
                return False
            if best is None or sc < best:
                best = sc
                best_dbg = dbg
                best_assign_start = dict(cur_start)
                best_assign_speed = dict(cur_speed)
                best_assign_end = dict(cur_end)
            return False
        v = vehs[i]
        for si in candidates_start[v]:
            cur_start[v] = si
            for ei in candidates_end[v]:
                if ei <= si + 1:
                    continue
                cur_end[v] = ei
                for spd in speed_opts[v]:
                    cur_speed[v] = spd
                    if rec(i + 1, cur_start, cur_speed, cur_end):
                        return True  # Early stop propagation
        return False

    t0_rec = time.time()
    rec(0, {}, {}, {})
    print(f"[DEBUG] refiner CSP: explored {iterations_count:,} leaf nodes in {time.time() - t0_rec:.2f}s", flush=True)

    # Build solution
    sol = {
        "score": float(best if best is not None else 0.0),
        "start_idx": {v: int((best_assign_start or {}).get(v, base[v]["start"])) for v in vehicles},
        "end_idx": {v: int((best_assign_end or {}).get(v, base[v]["end"])) for v in vehicles},
        "speed_mps": {v: float((best_assign_speed or {}).get(v, base_speed[v])) for v in vehicles},
        "debug": dict(best_dbg or {}, min_spawn_sep_m=min_spawn_sep_m_used),
        "conflict_count": int(len(conflicts)),
    }
    return sol


__all__ = [
    "_candidate_speeds",
    "_default_start_end_in_crop",
    "_eval_spawn_relation",
    "_speed_class_to_mps",
    "refine_spawn_and_speeds_soft_csp",
]
