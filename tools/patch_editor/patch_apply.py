"""
patch_apply.py  —  Apply a declarative patch to a pipeline dataset dict.

This is the pure-Python bridge between the editor's patch.json files and
the pipeline's dataset["tracks"] structure.  Call it AFTER
process_single_scenario() and BEFORE _export_carla_routes_for_dataset().

Usage in runtime_orchestration.py:

    if args.patch:
        from tools.patch_editor.patch_apply import apply_patch_to_dataset
        patch_data = json.loads(Path(args.patch).read_text())
        n_applied = apply_patch_to_dataset(dataset, patch_data, verbose=args.verbose)

apply_patch_to_dataset returns the number of overrides that were applied.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Geometry helpers (no external deps)
# ---------------------------------------------------------------------------

def _project_point_to_polyline(
    pts: List[Tuple[float, float]],
    px: float,
    py: float,
) -> Tuple[float, float, float]:
    """
    Return (proj_x, proj_y, heading_deg) of the nearest point on a polyline.
    heading_deg is the tangent direction at the projection, in degrees from +X.
    """
    best_dist2 = float("inf")
    best_x, best_y, best_heading = px, py, 0.0

    for i in range(len(pts) - 1):
        ax, ay = float(pts[i][0]),   float(pts[i][1])
        bx, by = float(pts[i+1][0]), float(pts[i+1][1])
        abx, aby = bx - ax, by - ay
        ab2 = abx * abx + aby * aby
        if ab2 < 1e-12:
            t = 0.0
        else:
            t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab2))
        qx = ax + t * abx
        qy = ay + t * aby
        d2 = (px - qx) ** 2 + (py - qy) ** 2
        if d2 < best_dist2:
            best_dist2 = d2
            best_x, best_y = qx, qy
            best_heading = math.degrees(math.atan2(aby, abx))

    return best_x, best_y, best_heading


def _smoothstep01(u: float) -> float:
    """C1-continuous blend ramp on [0,1]."""
    u = max(0.0, min(1.0, u))
    return u * u * (3.0 - 2.0 * u)


def _lerp_angle_deg(a_deg: float, b_deg: float, w: float) -> float:
    """Shortest-arc interpolation between headings in degrees."""
    da = ((b_deg - a_deg + 180.0) % 360.0) - 180.0
    return a_deg + da * w


def _build_line_lookup(
    dataset: Dict[str, Any],
) -> Dict[int, Dict[str, Any]]:
    """Build {line_idx → line_record} from dataset["carla_map"]["lines"]."""
    carla_map = dataset.get("carla_map") or {}
    lines = carla_map.get("lines") or []
    return {int(rec["index"]): rec for rec in lines if isinstance(rec, dict) and "index" in rec}


# ---------------------------------------------------------------------------
# Lane chain helpers — find continuation lanes at endpoints
# ---------------------------------------------------------------------------

def _polyline_heading(pts: List[Tuple[float, float]], at_end: bool) -> float:
    """Return heading (radians) at the start or end of a polyline."""
    if len(pts) < 2:
        return 0.0
    if at_end:
        return math.atan2(pts[-1][1] - pts[-2][1], pts[-1][0] - pts[-2][0])
    return math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0])


def _angle_diff(a: float, b: float) -> float:
    """Absolute angular difference in radians, wrapped to [0, pi]."""
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return abs(d)


def _build_lane_chain(
    start_idx: int,
    line_lookup: Dict[int, Dict[str, Any]],
    dist_thr: float = 5.0,
    angle_thr_deg: float = 36.0,
) -> List[int]:
    """
    Build an ordered chain of lane indices that form a continuous path
    through *start_idx*.  Walks backward (predecessors) and forward
    (successors) by matching polyline endpoints within *dist_thr* metres
    and *angle_thr_deg* heading tolerance.
    """
    angle_thr = math.radians(angle_thr_deg)

    def _get_pts(idx: int) -> List[Tuple[float, float]]:
        rec = line_lookup.get(idx)
        if rec is None:
            return []
        poly = rec.get("polyline") or []
        if len(poly) < 2:
            return []
        return [(float(p[0]), float(p[1])) for p in poly]

    def _find_successor(cur_idx: int, visited: set) -> Optional[int]:
        cur_pts = _get_pts(cur_idx)
        if not cur_pts:
            return None
        end_pt = cur_pts[-1]
        end_h = _polyline_heading(cur_pts, True)
        best_idx, best_d = None, float("inf")
        for idx, rec in line_lookup.items():
            if idx == cur_idx or idx in visited:
                continue
            cand_pts = _get_pts(idx)
            if not cand_pts:
                continue
            d = math.hypot(end_pt[0] - cand_pts[0][0], end_pt[1] - cand_pts[0][1])
            if d > dist_thr:
                continue
            start_h = _polyline_heading(cand_pts, False)
            if _angle_diff(end_h, start_h) > angle_thr:
                continue
            if d < best_d:
                best_d = d
                best_idx = idx
        return best_idx

    def _find_predecessor(cur_idx: int, visited: set) -> Optional[int]:
        cur_pts = _get_pts(cur_idx)
        if not cur_pts:
            return None
        start_pt = cur_pts[0]
        start_h = _polyline_heading(cur_pts, False)
        best_idx, best_d = None, float("inf")
        for idx, rec in line_lookup.items():
            if idx == cur_idx or idx in visited:
                continue
            cand_pts = _get_pts(idx)
            if not cand_pts:
                continue
            d = math.hypot(start_pt[0] - cand_pts[-1][0], start_pt[1] - cand_pts[-1][1])
            if d > dist_thr:
                continue
            end_h = _polyline_heading(cand_pts, True)
            if _angle_diff(start_h, end_h) > angle_thr:
                continue
            if d < best_d:
                best_d = d
                best_idx = idx
        return best_idx

    visited = {start_idx}

    # Walk backward
    preds: List[int] = []
    cur = start_idx
    for _ in range(50):
        p = _find_predecessor(cur, visited)
        if p is None:
            break
        preds.insert(0, p)
        visited.add(p)
        cur = p

    # Walk forward
    succs: List[int] = []
    cur = start_idx
    for _ in range(50):
        s = _find_successor(cur, visited)
        if s is None:
            break
        succs.append(s)
        visited.add(s)
        cur = s

    return preds + [start_idx] + succs


def _chain_polyline(
    chain: List[int],
    line_lookup: Dict[int, Dict[str, Any]],
) -> List[Tuple[float, float]]:
    """Concatenate polylines from a lane chain, deduplicating overlapping endpoints."""
    all_pts: List[Tuple[float, float]] = []
    for idx in chain:
        rec = line_lookup.get(idx)
        if rec is None:
            continue
        poly = rec.get("polyline") or []
        if len(poly) < 2:
            continue
        pts = [(float(p[0]), float(p[1])) for p in poly]
        # Skip first point if it overlaps with previous segment end
        start_i = 0
        if all_pts and math.hypot(all_pts[-1][0] - pts[0][0],
                                   all_pts[-1][1] - pts[0][1]) < 1.0:
            start_i = 1
        all_pts.extend(pts[start_i:])
    return all_pts


def _snap_frames_to_line(
    frames: List[Dict[str, Any]],
    line_rec: Dict[str, Any],
    start_t: Optional[float],
    end_t: Optional[float],
    chained_pts: Optional[List[Tuple[float, float]]] = None,
    blend_in_s: float = 0.0,
) -> int:
    """
    Project each frame (whose t is within [start_t, end_t]) onto the target
    CARLA polyline (or a chained polyline if provided). If blend_in_s>0 and
    start_t is provided, blend from original trajectory into snapped trajectory
    over [start_t, start_t+blend_in_s] for continuity.
    Updates cx, cy, cyaw, ccli, csource in-place.
    Returns count of modified frames.
    """
    if chained_pts is not None and len(chained_pts) >= 2:
        pts = chained_pts
    else:
        polyline = line_rec.get("polyline") or []
        if len(polyline) < 2:
            return 0
        pts = [(float(p[0]), float(p[1])) for p in polyline]

    line_idx = int(line_rec["index"])

    count = 0
    for fr in frames:
        t = float(fr.get("t", 0.0))
        if start_t is not None and t < start_t:
            continue
        if end_t is not None and t > end_t:
            continue

        # Use current CARLA-snapped position as the query point if available
        qx = float(fr.get("cx", fr.get("sx", fr.get("x", 0.0))))
        qy = float(fr.get("cy", fr.get("sy", fr.get("y", 0.0))))

        proj_x, proj_y, proj_heading = _project_point_to_polyline(pts, qx, qy)

        blend_w = 1.0
        if start_t is not None and blend_in_s > 1e-6:
            blend_w = _smoothstep01((t - start_t) / blend_in_s)
            if blend_w <= 1e-9:
                # Keep exact breakpoint frame untouched; snap starts after.
                continue

        if blend_w >= 0.999999:
            out_x, out_y, out_heading = proj_x, proj_y, proj_heading
            csource = "patch_lane_snap"
        else:
            base_heading = float(fr.get("cyaw", fr.get("yaw", proj_heading)))
            out_x = qx + (proj_x - qx) * blend_w
            out_y = qy + (proj_y - qy) * blend_w
            out_heading = _lerp_angle_deg(base_heading, proj_heading, blend_w)
            csource = "patch_lane_snap_blend"

        fr["cx"]     = out_x
        fr["cy"]     = out_y
        fr["cyaw"]   = out_heading
        fr["ccli"]   = line_idx
        fr["csource"]= csource
        count += 1

    return count


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def apply_patch_to_dataset(
    dataset: Dict[str, Any],
    patch: Dict[str, Any],
    verbose: bool = False,
) -> int:
    """
    Apply a declarative patch dict to the dataset in-place.

    Processes overrides in this order:
      1. delete          — remove track from dataset["tracks"]
      2. lane_segment_overrides — remap cx/cy/ccli for frame time ranges
      3. snap_to_outermost      — lateral projection to rightmost parallel lane
      4. waypoint_overrides     — delta-apply dx/dy to cx/cy
      5. phase_override         — tag track with phase metadata

    Returns total number of overrides applied.
    """
    overrides = patch.get("overrides") or []
    if not overrides:
        return 0

    tracks: List[Dict[str, Any]] = dataset.get("tracks") or []
    line_lookup = _build_line_lookup(dataset)

    # Build track lookup by id (both str and int keys)
    track_by_id: Dict[str, Dict[str, Any]] = {}
    for tr in tracks:
        tid = str(tr.get("id", ""))
        track_by_id[tid] = tr

    delete_ids: List[str] = []
    n_applied = 0

    for ov in overrides:
        if not isinstance(ov, dict):
            continue
        actor_id = str(ov.get("actor_id", ""))
        track = track_by_id.get(actor_id)

        # ── delete ──────────────────────────────────────────────────────────
        if ov.get("delete"):
            if track is not None:
                delete_ids.append(actor_id)
                n_applied += 1
                if verbose:
                    print(f"[PATCH] delete {actor_id}")
            continue  # skip other overrides for deleted actor

        if track is None:
            if verbose:
                print(f"[PATCH] WARN actor {actor_id!r} not found in dataset")
            continue

        frames: List[Dict[str, Any]] = track.get("frames") or []

        # ── lane_segment_overrides ───────────────────────────────────────────
        for lso in (ov.get("lane_segment_overrides") or []):
            if not isinstance(lso, dict):
                continue
            lane_id_str = str(lso.get("lane_id", ""))
            try:
                line_idx = int(lane_id_str)
            except ValueError:
                if verbose:
                    print(f"[PATCH] WARN invalid lane_id {lane_id_str!r} for actor {actor_id}")
                continue
            line_rec = line_lookup.get(line_idx)
            if line_rec is None:
                if verbose:
                    print(f"[PATCH] WARN CARLA line {line_idx} not found in dataset for actor {actor_id}")
                continue
            start_t = float(lso["start_t"]) if lso.get("start_t") is not None else None
            end_t   = float(lso["end_t"])   if lso.get("end_t")   is not None else None
            blend_in_s = 0.0
            if lso.get("blend_in_s") is not None:
                try:
                    blend_in_s = max(0.0, float(lso["blend_in_s"]))
                except (TypeError, ValueError):
                    blend_in_s = 0.0
            # Build lane chain for continuation
            chain = _build_lane_chain(line_idx, line_lookup)
            chained_pts = _chain_polyline(chain, line_lookup) if len(chain) > 1 else None
            n_frames = _snap_frames_to_line(frames, line_rec, start_t, end_t,
                                            chained_pts=chained_pts,
                                            blend_in_s=blend_in_s)
            n_applied += 1
            if verbose:
                chain_str = f" (chain: {len(chain)} segments)" if len(chain) > 1 else ""
                blend_str = f", blend={blend_in_s:.3f}s" if blend_in_s > 0 else ""
                print(f"[PATCH] lane_snap {actor_id} → line {line_idx}{chain_str} "
                      f"t=[{start_t},{end_t}]{blend_str}  {n_frames} frames updated")

        # ── snap_to_outermost ────────────────────────────────────────────────
        if ov.get("snap_to_outermost"):
            # Find modal ccli across all frames
            ccli_counts: Dict[int, int] = {}
            for fr in frames:
                cli = int(fr.get("ccli", -1))
                if cli >= 0:
                    ccli_counts[cli] = ccli_counts.get(cli, 0) + 1
            if ccli_counts:
                modal_cli = max(ccli_counts, key=ccli_counts.__getitem__)
                outer_line = _find_outermost_parallel(modal_cli, line_lookup)
                if outer_line is not None:
                    n_frames = _snap_frames_to_line(frames, outer_line, None, None)
                    n_applied += 1
                    if verbose:
                        print(f"[PATCH] snap_to_outermost {actor_id}: "
                              f"line {modal_cli} → {outer_line['index']}  "
                              f"{n_frames} frames")
                elif verbose:
                    print(f"[PATCH] snap_to_outermost {actor_id}: "
                          f"no rightmost parallel found for line {modal_cli}")
            elif verbose:
                print(f"[PATCH] snap_to_outermost {actor_id}: no valid ccli found")

        # ── waypoint_overrides ───────────────────────────────────────────────
        for wo in (ov.get("waypoint_overrides") or []):
            if not isinstance(wo, dict):
                continue
            fi = int(wo.get("frame_idx", -1))
            if fi < 0 or fi >= len(frames):
                if verbose:
                    print(f"[PATCH] WARN waypoint frame_idx {fi} out of range for {actor_id}")
                continue
            fr = frames[fi]
            dx = float(wo.get("dx", 0.0))
            dy = float(wo.get("dy", 0.0))
            fr["cx"] = float(fr.get("cx", fr.get("sx", fr.get("x", 0.0)))) + dx
            fr["cy"] = float(fr.get("cy", fr.get("sy", fr.get("y", 0.0)))) + dy
            fr["csource"] = "patch_waypoint"
            n_applied += 1
            if verbose:
                print(f"[PATCH] waypoint {actor_id}[{fi}] +({dx:.3f},{dy:.3f})")

        # ── phase_override ───────────────────────────────────────────────────
        if ov.get("phase_override"):
            phase = str(ov["phase_override"])
            track["patch_phase_override"] = phase
            # Also tag all frames for downstream consumers
            for fr in frames:
                fr["patch_phase"] = phase
            n_applied += 1
            if verbose:
                print(f"[PATCH] phase_override {actor_id} → {phase}")

    # Apply deletes last (after all other overrides have been processed)
    if delete_ids:
        delete_set = set(delete_ids)
        dataset["tracks"] = [
            tr for tr in tracks if str(tr.get("id", "")) not in delete_set
        ]
        if verbose:
            print(f"[PATCH] deleted {len(delete_ids)} track(s): {sorted(delete_ids)}")

    return n_applied


# ---------------------------------------------------------------------------
# Outermost snap geometry
# ---------------------------------------------------------------------------

def _find_outermost_parallel(
    modal_cli: int,
    line_lookup: Dict[int, Dict[str, Any]],
    max_lateral_m: float = 15.0,
    heading_tol_deg: float = 25.0,
) -> Optional[Dict[str, Any]]:
    """Return the rightmost CARLA line parallel to modal_cli."""
    cur = line_lookup.get(modal_cli)
    if cur is None:
        return None

    cur_pts = [(float(p[0]), float(p[1])) for p in (cur.get("polyline") or [])]
    if len(cur_pts) < 2:
        return None

    # Current line heading and right-perpendicular vector
    dx = cur_pts[-1][0] - cur_pts[0][0]
    dy = cur_pts[-1][1] - cur_pts[0][1]
    mag = math.hypot(dx, dy)
    if mag < 1e-6:
        return None
    heading_rad = math.atan2(dy, dx)
    fwd = (dx / mag, dy / mag)
    rgt = (fwd[1], -fwd[0])   # right-perpendicular (using right-hand rule)

    cx_mid = (cur_pts[0][0] + cur_pts[-1][0]) / 2.0
    cy_mid = (cur_pts[0][1] + cur_pts[-1][1]) / 2.0

    best_line = None
    best_offset = 0.0   # current line at 0; positive = rightward

    for idx, rec in line_lookup.items():
        if idx == modal_cli:
            continue
        cand_pts = [(float(p[0]), float(p[1])) for p in (rec.get("polyline") or [])]
        if len(cand_pts) < 2:
            continue

        cdx = cand_pts[-1][0] - cand_pts[0][0]
        cdy = cand_pts[-1][1] - cand_pts[0][1]
        cang = math.degrees(math.atan2(cdy, cdx))
        hang = math.degrees(heading_rad)
        hdiff = abs((cang - hang + 180) % 360 - 180)
        if hdiff > heading_tol_deg:
            continue

        c_mid_x = (cand_pts[0][0] + cand_pts[-1][0]) / 2.0
        c_mid_y = (cand_pts[0][1] + cand_pts[-1][1]) / 2.0

        lx = c_mid_x - cx_mid
        ly = c_mid_y - cy_mid

        # Lateral offset (right = positive)
        lateral = lx * rgt[0] + ly * rgt[1]
        # Longitudinal offset must be small
        longit = abs(lx * fwd[0] + ly * fwd[1])

        if abs(lateral) > max_lateral_m or longit > max_lateral_m * 3:
            continue

        if lateral > best_offset:
            best_offset = lateral
            best_line = rec

    return best_line
