"""Trajectory ingest internals: lane/object matching and metadata mapping."""

from __future__ import annotations

from v2xpnp.pipeline import trajectory_ingest_stage_01_types_io as _ingest_s1
from v2xpnp.pipeline.trajectory_ingest_stage_01_types_io import *  # noqa: F401,F403


for _name, _value in vars(_ingest_s1).items():
    if _name.startswith("__"):
        continue
    globals().setdefault(_name, _value)



def _candidate_waypoints_from_continuous_lane(
    prev_wp,
    step_dist: float,
) -> List[object]:
    """
    Return forward lane-follow candidates from the previous snapped waypoint.
    Uses multiple nearby step distances and deduplicates candidates.
    """
    if prev_wp is None:
        return []
    candidates: List[object] = []
    seen = set()

    def _add_wp(wp) -> None:
        if wp is None:
            return
        try:
            key = (
                int(getattr(wp, "road_id", 0)),
                int(getattr(wp, "lane_id", 0)),
                int(round(float(getattr(wp, "s", 0.0)) * 100.0)),
            )
        except Exception:
            key = (0, 0, id(wp))
        if key in seen:
            return
        seen.add(key)
        candidates.append(wp)

    # If ego barely moved, keep the same centerline point to avoid jitter.
    if step_dist <= 0.15:
        _add_wp(prev_wp)

    base = max(0.5, min(4.0, float(step_dist) if step_dist > 1e-6 else 1.0))
    probe_steps = sorted(
        set(
            [
                base,
                max(0.5, base * 0.7),
                min(4.0, max(0.8, base * 1.35)),
            ]
        )
    )
    for step in probe_steps:
        try:
            nxt = list(prev_wp.next(float(step)))
        except Exception:
            nxt = []
        for wp in nxt:
            _add_wp(wp)

    if not candidates:
        _add_wp(prev_wp)
    return candidates


def _trajectory_direction_and_hint(
    traj: List[Waypoint],
    idx: int,
    *,
    back_steps: int = 2,
    forward_steps: int = 6,
) -> Tuple[float, float, float, Tuple[float, float]]:
    """
    Estimate a stable trajectory direction + lookahead hint around index idx.
    Uses a local-to-future window so lane-locking follows path shape over time,
    not just the immediate previous step.
    """
    if not traj:
        return 1.0, 0.0, 1.0, (0.0, 0.0)

    n = len(traj)
    i = max(0, min(n - 1, int(idx)))
    back_idx = max(0, i - max(0, int(back_steps)))
    fwd_idx = min(n - 1, i + max(1, int(forward_steps)))

    base = traj[i]
    hint = traj[fwd_idx]
    dx = float(hint.x) - float(traj[back_idx].x)
    dy = float(hint.y) - float(traj[back_idx].y)
    norm = math.hypot(dx, dy)
    if norm <= 1e-6:
        yaw_rad = math.radians(float(base.yaw))
        dir_x = math.cos(yaw_rad)
        dir_y = math.sin(yaw_rad)
    else:
        dir_x = dx / norm
        dir_y = dy / norm

    forward_dist = math.hypot(float(hint.x) - float(base.x), float(hint.y) - float(base.y))
    forward_dist = max(0.5, float(forward_dist))
    return float(dir_x), float(dir_y), float(forward_dist), (float(hint.x), float(hint.y))


def _select_lane_locked_waypoint(
    prev_wp,
    raw_wp: Waypoint,
    raw_step_dx: float,
    raw_step_dy: float,
    local_best_info: Optional[Dict[str, float]],
    desired_dir_x: Optional[float] = None,
    desired_dir_y: Optional[float] = None,
    future_hint_xy: Optional[Tuple[float, float]] = None,
    future_lookahead_dist: Optional[float] = None,
) -> object:
    """
    Follow the previous snapped lane centerline (via waypoint graph) and pick
    the branch that best matches current trajectory direction/position.
    """
    if prev_wp is None:
        return None
    step_dist = math.hypot(float(raw_step_dx), float(raw_step_dy))
    candidates = _candidate_waypoints_from_continuous_lane(prev_wp, step_dist)
    if not candidates:
        return None

    if (
        desired_dir_x is not None
        and desired_dir_y is not None
        and math.hypot(float(desired_dir_x), float(desired_dir_y)) > 1e-6
    ):
        dnorm = math.hypot(float(desired_dir_x), float(desired_dir_y))
        raw_dir_x = float(desired_dir_x) / dnorm
        raw_dir_y = float(desired_dir_y) / dnorm
    else:
        raw_dir_norm = math.hypot(float(raw_step_dx), float(raw_step_dy))
        if raw_dir_norm <= 1e-6:
            raw_dir_x = math.cos(math.radians(float(raw_wp.yaw)))
            raw_dir_y = math.sin(math.radians(float(raw_wp.yaw)))
        else:
            raw_dir_x = float(raw_step_dx) / raw_dir_norm
            raw_dir_y = float(raw_step_dy) / raw_dir_norm

    try:
        prev_loc = prev_wp.transform.location
        prev_x = float(prev_loc.x)
        prev_y = float(prev_loc.y)
    except Exception:
        prev_x = float(raw_wp.x)
        prev_y = float(raw_wp.y)

    best_wp = None
    best_score: Optional[float] = None
    for cand_wp in candidates:
        try:
            cloc = cand_wp.transform.location
            cyaw = float(cand_wp.transform.rotation.yaw)
            cx = float(cloc.x)
            cy = float(cloc.y)
        except Exception:
            continue
        dist_to_raw = math.hypot(cx - float(raw_wp.x), cy - float(raw_wp.y))
        yaw_diff = _yaw_diff_deg(cyaw, float(raw_wp.yaw))
        score = float(dist_to_raw) + 0.20 * (float(yaw_diff) / 90.0)

        # Keep forward progression aligned with trajectory segment direction.
        mvx = cx - prev_x
        mvy = cy - prev_y
        mv_norm = math.hypot(mvx, mvy)
        if mv_norm > 1e-6:
            cosang = (mvx * raw_dir_x + mvy * raw_dir_y) / mv_norm
            cosang = max(-1.0, min(1.0, float(cosang)))
            if cosang < 0.0:
                score += 2.5 + 2.0 * abs(cosang)
            else:
                score += 0.30 * (1.0 - cosang)

        # Keep branch close to the local nearest lane-center projection.
        if isinstance(local_best_info, dict):
            try:
                ax = float(local_best_info["x"])
                ay = float(local_best_info["y"])
                score += 0.70 * math.hypot(cx - ax, cy - ay)
            except Exception:
                pass

        # Lookahead branch scoring against future trajectory hint to choose
        # the correct lane-node branch at intersections/curves.
        if future_hint_xy is not None:
            hint_x, hint_y = future_hint_xy
            probe_dist = float(future_lookahead_dist) if future_lookahead_dist is not None else max(1.5, 2.0 * step_dist)
            probe_dist = max(1.0, min(10.0, probe_dist))
            branch_score: Optional[float] = None
            try:
                future_nodes = list(cand_wp.next(probe_dist))
            except Exception:
                future_nodes = []

            if future_nodes:
                for next_wp in future_nodes:
                    try:
                        nloc = next_wp.transform.location
                        nx = float(nloc.x)
                        ny = float(nloc.y)
                    except Exception:
                        continue
                    dist_hint = math.hypot(nx - float(hint_x), ny - float(hint_y))
                    fvx = nx - cx
                    fvy = ny - cy
                    fvn = math.hypot(fvx, fvy)
                    if fvn <= 1e-6:
                        dir_pen = 1.0
                    else:
                        cos_future = max(-1.0, min(1.0, float((fvx * raw_dir_x + fvy * raw_dir_y) / fvn)))
                        dir_pen = 1.0 - max(0.0, cos_future)
                    cand_branch_score = 0.40 * float(dist_hint) + 1.25 * float(dir_pen)
                    if branch_score is None or cand_branch_score < branch_score:
                        branch_score = cand_branch_score
            else:
                # Fallback: project candidate heading forward when graph expansion fails.
                proj_x = cx + probe_dist * math.cos(math.radians(float(cyaw)))
                proj_y = cy + probe_dist * math.sin(math.radians(float(cyaw)))
                branch_score = 0.30 * math.hypot(proj_x - float(hint_x), proj_y - float(hint_y))

            if branch_score is not None:
                score += float(branch_score)

        if best_score is None or score < best_score:
            best_score = score
            best_wp = cand_wp
    return best_wp


def _compute_piecewise_ego_offsets(
    traj: List[Waypoint],
    times: List[float],
    world_map,
    max_shift: float,
    intent_margin: float,
    smooth_window: int,
    max_step_delta: float,
    bridge_max_gap_steps: int,
    bridge_straight_thresh_deg: float,
    snap_ego_to_lane: bool = False,
) -> Tuple[List[Tuple[float, float]], Dict[str, object]]:
    """Compute per-waypoint XY offsets for ego alignment while preserving trajectory timing."""
    if world_map is None or not traj:
        return [], {"status": "no_map_or_traj"}

    lane_names_all = ["Driving", "Shoulder", "Parking"]
    sample_count = max(8, min(len(traj), 24))
    samples = _select_alignment_samples(traj, times, sample_count)
    sample_projections: List[Dict[str, Dict[str, float]]] = []
    for s in samples:
        loc = carla.Location(x=float(s["x"]), y=float(s["y"]), z=float(s["z"]))
        sample_projections.append(_project_to_lane_types(world_map, loc, lane_names_all))
    intent_info = _infer_alignment_intent("npc", sample_projections, intent_margin)
    candidate_lanes = list(intent_info.get("candidate_lanes") or lane_names_all)
    if not candidate_lanes:
        candidate_lanes = lane_names_all

    raw_dx: List[float] = []
    raw_dy: List[float] = []
    valid_mask: List[bool] = []
    lane_keys: List[Optional[Tuple[int, int]]] = []
    yaws: List[float] = []
    lane_switches_raw = 0
    prev_lane_key: Optional[Tuple[int, int]] = None
    lane_dists: List[float] = []
    yaw_diffs: List[float] = []
    lane_lock_used = bool(snap_ego_to_lane)
    lane_lock_fallbacks = 0
    lane_lock_kept_prev = 0
    prev_snap_wp = None

    for idx, wp in enumerate(traj):
        loc = carla.Location(x=float(wp.x), y=float(wp.y), z=float(wp.z))
        if idx % 50 == 0:
            print(f"[SPAWN_PRE][EGO] piecewise ego offset: point {idx}/{len(traj)}, loc=({wp.x:.2f},{wp.y:.2f},{wp.z:.2f})", flush=True)
        best, best_lane_key, best_lane_name, best_wp = _best_lane_projection_with_waypoint(
            world_map=world_map,
            loc=loc,
            raw_yaw=float(wp.yaw),
            candidate_lanes=candidate_lanes,
            prev_lane_key=prev_lane_key,
        )

        if lane_lock_used and prev_snap_wp is not None:
            prev_raw = traj[idx - 1] if idx > 0 else wp
            raw_step_dx = float(wp.x) - float(prev_raw.x)
            raw_step_dy = float(wp.y) - float(prev_raw.y)
            traj_dir_x, traj_dir_y, traj_forward_dist, traj_future_xy = _trajectory_direction_and_hint(
                traj,
                idx,
                back_steps=2,
                forward_steps=6,
            )
            lane_locked_wp = _select_lane_locked_waypoint(
                prev_wp=prev_snap_wp,
                raw_wp=wp,
                raw_step_dx=raw_step_dx,
                raw_step_dy=raw_step_dy,
                local_best_info=best,
                desired_dir_x=traj_dir_x,
                desired_dir_y=traj_dir_y,
                future_hint_xy=traj_future_xy,
                future_lookahead_dist=traj_forward_dist,
            )
            if lane_locked_wp is None:
                lane_lock_fallbacks += 1
            else:
                try:
                    lwloc = lane_locked_wp.transform.location
                    lwyaw = float(lane_locked_wp.transform.rotation.yaw)
                    lroad = int(getattr(lane_locked_wp, "road_id", 0))
                    llane = int(getattr(lane_locked_wp, "lane_id", 0))
                    best = {
                        "x": float(lwloc.x),
                        "y": float(lwloc.y),
                        "z": float(lwloc.z),
                        "yaw": float(lwyaw),
                        "dist": float(math.hypot(float(lwloc.x) - float(wp.x), float(lwloc.y) - float(wp.y))),
                        "road_id": float(lroad),
                        "lane_id": float(llane),
                    }
                    best_lane_key = (lroad, llane)
                    best_wp = lane_locked_wp
                    if math.hypot(raw_step_dx, raw_step_dy) <= 0.15:
                        lane_lock_kept_prev += 1
                except Exception:
                    lane_lock_fallbacks += 1

        if best is None:
            raw_dx.append(0.0)
            raw_dy.append(0.0)
            valid_mask.append(False)
            lane_keys.append(None)
            yaws.append(float(wp.yaw))
            continue

        dx = float(best["x"]) - float(wp.x)
        dy = float(best["y"]) - float(wp.y)
        mag = math.hypot(dx, dy)
        if mag > max_shift > 0.0:
            scale = max_shift / max(mag, 1e-6)
            dx *= scale
            dy *= scale
        raw_dx.append(dx)
        raw_dy.append(dy)
        valid_mask.append(True)
        lane_keys.append(best_lane_key)
        yaws.append(float(wp.yaw))
        lane_dists.append(float(best["dist"]))
        yaw_diffs.append(_yaw_diff_deg(float(best["yaw"]), float(wp.yaw)))
        if prev_lane_key is not None and best_lane_key is not None and best_lane_key != prev_lane_key:
            lane_switches_raw += 1
        if best_lane_name is not None or best_lane_key is not None:
            prev_lane_key = best_lane_key
        if best_wp is not None:
            prev_snap_wp = best_wp

    bridge_info = {"bridged_missing": 0, "bridged_transient": 0}
    if int(bridge_max_gap_steps) > 0:
        bridge_info = _bridge_missing_or_transient_offsets(
            dxs=raw_dx,
            dys=raw_dy,
            valid_mask=valid_mask,
            lane_keys=lane_keys,
            yaws=yaws,
            max_gap_steps=int(bridge_max_gap_steps),
            straight_thresh_deg=float(bridge_straight_thresh_deg),
        )

    # Backfill any remaining missing offsets from nearest valid point.
    if any(valid_mask):
        valid_idx = [i for i, ok in enumerate(valid_mask) if ok]
        for i, ok in enumerate(valid_mask):
            if ok:
                continue
            nearest = min(valid_idx, key=lambda j: abs(j - i))
            raw_dx[i] = raw_dx[nearest]
            raw_dy[i] = raw_dy[nearest]
    else:
        return (
            [(0.0, 0.0) for _ in traj],
            {
                "status": "no_projection",
                "intent": intent_info,
                "candidate_lanes": candidate_lanes,
                "no_wp_ratio": 1.0,
            },
        )

    # Smooth offsets to avoid jagged lane oscillation.
    smooth_dx = _moving_average(raw_dx, smooth_window)
    smooth_dy = _moving_average(raw_dy, smooth_window)
    if smooth_dx:
        smooth_dx[0] = raw_dx[0]
        smooth_dy[0] = raw_dy[0]
        smooth_dx[-1] = raw_dx[-1]
        smooth_dy[-1] = raw_dy[-1]

    # Limit frame-to-frame offset delta to avoid sudden teleports.
    max_step_delta = max(0.05, float(max_step_delta))
    for i in range(1, len(smooth_dx)):
        ddx = smooth_dx[i] - smooth_dx[i - 1]
        ddy = smooth_dy[i] - smooth_dy[i - 1]
        step = math.hypot(ddx, ddy)
        if step > max_step_delta:
            scale = max_step_delta / max(step, 1e-6)
            smooth_dx[i] = smooth_dx[i - 1] + ddx * scale
            smooth_dy[i] = smooth_dy[i - 1] + ddy * scale

    # Re-clamp final magnitude.
    offsets: List[Tuple[float, float]] = []
    for dx, dy in zip(smooth_dx, smooth_dy):
        mag = math.hypot(dx, dy)
        if mag > max_shift > 0.0:
            scale = max_shift / max(mag, 1e-6)
            dx *= scale
            dy *= scale
        offsets.append((float(dx), float(dy)))

    shift_mags = [math.hypot(dx, dy) for dx, dy in offsets]
    lane_keys_valid = [lk for lk in lane_keys if lk is not None]
    lane_switches_post = 0
    prev_key: Optional[Tuple[int, int]] = None
    for lk in lane_keys:
        if lk is None:
            continue
        if prev_key is not None and lk != prev_key:
            lane_switches_post += 1
        prev_key = lk
    report = {
        "status": "ok",
        "mode": "piecewise",
        "intent": intent_info,
        "candidate_lanes": candidate_lanes,
        "samples": len(samples),
        "no_wp_ratio": float(sum(1 for ok in valid_mask if not ok) / max(1, len(valid_mask))),
        "lane_switches": int(lane_switches_post),
        "lane_switches_raw": int(lane_switches_raw),
        "bridge": bridge_info,
        "lane_lock": {
            "enabled": lane_lock_used,
            "fallbacks": int(lane_lock_fallbacks),
            "kept_prev_steps": int(lane_lock_kept_prev),
        },
        "lane_key_coverage": float(len(lane_keys_valid) / max(1, len(lane_keys))),
        "lane_dist_median": float(_median(lane_dists)) if lane_dists else 999.0,
        "yaw_diff_median": float(_median(yaw_diffs)) if yaw_diffs else 180.0,
        "shift_median": float(_median(shift_mags)) if shift_mags else 0.0,
        "shift_max": float(max(shift_mags)) if shift_mags else 0.0,
        "first_offset": [float(offsets[0][0]), float(offsets[0][1])] if offsets else [0.0, 0.0],
        "last_offset": [float(offsets[-1][0]), float(offsets[-1][1])] if offsets else [0.0, 0.0],
    }
    return offsets, report


def _compute_piecewise_actor_offsets(
    traj: List[Waypoint],
    times: List[float],
    world_map,
    role: str,
    base_dx: float,
    base_dy: float,
    max_shift: float,
    intent_margin: float,
    local_limit: float,
    smooth_window: int,
    max_step_delta: float,
    bridge_max_gap_steps: int,
    bridge_straight_thresh_deg: float,
    early_lane_lock_seconds: float,
    early_lane_switch_override_margin: float,
) -> Tuple[List[Tuple[float, float]], Dict[str, object]]:
    """
    Compute per-waypoint XY offsets around a global base offset.
    The refinement is bounded so it cannot drift far from the globally-selected shift.
    """
    if world_map is None or not traj:
        return [], {"status": "no_map_or_traj"}

    if role == "npc":
        lane_names_all = ["Driving", "Shoulder", "Parking"]
    elif role == "static":
        lane_names_all = ["Parking", "Shoulder", "Driving"]
    else:
        lane_names_all = ["Sidewalk", "Shoulder", "Driving"]
    sample_count = max(8, min(len(traj), 24))
    samples = _select_alignment_samples(traj, times, sample_count)
    sample_projections: List[Dict[str, Dict[str, float]]] = []
    for s in samples:
        loc = carla.Location(
            x=float(s["x"] + base_dx),
            y=float(s["y"] + base_dy),
            z=float(s["z"]),
        )
        sample_projections.append(_project_to_lane_types(world_map, loc, lane_names_all))
    intent_info = _infer_alignment_intent(role, sample_projections, intent_margin)
    candidate_lanes = list(intent_info.get("candidate_lanes") or lane_names_all)
    if not candidate_lanes:
        candidate_lanes = lane_names_all

    local_limit = max(0.0, float(local_limit))
    raw_dx: List[float] = []
    raw_dy: List[float] = []
    valid_mask: List[bool] = []
    lane_keys: List[Optional[Tuple[int, int]]] = []
    yaws: List[float] = []
    lane_switches_raw = 0
    prev_lane_key: Optional[Tuple[int, int]] = None
    initial_lane_key: Optional[Tuple[int, int]] = None
    lane_dists: List[float] = []
    yaw_diffs: List[float] = []
    early_lane_lock_forced = 0
    t0 = float(times[0]) if times else 0.0
    early_lane_lock_seconds = max(0.0, float(early_lane_lock_seconds))
    early_lane_switch_override_margin = max(0.0, float(early_lane_switch_override_margin))

    for idx, wp in enumerate(traj):
        loc = carla.Location(
            x=float(wp.x + base_dx),
            y=float(wp.y + base_dy),
            z=float(wp.z),
        )
        proj = _project_to_lane_types(world_map, loc, candidate_lanes)
        best = None
        best_lane_key: Optional[Tuple[int, int]] = None
        best_score = None
        best_lane_name = None
        initial_lane_best = None
        initial_lane_best_score = None
        for lane_name in candidate_lanes:
            info = proj.get(lane_name)
            if info is None:
                continue
            lane_key = (int(info["road_id"]), int(info["lane_id"]))
            yaw_diff = _yaw_diff_deg(float(info["yaw"]), float(wp.yaw))
            score = float(info["dist"]) + 0.35 * (yaw_diff / 90.0)
            if prev_lane_key is not None and lane_key != prev_lane_key:
                score += 0.15
            if role == "npc":
                if lane_name == "Shoulder":
                    score += 0.08
                elif lane_name == "Parking":
                    score += 0.20
            elif role == "static":
                if lane_name == "Driving":
                    score += 0.35
                elif lane_name == "Shoulder":
                    score += 0.10
            if best_score is None or score < best_score:
                best_score = score
                best = info
                best_lane_key = lane_key
                best_lane_name = lane_name
            if (
                initial_lane_key is not None
                and lane_key == initial_lane_key
                and (initial_lane_best_score is None or score < initial_lane_best_score)
            ):
                initial_lane_best_score = score
                initial_lane_best = info

        if best is None:
            raw_dx.append(float(base_dx))
            raw_dy.append(float(base_dy))
            valid_mask.append(False)
            lane_keys.append(None)
            yaws.append(float(wp.yaw))
            continue

        if initial_lane_key is None and best_lane_key is not None:
            initial_lane_key = best_lane_key

        if (
            role == "npc"
            and early_lane_lock_seconds > 0.0
            and initial_lane_key is not None
            and best_lane_key is not None
            and best_lane_key != initial_lane_key
        ):
            t_rel = float(times[idx]) - t0 if idx < len(times) else 0.0
            if t_rel <= early_lane_lock_seconds + 1e-6 and initial_lane_best is not None:
                best_score_val = float(best_score) if best_score is not None else 1e9
                initial_score_val = (
                    float(initial_lane_best_score)
                    if initial_lane_best_score is not None
                    else best_score_val
                )
                # Keep initial lane early unless alternative lane is significantly better.
                if (initial_score_val - best_score_val) < early_lane_switch_override_margin:
                    best = initial_lane_best
                    best_lane_key = initial_lane_key
                    best_lane_name = None
                    best_score = initial_score_val
                    early_lane_lock_forced += 1

        dx = float(best["x"]) - float(wp.x)
        dy = float(best["y"]) - float(wp.y)

        # Bound local refinement around the global offset.
        ddx = dx - float(base_dx)
        ddy = dy - float(base_dy)
        dmag = math.hypot(ddx, ddy)
        if dmag > local_limit > 0.0:
            scale = local_limit / max(dmag, 1e-6)
            dx = float(base_dx) + ddx * scale
            dy = float(base_dy) + ddy * scale

        # Bound absolute magnitude.
        mag = math.hypot(dx, dy)
        if mag > max_shift > 0.0:
            scale = max_shift / max(mag, 1e-6)
            dx *= scale
            dy *= scale

        raw_dx.append(dx)
        raw_dy.append(dy)
        valid_mask.append(True)
        lane_keys.append(best_lane_key)
        yaws.append(float(wp.yaw))
        lane_dists.append(float(best["dist"]))
        yaw_diffs.append(_yaw_diff_deg(float(best["yaw"]), float(wp.yaw)))
        if prev_lane_key is not None and best_lane_key is not None and best_lane_key != prev_lane_key:
            lane_switches_raw += 1
        if best_lane_name is not None:
            prev_lane_key = best_lane_key

    bridge_info = {"bridged_missing": 0, "bridged_transient": 0}
    if int(bridge_max_gap_steps) > 0:
        bridge_info = _bridge_missing_or_transient_offsets(
            dxs=raw_dx,
            dys=raw_dy,
            valid_mask=valid_mask,
            lane_keys=lane_keys,
            yaws=yaws,
            max_gap_steps=int(bridge_max_gap_steps),
            straight_thresh_deg=float(bridge_straight_thresh_deg),
        )

    if any(valid_mask):
        valid_idx = [i for i, ok in enumerate(valid_mask) if ok]
        for i, ok in enumerate(valid_mask):
            if ok:
                continue
            nearest = min(valid_idx, key=lambda j: abs(j - i))
            raw_dx[i] = raw_dx[nearest]
            raw_dy[i] = raw_dy[nearest]
    else:
        return (
            [(float(base_dx), float(base_dy)) for _ in traj],
            {
                "status": "no_projection",
                "intent": intent_info,
                "candidate_lanes": candidate_lanes,
                "no_wp_ratio": 1.0,
            },
        )

    smooth_dx = _moving_average(raw_dx, smooth_window)
    smooth_dy = _moving_average(raw_dy, smooth_window)
    if smooth_dx:
        smooth_dx[0] = raw_dx[0]
        smooth_dy[0] = raw_dy[0]
        smooth_dx[-1] = raw_dx[-1]
        smooth_dy[-1] = raw_dy[-1]

    # Prevent sudden frame-to-frame offset jumps.
    max_step_delta = max(0.05, float(max_step_delta))
    for i in range(1, len(smooth_dx)):
        ddx = smooth_dx[i] - smooth_dx[i - 1]
        ddy = smooth_dy[i] - smooth_dy[i - 1]
        step = math.hypot(ddx, ddy)
        if step > max_step_delta:
            scale = max_step_delta / max(step, 1e-6)
            smooth_dx[i] = smooth_dx[i - 1] + ddx * scale
            smooth_dy[i] = smooth_dy[i - 1] + ddy * scale

    offsets: List[Tuple[float, float]] = []
    for dx, dy in zip(smooth_dx, smooth_dy):
        # Re-enforce bounds.
        ddx = dx - float(base_dx)
        ddy = dy - float(base_dy)
        dmag = math.hypot(ddx, ddy)
        if dmag > local_limit > 0.0:
            scale = local_limit / max(dmag, 1e-6)
            dx = float(base_dx) + ddx * scale
            dy = float(base_dy) + ddy * scale
        mag = math.hypot(dx, dy)
        if mag > max_shift > 0.0:
            scale = max_shift / max(mag, 1e-6)
            dx *= scale
            dy *= scale
        offsets.append((float(dx), float(dy)))

    shift_mags = [math.hypot(dx, dy) for dx, dy in offsets]
    lane_keys_valid = [lk for lk in lane_keys if lk is not None]
    lane_switches_post = 0
    prev_key: Optional[Tuple[int, int]] = None
    for lk in lane_keys:
        if lk is None:
            continue
        if prev_key is not None and lk != prev_key:
            lane_switches_post += 1
        prev_key = lk
    report = {
        "status": "ok",
        "mode": "piecewise_refine",
        "intent": intent_info,
        "candidate_lanes": candidate_lanes,
        "samples": len(samples),
        "no_wp_ratio": float(sum(1 for ok in valid_mask if not ok) / max(1, len(valid_mask))),
        "lane_switches": int(lane_switches_post),
        "lane_switches_raw": int(lane_switches_raw),
        "early_lane_lock_seconds": float(early_lane_lock_seconds),
        "early_lane_switch_override_margin": float(early_lane_switch_override_margin),
        "early_lane_lock_forced": int(early_lane_lock_forced),
        "bridge": bridge_info,
        "lane_key_coverage": float(len(lane_keys_valid) / max(1, len(lane_keys))),
        "lane_dist_median": float(_median(lane_dists)) if lane_dists else 999.0,
        "yaw_diff_median": float(_median(yaw_diffs)) if yaw_diffs else 180.0,
        "base_shift": [float(base_dx), float(base_dy)],
        "shift_median": float(_median(shift_mags)) if shift_mags else 0.0,
        "shift_max": float(max(shift_mags)) if shift_mags else 0.0,
        "first_offset": [float(offsets[0][0]), float(offsets[0][1])] if offsets else [float(base_dx), float(base_dy)],
        "last_offset": [float(offsets[-1][0]), float(offsets[-1][1])] if offsets else [float(base_dx), float(base_dy)],
    }
    return offsets, report


def _sample_offset_profile(
    times: List[float],
    offsets_wp: List[Tuple[float, float]],
    sample_times: List[float],
    always_active: bool,
) -> List[Optional[Tuple[float, float]]]:
    if not offsets_wp:
        return [None for _ in sample_times]
    if len(offsets_wp) == 1 or always_active:
        val = (float(offsets_wp[0][0]), float(offsets_wp[0][1]))
        return [val for _ in sample_times]
    out: List[Optional[Tuple[float, float]]] = []
    idx = 0
    last = len(times) - 1
    for t in sample_times:
        if t < times[0] or t > times[-1]:
            out.append(None)
            continue
        while idx + 1 < last and times[idx + 1] < t:
            idx += 1
        if idx + 1 >= len(times):
            out.append((float(offsets_wp[-1][0]), float(offsets_wp[-1][1])))
            continue
        t0 = times[idx]
        t1 = times[idx + 1]
        if t1 <= t0:
            alpha = 0.0
        else:
            alpha = (t - t0) / (t1 - t0)
        dx = float(offsets_wp[idx][0]) + (float(offsets_wp[idx + 1][0]) - float(offsets_wp[idx][0])) * alpha
        dy = float(offsets_wp[idx][1]) + (float(offsets_wp[idx + 1][1]) - float(offsets_wp[idx][1])) * alpha
        out.append((dx, dy))
    return out


def _select_alignment_samples(
    traj: List[Waypoint],
    times: List[float],
    sample_count: int,
) -> List[Dict[str, float]]:
    """Pick evenly spaced samples along the trajectory distance."""
    if not traj:
        return []
    n = len(traj)
    sample_count = max(2, min(int(sample_count), n))
    # Build cumulative distance along path
    cum = [0.0]
    for i in range(1, n):
        dx = traj[i].x - traj[i - 1].x
        dy = traj[i].y - traj[i - 1].y
        cum.append(cum[-1] + math.hypot(dx, dy))
    total = cum[-1]
    if total < 1e-6:
        idxs = sorted(
            set(int(round(i * (n - 1) / max(1, sample_count - 1))) for i in range(sample_count))
        )
    else:
        targets = [total * (i / (sample_count - 1)) for i in range(sample_count)]
        idxs = []
        for d in targets:
            idx = bisect.bisect_left(cum, d)
            idx = max(0, min(n - 1, idx))
            idxs.append(idx)
        idxs = sorted(set(idxs))
    samples: List[Dict[str, float]] = []
    for idx in idxs:
        wp = traj[idx]
        t = float(times[idx]) if idx < len(times) else float(idx)
        samples.append(
            {
                "x": float(wp.x),
                "y": float(wp.y),
                "z": float(wp.z),
                "yaw": float(wp.yaw),
                "t": t,
                "idx": float(idx),
            }
        )
    return samples


def _project_to_lane_types(
    world_map,
    loc,
    lane_type_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """Project a location to multiple lane types, returning per-lane info."""
    results: Dict[str, Dict[str, float]] = {}
    if world_map is None:
        return results
    for name in lane_type_names:
        lane_val = _lane_type_value(name)
        if lane_val is None:
            continue
        try:
            wp = world_map.get_waypoint(loc, project_to_road=True, lane_type=lane_val)
        except Exception:
            wp = None
        if wp is None:
            continue
        wloc = wp.transform.location
        dist = math.hypot(float(wloc.x) - float(loc.x), float(wloc.y) - float(loc.y))
        yaw = float(wp.transform.rotation.yaw)
        results[name] = {
            "x": float(wloc.x),
            "y": float(wloc.y),
            "z": float(wloc.z),
            "yaw": yaw,
            "dist": float(dist),
            "road_id": float(getattr(wp, "road_id", 0)),
            "lane_id": float(getattr(wp, "lane_id", 0)),
        }
    return results


def _infer_alignment_intent(
    role: str,
    projections: List[Dict[str, Dict[str, float]]],
    intent_margin: float,
) -> Dict[str, object]:
    """Infer lane intent (vehicles vs walkers) from projection distances."""
    if role in ("npc", "static"):
        lane_pref = ("Driving", "Shoulder", "Parking") if role == "npc" else ("Parking", "Shoulder", "Driving")
        lane_names = [n for n in lane_pref if any(n in p for p in projections)]
        counts = {n: 0 for n in lane_names}
        dists: Dict[str, List[float]] = {n: [] for n in lane_names}
        for proj in projections:
            best_lane = None
            best_dist = None
            for ln in lane_names:
                info = proj.get(ln)
                if info is None:
                    continue
                dists[ln].append(float(info["dist"]))
                if best_dist is None or info["dist"] < best_dist:
                    best_dist = info["dist"]
                    best_lane = ln
            if best_lane is not None:
                counts[best_lane] += 1
        total = sum(counts.values()) or 1
        intent_lane = max(counts.items(), key=lambda kv: kv[1])[0] if counts else lane_pref[0]
        intent_ratio = counts.get(intent_lane, 0) / float(total)
        med_dist = _median(dists.get(intent_lane, []))
        intent = intent_lane.lower()
        if med_dist > 6.0 and intent_ratio < 0.6:
            intent = "unknown"
        return {
            "intent": intent,
            "intent_ratio": intent_ratio,
            "lane_counts": counts,
            "lane_median_dist": {k: _median(v) for k, v in dists.items()},
            "candidate_lanes": lane_names or [lane_pref[0]],
            "score_lanes": lane_names or [lane_pref[0]],
        }

    # Walker intent
    lane_names = [n for n in ("Sidewalk", "Shoulder", "Driving") if any(n in p for p in projections)]
    sidewalk_count = 0
    road_count = 0
    mixed = 0
    sidewalk_dists: List[float] = []
    road_dists: List[float] = []
    for proj in projections:
        ds = proj.get("Sidewalk", {}).get("dist") if proj.get("Sidewalk") else None
        dd = proj.get("Driving", {}).get("dist") if proj.get("Driving") else None
        if ds is not None:
            sidewalk_dists.append(float(ds))
        if dd is not None:
            road_dists.append(float(dd))
        if ds is None and dd is None:
            continue
        if ds is not None and dd is not None:
            if ds + intent_margin < dd:
                sidewalk_count += 1
            elif dd + intent_margin < ds:
                road_count += 1
            else:
                mixed += 1
        elif ds is not None:
            sidewalk_count += 1
        elif dd is not None:
            road_count += 1
    total = sidewalk_count + road_count + mixed
    total = max(1, total)
    sidewalk_ratio = sidewalk_count / float(total)
    road_ratio = road_count / float(total)
    sidewalk_med = _median(sidewalk_dists)
    road_med = _median(road_dists)
    if sidewalk_ratio >= 0.6 and sidewalk_med < 3.0:
        intent = "sidewalk"
    elif road_ratio >= 0.6 and road_med < 3.0:
        intent = "road"
    else:
        intent = "mixed"
    if intent == "sidewalk":
        candidate_lanes = [n for n in ("Sidewalk", "Shoulder") if n in lane_names]
    elif intent == "road":
        candidate_lanes = [n for n in ("Driving", "Shoulder") if n in lane_names]
    else:
        candidate_lanes = lane_names
    if not candidate_lanes:
        candidate_lanes = lane_names or ["Sidewalk"]
    return {
        "intent": intent,
        "intent_ratio": max(sidewalk_ratio, road_ratio),
        "lane_counts": {"sidewalk": sidewalk_count, "road": road_count, "mixed": mixed},
        "lane_median_dist": {"sidewalk": sidewalk_med, "road": road_med},
        "candidate_lanes": candidate_lanes,
        "score_lanes": lane_names or ["Sidewalk", "Driving"],
    }


def _score_alignment_candidate(
    world_map,
    samples: List[Dict[str, float]],
    dx: float,
    dy: float,
    role: str,
    intent_info: Dict[str, object],
    lane_change_ref: int,
) -> Tuple[float, Dict[str, float]]:
    """Score a constant XY offset against map geometry + intent."""
    if world_map is None or not samples:
        return 1e6, {
            "dist_median": 999.0,
            "dist_p95": 999.0,
            "yaw_median": 180.0,
            "intent_mismatch": 1.0,
            "lane_changes": 0.0,
            "no_wp_ratio": 1.0,
        }
    lane_names = list(intent_info.get("score_lanes") or [])
    yaw_weight = 0.4
    distances: List[float] = []
    yaw_diffs: List[float] = []
    lane_types: List[str] = []
    lane_ids: List[Tuple[int, int]] = []
    no_wp = 0
    for sample in samples:
        loc = carla.Location(
            x=float(sample["x"] + dx),
            y=float(sample["y"] + dy),
            z=float(sample["z"]),
        )
        proj = _project_to_lane_types(world_map, loc, lane_names)
        best = None
        best_score = None
        best_lane = None
        for lane_name, info in proj.items():
            dist = float(info["dist"])
            yaw_diff = _yaw_diff_deg(float(info["yaw"]), float(sample["yaw"]))
            score = dist + yaw_weight * (yaw_diff / 90.0)
            if best_score is None or score < best_score:
                best_score = score
                best = info
                best_lane = lane_name
        if best is None:
            no_wp += 1
            continue
        distances.append(float(best["dist"]))
        yaw_diffs.append(_yaw_diff_deg(float(best["yaw"]), float(sample["yaw"])))
        lane_types.append(str(best_lane))
        lane_ids.append((int(best["road_id"]), int(best["lane_id"])))
    total = max(1, len(samples))
    no_wp_ratio = no_wp / float(total)
    dist_med = _median(distances) if distances else 999.0
    dist_p95 = _quantile(distances, 0.95) if distances else 999.0
    yaw_med = _median(yaw_diffs) if yaw_diffs else 180.0
    # Lane change count
    lane_changes = 0
    for prev, nxt in zip(lane_ids, lane_ids[1:]):
        if prev != nxt:
            lane_changes += 1
    lane_change_penalty = abs(lane_changes - int(lane_change_ref))
    # Intent mismatch
    intent = str(intent_info.get("intent") or "")
    intent_mismatch = 0.0
    if role in ("npc", "static"):
        intent_lane = intent.upper() if intent else ""
        if intent_lane:
            mismatches = sum(1 for ln in lane_types if ln.upper() != intent_lane)
            intent_mismatch = mismatches / float(max(1, len(lane_types)))
    else:
        if intent == "sidewalk":
            mismatches = sum(1 for ln in lane_types if ln == "Driving")
            intent_mismatch = mismatches / float(max(1, len(lane_types)))
        elif intent == "road":
            mismatches = sum(1 for ln in lane_types if ln == "Sidewalk")
            intent_mismatch = mismatches / float(max(1, len(lane_types)))
        else:
            intent_mismatch = 0.0
    # Weighted cost
    w_dist = 1.0
    w_p95 = 0.5
    w_yaw = 0.25
    w_intent = 2.0
    w_lane_change = 0.4
    w_offset = 0.2
    w_no_wp = 3.0
    cost = (
        w_dist * dist_med
        + w_p95 * dist_p95
        + w_yaw * (yaw_med / 90.0)
        + w_intent * intent_mismatch
        + w_lane_change * lane_change_penalty
        + w_offset * math.hypot(dx, dy)
        + w_no_wp * no_wp_ratio
    )
    return cost, {
        "dist_median": float(dist_med),
        "dist_p95": float(dist_p95),
        "yaw_median": float(yaw_med),
        "intent_mismatch": float(intent_mismatch),
        "lane_changes": float(lane_changes),
        "no_wp_ratio": float(no_wp_ratio),
    }


def _build_alignment_candidates(
    traj: List[Waypoint],
    times: List[float],
    role: str,
    world_map,
    max_shift: float,
    sample_count: int,
    window_count: int,
    intent_margin: float,
) -> Tuple[List[SpawnCandidate], Dict[str, object]]:
    """Generate alignment-based candidates using multiple waypoints."""
    if world_map is None or not traj:
        return [], {"status": "no_map_or_traj"}
    samples = _select_alignment_samples(traj, times, sample_count)
    if not samples:
        return [], {"status": "no_samples"}
    if role == "npc":
        lane_names_all = ["Driving", "Shoulder", "Parking"]
    elif role == "static":
        # Parked/static actors should preserve parking-side intent, not be pulled into lane-following.
        lane_names_all = ["Parking", "Shoulder", "Driving"]
    else:
        lane_names_all = ["Sidewalk", "Shoulder", "Driving"]
    projections: List[Dict[str, Dict[str, float]]] = []
    for s in samples:
        loc = carla.Location(x=float(s["x"]), y=float(s["y"]), z=float(s["z"]))
        projections.append(_project_to_lane_types(world_map, loc, lane_names_all))
    intent_info = _infer_alignment_intent(role, projections, intent_margin)
    candidate_lanes = list(intent_info.get("candidate_lanes") or [])
    if not candidate_lanes:
        candidate_lanes = lane_names_all

    # Reference lane change count at zero offset
    ref_cost, ref_stats = _score_alignment_candidate(
        world_map, samples, 0.0, 0.0, role, intent_info, lane_change_ref=0
    )
    lane_change_ref = int(ref_stats.get("lane_changes", 0))

    candidates: Dict[Tuple[int, int], SpawnCandidate] = {}

    def _add_offset(dx: float, dy: float, source: str) -> None:
        dist = math.hypot(dx, dy)
        if dist > max_shift + 1e-6:
            return
        key = (int(round(dx * 100)), int(round(dy * 100)))
        if key in candidates:
            return
        cost, stats = _score_alignment_candidate(
            world_map, samples, dx, dy, role, intent_info, lane_change_ref
        )
        cand = SpawnCandidate(dx=float(dx), dy=float(dy), source=source, base_cost=float(cost))
        cand.align_stats = stats
        candidates[key] = cand

    # Always keep authored (zero-offset) as a candidate.
    _add_offset(0.0, 0.0, "align_authored")

    # Global median offsets per lane type
    for lane in candidate_lanes:
        offsets = []
        for s, proj in zip(samples, projections):
            info = proj.get(lane)
            if info is None:
                continue
            offsets.append((float(info["x"]) - float(s["x"]), float(info["y"]) - float(s["y"])))
        if offsets:
            dx = _median([o[0] for o in offsets])
            dy = _median([o[1] for o in offsets])
            _add_offset(dx, dy, f"align_{lane.lower()}")

    # Best-per-sample lane median
    offsets_best = []
    for s, proj in zip(samples, projections):
        best = None
        best_dist = None
        for lane in candidate_lanes:
            info = proj.get(lane)
            if info is None:
                continue
            if best_dist is None or info["dist"] < best_dist:
                best_dist = info["dist"]
                best = info
        if best is None:
            continue
        offsets_best.append((float(best["x"]) - float(s["x"]), float(best["y"]) - float(s["y"])))
    if offsets_best:
        dx = _median([o[0] for o in offsets_best])
        dy = _median([o[1] for o in offsets_best])
        _add_offset(dx, dy, "align_best")

    # Windowed medians for local slices
    window_count = max(1, int(window_count))
    if window_count > 1 and len(samples) >= window_count:
        total = len(samples)
        indices = list(range(total))
        for wi in range(window_count):
            start = int(round(wi * total / window_count))
            end = int(round((wi + 1) * total / window_count))
            window = indices[start:end]
            if not window:
                continue
            for lane in candidate_lanes:
                offsets = []
                for idx in window:
                    proj = projections[int(idx)]
                    info = proj.get(lane)
                    if info is None:
                        continue
                    s = samples[int(idx)]
                    offsets.append((float(info["x"]) - float(s["x"]), float(info["y"]) - float(s["y"])))
                if offsets:
                    dx = _median([o[0] for o in offsets])
                    dy = _median([o[1] for o in offsets])
                    _add_offset(dx, dy, f"align_{lane.lower()}_w{wi}")

    report = {
        "status": "ok",
        "intent": intent_info,
        "sample_count": len(samples),
        "lane_change_ref": lane_change_ref,
        "candidates": [
            {
                "dx": c.dx,
                "dy": c.dy,
                "source": c.source,
                "score": c.base_cost,
                "stats": c.align_stats or {},
            }
            for c in candidates.values()
        ],
    }
    return list(candidates.values()), report


def _align_ego_trajectories(
    ego_trajs: Sequence[List[Waypoint]] | None,
    ego_times_list: Sequence[List[float]] | None,
    world,
    world_map,
    blueprint_lib,
    args: argparse.Namespace,
) -> Dict[str, object]:
    """Align ego trajectories to map geometry using the same multi-waypoint alignment logic."""
    out: Dict[str, object] = {
        "status": "ok",
        "egos": [],
        "summary": {},
    }
    if not ego_trajs:
        out["status"] = "no_egos"
        out["summary"] = {
            "egos_considered": 0,
            "egos_aligned": 0,
            "egos_z_shifted": 0,
            "egos_no_candidates": 0,
            "egos_spawn_valid": 0,
        }
        return out
    if world_map is None:
        out["status"] = "no_map"
        out["summary"] = {
            "egos_considered": len(ego_trajs),
            "egos_aligned": 0,
            "egos_z_shifted": 0,
            "egos_no_candidates": len(ego_trajs),
            "egos_spawn_valid": 0,
        }
        return out

    max_shift = max(0.0, float(args.spawn_preprocess_max_shift))
    sample_count = int(args.spawn_preprocess_align_samples)
    window_count = int(args.spawn_preprocess_align_windows)
    intent_margin = float(args.spawn_preprocess_align_intent_margin)
    normalize_z = bool(args.spawn_preprocess_normalize_z)
    piecewise_mode = bool(getattr(args, "spawn_preprocess_align_ego_piecewise", True))
    piecewise_smooth_window = int(getattr(args, "spawn_preprocess_align_ego_smooth_window", 9))
    piecewise_max_step_delta = float(getattr(args, "spawn_preprocess_align_ego_max_step_delta", 0.45))
    bridge_max_gap_steps = int(getattr(args, "spawn_preprocess_bridge_max_gap_steps", 6))
    bridge_straight_thresh_deg = float(getattr(args, "spawn_preprocess_bridge_straight_thresh_deg", 18.0))
    snap_ego_to_lane = bool(getattr(args, "snap_ego_to_lane", False))

    ego_bp = None
    ego_bp_reason = "no_blueprint_library"
    ego_bp_model = str(getattr(args, "ego_model", ""))
    if blueprint_lib is not None:
        ego_bp, ego_bp_model, ego_bp_reason = _select_blueprint(
            blueprint_lib=blueprint_lib,
            model=ego_bp_model,
            kind="npc",
            obj_type_raw="car",
        )
    out["ego_blueprint"] = {
        "requested": str(getattr(args, "ego_model", "")),
        "used": ego_bp_model,
        "reason": ego_bp_reason,
    }

    aligned = 0
    z_shifted = 0
    no_candidates = 0
    spawn_valid = 0

    for ego_idx, traj in enumerate(ego_trajs):
        entry: Dict[str, object] = {
            "ego_index": int(ego_idx),
            "status": "ok",
            "chosen": None,
        }
        out["egos"].append(entry)
        if not traj:
            entry["status"] = "empty_traj"
            no_candidates += 1
            continue

        times = _ensure_times(
            traj,
            list(ego_times_list[ego_idx]) if ego_times_list and ego_idx < len(ego_times_list) else None,
            float(args.dt),
        )
        print(f"[SPAWN_PRE][EGO] ego_idx={ego_idx}, traj_len={len(traj)}, piecewise_mode={piecewise_mode}", flush=True)
        if piecewise_mode:
            print(f"[SPAWN_PRE][EGO] calling _compute_piecewise_ego_offsets for ego {ego_idx} with max_shift={max_shift}, snap_ego_to_lane={snap_ego_to_lane}", flush=True)
            offsets, align_report = _compute_piecewise_ego_offsets(
                traj=traj,
                times=times,
                world_map=world_map,
                max_shift=max_shift,
                intent_margin=intent_margin,
                smooth_window=piecewise_smooth_window,
                max_step_delta=piecewise_max_step_delta,
                bridge_max_gap_steps=bridge_max_gap_steps,
                bridge_straight_thresh_deg=bridge_straight_thresh_deg,
                snap_ego_to_lane=snap_ego_to_lane,
            )
            entry["alignment"] = align_report
            entry["candidate_count"] = len(offsets)
            if not offsets:
                entry["status"] = "no_candidates"
                no_candidates += 1
                continue

            chosen = SpawnCandidate(
                dx=float(offsets[0][0]),
                dy=float(offsets[0][1]),
                source="align_ego_piecewise",
                base_cost=0.0,
                valid=(world is None or ego_bp is None),
                reason=None if (world is None or ego_bp is None) else "spawn_unchecked",
            )
            if world is not None and ego_bp is not None:
                _try_spawn_candidate(
                    world=world,
                    world_map=world_map,
                    blueprint=ego_bp,
                    base_wp=traj[0],
                    cand=chosen,
                    normalize_z=normalize_z,
                )

                # If spawn check failed at the first offset, try local XY retries and
                # apply the best retry as a constant adjustment over the full path.
                if not chosen.valid:
                    retry_steps = [0.0, 0.2, -0.2, 0.4, -0.4, 0.8, -0.8]
                    retry_pairs: List[Tuple[float, float]] = []
                    for jx in retry_steps:
                        for jy in retry_steps:
                            retry_pairs.append((float(jx), float(jy)))
                    retry_pairs.sort(key=lambda p: math.hypot(p[0], p[1]))
                    base_dx = float(offsets[0][0])
                    base_dy = float(offsets[0][1])
                    retry_hit = None
                    for jx, jy in retry_pairs:
                        if abs(jx) < 1e-9 and abs(jy) < 1e-9:
                            continue
                        cand = SpawnCandidate(
                            dx=base_dx + jx,
                            dy=base_dy + jy,
                            source="align_ego_piecewise_retry",
                            base_cost=math.hypot(jx, jy),
                        )
                        _try_spawn_candidate(
                            world=world,
                            world_map=world_map,
                            blueprint=ego_bp,
                            base_wp=traj[0],
                            cand=cand,
                            normalize_z=normalize_z,
                        )
                        if cand.valid:
                            retry_hit = (jx, jy, cand)
                            break
                    if retry_hit is not None:
                        jx, jy, cand_ok = retry_hit
                        updated_offsets: List[Tuple[float, float]] = []
                        for dx, dy in offsets:
                            ndx = float(dx) + float(jx)
                            ndy = float(dy) + float(jy)
                            mag = math.hypot(ndx, ndy)
                            if mag > max_shift > 0.0:
                                scale = max_shift / max(mag, 1e-6)
                                ndx *= scale
                                ndy *= scale
                            updated_offsets.append((ndx, ndy))
                        offsets = updated_offsets
                        chosen = cand_ok
                        entry["spawn_retry"] = {"dx": float(jx), "dy": float(jy)}

            shift_mags = [math.hypot(dx, dy) for dx, dy in offsets]
            for wp, (dx, dy) in zip(traj, offsets):
                wp.x += float(dx)
                wp.y += float(dy)
                wp.z += float(chosen.dz)

            if any(abs(dx) >= 1e-6 or abs(dy) >= 1e-6 for dx, dy in offsets) or abs(float(chosen.dz)) >= 1e-6:
                aligned += 1
            if abs(float(chosen.dz)) >= 1e-6:
                z_shifted += 1
            if chosen.valid:
                spawn_valid += 1

            entry["offset_stats"] = {
                "median": float(_median(shift_mags)) if shift_mags else 0.0,
                "max": float(max(shift_mags)) if shift_mags else 0.0,
            }
            entry["chosen"] = {
                "dx": float(offsets[0][0]),
                "dy": float(offsets[0][1]),
                "dz": float(chosen.dz),
                "z_source": chosen.z_source,
                "source": chosen.source,
                "base_cost": float(chosen.base_cost),
                "valid_spawn": bool(chosen.valid),
                "reason": chosen.reason,
                "mode": "piecewise",
            }
            entry["valid_candidates"] = len(offsets) if chosen.valid else 0
            if args.spawn_preprocess_verbose:
                print(
                    f"[SPAWN_PRE][EGO] ego{ego_idx} piecewise "
                    f"first_dx={offsets[0][0]:.3f} first_dy={offsets[0][1]:.3f} "
                    f"dz={chosen.dz:.3f} valid={chosen.valid} "
                    f"median_shift={(entry['offset_stats']['median']):.3f} "
                    f"max_shift={(entry['offset_stats']['max']):.3f}"
                )
        else:
            candidates, align_report = _build_alignment_candidates(
                traj=traj,
                times=times,
                role="npc",
                world_map=world_map,
                max_shift=max_shift,
                sample_count=sample_count,
                window_count=window_count,
                intent_margin=intent_margin,
            )
            entry["alignment"] = align_report
            entry["candidate_count"] = len(candidates)
            if not candidates:
                entry["status"] = "no_candidates"
                no_candidates += 1
                continue

            candidates.sort(key=lambda c: (float(c.base_cost), math.hypot(float(c.dx), float(c.dy))))
            if world is not None and ego_bp is not None:
                for cand in candidates:
                    _try_spawn_candidate(
                        world=world,
                        world_map=world_map,
                        blueprint=ego_bp,
                        base_wp=traj[0],
                        cand=cand,
                        normalize_z=normalize_z,
                    )
            valid = [c for c in candidates if c.valid]
            choose_from = valid if valid else candidates
            chosen = min(choose_from, key=lambda c: (float(c.base_cost), math.hypot(float(c.dx), float(c.dy))))

            if abs(float(chosen.dx)) >= 1e-6 or abs(float(chosen.dy)) >= 1e-6 or abs(float(chosen.dz)) >= 1e-6:
                aligned += 1
            if abs(float(chosen.dz)) >= 1e-6:
                z_shifted += 1
            if chosen.valid:
                spawn_valid += 1

            for wp in traj:
                wp.x += float(chosen.dx)
                wp.y += float(chosen.dy)
                wp.z += float(chosen.dz)

            entry["chosen"] = {
                "dx": float(chosen.dx),
                "dy": float(chosen.dy),
                "dz": float(chosen.dz),
                "z_source": chosen.z_source,
                "source": chosen.source,
                "base_cost": float(chosen.base_cost),
                "valid_spawn": bool(chosen.valid),
                "reason": chosen.reason,
                "mode": "global",
            }
            entry["valid_candidates"] = sum(1 for c in candidates if c.valid)
            if args.spawn_preprocess_verbose:
                print(
                    f"[SPAWN_PRE][EGO] ego{ego_idx} chosen "
                    f"dx={chosen.dx:.3f} dy={chosen.dy:.3f} dz={chosen.dz:.3f} "
                    f"src={chosen.source} valid={chosen.valid} cost={chosen.base_cost:.3f}"
                )

    out["summary"] = {
        "egos_considered": len(ego_trajs),
        "egos_aligned": aligned,
        "egos_z_shifted": z_shifted,
        "egos_no_candidates": no_candidates,
        "egos_spawn_valid": spawn_valid,
    }
    return out


def _build_alignment_neighbor_map(
    vehicles: Dict[int, List[Waypoint]],
    actor_meta: Dict[int, Dict[str, object]],
    radius: float,
    heading_tol_deg: float,
) -> Dict[int, List[Dict[str, object]]]:
    """Build neighbor lists for vehicles with similar heading near spawn time."""
    radius = max(0.0, float(radius))
    heading_tol_deg = max(0.0, float(heading_tol_deg))
    items: List[Tuple[int, float, float, float]] = []
    for vid, traj in vehicles.items():
        if not traj:
            continue
        meta = actor_meta.get(vid)
        if meta is None:
            continue
        kind = str(meta.get("kind") or "")
        if kind != "npc":
            continue
        wp = traj[0]
        items.append((vid, float(wp.x), float(wp.y), float(wp.yaw)))
    neighbors: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for i in range(len(items)):
        vid_i, xi, yi, yaw_i = items[i]
        hi = (math.cos(math.radians(yaw_i)), math.sin(math.radians(yaw_i)))
        for j in range(i + 1, len(items)):
            vid_j, xj, yj, yaw_j = items[j]
            dx = xj - xi
            dy = yj - yi
            dist = math.hypot(dx, dy)
            if dist > radius:
                continue
            yaw_diff = _yaw_diff_deg(yaw_i, yaw_j)
            if yaw_diff > heading_tol_deg:
                continue
            cross = hi[0] * dy - hi[1] * dx
            side_i = "left" if cross > 0 else "right"
            side_j = "left" if cross < 0 else "right"
            neighbors[vid_i].append({"id": vid_j, "dist": dist, "yaw_diff": yaw_diff, "side": side_i})
            neighbors[vid_j].append({"id": vid_i, "dist": dist, "yaw_diff": yaw_diff, "side": side_j})
    return neighbors


def _lane_type_value(name: str):
    return getattr(carla.LaneType, name, None) if carla is not None else None


def _generate_spawn_candidates(
    base_wp: Waypoint,
    role: str,
    world_map,
    max_shift: float,
    grid_steps: List[float],
    lateral_margin: float,
    random_samples: int = 0,
    rng: Optional[random.Random] = None,
) -> List[SpawnCandidate]:
    candidates: Dict[Tuple[int, int], SpawnCandidate] = {}

    def _add_candidate(dx: float, dy: float, source: str, bias: float) -> None:
        dist = math.hypot(dx, dy)
        if dist > max_shift:
            return
        key = (int(round(dx * 100)), int(round(dy * 100)))
        base_cost = dist + bias
        existing = candidates.get(key)
        if existing is None or base_cost < existing.base_cost:
            candidates[key] = SpawnCandidate(dx=dx, dy=dy, source=source, base_cost=base_cost)

    _add_candidate(0.0, 0.0, "authored", 0.0)

    # Micro-jitter grid around authored pose
    for dx in grid_steps:
        for dy in grid_steps:
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                continue
            _add_candidate(dx, dy, "grid", 0.05)

    # Radial rings to increase coverage without biasing too far
    if max_shift > 0.0:
        ring_radii = []
        step = 0.5
        r = step
        while r <= max_shift + 1e-6:
            ring_radii.append(round(r, 2))
            r += step
        angles = [i * 30 for i in range(12)]
        for r in ring_radii:
            for ang in angles:
                rad = math.radians(float(ang))
                dx = r * math.cos(rad)
                dy = r * math.sin(rad)
                _add_candidate(dx, dy, f"ring_{r:.1f}", 0.08 + 0.01 * r)

    if world_map is None:
        # Random offsets (if requested)
        if random_samples > 0:
            rng = rng or random.Random(0)
            for _ in range(random_samples):
                r = max_shift * math.sqrt(rng.random())
                theta = 2.0 * math.pi * rng.random()
                dx = r * math.cos(theta)
                dy = r * math.sin(theta)
                _add_candidate(dx, dy, "random", 0.2 + 0.01 * r)
        return list(candidates.values())

    loc = carla.Location(x=base_wp.x, y=base_wp.y, z=base_wp.z)

    lane_candidates: List[Tuple[str, object]] = []
    if role == "npc":
        for lane_name in ("Driving", "Shoulder", "Parking"):
            lane_val = _lane_type_value(lane_name)
            if lane_val is not None:
                lane_candidates.append((lane_name, lane_val))
    elif role == "static":
        # For parked/static actors, prioritize off-lane placements and avoid lane-follow attraction.
        for lane_name in ("Parking", "Shoulder", "Driving"):
            lane_val = _lane_type_value(lane_name)
            if lane_val is not None:
                lane_candidates.append((lane_name, lane_val))
    else:
        for lane_name in ("Sidewalk", "Shoulder"):
            lane_val = _lane_type_value(lane_name)
            if lane_val is not None:
                lane_candidates.append((lane_name, lane_val))

    driving_wp = None
    for lane_name, lane_val in lane_candidates:
        try:
            wp = world_map.get_waypoint(loc, project_to_road=True, lane_type=lane_val)
        except Exception:
            wp = None
        if wp is None:
            continue
        dx = float(wp.transform.location.x) - base_wp.x
        dy = float(wp.transform.location.y) - base_wp.y
        _add_candidate(dx, dy, f"lane_{lane_name.lower()}", 0.1)
        if lane_name == "Driving":
            driving_wp = wp

    # Lateral offsets from driving lane (captures shoulder-like positions)
    if driving_wp is not None and role == "npc":
        try:
            yaw = math.radians(float(driving_wp.transform.rotation.yaw))
            right = (math.sin(yaw), -math.cos(yaw))
            lane_width = getattr(driving_wp, "lane_width", 3.5) or 3.5
            for mult in (0.5, 1.0, 1.5):
                offset = mult * float(lane_width) + float(lateral_margin)
                for sign in (-1.0, 1.0):
                    dx = float(driving_wp.transform.location.x) + sign * right[0] * offset - base_wp.x
                    dy = float(driving_wp.transform.location.y) + sign * right[1] * offset - base_wp.y
                    _add_candidate(dx, dy, f"lane_lateral_{mult:.1f}", 0.12 + 0.03 * mult)
            # along-lane offsets (forward/back)
            try:
                for dist in (0.5, 1.0, 2.0, 3.0, 4.0):
                    nxt = driving_wp.next(dist)
                    if nxt:
                        loc = nxt[0].transform.location
                        _add_candidate(loc.x - base_wp.x, loc.y - base_wp.y, "lane_forward", 0.15)
                    prev = driving_wp.previous(dist)
                    if prev:
                        loc = prev[0].transform.location
                        _add_candidate(loc.x - base_wp.x, loc.y - base_wp.y, "lane_backward", 0.15)
            except Exception:
                pass
        except Exception:
            pass

    # Random offsets (if requested)
    if random_samples > 0:
        rng = rng or random.Random(0)
        for _ in range(random_samples):
            r = max_shift * math.sqrt(rng.random())
            theta = 2.0 * math.pi * rng.random()
            dx = r * math.cos(theta)
            dy = r * math.sin(theta)
            _add_candidate(dx, dy, "random", 0.2 + 0.01 * r)

    return list(candidates.values())


def _connect_carla_for_spawn(
    host: str,
    port: int,
    expected_town: Optional[str],
):
    if carla is None:
        raise RuntimeError("carla module not available")
    _assert_carla_endpoint_reachable(host, port, timeout_s=2.0)
    client = carla.Client(host, port)
    client.set_timeout(30.0)
    world = client.get_world()
    cmap = world.get_map()
    if expected_town and expected_town not in (cmap.name or ""):
        available_maps = client.get_available_maps()
        candidates = [m for m in available_maps if expected_town in m]
        if candidates:
            target_map = candidates[0]
            print(f"[INFO] Loading map '{target_map}' for spawn preprocessing")
            world = client.load_world(target_map)
            cmap = world.get_map()
    return client, world, cmap


def _extend_grid_steps(base_steps: List[float], max_shift: float) -> List[float]:
    steps = list(base_steps)
    # Add finer steps up to 1.0
    for val in (0.6, 0.8, 1.0):
        steps.extend([val, -val])
    # Add coarser steps up to max_shift
    if max_shift > 1.0:
        step = 0.5
        v = 1.5
        while v <= max_shift + 1e-6:
            steps.extend([v, -v])
            v += step
    # Deduplicate and clamp
    uniq = []
    seen = set()
    for v in steps:
        if abs(v) > max_shift + 1e-6:
            continue
        key = round(float(v), 3)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(float(v))
    return uniq


def _select_blueprint(
    blueprint_lib,
    model: str,
    kind: str,
    obj_type_raw: str,
) -> Tuple[Optional[object], str, str]:
    """
    Return (blueprint, model_used, reason).
    Tries exact match, pattern match, then role-aware fallbacks.
    """
    if blueprint_lib is None:
        return None, model, "no_blueprint_lib"

    # Exact match
    try:
        bp = blueprint_lib.find(model)
        if bp is not None:
            if kind.startswith("walker") and _is_child_walker_blueprint(getattr(bp, "id", "")):
                bp = None
            else:
                return bp, model, "exact"
    except Exception:
        pass

    # Pattern match
    try:
        matches = blueprint_lib.filter(model)
        if matches:
            if kind.startswith("walker"):
                matches = [m for m in matches if not _is_child_walker_blueprint(getattr(m, "id", ""))]
            if matches:
                return matches[0], matches[0].id, "pattern"
    except Exception:
        pass

    obj_lower = str(obj_type_raw or "").lower()
    fallback_models: List[str] = []

    if kind.startswith("walker") or "pedestrian" in obj_lower or "walker" in obj_lower:
        fallback_models = list(ADULT_WALKER_BLUEPRINTS)[:3]
    elif "bicycle" in obj_lower or "cycl" in obj_lower:
        fallback_models = [
            "vehicle.diamondback.century",
            "vehicle.gazelle.omafiets",
            "vehicle.bh.crossbike",
        ]
    elif "motor" in obj_lower or "motorcycle" in obj_lower or ("bike" in obj_lower and "bicycle" not in obj_lower):
        fallback_models = [
            "vehicle.harley-davidson.low_rider",
            "vehicle.kawasaki.ninja",
            "vehicle.yamaha.yzf",
            "vehicle.vespa.zx125",
        ]
    elif "bus" in obj_lower:
        fallback_models = [
            "vehicle.volkswagen.t2",
            "vehicle.mitsubishi.fusorosa",
        ]
    elif "truck" in obj_lower:
        fallback_models = [
            "vehicle.carlamotors.carlacola",
        ]
    elif "van" in obj_lower or "sprinter" in obj_lower:
        fallback_models = [
            "vehicle.mercedes.sprinter",
            "vehicle.volkswagen.t2",
        ]
    elif "ambulance" in obj_lower:
        fallback_models = [
            "vehicle.ford.ambulance",
        ]
    elif "police" in obj_lower:
        fallback_models = [
            "vehicle.dodge.charger_police",
            "vehicle.dodge.charger_police_2020",
        ]
    else:
        fallback_models = [
            "vehicle.tesla.model3",
            "vehicle.audi.a2",
            "vehicle.lincoln.mkz_2017",
            "vehicle.nissan.micra",
        ]

    for fallback in fallback_models:
        try:
            bp = blueprint_lib.find(fallback)
            if bp is not None and not (kind.startswith("walker") and _is_child_walker_blueprint(getattr(bp, "id", ""))):
                return bp, fallback, "fallback"
        except Exception:
            pass
        try:
            matches = blueprint_lib.filter(fallback)
            if matches:
                if kind.startswith("walker"):
                    matches = [m for m in matches if not _is_child_walker_blueprint(getattr(m, "id", ""))]
                if matches:
                    return matches[0], matches[0].id, "fallback_pattern"
        except Exception:
            pass

    # Final generic fallback
    try:
        if kind.startswith("walker"):
            matches = blueprint_lib.filter("walker.pedestrian.*")
            if matches:
                matches = [m for m in matches if not _is_child_walker_blueprint(getattr(m, "id", ""))]
                if matches:
                    return matches[0], matches[0].id, "fallback_any_walker"
    except Exception:
        pass
    try:
        matches = blueprint_lib.filter("vehicle.*")
        if matches:
            return matches[0], matches[0].id, "fallback_any_vehicle"
    except Exception:
        pass

    return None, model, "missing_blueprint"


def _preprocess_spawn_positions(
    vehicles: Dict[int, List[Waypoint]],
    vehicle_times: Dict[int, List[float]],
    actor_meta: Dict[int, Dict[str, object]],
    args: argparse.Namespace,
    ego_trajs: Sequence[List[Waypoint]] | None = None,
    ego_times_list: Sequence[List[float]] | None = None,
) -> Dict[str, object]:
    report: Dict[str, object] = {
        "settings": {},
        "actors": {},
        "summary": {},
    }

    if carla is None:
        msg = "spawn preprocess requested but CARLA Python module is unavailable"
        if bool(getattr(args, "spawn_preprocess_require_carla", False)):
            raise RuntimeError(msg)
        print(f"[WARN] {msg}; skipping.")
        report["summary"]["status"] = "skipped_no_carla"
        return report

    try:
        client, world, world_map = _connect_carla_for_spawn(
            host=args.carla_host,
            port=args.carla_port,
            expected_town=args.expected_town,
        )
    except Exception as exc:
        msg = f"spawn preprocess failed to connect to CARLA: {exc}"
        if bool(getattr(args, "spawn_preprocess_require_carla", False)):
            raise RuntimeError(msg) from exc
        print(f"[WARN] {msg}")
        report["summary"]["status"] = "skipped_carla_connect"
        return report

    try:
        map_name = world_map.name if world_map is not None else "unknown"
    except Exception:
        map_name = "unknown"
    print(
        f"[SPAWN_PRE] Connected to CARLA {args.carla_host}:{int(args.carla_port)} "
        f"(map={map_name})",
        flush=True,
    )

    blueprint_lib = world.get_blueprint_library() if world else None

    existing_actors = 0
    cleared_actors = 0
    try:
        existing_actors = len(world.get_actors()) if world else 0
    except Exception:
        existing_actors = 0

    # Clear dynamic actors to reduce spawn-test interference.
    if world is not None and existing_actors:
        try:
            to_destroy = []
            for actor in world.get_actors():
                try:
                    tid = actor.type_id or ""
                except Exception:
                    tid = ""
                if (
                    tid.startswith("vehicle.")
                    or tid.startswith("walker.")
                    or tid.startswith("sensor.")
                    or tid.startswith("controller.ai.")
                ):
                    to_destroy.append(actor.id)
            if to_destroy:
                try:
                    if hasattr(carla, "command") and client is not None:
                        commands = [carla.command.DestroyActor(aid) for aid in to_destroy]
                        client.apply_batch_sync(commands, True)
                    else:
                        for actor in world.get_actors(to_destroy):
                            try:
                                actor.destroy()
                            except Exception:
                                pass
                except Exception:
                    for actor in world.get_actors(to_destroy):
                        try:
                            actor.destroy()
                        except Exception:
                            pass
                cleared_actors = len(to_destroy)
                try:
                    settings = world.get_settings()
                    if getattr(settings, "synchronous_mode", False):
                        world.tick()
                    else:
                        world.wait_for_tick()
                except Exception:
                    pass
        except Exception:
            pass

    if cleared_actors:
        print(f"[SPAWN_PRE] Cleared {cleared_actors} dynamic actors from CARLA world before spawn checks.")
    elif existing_actors > 10:
        print(f"[WARN] CARLA world has {existing_actors} existing actors; spawn tests may be affected.")

    max_shift = max(0.0, float(args.spawn_preprocess_max_shift))
    grid_steps = []
    for token in re.split(r"[,\s]+", str(args.spawn_preprocess_grid or "").strip()):
        if not token:
            continue
        try:
            grid_steps.append(float(token))
        except Exception:
            continue
    if not grid_steps:
        grid_steps = [0.0, 0.2, -0.2, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2]
    grid_steps = _extend_grid_steps(grid_steps, max_shift)
    lateral_margin = 0.6

    sample_dt = float(args.spawn_preprocess_sample_dt)
    grid_size = max(1.0, float(args.spawn_preprocess_grid_size))
    max_candidates = max(5, int(args.spawn_preprocess_max_candidates))
    collision_weight = float(args.spawn_preprocess_collision_weight)
    normalize_z = bool(args.spawn_preprocess_normalize_z)
    random_samples = max(0, int(args.spawn_preprocess_random_samples))
    debug_radius = float(args.spawn_preprocess_debug_radius)
    debug_max_items = int(args.spawn_preprocess_debug_max_items)
    align_enabled = bool(args.spawn_preprocess_align)
    align_samples = int(args.spawn_preprocess_align_samples)
    align_windows = int(args.spawn_preprocess_align_windows)
    align_intent_margin = float(args.spawn_preprocess_align_intent_margin)
    align_neighbor_radius = float(args.spawn_preprocess_align_neighbor_radius)
    align_neighbor_weight = float(args.spawn_preprocess_align_neighbor_weight)
    refine_piecewise = bool(getattr(args, "spawn_preprocess_refine_piecewise", True))
    refine_max_local = float(getattr(args, "spawn_preprocess_refine_max_local", 0.8))
    refine_smooth_window = int(getattr(args, "spawn_preprocess_refine_smooth_window", 7))
    refine_max_step_delta = float(getattr(args, "spawn_preprocess_refine_max_step_delta", 0.35))
    refine_collision_slack = float(getattr(args, "spawn_preprocess_refine_collision_slack", 0.0))
    bridge_max_gap_steps = int(getattr(args, "spawn_preprocess_bridge_max_gap_steps", 6))
    bridge_straight_thresh_deg = float(getattr(args, "spawn_preprocess_bridge_straight_thresh_deg", 18.0))

    all_times: List[float] = []
    for vid, traj in vehicles.items():
        meta = actor_meta.get(vid)
        if meta is None:
            continue
        times = _ensure_times(traj, vehicle_times.get(vid), args.dt)
        all_times.extend(times)
    sample_times = _build_time_grid(all_times, sample_dt)

    report["settings"] = {
        "max_shift": max_shift,
        "grid_steps": grid_steps,
        "sample_dt": sample_dt,
        "grid_size": grid_size,
        "max_candidates": max_candidates,
        "collision_weight": collision_weight,
        "normalize_z": normalize_z,
        "random_samples": random_samples,
        "debug_radius": debug_radius,
        "debug_max_items": debug_max_items,
        "cleared_dynamic_actors": cleared_actors,
        "sample_times": len(sample_times),
        "align_enabled": align_enabled,
        "align_samples": align_samples,
        "align_windows": align_windows,
        "align_intent_margin": align_intent_margin,
        "align_neighbor_radius": align_neighbor_radius,
        "align_neighbor_weight": align_neighbor_weight,
        "align_ego_enabled": bool(getattr(args, "spawn_preprocess_align_ego", True)),
        "align_ego_piecewise": bool(getattr(args, "spawn_preprocess_align_ego_piecewise", True)),
        "snap_ego_to_lane": bool(getattr(args, "snap_ego_to_lane", False)),
        "align_ego_smooth_window": int(getattr(args, "spawn_preprocess_align_ego_smooth_window", 9)),
        "align_ego_max_step_delta": float(getattr(args, "spawn_preprocess_align_ego_max_step_delta", 0.45)),
        "static_path_threshold": float(getattr(args, "static_path_threshold", 1.2)),
        "static_net_disp_threshold": float(getattr(args, "static_net_disp_threshold", 0.8)),
        "static_bbox_extent_threshold": float(getattr(args, "static_bbox_extent_threshold", 0.9)),
        "static_avg_speed_threshold": float(getattr(args, "static_avg_speed_threshold", 0.8)),
        "static_heavy_path_threshold": float(getattr(args, "static_heavy_path_threshold", 8.0)),
        "static_heavy_bbox_extent_threshold": float(getattr(args, "static_heavy_bbox_extent_threshold", 1.2)),
        "static_heavy_avg_speed_threshold": float(getattr(args, "static_heavy_avg_speed_threshold", 0.8)),
        "refine_piecewise": refine_piecewise,
        "refine_max_local": refine_max_local,
        "refine_smooth_window": refine_smooth_window,
        "refine_max_step_delta": refine_max_step_delta,
        "refine_early_lane_lock_seconds": float(
            getattr(args, "spawn_preprocess_refine_early_lane_lock_seconds", 1.0)
        ),
        "refine_early_lane_switch_override_margin": float(
            getattr(args, "spawn_preprocess_refine_early_lane_switch_override_margin", 0.9)
        ),
        "refine_collision_slack": refine_collision_slack,
        "bridge_max_gap_steps": bridge_max_gap_steps,
        "bridge_straight_thresh_deg": bridge_straight_thresh_deg,
    }

    # Precompute base positions and radii
    base_positions: Dict[int, List[Optional[Tuple[float, float]]]] = {}
    radii: Dict[int, float] = {}
    times_cache: Dict[int, List[float]] = {}
    for vid, traj in vehicles.items():
        meta = actor_meta.get(vid)
        if meta is None:
            continue
        times = _ensure_times(traj, vehicle_times.get(vid), args.dt)
        times_cache[vid] = times
        kind = str(meta.get("kind"))
        always_active = kind in ("static", "walker_static")
        base_positions[vid] = _sample_positions(traj, times, sample_times, always_active)
        radii[vid] = _actor_radius(
            kind,
            meta.get("length"),
            meta.get("width"),
            str(meta.get("model", "")),
        )

    # Cache world actors/env objects for debug
    actor_items: List[Dict[str, object]] = []
    env_items: List[Dict[str, object]] = []
    if world is not None:
        try:
            for actor in world.get_actors():
                try:
                    loc = actor.get_location()
                    tf = actor.get_transform()
                    bbox = actor.bounding_box
                except Exception:
                    continue
                actor_items.append(
                    {
                        "id": int(actor.id),
                        "type": getattr(actor, "type_id", "actor"),
                        "loc": loc,
                        "bbox": _bbox_corners_2d(bbox, tf),
                    }
                )
        except Exception:
            pass
        try:
            label_any = getattr(carla.CityObjectLabel, "Any", None)
            env_objs = world.get_environment_objects(label_any) if label_any is not None else world.get_environment_objects()
            for env in env_objs:
                try:
                    tf = env.transform
                    loc = tf.location
                    bbox = env.bounding_box
                except Exception:
                    continue
                env_items.append(
                    {
                        "id": int(getattr(env, "id", -1)),
                        "type": getattr(env, "type_id", getattr(env, "type", "env")),
                        "loc": loc,
                        "bbox": _bbox_corners_2d(bbox, tf),
                    }
                )
        except Exception:
            pass

    # Build candidate lists with spawn validity
    print(f"[SPAWN_PRE] Building spawn candidates for {len(vehicles)} actors ...", flush=True)
    candidates_by_actor: Dict[int, List[SpawnCandidate]] = {}
    bp_by_actor: Dict[int, object] = {}
    actors_processed = 0
    for vid, traj in vehicles.items():
        meta = actor_meta.get(vid)
        if meta is None or not traj:
            continue
        kind = str(meta.get("kind"))
        if kind == "npc":
            role = "npc"
        elif kind == "static":
            role = "static"
        else:
            role = "walker"
        model = str(meta.get("model") or "")
        actor_report = {
            "kind": kind,
            "model": model,
            "model_used": model,
            "candidates": [],
            "chosen": None,
        }
        report["actors"][str(vid)] = actor_report

        base_wp = traj[0]
        candidates = _generate_spawn_candidates(
            base_wp=base_wp,
            role=role,
            world_map=world_map,
            max_shift=max_shift,
            grid_steps=grid_steps,
            lateral_margin=lateral_margin,
            random_samples=random_samples,
            rng=random.Random(vid),
        )
        # Alignment-based candidates (multi-waypoint intent-aware offsets)
        if align_enabled:
            times = times_cache.get(vid) or _ensure_times(traj, vehicle_times.get(vid), args.dt)
            align_candidates, align_report = _build_alignment_candidates(
                traj=traj,
                times=times,
                role=role,
                world_map=world_map,
                max_shift=max_shift,
                sample_count=align_samples,
                window_count=align_windows,
                intent_margin=align_intent_margin,
            )
            if align_report:
                actor_report["alignment"] = align_report
            if align_candidates:
                merged: Dict[Tuple[int, int], SpawnCandidate] = {}
                for cand in candidates:
                    key = (int(round(cand.dx * 100)), int(round(cand.dy * 100)))
                    merged[key] = cand
                for cand in align_candidates:
                    key = (int(round(cand.dx * 100)), int(round(cand.dy * 100)))
                    existing = merged.get(key)
                    if existing is None or cand.base_cost < existing.base_cost:
                        merged[key] = cand
                candidates = list(merged.values())

        # Sort by base_cost and keep the best candidates first, ensuring alignment candidates stay.
        if candidates:
            align_list = [c for c in candidates if str(c.source).startswith("align_")]
            other_list = [c for c in candidates if c not in align_list]
            other_list.sort(key=lambda c: c.base_cost)
            keep_count = max(0, max_candidates - len(align_list))
            candidates = align_list + other_list[:keep_count]
        else:
            candidates = []

        if blueprint_lib is None:
            print(f"[WARN] Blueprint library unavailable; skipping spawn validation for actor {vid}.")
            for cand in candidates:
                cand.valid = True
                cand.reason = "no_blueprint_lib"
                cand.spawn_loc = (float(base_wp.x + cand.dx), float(base_wp.y + cand.dy), float(base_wp.z))
                cand.dz = 0.0
                cand.z_source = "authored"
            candidates_by_actor[vid] = candidates
            src_counts = {}
            for c in candidates:
                src_counts[c.source] = src_counts.get(c.source, 0) + 1
            actor_report["candidates"] = [
                {
                    "dx": c.dx,
                    "dy": c.dy,
                    "source": c.source,
                    "valid": c.valid,
                    "reason": c.reason,
                    "base_cost": c.base_cost,
                    "spawn_loc": c.spawn_loc,
                    "dz": c.dz,
                    "z_source": c.z_source,
                    "align_stats": c.align_stats,
                }
                for c in candidates
            ]
            actor_report["candidate_stats"] = {
                "total": len(candidates),
                "valid": len(candidates),
                "invalid": 0,
                "source_counts": src_counts,
                "failure_reasons": {},
            }
            actor_report["spawn_base"] = {
                "x": float(base_wp.x),
                "y": float(base_wp.y),
                "z": float(base_wp.z),
                "yaw": float(base_wp.yaw),
            }
            continue

        bp, model_used, reason = _select_blueprint(blueprint_lib, model, kind, str(meta.get("obj_type") or ""))
        actor_report["model_used"] = model_used
        actor_report["blueprint_reason"] = reason
        if bp is None:
            print(f"[WARN] No blueprint found for actor {vid} model '{model}'; leaving trajectory unchanged.")
            candidates_by_actor[vid] = []
            src_counts = {}
            for c in candidates:
                src_counts[c.source] = src_counts.get(c.source, 0) + 1
            actor_report["candidates"] = [
                {
                    "dx": c.dx,
                    "dy": c.dy,
                    "source": c.source,
                    "valid": False,
                    "reason": "missing_blueprint",
                    "base_cost": c.base_cost,
                    "spawn_loc": (float(base_wp.x + c.dx), float(base_wp.y + c.dy), float(base_wp.z)),
                    "dz": 0.0,
                    "z_source": "authored",
                    "align_stats": c.align_stats,
                }
                for c in candidates
            ]
            actor_report["status"] = "missing_blueprint"
            actor_report["candidate_stats"] = {
                "total": len(candidates),
                "valid": 0,
                "invalid": len(candidates),
                "source_counts": src_counts,
                "failure_reasons": {"missing_blueprint": len(candidates)} if candidates else {},
            }
            actor_report["spawn_base"] = {
                "x": float(base_wp.x),
                "y": float(base_wp.y),
                "z": float(base_wp.z),
                "yaw": float(base_wp.yaw),
            }
            continue
        bp_by_actor[vid] = bp
        if model_used and model_used != model:
            print(f"[WARN] Blueprint '{model}' unavailable; using '{model_used}' for actor {vid}.")
            meta["model"] = model_used
            model = model_used
            actor_report["model"] = model_used

        for cand in candidates:
            _try_spawn_candidate(world, world_map, bp, base_wp, cand, normalize_z)

        # If no valid candidates, expand search once more (try harder)
        if not any(c.valid for c in candidates):
            hard_max_shift = min(max_shift * 2.0, max_shift + 4.0)
            hard_grid_steps = _extend_grid_steps(grid_steps, hard_max_shift)
            hard_candidates = _generate_spawn_candidates(
                base_wp=base_wp,
                role=role,
                world_map=world_map,
                max_shift=hard_max_shift,
                grid_steps=hard_grid_steps,
                lateral_margin=lateral_margin,
                random_samples=random_samples * 2,
                rng=random.Random(vid + 100000),
            )
            hard_candidates.sort(key=lambda c: c.base_cost)
            hard_candidates = hard_candidates[: max_candidates * 3]
            for cand in hard_candidates:
                _try_spawn_candidate(world, world_map, bp, base_wp, cand, normalize_z)
            # merge candidates (keep best cost per offset)
            merged: Dict[Tuple[int, int], SpawnCandidate] = {}
            for cand in candidates + hard_candidates:
                key = (int(round(cand.dx * 100)), int(round(cand.dy * 100)))
                existing = merged.get(key)
                if existing is None or cand.base_cost < existing.base_cost:
                    merged[key] = cand
            candidates = sorted(merged.values(), key=lambda c: c.base_cost)
            candidates = candidates[: max_candidates * 2]

        candidates_by_actor[vid] = candidates
        valid_count = sum(1 for c in candidates if c.valid)
        invalid_count = len(candidates) - valid_count
        reasons: Dict[str, int] = {}
        sources: Dict[str, int] = {}
        for c in candidates:
            sources[c.source] = sources.get(c.source, 0) + 1
            if not c.valid:
                key = c.reason or "spawn_failed"
                reasons[key] = reasons.get(key, 0) + 1
        actor_report["candidate_stats"] = {
            "total": len(candidates),
            "valid": valid_count,
            "invalid": invalid_count,
            "source_counts": sources,
            "failure_reasons": reasons,
        }
        actor_report["spawn_base"] = {
            "x": float(base_wp.x),
            "y": float(base_wp.y),
            "z": float(base_wp.z),
            "yaw": float(base_wp.yaw),
        }
        actor_report["candidates"] = [
            {
                "dx": c.dx,
                "dy": c.dy,
                "source": c.source,
                "valid": c.valid,
                "reason": c.reason,
                "base_cost": c.base_cost,
                "spawn_loc": c.spawn_loc,
                "dz": c.dz,
                "z_source": c.z_source,
                "align_stats": c.align_stats,
            }
            for c in candidates
        ]
        actors_processed += 1
        if actors_processed % 10 == 0 or actors_processed == len(vehicles):
            print(f"[SPAWN_PRE] Spawn candidates: {actors_processed}/{len(vehicles)} actors done, actor {vid}: {valid_count}/{len(candidates)} valid", flush=True)

    print(f"[SPAWN_PRE] All spawn candidates built for {actors_processed} actors.", flush=True)
    neighbor_map: Dict[int, List[Dict[str, object]]] = {}
    if align_enabled and align_neighbor_weight > 0.0 and align_neighbor_radius > 0.0:
        neighbor_map = _build_alignment_neighbor_map(
            vehicles=vehicles,
            actor_meta=actor_meta,
            radius=align_neighbor_radius,
            heading_tol_deg=25.0,
        )
        report["settings"]["align_neighbor_pairs"] = sum(len(v) for v in neighbor_map.values())
        for vid, infos in neighbor_map.items():
            entry = report["actors"].get(str(vid))
            if entry is not None and isinstance(entry, dict):
                align_entry = entry.get("alignment")
                if isinstance(align_entry, dict):
                    align_entry["neighbors"] = infos

    # Global assignment with spatiotemporal collision avoidance
    occupancy: List[Dict[Tuple[int, int], List[Tuple[float, float, float, int]]]] = [
        defaultdict(list) for _ in sample_times
    ]

    def _collision_score(vid: int, cand: SpawnCandidate) -> float:
        positions = base_positions.get(vid, [])
        radius = radii.get(vid, 1.0)
        score = 0.0
        for t_idx, pos in enumerate(positions):
            if pos is None:
                continue
            x = pos[0] + cand.dx
            y = pos[1] + cand.dy
            cell_x = int(math.floor(x / grid_size))
            cell_y = int(math.floor(y / grid_size))
            cell_map = occupancy[t_idx]
            for gx in range(cell_x - 1, cell_x + 2):
                for gy in range(cell_y - 1, cell_y + 2):
                    for ox, oy, orad, oid in cell_map.get((gx, gy), []):
                        dist = math.hypot(x - ox, y - oy)
                        if dist < (radius + orad):
                            score += 1.0
        return score

    chosen_offsets: Dict[int, SpawnCandidate] = {}
    actor_order = sorted(
        [vid for vid in vehicles.keys() if vid in candidates_by_actor],
        key=lambda vid: len([c for c in candidates_by_actor[vid] if c.valid]) or 9999,
    )
    print(f"[SPAWN_PRE] Running global assignment for {len(actor_order)} actors ...", flush=True)

    for vid in actor_order:
        meta = actor_meta.get(vid)
        if meta is None:
            continue
        cands = [c for c in candidates_by_actor.get(vid, []) if c.valid]
        if not cands:
            # fallback to no shift
            fallback = SpawnCandidate(
                dx=0.0,
                dy=0.0,
                source="fallback",
                base_cost=0.0,
                valid=False,
                reason="no_valid_candidates",
                dz=0.0,
                z_source="fallback",
            )
            chosen_offsets[vid] = fallback
            report["actors"][str(vid)]["chosen"] = {
                "dx": 0.0,
                "dy": 0.0,
                "dz": 0.0,
                "z_source": "fallback",
                "source": "fallback",
                "collision_score": None,
                "status": "no_valid_candidates",
            }
            cand_all = candidates_by_actor.get(vid, [])
            if cand_all:
                best_invalid = min(cand_all, key=lambda c: c.base_cost)
                report["actors"][str(vid)]["best_invalid_candidate"] = {
                    "dx": best_invalid.dx,
                    "dy": best_invalid.dy,
                    "source": best_invalid.source,
                    "base_cost": best_invalid.base_cost,
                    "reason": best_invalid.reason,
                    "spawn_loc": best_invalid.spawn_loc,
                    "dz": best_invalid.dz,
                    "z_source": best_invalid.z_source,
                }
            # add debug info for failed spawns
            base_wp = vehicles.get(vid, [None])[0]
            entry = report["actors"].get(str(vid), {})
            if base_wp is not None:
                bp = bp_by_actor.get(vid)
                probe_yaw = str(entry.get("kind", "")).startswith("npc") or str(entry.get("kind", "")).startswith("static")
                collect_spawn_debug = globals().get("_collect_spawn_debug")
                if callable(collect_spawn_debug):
                    entry["debug"] = collect_spawn_debug(
                        actor_id=vid,
                        base_wp=base_wp,
                        entry=entry,
                        world=world,
                        world_map=world_map,
                        blueprint=bp,
                        actor_items=actor_items,
                        env_items=env_items,
                        max_dist=debug_radius,
                        max_items=debug_max_items,
                        probe_yaw=probe_yaw,
                    )
            continue

        best = None
        best_score = None
        best_collision = None
        best_neighbor = None
        for cand in cands:
            collision = _collision_score(vid, cand)
            neighbor_penalty = 0.0
            if neighbor_map and align_neighbor_weight > 0.0:
                offsets = []
                for info in neighbor_map.get(vid, []):
                    nid = int(info.get("id", -1))
                    chosen = chosen_offsets.get(nid)
                    if chosen is None:
                        continue
                    offsets.append((float(chosen.dx), float(chosen.dy)))
                if offsets:
                    avg_dx = sum(o[0] for o in offsets) / float(len(offsets))
                    avg_dy = sum(o[1] for o in offsets) / float(len(offsets))
                    neighbor_penalty = align_neighbor_weight * math.hypot(cand.dx - avg_dx, cand.dy - avg_dy)
            total = cand.base_cost + collision_weight * collision + neighbor_penalty
            if best_score is None or total < best_score:
                best = cand
                best_score = total
                best_collision = collision
                best_neighbor = neighbor_penalty

        if best is None:
            continue

        chosen_offsets[vid] = best
        report["actors"][str(vid)]["chosen"] = {
            "dx": best.dx,
            "dy": best.dy,
            "dz": best.dz,
            "z_source": best.z_source,
            "source": best.source,
            "collision_score": best_collision,
            "neighbor_penalty": best_neighbor,
            "status": "ok",
        }

        # Update occupancy
        positions = base_positions.get(vid, [])
        radius = radii.get(vid, 1.0)
        for t_idx, pos in enumerate(positions):
            if pos is None:
                continue
            x = pos[0] + best.dx
            y = pos[1] + best.dy
            cell = (int(math.floor(x / grid_size)), int(math.floor(y / grid_size)))
            occupancy[t_idx][cell].append((x, y, radius, vid))

        if args.spawn_preprocess_verbose:
            print(
                f"[SPAWN_PRE] actor {vid} kind={meta.get('kind')} model={meta.get('model')} "
                f"chosen dx={best.dx:.3f} dy={best.dy:.3f} dz={best.dz:.3f} "
                f"z_src={best.z_source} source={best.source} "
                f"collision={best_collision} neighbor={best_neighbor}"
            )

    # Optional second-stage refinement:
    #   1) keep globally optimized shift (collision-aware) as the anchor
    #   2) compute bounded per-waypoint local refinement around that anchor
    #   3) accept only if collision score does not get worse beyond slack
    piecewise_profiles: Dict[int, List[Tuple[float, float]]] = {}
    refined_accepted = 0
    refined_rejected = 0
    if refine_piecewise and world_map is not None and chosen_offsets:
        print(f"[SPAWN_PRE] Starting piecewise refinement for {len(chosen_offsets)} actors ...", flush=True)
        occupancy_const_all: List[Dict[Tuple[int, int], List[Tuple[float, float, float, int]]]] = [
            defaultdict(list) for _ in sample_times
        ]
        for oid, ocand in chosen_offsets.items():
            opos = base_positions.get(oid, [])
            orad = radii.get(oid, 1.0)
            for t_idx, pos in enumerate(opos):
                if pos is None:
                    continue
                x = pos[0] + float(ocand.dx)
                y = pos[1] + float(ocand.dy)
                cell = (int(math.floor(x / grid_size)), int(math.floor(y / grid_size)))
                occupancy_const_all[t_idx][cell].append((x, y, orad, oid))

        def _profile_collision_score(
            actor_id: int,
            sample_offsets: List[Optional[Tuple[float, float]]],
        ) -> float:
            score = 0.0
            apos = base_positions.get(actor_id, [])
            arad = radii.get(actor_id, 1.0)
            for t_idx, pos in enumerate(apos):
                if pos is None:
                    continue
                off = sample_offsets[t_idx] if t_idx < len(sample_offsets) else None
                if off is None:
                    continue
                x = float(pos[0]) + float(off[0])
                y = float(pos[1]) + float(off[1])
                cell_x = int(math.floor(x / grid_size))
                cell_y = int(math.floor(y / grid_size))
                cell_map = occupancy_const_all[t_idx]
                for gx in range(cell_x - 1, cell_x + 2):
                    for gy in range(cell_y - 1, cell_y + 2):
                        for ox, oy, orad, oid in cell_map.get((gx, gy), []):
                            if int(oid) == int(actor_id):
                                continue
                            if math.hypot(x - ox, y - oy) < (arad + orad):
                                score += 1.0
            return score

        refine_count = 0
        for vid, cand in chosen_offsets.items():
            refine_count += 1
            if refine_count % 10 == 0:
                print(f"[SPAWN_PRE] Refinement: {refine_count}/{len(chosen_offsets)} actors (accepted={refined_accepted}, rejected={refined_rejected})", flush=True)
            meta = actor_meta.get(vid)
            if meta is None:
                continue
            if not cand.valid:
                continue
            traj = vehicles.get(vid)
            if not traj or len(traj) < 3:
                continue
            times = times_cache.get(vid) or _ensure_times(traj, vehicle_times.get(vid), args.dt)
            kind = str(meta.get("kind") or "")
            if kind != "npc":
                entry = report["actors"].get(str(vid), {})
                entry["refinement"] = {
                    "accepted": False,
                    "reason": "skipped_non_moving_kind",
                    "kind": kind,
                }
                continue
            role = "npc"
            always_active = kind in ("static", "walker_static")

            refined_offsets, refine_report = _compute_piecewise_actor_offsets(
                traj=traj,
                times=times,
                world_map=world_map,
                role=role,
                base_dx=float(cand.dx),
                base_dy=float(cand.dy),
                max_shift=max_shift,
                intent_margin=align_intent_margin,
                local_limit=refine_max_local,
                smooth_window=refine_smooth_window,
                max_step_delta=refine_max_step_delta,
                bridge_max_gap_steps=bridge_max_gap_steps,
                bridge_straight_thresh_deg=bridge_straight_thresh_deg,
                early_lane_lock_seconds=float(
                    getattr(args, "spawn_preprocess_refine_early_lane_lock_seconds", 1.0)
                ),
                early_lane_switch_override_margin=float(
                    getattr(args, "spawn_preprocess_refine_early_lane_switch_override_margin", 0.9)
                ),
            )
            entry = report["actors"].get(str(vid), {})
            entry["refinement"] = refine_report
            if not refined_offsets:
                entry["refinement"]["accepted"] = False
                entry["refinement"]["reason"] = "no_offsets"
                refined_rejected += 1
                continue

            base_profile = _sample_offset_profile(
                times=times,
                offsets_wp=[(float(cand.dx), float(cand.dy)) for _ in traj],
                sample_times=sample_times,
                always_active=always_active,
            )
            refined_profile = _sample_offset_profile(
                times=times,
                offsets_wp=refined_offsets,
                sample_times=sample_times,
                always_active=always_active,
            )
            base_collision = _profile_collision_score(vid, base_profile)
            refined_collision = _profile_collision_score(vid, refined_profile)
            entry["refinement"]["base_collision"] = float(base_collision)
            entry["refinement"]["refined_collision"] = float(refined_collision)
            entry["refinement"]["collision_slack"] = float(refine_collision_slack)

            # Ensure first spawn remains valid for refined profile.
            spawn_ok = True
            if world is not None and vid in bp_by_actor and refined_offsets:
                spawn_probe = SpawnCandidate(
                    dx=float(refined_offsets[0][0]),
                    dy=float(refined_offsets[0][1]),
                    source="refine_probe",
                    base_cost=0.0,
                )
                _try_spawn_candidate(
                    world=world,
                    world_map=world_map,
                    blueprint=bp_by_actor[vid],
                    base_wp=traj[0],
                    cand=spawn_probe,
                    normalize_z=normalize_z,
                )
                spawn_ok = bool(spawn_probe.valid)
                entry["refinement"]["spawn_valid"] = bool(spawn_ok)
                entry["refinement"]["spawn_probe_reason"] = spawn_probe.reason

            if (not spawn_ok) or (refined_collision > base_collision + float(refine_collision_slack)):
                entry["refinement"]["accepted"] = False
                entry["refinement"]["reason"] = "spawn_invalid" if not spawn_ok else "collision_regression"
                refined_rejected += 1
                continue

            piecewise_profiles[vid] = refined_offsets
            entry["refinement"]["accepted"] = True
            refined_accepted += 1
            if args.spawn_preprocess_verbose:
                print(
                    f"[SPAWN_PRE][REFINE] actor {vid} accepted "
                    f"base_col={base_collision:.1f} refined_col={refined_collision:.1f}"
                )

    # Apply offsets to trajectories
    total_shifted = 0
    total_z_shifted = 0
    for vid, cand in chosen_offsets.items():
        profile = piecewise_profiles.get(vid)
        if profile is None and abs(cand.dx) < 1e-6 and abs(cand.dy) < 1e-6 and abs(cand.dz) < 1e-6:
            continue
        traj = vehicles.get(vid)
        if not traj:
            continue
        if profile is None:
            for wp in traj:
                wp.x += cand.dx
                wp.y += cand.dy
                wp.z += cand.dz
        else:
            for idx, wp in enumerate(traj):
                dx, dy = profile[min(idx, len(profile) - 1)]
                wp.x += float(dx)
                wp.y += float(dy)
                wp.z += cand.dz
            entry = report["actors"].get(str(vid))
            if isinstance(entry, dict):
                chosen_entry = entry.get("chosen")
                if isinstance(chosen_entry, dict):
                    chosen_entry["mode"] = "global_plus_piecewise_refine"
                    chosen_entry["first_dx"] = float(profile[0][0])
                    chosen_entry["first_dy"] = float(profile[0][1])
                    chosen_entry["last_dx"] = float(profile[-1][0])
                    chosen_entry["last_dy"] = float(profile[-1][1])
        total_shifted += 1
        if abs(cand.dz) >= 1e-6:
            total_z_shifted += 1

    ego_report: Dict[str, object] | None = None
    ego_summary: Dict[str, object] = {
        "egos_considered": 0,
        "egos_aligned": 0,
        "egos_z_shifted": 0,
        "egos_no_candidates": 0,
        "egos_spawn_valid": 0,
    }
    print(f"[SPAWN_PRE] Global assignment done. Starting ego trajectory alignment ...", flush=True)
    if bool(getattr(args, "spawn_preprocess_align_ego", True)):
        ego_report = _align_ego_trajectories(
            ego_trajs=ego_trajs,
            ego_times_list=ego_times_list,
            world=world,
            world_map=world_map,
            blueprint_lib=blueprint_lib,
            args=args,
        )
        report["ego"] = ego_report
        if isinstance(ego_report.get("summary"), dict):
            ego_summary = dict(ego_report["summary"])

    missing_bp = 0
    no_valid = 0
    fallback_bp = 0
    valid_counts: List[int] = []
    reason_totals: Dict[str, int] = {}
    source_totals: Dict[str, int] = {}
    bridge_missing_total = 0
    bridge_transient_total = 0
    for entry in report.get("actors", {}).values():
        if entry.get("status") == "missing_blueprint":
            missing_bp += 1
        chosen = entry.get("chosen") or {}
        if chosen.get("status") == "no_valid_candidates":
            no_valid += 1
        reason = str(entry.get("blueprint_reason") or "")
        if reason.startswith("fallback"):
            fallback_bp += 1
        stats = entry.get("candidate_stats") or {}
        if stats:
            valid_counts.append(int(stats.get("valid", 0)))
            for src, cnt in (stats.get("source_counts") or {}).items():
                source_totals[src] = source_totals.get(src, 0) + int(cnt)
            for r, cnt in (stats.get("failure_reasons") or {}).items():
                reason_totals[r] = reason_totals.get(r, 0) + int(cnt)
        refinement = entry.get("refinement")
        if isinstance(refinement, dict):
            bridge = refinement.get("bridge")
            if isinstance(bridge, dict):
                bridge_missing_total += int(bridge.get("bridged_missing", 0))
                bridge_transient_total += int(bridge.get("bridged_transient", 0))

    ego_bridge_missing = 0
    ego_bridge_transient = 0
    if isinstance(ego_report, dict):
        for ego_entry in ego_report.get("egos", []):
            if not isinstance(ego_entry, dict):
                continue
            alignment = ego_entry.get("alignment")
            if not isinstance(alignment, dict):
                continue
            bridge = alignment.get("bridge")
            if not isinstance(bridge, dict):
                continue
            ego_bridge_missing += int(bridge.get("bridged_missing", 0))
            ego_bridge_transient += int(bridge.get("bridged_transient", 0))

    report["summary"] = {
        "status": "ok",
        "actors_considered": len(actor_meta),
        "actors_shifted": total_shifted,
        "actors_z_shifted": total_z_shifted,
        "actors_missing_blueprint": missing_bp,
        "actors_no_valid_candidates": no_valid,
        "actors_with_fallback_blueprint": fallback_bp,
        "candidate_valid_counts": {
            "min": min(valid_counts) if valid_counts else 0,
            "max": max(valid_counts) if valid_counts else 0,
            "avg": (sum(valid_counts) / max(1, len(valid_counts))) if valid_counts else 0.0,
        },
        "candidate_source_totals": source_totals,
        "candidate_failure_reasons": reason_totals,
        "actors_refined_piecewise_accepted": int(refined_accepted),
        "actors_refined_piecewise_rejected": int(refined_rejected),
        "actors_bridged_missing_steps": int(bridge_missing_total),
        "actors_bridged_transient_steps": int(bridge_transient_total),
        "egos_considered": int(ego_summary.get("egos_considered", 0)),
        "egos_aligned": int(ego_summary.get("egos_aligned", 0)),
        "egos_z_shifted": int(ego_summary.get("egos_z_shifted", 0)),
        "egos_no_candidates": int(ego_summary.get("egos_no_candidates", 0)),
        "egos_spawn_valid": int(ego_summary.get("egos_spawn_valid", 0)),
        "egos_bridged_missing_steps": int(ego_bridge_missing),
        "egos_bridged_transient_steps": int(ego_bridge_transient),
    }
    ok_count = max(0, len(actor_meta) - missing_bp - no_valid)
    print(
        "[SPAWN_PRE] Summary: "
        f"actors={len(actor_meta)} ok={ok_count} "
        f"no_valid={no_valid} missing_bp={missing_bp} "
        f"fallback_bp={fallback_bp} shifted={total_shifted} z_shifted={total_z_shifted} "
        f"refine_ok={int(refined_accepted)} refine_reject={int(refined_rejected)} "
        f"bridge_missing={int(bridge_missing_total)} bridge_transient={int(bridge_transient_total)}"
    )
    if int(ego_summary.get("egos_considered", 0)) > 0:
        print(
            "[SPAWN_PRE] Ego alignment: "
            f"egos={int(ego_summary.get('egos_considered', 0))} "
            f"aligned={int(ego_summary.get('egos_aligned', 0))} "
            f"spawn_valid={int(ego_summary.get('egos_spawn_valid', 0))} "
            f"no_candidates={int(ego_summary.get('egos_no_candidates', 0))} "
            f"z_shifted={int(ego_summary.get('egos_z_shifted', 0))}"
        )
    if no_valid:
        samples = []
        for actor_id, entry in report.get("actors", {}).items():
            chosen = entry.get("chosen") or {}
            if chosen.get("status") != "no_valid_candidates":
                continue
            best = entry.get("best_invalid_candidate") or {}
            model_used = entry.get("model_used") or entry.get("model")
            samples.append(
                f"id={actor_id} kind={entry.get('kind')} model={model_used} "
                f"best_src={best.get('source')} cost={best.get('base_cost')}"
            )
            if len(samples) >= 10:
                break
        if samples:
            print("[SPAWN_PRE] No-valid examples: " + "; ".join(samples))
    return report
