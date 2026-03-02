"""Route export internals: alignment and transform helpers."""

from __future__ import annotations

from v2xpnp.pipeline.route_export_stage_01_foundation import *  # noqa: F401,F403
# Explicitly import private helpers excluded by the wildcard import above
from v2xpnp.pipeline.route_export_stage_01_foundation import (  # noqa: F401
    _safe_int,
    _safe_float,
    _scipy_lsa,
    _normalize_yaw_deg,
    _lerp_angle_deg,
    _progress_bar,
    _is_pedestrian_type,
    _is_cyclist_type,
    _is_parked_vehicle,
    _dominant_lane,
    _parse_lane_type_set,
    _grp_yaw_diff_deg,
    _grp_route_length,
    _grp_route_has_turn,
    _ensure_carla_grp,
)



def _grp_refine_trajectory_timing_preserving(
    carla_map,
    grp,
    waypoints_carla: List[Dict],
    timestamps: List[float],
    radius: float = 2.5,
    k: int = 6,
    heading_thresh: float = 40.0,
    lane_change_penalty: float = 50.0,
    turn_penalty: float = 8.0,
    deviation_weight: float = 1.0,
    route_weight: float = 0.8,
    waypoint_step: float = 1.0,
) -> List[Dict]:
    """GRP-aware trajectory alignment that preserves ALL timing waypoints.

    Unlike step_07's refine_waypoints_dp which may compress/skip waypoints,
    this variant keeps every input waypoint to maintain temporal fidelity.
    It snaps each waypoint to the nearest valid CARLA lane while penalising
    lane changes and GRP-illegal transitions.

    Parameters
    ----------
    carla_map : carla.Map
    grp : GlobalRoutePlanner
    waypoints_carla : list of dict with 'x', 'y', 'z', 'yaw'
    timestamps : parallel list of float timestamps
    radius : search radius for CARLA waypoint candidates
    k : max candidates per input waypoint
    heading_thresh : yaw tolerance for candidate filtering (degrees)
    lane_change_penalty : DP cost for switching lanes
    turn_penalty : DP cost for GRP-indicated turns
    deviation_weight : weight for spatial deviation from original position
    route_weight : weight for GRP route-length cost
    waypoint_step : CARLA map generate_waypoints resolution

    Returns
    -------
    list of dict : aligned waypoints with 'x', 'y', 'z', 'yaw', 't' fields
    """
    global _grp_carla_module  # Need carla module for Location creation
    
    n = len(waypoints_carla)
    if n == 0:
        return []
    if n == 1:
        wp = waypoints_carla[0]
        return [{"x": wp["x"], "y": wp["y"], "z": wp["z"], "yaw": wp["yaw"],
                 "t": timestamps[0] if timestamps else 0.0, "grp_aligned": True}]

    # Generate all map waypoints as snap candidates
    all_wps = carla_map.generate_waypoints(waypoint_step)
    if not all_wps:
        # Fallback: return unmodified
        return [
            {"x": wp["x"], "y": wp["y"], "z": wp["z"], "yaw": wp["yaw"],
             "t": timestamps[i] if i < len(timestamps) else 0.0, "grp_aligned": False}
            for i, wp in enumerate(waypoints_carla)
        ]

    coords = np.array([[w.transform.location.x, w.transform.location.y] for w in all_wps])

    # Recompute headings from the input trajectory for direction matching.
    # Use a wider forward-looking window (up to HEADING_WINDOW frames) so that
    # duplicate-position frames or micro-jitter don't yield garbage headings.
    # For the FIRST FEW frames, use an even wider window to capture overall
    # trajectory direction — critical for correct snapping in intersections.
    HEADING_WINDOW = 5
    FIRST_FRAME_LOOKAHEAD = 20  # Wider lookahead for first few frames
    FIRST_FRAME_COUNT = 5  # Apply wider lookahead to these many initial frames
    
    # First compute a "global" trajectory heading from overall displacement
    # This helps detect the intended direction for vehicles starting in intersections
    global_heading: Optional[float] = None
    if n >= 3:
        # Find a point with meaningful displacement from start
        for end_idx in range(min(FIRST_FRAME_LOOKAHEAD, n - 1), 0, -1):
            gdx = waypoints_carla[end_idx]["x"] - waypoints_carla[0]["x"]
            gdy = waypoints_carla[end_idx]["y"] - waypoints_carla[0]["y"]
            if abs(gdx) > 1.0 or abs(gdy) > 1.0:  # Need substantial displacement
                global_heading = math.degrees(math.atan2(gdy, gdx))
                break
    
    headings = [0.0] * n
    for i in range(n):
        dx, dy = 0.0, 0.0
        # For first few frames, use wider lookahead to get overall direction
        effective_window = FIRST_FRAME_LOOKAHEAD if i < FIRST_FRAME_COUNT else HEADING_WINDOW
        # Try forward window first
        for j in range(1, effective_window + 1):
            if i + j < n:
                dx = waypoints_carla[i + j]["x"] - waypoints_carla[i]["x"]
                dy = waypoints_carla[i + j]["y"] - waypoints_carla[i]["y"]
                if abs(dx) > 0.3 or abs(dy) > 0.3:  # need meaningful displacement
                    break
        if abs(dx) <= 0.3 and abs(dy) <= 0.3:
            # Forward didn't help — try backward window
            for j in range(1, effective_window + 1):
                if i - j >= 0:
                    dx = waypoints_carla[i]["x"] - waypoints_carla[i - j]["x"]
                    dy = waypoints_carla[i]["y"] - waypoints_carla[i - j]["y"]
                    if abs(dx) > 0.3 or abs(dy) > 0.3:
                        break
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            headings[i] = math.degrees(math.atan2(dy, dx))
        elif global_heading is not None and i < FIRST_FRAME_COUNT:
            # For first few frames with no displacement, use global heading
            headings[i] = global_heading
        elif i > 0:
            headings[i] = headings[i - 1]

    rad2 = radius * radius

    def _get_candidates(idx: int) -> List[int]:
        wp = waypoints_carla[idx]
        xy = np.array([wp["x"], wp["y"]])
        diff = coords - xy[None, :]
        dist2 = np.einsum("ij,ij->i", diff, diff)
        idxs = np.where(dist2 <= rad2)[0]
        if idxs.size == 0:
            # Expand search to nearest k
            nearest_k = min(k * 3, len(dist2))
            idxs = np.argsort(dist2)[:nearest_k]
        # Filter by heading — graduated fallback, never accept opposite direction
        traj_yaw = headings[idx]
        filtered = []
        for ci in idxs:
            wp_yaw = all_wps[ci].transform.rotation.yaw
            if _grp_yaw_diff_deg(wp_yaw, traj_yaw) <= heading_thresh:
                filtered.append((ci, dist2[ci]))
        if not filtered:
            # Relaxed filter: accept up to 90° but never opposite direction
            for ci in idxs:
                wp_yaw = all_wps[ci].transform.rotation.yaw
                if _grp_yaw_diff_deg(wp_yaw, traj_yaw) <= 90.0:
                    filtered.append((ci, dist2[ci]))
        if not filtered:
            # Last resort: accept up to 120° — still reject clearly opposite (>120°)
            for ci in idxs:
                wp_yaw = all_wps[ci].transform.rotation.yaw
                if _grp_yaw_diff_deg(wp_yaw, traj_yaw) <= 120.0:
                    filtered.append((ci, dist2[ci]))
        if not filtered:
            # Expanded-radius search: widen to 3× radius, still enforce heading
            expanded_rad2 = rad2 * 9.0  # 3× radius
            exp_idxs = np.where(dist2 <= expanded_rad2)[0]
            if exp_idxs.size == 0:
                exp_idxs = np.argsort(dist2)[:min(k * 6, len(dist2))]
            for ci in exp_idxs:
                wp_yaw = all_wps[ci].transform.rotation.yaw
                if _grp_yaw_diff_deg(wp_yaw, traj_yaw) <= 120.0:
                    filtered.append((ci, dist2[ci]))
        if not filtered:
            # Still empty — try 150° (reject only truly opposite > 150°)
            all_sorted = np.argsort(dist2)
            for ci in all_sorted[:min(k * 10, len(dist2))]:
                wp_yaw = all_wps[ci].transform.rotation.yaw
                if _grp_yaw_diff_deg(wp_yaw, traj_yaw) <= 150.0:
                    filtered.append((ci, dist2[ci]))
                    if len(filtered) >= k:
                        break
        if not filtered:
            # Absolute last resort: nearest single candidate (will be penalised
            # heavily by heading cost in the DP, but keeps DP from having empty
            # candidate sets which would break the algorithm).
            filtered = [(int(np.argmin(dist2)), float(np.min(dist2)))]
        filtered.sort(key=lambda x: x[1])
        return [ci for ci, _ in filtered[:k]]

    # Build candidates per waypoint
    cands_per_wp = [_get_candidates(i) for i in range(n)]

    # DP: for each waypoint i and each candidate ci, compute min cost
    INF = float("inf")
    best_cost = [dict() for _ in range(n)]
    backref = [dict() for _ in range(n)]

    # Initialize first waypoint — include heading penalty so wrong-direction
    # lanes don't seed the DP path at near-zero cost.
    # ENHANCED: Use much stronger heading penalty and forward-looking route check
    # to prevent wrong-direction snaps in intersections.
    
    # Find a forward reference point for route compatibility check
    forward_ref_idx = min(10, n - 1)  # Look ~10 frames ahead
    forward_ref_wp = waypoints_carla[forward_ref_idx]
    forward_ref_loc = None
    if forward_ref_idx > 0:
        try:
            forward_ref_loc = _grp_carla_module.Location(
                x=forward_ref_wp["x"],
                y=forward_ref_wp["y"],
                z=forward_ref_wp.get("z", 0.0),
            )
        except Exception:
            forward_ref_loc = None
    
    for ci in cands_per_wp[0]:
        dev = float(np.linalg.norm(np.array([waypoints_carla[0]["x"], waypoints_carla[0]["y"]]) - coords[ci]))
        heading_diff_0 = _grp_yaw_diff_deg(
            all_wps[ci].transform.rotation.yaw, headings[0]
        )
        
        # ENHANCED heading cost for first frame - be very strict about direction
        heading_cost_0 = 0.0
        if heading_diff_0 > 120.0:
            # Opposite direction - nearly impossible to recover
            heading_cost_0 = lane_change_penalty * 10.0
        elif heading_diff_0 > 90.0:
            # Wrong quadrant - strongly disfavour
            heading_cost_0 = lane_change_penalty * 6.0
        elif heading_diff_0 > 60.0:
            # Significant mismatch - moderate penalty
            heading_cost_0 = lane_change_penalty * 2.0
        elif heading_diff_0 > heading_thresh:
            # Minor mismatch - light penalty
            heading_cost_0 = deviation_weight * heading_diff_0 * 0.3
        
        # Forward-looking route compatibility check for first frame
        # Penalize candidates that can't route to where the trajectory goes
        route_compat_cost = 0.0
        if forward_ref_loc is not None and forward_ref_idx > 0:
            try:
                first_wp_loc = all_wps[ci].transform.location
                route_to_future = grp.trace_route(first_wp_loc, forward_ref_loc)
                if not route_to_future:
                    # No route found - heavily penalize
                    route_compat_cost = lane_change_penalty * 3.0
                else:
                    # Check if route involves excessive backtracking
                    route_len = _grp_route_length(route_to_future)
                    direct_dist = first_wp_loc.distance(forward_ref_loc)
                    if direct_dist > 1.0 and route_len > direct_dist * 3.0:
                        # Route is much longer than direct path - might be wrong direction
                        route_compat_cost = lane_change_penalty * 1.5
            except Exception:
                # GRP route check failed - small penalty
                route_compat_cost = lane_change_penalty * 0.5
        
        best_cost[0][ci] = deviation_weight * dev + heading_cost_0 + route_compat_cost
        backref[0][ci] = None

    # Forward pass — every waypoint MUST be kept (no skipping)
    for i in range(1, n):
        obs = np.array([waypoints_carla[i]["x"], waypoints_carla[i]["y"]])
        for ci in cands_per_wp[i]:
            dev = float(np.linalg.norm(obs - coords[ci]))
            base_cost = deviation_weight * dev
            best_val = INF
            best_prev_ci = None

            curr_wp = all_wps[ci]
            for prev_ci, prev_cost in best_cost[i - 1].items():
                prev_wp = all_wps[prev_ci]

                # Same lane bonus / lane change penalty
                same_lane = (
                    prev_wp.road_id == curr_wp.road_id
                    and prev_wp.lane_id == curr_wp.lane_id
                )

                # GRP route check — only compute for non-trivially-close waypoints
                dist_between = prev_wp.transform.location.distance(curr_wp.transform.location)
                route_cost = 0.0
                has_turn = False
                if dist_between > 0.5:
                    try:
                        route = grp.trace_route(prev_wp.transform.location, curr_wp.transform.location)
                        if route:
                            route_len = _grp_route_length(route)
                            has_turn = _grp_route_has_turn(route)
                            route_cost = route_weight * route_len
                        else:
                            route_cost = route_weight * dist_between * 3.0  # penalty for no-route
                    except Exception:
                        route_cost = route_weight * dist_between * 2.0
                else:
                    route_cost = route_weight * dist_between

                # Heading mismatch penalty: penalise candidates whose
                # lane direction disagrees with actual travel direction
                heading_diff = _grp_yaw_diff_deg(
                    curr_wp.transform.rotation.yaw, headings[i]
                )
                heading_cost = 0.0
                if heading_diff > 90.0:
                    heading_cost = lane_change_penalty * 4.0  # strongly disfavour opposite direction
                elif heading_diff > heading_thresh:
                    heading_cost = deviation_weight * heading_diff * 0.15

                total = (
                    prev_cost
                    + base_cost
                    + route_cost
                    + heading_cost
                    + (0.0 if same_lane else lane_change_penalty)
                    + (turn_penalty if has_turn else 0.0)
                )

                if total < best_val:
                    best_val = total
                    best_prev_ci = prev_ci

            if best_prev_ci is not None:
                best_cost[i][ci] = best_val
                backref[i][ci] = best_prev_ci

    # Backtrack
    if not best_cost[-1]:
        # Fallback: return unmodified
        return [
            {"x": wp["x"], "y": wp["y"], "z": wp["z"], "yaw": wp["yaw"],
             "t": timestamps[i] if i < len(timestamps) else 0.0, "grp_aligned": False}
            for i, wp in enumerate(waypoints_carla)
        ]

    end_ci = min(best_cost[-1], key=lambda ci: best_cost[-1][ci])
    chosen_cis = [0] * n
    ci = end_ci
    for i in range(n - 1, -1, -1):
        chosen_cis[i] = ci
        prev = backref[i].get(ci)
        if prev is not None:
            ci = prev

    # Build output — preserve ALL timestamps
    result = []
    for i in range(n):
        cw = all_wps[chosen_cis[i]]
        loc = cw.transform.location
        yaw = cw.transform.rotation.yaw
        result.append({
            "x": float(loc.x),
            "y": float(loc.y),
            "z": float(loc.z),
            "yaw": float(yaw),
            "t": float(timestamps[i]) if i < len(timestamps) else 0.0,
            "road_id": int(cw.road_id),
            "lane_id": int(cw.lane_id),
            "grp_aligned": True,
        })

    # Post-process: suppress micro lane-switches (A→B→A patterns)
    if len(result) >= 3:
        lane_ids = [(r["road_id"], r["lane_id"]) for r in result]
        # Build runs
        runs: List[Tuple[Tuple[int, int], int, int]] = []
        cur_lane = lane_ids[0]
        run_start = 0
        for ri in range(1, len(lane_ids)):
            if lane_ids[ri] != cur_lane:
                runs.append((cur_lane, run_start, ri - 1))
                cur_lane = lane_ids[ri]
                run_start = ri
        runs.append((cur_lane, run_start, len(lane_ids) - 1))

        # Suppress short B in A→B→A
        MAX_SHORT_RUN = 8
        changed = True
        while changed:
            changed = False
            for ri in range(1, len(runs) - 1):
                a_lane = runs[ri - 1][0]
                b_lane = runs[ri][0]
                c_lane = runs[ri + 1][0]
                b_len = runs[ri][2] - runs[ri][1] + 1
                if a_lane == c_lane and b_lane != a_lane and b_len <= MAX_SHORT_RUN:
                    # Re-snap B frames to A lane
                    for fi in range(runs[ri][1], runs[ri][2] + 1):
                        nearest_a = carla_map.get_waypoint(
                            _grp_carla_module.Location(
                                x=waypoints_carla[fi]["x"],
                                y=waypoints_carla[fi]["y"],
                                z=waypoints_carla[fi].get("z", 0.0),
                            ),
                            project_to_road=True,
                            lane_type=_grp_carla_module.LaneType.Driving,
                        )
                        if nearest_a is not None:
                            loc = nearest_a.transform.location
                            result[fi]["x"] = float(loc.x)
                            result[fi]["y"] = float(loc.y)
                            result[fi]["z"] = float(loc.z)
                            result[fi]["yaw"] = float(nearest_a.transform.rotation.yaw)
                            result[fi]["road_id"] = int(nearest_a.road_id)
                            result[fi]["lane_id"] = int(nearest_a.lane_id)
                    # Merge runs
                    runs[ri - 1] = (a_lane, runs[ri - 1][1], runs[ri + 1][2])
                    runs.pop(ri + 1)
                    runs.pop(ri)
                    changed = True
                    break

    return result


def _is_actor_confidently_stationary(
    traj: Sequence[Waypoint],
    times: Sequence[float] | None,
    default_dt: float,
    parked_cfg: Dict[str, float],
) -> Tuple[bool, str, Dict[str, float]]:
    """Enhanced stationary classification that refines (not replaces) _is_parked_vehicle.

    Adds:
    - Velocity-based sustained motion check
    - Positional variance check
    - Sustained displacement window check
    - Jitter detection (high path length but low net displacement)
    - Per-frame path efficiency check

    Returns (is_stationary, reason, stats).
    """
    is_parked, stats = _is_parked_vehicle(traj, times, default_dt, parked_cfg)
    if is_parked:
        return True, "parked_basic", stats

    n = len(traj)
    if n < 3:
        return False, "too_few_points", stats

    # Additional checks for borderline cases
    xs = np.array([float(wp.x) for wp in traj], dtype=np.float64)
    ys = np.array([float(wp.y) for wp in traj], dtype=np.float64)

    # Compute path metrics
    diffs = np.sqrt((xs[1:] - xs[:-1]) ** 2 + (ys[1:] - ys[:-1]) ** 2)
    path_len = float(np.sum(diffs)) if diffs.size else 0.0
    net_disp = float(math.hypot(xs[-1] - xs[0], ys[-1] - ys[0])) if n >= 2 else 0.0
    stats["jitter_path_len_m"] = float(path_len)
    stats["jitter_net_disp_m"] = float(net_disp)

    # 1. Positional variance check: if the variance is extremely low, it's stationary
    var_x = float(np.var(xs))
    var_y = float(np.var(ys))
    total_var = var_x + var_y
    stats["pos_variance_m2"] = float(total_var)
    if total_var < 0.15:
        stats["parked_vehicle"] = 1.0
        return True, "low_variance", stats

    # 2. Velocity-based check: compute per-step velocities
    dt_val = float(default_dt) if default_dt > 0 else 0.1
    velocities = diffs / dt_val
    stats["velocity_median_mps"] = float(np.median(velocities)) if velocities.size else 0.0
    stats["velocity_p95_mps"] = float(np.quantile(velocities, 0.95)) if velocities.size else 0.0

    # If median velocity is walking speed and p95 is still very low → stationary with jitter
    if float(np.median(velocities)) < 0.3 and float(np.quantile(velocities, 0.95)) < 0.8:
        stats["parked_vehicle"] = 1.0
        return True, "low_velocity", stats

    # 3. Sustained displacement: check if max displacement over any 2-second window is tiny
    window_frames = max(3, int(2.0 / dt_val))
    max_window_disp = 0.0
    for start in range(0, n - window_frames + 1):
        end = start + window_frames
        window_xs = xs[start:end]
        window_ys = ys[start:end]
        disp = math.hypot(float(window_xs[-1] - window_xs[0]), float(window_ys[-1] - window_ys[0]))
        max_window_disp = max(max_window_disp, disp)
    stats["max_2s_window_disp_m"] = float(max_window_disp)
    if max_window_disp < 0.5:
        stats["parked_vehicle"] = 1.0
        return True, "low_sustained_disp", stats

    # 4. Jitter detection: high path length with very low net displacement
    # This catches actors that vibrate in place (sensor noise, tracking jitter)
    jitter_net_disp_max_m = float(parked_cfg.get("jitter_net_disp_max_m", 5.0))
    jitter_path_ratio_min = float(parked_cfg.get("jitter_path_ratio_min", 3.0))
    jitter_variance_max_m2 = float(parked_cfg.get("jitter_variance_max_m2", 2.0))

    if net_disp < jitter_net_disp_max_m:
        # Low net displacement - check if it's jitter
        path_efficiency = net_disp / max(0.01, path_len)  # 0 = pure jitter, 1 = straight line
        stats["jitter_path_efficiency"] = float(path_efficiency)

        # Jitter: lots of movement but going nowhere (low efficiency) + low variance
        if path_len > 0.5 and path_efficiency < (1.0 / jitter_path_ratio_min) and total_var < jitter_variance_max_m2:
            stats["parked_vehicle"] = 1.0
            return True, "jitter_detected", stats

    # 5. Per-frame path check: if path per frame is extremely low for many frames, it's stationary
    path_per_frame = path_len / max(1, n - 1)
    stats["path_per_frame_m"] = float(path_per_frame)
    # Vehicles typically move at least 0.05m per frame (0.5 m/s at 10 Hz)
    # For many-frame trajectories with tiny per-frame movement, mark as jitter
    if n >= 50 and path_per_frame < 0.03 and net_disp < 3.0:
        stats["parked_vehicle"] = 1.0
        return True, "low_path_per_frame", stats

    return False, "moving", stats


# ---------------------------------------------------------------------------
# Outer-lane bias for parked vehicles
# ---------------------------------------------------------------------------


def _find_outermost_driving_lane_wp(carla_wp, carla_mod):
    """Walk from *carla_wp* toward the road edge and return the outermost
    **Driving** lane waypoint on the same road.

    CARLA convention: negative lane_id → right-hand side of the road (where
    parked cars typically sit).  We walk via ``get_right_lane()`` for
    negative lane_ids and ``get_left_lane()`` for positive ones.

    Returns *carla_wp* itself if it is already the outermost, or if the
    outermost lane is not a Driving lane.
    """
    if carla_wp is None:
        return carla_wp

    road_id = carla_wp.road_id
    best = carla_wp

    # Determine walk direction based on lane_id sign
    lid = carla_wp.lane_id
    if lid < 0:
        step_fn = lambda w: w.get_right_lane()
    elif lid > 0:
        step_fn = lambda w: w.get_left_lane()
    else:
        return carla_wp  # lane_id == 0 is center, nothing to do

    MAX_STEPS = 10  # safety limit
    cur = carla_wp
    for _ in range(MAX_STEPS):
        nxt = step_fn(cur)
        if nxt is None:
            break
        # Must stay on the same road
        if nxt.road_id != road_id:
            break
        # Must be a Driving lane
        if nxt.lane_type != carla_mod.LaneType.Driving:
            break
        best = nxt
        cur = nxt

    return best


def _build_moving_vehicle_spatial_set(
    vehicles: Dict[int, List],
    stationary_vids: set,
    to_carla_fn,
    cell_size: float = 2.0,
) -> set:
    """Build a set of (grid_col, grid_row) cells occupied by *moving*
    vehicles' trajectories (in CARLA coords).  Used to quickly check
    whether a candidate parked-vehicle position would interfere.

    *to_carla_fn* converts (v2x_x, v2x_y) → (carla_x, carla_y).
    """
    cells: set = set()
    inv = 1.0 / cell_size
    for vid, traj in vehicles.items():
        if vid in stationary_vids:
            continue
        for wp in traj:
            cx, cy = to_carla_fn(float(wp.x), float(wp.y))
            cells.add((int(math.floor(cx * inv)), int(math.floor(cy * inv))))
    return cells


def _point_in_moving_cells(
    cx: float, cy: float, cells: set, cell_size: float = 2.0, radius: float = 3.0,
) -> bool:
    """Return True if the CARLA point (cx, cy) is within *radius* of any
    cell in *cells* (approximated by checking the surrounding grid cells)."""
    inv = 1.0 / cell_size
    r_cells = int(math.ceil(radius / cell_size))
    gx = int(math.floor(cx * inv))
    gy = int(math.floor(cy * inv))
    for dx in range(-r_cells, r_cells + 1):
        for dy in range(-r_cells, r_cells + 1):
            if (gx + dx, gy + dy) in cells:
                return True
    return False


def _try_bias_parked_to_outer_lane(
    snapped_wp,           # CARLA waypoint from get_waypoint()
    carla_map,
    carla_mod,            # the carla module itself
    moving_cells: set,    # grid cells of moving vehicles
    cell_size: float = 2.0,
    max_lateral_m: float = 4.0,
    interference_radius: float = 3.0,
):
    """Attempt to move a parked vehicle to the outermost driving lane.

    Returns ``(outer_wp, did_move, reason)`` where *outer_wp* is the
    chosen CARLA waypoint (either the outer lane or the original if we
    decided not to move), *did_move* is a bool, and *reason* is a short
    string for the report.

    Rules:
      1. Find the outermost driving lane on the same road.
      2. If it's the same lane → no-op.
      3. If the lateral distance > *max_lateral_m* → too far, skip.
      4. If moving there would place the vehicle inside a moving-vehicle
         path cell that wasn't already interfered → revert.
    """
    if snapped_wp is None:
        return snapped_wp, False, "no_snap"

    outer_wp = _find_outermost_driving_lane_wp(snapped_wp, carla_mod)

    # Same lane already?
    if (outer_wp.road_id == snapped_wp.road_id
            and outer_wp.lane_id == snapped_wp.lane_id):
        return snapped_wp, False, "already_outermost"

    # Check lateral distance
    oloc = outer_wp.transform.location
    sloc = snapped_wp.transform.location
    lateral_dist = math.hypot(
        float(oloc.x) - float(sloc.x),
        float(oloc.y) - float(sloc.y),
    )
    if lateral_dist > max_lateral_m:
        return snapped_wp, False, f"too_far_{lateral_dist:.1f}m"

    # Interference check — does the new position create a new collision
    # with a moving vehicle path that the original position did not have?
    orig_interferes = _point_in_moving_cells(
        float(sloc.x), float(sloc.y), moving_cells,
        cell_size=cell_size, radius=interference_radius,
    )
    new_interferes = _point_in_moving_cells(
        float(oloc.x), float(oloc.y), moving_cells,
        cell_size=cell_size, radius=interference_radius,
    )

    if new_interferes and not orig_interferes:
        # Moving would create new interference — revert
        return snapped_wp, False, "would_add_interference"

    return outer_wp, True, f"biased_outer_lane_{lateral_dist:.1f}m"


def _check_parked_obstructs_moving_path(
    parked_x: float,
    parked_y: float,
    moving_cells: set,
    cell_size: float = 2.0,
    obstruction_radius: float = 4.0,
) -> bool:
    """Check if a parked vehicle at (parked_x, parked_y) in CARLA coords
    obstructs any moving vehicle path.
    
    Returns True if the parked vehicle is within obstruction_radius of any
    moving vehicle path cell.
    """
    return _point_in_moving_cells(
        parked_x, parked_y, moving_cells,
        cell_size=cell_size, radius=obstruction_radius,
    )


def _find_obstructing_parked_vehicles(
    parked_positions: Dict[int, Tuple[float, float]],  # vid -> (carla_x, carla_y)
    moving_cells: set,
    ego_cells: set,
    cell_size: float = 2.0,
    obstruction_radius: float = 4.0,
) -> Tuple[List[int], List[int]]:
    """Identify parked vehicles that obstruct moving actors or ego.
    
    Returns (obstructs_ego_vids, obstructs_actor_vids).
    Vehicles obstructing ego are higher priority for removal.
    """
    obstructs_ego: List[int] = []
    obstructs_actor: List[int] = []
    
    for vid, (cx, cy) in parked_positions.items():
        # Check if it obstructs ego path
        if _point_in_moving_cells(cx, cy, ego_cells, cell_size, obstruction_radius):
            obstructs_ego.append(vid)
        # Check if it obstructs any moving actor path
        elif _point_in_moving_cells(cx, cy, moving_cells, cell_size, obstruction_radius):
            obstructs_actor.append(vid)
    
    return obstructs_ego, obstructs_actor


def _grp_align_trajectories(
    vehicles: Dict[int, List[Waypoint]],
    vehicle_times: Dict[int, List[float]],
    ego_trajs: List[List[Waypoint]],
    ego_times: List[List[float]],
    obj_info: Dict[int, Dict[str, object]],
    parked_vehicle_cfg: Dict[str, float],
    align_cfg: Dict[str, object],
    carla_host: str = "localhost",
    carla_port: int = 2005,
    carla_map_name: str = "ucla_v2",
    sampling_resolution: float = 2.0,
    snap_radius: float = 2.5,
    snap_k: int = 6,
    heading_thresh: float = 40.0,
    lane_change_penalty: float = 50.0,
    default_dt: float = 0.1,
    actor_max_median_displacement_m: float = 2.0,
    actor_max_p90_displacement_m: float = 4.0,
    actor_max_displacement_m: float = 10.0,
    ego_max_median_displacement_m: float = 1.25,
    ego_max_p90_displacement_m: float = 2.5,
    ego_max_displacement_m: float = 6.0,
    enabled: bool = True,
) -> Tuple[
    Dict[int, List[Waypoint]],
    Dict[int, List[float]],
    List[List[Waypoint]],
    List[List[float]],
    Dict[str, object],
]:
    """Apply GRP-aware trajectory alignment for all non-stationary actors and egos.

    Connects to CARLA, transforms V2XPNP coordinates into CARLA frame,
    runs DP-based GRP alignment, then transforms back.

    Returns updated (vehicles, vehicle_times, ego_trajs, ego_times, report).
    """
    import inspect as _inspect

    report: Dict[str, object] = {
        "enabled": bool(enabled),
        "actors_processed": 0,
        "actors_skipped_stationary": 0,
        "actors_skipped_short": 0,
        "actors_aligned": 0,
        "actors_failed": 0,
        "actors_rejected_displacement": 0,
        "egos_processed": 0,
        "egos_aligned": 0,
        "egos_failed": 0,
        "egos_rejected_displacement": 0,
        "actor_details": {},
        "acceptance_thresholds_m": {
            "actor": {
                "max_median_displacement_m": float(actor_max_median_displacement_m),
                "max_p90_displacement_m": float(actor_max_p90_displacement_m),
                "max_displacement_m": float(actor_max_displacement_m),
            },
            "ego": {
                "max_median_displacement_m": float(ego_max_median_displacement_m),
                "max_p90_displacement_m": float(ego_max_p90_displacement_m),
                "max_displacement_m": float(ego_max_displacement_m),
            },
        },
    }

    if not enabled:
        report["reason"] = "disabled_by_flag"
        print("[GRP-ALIGN] Disabled by flag — skipping trajectory alignment.")
        return vehicles, vehicle_times, ego_trajs, ego_times, report

    # Coordinate transform parameters (V2XPNP → CARLA is the inverse)
    scale = float(align_cfg.get("scale", 1.0))
    theta_deg = float(align_cfg.get("theta_deg", 0.0))
    tx = float(align_cfg.get("tx", 0.0))
    ty = float(align_cfg.get("ty", 0.0))
    flip_y = bool(align_cfg.get("flip_y", False))
    inv_scale = 1.0 / scale if abs(scale) > 1e-12 else 1.0

    def v2x_to_carla(x: float, y: float) -> Tuple[float, float]:
        cx, cy = invert_se2((x, y), theta_deg, tx, ty, flip_y=flip_y)
        return cx * inv_scale, cy * inv_scale

    def carla_to_v2x(x: float, y: float) -> Tuple[float, float]:
        sx, sy = float(x) * scale, float(y) * scale
        return apply_se2((sx, sy), theta_deg, tx, ty, flip_y=flip_y)

    def yaw_v2x_to_carla(yaw_v2x: float) -> float:
        """Transform yaw from V2XPNP frame to CARLA frame."""
        adjusted = float(yaw_v2x) - float(theta_deg)
        if flip_y:
            adjusted = -adjusted
        return _normalize_yaw_deg(adjusted)

    def yaw_carla_to_v2x(yaw_carla: float) -> float:
        """Transform yaw from CARLA frame to V2XPNP frame."""
        adjusted = float(yaw_carla)
        if flip_y:
            adjusted = -adjusted
        return _normalize_yaw_deg(adjusted + float(theta_deg))

    def _is_ego_label(label: str) -> bool:
        return str(label).startswith("ego_")

    actor_accept_limits = {
        "max_median_displacement_m": max(0.0, float(actor_max_median_displacement_m)),
        "max_p90_displacement_m": max(0.0, float(actor_max_p90_displacement_m)),
        "max_displacement_m": max(0.0, float(actor_max_displacement_m)),
    }
    ego_accept_limits = {
        "max_median_displacement_m": max(0.0, float(ego_max_median_displacement_m)),
        "max_p90_displacement_m": max(0.0, float(ego_max_p90_displacement_m)),
        "max_displacement_m": max(0.0, float(ego_max_displacement_m)),
    }

    # --- Connect to CARLA ---
    print(f"[GRP-ALIGN] Connecting to CARLA at {carla_host}:{carla_port} ...")
    try:
        carla, GlobalRoutePlanner, GlobalRoutePlannerDAO = _ensure_carla_grp()
        client = carla.Client(carla_host, int(carla_port))
        client.set_timeout(30.0)
        world = client.get_world()

        current_map = world.get_map().name
        current_map_base = current_map.split("/")[-1] if "/" in current_map else current_map
        if current_map_base != carla_map_name:
            print(f"[GRP-ALIGN] Loading map '{carla_map_name}' (current: {current_map_base}) ...")
            world = client.load_world(carla_map_name)

        carla_map = world.get_map()
        print(f"[GRP-ALIGN] Connected — map: {carla_map.name}")
    except Exception as exc:
        print(f"[GRP-ALIGN] CARLA connection failed: {exc}")
        print("[GRP-ALIGN] Continuing without GRP alignment.")
        report["reason"] = f"carla_connection_failed: {exc}"
        return vehicles, vehicle_times, ego_trajs, ego_times, report

    # --- Set up GRP ---
    try:
        grp_init_params = list(_inspect.signature(GlobalRoutePlanner.__init__).parameters.values())[1:]
        if len(grp_init_params) >= 2 and grp_init_params[0].name != "dao":
            grp = GlobalRoutePlanner(carla_map, sampling_resolution)
        elif GlobalRoutePlannerDAO is not None:
            grp = GlobalRoutePlanner(GlobalRoutePlannerDAO(carla_map, sampling_resolution))
        else:
            grp = GlobalRoutePlanner(carla_map, sampling_resolution)
        if hasattr(grp, "setup"):
            grp.setup()
        print(f"[GRP-ALIGN] GlobalRoutePlanner ready (resolution={sampling_resolution}m).")
    except Exception as exc:
        print(f"[GRP-ALIGN] GRP setup failed: {exc}")
        report["reason"] = f"grp_setup_failed: {exc}"
        return vehicles, vehicle_times, ego_trajs, ego_times, report

    # --- Helper: align a single trajectory ---
    def _align_one(
        traj: List[Waypoint],
        times: List[float],
        label: str,
    ) -> Tuple[Optional[List[Waypoint]], Optional[List[float]], Dict[str, object]]:
        """Align one trajectory. Returns (new_traj, new_times, detail) or (None, None, detail)."""
        detail: Dict[str, object] = {"label": label, "original_len": len(traj)}

        if len(traj) < 2:
            detail["action"] = "skipped_short"
            return None, None, detail

        # Transform to CARLA coords
        wps_carla = []
        for wp in traj:
            cx, cy = v2x_to_carla(float(wp.x), float(wp.y))
            cyaw = yaw_v2x_to_carla(float(wp.yaw))
            wps_carla.append({"x": cx, "y": cy, "z": float(wp.z), "yaw": cyaw})

        ts = list(times) if times else [float(i) * default_dt for i in range(len(traj))]

        try:
            aligned = _grp_refine_trajectory_timing_preserving(
                carla_map=carla_map,
                grp=grp,
                waypoints_carla=wps_carla,
                timestamps=ts,
                radius=snap_radius,
                k=snap_k,
                heading_thresh=heading_thresh,
                lane_change_penalty=lane_change_penalty,
            )
        except Exception as exc:
            detail["action"] = f"alignment_failed: {exc}"
            return None, None, detail

        if not aligned or not any(r.get("grp_aligned", False) for r in aligned):
            detail["action"] = "alignment_produced_no_result"
            return None, None, detail

        # Transform back to V2XPNP coords
        new_traj = []
        new_times = []
        for r in aligned:
            vx, vy = carla_to_v2x(float(r["x"]), float(r["y"]))
            vyaw = yaw_carla_to_v2x(float(r["yaw"]))
            new_traj.append(Waypoint(x=vx, y=vy, z=float(r["z"]), yaw=vyaw))
            new_times.append(float(r["t"]))

        detail["action"] = "aligned"
        detail["aligned_len"] = len(new_traj)

        # Compute displacement stats
        orig_xy = np.array([[float(wp.x), float(wp.y)] for wp in traj[:len(new_traj)]])
        new_xy = np.array([[float(wp.x), float(wp.y)] for wp in new_traj])
        min_len = min(orig_xy.shape[0], new_xy.shape[0])
        if min_len > 0:
            disps = np.sqrt(np.sum((orig_xy[:min_len] - new_xy[:min_len]) ** 2, axis=1))
            detail["median_displacement_m"] = float(np.median(disps))
            detail["p90_displacement_m"] = float(np.quantile(disps, 0.9))
            detail["max_displacement_m"] = float(np.max(disps))
            limits = ego_accept_limits if _is_ego_label(label) else actor_accept_limits
            med_disp = float(detail["median_displacement_m"])
            p90_disp = float(detail["p90_displacement_m"])
            max_disp = float(detail["max_displacement_m"])
            if (
                med_disp > float(limits["max_median_displacement_m"])
                or p90_disp > float(limits["max_p90_displacement_m"])
                or max_disp > float(limits["max_displacement_m"])
            ):
                detail["action"] = "rejected_excessive_displacement"
                detail["rejection_thresholds_m"] = dict(limits)
                return None, None, detail

        return new_traj, new_times, detail

    # --- Process actor vehicles (two-pass) ---
    #
    # Pass 1: classify every actor as skip / stationary / moving so we can
    #         build a spatial index of moving-vehicle paths *before* we
    #         decide where to freeze parked vehicles.
    # Pass 2: freeze stationary vehicles (with outer-lane bias) and GRP-
    #         align moving ones.
    # ------------------------------------------------------------------
    actor_vids = sorted(vehicles.keys())
    n_actors = len(actor_vids)
    print(f"[GRP-ALIGN] Processing {n_actors} actor trajectories ...")

    # -- Pass 1: classify ------------------------------------------------
    _cls_skip: list = []       # (vid, reason_str)
    _cls_stationary: list = [] # (vid, stat_reason, stat_stats)
    _cls_moving: list = []     # vid

    for vid in actor_vids:
        traj = vehicles[vid]
        times = vehicle_times.get(vid, [])
        meta = dict(obj_info.get(vid, {}))
        obj_type = str(meta.get("obj_type") or "npc")

        if not is_vehicle_type(obj_type):
            _cls_skip.append((vid, "not_vehicle"))
            continue
        if _is_pedestrian_type(obj_type) or _is_cyclist_type(obj_type):
            _cls_skip.append((vid, f"non_motor ({obj_type})"))
            continue

        is_stat, stat_reason, stat_stats = _is_actor_confidently_stationary(
            traj, times, default_dt, parked_vehicle_cfg,
        )
        if is_stat:
            _cls_stationary.append((vid, stat_reason, stat_stats))
        else:
            _cls_moving.append(vid)

    stationary_vids = {v for v, _, _ in _cls_stationary}
    print(
        f"[GRP-ALIGN]   Classification: {len(_cls_skip)} skipped, "
        f"{len(_cls_stationary)} stationary, {len(_cls_moving)} moving"
    )

    # -- Build moving-vehicle spatial index (CARLA coords) ----------------
    # We include ego trajectories as well — a parked car should not be
    # pushed into the ego's path either.
    OUTER_LANE_CELL_SIZE = 2.0
    OBSTRUCTION_CHECK_RADIUS = 4.0  # Distance to check for parked vehicle obstructions
    
    moving_cells = _build_moving_vehicle_spatial_set(
        vehicles, stationary_vids, v2x_to_carla,
        cell_size=OUTER_LANE_CELL_SIZE,
    )
    
    # Build separate ego-specific cell set for priority obstruction detection
    ego_cells: set = set()
    inv_cs = 1.0 / OUTER_LANE_CELL_SIZE
    for etraj in ego_trajs:
        for wp in etraj:
            cx_e, cy_e = v2x_to_carla(float(wp.x), float(wp.y))
            ego_cells.add((
                int(math.floor(cx_e * inv_cs)),
                int(math.floor(cy_e * inv_cs)),
            ))
    # Also add ego cells to moving_cells
    moving_cells.update(ego_cells)

    # -- Pass 2a: handle skipped actors ----------------------------------
    for vid, reason in _cls_skip:
        report["actors_skipped_short"] = int(report["actors_skipped_short"]) + 1
        if "non_motor" in reason:
            report["actor_details"][int(vid)] = {
                "action": f"skipped_{reason}",
                "original_len": len(vehicles[vid]),
            }

    # -- Pass 2b: freeze stationary actors (with outer-lane bias) ---------
    # Track parked vehicle positions for obstruction removal
    parked_positions_carla: Dict[int, Tuple[float, float]] = {}  # vid -> (carla_x, carla_y)
    
    n_outer_biased = 0
    for vid, stat_reason, stat_stats in _progress_bar(
        _cls_stationary,
        total=len(_cls_stationary),
        desc="[GRP-ALIGN] Freezing parked",
    ):
        traj = vehicles[vid]
        report["actors_processed"] = int(report["actors_processed"]) + 1

        med_x = float(np.median([float(wp.x) for wp in traj]))
        med_y = float(np.median([float(wp.y) for wp in traj]))
        med_z = float(np.median([float(wp.z) for wp in traj]))
        med_yaw = float(traj[len(traj) // 2].yaw)
        cx, cy = v2x_to_carla(med_x, med_y)

        outer_bias_reason = "no_snap"
        try:
            carla_loc = carla.Location(x=float(cx), y=float(cy), z=float(med_z))
            snapped_wp = carla_map.get_waypoint(
                carla_loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if snapped_wp is not None:
                # --- Outer-lane bias ---
                final_wp, did_bias, outer_bias_reason = _try_bias_parked_to_outer_lane(
                    snapped_wp,
                    carla_map,
                    carla,
                    moving_cells,
                    cell_size=OUTER_LANE_CELL_SIZE,
                    max_lateral_m=4.0,
                    interference_radius=3.0,
                )
                if did_bias:
                    n_outer_biased += 1

                floc = final_wp.transform.location
                sx, sy = carla_to_v2x(float(floc.x), float(floc.y))
                syaw = yaw_carla_to_v2x(float(final_wp.transform.rotation.yaw))
                frozen = Waypoint(x=sx, y=sy, z=float(floc.z), yaw=syaw)
                # Track CARLA position for obstruction checking
                parked_positions_carla[vid] = (float(floc.x), float(floc.y))
            else:
                frozen = Waypoint(x=med_x, y=med_y, z=med_z, yaw=med_yaw)
                # Track CARLA position from original coords
                parked_positions_carla[vid] = (cx, cy)
        except Exception:
            frozen = Waypoint(x=med_x, y=med_y, z=med_z, yaw=med_yaw)
            parked_positions_carla[vid] = (cx, cy)

        vehicles[vid] = [frozen] * len(traj)
        report["actors_skipped_stationary"] = int(report["actors_skipped_stationary"]) + 1
        report["actor_details"][int(vid)] = {
            "action": f"frozen_stationary ({stat_reason})",
            "original_len": len(traj),
            "frozen_x": float(frozen.x),
            "frozen_y": float(frozen.y),
            "outer_lane_bias": outer_bias_reason,
        }

    if n_outer_biased:
        print(f"[GRP-ALIGN]   Outer-lane bias applied to {n_outer_biased}/{len(_cls_stationary)} parked vehicles")

    # -- Pass 2b-ii: Remove parked vehicles that obstruct moving actors or ego --
    obstructs_ego, obstructs_actor = _find_obstructing_parked_vehicles(
        parked_positions_carla,
        moving_cells,
        ego_cells,
        cell_size=OUTER_LANE_CELL_SIZE,
        obstruction_radius=OBSTRUCTION_CHECK_RADIUS,
    )
    
    # Initialize removal tracking in report
    report["parked_removed_obstructing_ego"] = []
    report["parked_removed_obstructing_actor"] = []
    
    # Remove parked vehicles that obstruct ego (highest priority)
    for vid in obstructs_ego:
        if vid in vehicles:
            print(f"[GRP-ALIGN]   REMOVED actor_{vid}: parked vehicle obstructing EGO path")
            del vehicles[vid]
            if vid in vehicle_times:
                del vehicle_times[vid]
            # Update report
            if int(vid) in report["actor_details"]:
                report["actor_details"][int(vid)]["action"] = "REMOVED_obstruct_ego"
            report["parked_removed_obstructing_ego"].append(int(vid))
    
    # Remove parked vehicles that obstruct other moving actors
    for vid in obstructs_actor:
        if vid in vehicles:
            print(f"[GRP-ALIGN]   REMOVED actor_{vid}: parked vehicle obstructing moving actor path")
            del vehicles[vid]
            if vid in vehicle_times:
                del vehicle_times[vid]
            # Update report
            if int(vid) in report["actor_details"]:
                report["actor_details"][int(vid)]["action"] = "REMOVED_obstruct_actor"
            report["parked_removed_obstructing_actor"].append(int(vid))
    
    total_removed = len(obstructs_ego) + len(obstructs_actor)
    if total_removed > 0:
        print(f"[GRP-ALIGN]   Removed {total_removed} obstructing parked vehicles "
              f"({len(obstructs_ego)} blocking ego, {len(obstructs_actor)} blocking actors)")

    # -- Pass 2c: GRP-align moving actors ---------------------------------
    for vid in _progress_bar(_cls_moving, total=len(_cls_moving), desc="[GRP-ALIGN] Aligning"):
        traj = vehicles[vid]
        times = vehicle_times.get(vid, [])
        report["actors_processed"] = int(report["actors_processed"]) + 1

        new_traj, new_times, detail = _align_one(traj, times, f"actor_{vid}")
        report["actor_details"][int(vid)] = detail

        if new_traj is not None and new_times is not None:
            vehicles[vid] = new_traj
            vehicle_times[vid] = new_times
            report["actors_aligned"] = int(report["actors_aligned"]) + 1
        else:
            if detail.get("action") == "rejected_excessive_displacement":
                report["actors_rejected_displacement"] = int(report["actors_rejected_displacement"]) + 1
            else:
                report["actors_failed"] = int(report["actors_failed"]) + 1

    # --- Process ego trajectories ---
    n_egos = len(ego_trajs)
    print(f"[GRP-ALIGN] Processing {n_egos} ego trajectories ...")

    for ego_idx in range(n_egos):
        traj = ego_trajs[ego_idx]
        times = ego_times[ego_idx] if ego_idx < len(ego_times) else []

        report["egos_processed"] = int(report["egos_processed"]) + 1

        # Check if ego is stationary (unlikely but possible)
        is_stat, stat_reason, _ = _is_actor_confidently_stationary(
            traj, times, default_dt, parked_vehicle_cfg,
        )
        if is_stat:
            report["actor_details"][f"ego_{ego_idx}"] = {  # type: ignore
                "action": f"skipped_stationary ({stat_reason})",
                "original_len": len(traj),
            }
            continue

        new_traj, new_times, detail = _align_one(traj, times, f"ego_{ego_idx}")
        report["actor_details"][f"ego_{ego_idx}"] = detail  # type: ignore

        if new_traj is not None and new_times is not None:
            ego_trajs[ego_idx] = new_traj
            if ego_idx < len(ego_times):
                ego_times[ego_idx] = new_times
            report["egos_aligned"] = int(report["egos_aligned"]) + 1
        else:
            if detail.get("action") == "rejected_excessive_displacement":
                report["egos_rejected_displacement"] = int(report["egos_rejected_displacement"]) + 1
            else:
                report["egos_failed"] = int(report["egos_failed"]) + 1

    # Summary
    print(
        f"[GRP-ALIGN] Done — "
        f"actors: {report['actors_aligned']} aligned, "
        f"{report['actors_skipped_stationary']} stationary, "
        f"{report['actors_rejected_displacement']} rejected, "
        f"{report['actors_failed']} failed | "
        f"egos: {report['egos_aligned']} aligned, "
        f"{report['egos_rejected_displacement']} rejected, "
        f"{report['egos_failed']} failed"
    )

    return vehicles, vehicle_times, ego_trajs, ego_times, report


def _suppress_short_lane_runs(
    lanes: List[int],
    max_short_run: int,
    endpoint_short_run: int,
) -> List[int]:
    if not lanes:
        return []
    out = [int(v) for v in lanes]

    def _build_runs(vals: List[int]) -> List[Tuple[int, int, int]]:
        runs: List[Tuple[int, int, int]] = []
        s = 0
        cur = int(vals[0])
        for i in range(1, len(vals)):
            if int(vals[i]) != cur:
                runs.append((cur, s, i - 1))
                s = i
                cur = int(vals[i])
        runs.append((cur, s, len(vals) - 1))
        return runs

    changed = True
    guard_iters = 0
    while changed and guard_iters < 6:
        guard_iters += 1
        changed = False
        runs = _build_runs(out)
        for i, (lane_id, s, e) in enumerate(runs):
            run_len = int(e - s + 1)
            if lane_id < 0:
                continue
            if 0 < i < len(runs) - 1:
                prev_lane = int(runs[i - 1][0])
                next_lane = int(runs[i + 1][0])
                if prev_lane == next_lane and prev_lane >= 0 and run_len <= int(max_short_run):
                    for j in range(s, e + 1):
                        out[j] = prev_lane
                    changed = True
            elif i == 0 and len(runs) > 1 and run_len <= int(endpoint_short_run):
                next_lane = int(runs[i + 1][0])
                if next_lane >= 0:
                    for j in range(s, e + 1):
                        out[j] = next_lane
                    changed = True
            elif i == len(runs) - 1 and len(runs) > 1 and run_len <= int(endpoint_short_run):
                prev_lane = int(runs[i - 1][0])
                if prev_lane >= 0:
                    for j in range(s, e + 1):
                        out[j] = prev_lane
                    changed = True
    return out


def _stabilize_lane_sequence(
    traj: Sequence[Waypoint],
    raw_candidates: Sequence[Sequence[Dict[str, object]]],
    matcher: LaneMatcher,
    cfg: Dict[str, float],
) -> List[int]:
    n = int(len(traj))
    if n <= 0:
        return []

    raw_lane: List[int] = []
    raw_dist: List[float] = []
    for cands in raw_candidates:
        if cands:
            raw_lane.append(int(cands[0]["lane_index"]))
            raw_dist.append(float(cands[0]["dist"]))
        else:
            raw_lane.append(-1)
            raw_dist.append(float("inf"))

    confirm_window = max(2, int(cfg.get("confirm_window", 5)))
    confirm_votes = max(2, int(cfg.get("confirm_votes", 3)))
    cooldown_frames = max(0, int(cfg.get("cooldown_frames", 3)))
    endpoint_guard = max(0, int(cfg.get("endpoint_guard_frames", 4)))
    endpoint_extra_votes = max(0, int(cfg.get("endpoint_extra_votes", 1)))
    min_improve_m = max(0.0, float(cfg.get("min_improvement_m", 0.2)))
    keep_lane_max_dist = max(0.0, float(cfg.get("keep_lane_max_dist", 3.0)))
    short_run_max = max(0, int(cfg.get("short_run_max", 2)))
    endpoint_short_run = max(0, int(cfg.get("endpoint_short_run", 2)))

    anchor_window = min(n, max(confirm_window + endpoint_guard, 6))
    start_anchor = _dominant_lane(raw_lane[:anchor_window])
    end_anchor = _dominant_lane(raw_lane[max(0, n - anchor_window):])

    current_lane = start_anchor if start_anchor >= 0 else next((lid for lid in raw_lane if lid >= 0), -1)
    cooldown = 0
    stable_lane: List[int] = []
    for i in range(n):
        raw_lid = int(raw_lane[i])
        chosen = int(current_lane)

        if chosen < 0:
            if raw_lid >= 0:
                chosen = raw_lid
            elif i < endpoint_guard and start_anchor >= 0:
                chosen = int(start_anchor)

        if chosen >= 0 and raw_lid >= 0 and raw_lid != chosen:
            required_votes = int(confirm_votes)
            if i < endpoint_guard or i >= max(0, n - endpoint_guard):
                required_votes += int(endpoint_extra_votes)

            w0 = i
            w1 = min(n, i + confirm_window)
            support_raw = sum(1 for j in range(w0, w1) if int(raw_lane[j]) == raw_lid)
            if i >= max(0, n - endpoint_guard):
                # Near tail, incorporate short look-back support too.
                wb0 = max(0, i - confirm_window + 1)
                support_raw = max(support_raw, sum(1 for j in range(wb0, i + 1) if int(raw_lane[j]) == raw_lid))

            cur_proj = matcher.project_to_lane(chosen, float(traj[i].x), float(traj[i].y), float(traj[i].z)) if chosen >= 0 else None
            cur_dist = float(cur_proj["dist"]) if cur_proj is not None else float("inf")
            raw_d = float(raw_dist[i])
            dist_improve = float(cur_dist - raw_d)

            if support_raw >= required_votes:
                support_strong = support_raw >= (required_votes + 1)
                if cooldown > 0 and not support_strong:
                    pass
                else:
                    if support_strong or dist_improve >= min_improve_m or cur_dist > keep_lane_max_dist:
                        chosen = raw_lid
                        cooldown = int(cooldown_frames)

        if i < endpoint_guard and start_anchor >= 0 and chosen != int(start_anchor):
            # Suppress startup lane jumps unless strongly supported.
            w1 = min(n, i + confirm_window + 1)
            support_chosen = sum(1 for j in range(i, w1) if int(raw_lane[j]) == int(chosen))
            if support_chosen < (confirm_votes + endpoint_extra_votes + 1):
                chosen = int(start_anchor)
                cooldown = max(cooldown, int(cooldown_frames))

        if i >= max(0, n - endpoint_guard) and end_anchor >= 0 and chosen != int(end_anchor):
            # Suppress tail-end flips unless they dominate tail support.
            w0 = max(0, n - max(confirm_window + endpoint_guard, 6))
            support_chosen = sum(1 for j in range(w0, n) if int(raw_lane[j]) == int(chosen))
            support_end = sum(1 for j in range(w0, n) if int(raw_lane[j]) == int(end_anchor))
            if support_chosen <= support_end:
                chosen = int(end_anchor)
                cooldown = max(cooldown, int(cooldown_frames))

        stable_lane.append(int(chosen))
        current_lane = int(chosen)
        if cooldown > 0:
            cooldown -= 1

    stable_lane = _suppress_short_lane_runs(
        stable_lane,
        max_short_run=int(short_run_max),
        endpoint_short_run=int(endpoint_short_run),
    )
    return stable_lane


def _build_track_frames(
    traj: Sequence[Waypoint],
    times: Sequence[float] | None,
    matcher: LaneMatcher,
    snap_to_map: bool,
    lane_change_cfg: Optional[Dict[str, object]] = None,
    lane_policy: Optional[Dict[str, object]] = None,
    skip_snap: bool = False,
) -> List[Dict[str, object]]:
    # --- Pedestrians / actors that should never be lane-snapped ---
    if skip_snap:
        frames: List[Dict[str, object]] = []
        for i, wp in enumerate(traj):
            t = float(times[i]) if (times is not None and i < len(times)) else float(i)
            frames.append(
                {
                    "t": float(round(t, 6)),
                    "rx": float(wp.x),
                    "ry": float(wp.y),
                    "rz": float(wp.z),
                    "ryaw": float(wp.yaw),
                    "mx": float(wp.x),
                    "my": float(wp.y),
                    "mz": float(wp.z),
                    "myaw": float(wp.yaw),
                    "x": float(wp.x),
                    "y": float(wp.y),
                    "z": float(wp.z),
                    "yaw": float(wp.yaw),
                    "li": -1,
                    "ld": float("inf"),
                    "li_raw": -1,
                    "ld_raw": float("inf"),
                }
            )
        return frames

    cfg = dict(lane_change_cfg or {})
    lane_filter_enabled = bool(cfg.get("enabled", True))
    lane_top_k = max(2, int(cfg.get("lane_top_k", 8)))
    policy = dict(lane_policy or {})
    allowed_lane_types_raw = policy.get("allowed_lane_types")
    allowed_lane_types: Optional[set[str]] = None
    if allowed_lane_types_raw is not None:
        allowed_lane_types = _parse_lane_type_set(allowed_lane_types_raw, fallback=[])
    stationary_lane_types = _parse_lane_type_set(policy.get("stationary_when_lane_types"), fallback=[])

    raw_candidates: List[List[Dict[str, object]]] = []
    raw_best_unfiltered: List[Optional[Dict[str, object]]] = []
    for wp in traj:
        nearest = matcher.match_candidates(float(wp.x), float(wp.y), float(wp.z), lane_top_k=lane_top_k)
        raw_best_unfiltered.append(nearest[0] if nearest else None)
        filtered = list(nearest)
        if allowed_lane_types is not None:
            filtered = [c for c in nearest if str(c.get("lane_type", "")) in allowed_lane_types]
            if not filtered:
                wide_k = max(int(lane_top_k) * 8, 48)
                if wide_k > int(lane_top_k):
                    wider = matcher.match_candidates(float(wp.x), float(wp.y), float(wp.z), lane_top_k=wide_k)
                    filtered = [c for c in wider if str(c.get("lane_type", "")) in allowed_lane_types]
        raw_candidates.append(filtered)

    stable_lane_idx: List[int]
    if lane_filter_enabled:
        stable_lane_idx = _stabilize_lane_sequence(
            traj=traj,
            raw_candidates=raw_candidates,
            matcher=matcher,
            cfg=cfg,
        )
    else:
        stable_lane_idx = [int(cands[0]["lane_index"]) if cands else -1 for cands in raw_candidates]

    force_stationary = False
    stationary_anchor: Optional[Dict[str, object]] = None
    stationary_lane_idx = -1
    if stationary_lane_types:
        lane_type_seq: List[str] = []
        for lane_idx in stable_lane_idx:
            if int(lane_idx) < 0:
                lane_type_seq.append("")
            else:
                lane_type_seq.append(str(matcher.map_data.lanes[int(lane_idx)].lane_type))
        if any(lt in stationary_lane_types for lt in lane_type_seq):
            preferred = [
                int(li)
                for li in stable_lane_idx
                if int(li) >= 0 and str(matcher.map_data.lanes[int(li)].lane_type) in stationary_lane_types
            ]
            if not preferred:
                preferred = [int(li) for li in stable_lane_idx if int(li) >= 0]
            stationary_lane_idx = _dominant_lane(preferred)
            if stationary_lane_idx >= 0:
                med_x = float(np.median([float(wp.x) for wp in traj])) if traj else 0.0
                med_y = float(np.median([float(wp.y) for wp in traj])) if traj else 0.0
                med_z = float(np.median([float(wp.z) for wp in traj])) if traj else 0.0
                stationary_anchor = matcher.project_to_lane(stationary_lane_idx, med_x, med_y, med_z)
                if stationary_anchor is None and traj:
                    wp0 = traj[0]
                    stationary_anchor = matcher.project_to_lane(
                        stationary_lane_idx,
                        float(wp0.x),
                        float(wp0.y),
                        float(wp0.z),
                    )
                if stationary_anchor is not None:
                    force_stationary = True
                    stable_lane_idx = [int(stationary_lane_idx) for _ in stable_lane_idx]

    frames: List[Dict[str, object]] = []
    for i, wp in enumerate(traj):
        t = float(times[i]) if (times is not None and i < len(times)) else float(i)
        raw_match = raw_best_unfiltered[i] if i < len(raw_best_unfiltered) else None
        chosen_lane = int(stable_lane_idx[i]) if i < len(stable_lane_idx) else -1

        if force_stationary and stationary_anchor is not None and int(stationary_lane_idx) >= 0:
            map_x = float(stationary_anchor["x"])
            map_y = float(stationary_anchor["y"])
            map_z = float(stationary_anchor["z"])
            map_yaw = float(stationary_anchor["yaw"])
            lane_idx = int(stationary_lane_idx)
            cur_proj = matcher.project_to_lane(int(stationary_lane_idx), float(wp.x), float(wp.y), float(wp.z))
            lane_dist = float(cur_proj["dist"]) if cur_proj is not None else float(stationary_anchor.get("dist", 0.0))
        else:
            match: Optional[Dict[str, object]] = None
            if chosen_lane >= 0:
                for cand in raw_candidates[i]:
                    if int(cand.get("lane_index", -1)) == chosen_lane:
                        match = cand
                        break
                if match is None:
                    match = matcher.project_to_lane(chosen_lane, float(wp.x), float(wp.y), float(wp.z))
                    if (
                        match is not None
                        and allowed_lane_types is not None
                        and str(match.get("lane_type", "")) not in allowed_lane_types
                    ):
                        match = None
            if match is None:
                if (
                    raw_match is not None
                    and (
                        allowed_lane_types is None
                        or str(raw_match.get("lane_type", "")) in allowed_lane_types
                    )
                ):
                    match = raw_match

            if match is None:
                map_x, map_y, map_z, map_yaw = float(wp.x), float(wp.y), float(wp.z), float(wp.yaw)
                lane_idx, lane_dist = -1, float("inf")
            else:
                map_x = float(match["x"])
                map_y = float(match["y"])
                map_z = float(match["z"])
                map_yaw = float(match["yaw"])
                lane_idx = int(match["lane_index"])
                lane_dist = float(match["dist"])

        if lane_idx < 0 and raw_match is None:
            map_x, map_y, map_z, map_yaw = float(wp.x), float(wp.y), float(wp.z), float(wp.yaw)
            lane_idx, lane_dist = -1, float("inf")

        display_x = map_x if snap_to_map else float(wp.x)
        display_y = map_y if snap_to_map else float(wp.y)
        display_z = map_z if snap_to_map else float(wp.z)
        display_yaw = map_yaw if snap_to_map else float(wp.yaw)
        frames.append(
            {
                "t": float(round(t, 6)),
                "rx": float(wp.x),
                "ry": float(wp.y),
                "rz": float(wp.z),
                "ryaw": float(wp.yaw),
                "mx": float(map_x),
                "my": float(map_y),
                "mz": float(map_z),
                "myaw": float(map_yaw),
                "x": float(display_x),
                "y": float(display_y),
                "z": float(display_z),
                "yaw": float(display_yaw),
                "li": int(lane_idx),
                "ld": float(lane_dist),
                "li_raw": int(raw_match["lane_index"]) if raw_match is not None else -1,
                "ld_raw": float(raw_match["dist"]) if raw_match is not None else float("inf"),
            }
        )
    return frames


def _trajectory_path_length(traj: Sequence[Waypoint]) -> float:
    if len(traj) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(traj, traj[1:]):
        total += math.hypot(float(b.x) - float(a.x), float(b.y) - float(a.y))
    return float(total)


def _downsample_line(points: np.ndarray, max_points: int) -> List[List[float]]:
    if points.shape[0] == 0:
        return []
    if max_points <= 0 or points.shape[0] <= max_points:
        return [[float(p[0]), float(p[1]), float(p[2])] for p in points]
    idx = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int32)
    sampled = points[idx]
    if idx[-1] != points.shape[0] - 1:
        sampled = np.vstack([sampled, points[-1:]])
    return [[float(p[0]), float(p[1]), float(p[2])] for p in sampled]


def _collect_timeline(ego_tracks: Sequence[Dict[str, object]], actor_tracks: Sequence[Dict[str, object]]) -> List[float]:
    times: set[float] = set()
    for track in list(ego_tracks) + list(actor_tracks):
        for fr in track.get("frames", []):
            try:
                times.add(float(fr["t"]))
            except Exception:
                continue
    if not times:
        return [0.0]
    return sorted(times)


def _refresh_payload_timeline_for_carla_exec(payload: Dict[str, object]) -> None:
    """Refresh timeline using exported replay paths when available."""
    ego_tracks = payload.get("ego_tracks", []) if isinstance(payload, dict) else []
    actor_tracks = payload.get("actor_tracks", []) if isinstance(payload, dict) else []
    times: set[float] = set()

    for track in ego_tracks:
        if not isinstance(track, dict):
            continue
        for fr in track.get("frames", []):
            if not isinstance(fr, dict):
                continue
            t = _safe_float(fr.get("t"), float("nan"))
            if math.isfinite(t):
                times.add(float(round(t, 6)))

    for track in actor_tracks:
        if not isinstance(track, dict):
            continue
        role = str(track.get("role", "")).strip().lower()
        is_walker_like = role in {"walker", "pedestrian", "cyclist", "bicycle"}
        control_mode = str(track.get("carla_control_mode", "")).strip().lower()
        use_exec = (
            control_mode == "replay"
            and not is_walker_like
            and isinstance(track.get("carla_exec_frames"), list)
            and len(track.get("carla_exec_frames", [])) > 0
        )
        src_frames = track.get("carla_exec_frames") if use_exec else track.get("frames", [])
        if not isinstance(src_frames, list):
            continue
        for fr in src_frames:
            if not isinstance(fr, dict):
                continue
            t = _safe_float(fr.get("t"), float("nan"))
            if math.isfinite(t):
                times.add(float(round(t, 6)))

    payload["timeline"] = sorted(times) if times else [0.0]


def _polyline_xy_from_points(points: object) -> np.ndarray:
    if not isinstance(points, (list, tuple)):
        return np.zeros((0, 2), dtype=np.float64)
    out: List[Tuple[float, float]] = []
    for p in points:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            x = _safe_float(p[0], float("nan"))
            y = _safe_float(p[1], float("nan"))
            if math.isfinite(x) and math.isfinite(y):
                out.append((float(x), float(y)))
    if len(out) < 2:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(out, dtype=np.float64)


def _polyline_cumlen_xy(poly_xy: np.ndarray) -> np.ndarray:
    n = int(poly_xy.shape[0])
    if n <= 0:
        return np.zeros((0,), dtype=np.float64)
    if n == 1:
        return np.zeros((1,), dtype=np.float64)
    seg = poly_xy[1:] - poly_xy[:-1]
    seg_len = np.sqrt(np.sum(seg * seg, axis=1))
    cum = np.zeros((n,), dtype=np.float64)
    cum[1:] = np.cumsum(seg_len)
    return cum


def _polyline_total_len(cumlen: np.ndarray) -> float:
    if cumlen.size <= 0:
        return 0.0
    return float(cumlen[-1])


def _yaw_abs_diff_deg(a: float, b: float) -> float:
    return abs(_normalize_yaw_deg(float(a) - float(b)))


def _project_point_to_polyline_xy(poly_xy: np.ndarray, cumlen: np.ndarray, x: float, y: float) -> Optional[Dict[str, float]]:
    if poly_xy.shape[0] <= 0:
        return None
    if poly_xy.shape[0] == 1:
        px = float(poly_xy[0, 0])
        py = float(poly_xy[0, 1])
        return {
            "x": px,
            "y": py,
            "yaw": 0.0,
            "dist": float(math.hypot(px - x, py - y)),
            "s": 0.0,
            "s_norm": 0.0,
            "segment_idx": 0.0,
        }

    p0 = poly_xy[:-1]
    p1 = poly_xy[1:]
    seg = p1 - p0
    seg_len2 = np.sum(seg * seg, axis=1)
    valid = seg_len2 > 1e-12
    if not np.any(valid):
        px = float(poly_xy[0, 0])
        py = float(poly_xy[0, 1])
        return {
            "x": px,
            "y": py,
            "yaw": 0.0,
            "dist": float(math.hypot(px - x, py - y)),
            "s": 0.0,
            "s_norm": 0.0,
            "segment_idx": 0.0,
        }

    xy = np.asarray([float(x), float(y)], dtype=np.float64)
    t = np.zeros((seg.shape[0],), dtype=np.float64)
    t[valid] = np.sum((xy - p0[valid]) * seg[valid], axis=1) / seg_len2[valid]
    t = np.clip(t, 0.0, 1.0)
    proj = p0 + seg * t[:, None]
    d2 = np.sum((proj - xy[None, :]) ** 2, axis=1)
    best = int(np.argmin(d2))
    best_t = float(t[best])
    seg_dx = float(seg[best, 0])
    seg_dy = float(seg[best, 1])
    seg_len = math.hypot(seg_dx, seg_dy)
    yaw = _normalize_yaw_deg(math.degrees(math.atan2(seg_dy, seg_dx))) if seg_len > 1e-9 else 0.0
    s_base = float(cumlen[best]) if cumlen.size > best else 0.0
    s = s_base + float(best_t) * float(seg_len)
    total_len = _polyline_total_len(cumlen)
    s_norm = (s / total_len) if total_len > 1e-9 else 0.0
    return {
        "x": float(proj[best, 0]),
        "y": float(proj[best, 1]),
        "yaw": float(yaw),
        "dist": float(math.sqrt(max(0.0, float(d2[best])))),
        "s": float(s),
        "s_norm": float(max(0.0, min(1.0, float(s_norm)))),
        "segment_idx": float(best + best_t),
    }


def _sample_polyline_at_s_xy(poly_xy: np.ndarray, cumlen: np.ndarray, s: float) -> Optional[Dict[str, float]]:
    n = int(poly_xy.shape[0])
    if n <= 0:
        return None
    if n == 1:
        return {
            "x": float(poly_xy[0, 0]),
            "y": float(poly_xy[0, 1]),
            "yaw": 0.0,
            "s": 0.0,
            "s_norm": 0.0,
            "segment_idx": 0.0,
        }
    total_len = _polyline_total_len(cumlen)
    if total_len <= 1e-9:
        return {
            "x": float(poly_xy[0, 0]),
            "y": float(poly_xy[0, 1]),
            "yaw": 0.0,
            "s": 0.0,
            "s_norm": 0.0,
            "segment_idx": 0.0,
        }
    ss = max(0.0, min(float(total_len), float(s)))
    idx = int(np.searchsorted(cumlen, ss, side="right") - 1)
    idx = max(0, min(idx, n - 2))
    s0 = float(cumlen[idx])
    s1 = float(cumlen[idx + 1])
    denom = max(1e-9, s1 - s0)
    t = max(0.0, min(1.0, (ss - s0) / denom))
    p0 = poly_xy[idx]
    p1 = poly_xy[idx + 1]
    x = float(p0[0] + t * (p1[0] - p0[0]))
    y = float(p0[1] + t * (p1[1] - p0[1]))
    yaw = _normalize_yaw_deg(math.degrees(math.atan2(float(p1[1] - p0[1]), float(p1[0] - p0[0]))))
    return {
        "x": x,
        "y": y,
        "yaw": float(yaw),
        "s": float(ss),
        "s_norm": float(ss / total_len),
        "segment_idx": float(idx + t),
    }


def _sample_polyline_at_ratio_xy(poly_xy: np.ndarray, cumlen: np.ndarray, ratio: float) -> Optional[Dict[str, float]]:
    total_len = _polyline_total_len(cumlen)
    r = max(0.0, min(1.0, float(ratio)))
    return _sample_polyline_at_s_xy(poly_xy, cumlen, r * total_len)


def _evaluate_lane_pair_metrics(v2_xy: np.ndarray, v2_cum: np.ndarray, carla_xy: np.ndarray, carla_cum: np.ndarray) -> Dict[str, float]:
    """Evaluate geometric similarity between a V2XPNP lane and a CARLA line.

    Returns a dict with individual metric values plus a composite *score*
    (lower = better).  A hard direction filter rejects anti-parallel
    candidates by setting score = inf when monotonic_ratio < 0.45.
    """
    _INF_METRICS: Dict[str, float] = {
        "score": float("inf"),
        "median_dist_m": float("inf"),
        "p90_dist_m": float("inf"),
        "mean_dist_m": float("inf"),
        "coverage_1m": 0.0,
        "coverage_2m": 0.0,
        "angle_median_deg": 180.0,
        "angle_p90_deg": 180.0,
        "monotonic_ratio": 0.0,
        "length_ratio": 0.0,
        "end_dist_m": float("inf"),
        "n_samples": 0.0,
    }
    v2_len = _polyline_total_len(v2_cum)
    carla_len = _polyline_total_len(carla_cum)
    if v2_xy.shape[0] < 2 or carla_xy.shape[0] < 2 or v2_len <= 1e-6 or carla_len <= 1e-6:
        return dict(_INF_METRICS)

    n_samples = int(max(14, min(84, round(v2_len / 1.6))))
    s_vals = np.linspace(0.0, float(v2_len), n_samples, dtype=np.float64)
    dists: List[float] = []
    angs: List[float] = []
    uvals: List[float] = []
    for sv in s_vals.tolist():
        sp = _sample_polyline_at_s_xy(v2_xy, v2_cum, float(sv))
        if sp is None:
            continue
        proj = _project_point_to_polyline_xy(carla_xy, carla_cum, float(sp["x"]), float(sp["y"]))
        if proj is None:
            continue
        dists.append(float(proj["dist"]))
        angs.append(float(_yaw_abs_diff_deg(float(sp["yaw"]), float(proj["yaw"]))))
        uvals.append(float(proj["s_norm"]))

    if not dists:
        return dict(_INF_METRICS)

    d_arr = np.asarray(dists, dtype=np.float64)
    a_arr = np.asarray(angs, dtype=np.float64)
    u_arr = np.asarray(uvals, dtype=np.float64)
    median_dist = float(np.quantile(d_arr, 0.5))
    p90_dist = float(np.quantile(d_arr, 0.9))
    mean_dist = float(np.mean(d_arr))
    coverage_1m = float(np.mean((d_arr <= 1.0).astype(np.float64)))
    coverage_2m = float(np.mean((d_arr <= 2.0).astype(np.float64)))
    angle_median = float(np.quantile(a_arr, 0.5))
    angle_p90 = float(np.quantile(a_arr, 0.9))
    mono = 1.0
    if u_arr.size >= 2:
        du = np.diff(u_arr)
        mono = float(np.mean((du >= -0.035).astype(np.float64)))
    length_ratio = float(min(v2_len, carla_len) / max(v2_len, carla_len))
    p_start = _project_point_to_polyline_xy(carla_xy, carla_cum, float(v2_xy[0, 0]), float(v2_xy[0, 1]))
    p_end = _project_point_to_polyline_xy(carla_xy, carla_cum, float(v2_xy[-1, 0]), float(v2_xy[-1, 1]))
    end_dist = float(0.5 * ((float(p_start["dist"]) if p_start else 4.0) + (float(p_end["dist"]) if p_end else 4.0)))

    # --- Hard direction filter ---
    # If the monotonic ratio is below threshold, the CARLA line is pointing
    # the wrong way or wraps around oddly.  Reject with infinite score.
    if mono < 0.45:
        out = dict(_INF_METRICS)
        out["monotonic_ratio"] = float(mono)
        out["angle_median_deg"] = float(angle_median)
        out["n_samples"] = float(len(dists))
        return out

    score = (
        1.00 * median_dist
        + 0.40 * p90_dist
        + 0.010 * angle_median
        + 0.90 * (1.0 - coverage_2m)
        + 0.55 * (1.0 - mono)
        + 0.30 * (1.0 - length_ratio)
        + 0.12 * end_dist
    )
    return {
        "score": float(score),
        "median_dist_m": float(median_dist),
        "p90_dist_m": float(p90_dist),
        "mean_dist_m": float(mean_dist),
        "coverage_1m": float(coverage_1m),
        "coverage_2m": float(coverage_2m),
        "angle_median_deg": float(angle_median),
        "angle_p90_deg": float(angle_p90),
        "monotonic_ratio": float(mono),
        "length_ratio": float(length_ratio),
        "end_dist_m": float(end_dist),
        "n_samples": float(len(dists)),
    }


def _quality_from_metrics(metrics: Dict[str, float]) -> str:
    med = float(metrics.get("median_dist_m", float("inf")))
    cov2 = float(metrics.get("coverage_2m", 0.0))
    ang = float(metrics.get("angle_median_deg", 180.0))
    mono = float(metrics.get("monotonic_ratio", 0.0))
    score = float(metrics.get("score", float("inf")))
    if med <= 1.25 and cov2 >= 0.85 and ang <= 25.0 and mono >= 0.85 and score <= 2.4:
        return "high"
    if med <= 2.2 and cov2 >= 0.55 and ang <= 55.0 and mono >= 0.55 and score <= 4.8:
        return "medium"
    if med <= 3.4 and cov2 >= 0.35 and score <= 7.0:
        return "low"
    return "poor"


# ---------------------------------------------------------------------------
# Connectivity / adjacency helpers for global lane correspondence
# ---------------------------------------------------------------------------

def _build_v2_connectivity_graph(
    v2_feats: Dict[int, Dict[str, object]],
    v2_lanes_raw: list,
) -> Dict[str, object]:
    """Build maps from V2XPNP lane index → connected lane indices.

    Uses *entry_lanes* / *exit_lanes* (lists of UID strings like "road_lane")
    stored in the raw lane dicts.

    Returns dict with:
      uid_to_index: {uid_str → lane_index}
      successors:   {lane_index → set of successor lane indices}
      predecessors: {lane_index → set of predecessor lane indices}
      adjacency:    {lane_index → set of adjacent lane indices (same road_id, |Δlane_id|=1)}
    """
    uid_to_index: Dict[str, int] = {}
    for lane in v2_lanes_raw:
        if not isinstance(lane, dict):
            continue
        li = _safe_int(lane.get("index"), -1)
        if li < 0:
            continue
        uid = f"{_safe_int(lane.get('road_id'), 0)}_{_safe_int(lane.get('lane_id'), 0)}"
        uid_to_index[uid] = int(li)

    successors: Dict[int, set] = {}
    predecessors: Dict[int, set] = {}
    for lane in v2_lanes_raw:
        if not isinstance(lane, dict):
            continue
        li = _safe_int(lane.get("index"), -1)
        if li < 0:
            continue
        for uid in (lane.get("exit_lanes") or []):
            si = uid_to_index.get(str(uid))
            if si is not None and si != li:
                successors.setdefault(li, set()).add(si)
                predecessors.setdefault(si, set()).add(li)
        for uid in (lane.get("entry_lanes") or []):
            pi = uid_to_index.get(str(uid))
            if pi is not None and pi != li:
                predecessors.setdefault(li, set()).add(pi)
                successors.setdefault(pi, set()).add(li)

    # Build spatial adjacency: same road_id, |lane_id difference| == 1
    road_lanes: Dict[int, List[Tuple[int, int]]] = {}  # road_id → [(lane_id, lane_index)]
    for li, lf in v2_feats.items():
        rid = int(lf.get("road_id", 0))
        lid = int(lf.get("lane_id", 0))
        road_lanes.setdefault(rid, []).append((lid, li))
    adjacency: Dict[int, set] = {}
    for rid, lanes in road_lanes.items():
        lid_to_idx = {lid: li for lid, li in lanes}
        for lid, li in lanes:
            for delta in (-1, 1):
                neighbor = lid_to_idx.get(lid + delta)
                if neighbor is not None and neighbor != li:
                    adjacency.setdefault(li, set()).add(neighbor)
                    adjacency.setdefault(neighbor, set()).add(li)

    return {
        "uid_to_index": uid_to_index,
        "successors": successors,
        "predecessors": predecessors,
        "adjacency": adjacency,
    }


def _carla_endpoint_proximity(
    carla_feats: Dict[int, Dict[str, object]],
    threshold_m: float = 3.0,
) -> Dict[int, set]:
    """For each CARLA line, find other CARLA lines whose start is close to
    this line's end (within *threshold_m* metres).  This approximates
    successor connectivity for unlabeled CARLA polylines."""
    endpoints: List[Tuple[int, float, float]] = []  # (line_index, end_x, end_y)
    startpoints: List[Tuple[int, float, float]] = []  # (line_index, start_x, start_y)
    for ci, cf in carla_feats.items():
        poly = cf["poly_xy"]
        endpoints.append((ci, float(poly[-1, 0]), float(poly[-1, 1])))
        startpoints.append((ci, float(poly[0, 0]), float(poly[0, 1])))

    if not startpoints:
        return {}

    start_xy = np.asarray([(s[1], s[2]) for s in startpoints], dtype=np.float64)
    start_idx = [s[0] for s in startpoints]
    start_tree = cKDTree(start_xy) if cKDTree is not None else None

    carla_successors: Dict[int, set] = {}
    thresh2 = threshold_m * threshold_m
    for ci, ex, ey in endpoints:
        if start_tree is not None:
            near = start_tree.query_ball_point(np.asarray([ex, ey], dtype=np.float64), r=threshold_m)
            for ni in near:
                cj = start_idx[ni]
                if cj != ci:
                    carla_successors.setdefault(ci, set()).add(cj)
        else:
            for cj, sx, sy in startpoints:
                if cj != ci and (ex - sx) ** 2 + (ey - sy) ** 2 <= thresh2:
                    carla_successors.setdefault(ci, set()).add(cj)
    return carla_successors


def _carla_lateral_adjacency(
    carla_feats: Dict[int, Dict[str, object]],
    threshold_m: float = 5.0,
) -> Dict[int, set]:
    """Detect pairs of CARLA lines that are roughly parallel and nearby
    (lateral adjacency).  Uses mid-point proximity + angle check."""
    mids: List[Tuple[int, float, float, float]] = []  # (ci, mx, my, heading_deg)
    for ci, cf in carla_feats.items():
        poly = cf["poly_xy"]
        mid = poly[poly.shape[0] // 2]
        dx = float(poly[-1, 0] - poly[0, 0])
        dy = float(poly[-1, 1] - poly[0, 1])
        heading = math.degrees(math.atan2(dy, dx)) if math.hypot(dx, dy) > 1e-6 else 0.0
        mids.append((ci, float(mid[0]), float(mid[1]), heading))

    if len(mids) < 2:
        return {}

    mid_xy = np.asarray([(m[1], m[2]) for m in mids], dtype=np.float64)
    mid_tree = cKDTree(mid_xy) if cKDTree is not None else None

    adjacency: Dict[int, set] = {}
    for i, (ci, mx, my, hi) in enumerate(mids):
        if mid_tree is not None:
            near = mid_tree.query_ball_point(np.asarray([mx, my], dtype=np.float64), r=threshold_m)
        else:
            near = [j for j in range(len(mids)) if math.hypot(mids[j][1] - mx, mids[j][2] - my) <= threshold_m]
        for j in near:
            if j == i:
                continue
            cj, _, _, hj = mids[j]
            angle_diff = abs(_normalize_yaw_deg(hi - hj))
            if angle_diff <= 30.0:  # roughly parallel (same direction)
                adjacency.setdefault(ci, set()).add(cj)
                adjacency.setdefault(cj, set()).add(ci)
    return adjacency


def _connectivity_consistency_penalty(
    assignment: Dict[int, int],  # v2_lane_index → carla_line_index
    v2_graph: Dict[str, object],
    carla_successors: Dict[int, set],
    weight: float = 0.5,
) -> float:
    """Compute a penalty for connectivity violations in the assignment.

    For each V2XPNP lane pair (A→B) that are connected by entry/exit_lanes,
    check whether their assigned CARLA lines are also connected (end-to-start
    proximity).  Returns a total penalty value.
    """
    v2_succs: Dict[int, set] = v2_graph.get("successors", {})
    penalty = 0.0
    n_checked = 0
    for li_a, ci_a in assignment.items():
        for li_b in v2_succs.get(li_a, set()):
            ci_b = assignment.get(li_b)
            if ci_b is None:
                continue
            n_checked += 1
            if ci_a == ci_b:
                continue  # same CARLA line — no penalty
            if ci_b in carla_successors.get(ci_a, set()):
                continue  # CARLA lines are connected — no penalty
            penalty += weight
    return penalty


def _adjacency_consistency_penalty(
    assignment: Dict[int, int],  # v2_lane_index → carla_line_index
    v2_graph: Dict[str, object],
    carla_adjacency: Dict[int, set],
    weight: float = 0.3,
) -> float:
    """Compute a penalty for adjacency violations.

    For V2XPNP lanes that share the same road_id with |Δlane_id|=1,
    their assigned CARLA lines should be laterally adjacent.
    """
    v2_adj: Dict[int, set] = v2_graph.get("adjacency", {})
    penalty = 0.0
    seen: set = set()
    for li_a, ci_a in assignment.items():
        for li_b in v2_adj.get(li_a, set()):
            pair = (min(li_a, li_b), max(li_a, li_b))
            if pair in seen:
                continue
            seen.add(pair)
            ci_b = assignment.get(li_b)
            if ci_b is None:
                continue
            if ci_a == ci_b:
                continue  # same line — could be a split, mild penalty
            if ci_b in carla_adjacency.get(ci_a, set()):
                continue  # properly adjacent
            penalty += weight
    return penalty


# ---------------------------------------------------------------------------
# Lane correspondence cache helpers
# ---------------------------------------------------------------------------

def _correspondence_cache_key(
    payload: Dict[str, object],
    driving_lane_types: Optional[Sequence[str]] = None,
    candidate_top_k: Optional[int] = None,
) -> str:
    """Compute a deterministic hash key from the map lane polylines and
    CARLA line polylines so we can cache the correspondence result.
    Includes the map name so intersection and corridor sections get
    distinct caches."""
    h = hashlib.sha256()
    # Hash map identity (name + source_path) for intersection vs corridor distinction
    map_meta = payload.get("map", {})
    map_name = str(map_meta.get("name", "unknown"))
    map_src = str(map_meta.get("source_path", ""))
    h.update(f"map_name:{map_name}|src:{map_src}|".encode())
    if driving_lane_types is not None:
        lane_types_norm = sorted({str(v).strip() for v in driving_lane_types if str(v).strip()})
        h.update(f"driving_lane_types:{','.join(lane_types_norm)}|".encode())
    if candidate_top_k is not None:
        h.update(f"candidate_top_k:{int(candidate_top_k)}|".encode())
    # Hash V2XPNP lanes
    lanes = payload.get("map", {}).get("lanes", [])
    for lane in (lanes if isinstance(lanes, list) else []):
        if not isinstance(lane, dict):
            continue
        li = _safe_int(lane.get("index"), -1)
        h.update(f"v2:{li}:".encode())
        for p in (lane.get("polyline") or []):
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                h.update(f"{float(p[0]):.4f},{float(p[1]):.4f};".encode())
    # Hash CARLA lines
    carla_lines = (payload.get("carla_map") or {}).get("lines", [])
    for i, ln in enumerate(carla_lines if isinstance(carla_lines, list) else []):
        h.update(f"c:{i}:".encode())
        pts = ln.get("polyline") if isinstance(ln, dict) else ln
        for p in (pts if isinstance(pts, (list, tuple)) else []):
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                h.update(f"{float(p[0]):.4f},{float(p[1]):.4f};".encode())
    return h.hexdigest()[:24]


def _load_cached_correspondence(cache_dir: Optional[Path], cache_key: str, map_name: str = "") -> Optional[Dict[str, object]]:
    if cache_dir is None:
        return None
    # Sanitise map name for use in filename
    safe_name = "".join(c if (c.isalnum() or c in "_-") else "_" for c in map_name)[:60]
    # Try pickle first (new format with features), then JSON (legacy)
    pkl_fname = f"lane_corr_{safe_name}_{cache_key}.pkl" if safe_name else f"lane_corr_{cache_key}.pkl"
    pkl_file = cache_dir / pkl_fname
    if pkl_file.exists():
        try:
            with pkl_file.open("rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict) and data.get("cache_key") == cache_key:
                print(f"[INFO] Loaded cached lane correspondence from {pkl_file}")
                return data
        except Exception:
            pass
    # Fall back to JSON (legacy format, no features)
    json_fname = f"lane_corr_{safe_name}_{cache_key}.json" if safe_name else f"lane_corr_{cache_key}.json"
    cache_file = cache_dir / json_fname
    if not cache_file.exists():
        # Also check legacy filename without map name
        legacy = cache_dir / f"lane_corr_{cache_key}.json"
        if legacy.exists():
            cache_file = legacy
        else:
            return None
    if not cache_file.exists():
        return None
    try:
        with cache_file.open("r") as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("cache_key") == cache_key:
            print(f"[INFO] Loaded cached lane correspondence from {cache_file} (legacy JSON, no features)")
            return data
    except Exception:
        pass
    return None


def _save_cached_correspondence(
    cache_dir: Optional[Path],
    cache_key: str,
    lane_to_carla: Dict[int, Dict[str, object]],
    carla_to_lanes: Dict[int, List[int]],
    driving_lane_types: list,
    map_name: str = "",
    v2_feats: Optional[Dict[int, Dict[str, object]]] = None,
    carla_feats: Optional[Dict[int, Dict[str, object]]] = None,
    v2_graph: Optional[Dict[str, object]] = None,
    carla_successors: Optional[Dict[int, set]] = None,
) -> None:
    if cache_dir is None:
        return
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(c if (c.isalnum() or c in "_-") else "_" for c in map_name)[:60]
        # Save as pickle (includes features with numpy arrays)
        pkl_fname = f"lane_corr_{safe_name}_{cache_key}.pkl" if safe_name else f"lane_corr_{cache_key}.pkl"
        pkl_file = cache_dir / pkl_fname
        serialisable = {
            "cache_key": cache_key,
            "lane_to_carla": {int(k): v for k, v in lane_to_carla.items()},
            "carla_to_lanes": {int(k): v for k, v in carla_to_lanes.items()},
            "driving_lane_types": list(driving_lane_types),
            "v2_feats": v2_feats,
            "carla_feats": carla_feats,
            "v2_graph": v2_graph,
            "carla_successors": {int(k): set(v) for k, v in (carla_successors or {}).items()},
        }
        with pkl_file.open("wb") as f:
            pickle.dump(serialisable, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[INFO] Saved lane correspondence cache to {pkl_file}")
    except Exception as e:
        print(f"[WARN] Failed to save lane correspondence cache: {e}")


# ---------------------------------------------------------------------------
# Main lane correspondence builder — global optimization
# ---------------------------------------------------------------------------

def _build_lane_correspondence(
    payload: Dict[str, object],
    candidate_top_k: int = 28,
    driving_lane_types: Sequence[str] = ("1",),
    cache_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Build V2XPNP-lane → CARLA-line correspondence using global
    optimisation (Hungarian / linear_sum_assignment) with multi-signal
    scoring: geometric distance, heading, connectivity consistency,
    adjacency consistency.

    Improvements over the previous greedy matcher:
    1. **Direction hard-filter** in _evaluate_lane_pair_metrics (mono < 0.45 → inf)
    2. **Global one-to-one assignment** via scipy linear_sum_assignment instead
       of greedy per-lane picking.
    3. **Connectivity consistency** — penalises assignments where connected
       V2XPNP lanes map to disconnected CARLA lines.
    4. **Adjacency consistency** — penalises assignments where spatially
       adjacent V2XPNP lanes map to non-adjacent CARLA lines.
    5. **Split/merge detection** — after global assignment, attempts to find
       secondary CARLA segments that cover gaps in partially-matched lanes.
    6. **Disk caching** — stores correspondence results keyed by content hash.
    """
    v2_lanes_raw = payload.get("map", {}).get("lanes", [])
    carla_map = payload.get("carla_map")
    if not isinstance(v2_lanes_raw, list) or not isinstance(carla_map, dict):
        return {"enabled": False, "reason": "missing_map_payload"}
    carla_lines_raw = carla_map.get("lines")
    if not isinstance(carla_lines_raw, list) or not carla_lines_raw:
        return {"enabled": False, "reason": "missing_carla_lines"}

    # --- Cache check ---
    _v2_map_name = str(payload.get("map", {}).get("name", ""))
    cache_key = _correspondence_cache_key(
        payload,
        driving_lane_types=driving_lane_types,
        candidate_top_k=int(candidate_top_k),
    )
    cached = _load_cached_correspondence(cache_dir, cache_key, map_name=_v2_map_name)

    def _normalize_int_set_map(raw_map: object) -> Dict[int, set]:
        out: Dict[int, set] = {}
        if not isinstance(raw_map, dict):
            return out
        for k, vals in raw_map.items():
            ki = _safe_int(k, -1)
            if ki < 0:
                continue
            bucket: set = set()
            if isinstance(vals, (list, tuple, set)):
                for v in vals:
                    vi = _safe_int(v, -1)
                    if vi >= 0:
                        bucket.add(int(vi))
            out[int(ki)] = bucket
        return out

    # --- If cache has features, use them directly (skip all phases) ---
    if cached is not None:
        cached_v2_feats = cached.get("v2_feats")
        cached_carla_feats = cached.get("carla_feats")
        if cached_v2_feats and cached_carla_feats:
            try:
                cached_l2c = cached.get("lane_to_carla", {})
                cached_c2l = cached.get("carla_to_lanes", {})
                lane_to_carla: Dict[int, Dict[str, object]] = {int(k): v for k, v in cached_l2c.items()}
                carla_to_lanes: Dict[int, List[int]] = {int(k): v for k, v in cached_c2l.items()}
                v2_feats_cached: Dict[int, Dict[str, object]] = {int(k): v for k, v in cached_v2_feats.items()}
                carla_feats_cached: Dict[int, Dict[str, object]] = {int(k): v for k, v in cached_carla_feats.items()}
                v2_graph_cached = cached.get("v2_graph")
                if not isinstance(v2_graph_cached, dict) or not v2_graph_cached:
                    v2_graph_cached = _build_v2_connectivity_graph(v2_feats_cached, v2_lanes_raw)
                carla_successors_cached = _normalize_int_set_map(cached.get("carla_successors"))
                if not carla_successors_cached:
                    carla_successors_cached = _carla_endpoint_proximity(carla_feats_cached, threshold_m=3.5)
                drive_types = {str(v) for v in driving_lane_types}
                print(f"  [CORR] Using fully cached correspondence (skipping all phases).")
                _save_cached_correspondence(
                    cache_dir,
                    cache_key,
                    lane_to_carla,
                    carla_to_lanes,
                    list(drive_types),
                    _v2_map_name,
                    v2_feats_cached,
                    carla_feats_cached,
                    v2_graph=v2_graph_cached,
                    carla_successors=carla_successors_cached,
                )
                return {
                    "enabled": True,
                    "v2_feats": v2_feats_cached,
                    "carla_feats": carla_feats_cached,
                    "lane_candidates": {},
                    "lane_to_carla": lane_to_carla,
                    "carla_to_lanes": carla_to_lanes,
                    "driving_lane_types": sorted(drive_types),
                    "split_merges": {},
                    "v2_graph": v2_graph_cached,
                    "carla_successors": carla_successors_cached,
                }
            except Exception:
                print("  [CORR] Cache corrupted, recomputing...")
                cached = None  # force recompute

    # ===== Phase 1: Feature extraction =====
    if cached is None:
        print(f"  [CORR] Phase 1/9: Feature extraction ({len(v2_lanes_raw)} v2 lanes, {len(carla_lines_raw)} CARLA lines)...")
    else:
        print(f"  [CORR] Phase 1/9: Feature extraction (legacy cache, rebuilding features)...")
    v2_feats: Dict[int, Dict[str, object]] = {}
    for lane in v2_lanes_raw:
        if not isinstance(lane, dict):
            continue
        li = _safe_int(lane.get("index"), -1)
        if li < 0:
            continue
        poly_xy = _polyline_xy_from_points(lane.get("polyline"))
        if poly_xy.shape[0] < 2:
            continue
        cum = _polyline_cumlen_xy(poly_xy)
        total_len = _polyline_total_len(cum)
        if total_len <= 1e-6:
            continue
        v2_feats[li] = {
            "lane_index": int(li),
            "road_id": _safe_int(lane.get("road_id"), 0),
            "lane_id": _safe_int(lane.get("lane_id"), 0),
            "lane_type": str(lane.get("lane_type", "")),
            "poly_xy": poly_xy,
            "cum": cum,
            "total_len": float(total_len),
            "label_x": _safe_float(lane.get("label_x"), float(poly_xy[poly_xy.shape[0] // 2, 0])),
            "label_y": _safe_float(lane.get("label_y"), float(poly_xy[poly_xy.shape[0] // 2, 1])),
            "entry_lanes": list(lane.get("entry_lanes") or []),
            "exit_lanes": list(lane.get("exit_lanes") or []),
        }

    carla_feats: Dict[int, Dict[str, object]] = {}
    for i, ln in enumerate(carla_lines_raw):
        if isinstance(ln, dict):
            poly_src = ln.get("polyline")
        else:
            poly_src = ln
        poly_xy = _polyline_xy_from_points(poly_src)
        if poly_xy.shape[0] < 2:
            continue
        cum = _polyline_cumlen_xy(poly_xy)
        total_len = _polyline_total_len(cum)
        if total_len <= 1e-6:
            continue
        poly_rev = poly_xy[::-1].copy()
        cum_rev = _polyline_cumlen_xy(poly_rev)
        label_mid = poly_xy[poly_xy.shape[0] // 2]
        carla_feats[i] = {
            "line_index": int(i),
            "poly_xy": poly_xy,
            "cum": cum,
            "total_len": float(total_len),
            "poly_xy_rev": poly_rev,
            "cum_rev": cum_rev,
            "label_x": float(label_mid[0]),
            "label_y": float(label_mid[1]),
        }

    if not v2_feats or not carla_feats:
        return {
            "enabled": False,
            "reason": "insufficient_lanes",
            "num_v2_lanes": int(len(v2_feats)),
            "num_carla_lines": int(len(carla_feats)),
        }

    # --- If we have a legacy cache (no features), use correspondence but we already rebuilt features ---
    if cached is not None:
        try:
            cached_l2c = cached.get("lane_to_carla", {})
            cached_c2l = cached.get("carla_to_lanes", {})
            lane_to_carla_cached: Dict[int, Dict[str, object]] = {int(k): v for k, v in cached_l2c.items()}
            carla_to_lanes_cached: Dict[int, List[int]] = {int(k): v for k, v in cached_c2l.items()}
            v2_graph_cached = cached.get("v2_graph")
            if not isinstance(v2_graph_cached, dict) or not v2_graph_cached:
                v2_graph_cached = _build_v2_connectivity_graph(v2_feats, v2_lanes_raw)
            carla_successors_cached = _normalize_int_set_map(cached.get("carla_successors"))
            if not carla_successors_cached:
                carla_successors_cached = _carla_endpoint_proximity(carla_feats, threshold_m=3.5)
            drive_types = {str(v) for v in driving_lane_types}
            print(f"  [CORR] Using cached correspondence (skipping phases 2-9).")
            # Re-save with features for next time
            _save_cached_correspondence(
                cache_dir,
                cache_key,
                lane_to_carla_cached,
                carla_to_lanes_cached,
                list(drive_types),
                _v2_map_name,
                v2_feats,
                carla_feats,
                v2_graph=v2_graph_cached,
                carla_successors=carla_successors_cached,
            )
            return {
                "enabled": True,
                "v2_feats": v2_feats,
                "carla_feats": carla_feats,
                "lane_candidates": {},
                "lane_to_carla": lane_to_carla_cached,
                "carla_to_lanes": carla_to_lanes_cached,
                "driving_lane_types": sorted(drive_types),
                "split_merges": {},
                "v2_graph": v2_graph_cached,
                "carla_successors": carla_successors_cached,
            }
        except Exception:
            print("  [CORR] Cache corrupted, recomputing...")
            pass  # cache corrupted, recompute

    # ===== Phase 2: Spatial indexing (KD-tree of all CARLA vertices) =====
    print(f"  [CORR] Phase 2/9: Spatial indexing (v2_feats={len(v2_feats)}, carla_feats={len(carla_feats)})...")
    vx: List[Tuple[float, float]] = []
    vl: List[int] = []
    for ci, cf in carla_feats.items():
        poly_xy = cf["poly_xy"]
        for p in poly_xy:
            vx.append((float(p[0]), float(p[1])))
            vl.append(int(ci))
    vertex_xy = np.asarray(vx, dtype=np.float64) if vx else np.zeros((0, 2), dtype=np.float64)
    vertex_line = np.asarray(vl, dtype=np.int32) if vl else np.zeros((0,), dtype=np.int32)
    tree = cKDTree(vertex_xy) if (cKDTree is not None and vertex_xy.shape[0] > 0) else None

    def candidate_carla_lines(x: float, y: float, k: int) -> List[int]:
        if vertex_xy.shape[0] <= 0:
            return []
        kk = max(1, min(int(k), int(vertex_xy.shape[0])))
        if tree is not None:
            _, idxs = tree.query(np.asarray([float(x), float(y)], dtype=np.float64), k=kk)
            if np.isscalar(idxs):
                idx_list = [int(idxs)]
            else:
                idx_list = [int(v) for v in np.asarray(idxs).reshape(-1)]
        else:
            d2 = np.sum((vertex_xy - np.asarray([float(x), float(y)], dtype=np.float64)[None, :]) ** 2, axis=1)
            idx_list = [int(v) for v in np.argsort(d2)[:kk]]
        out: List[int] = []
        seen: set = set()
        for vi in idx_list:
            li = int(vertex_line[vi])
            if li in seen:
                continue
            seen.add(li)
            out.append(li)
        return out

    # ===== Phase 3: Candidate scoring =====
    print(f"  [CORR] Phase 3/9: Candidate scoring ({len(v2_feats)} lanes × top-{candidate_top_k})...")
    lane_candidates: Dict[int, List[Dict[str, object]]] = {}
    drive_types = {str(v) for v in driving_lane_types}
    for li, lf in v2_feats.items():
        poly = lf["poly_xy"]
        n = int(poly.shape[0])
        # Sample more points along the lane for better candidate discovery
        sample_indices = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
        sample_pts = [poly[min(idx, n - 1)] for idx in sample_indices]
        cand_idx: set = set()
        for p in sample_pts:
            cands = candidate_carla_lines(float(p[0]), float(p[1]), k=max(8, int(candidate_top_k)))
            for ci in cands:
                cand_idx.add(int(ci))
        if not cand_idx:
            continue

        cand_rows: List[Dict[str, object]] = []
        for ci in sorted(cand_idx):
            cf = carla_feats.get(int(ci))
            if cf is None:
                continue
            m_fwd = _evaluate_lane_pair_metrics(lf["poly_xy"], lf["cum"], cf["poly_xy"], cf["cum"])
            m_rev = _evaluate_lane_pair_metrics(lf["poly_xy"], lf["cum"], cf["poly_xy_rev"], cf["cum_rev"])
            reversed_used = bool(float(m_rev.get("score", float("inf"))) < float(m_fwd.get("score", float("inf"))))
            met = m_rev if reversed_used else m_fwd
            q = _quality_from_metrics(met)
            row = {
                "lane_index": int(li),
                "carla_line_index": int(ci),
                "reversed": bool(reversed_used),
                "quality": str(q),
            }
            row.update({k: float(v) for k, v in met.items()})
            cand_rows.append(row)
        cand_rows.sort(key=lambda r: float(r.get("score", float("inf"))))
        lane_candidates[int(li)] = cand_rows

    # ===== Phase 4: Build connectivity/adjacency graphs =====
    print(f"  [CORR] Phase 4/9: Connectivity & adjacency graphs...")
    v2_graph = _build_v2_connectivity_graph(v2_feats, v2_lanes_raw)
    carla_successors = _carla_endpoint_proximity(carla_feats, threshold_m=3.5)
    carla_adj = _carla_lateral_adjacency(carla_feats, threshold_m=5.0)

    # ===== Phase 5: Global one-to-one assignment (Hungarian algorithm) =====
    # We solve the assignment for driving lanes first (one-to-one exclusive),
    # then assign non-driving lanes (allow shared CARLA lines).
    driving_lanes_list = sorted(
        [li for li, lf in v2_feats.items() if str(lf.get("lane_type", "")) in drive_types],
    )
    non_driving_lanes_list = sorted(
        [li for li in v2_feats if li not in set(driving_lanes_list)],
    )
    print(f"  [CORR] Phase 5/9: Hungarian assignment ({len(driving_lanes_list)} driving + {len(non_driving_lanes_list)} non-driving lanes)...")

    # Collect all CARLA line candidates that appear for any driving lane
    all_carla_candidates: set = set()
    for li in driving_lanes_list:
        for row in lane_candidates.get(li, []):
            all_carla_candidates.add(int(row["carla_line_index"]))
    carla_cand_list = sorted(all_carla_candidates)

    lane_to_carla: Dict[int, Dict[str, object]] = {}

    if driving_lanes_list and carla_cand_list and _scipy_lsa is not None:
        # Build cost matrix: rows = driving V2 lanes, cols = CARLA candidates
        n_v2 = len(driving_lanes_list)
        n_carla = len(carla_cand_list)
        v2_idx_map = {li: i for i, li in enumerate(driving_lanes_list)}
        carla_idx_map = {ci: j for j, ci in enumerate(carla_cand_list)}

        # Large penalty for unmatched or impossible pairs
        BIG_COST = 100.0
        cost_matrix = np.full((n_v2, n_carla), BIG_COST, dtype=np.float64)

        # Best candidate row for each (v2_lane, carla_line) pair
        best_row_lookup: Dict[Tuple[int, int], Dict[str, object]] = {}

        for li in driving_lanes_list:
            i = v2_idx_map[li]
            for row in lane_candidates.get(li, []):
                ci = int(row["carla_line_index"])
                j = carla_idx_map.get(ci)
                if j is None:
                    continue
                sc = float(row.get("score", float("inf")))
                if sc < cost_matrix[i, j]:
                    cost_matrix[i, j] = sc
                    best_row_lookup[(li, ci)] = row

        # Run Hungarian algorithm
        # Handle rectangular matrices: more V2 lanes or more CARLA lines
        if n_v2 <= n_carla:
            row_ind, col_ind = _scipy_lsa(cost_matrix)
        else:
            # Transpose so we have fewer rows
            col_ind_t, row_ind_t = _scipy_lsa(cost_matrix.T)
            row_ind = row_ind_t
            col_ind = col_ind_t

        # Build initial assignment
        initial_assignment: Dict[int, int] = {}
        for r, c in zip(row_ind, col_ind):
            if r < n_v2 and c < n_carla:
                li = driving_lanes_list[r]
                ci = carla_cand_list[c]
                if cost_matrix[r, c] < BIG_COST:
                    initial_assignment[li] = ci

        # ===== Phase 5b: Connectivity/adjacency refinement =====
        print(f"  [CORR] Phase 5b/9: Connectivity refinement (initial={len(initial_assignment)} assignments)...")
        # Evaluate current assignment quality including structural consistency
        # Attempt local swaps to reduce structural penalties
        improved = True
        assignment = dict(initial_assignment)
        max_swap_iters = min(50, n_v2 * n_v2)
        swap_iter = 0
        while improved and swap_iter < max_swap_iters:
            improved = False
            swap_iter += 1
            v2_assigned = list(assignment.keys())
            for idx_a in range(len(v2_assigned)):
                if improved:
                    break
                li_a = v2_assigned[idx_a]
                ci_a = assignment[li_a]
                # Try swapping with another assigned lane
                for idx_b in range(idx_a + 1, len(v2_assigned)):
                    li_b = v2_assigned[idx_b]
                    ci_b = assignment[li_b]
                    if ci_a == ci_b:
                        continue
                    # Check if swapped costs are not much worse
                    i_a, j_a = v2_idx_map[li_a], carla_idx_map[ci_a]
                    i_b, j_b = v2_idx_map[li_b], carla_idx_map[ci_b]
                    j_a_new, j_b_new = carla_idx_map[ci_b], carla_idx_map[ci_a]
                    old_cost = cost_matrix[i_a, j_a] + cost_matrix[i_b, j_b]
                    new_cost = cost_matrix[i_a, j_a_new] + cost_matrix[i_b, j_b_new]
                    # Only consider swaps where geometric cost increase is small
                    if new_cost > old_cost + 2.0:
                        continue
                    # Evaluate structural improvement
                    test = dict(assignment)
                    test[li_a] = ci_b
                    test[li_b] = ci_a
                    new_penalty = (
                        _connectivity_consistency_penalty(test, v2_graph, carla_successors, weight=0.5)
                        + _adjacency_consistency_penalty(test, v2_graph, carla_adj, weight=0.3)
                    )
                    cur_penalty = (
                        _connectivity_consistency_penalty(assignment, v2_graph, carla_successors, weight=0.5)
                        + _adjacency_consistency_penalty(assignment, v2_graph, carla_adj, weight=0.3)
                    )
                    total_improvement = (cur_penalty - new_penalty) - (new_cost - old_cost)
                    if total_improvement > 0.1:
                        assignment[li_a] = ci_b
                        assignment[li_b] = ci_a
                        improved = True
                        break
                # Also try swapping with an unassigned CARLA line
                if not improved:
                    used_carla_set = set(assignment.values())
                    for row in lane_candidates.get(li_a, []):
                        ci_new = int(row["carla_line_index"])
                        if ci_new in used_carla_set:
                            continue
                        j_new = carla_idx_map.get(ci_new)
                        if j_new is None:
                            continue
                        old_cost_a = cost_matrix[i_a, j_a]
                        new_cost_a = cost_matrix[i_a, j_new]
                        if new_cost_a >= BIG_COST:
                            continue
                        test = dict(assignment)
                        test[li_a] = ci_new
                        new_penalty = (
                            _connectivity_consistency_penalty(test, v2_graph, carla_successors, weight=0.5)
                            + _adjacency_consistency_penalty(test, v2_graph, carla_adj, weight=0.3)
                        )
                        cur_penalty = (
                            _connectivity_consistency_penalty(assignment, v2_graph, carla_successors, weight=0.5)
                            + _adjacency_consistency_penalty(assignment, v2_graph, carla_adj, weight=0.3)
                        )
                        total_improvement = (cur_penalty - new_penalty) - (new_cost_a - old_cost_a)
                        if total_improvement > 0.1:
                            assignment[li_a] = ci_new
                            improved = True
                            break

        # Convert assignment to lane_to_carla entries
        for li, ci in assignment.items():
            row = best_row_lookup.get((li, ci))
            if row is None:
                # Build metrics on the fly
                cf = carla_feats.get(ci)
                lf = v2_feats.get(li)
                if cf is None or lf is None:
                    continue
                m_fwd = _evaluate_lane_pair_metrics(lf["poly_xy"], lf["cum"], cf["poly_xy"], cf["cum"])
                m_rev = _evaluate_lane_pair_metrics(lf["poly_xy"], lf["cum"], cf["poly_xy_rev"], cf["cum_rev"])
                reversed_used = bool(float(m_rev.get("score", float("inf"))) < float(m_fwd.get("score", float("inf"))))
                met = m_rev if reversed_used else m_fwd
                q = _quality_from_metrics(met)
                row = {
                    "lane_index": int(li),
                    "carla_line_index": int(ci),
                    "reversed": bool(reversed_used),
                    "quality": str(q),
                }
                row.update({k: float(v) for k, v in met.items()})
            lane_to_carla[int(li)] = dict(row)

    elif driving_lanes_list:
        # Fallback: greedy assignment if scipy not available
        driving_lanes_sorted = sorted(
            driving_lanes_list,
            key=lambda li: float(lane_candidates.get(li, [{}])[0].get("score", float("inf"))),
        )
        used_carla: set = set()
        for li in driving_lanes_sorted:
            rows = lane_candidates.get(li, [])
            picked: Optional[Dict[str, object]] = None
            for r in rows:
                if int(r["carla_line_index"]) in used_carla:
                    continue
                if str(r.get("quality", "poor")) == "poor":
                    continue
                picked = r
                break
            if picked is None:
                for r in rows:
                    if int(r["carla_line_index"]) in used_carla:
                        continue
                    picked = r
                    break
            if picked is None and rows:
                picked = rows[0]
            if picked is None:
                continue
            lane_to_carla[int(li)] = dict(picked)
            used_carla.add(int(picked["carla_line_index"]))

    # ===== Phase 6: Non-driving lanes (allow shared CARLA lines) =====
    print(f"  [CORR] Phase 6/9: Non-driving lane assignment...")
    for li in non_driving_lanes_list:
        if li in lane_to_carla:
            continue
        rows = lane_candidates.get(li, [])
        if not rows:
            continue
        if str(rows[0].get("quality", "poor")) != "poor":
            lane_to_carla[int(li)] = dict(rows[0])

    # ===== Phase 7: Split detection =====
    print(f"  [CORR] Phase 7/9: Split detection ({len(lane_to_carla)} assigned so far)...")
    # For driving lanes with low coverage, try to find secondary CARLA
    # segments that cover the uncovered portion of the V2XPNP lane.
    used_primary_carla = set()
    for li, info in lane_to_carla.items():
        used_primary_carla.add(int(info.get("carla_line_index", -1)))

    split_merges: Dict[int, List[int]] = {}  # v2_lane → list of extra CARLA indices
    for li in driving_lanes_list:
        if li not in lane_to_carla:
            continue
        info = lane_to_carla[li]
        cov = float(info.get("coverage_2m", 1.0))
        if cov >= 0.75:
            continue  # good enough coverage, no split needed
        lf = v2_feats.get(li)
        if lf is None:
            continue
        primary_ci = int(info.get("carla_line_index", -1))
        primary_reversed = bool(info.get("reversed", False))
        pcf = carla_feats.get(primary_ci)
        if pcf is None:
            continue

        # Find which portion of the V2XPNP lane is uncovered
        v2_poly = lf["poly_xy"]
        v2_cum = lf["cum"]
        v2_len = float(lf["total_len"])
        n_probe = int(max(10, min(40, round(v2_len / 2.0))))
        s_probes = np.linspace(0.0, v2_len, n_probe, dtype=np.float64)

        c_poly = pcf["poly_xy_rev"] if primary_reversed else pcf["poly_xy"]
        c_cum = pcf["cum_rev"] if primary_reversed else pcf["cum"]

        uncovered_pts: List[Tuple[float, float]] = []
        for sv in s_probes.tolist():
            sp = _sample_polyline_at_s_xy(v2_poly, v2_cum, sv)
            if sp is None:
                continue
            proj = _project_point_to_polyline_xy(c_poly, c_cum, sp["x"], sp["y"])
            if proj is not None and proj["dist"] <= 2.0:
                continue
            uncovered_pts.append((sp["x"], sp["y"]))

        if not uncovered_pts:
            continue

        # Find candidate CARLA lines near uncovered points
        extra_cands: set = set()
        for px, py in uncovered_pts[:5]:  # sample a few
            cands = candidate_carla_lines(px, py, k=max(4, int(candidate_top_k) // 2))
            for ci in cands:
                if ci != primary_ci:
                    extra_cands.add(ci)

        best_extra_ci = -1
        best_extra_row: Optional[Dict[str, object]] = None
        for ci in sorted(extra_cands):
            cf = carla_feats.get(ci)
            if cf is None:
                continue
            # Evaluate how well this extra CARLA line covers the uncovered portion
            m_fwd = _evaluate_lane_pair_metrics(lf["poly_xy"], lf["cum"], cf["poly_xy"], cf["cum"])
            m_rev = _evaluate_lane_pair_metrics(lf["poly_xy"], lf["cum"], cf["poly_xy_rev"], cf["cum_rev"])
            rev_used = bool(float(m_rev.get("score", float("inf"))) < float(m_fwd.get("score", float("inf"))))
            met = m_rev if rev_used else m_fwd
            q = _quality_from_metrics(met)
            if q == "poor":
                continue
            if best_extra_row is None or float(met["score"]) < float(best_extra_row.get("score", float("inf"))):
                best_extra_ci = ci
                best_extra_row = {
                    "lane_index": int(li),
                    "carla_line_index": int(ci),
                    "reversed": bool(rev_used),
                    "quality": str(q),
                }
                best_extra_row.update({k: float(v) for k, v in met.items()})

        if best_extra_ci >= 0:
            split_merges.setdefault(li, []).append(best_extra_ci)

    # ===== Phase 8: Build reverse mapping =====
    print(f"  [CORR] Phase 8/9: Reverse mapping...")
    carla_to_lanes: Dict[int, List[int]] = {}
    for li, info in lane_to_carla.items():
        ci = int(info.get("carla_line_index", -1))
        if ci < 0:
            continue
        carla_to_lanes.setdefault(ci, []).append(int(li))
    # Include split/merge secondary mappings in the reverse map
    for li, extra_cis in split_merges.items():
        for ci in extra_cis:
            carla_to_lanes.setdefault(ci, []).append(int(li))

    for ci, lset in carla_to_lanes.items():
        lset_sorted = sorted(set(int(v) for v in lset))
        carla_to_lanes[ci] = lset_sorted
        for li in lset_sorted:
            if li in lane_to_carla:
                lane_to_carla[li]["shared_carla_line"] = bool(len(lset_sorted) > 1)

    # Add split_merges info to lane_to_carla entries
    for li, extra_cis in split_merges.items():
        if li in lane_to_carla:
            lane_to_carla[li]["split_extra_carla_lines"] = [int(c) for c in extra_cis]

    # ===== Phase 9: Cache result =====
    print(f"  [CORR] Phase 9/9: Caching result...")
    _save_cached_correspondence(
        cache_dir, cache_key, lane_to_carla, carla_to_lanes,
        sorted(drive_types), map_name=_v2_map_name,
        v2_feats=v2_feats,
        carla_feats=carla_feats,
        v2_graph=v2_graph,
        carla_successors=carla_successors,
    )

    return {
        "enabled": True,
        "v2_feats": v2_feats,
        "carla_feats": carla_feats,
        "lane_candidates": lane_candidates,
        "lane_to_carla": lane_to_carla,
        "carla_to_lanes": carla_to_lanes,
        "driving_lane_types": sorted(drive_types),
        "split_merges": split_merges,
        "v2_graph": v2_graph,
        "carla_successors": carla_successors,
    }


def _carla_lines_connected(
    ci_a: int,
    ci_b: int,
    carla_feats: Dict[int, Dict[str, object]],
    carla_succs: Dict[int, set],
    threshold_m: float = 6.0,
) -> bool:
    """Check whether two CARLA lines are directionally connected.

    Connectivity is accepted when:
      1) explicit successor relation exists, or
      2) end(A)->start(B) is within threshold and terminal headings are
         not anti-parallel.

    Note: start-start / end-end proximity is *not* considered connected, as
    that creates many false positives in dense intersections.
    """
    if ci_a == ci_b:
        return True
    if ci_b in carla_succs.get(ci_a, set()):
        return True
    if ci_a in carla_succs.get(ci_b, set()):
        return True
    cf_a = carla_feats.get(ci_a)
    cf_b = carla_feats.get(ci_b)
    if cf_a is None or cf_b is None:
        return False

    pa = cf_a.get("poly_xy")
    pb = cf_b.get("poly_xy")
    if not isinstance(pa, np.ndarray) or not isinstance(pb, np.ndarray):
        return False
    if pa.shape[0] < 2 or pb.shape[0] < 2:
        return False

    def _terminal_heading(poly_xy: np.ndarray, end_side: bool) -> float:
        if end_side:
            p0 = poly_xy[-2]
            p1 = poly_xy[-1]
        else:
            p0 = poly_xy[0]
            p1 = poly_xy[1]
        return _normalize_yaw_deg(math.degrees(math.atan2(float(p1[1] - p0[1]), float(p1[0] - p0[0]))))

    # A -> B
    a_end = pa[-1]
    b_start = pb[0]
    d_ab = math.hypot(float(a_end[0] - b_start[0]), float(a_end[1] - b_start[1]))
    if d_ab <= float(threshold_m):
        ha = _terminal_heading(pa, end_side=True)
        hb = _terminal_heading(pb, end_side=False)
        if _yaw_abs_diff_deg(float(ha), float(hb)) <= 120.0:
            return True

    return False


def _is_walker_like_role_name(role: object) -> bool:
    role_name = str(role or "").strip().lower()
    return role_name in {"walker", "pedestrian", "cyclist", "bicycle"}


def _compute_motion_heading_series_from_frames(
    frames: Sequence[object],
    lookahead_frames: int = 8,
    min_disp_m: float = 0.35,
) -> List[Optional[float]]:
    """Estimate per-frame motion heading from trajectory displacement."""
    n = int(len(frames))
    if n <= 0:
        return []

    out: List[Optional[float]] = [None] * n

    def _frame_xy(fi: int) -> Optional[Tuple[float, float]]:
        if fi < 0 or fi >= n:
            return None
        fr = frames[fi]
        if not isinstance(fr, dict):
            return None
        fx = _safe_float(fr.get("mx"), _safe_float(fr.get("x"), float("nan")))
        fy = _safe_float(fr.get("my"), _safe_float(fr.get("y"), float("nan")))
        if not (math.isfinite(fx) and math.isfinite(fy)):
            return None
        return float(fx), float(fy)

    min_disp = max(0.05, float(min_disp_m))
    win = max(1, int(lookahead_frames))

    for i in range(n):
        cur_xy = _frame_xy(i)
        if cur_xy is None:
            continue
        cx, cy = cur_xy
        found_yaw: Optional[float] = None

        for j in range(1, win + 1):
            nxt = _frame_xy(i + j)
            if nxt is None:
                continue
            dx = float(nxt[0] - cx)
            dy = float(nxt[1] - cy)
            if math.hypot(dx, dy) >= min_disp:
                found_yaw = _normalize_yaw_deg(math.degrees(math.atan2(dy, dx)))
                break

        if found_yaw is None:
            for j in range(1, win + 1):
                prv = _frame_xy(i - j)
                if prv is None:
                    continue
                dx = float(cx - prv[0])
                dy = float(cy - prv[1])
                if math.hypot(dx, dy) >= min_disp:
                    found_yaw = _normalize_yaw_deg(math.degrees(math.atan2(dy, dx)))
                    break

        out[i] = found_yaw

    return out
