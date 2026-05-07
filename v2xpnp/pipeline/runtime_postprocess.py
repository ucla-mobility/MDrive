"""Stage internals: overlap reduction and deduplication."""

from __future__ import annotations

from v2xpnp.pipeline import runtime_common as _s1
from v2xpnp.pipeline import runtime_projection as _s2

for _mod in (_s1, _s2):
    for _name, _value in vars(_mod).items():
        if _name.startswith("__"):
            continue
        globals()[_name] = _value

def _carla_track_key(track: Dict[str, object]) -> str:
    role = str(track.get("role", "")).strip().lower() or "actor"
    tid = str(track.get("id", ""))
    return f"{role}:{tid}"


def _track_raw_motion_stats(track: Dict[str, object]) -> Dict[str, float]:
    frames = track.get("frames", [])
    if not isinstance(frames, list) or len(frames) < 2:
        return {
            "n": float(len(frames) if isinstance(frames, list) else 0),
            "path_len_m": 0.0,
            "net_disp_m": 0.0,
            "robust_net_disp_m": 0.0,
            "anchor_disp_m": 0.0,
            "disp_p90_from_median_m": 0.0,
            "disp_p95_from_median_m": 0.0,
            "sustained_disp_ratio": 0.0,
            "max_disp_run_frames": 0.0,
            "max_disp_run_s": 0.0,
            "avg_speed_mps": 0.0,
            "p95_step_m": 0.0,
            "duration_s": 0.0,
        }
    path = 0.0
    steps: List[float] = []
    xs: List[float] = []
    ys: List[float] = []
    ts: List[float] = []
    for i in range(1, len(frames)):
        x0 = _safe_float(frames[i - 1].get("x"), 0.0)
        y0 = _safe_float(frames[i - 1].get("y"), 0.0)
        x1 = _safe_float(frames[i].get("x"), x0)
        y1 = _safe_float(frames[i].get("y"), y0)
        d = float(math.hypot(float(x1) - float(x0), float(y1) - float(y0)))
        path += d
        steps.append(d)
    for i, fr in enumerate(frames):
        xs.append(float(_safe_float(fr.get("x"), 0.0)))
        ys.append(float(_safe_float(fr.get("y"), 0.0)))
        ts.append(float(_safe_float(fr.get("t"), float(i) * 0.1)))
    x_start = _safe_float(frames[0].get("x"), 0.0)
    y_start = _safe_float(frames[0].get("y"), 0.0)
    x_end = _safe_float(frames[-1].get("x"), x_start)
    y_end = _safe_float(frames[-1].get("y"), y_start)
    net = float(math.hypot(float(x_end) - float(x_start), float(y_end) - float(y_start)))
    t0 = _safe_float(frames[0].get("t"), 0.0)
    t1 = _safe_float(frames[-1].get("t"), float(max(1, len(frames) - 1)) * 0.1)
    dur = max(0.1, float(t1) - float(t0))
    p95_step = float(np.percentile(np.asarray(steps, dtype=np.float64), 95.0)) if steps else 0.0
    dt_vals: List[float] = []
    for i in range(1, len(ts)):
        dt_vals.append(max(5e-2, float(ts[i]) - float(ts[i - 1])))
    dt_med = float(np.median(np.asarray(dt_vals, dtype=np.float64))) if dt_vals else 0.1

    # Robust displacement signatures (insensitive to sparse jitter spikes).
    arr_x = np.asarray(xs, dtype=np.float64)
    arr_y = np.asarray(ys, dtype=np.float64)
    med_x = float(np.median(arr_x))
    med_y = float(np.median(arr_y))
    d_med = np.hypot(arr_x - float(med_x), arr_y - float(med_y))
    d_p90 = float(np.percentile(d_med, 90.0)) if d_med.size > 0 else 0.0
    d_p95 = float(np.percentile(d_med, 95.0)) if d_med.size > 0 else 0.0

    anchor_win = _env_int("V2X_CARLA_OVERLAP_MOTION_ANCHOR_WIN_FRAMES", 5, minimum=2, maximum=50)
    w = int(max(2, min(int(anchor_win), len(frames) // 2 if len(frames) >= 4 else 2)))
    s_x = float(np.median(arr_x[:w])) if arr_x.size > 0 else float(x_start)
    s_y = float(np.median(arr_y[:w])) if arr_y.size > 0 else float(y_start)
    e_x = float(np.median(arr_x[-w:])) if arr_x.size > 0 else float(x_end)
    e_y = float(np.median(arr_y[-w:])) if arr_y.size > 0 else float(y_end)
    anchor_disp = float(math.hypot(float(e_x) - float(s_x), float(e_y) - float(s_y)))

    sustain_thr = _env_float("V2X_CARLA_OVERLAP_MOTION_SUSTAIN_DISP_THR_M", 1.2)
    sustain_mask = d_med >= float(max(0.2, sustain_thr))
    sustain_ratio = float(np.mean(sustain_mask.astype(np.float64))) if d_med.size > 0 else 0.0
    max_run = 0
    cur_run = 0
    for flag in sustain_mask.tolist():
        if bool(flag):
            cur_run += 1
            if cur_run > max_run:
                max_run = int(cur_run)
        else:
            cur_run = 0
    robust_net = float(min(float(net), float(anchor_disp)))
    max_run_s = float(max_run) * float(max(0.05, dt_med))
    return {
        "n": float(len(frames)),
        "path_len_m": float(path),
        "net_disp_m": float(net),
        "robust_net_disp_m": float(robust_net),
        "anchor_disp_m": float(anchor_disp),
        "disp_p90_from_median_m": float(d_p90),
        "disp_p95_from_median_m": float(d_p95),
        "sustained_disp_ratio": float(sustain_ratio),
        "max_disp_run_frames": float(max_run),
        "max_disp_run_s": float(max_run_s),
        "avg_speed_mps": float(path / dur),
        "p95_step_m": float(p95_step),
        "duration_s": float(dur),
    }


def _is_parked_vehicle_track_for_overlap(track: Dict[str, object]) -> bool:
    role = str(track.get("role", "")).strip().lower()
    if role != "vehicle":
        return False
    if bool(track.get("low_motion_vehicle", False)):
        return True
    stats = _track_raw_motion_stats(track)
    path_len = float(stats.get("path_len_m", 0.0))
    net_disp = float(stats.get("net_disp_m", 0.0))
    robust_net_disp = float(stats.get("robust_net_disp_m", net_disp))
    anchor_disp = float(stats.get("anchor_disp_m", net_disp))
    disp_p95 = float(stats.get("disp_p95_from_median_m", 0.0))
    sustain_ratio = float(stats.get("sustained_disp_ratio", 1.0))
    max_disp_run_s = float(stats.get("max_disp_run_s", 0.0))
    avg_speed = float(stats.get("avg_speed_mps", 0.0))
    p95_step = float(stats.get("p95_step_m", 0.0))
    parked_path_max = _env_float("V2X_CARLA_OVERLAP_PARKED_PATH_MAX_M", 16.0)
    parked_net_max = _env_float("V2X_CARLA_OVERLAP_PARKED_NET_MAX_M", 4.5)
    parked_speed_max = _env_float("V2X_CARLA_OVERLAP_PARKED_SPEED_MAX_MPS", 1.6)
    parked_step_max = _env_float("V2X_CARLA_OVERLAP_PARKED_P95_STEP_MAX_M", 0.28)
    if bool(
        path_len <= float(parked_path_max)
        and net_disp <= float(parked_net_max)
        and avg_speed <= float(parked_speed_max)
        and p95_step <= float(parked_step_max)
    ):
        return True

    # Lidar jitter can inflate path length for effectively parked vehicles.
    # Accept these as parked-like when net displacement is very small and
    # motion is dominated by local oscillation rather than translation.
    jitter_enabled = _env_int("V2X_CARLA_OVERLAP_PARKED_JITTER_ENABLED", 0, minimum=0, maximum=1) == 1
    jitter_path_max = _env_float("V2X_CARLA_OVERLAP_PARKED_JITTER_PATH_MAX_M", 24.0)
    jitter_net_max = _env_float("V2X_CARLA_OVERLAP_PARKED_JITTER_NET_MAX_M", 1.8)
    jitter_speed_max = _env_float("V2X_CARLA_OVERLAP_PARKED_JITTER_SPEED_MAX_MPS", 1.8)
    jitter_step_max = _env_float("V2X_CARLA_OVERLAP_PARKED_JITTER_P95_STEP_MAX_M", 0.34)
    jitter_path_net_ratio_min = _env_float("V2X_CARLA_OVERLAP_PARKED_JITTER_PATH_NET_RATIO_MIN", 7.0)
    path_net_ratio = float(path_len / max(0.2, net_disp))
    if (
        bool(jitter_enabled)
        and path_len <= float(jitter_path_max)
        and net_disp <= float(jitter_net_max)
        and avg_speed <= float(jitter_speed_max)
        and p95_step <= float(jitter_step_max)
        and path_net_ratio >= float(jitter_path_net_ratio_min)
    ):
        return True

    # Robust stationary signature: tolerate occasional jitter spikes while
    # requiring low sustained displacement from the median anchor region.
    robust_stationary_enabled = _env_int("V2X_CARLA_OVERLAP_PARKED_ROBUST_ENABLED", 1, minimum=0, maximum=1) == 1
    robust_path_max = _env_float("V2X_CARLA_OVERLAP_PARKED_ROBUST_PATH_MAX_M", 30.0)
    robust_anchor_max = _env_float("V2X_CARLA_OVERLAP_PARKED_ROBUST_ANCHOR_MAX_M", 2.3)
    robust_net_max = _env_float("V2X_CARLA_OVERLAP_PARKED_ROBUST_NET_MAX_M", 2.8)
    robust_disp_p95_max = _env_float("V2X_CARLA_OVERLAP_PARKED_ROBUST_DISP_P95_MAX_M", 2.0)
    robust_sustain_ratio_max = _env_float("V2X_CARLA_OVERLAP_PARKED_ROBUST_SUSTAIN_RATIO_MAX", 0.30)
    robust_max_disp_run_s = _env_float("V2X_CARLA_OVERLAP_PARKED_ROBUST_MAX_DISP_RUN_S", 2.2)
    robust_speed_max = _env_float("V2X_CARLA_OVERLAP_PARKED_ROBUST_SPEED_MAX_MPS", 2.2)
    if (
        bool(robust_stationary_enabled)
        and path_len <= float(robust_path_max)
        and anchor_disp <= float(robust_anchor_max)
        and robust_net_disp <= float(robust_net_max)
        and disp_p95 <= float(robust_disp_p95_max)
        and sustain_ratio <= float(robust_sustain_ratio_max)
        and max_disp_run_s <= float(robust_max_disp_run_s)
        and avg_speed <= float(robust_speed_max)
    ):
        return True

    return False


def _is_quasi_parked_vehicle_track_for_overlap(track: Dict[str, object]) -> bool:
    role = str(track.get("role", "")).strip().lower()
    if role != "vehicle":
        return False
    if _is_parked_vehicle_track_for_overlap(track):
        return True
    stats = _track_raw_motion_stats(track)
    path_len = float(stats.get("path_len_m", 0.0))
    robust_net_disp = float(stats.get("robust_net_disp_m", stats.get("net_disp_m", 0.0)))
    anchor_disp = float(stats.get("anchor_disp_m", stats.get("net_disp_m", 0.0)))
    disp_p95 = float(stats.get("disp_p95_from_median_m", 0.0))
    sustain_ratio = float(stats.get("sustained_disp_ratio", 1.0))
    max_disp_run_s = float(stats.get("max_disp_run_s", 0.0))
    avg_speed = float(stats.get("avg_speed_mps", 0.0))
    p95_step = float(stats.get("p95_step_m", 0.0))
    q_path_max = _env_float("V2X_CARLA_OVERLAP_QUASI_PARKED_PATH_MAX_M", 40.0)
    q_anchor_max = _env_float("V2X_CARLA_OVERLAP_QUASI_PARKED_ANCHOR_MAX_M", 3.4)
    q_robust_net_max = _env_float("V2X_CARLA_OVERLAP_QUASI_PARKED_ROBUST_NET_MAX_M", 4.2)
    q_disp_p95_max = _env_float("V2X_CARLA_OVERLAP_QUASI_PARKED_DISP_P95_MAX_M", 2.8)
    q_sustain_ratio_max = _env_float("V2X_CARLA_OVERLAP_QUASI_PARKED_SUSTAIN_RATIO_MAX", 0.42)
    q_max_disp_run_s = _env_float("V2X_CARLA_OVERLAP_QUASI_PARKED_MAX_DISP_RUN_S", 2.8)
    q_speed_max = _env_float("V2X_CARLA_OVERLAP_QUASI_PARKED_SPEED_MAX_MPS", 3.0)
    q_step_max = _env_float("V2X_CARLA_OVERLAP_QUASI_PARKED_P95_STEP_MAX_M", 1.0)
    return bool(
        path_len <= float(q_path_max)
        and anchor_disp <= float(q_anchor_max)
        and robust_net_disp <= float(q_robust_net_max)
        and disp_p95 <= float(q_disp_p95_max)
        and sustain_ratio <= float(q_sustain_ratio_max)
        and max_disp_run_s <= float(q_max_disp_run_s)
        and avg_speed <= float(q_speed_max)
        and p95_step <= float(q_step_max)
    )


def _vehicle_dims_for_overlap(track: Dict[str, object]) -> Tuple[float, float]:
    len_scale = _env_float("V2X_CARLA_OVERLAP_VEHICLE_LEN_SCALE", 1.0)
    wid_scale = _env_float("V2X_CARLA_OVERLAP_VEHICLE_WID_SCALE", 1.0)
    use_meta_dims = _env_int("V2X_CARLA_OVERLAP_USE_META_DIMS", 0, minimum=0, maximum=1) == 1
    role = str(track.get("role", "")).strip().lower()
    obj = str(track.get("obj_type", "")).strip().lower()
    meta_len = _safe_float(track.get("length"), float("nan"))
    meta_wid = _safe_float(track.get("width"), float("nan"))
    has_meta_dims = (
        math.isfinite(meta_len)
        and math.isfinite(meta_wid)
        and 1.6 <= float(meta_len) <= 28.0
        and 0.8 <= float(meta_wid) <= 4.5
    )
    base_len = 4.6
    base_wid = 2.0
    if bool(use_meta_dims) and bool(has_meta_dims):
        base_len, base_wid = float(meta_len), float(meta_wid)
    elif role == "ego":
        base_len, base_wid = 4.8, 2.1
    elif "bus" in obj:
        base_len, base_wid = 12.0, 2.9
    elif "concrete" in obj:
        base_len, base_wid = 9.5, 2.8
    elif "truck" in obj:
        base_len, base_wid = 8.8, 2.7
    elif "van" in obj:
        base_len, base_wid = 5.4, 2.1
    elif any(tok in obj for tok in ("motor", "scooter", "bike", "bicycle")):
        base_len, base_wid = 2.3, 1.0
    elif any(tok in obj for tok in ("trash", "cone", "barrier")):
        base_len, base_wid = 1.0, 1.0
    return (
        float(max(1.6, float(base_len) * float(len_scale))),
        float(max(0.8, float(base_wid) * float(wid_scale))),
    )


def _vehicle_bbox_uncertainty_m(track: Dict[str, object], length_m: float, width_m: float) -> float:
    # Conservative geometric uncertainty budget used to ignore near-threshold
    # overlap that can come from coarse actor size priors.
    base = _env_float("V2X_CARLA_OVERLAP_BBOX_UNCERT_BASE_M", 0.055)
    rel = _env_float("V2X_CARLA_OVERLAP_BBOX_UNCERT_REL", 0.010)
    min_m = _env_float("V2X_CARLA_OVERLAP_BBOX_UNCERT_MIN_M", 0.020)
    max_m = _env_float("V2X_CARLA_OVERLAP_BBOX_UNCERT_MAX_M", 0.180)
    with_meta_scale = _env_float("V2X_CARLA_OVERLAP_BBOX_UNCERT_WITH_META_SCALE", 0.58)
    ego_scale = _env_float("V2X_CARLA_OVERLAP_BBOX_UNCERT_EGO_SCALE", 0.85)
    heavy_scale = _env_float("V2X_CARLA_OVERLAP_BBOX_UNCERT_HEAVY_SCALE", 1.10)
    two_wheel_scale = _env_float("V2X_CARLA_OVERLAP_BBOX_UNCERT_TWO_WHEEL_SCALE", 1.25)
    use_meta_dims = _env_int("V2X_CARLA_OVERLAP_USE_META_DIMS", 0, minimum=0, maximum=1) == 1
    role = str(track.get("role", "")).strip().lower()
    obj = str(track.get("obj_type", "")).strip().lower()
    meta_len = _safe_float(track.get("length"), float("nan"))
    meta_wid = _safe_float(track.get("width"), float("nan"))
    has_meta_dims = (
        math.isfinite(meta_len)
        and math.isfinite(meta_wid)
        and 1.6 <= float(meta_len) <= 28.0
        and 0.8 <= float(meta_wid) <= 4.5
    )
    scale = 1.0
    if bool(use_meta_dims) and bool(has_meta_dims):
        scale *= float(with_meta_scale)
    if role == "ego":
        scale *= float(ego_scale)
    if ("bus" in obj) or ("truck" in obj) or ("concrete" in obj):
        scale *= float(heavy_scale)
    if ("motor" in obj) or ("bike" in obj) or ("bicycle" in obj) or ("scooter" in obj):
        scale *= float(two_wheel_scale)
    rel_term = float(rel) * 0.5 * (max(1.0, float(length_m)) + max(0.6, float(width_m)))
    unc = (float(base) + float(rel_term)) * float(scale)
    return float(max(float(min_m), min(float(max_m), float(unc))))


def _obb_overlap_penetration_xyyaw(
    x1: float,
    y1: float,
    yaw1_deg: float,
    len1: float,
    wid1: float,
    x2: float,
    y2: float,
    yaw2_deg: float,
    len2: float,
    wid2: float,
) -> float:
    h1 = 0.5 * float(max(0.2, len1))
    w1 = 0.5 * float(max(0.2, wid1))
    h2 = 0.5 * float(max(0.2, len2))
    w2 = 0.5 * float(max(0.2, wid2))

    yaw1 = math.radians(float(yaw1_deg))
    yaw2 = math.radians(float(yaw2_deg))
    f1 = (math.cos(yaw1), math.sin(yaw1))
    r1 = (-math.sin(yaw1), math.cos(yaw1))
    f2 = (math.cos(yaw2), math.sin(yaw2))
    r2 = (-math.sin(yaw2), math.cos(yaw2))
    axes = (f1, r1, f2, r2)

    tx = float(x2) - float(x1)
    ty = float(y2) - float(y1)
    min_pen = float("inf")
    for ax, ay in axes:
        tproj = abs(tx * ax + ty * ay)
        r1p = h1 * abs(f1[0] * ax + f1[1] * ay) + w1 * abs(r1[0] * ax + r1[1] * ay)
        r2p = h2 * abs(f2[0] * ax + f2[1] * ay) + w2 * abs(r2[0] * ax + r2[1] * ay)
        pen = float(r1p + r2p - tproj)
        if pen <= 0.0:
            return 0.0
        if pen < min_pen:
            min_pen = pen
    return float(min_pen if math.isfinite(min_pen) else 0.0)


def _reduce_carla_overlap_with_parked_guidance(
    tracks: List[Dict[str, object]],
    carla_context: Optional[Dict[str, object]],
    scenario_name: str = "",
    verbose: bool = False,
) -> Dict[str, object]:
    report: Dict[str, object] = {
        "enabled": bool(_env_int("V2X_CARLA_OVERLAP_REDUCE_ENABLED", 1, minimum=0, maximum=1) == 1),
        "scenario": str(scenario_name),
        "parked_tracks": 0,
        "quasi_parked_tracks": 0,
        "moving_tracks": 0,
        "parked_nudged_tracks": 0,
        "parked_nudged_frames": 0,
        "parked_micro_nudged_tracks": 0,
        "parked_micro_nudged_frames": 0,
        "moving_pair_nudged_tracks": 0,
        "moving_pair_nudged_frames": 0,
        "raw_fallback_recovered_tracks": 0,
        "raw_fallback_recovered_frames": 0,
        "parked_invariant_tracks": 0,
        "parked_invariant_frames": 0,
        "parked_invariant_edge_tracks": 0,
        "parked_invariant_centerline_tracks": 0,
        "parked_outer_edge_proven_tracks": 0,
        "parked_outer_edge_rejected_tracks": 0,
        "parked_outer_edge_reject_reasons": {},
        "parked_outermost_tiebreak_tracks": 0,
        "parked_overlap_conflict_edge_tracks": 0,
        "parked_inner_to_outer_centerline_tracks": 0,
        "residual_pruned_tracks": 0,
        "residual_pruned_ids": [],
        "parked_removed_tracks": 0,
        "parked_removed_frames": 0,
        "parked_removed_ids": [],
        "ego_actor_dup_removed_tracks": 0,
        "ego_actor_dup_removed_ids": [],
        "parked_dup_removed_tracks": 0,
        "parked_dup_removed_ids": [],
        "blocker_tracks": 0,
        "runs_considered": 0,
        "runs_adjusted": 0,
        "frames_adjusted": 0,
        "overlap_pen_before": 0.0,
        "overlap_pen_after": 0.0,
        "adjusted_track_ids": [],
    }
    if not bool(report["enabled"]):
        report["reason"] = "disabled_by_flag"
        return report
    if not carla_context or not bool(carla_context.get("enabled", False)):
        report["reason"] = "carla_projection_disabled"
        return report
    if not isinstance(tracks, list) or len(tracks) < 2:
        report["reason"] = "insufficient_tracks"
        return report

    dt_key = max(0.05, _env_float("V2X_CARLA_OVERLAP_DT_KEY", 0.1))
    inv_dt = 1.0 / float(dt_key)
    min_pen_thresh = _env_float("V2X_CARLA_OVERLAP_MIN_PEN_M", 0.08)
    min_speed_mps = _env_float("V2X_CARLA_OVERLAP_MIN_MOVING_SPEED_MPS", 0.25)
    max_shift_m = _env_float("V2X_CARLA_OVERLAP_MAX_SHIFT_M", 2.4)
    max_mean_shift_m = _env_float("V2X_CARLA_OVERLAP_MAX_MEAN_SHIFT_M", 1.35)
    cost_slack_per_frame = _env_float("V2X_CARLA_OVERLAP_COST_SLACK_PER_FRAME", 0.55)
    overlap_weight = _env_float("V2X_CARLA_OVERLAP_WEIGHT", 28.0)
    shift_weight = _env_float("V2X_CARLA_OVERLAP_SHIFT_WEIGHT", 0.85)
    disconnected_penalty = _env_float("V2X_CARLA_OVERLAP_DISCONNECTED_PENALTY", 3.2)
    max_curv_rate_proxy = _env_float("V2X_CARLA_OVERLAP_MAX_CURV_RATE_PROXY", 6.5)
    max_curv_rate_proxy_delta = _env_float("V2X_CARLA_OVERLAP_MAX_CURV_RATE_DELTA", 1.6)
    min_gain_abs = _env_float("V2X_CARLA_OVERLAP_MIN_GAIN_ABS", 0.14)
    min_gain_ratio = _env_float("V2X_CARLA_OVERLAP_MIN_GAIN_RATIO", 0.22)
    max_gap_between_overlap_frames = _env_int("V2X_CARLA_OVERLAP_MAX_GAP_FRAMES", 1, minimum=0, maximum=4)
    candidate_top_k = _env_int("V2X_CARLA_OVERLAP_CANDIDATE_TOP_K", 24, minimum=8, maximum=64)
    opposite_reject_deg = _env_float("V2X_CARLA_OPPOSITE_REJECT_DEG", 170.0)
    max_runs_per_track = _env_int("V2X_CARLA_OVERLAP_MAX_RUNS_PER_TRACK", 24, minimum=1, maximum=128)
    skip_transition_runs = _env_int("V2X_CARLA_OVERLAP_SKIP_LINE_TRANSITION_RUNS", 1, minimum=0, maximum=1) == 1
    blocker_stationary_speed_mps = _env_float("V2X_CARLA_OVERLAP_BLOCKER_STATIONARY_SPEED_MPS", 1.1)
    actor_divergence_trigger_m = _env_float("V2X_CARLA_OVERLAP_ACTOR_DIVERGENCE_TRIGGER_M", 1.25)
    divergence_pair_margin_m = _env_float("V2X_CARLA_OVERLAP_PAIR_DIVERGENCE_MARGIN_M", 0.18)
    raw_overlap_keep_ratio = _env_float("V2X_CARLA_OVERLAP_RAW_KEEP_RATIO", 0.72)
    raw_overlap_margin_m = _env_float("V2X_CARLA_OVERLAP_RAW_MARGIN_M", 0.04)
    raw_keep_div_medium_m = _env_float("V2X_CARLA_OVERLAP_RAW_KEEP_DIV_MEDIUM_M", 0.45)
    raw_keep_div_strong_m = _env_float("V2X_CARLA_OVERLAP_RAW_KEEP_DIV_STRONG_M", 0.85)
    raw_keep_ratio_medium_scale = _env_float("V2X_CARLA_OVERLAP_RAW_KEEP_MEDIUM_SCALE", 0.65)
    raw_keep_ratio_strong_scale = _env_float("V2X_CARLA_OVERLAP_RAW_KEEP_STRONG_SCALE", 0.42)
    raw_keep_ratio_parked_scale = _env_float("V2X_CARLA_OVERLAP_RAW_KEEP_PARKED_SCALE", 0.75)
    raw_fidelity_weight = _env_float("V2X_CARLA_OVERLAP_RAW_FIDELITY_WEIGHT", 0.65)
    raw_fallback_min_divergence_m = _env_float("V2X_CARLA_OVERLAP_RAW_FALLBACK_MIN_DIVERGENCE_M", 1.4)
    raw_fallback_max_shift_m = _env_float("V2X_CARLA_OVERLAP_RAW_FALLBACK_MAX_SHIFT_M", 3.6)
    raw_fallback_max_mean_shift_m = _env_float("V2X_CARLA_OVERLAP_RAW_FALLBACK_MAX_MEAN_SHIFT_M", 2.1)
    raw_fallback_shift_weight_scale = _env_float("V2X_CARLA_OVERLAP_RAW_FALLBACK_SHIFT_WEIGHT_SCALE", 0.35)
    raw_fallback_yaw_smooth = _env_int("V2X_CARLA_OVERLAP_RAW_FALLBACK_SMOOTH_YAW", 1, minimum=0, maximum=1) == 1
    raw_fallback_yaw_max_dev_deg = _env_float("V2X_CARLA_OVERLAP_RAW_FALLBACK_YAW_MAX_DEV_DEG", 14.0)
    raw_fallback_reassign_line = _env_int("V2X_CARLA_OVERLAP_RAW_FALLBACK_REASSIGN_LINE", 1, minimum=0, maximum=1) == 1
    raw_fallback_prefer_cbcli = _env_int("V2X_CARLA_OVERLAP_RAW_FALLBACK_PREFER_CBCLI", 1, minimum=0, maximum=1) == 1
    raw_fallback_line_max_dist_m = _env_float("V2X_CARLA_OVERLAP_RAW_FALLBACK_LINE_MAX_DIST_M", 4.0)
    raw_fallback_line_reject_wrong_way = (
        _env_int("V2X_CARLA_OVERLAP_RAW_FALLBACK_LINE_REJECT_WRONG_WAY", 1, minimum=0, maximum=1) == 1
    )
    frame_raw_restore_enabled = _env_int("V2X_CARLA_OVERLAP_FRAME_RAW_RESTORE_ENABLED", 1, minimum=0, maximum=1) == 1
    frame_raw_restore_allow_intersection = _env_int("V2X_CARLA_OVERLAP_FRAME_RAW_RESTORE_INTERSECTION", 0, minimum=0, maximum=1) == 1
    frame_raw_restore_min_gain = _env_float("V2X_CARLA_OVERLAP_FRAME_RAW_RESTORE_MIN_GAIN", 0.02)
    frame_raw_restore_max_shift_m = _env_float("V2X_CARLA_OVERLAP_FRAME_RAW_RESTORE_MAX_SHIFT_M", 5.5)
    frame_raw_restore_min_shift_m = _env_float("V2X_CARLA_OVERLAP_FRAME_RAW_RESTORE_MIN_SHIFT_M", 0.03)
    frame_raw_restore_speed_gate_mps = _env_float("V2X_CARLA_OVERLAP_FRAME_RAW_RESTORE_SPEED_GATE_MPS", 0.35)
    frame_raw_restore_max_step_m = _env_float("V2X_CARLA_OVERLAP_FRAME_RAW_RESTORE_MAX_STEP_M", 2.4)
    frame_raw_restore_step_ratio = _env_float("V2X_CARLA_OVERLAP_FRAME_RAW_RESTORE_STEP_RATIO", 2.8)
    frame_raw_restore_pair_min_pen_m = _env_float("V2X_CARLA_OVERLAP_FRAME_RAW_RESTORE_PAIR_MIN_PEN_M", 0.10)
    frame_raw_restore_pair_raw_ignore_m = _env_float("V2X_CARLA_OVERLAP_FRAME_RAW_RESTORE_PAIR_RAW_IGNORE_M", 0.03)
    # Legacy parked overlap nudges were a major source of between-lane placement
    # and jitter. Keep available behind env flags, but disabled by default.
    nudge_parked_enabled = _env_int("V2X_CARLA_OVERLAP_NUDGE_PARKED_ENABLED", 0, minimum=0, maximum=1) == 1
    nudge_parked_min_total_pen_m = _env_float("V2X_CARLA_OVERLAP_NUDGE_PARKED_MIN_TOTAL_PEN_M", 0.22)
    nudge_parked_min_pair_pen_m = _env_float("V2X_CARLA_OVERLAP_NUDGE_PARKED_MIN_PAIR_PEN_M", 0.04)
    nudge_parked_max_pair_pen_m = _env_float("V2X_CARLA_OVERLAP_NUDGE_PARKED_MAX_PAIR_PEN_M", 0.90)
    nudge_parked_max_shift_m = _env_float("V2X_CARLA_OVERLAP_NUDGE_PARKED_MAX_SHIFT_M", 0.75)
    nudge_parked_min_gain_m = _env_float("V2X_CARLA_OVERLAP_NUDGE_PARKED_MIN_GAIN_M", 0.10)
    nudge_parked_raw_err_med_cap_m = _env_float("V2X_CARLA_OVERLAP_NUDGE_PARKED_RAW_ERR_MED_CAP_M", 4.2)
    micro_nudge_parked_enabled = _env_int("V2X_CARLA_OVERLAP_MICRO_NUDGE_PARKED_ENABLED", 0, minimum=0, maximum=1) == 1
    micro_nudge_pair_pen_min_m = _env_float("V2X_CARLA_OVERLAP_MICRO_NUDGE_PAIR_PEN_MIN_M", 0.03)
    micro_nudge_pair_pen_max_m = _env_float("V2X_CARLA_OVERLAP_MICRO_NUDGE_PAIR_PEN_MAX_M", 0.30)
    micro_nudge_total_pen_min_m = _env_float("V2X_CARLA_OVERLAP_MICRO_NUDGE_TOTAL_PEN_MIN_M", 0.18)
    micro_nudge_max_shift_m = _env_float("V2X_CARLA_OVERLAP_MICRO_NUDGE_MAX_SHIFT_M", 0.28)
    micro_nudge_min_gain_m = _env_float("V2X_CARLA_OVERLAP_MICRO_NUDGE_MIN_GAIN_M", 0.05)
    micro_nudge_raw_err_med_cap_m = _env_float("V2X_CARLA_OVERLAP_MICRO_NUDGE_RAW_ERR_MED_CAP_M", 1.50)
    micro_nudge_raw_err_add_cap_m = _env_float("V2X_CARLA_OVERLAP_MICRO_NUDGE_RAW_ERR_ADD_CAP_M", 0.18)
    micro_nudge_max_tracks = _env_int("V2X_CARLA_OVERLAP_MICRO_NUDGE_MAX_TRACKS", 24, minimum=1, maximum=200)
    micro_nudge_include_quasi = _env_int("V2X_CARLA_OVERLAP_MICRO_NUDGE_INCLUDE_QUASI", 1, minimum=0, maximum=1) == 1
    moving_pair_nudge_enabled = _env_int("V2X_CARLA_OVERLAP_MOVING_PAIR_NUDGE_ENABLED", 1, minimum=0, maximum=1) == 1
    moving_pair_nudge_min_frames = _env_int("V2X_CARLA_OVERLAP_MOVING_PAIR_NUDGE_MIN_FRAMES", 8, minimum=3, maximum=300)
    moving_pair_nudge_min_pair_pen_m = _env_float("V2X_CARLA_OVERLAP_MOVING_PAIR_NUDGE_MIN_PAIR_PEN_M", 0.12)
    moving_pair_nudge_min_gain_m = _env_float("V2X_CARLA_OVERLAP_MOVING_PAIR_NUDGE_MIN_GAIN_M", 0.18)
    moving_pair_nudge_max_shift_m = _env_float("V2X_CARLA_OVERLAP_MOVING_PAIR_NUDGE_MAX_SHIFT_M", 0.95)
    moving_pair_nudge_raw_err_adv_m = _env_float("V2X_CARLA_OVERLAP_MOVING_PAIR_RAW_ERR_ADV_M", 0.25)
    moving_pair_nudge_raw_err_add_cap_m = _env_float("V2X_CARLA_OVERLAP_MOVING_PAIR_RAW_ERR_ADD_CAP_M", 0.95)
    moving_pair_nudge_step_ratio = _env_float("V2X_CARLA_OVERLAP_MOVING_PAIR_STEP_RATIO", 2.6)
    moving_pair_nudge_max_step_m = _env_float("V2X_CARLA_OVERLAP_MOVING_PAIR_MAX_STEP_M", 2.2)
    moving_pair_nudge_blend_radius = _env_int("V2X_CARLA_OVERLAP_MOVING_PAIR_BLEND_RADIUS", 3, minimum=0, maximum=6)
    moving_pair_nudge_max_tracks = _env_int("V2X_CARLA_OVERLAP_MOVING_PAIR_MAX_TRACKS", 24, minimum=1, maximum=200)
    moving_pair_nudge_min_spacing_adv_m = _env_float("V2X_CARLA_OVERLAP_MOVING_PAIR_MIN_SPACING_ADV_M", 0.55)
    bbox_uncert_scale = _env_float("V2X_CARLA_OVERLAP_BBOX_UNCERT_PAIR_SCALE", 0.25)
    bbox_uncert_pair_cap_m = _env_float("V2X_CARLA_OVERLAP_BBOX_UNCERT_PAIR_CAP_M", 0.08)
    remove_obstructing_parked_enabled = _env_int("V2X_CARLA_OVERLAP_REMOVE_OBSTRUCTING_PARKED_ENABLED", 0, minimum=0, maximum=1) == 1
    remove_obstructing_parked_min_overlap_frames = _env_int("V2X_CARLA_OVERLAP_REMOVE_OBSTRUCTING_PARKED_MIN_FRAMES", 10, minimum=2, maximum=500)
    remove_obstructing_parked_min_overlap_ratio = _env_float("V2X_CARLA_OVERLAP_REMOVE_OBSTRUCTING_PARKED_MIN_RATIO", 0.08)
    remove_obstructing_parked_min_pen_sum_m = _env_float("V2X_CARLA_OVERLAP_REMOVE_OBSTRUCTING_PARKED_MIN_PEN_SUM_M", 2.2)
    remove_obstructing_parked_min_pair_pen_m = _env_float("V2X_CARLA_OVERLAP_REMOVE_OBSTRUCTING_PARKED_MIN_PAIR_PEN_M", 0.09)
    remove_obstructing_parked_min_raw_err_med_m = _env_float("V2X_CARLA_OVERLAP_REMOVE_OBSTRUCTING_PARKED_MIN_RAW_ERR_MED_M", 0.45)
    remove_obstructing_parked_ratio_override = _env_float("V2X_CARLA_OVERLAP_REMOVE_OBSTRUCTING_PARKED_RATIO_OVERRIDE", 0.12)
    remove_obstructing_parked_hard_overlap_frames = _env_int("V2X_CARLA_OVERLAP_REMOVE_OBSTRUCTING_PARKED_HARD_FRAMES", 10, minimum=2, maximum=2000)
    remove_obstructing_parked_stationary_net_max_m = _env_float(
        "V2X_CARLA_OVERLAP_REMOVE_OBSTRUCTING_PARKED_STATIONARY_NET_MAX_M",
        1.8,
    )
    remove_obstructing_parked_max_fraction = _env_float("V2X_CARLA_OVERLAP_REMOVE_OBSTRUCTING_PARKED_MAX_FRACTION", 0.75)
    remove_obstructing_parked_max_count = _env_int("V2X_CARLA_OVERLAP_REMOVE_OBSTRUCTING_PARKED_MAX_COUNT", 40, minimum=1, maximum=200)
    remove_obstructing_parked_net_bonus_m = _env_float("V2X_CARLA_OVERLAP_REMOVE_OBSTRUCTING_PARKED_NET_BONUS_M", 2.2)
    ego_actor_dup_prune_enabled = _env_int("V2X_CARLA_OVERLAP_EGO_ACTOR_DUP_PRUNE_ENABLED", 1, minimum=0, maximum=1) == 1
    ego_actor_dup_min_common_frames = _env_int("V2X_CARLA_OVERLAP_EGO_ACTOR_DUP_MIN_COMMON_FRAMES", 12, minimum=3, maximum=400)
    ego_actor_dup_min_ratio_actor = _env_float("V2X_CARLA_OVERLAP_EGO_ACTOR_DUP_MIN_RATIO_ACTOR", 0.35)
    ego_actor_dup_min_ratio_ego = _env_float("V2X_CARLA_OVERLAP_EGO_ACTOR_DUP_MIN_RATIO_EGO", 0.12)
    ego_actor_dup_max_cdist_p90_m = _env_float("V2X_CARLA_OVERLAP_EGO_ACTOR_DUP_MAX_CDIST_P90_M", 1.10)
    ego_actor_dup_max_rawdist_p90_m = _env_float("V2X_CARLA_OVERLAP_EGO_ACTOR_DUP_MAX_RAWDIST_P90_M", 1.30)
    ego_actor_dup_max_yaw_med_deg = _env_float("V2X_CARLA_OVERLAP_EGO_ACTOR_DUP_MAX_YAW_MED_DEG", 12.0)
    parked_dup_prune_enabled = _env_int("V2X_CARLA_OVERLAP_PARKED_DUP_PRUNE_ENABLED", 1, minimum=0, maximum=1) == 1
    parked_dup_min_common_frames = _env_int("V2X_CARLA_OVERLAP_PARKED_DUP_MIN_COMMON_FRAMES", 10, minimum=3, maximum=400)
    parked_dup_min_ratio_each = _env_float("V2X_CARLA_OVERLAP_PARKED_DUP_MIN_RATIO_EACH", 0.40)
    parked_dup_max_rawdist_p90_m = _env_float("V2X_CARLA_OVERLAP_PARKED_DUP_MAX_RAWDIST_P90_M", 1.20)
    parked_dup_max_cdist_p90_m = _env_float("V2X_CARLA_OVERLAP_PARKED_DUP_MAX_CDIST_P90_M", 1.00)
    residual_prune_enabled = _env_int("V2X_CARLA_OVERLAP_RESIDUAL_PRUNE_ENABLED", 0, minimum=0, maximum=1) == 1
    residual_prune_pair_pen_min_m = _env_float("V2X_CARLA_OVERLAP_RESIDUAL_PAIR_PEN_MIN_M", 0.10)
    residual_prune_raw_pen_ignore_m = _env_float("V2X_CARLA_OVERLAP_RESIDUAL_RAW_PEN_IGNORE_M", 0.08)
    residual_prune_speed_gate_mps = _env_float("V2X_CARLA_OVERLAP_RESIDUAL_SPEED_GATE_MPS", 0.45)
    residual_prune_min_events = _env_int("V2X_CARLA_OVERLAP_RESIDUAL_MIN_EVENTS", 1, minimum=1, maximum=500)
    residual_prune_max_net_disp_m = _env_float("V2X_CARLA_OVERLAP_RESIDUAL_MAX_NET_DISP_M", 260.0)
    residual_prune_nonstationary_max_net_disp_m = _env_float(
        "V2X_CARLA_OVERLAP_RESIDUAL_NONSTATIONARY_MAX_NET_DISP_M", 40.0
    )
    residual_prune_nonstationary_max_avg_speed_mps = _env_float(
        "V2X_CARLA_OVERLAP_RESIDUAL_NONSTATIONARY_MAX_AVG_SPEED_MPS", 4.0
    )
    residual_prune_nonstationary_max_p95_step_m = _env_float(
        "V2X_CARLA_OVERLAP_RESIDUAL_NONSTATIONARY_MAX_P95_STEP_M", 1.0
    )
    residual_prune_nonstationary_max_sustain_ratio = _env_float(
        "V2X_CARLA_OVERLAP_RESIDUAL_NONSTATIONARY_MAX_SUSTAIN_RATIO", 0.90
    )
    residual_prune_max_count = _env_int("V2X_CARLA_OVERLAP_RESIDUAL_MAX_COUNT", 64, minimum=0, maximum=200)
    residual_prune_max_fraction = _env_float("V2X_CARLA_OVERLAP_RESIDUAL_MAX_FRACTION", 1.0)
    residual_prune_target_events = _env_int("V2X_CARLA_OVERLAP_RESIDUAL_TARGET_EVENTS", 0, minimum=0, maximum=5000)
    raw_fallback_recover_enabled = _env_int("V2X_CARLA_RAW_FALLBACK_RECOVER_ENABLED", 1, minimum=0, maximum=1) == 1
    raw_fallback_recover_max_dist_m = _env_float("V2X_CARLA_RAW_FALLBACK_RECOVER_MAX_DIST_M", 7.5)
    raw_fallback_recover_jump_floor_m = _env_float("V2X_CARLA_RAW_FALLBACK_RECOVER_JUMP_FLOOR_M", 2.8)
    raw_fallback_recover_jump_ratio = _env_float("V2X_CARLA_RAW_FALLBACK_RECOVER_JUMP_RATIO", 2.8)
    parked_invariant_enabled = _env_int("V2X_CARLA_PARKED_INVARIANT_ENABLED", 1, minimum=0, maximum=1) == 1
    parked_invariant_sample_stride = _env_int("V2X_CARLA_PARKED_INVARIANT_SAMPLE_STRIDE", 2, minimum=1, maximum=12)
    parked_invariant_edge_offset_m = _env_float("V2X_CARLA_PARKED_INVARIANT_EDGE_OFFSET_M", 1.55)
    parked_invariant_edge_proof_enabled = _env_int("V2X_CARLA_PARKED_EDGE_PROOF_ENABLED", 1, minimum=0, maximum=1) == 1
    parked_invariant_edge_proof_nearest_med_max_m = _env_float("V2X_CARLA_PARKED_EDGE_PROOF_NEAREST_MED_MAX_M", 2.8)
    parked_invariant_edge_proof_sample_max_dist_m = _env_float("V2X_CARLA_PARKED_EDGE_PROOF_SAMPLE_MAX_DIST_M", 3.6)
    parked_invariant_edge_proof_family_top_k = _env_int("V2X_CARLA_PARKED_EDGE_PROOF_FAMILY_TOP_K", 12, minimum=4, maximum=48)
    parked_invariant_edge_proof_family_dist_max_m = _env_float("V2X_CARLA_PARKED_EDGE_PROOF_FAMILY_DIST_MAX_M", 10.0)
    parked_invariant_edge_proof_parallel_max_deg = _env_float("V2X_CARLA_PARKED_EDGE_PROOF_PARALLEL_MAX_DEG", 24.0)
    parked_invariant_edge_proof_boundary_tol_m = _env_float("V2X_CARLA_PARKED_EDGE_PROOF_BOUNDARY_TOL_M", 0.35)
    parked_invariant_edge_proof_min_other_sep_m = _env_float("V2X_CARLA_PARKED_EDGE_PROOF_MIN_OTHER_SEP_M", 1.0)
    parked_invariant_edge_proof_raw_sign_consistency_min = _env_float(
        "V2X_CARLA_PARKED_EDGE_PROOF_RAW_SIGN_CONSISTENCY_MIN",
        0.98,
    )
    parked_invariant_edge_proof_raw_abs_med_min_m = _env_float("V2X_CARLA_PARKED_EDGE_PROOF_RAW_ABS_MED_MIN_M", 1.0)
    parked_invariant_static_net_max_m = _env_float("V2X_CARLA_PARKED_INVARIANT_STATIC_NET_MAX_M", 2.3)
    parked_invariant_static_p95_step_max_m = _env_float("V2X_CARLA_PARKED_INVARIANT_STATIC_P95_STEP_MAX_M", 0.55)
    parked_invariant_far_static_enabled = _env_int("V2X_CARLA_PARKED_INVARIANT_FAR_STATIC_ENABLED", 1, minimum=0, maximum=1) == 1
    parked_invariant_far_static_net_max_m = _env_float("V2X_CARLA_PARKED_INVARIANT_FAR_STATIC_NET_MAX_M", 8.0)
    parked_invariant_far_static_speed_max_mps = _env_float("V2X_CARLA_PARKED_INVARIANT_FAR_STATIC_SPEED_MAX_MPS", 2.2)
    parked_invariant_far_static_lane_med_min_m = _env_float("V2X_CARLA_PARKED_INVARIANT_FAR_STATIC_LANE_MED_MIN_M", 2.8)
    parked_invariant_far_static_lane_iqr_max_m = _env_float("V2X_CARLA_PARKED_INVARIANT_FAR_STATIC_LANE_IQR_MAX_M", 1.4)
    parked_outermost_tiebreak_enabled = _env_int(
        "V2X_CARLA_PARKED_OUTERMOST_TIEBREAK_ENABLED",
        1,
        minimum=0,
        maximum=1,
    ) == 1
    parked_outermost_tiebreak_count_gap_max = _env_int(
        "V2X_CARLA_PARKED_OUTERMOST_TIEBREAK_COUNT_GAP_MAX",
        3,
        minimum=0,
        maximum=12,
    )
    parked_outermost_tiebreak_query_gap_max_m = _env_float(
        "V2X_CARLA_PARKED_OUTERMOST_TIEBREAK_QUERY_GAP_MAX_M",
        0.50,
    )
    parked_outermost_tiebreak_raw_gap_max_m = _env_float(
        "V2X_CARLA_PARKED_OUTERMOST_TIEBREAK_RAW_GAP_MAX_M",
        0.45,
    )
    parked_outermost_tiebreak_raw_gain_min_m = _env_float(
        "V2X_CARLA_PARKED_OUTERMOST_TIEBREAK_RAW_GAIN_MIN_M",
        0.65,
    )
    # Generous gap tolerance for topology-only outermost enforcement (FC4):
    # allow committing to the outermost lane even when it is up to this many
    # metres farther from the raw actor position than the current inner lane.
    parked_outermost_force_raw_gap_max_m = _env_float(
        "V2X_CARLA_PARKED_OUTERMOST_FORCE_RAW_GAP_MAX_M",
        20.0,
    )
    parked_overlap_conflict_min_pair_pen_m = _env_float(
        "V2X_CARLA_PARKED_OVERLAP_CONFLICT_MIN_PAIR_PEN_M",
        0.10,
    )
    parked_overlap_conflict_min_frames = _env_int(
        "V2X_CARLA_PARKED_OVERLAP_CONFLICT_MIN_FRAMES",
        5,
        minimum=1,
        maximum=500,
    )
    parked_overlap_conflict_min_ratio = _env_float(
        "V2X_CARLA_PARKED_OVERLAP_CONFLICT_MIN_RATIO",
        0.08,
    )
    parked_overlap_conflict_min_pen_sum_m = _env_float(
        "V2X_CARLA_PARKED_OVERLAP_CONFLICT_MIN_PEN_SUM_M",
        1.8,
    )
    parked_overlap_conflict_force_same_ratio_min = _env_float(
        "V2X_CARLA_PARKED_OVERLAP_CONFLICT_FORCE_SAME_RATIO_MIN",
        0.05,
    )
    parked_overlap_conflict_relaxed_nearest_med_max_m = _env_float(
        "V2X_CARLA_PARKED_OVERLAP_CONFLICT_RELAXED_NEAREST_MED_MAX_M",
        9.5,
    )
    parked_overlap_conflict_relaxed_sample_max_dist_m = _env_float(
        "V2X_CARLA_PARKED_OVERLAP_CONFLICT_RELAXED_SAMPLE_MAX_DIST_M",
        9.0,
    )
    parked_overlap_conflict_relaxed_raw_sign_min = _env_float(
        "V2X_CARLA_PARKED_OVERLAP_CONFLICT_RELAXED_RAW_SIGN_MIN",
        0.55,
    )
    parked_overlap_conflict_relaxed_raw_abs_med_min_m = _env_float(
        "V2X_CARLA_PARKED_OVERLAP_CONFLICT_RELAXED_RAW_ABS_MED_MIN_M",
        0.45,
    )
    parked_outermost_far_edge_raw_abs_min_m = _env_float(
        "V2X_CARLA_PARKED_OUTERMOST_FAR_EDGE_RAW_ABS_MIN_M",
        4.0,
    )
    parked_outermost_far_edge_sign_ratio_min = _env_float(
        "V2X_CARLA_PARKED_OUTERMOST_FAR_EDGE_SIGN_RATIO_MIN",
        0.90,
    )
    parked_inner_to_outer_enabled = _env_int(
        "V2X_CARLA_PARKED_INNER_TO_OUTER_CENTERLINE_ENABLED",
        1,
        minimum=0,
        maximum=1,
    ) == 1
    parked_inner_to_outer_raw_gain_min_m = _env_float(
        "V2X_CARLA_PARKED_INNER_TO_OUTER_RAW_GAIN_MIN_M",
        0.65,
    )
    parked_duplicate_prune_enabled = _env_int(
        "V2X_CARLA_PARKED_DUPLICATE_PRUNE_ENABLED",
        1,
        minimum=0,
        maximum=1,
    ) == 1
    parked_duplicate_prune_min_overlap_frames = _env_int(
        "V2X_CARLA_PARKED_DUPLICATE_PRUNE_MIN_OVERLAP_FRAMES",
        8,
        minimum=2,
        maximum=500,
    )
    parked_duplicate_prune_min_overlap_ratio_each = _env_float(
        "V2X_CARLA_PARKED_DUPLICATE_PRUNE_MIN_OVERLAP_RATIO_EACH",
        0.70,
    )
    parked_duplicate_prune_min_pair_pen_m = _env_float(
        "V2X_CARLA_PARKED_DUPLICATE_PRUNE_MIN_PAIR_PEN_M",
        0.20,
    )
    parked_duplicate_prune_min_pen_sum_m = _env_float(
        "V2X_CARLA_PARKED_DUPLICATE_PRUNE_MIN_PEN_SUM_M",
        2.5,
    )
    parked_duplicate_prune_same_line_ratio_min = _env_float(
        "V2X_CARLA_PARKED_DUPLICATE_PRUNE_SAME_LINE_RATIO_MIN",
        0.85,
    )
    debug_track_ids = {
        tok.strip()
        for tok in re.split(r"[,\s]+", str(os.environ.get("V2X_CARLA_OVERLAP_DEBUG_TRACK_IDS", "")).strip())
        if tok.strip()
    }

    map_name = ""
    summary = carla_context.get("summary", {})
    if isinstance(summary, dict):
        map_name = str(summary.get("map_name", ""))
    is_intersection_map = "intersection" in str(map_name).lower()
    if bool(is_intersection_map) and not bool(frame_raw_restore_allow_intersection):
        frame_raw_restore_enabled = False

    track_by_key: Dict[str, Dict[str, object]] = {}
    frames_by_key: Dict[str, List[Dict[str, object]]] = {}
    dims_by_key: Dict[str, Tuple[float, float]] = {}
    bbox_uncert_by_key: Dict[str, float] = {}
    tick_to_index: Dict[str, Dict[int, int]] = {}
    speed_by_key: Dict[str, np.ndarray] = {}
    raw_err_by_key: Dict[str, np.ndarray] = {}
    motion_stats_by_key: Dict[str, Dict[str, float]] = {}
    parked_keys: set = set()
    quasi_parked_keys: set = set()
    moving_keys: set = set()

    for tr in tracks:
        if not isinstance(tr, dict):
            continue
        role = str(tr.get("role", "")).strip().lower()
        if role not in {"ego", "vehicle"}:
            continue
        frames = tr.get("frames", [])
        if not isinstance(frames, list) or len(frames) < 3:
            continue
        key = _carla_track_key(tr)
        track_by_key[key] = tr
        frames_by_key[key] = frames
        dims_by_key[key] = _vehicle_dims_for_overlap(tr)
        dl, dw = dims_by_key[key]
        bbox_uncert_by_key[key] = _vehicle_bbox_uncertainty_m(tr, float(dl), float(dw))
        motion_stats = _track_raw_motion_stats(tr)
        motion_stats_by_key[key] = motion_stats
        parked_like = _is_parked_vehicle_track_for_overlap(tr)
        quasi_parked_like = _is_quasi_parked_vehicle_track_for_overlap(tr)
        if parked_like:
            parked_keys.add(key)
        if quasi_parked_like:
            quasi_parked_keys.add(key)
        else:
            quasi_parked_keys.discard(key)
        if not parked_like:
            moving_keys.add(key)

        idx_map: Dict[int, int] = {}
        spd = np.zeros((len(frames),), dtype=np.float64)
        raw_err = np.zeros((len(frames),), dtype=np.float64)
        for i, fr in enumerate(frames):
            t = _safe_float(fr.get("t"), float(i) * float(dt_key))
            tk = int(round(float(t) * float(inv_dt)))
            if tk not in idx_map:
                idx_map[int(tk)] = int(i)
            rx = _safe_float(fr.get("x"), 0.0)
            ry = _safe_float(fr.get("y"), 0.0)
            cx = _safe_float(fr.get("cx"), rx)
            cy = _safe_float(fr.get("cy"), ry)
            raw_err[i] = float(math.hypot(float(cx) - float(rx), float(cy) - float(ry)))
            if i > 0:
                x0 = _safe_float(frames[i - 1].get("x"), 0.0)
                y0 = _safe_float(frames[i - 1].get("y"), 0.0)
                x1 = _safe_float(fr.get("x"), x0)
                y1 = _safe_float(fr.get("y"), y0)
                t0 = _safe_float(frames[i - 1].get("t"), float(i - 1) * float(dt_key))
                dt = max(5e-2, float(t) - float(t0))
                spd[i] = float(math.hypot(float(x1) - float(x0), float(y1) - float(y0)) / dt)
        tick_to_index[key] = idx_map
        speed_by_key[key] = spd
        raw_err_by_key[key] = raw_err

    ego_actor_dup_removed_ids: List[str] = []
    parked_dup_removed_ids: List[str] = []

    def _pair_alignment_stats(key_a: str, key_b: str) -> Dict[str, float]:
        idx_a = tick_to_index.get(key_a, {})
        idx_b = tick_to_index.get(key_b, {})
        if not idx_a or not idx_b:
            return {"common_n": 0.0}
        common = sorted(set(idx_a.keys()).intersection(idx_b.keys()))
        if not common:
            return {"common_n": 0.0}
        cdist: List[float] = []
        rawdist: List[float] = []
        yawdiff: List[float] = []
        valid_n = 0
        frames_a = frames_by_key.get(key_a, [])
        frames_b = frames_by_key.get(key_b, [])
        for tk in common:
            ia = int(idx_a.get(int(tk), -1))
            ib = int(idx_b.get(int(tk), -1))
            if ia < 0 or ib < 0 or ia >= len(frames_a) or ib >= len(frames_b):
                continue
            fa = frames_a[ia]
            fb = frames_b[ib]
            if not (_frame_has_carla_pose(fa) and _frame_has_carla_pose(fb)):
                continue
            cax = _safe_float(fa.get("cx"), float("nan"))
            cay = _safe_float(fa.get("cy"), float("nan"))
            cbx = _safe_float(fb.get("cx"), float("nan"))
            cby = _safe_float(fb.get("cy"), float("nan"))
            if not (math.isfinite(cax) and math.isfinite(cay) and math.isfinite(cbx) and math.isfinite(cby)):
                continue
            ray = _safe_float(fa.get("cyaw"), _safe_float(fa.get("yaw"), 0.0))
            rby = _safe_float(fb.get("cyaw"), _safe_float(fb.get("yaw"), 0.0))
            rax = _safe_float(fa.get("x"), cax)
            ray0 = _safe_float(fa.get("y"), cay)
            rbx = _safe_float(fb.get("x"), cbx)
            rby0 = _safe_float(fb.get("y"), cby)
            cdist.append(float(math.hypot(float(cax) - float(cbx), float(cay) - float(cby))))
            rawdist.append(float(math.hypot(float(rax) - float(rbx), float(ray0) - float(rby0))))
            yawdiff.append(float(_yaw_abs_diff_deg(float(ray), float(rby))))
            valid_n += 1
        if valid_n <= 0:
            return {"common_n": 0.0}
        arr_c = np.asarray(cdist, dtype=np.float64)
        arr_r = np.asarray(rawdist, dtype=np.float64)
        arr_y = np.asarray(yawdiff, dtype=np.float64)
        return {
            "common_n": float(valid_n),
            "ratio_a": float(valid_n) / max(1.0, float(len(idx_a))),
            "ratio_b": float(valid_n) / max(1.0, float(len(idx_b))),
            "cdist_med": float(np.median(arr_c)),
            "cdist_p90": float(np.percentile(arr_c, 90.0)),
            "rawdist_med": float(np.median(arr_r)),
            "rawdist_p90": float(np.percentile(arr_r, 90.0)),
            "yaw_med": float(np.median(arr_y)),
        }

    drop_keys: set = set()
    if bool(ego_actor_dup_prune_enabled):
        ego_keys = [k for k in track_by_key.keys() if str(k).startswith("ego:")]
        veh_keys = [k for k in track_by_key.keys() if str(k).startswith("vehicle:")]
        for vk in veh_keys:
            if vk in drop_keys:
                continue
            for ek in ego_keys:
                st = _pair_alignment_stats(str(vk), str(ek))
                if float(st.get("common_n", 0.0)) < float(ego_actor_dup_min_common_frames):
                    continue
                if float(st.get("ratio_a", 0.0)) < float(ego_actor_dup_min_ratio_actor):
                    continue
                if float(st.get("ratio_b", 0.0)) < float(ego_actor_dup_min_ratio_ego):
                    continue
                if float(st.get("cdist_p90", 1e9)) > float(ego_actor_dup_max_cdist_p90_m):
                    continue
                if float(st.get("rawdist_p90", 1e9)) > float(ego_actor_dup_max_rawdist_p90_m):
                    continue
                if float(st.get("yaw_med", 1e9)) > float(ego_actor_dup_max_yaw_med_deg):
                    continue
                drop_keys.add(str(vk))
                tr_vk = track_by_key.get(vk, {})
                ego_actor_dup_removed_ids.append(str(tr_vk.get("id", vk)))
                break

    if bool(parked_dup_prune_enabled):
        pk = [k for k in parked_keys if k not in drop_keys]
        dup_pairs: List[Tuple[int, str, str]] = []
        for i in range(len(pk)):
            ka = str(pk[i])
            for j in range(i + 1, len(pk)):
                kb = str(pk[j])
                st = _pair_alignment_stats(ka, kb)
                common_n = int(round(float(st.get("common_n", 0.0))))
                if common_n < int(parked_dup_min_common_frames):
                    continue
                if min(float(st.get("ratio_a", 0.0)), float(st.get("ratio_b", 0.0))) < float(parked_dup_min_ratio_each):
                    continue
                if float(st.get("cdist_p90", 1e9)) > float(parked_dup_max_cdist_p90_m):
                    continue
                if float(st.get("rawdist_p90", 1e9)) > float(parked_dup_max_rawdist_p90_m):
                    continue
                dup_pairs.append((int(common_n), ka, kb))
        dup_pairs.sort(reverse=True)
        for _, ka, kb in dup_pairs:
            if ka in drop_keys or kb in drop_keys:
                continue
            len_a = len(tick_to_index.get(ka, {}))
            len_b = len(tick_to_index.get(kb, {}))
            remove_key = ka if int(len_a) < int(len_b) else kb
            drop_keys.add(str(remove_key))
            tr_rm = track_by_key.get(remove_key, {})
            parked_dup_removed_ids.append(str(tr_rm.get("id", remove_key)))

    if drop_keys:
        tracks[:] = [tr for tr in tracks if _carla_track_key(tr) not in drop_keys]
        for dk in list(drop_keys):
            track_by_key.pop(dk, None)
            frames_by_key.pop(dk, None)
            dims_by_key.pop(dk, None)
            bbox_uncert_by_key.pop(dk, None)
            tick_to_index.pop(dk, None)
            speed_by_key.pop(dk, None)
            raw_err_by_key.pop(dk, None)
            motion_stats_by_key.pop(dk, None)
            parked_keys.discard(dk)
            quasi_parked_keys.discard(dk)
            moving_keys.discard(dk)

    report["ego_actor_dup_removed_tracks"] = int(len(ego_actor_dup_removed_ids))
    report["ego_actor_dup_removed_ids"] = sorted(ego_actor_dup_removed_ids)
    report["parked_dup_removed_tracks"] = int(len(parked_dup_removed_ids))
    report["parked_dup_removed_ids"] = sorted(parked_dup_removed_ids)

    report["parked_tracks"] = int(len(parked_keys))
    report["quasi_parked_tracks"] = int(len(quasi_parked_keys))
    report["moving_tracks"] = int(len(moving_keys))
    if not moving_keys:
        report["reason"] = "missing_moving_tracks"
        return report

    blockers_by_tick: Dict[int, List[Tuple[str, float, float, float, float, float, float, bool]]] = {}
    blocker_keys = set(track_by_key.keys())
    report["blocker_tracks"] = int(len(blocker_keys))
    for key in blocker_keys:
        frames = frames_by_key.get(key, [])
        l, w = dims_by_key.get(key, (4.6, 2.0))
        speeds = speed_by_key.get(key, np.zeros((len(frames),), dtype=np.float64))
        is_parked = bool(key in parked_keys)
        for tk, i in tick_to_index.get(key, {}).items():
            if i < 0 or i >= len(frames):
                continue
            fr = frames[i]
            x = _safe_float(fr.get("cx"), float("nan"))
            y = _safe_float(fr.get("cy"), float("nan"))
            yaw = _safe_float(fr.get("cyaw"), _safe_float(fr.get("yaw"), 0.0))
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            bspd = float(speeds[i]) if i < len(speeds) else 0.0
            blockers_by_tick.setdefault(int(tk), []).append(
                (key, float(x), float(y), float(yaw), float(l), float(w), float(bspd), bool(is_parked))
            )

    if not blockers_by_tick:
        report["reason"] = "no_blockers"
        return report

    carla_successors = carla_context.get("carla_successors", {})
    if not isinstance(carla_successors, dict):
        carla_successors = {}

    def _is_line_connected(a: int, b: int) -> bool:
        if int(a) < 0 or int(b) < 0:
            return False
        if int(a) == int(b):
            return True
        a_succ = carla_successors.get(int(a), set())
        b_succ = carla_successors.get(int(b), set())
        if int(b) in a_succ or int(a) in b_succ:
            return True
        return False

    def _pair_bbox_uncertainty_m(a_key: str, b_key: str) -> float:
        ua = float(bbox_uncert_by_key.get(str(a_key), 0.0))
        ub = float(bbox_uncert_by_key.get(str(b_key), 0.0))
        pair_unc = float(0.5 * (float(ua) + float(ub)) * float(bbox_uncert_scale))
        return float(max(0.0, min(float(bbox_uncert_pair_cap_m), float(pair_unc))))

    def _effective_overlap_pen_m(raw_pen_m: float, a_key: str, b_key: str) -> float:
        if float(raw_pen_m) <= 0.0:
            return 0.0
        return float(max(0.0, float(raw_pen_m) - float(_pair_bbox_uncertainty_m(str(a_key), str(b_key)))))

    def _curv_rate_proxy_xy(points_xy: Sequence[Tuple[float, float]]) -> float:
        if len(points_xy) < 4:
            return 0.0
        arr = np.asarray(points_xy, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return 0.0
        dxy = np.diff(arr, axis=0)
        step = np.linalg.norm(dxy, axis=1)
        if step.size < 3:
            return 0.0
        heading = np.unwrap(np.arctan2(dxy[:, 1], dxy[:, 0]))
        dhead = np.abs(np.diff(heading))
        if dhead.size < 2:
            return float(np.percentile(dhead, 95)) if dhead.size > 0 else 0.0
        step_mid = np.maximum(0.2, 0.5 * (step[1:] + step[:-1]))
        kappa = dhead / step_mid
        if kappa.size < 2:
            return float(np.percentile(kappa, 95)) if kappa.size > 0 else 0.0
        dk = np.abs(np.diff(kappa)) / max(0.05, float(dt_key))
        if dk.size <= 0:
            return 0.0
        return float(np.percentile(dk, 95))

    adjusted_tracks: set = set()
    overlap_pen_before = 0.0
    overlap_pen_after = 0.0
    total_runs_considered = 0
    total_runs_adjusted = 0
    total_frames_adjusted = 0
    frame_raw_restored = 0

    for key in sorted(moving_keys):
        frames = frames_by_key.get(key, [])
        if not isinstance(frames, list) or len(frames) < 3:
            continue
        tr = track_by_key.get(key, {})
        tr_id = str(tr.get("id", key))
        debug_this_track = bool(tr_id in debug_track_ids or key in debug_track_ids)
        l, w = dims_by_key.get(key, (4.6, 2.0))
        speeds = speed_by_key.get(key, np.zeros((len(frames),), dtype=np.float64))
        raw_errs = raw_err_by_key.get(key, np.zeros((len(frames),), dtype=np.float64))
        actor_frames = frames

        def _pair_overlap_excess(
            fi: int,
            tk: int,
            ax: float,
            ay: float,
            ayaw: float,
            b_key: str,
            bx: float,
            by: float,
            byaw: float,
            bl: float,
            bw: float,
            bspd: float,
            b_parked: bool,
        ) -> float:
            actor_div = float(raw_errs[fi]) if fi < len(raw_errs) else 0.0
            if (not bool(b_parked)) and float(bspd) > float(blocker_stationary_speed_mps) and actor_div < float(actor_divergence_trigger_m):
                return 0.0

            b_idx = int(tick_to_index.get(b_key, {}).get(int(tk), -1))
            blocker_div = 0.0
            if b_idx >= 0:
                b_err = raw_err_by_key.get(b_key)
                if isinstance(b_err, np.ndarray) and b_idx < len(b_err):
                    blocker_div = float(b_err[b_idx])
            div_priority = 1.0
            if actor_div + float(divergence_pair_margin_m) < float(blocker_div):
                denom = max(1e-3, float(blocker_div))
                div_priority = max(0.18, min(0.70, float(actor_div) / denom))

            pen = _obb_overlap_penetration_xyyaw(
                x1=float(ax),
                y1=float(ay),
                yaw1_deg=float(ayaw),
                len1=float(l),
                wid1=float(w),
                x2=float(bx),
                y2=float(by),
                yaw2_deg=float(byaw),
                len2=float(bl),
                wid2=float(bw),
            )
            eff_pen = _effective_overlap_pen_m(float(pen), str(key), str(b_key))
            if eff_pen < float(min_pen_thresh):
                return 0.0

            raw_pen = 0.0
            if b_idx >= 0:
                b_frames = frames_by_key.get(b_key, [])
                if b_idx < len(b_frames):
                    afr = actor_frames[fi]
                    bfr = b_frames[b_idx]
                    arx = _safe_float(afr.get("x"), float(ax))
                    ary = _safe_float(afr.get("y"), float(ay))
                    aryaw = _safe_float(afr.get("yaw"), float(ayaw))
                    brx = _safe_float(bfr.get("x"), float(bx))
                    bry = _safe_float(bfr.get("y"), float(by))
                    bryaw = _safe_float(bfr.get("yaw"), float(byaw))
                    raw_pen = _obb_overlap_penetration_xyyaw(
                        x1=float(arx),
                        y1=float(ary),
                        yaw1_deg=float(aryaw),
                        len1=float(l),
                        wid1=float(w),
                        x2=float(brx),
                        y2=float(bry),
                        yaw2_deg=float(bryaw),
                        len2=float(bl),
                        wid2=float(bw),
                    )
                    raw_pen = _effective_overlap_pen_m(float(raw_pen), str(key), str(b_key))
            keep_ratio = float(raw_overlap_keep_ratio)
            div_gap = float(actor_div) - float(blocker_div)
            if div_gap >= float(raw_keep_div_strong_m):
                keep_ratio *= float(raw_keep_ratio_strong_scale)
            elif div_gap >= float(raw_keep_div_medium_m):
                keep_ratio *= float(raw_keep_ratio_medium_scale)
            if bool(b_parked) and div_gap >= float(raw_keep_div_medium_m):
                keep_ratio *= float(raw_keep_ratio_parked_scale)
            keep_ratio = float(max(0.12, min(1.0, keep_ratio)))
            keep_pen = max(float(raw_overlap_margin_m), float(keep_ratio) * float(raw_pen))
            excess_pen = float(eff_pen) - float(keep_pen)
            if excess_pen <= 0.0:
                return 0.0
            return float(excess_pen) * float(div_priority)

        overlap_idx: List[int] = []
        per_frame_overlap_before: Dict[int, float] = {}
        for i, fr in enumerate(frames):
            if not _frame_has_carla_pose(fr):
                continue
            if i < len(speeds) and float(speeds[i]) < float(min_speed_mps):
                continue
            t = _safe_float(fr.get("t"), float(i) * float(dt_key))
            tk = int(round(float(t) * float(inv_dt)))
            blockers = blockers_by_tick.get(int(tk), [])
            if not blockers:
                continue
            x = _safe_float(fr.get("cx"), float("nan"))
            y = _safe_float(fr.get("cy"), float("nan"))
            yaw = _safe_float(fr.get("cyaw"), _safe_float(fr.get("yaw"), 0.0))
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            frame_pen = 0.0
            for b_key, bx, by, byaw, bl, bw, bspd, b_parked in blockers:
                if b_key == key:
                    continue
                center_dist = float(math.hypot(float(x) - float(bx), float(y) - float(by)))
                diag_bound = 0.62 * (math.hypot(float(l), float(w)) + math.hypot(float(bl), float(bw)))
                if center_dist > diag_bound + 0.45:
                    continue
                pen = _pair_overlap_excess(
                    fi=int(i),
                    tk=int(tk),
                    ax=float(x),
                    ay=float(y),
                    ayaw=float(yaw),
                    b_key=str(b_key),
                    bx=float(bx),
                    by=float(by),
                    byaw=float(byaw),
                    bl=float(bl),
                    bw=float(bw),
                    bspd=float(bspd),
                    b_parked=bool(b_parked),
                )
                if pen > 0.0:
                    frame_pen += float(pen)
            if frame_pen > 0.0:
                overlap_idx.append(int(i))
                per_frame_overlap_before[int(i)] = float(frame_pen)
                overlap_pen_before += float(frame_pen)

        if not overlap_idx:
            if debug_this_track:
                print(f"[CARLA_OVERLAP][DEBUG] track={tr_id} no_excess_overlap_frames")
            continue

        runs: List[Tuple[int, int]] = []
        s = overlap_idx[0]
        e = overlap_idx[0]
        for i in overlap_idx[1:]:
            if int(i) <= int(e) + int(max_gap_between_overlap_frames) + 1:
                e = int(i)
            else:
                runs.append((int(s), int(e)))
                s = int(i)
                e = int(i)
        runs.append((int(s), int(e)))

        runs = runs[: max(1, int(max_runs_per_track))]
        total_runs_considered += int(len(runs))

        for run_start, run_end in runs:
            curr_lines = [
                _safe_int(frames[i].get("ccli"), -1)
                for i in range(int(run_start), int(run_end) + 1)
                if _safe_int(frames[i].get("ccli"), -1) >= 0
            ]
            if not curr_lines:
                continue
            run_has_line_transition = any(int(curr_lines[j]) != int(curr_lines[j - 1]) for j in range(1, len(curr_lines)))
            if bool(skip_transition_runs) and bool(run_has_line_transition):
                current_overlap = 0.0
                for fi in range(int(run_start), int(run_end) + 1):
                    current_overlap += float(per_frame_overlap_before.get(int(fi), 0.0))
                overlap_pen_after += float(current_overlap)
                if debug_this_track:
                    print(
                        f"[CARLA_OVERLAP][DEBUG] track={tr_id} run={run_start}-{run_end} "
                        "skip_transition_run"
                    )
                continue
            curr_line = max(set(curr_lines), key=curr_lines.count)
            mid_i = int((int(run_start) + int(run_end)) // 2)
            qx_mid, qy_mid, _ = _frame_query_pose(frames[mid_i])
            candidate_lines: List[int] = []
            seen_lines: set = set()

            def _push_line(li: int) -> None:
                if int(li) < 0 or int(li) in seen_lines:
                    return
                seen_lines.add(int(li))
                candidate_lines.append(int(li))

            _push_line(int(curr_line))
            prev_line = _safe_int(frames[run_start - 1].get("ccli"), -1) if int(run_start) > 0 else -1
            next_line = _safe_int(frames[run_end + 1].get("ccli"), -1) if int(run_end + 1) < len(frames) else -1
            _push_line(int(prev_line))
            _push_line(int(next_line))
            for li in _nearby_carla_lines(carla_context, float(qx_mid), float(qy_mid), top_k=int(candidate_top_k)):
                _push_line(int(li))

            if len(candidate_lines) <= 1:
                if debug_this_track:
                    print(
                        f"[CARLA_OVERLAP][DEBUG] track={tr_id} run={run_start}-{run_end} "
                        f"insufficient_candidates curr_line={curr_line}"
                    )
                continue

            current_cost = 0.0
            current_raw_cost = 0.0
            current_overlap = 0.0
            current_xy_points: List[Tuple[float, float]] = []
            for fi in range(int(run_start), int(run_end) + 1):
                fr = frames[fi]
                if not _frame_has_carla_pose(fr):
                    continue
                qx, qy, qyaw = _frame_query_pose(fr)
                cx = _safe_float(fr.get("cx"), float(qx))
                cy = _safe_float(fr.get("cy"), float(qy))
                cyaw = _safe_float(fr.get("cyaw"), float(qyaw))
                current_xy_points.append((float(cx), float(cy)))
                current_cost += float(math.hypot(float(cx) - float(qx), float(cy) - float(qy)))
                current_cost += 0.018 * float(_yaw_abs_diff_deg(float(cyaw), float(qyaw)))
                rx = _safe_float(fr.get("x"), float(qx))
                ry = _safe_float(fr.get("y"), float(qy))
                current_raw_cost += float(math.hypot(float(cx) - float(rx), float(cy) - float(ry)))
                current_overlap += float(per_frame_overlap_before.get(int(fi), 0.0))
            current_curv_rate_proxy = _curv_rate_proxy_xy(current_xy_points)
            current_total_ref = (
                float(current_cost)
                + float(raw_fidelity_weight) * float(current_raw_cost)
                + float(overlap_weight) * float(current_overlap)
            )

            best_line = int(curr_line)
            best_total = float("inf")
            best_fit_cost = float("inf")
            best_overlap = float("inf")
            best_shift_mean = float("inf")
            best_rows: List[Tuple[int, float, float, float, float]] = []

            for cand_line in candidate_lines:
                cand_rows: List[Tuple[int, float, float, float, float]] = []
                cand_cost = 0.0
                cand_raw_cost = 0.0
                cand_overlap = 0.0
                shift_sum = 0.0
                frame_count = 0
                valid = True
                for fi in range(int(run_start), int(run_end) + 1):
                    fr = frames[fi]
                    if not _frame_has_carla_pose(fr):
                        continue
                    qx, qy, qyaw = _frame_query_pose(fr)
                    cand = _best_projection_on_carla_line(
                        carla_context=carla_context,
                        line_index=int(cand_line),
                        qx=float(qx),
                        qy=float(qy),
                        qyaw=float(qyaw),
                        prefer_reversed=None,
                        enforce_node_direction=True,
                        opposite_reject_deg=float(opposite_reject_deg),
                    )
                    if cand is None:
                        valid = False
                        break
                    proj = cand.get("projection", {})
                    px = _safe_float(proj.get("x"), float("nan"))
                    py = _safe_float(proj.get("y"), float("nan"))
                    pyaw = _safe_float(proj.get("yaw"), float(qyaw))
                    pd = _safe_float(proj.get("dist"), float("inf"))
                    if not (math.isfinite(px) and math.isfinite(py) and math.isfinite(pd)):
                        valid = False
                        break
                    cur_x = _safe_float(fr.get("cx"), float(px))
                    cur_y = _safe_float(fr.get("cy"), float(py))
                    shift = float(math.hypot(float(px) - float(cur_x), float(py) - float(cur_y)))
                    if shift > float(max_shift_m):
                        valid = False
                        break
                    shift_sum += float(shift)
                    cand_cost += float(cand.get("score", pd))
                    rx = _safe_float(fr.get("x"), float(qx))
                    ry = _safe_float(fr.get("y"), float(qy))
                    cand_raw_cost += float(math.hypot(float(px) - float(rx), float(py) - float(ry)))
                    t = _safe_float(fr.get("t"), float(fi) * float(dt_key))
                    tk = int(round(float(t) * float(inv_dt)))
                    blockers = blockers_by_tick.get(int(tk), [])
                    frame_pen = 0.0
                    for b_key, bx, by, byaw, bl, bw, bspd, b_parked in blockers:
                        if b_key == key:
                            continue
                        center_dist = float(math.hypot(float(px) - float(bx), float(py) - float(by)))
                        diag_bound = 0.62 * (math.hypot(float(l), float(w)) + math.hypot(float(bl), float(bw)))
                        if center_dist > diag_bound + 0.45:
                            continue
                        pen = _pair_overlap_excess(
                            fi=int(fi),
                            tk=int(tk),
                            ax=float(px),
                            ay=float(py),
                            ayaw=float(pyaw),
                            b_key=str(b_key),
                            bx=float(bx),
                            by=float(by),
                            byaw=float(byaw),
                            bl=float(bl),
                            bw=float(bw),
                            bspd=float(bspd),
                            b_parked=bool(b_parked),
                        )
                        if pen > 0.0:
                            frame_pen += float(pen)
                    cand_overlap += float(frame_pen)
                    cand_rows.append((int(fi), float(px), float(py), float(pyaw), float(pd)))
                    frame_count += 1

                if not valid or frame_count <= 0:
                    continue

                cand_curv_rate_proxy = _curv_rate_proxy_xy([(float(px), float(py)) for _, px, py, _, _ in cand_rows])
                if (
                    float(cand_curv_rate_proxy) > float(max_curv_rate_proxy)
                    and float(cand_curv_rate_proxy) > float(current_curv_rate_proxy) + float(max_curv_rate_proxy_delta)
                ):
                    continue

                continuity = 0.0
                if int(prev_line) >= 0 and int(cand_line) != int(prev_line):
                    if not _is_line_connected(int(prev_line), int(cand_line)):
                        continuity += float(disconnected_penalty)
                if int(next_line) >= 0 and int(cand_line) != int(next_line):
                    if not _is_line_connected(int(next_line), int(cand_line)):
                        continuity += float(disconnected_penalty)

                shift_mean = float(shift_sum / max(1, frame_count))
                fit_cost = float(cand_cost) + float(raw_fidelity_weight) * float(cand_raw_cost)
                total = (
                    float(fit_cost)
                    + float(overlap_weight) * float(cand_overlap)
                    + float(shift_weight) * float(shift_sum)
                    + float(continuity)
                )
                if total < best_total:
                    best_total = float(total)
                    best_line = int(cand_line)
                    best_fit_cost = float(fit_cost)
                    best_overlap = float(cand_overlap)
                    best_shift_mean = float(shift_mean)
                    best_rows = cand_rows

            # Divergence-gated raw fallback:
            # if snapped poses overlap and are materially divergent from raw, allow
            # local reversion to raw geometry (timing preserved) for this run.
            raw_fallback_applied = False
            raw_run_errs: List[float] = []
            for fi in range(int(run_start), int(run_end) + 1):
                if fi < len(raw_errs):
                    raw_run_errs.append(float(raw_errs[fi]))
            raw_run_med = float(np.median(np.asarray(raw_run_errs, dtype=np.float64))) if raw_run_errs else 0.0
            if raw_run_med >= float(raw_fallback_min_divergence_m):
                fallback_line = int(curr_line)
                raw_rows: List[Tuple[int, float, float, float, float, int]] = []
                raw_overlap = 0.0
                raw_shift_sum = 0.0
                raw_cost = 0.0
                raw_valid = True
                raw_count = 0
                for fi in range(int(run_start), int(run_end) + 1):
                    fr = frames[fi]
                    if not _frame_has_carla_pose(fr):
                        continue
                    qx, qy, qyaw = _frame_query_pose(fr)
                    rx = _safe_float(fr.get("x"), float(qx))
                    ry = _safe_float(fr.get("y"), float(qy))
                    ryaw = _safe_float(fr.get("yaw"), float(qyaw))
                    cur_x = _safe_float(fr.get("cx"), float(qx))
                    cur_y = _safe_float(fr.get("cy"), float(qy))
                    shift = float(math.hypot(float(rx) - float(cur_x), float(ry) - float(cur_y)))
                    if shift > float(raw_fallback_max_shift_m):
                        raw_valid = False
                        break
                    raw_shift_sum += float(shift)
                    # Keep a small shift term so this remains conservative.
                    raw_cost += 0.05 * float(shift)
                    t = _safe_float(fr.get("t"), float(fi) * float(dt_key))
                    tk = int(round(float(t) * float(inv_dt)))
                    blockers = blockers_by_tick.get(int(tk), [])
                    frame_pen = 0.0
                    for b_key, bx, by, byaw, bl, bw, bspd, b_parked in blockers:
                        if b_key == key:
                            continue
                        center_dist = float(math.hypot(float(rx) - float(bx), float(ry) - float(by)))
                        diag_bound = 0.62 * (math.hypot(float(l), float(w)) + math.hypot(float(bl), float(bw)))
                        if center_dist > diag_bound + 0.45:
                            continue
                        pen = _pair_overlap_excess(
                            fi=int(fi),
                            tk=int(tk),
                            ax=float(rx),
                            ay=float(ry),
                            ayaw=float(ryaw),
                            b_key=str(b_key),
                            bx=float(bx),
                            by=float(by),
                            byaw=float(byaw),
                            bl=float(bl),
                            bw=float(bw),
                            bspd=float(bspd),
                            b_parked=bool(b_parked),
                        )
                        if pen > 0.0:
                            frame_pen += float(pen)
                    raw_overlap += float(frame_pen)
                    # Use local motion tangent for fallback yaw so wrong-way checks
                    # don't spike when raw yaw metadata is noisy.
                    myaw = float(ryaw)
                    if fi > 0 and fi + 1 < len(frames):
                        dx = _safe_float(frames[fi + 1].get("x"), float(rx)) - _safe_float(frames[fi - 1].get("x"), float(rx))
                        dy = _safe_float(frames[fi + 1].get("y"), float(ry)) - _safe_float(frames[fi - 1].get("y"), float(ry))
                        if math.hypot(float(dx), float(dy)) > 1e-4:
                            myaw = float(math.degrees(math.atan2(float(dy), float(dx))))
                    elif fi > 0:
                        dx = _safe_float(frames[fi].get("x"), float(rx)) - _safe_float(frames[fi - 1].get("x"), float(rx))
                        dy = _safe_float(frames[fi].get("y"), float(ry)) - _safe_float(frames[fi - 1].get("y"), float(ry))
                        if math.hypot(float(dx), float(dy)) > 1e-4:
                            myaw = float(math.degrees(math.atan2(float(dy), float(dx))))
                    fallback_line_i = int(fallback_line)
                    fallback_dist_i = 0.0
                    if bool(raw_fallback_reassign_line):
                        best_line = int(fallback_line_i)
                        best_dist = float("inf")

                        def _try_line(cand_line: int) -> None:
                            nonlocal best_line, best_dist
                            if int(cand_line) < 0:
                                return
                            cand = _best_projection_on_carla_line(
                                carla_context=carla_context,
                                line_index=int(cand_line),
                                qx=float(rx),
                                qy=float(ry),
                                qyaw=float(myaw),
                                prefer_reversed=None,
                                enforce_node_direction=bool(raw_fallback_line_reject_wrong_way),
                                opposite_reject_deg=float(opposite_reject_deg),
                            )
                            if cand is None:
                                cand = _best_projection_on_carla_line(
                                    carla_context=carla_context,
                                    line_index=int(cand_line),
                                    qx=float(rx),
                                    qy=float(ry),
                                    qyaw=float(myaw),
                                    prefer_reversed=None,
                                    enforce_node_direction=False,
                                    opposite_reject_deg=float(opposite_reject_deg),
                                )
                            if not isinstance(cand, dict):
                                return
                            proj = cand.get("projection", {})
                            if not isinstance(proj, dict):
                                return
                            d = _safe_float(proj.get("dist"), float("inf"))
                            if not math.isfinite(d):
                                return
                            if float(d) > float(raw_fallback_line_max_dist_m):
                                return
                            if float(d) < float(best_dist):
                                best_dist = float(d)
                                best_line = int(cand_line)

                        if bool(raw_fallback_prefer_cbcli):
                            _try_line(_safe_int(fr.get("cbcli"), -1))
                        _try_line(int(fallback_line))
                        near = _nearest_projection_any_line(
                            carla_context=carla_context,
                            qx=float(rx),
                            qy=float(ry),
                            qyaw=float(myaw),
                            enforce_node_direction=bool(raw_fallback_line_reject_wrong_way),
                            opposite_reject_deg=float(opposite_reject_deg),
                        )
                        if near is None:
                            near = _nearest_projection_any_line(
                                carla_context=carla_context,
                                qx=float(rx),
                                qy=float(ry),
                                qyaw=float(myaw),
                                enforce_node_direction=False,
                                opposite_reject_deg=float(opposite_reject_deg),
                            )
                        if isinstance(near, dict):
                            near_line = _safe_int(near.get("line_index"), -1)
                            near_proj = near.get("projection", {})
                            near_dist = _safe_float(
                                near_proj.get("dist"),
                                float("inf"),
                            ) if isinstance(near_proj, dict) else float("inf")
                            if (
                                int(near_line) >= 0
                                and math.isfinite(float(near_dist))
                                and float(near_dist) <= float(raw_fallback_line_max_dist_m)
                                and float(near_dist) + 1e-6 < float(best_dist)
                            ):
                                best_line = int(near_line)
                                best_dist = float(near_dist)
                        fallback_line_i = int(best_line)
                        if math.isfinite(float(best_dist)):
                            fallback_dist_i = float(best_dist)

                    raw_rows.append(
                        (
                            int(fi),
                            float(rx),
                            float(ry),
                            float(_normalize_yaw_deg(myaw)),
                            float(fallback_dist_i),
                            int(fallback_line_i),
                        )
                    )
                    raw_count += 1

                if raw_valid and raw_count > 0:
                    if bool(raw_fallback_yaw_smooth) and len(raw_rows) >= 3:
                        yaws = np.asarray([float(row[3]) for row in raw_rows], dtype=np.float64)
                        yaws_rad = np.unwrap(np.radians(yaws))
                        sm = np.array(yaws_rad, copy=True)
                        tmp = np.array(sm, copy=True)
                        tmp[1:-1] = 0.2 * sm[:-2] + 0.6 * sm[1:-1] + 0.2 * sm[2:]
                        sm = tmp
                        max_dev_rad = math.radians(float(max(0.0, raw_fallback_yaw_max_dev_deg)))
                        sm = np.clip(sm, yaws_rad - max_dev_rad, yaws_rad + max_dev_rad)
                        for j, row in enumerate(raw_rows):
                            fi_j, px_j, py_j, _, pd_j, pli_j = row
                            raw_rows[j] = (
                                int(fi_j),
                                float(px_j),
                                float(py_j),
                                float(_normalize_yaw_deg(math.degrees(float(sm[j])))),
                                float(pd_j),
                                int(pli_j),
                            )
                    raw_curv_rate_proxy = _curv_rate_proxy_xy([(float(px), float(py)) for _, px, py, _, _, _ in raw_rows])
                    if (
                        float(raw_curv_rate_proxy) > float(max_curv_rate_proxy)
                        and float(raw_curv_rate_proxy) > float(current_curv_rate_proxy) + float(max_curv_rate_proxy_delta)
                    ):
                        raw_valid = False
                    if raw_valid:
                        raw_shift_mean = float(raw_shift_sum / max(1, raw_count))
                        raw_total = (
                            float(raw_cost)
                            + float(overlap_weight) * float(raw_overlap)
                            + float(shift_weight) * float(raw_shift_sum) * float(raw_fallback_shift_weight_scale)
                        )
                        raw_gain = float(current_overlap) - float(raw_overlap)
                        required_gain = max(float(min_gain_abs), float(min_gain_ratio) * max(0.0, float(current_overlap)))
                        if (
                            raw_gain >= required_gain
                            and raw_shift_mean <= float(raw_fallback_max_mean_shift_m)
                            and raw_total + 1e-6 < float(current_total_ref)
                        ):
                            for fi, px, py, pyaw, pd, pli in raw_rows:
                                fr = frames[fi]
                                _set_carla_pose(
                                    frame=fr,
                                    line_index=int(pli),
                                    x=float(px),
                                    y=float(py),
                                    yaw=float(pyaw),
                                    dist=float(pd),
                                    source="overlap_raw_fallback",
                                    quality=str(fr.get("cquality", "none")),
                                )
                            total_runs_adjusted += 1
                            total_frames_adjusted += int(len(raw_rows))
                            adjusted_tracks.add(key)
                            overlap_pen_after += float(raw_overlap)
                            raw_fallback_applied = True

            if raw_fallback_applied:
                if debug_this_track:
                    print(
                        f"[CARLA_OVERLAP][DEBUG] track={tr_id} run={run_start}-{run_end} "
                        f"applied=raw_fallback current_overlap={current_overlap:.3f}"
                    )
                continue

            if int(best_line) == int(curr_line) or not best_rows:
                if debug_this_track:
                    print(
                        f"[CARLA_OVERLAP][DEBUG] track={tr_id} run={run_start}-{run_end} "
                        f"no_better_line best_line={best_line} curr_line={curr_line}"
                    )
                overlap_pen_after += float(current_overlap)
                continue

            overlap_gain = float(current_overlap) - float(best_overlap)
            required_gain = max(float(min_gain_abs), float(min_gain_ratio) * max(0.0, float(current_overlap)))
            run_len = max(1, int(run_end) - int(run_start) + 1)
            if overlap_gain < required_gain:
                if debug_this_track:
                    print(
                        f"[CARLA_OVERLAP][DEBUG] track={tr_id} run={run_start}-{run_end} "
                        f"reject_gain gain={overlap_gain:.3f} required={required_gain:.3f}"
                    )
                overlap_pen_after += float(current_overlap)
                continue
            if best_shift_mean > float(max_mean_shift_m):
                if debug_this_track:
                    print(
                        f"[CARLA_OVERLAP][DEBUG] track={tr_id} run={run_start}-{run_end} "
                        f"reject_shift mean_shift={best_shift_mean:.3f} max={max_mean_shift_m:.3f}"
                    )
                overlap_pen_after += float(current_overlap)
                continue
            if best_fit_cost > (
                float(current_cost)
                + float(raw_fidelity_weight) * float(current_raw_cost)
                + float(cost_slack_per_frame) * float(run_len)
            ):
                if debug_this_track:
                    print(
                        f"[CARLA_OVERLAP][DEBUG] track={tr_id} run={run_start}-{run_end} "
                        f"reject_fit best_fit={best_fit_cost:.3f} ref={current_cost + raw_fidelity_weight * current_raw_cost:.3f}"
                    )
                overlap_pen_after += float(current_overlap)
                continue

            for fi, px, py, pyaw, pd in best_rows:
                fr = frames[fi]
                _set_carla_pose(
                    frame=fr,
                    line_index=int(best_line),
                    x=float(px),
                    y=float(py),
                    yaw=float(pyaw),
                    dist=float(pd),
                    source="overlap_lane_avoid",
                    quality=str(fr.get("cquality", "none")),
                )
            total_runs_adjusted += 1
            total_frames_adjusted += int(len(best_rows))
            adjusted_tracks.add(key)
            overlap_pen_after += float(best_overlap)
            if debug_this_track:
                print(
                    f"[CARLA_OVERLAP][DEBUG] track={tr_id} run={run_start}-{run_end} "
                    f"applied=line {curr_line}->{best_line} overlap {current_overlap:.3f}->{best_overlap:.3f}"
                )

    def _raw_motion_yaw_deg(frames_local: List[Dict[str, object]], fi: int, fallback_deg: float) -> float:
        nloc = len(frames_local)
        if fi > 0 and fi + 1 < nloc:
            dx = _safe_float(frames_local[fi + 1].get("x"), _safe_float(frames_local[fi].get("x"), 0.0)) - _safe_float(frames_local[fi - 1].get("x"), _safe_float(frames_local[fi].get("x"), 0.0))
            dy = _safe_float(frames_local[fi + 1].get("y"), _safe_float(frames_local[fi].get("y"), 0.0)) - _safe_float(frames_local[fi - 1].get("y"), _safe_float(frames_local[fi].get("y"), 0.0))
            if math.hypot(float(dx), float(dy)) > 1e-4:
                return float(_normalize_yaw_deg(math.degrees(math.atan2(float(dy), float(dx)))))
        if fi > 0:
            dx = _safe_float(frames_local[fi].get("x"), 0.0) - _safe_float(frames_local[fi - 1].get("x"), 0.0)
            dy = _safe_float(frames_local[fi].get("y"), 0.0) - _safe_float(frames_local[fi - 1].get("y"), 0.0)
            if math.hypot(float(dx), float(dy)) > 1e-4:
                return float(_normalize_yaw_deg(math.degrees(math.atan2(float(dy), float(dx)))))
        return float(_normalize_yaw_deg(float(fallback_deg)))

    def _metric_overlap_pen_for_pose(key: str, fi: int, tk: int, px: float, py: float, pyaw: float) -> float:
        frames_local = frames_by_key.get(key, [])
        if fi < 0 or fi >= len(frames_local):
            return 0.0
        l1, w1 = dims_by_key.get(key, (4.6, 2.0))
        spd_map = speed_by_key.get(key, np.zeros((len(frames_local),), dtype=np.float64))
        spd_actor = float(spd_map[fi]) if fi < len(spd_map) else 0.0
        total_pen = 0.0
        for b_key in blocker_keys:
            if b_key == key:
                continue
            b_idx = int(tick_to_index.get(b_key, {}).get(int(tk), -1))
            if b_idx < 0:
                continue
            b_frames = frames_by_key.get(b_key, [])
            if b_idx >= len(b_frames):
                continue
            b_fr = b_frames[b_idx]
            if not _frame_has_carla_pose(b_fr):
                continue
            bx = _safe_float(b_fr.get("cx"), float("nan"))
            by = _safe_float(b_fr.get("cy"), float("nan"))
            byaw = _safe_float(b_fr.get("cyaw"), _safe_float(b_fr.get("yaw"), 0.0))
            if not (math.isfinite(bx) and math.isfinite(by)):
                continue
            l2, w2 = dims_by_key.get(b_key, (4.6, 2.0))
            spd_b_map = speed_by_key.get(b_key, np.zeros((len(b_frames),), dtype=np.float64))
            spd_block = float(spd_b_map[b_idx]) if b_idx < len(spd_b_map) else 0.0
            if float(spd_actor) <= float(frame_raw_restore_speed_gate_mps) and float(spd_block) <= float(frame_raw_restore_speed_gate_mps):
                continue
            center_dist = float(math.hypot(float(px) - float(bx), float(py) - float(by)))
            diag_bound = 0.6 * (math.hypot(float(l1), float(w1)) + math.hypot(float(l2), float(w2)))
            if center_dist > diag_bound + 0.5:
                continue
            pen = _obb_overlap_penetration_xyyaw(
                x1=float(px),
                y1=float(py),
                yaw1_deg=float(pyaw),
                len1=float(l1),
                wid1=float(w1),
                x2=float(bx),
                y2=float(by),
                yaw2_deg=float(byaw),
                len2=float(l2),
                wid2=float(w2),
            )
            pen = _effective_overlap_pen_m(float(pen), str(key), str(b_key))
            if pen < float(frame_raw_restore_pair_min_pen_m):
                continue
            a_fr = frames_local[fi]
            arx = _safe_float(a_fr.get("x"), float(px))
            ary = _safe_float(a_fr.get("y"), float(py))
            aryaw = _safe_float(a_fr.get("yaw"), float(pyaw))
            brx = _safe_float(b_fr.get("x"), float(bx))
            bry = _safe_float(b_fr.get("y"), float(by))
            bryaw = _safe_float(b_fr.get("yaw"), float(byaw))
            raw_pen = _obb_overlap_penetration_xyyaw(
                x1=float(arx),
                y1=float(ary),
                yaw1_deg=float(aryaw),
                len1=float(l1),
                wid1=float(w1),
                x2=float(brx),
                y2=float(bry),
                yaw2_deg=float(bryaw),
                len2=float(l2),
                wid2=float(w2),
            )
            raw_pen = _effective_overlap_pen_m(float(raw_pen), str(key), str(b_key))
            if raw_pen >= float(frame_raw_restore_pair_raw_ignore_m):
                continue
            total_pen += float(max(0.0, float(pen) - float(frame_raw_restore_pair_raw_ignore_m)))
        return float(total_pen)

    if bool(frame_raw_restore_enabled):
        key_order: List[Tuple[float, str]] = []
        for key in moving_keys:
            errs = raw_err_by_key.get(key)
            med = float(np.median(errs)) if isinstance(errs, np.ndarray) and errs.size > 0 else 0.0
            key_order.append((float(med), str(key)))
        key_order.sort(reverse=True)
        for _, key in key_order:
            frames = frames_by_key.get(key, [])
            if not isinstance(frames, list) or len(frames) < 3:
                continue
            if key not in track_by_key:
                continue
            for fi, fr in enumerate(frames):
                if not _frame_has_carla_pose(fr):
                    continue
                t = _safe_float(fr.get("t"), float(fi) * float(dt_key))
                tk = int(round(float(t) * float(inv_dt)))
                cx = _safe_float(fr.get("cx"), _safe_float(fr.get("x"), 0.0))
                cy = _safe_float(fr.get("cy"), _safe_float(fr.get("y"), 0.0))
                cyaw = _safe_float(fr.get("cyaw"), _safe_float(fr.get("yaw"), 0.0))
                cur_pen = _metric_overlap_pen_for_pose(str(key), int(fi), int(tk), float(cx), float(cy), float(cyaw))
                if cur_pen <= 0.0:
                    continue
                rx = _safe_float(fr.get("x"), float(cx))
                ry = _safe_float(fr.get("y"), float(cy))
                shift = float(math.hypot(float(rx) - float(cx), float(ry) - float(cy)))
                if shift < float(frame_raw_restore_min_shift_m) or shift > float(frame_raw_restore_max_shift_m):
                    continue
                ryaw = _raw_motion_yaw_deg(frames, int(fi), _safe_float(fr.get("yaw"), float(cyaw)))
                cand_pen = _metric_overlap_pen_for_pose(str(key), int(fi), int(tk), float(rx), float(ry), float(ryaw))
                if float(cand_pen) > float(cur_pen) - float(frame_raw_restore_min_gain):
                    continue
                if fi > 0 and _frame_has_carla_pose(frames[fi - 1]):
                    px0 = _safe_float(frames[fi - 1].get("cx"), _safe_float(frames[fi - 1].get("x"), float(rx)))
                    py0 = _safe_float(frames[fi - 1].get("cy"), _safe_float(frames[fi - 1].get("y"), float(ry)))
                    old_step = float(math.hypot(float(cx) - float(px0), float(cy) - float(py0)))
                    new_step = float(math.hypot(float(rx) - float(px0), float(ry) - float(py0)))
                    step_limit = min(float(frame_raw_restore_max_step_m), float(frame_raw_restore_step_ratio) * max(0.2, old_step))
                    if new_step > step_limit + 1e-6:
                        continue
                if fi + 1 < len(frames) and _frame_has_carla_pose(frames[fi + 1]):
                    px1 = _safe_float(frames[fi + 1].get("cx"), _safe_float(frames[fi + 1].get("x"), float(rx)))
                    py1 = _safe_float(frames[fi + 1].get("cy"), _safe_float(frames[fi + 1].get("y"), float(ry)))
                    old_step = float(math.hypot(float(px1) - float(cx), float(py1) - float(cy)))
                    new_step = float(math.hypot(float(px1) - float(rx), float(py1) - float(ry)))
                    step_limit = min(float(frame_raw_restore_max_step_m), float(frame_raw_restore_step_ratio) * max(0.2, old_step))
                    if new_step > step_limit + 1e-6:
                        continue
                _set_carla_pose(
                    frame=fr,
                    line_index=_safe_int(fr.get("ccli"), -1),
                    x=float(rx),
                    y=float(ry),
                    yaw=float(ryaw),
                    dist=0.0,
                    source="overlap_raw_restore_frame",
                    quality=str(fr.get("cquality", "none")),
                )
                frame_raw_restored += 1
                adjusted_tracks.add(key)

    if adjusted_tracks:
        for key in adjusted_tracks:
            frames = frames_by_key.get(key, [])
            if isinstance(frames, list) and frames:
                _stabilize_carla_line_ids(
                    frames,
                    max_semantic_hold_run=_env_int("V2X_CARLA_SEMANTIC_LINE_HOLD_MAX_RUN", 30, minimum=1, maximum=60),
                    carla_context=carla_context,
                    opposite_reject_deg=_env_float("V2X_CARLA_OPPOSITE_REJECT_DEG", 165.0),
                )
                # Overlap corrections happen after the main CARLA smoothing stack.
                # Run a conservative local transition polish to avoid introducing
                # small boundary artifacts on corrected tracks.
                _smooth_carla_transition_windows(
                    frames,
                    window_radius=_env_int("V2X_CARLA_OVERLAP_POST_WINDOW_RADIUS", 2, minimum=1, maximum=3),
                    max_shift_m=_env_float("V2X_CARLA_OVERLAP_POST_WINDOW_MAX_SHIFT_M", 0.75),
                    max_query_cost_delta=_env_float("V2X_CARLA_OVERLAP_POST_WINDOW_MAX_QUERY_DELTA", 0.6),
                    max_total_cost_delta=_env_float("V2X_CARLA_OVERLAP_POST_WINDOW_MAX_TOTAL_DELTA", 1.2),
                    passes=_env_int("V2X_CARLA_OVERLAP_POST_WINDOW_PASSES", 1, minimum=1, maximum=2),
                )
                _soften_semantic_boundary_jumps(
                    frames,
                    jump_threshold_m=_env_float("V2X_CARLA_OVERLAP_POST_BOUNDARY_JUMP_M", 1.6),
                    jump_ratio_vs_query=_env_float("V2X_CARLA_OVERLAP_POST_BOUNDARY_RATIO", 1.9),
                    move_toward_mid=_env_float("V2X_CARLA_OVERLAP_POST_BOUNDARY_BLEND", 0.45),
                    max_shift_m=_env_float("V2X_CARLA_OVERLAP_POST_BOUNDARY_MAX_SHIFT_M", 0.75),
                    max_query_cost_delta=_env_float("V2X_CARLA_OVERLAP_POST_BOUNDARY_MAX_QUERY_DELTA", 0.6),
                )
                _stabilize_carla_yaw_at_transitions(
                    frames,
                    max_low_step_yaw_jump_deg=_env_float("V2X_CARLA_OVERLAP_POST_YAW_MAX_JUMP_DEG", 20.0),
                )

    if bool(nudge_parked_enabled) and parked_keys and moving_keys:
        nudged_tracks = 0
        nudged_frames = 0

        def _parked_overlap_pen_with_offset(key: str, dx: float, dy: float) -> Tuple[float, int]:
            frames = frames_by_key.get(key, [])
            if not isinstance(frames, list) or not frames:
                return (0.0, 0)
            idx_map = tick_to_index.get(key, {})
            if not idx_map:
                return (0.0, 0)
            l1, w1 = dims_by_key.get(key, (4.6, 2.0))
            total_pen = 0.0
            frame_hits = 0
            for tk, i in idx_map.items():
                if i < 0 or i >= len(frames):
                    continue
                fr = frames[i]
                if not _frame_has_carla_pose(fr):
                    continue
                px = _safe_float(fr.get("cx"), float("nan")) + float(dx)
                py = _safe_float(fr.get("cy"), float("nan")) + float(dy)
                pyaw = _safe_float(fr.get("cyaw"), _safe_float(fr.get("yaw"), 0.0))
                if not (math.isfinite(px) and math.isfinite(py)):
                    continue
                frame_pen = 0.0
                for mkey in moving_keys:
                    j = int(tick_to_index.get(mkey, {}).get(int(tk), -1))
                    if j < 0:
                        continue
                    mframes = frames_by_key.get(mkey, [])
                    if j >= len(mframes):
                        continue
                    mfr = mframes[j]
                    if not _frame_has_carla_pose(mfr):
                        continue
                    bx = _safe_float(mfr.get("cx"), float("nan"))
                    by = _safe_float(mfr.get("cy"), float("nan"))
                    byaw = _safe_float(mfr.get("cyaw"), _safe_float(mfr.get("yaw"), 0.0))
                    if not (math.isfinite(bx) and math.isfinite(by)):
                        continue
                    l2, w2 = dims_by_key.get(mkey, (4.6, 2.0))
                    center_dist = float(math.hypot(float(px) - float(bx), float(py) - float(by)))
                    diag_bound = 0.62 * (math.hypot(float(l1), float(w1)) + math.hypot(float(l2), float(w2)))
                    if center_dist > diag_bound + 0.45:
                        continue
                    pen = _obb_overlap_penetration_xyyaw(
                        x1=float(px),
                        y1=float(py),
                        yaw1_deg=float(pyaw),
                        len1=float(l1),
                        wid1=float(w1),
                        x2=float(bx),
                        y2=float(by),
                        yaw2_deg=float(byaw),
                        len2=float(l2),
                        wid2=float(w2),
                    )
                    pen = _effective_overlap_pen_m(float(pen), str(key), str(mkey))
                    if pen >= float(nudge_parked_min_pair_pen_m):
                        frame_pen += float(pen)
                if frame_pen > 0.0:
                    frame_hits += 1
                    total_pen += float(frame_pen)
            return (float(total_pen), int(frame_hits))

        for key in sorted(parked_keys):
            tr = track_by_key.get(key)
            if not isinstance(tr, dict):
                continue
            frames = frames_by_key.get(key, [])
            if not isinstance(frames, list) or not frames:
                continue
            idx_map = tick_to_index.get(key, {})
            if not idx_map:
                continue

            base_pen, _ = _parked_overlap_pen_with_offset(str(key), 0.0, 0.0)
            if base_pen < float(nudge_parked_min_total_pen_m):
                continue

            raw_err_vals: List[float] = []
            for fr in frames:
                rx = _safe_float(fr.get("x"), 0.0)
                ry = _safe_float(fr.get("y"), 0.0)
                cx = _safe_float(fr.get("cx"), rx)
                cy = _safe_float(fr.get("cy"), ry)
                raw_err_vals.append(float(math.hypot(float(cx) - float(rx), float(cy) - float(ry))))
            raw_err_med = (
                float(np.median(np.asarray(raw_err_vals, dtype=np.float64)))
                if raw_err_vals
                else 0.0
            )
            if raw_err_med > float(nudge_parked_raw_err_med_cap_m):
                continue

            push_x = 0.0
            push_y = 0.0
            l1, w1 = dims_by_key.get(key, (4.6, 2.0))
            for tk, i in idx_map.items():
                if i < 0 or i >= len(frames):
                    continue
                fr = frames[i]
                if not _frame_has_carla_pose(fr):
                    continue
                px = _safe_float(fr.get("cx"), float("nan"))
                py = _safe_float(fr.get("cy"), float("nan"))
                pyaw = _safe_float(fr.get("cyaw"), _safe_float(fr.get("yaw"), 0.0))
                if not (math.isfinite(px) and math.isfinite(py)):
                    continue
                for mkey in moving_keys:
                    j = int(tick_to_index.get(mkey, {}).get(int(tk), -1))
                    if j < 0:
                        continue
                    mframes = frames_by_key.get(mkey, [])
                    if j >= len(mframes):
                        continue
                    mfr = mframes[j]
                    if not _frame_has_carla_pose(mfr):
                        continue
                    bx = _safe_float(mfr.get("cx"), float("nan"))
                    by = _safe_float(mfr.get("cy"), float("nan"))
                    byaw = _safe_float(mfr.get("cyaw"), _safe_float(mfr.get("yaw"), 0.0))
                    if not (math.isfinite(bx) and math.isfinite(by)):
                        continue
                    l2, w2 = dims_by_key.get(mkey, (4.6, 2.0))
                    center_dist = float(math.hypot(float(px) - float(bx), float(py) - float(by)))
                    diag_bound = 0.62 * (math.hypot(float(l1), float(w1)) + math.hypot(float(l2), float(w2)))
                    if center_dist > diag_bound + 0.45:
                        continue
                    pen = _obb_overlap_penetration_xyyaw(
                        x1=float(px),
                        y1=float(py),
                        yaw1_deg=float(pyaw),
                        len1=float(l1),
                        wid1=float(w1),
                        x2=float(bx),
                        y2=float(by),
                        yaw2_deg=float(byaw),
                        len2=float(l2),
                        wid2=float(w2),
                    )
                    pen = _effective_overlap_pen_m(float(pen), str(key), str(mkey))
                    if pen < float(nudge_parked_min_pair_pen_m) or pen > float(nudge_parked_max_pair_pen_m):
                        continue
                    vx = float(px) - float(bx)
                    vy = float(py) - float(by)
                    vn = float(math.hypot(vx, vy))
                    if vn < 1e-5:
                        nyaw = math.radians(float(pyaw))
                        vx = -math.sin(float(nyaw))
                        vy = math.cos(float(nyaw))
                        vn = float(math.hypot(vx, vy))
                    if vn <= 1e-5:
                        continue
                    w_push = float(min(float(nudge_parked_max_pair_pen_m), float(pen)))
                    push_x += float(vx / vn) * float(w_push)
                    push_y += float(vy / vn) * float(w_push)

            vnorm = float(math.hypot(float(push_x), float(push_y)))
            if vnorm <= 1e-4:
                continue
            ux = float(push_x / vnorm)
            uy = float(push_y / vnorm)
            candidate_shifts = [0.15, 0.28, 0.40, float(max(0.15, nudge_parked_max_shift_m))]
            best_dx = 0.0
            best_dy = 0.0
            best_pen = float(base_pen)
            for mag in candidate_shifts:
                m = float(min(float(nudge_parked_max_shift_m), max(0.05, float(mag))))
                dx = float(ux) * float(m)
                dy = float(uy) * float(m)
                cand_pen, _ = _parked_overlap_pen_with_offset(str(key), float(dx), float(dy))
                if cand_pen + 1e-6 < float(best_pen):
                    best_pen = float(cand_pen)
                    best_dx = float(dx)
                    best_dy = float(dy)

            if float(base_pen) - float(best_pen) < float(nudge_parked_min_gain_m):
                continue
            if abs(float(best_dx)) < 1e-6 and abs(float(best_dy)) < 1e-6:
                continue

            local_adjusted = 0
            for fr in frames:
                if not _frame_has_carla_pose(fr):
                    continue
                cx = _safe_float(fr.get("cx"), _safe_float(fr.get("x"), 0.0))
                cy = _safe_float(fr.get("cy"), _safe_float(fr.get("y"), 0.0))
                cyaw = _safe_float(fr.get("cyaw"), _safe_float(fr.get("yaw"), 0.0))
                cli = _safe_int(fr.get("ccli"), -1)
                cdist = _safe_float(fr.get("cdist"), 0.0)
                _set_carla_pose(
                    frame=fr,
                    line_index=int(cli),
                    x=float(cx) + float(best_dx),
                    y=float(cy) + float(best_dy),
                    yaw=float(cyaw),
                    dist=float(cdist),
                    source="parked_overlap_nudge",
                    quality=str(fr.get("cquality", "none")),
                )
                local_adjusted += 1

            if local_adjusted > 0:
                nudged_tracks += 1
                nudged_frames += int(local_adjusted)

        report["parked_nudged_tracks"] = int(nudged_tracks)
        report["parked_nudged_frames"] = int(nudged_frames)

    micro_stationary_keys = set(parked_keys)
    if bool(micro_nudge_include_quasi):
        micro_stationary_keys.update(set(quasi_parked_keys))

    if bool(micro_nudge_parked_enabled) and micro_stationary_keys and moving_keys:
        def _parked_eval_with_offset(
            key: str,
            dx: float,
            dy: float,
            pen_min: float,
            pen_max: float,
        ) -> Tuple[float, float, float, int]:
            frames = frames_by_key.get(key, [])
            if not isinstance(frames, list) or not frames:
                return (0.0, 0.0, 0.0, 0)
            idx_map = tick_to_index.get(key, {})
            if not idx_map:
                return (0.0, 0.0, 0.0, 0)
            l1, w1 = dims_by_key.get(key, (4.6, 2.0))
            pen_all = 0.0
            pen_band = 0.0
            max_pair = 0.0
            frame_hits = 0
            for tk, i in idx_map.items():
                if i < 0 or i >= len(frames):
                    continue
                fr = frames[i]
                if not _frame_has_carla_pose(fr):
                    continue
                px = _safe_float(fr.get("cx"), float("nan")) + float(dx)
                py = _safe_float(fr.get("cy"), float("nan")) + float(dy)
                pyaw = _safe_float(fr.get("cyaw"), _safe_float(fr.get("yaw"), 0.0))
                if not (math.isfinite(px) and math.isfinite(py)):
                    continue
                frame_has_band = False
                for mkey in moving_keys:
                    j = int(tick_to_index.get(mkey, {}).get(int(tk), -1))
                    if j < 0:
                        continue
                    mframes = frames_by_key.get(mkey, [])
                    if j >= len(mframes):
                        continue
                    mfr = mframes[j]
                    if not _frame_has_carla_pose(mfr):
                        continue
                    bx = _safe_float(mfr.get("cx"), float("nan"))
                    by = _safe_float(mfr.get("cy"), float("nan"))
                    byaw = _safe_float(mfr.get("cyaw"), _safe_float(mfr.get("yaw"), 0.0))
                    if not (math.isfinite(bx) and math.isfinite(by)):
                        continue
                    l2, w2 = dims_by_key.get(mkey, (4.6, 2.0))
                    center_dist = float(math.hypot(float(px) - float(bx), float(py) - float(by)))
                    diag_bound = 0.62 * (math.hypot(float(l1), float(w1)) + math.hypot(float(l2), float(w2)))
                    if center_dist > diag_bound + 0.45:
                        continue
                    pen_raw = _obb_overlap_penetration_xyyaw(
                        x1=float(px),
                        y1=float(py),
                        yaw1_deg=float(pyaw),
                        len1=float(l1),
                        wid1=float(w1),
                        x2=float(bx),
                        y2=float(by),
                        yaw2_deg=float(byaw),
                        len2=float(l2),
                        wid2=float(w2),
                    )
                    pen = _effective_overlap_pen_m(float(pen_raw), str(key), str(mkey))
                    if pen <= float(pen_min):
                        continue
                    pen_all += float(pen)
                    if pen >= float(pen_min) and pen <= float(pen_max):
                        pen_band += float(pen)
                        frame_has_band = True
                    if pen > float(max_pair):
                        max_pair = float(pen)
                if frame_has_band:
                    frame_hits += 1
            return (float(pen_all), float(pen_band), float(max_pair), int(frame_hits))

        micro_candidates: List[Tuple[float, str]] = []
        for key in sorted(micro_stationary_keys):
            pen_all, pen_band, _, band_hits = _parked_eval_with_offset(
                str(key),
                0.0,
                0.0,
                float(micro_nudge_pair_pen_min_m),
                float(micro_nudge_pair_pen_max_m),
            )
            if pen_band < float(micro_nudge_total_pen_min_m):
                continue
            if band_hits <= 0:
                continue
            micro_candidates.append((float(pen_all + pen_band), str(key)))

        if micro_candidates:
            micro_candidates.sort(reverse=True)
            applied_tracks = 0
            micro_tracks = 0
            micro_frames = 0
            for _, key in micro_candidates:
                if applied_tracks >= int(micro_nudge_max_tracks):
                    break
                tr = track_by_key.get(key)
                if not isinstance(tr, dict):
                    continue
                frames = frames_by_key.get(key, [])
                if not isinstance(frames, list) or not frames:
                    continue
                idx_map = tick_to_index.get(key, {})
                if not idx_map:
                    continue

                raw_err_vals: List[float] = []
                for fr in frames:
                    rx = _safe_float(fr.get("x"), 0.0)
                    ry = _safe_float(fr.get("y"), 0.0)
                    cx = _safe_float(fr.get("cx"), rx)
                    cy = _safe_float(fr.get("cy"), ry)
                    raw_err_vals.append(float(math.hypot(float(cx) - float(rx), float(cy) - float(ry))))
                raw_err_med = float(np.median(np.asarray(raw_err_vals, dtype=np.float64))) if raw_err_vals else 0.0
                if raw_err_med > float(micro_nudge_raw_err_med_cap_m):
                    continue

                base_all, base_band, base_max_pair, _ = _parked_eval_with_offset(
                    str(key),
                    0.0,
                    0.0,
                    float(micro_nudge_pair_pen_min_m),
                    float(micro_nudge_pair_pen_max_m),
                )
                if base_band < float(micro_nudge_total_pen_min_m):
                    continue

                push_x = 0.0
                push_y = 0.0
                l1, w1 = dims_by_key.get(key, (4.6, 2.0))
                for tk, i in idx_map.items():
                    if i < 0 or i >= len(frames):
                        continue
                    fr = frames[i]
                    if not _frame_has_carla_pose(fr):
                        continue
                    px = _safe_float(fr.get("cx"), float("nan"))
                    py = _safe_float(fr.get("cy"), float("nan"))
                    pyaw = _safe_float(fr.get("cyaw"), _safe_float(fr.get("yaw"), 0.0))
                    if not (math.isfinite(px) and math.isfinite(py)):
                        continue
                    for mkey in moving_keys:
                        j = int(tick_to_index.get(mkey, {}).get(int(tk), -1))
                        if j < 0:
                            continue
                        mframes = frames_by_key.get(mkey, [])
                        if j >= len(mframes):
                            continue
                        mfr = mframes[j]
                        if not _frame_has_carla_pose(mfr):
                            continue
                        bx = _safe_float(mfr.get("cx"), float("nan"))
                        by = _safe_float(mfr.get("cy"), float("nan"))
                        byaw = _safe_float(mfr.get("cyaw"), _safe_float(mfr.get("yaw"), 0.0))
                        if not (math.isfinite(bx) and math.isfinite(by)):
                            continue
                        l2, w2 = dims_by_key.get(mkey, (4.6, 2.0))
                        center_dist = float(math.hypot(float(px) - float(bx), float(py) - float(by)))
                        diag_bound = 0.62 * (math.hypot(float(l1), float(w1)) + math.hypot(float(l2), float(w2)))
                        if center_dist > diag_bound + 0.45:
                            continue
                        pen_raw = _obb_overlap_penetration_xyyaw(
                            x1=float(px),
                            y1=float(py),
                            yaw1_deg=float(pyaw),
                            len1=float(l1),
                            wid1=float(w1),
                            x2=float(bx),
                            y2=float(by),
                            yaw2_deg=float(byaw),
                            len2=float(l2),
                            wid2=float(w2),
                        )
                        pen = _effective_overlap_pen_m(float(pen_raw), str(key), str(mkey))
                        if pen < float(micro_nudge_pair_pen_min_m) or pen > float(micro_nudge_pair_pen_max_m):
                            continue
                        vx = float(px) - float(bx)
                        vy = float(py) - float(by)
                        vn = float(math.hypot(vx, vy))
                        if vn <= 1e-5:
                            nyaw = math.radians(float(pyaw))
                            vx = -math.sin(float(nyaw))
                            vy = math.cos(float(nyaw))
                            vn = float(math.hypot(vx, vy))
                        if vn <= 1e-5:
                            continue
                        w_push = float(min(float(micro_nudge_pair_pen_max_m), float(pen)))
                        push_x += float(vx / vn) * float(w_push)
                        push_y += float(vy / vn) * float(w_push)

                pnorm = float(math.hypot(float(push_x), float(push_y)))
                if pnorm <= 1e-5:
                    continue
                ux = float(push_x / pnorm)
                uy = float(push_y / pnorm)
                mag_candidates = [0.05, 0.09, 0.13, float(max(0.05, micro_nudge_max_shift_m))]
                best_dx = 0.0
                best_dy = 0.0
                best_all = float(base_all)
                best_band = float(base_band)
                best_max_pair = float(base_max_pair)
                for mag in mag_candidates:
                    shift_mag = float(min(float(micro_nudge_max_shift_m), max(0.02, float(mag))))
                    if shift_mag > float(micro_nudge_raw_err_add_cap_m):
                        continue
                    dx = float(ux) * float(shift_mag)
                    dy = float(uy) * float(shift_mag)
                    cand_all, cand_band, cand_max_pair, _ = _parked_eval_with_offset(
                        str(key),
                        float(dx),
                        float(dy),
                        float(micro_nudge_pair_pen_min_m),
                        float(micro_nudge_pair_pen_max_m),
                    )
                    if cand_max_pair > float(base_max_pair) + 1e-6:
                        continue
                    if (float(base_all) - float(cand_all)) < float(micro_nudge_min_gain_m):
                        continue
                    if float(cand_band) > float(best_band) + 1e-6:
                        continue
                    if float(cand_all) + 1e-6 < float(best_all):
                        best_dx = float(dx)
                        best_dy = float(dy)
                        best_all = float(cand_all)
                        best_band = float(cand_band)
                        best_max_pair = float(cand_max_pair)

                if abs(float(best_dx)) < 1e-6 and abs(float(best_dy)) < 1e-6:
                    continue

                local_adjusted = 0
                for fr in frames:
                    if not _frame_has_carla_pose(fr):
                        continue
                    cx = _safe_float(fr.get("cx"), _safe_float(fr.get("x"), 0.0))
                    cy = _safe_float(fr.get("cy"), _safe_float(fr.get("y"), 0.0))
                    cyaw = _safe_float(fr.get("cyaw"), _safe_float(fr.get("yaw"), 0.0))
                    cli = _safe_int(fr.get("ccli"), -1)
                    cdist = _safe_float(fr.get("cdist"), 0.0)
                    _set_carla_pose(
                        frame=fr,
                        line_index=int(cli),
                        x=float(cx) + float(best_dx),
                        y=float(cy) + float(best_dy),
                        yaw=float(cyaw),
                        dist=float(cdist),
                        source="parked_overlap_micro_nudge",
                        quality=str(fr.get("cquality", "none")),
                    )
                    local_adjusted += 1

                if local_adjusted > 0:
                    micro_tracks += 1
                    micro_frames += int(local_adjusted)
                    applied_tracks += 1

            report["parked_micro_nudged_tracks"] = int(micro_tracks)
            report["parked_micro_nudged_frames"] = int(micro_frames)

    if bool(moving_pair_nudge_enabled) and len(moving_keys) >= 2:
        pair_candidates: List[Tuple[float, str, str, str, List[Tuple[int, int, int, float, float, float, float, float, float, float]]]] = []
        moving_list = sorted(moving_keys)
        for i in range(len(moving_list)):
            a_key = str(moving_list[i])
            idx_a = tick_to_index.get(a_key, {})
            frames_a = frames_by_key.get(a_key, [])
            if not idx_a or not isinstance(frames_a, list):
                continue
            l_a, w_a = dims_by_key.get(a_key, (4.6, 2.0))
            for j in range(i + 1, len(moving_list)):
                b_key = str(moving_list[j])
                idx_b = tick_to_index.get(b_key, {})
                frames_b = frames_by_key.get(b_key, [])
                if not idx_b or not isinstance(frames_b, list):
                    continue
                l_b, w_b = dims_by_key.get(b_key, (4.6, 2.0))
                common_ticks = sorted(set(idx_a.keys()).intersection(idx_b.keys()))
                if len(common_ticks) < int(moving_pair_nudge_min_frames):
                    continue
                rows: List[Tuple[int, int, int, float, float, float, float, float, float, float]] = []
                for tk in common_ticks:
                    ia = int(idx_a.get(int(tk), -1))
                    ib = int(idx_b.get(int(tk), -1))
                    if ia < 0 or ib < 0 or ia >= len(frames_a) or ib >= len(frames_b):
                        continue
                    fa = frames_a[ia]
                    fb = frames_b[ib]
                    if not (_frame_has_carla_pose(fa) and _frame_has_carla_pose(fb)):
                        continue
                    ax = _safe_float(fa.get("cx"), _safe_float(fa.get("x"), float("nan")))
                    ay = _safe_float(fa.get("cy"), _safe_float(fa.get("y"), float("nan")))
                    ayaw = _safe_float(fa.get("cyaw"), _safe_float(fa.get("yaw"), 0.0))
                    bx = _safe_float(fb.get("cx"), _safe_float(fb.get("x"), float("nan")))
                    by = _safe_float(fb.get("cy"), _safe_float(fb.get("y"), float("nan")))
                    byaw = _safe_float(fb.get("cyaw"), _safe_float(fb.get("yaw"), 0.0))
                    if not (math.isfinite(ax) and math.isfinite(ay) and math.isfinite(bx) and math.isfinite(by)):
                        continue
                    center_dist = float(math.hypot(float(ax) - float(bx), float(ay) - float(by)))
                    diag_bound = 0.62 * (math.hypot(float(l_a), float(w_a)) + math.hypot(float(l_b), float(w_b)))
                    if center_dist > diag_bound + 0.45:
                        continue
                    pen_raw = _obb_overlap_penetration_xyyaw(
                        x1=float(ax),
                        y1=float(ay),
                        yaw1_deg=float(ayaw),
                        len1=float(l_a),
                        wid1=float(w_a),
                        x2=float(bx),
                        y2=float(by),
                        yaw2_deg=float(byaw),
                        len2=float(l_b),
                        wid2=float(w_b),
                    )
                    pen = _effective_overlap_pen_m(float(pen_raw), str(a_key), str(b_key))
                    if pen < float(moving_pair_nudge_min_pair_pen_m):
                        continue
                    rows.append(
                        (
                            int(tk),
                            int(ia),
                            int(ib),
                            float(ax),
                            float(ay),
                            float(ayaw),
                            float(bx),
                            float(by),
                            float(byaw),
                            float(pen),
                        )
                    )
                if len(rows) < int(moving_pair_nudge_min_frames):
                    continue

                a_quasi = bool(a_key in quasi_parked_keys)
                b_quasi = bool(b_key in quasi_parked_keys)
                adjust_key = ""
                if a_quasi ^ b_quasi:
                    adjust_key = str(a_key if a_quasi else b_key)
                elif a_quasi and b_quasi:
                    a_net = float(motion_stats_by_key.get(a_key, {}).get("robust_net_disp_m", 1e9))
                    b_net = float(motion_stats_by_key.get(b_key, {}).get("robust_net_disp_m", 1e9))
                    adjust_key = str(a_key if a_net <= b_net else b_key)
                else:
                    a_err = raw_err_by_key.get(a_key, np.zeros((0,), dtype=np.float64))
                    b_err = raw_err_by_key.get(b_key, np.zeros((0,), dtype=np.float64))
                    a_med = float(np.median(a_err)) if isinstance(a_err, np.ndarray) and a_err.size > 0 else 0.0
                    b_med = float(np.median(b_err)) if isinstance(b_err, np.ndarray) and b_err.size > 0 else 0.0
                    prefer_key = str(a_key if a_med >= b_med else b_key)
                    spacing_adv = 0.0
                    raw_sep_vals: List[float] = []
                    carla_sep_vals: List[float] = []
                    for _, ia, ib, ax, ay, _, bx, by, _, _ in rows:
                        if ia < 0 or ib < 0 or ia >= len(frames_a) or ib >= len(frames_b):
                            continue
                        fa = frames_a[int(ia)]
                        fb = frames_b[int(ib)]
                        rax = _safe_float(fa.get("x"), float("nan"))
                        ray = _safe_float(fa.get("y"), float("nan"))
                        rbx = _safe_float(fb.get("x"), float("nan"))
                        rby = _safe_float(fb.get("y"), float("nan"))
                        if math.isfinite(rax) and math.isfinite(ray) and math.isfinite(rbx) and math.isfinite(rby):
                            raw_sep_vals.append(float(math.hypot(float(rax) - float(rbx), float(ray) - float(rby))))
                        carla_sep_vals.append(float(math.hypot(float(ax) - float(bx), float(ay) - float(by))))
                    if raw_sep_vals and carla_sep_vals:
                        spacing_adv = float(np.median(np.asarray(raw_sep_vals, dtype=np.float64))) - float(
                            np.median(np.asarray(carla_sep_vals, dtype=np.float64))
                        )
                    if abs(float(a_med) - float(b_med)) >= float(moving_pair_nudge_raw_err_adv_m):
                        adjust_key = str(prefer_key)
                    elif float(spacing_adv) >= float(moving_pair_nudge_min_spacing_adv_m):
                        role_a = str(track_by_key.get(str(a_key), {}).get("role", "")).strip().lower()
                        role_b = str(track_by_key.get(str(b_key), {}).get("role", "")).strip().lower()
                        if role_a == "ego" and role_b != "ego":
                            adjust_key = str(b_key)
                        elif role_b == "ego" and role_a != "ego":
                            adjust_key = str(a_key)
                        else:
                            adjust_key = str(prefer_key)
                    else:
                        continue

                score = float(sum(float(r[9]) for r in rows))
                pair_candidates.append((float(score), str(a_key), str(b_key), str(adjust_key), rows))

        if pair_candidates:
            pair_candidates.sort(reverse=True, key=lambda r: float(r[0]))
            moved_tracks: set = set()
            moved_frames_total = 0
            for _, a_key, b_key, adjust_key, rows in pair_candidates:
                if len(moved_tracks) >= int(moving_pair_nudge_max_tracks):
                    break
                if adjust_key in moved_tracks:
                    continue
                other_key = str(b_key if str(adjust_key) == str(a_key) else a_key)
                frames_adj = frames_by_key.get(adjust_key, [])
                if not isinstance(frames_adj, list) or not frames_adj:
                    continue
                idx_adj = tick_to_index.get(adjust_key, {})
                if not idx_adj:
                    continue
                l_adj, w_adj = dims_by_key.get(adjust_key, (4.6, 2.0))
                l_oth, w_oth = dims_by_key.get(other_key, (4.6, 2.0))

                event_fi_set: set = set()
                base_pen_sum = 0.0
                push_x = 0.0
                push_y = 0.0
                for tk, ia, ib, ax, ay, ayaw, bx, by, byaw, pen in rows:
                    if str(adjust_key) == str(a_key):
                        fi_adj = int(ia)
                        px, py, pyaw = float(ax), float(ay), float(ayaw)
                        ox, oy, oyaw = float(bx), float(by), float(byaw)
                    else:
                        fi_adj = int(ib)
                        px, py, pyaw = float(bx), float(by), float(byaw)
                        ox, oy, oyaw = float(ax), float(ay), float(ayaw)
                    event_fi_set.add(int(fi_adj))
                    base_pen_sum += float(pen)
                    vx = float(px) - float(ox)
                    vy = float(py) - float(oy)
                    vn = float(math.hypot(float(vx), float(vy)))
                    if vn <= 1e-5:
                        nyaw = math.radians(float(pyaw))
                        vx = -math.sin(float(nyaw))
                        vy = math.cos(float(nyaw))
                        vn = float(math.hypot(float(vx), float(vy)))
                    if vn <= 1e-5:
                        continue
                    w_push = float(min(2.0, max(0.05, float(pen))))
                    push_x += float(vx / vn) * float(w_push)
                    push_y += float(vy / vn) * float(w_push)
                if not event_fi_set or base_pen_sum <= 0.0:
                    continue
                push_norm = float(math.hypot(float(push_x), float(push_y)))
                if push_norm <= 1e-5:
                    continue
                ux = float(push_x / push_norm)
                uy = float(push_y / push_norm)

                event_indices = sorted(int(v) for v in event_fi_set)
                blend_radius = int(max(0, moving_pair_nudge_blend_radius))
                quasi_like = bool(adjust_key in quasi_parked_keys)
                weights = np.zeros((len(frames_adj),), dtype=np.float64)
                if quasi_like:
                    for fi, fr in enumerate(frames_adj):
                        if _frame_has_carla_pose(fr):
                            weights[fi] = 1.0
                else:
                    for fi, fr in enumerate(frames_adj):
                        if not _frame_has_carla_pose(fr):
                            continue
                        nearest = min((abs(int(fi) - int(ev)) for ev in event_indices), default=10**6)
                        if int(nearest) <= int(blend_radius):
                            weights[fi] = float(int(blend_radius) + 1 - int(nearest)) / float(int(blend_radius) + 1)

                candidate_mags = [
                    0.10,
                    0.22,
                    0.36,
                    0.52,
                    0.72,
                    float(max(0.10, moving_pair_nudge_max_shift_m)),
                ]
                best_dx = 0.0
                best_dy = 0.0
                best_pen_sum = float(base_pen_sum)
                for mag in candidate_mags:
                    shift_mag = float(min(float(moving_pair_nudge_max_shift_m), max(0.02, float(mag))))
                    dx = float(ux) * float(shift_mag)
                    dy = float(uy) * float(shift_mag)

                    valid = True
                    for fi, fr in enumerate(frames_adj):
                        wfi = float(weights[fi]) if fi < len(weights) else 0.0
                        if wfi <= 1e-9:
                            continue
                        cx = _safe_float(fr.get("cx"), _safe_float(fr.get("x"), 0.0))
                        cy = _safe_float(fr.get("cy"), _safe_float(fr.get("y"), 0.0))
                        rx = _safe_float(fr.get("x"), cx)
                        ry = _safe_float(fr.get("y"), cy)
                        old_err = float(math.hypot(float(cx) - float(rx), float(cy) - float(ry)))
                        nx = float(cx) + float(dx) * float(wfi)
                        ny = float(cy) + float(dy) * float(wfi)
                        new_err = float(math.hypot(float(nx) - float(rx), float(ny) - float(ry)))
                        if float(new_err) > float(old_err) + float(moving_pair_nudge_raw_err_add_cap_m):
                            valid = False
                            break
                    if not valid:
                        continue

                    for fi in range(1, len(frames_adj)):
                        fr0 = frames_adj[fi - 1]
                        fr1 = frames_adj[fi]
                        if not (_frame_has_carla_pose(fr0) and _frame_has_carla_pose(fr1)):
                            continue
                        w0 = float(weights[fi - 1]) if fi - 1 < len(weights) else 0.0
                        w1 = float(weights[fi]) if fi < len(weights) else 0.0
                        if max(float(w0), float(w1)) <= 1e-9:
                            continue
                        c0x = _safe_float(fr0.get("cx"), _safe_float(fr0.get("x"), 0.0))
                        c0y = _safe_float(fr0.get("cy"), _safe_float(fr0.get("y"), 0.0))
                        c1x = _safe_float(fr1.get("cx"), _safe_float(fr1.get("x"), 0.0))
                        c1y = _safe_float(fr1.get("cy"), _safe_float(fr1.get("y"), 0.0))
                        n0x = float(c0x) + float(dx) * float(w0)
                        n0y = float(c0y) + float(dy) * float(w0)
                        n1x = float(c1x) + float(dx) * float(w1)
                        n1y = float(c1y) + float(dy) * float(w1)
                        old_step = float(math.hypot(float(c1x) - float(c0x), float(c1y) - float(c0y)))
                        new_step = float(math.hypot(float(n1x) - float(n0x), float(n1y) - float(n0y)))
                        step_lim = min(
                            float(moving_pair_nudge_max_step_m),
                            float(moving_pair_nudge_step_ratio) * max(0.2, float(old_step)),
                        )
                        if float(new_step) > float(step_lim) + 1e-6:
                            valid = False
                            break
                    if not valid:
                        continue

                    cand_pen_sum = 0.0
                    for tk, ia, ib, ax, ay, ayaw, bx, by, byaw, _ in rows:
                        if str(adjust_key) == str(a_key):
                            fi_adj = int(ia)
                            wfi = float(weights[fi_adj]) if fi_adj < len(weights) else 0.0
                            px = float(ax) + float(dx) * float(wfi)
                            py = float(ay) + float(dy) * float(wfi)
                            pyaw = float(ayaw)
                            ox = float(bx)
                            oy = float(by)
                            oyaw = float(byaw)
                        else:
                            fi_adj = int(ib)
                            wfi = float(weights[fi_adj]) if fi_adj < len(weights) else 0.0
                            px = float(bx) + float(dx) * float(wfi)
                            py = float(by) + float(dy) * float(wfi)
                            pyaw = float(byaw)
                            ox = float(ax)
                            oy = float(ay)
                            oyaw = float(ayaw)
                        pen_raw = _obb_overlap_penetration_xyyaw(
                            x1=float(px),
                            y1=float(py),
                            yaw1_deg=float(pyaw),
                            len1=float(l_adj),
                            wid1=float(w_adj),
                            x2=float(ox),
                            y2=float(oy),
                            yaw2_deg=float(oyaw),
                            len2=float(l_oth),
                            wid2=float(w_oth),
                        )
                        cand_pen_sum += float(_effective_overlap_pen_m(float(pen_raw), str(adjust_key), str(other_key)))

                    gain = float(base_pen_sum) - float(cand_pen_sum)
                    if gain < float(moving_pair_nudge_min_gain_m):
                        continue
                    if float(cand_pen_sum) + 1e-6 < float(best_pen_sum):
                        best_pen_sum = float(cand_pen_sum)
                        best_dx = float(dx)
                        best_dy = float(dy)

                if abs(float(best_dx)) < 1e-6 and abs(float(best_dy)) < 1e-6:
                    continue

                local_moved = 0
                raw_err_arr = raw_err_by_key.get(adjust_key)
                for fi, fr in enumerate(frames_adj):
                    wfi = float(weights[fi]) if fi < len(weights) else 0.0
                    if wfi <= 1e-9 or not _frame_has_carla_pose(fr):
                        continue
                    cx = _safe_float(fr.get("cx"), _safe_float(fr.get("x"), 0.0))
                    cy = _safe_float(fr.get("cy"), _safe_float(fr.get("y"), 0.0))
                    cyaw = _safe_float(fr.get("cyaw"), _safe_float(fr.get("yaw"), 0.0))
                    cli = _safe_int(fr.get("ccli"), -1)
                    cdist = _safe_float(fr.get("cdist"), 0.0)
                    nx = float(cx) + float(best_dx) * float(wfi)
                    ny = float(cy) + float(best_dy) * float(wfi)
                    _set_carla_pose(
                        frame=fr,
                        line_index=int(cli),
                        x=float(nx),
                        y=float(ny),
                        yaw=float(cyaw),
                        dist=float(cdist),
                        source="moving_pair_overlap_nudge",
                        quality=str(fr.get("cquality", "none")),
                    )
                    if isinstance(raw_err_arr, np.ndarray) and fi < len(raw_err_arr):
                        rx = _safe_float(fr.get("x"), float(nx))
                        ry = _safe_float(fr.get("y"), float(ny))
                        raw_err_arr[fi] = float(math.hypot(float(nx) - float(rx), float(ny) - float(ry)))
                    local_moved += 1

                if local_moved > 0:
                    moved_tracks.add(str(adjust_key))
                    moved_frames_total += int(local_moved)

            report["moving_pair_nudged_tracks"] = int(len(moved_tracks))
            report["moving_pair_nudged_frames"] = int(moved_frames_total)

    stationary_remove_keys = set(parked_keys)
    if _env_int("V2X_CARLA_OVERLAP_REMOVE_INCLUDE_QUASI", 1, minimum=0, maximum=1) == 1:
        quasi_remove_robust_net_max = _env_float("V2X_CARLA_OVERLAP_REMOVE_QUASI_ROBUST_NET_MAX_M", 3.4)
        quasi_remove_sustain_ratio_max = _env_float("V2X_CARLA_OVERLAP_REMOVE_QUASI_SUSTAIN_RATIO_MAX", 0.35)
        for key in quasi_parked_keys:
            ms = motion_stats_by_key.get(str(key), {})
            if not isinstance(ms, dict):
                continue
            if (
                float(ms.get("robust_net_disp_m", 1e9)) <= float(quasi_remove_robust_net_max)
                and float(ms.get("sustained_disp_ratio", 1.0)) <= float(quasi_remove_sustain_ratio_max)
            ):
                stationary_remove_keys.add(str(key))

    if bool(remove_obstructing_parked_enabled) and stationary_remove_keys and moving_keys:
        parked_candidates: List[Tuple[float, float, int, str]] = []
        for key in sorted(stationary_remove_keys):
            tr = track_by_key.get(key)
            if not isinstance(tr, dict):
                continue
            frames = frames_by_key.get(key, [])
            if not isinstance(frames, list) or not frames:
                continue

            idx_map = tick_to_index.get(key, {})
            if not idx_map:
                continue

            raw_err = raw_err_by_key.get(key, np.zeros((0,), dtype=np.float64))
            raw_err_med = float(np.median(raw_err)) if isinstance(raw_err, np.ndarray) and raw_err.size > 0 else 0.0
            overlap_frames = 0
            overlap_pen_sum = 0.0

            l1, w1 = dims_by_key.get(key, (4.6, 2.0))
            for tk, i in idx_map.items():
                if i < 0 or i >= len(frames):
                    continue
                fr = frames[i]
                px = _safe_float(fr.get("cx"), _safe_float(fr.get("x"), float("nan")))
                py = _safe_float(fr.get("cy"), _safe_float(fr.get("y"), float("nan")))
                pyaw = _safe_float(fr.get("cyaw"), _safe_float(fr.get("yaw"), 0.0))
                if not (math.isfinite(px) and math.isfinite(py)):
                    continue

                frame_pen = 0.0
                for mkey in moving_keys:
                    j = int(tick_to_index.get(mkey, {}).get(int(tk), -1))
                    if j < 0:
                        continue
                    mframes = frames_by_key.get(mkey, [])
                    if j >= len(mframes):
                        continue
                    mfr = mframes[j]
                    bx = _safe_float(mfr.get("cx"), _safe_float(mfr.get("x"), float("nan")))
                    by = _safe_float(mfr.get("cy"), _safe_float(mfr.get("y"), float("nan")))
                    byaw = _safe_float(mfr.get("cyaw"), _safe_float(mfr.get("yaw"), 0.0))
                    if not (math.isfinite(bx) and math.isfinite(by)):
                        continue
                    l2, w2 = dims_by_key.get(mkey, (4.6, 2.0))
                    center_dist = float(math.hypot(float(px) - float(bx), float(py) - float(by)))
                    diag_bound = 0.62 * (math.hypot(float(l1), float(w1)) + math.hypot(float(l2), float(w2)))
                    if center_dist > diag_bound + 0.45:
                        continue
                    pen = _obb_overlap_penetration_xyyaw(
                        x1=float(px),
                        y1=float(py),
                        yaw1_deg=float(pyaw),
                        len1=float(l1),
                        wid1=float(w1),
                        x2=float(bx),
                        y2=float(by),
                        yaw2_deg=float(byaw),
                        len2=float(l2),
                        wid2=float(w2),
                    )
                    pen = _effective_overlap_pen_m(float(pen), str(key), str(mkey))
                    if pen >= float(remove_obstructing_parked_min_pair_pen_m):
                        frame_pen += float(pen)

                if frame_pen > 0.0:
                    overlap_frames += 1
                    overlap_pen_sum += float(frame_pen)

            if overlap_frames <= 0:
                continue

            overlap_ratio = float(overlap_frames) / max(1.0, float(len(idx_map)))
            if overlap_frames < int(remove_obstructing_parked_min_overlap_frames):
                continue
            if overlap_ratio < float(remove_obstructing_parked_min_overlap_ratio):
                continue
            if overlap_pen_sum < float(remove_obstructing_parked_min_pen_sum_m):
                continue

            motion_stats = motion_stats_by_key.get(str(key), _track_raw_motion_stats(tr))
            net_disp = float(
                min(
                    float(motion_stats.get("net_disp_m", 0.0)),
                    float(motion_stats.get("robust_net_disp_m", motion_stats.get("net_disp_m", 0.0))),
                )
            )
            stationary_confident = bool(net_disp <= float(remove_obstructing_parked_stationary_net_max_m))

            # For strongly stationary blockers, prioritize overlap removal even when
            # raw-fit error is low. For less-stationary tracks keep the raw-fit gate.
            if (
                (not stationary_confident)
                and
                overlap_ratio < float(remove_obstructing_parked_ratio_override)
                and overlap_frames < int(remove_obstructing_parked_hard_overlap_frames)
                and raw_err_med < float(remove_obstructing_parked_min_raw_err_med_m)
            ):
                continue

            bonus_den = max(0.2, float(remove_obstructing_parked_net_bonus_m))
            net_bonus = 1.0 + 0.6 * max(0.0, float(bonus_den - float(net_disp)) / float(bonus_den))
            score = float(overlap_pen_sum) * (1.0 + float(overlap_ratio)) * float(net_bonus)
            parked_candidates.append((float(score), float(overlap_ratio), int(overlap_frames), str(key)))

        if parked_candidates:
            parked_candidates.sort(reverse=True)
            max_remove = int(
                min(
                    int(remove_obstructing_parked_max_count),
                    max(
                        1,
                        int(round(float(len(stationary_remove_keys)) * float(remove_obstructing_parked_max_fraction))),
                    ),
                )
            )
            remove_keys = {row[3] for row in parked_candidates[:max_remove]}
            if remove_keys:
                kept: List[Dict[str, object]] = []
                removed_ids: List[str] = []
                removed_frames = 0
                for tr in tracks:
                    if not isinstance(tr, dict):
                        kept.append(tr)
                        continue
                    key = _carla_track_key(tr)
                    if key not in remove_keys:
                        kept.append(tr)
                        continue
                    removed_ids.append(str(tr.get("id", key)))
                    frames = tr.get("frames", [])
                    if isinstance(frames, list):
                        removed_frames += int(len(frames))
                tracks[:] = kept
                report["parked_removed_tracks"] = int(len(removed_ids))
                report["parked_removed_frames"] = int(removed_frames)
                report["parked_removed_ids"] = sorted(removed_ids)
                if verbose and removed_ids:
                    print(
                        "[CARLA_OVERLAP] scenario={} removed_obstructing_parked={} ids={}".format(
                            str(scenario_name),
                            int(len(removed_ids)),
                            ",".join(sorted(removed_ids)),
                        )
                    )

    if bool(residual_prune_enabled) and int(residual_prune_max_count) > 0:
        live_keys: List[str] = []
        live_roles: Dict[str, str] = {}
        for tr in tracks:
            if not isinstance(tr, dict):
                continue
            role = str(tr.get("role", "")).strip().lower()
            if role not in {"ego", "vehicle"}:
                continue
            frames = tr.get("frames", [])
            if not isinstance(frames, list) or len(frames) < 2:
                continue
            key = _carla_track_key(tr)
            if key not in track_by_key or key not in tick_to_index:
                continue
            live_keys.append(str(key))
            live_roles[str(key)] = str(role)

        if len(live_keys) >= 2:
            live_key_set = set(live_keys)
            all_tks: set = set()
            for key in live_keys:
                all_tks.update(tick_to_index.get(str(key), {}).keys())

            event_to_pair: Dict[str, Tuple[str, str]] = {}
            events_by_key: Dict[str, set] = {str(k): set() for k in live_keys}

            def _carla_speed_at_frame(frames_local: List[Dict[str, object]], fi: int) -> float:
                if fi < 0 or fi >= len(frames_local):
                    return 0.0
                spd = 0.0
                t = _safe_float(frames_local[fi].get("t"), float(fi) * float(dt_key))
                x = _safe_float(
                    frames_local[fi].get("cx"),
                    _safe_float(frames_local[fi].get("sx"), _safe_float(frames_local[fi].get("x"), 0.0)),
                )
                y = _safe_float(
                    frames_local[fi].get("cy"),
                    _safe_float(frames_local[fi].get("sy"), _safe_float(frames_local[fi].get("y"), 0.0)),
                )
                if fi > 0:
                    t0 = _safe_float(frames_local[fi - 1].get("t"), float(fi - 1) * float(dt_key))
                    x0 = _safe_float(
                        frames_local[fi - 1].get("cx"),
                        _safe_float(frames_local[fi - 1].get("sx"), _safe_float(frames_local[fi - 1].get("x"), x)),
                    )
                    y0 = _safe_float(
                        frames_local[fi - 1].get("cy"),
                        _safe_float(frames_local[fi - 1].get("sy"), _safe_float(frames_local[fi - 1].get("y"), y)),
                    )
                    spd = max(spd, float(math.hypot(float(x) - float(x0), float(y) - float(y0))) / max(5e-2, float(t) - float(t0)))
                if fi + 1 < len(frames_local):
                    t1 = _safe_float(frames_local[fi + 1].get("t"), float(fi + 1) * float(dt_key))
                    x1 = _safe_float(
                        frames_local[fi + 1].get("cx"),
                        _safe_float(frames_local[fi + 1].get("sx"), _safe_float(frames_local[fi + 1].get("x"), x)),
                    )
                    y1 = _safe_float(
                        frames_local[fi + 1].get("cy"),
                        _safe_float(frames_local[fi + 1].get("sy"), _safe_float(frames_local[fi + 1].get("y"), y)),
                    )
                    spd = max(spd, float(math.hypot(float(x1) - float(x), float(y1) - float(y))) / max(5e-2, float(t1) - float(t)))
                return float(spd)

            for tk in sorted(int(v) for v in all_tks):
                active: List[Tuple[str, int]] = []
                for key in live_keys:
                    fi = int(tick_to_index.get(str(key), {}).get(int(tk), -1))
                    if fi < 0:
                        continue
                    frames_local = frames_by_key.get(str(key), [])
                    if fi >= len(frames_local):
                        continue
                    fr = frames_local[fi]
                    px = _safe_float(fr.get("cx"), _safe_float(fr.get("sx"), _safe_float(fr.get("x"), float("nan"))))
                    py = _safe_float(fr.get("cy"), _safe_float(fr.get("sy"), _safe_float(fr.get("y"), float("nan"))))
                    if not (math.isfinite(px) and math.isfinite(py)):
                        continue
                    active.append((str(key), int(fi)))
                if len(active) < 2:
                    continue

                for i in range(len(active)):
                    a_key, a_fi = active[i]
                    a_frames = frames_by_key.get(str(a_key), [])
                    if a_fi >= len(a_frames):
                        continue
                    a_fr = a_frames[a_fi]
                    ax = _safe_float(a_fr.get("cx"), _safe_float(a_fr.get("sx"), _safe_float(a_fr.get("x"), float("nan"))))
                    ay = _safe_float(a_fr.get("cy"), _safe_float(a_fr.get("sy"), _safe_float(a_fr.get("y"), float("nan"))))
                    ayaw = _safe_float(a_fr.get("cyaw"), _safe_float(a_fr.get("yaw"), 0.0))
                    if not (math.isfinite(ax) and math.isfinite(ay)):
                        continue
                    al, aw = dims_by_key.get(str(a_key), (4.6, 2.0))
                    aspd = _carla_speed_at_frame(a_frames, int(a_fi))
                    for j in range(i + 1, len(active)):
                        b_key, b_fi = active[j]
                        b_frames = frames_by_key.get(str(b_key), [])
                        if b_fi >= len(b_frames):
                            continue
                        b_fr = b_frames[b_fi]
                        bx = _safe_float(b_fr.get("cx"), _safe_float(b_fr.get("sx"), _safe_float(b_fr.get("x"), float("nan"))))
                        by = _safe_float(b_fr.get("cy"), _safe_float(b_fr.get("sy"), _safe_float(b_fr.get("y"), float("nan"))))
                        byaw = _safe_float(b_fr.get("cyaw"), _safe_float(b_fr.get("yaw"), 0.0))
                        if not (math.isfinite(bx) and math.isfinite(by)):
                            continue
                        bl, bw = dims_by_key.get(str(b_key), (4.6, 2.0))
                        bspd = _carla_speed_at_frame(b_frames, int(b_fi))
                        if (
                            float(aspd) <= float(residual_prune_speed_gate_mps)
                            and float(bspd) <= float(residual_prune_speed_gate_mps)
                        ):
                            continue
                        center_dist = float(math.hypot(float(ax) - float(bx), float(ay) - float(by)))
                        diag_bound = 0.6 * (math.hypot(float(al), float(aw)) + math.hypot(float(bl), float(bw)))
                        if center_dist > diag_bound + 0.5:
                            continue
                        pen = _obb_overlap_penetration_xyyaw(
                            x1=float(ax),
                            y1=float(ay),
                            yaw1_deg=float(ayaw),
                            len1=float(al),
                            wid1=float(aw),
                            x2=float(bx),
                            y2=float(by),
                            yaw2_deg=float(byaw),
                            len2=float(bl),
                            wid2=float(bw),
                        )
                        if pen < float(residual_prune_pair_pen_min_m):
                            continue

                        arx = _safe_float(a_fr.get("x"), float(ax))
                        ary = _safe_float(a_fr.get("y"), float(ay))
                        aryaw = _safe_float(a_fr.get("yaw"), float(ayaw))
                        brx = _safe_float(b_fr.get("x"), float(bx))
                        bry = _safe_float(b_fr.get("y"), float(by))
                        bryaw = _safe_float(b_fr.get("yaw"), float(byaw))
                        raw_pen = _obb_overlap_penetration_xyyaw(
                            x1=float(arx),
                            y1=float(ary),
                            yaw1_deg=float(aryaw),
                            len1=float(al),
                            wid1=float(aw),
                            x2=float(brx),
                            y2=float(bry),
                            yaw2_deg=float(bryaw),
                            len2=float(bl),
                            wid2=float(bw),
                        )
                        if raw_pen >= float(max(float(residual_prune_raw_pen_ignore_m), 0.75 * float(residual_prune_pair_pen_min_m))):
                            continue

                        k1 = str(min(str(a_key), str(b_key)))
                        k2 = str(max(str(a_key), str(b_key)))
                        ev_id = f"{k1}|{k2}|{int(tk)}"
                        event_to_pair[ev_id] = (k1, k2)
                        events_by_key.setdefault(str(a_key), set()).add(ev_id)
                        events_by_key.setdefault(str(b_key), set()).add(ev_id)

            all_events: set = set(event_to_pair.keys())
            vehicle_keys = [k for k in live_keys if live_roles.get(str(k), "") == "vehicle"]
            if all_events and vehicle_keys:
                max_remove_residual = int(
                    min(
                        int(residual_prune_max_count),
                        max(1, int(round(float(len(vehicle_keys)) * float(residual_prune_max_fraction)))),
                    )
                )
                removed_keys_residual: List[str] = []
                remaining_events: set = set(all_events)
                while remaining_events and len(removed_keys_residual) < max_remove_residual:
                    if len(remaining_events) <= int(residual_prune_target_events):
                        break
                    best_key = ""
                    best_score = -1.0
                    for key in vehicle_keys:
                        if key in removed_keys_residual:
                            continue
                        evs = events_by_key.get(str(key), set()).intersection(remaining_events)
                        if len(evs) < int(residual_prune_min_events):
                            continue
                        tr = track_by_key.get(str(key), {})
                        ms = motion_stats_by_key.get(str(key), _track_raw_motion_stats(tr))
                        net_disp = float(
                            min(
                                float(ms.get("net_disp_m", 0.0)),
                                float(ms.get("robust_net_disp_m", ms.get("net_disp_m", 0.0))),
                            )
                        )
                        is_stationary_like = bool((key in parked_keys) or (key in quasi_parked_keys))
                        avg_speed = float(ms.get("avg_speed_mps", 0.0))
                        p95_step = float(ms.get("p95_step_m", 0.0))
                        sustain_ratio = float(ms.get("sustained_disp_ratio", 0.0))
                        if not is_stationary_like:
                            low_motion_like = bool(
                                float(net_disp) <= float(residual_prune_nonstationary_max_net_disp_m)
                                and float(avg_speed) <= float(residual_prune_nonstationary_max_avg_speed_mps)
                                and float(p95_step) <= float(residual_prune_nonstationary_max_p95_step_m)
                                and float(sustain_ratio) <= float(residual_prune_nonstationary_max_sustain_ratio)
                            )
                            if not low_motion_like:
                                continue
                        if float(net_disp) > float(residual_prune_max_net_disp_m):
                            continue
                        err_arr = raw_err_by_key.get(str(key), np.zeros((0,), dtype=np.float64))
                        raw_med = float(np.median(err_arr)) if isinstance(err_arr, np.ndarray) and err_arr.size > 0 else 0.0
                        cost = 1.0 + 0.08 * min(80.0, float(net_disp)) + 0.10 * min(12.0, float(raw_med))
                        if is_stationary_like:
                            cost *= 0.55
                        score = float(len(evs)) / max(0.2, float(cost))
                        if score > best_score:
                            best_score = float(score)
                            best_key = str(key)
                    if not best_key:
                        break
                    removed_keys_residual.append(str(best_key))
                    remaining_events.difference_update(events_by_key.get(str(best_key), set()))

                if removed_keys_residual:
                    remove_set = set(str(k) for k in removed_keys_residual)
                    kept: List[Dict[str, object]] = []
                    removed_ids: List[str] = []
                    for tr in tracks:
                        if not isinstance(tr, dict):
                            kept.append(tr)
                            continue
                        key = _carla_track_key(tr)
                        if key not in remove_set:
                            kept.append(tr)
                            continue
                        if str(tr.get("role", "")).strip().lower() == "ego":
                            kept.append(tr)
                            continue
                        removed_ids.append(str(tr.get("id", key)))
                    tracks[:] = kept
                    report["residual_pruned_tracks"] = int(len(removed_ids))
                    report["residual_pruned_ids"] = sorted(removed_ids)
                    if verbose and removed_ids:
                        print(
                            "[CARLA_OVERLAP] scenario={} residual_pruned={} ids={}".format(
                                str(scenario_name),
                                int(len(removed_ids)),
                                ",".join(sorted(removed_ids)),
                            )
                        )

    # Last-resort raw fallback recovery: prefer geometric nearest line whenever
    # CARLA topology is available and the snap is not a teleport.
    if bool(raw_fallback_recover_enabled):
        recover_sources = {
            "raw_fallback",
            "raw_fallback_far",
            "non_vehicle_raw",
            "raw_fallback_wrong_way",
        }
        recovered_tracks: set = set()
        recovered_frames = 0
        for tr in tracks:
            if not isinstance(tr, dict):
                continue
            role = str(tr.get("role", "")).strip().lower()
            if role not in {"ego", "vehicle"}:
                continue
            key = _carla_track_key(tr)
            frames = tr.get("frames", [])
            if not isinstance(frames, list) or not frames:
                continue
            local_recovered = 0
            for fi, fr in enumerate(frames):
                src = str(fr.get("csource", "")).strip().lower()
                if src not in recover_sources and not src.startswith("max_divergence_fallback"):
                    continue
                qx, qy, qyaw = _frame_query_pose(fr)
                # For raw-fallback frames, recover from raw actor pose first.
                # Query/smoothed pose can itself be biased away from drivable
                # lanes in these failure cases.
                rx = _safe_float(fr.get("x"), float(qx))
                ry = _safe_float(fr.get("y"), float(qy))
                ryaw = _safe_float(fr.get("yaw"), float(qyaw))
                recover_qx = float(rx) if math.isfinite(rx) else float(qx)
                recover_qy = float(ry) if math.isfinite(ry) else float(qy)
                recover_qyaw = float(ryaw) if math.isfinite(ryaw) else float(qyaw)
                cand = _nearest_projection_any_line(
                    carla_context=carla_context,
                    qx=float(recover_qx),
                    qy=float(recover_qy),
                    qyaw=float(recover_qyaw),
                    enforce_node_direction=True,
                    opposite_reject_deg=float(opposite_reject_deg),
                )
                if cand is None:
                    cand = _nearest_projection_any_line(
                        carla_context=carla_context,
                        qx=float(recover_qx),
                        qy=float(recover_qy),
                        qyaw=float(recover_qyaw),
                        enforce_node_direction=False,
                        opposite_reject_deg=float(opposite_reject_deg),
                    )
                if not isinstance(cand, dict):
                    continue
                proj = cand.get("projection", {})
                dist = _safe_float(proj.get("dist"), float("inf"))
                if not math.isfinite(dist) or float(dist) > float(raw_fallback_recover_max_dist_m):
                    continue
                px = _safe_float(proj.get("x"), float("nan"))
                py = _safe_float(proj.get("y"), float("nan"))
                pyaw = _safe_float(proj.get("yaw"), float("nan"))
                if not (math.isfinite(px) and math.isfinite(py) and math.isfinite(pyaw)):
                    continue
                if fi > 0 and _frame_has_carla_pose(frames[fi - 1]):
                    prev = frames[fi - 1]
                    pcx = _safe_float(prev.get("cx"), float("nan"))
                    pcy = _safe_float(prev.get("cy"), float("nan"))
                    if math.isfinite(pcx) and math.isfinite(pcy):
                        jump = float(math.hypot(float(px) - float(pcx), float(py) - float(pcy)))
                        pqx, pqy, _ = _frame_query_pose(prev)
                        prx = _safe_float(prev.get("x"), float(pqx))
                        pry = _safe_float(prev.get("y"), float(pqy))
                        raw_step = float(math.hypot(float(recover_qx) - float(prx), float(recover_qy) - float(pry)))
                        jump_allow = max(
                            float(raw_fallback_recover_jump_floor_m),
                            float(raw_fallback_recover_jump_ratio) * max(0.2, float(raw_step)),
                        )
                        if float(jump) > float(jump_allow):
                            continue
                _set_carla_pose(
                    frame=fr,
                    line_index=int(_safe_int(cand.get("line_index"), -1)),
                    x=float(px),
                    y=float(py),
                    yaw=float(pyaw),
                    dist=float(dist),
                    source="raw_fallback_recovered_nearest",
                    quality="none",
                )
                local_recovered += 1
            # Fill isolated residual raw-fallback frames from temporal neighbors
            # when geometry is already stable on the same lane.
            for fi in range(1, max(1, len(frames) - 1)):
                fr = frames[fi]
                src = str(fr.get("csource", "")).strip().lower()
                if src not in recover_sources and not src.startswith("max_divergence_fallback"):
                    continue
                prev_fr = frames[fi - 1]
                next_fr = frames[fi + 1]
                if not (_frame_has_carla_pose(prev_fr) and _frame_has_carla_pose(next_fr)):
                    continue
                prev_cli = _safe_int(prev_fr.get("ccli"), -1)
                next_cli = _safe_int(next_fr.get("ccli"), -1)
                if prev_cli < 0 or next_cli < 0 or prev_cli != next_cli:
                    continue
                px0 = _safe_float(prev_fr.get("cx"), float("nan"))
                py0 = _safe_float(prev_fr.get("cy"), float("nan"))
                pyaw0 = _safe_float(prev_fr.get("cyaw"), float("nan"))
                px1 = _safe_float(next_fr.get("cx"), float("nan"))
                py1 = _safe_float(next_fr.get("cy"), float("nan"))
                pyaw1 = _safe_float(next_fr.get("cyaw"), float("nan"))
                if not (
                    math.isfinite(px0)
                    and math.isfinite(py0)
                    and math.isfinite(pyaw0)
                    and math.isfinite(px1)
                    and math.isfinite(py1)
                    and math.isfinite(pyaw1)
                ):
                    continue
                t0 = _safe_float(prev_fr.get("t"), float(fi - 1) * float(dt_key))
                t1 = _safe_float(next_fr.get("t"), float(fi + 1) * float(dt_key))
                t = _safe_float(fr.get("t"), float(fi) * float(dt_key))
                if float(t1) <= float(t0) + 1e-6:
                    alpha = 0.5
                else:
                    alpha = max(0.0, min(1.0, (float(t) - float(t0)) / (float(t1) - float(t0))))
                ix = (1.0 - float(alpha)) * float(px0) + float(alpha) * float(px1)
                iy = (1.0 - float(alpha)) * float(py0) + float(alpha) * float(py1)
                iyaw = _interp_yaw_deg(float(pyaw0), float(pyaw1), float(alpha))
                qx, qy, _ = _frame_query_pose(fr)
                _set_carla_pose(
                    frame=fr,
                    line_index=int(prev_cli),
                    x=float(ix),
                    y=float(iy),
                    yaw=float(iyaw),
                    dist=float(math.hypot(float(ix) - float(qx), float(iy) - float(qy))),
                    source="raw_fallback_recovered_interp",
                    quality="none",
                )
                local_recovered += 1
            if local_recovered > 0:
                recovered_tracks.add(str(key))
                recovered_frames += int(local_recovered)
        report["raw_fallback_recovered_tracks"] = int(len(recovered_tracks))
        report["raw_fallback_recovered_frames"] = int(recovered_frames)

    # Deterministic parked placement invariant:
    # - never preserve parked raw lateral jitter
    # - never place parked actors between active lanes
    # - edge snapping is allowed only when strict outermost-edge proof passes
    if bool(parked_invariant_enabled):
        line_lane_id: Dict[int, int] = {}
        line_road_id: Dict[int, int] = {}
        road_side_max_abs_lane: Dict[Tuple[int, int], int] = {}
        lines_data = carla_context.get("carla_lines_data", [])
        if isinstance(lines_data, list):
            for row in lines_data:
                if not isinstance(row, dict):
                    continue
                ci = _safe_int(row.get("index"), -1)
                rid = _safe_int(row.get("road_id"), 0)
                lid = _safe_int(row.get("lane_id"), 0)
                if ci < 0 or rid == 0 or lid == 0:
                    continue
                line_lane_id[int(ci)] = int(lid)
                line_road_id[int(ci)] = int(rid)
                side = 1 if int(lid) > 0 else -1
                key = (int(rid), int(side))
                road_side_max_abs_lane[key] = max(
                    int(road_side_max_abs_lane.get(key, 0)),
                    int(abs(int(lid))),
                )

        active_frames_by_key: Dict[str, List[Dict[str, object]]] = {}
        active_track_by_key: Dict[str, Dict[str, object]] = {}
        active_ticks: Dict[str, Dict[int, int]] = {}
        active_dims: Dict[str, Tuple[float, float]] = {}
        active_motion: Dict[str, Dict[str, float]] = {}
        parked_keys_final: set = set()
        moving_keys_final: set = set()

        for tr in tracks:
            if not isinstance(tr, dict):
                continue
            role = str(tr.get("role", "")).strip().lower()
            if role not in {"ego", "vehicle"}:
                continue
            frames = tr.get("frames", [])
            if not isinstance(frames, list) or len(frames) < 2:
                continue
            key = _carla_track_key(tr)
            active_track_by_key[str(key)] = tr
            active_frames_by_key[str(key)] = frames
            active_dims[str(key)] = _vehicle_dims_for_overlap(tr)
            motion_stats = _track_raw_motion_stats(tr)
            active_motion[str(key)] = motion_stats
            idx_map: Dict[int, int] = {}
            for i, fr in enumerate(frames):
                t = _safe_float(fr.get("t"), float(i) * float(dt_key))
                tk = int(round(float(t) * float(inv_dt)))
                if tk not in idx_map:
                    idx_map[int(tk)] = int(i)
            active_ticks[str(key)] = idx_map

            parked_like = bool(role == "vehicle" and _is_parked_vehicle_track_for_overlap(tr))
            if (
                (not bool(parked_like))
                and bool(role == "vehicle")
                and bool(parked_invariant_far_static_enabled)
            ):
                robust_net = float(motion_stats.get("robust_net_disp_m", motion_stats.get("net_disp_m", 0.0)))
                avg_speed = float(motion_stats.get("avg_speed_mps", 0.0))
                if (
                    robust_net <= float(parked_invariant_far_static_net_max_m)
                    and avg_speed <= float(parked_invariant_far_static_speed_max_mps)
                ):
                    dists: List[float] = []
                    for fi in range(0, len(frames), max(1, int(parked_invariant_sample_stride))):
                        qx, qy, qyaw = _frame_query_pose(frames[fi])
                        cand = _nearest_projection_any_line(
                            carla_context=carla_context,
                            qx=float(qx),
                            qy=float(qy),
                            qyaw=float(qyaw),
                            enforce_node_direction=False,
                            opposite_reject_deg=float(opposite_reject_deg),
                        )
                        if not isinstance(cand, dict):
                            continue
                        proj = cand.get("projection", {})
                        dd = _safe_float(proj.get("dist"), float("inf"))
                        if math.isfinite(dd):
                            dists.append(float(dd))
                    if len(dists) >= 5:
                        arr_d = np.asarray(dists, dtype=np.float64)
                        med_d = float(np.median(arr_d))
                        iqr_d = float(np.percentile(arr_d, 75.0) - np.percentile(arr_d, 25.0))
                        if (
                            med_d >= float(parked_invariant_far_static_lane_med_min_m)
                            and iqr_d <= float(parked_invariant_far_static_lane_iqr_max_m)
                        ):
                            parked_like = True

            if bool(role == "vehicle" and parked_like):
                parked_keys_final.add(str(key))
            else:
                moving_keys_final.add(str(key))

        parked_tracks_applied = 0
        parked_frames_applied = 0
        parked_edge_tracks = 0
        parked_centerline_tracks = 0
        parked_outermost_tiebreak_tracks = 0
        parked_overlap_conflict_edge_tracks = 0
        parked_inner_to_outer_centerline_tracks = 0
        parked_edge_reject_reasons: Dict[str, int] = {}

        def _line_is_outermost(line_idx: int) -> bool:
            rid = int(line_road_id.get(int(line_idx), 0))
            lid = int(line_lane_id.get(int(line_idx), 0))
            if rid == 0 or lid == 0:
                return False
            side = 1 if int(lid) > 0 else -1
            max_abs = int(road_side_max_abs_lane.get((int(rid), int(side)), 0))
            if max_abs <= 0:
                return False
            return bool(int(abs(int(lid))) >= int(max_abs))

        def _parked_overlap_conflict_stats(track_key: str, line_idx: int) -> Dict[str, float]:
            frames_local = active_frames_by_key.get(str(track_key), [])
            ticks_local = active_ticks.get(str(track_key), {})
            dims_local = active_dims.get(str(track_key), (4.8, 2.0))
            if not isinstance(frames_local, list) or not isinstance(ticks_local, dict) or not ticks_local:
                return {
                    "frames": 0.0,
                    "ratio": 0.0,
                    "pen_sum": 0.0,
                    "max_pen": 0.0,
                    "same_line_hits": 0.0,
                }
            l0 = float(dims_local[0]) if isinstance(dims_local, (list, tuple)) and len(dims_local) >= 2 else 4.8
            w0 = float(dims_local[1]) if isinstance(dims_local, (list, tuple)) and len(dims_local) >= 2 else 2.0
            overlap_frames = 0
            same_line_hits = 0
            pen_sum = 0.0
            max_pen = 0.0
            for tk, fi in ticks_local.items():
                if fi < 0 or fi >= len(frames_local):
                    continue
                fr = frames_local[int(fi)]
                if not _frame_has_carla_pose(fr):
                    continue
                ax = float(_safe_float(fr.get("cx"), _safe_float(fr.get("x"), 0.0)))
                ay = float(_safe_float(fr.get("cy"), _safe_float(fr.get("y"), 0.0)))
                ayaw = float(_safe_float(fr.get("cyaw"), _safe_float(fr.get("yaw"), 0.0)))
                frame_pen = 0.0
                frame_same = False
                partner_keys = set(moving_keys_final) | {str(k) for k in parked_keys_final if str(k) != str(track_key)}
                for mkey in partner_keys:
                    m_ticks = active_ticks.get(str(mkey), {})
                    if not isinstance(m_ticks, dict):
                        continue
                    m_fi = m_ticks.get(int(tk))
                    if m_fi is None:
                        continue
                    m_frames = active_frames_by_key.get(str(mkey), [])
                    if not isinstance(m_frames, list) or m_fi < 0 or m_fi >= len(m_frames):
                        continue
                    m_fr = m_frames[int(m_fi)]
                    if not _frame_has_carla_pose(m_fr):
                        continue
                    dims_m = active_dims.get(str(mkey), (4.8, 2.0))
                    l1 = float(dims_m[0]) if isinstance(dims_m, (list, tuple)) and len(dims_m) >= 2 else 4.8
                    w1 = float(dims_m[1]) if isinstance(dims_m, (list, tuple)) and len(dims_m) >= 2 else 2.0
                    bx = float(_safe_float(m_fr.get("cx"), _safe_float(m_fr.get("x"), 0.0)))
                    by = float(_safe_float(m_fr.get("cy"), _safe_float(m_fr.get("y"), 0.0)))
                    byaw = float(_safe_float(m_fr.get("cyaw"), _safe_float(m_fr.get("yaw"), 0.0)))
                    pen = float(
                        _obb_overlap_penetration_xyyaw(
                            float(ax),
                            float(ay),
                            float(ayaw),
                            float(l0),
                            float(w0),
                            float(bx),
                            float(by),
                            float(byaw),
                            float(l1),
                            float(w1),
                        )
                    )
                    if pen < float(parked_overlap_conflict_min_pair_pen_m):
                        continue
                    frame_pen += float(pen)
                    max_pen = max(float(max_pen), float(pen))
                    if int(_safe_int(m_fr.get("ccli"), -1)) == int(line_idx):
                        frame_same = True
                if frame_pen > 0.0:
                    overlap_frames += 1
                    pen_sum += float(frame_pen)
                    if bool(frame_same):
                        same_line_hits += 1
            ratio = float(overlap_frames) / float(max(1, len(ticks_local)))
            return {
                "frames": float(overlap_frames),
                "ratio": float(ratio),
                "pen_sum": float(pen_sum),
                "max_pen": float(max_pen),
                "same_line_hits": float(same_line_hits),
            }

        def _strict_outer_edge_proof(
            frames_local: List[Dict[str, object]],
            sample_rows_local: List[Tuple[int, int, float, float, float, int, int, float]],
            committed_line_local: int,
            nearest_med_local: float,
            relaxed: bool = False,
        ) -> Tuple[bool, float, str]:
            if not bool(parked_invariant_edge_proof_enabled):
                return (False, 0.0, "edge_proof_disabled")
            if not sample_rows_local:
                return (False, 0.0, "insufficient_samples")
            nearest_med_max = float(parked_invariant_edge_proof_nearest_med_max_m)
            sample_max_dist = float(parked_invariant_edge_proof_sample_max_dist_m)
            raw_sign_min = float(parked_invariant_edge_proof_raw_sign_consistency_min)
            raw_abs_med_min = float(parked_invariant_edge_proof_raw_abs_med_min_m)
            if bool(relaxed):
                nearest_med_max = max(float(nearest_med_max), float(parked_overlap_conflict_relaxed_nearest_med_max_m))
                sample_max_dist = max(float(sample_max_dist), float(parked_overlap_conflict_relaxed_sample_max_dist_m))
                raw_sign_min = min(float(raw_sign_min), float(parked_overlap_conflict_relaxed_raw_sign_min))
                raw_abs_med_min = min(float(raw_abs_med_min), float(parked_overlap_conflict_relaxed_raw_abs_med_min_m))
            if not math.isfinite(float(nearest_med_local)) or float(nearest_med_local) > float(nearest_med_max):
                return (False, 0.0, "lane_too_far")
            road_id_local = int(line_road_id.get(int(committed_line_local), 0))
            lane_id_local = int(line_lane_id.get(int(committed_line_local), 0))
            if road_id_local == 0 or lane_id_local == 0:
                return (False, 0.0, "missing_lane_meta")
            side_local = 1 if int(lane_id_local) > 0 else -1
            max_abs_local = int(road_side_max_abs_lane.get((int(road_id_local), int(side_local)), 0))
            if max_abs_local <= 0 or int(abs(int(lane_id_local))) < int(max_abs_local):
                return (False, 0.0, "not_metadata_outermost")

            outward_votes: List[float] = []
            raw_sign_hits = 0
            raw_sign_total = 0
            raw_abs_vals: List[float] = []

            for fi, _, base_px, base_py, base_yaw, _, _, lat_raw in sample_rows_local:
                if fi < 0 or fi >= len(frames_local):
                    return (False, 0.0, "sample_oob")
                qx, qy, qyaw = _frame_query_pose(frames_local[int(fi)])
                cc = _best_projection_on_carla_line(
                    carla_context=carla_context,
                    line_index=int(committed_line_local),
                    qx=float(qx),
                    qy=float(qy),
                    qyaw=float(qyaw),
                    prefer_reversed=None,
                    enforce_node_direction=False,
                    opposite_reject_deg=float(opposite_reject_deg),
                )
                if not isinstance(cc, dict):
                    return (False, 0.0, "missing_committed_projection")
                proj0 = cc.get("projection", {})
                dist0 = _safe_float(proj0.get("dist"), float("inf"))
                if not math.isfinite(dist0) or float(dist0) > float(sample_max_dist):
                    return (False, 0.0, "sample_too_far")
                px0 = _safe_float(proj0.get("x"), float(base_px))
                py0 = _safe_float(proj0.get("y"), float(base_py))
                yaw0 = _safe_float(proj0.get("yaw"), float(base_yaw))
                if not (math.isfinite(px0) and math.isfinite(py0) and math.isfinite(yaw0)):
                    return (False, 0.0, "invalid_committed_projection")
                nyaw0 = math.radians(float(yaw0))
                nx0 = -math.sin(float(nyaw0))
                ny0 = math.cos(float(nyaw0))

                lat_by_line: Dict[int, float] = {int(committed_line_local): 0.0}
                near_ids = _nearby_carla_lines(
                    carla_context=carla_context,
                    qx=float(qx),
                    qy=float(qy),
                    top_k=int(parked_invariant_edge_proof_family_top_k),
                )
                for ci in near_ids:
                    road_ci = int(line_road_id.get(int(ci), 0))
                    if road_ci != int(road_id_local):
                        continue
                    cc_i = _best_projection_on_carla_line(
                        carla_context=carla_context,
                        line_index=int(ci),
                        qx=float(qx),
                        qy=float(qy),
                        qyaw=float(qyaw),
                        prefer_reversed=None,
                        enforce_node_direction=False,
                        opposite_reject_deg=float(opposite_reject_deg),
                    )
                    if not isinstance(cc_i, dict):
                        continue
                    proj_i = cc_i.get("projection", {})
                    di = _safe_float(proj_i.get("dist"), float("inf"))
                    if not math.isfinite(di) or float(di) > float(parked_invariant_edge_proof_family_dist_max_m):
                        continue
                    yi = _safe_float(proj_i.get("yaw"), float("nan"))
                    if not math.isfinite(yi):
                        continue
                    if float(_yaw_abs_diff_deg(float(yaw0), float(yi))) > float(parked_invariant_edge_proof_parallel_max_deg):
                        continue
                    xi = _safe_float(proj_i.get("x"), float("nan"))
                    yi_pos = _safe_float(proj_i.get("y"), float("nan"))
                    if not (math.isfinite(xi) and math.isfinite(yi_pos)):
                        continue
                    lat_by_line[int(ci)] = float((float(xi) - float(px0)) * float(nx0) + (float(yi_pos) - float(py0)) * float(ny0))

                other_lats = [float(v) for k, v in lat_by_line.items() if int(k) != int(committed_line_local)]
                if len(other_lats) <= 0:
                    if bool(relaxed):
                        outward_sign_local = -1.0 if int(lane_id_local) > 0 else 1.0
                    else:
                        return (False, 0.0, "insufficient_family")
                else:
                    tol = float(parked_invariant_edge_proof_boundary_tol_m)
                    all_nonneg = all(float(v) >= -float(tol) for v in other_lats)
                    all_nonpos = all(float(v) <= float(tol) for v in other_lats)
                    if bool(all_nonneg) == bool(all_nonpos):
                        if bool(relaxed):
                            outward_sign_local = -1.0 if int(lane_id_local) > 0 else 1.0
                        else:
                            return (False, 0.0, "interior_or_ambiguous_boundary")
                    elif bool(all_nonneg):
                        outward_sign_local = -1.0
                        if (not bool(relaxed)) and float(max(other_lats)) < float(parked_invariant_edge_proof_min_other_sep_m):
                            return (False, 0.0, "weak_boundary_separation")
                    else:
                        outward_sign_local = 1.0
                        if (not bool(relaxed)) and float(abs(min(other_lats))) < float(parked_invariant_edge_proof_min_other_sep_m):
                            return (False, 0.0, "weak_boundary_separation")

                tol = float(parked_invariant_edge_proof_boundary_tol_m)
                outward_votes.append(float(outward_sign_local))

                if math.isfinite(float(lat_raw)):
                    raw_abs_vals.append(float(abs(float(lat_raw))))
                    raw_sign_total += 1
                    if float(lat_raw) * float(outward_sign_local) > float(tol):
                        raw_sign_hits += 1

            if not outward_votes:
                return (False, 0.0, "missing_outward_votes")
            out_sign = float(np.median(np.asarray(outward_votes, dtype=np.float64)))
            if any((float(v) * float(out_sign)) <= 0.0 for v in outward_votes):
                return (False, 0.0, "side_flip")
            if raw_sign_total <= 0 or not raw_abs_vals:
                return (False, 0.0, "missing_raw_support")
            raw_ratio = float(raw_sign_hits) / float(max(1, raw_sign_total))
            if float(raw_ratio) < float(raw_sign_min):
                return (False, 0.0, "raw_sign_inconsistent")
            raw_abs_med = float(np.median(np.asarray(raw_abs_vals, dtype=np.float64)))
            if float(raw_abs_med) < float(raw_abs_med_min):
                return (False, 0.0, "raw_offset_too_small")
            return (True, float(1.0 if out_sign >= 0.0 else -1.0), "ok")

        for key in sorted(parked_keys_final):
            frames = active_frames_by_key.get(str(key), [])
            if not isinstance(frames, list) or not frames:
                continue
            tr_dbg = active_track_by_key.get(str(key), {})
            tr_dbg_id = str(tr_dbg.get("id", str(key))) if isinstance(tr_dbg, dict) else str(key)
            debug_this_parked = bool(str(key) in debug_track_ids or str(tr_dbg_id) in debug_track_ids)

            sample_rows: List[Tuple[int, int, float, float, float, int, int, float]] = []
            for fi in range(0, len(frames), max(1, int(parked_invariant_sample_stride))):
                fr = frames[fi]
                qx, qy, qyaw = _frame_query_pose(fr)
                cand = _nearest_projection_any_line(
                    carla_context=carla_context,
                    qx=float(qx),
                    qy=float(qy),
                    qyaw=float(qyaw),
                    enforce_node_direction=False,
                    opposite_reject_deg=float(opposite_reject_deg),
                )
                if not isinstance(cand, dict):
                    continue
                proj = cand.get("projection", {})
                line_idx = int(_safe_int(cand.get("line_index"), -1))
                px = _safe_float(proj.get("x"), float("nan"))
                py = _safe_float(proj.get("y"), float("nan"))
                pyaw = _safe_float(proj.get("yaw"), float("nan"))
                dist = _safe_float(proj.get("dist"), float("inf"))
                if not (
                    line_idx >= 0
                    and math.isfinite(px)
                    and math.isfinite(py)
                    and math.isfinite(pyaw)
                    and math.isfinite(dist)
                ):
                    continue
                rid = int(line_road_id.get(int(line_idx), 0))
                lid = int(line_lane_id.get(int(line_idx), 0))
                nyaw = math.radians(float(pyaw))
                nx = -math.sin(float(nyaw))
                ny = math.cos(float(nyaw))
                rx = _safe_float(fr.get("x"), float(qx))
                ry = _safe_float(fr.get("y"), float(qy))
                lat_raw = (float(rx) - float(px)) * float(nx) + (float(ry) - float(py)) * float(ny)
                sample_rows.append(
                    (
                        int(fi),
                        int(line_idx),
                        float(px),
                        float(py),
                        float(pyaw),
                        int(rid),
                        int(lid),
                        float(lat_raw),
                    )
                )

            if not sample_rows:
                continue

            line_counts: Dict[int, int] = {}
            line_dist: Dict[int, List[float]] = {}
            dist_vals: List[float] = []
            for fi, li, px, py, pyaw, rid, lid, lat_raw in sample_rows:
                line_counts[int(li)] = int(line_counts.get(int(li), 0)) + 1
                qx, qy, _ = _frame_query_pose(frames[int(fi)])
                dist_vals.append(float(math.hypot(float(px) - float(qx), float(py) - float(qy))))
                line_dist.setdefault(int(li), []).append(float(dist_vals[-1]))

            if not line_counts:
                continue

            nearest_med = float(np.median(np.asarray(dist_vals, dtype=np.float64))) if dist_vals else float("inf")
            line_rank: List[Tuple[int, float, int]] = []
            for li, cnt in line_counts.items():
                d_arr = np.asarray(line_dist.get(int(li), [float("inf")]), dtype=np.float64)
                med_d = float(np.median(d_arr)) if d_arr.size > 0 else float("inf")
                line_rank.append((int(li), float(med_d), int(cnt)))
            line_rank.sort(key=lambda row: (-int(row[2]), float(row[1]), int(row[0])))
            if not line_rank:
                continue
            base_line = int(line_rank[0][0])
            committed_line = int(base_line)
            base_count = int(line_rank[0][2])
            base_med = float(line_rank[0][1])

            raw_dist_cache: Dict[int, float] = {}
            sample_cache_by_line: Dict[int, List[Tuple[int, float, float, float, float]]] = {}

            def _raw_median_dist_for_line(line_idx: int) -> float:
                if int(line_idx) in raw_dist_cache:
                    return float(raw_dist_cache[int(line_idx)])
                vals: List[float] = []
                cached_rows: List[Tuple[int, float, float, float, float]] = []
                for fi2, _, _, _, _, _, _, _ in sample_rows:
                    if fi2 < 0 or fi2 >= len(frames):
                        continue
                    fr2 = frames[int(fi2)]
                    rx = _safe_float(fr2.get("x"), float("nan"))
                    ry = _safe_float(fr2.get("y"), float("nan"))
                    ryaw = _safe_float(fr2.get("yaw"), float("nan"))
                    if not (math.isfinite(rx) and math.isfinite(ry) and math.isfinite(ryaw)):
                        qx2, qy2, qyaw2 = _frame_query_pose(fr2)
                        rx = float(qx2)
                        ry = float(qy2)
                        ryaw = float(qyaw2)
                    cc2 = _best_projection_on_carla_line(
                        carla_context=carla_context,
                        line_index=int(line_idx),
                        qx=float(rx),
                        qy=float(ry),
                        qyaw=float(ryaw),
                        prefer_reversed=None,
                        enforce_node_direction=False,
                        opposite_reject_deg=float(opposite_reject_deg),
                    )
                    if not isinstance(cc2, dict):
                        continue
                    proj2 = cc2.get("projection", {})
                    d2 = _safe_float(proj2.get("dist"), float("inf"))
                    px2 = _safe_float(proj2.get("x"), float("nan"))
                    py2 = _safe_float(proj2.get("y"), float("nan"))
                    pyaw2 = _safe_float(proj2.get("yaw"), float("nan"))
                    if not (math.isfinite(d2) and math.isfinite(px2) and math.isfinite(py2) and math.isfinite(pyaw2)):
                        continue
                    vals.append(float(d2))
                    cached_rows.append((int(fi2), float(px2), float(py2), float(pyaw2), float(d2)))
                med = float(np.median(np.asarray(vals, dtype=np.float64))) if vals else float("inf")
                raw_dist_cache[int(line_idx)] = float(med)
                sample_cache_by_line[int(line_idx)] = list(cached_rows)
                return float(med)

            def _adjacent_outer_pair(line_a: int, line_b: int) -> Optional[Tuple[int, int]]:
                rid_a = int(line_road_id.get(int(line_a), 0))
                rid_b = int(line_road_id.get(int(line_b), 0))
                lid_a = int(line_lane_id.get(int(line_a), 0))
                lid_b = int(line_lane_id.get(int(line_b), 0))
                if rid_a == 0 or rid_b == 0 or lid_a == 0 or lid_b == 0:
                    return None
                if int(rid_a) != int(rid_b):
                    return None
                side_a = 1 if int(lid_a) > 0 else -1
                side_b = 1 if int(lid_b) > 0 else -1
                if int(side_a) != int(side_b):
                    return None
                aa = int(abs(int(lid_a)))
                bb = int(abs(int(lid_b)))
                if abs(int(aa) - int(bb)) != 1:
                    return None
                max_abs_local = int(road_side_max_abs_lane.get((int(rid_a), int(side_a)), 0))
                if max_abs_local <= 0:
                    return None
                if max(int(aa), int(bb)) < int(max_abs_local):
                    return None
                outer = int(line_a) if int(aa) > int(bb) else int(line_b)
                inner = int(line_b) if int(aa) > int(bb) else int(line_a)
                if int(abs(int(line_lane_id.get(int(outer), 0)))) != int(max_abs_local):
                    return None
                return (int(outer), int(inner))

            # Goal A: ambiguity-only tie-breaker to prefer outermost lane centerline
            # when parked placement has comparable evidence.
            if bool(parked_outermost_tiebreak_enabled) and len(line_rank) >= 2:
                li2 = int(line_rank[1][0])
                med2 = float(line_rank[1][1])
                cnt2 = int(line_rank[1][2])
                pair = _adjacent_outer_pair(int(base_line), int(li2))
                if pair is not None:
                    outer_li, inner_li = int(pair[0]), int(pair[1])
                    outer_cnt = int(line_counts.get(int(outer_li), 0))
                    inner_cnt = int(line_counts.get(int(inner_li), 0))
                    outer_q = float(np.median(np.asarray(line_dist.get(int(outer_li), [float("inf")]), dtype=np.float64)))
                    inner_q = float(np.median(np.asarray(line_dist.get(int(inner_li), [float("inf")]), dtype=np.float64)))
                    outer_raw = float(_raw_median_dist_for_line(int(outer_li)))
                    inner_raw = float(_raw_median_dist_for_line(int(inner_li)))
                    ambiguous = (
                        abs(int(outer_cnt) - int(inner_cnt)) <= int(parked_outermost_tiebreak_count_gap_max)
                        and abs(float(outer_q) - float(inner_q)) <= float(parked_outermost_tiebreak_query_gap_max_m)
                    )
                    outer_supported = bool(
                        math.isfinite(outer_raw)
                        and math.isfinite(inner_raw)
                        and float(outer_raw) <= float(inner_raw) + float(parked_outermost_tiebreak_raw_gap_max_m)
                    )
                    if bool(ambiguous) and bool(outer_supported):
                        committed_line = int(outer_li)
                        parked_outermost_tiebreak_tracks += 1
            if bool(parked_outermost_tiebreak_enabled) and (not bool(_line_is_outermost(int(committed_line)))):
                rid0 = int(line_road_id.get(int(committed_line), 0))
                lid0 = int(line_lane_id.get(int(committed_line), 0))
                if rid0 != 0 and lid0 != 0:
                    side0 = 1 if int(lid0) > 0 else -1
                    abs0 = int(abs(int(lid0)))
                    max_abs0 = int(road_side_max_abs_lane.get((int(rid0), int(side0)), 0))
                    if max_abs0 > 0 and int(abs0) < int(max_abs0):
                        outer_candidates: set = set()
                        # Search ALL known CARLA lines (not just observed ones) so that
                        # outermost lanes never traversed by any actor are still found.
                        for li in line_road_id.keys():
                            rid_i = int(line_road_id.get(int(li), 0))
                            lid_i = int(line_lane_id.get(int(li), 0))
                            if rid_i != int(rid0) or lid_i == 0:
                                continue
                            side_i = 1 if int(lid_i) > 0 else -1
                            if int(side_i) != int(side0):
                                continue
                            if int(abs(int(lid_i))) == int(max_abs0):
                                outer_candidates.add(int(li))
                        for fi2, _, _, _, _, _, _, _ in sample_rows:
                            if fi2 < 0 or fi2 >= len(frames):
                                continue
                            qx2, qy2, _ = _frame_query_pose(frames[int(fi2)])
                            near_ids2 = _nearby_carla_lines(
                                carla_context=carla_context,
                                qx=float(qx2),
                                qy=float(qy2),
                                top_k=12,
                            )
                            for li in near_ids2:
                                rid_i = int(line_road_id.get(int(li), 0))
                                lid_i = int(line_lane_id.get(int(li), 0))
                                if rid_i != int(rid0) or lid_i == 0:
                                    continue
                                side_i = 1 if int(lid_i) > 0 else -1
                                if int(side_i) != int(side0):
                                    continue
                                if int(abs(int(lid_i))) == int(max_abs0):
                                    outer_candidates.add(int(li))
                        if outer_candidates:
                            outer_line = int(min(outer_candidates, key=lambda li: float(_raw_median_dist_for_line(int(li)))))
                            base_raw = float(_raw_median_dist_for_line(int(committed_line)))
                            outer_raw = float(_raw_median_dist_for_line(int(outer_line)))
                            # Topology-only enforcement: always commit to the outermost
                            # lane as long as it is within a generous spatial tolerance
                            # (same road/side already guaranteed above).
                            if (
                                math.isfinite(outer_raw)
                                and float(outer_raw) <= float(base_raw) + float(parked_outermost_force_raw_gap_max_m)
                            ):
                                committed_line = int(outer_line)
                                parked_outermost_tiebreak_tracks += 1

            conflict_stats = _parked_overlap_conflict_stats(str(key), int(committed_line))
            conflict_strong = bool(
                int(round(float(conflict_stats.get("frames", 0.0)))) >= int(parked_overlap_conflict_min_frames)
                and (
                    float(conflict_stats.get("ratio", 0.0)) >= float(parked_overlap_conflict_min_ratio)
                    or float(conflict_stats.get("pen_sum", 0.0)) >= float(parked_overlap_conflict_min_pen_sum_m)
                )
            )
            if bool(debug_this_parked):
                print(
                    "[PARKED_DBG] id={} line={} nearest_med={:.3f} conflict(frames={:.1f}, ratio={:.3f}, pen={:.3f}, same_hits={:.1f})".format(
                        str(tr_dbg_id),
                        int(committed_line),
                        float(nearest_med),
                        float(conflict_stats.get("frames", 0.0)),
                        float(conflict_stats.get("ratio", 0.0)),
                        float(conflict_stats.get("pen_sum", 0.0)),
                        float(conflict_stats.get("same_line_hits", 0.0)),
                    )
                )

            # Goal B edge-case: only with extreme evidence, allow promoting an
            # inner parked lane to the outermost centerline.
            if bool(parked_inner_to_outer_enabled) and bool(conflict_strong):
                rid0 = int(line_road_id.get(int(committed_line), 0))
                lid0 = int(line_lane_id.get(int(committed_line), 0))
                if rid0 != 0 and lid0 != 0:
                    side0 = 1 if int(lid0) > 0 else -1
                    max_abs0 = int(road_side_max_abs_lane.get((int(rid0), int(side0)), 0))
                    abs0 = int(abs(int(lid0)))
                    if max_abs0 > 0 and abs0 == max(1, int(max_abs0) - 1):
                        outer_candidates = [
                            int(li)
                            for li in line_counts.keys()
                            if int(line_road_id.get(int(li), 0)) == int(rid0)
                            and int(line_lane_id.get(int(li), 0)) != 0
                            and (1 if int(line_lane_id.get(int(li), 0)) > 0 else -1) == int(side0)
                            and int(abs(int(line_lane_id.get(int(li), 0)))) == int(max_abs0)
                        ]
                        if outer_candidates:
                            outer_li = int(min(outer_candidates, key=lambda li: float(_raw_median_dist_for_line(int(li)))))
                            inner_raw = float(_raw_median_dist_for_line(int(committed_line)))
                            outer_raw = float(_raw_median_dist_for_line(int(outer_li)))
                            if (
                                math.isfinite(inner_raw)
                                and math.isfinite(outer_raw)
                                and float(outer_raw) + float(parked_inner_to_outer_raw_gain_min_m) < float(inner_raw)
                            ):
                                committed_line = int(outer_li)
                                parked_inner_to_outer_centerline_tracks += 1

            proof_ok, outward_sign, proof_reason = _strict_outer_edge_proof(
                frames_local=frames,
                sample_rows_local=sample_rows,
                committed_line_local=int(committed_line),
                nearest_med_local=float(nearest_med),
                relaxed=False,
            )
            use_curb = bool(proof_ok)
            lat_vals_forced = [float(row[7]) for row in sample_rows if math.isfinite(float(row[7]))]
            lat_abs_med_forced = float(np.median(np.abs(np.asarray(lat_vals_forced, dtype=np.float64)))) if lat_vals_forced else 0.0
            sign_ratio_forced = 0.0
            if lat_vals_forced:
                lat_arr_forced = np.asarray(lat_vals_forced, dtype=np.float64)
                pos_forced = int(np.sum(lat_arr_forced >= 0.0))
                neg_forced = int(np.sum(lat_arr_forced <= 0.0))
                sign_ratio_forced = float(max(pos_forced, neg_forced)) / float(max(1, len(lat_vals_forced)))
            far_side_raw_support = bool(
                bool(_line_is_outermost(int(committed_line)))
                and bool(lat_vals_forced)
                and float(lat_abs_med_forced) >= float(parked_outermost_far_edge_raw_abs_min_m)
                and float(sign_ratio_forced) >= float(parked_outermost_far_edge_sign_ratio_min)
            )
            if (
                (not bool(use_curb))
                and bool(_line_is_outermost(int(committed_line)))
                and (bool(conflict_strong) or bool(far_side_raw_support))
            ):
                proof_ok_relaxed, outward_sign_relaxed, proof_reason_relaxed = _strict_outer_edge_proof(
                    frames_local=frames,
                    sample_rows_local=sample_rows,
                    committed_line_local=int(committed_line),
                    nearest_med_local=float(nearest_med),
                    relaxed=True,
                )
                if bool(proof_ok_relaxed):
                    use_curb = True
                    outward_sign = float(outward_sign_relaxed)
                    proof_reason = str(proof_reason_relaxed)
                    if bool(conflict_strong):
                        parked_overlap_conflict_edge_tracks += 1
                else:
                    same_hits = float(conflict_stats.get("same_line_hits", 0.0))
                    conf_frames = max(1.0, float(conflict_stats.get("frames", 0.0)))
                    same_ratio = float(same_hits) / float(conf_frames)
                    if bool(conflict_strong) and float(same_ratio) >= float(parked_overlap_conflict_force_same_ratio_min):
                        lat_vals = [float(row[7]) for row in sample_rows if math.isfinite(float(row[7]))]
                        lat_med = float(np.median(np.asarray(lat_vals, dtype=np.float64))) if lat_vals else 0.0
                        if abs(float(lat_med)) >= 0.20:
                            outward_sign = float(1.0 if float(lat_med) >= 0.0 else -1.0)
                        else:
                            lid_local = int(line_lane_id.get(int(committed_line), 0))
                            outward_sign = float(-1.0 if int(lid_local) > 0 else 1.0)
                        use_curb = True
                        proof_reason = "conflict_forced"
                        parked_overlap_conflict_edge_tracks += 1
            if (not bool(use_curb)) and bool(_line_is_outermost(int(committed_line))) and bool(conflict_strong):
                same_hits_far = float(conflict_stats.get("same_line_hits", 0.0))
                conf_frames_far = max(1.0, float(conflict_stats.get("frames", 0.0)))
                same_ratio_far = float(same_hits_far) / float(conf_frames_far)
                lat_vals = [float(row[7]) for row in sample_rows if math.isfinite(float(row[7]))]
                if lat_vals and float(same_ratio_far) >= float(parked_overlap_conflict_force_same_ratio_min):
                    lat_arr = np.asarray(lat_vals, dtype=np.float64)
                    lat_abs_med = float(np.median(np.abs(lat_arr)))
                    pos = int(np.sum(lat_arr >= 0.0))
                    neg = int(np.sum(lat_arr <= 0.0))
                    sign_ratio = float(max(pos, neg)) / float(max(1, len(lat_vals)))
                    if (
                        float(lat_abs_med) >= float(parked_outermost_far_edge_raw_abs_min_m)
                        and float(sign_ratio) >= float(parked_outermost_far_edge_sign_ratio_min)
                    ):
                        lat_med = float(np.median(lat_arr))
                        outward_sign = float(1.0 if float(lat_med) >= 0.0 else -1.0)
                        use_curb = True
                        proof_reason = "outermost_far_raw"
            if bool(use_curb):
                if not bool(_line_is_outermost(int(committed_line))):
                    use_curb = False
                    proof_reason = "not_outermost_committed"
                else:
                    # Hard invariant: curb offset must always point to the metadata
                    # outer side of the committed outermost lane.
                    lid_committed = int(line_lane_id.get(int(committed_line), 0))
                    if int(lid_committed) != 0:
                        outward_sign = float(-1.0 if int(lid_committed) > 0 else 1.0)
            if bool(debug_this_parked):
                lat_vals_dbg = [float(row[7]) for row in sample_rows if math.isfinite(float(row[7]))]
                lat_med_dbg = float(np.median(np.asarray(lat_vals_dbg, dtype=np.float64))) if lat_vals_dbg else float("nan")
                lat_abs_dbg = float(np.median(np.abs(np.asarray(lat_vals_dbg, dtype=np.float64)))) if lat_vals_dbg else float("nan")
                print(
                    "[PARKED_DBG] id={} curb={} reason={} out_sign={:.1f} lat_med={} lat_abs_med={}".format(
                        str(tr_dbg_id),
                        int(bool(use_curb)),
                        str(proof_reason),
                        float(outward_sign),
                        f"{float(lat_med_dbg):.3f}" if math.isfinite(float(lat_med_dbg)) else "nan",
                        f"{float(lat_abs_dbg):.3f}" if math.isfinite(float(lat_abs_dbg)) else "nan",
                    )
                )
            if not bool(use_curb):
                parked_edge_reject_reasons[str(proof_reason)] = int(parked_edge_reject_reasons.get(str(proof_reason), 0)) + 1

            pose_rows: Dict[int, Tuple[float, float, float, int]] = {}
            prev_row: Optional[Tuple[float, float, float, int]] = None
            for fi, fr in enumerate(frames):
                qx, qy, qyaw = _frame_query_pose(fr)
                cc = _best_projection_on_carla_line(
                    carla_context=carla_context,
                    line_index=int(committed_line),
                    qx=float(qx),
                    qy=float(qy),
                    qyaw=float(qyaw),
                    prefer_reversed=None,
                    enforce_node_direction=False,
                    opposite_reject_deg=float(opposite_reject_deg),
                )
                if not isinstance(cc, dict):
                    if prev_row is not None:
                        pose_rows[int(fi)] = prev_row
                    continue
                pp = cc.get("projection", {})
                px = _safe_float(pp.get("x"), float("nan"))
                py = _safe_float(pp.get("y"), float("nan"))
                pyaw = _safe_float(pp.get("yaw"), float("nan"))
                if not (math.isfinite(px) and math.isfinite(py) and math.isfinite(pyaw)):
                    if prev_row is not None:
                        pose_rows[int(fi)] = prev_row
                    continue
                if bool(use_curb):
                    nyaw = math.radians(float(pyaw))
                    nx = -math.sin(float(nyaw))
                    ny = math.cos(float(nyaw))
                    px = float(px) + float(outward_sign) * float(parked_invariant_edge_offset_m) * float(nx)
                    py = float(py) + float(outward_sign) * float(parked_invariant_edge_offset_m) * float(ny)
                row = (float(px), float(py), float(_normalize_yaw_deg(pyaw)), int(committed_line))
                pose_rows[int(fi)] = row
                prev_row = row
            if not pose_rows:
                continue
            if len(pose_rows) < len(frames):
                carry: Optional[Tuple[float, float, float, int]] = None
                for fi in range(len(frames)):
                    row = pose_rows.get(int(fi))
                    if row is not None:
                        carry = row
                    elif carry is not None:
                        pose_rows[int(fi)] = carry
                carry = None
                for fi in range(len(frames) - 1, -1, -1):
                    row = pose_rows.get(int(fi))
                    if row is not None:
                        carry = row
                    elif carry is not None:
                        pose_rows[int(fi)] = carry

            motion_stats = active_motion.get(str(key), {})
            static_like = bool(
                float(motion_stats.get("robust_net_disp_m", motion_stats.get("net_disp_m", 0.0)))
                <= float(parked_invariant_static_net_max_m)
                and float(motion_stats.get("p95_step_m", 0.0)) <= float(parked_invariant_static_p95_step_max_m)
            )
            source = "parked_invariant_edge" if bool(use_curb) else "parked_invariant_centerline"
            if bool(static_like):
                arr_x = np.asarray([float(v[0]) for v in pose_rows.values()], dtype=np.float64)
                arr_y = np.asarray([float(v[1]) for v in pose_rows.values()], dtype=np.float64)
                arr_yaw = np.asarray([math.radians(float(v[2])) for v in pose_rows.values()], dtype=np.float64)
                ax = float(np.median(arr_x))
                ay = float(np.median(arr_y))
                ayaw = float(
                    _normalize_yaw_deg(
                        math.degrees(
                            math.atan2(
                                float(np.mean(np.sin(arr_yaw))),
                                float(np.mean(np.cos(arr_yaw))),
                            )
                        )
                    )
                )
                for fi, fr in enumerate(frames):
                    qx, qy, _ = _frame_query_pose(fr)
                    _set_carla_pose(
                        frame=fr,
                        line_index=int(committed_line),
                        x=float(ax),
                        y=float(ay),
                        yaw=float(ayaw),
                        dist=float(math.hypot(float(ax) - float(qx), float(ay) - float(qy))),
                        source=f"{source}_static",
                        quality="none",
                    )
            else:
                for fi, fr in enumerate(frames):
                    row = pose_rows.get(int(fi))
                    if row is None:
                        continue
                    px, py, pyaw, pli = row
                    qx, qy, _ = _frame_query_pose(fr)
                    _set_carla_pose(
                        frame=fr,
                        line_index=int(pli),
                        x=float(px),
                        y=float(py),
                        yaw=float(pyaw),
                        dist=float(math.hypot(float(px) - float(qx), float(py) - float(qy))),
                        source=f"{source}_track",
                        quality="none",
                    )

            parked_tracks_applied += 1
            parked_frames_applied += int(len(frames))
            if bool(use_curb):
                parked_edge_tracks += 1
            else:
                parked_centerline_tracks += 1

        parked_duplicate_pruned_ids: List[str] = []
        if bool(parked_duplicate_prune_enabled):
            parked_keys_local = [
                str(k)
                for k in sorted(parked_keys_final)
                if str(k) in active_frames_by_key and str(k) in active_ticks
            ]
            remove_dup_keys: set = set()

            def _dup_track_quality(track_key: str) -> Tuple[int, float, float, int]:
                tr_local = active_track_by_key.get(str(track_key), {})
                frames_local = active_frames_by_key.get(str(track_key), [])
                motion_local = active_motion.get(str(track_key), {})
                path_len_local = float(motion_local.get("path_len_m", 0.0))
                raw_arr = raw_err_by_key.get(str(track_key), np.zeros((0,), dtype=np.float64))
                raw_med_local = (
                    float(np.median(raw_arr))
                    if isinstance(raw_arr, np.ndarray) and raw_arr.size > 0
                    else 0.0
                )
                tid = tr_local.get("id", track_key) if isinstance(tr_local, dict) else track_key
                try:
                    tid_int = int(tid)
                except Exception:
                    tid_int = -1
                return (int(len(frames_local)), float(path_len_local), -float(raw_med_local), int(tid_int))

            for i in range(len(parked_keys_local)):
                a_key = str(parked_keys_local[i])
                if a_key in remove_dup_keys:
                    continue
                a_ticks = active_ticks.get(str(a_key), {})
                a_frames = active_frames_by_key.get(str(a_key), [])
                if not isinstance(a_ticks, dict) or not isinstance(a_frames, list) or not a_ticks:
                    continue
                for j in range(i + 1, len(parked_keys_local)):
                    b_key = str(parked_keys_local[j])
                    if b_key in remove_dup_keys or b_key == a_key:
                        continue
                    b_ticks = active_ticks.get(str(b_key), {})
                    b_frames = active_frames_by_key.get(str(b_key), [])
                    if not isinstance(b_ticks, dict) or not isinstance(b_frames, list) or not b_ticks:
                        continue
                    common_tks = sorted(set(a_ticks.keys()).intersection(set(b_ticks.keys())))
                    if len(common_tks) < int(parked_duplicate_prune_min_overlap_frames):
                        continue
                    l1, w1 = active_dims.get(str(a_key), (4.8, 2.0))
                    l2, w2 = active_dims.get(str(b_key), (4.8, 2.0))
                    overlap_hits = 0
                    same_line_hits = 0
                    pen_sum = 0.0
                    for tk in common_tks:
                        ai = int(a_ticks.get(int(tk), -1))
                        bi = int(b_ticks.get(int(tk), -1))
                        if ai < 0 or bi < 0 or ai >= len(a_frames) or bi >= len(b_frames):
                            continue
                        af = a_frames[int(ai)]
                        bf = b_frames[int(bi)]
                        if (not _frame_has_carla_pose(af)) or (not _frame_has_carla_pose(bf)):
                            continue
                        ax = float(_safe_float(af.get("cx"), _safe_float(af.get("x"), 0.0)))
                        ay = float(_safe_float(af.get("cy"), _safe_float(af.get("y"), 0.0)))
                        ayaw = float(_safe_float(af.get("cyaw"), _safe_float(af.get("yaw"), 0.0)))
                        bx = float(_safe_float(bf.get("cx"), _safe_float(bf.get("x"), 0.0)))
                        by = float(_safe_float(bf.get("cy"), _safe_float(bf.get("y"), 0.0)))
                        byaw = float(_safe_float(bf.get("cyaw"), _safe_float(bf.get("yaw"), 0.0)))
                        pen = float(
                            _obb_overlap_penetration_xyyaw(
                                float(ax),
                                float(ay),
                                float(ayaw),
                                float(l1),
                                float(w1),
                                float(bx),
                                float(by),
                                float(byaw),
                                float(l2),
                                float(w2),
                            )
                        )
                        if float(pen) < float(parked_duplicate_prune_min_pair_pen_m):
                            continue
                        overlap_hits += 1
                        pen_sum += float(pen)
                        if int(_safe_int(af.get("ccli"), -1)) == int(_safe_int(bf.get("ccli"), -2)):
                            same_line_hits += 1
                    if int(overlap_hits) < int(parked_duplicate_prune_min_overlap_frames):
                        continue
                    ratio_a = float(overlap_hits) / float(max(1, len(a_ticks)))
                    ratio_b = float(overlap_hits) / float(max(1, len(b_ticks)))
                    if (
                        float(ratio_a) < float(parked_duplicate_prune_min_overlap_ratio_each)
                        or float(ratio_b) < float(parked_duplicate_prune_min_overlap_ratio_each)
                    ):
                        continue
                    if float(pen_sum) < float(parked_duplicate_prune_min_pen_sum_m):
                        continue
                    same_line_ratio = float(same_line_hits) / float(max(1, overlap_hits))
                    if float(same_line_ratio) < float(parked_duplicate_prune_same_line_ratio_min):
                        continue

                    qa = _dup_track_quality(str(a_key))
                    qb = _dup_track_quality(str(b_key))
                    drop_key = str(b_key) if qa >= qb else str(a_key)
                    remove_dup_keys.add(str(drop_key))

            if remove_dup_keys:
                kept_tracks: List[Dict[str, object]] = []
                for tr in tracks:
                    if not isinstance(tr, dict):
                        kept_tracks.append(tr)
                        continue
                    tkey = _carla_track_key(tr)
                    if str(tkey) in remove_dup_keys:
                        parked_duplicate_pruned_ids.append(str(tr.get("id", tkey)))
                        continue
                    kept_tracks.append(tr)
                tracks[:] = kept_tracks
                for rk in list(remove_dup_keys):
                    track_by_key.pop(str(rk), None)
                    frames_by_key.pop(str(rk), None)
                    dims_by_key.pop(str(rk), None)
                    bbox_uncert_by_key.pop(str(rk), None)
                    tick_to_index.pop(str(rk), None)
                    speed_by_key.pop(str(rk), None)
                    raw_err_by_key.pop(str(rk), None)
                    motion_stats_by_key.pop(str(rk), None)
                    parked_keys.discard(str(rk))
                    quasi_parked_keys.discard(str(rk))
                    moving_keys.discard(str(rk))
                    stationary_remove_keys.discard(str(rk))
                    active_frames_by_key.pop(str(rk), None)
                    active_track_by_key.pop(str(rk), None)
                    active_ticks.pop(str(rk), None)
                    active_dims.pop(str(rk), None)
                    active_motion.pop(str(rk), None)
                    parked_keys_final.discard(str(rk))
                    moving_keys_final.discard(str(rk))

        report["parked_invariant_tracks"] = int(parked_tracks_applied)
        report["parked_invariant_frames"] = int(parked_frames_applied)
        report["parked_invariant_edge_tracks"] = int(parked_edge_tracks)
        report["parked_invariant_centerline_tracks"] = int(parked_centerline_tracks)
        report["parked_outer_edge_proven_tracks"] = int(parked_edge_tracks)
        report["parked_outer_edge_rejected_tracks"] = int(max(0, parked_tracks_applied - parked_edge_tracks))
        report["parked_outermost_tiebreak_tracks"] = int(parked_outermost_tiebreak_tracks)
        report["parked_overlap_conflict_edge_tracks"] = int(parked_overlap_conflict_edge_tracks)
        report["parked_inner_to_outer_centerline_tracks"] = int(parked_inner_to_outer_centerline_tracks)
        report["parked_duplicate_pruned_tracks"] = int(len(parked_duplicate_pruned_ids))
        report["parked_duplicate_pruned_ids"] = sorted(set(str(v) for v in parked_duplicate_pruned_ids))
        report["parked_outer_edge_reject_reasons"] = dict(
            sorted(parked_edge_reject_reasons.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
        )

    report["runs_considered"] = int(total_runs_considered)
    report["runs_adjusted"] = int(total_runs_adjusted)
    report["frames_adjusted"] = int(total_frames_adjusted)
    report["frame_raw_restored"] = int(frame_raw_restored)
    report["overlap_pen_before"] = float(overlap_pen_before)
    report["overlap_pen_after"] = float(overlap_pen_after)
    report["adjusted_track_ids"] = sorted(str(track_by_key[k].get("id", "")) for k in adjusted_tracks if k in track_by_key)
    if verbose and int(total_runs_adjusted) > 0:
        print(
            "[CARLA_OVERLAP] scenario={} parked={} moving={} adjusted_runs={} adjusted_frames={} "
            "pen_before={:.3f} pen_after={:.3f}".format(
                str(scenario_name),
                int(report["parked_tracks"]),
                int(report["moving_tracks"]),
                int(report["runs_adjusted"]),
                int(report["frames_adjusted"]),
                float(report["overlap_pen_before"]),
                float(report["overlap_pen_after"]),
            )
        )
    return report


# =============================================================================
# Trajectory Loading
# =============================================================================


def _copy_waypoint(wp: Waypoint) -> Waypoint:
    return Waypoint(
        x=float(wp.x),
        y=float(wp.y),
        z=float(wp.z),
        yaw=float(wp.yaw),
        pitch=float(getattr(wp, "pitch", 0.0)),
        roll=float(getattr(wp, "roll", 0.0)),
    )


def _trajectory_path_length_m(traj: Sequence[Waypoint]) -> float:
    if len(traj) < 2:
        return 0.0
    total = 0.0
    prev = traj[0]
    for wp in traj[1:]:
        total += math.hypot(float(wp.x) - float(prev.x), float(wp.y) - float(prev.y))
        prev = wp
    return float(total)


def _track_ticks_for_dedup(times: Sequence[float] | None, n: int, dt: float) -> List[int]:
    if times and len(times) == n:
        base_dt = max(1e-6, float(dt))
        out: List[int] = []
        for i, t in enumerate(times):
            tv = _safe_float(t, float(i) * base_dt)
            out.append(int(round(float(tv) / base_dt)))
        return out
    return list(range(int(n)))


def _infer_actor_role_for_dedup(obj_type: str | None, traj: Sequence[Waypoint]) -> str:
    if hasattr(ytm, "_actor_role_for_dedup"):
        try:
            role = str(ytm._actor_role_for_dedup(obj_type, traj))
            if role:
                return role
        except Exception:
            pass
    obj = str(obj_type or "").lower()
    if "walker" in obj or "pedestrian" in obj:
        return "walker"
    if "cyclist" in obj or "bicycle" in obj or "bike" in obj:
        return "cyclist"
    if "static" in obj:
        return "static"
    return "vehicle"


def _merge_actor_meta_local(acc: Dict[str, object], incoming: Dict[str, object]) -> Dict[str, object]:
    if hasattr(ytm, "_merge_actor_meta"):
        try:
            return dict(ytm._merge_actor_meta(acc, incoming))
        except Exception:
            pass
    out = dict(acc)
    if not out.get("obj_type") and incoming.get("obj_type"):
        out["obj_type"] = incoming.get("obj_type")
    if not out.get("model") and incoming.get("model"):
        out["model"] = incoming.get("model")
    if out.get("length") is None and incoming.get("length") is not None:
        out["length"] = incoming.get("length")
    if out.get("width") is None and incoming.get("width") is not None:
        out["width"] = incoming.get("width")
    return out


def _pair_overlap_metrics_shifted(
    traj_a: Sequence[Waypoint],
    times_a: Sequence[float] | None,
    traj_b: Sequence[Waypoint],
    times_b: Sequence[float] | None,
    dt: float,
    max_shift_ticks: int,
) -> Optional[Dict[str, float]]:
    if not traj_a or not traj_b:
        return None

    ticks_a = _track_ticks_for_dedup(times_a, len(traj_a), dt=float(dt))
    ticks_b = _track_ticks_for_dedup(times_b, len(traj_b), dt=float(dt))
    idx_a: Dict[int, int] = {}
    idx_b: Dict[int, int] = {}
    for i, tk in enumerate(ticks_a):
        if tk not in idx_a:
            idx_a[tk] = i
    for i, tk in enumerate(ticks_b):
        if tk not in idx_b:
            idx_b[tk] = i
    if not idx_a or not idx_b:
        return None

    rows: List[Dict[str, float]] = []
    for shift in range(-max(0, int(max_shift_ticks)), max(0, int(max_shift_ticks)) + 1):
        idx_b_shifted: Dict[int, int] = {int(tk + shift): int(i) for tk, i in idx_b.items()}
        common_ticks = sorted(set(idx_a.keys()).intersection(idx_b_shifted.keys()))
        if not common_ticks:
            continue

        dists: List[float] = []
        yaw_diffs: List[float] = []
        step_diffs: List[float] = []
        max_run = 0
        run = 0
        prev_tick: Optional[int] = None

        for tk in common_ticks:
            if prev_tick is None or int(tk) == int(prev_tick) + 1:
                run += 1
            else:
                run = 1
            prev_tick = int(tk)
            max_run = max(max_run, run)

            ia = idx_a[int(tk)]
            ib = idx_b_shifted[int(tk)]
            wa = traj_a[ia]
            wb = traj_b[ib]
            dists.append(math.hypot(float(wa.x) - float(wb.x), float(wa.y) - float(wb.y)))
            yaw_abs = abs(_normalize_yaw_deg(float(wa.yaw) - float(wb.yaw)))
            yaw_bidir = min(float(yaw_abs), abs(180.0 - float(yaw_abs)))
            yaw_diffs.append(float(yaw_bidir))

            if ia > 0 and ib > 0:
                wa_prev = traj_a[ia - 1]
                wb_prev = traj_b[ib - 1]
                step_a = math.hypot(float(wa.x) - float(wa_prev.x), float(wa.y) - float(wa_prev.y))
                step_b = math.hypot(float(wb.x) - float(wb_prev.x), float(wb.y) - float(wb_prev.y))
                step_diffs.append(abs(float(step_a) - float(step_b)))

        arr_d = np.asarray(dists, dtype=np.float64)
        arr_y = np.asarray(yaw_diffs, dtype=np.float64)
        arr_step = np.asarray(step_diffs, dtype=np.float64) if step_diffs else np.zeros((0,), dtype=np.float64)
        common_n = int(len(common_ticks))
        row = {
            "shift_ticks": float(shift),
            "common_points": float(common_n),
            "max_contiguous_common_points": float(max_run),
            "overlap_ratio_a": float(common_n) / max(1, len(idx_a)),
            "overlap_ratio_b": float(common_n) / max(1, len(idx_b)),
            "median_dist_m": float(np.median(arr_d)),
            "p90_dist_m": float(np.quantile(arr_d, 0.9)),
            "max_dist_m": float(np.max(arr_d)),
            "median_yaw_diff_deg": float(np.median(arr_y)),
            "p90_yaw_diff_deg": float(np.quantile(arr_y, 0.9)),
            "median_step_diff_m": float(np.median(arr_step)) if arr_step.size > 0 else float("inf"),
        }
        rows.append(row)

    if not rows:
        return None
    viable = list(rows)
    best = min(
        viable,
        key=lambda r: (
            (
                float(r.get("median_dist_m", float("inf")))
                + 0.25 * float(r.get("p90_dist_m", float("inf")))
                - 0.03 * min(25.0, float(r.get("common_points", 0.0)))
            ),
            float(r.get("median_dist_m", float("inf"))),
            -float(r.get("common_points", 0.0)),
            abs(float(r.get("shift_ticks", 0.0))),
        ),
    )
    return dict(best)


def _merge_duplicate_cluster_by_ticks(
    members: List[int],
    vehicles: Dict[int, List[Waypoint]],
    vehicle_times: Dict[int, List[float]],
    dt: float,
) -> Tuple[int, List[Waypoint], List[float], List[int]]:
    if not members:
        return -1, [], [], []

    def _quality_key(vid: int) -> Tuple[float, int, int]:
        traj = vehicles.get(int(vid), [])
        times = vehicle_times.get(int(vid), [])
        return (
            float(_trajectory_path_length_m(traj)),
            int(len(times) if times else len(traj)),
            int(len(traj)),
        )

    ordered = sorted([int(v) for v in members], key=lambda vid: (_quality_key(vid), -int(vid)), reverse=True)
    rep = int(ordered[0])

    # Union timestamps across duplicates. For overlapping samples at the same tick,
    # use a robust geometric median blend (instead of representative priority) to
    # reduce ID-handoff jitter from any single source.
    tick_samples: Dict[int, List[Waypoint]] = {}
    for vid in ordered:
        traj = vehicles.get(int(vid), [])
        if not traj:
            continue
        times = vehicle_times.get(int(vid), [])
        ticks = _track_ticks_for_dedup(times, len(traj), dt=float(dt))
        for i, tk in enumerate(ticks):
            if i < 0 or i >= len(traj):
                continue
            tick_samples.setdefault(int(tk), []).append(traj[i])

    sorted_ticks = sorted(tick_samples.keys())
    merged_traj: List[Waypoint] = []
    merged_times: List[float] = []
    for tk in sorted_ticks:
        samples = tick_samples.get(int(tk), [])
        if not samples:
            continue
        if len(samples) == 1:
            merged_wp = _copy_waypoint(samples[0])
        else:
            xs = np.asarray([float(wp.x) for wp in samples], dtype=np.float64)
            ys = np.asarray([float(wp.y) for wp in samples], dtype=np.float64)
            zs = np.asarray([float(getattr(wp, "z", 0.0)) for wp in samples], dtype=np.float64)
            pitches = np.asarray([float(getattr(wp, "pitch", 0.0)) for wp in samples], dtype=np.float64)
            rolls = np.asarray([float(getattr(wp, "roll", 0.0)) for wp in samples], dtype=np.float64)
            yaws_rad = np.asarray([math.radians(float(wp.yaw)) for wp in samples], dtype=np.float64)
            yaw_blend = math.degrees(math.atan2(float(np.mean(np.sin(yaws_rad))), float(np.mean(np.cos(yaws_rad)))))
            merged_wp = Waypoint(
                x=float(np.median(xs)),
                y=float(np.median(ys)),
                z=float(np.median(zs)),
                yaw=float(_normalize_yaw_deg(float(yaw_blend))),
                pitch=float(np.median(pitches)),
                roll=float(np.median(rolls)),
            )
        merged_traj.append(merged_wp)
        merged_times.append(float(tk) * float(dt))

    return int(rep), merged_traj, merged_times, ordered


def _apply_post_timing_handoff_dedup(
    vehicles: Dict[int, List[Waypoint]],
    vehicle_times: Dict[int, List[float]],
    obj_info: Dict[int, Dict[str, object]],
) -> Tuple[Dict[int, List[Waypoint]], Dict[int, List[float]], Dict[int, Dict[str, object]]]:
    """Remove predecessor actors in terminal handoff pairs found after timing optimization.

    The early-spawn timing optimization extends actor tracks backward in time, which can
    create new handoff pairs not visible at overlap-dedup time.  This pass runs on the
    final vehicles dict and removes the *predecessor* (the actor whose track ends at the
    handoff point) to eliminate the duplicate.
    """
    enabled = _env_int("V2X_POST_TIMING_HANDOFF_DEDUP_ENABLED", 1, minimum=0, maximum=1) == 1
    if not enabled or len(vehicles) < 2:
        return vehicles, vehicle_times, obj_info

    max_gap_s = _env_float("V2X_POST_TIMING_HANDOFF_MAX_GAP_S", 1.5)
    max_dist_m = _env_float("V2X_POST_TIMING_HANDOFF_MAX_DIST_M", 2.0)

    role_by_id: Dict[int, str] = {}
    for vid in vehicles.keys():
        meta = obj_info.get(int(vid), {})
        role_by_id[int(vid)] = _infer_actor_role_for_dedup(
            str(meta.get("obj_type") or ""), vehicles.get(int(vid), [])
        )

    to_remove: Set[int] = set()
    ids = sorted(int(v) for v in vehicles.keys())

    for i in range(len(ids)):
        va = int(ids[i])
        if va in to_remove:
            continue
        t_a = vehicle_times.get(va, [])
        tr_a = vehicles.get(va, [])
        if not t_a or not tr_a:
            continue
        # Normalize "static" (1-frame parked vehicles) to "vehicle" for handoff matching
        role_a = role_by_id.get(va, "vehicle")
        if role_a == "static":
            role_a = "vehicle"

        for j in range(i + 1, len(ids)):
            vb = int(ids[j])
            if vb in to_remove:
                continue
            t_b = vehicle_times.get(vb, [])
            tr_b = vehicles.get(vb, [])
            if not t_b or not tr_b:
                continue
            role_b = role_by_id.get(vb, "vehicle")
            if role_b == "static":
                role_b = "vehicle"
            if role_a != role_b:
                continue

            # Check a ends -> b starts (a is predecessor, remove a)
            gap_ab = abs(float(t_b[0]) - float(t_a[-1]))
            if gap_ab <= float(max_gap_s):
                dist_ab = math.hypot(
                    float(tr_a[-1].x) - float(tr_b[0].x),
                    float(tr_a[-1].y) - float(tr_b[0].y),
                )
                if dist_ab <= float(max_dist_m):
                    to_remove.add(va)
                    break

            # Check b ends -> a starts (b is predecessor, remove b)
            gap_ba = abs(float(t_a[0]) - float(t_b[-1]))
            if gap_ba <= float(max_gap_s):
                dist_ba = math.hypot(
                    float(tr_b[-1].x) - float(tr_a[0].x),
                    float(tr_b[-1].y) - float(tr_a[0].y),
                )
                if dist_ba <= float(max_dist_m):
                    to_remove.add(vb)

    if to_remove:
        print(
            "[INFO] Post-timing handoff dedup: removed {} predecessor actors: {}".format(
                len(to_remove), sorted(to_remove)
            )
        )
        vehicles = {k: v for k, v in vehicles.items() if int(k) not in to_remove}
        vehicle_times = {k: v for k, v in vehicle_times.items() if int(k) not in to_remove}
        obj_info = {k: v for k, v in obj_info.items() if int(k) not in to_remove}

    return vehicles, vehicle_times, obj_info


def _is_terminal_handoff(
    va: int,
    vb: int,
    vehicles: "Dict[int, List[Any]]",
    vehicle_times: "Dict[int, List[float]]",
    max_gap_s: float,
    max_dist_m: float,
) -> bool:
    """Return True if one actor's track ends where and when the other begins."""
    t_a = vehicle_times.get(int(va), [])
    t_b = vehicle_times.get(int(vb), [])
    tr_a = vehicles.get(int(va), [])
    tr_b = vehicles.get(int(vb), [])
    if not t_a or not t_b or not tr_a or not tr_b:
        return False
    for tend, pend, tstart, pstart in (
        (float(t_a[-1]), tr_a[-1], float(t_b[0]), tr_b[0]),
        (float(t_b[-1]), tr_b[-1], float(t_a[0]), tr_a[0]),
    ):
        if abs(float(tstart) - float(tend)) <= float(max_gap_s):
            if (
                math.hypot(
                    float(pend.x) - float(pstart.x),
                    float(pend.y) - float(pstart.y),
                )
                <= float(max_dist_m)
            ):
                return True
    return False


def _deduplicate_actor_tracks_by_overlap(
    vehicles: Dict[int, List[Waypoint]],
    vehicle_times: Dict[int, List[float]],
    obj_info: Dict[int, Dict[str, object]],
    actor_source_subdir: Dict[int, str],
    actor_orig_vid: Dict[int, int],
    dt: float,
) -> Tuple[
    Dict[int, List[Waypoint]],
    Dict[int, List[float]],
    Dict[int, Dict[str, object]],
    Dict[int, str],
    Dict[int, int],
    Dict[str, object],
]:
    enabled = _env_int("V2X_ACTOR_OVERLAP_DEDUP_ENABLED", 1, minimum=0, maximum=1) == 1
    if not enabled or len(vehicles) < 2:
        return vehicles, vehicle_times, obj_info, actor_source_subdir, actor_orig_vid, {
            "enabled": bool(enabled),
            "pair_checks": 0,
            "candidate_pairs": 0,
            "clusters": 0,
            "removed_ids": [],
        }

    max_shift_ticks = _env_int("V2X_ACTOR_OVERLAP_DEDUP_MAX_SHIFT_TICKS", 1, minimum=0, maximum=3)
    min_common_points = _env_int("V2X_ACTOR_OVERLAP_DEDUP_MIN_COMMON_POINTS", 8, minimum=2, maximum=300)
    min_overlap_ratio_each = _env_float("V2X_ACTOR_OVERLAP_DEDUP_MIN_OVERLAP_EACH", 0.30)
    min_overlap_ratio_any = _env_float("V2X_ACTOR_OVERLAP_DEDUP_MIN_OVERLAP_ANY", 0.75)
    max_median_dist_m = _env_float("V2X_ACTOR_OVERLAP_DEDUP_MAX_MEDIAN_DIST_M", 1.20)
    max_p90_dist_m = _env_float("V2X_ACTOR_OVERLAP_DEDUP_MAX_P90_DIST_M", 2.00)
    max_median_yaw_diff_deg = _env_float("V2X_ACTOR_OVERLAP_DEDUP_MAX_MEDIAN_YAW_DEG", 35.0)

    # Extra mode for short-overlap handoff duplicates across different source subdirs.
    short_min_common_points = _env_int("V2X_ACTOR_SHORT_OVERLAP_MIN_COMMON_POINTS", 5, minimum=2, maximum=80)
    short_min_overlap_any = _env_float("V2X_ACTOR_SHORT_OVERLAP_MIN_OVERLAP_ANY", 0.18)
    short_max_median_dist_m = _env_float("V2X_ACTOR_SHORT_OVERLAP_MAX_MEDIAN_DIST_M", 0.85)
    short_max_p90_dist_m = _env_float("V2X_ACTOR_SHORT_OVERLAP_MAX_P90_DIST_M", 1.35)
    short_max_dist_m = _env_float("V2X_ACTOR_SHORT_OVERLAP_MAX_DIST_M", 2.20)
    short_max_median_yaw_deg = _env_float("V2X_ACTOR_SHORT_OVERLAP_MAX_MEDIAN_YAW_DEG", 22.0)
    short_max_step_diff_m = _env_float("V2X_ACTOR_SHORT_OVERLAP_MAX_MEDIAN_STEP_DIFF_M", 0.60)
    short_require_diff_source = _env_int("V2X_ACTOR_SHORT_OVERLAP_REQUIRE_DIFF_SOURCE", 0, minimum=0, maximum=1) == 1
    # Segment-level duplicate mode: allow partial-run merges when a long
    # contiguous raw segment is nearly identical (typical ID handoff duplicates).
    segment_min_common_points = _env_int("V2X_ACTOR_SEGMENT_DEDUP_MIN_COMMON_POINTS", 18, minimum=4, maximum=120)
    segment_min_contiguous_points = _env_int("V2X_ACTOR_SEGMENT_DEDUP_MIN_CONTIGUOUS_POINTS", 16, minimum=4, maximum=120)
    segment_min_overlap_any = _env_float("V2X_ACTOR_SEGMENT_DEDUP_MIN_OVERLAP_ANY", 0.45)
    segment_max_median_dist_m = _env_float("V2X_ACTOR_SEGMENT_DEDUP_MAX_MEDIAN_DIST_M", 1.70)
    segment_max_p90_dist_m = _env_float("V2X_ACTOR_SEGMENT_DEDUP_MAX_P90_DIST_M", 2.90)
    segment_max_dist_m = _env_float("V2X_ACTOR_SEGMENT_DEDUP_MAX_DIST_M", 4.80)
    segment_max_median_yaw_deg = _env_float("V2X_ACTOR_SEGMENT_DEDUP_MAX_MEDIAN_YAW_DEG", 14.0)
    segment_max_step_diff_m = _env_float("V2X_ACTOR_SEGMENT_DEDUP_MAX_MEDIAN_STEP_DIFF_M", 0.90)
    segment_min_path_len_m = _env_float("V2X_ACTOR_SEGMENT_DEDUP_MIN_PATH_LEN_M", 25.0)
    segment_require_diff_source = _env_int("V2X_ACTOR_SEGMENT_DEDUP_REQUIRE_DIFF_SOURCE", 0, minimum=0, maximum=1) == 1
    skip_low_motion_pairs = _env_int("V2X_ACTOR_OVERLAP_DEDUP_SKIP_LOW_MOTION", 1, minimum=0, maximum=1) == 1
    skip_low_motion_path_max_m = _env_float("V2X_ACTOR_OVERLAP_DEDUP_SKIP_LOW_MOTION_PATH_MAX_M", 18.0)
    skip_low_motion_net_max_m = _env_float("V2X_ACTOR_OVERLAP_DEDUP_SKIP_LOW_MOTION_NET_MAX_M", 4.0)
    skip_low_motion_avg_speed_max_mps = _env_float("V2X_ACTOR_OVERLAP_DEDUP_SKIP_LOW_MOTION_AVG_SPEED_MAX_MPS", 2.2)
    low_motion_same_orig_strict_enabled = (
        _env_int("V2X_ACTOR_OVERLAP_DEDUP_LOW_MOTION_SAME_ORIG_STRICT_ENABLED", 1, minimum=0, maximum=1) == 1
    )
    low_motion_same_orig_min_common_points = _env_int(
        "V2X_ACTOR_OVERLAP_DEDUP_LOW_MOTION_SAME_ORIG_MIN_COMMON_POINTS",
        8,
        minimum=3,
        maximum=120,
    )
    low_motion_same_orig_min_contiguous_points = _env_int(
        "V2X_ACTOR_OVERLAP_DEDUP_LOW_MOTION_SAME_ORIG_MIN_CONTIGUOUS_POINTS",
        6,
        minimum=3,
        maximum=120,
    )
    low_motion_same_orig_min_overlap_any = _env_float(
        "V2X_ACTOR_OVERLAP_DEDUP_LOW_MOTION_SAME_ORIG_MIN_OVERLAP_ANY",
        0.50,
    )
    low_motion_same_orig_max_median_dist_m = _env_float(
        "V2X_ACTOR_OVERLAP_DEDUP_LOW_MOTION_SAME_ORIG_MAX_MEDIAN_DIST_M",
        1.0,
    )
    low_motion_same_orig_max_p90_dist_m = _env_float(
        "V2X_ACTOR_OVERLAP_DEDUP_LOW_MOTION_SAME_ORIG_MAX_P90_DIST_M",
        1.8,
    )
    low_motion_same_orig_max_dist_m = _env_float(
        "V2X_ACTOR_OVERLAP_DEDUP_LOW_MOTION_SAME_ORIG_MAX_DIST_M",
        3.2,
    )
    low_motion_same_orig_max_median_yaw_deg = _env_float(
        "V2X_ACTOR_OVERLAP_DEDUP_LOW_MOTION_SAME_ORIG_MAX_MEDIAN_YAW_DEG",
        16.0,
    )
    handoff_dedup_enabled = (
        _env_int("V2X_ACTOR_HANDOFF_DEDUP_ENABLED", 1, minimum=0, maximum=1) == 1
    )
    handoff_max_gap_s = _env_float("V2X_ACTOR_HANDOFF_DEDUP_MAX_GAP_S", 1.5)
    handoff_max_dist_m = _env_float("V2X_ACTOR_HANDOFF_DEDUP_MAX_DIST_M", 2.0)

    vehicles = {int(k): list(v) for k, v in vehicles.items()}
    vehicle_times = {int(k): [float(t) for t in v] for k, v in vehicle_times.items()}
    obj_info = {int(k): dict(v) for k, v in obj_info.items()}
    actor_source_subdir = {int(k): str(v) for k, v in actor_source_subdir.items()}
    actor_orig_vid = {int(k): int(v) for k, v in actor_orig_vid.items()}

    ids = sorted(int(v) for v in vehicles.keys())
    id_to_pos = {int(v): i for i, v in enumerate(ids)}
    parents = list(range(len(ids)))

    def _find(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def _union(i: int, j: int) -> None:
        ri = _find(i)
        rj = _find(j)
        if ri == rj:
            return
        if ri < rj:
            parents[rj] = ri
        else:
            parents[ri] = rj

    role_by_id: Dict[int, str] = {}
    path_len_by_id: Dict[int, float] = {}
    low_motion_by_id: Dict[int, bool] = {}
    stationary_like_by_id: Dict[int, bool] = {}
    for vid in ids:
        meta = obj_info.get(int(vid), {})
        role_by_id[int(vid)] = _infer_actor_role_for_dedup(str(meta.get("obj_type") or ""), vehicles.get(int(vid), []))
        traj = vehicles.get(int(vid), [])
        path_len = float(_trajectory_path_length_m(traj))
        path_len_by_id[int(vid)] = float(path_len)
        low_motion = bool(meta.get("low_motion_vehicle", False))
        low_motion_by_id[int(vid)] = bool(low_motion)
        net_disp = 0.0
        if len(traj) >= 2:
            net_disp = float(
                math.hypot(
                    float(traj[-1].x) - float(traj[0].x),
                    float(traj[-1].y) - float(traj[0].y),
                )
            )
        times = vehicle_times.get(int(vid), [])
        if len(times) >= 2:
            duration = max(1e-3, float(times[-1]) - float(times[0]))
        else:
            duration = max(1e-3, float(max(0, len(traj) - 1)) * float(dt))
        avg_speed = float(path_len) / float(duration)
        stationary_like_by_id[int(vid)] = bool(
            low_motion
            or (
                float(path_len) <= float(skip_low_motion_path_max_m)
                and float(net_disp) <= float(skip_low_motion_net_max_m)
                and float(avg_speed) <= float(skip_low_motion_avg_speed_max_mps)
            )
        )

    pair_checks = 0
    candidate_pairs = 0
    short_pairs = 0
    segment_pairs = 0
    low_motion_same_orig_pairs = 0
    handoff_pairs = 0
    for i in range(len(ids)):
        va = int(ids[i])
        role_a = role_by_id.get(va, "vehicle")
        for j in range(i + 1, len(ids)):
            vb = int(ids[j])
            role_b = role_by_id.get(vb, "vehicle")
            if role_a != role_b:
                continue
            # Terminal handoff: one actor ends where/when the other begins.
            # Check before the stationary-like skip so short-duration actors aren't excluded.
            if bool(handoff_dedup_enabled) and _is_terminal_handoff(
                int(va),
                int(vb),
                vehicles,
                vehicle_times,
                float(handoff_max_gap_s),
                float(handoff_max_dist_m),
            ):
                candidate_pairs += 1
                handoff_pairs += 1
                _union(id_to_pos[int(va)], id_to_pos[int(vb)])
                continue
            same_orig_id = (
                int(actor_orig_vid.get(int(va), int(va)))
                == int(actor_orig_vid.get(int(vb), int(vb)))
            )
            allow_low_motion_same_orig = bool(low_motion_same_orig_strict_enabled and bool(same_orig_id))
            if bool(skip_low_motion_pairs):
                if (
                    bool(stationary_like_by_id.get(int(va), False))
                    or bool(stationary_like_by_id.get(int(vb), False))
                ) and (not bool(allow_low_motion_same_orig)):
                    continue
            pair_checks += 1
            metrics = _pair_overlap_metrics_shifted(
                traj_a=vehicles.get(va, []),
                times_a=vehicle_times.get(va, []),
                traj_b=vehicles.get(vb, []),
                times_b=vehicle_times.get(vb, []),
                dt=float(dt),
                max_shift_ticks=int(max_shift_ticks),
            )
            if metrics is None:
                continue

            common = int(round(float(metrics.get("common_points", 0.0))))
            overlap_a = float(metrics.get("overlap_ratio_a", 0.0))
            overlap_b = float(metrics.get("overlap_ratio_b", 0.0))
            median_dist = float(metrics.get("median_dist_m", float("inf")))
            p90_dist = float(metrics.get("p90_dist_m", float("inf")))
            max_dist = float(metrics.get("max_dist_m", float("inf")))
            median_yaw = float(metrics.get("median_yaw_diff_deg", float("inf")))
            median_step_diff = float(metrics.get("median_step_diff_m", float("inf")))
            contiguous_common = int(round(float(metrics.get("max_contiguous_common_points", 0.0))))

            strong_match = (
                common >= int(min_common_points)
                and min(overlap_a, overlap_b) >= float(min_overlap_ratio_each)
                and max(overlap_a, overlap_b) >= float(min_overlap_ratio_any)
                and median_dist <= float(max_median_dist_m)
                and p90_dist <= float(max_p90_dist_m)
                and (role_a == "walker" or median_yaw <= float(max_median_yaw_diff_deg))
            )

            short_match = (
                common >= int(short_min_common_points)
                and (role_a == "walker" or contiguous_common >= int(short_min_common_points))
                and max(overlap_a, overlap_b) >= float(short_min_overlap_any)
                and median_dist <= float(short_max_median_dist_m)
                and p90_dist <= float(short_max_p90_dist_m)
                and max_dist <= float(short_max_dist_m)
                and (role_a == "walker" or median_yaw <= float(short_max_median_yaw_deg))
                and (role_a == "walker" or median_step_diff <= float(short_max_step_diff_m))
            )

            if short_match and not strong_match and short_require_diff_source:
                src_a = str(actor_source_subdir.get(int(va), ""))
                src_b = str(actor_source_subdir.get(int(vb), ""))
                if src_a and src_b and src_a == src_b:
                    short_match = False

            segment_match = (
                common >= int(segment_min_common_points)
                and (role_a == "walker" or contiguous_common >= int(segment_min_contiguous_points))
                and (not bool(low_motion_by_id.get(int(va), False)))
                and (not bool(low_motion_by_id.get(int(vb), False)))
                and min(
                    float(path_len_by_id.get(int(va), 0.0)),
                    float(path_len_by_id.get(int(vb), 0.0)),
                ) >= float(segment_min_path_len_m)
                and max(overlap_a, overlap_b) >= float(segment_min_overlap_any)
                and median_dist <= float(segment_max_median_dist_m)
                and p90_dist <= float(segment_max_p90_dist_m)
                and max_dist <= float(segment_max_dist_m)
                and (role_a == "walker" or median_yaw <= float(segment_max_median_yaw_deg))
                and (role_a == "walker" or median_step_diff <= float(segment_max_step_diff_m))
            )
            if segment_match and not strong_match and not short_match and segment_require_diff_source:
                src_a = str(actor_source_subdir.get(int(va), ""))
                src_b = str(actor_source_subdir.get(int(vb), ""))
                if src_a and src_b and src_a == src_b:
                    segment_match = False

            low_motion_same_orig_match = (
                bool(allow_low_motion_same_orig)
                and common >= int(low_motion_same_orig_min_common_points)
                and contiguous_common >= int(low_motion_same_orig_min_contiguous_points)
                and max(overlap_a, overlap_b) >= float(low_motion_same_orig_min_overlap_any)
                and median_dist <= float(low_motion_same_orig_max_median_dist_m)
                and p90_dist <= float(low_motion_same_orig_max_p90_dist_m)
                and max_dist <= float(low_motion_same_orig_max_dist_m)
                and (role_a == "walker" or median_yaw <= float(low_motion_same_orig_max_median_yaw_deg))
            )

            if not (strong_match or short_match or segment_match or low_motion_same_orig_match):
                continue

            candidate_pairs += 1
            if short_match and not strong_match:
                short_pairs += 1
            if segment_match and not strong_match and not short_match:
                segment_pairs += 1
            if low_motion_same_orig_match and not strong_match and not short_match and not segment_match:
                low_motion_same_orig_pairs += 1
            _union(id_to_pos[int(va)], id_to_pos[int(vb)])

    clusters: Dict[int, List[int]] = {}
    for vid in ids:
        root = _find(id_to_pos[int(vid)])
        clusters.setdefault(int(root), []).append(int(vid))

    merged_clusters = 0
    removed_ids: List[int] = []
    for members in sorted(clusters.values(), key=lambda arr: (len(arr), arr), reverse=True):
        if len(members) <= 1:
            continue
        rep, merged_traj, merged_times, ordered = _merge_duplicate_cluster_by_ticks(
            members=[int(v) for v in members],
            vehicles=vehicles,
            vehicle_times=vehicle_times,
            dt=float(dt),
        )
        if rep < 0 or len(merged_traj) <= 0:
            continue

        merged_clusters += 1
        merged_meta: Dict[str, object] = {}
        source_unique: List[str] = []
        merged_vids: List[int] = []
        rep_source = str(actor_source_subdir.get(int(rep), ""))
        for vid in ordered:
            v = int(vid)
            merged_meta = _merge_actor_meta_local(merged_meta, obj_info.get(v, {}))
            merged_vids.append(v)
            src = str(actor_source_subdir.get(v, ""))
            if src:
                source_unique.append(src)
        source_unique = sorted(set(source_unique))
        if source_unique:
            merged_meta["_merged_sources"] = source_unique
        merged_meta["_merged_vids"] = sorted(set(int(v) for v in merged_vids))

        vehicles[int(rep)] = merged_traj
        vehicle_times[int(rep)] = merged_times
        obj_info[int(rep)] = merged_meta
        if not rep_source and source_unique:
            actor_source_subdir[int(rep)] = str(source_unique[0])
        elif rep_source:
            actor_source_subdir[int(rep)] = rep_source

        for vid in ordered:
            v = int(vid)
            if v == int(rep):
                continue
            removed_ids.append(int(v))
            vehicles.pop(int(v), None)
            vehicle_times.pop(int(v), None)
            obj_info.pop(int(v), None)
            actor_source_subdir.pop(int(v), None)
            actor_orig_vid.pop(int(v), None)

    stats = {
        "enabled": True,
        "pair_checks": int(pair_checks),
        "candidate_pairs": int(candidate_pairs),
        "short_overlap_pairs": int(short_pairs),
        "segment_overlap_pairs": int(segment_pairs),
        "low_motion_same_orig_pairs": int(low_motion_same_orig_pairs),
        "handoff_pairs": int(handoff_pairs),
        "clusters": int(merged_clusters),
        "removed_ids": [int(v) for v in sorted(set(removed_ids))],
    }
    return vehicles, vehicle_times, obj_info, actor_source_subdir, actor_orig_vid, stats


def _deduplicate_ego_actor_overlap(
    vehicles: Dict[int, List[Waypoint]],
    vehicle_times: Dict[int, List[float]],
    ego_trajs: Sequence[Sequence[Waypoint]],
    ego_times: Sequence[Sequence[float]],
    obj_info: Dict[int, Dict[str, object]],
    dt: float,
) -> Tuple[
    Dict[int, List[Waypoint]],
    Dict[int, List[float]],
    Dict[int, Dict[str, object]],
    Dict[str, object],
]:
    enabled = _env_int("V2X_EGO_ACTOR_DEDUP_ENABLED", 1, minimum=0, maximum=1) == 1
    if not enabled or not vehicles or not ego_trajs:
        return vehicles, vehicle_times, obj_info, {
            "enabled": bool(enabled),
            "pair_checks": 0,
            "removed_ids": [],
        }

    max_shift_ticks = _env_int("V2X_EGO_ACTOR_DEDUP_MAX_SHIFT_TICKS", 1, minimum=0, maximum=2)
    min_common_points = _env_int("V2X_EGO_ACTOR_DEDUP_MIN_COMMON_POINTS", 8, minimum=3, maximum=200)
    min_overlap_ratio_actor = _env_float("V2X_EGO_ACTOR_DEDUP_MIN_OVERLAP_ACTOR", 0.55)
    min_overlap_ratio_any = _env_float("V2X_EGO_ACTOR_DEDUP_MIN_OVERLAP_ANY", 0.75)
    max_median_dist_m = _env_float("V2X_EGO_ACTOR_DEDUP_MAX_MEDIAN_DIST_M", 1.20)
    max_p90_dist_m = _env_float("V2X_EGO_ACTOR_DEDUP_MAX_P90_DIST_M", 2.10)
    max_max_dist_m = _env_float("V2X_EGO_ACTOR_DEDUP_MAX_DIST_M", 3.20)
    max_median_yaw_diff_deg = _env_float("V2X_EGO_ACTOR_DEDUP_MAX_MEDIAN_YAW_DEG", 22.0)

    subset_min_common = _env_int("V2X_EGO_ACTOR_DEDUP_SUBSET_MIN_COMMON", 6, minimum=3, maximum=200)
    subset_max_median_dist = _env_float("V2X_EGO_ACTOR_DEDUP_SUBSET_MAX_MEDIAN_DIST_M", 0.70)
    subset_max_p90_dist = _env_float("V2X_EGO_ACTOR_DEDUP_SUBSET_MAX_P90_DIST_M", 1.35)
    subset_min_actor_overlap = _env_float("V2X_EGO_ACTOR_DEDUP_SUBSET_MIN_ACTOR_OVERLAP", 0.25)
    skip_low_motion = _env_int("V2X_EGO_ACTOR_DEDUP_SKIP_LOW_MOTION", 1, minimum=0, maximum=1) == 1
    skip_stationary_path_max_m = _env_float("V2X_EGO_ACTOR_DEDUP_SKIP_STATIONARY_PATH_MAX_M", 18.0)
    skip_stationary_net_max_m = _env_float("V2X_EGO_ACTOR_DEDUP_SKIP_STATIONARY_NET_MAX_M", 4.0)
    skip_stationary_avg_speed_max_mps = _env_float("V2X_EGO_ACTOR_DEDUP_SKIP_STATIONARY_AVG_SPEED_MAX_MPS", 2.2)

    vehicles = {int(k): list(v) for k, v in vehicles.items()}
    vehicle_times = {int(k): [float(t) for t in v] for k, v in vehicle_times.items()}
    obj_info = {int(k): dict(v) for k, v in obj_info.items()}

    removed_ids: List[int] = []
    pair_checks = 0
    for vid in sorted(list(vehicles.keys())):
        traj = vehicles.get(int(vid), [])
        if not traj:
            continue
        if bool(skip_low_motion):
            meta = obj_info.get(int(vid), {})
            if bool(meta.get("low_motion_vehicle", False)):
                continue
            if len(traj) >= 2:
                path_len = float(_trajectory_path_length_m(traj))
                net_disp = float(
                    math.hypot(
                        float(traj[-1].x) - float(traj[0].x),
                        float(traj[-1].y) - float(traj[0].y),
                    )
                )
                times = vehicle_times.get(int(vid), [])
                if len(times) >= 2:
                    duration = max(1e-3, float(times[-1]) - float(times[0]))
                else:
                    duration = max(1e-3, float(max(0, len(traj) - 1)) * float(dt))
                avg_speed = float(path_len) / float(duration)
                if (
                    float(path_len) <= float(skip_stationary_path_max_m)
                    and float(net_disp) <= float(skip_stationary_net_max_m)
                    and float(avg_speed) <= float(skip_stationary_avg_speed_max_mps)
                ):
                    continue
        role = _infer_actor_role_for_dedup(str(obj_info.get(int(vid), {}).get("obj_type") or ""), traj)
        if role in ("walker", "static"):
            continue

        times = vehicle_times.get(int(vid), [])
        matched = False
        for ego_idx, ego_traj in enumerate(ego_trajs):
            if not ego_traj:
                continue
            ego_t = ego_times[ego_idx] if ego_idx < len(ego_times) else []
            pair_checks += 1
            metrics = _pair_overlap_metrics_shifted(
                traj_a=traj,
                times_a=times,
                traj_b=ego_traj,
                times_b=ego_t,
                dt=float(dt),
                max_shift_ticks=int(max_shift_ticks),
            )
            if metrics is None:
                continue

            common = int(round(float(metrics.get("common_points", 0.0))))
            overlap_actor = float(metrics.get("overlap_ratio_a", 0.0))
            overlap_ego = float(metrics.get("overlap_ratio_b", 0.0))
            median_dist = float(metrics.get("median_dist_m", float("inf")))
            p90_dist = float(metrics.get("p90_dist_m", float("inf")))
            max_dist = float(metrics.get("max_dist_m", float("inf")))
            median_yaw = float(metrics.get("median_yaw_diff_deg", float("inf")))

            strong_full = (
                common >= int(min_common_points)
                and overlap_actor >= float(min_overlap_ratio_actor)
                and max(overlap_actor, overlap_ego) >= float(min_overlap_ratio_any)
                and median_dist <= float(max_median_dist_m)
                and p90_dist <= float(max_p90_dist_m)
                and max_dist <= float(max_max_dist_m)
                and median_yaw <= float(max_median_yaw_diff_deg)
            )
            strong_subset = (
                common >= int(subset_min_common)
                and overlap_actor >= float(subset_min_actor_overlap)
                and median_dist <= float(subset_max_median_dist)
                and p90_dist <= float(subset_max_p90_dist)
            )
            if strong_full or strong_subset:
                matched = True
                break

        if matched:
            removed_ids.append(int(vid))
            vehicles.pop(int(vid), None)
            vehicle_times.pop(int(vid), None)
            obj_info.pop(int(vid), None)

    stats = {
        "enabled": True,
        "pair_checks": int(pair_checks),
        "removed_ids": [int(v) for v in sorted(set(removed_ids))],
    }
    return vehicles, vehicle_times, obj_info, stats


def _apply_overlap_dedup_pipeline(
    vehicles: Dict[int, List[Waypoint]],
    vehicle_times: Dict[int, List[float]],
    ego_trajs: Sequence[Sequence[Waypoint]],
    ego_times: Sequence[Sequence[float]],
    obj_info: Dict[int, Dict[str, object]],
    actor_source_subdir: Optional[Dict[int, str]],
    actor_orig_vid: Optional[Dict[int, int]],
    dt: float,
) -> Tuple[
    Dict[int, List[Waypoint]],
    Dict[int, List[float]],
    Dict[int, Dict[str, object]],
]:
    if not vehicles:
        return vehicles, vehicle_times, obj_info

    actor_source_subdir = {int(k): str(v) for k, v in (actor_source_subdir or {}).items()}
    actor_orig_vid = {int(k): int(v) for k, v in (actor_orig_vid or {}).items()}
    for vid in vehicles.keys():
        actor_source_subdir.setdefault(int(vid), "")
        actor_orig_vid.setdefault(int(vid), int(vid))

    # Pass 1: robust global cross-ID dedup from yaml_to_map (full overlap focused).
    use_cross_id = _env_int("V2X_ENABLE_CROSS_ID_DEDUP", 1, minimum=0, maximum=1) == 1
    if use_cross_id and hasattr(ytm, "_deduplicate_cross_id_tracks"):
        cross_skip_low_motion = _env_int("V2X_CROSS_ID_DEDUP_SKIP_LOW_MOTION", 1, minimum=0, maximum=1) == 1
        hold_vehicles: Dict[int, List[Waypoint]] = {}
        hold_vehicle_times: Dict[int, List[float]] = {}
        hold_obj_info: Dict[int, Dict[str, object]] = {}
        hold_actor_source: Dict[int, str] = {}
        hold_actor_orig: Dict[int, int] = {}
        cross_vehicles = {int(k): list(v) for k, v in vehicles.items()}
        cross_vehicle_times = {int(k): [float(t) for t in v] for k, v in vehicle_times.items()}
        cross_obj_info = {int(k): dict(v) for k, v in obj_info.items()}
        cross_actor_source = {int(k): str(v) for k, v in actor_source_subdir.items()}
        cross_actor_orig = {int(k): int(v) for k, v in actor_orig_vid.items()}
        if bool(cross_skip_low_motion):
            cross_skip_path_max_m = _env_float("V2X_CROSS_ID_DEDUP_SKIP_LOW_MOTION_PATH_MAX_M", 18.0)
            cross_skip_net_max_m = _env_float("V2X_CROSS_ID_DEDUP_SKIP_LOW_MOTION_NET_MAX_M", 4.0)
            cross_skip_avg_speed_max_mps = _env_float("V2X_CROSS_ID_DEDUP_SKIP_LOW_MOTION_AVG_SPEED_MAX_MPS", 2.2)
            for vid in sorted(list(cross_vehicles.keys())):
                meta = cross_obj_info.get(int(vid), {})
                skip_vid = bool(meta.get("low_motion_vehicle", False))
                if not bool(skip_vid):
                    traj = cross_vehicles.get(int(vid), [])
                    if len(traj) >= 2:
                        path_len = float(_trajectory_path_length_m(traj))
                        net_disp = float(
                            math.hypot(
                                float(traj[-1].x) - float(traj[0].x),
                                float(traj[-1].y) - float(traj[0].y),
                            )
                        )
                        times = cross_vehicle_times.get(int(vid), [])
                        if len(times) >= 2:
                            duration = max(1e-3, float(times[-1]) - float(times[0]))
                        else:
                            duration = max(1e-3, float(max(0, len(traj) - 1)) * float(dt))
                        avg_speed = float(path_len) / float(duration)
                        skip_vid = bool(
                            float(path_len) <= float(cross_skip_path_max_m)
                            and float(net_disp) <= float(cross_skip_net_max_m)
                            and float(avg_speed) <= float(cross_skip_avg_speed_max_mps)
                        )
                if not bool(skip_vid):
                    continue
                hold_vehicles[int(vid)] = cross_vehicles.pop(int(vid), [])
                hold_vehicle_times[int(vid)] = cross_vehicle_times.pop(int(vid), [])
                hold_obj_info[int(vid)] = cross_obj_info.pop(int(vid), {})
                hold_actor_source[int(vid)] = cross_actor_source.pop(int(vid), "")
                hold_actor_orig[int(vid)] = int(cross_actor_orig.pop(int(vid), int(vid)))
        try:
            if len(cross_vehicles) >= 2:
                (
                    cross_vehicles,
                    cross_vehicle_times,
                    cross_obj_info,
                    cross_actor_source,
                    cross_actor_orig,
                    _actor_alias_vids,
                    cross_stats,
                ) = ytm._deduplicate_cross_id_tracks(
                    vehicles=cross_vehicles,
                    vehicle_times=cross_vehicle_times,
                    obj_info=cross_obj_info,
                    actor_source_subdir=cross_actor_source,
                    actor_orig_vid=cross_actor_orig,
                    dt=float(dt),
                    max_median_dist_m=float(_env_float("V2X_CROSS_ID_DEDUP_MAX_MEDIAN_DIST_M", 1.2)),
                    max_p90_dist_m=float(_env_float("V2X_CROSS_ID_DEDUP_MAX_P90_DIST_M", 2.0)),
                    max_median_yaw_diff_deg=float(_env_float("V2X_CROSS_ID_DEDUP_MAX_MEDIAN_YAW_DEG", 35.0)),
                    min_common_points=int(_env_int("V2X_CROSS_ID_DEDUP_MIN_COMMON_POINTS", 8, minimum=2, maximum=500)),
                    min_overlap_ratio_each=float(_env_float("V2X_CROSS_ID_DEDUP_MIN_OVERLAP_EACH", 0.30)),
                    min_overlap_ratio_any=float(_env_float("V2X_CROSS_ID_DEDUP_MIN_OVERLAP_ANY", 0.75)),
                )
            else:
                cross_stats = {
                    "cross_id_removed": 0,
                    "cross_id_clusters": 0,
                    "cross_id_pair_checks": 0,
                }
            vehicles = {int(k): list(v) for k, v in cross_vehicles.items()}
            vehicle_times = {int(k): [float(t) for t in v] for k, v in cross_vehicle_times.items()}
            obj_info = {int(k): dict(v) for k, v in cross_obj_info.items()}
            actor_source_subdir = {int(k): str(v) for k, v in cross_actor_source.items()}
            actor_orig_vid = {int(k): int(v) for k, v in cross_actor_orig.items()}
            if hold_vehicles:
                for vid in sorted(hold_vehicles.keys()):
                    if int(vid) in vehicles:
                        continue
                    vehicles[int(vid)] = list(hold_vehicles[int(vid)])
                    vehicle_times[int(vid)] = [float(t) for t in hold_vehicle_times.get(int(vid), [])]
                    obj_info[int(vid)] = dict(hold_obj_info.get(int(vid), {}))
                    actor_source_subdir[int(vid)] = str(hold_actor_source.get(int(vid), ""))
                    actor_orig_vid[int(vid)] = int(hold_actor_orig.get(int(vid), int(vid)))
            removed = _safe_int(cross_stats.get("cross_id_removed", 0), 0) if isinstance(cross_stats, dict) else 0
            clusters = _safe_int(cross_stats.get("cross_id_clusters", 0), 0) if isinstance(cross_stats, dict) else 0
            if removed > 0:
                print(
                    "[INFO] Cross-ID dedup: removed={} clusters={} pairs_checked={}".format(
                        int(removed),
                        int(clusters),
                        int(_safe_int(cross_stats.get("cross_id_pair_checks", 0), 0)) if isinstance(cross_stats, dict) else 0,
                    )
                )
        except Exception as exc:
            print(f"[WARN] Cross-ID dedup pass failed; continuing with local overlap dedup only: {exc}")

    # Pass 2: local overlap-aware dedup (adds short overlap handoff handling).
    (
        vehicles,
        vehicle_times,
        obj_info,
        actor_source_subdir,
        actor_orig_vid,
        local_stats,
    ) = _deduplicate_actor_tracks_by_overlap(
        vehicles=vehicles,
        vehicle_times=vehicle_times,
        obj_info=obj_info,
        actor_source_subdir=actor_source_subdir,
        actor_orig_vid=actor_orig_vid,
        dt=float(dt),
    )
    local_removed = len(local_stats.get("removed_ids", [])) if isinstance(local_stats, dict) else 0
    if local_removed > 0:
        print(
            "[INFO] Overlap dedup: removed={} clusters={} short_pairs={} segment_pairs={} handoff_pairs={} low_motion_same_orig_pairs={} pair_checks={}".format(
                int(local_removed),
                int(_safe_int(local_stats.get("clusters", 0), 0)),
                int(_safe_int(local_stats.get("short_overlap_pairs", 0), 0)),
                int(_safe_int(local_stats.get("segment_overlap_pairs", 0), 0)),
                int(_safe_int(local_stats.get("handoff_pairs", 0), 0)),
                int(_safe_int(local_stats.get("low_motion_same_orig_pairs", 0), 0)),
                int(_safe_int(local_stats.get("pair_checks", 0), 0)),
            )
        )

    # Pass 3: strict ego-vs-actor dedup using time-aligned overlap metrics.
    vehicles, vehicle_times, obj_info, ego_stats = _deduplicate_ego_actor_overlap(
        vehicles=vehicles,
        vehicle_times=vehicle_times,
        ego_trajs=ego_trajs,
        ego_times=ego_times,
        obj_info=obj_info,
        dt=float(dt),
    )
    ego_removed = len(ego_stats.get("removed_ids", [])) if isinstance(ego_stats, dict) else 0
    if ego_removed > 0:
        print(
            "[INFO] Ego-actor dedup: removed={} pair_checks={}".format(
                int(ego_removed),
                int(_safe_int(ego_stats.get("pair_checks", 0), 0)),
            )
        )

    return vehicles, vehicle_times, obj_info


def _track_significance_score(track: Dict[str, object]) -> float:
    """Higher = more important to keep. Used to pick a victim for deletion
    when two vehicles collide and neither can be un-snapped without overlap."""
    role = str(track.get("role", "")).strip().lower()
    if role == "ego":
        return 1e9  # never delete ego
    n_frames = len(track.get("frames") or [])
    parked_pen = -200.0 if _is_parked_vehicle_track_for_overlap(track) else 0.0
    quasi_parked_pen = -80.0 if _is_quasi_parked_vehicle_track_for_overlap(track) else 0.0
    # Long trajectories beat short ones; non-parked beats parked.
    return float(n_frames) + parked_pen + quasi_parked_pen


def _merge_duplicate_vehicle_tracks(
    tracks: List[Dict[str, object]],
    scenario_name: str = "",
    verbose: bool = False,
) -> Dict[str, object]:
    """Detect and remove duplicate-detection tracks.

    V2X-PnP-real often contains multiple sensor-agent detections of the same
    physical vehicle that are propagated as separate tracks. They typically
    overlap heavily in time and in space (center-to-center distance well
    under one half-vehicle-length), and have correlated headings. The
    downstream residual-collision resolver only fixes per-frame overlaps
    — it cannot recognise that two whole tracks describe the same actor.

    Heuristic: pair tracks A and B where, during their temporal overlap
    (>=5 shared frames), median raw center-to-center distance <
    `(L_a + L_b) * 0.45`, and median yaw difference < 35°. Drop the less
    significant of the pair. Ego tracks are never dropped.
    """
    report: Dict[str, object] = {
        "enabled": True,
        "scenario": str(scenario_name),
        "duplicate_pairs": [],
        "tracks_dropped": [],
    }
    if _env_int("V2X_CARLA_DUPLICATE_TRACK_MERGE_ENABLED", 1, minimum=0, maximum=1) != 1:
        report["enabled"] = False
        return report

    overlap_min_frames = _env_int(
        "V2X_CARLA_DUPLICATE_TRACK_MIN_OVERLAP_FRAMES", 5, minimum=2, maximum=200,
    )
    median_dist_factor = _env_float(
        "V2X_CARLA_DUPLICATE_TRACK_MEDIAN_DIST_FACTOR", 0.45,
    )
    yaw_match_deg = _env_float(
        "V2X_CARLA_DUPLICATE_TRACK_YAW_MATCH_DEG", 35.0,
    )

    # Only consider vehicle/ego role tracks
    vehicle_tracks: List[Dict[str, object]] = []
    for tr in tracks:
        if not isinstance(tr, dict):
            continue
        role = str(tr.get("role", "")).strip().lower()
        if role in ("vehicle", "ego"):
            vehicle_tracks.append(tr)

    if len(vehicle_tracks) < 2:
        return report

    # Pre-extract per-track frame index → (rx, ry, ryaw, length, width)
    def _idx_table(tr: Dict[str, object]) -> Tuple[Dict[int, Tuple[float, float, float]], float, float]:
        out: Dict[int, Tuple[float, float, float]] = {}
        L = float(tr.get("length", 4.5) or 4.5)
        W = float(tr.get("width", 2.0) or 2.0)
        frames = tr.get("frames") or []
        for i, f in enumerate(frames):
            if not isinstance(f, dict):
                continue
            rx = f.get("x"); ry = f.get("y"); ryaw = f.get("yaw")
            try:
                rx_v = float(rx); ry_v = float(ry); ryaw_v = float(ryaw)
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(rx_v) and math.isfinite(ry_v) and math.isfinite(ryaw_v)):
                continue
            # Use the frame's own time index when available (some pipelines
            # have non-contiguous frame lists). Fall back to position.
            t = f.get("t")
            try:
                key = int(round(float(t) * 100.0))  # 10ms buckets
            except (TypeError, ValueError):
                key = i
            out[key] = (rx_v, ry_v, ryaw_v)
        return out, L, W

    tables: List[Tuple[Dict[str, object], Dict[int, Tuple[float, float, float]], float, float]] = []
    for tr in vehicle_tracks:
        tab, L, W = _idx_table(tr)
        if tab:
            tables.append((tr, tab, L, W))

    drop_set: set = set()  # python ids of tracks to drop
    pairs_recorded: List[Dict[str, object]] = []
    spawn_coincident_thresh_m = _env_float(
        "V2X_CARLA_DUPLICATE_TRACK_SPAWN_COINCIDENT_M", 1.5,
    )
    spawn_coincident_min_frames = _env_int(
        "V2X_CARLA_DUPLICATE_TRACK_SPAWN_COINCIDENT_FRAMES", 3, minimum=1, maximum=20,
    )
    for i in range(len(tables)):
        tr_i, tab_i, L_i, W_i = tables[i]
        if id(tr_i) in drop_set:
            continue
        for j in range(i + 1, len(tables)):
            tr_j, tab_j, L_j, W_j = tables[j]
            if id(tr_j) in drop_set:
                continue
            shared_keys = set(tab_i.keys()) & set(tab_j.keys())
            if len(shared_keys) < overlap_min_frames:
                continue
            sorted_shared = sorted(shared_keys)
            dists: List[float] = []
            yaw_diffs: List[float] = []
            for k in sorted_shared:
                xa, ya, yawa = tab_i[k]
                xb, yb, yawb = tab_j[k]
                dists.append(math.hypot(xa - xb, ya - yb))
                d_yaw = ((yawb - yawa + 540.0) % 360.0) - 180.0
                yaw_diffs.append(abs(d_yaw))
            sorted_dists = sorted(dists); sorted_yaws = sorted(yaw_diffs)
            med_d = sorted_dists[len(sorted_dists) // 2]
            med_yaw = sorted_yaws[len(sorted_yaws) // 2]
            threshold_dist = (L_i + L_j) * 0.5 * median_dist_factor + 0.4

            # A) full-trajectory duplicate: median distance is small along the
            #    whole shared window
            full_dup = (med_d <= threshold_dist) and (med_yaw <= yaw_match_deg)

            # B) spawn-coincident duplicate: same vehicle whose perception
            #    track-ID flipped mid-recording — the first several frames
            #    are near-identical, even if later they diverge.
            n_head = min(int(spawn_coincident_min_frames + 4), len(sorted_shared))
            head_max_d = max(dists[:n_head]) if dists[:n_head] else float("inf")
            head_max_yaw = max(yaw_diffs[:n_head]) if yaw_diffs[:n_head] else 180.0
            spawn_dup = (
                head_max_d <= float(spawn_coincident_thresh_m)
                and head_max_yaw <= yaw_match_deg
                and len(sorted_shared) >= int(spawn_coincident_min_frames)
            )
            if not (full_dup or spawn_dup):
                continue
            # Pick which to drop. Never drop ego.
            role_i = str(tr_i.get("role", "")).strip().lower()
            role_j = str(tr_j.get("role", "")).strip().lower()
            if role_i == "ego" and role_j == "ego":
                # Different egos collocated — leave alone.
                continue
            if role_i == "ego":
                drop_id = id(tr_j); keep_tr = tr_i; drop_tr = tr_j
            elif role_j == "ego":
                drop_id = id(tr_i); keep_tr = tr_j; drop_tr = tr_i
            else:
                # Higher significance score wins.
                sig_i = _track_significance_score(tr_i)
                sig_j = _track_significance_score(tr_j)
                if sig_i >= sig_j:
                    drop_id = id(tr_j); keep_tr = tr_i; drop_tr = tr_j
                else:
                    drop_id = id(tr_i); keep_tr = tr_j; drop_tr = tr_i
            drop_set.add(drop_id)
            pairs_recorded.append({
                "keep_id": str(keep_tr.get("id", "?")),
                "drop_id": str(drop_tr.get("id", "?")),
                "median_dist_m": round(med_d, 3),
                "median_yaw_diff_deg": round(med_yaw, 2),
                "head_max_dist_m": round(head_max_d, 3),
                "shared_frames": int(len(shared_keys)),
                "threshold_dist_m": round(threshold_dist, 3),
                "match_kind": "full" if full_dup else "spawn",
            })

    # C) Cross-time spawn coincidence: same vehicle re-IDed across a temporal
    #    gap (e.g. perception lost track and re-acquired). Compares each pair's
    #    frame[0] xy/yaw directly, regardless of t-key alignment. Only fires
    #    when at least one of the two tracks is short (likely a re-ID sliver).
    spawn_xy_thresh = _env_float("V2X_CARLA_DUPLICATE_TRACK_SPAWN_XY_THRESH_M", 1.2)
    spawn_xt_yaw_deg = _env_float("V2X_CARLA_DUPLICATE_TRACK_SPAWN_XT_YAW_DEG", 35.0)
    spawn_xt_short_max = _env_int(
        "V2X_CARLA_DUPLICATE_TRACK_SPAWN_XT_SHORT_MAX_FRAMES",
        25, minimum=2, maximum=200,
    )
    for i in range(len(tables)):
        tr_i, tab_i, L_i, W_i = tables[i]
        if id(tr_i) in drop_set:
            continue
        fi0 = (tr_i.get("frames") or [None])[0]
        if not isinstance(fi0, dict):
            continue
        for j in range(i + 1, len(tables)):
            tr_j, tab_j, L_j, W_j = tables[j]
            if id(tr_j) in drop_set:
                continue
            fj0 = (tr_j.get("frames") or [None])[0]
            if not isinstance(fj0, dict):
                continue
            try:
                xa = float(fi0["x"]); ya = float(fi0["y"]); yawa = float(fi0["yaw"])
                xb = float(fj0["x"]); yb = float(fj0["y"]); yawb = float(fj0["yaw"])
            except (KeyError, TypeError, ValueError):
                continue
            d0 = math.hypot(xa - xb, ya - yb)
            if d0 > float(spawn_xy_thresh):
                continue
            dyaw0 = abs(((yawb - yawa + 540.0) % 360.0) - 180.0)
            if dyaw0 > float(spawn_xt_yaw_deg):
                continue
            n_i = len(tr_i.get("frames") or [])
            n_j = len(tr_j.get("frames") or [])
            if min(n_i, n_j) > int(spawn_xt_short_max):
                continue  # both long-lived → safer to leave alone
            role_i = str(tr_i.get("role", "")).strip().lower()
            role_j = str(tr_j.get("role", "")).strip().lower()
            if role_i == "ego" and role_j == "ego":
                continue
            if role_i == "ego":
                drop_id = id(tr_j); keep_tr = tr_i; drop_tr = tr_j
            elif role_j == "ego":
                drop_id = id(tr_i); keep_tr = tr_j; drop_tr = tr_i
            else:
                if n_i < n_j:
                    drop_id = id(tr_i); keep_tr = tr_j; drop_tr = tr_i
                elif n_j < n_i:
                    drop_id = id(tr_j); keep_tr = tr_i; drop_tr = tr_j
                else:
                    sig_i = _track_significance_score(tr_i)
                    sig_j = _track_significance_score(tr_j)
                    if sig_i >= sig_j:
                        drop_id = id(tr_j); keep_tr = tr_i; drop_tr = tr_j
                    else:
                        drop_id = id(tr_i); keep_tr = tr_j; drop_tr = tr_i
            drop_set.add(drop_id)
            pairs_recorded.append({
                "keep_id": str(keep_tr.get("id", "?")),
                "drop_id": str(drop_tr.get("id", "?")),
                "frame0_dist_m": round(d0, 3),
                "frame0_yaw_diff_deg": round(dyaw0, 2),
                "shorter_n_frames": int(min(n_i, n_j)),
                "match_kind": "spawn_xtime",
            })

    if drop_set:
        # Mutate `tracks` in place: remove dropped entries.
        kept: List[Dict[str, object]] = []
        dropped_ids: List[str] = []
        for tr in tracks:
            if id(tr) in drop_set:
                dropped_ids.append(str(tr.get("id", "?")))
                continue
            kept.append(tr)
        tracks[:] = kept
        report["tracks_dropped"] = dropped_ids
    report["duplicate_pairs"] = pairs_recorded
    return report


def _resolve_residual_vehicle_collisions(
    tracks: List[Dict[str, object]],
    scenario_name: str = "",
    verbose: bool = False,
) -> Dict[str, object]:
    """Final-pass collision resolver for vehicle/ego pairs.

    After the upstream overlap reducer runs, any *remaining* sustained
    collision is treated as a snap artifact. For each colliding pair:

    1. Compute average snap displacement |(cx, cy) - (x, y)| per vehicle
       across the colliding run. The vehicle with greater displacement is
       furthest from its raw observation → most likely on the wrong lane.
    2. Replace that vehicle's (cx, cy) with its raw (x, y) at those frames
       (un-snap toward the actually-observed position).
    3. Re-check OBB overlap. If now resolved, keep the un-snap. If still
       overlapping (raw observations themselves were too close), delete
       the LESS significant track.

    Significance: ego >> long-trajectory moving > short moving > parked.
    """
    report: Dict[str, object] = {
        "enabled": True,
        "scenario": str(scenario_name),
        "collision_runs": 0,
        "frames_unsnapped": 0,
        "tracks_unsnapped_ids": [],
        "tracks_deleted_ids": [],
    }
    if _env_int("V2X_CARLA_RESIDUAL_COLLISION_RESOLVE_ENABLED", 1, minimum=0, maximum=1) != 1:
        report["enabled"] = False
        report["reason"] = "disabled_by_flag"
        return report
    if not isinstance(tracks, list) or len(tracks) < 2:
        report["reason"] = "too_few_tracks"
        return report

    pen_thresh_m = _env_float("V2X_CARLA_RESIDUAL_PEN_THRESH_M", 0.12)
    min_run_frames = _env_int("V2X_CARLA_RESIDUAL_MIN_COLLISION_FRAMES", 3, minimum=1, maximum=30)
    safety_inflate_m = _env_float("V2X_CARLA_RESIDUAL_SAFETY_MARGIN_M", 0.10)
    t_round_decimals = _env_int("V2X_CARLA_RESIDUAL_T_ROUND_DECIMALS", 2, minimum=1, maximum=4)
    skip_parked_pairs = _env_int("V2X_CARLA_RESIDUAL_SKIP_PARKED_PAIRS", 1, minimum=0, maximum=1) == 1

    # Build a per-track index (id, dims, time→frame_idx).
    metas: List[Dict[str, object]] = []
    for tr in tracks:
        role = str(tr.get("role", "")).strip().lower()
        if role not in ("vehicle", "ego"):
            continue
        frames = tr.get("frames") or []
        if not frames:
            continue
        L, W = _vehicle_dims_for_overlap(tr)
        L_eff = float(L) + float(safety_inflate_m)
        W_eff = float(W) + float(safety_inflate_m)
        t_to_fi: Dict[float, int] = {}
        for fi, fr in enumerate(frames):
            t_val = _safe_float(fr.get("t"), float("nan"))
            if math.isfinite(t_val):
                t_to_fi[round(float(t_val), int(t_round_decimals))] = int(fi)
        metas.append({
            "track": tr,
            "id": str(tr.get("id", "")),
            "role": role,
            "L": L_eff,
            "W": W_eff,
            "t_to_fi": t_to_fi,
            "n_frames": len(frames),
            "is_parked": _is_parked_vehicle_track_for_overlap(tr),
            "is_quasi_parked": _is_quasi_parked_vehicle_track_for_overlap(tr),
        })
    if len(metas) < 2:
        report["reason"] = "fewer_than_2_vehicle_tracks"
        return report

    deleted_ids: set = set()

    def _pose(meta: Dict[str, object], fi: int) -> Tuple[float, float, float, float, float]:
        fr = meta["track"]["frames"][int(fi)]
        cx = _safe_float(fr.get("cx"), float("nan"))
        cy = _safe_float(fr.get("cy"), float("nan"))
        cyaw = _safe_float(fr.get("cyaw"), _safe_float(fr.get("yaw"), 0.0))
        x = _safe_float(fr.get("x"), float("nan"))
        y = _safe_float(fr.get("y"), float("nan"))
        # If snap missing, fall back to raw.
        if not math.isfinite(cx) or not math.isfinite(cy):
            cx, cy = x, y
        return float(cx), float(cy), float(cyaw), float(x), float(y)

    def _pen(ma, fi_a, mb, fi_b, *, use_raw_a=False, use_raw_b=False) -> float:
        ax, ay, ayaw, axr, ayr = _pose(ma, fi_a)
        bx, by, byaw, bxr, byr = _pose(mb, fi_b)
        if use_raw_a and math.isfinite(axr) and math.isfinite(ayr):
            ax, ay = axr, ayr
        if use_raw_b and math.isfinite(bxr) and math.isfinite(byr):
            bx, by = bxr, byr
        if not all(math.isfinite(v) for v in (ax, ay, bx, by)):
            return 0.0
        return _obb_overlap_penetration_xyyaw(
            ax, ay, ayaw, float(ma["L"]), float(ma["W"]),
            bx, by, byaw, float(mb["L"]), float(mb["W"]),
        )

    def _snap_disp(meta, fi) -> float:
        fr = meta["track"]["frames"][int(fi)]
        cx = _safe_float(fr.get("cx"), float("nan"))
        cy = _safe_float(fr.get("cy"), float("nan"))
        x = _safe_float(fr.get("x"), float("nan"))
        y = _safe_float(fr.get("y"), float("nan"))
        if not all(math.isfinite(v) for v in (cx, cy, x, y)):
            return 0.0
        return float(math.hypot(cx - x, cy - y))

    for i in range(len(metas)):
        if metas[i]["id"] in deleted_ids:
            continue
        for j in range(i + 1, len(metas)):
            if metas[i]["id"] in deleted_ids or metas[j]["id"] in deleted_ids:
                continue
            ma, mb = metas[i], metas[j]
            if skip_parked_pairs and ma["is_parked"] and mb["is_parked"]:
                continue  # parked-vs-parked is handled by upstream reducer

            shared = sorted(set(ma["t_to_fi"]) & set(mb["t_to_fi"]))
            if not shared:
                continue

            # Group consecutive collision frames into runs.
            runs: List[List[Tuple[float, int, int]]] = []
            cur: List[Tuple[float, int, int]] = []
            for t_val in shared:
                fi_a = ma["t_to_fi"][t_val]
                fi_b = mb["t_to_fi"][t_val]
                if _pen(ma, fi_a, mb, fi_b) >= float(pen_thresh_m):
                    cur.append((t_val, fi_a, fi_b))
                else:
                    if len(cur) >= int(min_run_frames):
                        runs.append(cur)
                    cur = []
            if len(cur) >= int(min_run_frames):
                runs.append(cur)
            if not runs:
                continue

            for run in runs:
                report["collision_runs"] = int(report["collision_runs"]) + 1
                # Average snap displacement → pick the more-snapped vehicle.
                disp_a = sum(_snap_disp(ma, fa) for _, fa, _ in run) / float(len(run))
                disp_b = sum(_snap_disp(mb, fb) for _, _, fb in run) / float(len(run))
                # Strong preference: never modify a parked vehicle's position
                # (it should stay locked to its static pose). If exactly one
                # of the pair is parked, the OTHER one must absorb the fix.
                a_parked = bool(ma["is_parked"]) or bool(ma["is_quasi_parked"])
                b_parked = bool(mb["is_parked"]) or bool(mb["is_quasi_parked"])
                if a_parked and not b_parked:
                    fix_meta, fix_idx_run = mb, [fb for _, _, fb in run]
                    keep_meta, keep_idx_run = ma, [fa for _, fa, _ in run]
                    fix_is_a = False
                elif b_parked and not a_parked:
                    fix_meta, fix_idx_run = ma, [fa for _, fa, _ in run]
                    keep_meta, keep_idx_run = mb, [fb for _, _, fb in run]
                    fix_is_a = True
                elif disp_a >= disp_b:
                    fix_meta, fix_idx_run = ma, [fa for _, fa, _ in run]
                    keep_meta, keep_idx_run = mb, [fb for _, _, fb in run]
                    fix_is_a = True
                else:
                    fix_meta, fix_idx_run = mb, [fb for _, _, fb in run]
                    keep_meta, keep_idx_run = ma, [fa for _, fa, _ in run]
                    fix_is_a = False

                # Try un-snapping: probe with raw_a / raw_b.
                still_collide = False
                for k, fi_fix in enumerate(fix_idx_run):
                    fi_keep = keep_idx_run[k]
                    if fix_is_a:
                        pen_after = _pen(ma, fi_fix, mb, fi_keep, use_raw_a=True)
                    else:
                        pen_after = _pen(ma, fi_keep, mb, fi_fix, use_raw_b=True)
                    if pen_after >= float(pen_thresh_m):
                        still_collide = True
                        break

                if not still_collide:
                    # Apply the un-snap to the fix vehicle's frames in this run.
                    n_modified = 0
                    for fi_fix in fix_idx_run:
                        fr = fix_meta["track"]["frames"][fi_fix]
                        x = _safe_float(fr.get("x"), float("nan"))
                        y = _safe_float(fr.get("y"), float("nan"))
                        if not (math.isfinite(x) and math.isfinite(y)):
                            continue
                        fr["cx"] = float(x)
                        fr["cy"] = float(y)
                        # Yaw: keep raw yaw if available; else current cyaw.
                        raw_yaw = _safe_float(fr.get("yaw"), float("nan"))
                        if math.isfinite(raw_yaw):
                            fr["cyaw"] = float(raw_yaw)
                        fr["csource"] = "collision_unsnap"
                        fr["collision_unsnap"] = True
                        n_modified += 1
                    if n_modified > 0:
                        report["frames_unsnapped"] = int(report["frames_unsnapped"]) + int(n_modified)
                        ids_list = report["tracks_unsnapped_ids"]  # type: ignore
                        if isinstance(ids_list, list) and fix_meta["id"] not in ids_list:
                            ids_list.append(fix_meta["id"])
                        # Smooth the boundary transitions back to the surrounding
                        # snapped trajectory using shape-preserving blend.
                        # _shape_preserving_blend is exported from runtime_projection
                        # via the module wildcard import at file top.
                        boundary_radius = 3
                        fix_frames = fix_meta["track"]["frames"]
                        s_run = int(min(fix_idx_run))
                        e_run = int(max(fix_idx_run))
                        if s_run - boundary_radius >= 0:
                            _shape_preserving_blend(  # type: ignore[name-defined]
                                fix_frames,
                                s_run - boundary_radius,
                                s_run,
                                "collision_unsnap_boundary",
                            )
                        if e_run + boundary_radius < len(fix_frames):
                            _shape_preserving_blend(  # type: ignore[name-defined]
                                fix_frames,
                                e_run,
                                e_run + boundary_radius,
                                "collision_unsnap_boundary",
                            )
                    if verbose:
                        print(
                            f"[COLLISION] un-snapped {fix_meta['id']} for "
                            f"{n_modified} frame(s) (collision with {keep_meta['id']})"
                        )
                else:
                    # Un-snap doesn't help → delete the less significant track.
                    sig_a = _track_significance_score(ma["track"])
                    sig_b = _track_significance_score(mb["track"])
                    loser_meta = ma if sig_a < sig_b else mb
                    deleted_ids.add(loser_meta["id"])
                    ids_list = report["tracks_deleted_ids"]  # type: ignore
                    if isinstance(ids_list, list):
                        ids_list.append(loser_meta["id"])
                    if verbose:
                        survivor = mb if loser_meta is ma else ma
                        print(
                            f"[COLLISION] deleting {loser_meta['id']} "
                            f"(collides with {survivor['id']}, un-snap insufficient, "
                            f"sig_a={sig_a:.0f} sig_b={sig_b:.0f})"
                        )
                    # Move on to next pair (the loser is removed, no further runs).
                    break

    # Apply deletions.
    if deleted_ids:
        original_n = len(tracks)
        tracks[:] = [t for t in tracks if str(t.get("id", "")) not in deleted_ids]
        if verbose:
            print(f"[COLLISION] Removed {original_n - len(tracks)} track(s) from dataset")

    # Raw-teleport trim: drop frames where raw position teleports by an
    # impossible distance in one timestep (e.g., perception track-ID mix-up
    # where a far-away detection gets the same ID as our actor). At 10 Hz
    # with dt=0.1 s, even 50 m/s ≈ 180 km/h is only 5 m/frame — anything
    # over the threshold is a track confusion, not real motion.
    if _env_int("V2X_CARLA_RESIDUAL_RAW_TELEPORT_TRIM_ENABLED", 1, minimum=0, maximum=1) == 1:
        teleport_thresh_m = _env_float("V2X_CARLA_RESIDUAL_RAW_TELEPORT_THRESH_M", 10.0)
        for tr in tracks:
            role = str(tr.get("role", "")).strip().lower()
            if role not in ("vehicle", "ego"):
                continue
            frames_t = tr.get("frames") or []
            if len(frames_t) < 4:
                continue
            n_frames_t = len(frames_t)
            # Find first/last frame in the longest contiguous run of "no
            # teleports". Drop frames outside that run — preserves the
            # main trajectory and discards any teleport-prefix or suffix.
            best_run = (0, n_frames_t - 1)
            best_len = 0
            cur_start = 0
            for i in range(1, n_frames_t):
                f0 = frames_t[i - 1]; f1 = frames_t[i]
                x0 = _safe_float(f0.get("x"), float("nan"))
                y0 = _safe_float(f0.get("y"), float("nan"))
                x1 = _safe_float(f1.get("x"), float("nan"))
                y1 = _safe_float(f1.get("y"), float("nan"))
                if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
                    continue
                step = math.hypot(x1 - x0, y1 - y0)
                if step > float(teleport_thresh_m):
                    run_len = i - 1 - cur_start
                    if run_len > best_len:
                        best_len = run_len
                        best_run = (cur_start, i - 1)
                    cur_start = i
            run_len = n_frames_t - 1 - cur_start
            if run_len > best_len:
                best_run = (cur_start, n_frames_t - 1)
                best_len = run_len
            s_run, e_run = best_run
            if best_len < 4:
                continue
            # If the longest non-teleport run isn't the entire track, trim.
            if s_run > 0 or e_run < (n_frames_t - 1):
                tr["frames"] = frames_t[s_run : e_run + 1]

    # If a track had ANY frames un-snapped to raw by the collision resolver,
    # propagate the un-snap to the whole track. The user's principle:
    # "collision usually means the lanes aren't what we think — the right
    # answer is a long-term lane assumption change, not a brief jump out of
    # a lane to dodge a single colliding frame."
    #
    # Only fires when the un-snapped frames represent a meaningful portion of
    # the track (>= MIN_FRAC) and the snap is otherwise far enough from raw
    # that the snap is genuinely contested (median snap_disp > MIN_DISP).
    if _env_int("V2X_CARLA_RESIDUAL_PROPAGATE_UNSNAP_ENABLED", 1, minimum=0, maximum=1) == 1:
        prop_min_frac = _env_float("V2X_CARLA_RESIDUAL_PROPAGATE_UNSNAP_MIN_FRAC", 0.05)
        prop_min_disp_m = _env_float("V2X_CARLA_RESIDUAL_PROPAGATE_UNSNAP_MIN_DISP_M", 0.7)
        for tr in tracks:
            role = str(tr.get("role", "")).strip().lower()
            if role not in ("vehicle", "ego"):
                continue
            frames_t = tr.get("frames") or []
            if len(frames_t) < 6:
                continue
            n_un = sum(
                1
                for fr in frames_t
                if str(fr.get("csource", "")).startswith("collision_unsnap")
            )
            if n_un == 0:
                continue
            if n_un < prop_min_frac * len(frames_t):
                continue
            # Median snap_disp guard — only propagate when snap is consistently
            # offset from raw (otherwise we'd be erasing valid snapping).
            disps: List[float] = []
            for fr in frames_t:
                cx = _safe_float(fr.get("cx"), float("nan"))
                cy = _safe_float(fr.get("cy"), float("nan"))
                x = _safe_float(fr.get("x"), float("nan"))
                y = _safe_float(fr.get("y"), float("nan"))
                if all(math.isfinite(v) for v in (cx, cy, x, y)):
                    disps.append(math.hypot(cx - x, cy - y))
            if not disps:
                continue
            sorted_disps = sorted(disps)
            med = sorted_disps[len(sorted_disps) // 2]
            if med < float(prop_min_disp_m):
                continue
            # Force the rest of the frames to raw too — long-term lane change.
            for fr in frames_t:
                cs = str(fr.get("csource", ""))
                if cs.startswith("collision_unsnap"):
                    continue
                x = _safe_float(fr.get("x"), float("nan"))
                y = _safe_float(fr.get("y"), float("nan"))
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue
                fr["cx"] = float(x)
                fr["cy"] = float(y)
                raw_yaw = _safe_float(fr.get("yaw"), float("nan"))
                if math.isfinite(raw_yaw):
                    fr["cyaw"] = float(raw_yaw)
                fr["csource"] = "collision_unsnap_propagated"
                fr["collision_unsnap_propagated"] = True

    # Force-to-raw fallback for tracks where snap is consistently far from raw.
    # If median snap-displacement > threshold, the snapper picked a wrong lane
    # entirely (no nearby CARLA lane fits the raw trajectory). The collision
    # logic may have un-snapped a few frames but left others at the wrong
    # offset, creating zigzag. Force-snap all to raw to keep the trajectory
    # continuous with raw observations.
    if _env_int("V2X_CARLA_RESIDUAL_FORCE_RAW_FOR_BAD_SNAP_ENABLED", 1, minimum=0, maximum=1) == 1:
        force_raw_med_thresh = _env_float("V2X_CARLA_RESIDUAL_FORCE_RAW_MEDIAN_THRESH_M", 2.5)
        for tr in tracks:
            role = str(tr.get("role", "")).strip().lower()
            if role not in ("vehicle", "ego"):
                continue
            frames_t = tr.get("frames") or []
            if len(frames_t) < 6:
                continue
            disps: List[float] = []
            for fr in frames_t:
                cx = _safe_float(fr.get("cx"), float("nan"))
                cy = _safe_float(fr.get("cy"), float("nan"))
                x = _safe_float(fr.get("x"), float("nan"))
                y = _safe_float(fr.get("y"), float("nan"))
                if all(math.isfinite(v) for v in (cx, cy, x, y)):
                    disps.append(math.hypot(cx - x, cy - y))
            if len(disps) < 6:
                continue
            sorted_disps = sorted(disps)
            med = sorted_disps[len(sorted_disps) // 2]
            if med < float(force_raw_med_thresh):
                continue
            # Snap is consistently wrong — fall back to raw for the whole track.
            for fr in frames_t:
                x = _safe_float(fr.get("x"), float("nan"))
                y = _safe_float(fr.get("y"), float("nan"))
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue
                fr["cx"] = float(x)
                fr["cy"] = float(y)
                raw_yaw = _safe_float(fr.get("yaw"), float("nan"))
                if math.isfinite(raw_yaw):
                    fr["cyaw"] = float(raw_yaw)
                fr["csource"] = "force_raw_bad_snap"
                fr["force_raw_bad_snap"] = True

    # Snap-outlier trim. If most frames have snap ≈ raw (median snap_disp
    # below threshold), any frame whose snap_disp is much larger is a snap
    # outlier from spawn/teardown that should match the rest of the track.
    # Force such outliers to raw. The MEDIAN guard makes this safe for
    # legitimately-snapped tracks (which have non-trivial median disp).
    if _env_int("V2X_CARLA_RESIDUAL_OUTLIER_TRIM_ENABLED", 1, minimum=0, maximum=1) == 1:
        outlier_thresh_m = _env_float("V2X_CARLA_RESIDUAL_OUTLIER_TRIM_THRESH_M", 1.5)
        max_median_disp_m = _env_float("V2X_CARLA_RESIDUAL_OUTLIER_TRIM_MAX_MEDIAN_M", 0.4)
        for tr in tracks:
            role = str(tr.get("role", "")).strip().lower()
            if role not in ("vehicle", "ego"):
                continue
            frames_t = tr.get("frames") or []
            if len(frames_t) < 6:
                continue
            disps: List[Tuple[int, float]] = []
            for fi, fr in enumerate(frames_t):
                cx = _safe_float(fr.get("cx"), float("nan"))
                cy = _safe_float(fr.get("cy"), float("nan"))
                x = _safe_float(fr.get("x"), float("nan"))
                y = _safe_float(fr.get("y"), float("nan"))
                if all(math.isfinite(v) for v in (cx, cy, x, y)):
                    disps.append((fi, math.hypot(cx - x, cy - y)))
            if len(disps) < 6:
                continue
            sorted_disps = sorted(d for _, d in disps)
            med = sorted_disps[len(sorted_disps) // 2]
            if med >= float(max_median_disp_m):
                continue  # most frames not on raw → not the spawn-outlier pattern
            # Force outlier frames to raw.
            for fi, dval in disps:
                if dval > float(outlier_thresh_m):
                    fr = frames_t[fi]
                    x = _safe_float(fr.get("x"), float("nan"))
                    y = _safe_float(fr.get("y"), float("nan"))
                    if math.isfinite(x) and math.isfinite(y):
                        fr["cx"] = float(x)
                        fr["cy"] = float(y)
                        raw_yaw = _safe_float(fr.get("yaw"), float("nan"))
                        if math.isfinite(raw_yaw):
                            fr["cyaw"] = float(raw_yaw)
                        fr["csource"] = "outlier_trim_to_raw"
                        fr["outlier_trim_to_raw"] = True

    return report

    return vehicles, vehicle_times, obj_info
