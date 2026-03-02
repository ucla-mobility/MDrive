#!/usr/bin/env python3
"""
Planner-likeness metric suite for plot_trajectories_on_map HTML outputs.

This script evaluates whether CARLA-projected trajectories look like planner
outputs while preserving timing and staying close to raw trajectories.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        out = float(v)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


def _safe_int(v: object, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(str(name), None)
    if raw is None:
        return float(default)
    return _safe_float(raw, float(default))


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(str(name), None)
    if raw is None:
        return int(default)
    return _safe_int(raw, int(default))


def _yaw_wrap_rad(a: float) -> float:
    return float((float(a) + math.pi) % (2.0 * math.pi) - math.pi)


def _yaw_abs_diff_deg(a_deg: float, b_deg: float) -> float:
    a = math.radians(float(a_deg))
    b = math.radians(float(b_deg))
    return float(abs(math.degrees(_yaw_wrap_rad(a - b))))


def _p95(vals: Sequence[float]) -> float:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size <= 0:
        return 0.0
    return float(np.percentile(arr, 95.0))


def _mad(vals: Sequence[float]) -> float:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size <= 0:
        return 0.0
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)))


def _extract_dataset_json(html_text: str) -> Dict[str, object]:
    m = re.search(
        r'<script id="dataset" type="application/json">(.*?)</script>',
        html_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not m:
        raise RuntimeError("Could not find embedded dataset JSON in HTML.")
    return json.loads(m.group(1))


def _to_scenarios(root: Dict[str, object]) -> List[Dict[str, object]]:
    if isinstance(root.get("scenarios"), list):
        return [s for s in root["scenarios"] if isinstance(s, dict)]
    return [root]


def _frame_raw_xy(fr: Dict[str, object]) -> Tuple[float, float]:
    return (_safe_float(fr.get("x"), 0.0), _safe_float(fr.get("y"), 0.0))


def _frame_v2_xy(fr: Dict[str, object]) -> Tuple[float, float]:
    return (
        _safe_float(fr.get("sx"), _safe_float(fr.get("x"), 0.0)),
        _safe_float(fr.get("sy"), _safe_float(fr.get("y"), 0.0)),
    )


def _frame_carla_xy(fr: Dict[str, object]) -> Tuple[float, float]:
    return (
        _safe_float(fr.get("cx"), _safe_float(fr.get("sx"), _safe_float(fr.get("x"), 0.0))),
        _safe_float(fr.get("cy"), _safe_float(fr.get("sy"), _safe_float(fr.get("y"), 0.0))),
    )


def _frame_carla_pre_xy(fr: Dict[str, object]) -> Tuple[float, float]:
    return (
        _safe_float(
            fr.get("cbx"),
            _safe_float(fr.get("cx"), _safe_float(fr.get("sx"), _safe_float(fr.get("x"), 0.0))),
        ),
        _safe_float(
            fr.get("cby"),
            _safe_float(fr.get("cy"), _safe_float(fr.get("sy"), _safe_float(fr.get("y"), 0.0))),
        ),
    )


def _frame_raw_yaw(fr: Dict[str, object]) -> float:
    return _safe_float(fr.get("yaw"), 0.0)


def _frame_v2_yaw(fr: Dict[str, object], fallback: float) -> float:
    return _safe_float(fr.get("syaw"), fallback)


def _frame_carla_yaw(fr: Dict[str, object], fallback: float) -> float:
    return _safe_float(fr.get("cyaw"), fallback)


def _frame_carla_pre_yaw(fr: Dict[str, object], fallback: float) -> float:
    return _safe_float(fr.get("cbyaw"), _safe_float(fr.get("cyaw"), fallback))


def _point_seg_dist(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    vx = float(bx) - float(ax)
    vy = float(by) - float(ay)
    wx = float(px) - float(ax)
    wy = float(py) - float(ay)
    den = vx * vx + vy * vy
    if den <= 1e-12:
        return float(math.hypot(wx, wy))
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / den))
    qx = float(ax) + t * vx
    qy = float(ay) + t * vy
    return float(math.hypot(float(px) - qx, float(py) - qy))


def _point_polyline_dist(pt: Tuple[float, float], poly: np.ndarray) -> float:
    if poly.shape[0] <= 0:
        return 0.0
    if poly.shape[0] == 1:
        return float(math.hypot(float(pt[0]) - float(poly[0, 0]), float(pt[1]) - float(poly[0, 1])))
    px, py = float(pt[0]), float(pt[1])
    best = float("inf")
    for i in range(1, int(poly.shape[0])):
        d = _point_seg_dist(px, py, float(poly[i - 1, 0]), float(poly[i - 1, 1]), float(poly[i, 0]), float(poly[i, 1]))
        if d < best:
            best = d
    return float(best)


def _directed_hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] <= 0 or b.shape[0] <= 0:
        return 0.0
    vals: List[float] = []
    for i in range(int(a.shape[0])):
        vals.append(_point_polyline_dist((float(a[i, 0]), float(a[i, 1])), b))
    return float(max(vals)) if vals else 0.0


def _hausdorff_sym(a: np.ndarray, b: np.ndarray) -> float:
    return float(max(_directed_hausdorff(a, b), _directed_hausdorff(b, a)))


def _downsample_xy(arr: np.ndarray, max_n: int = 160) -> np.ndarray:
    n = int(arr.shape[0])
    if n <= max_n:
        return arr
    idx = np.linspace(0, n - 1, num=int(max_n), dtype=np.int32)
    return arr[idx]


def _discrete_frechet(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] <= 0 or b.shape[0] <= 0:
        return 0.0
    n = int(a.shape[0])
    m = int(b.shape[0])
    ca = np.full((n, m), -1.0, dtype=np.float64)

    def _dist(i: int, j: int) -> float:
        dx = float(a[i, 0]) - float(b[j, 0])
        dy = float(a[i, 1]) - float(b[j, 1])
        return float(math.hypot(dx, dy))

    for i in range(n):
        for j in range(m):
            d = _dist(i, j)
            if i == 0 and j == 0:
                ca[i, j] = d
            elif i == 0:
                ca[i, j] = max(ca[i, j - 1], d)
            elif j == 0:
                ca[i, j] = max(ca[i - 1, j], d)
            else:
                ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), d)
    return float(ca[n - 1, m - 1])


def _steps_and_speed(xy: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = int(xy.shape[0])
    if n <= 1:
        return np.zeros((n,), dtype=np.float64), np.zeros((n,), dtype=np.float64)
    ds = np.zeros((n,), dtype=np.float64)
    speed = np.zeros((n,), dtype=np.float64)
    dt_min = _env_float("PLM_SPEED_DT_MIN_S", 5e-2)
    for i in range(1, n):
        d = float(math.hypot(float(xy[i, 0] - xy[i - 1, 0]), float(xy[i, 1] - xy[i - 1, 1])))
        dt = max(float(dt_min), float(t[i] - t[i - 1]))
        ds[i] = d
        speed[i] = d / dt
    return ds, speed


def _first_diff(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    n = int(v.shape[0])
    out = np.zeros((n,), dtype=np.float64)
    if n <= 1:
        return out
    dt_min = _env_float("PLM_DERIV_DT_MIN_S", 5e-2)
    for i in range(1, n):
        dt = max(float(dt_min), float(t[i] - t[i - 1]))
        out[i] = float(v[i] - v[i - 1]) / dt
    return out


def _motion_yaw_deg(xy: np.ndarray) -> np.ndarray:
    n = int(xy.shape[0])
    out = np.zeros((n,), dtype=np.float64)
    if n <= 1:
        return out
    for i in range(n):
        if 0 < i < (n - 1):
            dx = float(xy[i + 1, 0] - xy[i - 1, 0])
            dy = float(xy[i + 1, 1] - xy[i - 1, 1])
        elif i + 1 < n:
            dx = float(xy[i + 1, 0] - xy[i, 0])
            dy = float(xy[i + 1, 1] - xy[i, 1])
        else:
            dx = float(xy[i, 0] - xy[i - 1, 0])
            dy = float(xy[i, 1] - xy[i - 1, 1])
        if math.hypot(dx, dy) < 1e-5:
            out[i] = out[i - 1] if i > 0 else 0.0
        else:
            out[i] = float(math.degrees(math.atan2(dy, dx)))
    return out


def _curvature_series(xy: np.ndarray) -> np.ndarray:
    n = int(xy.shape[0])
    out = np.zeros((n,), dtype=np.float64)
    if n < 3:
        return out
    seg_min_m = _env_float("PLM_CURV_SEG_MIN_M", 0.12)
    for i in range(1, n - 1):
        x0, y0 = float(xy[i - 1, 0]), float(xy[i - 1, 1])
        x1, y1 = float(xy[i, 0]), float(xy[i, 1])
        x2, y2 = float(xy[i + 1, 0]), float(xy[i + 1, 1])
        yaw1 = math.atan2(y1 - y0, x1 - x0)
        yaw2 = math.atan2(y2 - y1, x2 - x1)
        dpsi = _yaw_wrap_rad(yaw2 - yaw1)
        ds1 = math.hypot(x1 - x0, y1 - y0)
        ds2 = math.hypot(x2 - x1, y2 - y1)
        if ds1 < float(seg_min_m) or ds2 < float(seg_min_m):
            out[i] = out[i - 1] if i > 1 else 0.0
            continue
        ds = max(1e-3, 0.5 * (ds1 + ds2))
        out[i] = float(dpsi / ds)
    out[0] = out[1]
    out[-1] = out[-2]
    return out


def _signed_yaw_delta_deg(curr_deg: float, prev_deg: float) -> float:
    d = _yaw_wrap_rad(math.radians(float(curr_deg) - float(prev_deg)))
    return float(math.degrees(d))


def _count_id_aba(ids: np.ndarray, min_valid_id: int = 0) -> int:
    n = int(ids.shape[0])
    if n < 3:
        return 0
    out = 0
    for i in range(1, n - 1):
        a = int(ids[i - 1])
        b = int(ids[i])
        c = int(ids[i + 1])
        if a < int(min_valid_id) or b < int(min_valid_id) or c < int(min_valid_id):
            continue
        if a == c and b != a:
            out += 1
    return int(out)


def _transition_indices_from_ids(ids: np.ndarray, min_valid_id: int = 0) -> List[int]:
    n = int(ids.shape[0])
    if n < 2:
        return []
    out: List[int] = []
    for i in range(1, n):
        a = int(ids[i - 1])
        b = int(ids[i])
        if a < int(min_valid_id) or b < int(min_valid_id):
            continue
        if a != b:
            out.append(int(i))
    return out


def _semantic_transition_indices(frames: Sequence[Dict[str, object]]) -> List[int]:
    n = len(frames)
    if n < 2:
        return []
    out: List[int] = []
    for i in range(1, n):
        a = frames[i - 1]
        b = frames[i]
        a_assigned = _safe_int(a.get("assigned_lane_id"), 0)
        b_assigned = _safe_int(b.get("assigned_lane_id"), 0)
        if a_assigned != 0 and b_assigned != 0:
            if int(a_assigned) != int(b_assigned):
                out.append(int(i))
            continue
        a_lane = _safe_int(a.get("lane_id"), 0)
        b_lane = _safe_int(b.get("lane_id"), 0)
        if a_lane != 0 and b_lane != 0 and int(a_lane) != int(b_lane):
            out.append(int(i))
            continue
        a_idx = _safe_int(a.get("lane_index"), -1)
        b_idx = _safe_int(b.get("lane_index"), -1)
        if a_idx >= 0 and b_idx >= 0 and int(a_idx) != int(b_idx):
            out.append(int(i))
    return out


def _intersection_inconsistency_events(
    motion_yaw_deg: np.ndarray,
    transition_indices: Sequence[int],
    n: int,
) -> int:
    if n < 4:
        return 0
    if not transition_indices:
        return 0
    window = max(1, _env_int("PLM_STAGE_INTERSECTION_WINDOW_FRAMES", 4))
    step_thresh = _env_float("PLM_STAGE_INTERSECTION_STEP_YAW_MIN_DEG", 4.0)
    straight_net_thresh = _env_float("PLM_STAGE_INTERSECTION_STRAIGHT_NET_DEG", 14.0)
    opposite_ratio_max = _env_float("PLM_STAGE_INTERSECTION_OPPOSITE_RATIO_MAX", 0.35)
    min_steps = max(2, _env_int("PLM_STAGE_INTERSECTION_MIN_STEPS", 3))
    events = 0
    for idx in transition_indices:
        i = int(idx)
        lo = max(1, int(i) - int(window))
        hi = min(int(n) - 1, int(i) + int(window))
        signed_steps: List[float] = []
        for k in range(lo, hi + 1):
            d = float(_signed_yaw_delta_deg(float(motion_yaw_deg[k]), float(motion_yaw_deg[k - 1])))
            if abs(float(d)) >= float(step_thresh):
                signed_steps.append(float(d))
        if len(signed_steps) < int(min_steps):
            continue
        pos = int(sum(1 for v in signed_steps if float(v) > 0.0))
        neg = int(sum(1 for v in signed_steps if float(v) < 0.0))
        net = float(_signed_yaw_delta_deg(float(motion_yaw_deg[hi]), float(motion_yaw_deg[lo - 1])))
        if abs(float(net)) < float(straight_net_thresh):
            if pos > 0 and neg > 0:
                events += 1
        else:
            turn_sign = 1.0 if float(net) > 0.0 else -1.0
            opposite = int(sum(1 for v in signed_steps if float(v) * float(turn_sign) < 0.0))
            if float(opposite) / max(1.0, float(len(signed_steps))) > float(opposite_ratio_max):
                events += 1
    return int(events)


def _stage_metric_bundle(
    xy: np.ndarray,
    yaw_deg: np.ndarray,
    raw_xy: np.ndarray,
    raw_step: np.ndarray,
    t: np.ndarray,
    line_ids: Optional[np.ndarray],
    lane_ids: Optional[np.ndarray],
    transition_indices: Sequence[int],
    synthetic_turn_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    n = int(xy.shape[0])
    if n <= 0:
        return {
            "step_p95_m": 0.0,
            "accel_p95": 0.0,
            "jerk_p95": 0.0,
            "low_motion_jump_events": 0.0,
            "catastrophic_jump_events": 0.0,
            "line_oscillation_events": 0.0,
            "lane_oscillation_events": 0.0,
            "raw_fidelity_median_m": 0.0,
            "raw_fidelity_p95_m": 0.0,
            "raw_fidelity_max_m": 0.0,
            "intersection_inconsistent_events": 0.0,
            "wrong_way_events": 0.0,
            "facing_violation_events": 0.0,
        }

    step, speed = _steps_and_speed(xy, t)
    accel = _first_diff(speed, t)
    jerk = _first_diff(accel, t)
    err_raw = np.linalg.norm(xy - raw_xy, axis=1)
    motion_yaw = _motion_yaw_deg(xy)

    low_raw_thr = _env_float("PLM_STAGE_LOW_MOTION_RAW_STEP_MAX_M", 0.9)
    jump_abs_min = _env_float("PLM_STAGE_LOW_MOTION_JUMP_ABS_MIN_M", 1.35)
    jump_ratio_min = _env_float("PLM_STAGE_LOW_MOTION_JUMP_RATIO", 2.6)
    cat_abs_min = _env_float("PLM_STAGE_CATASTROPHIC_JUMP_ABS_MIN_M", 4.5)
    cat_ratio_min = _env_float("PLM_STAGE_CATASTROPHIC_JUMP_RATIO", 6.0)

    low_motion_jump_events = 0
    catastrophic_jump_events = 0
    for i in range(1, n):
        rs = float(raw_step[i]) if i < int(raw_step.shape[0]) else 0.0
        ss = float(step[i])
        if rs < float(low_raw_thr) and ss > max(float(jump_abs_min), float(jump_ratio_min) * float(rs)):
            low_motion_jump_events += 1
        if rs < float(low_raw_thr) and ss > max(float(cat_abs_min), float(cat_ratio_min) * float(rs)):
            catastrophic_jump_events += 1

    line_osc = 0
    if isinstance(line_ids, np.ndarray) and int(line_ids.shape[0]) == n:
        line_osc = _count_id_aba(line_ids, min_valid_id=0)
    lane_osc = 0
    if isinstance(lane_ids, np.ndarray) and int(lane_ids.shape[0]) == n:
        lane_osc = _count_id_aba(lane_ids, min_valid_id=1)

    facing_min_step = _env_float("PLM_STAGE_FACING_MIN_STAGE_STEP_M", 0.20)
    facing_min_raw_step = _env_float("PLM_STAGE_FACING_MIN_RAW_STEP_M", 0.12)
    facing_delta_min = _env_float("PLM_STAGE_FACING_YAW_DELTA_MIN_DEG", 120.0)
    wrong_way_delta_min = _env_float("PLM_STAGE_WRONG_WAY_YAW_DELTA_MIN_DEG", 150.0)
    wrong_way_min_run = max(1, _env_int("PLM_STAGE_WRONG_WAY_MIN_RUN", 2))

    facing_violation_events = 0
    wrong_way_raw_idx: List[int] = []
    for i in range(1, n):
        rs = float(raw_step[i]) if i < int(raw_step.shape[0]) else 0.0
        ss = float(step[i])
        if ss < float(facing_min_step) or rs < float(facing_min_raw_step):
            continue
        dy = float(_yaw_abs_diff_deg(float(yaw_deg[i]), float(motion_yaw[i])))
        if dy >= float(facing_delta_min):
            facing_violation_events += 1
        if isinstance(line_ids, np.ndarray):
            if i < int(line_ids.shape[0]) and int(line_ids[i]) >= 0 and dy >= float(wrong_way_delta_min):
                if synthetic_turn_mask is not None and i < int(synthetic_turn_mask.shape[0]) and bool(synthetic_turn_mask[i]):
                    continue
                wrong_way_raw_idx.append(int(i))

    wrong_way_events = 0
    if wrong_way_raw_idx:
        s = int(wrong_way_raw_idx[0])
        e = int(wrong_way_raw_idx[0])
        for ii in wrong_way_raw_idx[1:]:
            i = int(ii)
            if i <= int(e) + 1:
                e = i
            else:
                if int(e) - int(s) + 1 >= int(wrong_way_min_run):
                    wrong_way_events += int(e) - int(s) + 1
                s = i
                e = i
        if int(e) - int(s) + 1 >= int(wrong_way_min_run):
            wrong_way_events += int(e) - int(s) + 1

    intersection_inconsistent_events = _intersection_inconsistency_events(
        motion_yaw_deg=motion_yaw,
        transition_indices=transition_indices,
        n=n,
    )

    return {
        "step_p95_m": float(_p95(step)),
        "accel_p95": float(_p95(np.abs(accel))),
        "jerk_p95": float(_p95(np.abs(jerk))),
        "low_motion_jump_events": float(low_motion_jump_events),
        "catastrophic_jump_events": float(catastrophic_jump_events),
        "line_oscillation_events": float(line_osc),
        "lane_oscillation_events": float(lane_osc),
        "raw_fidelity_median_m": float(np.median(err_raw)),
        "raw_fidelity_p95_m": float(_p95(err_raw)),
        "raw_fidelity_max_m": float(np.max(err_raw)),
        "intersection_inconsistent_events": float(intersection_inconsistent_events),
        "wrong_way_events": float(wrong_way_events),
        "facing_violation_events": float(facing_violation_events),
    }


def _vehicle_dims_from_track(track: Dict[str, object]) -> Tuple[float, float]:
    role = str(track.get("role", "")).strip().lower()
    obj = str(track.get("obj_type", "")).strip().lower()
    if role == "ego":
        return (4.8, 2.1)
    if "bus" in obj:
        return (12.0, 2.9)
    if "concrete" in obj:
        return (9.5, 2.8)
    if "truck" in obj:
        return (8.8, 2.7)
    if "van" in obj:
        return (5.4, 2.1)
    if any(tok in obj for tok in ("motor", "scooter", "bike", "bicycle")):
        return (2.3, 1.0)
    if any(tok in obj for tok in ("trash", "cone", "barrier")):
        return (1.0, 1.0)
    return (4.6, 2.0)


def _obb_overlap_penetration(
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


def _annotate_scenario_overlaps(
    scenario_name: str,
    tracks: List[Dict[str, object]],
    rows: List[Dict[str, object]],
) -> Dict[str, object]:
    if not tracks or not rows:
        return {
            "scenario": str(scenario_name),
            "overlap_events": 0,
            "overlap_pair_events": 0,
            "max_penetration_m": 0.0,
            "tracks_with_overlap": 0,
        }

    track_by_id: Dict[str, Dict[str, object]] = {}
    frame_maps: Dict[str, Dict[int, int]] = {}
    dims_by_id: Dict[str, Tuple[float, float]] = {}
    speeds_by_id: Dict[str, Dict[int, float]] = {}
    parked_like_by_id: Dict[str, bool] = {}

    dt_key = 0.1
    inv_dt_key = 1.0 / dt_key

    for tr in tracks:
        role = str(tr.get("role", "")).strip().lower()
        if role not in {"ego", "vehicle"}:
            continue
        tid = str(tr.get("id"))
        frames = tr.get("frames", [])
        if not isinstance(frames, list) or len(frames) < 2:
            continue
        track_by_id[tid] = tr
        dims_by_id[tid] = _vehicle_dims_from_track(tr)
        parked_like_by_id[tid] = bool(tr.get("low_motion_vehicle", False))

        fmap: Dict[int, int] = {}
        smap: Dict[int, float] = {}
        for i, fr in enumerate(frames):
            t = _safe_float(fr.get("t"), float(i) * dt_key)
            tk = int(round(float(t) * inv_dt_key))
            fmap[tk] = int(i)
        for i, fr in enumerate(frames):
            t = _safe_float(fr.get("t"), float(i) * dt_key)
            tk = int(round(float(t) * inv_dt_key))
            x, y = _frame_carla_xy(fr)
            spd = 0.0
            if i > 0:
                x0, y0 = _frame_carla_xy(frames[i - 1])
                t0 = _safe_float(frames[i - 1].get("t"), float(i - 1) * dt_key)
                dt = max(5e-2, float(t) - float(t0))
                spd = max(spd, float(math.hypot(float(x) - float(x0), float(y) - float(y0))) / dt)
            if i + 1 < len(frames):
                x1, y1 = _frame_carla_xy(frames[i + 1])
                t1 = _safe_float(frames[i + 1].get("t"), float(i + 1) * dt_key)
                dt = max(5e-2, float(t1) - float(t))
                spd = max(spd, float(math.hypot(float(x1) - float(x), float(y1) - float(y))) / dt)
            smap[tk] = float(spd)
        frame_maps[tid] = fmap
        speeds_by_id[tid] = smap

    ids = sorted(frame_maps.keys())
    if len(ids) < 2:
        for row in rows:
            row["overlap_events"] = 0
            row["overlap_pair_events"] = 0
            row["overlap_max_penetration_m"] = 0.0
            hc = row.get("hard_constraints", {})
            if isinstance(hc, dict):
                hc["no_overlap"] = True
                hc["all"] = bool(hc.get("all", False))
                row["hard_constraints"] = hc
        return {
            "scenario": str(scenario_name),
            "overlap_events": 0,
            "overlap_pair_events": 0,
            "max_penetration_m": 0.0,
            "tracks_with_overlap": 0,
        }

    # Collect all discrete time buckets where any vehicle/ego appears.
    all_tk: set = set()
    for tid in ids:
        all_tk.update(frame_maps[tid].keys())

    per_track_event_keys: Dict[str, set] = {tid: set() for tid in ids}
    per_track_pair_keys: Dict[str, set] = {tid: set() for tid in ids}
    per_track_max_pen: Dict[str, float] = {tid: 0.0 for tid in ids}
    overlap_pair_events = 0
    overlap_max_pen = 0.0
    min_pen_thresh = _env_float("PLM_OVERLAP_MIN_PEN_M", 0.10)
    moving_speed_thresh = _env_float("PLM_OVERLAP_MOVING_SPEED_THRESH_MPS", 0.45)
    skip_low_speed_pairs = _env_int("PLM_OVERLAP_SKIP_LOW_SPEED_PAIRS", 1) == 1
    ignore_parked_like = _env_int("PLM_OVERLAP_IGNORE_PARKED_LIKE", 1) == 1
    raw_overlap_filter_enabled = _env_int("PLM_OVERLAP_RAW_FILTER_ENABLED", 1) == 1
    raw_overlap_filter_scale = _env_float("PLM_OVERLAP_RAW_FILTER_SCALE", 0.75)
    raw_overlap_filter_min_m = _env_float("PLM_OVERLAP_RAW_FILTER_MIN_M", 0.08)

    for tk in sorted(all_tk):
        active: List[Tuple[str, Dict[str, object], int]] = []
        for tid in ids:
            fi = frame_maps[tid].get(int(tk))
            if fi is None:
                continue
            tr = track_by_id.get(tid)
            if tr is None:
                continue
            frames = tr.get("frames", [])
            if not isinstance(frames, list) or fi < 0 or fi >= len(frames):
                continue
            active.append((tid, tr, fi))
        if len(active) < 2:
            continue

        for i in range(len(active)):
            tid_i, tr_i, fi_i = active[i]
            fr_i = tr_i["frames"][fi_i]
            xi, yi = _frame_carla_xy(fr_i)
            yiw = _frame_carla_yaw(fr_i, _safe_float(fr_i.get("yaw"), 0.0))
            li, wi = dims_by_id.get(tid_i, (4.6, 2.0))
            spi = _safe_float(speeds_by_id.get(tid_i, {}).get(int(tk), 0.0), 0.0)
            for j in range(i + 1, len(active)):
                tid_j, tr_j, fi_j = active[j]
                fr_j = tr_j["frames"][fi_j]
                xj, yj = _frame_carla_xy(fr_j)
                ywj = _frame_carla_yaw(fr_j, _safe_float(fr_j.get("yaw"), 0.0))
                lj, wj = dims_by_id.get(tid_j, (4.6, 2.0))
                spj = _safe_float(speeds_by_id.get(tid_j, {}).get(int(tk), 0.0), 0.0)
                parked_i = bool(parked_like_by_id.get(tid_i, False))
                parked_j = bool(parked_like_by_id.get(tid_j, False))

                # Overlap quality is currently low-priority; by default skip
                # parked-like actor interactions entirely.
                if ignore_parked_like and (parked_i or parked_j):
                    continue

                # Optionally skip low-speed/low-speed contacts (parked-vs-parked).
                if skip_low_speed_pairs and spi <= moving_speed_thresh and spj <= moving_speed_thresh:
                    continue

                center_dist = float(math.hypot(float(xi) - float(xj), float(yi) - float(yj)))
                diag_bound = 0.6 * (math.hypot(li, wi) + math.hypot(lj, wj))
                if center_dist > diag_bound + 0.5:
                    continue

                pen = _obb_overlap_penetration(
                    x1=float(xi),
                    y1=float(yi),
                    yaw1_deg=float(yiw),
                    len1=float(li),
                    wid1=float(wi),
                    x2=float(xj),
                    y2=float(yj),
                    yaw2_deg=float(ywj),
                    len2=float(lj),
                    wid2=float(wj),
                )
                if pen < min_pen_thresh:
                    continue
                # Only count overlap that is introduced/amplified by CARLA projection.
                # If raw trajectory already overlaps similarly, do not flag it here.
                rxi, ryi = _frame_raw_xy(fr_i)
                rxj, ryj = _frame_raw_xy(fr_j)
                ryawi = _frame_raw_yaw(fr_i)
                ryawj = _frame_raw_yaw(fr_j)
                raw_pen = _obb_overlap_penetration(
                    x1=float(rxi),
                    y1=float(ryi),
                    yaw1_deg=float(ryawi),
                    len1=float(li),
                    wid1=float(wi),
                    x2=float(rxj),
                    y2=float(ryj),
                    yaw2_deg=float(ryawj),
                    len2=float(lj),
                    wid2=float(wj),
                )
                if raw_overlap_filter_enabled:
                    if raw_pen >= max(float(raw_overlap_filter_min_m), float(raw_overlap_filter_scale) * float(min_pen_thresh)):
                        continue

                overlap_pair_events += 1
                overlap_max_pen = max(overlap_max_pen, float(pen))
                pair_key = f"{min(tid_i, tid_j)}|{max(tid_i, tid_j)}|{int(tk)}"
                per_track_event_keys[tid_i].add(int(tk))
                per_track_event_keys[tid_j].add(int(tk))
                per_track_pair_keys[tid_i].add(pair_key)
                per_track_pair_keys[tid_j].add(pair_key)
                per_track_max_pen[tid_i] = max(float(per_track_max_pen[tid_i]), float(pen))
                per_track_max_pen[tid_j] = max(float(per_track_max_pen[tid_j]), float(pen))

    row_by_id: Dict[str, Dict[str, object]] = {str(r.get("id")): r for r in rows}
    tracks_with_overlap = 0
    overlap_events_total = 0
    for tid, row in row_by_id.items():
        ev = int(len(per_track_event_keys.get(tid, set())))
        pe = int(len(per_track_pair_keys.get(tid, set())))
        max_pen = float(per_track_max_pen.get(tid, 0.0))
        overlap_events_total += ev
        if ev > 0:
            tracks_with_overlap += 1
        row["overlap_events"] = int(ev)
        row["overlap_pair_events"] = int(pe)
        row["overlap_max_penetration_m"] = float(max_pen)
        # Penalize suspicious overlap behavior.
        overlap_event_susp_w = _env_float("PLM_SUSPICION_OVERLAP_EVENT_W", 0.0)
        overlap_pen_susp_w = _env_float("PLM_SUSPICION_OVERLAP_PEN_W", 0.0)
        row["planner_suspicion_score"] = float(_safe_float(row.get("planner_suspicion_score"), 0.0)) + (
            float(overlap_event_susp_w) * min(10.0, float(ev))
            + float(overlap_pen_susp_w) * min(6.0, float(max_pen / 0.15))
        )
        hc = row.get("hard_constraints", {})
        if isinstance(hc, dict):
            hc["no_overlap"] = bool(ev == 0)
            all_keys = [
                "no_lane_jumps",
                "low_variance",
                "time_fidelity",
                "lane_semantics",
                "intersection_smooth",
                "similarity",
                "low_noise",
            ]
            if _env_int("PLM_OVERLAP_INCLUDE_IN_HARD", 0) == 1:
                all_keys.append("no_overlap")
            hc["all"] = bool(all(bool(hc.get(k, False)) for k in all_keys))
            row["hard_constraints"] = hc

    return {
        "scenario": str(scenario_name),
        "overlap_events": int(overlap_events_total),
        "overlap_pair_events": int(overlap_pair_events),
        "max_penetration_m": float(overlap_max_pen),
        "tracks_with_overlap": int(tracks_with_overlap),
    }


@dataclass
class TrackEval:
    row: Dict[str, object]
    hard_ok: bool


def _merge_indices(indices: Iterable[int], pad: int = 1) -> List[Tuple[int, int]]:
    vals = sorted({int(i) for i in indices if int(i) >= 0})
    if not vals:
        return []
    out: List[Tuple[int, int]] = []
    s = vals[0]
    e = vals[0]
    for i in vals[1:]:
        if i <= (e + 1 + int(pad)):
            e = i
        else:
            out.append((int(s), int(e)))
            s, e = i, i
    out.append((int(s), int(e)))
    return out


def _analyze_track(
    scenario_name: str,
    track: Dict[str, object],
    min_frames: int,
    include_roles: Sequence[str],
) -> Optional[TrackEval]:
    role = str(track.get("role", ""))
    if role not in include_roles:
        return None
    frames = track.get("frames", [])
    if not isinstance(frames, list) or len(frames) < int(min_frames):
        return None

    n = len(frames)
    t = np.asarray([_safe_float(fr.get("t"), float(i) * 0.1) for i, fr in enumerate(frames)], dtype=np.float64)
    raw_xy = np.asarray([_frame_raw_xy(fr) for fr in frames], dtype=np.float64)
    v2_xy = np.asarray([_frame_v2_xy(fr) for fr in frames], dtype=np.float64)
    carla_pre_xy = np.asarray([_frame_carla_pre_xy(fr) for fr in frames], dtype=np.float64)
    carla_xy = np.asarray([_frame_carla_xy(fr) for fr in frames], dtype=np.float64)
    raw_yaw = np.asarray([_frame_raw_yaw(fr) for fr in frames], dtype=np.float64)
    v2_motion_yaw = _motion_yaw_deg(v2_xy)
    v2_yaw = np.asarray(
        [_frame_v2_yaw(fr, float(v2_motion_yaw[i])) for i, fr in enumerate(frames)],
        dtype=np.float64,
    )
    carla_pre_motion_yaw = _motion_yaw_deg(carla_pre_xy)
    carla_pre_yaw = np.asarray(
        [_frame_carla_pre_yaw(fr, float(carla_pre_motion_yaw[i])) for i, fr in enumerate(frames)],
        dtype=np.float64,
    )
    carla_motion_yaw = _motion_yaw_deg(carla_xy)
    carla_yaw = np.asarray(
        [_frame_carla_yaw(fr, float(carla_motion_yaw[i])) for i, fr in enumerate(frames)],
        dtype=np.float64,
    )

    semantic_lane_ids = np.asarray(
        [
            _safe_int(
                fr.get("assigned_lane_id"),
                _safe_int(fr.get("lane_id"), 0),
            )
            for fr in frames
        ],
        dtype=np.int64,
    )
    ccli_pre = np.asarray([_safe_int(fr.get("cbcli"), _safe_int(fr.get("ccli"), -1)) for fr in frames], dtype=np.int64)
    ccli = np.asarray([_safe_int(fr.get("ccli"), -1) for fr in frames], dtype=np.int64)
    synthetic_turn_mask = np.asarray([bool(fr.get("synthetic_turn", False)) for fr in frames], dtype=bool)

    dt = np.diff(t)
    nonmono = int(np.sum(dt < -1e-8))
    duplicates = int(np.sum(np.abs(dt) <= 1e-8))
    dt_pos = dt[dt > 1e-8]
    dt_med = float(np.median(dt_pos)) if dt_pos.size > 0 else 0.1
    dt_cv = float(np.std(dt_pos) / max(1e-6, np.mean(dt_pos))) if dt_pos.size > 0 else 0.0

    d_raw, v_raw = _steps_and_speed(raw_xy, t)
    d_v2, v_v2 = _steps_and_speed(v2_xy, t)
    d_carla, v_carla = _steps_and_speed(carla_xy, t)
    jerk_smooth_win = max(1, _env_int("PLM_JERK_SMOOTH_WIN", 1))
    if jerk_smooth_win > 1:
        ker = np.ones((int(jerk_smooth_win),), dtype=np.float64) / float(jerk_smooth_win)
        v_raw_eval = np.convolve(v_raw, ker, mode="same")
        v_carla_eval = np.convolve(v_carla, ker, mode="same")
    else:
        v_raw_eval = v_raw
        v_carla_eval = v_carla
    a_raw = _first_diff(v_raw_eval, t)
    a_carla = _first_diff(v_carla_eval, t)
    j_raw = _first_diff(a_raw, t)
    j_carla = _first_diff(a_carla, t)

    err_raw_v2 = np.linalg.norm(v2_xy - raw_xy, axis=1)
    err_raw_carla = np.linalg.norm(carla_xy - raw_xy, axis=1)
    err_v2_carla = np.linalg.norm(carla_xy - v2_xy, axis=1)
    per_step_disp_err = np.abs(d_carla - d_raw)

    lateral_err: List[float] = []
    heading_err: List[float] = []
    for i in range(n):
        yaw = math.radians(float(raw_yaw[i]))
        dx = float(carla_xy[i, 0] - raw_xy[i, 0])
        dy = float(carla_xy[i, 1] - raw_xy[i, 1])
        lat = abs(-math.sin(yaw) * dx + math.cos(yaw) * dy)
        lateral_err.append(float(lat))
        heading_err.append(float(_yaw_abs_diff_deg(float(raw_yaw[i]), float(carla_yaw[i]))))

    lane_jump_raw_step_max = _env_float("PLM_LANE_JUMP_RAW_STEP_MAX_M", 0.9)
    lane_jump_carla_abs_min = _env_float("PLM_LANE_JUMP_CARLA_ABS_MIN_M", 1.35)
    lane_jump_ratio_min = _env_float("PLM_LANE_JUMP_RATIO", 2.6)
    lane_jump_include_aba = _env_int("PLM_LANE_JUMP_COUNT_ABA", 1) == 1
    lane_changes = 0
    lane_jump_indices: List[int] = []
    ccli_transition_indices: List[int] = []
    for i in range(1, n):
        if ccli[i] >= 0 and ccli[i - 1] >= 0 and int(ccli[i]) != int(ccli[i - 1]):
            lane_changes += 1
            ccli_transition_indices.append(i)
            if float(d_raw[i]) < float(lane_jump_raw_step_max) and float(d_carla[i]) > max(
                float(lane_jump_carla_abs_min), float(lane_jump_ratio_min) * float(d_raw[i])
            ):
                lane_jump_indices.append(i)

    aba_indices: List[int] = []
    for i in range(1, n - 1):
        a = int(ccli[i - 1])
        b = int(ccli[i])
        c = int(ccli[i + 1])
        if a >= 0 and b >= 0 and c >= 0 and a == c and b != a:
            if _env_int("PLM_ABA_ENABLE", 1) == 1:
                aba_indices.append(i)
                if lane_jump_include_aba:
                    lane_jump_indices.append(i)

    wrong_way_min_carla_step = _env_float("PLM_WRONG_WAY_MIN_CARLA_STEP_M", 0.28)
    wrong_way_min_raw_step = _env_float("PLM_WRONG_WAY_MIN_RAW_STEP_M", 0.20)
    wrong_way_yaw_delta_min = _env_float("PLM_WRONG_WAY_YAW_DELTA_MIN_DEG", 150.0)
    wrong_way_min_run = max(1, _env_int("PLM_WRONG_WAY_MIN_RUN", 2))
    wrong_way_require_adjacent = _env_int("PLM_WRONG_WAY_REQUIRE_ADJACENT", 1) == 1
    wrong_way_raw: List[int] = []
    for i in range(n):
        if int(ccli[i]) < 0:
            continue
        if bool(frames[i].get("synthetic_turn", False)):
            continue
        if float(d_carla[i]) < float(wrong_way_min_carla_step) or float(d_raw[i]) < float(wrong_way_min_raw_step):
            continue
        dy = _yaw_abs_diff_deg(float(carla_yaw[i]), float(carla_motion_yaw[i]))
        if dy >= float(wrong_way_yaw_delta_min):
            wrong_way_raw.append(i)
    wrong_way_idx: List[int] = []
    if wrong_way_raw:
        if not wrong_way_require_adjacent and wrong_way_min_run <= 1:
            wrong_way_idx = [int(i) for i in wrong_way_raw]
        else:
            s = int(wrong_way_raw[0])
            e = int(wrong_way_raw[0])
            for i in wrong_way_raw[1:]:
                ii = int(i)
                if ii <= e + 1:
                    e = ii
                else:
                    run_len = int(e - s + 1)
                    if run_len >= int(wrong_way_min_run):
                        wrong_way_idx.extend(range(int(s), int(e) + 1))
                    s = ii
                    e = ii
            run_len = int(e - s + 1)
            if run_len >= int(wrong_way_min_run):
                wrong_way_idx.extend(range(int(s), int(e) + 1))
            elif wrong_way_require_adjacent and run_len >= 2:
                wrong_way_idx.extend(range(int(s), int(e) + 1))

    cdist = np.asarray([_safe_float(fr.get("cdist"), 0.0) for fr in frames], dtype=np.float64)
    lane_boundary_violations = int(np.sum(cdist > 3.2))

    frechet = _discrete_frechet(_downsample_xy(raw_xy), _downsample_xy(carla_xy))
    haus = _hausdorff_sym(_downsample_xy(raw_xy), _downsample_xy(carla_xy))
    raw_poly = _downsample_xy(raw_xy, max_n=240)
    lat_to_raw = [_point_polyline_dist((float(carla_xy[i, 0]), float(carla_xy[i, 1])), raw_poly) for i in range(n)]

    yaw_step = [
        float(_yaw_abs_diff_deg(float(carla_yaw[i]), float(carla_yaw[i - 1])))
        for i in range(1, n)
        if float(d_carla[i]) >= 0.18
    ]
    yaw_rate = np.asarray([0.0] + [float(_yaw_wrap_rad(math.radians(carla_yaw[i] - carla_yaw[i - 1]))) for i in range(1, n)], dtype=np.float64)
    yaw_osc = 0
    for i in range(2, n):
        if float(d_carla[i]) < 0.18 or float(d_carla[i - 1]) < 0.18:
            continue
        a = float(yaw_rate[i - 1])
        b = float(yaw_rate[i])
        if abs(a) < math.radians(0.8) or abs(b) < math.radians(0.8):
            continue
        if a * b < 0.0:
            yaw_osc += 1

    k_raw = _curvature_series(raw_xy)
    k_carla = _curvature_series(carla_xy)
    dk_raw = _first_diff(k_raw, t)
    dk_carla = _first_diff(k_carla, t)
    curv_min_step_m = _env_float("PLM_CURV_MIN_STEP_M", 0.0)
    if curv_min_step_m > 0.0:
        curv_vals = [abs(float(dk_carla[i])) for i in range(n) if float(d_carla[i]) >= float(curv_min_step_m)]
        k_jitter = _p95(curv_vals) if curv_vals else _p95(np.abs(dk_carla))
    else:
        k_jitter = _p95(np.abs(dk_carla))
    k_sign_flip = 0
    k_sign_flip_step_gate = max(0.18, float(curv_min_step_m))
    for i in range(2, n):
        if float(d_carla[i]) < k_sign_flip_step_gate or float(d_carla[i - 1]) < k_sign_flip_step_gate:
            continue
        a = float(k_carla[i - 1])
        b = float(k_carla[i])
        if abs(a) < 0.025 or abs(b) < 0.025:
            continue
        if a * b < 0:
            k_sign_flip += 1

    speed_hf = v_carla - np.convolve(v_carla, np.ones((5,)) / 5.0, mode="same")
    speed_hf_rms = float(np.sqrt(np.mean(np.square(speed_hf)))) if n > 1 else 0.0
    jerk_p95 = _p95(np.abs(j_carla))
    jerk_min_speed_mps = _env_float("PLM_JERK_MIN_SPEED_MPS", 0.0)
    if jerk_min_speed_mps > 0.0:
        carla_mask = np.asarray(v_carla >= float(jerk_min_speed_mps), dtype=bool)
        raw_mask = np.asarray(v_raw >= float(jerk_min_speed_mps), dtype=bool)
        carla_vals = np.abs(j_carla[carla_mask]) if np.any(carla_mask) else np.abs(j_carla)
        raw_vals = np.abs(j_raw[raw_mask]) if np.any(raw_mask) else np.abs(j_raw)
        jerk_p95_eval = _p95(carla_vals)
        jerk_raw_p95_eval = _p95(raw_vals)
    else:
        jerk_p95_eval = jerk_p95
        jerk_raw_p95_eval = _p95(np.abs(j_raw))
    jerk_ratio = float(jerk_p95_eval / max(0.5, float(jerk_raw_p95_eval)))

    stop_go_osc = 0
    stop_go_moving_thresh = _env_float("PLM_STOP_GO_MOVING_THRESH_MPS", 0.45)
    moving = v_carla > float(stop_go_moving_thresh)
    for i in range(1, n):
        if bool(moving[i]) != bool(moving[i - 1]):
            stop_go_osc += 1
    stop_go_rate = float(stop_go_osc / max(1, n - 1))

    reverse_idx: List[int] = []
    reverse_min_step_m = _env_float("PLM_REVERSE_MIN_STEP_M", 0.15)
    reverse_dot_thresh = _env_float("PLM_REVERSE_DOT_THRESH", -0.15)
    reverse_raw_dot_thresh = _env_float("PLM_REVERSE_RAW_DOT_THRESH", -0.05)
    reverse_min_run = max(1, _env_int("PLM_REVERSE_MIN_RUN", 1))
    for i in range(1, n):
        dx = float(carla_xy[i, 0] - carla_xy[i - 1, 0])
        dy = float(carla_xy[i, 1] - carla_xy[i - 1, 1])
        if math.hypot(dx, dy) < float(reverse_min_step_m):
            continue
        hy = math.radians(float(carla_yaw[i]))
        dot = math.cos(hy) * dx + math.sin(hy) * dy
        hy_raw = math.radians(float(raw_yaw[i]))
        dot_raw = math.cos(hy_raw) * float(raw_xy[i, 0] - raw_xy[i - 1, 0]) + math.sin(hy_raw) * float(raw_xy[i, 1] - raw_xy[i - 1, 1])
        if dot < float(reverse_dot_thresh) and dot_raw > float(reverse_raw_dot_thresh):
            reverse_idx.append(i)
    if reverse_min_run > 1 and reverse_idx:
        filtered_reverse: List[int] = []
        s = int(reverse_idx[0])
        e = int(reverse_idx[0])
        for i in reverse_idx[1:]:
            ii = int(i)
            if ii <= e + 1:
                e = ii
            else:
                if (e - s + 1) >= int(reverse_min_run):
                    filtered_reverse.extend(range(int(s), int(e) + 1))
                s = ii
                e = ii
        if (e - s + 1) >= int(reverse_min_run):
            filtered_reverse.extend(range(int(s), int(e) + 1))
        reverse_idx = filtered_reverse

    intersection_bad_idx: List[int] = []
    intersection_heading_jump = 0.0
    intersection_window = max(1, _env_int("PLM_INTERSECTION_WINDOW_FRAMES", 3))
    intersection_max_yaw_step = _env_float("PLM_INTERSECTION_MAX_YAW_STEP_DEG", 26.0)
    intersection_max_curv_rate = _env_float("PLM_INTERSECTION_MAX_CURV_RATE", 2.2)
    intersection_max_raw_err = _env_float("PLM_INTERSECTION_MAX_RAW_ERR_M", 3.5)
    for i in ccli_transition_indices:
        lo = max(1, int(i) - int(intersection_window))
        hi = min(n - 2, int(i) + int(intersection_window))
        local_yaw_step = 0.0
        local_dk = 0.0
        for k in range(lo, hi + 1):
            local_yaw_step = max(local_yaw_step, float(_yaw_abs_diff_deg(float(carla_yaw[k]), float(carla_yaw[k - 1]))))
            local_dk = max(local_dk, abs(float(dk_carla[k])))
        intersection_heading_jump = max(intersection_heading_jump, local_yaw_step)
        if (
            local_yaw_step > float(intersection_max_yaw_step)
            or local_dk > float(intersection_max_curv_rate)
            or float(err_raw_carla[i]) > float(intersection_max_raw_err)
        ):
            intersection_bad_idx.append(int(i))

    semantic_transition_idx = _semantic_transition_indices(frames)
    pre_transition_idx = _transition_indices_from_ids(ccli_pre, min_valid_id=0)
    final_transition_idx = _transition_indices_from_ids(ccli, min_valid_id=0)
    pre_transition_union = sorted(set(int(v) for v in (list(pre_transition_idx) + list(semantic_transition_idx))))
    final_transition_union = sorted(set(int(v) for v in (list(final_transition_idx) + list(semantic_transition_idx))))

    stage_metrics: Dict[str, Dict[str, float]] = {
        "raw": _stage_metric_bundle(
            xy=raw_xy,
            yaw_deg=raw_yaw,
            raw_xy=raw_xy,
            raw_step=d_raw,
            t=t,
            line_ids=None,
            lane_ids=None,
            transition_indices=semantic_transition_idx,
            synthetic_turn_mask=synthetic_turn_mask,
        ),
        "v2": _stage_metric_bundle(
            xy=v2_xy,
            yaw_deg=v2_yaw,
            raw_xy=raw_xy,
            raw_step=d_raw,
            t=t,
            line_ids=None,
            lane_ids=semantic_lane_ids,
            transition_indices=semantic_transition_idx,
            synthetic_turn_mask=synthetic_turn_mask,
        ),
        "carla_pre": _stage_metric_bundle(
            xy=carla_pre_xy,
            yaw_deg=carla_pre_yaw,
            raw_xy=raw_xy,
            raw_step=d_raw,
            t=t,
            line_ids=ccli_pre,
            lane_ids=semantic_lane_ids,
            transition_indices=pre_transition_union,
            synthetic_turn_mask=synthetic_turn_mask,
        ),
        "carla_final": _stage_metric_bundle(
            xy=carla_xy,
            yaw_deg=carla_yaw,
            raw_xy=raw_xy,
            raw_step=d_raw,
            t=t,
            line_ids=ccli,
            lane_ids=semantic_lane_ids,
            transition_indices=final_transition_union,
            synthetic_turn_mask=synthetic_turn_mask,
        ),
    }
    final_stage = stage_metrics.get("carla_final", {})
    final_stage_low_motion_jumps = int(_safe_int(final_stage.get("low_motion_jump_events"), 0))
    final_stage_catastrophic_jumps = int(_safe_int(final_stage.get("catastrophic_jump_events"), 0))
    final_stage_line_osc = int(_safe_int(final_stage.get("line_oscillation_events"), 0))
    final_stage_lane_osc = int(_safe_int(final_stage.get("lane_oscillation_events"), 0))
    final_stage_intersection_inconsistent = int(
        _safe_int(final_stage.get("intersection_inconsistent_events"), 0)
    )
    final_stage_facing_violations = int(_safe_int(final_stage.get("facing_violation_events"), 0))
    final_stage_wrong_way = int(_safe_int(final_stage.get("wrong_way_events"), 0))

    kappa_abs = np.abs(k_carla)
    radius_min = float(1.0 / max(1e-3, float(np.max(kappa_abs)))) if kappa_abs.size > 0 else 1e9

    events: List[Tuple[str, int, float]] = []
    for i in lane_jump_indices:
        events.append(("lane_jump", int(i), float(err_raw_carla[i])))
    for i in aba_indices:
        events.append(("lane_flicker", int(i), 1.0))
    for i in wrong_way_idx:
        events.append(("wrong_way", int(i), 1.0))
    for i in intersection_bad_idx:
        events.append(("intersection_discontinuity", int(i), float(err_raw_carla[i])))
    for i in range(1, n):
        if float(d_carla[i]) >= 0.20 and abs(float(dk_carla[i])) > 2.4:
            events.append(("curvature_jitter", int(i), abs(float(dk_carla[i]))))
        if abs(float(j_carla[i])) > max(8.0, 2.8 * _p95(np.abs(j_raw))):
            events.append(("jerk_noise", int(i), abs(float(j_carla[i]))))
        if i > 0 and float(per_step_disp_err[i]) > 1.4:
            events.append(("step_divergence", int(i), float(per_step_disp_err[i])))

    segs = []
    for s, e in _merge_indices([idx for _, idx, _ in events], pad=1):
        reasons = sorted({typ for typ, idx, _ in events if s <= idx <= e})
        mags = [mag for _, idx, mag in events if s <= idx <= e]
        segs.append(
            {
                "i0": int(s),
                "i1": int(e),
                "t0": float(t[s]),
                "t1": float(t[e]),
                "reasons": reasons,
                "n_events": int(sum(1 for _, idx, _ in events if s <= idx <= e)),
                "max_mag": float(max(mags)) if mags else 0.0,
            }
        )

    lane_jump_events_final = max(int(len(lane_jump_indices)), int(final_stage_low_motion_jumps))
    wrong_way_events_final = max(int(len(wrong_way_idx)), int(final_stage_wrong_way))
    intersection_bad_events_final = max(int(len(intersection_bad_idx)), int(final_stage_intersection_inconsistent))
    catastrophic_jump_events_final = int(final_stage_catastrophic_jumps)
    hc_no_lane_jumps = int(lane_jump_events_final) == 0 and len(aba_indices) <= 1
    hc_low_variance = float(_p95(err_raw_carla)) <= 3.6 and float(np.mean(err_raw_carla)) <= 1.8
    hc_time_fidelity = nonmono == 0 and duplicates == 0
    hc_lane_semantics = (float(wrong_way_events_final) / float(n)) <= 0.03 and (lane_boundary_violations / float(n)) <= 0.10
    hc_intersection = int(intersection_bad_events_final) == 0 and intersection_heading_jump <= 26.0
    hc_similarity = float(frechet) <= 6.0 and float(haus) <= 7.5
    hc_low_noise = float(k_jitter) <= 2.2 and float(_p95(yaw_step)) <= 24.0 and float(jerk_ratio) <= 3.2
    hc_no_catastrophic = int(catastrophic_jump_events_final) == 0

    hard_ok = bool(
        hc_no_lane_jumps
        and hc_low_variance
        and hc_time_fidelity
        and hc_lane_semantics
        and hc_intersection
        and hc_similarity
        and hc_low_noise
        and hc_no_catastrophic
    )

    suspicion = (
        2.6 * min(6.0, float(lane_jump_events_final))
        + 1.2 * min(6.0, float(len(aba_indices)))
        + 1.0 * min(4.0, float(_p95(err_raw_carla) / 2.4))
        + 0.8 * min(4.0, float(frechet / 3.8))
        + 0.8 * min(4.0, float(haus / 4.8))
        + 0.8 * min(4.0, float(k_jitter / 1.5))
        + 0.7 * min(4.0, float(_p95(yaw_step) / 18.0))
        + 0.7 * min(4.0, float(jerk_ratio / 2.0))
        + 1.0 * min(6.0, float(intersection_bad_events_final))
        + 2.2 * min(6.0, float(catastrophic_jump_events_final))
        + 0.8 * min(6.0, float(final_stage_facing_violations))
        + 0.6 * min(6.0, float(stop_go_rate * 10.0))
        + 0.8 * min(6.0, float(len(reverse_idx)))
    )

    row: Dict[str, object] = {
        "scenario": str(scenario_name),
        "id": str(track.get("id")),
        "role": role,
        "n": int(n),
        "duration_s": float(max(0.0, float(t[-1] - t[0]))),
        "distance_raw_m": float(np.sum(d_raw)),
        "distance_carla_m": float(np.sum(d_carla)),
        "lane_changes": int(lane_changes),
        "lane_jump_events": int(lane_jump_events_final),
        "low_motion_jump_events": int(final_stage_low_motion_jumps),
        "catastrophic_jump_events": int(catastrophic_jump_events_final),
        "aba_flicker_events": int(len(aba_indices)),
        "line_oscillation_events": int(final_stage_line_osc),
        "lane_oscillation_events": int(final_stage_lane_osc),
        "wrong_way_events": int(wrong_way_events_final),
        "facing_violation_events": int(final_stage_facing_violations),
        "intersection_bad_events": int(intersection_bad_events_final),
        "intersection_maneuver_inconsistent_events": int(final_stage_intersection_inconsistent),
        "nonmono_timestamps": int(nonmono),
        "duplicate_timestamps": int(duplicates),
        "dt_cv": float(dt_cv),
        "raw_carla_mean_m": float(np.mean(err_raw_carla)),
        "raw_carla_p95_m": float(_p95(err_raw_carla)),
        "raw_carla_max_m": float(np.max(err_raw_carla)),
        "raw_v2_p95_m": float(_p95(err_raw_v2)),
        "v2_carla_p95_m": float(_p95(err_v2_carla)),
        "lateral_error_p95_m": float(_p95(lateral_err)),
        "heading_error_p95_deg": float(_p95(heading_err)),
        "disp_error_p95_m": float(_p95(per_step_disp_err)),
        "frechet_m": float(frechet),
        "hausdorff_m": float(haus),
        "polyline_lateral_p95_m": float(_p95(lat_to_raw)),
        "speed_rmse_vs_raw": float(np.sqrt(np.mean(np.square(v_carla - v_raw)))),
        "speed_hf_rms": float(speed_hf_rms),
        "accel_p95": float(_p95(np.abs(a_carla))),
        "jerk_p95": float(jerk_p95),
        "jerk_ratio_vs_raw": float(jerk_ratio),
        "curvature_p95": float(_p95(np.abs(k_carla))),
        "curvature_rate_p95": float(_p95(np.abs(dk_carla))),
        "curvature_jitter": float(k_jitter),
        "curvature_sign_flips": int(k_sign_flip),
        "yaw_step_p95_deg": float(_p95(yaw_step)),
        "yaw_osc_rate": float(yaw_osc / max(1, n - 1)),
        "stop_go_osc_rate": float(stop_go_rate),
        "reverse_motion_events": int(len(reverse_idx)),
        "min_turn_radius_m": float(radius_min),
        "intersection_heading_jump_deg": float(intersection_heading_jump),
        "lane_boundary_violations": int(lane_boundary_violations),
        "hard_constraints": {
            "no_lane_jumps": bool(hc_no_lane_jumps),
            "low_variance": bool(hc_low_variance),
            "time_fidelity": bool(hc_time_fidelity),
            "lane_semantics": bool(hc_lane_semantics),
            "intersection_smooth": bool(hc_intersection),
            "similarity": bool(hc_similarity),
            "low_noise": bool(hc_low_noise),
            "no_catastrophic_jumps": bool(hc_no_catastrophic),
            "all": bool(hard_ok),
        },
        "events_total": int(len(events)),
        "segments": segs,
        "stage_metrics": stage_metrics,
        "planner_suspicion_score": float(suspicion),
    }
    return TrackEval(row=row, hard_ok=hard_ok)


def _aggregate_stage_metrics(rows: List[Dict[str, object]]) -> Dict[str, object]:
    stages = ("raw", "v2", "carla_pre", "carla_final")
    out: Dict[str, object] = {}
    for stage in stages:
        stage_rows: List[Dict[str, float]] = []
        for row in rows:
            sm = row.get("stage_metrics", {})
            if not isinstance(sm, dict):
                continue
            sr = sm.get(stage)
            if isinstance(sr, dict):
                stage_rows.append(sr)  # type: ignore[arg-type]
        if not stage_rows:
            continue

        def _mean_key(key: str) -> float:
            vals = [float(_safe_float(r.get(key), 0.0)) for r in stage_rows]
            return float(np.mean(np.asarray(vals, dtype=np.float64))) if vals else 0.0

        def _sum_key(key: str) -> int:
            return int(sum(int(_safe_int(r.get(key), 0)) for r in stage_rows))

        out[str(stage)] = {
            "continuity": {
                "step_p95_mean_m": float(_mean_key("step_p95_m")),
                "accel_p95_mean": float(_mean_key("accel_p95")),
                "jerk_p95_mean": float(_mean_key("jerk_p95")),
            },
            "raw_fidelity": {
                "median_mean_m": float(_mean_key("raw_fidelity_median_m")),
                "p95_mean_m": float(_mean_key("raw_fidelity_p95_m")),
                "max_mean_m": float(_mean_key("raw_fidelity_max_m")),
            },
            "events": {
                "low_motion_jump_events": int(_sum_key("low_motion_jump_events")),
                "catastrophic_jump_events": int(_sum_key("catastrophic_jump_events")),
                "line_oscillation_events": int(_sum_key("line_oscillation_events")),
                "lane_oscillation_events": int(_sum_key("lane_oscillation_events")),
                "intersection_inconsistent_events": int(_sum_key("intersection_inconsistent_events")),
                "wrong_way_events": int(_sum_key("wrong_way_events")),
                "facing_violation_events": int(_sum_key("facing_violation_events")),
            },
        }
    return out


def _aggregate(rows: List[Dict[str, object]], hard_ok_count: int, total_tracks: int) -> Dict[str, object]:
    if not rows:
        return {
            "total_tracks": int(total_tracks),
            "analyzed_tracks": 0,
            "hard_pass_rate": 0.0,
            "hard_ok_tracks": 0,
            "stage_metrics": {},
        }

    def _vals(key: str) -> List[float]:
        return [float(_safe_float(r.get(key), 0.0)) for r in rows]

    def _count_ev(key: str) -> int:
        return int(sum(int(_safe_int(r.get(key), 0)) for r in rows))

    hard_rates: Dict[str, float] = {}
    hc_keys = [
        "no_lane_jumps",
        "low_variance",
        "time_fidelity",
        "lane_semantics",
        "intersection_smooth",
        "similarity",
        "low_noise",
        "no_catastrophic_jumps",
    ]
    if _env_int("PLM_OVERLAP_INCLUDE_IN_HARD", 0) == 1:
        hc_keys.append("no_overlap")
    for k in hc_keys:
        c = 0
        for r in rows:
            hc = r.get("hard_constraints", {})
            if isinstance(hc, dict) and bool(hc.get(k, False)):
                c += 1
        hard_rates[k] = float(c / max(1, len(rows)))

    lane_jump_total = _count_ev("lane_jump_events")
    low_motion_jump_total = _count_ev("low_motion_jump_events")
    catastrophic_jump_total = _count_ev("catastrophic_jump_events")
    aba_total = _count_ev("aba_flicker_events")
    line_osc_total = _count_ev("line_oscillation_events")
    lane_osc_total = _count_ev("lane_oscillation_events")
    wrong_way_total = _count_ev("wrong_way_events")
    facing_violation_total = _count_ev("facing_violation_events")
    intersection_bad_total = _count_ev("intersection_bad_events")
    intersection_maneuver_total = _count_ev("intersection_maneuver_inconsistent_events")
    reverse_total = _count_ev("reverse_motion_events")
    overlap_total = _count_ev("overlap_events")
    overlap_pair_total = _count_ev("overlap_pair_events")

    objective_components = {
        "lane_jump_total": float(lane_jump_total),
        "low_motion_jump_total": float(low_motion_jump_total),
        "catastrophic_jump_total": float(catastrophic_jump_total),
        "aba_flicker_total": float(aba_total),
        "line_oscillation_total": float(line_osc_total),
        "lane_oscillation_total": float(lane_osc_total),
        "wrong_way_total": float(wrong_way_total),
        "facing_violation_total": float(facing_violation_total),
        "intersection_bad_total": float(intersection_bad_total),
        "intersection_maneuver_inconsistent_total": float(intersection_maneuver_total),
        "overlap_total": float(overlap_total),
        "overlap_max_pen_mean": float(np.mean(_vals("overlap_max_penetration_m"))),
        "raw_carla_p95_mean": float(np.mean(_vals("raw_carla_p95_m"))),
        "frechet_mean": float(np.mean(_vals("frechet_m"))),
        "hausdorff_mean": float(np.mean(_vals("hausdorff_m"))),
        "curvature_jitter_mean": float(np.mean(_vals("curvature_jitter"))),
        "yaw_step_p95_mean": float(np.mean(_vals("yaw_step_p95_deg"))),
        "jerk_ratio_mean": float(np.mean(_vals("jerk_ratio_vs_raw"))),
        "stop_go_osc_mean": float(np.mean(_vals("stop_go_osc_rate"))),
        "hard_violation_tracks": float(len(rows) - int(hard_ok_count)),
    }
    w_lane_jump_total = _env_float("PLM_OBJECTIVE_W_LANE_JUMP_TOTAL", 2.8)
    w_low_motion_jump_total = _env_float("PLM_OBJECTIVE_W_LOW_MOTION_JUMP_TOTAL", 1.2)
    w_catastrophic_jump_total = _env_float("PLM_OBJECTIVE_W_CATASTROPHIC_JUMP_TOTAL", 6.0)
    w_aba_flicker_total = _env_float("PLM_OBJECTIVE_W_ABA_FLICKER_TOTAL", 1.2)
    w_line_osc_total = _env_float("PLM_OBJECTIVE_W_LINE_OSC_TOTAL", 1.4)
    w_lane_osc_total = _env_float("PLM_OBJECTIVE_W_LANE_OSC_TOTAL", 1.0)
    w_wrong_way_total = _env_float("PLM_OBJECTIVE_W_WRONG_WAY_TOTAL", 1.8)
    w_facing_violation_total = _env_float("PLM_OBJECTIVE_W_FACING_VIOLATION_TOTAL", 1.2)
    w_intersection_bad_total = _env_float("PLM_OBJECTIVE_W_INTERSECTION_BAD_TOTAL", 2.2)
    w_intersection_maneuver_total = _env_float("PLM_OBJECTIVE_W_INTERSECTION_MANEUVER_TOTAL", 1.8)
    # Overlap is intentionally very low-priority by default right now.
    w_overlap_total = _env_float("PLM_OBJECTIVE_W_OVERLAP_TOTAL", 0.0)
    w_overlap_max_pen = _env_float("PLM_OBJECTIVE_W_OVERLAP_MAX_PEN", 0.0)
    w_raw_carla_p95_mean = _env_float("PLM_OBJECTIVE_W_RAW_CARLA_P95_MEAN", 5.5)
    w_frechet_mean = _env_float("PLM_OBJECTIVE_W_FRECHET_MEAN", 3.0)
    w_hausdorff_mean = _env_float("PLM_OBJECTIVE_W_HAUSDORFF_MEAN", 2.2)
    w_curvature_jitter_mean = _env_float("PLM_OBJECTIVE_W_CURVATURE_JITTER_MEAN", 3.5)
    w_yaw_step_p95_mean = _env_float("PLM_OBJECTIVE_W_YAW_STEP_P95_MEAN", 0.8)
    w_jerk_ratio_mean = _env_float("PLM_OBJECTIVE_W_JERK_RATIO_MEAN", 1.4)
    w_stop_go_osc_mean = _env_float("PLM_OBJECTIVE_W_STOP_GO_OSC_MEAN", 12.0)
    w_hard_violation_tracks = _env_float("PLM_OBJECTIVE_W_HARD_VIOLATION_TRACKS", 4.0)
    objective_score = (
        float(w_lane_jump_total) * objective_components["lane_jump_total"]
        + float(w_low_motion_jump_total) * objective_components["low_motion_jump_total"]
        + float(w_catastrophic_jump_total) * objective_components["catastrophic_jump_total"]
        + float(w_aba_flicker_total) * objective_components["aba_flicker_total"]
        + float(w_line_osc_total) * objective_components["line_oscillation_total"]
        + float(w_lane_osc_total) * objective_components["lane_oscillation_total"]
        + float(w_wrong_way_total) * objective_components["wrong_way_total"]
        + float(w_facing_violation_total) * objective_components["facing_violation_total"]
        + float(w_intersection_bad_total) * objective_components["intersection_bad_total"]
        + float(w_intersection_maneuver_total) * objective_components["intersection_maneuver_inconsistent_total"]
        + float(w_overlap_total) * objective_components["overlap_total"]
        + float(w_overlap_max_pen) * objective_components["overlap_max_pen_mean"]
        + float(w_raw_carla_p95_mean) * objective_components["raw_carla_p95_mean"]
        + float(w_frechet_mean) * objective_components["frechet_mean"]
        + float(w_hausdorff_mean) * objective_components["hausdorff_mean"]
        + float(w_curvature_jitter_mean) * objective_components["curvature_jitter_mean"]
        + float(w_yaw_step_p95_mean) * objective_components["yaw_step_p95_mean"]
        + float(w_jerk_ratio_mean) * objective_components["jerk_ratio_mean"]
        + float(w_stop_go_osc_mean) * objective_components["stop_go_osc_mean"]
        + float(w_hard_violation_tracks) * objective_components["hard_violation_tracks"]
    )

    return {
        "total_tracks": int(total_tracks),
        "analyzed_tracks": int(len(rows)),
        "hard_ok_tracks": int(hard_ok_count),
        "hard_pass_rate": float(hard_ok_count / max(1, len(rows))),
        "hard_constraint_rates": hard_rates,
        "event_totals": {
            "lane_jump_events": int(lane_jump_total),
            "low_motion_jump_events": int(low_motion_jump_total),
            "catastrophic_jump_events": int(catastrophic_jump_total),
            "aba_flicker_events": int(aba_total),
            "line_oscillation_events": int(line_osc_total),
            "lane_oscillation_events": int(lane_osc_total),
            "wrong_way_events": int(wrong_way_total),
            "facing_violation_events": int(facing_violation_total),
            "intersection_bad_events": int(intersection_bad_total),
            "intersection_maneuver_inconsistent_events": int(intersection_maneuver_total),
            "reverse_motion_events": int(reverse_total),
            "overlap_events": int(overlap_total),
            "overlap_pair_events": int(overlap_pair_total),
        },
        "global_metrics": {
            "raw_carla_mean_mean_m": float(np.mean(_vals("raw_carla_mean_m"))),
            "raw_carla_p95_mean_m": float(np.mean(_vals("raw_carla_p95_m"))),
            "raw_carla_p95_p90_m": float(np.percentile(np.asarray(_vals("raw_carla_p95_m"), dtype=np.float64), 90.0)),
            "frechet_mean_m": float(np.mean(_vals("frechet_m"))),
            "frechet_p90_m": float(np.percentile(np.asarray(_vals("frechet_m"), dtype=np.float64), 90.0)),
            "hausdorff_mean_m": float(np.mean(_vals("hausdorff_m"))),
            "curvature_jitter_mean": float(np.mean(_vals("curvature_jitter"))),
            "yaw_step_p95_mean_deg": float(np.mean(_vals("yaw_step_p95_deg"))),
            "jerk_ratio_mean": float(np.mean(_vals("jerk_ratio_vs_raw"))),
            "stop_go_osc_rate_mean": float(np.mean(_vals("stop_go_osc_rate"))),
            "overlap_max_pen_mean_m": float(np.mean(_vals("overlap_max_penetration_m"))),
            "overlap_max_pen_p95_m": float(_p95(_vals("overlap_max_penetration_m"))),
            "timestamp_nonmono_tracks": int(sum(1 for r in rows if _safe_int(r.get("nonmono_timestamps"), 0) > 0)),
            "timestamp_duplicate_tracks": int(sum(1 for r in rows if _safe_int(r.get("duplicate_timestamps"), 0) > 0)),
        },
        "stage_metrics": _aggregate_stage_metrics(rows),
        "objective": {
            "score": float(objective_score),
            "components": objective_components,
            "weights": {
                "lane_jump_total": float(w_lane_jump_total),
                "low_motion_jump_total": float(w_low_motion_jump_total),
                "catastrophic_jump_total": float(w_catastrophic_jump_total),
                "aba_flicker_total": float(w_aba_flicker_total),
                "line_oscillation_total": float(w_line_osc_total),
                "lane_oscillation_total": float(w_lane_osc_total),
                "wrong_way_total": float(w_wrong_way_total),
                "facing_violation_total": float(w_facing_violation_total),
                "intersection_bad_total": float(w_intersection_bad_total),
                "intersection_maneuver_inconsistent_total": float(w_intersection_maneuver_total),
                "overlap_total": float(w_overlap_total),
                "overlap_max_pen_mean": float(w_overlap_max_pen),
                "raw_carla_p95_mean": float(w_raw_carla_p95_mean),
                "frechet_mean": float(w_frechet_mean),
                "hausdorff_mean": float(w_hausdorff_mean),
                "curvature_jitter_mean": float(w_curvature_jitter_mean),
                "yaw_step_p95_mean": float(w_yaw_step_p95_mean),
                "jerk_ratio_mean": float(w_jerk_ratio_mean),
                "stop_go_osc_mean": float(w_stop_go_osc_mean),
                "hard_violation_tracks": float(w_hard_violation_tracks),
            },
        },
    }


def _apply_metric_profile(profile: str) -> Dict[str, str]:
    p = str(profile or "current").strip().lower()
    if p in {"default", "latest"}:
        p = "current"
    if p == "current":
        return {}
    if p != "overlap_check_compat":
        raise ValueError(f"Unknown metric profile: {profile}")

    # Compatibility profile to recover the historical overlap-checkpoint
    # detector behavior used during previous optimization loops.
    overrides: Dict[str, str] = {
        "PLM_OVERLAP_SKIP_LOW_SPEED_PAIRS": "0",
        "PLM_OVERLAP_RAW_FILTER_ENABLED": "0",
        "PLM_OVERLAP_MIN_PEN_M": "0.04",
        "PLM_LANE_JUMP_RAW_STEP_MAX_M": "0.50",
        "PLM_LANE_JUMP_CARLA_ABS_MIN_M": "2.2",
        "PLM_LANE_JUMP_RATIO": "4.2",
        "PLM_LANE_JUMP_COUNT_ABA": "0",
        "PLM_ABA_ENABLE": "0",
        "PLM_WRONG_WAY_YAW_DELTA_MIN_DEG": "181",
        "PLM_WRONG_WAY_MIN_RUN": "8",
        "PLM_INTERSECTION_MAX_YAW_STEP_DEG": "95",
        "PLM_INTERSECTION_MAX_CURV_RATE": "14.0",
        "PLM_INTERSECTION_MAX_RAW_ERR_M": "11.0",
        "PLM_DERIV_DT_MIN_S": "0.10",
        "PLM_SPEED_DT_MIN_S": "0.10",
        "PLM_CURV_SEG_MIN_M": "0.20",
        "PLM_JERK_SMOOTH_WIN": "9",
        "PLM_JERK_MIN_SPEED_MPS": "1.00",
        "PLM_STOP_GO_MOVING_THRESH_MPS": "0.50",
        "PLM_REVERSE_MIN_STEP_M": "0.20",
        "PLM_REVERSE_DOT_THRESH": "-0.20",
        "PLM_REVERSE_RAW_DOT_THRESH": "-0.08",
        "PLM_REVERSE_MIN_RUN": "2",
    }
    for k, v in overrides.items():
        os.environ[str(k)] = str(v)
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Planner-likeness metrics for trajectory HTML outputs.")
    parser.add_argument("html", type=Path, help="Path to trajectory_plot.html or trajectories_multi.html")
    parser.add_argument(
        "--roles",
        type=str,
        default="ego,vehicle",
        help="Comma-separated roles to include (default: ego,vehicle)",
    )
    parser.add_argument("--min-frames", type=int, default=12, help="Minimum track length to analyze")
    parser.add_argument("--top-k", type=int, default=120, help="Top suspicious rows to keep")
    parser.add_argument(
        "--metric-profile",
        type=str,
        choices=("current", "overlap_check_compat"),
        default="current",
        help="Metric detector profile (current defaults or historical compatibility mode).",
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/planner_likeness_report.json"), help="Output JSON path")
    args = parser.parse_args()

    html_path = args.html.expanduser().resolve()
    if not html_path.exists():
        raise SystemExit(f"HTML file not found: {html_path}")

    include_roles = [r.strip() for r in str(args.roles).split(",") if r.strip()]
    if not include_roles:
        include_roles = ["ego", "vehicle"]

    profile_overrides = _apply_metric_profile(str(args.metric_profile))

    data = _extract_dataset_json(html_path.read_text(encoding="utf-8"))
    scenarios = _to_scenarios(data)

    rows: List[Dict[str, object]] = []
    total_tracks = 0
    hard_ok = 0
    scenario_rows: Dict[str, List[Dict[str, object]]] = {}
    scenario_overlap: Dict[str, Dict[str, object]] = {}

    for sc in scenarios:
        sc_name = str(sc.get("scenario_name", "scenario"))
        tracks = sc.get("tracks", [])
        if not isinstance(tracks, list):
            continue
        total_tracks += int(len(tracks))
        sc_rows = scenario_rows.setdefault(sc_name, [])
        for tr in tracks:
            if not isinstance(tr, dict):
                continue
            ev = _analyze_track(
                scenario_name=sc_name,
                track=tr,
                min_frames=int(args.min_frames),
                include_roles=include_roles,
            )
            if ev is None:
                continue
            rows.append(ev.row)
            sc_rows.append(ev.row)
        if sc_rows:
            scenario_overlap[sc_name] = _annotate_scenario_overlaps(
                scenario_name=sc_name,
                tracks=tracks,
                rows=sc_rows,
            )

    # Recompute hard-pass after scenario-level overlap annotation updates.
    hard_ok = int(
        sum(
            1
            for r in rows
            if bool((r.get("hard_constraints") or {}).get("all", False))
        )
    )

    rows_sorted = sorted(
        rows,
        key=lambda r: float(_safe_float(r.get("planner_suspicion_score"), 0.0)),
        reverse=True,
    )

    by_scenario: Dict[str, object] = {}
    for sc_name, sc_rows in scenario_rows.items():
        sc_hard = int(sum(1 for r in sc_rows if bool((r.get("hard_constraints") or {}).get("all", False))))
        sc_agg = _aggregate(sc_rows, hard_ok_count=sc_hard, total_tracks=len(sc_rows))
        sc_agg["overlap"] = scenario_overlap.get(
            sc_name,
            {
                "scenario": str(sc_name),
                "overlap_events": 0,
                "overlap_pair_events": 0,
                "max_penetration_m": 0.0,
                "tracks_with_overlap": 0,
            },
        )
        by_scenario[sc_name] = sc_agg

    report = {
        "source_html": str(html_path),
        "metric_profile": str(args.metric_profile),
        "metric_profile_env": dict(profile_overrides),
        "tracks_analyzed": int(len(rows)),
        "summary": _aggregate(rows, hard_ok_count=hard_ok, total_tracks=total_tracks),
        "scenario_summary": by_scenario,
        "rows_sorted": rows_sorted[: max(1, int(args.top_k))],
    }

    out_path = args.output.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summ = report.get("summary", {})
    objective = ((summ.get("objective", {}) if isinstance(summ, dict) else {}) if isinstance(summ, dict) else {})
    g = summ.get("global_metrics", {}) if isinstance(summ, dict) else {}
    e = summ.get("event_totals", {}) if isinstance(summ, dict) else {}
    print(
        "[METRICS] tracks={} hard_pass={:.3f} objective={:.3f} lane_jumps={} cat_jumps={} wrong_way={} "
        "intersection_bad={} overlap_events={} overlap_pair_events={} raw_carla_p95_mean={:.3f} frechet_p90={:.3f}".format(
            int(summ.get("analyzed_tracks", 0)) if isinstance(summ, dict) else 0,
            float(_safe_float(summ.get("hard_pass_rate"), 0.0)) if isinstance(summ, dict) else 0.0,
            float(_safe_float(objective.get("score"), 0.0)) if isinstance(objective, dict) else 0.0,
            int(_safe_int(e.get("lane_jump_events"), 0)) if isinstance(e, dict) else 0,
            int(_safe_int(e.get("catastrophic_jump_events"), 0)) if isinstance(e, dict) else 0,
            int(_safe_int(e.get("wrong_way_events"), 0)) if isinstance(e, dict) else 0,
            int(_safe_int(e.get("intersection_bad_events"), 0)) if isinstance(e, dict) else 0,
            int(_safe_int(e.get("overlap_events"), 0)) if isinstance(e, dict) else 0,
            int(_safe_int(e.get("overlap_pair_events"), 0)) if isinstance(e, dict) else 0,
            float(_safe_float(g.get("raw_carla_p95_mean_m"), 0.0)) if isinstance(g, dict) else 0.0,
            float(_safe_float(g.get("frechet_p90_m"), 0.0)) if isinstance(g, dict) else 0.0,
        )
    )
    print(f"[OK] Wrote metrics report: {out_path}")


if __name__ == "__main__":
    main()
