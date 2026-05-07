"""V2XPNP trajectory evaluation framework.

Computes 30+ actor-level + scenario-level metrics on a converted V2X-PnP-real
scenario, with explicit per-metric pass/fail thresholds, so we can answer
"which specific actors fail, and on which axis?" instead of just a global
score.

This is an offline tool — it consumes the embedded JSON dataset from a
trajectory_plot HTML file (or a list of them), or accepts a dataset dict
directly.

Usage (single HTML):
    python3 -m tools.v2xpnp_eval_framework \
        --html /tmp/run/scenario_x.html \
        --out  /tmp/run/scenario_x.eval.json

Usage (multiple, by scenario root):
    python3 -m tools.v2xpnp_eval_framework \
        --html /tmp/run/2023-03-17-15-53-02_1_0.html \
               /tmp/run/2023-03-17-16-10-12_1_1.html \
               /tmp/run/2023-03-17-16-11-12_2_0.html \
        --out-dir /tmp/eval_run

Output:
    Per scenario: <out>/<scenario>.eval.json (full metrics + failures)
                  <out>/<scenario>.eval.txt  (human-readable report)
    Across:       <out>/eval_summary.json
                  <out>/eval_summary.txt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Pass / fail thresholds. Set with deliberate margins; tighten as the pipeline
# improves. Each metric has an optional lower / upper bound; the actor "fails"
# the metric if the value is outside that bound.
# ---------------------------------------------------------------------------
THRESHOLDS: Dict[str, Tuple[Optional[float], Optional[float], str]] = {
    # name: (min_ok, max_ok, brief)
    # Tuned 2026-05-05 after fixing OBB-length bug. Aim: catch true pipeline
    # bugs, tolerate isolated single-frame snap-discontinuity spikes that
    # CARLA replay handles fine.
    # Kinematic thresholds calibrated against the existing reference scenarioset
    # (which the user has already accepted as functional in CARLA replay).
    # Reference NPCs average ~65 jerk-over-5 frames each; we set thresholds that
    # tolerate the V2X-PnP data quality without flagging every NPC as broken.
    "n_jumps_over_2m":            (None, 1,    "frames with >2m position jump"),
    "n_jumps_over_1m":            (None, 8,    "frames with >1m position jump"),
    "max_xy_jump_m":              (None, 3.5,  "single largest position jump"),
    "mean_jerk":                  (None, 0.5,  "mean jerk (legacy units, very loose)"),
    "n_jerk_over_5_mps3":         (None, 80,   "frames with jerk > 5 m/s^3 (V2X-PnP raw is noisy)"),
    "n_accel_over_6_mps2":        (None, 60,   "frames with acceleration > 6 m/s^2"),
    "max_speed_m_s":              (None, 35.0, "max speed (~125 km/h)"),
    "max_yaw_step_deg":           (None, 60.0, "max single-frame yaw change"),
    "max_yaw_sliding_deg":        (None, 130.0, "max yaw change over 5-frame window"),
    "n_yaw_jumps":                (None, 8,    "frames with yaw step > 30 deg"),

    "pct_on_lane_1m":             (0.80, None, "fraction of frames within 1m of lane"),
    "pct_on_lane_05m":            (0.30, None, "fraction of frames within 0.5m of lane"),
    "max_lane_lat_offset_m":      (None, 3.0,  "max lateral distance to nearest lane"),
    "mean_lane_lat_offset_m":     (None, 1.0,  "mean lateral distance to nearest lane"),
    "n_off_route_frames":         (None, 0,    "frames > 3m from any lane"),
    "n_wrongway_frames":          (None, 0,    "frames moving against lane direction"),
    "pct_lane_aligned":           (0.75, None, "fraction of moving frames whose heading aligns with nearest-lane heading"),
    "n_lane_changes":             (None, 8,    "CCLI-change count (excessive churn)"),

    # polyline_follow_err measures distance to ASSIGNED ccli's polyline. If the
    # ccli got mis-stamped by an internal stabilization pass, this metric flags
    # an internal inconsistency, not a CARLA-visible problem (cx/cy may still be
    # on a valid drivable lane — see max_lane_lat_offset_m). Keep as informational.
    "polyline_follow_err_mean_m": (None, 3.0,  "mean distance from snapped point to assigned-ccli polyline (informational)"),
    "polyline_follow_err_max_m":  (None, 30.0, "max distance from snapped point to assigned-ccli polyline (informational)"),

    "raw_aligned_max_lat_m":      (None, 5.0,  "max lateral deviation snap-vs-raw"),
    "raw_aligned_mean_lat_m":     (None, 2.0,  "mean lateral deviation snap-vs-raw"),
    "snap_lateral_excess_m":      (None, 2.5,  "snap zigzags more laterally than raw"),

    "spawn_lane_offset_m":        (None, 2.0,  "first-frame distance from nearest lane"),
    "spawn_lane_yaw_diff_deg":    (None, 35.0, "first-frame heading vs nearest-lane heading"),
    "spawn_collision":            (None, 0,    "first frame OBB-collides with another actor"),
    "n_collisions_vs_moving":     (None, 5,    "frames where this actor's OBB intersects another MOVING actor (tolerance for following-traffic touch)"),

    "intersection_connector_valid": (1.0, None, "1.0 if all intersections used valid connectors"),

    "monotonicity_violations":    (None, 25,   "frames where actor moves backward along lane (V2X-PnP raw has noise)"),
    "n_outside_map_bbox":         (None, 0,    "frames outside the dataset's xy bbox"),
}


# ---------------------------------------------------------------------------
# Tiny numeric utilities
# ---------------------------------------------------------------------------
def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except Exception:
        return default


def _safe_int(v: Any, default: int = -1) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _pct(num: float, denom: float) -> float:
    return float(num) / float(denom) if denom > 0 else 0.0


def _percentile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return float(s[idx])


def _wrap_180(d: float) -> float:
    return ((d + 540.0) % 360.0) - 180.0


def _yaw_diff_deg(a: float, b: float) -> float:
    return abs(_wrap_180(b - a))


# ---------------------------------------------------------------------------
# Lane / polyline geometry
# ---------------------------------------------------------------------------
@dataclass
class LaneSeg:
    x0: float; y0: float; x1: float; y1: float
    sdx: float; sdy: float; L2: float
    line_idx: int
    heading_deg: float


def _build_lane_segs(dataset: Dict[str, Any]) -> List[LaneSeg]:
    segs: List[LaneSeg] = []
    lines = (dataset.get("carla_map") or {}).get("lines") or []
    for line in lines:
        idx = _safe_int(line.get("index"), -1)
        pts = line.get("polyline") or []
        for i in range(len(pts) - 1):
            try:
                x0, y0 = float(pts[i][0]), float(pts[i][1])
                x1, y1 = float(pts[i + 1][0]), float(pts[i + 1][1])
            except Exception:
                continue
            sdx = x1 - x0
            sdy = y1 - y0
            L2 = sdx * sdx + sdy * sdy
            if L2 < 1e-9:
                continue
            segs.append(LaneSeg(
                x0=x0, y0=y0, x1=x1, y1=y1,
                sdx=sdx, sdy=sdy, L2=L2,
                line_idx=idx,
                heading_deg=math.degrees(math.atan2(sdy, sdx)),
            ))
    return segs


def _project_to_seg(s: LaneSeg, px: float, py: float) -> Tuple[float, float, float, float]:
    """Return (cx, cy, t in [0,1], dist) — closest point on the segment."""
    t = ((px - s.x0) * s.sdx + (py - s.y0) * s.sdy) / s.L2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    cx = s.x0 + t * s.sdx
    cy = s.y0 + t * s.sdy
    return cx, cy, t, math.hypot(cx - px, cy - py)


def _nearest_lane(segs: List[LaneSeg], px: float, py: float) -> Optional[Tuple[float, float, int]]:
    """Return (min_dist, lane_heading_deg, line_idx) of the nearest lane segment."""
    if not segs:
        return None
    best_d = float("inf")
    best_h = 0.0
    best_idx = -1
    for s in segs:
        _, _, _, d = _project_to_seg(s, px, py)
        if d < best_d:
            best_d = d
            best_h = s.heading_deg
            best_idx = s.line_idx
    return (float(best_d), float(best_h), int(best_idx))


def _ccli_polyline(dataset: Dict[str, Any], line_idx: int) -> List[Tuple[float, float]]:
    lines = (dataset.get("carla_map") or {}).get("lines") or []
    for line in lines:
        if _safe_int(line.get("index"), -2) == line_idx:
            return [(float(p[0]), float(p[1])) for p in (line.get("polyline") or []) if len(p) >= 2]
    return []


def _dist_to_polyline(poly: List[Tuple[float, float]], px: float, py: float) -> float:
    if len(poly) < 2:
        return float("inf")
    best = float("inf")
    for i in range(len(poly) - 1):
        x0, y0 = poly[i]
        x1, y1 = poly[i + 1]
        sdx = x1 - x0
        sdy = y1 - y0
        L2 = sdx * sdx + sdy * sdy
        if L2 < 1e-9:
            continue
        t = ((px - x0) * sdx + (py - y0) * sdy) / L2
        t = max(0.0, min(1.0, t))
        cx = x0 + t * sdx
        cy = y0 + t * sdy
        d = math.hypot(cx - px, cy - py)
        if d < best:
            best = d
    return best


def _polyline_heading_at(poly: List[Tuple[float, float]], px: float, py: float) -> float:
    if len(poly) < 2:
        return 0.0
    best_h = 0.0
    best_d = float("inf")
    for i in range(len(poly) - 1):
        x0, y0 = poly[i]
        x1, y1 = poly[i + 1]
        sdx = x1 - x0
        sdy = y1 - y0
        L2 = sdx * sdx + sdy * sdy
        if L2 < 1e-9:
            continue
        t = max(0.0, min(1.0, ((px - x0) * sdx + (py - y0) * sdy) / L2))
        cx = x0 + t * sdx
        cy = y0 + t * sdy
        d = math.hypot(cx - px, cy - py)
        if d < best_d:
            best_d = d
            best_h = math.degrees(math.atan2(sdy, sdx))
    return best_h


# ---------------------------------------------------------------------------
# OBB collision
# ---------------------------------------------------------------------------
def _obb_corners(cx: float, cy: float, yaw_deg: float, length: float, width: float) -> List[Tuple[float, float]]:
    cyaw = math.cos(math.radians(yaw_deg))
    syaw = math.sin(math.radians(yaw_deg))
    hl = length * 0.5
    hw = width * 0.5
    corners_local = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
    return [
        (cx + cl * cyaw - cw * syaw, cy + cl * syaw + cw * cyaw)
        for (cl, cw) in corners_local
    ]


def _obb_overlap(a: List[Tuple[float, float]], b: List[Tuple[float, float]]) -> bool:
    """SAT (Separating Axis Theorem) for two convex polygons."""
    for poly in (a, b):
        n = len(poly)
        for i in range(n):
            x0, y0 = poly[i]
            x1, y1 = poly[(i + 1) % n]
            ex = x1 - x0
            ey = y1 - y0
            # Normal axis (perpendicular to edge)
            nx, ny = -ey, ex
            mag = math.hypot(nx, ny)
            if mag < 1e-9:
                continue
            nx /= mag
            ny /= mag
            a_proj = [px * nx + py * ny for (px, py) in a]
            b_proj = [px * nx + py * ny for (px, py) in b]
            if max(a_proj) < min(b_proj) - 1e-6 or max(b_proj) < min(a_proj) - 1e-6:
                return False
    return True


def _obb_distance(a: List[Tuple[float, float]], b: List[Tuple[float, float]]) -> float:
    """Approximate min distance between two OBBs by sampling vertices and edges.
    Returns 0.0 if overlapping. Approximation is conservative for our use."""
    if _obb_overlap(a, b):
        return 0.0
    best = float("inf")
    # vertex-to-vertex
    for ax, ay in a:
        for bx, by in b:
            d = math.hypot(ax - bx, ay - by)
            if d < best:
                best = d
    return float(best)


def _obb_penetration(a: List[Tuple[float, float]], b: List[Tuple[float, float]]) -> float:
    """Min penetration depth using SAT. 0.0 if not overlapping."""
    if not _obb_overlap(a, b):
        return 0.0
    min_pen = float("inf")
    for poly in (a, b):
        n = len(poly)
        for i in range(n):
            x0, y0 = poly[i]
            x1, y1 = poly[(i + 1) % n]
            ex = x1 - x0
            ey = y1 - y0
            nx, ny = -ey, ex
            mag = math.hypot(nx, ny)
            if mag < 1e-9:
                continue
            nx /= mag; ny /= mag
            a_proj = [px * nx + py * ny for (px, py) in a]
            b_proj = [px * nx + py * ny for (px, py) in b]
            overlap = min(max(a_proj), max(b_proj)) - max(min(a_proj), min(b_proj))
            if overlap < min_pen:
                min_pen = overlap
    return max(0.0, float(min_pen if min_pen != float("inf") else 0.0))


# ---------------------------------------------------------------------------
# Per-actor metric computation
# ---------------------------------------------------------------------------
@dataclass
class ActorMetrics:
    track_id: str
    role: str
    n_frames: int

    # Movement classification
    total_raw_displacement_m: float = 0.0
    raw_path_length_m: float = 0.0
    is_static: int = 0  # 1 if this is a parked vehicle (raw total motion < 2 m)

    # Geometry quality
    mean_speed_m_s: float = 0.0
    max_speed_m_s: float = 0.0
    mean_accel_mag: float = 0.0
    max_accel_mag: float = 0.0
    mean_jerk: float = 0.0
    p95_jerk: float = 0.0
    max_jerk: float = 0.0
    n_accel_over_6_mps2: int = 0   # per-frame count of |a| > 6 m/s^2
    n_jerk_over_5_mps3: int = 0    # per-frame count of |j| > 5 m/s^3
    max_xy_jump_m: float = 0.0
    n_jumps_over_05m: int = 0
    n_jumps_over_1m: int = 0
    n_jumps_over_2m: int = 0
    max_yaw_step_deg: float = 0.0
    max_yaw_sliding_deg: float = 0.0
    n_yaw_jumps: int = 0
    curvature_std: float = 0.0
    n_lane_changes: int = 0
    n_ccli_runs: int = 1

    # Lane / map
    pct_on_lane_1m: float = 0.0
    pct_on_lane_05m: float = 0.0
    mean_lane_lat_offset_m: float = 0.0
    max_lane_lat_offset_m: float = 0.0
    n_off_route_frames: int = 0
    n_wrongway_frames: int = 0
    pct_lane_aligned: float = 0.0
    monotonicity_violations: int = 0

    # Polyline-following error vs the actor's own current CCLI
    polyline_follow_err_mean_m: float = 0.0
    polyline_follow_err_max_m: float = 0.0

    # Raw vs aligned
    raw_aligned_mean_lat_m: float = 0.0
    raw_aligned_max_lat_m: float = 0.0
    snap_lateral_excess_m: float = 0.0

    # Spawn / boundary
    spawn_lane_offset_m: float = 0.0
    spawn_lane_yaw_diff_deg: float = 0.0
    spawn_collision: int = 0
    n_outside_map_bbox: int = 0

    # Inter-actor
    n_inter_actor_collisions: int = 0   # total collision-frames (with any sibling)
    n_collisions_vs_moving: int = 0
    n_collisions_vs_static: int = 0
    min_inter_actor_dist_m: float = float("inf")
    spawn_collision_partners: List[str] = field(default_factory=list)
    n_alignment_eval_frames: int = 0  # if 0, pct_lane_aligned is not meaningful

    # Intersection
    intersection_connector_valid: float = 1.0  # 1 = all valid

    # Failure summary
    failures: List[str] = field(default_factory=list)
    score: float = 0.0


def _track_kind(trk: Dict[str, Any]) -> str:
    role = str(trk.get("role", "")).strip().lower()
    if role in ("ego", "vehicle"):
        return "vehicle"
    return role or "other"


def _frames_xy(frames: List[Dict[str, Any]], use: str = "snap") -> List[Tuple[float, float]]:
    """use='snap' -> (cx,cy); use='raw' -> (x,y)."""
    out = []
    for f in frames:
        if use == "snap":
            x = _safe_float(f.get("cx"))
            y = _safe_float(f.get("cy"))
        else:
            x = _safe_float(f.get("x"))
            y = _safe_float(f.get("y"))
        out.append((x, y))
    return out


def _frames_yaw(frames: List[Dict[str, Any]], use: str = "snap") -> List[float]:
    out = []
    for f in frames:
        if use == "snap":
            y = _safe_float(f.get("cyaw"), default=_safe_float(f.get("yaw")))
        else:
            y = _safe_float(f.get("yaw"))
        out.append(y)
    return out


def _compute_actor(
    trk: Dict[str, Any],
    dataset: Dict[str, Any],
    segs: List[LaneSeg],
    bbox: Optional[Tuple[float, float, float, float]],
    other_tracks: List[Dict[str, Any]],
    fps: float,
) -> Optional[ActorMetrics]:
    role = _track_kind(trk)
    if role != "vehicle":
        return None
    frames = trk.get("frames") or []
    if len(frames) < 4:
        return None
    n = len(frames)
    L = _safe_float(trk.get("length"), 4.5)
    W = _safe_float(trk.get("width"), 2.0)

    m = ActorMetrics(
        track_id=str(trk.get("id", "?")),
        role=role,
        n_frames=n,
    )

    # ---- Geometry / kinematics on snapped trajectory ----
    sxy = _frames_xy(frames, "snap")
    syaw = _frames_yaw(frames, "snap")
    rxy = _frames_xy(frames, "raw")
    ryaw = _frames_yaw(frames, "raw")
    cclis = [_safe_int(f.get("ccli"), -1) for f in frames]

    # Movement classification — first prefer pipeline's own tag, then fall back
    # to raw displacement.
    raw_finite = [(rx, ry) for rx, ry in rxy if math.isfinite(rx) and math.isfinite(ry)]
    if len(raw_finite) >= 2:
        m.total_raw_displacement_m = math.hypot(
            raw_finite[-1][0] - raw_finite[0][0],
            raw_finite[-1][1] - raw_finite[0][1],
        )
        path = 0.0
        for i in range(1, len(raw_finite)):
            dx = raw_finite[i][0] - raw_finite[i - 1][0]
            dy = raw_finite[i][1] - raw_finite[i - 1][1]
            path += math.hypot(dx, dy)
        m.raw_path_length_m = path
    pipeline_low_motion = bool(trk.get("low_motion_vehicle", False))
    # Also consider parked if a high fraction of frames have parked-style csource
    parked_csources = {
        "parked_invariant_centerline_static",
        "parked_invariant_static",
        "parked_static",
    }
    n_parked_csource = 0
    for f in frames:
        cs = str(f.get("csource") or "")
        if cs in parked_csources:
            n_parked_csource += 1
    pct_parked_csource = _pct(n_parked_csource, len(frames))
    # Static if any of these hold:
    #   - pipeline tagged low_motion_vehicle
    #   - >50% of frames have parked-static csource
    #   - net displacement < 2 m
    m.is_static = 1 if (
        pipeline_low_motion
        or pct_parked_csource > 0.5
        or m.total_raw_displacement_m < 2.0
    ) else 0

    n_runs = 1
    n_lane_changes = 0
    for i in range(1, n):
        if cclis[i] != cclis[i - 1]:
            n_runs += 1
            n_lane_changes += 1
    m.n_ccli_runs = n_runs
    m.n_lane_changes = n_lane_changes

    speeds: List[float] = []
    accels: List[float] = []
    jerks: List[float] = []
    yaw_steps: List[float] = []
    curvatures: List[float] = []
    monot_viol = 0

    last_v = (0.0, 0.0)
    last_a = (0.0, 0.0)
    have_last_v = False
    have_last_a = False
    dt = 1.0 / max(fps, 1.0)
    n_accel6 = 0
    n_jerk5 = 0
    for i in range(1, n):
        x0, y0 = sxy[i - 1]
        x1, y1 = sxy[i]
        if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
            continue
        dx = x1 - x0
        dy = y1 - y0
        step = math.hypot(dx, dy)
        if step > m.max_xy_jump_m:
            m.max_xy_jump_m = step
        if step > 0.5:
            m.n_jumps_over_05m += 1
        if step > 1.0:
            m.n_jumps_over_1m += 1
        if step > 2.0:
            m.n_jumps_over_2m += 1
        v = (dx / dt, dy / dt)
        spd = math.hypot(*v)
        speeds.append(spd)
        if have_last_v:
            ax = (v[0] - last_v[0]) / dt
            ay = (v[1] - last_v[1]) / dt
            am = math.hypot(ax, ay)
            accels.append(am)
            if am > 6.0:
                n_accel6 += 1
            if have_last_a:
                jx = (ax - last_a[0]) / dt
                jy = (ay - last_a[1]) / dt
                jm_phys = math.hypot(jx, jy)         # physical m/s^3
                if jm_phys > 5.0:
                    n_jerk5 += 1
                jerks.append(jm_phys * dt * dt)      # kept comparable to original 3rd-diff units
            last_a = (ax, ay)
            have_last_a = True
        last_v = v
        have_last_v = True

        # yaw step
        if math.isfinite(syaw[i]) and math.isfinite(syaw[i - 1]):
            ys = abs(_wrap_180(syaw[i] - syaw[i - 1]))
            yaw_steps.append(ys)
            if ys > m.max_yaw_step_deg:
                m.max_yaw_step_deg = ys
            if ys > 30.0:
                m.n_yaw_jumps += 1
    m.n_accel_over_6_mps2 = n_accel6
    m.n_jerk_over_5_mps3 = n_jerk5

    # Curvature: angular change between consecutive headings normalized by step length.
    for i in range(2, n):
        x0, y0 = sxy[i - 2]; x1, y1 = sxy[i - 1]; x2, y2 = sxy[i]
        if not all(math.isfinite(v) for v in (x0, y0, x1, y1, x2, y2)):
            continue
        h0 = math.degrees(math.atan2(y1 - y0, x1 - x0))
        h1 = math.degrees(math.atan2(y2 - y1, x2 - x1))
        dh = abs(_wrap_180(h1 - h0))
        d = math.hypot(x2 - x1, y2 - y1)
        if d > 0.05:
            curvatures.append(dh / d)
    if len(curvatures) >= 2:
        mean_c = sum(curvatures) / len(curvatures)
        m.curvature_std = math.sqrt(sum((c - mean_c) ** 2 for c in curvatures) / len(curvatures))

    # Sliding-window yaw (5 frames)
    win = 5
    for i in range(win, n):
        x_a, y_a = sxy[i - win]
        x_b, y_b = sxy[i]
        if not all(math.isfinite(v) for v in (x_a, y_a, x_b, y_b)):
            continue
        dx0 = sxy[i - win + 1][0] - x_a
        dy0 = sxy[i - win + 1][1] - y_a
        dx1 = x_b - sxy[i - 1][0]
        dy1 = y_b - sxy[i - 1][1]
        if math.hypot(dx0, dy0) < 0.1 or math.hypot(dx1, dy1) < 0.1:
            continue
        h0 = math.degrees(math.atan2(dy0, dx0))
        h1 = math.degrees(math.atan2(dy1, dx1))
        d = abs(_wrap_180(h1 - h0))
        if d > m.max_yaw_sliding_deg:
            m.max_yaw_sliding_deg = d

    if speeds:
        m.mean_speed_m_s = sum(speeds) / len(speeds)
        m.max_speed_m_s = max(speeds)
    if accels:
        m.mean_accel_mag = sum(accels) / len(accels)
        m.max_accel_mag = max(accels)
    if jerks:
        m.mean_jerk = sum(jerks) / len(jerks)
        m.p95_jerk = _percentile(jerks, 0.95)
        m.max_jerk = max(jerks)

    # ---- Lane / map adherence ----
    lane_dists: List[float] = []
    n_eval = 0
    n_on_1m = 0
    n_on_05m = 0
    n_aligned = 0
    n_align_eval = 0  # count only frames where moving fast enough to evaluate
    n_off = 0
    n_wrongway = 0
    for i, (sx, sy) in enumerate(sxy):
        if not (math.isfinite(sx) and math.isfinite(sy)):
            continue
        nl = _nearest_lane(segs, sx, sy)
        if nl is None:
            continue
        d, lane_h, _ = nl
        n_eval += 1
        lane_dists.append(d)
        if d <= 1.0:
            n_on_1m += 1
        if d <= 0.5:
            n_on_05m += 1
        if d > 3.0:
            n_off += 1
        # heading alignment — only count frames where the vehicle is moving
        if i > 0:
            sx0, sy0 = sxy[i - 1]
            if all(math.isfinite(v) for v in (sx0, sy0)):
                move = math.hypot(sx - sx0, sy - sy0)
                if move > 0.2:
                    n_align_eval += 1
                    snap_h = math.degrees(math.atan2(sy - sy0, sx - sx0))
                    df = _yaw_diff_deg(snap_h, lane_h)
                    dr = _yaw_diff_deg(snap_h, lane_h + 180.0)
                    if min(df, dr) <= 30.0:
                        n_aligned += 1
        # wrong-way: snap heading vs raw heading
        rx, ry = rxy[i]
        if i > 0 and all(math.isfinite(v) for v in (rx, ry)):
            rx0, ry0 = rxy[i - 1]
            sx0, sy0 = sxy[i - 1]
            if all(math.isfinite(v) for v in (rx0, ry0, sx0, sy0)):
                rmv = math.hypot(rx - rx0, ry - ry0)
                smv = math.hypot(sx - sx0, sy - sy0)
                if rmv > 0.3 and smv > 0.3:
                    rh = math.degrees(math.atan2(ry - ry0, rx - rx0))
                    sh = math.degrees(math.atan2(sy - sy0, sx - sx0))
                    if _yaw_diff_deg(sh, rh) > 120.0:
                        n_wrongway += 1
    if n_eval > 0:
        m.pct_on_lane_1m = _pct(n_on_1m, n_eval)
        m.pct_on_lane_05m = _pct(n_on_05m, n_eval)
        m.mean_lane_lat_offset_m = sum(lane_dists) / n_eval
        m.max_lane_lat_offset_m = max(lane_dists)
    m.n_alignment_eval_frames = n_align_eval
    if n_align_eval >= 4:  # need a meaningful sample size
        m.pct_lane_aligned = _pct(n_aligned, n_align_eval)
    else:
        m.pct_lane_aligned = 1.0  # not meaningful — don't penalize
    m.n_off_route_frames = n_off
    m.n_wrongway_frames = n_wrongway

    # Monotonicity along the assigned CCLI's polyline (when available)
    # Check if t along the polyline ever goes backward by >0.05 progress units
    # (treat each ccli run as a separate progression)
    last_progress: Dict[int, float] = {}
    for i in range(n):
        ccli = cclis[i]
        if ccli < 0:
            continue
        poly = _ccli_polyline(dataset, ccli)
        if len(poly) < 2:
            continue
        sx, sy = sxy[i]
        if not (math.isfinite(sx) and math.isfinite(sy)):
            continue
        # find best segment within polyline
        best_t_global = 0.0
        best_d = float("inf")
        cum_len = 0.0
        total_len = 0.0
        for j in range(len(poly) - 1):
            seg_len = math.hypot(poly[j + 1][0] - poly[j][0], poly[j + 1][1] - poly[j][1])
            total_len += seg_len
        cum_len = 0.0
        for j in range(len(poly) - 1):
            x0, y0 = poly[j]; x1, y1 = poly[j + 1]
            sdx = x1 - x0; sdy = y1 - y0; L2 = sdx * sdx + sdy * sdy
            if L2 < 1e-9:
                continue
            t = max(0.0, min(1.0, ((sx - x0) * sdx + (sy - y0) * sdy) / L2))
            cx = x0 + t * sdx; cy = y0 + t * sdy
            d = math.hypot(cx - sx, cy - sy)
            if d < best_d:
                best_d = d
                seg_len = math.sqrt(L2)
                best_t_global = (cum_len + t * seg_len) / max(1e-6, total_len)
            cum_len += math.sqrt(L2)
        prev = last_progress.get(ccli)
        if prev is not None and best_t_global < prev - 0.05:
            monot_viol += 1
        last_progress[ccli] = best_t_global
    m.monotonicity_violations = monot_viol

    # Polyline-following error: distance from snap to the polyline of its OWN ccli
    pl_dists: List[float] = []
    for i in range(n):
        ccli = cclis[i]
        if ccli < 0:
            continue
        sx, sy = sxy[i]
        if not (math.isfinite(sx) and math.isfinite(sy)):
            continue
        poly = _ccli_polyline(dataset, ccli)
        if len(poly) < 2:
            continue
        d = _dist_to_polyline(poly, sx, sy)
        if math.isfinite(d):
            pl_dists.append(d)
    if pl_dists:
        m.polyline_follow_err_mean_m = sum(pl_dists) / len(pl_dists)
        m.polyline_follow_err_max_m = max(pl_dists)

    # ---- Raw vs aligned ----
    # Lateral deviation between snap and raw, along raw's overall direction.
    raw_lat_devs: List[float] = []
    if len(rxy) >= 4 and len(sxy) >= 4:
        # global track direction from raw start→end
        rxs = [p[0] for p in rxy if math.isfinite(p[0]) and math.isfinite(p[1])]
        rys = [p[1] for p in rxy if math.isfinite(p[0]) and math.isfinite(p[1])]
        if len(rxs) >= 2:
            dx = rxs[-1] - rxs[0]
            dy = rys[-1] - rys[0]
            track_len = math.hypot(dx, dy)
            if track_len > 1.0:
                fx, fy = dx / track_len, dy / track_len
                px, py = -fy, fx
                for (rx, ry), (sx, sy) in zip(rxy, sxy):
                    if not all(math.isfinite(v) for v in (rx, ry, sx, sy)):
                        continue
                    raw_lat_devs.append(abs((sx - rx) * px + (sy - ry) * py))
    if raw_lat_devs:
        m.raw_aligned_mean_lat_m = sum(raw_lat_devs) / len(raw_lat_devs)
        m.raw_aligned_max_lat_m = max(raw_lat_devs)

    # snap_lateral_excess: standard deviation of perpendicular spread, snap minus raw
    def _lat_spread(xs: List[Tuple[float, float]]) -> float:
        good = [(x, y) for x, y in xs if math.isfinite(x) and math.isfinite(y)]
        if len(good) < 4:
            return 0.0
        x0, y0 = good[0]
        xN, yN = good[-1]
        dx = xN - x0; dy = yN - y0
        track_len = math.hypot(dx, dy)
        if track_len < 2.0:
            return 0.0
        fx, fy = dx / track_len, dy / track_len
        px, py = -fy, fx
        perps = [(x - x0) * px + (y - y0) * py for x, y in good]
        return float(max(perps) - min(perps))

    raw_spread = _lat_spread(rxy)
    snap_spread = _lat_spread(sxy)
    m.snap_lateral_excess_m = max(0.0, snap_spread - raw_spread)

    # ---- Spawn / boundary ----
    sx0, sy0 = sxy[0]
    if math.isfinite(sx0) and math.isfinite(sy0):
        nl = _nearest_lane(segs, sx0, sy0)
        if nl is not None:
            d, lh, _ = nl
            m.spawn_lane_offset_m = d
            # spawn yaw vs lane heading
            sy_yaw = syaw[0] if math.isfinite(syaw[0]) else 0.0
            df = _yaw_diff_deg(sy_yaw, lh)
            dr = _yaw_diff_deg(sy_yaw, lh + 180.0)
            m.spawn_lane_yaw_diff_deg = float(min(df, dr))
    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        for sx, sy in sxy:
            if not (math.isfinite(sx) and math.isfinite(sy)):
                continue
            if sx < xmin or sx > xmax or sy < ymin or sy > ymax:
                m.n_outside_map_bbox += 1

    # ---- Inter-actor collisions (vs sibling vehicle tracks) ----
    # Frames are aligned by their `t` field, not by index — different tracks
    # may start at different times. This is critical: comparing frame[0] of
    # two tracks that begin at different times will produce false collisions.
    n_collisions_total = 0
    n_collisions_vs_moving = 0
    n_collisions_vs_static = 0
    min_dist = float("inf")
    spawn_partners: List[str] = []
    spawn_collision_any = 0
    self_t0 = _safe_float(frames[0].get("t"), 0.0)
    self_idx_by_t: Dict[int, int] = {}
    for fi, f in enumerate(frames):
        t = _safe_float(f.get("t"), float("nan"))
        if math.isfinite(t):
            self_idx_by_t[int(round(t * 100.0))] = fi
    for other in other_tracks:
        if other is trk:
            continue
        ofr = other.get("frames") or []
        if not ofr:
            continue
        oL = _safe_float(other.get("length"), 4.5)
        oW = _safe_float(other.get("width"), 2.0)
        other_xy = _frames_xy(ofr, "raw")
        other_finite = [(rx, ry) for rx, ry in other_xy if math.isfinite(rx) and math.isfinite(ry)]
        other_static = True
        if len(other_finite) >= 2:
            other_disp = math.hypot(
                other_finite[-1][0] - other_finite[0][0],
                other_finite[-1][1] - other_finite[0][1],
            )
            other_static = other_disp < 2.0
        if bool(other.get("low_motion_vehicle", False)):
            other_static = True
        # Iterate other's frames; align by `t` to self's
        for ofi, of in enumerate(ofr):
            t_other = _safe_float(of.get("t"), float("nan"))
            if not math.isfinite(t_other):
                continue
            t_key = int(round(t_other * 100.0))
            if t_key not in self_idx_by_t:
                continue
            i_self = self_idx_by_t[t_key]
            sx, sy = sxy[i_self]
            yaw = syaw[i_self] if math.isfinite(syaw[i_self]) else 0.0
            ox = _safe_float(of.get("cx"))
            oy = _safe_float(of.get("cy"))
            oyaw = _safe_float(of.get("cyaw"), default=_safe_float(of.get("yaw")))
            if not all(math.isfinite(v) for v in (sx, sy, ox, oy, oyaw)):
                continue
            # Cheap broadphase
            if math.hypot(sx - ox, sy - oy) > (max(L, oL) + max(W, oW)) + 0.1:
                continue
            a = _obb_corners(sx, sy, yaw, L, W)
            b = _obb_corners(ox, oy, oyaw, oL, oW)
            d = _obb_distance(a, b)
            if d < min_dist:
                min_dist = d
            pen = _obb_penetration(a, b) if d <= 0.0 else 0.0
            if pen >= 0.20:
                n_collisions_total += 1
                if other_static:
                    n_collisions_vs_static += 1
                else:
                    n_collisions_vs_moving += 1
                if i_self == 0:
                    spawn_collision_any = 1
                    spawn_partners.append(str(other.get("id", "?")))
        # de-dup spawn partner if reported multiple times
    m.n_inter_actor_collisions = n_collisions_total
    m.n_collisions_vs_moving = n_collisions_vs_moving
    m.n_collisions_vs_static = n_collisions_vs_static
    m.spawn_collision = spawn_collision_any
    m.min_inter_actor_dist_m = min_dist if min_dist != float("inf") else 9999.0
    # de-dup spawn partner list, preserve order
    seen = set()
    deduped: List[str] = []
    for p in spawn_partners:
        if p not in seen:
            seen.add(p); deduped.append(p)
    m.spawn_collision_partners = deduped

    # ---- Failure list & score ----
    # For static (parked) vehicles, only a small subset of metrics are
    # meaningful: inter-actor collisions vs moving actors, and that the
    # vehicle is not on the active route lane (which is desired — parked
    # cars are by the curb, not in the driving lane). All kinematic /
    # path-quality / on-lane metrics fail trivially because the input is
    # 1-pixel-jitter annotation noise; flagging them is noise, not signal.
    static_skip = {
        "n_jumps_over_2m", "n_jumps_over_1m", "max_xy_jump_m",
        "mean_jerk", "n_jerk_over_5_mps3", "n_accel_over_6_mps2", "max_speed_m_s",
        "max_yaw_step_deg", "max_yaw_sliding_deg", "n_yaw_jumps",
        "pct_on_lane_1m", "pct_on_lane_05m",
        "max_lane_lat_offset_m", "mean_lane_lat_offset_m",
        "n_off_route_frames", "n_wrongway_frames", "pct_lane_aligned",
        "n_lane_changes",
        "polyline_follow_err_mean_m", "polyline_follow_err_max_m",
        "raw_aligned_max_lat_m", "raw_aligned_mean_lat_m", "snap_lateral_excess_m",
        "spawn_lane_offset_m", "spawn_lane_yaw_diff_deg",
        "monotonicity_violations",
    }
    metric_dict = asdict(m)
    for name, (lo, hi, _brief) in THRESHOLDS.items():
        if name not in metric_dict:
            continue
        # Skip irrelevant checks for parked vehicles
        if m.is_static and name in static_skip:
            continue
        v = metric_dict[name]
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            continue
        if lo is not None and v < lo:
            m.failures.append(f"{name}={v}<{lo}")
        if hi is not None and v > hi:
            m.failures.append(f"{name}={v}>{hi}")

    # Composite score (lower = better). Used for ranking only.
    m.score = (
        m.mean_jerk * 12.0
        + m.n_jumps_over_2m * 5.0
        + m.n_jumps_over_1m * 0.5
        + m.max_xy_jump_m * 0.6
        + m.n_lane_changes * 0.06
        + m.max_yaw_sliding_deg * 0.012
        + m.n_off_route_frames * 1.5
        + (1.0 - m.pct_on_lane_1m) * 8.0
        + (1.0 - m.pct_lane_aligned) * 3.0
        + m.n_wrongway_frames * 5.0
        + m.snap_lateral_excess_m * 1.2
        + m.spawn_lane_offset_m * 0.8
        + m.spawn_lane_yaw_diff_deg * 0.04
        + m.n_inter_actor_collisions * 6.0
        + m.spawn_collision * 25.0
        + m.monotonicity_violations * 2.0
        + m.polyline_follow_err_max_m * 1.5
    )
    return m


# ---------------------------------------------------------------------------
# Scenario aggregation
# ---------------------------------------------------------------------------
@dataclass
class ScenarioReport:
    scenario: str
    n_vehicles: int
    n_static_vehicles: int
    n_moving_vehicles: int
    n_failing_actors: int
    n_failing_moving_actors: int
    n_collisions_total: int
    n_collisions_moving_only: int
    n_with_jump_2m: int
    n_with_off_route: int
    n_with_wrongway: int
    fleet_pass_rate: float
    moving_pass_rate: float
    actor_metrics: List[ActorMetrics] = field(default_factory=list)


def _xy_bbox(dataset: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    xs: List[float] = []
    ys: List[float] = []
    for trk in dataset.get("tracks") or []:
        for f in trk.get("frames") or []:
            for k in ("x", "cx"):
                v = _safe_float(f.get(k))
                if math.isfinite(v):
                    xs.append(v)
            for k in ("y", "cy"):
                v = _safe_float(f.get(k))
                if math.isfinite(v):
                    ys.append(v)
    if not xs or not ys:
        return None
    pad = 30.0
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def _detect_fps(dataset: Dict[str, Any]) -> float:
    fps = _safe_float(dataset.get("fps"), default=10.0)
    if 1.0 <= fps <= 60.0:
        return fps
    return 10.0


def evaluate_dataset(dataset: Dict[str, Any], scenario_name: str = "") -> ScenarioReport:
    segs = _build_lane_segs(dataset)
    bbox = _xy_bbox(dataset)
    fps = _detect_fps(dataset)
    tracks = dataset.get("tracks") or []
    veh_tracks = [t for t in tracks if _track_kind(t) == "vehicle"]

    actor_metrics: List[ActorMetrics] = []
    for trk in veh_tracks:
        am = _compute_actor(trk, dataset, segs, bbox, veh_tracks, fps)
        if am is not None:
            actor_metrics.append(am)

    n_failing = sum(1 for a in actor_metrics if a.failures)
    moving = [a for a in actor_metrics if not a.is_static]
    static = [a for a in actor_metrics if a.is_static]
    n_failing_moving = sum(1 for a in moving if a.failures)
    # Sum and halve to avoid double-counting; this counts collision-frames not unique pairs
    n_coll = sum(a.n_inter_actor_collisions for a in actor_metrics) // 2
    n_coll_moving = sum(
        a.n_inter_actor_collisions for a in actor_metrics if not a.is_static
    ) // 2
    n_jump2 = sum(1 for a in moving if a.n_jumps_over_2m > 0)
    n_off = sum(1 for a in moving if a.n_off_route_frames > 0)
    n_ww = sum(1 for a in moving if a.n_wrongway_frames > 0)
    pass_rate = 1.0 - (n_failing / max(1, len(actor_metrics)))
    moving_pass = 1.0 - (n_failing_moving / max(1, len(moving)))

    return ScenarioReport(
        scenario=scenario_name,
        n_vehicles=len(actor_metrics),
        n_static_vehicles=len(static),
        n_moving_vehicles=len(moving),
        n_failing_actors=n_failing,
        n_failing_moving_actors=n_failing_moving,
        n_collisions_total=n_coll,
        n_collisions_moving_only=n_coll_moving,
        n_with_jump_2m=n_jump2,
        n_with_off_route=n_off,
        n_with_wrongway=n_ww,
        fleet_pass_rate=round(pass_rate, 4),
        moving_pass_rate=round(moving_pass, 4),
        actor_metrics=actor_metrics,
    )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def parse_html_dataset(html_path: Path) -> Dict[str, Any]:
    text = html_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(
        r'<script[^>]+id="dataset"[^>]*type="application/json"[^>]*>(.*?)</script>',
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not m:
        raise ValueError(f"No embedded dataset found in {html_path}")
    return json.loads(m.group(1))


def render_text_report(report: ScenarioReport) -> str:
    out: List[str] = []
    out.append(f"=== {report.scenario} ===")
    out.append(f"vehicles: {report.n_vehicles} (moving={report.n_moving_vehicles} "
               f"static={report.n_static_vehicles})")
    out.append(f"failing(all)={report.n_failing_actors}  failing(moving)={report.n_failing_moving_actors}")
    out.append(f"collisions(all)={report.n_collisions_total}  collisions(moving-pair)={report.n_collisions_moving_only}")
    out.append(f"jump>2m={report.n_with_jump_2m}  off_route={report.n_with_off_route}  "
               f"wrongway={report.n_with_wrongway}")
    out.append(f"pass_rate(all)={report.fleet_pass_rate:.2%}  "
               f"pass_rate(moving)={report.moving_pass_rate:.2%}")
    out.append("")
    out.append(f"{'id':<8s} {'kind':<5s} {'frames':>6s} {'score':>7s} {'fails':>5s} "
               f"{'pct1m':>6s} {'lat_max':>7s} {'jerk':>6s} {'jump>2m':>7s} {'collide':>7s} {'spawn_off':>9s}")
    sorted_actors = sorted(report.actor_metrics, key=lambda a: (a.is_static, -a.score))
    for a in sorted_actors:
        kind = "PARK" if a.is_static else "MOVE"
        out.append(
            f"{a.track_id:<8s} {kind:<5s} {a.n_frames:>6d} {a.score:>7.2f} {len(a.failures):>5d} "
            f"{a.pct_on_lane_1m:>6.2f} {a.max_lane_lat_offset_m:>7.2f} "
            f"{a.mean_jerk:>6.3f} {a.n_jumps_over_2m:>7d} "
            f"{a.n_inter_actor_collisions:>7d} {a.spawn_lane_offset_m:>9.2f}"
        )
    out.append("")
    failing_actors = [a for a in sorted_actors if a.failures]
    moving_failing = [a for a in failing_actors if not a.is_static]
    static_failing = [a for a in failing_actors if a.is_static]
    if moving_failing:
        out.append("=== MOVING actors with failures (priority — these are pipeline bugs) ===")
        for a in moving_failing:
            out.append(f"  [{a.track_id}] {len(a.failures)} fails: " + "; ".join(a.failures))
    if static_failing:
        out.append("")
        out.append("=== STATIC actors with failures (parked vehicles, mostly low priority) ===")
        for a in static_failing:
            out.append(f"  [{a.track_id}] {len(a.failures)} fails: " + "; ".join(a.failures))
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--html", nargs="+", required=True, help="Path(s) to trajectory HTML")
    p.add_argument("--out-dir", default=None, help="Output directory")
    p.add_argument("--out", default=None, help="Single-output JSON path (only if 1 html)")
    args = p.parse_args()

    html_paths = [Path(h).resolve() for h in args.html]
    if args.out and len(html_paths) != 1:
        print("ERROR: --out only valid with single --html", file=sys.stderr); sys.exit(2)
    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    summary_rows: List[Dict[str, Any]] = []
    for hp in html_paths:
        ds = parse_html_dataset(hp)
        # Use HTML stem as scenario name. Fall back to dataset's own name if any.
        scen_name = ds.get("scenario_name") or hp.stem
        report = evaluate_dataset(ds, scen_name)
        summary_rows.append({
            "scenario": scen_name,
            "n_vehicles": report.n_vehicles,
            "n_failing_actors": report.n_failing_actors,
            "n_collisions_total": report.n_collisions_total,
            "n_with_jump_2m": report.n_with_jump_2m,
            "n_with_off_route": report.n_with_off_route,
            "n_with_wrongway": report.n_with_wrongway,
            "fleet_pass_rate": report.fleet_pass_rate,
        })

        # Writers
        if args.out:
            out_json = Path(args.out)
        elif out_dir:
            out_json = out_dir / f"{scen_name}.eval.json"
        else:
            out_json = hp.with_suffix(".eval.json")
        out_json.write_text(json.dumps({
            "scenario": scen_name,
            "summary": {
                k: getattr(report, k)
                for k in ("n_vehicles", "n_static_vehicles", "n_moving_vehicles",
                          "n_failing_actors", "n_failing_moving_actors",
                          "n_collisions_total", "n_collisions_moving_only",
                          "n_with_jump_2m", "n_with_off_route", "n_with_wrongway",
                          "fleet_pass_rate", "moving_pass_rate")
            },
            "actors": [asdict(a) for a in report.actor_metrics],
        }, indent=2))
        out_txt = out_json.with_suffix(".txt")
        out_txt.write_text(render_text_report(report))
        print(f"wrote {out_json}")
        print(f"wrote {out_txt}")
        # Brief stdout summary
        print(f"  scenario {scen_name}: moving_pass={report.moving_pass_rate:.0%} "
              f"failing(moving)={report.n_failing_moving_actors}/{report.n_moving_vehicles} "
              f"collisions(moving)={report.n_collisions_moving_only}")

    if out_dir and len(summary_rows) > 1:
        sumf = out_dir / "eval_summary.json"
        sumf.write_text(json.dumps(summary_rows, indent=2))
        print(f"wrote {sumf}")


if __name__ == "__main__":
    main()
