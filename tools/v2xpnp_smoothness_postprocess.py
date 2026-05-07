"""Smoothness post-processing helpers for the v2xpnp -> scenarioset pipeline.

Drop-in module called from ``tools/v2xpnp_html_to_scenarioset.py`` AFTER
``convert_dataset`` builds the per-actor waypoint list (CARLA-frame x/y/yaw/t)
but BEFORE the XML is written.

Two passes are exposed:

1. ``freeze_low_motion_segments`` -- locks (x, y) inside continuous slow
   stretches so stop-and-go annotation jitter (the +/- 0.05m we observed in
   tracks 1, 7, 13 of scenario 2023-03-17-15-53-02_1_0) does not flow into
   CARLA replay. Yaw is intentionally left untouched so the heading still
   tracks the raw annotation.

2. ``gaussian_smooth_xy`` -- a separable 1D Gaussian on (x, y) that runs only
   inside a "stable run" (frames sharing the same ccli / lane / road id).
   The kernel is small (sigma ~ 0.6 frames at 10 Hz by default) so genuine
   accelerations are preserved; it just nibbles off the residual ~1-2 cm
   wobble that survives lane snapping (see vehicle 0's snap deltas oscillating
   between -0.005 and +0.025 m on adjacent frames).

Both helpers are pure functions over the python-list waypoint format
``List[Tuple[x, y, yaw_deg, t]]`` plus an optional list of per-frame
metadata dicts (matching ``track["frames"]``) used to gate the smoothing
to safe regions.

Typical hookup in ``convert_dataset``:

    from tools.smoothness_postprocess import smooth_track_waypoints
    waypoints = smooth_track_waypoints(waypoints, frames, kind=kind)

The functions never raise on degenerate input -- short tracks pass through
unchanged.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

Waypoint = Tuple[float, float, float, float]  # x, y, yaw_deg, t


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Speed (m/s) below which a frame is considered "low motion".  Picked so that
# annotation jitter (~0.05 m / 0.1 s = 0.5 m/s) is almost always above this
# threshold but real crawling traffic (>~1 m/s sustained) is not frozen.
DEFAULT_SLOW_SPEED_MPS = 0.30

# Required run length (in frames) before we freeze.  At 10 Hz this is 1.0 s.
# Anything shorter is likely a transient deceleration we want to preserve.
DEFAULT_MIN_FREEZE_FRAMES = 10

# Gaussian smoothing sigma in frames.  At 10 Hz a sigma of 0.6 corresponds to
# a kernel that mostly weights +/- 1 frame -- enough to wash out 1-2 cm wobble
# without rounding off real accelerations.
DEFAULT_SMOOTH_SIGMA_FRAMES = 0.6

# Half-window (frames) for the FIR Gaussian.  Three is plenty for sigma <= 1.
DEFAULT_SMOOTH_HALF_WINDOW = 3

# Maximum per-frame correction the Gaussian may apply.  Caps any pathological
# step where the smoother would shift a real waypoint by an unphysical amount.
DEFAULT_SMOOTH_MAX_DEVIATION_M = 0.15

# Number of frames over which to linearly blend INTO and OUT OF a freeze.
# Without a ramp, freezing teleports the actor at the segment boundary and
# adds a step discontinuity that the jerk metric picks up.  With a ramp,
# the blend is continuous and the jerk it introduces is bounded.
DEFAULT_FREEZE_RAMP_FRAMES = 3


# ---------------------------------------------------------------------------
# Pass 1: freeze low-motion segments
# ---------------------------------------------------------------------------


def _segment_speed(p0: Waypoint, p1: Waypoint) -> float:
    dt = p1[3] - p0[3]
    if dt <= 1e-6:
        return 0.0
    return math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / dt


def _raw_segment_speed(f0: Dict[str, Any], f1: Dict[str, Any]) -> float:
    try:
        dt = float(f1.get("t", 0.0)) - float(f0.get("t", 0.0))
        if dt <= 1e-6:
            return 0.0
        dx = float(f1.get("x", 0.0)) - float(f0.get("x", 0.0))
        dy = float(f1.get("y", 0.0)) - float(f0.get("y", 0.0))
        return math.hypot(dx, dy) / dt
    except (TypeError, ValueError):
        return 0.0


def _windowed_speed(
    pts_xy: Sequence[Tuple[float, float]],
    ts: Sequence[float],
    i: int,
    half_window: int,
) -> float:
    """Speed estimated as |pos(i+h) - pos(i-h)| / (t(i+h) - t(i-h)).

    Using endpoints of a small window cancels per-frame jitter so we don't
    confuse +/- 0.05 m noise with real motion.  Falls back to one-sided diffs
    near the track edges.
    """
    n = len(pts_xy)
    if n < 2:
        return 0.0
    lo = max(0, i - half_window)
    hi = min(n - 1, i + half_window)
    if hi <= lo:
        return 0.0
    dt = ts[hi] - ts[lo]
    if dt <= 1e-6:
        return 0.0
    return math.hypot(pts_xy[hi][0] - pts_xy[lo][0], pts_xy[hi][1] - pts_xy[lo][1]) / dt


def _slow_mask(
    waypoints: Sequence[Waypoint],
    frames: Optional[Sequence[Dict[str, Any]]],
    slow_speed_mps: float,
    half_window: int = 3,
) -> List[bool]:
    """Per-frame True where windowed snap speed (and windowed raw speed when
    available) is < threshold.

    The windowed estimator (default +/- 3 frames = 0.6 s at 10 Hz) is what
    distinguishes annotation jitter (random walk, near-zero net displacement)
    from real crawling (consistent net drift).  Single-segment speed is too
    noisy: 0.05 m of jitter at 10 Hz already shows 0.5 m/s instantaneous
    speed.
    """
    n = len(waypoints)
    if n == 0:
        return []

    snap_xy = [(w[0], w[1]) for w in waypoints]
    ts = [w[3] for w in waypoints]
    raw_xy: Optional[List[Tuple[float, float]]] = None
    raw_ts: Optional[List[float]] = None
    if frames is not None:
        raw_xy = []
        raw_ts = []
        for f in frames:
            try:
                raw_xy.append((float(f.get("x", 0.0)), float(f.get("y", 0.0))))
                raw_ts.append(float(f.get("t", 0.0)))
            except (TypeError, ValueError):
                raw_xy.append((0.0, 0.0))
                raw_ts.append(0.0)

    mask = [False] * n
    for i in range(n):
        snap_v = _windowed_speed(snap_xy, ts, i, half_window)
        raw_v = 0.0
        if raw_xy is not None and raw_ts is not None and i < len(raw_xy):
            raw_v = _windowed_speed(raw_xy, raw_ts, i, half_window)
            mask[i] = (snap_v < slow_speed_mps) and (raw_v < slow_speed_mps)
        else:
            mask[i] = snap_v < slow_speed_mps
    return mask


def _runs(mask: Sequence[bool]) -> List[Tuple[int, int]]:
    """Return [start, end) index spans where mask is True."""
    runs: List[Tuple[int, int]] = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        runs.append((i, j))
        i = j
    return runs


def freeze_low_motion_segments(
    waypoints: Sequence[Waypoint],
    frames: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    slow_speed_mps: float = DEFAULT_SLOW_SPEED_MPS,
    min_freeze_frames: int = DEFAULT_MIN_FREEZE_FRAMES,
    ramp_frames: int = DEFAULT_FREEZE_RAMP_FRAMES,
) -> Tuple[List[Waypoint], List[bool]]:
    """Hold (x, y) constant inside continuous low-motion segments.

    For every run of consecutive frames where the windowed snap speed (and
    windowed raw speed, when available) stays below ``slow_speed_mps`` for at
    least ``min_freeze_frames`` frames, the run is pinned to a single
    (anchor_x, anchor_y) -- the median of (x, y) inside the run.  We use the
    median instead of the first/last sample so the anchor is robust to
    annotation outliers at the boundary.

    To avoid step discontinuities at the run edges (which themselves create
    jerk spikes), we linearly blend between the surrounding free position
    and the anchor over ``ramp_frames`` frames on each side, BUT we only
    blend OUTSIDE the freeze run (using the free frames just before/after).
    Inside the run the position remains exactly the anchor.

    Returns:
      (smoothed_waypoints, frozen_mask) where ``frozen_mask[i]`` is True iff
      frame i had its (x, y) pinned exactly to the anchor (i.e. it lies
      inside the core of a freeze run, not in the ramp region).
    """
    n = len(waypoints)
    frozen_mask = [False] * n
    if n < 2:
        return list(waypoints), frozen_mask

    mask = _slow_mask(waypoints, frames, slow_speed_mps)
    out: List[Waypoint] = list(waypoints)

    accepted_runs: List[Tuple[int, int, float, float]] = []
    for start, end in _runs(mask):
        if end - start < min_freeze_frames:
            continue
        xs = sorted(waypoints[k][0] for k in range(start, end))
        ys = sorted(waypoints[k][1] for k in range(start, end))
        mid = (end - start) // 2
        anchor_x = xs[mid]
        anchor_y = ys[mid]
        accepted_runs.append((start, end, anchor_x, anchor_y))
        for k in range(start, end):
            _, _, yaw, t = out[k]
            out[k] = (anchor_x, anchor_y, yaw, t)
            frozen_mask[k] = True

    if ramp_frames > 0:
        # Ramp the frames immediately surrounding each accepted run from the
        # original (pre-freeze) trajectory toward / away from the anchor.
        for idx, (start, end, ax, ay) in enumerate(accepted_runs):
            prev_end = accepted_runs[idx - 1][1] if idx > 0 else 0
            next_start = accepted_runs[idx + 1][0] if idx + 1 < len(accepted_runs) else n

            # Pre-ramp: frames [start - ramp_frames, start) blended from
            # original toward anchor.  Stop if we'd cross into another run.
            ramp_lo = max(start - ramp_frames, prev_end)
            for k in range(ramp_lo, start):
                alpha = (k - ramp_lo + 1) / (start - ramp_lo + 1)
                ox, oy, oyaw, ot = waypoints[k]
                rx = ox * (1.0 - alpha) + ax * alpha
                ry = oy * (1.0 - alpha) + ay * alpha
                out[k] = (rx, ry, oyaw, ot)

            # Post-ramp: frames [end, end + ramp_frames) blended from anchor
            # toward original.
            ramp_hi = min(end + ramp_frames, next_start)
            for k in range(end, ramp_hi):
                alpha = (k - end + 1) / (ramp_hi - end + 1)
                ox, oy, oyaw, ot = waypoints[k]
                rx = ax * (1.0 - alpha) + ox * alpha
                ry = ay * (1.0 - alpha) + oy * alpha
                out[k] = (rx, ry, oyaw, ot)

    return out, frozen_mask


# ---------------------------------------------------------------------------
# Pass 2: Gaussian smoothing inside stable lane runs
# ---------------------------------------------------------------------------


def _gaussian_kernel(sigma: float, half_window: int) -> List[float]:
    if sigma <= 0:
        return [1.0]
    xs = list(range(-half_window, half_window + 1))
    weights = [math.exp(-0.5 * (x / sigma) ** 2) for x in xs]
    s = sum(weights)
    return [w / s for w in weights]


def _stable_run_groups(
    n: int,
    frames: Optional[Sequence[Dict[str, Any]]],
) -> List[Tuple[int, int]]:
    """Split [0, n) into runs sharing the same (ccli, road_id, assigned_lane_id).

    If ``frames`` is None we return a single run covering everything; smoothing
    will still work but it won't respect lane boundaries.
    """
    if not frames:
        return [(0, n)]
    runs: List[Tuple[int, int]] = []
    start = 0

    def key(i: int) -> Tuple[Any, Any, Any]:
        f = frames[i] if i < len(frames) else {}
        return (
            f.get("ccli"),
            f.get("road_id"),
            f.get("assigned_lane_id"),
        )

    last = key(0)
    for i in range(1, n):
        cur = key(i)
        if cur != last:
            runs.append((start, i))
            start = i
            last = cur
    runs.append((start, n))
    return runs


def gaussian_smooth_xy(
    waypoints: Sequence[Waypoint],
    frames: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    sigma_frames: float = DEFAULT_SMOOTH_SIGMA_FRAMES,
    half_window: int = DEFAULT_SMOOTH_HALF_WINDOW,
    max_deviation_m: float = DEFAULT_SMOOTH_MAX_DEVIATION_M,
    skip_mask: Optional[Sequence[bool]] = None,
) -> List[Waypoint]:
    """Smooth (x, y) with a separable 1D Gaussian, scoped to stable lane runs.

    Args:
      waypoints: List of (x, y, yaw_deg, t) tuples.
      frames: Optional per-frame metadata dicts (same length as waypoints) used
        to detect stable-CCLI runs.  Smoothing never crosses run boundaries.
      sigma_frames: Gaussian sigma expressed in frames.  0.6 is a good default
        at 10 Hz; raise to ~1.0 for noisier sources.
      half_window: FIR half-window in frames.  The kernel covers
        [-half_window, +half_window].
      max_deviation_m: Maximum absolute correction the smoother is allowed to
        apply to any single frame.  Caps pathological pulls.
      skip_mask: Optional per-frame mask. Frames where mask is True keep their
        original (x, y) -- useful to skip freeze-locked segments so smoothing
        doesn't undo them.
    """
    n = len(waypoints)
    if n == 0:
        return []
    if sigma_frames <= 0 or half_window <= 0 or n < 3:
        return list(waypoints)

    kernel = _gaussian_kernel(sigma_frames, half_window)
    out: List[Waypoint] = list(waypoints)

    for start, end in _stable_run_groups(n, frames):
        run_len = end - start
        if run_len < 3:
            continue
        for i in range(start, end):
            if skip_mask is not None and i < len(skip_mask) and skip_mask[i]:
                continue
            wsum = 0.0
            xacc = 0.0
            yacc = 0.0
            for k, w in enumerate(kernel):
                j = i + (k - half_window)
                if j < start or j >= end:
                    continue
                if skip_mask is not None and j < len(skip_mask) and skip_mask[j]:
                    # Don't pull samples from frozen frames (would smear the
                    # freeze across the boundary); fall back to original.
                    continue
                wsum += w
                xacc += w * waypoints[j][0]
                yacc += w * waypoints[j][1]
            if wsum <= 0:
                continue
            sx, sy = xacc / wsum, yacc / wsum
            ox, oy, yaw, t = waypoints[i]
            dx, dy = sx - ox, sy - oy
            mag = math.hypot(dx, dy)
            if mag > max_deviation_m and mag > 0:
                scale = max_deviation_m / mag
                sx = ox + dx * scale
                sy = oy + dy * scale
            out[i] = (sx, sy, yaw, t)
    return out


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def smooth_track_waypoints(
    waypoints: Sequence[Waypoint],
    frames: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    kind: str = "npc",
    enable_freeze: bool = True,
    enable_gaussian: bool = True,
    slow_speed_mps: float = DEFAULT_SLOW_SPEED_MPS,
    min_freeze_frames: int = DEFAULT_MIN_FREEZE_FRAMES,
    sigma_frames: float = DEFAULT_SMOOTH_SIGMA_FRAMES,
    half_window: int = DEFAULT_SMOOTH_HALF_WINDOW,
    max_deviation_m: float = DEFAULT_SMOOTH_MAX_DEVIATION_M,
) -> List[Waypoint]:
    """Run freeze + Gaussian smoothing in the right order.

    Static actors are already pinned by ``convert_dataset`` (the existing
    ``kind == "static"`` branch), so we don't touch them here.

    Returns a new waypoint list of the same length.
    """
    if kind == "static" or len(waypoints) < 3:
        return list(waypoints)

    wp = list(waypoints)

    # Pass 1: freeze stop-and-go segments.
    frozen_mask: Optional[List[bool]] = None
    if enable_freeze:
        wp, frozen_mask = freeze_low_motion_segments(
            wp,
            frames,
            slow_speed_mps=slow_speed_mps,
            min_freeze_frames=min_freeze_frames,
        )

    # Pass 2: Gaussian smoothing inside stable-lane runs.
    if enable_gaussian:
        wp = gaussian_smooth_xy(
            wp,
            frames,
            sigma_frames=sigma_frames,
            half_window=half_window,
            max_deviation_m=max_deviation_m,
            skip_mask=frozen_mask,
        )

    return wp


def reject_speed_anomalies(
    waypoints: Sequence[Waypoint],
    *,
    max_speed_mps: float = 25.0,
    smoothing_passes: int = 3,
) -> List[Waypoint]:
    """Replace frames whose inter-frame speed exceeds `max_speed_mps` (or that
    cause a >1.8x speed-ratio anomaly with neighbors) with the time-weighted
    linear interpolation between their neighbors. Iterative.

    Yaw and t are preserved.
    """
    n = len(waypoints)
    if n < 5:
        return list(waypoints)
    out = list(waypoints)
    for _ in range(int(smoothing_passes)):
        changed = False
        for i in range(1, n - 1):
            x_p, y_p, _, t_p = out[i - 1]
            x_c, y_c, yaw, t_c = out[i]
            x_n, y_n, _, t_n = out[i + 1]
            dt_pc = max(t_c - t_p, 1e-6)
            dt_cn = max(t_n - t_c, 1e-6)
            v_pc = math.hypot(x_c - x_p, y_c - y_p) / dt_pc
            v_cn = math.hypot(x_n - x_c, y_n - y_c) / dt_cn
            speed_anom = (v_pc > max_speed_mps) or (v_cn > max_speed_mps)
            jerk_anom = False
            if min(v_pc, v_cn) > 0.5:
                ratio = max(v_pc, v_cn) / min(v_pc, v_cn)
                jerk_anom = ratio > 1.8
            if speed_anom or jerk_anom:
                alpha = (t_c - t_p) / max(t_n - t_p, 1e-6)
                new_x = x_p + alpha * (x_n - x_p)
                new_y = y_p + alpha * (y_n - y_p)
                new_v_pc = math.hypot(new_x - x_p, new_y - y_p) / dt_pc
                new_v_cn = math.hypot(x_n - new_x, y_n - new_y) / dt_cn
                if max(new_v_pc, new_v_cn) < max(v_pc, v_cn) * 0.95:
                    out[i] = (new_x, new_y, yaw, t_c)
                    changed = True
        if not changed:
            break
    return out


def trim_end_anomalies(
    waypoints: Sequence[Waypoint],
    *,
    max_speed_mps: float = 25.0,
    look_back: int = 6,
) -> List[Waypoint]:
    """Drop trailing frames whose inter-frame speed exceeds `max_speed_mps`.

    Snap pipelines sometimes produce end-of-track teleports where the last
    1-3 frames jumped to catch up to raw across a frame gap.
    """
    n = len(waypoints)
    if n < 5:
        return list(waypoints)
    out = list(waypoints)
    end = n
    last_to_check = max(0, n - look_back)
    for i in range(n - 1, last_to_check, -1):
        dt = out[i][3] - out[i - 1][3]
        if dt <= 0:
            continue
        v = math.hypot(out[i][0] - out[i - 1][0], out[i][1] - out[i - 1][1]) / dt
        if v > max_speed_mps:
            end = i
        else:
            break
    return out[:end]


__all__ = [
    "freeze_low_motion_segments",
    "gaussian_smooth_xy",
    "smooth_track_waypoints",
    "reject_speed_anomalies",
    "trim_end_anomalies",
    "DEFAULT_SLOW_SPEED_MPS",
    "DEFAULT_MIN_FREEZE_FRAMES",
    "DEFAULT_SMOOTH_SIGMA_FRAMES",
    "DEFAULT_SMOOTH_HALF_WINDOW",
    "DEFAULT_SMOOTH_MAX_DEVIATION_M",
]


# ---------------------------------------------------------------------------
# Self-test (executable as a script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    random.seed(0)

    # Build a synthetic stop-and-go track: vehicle creeps at 0.1 m/s along x
    # with +/-0.05 m noise on y for 30 frames at 10 Hz.
    wps: List[Waypoint] = []
    fs: List[Dict[str, Any]] = []
    for i in range(30):
        t = i * 0.1
        x = 100.0 + 0.01 * i  # 0.1 m/s
        y = 50.0 + random.uniform(-0.05, 0.05)
        yaw = 0.0
        wps.append((x, y, yaw, t))
        fs.append({"t": t, "x": x, "y": y, "ccli": 7, "road_id": 1, "assigned_lane_id": 0})

    smoothed = smooth_track_waypoints(wps, fs)
    # All y values inside the slow segment should be identical (frozen).
    ys = sorted({round(w[1], 6) for w in smoothed})
    print(f"unique y values after freeze+smooth: {len(ys)}  (was 30)")
    print(f"first 5 smoothed: {smoothed[:5]}")
    assert len(ys) <= 2, "freeze should collapse y noise"
    print("self-test ok")
