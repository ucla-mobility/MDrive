#!/usr/bin/env python3
"""fix_actor_collisions.py — in-place collision-free editing of v2xpnp scene XMLs.

For one scenario folder under ``scenarioset/v2xpnp/``, eliminate every
pairwise OBB-OBB collision above the sub-noise threshold by editing
actor XMLs in place. **The v2xpnp pipeline itself is NOT touched** — we
only modify the per-actor XML files that already live under the scene
directory.

Four fix mechanisms, in priority order:

1. **STAGGER (identical-spawn cluster)**
   Two or more actors whose first waypoint is within 0.3 m / 5° of each
   other are treated as a colliding spawn cluster. We anchor the actor
   with the longest arc length at its current ``time=0``, then for each
   sibling we binary-search the smallest non-negative time offset Δt
   such that the sibling's OBB never overlaps any already-placed actor's
   OBB at any common tick. Δt is *added to every waypoint's* ``time``
   attribute. If no Δt within ``--max-stagger-s`` clears the conflict,
   fall back to REMOVE.

2. **TRUNCATE (tail collision)**
   For a pair of actors whose collision is contained to a contiguous
   tail segment of one trajectory (both share heading and lane), drop
   the trailing waypoints of the "later-arriving" actor up to the last
   one where the OBB no longer overlaps the leader's OBB.

3. **SNAP-OR-REMOVE STATIC**
   For pairs where one actor is ``role=static`` and the other is a
   moving NPC, the static is blocking the lane. Try snapping the static
   onto the nearest Parking/Shoulder lane (≤ 8 m), preserving its yaw.
   If the snap pose introduces a new collision, REMOVE.

4. **IGNORE (sub-noise)**
   Pairs with ``n_ticks < SUBNOISE_MIN_TICKS`` AND
   ``peak < SUBNOISE_MIN_DEPTH_M`` are V2X-perception noise and not
   actionable.

Always idempotent: a re-run of the tool on already-fixed XMLs detects no
non-sub-noise collisions and writes nothing.

Usage
-----
    # Dry-run on a single scene
    python v2xpnp/scripts/fix_actor_collisions.py --scene scenarioset/v2xpnp/2023-04-07-15-05-15_4_1

    # Apply
    python v2xpnp/scripts/fix_actor_collisions.py --scene ... --apply
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROLE_DIMS: Dict[str, Tuple[float, float]] = {
    "ego":     (4.8, 2.0),
    "npc":     (4.6, 2.0),
    "static":  (4.6, 2.0),
    "walker":  (0.6, 0.6),
    "cyclist": (1.8, 0.7),
}

# Per-model OBB dimensions from CARLA 0.9.12. Used in preference to the
# coarse ROLE_DIMS so the audit matches what's rendered at runtime —
# CARLA bumper-to-bumper rendering of a Lincoln MKZ vs a Mini Cooper
# differs by ~1.1m, which is enough to turn a "0.2m clearance" model
# call into a real visual stack on the road.
MODEL_DIMS: Dict[str, Tuple[float, float]] = {
    "vehicle.audi.a2":                (3.95, 1.85),
    "vehicle.audi.tt":                (4.18, 2.00),
    "vehicle.audi.etron":             (4.60, 2.10),
    "vehicle.bmw.grandtourer":        (4.79, 2.16),
    "vehicle.bmw.isetta":             (2.40, 1.60),
    "vehicle.chevrolet.impala":       (5.30, 2.10),
    "vehicle.citroen.c3":             (4.00, 1.85),
    "vehicle.dodge.charger_2020":     (5.20, 2.10),
    "vehicle.dodge.charger_police":   (5.20, 2.10),
    "vehicle.dodge.charger_police_2020": (5.20, 2.10),
    "vehicle.ford.ambulance":         (5.85, 2.40),
    "vehicle.ford.crown":             (5.40, 2.10),
    "vehicle.ford.mustang":           (4.85, 2.10),
    "vehicle.harley-davidson.low_rider": (2.40, 0.95),
    "vehicle.jeep.wrangler_rubicon":  (4.34, 1.92),
    "vehicle.kawasaki.ninja":         (2.05, 0.90),
    "vehicle.lincoln.mkz_2017":       (4.92, 2.13),
    "vehicle.lincoln.mkz_2020":       (4.92, 2.13),
    "vehicle.mercedes.coupe":         (4.85, 2.10),
    "vehicle.mercedes.coupe_2020":    (4.95, 2.10),
    "vehicle.mercedes.sprinter":      (5.85, 2.40),
    "vehicle.mini.cooper_s":          (3.85, 1.84),
    "vehicle.mini.cooper_s_2021":     (3.78, 1.84),
    "vehicle.nissan.micra":           (3.71, 1.78),
    "vehicle.nissan.patrol":          (4.78, 1.94),
    "vehicle.nissan.patrol_2021":     (4.85, 1.95),
    "vehicle.seat.leon":              (4.35, 1.92),
    "vehicle.tesla.cybertruck":       (5.88, 2.20),
    "vehicle.tesla.model3":           (4.79, 2.06),
    "vehicle.toyota.prius":           (4.70, 1.90),
    "vehicle.vespa.zx125":            (2.05, 0.85),
    "vehicle.volkswagen.t2":          (4.50, 1.94),
    "vehicle.yamaha.yzf":             (2.05, 0.90),
}


def _dims_for(model: str, role: str) -> Tuple[float, float]:
    """Return (length, width) for an actor's bounding box. Prefers
    MODEL_DIMS lookup; falls back to ROLE_DIMS."""
    m = (model or "").strip()
    if m in MODEL_DIMS:
        return MODEL_DIMS[m]
    return ROLE_DIMS.get(role, ROLE_DIMS["npc"])

TICK_DT_S = 0.1

# Sub-noise floor: pairs below BOTH thresholds are ignored. Above either,
# we attempt a fix.
SUBNOISE_MIN_TICKS    = 3
SUBNOISE_MIN_DEPTH_M  = 0.3

# Cluster detection (identical-spawn).
SPAWN_CLUSTER_DIST_M  = 0.3
SPAWN_CLUSTER_YAW_DEG = 5.0

# Stagger search.
DEFAULT_MAX_STAGGER_S = 5.0
STAGGER_STEP_S        = 0.1

# Static snap-or-remove (uses CARLA xodr).
DEFAULT_XODR    = Path("v2xpnp/map/ucla_v2.xodr")
DEFAULT_CARLA_EGG = Path("carla912/PythonAPI/carla/dist/carla-0.9.12-py3.7-linux-x86_64.egg")
SNAP_NONDRIVING_MAX_M = 8.0   # how far we'll reach to find Parking/Shoulder

COLLISION_PADDING_M  = 0.0    # 0 = detect real OBB overlap (no margin
                              # either way). Static-pair spacing has its
                              # own 1m clearance requirement; other paths
                              # just need to avoid the OBBs actually
                              # intersecting.

# Static-pair spacing: minimum bumper-to-bumper clearance for parked
# vehicles in the same lane (same yaw within tolerance). 1m is enough
# that the rendered CARLA bounding boxes (which include side mirrors,
# bumpers, etc.) don't visually overlap.
STATIC_PAIR_MIN_CLEAR_M = 1.0
STATIC_PAIR_YAW_TOL_DEG = 15.0   # actors with yaw within this are in same lane direction
STATIC_PAIR_MAX_SCAN_DIST_M = 12.0   # only consider pairs whose centers are within this


# ---------------------------------------------------------------------------
# Geometry helpers (same as audit_actor_collisions.py)
# ---------------------------------------------------------------------------

def _wrap180(d: float) -> float:
    return ((float(d) + 180.0) % 360.0) - 180.0


def _normalize_yaw(d: float) -> float:
    d = float(d) % 360.0
    return d if d >= 0 else d + 360.0


def _obb_corners(x: float, y: float, yaw_deg: float, hl: float, hw: float
                 ) -> List[Tuple[float, float]]:
    c = math.cos(math.radians(yaw_deg))
    s = math.sin(math.radians(yaw_deg))
    out = []
    for sx, sy in ((+hl, +hw), (+hl, -hw), (-hl, -hw), (-hl, +hw)):
        out.append((x + sx * c - sy * s, y + sx * s + sy * c))
    return out


def _project(corners, ax, ay):
    proj = [cx * ax + cy * ay for cx, cy in corners]
    return min(proj), max(proj)


def _obb_overlap(c1, c2, padding: float = 0.0) -> Optional[float]:
    """Return penetration depth (m) on the smallest separating axis if OBBs
    overlap; ``None`` if separated. ``padding`` shrinks both OBBs."""
    min_o = math.inf
    for corners in (c1, c2):
        for i in range(4):
            x0, y0 = corners[i]
            x1, y1 = corners[(i + 1) % 4]
            ex, ey = x1 - x0, y1 - y0
            nx, ny = ey, -ex
            n_len = math.hypot(nx, ny)
            if n_len < 1e-9: continue
            nx /= n_len; ny /= n_len
            a_min, a_max = _project(c1, nx, ny)
            b_min, b_max = _project(c2, nx, ny)
            o = min(a_max, b_max) - max(a_min, b_min) - padding
            if o < 0: return None
            if o < min_o: min_o = o
    return min_o


# ---------------------------------------------------------------------------
# Trajectory data
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    actor_id: str           # e.g. "ucla_v2_custom_Vehicle_19_npc"
    xml_path: Path
    role: str
    model: str
    length: float
    width: float
    times: List[float]
    xs:    List[float]
    ys:    List[float]
    yaws:  List[float]
    # Mutation state — what we want to write back. Initially the identity
    # transform on the source XML.
    time_offset_s: float = 0.0      # add to every waypoint's `time`
    truncate_after_idx: Optional[int] = None  # drop waypoints after this index (inclusive)
    snap_pose: Optional[Tuple[float, float, float]] = None  # static snap to (x, y, yaw)
    remove: bool = False
    fix_note: str = ""


def _parse_xml(xml_path: Path) -> Optional[Trajectory]:
    try:
        tree = ET.parse(xml_path)
    except Exception:
        return None
    root = tree.getroot()
    route = root.find("route") if root.tag != "route" else root
    if route is None: return None
    role = (route.attrib.get("role") or "").lower()
    if role not in ROLE_DIMS: role = "npc"
    model = route.attrib.get("model", "?")
    length, width = _dims_for(model, role)
    wps = route.findall("waypoint")
    times, xs, ys, yaws = [], [], [], []
    for w in wps:
        try:
            t  = float(w.attrib.get("time", "0"))
            x  = float(w.attrib["x"])
            y  = float(w.attrib["y"])
            yw = float(w.attrib.get("yaw", "0"))
        except Exception:
            continue
        times.append(t); xs.append(x); ys.append(y); yaws.append(yw)
    if len(times) < 2: return None
    return Trajectory(
        actor_id=xml_path.stem, xml_path=xml_path, role=role, model=model,
        length=length, width=width,
        times=times, xs=xs, ys=ys, yaws=yaws,
    )


def _interpolate(t: Trajectory, real_t: float) -> Optional[Tuple[float, float, float]]:
    """Pose at real-world time, accounting for time_offset and truncation."""
    times = t.times
    if t.truncate_after_idx is not None:
        last = t.truncate_after_idx
        if last < len(times):
            times = times[:last + 1]
    if not times: return None
    shifted_start = times[0]  + t.time_offset_s
    shifted_end   = times[-1] + t.time_offset_s
    if real_t < shifted_start - 1e-6 or real_t > shifted_end + 1e-6:
        return None
    internal_t = real_t - t.time_offset_s
    lo, hi = 0, len(times) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if times[mid] <= internal_t: lo = mid
        else: hi = mid
    t0, t1 = times[lo], times[hi]
    if t1 == t0:
        if t.snap_pose is not None: return t.snap_pose
        return (t.xs[lo], t.ys[lo], t.yaws[lo])
    alpha = (internal_t - t0) / (t1 - t0)
    if t.snap_pose is not None:
        # Static-snap: pose is constant across all ticks.
        return t.snap_pose
    x   = t.xs[lo]   + alpha * (t.xs[hi]   - t.xs[lo])
    y   = t.ys[lo]   + alpha * (t.ys[hi]   - t.ys[lo])
    yaw = t.yaws[lo] + alpha * _wrap180(t.yaws[hi] - t.yaws[lo])
    return (x, y, yaw)


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

@dataclass
class PairCollision:
    a: str
    b: str
    role_a: str
    role_b: str
    n_ticks: int
    peak: float
    first_t: float
    last_t: float


def _audit(trajs: List[Trajectory], tick_dt: float, padding: float
           ) -> List[PairCollision]:
    if not trajs: return []
    # Determine real-world time range across all trajectories
    times = []
    for t in trajs:
        if t.remove: continue
        ts = t.times
        if t.truncate_after_idx is not None:
            ts = ts[:t.truncate_after_idx + 1]
        if not ts: continue
        times.append(ts[0]  + t.time_offset_s)
        times.append(ts[-1] + t.time_offset_s)
    if not times: return []
    t_min, t_max = min(times), max(times)
    pair: Dict[Tuple[str, str], dict] = {}
    real_t = t_min
    while real_t <= t_max + 1e-6:
        live: List[Tuple[Trajectory, float, float, float]] = []
        for tr in trajs:
            if tr.remove: continue
            p = _interpolate(tr, real_t)
            if p is None: continue
            live.append((tr, *p))
        corners = []
        for tr, x, y, yaw in live:
            corners.append(_obb_corners(x, y, yaw, tr.length / 2.0, tr.width / 2.0))
        for i in range(len(live)):
            ax = [c[0] for c in corners[i]]; ay = [c[1] for c in corners[i]]
            for j in range(i + 1, len(live)):
                bx = [c[0] for c in corners[j]]; by = [c[1] for c in corners[j]]
                if (max(ax) < min(bx) or max(bx) < min(ax) or
                    max(ay) < min(by) or max(by) < min(ay)):
                    continue
                d = _obb_overlap(corners[i], corners[j], padding)
                if d is None: continue
                ka = live[i][0].actor_id; kb = live[j][0].actor_id
                key = (ka, kb) if ka < kb else (kb, ka)
                st = pair.setdefault(key, {"n": 0, "peak": 0.0,
                                            "first_t": real_t,
                                            "last_t": real_t,
                                            "role_a": live[i][0].role,
                                            "role_b": live[j][0].role})
                st["n"] += 1
                if d > st["peak"]: st["peak"] = d
                st["last_t"] = real_t
        real_t += tick_dt
    out = []
    for (a, b), st in pair.items():
        out.append(PairCollision(a=a, b=b, role_a=st["role_a"], role_b=st["role_b"],
                                 n_ticks=st["n"], peak=st["peak"],
                                 first_t=st["first_t"], last_t=st["last_t"]))
    out.sort(key=lambda p: (-p.n_ticks, -p.peak))
    return out


def _is_subnoise(p: PairCollision) -> bool:
    """Sub-noise = V2X perception noise we accept and don't fix.

    **Zero-tolerance for ego.** Any pair involving an ego actor is never
    sub-noise, no matter how brief or shallow the OBB overlap — the
    logreplay follower can't be relied on to spirit non-ego actors out
    of the ego's way, so the XML itself must be free of ego-collisions.
    """
    if p.role_a == "ego" or p.role_b == "ego":
        return False
    return p.n_ticks < SUBNOISE_MIN_TICKS and p.peak < SUBNOISE_MIN_DEPTH_M


# ---------------------------------------------------------------------------
# Cluster 1: identical-spawn stagger
# ---------------------------------------------------------------------------

def _group_identical_spawn(trajs: List[Trajectory]) -> List[List[Trajectory]]:
    """Group trajectories whose first waypoint is within
    SPAWN_CLUSTER_DIST_M and SPAWN_CLUSTER_YAW_DEG of each other.
    Walkers/cyclists/egos excluded."""
    candidates = [t for t in trajs
                  if t.role in ("npc", "static") and not t.remove]
    clusters: List[List[Trajectory]] = []
    used = set()
    for i, ta in enumerate(candidates):
        if ta.actor_id in used: continue
        group = [ta]; used.add(ta.actor_id)
        for tb in candidates[i + 1:]:
            if tb.actor_id in used: continue
            d = math.hypot(ta.xs[0] - tb.xs[0], ta.ys[0] - tb.ys[0])
            ydiff = abs(_wrap180(ta.yaws[0] - tb.yaws[0]))
            if d < SPAWN_CLUSTER_DIST_M and ydiff < SPAWN_CLUSTER_YAW_DEG:
                group.append(tb); used.add(tb.actor_id)
        if len(group) > 1:
            clusters.append(group)
    return clusters


def _arc_length(t: Trajectory) -> float:
    s = 0.0
    for i in range(len(t.xs) - 1):
        s += math.hypot(t.xs[i + 1] - t.xs[i], t.ys[i + 1] - t.ys[i])
    return s


def _check_pair_clear(a: Trajectory, b: Trajectory, b_offset_s: float,
                      tick_dt: float, padding: float) -> bool:
    """Would B (with a hypothetical time_offset_s = b_offset_s) overlap A
    at any common real-world tick? Returns True if CLEAR (no overlap)."""
    a_ts = a.times
    if a.truncate_after_idx is not None: a_ts = a_ts[:a.truncate_after_idx + 1]
    b_ts = b.times
    if b.truncate_after_idx is not None: b_ts = b_ts[:b.truncate_after_idx + 1]
    if not a_ts or not b_ts: return True
    a_start = a_ts[0]  + a.time_offset_s
    a_end   = a_ts[-1] + a.time_offset_s
    b_start = b_ts[0]  + b_offset_s
    b_end   = b_ts[-1] + b_offset_s
    lo = max(a_start, b_start)
    hi = min(a_end,   b_end)
    if hi < lo: return True
    real_t = lo
    while real_t <= hi + 1e-6:
        pa = _interpolate(a, real_t)
        # Use a temporary lookup for B with the hypothetical offset
        if pa is None: real_t += tick_dt; continue
        # mimic _interpolate but with b_offset_s
        internal_t = real_t - b_offset_s
        if internal_t < b_ts[0] - 1e-6 or internal_t > b_ts[-1] + 1e-6:
            real_t += tick_dt; continue
        # Linear interp on B
        lo_i, hi_i = 0, len(b_ts) - 1
        while hi_i - lo_i > 1:
            mid = (lo_i + hi_i) // 2
            if b_ts[mid] <= internal_t: lo_i = mid
            else: hi_i = mid
        t0, t1 = b_ts[lo_i], b_ts[hi_i]
        if t1 == t0:
            xb, yb, ywb = b.xs[lo_i], b.ys[lo_i], b.yaws[lo_i]
        else:
            alpha = (internal_t - t0) / (t1 - t0)
            xb = b.xs[lo_i] + alpha * (b.xs[hi_i] - b.xs[lo_i])
            yb = b.ys[lo_i] + alpha * (b.ys[hi_i] - b.ys[lo_i])
            ywb = b.yaws[lo_i] + alpha * _wrap180(b.yaws[hi_i] - b.yaws[lo_i])
        # B's hypothetical pose (overrides any snap because we're searching offset)
        if b.snap_pose is not None: xb, yb, ywb = b.snap_pose
        ca = _obb_corners(*pa, a.length / 2.0, a.width / 2.0)
        cb = _obb_corners(xb, yb, ywb, b.length / 2.0, b.width / 2.0)
        if _obb_overlap(ca, cb, padding) is not None:
            return False
        real_t += tick_dt
    return True


def _find_min_offset(b: Trajectory, placed: List[Trajectory], max_s: float,
                     step_s: float, tick_dt: float, padding: float
                     ) -> Optional[float]:
    """Smallest non-negative offset Δt for B that clears every actor in
    ``placed``. None if none ≤ max_s works."""
    delta = 0.0
    while delta <= max_s + 1e-6:
        if all(_check_pair_clear(p, b, delta, tick_dt, padding) for p in placed):
            return delta
        delta += step_s
    return None


def _stagger_cluster(cluster: List[Trajectory], all_placed_others: List[Trajectory],
                     max_stagger_s: float, tick_dt: float, padding: float
                     ) -> None:
    """Time-stagger members of an identical-spawn cluster.

    Cluster members are placed in arc-length-descending order. For *each*
    member (including the largest), we search the smallest non-negative Δt
    that clears all already-placed actors — egos and other immutable
    actors live in ``all_placed_others`` so they act as zero-shift
    constraints. The largest-arc member usually settles at Δt=0 but is
    not assumed to; if it overlaps an ego at t=0, it gets a delay too.
    """
    cluster_sorted = sorted(cluster, key=lambda t: -_arc_length(t))
    placed = list(all_placed_others)
    for member in cluster_sorted:
        delta = _find_min_offset(member, placed, max_stagger_s, STAGGER_STEP_S,
                                  tick_dt, padding)
        if delta is None:
            member.remove = True
            member.fix_note = (member.fix_note + "; " if member.fix_note else "") + \
                              f"REMOVE (no Δt ≤ {max_stagger_s}s clears spawn cluster + immutable actors)"
            continue
        member.time_offset_s = delta
        tag = "cluster-anchor" if delta == 0.0 else f"stagger Δt=+{delta:.1f}s"
        member.fix_note = (member.fix_note + "; " if member.fix_note else "") + \
                          f"{tag} (arc={_arc_length(member):.0f}m, cluster of {len(cluster_sorted)})"
        placed.append(member)


# ---------------------------------------------------------------------------
# Static-pair spacing
#
# Two parked static actors in the same lane direction that are
# bumper-to-bumper (< STATIC_PAIR_MIN_CLEAR_M clearance) read as a
# visual stack in BEV even though SAT might call them "separated".
# Shift the behind one backward along the lane heading by the missing
# clearance. We walk same-lane chains in along-heading order so chains
# of 3+ parked cars cascade correctly (V_30 → V_31 → V_39 etc).
# ---------------------------------------------------------------------------

def _space_static_pairs(trajs: List[Trajectory],
                        tick_dt: float, padding: float
                        ) -> int:
    """Iteratively shift tight static pairs backward along their lane
    direction so each pair has ≥ STATIC_PAIR_MIN_CLEAR_M clearance.

    Each iteration: find the tightest static-static pair that
    pairwise-time-overlaps and shares a lane (yaw within tol, lateral
    within tol, distance within scan radius). Shift the one with the
    smaller along-heading projection (the "behind" one) backward by
    just enough to give 1m clearance. Iterate until no tight pair
    remains or we can't shift any more without colliding with a
    non-static / non-pair actor.

    This handles chains (V_25 ↔ V_30 ↔ V_31 ↔ V_39) by re-evaluating
    after each shift: once V_31 moves back, the V_31-V_39 pair tightens
    and is fixed in the next iteration.
    """
    statics = [t for t in trajs if t.role == "static" and not t.remove]
    if len(statics) < 2: return 0

    def _t_range(t: Trajectory) -> Tuple[float, float]:
        return (t.times[0] + t.time_offset_s, t.times[-1] + t.time_offset_s)

    def _cur_pose(t: Trajectory) -> Tuple[float, float, float]:
        if t.snap_pose is not None: return t.snap_pose
        return (t.xs[0], t.ys[0], t.yaws[0])

    def _compat(a: Trajectory, b: Trajectory) -> bool:
        ax, ay, ayaw = _cur_pose(a); bx, by, byaw = _cur_pose(b)
        if abs(_wrap180(ayaw - byaw)) > STATIC_PAIR_YAW_TOL_DEG: return False
        dx, dy = bx - ax, by - ay
        if math.hypot(dx, dy) > STATIC_PAIR_MAX_SCAN_DIST_M: return False
        a_perp = (-math.sin(math.radians(ayaw)), math.cos(math.radians(ayaw)))
        if abs(dx * a_perp[0] + dy * a_perp[1]) > (a.width / 2.0 + b.width / 2.0 + 0.5):
            return False
        a_s, a_e = _t_range(a); b_s, b_e = _t_range(b)
        if max(a_s, b_s) > min(a_e, b_e) + 1e-6: return False
        return True

    def _along_gap(a: Trajectory, b: Trajectory
                   ) -> Tuple[float, float, Tuple[float, float], Trajectory, Trajectory]:
        """Return (gap, required, h, ahead, behind) for compat pair (a, b)."""
        ax, ay, ayaw = _cur_pose(a); bx, by, byaw = _cur_pose(b)
        mean_yaw = (ayaw + byaw) / 2.0
        h = (math.cos(math.radians(mean_yaw)), math.sin(math.radians(mean_yaw)))
        a_along = ax * h[0] + ay * h[1]
        b_along = bx * h[0] + by * h[1]
        if a_along >= b_along: ahead, behind = a, b
        else: ahead, behind = b, a
        ahead_along = max(a_along, b_along)
        behind_along = min(a_along, b_along)
        gap = ahead_along - behind_along
        required = ahead.length / 2.0 + behind.length / 2.0 + STATIC_PAIR_MIN_CLEAR_M
        return gap, required, h, ahead, behind

    n_shifted = 0
    for _ in range(50):  # safety bound — chains rarely need more than n−1 iters
        # Find tightest tight pair
        worst = None  # (deficit, a, b)
        for i in range(len(statics)):
            a = statics[i]
            if a.remove: continue
            for j in range(i + 1, len(statics)):
                b = statics[j]
                if b.remove: continue
                if not _compat(a, b): continue
                gap, required, h, ahead, behind = _along_gap(a, b)
                deficit = required - gap
                if deficit > 0.01 and (worst is None or deficit > worst[0]):
                    worst = (deficit, a, b)
        if worst is None: break
        deficit, a, b = worst
        gap, required, h, ahead, behind = _along_gap(a, b)
        # Add a small epsilon so we exceed the required threshold and
        # don't sit on the SAT's "exactly 0 overlap with padding" edge.
        shift = deficit + 0.05
        cur = _cur_pose(behind)
        new_x = cur[0] - shift * h[0]
        new_y = cur[1] - shift * h[1]
        new_corners = _obb_corners(new_x, new_y, cur[2],
                                    behind.length / 2.0, behind.width / 2.0)
        # Don't collide with any other actor's existing OBB at any tick
        collides_with = None
        for other in trajs:
            if other.actor_id == behind.actor_id or other.remove: continue
            other_ts = other.times
            if other.truncate_after_idx is not None:
                other_ts = other_ts[:other.truncate_after_idx + 1]
            for ti, t_int in enumerate(other_ts):
                ox, oy, oyw = other.xs[ti], other.ys[ti], other.yaws[ti]
                if other.snap_pose is not None:
                    ox, oy, oyw = other.snap_pose
                # Skip checking against time slots where 'other' isn't alive
                # at any time overlapping 'behind' (statics are alive their
                # whole time range; this filters out time-disjoint clones
                # like V_30 vs V_76).
                bs, be = _t_range(behind)
                os = other_ts[0]  + other.time_offset_s
                oe = other_ts[-1] + other.time_offset_s
                if max(bs, os) > min(be, oe) + 1e-6:
                    break  # break inner loop, move to next actor
                oc = _obb_corners(ox, oy, oyw,
                                   other.length / 2.0, other.width / 2.0)
                if _obb_overlap(new_corners, oc, padding) is not None:
                    collides_with = other.actor_id; break
            if collides_with is not None: break
        if collides_with is not None:
            behind.fix_note = (behind.fix_note + "; " if behind.fix_note else "") + \
                              f"SKIP spacing shift {shift:.2f}m vs {ahead.actor_id} — would collide with {collides_with}"
            # Mark this pair as unresolvable to avoid infinite loop
            # by removing one of them from consideration in this routine
            # (we'll let the residual step REMOVE if needed)
            break
        behind.snap_pose = (new_x, new_y, cur[2])
        behind.fix_note = (behind.fix_note + "; " if behind.fix_note else "") + \
                          f"space-shift {shift:.2f}m backward (vs {ahead.actor_id}, gap was {gap:.2f}m, required ≥ {required:.2f}m)"
        n_shifted += 1
    return n_shifted


# ---------------------------------------------------------------------------
# Cluster 2: tail-truncate
# ---------------------------------------------------------------------------

def _tail_truncate(b: Trajectory, a: Trajectory, tick_dt: float, padding: float
                   ) -> bool:
    """If B's collision with A is contained to a contiguous tail of B's
    trajectory, find the latest index where B no longer overlaps A and
    truncate B after that. Returns True if applied, False if not
    applicable (collision not tail-shaped)."""
    # Compute per-waypoint of B whether overlap with A exists at that real-world time
    b_ts = b.times
    if b.truncate_after_idx is not None: b_ts = b_ts[:b.truncate_after_idx + 1]
    if not b_ts: return False
    overlap_flags = []
    for i, t_int in enumerate(b_ts):
        real_t = t_int + b.time_offset_s
        pa = _interpolate(a, real_t)
        if pa is None:
            overlap_flags.append(False); continue
        cb = _obb_corners(b.xs[i], b.ys[i], b.yaws[i],
                          b.length / 2.0, b.width / 2.0)
        ca = _obb_corners(*pa, a.length / 2.0, a.width / 2.0)
        d = _obb_overlap(ca, cb, padding)
        overlap_flags.append(d is not None)
    # Tail-shaped if there's a final contiguous run of True after the last False
    if not any(overlap_flags): return False
    last_false = -1
    for i, f in enumerate(overlap_flags):
        if not f: last_false = i
    if last_false < 0:
        # All collide → can't truncate, would empty the trajectory
        return False
    if last_false == len(overlap_flags) - 1:
        # No tail collision
        return False
    # Truncate after the last clean index
    b.truncate_after_idx = last_false
    b.fix_note = (b.fix_note + "; " if b.fix_note else "") + \
                 f"truncate tail at idx {last_false}/{len(b_ts) - 1} (was colliding with {a.actor_id})"
    return True


# ---------------------------------------------------------------------------
# Cluster 3: static snap or remove (uses CARLA xodr)
# ---------------------------------------------------------------------------

class _CarlaMap:
    """Thin wrapper around offline carla.Map for nearest-lane queries."""

    def __init__(self, xodr_path: Path, egg_path: Path):
        if not xodr_path.exists():
            self.map = None; self.carla = None; return
        if not egg_path.exists():
            self.map = None; self.carla = None; return
        sys.path.insert(0, str(egg_path))
        try:
            import carla  # noqa: F401
            self.carla = carla
        except Exception:
            self.map = None; self.carla = None; return
        try:
            old_fd = os.dup(2); dn = os.open(os.devnull, os.O_WRONLY)
            os.dup2(dn, 2)
            self.map = self.carla.Map("ucla_v2", xodr_path.read_text())
            os.dup2(old_fd, 2); os.close(dn); os.close(old_fd)
        except Exception:
            self.map = None

    def ok(self) -> bool:
        return self.map is not None

    def nearest_parking_or_shoulder(self, x: float, y: float
                                    ) -> Optional[Tuple[float, float, float]]:
        if not self.ok(): return None
        loc = self.carla.Location(x=x, y=y, z=0.0)
        best = None; best_d = math.inf
        for lt_name in ("Parking", "Shoulder"):
            lt = getattr(self.carla.LaneType, lt_name, None)
            if lt is None: continue
            try:
                wp = self.map.get_waypoint(loc, project_to_road=True, lane_type=lt)
            except Exception:
                continue
            if wp is None: continue
            d = math.hypot(wp.transform.location.x - x,
                           wp.transform.location.y - y)
            if d < best_d:
                best_d = d
                best = (wp.transform.location.x, wp.transform.location.y,
                        wp.transform.rotation.yaw)
        if best is None or best_d > SNAP_NONDRIVING_MAX_M: return None
        return best


def _snap_static(static: Trajectory, other_npc: Trajectory,
                 carla_map: Optional[_CarlaMap],
                 all_trajs: List[Trajectory],
                 tick_dt: float, padding: float) -> None:
    """Attempt to snap a static-vs-NPC pair by moving the static to nearest
    Parking/Shoulder. If snapping introduces a new collision (with anyone),
    REMOVE the static."""
    if carla_map is None or not carla_map.ok():
        static.remove = True
        static.fix_note = (static.fix_note + "; " if static.fix_note else "") + \
                          f"REMOVE (no xodr available; was colliding with {other_npc.actor_id})"
        return
    spot = carla_map.nearest_parking_or_shoulder(static.xs[0], static.ys[0])
    if spot is None:
        static.remove = True
        static.fix_note = (static.fix_note + "; " if static.fix_note else "") + \
                          f"REMOVE (no Parking/Shoulder within {SNAP_NONDRIVING_MAX_M}m; was colliding with {other_npc.actor_id})"
        return
    sx, sy, sy_yaw = spot
    # Preserve the static's stored yaw rather than overwriting with lane yaw
    # — parking lanes can be in either direction, and the original yaw came
    # from perception.
    static.snap_pose = (sx, sy, static.yaws[0])
    # Collision-check the snap pose against all other actors at every tick
    static_corners = _obb_corners(sx, sy, static.yaws[0],
                                  static.length / 2.0, static.width / 2.0)
    bad = None
    for other in all_trajs:
        if other.actor_id == static.actor_id or other.remove: continue
        other_ts = other.times
        if other.truncate_after_idx is not None:
            other_ts = other_ts[:other.truncate_after_idx + 1]
        for i, t_int in enumerate(other_ts):
            real_t = t_int + other.time_offset_s
            ox, oy, oyw = other.xs[i], other.ys[i], other.yaws[i]
            if other.snap_pose is not None: ox, oy, oyw = other.snap_pose
            oc = _obb_corners(ox, oy, oyw, other.length / 2.0, other.width / 2.0)
            if _obb_overlap(static_corners, oc, padding) is not None:
                bad = other.actor_id; break
        if bad: break
    if bad:
        static.snap_pose = None
        static.remove = True
        static.fix_note = (static.fix_note + "; " if static.fix_note else "") + \
                          f"REMOVE (snap to ({sx:.1f},{sy:.1f}) collides with {bad})"
    else:
        static.fix_note = (static.fix_note + "; " if static.fix_note else "") + \
                          f"snap to ({sx:.1f},{sy:.1f}) (was colliding with {other_npc.actor_id})"


# ---------------------------------------------------------------------------
# XML write helpers — preserve byte-exact formatting via regex substitution
# ---------------------------------------------------------------------------

def _add_time_offset_xml(xml_path: Path, dt: float) -> int:
    """Add ``dt`` to every waypoint's ``time`` attribute."""
    text = xml_path.read_text(encoding="utf-8")
    n = [0]
    def sub_t(m: re.Match) -> str:
        old = float(m.group(2))
        new = f"{old + dt:.6f}"
        n[0] += 1
        return m.group(1) + new + m.group(3)
    new_text = re.sub(r'(\btime=")(-?\d+(?:\.\d+)?)(")', sub_t, text)
    xml_path.write_text(new_text, encoding="utf-8")
    return n[0]


def _truncate_waypoints_xml(xml_path: Path, keep_up_to_idx: int) -> int:
    """Keep waypoints with index <= keep_up_to_idx, drop the rest.
    Returns the number of waypoints removed."""
    text = xml_path.read_text(encoding="utf-8")
    # Iterate through every `<waypoint .../>` element and keep only the
    # first (keep_up_to_idx+1) ones.
    pattern = re.compile(r'\n?\s*<waypoint\b[^/]*?/>', flags=re.DOTALL)
    kept = [0]
    n_removed = [0]
    def sub_wp(m: re.Match) -> str:
        if kept[0] <= keep_up_to_idx:
            kept[0] += 1
            return m.group(0)
        n_removed[0] += 1
        return ""
    new_text = pattern.sub(sub_wp, text)
    xml_path.write_text(new_text, encoding="utf-8")
    return n_removed[0]


def _snap_static_xml(xml_path: Path, new_x: float, new_y: float, new_yaw: float
                     ) -> int:
    """Rewrite every waypoint's x/y/yaw to the snap pose (static actor)."""
    text = xml_path.read_text(encoding="utf-8")
    x_s   = f"{new_x:.6f}"
    y_s   = f"{new_y:.6f}"
    yaw_s = f"{_normalize_yaw(new_yaw):.6f}"
    n = [0]
    def sub_wp(m: re.Match) -> str:
        block = m.group(0)
        b = re.sub(r'(\bx=")(-?\d+(?:\.\d+)?)(")', r'\g<1>' + x_s + r'\g<3>',
                   block, count=1)
        b = re.sub(r'(\by=")(-?\d+(?:\.\d+)?)(")', r'\g<1>' + y_s + r'\g<3>',
                   b, count=1)
        b = re.sub(r'(\byaw=")(-?\d+(?:\.\d+)?)(")', r'\g<1>' + yaw_s + r'\g<3>',
                   b, count=1)
        if b != block: n[0] += 1
        return b
    new_text = re.sub(r'<waypoint\b[^/]*?/>', sub_wp, text, flags=re.DOTALL)
    xml_path.write_text(new_text, encoding="utf-8")
    return n[0]


def _remove_xml(xml_path: Path, scene_dir: Path) -> bool:
    manifest_path = scene_dir / "actors_manifest.json"
    fname = xml_path.name
    try:
        rel = str(xml_path.resolve().relative_to(scene_dir.resolve()))
    except Exception:
        rel = fname
    candidates = {fname, rel, rel.replace("\\", "/")}
    try:
        xml_path.unlink()
    except FileNotFoundError:
        return False
    if manifest_path.exists():
        try:
            mf = json.loads(manifest_path.read_text())
            changed = False
            for kind in ("ego", "npc", "static", "walker", "cyclist"):
                lst = mf.get(kind) or []
                new = [e for e in lst
                       if not (isinstance(e, dict) and str(e.get("file")) in candidates)]
                if len(new) != len(lst):
                    mf[kind] = new; changed = True
            if changed: manifest_path.write_text(json.dumps(mf, indent=2))
        except Exception:
            pass
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--apply", action="store_true",
                    help="Actually write XMLs (default: dry-run)")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--tick-dt", type=float, default=TICK_DT_S)
    ap.add_argument("--padding", type=float, default=COLLISION_PADDING_M)
    ap.add_argument("--max-stagger-s", type=float, default=DEFAULT_MAX_STAGGER_S)
    ap.add_argument("--xodr", type=Path, default=DEFAULT_XODR)
    ap.add_argument("--carla-egg", type=Path, default=DEFAULT_CARLA_EGG)
    args = ap.parse_args()

    if not args.scene.is_dir():
        print(f"ERROR: scene not found: {args.scene}", file=sys.stderr); sys.exit(2)

    # Load
    trajs: List[Trajectory] = []
    for p in sorted(glob.glob(str(args.scene / "**/*.xml"), recursive=True)):
        if any(p.endswith(s) for s in (".preflip", ".presnap", ".predelete",
                                        "_REPLAY.xml", "_FIXED.xml")):
            continue
        t = _parse_xml(Path(p))
        if t is None: continue
        if t.role in ("walker", "cyclist"): continue  # different model, skip
        trajs.append(t)
    print(f"Loaded {len(trajs)} vehicle trajectories from {args.scene}")

    # ── Audit BEFORE ────────────────────────────────────────────────────
    before = _audit(trajs, args.tick_dt, args.padding)
    above = [p for p in before if not _is_subnoise(p)]
    print(f"\n[audit BEFORE]  collisions above sub-noise threshold: {len(above)} "
          f"(of {len(before)} total)")
    for p in above:
        print(f"  {p.a:<42} ↔ {p.b:<42} ticks={p.n_ticks:>3} peak={p.peak:.2f}m  "
              f"t={p.first_t:.1f}-{p.last_t:.1f}  {p.role_a}/{p.role_b}")

    # ── Carla map for static snap path ───────────────────────────────────
    carla_map = _CarlaMap(args.xodr, args.carla_egg)
    if carla_map.ok():
        print(f"[xodr] loaded {args.xodr} for static-snap queries")
    else:
        print(f"[xodr] WARNING: could not load — static-snap path will REMOVE instead")

    # ── Step 0: space out tight static pairs (parked cars bumper-to-bumper)
    n_static_shifted = _space_static_pairs(trajs, args.tick_dt, args.padding)
    if n_static_shifted:
        print(f"\n[step 0] static-pair spacing: {n_static_shifted} actor(s) shifted backward along lane")

    # ── Step 1: stagger identical-spawn clusters ────────────────────────
    clusters = _group_identical_spawn(trajs)
    if clusters:
        print(f"\n[step 1] identical-spawn clusters: {len(clusters)}")
        for cl in clusters:
            ids = ", ".join(c.actor_id for c in cl)
            print(f"  cluster (n={len(cl)}) @ ({cl[0].xs[0]:.2f},{cl[0].ys[0]:.2f},{cl[0].yaws[0]:+.1f}°): {ids}")
        for cl in clusters:
            others = [t for t in trajs if t not in cl and not t.remove]
            _stagger_cluster(cl, others, args.max_stagger_s,
                             args.tick_dt, args.padding)

    # ── Step 2: static-in-lane snap or remove ───────────────────────────
    # Re-audit after step 1 so we only act on remaining static/npc collisions
    mid = _audit(trajs, args.tick_dt, args.padding)
    handled_statics = set()
    for p in mid:
        if _is_subnoise(p): continue
        # Find the static side
        a = next(t for t in trajs if t.actor_id == p.a)
        b = next(t for t in trajs if t.actor_id == p.b)
        s = None; n = None
        if a.role == "static" and b.role != "static": s, n = a, b
        elif b.role == "static" and a.role != "static": s, n = b, a
        if s is None: continue
        if s.actor_id in handled_statics: continue
        _snap_static(s, n, carla_map, trajs, args.tick_dt, args.padding)
        handled_statics.add(s.actor_id)
    if handled_statics:
        print(f"\n[step 2] static-vs-NPC: {len(handled_statics)} static(s) snapped or removed")

    # ── Step 3: tail truncate the remaining ─────────────────────────────
    # Ego is never the truncate target — it's immutable. For ego-vs-X
    # collisions, force X to be the truncate target. For non-ego pairs,
    # pick the actor with the later end-of-trajectory (more truncatable
    # tail).
    mid2 = _audit(trajs, args.tick_dt, args.padding)
    handled_trunc = 0
    for p in mid2:
        if _is_subnoise(p): continue
        a = next(t for t in trajs if t.actor_id == p.a)
        b = next(t for t in trajs if t.actor_id == p.b)
        if a.role == "ego" and b.role != "ego":
            target, leader = b, a
        elif b.role == "ego" and a.role != "ego":
            target, leader = a, b
        elif a.role == "ego" and b.role == "ego":
            continue  # ego-ego — handled by leaderboard, we don't touch
        else:
            a_end = (a.times[-1] if a.truncate_after_idx is None
                      else a.times[a.truncate_after_idx]) + a.time_offset_s
            b_end = (b.times[-1] if b.truncate_after_idx is None
                      else b.times[b.truncate_after_idx]) + b.time_offset_s
            target, leader = (a, b) if a_end >= b_end else (b, a)
        if _tail_truncate(target, leader, args.tick_dt, args.padding):
            handled_trunc += 1
    if handled_trunc:
        print(f"\n[step 3] tail truncations: {handled_trunc}")

    # ── Step 4: residual — anything still colliding, time-shift the
    # mutable side; if no Δt works, REMOVE it. Egos and previously-snapped
    # statics are immutable here too.
    mid3 = _audit(trajs, args.tick_dt, args.padding)
    handled_residual = 0
    for p in mid3:
        if _is_subnoise(p): continue
        a = next(t for t in trajs if t.actor_id == p.a)
        b = next(t for t in trajs if t.actor_id == p.b)
        # Pick the mutable side: never ego; otherwise prefer the one with
        # shorter arc length (less narrative loss).
        if a.role == "ego" and b.role != "ego":
            target = b
        elif b.role == "ego" and a.role != "ego":
            target = a
        elif a.role == "ego" and b.role == "ego":
            continue
        else:
            target = a if _arc_length(a) <= _arc_length(b) else b
        # The other actors (with their already-applied fixes) are the
        # constraint set.
        constraints = [t for t in trajs
                       if t.actor_id != target.actor_id and not t.remove]
        delta = _find_min_offset(target, constraints, args.max_stagger_s,
                                 STAGGER_STEP_S, args.tick_dt, args.padding)
        if delta is None:
            target.remove = True
            target.fix_note = (target.fix_note + "; " if target.fix_note else "") + \
                              f"REMOVE (residual collision with {b.actor_id if target is a else a.actor_id}, no Δt clears)"
        else:
            target.time_offset_s = (target.time_offset_s or 0.0) + delta
            target.fix_note = (target.fix_note + "; " if target.fix_note else "") + \
                              f"residual stagger Δt=+{delta:.1f}s"
        handled_residual += 1
    if handled_residual:
        print(f"\n[step 4] residual fixes: {handled_residual}")

    # ── Audit AFTER ─────────────────────────────────────────────────────
    after = _audit(trajs, args.tick_dt, args.padding)
    above_after = [p for p in after if not _is_subnoise(p)]
    print(f"\n[audit AFTER]   collisions above sub-noise threshold: {len(above_after)} "
          f"(of {len(after)} total)")
    for p in above_after:
        print(f"  REMAINS  {p.a:<42} ↔ {p.b:<42} ticks={p.n_ticks:>3} peak={p.peak:.2f}m  {p.role_a}/{p.role_b}")

    # ── Summary of decisions ────────────────────────────────────────────
    fixed = [t for t in trajs
             if t.time_offset_s != 0.0 or t.truncate_after_idx is not None
             or t.snap_pose is not None or t.remove]
    print(f"\n[decisions] {len(fixed)} actor(s) will be edited:")
    for t in fixed:
        flags = []
        if t.time_offset_s != 0.0: flags.append(f"Δt=+{t.time_offset_s:.1f}s")
        if t.truncate_after_idx is not None: flags.append(f"truncate@{t.truncate_after_idx}")
        if t.snap_pose is not None: flags.append(f"snap→({t.snap_pose[0]:.1f},{t.snap_pose[1]:.1f})")
        if t.remove: flags.append("REMOVE")
        print(f"  {t.actor_id:<48} {', '.join(flags)}  | {t.fix_note}")

    # ── Apply ───────────────────────────────────────────────────────────
    if args.apply:
        print(f"\n[apply] writing changes...")
        for t in fixed:
            if t.role == "ego":
                # Defence-in-depth: ego XMLs are immutable. If anything tried
                # to mark one fixed, refuse to write it.
                print(f"  SKIP    {t.actor_id} (ego — immutable)")
                continue
            xp = t.xml_path
            if t.remove:
                if not args.no_backup:
                    bak = xp.with_suffix(xp.suffix + ".predelete")
                    if not bak.exists(): shutil.copy2(xp, bak)
                _remove_xml(xp, args.scene)
                print(f"  REMOVED {t.actor_id}")
                continue
            if t.snap_pose is not None:
                if not args.no_backup:
                    bak = xp.with_suffix(xp.suffix + ".presnap")
                    if not bak.exists(): shutil.copy2(xp, bak)
                _snap_static_xml(xp, *t.snap_pose)
                print(f"  SNAP    {t.actor_id} -> {t.snap_pose}")
            if t.truncate_after_idx is not None:
                if not args.no_backup:
                    bak = xp.with_suffix(xp.suffix + ".pretrunc")
                    if not bak.exists(): shutil.copy2(xp, bak)
                n_drop = _truncate_waypoints_xml(xp, t.truncate_after_idx)
                print(f"  TRUNC   {t.actor_id}: dropped {n_drop} waypoints")
            if t.time_offset_s != 0.0:
                if not args.no_backup:
                    bak = xp.with_suffix(xp.suffix + ".prestagger")
                    if not bak.exists(): shutil.copy2(xp, bak)
                n_t = _add_time_offset_xml(xp, t.time_offset_s)
                print(f"  STAGGER {t.actor_id}: +{t.time_offset_s:.2f}s on {n_t} waypoints")

    # Post-apply verification
    if args.apply:
        # Re-parse from disk to confirm
        re_trajs: List[Trajectory] = []
        for p in sorted(glob.glob(str(args.scene / "**/*.xml"), recursive=True)):
            if any(p.endswith(s) for s in (".preflip", ".presnap", ".predelete",
                                            ".pretrunc", ".prestagger",
                                            "_REPLAY.xml", "_FIXED.xml")):
                continue
            t = _parse_xml(Path(p))
            if t is None or t.role in ("walker", "cyclist"): continue
            re_trajs.append(t)
        verify = _audit(re_trajs, args.tick_dt, args.padding)
        above_verify = [p for p in verify if not _is_subnoise(p)]
        print(f"\n[verify] post-write audit: {len(above_verify)} above sub-noise "
              f"(of {len(verify)})")
        for p in above_verify:
            print(f"  STILL  {p.a:<42} ↔ {p.b:<42} ticks={p.n_ticks} peak={p.peak:.2f}m")


if __name__ == "__main__":
    main()
