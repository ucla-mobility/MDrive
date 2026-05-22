#!/usr/bin/env python3
"""
fix_actor_yaw_reversal.py — flip 180°-reversed actor yaw in-place.

Many actors in ``scenarioset/v2xpnp/`` ship with a ``yaw`` attribute that is
exactly opposite to the actor's motion direction — an artifact of the
upstream v2x-real-pnp dataset's bounding-box labelling (boxes have no
inherent front/back; the labeller picked one of two equivalent
orientations). The v2xpnp pipeline propagates the labelled yaw straight
through ``yaw_v2x_to_carla()`` without reconciling against velocity, so
CARLA renders those vehicles driving in reverse.

This tool detects whole-track 180° reversals with high confidence and
flips them in-place. It is intentionally conservative: a track is only
flipped when ≥90 % of its moving waypoints disagree with motion direction
by >135°, ambiguous frames (45°–135°) are <10 %, and the wrong-way mean
is within 15° of a pure half-turn. Borderline tracks are reported but
never edited. The operation is idempotent: a flipped track no longer
matches the detection rule, so re-running is a no-op.

Coexists with the patch editor: any actor that already has a
``yaw_offset_deg`` or ``yaw_segment_offsets`` entry in the scene's
``_patch_editor.patch.json`` is skipped (manual user edits win).

Usage
-----
    # Dry-run audit across the whole scenarioset (default):
    python v2xpnp/scripts/fix_actor_yaw_reversal.py

    # Limit to one scene:
    python v2xpnp/scripts/fix_actor_yaw_reversal.py \\
        --scenes 2023-04-07-15-05-15_4_1

    # Actually apply (writes .preflip.xml backups next to each fix):
    python v2xpnp/scripts/fix_actor_yaw_reversal.py --apply

    # Whole-scenarioset apply with a JSON decisions report:
    python v2xpnp/scripts/fix_actor_yaw_reversal.py --apply \\
        --report /tmp/yaw-fix.json
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
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Detection thresholds — chosen conservatively to minimise false flips.
# Auditing the current scenarioset (5823 actors) shows these values isolate
# the 78 clean whole-track reversals from the 45 borderline / mid-track-
# reversal cases without touching any track whose stored yaw already
# tracks its motion direction.
# ---------------------------------------------------------------------------

MIN_ARC_M               = 1.0     # below this arc length, treat as static
MIN_NET_DISPLACEMENT_M  = 2.0     # net spawn→last displacement; below this
                                  # treat as static regardless of arc length.
                                  # Catches noisy-static actors like
                                  # Vehicle_28 (arc 13.8m but net 1.3m — sensor
                                  # jitter on a parked car, not real motion).
MIN_STEP_M              = 0.3     # per-step displacement that counts as "moving"
WRONG_THRESHOLD_DEG     = 135.0   # |diff| > this  → wrong-way frame
RIGHT_THRESHOLD_DEG     = 45.0    # |diff| < this  → right-way frame
MIN_SAMPLES             = 5       # require at least N moving frames
WRONG_RATIO_TO_FLIP     = 0.90    # n_wrong / (n_wrong + n_right) ≥ this
AMBIG_RATIO_MAX         = 0.10    # n_amb / total ≤ this
MEAN_WRONG_TO_180_MAX   = 15.0    # mean(|diff − 180°|) on wrong-way frames

# Verified-lane second pass (primary fallback for STATIC / TOO_FEW vehicles).
# We load CARLA's OpenDRIVE source offline via ``carla.Map(name, xodr_string)``
# — no server needed — and look up the nearest Driving lane for each static
# actor. The waypoint's ``rotation.yaw`` is the lane forward direction; an
# accompanying ``verified_lane_directions.json`` (produced by
# v2xpnp/scripts/build_verified_lane_directions.py) tells us which lanes were
# cross-validated against motion across the entire scenarioset and which to
# invert. Lanes flagged UNRELIABLE/UNDER_SAMPLED are skipped.
LANE_WP_MAX_LATERAL_M   = 3.0
LANE_EDGE_OFF_M         = 2.0  # An actor more than this far from the lane
                               # centerline (~lane_width/2 + a small margin)
                               # is materially outside the lane edge. Even if
                               # yaw-aligned with the lane, treat as off-lane
                               # so the snap path runs (default CARLA lane
                               # width is 3.5m → 1.75m half-width).
LANE_ALIGN_TOL_DEG      = 30.0
LANE_REV_TOL_DEG        = 30.0
DEFAULT_XODR = Path("v2xpnp/map/ucla_v2.xodr")
DEFAULT_VERIFIED_LANES = Path("v2xpnp/map/verified_lane_directions.json")
DEFAULT_CARLA_EGG = Path("carla912/PythonAPI/carla/dist/carla-0.9.12-py3.7-linux-x86_64.egg")

# Neighbour-motion fallback (used when the verified-lane pass returns
# UNRELIABLE / UNDER_SAMPLED / off-road). Borrows direction from
# neighbouring vehicles' velocity in the same scene.
NEIGHBOR_RADIUS_M       = 5.0
NEIGHBOR_MIN_SEGS       = 3
NEIGHBOR_CLUSTER_TOL    = 30.0
NEIGHBOR_ALIGN_TOL_DEG  = 30.0
NEIGHBOR_REV_TOL_DEG    = 30.0

REPLAY_SUFFIX = "_REPLAY.xml"
PATCH_SIDECAR_NAME = "_patch_editor.patch.json"

# Snap-or-remove pass (NEW). For static actors whose stored yaw is neither
# aligned nor anti-aligned with the lane direction — e.g. a parked car facing
# the curb perpendicular to the road, or an actor labeled completely off the
# road — propose either:
#   * SNAP: move (x, y) onto the lane centerline and set yaw to the derived
#     lane direction (picking forward or reverse so the rotational change
#     from the original yaw is minimised). Collision-check the snapped OBB
#     against every other actor at every frame; if clear → apply.
#   * REMOVE: collision check fails for both forward and reverse snap → the
#     actor would block traffic wherever it goes → delete the XML and the
#     manifest entry.
SNAP_MAX_LATERAL_M       = 12.0  # if lateral > this, try to snap (treat as off-lane).
NONDRIVING_NEAR_ROAD_MAX_M = 2.0 # OK_NONDRIVING_LANE only when the Parking/
                                 # Shoulder lane is within this distance of a
                                 # real Driving lane. Beyond this, the xodr
                                 # marker is a stranded "Shoulder" that
                                 # corresponds to grass/median in the
                                 # rendered map — treat as off-lane and snap.
UNRELIABLE_LATERAL_SNAP_M  = 2.0 # When the projected lane is UNRELIABLE,
                                 # still snap position when lateral > this
                                 # (direction comes from carla_yaw fallback +
                                 # _choose_snap_yaw to preserve actor facing).
STRONG_OFF_LANE_REMOVE_M   = 5.0 # Moving NPCs whose spawn pose is this far
                                 # from any Driving lane get REMOVED outright,
                                 # even if their trajectory eventually rejoins
                                 # the road. A vehicle that materializes 5+m
                                 # off the asphalt is visually broken.
                                 # Widened from 8m so actors parked on grass /
                                 # between sidewalk and driving lane (e.g.
                                 # Vehicle_58 at 8.25m off the road) enter the
                                 # snap-or-delete path. CARLA's roads + sidewalks
                                 # span ~8m together, so 12m comfortably covers
                                 # the grass margin without sweeping unrelated
                                 # actors on other roads into the snap.
SNAP_PERP_MIN_DEG        = 30.0  # |yaw-fwd| or |yaw-rev| > this → not "aligned"
SNAP_PERP_MAX_DEG        = 150.0 # |diff| > this → "reversed" (handled by FLIP, not SNAP)
DEFAULT_VEHICLE_LENGTH_M = 4.6
DEFAULT_VEHICLE_WIDTH_M  = 2.0
COLLISION_FRAME_STRIDE   = 4     # sample every Nth moving-actor waypoint
COLLISION_PADDING_M      = 0.05  # small margin so we don't false-positive on touching boxes

# Off-lane mover handling.
#
# Some actors have own-motion yaw that perfectly matches velocity (so the
# velocity pass returns OK), yet their spawn pose — and often the whole
# trajectory — sits off any driving lane. The blue tesla in the test scene
# (Vehicle_12) drives 100 m parallel to the road but on grass; its yaw is
# consistent, so the velocity check is satisfied. We need a separate
# lane-membership pass for these.
#
# Lateral-shifting the trajectory was considered and rejected: lanes curve,
# and a rigid offset that puts the actor on the lane at frame 0 drifts off
# again wherever the road bends. The robust choice is to REMOVE.
#
# Two cases handled:
#   * NOISY_STATIC — arc/net ratio > 3 AND net < 10 m. The track is
#     LiDAR-jitter on a parked car (e.g. Vehicle_38: arc 24 m, net 5.9 m).
#     Treat as static and run the existing snap-or-remove pass.
#   * OFF-LANE TRAJECTORY — frame-0 off lane by > LANE_WP_MAX_LATERAL_M
#     AND < OFF_LANE_ON_RATIO_OK of trajectory frames on a driving lane.
#     REMOVE the actor (and back up the XML to .predelete).
NOISY_STATIC_ARC_OVER_NET = 3.0   # path length 3× straight-line → likely jitter
NOISY_STATIC_NET_MAX_M    = 10.0  # but only if total displacement < 10 m;
                                  # protects legit U-turners (net > 10 m)
OFF_LANE_ON_RATIO_OK      = 0.95  # ≥ 95% of waypoints on a Driving lane → keep
TRUNC_MIN_ON_LANE_RATIO   = 0.50  # An NPC with at least this fraction of
                                  # contiguous on-lane waypoints is salvaged
                                  # via TRUNCATE_PREFIX / TRUNCATE_SUFFIX
                                  # rather than REMOVE'd. Drops the off-lane
                                  # prefix/suffix and keeps the on-lane core.
                                  # (Vehicle_14 has 95.4% — spawn-frame just
                                  # touches the curb; the actor is fine)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap180(deg: float) -> float:
    """Normalize ``deg`` into (-180, 180]."""
    return ((float(deg) + 180.0) % 360.0) - 180.0


def _normalize_yaw(deg: float) -> float:
    """Normalize ``deg`` into [-180, 180]. Used when writing back."""
    v = ((float(deg) + 180.0) % 360.0) - 180.0
    return 180.0 if v <= -180.0 else v


# ---------------------------------------------------------------------------
# OBB (oriented bounding box) collision check via Separating Axis Theorem.
# Each box is (cx, cy, half_length, half_width, yaw_deg) where length is
# along the box's forward axis (yaw direction) and width perpendicular.
# ---------------------------------------------------------------------------

OBB = Tuple[float, float, float, float, float]   # (cx, cy, hl, hw, yaw_deg)


def _obb_corners(box: OBB) -> List[Tuple[float, float]]:
    cx, cy, hl, hw, yaw = box
    a = math.radians(yaw)
    fx, fy = math.cos(a), math.sin(a)        # forward unit vector
    # Right vector (90° CW from forward, consistent with the box's local
    # right-handed frame regardless of canvas Y-flip).
    rx, ry = math.sin(a), -math.cos(a)
    return [
        (cx + hl * fx + hw * rx, cy + hl * fy + hw * ry),   # FR
        (cx + hl * fx - hw * rx, cy + hl * fy - hw * ry),   # FL
        (cx - hl * fx - hw * rx, cy - hl * fy - hw * ry),   # BL
        (cx - hl * fx + hw * rx, cy - hl * fy + hw * ry),   # BR
    ]


def _project(corners: List[Tuple[float, float]], ax: Tuple[float, float]
             ) -> Tuple[float, float]:
    vals = [p[0] * ax[0] + p[1] * ax[1] for p in corners]
    return min(vals), max(vals)


def _obb_overlap(box1: OBB, box2: OBB, padding: float = 0.0) -> bool:
    """SAT overlap test for two OBBs. ``padding`` inflates both boxes by
    that many metres on every side."""
    if padding != 0.0:
        box1 = (box1[0], box1[1], box1[2] + padding, box1[3] + padding, box1[4])
        box2 = (box2[0], box2[1], box2[2] + padding, box2[3] + padding, box2[4])
    # Quick AABB reject first — circle-radius-based.
    r1 = math.hypot(box1[2], box1[3])
    r2 = math.hypot(box2[2], box2[3])
    if (box1[0] - box2[0]) ** 2 + (box1[1] - box2[1]) ** 2 > (r1 + r2) ** 2:
        return False
    c1 = _obb_corners(box1)
    c2 = _obb_corners(box2)
    # Axes: forward + right of each box (4 total in 2D OBB-OBB).
    for box in (box1, box2):
        a = math.radians(box[4])
        for ax in ((math.cos(a), math.sin(a)),
                   (math.sin(a), -math.cos(a))):
            lo1, hi1 = _project(c1, ax)
            lo2, hi2 = _project(c2, ax)
            if hi1 < lo2 or hi2 < lo1:
                return False
    return True


# ---------------------------------------------------------------------------
# Per-actor analysis
# ---------------------------------------------------------------------------

@dataclass
class ActorDecision:
    xml_rel: str                    # relative to scene_dir
    role: str
    n_waypoints: int
    arc_m: float
    net_displacement_m: float        # ‖spawn − last waypoint‖ (straight-line)
    n_wrong: int
    n_right: int
    n_amb: int
    wrong_ratio: float              # n_wrong / (n_wrong + n_right)
    amb_ratio: float                # n_amb / total decisive
    wrong_mean_dev_from_180: float  # mean(|diff − 180°|) on wrong-way frames
    verdict: str                    # FLIP / FLIP_LANE / FLIP_NEIGHBOR / SNAP / REMOVE / OK / STATIC / WALKER / REVIEW / PATCHED / TOO_FEW / OFF_LANE
    note: str = ""
    # Position of the first waypoint — used for the lane second pass.
    spawn_x: float = 0.0
    spawn_y: float = 0.0
    spawn_yaw: float = 0.0
    # For SNAP verdicts: target pose ``(x, y, yaw_deg)`` to write back.
    snap_pose: Optional[Tuple[float, float, float]] = None


def _analyse_xml(xml_path: Path) -> Optional[ActorDecision]:
    """Read one route XML and compute its yaw-vs-motion stats. Returns None
    if the file is unparseable. Caller decides the final verdict (because
    some inputs — e.g. patch-sidecar / role checks — live outside the XML).
    """
    try:
        tree = ET.parse(xml_path)
    except Exception:
        return None
    root = tree.getroot()
    route = root.find("route") if root.tag != "route" else root
    if route is None:
        return None
    role = str(route.attrib.get("role", "?"))
    wps = route.findall("waypoint")
    pts: List[Tuple[float, float, float]] = []
    for wp in wps:
        try:
            pts.append((
                float(wp.attrib["x"]),
                float(wp.attrib["y"]),
                float(wp.attrib["yaw"]),
            ))
        except (KeyError, ValueError):
            continue
    # Single-waypoint XMLs (statics with a degenerate trajectory) still need
    # to be classified — they can be far off-lane and need to be REMOVE'd.
    # Skip only when there's literally nothing to read.
    if not pts:
        return None

    spawn_x, spawn_y, spawn_yaw = pts[0]
    last_x, last_y, _ = pts[-1]
    net_displacement = math.hypot(last_x - spawn_x, last_y - spawn_y)
    arc = 0.0
    n_wrong = n_right = n_amb = 0
    sum_dev_from_180 = 0.0
    for i in range(len(pts) - 1):
        dx = pts[i + 1][0] - pts[i][0]
        dy = pts[i + 1][1] - pts[i][1]
        step = math.hypot(dx, dy)
        arc += step
        if step < MIN_STEP_M:
            continue
        motion_yaw = math.degrees(math.atan2(dy, dx))
        diff = abs(_wrap180(pts[i][2] - motion_yaw))
        if diff > WRONG_THRESHOLD_DEG:
            n_wrong += 1
            sum_dev_from_180 += abs(diff - 180.0)
        elif diff < RIGHT_THRESHOLD_DEG:
            n_right += 1
        else:
            n_amb += 1

    decisive = n_wrong + n_right
    total = decisive + n_amb
    wrong_ratio = (n_wrong / decisive) if decisive > 0 else 0.0
    amb_ratio = (n_amb / total) if total > 0 else 0.0
    wrong_mean_dev = (sum_dev_from_180 / n_wrong) if n_wrong > 0 else 0.0

    result = ActorDecision(
        xml_rel="",  # filled in by caller
        role=role,
        n_waypoints=len(pts),
        arc_m=arc,
        net_displacement_m=net_displacement,
        n_wrong=n_wrong,
        n_right=n_right,
        n_amb=n_amb,
        wrong_ratio=wrong_ratio,
        amb_ratio=amb_ratio,
        wrong_mean_dev_from_180=wrong_mean_dev,
        verdict="OK",
        spawn_x=spawn_x,
        spawn_y=spawn_y,
        spawn_yaw=spawn_yaw,
    )
    # Carry the trajectory through to _classify (lane membership profile).
    # Not a dataclass field on purpose — asdict() would bloat the JSON
    # report. Side-channel attribute via setattr.
    result._pts = pts  # type: ignore[attr-defined]
    return result


# ---------------------------------------------------------------------------
# Verified-lane direction pass — uses CARLA's offline xodr + a per-lane
# motion-validated direction table.
# ---------------------------------------------------------------------------

class _LaneDirectionService:
    """Thin wrapper around an offline CARLA Map + verified-lanes JSON."""

    def __init__(self, xodr_path: Path, verified_path: Optional[Path]):
        self.carla_map = None
        self.carla = None
        self.verified: Dict[Tuple[int, int], dict] = {}
        if xodr_path.is_file():
            try:
                self._load_carla(xodr_path)
            except Exception as exc:
                print(f"[WARN] could not load xodr {xodr_path}: {exc}",
                      file=sys.stderr)
        else:
            print(f"[WARN] xodr not found at {xodr_path}; lane pass disabled",
                  file=sys.stderr)
        if verified_path is not None and verified_path.is_file():
            try:
                blob = json.loads(verified_path.read_text())
                for k, v in (blob.get("lanes") or {}).items():
                    try:
                        rid, lid = (int(x) for x in k.split(":"))
                        self.verified[(rid, lid)] = v
                    except Exception:
                        continue
                print(f"[lane-check] loaded {len(self.verified)} verified lane "
                      f"verdicts from {verified_path}")
            except Exception as exc:
                print(f"[WARN] couldn't parse verified-lanes JSON at "
                      f"{verified_path}: {exc}", file=sys.stderr)
        else:
            if verified_path is not None:
                print(f"[WARN] verified-lanes JSON not found at "
                      f"{verified_path}; without it, only TRUST_CARLA / "
                      f"UNDER_SAMPLED behaviour is available", file=sys.stderr)
        # Index verified lanes by road_id for sibling derivation.
        self.by_road: Dict[int, List[Tuple[int, dict]]] = {}
        for (rid, lid), v in self.verified.items():
            self.by_road.setdefault(rid, []).append((lid, v))

    def _load_carla(self, xodr_path: Path) -> None:
        # Add the bundled 0.9.12 egg if it exists; otherwise rely on whatever
        # ``import carla`` finds.
        # __file__ lives at v2xpnp/scripts/<this>.py — walk up three levels
        # to reach the repo root before joining the relative egg path.
        egg = (Path(__file__).resolve().parent.parent.parent / DEFAULT_CARLA_EGG).resolve()
        if egg.exists() and str(egg) not in sys.path:
            sys.path.insert(0, str(egg))
        import carla as _carla   # type: ignore
        xodr_str = xodr_path.read_text()
        # Suppress C-level stderr ('Traffic sign overlaps a driving lane'
        # warnings) during map load.
        old_fd = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        try:
            self.carla_map = _carla.Map(xodr_path.stem, xodr_str)
        finally:
            os.dup2(old_fd, 2)
            os.close(devnull)
            os.close(old_fd)
        self.carla = _carla
        print(f"[lane-check] loaded CARLA map from {xodr_path}")

    def is_available(self) -> bool:
        return self.carla_map is not None

    def _derive_from_siblings(self, road_id: int, lane_id: int
                              ) -> Optional[Tuple[float, str]]:
        """Compute the lane-forward direction for an UNDER_SAMPLED / unseen
        lane by polling sampled siblings on the same road. Returns
        ``(forward_yaw_deg, source_note)`` or None if there's no consensus.

        OpenDRIVE convention: lanes with the same lane_id sign travel the
        same physical direction (parallel one-side lanes); opposite-sign
        lanes travel in opposite directions (other side of the reference
        line). We use a sampled sibling's ``motion_forward_yaw_deg`` (the
        cross-validated true travel direction at that sibling) and apply
        a 180° flip if the under-sampled lane is on the opposite side.
        """
        siblings = self.by_road.get(road_id) or []
        votes: List[float] = []
        sources: List[str] = []
        for sib_lane_id, sib_v in siblings:
            if sib_lane_id == lane_id: continue
            if sib_v.get("verdict") not in ("TRUST_CARLA", "INVERT_CARLA"): continue
            sib_motion = sib_v.get("motion_forward_yaw_deg")
            if sib_motion is None: continue
            same_sign = (sib_lane_id > 0) == (lane_id > 0)
            derived = float(sib_motion) if same_sign else _wrap180(float(sib_motion) + 180.0)
            votes.append(derived)
            sources.append(f"lane {sib_lane_id} ({sib_v['verdict']}, {sib_v.get('n_samples',0)} samples)")
        if not votes:
            return None
        # Circular mean
        sx = sum(math.cos(math.radians(v)) for v in votes)
        sy = sum(math.sin(math.radians(v)) for v in votes)
        mean = math.degrees(math.atan2(sy, sx))
        max_dev = max(abs(_wrap180(v - mean)) for v in votes)
        if max_dev > 30.0:
            return None  # siblings disagree → no derivation
        return mean, "siblings: " + ", ".join(sources[:3])

    def lane_membership_profile(self, pts: List[Tuple[float, float, float]]
                                ) -> Tuple[int, int, float]:
        """For a trajectory, count how many waypoints sit on a Driving lane.

        Returns ``(n_on, n_off, first_off_lateral_m)``. ``first_off_lateral_m``
        is the distance from the *first* waypoint to its nearest Driving lane
        when frame 0 is off-lane (0.0 otherwise). Used to decide whether a
        moving NPC's spawn pose is acceptable, in tandem with on/off ratio.
        """
        if self.carla_map is None:
            return (0, 0, 0.0)
        n_on = n_off = 0
        first_off_lateral = 0.0
        for i, (x, y, _yaw) in enumerate(pts):
            loc = self.carla.Location(x=x, y=y, z=0.0)
            wp_strict = self.carla_map.get_waypoint(
                loc, project_to_road=False,
                lane_type=self.carla.LaneType.Driving,
            )
            if wp_strict is None:
                n_off += 1
                if i == 0:
                    wp_proj = self.carla_map.get_waypoint(
                        loc, project_to_road=True,
                        lane_type=self.carla.LaneType.Driving,
                    )
                    if wp_proj is not None:
                        first_off_lateral = math.hypot(
                            wp_proj.transform.location.x - x,
                            wp_proj.transform.location.y - y,
                        )
            else:
                n_on += 1
        return (n_on, n_off, first_off_lateral)

    def _nearest_nondriving_lane(self, x: float, y: float):
        """Return the nearest Parking/Sidewalk/Shoulder/Median waypoint
        (whichever is closest) along with its lateral distance. Used to
        detect actors that are parked legitimately on a non-driving
        feature — those should not be snapped onto a driving lane.
        """
        if self.carla_map is None or self.carla is None:
            return None, math.inf
        loc = self.carla.Location(x=x, y=y, z=0.0)
        best_wp = None; best_d = math.inf
        for lt_name in ("Parking", "Shoulder", "Sidewalk", "Median"):
            lt = getattr(self.carla.LaneType, lt_name, None)
            if lt is None: continue
            try:
                wp = self.carla_map.get_waypoint(
                    loc, project_to_road=True, lane_type=lt
                )
            except Exception:
                continue
            if wp is None: continue
            d = math.hypot(wp.transform.location.x - x,
                           wp.transform.location.y - y)
            if d < best_d:
                best_d, best_wp = d, wp
        return best_wp, best_d

    def classify(self, x: float, y: float, yaw_deg: float
                 ) -> Tuple[str, str, Optional[dict]]:
        """Return (verdict, note, snap_info).

        See module docstring for verdict descriptions. Adds:
          * ``OK_NONDRIVING_LANE`` — closest lane of any type is non-driving
                                     (sidewalk/parking/shoulder/median); the
                                     actor is legitimately parked off-road
                                     and should be left alone.
        """
        if self.carla_map is None:
            return "OFF_LANE", "xodr not loaded", None
        loc = self.carla.Location(x=x, y=y, z=0.0)
        wp = self.carla_map.get_waypoint(
            loc, project_to_road=True,
            lane_type=self.carla.LaneType.Driving,
        )
        if wp is None:
            return "OFF_LANE", "no driving lane via get_waypoint", None
        wpx = wp.transform.location.x
        wpy = wp.transform.location.y
        lateral = math.hypot(wpx - x, wpy - y)
        if lateral > SNAP_MAX_LATERAL_M:
            return "OFF_LANE", (
                f"projected lane is {lateral:.2f}m away "
                f"(> {SNAP_MAX_LATERAL_M:.1f}m)"
            ), None

        # Non-driving-lane guard: if the actor is sitting *directly on top*
        # of a Parking or Shoulder lane (≤ 0.7 m), it is legitimately
        # parked there — *as long as its yaw also matches the non-driving
        # lane's direction*. If the actor's yaw is 180° opposite the
        # parking-lane direction (e.g. Vehicle_28: facing into oncoming
        # traffic on a parking strip), we still need to flip its yaw.
        # Position stays the same; only the yaw rotates.
        #
        # Sidewalk and Median are *not* legitimate vehicle placements —
        # if the actor's closest non-driving feature is one of those,
        # fall through to the snap-or-remove path so the vehicle either
        # gets pulled onto the road or deleted.
        nd_wp, nd_lat = self._nearest_nondriving_lane(x, y)
        nd_type_ok = False
        if nd_wp is not None:
            nd_type_str = str(nd_wp.lane_type)
            nd_type_ok = ("Parking" in nd_type_str) or ("Shoulder" in nd_type_str)
        # Only trust an OK_NONDRIVING_LANE verdict when the Parking/Shoulder
        # lane is itself reasonably close to the road network. A "Shoulder"
        # that's 5+m from any Driving lane usually corresponds to a grass
        # strip in the rendered map (the xodr declares it as Shoulder but
        # there's no asphalt under it). For those, fall through to the
        # snap path so the actor gets pulled back onto the road.
        if (nd_wp is not None and nd_type_ok and nd_lat < 0.7
                and nd_lat < lateral
                and lateral <= NONDRIVING_NEAR_ROAD_MAX_M):
            nd_yaw = nd_wp.transform.rotation.yaw
            nd_diff = abs(_wrap180(yaw_deg - nd_yaw))
            base_note_nd = (
                f"sits on a {nd_wp.lane_type} lane at {nd_lat:.2f}m "
                f"(road {nd_wp.road_id} lane {nd_wp.lane_id})"
            )
            snap_info_nd = {
                "wpx": nd_wp.transform.location.x,
                "wpy": nd_wp.transform.location.y,
                "lateral": nd_lat,
                "lane_fwd": nd_yaw,
                "source": f"non-driving lane ({nd_wp.lane_type})",
                "road_id": nd_wp.road_id, "lane_id": nd_wp.lane_id,
            }
            if nd_diff < LANE_ALIGN_TOL_DEG:
                return "OK_NONDRIVING_LANE", (
                    base_note_nd + f"; yaw aligned (diff={nd_diff:.1f}°) — leave alone"
                ), None
            if abs(nd_diff - 180.0) < LANE_REV_TOL_DEG:
                # Actor's yaw is 180° opposite the parking/sidewalk lane
                # direction → in-place yaw flip (no snap, no remove).
                return "FLIP_LANE_PRIOR", (
                    base_note_nd +
                    f"; yaw is reversed wrt {nd_wp.lane_type} lane "
                    f"fwd {nd_yaw:+.1f}° (diff={nd_diff:.1f}° ~ 180°)"
                ), snap_info_nd
            # Yaw is neither aligned nor opposite (e.g. perpendicular
            # parking) — still trust the placement. Leave alone.
            return "OK_NONDRIVING_LANE", (
                base_note_nd + f"; yaw at {nd_diff:.1f}° from lane "
                f"(perpendicular parking) — leave alone"
            ), None

        carla_yaw = wp.transform.rotation.yaw

        # ── Resolve the lane's forward direction with the cascade ─────────
        key = (wp.road_id, wp.lane_id)
        verdict_info = self.verified.get(key)
        lane_fwd: Optional[float] = None
        source = ""
        unreliable_note = None
        if verdict_info is not None:
            vv = verdict_info.get("verdict", "UNDER_SAMPLED")
            if vv == "TRUST_CARLA":
                lane_fwd, source = carla_yaw, "verified TRUST_CARLA"
            elif vv == "INVERT_CARLA":
                lane_fwd, source = _wrap180(carla_yaw + 180.0), "verified INVERT_CARLA"
            elif vv == "UNRELIABLE":
                # UNRELIABLE lanes (intersection connectors, turn lanes,
                # mixed-motion segments) don't give us a per-lane direction,
                # but sibling lanes on the same road often *do*. Fall
                # through to sibling derivation; if siblings agree we can
                # still use that direction. Otherwise we'll skip below.
                unreliable_note = (
                    f"road {wp.road_id} lane {wp.lane_id}: UNRELIABLE "
                    f"({verdict_info.get('n_agree')} agree, "
                    f"{verdict_info.get('n_oppose')} opp, "
                    f"{verdict_info.get('n_other')} other) — trying siblings"
                )
            # else UNDER_SAMPLED → fall through to derivation
        if lane_fwd is None:
            derived = self._derive_from_siblings(wp.road_id, wp.lane_id)
            if derived is not None:
                lane_fwd, source = derived
        if lane_fwd is None:
            # No usable direction. For UNRELIABLE lanes the prior is
            # meaningless for *direction* — but if the actor is materially
            # off-lane (lateral > LANE_WP_MAX_LATERAL_M), the *position* is
            # still wrong and worth snapping. Use carla_yaw as a position-
            # snap reference; _choose_snap_yaw will pick whichever side of
            # 180° is closer to the actor's current yaw, so the visual
            # direction is preserved.
            if unreliable_note is not None:
                if lateral > UNRELIABLE_LATERAL_SNAP_M:
                    return "SNAP_OFFLANE", (
                        unreliable_note +
                        f"; lateral={lateral:.2f}m > {UNRELIABLE_LATERAL_SNAP_M:.1f}m "
                        f"— snapping position despite direction uncertainty"
                    ), {
                        "wpx": wpx, "wpy": wpy, "lateral": lateral,
                        "lane_fwd": carla_yaw,
                        "source": "unreliable-position-snap (carla yaw fallback)",
                        "road_id": wp.road_id, "lane_id": wp.lane_id,
                    }
                return "LANE_UNRELIABLE", unreliable_note + "; no sibling consensus", {
                    "wpx": wpx, "wpy": wpy, "lateral": lateral,
                    "lane_fwd": None, "source": "unreliable",
                    "road_id": wp.road_id, "lane_id": wp.lane_id,
                }
            # No verified verdict + no siblings: 91% CARLA prior.
            lane_fwd = carla_yaw
            source = "carla 91% prior"
        elif unreliable_note is not None:
            source = source + " (UNRELIABLE lane, sibling-derived)"

        snap_info = {
            "wpx": wpx, "wpy": wpy, "lateral": lateral,
            "lane_fwd": lane_fwd, "source": source,
            "road_id": wp.road_id, "lane_id": wp.lane_id,
        }

        diff = abs(_wrap180(yaw_deg - lane_fwd))

        # Off-lane (but inside snap radius): candidate for snap regardless of yaw.
        # NB: threshold is LANE_EDGE_OFF_M (2.0m, just past the lane edge), not
        # LANE_WP_MAX_LATERAL_M (3.0m). A yaw-aligned actor 2-3m off the lane
        # is still in grass/shoulder; "yaw-aligned but laterally off" is the
        # exact failure mode we hit on Vehicle_15 (sat on a Shoulder lane that
        # was itself 2.5m from the road).
        if lateral > LANE_EDGE_OFF_M:
            return "SNAP_OFFLANE", (
                f"projected lane is {lateral:.2f}m away "
                f"(> {LANE_EDGE_OFF_M:.1f}m); lane fwd {lane_fwd:+.1f}° "
                f"({source}, road {wp.road_id} lane {wp.lane_id})"
            ), snap_info

        if diff < LANE_ALIGN_TOL_DEG:
            return "OK_LANE", (
                f"aligned with lane fwd {lane_fwd:+.1f}° "
                f"(road {wp.road_id} lane {wp.lane_id}, {source}, "
                f"diff={diff:.1f}°)"
            ), snap_info
        if abs(diff - 180.0) < LANE_REV_TOL_DEG:
            # Distinguish source to mark confidence in the verdict tag.
            if source.startswith("verified"):
                tag = "FLIP_LANE_VERIFIED"
            elif source.startswith("siblings"):
                tag = "FLIP_LANE_DERIVED"
            else:
                tag = "FLIP_LANE_PRIOR"
            return tag, (
                f"opposite of lane fwd {lane_fwd:+.1f}° "
                f"(road {wp.road_id} lane {wp.lane_id}, {source}, "
                f"diff={diff:.1f}° ~ 180°)"
            ), snap_info
        # Perpendicular-ish.
        return "SNAP_PERP", (
            f"lane fwd {lane_fwd:+.1f}°, actor {yaw_deg:+.1f}°, "
            f"diff {diff:.1f}° (perpendicular-ish); "
            f"road {wp.road_id} lane {wp.lane_id} ({source})"
        ), snap_info


# ---------------------------------------------------------------------------
# Motion-consensus second pass.
#
# A motion segment is a single per-step ``(xa, ya, xb, yb, heading_deg)``
# extracted from any vehicle whose step length is ≥ MIN_STEP_M. Per-scene
# we gather all segments, then for each static actor we look at the
# segments within NEIGHBOR_RADIUS_M of the actor's spawn point, cluster
# them by direction modulo 180°, and treat the dominant cluster's mean
# heading as the road's "true" travel direction at that point. The static
# actor's stored yaw is compared to that consensus.
# ---------------------------------------------------------------------------

MotionSeg = Tuple[float, float, float, float, float]   # (xa, ya, xb, yb, heading_deg)


def _collect_motion_segments(scene_dir: Path) -> List[MotionSeg]:
    """Gather every ≥ MIN_STEP_M motion step from every non-walker XML in
    ``scene_dir``. Walkers/cyclists are excluded — pedestrian motion is too
    chaotic to encode road direction reliably.
    """
    out: List[MotionSeg] = []
    for xml_path in _iter_scene_xmls(scene_dir):
        if _is_walker_xml(xml_path, ""):     # path-based walker detection
            continue
        try:
            tree = ET.parse(xml_path)
        except Exception:
            continue
        root = tree.getroot()
        route = root.find("route") if root.tag != "route" else root
        if route is None:
            continue
        # Skip walkers by role too
        if (route.attrib.get("role") or "").lower() in ("walker", "cyclist"):
            continue
        wps = route.findall("waypoint")
        for i in range(len(wps) - 1):
            try:
                xa, ya = float(wps[i].attrib["x"]), float(wps[i].attrib["y"])
                xb, yb = float(wps[i + 1].attrib["x"]), float(wps[i + 1].attrib["y"])
            except (KeyError, ValueError):
                continue
            dx, dy = xb - xa, yb - ya
            if math.hypot(dx, dy) < MIN_STEP_M:
                continue
            out.append((xa, ya, xb, yb, math.degrees(math.atan2(dy, dx))))
    return out


def _motion_consensus_at(px: float, py: float, segs: List[MotionSeg]
                         ) -> Tuple[Optional[float], int]:
    """Return ``(consensus_heading_deg, n_segs)`` for the motion segments
    within ``NEIGHBOR_RADIUS_M`` of ``(px, py)``. Returns ``(None, 0)`` if
    no cluster of size ≥ NEIGHBOR_MIN_SEGS is found.
    """
    nearby: List[float] = []
    R2 = NEIGHBOR_RADIUS_M * NEIGHBOR_RADIUS_M
    for xa, ya, xb, yb, h in segs:
        dxs, dys = xb - xa, yb - ya
        L2 = dxs * dxs + dys * dys
        if L2 < 1e-9:
            continue
        t = max(0.0, min(1.0, ((px - xa) * dxs + (py - ya) * dys) / L2))
        cx, cy = xa + t * dxs, ya + t * dys
        if (px - cx) ** 2 + (py - cy) ** 2 < R2:
            nearby.append(h)
    if len(nearby) < NEIGHBOR_MIN_SEGS:
        return None, 0
    # Find the largest cluster of mutually-similar headings (within tolerance).
    # Direction is sign-sensitive here — we explicitly want to distinguish
    # north-bound from south-bound traffic on the same road.
    best_cluster: List[float] = []
    for h0 in nearby:
        cluster = [h for h in nearby
                   if abs(_wrap180(h - h0)) < NEIGHBOR_CLUSTER_TOL]
        if len(cluster) > len(best_cluster):
            best_cluster = cluster
    if len(best_cluster) < NEIGHBOR_MIN_SEGS:
        return None, 0
    sx = sum(math.cos(math.radians(h)) for h in best_cluster)
    sy = sum(math.sin(math.radians(h)) for h in best_cluster)
    return math.degrees(math.atan2(sy, sx)), len(best_cluster)


def _neighbor_classify(x: float, y: float, yaw_deg: float,
                       segs: List[MotionSeg]) -> Tuple[str, str]:
    """Decide whether ``yaw_deg`` agrees with the motion direction of
    vehicles passing within NEIGHBOR_RADIUS_M.

    Returns ``(verdict, note)`` ∈ {OK_NEIGHBOR, FLIP_NEIGHBOR, NO_NEIGHBORS,
    AMBIG_NEIGHBOR}.
    """
    consensus, n = _motion_consensus_at(x, y, segs)
    if consensus is None:
        return "NO_NEIGHBORS", "no vehicle motion within {:.1f}m".format(NEIGHBOR_RADIUS_M)
    diff = abs(_wrap180(yaw_deg - consensus))
    if diff < NEIGHBOR_ALIGN_TOL_DEG:
        return "OK_NEIGHBOR", (
            f"aligned with traffic at {consensus:+.1f}° (n={n}, diff={diff:.1f}°)"
        )
    if abs(diff - 180.0) < NEIGHBOR_REV_TOL_DEG:
        return "FLIP_NEIGHBOR", (
            f"opposite of traffic at {consensus:+.1f}° (n={n}, diff={diff:.1f}° ~ 180°)"
        )
    return "AMBIG_NEIGHBOR", (
        f"traffic at {consensus:+.1f}° (n={n}), actor at {yaw_deg:+.1f}°, diff={diff:.1f}°"
    )


# ---------------------------------------------------------------------------
# Per-scene OBB index — used to collision-check snap-candidate poses.
# ---------------------------------------------------------------------------

@dataclass
class SceneOBBs:
    """All actor OBBs in a scene, keyed by source XML for exclusion."""
    static: List[Tuple[str, OBB]] = field(default_factory=list)
    moving: List[Tuple[str, OBB]] = field(default_factory=list)


def _actor_dims_for_role(role: str) -> Tuple[float, float]:
    """Return (length_m, width_m) for an actor role. The dataset's
    manifest doesn't always reliably carry dims; use generous defaults.
    Walkers/cyclists are slimmer."""
    r = (role or "").lower()
    if r in ("walker",):
        return 0.6, 0.5
    if r in ("cyclist",):
        return 1.8, 0.7
    # vehicles / static / ego: pick a body that doesn't massively overshoot
    # but also won't falsely declare collision-free for clearly-blocked spots.
    return DEFAULT_VEHICLE_LENGTH_M, DEFAULT_VEHICLE_WIDTH_M


def _xml_role_and_arc(xml_path: Path) -> Tuple[str, float, List[Tuple[float, float, float]]]:
    """Return (role, arc_length_m, [(x,y,yaw) ...]) for an XML route."""
    try:
        tree = ET.parse(xml_path)
    except Exception:
        return "?", 0.0, []
    root = tree.getroot()
    route = root.find("route") if root.tag != "route" else root
    if route is None:
        return "?", 0.0, []
    role = str(route.attrib.get("role", "?"))
    pts: List[Tuple[float, float, float]] = []
    for wp in route.findall("waypoint"):
        try:
            pts.append((
                float(wp.attrib["x"]),
                float(wp.attrib["y"]),
                float(wp.attrib["yaw"]),
            ))
        except (KeyError, ValueError):
            continue
    arc = 0.0
    for i in range(len(pts) - 1):
        arc += math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
    return role, arc, pts


def _build_scene_obbs(scene_dir: Path,
                      stride: int = COLLISION_FRAME_STRIDE) -> SceneOBBs:
    """Pre-collect OBBs for every actor in ``scene_dir``. Static actors
    contribute a single OBB at their spawn; moving actors contribute a
    sampled OBB every ``stride`` waypoints (plus the last).

    Each entry is ``(xml_relpath, obb)``. The relpath lets the collision
    checker exclude the candidate's own OBBs.
    """
    scene = SceneOBBs()
    for xml_path in _iter_scene_xmls(scene_dir):
        role, arc, pts = _xml_role_and_arc(xml_path)
        if not pts: continue
        hl, hw = _actor_dims_for_role(role)
        hl /= 2.0
        hw /= 2.0
        rel = str(xml_path.relative_to(scene_dir))
        if arc < MIN_ARC_M:
            x, y, yaw = pts[0]
            scene.static.append((rel, (x, y, hl, hw, yaw)))
        else:
            n = len(pts)
            for i in range(0, n, max(1, stride)):
                x, y, yaw = pts[i]
                scene.moving.append((rel, (x, y, hl, hw, yaw)))
            # Always include the last waypoint so end-of-route position is covered.
            if (n - 1) % max(1, stride) != 0 and n >= 1:
                x, y, yaw = pts[-1]
                scene.moving.append((rel, (x, y, hl, hw, yaw)))
    return scene


def _snap_collides(snap: OBB, scene: SceneOBBs, exclude_rel: str) -> Optional[str]:
    """Check whether ``snap`` overlaps any non-self OBB in ``scene``.
    Returns the colliding actor's relpath, or None if clear."""
    for rel, obb in scene.static:
        if rel == exclude_rel: continue
        if _obb_overlap(snap, obb, padding=COLLISION_PADDING_M):
            return rel
    for rel, obb in scene.moving:
        if rel == exclude_rel: continue
        if _obb_overlap(snap, obb, padding=COLLISION_PADDING_M):
            return rel
    return None


def _choose_snap_yaw(actor_yaw: float, lane_fwd: float) -> float:
    """Snap yaw is always the lane's forward direction at this point.

    Earlier versions of this function picked whichever of ``lane_fwd`` or
    ``lane_fwd + 180°`` was closer to ``actor_yaw`` — but on a typical
    two-way road, each lane has only ONE valid travel direction (the
    opposite direction belongs to a different lane on the other side of
    the reference line). The "closer of two" heuristic would happily snap
    an actor parked perpendicular to a lane to face *against* traffic on
    that lane, which the user (correctly) flagged as a regression.

    For perpendicular cases, the actor is being repositioned anyway; the
    rotational change is large regardless. Choosing the correct lane
    direction is what matters."""
    return _normalize_yaw(lane_fwd)


# ---------------------------------------------------------------------------
# XML write helpers
# ---------------------------------------------------------------------------

_WAYPOINT_XYZ_YAW_RE = re.compile(
    r'(<waypoint\b[^>]*?)(?: x="(-?\d+(?:\.\d+)?)")?(?: y="(-?\d+(?:\.\d+)?)")?(?: yaw="(-?\d+(?:\.\d+)?)")?'
)


def _snap_xml_pose(xml_path: Path, new_x: float, new_y: float, new_yaw: float) -> int:
    """Rewrite every waypoint in ``xml_path`` to ``(new_x, new_y, new_yaw)``.
    Used for snap-to-lane on STATIC actors. Returns the number of waypoints
    rewritten. Preserves all other attributes (time, z, pitch, roll).

    Like the yaw-only flip, this uses regex substitution on the raw text so
    XML byte ordering / indentation is preserved exactly. The substitution
    is per-attribute (x= … y= … yaw= …) so non-affected attributes survive
    intact.
    """
    text = xml_path.read_text(encoding="utf-8")
    new_x_s = f"{new_x:.6f}"
    new_y_s = f"{new_y:.6f}"
    new_yaw_s = f"{_normalize_yaw(new_yaw):.6f}"
    n = 0

    # Match each <waypoint ...> element with its full attribute block and
    # update x/y/yaw individually inside that block.
    def _sub_wp(m: re.Match) -> str:
        nonlocal n
        block = m.group(0)
        # Use \g<N> backreferences (not \N) — bare \1 followed by a numeric
        # literal value would re-parse as group 19 etc. and raise.
        new_block = re.sub(r'(\bx=")(-?\d+(?:\.\d+)?)(")',
                           r'\g<1>' + new_x_s + r'\g<3>', block, count=1)
        new_block = re.sub(r'(\by=")(-?\d+(?:\.\d+)?)(")',
                           r'\g<1>' + new_y_s + r'\g<3>', new_block, count=1)
        new_block = re.sub(r'(\byaw=")(-?\d+(?:\.\d+)?)(")',
                           r'\g<1>' + new_yaw_s + r'\g<3>', new_block, count=1)
        if new_block != block:
            n += 1
        return new_block

    new_text = re.sub(r'<waypoint\b[^/]*?/>', _sub_wp, text, flags=re.DOTALL)
    if n > 0:
        xml_path.write_text(new_text, encoding="utf-8")
    return n


def _neighbor_yaw_consensus(px: float, py: float,
                            scene_obbs: Optional["SceneOBBs"],
                            exclude_rel: str = "",
                            radius_m: float = 8.0,
                            ) -> Optional[float]:
    """Modal yaw direction of vehicles within ``radius_m`` of ``(px, py)``.

    Used to disambiguate the 180° choice in ``_choose_snap_yaw`` for static
    snaps: when carla_yaw flips between adjacent opposing lanes, the actor's
    own (broken) yaw is a poor reference. Other actors already-placed on
    either side of the same street give us a robust direction signal.

    Returns ``None`` when fewer than 2 neighbours are nearby.
    """
    if scene_obbs is None:
        return None
    yaws: List[float] = []
    R2 = radius_m * radius_m
    for src_rel, (cx, cy, _hl, _hw, yaw_deg) in (scene_obbs.static + scene_obbs.moving):
        if src_rel == exclude_rel:
            continue
        if (cx - px) * (cx - px) + (cy - py) * (cy - py) > R2:
            continue
        yaws.append(yaw_deg)
    if len(yaws) < 2:
        return None
    # Project onto unit vectors so circular mean is correct.
    sx = sum(math.cos(math.radians(y)) for y in yaws)
    sy = sum(math.sin(math.radians(y)) for y in yaws)
    if (sx * sx + sy * sy) < 1e-3:
        return None  # too dispersed
    return math.degrees(math.atan2(sy, sx))


def _truncate_xml_to_indices(xml_path: Path, keep_first: int, keep_last: int) -> int:
    """Rewrite ``xml_path`` to keep only waypoints whose index is in
    ``[keep_first, keep_last]`` (inclusive). The retained waypoints have
    their ``time`` attribute re-based to start at 0.0 so the actor spawns
    at scenario start, not at the original waypoint's time. Returns the
    number of waypoints retained, or -1 on failure.

    Used to drop off-lane prefix/suffix from a moving NPC's trajectory
    while preserving the on-lane core. Backup is left to the caller (the
    standard ``.predelete`` / ``.pretrunc`` flow in main()).
    """
    try:
        tree = ET.parse(xml_path)
    except Exception:
        return -1
    root = tree.getroot()
    route = root.find("route") if root.tag != "route" else root
    if route is None:
        return -1
    waypoints = list(route.findall("waypoint"))
    if not waypoints or keep_first < 0 or keep_last >= len(waypoints) or keep_first > keep_last:
        return -1
    # Determine time rebase offset from the first retained waypoint.
    try:
        t0 = float(waypoints[keep_first].attrib.get("time", "0"))
    except ValueError:
        t0 = 0.0
    # Remove waypoints outside the kept range; rewrite time on the rest.
    for i, wp in enumerate(waypoints):
        if i < keep_first or i > keep_last:
            route.remove(wp)
        else:
            try:
                t = float(wp.attrib.get("time", "0"))
                wp.set("time", f"{(t - t0):.6f}")
            except ValueError:
                pass
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    return keep_last - keep_first + 1


def _longest_on_lane_run(pts: List[Tuple[float, float, float]],
                         lane_service: "_LaneDirectionService"
                         ) -> Tuple[int, int, int]:
    """Find the longest contiguous run of on-Driving-lane waypoints.
    Returns ``(start_idx, end_idx, length)``. Both indices are inclusive.
    Returns ``(-1, -1, 0)`` if no waypoint is on-lane.
    """
    if lane_service.carla_map is None:
        return (-1, -1, 0)
    best_s = best_e = -1
    best_len = 0
    cur_s = -1
    for i, (x, y, _yaw) in enumerate(pts):
        loc = lane_service.carla.Location(x=x, y=y, z=0.0)
        wp_strict = lane_service.carla_map.get_waypoint(
            loc, project_to_road=False,
            lane_type=lane_service.carla.LaneType.Driving,
        )
        if wp_strict is not None:
            if cur_s < 0:
                cur_s = i
            run_len = i - cur_s + 1
            if run_len > best_len:
                best_len, best_s, best_e = run_len, cur_s, i
        else:
            cur_s = -1
    return (best_s, best_e, best_len)


def _remove_actor_xml(xml_path: Path, scene_dir: Path) -> bool:
    """Delete ``xml_path`` and remove its entry from the scene's
    ``actors_manifest.json`` if present. Returns True on success."""
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
    # Best-effort manifest update.
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            changed = False
            for kind in ("ego", "npc", "static", "walker", "cyclist"):
                lst = manifest.get(kind) or []
                new = [e for e in lst
                       if not (isinstance(e, dict) and str(e.get("file")) in candidates)]
                if len(new) != len(lst):
                    manifest[kind] = new
                    changed = True
            if changed:
                manifest_path.write_text(json.dumps(manifest, indent=2))
        except Exception:
            pass
    return True


def _classify(d: ActorDecision, *, is_walker: bool, is_patched: bool,
              include_walkers: bool,
              motion_segs: Optional[List[MotionSeg]] = None,
              lane_service: Optional["_LaneDirectionService"] = None,
              scene_obbs: Optional[SceneOBBs] = None) -> None:
    """Mutate ``d.verdict`` (and ``d.note``) per the conservative rule set."""
    # ── Hard skip guards (apply regardless of velocity / lane outcome) ───
    if is_walker and not include_walkers:
        d.verdict = "WALKER"
        d.note = "walker/cyclist (use --include-walkers to consider)"
        return
    if is_patched:
        d.verdict = "PATCHED"
        d.note = "patch sidecar already overrides yaw on this actor"
        return

    # ── Pass 1: velocity-based detection (most authoritative for movers) ──
    # A track is "static" when EITHER:
    #   - total arc length < 1m (parked car, sensor caught no motion), OR
    #   - net spawn→last displacement < 2m (jitter without real translation —
    #     e.g. Vehicle_28-style: 13.8m of LiDAR jitter on a parked car that
    #     actually didn't go anywhere).
    # The net-displacement check catches actors that pass the velocity test
    # (own motion matches own yaw) but aren't actually going anywhere, so
    # lane-direction validation is appropriate.
    velocity_static  = (d.arc_m < MIN_ARC_M or
                        d.net_displacement_m < MIN_NET_DISPLACEMENT_M)
    # Noisy-static: the actor's arc length is large but its net displacement
    # is small AND arc/net is high. That signature is LiDAR jitter on a
    # parked car (e.g. Vehicle_38: arc 24 m, net 5.9 m, ratio 4.1). Treat
    # those as static so the lane-pass / snap-or-remove logic runs.
    if (not velocity_static
            and d.net_displacement_m < NOISY_STATIC_NET_MAX_M
            and d.net_displacement_m > 0
            and (d.arc_m / d.net_displacement_m) > NOISY_STATIC_ARC_OVER_NET):
        velocity_static = True
    velocity_too_few = (d.n_wrong + d.n_right + d.n_amb) < MIN_SAMPLES

    if not velocity_static and not velocity_too_few:
        # Spawn-grossly-off-lane is a hard REMOVE that overrides any yaw or
        # velocity verdict. Check BEFORE the FLIP/REVIEW branches return —
        # an actor 5+ m off the road shouldn't be salvaged by a yaw flip.
        if lane_service is not None and lane_service.is_available():
            _pts_check = getattr(d, "_pts", None) or []
            if _pts_check:
                _, _, _first_off_check = lane_service.lane_membership_profile(_pts_check)
                if _first_off_check > STRONG_OFF_LANE_REMOVE_M:
                    d.verdict = "REMOVE"
                    d.note = (
                        f"moving NPC spawn is grossly off-lane "
                        f"(spawn {_first_off_check:.2f}m > "
                        f"{STRONG_OFF_LANE_REMOVE_M:.1f}m from nearest Driving lane); "
                        f"cannot rigidly relocate a moving trajectory"
                    )
                    return
        if (
            d.wrong_ratio >= WRONG_RATIO_TO_FLIP
            and d.amb_ratio <= AMBIG_RATIO_MAX
            and d.wrong_mean_dev_from_180 <= MEAN_WRONG_TO_180_MAX
        ):
            d.verdict = "FLIP"
            d.note = (
                f"velocity: {d.n_wrong}/{d.n_wrong + d.n_right} wrong-way "
                f"({100*d.wrong_ratio:.0f}%), "
                f"amb={100*d.amb_ratio:.0f}%, "
                f"mean dev from 180°={d.wrong_mean_dev_from_180:.1f}°"
            )
            return
        if d.n_wrong >= 1 and d.wrong_ratio >= 0.50:
            d.verdict = "REVIEW"
            d.note = (
                f"velocity borderline: {d.n_wrong} wrong, {d.n_right} right, "
                f"{d.n_amb} amb; manual check required"
            )
            return
        # Velocity says the actor's own motion is consistent — but the
        # trajectory might still drive entirely off the driving lanes.
        # Cross-check lane membership: REMOVE the actor if frame 0 is off-lane
        # by > 3 m AND less than 95% of waypoints sit on a Driving lane.
        # Lateral-shifting was rejected (lanes curve → shift drifts off-lane
        # downstream), so the safe choice for unfixable trajectories is to
        # delete them.
        if lane_service is not None and lane_service.is_available():
            pts = getattr(d, "_pts", None) or []
            if pts:
                n_on, n_off, first_off = lane_service.lane_membership_profile(pts)
                total = n_on + n_off
                on_ratio = (n_on / total) if total > 0 else 1.0
                # REMOVE only when BOTH the spawn pose is meaningfully off-lane
                # (> 1.5 m) AND the trajectory as a whole is not predominantly
                # on driving lanes. Movers that spawn *on* a driving lane and
                # briefly cross off-lane mid-trajectory are usually showing
                # intersection / lane-change frames where the xodr doesn't
                # define a lane in that gap — they're legitimately driving,
                # not on grass.
                # Force-REMOVE when the spawn pose is grossly off-lane
                # (> STRONG_OFF_LANE_REMOVE_M), even if downstream waypoints
                # eventually drift onto a lane. A vehicle spawned 5+ m off
                # the road is visually wrong regardless of the rest of the
                # trajectory.
                if first_off > STRONG_OFF_LANE_REMOVE_M:
                    d.verdict = "REMOVE"
                    d.note = (
                        f"moving NPC spawn is grossly off-lane "
                        f"(spawn {first_off:.2f}m > {STRONG_OFF_LANE_REMOVE_M:.1f}m "
                        f"from nearest Driving lane); cannot rigidly relocate "
                        f"a moving trajectory"
                    )
                    return
                # If the spawn is off-lane (>1.5m) but the trajectory has a
                # long contiguous on-lane segment, salvage by truncating
                # the off-lane prefix/suffix instead of deleting the actor.
                # This preserves the bulk of the NPC's planned motion and
                # avoids the visual "spawn on grass / drive off road at end"
                # artifacts.
                if first_off > 1.5:
                    s_idx, e_idx, run_len = _longest_on_lane_run(pts, lane_service)
                    if run_len >= max(MIN_SAMPLES,
                                       int(TRUNC_MIN_ON_LANE_RATIO * total)):
                        # Salvageable via prefix/suffix trim.
                        d.verdict = "TRUNCATE"
                        d.truncate_range = (s_idx, e_idx)  # type: ignore[attr-defined]
                        dropped_pre  = s_idx
                        dropped_post = total - 1 - e_idx
                        d.note = (
                            f"moving NPC truncated to on-lane core "
                            f"({run_len}/{total} = {100*run_len/total:.0f}% on-lane; "
                            f"dropped {dropped_pre} prefix + {dropped_post} suffix "
                            f"waypoints; spawn was {first_off:.2f}m off-lane)"
                        )
                        return
                    # No long enough on-lane segment → remove.
                    d.verdict = "REMOVE"
                    d.note = (
                        f"moving NPC trajectory is off-lane "
                        f"(spawn {first_off:.2f}m from nearest Driving lane, "
                        f"only {n_on}/{total} = {100*on_ratio:.0f}% of "
                        f"waypoints on a Driving lane; no on-lane run "
                        f">= {int(TRUNC_MIN_ON_LANE_RATIO*100)}% of trajectory)"
                    )
                    return
        d.verdict = "OK"
        d.note = ""
        return

    # ── Pass 2: verified-lane direction (cascade — verified, then siblings, then prior)
    # The lane service returns a derived lane-forward direction PLUS a snap pose
    # (lane projection point) so the caller can plan an actual snap+yaw change.
    base_verdict = "STATIC" if velocity_static else "TOO_FEW"
    if velocity_static:
        # Show whichever condition was decisive
        if d.arc_m < MIN_ARC_M:
            base_note = f"arc={d.arc_m:.2f}m < {MIN_ARC_M}m"
        else:
            base_note = (f"net displacement={d.net_displacement_m:.2f}m "
                         f"< {MIN_NET_DISPLACEMENT_M}m (arc={d.arc_m:.2f}m, "
                         f"likely jitter)")
    else:
        base_note = f"only {d.n_wrong + d.n_right + d.n_amb} moving frames"
    lane_verdict = lane_note = lane_snap_info = None
    if lane_service is not None and lane_service.is_available():
        lane_verdict, lane_note, lane_snap_info = lane_service.classify(
            d.spawn_x, d.spawn_y, d.spawn_yaw,
        )
        # Aligned with derived/verified lane direction → already correct.
        if lane_verdict == "OK_LANE":
            d.verdict = "OK"
            d.note = f"{base_verdict.lower()}; {lane_note}"
            return
        # Actor sits on a non-driving feature (sidewalk / parking / shoulder
        # / median). Leave it alone — that's where it's supposed to be.
        if lane_verdict == "OK_NONDRIVING_LANE":
            d.verdict = "OK"
            d.note = f"{base_verdict.lower()}; {lane_note}"
            return
        # Reversed wrt lane direction → straight 180° in-place flip (no position
        # change). Collision check is unnecessary because the position stays the
        # same and the OBB only rotates 180° — its footprint occupies the same
        # cells either way (a rotated rectangle by 180° equals itself).
        if lane_verdict in ("FLIP_LANE_VERIFIED", "FLIP_LANE_DERIVED",
                            "FLIP_LANE_PRIOR"):
            d.verdict = {
                "FLIP_LANE_VERIFIED": "FLIP_LANE",
                "FLIP_LANE_DERIVED":  "FLIP_LANE",
                "FLIP_LANE_PRIOR":    "FLIP_LANE",
            }[lane_verdict]
            d.note = f"{base_verdict.lower()}; {lane_note}"
            d.snap_pose = None
            return
        # Perpendicular or off-lane → propose a snap. The snap pose moves
        # (x, y) onto the lane and sets yaw to the closer of (lane_fwd) /
        # (lane_fwd+180°). Collision check decides snap-or-remove.
        if lane_verdict in ("SNAP_PERP", "SNAP_OFFLANE") and lane_snap_info is not None:
            snap_x = lane_snap_info["wpx"]
            snap_y = lane_snap_info["wpy"]
            lane_fwd = lane_snap_info["lane_fwd"]
            # Prefer neighbour consensus over the actor's own yaw — the
            # actor's pre-snap yaw is often wrong (e.g. 90° rotated by a
            # sensor calibration error) and would pick the wrong side of
            # the 180° flip. Nearby parked cars and moving traffic are a
            # robust direction reference.
            ref_yaw = d.spawn_yaw
            consensus = _neighbor_yaw_consensus(snap_x, snap_y, scene_obbs, d.xml_rel)
            if consensus is not None:
                ref_yaw = consensus
            snap_yaw = _choose_snap_yaw(ref_yaw, lane_fwd)
            d.snap_pose = (snap_x, snap_y, snap_yaw)
            # Collision check
            if scene_obbs is not None:
                hl, hw = _actor_dims_for_role(d.role)
                candidate_obb: OBB = (snap_x, snap_y, hl/2.0, hw/2.0, snap_yaw)
                exclude_rel = d.xml_rel
                colliding = _snap_collides(candidate_obb, scene_obbs, exclude_rel)
                if colliding is None:
                    d.verdict = "SNAP"
                    d.note = f"{base_verdict.lower()}; {lane_note}; snap to ({snap_x:.2f},{snap_y:.2f}) yaw={snap_yaw:+.1f}°"
                    return
                else:
                    d.verdict = "REMOVE"
                    d.note = f"{base_verdict.lower()}; {lane_note}; snap collides with {colliding}"
                    return
            else:
                # No collision context available — degrade to base verdict and
                # surface the snap proposal as info only.
                d.verdict = base_verdict
                d.note = f"{base_note}; {lane_note}; (no scene OBBs — snap not proposed)"
                return
        # OFF_LANE (no driving lane within SNAP_MAX_LATERAL_M) → vehicle is
        # >12m off any road. Snap not feasible, neighbour motion won't help
        # (it can't relocate the actor). REMOVE.
        if lane_verdict == "OFF_LANE":
            d.verdict = "REMOVE"
            d.note = (
                f"{base_verdict.lower()}; {lane_note}; no driving lane within "
                f"{SNAP_MAX_LATERAL_M:.0f}m — cannot snap"
            )
            return
        # LANE_UNRELIABLE → fall through to neighbour motion.

    # ── Pass 3: neighbour-motion fallback (best-effort) ────────────────
    extras = []
    if lane_note is not None:
        extras.append(f"lane: {lane_note}")
    if motion_segs is None:
        d.verdict = base_verdict
        d.note = "; ".join([base_note] + extras)
        return
    n_verdict, n_note = _neighbor_classify(d.spawn_x, d.spawn_y, d.spawn_yaw,
                                           motion_segs)
    extras.append(f"neighbour: {n_note}")
    if n_verdict == "FLIP_NEIGHBOR":
        d.verdict = "FLIP_NEIGHBOR"
        d.note = "; ".join([base_verdict.lower()] + extras)
        return
    if n_verdict == "OK_NEIGHBOR":
        d.verdict = "OK"
        d.note = "; ".join([base_verdict.lower()] + extras)
        return
    # NO_NEIGHBORS / AMBIG_NEIGHBOR → keep the base verdict
    d.verdict = base_verdict
    d.note = "; ".join([base_note] + extras)


# ---------------------------------------------------------------------------
# Patch sidecar — coexistence guard
# ---------------------------------------------------------------------------

def _patched_actor_ids(scene_dir: Path) -> set:
    p = scene_dir / PATCH_SIDECAR_NAME
    if not p.exists():
        return set()
    try:
        blob = json.loads(p.read_text())
    except Exception:
        return set()
    out: set = set()
    for ov in (blob.get("overrides") or []):
        if not isinstance(ov, dict):
            continue
        if ov.get("yaw_offset_deg") or ov.get("yaw_segment_offsets"):
            aid = ov.get("actor_id")
            if aid:
                out.add(str(aid))
    return out


def _actor_id_for_xml(xml_path: Path, manifest: Optional[dict]) -> str:
    """Best-effort: return the actor id that the patch editor uses.

    The patch editor builds its id from the manifest entry's ``name`` field
    if present; otherwise the XML stem with the ``<town>_custom_`` prefix
    stripped (see ``loader._read_xml_actor_id``). We replicate that here so
    we match the same key used in ``_patch_editor.patch.json``.
    """
    fname = xml_path.name
    if manifest:
        for kind in ("ego", "npc", "static", "walker", "cyclist"):
            for entry in (manifest.get(kind) or []):
                if isinstance(entry, dict) and entry.get("file"):
                    # manifest stores either basename or relative path
                    raw = str(entry["file"])
                    if raw == fname or os.path.basename(raw) == fname:
                        return str(entry.get("name") or entry.get("id") or
                                   xml_path.stem)
    # Fallback: strip "<town>_custom_" prefix
    parts = xml_path.stem.split("_custom_", 1)
    return parts[1] if len(parts) == 2 else xml_path.stem


# ---------------------------------------------------------------------------
# Apply the flip
# ---------------------------------------------------------------------------

# Matches ``yaw="<float>"`` only inside ``<waypoint …>`` tags. We capture
# the lead-in / numeric value / closing quote so the substitution preserves
# everything else byte-for-byte (whitespace, attribute order, comments).
# Using ET.write() instead would (a) require Python 3.9+ for ET.indent and
# (b) silently re-sort attributes on Python 3.7 — both unacceptable since
# the editor and the simulator both read these XMLs.
_WAYPOINT_YAW_RE = re.compile(
    r'(<waypoint\b[^>]*?\byaw=")(-?\d+(?:\.\d+)?)(")',
    flags=re.DOTALL,
)


def _flip_xml_yaw(xml_path: Path) -> int:
    """Add 180° to every ``<waypoint yaw=...>`` value in ``xml_path`` in place.

    Returns the number of waypoints modified. Touches only the yaw value of
    each ``<waypoint>``; everything else (route attrs, x/y/z/pitch/roll/time,
    XML declaration, indentation, attribute order) is preserved verbatim.
    """
    text = xml_path.read_text(encoding="utf-8")
    n = 0

    def _sub(m):
        nonlocal n
        try:
            y = float(m.group(2))
        except ValueError:
            return m.group(0)
        new_val = f"{_normalize_yaw(y + 180.0):.6f}"
        n += 1
        return m.group(1) + new_val + m.group(3)

    new_text = _WAYPOINT_YAW_RE.sub(_sub, text)
    if n > 0:
        xml_path.write_text(new_text, encoding="utf-8")
    return n


# ---------------------------------------------------------------------------
# Scenario sweep
# ---------------------------------------------------------------------------

def _iter_scene_xmls(scene_dir: Path) -> List[Path]:
    out: List[Path] = []
    for pat in (
        scene_dir / "*.xml",
        scene_dir / "actors" / "*" / "*.xml",
    ):
        for p in sorted(glob.glob(str(pat))):
            pp = Path(p)
            if pp.name.endswith(REPLAY_SUFFIX):
                continue
            out.append(pp)
    return out


def _load_manifest(scene_dir: Path) -> Optional[dict]:
    p = scene_dir / "actors_manifest.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _is_walker_xml(xml_path: Path, role: str) -> bool:
    role_l = (role or "").lower()
    if role_l in ("walker", "cyclist"):
        return True
    # Defensive: actors/{walker,cyclist}/ subdir naming
    parent = xml_path.parent.name.lower()
    return parent in ("walker", "cyclist")


@dataclass
class SceneReport:
    name: str
    decisions: List[ActorDecision] = field(default_factory=list)

    @property
    def n_flip(self) -> int:
        return sum(1 for d in self.decisions if d.verdict == "FLIP")
    @property
    def n_flip_lane(self) -> int:
        return sum(1 for d in self.decisions if d.verdict == "FLIP_LANE")
    @property
    def n_flip_neighbor(self) -> int:
        return sum(1 for d in self.decisions if d.verdict == "FLIP_NEIGHBOR")
    @property
    def n_snap(self) -> int:
        return sum(1 for d in self.decisions if d.verdict == "SNAP")
    @property
    def n_remove(self) -> int:
        return sum(1 for d in self.decisions if d.verdict == "REMOVE")
    @property
    def n_truncate(self) -> int:
        return sum(1 for d in self.decisions if d.verdict == "TRUNCATE")
    @property
    def n_review(self) -> int:
        return sum(1 for d in self.decisions if d.verdict == "REVIEW")
    @property
    def n_static(self) -> int:
        return sum(1 for d in self.decisions if d.verdict == "STATIC")
    @property
    def n_walker(self) -> int:
        return sum(1 for d in self.decisions if d.verdict == "WALKER")
    @property
    def n_patched(self) -> int:
        return sum(1 for d in self.decisions if d.verdict == "PATCHED")
    @property
    def n_ok(self) -> int:
        return sum(1 for d in self.decisions if d.verdict == "OK")
    @property
    def n_too_few(self) -> int:
        return sum(1 for d in self.decisions if d.verdict == "TOO_FEW")


def process_scene(scene_dir: Path, *, include_walkers: bool,
                  use_motion_consensus: bool = True,
                  lane_service: Optional[_LaneDirectionService] = None,
                  use_snap_pass: bool = True,
                  ) -> SceneReport:
    rep = SceneReport(name=scene_dir.name)
    manifest = _load_manifest(scene_dir)
    patched_ids = _patched_actor_ids(scene_dir)
    motion_segs: Optional[List[MotionSeg]] = (
        _collect_motion_segments(scene_dir) if use_motion_consensus else None
    )
    scene_obbs: Optional[SceneOBBs] = (
        _build_scene_obbs(scene_dir) if use_snap_pass else None
    )
    for xml_path in _iter_scene_xmls(scene_dir):
        decision = _analyse_xml(xml_path)
        if decision is None:
            continue
        decision.xml_rel = str(xml_path.relative_to(scene_dir))
        actor_id = _actor_id_for_xml(xml_path, manifest)
        _classify(
            decision,
            is_walker=_is_walker_xml(xml_path, decision.role),
            is_patched=(actor_id in patched_ids),
            include_walkers=include_walkers,
            motion_segs=motion_segs,
            lane_service=lane_service,
            scene_obbs=scene_obbs,
        )
        rep.decisions.append(decision)
    return rep


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarioset", type=Path,
                    default=Path("scenarioset/v2xpnp"),
                    help="Root containing v2xpnp scenario subfolders "
                         "(default: scenarioset/v2xpnp)")
    ap.add_argument("--scenes", default="",
                    help="Comma-separated subset of scene names to process")
    ap.add_argument("--apply", action="store_true",
                    help="Actually rewrite XMLs (default: dry-run)")
    ap.add_argument("--no-backup", action="store_true",
                    help="Skip writing <file>.preflip.xml backups on --apply")
    ap.add_argument("--include-walkers", action="store_true",
                    help="Also consider walker/cyclist actors")
    ap.add_argument("--xodr", type=Path, default=DEFAULT_XODR,
                    help=f"OpenDRIVE source for offline lane direction queries "
                         f"(default: {DEFAULT_XODR})")
    ap.add_argument("--verified-lanes", type=Path, default=DEFAULT_VERIFIED_LANES,
                    help=f"JSON of motion-validated lane verdicts, produced by "
                         f"v2xpnp/scripts/build_verified_lane_directions.py (default: "
                         f"{DEFAULT_VERIFIED_LANES})")
    ap.add_argument("--no-lane-check", action="store_true",
                    help="Skip the verified-lane direction pass entirely")
    ap.add_argument("--no-motion-consensus", action="store_true",
                    help="Skip the neighbour-motion fallback pass entirely")
    ap.add_argument("--no-snap", action="store_true",
                    help="Skip the snap-or-remove pass for perpendicular / "
                         "off-lane static actors (lane-pass and yaw-only flips "
                         "still run as usual)")
    ap.add_argument("--report", type=Path, default=None,
                    help="Write a JSON report of every decision to this path")
    ap.add_argument("--quiet", action="store_true",
                    help="Only print the summary, not per-actor lines")
    args = ap.parse_args()

    if not args.scenarioset.is_dir():
        print(f"ERROR: scenarioset root not found: {args.scenarioset}",
              file=sys.stderr)
        sys.exit(2)

    wanted = {s.strip() for s in args.scenes.split(",") if s.strip()}
    scene_dirs = []
    for d in sorted(args.scenarioset.iterdir()):
        if not d.is_dir(): continue
        if d.name == "carla_routes_patched": continue   # editor output dir
        if wanted and d.name not in wanted: continue
        scene_dirs.append(d)
    if not scene_dirs:
        print(f"No scenes selected under {args.scenarioset}", file=sys.stderr)
        sys.exit(2)

    use_consensus = not args.no_motion_consensus
    use_lane = not args.no_lane_check
    mode = "APPLY" if args.apply else "DRY-RUN"

    # Build the lane-direction service once (xodr load is slow ~5s).
    lane_service: Optional[_LaneDirectionService] = None
    if use_lane:
        verified_path = args.verified_lanes if args.verified_lanes.exists() else None
        lane_service = _LaneDirectionService(args.xodr, verified_path)
        if not lane_service.is_available():
            lane_service = None
            use_lane = False

    use_snap = not args.no_snap
    tags = []
    tags.append("lane on" if use_lane else "lane OFF")
    tags.append("neighbour on" if use_consensus else "neighbour OFF")
    tags.append("snap on" if use_snap else "snap OFF")
    print(f"=== Yaw reversal fixer ({mode}, {', '.join(tags)}) — "
          f"{len(scene_dirs)} scene(s) ===")

    reports: List[SceneReport] = []
    n_flipped_now = 0
    n_snapped_now = 0
    n_removed_now = 0
    n_truncated_now = 0
    for sd in scene_dirs:
        rep = process_scene(sd, include_walkers=args.include_walkers,
                            use_motion_consensus=use_consensus,
                            lane_service=lane_service,
                            use_snap_pass=use_snap)
        reports.append(rep)
        flips      = [d for d in rep.decisions if d.verdict == "FLIP"]
        flips_lane = [d for d in rep.decisions if d.verdict == "FLIP_LANE"]
        flips_nbr  = [d for d in rep.decisions if d.verdict == "FLIP_NEIGHBOR"]
        snaps      = [d for d in rep.decisions if d.verdict == "SNAP"]
        removes    = [d for d in rep.decisions if d.verdict == "REMOVE"]
        truncates  = [d for d in rep.decisions if d.verdict == "TRUNCATE"]
        reviews    = [d for d in rep.decisions if d.verdict == "REVIEW"]
        if not args.quiet and (flips or flips_lane or flips_nbr or snaps or removes or truncates or reviews):
            print(f"\nSCENE {rep.name}")
            for d in flips:
                print(f"  FLIP   {d.xml_rel:<60s}  {d.note}")
            for d in flips_lane:
                print(f"  FLIP-L {d.xml_rel:<60s}  {d.note}")
            for d in flips_nbr:
                print(f"  FLIP-N {d.xml_rel:<60s}  {d.note}")
            for d in snaps:
                print(f"  SNAP   {d.xml_rel:<60s}  {d.note}")
            for d in truncates:
                print(f"  TRUNC  {d.xml_rel:<60s}  {d.note}")
            for d in removes:
                print(f"  REMOVE {d.xml_rel:<60s}  {d.note}")
            for d in reviews:
                print(f"  REVIEW {d.xml_rel:<60s}  {d.note}")
        if args.apply:
            # Yaw-only flips first (no position change, OBB unchanged → safe).
            for d in (flips + flips_lane + flips_nbr):
                xml_path = sd / d.xml_rel
                if not args.no_backup:
                    bak = xml_path.with_suffix(xml_path.suffix + ".preflip")
                    if not bak.exists():
                        shutil.copy2(xml_path, bak)
                try:
                    n_wps = _flip_xml_yaw(xml_path)
                    n_flipped_now += 1
                    if not args.quiet:
                        print(f"    wrote {n_wps} flipped waypoints")
                except Exception as exc:
                    print(f"    ERROR writing {xml_path}: {exc}", file=sys.stderr)
            # Snap-to-lane: rewrites x/y/yaw on every waypoint.
            for d in snaps:
                if d.snap_pose is None:
                    continue
                xml_path = sd / d.xml_rel
                if not args.no_backup:
                    bak = xml_path.with_suffix(xml_path.suffix + ".presnap")
                    if not bak.exists():
                        shutil.copy2(xml_path, bak)
                try:
                    n_wps = _snap_xml_pose(xml_path, *d.snap_pose)
                    n_snapped_now += 1
                    if not args.quiet:
                        print(f"    snapped {n_wps} waypoints to "
                              f"({d.snap_pose[0]:.2f},{d.snap_pose[1]:.2f}) "
                              f"yaw={d.snap_pose[2]:+.1f}°")
                except Exception as exc:
                    print(f"    ERROR snapping {xml_path}: {exc}", file=sys.stderr)
            # Trajectory truncation: keep a .pretrunc backup, drop off-lane
            # prefix/suffix waypoints, rebase the time column on the rest.
            for d in truncates:
                rng = getattr(d, "truncate_range", None)
                if rng is None:
                    continue
                xml_path = sd / d.xml_rel
                if not args.no_backup:
                    bak = xml_path.with_suffix(xml_path.suffix + ".pretrunc")
                    if not bak.exists():
                        try:
                            shutil.copy2(xml_path, bak)
                        except Exception as exc:
                            print(f"    ERROR backing up {xml_path}: {exc}", file=sys.stderr)
                            continue
                try:
                    n_kept = _truncate_xml_to_indices(xml_path, rng[0], rng[1])
                    if n_kept > 0:
                        n_truncated_now += 1
                        if not args.quiet:
                            print(f"    truncated to {n_kept} waypoints ({rng[0]}..{rng[1]})")
                except Exception as exc:
                    print(f"    ERROR truncating {xml_path}: {exc}", file=sys.stderr)
            # Removals: archive a .predelete backup, then delete the XML and
            # the manifest entry.
            for d in removes:
                xml_path = sd / d.xml_rel
                if not args.no_backup:
                    bak = xml_path.with_suffix(xml_path.suffix + ".predelete")
                    if not bak.exists():
                        try:
                            shutil.copy2(xml_path, bak)
                        except Exception as exc:
                            print(f"    ERROR backing up {xml_path}: {exc}", file=sys.stderr)
                            continue
                try:
                    ok = _remove_actor_xml(xml_path, sd)
                    if ok:
                        n_removed_now += 1
                        if not args.quiet:
                            print(f"    removed XML + manifest entry")
                except Exception as exc:
                    print(f"    ERROR removing {xml_path}: {exc}", file=sys.stderr)

    # ── Aggregate ────────────────────────────────────────────────────────
    n_total = sum(len(r.decisions) for r in reports)
    n_flip          = sum(r.n_flip          for r in reports)
    n_flip_lane     = sum(r.n_flip_lane     for r in reports)
    n_flip_neighbor = sum(r.n_flip_neighbor for r in reports)
    n_snap          = sum(r.n_snap          for r in reports)
    n_truncate      = sum(r.n_truncate      for r in reports)
    n_remove        = sum(r.n_remove        for r in reports)
    n_review = sum(r.n_review for r in reports)
    n_static = sum(r.n_static for r in reports)
    n_walker = sum(r.n_walker for r in reports)
    n_patched = sum(r.n_patched for r in reports)
    n_too_few = sum(r.n_too_few for r in reports)
    n_ok     = sum(r.n_ok     for r in reports)
    print()
    print(f"=== Summary ===")
    print(f"  examined          : {n_total}")
    print(f"  FLIP (own motion) : {n_flip}{' (applied)' if args.apply else ' (dry-run)'}")
    print(f"  FLIP (lane GT)    : {n_flip_lane}{' (applied)' if args.apply else ' (dry-run)'}")
    print(f"  FLIP (neighbours) : {n_flip_neighbor}{' (applied)' if args.apply else ' (dry-run)'}")
    print(f"  SNAP (to lane)    : {n_snap}{' (applied)' if args.apply else ' (dry-run)'}")
    print(f"  TRUNC (on-lane)   : {n_truncate}{' (applied)' if args.apply else ' (dry-run)'}")
    print(f"  REMOVE (blocked)  : {n_remove}{' (applied)' if args.apply else ' (dry-run)'}")
    print(f"  REVIEW            : {n_review}  (need human judgement; not auto-edited)")
    print(f"  STATIC            : {n_static}  (no decisive signal — left alone)")
    print(f"  WALKER            : {n_walker}{' (use --include-walkers)' if not args.include_walkers else ''}")
    print(f"  PATCHED           : {n_patched}  (already manually overridden)")
    print(f"  TOO_FEW           : {n_too_few}")
    print(f"  OK                : {n_ok}")
    if args.apply:
        print(f"  Wrote             : {n_flipped_now} flips, {n_snapped_now} snaps, "
              f"{n_truncated_now} truncated, {n_removed_now} removed"
              f"{' (backups: .preflip / .presnap / .pretrunc / .predelete)' if not args.no_backup else ''}")

    # ── JSON report ──────────────────────────────────────────────────────
    if args.report is not None:
        out = {
            "scenarioset": str(args.scenarioset.resolve()),
            "mode": mode,
            "summary": {
                "examined": n_total,
                "flip_velocity": n_flip,
                "flip_lane": n_flip_lane,
                "flip_neighbor": n_flip_neighbor,
                "snap": n_snap,
                "remove": n_remove,
                "review": n_review,
                "static": n_static, "walker": n_walker, "patched": n_patched,
                "too_few": n_too_few, "ok": n_ok,
            },
            "scenes": {
                r.name: [asdict(d) for d in r.decisions]
                for r in reports
            },
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(out, indent=2))
        print(f"  Report   : {args.report}")


if __name__ == "__main__":
    main()
