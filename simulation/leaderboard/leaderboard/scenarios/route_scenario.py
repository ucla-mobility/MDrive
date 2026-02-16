#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides Challenge routes as standalone scenarios
"""

from __future__ import print_function

import math
import os
import re
import queue
import xml.etree.ElementTree as ET
import numpy.random as random
import torch
import py_trees
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
import carla

from agents.navigation.local_planner import RoadOption
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

# ---------------------- Heading Helpers ---------------------- #

def _angle_delta(a: float, b: float) -> float:
    diff = (a - b + 180.0) % 360.0 - 180.0
    return abs(diff)


def _heading_from_points(points: List[carla.Location], min_dist: float = 2.0) -> Optional[float]:
    if len(points) < 2:
        return None
    p0 = points[0]
    for idx in range(1, len(points)):
        dx = float(points[idx].x) - float(p0.x)
        dy = float(points[idx].y) - float(p0.y)
        if (dx * dx + dy * dy) >= (min_dist * min_dist):
            return math.degrees(math.atan2(dy, dx))
    return None


def _resolve_ground_z(world: Optional[carla.World], location: carla.Location) -> Optional[float]:
    if world is None:
        return None
    ground_projection = getattr(world, "ground_projection", None)
    if callable(ground_projection):
        try:
            probe = carla.Location(
                x=location.x,
                y=location.y,
                z=location.z + 50.0,
            )
            result = ground_projection(probe, 100.0)
            if result is not None:
                if hasattr(result, "z"):
                    return float(result.z)
                if isinstance(result, (tuple, list)) and result:
                    first = result[0]
                    if hasattr(first, "z"):
                        return float(first.z)
        except Exception:  # pylint: disable=broad-except
            pass
    cast_ray = getattr(world, "cast_ray", None)
    if callable(cast_ray):
        try:
            start = carla.Location(
                x=location.x,
                y=location.y,
                z=location.z + 50.0,
            )
            end = carla.Location(
                x=location.x,
                y=location.y,
                z=location.z - 50.0,
            )
            hits = cast_ray(start, end)
            if hits:
                best_z = None
                best_dist = None
                for hit in hits:
                    hit_loc = getattr(hit, "location", None) or getattr(hit, "point", None)
                    if hit_loc is None:
                        continue
                    dz = abs(float(hit_loc.z) - float(location.z))
                    if best_dist is None or dz < best_dist:
                        best_dist = dz
                        best_z = float(hit_loc.z)
                if best_z is not None:
                    return best_z
        except Exception:  # pylint: disable=broad-except
            pass
    return None


def _glue_plan_to_ground(
    plan: List[carla.Transform],
    actor: carla.Actor,
    world_map: Optional[carla.Map],
    lane_type: carla.LaneType,
    world: Optional[carla.World],
    prefer_ray_ground: bool = False,
    z_extra: float = 0.0,
) -> None:
    if not plan:
        return
    try:
        bbox = actor.bounding_box
        base_offset = float(bbox.extent.z) - float(bbox.location.z)
    except Exception:  # pylint: disable=broad-except
        base_offset = 0.0
    for tf in plan:
        candidates: list[float] = []
        ground_z = _resolve_ground_z(world, tf.location)
        if ground_z is not None:
            candidates.append(ground_z)
        if world_map is not None:
            snapped_wp = world_map.get_waypoint(
                tf.location,
                project_to_road=True,
                lane_type=lane_type,
            )
            if snapped_wp is not None:
                candidates.append(float(snapped_wp.transform.location.z))
        if not candidates:
            continue
        if prefer_ray_ground and ground_z is not None:
            target_z = float(ground_z)
        else:
            target_z = min(candidates, key=lambda z: abs(z - float(tf.location.z)))
        tf.location.z = float(target_z) + base_offset + float(z_extra)


def _interp_angle(a: float, b: float, alpha: float) -> float:
    delta = (b - a + 180.0) % 360.0 - 180.0
    return a + delta * alpha


def _interp_transform(a: carla.Transform, b: carla.Transform, alpha: float) -> carla.Transform:
    loc = carla.Location(
        x=a.location.x + (b.location.x - a.location.x) * alpha,
        y=a.location.y + (b.location.y - a.location.y) * alpha,
        z=a.location.z + (b.location.z - a.location.z) * alpha,
    )
    rot = carla.Rotation(
        pitch=_interp_angle(a.rotation.pitch, b.rotation.pitch, alpha),
        yaw=_interp_angle(a.rotation.yaw, b.rotation.yaw, alpha),
        roll=_interp_angle(a.rotation.roll, b.rotation.roll, alpha),
    )
    return carla.Transform(loc, rot)


def _copy_transform(tf: carla.Transform) -> carla.Transform:
    return carla.Transform(
        carla.Location(
            x=float(tf.location.x),
            y=float(tf.location.y),
            z=float(tf.location.z),
        ),
        carla.Rotation(
            pitch=float(tf.rotation.pitch),
            yaw=float(tf.rotation.yaw),
            roll=float(tf.rotation.roll),
        ),
    )


def _is_walker_actor(actor: Optional[carla.Actor]) -> bool:
    if actor is None:
        return False
    try:
        if isinstance(actor, carla.Walker):
            return True
    except Exception:  # pylint: disable=broad-except
        pass
    try:
        return str(actor.type_id).startswith("walker.")
    except Exception:  # pylint: disable=broad-except
        return False


class _OneEuroScalar:
    """
    Lightweight One Euro filter for adaptive smoothing.
    """

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self._min_cutoff = max(1e-4, float(min_cutoff))
        self._beta = max(0.0, float(beta))
        self._d_cutoff = max(1e-4, float(d_cutoff))
        self._x_hat: Optional[float] = None
        self._dx_hat: float = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(1e-4, cutoff))
        return 1.0 / (1.0 + tau / max(1e-4, dt))

    def filter(self, value: float, dt: float) -> float:
        value = float(value)
        dt = max(1e-4, float(dt))
        if self._x_hat is None:
            self._x_hat = value
            self._dx_hat = 0.0
            return value

        dx = (value - self._x_hat) / dt
        a_d = self._alpha(self._d_cutoff, dt)
        self._dx_hat = a_d * dx + (1.0 - a_d) * self._dx_hat

        cutoff = self._min_cutoff + self._beta * abs(self._dx_hat)
        a = self._alpha(cutoff, dt)
        self._x_hat = a * value + (1.0 - a) * self._x_hat
        return self._x_hat


def _unwrap_angle_rad(current: float, previous: float, previous_unwrapped: float) -> float:
    delta = (float(current) - float(previous) + math.pi) % (2.0 * math.pi) - math.pi
    return float(previous_unwrapped) + delta


def _wrap_angle_deg(angle_rad: float) -> float:
    return math.degrees((float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi)


def _delta_angle_deg(current: float, previous: float) -> float:
    return (float(current) - float(previous) + 180.0) % 360.0 - 180.0


def _distance_xy(a: carla.Transform, b: carla.Transform) -> float:
    dx = float(b.location.x) - float(a.location.x)
    dy = float(b.location.y) - float(a.location.y)
    return math.sqrt(dx * dx + dy * dy)


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(float(v) for v in values)
    if len(arr) == 1:
        return arr[0]
    q = max(0.0, min(100.0, float(q)))
    rank = (q / 100.0) * (len(arr) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return arr[lo]
    w = rank - lo
    return arr[lo] * (1.0 - w) + arr[hi] * w


def _closest_point_on_segment_xy(
    point_x: float,
    point_y: float,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
) -> Tuple[float, float]:
    seg_x = float(end_x) - float(start_x)
    seg_y = float(end_y) - float(start_y)
    denom = seg_x * seg_x + seg_y * seg_y
    if denom <= 1e-9:
        return float(start_x), float(start_y)
    proj = ((float(point_x) - float(start_x)) * seg_x + (float(point_y) - float(start_y)) * seg_y) / denom
    proj = max(0.0, min(1.0, float(proj)))
    return float(start_x) + seg_x * proj, float(start_y) + seg_y * proj


def _turn_angle_deg(prev_tf: carla.Transform, cur_tf: carla.Transform, next_tf: carla.Transform) -> float:
    h0 = math.degrees(
        math.atan2(
            float(cur_tf.location.y) - float(prev_tf.location.y),
            float(cur_tf.location.x) - float(prev_tf.location.x),
        )
    )
    h1 = math.degrees(
        math.atan2(
            float(next_tf.location.y) - float(cur_tf.location.y),
            float(next_tf.location.x) - float(cur_tf.location.x),
        )
    )
    return abs(_delta_angle_deg(h1, h0))


def _suppress_vehicle_lateral_jitter(
    transforms: List[carla.Transform],
    *,
    damping: float,
    passes: int,
    max_correction: float,
    turn_keep: float,
    turn_angle_deg: float,
) -> List[carla.Transform]:
    """
    Reduce side-to-side wobble by pulling each point toward the local centerline
    defined by its neighbors. This keeps large trajectory changes while damping
    high-frequency lateral jitter.
    """
    if len(transforms) < 3:
        return [_copy_transform(tf) for tf in transforms]

    out = [_copy_transform(tf) for tf in transforms]
    damping = max(0.0, min(1.0, float(damping)))
    passes = max(1, int(passes))
    max_correction = max(0.0, float(max_correction))
    turn_keep = max(0.0, min(1.0, float(turn_keep)))
    turn_angle_deg = max(0.0, float(turn_angle_deg))

    for _ in range(passes):
        src = [_copy_transform(tf) for tf in out]
        for idx in range(1, len(src) - 1):
            prev_tf = src[idx - 1]
            cur_tf = src[idx]
            next_tf = src[idx + 1]

            cx, cy = _closest_point_on_segment_xy(
                float(cur_tf.location.x),
                float(cur_tf.location.y),
                float(prev_tf.location.x),
                float(prev_tf.location.y),
                float(next_tf.location.x),
                float(next_tf.location.y),
            )
            err_x = float(cur_tf.location.x) - cx
            err_y = float(cur_tf.location.y) - cy
            err_norm = math.sqrt(err_x * err_x + err_y * err_y)
            if err_norm <= 1e-5:
                continue

            keep = 1.0 - damping
            if _turn_angle_deg(prev_tf, cur_tf, next_tf) >= turn_angle_deg:
                keep = max(keep, turn_keep)
            corr_scale = 1.0 - keep
            corr_mag = min(err_norm * corr_scale, max_correction)
            corr_ratio = corr_mag / err_norm if err_norm > 1e-8 else 0.0

            out[idx].location.x = float(cur_tf.location.x) - err_x * corr_ratio
            out[idx].location.y = float(cur_tf.location.y) - err_y * corr_ratio

    out[0] = _copy_transform(transforms[0])
    out[-1] = _copy_transform(transforms[-1])
    return out


def _stabilize_near_static_vehicle_segments(
    transforms: List[carla.Transform],
    times: Optional[List[float]],
    *,
    total_displacement_threshold: float,
    total_path_length_threshold: float,
    window_min_duration: float,
    window_max_displacement: float,
    window_max_speed: float,
    window_max_yaw_delta: float,
) -> List[carla.Transform]:
    """
    Freeze near-static sub-segments in a replay plan to suppress parked-car jitter.
    Detection is interval-based: a vehicle can be frozen while stopped/parked and
    then smoothly resume movement later in the same trajectory.
    """
    if len(transforms) < 2:
        return [_copy_transform(tf) for tf in transforms]

    out = [_copy_transform(tf) for tf in transforms]
    has_times = bool(times) and len(times) == len(out)
    default_dt = 0.05

    seg_dt: List[float] = []
    seg_dist: List[float] = []
    seg_speed: List[float] = []
    seg_yaw_delta: List[float] = []
    for idx in range(len(out) - 1):
        if has_times:
            dt_raw = float(times[idx + 1]) - float(times[idx])
            # Use unclamped dt for motion statistics so sparse logs do not inflate speed.
            dt = max(1e-3, dt_raw)
        else:
            dt = default_dt
        dist_xy = _distance_xy(out[idx], out[idx + 1])
        seg_dt.append(dt)
        seg_dist.append(dist_xy)
        seg_speed.append(dist_xy / dt)
        seg_yaw_delta.append(
            abs(_delta_angle_deg(out[idx + 1].rotation.yaw, out[idx].rotation.yaw))
        )

    disp_thr = max(0.0, float(total_displacement_threshold))
    path_thr = max(0.0, float(total_path_length_threshold))
    win_disp_thr = max(0.0, float(window_max_displacement))
    win_speed_thr = max(0.0, float(window_max_speed))
    win_yaw_thr = max(0.0, float(window_max_yaw_delta))

    def _segment_static_stats(
        segment_tfs: List[carla.Transform],
        segment_dist: List[float],
        segment_speed: List[float],
        segment_yaw: List[float],
    ) -> Dict[str, object]:
        if len(segment_tfs) < 2:
            return {
                "is_static": False,
                "center_x": float(segment_tfs[0].location.x) if segment_tfs else 0.0,
                "center_y": float(segment_tfs[0].location.y) if segment_tfs else 0.0,
                "mean_z": float(segment_tfs[0].location.z) if segment_tfs else 0.0,
                "mean_yaw": float(segment_tfs[0].rotation.yaw) if segment_tfs else 0.0,
            }

        net_displacement = _distance_xy(segment_tfs[0], segment_tfs[-1])
        total_segment_path = sum(float(v) for v in segment_dist)
        max_radius_from_start = max(_distance_xy(segment_tfs[0], tf) for tf in segment_tfs)
        xs = [float(tf.location.x) for tf in segment_tfs]
        ys = [float(tf.location.y) for tf in segment_tfs]
        zs = [float(tf.location.z) for tf in segment_tfs]
        center_x = _percentile(xs, 50.0)
        center_y = _percentile(ys, 50.0)
        radial = [math.hypot(x - center_x, y - center_y) for x, y in zip(xs, ys)]
        p95_radius = _percentile(radial, 95.0)
        bbox_diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        p90_speed = _percentile(segment_speed, 90.0) if segment_speed else 0.0
        max_speed = max(segment_speed) if segment_speed else 0.0
        p90_step_dist = _percentile(segment_dist, 90.0) if segment_dist else 0.0
        p95_step_dist = _percentile(segment_dist, 95.0) if segment_dist else 0.0
        max_step_dist = max(segment_dist) if segment_dist else 0.0
        p95_step_yaw = _percentile(segment_yaw, 95.0) if segment_yaw else 0.0
        path_to_net_ratio = total_segment_path / max(0.05, net_displacement)
        path_to_bbox_ratio = total_segment_path / max(0.05, bbox_diag)

        unwrapped_yaw: List[float] = [float(segment_tfs[0].rotation.yaw)]
        for k in range(1, len(segment_tfs)):
            unwrapped_yaw.append(
                unwrapped_yaw[-1]
                + _delta_angle_deg(
                    float(segment_tfs[k].rotation.yaw),
                    float(segment_tfs[k - 1].rotation.yaw),
                )
            )
        yaw_span = max(unwrapped_yaw) - min(unwrapped_yaw)

        globally_static = (
            net_displacement <= disp_thr
            and max_radius_from_start <= max(0.0, 1.6 * win_disp_thr, 0.40)
            and p95_radius <= max(1.8 * win_disp_thr, 0.32)
            and yaw_span <= max(4.0 * win_yaw_thr, 10.0)
        )
        compact_jitter_static = (
            net_displacement <= max(1.60 * disp_thr, 0.65)
            and p95_radius <= max(3.2 * win_disp_thr, 0.72)
            and bbox_diag <= max(6.0 * win_disp_thr, 1.35)
            and p90_speed <= max(5.0 * win_speed_thr, 0.42)
            and max_speed <= max(7.0 * win_speed_thr, 0.95)
            and p95_step_dist <= max(2.6 * win_disp_thr, 0.36)
            and max_step_dist <= max(5.8 * win_disp_thr, 0.98)
            and yaw_span <= max(6.5 * win_yaw_thr, 16.0)
            and p95_step_yaw <= max(4.0 * win_yaw_thr, 6.5)
        )
        path_noisy_compact_static = (
            net_displacement <= max(1.45 * disp_thr, 0.55)
            and bbox_diag <= max(6.4 * win_disp_thr, 1.45)
            and p95_radius <= max(3.4 * win_disp_thr, 0.78)
            and p90_step_dist <= max(2.9 * win_disp_thr, 0.40)
            and p95_step_yaw <= max(4.6 * win_yaw_thr, 7.5)
            and path_to_net_ratio >= max(4.0, 2.0 * max(1e-3, path_thr / max(0.05, disp_thr)))
            and path_to_bbox_ratio <= max(8.0, 1.5 * max(1e-3, path_thr / max(0.05, win_disp_thr)))
        )
        noisy_low_drift_static = (
            net_displacement <= max(1.30 * disp_thr, 0.52)
            and bbox_diag <= max(7.0 * win_disp_thr, 1.55)
            and p95_radius <= max(3.7 * win_disp_thr, 0.84)
            and p95_step_dist <= max(3.3 * win_disp_thr, 0.46)
            and p95_step_yaw <= max(5.0 * win_yaw_thr, 8.5)
            and p90_speed <= max(8.0 * win_speed_thr, 0.68)
            and max_speed <= max(18.0 * win_speed_thr, 2.35)
            and path_to_net_ratio >= max(4.0, 1.8 * max(1e-3, path_thr / max(0.05, disp_thr)))
            and path_to_bbox_ratio <= max(9.0, 1.7 * max(1e-3, path_thr / max(0.05, win_disp_thr)))
        )
        static_hit = (
            globally_static
            or compact_jitter_static
            or path_noisy_compact_static
            or noisy_low_drift_static
        )

        sin_sum = sum(math.sin(math.radians(float(tf.rotation.yaw))) for tf in segment_tfs)
        cos_sum = sum(math.cos(math.radians(float(tf.rotation.yaw))) for tf in segment_tfs)
        if abs(sin_sum) + abs(cos_sum) > 1e-6:
            mean_yaw = math.degrees(math.atan2(sin_sum, cos_sum))
        else:
            mean_yaw = float(segment_tfs[0].rotation.yaw)

        return {
            "is_static": bool(static_hit),
            "center_x": float(center_x),
            "center_y": float(center_y),
            "mean_z": float(sum(zs) / max(1.0, float(len(zs)))),
            "mean_yaw": float(mean_yaw),
        }

    whole_stats = _segment_static_stats(out, seg_dist, seg_speed, seg_yaw_delta)
    if bool(whole_stats.get("is_static")):
        anchor = _copy_transform(out[0])
        anchor.location.x = float(whole_stats["center_x"])
        anchor.location.y = float(whole_stats["center_y"])
        anchor.location.z = float(whole_stats["mean_z"])
        anchor.rotation.yaw = float(whole_stats["mean_yaw"])
        return [_copy_transform(anchor) for _ in out]

    low_motion = [
        (seg_speed[i] <= max(5.0 * win_speed_thr, 0.35))
        and (seg_yaw_delta[i] <= max(4.0 * win_yaw_thr, 6.0))
        for i in range(len(seg_speed))
    ]
    for idx in range(1, len(low_motion) - 1):
        if low_motion[idx]:
            continue
        if (
            low_motion[idx - 1]
            and low_motion[idx + 1]
            and seg_speed[idx] <= max(8.0 * win_speed_thr, 0.90)
            and seg_yaw_delta[idx] <= max(5.5 * win_yaw_thr, 11.0)
        ):
            low_motion[idx] = True

    idx = 0
    while idx < len(low_motion):
        if not low_motion[idx]:
            idx += 1
            continue
        start_seg = idx
        while idx + 1 < len(low_motion) and low_motion[idx + 1]:
            idx += 1
        end_seg = idx

        start_idx = start_seg
        end_idx = end_seg + 1
        duration = sum(seg_dt[start_seg : end_seg + 1])
        duration_ok = duration >= max(0.0, 0.50 * float(window_min_duration), 0.35)
        seg_stats = _segment_static_stats(
            out[start_idx : end_idx + 1],
            seg_dist[start_seg : end_seg + 1],
            seg_speed[start_seg : end_seg + 1],
            seg_yaw_delta[start_seg : end_seg + 1],
        )
        if duration_ok and bool(seg_stats.get("is_static")):
            anchor_x = float(seg_stats["center_x"])
            anchor_y = float(seg_stats["center_y"])
            anchor_z = float(seg_stats["mean_z"])
            anchor_yaw = float(seg_stats["mean_yaw"])

            seg_count = end_idx - start_idx + 1
            mean_dt = duration / max(1.0, float(seg_count - 1))
            ramp_frames = max(1, min(6, int(round(0.35 / max(1e-3, mean_dt)))))
            ramp_frames = max(1, min(ramp_frames, max(1, (seg_count - 1) // 2)))
            has_left_neighbor = start_idx > 0
            has_right_neighbor = end_idx < len(out) - 1

            for j in range(start_idx, end_idx + 1):
                src_tf = out[j]
                w_left = 1.0
                w_right = 1.0
                if has_left_neighbor:
                    w_left = min(1.0, float(j - start_idx) / float(ramp_frames))
                if has_right_neighbor:
                    w_right = min(1.0, float(end_idx - j) / float(ramp_frames))
                freeze_weight = max(0.0, min(1.0, min(w_left, w_right)))

                out[j].location.x = float(src_tf.location.x) * (1.0 - freeze_weight) + anchor_x * freeze_weight
                out[j].location.y = float(src_tf.location.y) * (1.0 - freeze_weight) + anchor_y * freeze_weight
                out[j].location.z = float(src_tf.location.z) * (1.0 - freeze_weight) + anchor_z * freeze_weight
                yaw_delta = _delta_angle_deg(float(anchor_yaw), float(src_tf.rotation.yaw))
                out[j].rotation.yaw = _wrap_angle_deg(
                    math.radians(float(src_tf.rotation.yaw) + freeze_weight * yaw_delta)
                )

        idx += 1

    return out


def _smooth_vehicle_replay_plan(
    transforms: List[carla.Transform],
    times: Optional[List[float]],
    *,
    min_cutoff_xy: float,
    beta_xy: float,
    min_cutoff_z: float,
    beta_z: float,
    min_cutoff_yaw: float,
    beta_yaw: float,
    d_cutoff: float,
    static_hold_enabled: bool = True,
    static_total_displacement: float = 0.35,
    static_total_path_length: float = 1.6,
    static_window_min_duration: float = 0.8,
    static_window_max_displacement: float = 0.22,
    static_window_max_speed: float = 0.08,
    static_window_max_yaw_delta: float = 2.5,
    lateral_damping: float = 0.72,
    lateral_passes: int = 2,
    lateral_max_correction: float = 0.40,
    lateral_turn_keep: float = 0.62,
    lateral_turn_angle_deg: float = 10.0,
) -> List[carla.Transform]:
    """
    Smooth replay transforms with adaptive filtering while preserving endpoints.
    This removes high-frequency lateral jitter from logged actor trajectories.
    """
    if len(transforms) < 3:
        return [_copy_transform(tf) for tf in transforms]

    src = [_copy_transform(tf) for tf in transforms]
    out: List[carla.Transform] = []

    fx = _OneEuroScalar(min_cutoff=min_cutoff_xy, beta=beta_xy, d_cutoff=d_cutoff)
    fy = _OneEuroScalar(min_cutoff=min_cutoff_xy, beta=beta_xy, d_cutoff=d_cutoff)
    fz = _OneEuroScalar(min_cutoff=min_cutoff_z, beta=beta_z, d_cutoff=d_cutoff)
    fyaw = _OneEuroScalar(min_cutoff=min_cutoff_yaw, beta=beta_yaw, d_cutoff=d_cutoff)

    has_times = bool(times) and len(times) == len(src)
    default_dt = 0.05

    prev_raw_yaw = math.radians(float(src[0].rotation.yaw))
    prev_unwrapped_yaw = prev_raw_yaw
    prev_time = float(times[0]) if has_times else 0.0

    for idx, tf in enumerate(src):
        if idx == 0:
            dt = default_dt
        else:
            if has_times:
                cur_t = float(times[idx])
                dt = cur_t - prev_time
                prev_time = cur_t
            else:
                dt = default_dt
            dt = max(1e-3, min(0.25, float(dt)))

        x = fx.filter(float(tf.location.x), dt)
        y = fy.filter(float(tf.location.y), dt)
        z = fz.filter(float(tf.location.z), dt)

        raw_yaw = math.radians(float(tf.rotation.yaw))
        if idx == 0:
            unwrapped_yaw = raw_yaw
        else:
            unwrapped_yaw = _unwrap_angle_rad(raw_yaw, prev_raw_yaw, prev_unwrapped_yaw)
        prev_raw_yaw = raw_yaw
        prev_unwrapped_yaw = unwrapped_yaw
        yaw = _wrap_angle_deg(fyaw.filter(unwrapped_yaw, dt))

        out.append(
            carla.Transform(
                carla.Location(x=x, y=y, z=z),
                carla.Rotation(
                    pitch=float(tf.rotation.pitch),
                    yaw=yaw,
                    roll=float(tf.rotation.roll),
                ),
            )
        )

    # Preserve original endpoints exactly to keep replay alignment and completion timing stable.
    out[0] = _copy_transform(src[0])
    out[-1] = _copy_transform(src[-1])
    out = _suppress_vehicle_lateral_jitter(
        out,
        damping=lateral_damping,
        passes=lateral_passes,
        max_correction=lateral_max_correction,
        turn_keep=lateral_turn_keep,
        turn_angle_deg=lateral_turn_angle_deg,
    )
    if static_hold_enabled:
        out = _stabilize_near_static_vehicle_segments(
            out,
            times,
            total_displacement_threshold=static_total_displacement,
            total_path_length_threshold=static_total_path_length,
            window_min_duration=static_window_min_duration,
            window_max_displacement=static_window_max_displacement,
            window_max_speed=static_window_max_speed,
            window_max_yaw_delta=static_window_max_yaw_delta,
        )
    return out


def _make_stage_transform(index: int, base_transform: Optional[carla.Transform] = None) -> carla.Transform:
    """
    Compute a deterministic below-ground staging transform for log replay actors.
    Actors parked here are effectively invisible to the scene and won't collide.
    """
    if base_transform is not None:
        base_x = float(base_transform.location.x)
        base_y = float(base_transform.location.y)
        base_z = float(base_transform.location.z) - 500.0
    else:
        base_x = 0.0
        base_y = 0.0
        base_z = -500.0
    spacing = 0.5
    cols = 10
    row = index // cols
    col = index % cols
    return carla.Transform(
        carla.Location(
            x=base_x + col * spacing,
            y=base_y + row * spacing,
            z=base_z,
        ),
        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
    )


def _emit_spawn_debug(
    reason: str,
    cfg: dict,
    spawn_tf_dbg: carla.Transform,
    snap_to_road: bool,
    ground_z_dbg: Optional[float],
    stage_replay_dbg: bool,
    world: Optional[carla.World],
    world_map: Optional[carla.Map],
    normalize_actor_z: bool,
    follow_exact: bool,
    log_replay: bool,
    debug_spawn: bool,
) -> None:
    if not debug_spawn:
        return
    try:
        bp_lib = world.get_blueprint_library() if world else None
        model_dbg = cfg.get("model", "unknown")
        bp_available = False
        bp_count = 0
        if bp_lib is not None:
            try:
                bps = bp_lib.filter(str(model_dbg))
                bp_count = len(bps)
                bp_available = bp_count > 0
            except Exception:  # pylint: disable=broad-except
                bp_available = False
        loc = spawn_tf_dbg.location
        rot = spawn_tf_dbg.rotation
        wp_any = None
        wp_drive = None
        wp_sidewalk = None
        if world_map is not None:
            try:
                wp_any = world_map.get_waypoint(
                    loc, project_to_road=False, lane_type=carla.LaneType.Any
                )
            except Exception:  # pylint: disable=broad-except
                wp_any = None
            try:
                wp_drive = world_map.get_waypoint(
                    loc, project_to_road=True, lane_type=carla.LaneType.Driving
                )
            except Exception:  # pylint: disable=broad-except
                wp_drive = None
            try:
                wp_sidewalk = world_map.get_waypoint(
                    loc, project_to_road=True, lane_type=carla.LaneType.Sidewalk
                )
            except Exception:  # pylint: disable=broad-except
                wp_sidewalk = None
        sidewalk_dist = None
        if wp_sidewalk is not None:
            try:
                sidewalk_dist = wp_sidewalk.transform.location.distance(loc)
            except Exception:  # pylint: disable=broad-except
                sidewalk_dist = None
        ground_z = ground_z_dbg
        if ground_z is None:
            ground_z = _resolve_ground_z(world, loc) if world else None
        z_delta = None
        if ground_z is not None:
            try:
                z_delta = float(loc.z) - float(ground_z)
            except Exception:  # pylint: disable=broad-except
                z_delta = None
        # nearest actor distance (to highlight collision/overlap issues)
        nearest = None
        nearest_id = None
        nearest_type = None
        if world is not None:
            try:
                for actor in world.get_actors():
                    try:
                        a_loc = actor.get_location()
                    except Exception:  # pylint: disable=broad-except
                        continue
                    d = a_loc.distance(loc)
                    if nearest is None or d < nearest:
                        nearest = d
                        nearest_id = actor.id
                        nearest_type = actor.type_id
            except Exception:  # pylint: disable=broad-except
                pass
        overlaps = []
        nearest_bbox = None

        def _point_bbox_distance(point_loc, bbox, bbox_tf):
            try:
                inv = bbox_tf.get_inverse_matrix()
            except Exception:  # pylint: disable=broad-except
                return None
            try:
                lx = (
                    inv[0][0] * point_loc.x
                    + inv[0][1] * point_loc.y
                    + inv[0][2] * point_loc.z
                    + inv[0][3]
                )
                ly = (
                    inv[1][0] * point_loc.x
                    + inv[1][1] * point_loc.y
                    + inv[1][2] * point_loc.z
                    + inv[1][3]
                )
                lz = (
                    inv[2][0] * point_loc.x
                    + inv[2][1] * point_loc.y
                    + inv[2][2] * point_loc.z
                    + inv[2][3]
                )
            except Exception:  # pylint: disable=broad-except
                return None
            try:
                lx -= float(bbox.location.x)
                ly -= float(bbox.location.y)
                lz -= float(bbox.location.z)
            except Exception:  # pylint: disable=broad-except
                pass
            try:
                dx = max(abs(lx) - float(bbox.extent.x), 0.0)
                dy = max(abs(ly) - float(bbox.extent.y), 0.0)
                dz = max(abs(lz) - float(bbox.extent.z), 0.0)
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                inside = dx == 0.0 and dy == 0.0 and dz == 0.0
                return dist, inside
            except Exception:  # pylint: disable=broad-except
                return None

        def _consider_bbox(obj_kind, obj_id, obj_type, obj_loc, bbox, bbox_tf):
            nonlocal nearest_bbox
            result = _point_bbox_distance(loc, bbox, bbox_tf)
            if result is None:
                return
            dist, inside = result
            if nearest_bbox is None or dist < nearest_bbox[0]:
                nearest_bbox = (dist, obj_kind, obj_id, obj_type)
            if inside or dist < 0.25:
                overlaps.append((dist, obj_kind, obj_id, obj_type, inside))

        if world is not None:
            try:
                for actor in world.get_actors():
                    try:
                        a_loc = actor.get_location()
                    except Exception:  # pylint: disable=broad-except
                        continue
                    try:
                        if a_loc.distance(loc) > 20.0:
                            continue
                    except Exception:  # pylint: disable=broad-except
                        pass
                    try:
                        bbox = actor.bounding_box
                        tf = actor.get_transform()
                    except Exception:  # pylint: disable=broad-except
                        continue
                    _consider_bbox("actor", actor.id, actor.type_id, a_loc, bbox, tf)
            except Exception:  # pylint: disable=broad-except
                pass
            env_objects = []
            try:
                label_any = getattr(carla.CityObjectLabel, "Any", None)
                if label_any is not None:
                    env_objects = world.get_environment_objects(label_any)
                else:
                    env_objects = world.get_environment_objects()
            except Exception:  # pylint: disable=broad-except
                env_objects = []
            if not env_objects:
                for label_name in (
                    "TrafficLight",
                    "TrafficSign",
                    "Pole",
                    "Static",
                    "Buildings",
                    "Wall",
                    "Fence",
                    "Other",
                ):
                    label = getattr(carla.CityObjectLabel, label_name, None)
                    if label is None:
                        continue
                    try:
                        env_objects.extend(world.get_environment_objects(label))
                    except Exception:  # pylint: disable=broad-except
                        continue
            for env in env_objects:
                try:
                    env_tf = env.transform
                    env_loc = env_tf.location
                except Exception:  # pylint: disable=broad-except
                    env_tf = None
                    env_loc = None
                if env_loc is not None:
                    try:
                        if env_loc.distance(loc) > 20.0:
                            continue
                    except Exception:  # pylint: disable=broad-except
                        pass
                try:
                    bbox = env.bounding_box
                except Exception:  # pylint: disable=broad-except
                    continue
                if env_tf is None:
                    try:
                        env_tf = carla.Transform(bbox.location)
                    except Exception:  # pylint: disable=broad-except
                        env_tf = None
                if env_tf is None:
                    continue
                env_id = getattr(env, "id", None)
                env_type = getattr(env, "type_id", None)
                if env_type is None:
                    env_type = getattr(env, "type", None)
                if env_type is None:
                    env_type = "env_object"
                _consider_bbox("env", env_id, env_type, env_loc, bbox, env_tf)
        probe_logs = []
        if world is not None and bp_lib is not None and bp_count > 0:
            try:
                bp = bp_lib.filter(str(model_dbg))[0]
                for dz in (0.5, 1.0):
                    probe_tf = carla.Transform(
                        carla.Location(x=loc.x, y=loc.y, z=loc.z + dz),
                        carla.Rotation(
                            pitch=rot.pitch,
                            yaw=rot.yaw,
                            roll=rot.roll,
                        ),
                    )
                    probe_actor = None
                    try:
                        probe_actor = world.try_spawn_actor(bp, probe_tf)
                    except Exception as exc:  # pylint: disable=broad-except
                        probe_logs.append("z+{:.1f}=err({})".format(dz, exc))
                        continue
                    if probe_actor is not None:
                        probe_logs.append("z+{:.1f}=ok".format(dz))
                        try:
                            probe_actor.destroy()
                        except Exception:  # pylint: disable=broad-except
                            pass
                        break
                    probe_logs.append("z+{:.1f}=fail".format(dz))
            except Exception as exc:  # pylint: disable=broad-except
                probe_logs.append("err({})".format(exc))
        print("[RouteScenario][SpawnDebug] reason={}".format(reason))
        print("  name={} role={} model={} bp_available={} bp_count={}".format(
            cfg.get("name"), cfg.get("role"), model_dbg, bp_available, bp_count
        ))
        print("  loc=({:.3f},{:.3f},{:.3f}) rot=({:.1f},{:.1f},{:.1f})".format(
            loc.x, loc.y, loc.z, rot.pitch, rot.yaw, rot.roll
        ))
        print("  normalize_z={} snap_to_road={} follow_exact={} log_replay={} stage_replay={}".format(
            normalize_actor_z, snap_to_road, follow_exact, log_replay, stage_replay_dbg
        ))
        if z_delta is not None:
            print("  ground_z={:.3f} z_delta={:.3f}".format(float(ground_z), float(z_delta)))
        if wp_any is not None:
            print("  wp_any: road_id={} lane_id={} lane_type={} junction={}".format(
                wp_any.road_id, wp_any.lane_id, wp_any.lane_type, wp_any.is_junction
            ))
        if wp_drive is not None:
            print("  wp_drive: road_id={} lane_id={} lane_type={} junction={}".format(
                wp_drive.road_id, wp_drive.lane_id, wp_drive.lane_type, wp_drive.is_junction
            ))
        if wp_sidewalk is not None:
            print("  wp_sidewalk: road_id={} lane_id={} lane_type={} junction={}".format(
                wp_sidewalk.road_id, wp_sidewalk.lane_id, wp_sidewalk.lane_type, wp_sidewalk.is_junction
            ))
        is_walker = str(cfg.get("role", "")).lower() in ("walker", "pedestrian")
        if not is_walker and str(model_dbg).lower().startswith("walker."):
            is_walker = True
        if is_walker:
            if wp_sidewalk is None:
                print("  walker_nav: no_sidewalk_wp")
            elif sidewalk_dist is not None:
                print("  walker_nav: sidewalk_dist={:.3f}m".format(float(sidewalk_dist)))
        if nearest is not None:
            print("  nearest_actor: id={} type={} dist={:.3f}m".format(
                nearest_id, nearest_type, float(nearest)
            ))
        if overlaps:
            overlaps.sort(key=lambda item: item[0])
            for dist, kind, obj_id, obj_type, inside in overlaps[:5]:
                print(
                    "  overlap_{}: id={} type={} dist={:.3f}m inside={}".format(
                        kind, obj_id, obj_type, float(dist), inside
                    )
                )
        if nearest_bbox is not None:
            print(
                "  nearest_bbox: kind={} id={} type={} dist={:.3f}m".format(
                    nearest_bbox[1],
                    nearest_bbox[2],
                    nearest_bbox[3],
                    float(nearest_bbox[0]),
                )
            )
        if probe_logs:
            print("  probe_spawn: {}".format(", ".join(probe_logs)))
    except Exception:  # pylint: disable=broad-except
        pass


def _parse_lift_steps(raw: str) -> List[float]:
    lifts = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            lifts.append(float(part))
        except Exception:  # pylint: disable=broad-except
            continue
    return lifts


def _parse_offset_values(raw: str, default_values: Optional[List[float]] = None) -> List[float]:
    values: List[float] = []
    for token in re.split(r"[,\s]+", str(raw or "").strip()):
        if not token:
            continue
        try:
            values.append(float(token))
        except Exception:  # pylint: disable=broad-except
            continue
    if not values and default_values is not None:
        values = [float(v) for v in default_values]
    if not values:
        values = [0.0]
    # unique with stable order (rounded to reduce floating point duplicate noise)
    dedup: List[float] = []
    seen = set()
    for value in values:
        key = round(float(value), 6)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(float(value))
    if 0.0 not in seen:
        dedup.insert(0, 0.0)
    return dedup


def _build_spawn_retry_offsets(
    xy_offsets: List[float],
    z_offsets: List[float],
    max_attempts: int,
) -> List[Tuple[float, float, float]]:
    combos: List[Tuple[float, float, float]] = []
    for dx in xy_offsets:
        for dy in xy_offsets:
            for dz in z_offsets:
                if abs(dx) < 1e-8 and abs(dy) < 1e-8 and abs(dz) < 1e-8:
                    continue
                combos.append((float(dx), float(dy), float(dz)))
    combos.sort(
        key=lambda item: (
            abs(item[0]) + abs(item[1]) + 0.75 * abs(item[2]),
            abs(item[2]),
            abs(item[0]) + abs(item[1]),
        )
    )
    dedup: List[Tuple[float, float, float]] = []
    seen = set()
    for dx, dy, dz in combos:
        key = (round(dx, 4), round(dy, 4), round(dz, 4))
        if key in seen:
            continue
        seen.add(key)
        dedup.append((dx, dy, dz))
        if max_attempts > 0 and len(dedup) >= max_attempts:
            break
    return dedup


def _spawn_with_offsets(
    model: str,
    rolename: str,
    spawn_tf: carla.Transform,
    offsets: List[Tuple[float, float, float]],
    world: Optional[carla.World],
    world_map: Optional[carla.Map],
    lane_type: carla.LaneType,
    normalize_actor_z: bool = False,
    autopilot: bool = False,
) -> tuple:
    last_exc = None
    attempts = 0
    for dx, dy, dz in offsets:
        attempts += 1
        candidate = carla.Transform(
            carla.Location(
                x=float(spawn_tf.location.x) + float(dx),
                y=float(spawn_tf.location.y) + float(dy),
                z=float(spawn_tf.location.z) + float(dz),
            ),
            carla.Rotation(
                pitch=spawn_tf.rotation.pitch,
                yaw=spawn_tf.rotation.yaw,
                roll=spawn_tf.rotation.roll,
            ),
        )
        if normalize_actor_z:
            candidates: List[float] = []
            ground_z = _resolve_ground_z(world, candidate.location)
            if ground_z is not None:
                candidates.append(float(ground_z))
            if world_map is not None:
                try:
                    snapped_wp = world_map.get_waypoint(
                        candidate.location,
                        project_to_road=True,
                        lane_type=lane_type,
                    )
                    if snapped_wp is not None:
                        candidates.append(float(snapped_wp.transform.location.z))
                except Exception:  # pylint: disable=broad-except
                    pass
            if candidates:
                target_z = min(candidates, key=lambda z: abs(z - float(candidate.location.z)))
                candidate.location.z = float(target_z)
        try:
            actor = CarlaDataProvider.request_new_actor(
                model,
                candidate,
                rolename=rolename,
                autopilot=autopilot,
            )
            if actor is not None:
                return actor, candidate, (float(dx), float(dy), float(dz)), None, attempts
        except Exception as exc:  # pylint: disable=broad-except
            last_exc = exc
            continue
    return None, None, None, last_exc, attempts


def _spawn_with_lifts(
    model: str,
    rolename: str,
    spawn_tf: carla.Transform,
    lifts: List[float],
    autopilot: bool = False,
) -> tuple:
    last_exc = None
    for lift in lifts:
        try:
            candidate = carla.Transform(
                carla.Location(
                    x=spawn_tf.location.x,
                    y=spawn_tf.location.y,
                    z=spawn_tf.location.z + float(lift),
                ),
                spawn_tf.rotation,
            )
            actor = CarlaDataProvider.request_new_actor(
                model,
                candidate,
                rolename=rolename,
                autopilot=autopilot,
            )
            if actor is not None:
                return actor, lift, None
        except Exception as exc:  # pylint: disable=broad-except
            last_exc = exc
            continue
    return None, None, last_exc


def _ground_actor_transform(
    actor: carla.Actor,
    base_tf: carla.Transform,
    world_map: Optional[carla.Map],
    world: Optional[carla.World],
    lane_type: carla.LaneType,
) -> Optional[carla.Transform]:
    if actor is None:
        return None
    temp = [
        carla.Transform(
            carla.Location(
                x=base_tf.location.x,
                y=base_tf.location.y,
                z=base_tf.location.z,
            ),
            carla.Rotation(
                pitch=base_tf.rotation.pitch,
                yaw=base_tf.rotation.yaw,
                roll=base_tf.rotation.roll,
            ),
        )
    ]
    _glue_plan_to_ground(temp, actor, world_map, lane_type, world)
    return temp[0] if temp else None


class LogReplayFollower(py_trees.behaviour.Behaviour):
    """
    Replay a recorded trajectory with explicit timing. Each waypoint has an absolute time in seconds.
    Optionally stage the actor off-map before the first timestamp and after the last timestamp.
    """

    def __init__(
        self,
        actor,
        plan_transforms,
        plan_times,
        name: str = "LogReplayFollower",
        stage_transform: Optional[carla.Transform] = None,
        stage_before: bool = False,
        stage_after: bool = False,
        fail_on_exception: bool = True,
        done_blackboard_key: Optional[str] = None,
        capture_callback=None,
        capture_mid_time: Optional[float] = None,
        record_list=None,
        finalize_callback=None,
        spawn_cb=None,
        spawn_time: Optional[float] = None,
        despawn_cb=None,
        spawn_grace: Optional[float] = None,
        ground_each_tick: bool = False,
        ground_lane_type: Optional[carla.LaneType] = None,
        ground_world_map: Optional[carla.Map] = None,
        ground_world: Optional[carla.World] = None,
        ground_prefer_ray: bool = False,
        ground_z_extra: float = 0.0,
        animate_walkers: Optional[bool] = None,
        walker_max_speed: Optional[float] = None,
        walker_teleport_distance: Optional[float] = None,
    ):
        super().__init__(name)
        self._actor = actor
        self._plan = list(plan_transforms or [])
        self._times = list(plan_times or [])
        self._start_time = None
        self._last_index = 0
        self._spawn_cb = spawn_cb
        self._spawn_time = None
        if spawn_time is not None:
            try:
                self._spawn_time = float(spawn_time)
            except Exception:  # pylint: disable=broad-except
                self._spawn_time = None
        self._despawn_cb = despawn_cb
        self._spawn_grace = float(spawn_grace) if spawn_grace is not None else None
        self._ground_each_tick = bool(ground_each_tick)
        self._ground_lane_type = (
            ground_lane_type if ground_lane_type is not None else carla.LaneType.Any
        )
        self._ground_world_map = ground_world_map
        self._ground_world = ground_world
        self._ground_prefer_ray = bool(ground_prefer_ray)
        self._ground_z_extra = float(ground_z_extra)
        if animate_walkers is None:
            animate_walkers = os.environ.get("CUSTOM_LOG_REPLAY_ANIMATE_WALKERS", "1").lower() in (
                "1",
                "true",
                "yes",
            )
        self._animate_walkers = bool(animate_walkers)
        try:
            self._walker_max_speed = float(
                walker_max_speed
                if walker_max_speed is not None
                else os.environ.get("CUSTOM_LOG_REPLAY_WALKER_MAX_SPEED", "3.0")
            )
        except Exception:  # pylint: disable=broad-except
            self._walker_max_speed = 3.0
        self._walker_max_speed = max(0.2, self._walker_max_speed)
        try:
            self._walker_teleport_distance = float(
                walker_teleport_distance
                if walker_teleport_distance is not None
                else os.environ.get("CUSTOM_LOG_REPLAY_WALKER_TELEPORT_DIST", "1.2")
            )
        except Exception:  # pylint: disable=broad-except
            self._walker_teleport_distance = 1.2
        self._walker_teleport_distance = max(0.1, self._walker_teleport_distance)
        self._last_walker_time: Optional[float] = None
        self._spawned_once = bool(self._actor)
        self._valid = len(self._plan) == len(self._times) and len(self._plan) >= 1 and (
            self._actor is not None or self._spawn_cb is not None
        )
        self._stage_tf = stage_transform
        self._stage_before = bool(stage_before and self._stage_tf is not None)
        self._stage_after = bool(stage_after and self._stage_tf is not None)
        self._fail_on_exception = bool(fail_on_exception)
        self._disabled = False
        self._completed = False
        self._done_bb_key = done_blackboard_key
        self._staged_before = False
        self._staged_after = False
        self._capture_cb = capture_callback
        self._captured_spawn = False
        self._captured_mid = False
        self._captured_post = False
        self._record_list = record_list
        self._finalize_cb = finalize_callback
        self._finalized = False
        if capture_mid_time is not None:
            self._capture_mid_time = float(capture_mid_time)
        elif self._times:
            self._capture_mid_time = 0.5 * (float(self._times[0]) + float(self._times[-1]))
        else:
            self._capture_mid_time = None
        if self._spawn_time is None and self._times:
            self._spawn_time = float(self._times[0])
        if (
            self._spawn_grace is None
            and self._spawn_time is not None
            and self._times
        ):
            try:
                self._spawn_grace = max(0.0, float(self._times[-1]) - float(self._spawn_time))
            except Exception:  # pylint: disable=broad-except
                self._spawn_grace = None
        if self._spawn_cb is not None:
            # Disable staging when using deferred spawn.
            self._stage_before = False
            self._stage_after = False

        self._debug = os.environ.get("CUSTOM_LOG_REPLAY_DEBUG", "").lower() in ("1", "true", "yes")
        try:
            self._debug_interval = float(os.environ.get("CUSTOM_LOG_REPLAY_DEBUG_INTERVAL", "2.0"))
        except Exception:  # pylint: disable=broad-except
            self._debug_interval = 2.0
        lead_raw = str(os.environ.get("CUSTOM_LOG_REPLAY_TIME_LEAD", "auto")).strip().lower()
        if lead_raw in ("", "auto"):
            lead_seconds = 0.0
            try:
                world = CarlaDataProvider.get_world()
                if world is not None:
                    settings = world.get_settings()
                    fixed_delta = getattr(settings, "fixed_delta_seconds", None)
                    if fixed_delta is not None:
                        lead_seconds = float(fixed_delta)
            except Exception:  # pylint: disable=broad-except
                lead_seconds = 0.0
        else:
            try:
                lead_seconds = float(lead_raw)
            except Exception:  # pylint: disable=broad-except
                lead_seconds = 0.0
        self._replay_time_lead = max(0.0, min(0.25, float(lead_seconds)))
        self._last_debug_time = None
        self._last_sim_time = None
        self._last_index_dbg = None

    def _should_animate_walker(self) -> bool:
        return self._animate_walkers and _is_walker_actor(self._actor)

    def _estimate_segment_speed(self) -> float:
        if self._last_index >= len(self._plan) - 1 or self._last_index >= len(self._times) - 1:
            return 0.0
        t0 = float(self._times[self._last_index])
        t1 = float(self._times[self._last_index + 1])
        dt = max(1e-3, t1 - t0)
        a = self._plan[self._last_index].location
        b = self._plan[self._last_index + 1].location
        dist = math.sqrt(
            (float(b.x) - float(a.x)) ** 2
            + (float(b.y) - float(a.y)) ** 2
            + (float(b.z) - float(a.z)) ** 2
        )
        return dist / dt

    def _apply_walker_replay_control(self, target: carla.Transform, sim_time: float) -> bool:
        if self._actor is None:
            return False
        try:
            loc = self._actor.get_location()
        except Exception:  # pylint: disable=broad-except
            return False

        dx = float(target.location.x) - float(loc.x)
        dy = float(target.location.y) - float(loc.y)
        dz = float(target.location.z) - float(loc.z)
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        if dist > self._walker_teleport_distance:
            try:
                self._actor.set_transform(target)
            except Exception:  # pylint: disable=broad-except
                return False
            self._last_walker_time = sim_time
            return True

        dt = 0.05
        if self._last_walker_time is not None:
            dt = max(1e-3, min(0.25, float(sim_time - self._last_walker_time)))
        self._last_walker_time = sim_time

        desired_speed = self._estimate_segment_speed()
        if desired_speed <= 1e-3:
            desired_speed = dist / dt
        desired_speed = min(self._walker_max_speed, max(0.0, float(desired_speed)))
        if dist < 0.02:
            desired_speed = 0.0

        if dist > 1e-4:
            direction = carla.Vector3D(dx / dist, dy / dist, dz / dist)
        else:
            try:
                direction = self._actor.get_transform().get_forward_vector()
            except Exception:  # pylint: disable=broad-except
                direction = carla.Vector3D(1.0, 0.0, 0.0)

        try:
            control = self._actor.get_control()
            if not isinstance(control, carla.WalkerControl):
                control = carla.WalkerControl()
        except Exception:  # pylint: disable=broad-except
            control = carla.WalkerControl()
        control.speed = float(desired_speed)
        control.direction = direction

        try:
            self._actor.apply_control(control)
            return True
        except Exception:  # pylint: disable=broad-except
            return False

    def _maybe_capture(self, stage: str):
        if self._capture_cb is None:
            return
        try:
            self._capture_cb(stage)
        except Exception:  # pylint: disable=broad-except
            pass

    def _maybe_finalize(self):
        if self._finalized or self._finalize_cb is None:
            return
        try:
            self._finalize_cb()
        except Exception:  # pylint: disable=broad-except
            pass
        self._finalized = True

    def _debug_log(self, sim_time: float, note: str = "", target: Optional[carla.Transform] = None) -> None:
        if not self._debug:
            return
        if self._last_debug_time is not None and sim_time - self._last_debug_time < self._debug_interval:
            return
        self._last_debug_time = sim_time
        try:
            abs_time = float(GameTime.get_time())
        except Exception:  # pylint: disable=broad-except
            abs_time = None
        msg = f"[LOG_REPLAY_DEBUG] {self.name}: t={sim_time:.3f}"
        if abs_time is not None:
            msg += f" abs={abs_time:.3f}"
        msg += f" idx={self._last_index}/{max(0, len(self._times)-1)}"
        if note:
            msg += f" {note}"
        if target is not None:
            loc = target.location
            msg += f" target=({loc.x:.2f},{loc.y:.2f},{loc.z:.2f})"
        if self._actor is None:
            msg += " actor=None"
        else:
            try:
                aloc = self._actor.get_location()
                msg += f" actor=({aloc.x:.2f},{aloc.y:.2f},{aloc.z:.2f})"
            except Exception:  # pylint: disable=broad-except
                msg += " actor=unavailable"
        print(msg)

    def initialise(self):
        if self._completed:
            self._disabled = True
            return
        reset_on_init = os.environ.get("CUSTOM_LOG_REPLAY_RESET_ON_INIT", "").lower() in ("1", "true", "yes")
        if reset_on_init or self._start_time is None:
            self._start_time = GameTime.get_time()
        self._last_index = 0
        if self._done_bb_key:
            try:
                py_trees.blackboard.Blackboard().set(self._done_bb_key, False, overwrite=True)
            except Exception:  # pylint: disable=broad-except
                pass
        if self._valid and self._actor is not None:
            if not self._should_animate_walker():
                try:
                    self._actor.set_simulate_physics(False)
                except Exception:  # pylint: disable=broad-except
                    pass
            try:
                if self._stage_before and self._times[0] > 0.0:
                    self._actor.set_transform(self._stage_tf)
                    self._staged_before = True
                else:
                    self._actor.set_transform(self._plan[0])
            except Exception:  # pylint: disable=broad-except
                pass
            if not self._captured_spawn:
                self._maybe_capture("spawn")
                self._captured_spawn = True
        self._last_debug_time = None
        self._last_sim_time = None
        self._last_index_dbg = None
        self._last_walker_time = None

    def _compute_target(self, sim_time: float) -> carla.Transform:
        if sim_time <= self._times[0]:
            return self._plan[0]
        if sim_time >= self._times[-1]:
            return self._plan[-1]
        while self._last_index < len(self._times) - 2 and self._times[self._last_index + 1] <= sim_time:
            self._last_index += 1
        t0 = self._times[self._last_index]
        t1 = self._times[self._last_index + 1]
        if t1 <= t0:
            alpha = 0.0
        else:
            alpha = (sim_time - t0) / (t1 - t0)
        alpha = max(0.0, min(1.0, alpha))
        return _interp_transform(self._plan[self._last_index], self._plan[self._last_index + 1], alpha)

    def update(self):
        if self._disabled:
            return py_trees.common.Status.SUCCESS
        if not self._valid:
            return py_trees.common.Status.SUCCESS

        if self._start_time is None:
            self._start_time = GameTime.get_time()

        sim_time = GameTime.get_time() - self._start_time
        replay_time = sim_time + self._replay_time_lead

        if self._debug and self._last_sim_time is not None and sim_time + 1e-3 < self._last_sim_time:
            print(f"[LOG_REPLAY_DEBUG] {self.name}: sim_time went backwards ({self._last_sim_time:.3f} -> {sim_time:.3f})")
        if self._debug:
            self._last_sim_time = sim_time

        if self._actor is None and self._spawn_cb is not None:
            spawn_time = self._spawn_time if self._spawn_time is not None else 0.0
            if replay_time < spawn_time:
                self._debug_log(sim_time, note=f"waiting_spawn t0={spawn_time:.2f}")
                return py_trees.common.Status.RUNNING
            try:
                self._actor = self._spawn_cb()
                if self._actor is not None:
                    self._spawned_once = True
                    if not self._should_animate_walker():
                        try:
                            self._actor.set_simulate_physics(False)
                        except Exception:  # pylint: disable=broad-except
                            pass
                    if not self._captured_spawn:
                        self._maybe_capture("spawn")
                        self._captured_spawn = True
                else:
                    if self._spawn_grace is not None and replay_time > (spawn_time + self._spawn_grace):
                        self._completed = True
                        self._disabled = True
                        self._maybe_finalize()
                        if self._done_bb_key:
                            try:
                                py_trees.blackboard.Blackboard().set(self._done_bb_key, True, overwrite=True)
                            except Exception:  # pylint: disable=broad-except
                                pass
                        return py_trees.common.Status.SUCCESS
                    self._debug_log(sim_time, note="spawn_cb_none")
                    return py_trees.common.Status.RUNNING
            except Exception as exc:  # pylint: disable=broad-except
                if self._fail_on_exception:
                    return py_trees.common.Status.FAILURE
                print(f"[LogReplayFollower] {self.name}: spawn failed ({exc}); retrying.")
                self._debug_log(sim_time, note="spawn_cb_exception")
                return py_trees.common.Status.RUNNING

        if self._actor is None:
            self._debug_log(sim_time, note="actor_missing")
            return py_trees.common.Status.RUNNING

        if self._stage_before and replay_time < self._times[0]:
            if not self._staged_before:
                try:
                    self._actor.set_transform(self._stage_tf)
                except Exception:  # pylint: disable=broad-except
                    if self._fail_on_exception:
                        return py_trees.common.Status.FAILURE
                    print(f"[LogReplayFollower] {self.name}: failed to stage actor; disabling replay.")
                    self._completed = True
                    self._disabled = True
                    self._maybe_finalize()
                    return py_trees.common.Status.SUCCESS
                self._staged_before = True
            self._debug_log(sim_time, note="staged_before")
            return py_trees.common.Status.RUNNING

        target = self._compute_target(replay_time)
        if self._ground_each_tick and self._actor is not None:
            if self._ground_world is None:
                try:
                    self._ground_world = CarlaDataProvider.get_world()
                except Exception:  # pylint: disable=broad-except
                    self._ground_world = None
            if self._ground_world_map is None:
                try:
                    self._ground_world_map = CarlaDataProvider.get_map()
                except Exception:  # pylint: disable=broad-except
                    self._ground_world_map = None
            grounded_target = _copy_transform(target)
            _glue_plan_to_ground(
                [grounded_target],
                self._actor,
                self._ground_world_map,
                self._ground_lane_type,
                self._ground_world,
                prefer_ray_ground=self._ground_prefer_ray,
                z_extra=self._ground_z_extra,
            )
            target = grounded_target
        if self._debug and self._last_index_dbg is not None and self._last_index < self._last_index_dbg:
            print(
                f"[LOG_REPLAY_DEBUG] {self.name}: index regressed ({self._last_index_dbg} -> {self._last_index})"
            )
        if self._debug:
            self._last_index_dbg = self._last_index
        self._debug_log(sim_time, note="replay", target=target)

        if self._should_animate_walker():
            applied = self._apply_walker_replay_control(target, sim_time)
            if not applied:
                try:
                    self._actor.set_transform(target)
                except Exception as exc:  # pylint: disable=broad-except
                    if self._fail_on_exception:
                        return py_trees.common.Status.FAILURE
                    print(f"[LogReplayFollower] {self.name}: set_transform failed ({exc}); disabling replay.")
                    self._completed = True
                    self._disabled = True
                    self._maybe_finalize()
                    return py_trees.common.Status.SUCCESS
        else:
            try:
                self._actor.set_transform(target)
            except Exception as exc:  # pylint: disable=broad-except
                if self._fail_on_exception:
                    return py_trees.common.Status.FAILURE
                print(f"[LogReplayFollower] {self.name}: set_transform failed ({exc}); disabling replay.")
                self._completed = True
                self._disabled = True
                self._maybe_finalize()
                return py_trees.common.Status.SUCCESS

        if self._record_list is not None:
            try:
                self._record_list.append((float(replay_time), target))
            except Exception:  # pylint: disable=broad-except
                pass

        if (
            not self._captured_mid
            and self._capture_mid_time is not None
            and replay_time >= self._capture_mid_time
            and replay_time <= self._times[-1]
        ):
            self._maybe_capture("mid")
            self._captured_mid = True

        if replay_time >= self._times[-1]:
            if self._should_animate_walker():
                try:
                    stop_control = self._actor.get_control()
                    if not isinstance(stop_control, carla.WalkerControl):
                        stop_control = carla.WalkerControl()
                    stop_control.speed = 0.0
                    self._actor.apply_control(stop_control)
                except Exception:  # pylint: disable=broad-except
                    pass
            if self._stage_after and not self._staged_after:
                try:
                    self._actor.set_transform(self._stage_tf)
                except Exception:  # pylint: disable=broad-except
                    if self._fail_on_exception:
                        return py_trees.common.Status.FAILURE
                    print(f"[LogReplayFollower] {self.name}: failed to stage after replay; disabling replay.")
                    self._completed = True
                    self._disabled = True
                    self._maybe_finalize()
                    return py_trees.common.Status.SUCCESS
                self._staged_after = True
            if not self._captured_post:
                self._maybe_capture("post")
                self._captured_post = True
            if self._despawn_cb is not None and self._actor is not None:
                try:
                    self._despawn_cb(self._actor)
                except Exception:  # pylint: disable=broad-except
                    pass
                self._actor = None
            if self._done_bb_key:
                try:
                    py_trees.blackboard.Blackboard().set(self._done_bb_key, True, overwrite=True)
                except Exception:  # pylint: disable=broad-except
                    pass
            self._completed = True
            self._disabled = True
            self._maybe_finalize()
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

# pylint: disable=line-too-long
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration, ActorConfigurationData
# pylint: enable=line-too-long
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    Idle,
    ScenarioTriggerer,
    WaypointFollower,
    StopVehicle,
    LaneChange,
    HandBrakeVehicle,
    TerminateWaypointFollower,
)
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import InTriggerDistanceToVehicle
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenarios.control_loss import ControlLoss
from srunner.scenarios.follow_leading_vehicle import FollowLeadingVehicle
from srunner.scenarios.object_crash_vehicle import DynamicObjectCrossing
from srunner.scenarios.object_crash_intersection import VehicleTurningRoute
from srunner.scenarios.other_leading_vehicle import OtherLeadingVehicle
from srunner.scenarios.maneuver_opposite_direction import ManeuverOppositeDirection
from srunner.scenarios.junction_crossing_route import SignalJunctionCrossingRoute, NoSignalJunctionCrossingRoute

from srunner.scenariomanager.scenarioatomics.atomic_criteria import (CollisionTest,
                                                                     InRouteTest,
                                                                     RouteCompletionTest,
                                                                     OutsideRouteLanesTest,
                                                                     RunningRedLightTest,
                                                                     RunningStopTest,
                                                                     ActorSpeedAboveThresholdTest)
from srunner.tools.scenario_helper import get_location_in_distance_from_wp

from leaderboard.utils.route_parser import RouteParser, TRIGGER_THRESHOLD, TRIGGER_ANGLE_THRESHOLD, get_ego_vehicle_model
from leaderboard.utils.route_manipulation import interpolate_trajectory
from leaderboard.sensors.fixed_sensors import TrafficLightSensor
from simulation.scenario_runner.srunner.scenarios import ScenarioClassRegistry


ROUTESCENARIO = ["RouteScenario"]

SECONDS_GIVEN_PER_METERS = 0.8 # for timeout
INITIAL_SECONDS_DELAY = 8.0

NUMBER_CLASS_TRANSLATION = {
    "Scenario1": ControlLoss,
    "Scenario2": FollowLeadingVehicle,
    "Scenario3": DynamicObjectCrossing,
    "Scenario4": VehicleTurningRoute,
    "Scenario5": OtherLeadingVehicle,
    "Scenario6": ManeuverOppositeDirection,
    "Scenario7": SignalJunctionCrossingRoute,
    "Scenario8": SignalJunctionCrossingRoute,
    "Scenario9": SignalJunctionCrossingRoute,
    "Scenario10": NoSignalJunctionCrossingRoute
}


def oneshot_behavior(name, variable_name, behaviour):
    """
    This is taken from py_trees.idiom.oneshot.
    """
    # Initialize the variables
    blackboard = py_trees.blackboard.Blackboard()
    _ = blackboard.set(variable_name, False)

    # Wait until the scenario has ended
    subtree_root = py_trees.composites.Selector(name=name)
    check_flag = py_trees.blackboard.CheckBlackboardVariable(
        name=variable_name + " Done?",
        variable_name=variable_name,
        expected_value=True,
        clearing_policy=py_trees.common.ClearingPolicy.ON_INITIALISE
    )
    set_flag = py_trees.blackboard.SetBlackboardVariable(
        name="Mark Done",
        variable_name=variable_name,
        variable_value=True
    )
    # If it's a sequence, don't double-nest it in a redundant manner
    if isinstance(behaviour, py_trees.composites.Sequence):
        behaviour.add_child(set_flag)
        sequence = behaviour
    else:
        sequence = py_trees.composites.Sequence(name="OneShot")
        sequence.add_children([behaviour, set_flag])

    subtree_root.add_children([check_flag, sequence])
    return subtree_root


def convert_json_to_transform(actor_dict):
    """
    Convert a JSON string to a CARLA transform
    """
    return carla.Transform(location=carla.Location(x=float(actor_dict['x']), y=float(actor_dict['y']),
                                                   z=float(actor_dict['z'])),
                           rotation=carla.Rotation(roll=0.0, pitch=0.0, yaw=float(actor_dict['yaw'])))

# NOTE（GJH): Select the scenario according to the proportion of each scenario
def selScenario(scenario_config: dict) -> str:
    """
    
    Select a scenario according to the proportion of each scenario

    Args:
        scenario_config: a dict of scenario definition config containing proportion in this Scenario

    Returns:
        selected_scenario: a string of selected scenario

    """
    try:
        scenarios = list(scenario_config.keys())
        scenario_proportion = []
        for scenario_name in scenarios:
            scenario_proportion.append(scenario_config[scenario_name]["proportion"])
        return np.random.choice(scenarios, 1, p=scenario_proportion).tolist()[0]
    except Exception as e:
        print("Select Scenario Error: ", e, "Using the first scenario in the config file.")
        scenarios = list(scenario_config.keys())
        print(scenarios)
        return scenarios[0]


def convert_json_to_actor(actor_dict):
    """
    Convert a JSON string to an ActorConfigurationData dictionary
    """
    node = ET.Element('waypoint')
    node.set('x', actor_dict['x'])
    node.set('y', actor_dict['y'])
    node.set('z', actor_dict['z'])
    node.set('yaw', actor_dict['yaw'])

    return ActorConfigurationData.parse_from_node(node, 'simulation')


def convert_transform_to_location(transform_vec):
    """
    Convert a vector of transforms to a vector of locations
    """
    location_vec = []
    for transform_tuple in transform_vec:
        location_vec.append((transform_tuple[0].location, transform_tuple[1]))

    return location_vec


def compare_scenarios(scenario_choice, existent_scenario):
    """
    Compare function for scenarios based on distance of the scenario start position
    """
    def transform_to_pos_vec(scenario):
        """
        Convert left/right/front to a meaningful CARLA position
        """
        position_vec = [scenario['trigger_position']]
        if scenario['other_actors'] is not None:
            if 'left' in scenario['other_actors']:
                position_vec += scenario['other_actors']['left']
            if 'front' in scenario['other_actors']:
                position_vec += scenario['other_actors']['front']
            if 'right' in scenario['other_actors']:
                position_vec += scenario['other_actors']['right']

        return position_vec

    # put the positions of the scenario choice into a vec of positions to be able to compare

    choice_vec = transform_to_pos_vec(scenario_choice)
    existent_vec = transform_to_pos_vec(existent_scenario)
    for pos_choice in choice_vec:
        for pos_existent in existent_vec:

            dx = float(pos_choice['x']) - float(pos_existent['x'])
            dy = float(pos_choice['y']) - float(pos_existent['y'])
            dz = float(pos_choice['z']) - float(pos_existent['z'])
            dist_position = math.sqrt(dx * dx + dy * dy + dz * dz)
            dyaw = float(pos_choice['yaw']) - float(pos_choice['yaw'])
            dist_angle = math.sqrt(dyaw * dyaw)
            if dist_position < TRIGGER_THRESHOLD and dist_angle < TRIGGER_ANGLE_THRESHOLD:
                return True

    return False


class RouteScenario(BasicScenario):

    """
    Implementation of a RouteScenario, i.e. a scenario that consists of driving along a pre-defined route,
    along which several smaller scenarios are triggered
    """

    category = "RouteScenario"

    def __init__(self, world, config, debug_mode=0, criteria_enable=True, ego_vehicles_num=1,log_dir=None, scenario_parameter=None,trigger_distance=10):
        """
        Setup all relevant parameters and create scenarios along route

        Args:
            world: carla.libcarla.World
            config: srunner.scenarioconfigs.route_scenario_configuration.RouteScenarioConfiguration,
                    route information(name, town, trajectory, weather)
            ego_vehicles_num: int, number of communicating vehicles
            log_dir: str, directory to save log
            scenario information:
                {crazy_level,
                crazy_propotion,
                trigger_distance
                }

        """

        # load or initialize params
        self.config = config
        self.route = None
        self.sampled_scenarios_definitions = None
        self.ego_vehicles_num=ego_vehicles_num
        self.new_config_trajectory=None
        self.crazy_level = 0
        self.crazy_proportion = 0
        self.trigger_distance = trigger_distance
        self.sensor_tf_num = 0
        self.sensor_tf_list = []
        self.log_dir = log_dir
        
        self.scenario_parameter = scenario_parameter
        self.background_params = scenario_parameter.get('Background',{})

        self.route_scenario_dic = {}
        self._custom_actor_configs = list(getattr(config, "custom_actors", []) or [])
        self._custom_actor_plans: List[dict] = []
        self._custom_actor_spawn_states: Dict[str, dict] = {}
        self._custom_actor_spawn_summary_printed = False
        self._custom_actor_behaviors = list(getattr(config, "custom_actor_behaviors", []) or [])
        self._custom_actor_behavior_by_name = {}
        self._ego_replay_transforms: List[List[carla.Transform]] = []
        self._ego_replay_times: List[Optional[List[float]]] = []
        for entry in self._custom_actor_behaviors:
            if not isinstance(entry, dict):
                continue
            name = (
                entry.get("actor_name")
                or entry.get("name")
                or entry.get("actor")
                or entry.get("id")
            )
            if name:
                self._custom_actor_behavior_by_name[str(name)] = entry

        # store original ego replay plans before route alignment modifies them
        self._build_ego_replay_plans(config)

        # update waypoints and scenarios along the routes
        self._update_route(world, config, debug_mode>0)

        # set traffic sensors
        for j in range(self.sensor_tf_num):
            tf_sensor=TrafficLightSensor(config.save_path_root,j)
            self.sensor_tf_list.append(tf_sensor)
        self._init_tf_sensors()

        # spawn ego vehicles
        ego_vehicles = self._update_ego_vehicle(world)
        # update ego_num, for some ego may fail to spawn
        # self.ego_vehicles_num = len(ego_vehicles)

        
        if self.scenario_parameter is not None:
            self.list_scenarios = self._build_scenario_parameter_instances(world,
                                                             ego_vehicles,
                                                             self.sampled_scenarios_definitions,
                                                             scenarios_per_tick=10,
                                                             timeout=self.timeout,
                                                             debug_mode=debug_mode>1,
                                                             scenario_parameter=self.scenario_parameter)
        else:
            self.list_scenarios = self._build_scenario_instances(world,
                                                                ego_vehicles,
                                                                self.sampled_scenarios_definitions,
                                                                scenarios_per_tick=10,
                                                                timeout=self.timeout,
                                                                debug_mode=debug_mode>1)
        super(RouteScenario, self).__init__(name=config.name,
                                            ego_vehicles=ego_vehicles,
                                            config=config,
                                            world=world,
                                            debug_mode=debug_mode>1,
                                            terminate_on_failure=False,
                                            criteria_enable=criteria_enable)
        print('route_scenarios:',self.route_scenario_dic)

    def _get_multi_tf(self, trajectory, tf_num=1) -> List:
        """
        calculate the closest tfs and return them
        """
        decay_factor=0.5
        tf_list = []
        world = CarlaDataProvider.get_world()
        # lights_list is a list of all tfs
        lights_list = world.get_actors().filter("*traffic_light*")
        tf_tensor = torch.tensor([[light.get_transform().location.x,
                       light.get_transform().location.y,
                       light.get_transform().location.z 
                    ] for light in lights_list],dtype=float)
        # Find top k closest tfs
        dist_tensor=torch.zeros(tf_tensor.shape[0],dtype=float)
        for waypoint in trajectory[0]:
            waypoint_tensor=torch.tensor([waypoint.x,waypoint.y,waypoint.z] 
                                ,dtype=float).reshape(1,3).repeat(tf_tensor.shape[0],1)
            dist_tensor += ((tf_tensor-waypoint_tensor)**2).sum(dim=1,keepdim=False)*decay_factor
            decay_factor = decay_factor ** 2
        # get the idx of top k closest tfs
        _,idx=dist_tensor.topk(tf_num,
                                    largest=False,
                                    sorted=False)
        [tf_list.append(lights_list[idx[j].item()]) for j in range(tf_num)]
        return tf_list

    def _init_tf_sensors(self) -> None:
        """
        This function is to find the proper tf and set up sensors on it.
        """
        tf_list = self._get_multi_tf(self.get_new_config_trajectory().copy(), 
                                          tf_num=self.sensor_tf_num)
        for j in range(self.sensor_tf_num):
            self.sensor_tf_list[j].setup_sensors(tf_list[j])
        return

    def get_sensor_tf(self) -> List:
        return self.sensor_tf_list

    def _build_ego_replay_plans(self, config) -> None:
        """
        Build per-ego transform + timing plans for log replay, using the raw route XML data.
        """
        self._ego_replay_transforms = []
        self._ego_replay_times = []

        if hasattr(config, "multi_traj") and config.multi_traj:
            multi_traj = config.multi_traj
            multi_yaws = getattr(config, "multi_traj_yaws", None)
            multi_pitches = getattr(config, "multi_traj_pitches", None)
            multi_rolls = getattr(config, "multi_traj_rolls", None)
            multi_times = getattr(config, "multi_traj_times", None)
        else:
            multi_traj = [getattr(config, "trajectory", None)]
            multi_yaws = [getattr(config, "trajectory_yaws", None)]
            multi_pitches = [getattr(config, "trajectory_pitches", None)]
            multi_rolls = [getattr(config, "trajectory_rolls", None)]
            multi_times = [getattr(config, "trajectory_times", None)]

        def _yaw_from_segment(a: carla.Location, b: carla.Location) -> float:
            return math.degrees(math.atan2(float(b.y) - float(a.y), float(b.x) - float(a.x)))

        for idx, traj in enumerate(multi_traj):
            if not traj:
                self._ego_replay_transforms.append([])
                self._ego_replay_times.append(None)
                continue

            yaws = multi_yaws[idx] if multi_yaws and idx < len(multi_yaws) else None
            pitches = multi_pitches[idx] if multi_pitches and idx < len(multi_pitches) else None
            rolls = multi_rolls[idx] if multi_rolls and idx < len(multi_rolls) else None
            times = multi_times[idx] if multi_times and idx < len(multi_times) else None

            transforms: List[carla.Transform] = []
            for i, loc in enumerate(traj):
                yaw = yaws[i] if yaws and i < len(yaws) else None
                pitch = pitches[i] if pitches and i < len(pitches) else None
                roll = rolls[i] if rolls and i < len(rolls) else None

                if yaw is None:
                    if i + 1 < len(traj):
                        yaw = _yaw_from_segment(loc, traj[i + 1])
                    elif i > 0:
                        yaw = _yaw_from_segment(traj[i - 1], loc)
                    else:
                        yaw = 0.0
                if pitch is None:
                    pitch = 0.0
                if roll is None:
                    roll = 0.0

                transforms.append(
                    carla.Transform(
                        carla.Location(x=loc.x, y=loc.y, z=loc.z),
                        carla.Rotation(pitch=float(pitch), yaw=float(yaw), roll=float(roll)),
                    )
                )

            times_out = None
            if times and len(times) == len(traj) and all(t is not None for t in times):
                times_out = [float(t) for t in times]

            self._ego_replay_transforms.append(transforms)
            self._ego_replay_times.append(times_out)

    def _cal_multi_routes(self, world: carla.libcarla.World, config) -> List:
        """
        Given the waypoints of one route as anchors, computes waypoints that those ego vehicles will pass by around those anchors.
        Args:
            world: Carla world
            config:
                config.trajectory: list of carla.libcarla.Location, sparse waypoints list
        Returns:
            trajectory: trajectory[i] represents waypoints of the ith ego vehicle, trajectory[i] is list of carla.libcarla.Location
        
        Given config.trajectory[A(start point), B, C, ..., L(end point)]
        ego 0, ego 1, ego 2 will reach B, C, ..., L individually, but start at different location around A, like
        ////////////////////////////
        //      |       ||[ego 1]|        //
        //      |       ||       |        //
        //      |       ||       |        //
        //      |       ||       |        //
        //      |       ||       |        //
        //      |[ego 3]||[ego 0]|        //
        //      |       ||       |        //
        //      |       ||       |        //
        //      |       ||       |        //
        //      |       ||[ego 2]|        //
        //      |       ||       |        //
            

        """
        trajectory=[]
        distance_gap_straight = 12
        distance_gap_left = 0
        distance_gap_right = 0
        distance_gap_rear = 0

        # trajectories for multi-ego-vehicle
        if hasattr(self.config, 'multi_traj'):
            self.new_config_trajectory=config.multi_traj.copy()
            return config.multi_traj

        # trajectory's element is a list of waypoint, a carla.Location object
        trajectory.append(config.trajectory) 
        # initialize trajectory
        trajectory.extend([[] for _ in range(1,self.ego_vehicles_num)])

        # spawn_points is a list of carla.Transform
        spawn_points=world.get_map().get_spawn_points()
        spawn_tensor=torch.tensor([[spawn_point.location.x,
                       spawn_point.location.y,
                       spawn_point.location.z 
                    ] for spawn_point in spawn_points],dtype=float)

        # calculate waypoints via knn
        for point, waypoint in enumerate(trajectory[0]):
            waypoint_tensor=torch.tensor([waypoint.x,waypoint.y,waypoint.z] 
                                ,dtype=float).reshape(1,3).repeat(spawn_tensor.shape[0],1)
            dist_tensor=((spawn_tensor-waypoint_tensor)**2).sum(dim=1,keepdim=False)
            val,idx=dist_tensor.topk(7*self.ego_vehicles_num,
                                        largest=False,
                                        sorted=True
                                        )
            curk=0
            travel_distance_start = 12
            travel_distance_start_pre = 0
            travel_distance_start_rear = 12
            travel_distance_start_rear_pre = 0
            travel_distance_start_right = 10
            travel_distance_start_right_pre = 0
            travel_distance_start_left = 10
            travel_distance_start_left_pre = 0
            waypoint_start = None
            for k in range(1,self.ego_vehicles_num):
                if point == 0:
                    # curk+=1
                    for _ in range(curk, len(val)):
                        waypoint_carla = CarlaDataProvider.get_map().get_waypoint(waypoint)
                        location, travel_distance_start = get_location_in_distance_from_wp(waypoint_carla, travel_distance_start, direction='foward')
                        if abs(travel_distance_start_pre - travel_distance_start)<5.0:
                            waypoint_start_right = waypoint_carla.get_right_lane()
                            if waypoint_start_right:
                                location_right, travel_distance_start_right = get_location_in_distance_from_wp(waypoint_start_right, travel_distance_start_right, direction='foward')
                            
                            waypoint_start_left = waypoint_carla.get_left_lane()
                            if waypoint_start_left:
                                location_left, travel_distance_start_left = get_location_in_distance_from_wp(waypoint_start_left, travel_distance_start_left, direction='rear')
                            
                            location_rear, travel_distance_start_rear = get_location_in_distance_from_wp(waypoint_carla, travel_distance_start_rear, direction='rear')

                            if abs(travel_distance_start_right-travel_distance_start_right_pre)>7.0 \
                                and (not(waypoint_start_right is None or waypoint_start_right.lane_type == carla.LaneType.Sidewalk\
                                    or waypoint_start_right.lane_type == carla.LaneType.Shoulder)):
                                trajectory[k].append(location_right)
                                print("ego{} located at right".format(k))
                                travel_distance_start_right_pre = travel_distance_start_right
                                travel_distance_start_right += distance_gap_right
                                break

                            elif abs(travel_distance_start_rear-travel_distance_start_rear_pre)>8.0:
                                trajectory[k].append(location_rear)
                                print("ego{} located at rear".format(k))
                                travel_distance_start_rear_pre = travel_distance_start_rear
                                travel_distance_start_rear += distance_gap_rear
                                break                                    
                            elif abs(travel_distance_start_left-travel_distance_start_left_pre)>3.0 \
                                and (not(waypoint_start_left is None or waypoint_start_left.lane_type == carla.LaneType.Sidewalk\
                                    or waypoint_start_left.lane_type == carla.LaneType.Shoulder)):                         
                                trajectory[k].append(location_left)
                                print("ego{} located at left".format(k))
                                travel_distance_start_left_pre = travel_distance_start_left
                                travel_distance_start_left += distance_gap_left
                                break
                            else :
                                while min([spawn_points[idx[curk]].location.distance(trajectory[kk-1][0]) for kk in range(1,k+1)]) <20 and curk<len(idx)-2:
                                        curk += 1
                                trajectory[k].append(spawn_points[idx[curk]].location)
                                curk = (curk + 1)%len(idx)
                                print("ego{} located at the closest spawn point".format(k))
                                break
                        else:
                            trajectory[k].append(location)
                            print("ego{} located at forward".format(k))
                            travel_distance_start_pre = travel_distance_start 
                            travel_distance_start  += distance_gap_straight
                            break 
                else:
                    if val[curk]>5.0:
                        trajectory[k].append(waypoint)
                    else:
                        trajectory[k].append(spawn_points[idx[curk]].location)
                        curk = (curk + 1)%len(idx)
        self.new_config_trajectory=trajectory.copy()
        return trajectory

    def _cal_multi_routes_for_parallel_driving(self, world, config):
        """
        Make cars drive in parallel from the start
        """
        trajectory=[]
        # trajectory's element is a list of waypoint, a carla.Location object
        trajectory.append(config.trajectory) 
        # initialize trajectory
        trajectory.extend([[] for _ in range(1,self.ego_vehicles_num)])

        # spawn_points is a list of carla.Transform
        spawn_points=world.get_map().get_spawn_points()
        spawn_tensor=torch.tensor([[spawn_point.location.x,
                       spawn_point.location.y,
                       spawn_point.location.z 
                    ] for spawn_point in spawn_points],dtype=float)

        # calculate waypoints via knn
        for point, waypoint in enumerate(trajectory[0]):
            waypoint_tensor=torch.tensor([waypoint.x,waypoint.y,waypoint.z] 
                                ,dtype=float).reshape(1,3).repeat(spawn_tensor.shape[0],1)
            dist_tensor=((spawn_tensor-waypoint_tensor)**2).sum(dim=1,keepdim=False)
            val,idx=dist_tensor.topk(7*self.ego_vehicles_num,
                                        largest=False,
                                        sorted=True
                                        )
            curk=0

            waypoint_carla = CarlaDataProvider.get_map().get_waypoint(waypoint)
            waypoint_start_right = waypoint_carla.get_right_lane()

            # record lane in this row
            lane_list = []
            lane_list.append(waypoint)
            max_lane_num = 20
            # record lane on the right
            for lane in range(max_lane_num):
                waypoint_lane_changed = waypoint_carla.get_right_lane()
                if waypoint_lane_changed.lane_type == carla.LaneType.Driving:
                    lane_list.insert(0,waypoint_lane_changed)
                else:
                    break
            # record lane on the left
            for lane in range(max_lane_num):
                waypoint_lane_changed = waypoint_carla.get_left_lane()
                if waypoint_lane_changed.lane_type == carla.LaneType.Driving:
                    lane_list.append(waypoint_lane_changed)
                else:
                    break

            print('initial lane num:', len(lane_list))


            for k in range(1,self.ego_vehicles_num):
                if point == 0:
                    trajectory[k].append(lane_list[k].transform.location)

                else:
                    if val[curk]>5.0:
                        trajectory[k].append(waypoint)
                    else:
                        # _route_distance += travel_distance_route
                        # location, travel_distance_route = get_location_in_distance_from_wp(waypoint_carla, _start_distance)
                        trajectory[k].append(spawn_points[idx[curk]].location)
                        curk = (curk + 1)%len(idx)
        # route = open("route.txt",'w')
        # route.write(str(trajectory))
        self.new_config_trajectory=trajectory.copy()
        return trajectory

    def get_new_config_trajectory(self):
        return self.new_config_trajectory

    def draw_route(self):
        """
        draw waypoints coordinates from self.route
        """
        fig = plt.figure(dpi=400)
        colors = ['tab:red','tab:blue','tab:orange', 'tab:purple','tab:green','tab:pink', 'tab:brown', 'tab:gray', 'tab:olive', 'tab:cyan']
        center_x = self.route[0][0][0].location.x
        center_y = self.route[0][0][0].location.y
        for i in range(len(self.route)):
            for j in range(len(self.route[i])):
                point_x = self.route[i][j][0].location.x - center_x + 1*i
                point_y = self.route[i][j][0].location.y - center_y + 1*i
                if j==0:
                    plt.scatter(point_x, point_y, s=50, c=colors[i], label='ego{}'.format(i))
                    plt.text(point_x+0.1, point_y+0.1, 'ego{} start'.format(i))
                elif j==(len(self.route[i])-1):
                    plt.scatter(point_x, point_y, s=50, c=colors[i])
                    plt.text(point_x+0.1, point_y+2*(i+1), 'ego{} end'.format(i))
                else:
                    plt.scatter(point_x, point_y, s=20, c=colors[i])
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(self.log_dir,'point_coordinates.png'))
        plt.close()

    def _update_route(self, world, config, debug_mode):
        """
        Update the input route, i.e. refine waypoint list, and extract possible scenario locations

        Parameters:
            world: CARLA world
            config: Scenario configuration (RouteConfiguration)
        Main target variable:
            self.route: trajectory plan to reach for each ego vehicle
            self.sampled_scenarios_definitions: scenarios will be triggered on each ego vehicle's route
        """

        # Transform the scenario file into a dictionary, defines possible trigger position for each type of scenario 
        world_annotations = RouteParser.parse_annotations_file(config.scenario_file)

        # generate trajectory for ego-vehicles
        # trajectory's element is a list of waypoint(carla.Location object)
        trajectory = self._cal_multi_routes(world, config)
        if os.environ.get("CUSTOM_EGO_LOG_REPLAY", "").lower() not in ("1", "true", "yes"):
            self._align_start_waypoints(world, trajectory, config)
        gps_route=[]
        route=[]
        potential_scenarios_definitions=[]

        # prepare route's trajectory (interpolate and add the GPS route)
        for i, tr in enumerate(trajectory):
            # tr is a list of waypoint, each a carla.Location object
            gps, r = interpolate_trajectory(world, tr)
            gps_route.append(gps)
            route.append(r)
            print('load scenarios for ego{}'.format(i))
            potential_scenarios_definition, _ = RouteParser.scan_route_for_scenarios(
                config.town, r, world_annotations)
            potential_scenarios_definitions.append(potential_scenarios_definition)
        # print(potential_scenarios_definitions)
        # self.route is a list of ego_vehicles' routes
        self.route = route
        if self.log_dir is not None:
            # plot waypoints coordinates
            self.draw_route()
        CarlaDataProvider.set_ego_vehicle_route([convert_transform_to_location(self.route[j]) for j in range(self.ego_vehicles_num)])
        config.agent.set_global_plan(gps_route, self.route)

        # Sample the scenarios to be used for this route instance. A list for ego_vehicles.
        self.sampled_scenarios_definitions = [self._scenario_sampling(potential_scenarios_definition) 
                                                for potential_scenarios_definition in potential_scenarios_definitions]

        # Timeout of each ego_vehicle in scenario in seconds
        self.timeout = self._estimate_route_timeout()

        # Print route in debug mode
        if debug_mode:
            [self._draw_waypoints(world, self.route[j], vertical_shift=1.0, persistency=50000.0) for j in range(self.ego_vehicles_num)]

    def _align_start_waypoints(self, world, trajectory, config) -> None:
        """
        Snap the first waypoint of each ego trajectory to the closest drivable lane that best
        matches the XML heading. This keeps spawn heading aligned with the planned route.
        """
        world_map = CarlaDataProvider.get_map()
        if world_map is None:
            return

        max_snap_dist = 5.0

        for ego_idx, tr in enumerate(trajectory):
            if not tr:
                continue

            traj_yaws = None
            if hasattr(config, "multi_traj_yaws") and config.multi_traj_yaws:
                if ego_idx < len(config.multi_traj_yaws):
                    traj_yaws = config.multi_traj_yaws[ego_idx]
            if traj_yaws is None:
                traj_yaws = getattr(config, "trajectory_yaws", None)

            xml_yaw = None
            if traj_yaws and len(traj_yaws) > 0:
                xml_yaw = traj_yaws[0]

            xml_heading = _heading_from_points(tr)
            desired_yaw = None
            if xml_heading is not None:
                if xml_yaw is None or _angle_delta(xml_heading, xml_yaw) > 45.0:
                    desired_yaw = xml_heading
                else:
                    desired_yaw = xml_yaw
            else:
                desired_yaw = xml_yaw

            if desired_yaw is None:
                continue

            wp = world_map.get_waypoint(
                tr[0],
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if wp is None:
                continue

            candidates = [wp]
            left = wp.get_left_lane()
            right = wp.get_right_lane()
            if left is not None and left.lane_type == carla.LaneType.Driving:
                candidates.append(left)
            if right is not None and right.lane_type == carla.LaneType.Driving:
                candidates.append(right)

            best = min(candidates, key=lambda c: _angle_delta(desired_yaw, c.transform.rotation.yaw))
            if best.transform.location.distance(tr[0]) <= max_snap_dist:
                tr[0] = best.transform.location

    def _update_ego_vehicle(self, world) -> List:
        """
        Set/Update the start position of the ego_vehicles
        Returns:
            ego_vehicles (list): list of ego_vehicles.
        """
        # move ego vehicles to correct position
        ego_vehicles=[]
        normalize_ego_z = os.environ.get("CUSTOM_EGO_NORMALIZE_Z", "").lower() in (
            "1",
            "true",
            "yes",
        )
        normalize_actor_z = os.environ.get("CUSTOM_ACTOR_NORMALIZE_Z", "").lower() in (
            "1",
            "true",
            "yes",
        )
        log_replay_ego = os.environ.get("CUSTOM_EGO_LOG_REPLAY", "").lower() in (
            "1",
            "true",
            "yes",
        )
        world = CarlaDataProvider.get_world()
        world_map = CarlaDataProvider.get_map()


        for j in range(self.ego_vehicles_num):
            if log_replay_ego and j < len(self._ego_replay_transforms) and self._ego_replay_transforms[j]:
                if (normalize_ego_z or normalize_actor_z) and world_map is not None:
                    # Glue the replay plan to the ground using the ego bounding box offset
                    # (actor must exist, so perform after spawn)
                    elevate_transform = self._ego_replay_transforms[j][0]
            else:
                elevate_transform = self.route[j][0][0]
            spawn_tf = carla.Transform(
                carla.Location(
                    x=elevate_transform.location.x,
                    y=elevate_transform.location.y,
                    z=elevate_transform.location.z,
                ),
                carla.Rotation(
                    pitch=elevate_transform.rotation.pitch,
                    yaw=elevate_transform.rotation.yaw,
                    roll=elevate_transform.rotation.roll,
                ),
            )

            if (normalize_ego_z or (log_replay_ego and normalize_actor_z)) and world_map is not None:
                candidates: list[float] = []
                ground_z = _resolve_ground_z(world, spawn_tf.location)
                if ground_z is not None:
                    candidates.append(ground_z)
                snapped_wp = world_map.get_waypoint(
                    spawn_tf.location,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
                if snapped_wp is not None:
                    candidates.append(float(snapped_wp.transform.location.z))
                if candidates:
                    spawn_tf.location.z = min(candidates, key=lambda z: abs(z - float(spawn_tf.location.z)))
            # ego vehicle will float in the air at a height of 0.5m in the first frame
            # (especially useful when spawning from log replay)
            spawn_tf.location.z += 0.5
            print("ego id:{}".format(j))
            print("transform:{}".format(spawn_tf))
            # Get vehicle model from manifest (for promoted NPCs) or use default
            vehicle_model = get_ego_vehicle_model(j, default='vehicle.lincoln.mkz2017')
            print("vehicle model:{}".format(vehicle_model))
            ego_vehicle = CarlaDataProvider.request_new_actor(vehicle_model,
                                                            spawn_tf,
                                                            rolename='hero_{}'.format(j))
            ego_vehicles.append(ego_vehicle)

            if log_replay_ego and ego_vehicle is not None:
                try:
                    ego_vehicle.set_simulate_physics(False)
                except Exception:  # pylint: disable=broad-except
                    pass
                if (normalize_ego_z or normalize_actor_z) and self._ego_replay_transforms[j]:
                    _glue_plan_to_ground(
                        self._ego_replay_transforms[j],
                        ego_vehicle,
                        world_map,
                        carla.LaneType.Driving,
                        world,
                    )
                    try:
                        ego_vehicle.set_transform(self._ego_replay_transforms[j][0])
                    except Exception:  # pylint: disable=broad-except
                        pass

            # set the spectator location above the first ego vehicle
            if j==0:
                spectator = CarlaDataProvider.get_world().get_spectator()
                ego_trans = ego_vehicle.get_transform()
                spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=50),
                                                            carla.Rotation(pitch=-90)))
        return ego_vehicles

    def _estimate_route_timeout(self):
        """
        Estimate the duration of the route
        """
        if os.environ.get("CUSTOM_EGO_LOG_REPLAY", "").lower() in ("1", "true", "yes"):
            max_time = None
            for times in getattr(self, "_ego_replay_times", []) or []:
                if times:
                    t_last = times[-1]
                    if max_time is None or t_last > max_time:
                        max_time = t_last
            if max_time is not None:
                return int(max_time + INITIAL_SECONDS_DELAY)

        route_length = 0.0  # in meters

        prev_point = self.route[0][0][0]
        for current_point, _ in self.route[0][1:]:
            dist = current_point.location.distance(prev_point.location)
            route_length += dist
            prev_point = current_point

        return int(SECONDS_GIVEN_PER_METERS * route_length + INITIAL_SECONDS_DELAY)

    # pylint: disable=no-self-use
    def _draw_waypoints(self, world, waypoints, vertical_shift, persistency=-1):
        """
        Draw a list of waypoints at a certain height given in vertical_shift.
        """
        for w in waypoints:
            wp = w[0].location + carla.Location(z=vertical_shift)

            size = 0.2
            if w[1] == RoadOption.LEFT:  # Yellow
                color = carla.Color(255, 255, 0)
            elif w[1] == RoadOption.RIGHT:  # Cyan
                color = carla.Color(0, 255, 255)
            elif w[1] == RoadOption.CHANGELANELEFT:  # Orange
                color = carla.Color(255, 64, 0)
            elif w[1] == RoadOption.CHANGELANERIGHT:  # Dark Cyan
                color = carla.Color(0, 64, 255)
            elif w[1] == RoadOption.STRAIGHT:  # Gray
                color = carla.Color(128, 128, 128)
            else:  # LANEFOLLOW
                color = carla.Color(0, 255, 0) # Green
                size = 0.1

            world.debug.draw_point(wp, size=size, color=color, life_time=persistency)

        world.debug.draw_point(waypoints[0][0].location + carla.Location(z=vertical_shift), size=0.2,
                               color=carla.Color(0, 0, 255), life_time=persistency)
        world.debug.draw_point(waypoints[-1][0].location + carla.Location(z=vertical_shift), size=0.2,
                               color=carla.Color(255, 0, 0), life_time=persistency)

    def _scenario_sampling(self, potential_scenarios_definitions, random_seed=0):
        """
        The function used to sample the scenarios that are going to happen for this route.
        Args:
            potential_scenarios_definitions: OrderedDict, len(possible_scenarios) is the number of position that is possible to trigger scenarios along this route, 
                                and possible_scenarios[i] is a list of scenarios that is possible to be triggered at the ith position
        Returns:
            sampled_scenarios: list of Dict(), sampled from possible scenarios, sampled_scenarios[i] represents the ith scenario to be triggered along this route
        """

        # fix the random seed for reproducibility
        rgn = random.RandomState(random_seed)

        def position_sampled(scenario_choice, sampled_scenarios):
            """
            Check if a position was already sampled, i.e. used for another scenario
            """
            for existent_scenario in sampled_scenarios:
                # If the scenarios have equal positions then it is true.
                if compare_scenarios(scenario_choice, existent_scenario):
                    return True

            return False

        def select_scenario(list_scenarios):
            # priority to the scenarios with higher number: 10 has priority over 9, etc.
            higher_id = -1
            selected_scenario = None
            for scenario in list_scenarios:
                try:
                    scenario_number = int(scenario['name'].split('Scenario')[1])
                except:
                    scenario_number = -1

                if scenario_number >= higher_id:
                    higher_id = scenario_number
                    selected_scenario = scenario
            if not selected_scenario['name'] in self.route_scenario_dic:
                self.route_scenario_dic[selected_scenario['name']] = 1
                if selected_scenario['name'] == 'Scenario3':
                    return None
            else:
                self.route_scenario_dic[selected_scenario['name']] += 1
                if selected_scenario['name'] == 'Scenario3' and self.route_scenario_dic[selected_scenario['name']]!=5:
                    return None


            return selected_scenario

        def select_scenario_randomly(list_scenarios):
            # randomly select a scenario
            # if scenario3 in select list, select it with a probability, if not select randomly
            selected_scenario = None
            # for scenario in list_scenarios:
            #     if scenario['name'] == 'Scenario3':
            #         if rgn.random()>0.0:
            #             selected_scenario = rgn.choice(list_scenarios)
            #         selected_scenario = scenario
            selected_scenario = rgn.choice(list_scenarios)
            # if selected_scenario == None:
            #     selected_scenario = rgn.choice(list_scenarios)
            # record number of each type of scenario along this route
            if not selected_scenario['name'] in self.route_scenario_dic:
                self.route_scenario_dic[selected_scenario['name']] = 1
            else:
                self.route_scenario_dic[selected_scenario['name']] += 1

            # if selected_scenario['name'] != 'Scenario3':
            #     print(selected_scenario['name'])
            return selected_scenario

        # The idea is to randomly sample a scenario per trigger position.
        sampled_scenarios = []
        for trigger in potential_scenarios_definitions.keys():
            possible_scenarios = potential_scenarios_definitions[trigger]

            scenario_choice = select_scenario(possible_scenarios) # original prioritized sampling
            # scenario_choice = select_scenario_randomly(possible_scenarios) # random sampling
            if scenario_choice == None:
                continue
            print('load {} at (x={}, y={})'.format(scenario_choice['name'], scenario_choice['trigger_position']['x'], scenario_choice['trigger_position']['y']))
            del possible_scenarios[possible_scenarios.index(scenario_choice)]
            # Keep sampling and testing if this position is present on any of the scenarios.
            while position_sampled(scenario_choice, sampled_scenarios):
                if possible_scenarios is None or not possible_scenarios:
                    scenario_choice = None
                    break
                scenario_choice = rgn.choice(possible_scenarios)
                del possible_scenarios[possible_scenarios.index(scenario_choice)]

            if scenario_choice is not None:
                sampled_scenarios.append(scenario_choice)

        return sampled_scenarios

    def _build_scenario_instances(self, world, ego_vehicles, scenario_definitions,
                                  scenarios_per_tick=5, timeout=300, debug_mode=False) -> List:
        """
        Based on the parsed route and possible scenarios, build all the scenario classes.
        Args:
            world: Carla world
            ego_vehicles: list of Carla vehicle
            scenario_definitions: scenario_definitions[j] represents scenario to be triggered on the jth ego vehicle's route
        Returns:
            scenario_instance_vecs: scenario_instance_vecs[j] represents a list of scenario instance to meet with the jth ego vehicle
        """
        scenario_instance_vecs = []

        for j in range(len(scenario_definitions)):
            scenario_definition = scenario_definitions[j]
            scenario_instance_vec = []
            if debug_mode:
                for scenario in scenario_definition:
                    loc = carla.Location(scenario['trigger_position']['x'],
                                        scenario['trigger_position']['y'],
                                        scenario['trigger_position']['z']) + carla.Location(z=2.0)
                    world.debug.draw_point(loc, size=0.3, color=carla.Color(255, 0, 0), life_time=100000)
                    world.debug.draw_string(loc, str(scenario['name']), draw_shadow=False,
                                            color=carla.Color(0, 0, 255), life_time=100000, persistent_lines=True)

            for scenario_number, definition in enumerate(scenario_definition):
                # Get the class possibilities for this scenario number
                # TODO(gjh): USE REGISTRY TO DEFINE SCENARIOS
                scenario_class = NUMBER_CLASS_TRANSLATION[definition['name']]

                # Create the other actors that are going to appear
                if definition['other_actors'] is not None:
                    list_of_actor_conf_instances = self._get_actors_instances(definition['other_actors'])
                else:
                    list_of_actor_conf_instances = []
                # Create an actor configuration for the ego-vehicle trigger position

                egoactor_trigger_position = convert_json_to_transform(definition['trigger_position'])
                scenario_configuration = ScenarioConfiguration()
                scenario_configuration.other_actors = list_of_actor_conf_instances
                scenario_configuration.trigger_points = [egoactor_trigger_position]
                scenario_configuration.subtype = definition['scenario_type']
                scenario_configuration.ego_vehicle = ActorConfigurationData('vehicle.lincoln.mkz2017',
                                                                            ego_vehicles[j].get_transform(),
                                                                            'hero_{}'.format(j))
                route_var_name = "ScenarioRouteNumber{}".format(scenario_number)
                scenario_configuration.route_var_name = route_var_name
                try:
                    if definition['name']=='Scenario3':
                        scenario_instance = scenario_class(world, [ego_vehicles[j]], scenario_configuration,
                                                    criteria_enable=False, timeout=timeout, trigger_distance=self.trigger_distance)
                    else:
                        scenario_instance = scenario_class(world, [ego_vehicles[j]], scenario_configuration,
                                                    criteria_enable=False, timeout=timeout)
                    # Do a tick every once in a while to avoid spawning everything at the same time
                    if scenario_number % scenarios_per_tick == 0:
                        if CarlaDataProvider.is_sync_mode():
                            world.tick()
                        else:
                            world.wait_for_tick()
                except Exception as e:
                    print("Skipping scenario '{}' due to setup error: {}".format(definition['name'], e))
                    continue

                scenario_instance_vec.append(scenario_instance)
            scenario_instance_vecs.append(scenario_instance_vec)

        return scenario_instance_vecs

    def _build_scenario_parameter_instances(self, world:carla.libcarla.World, ego_vehicles: List, scenario_definitions: List,
                                  scenarios_per_tick:int=5, timeout:int=300, debug_mode:bool=False, scenario_parameter:dict=None)-> List:
        """
        Based on the parsed route and possible scenarios, build all the scenario classes.

        Args:
            world: carla world
            ego_vehicles: list of ego_vehicles
            scenario_definitions: list of scenario_definitions
            scenarios_per_tick: number of scenarios per tick
            timeout: number of timeout
            debug_mode: if open debug_mode
            scenario_parameter: a dict of predefined scenario_parameter in yaml file

        Returns:
            scenario_instance_vecs: list of scenario_instance_vecs
        """
        scenario_instance_vecs = []
        for j in range(len(scenario_definitions)):
            scenario_definition = scenario_definitions[j]
            scenario_instance_vec = []
            if debug_mode:
                for scenario in scenario_definition:
                    loc = carla.Location(scenario['trigger_position']['x'],
                                        scenario['trigger_position']['y'],
                                        scenario['trigger_position']['z']) + carla.Location(z=2.0)
                    world.debug.draw_point(loc, size=0.3, color=carla.Color(255, 0, 0), life_time=100000)
                    world.debug.draw_string(loc, str(scenario['name']), draw_shadow=False,
                                            color=carla.Color(0, 0, 255), life_time=100000, persistent_lines=True)

            for scenario_number, definition in enumerate(scenario_definition):
                # Get the class possibilities for this scenario number
                # NOTE(GJH): Use scenario_config.yaml to define scenarios
                scenario = scenario_parameter[definition['name']]
                scenario_class_name = selScenario(scenario)
                scenario_class = ScenarioClassRegistry[scenario_class_name]
                scenario_class_parameter = scenario[scenario_class_name]
                # Create the other actors that are going to appear
                if definition['other_actors'] is not None:
                    list_of_actor_conf_instances = self._get_actors_instances(definition['other_actors'])
                else:
                    list_of_actor_conf_instances = []
                # Create an actor configuration for the ego-vehicle trigger position

                egoactor_trigger_position = convert_json_to_transform(definition['trigger_position'])
                scenario_configuration = ScenarioConfiguration()
                scenario_configuration.other_actors = list_of_actor_conf_instances
                scenario_configuration.trigger_points = [egoactor_trigger_position]
                scenario_configuration.subtype = definition['scenario_type']
                scenario_configuration.ego_vehicle = ActorConfigurationData('vehicle.lincoln.mkz2017',
                                                                            ego_vehicles[j].get_transform(),
                                                                            'hero_{}'.format(j))
                route_var_name = "ScenarioRouteNumber{}".format(scenario_number)
                scenario_configuration.route_var_name = route_var_name
                scenario_instance = scenario_class(world, [ego_vehicles[j]], scenario_configuration,
                                                    criteria_enable=False, timeout=timeout, scenario_parameter=scenario_class_parameter)
                
                # Do a tick every once in a while to avoid spawning everything at the same time
                if scenario_number % scenarios_per_tick == 0:
                    if CarlaDataProvider.is_sync_mode():
                        world.tick()
                    else:
                        world.wait_for_tick()

                # except Exception as e:
                #     print("Skipping scenario '{}' due to setup error: {}".format(definition['name'], e))
                #     continue

                scenario_instance_vec.append(scenario_instance)
            scenario_instance_vecs.append(scenario_instance_vec)

        return scenario_instance_vecs


    def _get_actors_instances(self, list_of_antagonist_actors):
        """
        Get the full list of actor instances.
        """

        def get_actors_from_list(list_of_actor_def):
            """
                Receives a list of actor definitions and creates an actual list of ActorConfigurationObjects
            """
            sublist_of_actors = []
            for actor_def in list_of_actor_def:
                sublist_of_actors.append(convert_json_to_actor(actor_def))

            return sublist_of_actors

        list_of_actors = []
        # Parse vehicles to the left
        if 'front' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['front'])

        if 'left' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['left'])

        if 'right' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['right'])

        return list_of_actors

    # pylint: enable=no-self-use

    def _initialize_actors(self, config):
        """
        Set other_actors to the superset of all scenario actors
        """
        # Create the background activity of the route
        town_amount = {
            'Town01': 120,
            'Town02': 100,
            'Town03': 120,
            'Town04': 200,
            'Town05': 120, #120
            'Town06': 150,
            'Town07': 110,
            'Town08': 180,
            'Town09': 300,
            'Town10HD': 120, # town10 doesn't load properly for some reason
        }

        amount = town_amount[config.town] if config.town in town_amount else 0
        # amount_vehicle = amount
        amount_vehicle = self.background_params.get('vehicle_amount', int(amount)) 
        amount_pedestrian = self.background_params.get('pedestrian_amount', int(amount)) 
        self.crazy_level = self.background_params.get('CRAZY_LEVEL', self.crazy_level)
        self.crazy_proportion = self.background_params.get('CRAZY_PROPORTION', self.crazy_proportion)

        new_actors_vehicle = CarlaDataProvider.request_new_batch_actors('vehicle.*',
                                                                amount_vehicle,
                                                                carla.Transform(),
                                                                autopilot=True,
                                                                random_location=True,
                                                                rolename='background',
                                                                crazy_level = 0,
                                                                crazy_proportion = 0
                                                                )
        new_actors_pedestrian = CarlaDataProvider.request_new_batch_actors('walker.pedestrian.*',
                                                                amount_pedestrian,
                                                                carla.Transform(),
                                                                autopilot=True,
                                                                random_location=True,
                                                                rolename='background',
                                                                crazy_level = self.crazy_level,
                                                                crazy_proportion = self.crazy_proportion
                                                                )
        new_actors_bicycle = CarlaDataProvider.request_new_batch_actors('vehicle.diamondback.century',
                                                                amount_pedestrian,
                                                                carla.Transform(),
                                                                autopilot=True,
                                                                random_location=True,
                                                                rolename='background',
                                                                crazy_level = self.crazy_level,
                                                                crazy_proportion = self.crazy_proportion
                                                                )
                                                                
        # TODO: add other types of actors
        new_actors = new_actors_vehicle + new_actors_pedestrian + new_actors_bicycle

        if new_actors is None:
            raise Exception("Error: Unable to add the background activity, all spawn points were occupied")

        for _actor in new_actors:
            self.other_actors.append(_actor)

        self._spawn_custom_route_actors()

        # Add all the actors of the specific scenarios to self.other_actors
        for list_scenarios in self.list_scenarios:
            for scenario in list_scenarios:
                self.other_actors.extend(scenario.other_actors)

    def _spawn_custom_route_actors(self) -> None:
        """
        Spawn user-provided NPC actors that should follow predetermined paths.
        """
        if not self._custom_actor_configs:
            print("[RouteScenario] No custom actors defined for this route.")
            return

        actor_image_dir = os.environ.get("CUSTOM_ACTOR_IMAGE_DIR")
        normalize_actor_z = os.environ.get("CUSTOM_ACTOR_NORMALIZE_Z", "").lower() in (
            "1",
            "true",
            "yes",
        )
        follow_exact = os.environ.get("CUSTOM_ACTOR_FOLLOW_EXACT", "").lower() in (
            "1",
            "true",
            "yes",
        )
        log_replay = os.environ.get("CUSTOM_ACTOR_LOG_REPLAY", "").lower() in (
            "1",
            "true",
            "yes",
        )
        use_staging = os.environ.get("CUSTOM_LOG_REPLAY_STAGE", "").lower() in (
            "1",
            "true",
            "yes",
        )
        defer_log_replay_spawn = os.environ.get("CUSTOM_LOG_REPLAY_DEFER_SPAWN", "1").lower() in (
            "1",
            "true",
            "yes",
        )
        debug_spawn = os.environ.get("CUSTOM_ACTOR_SPAWN_DEBUG", "").lower() in (
            "1",
            "true",
            "yes",
        )
        dynamic_spawn_lift = log_replay and os.environ.get("CUSTOM_ACTOR_DYNAMIC_Z", "1").lower() in (
            "1",
            "true",
            "yes",
        )
        try:
            walker_ground_lift = float(os.environ.get("CUSTOM_WALKER_GROUND_LIFT", "0.06"))
        except Exception:  # pylint: disable=broad-except
            walker_ground_lift = 0.06
        animate_walkers = os.environ.get("CUSTOM_LOG_REPLAY_ANIMATE_WALKERS", "1").lower() in (
            "1",
            "true",
            "yes",
        )
        smooth_vehicle_replay = os.environ.get("CUSTOM_LOG_REPLAY_SMOOTH_VEHICLES", "1").lower() in (
            "1",
            "true",
            "yes",
        )
        try:
            smooth_min_cutoff_xy = float(os.environ.get("CUSTOM_LOG_REPLAY_SMOOTH_MIN_CUTOFF", "0.65"))
        except Exception:  # pylint: disable=broad-except
            smooth_min_cutoff_xy = 0.65
        try:
            smooth_beta_xy = float(os.environ.get("CUSTOM_LOG_REPLAY_SMOOTH_BETA", "0.18"))
        except Exception:  # pylint: disable=broad-except
            smooth_beta_xy = 0.18
        try:
            smooth_min_cutoff_yaw = float(os.environ.get("CUSTOM_LOG_REPLAY_SMOOTH_YAW_MIN_CUTOFF", "0.5"))
        except Exception:  # pylint: disable=broad-except
            smooth_min_cutoff_yaw = 0.5
        try:
            smooth_beta_yaw = float(os.environ.get("CUSTOM_LOG_REPLAY_SMOOTH_YAW_BETA", "0.12"))
        except Exception:  # pylint: disable=broad-except
            smooth_beta_yaw = 0.12
        try:
            smooth_min_cutoff_z = float(os.environ.get("CUSTOM_LOG_REPLAY_SMOOTH_Z_MIN_CUTOFF", "0.8"))
        except Exception:  # pylint: disable=broad-except
            smooth_min_cutoff_z = 0.8
        try:
            smooth_beta_z = float(os.environ.get("CUSTOM_LOG_REPLAY_SMOOTH_Z_BETA", "0.12"))
        except Exception:  # pylint: disable=broad-except
            smooth_beta_z = 0.12
        try:
            smooth_d_cutoff = float(os.environ.get("CUSTOM_LOG_REPLAY_SMOOTH_D_CUTOFF", "1.0"))
        except Exception:  # pylint: disable=broad-except
            smooth_d_cutoff = 1.0
        try:
            smooth_lateral_damping = float(
                os.environ.get("CUSTOM_LOG_REPLAY_LATERAL_DAMPING", "0.72")
            )
        except Exception:  # pylint: disable=broad-except
            smooth_lateral_damping = 0.72
        try:
            smooth_lateral_passes = int(
                os.environ.get("CUSTOM_LOG_REPLAY_LATERAL_PASSES", "2")
            )
        except Exception:  # pylint: disable=broad-except
            smooth_lateral_passes = 2
        try:
            smooth_lateral_max_correction = float(
                os.environ.get("CUSTOM_LOG_REPLAY_LATERAL_MAX_CORRECTION", "0.40")
            )
        except Exception:  # pylint: disable=broad-except
            smooth_lateral_max_correction = 0.40
        try:
            smooth_lateral_turn_keep = float(
                os.environ.get("CUSTOM_LOG_REPLAY_LATERAL_TURN_KEEP", "0.62")
            )
        except Exception:  # pylint: disable=broad-except
            smooth_lateral_turn_keep = 0.62
        try:
            smooth_lateral_turn_angle = float(
                os.environ.get("CUSTOM_LOG_REPLAY_LATERAL_TURN_ANGLE_DEG", "10.0")
            )
        except Exception:  # pylint: disable=broad-except
            smooth_lateral_turn_angle = 10.0
        stabilize_static_segments = os.environ.get(
            "CUSTOM_LOG_REPLAY_STABILIZE_STATIC_VEHICLES", "1"
        ).lower() in (
            "1",
            "true",
            "yes",
        )
        try:
            static_total_disp = float(
                os.environ.get("CUSTOM_LOG_REPLAY_STATIC_TOTAL_DISP", "0.35")
            )
        except Exception:  # pylint: disable=broad-except
            static_total_disp = 0.35
        try:
            static_total_path = float(
                os.environ.get("CUSTOM_LOG_REPLAY_STATIC_TOTAL_PATH", "1.6")
            )
        except Exception:  # pylint: disable=broad-except
            static_total_path = 1.6
        try:
            static_window_min_duration = float(
                os.environ.get("CUSTOM_LOG_REPLAY_STATIC_WINDOW_MIN_DURATION", "0.8")
            )
        except Exception:  # pylint: disable=broad-except
            static_window_min_duration = 0.8
        try:
            static_window_max_disp = float(
                os.environ.get("CUSTOM_LOG_REPLAY_STATIC_WINDOW_MAX_DISP", "0.22")
            )
        except Exception:  # pylint: disable=broad-except
            static_window_max_disp = 0.22
        try:
            static_window_max_speed = float(
                os.environ.get("CUSTOM_LOG_REPLAY_STATIC_WINDOW_MAX_SPEED", "0.08")
            )
        except Exception:  # pylint: disable=broad-except
            static_window_max_speed = 0.08
        try:
            static_window_max_yaw_delta = float(
                os.environ.get("CUSTOM_LOG_REPLAY_STATIC_WINDOW_MAX_YAW_DELTA", "2.5")
            )
        except Exception:  # pylint: disable=broad-except
            static_window_max_yaw_delta = 2.5
        lift_steps = _parse_lift_steps(
            os.environ.get("CUSTOM_ACTOR_SPAWN_LIFT_STEPS", "0.2,0.5,1.0")
        )
        if log_replay:
            follow_exact = True
            print(
                "[RouteScenario] Custom actors will replay logged transforms with timing "
                "(no road snapping or route planning)."
            )
        if follow_exact:
            print(
                "[RouteScenario] Custom actors will follow exact authored trajectories "
                "(no road snapping or route planning)."
            )
        if actor_image_dir:
            try:
                os.makedirs(actor_image_dir, exist_ok=True)
            except OSError:
                actor_image_dir = None

        world = CarlaDataProvider.get_world()
        world_map = CarlaDataProvider.get_map()
        planner = None
        if not follow_exact and world_map is not None:
            try:
                # Try CARLA 9.12+ API first (takes wmap, sampling_resolution directly)
                planner = GlobalRoutePlanner(world.get_map(), 1.0)
            except TypeError:
                # Fall back to older CARLA API (uses DAO pattern)
                try:
                    dao = GlobalRoutePlannerDAO(world_map, 1.0)
                    planner = GlobalRoutePlanner(dao)
                    planner.setup()
                except Exception:  # pylint: disable=broad-except
                    planner = None
            except Exception:  # pylint: disable=broad-except
                planner = None

        xy_retry_offsets = _parse_offset_values(
            os.environ.get("CUSTOM_ACTOR_SPAWN_XY_OFFSETS", "0.0,0.25,-0.25,0.5,-0.5,1.0,-1.0"),
            default_values=[0.0, 0.25, -0.25, 0.5, -0.5, 1.0, -1.0],
        )
        z_retry_offsets = _parse_offset_values(
            os.environ.get("CUSTOM_ACTOR_SPAWN_Z_OFFSETS", "0.0,0.2,-0.2,0.5,-0.5,1.0"),
            default_values=[0.0, 0.2, -0.2, 0.5, -0.5, 1.0],
        )
        try:
            max_spawn_attempts = max(0, int(os.environ.get("CUSTOM_ACTOR_SPAWN_MAX_ATTEMPTS", "180")))
        except Exception:  # pylint: disable=broad-except
            max_spawn_attempts = 180
        spawn_retry_offsets = _build_spawn_retry_offsets(
            xy_retry_offsets,
            z_retry_offsets,
            max_spawn_attempts,
        )

        total = len(self._custom_actor_configs)
        spawned = 0
        failed = 0
        deferred = 0
        self._custom_actor_spawn_summary_printed = False
        self._custom_actor_spawn_states = {}
        for actor_idx, actor_cfg in enumerate(self._custom_actor_configs):
            role = actor_cfg.get("role", "npc")
            name = actor_cfg.get("rolename") or actor_cfg.get("name") or "unknown"
            model = actor_cfg.get("model", "unknown")
            self._custom_actor_spawn_states[name] = {
                "name": name,
                "role": role,
                "model": model,
                "deferred": False,
                "spawned": False,
                "attempts": 0,
                "last_failure": None,
                "spawn_time": None,
                "spawn_offset": (0.0, 0.0, 0.0),
            }
            plan_transforms = actor_cfg.get("plan_transforms") or []
            plan_times = actor_cfg.get("plan_times")
            spawn_tf_src: carla.Transform = actor_cfg["spawn_transform"]
            spawn_tf_plan = carla.Transform(
                carla.Location(
                    x=spawn_tf_src.location.x,
                    y=spawn_tf_src.location.y,
                    z=spawn_tf_src.location.z,
                ),
                carla.Rotation(
                    pitch=spawn_tf_src.rotation.pitch,
                    yaw=spawn_tf_src.rotation.yaw,
                    roll=spawn_tf_src.rotation.roll,
                ),
            )
            spawn_tf = carla.Transform(
                carla.Location(
                    x=spawn_tf_plan.location.x,
                    y=spawn_tf_plan.location.y,
                    z=spawn_tf_plan.location.z,
                ),
                carla.Rotation(
                    pitch=spawn_tf_plan.rotation.pitch,
                    yaw=spawn_tf_plan.rotation.yaw,
                    roll=spawn_tf_plan.rotation.roll,
                ),
            )
            adjusted_plan_transforms = [
                carla.Transform(
                    carla.Location(x=tf.location.x, y=tf.location.y, z=tf.location.z),
                    carla.Rotation(
                        pitch=tf.rotation.pitch,
                        yaw=tf.rotation.yaw,
                        roll=tf.rotation.roll,
                    ),
                )
                for tf in plan_transforms
            ]
            is_non_vehicle_role = role in ("pedestrian", "walker", "bicycle", "cyclist")
            is_vehicle_model = str(model).lower().startswith("vehicle.")
            if (
                log_replay
                and smooth_vehicle_replay
                and adjusted_plan_transforms
                and not is_non_vehicle_role
                and is_vehicle_model
            ):
                adjusted_plan_transforms = _smooth_vehicle_replay_plan(
                    adjusted_plan_transforms,
                    plan_times,
                    min_cutoff_xy=smooth_min_cutoff_xy,
                    beta_xy=smooth_beta_xy,
                    min_cutoff_z=smooth_min_cutoff_z,
                    beta_z=smooth_beta_z,
                    min_cutoff_yaw=smooth_min_cutoff_yaw,
                    beta_yaw=smooth_beta_yaw,
                    d_cutoff=smooth_d_cutoff,
                    static_hold_enabled=stabilize_static_segments,
                    static_total_displacement=static_total_disp,
                    static_total_path_length=static_total_path,
                    static_window_min_duration=static_window_min_duration,
                    static_window_max_displacement=static_window_max_disp,
                    static_window_max_speed=static_window_max_speed,
                    static_window_max_yaw_delta=static_window_max_yaw_delta,
                    lateral_damping=smooth_lateral_damping,
                    lateral_passes=smooth_lateral_passes,
                    lateral_max_correction=smooth_lateral_max_correction,
                    lateral_turn_keep=smooth_lateral_turn_keep,
                    lateral_turn_angle_deg=smooth_lateral_turn_angle,
                )
            stage_replay = bool(
                log_replay
                and use_staging
                and plan_times
                and adjusted_plan_transforms
                and len(plan_times) == len(adjusted_plan_transforms)
            )
            stage_tf = _make_stage_transform(actor_idx, spawn_tf_plan) if stage_replay else None

            snapped_to_road = False
            ground_z_for_actor: Optional[float] = None
            if normalize_actor_z:
                # Adjust only Z to the nearest ground height; preserve authored x/y and rotation.
                candidates: list[float] = []
                ground_z = _resolve_ground_z(world, spawn_tf.location)
                if ground_z is not None:
                    candidates.append(ground_z)
                if world_map is not None:
                    snapped_wp = world_map.get_waypoint(
                        spawn_tf.location,
                        project_to_road=True,
                        lane_type=carla.LaneType.Any,
                    )
                    if snapped_wp is not None:
                        candidates.append(float(snapped_wp.transform.location.z))
                if candidates:
                    orig_z = float(spawn_tf.location.z)
                    best_z = min(candidates, key=lambda z: abs(z - orig_z))
                    spawn_tf.location.z = best_z
                    ground_z_for_actor = best_z
                    snapped_to_road = True

            snap_pref = actor_cfg.get("snap_to_road", True)
            if follow_exact:
                snap_pref = False
            should_snap_to_road = (
                snap_pref
                and role not in ("static", "static_prop", "pedestrian", "walker", "bicycle", "cyclist")
            )  # keep static props and pedestrians at their authored pose
            if world_map is not None and should_snap_to_road:
                snapped_wp = world_map.get_waypoint(
                    spawn_tf.location,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
                if snapped_wp is not None:
                    spawn_tf = snapped_wp.transform
                    snapped_to_road = True

            # Add z-offset to prevent ground clipping (vehicles need this even when static)
            model_str = str(actor_cfg.get("model", "")).lower()
            is_vehicle_model = model_str.startswith("vehicle.")
            if not snapped_to_road and not follow_exact:
                if role not in ("static", "static_prop"):
                    spawn_tf.location.z += 0.5
                elif is_vehicle_model and not normalize_actor_z:
                    # Smaller offset for static/parked vehicles to avoid visible floating
                    spawn_tf.location.z += 0.1

            if log_replay and adjusted_plan_transforms and not normalize_actor_z:
                try:
                    z_offset = float(spawn_tf_plan.location.z) - float(adjusted_plan_transforms[0].location.z)
                    if abs(z_offset) > 1e-4:
                        for tf in adjusted_plan_transforms:
                            tf.location.z = float(tf.location.z) + z_offset
                except Exception:  # pylint: disable=broad-except
                    pass

            defer_spawn = False
            spawn_time = None
            if (
                log_replay
                and defer_log_replay_spawn
                and not use_staging
                and plan_times
                and adjusted_plan_transforms
                and len(plan_times) == len(adjusted_plan_transforms)
            ):
                try:
                    spawn_time = float(plan_times[0])
                except Exception:  # pylint: disable=broad-except
                    spawn_time = None
                if spawn_time is not None and spawn_time > 0.0:
                    defer_spawn = True

            rolename = name
            is_walker_like = role in ("pedestrian", "walker", "bicycle", "cyclist")
            if defer_spawn:
                deferred += 1
                state = self._custom_actor_spawn_states.get(rolename)
                if state is not None:
                    state["deferred"] = True
                    state["spawn_time"] = float(spawn_time) if spawn_time is not None else None
                if spawn_time is not None:
                    print(
                        f"[RouteScenario] Log replay actor {rolename} deferred spawn until t={spawn_time:.3f}s"
                    )
                self._custom_actor_plans.append(
                    {
                        "actor": None,
                        "name": actor_cfg["name"],
                        "model": actor_cfg.get("model"),
                        "plan": [],
                        "plan_transforms": adjusted_plan_transforms,
                        "plan_times": plan_times,
                        "target_speed": actor_cfg["target_speed"],
                        "avoid_collision": actor_cfg.get("avoid_collision", False),
                        "behavior": self._custom_actor_behavior_by_name.get(actor_cfg["name"]),
                        "role": actor_cfg.get("role", "npc"),
                        "stage_transform": None,
                        "stage_before": False,
                        "stage_after": False,
                        "realized_path": [],
                        "spawn_model": actor_cfg.get("model"),
                        "spawn_transform": carla.Transform(
                            carla.Location(
                                x=spawn_tf.location.x,
                                y=spawn_tf.location.y,
                                z=spawn_tf.location.z,
                            ),
                            carla.Rotation(
                                pitch=spawn_tf.rotation.pitch,
                                yaw=spawn_tf.rotation.yaw,
                                roll=spawn_tf.rotation.roll,
                            ),
                        ),
                        "spawn_time": spawn_time,
                        "defer_spawn": True,
                        "normalize_actor_z": normalize_actor_z,
                        "ground_prefer_ray": is_walker_like,
                        "ground_z_extra": float(walker_ground_lift) if is_walker_like else 0.0,
                        "animate_walkers": animate_walkers,
                        "snap_to_road": should_snap_to_road,
                        "follow_exact": follow_exact,
                        "spawn_retry_offsets": spawn_retry_offsets,
                        "spawn_debug": {
                            "snapped_to_road": snapped_to_road,
                            "ground_z": ground_z_for_actor,
                            "stage_replay": stage_replay,
                        },
                    }
                )
                continue
            spawn_candidates = [spawn_tf]
            new_actor = None
            last_exc = None
            lift_used = None
            offset_used = None
            attempts_for_actor = 0
            lane_type_for_spawn = carla.LaneType.Driving
            if role in ("pedestrian", "walker", "bicycle", "cyclist"):
                lane_type_for_spawn = carla.LaneType.Any
            for candidate in spawn_candidates:
                attempts_for_actor += 1
                try:
                    new_actor = CarlaDataProvider.request_new_actor(
                        actor_cfg["model"],
                        candidate,
                        rolename=rolename,
                        autopilot=False,
                    )
                    if new_actor is not None:
                        spawn_tf = candidate
                        break
                except Exception as exc:  # pylint: disable=broad-except
                    last_exc = exc
                    new_actor = None
            if new_actor is None and spawn_retry_offsets:
                (
                    retry_actor,
                    retry_tf,
                    retry_offset,
                    retry_exc,
                    retry_attempts,
                ) = _spawn_with_offsets(
                    actor_cfg["model"],
                    rolename,
                    spawn_tf,
                    spawn_retry_offsets,
                    world,
                    world_map,
                    lane_type_for_spawn,
                    normalize_actor_z=normalize_actor_z,
                    autopilot=False,
                )
                attempts_for_actor += retry_attempts
                if retry_actor is not None and retry_tf is not None:
                    new_actor = retry_actor
                    spawn_tf = retry_tf
                    offset_used = retry_offset
                elif retry_exc is not None:
                    last_exc = retry_exc
            if new_actor is None and dynamic_spawn_lift and lift_steps:
                retry_actor, retry_lift, retry_exc = _spawn_with_lifts(
                    actor_cfg["model"],
                    rolename,
                    spawn_tf,
                    lift_steps,
                    autopilot=False,
                )
                attempts_for_actor += len(lift_steps)
                if retry_actor is not None:
                    new_actor = retry_actor
                    lift_used = retry_lift
                elif retry_exc is not None:
                    last_exc = retry_exc
            state = self._custom_actor_spawn_states.get(rolename)
            if state is not None:
                state["attempts"] = int(state.get("attempts", 0)) + int(attempts_for_actor)
                if offset_used is not None:
                    state["spawn_offset"] = offset_used
            if new_actor is None:
                failed += 1
                if state is not None:
                    state["last_failure"] = str(last_exc) if last_exc is not None else "request_new_actor returned None"
                msg = (
                    f"[RouteScenario] Failed to spawn custom actor name={rolename} role={role} model={model}"
                )
                if last_exc is not None:
                    msg = f"{msg}: {last_exc}"
                if attempts_for_actor > 0:
                    msg = f"{msg} (attempts={attempts_for_actor})"
                print(msg)
                _emit_spawn_debug(
                    "request_new_actor_failed",
                    actor_cfg,
                    spawn_tf,
                    snapped_to_road,
                    ground_z_for_actor,
                    stage_replay,
                    world,
                    world_map,
                    normalize_actor_z,
                    follow_exact,
                    log_replay,
                    debug_spawn,
                )
                continue

            if state is not None:
                state["spawned"] = True
                state["last_failure"] = None

            if lift_used is not None:
                if normalize_actor_z:
                    lane_type = carla.LaneType.Driving
                    if role in ("pedestrian", "walker", "bicycle", "cyclist"):
                        lane_type = carla.LaneType.Any
                    grounded_tf = _ground_actor_transform(
                        new_actor,
                        spawn_tf,
                        world_map,
                        world,
                        lane_type,
                    )
                    if grounded_tf is not None:
                        try:
                            new_actor.set_transform(grounded_tf)
                            spawn_tf = grounded_tf
                        except Exception:  # pylint: disable=broad-except
                            pass
                else:
                    try:
                        new_actor.set_transform(spawn_tf)
                    except Exception:  # pylint: disable=broad-except
                        pass
            elif offset_used is not None:
                print(
                    "[RouteScenario] Spawn retry succeeded name={} offset=({:+.2f},{:+.2f},{:+.2f})".format(
                        rolename,
                        float(offset_used[0]),
                        float(offset_used[1]),
                        float(offset_used[2]),
                    )
                )

            if normalize_actor_z and ground_z_for_actor is not None and role in ("static", "static_prop"):
                try:
                    bbox = new_actor.bounding_box
                    target_z = float(ground_z_for_actor) - float(bbox.location.z) + float(bbox.extent.z)
                    tf = new_actor.get_transform()
                    tf.location.z = target_z
                    new_actor.set_transform(tf)
                except Exception:  # pylint: disable=broad-except
                    pass

            if log_replay and normalize_actor_z and adjusted_plan_transforms:
                lane_type = carla.LaneType.Driving
                if role in ("pedestrian", "walker", "bicycle", "cyclist"):
                    lane_type = carla.LaneType.Any
                _glue_plan_to_ground(
                    adjusted_plan_transforms,
                    new_actor,
                    world_map,
                    lane_type,
                    world,
                    prefer_ray_ground=is_walker_like,
                    z_extra=float(walker_ground_lift) if is_walker_like else 0.0,
                )
                if not (stage_replay and plan_times and plan_times[0] > 0.0):
                    try:
                        new_actor.set_transform(adjusted_plan_transforms[0])
                    except Exception:  # pylint: disable=broad-except
                        pass

            spawned += 1
            print(
                f"[RouteScenario] Spawned custom actor name={rolename} role={role} model={model} at {spawn_tf}"
            )
            if stage_replay and stage_tf is not None and plan_times:
                try:
                    t0 = float(plan_times[0])
                except Exception:  # pylint: disable=broad-except
                    t0 = None
                if t0 is not None and t0 > 0.0:
                    if not (is_walker_like and animate_walkers):
                        try:
                            new_actor.set_simulate_physics(False)
                        except Exception:  # pylint: disable=broad-except
                            pass
                    try:
                        new_actor.set_transform(stage_tf)
                    except Exception:  # pylint: disable=broad-except
                        pass
                    print(
                        f"[RouteScenario] Log replay actor {rolename} staged until t={t0:.3f}s"
                    )
            self.other_actors.append(new_actor)

            if actor_image_dir and not log_replay:
                self._capture_custom_actor_image(new_actor, rolename, role, actor_image_dir)

            if role in ("static", "static_prop"):
                try:
                    new_actor.set_simulate_physics(False)
                except Exception:  # pylint: disable=broad-except
                    pass
                continue

            # For pedestrians and cyclists, use the authored plan directly without road snapping
            is_non_vehicle = role in ("pedestrian", "walker", "bicycle", "cyclist")
            snap_plan = should_snap_to_road and not is_non_vehicle

            plan_locations = []
            for loc in actor_cfg["plan"]:
                if not snap_plan:
                    # Keep original waypoint for pedestrians/cyclists
                    plan_locations.append(carla.Location(x=loc.x, y=loc.y, z=loc.z))
                else:
                    snapped_loc = carla.Location(x=loc.x, y=loc.y, z=loc.z)
                    if world_map is not None:
                        wp = world_map.get_waypoint(
                            loc,
                            project_to_road=True,
                            lane_type=carla.LaneType.Driving,
                        )
                        if wp is not None:
                            snapped_loc = wp.transform.location
                    plan_locations.append(snapped_loc)

            dense_plan = []
            if snap_plan and planner is not None and len(plan_locations) >= 2:
                route_plan = []
                for idx in range(len(plan_locations) - 1):
                    start_loc = plan_locations[idx]
                    end_loc = plan_locations[idx + 1]
                    segment = planner.trace_route(start_loc, end_loc)
                    if segment:
                        route_plan.extend(segment)
                if route_plan:
                    if world_map is not None:
                        spawn_wp = world_map.get_waypoint(
                            spawn_tf.location,
                            project_to_road=True,
                            lane_type=carla.LaneType.Driving,
                        )
                        if spawn_wp is not None:
                            first_option = route_plan[0][1] if isinstance(route_plan[0], tuple) and len(route_plan[0]) > 1 else RoadOption.LANEFOLLOW
                            route_plan[0] = (spawn_wp, first_option)
                    dense_plan = route_plan
            if not dense_plan:
                if plan_locations and snap_plan:
                    plan_locations[0] = carla.Location(
                        x=spawn_tf.location.x,
                        y=spawn_tf.location.y,
                        z=spawn_tf.location.z,
                    )
                dense_plan = plan_locations

            self._custom_actor_plans.append(
                {
                    "actor": new_actor,
                    "name": actor_cfg["name"],
                    "plan": dense_plan,
                    "plan_transforms": adjusted_plan_transforms,
                    "plan_times": plan_times,
                    "target_speed": actor_cfg["target_speed"],
                    "avoid_collision": actor_cfg.get("avoid_collision", False),
                    "behavior": self._custom_actor_behavior_by_name.get(actor_cfg["name"]),
                    "role": actor_cfg.get("role", "npc"),
                    "stage_transform": stage_tf,
                    "stage_before": stage_replay,
                    "stage_after": stage_replay,
                    "ground_prefer_ray": is_walker_like,
                    "ground_z_extra": float(walker_ground_lift) if is_walker_like else 0.0,
                    "animate_walkers": animate_walkers,
                    "realized_path": [],
                }
            )

        if deferred:
            print(
                f"[RouteScenario] Custom actor initial spawn summary: total={total} spawned={spawned} "
                f"deferred={deferred} failed={failed}"
            )
        else:
            print(
                f"[RouteScenario] Custom actor initial spawn summary: total={total} spawned={spawned} failed={failed}"
            )
        if actor_image_dir:
            self._capture_overhead_scene_image(actor_image_dir)

    def _build_custom_actor_spawn_summary(self) -> dict:
        states = self._custom_actor_spawn_states or {}
        total = len(states)
        immediate_states = [s for s in states.values() if not bool(s.get("deferred"))]
        deferred_states = [s for s in states.values() if bool(s.get("deferred"))]

        immediate_spawned = sum(1 for s in immediate_states if bool(s.get("spawned")))
        immediate_failed = sum(1 for s in immediate_states if not bool(s.get("spawned")))
        deferred_spawned = sum(1 for s in deferred_states if bool(s.get("spawned")))
        deferred_attempted = sum(1 for s in deferred_states if int(s.get("attempts", 0)) > 0)
        deferred_failed = sum(
            1
            for s in deferred_states
            if int(s.get("attempts", 0)) > 0 and not bool(s.get("spawned"))
        )
        deferred_pending = sum(
            1
            for s in deferred_states
            if int(s.get("attempts", 0)) == 0 and not bool(s.get("spawned"))
        )

        failed_details = []
        pending_details = []
        for state in states.values():
            if bool(state.get("spawned")):
                continue
            record = {
                "name": state.get("name"),
                "role": state.get("role"),
                "model": state.get("model"),
                "deferred": bool(state.get("deferred")),
                "attempts": int(state.get("attempts", 0)),
                "spawn_time": state.get("spawn_time"),
                "last_failure": state.get("last_failure"),
            }
            if bool(state.get("deferred")) and int(state.get("attempts", 0)) == 0:
                pending_details.append(record)
            else:
                failed_details.append(record)

        return {
            "total": total,
            "attempts_total": sum(int(s.get("attempts", 0)) for s in states.values()),
            "immediate_total": len(immediate_states),
            "immediate_spawned": immediate_spawned,
            "immediate_failed": immediate_failed,
            "deferred_total": len(deferred_states),
            "deferred_spawned": deferred_spawned,
            "deferred_attempted": deferred_attempted,
            "deferred_failed": deferred_failed,
            "deferred_pending": deferred_pending,
            "failed_total": immediate_failed + deferred_failed,
            "failed_details": failed_details,
            "pending_details": pending_details,
        }

    def _print_custom_actor_spawn_summary(self) -> None:
        if self._custom_actor_spawn_summary_printed:
            return
        if not self._custom_actor_spawn_states:
            return

        summary = self._build_custom_actor_spawn_summary()
        print(
            "[RouteScenario] Custom actor final spawn summary: "
            f"total={summary['total']} attempts={summary['attempts_total']} "
            f"immediate_ok={summary['immediate_spawned']}/{summary['immediate_total']} "
            f"immediate_fail={summary['immediate_failed']} "
            f"deferred_ok={summary['deferred_spawned']}/{summary['deferred_total']} "
            f"deferred_fail={summary['deferred_failed']} "
            f"deferred_pending={summary['deferred_pending']} "
            f"failed_total={summary['failed_total']}"
        )
        if summary["failed_details"]:
            preview = []
            for item in summary["failed_details"][:15]:
                preview.append(
                    "{}(role={}, model={}, attempts={}, reason={})".format(
                        item.get("name"),
                        item.get("role"),
                        item.get("model"),
                        item.get("attempts"),
                        item.get("last_failure"),
                    )
                )
            print("[RouteScenario] Failed spawn actors: " + "; ".join(preview))
        if summary["pending_details"]:
            preview = []
            for item in summary["pending_details"][:15]:
                preview.append(
                    "{}(role={}, model={}, spawn_time={})".format(
                        item.get("name"),
                        item.get("role"),
                        item.get("model"),
                        item.get("spawn_time"),
                    )
                )
            print("[RouteScenario] Deferred actors not yet attempted: " + "; ".join(preview))
        self._custom_actor_spawn_summary_printed = True

    def _capture_custom_actor_image(self, actor, name: str, role: str, output_dir: str) -> None:
        """
        Capture a single RGB image of a spawned custom actor.
        """
        if actor is None:
            return
        try:
            world = actor.get_world()
            if world is None:
                return
            blueprint_library = world.get_blueprint_library()
            cam_bp = blueprint_library.find("sensor.camera.rgb")
            if cam_bp is None:
                print(f"[RouteScenario] Actor image capture skipped (no camera blueprint) for {name}")
                return
            cam_bp.set_attribute("image_size_x", "800")
            cam_bp.set_attribute("image_size_y", "600")
            cam_bp.set_attribute("fov", "90")

            # Set a reasonable camera offset depending on actor role.
            role_lower = str(role or "").lower()
            if role_lower in ("walker", "pedestrian", "cyclist", "bicycle"):
                rel_loc = carla.Location(x=-3.0, y=0.0, z=1.8)
                rel_rot = carla.Rotation(pitch=-10.0)
            else:
                rel_loc = carla.Location(x=-6.0, y=0.0, z=3.0)
                rel_rot = carla.Rotation(pitch=-10.0)
            cam_tf = carla.Transform(rel_loc, rel_rot)

            camera = world.spawn_actor(
                cam_bp,
                cam_tf,
                attach_to=actor,
                attachment_type=carla.AttachmentType.Rigid,
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[RouteScenario] Actor image capture failed to spawn camera for {name}: {exc}")
            return

        image_queue = queue.Queue()
        try:
            camera.listen(lambda image: image_queue.put(image))
            settings = world.get_settings()
            if getattr(settings, "synchronous_mode", False):
                world.tick()
            else:
                world.wait_for_tick()
            image = image_queue.get(timeout=2.0)
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))
            try:
                capture_t = float(GameTime.get_time())
            except Exception:  # pylint: disable=broad-except
                capture_t = None
            if capture_t is None:
                filename = f"{role}_{safe_name}_{actor.id}.png"
            else:
                filename = f"{role}_{safe_name}_{actor.id}_t{capture_t:.2f}.png"
            image.save_to_disk(os.path.join(output_dir, filename))
            print(f"[RouteScenario] Saved actor image: {filename}")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[RouteScenario] Actor image capture failed for {name}: {exc}")
        finally:
            try:
                camera.stop()
            except Exception:  # pylint: disable=broad-except
                pass
            try:
                camera.destroy()
            except Exception:  # pylint: disable=broad-except
                pass

    def _capture_custom_actor_image_stage(
        self,
        actor,
        name: str,
        role: str,
        output_dir: str,
        stage: str,
    ) -> None:
        """
        Capture an actor image into a stage subfolder (spawn/mid/post).
        """
        if not output_dir:
            return
        stage_dir = os.path.join(output_dir, str(stage))
        try:
            os.makedirs(stage_dir, exist_ok=True)
        except OSError:
            return
        self._capture_custom_actor_image(actor, name, role, stage_dir)

    def _save_custom_actor_path_visualization(self, actor_plan: dict, output_dir: str) -> None:
        """
        Save a visualization of the ground-truth route and realized path for a custom actor.
        """
        if not output_dir or not actor_plan:
            return
        plan_transforms = actor_plan.get("plan_transforms") or []
        plan_times = actor_plan.get("plan_times") or []
        realized = actor_plan.get("realized_path") or []
        if not plan_transforms or not plan_times or len(plan_transforms) != len(plan_times):
            return

        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError:
            return

        out_dir = os.path.join(output_dir, "paths")
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError:
            return

        gt_x = [tf.location.x for tf in plan_transforms]
        gt_y = [tf.location.y for tf in plan_transforms]
        gt_t = [float(t) for t in plan_times]

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(gt_x, gt_y, color="black", linewidth=1.0, alpha=0.6, label="GT path")
        sc = ax.scatter(gt_x, gt_y, c=gt_t, cmap="viridis", s=10, label="GT time")
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Time (s)")

        # Annotate spawn/despawn times
        ax.scatter([gt_x[0]], [gt_y[0]], color="green", s=40, label="Spawn")
        ax.scatter([gt_x[-1]], [gt_y[-1]], color="red", s=40, label="Despawn")
        ax.annotate(f"t={gt_t[0]:.2f}", (gt_x[0], gt_y[0]), textcoords="offset points", xytext=(6, 6))
        ax.annotate(f"t={gt_t[-1]:.2f}", (gt_x[-1], gt_y[-1]), textcoords="offset points", xytext=(6, 6))

        if realized:
            rx = [tf.location.x for _, tf in realized]
            ry = [tf.location.y for _, tf in realized]
            ax.plot(rx, ry, color="tab:red", linewidth=1.0, alpha=0.8, label="Realized path")

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Actor Path: {actor_plan.get('name', 'unknown')}")
        ax.legend(loc="best")

        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(actor_plan.get("name", "actor")))
        filename = f"path_{safe_name}.png"
        try:
            fig.savefig(os.path.join(out_dir, filename), dpi=150, bbox_inches="tight")
        except Exception:  # pylint: disable=broad-except
            pass
        finally:
            plt.close(fig)

    def _capture_overhead_scene_image(self, output_dir: str) -> None:
        """
        Capture a top-down image that bounds all custom actors and ego vehicles.
        """
        world = CarlaDataProvider.get_world()
        if world is None:
            return

        actors = []
        actors.extend(self.other_actors or [])
        actors.extend(self.ego_vehicles or [])
        actors = [a for a in actors if a is not None]
        if not actors:
            return

        locations = []
        for actor in actors:
            try:
                loc = actor.get_location()
            except Exception:  # pylint: disable=broad-except
                continue
            if loc is not None:
                locations.append(loc)
        if not locations:
            return

        min_x = min(loc.x for loc in locations)
        max_x = max(loc.x for loc in locations)
        min_y = min(loc.y for loc in locations)
        max_y = max(loc.y for loc in locations)
        max_z = max(loc.z for loc in locations)

        center_x = 0.5 * (min_x + max_x)
        center_y = 0.5 * (min_y + max_y)
        extent = max(max_x - min_x, max_y - min_y)
        extent *= 1.2  # margin
        extent = max(extent, 20.0)

        fov = 90.0
        height = max(20.0, 0.5 * extent / max(1e-3, math.tan(math.radians(fov * 0.5))))

        blueprint_library = world.get_blueprint_library()
        cam_bp = blueprint_library.find("sensor.camera.rgb")
        if cam_bp is None:
            print("[RouteScenario] Overhead capture skipped (no camera blueprint).")
            return
        cam_bp.set_attribute("image_size_x", "1024")
        cam_bp.set_attribute("image_size_y", "1024")
        cam_bp.set_attribute("fov", f"{fov}")

        cam_tf = carla.Transform(
            carla.Location(x=center_x, y=center_y, z=max_z + height),
            carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
        )

        try:
            camera = world.spawn_actor(cam_bp, cam_tf)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[RouteScenario] Overhead capture failed to spawn camera: {exc}")
            return

        image_queue = queue.Queue()
        try:
            camera.listen(lambda image: image_queue.put(image))
            settings = world.get_settings()
            if getattr(settings, "synchronous_mode", False):
                world.tick()
            else:
                world.wait_for_tick()
            image = image_queue.get(timeout=2.0)
            filename = "overhead_all_actors.png"
            image.save_to_disk(os.path.join(output_dir, filename))
            print(f"[RouteScenario] Saved overhead image: {filename}")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[RouteScenario] Overhead capture failed: {exc}")
        finally:
            try:
                camera.stop()
            except Exception:  # pylint: disable=broad-except
                pass
            try:
                camera.destroy()
            except Exception:  # pylint: disable=broad-except
                pass

    def remove_all_actors(self):
        # Emit a final spawn summary before actor cleanup so end-of-run logs include totals.
        self._print_custom_actor_spawn_summary()
        super().remove_all_actors()

    def _resolve_ego_vehicle(self, vehicle_name: str):
        if not vehicle_name:
            return None
        name = str(vehicle_name).strip()
        if name.lower() in ("ego", "ego_vehicle", "ego vehicle"):
            return self.ego_vehicles[0] if self.ego_vehicles else None
        match = re.search(r"(\d+)", name)
        if not match:
            return None
        idx = int(match.group(1)) - 1
        if idx < 0 or idx >= len(self.ego_vehicles):
            return None
        return self.ego_vehicles[idx]

    def _build_trigger_condition(self, trigger_spec: dict, actor):
        if not isinstance(trigger_spec, dict):
            return None
        ttype = str(trigger_spec.get("type", "")).strip()
        if ttype != "distance_to_vehicle":
            return None
        ego_name = trigger_spec.get("vehicle")
        ego_actor = self._resolve_ego_vehicle(ego_name)
        if ego_actor is None:
            return None
        try:
            dist = float(trigger_spec.get("distance_m", 8.0))
        except Exception:
            dist = 8.0
        dist = max(1.0, min(50.0, dist))
        return InTriggerDistanceToVehicle(ego_actor, actor, dist)

    def _infer_lane_change_direction(self, actor, target_vehicle):
        world_map = CarlaDataProvider.get_map()
        if world_map is None:
            return None
        actor_wp = world_map.get_waypoint(
            actor.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
        )
        target_wp = world_map.get_waypoint(
            target_vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if actor_wp is None or target_wp is None:
            return None
        left_wp = actor_wp.get_left_lane()
        if left_wp is not None and left_wp.lane_id == target_wp.lane_id:
            return "left"
        right_wp = actor_wp.get_right_lane()
        if right_wp is not None and right_wp.lane_id == target_wp.lane_id:
            return "right"
        return None

    def _is_vehicle_actor(self, actor_plan: dict) -> bool:
        """Check if this actor is a vehicle (not pedestrian/walker)."""
        role = actor_plan.get("role", "npc")
        name = actor_plan.get("name", "").lower()
        # Check if it's a vehicle based on name or role
        if "walker" in name or "pedestrian" in name:
            return False
        return True

    def _build_custom_actor_behavior(self, actor_plan: dict):
        behavior_spec = actor_plan.get("behavior")
        if not isinstance(behavior_spec, dict):
            if os.environ.get('DEBUG_FINISH', '').lower() in ('1', 'true', 'yes'):
                print(f"[DEBUG FINISH] _build_custom_actor_behavior: No behavior_spec for {actor_plan.get('name')}")
            return None
        trigger_spec = behavior_spec.get("trigger")
        action_spec = behavior_spec.get("action") or behavior_spec.get("behavior")
        if os.environ.get('DEBUG_FINISH', '').lower() in ('1', 'true', 'yes'):
            print(f"[DEBUG FINISH] _build_custom_actor_behavior: trigger_spec={trigger_spec}, action_spec={action_spec}")

        actor = actor_plan.get("actor")
        if actor is None:
            if os.environ.get('DEBUG_FINISH', '').lower() in ('1', 'true', 'yes'):
                print(f"[DEBUG FINISH] _build_custom_actor_behavior: No actor")
            return None

        trigger_cond = None
        ego_actor = None
        trigger_distance = None
        if isinstance(trigger_spec, dict):
            trigger_cond = self._build_trigger_condition(trigger_spec, actor)
            if os.environ.get('DEBUG_FINISH', '').lower() in ('1', 'true', 'yes'):
                print(f"[DEBUG FINISH] _build_custom_actor_behavior: trigger_cond={trigger_cond}")
            # Get ego actor and trigger distance for smart speed calculation
            if trigger_spec.get("type") == "distance_to_vehicle":
                ego_actor = self._resolve_ego_vehicle(trigger_spec.get("vehicle"))
                try:
                    trigger_distance = float(trigger_spec.get("distance_m", 8.0))
                except Exception:
                    trigger_distance = 8.0
                trigger_distance = max(1.0, min(50.0, trigger_distance))

        if not isinstance(action_spec, dict):
            action_spec = {}

        action_type = str(action_spec.get("type", "")).strip()
        if not action_type:
            # Default action: start motion when trigger is present.
            action_type = "start_motion"

        target_speed = actor_plan.get("target_speed")
        plan = actor_plan.get("plan")
        avoid_collision = actor_plan.get("avoid_collision", False)
        
        # NPC distance-trigger pacing:
        # - Compute a base speed from initial spawn distance.
        # - While waiting for the trigger, pace off ego speed so the NPC never outruns.
        
        is_vehicle = self._is_vehicle_actor(actor_plan)
        has_distance_trigger = (
            isinstance(trigger_spec, dict) and 
            trigger_spec.get("type") == "distance_to_vehicle"
        )
        needs_catchup = is_vehicle and has_distance_trigger and action_type != "start_motion"
        
        effective_speed = target_speed if target_speed is not None else 8.0
        if needs_catchup and ego_actor is not None and actor is not None:
            try:
                ego_loc = CarlaDataProvider.get_location(ego_actor) or ego_actor.get_location()
                npc_loc = CarlaDataProvider.get_location(actor) or actor.get_location()
                spawn_distance = ego_loc.distance(npc_loc)
            except Exception:
                spawn_distance = 30.0

            close_distance = max(12.0, float(trigger_distance or 8.0) + 4.0)
            far_distance = max(60.0, close_distance + 30.0)
            max_speed = max(0.0, float(effective_speed or 8.0))
            min_speed = min(2.0, max_speed)

            if spawn_distance >= far_distance:
                effective_speed = min_speed
            elif spawn_distance <= close_distance:
                effective_speed = max_speed
            else:
                t = (spawn_distance - close_distance) / (far_distance - close_distance)
                effective_speed = max_speed - t * (max_speed - min_speed)

        speed_callback = None
        _debug_counter = [0]  # Use list to allow mutation in closure
        if needs_catchup and ego_actor is not None and trigger_distance is not None:
            def speed_callback(_actor):
                _debug_counter[0] += 1
                try:
                    ego_speed = CarlaDataProvider.get_velocity(ego_actor)
                except Exception:
                    ego_speed = None
                if ego_speed is None:
                    try:
                        vel = ego_actor.get_velocity()
                        ego_speed = math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)
                    except Exception:
                        return effective_speed
                try:
                    ego_speed = float(ego_speed)
                except Exception:
                    return effective_speed
                if ego_speed < 0.1:
                    return 0.0
                try:
                    ego_loc = CarlaDataProvider.get_location(ego_actor) or ego_actor.get_location()
                    npc_loc = CarlaDataProvider.get_location(_actor) or _actor.get_location()
                    dist = ego_loc.distance(npc_loc)
                except Exception:
                    return min(effective_speed, ego_speed)
                dist_gap = max(0.0, dist - float(trigger_distance or 8.0))
                max_catchup_time = 10.0
                max_speed = max(0.0, float(target_speed if target_speed is not None else effective_speed))
                gap_norm = min(1.0, dist_gap / 25.0)
                min_speed = 1.0 + (2.0 * (1.0 - gap_norm))
                if max_speed > 0.0:
                    min_speed = min(min_speed, max_speed)
                else:
                    min_speed = 0.0
                desired = ego_speed - (dist_gap / max(1e-3, max_catchup_time))
                desired = max(min_speed, min(max_speed, desired))
                desired = min(desired, ego_speed * 0.98)
                result_speed = max(0.0, desired)
                # Debug: print every 20 ticks
                if _debug_counter[0] % 20 == 0:
                    print(f"[SPEED_CB] dist={dist:.1f}m, trigger_dist={trigger_distance}, gap={dist_gap:.1f}m, ego_spd={ego_speed:.1f}, npc_spd={result_speed:.1f}")
                return result_speed

        # MINIMUM DRIVE TIME: Ensure NPC drives naturally for a bit before trigger can fire
        # This prevents immediate triggering if NPC spawns within trigger distance
        min_drive_time = 3.0  # seconds

        follow_plan = plan
        resume_plan = plan
        if needs_catchup and has_distance_trigger:
            # Avoid short plans ending before the trigger fires.
            follow_plan = None
            resume_plan = None

        if action_type == "start_motion":
            if trigger_cond is None:
                return WaypointFollower(
                    actor,
                    target_speed=target_speed,
                    plan=plan,
                    avoid_collision=avoid_collision,
                    name=f"FollowWaypoints-{actor_plan.get('name')}",
                )
            seq = py_trees.composites.Sequence(name=f"TriggerStart-{actor_plan.get('name')}")
            seq.add_child(trigger_cond)
            seq.add_child(
                WaypointFollower(
                    actor,
                    target_speed=target_speed,
                    plan=plan,
                    avoid_collision=avoid_collision,
                    name=f"FollowWaypoints-{actor_plan.get('name')}",
                )
            )
            return seq

        if action_type == "hard_brake":
            # HARD BRAKE BEHAVIOR:
            # - NPC drives at effective_speed (distance-based pacing)
            # - Must drive for minimum time before trigger can fire (prevents immediate trigger)
            # - When trigger fires, NPC brakes hard
            # - After 2 seconds, NPC releases handbrake and resumes driving at normal speed
            
            follow = WaypointFollower(
                actor,
                target_speed=effective_speed,
                plan=follow_plan,
                avoid_collision=avoid_collision,
                speed_callback=speed_callback,
                name=f"FollowWaypoints-{actor_plan.get('name')}",
            )
            
            # Brake sequence: wait for combined trigger -> terminate follower -> brake -> handbrake
            brake_seq = py_trees.composites.Sequence(name=f"TriggerBrake-{actor_plan.get('name')}")
            
            if trigger_cond is not None and needs_catchup:
                # Combined trigger: BOTH minimum drive time AND distance trigger must be satisfied
                # This prevents immediate triggering if NPC spawns within trigger distance
                combined_trigger = py_trees.composites.Parallel(
                    name=f"CombinedTrigger-{actor_plan.get('name')}",
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL,
                )
                combined_trigger.add_child(Idle(duration=min_drive_time, name=f"MinDriveTime-{actor_plan.get('name')}"))
                combined_trigger.add_child(trigger_cond)
                brake_seq.add_child(combined_trigger)
            elif trigger_cond is not None:
                brake_seq.add_child(trigger_cond)
            # else: no trigger, brake immediately
            
            brake_seq.add_child(TerminateWaypointFollower(actor))
            brake_seq.add_child(StopVehicle(actor, brake_value=1.0))
            brake_seq.add_child(HandBrakeVehicle(actor, hand_brake_value=1.0))
            
            # Parallel: follow while waiting for brake trigger
            # Ends when brake_seq completes (SUCCESS_ON_ONE)
            par = py_trees.composites.Parallel(
                name=f"BrakeParallel-{actor_plan.get('name')}",
                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
            )
            par.add_child(follow)
            par.add_child(brake_seq)
            
            # Main sequence: parallel (drive+brake) -> wait -> release -> resume driving
            main_seq = py_trees.composites.Sequence(name=f"HardBrakeBehavior-{actor_plan.get('name')}")
            main_seq.add_child(par)
            # Wait 2 seconds while stopped
            main_seq.add_child(Idle(duration=2.0, name=f"BrakeWait-{actor_plan.get('name')}"))
            # Release handbrake
            main_seq.add_child(HandBrakeVehicle(actor, hand_brake_value=0.0))
            # Resume driving with a fresh follower so the actor doesn't remain stationary.
            resume_speed = target_speed if target_speed is not None else effective_speed
            main_seq.add_child(
                WaypointFollower(
                    actor,
                    target_speed=resume_speed,
                    plan=resume_plan,
                    avoid_collision=avoid_collision,
                    name=f"FollowWaypoints-{actor_plan.get('name')}-resume",
                )
            )
            return main_seq

        if action_type == "lane_change":
            direction = str(action_spec.get("direction") or "").strip().lower()
            if direction not in ("left", "right"):
                target_vehicle_name = action_spec.get("target_vehicle")
                target_vehicle = self._resolve_ego_vehicle(target_vehicle_name)
                if target_vehicle is not None:
                    direction = self._infer_lane_change_direction(actor, target_vehicle) or direction
            if direction not in ("left", "right"):
                direction = "left"

            # LANE CHANGE BEHAVIOR:
            # - NPC drives at distance-based pacing
            # - Must drive for minimum time before trigger can fire (prevents immediate trigger)
            # - When trigger fires, NPC changes lanes
            # - Then resumes driving at normal speed
            
            follow = WaypointFollower(
                actor,
                target_speed=effective_speed,
                plan=follow_plan,
                avoid_collision=avoid_collision,
                speed_callback=speed_callback,
                name=f"FollowWaypoints-{actor_plan.get('name')}",
            )

            action_seq = py_trees.composites.Sequence(name=f"TriggerLaneChange-{actor_plan.get('name')}")
            
            if trigger_cond is not None and needs_catchup:
                # Combined trigger: BOTH minimum drive time AND distance trigger must be satisfied
                combined_trigger = py_trees.composites.Parallel(
                    name=f"CombinedTrigger-{actor_plan.get('name')}",
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL,
                )
                combined_trigger.add_child(Idle(duration=min_drive_time, name=f"MinDriveTime-{actor_plan.get('name')}"))
                combined_trigger.add_child(trigger_cond)
                action_seq.add_child(combined_trigger)
            elif trigger_cond is not None:
                action_seq.add_child(trigger_cond)
            
            # Terminate the parallel WaypointFollower before starting lane change
            # This prevents two controllers from fighting over the actor
            action_seq.add_child(TerminateWaypointFollower(actor))
            
            action_seq.add_child(
                LaneChange(
                    actor,
                    speed=effective_speed,
                    direction=direction,
                )
            )
            # Resume at normal speed after lane change
            resume_speed = target_speed if target_speed is not None else effective_speed
            action_seq.add_child(
                WaypointFollower(
                    actor,
                    target_speed=resume_speed,
                    plan=resume_plan,
                    avoid_collision=avoid_collision,
                    name=f"FollowWaypoints-{actor_plan.get('name')}-resume",
                )
            )

            par = py_trees.composites.Parallel(
                name=f"LaneChangeParallel-{actor_plan.get('name')}",
                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
            )
            par.add_child(follow)
            par.add_child(action_seq)
            return par

        return None

    def _create_behavior(self):
        """
        Basic behavior do nothing, i.e. Idle
        """
        scenario_trigger_distance = 1.5  # Max trigger distance between route and scenario
        behavior = []
        log_replay = os.environ.get("CUSTOM_ACTOR_LOG_REPLAY", "").lower() in (
            "1",
            "true",
            "yes",
        )
        log_replay_ego = os.environ.get("CUSTOM_EGO_LOG_REPLAY", "").lower() in (
            "1",
            "true",
            "yes",
        )
        use_staging = os.environ.get("CUSTOM_LOG_REPLAY_STAGE", "").lower() in (
            "1",
            "true",
            "yes",
        )
        actor_image_dir = os.environ.get("CUSTOM_ACTOR_IMAGE_DIR")
        world = CarlaDataProvider.get_world()
        world_map = CarlaDataProvider.get_map()

        for ego_vehicle_id in range(len(self.list_scenarios)):
            behavior_tmp = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

            subbehavior = py_trees.composites.Parallel(name="Behavior",
                                                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)

            if log_replay_ego:
                transforms = (
                    self._ego_replay_transforms[ego_vehicle_id]
                    if ego_vehicle_id < len(self._ego_replay_transforms)
                    else []
                )
                times = (
                    self._ego_replay_times[ego_vehicle_id]
                    if ego_vehicle_id < len(self._ego_replay_times)
                    else None
                )
                if transforms and times:
                    stage_tf = None
                    if use_staging:
                        stage_tf = _make_stage_transform(
                            100000 + ego_vehicle_id,
                            transforms[0],
                        )
                    subbehavior.add_child(
                        LogReplayFollower(
                            self.ego_vehicles[ego_vehicle_id],
                            transforms,
                            times,
                            name=f"LogReplayEgo-{ego_vehicle_id}",
                            stage_transform=stage_tf,
                            stage_before=use_staging,
                            stage_after=use_staging,
                            done_blackboard_key=f"log_replay_done_ego_{ego_vehicle_id}",
                        )
                    )
                else:
                    print(
                        f"[RouteScenario] Ego log replay requested but missing timing data for ego {ego_vehicle_id}."
                    )

            if ego_vehicle_id == 0 and self._custom_actor_plans:
                for actor_plan in self._custom_actor_plans:
                    if os.environ.get('DEBUG_FINISH', '').lower() in ('1', 'true', 'yes'):
                        print(f"[DEBUG FINISH] Building behavior for actor: {actor_plan.get('name')}")
                        print(f"[DEBUG FINISH]   behavior spec: {actor_plan.get('behavior')}")
                        print(f"[DEBUG FINISH]   target_speed: {actor_plan.get('target_speed')}")
                        print(f"[DEBUG FINISH]   plan length: {len(actor_plan.get('plan') or [])}")
                    if log_replay and actor_plan.get("plan_times") and actor_plan.get("plan_transforms"):
                        capture_cb = None
                        finalize_cb = None
                        record_list = actor_plan.get("realized_path")
                        if actor_image_dir:
                            def _make_cb(plan=actor_plan):
                                return lambda stage: self._capture_custom_actor_image_stage(
                                    plan.get("actor"),
                                    plan.get("name", "unknown"),
                                    plan.get("role", "npc"),
                                    actor_image_dir,
                                    stage,
                                )
                            capture_cb = _make_cb()
                            def _make_finalize_cb(plan=actor_plan):
                                return lambda: self._save_custom_actor_path_visualization(plan, actor_image_dir)
                            finalize_cb = _make_finalize_cb()
                        spawn_cb = None
                        despawn_cb = None
                        spawn_time = actor_plan.get("spawn_time")
                        def _make_despawn_cb(plan=actor_plan):
                            def _despawn(actor):
                                try:
                                    CarlaDataProvider.remove_actor_by_id(actor.id)
                                except Exception:  # pylint: disable=broad-except
                                    try:
                                        actor.destroy()
                                    except Exception:  # pylint: disable=broad-except
                                        pass
                                try:
                                    self.other_actors.remove(actor)
                                except Exception:  # pylint: disable=broad-except
                                    pass
                                plan["actor"] = None
                            return _despawn
                        if actor_plan.get("actor") is None and actor_plan.get("spawn_model") is not None:
                            def _make_spawn_cb(plan=actor_plan):
                                def _spawn():
                                    world = CarlaDataProvider.get_world()
                                    world_map = CarlaDataProvider.get_map()
                                    model = plan.get("spawn_model")
                                    rolename = plan.get("name") or "unknown"
                                    spawn_tf = plan.get("spawn_transform")
                                    state = self._custom_actor_spawn_states.get(rolename)
                                    attempts = 0
                                    try:
                                        actor = None
                                        lift_used = None
                                        offset_used = None
                                        last_exc = None
                                        attempts += 1
                                        try:
                                            actor = CarlaDataProvider.request_new_actor(
                                                model,
                                                spawn_tf,
                                                rolename=rolename,
                                                autopilot=False,
                                            )
                                        except Exception as exc:  # pylint: disable=broad-except
                                            actor = None
                                            last_exc = exc

                                        if actor is None:
                                            retry_offsets = plan.get("spawn_retry_offsets") or []
                                            if retry_offsets:
                                                lane_type = carla.LaneType.Driving
                                                if plan.get("role") in ("pedestrian", "walker", "bicycle", "cyclist"):
                                                    lane_type = carla.LaneType.Any
                                                (
                                                    actor,
                                                    retry_tf,
                                                    retry_offset,
                                                    retry_exc,
                                                    retry_attempts,
                                                ) = _spawn_with_offsets(
                                                    model,
                                                    rolename,
                                                    spawn_tf,
                                                    retry_offsets,
                                                    world,
                                                    world_map,
                                                    lane_type,
                                                    normalize_actor_z=bool(plan.get("normalize_actor_z")),
                                                    autopilot=False,
                                                )
                                                attempts += retry_attempts
                                                if actor is not None and retry_tf is not None:
                                                    spawn_tf = retry_tf
                                                    plan["spawn_transform"] = retry_tf
                                                    offset_used = retry_offset
                                                elif retry_exc is not None:
                                                    last_exc = retry_exc

                                        if actor is None and os.environ.get("CUSTOM_ACTOR_DYNAMIC_Z", "1").lower() in (
                                            "1",
                                            "true",
                                            "yes",
                                        ):
                                            lift_steps = _parse_lift_steps(
                                                os.environ.get(
                                                    "CUSTOM_ACTOR_SPAWN_LIFT_STEPS", "0.2,0.5,1.0"
                                                )
                                            )
                                            actor, lift_used, lift_exc = _spawn_with_lifts(
                                                model,
                                                rolename,
                                                spawn_tf,
                                                lift_steps,
                                                autopilot=False,
                                            )
                                            attempts += len(lift_steps)
                                            if actor is None and lift_exc is not None:
                                                last_exc = lift_exc

                                        if actor is None:
                                            if last_exc is None:
                                                raise RuntimeError("request_new_actor returned None")
                                            raise last_exc
                                    except Exception as exc:  # pylint: disable=broad-except
                                        if state is not None:
                                            state["attempts"] = int(state.get("attempts", 0)) + int(attempts)
                                            state["last_failure"] = str(exc)
                                        print(
                                            f"[RouteScenario] Log replay spawn failed name={rolename} model={model}: {exc}"
                                        )
                                        _emit_spawn_debug(
                                            "log_replay_spawn_exception",
                                            plan,
                                            spawn_tf,
                                            plan.get("snap_to_road", False),
                                            plan.get("spawn_debug", {}).get("ground_z"),
                                            plan.get("spawn_debug", {}).get("stage_replay", False),
                                            world,
                                            world_map,
                                            bool(plan.get("normalize_actor_z")),
                                            bool(plan.get("follow_exact")),
                                            True,
                                            os.environ.get("CUSTOM_ACTOR_SPAWN_DEBUG", "").lower() in ("1", "true", "yes"),
                                        )
                                        return None
                                    if state is not None:
                                        state["attempts"] = int(state.get("attempts", 0)) + int(attempts)
                                    if actor is not None and lift_used is not None:
                                        if plan.get("normalize_actor_z"):
                                            lane_type = carla.LaneType.Driving
                                            if plan.get("role") in ("pedestrian", "walker", "bicycle", "cyclist"):
                                                lane_type = carla.LaneType.Any
                                            grounded_tf = _ground_actor_transform(
                                                actor,
                                                spawn_tf,
                                                world_map,
                                                world,
                                                lane_type,
                                            )
                                            if grounded_tf is not None:
                                                try:
                                                    actor.set_transform(grounded_tf)
                                                except Exception:  # pylint: disable=broad-except
                                                    pass
                                        else:
                                            try:
                                                actor.set_transform(spawn_tf)
                                            except Exception:  # pylint: disable=broad-except
                                                pass
                                    elif actor is not None and offset_used is not None:
                                        print(
                                            "[RouteScenario] Log replay spawn retry succeeded name={} offset=({:+.2f},{:+.2f},{:+.2f})".format(
                                                rolename,
                                                float(offset_used[0]),
                                                float(offset_used[1]),
                                                float(offset_used[2]),
                                            )
                                        )
                                    if actor is None:
                                        if state is not None:
                                            state["last_failure"] = "request_new_actor returned None"
                                        print(
                                            f"[RouteScenario] Log replay spawn returned None name={rolename} model={model}"
                                        )
                                        _emit_spawn_debug(
                                            "log_replay_spawn_none",
                                            plan,
                                            spawn_tf,
                                            plan.get("snap_to_road", False),
                                            plan.get("spawn_debug", {}).get("ground_z"),
                                            plan.get("spawn_debug", {}).get("stage_replay", False),
                                            world,
                                            world_map,
                                            bool(plan.get("normalize_actor_z")),
                                            bool(plan.get("follow_exact")),
                                            True,
                                            os.environ.get("CUSTOM_ACTOR_SPAWN_DEBUG", "").lower() in ("1", "true", "yes"),
                                        )
                                        return None
                                    if state is not None:
                                        state["spawned"] = True
                                        state["last_failure"] = None
                                        if offset_used is not None:
                                            state["spawn_offset"] = offset_used
                                    plan["actor"] = actor
                                    self.other_actors.append(actor)
                                    if plan.get("normalize_actor_z") and plan.get("plan_transforms"):
                                        lane_type = carla.LaneType.Driving
                                        if plan.get("role") in ("pedestrian", "walker", "bicycle", "cyclist"):
                                            lane_type = carla.LaneType.Any
                                        _glue_plan_to_ground(
                                            plan.get("plan_transforms"),
                                            actor,
                                            world_map,
                                            lane_type,
                                            world,
                                            prefer_ray_ground=bool(plan.get("ground_prefer_ray")),
                                            z_extra=float(plan.get("ground_z_extra", 0.0)),
                                        )
                                        try:
                                            actor.set_transform(plan["plan_transforms"][0])
                                        except Exception:  # pylint: disable=broad-except
                                            pass
                                    return actor
                                return _spawn
                            spawn_cb = _make_spawn_cb()
                            despawn_cb = _make_despawn_cb()
                        if despawn_cb is None and not actor_plan.get("stage_after", False):
                            despawn_cb = _make_despawn_cb()
                        custom_behavior = LogReplayFollower(
                            actor_plan["actor"],
                            actor_plan.get("plan_transforms"),
                            actor_plan.get("plan_times"),
                            name=f"LogReplay-{actor_plan['name']}",
                            stage_transform=actor_plan.get("stage_transform"),
                            stage_before=actor_plan.get("stage_before", False),
                            stage_after=actor_plan.get("stage_after", False),
                            fail_on_exception=False,
                            capture_callback=capture_cb,
                            record_list=record_list,
                            finalize_callback=finalize_cb,
                            spawn_cb=spawn_cb,
                            spawn_time=spawn_time,
                            despawn_cb=despawn_cb,
                            ground_each_tick=bool(actor_plan.get("normalize_actor_z")),
                            ground_lane_type=(
                                carla.LaneType.Any
                                if actor_plan.get("role") in ("pedestrian", "walker", "bicycle", "cyclist")
                                else carla.LaneType.Driving
                            ),
                            ground_world_map=world_map,
                            ground_world=world,
                            ground_prefer_ray=bool(actor_plan.get("ground_prefer_ray")),
                            ground_z_extra=float(actor_plan.get("ground_z_extra", 0.0)),
                            animate_walkers=bool(actor_plan.get("animate_walkers", True)),
                        )
                        if os.environ.get('DEBUG_FINISH', '').lower() in ('1', 'true', 'yes'):
                            print("[DEBUG FINISH]   -> Using LogReplayFollower")
                    else:
                        custom_behavior = self._build_custom_actor_behavior(actor_plan)
                        if custom_behavior is None:
                            if os.environ.get('DEBUG_FINISH', '').lower() in ('1', 'true', 'yes'):
                                print(f"[DEBUG FINISH]   -> Behavior is None, using default WaypointFollower")
                            custom_behavior = WaypointFollower(
                                actor_plan["actor"],
                                target_speed=actor_plan["target_speed"],
                                plan=actor_plan["plan"],
                                avoid_collision=actor_plan["avoid_collision"],
                                name=f"FollowWaypoints-{actor_plan['name']}",
                            )
                        else:
                            if os.environ.get('DEBUG_FINISH', '').lower() in ('1', 'true', 'yes'):
                                print(f"[DEBUG FINISH]   -> Built custom behavior: {type(custom_behavior).__name__}")
                    subbehavior.add_child(custom_behavior)

            scenario_behaviors = []
            blackboard_list = []
            list_scenarios = self.list_scenarios[ego_vehicle_id]
            for i, scenario in enumerate(list_scenarios):
                if scenario.scenario.behavior is not None:
                    route_var_name = scenario.config.route_var_name

                    if route_var_name is not None:
                        scenario_behaviors.append(scenario.scenario.behavior)
                        blackboard_list.append([scenario.config.route_var_name,
                                                scenario.config.trigger_points[0].location])
                    else:
                        name = "{} - {}".format(i, scenario.scenario.behavior.name)
                        oneshot_idiom = oneshot_behavior(
                            name=name,
                            variable_name=name,
                            behaviour=scenario.scenario.behavior)
                        scenario_behaviors.append(oneshot_idiom)

            # Add behavior that manages the scenarios trigger conditions
            scenario_triggerer = ScenarioTriggerer(
                self.ego_vehicles[ego_vehicle_id],
                self.route[ego_vehicle_id],
                blackboard_list,
                scenario_trigger_distance,
                repeat_scenarios=False
            )

            subbehavior.add_child(scenario_triggerer)  # make ScenarioTriggerer the first thing to be checked
            subbehavior.add_children(scenario_behaviors)
            subbehavior.add_child(Idle())  # The behaviours cannot make the route scenario stop
            behavior_tmp.add_child(subbehavior)
            behavior.append(behavior_tmp)
        return behavior

    def _create_test_criteria(self):
        """
        """
        criteria_all = []
        log_replay_ego = os.environ.get("CUSTOM_EGO_LOG_REPLAY", "").lower() in (
            "1",
            "true",
            "yes",
        )
        for ego_vehicle_id in range(len(self.list_scenarios)):
            criteria = []
            route = convert_transform_to_location(self.route[ego_vehicle_id])
            if log_replay_ego and ego_vehicle_id < len(self._ego_replay_transforms):
                replay_transforms = self._ego_replay_transforms[ego_vehicle_id]
                if replay_transforms:
                    replay_route = []
                    for tf in replay_transforms:
                        loc = tf.location
                        if not (math.isfinite(loc.x) and math.isfinite(loc.y) and math.isfinite(loc.z)):
                            continue
                        replay_route.append((carla.Location(x=loc.x, y=loc.y, z=loc.z), RoadOption.LANEFOLLOW))
                    if len(replay_route) >= 2:
                        route = replay_route
            collision_criterion = CollisionTest(self.ego_vehicles[ego_vehicle_id], terminate_on_failure=False)

            route_criterion = InRouteTest(self.ego_vehicles[ego_vehicle_id],
                                        route=route,
                                        offroad_max=30,
                                        terminate_on_failure=not log_replay_ego)
                                        
            completion_criterion = RouteCompletionTest(self.ego_vehicles[ego_vehicle_id], route=route)

            outsidelane_criterion = OutsideRouteLanesTest(self.ego_vehicles[ego_vehicle_id], route=route)

            red_light_criterion = RunningRedLightTest(self.ego_vehicles[ego_vehicle_id])

            stop_criterion = RunningStopTest(self.ego_vehicles[ego_vehicle_id])

            blocked_criterion = ActorSpeedAboveThresholdTest(self.ego_vehicles[ego_vehicle_id],
                                                            speed_threshold=0.5,
                                                            below_threshold_max_time=30.0,
                                                            terminate_on_failure=True,
                                                            name="AgentBlockedTest")

            criteria.append(completion_criterion)
            criteria.append(outsidelane_criterion)
            criteria.append(collision_criterion)
            criteria.append(red_light_criterion)
            criteria.append(stop_criterion)
            criteria.append(route_criterion)
            criteria.append(blocked_criterion)
            criteria_all.append(criteria)
            
        return criteria_all

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
