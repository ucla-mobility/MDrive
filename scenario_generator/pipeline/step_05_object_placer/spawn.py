import os
from typing import Any, Dict, List, Optional

import numpy as np

from .constants import LANE_WIDTH_M, LATERAL_TO_M, SIDEWALK_OFFSET_M
from .geometry import (
    cumulative_dist,
    heading_deg_from_vec,
    left_normal_world,
    point_and_tangent_at_s,
    right_normal_world,
    wrap180,
)

def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return default


def infer_speed_mps(category: str, speed_profile: str) -> float:
    # Baselines: walkers ~1.7, cyclists ~4.0, vehicles ~8.0
    speed_profile = (speed_profile or "normal").lower()
    base = 1.7 if category == "walker" else (4.0 if category == "cyclist" else 8.0)
    if speed_profile in ("stopped", "stop", "static"):
        return 0.0
    if speed_profile == "slow":
        return 0.6 * base
    if speed_profile == "fast":
        return 1.6 * base
    if speed_profile == "erratic":
        # Erratic uses a higher base speed (actual speed varies in waypoint generation)
        return 1.3 * base
    return base


def resolve_nodes_path(picked_path_json_path: str, nodes_field: str, nodes_root: Optional[str]) -> str:
    # If nodes_field is absolute, use it. Otherwise resolve relative to picked_paths_detailed.json dir or --nodes-root.
    if os.path.isabs(nodes_field) and os.path.exists(nodes_field):
        return nodes_field
    # Try nodes_root first (if provided)
    if nodes_root:
        cand = os.path.join(nodes_root, nodes_field)
        if os.path.exists(cand):
            return cand
    # Resolve relative to the picked json file location
    base = os.path.dirname(os.path.abspath(picked_path_json_path))
    cand = os.path.join(base, nodes_field)
    if os.path.exists(cand):
        return cand
    # Final fallback: keep as-is (caller may want to handle)
    return nodes_field


def compute_spawn_from_anchor(
    seg_points: np.ndarray,
    s_along: float,
    lateral_relation: str,
    lateral_offset_m: Optional[float] = None,
) -> Dict[str, float]:
    p, t = point_and_tangent_at_s(seg_points, s_along)
    yaw = wrap180(heading_deg_from_vec(t))
    lat_m = float(lateral_offset_m) if lateral_offset_m is not None else float(LATERAL_TO_M.get(lateral_relation, 0.0))

    if abs(lat_m) > 1e-9:
        if lat_m > 0:
            n = right_normal_world(t)
        else:
            n = left_normal_world(t)
        p = p + abs(lat_m) * n

    return {"x": float(p[0]), "y": float(p[1]), "yaw_deg": float(yaw)}


def build_motion_waypoints(
    motion: Dict[str, Any],
    category: str,
    anchor_spawn: Dict[str, float],
    seg_points: np.ndarray,
) -> List[Dict[str, Any]]:
    mtype = motion.get("type", "static")
    speed_profile = motion.get("speed_profile", "normal")
    is_erratic = speed_profile == "erratic"
    speed = infer_speed_mps(category, speed_profile)

    if mtype == "static":
        return [{
            "x": anchor_spawn["x"],
            "y": anchor_spawn["y"],
            "yaw_deg": anchor_spawn["yaw_deg"],
            "speed_mps": 0.0,
        }]

    if mtype == "follow_lane":
        # Create a short forward trajectory along the segment
        s0 = float(motion.get("start_s_along", None) or 0.0)
        if s0 <= 0.0:
            s0 = None
        # start at anchor, then go forward by delta_s
        delta_m = float(motion.get("travel_distance_m", 18.0))
        # Convert delta_m to delta_s by arc length
        cum = cumulative_dist(seg_points)
        total = float(cum[-1]) if len(cum) else 0.0
        if total < 1e-6:
            return [{
                "x": anchor_spawn["x"],
                "y": anchor_spawn["y"],
                "yaw_deg": anchor_spawn["yaw_deg"],
                "speed_mps": speed,
            }]

        # anchor at s_anchor, then advance by delta_m
        # Determine current arc distance for anchor
        p_anchor, _ = point_and_tangent_at_s(seg_points, float(motion.get("anchor_s_along", 0.5)))
        # Find closest point index for approximate s distance
        dists = np.linalg.norm(seg_points - np.array([anchor_spawn["x"], anchor_spawn["y"]])[None, :], axis=1)
        idx = int(np.argmin(dists))
        s_anchor_dist = float(cum[idx])
        s_end_dist = min(total, s_anchor_dist + delta_m)

        # Sample along [s_anchor_dist, s_end_dist]
        num = int(motion.get("num_waypoints", 8))
        num = max(2, min(40, num))
        waypoints = []
        
        if is_erratic:
            # Erratic driving: variable speed, lane weaving, unpredictable heading
            import random
            random.seed(hash(str(anchor_spawn)))  # Deterministic but varied
            
            for k in range(num):
                target = s_anchor_dist + (s_end_dist - s_anchor_dist) * (k / (num - 1))
                s_frac = target / total if total > 1e-6 else 0.0
                spawn = compute_spawn_from_anchor(seg_points, s_frac, "center")
                
                # Erratic speed: varies between 50% and 150% of base speed
                erratic_speed = speed * (0.5 + random.random())
                
                # Erratic lateral offset: weave side to side (up to 1.5m)
                lat_offset = (random.random() - 0.5) * 3.0  # -1.5m to +1.5m
                _, t = point_and_tangent_at_s(seg_points, s_frac)
                n = right_normal_world(t) if lat_offset >= 0 else left_normal_world(t)
                spawn["x"] += abs(lat_offset) * n[0]
                spawn["y"] += abs(lat_offset) * n[1]
                
                # Erratic heading: slight random deviation (up to ±10 degrees)
                spawn["yaw_deg"] = wrap180(spawn["yaw_deg"] + (random.random() - 0.5) * 20)
                
                waypoints.append({**spawn, "speed_mps": erratic_speed, "erratic": True})
        else:
            # Honor actor's lateral relation for lane-following.
            lateral = str(motion.get("start_lateral", "center") or "center").lower()
            for k in range(num):
                target = s_anchor_dist + (s_end_dist - s_anchor_dist) * (k / (num - 1))
                s_frac = target / total if total > 1e-6 else 0.0
                spawn = compute_spawn_from_anchor(seg_points, s_frac, lateral)
                waypoints.append({**spawn, "speed_mps": speed})
        return waypoints

    if mtype == "cross_perpendicular":
        # For crossing motion, the pedestrian should:
        # 1. ALWAYS start from the sidewalk (off-road), not lane edge or center
        # 2. Cross the entire road width to the opposite sidewalk
        
        # CRITICAL FIX: For pedestrian crosswalks, ALWAYS use sidewalk offsets
        # Never spawn pedestrians in the middle of lanes or at lane edges
        
        # Determine normal from segment tangent at anchor
        _, t = point_and_tangent_at_s(seg_points, float(motion.get("anchor_s_along", 0.5)))
        n_right = right_normal_world(t)
        
        # Cross direction: use explicit direction, or infer from start lateral
        side = str(motion.get("cross_direction", "unknown")).lower()
        start_lat = str(motion.get("start_lateral", "")).lower()
        
        if side not in ("left", "right"):
            if "right" in start_lat:
                side = "left"  # starting on right side, cross to left
            elif "left" in start_lat:
                side = "right"  # starting on left side, cross to right
            else:
                side = "left"  # default to crossing left
        
        # If local road-edge bounds are provided, use them directly so start/end are
        # guaranteed to be outside the outermost drivable lane by edge_clearance_m.
        road_lat_min = _safe_float(motion.get("road_lat_min_m"))
        road_lat_max = _safe_float(motion.get("road_lat_max_m"))
        edge_clearance_m = _safe_float(motion.get("edge_clearance_m"), 0.5)
        if edge_clearance_m is None:
            edge_clearance_m = 0.5
        edge_clearance_m = max(0.0, float(edge_clearance_m))

        # Get base point from lane center of the anchor segment
        center_spawn = compute_spawn_from_anchor(seg_points, float(motion.get("anchor_s_along", 0.5)), "center")
        p_center = np.array([center_spawn["x"], center_spawn["y"]], dtype=float)

        if (
            road_lat_min is not None
            and road_lat_max is not None
            and float(road_lat_max) > float(road_lat_min)
        ):
            if side == "left":
                # Start from right side of roadway envelope, cross toward left.
                start_lat = float(road_lat_max) + edge_clearance_m
                end_lat = float(road_lat_min) - edge_clearance_m
            else:
                # Start from left side of roadway envelope, cross toward right.
                start_lat = float(road_lat_min) - edge_clearance_m
                end_lat = float(road_lat_max) + edge_clearance_m

            p0 = p_center + start_lat * n_right
            p1 = p_center + end_lat * n_right
            cross_vec = p1 - p0
            norm = float(np.linalg.norm(cross_vec))
            if norm > 1e-9:
                yaw = wrap180(heading_deg_from_vec(cross_vec / norm))
            else:
                yaw = wrap180(heading_deg_from_vec(n_right))
            return [
                {"x": float(p0[0]), "y": float(p0[1]), "yaw_deg": float(yaw), "speed_mps": speed},
                {"x": float(p1[0]), "y": float(p1[1]), "yaw_deg": float(yaw), "speed_mps": speed},
            ]

        # Fallback: infer sidewalk offset scalar and crossing distance.
        start_offset_m = float(motion.get("start_offset_m", SIDEWALK_OFFSET_M))
        if start_offset_m < SIDEWALK_OFFSET_M:
            start_offset_m = SIDEWALK_OFFSET_M

        # Calculate road crossing geometry
        # Crossing distance: from sidewalk to opposite sidewalk across the entire road
        # Need to cross: sidewalk + all lanes + opposite sidewalk
        # For multi-lane roads, use a larger crossing distance
        default_road_crossing_m = 2 * start_offset_m + 2 * LANE_WIDTH_M
        dist = float(motion.get("cross_distance_m", default_road_crossing_m))
        
        # Determine start side based on cross direction
        # If crossing "left" (rightward to leftward), start from right side (offroad_right)
        # If crossing "right" (leftward to rightward), start from left side (offroad_left)
        if side == "left":
            # Start from right sidewalk, cross left
            start_normal = n_right
            cross_normal = left_normal_world(t)
        else:
            # Start from left sidewalk, cross right
            start_normal = left_normal_world(t)
            cross_normal = n_right
        
        # Calculate start point on the sidewalk (off-road)
        p0 = p_center + start_offset_m * start_normal
        
        # Calculate end point on the opposite sidewalk
        p1 = p0 + dist * cross_normal
        
        yaw = wrap180(heading_deg_from_vec(cross_normal))
        return [
            {"x": float(p0[0]), "y": float(p0[1]), "yaw_deg": float(yaw), "speed_mps": speed},
            {"x": float(p1[0]), "y": float(p1[1]), "yaw_deg": float(yaw), "speed_mps": speed},
        ]

    if mtype == "straight_line":
        end_anchor = motion.get("end_anchor", {})
        wp = end_anchor.get("world_point")
        if not isinstance(wp, dict) or "x" not in wp or "y" not in wp:
            # If no end point, just hold
            return [{"x": anchor_spawn["x"], "y": anchor_spawn["y"], "yaw_deg": anchor_spawn["yaw_deg"], "speed_mps": speed}]
        p0 = np.array([anchor_spawn["x"], anchor_spawn["y"]], dtype=float)
        p1 = np.array([float(wp["x"]), float(wp["y"])] , dtype=float)
        v = p1 - p0
        n = float(np.linalg.norm(v))
        yaw = wrap180(heading_deg_from_vec(v / n)) if n > 1e-9 else anchor_spawn["yaw_deg"]
        return [
            {"x": float(p0[0]), "y": float(p0[1]), "yaw_deg": float(yaw), "speed_mps": speed},
            {"x": float(p1[0]), "y": float(p1[1]), "yaw_deg": float(yaw), "speed_mps": speed},
        ]

    # fallback
    return [{"x": anchor_spawn["x"], "y": anchor_spawn["y"], "yaw_deg": anchor_spawn["yaw_deg"], "speed_mps": speed}]


__all__ = [
    "build_motion_waypoints",
    "compute_spawn_from_anchor",
    "infer_speed_mps",
    "resolve_nodes_path",
]
