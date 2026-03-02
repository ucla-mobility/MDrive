"""
Lightweight PDM-style metric helpers adapted from navsim's PDMScorer.
Copied from navsim/planning/simulation/planner/pdm_planner/scoring/pdm_comfort_metrics.py
with minimal changes to remove nuPlan/navsim dependencies and use CARLA-recorded states.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Enable debug output only when DEBUG_PDM environment variable is set
_DEBUG_PDM = os.environ.get('DEBUG_PDM', '').lower() in ('1', 'true', 'yes')
if _DEBUG_PDM:
    print(f"[DEBUG PDM] Module loaded with DEBUG_PDM enabled")
try:
    import numpy.typing as npt
except Exception:
    # Fallback for older numpy that lacks numpy.typing
    class _NptShim:
        NDArray = np.ndarray
    npt = _NptShim()

try:
    from scipy.signal import savgol_filter
except ImportError:  # fallback to identity if SciPy isn't available
    def savgol_filter(x, *args, **kwargs):
        return x

try:
    import carla
except Exception:
    carla = None

try:
    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
except Exception:
    CarlaDataProvider = None

try:
    from shapely.geometry import Polygon, LineString, Point
except Exception:
    Polygon = None
    LineString = None
    Point = None


def _safe_savgol(y: np.ndarray, polyorder: int, window_length: int, axis: int = -1):
    """
    Ensure window_length is valid (odd, >= polyorder+2, <= len) for savgol_filter.
    Falls back to unfiltered data if constraints cannot be met.
    """
    n = y.shape[axis]
    if n < 3:
        return y

    wl = min(window_length, n)
    # make odd
    if wl % 2 == 0:
        wl = wl - 1
    # ensure still positive
    if wl < 3:
        return y
    # ensure large enough for polyorder
    if wl <= polyorder:
        # try smallest odd greater than polyorder within n
        wl = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3
        if wl > n:
            return y
    try:
        return savgol_filter(y, window_length=wl, polyorder=polyorder, axis=axis, mode="interp")
    except Exception:
        return y


# ---- State layout (copied from navsim/planning/simulation/planner/pdm_planner/utils/pdm_enums.py) ----
class StateIndex:
    _X = 0
    _Y = 1
    _HEADING = 2
    _VELOCITY_X = 3
    _VELOCITY_Y = 4
    _ACCELERATION_X = 5
    _ACCELERATION_Y = 6
    _STEERING_ANGLE = 7
    _STEERING_RATE = 8
    _ANGULAR_VELOCITY = 9
    _ANGULAR_ACCELERATION = 10

    @classmethod
    def size(cls) -> int:
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_") and not attribute.startswith("__") and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    # property accessors (match navsim usage)
    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def VELOCITY_X(cls):
        return cls._VELOCITY_X

    @classmethod
    @property
    def VELOCITY_Y(cls):
        return cls._VELOCITY_Y

    @classmethod
    @property
    def ACCELERATION_X(cls):
        return cls._ACCELERATION_X

    @classmethod
    @property
    def ACCELERATION_Y(cls):
        return cls._ACCELERATION_Y

    @classmethod
    @property
    def STEERING_ANGLE(cls):
        return cls._STEERING_ANGLE

    @classmethod
    @property
    def STEERING_RATE(cls):
        return cls._STEERING_RATE

    @classmethod
    @property
    def ANGULAR_VELOCITY(cls):
        return cls._ANGULAR_VELOCITY

    @classmethod
    @property
    def ANGULAR_ACCELERATION(cls):
        return cls._ANGULAR_ACCELERATION

    @classmethod
    @property
    def STATE_SE2(cls):
        return slice(cls._X, cls._HEADING + 1)

    @classmethod
    @property
    def VELOCITY_2D(cls):
        return slice(cls._VELOCITY_X, cls._VELOCITY_Y + 1)

    @classmethod
    @property
    def ACCELERATION_2D(cls):
        return slice(cls._ACCELERATION_X, cls._ACCELERATION_Y + 1)


# ---- Thresholds copied from navsim comfort metrics ----
MAX_ABS_MAG_JERK: float = 8.37  # [m/s^3]
MAX_ABS_LAT_ACCEL: float = 4.89  # [m/s^2]
MAX_LON_ACCEL: float = 2.40  # [m/s^2]
MIN_LON_ACCEL: float = -4.05
MAX_ABS_YAW_ACCEL: float = 1.93  # [rad/s^2]
MAX_ABS_LON_JERK: float = 4.13  # [m/s^3]
MAX_ABS_YAW_RATE: float = 0.95  # [rad/s]


def _within_bound(
    metric_value: npt.NDArray[np.float64],
    min_bound: Optional[float] = None,
    max_bound: Optional[float] = None,
) -> npt.NDArray[np.bool_]:
    mask = np.ones_like(metric_value, dtype=np.bool_)
    if min_bound is not None:
        mask = np.logical_and(mask, metric_value >= min_bound)
    if max_bound is not None:
        mask = np.logical_and(mask, metric_value <= max_bound)
    return mask


def _approximate_derivatives(
    y: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    window_length: int = 5,
    poly_order: int = 2,
    deriv_order: int = 1,
    axis: int = -1,
) -> npt.NDArray[np.float64]:
    """
    Savitzky-Golay derivative (copied from navsim). Assumes equally spaced samples.
    """
    n = y.shape[axis]
    if n < 3:
        return np.gradient(y, x, axis=axis)

    # ensure odd window length, large enough for polyorder, and <= n
    candidate = max(3, min(window_length, n))
    if candidate % 2 == 0:
        candidate -= 1
    if candidate <= poly_order:
        candidate = poly_order + 2 if (poly_order + 2) % 2 == 1 else poly_order + 3
    if candidate > n:
        return np.gradient(y, x, axis=axis)

    try:
        return savgol_filter(
            y,
            polyorder=min(poly_order, candidate - 1),
            window_length=candidate,
            deriv=deriv_order,
            delta=float(np.mean(np.diff(x))) if len(x) > 1 else 1.0,
            axis=axis,
            mode="interp",
        )
    except Exception:
        return np.gradient(y, x, axis=axis)


def _phase_unwrap(headings: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    two_pi = 2.0 * np.pi
    adjustments = np.zeros_like(headings)
    adjustments[..., 1:] = np.cumsum(np.round(np.diff(headings, axis=-1) / two_pi), axis=-1)
    unwrapped = headings - two_pi * adjustments
    return unwrapped


def _extract_ego_acceleration(
    states: npt.NDArray[np.float64],
    acceleration_coordinate: str,
    decimals: int = 8,
    poly_order: int = 2,
    window_length: int = 8,
) -> npt.NDArray[np.float64]:
    if acceleration_coordinate == "x":
        acceleration = states[..., StateIndex._ACCELERATION_X]
    elif acceleration_coordinate == "y":
        acceleration = states[..., StateIndex._ACCELERATION_Y]
    elif acceleration_coordinate == "magnitude":
        acceleration = np.hypot(
            states[..., StateIndex._ACCELERATION_X],
            states[..., StateIndex._ACCELERATION_Y],
        )
    else:
        raise ValueError(f"acceleration_coordinate {acceleration_coordinate} not supported")

    acceleration_filtered = _safe_savgol(
        acceleration,
        polyorder=min(poly_order, max(2, acceleration.shape[-1] - 1)),
        window_length=min(window_length, acceleration.shape[-1]),
        axis=-1,
    )
    return np.round(acceleration_filtered, decimals=decimals)


def _extract_ego_jerk(
    states: npt.NDArray[np.float64],
    acceleration_coordinate: str,
    time_steps_s: npt.NDArray[np.float64],
    decimals: int = 8,
    deriv_order: int = 1,
    poly_order: int = 2,
    window_length: int = 15,
) -> npt.NDArray[np.float64]:
    ego_acceleration = _extract_ego_acceleration(states, acceleration_coordinate=acceleration_coordinate)
    jerk = _approximate_derivatives(
        ego_acceleration,
        time_steps_s,
        deriv_order=deriv_order,
        poly_order=poly_order,
        window_length=min(window_length, ego_acceleration.shape[-1]),
    )
    return np.round(jerk, decimals=decimals)


def _extract_ego_yaw_rate(
    states: npt.NDArray[np.float64],
    time_steps_s: npt.NDArray[np.float64],
    deriv_order: int = 1,
    poly_order: int = 2,
    decimals: int = 8,
    window_length: int = 15,
) -> npt.NDArray[np.float64]:
    ego_headings = states[..., StateIndex._HEADING]
    ego_yaw_rate = _approximate_derivatives(
        _phase_unwrap(ego_headings),
        time_steps_s,
        deriv_order=deriv_order,
        poly_order=poly_order,
        window_length=max(3, min(window_length, ego_headings.shape[-1] - (ego_headings.shape[-1] + 1) % 2)),
    )
    return np.round(ego_yaw_rate, decimals=decimals)


def _compute_lon_acceleration(states, time_steps_s, _) -> npt.NDArray[np.bool_]:
    return _within_bound(
        _extract_ego_acceleration(states, "x"),
        min_bound=MIN_LON_ACCEL,
        max_bound=MAX_LON_ACCEL,
    ).all(axis=-1)


def _compute_lat_acceleration(states, time_steps_s, _) -> npt.NDArray[np.bool_]:
    return _within_bound(
        _extract_ego_acceleration(states, "y"),
        min_bound=-MAX_ABS_LAT_ACCEL,
        max_bound=MAX_ABS_LAT_ACCEL,
    ).all(axis=-1)


def _compute_jerk_metric(states, time_steps_s, _) -> npt.NDArray[np.bool_]:
    return _within_bound(
        _extract_ego_jerk(states, "magnitude", time_steps_s),
        min_bound=-MAX_ABS_MAG_JERK,
        max_bound=MAX_ABS_MAG_JERK,
    ).all(axis=-1)


def _compute_lon_jerk_metric(states, time_steps_s, _) -> npt.NDArray[np.bool_]:
    return _within_bound(
        _extract_ego_jerk(states, "x", time_steps_s),
        min_bound=-MAX_ABS_LON_JERK,
        max_bound=MAX_ABS_LON_JERK,
    ).all(axis=-1)


def _compute_yaw_accel(states, time_steps_s, _) -> npt.NDArray[np.bool_]:
    yaw_accel_metric = _extract_ego_yaw_rate(states, time_steps_s, deriv_order=2, poly_order=3)
    return _within_bound(yaw_accel_metric, min_bound=-MAX_ABS_YAW_ACCEL, max_bound=MAX_ABS_YAW_ACCEL).all(axis=-1)


def _compute_yaw_rate(states, time_steps_s, _) -> npt.NDArray[np.bool_]:
    yaw_rate_metric = _extract_ego_yaw_rate(states, time_steps_s)
    return _within_bound(yaw_rate_metric, min_bound=-MAX_ABS_YAW_RATE, max_bound=MAX_ABS_YAW_RATE).all(axis=-1)


def ego_is_comfortable(
    states: npt.NDArray[np.float64],
    time_point_s: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """
    Copied from navsim: evaluates six comfort metrics.
    """
    n_batch, n_time, n_states = states.shape
    assert n_time == len(time_point_s)
    assert n_states == StateIndex.size()

    comfort_metric_functions = [
        _compute_lon_acceleration,
        _compute_lat_acceleration,
        _compute_jerk_metric,
        _compute_lon_jerk_metric,
        _compute_yaw_accel,
        _compute_yaw_rate,
    ]
    results: npt.NDArray[np.bool_] = np.zeros((n_batch, len(comfort_metric_functions)), dtype=np.bool_)
    for idx, metric_function in enumerate(comfort_metric_functions):
        results[:, idx] = metric_function(states, time_point_s, None)
    return results


# ---- PDM-style aggregation helpers ---------------------------------------------------------------

PDM_WEIGHTS = np.array([5.0, 5.0, 2.0, 2.0, 2.0], dtype=np.float64)

# PDMScorerConfig defaults (navsim)
DRIVING_DIRECTION_HORIZON = 1.0
DRIVING_DIRECTION_COMPLIANCE_THRESHOLD = 2.0
DRIVING_DIRECTION_VIOLATION_THRESHOLD = 6.0
STOPPED_SPEED_THRESHOLD = 5e-03
FUTURE_COLLISION_HORIZON_WINDOW = 1.0
PROGRESS_DISTANCE_THRESHOLD = 5.0
LANE_KEEPING_DEVIATION_LIMIT = 0.5
LANE_KEEPING_HORIZON_WINDOW = 2.0


@dataclass
class PDMRouteMetrics:
    no_at_fault_collisions: float
    drivable_area_compliance: float
    driving_direction_compliance: float
    traffic_light_compliance: float
    ego_progress: float
    time_to_collision_within_bound: float
    lane_keeping: float
    history_comfort: float
    weighted_metrics: List[float]
    weighted_metrics_array: List[float]
    multiplicative_metrics_prod: float
    pdm_score: float
    comfort_components: Dict[str, float]
    available: bool
    ttc_min_time_s: Optional[float] = None
    ttc_min_distance_m: Optional[float] = None


def _trace_to_state_array(pdm_trace: List[Dict]) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Convert recorded CARLA trace to navsim-style state array and time vector.
    """
    times = np.array([sample["t"] for sample in pdm_trace], dtype=np.float64)
    if len(times) == 0:
        return np.zeros((1, 0, StateIndex.size())), times
    times = times - times[0]  # start at 0 for stability

    states = np.zeros((1, len(pdm_trace), StateIndex.size()), dtype=np.float64)
    for idx, sample in enumerate(pdm_trace):
        states[0, idx, StateIndex._X] = sample.get("x", 0.0)
        states[0, idx, StateIndex._Y] = sample.get("y", 0.0)
        states[0, idx, StateIndex._HEADING] = sample.get("yaw", 0.0)
        states[0, idx, StateIndex._VELOCITY_X] = sample.get("vel_x", 0.0)
        states[0, idx, StateIndex._VELOCITY_Y] = sample.get("vel_y", 0.0)
        states[0, idx, StateIndex._ACCELERATION_X] = sample.get("accel_x", 0.0)
        states[0, idx, StateIndex._ACCELERATION_Y] = sample.get("accel_y", 0.0)
        states[0, idx, StateIndex._ANGULAR_VELOCITY] = sample.get("ang_vel", 0.0)
        states[0, idx, StateIndex._ANGULAR_ACCELERATION] = sample.get("ang_acc", 0.0)
        # steering angle/rate not available from CARLA logs; leave zeros
    return states, times


def _oriented_box_with_front(x: float, y: float, yaw: float, extent_x: float, extent_y: float):
    """
    Build shapely polygon for an oriented bounding box and return front edge.
    """
    if Polygon is None:
        return None, None
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    local = [
        (extent_x, extent_y),   # front-left
        (extent_x, -extent_y),  # front-right
        (-extent_x, -extent_y), # rear-right
        (-extent_x, extent_y),  # rear-left
    ]
    world = []
    for lx, ly in local:
        wx = x + lx * cos_y - ly * sin_y
        wy = y + lx * sin_y + ly * cos_y
        world.append((wx, wy))
    poly = Polygon(world)
    front_edge = (world[0], world[1])
    return poly, front_edge


def _lane_key(wp):
    if wp is None:
        return None
    return (wp.road_id, wp.lane_id)


def _is_drivable_wp(wp):
    if wp is None or carla is None:
        return False
    drivable_mask = carla.LaneType.Driving | carla.LaneType.Bidirectional | carla.LaneType.Shoulder
    return bool(wp.lane_type & drivable_mask)


def _build_route_line_and_lane_ids(config, ego_id: int, carla_map):
    if config is None or LineString is None or carla_map is None:
        return None, set()
    trajectory = None
    if hasattr(config, "multi_traj") and config.multi_traj:
        try:
            trajectory = config.multi_traj[ego_id]
        except Exception:
            trajectory = config.multi_traj[0]
    if trajectory is None:
        trajectory = config.trajectory
    coords = [(loc.x, loc.y) for loc in trajectory if loc is not None]
    if len(coords) < 2:
        return None, set()
    route_line = LineString(coords)
    route_lane_ids = set()
    for x, y in coords:
        try:
            wp = carla_map.get_waypoint(carla.Location(x=x, y=y, z=0.0), project_to_road=True)
        except Exception:
            wp = None
        key = _lane_key(wp)
        if key is not None:
            route_lane_ids.add(key)
    return route_line, route_lane_ids


def _compute_ego_areas(pdm_trace, carla_map, route_lane_ids):
    n = len(pdm_trace)
    multiple_lanes = np.zeros(n, dtype=bool)
    non_drivable = np.zeros(n, dtype=bool)
    oncoming = np.zeros(n, dtype=bool)
    in_intersection = np.zeros(n, dtype=bool)
    if carla_map is None:
        return multiple_lanes, non_drivable, oncoming, in_intersection

    for i, sample in enumerate(pdm_trace):
        extent_x = sample.get("extent_x", 1.0)
        extent_y = sample.get("extent_y", 0.5)
        poly, _ = _oriented_box_with_front(sample["x"], sample["y"], sample["yaw"], extent_x, extent_y)
        if poly is None:
            continue
        corners = list(poly.exterior.coords)[:-1]
        corner_lane_keys = []
        drivable_corners = 0
        for cx, cy in corners:
            try:
                wp = carla_map.get_waypoint(carla.Location(x=cx, y=cy, z=0.0), project_to_road=False)
            except Exception:
                wp = None
            if _is_drivable_wp(wp):
                drivable_corners += 1
            key = _lane_key(wp)
            if key is not None:
                corner_lane_keys.append(key)
        non_drivable[i] = drivable_corners < 4
        unique_lanes = set(corner_lane_keys)
        if len(unique_lanes) > 1:
            multiple_lanes[i] = True
        try:
            center_wp = carla_map.get_waypoint(carla.Location(x=sample["x"], y=sample["y"], z=0.0), project_to_road=True)
        except Exception:
            center_wp = None
        if center_wp is not None:
            in_intersection[i] = bool(center_wp.is_junction)
            key = _lane_key(center_wp)
            if key is not None and key not in route_lane_ids:
                oncoming[i] = True

    return multiple_lanes, non_drivable, oncoming, in_intersection


def _compute_driving_direction_compliance(centers, oncoming_mask, intersection_mask, dt):
    if len(centers) < 2:
        return 1.0
    centers = np.array(centers, dtype=np.float64)
    progress = np.zeros(len(centers), dtype=np.float64)
    progress[1:] = np.linalg.norm(centers[1:] - centers[:-1], axis=-1)
    progress[~oncoming_mask] = 0.0
    progress[intersection_mask] = 0.0

    horizon_steps = int(DRIVING_DIRECTION_HORIZON / max(dt, 1e-3))
    if horizon_steps <= 0:
        horizon_steps = 1
    windowed = []
    for t in range(len(progress)):
        start = max(0, t - horizon_steps)
        windowed.append(progress[start : t + 1].sum())
    max_progress = max(windowed) if windowed else 0.0
    if max_progress < DRIVING_DIRECTION_COMPLIANCE_THRESHOLD:
        return 1.0
    if max_progress < DRIVING_DIRECTION_VIOLATION_THRESHOLD:
        return 0.5
    return 0.0


def _compute_lane_keeping(route_line, centers, intersection_mask, dt):
    if route_line is None or LineString is None or len(centers) < 2:
        return 1.0
    continuous_steps_required = int(math.ceil(LANE_KEEPING_HORIZON_WINDOW / max(dt, 1e-3)))
    consecutive_exceeds = 0
    for idx, center in enumerate(centers):
        if intersection_mask[idx]:
            continue
        pt = Point(center[0], center[1])
        lateral_deviation = pt.distance(route_line)
        if lateral_deviation > LANE_KEEPING_DEVIATION_LIMIT:
            consecutive_exceeds += 1
        else:
            consecutive_exceeds = 0
        if consecutive_exceeds >= continuous_steps_required:
            return 0.0
    return 1.0


def _compute_traffic_light_compliance(ego_polys, world_trace, tl_polygons):
    if Polygon is None or not world_trace or not tl_polygons or carla is None:
        return 1.0
    red_val = int(carla.TrafficLightState.Red)
    for t, ego_poly in enumerate(ego_polys):
        if ego_poly is None:
            continue
        tl_states = world_trace[t].get("traffic_lights", {}) if t < len(world_trace) else {}
        for tl_id, state in tl_states.items():
            if int(state) != red_val:
                continue
            coords = tl_polygons.get(tl_id)
            if not coords:
                continue
            tl_poly = Polygon(coords)
            if ego_poly.intersects(tl_poly):
                return 0.0
    return 1.0


def _compute_no_at_fault_collision(ego_states, ego_polys, ego_front_edges, world_trace, multiple_lanes, non_drivable):
    if Polygon is None or not world_trace:
        return 1.0, np.inf
    no_at_fault = 1.0
    collision_time_idx = np.inf
    collided_ids = set()
    for t in range(len(ego_polys)):
        ego_poly = ego_polys[t]
        if ego_poly is None:
            continue
        for actor in world_trace[t].get("actors", []):
            if actor["id"] in collided_ids:
                continue
            actor_poly, _ = _oriented_box_with_front(
                actor["x"], actor["y"], actor["yaw"], actor["extent_x"], actor["extent_y"]
            )
            if actor_poly is None or not ego_poly.intersects(actor_poly):
                continue
            # classify collision
            ego_speed = math.hypot(ego_states[t, StateIndex._VELOCITY_X], ego_states[t, StateIndex._VELOCITY_Y])
            actor_speed = math.hypot(actor["vel_x"], actor["vel_y"])
            if ego_speed <= STOPPED_SPEED_THRESHOLD:
                collision_type = "STOPPED_EGO"
            elif actor_speed <= STOPPED_SPEED_THRESHOLD:
                collision_type = "STOPPED_TRACK"
            else:
                # check if actor is behind ego
                rel_x = actor["x"] - ego_states[t, StateIndex._X]
                rel_y = actor["y"] - ego_states[t, StateIndex._Y]
                forward = (math.cos(ego_states[t, StateIndex._HEADING]), math.sin(ego_states[t, StateIndex._HEADING]))
                if rel_x * forward[0] + rel_y * forward[1] < 0:
                    collision_type = "ACTIVE_REAR"
                else:
                    # front edge intersection
                    front_edge = ego_front_edges[t]
                    if front_edge and LineString is not None and LineString([front_edge[0], front_edge[1]]).intersects(actor_poly):
                        collision_type = "ACTIVE_FRONT"
                    else:
                        collision_type = "ACTIVE_LATERAL"

            collisions_at_stopped_or_front = collision_type in ["STOPPED_TRACK", "ACTIVE_FRONT"]
            collision_at_lateral = collision_type == "ACTIVE_LATERAL"
            ego_bad_area = bool(multiple_lanes[t] or non_drivable[t])
            if collisions_at_stopped_or_front or (ego_bad_area and collision_at_lateral):
                no_at_fault = 0.0
                collision_time_idx = min(collision_time_idx, t)
            else:
                collided_ids.add(actor["id"])
    return no_at_fault, collision_time_idx


def _compute_ttc(ego_states, ego_polys, world_trace, multiple_lanes, non_drivable, in_intersection, dt, ego_actor_id=None):
    """
    CARLA equivalent of navsim TTC:
    - Project ego footprint forward.
    - Query environment at future time index (world_trace).
    - Exclude the current ego vehicle but include OTHER ego vehicles.
    """
    if Polygon is None or not world_trace:
        if _DEBUG_PDM:
            print(f"[DEBUG PDM] Early return: Polygon={Polygon is not None}, world_trace_len={len(world_trace) if world_trace else 0}")
        return 1.0, np.inf, None, None
    ttc_score = 1.0
    ttc_time_idx = np.inf
    min_ttc_time_s = np.inf
    min_ttc_distance_m = np.inf
    temp_collided_ids = set()

    # navsim uses 0.0, 0.3, 0.6, 0.9s for 1s horizon (step of 3 at 10Hz)
    future_times = np.arange(0.0, FUTURE_COLLISION_HORIZON_WINDOW + 1e-6, 0.3)
    n = min(len(ego_polys), len(world_trace))
    
    if _DEBUG_PDM:
        print(f"[DEBUG PDM] Starting TTC computation: n={n}, len(world_trace)={len(world_trace)}, ego_actor_id={ego_actor_id}")
    total_actors_checked = 0
    timesteps_with_actors = 0

    for t in range(n):
        ego_speed = math.hypot(ego_states[t, StateIndex._VELOCITY_X], ego_states[t, StateIndex._VELOCITY_Y])
        ego_poly = ego_polys[t]
        if ego_poly is None:
            continue

        heading = ego_states[t, StateIndex._HEADING]
        forward = (math.cos(heading), math.sin(heading))
        ego_bad_area = bool(multiple_lanes[t] or non_drivable[t] or in_intersection[t])
        
        # Skip TTC scoring if stopped, but still track min distances for diagnostics
        ego_moving = ego_speed >= STOPPED_SPEED_THRESHOLD

        for future_time in future_times:
            offset = int(round(future_time / max(dt, 1e-3)))
            current_time_idx = t + offset
            if current_time_idx >= n:
                continue

            # Use actual future ego position from trajectory instead of straight-line projection
            if current_time_idx < len(ego_polys) and ego_polys[current_time_idx] is not None:
                projected_ego = ego_polys[current_time_idx]
            else:
                # Fallback to straight-line projection if future position not available
                dx = forward[0] * ego_speed * future_time
                dy = forward[1] * ego_speed * future_time
                projected_ego = Polygon([(x + dx, y + dy) for x, y in ego_poly.exterior.coords])

            actors = world_trace[current_time_idx].get("actors", [])
            if _DEBUG_PDM and len(actors) > 0 and t == 0 and future_time == 0.0:
                timesteps_with_actors += 1
                print(f"[DEBUG PDM] t={t}, future_time={future_time}, actors={len(actors)}, ego_speed={ego_speed:.3f}, ego_moving={ego_moving}")
            
            for actor in actors:
                # Skip the current ego vehicle but allow other ego vehicles
                if ego_actor_id is not None and actor["id"] == ego_actor_id:
                    continue
                if actor["id"] in temp_collided_ids:
                    continue
                actor_poly, _ = _oriented_box_with_front(
                    actor["x"], actor["y"], actor["yaw"], actor["extent_x"], actor["extent_y"]
                )
                if actor_poly is None:
                    continue
                dist = projected_ego.distance(actor_poly)
                total_actors_checked += 1
                
                # Calculate actual TTC based on relative velocity
                ego_vel = np.array([ego_states[t, StateIndex._VELOCITY_X], ego_states[t, StateIndex._VELOCITY_Y]])
                actor_vel = np.array([actor["vel_x"], actor["vel_y"]])
                relative_vel = ego_vel - actor_vel
                
                # Vector from actor to ego
                actor_to_ego = np.array([ego_states[t, StateIndex._X] - actor["x"], 
                                        ego_states[t, StateIndex._Y] - actor["y"]])
                
                # Closing speed (negative dot product means approaching)
                closing_speed = -np.dot(relative_vel, actor_to_ego / (np.linalg.norm(actor_to_ego) + 1e-6))
                
                # Calculate TTC: only meaningful if vehicles are closing (closing_speed > threshold)
                if closing_speed > 0.1:  # Only consider if closing faster than 0.1 m/s
                    ttc = dist / closing_speed
                else:
                    ttc = np.inf
                
                if _DEBUG_PDM and t == 0 and future_time == 0.0 and total_actors_checked <= 3:
                    min_dist_str = f"{min_ttc_distance_m:.3f}" if min_ttc_distance_m != np.inf else "inf"
                    print(f"[DEBUG PDM] Actor check: dist={dist:.3f}, closing_speed={closing_speed:.3f} m/s, ttc={ttc:.3f}s, min_ttc_distance_m={min_dist_str}")
                
                # Track minimum TTC
                if ttc < min_ttc_time_s:
                    min_ttc_time_s = ttc
                    min_ttc_distance_m = dist
                    if _DEBUG_PDM and total_actors_checked <= 3:
                        print(f"[DEBUG PDM] Updated min_ttc_time_s={min_ttc_time_s:.3f}s, min_ttc_distance_m={min_ttc_distance_m:.3f}m")
                # Also track minimum distance independently
                elif dist < min_ttc_distance_m and ttc != np.inf:
                    min_ttc_distance_m = dist

                # Only score TTC violations if ego is moving
                if ego_moving and dist <= 0.0:
                    rel_x = actor["x"] - ego_states[t, StateIndex._X]
                    rel_y = actor["y"] - ego_states[t, StateIndex._Y]
                    ahead = rel_x * forward[0] + rel_y * forward[1] > 0
                    behind = rel_x * forward[0] + rel_y * forward[1] < 0
                    if ahead or (ego_bad_area and not behind):
                        ttc_score = 0.0
                        ttc_time_idx = min(ttc_time_idx, t)
                    else:
                        temp_collided_ids.add(actor["id"])
            if ttc_score == 0.0:
                break
        if ttc_score == 0.0:
            break

    if _DEBUG_PDM:
        print(f"[DEBUG PDM] Final: total_actors_checked={total_actors_checked}, timesteps_with_actors={timesteps_with_actors}")
        print(f"[DEBUG PDM] min_ttc_distance_m={min_ttc_distance_m if min_ttc_distance_m != np.inf else 'inf'}, min_ttc_time_s={min_ttc_time_s if min_ttc_time_s != np.inf else 'inf'}")
    
    if min_ttc_time_s == np.inf:
        min_ttc_time_s = None
    if min_ttc_distance_m == np.inf:
        min_ttc_distance_m = None
    return ttc_score, ttc_time_idx, min_ttc_time_s, min_ttc_distance_m


def compute_pdm_route_metrics(
    route_record,
    pdm_trace: Optional[List[Dict]],
    pdm_world_trace: Optional[List[Dict]] = None,
    pdm_tl_polygons: Optional[Dict] = None,
    config=None,
    ego_id: int = 0,
) -> PDMRouteMetrics:
    """
    CARLA-side PDM scoring using navsim-equivalent logic with CARLA observations.
    """
    if not pdm_trace or len(pdm_trace) < 2:
        return PDMRouteMetrics(
            no_at_fault_collisions=0.0,
            drivable_area_compliance=0.0,
            driving_direction_compliance=0.0,
            traffic_light_compliance=0.0,
            ego_progress=0.0,
            time_to_collision_within_bound=0.0,
            lane_keeping=0.0,
            history_comfort=0.0,
            weighted_metrics=[0.0] * 5,
            weighted_metrics_array=PDM_WEIGHTS.tolist(),
            multiplicative_metrics_prod=0.0,
            pdm_score=0.0,
            comfort_components={},
            available=False,
            ttc_min_time_s=None,
            ttc_min_distance_m=None,
        ).__dict__

    carla_map = CarlaDataProvider.get_map() if CarlaDataProvider else None
    route_line, route_lane_ids = _build_route_line_and_lane_ids(config, ego_id, carla_map)

    n = len(pdm_trace)
    if pdm_world_trace:
        n = min(n, len(pdm_world_trace))
    pdm_trace = pdm_trace[:n]
    pdm_world_trace = pdm_world_trace[:n] if pdm_world_trace else []
    
    if _DEBUG_PDM:
        print(f"[DEBUG PDM] compute_pdm_route_metrics called: ego_id={ego_id}, n={n}, pdm_world_trace_len={len(pdm_world_trace)}")
        if pdm_world_trace:
            actor_counts = [len(wt.get("actors", [])) for wt in pdm_world_trace[:5]]
            print(f"[DEBUG PDM] First 5 timesteps actor counts: {actor_counts}")
            total_actors = sum(len(wt.get("actors", [])) for wt in pdm_world_trace)
            print(f"[DEBUG PDM] Total actors across all timesteps: {total_actors}")
        else:
            print(f"[DEBUG PDM] pdm_world_trace is empty or None!")

    states, time_s = _trace_to_state_array(pdm_trace)
    dt = float(np.median(np.diff(time_s))) if len(time_s) > 1 else 0.1
    
    # Get the actor_id of the current ego vehicle from the trace
    ego_actor_id = pdm_trace[0].get("actor_id") if pdm_trace else None
    if _DEBUG_PDM:
        print(f"[DEBUG PDM] Ego vehicle actor_id from trace: {ego_actor_id}")

    # ego polygons / centers
    ego_polys = []
    ego_front_edges = []
    centers = []
    for sample in pdm_trace:
        extent_x = sample.get("extent_x", 1.0)
        extent_y = sample.get("extent_y", 0.5)
        poly, front_edge = _oriented_box_with_front(
            sample["x"], sample["y"], sample["yaw"], extent_x, extent_y
        )
        ego_polys.append(poly)
        ego_front_edges.append(front_edge)
        centers.append((sample["x"], sample["y"]))

    # area masks
    multiple_lanes, non_drivable, oncoming, in_intersection = _compute_ego_areas(
        pdm_trace, carla_map, route_lane_ids
    )

    # multiplicative metrics
    no_at_fault_collision, _ = _compute_no_at_fault_collision(
        states[0], ego_polys, ego_front_edges, pdm_world_trace, multiple_lanes, non_drivable
    )
    drivable_area_compliance = 0.0 if non_drivable.any() else 1.0
    traffic_light_compliance = _compute_traffic_light_compliance(
        ego_polys, pdm_world_trace, pdm_tl_polygons or {}
    )
    driving_direction_compliance = _compute_driving_direction_compliance(
        centers, oncoming, in_intersection, dt
    )

    # weighted metrics
    progress_raw = 0.0
    if route_line is not None and LineString is not None and len(centers) >= 2:
        start = Point(centers[0][0], centers[0][1])
        end = Point(centers[-1][0], centers[-1][1])
        progress_raw = float(route_line.project(end) - route_line.project(start))
    progress_raw = max(progress_raw, 0.0)

    # normalize progress using a reference proposal along route (closest navsim behavior)
    max_progress = progress_raw
    if route_line is not None and LineString is not None:
        max_progress = max(progress_raw, float(route_line.length))
    if max_progress > PROGRESS_DISTANCE_THRESHOLD:
        ego_progress = float(np.clip(progress_raw / max_progress, 0.0, 1.0))
    else:
        ego_progress = 1.0

    time_to_collision_within_bound, _, min_ttc_time_s, min_ttc_distance_m = _compute_ttc(
        states[0], ego_polys, pdm_world_trace, multiple_lanes, non_drivable, in_intersection, dt, ego_actor_id
    )
    if _DEBUG_PDM:
        print(f"[DEBUG PDM] After _compute_ttc: ttc_score={time_to_collision_within_bound}, min_ttc_time_s={min_ttc_time_s}, min_ttc_distance_m={min_ttc_distance_m}")
    lane_keeping = _compute_lane_keeping(route_line, centers, in_intersection, dt)

    # comfort
    comfort_components_matrix = ego_is_comfortable(states, time_s)
    comfort_components_names = [
        "lon_acceleration",
        "lat_acceleration",
        "jerk",
        "lon_jerk",
        "yaw_acceleration",
        "yaw_rate",
    ]
    comfort_components: Dict[str, float] = {}
    for idx, name in enumerate(comfort_components_names):
        comfort_components[name] = float(bool(comfort_components_matrix[0, idx]))
    history_comfort = float(bool(comfort_components_matrix.all(axis=-1)[0]))

    weighted_metrics = np.array(
        [ego_progress, time_to_collision_within_bound, lane_keeping, history_comfort, 1.0],
        dtype=np.float64,
    )
    mask = np.ones_like(PDM_WEIGHTS, dtype=bool)
    mask[4] = False
    weighted_score = float((weighted_metrics[mask] * PDM_WEIGHTS[mask]).sum() / PDM_WEIGHTS[mask].sum())

    multiplicative_metrics_prod = (
        no_at_fault_collision * drivable_area_compliance * driving_direction_compliance * traffic_light_compliance
    )
    pdm_score = float(weighted_score * multiplicative_metrics_prod)

    return PDMRouteMetrics(
        no_at_fault_collisions=no_at_fault_collision,
        drivable_area_compliance=drivable_area_compliance,
        driving_direction_compliance=driving_direction_compliance,
        traffic_light_compliance=traffic_light_compliance,
        ego_progress=ego_progress,
        time_to_collision_within_bound=time_to_collision_within_bound,
        comfort_components=comfort_components,
        available=True,
        weighted_metrics=weighted_metrics.tolist(),
        weighted_metrics_array=PDM_WEIGHTS.tolist(),
        multiplicative_metrics_prod=multiplicative_metrics_prod,
        pdm_score=pdm_score,
        # near-collision diagnostic (not used in aggregation)
        ttc_min_time_s=min_ttc_time_s,
        ttc_min_distance_m=min_ttc_distance_m,
        lane_keeping=lane_keeping,
        history_comfort=history_comfort,
    ).__dict__
