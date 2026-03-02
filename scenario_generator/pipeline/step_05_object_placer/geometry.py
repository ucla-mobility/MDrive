import math
from typing import Any, Dict, Optional, Tuple

try:
    import numpy as np
except Exception as e:
    raise RuntimeError("This script requires numpy") from e


def wrap180(deg: float) -> float:
    return ((float(deg) + 180.0) % 360.0) - 180.0


def heading_deg_from_vec(v: np.ndarray) -> float:
    return float(math.degrees(math.atan2(float(v[1]), float(v[0]))))


def cumulative_dist(points_xy: np.ndarray) -> np.ndarray:
    if len(points_xy) < 2:
        return np.array([0.0], dtype=float)
    seg = np.linalg.norm(points_xy[1:] - points_xy[:-1], axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)], axis=0)


def point_and_tangent_at_s(points_xy: np.ndarray, s_along: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (point, unit_tangent) at fractional arc-length s_along ∈ [0,1].
    Uses piecewise-linear interpolation on the polyline.
    """
    pts = np.asarray(points_xy, dtype=float)
    s = float(min(1.0, max(0.0, s_along)))
    if len(pts) == 1:
        return pts[0].copy(), np.array([1.0, 0.0], dtype=float)

    cum = cumulative_dist(pts)
    total = float(cum[-1])
    if total < 1e-9:
        # Degenerate
        v = pts[-1] - pts[0]
        n = float(np.linalg.norm(v))
        t = v / n if n > 1e-9 else np.array([1.0, 0.0], dtype=float)
        return pts[0].copy(), t

    target = s * total
    idx = int(np.searchsorted(cum, target, side="right") - 1)
    idx = max(0, min(idx, len(pts) - 2))

    a = pts[idx]
    b = pts[idx + 1]
    seg_len = float(np.linalg.norm(b - a))
    if seg_len < 1e-9:
        # Find a non-degenerate neighbor for tangent
        j = idx
        while j + 1 < len(pts) and float(np.linalg.norm(pts[j + 1] - pts[j])) < 1e-9:
            j += 1
        if j + 1 < len(pts):
            v = pts[j + 1] - pts[j]
        else:
            v = pts[-1] - pts[0]
        n = float(np.linalg.norm(v))
        t = v / n if n > 1e-9 else np.array([1.0, 0.0], dtype=float)
        return a.copy(), t

    seg_start = float(cum[idx])
    alpha = (target - seg_start) / seg_len
    p = a + alpha * (b - a)
    t = (b - a) / seg_len
    return p, t


def _closest_point_s_m_on_polyline(points_xy: np.ndarray, point_xy: np.ndarray) -> Tuple[float, float]:
    """
    Return (min_dist_m, s_m) for the closest point on a polyline to point_xy.
    s_m is distance along the polyline from its start.
    """
    pts = np.asarray(points_xy, dtype=float)
    p = np.asarray(point_xy, dtype=float).reshape(2)
    if len(pts) < 2:
        d = float(np.linalg.norm(p - pts[0])) if len(pts) == 1 else 0.0
        return d, 0.0

    cum = cumulative_dist(pts)
    best_dist = float("inf")
    best_s_m = 0.0
    for i in range(len(pts) - 1):
        a = pts[i]
        b = pts[i + 1]
        ab = b - a
        ab_len2 = float(np.dot(ab, ab))
        if ab_len2 < 1e-12:
            continue
        t = float(np.dot(p - a, ab) / ab_len2)
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        dist = float(np.linalg.norm(p - proj))
        s_m = float(cum[i] + t * math.sqrt(ab_len2))
        if dist < best_dist:
            best_dist = dist
            best_s_m = s_m
    return best_dist, best_s_m


def _project_point_to_path_s_m(
    picked_entry: Dict[str, Any],
    seg_by_id: Dict[int, np.ndarray],
    point_xy: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """
    Project a world point onto a vehicle's path and return (dist_m, path_s_m).
    path_s_m is distance along the full path from its start.
    """
    seg_ids = picked_entry.get("signature", {}).get("segment_ids", [])
    if not isinstance(seg_ids, list) or not seg_ids:
        return None

    total_offset = 0.0
    best_dist = None
    best_path_s = None
    for seg_id_raw in seg_ids:
        try:
            seg_id = int(seg_id_raw)
        except Exception:
            continue
        pts = seg_by_id.get(seg_id)
        if pts is None or len(pts) < 2:
            continue
        dist, s_m = _closest_point_s_m_on_polyline(pts, point_xy)
        path_s = total_offset + s_m
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_path_s = path_s
        total_offset += float(cumulative_dist(pts)[-1])
    if best_dist is None or best_path_s is None:
        return None
    return best_dist, best_path_s


def right_normal_world(tangent: np.ndarray) -> np.ndarray:
    """
    Get the right-side normal vector for a given tangent direction.
    
    In CARLA's coordinate system where +x is typically west and +y is north:
    - If facing west (+x), RIGHT is north (+y)
    - If facing north (+y), RIGHT is east (-x)
    
    So right_normal = rotate tangent by +90 degrees = (-dy, dx)
    """
    dx, dy = float(tangent[0]), float(tangent[1])
    n = np.array([-dy, dx], dtype=float)
    nn = float(np.linalg.norm(n))
    return n / nn if nn > 1e-9 else np.array([0.0, 1.0], dtype=float)


def left_normal_world(tangent: np.ndarray) -> np.ndarray:
    """
    Get the left-side normal vector for a given tangent direction.
    
    In CARLA's coordinate system where +x is typically west and +y is north:
    - If facing west (+x), LEFT is south (-y)
    - If facing north (+y), LEFT is west (+x)
    
    So left_normal = rotate tangent by -90 degrees = (dy, -dx)
    """
    dx, dy = float(tangent[0]), float(tangent[1])
    n = np.array([dy, -dx], dtype=float)
    nn = float(np.linalg.norm(n))
    return n / nn if nn > 1e-9 else np.array([0.0, -1.0], dtype=float)


__all__ = [
    "_closest_point_s_m_on_polyline",
    "_project_point_to_path_s_m",
    "cumulative_dist",
    "heading_deg_from_vec",
    "left_normal_world",
    "point_and_tangent_at_s",
    "right_normal_world",
    "wrap180",
]
