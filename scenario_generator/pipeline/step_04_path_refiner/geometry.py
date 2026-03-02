import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class CropBox:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def contains(self, x: float, y: float) -> bool:
        return (self.xmin <= x <= self.xmax) and (self.ymin <= y <= self.ymax)


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.hypot(dx, dy)


def _polyline_cumdist(pts: Sequence[Tuple[float, float]]) -> List[float]:
    out = [0.0]
    for i in range(1, len(pts)):
        out.append(out[-1] + _dist(pts[i - 1], pts[i]))
    return out


def _heading_deg(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    # world frame heading: atan2(dy, dx) in degrees
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def _unit(v: Tuple[float, float]) -> Tuple[float, float]:
    n = math.hypot(v[0], v[1])
    if n < 1e-9:
        return (1.0, 0.0)
    return (v[0] / n, v[1] / n)


def _dot(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _cross2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _closest_point_on_segment(
    p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]
) -> Tuple[Tuple[float, float], float]:
    # Returns (closest_point, t in [0,1])
    ax, ay = a
    bx, by = b
    px, py = p
    vx, vy = (bx - ax), (by - ay)
    denom = vx * vx + vy * vy
    if denom < 1e-9:
        return a, 0.0
    t = ((px - ax) * vx + (py - ay) * vy) / denom
    t = max(0.0, min(1.0, t))
    cp = (ax + t * vx, ay + t * vy)
    return cp, t


def _segment_segment_closest(
    a0: Tuple[float, float],
    a1: Tuple[float, float],
    b0: Tuple[float, float],
    b1: Tuple[float, float],
) -> Tuple[float, Tuple[float, float], Tuple[float, float], float, float]:
    """
    Approx closest points between segments [a0,a1] and [b0,b1].

    Returns:
      (min_dist, pa, pb, ta, tb)
      where pa = a0 + ta*(a1-a0), pb = b0 + tb*(b1-b0), ta,tb in [0,1]
    """
    # Small, robust approximation: check endpoints projected onto the other segment.
    candidates: List[Tuple[float, Tuple[float, float], Tuple[float, float], float, float]] = []

    pb, tb = _closest_point_on_segment(a0, b0, b1)
    candidates.append((_dist(a0, pb), a0, pb, 0.0, tb))

    pb, tb = _closest_point_on_segment(a1, b0, b1)
    candidates.append((_dist(a1, pb), a1, pb, 1.0, tb))

    pa, ta = _closest_point_on_segment(b0, a0, a1)
    candidates.append((_dist(pa, b0), pa, b0, ta, 0.0))

    pa, ta = _closest_point_on_segment(b1, a0, a1)
    candidates.append((_dist(pa, b1), pa, b1, ta, 1.0))

    # Pick best
    return min(candidates, key=lambda x: x[0])


def find_conflict_between_polylines(
    p1: Sequence[Tuple[float, float]],
    p2: Sequence[Tuple[float, float]],
    dist_thresh_m: float = 3.0,
    min_angle_deg: float = 12.0,
) -> Optional[Dict[str, Any]]:
    """
    Returns an approximate conflict if within threshold.

    First-encounter behavior:
    - Among all near-approaches within dist_thresh_m, pick the earliest encounter along the
      polylines (minimize max(s1, s2)), ignoring trivially parallel segment pairs.
    """
    if len(p1) < 2 or len(p2) < 2:
        return None

    c1 = _polyline_cumdist(p1)
    c2 = _polyline_cumdist(p2)

    # Collect near-approach candidates that are not trivially parallel
    candidates: List[Tuple[float, float, float, int, int, Tuple[float, float], Tuple[float, float], float, float]] = []
    # (s1, s2, d, i, j, pa, pb, ta, tb)

    for i in range(len(p1) - 1):
        a0, a1 = p1[i], p1[i + 1]
        va = (a1[0] - a0[0], a1[1] - a0[1])
        na = math.hypot(va[0], va[1])
        for j in range(len(p2) - 1):
            b0, b1 = p2[j], p2[j + 1]
            vb = (b1[0] - b0[0], b1[1] - b0[1])
            nb = math.hypot(vb[0], vb[1])

            d, pa, pb, ta, tb = _segment_segment_closest(a0, a1, b0, b1)
            if d > dist_thresh_m:
                continue

            ok = True
            if na > 1e-6 and nb > 1e-6:
                cosang = max(-1.0, min(1.0, (va[0] * vb[0] + va[1] * vb[1]) / (na * nb)))
                ang = math.degrees(math.acos(cosang))
                if ang < float(min_angle_deg):
                    ok = False
            if not ok:
                continue

            # Arc-length along each at the closest point
            s1 = c1[i] + ta * _dist(p1[i], p1[i + 1])
            s2 = c2[j] + tb * _dist(p2[j], p2[j + 1])
            candidates.append((s1, s2, d, i, j, pa, pb, ta, tb))

    if not candidates:
        return None

    # First encounter: minimize max(s1, s2); tie-break by smaller distance
    s1, s2, d, i, j, pa, pb, ta, tb = min(candidates, key=lambda t: (max(t[0], t[1]), t[2]))

    conflict = ((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0)
    return {
        "dist_m": float(d),
        "point": {"x": float(conflict[0]), "y": float(conflict[1])},
        "s_along": {"p1_m": float(s1), "p2_m": float(s2)},
        "seg_index": {"p1": int(i), "p2": int(j)},
        "param": {"p1": float(ta), "p2": float(tb)},
    }


def _segments_to_polyline_with_map(segments_detailed: List[Dict[str, Any]]) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int]]]:
    """
    Concatenate per-segment polyline_sample into one polyline.
    Returns:
      points: [(x,y), ...]
      mapping: [(seg_i, local_pt_i), ...] for each point in points
    """
    pts: List[Tuple[float, float]] = []
    mp: List[Tuple[int, int]] = []
    for si, seg in enumerate(segments_detailed):
        pl = seg.get("polyline_sample") or []
        local: List[Tuple[float, float]] = [(float(p["x"]), float(p["y"])) for p in pl if "x" in p and "y" in p]
        if not local:
            continue
        for li, pxy in enumerate(local):
            # Avoid duplicating the seam point
            if pts and _dist(pts[-1], pxy) < 1e-6:
                continue
            pts.append(pxy)
            mp.append((si, li))
    return pts, mp


def _first_last_idx_in_crop(pts: Sequence[Tuple[float, float]], crop: CropBox) -> Optional[Tuple[int, int]]:
    inside = [i for i, (x, y) in enumerate(pts) if crop.contains(x, y)]
    if not inside:
        return None
    return inside[0], inside[-1]


def _polyline_slice(pts: Sequence[Tuple[float, float]], start_i: int, end_i: int) -> List[Tuple[float, float]]:
    start_i = max(0, min(start_i, len(pts) - 1))
    end_i = max(0, min(end_i, len(pts) - 1))
    if end_i <= start_i:
        end_i = min(len(pts) - 1, start_i + 1)
    return [tuple(map(float, pts[i])) for i in range(start_i, end_i + 1)]


def _polyline_bbox(pts: Sequence[Tuple[float, float]]) -> Dict[str, float]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return {"xmin": float(min(xs)), "xmax": float(max(xs)), "ymin": float(min(ys)), "ymax": float(max(ys))}


def _polyline_length(pts: Sequence[Tuple[float, float]]) -> float:
    if len(pts) < 2:
        return 0.0
    return _polyline_cumdist(pts)[-1]


def _build_segment_payload_from_polyline(
    pts: Sequence[Tuple[float, float]],
    seg_template: Optional[Dict[str, Any]] = None,
    seg_id: Optional[int] = None,
) -> Dict[str, Any]:
    if len(pts) < 2:
        raise ValueError("segment polyline must have >=2 points")

    hs = _heading_deg(pts[0], pts[1])
    he = _heading_deg(pts[-2], pts[-1])
    out: Dict[str, Any] = {}
    if seg_template:
        for k in ["road_id", "section_id", "lane_id"]:
            if k in seg_template:
                out[k] = seg_template[k]
    out["seg_id"] = int(seg_id if seg_id is not None else (seg_template.get("seg_id") if seg_template else 0))
    out["length_m"] = float(_polyline_length(pts))
    out["bbox"] = _polyline_bbox(pts)
    out["start"] = {"point": {"x": float(pts[0][0]), "y": float(pts[0][1])}, "heading_deg": float(hs)}
    out["end"] = {"point": {"x": float(pts[-1][0]), "y": float(pts[-1][1])}, "heading_deg": float(he)}
    out["polyline_sample"] = [{"x": float(x), "y": float(y)} for (x, y) in pts]
    return out


def _slice_segments_detailed(
    segments_detailed: List[Dict[str, Any]],
    start_idx_global: int,
    end_idx_global: int,
) -> List[Dict[str, Any]]:
    """
    Slice original segments_detailed based on global point indices from the concatenated polyline.
    Keeps as much original segment structure as possible.
    """
    pts, mp = _segments_to_polyline_with_map(segments_detailed)
    if not pts or not mp:
        return []

    start_idx_global = max(0, min(start_idx_global, len(pts) - 1))
    end_idx_global = max(0, min(end_idx_global, len(pts) - 1))
    if end_idx_global <= start_idx_global:
        end_idx_global = min(len(pts) - 1, start_idx_global + 1)

    # Determine which segment/local indices are included
    included = list(range(start_idx_global, end_idx_global + 1))

    seg_points: Dict[int, List[Tuple[float, float]]] = {}
    for gi in included:
        si, _li = mp[gi]
        seg_points.setdefault(si, []).append(pts[gi])

    out: List[Dict[str, Any]] = []
    for si in sorted(seg_points.keys()):
        pl = seg_points[si]
        if len(pl) < 2:
            continue
        templ = segments_detailed[si]
        out.append(_build_segment_payload_from_polyline(pl, seg_template=templ, seg_id=int(templ.get("seg_id", si))))
    return out


def _resample_polyline(pts: Sequence[Tuple[float, float]], n: int) -> List[Tuple[float, float]]:
    if n <= 2 or len(pts) <= 2:
        return list(pts)
    # sample by arc-length
    cum = _polyline_cumdist(pts)
    total = cum[-1]
    if total < 1e-6:
        return [pts[0]] * n
    targets = [total * (i / (n - 1)) for i in range(n)]
    out = []
    j = 0
    for t in targets:
        while j + 1 < len(cum) and cum[j + 1] < t:
            j += 1
        if j + 1 >= len(cum):
            out.append(pts[-1])
        else:
            t0, t1 = cum[j], cum[j + 1]
            alpha = 0.0 if abs(t1 - t0) < 1e-9 else (t - t0) / (t1 - t0)
            x = pts[j][0] + alpha * (pts[j + 1][0] - pts[j][0])
            y = pts[j][1] + alpha * (pts[j + 1][1] - pts[j][1])
            out.append((x, y))
    return out


def _find_closest_idx(pts: Sequence[Tuple[float, float]], p: Tuple[float, float]) -> int:
    best_i = 0
    best_d = float("inf")
    for i, q in enumerate(pts):
        d = _dist(q, p)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _forward_axis_at(pts: Sequence[Tuple[float, float]], idx: int) -> Tuple[float, float]:
    j = min(len(pts) - 1, max(0, idx))
    k = min(len(pts) - 1, j + 1)
    if k == j and j - 1 >= 0:
        k = j
        j = j - 1
    return _unit((pts[k][0] - pts[j][0], pts[k][1] - pts[j][1]))


def split_corridor_segment_detailed(
    seg: Dict[str, Any],
    target_length_m: float = 25.0,
    min_splits: int = 3,
) -> List[Dict[str, Any]]:
    """
    Split a single long corridor segment_detailed into multiple virtual sub-segments.
    
    This is used for TWO_LANE_CORRIDOR topologies where the path typically has only
    1 segment (since it's a straight road), but the object placer needs multiple
    segments to place objects at different positions along the path.
    
    Args:
        seg: A single segment_detailed dict with polyline_sample, length_m, etc.
        target_length_m: Target length for each sub-segment (default 25m)
        min_splits: Minimum number of sub-segments to create (default 3)
        
    Returns:
        List of segment_detailed dicts representing the split sub-segments.
        Each sub-segment retains the same seg_id, road_id, lane_id but with
        adjusted length_m, start/end points, and polyline_sample.
    """
    length_m = float(seg.get("length_m", 0.0))
    if length_m < 1e-6:
        return [seg]
    
    # Determine number of splits
    n_splits = max(min_splits, int(math.ceil(length_m / target_length_m)))
    if n_splits <= 1:
        return [seg]
    
    # Get the polyline sample points
    polyline = seg.get("polyline_sample", [])
    if not polyline or len(polyline) < 2:
        return [seg]
    
    pts = [(float(p["x"]), float(p["y"])) for p in polyline if "x" in p and "y" in p]
    if len(pts) < 2:
        return [seg]
    
    # Compute cumulative distances
    cumdist = _polyline_cumdist(pts)
    total_length = cumdist[-1]
    if total_length < 1e-6:
        return [seg]
    
    # Split points: divide into n_splits equal segments
    split_distances = [total_length * i / n_splits for i in range(n_splits + 1)]
    
    # Find indices for each split point
    def find_idx_and_interp(target_dist: float) -> Tuple[int, float, Tuple[float, float]]:
        """Find index and interpolation for a given distance along the polyline."""
        for i in range(len(cumdist) - 1):
            if cumdist[i] <= target_dist <= cumdist[i + 1]:
                seg_len = cumdist[i + 1] - cumdist[i]
                if seg_len < 1e-9:
                    alpha = 0.0
                else:
                    alpha = (target_dist - cumdist[i]) / seg_len
                x = pts[i][0] + alpha * (pts[i + 1][0] - pts[i][0])
                y = pts[i][1] + alpha * (pts[i + 1][1] - pts[i][1])
                return i, alpha, (x, y)
        return len(pts) - 1, 0.0, pts[-1]
    
    # Build sub-segments
    sub_segments: List[Dict[str, Any]] = []
    base_seg_id = seg.get("seg_id", 0)
    
    for split_idx in range(n_splits):
        start_dist = split_distances[split_idx]
        end_dist = split_distances[split_idx + 1]
        sub_length = end_dist - start_dist
        
        _, _, start_pt = find_idx_and_interp(start_dist)
        _, _, end_pt = find_idx_and_interp(end_dist)
        
        # Compute heading from start to end
        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]
        heading = math.degrees(math.atan2(dy, dx)) if (abs(dx) > 1e-6 or abs(dy) > 1e-6) else 0.0
        
        # Cardinal direction from heading
        if -45 <= heading < 45:
            cardinal = "E"
        elif 45 <= heading < 135:
            cardinal = "N"
        elif -135 <= heading < -45:
            cardinal = "S"
        else:
            cardinal = "W"
        bound = {"E": "eastbound", "W": "westbound", "N": "northbound", "S": "southbound"}.get(cardinal, "forward")
        
        # Create polyline sample for this sub-segment (just start and end points)
        sub_polyline = [
            {"x": start_pt[0], "y": start_pt[1]},
            {"x": end_pt[0], "y": end_pt[1]},
        ]
        
        # Build the sub-segment dict
        sub_seg = {
            "seg_id": int(base_seg_id),  # Keep same seg_id (virtual split)
            "virtual_split_index": split_idx,  # Mark as virtual sub-segment
            "road_id": seg.get("road_id"),
            "section_id": seg.get("section_id"),
            "lane_id": seg.get("lane_id"),
            "length_m": float(sub_length),
            "bbox": seg.get("bbox"),  # Keep original bbox
            "start": {
                "point": {"x": float(start_pt[0]), "y": float(start_pt[1])},
                "heading_deg": float(heading),
                "cardinal4": cardinal,
                "bound": bound,
                "orig_idx": seg.get("start", {}).get("orig_idx") if split_idx == 0 else None,
            },
            "end": {
                "point": {"x": float(end_pt[0]), "y": float(end_pt[1])},
                "heading_deg": float(heading),
                "cardinal4": cardinal,
                "bound": bound,
                "orig_idx": seg.get("end", {}).get("orig_idx") if split_idx == n_splits - 1 else None,
            },
            "polyline_sample": sub_polyline,
        }
        sub_segments.append(sub_seg)
    
    return sub_segments


def split_corridor_path_segments(
    picked_entry: Dict[str, Any],
    target_length_m: float = 25.0,
    min_splits: int = 3,
) -> Dict[str, Any]:
    """
    Process a picked path entry and split its corridor segments into sub-segments.
    
    This should be applied to TWO_LANE_CORRIDOR topologies after path picking
    but before object placement, so that the LLM has multiple segment_index values
    to choose from when placing objects.
    
    Args:
        picked_entry: A single entry from the "picked" list in picked_paths.json
        target_length_m: Target length for each sub-segment
        min_splits: Minimum number of sub-segments to create
        
    Returns:
        Modified picked_entry with split segments_detailed
    """
    sig = picked_entry.get("signature")
    if not isinstance(sig, dict):
        return picked_entry
    
    segments_detailed = sig.get("segments_detailed", [])
    if not isinstance(segments_detailed, list) or not segments_detailed:
        return picked_entry
    
    # Only split if we have few segments (corridor-like)
    # Intersections/T-junctions typically have 3+ segments
    if len(segments_detailed) > 2:
        return picked_entry
    
    # Split each segment
    new_segments: List[Dict[str, Any]] = []
    for seg in segments_detailed:
        length_m = float(seg.get("length_m", 0.0))
        # Only split if segment is long enough
        if length_m >= target_length_m * 1.5:
            sub_segs = split_corridor_segment_detailed(seg, target_length_m, min_splits)
            new_segments.extend(sub_segs)
        else:
            new_segments.append(seg)
    
    # Update the signature
    new_sig = dict(sig)
    new_sig["segments_detailed"] = new_segments
    new_sig["num_segments"] = len(new_segments)
    # segment_ids stays the same (virtual splits share the same underlying seg_id)
    
    new_entry = dict(picked_entry)
    new_entry["signature"] = new_sig
    new_entry["corridor_split_applied"] = True
    
    return new_entry


__all__ = [
    "CropBox",
    "_build_segment_payload_from_polyline",
    "_closest_point_on_segment",
    "_cross2",
    "_dist",
    "_dot",
    "_find_closest_idx",
    "_first_last_idx_in_crop",
    "_forward_axis_at",
    "_heading_deg",
    "_polyline_bbox",
    "_polyline_cumdist",
    "_polyline_length",
    "_polyline_slice",
    "_resample_polyline",
    "_segment_segment_closest",
    "_segments_to_polyline_with_map",
    "_slice_segments_detailed",
    "_unit",
    "find_conflict_between_polylines",
    "split_corridor_path_segments",
    "split_corridor_segment_detailed",
]
