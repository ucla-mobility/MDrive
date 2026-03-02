import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .geometry import unit_from_yaw_deg, wrap180
from .models import CropBox, LaneSegment


def orient_polyline(points_xy: np.ndarray, yaws_deg: np.ndarray,
                    orig_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ensure polyline direction agrees with yaw direction.
    If average dot(direction_vec, yaw_vec) < 0, reverse the polyline.
    """
    pts = np.asarray(points_xy, dtype=float)
    yaws = wrap180(np.asarray(yaws_deg, dtype=float))
    idxs = np.asarray(orig_idx, dtype=int)

    if len(pts) < 2:
        return pts, yaws, idxs

    vecs = pts[1:] - pts[:-1]
    norms = np.linalg.norm(vecs, axis=1) + 1e-9
    dir_vecs = vecs / norms[:, None]

    yaw_vecs = np.vstack([unit_from_yaw_deg(y) for y in yaws[:-1]])
    dots = np.sum(dir_vecs * yaw_vecs, axis=1)

    if float(np.nanmean(dots)) < 0.0:
        pts = pts[::-1].copy()
        yaws = yaws[::-1].copy()
        idxs = idxs[::-1].copy()

    return pts, yaws, idxs


def split_by_gaps(idxs_sorted: np.ndarray, pts: np.ndarray, yaws: np.ndarray,
                  gap_m: float = 6.0) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Split a polyline into continuous chunks at large gaps."""
    if len(pts) < 2:
        return [(idxs_sorted, pts, yaws)] if len(pts) > 0 else []

    jumps = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    cuts = [0]
    for i, d in enumerate(jumps):
        if float(d) > gap_m:
            cuts.append(i + 1)
    cuts.append(len(pts))

    chunks = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a >= 2:
            chunks.append((idxs_sorted[a:b], pts[a:b], yaws[a:b]))

    return chunks


def load_nodes(path: str) -> Dict[str, Any]:
    """Load town nodes JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    if "payload" not in data:
        raise ValueError(f"{path} does not contain a top-level 'payload' field")
    return data


def build_segments(data: Dict[str, Any], min_points: int = 6) -> List[LaneSegment]:
    """
    Build lane segments from node data.
    Groups waypoints by (road_id, lane_id, section_id) and orients them correctly.
    """
    payload = data["payload"]
    required_keys = ["x", "y", "yaw", "road_id", "lane_id", "section_id"]
    for k in required_keys:
        if k not in payload:
            raise ValueError(f"payload missing required key '{k}'")

    x = np.asarray(payload["x"], dtype=float)
    y = np.asarray(payload["y"], dtype=float)
    yaw = np.asarray(payload["yaw"], dtype=float)
    road_id = np.asarray(payload["road_id"], dtype=int)
    lane_id = np.asarray(payload["lane_id"], dtype=int)
    section_id = np.asarray(payload["section_id"], dtype=int)

    grouped: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for i in range(len(x)):
        grouped[(int(road_id[i]), int(lane_id[i]), int(section_id[i]))].append(i)

    segments: List[LaneSegment] = []
    seg_id = 0

    for (rid, lid, sid), idxs in grouped.items():
        idxs_sorted = np.asarray(sorted(idxs), dtype=int)
        pts = np.vstack([x[idxs_sorted], y[idxs_sorted]]).T
        yaws_data = yaw[idxs_sorted]

        for idxs_chunk, pts_chunk, yaws_chunk in split_by_gaps(idxs_sorted, pts, yaws_data):
            pts_o, yaws_o, idxs_o = orient_polyline(pts_chunk, yaws_chunk, idxs_chunk)
            if len(pts_o) < min_points:
                continue

            segments.append(
                LaneSegment(
                    seg_id=seg_id,
                    road_id=int(rid),
                    lane_id=int(lid),
                    section_id=int(sid),
                    points=pts_o,
                    yaws=wrap180(yaws_o),
                    orig_idx=idxs_o,
                )
            )
            seg_id += 1

    return segments


def crop_segments(segments: List[LaneSegment], crop: CropBox) -> List[LaneSegment]:
    """Filter segments that intersect with the crop box."""
    return [s for s in segments if crop.intersects_bbox(s.bbox())]


def _clip_segment_to_crop(seg: LaneSegment, crop: CropBox, margin: float = 1.0) -> Optional[LaneSegment]:
    """
    Clip a segment's points to stay within the crop region (with a small margin).
    
    Returns a new LaneSegment with only points inside the expanded crop region,
    or None if no points remain after clipping.
    """
    pts = seg.points
    yaws = seg.yaws
    
    # Find indices of points inside the crop (with margin)
    inside_mask = (
        (pts[:, 0] >= crop.xmin - margin) & (pts[:, 0] <= crop.xmax + margin) &
        (pts[:, 1] >= crop.ymin - margin) & (pts[:, 1] <= crop.ymax + margin)
    )
    
    if not np.any(inside_mask):
        return None
    
    # Find contiguous run of inside points (keep the longest contiguous section)
    # For simplicity, just keep all inside points
    inside_indices = np.where(inside_mask)[0]
    
    if len(inside_indices) < 2:
        return None
    
    # Get the contiguous range from first to last inside point
    first_inside = inside_indices[0]
    last_inside = inside_indices[-1]
    
    clipped_pts = pts[first_inside:last_inside + 1]
    clipped_yaws = yaws[first_inside:last_inside + 1]
    
    if len(clipped_pts) < 2:
        return None
    
    return LaneSegment(
        seg_id=seg.seg_id,
        road_id=seg.road_id,
        lane_id=seg.lane_id,
        section_id=seg.section_id,
        points=clipped_pts,
        yaws=clipped_yaws,
        orig_idx=seg.orig_idx[first_inside:last_inside + 1] if seg.orig_idx is not None else None,
    )


def crop_segments_t_junction(
    segments: List[LaneSegment], 
    crop: CropBox,
    junction_center: Optional[Tuple[float, float]] = None,
    max_heading_change_deg: float = 30.0,
    clip_margin: float = 5.0,
    junction_radius: float = 25.0,
) -> List[LaneSegment]:
    """
    Filter and CLIP segments for T-junction scenarios.
    
    For T-junctions:
    1. Segments must pass near the junction center (within junction_radius)
    2. Segments fully inside the crop are included as-is
    3. Segments partially outside are included ONLY if straight, AND they are 
       clipped to the crop boundary (with margin) so they don't extend to other junctions
    4. Curved segments that extend outside the crop are excluded (likely from other junctions)
    
    Args:
        segments: List of lane segments to filter
        crop: The crop box defining the region of interest
        junction_center: (x, y) center of the junction. If provided, segments that
                        don't pass near this point are excluded.
        max_heading_change_deg: Maximum allowed heading change for "straight" segments
        clip_margin: Margin beyond crop boundary to keep points (for connectivity)
        junction_radius: Maximum distance from junction center for a segment to be included.
                        A segment is included if ANY of its points are within this radius.
        
    Returns:
        Filtered and clipped list of segments
    """
    
    def segment_passes_near_junction(seg: LaneSegment) -> bool:
        """Check if any point of the segment is within junction_radius of center."""
        if junction_center is None:
            return True
        cx, cy = junction_center
        for pt in seg.points:
            dist = np.sqrt((pt[0] - cx)**2 + (pt[1] - cy)**2)
            if dist <= junction_radius:
                return True
        return False
    
    result = []
    for seg in segments:
        if not crop.intersects_bbox(seg.bbox()):
            # Segment doesn't intersect at all - skip
            continue
        
        # Check if segment passes near the junction center
        if not segment_passes_near_junction(seg):
            # Segment is from another junction - skip
            continue
        
        # Check if segment is fully inside crop
        pts = seg.points
        start_inside = crop.contains(pts[0])
        end_inside = crop.contains(pts[-1])
        
        if start_inside and end_inside:
            # Fully inside - always include as-is
            result.append(seg)
        else:
            # Partially outside - only include if straight, and CLIP it
            if seg.is_straight(max_heading_change_deg):
                clipped = _clip_segment_to_crop(seg, crop, margin=clip_margin)
                if clipped is not None:
                    result.append(clipped)
            # else: skip curved segments that extend outside crop
    
    return result
