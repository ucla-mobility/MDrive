from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .geometry import cumulative_dist, heading_deg_from_vec, wrap180
from .utils import _parse_vehicle_num


def _heading_at_end(pts: np.ndarray, k: int = 6) -> float:
    """Compute heading at end of polyline using last k+1 points."""
    k = min(k, len(pts) - 1)
    if k == 0:
        return 0.0
    v = pts[-1] - pts[-(k + 1)]
    n = np.linalg.norm(v)
    if n < 1e-6:
        return 0.0
    return heading_deg_from_vec(v)


def _heading_at_start(pts: np.ndarray, k: int = 6) -> float:
    """Compute heading at start of polyline using first k+1 points."""
    k = min(k, len(pts) - 1)
    if k == 0:
        return 0.0
    v = pts[k] - pts[0]
    n = np.linalg.norm(v)
    if n < 1e-6:
        return 0.0
    return heading_deg_from_vec(v)


def _ang_diff_deg(a: float, b: float) -> float:
    """Absolute wrapped difference in degrees."""
    return abs(wrap180(a - b))


def _segment_length(pts: np.ndarray) -> float:
    """Compute total arc length of a polyline."""
    if len(pts) < 2:
        return 0.0
    return float(cumulative_dist(pts)[-1])


def _find_best_successor_segment(
    end_pt: np.ndarray,
    end_heading: float,
    all_segments: List[Dict[str, Any]],
    excluded_seg_ids: set,
    connect_radius_m: float = 6.0,
    connect_yaw_tol_deg: float = 45.0,  # Stricter than normal for "straight-through"
) -> Optional[Dict[str, Any]]:
    """
    Find the best successor segment for path extension.
    
    Returns the segment that:
    1. Has start point within connect_radius_m of end_pt
    2. Has start heading within connect_yaw_tol_deg of end_heading
    3. Among valid candidates, picks the one with smallest heading difference (most aligned)
    
    Returns None if no valid successor found.
    """
    best_seg = None
    best_ang_diff = float('inf')
    
    for seg in all_segments:
        seg_id = seg.get("seg_id")
        if seg_id in excluded_seg_ids:
            continue
        
        pts = seg.get("points")
        if pts is None or len(pts) < 2:
            continue
        pts = np.asarray(pts, dtype=float)
        
        # Check distance from end_pt to segment start
        start_pt = pts[0]
        dist = np.linalg.norm(start_pt - end_pt)
        if dist > connect_radius_m:
            continue
        
        # Check heading alignment
        start_heading = _heading_at_start(pts)
        ang_diff = _ang_diff_deg(end_heading, start_heading)
        if ang_diff > connect_yaw_tol_deg:
            continue
        
        # Pick the most aligned one
        if ang_diff < best_ang_diff:
            best_ang_diff = ang_diff
            best_seg = seg
    
    return best_seg


def _polyline_sample_from_pts(pts: np.ndarray, max_points: int = 12) -> List[Dict[str, float]]:
    """Convert numpy points to polyline_sample format."""
    n = len(pts)
    if n == 0:
        return []
    if n <= max_points:
        return [{"x": float(p[0]), "y": float(p[1])} for p in pts]
    idxs = np.linspace(0, n - 1, num=max_points, dtype=int)
    return [{"x": float(pts[i, 0]), "y": float(pts[i, 1])} for i in idxs]


def _compute_path_length(picked_entry: Dict[str, Any], seg_by_id: Dict[int, np.ndarray]) -> float:
    """Compute total path length for a picked vehicle path."""
    sig = picked_entry.get("signature", {})
    seg_ids = sig.get("segment_ids", [])
    total = 0.0
    for sid in seg_ids:
        pts = seg_by_id.get(int(sid))
        if pts is not None and len(pts) >= 2:
            total += _segment_length(pts)
    return total


def _estimate_required_path_length(
    actor_specs: List[Dict[str, Any]],
    picked_list: List[Dict[str, Any]],
    seg_by_id: Dict[int, np.ndarray],
) -> Dict[int, float]:
    """
    Estimate how much path length each vehicle needs based on entity relations.
    
    This is a conservative estimate used for initial path extension.
    More precise extension happens during CSP solve.
    
    Returns a dict mapping vehicle_num -> required_length_m
    """
    # Distance mapping from qualitative to meters (same as Stage2 prompt)
    DISTANCE_TO_M = {
        "touching": 2.0,
        "close": 6.0,
        "medium": 12.0,
        "far": 23.0,  # For "Twenty meters later" type phrases
    }
    
    required: Dict[int, float] = {}
    
    # Compute current path lengths
    for p in picked_list:
        veh_num = _parse_vehicle_num(p.get("vehicle"))
        if veh_num is not None:
            current_len = _compute_path_length(p, seg_by_id)
            required[veh_num] = current_len
    
    # Build relation chains and compute cumulative distance needs
    # For chains like: entity_1 -> entity_2 (20m) -> entity_3 (20m), we need base_position + 40m
    
    entity_to_veh: Dict[str, int] = {}
    for spec in actor_specs:
        eid = str(spec.get("id", ""))
        veh_num = spec.get("vehicle_num")
        if veh_num is not None:
            entity_to_veh[eid] = int(veh_num)
    
    # Build dependency graph and compute chain lengths
    # ahead_of means: this entity is ahead_of other (so this entity is further along path)
    depends_on: Dict[str, Tuple[str, float]] = {}  # entity -> (depends_on_entity, distance_m)
    for spec in actor_specs:
        eid = str(spec.get("id", ""))
        for rel in spec.get("relations", []):
            if rel.get("type") in ("ahead_of",):
                other_id = str(rel.get("other_id", ""))
                distance = rel.get("distance", "medium")
                dist_m = DISTANCE_TO_M.get(distance, 12.0)
                depends_on[eid] = (other_id, dist_m)
    
    # For each chain root (entity with no dependencies), compute total chain length
    def compute_chain_length(eid: str, visited: set) -> float:
        """Compute cumulative distance from this entity to end of chain."""
        if eid in visited:
            return 0.0
        visited.add(eid)
        
        # Find entities that depend on this one
        chain_len = 0.0
        for other_eid, (dep_id, dist_m) in depends_on.items():
            if dep_id == eid:
                chain_len = max(chain_len, dist_m + compute_chain_length(other_eid, visited))
        return chain_len
    
    # Entities that have dependencies on them need extra path length
    for spec in actor_specs:
        eid = str(spec.get("id", ""))
        veh_num = spec.get("vehicle_num")
        if veh_num is None:
            continue
        
        chain_len = compute_chain_length(eid, set())
        if chain_len > 0:
            # This entity has things ahead of it - we need current + chain_len + margin
            current = required.get(veh_num, 0.0)
            # Assume base entity is placed around 60% of path (typical for after_exit)
            base_position_estimate = current * 0.6
            needed = base_position_estimate + chain_len + 10.0  # 10m margin
            required[veh_num] = max(current, needed)
    
    return required


def _find_parallel_vehicles(
    primary_veh_num: int,
    picked_list: List[Dict[str, Any]],
) -> List[Tuple[int, bool, str]]:
    """
    Find vehicles that share the same exit road as the primary vehicle,
    OR that enter on the primary's exit road (opposite direction).
    
    These vehicles travel in parallel, merge, or travel in the opposite
    direction on the same road - they should all be extended together when
    the primary vehicle is extended, so entities can be placed correctly.
    
    Returns list of (vehicle_number, is_same_direction, extend_mode) tuples.
    extend_mode is 'append' for same-direction, 'prepend' for opposite-direction.
    """
    # Get the primary vehicle's final road and heading
    primary_entry = None
    for p in picked_list:
        if _parse_vehicle_num(p.get("vehicle")) == primary_veh_num:
            primary_entry = p
            break
    
    if primary_entry is None:
        return []
    
    primary_roads = primary_entry.get("signature", {}).get("roads", [])
    if not primary_roads:
        return []
    
    primary_exit_road = primary_roads[-1]
    primary_exit_heading = primary_entry.get("signature", {}).get("exit", {}).get("heading_deg", 0)
    
    parallel_vehs = []
    for p in picked_list:
        veh_num = _parse_vehicle_num(p.get("vehicle"))
        if veh_num is None or veh_num == primary_veh_num:
            continue
        
        roads = p.get("signature", {}).get("roads", [])
        if not roads:
            continue
        
        exit_road = roads[-1]
        entry_road = roads[0]
        exit_heading = p.get("signature", {}).get("exit", {}).get("heading_deg", 0)
        entry_heading = p.get("signature", {}).get("entry", {}).get("heading_deg", 0)
        
        # Check if this vehicle shares the same EXIT road (same or opposite direction)
        if exit_road == primary_exit_road:
            # Determine if same direction (within 90 degrees) or opposite
            heading_diff = abs(exit_heading - primary_exit_heading)
            heading_diff = min(heading_diff, 360 - heading_diff)
            is_same_direction = heading_diff < 90
            parallel_vehs.append((veh_num, is_same_direction, 'append'))
        
        # Also check if this vehicle ENTERS on the primary's exit road (opposite direction)
        # This catches vehicles that start on the primary's exit road and turn off
        elif entry_road == primary_exit_road:
            # This vehicle enters where the primary exits - opposite direction
            # Check heading to confirm it's traveling opposite
            heading_diff = abs(entry_heading - primary_exit_heading)
            heading_diff = min(heading_diff, 360 - heading_diff)
            if heading_diff > 90:  # Opposite direction (more than 90 degrees different)
                parallel_vehs.append((veh_num, False, 'prepend'))
    
    return parallel_vehs


def _extend_parallel_paths(
    primary_veh_num: int,
    extended_pts: np.ndarray,
    picked_list: List[Dict[str, Any]],
    seg_by_id: Dict[int, np.ndarray],
    primary_end_pt_before_extension: np.ndarray,
) -> List[int]:
    """
    Extend parallel vehicles' paths using the same extension points.
    
    When we extend the primary vehicle's path, we want parallel vehicles
    (same exit road) to also extend so they can interact with entities 
    placed in the extended region.
    
    For same-direction vehicles: append offset extension points to the end
    For opposite-direction vehicles: prepend reversed extension points to the start
    
    Returns list of vehicle numbers that were extended.
    """
    parallel_vehs = _find_parallel_vehicles(primary_veh_num, picked_list)
    extended_vehs = []
    
    if len(extended_pts) == 0:
        return extended_vehs
    
    for veh_num, is_same_direction, extend_mode in parallel_vehs:
        picked_entry = None
        for p in picked_list:
            if _parse_vehicle_num(p.get("vehicle")) == veh_num:
                picked_entry = p
                break
        
        if picked_entry is None:
            continue
        
        sig = picked_entry.get("signature", {})
        seg_ids = sig.get("segment_ids", [])
        if not seg_ids:
            continue
        
        if extend_mode == 'append':
            # Same direction: extend from the END of the path
            last_seg_id = int(seg_ids[-1])
            last_pts = seg_by_id.get(last_seg_id)
            if last_pts is None or len(last_pts) < 2:
                continue
            
            last_pts = np.asarray(last_pts, dtype=float)
            end_pt = last_pts[-1]
            
            # Compute offset from primary vehicle's endpoint (before extension) to this vehicle's endpoint
            # This maintains the lane offset regardless of how far apart they are
            offset = end_pt - primary_end_pt_before_extension
            
            # Apply the offset to each extension point
            offset_pts = extended_pts + offset
            
            # Append to this vehicle's last segment
            new_pts = np.vstack([last_pts, offset_pts])
            seg_by_id[last_seg_id] = new_pts
            
            # Update segments_detailed
            segments_detailed = sig.get("segments_detailed", [])
            for sd in segments_detailed:
                if sd.get("seg_id") == last_seg_id:
                    sd["polyline_sample"] = _polyline_sample_from_pts(new_pts, max_points=20)
                    sd["length_m"] = float(_segment_length(new_pts))
            
            # Update the exit point
            new_end_pt = new_pts[-1]
            new_end_heading = _heading_at_end(new_pts)
            if "exit" in sig:
                sig["exit"]["point"] = {"x": float(new_end_pt[0]), "y": float(new_end_pt[1])}
                sig["exit"]["heading_deg"] = float(new_end_heading)
            
            extended_vehs.append(veh_num)
            print(f"[INFO] Extended parallel Vehicle {veh_num} (same dir) with {len(extended_pts)} waypoints")
            
        elif extend_mode == 'prepend':
            # Opposite direction relative to the primary: extend at this vehicle's ENTRY end.
            first_seg_id = int(seg_ids[0])  # Entry segment for this vehicle
            first_pts = seg_by_id.get(first_seg_id)
            if first_pts is None or len(first_pts) < 2:
                continue
            
            first_pts = np.asarray(first_pts, dtype=float)

            entry_pt = None
            if isinstance(sig.get("entry", {}), dict):
                ep = sig.get("entry", {}).get("point", {})
                if isinstance(ep, dict) and "x" in ep and "y" in ep:
                    entry_pt = np.array([float(ep["x"]), float(ep["y"])], dtype=float)

            start_pt = first_pts[0]
            end_pt = first_pts[-1]
            if entry_pt is None:
                entry_pt = start_pt

            d_start = float(np.linalg.norm(entry_pt - start_pt))
            d_end = float(np.linalg.norm(entry_pt - end_pt))
            entry_at_start = d_start <= d_end
            entry_end_pt = start_pt if entry_at_start else end_pt

            # Compute offset from primary's old end to this vehicle's entry endpoint
            offset = entry_end_pt - primary_end_pt_before_extension
            offset_pts = extended_pts + offset

            if len(offset_pts) >= 2:
                if entry_at_start:
                    # Ensure the last extension point lands near the existing entry.
                    if float(np.linalg.norm(offset_pts[-1] - entry_end_pt)) > float(np.linalg.norm(offset_pts[0] - entry_end_pt)):
                        offset_pts = offset_pts[::-1]
                    new_pts = np.vstack([offset_pts, first_pts])
                else:
                    # Entry is at the array end; extend after it and keep continuity.
                    if float(np.linalg.norm(offset_pts[0] - entry_end_pt)) > float(np.linalg.norm(offset_pts[-1] - entry_end_pt)):
                        offset_pts = offset_pts[::-1]
                    new_pts = np.vstack([first_pts, offset_pts])
            else:
                new_pts = np.vstack([offset_pts, first_pts]) if entry_at_start else np.vstack([first_pts, offset_pts])
            seg_by_id[first_seg_id] = new_pts
            
            # Update segments_detailed
            segments_detailed = sig.get("segments_detailed", [])
            for sd in segments_detailed:
                if sd.get("seg_id") == first_seg_id:
                    sd["polyline_sample"] = _polyline_sample_from_pts(new_pts, max_points=20)
                    sd["length_m"] = float(_segment_length(new_pts))
            
            # Update the entry point (where the opposite-direction vehicle actually starts)
            new_start_pt = new_pts[0] if entry_at_start else new_pts[-1]
            new_start_heading = _heading_at_start(new_pts) if entry_at_start else _heading_at_end(new_pts)
            if "entry" in sig:
                sig["entry"]["point"] = {"x": float(new_start_pt[0]), "y": float(new_start_pt[1])}
                if entry_at_start:
                    sig["entry"]["heading_deg"] = float(new_start_heading)
                else:
                    # Entry at array end implies travel opposite array order.
                    sig["entry"]["heading_deg"] = float((new_start_heading + 180) % 360)
            
            extended_vehs.append(veh_num)
            print(f"[INFO] Extended parallel Vehicle {veh_num} (opposite dir) with {len(extended_pts)} waypoints appended to entry")
    
    return extended_vehs


def extend_path_if_needed(
    picked_entry: Dict[str, Any],
    seg_by_id: Dict[int, np.ndarray],
    all_segments: List[Dict[str, Any]],
    target_length: float,
    max_extensions: int = 5,
    nodes: Optional[Dict[str, Any]] = None,
    picked_list: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[bool, float, List[int]]:
    """
    Extend a vehicle's path geometry if it's shorter than target_length.
    
    Works purely with geometry - finds waypoints that continue the path
    and appends them directly to the last segment's polyline.
    
    Also extends parallel vehicles (same exit road) with the same extension points.
    
    Modifies seg_by_id in-place (extends the last segment's geometry).
    
    Returns (was_extended, new_total_length, list_of_parallel_vehicles_extended)
    """
    sig = picked_entry.get("signature", {})
    seg_ids = sig.get("segment_ids", [])
    
    # Get the primary vehicle number for extending parallel paths
    primary_veh_num = _parse_vehicle_num(picked_entry.get("vehicle"))
    
    if not seg_ids:
        return False, 0.0, []
    
    current_length = _compute_path_length(picked_entry, seg_by_id)
    if current_length >= target_length:
        return False, current_length, []
    
    # Get the last segment's geometry
    last_seg_id = int(seg_ids[-1])
    last_pts = seg_by_id.get(last_seg_id)
    if last_pts is None or len(last_pts) < 2:
        return False, current_length, []
    
    last_pts = np.asarray(last_pts, dtype=float)
    end_pt = last_pts[-1]
    end_heading = _heading_at_end(last_pts)
    
    # Save the endpoint BEFORE extension for computing offsets for parallel vehicles
    primary_end_pt_before_extension = end_pt.copy()
    
    extended_parallel_vehs = []
    
    # If we have raw nodes, use them to find continuation waypoints
    if nodes is not None and "payload" in nodes:
        extended_pts = _extend_path_from_nodes(
            end_pt, end_heading, nodes,
            target_extension=target_length - current_length + 10.0,  # Add margin
            connect_radius_m=8.0,
            connect_yaw_tol_deg=30.0,  # Strict for straight-through
        )
        
        if len(extended_pts) > 0:
            # Append extended points to the last segment
            new_pts = np.vstack([last_pts, extended_pts])
            seg_by_id[last_seg_id] = new_pts
            
            # Also update segments_detailed polyline_sample if present
            segments_detailed = sig.get("segments_detailed", [])
            for sd in segments_detailed:
                if sd.get("seg_id") == last_seg_id:
                    sd["polyline_sample"] = _polyline_sample_from_pts(new_pts, max_points=20)
                    sd["length_m"] = float(_segment_length(new_pts))
            
            # Update the exit point in the signature to reflect the new end
            new_end_pt = new_pts[-1]
            new_end_heading = _heading_at_end(new_pts)
            if "exit" in sig:
                sig["exit"]["point"] = {"x": float(new_end_pt[0]), "y": float(new_end_pt[1])}
                sig["exit"]["heading_deg"] = float(new_end_heading)
            
            new_length = _compute_path_length(picked_entry, seg_by_id)
            print(f"[INFO] Path extension: extended from {current_length:.1f}m to {new_length:.1f}m "
                  f"(added {len(extended_pts)} waypoints)")
            
            # Extend parallel vehicles with the same extension points
            if picked_list is not None and primary_veh_num is not None:
                extended_parallel_vehs = _extend_parallel_paths(
                    primary_veh_num, extended_pts, picked_list, seg_by_id,
                    primary_end_pt_before_extension
                )
            
            return True, new_length, extended_parallel_vehs
    
    # Fallback: try segment-based extension (original approach)
    excluded = set(int(s) for s in seg_ids)
    extensions = 0
    fallback_extended_pts = []
    
    while current_length < target_length and extensions < max_extensions:
        end_pt = last_pts[-1]
        end_heading = _heading_at_end(last_pts)
        
        next_seg = _find_best_successor_segment(
            end_pt, end_heading, all_segments, excluded,
            connect_radius_m=10.0,
            connect_yaw_tol_deg=45.0,
        )
        
        if next_seg is None:
            break
        
        next_seg_id = int(next_seg["seg_id"])
        next_pts = np.asarray(next_seg["points"], dtype=float)
        
        # Append to last segment geometry
        new_pts = np.vstack([last_pts, next_pts])
        seg_by_id[last_seg_id] = new_pts
        last_pts = new_pts
        fallback_extended_pts = np.vstack([fallback_extended_pts, next_pts]) if len(fallback_extended_pts) > 0 else next_pts
        
        excluded.add(next_seg_id)
        current_length = _segment_length(new_pts)
        extensions += 1
        
        print(f"[INFO] Path extension: added segment {next_seg_id}, path now {current_length:.1f}m")
    
    new_length = _compute_path_length(picked_entry, seg_by_id)
    
    # Extend parallel vehicles with the same extension points (fallback path)
    if extensions > 0 and picked_list is not None and primary_veh_num is not None and len(fallback_extended_pts) > 0:
        extended_parallel_vehs = _extend_parallel_paths(
            primary_veh_num, np.asarray(fallback_extended_pts), picked_list, seg_by_id,
            primary_end_pt_before_extension
        )
    
    return extensions > 0, new_length, extended_parallel_vehs


def _extend_path_from_nodes(
    start_pt: np.ndarray,
    start_heading: float,
    nodes: Dict[str, Any],
    target_extension: float,
    connect_radius_m: float = 8.0,
    connect_yaw_tol_deg: float = 30.0,
) -> np.ndarray:
    """
    Find waypoints from raw nodes that continue the path from start_pt/start_heading.
    
    Returns array of (N, 2) points to append to the path.
    """
    payload = nodes.get("payload", {})
    x = np.asarray(payload.get("x", []), dtype=float)
    y = np.asarray(payload.get("y", []), dtype=float)
    yaw = np.asarray(payload.get("yaw", []), dtype=float)
    
    if len(x) == 0:
        return np.empty((0, 2), dtype=float)
    
    all_pts = np.vstack([x, y]).T
    
    # Find starting candidates: close to start_pt and heading-aligned
    dists = np.linalg.norm(all_pts - start_pt, axis=1)
    close_mask = dists < connect_radius_m
    
    if not np.any(close_mask):
        return np.empty((0, 2), dtype=float)
    
    # Check heading alignment
    heading_diffs = np.abs(np.mod(yaw[close_mask] - start_heading + 180, 360) - 180)
    aligned_mask = heading_diffs < connect_yaw_tol_deg
    
    close_indices = np.where(close_mask)[0]
    aligned_indices = close_indices[aligned_mask]
    
    if len(aligned_indices) == 0:
        return np.empty((0, 2), dtype=float)
    
    # Pick the closest aligned point as the seed
    seed_idx = aligned_indices[np.argmin(dists[aligned_indices])]
    
    # Greedy walk: follow waypoints that continue in roughly the same direction
    extended_points = []
    current_pt = all_pts[seed_idx]
    current_heading = float(yaw[seed_idx])
    visited = {seed_idx}
    total_dist = 0.0
    
    while total_dist < target_extension:
        # Find next waypoint: ahead of current, aligned heading
        candidates = []
        for i in range(len(x)):
            if i in visited:
                continue
            pt = all_pts[i]
            
            # Must be ahead (in direction of travel)
            vec_to_pt = pt - current_pt
            dist_to_pt = np.linalg.norm(vec_to_pt)
            if dist_to_pt < 0.5 or dist_to_pt > 15.0:  # Skip too close or too far
                continue
            
            # Check if it's roughly ahead (within 60° of current heading)
            angle_to_pt = np.degrees(np.arctan2(vec_to_pt[1], vec_to_pt[0]))
            angle_diff = abs(wrap180(angle_to_pt - current_heading))
            if angle_diff > 60:
                continue
            
            # Check heading alignment of the waypoint itself
            heading_diff = abs(wrap180(float(yaw[i]) - current_heading))
            if heading_diff > connect_yaw_tol_deg:
                continue
            
            candidates.append((i, dist_to_pt, angle_diff))
        
        if not candidates:
            break
        
        # Pick the best candidate (closest and most aligned)
        candidates.sort(key=lambda c: (c[2], c[1]))  # Sort by angle_diff, then distance
        best_idx = candidates[0][0]
        
        next_pt = all_pts[best_idx]
        step_dist = np.linalg.norm(next_pt - current_pt)
        
        extended_points.append(next_pt)
        visited.add(best_idx)
        total_dist += step_dist
        current_pt = next_pt
        current_heading = float(yaw[best_idx])
    
    if not extended_points:
        return np.empty((0, 2), dtype=float)
    
    return np.array(extended_points, dtype=float)


def extend_paths_for_entity_spacing(
    actor_specs: List[Dict[str, Any]],
    picked_list: List[Dict[str, Any]],
    seg_by_id: Dict[int, np.ndarray],
    all_segments: List[Dict[str, Any]],
) -> Dict[int, List[int]]:
    """
    Extend vehicle paths as needed to accommodate entity spacing requirements.
    
    Analyzes actor_specs for ahead_of/behind_of relations and extends paths
    that are too short.
    
    Returns dict mapping vehicle_num -> list of added segment IDs
    """
    # Estimate required lengths
    required_lengths = _estimate_required_path_length(actor_specs, picked_list, seg_by_id)
    
    extensions_made: Dict[int, List[int]] = {}
    
    for picked in picked_list:
        veh_num = _parse_vehicle_num(picked.get("vehicle"))
        if veh_num is None:
            continue
        
        current_len = _compute_path_length(picked, seg_by_id)
        target_len = required_lengths.get(veh_num, current_len)
        
        if target_len > current_len:
            print(f"[INFO] Vehicle {veh_num}: current path {current_len:.1f}m, "
                  f"need ~{target_len:.1f}m for entity spacing")
            
            was_extended, added_ids = extend_path_if_needed(
                picked, seg_by_id, all_segments, target_len
            )
            
            if was_extended:
                extensions_made[veh_num] = added_ids
                new_len = _compute_path_length(picked, seg_by_id)
                print(f"[INFO] Vehicle {veh_num}: extended to {new_len:.1f}m")
    
    return extensions_made


__all__ = [
    "_ang_diff_deg",
    "_compute_path_length",
    "_estimate_required_path_length",
    "_extend_path_from_nodes",
    "_extend_parallel_paths",
    "_find_best_successor_segment",
    "_find_parallel_vehicles",
    "_heading_at_end",
    "_heading_at_start",
    "_polyline_sample_from_pts",
    "_segment_length",
    "extend_path_if_needed",
    "extend_paths_for_entity_spacing",
]
