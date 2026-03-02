from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .geometry import ang_diff_deg
from .models import CropBox, LaneSegment, LegalPath


def build_connectivity(segments: List[LaneSegment],
                       connect_radius_m: float = 6.0,
                       connect_yaw_tol_deg: float = 60.0,
                       allow_cross_lane: bool = False,
                       strict_endpoint_dist_m: Optional[float] = 4.5) -> List[List[int]]:
    """
    Build segment-to-segment connectivity graph.
    Two segments are connected if:
    1. End of segment A is close to start of segment B (within connect_radius_m)
    2. Heading at end of A aligns with heading at start of B (within connect_yaw_tol_deg)
    3. If on the same road, they must be on the same lane (unless allow_cross_lane=True)
    4. Optional: enforce endpoint distance for cross-road connections via strict_endpoint_dist_m
    """
    n = len(segments)
    adj: List[List[int]] = [[] for _ in range(n)]
    if n == 0:
        return adj

    starts = np.vstack([seg.points[0] for seg in segments])
    tree = cKDTree(starts)

    for i, seg_a in enumerate(segments):
        end_pt = seg_a.points[-1]
        end_heading = seg_a.heading_at_end()

        candidates = tree.query_ball_point(end_pt, r=connect_radius_m)
        for j in candidates:
            if i == j:
                continue
            seg_b = segments[j]
            
            # Check lane consistency: if same road, must be same lane
            # This prevents spurious cross-lane connections on parallel lanes
            if not allow_cross_lane:
                if seg_a.road_id == seg_b.road_id and seg_a.lane_id != seg_b.lane_id:
                    continue
            
            # For cross-road connections, require stricter endpoint proximity
            # to prevent lateral jumps to parallel lanes on different roads
            if strict_endpoint_dist_m is not None:
                endpoint_dist = float(np.linalg.norm(end_pt - seg_b.points[0]))
                if seg_a.road_id != seg_b.road_id and endpoint_dist > strict_endpoint_dist_m:
                    continue
            
            start_heading = seg_b.heading_at_start()
            if ang_diff_deg(end_heading, start_heading) <= connect_yaw_tol_deg:
                adj[i].append(j)

    return adj


def identify_boundary_segments(segments: List[LaneSegment],
                               crop: CropBox,
                               corridor_mode: bool = False,
                               boundary_margin: float = 3.0,
                               adj: Optional[List[List[int]]] = None) -> Tuple[List[int], List[int]]:
    """
    Identify segments that cross or are near the crop boundary.

    Entry segments: start outside (or near boundary), enter inside
    Exit segments: start inside, exit outside (or near boundary)
    
    Additionally, "terminal" segments (one end has no connectivity) are also
    treated as entry/exit. This handles cases where roads end within the crop
    region (e.g., dead ends or roads that were clipped).
    
    Args:
        segments: List of lane segments
        crop: Crop bounding box
        corridor_mode: If True, allow pass-through segments (start outside, end outside,
                       middle inside) to be classified as BOTH entry AND exit.
                       This is needed for corridor topologies where a road runs
                       entirely through the crop region.
        boundary_margin: Distance from boundary to consider as "at the boundary".
                        This handles segments that start/end slightly inside the crop
                        but should still be treated as entry/exit segments.
        adj: Adjacency list for segment connectivity. If provided, segments with
             terminal endpoints (no incoming/outgoing connections) are also
             classified as entry/exit segments.
    """
    entry_segments = []
    exit_segments = []
    
    def is_near_boundary(pt: np.ndarray) -> bool:
        """Check if a point is near (within margin of) any crop boundary."""
        x, y = float(pt[0]), float(pt[1])
        near_xmin = abs(x - crop.xmin) < boundary_margin
        near_xmax = abs(x - crop.xmax) < boundary_margin
        near_ymin = abs(y - crop.ymin) < boundary_margin
        near_ymax = abs(y - crop.ymax) < boundary_margin
        return near_xmin or near_xmax or near_ymin or near_ymax
    
    def is_deeply_inside(pt: np.ndarray) -> bool:
        """Check if point is well inside the crop (not near any boundary)."""
        return crop.contains(pt) and not is_near_boundary(pt)
    
    # Build reverse adjacency to detect incoming connections
    incoming = [[] for _ in segments] if adj else None
    if adj:
        for i, neighbors in enumerate(adj):
            for j in neighbors:
                incoming[j].append(i)
    
    def has_no_incoming(seg_idx: int) -> bool:
        """Check if segment has no incoming connections (terminal start)."""
        if incoming is None:
            return False
        return len(incoming[seg_idx]) == 0
    
    def has_no_outgoing(seg_idx: int) -> bool:
        """Check if segment has no outgoing connections (terminal end)."""
        if adj is None:
            return False
        return len(adj[seg_idx]) == 0

    for i, seg in enumerate(segments):
        start_pt = seg.points[0]
        end_pt = seg.points[-1]
        
        start_inside = crop.contains(start_pt)
        end_inside = crop.contains(end_pt)
        start_at_boundary = is_near_boundary(start_pt)
        end_at_boundary = is_near_boundary(end_pt)

        # Entry segment: starts outside/at-boundary, ends inside
        if not start_inside and end_inside:
            entry_segments.append(i)
        elif start_inside and start_at_boundary and is_deeply_inside(end_pt):
            # Start is technically inside but very near boundary, end is deep inside
            entry_segments.append(i)
        elif not start_inside and not end_inside:
            for pt in seg.points[1:-1]:
                if crop.contains(pt):
                    entry_segments.append(i)
                    break
        elif start_inside and end_inside and has_no_incoming(i):
            # Fully inside but no incoming connections - terminal start = entry
            entry_segments.append(i)

        # Exit segment: starts inside, ends outside/at-boundary
        if start_inside and not end_inside:
            exit_segments.append(i)
        elif is_deeply_inside(start_pt) and end_inside and end_at_boundary:
            # Start is deep inside, end is technically inside but very near boundary
            exit_segments.append(i)
        elif not start_inside and not end_inside:
            # Pass-through segment: starts outside, ends outside, but passes through crop
            if corridor_mode:
                # In corridor mode, pass-through segments can serve as BOTH entry and exit
                # This allows straight roads that run entirely through the crop to generate paths
                for pt in seg.points[1:-1]:
                    if crop.contains(pt):
                        if i not in exit_segments:
                            exit_segments.append(i)
                        break
            else:
                # Original behavior: only add to exit if not already an entry
                if i not in entry_segments:
                    for pt in seg.points[1:-1]:
                        if crop.contains(pt):
                            exit_segments.append(i)
                            break
        elif start_inside and end_inside and has_no_outgoing(i):
            # Fully inside but no outgoing connections - terminal end = exit
            if i not in exit_segments:
                exit_segments.append(i)

    return entry_segments, exit_segments


def generate_legal_paths(segments: List[LaneSegment],
                         adj: List[List[int]],
                         crop: CropBox,
                         min_path_length: float = 20.0,
                         max_paths: int = 100,
                         max_depth: int = 10,
                         allow_within_region_fallback: bool = True,
                         corridor_mode: bool = False,
                         roundabout_mode: bool = False,
                         t_junction_mode: bool = False) -> List[LegalPath]:
    """
    Generate legal paths that go from outside the crop area to outside.
    
    Args:
        segments: List of lane segments
        adj: Adjacency list for segment connectivity
        crop: Crop bounding box
        min_path_length: Minimum path length in meters
        max_paths: Maximum number of paths to generate
        max_depth: Maximum DFS depth
        allow_within_region_fallback: If True, fall back to any paths within region
        corridor_mode: If True, use corridor-specific boundary segment detection
                       that allows pass-through segments as both entry and exit.
                       Also allows single-segment paths for corridors.
        roundabout_mode: If True, apply roundabout-specific filtering to remove
                         jittery paths, short bypasses, and keep only the smoothest
                         paths per cardinal direction combo.
        t_junction_mode: If True, use terminal segment detection for T-junctions
                         where roads may end within the crop region.
    """
    legal_paths: List[LegalPath] = []

    # Only pass adj for T-junction mode (terminal segment detection)
    # For other topologies, this causes spurious entry/exit detection of junction connectors
    adj_for_boundary = adj if t_junction_mode else None
    entry_segments, exit_segments = identify_boundary_segments(segments, crop, corridor_mode=corridor_mode, adj=adj_for_boundary)
    print(f"[INFO] Found {len(entry_segments)} entry segments and {len(exit_segments)} exit segments")

    if len(entry_segments) == 0 or len(exit_segments) == 0:
        print("[WARNING] No entry or exit segments found. Paths must cross crop boundary.")
        if not allow_within_region_fallback:
            print("[INFO] Returning 0 legal paths for this crop (requires boundary-crossing paths).")
            return []
        print("[INFO] Falling back to any paths within the region...")
        entry_segments = list(range(len(segments)))
        exit_segments = list(range(len(segments)))

    exit_set = set(exit_segments)
    
    # In corridor mode, allow single-segment paths (a road that passes entirely through)
    min_path_segments = 1 if corridor_mode else 2
    
    # Collect all paths from each entry, then interleave for diversity
    paths_by_entry: Dict[int, List[LegalPath]] = {entry_idx: [] for entry_idx in entry_segments}
    # For roundabout mode, collect many paths per entry to ensure we find all cardinal combos
    max_per_entry = 500 if roundabout_mode else max(1, max_paths // len(entry_segments) * 3)

    def dfs(current_idx: int, path: List[int], total_length: float, depth: int, 
            entry_idx: int, collected: List[LegalPath], max_collect: int):
        if len(collected) >= max_collect:
            return

        if current_idx in exit_set and len(path) >= min_path_segments:
            if total_length >= min_path_length:
                path_segments = [segments[i] for i in path]
                collected.append(LegalPath(path_segments, total_length))
                # In corridor mode with single-segment path, don't continue exploring
                if corridor_mode and len(path) == 1:
                    return

        if depth >= max_depth:
            return

        for next_idx in adj[current_idx]:
            if next_idx in path:
                continue
            if len(collected) >= max_collect:
                return
            next_seg = segments[next_idx]
            new_length = total_length + next_seg.length()
            dfs(next_idx, path + [next_idx], new_length, depth + 1, 
                entry_idx, collected, max_collect)

    # Collect paths from each entry
    for entry_idx in entry_segments:
        collected: List[LegalPath] = []
        dfs(entry_idx, [entry_idx], segments[entry_idx].length(), 1,
            entry_idx, collected, max_per_entry)
        paths_by_entry[entry_idx] = collected
    
    # Roundabout-specific filtering to remove jittery paths and keep clean routes
    if roundabout_mode:
        def has_road_revisit(path: LegalPath) -> bool:
            """Check if path revisits the same road_id after leaving it (indicates jitter/backtracking).
            
            Consecutive segments on the same road are OK (they're part of the same road stretch).
            Revisiting a road after leaving it to visit other roads is the jitter we want to filter.
            """
            if len(path.segments) < 3:
                return False
                
            # Track roads we've completely left (not just the previous segment)
            left_roads: Set[int] = set()
            prev_road = path.segments[0].road_id
            
            for seg in path.segments[1:]:
                curr_road = seg.road_id
                if curr_road != prev_road:
                    # We're changing roads - mark the previous road as "left"
                    left_roads.add(prev_road)
                    # Check if we're returning to a road we already left
                    if curr_road in left_roads:
                        return True
                prev_road = curr_road
            return False
        
        def is_too_short(path: LegalPath, min_unique_roads: int = 4) -> bool:
            """Filter out paths that are too short to actually traverse the roundabout."""
            unique_roads = len(set(seg.road_id for seg in path.segments))
            return unique_roads < min_unique_roads
        
        def has_consecutive_same_road(path: LegalPath, max_consecutive: int = 2) -> bool:
            """Filter out paths with >2 consecutive segments on same road (excessive jitter)."""
            if len(path.segments) < 2:
                return False
            consecutive_count = 1
            prev_road = path.segments[0].road_id
            for seg in path.segments[1:]:
                if seg.road_id == prev_road:
                    consecutive_count += 1
                    if consecutive_count > max_consecutive:
                        return True
                else:
                    consecutive_count = 1
                    prev_road = seg.road_id
            return False
        
        def path_smoothness_score(path: LegalPath) -> float:
            """Lower score = smoother path. Penalize many segments and short segments."""
            num_segs = len(path.segments)
            # Count unique roads (fewer is better)
            unique_roads = len(set(seg.road_id for seg in path.segments))
            # Average segment length (longer is better, so invert)
            avg_seg_len = path.total_length / num_segs if num_segs > 0 else 0
            # Score: prefer fewer segments, fewer road changes, longer avg segments
            return num_segs + unique_roads - (avg_seg_len / 20.0)
        
        def get_cardinal_direction(heading_deg: float) -> str:
            """Convert heading to cardinal direction."""
            h = heading_deg % 360
            if h > 180:
                h -= 360
            if -45 <= h < 45:
                return "W"
            if 45 <= h < 135:
                return "N"
            if h >= 135 or h < -135:
                return "E"
            return "S"
        
        for entry_idx in paths_by_entry:
            # Apply all filters
            paths_by_entry[entry_idx] = [
                p for p in paths_by_entry[entry_idx] 
                if not has_road_revisit(p) 
                and not is_too_short(p)
                and not has_consecutive_same_road(p)
            ]
        
        # Collect all filtered paths, then dedupe by cardinal direction combo
        all_filtered: List[LegalPath] = []
        for paths in paths_by_entry.values():
            all_filtered.extend(paths)
        
        # Group by (entry_cardinal, exit_cardinal) and keep top 2 smoothest per combo
        from collections import defaultdict
        paths_by_combo: Dict[Tuple[str, str], List[LegalPath]] = defaultdict(list)
        
        for path in all_filtered:
            entry_heading = path.segments[0].heading_at_start()
            exit_heading = path.segments[-1].heading_at_end()
            entry_card = get_cardinal_direction(entry_heading)
            exit_card = get_cardinal_direction(exit_heading)
            paths_by_combo[(entry_card, exit_card)].append(path)
        
        # Sort each combo by smoothness and keep top 2
        final_paths: List[LegalPath] = []
        for combo, paths in sorted(paths_by_combo.items()):
            sorted_paths = sorted(paths, key=path_smoothness_score)
            final_paths.extend(sorted_paths[:2])  # Keep top 2 smoothest per combo
        
        return final_paths
    
    # Standard mode: interleave paths from different entries for diversity
    entry_iters = {entry_idx: iter(paths) for entry_idx, paths in paths_by_entry.items()}
    active_entries = list(entry_iters.keys())
    
    while len(legal_paths) < max_paths and active_entries:
        next_active = []
        for entry_idx in active_entries:
            if len(legal_paths) >= max_paths:
                break
            try:
                path = next(entry_iters[entry_idx])
                legal_paths.append(path)
                next_active.append(entry_idx)
            except StopIteration:
                # This entry is exhausted
                pass
        active_entries = next_active

    return legal_paths
