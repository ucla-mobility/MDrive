#!/usr/bin/env python3
"""
Route Alignment & Trimming Module

This module takes generated route XMLs and refines their waypoints using CARLA's
GlobalRoutePlanner to ensure they follow the correct road paths without detours
caused by wrong-direction snapping.

Key steps:
1. Load waypoints from route XML files
2. Snap waypoints to CARLA map using dynamic programming to minimize detours
3. Compress waypoints by removing redundant mid-points on straight segments
4. Recompute headings based on actual road direction
5. Write aligned XMLs back

Based on ChatScene/trajectory_to_xml.py snapping refinement logic.

Usage:
    python -m scenario_builder_api.pipeline.step_07_route_alignment.main \
        --routes-dir scenario_builder_api/routes/Wide_Turn_Negotiation_1 \
        --town Town05 \
        --carla-port 2012
"""

import argparse
import math
import sys
import inspect
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple
from xml.dom import minidom

import numpy as np


def _setup_carla_paths():
    """Try to add CARLA Python API to sys.path if not already available."""
    # Check common CARLA locations relative to this project
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    carla_candidates = [
        project_root / 'carla' / 'PythonAPI' / 'carla',
        project_root / 'carla912' / 'PythonAPI' / 'carla',
        project_root / 'external_paths' / 'carla_root' / 'PythonAPI' / 'carla',
        Path('/opt/carla/PythonAPI/carla'),
        Path.home() / 'carla' / 'PythonAPI' / 'carla',
    ]
    
    # IMPORTANT: Remove project root and CWD from sys.path to avoid importing
    # the wrong 'carla' module (the project has a 'carla/' directory that shadows
    # the CARLA Python API egg).
    paths_to_remove = []
    for p in list(sys.path):
        if not p:
            continue
        try:
            path_obj = Path(p).resolve()
        except Exception:
            continue
        # Never drop global site-packages; we still need stdlib extras like pkg_resources
        if "site-packages" in path_obj.as_posix().split("/"):
            continue
        # Only scrub paths that live inside the repository to avoid nuking the environment
        if not (path_obj == project_root or project_root in path_obj.parents):
            continue
        potential_shadow = path_obj / 'carla'
        if potential_shadow.exists() and potential_shadow.is_dir() and not (potential_shadow / 'dist').exists():
            paths_to_remove.append(p)
    
    for p in paths_to_remove:
        if p in sys.path:
            sys.path.remove(p)
    
    for carla_path in carla_candidates:
        if carla_path.exists():
            # Find and add the .egg file for carla module FIRST
            dist_dir = carla_path / 'dist'
            if dist_dir.exists():
                # Prefer py3.7 egg
                egg_files = sorted(dist_dir.glob('carla-*-py3*.egg'), reverse=True)
                if not egg_files:
                    egg_files = list(dist_dir.glob('carla-*.egg'))
                for egg in egg_files:
                    if str(egg) not in sys.path:
                        sys.path.append(str(egg))
                        break  # Only add one egg
            
            # Add the carla directory for agents module
            if str(carla_path) not in sys.path:
                sys.path.append(str(carla_path))
            
            return True
    return False


def _is_valid_carla(carla_module) -> bool:
    return all(hasattr(carla_module, attr) for attr in ("Client", "Location", "Transform"))


def _import_carla():
    import carla
    if not _is_valid_carla(carla):
        carla_path = getattr(carla, "__file__", "unknown location")
        raise ImportError(
            f"Imported 'carla' from {carla_path}, but it does not look like CARLA PythonAPI (missing Client)."
        )
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
    return carla, GlobalRoutePlanner, GlobalRoutePlannerDAO


def _ensure_carla():
    """Ensure CARLA is available, setting up paths if needed. Returns (carla, GRP, GRPDAO) or raises ImportError."""
    global _carla_module, _grp_class, _grpdao_class

    if _carla_module and _grp_class and _grpdao_class:
        return _carla_module, _grp_class, _grpdao_class

    # ALWAYS set up paths first to avoid importing the wrong carla module
    # (the project has a 'carla/' directory that can shadow the PythonAPI)
    _setup_carla_paths()
    
    # Clear any previously cached wrong carla module
    sys.modules.pop("carla", None)
    sys.modules.pop("agents", None)
    
    try:
        carla, GlobalRoutePlanner, GlobalRoutePlannerDAO = _import_carla()
        _carla_module, _grp_class, _grpdao_class = carla, GlobalRoutePlanner, GlobalRoutePlannerDAO
        return _carla_module, _grp_class, _grpdao_class
    except ImportError as e:
        raise ImportError(f"CARLA Python API not found. Set PYTHONPATH or install CARLA. Error: {e}")


# Module-level cache for CARLA imports
_carla_module = None
_grp_class = None
_grpdao_class = None


def prettify_xml(elem: ET.Element) -> str:
    """Return a pretty-printed XML string."""
    rough_string = ET.tostring(elem, encoding='utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ", encoding='utf-8').decode('utf-8')


def load_route_xml(xml_path: Path) -> Tuple[Dict, List[Dict]]:
    """
    Load a route XML file and extract route metadata and waypoints.
    
    Returns:
        (route_attrs, waypoints) where waypoints is a list of dicts with x, y, z, yaw
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    route_elem = root.find('route')
    if route_elem is None:
        raise ValueError(f"No <route> element found in {xml_path}")
    
    route_attrs = dict(route_elem.attrib)
    waypoints = []
    
    for wp in route_elem.findall('waypoint'):
        waypoints.append({
            'x': float(wp.get('x', 0)),
            'y': float(wp.get('y', 0)),
            'z': float(wp.get('z', 0)),
            'yaw': float(wp.get('yaw', 0)),
            'pitch': float(wp.get('pitch', 0)) if wp.get('pitch') else 0.0,
            'roll': float(wp.get('roll', 0)) if wp.get('roll') else 0.0,
        })
    
    return route_attrs, waypoints


def save_route_xml(xml_path: Path, route_attrs: Dict, waypoints: List[Dict]):
    """Save aligned waypoints back to route XML."""
    routes = ET.Element('routes')
    route = ET.SubElement(routes, 'route')
    
    for key, value in route_attrs.items():
        route.set(key, str(value))
    
    for wp in waypoints:
        waypoint = ET.SubElement(route, 'waypoint')
        waypoint.set('x', f"{wp['x']:.3f}")
        waypoint.set('y', f"{wp['y']:.3f}")
        waypoint.set('z', f"{wp.get('z', 0.0):.3f}")
        waypoint.set('yaw', f"{wp['yaw']:.6f}")
        if wp.get('pitch'):
            waypoint.set('pitch', f"{wp['pitch']:.6f}")
        if wp.get('roll'):
            waypoint.set('roll', f"{wp['roll']:.6f}")
    
    xml_string = prettify_xml(routes)
    # Remove extra blank lines
    import re
    xml_string = re.sub(r'\n\s*\n', '\n', xml_string)
    
    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write(xml_string)


def recompute_headings(waypoints: List[Dict]) -> None:
    """Overwrite yaw based on direction to next waypoint."""
    if len(waypoints) < 2:
        return
    
    for idx in range(len(waypoints)):
        if idx < len(waypoints) - 1:
            dx = waypoints[idx + 1]['x'] - waypoints[idx]['x']
            dy = waypoints[idx + 1]['y'] - waypoints[idx]['y']
        else:
            dx = waypoints[idx]['x'] - waypoints[idx - 1]['x']
            dy = waypoints[idx]['y'] - waypoints[idx - 1]['y']
        
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            waypoints[idx]['yaw'] = math.degrees(math.atan2(dy, dx))


def yaw_diff_deg(a: float, b: float) -> float:
    """Compute smallest angle difference in degrees."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def route_metrics(route: List) -> Tuple[float, bool, List[str]]:
    """
    Compute total length and detect if route contains turns.
    
    Args:
        route: List of (waypoint, road_option) tuples from GlobalRoutePlanner
    
    Returns:
        (length, has_turn, options_list)
    """
    length = 0.0
    has_turn = False
    options = []
    
    if not route:
        return length, has_turn, options
    
    prev_loc = route[0][0].transform.location
    for wp, opt in route[1:]:
        loc = wp.transform.location
        length += loc.distance(prev_loc)
        prev_loc = loc
        opt_name = getattr(opt, 'name', str(opt))
        options.append(opt_name)
        if opt_name.upper() not in ('LANEFOLLOW', 'VOID'):
            has_turn = True
    
    return length, has_turn, options


def max_segment_deviation(
    waypoints: List[Dict],
    start_idx: int,
    end_idx: int,
    start_loc,
    end_loc,
) -> float:
    """Maximum lateral deviation of intermediate points from the line between start/end."""
    if end_idx - start_idx <= 1:
        return 0.0
    
    sx, sy = start_loc.x, start_loc.y
    ex, ey = end_loc.x, end_loc.y
    dx = ex - sx
    dy = ey - sy
    seg_len_sq = dx * dx + dy * dy
    max_dev = 0.0
    
    for idx in range(start_idx + 1, end_idx):
        px = waypoints[idx]['x']
        py = waypoints[idx]['y']
        if seg_len_sq == 0:
            dist = math.hypot(px - sx, py - sy)
        else:
            t = ((px - sx) * dx + (py - sy) * dy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            proj_x = sx + t * dx
            proj_y = sy + t * dy
            dist = math.hypot(px - proj_x, py - proj_y)
        if dist > max_dev:
            max_dev = dist
    
    return max_dev


def refine_waypoints_dp(
    carla_map,
    waypoints: List[Dict],
    grp,
    radius: float = 1.5,
    k: int = 5,
    w_deviation: float = 1.0,
    w_route: float = 1.0,
    turn_penalty: float = 5.0,
    skip_penalty: float = 0.5,
    shape_penalty: float = 0.2,
    max_skip: int = 20,
    waypoint_step: float = 1.0,
) -> Tuple[List[Dict], List]:
    """
    Use dynamic programming to find optimal waypoint snapping that minimizes
    detours and wrong-direction routing.
    
    Returns:
        (refined_waypoints, global_plan)
    """
    if not waypoints:
        return [], []
    
    # Recompute headings based on actual waypoint directions
    recompute_headings(waypoints)
    
    heading_thresh = 30.0
    straight_thresh = 15.0
    lane_change_penalty = 30.0
    straight_turn_penalty = 30.0
    
    # Get all map waypoints
    all_wps = carla_map.generate_waypoints(waypoint_step)
    if not all_wps:
        raise RuntimeError("No waypoints generated from CARLA map.")
    
    coords = np.array([[wp.transform.location.x, wp.transform.location.y] for wp in all_wps])
    rad2 = radius * radius
    n = len(waypoints)
    
    def candidate_indices(idx: int, xy: np.ndarray) -> List[int]:
        traj_yaw = waypoints[idx]['yaw']
        diff = coords - xy[None, :]
        dist2 = np.einsum("ij,ij->i", diff, diff)
        idxs = np.where(dist2 <= rad2)[0]
        if idxs.size == 0:
            idxs = np.array([int(np.argmin(dist2))])
        
        filtered = []
        for ci in idxs:
            wp_yaw = all_wps[ci].transform.rotation.yaw
            if yaw_diff_deg(wp_yaw, traj_yaw) <= heading_thresh:
                filtered.append((ci, dist2[ci]))
        
        if not filtered:
            filtered = [(ci, dist2[ci]) for ci in idxs]
        
        filtered.sort(key=lambda x: x[1])
        return [ci for ci, _ in filtered[:k]]
    
    # Build candidates for each waypoint
    candidates_per_step = []
    for idx, wp in enumerate(waypoints):
        xy = np.array([wp['x'], wp['y']])
        candidates_per_step.append(candidate_indices(idx, xy))
    
    # DP tables
    best_cost = [dict() for _ in range(n)]
    backref = [dict() for _ in range(n)]
    
    # Initialize first step
    for ci in candidates_per_step[0]:
        dev = np.linalg.norm(np.array([waypoints[0]['x'], waypoints[0]['y']]) - coords[ci])
        best_cost[0][ci] = w_deviation * dev
        backref[0][ci] = None
    
    # Forward pass
    for i in range(1, n):
        obs = np.array([waypoints[i]['x'], waypoints[i]['y']])
        traj_yaw_i = waypoints[i]['yaw']
        
        for ci in candidates_per_step[i]:
            dev = np.linalg.norm(obs - coords[ci])
            base_cost = w_deviation * dev
            best_val = None
            best_prev = None
            j_start = max(0, i - max_skip)
            
            for j in range(i - 1, j_start - 1, -1):
                if not best_cost[j]:
                    continue
                
                traj_yaw_j = waypoints[j]['yaw']
                delta_heading = yaw_diff_deg(traj_yaw_i, traj_yaw_j)
                is_straight_traj = delta_heading < straight_thresh
                gap = i - j
                
                for prev_ci, prev_cost in best_cost[j].items():
                    prev_wp = all_wps[prev_ci]
                    curr_wp = all_wps[ci]
                    
                    route = grp.trace_route(prev_wp.transform.location, curr_wp.transform.location)
                    if not route:
                        continue
                    
                    route_len, has_turn, _ = route_metrics(route)
                    skip_cost = skip_penalty * max(0, gap - 1)
                    
                    shape_cost = 0.0
                    if gap > 1:
                        shape_cost = shape_penalty * max_segment_deviation(
                            waypoints, j, i, prev_wp.transform.location, curr_wp.transform.location
                        )
                    
                    same_lane = (
                        prev_wp.road_id == curr_wp.road_id and 
                        prev_wp.lane_id == curr_wp.lane_id
                    )
                    
                    total = (
                        prev_cost
                        + base_cost
                        + w_route * route_len
                        + (turn_penalty if has_turn else 0.0)
                        + skip_cost
                        + shape_cost
                        + (0.0 if same_lane else lane_change_penalty)
                    )
                    
                    if is_straight_traj and has_turn:
                        total += straight_turn_penalty
                    
                    if best_val is None or total < best_val:
                        best_val = total
                        best_prev = (j, prev_ci)
            
            if best_val is not None:
                best_cost[i][ci] = best_val
                backref[i][ci] = best_prev
    
    if not best_cost[-1]:
        raise RuntimeError("Failed to align waypoints to map (no terminal state).")
    
    # Backtrack to find best path
    end_ci = min(best_cost[-1], key=lambda ci: best_cost[-1][ci])
    path = []
    idx = n - 1
    ci = end_ci
    
    while True:
        path.append((idx, ci))
        prev = backref[idx].get(ci)
        if prev is None:
            break
        idx, ci = prev
    
    path.reverse()
    
    # Build state records
    state_records = []
    for sample_idx, cand_idx in path:
        wp = all_wps[cand_idx]
        loc = wp.transform.location
        yaw = wp.transform.rotation.yaw
        state_records.append({
            'sample_index': sample_idx,
            'candidate_index': cand_idx,
            'wp': wp,
            'state': {
                'x': loc.x,
                'y': loc.y,
                'z': loc.z,
                'yaw': yaw,
            }
        })
    
    if len(state_records) < 2:
        raise RuntimeError("Refined trajectory must contain at least start and end waypoints.")
    
    # Compute segment data for compression
    segment_data = []
    for prev_state, curr_state in zip(state_records[:-1], state_records[1:]):
        prev_wp = prev_state['wp']
        curr_wp = curr_state['wp']
        route = grp.trace_route(prev_wp.transform.location, curr_wp.transform.location)
        route_len, has_turn, options = route_metrics(route)
        segment_data.append({
            'route': route,
            'options': options,
            'has_turn': has_turn,
            'start_wp': prev_wp,
            'end_wp': curr_wp,
        })
    
    # Compress: keep only waypoints where lane/road changes or turns occur
    keep_flags = [False] * len(state_records)
    keep_flags[0] = True
    keep_flags[-1] = True
    
    for mid in range(1, len(state_records) - 1):
        prev_state = state_records[mid - 1]
        curr_state = state_records[mid]
        next_state = state_records[mid + 1]
        prev_wp = prev_state['wp']
        curr_wp = curr_state['wp']
        next_wp = next_state['wp']
        
        lane_change = (
            curr_wp.road_id != prev_wp.road_id or
            curr_wp.lane_id != prev_wp.lane_id or
            next_wp.road_id != curr_wp.road_id or
            next_wp.lane_id != curr_wp.lane_id
        )
        turn_near = segment_data[mid - 1]['has_turn'] or segment_data[mid]['has_turn']
        
        if lane_change or turn_near:
            keep_flags[mid] = True
            continue
        
        # Check if we can merge through this waypoint
        merged_route = grp.trace_route(prev_wp.transform.location, next_wp.transform.location)
        if not merged_route:
            keep_flags[mid] = True
            continue
        
        _, merged_turn, _ = route_metrics(merged_route)
        if merged_turn:
            keep_flags[mid] = True
    
    compressed_states = [state for flag, state in zip(keep_flags, state_records) if flag]
    if len(compressed_states) < 2:
        compressed_states = state_records
    
    # Build final waypoints
    final_waypoints = [state['state'] for state in compressed_states]
    recompute_headings(final_waypoints)
    
    # Build global plan for the compressed route
    plan = []
    for prev_state, curr_state in zip(compressed_states[:-1], compressed_states[1:]):
        prev_wp = prev_state['wp']
        curr_wp = curr_state['wp']
        route_seg = grp.trace_route(prev_wp.transform.location, curr_wp.transform.location)
        if plan and route_seg:
            route_seg = route_seg[1:]
        plan.extend(route_seg)
    
    return final_waypoints, plan


def align_route_file(
    xml_path: Path,
    carla_map,
    grp,
    backup: bool = True,
    **snap_kwargs,
) -> Tuple[int, int]:
    """
    Align a single route XML file.
    
    Returns:
        (original_count, aligned_count) waypoint counts
    """
    route_attrs, waypoints = load_route_xml(xml_path)
    original_count = len(waypoints)
    
    if original_count < 2:
        print(f"  [SKIP] {xml_path.name}: too few waypoints ({original_count})")
        return original_count, original_count
    
    try:
        aligned_waypoints, plan = refine_waypoints_dp(
            carla_map, waypoints, grp, **snap_kwargs
        )
    except Exception as e:
        print(f"  [FAIL] {xml_path.name}: {e}")
        return original_count, original_count
    
    aligned_count = len(aligned_waypoints)
    
    # Backup original
    if backup:
        backup_path = xml_path.with_suffix('.xml.orig')
        if not backup_path.exists():
            import shutil
            shutil.copy(xml_path, backup_path)
    
    # Save aligned
    save_route_xml(xml_path, route_attrs, aligned_waypoints)
    
    return original_count, aligned_count


def align_routes_in_directory(
    routes_dir: Path,
    town: str,
    carla_host: str = '127.0.0.1',
    carla_port: int = 2000,
    backup: bool = True,
    sampling_resolution: float = 2.0,
    **snap_kwargs,
) -> Dict[str, Tuple[int, int]]:
    """
    Align all route XML files in a directory.
    
    Returns:
        Dict mapping filename to (original_count, aligned_count)
    """
    # Ensure CARLA is available (with automatic path setup)
    carla, GlobalRoutePlanner, GlobalRoutePlannerDAO = _ensure_carla()
    
    # Connect to CARLA
    print(f"Connecting to CARLA at {carla_host}:{carla_port}...")
    client = carla.Client(carla_host, carla_port)
    client.set_timeout(30.0)
    world = client.get_world()
    
    current_map = world.get_map().name
    # Handle map name variations (e.g., "Town05" vs "/Game/Carla/Maps/Town05")
    current_map_base = current_map.split('/')[-1] if '/' in current_map else current_map
    
    if current_map_base != town:
        print(f"Loading {town}...")
        world = client.load_world(town)
    
    carla_map = world.get_map()
    print(f"Using map: {carla_map.name}")
    
    # Create GlobalRoutePlanner (API changed in CARLA 0.9.12+)
    # 0.9.10: GlobalRoutePlanner(dao)
    # 0.9.12: GlobalRoutePlanner(world_map, sampling_resolution)
    # Detect signature to stay compatible.
    grp_init_params = list(inspect.signature(GlobalRoutePlanner.__init__).parameters.values())[1:]  # skip self
    try:
        if len(grp_init_params) >= 2 and grp_init_params[0].name != "dao":
            grp = GlobalRoutePlanner(carla_map, sampling_resolution)
        else:
            grp = GlobalRoutePlanner(GlobalRoutePlannerDAO(carla_map, sampling_resolution))
    except TypeError:
        # Fallback: try both styles explicitly before bubbling up
        try:
            grp = GlobalRoutePlanner(carla_map, sampling_resolution)
        except Exception:
            grp = GlobalRoutePlanner(GlobalRoutePlannerDAO(carla_map, sampling_resolution))
    # CARLA 0.9.10 expected an explicit setup(); 0.9.12 builds the graph in __init__.
    if hasattr(grp, "setup"):
        grp.setup()
    
    # Find all route XMLs
    xml_files = list(routes_dir.glob('*.xml'))
    # Also check actors subdirectory
    actors_dir = routes_dir / 'actors'
    if actors_dir.exists():
        for subdir in actors_dir.iterdir():
            if subdir.is_dir():
                xml_files.extend(subdir.glob('*.xml'))
    
    if not xml_files:
        print(f"No XML files found in {routes_dir}")
        return {}
    
    print(f"Found {len(xml_files)} route XML files to align")
    
    results = {}
    for xml_path in xml_files:
        print(f"  Aligning {xml_path.name}...")
        orig, aligned = align_route_file(
            xml_path, carla_map, grp, backup=backup, **snap_kwargs
        )
        results[xml_path.name] = (orig, aligned)
        
        if orig != aligned:
            print(f"    {orig} waypoints -> {aligned} waypoints")
        else:
            print(f"    {aligned} waypoints (unchanged)")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Align and trim route XML waypoints using CARLA GlobalRoutePlanner'
    )
    parser.add_argument(
        '--routes-dir', '-r',
        type=Path,
        required=True,
        help='Directory containing route XML files'
    )
    parser.add_argument(
        '--town', '-t',
        type=str,
        default='Town05',
        help='CARLA town name (default: Town05)'
    )
    parser.add_argument(
        '--carla-host',
        type=str,
        default='127.0.0.1',
        help='CARLA server host (default: 127.0.0.1)'
    )
    parser.add_argument(
        '--carla-port',
        type=int,
        default=2000,
        help='CARLA server port (default: 2000)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip backing up original XML files'
    )
    parser.add_argument(
        '--sampling-resolution',
        type=float,
        default=2.0,
        help='GlobalRoutePlanner sampling resolution in meters (default: 2.0)'
    )
    # Snapping parameters
    parser.add_argument('--snap-radius', type=float, default=1.5,
                        help='Search radius for candidate waypoints (default: 1.5m)')
    parser.add_argument('--snap-k', type=int, default=5,
                        help='Max candidates per waypoint (default: 5)')
    parser.add_argument('--snap-w-deviation', type=float, default=1.0,
                        help='Weight for deviation cost (default: 1.0)')
    parser.add_argument('--snap-w-route', type=float, default=1.0,
                        help='Weight for route length cost (default: 1.0)')
    parser.add_argument('--snap-turn-penalty', type=float, default=5.0,
                        help='Penalty for introducing turns (default: 5.0)')
    parser.add_argument('--snap-skip-penalty', type=float, default=0.5,
                        help='Penalty per skipped waypoint (default: 0.5)')
    parser.add_argument('--snap-shape-penalty', type=float, default=0.2,
                        help='Penalty for deviation from original shape (default: 0.2)')
    parser.add_argument('--snap-max-skip', type=int, default=20,
                        help='Max waypoints to skip in one jump (default: 20)')
    
    args = parser.parse_args()
    
    routes_dir = args.routes_dir.resolve()
    if not routes_dir.exists():
        raise FileNotFoundError(f"Routes directory not found: {routes_dir}")
    
    print(f"\n=== Route Alignment & Trimming ===")
    print(f"Routes dir: {routes_dir}")
    print(f"Town: {args.town}")
    print()
    
    snap_kwargs = {
        'radius': args.snap_radius,
        'k': args.snap_k,
        'w_deviation': args.snap_w_deviation,
        'w_route': args.snap_w_route,
        'turn_penalty': args.snap_turn_penalty,
        'skip_penalty': args.snap_skip_penalty,
        'shape_penalty': args.snap_shape_penalty,
        'max_skip': args.snap_max_skip,
    }
    
    results = align_routes_in_directory(
        routes_dir=routes_dir,
        town=args.town,
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        backup=not args.no_backup,
        sampling_resolution=args.sampling_resolution,
        **snap_kwargs,
    )
    
    # Summary
    total_orig = sum(r[0] for r in results.values())
    total_aligned = sum(r[1] for r in results.values())
    
    print(f"\n=== Summary ===")
    print(f"Processed {len(results)} route files")
    print(f"Total waypoints: {total_orig} -> {total_aligned}")
    if total_orig > 0:
        print(f"Reduction: {100 * (1 - total_aligned / total_orig):.1f}%")
    print(f"\nAligned routes saved to: {routes_dir}")
    if not args.no_backup:
        print(f"Original files backed up as *.xml.orig")


if __name__ == '__main__':
    main()
