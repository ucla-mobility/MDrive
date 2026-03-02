#!/usr/bin/env python3
"""
Convert scene_objects.json to CoLMDriver route XML format.

This script takes a scene_objects.json file (output from the scenario builder pipeline)
and generates:
1. Individual XML route files for each vehicle (ego and NPCs)
2. Individual XML files for static actors
3. An actors_manifest.json that references all the XML files

Usage:
    python convert_scene_to_routes.py --input log/scenarios_001/scene_objects.json --output-dir routes_output
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any
import xml.etree.ElementTree as ET
from xml.dom import minidom
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


def prettify_xml(elem: ET.Element) -> str:
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, encoding='utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ", encoding='utf-8').decode('utf-8')


def safe_filename(text: str) -> str:
    """Convert text to safe filename."""
    text = re.sub(r'[^a-zA-Z0-9_-]+', '_', text)
    return text.strip('_').lower()


def extract_town_from_nodes_path(nodes_path: str) -> str:
    """Extract town name from nodes path like 'town_nodes/Town05.json'."""
    basename = os.path.basename(nodes_path)
    town = os.path.splitext(basename)[0]  # Remove .json
    return town


def create_route_xml(route_id: str, town: str, role: str, waypoints: List[Dict[str, float]], 
                      model: str = None, speed: float = None, length: str = None, 
                      width: str = None) -> ET.Element:
    """Create XML route structure."""
    routes = ET.Element('routes')
    route = ET.SubElement(routes, 'route')
    route.set('id', route_id)
    route.set('town', town)
    route.set('role', role)
    
    if model:
        route.set('model', model)
    if speed is not None:
        route.set('speed', str(speed))
    if length:
        route.set('length', str(length))
    if width:
        route.set('width', str(width))
    
    for wp in waypoints:
        waypoint = ET.SubElement(route, 'waypoint')
        waypoint.set('x', f"{wp['x']:.3f}")
        waypoint.set('y', f"{wp['y']:.3f}")
        waypoint.set('z', f"{wp.get('z', 0.0):.3f}")
        waypoint.set('yaw', f"{wp['yaw_deg']:.6f}")
        if 'pitch' in wp:
            waypoint.set('pitch', f"{wp['pitch']:.6f}")
        if 'roll' in wp:
            waypoint.set('roll', f"{wp['roll']:.6f}")
    
    return routes


def write_xml_file(filepath: str, xml_root: ET.Element):
    """Write XML to file with proper formatting."""
    xml_string = prettify_xml(xml_root)
    # Remove extra blank lines
    xml_string = re.sub(r'\n\s*\n', '\n', xml_string)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(xml_string)


def extract_waypoints_from_path(path_data: Dict[str, Any]) -> List[Dict[str, float]]:
    """Extract waypoints from path signature's polyline samples."""
    waypoints = []
    
    if 'signature' in path_data and 'segments_detailed' in path_data['signature']:
        for segment in path_data['signature']['segments_detailed']:
            if 'polyline_sample' in segment:
                for point in segment['polyline_sample']:
                    # Check if this point is different from the last added point
                    if not waypoints or (waypoints[-1]['x'] != point['x'] or waypoints[-1]['y'] != point['y']):
                        wp = {
                            'x': point['x'],
                            'y': point['y'],
                            'z': point.get('z', 0.0),
                            'yaw_deg': segment['start']['heading_deg']  # Use segment heading
                        }
                        waypoints.append(wp)
    
    # If no waypoints from detailed segments, try using refined data or original
    if not waypoints and 'refined' in path_data:
        # Use original signature polyline
        if 'signature_original' in path_data and 'segments_detailed' in path_data['signature_original']:
            for segment in path_data['signature_original']['segments_detailed']:
                if 'polyline_sample' in segment:
                    for point in segment['polyline_sample']:
                        if not waypoints or (waypoints[-1]['x'] != point['x'] or waypoints[-1]['y'] != point['y']):
                            wp = {
                                'x': point['x'],
                                'y': point['y'],
                                'z': point.get('z', 0.0),
                                'yaw_deg': segment['start']['heading_deg']
                            }
                            waypoints.append(wp)
    
    return waypoints


def convert_scene_to_routes(
    scene_json_path: str,
    output_dir: str,
    ego_num: int = None,
    align_routes: bool = True,
    carla_host: str = '127.0.0.1',
    carla_port: int = 2000,
):
    """
    Convert scene_objects.json to route XML files and actors_manifest.json.
    
    Args:
        scene_json_path: Path to scene_objects.json file
        output_dir: Directory to write output files
        ego_num: Number of ego vehicles (first N vehicles in ego_picked). If None, auto-detect or default to 1.
        align_routes: If True, align and trim route waypoints using CARLA GlobalRoutePlanner
        carla_host: CARLA server host for route alignment
        carla_port: CARLA server port for route alignment
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load scene data
    with open(scene_json_path, 'r', encoding='utf-8') as f:
        scene = json.load(f)
    
    # Extract town from nodes path
    town = extract_town_from_nodes_path(scene.get('nodes', 'town_nodes/Town05.json'))
    
    # Auto-detect ego_num if not specified: use the number of vehicles in ego_picked
    ego_picked = scene.get('ego_picked', [])
    if ego_num is None:
        ego_num = len(ego_picked)
        print(f"[INFO] Auto-detected {ego_num} ego vehicle(s) from ego_picked array")
    else:
        print(f"[INFO] Using explicit --ego-num={ego_num} (overriding auto-detection of {len(ego_picked)} vehicles)")
    
    # Prepare manifest
    manifest = {
        'ego': [],
        'npc': [],
        'static': [],
        'pedestrian': [],
        'bicycle': []
    }
    
    # Extract base route ID from scene or generate one
    route_id_base = Path(scene_json_path).parent.name.replace('scenarios_', '')
    
    # Process ego vehicles
    for idx, ego_data in enumerate(ego_picked):
        vehicle_name = ego_data.get('vehicle', f'Vehicle_{idx+1}')
        vehicle_safe = safe_filename(vehicle_name)
        
        # First ego_num vehicles are egos, rest are NPCs
        is_ego = (idx < ego_num)
        role = 'ego' if is_ego else 'npc'
        
        # Extract waypoints
        waypoints = extract_waypoints_from_path(ego_data)
        
        if not waypoints:
            print(f"[WARN] No waypoints found for {vehicle_name}, skipping")
            continue
        
        # Get speed from refined data
        speed = None
        if 'refined' in ego_data and 'speed_mps' in ego_data['refined']:
            speed = ego_data['refined']['speed_mps']
        
        # Extract vehicle number from name (e.g., "Vehicle 1" -> 1, convert to 0-indexed -> 0)
        vehicle_number = idx
        match = re.search(r'\d+', vehicle_name)
        if match:
            vehicle_number = int(match.group()) - 1  # Convert to 0-indexed
        
        # Create filename with 0-indexed vehicle number for proper ego_id matching
        filename = f"{town.lower()}_{vehicle_safe}_{vehicle_number}.xml"
        filepath = output_path / filename
        
        # Create XML
        xml_root = create_route_xml(
            route_id=f"{route_id_base}",
            town=town,
            role=role,
            waypoints=waypoints,
            speed=speed
        )
        
        # Write XML file
        write_xml_file(str(filepath), xml_root)
        print(f"[OK] Created {filename} ({len(waypoints)} waypoints)")
        
        # Add to manifest
        manifest_entry = {
            'file': filename,
            'route_id': route_id_base,
            'town': town,
            'name': vehicle_name,
            'kind': role
        }
        
        if speed is not None:
            manifest_entry['speed'] = speed
        
        manifest[role].append(manifest_entry)
    
    # Process all actors (static, pedestrians, cyclists)
    actors = scene.get('actors', [])
    
    for actor in actors:
        actor_id = actor.get('id', 'unknown')
        category = actor.get('category', 'static')
        
        # Determine role based on category
        if category == 'walker':
            role = 'pedestrian'
            subdir = 'pedestrians'
        elif category == 'cyclist':
            role = 'bicycle'
            subdir = 'bicycles'
        elif category == 'static':
            role = 'static'
            subdir = 'static'
        elif category == 'vehicle':
            # Check if it's a moving vehicle or static/parked
            motion = actor.get('motion', {})
            motion_type = motion.get('type', 'static')
            if motion_type == 'static':
                role = 'static'
                subdir = 'static'
            else:
                role = 'npc'
                subdir = 'npc'
        else:
            print(f"[WARN] Unknown actor category '{category}' for {actor_id}, treating as static")
            role = 'static'
            subdir = 'static'
        
        # Get spawn location
        spawn = actor.get('spawn', {})
        if not spawn:
            print(f"[WARN] No spawn location for actor {actor_id}, skipping")
            continue
        
        # Get waypoints (either world_waypoints or just spawn point)
        world_waypoints = actor.get('world_waypoints', [])
        if world_waypoints:
            # Use provided waypoints
            waypoints = []
            for wp in world_waypoints:
                waypoints.append({
                    'x': wp['x'],
                    'y': wp['y'],
                    'z': wp.get('z', 0.0),
                    'yaw_deg': wp['yaw_deg']
                })
        else:
            # Create single waypoint from spawn point
            waypoints = [{
                'x': spawn['x'],
                'y': spawn['y'],
                'z': spawn.get('z', 0.0),
                'yaw_deg': spawn['yaw_deg']
            }]
        
        # Get asset ID (CARLA model)
        model = actor.get('asset_id', None)
        if not model:
            # Set default models based on role
            if role == 'pedestrian':
                model = 'walker.pedestrian.0001'
            elif role == 'bicycle':
                model = 'vehicle.bh.crossbike'
            elif role == 'npc':
                model = 'vehicle.audi.a2'
            else:
                model = 'static.prop.trafficcone'
        
        # Get speed from waypoints or motion
        speed = None
        if world_waypoints and len(world_waypoints) > 0:
            speed = world_waypoints[0].get('speed_mps')
        if speed is None:
            motion = actor.get('motion', {})
            if motion.get('type') != 'static':
                # Set default speeds
                if role == 'pedestrian':
                    speed = 1.5
                elif role == 'bicycle':
                    speed = 4.0
        
        # Create subdirectory for actor type
        actor_dir = output_path / 'actors' / subdir
        actor_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        semantic = actor.get('semantic', role)
        filename = f"{town.lower()}_{safe_filename(actor_id)}_{safe_filename(semantic)}.xml"
        filepath = actor_dir / filename
        
        # Create XML
        xml_root = create_route_xml(
            route_id=route_id_base,
            town=town,
            role=role,
            waypoints=waypoints,
            model=model,
            speed=speed
        )
        
        # Write XML file
        write_xml_file(str(filepath), xml_root)
        print(f"[OK] Created actors/{subdir}/{filename} ({len(waypoints)} waypoints)")
        
        # Add to manifest
        manifest_entry = {
            'file': f"actors/{subdir}/{filename}",
            'route_id': route_id_base,
            'town': town,
            'name': actor_id,
            'kind': role,
            'model': model
        }
        
        if speed is not None:
            manifest_entry['speed'] = speed
        
        manifest[role].append(manifest_entry)
    
    # Write actors_manifest.json
    manifest_path = output_path / 'actors_manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    
    print(f"\n[OK] Created actors_manifest.json")

    # Write actors_behavior.json (optional)
    behavior_entries = []
    for actor in actors:
        trigger = actor.get("trigger")
        action = actor.get("action")
        if trigger is None and action is None:
            continue
        actor_name = actor.get("id", "unknown")
        behavior_entries.append(
            {
                "route_id": route_id_base,
                "actor_name": actor_name,
                "trigger": trigger,
                "action": action,
            }
        )

    if behavior_entries:
        behavior_path = output_path / "actors_behavior.json"
        with open(behavior_path, "w", encoding="utf-8") as f:
            json.dump({"behaviors": behavior_entries}, f, indent=2)
        print(f"[OK] Created actors_behavior.json ({len(behavior_entries)} entries)")
    
    # Align routes using CARLA GlobalRoutePlanner (if requested)
    if align_routes:
        try:
            from ..step_07_route_alignment import align_routes_in_directory
            print(f"[INFO] Aligning routes with CARLA at {carla_host}:{carla_port}...")
            align_routes_in_directory(
                routes_dir=output_path,
                town=town,
                carla_host=carla_host,
                carla_port=carla_port,
                backup=True,
            )
            print(f"[OK] Route alignment complete!")
            try:
                _plot_routes_after_alignment(output_path)
            except Exception as viz_exc:
                print(f"[WARN] Could not plot aligned routes: {viz_exc}")
        except ImportError as e:
            print(f"[WARN] Route alignment not available (CARLA Python API not found): {e}")
        except Exception as e:
            print(f"[WARN] Route alignment failed: {e}")
    
    print(f"[OK] Conversion complete!")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Generated {len(manifest['ego'])} ego, {len(manifest['npc'])} npc, "
          f"{len(manifest['static'])} static, {len(manifest['pedestrian'])} pedestrian, "
          f"{len(manifest['bicycle'])} bicycle actors")


def _plot_routes_after_alignment(routes_dir: Path) -> None:
    """
    Save a quick visualization of all aligned route XMLs (ego + actors) to routes_dir/aligned_routes.png.
    """
    xml_files = list(routes_dir.glob("*.xml"))
    actors_dir = routes_dir / "actors"
    if actors_dir.exists():
        xml_files.extend(list(actors_dir.rglob("*.xml")))
    if not xml_files:
        return

    plt.figure(figsize=(8, 6))
    for xf in xml_files:
        try:
            tree = ET.parse(str(xf))
            root = tree.getroot()
            for route in root.iter("route"):
                xs, ys = [], []
                for wp in route.iter("waypoint"):
                    xs.append(float(wp.get("x", 0)))
                    ys.append(float(wp.get("y", 0)))
                if len(xs) >= 2:
                    plt.plot(xs, ys, linewidth=1.0, alpha=0.7)
        except Exception:
            continue

    plt.title("Aligned routes (post-CARLA alignment)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    out_path = routes_dir / "aligned_routes.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Convert scene_objects.json to CoLMDriver route XML format',
        epilog='''
Examples:
  # Convert single file
  %(prog)s --input log/scenarios_001/scene_objects.json --output-dir routes_001
  
  # Batch convert all scenarios in a directory
  %(prog)s --batch-dir log --output-dir routes_all
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--input',
        '-i',
        help='Path to scene_objects.json file'
    )
    parser.add_argument(
        '--output-dir',
        '-o',
        required=True,
        help='Output directory for XML files and manifest'
    )
    parser.add_argument(
        '--route-id',
        default=None,
        help='Override route ID (default: extracted from input path)'
    )
    parser.add_argument(
        '--batch-dir',
        '-b',
        help='Process all scene_objects.json files in this directory (recursively)'
    )
    parser.add_argument(
        '--ego-num',
        type=int,
        default=None,
        help='Number of ego vehicles (first N in ego_picked are egos, rest are NPCs). Default: auto-detect from ego_picked length'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output directory if it exists'
    )
    parser.add_argument(
        '--align-routes',
        action='store_true',
        help='Align and trim route waypoints using CARLA GlobalRoutePlanner (requires CARLA running)'
    )
    parser.add_argument(
        '--carla-host',
        type=str,
        default='127.0.0.1',
        help='CARLA server host for route alignment (default: 127.0.0.1)'
    )
    parser.add_argument(
        '--carla-port',
        type=int,
        default=2000,
        help='CARLA server port for route alignment (default: 2000)'
    )
    
    args = parser.parse_args()
    
    if not args.input and not args.batch_dir:
        parser.error("Either --input or --batch-dir must be specified")
    
    if args.batch_dir:
        # Batch mode: process all scene_objects.json files
        batch_path = Path(args.batch_dir)
        if not batch_path.exists():
            print(f"[ERROR] Batch directory not found: {args.batch_dir}")
            return 1
        
        scene_files = list(batch_path.rglob("scene_objects.json"))
        if not scene_files:
            print(f"[ERROR] No scene_objects.json files found in {args.batch_dir}")
            return 1
        
        print(f"[INFO] Found {len(scene_files)} scene_objects.json files")
        
        success_count = 0
        for scene_file in scene_files:
            # Create output subdirectory based on scenario name
            scenario_name = scene_file.parent.name
            output_subdir = Path(args.output_dir) / scenario_name
            
            print(f"\n{'='*80}")
            print(f"[INFO] Processing {scenario_name}: {scene_file}")
            print(f"{'='*80}")
            
            try:
                convert_scene_to_routes(str(scene_file), str(output_subdir), ego_num=args.ego_num)
                
                # Optional: align routes using CARLA GlobalRoutePlanner
                if args.align_routes:
                    try:
                        from ..step_07_route_alignment import align_routes_in_directory
                        # Extract town from scene
                        with open(scene_file, 'r') as f:
                            scene = json.load(f)
                        town = extract_town_from_nodes_path(scene.get('nodes', 'town_nodes/Town05.json'))
                        
                        align_routes_in_directory(
                            routes_dir=output_subdir,
                            town=town,
                            carla_host=args.carla_host,
                            carla_port=args.carla_port,
                            backup=True,
                        )
                        print(f"[OK] Route alignment complete for {scenario_name}")
                    except ImportError as e:
                        print(f"[WARN] Route alignment not available: {e}")
                    except Exception as e:
                        print(f"[WARN] Route alignment failed for {scenario_name}: {e}")
                
                success_count += 1
            except Exception as e:
                print(f"[ERROR] Failed to convert {scenario_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n{'='*80}")
        print(f"[INFO] Batch conversion complete: {success_count}/{len(scene_files)} successful")
        print(f"{'='*80}")
        return 0 if success_count > 0 else 1
    
    else:
        # Single file mode
        if not os.path.exists(args.input):
            print(f"[ERROR] Input file not found: {args.input}")
            return 1
        
        try:
            convert_scene_to_routes(args.input, args.output_dir, ego_num=args.ego_num)
            
            # Optional: align routes using CARLA GlobalRoutePlanner
            if args.align_routes:
                print(f"\n{'='*80}")
                print("[INFO] Aligning routes with CARLA GlobalRoutePlanner...")
                print(f"{'='*80}")
                try:
                    from ..step_07_route_alignment import align_routes_in_directory
                    # Extract town from scene
                    import json
                    with open(args.input, 'r') as f:
                        scene = json.load(f)
                    town = extract_town_from_nodes_path(scene.get('nodes', 'town_nodes/Town05.json'))
                    
                    align_routes_in_directory(
                        routes_dir=Path(args.output_dir),
                        town=town,
                        carla_host=args.carla_host,
                        carla_port=args.carla_port,
                        backup=True,
                    )
                    print("[OK] Route alignment complete!")
                except ImportError as e:
                    print(f"[WARN] Route alignment not available: {e}")
                except Exception as e:
                    print(f"[WARN] Route alignment failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            return 0
        except Exception as e:
            print(f"[ERROR] Conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return 1


if __name__ == '__main__':
    exit(main())
