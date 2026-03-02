#!/usr/bin/env python3
"""
Verify whether ego vehicle XML waypoints align with CARLA's lane centerlines.

Connects to a running CARLA server, loads the ucla_v2 map, then for each ego
XML in the patched routes directory:
  1. Parses waypoints from the XML
  2. For each waypoint, queries map.get_waypoint() for the nearest lane
  3. Measures the lateral offset between the XML position and the lane center
  4. Reports statistics

Usage:
    python tools/verify_ego_lane_alignment.py \
        --routes-dir v2xpnp/dataset/carla_routes_patched/2023-03-17-15-53-02_1_0 \
        --port 2050
"""

import argparse
import glob
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Add CARLA egg
carla_root = Path(__file__).resolve().parents[1] / "carla912"
egg_glob = list(carla_root.glob("PythonAPI/carla/dist/carla-*.egg"))
if egg_glob:
    sys.path.insert(0, str(egg_glob[0]))
sys.path.insert(0, str(carla_root / "PythonAPI/carla"))

import carla  # noqa: E402


def parse_waypoints_from_xml(xml_path: str):
    """Parse waypoints from an ego XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    route = root.find("route")
    if route is None:
        return [], {}
    
    attrs = dict(route.attrib)
    waypoints = []
    for wp in route.findall("waypoint"):
        waypoints.append({
            "x": float(wp.get("x", 0)),
            "y": float(wp.get("y", 0)),
            "z": float(wp.get("z", 0)),
            "yaw": float(wp.get("yaw", 0)),
            "time": float(wp.get("time", 0)),
        })
    return waypoints, attrs


def measure_lane_offset(carla_map, waypoints, sample_every=10):
    """
    For sampled waypoints, measure lateral offset from CARLA lane center.
    Returns list of (time, xml_x, xml_y, center_x, center_y, lateral_offset, lane_id, road_id).
    """
    results = []
    for i, wp in enumerate(waypoints):
        if i % sample_every != 0:
            continue
        
        loc = carla.Location(x=wp["x"], y=wp["y"], z=wp["z"])
        carla_wp = carla_map.get_waypoint(loc, project_to_road=True, 
                                           lane_type=carla.LaneType.Driving)
        if carla_wp is None:
            # Try any lane type
            carla_wp = carla_map.get_waypoint(loc, project_to_road=True,
                                               lane_type=carla.LaneType.Any)
        
        if carla_wp is None:
            results.append({
                "time": wp["time"],
                "xml_x": wp["x"],
                "xml_y": wp["y"],
                "center_x": None,
                "center_y": None,
                "offset": float("inf"),
                "lane_id": None,
                "road_id": None,
                "lane_width": None,
            })
            continue
        
        ct = carla_wp.transform
        cx, cy = ct.location.x, ct.location.y
        
        # Lateral offset = distance from XML point to lane center
        dx = wp["x"] - cx
        dy = wp["y"] - cy
        offset = math.sqrt(dx*dx + dy*dy)
        
        # Also compute signed lateral offset (perpendicular to lane direction)
        lane_yaw_rad = math.radians(ct.rotation.yaw)
        # Lane forward vector
        fwd_x = math.cos(lane_yaw_rad)
        fwd_y = math.sin(lane_yaw_rad) 
        # Lane right vector (perpendicular)
        right_x = -fwd_y
        right_y = fwd_x
        # Signed lateral offset (positive = right of center)
        signed_offset = dx * right_x + dy * right_y
        
        results.append({
            "time": wp["time"],
            "xml_x": wp["x"],
            "xml_y": wp["y"],
            "xml_yaw": wp["yaw"],
            "center_x": cx,
            "center_y": cy,
            "center_yaw": ct.rotation.yaw,
            "offset": offset,
            "signed_offset": signed_offset,
            "lane_id": carla_wp.lane_id,
            "road_id": carla_wp.road_id,
            "lane_width": carla_wp.lane_width,
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=2050)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--sample-every", type=int, default=10,
                        help="Sample every Nth waypoint (default: 10 = every 1 second)")
    args = parser.parse_args()

    # Connect to CARLA
    print(f"Connecting to CARLA at {args.host}:{args.port}...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    
    # Load map
    print("Loading ucla_v2 map...")
    world = client.load_world("ucla_v2")
    carla_map = world.get_map()
    print(f"Map loaded: {carla_map.name}")
    
    # Find ego XMLs
    ego_xmls = sorted(args.routes_dir.glob("*ego*.xml"))
    if not ego_xmls:
        ego_xmls = sorted(args.routes_dir.glob("*.xml"))
    
    print(f"\nFound {len(ego_xmls)} ego XML files")
    print("=" * 80)
    
    for xml_path in ego_xmls:
        print(f"\n{'=' * 80}")
        print(f"FILE: {xml_path.name}")
        print(f"{'=' * 80}")
        
        waypoints, attrs = parse_waypoints_from_xml(str(xml_path))
        print(f"  snap_to_road: {attrs.get('snap_to_road', 'N/A')}")
        print(f"  Total waypoints: {len(waypoints)}")
        
        if not waypoints:
            print("  No waypoints found!")
            continue
        
        results = measure_lane_offset(carla_map, waypoints, args.sample_every)
        
        if not results:
            print("  No results!")
            continue
        
        # Compute statistics
        offsets = [r["offset"] for r in results if r["offset"] != float("inf")]
        signed = [r["signed_offset"] for r in results if "signed_offset" in r]
        widths = [r["lane_width"] for r in results if r.get("lane_width")]
        
        if not offsets:
            print("  Could not match any waypoints to lanes!")
            continue
        
        avg_offset = sum(offsets) / len(offsets)
        max_offset = max(offsets)
        min_offset = min(offsets)
        
        avg_width = sum(widths) / len(widths) if widths else 0
        half_width = avg_width / 2
        
        # Count how many waypoints are outside lane center by more than thresholds
        n_over_05m = sum(1 for o in offsets if o > 0.5)
        n_over_1m = sum(1 for o in offsets if o > 1.0)
        n_over_half_lane = sum(1 for i, o in enumerate(offsets) if widths and o > widths[min(i, len(widths)-1)] / 2)
        
        print(f"\n  LANE ALIGNMENT STATISTICS (sampled every {args.sample_every} frames):")
        print(f"  {'─' * 60}")
        print(f"  Waypoints analyzed: {len(offsets)}")
        print(f"  Average lane width: {avg_width:.2f} m (half = {half_width:.2f} m)")
        print(f"  ")
        print(f"  Distance from lane center (XY):")
        print(f"    Mean offset:  {avg_offset:.3f} m")
        print(f"    Max offset:   {max_offset:.3f} m")
        print(f"    Min offset:   {min_offset:.3f} m")
        print(f"  ")
        print(f"  Signed lateral offset (+ = right of center):")
        if signed:
            avg_signed = sum(signed) / len(signed)
            print(f"    Mean:  {avg_signed:.3f} m  ({'right' if avg_signed > 0 else 'left'} of center)")
            print(f"    Range: [{min(signed):.3f}, {max(signed):.3f}] m")
        print(f"  ")
        print(f"  Violation counts:")
        print(f"    > 0.5m from center: {n_over_05m}/{len(offsets)} ({100*n_over_05m/len(offsets):.1f}%)")
        print(f"    > 1.0m from center: {n_over_1m}/{len(offsets)} ({100*n_over_1m/len(offsets):.1f}%)")
        print(f"    > half lane width:  {n_over_half_lane}/{len(offsets)} ({100*n_over_half_lane/len(offsets):.1f}%)")
        
        # Show worst offenders
        sorted_results = sorted(results, key=lambda r: r["offset"], reverse=True)
        print(f"\n  TOP 10 WORST OFFSETS:")
        print(f"  {'Time':>8s} {'XML X':>10s} {'XML Y':>10s} {'Ctr X':>10s} {'Ctr Y':>10s} {'Offset':>8s} {'Signed':>8s} {'LaneW':>7s} {'Road':>5s} {'Lane':>5s}")
        for r in sorted_results[:10]:
            if r.get("center_x") is None:
                print(f"  {r['time']:8.1f} {r['xml_x']:10.3f} {r['xml_y']:10.3f} {'N/A':>10s} {'N/A':>10s} {'INF':>8s}")
            else:
                print(f"  {r['time']:8.1f} {r['xml_x']:10.3f} {r['xml_y']:10.3f} {r['center_x']:10.3f} {r['center_y']:10.3f} {r['offset']:8.3f} {r.get('signed_offset',0):8.3f} {r.get('lane_width',0):7.2f} {r.get('road_id','?'):>5} {r.get('lane_id','?'):>5}")
        
        # Show a few samples across the trajectory
        print(f"\n  SAMPLE WAYPOINTS ACROSS TRAJECTORY:")
        step = max(1, len(results) // 10)
        print(f"  {'Time':>8s} {'XML X':>10s} {'XML Y':>10s} {'Ctr X':>10s} {'Ctr Y':>10s} {'Offset':>8s} {'Signed':>8s} {'Road':>5s} {'Lane':>5s}")
        for i in range(0, len(results), step):
            r = results[i]
            if r.get("center_x") is None:
                continue
            print(f"  {r['time']:8.1f} {r['xml_x']:10.3f} {r['xml_y']:10.3f} {r['center_x']:10.3f} {r['center_y']:10.3f} {r['offset']:8.3f} {r.get('signed_offset',0):8.3f} {r.get('road_id','?'):>5} {r.get('lane_id','?'):>5}")

    print(f"\n{'=' * 80}")
    print("DONE")


if __name__ == "__main__":
    main()
