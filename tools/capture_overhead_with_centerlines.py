#!/usr/bin/env python3
"""
Capture an overhead CARLA screenshot centred on given x,y coordinates,
with lane centerlines, road edges, and lane markings overlaid.

Usage examples:

  # Centered on (x=100, y=-50) with a 100m margin around that point:
  python tools/capture_overhead_with_centerlines.py --port 2005 --x 100 --y -50 --margin 100

  # Larger area, higher resolution:
  python tools/capture_overhead_with_centerlines.py --port 2005 --x 100 --y -50 --margin 200 --width 4096 --height 4096

  # Custom output path:
  python tools/capture_overhead_with_centerlines.py --port 2005 --x 100 --y -50 --margin 150 -o my_overhead.png

  # With route waypoints from a route XML overlaid:
  python tools/capture_overhead_with_centerlines.py --port 2005 --x 100 --y -50 --margin 150 --route-xml path/to/route.xml

The script:
  1. Connects to a running CARLA server.
  2. Places an RGB camera directly overhead looking down at (x, y, z_high).
  3. Captures the rendered frame.
  4. Queries the CARLA map for all waypoints within the bounding box.
  5. Draws lane centerlines, road boundaries, and lane marks as an overlay.
  6. Optionally draws route waypoints from a route XML file.
  7. Saves the composited image.
"""

from __future__ import annotations

import argparse
import math
import queue
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

# ── CARLA import ────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
CARLA_EGG = REPO_ROOT / "carla912" / "PythonAPI" / "carla" / "dist" / "carla-0.9.12-py3.7-linux-x86_64.egg"
if CARLA_EGG.exists() and str(CARLA_EGG) not in sys.path:
    sys.path.insert(0, str(CARLA_EGG))

try:
    import carla  # type: ignore
except ImportError as exc:
    raise RuntimeError(
        "Cannot import carla.  Set PYTHONPATH to include the CARLA egg."
    ) from exc


# ── helpers ─────────────────────────────────────────────────────────────────

def build_intrinsic(width: int, height: int, fov_deg: float) -> np.ndarray:
    f = height / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return np.array([[f, 0, width / 2.0],
                     [0, f, height / 2.0],
                     [0, 0, 1.0]], dtype=np.float64)


def world_to_pixel(
    pts: np.ndarray,            # (N, 3) world XYZ
    cam_x: float, cam_y: float, cam_z: float,
    width: int, height: int, fov_deg: float,
) -> np.ndarray:
    """Project world coordinates to pixel coords for a downward-facing camera.

    Camera at (cam_x, cam_y, cam_z) looking straight down (pitch=-90).
    Returns (N, 2) float array of (px_x, px_y).
    """
    K = build_intrinsic(width, height, fov_deg)

    # Camera-relative coords.  For pitch=-90 yaw=0:
    #   cam_x_axis → world +X   (right in image)
    #   cam_y_axis → world +Y   (down in image)
    #   cam_z_axis → world -Z   (into the scene)
    # So camera coords = R @ (world - cam_pos), with R = diag(1,1,-1) * Ry(-90)
    # Simplification for straight-down:
    rel = pts - np.array([cam_x, cam_y, cam_z])
    # cam coord: x_c = rel_x, y_c = rel_y, z_c = -rel_z (pointing downward)
    cam_pts = np.stack([rel[:, 0], rel[:, 1], -rel[:, 2]], axis=-1)

    # Project
    proj = (K @ cam_pts.T).T           # (N, 3)
    proj[:, :2] /= proj[:, 2:3]       # normalise
    return proj[:, :2]


def lateral_shift_loc(transform: carla.Transform, shift: float) -> carla.Location:
    """Shift a transform sideways by *shift* metres."""
    t = carla.Transform(transform.location, transform.rotation)
    t.rotation.yaw += 90
    fwd = t.get_forward_vector()
    return transform.location + carla.Location(x=shift * fwd.x, y=shift * fwd.y, z=shift * fwd.z)


def gather_centerlines(carla_map: carla.Map, x: float, y: float, margin: float,
                        spacing: float = 2.0) -> List[List[Tuple[float, float, float]]]:
    """Return polylines for every road-segment centerline within the ROI."""
    wps = carla_map.generate_waypoints(spacing)

    # Group by (road_id, lane_id)
    from collections import defaultdict
    road_lanes: dict[tuple, list] = defaultdict(list)
    for wp in wps:
        loc = wp.transform.location
        if abs(loc.x - x) > margin or abs(loc.y - y) > margin:
            continue
        road_lanes[(wp.road_id, wp.lane_id, wp.section_id)].append(wp)

    lines = []
    for key, lane_wps in road_lanes.items():
        # Sort along s parameter
        lane_wps.sort(key=lambda w: w.s)
        pts = [(w.transform.location.x, w.transform.location.y, w.transform.location.z)
               for w in lane_wps]
        if len(pts) >= 2:
            lines.append(pts)
    return lines


def gather_road_edges(carla_map: carla.Map, x: float, y: float, margin: float,
                       spacing: float = 2.0) -> Tuple[List[List[Tuple[float, float, float]]],
                                                        List[List[Tuple[float, float, float]]]]:
    """Return left-edge and right-edge polylines for roads within the ROI."""
    wps = carla_map.generate_waypoints(spacing)

    from collections import defaultdict
    road_lanes: dict[tuple, list] = defaultdict(list)
    for wp in wps:
        loc = wp.transform.location
        if abs(loc.x - x) > margin or abs(loc.y - y) > margin:
            continue
        road_lanes[(wp.road_id, wp.lane_id, wp.section_id)].append(wp)

    left_edges, right_edges = [], []
    for key, lane_wps in road_lanes.items():
        lane_wps.sort(key=lambda w: w.s)
        left_pts, right_pts = [], []
        for w in lane_wps:
            hw = w.lane_width * 0.5
            ll = lateral_shift_loc(w.transform, -hw)
            rl = lateral_shift_loc(w.transform, hw)
            left_pts.append((ll.x, ll.y, ll.z))
            right_pts.append((rl.x, rl.y, rl.z))
        if len(left_pts) >= 2:
            left_edges.append(left_pts)
            right_edges.append(right_pts)
    return left_edges, right_edges


def parse_route_waypoints(route_xml: str) -> List[Tuple[float, float, float]]:
    """Extract (x, y, z) waypoints from a CARLA route XML file."""
    tree = ET.parse(route_xml)
    root = tree.getroot()
    pts = []
    for wp in root.iter("waypoint"):
        pts.append((float(wp.get("x", 0)), float(wp.get("y", 0)), float(wp.get("z", 0))))
    if not pts:
        # Try <position> tags
        for pos in root.iter("position"):
            pts.append((float(pos.get("x", 0)), float(pos.get("y", 0)), float(pos.get("z", 0))))
    return pts


def draw_polyline_on_image(
    draw: ImageDraw.ImageDraw,
    pts_world: List[Tuple[float, float, float]],
    cam_x: float, cam_y: float, cam_z: float,
    width: int, height: int, fov_deg: float,
    color: Tuple[int, ...] = (0, 255, 255),
    line_width: int = 2,
):
    arr = np.array(pts_world)
    px = world_to_pixel(arr, cam_x, cam_y, cam_z, width, height, fov_deg)
    coords = [(int(round(p[0])), int(round(p[1]))) for p in px
              if 0 <= p[0] < width and 0 <= p[1] < height]
    if len(coords) >= 2:
        draw.line(coords, fill=color, width=line_width)


# ── main ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--town", type=str, default="current",
                   help="CARLA map/town to load (default: current = keep loaded map). "
                        "E.g. --town ucla_v2 or --town Town05.")
    p.add_argument("--x", type=float, required=True, help="Centre X coordinate (CARLA world)")
    p.add_argument("--y", type=float, required=True, help="Centre Y coordinate (CARLA world)")
    p.add_argument("--margin", type=float, default=100.0,
                   help="Half-extent of the bounding box in metres (default: 100)")
    p.add_argument("--width", type=int, default=2048, help="Image width in pixels")
    p.add_argument("--height", type=int, default=2048, help="Image height in pixels")
    p.add_argument("--fov", type=float, default=90.0, help="Camera FOV in degrees")
    p.add_argument("--route-xml", type=str, default=None,
                   help="Optional route XML to overlay route waypoints")
    p.add_argument("--no-centerlines", action="store_true",
                   help="Skip drawing centerlines")
    p.add_argument("--no-road-edges", action="store_true",
                   help="Skip drawing road edge lines")
    p.add_argument("--no-screenshot", action="store_true",
                   help="Skip the rendered screenshot and produce a schematic-only BEV")
    p.add_argument("--centerline-color", type=str, default="0,255,255",
                   help="Centerline RGB colour as R,G,B (default: 0,255,255 = cyan)")
    p.add_argument("--road-edge-color", type=str, default="255,255,0",
                   help="Road edge RGB colour as R,G,B (default: 255,255,0 = yellow)")
    p.add_argument("--route-color", type=str, default="255,0,0",
                   help="Route waypoints RGB colour (default: 255,0,0 = red)")
    p.add_argument("--line-width", type=int, default=2)
    p.add_argument("--route-line-width", type=int, default=4)
    p.add_argument("--overlay-alpha", type=float, default=0.6,
                   help="Opacity of the overlay lines (0.0-1.0, default 0.6)")
    p.add_argument("-o", "--output", type=str, default=None,
                   help="Output file path (default: overhead_<x>_<y>.png)")
    return p.parse_args()


def parse_color(s: str) -> Tuple[int, int, int]:
    parts = [int(c.strip()) for c in s.split(",")]
    assert len(parts) == 3
    return tuple(parts)  # type: ignore


def main() -> None:
    args = parse_args()
    cx, cy = args.x, args.y
    margin = args.margin
    W, H = args.width, args.height
    fov = args.fov

    cl_color = parse_color(args.centerline_color)
    re_color = parse_color(args.road_edge_color)
    rt_color = parse_color(args.route_color)

    output = args.output or f"overhead_{cx:.0f}_{cy:.0f}.png"

    # ── connect ──────────────────────────────────────────────────────────
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()

    # ── load requested map ───────────────────────────────────────────────
    desired_town = args.town.strip()
    if desired_town.lower() != "current":
        current_map = ""
        try:
            current_map = world.get_map().name
        except RuntimeError:
            pass
        # Check if already on the right map
        if desired_town.lower() not in current_map.lower():
            avail = client.get_available_maps()
            # Try exact match, then substring match
            match = None
            for m in avail:
                if m.split("/")[-1].lower() == desired_town.lower():
                    match = m
                    break
            if match is None:
                for m in avail:
                    if desired_town.lower() in m.lower():
                        match = m
                        break
            if match is None:
                print(f"WARNING: town '{desired_town}' not found in available maps:")
                print("  " + "\n  ".join(avail))
                print("Continuing with currently loaded map.")
            else:
                print(f"Loading map: {match} ...")
                world = client.load_world(match)
                time.sleep(3.0)  # give UE4 time to settle
                world = client.get_world()
                print(f"Map loaded: {world.get_map().name}")

    carla_map = world.get_map()
    print(f"Connected to CARLA ({carla_map.name}) on port {args.port}")

    # ── compute camera height ────────────────────────────────────────────
    fov_rad = math.radians(max(1.0, min(fov, 175.0)))
    tan_half = math.tan(fov_rad / 2.0)
    # Camera height so that `margin` metres are visible from centre to edge
    cam_z = margin / tan_half + 20.0  # +20 m safety buffer
    cam_z = max(cam_z, 80.0)

    print(f"Centre=({cx}, {cy}), margin={margin}m, camera Z={cam_z:.1f}m")

    # ── capture screenshot (unless --no-screenshot) ──────────────────────
    if not args.no_screenshot:
        cam_transform = carla.Transform(
            carla.Location(x=cx, y=cy, z=cam_z),
            carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
        )

        # Put world in synchronous mode temporarily
        orig_settings = world.get_settings()
        new_settings = carla.WorldSettings()
        new_settings.synchronous_mode = True
        new_settings.fixed_delta_seconds = 0.05
        if hasattr(new_settings, "no_rendering_mode"):
            new_settings.no_rendering_mode = False
        world.apply_settings(new_settings)

        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(W))
        bp.set_attribute("image_size_y", str(H))
        bp.set_attribute("fov", f"{fov:.2f}")

        sensor = world.spawn_actor(bp, cam_transform)
        img_queue: queue.Queue = queue.Queue()
        sensor.listen(img_queue.put)

        # Warm-up ticks
        for _ in range(5):
            world.tick()

        try:
            raw_img = img_queue.get(timeout=5.0)
        except queue.Empty:
            raise RuntimeError("Timeout waiting for camera frame")
        finally:
            sensor.stop()
            sensor.destroy()
            world.apply_settings(orig_settings)

        arr = np.frombuffer(raw_img.raw_data, dtype=np.uint8).reshape(H, W, 4)
        base_rgb = arr[:, :, :3][:, :, ::-1].copy()  # BGRA -> RGB
        base_img = Image.fromarray(base_rgb)
        print(f"Captured {W}x{H} screenshot")
    else:
        # Blank dark canvas for schematic-only mode
        base_img = Image.new("RGB", (W, H), (30, 30, 30))
        print("Schematic-only mode (no screenshot)")

    # ── gather map data ──────────────────────────────────────────────────
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if not args.no_road_edges:
        print("Drawing road edges …")
        left_edges, right_edges = gather_road_edges(carla_map, cx, cy, margin * 1.2)
        for pts in left_edges + right_edges:
            draw_polyline_on_image(draw, pts, cx, cy, cam_z, W, H, fov,
                                   color=re_color + (int(255 * args.overlay_alpha),),
                                   line_width=args.line_width)
        print(f"  {len(left_edges) + len(right_edges)} edge polylines")

    if not args.no_centerlines:
        print("Drawing centerlines …")
        centerlines = gather_centerlines(carla_map, cx, cy, margin * 1.2)
        for pts in centerlines:
            draw_polyline_on_image(draw, pts, cx, cy, cam_z, W, H, fov,
                                   color=cl_color + (int(255 * args.overlay_alpha),),
                                   line_width=args.line_width)
        print(f"  {len(centerlines)} centerline polylines")

    # ── optional route overlay ───────────────────────────────────────────
    if args.route_xml:
        print(f"Drawing route from {args.route_xml} …")
        route_pts = parse_route_waypoints(args.route_xml)
        if route_pts:
            draw_polyline_on_image(draw, route_pts, cx, cy, cam_z, W, H, fov,
                                   color=rt_color + (255,),
                                   line_width=args.route_line_width)
            # Also draw circles at each route waypoint
            arr_pts = np.array(route_pts)
            px = world_to_pixel(arr_pts, cx, cy, cam_z, W, H, fov)
            r = max(3, args.route_line_width + 1)
            for i, (px_x, px_y) in enumerate(px):
                if 0 <= px_x < W and 0 <= px_y < H:
                    ix, iy = int(round(px_x)), int(round(px_y))
                    draw.ellipse([ix - r, iy - r, ix + r, iy + r],
                                 fill=rt_color + (255,))
            print(f"  {len(route_pts)} route waypoints")
        else:
            print("  WARNING: no waypoints found in route XML")

    # ── composite and save ───────────────────────────────────────────────
    base_img = base_img.convert("RGBA")
    result = Image.alpha_composite(base_img, overlay).convert("RGB")
    result.save(output)
    print(f"Saved → {output}")


if __name__ == "__main__":
    main()
