#!/usr/bin/env python3
"""
Capture a true CARLA-rendered overhead image around one ego route and overlay XY grid boxes.

Example:
  conda run -n colmdrivermarco2 python tools/render_ego_region_grid.py \
    --routes-dir /data2/marco/CoLMDriver/v2xpnp/dataset/train1/2023-03-17-15-53-02_1_0/carla_log_export \
    --ego-number 1 \
    --carla-port 2005
"""

from __future__ import annotations

import argparse
import math
import queue
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Sequence, Tuple

def _import_carla():
    try:
        import carla as _carla  # type: ignore

        return _carla
    except Exception as primary_exc:
        repo_root = Path(__file__).resolve().parents[1]
        dist_dir = repo_root / "carla912" / "PythonAPI" / "carla" / "dist"
        egg_candidates = sorted(dist_dir.glob("carla-*.egg"))
        for egg in egg_candidates:
            egg_str = str(egg)
            if egg_str not in sys.path:
                sys.path.insert(0, egg_str)
            try:
                import carla as _carla  # type: ignore

                return _carla
            except Exception:
                continue
        raise SystemExit(
            "carla Python module is required for rendered capture. "
            f"Tried default import and eggs under {dist_dir}. Last error: {primary_exc}"
        )


carla = _import_carla()

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FuncFormatter
except Exception as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required. Install with: pip install matplotlib") from exc

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise SystemExit("numpy is required. Install with: pip install numpy") from exc


Point2D = Tuple[float, float]
Bounds2D = Tuple[float, float, float, float]


def _load_ego_route_xy(xml_path: Path) -> List[Point2D]:
    root = ET.parse(xml_path).getroot()
    route = root.find("route")
    if route is None:
        raise RuntimeError(f"No <route> found in {xml_path}")
    pts: List[Point2D] = []
    for wp in route.findall("waypoint"):
        pts.append((float(wp.attrib["x"]), float(wp.attrib["y"])))
    if not pts:
        raise RuntimeError(f"No waypoints found in {xml_path}")
    return pts


def _bounds_from_points(points: Sequence[Point2D]) -> Bounds2D:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), max(xs), min(ys), max(ys))


def _expand_bounds(bounds: Bounds2D, margin: float) -> Bounds2D:
    min_x, max_x, min_y, max_y = bounds
    return (min_x - margin, max_x + margin, min_y - margin, max_y + margin)


def _ticks(start: float, end: float, step: float) -> List[float]:
    lo = math.floor(start / step) * step
    hi = math.ceil(end / step) * step
    out: List[float] = []
    val = lo
    while val <= hi + 1e-9:
        out.append(round(val, 6))
        val += step
    return out


def _resolve_ego_xml(routes_dir: Path, ego_number_one_based: int) -> Path:
    if ego_number_one_based <= 0:
        raise ValueError("--ego-number must be >= 1")
    ego_zero_based = ego_number_one_based - 1
    xml_path = routes_dir / f"ucla_v2_custom_ego_vehicle_{ego_zero_based}.xml"
    if not xml_path.exists():
        raise FileNotFoundError(f"Ego XML not found: {xml_path}")
    return xml_path


def _capture_topdown_rgb(
    *,
    host: str,
    port: int,
    bounds: Bounds2D,
    image_w: int,
    image_h: int,
    fov_deg: float,
    expected_town_substr: str,
    map_ground_z: float = 0.0,
) -> Tuple[np.ndarray, Bounds2D]:
    client = carla.Client(host, int(port))
    client.set_timeout(20.0)

    world = client.get_world()
    cmap = world.get_map()
    if expected_town_substr and expected_town_substr not in str(cmap.name or ""):
        candidates = [m for m in client.get_available_maps() if expected_town_substr in m]
        if candidates:
            world = client.load_world(candidates[0])
            cmap = world.get_map()
        else:
            raise RuntimeError(
                f"Current map '{cmap.name}' does not match '{expected_town_substr}' and no candidate map found."
            )

    min_x, max_x, min_y, max_y = bounds
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    span_x = max(5.0, max_x - min_x)
    span_y = max(5.0, max_y - min_y)

    # Compute camera height to fully cover requested bounds.
    hfov = math.radians(float(fov_deg))
    vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * (float(image_h) / float(image_w)))
    alt_x = span_x / (2.0 * math.tan(hfov / 2.0))
    alt_y = span_y / (2.0 * math.tan(vfov / 2.0))

    # Ground estimate at route center, fallback to provided ground z.
    ground_z = float(map_ground_z)
    try:
        center_wp = cmap.get_waypoint(
            carla.Location(x=float(cx), y=float(cy), z=0.0),
            project_to_road=True,
            lane_type=carla.LaneType.Any,
        )
        if center_wp is not None:
            ground_z = float(center_wp.transform.location.z)
    except Exception:
        pass

    cam_alt = max(float(alt_x), float(alt_y)) + 2.0
    cam_z = ground_z + cam_alt
    cam_tf = carla.Transform(
        carla.Location(x=float(cx), y=float(cy), z=float(cam_z)),
        carla.Rotation(pitch=-90.0, yaw=-90.0, roll=0.0),
    )

    # World extent represented by image at ground plane for nadir camera.
    half_w = math.tan(hfov / 2.0) * cam_alt
    half_h = math.tan(vfov / 2.0) * cam_alt
    image_extent: Bounds2D = (cx - half_w, cx + half_w, cy - half_h, cy + half_h)

    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(int(image_w)))
    cam_bp.set_attribute("image_size_y", str(int(image_h)))
    cam_bp.set_attribute("fov", str(float(fov_deg)))
    cam_bp.set_attribute("sensor_tick", "0.0")
    cam_bp.set_attribute("motion_blur_intensity", "0.0")

    q: "queue.Queue[carla.Image]" = queue.Queue()
    cam_actor = None
    img_rgb: np.ndarray | None = None
    try:
        cam_actor = world.spawn_actor(cam_bp, cam_tf)
        cam_actor.listen(q.put)

        is_sync = bool(getattr(world.get_settings(), "synchronous_mode", False))
        for _ in range(30):
            if is_sync:
                world.tick()
            else:
                world.wait_for_tick(seconds=1.0)
            try:
                img = q.get(timeout=0.2)
                raw = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
                # CARLA -> BGRA, convert to RGB
                img_rgb = raw[:, :, :3][:, :, ::-1].copy()
                break
            except queue.Empty:
                continue

        if img_rgb is None:
            raise RuntimeError("Failed to capture image from CARLA top-down camera.")
    finally:
        if cam_actor is not None:
            try:
                cam_actor.stop()
            except Exception:
                pass
            try:
                cam_actor.destroy()
            except Exception:
                pass

    return img_rgb, image_extent


def _draw_grid_boxes(ax: plt.Axes, bounds: Bounds2D, box_step: float, minor_step: float) -> None:
    min_x, max_x, min_y, max_y = bounds
    major_x = _ticks(min_x, max_x, box_step)
    major_y = _ticks(min_y, max_y, box_step)

    # Cell shading for easy coordinate lookup.
    for ix in range(len(major_x) - 1):
        for iy in range(len(major_y) - 1):
            if (ix + iy) % 2 == 0:
                ax.add_patch(
                    Rectangle(
                        (major_x[ix], major_y[iy]),
                        major_x[ix + 1] - major_x[ix],
                        major_y[iy + 1] - major_y[iy],
                        facecolor="#ffffff",
                        edgecolor="none",
                        alpha=0.07,
                        zorder=2,
                    )
                )

    if minor_step > 0.0:
        for x in _ticks(min_x, max_x, minor_step):
            ax.axvline(x, color="#d6dce4", linewidth=0.5, alpha=0.7, zorder=3)
        for y in _ticks(min_y, max_y, minor_step):
            ax.axhline(y, color="#d6dce4", linewidth=0.5, alpha=0.7, zorder=3)

    for x in major_x:
        ax.axvline(x, color="#8c96a3", linewidth=1.0, alpha=0.9, zorder=4)
    for y in major_y:
        ax.axhline(y, color="#8c96a3", linewidth=1.0, alpha=0.9, zorder=4)

    ax.set_xticks(major_x)
    ax.set_yticks(major_y)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v))}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v))}"))
    ax.tick_params(axis="both", labelsize=10, top=True, right=True, labeltop=True, labelright=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture rendered CARLA overhead image with XY coordinate grid around ego route."
    )
    parser.add_argument("--routes-dir", type=Path, required=True, help="Path to carla_log_export directory.")
    parser.add_argument("--ego-number", type=int, default=1, help="Ego vehicle number, 1-based. Default: 1.")
    parser.add_argument("--margin", type=float, default=35.0, help="Margin (meters) around ego route. Default: 35.")
    parser.add_argument("--box-step", type=float, default=5.0, help="Major grid box size (meters). Default: 5.")
    parser.add_argument("--minor-step", type=float, default=1.0, help="Minor grid spacing (meters). Default: 1.")
    parser.add_argument("--carla-host", type=str, default="127.0.0.1", help="CARLA host. Default: 127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=2005, help="CARLA port. Default: 2005")
    parser.add_argument(
        "--expected-town",
        type=str,
        default="ucla_v2",
        help="Expected map name substring. Script loads matching map if needed.",
    )
    parser.add_argument("--width", type=int, default=2304, help="Capture width in pixels. Default: 2304")
    parser.add_argument("--height", type=int, default=2304, help="Capture height in pixels. Default: 2304")
    parser.add_argument("--fov", type=float, default=70.0, help="Camera horizontal FOV deg. Default: 70")
    parser.add_argument("--dpi", type=int, default=220, help="Output DPI. Default: 220")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Default: <routes-dir>/ego_<n>_region_grid_rendered.png",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    routes_dir = args.routes_dir.expanduser().resolve()
    if not routes_dir.exists():
        raise FileNotFoundError(f"Routes dir not found: {routes_dir}")

    ego_xml = _resolve_ego_xml(routes_dir, int(args.ego_number))
    ego_xy = _load_ego_route_xy(ego_xml)
    zoom_bounds = _expand_bounds(_bounds_from_points(ego_xy), float(args.margin))

    image_rgb, image_extent = _capture_topdown_rgb(
        host=str(args.carla_host),
        port=int(args.carla_port),
        bounds=zoom_bounds,
        image_w=int(args.width),
        image_h=int(args.height),
        fov_deg=float(args.fov),
        expected_town_substr=str(args.expected_town),
    )

    out_path = args.output
    if out_path is None:
        out_path = routes_dir / f"ego_{int(args.ego_number)}_region_grid_rendered.png"
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect("equal", adjustable="box")
    # Rendered image background (actual simulator frame)
    ax.imshow(
        image_rgb,
        extent=(image_extent[0], image_extent[1], image_extent[2], image_extent[3]),
        origin="lower",
        zorder=1,
    )

    _draw_grid_boxes(
        ax=ax,
        bounds=zoom_bounds,
        box_step=float(args.box_step),
        minor_step=float(args.minor_step),
    )

    route_x = [p[0] for p in ego_xy]
    route_y = [p[1] for p in ego_xy]
    ax.plot(route_x, route_y, color="#ff2d2d", linewidth=2.7, alpha=0.95, zorder=5, label="ego route")
    ax.scatter([route_x[0]], [route_y[0]], color="#00b894", s=55, zorder=6, label="start")
    ax.scatter([route_x[-1]], [route_y[-1]], color="#1f3a93", s=55, zorder=6, label="end")

    min_x, max_x, min_y, max_y = zoom_bounds
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_xlabel("X (CARLA world meters)")
    ax.set_ylabel("Y (CARLA world meters)")
    ax.set_title(
        f"Rendered overhead (ucla_v2) for ego #{int(args.ego_number)} | "
        f"margin={float(args.margin):.1f}m, box={float(args.box_step):.1f}m"
    )
    ax.legend(loc="upper right", frameon=True)
    ax.text(
        0.01,
        0.01,
        f"x:[{min_x:.1f}, {max_x:.1f}]  y:[{min_y:.1f}, {max_y:.1f}]",
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#999999"},
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=int(args.dpi))
    plt.close(fig)
    print(f"[OK] Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
