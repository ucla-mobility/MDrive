#!/usr/bin/env python3
"""
Visualize all 209 diverse scenarios: one PNG per category (11 total).

Each PNG is a grid of subplots (one per scenario in that category).
Each subplot shows:
  - Ego routes as colored polylines with arrow heads for direction
  - Actors/props as colored markers (triangles for walkers, squares for static,
    diamonds for parked vehicles, pentagons for NPC vehicles)
  - Town name and scenario seed in the title
  - Light road context from the birdview cache (where available)
"""

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────

SRC_DIR = Path("/data2/marco/CoLMDriver/debug_runs_diverse_batch2")
OUT_DIR = Path("/data2/marco/CoLMDriver/debug_runs_diverse_batch2")
BEV_CACHE = Path("/data2/marco/CoLMDriver/birdview_v2_cache")
PX_PER_METER = 5
MARGIN = 300  # MAP_BOUNDARY_MARGIN used when generating the cache

# Map of town -> (min_x_world, min_y_world) computed from the birdview cache
# These are derived from the map waypoints minus MARGIN.
# We'll compute them on the fly from the .npy filename / known map extents.

# Ego route color palette (distinct colors for multiple egos)
EGO_COLORS = [
    "#2196F3",  # blue
    "#FF5722",  # deep orange
    "#4CAF50",  # green
    "#9C27B0",  # purple
    "#FF9800",  # orange
    "#00BCD4",  # cyan
    "#E91E63",  # pink
    "#795548",  # brown
]

ACTOR_MARKERS = {
    "pedestrian": ("^", "#E91E63", 60, "Pedestrian"),   # triangle, pink
    "static": ("s", "#607D8B", 50, "Static prop"),       # square, grey-blue
    "npc": ("D", "#FF9800", 55, "NPC vehicle"),          # diamond, orange
    "bicycle": ("p", "#8BC34A", 55, "Bicycle"),          # pentagon, green
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _parse_waypoints_from_xml(xml_path: Path) -> List[Tuple[float, float]]:
    """Parse (x, y) waypoints from a route XML file."""
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return []
    pts = []
    for wp in root.iter("waypoint"):
        try:
            x = float(wp.attrib.get("x", "nan"))
            y = float(wp.attrib.get("y", "nan"))
            if math.isfinite(x) and math.isfinite(y):
                pts.append((x, y))
        except (ValueError, TypeError):
            continue
    return pts


def _load_bev_image(town: str) -> Optional[Tuple[np.ndarray, float, float]]:
    """
    Load the birdview cache image for a town.
    Returns (image_rgb, min_x_world, min_y_world) or None.
    The image pixel (px, py) corresponds to world coords:
      world_x = min_x_world + px / PX_PER_METER
      world_y = min_y_world + py / PX_PER_METER
    """
    import glob

    pattern = str(BEV_CACHE / f"{town}__px_per_meter={PX_PER_METER}__*.npy")
    files = glob.glob(pattern)
    if not files:
        return None

    data = np.load(files[0])  # shape: (3, H, W), dtype uint8
    # Transpose to (H, W, 3) for display
    img = np.transpose(data, (1, 2, 0))
    # We need to know the world offset. The map boundaries are:
    # min_x = min(waypoint_x) - MARGIN
    # min_y = min(waypoint_y) - MARGIN
    # We can infer them from known town extents or we'll just use
    # the birdview with the mapping: world_to_pixel
    # For now, we'll compute them from known map data.
    return img


# Pre-computed world origins for each town's birdview cache.
# Calibrated by matching route waypoints to road pixels in the BEV cache.
# pixel = (world - origin) * PX_PER_METER
TOWN_WORLD_ORIGINS: Dict[str, Tuple[float, float]] = {
    "Town01": (-570.0, -304.8),
    "Town02": (-307.6, -262.4),
    "Town05": (-518.0, -510.0),
    "Town06": (-580.0, -551.5),
}


def _get_town_origin(town: str) -> Tuple[float, float]:
    """Get the world origin (min_x, min_y) for a town's birdview."""
    return TOWN_WORLD_ORIGINS.get(town, (-300.0, -300.0))


def _world_to_pixel(x: float, y: float, origin_x: float, origin_y: float) -> Tuple[int, int]:
    """Convert world coordinates to pixel coordinates in the birdview."""
    px = int(PX_PER_METER * (x - origin_x))
    py = int(PX_PER_METER * (y - origin_y))
    return px, py


def _collect_scenario_data(run_dir: Path) -> dict:
    """Collect all route and actor data for a single scenario."""
    result = {
        "run_name": run_dir.name,
        "town": "Town05",
        "seed": None,
        "category": None,
        "ego_routes": [],       # List of [(x,y), ...] per ego
        "ego_speeds": [],       # speed per ego
        "actors": [],           # List of {kind, positions: [(x,y),...], model}
    }

    # Summary
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            s = json.load(f)
        result["town"] = s.get("town", "Town05")
        result["seed"] = s.get("seed")
        result["category"] = s.get("category")

    # Actors manifest
    manifest_path = run_dir / "09_routes" / "routes" / "actors_manifest.json"
    if not manifest_path.exists():
        return result

    with open(manifest_path) as f:
        manifest = json.load(f)

    routes_dir = run_dir / "09_routes" / "routes"

    # Ego routes
    for ego in manifest.get("ego", []):
        xml_path = routes_dir / ego["file"]
        wps = _parse_waypoints_from_xml(xml_path)
        result["ego_routes"].append(wps)
        result["ego_speeds"].append(ego.get("speed", 0.0))

    # Actors (pedestrian, static, npc, bicycle)
    for kind in ["pedestrian", "static", "npc", "bicycle"]:
        for actor in manifest.get(kind, []):
            xml_path = routes_dir / actor["file"]
            wps = _parse_waypoints_from_xml(xml_path)
            result["actors"].append({
                "kind": kind,
                "positions": wps,
                "model": actor.get("model", ""),
                "name": actor.get("name", ""),
            })

    return result


def _plot_scenario(ax: plt.Axes, data: dict, bev_images: dict, idx: int, run_name: str = ""):
    """Plot a single scenario on an axes."""
    town = data["town"]
    ego_routes = data["ego_routes"]
    actors = data["actors"]

    if not ego_routes or all(len(r) == 0 for r in ego_routes):
        ax.text(0.5, 0.5, "No routes", ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color="gray")
        ax.set_title(run_name or f"s{data['seed']}", fontsize=5, pad=2)
        return

    # Gather all points to compute bounds
    all_x, all_y = [], []
    for route in ego_routes:
        for x, y in route:
            all_x.append(x)
            all_y.append(y)
    for actor in actors:
        for x, y in actor["positions"]:
            all_x.append(x)
            all_y.append(y)

    if not all_x:
        return

    pad = 15.0
    x_min, x_max = min(all_x) - pad, max(all_x) + pad
    y_min, y_max = min(all_y) - pad, max(all_y) + pad

    # Make square aspect
    dx = x_max - x_min
    dy = y_max - y_min
    if dx > dy:
        mid_y = (y_min + y_max) / 2
        y_min = mid_y - dx / 2
        y_max = mid_y + dx / 2
    else:
        mid_x = (x_min + x_max) / 2
        x_min = mid_x - dy / 2
        x_max = mid_x + dy / 2

    # Draw birdview road mask as background
    if town in bev_images and bev_images[town] is not None:
        bev_img = bev_images[town]  # binary mask (H, W, 3) with values 0/1
        origin = _get_town_origin(town)
        # Convert world bounds to pixel bounds
        px_min, py_min = _world_to_pixel(x_min, y_min, origin[0], origin[1])
        px_max, py_max = _world_to_pixel(x_max, y_max, origin[0], origin[1])
        # Clamp to image bounds
        h, w = bev_img.shape[:2]
        px_min_c = max(0, min(px_min, w - 1))
        px_max_c = max(0, min(px_max, w - 1))
        py_min_c = max(0, min(py_min, h - 1))
        py_max_c = max(0, min(py_max, h - 1))

        if px_max_c > px_min_c and py_max_c > py_min_c:
            crop = bev_img[py_min_c:py_max_c, px_min_c:px_max_c]
            # Create a road overlay: road pixels in light gray, background white
            road_any = crop.any(axis=2)  # boolean mask
            rgb = np.full((*crop.shape[:2], 3), 245, dtype=np.uint8)  # white bg
            rgb[road_any] = [210, 210, 210]  # light gray roads
            # Map pixel bounds back to world for extent
            wx_min = origin[0] + px_min_c / PX_PER_METER
            wx_max = origin[0] + px_max_c / PX_PER_METER
            wy_min = origin[1] + py_min_c / PX_PER_METER
            wy_max = origin[1] + py_max_c / PX_PER_METER
            ax.imshow(rgb, extent=[wx_min, wx_max, wy_max, wy_min],
                      aspect="equal", alpha=0.8, interpolation="nearest")

    # Set background
    ax.set_facecolor("#F5F5F5")

    # Plot ego routes
    for i, route in enumerate(ego_routes):
        if len(route) < 1:
            continue
        color = EGO_COLORS[i % len(EGO_COLORS)]
        xs = [p[0] for p in route]
        ys = [p[1] for p in route]

        ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.85, zorder=3)

        # Start marker (circle)
        ax.plot(xs[0], ys[0], "o", color=color, markersize=4, zorder=4)

        # Direction arrow at midpoint
        if len(route) >= 2:
            mid = len(route) // 2
            dx_arrow = xs[min(mid + 1, len(xs) - 1)] - xs[max(mid - 1, 0)]
            dy_arrow = ys[min(mid + 1, len(ys) - 1)] - ys[max(mid - 1, 0)]
            norm = math.hypot(dx_arrow, dy_arrow)
            if norm > 1e-6:
                dx_arrow /= norm
                dy_arrow /= norm
                ax.annotate(
                    "",
                    xy=(xs[mid] + dx_arrow * 2, ys[mid] + dy_arrow * 2),
                    xytext=(xs[mid], ys[mid]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                    zorder=5,
                )

        # End marker (arrow tip is already there, add small x)
        ax.plot(xs[-1], ys[-1], "x", color=color, markersize=4, markeredgewidth=1.5, zorder=4)

    # Plot actors
    for actor in actors:
        kind = actor["kind"]
        marker_info = ACTOR_MARKERS.get(kind, ("o", "#999999", 40, kind))
        marker, color, size, label = marker_info

        for x, y in actor["positions"]:
            ax.scatter(x, y, marker=marker, c=color, s=size, edgecolors="white",
                       linewidths=0.5, zorder=6, alpha=0.9)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # Invert Y to match CARLA convention (y increases downward in vis)
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Title: use folder name for easy cross-referencing
    n_egos = len(ego_routes)
    n_actors = len(actors)
    if run_name:
        # Strip category suffix for brevity – keep timestamp+seed prefix
        # e.g. "20260227_015140_957369_..._s2" → show full name but small font
        title = run_name
    else:
        title = f"s{data['seed']} | {town} | {n_egos}E"
        if n_actors > 0:
            title += f" {n_actors}A"
    ax.set_title(title, fontsize=4.5, pad=2, fontweight="bold")


def main():
    src = SRC_DIR
    out = OUT_DIR

    # Discover runs by category
    print(f"Scanning {src} ...")
    runs_by_cat: Dict[str, List[Path]] = {}
    for d in sorted(src.iterdir()):
        if d.name.startswith("_"):
            continue
        summary_path = d / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            s = json.load(f)
        cat = s.get("category", "unknown")
        runs_by_cat.setdefault(cat, []).append(d)

    print(f"Found {sum(len(v) for v in runs_by_cat.values())} runs in {len(runs_by_cat)} categories")

    # Pre-load birdview images
    print("Loading birdview maps ...")
    bev_images: Dict[str, Optional[np.ndarray]] = {}
    for town in ["Town01", "Town02", "Town03", "Town05", "Town06"]:
        import glob
        pattern = str(BEV_CACHE / f"{town}__px_per_meter={PX_PER_METER}__*.npy")
        files = glob.glob(pattern)
        if files:
            data = np.load(files[0])
            bev_images[town] = np.transpose(data, (1, 2, 0))
            print(f"  {town}: loaded {bev_images[town].shape}")
        else:
            bev_images[town] = None
            print(f"  {town}: no cache found")

    # Generate one PNG per category
    out.mkdir(parents=True, exist_ok=True)

    for cat in sorted(runs_by_cat.keys()):
        run_dirs = sorted(runs_by_cat[cat])
        n = len(run_dirs)
        print(f"\nCategory: {cat} ({n} scenarios)")

        # Collect all data
        all_data = []
        for rd in run_dirs:
            all_data.append(_collect_scenario_data(rd))

        # Grid layout: aim for roughly square
        ncols = min(5, n)
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 3.2))
        fig.suptitle(f"{cat} ({n} scenarios)", fontsize=14, fontweight="bold", y=0.98)

        # Flatten axes array
        if nrows == 1 and ncols == 1:
            axes_flat = [axes]
        elif nrows == 1 or ncols == 1:
            axes_flat = list(axes)
        else:
            axes_flat = [ax for row in axes for ax in row]

        for i, (data, rd) in enumerate(zip(all_data, run_dirs)):
            _plot_scenario(axes_flat[i], data, bev_images, i, run_name=rd.name)

        # Hide unused axes
        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)

        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor=EGO_COLORS[0], label="Ego routes"),
        ]
        for kind, (marker, color, size, label) in ACTOR_MARKERS.items():
            legend_elements.append(
                plt.scatter([], [], marker=marker, c=color, s=30, label=label,
                            edgecolors="white", linewidths=0.3)
            )
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=5,
            fontsize=8,
            frameon=True,
            fancybox=True,
            shadow=False,
        )

        plt.tight_layout(rect=[0, 0.04, 1, 0.96])

        # Save
        safe_name = cat.replace("/", "_").replace(" ", "_")
        out_path = out / f"viz_{safe_name}.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    print(f"\nDone! {len(runs_by_cat)} PNGs saved to {out}")


if __name__ == "__main__":
    main()
