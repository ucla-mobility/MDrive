"""Render per-scenario PNG and animated GIF visualizations of a converted
scenarioset directory.

Each scenario gets:
  <out>/<scenario>.png     -- BEV with all actor trajectories drawn over lane polylines
  <out>/<scenario>.gif     -- 10 Hz animation of the run (optional, --gif)

Trajectories are color-coded by actor kind:
  ego: thick blue, with start (green dot) and end (red square)
  npc: thin per-actor color (HSV)
  walker: thin green dashed
  static: orange dot

If a spawn-validation JSON is supplied (--spawn-json), actors that failed
to spawn are highlighted with a red X at their first waypoint.

Usage:
    python3 -m tools.v2xpnp_visualize_scenarioset \
        --scenarios /tmp/scenarioset_full \
        --map-pkl v2xpnp/map/v2x_intersection_vector_map.pkl \
        --out /tmp/scenarioset_full_viz \
        --spawn-json /tmp/eval_full/spawn_v3.json \
        --gif
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib as _mpl
_mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors as mcolors


def _load_lane_polylines(map_pkl: Path) -> List[List[Tuple[float, float]]]:
    """Load lane polylines from one of the v2xpnp map pickles. The pickles
    are dicts that contain a 'lines' key (list of dicts with 'polyline')."""
    obj = pickle.load(open(map_pkl, "rb"))
    polys: List[List[Tuple[float, float]]] = []
    lines = []
    if isinstance(obj, dict):
        lines = obj.get("lines") or obj.get("polylines") or []
    for line in lines:
        if isinstance(line, dict):
            pts = line.get("polyline") or line.get("points") or []
        else:
            pts = line
        seq: List[Tuple[float, float]] = []
        for p in pts:
            try:
                seq.append((float(p[0]), float(p[1])))
            except (TypeError, ValueError, IndexError):
                continue
        if len(seq) >= 2:
            polys.append(seq)
    return polys


def _load_xml_waypoints(xml_path: Path) -> List[Tuple[float, float]]:
    try:
        root = ET.parse(str(xml_path)).getroot()
    except ET.ParseError:
        return []
    out = []
    for w in root.findall(".//waypoint"):
        try:
            out.append((float(w.attrib["x"]), float(w.attrib["y"])))
        except (KeyError, ValueError):
            continue
    return out


def _hsv_color(i: int, n: int) -> Tuple[float, float, float]:
    h = (i / max(1, n)) % 1.0
    return mcolors.hsv_to_rgb((h, 0.7, 0.85))


def render_scenario_png(
    scen_dir: Path,
    out_png: Path,
    lane_polys: Optional[List[List[Tuple[float, float]]]] = None,
    failed_set: Optional[set] = None,
    transform: Optional[Dict[str, float]] = None,
) -> bool:
    """Render a single scenario to PNG. Returns True on success."""
    manifest_path = scen_dir / "actors_manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())

    egos = manifest.get("ego", [])
    npcs = manifest.get("npc", [])
    walkers = manifest.get("walker", [])
    statics = manifest.get("static", [])

    fig, ax = plt.subplots(figsize=(14, 14))

    # Lane polylines (gray) -- transform from PKL frame to CARLA frame if needed
    if lane_polys:
        for poly in lane_polys:
            xs, ys = zip(*poly)
            if transform is not None:
                xs2, ys2 = [], []
                for x, y in zip(xs, ys):
                    nx = x + (-transform["tx"])
                    ny = (-y if transform["flip_y"] else y) + (
                        transform["ty"] if transform["flip_y"] else -transform["ty"]
                    )
                    xs2.append(nx); ys2.append(ny)
                xs, ys = xs2, ys2
            ax.plot(xs, ys, color="#999999", linewidth=0.6, zorder=1)

    # Static actors (orange dots)
    for e in statics:
        wps = _load_xml_waypoints(scen_dir / e["file"])
        if not wps:
            continue
        x, y = wps[0]
        is_failed = failed_set and (("static", str(e.get("name", "?"))) in failed_set)
        ax.plot(
            x, y,
            marker="s", color="#ff8800" if not is_failed else "#ff0000",
            markersize=5 if not is_failed else 9,
            markeredgecolor="black", markeredgewidth=0.4, zorder=4,
        )

    # NPCs (per-actor color, thin line)
    n_npcs = len(npcs)
    for i, e in enumerate(npcs):
        wps = _load_xml_waypoints(scen_dir / e["file"])
        if len(wps) < 2:
            continue
        xs, ys = zip(*wps)
        color = _hsv_color(i, n_npcs)
        ax.plot(xs, ys, color=color, linewidth=1.0, alpha=0.85, zorder=3)
        # mark spawn point with small dot
        is_failed = failed_set and (("npc", str(e.get("name", "?"))) in failed_set)
        if is_failed:
            ax.plot(xs[0], ys[0], marker="x", color="red", markersize=14, markeredgewidth=2.5, zorder=10)
            ax.plot(xs[0], ys[0], marker="o", color="red", markersize=8, zorder=9, alpha=0.4)
        else:
            ax.plot(xs[0], ys[0], marker="o", color=color, markersize=4, zorder=4)

    # Walkers (green dashed)
    for e in walkers:
        wps = _load_xml_waypoints(scen_dir / e["file"])
        if len(wps) < 2:
            continue
        xs, ys = zip(*wps)
        ax.plot(xs, ys, color="#22aa22", linewidth=0.5, linestyle="--", alpha=0.6, zorder=2)

    # Egos last (top, thick blue)
    for i, e in enumerate(egos):
        wps = _load_xml_waypoints(scen_dir / e["file"])
        if len(wps) < 2:
            continue
        xs, ys = zip(*wps)
        ego_color = "#0033ff" if i == 0 else "#0099ff"
        ax.plot(xs, ys, color=ego_color, linewidth=3.0, alpha=0.95, zorder=5,
                label=f"ego_{i}")
        ax.plot(xs[0], ys[0], marker="^", color="lime", markersize=14,
                markeredgecolor="black", markeredgewidth=1.0, zorder=6)
        ax.plot(xs[-1], ys[-1], marker="s", color="red", markersize=12,
                markeredgecolor="black", markeredgewidth=1.0, zorder=6)

    # Bounds: union of ego bounds + 80m padding
    all_xs: List[float] = []
    all_ys: List[float] = []
    for e in egos + npcs:
        wps = _load_xml_waypoints(scen_dir / e["file"])
        for x, y in wps:
            all_xs.append(x); all_ys.append(y)
    if all_xs and all_ys:
        pad = 60.0
        ax.set_xlim(min(all_xs) - pad, max(all_xs) + pad)
        ax.set_ylim(min(all_ys) - pad, max(all_ys) + pad)

    ax.set_aspect("equal")
    ax.set_xlabel("x (CARLA frame, m)")
    ax.set_ylabel("y (CARLA frame, m)")
    ax.grid(True, alpha=0.25)

    n_ego = len(egos); n_npc = len(npcs); n_walker = len(walkers); n_static = len(statics)
    n_failed = (
        sum(1 for f in failed_set if f[0] in ("npc", "static")) if failed_set else 0
    )
    title = (
        f"{scen_dir.name}\n"
        f"ego={n_ego}  npc={n_npc}  walker={n_walker}  static={n_static}"
        f"  spawn-failed={n_failed}"
    )
    ax.set_title(title, fontsize=11)

    legend_handles = [
        mpatches.Patch(color="#0033ff", label="ego"),
        plt.Line2D([], [], marker="^", color="lime", linestyle="None", label="ego start"),
        plt.Line2D([], [], marker="s", color="red", linestyle="None", label="ego end"),
        mpatches.Patch(color="#888", label="lanes"),
        plt.Line2D([], [], color="#22aa22", linestyle="--", label="walkers"),
        plt.Line2D([], [], marker="s", color="#ff8800", linestyle="None", label="static"),
        plt.Line2D([], [], marker="x", color="red", linestyle="None",
                   markersize=10, markeredgewidth=2, label="spawn failed"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.85)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def render_scenario_gif(
    scen_dir: Path,
    out_gif: Path,
    lane_polys: Optional[List[List[Tuple[float, float]]]] = None,
    failed_set: Optional[set] = None,
    fps: int = 10,
    max_frames: int = 200,
    skip_frames: int = 2,
    transform: Optional[Dict[str, float]] = None,
) -> bool:
    """Render an animated GIF of the scenario."""
    try:
        import imageio
    except ImportError:
        print("imageio not installed; skipping GIF for", scen_dir.name)
        return False

    manifest = json.loads((scen_dir / "actors_manifest.json").read_text())
    egos = manifest.get("ego", [])
    npcs = manifest.get("npc", [])
    walkers = manifest.get("walker", [])
    statics = manifest.get("static", [])

    # Pre-load all waypoints with their times
    def load_wps_with_t(path: Path) -> List[Tuple[float, float, float]]:
        try:
            root = ET.parse(str(path)).getroot()
        except ET.ParseError:
            return []
        out = []
        for w in root.findall(".//waypoint"):
            try:
                out.append((
                    float(w.attrib["x"]),
                    float(w.attrib["y"]),
                    float(w.attrib.get("time", "0")),
                ))
            except (KeyError, ValueError):
                continue
        return out

    ego_data = [load_wps_with_t(scen_dir / e["file"]) for e in egos]
    npc_data = [load_wps_with_t(scen_dir / e["file"]) for e in npcs]
    static_data = [load_wps_with_t(scen_dir / e["file"]) for e in statics]
    walker_data = [load_wps_with_t(scen_dir / e["file"]) for e in walkers]

    # Bounds
    all_xs: List[float] = []
    all_ys: List[float] = []
    for trk in ego_data + npc_data:
        for x, y, _ in trk:
            all_xs.append(x); all_ys.append(y)
    if not all_xs:
        return False
    pad = 50.0
    xlim = (min(all_xs) - pad, max(all_xs) + pad)
    ylim = (min(all_ys) - pad, max(all_ys) + pad)

    # Time axis: union of all ego time samples
    times = sorted({t for trk in ego_data for _, _, t in trk})
    times = times[::skip_frames][:max_frames]

    frames_imgs = []
    for t_now in times:
        fig, ax = plt.subplots(figsize=(10, 10))
        if lane_polys:
            for poly in lane_polys:
                xs, ys = zip(*poly)
                if transform is not None:
                    xs2, ys2 = [], []
                    for x, y in zip(xs, ys):
                        nx = x + (-transform["tx"])
                        ny = (-y if transform["flip_y"] else y) + (
                            transform["ty"] if transform["flip_y"] else -transform["ty"]
                        )
                        xs2.append(nx); ys2.append(ny)
                    xs, ys = xs2, ys2
                ax.plot(xs, ys, color="#aaa", linewidth=0.5, zorder=1)

        # Statics: just show all (pose doesn't change after frame 0)
        for trk in static_data:
            if trk:
                ax.plot(trk[0][0], trk[0][1], marker="s", color="#ff8800",
                        markersize=4, markeredgecolor="black", markeredgewidth=0.3, zorder=3)

        # Walkers
        for trk in walker_data:
            cur = next(((x, y) for x, y, t in trk if abs(t - t_now) < 0.06), None)
            if cur is None and trk:
                cur = trk[0][:2]
            if cur:
                ax.plot(cur[0], cur[1], marker="o", color="#22aa22", markersize=4, zorder=3)

        # NPCs
        for i, trk in enumerate(npc_data):
            color = _hsv_color(i, max(1, len(npc_data)))
            # Draw the path so far (faint)
            past = [(x, y) for x, y, t in trk if t <= t_now]
            if len(past) >= 2:
                xs, ys = zip(*past)
                ax.plot(xs, ys, color=color, linewidth=1.0, alpha=0.45, zorder=2)
            cur = next(((x, y) for x, y, t in trk if abs(t - t_now) < 0.06), None)
            if cur:
                ax.plot(cur[0], cur[1], marker="o", color=color,
                        markersize=7, markeredgecolor="black", markeredgewidth=0.4, zorder=4)

        # Egos
        for i, trk in enumerate(ego_data):
            ego_color = "#0033ff" if i == 0 else "#0099ff"
            past = [(x, y) for x, y, t in trk if t <= t_now]
            if len(past) >= 2:
                xs, ys = zip(*past)
                ax.plot(xs, ys, color=ego_color, linewidth=2.5, alpha=0.85, zorder=5)
            cur = next(((x, y) for x, y, t in trk if abs(t - t_now) < 0.06), None)
            if cur:
                ax.plot(cur[0], cur[1], marker="^", color=ego_color,
                        markersize=12, markeredgecolor="white", markeredgewidth=1.0, zorder=6)

        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_title(f"{scen_dir.name}  t={t_now:.1f}s", fontsize=11)
        ax.grid(True, alpha=0.2)

        fig.canvas.draw()
        # extract RGB image
        try:
            img = fig.canvas.buffer_rgba()
            import numpy as np
            arr = np.asarray(img)
            frames_imgs.append(arr[:, :, :3].copy())
        except Exception:
            pass
        plt.close(fig)

    if frames_imgs:
        out_gif.parent.mkdir(parents=True, exist_ok=True)
        # Newer imageio uses `duration` (ms per frame) not `fps`
        try:
            imageio.mimsave(str(out_gif), frames_imgs, duration=1000.0 / float(fps))
        except TypeError:
            imageio.mimsave(str(out_gif), frames_imgs, fps=fps)
        return True
    return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenarios", required=True, help="Root containing per-scenario subdirs")
    p.add_argument("--map-pkl", default=None, help="Optional v2x map pickle for lane polylines")
    p.add_argument("--map-offset-json", default=None, help="Optional ucla_map_offset_carla.json (transforms PKL→CARLA)")
    p.add_argument("--out", required=True, help="Output directory for PNGs / GIFs")
    p.add_argument("--gif", action="store_true", help="Also render animated GIFs (slower)")
    p.add_argument("--gif-fps", type=int, default=10)
    p.add_argument("--gif-skip", type=int, default=2, help="Subsample every Nth ego time step")
    p.add_argument("--spawn-json", default=None, help="Optional spawn validation JSON to mark failed actors")
    p.add_argument("--scenario", default=None, help="Render only one scenario by name")
    args = p.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    lane_polys: Optional[List[List[Tuple[float, float]]]] = None
    if args.map_pkl:
        try:
            lane_polys = _load_lane_polylines(Path(args.map_pkl))
            print(f"Loaded {len(lane_polys)} lane polylines from {args.map_pkl}")
        except Exception as exc:
            print(f"Could not load map polylines: {exc}")

    transform: Optional[Dict[str, float]] = None
    if args.map_offset_json:
        cfg = json.loads(Path(args.map_offset_json).read_text())
        transform = {
            "tx": float(cfg.get("tx", 0.0)),
            "ty": float(cfg.get("ty", 0.0)),
            "flip_y": bool(cfg.get("flip_y", False)),
        }

    failed_by_scenario: Dict[str, set] = {}
    if args.spawn_json:
        spawn_data = json.loads(Path(args.spawn_json).read_text())
        for r in spawn_data:
            failed_by_scenario[r["scenario"]] = {
                (fl["kind"], str(fl["name"])) for fl in r.get("failures", [])
            }

    scen_dirs = sorted([d for d in Path(args.scenarios).iterdir() if d.is_dir()])
    if args.scenario:
        scen_dirs = [d for d in scen_dirs if d.name == args.scenario]
    print(f"Rendering {len(scen_dirs)} scenarios → {out_dir}")

    n_done = 0
    for scen_dir in scen_dirs:
        png_path = out_dir / f"{scen_dir.name}.png"
        ok = render_scenario_png(
            scen_dir, png_path,
            lane_polys=lane_polys,
            failed_set=failed_by_scenario.get(scen_dir.name),
            transform=transform,
        )
        if ok:
            n_done += 1
            print(f"  [{n_done}/{len(scen_dirs)}] PNG {scen_dir.name}")
        if args.gif and ok:
            gif_path = out_dir / f"{scen_dir.name}.gif"
            try:
                gif_ok = render_scenario_gif(
                    scen_dir, gif_path,
                    lane_polys=lane_polys,
                    failed_set=failed_by_scenario.get(scen_dir.name),
                    fps=args.gif_fps,
                    skip_frames=args.gif_skip,
                    transform=transform,
                )
                if gif_ok:
                    print(f"      GIF {scen_dir.name}")
            except Exception as exc:
                print(f"      GIF failed for {scen_dir.name}: {exc}")

    print(f"Wrote {n_done} PNGs to {out_dir}")


if __name__ == "__main__":
    main()
