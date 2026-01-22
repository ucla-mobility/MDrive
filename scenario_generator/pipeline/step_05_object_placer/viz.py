import json
import math
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from matplotlib.lines import Line2D

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from .assets import get_asset_bbox
from .geometry import left_normal_world, right_normal_world


def visualize(
    picked: List[Dict[str, Any]],
    seg_by_id: Dict[int, np.ndarray],
    actors_world: List[Dict[str, Any]],
    crop_region: Optional[Dict[str, Any]],
    out_path: str,
    description: Optional[str] = None,
    show: bool = False,
) -> None:
    if plt is None:
        print("[WARNING] matplotlib not installed; skipping visualization")
        return

    import matplotlib.patches as mpatches
    from matplotlib.transforms import Bbox

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    # -------------------------
    # Crop region / axes config
    # -------------------------
    if crop_region and all(k in crop_region for k in ("xmin", "xmax", "ymin", "ymax")):
        xmin, xmax, ymin, ymax = crop_region["xmin"], crop_region["xmax"], crop_region["ymin"], crop_region["ymax"]
        max_range = max(float(xmax - xmin), float(ymax - ymin))
        margin_m = min(max(12.0, 0.12 * max_range), 60.0)
        ax.set_xlim(xmin - margin_m, xmax + margin_m)
        ax.set_ylim(ymin - margin_m, ymax + margin_m)
        ax.invert_xaxis()
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, linestyle="--", linewidth=2)
        ax.add_patch(rect)

    cmap = plt.cm.get_cmap("tab10")

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _offset_polyline(poly: np.ndarray, offset_m: float) -> np.ndarray:
        """Laterally offset a polyline for visual separation."""
        if abs(offset_m) < 1e-6 or poly is None or len(poly) < 2:
            return poly
        pts = np.asarray(poly, dtype=float)
        out = []
        n = len(pts)
        for k in range(n):
            if k == 0:
                t = pts[1] - pts[0]
            elif k == n - 1:
                t = pts[k] - pts[k - 1]
            else:
                t = pts[k + 1] - pts[k - 1]
            if offset_m >= 0:
                nvec = right_normal_world(t)
            else:
                nvec = left_normal_world(t)
            out.append(pts[k] + abs(offset_m) * nvec)
        return np.vstack(out)

    def _bbox_intersects(b1: Bbox, b2: Bbox) -> bool:
        return not (b1.x1 < b2.x0 or b1.x0 > b2.x1 or b1.y1 < b2.y0 or b1.y0 > b2.y1)

    def _inflate_bbox(bb: Bbox, px: float = 2.0) -> Bbox:
        return Bbox.from_extents(bb.x0 - px, bb.y0 - px, bb.x1 + px, bb.y1 + px)

    def _actor_marker(cat: str) -> str:
        cat = (cat or "").lower()
        if cat == "walker":
            return "P"
        if cat == "cyclist":
            return "D"
        if cat == "vehicle":
            return "s"
        return "x"

    def _get_asset_short(asset_id: str, fallback: str) -> str:
        asset_id = str(asset_id or "")
        if not asset_id:
            return fallback
        parts = asset_id.split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else asset_id

    def _format_distance_m(value: Any) -> Optional[str]:
        if value is None:
            return None
        try:
            dist = float(value)
        except Exception:
            return None
        if abs(dist - round(dist)) < 1e-6:
            return f"{int(round(dist))}m"
        return f"{dist:.1f}m"

    def _format_trigger(trigger: Any) -> Optional[str]:
        if not isinstance(trigger, dict):
            return None
        ttype = str(trigger.get("type", "")).strip()
        if ttype != "distance_to_vehicle":
            return None
        vehicle = str(trigger.get("vehicle", "")).strip()
        if not vehicle:
            return None
        dist_str = _format_distance_m(trigger.get("distance_m"))
        if dist_str:
            return f"trigger: {vehicle} <= {dist_str}"
        return f"trigger: {vehicle} distance"

    def _format_action(action: Any) -> Optional[str]:
        if not isinstance(action, dict):
            return None
        atype = str(action.get("type", "")).strip()
        if not atype:
            return None
        if atype == "hard_brake":
            return "action: hard_brake"
        if atype == "start_motion":
            direction = str(action.get("direction") or "").strip()
            if direction:
                return f"action: start_motion {direction}"
            return "action: start_motion"
        if atype == "lane_change":
            direction = str(action.get("direction") or "").strip()
            target_vehicle = str(action.get("target_vehicle") or "").strip()
            if direction:
                return f"action: lane_change {direction}"
            if target_vehicle:
                return f"action: lane_change -> {target_vehicle} lane"
            return "action: lane_change"
        return f"action: {atype}"

    def _format_behavior_label(a: Dict[str, Any]) -> Optional[str]:
        lines = []
        tline = _format_trigger(a.get("trigger"))
        if tline:
            lines.append(tline)
        aline = _format_action(a.get("action"))
        if aline:
            lines.append(aline)
        if not lines:
            return None
        return "\n".join(lines)

    def _behavior_signature(a: Dict[str, Any]) -> str:
        trigger = a.get("trigger")
        action = a.get("action")
        parts = []
        if isinstance(trigger, dict):
            ttype = str(trigger.get("type", "")).strip()
            vehicle = str(trigger.get("vehicle", "")).strip()
            dist_str = _format_distance_m(trigger.get("distance_m")) or ""
            parts.append(f"t:{ttype}:{vehicle}:{dist_str}")
        if isinstance(action, dict):
            atype = str(action.get("type", "")).strip()
            direction = str(action.get("direction") or "").strip()
            target_vehicle = str(action.get("target_vehicle") or "").strip()
            parts.append(f"a:{atype}:{direction}:{target_vehicle}")
        return "|".join(parts) if parts else "none"

    def _wrap_label(label: str, width: int = 32) -> str:
        return "\n".join(textwrap.fill(line, width=width) for line in label.splitlines())

    def _scene_summary_text(raw_description: Any) -> Optional[str]:
        """
        Prefer the `description` field when the provided description is a schema JSON.
        Falls back to the raw string when parsing fails or no description exists.
        """
        if raw_description is None:
            return None
        if isinstance(raw_description, dict):
            desc_val = raw_description.get("description")
            if isinstance(desc_val, str) and desc_val.strip():
                return desc_val.strip()
            raw_str = str(raw_description).strip()
            return raw_str or None

        raw_str = str(raw_description).strip()
        if not raw_str:
            return None

        try:
            parsed = json.loads(raw_str)
            if isinstance(parsed, dict):
                desc_val = parsed.get("description")
                if isinstance(desc_val, str) and desc_val.strip():
                    return desc_val.strip()
        except Exception:
            pass
        return raw_str

    def _actor_bbox_dims_from_actor_or_cache(a: Dict[str, Any]) -> Tuple[float, float]:
        """
        Returns (length,width) in meters, or (0,0) if unknown.
        Prefers a['bbox'], else falls back to get_asset_bbox(asset_id).
        """
        asset_id = str(a.get("asset_id", ""))
        actor_bbox = a.get("bbox")
        if isinstance(actor_bbox, dict):
            try:
                length = float(actor_bbox.get("length", 0.0))
                width = float(actor_bbox.get("width", 0.0))
                if length > 0.05 and width > 0.05:
                    return length, width
            except Exception:
                pass

        asset_bbox = get_asset_bbox(asset_id)
        if asset_bbox:
            return float(asset_bbox.length), float(asset_bbox.width)
        return 0.0, 0.0

    def _draw_filled_oriented_bbox(x: float, y: float, yaw_deg: float, length: float, width: float,
                                   facecolor: Any, edgecolor: Any, alpha: float, zorder: int):
        # local corners centered at origin
        half_l, half_w = length / 2.0, width / 2.0
        corners_local = np.array([
            [-half_l, -half_w],
            [ half_l, -half_w],
            [ half_l,  half_w],
            [-half_l,  half_w],
        ], dtype=float)
        yaw_rad = math.radians(float(yaw_deg))
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
        rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]], dtype=float)
        corners_world = corners_local @ rot.T + np.array([x, y], dtype=float)

        poly = mpatches.Polygon(
            corners_world,
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.3,
            alpha=alpha,
            zorder=zorder,
        )
        ax.add_patch(poly)
        return poly

    def _collect_forbidden_bboxes(fig, ax, artists, pad_px: float = 3.0) -> List[Bbox]:
        """
        Collect screen-space bboxes for "things labels should not overlap":
        - ego path lines
        - motion lines
        - bbox patches
        """
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        forb: List[Bbox] = []
        for art in artists:
            try:
                bb = art.get_window_extent(renderer=renderer)
                if bb is not None:
                    forb.append(_inflate_bbox(bb, px=pad_px))
            except Exception:
                continue
        return forb

    def _place_labels_repel(ax, fig, items, forbidden_bboxes, fontsize=8, crop_region: Optional[Dict[str, Any]] = None):
        """
        Greedy label placement that avoids:
          - other labels
          - forbidden_bboxes (paths, motion polylines, bbox patches)
        Adds leader lines.
        """
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        placed_bboxes: List[Bbox] = []
        used_annotations = []

        # Offset candidates: try lots, progressively farther away.
        offsets = []
        for r in (10, 16, 24, 34, 46, 60, 78, 98):
            offsets += [( r, 0), (-r, 0), (0, r), (0, -r)]
            offsets += [( r, r), ( r, -r), (-r, r), (-r, -r)]
            offsets += [( int(r*0.9), int(r*0.4)), ( int(r*0.9), -int(r*0.4)),
                        (-int(r*0.9), int(r*0.4)), (-int(r*0.9), -int(r*0.4))]
            offsets += [( int(r*0.4), int(r*0.9)), (-int(r*0.4), int(r*0.9)),
                        ( int(r*0.4), -int(r*0.9)), (-int(r*0.4), -int(r*0.9))]

        def ok_bbox(bb: Bbox) -> bool:
            for prev in placed_bboxes:
                if _bbox_intersects(bb, prev):
                    return False
            for fb in forbidden_bboxes:
                if _bbox_intersects(bb, fb):
                    return False
            return True

        def _whitespace_candidates(x: float, y: float) -> List[Tuple[float, float]]:
            if not crop_region or not all(k in crop_region for k in ("xmin", "xmax", "ymin", "ymax")):
                return []

            xmin, xmax = float(crop_region["xmin"]), float(crop_region["xmax"])
            ymin, ymax = float(crop_region["ymin"]), float(crop_region["ymax"])
            ax_xmin, ax_xmax = sorted(ax.get_xlim())
            ax_ymin, ax_ymax = sorted(ax.get_ylim())

            max_range = max(xmax - xmin, ymax - ymin)
            pad_m = max(1.5, 0.02 * max_range)

            margins = {
                "left": xmin - ax_xmin,
                "right": ax_xmax - xmax,
                "bottom": ymin - ax_ymin,
                "top": ax_ymax - ymax,
            }
            side_order = sorted(margins.items(), key=lambda kv: kv[1], reverse=True)
            rail_offsets = [0.0, 4.0, -4.0, 8.0, -8.0, 12.0, -12.0]

            positions: List[Tuple[float, float]] = []
            for side, margin in side_order:
                if margin <= pad_m * 1.2:
                    continue
                if side in ("left", "right"):
                    base_x = xmin - pad_m if side == "left" else xmax + pad_m
                    base_x = min(max(base_x, ax_xmin + pad_m), ax_xmax - pad_m)
                    for dy in rail_offsets:
                        y_text = float(np.clip(y + dy, ax_ymin + pad_m, ax_ymax - pad_m))
                        positions.append((base_x, y_text))
                else:
                    base_y = ymin - pad_m if side == "bottom" else ymax + pad_m
                    base_y = min(max(base_y, ax_ymin + pad_m), ax_ymax - pad_m)
                    for dx in rail_offsets:
                        x_text = float(np.clip(x + dx, ax_xmin + pad_m, ax_xmax - pad_m))
                        positions.append((x_text, base_y))

            return positions

        for it in items:
            x, y = float(it["x"]), float(it["y"])
            label = str(it["label"])
            color = it.get("color", "black")
            zorder = int(it.get("zorder", 9))

            placed = False
            whitespace_positions = _whitespace_candidates(x, y)
            for (tx, ty) in whitespace_positions:
                ann = ax.annotate(
                    label,
                    xy=(x, y),
                    xytext=(tx, ty),
                    textcoords="data",
                    fontsize=fontsize,
                    color=color,
                    zorder=zorder,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.88),
                    arrowprops=dict(arrowstyle="-", lw=0.8, color=color, alpha=0.55),
                    annotation_clip=False,
                )
                bb = ann.get_window_extent(renderer=renderer)
                bb = _inflate_bbox(bb, px=2.0)
                if ok_bbox(bb):
                    placed_bboxes.append(bb)
                    used_annotations.append(ann)
                    placed = True
                    break
                ann.remove()

            if not placed:
                for (dx, dy) in offsets:
                    ann = ax.annotate(
                        label,
                        xy=(x, y),
                        xytext=(dx, dy),
                        textcoords="offset points",
                        fontsize=fontsize,
                        color=color,
                        zorder=zorder,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.88),
                        arrowprops=dict(arrowstyle="-", lw=0.8, color=color, alpha=0.55),
                    )
                    bb = ann.get_window_extent(renderer=renderer)
                    bb = _inflate_bbox(bb, px=2.0)
                    if ok_bbox(bb):
                        placed_bboxes.append(bb)
                        used_annotations.append(ann)
                        placed = True
                        break
                    ann.remove()

            if not placed:
                # Absolute last resort: still draw it (but at least boxed)
                ann = ax.annotate(
                    label,
                    xy=(x, y),
                    xytext=(offsets[-1][0], offsets[-1][1]),
                    textcoords="offset points",
                    fontsize=fontsize,
                    color=color,
                    zorder=zorder,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.88),
                    arrowprops=dict(arrowstyle="-", lw=0.8, color=color, alpha=0.55),
                )
                used_annotations.append(ann)

        return used_annotations

    # ------------------------------------------------------------
    # Background lane geometry (de-emphasized)
    # ------------------------------------------------------------
    all_pts = []
    for pts in seg_by_id.values():
        if pts is not None and len(pts):
            all_pts.append(np.asarray(pts, dtype=float))
    if all_pts:
        pts_concat = np.vstack(all_pts)
        ax.scatter(
            pts_concat[:, 0],
            pts_concat[:, 1],
            s=4,
            color="lightgray",
            alpha=0.25,
            zorder=0,
            label=None
        )

    # ------------------------------------------------------------
    # Ego paths (offset a bit to avoid perfect overlap)
    # ------------------------------------------------------------
    ego_line_artists = []
    marker_artists = []
    n_paths = max(1, len(picked))
    offset_step = 0.6  # meters
    center = (n_paths - 1) / 2.0

    for i, p in enumerate(picked):
        veh = p.get("vehicle", f"Vehicle {i+1}")
        sig = p.get("signature", {}) if isinstance(p.get("signature", {}), dict) else {}
        color = cmap(i % 10)
        offset_m = (i - center) * offset_step

        all_segment_pts = []
        segments_detailed = sig.get("segments_detailed", [])
        if isinstance(segments_detailed, list) and segments_detailed:
            for seg in segments_detailed:
                if not isinstance(seg, dict):
                    continue
                poly = seg.get("polyline_sample", [])
                if isinstance(poly, list) and poly:
                    for pt in poly:
                        if isinstance(pt, dict) and "x" in pt and "y" in pt:
                            all_segment_pts.append(np.array([float(pt["x"]), float(pt["y"])], dtype=float))

        if not all_segment_pts:
            seg_ids = sig.get("segment_ids", [])
            if isinstance(seg_ids, list):
                for sid in seg_ids:
                    try:
                        sid_i = int(sid)
                    except Exception:
                        continue
                    pts = seg_by_id.get(sid_i)
                    if pts is not None and len(pts) > 0:
                        pts = np.asarray(pts, dtype=float)
                        for pt in pts:
                            all_segment_pts.append(np.asarray(pt, dtype=float))

        if not all_segment_pts:
            continue

        pts = np.vstack(all_segment_pts)
        pts_off = _offset_polyline(pts, offset_m)

        ln, = ax.plot(
            pts_off[:, 0],
            pts_off[:, 1],
            linewidth=3.0,
            alpha=0.82,
            color=color,
            label=veh,
            zorder=2,
        )
        ego_line_artists.append(ln)

        # Direction arrow
        if len(pts_off) >= 2:
            arr = ax.annotate(
                "",
                xy=(pts_off[-1, 0], pts_off[-1, 1]),
                xytext=(pts_off[-2, 0], pts_off[-2, 1]),
                arrowprops=dict(arrowstyle="->", lw=3, color=color, alpha=0.95, mutation_scale=18),
                zorder=3,
            )

        # Start/end markers (first/last within crop if crop exists)
        if crop_region and all(k in crop_region for k in ("xmin", "xmax", "ymin", "ymax")):
            xmin, xmax = crop_region["xmin"], crop_region["xmax"]
            ymin, ymax = crop_region["ymin"], crop_region["ymax"]
            in_crop = [(idx, pt) for idx, pt in enumerate(pts) if xmin <= pt[0] <= xmax and ymin <= pt[1] <= ymax]
            if in_crop:
                first_idx = in_crop[0][0]
                last_idx = in_crop[-1][0]
                if 0 <= first_idx < len(pts_off) and 0 <= last_idx < len(pts_off):
                    first_pt = pts_off[first_idx]
                    last_pt = pts_off[last_idx]
                    sc = ax.scatter(
                        [first_pt[0], last_pt[0]],
                        [first_pt[1], last_pt[1]],
                        s=90,
                        facecolors=color,
                        edgecolors="white",
                        linewidths=1.5,
                        alpha=0.95,
                        zorder=6,
                    )
                    marker_artists.append(sc)

    legend = None
    if len(picked) > 0:
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], marker="o", linestyle="None", markersize=8, color="gray"))
        labels.append("Start/End")
        legend = ax.legend(handles=handles, labels=labels, loc="upper right", fontsize=9, framealpha=0.9)

    # ------------------------------------------------------------
    # Actors: draw motion first, then bboxes/markers, then labels last
    # ------------------------------------------------------------
    motion_line_artists = []
    bbox_patch_artists = []
    # marker_artists already contains start/end markers

    # Cluster labels by approximate position + asset_short to avoid stacked cone labels.
    BUCKET_M = 1.2
    clusters: Dict[Tuple[int, int, str, str], List[int]] = {}

    actor_pts = []
    for j, a in enumerate(actors_world):
        spawn = a.get("spawn", {})
        x, y = spawn.get("x"), spawn.get("y")
        if x is None or y is None:
            continue

        asset_short = _get_asset_short(str(a.get("asset_id", "")), str(a.get("category", "object")))
        bx = int(round(float(x) / BUCKET_M))
        by = int(round(float(y) / BUCKET_M))
        behavior_sig = _behavior_signature(a)
        key = (bx, by, asset_short, behavior_sig)
        clusters.setdefault(key, []).append(j)
        actor_pts.append((float(x), float(y), a))

    # 1) Draw motion polylines (so labels avoid them)
    for (x, y, a) in actor_pts:
        wps = a.get("world_waypoints", [])
        if isinstance(wps, list) and len(wps) >= 2:
            xs = [w["x"] for w in wps if "x" in w and "y" in w]
            ys = [w["y"] for w in wps if "x" in w and "y" in w]
            if len(xs) >= 2:
                is_erratic = (a.get("motion", {}) or {}).get("speed_profile") == "erratic"
                if is_erratic:
                    ln, = ax.plot(xs, ys, linestyle="-", linewidth=2.5, alpha=0.80, zorder=4)
                    motion_line_artists.append(ln)
                    sc = ax.scatter(xs, ys, s=18, marker="o", alpha=0.75, zorder=5)
                    marker_artists.append(sc)
                else:
                    ln, = ax.plot(xs, ys, linestyle=":", linewidth=1.6, alpha=0.75, zorder=4)
                    motion_line_artists.append(ln)

                arr = ax.annotate(
                    "",
                    xy=(xs[-1], ys[-1]),
                    xytext=(xs[-2], ys[-2]),
                    arrowprops=dict(arrowstyle="->", lw=1.4, alpha=0.8),
                    zorder=5,
                )

    # 2) Draw bboxes if available; otherwise draw a marker
    for (x, y, a) in actor_pts:
        spawn = a.get("spawn", {}) or {}
        yaw_deg = float(spawn.get("yaw_deg", 0.0))

        cat = str(a.get("category", "static")).lower()
        length, width = _actor_bbox_dims_from_actor_or_cache(a)

        # If bbox exists: draw ONLY bbox (filled), no "x"/"square" marker
        if length > 0.10 and width > 0.10:
            # Use category-based color, but keep it fairly subtle.
            if cat == "vehicle":
                face = "tab:blue"
                edge = "tab:blue"
            elif cat == "walker":
                face = "tab:green"
                edge = "tab:green"
            elif cat == "cyclist":
                face = "tab:orange"
                edge = "tab:orange"
            else:
                face = "tab:gray"
                edge = "tab:gray"

            poly = _draw_filled_oriented_bbox(
                x=x, y=y, yaw_deg=yaw_deg,
                length=length, width=width,
                facecolor=face, edgecolor=edge,
                alpha=0.25, zorder=6
            )
            bbox_patch_artists.append(poly)
        else:
            m = _actor_marker(cat)
            sc = ax.scatter([x], [y], s=55, marker=m, zorder=7)
            marker_artists.append(sc)

    # ------------------------------------------------------------
    # Labels: one label per cluster, repel away from paths/motion/bboxes
    # ------------------------------------------------------------
    # Compute forbidden regions for labels (screen-space bboxes)
    forbidden_artists = ego_line_artists + motion_line_artists + bbox_patch_artists + marker_artists
    if legend is not None:
        forbidden_artists.append(legend)
    forbidden = _collect_forbidden_bboxes(fig, ax, forbidden_artists, pad_px=5.0)

    label_items = []
    for (bx, by, asset_short, behavior_sig), idxs in clusters.items():
        # centroid anchor (better than "first actor")
        xs = []
        ys = []
        for j in idxs:
            sp = actors_world[j].get("spawn", {}) or {}
            if "x" in sp and "y" in sp:
                xs.append(float(sp["x"]))
                ys.append(float(sp["y"]))
        if not xs:
            continue
        x0 = float(sum(xs) / len(xs))
        y0 = float(sum(ys) / len(ys))

        count = len(idxs)
        a0 = actors_world[idxs[0]]
        if count > 1:
            label = f"{asset_short} ×{count}"
        else:
            actor_id = a0.get("id", "obj")
            label = f"{actor_id}: {asset_short}"

        behavior_label = _format_behavior_label(a0)
        if behavior_label:
            label = f"{label}\n{behavior_label}"
        label = _wrap_label(label, width=36)

        label_items.append({"x": x0, "y": y0, "label": label, "zorder": 10})

    _place_labels_repel(ax, fig, label_items, forbidden_bboxes=forbidden, fontsize=8, crop_region=crop_region)

    # ------------------------------------------------------------
    # Title
    # ------------------------------------------------------------
    lines = []
    scene_summary = _scene_summary_text(description)
    if scene_summary:
        desc_clean = " ".join(scene_summary.split())
        scene_text = textwrap.fill(desc_clean, width=90)
        lines.append(r"$\bf{Scene:}$ " + scene_text)
    lines.append(rf"$\bf{{Placed\ actors\ (n={len(actors_world)})}}$")
    ax.set_title("\n".join(lines), fontsize=12, loc="left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Visualization saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


__all__ = [
    "visualize",
    "plt",
]
