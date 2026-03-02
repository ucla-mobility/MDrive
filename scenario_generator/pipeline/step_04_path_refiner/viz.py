import math
from typing import Any, Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from .geometry import CropBox, _segments_to_polyline_with_map


def visualize_refinement(
    out_png: str,
    crop: CropBox,
    picked_entries: List[Dict[str, Any]],
    conflicts: List[Dict[str, Any]],
    lane_change_debug: Optional[List[Dict[str, Any]]] = None,
    conflict_times: Optional[List[Dict[str, Any]]] = None,
    seg_by_id: Optional[Dict[int, Any]] = None,
    show: bool = False,
):
    if plt is None:
        print("[WARNING] matplotlib not available; skipping viz.")
        return

    def _cluster_xy(conf_pts: List[Tuple[float, float, Any]], thresh_m: float = 2.5) -> List[Dict[str, Any]]:
        """
        Simple agglomerative clustering in world coords to merge near-duplicate conflict points.
        conf_pts: [(x,y,meta), ...]
        """
        clusters: List[Dict[str, Any]] = []
        for x, y, meta in conf_pts:
            placed = False
            for c in clusters:
                if math.hypot(x - c["x"], y - c["y"]) <= thresh_m:
                    c["members"].append(meta)
                    n = len(c["members"])
                    c["x"] = (c["x"] * (n - 1) + x) / n
                    c["y"] = (c["y"] * (n - 1) + y) / n
                    placed = True
                    break
            if not placed:
                clusters.append({"x": float(x), "y": float(y), "members": [meta]})
        return clusters

    def _place_label(
        ax,
        xy: Tuple[float, float],
        text: str,
        used_boxes: List[Tuple[float, float, float, float]],
        *,
        color: str = "black",
        fontsize: int = 8,
        weight: Optional[str] = None,
    ):
        """
        Greedy label placer in screen coords, but avoids overlaps using an approximate
        pixel-space bounding box for each label (much better than tracking only points).
        """
        offsets = (
            (20, 20), (20, -20), (-20, 20), (-20, -20),
            (30, 0), (-30, 0), (0, 30), (0, -30),
            (45, 15), (45, -15), (-45, 15), (-45, -15),
            (15, 45), (15, -45), (-15, 45), (-15, -45),
            (60, 0), (-60, 0), (0, 60), (0, -60),
        )
        x, y = xy
        x0, y0 = ax.transData.transform((x, y))

        # Cheap text box size estimate in pixels (good enough for decluttering)
        w = max(70.0, 7.0 * len(text) * (fontsize / 8.0))
        h = 18.0 * (fontsize / 8.0)

        def intersects(a, b) -> bool:
            return not (a[1] < b[0] or a[0] > b[1] or a[3] < b[2] or a[2] > b[3])

        best_dxdy = (20, 20)
        best_box = (x0 + 20, x0 + 20 + w, y0 + 20, y0 + 20 + h)
        best_score = -1e18

        for dx, dy in offsets:
            xp, yp = x0 + dx, y0 + dy
            box = (xp, xp + w, yp, yp + h)
            overlaps = sum(1 for ob in used_boxes if intersects(box, ob))

            # Prefer 0 overlaps; otherwise maximize distance to nearest label box center
            if overlaps == 0:
                score = 1e9
            else:
                cx, cy = xp + w / 2.0, yp + h / 2.0
                if used_boxes:
                    mind = min(
                        (cx - (ob[0] + ob[1]) / 2.0) ** 2 + (cy - (ob[2] + ob[3]) / 2.0) ** 2
                        for ob in used_boxes
                    )
                else:
                    mind = 0.0
                score = mind - 1e6 * overlaps

            if score > best_score:
                best_score = score
                best_dxdy = (dx, dy)
                best_box = box

            if overlaps == 0:
                break

        used_boxes.append(best_box)

        ax.annotate(
            text,
            xy=(x, y),
            xytext=best_dxdy,
            textcoords="offset points",
            fontsize=fontsize,
            color=color,
            fontweight=weight,
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
            arrowprops=dict(arrowstyle="-", lw=0.6, color="gray", alpha=0.6),
            zorder=10,
        )

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    used_boxes: List[Tuple[float, float, float, float]] = []

    # Set limits and invert X axis to match scene_objects viz
    ax.set_xlim(crop.xmin - 10, crop.xmax + 10)
    ax.set_ylim(crop.ymin - 10, crop.ymax + 10)
    ax.invert_xaxis()

    # Draw road network nodes in background (if seg_by_id provided)
    if seg_by_id:
        import numpy as _np
        all_pts = []
        for pts in seg_by_id.values():
            if pts is not None and len(pts):
                all_pts.append(pts)
        if all_pts:
            pts_concat = _np.vstack(all_pts)
            ax.scatter(pts_concat[:, 0], pts_concat[:, 1], s=6, color="lightgray", alpha=0.35, zorder=0)

    # Crop box (dashed rectangle)
    rect = plt.Rectangle(
        (crop.xmin, crop.ymin),
        crop.xmax - crop.xmin,
        crop.ymax - crop.ymin,
        fill=False,
        linestyle="--",
        linewidth=2,
        edgecolor="blue",
    )
    ax.add_patch(rect)

    cmap = plt.cm.get_cmap("tab10")

    # paths and markers
    for i, pe in enumerate(picked_entries):
        v = pe.get("vehicle", "?")
        sig = (pe.get("signature") or {}) if isinstance(pe.get("signature"), dict) else {}
        segs = sig.get("segments_detailed", []) if isinstance(sig.get("segments_detailed", []), list) else []
        pts, _ = _segments_to_polyline_with_map(segs)
        if len(pts) < 2:
            continue
        color = cmap(i % 10)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, linewidth=3.0, alpha=0.85, color=color, label=v)

        # start/end markers
        ax.scatter([pts[0][0]], [pts[0][1]], marker="o", s=90, facecolors=color, edgecolors="white", linewidths=1.5, zorder=6)
        ax.scatter([pts[-1][0]], [pts[-1][1]], marker="s", s=90, facecolors=color, edgecolors="white", linewidths=1.5, zorder=6)

        _place_label(ax, (pts[0][0], pts[0][1]), f"{v} start", used_boxes, fontsize=9)
        _place_label(ax, (pts[-1][0], pts[-1][1]), f"{v} end", used_boxes, fontsize=9)

    # conflict points (cluster near-duplicates)
    conf_pts: List[Tuple[float, float, Any]] = []
    for c in conflicts:
        p = c.get("point") or {}
        if "x" in p and "y" in p:
            conf_pts.append((float(p["x"]), float(p["y"]), c))

    for cl in _cluster_xy(conf_pts, thresh_m=2.5):
        x, y = float(cl["x"]), float(cl["y"])
        ax.scatter([x], [y], marker="x", s=90, color="red", zorder=8)
        label = "conflict" if len(cl["members"]) == 1 else f"conflict ×{len(cl['members'])}"
        _place_label(ax, (x, y), label, used_boxes, fontsize=9, color="red", weight="bold")

    # conflict timing (HUD box in axes coords, not stacked at intersection)
    if conflict_times:
        lines: List[str] = []
        for ct in conflict_times:
            tmap = ct.get("t") or {}
            keys = list(tmap.keys())
            if len(keys) == 2:
                a, b = keys[0], keys[1]
                ta, tb = float(tmap[a]), float(tmap[b])
                lines.append(f"{a} vs {b}: {ta:.1f}s / {tb:.1f}s (Δ={abs(ta - tb):.1f}s)")
            elif tmap:
                lines.append(", ".join(f"{k}:{float(v):.1f}s" for k, v in tmap.items()))
        if lines:
            ax.text(
                0.01, 0.99,
                "Conflict timing\n" + "\n".join(lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9),
                zorder=20,
            )

    # lane-change points (cut-start / merge)
    if lane_change_debug:
        for item in lane_change_debug:
            dbg = item.get("debug") or {}
            if isinstance(dbg, dict) and dbg.get("applied"):
                cs = dbg.get("cut_start_point") or {}
                mp = dbg.get("merge_point") or {}
                if "x" in cs and "y" in cs:
                    ax.scatter([cs["x"]], [cs["y"]], marker="^", s=70, color="purple", zorder=7)
                    _place_label(ax, (float(cs["x"]), float(cs["y"])), "cut_start", used_boxes, fontsize=8, color="purple")
                if "x" in mp and "y" in mp:
                    ax.scatter([mp["x"]], [mp["y"]], marker="v", s=70, color="purple", zorder=7)
                    _place_label(ax, (float(mp["x"]), float(mp["y"])), "merge", used_boxes, fontsize=8, color="purple")

    # Build legend with Start/End marker, place outside
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker="o", linestyle="None", markersize=8, color="gray"))
    labels.append("Start (○) / End (□)")

    ax.legend(
        handles=handles,
        labels=labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=9,
        framealpha=0.9,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(right=0.78)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    if show:
        plt.show()
    plt.close()


__all__ = [
    "visualize_refinement",
    "plt",
]
