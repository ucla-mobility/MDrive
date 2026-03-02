import json
import math
from typing import Any, Dict, List, Optional

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _load_nodes(nodes_path: str) -> Dict[str, Any]:
    with open(nodes_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "payload" not in data:
        raise ValueError(f"{nodes_path} missing top-level 'payload'")
    return data


def _wrap180(deg: float) -> float:
    return ((deg + 180.0) % 360.0) - 180.0


def _build_segments_minimal(nodes: Dict[str, Any], min_points: int = 6) -> List[Dict[str, Any]]:
    """
    Minimal reimplementation of build_segments() just for visualization.
    Returns a list of dict segments:
      {seg_id, road_id, lane_id, section_id, points: [(x,y),...]}
    """
    import numpy as np
    from collections import defaultdict

    payload = nodes["payload"]
    x = np.asarray(payload["x"], dtype=float)
    y = np.asarray(payload["y"], dtype=float)
    yaw = np.asarray(payload["yaw"], dtype=float)
    road_id = np.asarray(payload["road_id"], dtype=int)
    lane_id = np.asarray(payload["lane_id"], dtype=int)
    section_id = np.asarray(payload["section_id"], dtype=int)

    grouped: Dict[tuple, List[int]] = defaultdict(list)
    for i in range(len(x)):
        grouped[(int(road_id[i]), int(lane_id[i]), int(section_id[i]))].append(i)

    def unit_from_yaw(yaw_deg: float) -> np.ndarray:
        r = np.radians(_wrap180(float(yaw_deg)))
        return np.array([np.cos(r), np.sin(r)], dtype=float)

    def orient_polyline(pts: np.ndarray, yaws_deg: np.ndarray, idxs: np.ndarray) -> tuple:
        if len(pts) < 2:
            return pts, yaws_deg, idxs
        vecs = pts[1:] - pts[:-1]
        norms = (np.linalg.norm(vecs, axis=1) + 1e-9)
        dir_vecs = vecs / norms[:, None]
        yaw_vecs = np.vstack([unit_from_yaw(y) for y in yaws_deg[:-1]])
        dots = np.sum(dir_vecs * yaw_vecs, axis=1)
        if float(np.nanmean(dots)) < 0.0:
            return pts[::-1].copy(), yaws_deg[::-1].copy(), idxs[::-1].copy()
        return pts, yaws_deg, idxs

    def split_by_gaps(idxs_sorted: np.ndarray, pts: np.ndarray, yaws_deg: np.ndarray, gap_m: float = 6.0):
        if len(pts) < 2:
            return [(idxs_sorted, pts, yaws_deg)] if len(pts) > 0 else []
        jumps = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        cuts = [0]
        for i, d in enumerate(jumps):
            if float(d) > gap_m:
                cuts.append(i + 1)
        cuts.append(len(pts))
        out = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a >= 2:
                out.append((idxs_sorted[a:b], pts[a:b], yaws_deg[a:b]))
        return out

    segments = []
    seg_id_counter = 0

    for (rid, lid, sid), idxs in grouped.items():
        idxs_sorted = np.asarray(sorted(idxs), dtype=int)
        pts = np.vstack([x[idxs_sorted], y[idxs_sorted]]).T
        yaws_data = yaw[idxs_sorted]
        for idxs_chunk, pts_chunk, yaws_chunk in split_by_gaps(idxs_sorted, pts, yaws_data):
            pts_o, yaws_o, idxs_o = orient_polyline(pts_chunk, yaws_chunk, idxs_chunk)
            if len(pts_o) < min_points:
                continue
            segments.append({
                "seg_id": int(seg_id_counter),
                "road_id": int(rid),
                "lane_id": int(lid),
                "section_id": int(sid),
                "points": [(float(p[0]), float(p[1])) for p in pts_o],
            })
            seg_id_counter += 1

    return segments


def _plot_paths_together(
    all_segments: List[Dict[str, Any]],
    picked: List[Dict[str, Any]],
    crop: Optional[Dict[str, Any]],
    out_path: Optional[str] = None,
    show: bool = False,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib not available; install it or disable --viz")

    seg_by_id = {s["seg_id"]: s for s in all_segments}

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect("equal", adjustable="box")

    if crop and all(k in crop for k in ("xmin", "xmax", "ymin", "ymax")):
        xmin, xmax, ymin, ymax = crop["xmin"], crop["xmax"], crop["ymin"], crop["ymax"]
        ax.set_xlim(xmin - 5, xmax + 5)
        ax.set_ylim(ymin - 5, ymax + 5)
        ax.invert_xaxis()
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, linestyle="--", linewidth=2)
        ax.add_patch(rect)

    ax.grid(True, alpha=0.3)
    ax.set_title(f"Picked Paths (n={len(picked)})")

    cmap = plt.cm.get_cmap("tab20")

    def _draw_segment_arrow(pts: List[tuple], color: str, alpha: float) -> None:
        if len(pts) < 2:
            return
        mid = max(1, len(pts) // 2)
        x0, y0 = pts[mid - 1]
        x1, y1 = pts[mid]
        dx = x1 - x0
        dy = y1 - y0
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-3:
            return
        arrow_len = min(4.0, max(1.6, seg_len * 0.35))
        scale = arrow_len / seg_len
        ax.arrow(
            x0,
            y0,
            dx * scale,
            dy * scale,
            width=0.003,
            head_width=1.2,
            head_length=1.6,
            length_includes_head=True,
            color=color,
            alpha=alpha,
            linewidth=0,
            zorder=3,
        )

    for i, entry in enumerate(picked):
        sig = entry.get("signature", {})
        seg_ids = sig.get("segment_ids", [])
        color = cmap(i % 20)

        for sid in seg_ids:
            sid = int(sid)
            seg = seg_by_id.get(sid)
            if not seg:
                continue
            pts = seg["points"]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, linewidth=2.5, alpha=0.85, color=color)
            _draw_segment_arrow(pts, color=color, alpha=0.85)

        ent = sig.get("entry", {}).get("point", None)
        ex = sig.get("exit", {}).get("point", None)
        if ent:
            ax.plot(ent["x"], ent["y"], marker="o", markersize=8, color=color)
        if ex:
            ax.plot(ex["x"], ex["y"], marker="s", markersize=8, color=color)

    if out_path:
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Visualization saved: {out_path}")

    if show:
        plt.show()

    plt.close(fig)


__all__ = [
    "_build_segments_minimal",
    "_load_nodes",
    "_plot_paths_together",
    "_wrap180",
    "plt",
]
