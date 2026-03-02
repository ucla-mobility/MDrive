import math
import os
import re

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except Exception:
    plt = None
    mpatches = None

import generate_legal_paths as glp

from .features import _crop_contains_point
from .models import CropFeatures, CropKey


def _wrap_text(s: str, width: int = 120) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) <= width:
        return s
    import textwrap as _tw
    return "\n".join(_tw.wrap(s, width=width))


def _draw_segment_arrow(ax, pts: np.ndarray, color: str, alpha: float) -> None:
    if pts.shape[0] < 2:
        return
    mid = max(1, pts.shape[0] // 2)
    x0, y0 = float(pts[mid - 1, 0]), float(pts[mid - 1, 1])
    x1, y1 = float(pts[mid, 0]), float(pts[mid, 1])
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
        width=0.002,
        head_width=0.9,
        head_length=1.2,
        length_includes_head=True,
        color=color,
        alpha=alpha,
        linewidth=0,
        zorder=2,
    )


def save_viz(
    out_png: str,
    scenario_id: str,
    scenario_text: str,
    crop: CropKey,
    crop_feat: CropFeatures,
    invert_x: bool,
    dpi: int,
) -> None:
    if plt is None or mpatches is None:
        raise RuntimeError("matplotlib not available")

    seg_full = crop_feat._segments_full
    jcenters = crop_feat._junction_centers
    if seg_full is None or jcenters is None:
        raise RuntimeError("missing cached segments/junction centers")

    cb = glp.CropBox(crop.xmin, crop.xmax, crop.ymin, crop.ymax)
    segs_crop = glp.crop_segments(seg_full, cb)

    fig = plt.figure(figsize=(9.2, 7.6), dpi=dpi)
    ax = plt.gca()

    for seg in segs_crop:
        pts = np.asarray(seg.points, dtype=float)
        ax.plot(pts[:, 0], pts[:, 1], linewidth=1.0, alpha=0.55)
        ax.scatter([pts[0, 0], pts[-1, 0]], [pts[0, 1], pts[-1, 1]], s=8, alpha=0.75)
        _draw_segment_arrow(ax, pts, color="black", alpha=0.4)

    rect = mpatches.Rectangle((crop.xmin, crop.ymin), crop.xmax - crop.xmin, crop.ymax - crop.ymin,
                              fill=False, linewidth=2.0)
    ax.add_patch(rect)

    xs, ys = [], []
    for jc in jcenters:
        if _crop_contains_point(crop, jc):
            xs.append(float(jc[0]))
            ys.append(float(jc[1]))
    if xs:
        ax.scatter(xs, ys, s=38, marker="x")

    cx, cy = crop_feat.center_xy
    ax.scatter([cx], [cy], s=55, marker="o", alpha=0.8)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.15)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    title = f"{scenario_id} | {crop_feat.town} | {crop.to_str()}"
    ax.set_title(title, fontsize=10)
    fig.suptitle(_wrap_text(scenario_text, width=110), fontsize=9, y=0.98)

    pad = 6.0
    ax.set_xlim(crop.xmin - pad, crop.xmax + pad)
    ax.set_ylim(crop.ymin - pad, crop.ymax + pad)

    if invert_x:
        ax.invert_xaxis()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
