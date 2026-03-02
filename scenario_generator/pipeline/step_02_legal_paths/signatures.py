from typing import Any, Dict, List

import numpy as np

from .geometry import sanitize, wrap180
from .models import LegalPath


def classify_turn_world(in_heading: float, out_heading: float) -> str:
    """
    Classify turn in canonical world frame used here:
    - Heading computed from atan2(y, x)
    - X increases → West, X decreases → East (mirrored X)

    Mirroring X flips handedness; thus left/right are inverted compared to the
    standard math frame. We flip the sign test accordingly.
    """
    d = wrap180(out_heading - in_heading)
    ad = abs(d)
    if ad <= 35:
        return "straight"
    # Note the swapped semantics due to mirrored X axis
    if 35 < d < 145:
        return "right"
    if -145 < d < -35:
        return "left"
    return "uturn"


def heading_to_cardinal4_world(h: float) -> str:
    """
    4-way cardinal (E/N/W/S) in canonical world frame where:
    - X+ is West, X- is East. Therefore:
      - Heading ~ 0° (along +X) → "W"
      - Heading ~ ±180° (along -X) → "E"
      - Heading ~ 90° → "N"; ~ -90° → "S"
    """
    hd = wrap180(float(h))
    snapped = int(round(hd / 90.0)) * 90
    snapped = int(wrap180(snapped))
    if -45 <= snapped < 45:
        return "W"
    if 45 <= snapped < 135:
        return "N"
    if snapped >= 135 or snapped < -135:
        return "E"
    return "S"


def bound_word(card4: str) -> str:
    return {"E": "eastbound", "W": "westbound", "N": "northbound", "S": "southbound"}.get(card4, "forward")


def _pt2(p: np.ndarray) -> Dict[str, float]:
    return {"x": float(p[0]), "y": float(p[1])}


def _sample_polyline(points_xy: np.ndarray, max_points: int = 10) -> List[Dict[str, float]]:
    """
    Small, deterministic polyline sample for logging/debug.
    This is NOT used for any connectivity/path logic.
    """
    pts = np.asarray(points_xy, dtype=float)
    n = len(pts)
    if n == 0:
        return []
    if n <= max_points:
        return [{"x": float(p[0]), "y": float(p[1])} for p in pts]
    idxs = np.linspace(0, n - 1, num=max_points, dtype=int)
    return [{"x": float(pts[i, 0]), "y": float(pts[i, 1])} for i in idxs]


def build_segments_detailed_for_path(path: LegalPath, polyline_sample_n: int = 10) -> List[Dict[str, Any]]:
    """
    Extra per-segment logging payload for --out-json-detailed ONLY.
    No effect on any functional behavior; purely additional metadata export.
    """
    out: List[Dict[str, Any]] = []
    for seg in path.segments:
        start_pt = seg.points[0]
        end_pt = seg.points[-1]

        hs = float(wrap180(seg.heading_at_start()))
        he = float(wrap180(seg.heading_at_end()))
        cs = heading_to_cardinal4_world(hs)
        ce = heading_to_cardinal4_world(he)

        bb = seg.bbox()
        out.append({
            "seg_id": int(seg.seg_id),
            "road_id": int(seg.road_id),
            "section_id": int(seg.section_id),
            "lane_id": int(seg.lane_id),
            "length_m": float(seg.length()),
            "bbox": {
                "xmin": float(bb[0]),
                "xmax": float(bb[1]),
                "ymin": float(bb[2]),
                "ymax": float(bb[3]),
            },
            "start": {
                "point": _pt2(start_pt),
                "heading_deg": hs,
                "cardinal4": cs,
                "bound": bound_word(cs),
                "orig_idx": int(seg.orig_idx[0]) if len(seg.orig_idx) > 0 else None,
            },
            "end": {
                "point": _pt2(end_pt),
                "heading_deg": he,
                "cardinal4": ce,
                "bound": bound_word(ce),
                "orig_idx": int(seg.orig_idx[-1]) if len(seg.orig_idx) > 0 else None,
            },
            "polyline_sample": _sample_polyline(seg.points, max_points=polyline_sample_n),
        })
    return out


def build_path_signature(path: LegalPath) -> Dict[str, Any]:
    """
    Compact, order-explicit representation that an LLM can match reliably.
    Uses WORLD-FRAME turn classification (recommended).
    """
    segs = path.segments
    assert len(segs) > 0

    entry_h = float(wrap180(segs[0].heading_at_start()))
    exit_h = float(wrap180(segs[-1].heading_at_end()))
    entry_card4 = heading_to_cardinal4_world(entry_h)
    exit_card4 = heading_to_cardinal4_world(exit_h)

    # Ordered maneuvers between segments (length = nseg-1)
    maneuvers_between = []
    for i in range(len(segs) - 1):
        in_h = segs[i].heading_at_end()
        out_h = segs[i + 1].heading_at_start()
        maneuvers_between.append(classify_turn_world(in_h, out_h))

    # Roads / sections / lanes sequences
    roads = [int(s.road_id) for s in segs]
    sections = [int(s.section_id) for s in segs]
    lanes = [int(s.lane_id) for s in segs]
    seg_ids = [int(s.seg_id) for s in segs]

    # Start/end points
    start_pt = segs[0].points[0]
    end_pt = segs[-1].points[-1]

    return {
        "entry": {
            "point": _pt2(start_pt),
            "heading_deg": entry_h,
            "cardinal4": entry_card4,
            "bound": bound_word(entry_card4),
            "road_id": int(segs[0].road_id),
            "section_id": int(segs[0].section_id),
            "lane_id": int(segs[0].lane_id),
        },
        "exit": {
            "point": _pt2(end_pt),
            "heading_deg": exit_h,
            "cardinal4": exit_card4,
            "bound": bound_word(exit_card4),
            "road_id": int(segs[-1].road_id),
            "section_id": int(segs[-1].section_id),
            "lane_id": int(segs[-1].lane_id),
        },
        "length_m": float(path.total_length),
        "num_segments": int(len(segs)),
        "segment_ids": seg_ids,
        "roads": roads,
        "sections": sections,
        "lanes": lanes,
        "maneuvers_between": maneuvers_between,  # order-explicit
        # Helpful redundancy: coarse path “turn” from entry->exit (not perfect, but quick filter)
        "entry_to_exit_turn": classify_turn_world(entry_h, exit_h),
    }


def make_path_name(path_idx: int, sig: Dict[str, Any]) -> str:
    """Stable-ish name for humans + logs."""
    ent = sig["entry"]
    ex = sig["exit"]
    man = sig["entry_to_exit_turn"]
    nseg = sig["num_segments"]
    length = sig["length_m"]
    return (
        f"path_{path_idx + 1:03d}"
        f"__man={sanitize(man)}"
        f"__entry={ent['cardinal4']}({int(round(ent['heading_deg']))}deg)"
        f"__exit={ex['cardinal4']}({int(round(ex['heading_deg']))}deg)"
        f"__len={length:.1f}m__n={nseg}segs"
    )
