import random
import hashlib
from typing import Any, Dict, List, Tuple

from .geometry import (
    CropBox,
    _build_segment_payload_from_polyline,
    _dist,
    _find_closest_idx,
    _polyline_cumdist,
    _resample_polyline,
    _segments_to_polyline_with_map,
    _unit,
    find_conflict_between_polylines,
)


def apply_merge_into_lane_of(
    mover: Dict[str, Any],
    target: Dict[str, Any],
    crop: CropBox,
    style: str = "polite",
    timing: str = "near_conflict",
    synthetic_seg_id_base: int = 900000,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns (updated_mover_entry, debug_info).

    Best-effort macro:
      mover follows its picked path until a cut-start point, then transitions (synthetic segment)
      to a merge point on target's refined path, then follows target's path onwards.

    We choose merge point as:
      - conflict between mover/target if exists (preferred)
      - else nearest approach within crop
    """
    m_sig = (mover.get("signature") or {}) if isinstance(mover.get("signature"), dict) else {}
    t_sig = (target.get("signature") or {}) if isinstance(target.get("signature"), dict) else {}
    m_segs = m_sig.get("segments_detailed", []) if isinstance(m_sig.get("segments_detailed", []), list) else []
    t_segs = t_sig.get("segments_detailed", []) if isinstance(t_sig.get("segments_detailed", []), list) else []

    # Stable deterministic jitter so each pair has distinct merge geometry across processes.
    seed_key = f"{mover.get('vehicle', '')}::{target.get('vehicle', '')}".encode("utf-8")
    seed = int.from_bytes(hashlib.blake2b(seed_key, digest_size=8).digest(), "big")
    rng = random.Random(seed)

    m_pts, _ = _segments_to_polyline_with_map(m_segs)
    t_pts, _ = _segments_to_polyline_with_map(t_segs)
    if len(m_pts) < 4 or len(t_pts) < 4:
        return mover, {"applied": False, "reason": "insufficient polyline points"}

    # pick merge candidate
    conflict = find_conflict_between_polylines(m_pts, t_pts, dist_thresh_m=4.0)
    if conflict is not None:
        merge_p = (float(conflict["point"]["x"]), float(conflict["point"]["y"]))
    else:
        # brute nearest pair of vertices, preferring inside crop
        best = None
        for i, p in enumerate(m_pts):
            for j, q in enumerate(t_pts):
                if not crop.contains(q[0], q[1]):
                    continue
                d = _dist(p, q)
                if best is None or d < best[0]:
                    best = (d, i, j)
        if best is None:
            return mover, {"applied": False, "reason": "no merge candidate in crop"}
        _d, _i, j = best
        merge_p = t_pts[j]

    merge_idx_t = _find_closest_idx(t_pts, merge_p)

    # gap (cut-off means tighter)
    gap_m = 5.0 if style == "cut_off" else 10.0
    # Larger jitter so merges don't collapse to the same polyline sample
    gap_m = max(2.0, gap_m + rng.uniform(-5.0, 5.0))
    # move forward along target by gap
    t_cum = _polyline_cumdist(t_pts)
    desired_s = min(t_cum[-1], t_cum[merge_idx_t] + gap_m + rng.uniform(-5.0, 5.0))
    # find idx at desired_s
    j = merge_idx_t
    while j + 1 < len(t_cum) and t_cum[j] < desired_s:
        j += 1
    merge_idx_t2 = j
    # Small index jitter to avoid all merges snapping to the same vertex
    merge_idx_t2 = max(0, min(len(t_pts) - 1, merge_idx_t2 + int(rng.choice([-2, -1, 0, 1, 2]))))
    merge_p2 = t_pts[merge_idx_t2]

    # choose cut start on mover some distance before reaching a point closest to merge
    m_idx_near = _find_closest_idx(m_pts, merge_p2)
    m_cum = _polyline_cumdist(m_pts)
    back_m = max(6.0, 12.0 + rng.uniform(-8.0, 8.0))  # start lane change with stronger jitter
    desired_m_s = max(0.0, m_cum[m_idx_near] - back_m)
    i = m_idx_near
    while i - 1 >= 0 and m_cum[i] > desired_m_s:
        i -= 1
    cut_idx_m = i
    cut_idx_m = max(0, min(len(m_pts) - 1, cut_idx_m + int(rng.choice([-2, -1, 0, 1, 2]))))
    cut_p = m_pts[cut_idx_m]

    # build synthetic transition polyline (simple cubic-like control)
    fwd_m = _unit((m_pts[min(cut_idx_m + 1, len(m_pts) - 1)][0] - cut_p[0], m_pts[min(cut_idx_m + 1, len(m_pts) - 1)][1] - cut_p[1]))
    fwd_t = _unit((merge_p2[0] - t_pts[max(0, merge_idx_t2 - 1)][0], merge_p2[1] - t_pts[max(0, merge_idx_t2 - 1)][1]))
    d = _dist(cut_p, merge_p2)
    ctrl1 = (cut_p[0] + fwd_m[0] * d * 0.33, cut_p[1] + fwd_m[1] * d * 0.33)
    ctrl2 = (merge_p2[0] - fwd_t[0] * d * 0.33, merge_p2[1] - fwd_t[1] * d * 0.33)

    def bezier(t: float) -> Tuple[float, float]:
        # cubic Bezier: P0=cut, P1=ctrl1, P2=ctrl2, P3=merge
        x = (1 - t) ** 3 * cut_p[0] + 3 * (1 - t) ** 2 * t * ctrl1[0] + 3 * (1 - t) * t ** 2 * ctrl2[0] + t ** 3 * merge_p2[0]
        y = (1 - t) ** 3 * cut_p[1] + 3 * (1 - t) ** 2 * t * ctrl1[1] + 3 * (1 - t) * t ** 2 * ctrl2[1] + t ** 3 * merge_p2[1]
        return (x, y)

    trans = [bezier(k / 19.0) for k in range(20)]
    trans = _resample_polyline(trans, 12)

    # Build new per-segment structure: prefix from mover, synthetic, suffix from target
    prefix_pts = m_pts[: cut_idx_m + 1]
    suffix_pts = t_pts[merge_idx_t2:]


    # Build segments_detailed for new path: keep mover segments count low -> single segment each section
    new_segs: List[Dict[str, Any]] = []
    # prefix as one segment
    if len(prefix_pts) >= 2:
        new_segs.append(_build_segment_payload_from_polyline(_resample_polyline(prefix_pts, 10), seg_template=(m_segs[0] if m_segs else None), seg_id=synthetic_seg_id_base + 1))
    # transition segment
    new_segs.append(_build_segment_payload_from_polyline(trans, seg_template=(t_segs[0] if t_segs else None), seg_id=synthetic_seg_id_base + 2))
    # suffix as one segment
    if len(suffix_pts) >= 2:
        new_segs.append(_build_segment_payload_from_polyline(_resample_polyline(suffix_pts, 10), seg_template=(t_segs[-1] if t_segs else None), seg_id=synthetic_seg_id_base + 3))

    # Build updated signature
    new_sig = dict(m_sig)
    new_sig["segments_detailed"] = new_segs
    new_sig["segment_ids"] = [int(s.get("seg_id", 0)) for s in new_segs]
    new_sig["num_segments"] = int(len(new_segs))
    new_sig["length_m"] = float(sum(float(s.get("length_m", 0.0)) for s in new_segs))

    # entry/exit points
    if new_segs:
        new_sig["entry"] = dict(new_sig.get("entry", {}))
        new_sig["entry"]["point"] = {"x": float(new_segs[0]["start"]["point"]["x"]), "y": float(new_segs[0]["start"]["point"]["y"])}
        new_sig["exit"] = dict(new_sig.get("exit", {}))
        new_sig["exit"]["point"] = {"x": float(new_segs[-1]["end"]["point"]["x"]), "y": float(new_segs[-1]["end"]["point"]["y"])}

    mover2 = dict(mover)
    mover2["signature_original"] = mover.get("signature")
    mover2["signature"] = new_sig
    mover2["name_refined"] = f"{mover.get('name','')}__merge_into_{target.get('vehicle','')}"
    return mover2, {
        "applied": True,
        "style": style,
        "timing": timing,
        "cut_start_point": {"x": float(cut_p[0]), "y": float(cut_p[1])},
        "merge_point": {"x": float(merge_p2[0]), "y": float(merge_p2[1])},
    }


__all__ = [
    "apply_merge_into_lane_of",
]
