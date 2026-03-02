#!/usr/bin/env python3
"""
Build robust V2XPNP<->CARLA lane correspondence diagnostics.

This tool:
1) Loads a V2XPNP vector map and CARLA map-cache polylines.
2) Applies the configured CARLA->V2 alignment transform.
3) Optionally refines that alignment with robust SE(2) ICP.
4) Runs the global correspondence matcher from route_export.
5) Exports an interactive HTML report for visual QA.

Example:
  python3 -m v2xpnp.pipeline.correspondence_diagnostics \
    --v2-map v2xpnp/map/v2x_intersection_vector_map.pkl \
    --carla-map-cache v2xpnp/map/carla_map_cache.pkl \
    --carla-align v2xpnp/map/ucla_map_offset_carla.json \
    --out v2xpnp/map/lane_correspondence_diagnostics.html
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v2xpnp.pipeline.pipeline_runtime import VectorMapData, load_vector_map  # noqa: E402
from v2xpnp.pipeline import route_export as ytm  # noqa: E402


@dataclass
class ICPStats:
    applied: bool
    iterations: int
    inliers: int
    before_median: float
    before_mean: float
    before_p90: float
    after_median: float
    after_mean: float
    after_p90: float
    delta_theta_deg: float
    delta_tx: float
    delta_ty: float
    history: List[Dict[str, float]]


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _polyline_points_xy(poly: np.ndarray) -> List[List[float]]:
    if not isinstance(poly, np.ndarray) or poly.size == 0:
        return []
    arr = poly[:, :2] if poly.ndim == 2 else np.zeros((0, 2), dtype=np.float64)
    return [[float(p[0]), float(p[1])] for p in arr]


def _flatten_lines(lines: Sequence[Sequence[Tuple[float, float]]]) -> np.ndarray:
    out: List[Tuple[float, float]] = []
    for ln in lines:
        for p in ln:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                out.append((float(p[0]), float(p[1])))
    if not out:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(out, dtype=np.float64)


def _compute_nn_stats(src_xy: np.ndarray, dst_tree: cKDTree) -> Tuple[float, float, float]:
    if src_xy.shape[0] <= 0:
        return float("inf"), float("inf"), float("inf")
    d, _ = dst_tree.query(src_xy, k=1)
    d_arr = np.asarray(d, dtype=np.float64)
    return float(np.median(d_arr)), float(np.mean(d_arr)), float(np.quantile(d_arr, 0.9))


def _kabsch_2d(src_xy: np.ndarray, dst_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return rigid transform (R, t) s.t. dst ~= R*src + t."""
    src_cent = np.mean(src_xy, axis=0)
    dst_cent = np.mean(dst_xy, axis=0)
    src0 = src_xy - src_cent
    dst0 = dst_xy - dst_cent
    h = src0.T @ dst0
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0.0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    t = dst_cent - (r @ src_cent)
    return r, t


def _apply_rigid_to_points(points_xy: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    if points_xy.shape[0] <= 0:
        return points_xy
    return (r @ points_xy.T).T + t[None, :]


def _apply_rigid_to_lines(
    lines: Sequence[Sequence[Tuple[float, float]]],
    r: np.ndarray,
    t: np.ndarray,
) -> List[List[Tuple[float, float]]]:
    out: List[List[Tuple[float, float]]] = []
    for ln in lines:
        if len(ln) < 2:
            continue
        arr = np.asarray([[float(p[0]), float(p[1])] for p in ln], dtype=np.float64)
        arr2 = _apply_rigid_to_points(arr, r, t)
        out.append([(float(p[0]), float(p[1])) for p in arr2])
    return out


def _rotation_matrix(theta_deg: float) -> np.ndarray:
    th = math.radians(float(theta_deg))
    c = math.cos(th)
    s = math.sin(th)
    return np.asarray([[c, -s], [s, c]], dtype=np.float64)


def _rotation_deg_from_matrix(r: np.ndarray) -> float:
    return math.degrees(math.atan2(float(r[1, 0]), float(r[0, 0])))


def _refine_alignment_icp(
    transformed_lines: Sequence[Sequence[Tuple[float, float]]],
    v2_vertices_xy: np.ndarray,
    enabled: bool,
    max_iters: int = 10,
    inlier_quantile: float = 0.35,
    inlier_max_m: float = 15.0,
    max_step_deg: float = 1.25,
    max_step_trans_m: float = 2.0,
    stop_theta_deg: float = 0.02,
    stop_trans_m: float = 0.03,
) -> Tuple[List[List[Tuple[float, float]]], ICPStats]:
    if not enabled:
        stats = ICPStats(
            applied=False,
            iterations=0,
            inliers=0,
            before_median=float("nan"),
            before_mean=float("nan"),
            before_p90=float("nan"),
            after_median=float("nan"),
            after_mean=float("nan"),
            after_p90=float("nan"),
            delta_theta_deg=0.0,
            delta_tx=0.0,
            delta_ty=0.0,
            history=[],
        )
        return [list(ln) for ln in transformed_lines], stats

    src0 = _flatten_lines(transformed_lines)
    if src0.shape[0] <= 0 or v2_vertices_xy.shape[0] <= 0:
        stats = ICPStats(
            applied=False,
            iterations=0,
            inliers=0,
            before_median=float("nan"),
            before_mean=float("nan"),
            before_p90=float("nan"),
            after_median=float("nan"),
            after_mean=float("nan"),
            after_p90=float("nan"),
            delta_theta_deg=0.0,
            delta_tx=0.0,
            delta_ty=0.0,
            history=[],
        )
        return [list(ln) for ln in transformed_lines], stats

    # Subsample source points for speed/stability.
    if src0.shape[0] > 30000:
        idx = np.linspace(0, src0.shape[0] - 1, 30000, dtype=np.int32)
        src = src0[idx]
    else:
        src = src0.copy()

    dst_tree = cKDTree(v2_vertices_xy)
    before_med, before_mean, before_p90 = _compute_nn_stats(src, dst_tree)

    r_total = np.eye(2, dtype=np.float64)
    t_total = np.zeros((2,), dtype=np.float64)
    current = src.copy()
    history: List[Dict[str, float]] = []
    final_inliers = 0

    q = min(0.95, max(0.05, float(inlier_quantile)))
    for it in range(max(1, int(max_iters))):
        d, idx = dst_tree.query(current, k=1)
        d_arr = np.asarray(d, dtype=np.float64)
        dst_match = v2_vertices_xy[np.asarray(idx, dtype=np.int32)]

        thresh = min(float(inlier_max_m), float(np.quantile(d_arr, q)))
        inlier_mask = d_arr <= thresh
        inlier_count = int(np.sum(inlier_mask))
        if inlier_count < 20:
            break

        src_in = current[inlier_mask]
        dst_in = dst_match[inlier_mask]
        r_step, t_step = _kabsch_2d(src_in, dst_in)

        step_theta = _rotation_deg_from_matrix(r_step)
        step_trans = float(np.linalg.norm(t_step))

        # Clamp large updates for stability.
        if abs(step_theta) > float(max_step_deg):
            clipped_theta = float(max_step_deg) * (1.0 if step_theta >= 0.0 else -1.0)
            r_step = _rotation_matrix(clipped_theta)
            step_theta = clipped_theta
        if step_trans > float(max_step_trans_m):
            scale = float(max_step_trans_m) / max(step_trans, 1e-9)
            t_step = t_step * scale
            step_trans = float(np.linalg.norm(t_step))

        current = _apply_rigid_to_points(current, r_step, t_step)
        r_total = r_step @ r_total
        t_total = (r_step @ t_total) + t_step
        final_inliers = inlier_count

        med_now, mean_now, p90_now = _compute_nn_stats(current, dst_tree)
        history.append(
            {
                "iter": float(it + 1),
                "inliers": float(inlier_count),
                "inlier_thresh_m": float(thresh),
                "step_theta_deg": float(step_theta),
                "step_trans_m": float(step_trans),
                "median_m": float(med_now),
                "mean_m": float(mean_now),
                "p90_m": float(p90_now),
            }
        )
        if abs(step_theta) <= float(stop_theta_deg) and step_trans <= float(stop_trans_m):
            break

    corrected_lines = _apply_rigid_to_lines(transformed_lines, r_total, t_total)
    corrected_pts = _flatten_lines(corrected_lines)
    if corrected_pts.shape[0] > 30000:
        idx = np.linspace(0, corrected_pts.shape[0] - 1, 30000, dtype=np.int32)
        corrected_pts = corrected_pts[idx]
    after_med, after_mean, after_p90 = _compute_nn_stats(corrected_pts, dst_tree)

    stats = ICPStats(
        applied=True,
        iterations=len(history),
        inliers=final_inliers,
        before_median=float(before_med),
        before_mean=float(before_mean),
        before_p90=float(before_p90),
        after_median=float(after_med),
        after_mean=float(after_mean),
        after_p90=float(after_p90),
        delta_theta_deg=float(_rotation_deg_from_matrix(r_total)),
        delta_tx=float(t_total[0]),
        delta_ty=float(t_total[1]),
        history=history,
    )
    return corrected_lines, stats


def _build_payload_from_maps(
    v2_map: VectorMapData,
    carla_lines_xy: Sequence[Sequence[Tuple[float, float]]],
    carla_source: str,
    carla_name: str,
) -> Dict[str, object]:
    lanes_payload: List[Dict[str, object]] = []
    for lane in v2_map.lanes:
        poly = lane.polyline[:, :2]
        if poly.shape[0] < 2:
            continue
        mid = poly[poly.shape[0] // 2]
        lanes_payload.append(
            {
                "index": int(lane.index),
                "road_id": int(lane.road_id),
                "lane_id": int(lane.lane_id),
                "lane_type": str(lane.lane_type),
                "polyline": _polyline_points_xy(lane.polyline),
                "label_x": float(mid[0]),
                "label_y": float(mid[1]),
                "entry_lanes": list(lane.entry_lanes or []),
                "exit_lanes": list(lane.exit_lanes or []),
            }
        )
    carla_payload_lines: List[Dict[str, object]] = []
    for ln in carla_lines_xy:
        if len(ln) < 2:
            continue
        carla_payload_lines.append(
            {"polyline": [[float(p[0]), float(p[1])] for p in ln]}
        )

    return {
        "map": {
            "name": str(v2_map.name),
            "source_path": str(v2_map.source_path),
            "lanes": lanes_payload,
        },
        "carla_map": {
            "name": str(carla_name or "carla_map_cache"),
            "source_path": str(carla_source),
            "lines": carla_payload_lines,
        },
    }


def _lane_label(lane: Dict[str, object]) -> str:
    return (
        f"idx={int(lane.get('index', -1))} "
        f"r{int(lane.get('road_id', 0))} "
        f"l{int(lane.get('lane_id', 0))} "
        f"t{str(lane.get('lane_type', ''))}"
    )


def _quality_rank(q: str) -> int:
    qv = str(q or "").lower()
    if qv == "high":
        return 4
    if qv == "medium":
        return 3
    if qv == "low":
        return 2
    if qv == "poor":
        return 1
    return 0


def _quality_color(q: str) -> str:
    qv = str(q or "").lower()
    if qv == "high":
        return "#2ecc71"
    if qv == "medium":
        return "#f1c40f"
    if qv == "low":
        return "#e67e22"
    if qv == "poor":
        return "#e74c3c"
    return "#95a5a6"


def _normalize_int_set_map(raw_map: object) -> Dict[int, set]:
    out: Dict[int, set] = {}
    if not isinstance(raw_map, dict):
        return out
    for k, vals in raw_map.items():
        ki = int(_safe_float(k, -1))
        if ki < 0:
            continue
        bucket: set = set()
        if isinstance(vals, (list, tuple, set)):
            for v in vals:
                vi = int(_safe_float(v, -1))
                if vi >= 0:
                    bucket.add(vi)
        out[ki] = bucket
    return out


def _candidate_quality_penalty(q: str) -> float:
    qv = str(q or "").lower()
    if qv == "high":
        return 0.0
    if qv == "medium":
        return 0.6
    if qv == "low":
        return 1.8
    if qv == "poor":
        return 5.5
    return 7.0


def _is_driving_lane(
    lane_index: int,
    v2_feats: Dict[int, Dict[str, object]],
    driving_types: set,
) -> bool:
    lf = v2_feats.get(int(lane_index))
    if not isinstance(lf, dict):
        return False
    return str(lf.get("lane_type", "")) in driving_types


def _legal_route_stats(
    lane_to_carla: Dict[int, Dict[str, object]],
    v2_graph: Dict[str, object],
    carla_feats: Dict[int, Dict[str, object]],
    carla_successors: Dict[int, set],
    driving_lanes: set,
) -> Dict[str, float]:
    succs_raw = v2_graph.get("successors", {})
    succs = _normalize_int_set_map(succs_raw)
    total = 0
    ok = 0
    bad = 0
    for li_a in sorted(driving_lanes):
        info_a = lane_to_carla.get(int(li_a))
        if not isinstance(info_a, dict):
            continue
        ci_a = int(_safe_float(info_a.get("carla_line_index"), -1))
        if ci_a < 0:
            continue
        for li_b in sorted(succs.get(int(li_a), set())):
            if li_b not in driving_lanes:
                continue
            info_b = lane_to_carla.get(int(li_b))
            if not isinstance(info_b, dict):
                continue
            ci_b = int(_safe_float(info_b.get("carla_line_index"), -1))
            if ci_b < 0:
                continue
            total += 1
            if ci_a == ci_b or ytm._carla_lines_connected(
                ci_a,
                ci_b,
                carla_feats,
                carla_successors,
                threshold_m=6.0,
            ):
                ok += 1
            else:
                bad += 1
    ratio = float(ok / total) if total > 0 else float("nan")
    return {
        "total": float(total),
        "ok": float(ok),
        "bad": float(bad),
        "ratio": float(ratio),
    }


def _postprocess_correspondence(
    corr: Dict[str, object],
    driving_types: set,
    keep_non_driving_matches: bool,
    fill_max_candidates: int,
    legality_repair_passes: int,
    strict_legal_routes: bool,
    verbose: bool = False,
) -> Dict[str, object]:
    """Improve correspondence by:
    1) optional non-driving match pruning,
    2) filling unmatched driving lanes with legal-route-aware candidates,
    3) local legality repair on successor edges.
    """
    if not bool(corr.get("enabled", False)):
        return corr

    v2_feats_raw = corr.get("v2_feats", {})
    carla_feats_raw = corr.get("carla_feats", {})
    lane_candidates_raw = corr.get("lane_candidates", {})
    lane_to_carla_raw = corr.get("lane_to_carla", {})
    v2_graph = corr.get("v2_graph", {})
    carla_succs_raw = corr.get("carla_successors", {})
    split_merges_raw = corr.get("split_merges", {})

    v2_feats: Dict[int, Dict[str, object]] = {int(_safe_float(k, -1)): v for k, v in dict(v2_feats_raw).items()}
    v2_feats = {k: v for k, v in v2_feats.items() if k >= 0 and isinstance(v, dict)}
    carla_feats: Dict[int, Dict[str, object]] = {int(_safe_float(k, -1)): v for k, v in dict(carla_feats_raw).items()}
    carla_feats = {k: v for k, v in carla_feats.items() if k >= 0 and isinstance(v, dict)}
    lane_to_carla: Dict[int, Dict[str, object]] = {int(_safe_float(k, -1)): dict(v) for k, v in dict(lane_to_carla_raw).items()}
    lane_to_carla = {k: v for k, v in lane_to_carla.items() if k >= 0}
    lane_candidates: Dict[int, List[Dict[str, object]]] = {}
    for k, rows in dict(lane_candidates_raw).items():
        ki = int(_safe_float(k, -1))
        if ki < 0 or not isinstance(rows, list):
            continue
        lane_candidates[ki] = [dict(r) for r in rows if isinstance(r, dict)]

    carla_successors = _normalize_int_set_map(carla_succs_raw)
    split_merges: Dict[int, List[int]] = {}
    if isinstance(split_merges_raw, dict):
        for k, vals in split_merges_raw.items():
            ki = int(_safe_float(k, -1))
            if ki < 0:
                continue
            if isinstance(vals, (list, tuple, set)):
                split_merges[ki] = [int(_safe_float(v, -1)) for v in vals if int(_safe_float(v, -1)) >= 0]

    succs = _normalize_int_set_map(v2_graph.get("successors", {}))
    preds = _normalize_int_set_map(v2_graph.get("predecessors", {}))
    adjacency = _normalize_int_set_map(v2_graph.get("adjacency", {}))

    driving_lanes = {
        int(li) for li, lf in v2_feats.items()
        if str(lf.get("lane_type", "")) in driving_types
    }
    initial_driving_mapping = {
        int(li): dict(info)
        for li, info in lane_to_carla.items()
        if li in driving_lanes
    }
    legal_before = _legal_route_stats(
        lane_to_carla=initial_driving_mapping,
        v2_graph=v2_graph if isinstance(v2_graph, dict) else {},
        carla_feats=carla_feats,
        carla_successors=carla_successors,
        driving_lanes=driving_lanes,
    )

    removed_non_driving = 0
    if not keep_non_driving_matches:
        pruned: Dict[int, Dict[str, object]] = {}
        for li, info in lane_to_carla.items():
            if li in driving_lanes:
                pruned[int(li)] = info
            else:
                removed_non_driving += 1
        lane_to_carla = pruned

    group_key_of: Dict[int, Tuple[int, int]] = {}
    for li, lf in v2_feats.items():
        group_key_of[int(li)] = (
            int(_safe_float(lf.get("road_id"), 0)),
            int(_safe_float(lf.get("lane_id"), 0)),
        )

    def _is_connected(ci_a: int, ci_b: int) -> bool:
        if ci_a < 0 or ci_b < 0:
            return False
        if ci_a == ci_b:
            return True
        return bool(
            ytm._carla_lines_connected(
                ci_a,
                ci_b,
                carla_feats,
                carla_successors,
                threshold_m=6.0,
            )
        )

    def _group_lines(mapping: Dict[int, Dict[str, object]]) -> Dict[Tuple[int, int], set]:
        out: Dict[Tuple[int, int], set] = defaultdict(set)
        for li, info in mapping.items():
            gk = group_key_of.get(int(li))
            if gk is None:
                continue
            ci = int(_safe_float(info.get("carla_line_index"), -1))
            if ci >= 0:
                out[gk].add(ci)
        return out

    def _candidate_cost(
        li: int,
        row: Dict[str, object],
        mapping: Dict[int, Dict[str, object]],
        group_lines_map: Dict[Tuple[int, int], set],
    ) -> float:
        ci = int(_safe_float(row.get("carla_line_index"), -1))
        if ci < 0:
            return float("inf")
        score = float(_safe_float(row.get("score"), float("inf")))
        if not math.isfinite(score):
            return float("inf")
        q = str(row.get("quality", "poor"))
        total = score + _candidate_quality_penalty(q)

        for pj in sorted(preds.get(int(li), set())):
            if pj not in driving_lanes:
                continue
            p_info = mapping.get(int(pj))
            if not isinstance(p_info, dict):
                continue
            cj = int(_safe_float(p_info.get("carla_line_index"), -1))
            if cj >= 0 and not _is_connected(cj, ci):
                total += 2.6
        for sj in sorted(succs.get(int(li), set())):
            if sj not in driving_lanes:
                continue
            s_info = mapping.get(int(sj))
            if not isinstance(s_info, dict):
                continue
            cj = int(_safe_float(s_info.get("carla_line_index"), -1))
            if cj >= 0 and not _is_connected(ci, cj):
                total += 2.6

        # Keep laterally adjacent lanes spatially coherent.
        for aj in sorted(adjacency.get(int(li), set())):
            if aj not in driving_lanes:
                continue
            a_info = mapping.get(int(aj))
            if not isinstance(a_info, dict):
                continue
            cj = int(_safe_float(a_info.get("carla_line_index"), -1))
            if cj < 0:
                continue
            if ci != cj and not _is_connected(ci, cj) and not _is_connected(cj, ci):
                total += 0.7

        gk = group_key_of.get(int(li))
        if gk is not None:
            same_group = group_lines_map.get(gk, set())
            if same_group:
                if ci in same_group:
                    total -= 1.8
                elif all((not _is_connected(ci, gx) and not _is_connected(gx, ci)) for gx in same_group):
                    total += 0.8
        return total

    def _local_edge_violations(li: int, ci: int, mapping: Dict[int, Dict[str, object]]) -> int:
        bad = 0
        for pj in sorted(preds.get(int(li), set())):
            if pj not in driving_lanes:
                continue
            p_info = mapping.get(int(pj))
            if not isinstance(p_info, dict):
                continue
            cj = int(_safe_float(p_info.get("carla_line_index"), -1))
            if cj >= 0 and not _is_connected(cj, ci):
                bad += 1
        for sj in sorted(succs.get(int(li), set())):
            if sj not in driving_lanes:
                continue
            s_info = mapping.get(int(sj))
            if not isinstance(s_info, dict):
                continue
            cj = int(_safe_float(s_info.get("carla_line_index"), -1))
            if cj >= 0 and not _is_connected(ci, cj):
                bad += 1
        return bad

    def _collect_illegal_edges(mapping: Dict[int, Dict[str, object]]) -> List[Tuple[int, int]]:
        bad_edges: List[Tuple[int, int]] = []
        for li_a in sorted(driving_lanes):
            info_a = mapping.get(int(li_a))
            if not isinstance(info_a, dict):
                continue
            ci_a = int(_safe_float(info_a.get("carla_line_index"), -1))
            if ci_a < 0:
                continue
            for li_b in sorted(succs.get(int(li_a), set())):
                if li_b not in driving_lanes:
                    continue
                info_b = mapping.get(int(li_b))
                if not isinstance(info_b, dict):
                    continue
                ci_b = int(_safe_float(info_b.get("carla_line_index"), -1))
                if ci_b < 0:
                    continue
                if not _is_connected(ci_a, ci_b):
                    bad_edges.append((int(li_a), int(li_b)))
        return bad_edges

    added_by_fill = 0
    # Fill unmatched driving lanes (allowing shared CARLA lines).
    for _ in range(4):
        changed = False
        group_lines_map = _group_lines(lane_to_carla)
        for li in sorted(driving_lanes):
            if li in lane_to_carla:
                continue
            rows = lane_candidates.get(int(li), [])
            if not rows:
                continue
            best_row: Optional[Dict[str, object]] = None
            best_cost = float("inf")
            max_rows = max(1, int(fill_max_candidates))
            for row in rows[:max_rows]:
                cost = _candidate_cost(li, row, lane_to_carla, group_lines_map)
                if cost < best_cost:
                    best_cost = cost
                    best_row = row
            if best_row is None:
                continue
            q = str(best_row.get("quality", "poor"))
            med = float(_safe_float(best_row.get("median_dist_m"), float("inf")))
            cov2 = float(_safe_float(best_row.get("coverage_2m"), 0.0))
            # Accept all non-poor. For poor, require at least modest geometric support.
            if q == "poor" and not (med <= 3.8 and cov2 >= 0.20 and best_cost <= 9.5):
                continue
            lane_to_carla[int(li)] = dict(best_row)
            added_by_fill += 1
            changed = True
        if not changed:
            break

    # Local legality repair for driving-lane successor edges.
    repaired_changes = 0
    for _ in range(max(0, int(legality_repair_passes))):
        pass_changed = False
        group_lines_map = _group_lines(lane_to_carla)
        for li in sorted(driving_lanes):
            cur_info = lane_to_carla.get(int(li))
            if not isinstance(cur_info, dict):
                continue
            cur_ci = int(_safe_float(cur_info.get("carla_line_index"), -1))
            if cur_ci < 0:
                continue
            rows = lane_candidates.get(int(li), [])
            if not rows:
                continue
            cur_cost = _candidate_cost(li, cur_info, lane_to_carla, group_lines_map)
            cur_total = cur_cost + 3.0 * float(_local_edge_violations(li, cur_ci, lane_to_carla))

            best_info = cur_info
            best_total = cur_total
            max_rows = max(1, int(fill_max_candidates))
            for row in rows[:max_rows]:
                ci = int(_safe_float(row.get("carla_line_index"), -1))
                if ci < 0:
                    continue
                q = str(row.get("quality", "poor"))
                med = float(_safe_float(row.get("median_dist_m"), float("inf")))
                cov2 = float(_safe_float(row.get("coverage_2m"), 0.0))
                if q == "poor" and not (med <= 3.8 and cov2 >= 0.20):
                    continue
                tmp = lane_to_carla.get(int(li))
                lane_to_carla[int(li)] = dict(row)
                c_cost = _candidate_cost(li, row, lane_to_carla, group_lines_map)
                c_total = c_cost + 3.0 * float(_local_edge_violations(li, ci, lane_to_carla))
                if c_total + 0.15 < best_total:
                    best_total = c_total
                    best_info = dict(row)
                if tmp is not None:
                    lane_to_carla[int(li)] = tmp
            if best_info is not cur_info:
                lane_to_carla[int(li)] = dict(best_info)
                repaired_changes += 1
                pass_changed = True
        if not pass_changed:
            break

    dropped_for_legality = 0
    remapped_for_legality = 0
    if bool(strict_legal_routes):
        for _ in range(200):
            illegal_edges = _collect_illegal_edges(lane_to_carla)
            if not illegal_edges:
                break
            bad_count: Counter = Counter()
            for li_a, li_b in illegal_edges:
                bad_count[int(li_a)] += 1
                bad_count[int(li_b)] += 1
            # Focus on the lane causing the most violations; prefer removing weaker matches.
            worst_li = max(
                bad_count.keys(),
                key=lambda li: (
                    int(bad_count[li]),
                    float(_safe_float(lane_to_carla.get(li, {}).get("score"), float("inf"))),
                    _candidate_quality_penalty(str(lane_to_carla.get(li, {}).get("quality", "poor"))),
                ),
            )
            rows = lane_candidates.get(int(worst_li), [])
            replaced = False
            if rows:
                max_rows = max(1, int(fill_max_candidates))
                cur_info = lane_to_carla.get(int(worst_li), {})
                cur_ci = int(_safe_float(cur_info.get("carla_line_index"), -1))
                cur_bad = _local_edge_violations(int(worst_li), cur_ci, lane_to_carla) if cur_ci >= 0 else 999
                best_row: Optional[Dict[str, object]] = None
                best_total = float("inf")
                group_lines_map = _group_lines(lane_to_carla)
                for row in rows[:max_rows]:
                    ci = int(_safe_float(row.get("carla_line_index"), -1))
                    if ci < 0:
                        continue
                    q = str(row.get("quality", "poor"))
                    med = float(_safe_float(row.get("median_dist_m"), float("inf")))
                    cov2 = float(_safe_float(row.get("coverage_2m"), 0.0))
                    if q == "poor" and not (med <= 3.8 and cov2 >= 0.20):
                        continue
                    tmp = lane_to_carla.get(int(worst_li))
                    lane_to_carla[int(worst_li)] = dict(row)
                    bad_now = _local_edge_violations(int(worst_li), ci, lane_to_carla)
                    total_now = float(bad_now) * 4.0 + _candidate_cost(int(worst_li), row, lane_to_carla, group_lines_map)
                    if total_now < best_total:
                        best_total = total_now
                        best_row = dict(row)
                    if tmp is not None:
                        lane_to_carla[int(worst_li)] = tmp
                if best_row is not None:
                    best_ci = int(_safe_float(best_row.get("carla_line_index"), -1))
                    best_bad = _local_edge_violations(int(worst_li), best_ci, lane_to_carla)
                    if best_bad < cur_bad:
                        lane_to_carla[int(worst_li)] = dict(best_row)
                        replaced = True
                        remapped_for_legality += 1
            if not replaced:
                if int(worst_li) in lane_to_carla:
                    lane_to_carla.pop(int(worst_li), None)
                    dropped_for_legality += 1

    # Rebuild reverse map.
    carla_to_lanes: Dict[int, List[int]] = {}
    for li, info in lane_to_carla.items():
        ci = int(_safe_float(info.get("carla_line_index"), -1))
        if ci >= 0:
            carla_to_lanes.setdefault(ci, []).append(int(li))
        for c in info.get("split_extra_carla_lines", []) or []:
            ci2 = int(_safe_float(c, -1))
            if ci2 >= 0:
                carla_to_lanes.setdefault(ci2, []).append(int(li))
    for ci, lset in carla_to_lanes.items():
        uniq = sorted(set(int(v) for v in lset))
        carla_to_lanes[ci] = uniq
        if len(uniq) > 1:
            for li in uniq:
                if li in lane_to_carla:
                    lane_to_carla[li]["shared_carla_line"] = True

    final_driving_mapping = {
        int(li): dict(info)
        for li, info in lane_to_carla.items()
        if li in driving_lanes
    }
    legal_after = _legal_route_stats(
        lane_to_carla=final_driving_mapping,
        v2_graph=v2_graph if isinstance(v2_graph, dict) else {},
        carla_feats=carla_feats,
        carla_successors=carla_successors,
        driving_lanes=driving_lanes,
    )

    if verbose:
        print(
            "[POST] "
            f"removed_non_driving={removed_non_driving}, "
            f"added_by_fill={added_by_fill}, repaired_changes={repaired_changes}, "
            f"strict_dropped={dropped_for_legality}, strict_remapped={remapped_for_legality}, "
            f"legal_edges {int(legal_before['ok'])}/{int(legal_before['total'])} -> "
            f"{int(legal_after['ok'])}/{int(legal_after['total'])}"
        )

    corr["lane_to_carla"] = lane_to_carla
    corr["carla_to_lanes"] = carla_to_lanes
    corr["split_merges"] = split_merges
    corr["postprocess"] = {
        "keep_non_driving_matches": bool(keep_non_driving_matches),
        "removed_non_driving": int(removed_non_driving),
        "added_by_fill": int(added_by_fill),
        "repaired_changes": int(repaired_changes),
        "strict_legal_routes": bool(strict_legal_routes),
        "dropped_for_legality": int(dropped_for_legality),
        "remapped_for_legality": int(remapped_for_legality),
        "driving_lane_count": int(len(driving_lanes)),
        "driving_mapped_count": int(len(final_driving_mapping)),
        "legal_before": legal_before,
        "legal_after": legal_after,
    }
    return corr


def _build_report_dataset(
    payload: Dict[str, object],
    corr: Dict[str, object],
    icp_stats: ICPStats,
    align_cfg: Dict[str, object],
    top_candidates: int,
) -> Dict[str, object]:
    lanes_raw = payload.get("map", {}).get("lanes", [])
    carla_lines_raw = payload.get("carla_map", {}).get("lines", [])
    lane_to_carla = dict(corr.get("lane_to_carla", {}))
    carla_to_lanes = dict(corr.get("carla_to_lanes", {}))
    lane_candidates = dict(corr.get("lane_candidates", {}))
    v2_feats = {
        int(_safe_float(k, -1)): v
        for k, v in dict(corr.get("v2_feats", {})).items()
        if int(_safe_float(k, -1)) >= 0 and isinstance(v, dict)
    }
    carla_feats = {
        int(_safe_float(k, -1)): v
        for k, v in dict(corr.get("carla_feats", {})).items()
        if int(_safe_float(k, -1)) >= 0 and isinstance(v, dict)
    }
    driving_types = {str(v) for v in (corr.get("driving_lane_types") or ["1"])}
    driving_lanes = {
        int(li) for li, lf in v2_feats.items()
        if str(lf.get("lane_type", "")) in driving_types
    }
    legal_stats = _legal_route_stats(
        lane_to_carla={int(k): dict(v) for k, v in lane_to_carla.items() if int(k) in driving_lanes and isinstance(v, dict)},
        v2_graph=dict(corr.get("v2_graph", {})) if isinstance(corr.get("v2_graph", {}), dict) else {},
        carla_feats=carla_feats,
        carla_successors=_normalize_int_set_map(corr.get("carla_successors", {})),
        driving_lanes=driving_lanes,
    )
    postprocess_meta = corr.get("postprocess", {}) if isinstance(corr.get("postprocess", {}), dict) else {}

    lanes: List[Dict[str, object]] = []
    lane_mapped_flag: Dict[int, bool] = {}
    q_counter: Counter = Counter()
    per_type_counter: Dict[str, Counter] = defaultdict(Counter)
    med_dists: List[float] = []
    p90_dists: List[float] = []
    cov2_vals: List[float] = []
    mono_vals: List[float] = []
    ang_vals: List[float] = []

    for lane in lanes_raw if isinstance(lanes_raw, list) else []:
        if not isinstance(lane, dict):
            continue
        li = int(_safe_float(lane.get("index"), -1))
        match = lane_to_carla.get(li)
        match_payload: Optional[Dict[str, object]] = None
        quality = "unmatched"
        if isinstance(match, dict):
            quality = str(match.get("quality", "poor"))
            match_payload = {
                "carla_line_index": int(_safe_float(match.get("carla_line_index"), -1)),
                "quality": quality,
                "score": float(_safe_float(match.get("score"), float("inf"))),
                "median_dist_m": float(_safe_float(match.get("median_dist_m"), float("inf"))),
                "p90_dist_m": float(_safe_float(match.get("p90_dist_m"), float("inf"))),
                "coverage_2m": float(_safe_float(match.get("coverage_2m"), 0.0)),
                "angle_median_deg": float(_safe_float(match.get("angle_median_deg"), 180.0)),
                "monotonic_ratio": float(_safe_float(match.get("monotonic_ratio"), 0.0)),
                "length_ratio": float(_safe_float(match.get("length_ratio"), 0.0)),
                "reversed": bool(match.get("reversed", False)),
                "shared_carla_line": bool(match.get("shared_carla_line", False)),
                "split_extra_carla_lines": [
                    int(_safe_float(v, -1))
                    for v in (match.get("split_extra_carla_lines") or [])
                    if int(_safe_float(v, -1)) >= 0
                ],
            }
            med_dists.append(float(match_payload["median_dist_m"]))
            p90_dists.append(float(match_payload["p90_dist_m"]))
            cov2_vals.append(float(match_payload["coverage_2m"]))
            mono_vals.append(float(match_payload["monotonic_ratio"]))
            ang_vals.append(float(match_payload["angle_median_deg"]))

        q_counter[quality] += 1
        per_type_counter[str(lane.get("lane_type", "unknown"))][quality] += 1
        lanes.append(
            {
                "index": li,
                "road_id": int(_safe_float(lane.get("road_id"), 0)),
                "lane_id": int(_safe_float(lane.get("lane_id"), 0)),
                "lane_type": str(lane.get("lane_type", "")),
                "label": _lane_label(lane),
                "label_x": float(_safe_float(lane.get("label_x"), 0.0)),
                "label_y": float(_safe_float(lane.get("label_y"), 0.0)),
                "polyline": lane.get("polyline") or [],
                "quality": quality,
                "quality_color": _quality_color(quality),
                "match": match_payload,
            }
        )
        lane_mapped_flag[int(li)] = bool(match_payload is not None)

    def _poly_to_xy(poly_obj: object) -> np.ndarray:
        if not isinstance(poly_obj, list):
            return np.zeros((0, 2), dtype=np.float64)
        pts: List[Tuple[float, float]] = []
        for p in poly_obj:
            if not (isinstance(p, (list, tuple)) and len(p) >= 2):
                continue
            x = _safe_float(p[0], float("nan"))
            y = _safe_float(p[1], float("nan"))
            if math.isfinite(x) and math.isfinite(y):
                pts.append((float(x), float(y)))
        if not pts:
            return np.zeros((0, 2), dtype=np.float64)
        return np.asarray(pts, dtype=np.float64)

    all_v2_chunks: List[np.ndarray] = []
    type1_chunks: List[np.ndarray] = []
    type1_point_lane: List[int] = []
    for lane in lanes:
        arr = _poly_to_xy(lane.get("polyline"))
        if arr.shape[0] <= 0:
            continue
        all_v2_chunks.append(arr)
        if str(lane.get("lane_type", "")) == "1":
            type1_chunks.append(arr)
            type1_point_lane.extend([int(_safe_float(lane.get("index"), -1))] * int(arr.shape[0]))

    all_v2_pts = np.vstack(all_v2_chunks) if all_v2_chunks else np.zeros((0, 2), dtype=np.float64)
    type1_v2_pts = np.vstack(type1_chunks) if type1_chunks else np.zeros((0, 2), dtype=np.float64)
    all_v2_tree = cKDTree(all_v2_pts) if all_v2_pts.shape[0] > 0 else None
    type1_v2_tree = cKDTree(type1_v2_pts) if type1_v2_pts.shape[0] > 0 else None

    if all_v2_pts.shape[0] > 0:
        v2_min_x = float(np.min(all_v2_pts[:, 0]))
        v2_max_x = float(np.max(all_v2_pts[:, 0]))
        v2_min_y = float(np.min(all_v2_pts[:, 1]))
        v2_max_y = float(np.max(all_v2_pts[:, 1]))
    else:
        v2_min_x = v2_max_x = v2_min_y = v2_max_y = float("nan")

    def _accept_supplemental_counterpart(
        quality: str,
        score: float,
        med: float,
        ang: float,
        cov2: float,
        mono: float,
    ) -> bool:
        q = str(quality)
        if not math.isfinite(score) or not math.isfinite(med) or not math.isfinite(ang):
            return False
        if q == "high":
            return bool(med <= 1.6 and ang <= 18.0 and score <= 4.5 and mono >= 0.55)
        if q == "medium":
            return bool(med <= 1.35 and ang <= 13.0 and score <= 4.3 and mono >= 0.55)
        if q == "low":
            return bool(med <= 1.0 and ang <= 8.5 and score <= 4.5 and cov2 >= 0.40 and mono >= 0.65)
        return False

    supplemental_best_by_carla: Dict[int, Dict[str, object]] = {}
    for li_raw, rows in lane_candidates.items():
        li = int(_safe_float(li_raw, -1))
        if li < 0 or not isinstance(rows, list):
            continue
        lf = v2_feats.get(int(li), {})
        lane_type = str(lf.get("lane_type", ""))
        for row in rows:
            if not isinstance(row, dict):
                continue
            ci = int(_safe_float(row.get("carla_line_index"), -1))
            if ci < 0:
                continue
            primary_lanes = carla_to_lanes.get(ci)
            if primary_lanes is None:
                primary_lanes = carla_to_lanes.get(str(ci))
            if isinstance(primary_lanes, (list, tuple)) and len(primary_lanes) > 0:
                continue
            quality = str(row.get("quality", "poor"))
            score = float(_safe_float(row.get("score"), float("inf")))
            med = float(_safe_float(row.get("median_dist_m"), float("inf")))
            ang = float(_safe_float(row.get("angle_median_deg"), 180.0))
            cov2 = float(_safe_float(row.get("coverage_2m"), 0.0))
            mono = float(_safe_float(row.get("monotonic_ratio"), 0.0))
            if not _accept_supplemental_counterpart(quality, score, med, ang, cov2, mono):
                continue
            q_rank = 0 if quality == "high" else 1 if quality == "medium" else 2
            rank = (q_rank, score, med, ang, -mono, int(li))
            prev = supplemental_best_by_carla.get(int(ci))
            prev_rank = prev.get("rank") if isinstance(prev, dict) else None
            if prev is None or not isinstance(prev_rank, tuple) or rank < prev_rank:
                supplemental_best_by_carla[int(ci)] = {
                    "rank": rank,
                    "lane_index": int(li),
                    "lane_type": lane_type,
                    "quality": quality,
                    "score": score,
                    "median_dist_m": med,
                    "angle_median_deg": ang,
                    "coverage_2m": cov2,
                    "monotonic_ratio": mono,
                }

    carla_lines: List[Dict[str, object]] = []
    for i, ln in enumerate(carla_lines_raw if isinstance(carla_lines_raw, list) else []):
        if isinstance(ln, dict):
            poly = ln.get("polyline") or []
        else:
            poly = ln if isinstance(ln, list) else []
        if not isinstance(poly, list) or len(poly) < 2:
            continue
        mid = poly[len(poly) // 2]
        if not (isinstance(mid, (list, tuple)) and len(mid) >= 2):
            mid = [0.0, 0.0]
        matched_raw = carla_to_lanes.get(i)
        if matched_raw is None:
            matched_raw = carla_to_lanes.get(str(i), [])
        matched_v2 = [int(_safe_float(v, -1)) for v in (matched_raw or []) if int(_safe_float(v, -1)) >= 0]
        matched_v2 = sorted(set(matched_v2))
        supp = supplemental_best_by_carla.get(int(i))
        supplemental_v2 = [int(_safe_float(supp.get("lane_index"), -1))] if isinstance(supp, dict) else []
        supplemental_v2 = [v for v in supplemental_v2 if v >= 0 and v not in matched_v2]
        counterpart_v2 = sorted(set(matched_v2 + supplemental_v2))
        has_counterpart = bool(len(counterpart_v2) > 0)
        counterpart_source = "primary" if matched_v2 else ("supplemental" if supplemental_v2 else "none")
        best_q = "unmatched"
        for li in matched_v2:
            m = lane_to_carla.get(int(li))
            if isinstance(m, dict):
                qv = str(m.get("quality", "poor"))
                if _quality_rank(qv) > _quality_rank(best_q):
                    best_q = qv
        if best_q == "unmatched" and isinstance(supp, dict):
            best_q = str(supp.get("quality", "unmatched"))

        arr = _poly_to_xy(poly)
        dist_any_min = float("inf")
        dist_type1_min = float("inf")
        nearest_type1_lane = -1
        nearest_type1_lane_mapped = False
        in_v2_bbox20 = False
        if arr.shape[0] > 0:
            if all_v2_tree is not None:
                d_any, _ = all_v2_tree.query(arr, k=1)
                d_any_arr = np.asarray(d_any, dtype=np.float64)
                if d_any_arr.size > 0:
                    dist_any_min = float(np.min(d_any_arr))
            if type1_v2_tree is not None and len(type1_point_lane) > 0:
                d_t1, idx_t1 = type1_v2_tree.query(arr, k=1)
                d_t1_arr = np.asarray(d_t1, dtype=np.float64)
                idx_t1_arr = np.asarray(idx_t1, dtype=np.int64)
                if d_t1_arr.size > 0 and idx_t1_arr.size > 0:
                    min_j = int(np.argmin(d_t1_arr))
                    dist_type1_min = float(d_t1_arr[min_j])
                    pt_idx = int(idx_t1_arr[min_j])
                    if 0 <= pt_idx < len(type1_point_lane):
                        nearest_type1_lane = int(type1_point_lane[pt_idx])
                        nearest_type1_lane_mapped = bool(lane_mapped_flag.get(nearest_type1_lane, False))
            if all(math.isfinite(v) for v in [v2_min_x, v2_max_x, v2_min_y, v2_max_y]):
                lx0 = float(np.min(arr[:, 0]))
                lx1 = float(np.max(arr[:, 0]))
                ly0 = float(np.min(arr[:, 1]))
                ly1 = float(np.max(arr[:, 1]))
                m = 20.0
                in_v2_bbox20 = not (
                    lx1 < (v2_min_x - m)
                    or lx0 > (v2_max_x + m)
                    or ly1 < (v2_min_y - m)
                    or ly0 > (v2_max_y + m)
                )

        is_primary_unmatched = len(matched_v2) <= 0
        is_unmatched = not bool(has_counterpart)
        unmatched_bucket = "matched"
        if is_unmatched:
            if not in_v2_bbox20:
                unmatched_bucket = "out_bbox"
            elif math.isfinite(dist_type1_min) and dist_type1_min <= 1.0:
                unmatched_bucket = "near_1m"
            elif math.isfinite(dist_type1_min) and dist_type1_min <= 5.0:
                unmatched_bucket = "near_5m"
            elif math.isfinite(dist_type1_min) and dist_type1_min <= 20.0:
                unmatched_bucket = "near_20m"
            else:
                unmatched_bucket = "far_20m"
        likely_true_miss = bool(
            is_unmatched
            and in_v2_bbox20
            and math.isfinite(dist_type1_min)
            and dist_type1_min <= 1.0
            and nearest_type1_lane_mapped
        )

        dist_any_out: Optional[float] = float(dist_any_min) if math.isfinite(dist_any_min) else None
        dist_type1_out: Optional[float] = float(dist_type1_min) if math.isfinite(dist_type1_min) else None
        carla_lines.append(
            {
                "index": int(i),
                "label_x": float(_safe_float(mid[0], 0.0)),
                "label_y": float(_safe_float(mid[1], 0.0)),
                "polyline": poly,
                "matched_v2_lane_indices": matched_v2,
                "supplemental_v2_lane_indices": supplemental_v2,
                "counterpart_v2_lane_indices": counterpart_v2,
                "counterpart_source": str(counterpart_source),
                "has_counterpart": bool(has_counterpart),
                "best_quality": best_q,
                "best_quality_color": _quality_color(best_q),
                "is_primary_unmatched": bool(is_primary_unmatched),
                "is_unmatched": bool(is_unmatched),
                "unmatched_bucket": str(unmatched_bucket),
                "in_v2_bbox_plus_20m": bool(in_v2_bbox20),
                "likely_true_miss": bool(likely_true_miss),
                "dist_to_any_v2_min_m": dist_any_out,
                "dist_to_type1_v2_min_m": dist_type1_out,
                "nearest_type1_lane_index": int(nearest_type1_lane),
                "nearest_type1_lane_mapped": bool(nearest_type1_lane_mapped),
                "supplemental_match": {
                    "lane_index": int(_safe_float(supp.get("lane_index"), -1)),
                    "lane_type": str(supp.get("lane_type", "")),
                    "quality": str(supp.get("quality", "unmatched")),
                    "score": float(_safe_float(supp.get("score"), float("inf"))),
                    "median_dist_m": float(_safe_float(supp.get("median_dist_m"), float("inf"))),
                    "angle_median_deg": float(_safe_float(supp.get("angle_median_deg"), 180.0)),
                    "coverage_2m": float(_safe_float(supp.get("coverage_2m"), 0.0)),
                    "monotonic_ratio": float(_safe_float(supp.get("monotonic_ratio"), 0.0)),
                } if isinstance(supp, dict) else None,
            }
        )

    candidate_report: Dict[str, List[Dict[str, object]]] = {}
    for li_raw, rows in lane_candidates.items():
        li = int(_safe_float(li_raw, -1))
        if li < 0 or not isinstance(rows, list):
            continue
        out_rows: List[Dict[str, object]] = []
        for row in rows[: max(1, int(top_candidates))]:
            if not isinstance(row, dict):
                continue
            out_rows.append(
                {
                    "carla_line_index": int(_safe_float(row.get("carla_line_index"), -1)),
                    "quality": str(row.get("quality", "poor")),
                    "score": float(_safe_float(row.get("score"), float("inf"))),
                    "median_dist_m": float(_safe_float(row.get("median_dist_m"), float("inf"))),
                    "p90_dist_m": float(_safe_float(row.get("p90_dist_m"), float("inf"))),
                    "coverage_2m": float(_safe_float(row.get("coverage_2m"), 0.0)),
                    "angle_median_deg": float(_safe_float(row.get("angle_median_deg"), 180.0)),
                    "monotonic_ratio": float(_safe_float(row.get("monotonic_ratio"), 0.0)),
                    "length_ratio": float(_safe_float(row.get("length_ratio"), 0.0)),
                    "reversed": bool(row.get("reversed", False)),
                }
            )
        candidate_report[str(li)] = out_rows

    used_carla = set()
    for lane in lanes:
        m = lane.get("match")
        if isinstance(m, dict):
            ci = int(_safe_float(m.get("carla_line_index"), -1))
            if ci >= 0:
                used_carla.add(ci)
            for c in (m.get("split_extra_carla_lines") or []):
                ci2 = int(_safe_float(c, -1))
                if ci2 >= 0:
                    used_carla.add(ci2)

    metric_summary = {
        "median_of_median_dist_m": float(statistics.median(med_dists)) if med_dists else float("nan"),
        "mean_of_median_dist_m": float(statistics.fmean(med_dists)) if med_dists else float("nan"),
        "median_of_p90_dist_m": float(statistics.median(p90_dists)) if p90_dists else float("nan"),
        "mean_of_coverage_2m": float(statistics.fmean(cov2_vals)) if cov2_vals else float("nan"),
        "mean_of_monotonic_ratio": float(statistics.fmean(mono_vals)) if mono_vals else float("nan"),
        "mean_of_angle_median_deg": float(statistics.fmean(ang_vals)) if ang_vals else float("nan"),
    }

    counterpart_carla_rows = [c for c in carla_lines if bool(c.get("has_counterpart", False))]
    supplemental_only_rows = [c for c in carla_lines if str(c.get("counterpart_source", "none")) == "supplemental"]
    unmatched_carla_rows = [c for c in carla_lines if bool(c.get("is_unmatched", False))]
    unmatched_bucket_counts = Counter(str(c.get("unmatched_bucket", "unknown")) for c in unmatched_carla_rows)
    likely_true_miss_count = int(sum(1 for c in unmatched_carla_rows if bool(c.get("likely_true_miss", False))))

    summary = {
        "v2_lane_count": int(len(lanes)),
        "carla_line_count": int(len(carla_lines)),
        "mapped_lane_count": int(sum(1 for x in lanes if x.get("match") is not None)),
        "usable_lane_count": int(sum(1 for x in lanes if str(x.get("quality", "")) in {"high", "medium", "low"})),
        "poor_lane_count": int(sum(1 for x in lanes if str(x.get("quality", "")) == "poor")),
        "unmatched_lane_count": int(sum(1 for x in lanes if str(x.get("quality", "")) == "unmatched")),
        "mapped_carla_line_count": int(len(used_carla)),
        "counterpart_carla_line_count": int(len(counterpart_carla_rows)),
        "supplemental_counterpart_carla_count": int(len(supplemental_only_rows)),
        "quality_counts": {k: int(v) for k, v in dict(q_counter).items()},
        "lane_type_quality_counts": {
            lane_type: {q: int(c) for q, c in dict(cnt).items()}
            for lane_type, cnt in per_type_counter.items()
        },
        "driving_lane_count": int(len(driving_lanes)),
        "driving_mapped_count": int(
            sum(
                1 for li in driving_lanes
                if isinstance(lane_to_carla.get(int(li)), dict)
            )
        ),
        "legal_driving_edge_total": int(_safe_float(legal_stats.get("total"), 0.0)),
        "legal_driving_edge_ok": int(_safe_float(legal_stats.get("ok"), 0.0)),
        "legal_driving_edge_bad": int(_safe_float(legal_stats.get("bad"), 0.0)),
        "legal_driving_edge_ratio": float(_safe_float(legal_stats.get("ratio"), float("nan"))),
        "unmatched_carla_line_count": int(len(unmatched_carla_rows)),
        "likely_true_miss_carla_count": int(likely_true_miss_count),
        "unmatched_carla_bucket_counts": {k: int(v) for k, v in dict(unmatched_bucket_counts).items()},
        "postprocess": postprocess_meta,
    }

    align_meta = {
        "source_cfg": {
            "scale": float(_safe_float(align_cfg.get("scale"), 1.0)),
            "theta_deg": float(_safe_float(align_cfg.get("theta_deg"), 0.0)),
            "tx": float(_safe_float(align_cfg.get("tx"), 0.0)),
            "ty": float(_safe_float(align_cfg.get("ty"), 0.0)),
            "flip_y": bool(align_cfg.get("flip_y", False)),
            "source_path": str(align_cfg.get("source_path", "")),
        },
        "icp_refine": {
            "applied": bool(icp_stats.applied),
            "iterations": int(icp_stats.iterations),
            "inliers": int(icp_stats.inliers),
            "before_median_m": float(icp_stats.before_median),
            "before_mean_m": float(icp_stats.before_mean),
            "before_p90_m": float(icp_stats.before_p90),
            "after_median_m": float(icp_stats.after_median),
            "after_mean_m": float(icp_stats.after_mean),
            "after_p90_m": float(icp_stats.after_p90),
            "delta_theta_deg": float(icp_stats.delta_theta_deg),
            "delta_tx": float(icp_stats.delta_tx),
            "delta_ty": float(icp_stats.delta_ty),
            "history": icp_stats.history,
        },
    }

    return {
        "summary": summary,
        "metric_summary": metric_summary,
        "alignment": align_meta,
        "v2_lanes": lanes,
        "carla_lines": carla_lines,
        "lane_candidates": candidate_report,
    }


def _build_html(report: Dict[str, object]) -> str:
    dataset_json = json.dumps(report, ensure_ascii=True, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Lane Correspondence Diagnostics</title>
  <style>
    :root {{
      --bg: #0c1720;
      --panel: #102230;
      --panel2: #0f1d2a;
      --border: #2c4458;
      --text: #e4ecf1;
      --muted: #9fb0bc;
      --accent: #69d2ff;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; height: 100%; background: var(--bg); color: var(--text); font-family: "Segoe UI", sans-serif; }}
    #app {{ height: 100%; display: grid; grid-template-columns: 1fr 420px; gap: 10px; padding: 10px; }}
    #main {{ border: 1px solid var(--border); border-radius: 10px; overflow: hidden; position: relative; background: #09131b; }}
    #canvas {{ width: 100%; height: 100%; display: block; }}
    #hud {{
      position: absolute; left: 10px; top: 10px; z-index: 5;
      background: rgba(8, 18, 26, 0.88); border: 1px solid #37546b; border-radius: 8px;
      padding: 8px 10px; font-size: 12px; line-height: 1.35;
    }}
    #hud .line {{ margin-bottom: 2px; }}
    #sidebar {{ border: 1px solid var(--border); border-radius: 10px; overflow: auto; background: var(--panel); padding: 10px; }}
    .section {{ border: 1px solid #2f4a60; border-radius: 8px; background: var(--panel2); margin-bottom: 10px; padding: 8px; }}
    h3 {{ margin: 0 0 8px 0; font-size: 13px; color: var(--accent); }}
    .row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 12px; }}
    .rowWrap {{ display: flex; flex-wrap: wrap; gap: 8px 10px; margin-bottom: 6px; font-size: 11px; color: #cddae3; }}
    .row input[type="text"] {{
      flex: 1; min-width: 0; border: 1px solid #3a5a73; border-radius: 6px; background: #0b1d2a; color: var(--text); padding: 5px 8px;
    }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; color: #c9d8e2; white-space: pre-wrap; }}
    .chip {{ display: inline-flex; align-items: center; gap: 6px; border: 1px solid #3b5e78; border-radius: 999px; padding: 2px 8px; font-size: 11px; margin: 2px 4px 0 0; }}
    .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
    .legendGrid {{ display: grid; grid-template-columns: 26px 1fr; gap: 6px 8px; margin-top: 6px; font-size: 11px; color: #cad8e2; }}
    .legendSwatch {{ height: 0; border-top: 3px solid #fff; align-self: center; position: relative; }}
    .legendV2 {{ border-top-color: #f5a65f; border-top-width: 3px; }}
    .legendCarla {{ border-top-color: #63c7ef; border-top-width: 2px; border-top-style: dashed; }}
    .legendCarlaMatched {{ border-top-color: #2be8ff; border-top-width: 3px; }}
    .legendMatch {{ border-top-color: #d9a8ff; border-top-width: 2px; border-top-style: dashed; }}
    .legendMatch::after {{
      content: "";
      position: absolute;
      right: -1px;
      top: -5px;
      border-top: 5px solid transparent;
      border-bottom: 5px solid transparent;
      border-left: 8px solid #d9a8ff;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 11px; }}
    th, td {{ border-bottom: 1px solid #2b4357; padding: 4px 6px; text-align: left; }}
    th {{ color: #b5c8d6; font-weight: 600; position: sticky; top: 0; background: #112535; }}
    tbody tr {{ cursor: pointer; }}
    tbody tr:hover {{ background: rgba(86, 169, 214, 0.14); }}
    tbody tr.sel {{ background: rgba(105, 210, 255, 0.25); }}
    #laneTableWrap {{ max-height: 290px; overflow: auto; border: 1px solid #2d475d; border-radius: 6px; }}
    #candidateTableWrap {{ max-height: 220px; overflow: auto; border: 1px solid #2d475d; border-radius: 6px; }}
    .small {{ font-size: 11px; color: var(--muted); }}
    @media (max-width: 1200px) {{
      #app {{ grid-template-columns: 1fr; grid-template-rows: 58% 42%; }}
      #sidebar {{ min-height: 300px; }}
    }}
  </style>
</head>
<body>
  <div id="app">
    <div id="main">
      <canvas id="canvas"></canvas>
      <div id="hud">
        <div class="line" id="hudSummary">-</div>
        <div class="line" id="hudFilter">-</div>
        <div class="line" id="hudSelected">-</div>
      </div>
    </div>
    <aside id="sidebar">
      <div class="section">
        <h3>Overview</h3>
        <div id="summaryText" class="mono">-</div>
        <div id="qualityChips"></div>
        <div id="unmatchedChips"></div>
      </div>

      <div class="section">
        <h3>Display</h3>
        <label class="row"><input type="checkbox" id="showV2" checked />Show V2 lanes</label>
        <label class="row"><input type="checkbox" id="showCarla" checked />Show CARLA lines</label>
        <label class="row"><input type="checkbox" id="showCarlaMatchedLayer" />Show CARLA counterpart-only layer</label>
        <label class="row"><input type="checkbox" id="showConnectors" checked />Show match arrows</label>
        <label class="row"><input type="checkbox" id="showPairTags" />Show pair IDs (v2->c) on map</label>
        <label class="row"><input type="checkbox" id="showUnmatchedCarla" />Highlight unmatched CARLA lines</label>
        <div class="rowWrap">
          <label><input type="checkbox" class="uf" value="near_1m" checked />&lt;=1m</label>
          <label><input type="checkbox" class="uf" value="near_5m" checked />1-5m</label>
          <label><input type="checkbox" class="uf" value="near_20m" checked />5-20m</label>
          <label><input type="checkbox" class="uf" value="far_20m" checked />&gt;20m</label>
          <label><input type="checkbox" class="uf" value="out_bbox" checked />outside bbox+20m</label>
        </div>
        <label class="row"><input type="checkbox" id="onlyLikelyTrueMiss" />Only likely true misses</label>
        <label class="row"><input type="checkbox" id="showLabels" />Show labels</label>
        <label class="row"><input type="checkbox" id="onlySelected" />Draw only selected lane + match</label>
        <div class="legendGrid">
          <span class="legendSwatch legendV2"></span><span>V2XPNP lane centerline (solid)</span>
          <span class="legendSwatch legendCarla"></span><span>CARLA centerline (dashed)</span>
          <span class="legendSwatch legendCarlaMatched"></span><span>CARLA lines that have V2 counterparts</span>
          <span class="legendSwatch legendMatch"></span><span>Lane correspondence link, V2 -> CARLA</span>
        </div>
        <div class="row small">CARLA colors: cyan=primary match, green=supplemental counterpart, slate=no counterpart.</div>
        <div class="row small">Unmatched overlay colors: pink <=1m, orange 1-5m, yellow 5-20m, gray >20m, slate out-of-region.</div>
        <div class="row small">Mouse wheel: zoom, drag: pan, click lane/table row: inspect.</div>
      </div>

      <div class="section">
        <h3>Filters</h3>
        <div class="row"><input id="searchBox" type="text" placeholder="Search: road/lane/type/index..." /></div>
        <div class="row">
          <label><input type="checkbox" class="qf" value="high" checked />high</label>
          <label><input type="checkbox" class="qf" value="medium" checked />medium</label>
          <label><input type="checkbox" class="qf" value="low" checked />low</label>
          <label><input type="checkbox" class="qf" value="poor" checked />poor</label>
          <label><input type="checkbox" class="qf" value="unmatched" checked />unmatched</label>
        </div>
      </div>

      <div class="section">
        <h3>Lane Matches</h3>
        <div id="laneTableWrap">
          <table>
            <thead>
              <tr>
                <th>V2 lane</th><th>Q</th><th>CARLA</th><th>med</th><th>p90</th><th>cov2</th>
              </tr>
            </thead>
            <tbody id="laneTableBody"></tbody>
          </table>
        </div>
      </div>

      <div class="section">
        <h3>Selected Lane</h3>
        <div id="selectedInfo" class="mono">Click a lane to inspect match metrics.</div>
      </div>

      <div class="section">
        <h3>Top Candidates</h3>
        <div id="candidateTableWrap">
          <table>
            <thead>
              <tr>
                <th>CARLA</th><th>Q</th><th>score</th><th>med</th><th>p90</th><th>cov2</th><th>ang</th>
              </tr>
            </thead>
            <tbody id="candidateBody"></tbody>
          </table>
        </div>
      </div>
    </aside>
  </div>

  <script id="dataset" type="application/json">{dataset_json}</script>
  <script>
  (() => {{
    'use strict';
    const DATA = JSON.parse(document.getElementById('dataset').textContent);

    const lanes = DATA.v2_lanes || [];
    const carlaLines = DATA.carla_lines || [];
    const laneCandidates = DATA.lane_candidates || {{}};
    const summary = DATA.summary || {{}};
    const metricSummary = DATA.metric_summary || {{}};
    const align = DATA.alignment || {{}};

    const qualityColor = {{
      high: '#2ecc71',
      medium: '#f1c40f',
      low: '#e67e22',
      poor: '#e74c3c',
      unmatched: '#95a5a6'
    }};

    const unmatchedBucketColor = {{
      near_1m: '#ff59d2',
      near_5m: '#ff9f43',
      near_20m: '#f4d35e',
      far_20m: '#8d99a6',
      out_bbox: '#516273',
    }};

    const unmatchedBucketLabel = {{
      near_1m: '<=1m to V2 type1',
      near_5m: '1-5m to V2 type1',
      near_20m: '5-20m to V2 type1',
      far_20m: '>20m to V2 type1',
      out_bbox: 'outside V2 bbox+20m',
    }};

    const sourceColor = {{
      v2Selected: '#ffd79b',
      carlaMapped: '#63c7ef',
      carlaSupplemental: '#7ef2ad',
      carlaUnmapped: '#2f4e63',
      carlaSelected: '#91e4ff',
      pair: '#d9a8ff',
      pairSelected: '#fff08f',
      carlaMatchedOnly: '#2be8ff',
      labelBg: 'rgba(10, 22, 32, 0.92)',
      labelStroke: '#38566e',
      labelText: '#f0f6fb'
    }};

    const state = {{
      selectedLaneIndex: null,
      filterQuality: new Set(['high','medium','low','poor','unmatched']),
      search: '',
      showV2: true,
      showCarla: true,
      showCarlaMatchedLayer: false,
      showConnectors: true,
      showPairTags: false,
      showUnmatchedCarla: false,
      onlyLikelyTrueMiss: false,
      unmatchedBucketFilter: new Set(['near_1m', 'near_5m', 'near_20m', 'far_20m', 'out_bbox']),
      showLabels: false,
      onlySelected: false,
      view: {{ cx: 0, cy: 0, scale: 1 }},
      drag: null,
    }};

    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const hudSummary = document.getElementById('hudSummary');
    const hudFilter = document.getElementById('hudFilter');
    const hudSelected = document.getElementById('hudSelected');
    const summaryText = document.getElementById('summaryText');
    const qualityChips = document.getElementById('qualityChips');
    const unmatchedChips = document.getElementById('unmatchedChips');
    const laneTableBody = document.getElementById('laneTableBody');
    const selectedInfo = document.getElementById('selectedInfo');
    const candidateBody = document.getElementById('candidateBody');

    const showV2 = document.getElementById('showV2');
    const showCarla = document.getElementById('showCarla');
    const showCarlaMatchedLayer = document.getElementById('showCarlaMatchedLayer');
    const showConnectors = document.getElementById('showConnectors');
    const showPairTags = document.getElementById('showPairTags');
    const showUnmatchedCarla = document.getElementById('showUnmatchedCarla');
    const onlyLikelyTrueMiss = document.getElementById('onlyLikelyTrueMiss');
    const showLabels = document.getElementById('showLabels');
    const onlySelected = document.getElementById('onlySelected');
    const searchBox = document.getElementById('searchBox');
    const qChecks = [...document.querySelectorAll('.qf')];
    const unmatchedChecks = [...document.querySelectorAll('.uf')];

    function laneQuality(lane) {{
      return (lane && lane.quality) ? String(lane.quality) : 'unmatched';
    }}

    function laneMatchesFilter(lane) {{
      const q = laneQuality(lane);
      if (!state.filterQuality.has(q)) return false;
      if (!state.search) return true;
      const s = state.search.toLowerCase();
      const txt = `${{lane.label || ''}} r${{lane.road_id}} l${{lane.lane_id}} t${{lane.lane_type}} idx${{lane.index}}`.toLowerCase();
      return txt.includes(s);
    }}

    function filteredLanes() {{
      return lanes.filter(laneMatchesFilter);
    }}

    function fitView() {{
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      const push = (poly) => {{
        for (const p of (poly || [])) {{
          if (!Array.isArray(p) || p.length < 2) continue;
          const x = Number(p[0]), y = Number(p[1]);
          if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
          minX = Math.min(minX, x); maxX = Math.max(maxX, x);
          minY = Math.min(minY, y); maxY = Math.max(maxY, y);
        }}
      }};
      for (const l of lanes) push(l.polyline);
      for (const c of carlaLines) push(c.polyline);
      if (!Number.isFinite(minX) || !Number.isFinite(minY)) {{
        state.view = {{ cx: 0, cy: 0, scale: 1 }};
        return;
      }}
      const w = Math.max(1e-3, maxX - minX);
      const h = Math.max(1e-3, maxY - minY);
      state.view.cx = (minX + maxX) / 2.0;
      state.view.cy = (minY + maxY) / 2.0;
      const sx = (canvas.clientWidth * 0.9) / w;
      const sy = (canvas.clientHeight * 0.9) / h;
      state.view.scale = Math.max(0.04, Math.min(800, Math.min(sx, sy)));
    }}

    function resizeCanvas() {{
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
      canvas.height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }}

    function worldToScreen(x, y) {{
      const sx = (x - state.view.cx) * state.view.scale + canvas.clientWidth / 2;
      const sy = canvas.clientHeight / 2 - (y - state.view.cy) * state.view.scale;
      return {{ x: sx, y: sy }};
    }}

    function drawPolyline(poly, color, width, alpha, dash) {{
      if (!Array.isArray(poly) || poly.length < 2) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.globalAlpha = alpha;
      ctx.setLineDash(Array.isArray(dash) ? dash : []);
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.beginPath();
      for (let i = 0; i < poly.length; i++) {{
        const p = poly[i];
        if (!Array.isArray(p) || p.length < 2) continue;
        const s = worldToScreen(Number(p[0]), Number(p[1]));
        if (i === 0) ctx.moveTo(s.x, s.y);
        else ctx.lineTo(s.x, s.y);
      }}
      ctx.stroke();
      ctx.globalAlpha = 1.0;
      ctx.setLineDash([]);
    }}

    function drawArrow(a, b, color, width, alpha, dash) {{
      if (!a || !b) return;
      const dx = Number(b.x) - Number(a.x);
      const dy = Number(b.y) - Number(a.y);
      const len = Math.hypot(dx, dy);
      if (!(len > 1e-3)) return;
      const ux = dx / len;
      const uy = dy / len;
      const head = Math.max(8.0, 4.0 + 1.7 * Number(width || 1.0));
      const endX = Number(b.x) - ux * head;
      const endY = Number(b.y) - uy * head;

      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.globalAlpha = alpha;
      ctx.setLineDash(Array.isArray(dash) ? dash : []);
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(Number(a.x), Number(a.y));
      ctx.lineTo(endX, endY);
      ctx.stroke();

      const nx = -uy;
      const ny = ux;
      const wing = Math.max(3.0, 0.45 * head);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(Number(b.x), Number(b.y));
      ctx.lineTo(endX + nx * wing, endY + ny * wing);
      ctx.lineTo(endX - nx * wing, endY - ny * wing);
      ctx.closePath();
      ctx.fill();

      ctx.globalAlpha = 1.0;
      ctx.setLineDash([]);
    }}

    function drawLabelTag(x, y, text, bgColor, strokeColor, textColor) {{
      const padX = 5;
      const padY = 3;
      ctx.font = '10px ui-monospace, monospace';
      const w = ctx.measureText(String(text)).width;
      const h = 13;
      const rx = Number(x) + 6;
      const ry = Number(y) - h - 6;
      ctx.fillStyle = bgColor;
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 1;
      ctx.fillRect(rx, ry, w + 2 * padX, h + 2 * padY);
      ctx.strokeRect(rx, ry, w + 2 * padX, h + 2 * padY);
      ctx.fillStyle = textColor;
      ctx.fillText(String(text), rx + padX, ry + h);
    }}

    function drawMarker(x, y, fillColor, strokeColor, radius) {{
      ctx.beginPath();
      ctx.arc(Number(x), Number(y), Number(radius), 0, Math.PI * 2.0);
      ctx.fillStyle = fillColor;
      ctx.fill();
      ctx.lineWidth = 1.4;
      ctx.strokeStyle = strokeColor;
      ctx.stroke();
    }}

    function getLaneByIndex(idx) {{
      return lanes.find(l => Number(l.index) === Number(idx)) || null;
    }}

    function getCarlaByIndex(idx) {{
      return carlaLines.find(c => Number(c.index) === Number(idx)) || null;
    }}

    function draw() {{
      ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);

      const filtered = filteredLanes();
      const visibleLaneIds = new Set(filtered.map(l => Number(l.index)));
      const selectedLane = state.selectedLaneIndex != null ? getLaneByIndex(state.selectedLaneIndex) : null;
      const selectedCarlaMain = (selectedLane && selectedLane.match) ? Number(selectedLane.match.carla_line_index) : null;
      const selectedCarlaExtra = (selectedLane && selectedLane.match && Array.isArray(selectedLane.match.split_extra_carla_lines))
        ? new Set(selectedLane.match.split_extra_carla_lines.map(Number))
        : new Set();
      const selectedCarlaSet = new Set();
      if (selectedCarlaMain != null && Number.isFinite(selectedCarlaMain) && selectedCarlaMain >= 0) {{
        selectedCarlaSet.add(Number(selectedCarlaMain));
      }}
      for (const ci of selectedCarlaExtra) selectedCarlaSet.add(Number(ci));

      const allowLane = (lane) => {{
        if (state.onlySelected) return selectedLane && Number(lane.index) === Number(selectedLane.index);
        return visibleLaneIds.has(Number(lane.index));
      }};

      const allowCarla = (ci) => {{
        if (!state.onlySelected) return true;
        return selectedCarlaSet.has(Number(ci));
      }};

      let unmatchedVisibleCount = 0;

      if (state.showCarla) {{
        for (const c of carlaLines) {{
          if (!allowCarla(c.index)) continue;
          const ci = Number(c.index);
          const isSel = selectedCarlaSet.has(ci);
          const matchedCount = Array.isArray(c.matched_v2_lane_indices) ? c.matched_v2_lane_indices.length : 0;
          const supplementalCount = Array.isArray(c.supplemental_v2_lane_indices) ? c.supplemental_v2_lane_indices.length : 0;
          const counterpartCount = Array.isArray(c.counterpart_v2_lane_indices) ? c.counterpart_v2_lane_indices.length : (matchedCount + supplementalCount);
          const onlySupplemental = (matchedCount <= 0 && supplementalCount > 0);
          const col = isSel
            ? sourceColor.carlaSelected
            : (matchedCount > 0 ? sourceColor.carlaMapped : (onlySupplemental ? sourceColor.carlaSupplemental : sourceColor.carlaUnmapped));
          const alpha = isSel ? 0.95 : (counterpartCount > 0 ? 0.64 : 0.24);
          const width = isSel ? 3.0 : (counterpartCount > 0 ? 1.8 : 1.1);
          drawPolyline(c.polyline, col, width, alpha, onlySupplemental ? [5, 3] : [7, 4]);
          if (state.showLabels && (isSel || counterpartCount > 0)) {{
            const s = worldToScreen(Number(c.label_x), Number(c.label_y));
            ctx.fillStyle = '#b4cad8';
            ctx.font = '10px ui-monospace, monospace';
            const tag = matchedCount > 0 ? 'primary' : (onlySupplemental ? 'supp' : '');
            ctx.fillText('CARLA c' + String(c.index) + (tag ? (' (' + tag + ')') : ''), s.x + 4, s.y - 4);
          }}
        }}
      }}

      if (state.showCarlaMatchedLayer) {{
        for (const c of carlaLines) {{
          if (!allowCarla(c.index)) continue;
          const matchedCount = Array.isArray(c.matched_v2_lane_indices) ? c.matched_v2_lane_indices.length : 0;
          const supplementalCount = Array.isArray(c.supplemental_v2_lane_indices) ? c.supplemental_v2_lane_indices.length : 0;
          const counterpartCount = Array.isArray(c.counterpart_v2_lane_indices) ? c.counterpart_v2_lane_indices.length : (matchedCount + supplementalCount);
          if (counterpartCount <= 0) continue;
          const onlySupplemental = (matchedCount <= 0 && supplementalCount > 0);
          const isSel = selectedCarlaSet.has(Number(c.index));
          const col = isSel ? '#9ef4ff' : (onlySupplemental ? sourceColor.carlaSupplemental : sourceColor.carlaMatchedOnly);
          const width = isSel ? 3.8 : 3.0;
          const alpha = isSel ? 0.98 : 0.9;
          drawPolyline(c.polyline, col, width, alpha, onlySupplemental ? [5, 3] : []);
          if (state.showLabels || isSel) {{
            const s = worldToScreen(Number(c.label_x), Number(c.label_y));
            ctx.fillStyle = '#d9fbff';
            ctx.font = '10px ui-monospace, monospace';
            const tag = onlySupplemental ? 'counterpart:supp' : 'counterpart:primary';
            ctx.fillText(`CARLA c${{c.index}} (${{tag}})`, s.x + 4, s.y - 4);
          }}
        }}
      }}

      if (state.showV2) {{
        for (const lane of lanes) {{
          if (!allowLane(lane)) continue;
          const q = laneQuality(lane);
          const isSel = selectedLane && Number(selectedLane.index) === Number(lane.index);
          const col = isSel ? sourceColor.v2Selected : (qualityColor[q] || '#95a5a6');
          const width = isSel ? 3.4 : 2.35;
          const alpha = isSel ? 1.0 : 0.88;
          drawPolyline(lane.polyline, col, width, alpha, []);
          if (state.showLabels || isSel) {{
            const s = worldToScreen(Number(lane.label_x), Number(lane.label_y));
            ctx.fillStyle = '#eef6fb';
            ctx.font = '10px ui-monospace, monospace';
            ctx.fillText(`V2 r${{lane.road_id}} l${{lane.lane_id}} t${{lane.lane_type}}`, s.x + 5, s.y - 5);
          }}
        }}
      }}

      if (state.showConnectors) {{
        for (const lane of lanes) {{
          if (!allowLane(lane)) continue;
          if (!lane.match) continue;
          const ci = Number(lane.match.carla_line_index);
          if (!allowCarla(ci)) continue;
          const carla = getCarlaByIndex(ci);
          if (!carla) continue;
          const a = worldToScreen(Number(lane.label_x), Number(lane.label_y));
          const b = worldToScreen(Number(carla.label_x), Number(carla.label_y));
          const isSel = selectedLane && Number(selectedLane.index) === Number(lane.index);
          const col = isSel ? sourceColor.pairSelected : sourceColor.pair;
          const width = isSel ? 2.2 : 1.1;
          const alpha = isSel ? 0.95 : 0.42;
          drawArrow(a, b, col, width, alpha, [4, 4]);
          if (state.showPairTags && (!state.onlySelected || isSel)) {{
            const mx = (Number(a.x) + Number(b.x)) * 0.5;
            const my = (Number(a.y) + Number(b.y)) * 0.5;
            drawLabelTag(
              mx,
              my,
              `v${{lane.index}} -> c${{ci}}`,
              sourceColor.labelBg,
              sourceColor.labelStroke,
              sourceColor.labelText
            );
          }}
        }}
      }}

      if (state.showUnmatchedCarla) {{
        for (const c of carlaLines) {{
          if (!allowCarla(c.index)) continue;
          if (!Boolean(c.is_unmatched)) continue;
          if (state.onlyLikelyTrueMiss && !Boolean(c.likely_true_miss)) continue;
          const bucket = String(c.unmatched_bucket || 'out_bbox');
          if (!state.unmatchedBucketFilter.has(bucket)) continue;
          const likely = Boolean(c.likely_true_miss);
          const col = likely ? '#ff2cc9' : (unmatchedBucketColor[bucket] || '#7f8c8d');
          const width = likely ? 3.7 : 2.6;
          const alpha = likely ? 0.98 : 0.86;
          drawPolyline(c.polyline, col, width, alpha, likely ? [] : [4, 2]);
          const s = worldToScreen(Number(c.label_x), Number(c.label_y));
          drawMarker(s.x, s.y, col, '#1d2731', likely ? 3.6 : 2.8);
          if (state.showLabels || likely) {{
            const d = c.dist_to_type1_v2_min_m;
            const dTxt = Number.isFinite(Number(d)) ? Number(d).toFixed(2) : '-';
            drawLabelTag(
              s.x,
              s.y,
              `c${{c.index}} ${{bucket}} d=${{dTxt}}m`,
              sourceColor.labelBg,
              sourceColor.labelStroke,
              sourceColor.labelText
            );
          }}
          unmatchedVisibleCount += 1;
        }}
      }}

      if (selectedLane) {{
        const a = worldToScreen(Number(selectedLane.label_x), Number(selectedLane.label_y));
        drawMarker(a.x, a.y, '#ffd79b', '#83572e', 4.4);
        drawLabelTag(
          a.x,
          a.y,
          `V2 ${{selectedLane.label}}`,
          sourceColor.labelBg,
          sourceColor.labelStroke,
          sourceColor.labelText
        );
        if (selectedCarlaMain != null && Number.isFinite(selectedCarlaMain) && selectedCarlaMain >= 0) {{
          const main = getCarlaByIndex(selectedCarlaMain);
          if (main) {{
            const b = worldToScreen(Number(main.label_x), Number(main.label_y));
            drawMarker(b.x, b.y, '#91e4ff', '#1e4f63', 4.4);
            drawArrow(a, b, sourceColor.pairSelected, 2.4, 0.95, []);
            drawLabelTag(
              b.x,
              b.y,
              `CARLA c${{selectedCarlaMain}}`,
              sourceColor.labelBg,
              sourceColor.labelStroke,
              sourceColor.labelText
            );
          }}
        }}
      }}

      const visibleCount = filtered.length;
      const matchedCarlaCount = carlaLines.reduce((acc, c) => {{
        const n = Array.isArray(c.matched_v2_lane_indices) ? c.matched_v2_lane_indices.length : 0;
        return acc + (n > 0 ? 1 : 0);
      }}, 0);
      const counterpartCarlaCount = carlaLines.reduce((acc, c) => {{
        const n = Array.isArray(c.counterpart_v2_lane_indices) ? c.counterpart_v2_lane_indices.length : 0;
        return acc + (n > 0 ? 1 : 0);
      }}, 0);
      hudSummary.textContent =
        `V2 lanes shown: ${{visibleCount}}/${{lanes.length}} | CARLA lines: ${{carlaLines.length}} | primary: ${{matchedCarlaCount}} | counterpart: ${{counterpartCarlaCount}}`;
      hudFilter.textContent =
        `Q filter: ${{[...state.filterQuality].join(', ') || '-'}} | search: ${{state.search || '-'}}` +
        ` | unmatched overlay: ${{state.showUnmatchedCarla ? ('on (' + unmatchedVisibleCount + ')') : 'off'}}`;
      if (selectedLane) {{
        const m = selectedLane.match;
        hudSelected.textContent = m
          ? `Selected pair: V2 ${{selectedLane.label}} -> CARLA c${{m.carla_line_index}} (${{selectedLane.quality}})`
          : `Selected: ${{selectedLane.label}} -> unmatched`;
      }} else {{
        hudSelected.textContent = 'Selected: -';
      }}
    }}

    function updateSummaryPanel() {{
      const q = summary.quality_counts || {{}};
      const metricTxt = [
        `median(med_dist)=${{Number(metricSummary.median_of_median_dist_m).toFixed(3)}}m`,
        `mean(cov2)=${{Number(metricSummary.mean_of_coverage_2m).toFixed(3)}}`,
        `mean(mono)=${{Number(metricSummary.mean_of_monotonic_ratio).toFixed(3)}}`,
      ].join(' | ');
      const legalTot = Number(summary.legal_driving_edge_total || 0);
      const legalOk = Number(summary.legal_driving_edge_ok || 0);
      const legalBad = Number(summary.legal_driving_edge_bad || 0);
      const legalRatio = (legalTot > 0) ? ((100.0 * legalOk / legalTot).toFixed(1) + '%') : '-';
      const pp = summary.postprocess || {{}};
      const unmatchedCarla = Number(summary.unmatched_carla_line_count || 0);
      const likelyMiss = Number(summary.likely_true_miss_carla_count || 0);
      const unmatchedBuckets = summary.unmatched_carla_bucket_counts || {{}};
      const counterpartCarla = Number(summary.counterpart_carla_line_count || 0);
      const supplementalCounterpart = Number(summary.supplemental_counterpart_carla_count || 0);

      const icp = align.icp_refine || {{}};
      const icpTxt = icp.applied
        ? `ICP: iters=${{icp.iterations}} inliers=${{icp.inliers}} ` +
          `median ${{Number(icp.before_median_m).toFixed(2)}} -> ${{Number(icp.after_median_m).toFixed(2)}} m ` +
          `(dtheta=${{Number(icp.delta_theta_deg).toFixed(3)}}deg, dtx=${{Number(icp.delta_tx).toFixed(3)}}, dty=${{Number(icp.delta_ty).toFixed(3)}})`
        : 'ICP: disabled';

      summaryText.textContent =
        `v2_lanes=${{summary.v2_lane_count}} mapped=${{summary.mapped_lane_count}} usable=${{summary.usable_lane_count}} ` +
        `poor=${{summary.poor_lane_count}} unmatched=${{summary.unmatched_lane_count}}\\n` +
        `driving_lanes=${{summary.driving_lane_count}} mapped_driving=${{summary.driving_mapped_count}}\\n` +
        `carla_lines=${{summary.carla_line_count}} primary_mapped=${{summary.mapped_carla_line_count}} ` +
        `counterpart_mapped=${{counterpartCarla}} (supp=${{supplementalCounterpart}})\\n` +
        `carla_unmatched_no_counterpart=${{unmatchedCarla}} likely_true_miss=${{likelyMiss}}\\n` +
        `legal_edges=${{legalOk}}/${{legalTot}} bad=${{legalBad}} ratio=${{legalRatio}}\\n` +
        `post: removed_non_driving=${{Number(pp.removed_non_driving || 0)}} ` +
        `added_fill=${{Number(pp.added_by_fill || 0)}} repaired=${{Number(pp.repaired_changes || 0)}} ` +
        `strict_dropped=${{Number(pp.dropped_for_legality || 0)}} strict_remapped=${{Number(pp.remapped_for_legality || 0)}}\\n` +
        `${{metricTxt}}\\n${{icpTxt}}`;

      qualityChips.innerHTML = '';
      for (const key of ['high','medium','low','poor','unmatched']) {{
        const val = Number(q[key] || 0);
        const div = document.createElement('span');
        div.className = 'chip';
        div.innerHTML = `<span class="dot" style="background:${{qualityColor[key] || '#95a5a6'}}"></span>${{key}}: ${{val}}`;
        qualityChips.appendChild(div);
      }}

      unmatchedChips.innerHTML = '';
      for (const key of ['near_1m', 'near_5m', 'near_20m', 'far_20m', 'out_bbox']) {{
        const val = Number(unmatchedBuckets[key] || 0);
        const div = document.createElement('span');
        div.className = 'chip';
        div.innerHTML = `<span class="dot" style="background:${{unmatchedBucketColor[key] || '#95a5a6'}}"></span>${{unmatchedBucketLabel[key] || key}}: ${{val}}`;
        unmatchedChips.appendChild(div);
      }}
    }}

    function rebuildLaneTable() {{
      const rows = filteredLanes().sort((a, b) => {{
        const qa = ['high','medium','low','poor','unmatched'].indexOf(String(a.quality || 'unmatched'));
        const qb = ['high','medium','low','poor','unmatched'].indexOf(String(b.quality || 'unmatched'));
        if (qa !== qb) return qa - qb;
        return Number(a.index) - Number(b.index);
      }});
      laneTableBody.innerHTML = '';
      for (const lane of rows) {{
        const tr = document.createElement('tr');
        if (state.selectedLaneIndex != null && Number(state.selectedLaneIndex) === Number(lane.index)) {{
          tr.className = 'sel';
        }}
        const m = lane.match;
        const ci = m ? ('c' + String(m.carla_line_index)) : '-';
        const med = m ? Number(m.median_dist_m).toFixed(2) : '-';
        const p90 = m ? Number(m.p90_dist_m).toFixed(2) : '-';
        const cov = m ? Number(m.coverage_2m).toFixed(2) : '-';
        tr.innerHTML =
          `<td>${{lane.label}}</td>` +
          `<td style="color:${{lane.quality_color || '#95a5a6'}}">${{lane.quality}}</td>` +
          `<td>${{ci}}</td><td>${{med}}</td><td>${{p90}}</td><td>${{cov}}</td>`;
        tr.addEventListener('click', () => {{
          state.selectedLaneIndex = Number(lane.index);
          rebuildLaneTable();
          rebuildSelectedPanel();
          draw();
        }});
        laneTableBody.appendChild(tr);
      }}
    }}

    function rebuildSelectedPanel() {{
      const lane = (state.selectedLaneIndex != null) ? getLaneByIndex(state.selectedLaneIndex) : null;
      if (!lane) {{
        selectedInfo.textContent = 'Click a lane to inspect match metrics.';
        candidateBody.innerHTML = '';
        return;
      }}
      if (!lane.match) {{
        selectedInfo.textContent = `V2 ${{lane.label}}\\nmatch=unmatched`;
      }} else {{
        const m = lane.match;
        selectedInfo.textContent =
          `V2 ${{lane.label}}\\n` +
          `match=CARLA c${{m.carla_line_index}} quality=${{lane.quality}} reversed=${{Boolean(m.reversed)}} shared=${{Boolean(m.shared_carla_line)}}\\n` +
          `score=${{Number(m.score).toFixed(3)}} med=${{Number(m.median_dist_m).toFixed(3)}}m p90=${{Number(m.p90_dist_m).toFixed(3)}}m\\n` +
          `cov2=${{Number(m.coverage_2m).toFixed(3)}} mono=${{Number(m.monotonic_ratio).toFixed(3)}} ` +
          `ang=${{Number(m.angle_median_deg).toFixed(2)}}deg len_ratio=${{Number(m.length_ratio).toFixed(3)}}\\n` +
          `split_extra=[${{(m.split_extra_carla_lines || []).join(', ')}}]`;
      }}
      const cands = laneCandidates[String(lane.index)] || [];
      candidateBody.innerHTML = '';
      for (const c of cands) {{
        const tr = document.createElement('tr');
        tr.innerHTML =
          `<td>c${{Number(c.carla_line_index)}}</td>` +
          `<td style="color:${{qualityColor[String(c.quality)] || '#95a5a6'}}">${{c.quality}}</td>` +
          `<td>${{Number(c.score).toFixed(3)}}</td>` +
          `<td>${{Number(c.median_dist_m).toFixed(2)}}</td>` +
          `<td>${{Number(c.p90_dist_m).toFixed(2)}}</td>` +
          `<td>${{Number(c.coverage_2m).toFixed(2)}}</td>` +
          `<td>${{Number(c.angle_median_deg).toFixed(1)}}</td>`;
        candidateBody.appendChild(tr);
      }}
    }}

    function pickLaneAtScreen(sx, sy) {{
      let best = null;
      let bestDist = 16.0;
      for (const lane of filteredLanes()) {{
        const poly = lane.polyline || [];
        for (const p of poly) {{
          if (!Array.isArray(p) || p.length < 2) continue;
          const s = worldToScreen(Number(p[0]), Number(p[1]));
          const d = Math.hypot(s.x - sx, s.y - sy);
          if (d < bestDist) {{
            bestDist = d;
            best = lane;
          }}
        }}
      }}
      return best;
    }}

    function bindUI() {{
      showV2.addEventListener('change', () => {{ state.showV2 = !!showV2.checked; draw(); }});
      showCarla.addEventListener('change', () => {{ state.showCarla = !!showCarla.checked; draw(); }});
      showCarlaMatchedLayer.addEventListener('change', () => {{ state.showCarlaMatchedLayer = !!showCarlaMatchedLayer.checked; draw(); }});
      showConnectors.addEventListener('change', () => {{ state.showConnectors = !!showConnectors.checked; draw(); }});
      showPairTags.addEventListener('change', () => {{ state.showPairTags = !!showPairTags.checked; draw(); }});
      showUnmatchedCarla.addEventListener('change', () => {{ state.showUnmatchedCarla = !!showUnmatchedCarla.checked; draw(); }});
      onlyLikelyTrueMiss.addEventListener('change', () => {{ state.onlyLikelyTrueMiss = !!onlyLikelyTrueMiss.checked; draw(); }});
      showLabels.addEventListener('change', () => {{ state.showLabels = !!showLabels.checked; draw(); }});
      onlySelected.addEventListener('change', () => {{ state.onlySelected = !!onlySelected.checked; draw(); }});
      searchBox.addEventListener('input', () => {{
        state.search = String(searchBox.value || '').trim();
        rebuildLaneTable();
        draw();
      }});
      for (const qc of qChecks) {{
        qc.addEventListener('change', () => {{
          const val = String(qc.value || '');
          if (qc.checked) state.filterQuality.add(val);
          else state.filterQuality.delete(val);
          rebuildLaneTable();
          draw();
        }});
      }}
      for (const uc of unmatchedChecks) {{
        uc.addEventListener('change', () => {{
          const val = String(uc.value || '');
          if (uc.checked) state.unmatchedBucketFilter.add(val);
          else state.unmatchedBucketFilter.delete(val);
          draw();
        }});
      }}

      canvas.addEventListener('wheel', (e) => {{
        e.preventDefault();
        const factor = e.deltaY < 0 ? 1.15 : 0.87;
        state.view.scale = Math.max(0.03, Math.min(2000, state.view.scale * factor));
        draw();
      }}, {{ passive: false }});

      canvas.addEventListener('mousedown', (e) => {{
        state.drag = {{ x: e.clientX, y: e.clientY, cx: state.view.cx, cy: state.view.cy }};
      }});
      canvas.addEventListener('mousemove', (e) => {{
        if (!state.drag) return;
        const dx = e.clientX - state.drag.x;
        const dy = e.clientY - state.drag.y;
        state.view.cx = state.drag.cx - dx / state.view.scale;
        state.view.cy = state.drag.cy + dy / state.view.scale;
        draw();
      }});
      const clearDrag = () => {{ state.drag = null; }};
      canvas.addEventListener('mouseup', clearDrag);
      canvas.addEventListener('mouseleave', clearDrag);

      canvas.addEventListener('click', (e) => {{
        const rect = canvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;
        const lane = pickLaneAtScreen(sx, sy);
        if (!lane) return;
        state.selectedLaneIndex = Number(lane.index);
        rebuildLaneTable();
        rebuildSelectedPanel();
        draw();
      }});

      window.addEventListener('resize', () => {{
        resizeCanvas();
        fitView();
        draw();
      }});
    }}

    function init() {{
      resizeCanvas();
      fitView();
      updateSummaryPanel();
      rebuildLaneTable();
      rebuildSelectedPanel();
      bindUI();
      draw();
    }}

    init();
  }})();
  </script>
</body>
</html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build V2XPNP<->CARLA lane correspondence diagnostics HTML.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--v2-map",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Path(s) to V2XPNP vector map PKL. "
            "When omitted, runs both default maps: corridors + intersection."
        ),
    )
    parser.add_argument(
        "--carla-map-cache",
        type=str,
        default="/data2/marco/CoLMDriver/v2xpnp/map/carla_map_cache.pkl",
        help="Path to CARLA map cache pickle with line polylines.",
    )
    parser.add_argument(
        "--carla-align",
        type=str,
        default="/data2/marco/CoLMDriver/v2xpnp/map/ucla_map_offset_carla.json",
        help="Path to alignment config JSON for CARLA map lines.",
    )
    parser.add_argument(
        "--driving-types",
        type=str,
        default="1",
        help="Comma/space-separated V2 lane types treated as driving lanes in one-to-one assignment.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=56,
        help="Candidate CARLA lines per V2 lane for correspondence scoring.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="v2xpnp/scripts/lane_corr_cache",
        help="Cache directory for correspondence internals.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="v2xpnp/map/lane_correspondence_diagnostics.html",
        help="Output HTML path.",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="",
        help="Optional output JSON report path.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=8,
        help="How many top candidates to include per lane in diagnostics output.",
    )
    parser.add_argument(
        "--candidate-refresh-top-k",
        type=int,
        default=96,
        help="If cached correspondence omits candidate rows, recompute with this top-k (no cache) for diagnostics/postprocess.",
    )
    parser.add_argument(
        "--keep-non-driving-matches",
        action="store_true",
        help="Keep final mappings for non-driving lane types. Default behavior prunes them from final output.",
    )
    parser.add_argument(
        "--fill-max-candidates",
        type=int,
        default=16,
        help="Max candidate rows examined per lane during unmatched-driving fill and legality repair.",
    )
    parser.add_argument(
        "--legality-repair-passes",
        type=int,
        default=3,
        help="Local repair passes that optimize mapped driving lanes for legal successor connectivity.",
    )
    parser.add_argument(
        "--strict-legal-routes",
        action="store_true",
        help="Enforce legal successor routes by remapping or dropping violating driving-lane assignments.",
    )
    parser.add_argument(
        "--no-icp-refine",
        action="store_true",
        help="Disable robust ICP alignment refinement before lane correspondence.",
    )
    parser.add_argument(
        "--icp-max-iters",
        type=int,
        default=10,
        help="Maximum ICP refinement iterations.",
    )
    parser.add_argument(
        "--icp-inlier-quantile",
        type=float,
        default=0.35,
        help="Inlier quantile used by ICP each iteration.",
    )
    parser.add_argument(
        "--icp-inlier-max-m",
        type=float,
        default=15.0,
        help="Max inlier distance gate for ICP (meters).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable extra logs.",
    )
    return parser.parse_args()


def _parse_type_set(raw: str) -> List[str]:
    if raw is None:
        return ["1"]
    toks: List[str] = []
    for chunk in str(raw).replace(",", " ").split():
        t = str(chunk).strip()
        if t:
            toks.append(t)
    return sorted(set(toks)) if toks else ["1"]


def _run_single_map(
    args: argparse.Namespace,
    v2_map_path: Path,
    carla_cache_path: Path,
    align_path: Optional[Path],
    out_html: Path,
    out_json: Optional[Path],
    cache_dir: Optional[Path],
    driving_types: Sequence[str],
) -> Dict[str, object]:
    if args.verbose:
        print(f"[INFO] Loading V2 map: {v2_map_path}")
    v2_map = load_vector_map(v2_map_path)

    if args.verbose:
        print(f"[INFO] Loading CARLA cache lines: {carla_cache_path}")
    raw_lines, _, carla_name = ytm._load_carla_map_cache_lines(carla_cache_path)
    if not raw_lines:
        raise SystemExit("CARLA cache has no valid line polylines.")

    align_cfg = ytm._load_carla_alignment_cfg(align_path)
    transformed_lines, _ = ytm._transform_carla_lines(raw_lines, align_cfg)
    if not transformed_lines:
        raise SystemExit("No CARLA lines remained after alignment transform.")

    v2_vertices = np.asarray(
        [
            (float(p[0]), float(p[1]))
            for lane in v2_map.lanes
            for p in lane.polyline[:, :2]
        ],
        dtype=np.float64,
    )
    if v2_vertices.shape[0] <= 0:
        raise SystemExit("V2 map has no lane vertices.")

    refined_lines, icp_stats = _refine_alignment_icp(
        transformed_lines=transformed_lines,
        v2_vertices_xy=v2_vertices,
        enabled=not bool(args.no_icp_refine),
        max_iters=int(args.icp_max_iters),
        inlier_quantile=float(args.icp_inlier_quantile),
        inlier_max_m=float(args.icp_inlier_max_m),
    )

    if args.verbose:
        if icp_stats.applied:
            print(
                "[INFO] ICP refine: "
                f"iters={icp_stats.iterations}, inliers={icp_stats.inliers}, "
                f"median {icp_stats.before_median:.3f} -> {icp_stats.after_median:.3f} m, "
                f"dtheta={icp_stats.delta_theta_deg:.4f} deg, "
                f"dtx={icp_stats.delta_tx:.4f}, dty={icp_stats.delta_ty:.4f}"
            )
        else:
            print("[INFO] ICP refine disabled.")

    payload = _build_payload_from_maps(
        v2_map=v2_map,
        carla_lines_xy=refined_lines,
        carla_source=str(carla_cache_path),
        carla_name=str(carla_name),
    )

    if args.verbose:
        print(
            f"[INFO] Running correspondence: v2_lanes={len(payload['map']['lanes'])}, "
            f"carla_lines={len(payload['carla_map']['lines'])}, driving_types={driving_types}, top_k={int(args.top_k)}"
        )
    corr = ytm._build_lane_correspondence(
        payload=payload,
        candidate_top_k=int(args.top_k),
        driving_lane_types=driving_types,
        cache_dir=cache_dir,
    )
    if not bool(corr.get("enabled", False)):
        raise SystemExit(f"Lane correspondence failed: {corr.get('reason', 'unknown')}")

    lane_candidates_raw = corr.get("lane_candidates", {})
    has_candidates = bool(isinstance(lane_candidates_raw, dict) and len(lane_candidates_raw) > 0)
    if not has_candidates:
        refresh_top_k = max(int(args.top_k), int(args.candidate_refresh_top_k))
        if args.verbose:
            print(
                "[INFO] Candidate rows missing in correspondence payload; "
                f"recomputing without cache (top_k={refresh_top_k})..."
            )
        corr = ytm._build_lane_correspondence(
            payload=payload,
            candidate_top_k=int(refresh_top_k),
            driving_lane_types=driving_types,
            cache_dir=None,
        )
        if not bool(corr.get("enabled", False)):
            raise SystemExit(
                f"Lane correspondence refresh failed: {corr.get('reason', 'unknown')}"
            )

    corr = _postprocess_correspondence(
        corr=corr,
        driving_types=set(driving_types),
        keep_non_driving_matches=bool(args.keep_non_driving_matches),
        fill_max_candidates=int(args.fill_max_candidates),
        legality_repair_passes=int(args.legality_repair_passes),
        strict_legal_routes=bool(args.strict_legal_routes),
        verbose=bool(args.verbose),
    )

    report = _build_report_dataset(
        payload=payload,
        corr=corr,
        icp_stats=icp_stats,
        align_cfg=align_cfg,
        top_candidates=int(args.max_candidates),
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(_build_html(report), encoding="utf-8")
    print(f"[OK] Wrote diagnostics HTML: {out_html}")

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"[OK] Wrote diagnostics JSON: {out_json}")

    summary = report.get("summary", {})
    legal_ok = int(_safe_float(summary.get("legal_driving_edge_ok"), 0.0))
    legal_total = int(_safe_float(summary.get("legal_driving_edge_total"), 0.0))
    legal_bad = int(_safe_float(summary.get("legal_driving_edge_bad"), 0.0))
    pp = summary.get("postprocess", {}) if isinstance(summary.get("postprocess", {}), dict) else {}
    print(
        "[INFO] Summary: "
        f"map={v2_map_path.stem} "
        f"v2={summary.get('v2_lane_count', 0)} mapped={summary.get('mapped_lane_count', 0)} "
        f"usable={summary.get('usable_lane_count', 0)} poor={summary.get('poor_lane_count', 0)} "
        f"unmatched={summary.get('unmatched_lane_count', 0)} "
        f"carla={summary.get('carla_line_count', 0)} "
        f"mapped_carla_primary={summary.get('mapped_carla_line_count', 0)} "
        f"mapped_carla_counterpart={summary.get('counterpart_carla_line_count', 0)} "
        f"supp_counterpart={summary.get('supplemental_counterpart_carla_count', 0)} "
        f"driving={summary.get('driving_mapped_count', 0)}/{summary.get('driving_lane_count', 0)} "
        f"legal_edges={legal_ok}/{legal_total} bad={legal_bad} "
        f"post(removed_non_driving={int(_safe_float(pp.get('removed_non_driving'), 0.0))}, "
        f"added_fill={int(_safe_float(pp.get('added_by_fill'), 0.0))}, "
        f"repaired={int(_safe_float(pp.get('repaired_changes'), 0.0))})"
    )
    return report


def main() -> None:
    args = parse_args()
    requested_maps = [str(v).strip() for v in (args.v2_map or []) if str(v).strip()]
    if requested_maps:
        v2_map_paths = [Path(v).expanduser().resolve() for v in requested_maps]
    else:
        v2_map_paths = [
            Path("/data2/marco/CoLMDriver/v2xpnp/map/v2v_corridors_vector_map.pkl").resolve(),
            Path("/data2/marco/CoLMDriver/v2xpnp/map/v2x_intersection_vector_map.pkl").resolve(),
        ]
    # Preserve order but de-dup.
    seen_paths: set = set()
    map_paths: List[Path] = []
    for p in v2_map_paths:
        ps = str(p)
        if ps in seen_paths:
            continue
        seen_paths.add(ps)
        map_paths.append(p)

    carla_cache_path = Path(args.carla_map_cache).expanduser().resolve()
    align_path = Path(args.carla_align).expanduser().resolve() if str(args.carla_align).strip() else None
    out_html_base = Path(args.out).expanduser().resolve()
    out_json_base = Path(args.out_json).expanduser().resolve() if str(args.out_json).strip() else None
    cache_dir = Path(args.cache_dir).expanduser().resolve() if str(args.cache_dir).strip() else None
    driving_types = _parse_type_set(args.driving_types)

    for p in map_paths:
        if not p.exists():
            raise SystemExit(f"V2 map not found: {p}")
    if not carla_cache_path.exists():
        raise SystemExit(f"CARLA cache not found: {carla_cache_path}")

    multi_map = len(map_paths) > 1
    reports: List[Tuple[str, Path, Dict[str, object]]] = []
    for v2_map_path in map_paths:
        if multi_map:
            out_html = out_html_base.with_name(f"{out_html_base.stem}_{v2_map_path.stem}{out_html_base.suffix or '.html'}")
            out_json = (
                out_json_base.with_name(f"{out_json_base.stem}_{v2_map_path.stem}{out_json_base.suffix or '.json'}")
                if out_json_base is not None
                else None
            )
        else:
            out_html = out_html_base
            out_json = out_json_base

        report = _run_single_map(
            args=args,
            v2_map_path=v2_map_path,
            carla_cache_path=carla_cache_path,
            align_path=align_path,
            out_html=out_html,
            out_json=out_json,
            cache_dir=cache_dir,
            driving_types=driving_types,
        )
        reports.append((str(v2_map_path.stem), out_html, report))

    if len(reports) > 1:
        index_path = out_html_base.with_name(f"{out_html_base.stem}_index.html")
        rows = []
        for map_name, html_path, report in reports:
            summary = report.get("summary", {}) if isinstance(report, dict) else {}
            rows.append(
                "<tr>"
                f"<td>{map_name}</td>"
                f"<td>{int(_safe_float(summary.get('mapped_lane_count', 0)))} / {int(_safe_float(summary.get('v2_lane_count', 0)))}</td>"
                f"<td>{int(_safe_float(summary.get('mapped_carla_line_count', 0)))} / {int(_safe_float(summary.get('carla_line_count', 0)))}</td>"
                f"<td><a href=\"{html_path.name}\">{html_path.name}</a></td>"
                "</tr>"
            )
        index_html = (
            "<!doctype html><html><head><meta charset=\"utf-8\"><title>Lane Correspondence Multi-Map</title>"
            "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#0f1720;color:#e5eef5}"
            "table{border-collapse:collapse;width:100%;max-width:1200px}th,td{border:1px solid #365066;padding:8px 10px}"
            "th{background:#1b2b3a}a{color:#6bd4ff;text-decoration:none}a:hover{text-decoration:underline}</style>"
            "</head><body><h2>Lane Correspondence Reports (Multi-Map)</h2>"
            "<table><thead><tr><th>V2 Map</th><th>Mapped V2 Lanes</th><th>Primary Mapped CARLA Lines</th><th>HTML Report</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></body></html>"
        )
        index_path.write_text(index_html, encoding="utf-8")
        print(f"[OK] Wrote multi-map index: {index_path}")


if __name__ == "__main__":
    main()
