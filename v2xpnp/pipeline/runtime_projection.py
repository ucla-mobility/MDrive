"""Stage internals: CARLA projection and projection-time smoothing."""

from __future__ import annotations

import time

from v2xpnp.pipeline import runtime_common as _s1

for _name, _value in vars(_s1).items():
    if _name.startswith("__"):
        continue
    globals()[_name] = _value

def _normalize_yaw_deg(yaw: float) -> float:
    y = float(yaw)
    while y > 180.0:
        y -= 360.0
    while y <= -180.0:
        y += 360.0
    return y


def _yaw_abs_diff_deg(a: float, b: float) -> float:
    return abs(_normalize_yaw_deg(float(a) - float(b)))


def _interp_yaw_deg(a: float, b: float, alpha: float) -> float:
    aa = _normalize_yaw_deg(float(a))
    bb = _normalize_yaw_deg(float(b))
    delta = _normalize_yaw_deg(bb - aa)
    return _normalize_yaw_deg(aa + float(alpha) * delta)


def _polyline_cumlen_xy(poly_xy: np.ndarray) -> np.ndarray:
    n = int(poly_xy.shape[0])
    if n <= 0:
        return np.zeros((0,), dtype=np.float64)
    if n == 1:
        return np.zeros((1,), dtype=np.float64)
    seg = poly_xy[1:] - poly_xy[:-1]
    seg_len = np.sqrt(np.sum(seg * seg, axis=1))
    out = np.zeros((n,), dtype=np.float64)
    out[1:] = np.cumsum(seg_len)
    return out


def _project_point_to_polyline_xy(
    poly_xy: np.ndarray,
    cumlen: np.ndarray,
    qx: float,
    qy: float,
) -> Optional[Dict[str, float]]:
    n = int(poly_xy.shape[0])
    if n <= 0:
        return None
    if n == 1:
        px = float(poly_xy[0, 0])
        py = float(poly_xy[0, 1])
        return {
            "x": px,
            "y": py,
            "dist": float(math.hypot(px - float(qx), py - float(qy))),
            "yaw": 0.0,
            "s": 0.0,
            "s_norm": 0.0,
        }

    p0 = poly_xy[:-1, :]
    p1 = poly_xy[1:, :]
    seg = p1 - p0
    seg_len2 = np.sum(seg * seg, axis=1)
    valid = seg_len2 > 1e-12
    if not np.any(valid):
        px = float(poly_xy[0, 0])
        py = float(poly_xy[0, 1])
        return {
            "x": px,
            "y": py,
            "dist": float(math.hypot(px - float(qx), py - float(qy))),
            "yaw": 0.0,
            "s": 0.0,
            "s_norm": 0.0,
        }

    q = np.asarray([float(qx), float(qy)], dtype=np.float64)
    t = np.zeros((seg.shape[0],), dtype=np.float64)
    t[valid] = np.sum((q[None, :] - p0[valid]) * seg[valid], axis=1) / seg_len2[valid]
    t = np.clip(t, 0.0, 1.0)
    proj = p0 + seg * t[:, None]
    diff = proj - q[None, :]
    dist2 = np.sum(diff * diff, axis=1)
    best = int(np.argmin(dist2))

    seg_dx = float(seg[best, 0])
    seg_dy = float(seg[best, 1])
    yaw = _normalize_yaw_deg(math.degrees(math.atan2(seg_dy, seg_dx))) if (abs(seg_dx) + abs(seg_dy)) > 1e-9 else 0.0
    seg_len = math.sqrt(max(1e-12, float(seg_len2[best])))
    s0 = float(cumlen[best]) if best < int(cumlen.size) else 0.0
    s = float(s0 + float(t[best]) * seg_len)
    total = float(cumlen[-1]) if cumlen.size > 0 else 0.0
    s_norm = float(s / total) if total > 1e-9 else 0.0
    return {
        "x": float(proj[best, 0]),
        "y": float(proj[best, 1]),
        "dist": float(math.sqrt(max(0.0, float(dist2[best])))),
        "yaw": float(yaw),
        "s": float(s),
        "s_norm": float(max(0.0, min(1.0, s_norm))),
    }


def _build_lane_corr_payload(
    v2_map: VectorMapData,
    carla_lines_xy: Sequence[Sequence[Tuple[float, float]]],
    carla_line_records: Optional[Sequence[Dict[str, object]]],
    carla_source_path: str,
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
                "polyline": [[float(p[0]), float(p[1])] for p in poly],
                "label_x": float(mid[0]),
                "label_y": float(mid[1]),
                "entry_lanes": list(lane.entry_lanes or []),
                "exit_lanes": list(lane.exit_lanes or []),
            }
        )

    carla_lines_payload: List[Dict[str, object]] = []
    for i, ln in enumerate(carla_lines_xy):
        if len(ln) < 2:
            continue
        row: Dict[str, object] = {
            "index": int(i),
            "polyline": [[float(p[0]), float(p[1])] for p in ln],
        }
        if isinstance(carla_line_records, (list, tuple)) and i < len(carla_line_records):
            rec = carla_line_records[i]
            if isinstance(rec, dict):
                road_id = rec.get("road_id")
                lane_id = rec.get("lane_id")
                dir_sign = rec.get("dir_sign")
                try:
                    road_id = int(road_id) if road_id is not None else None
                except Exception:
                    road_id = None
                try:
                    lane_id = int(lane_id) if lane_id is not None else None
                except Exception:
                    lane_id = None
                try:
                    dir_sign = int(dir_sign) if dir_sign is not None else None
                except Exception:
                    dir_sign = None
                if dir_sign not in (-1, 1):
                    dir_sign = None
                row["road_id"] = road_id
                row["lane_id"] = lane_id
                row["dir_sign"] = dir_sign
        carla_lines_payload.append(row)

    return {
        "map": {
            "name": str(v2_map.name),
            "source_path": str(v2_map.source_path),
            "lanes": lanes_payload,
        },
        "carla_map": {
            "name": str(carla_name or "carla_map_cache"),
            "source_path": str(carla_source_path),
            "lines": carla_lines_payload,
        },
    }


def _build_carla_feats_from_lines(
    carla_lines_xy: Sequence[Sequence[Tuple[float, float]]],
) -> Dict[int, Dict[str, object]]:
    out: Dict[int, Dict[str, object]] = {}
    for i, ln in enumerate(carla_lines_xy):
        if len(ln) < 2:
            continue
        poly_xy = np.asarray([[float(p[0]), float(p[1])] for p in ln], dtype=np.float64)
        if poly_xy.shape[0] < 2:
            continue
        cum = _polyline_cumlen_xy(poly_xy)
        total = float(cum[-1]) if cum.size > 0 else 0.0
        if total <= 1e-6:
            continue
        poly_rev = poly_xy[::-1].copy()
        cum_rev = _polyline_cumlen_xy(poly_rev)
        mid = poly_xy[poly_xy.shape[0] // 2]
        out[int(i)] = {
            "line_index": int(i),
            "poly_xy": poly_xy,
            "cum": cum,
            "poly_xy_rev": poly_rev,
            "cum_rev": cum_rev,
            "label_x": float(mid[0]),
            "label_y": float(mid[1]),
            "total_len": total,
        }
    return out


def _resolve_carla_orientation_signs(
    line_indices: Sequence[int],
    rev_votes: Dict[int, float],
    fwd_votes: Dict[int, float],
    margin_ratio: float,
    min_vote: float,
) -> Dict[int, int]:
    sign_by_line: Dict[int, int] = {}
    for ci in line_indices:
        rv = float(rev_votes.get(int(ci), 0.0))
        fv = float(fwd_votes.get(int(ci), 0.0))
        sign = 0
        if rv >= float(min_vote) and rv > float(margin_ratio) * fv:
            sign = -1
        elif fv >= float(min_vote) and fv > float(margin_ratio) * rv:
            sign = 1
        sign_by_line[int(ci)] = int(sign)
    return sign_by_line


def _apply_orientation_to_lines_data(
    lines_data: List[Dict[str, object]],
    sign_by_line: Dict[int, int],
    rev_votes: Dict[int, float],
    fwd_votes: Dict[int, float],
) -> None:
    if not isinstance(lines_data, list):
        return
    for row in lines_data:
        if not isinstance(row, dict):
            continue
        ci = _safe_int(row.get("index"), -1)
        if ci < 0:
            continue
        sign = _safe_int(sign_by_line.get(int(ci), 0), 0)
        row["orientation_hint"] = (
            "reversed" if int(sign) < 0 else ("forward" if int(sign) > 0 else "unknown")
        )
        row["orientation_rev_vote"] = float(rev_votes.get(int(ci), 0.0))
        row["orientation_fwd_vote"] = float(fwd_votes.get(int(ci), 0.0))


def _normalize_int_key_dict(raw_map: object) -> Dict[int, object]:
    out: Dict[int, object] = {}
    if not isinstance(raw_map, dict):
        return out
    for k, v in raw_map.items():
        ki = _safe_int(k, -1)
        if ki >= 0:
            out[int(ki)] = v
    return out


def _normalize_int_set_map(raw_map: object) -> Dict[int, set]:
    out: Dict[int, set] = {}
    if not isinstance(raw_map, dict):
        return out
    for k, vals in raw_map.items():
        ki = _safe_int(k, -1)
        if ki < 0:
            continue
        bucket: set = set()
        if isinstance(vals, (list, tuple, set)):
            for v in vals:
                vi = _safe_int(v, -1)
                if vi >= 0:
                    bucket.add(int(vi))
        out[int(ki)] = bucket
    return out


def build_carla_projection_context(
    chosen_map: VectorMapData,
    carla_lines_xy: Sequence[Sequence[Tuple[float, float]]],
    carla_bbox: Tuple[float, float, float, float],
    carla_source_path: str,
    carla_name: str,
    lane_corr_top_k: int,
    lane_corr_cache_dir: Optional[Path],
    lane_corr_driving_types: Sequence[str],
    carla_line_records: Optional[Sequence[Dict[str, object]]] = None,
    verbose: bool = False,
) -> Dict[str, object]:
    carla_lines_for_corr = [list(ln) for ln in carla_lines_xy]
    icp_meta: Dict[str, object] = {}
    if _env_int("V2X_LANE_CORR_ENABLE_ICP_REFINE", 1, minimum=0, maximum=1) == 1:
        lcd = _load_lane_corr_diag_module()
        if lcd is not None:
            try:
                v2_vertices = np.asarray(
                    [
                        (float(p[0]), float(p[1]))
                        for lane in chosen_map.lanes
                        for p in lane.polyline[:, :2]
                    ],
                    dtype=np.float64,
                )
                if v2_vertices.shape[0] > 0:
                    refined_lines, icp_stats = lcd._refine_alignment_icp(
                        transformed_lines=carla_lines_for_corr,
                        v2_vertices_xy=v2_vertices,
                        enabled=True,
                        max_iters=_env_int("V2X_LANE_CORR_ICP_MAX_ITERS", 10, minimum=1, maximum=30),
                        inlier_quantile=_env_float("V2X_LANE_CORR_ICP_INLIER_QUANTILE", 0.35),
                        inlier_max_m=_env_float("V2X_LANE_CORR_ICP_INLIER_MAX_M", 15.0),
                    )
                    if refined_lines:
                        carla_lines_for_corr = refined_lines
                    icp_meta = {
                        "applied": bool(getattr(icp_stats, "applied", False)),
                        "iterations": int(getattr(icp_stats, "iterations", 0)),
                        "inliers": int(getattr(icp_stats, "inliers", 0)),
                        "before_median_m": float(_safe_float(getattr(icp_stats, "before_median", float("nan")), float("nan"))),
                        "after_median_m": float(_safe_float(getattr(icp_stats, "after_median", float("nan")), float("nan"))),
                        "delta_theta_deg": float(_safe_float(getattr(icp_stats, "delta_theta_deg", 0.0), 0.0)),
                        "delta_tx": float(_safe_float(getattr(icp_stats, "delta_tx", 0.0), 0.0)),
                        "delta_ty": float(_safe_float(getattr(icp_stats, "delta_ty", 0.0), 0.0)),
                    }
                    if verbose and bool(icp_meta.get("applied", False)):
                        print(
                            f"[INFO] Lane-corr ICP ({chosen_map.name}): "
                            f"iters={icp_meta['iterations']} inliers={icp_meta['inliers']} "
                            f"median {icp_meta['before_median_m']:.3f}->{icp_meta['after_median_m']:.3f} m "
                            f"dtheta={icp_meta['delta_theta_deg']:.4f}deg "
                            f"dtx={icp_meta['delta_tx']:.3f} dty={icp_meta['delta_ty']:.3f}"
                        )
            except Exception as e:
                if verbose:
                    print(f"[WARN] Lane-corr ICP refine failed: {e}")

    payload = _build_lane_corr_payload(
        v2_map=chosen_map,
        carla_lines_xy=carla_lines_for_corr,
        carla_line_records=carla_line_records,
        carla_source_path=carla_source_path,
        carla_name=carla_name,
    )
    if verbose:
        print(
            f"[INFO] Building lane correspondence for {chosen_map.name}: "
            f"v2_lanes={len(payload['map']['lanes'])} carla_lines={len(payload['carla_map']['lines'])}"
        )

    corr = ytm._build_lane_correspondence(
        payload=payload,
        candidate_top_k=max(8, int(lane_corr_top_k)),
        driving_lane_types=[str(v) for v in lane_corr_driving_types],
        cache_dir=lane_corr_cache_dir,
    )
    lane_candidates_raw = corr.get("lane_candidates", {})
    has_candidate_rows = bool(isinstance(lane_candidates_raw, dict) and len(lane_candidates_raw) > 0)
    refresh_missing_candidates = _env_int(
        "V2X_LANE_CORR_REFRESH_WHEN_MISSING_CANDIDATES",
        0,
        minimum=0,
        maximum=1,
    ) == 1
    if (not has_candidate_rows) and bool(refresh_missing_candidates):
        refresh_top_k = max(
            max(8, int(lane_corr_top_k)),
            _env_int("V2X_LANE_CORR_REFRESH_TOP_K", 96, minimum=16, maximum=256),
        )
        if verbose:
            print(
                "[INFO] Lane-corr candidates missing (cache payload). "
                f"Rebuilding uncached for postprocess (top_k={refresh_top_k})..."
            )
        corr = ytm._build_lane_correspondence(
            payload=payload,
            candidate_top_k=int(refresh_top_k),
            driving_lane_types=[str(v) for v in lane_corr_driving_types],
            cache_dir=None,
        )
    elif (not has_candidate_rows) and bool(verbose):
        print(
            "[INFO] Lane-corr cache missing candidate rows; using cached "
            "lane_to_carla directly (refresh disabled)."
        )

    if _env_int("V2X_ENABLE_LANE_CORR_POSTPROCESS", 1, minimum=0, maximum=1) == 1:
        lcd = _load_lane_corr_diag_module()
        if lcd is not None:
            try:
                corr = lcd._postprocess_correspondence(
                    corr=corr,
                    driving_types={str(v) for v in lane_corr_driving_types},
                    keep_non_driving_matches=bool(
                        _env_int("V2X_LANE_CORR_KEEP_NON_DRIVING", 0, minimum=0, maximum=1)
                    ),
                    fill_max_candidates=_env_int(
                        "V2X_LANE_CORR_FILL_MAX_CANDIDATES",
                        16,
                        minimum=1,
                        maximum=64,
                    ),
                    legality_repair_passes=_env_int(
                        "V2X_LANE_CORR_LEGALITY_REPAIR_PASSES",
                        3,
                        minimum=0,
                        maximum=12,
                    ),
                    strict_legal_routes=bool(
                        _env_int("V2X_LANE_CORR_STRICT_LEGAL_ROUTES", 0, minimum=0, maximum=1)
                    ),
                    verbose=bool(verbose),
                )
            except Exception as e:
                if verbose:
                    print(f"[WARN] Lane-corr postprocess failed; using raw correspondence: {e}")

    lane_to_carla_raw = _normalize_int_key_dict(corr.get("lane_to_carla", {}))
    carla_to_lanes_raw = _normalize_int_key_dict(corr.get("carla_to_lanes", {}))
    v2_feats_raw = _normalize_int_key_dict(corr.get("v2_feats", {}))
    carla_successors = _normalize_int_set_map(corr.get("carla_successors", {}))
    lane_to_carla: Dict[int, Dict[str, object]] = {}
    for li, row in lane_to_carla_raw.items():
        if isinstance(row, dict):
            lane_to_carla[int(li)] = dict(row)
    v2_feats: Dict[int, Dict[str, object]] = {}
    for li, row in v2_feats_raw.items():
        if isinstance(row, dict):
            v2_feats[int(li)] = dict(row)
    carla_to_lanes: Dict[int, List[int]] = {}
    for ci, vals in carla_to_lanes_raw.items():
        if isinstance(vals, (list, tuple, set)):
            carla_to_lanes[int(ci)] = sorted(
                set(int(_safe_int(v, -1)) for v in vals if _safe_int(v, -1) >= 0)
            )

    v2_label_by_idx: Dict[int, str] = {}
    for lane in chosen_map.lanes:
        v2_label_by_idx[int(lane.index)] = (
            f"r{int(lane.road_id)}_l{int(lane.lane_id)}_t{str(lane.lane_type)}"
        )

    carla_feats_raw = _normalize_int_key_dict(corr.get("carla_feats", {}))
    carla_feats: Dict[int, Dict[str, object]] = {}
    for ci, row in carla_feats_raw.items():
        if isinstance(row, dict):
            carla_feats[int(ci)] = dict(row)
    if not carla_feats:
        carla_feats = _build_carla_feats_from_lines(carla_lines_for_corr)

    carla_line_records_by_index: Dict[int, Dict[str, object]] = {}
    if isinstance(carla_line_records, (list, tuple)):
        for i, rec in enumerate(carla_line_records):
            if isinstance(rec, dict):
                carla_line_records_by_index[int(i)] = dict(rec)

    # Orientation calibration hints from correspondence + map metadata.
    # CARLA cache polyline order is not guaranteed to equal travel direction,
    # so downstream direction checks use this calibrated per-line heading.
    orient_votes_rev: Dict[int, float] = {}
    orient_votes_fwd: Dict[int, float] = {}
    for li, row in lane_to_carla.items():
        if not isinstance(row, dict):
            continue
        ci = _safe_int(row.get("carla_line_index"), -1)
        if ci < 0:
            continue
        q = str(row.get("quality", "poor")).strip().lower()
        w = 2.0 if q in {"high", "good"} else (1.0 if q in {"medium"} else 0.5)
        if bool(row.get("reversed", False)):
            orient_votes_rev[int(ci)] = float(orient_votes_rev.get(int(ci), 0.0) + float(w))
        else:
            orient_votes_fwd[int(ci)] = float(orient_votes_fwd.get(int(ci), 0.0) + float(w))

    metadata_vote_weight = _env_float("V2X_CARLA_ORIENTATION_METADATA_VOTE_WEIGHT", 4.0)
    metadata_dirsign_lines = 0
    metadata_sign_by_line: Dict[int, int] = {}
    for ci in carla_feats.keys():
        rec = carla_line_records_by_index.get(int(ci), {})
        if not isinstance(rec, dict):
            continue
        dsgn = rec.get("dir_sign")
        try:
            dsgn = int(dsgn) if dsgn is not None else None
        except Exception:
            dsgn = None
        if dsgn == -1:
            metadata_sign_by_line[int(ci)] = -1
            orient_votes_rev[int(ci)] = float(orient_votes_rev.get(int(ci), 0.0) + float(metadata_vote_weight))
            metadata_dirsign_lines += 1
        elif dsgn == 1:
            metadata_sign_by_line[int(ci)] = 1
            orient_votes_fwd[int(ci)] = float(orient_votes_fwd.get(int(ci), 0.0) + float(metadata_vote_weight))
            metadata_dirsign_lines += 1

    orient_margin_ratio = _env_float("V2X_CARLA_ORIENTATION_HINT_MARGIN_RATIO", 1.15)
    orient_min_vote = _env_float("V2X_CARLA_ORIENTATION_HINT_MIN_VOTE", 0.8)
    orient_sign_by_line = _resolve_carla_orientation_signs(
        line_indices=[int(ci) for ci in carla_feats.keys()],
        rev_votes=orient_votes_rev,
        fwd_votes=orient_votes_fwd,
        margin_ratio=float(orient_margin_ratio),
        min_vote=float(orient_min_vote),
    )
    metadata_hard_override = (
        _env_int("V2X_CARLA_ORIENTATION_METADATA_HARD_OVERRIDE", 1, minimum=0, maximum=1) == 1
    )
    if bool(metadata_hard_override):
        for ci, sign in metadata_sign_by_line.items():
            if int(sign) < 0:
                orient_sign_by_line[int(ci)] = -1
            elif int(sign) > 0:
                orient_sign_by_line[int(ci)] = 1
    orient_sign_by_line_base = dict(orient_sign_by_line)
    orient_votes_rev_base = {int(k): float(v) for k, v in orient_votes_rev.items()}
    orient_votes_fwd_base = {int(k): float(v) for k, v in orient_votes_fwd.items()}

    lines_data: List[Dict[str, object]] = []
    all_xy: List[Tuple[float, float]] = []
    for ci in sorted(carla_feats.keys()):
        cf = carla_feats[ci]
        poly = cf.get("poly_xy")
        if not isinstance(poly, np.ndarray) or poly.shape[0] < 2:
            continue
        mid = poly[poly.shape[0] // 2]
        matched = carla_to_lanes.get(int(ci), [])
        matched_labels: List[str] = []
        for li in matched:
            if int(li) in v2_label_by_idx:
                matched_labels.append(str(v2_label_by_idx[int(li)]))
            else:
                vf = v2_feats.get(int(li), {})
                matched_labels.append(
                    f"r{_safe_int(vf.get('road_id'), 0)}_l{_safe_int(vf.get('lane_id'), 0)}_t{str(vf.get('lane_type', ''))}"
                )
        best_quality = "none"
        for li in matched:
            m = lane_to_carla.get(int(li), {})
            q = str(m.get("quality", "poor"))
            if q == "high":
                best_quality = "high"
                break
            if q == "medium" and best_quality not in {"high"}:
                best_quality = "medium"
            if q == "low" and best_quality not in {"high", "medium"}:
                best_quality = "low"
            if q == "poor" and best_quality == "none":
                best_quality = "poor"
        lines_data.append(
            {
                "index": int(ci),
                "label": f"c{int(ci)}",
                "polyline": [[float(p[0]), float(p[1])] for p in poly],
                "mid_x": float(mid[0]),
                "mid_y": float(mid[1]),
                "label_x": float(mid[0]),
                "label_y": float(mid[1]),
                "matched_v2_lane_indices": [int(v) for v in matched],
                "matched_v2_labels": matched_labels,
                "matched_v2_count": int(len(matched)),
                "match_quality": str(best_quality),
                "orientation_hint": "unknown",
                "orientation_rev_vote": 0.0,
                "orientation_fwd_vote": 0.0,
                "road_id": carla_line_records_by_index.get(int(ci), {}).get("road_id"),
                "lane_id": carla_line_records_by_index.get(int(ci), {}).get("lane_id"),
                "dir_sign": carla_line_records_by_index.get(int(ci), {}).get("dir_sign"),
            }
        )
        all_xy.extend((float(p[0]), float(p[1])) for p in poly)

    _apply_orientation_to_lines_data(
        lines_data=lines_data,
        sign_by_line=orient_sign_by_line,
        rev_votes=orient_votes_rev,
        fwd_votes=orient_votes_fwd,
    )

    vertex_xy: List[Tuple[float, float]] = []
    vertex_line: List[int] = []
    for ci, cf in carla_feats.items():
        poly = cf.get("poly_xy")
        if not isinstance(poly, np.ndarray):
            continue
        for p in poly:
            vertex_xy.append((float(p[0]), float(p[1])))
            vertex_line.append(int(ci))
    vertex_arr = np.asarray(vertex_xy, dtype=np.float64) if vertex_xy else np.zeros((0, 2), dtype=np.float64)
    vertex_line_arr = np.asarray(vertex_line, dtype=np.int32) if vertex_line else np.zeros((0,), dtype=np.int32)
    vertex_tree = cKDTree(vertex_arr) if (cKDTree is not None and vertex_arr.shape[0] > 0) else None

    bbox = _compute_bbox_xy(np.asarray(all_xy, dtype=np.float64)) if all_xy else carla_bbox
    corr_enabled = bool(corr.get("enabled", False))
    summary = {
        "enabled": corr_enabled,
        "reason": str(corr.get("reason", "")) if not corr_enabled else "",
        "v2_lane_count": int(len(payload["map"]["lanes"])),
        "mapped_v2_lane_count": int(len(lane_to_carla)),
        "carla_line_count": int(len(lines_data)),
        "matched_carla_line_count": int(sum(1 for row in lines_data if int(row.get("matched_v2_count", 0)) > 0)),
        "map_name": str(chosen_map.name),
        "postprocess": dict(corr.get("postprocess", {})) if isinstance(corr.get("postprocess", {}), dict) else {},
        "icp_refine": dict(icp_meta),
        "orientation_hints": {
            "lines_with_hint": int(sum(1 for v in orient_sign_by_line.values() if int(v) != 0)),
            "reversed_hint_lines": int(sum(1 for v in orient_sign_by_line.values() if int(v) < 0)),
            "forward_hint_lines": int(sum(1 for v in orient_sign_by_line.values() if int(v) > 0)),
            "metadata_dirsign_lines": int(metadata_dirsign_lines),
        },
    }
    if verbose:
        print(
            f"[INFO] Lane correspondence ({chosen_map.name}): enabled={corr_enabled} "
            f"mapped_v2={summary['mapped_v2_lane_count']}/{summary['v2_lane_count']} "
            f"matched_carla={summary['matched_carla_line_count']}/{summary['carla_line_count']}"
        )

    return {
        "enabled": bool(corr_enabled),
        "carla_name": str(carla_name or "carla_map_cache"),
        "summary": summary,
        "lane_to_carla": lane_to_carla,
        "carla_to_lanes": carla_to_lanes,
        "carla_successors": carla_successors,
        "carla_feats": carla_feats,
        "carla_line_records": carla_line_records_by_index,
        "carla_lines_data": lines_data,
        "carla_line_orientation_sign": orient_sign_by_line,
        "carla_line_orientation_rev_vote": orient_votes_rev,
        "carla_line_orientation_fwd_vote": orient_votes_fwd,
        "carla_line_orientation_sign_base": orient_sign_by_line_base,
        "carla_line_orientation_rev_vote_base": orient_votes_rev_base,
        "carla_line_orientation_fwd_vote_base": orient_votes_fwd_base,
        "carla_bbox": bbox,
        "vertex_arr": vertex_arr,
        "vertex_line_arr": vertex_line_arr,
        "vertex_tree": vertex_tree,
    }


def _nearby_carla_lines(
    carla_context: Dict[str, object],
    qx: float,
    qy: float,
    top_k: int = 28,
) -> List[int]:
    vertex_arr = carla_context.get("vertex_arr")
    vertex_line_arr = carla_context.get("vertex_line_arr")
    if not isinstance(vertex_arr, np.ndarray) or vertex_arr.shape[0] <= 0:
        return []
    if not isinstance(vertex_line_arr, np.ndarray) or vertex_line_arr.shape[0] <= 0:
        return []
    kk = max(1, min(int(top_k), int(vertex_arr.shape[0])))
    tree = carla_context.get("vertex_tree")
    if tree is not None:
        _, idxs = tree.query(np.asarray([float(qx), float(qy)], dtype=np.float64), k=kk)
        if np.isscalar(idxs):
            idx_list = [int(idxs)]
        else:
            idx_list = [int(v) for v in np.asarray(idxs).reshape(-1)]
    else:
        d2 = np.sum((vertex_arr - np.asarray([float(qx), float(qy)], dtype=np.float64)[None, :]) ** 2, axis=1)
        idx_list = [int(v) for v in np.argsort(d2)[:kk]]

    out: List[int] = []
    seen: set = set()
    for vi in idx_list:
        if vi < 0 or vi >= int(vertex_line_arr.shape[0]):
            continue
        ci = int(vertex_line_arr[vi])
        if ci in seen:
            continue
        seen.add(ci)
        out.append(ci)
    return out


def _best_projection_on_carla_line(
    carla_context: Dict[str, object],
    line_index: int,
    qx: float,
    qy: float,
    qyaw: float,
    prefer_reversed: Optional[bool] = None,
    enforce_node_direction: bool = False,
    opposite_reject_deg: float = 135.0,
    allow_reversed: Optional[bool] = None,
) -> Optional[Dict[str, object]]:
    carla_feats = carla_context.get("carla_feats", {})
    if not isinstance(carla_feats, dict):
        return None
    cf = carla_feats.get(int(line_index))
    if not isinstance(cf, dict):
        return None

    forward_poly = cf.get("poly_xy")
    forward_cum = cf.get("cum")
    if not isinstance(forward_poly, np.ndarray) or forward_poly.shape[0] < 2:
        return None
    if not isinstance(forward_cum, np.ndarray) or forward_cum.shape[0] != forward_poly.shape[0]:
        return None

    if allow_reversed is None:
        allow_reversed = (
            _env_int("V2X_CARLA_ALLOW_REVERSED_LINE_VARIANT", 1, minimum=0, maximum=1) == 1
        )
    if prefer_reversed is None:
        line_sign = _carla_line_orientation_sign(carla_context, int(line_index))
        if int(line_sign) != 0:
            prefer_reversed = bool(int(line_sign) < 0)
    strict_orientation_hint = (
        _env_int("V2X_CARLA_STRICT_ORIENTATION_HINT", 1, minimum=0, maximum=1) == 1
    )

    # Evaluate forward/reversed variants; by default driving tracks should follow
    # correspondence orientation hint unless explicitly disabled.
    variants: List[Tuple[object, object, bool]]
    rev_poly = cf.get("poly_xy_rev")
    rev_cum = cf.get("cum_rev")
    if not bool(allow_reversed):
        variants = [(forward_poly, forward_cum, False)]
    else:
        if bool(strict_orientation_hint) and prefer_reversed is True:
            variants = [(rev_poly, rev_cum, True)]
        elif bool(strict_orientation_hint) and prefer_reversed is False:
            variants = [(forward_poly, forward_cum, False)]
        elif prefer_reversed is True:
            variants = [
                (rev_poly, rev_cum, True),
                (forward_poly, forward_cum, False),
            ]
        elif prefer_reversed is False:
            variants = [
                (forward_poly, forward_cum, False),
                (rev_poly, rev_cum, True),
            ]
        else:
            variants = [
                (forward_poly, forward_cum, False),
                (rev_poly, rev_cum, True),
            ]

    best: Optional[Dict[str, object]] = None
    for poly_xy, cum, used_rev in variants:
        if not isinstance(poly_xy, np.ndarray) or poly_xy.shape[0] < 2:
            continue
        if not isinstance(cum, np.ndarray) or cum.shape[0] != poly_xy.shape[0]:
            continue
        proj = _project_point_to_polyline_xy(poly_xy, cum, float(qx), float(qy))
        if proj is None:
            continue
        if bool(enforce_node_direction):
            yaw_diff = float(_yaw_abs_diff_deg(float(qyaw), float(proj["yaw"])))
            if yaw_diff >= float(opposite_reject_deg):
                continue
        score = float(proj["dist"]) + 0.018 * float(_yaw_abs_diff_deg(float(qyaw), float(proj["yaw"])))
        cand = {
            "line_index": int(line_index),
            "projection": proj,
            "score": score,
            "reversed": bool(used_rev),
        }
        if best is None or float(cand["score"]) < float(best["score"]):
            best = cand
    return best


def _nearest_projection_any_line(
    carla_context: Dict[str, object],
    qx: float,
    qy: float,
    qyaw: float,
    enforce_node_direction: bool = False,
    opposite_reject_deg: float = 135.0,
    allow_reversed: Optional[bool] = None,
) -> Optional[Dict[str, object]]:
    carla_feats = carla_context.get("carla_feats", {})
    if not isinstance(carla_feats, dict) or not carla_feats:
        return None
    near = _nearby_carla_lines(carla_context, float(qx), float(qy), top_k=48)
    if not near:
        near = [int(ci) for ci in sorted(carla_feats.keys())[:96]]
    best: Optional[Dict[str, object]] = None
    for ci in near:
        cand = _best_projection_on_carla_line(
            carla_context=carla_context,
            line_index=int(ci),
            qx=float(qx),
            qy=float(qy),
            qyaw=float(qyaw),
            prefer_reversed=None,
            enforce_node_direction=bool(enforce_node_direction),
            opposite_reject_deg=float(opposite_reject_deg),
            allow_reversed=allow_reversed,
        )
        if cand is None:
            continue
        if best is None or float(cand["score"]) < float(best["score"]):
            best = cand
    return best


def _carla_line_forward_yaw_at_point(
    carla_context: Dict[str, object],
    line_index: int,
    x: float,
    y: float,
) -> Optional[float]:
    carla_feats = carla_context.get("carla_feats", {})
    if not isinstance(carla_feats, dict):
        return None
    cf = carla_feats.get(int(line_index))
    if not isinstance(cf, dict):
        return None
    poly = cf.get("poly_xy")
    cum = cf.get("cum")
    if not isinstance(poly, np.ndarray) or poly.shape[0] < 2:
        return None
    if not isinstance(cum, np.ndarray) or cum.shape[0] != poly.shape[0]:
        return None
    proj = _project_point_to_polyline_xy(poly, cum, float(x), float(y))
    if not isinstance(proj, dict):
        return None
    yaw = _safe_float(proj.get("yaw"), float("nan"))
    if not math.isfinite(yaw):
        return None
    return float(yaw)


def _carla_line_orientation_sign(
    carla_context: Dict[str, object],
    line_index: int,
) -> int:
    orient = carla_context.get("carla_line_orientation_sign", {})
    if not isinstance(orient, dict):
        return 0
    s = _safe_int(orient.get(int(line_index), orient.get(str(int(line_index)), 0)), 0)
    if s < 0:
        return -1
    if s > 0:
        return 1
    return 0


def _carla_line_calibrated_yaw_at_point(
    carla_context: Dict[str, object],
    line_index: int,
    x: float,
    y: float,
) -> Optional[float]:
    yaw = _carla_line_forward_yaw_at_point(
        carla_context=carla_context,
        line_index=int(line_index),
        x=float(x),
        y=float(y),
    )
    if yaw is None:
        return None
    sign = _carla_line_orientation_sign(carla_context, int(line_index))
    if sign < 0:
        return float(_normalize_yaw_deg(float(yaw) + 180.0))
    return float(yaw)


def _frame_query_pose(frame: Dict[str, object]) -> Tuple[float, float, float]:
    qx = _safe_float(frame.get("sx"), _safe_float(frame.get("x"), 0.0))
    qy = _safe_float(frame.get("sy"), _safe_float(frame.get("y"), 0.0))
    qyaw = _safe_float(frame.get("syaw"), _safe_float(frame.get("yaw"), 0.0))
    return (float(qx), float(qy), float(qyaw))


def _frame_has_carla_pose(frame: Dict[str, object]) -> bool:
    cx = _safe_float(frame.get("cx"), float("nan"))
    cy = _safe_float(frame.get("cy"), float("nan"))
    return math.isfinite(cx) and math.isfinite(cy)


def _set_carla_pose(
    frame: Dict[str, object],
    line_index: int,
    x: float,
    y: float,
    yaw: float,
    dist: float,
    source: str,
    quality: str = "none",
) -> None:
    frame["cx"] = float(x)
    frame["cy"] = float(y)
    frame["cyaw"] = float(yaw)
    frame["ccli"] = int(line_index)
    frame["cdist"] = float(dist)
    frame["csource"] = str(source)
    frame["cquality"] = str(quality)


def _snapshot_carla_pose_frame(frame: Dict[str, object], prefix: str = "cb") -> None:
    """
    Persist current CARLA pose into a secondary namespace on the frame.
    Used for visualization of before/after CARLA post-processing.
    """
    if not isinstance(frame, dict) or not str(prefix):
        return
    if not _frame_has_carla_pose(frame):
        return
    p = str(prefix)
    frame[f"{p}x"] = float(_safe_float(frame.get("cx"), 0.0))
    frame[f"{p}y"] = float(_safe_float(frame.get("cy"), 0.0))
    frame[f"{p}yaw"] = float(_safe_float(frame.get("cyaw"), 0.0))
    frame[f"{p}cli"] = int(_safe_int(frame.get("ccli"), -1))
    dist = _safe_float(frame.get("cdist"), float("nan"))
    if math.isfinite(dist):
        frame[f"{p}dist"] = float(dist)


def _snapshot_carla_pose_frames(frames: List[Dict[str, object]], prefix: str = "cb") -> None:
    if not isinstance(frames, list) or not frames:
        return
    for fr in frames:
        _snapshot_carla_pose_frame(fr, prefix=prefix)


def _regenerate_carla_turn_segments(frames: List[Dict[str, object]]) -> None:
    if len(frames) < 3:
        return

    use_single_turn_line = _env_int("V2X_CARLA_TURN_REGEN_SINGLE_LINE", 1, minimum=0, maximum=1) == 1
    max_anchor_cdist = _env_float("V2X_CARLA_TURN_REGEN_MAX_ANCHOR_CDIST_M", 2.4)
    max_anchor_cdist_delta = _env_float("V2X_CARLA_TURN_REGEN_MAX_ANCHOR_CDIST_DELTA_M", 1.2)

    def _bezier_pose(
        p0: Tuple[float, float],
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        u: float,
    ) -> Tuple[float, float, float]:
        uu = max(0.0, min(1.0, float(u)))
        om = 1.0 - uu
        b0 = om * om * om
        b1 = 3.0 * om * om * uu
        b2 = 3.0 * om * uu * uu
        b3 = uu * uu * uu
        x = b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0]
        y = b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1]
        dx = (
            3.0 * om * om * (p1[0] - p0[0])
            + 6.0 * om * uu * (p2[0] - p1[0])
            + 3.0 * uu * uu * (p3[0] - p2[0])
        )
        dy = (
            3.0 * om * om * (p1[1] - p0[1])
            + 6.0 * om * uu * (p2[1] - p1[1])
            + 3.0 * uu * uu * (p3[1] - p2[1])
        )
        yaw = math.degrees(math.atan2(dy, dx)) if (abs(dx) + abs(dy)) > 1e-9 else 0.0
        return (float(x), float(y), float(_normalize_yaw_deg(yaw)))

    n = len(frames)
    i = 0
    while i < n:
        if not bool(frames[i].get("synthetic_turn", False)):
            i += 1
            continue
        start = i
        while i + 1 < n and bool(frames[i + 1].get("synthetic_turn", False)):
            i += 1
        end = i

        prev_i = start - 1
        next_i = end + 1
        has_prev_anchor = prev_i >= 0 and _frame_has_carla_pose(frames[prev_i])
        has_next_anchor = next_i < n and _frame_has_carla_pose(frames[next_i])
        # If boundary anchor samples are already far from query/raw, they tend to
        # drag regenerated turn arcs into wrong segments. Fall back to query
        # anchors in that case.
        if has_prev_anchor:
            prev_cdist = _safe_float(frames[prev_i].get("cdist"), float("inf"))
            turn_cdist = _safe_float(frames[start].get("cdist"), float("inf"))
            if (
                math.isfinite(prev_cdist)
                and prev_cdist > float(max_anchor_cdist)
                and (not math.isfinite(turn_cdist) or prev_cdist > turn_cdist + float(max_anchor_cdist_delta))
            ):
                has_prev_anchor = False
        if has_next_anchor:
            next_cdist = _safe_float(frames[next_i].get("cdist"), float("inf"))
            turn_cdist = _safe_float(frames[end].get("cdist"), float("inf"))
            if (
                math.isfinite(next_cdist)
                and next_cdist > float(max_anchor_cdist)
                and (not math.isfinite(turn_cdist) or next_cdist > turn_cdist + float(max_anchor_cdist_delta))
            ):
                has_next_anchor = False
        # Support boundary synthetic-turn runs (head/tail) with virtual query anchors.
        # This is important for tracks whose turn segment extends to the final frame.
        if not has_prev_anchor and not has_next_anchor:
            i += 1
            continue

        if has_prev_anchor:
            p0 = (float(frames[prev_i]["cx"]), float(frames[prev_i]["cy"]))
            y0 = math.radians(_safe_float(frames[prev_i].get("cyaw"), 0.0))
            prev_line_idx = _safe_int(frames[prev_i].get("ccli"), -1)
        else:
            qx0, qy0, qyaw0 = _frame_query_pose(frames[start])
            p0 = (float(qx0), float(qy0))
            y0 = math.radians(float(qyaw0))
            prev_line_idx = _safe_int(frames[start].get("ccli"), -1)

        if has_next_anchor:
            p3 = (float(frames[next_i]["cx"]), float(frames[next_i]["cy"]))
            y3 = math.radians(_safe_float(frames[next_i].get("cyaw"), 0.0))
            next_line_idx = _safe_int(frames[next_i].get("ccli"), prev_line_idx)
        else:
            qx3, qy3, qyaw3 = _frame_query_pose(frames[end])
            p3 = (float(qx3), float(qy3))
            y3 = math.radians(float(qyaw3))
            next_line_idx = _safe_int(frames[end].get("ccli"), prev_line_idx)

        span = math.hypot(p3[0] - p0[0], p3[1] - p0[1])
        if span < 1e-3:
            i += 1
            continue

        # Guard against long cross-map jumps: the CARLA anchor span should be
        # reasonably close to the raw/V2 query span across the same interval.
        qx_start, qy_start, _ = _frame_query_pose(frames[start])
        qx_end, qy_end, _ = _frame_query_pose(frames[end])
        raw_span = math.hypot(float(qx_end) - float(qx_start), float(qy_end) - float(qy_start))
        if span > max(18.0, float(raw_span) + 10.0):
            i += 1
            continue

        d = max(2.0, min(18.0, 0.45 * span))
        p1 = (p0[0] + d * math.cos(y0), p0[1] + d * math.sin(y0))
        p2 = (p3[0] - d * math.cos(y3), p3[1] - d * math.sin(y3))

        orig_dists: List[float] = []
        for fi in range(start, end + 1):
            fr0 = frames[fi]
            qx0, qy0, _ = _frame_query_pose(fr0)
            ox = _safe_float(fr0.get("cx"), float(qx0))
            oy = _safe_float(fr0.get("cy"), float(qy0))
            orig_dists.append(float(math.hypot(float(ox) - float(qx0), float(oy) - float(qy0))))

        candidate_rows: List[Tuple[int, float, float, float, float, int]] = []
        denom = max(1, end - start)
        for fi in range(start, end + 1):
            fr = frames[fi]
            u_raw = _safe_float(fr.get("turn_u"), float(fi - start) / float(denom))
            u = max(0.0, min(1.0, float(u_raw)))
            x, y, yaw = _bezier_pose(p0, p1, p2, p3, u)
            qx, qy, _ = _frame_query_pose(fr)
            if bool(use_single_turn_line):
                line_idx = int(prev_line_idx if prev_line_idx >= 0 else next_line_idx)
            else:
                line_idx = int(prev_line_idx) if u < 0.5 else int(next_line_idx)
            dist = float(math.hypot(float(x) - float(qx), float(y) - float(qy)))
            candidate_rows.append((fi, float(x), float(y), float(yaw), dist, int(line_idx)))

        if not candidate_rows:
            i += 1
            continue

        new_dists = [float(row[4]) for row in candidate_rows]
        orig_sorted = sorted(orig_dists)
        new_sorted = sorted(new_dists)
        orig_sum = float(sum(orig_sorted))
        new_sum = float(sum(new_sorted))
        orig_max = float(orig_sorted[-1]) if orig_sorted else 0.0
        new_max = float(new_sorted[-1]) if new_sorted else 0.0
        orig_p90 = float(orig_sorted[min(len(orig_sorted) - 1, int(0.9 * len(orig_sorted)))]) if orig_sorted else 0.0
        new_p90 = float(new_sorted[min(len(new_sorted) - 1, int(0.9 * len(new_sorted)))]) if new_sorted else 0.0

        # Only keep regenerated CARLA turn curves when they stay close to raw/V2
        # and do not meaningfully degrade the already-projected turn section.
        interval_len = max(1, end - start + 1)
        if new_sum > (orig_sum + 0.25 * float(interval_len)):
            i += 1
            continue
        if new_p90 > min(8.0, max(4.5, orig_p90 + 0.8)):
            i += 1
            continue
        if new_max > min(14.0, max(7.0, orig_max + 1.4)):
            i += 1
            continue

        for fi, x, y, yaw, dist, line_idx in candidate_rows:
            fr = frames[fi]
            _set_carla_pose(
                frame=fr,
                line_index=int(line_idx),
                x=float(x),
                y=float(y),
                yaw=float(yaw),
                dist=float(dist),
                source="turn_regen",
                quality=str(fr.get("cquality", "none")),
            )
        i += 1


def _bridge_missing_carla_gaps(
    frames: List[Dict[str, object]],
    carla_context: Dict[str, object],
    max_gap_frames: int = 30,
    max_bridge_dist_m: float = 42.0,
    enforce_node_direction: bool = False,
    opposite_reject_deg: float = 135.0,
) -> None:
    if not frames:
        return
    n = len(frames)
    i = 0
    while i < n:
        if _frame_has_carla_pose(frames[i]):
            i += 1
            continue
        start = i
        while i < n and not _frame_has_carla_pose(frames[i]):
            i += 1
        end = i - 1
        prev_i = start - 1 if start - 1 >= 0 and _frame_has_carla_pose(frames[start - 1]) else -1
        next_i = i if i < n and _frame_has_carla_pose(frames[i]) else -1
        gap_len = end - start + 1

        bridged = False
        if prev_i >= 0 and next_i >= 0 and gap_len <= int(max_gap_frames):
            px = float(frames[prev_i]["cx"])
            py = float(frames[prev_i]["cy"])
            nx = float(frames[next_i]["cx"])
            ny = float(frames[next_i]["cy"])
            jump = math.hypot(nx - px, ny - py)
            if jump <= float(max_bridge_dist_m):
                pyaw = _safe_float(frames[prev_i].get("cyaw"), 0.0)
                nyaw = _safe_float(frames[next_i].get("cyaw"), pyaw)
                pcli = _safe_int(frames[prev_i].get("ccli"), -1)
                ncli = _safe_int(frames[next_i].get("ccli"), pcli)
                bridge_rows: List[Tuple[int, float, float, float, float, int]] = []
                bridge_valid = True
                for fi in range(start, end + 1):
                    alpha = float(fi - prev_i) / float(next_i - prev_i)
                    x = (1.0 - alpha) * px + alpha * nx
                    y = (1.0 - alpha) * py + alpha * ny
                    yaw = _interp_yaw_deg(pyaw, nyaw, alpha)
                    qx, qy, _ = _frame_query_pose(frames[fi])
                    qyaw = _safe_float(frames[fi].get("syaw"), _safe_float(frames[fi].get("yaw"), 0.0))
                    if bool(enforce_node_direction):
                        yaw_diff = float(_yaw_abs_diff_deg(float(qyaw), float(yaw)))
                        if yaw_diff >= float(opposite_reject_deg):
                            bridge_valid = False
                            break
                    line_idx = pcli if alpha < 0.5 else ncli
                    dist = float(math.hypot(float(x) - float(qx), float(y) - float(qy)))
                    bridge_rows.append((int(fi), float(x), float(y), float(yaw), dist, int(line_idx)))
                if bridge_valid and bridge_rows:
                    for fi, x, y, yaw, dist, line_idx in bridge_rows:
                        _set_carla_pose(
                            frame=frames[fi],
                            line_index=int(line_idx),
                            x=float(x),
                            y=float(y),
                            yaw=float(yaw),
                            dist=float(dist),
                            source="gap_bridge",
                            quality="none",
                        )
                    bridged = True

        if bridged:
            continue

        for fi in range(start, end + 1):
            fr = frames[fi]
            qx, qy, qyaw = _frame_query_pose(fr)
            cand = _nearest_projection_any_line(
                carla_context=carla_context,
                qx=float(qx),
                qy=float(qy),
                qyaw=float(qyaw),
                enforce_node_direction=bool(enforce_node_direction),
                opposite_reject_deg=float(opposite_reject_deg),
            )
            if cand is None:
                _set_carla_pose(
                    frame=fr,
                    line_index=-1,
                    x=float(qx),
                    y=float(qy),
                    yaw=float(qyaw),
                    dist=0.0,
                    source="raw_fallback",
                    quality="none",
                )
                continue
            proj = cand["projection"]
            _set_carla_pose(
                frame=fr,
                line_index=int(cand["line_index"]),
                x=float(proj["x"]),
                y=float(proj["y"]),
                yaw=float(proj["yaw"]),
                dist=float(proj["dist"]),
                source="nearest_gap",
                quality="none",
            )


def _smooth_carla_jump_stall_artifacts(frames: List[Dict[str, object]]) -> None:
    """
    Reduce CARLA projection jump-then-wait artifacts by redistributing motion
    over stalled windows using query displacement timing.
    """
    if len(frames) < 4:
        return

    RAW_MIN_STEP = _env_float("V2X_CARLA_JUMP_RAW_MIN_STEP", 0.25)
    RAW_MAX_STEP = _env_float("V2X_CARLA_JUMP_RAW_MAX_STEP", 2.50)
    MIN_JUMP = _env_float("V2X_CARLA_JUMP_MIN", 2.20)
    JUMP_RATIO = _env_float("V2X_CARLA_JUMP_RATIO", 2.80)
    STALL_STEP = _env_float("V2X_CARLA_STALL_STEP", 0.12)
    RESUME_STEP = _env_float("V2X_CARLA_RESUME_STEP", 0.35)
    MIN_STALL_FRAMES = _env_int("V2X_CARLA_MIN_STALL_FRAMES", 2, minimum=1, maximum=12)
    LOOKAHEAD = _env_int("V2X_CARLA_JUMP_LOOKAHEAD", 14, minimum=2, maximum=40)
    COST_SLACK_BASE = _env_float("V2X_CARLA_JUMP_COST_SLACK_BASE", 0.80)
    COST_SLACK_PER_FRAME = _env_float("V2X_CARLA_JUMP_COST_SLACK_PER_FRAME", 0.35)
    MAX_POINT_DELTA = _env_float("V2X_CARLA_JUMP_MAX_POINT_DELTA", 2.0)
    MAX_POINT_DIST = _env_float("V2X_CARLA_JUMP_MAX_POINT_DIST", 12.0)
    SKIP_ON_LINE_CHANGE = _env_int("V2X_CARLA_JUMP_SKIP_ON_LINE_CHANGE", 1, minimum=0, maximum=1) == 1
    SKIP_ON_SEMANTIC_TRANSITION = _env_int("V2X_CARLA_JUMP_SKIP_ON_SEMANTIC_TRANSITION", 1, minimum=0, maximum=1) == 1
    SKIP_NEAR_SYNTHETIC_TURN = _env_int("V2X_CARLA_JUMP_SKIP_NEAR_SYNTHETIC_TURN", 1, minimum=0, maximum=1) == 1

    n = len(frames)

    def _semantic_lane_key(fr: Dict[str, object]) -> Optional[Tuple[int, int]]:
        rid = _safe_int(fr.get("road_id"), -1)
        assigned_lid = _safe_int(fr.get("assigned_lane_id"), 0)
        lane_id = _safe_int(fr.get("lane_id"), 0)
        if rid >= 0:
            if assigned_lid != 0:
                return (int(rid), int(assigned_lid))
            if lane_id != 0:
                return (int(rid), int(lane_id))
        return None

    def _has_semantic_lane_transition(fr0: Dict[str, object], fr1: Dict[str, object]) -> bool:
        k0 = _semantic_lane_key(fr0)
        k1 = _semantic_lane_key(fr1)
        if k0 is not None and k1 is not None:
            return bool(k0 != k1)
        prev_assigned = _safe_int(fr0.get("assigned_lane_id"), 0)
        curr_assigned = _safe_int(fr1.get("assigned_lane_id"), 0)
        if prev_assigned != 0 and curr_assigned != 0:
            return bool(prev_assigned != curr_assigned)
        prev_idx = _safe_int(fr0.get("lane_index"), -1)
        curr_idx = _safe_int(fr1.get("lane_index"), -1)
        if prev_idx >= 0 and curr_idx >= 0:
            return bool(prev_idx != curr_idx)
        return False

    def _query_step(i: int) -> float:
        if i <= 0 or i >= n:
            return 0.0
        x0, y0, _ = _frame_query_pose(frames[i - 1])
        x1, y1, _ = _frame_query_pose(frames[i])
        return float(math.hypot(float(x1) - float(x0), float(y1) - float(y0)))

    def _carla_step(i: int) -> float:
        if i <= 0 or i >= n:
            return 0.0
        if not _frame_has_carla_pose(frames[i - 1]) or not _frame_has_carla_pose(frames[i]):
            return 0.0
        x0 = float(frames[i - 1]["cx"])
        y0 = float(frames[i - 1]["cy"])
        x1 = float(frames[i]["cx"])
        y1 = float(frames[i]["cy"])
        return float(math.hypot(x1 - x0, y1 - y0))

    i = 1
    while i < n - 2:
        if not _frame_has_carla_pose(frames[i - 1]) or not _frame_has_carla_pose(frames[i]):
            i += 1
            continue

        if bool(SKIP_NEAR_SYNTHETIC_TURN):
            lo = max(0, int(i) - 2)
            hi = min(n - 1, int(i) + 2)
            if any(bool(frames[k].get("synthetic_turn", False)) for k in range(lo, hi + 1)):
                i += 1
                continue

        if bool(SKIP_ON_LINE_CHANGE):
            prev_line = _safe_int(frames[i - 1].get("ccli"), -1)
            curr_line = _safe_int(frames[i].get("ccli"), -1)
            if prev_line >= 0 and curr_line >= 0 and prev_line != curr_line:
                i += 1
                continue

        if bool(SKIP_ON_SEMANTIC_TRANSITION) and _has_semantic_lane_transition(frames[i - 1], frames[i]):
            i += 1
            continue

        raw_jump = _query_step(i)
        carla_jump = _carla_step(i)
        if raw_jump < RAW_MIN_STEP or raw_jump > RAW_MAX_STEP:
            i += 1
            continue
        if carla_jump < max(MIN_JUMP, JUMP_RATIO * raw_jump):
            i += 1
            continue

        scan_end = min(n - 1, i + LOOKAHEAD)
        stall_last = i
        stall_count = 0
        j = i + 1
        while j <= scan_end:
            if not _frame_has_carla_pose(frames[j - 1]) or not _frame_has_carla_pose(frames[j]):
                break
            raw_step = _query_step(j)
            carla_step = _carla_step(j)
            if raw_step > (2.0 * RAW_MAX_STEP):
                break
            if raw_step >= RAW_MIN_STEP and carla_step <= STALL_STEP:
                stall_last = j
                stall_count += 1
                j += 1
                continue
            if carla_step >= RESUME_STEP:
                break
            if raw_step >= RAW_MIN_STEP and carla_step <= (1.8 * STALL_STEP):
                stall_last = j
                stall_count += 1
                j += 1
                continue
            break

        # For extreme jump artifacts, a single immediate stall frame is often
        # enough evidence (e.g., jump then freeze for one frame before resume).
        required_stall_frames = int(MIN_STALL_FRAMES)
        if carla_jump >= max(5.0, 5.5 * raw_jump):
            required_stall_frames = 1
        if stall_count < required_stall_frames:
            i += 1
            continue

        start_x = float(frames[i - 1]["cx"])
        start_y = float(frames[i - 1]["cy"])
        end_idx = -1
        use_query_anchor = False
        for k in range(stall_last + 1, scan_end + 1):
            if not _frame_has_carla_pose(frames[k]):
                break
            d = math.hypot(float(frames[k]["cx"]) - start_x, float(frames[k]["cy"]) - start_y)
            if d >= max(0.8, 1.5 * raw_jump):
                end_idx = k
                break
        if end_idx < 0:
            fallback = stall_last + 1
            if fallback <= scan_end and _frame_has_carla_pose(frames[fallback]):
                end_idx = int(fallback)
            elif stall_last >= (n - 1):
                # Terminal stall: no future CARLA anchor exists.
                # Use query pose tail as a soft anchor to avoid visible freeze.
                end_idx = int(stall_last)
                use_query_anchor = True
            else:
                i += 1
                continue
        if end_idx <= i:
            i += 1
            continue
        if not use_query_anchor and not _frame_has_carla_pose(frames[end_idx]):
            i += 1
            continue

        qx_end, qy_end, qyaw_end = _frame_query_pose(frames[end_idx])
        end_x = float(qx_end) if use_query_anchor else float(frames[end_idx]["cx"])
        end_y = float(qy_end) if use_query_anchor else float(frames[end_idx]["cy"])
        if math.hypot(end_x - start_x, end_y - start_y) < 0.5:
            i = end_idx + 1
            continue

        cum: List[float] = [0.0]
        total = 0.0
        for t in range(i, end_idx + 1):
            step = min(_query_step(t), 1.25 * RAW_MAX_STEP)
            total += max(0.0, float(step))
            cum.append(float(total))
        if total <= 1e-6:
            total = float(max(1, end_idx - (i - 1)))
            cum = [float(v) for v in range(0, end_idx - (i - 1) + 1)]

        yaw_start = _safe_float(frames[i - 1].get("cyaw"), 0.0)
        yaw_end = float(qyaw_end) if use_query_anchor else _safe_float(frames[end_idx].get("cyaw"), yaw_start)
        cli_start = _safe_int(frames[i - 1].get("ccli"), -1)
        cli_end = _safe_int(frames[i].get("ccli"), cli_start) if use_query_anchor else _safe_int(frames[end_idx].get("ccli"), cli_start)
        quality_start = str(frames[i - 1].get("cquality", "none"))
        quality_end = quality_start if use_query_anchor else str(frames[end_idx].get("cquality", quality_start))

        old_cost = 0.0
        new_cost = 0.0
        old_max = 0.0
        new_max = 0.0
        patch_rows: List[Tuple[int, float, float, float, float, int, str]] = []
        denom = max(1e-6, float(total))

        for off, t in enumerate(range(i, end_idx + 1), start=1):
            if not _frame_has_carla_pose(frames[t]):
                patch_rows = []
                break
            alpha = max(0.0, min(1.0, float(cum[off]) / denom))
            x = (1.0 - alpha) * start_x + alpha * end_x
            y = (1.0 - alpha) * start_y + alpha * end_y
            yaw = _interp_yaw_deg(yaw_start, yaw_end, alpha)
            qx, qy, _ = _frame_query_pose(frames[t])
            old_x = float(frames[t]["cx"])
            old_y = float(frames[t]["cy"])
            old_dist = math.hypot(old_x - float(qx), old_y - float(qy))
            new_dist = math.hypot(float(x) - float(qx), float(y) - float(qy))
            old_cost += float(old_dist)
            new_cost += float(new_dist)
            old_max = max(old_max, float(old_dist))
            new_max = max(new_max, float(new_dist))
            line_idx = cli_start if alpha < 0.5 else cli_end
            quality = quality_start if alpha < 0.5 else quality_end
            patch_rows.append((t, float(x), float(y), float(yaw), float(new_dist), int(line_idx), str(quality)))

        if len(patch_rows) < 2:
            i += 1
            continue

        local_slack = float(COST_SLACK_BASE) + float(COST_SLACK_PER_FRAME) * float(len(patch_rows))
        if new_cost > old_cost + local_slack:
            i += 1
            continue
        if new_max > max(float(MAX_POINT_DIST), old_max + float(MAX_POINT_DELTA)):
            i += 1
            continue

        for t, x, y, yaw, dist, line_idx, quality in patch_rows:
            _set_carla_pose(
                frame=frames[t],
                line_index=int(line_idx),
                x=float(x),
                y=float(y),
                yaw=float(yaw),
                dist=float(dist),
                source="jump_stall_bridge",
                quality=quality,
            )
            frames[t]["continuity_bridge"] = True

        i = end_idx + 1


def _smooth_short_carla_line_runs(
    frames: List[Dict[str, object]],
    carla_context: Dict[str, object],
    track_label: str = "",
    max_mid_run_frames: int = 12,
    max_edge_run_frames: int = 2,
    max_head_run_frames: int = 2,
    min_stable_neighbor_frames: int = 30,
    transition_spike_max_frames: int = 3,
    transition_min_neighbor_frames: int = 10,
    transition_cost_slack_base: float = 1.2,
    transition_cost_slack_per_frame: float = 0.45,
    transition_max_point_delta: float = 2.0,
    cost_slack_base: float = 0.5,
    cost_slack_per_frame: float = 0.35,
    aba_cost_slack_base: float = 4.0,
    aba_cost_slack_per_frame: float = 1.0,
    max_point_delta: float = 1.5,
    aba_max_point_delta: float = 4.0,
    max_point_dist: float = 12.0,
    aba_max_point_dist: float = 12.0,
    aba_enforce_node_direction: bool = False,
    enforce_node_direction: bool = False,
    opposite_reject_deg: float = 135.0,
) -> None:
    """
    Smooth short CARLA line-index flickers (ccli) caused by local projection noise.

    Patterns handled:
    - Internal A->B->A where B is short.
    - Edge spikes at trajectory head/tail where a tiny run differs from a long stable run.
    - Internal A->B->C transition spikes where B is a very short flicker.

    Replacements are only accepted when they do not materially worsen query fit.
    """
    if len(frames) < 2:
        return
    smooth_t0 = time.perf_counter()
    actor_name = str(track_label).strip() or "unknown"
    smooth_progress_every = _env_int(
        "V2X_CARLA_SMOOTH_PROGRESS_EVERY_ITERS",
        200,
        minimum=10,
        maximum=1000000,
    )
    print(
        "[PERF] smooth_short_runs start: "
        f"actor={actor_name} frames={int(len(frames))}"
    )
    smooth_iter_cap = _env_int(
        "V2X_CARLA_SMOOTH_SHORT_RUN_MAX_ITERS",
        0,
        minimum=0,
        maximum=50000000,
    )
    smooth_iter_count = 0
    stop_reason = "unknown"

    line_meta_by_index: Dict[int, Dict[str, object]] = {}
    raw_lines_data = carla_context.get("carla_lines_data", [])
    if isinstance(raw_lines_data, list):
        for row in raw_lines_data:
            if not isinstance(row, dict):
                continue
            li = _safe_int(row.get("index"), -1)
            if li >= 0:
                line_meta_by_index[int(li)] = row

    line_semantic_cache: Dict[int, set] = {}

    def _line_semantic_tokens(line_idx: int) -> set:
        li = int(line_idx)
        cached = line_semantic_cache.get(li)
        if cached is not None:
            return cached
        out: set = set()
        row = line_meta_by_index.get(li, {})
        if isinstance(row, dict):
            lane_idxs = row.get("matched_v2_lane_indices", [])
            if isinstance(lane_idxs, (list, tuple, set)):
                for v in lane_idxs:
                    vi = _safe_int(v, -1)
                    if vi >= 0:
                        out.add(("v2_idx", int(vi)))
            labels = row.get("matched_v2_labels", [])
            if isinstance(labels, (list, tuple, set)):
                for lbl in labels:
                    text = str(lbl)
                    m = re.search(r"_l(-?\d+)_t([^_]+)$", text)
                    if m:
                        lane_i = _safe_int(m.group(1), 0)
                        lane_t = str(m.group(2))
                        out.add(("lane", int(lane_i), lane_t))
        line_semantic_cache[li] = out
        return out

    def _lines_semantically_equivalent(line_a: int, line_b: int) -> bool:
        a = int(line_a)
        b = int(line_b)
        if a == b:
            return True
        sa = _line_semantic_tokens(a)
        sb = _line_semantic_tokens(b)
        if not sa or not sb:
            return False
        return bool(sa & sb)

    def _frame_line(fi: int) -> Optional[int]:
        if fi < 0 or fi >= len(frames):
            return None
        fr = frames[fi]
        if not _frame_has_carla_pose(fr):
            return None
        li = _safe_int(fr.get("ccli"), -1)
        return int(li) if li >= 0 else None

    def _build_runs() -> List[Tuple[int, int, int]]:
        runs: List[Tuple[int, int, int]] = []  # (start, end, ccli)
        n = len(frames)
        i = 0
        while i < n:
            li = _frame_line(i)
            if li is None:
                i += 1
                continue
            start = i
            while i + 1 < n and _frame_line(i + 1) == li:
                i += 1
            runs.append((int(start), int(i), int(li)))
            i += 1
        return runs

    def _collect_replace_plan(start: int, end: int, target_line: int, _dir_override: Optional[bool] = None) -> Optional[Dict[str, object]]:
        rows: List[Tuple[int, Dict[str, object]]] = []
        old_cost = 0.0
        new_cost = 0.0
        old_max = 0.0
        new_max = 0.0
        _eff_enforce_dir = bool(enforce_node_direction) if _dir_override is None else bool(_dir_override)

        for fi in range(int(start), int(end) + 1):
            fr = frames[fi]
            if not _frame_has_carla_pose(fr):
                return None
            qx, qy, qyaw = _frame_query_pose(fr)
            cand = _best_projection_on_carla_line(
                carla_context=carla_context,
                line_index=int(target_line),
                qx=float(qx),
                qy=float(qy),
                qyaw=float(qyaw),
                prefer_reversed=None,
                enforce_node_direction=_eff_enforce_dir,
                opposite_reject_deg=float(opposite_reject_deg),
            )
            if cand is None:
                return None
            proj = cand.get("projection", {})
            if not isinstance(proj, dict):
                return None

            old_dist = _safe_float(fr.get("cdist"), float("nan"))
            if not math.isfinite(old_dist):
                old_dist = math.hypot(
                    _safe_float(fr.get("cx"), float(qx)) - float(qx),
                    _safe_float(fr.get("cy"), float(qy)) - float(qy),
                )
            new_dist = float(_safe_float(proj.get("dist"), 0.0))
            old_cost += float(old_dist)
            new_cost += float(new_dist)
            old_max = max(float(old_max), float(old_dist))
            new_max = max(float(new_max), float(new_dist))
            rows.append((fi, cand))

        return {
            "rows": rows,
            "old_cost": float(old_cost),
            "new_cost": float(new_cost),
            "old_max": float(old_max),
            "new_max": float(new_max),
        }

    def _apply_replace_plan(plan: Dict[str, object], target_line: int, source_tag: str) -> None:
        rows = plan.get("rows", [])
        if not isinstance(rows, list):
            return
        for fi, cand in rows:
            proj = cand["projection"]
            fr = frames[int(fi)]
            _set_carla_pose(
                frame=fr,
                line_index=int(target_line),
                x=float(proj["x"]),
                y=float(proj["y"]),
                yaw=float(proj["yaw"]),
                dist=float(proj["dist"]),
                source=str(source_tag),
                quality=str(fr.get("cquality", "none")),
            )
            fr["line_flicker_smooth"] = True

    def _try_replace_run(
        start: int,
        end: int,
        target_line: int,
        source_tag: str,
        local_cost_slack_base: float,
        local_cost_slack_per_frame: float,
        local_max_point_delta: float,
    ) -> bool:
        plan = _collect_replace_plan(int(start), int(end), int(target_line))
        if not isinstance(plan, dict):
            return False

        run_len = max(1, int(end - start + 1))
        local_slack = float(local_cost_slack_base) + float(local_cost_slack_per_frame) * float(run_len)
        old_cost = float(plan.get("old_cost", 0.0))
        new_cost = float(plan.get("new_cost", 0.0))
        old_max = float(plan.get("old_max", 0.0))
        new_max = float(plan.get("new_max", 0.0))
        if new_cost > old_cost + local_slack:
            return False
        if new_max > max(float(max_point_dist), old_max + float(local_max_point_delta)):
            return False

        _apply_replace_plan(plan, int(target_line), str(source_tag))
        return True

    while True:
        smooth_iter_count += 1
        if int(smooth_iter_cap) > 0 and int(smooth_iter_count) > int(smooth_iter_cap):
            actor_txt = f" actor={str(track_label)}" if str(track_label).strip() else ""
            print(
                "[WARN] _smooth_short_carla_line_runs: max-iters cap reached "
                f"(cap={int(smooth_iter_cap)}). Leaving loop early.{actor_txt} "
                f"frames={int(len(frames))}"
            )
            stop_reason = f"max_iters_cap_{int(smooth_iter_cap)}"
            break
        runs = _build_runs()
        if not runs:
            stop_reason = "no_runs"
            break
        if int(smooth_iter_count) == 1 or (int(smooth_iter_count) % int(smooth_progress_every) == 0):
            print(
                "[PERF] smooth_short_runs progress: "
                f"actor={actor_name} iter={int(smooth_iter_count)} runs={int(len(runs))} "
                f"elapsed={time.perf_counter() - smooth_t0:.2f}s"
            )

        changed = False

        # Head spike — also handles longer wrong-head runs (spawn-mid-turn) via
        # max_head_run_frames. Use ABA-style generous slacks for head spikes
        # since initial frames often have misassignments due to lane_index
        # correspondence not matching the actual intended lane.
        if len(runs) >= 2:
            hs, he, hline = runs[0]
            ns, ne, nline = runs[1]
            hlen = int(he - hs + 1)
            nlen = int(ne - ns + 1)
            _head_thresh = max(int(max_edge_run_frames), int(max_head_run_frames))
            if (
                int(hlen) <= int(_head_thresh)
                and int(nlen) >= int(min_stable_neighbor_frames)
                and int(hline) != int(nline)
            ):
                # Always use generous ABA-style slack for head spikes, since
                # spawn-time lane assignments are often based on incomplete
                # trajectory context (no motion history), leading to
                # misassignments that should be corrected based on the
                # dominant subsequent lane assignment.
                if _try_replace_run(
                    hs,
                    he,
                    int(nline),
                    source_tag="head_line_spike",
                    local_cost_slack_base=float(aba_cost_slack_base),
                    local_cost_slack_per_frame=float(aba_cost_slack_per_frame),
                    local_max_point_delta=float(aba_max_point_delta),
                ):
                    changed = True

        if changed:
            continue

        # Tail spike
        if len(runs) >= 2:
            ps, pe, pline = runs[-2]
            ts, te, tline = runs[-1]
            tlen = int(te - ts + 1)
            plen = int(pe - ps + 1)
            if (
                int(tlen) <= int(max_edge_run_frames)
                and int(plen) >= int(min_stable_neighbor_frames)
                and int(pline) != int(tline)
            ):
                if _try_replace_run(
                    ts,
                    te,
                    int(pline),
                    source_tag="tail_line_spike",
                    local_cost_slack_base=float(cost_slack_base),
                    local_cost_slack_per_frame=float(cost_slack_per_frame),
                    local_max_point_delta=float(max_point_delta),
                ):
                    changed = True

        if changed:
            continue

        # Internal A->B->C transition spikes where B is very short.
        # This catches one-frame line-id glitches during lane transitions.
        for ridx in range(1, len(runs) - 1):
            bs, be, bline = runs[ridx]
            as_, ae, aline = runs[ridx - 1]
            cs, ce, cline = runs[ridx + 1]
            blen = int(be - bs + 1)
            if blen > int(transition_spike_max_frames):
                continue
            if int(aline) == int(cline):
                continue
            if int(bline) == int(aline) or int(bline) == int(cline):
                continue
            prev_len = int(ae - as_ + 1)
            next_len = int(ce - cs + 1)
            if prev_len < int(transition_min_neighbor_frames) and next_len < int(transition_min_neighbor_frames):
                continue

            candidate_plans: List[Tuple[float, float, int, str, Dict[str, object]]] = []
            run_len = max(1, blen)
            local_slack = float(transition_cost_slack_base) + float(transition_cost_slack_per_frame) * float(run_len)
            max_delta = float(max(transition_max_point_delta, max_point_delta))

            for target_line, source_tag in (
                (int(aline), "transition_line_spike_prev"),
                (int(cline), "transition_line_spike_next"),
            ):
                plan = _collect_replace_plan(int(bs), int(be), int(target_line))
                if not isinstance(plan, dict):
                    continue
                old_cost = float(plan.get("old_cost", 0.0))
                new_cost = float(plan.get("new_cost", 0.0))
                old_max = float(plan.get("old_max", 0.0))
                new_max = float(plan.get("new_max", 0.0))
                if new_cost > old_cost + local_slack:
                    continue
                if new_max > max(float(max_point_dist), old_max + max_delta):
                    continue
                candidate_plans.append((new_cost, new_max, target_line, source_tag, plan))

            if not candidate_plans:
                continue
            candidate_plans.sort(key=lambda row: (float(row[0]), float(row[1])))
            _, _, target_line, source_tag, best_plan = candidate_plans[0]
            _apply_replace_plan(best_plan, int(target_line), str(source_tag))
            changed = True
            break

        if changed:
            continue

        # Internal A->B->A short flickers
        for ridx in range(1, len(runs) - 1):
            bs, be, bline = runs[ridx]
            as_, ae, aline = runs[ridx - 1]
            cs, ce, cline = runs[ridx + 1]
            blen = int(be - bs + 1)
            if blen > int(max_mid_run_frames):
                continue
            if not _lines_semantically_equivalent(int(aline), int(cline)):
                continue
            if int(bline) == int(aline):
                continue
            prev_len = int(ae - as_ + 1)
            next_len = int(ce - cs + 1)
            is_edge_tail = (ridx + 1 == len(runs) - 1)
            is_edge_head = (ridx - 1 == 0)
            if prev_len < 4 or next_len < 4:
                # Allow tiny A->B->A flickers at sequence edges where one side is
                # naturally short but the other side is stable.
                edge_ok = (
                    (is_edge_tail and prev_len >= 6 and next_len >= 1 and blen <= max(4, int(max_mid_run_frames)))
                    or (is_edge_head and next_len >= 6 and prev_len >= 1 and blen <= max(4, int(max_mid_run_frames)))
                )
                if not edge_ok:
                    continue
            run_len = max(1, int(be - bs + 1))
            local_slack = (
                float(max(cost_slack_base, aba_cost_slack_base))
                + float(max(cost_slack_per_frame, aba_cost_slack_per_frame)) * float(run_len)
            )
            local_max_delta = float(max(max_point_delta, aba_max_point_delta))
            candidate_lines: List[Tuple[int, str]] = [(int(aline), "aba_line_flicker")]
            if int(cline) != int(aline):
                candidate_lines.append((int(cline), "aba_line_flicker_semantic"))
            best_choice: Optional[Tuple[float, float, int, str, Dict[str, object]]] = None
            for target_line, source_tag in candidate_lines:
                plan = _collect_replace_plan(int(bs), int(be), int(target_line), _dir_override=aba_enforce_node_direction)
                if not isinstance(plan, dict):
                    continue
                old_cost = float(plan.get("old_cost", 0.0))
                new_cost = float(plan.get("new_cost", 0.0))
                old_max = float(plan.get("old_max", 0.0))
                new_max = float(plan.get("new_max", 0.0))
                if new_cost > old_cost + local_slack:
                    continue
                if new_max > max(float(max_point_dist), float(aba_max_point_dist), old_max + local_max_delta):
                    continue
                row = (new_cost, new_max, int(target_line), str(source_tag), plan)
                if best_choice is None or (row[0], row[1]) < (best_choice[0], best_choice[1]):
                    best_choice = row
            if best_choice is None:
                continue
            _, _, target_line, source_tag, plan = best_choice
            _apply_replace_plan(plan, int(target_line), str(source_tag))
            changed = True
            break

        if not changed:
            stop_reason = "converged_no_change"
            break
    print(
        "[PERF] smooth_short_runs done: "
        f"actor={actor_name} iter={int(smooth_iter_count)} reason={stop_reason} "
        f"elapsed={time.perf_counter() - smooth_t0:.2f}s"
    )


def _repair_carla_transition_spikes(
    frames: List[Dict[str, object]],
    carla_context: Dict[str, object],
    enforce_node_direction: bool = False,
    opposite_reject_deg: float = 135.0,
) -> None:
    """
    Clamp abrupt one-frame/two-frame CARLA line switches that are inconsistent
    with tiny raw displacement, then quickly revert.
    """
    if len(frames) < 2:
        return

    max_passes = _env_int("V2X_CARLA_TRANSITION_REPAIR_PASSES", 2, minimum=1, maximum=5)
    raw_step_max = _env_float("V2X_CARLA_TRANSITION_REPAIR_RAW_STEP_MAX_M", 0.95)
    jump_abs = _env_float("V2X_CARLA_TRANSITION_REPAIR_JUMP_ABS_M", 1.35)
    jump_ratio = _env_float("V2X_CARLA_TRANSITION_REPAIR_JUMP_RATIO", 2.6)
    prev_dist_abs_cap = _env_float("V2X_CARLA_TRANSITION_REPAIR_PREV_DIST_ABS_M", 4.2)
    prev_dist_delta_cap = _env_float("V2X_CARLA_TRANSITION_REPAIR_PREV_DIST_DELTA_M", 1.2)
    prev_step_abs_cap = _env_float("V2X_CARLA_TRANSITION_REPAIR_PREV_STEP_ABS_M", 2.6)
    prev_step_ratio_cap = _env_float("V2X_CARLA_TRANSITION_REPAIR_PREV_STEP_RATIO", 2.8)
    next_step_abs_cap = _env_float("V2X_CARLA_TRANSITION_REPAIR_NEXT_STEP_ABS_M", 3.0)
    next_step_ratio_cap = _env_float("V2X_CARLA_TRANSITION_REPAIR_NEXT_STEP_RATIO", 3.2)

    n = len(frames)

    def _raw_step(i0: int, i1: int) -> float:
        if i0 < 0 or i1 < 0 or i0 >= n or i1 >= n or i0 == i1:
            return 0.0
        f0 = frames[i0]
        f1 = frames[i1]
        x0 = _safe_float(f0.get("x"), float("nan"))
        y0 = _safe_float(f0.get("y"), float("nan"))
        x1 = _safe_float(f1.get("x"), float("nan"))
        y1 = _safe_float(f1.get("y"), float("nan"))
        if math.isfinite(x0) and math.isfinite(y0) and math.isfinite(x1) and math.isfinite(y1):
            return float(math.hypot(float(x1) - float(x0), float(y1) - float(y0)))
        qx0, qy0, _ = _frame_query_pose(f0)
        qx1, qy1, _ = _frame_query_pose(f1)
        return float(math.hypot(float(qx1) - float(qx0), float(qy1) - float(qy0)))

    for _ in range(max(1, int(max_passes))):
        changed = False
        for i in range(1, n):
            prev = frames[i - 1]
            cur = frames[i]
            if not (_frame_has_carla_pose(prev) and _frame_has_carla_pose(cur)):
                continue
            if bool(prev.get("synthetic_turn", False)) or bool(cur.get("synthetic_turn", False)):
                continue
            prev_line = _safe_int(prev.get("ccli"), -1)
            cur_line = _safe_int(cur.get("ccli"), -1)
            if prev_line < 0 or cur_line < 0 or prev_line == cur_line:
                continue

            raw_step = float(_raw_step(i - 1, i))
            carla_step = float(
                math.hypot(
                    float(_safe_float(cur.get("cx"), 0.0)) - float(_safe_float(prev.get("cx"), 0.0)),
                    float(_safe_float(cur.get("cy"), 0.0)) - float(_safe_float(prev.get("cy"), 0.0)),
                )
            )
            has_aba = bool(
                (i + 1) < n
                and _frame_has_carla_pose(frames[i + 1])
                and _safe_int(frames[i + 1].get("ccli"), -1) == int(prev_line)
            )
            is_jump_like = bool(
                float(raw_step) <= float(raw_step_max)
                and float(carla_step) > max(float(jump_abs), float(jump_ratio) * max(0.2, float(raw_step)))
            )
            if not has_aba and not is_jump_like:
                continue

            qx, qy, qyaw = _frame_query_pose(cur)
            prev_proj = _best_projection_on_carla_line(
                carla_context=carla_context,
                line_index=int(prev_line),
                qx=float(qx),
                qy=float(qy),
                qyaw=float(qyaw),
                prefer_reversed=None,
                enforce_node_direction=bool(enforce_node_direction),
                opposite_reject_deg=float(opposite_reject_deg),
            )
            if prev_proj is None:
                prev_proj = _best_projection_on_carla_line(
                    carla_context=carla_context,
                    line_index=int(prev_line),
                    qx=float(qx),
                    qy=float(qy),
                    qyaw=float(qyaw),
                    prefer_reversed=None,
                    enforce_node_direction=False,
                    opposite_reject_deg=float(opposite_reject_deg),
                )
            if prev_proj is None:
                continue
            proj = prev_proj.get("projection", {})
            if not isinstance(proj, dict):
                continue
            px = _safe_float(proj.get("x"), float("nan"))
            py = _safe_float(proj.get("y"), float("nan"))
            pyaw = _safe_float(proj.get("yaw"), _safe_float(cur.get("cyaw"), qyaw))
            pdist = _safe_float(proj.get("dist"), float("inf"))
            if not (math.isfinite(px) and math.isfinite(py) and math.isfinite(pdist)):
                continue
            if float(pdist) > float(prev_dist_abs_cap):
                continue

            cur_dist = _safe_float(cur.get("cdist"), float("inf"))
            if math.isfinite(cur_dist) and float(pdist) > float(cur_dist) + float(prev_dist_delta_cap):
                continue

            prev_cx = _safe_float(prev.get("cx"), float("nan"))
            prev_cy = _safe_float(prev.get("cy"), float("nan"))
            if not (math.isfinite(prev_cx) and math.isfinite(prev_cy)):
                continue
            prev_step = float(math.hypot(float(px) - float(prev_cx), float(py) - float(prev_cy)))
            prev_step_lim = max(float(prev_step_abs_cap), float(prev_step_ratio_cap) * max(0.2, float(raw_step)))
            if float(prev_step) > float(prev_step_lim):
                continue

            if (i + 1) < n and _frame_has_carla_pose(frames[i + 1]):
                next_fr = frames[i + 1]
                next_raw = float(_raw_step(i, i + 1))
                nx = _safe_float(next_fr.get("cx"), float("nan"))
                ny = _safe_float(next_fr.get("cy"), float("nan"))
                if math.isfinite(nx) and math.isfinite(ny):
                    next_step = float(math.hypot(float(nx) - float(px), float(ny) - float(py)))
                    next_lim = max(
                        float(next_step_abs_cap),
                        float(next_step_ratio_cap) * max(0.2, float(next_raw)),
                    )
                    if float(next_step) > float(next_lim) and not has_aba:
                        continue

            _set_carla_pose(
                frame=cur,
                line_index=int(prev_line),
                x=float(px),
                y=float(py),
                yaw=float(pyaw),
                dist=float(pdist),
                source="transition_spike_hold_prev",
                quality=str(cur.get("cquality", "none")),
            )
            cur["transition_spike_repair"] = True
            changed = True

        if not changed:
            break


def _suppress_short_carla_switchbacks(
    frames: List[Dict[str, object]],
    carla_context: Dict[str, object],
    max_run_frames: int = 8,
    cost_slack_base: float = 0.8,
    cost_slack_per_frame: float = 0.45,
    max_point_delta: float = 2.2,
    max_point_dist: float = 10.0,
    enforce_node_direction: bool = False,
    opposite_reject_deg: float = 135.0,
) -> None:
    """
    Collapse short A->B->A CARLA switchback runs using per-frame reprojection
    onto the stable surrounding line when geometric fit remains comparable.
    """
    if len(frames) < 3:
        return

    max_passes = _env_int("V2X_CARLA_SWITCHBACK_PASSES", 2, minimum=1, maximum=5)

    def _build_runs() -> List[Tuple[int, int, int]]:
        runs: List[Tuple[int, int, int]] = []
        i = 0
        n = len(frames)
        while i < n:
            fr = frames[i]
            if not _frame_has_carla_pose(fr):
                i += 1
                continue
            li = _safe_int(fr.get("ccli"), -1)
            if li < 0:
                i += 1
                continue
            s = i
            while i + 1 < n:
                nxt = frames[i + 1]
                if not _frame_has_carla_pose(nxt):
                    break
                nli = _safe_int(nxt.get("ccli"), -1)
                if nli != li:
                    break
                i += 1
            runs.append((int(s), int(i), int(li)))
            i += 1
        return runs

    def _plan_replace(start: int, end: int, line_idx: int) -> Optional[Dict[str, object]]:
        rows: List[Tuple[int, Dict[str, object]]] = []
        old_cost = 0.0
        new_cost = 0.0
        old_max = 0.0
        new_max = 0.0
        for fi in range(int(start), int(end) + 1):
            fr = frames[fi]
            if not _frame_has_carla_pose(fr):
                return None
            qx, qy, qyaw = _frame_query_pose(fr)
            cand = _best_projection_on_carla_line(
                carla_context=carla_context,
                line_index=int(line_idx),
                qx=float(qx),
                qy=float(qy),
                qyaw=float(qyaw),
                prefer_reversed=None,
                enforce_node_direction=bool(enforce_node_direction),
                opposite_reject_deg=float(opposite_reject_deg),
            )
            if cand is None:
                cand = _best_projection_on_carla_line(
                    carla_context=carla_context,
                    line_index=int(line_idx),
                    qx=float(qx),
                    qy=float(qy),
                    qyaw=float(qyaw),
                    prefer_reversed=None,
                    enforce_node_direction=False,
                    opposite_reject_deg=float(opposite_reject_deg),
                )
            if cand is None:
                return None
            proj = cand.get("projection", {})
            if not isinstance(proj, dict):
                return None

            old_dist = _safe_float(fr.get("cdist"), float("nan"))
            if not math.isfinite(old_dist):
                ox = _safe_float(fr.get("cx"), float(qx))
                oy = _safe_float(fr.get("cy"), float(qy))
                old_dist = float(math.hypot(float(ox) - float(qx), float(oy) - float(qy)))
            new_dist = _safe_float(proj.get("dist"), float("inf"))
            if not math.isfinite(new_dist):
                return None
            old_cost += float(old_dist)
            new_cost += float(new_dist)
            old_max = max(float(old_max), float(old_dist))
            new_max = max(float(new_max), float(new_dist))
            rows.append((int(fi), cand))

        return {
            "rows": rows,
            "old_cost": float(old_cost),
            "new_cost": float(new_cost),
            "old_max": float(old_max),
            "new_max": float(new_max),
        }

    for _ in range(max(1, int(max_passes))):
        runs = _build_runs()
        if len(runs) < 3:
            break
        changed = False

        for ridx in range(1, len(runs) - 1):
            bs, be, bline = runs[ridx]
            as_, ae, aline = runs[ridx - 1]
            cs, ce, cline = runs[ridx + 1]
            run_len = int(be - bs + 1)
            if run_len <= 0 or run_len > int(max_run_frames):
                continue
            if int(aline) != int(cline) or int(bline) == int(aline):
                continue

            # Preserve true lane changes: require at least modest support on
            # both sides of the switchback.
            prev_len = int(ae - as_ + 1)
            next_len = int(ce - cs + 1)
            if prev_len < 2 or next_len < 2:
                continue

            plan = _plan_replace(int(bs), int(be), int(aline))
            if not isinstance(plan, dict):
                continue
            old_cost = float(plan.get("old_cost", 0.0))
            new_cost = float(plan.get("new_cost", 0.0))
            old_max = float(plan.get("old_max", 0.0))
            new_max = float(plan.get("new_max", 0.0))
            local_slack = float(cost_slack_base) + float(cost_slack_per_frame) * float(run_len)
            if new_cost > old_cost + float(local_slack):
                continue
            if new_max > max(float(max_point_dist), old_max + float(max_point_delta)):
                continue

            rows = plan.get("rows", [])
            if not isinstance(rows, list) or not rows:
                continue
            for fi, cand in rows:
                proj = cand["projection"]
                fr = frames[int(fi)]
                _set_carla_pose(
                    frame=fr,
                    line_index=int(aline),
                    x=float(proj["x"]),
                    y=float(proj["y"]),
                    yaw=float(proj["yaw"]),
                    dist=float(proj["dist"]),
                    source="short_switchback_hold",
                    quality=str(fr.get("cquality", "none")),
                )
                fr["switchback_suppressed"] = True
            changed = True
            break

        if not changed:
            break


def _soften_small_step_transition_jumps(
    frames: List[Dict[str, object]],
    jump_threshold_m: float = 1.5,
    jump_ratio_vs_raw: float = 2.2,
    max_shift_m: float = 1.2,
    max_query_cost_delta: float = 1.0,
) -> None:
    """
    For short-step line transitions, shrink abrupt CARLA point-to-point jumps
    by nudging both boundary points toward their midpoint.
    """
    if len(frames) < 2:
        return

    n = len(frames)
    teleport_raw_step_m = _env_float("V2X_CARLA_SMALL_TRANSITION_TELEPORT_RAW_STEP_M", 0.6)
    teleport_snap_step_m = _env_float("V2X_CARLA_SMALL_TRANSITION_TELEPORT_SNAP_STEP_M", 1.8)

    def _raw_step(i0: int, i1: int) -> float:
        if i0 < 0 or i1 < 0 or i0 >= n or i1 >= n or i0 == i1:
            return 0.0
        f0 = frames[i0]
        f1 = frames[i1]
        x0 = _safe_float(f0.get("x"), float("nan"))
        y0 = _safe_float(f0.get("y"), float("nan"))
        x1 = _safe_float(f1.get("x"), float("nan"))
        y1 = _safe_float(f1.get("y"), float("nan"))
        if math.isfinite(x0) and math.isfinite(y0) and math.isfinite(x1) and math.isfinite(y1):
            return float(math.hypot(float(x1) - float(x0), float(y1) - float(y0)))
        qx0, qy0, _ = _frame_query_pose(f0)
        qx1, qy1, _ = _frame_query_pose(f1)
        return float(math.hypot(float(qx1) - float(qx0), float(qy1) - float(qy0)))

    def _carla_step_with_overrides(
        step_idx: int,
        overrides: Dict[int, Tuple[float, float]],
    ) -> Optional[float]:
        if step_idx <= 0 or step_idx >= n:
            return None
        a = frames[step_idx - 1]
        b = frames[step_idx]
        if not (_frame_has_carla_pose(a) and _frame_has_carla_pose(b)):
            return None
        ax, ay = overrides.get(
            int(step_idx - 1),
            (float(_safe_float(a.get("cx"), 0.0)), float(_safe_float(a.get("cy"), 0.0))),
        )
        bx, by = overrides.get(
            int(step_idx),
            (float(_safe_float(b.get("cx"), 0.0)), float(_safe_float(b.get("cy"), 0.0))),
        )
        return float(math.hypot(float(bx) - float(ax), float(by) - float(ay)))

    for i in range(1, n):
        a = frames[i - 1]
        b = frames[i]
        if not (_frame_has_carla_pose(a) and _frame_has_carla_pose(b)):
            continue
        if bool(a.get("synthetic_turn", False)) or bool(b.get("synthetic_turn", False)):
            continue

        al = _safe_int(a.get("ccli"), -1)
        bl = _safe_int(b.get("ccli"), -1)
        if al < 0 or bl < 0 or al == bl:
            continue

        raw_step = float(_raw_step(i - 1, i))
        carla_step = float(
            math.hypot(
                float(_safe_float(b.get("cx"), 0.0)) - float(_safe_float(a.get("cx"), 0.0)),
                float(_safe_float(b.get("cy"), 0.0)) - float(_safe_float(a.get("cy"), 0.0)),
            )
        )
        if carla_step <= max(float(jump_threshold_m), float(jump_ratio_vs_raw) * max(0.2, float(raw_step))):
            continue

        cdist_a = _safe_float(a.get("cdist"), float("inf"))
        cdist_b = _safe_float(b.get("cdist"), float("inf"))
        if cdist_a > 6.0 or cdist_b > 6.0:
            continue

        ax = float(a["cx"])
        ay = float(a["cy"])
        bx = float(b["cx"])
        by = float(b["cy"])
        mx = 0.5 * (ax + bx)
        my = 0.5 * (ay + by)
        target = max(0.9, 1.8 * max(0.2, float(raw_step)))
        if carla_step <= target:
            continue
        scale = float(target) / max(1e-6, float(carla_step))
        lax = float(mx) + (float(ax) - float(mx)) * float(scale)
        lay = float(my) + (float(ay) - float(my)) * float(scale)
        lbx = float(mx) + (float(bx) - float(mx)) * float(scale)
        lby = float(my) + (float(by) - float(my)) * float(scale)

        shift_a = float(math.hypot(float(lax) - float(ax), float(lay) - float(ay)))
        shift_b = float(math.hypot(float(lbx) - float(bx), float(lby) - float(by)))
        if shift_a > float(max_shift_m):
            sa = float(max_shift_m) / max(1e-6, float(shift_a))
            lax = float(ax) + (float(lax) - float(ax)) * float(sa)
            lay = float(ay) + (float(lay) - float(ay)) * float(sa)
        if shift_b > float(max_shift_m):
            sb = float(max_shift_m) / max(1e-6, float(shift_b))
            lbx = float(bx) + (float(lbx) - float(bx)) * float(sb)
            lby = float(by) + (float(lby) - float(by)) * float(sb)

        qax, qay, _ = _frame_query_pose(a)
        qbx, qby, _ = _frame_query_pose(b)
        old_da = float(math.hypot(float(ax) - float(qax), float(ay) - float(qay)))
        old_db = float(math.hypot(float(bx) - float(qbx), float(by) - float(qby)))
        new_da = float(math.hypot(float(lax) - float(qax), float(lay) - float(qay)))
        new_db = float(math.hypot(float(lbx) - float(qbx), float(lby) - float(qby)))
        if new_da > old_da + float(max_query_cost_delta):
            continue
        if new_db > old_db + float(max_query_cost_delta):
            continue

        overrides = {
            int(i - 1): (float(lax), float(lay)),
            int(i): (float(lbx), float(lby)),
        }
        affected_steps = [idx for idx in (i - 1, i, i + 1) if 0 < idx < n]
        old_local_max = 0.0
        new_local_max = 0.0
        has_step = False
        creates_teleport = False
        for step_idx in affected_steps:
            old_step = _carla_step_with_overrides(int(step_idx), overrides={})
            new_step = _carla_step_with_overrides(int(step_idx), overrides=overrides)
            if old_step is None or new_step is None:
                continue
            has_step = True
            old_local_max = max(float(old_local_max), float(old_step))
            new_local_max = max(float(new_local_max), float(new_step))
            raw_step_local = float(_raw_step(int(step_idx - 1), int(step_idx)))
            if float(raw_step_local) < float(teleport_raw_step_m) and float(new_step) > float(teleport_snap_step_m):
                creates_teleport = True
                break
        if not bool(has_step):
            continue
        if bool(creates_teleport):
            continue
        if float(new_local_max) >= float(old_local_max) - 1e-6:
            continue

        a["cx"] = float(lax)
        a["cy"] = float(lay)
        b["cx"] = float(lbx)
        b["cy"] = float(lby)
        a["small_step_transition_soften"] = True
        b["small_step_transition_soften"] = True
        a["csource"] = "small_step_transition_soften"
        b["csource"] = "small_step_transition_soften"


def _stabilize_carla_yaw_at_transitions(
    frames: List[Dict[str, object]],
    max_low_step_yaw_jump_deg: float = 26.0,
) -> None:
    """
    Reduce abrupt yaw flips around CARLA line transitions while preserving
    frame timing and XY geometry.
    """
    if len(frames) < 2:
        return

    n = len(frames)

    def _carla_motion_yaw(i: int) -> Optional[float]:
        if i < 0 or i >= n:
            return None
        if 0 < i < (n - 1) and _frame_has_carla_pose(frames[i - 1]) and _frame_has_carla_pose(frames[i + 1]):
            dx = float(frames[i + 1]["cx"]) - float(frames[i - 1]["cx"])
            dy = float(frames[i + 1]["cy"]) - float(frames[i - 1]["cy"])
            if math.hypot(dx, dy) >= 0.30:
                return float(_normalize_yaw_deg(math.degrees(math.atan2(dy, dx))))
        if i > 0 and _frame_has_carla_pose(frames[i - 1]) and _frame_has_carla_pose(frames[i]):
            dx = float(frames[i]["cx"]) - float(frames[i - 1]["cx"])
            dy = float(frames[i]["cy"]) - float(frames[i - 1]["cy"])
            if math.hypot(dx, dy) >= 0.20:
                return float(_normalize_yaw_deg(math.degrees(math.atan2(dy, dx))))
        if i + 1 < n and _frame_has_carla_pose(frames[i]) and _frame_has_carla_pose(frames[i + 1]):
            dx = float(frames[i + 1]["cx"]) - float(frames[i]["cx"])
            dy = float(frames[i + 1]["cy"]) - float(frames[i]["cy"])
            if math.hypot(dx, dy) >= 0.20:
                return float(_normalize_yaw_deg(math.degrees(math.atan2(dy, dx))))
        return _estimate_query_motion_yaw(frames, i, max_span=5, min_disp_m=0.20)

    # First, align transition boundary yaws to motion direction.
    for i in range(1, n):
        a = frames[i - 1]
        b = frames[i]
        if not (_frame_has_carla_pose(a) and _frame_has_carla_pose(b)):
            continue
        al = _safe_int(a.get("ccli"), -1)
        bl = _safe_int(b.get("ccli"), -1)
        if al < 0 or bl < 0 or al == bl:
            continue
        for j in (i - 1, i):
            fr = frames[j]
            cyaw = _safe_float(fr.get("cyaw"), float("nan"))
            if not math.isfinite(cyaw):
                continue
            myaw = _carla_motion_yaw(j)
            if myaw is None:
                continue
            diff = float(_yaw_abs_diff_deg(float(cyaw), float(myaw)))
            if diff <= 12.0:
                continue
            alpha = 0.55 if diff < 70.0 else 0.78
            fr["cyaw"] = float(_normalize_yaw_deg(_interp_yaw_deg(float(cyaw), float(myaw), float(alpha))))
            fr["cyaw_transition_stabilized"] = True

    # Then clamp residual adjacent yaw jumps on low-motion steps.
    max_jump = float(max_low_step_yaw_jump_deg)
    for i in range(1, n):
        a = frames[i - 1]
        b = frames[i]
        if not (_frame_has_carla_pose(a) and _frame_has_carla_pose(b)):
            continue
        ay = _safe_float(a.get("cyaw"), float("nan"))
        by = _safe_float(b.get("cyaw"), float("nan"))
        if not (math.isfinite(ay) and math.isfinite(by)):
            continue
        step = float(
            math.hypot(
                float(_safe_float(b.get("cx"), 0.0)) - float(_safe_float(a.get("cx"), 0.0)),
                float(_safe_float(b.get("cy"), 0.0)) - float(_safe_float(a.get("cy"), 0.0)),
            )
        )
        if step > 1.25:
            continue
        jump = float(_yaw_abs_diff_deg(float(by), float(ay)))
        if jump <= float(max_jump):
            continue
        mid = float(_interp_yaw_deg(float(ay), float(by), 0.5))
        a["cyaw"] = float(_normalize_yaw_deg(_interp_yaw_deg(float(ay), float(mid), 0.70)))
        b["cyaw"] = float(_normalize_yaw_deg(_interp_yaw_deg(float(by), float(mid), 0.70)))
        a["cyaw_transition_stabilized"] = True
        b["cyaw_transition_stabilized"] = True


def _smooth_carla_transition_windows(
    frames: List[Dict[str, object]],
    window_radius: int = 2,
    max_shift_m: float = 0.9,
    max_query_cost_delta: float = 0.8,
    max_total_cost_delta: float = 1.2,
    passes: int = 2,
) -> None:
    """
    Locally smooth XY around CARLA line transitions to reduce curvature-rate
    spikes while keeping raw-query fit bounded.
    """
    if len(frames) < 5:
        return

    n = len(frames)
    rad = max(1, int(window_radius))
    teleport_raw_step_m = _env_float("V2X_CARLA_TRANSITION_WINDOW_TELEPORT_RAW_STEP_M", 0.6)
    teleport_snap_step_m = _env_float("V2X_CARLA_TRANSITION_WINDOW_TELEPORT_SNAP_STEP_M", 1.8)
    skip_semantic_transition = (
        _env_int("V2X_CARLA_TRANSITION_WINDOW_SKIP_SEMANTIC_TRANSITION", 1, minimum=0, maximum=1) == 1
    )
    changed_idx: set = set()

    def _raw_step(step_idx: int) -> float:
        if step_idx <= 0 or step_idx >= n:
            return 0.0
        a = frames[step_idx - 1]
        b = frames[step_idx]
        ax = _safe_float(a.get("x"), float("nan"))
        ay = _safe_float(a.get("y"), float("nan"))
        bx = _safe_float(b.get("x"), float("nan"))
        by = _safe_float(b.get("y"), float("nan"))
        if math.isfinite(ax) and math.isfinite(ay) and math.isfinite(bx) and math.isfinite(by):
            return float(math.hypot(float(bx) - float(ax), float(by) - float(ay)))
        qax, qay, _ = _frame_query_pose(a)
        qbx, qby, _ = _frame_query_pose(b)
        return float(math.hypot(float(qbx) - float(qax), float(qby) - float(qay)))

    def _carla_step(
        step_idx: int,
        overrides: Dict[int, Tuple[float, float]],
    ) -> Optional[float]:
        if step_idx <= 0 or step_idx >= n:
            return None
        a = frames[step_idx - 1]
        b = frames[step_idx]
        if not (_frame_has_carla_pose(a) and _frame_has_carla_pose(b)):
            return None
        ax, ay = overrides.get(
            int(step_idx - 1),
            (float(_safe_float(a.get("cx"), 0.0)), float(_safe_float(a.get("cy"), 0.0))),
        )
        bx, by = overrides.get(
            int(step_idx),
            (float(_safe_float(b.get("cx"), 0.0)), float(_safe_float(b.get("cy"), 0.0))),
        )
        return float(math.hypot(float(bx) - float(ax), float(by) - float(ay)))

    for i in range(1, n):
        if not (_frame_has_carla_pose(frames[i - 1]) and _frame_has_carla_pose(frames[i])):
            continue
        li_prev = _safe_int(frames[i - 1].get("ccli"), -1)
        li_cur = _safe_int(frames[i].get("ccli"), -1)
        if li_prev < 0 or li_cur < 0 or li_prev == li_cur:
            continue
        if bool(frames[i - 1].get("synthetic_turn", False)) or bool(frames[i].get("synthetic_turn", False)):
            continue
        if bool(skip_semantic_transition):
            sk_prev = _frame_semantic_lane_key(frames[i - 1])
            sk_cur = _frame_semantic_lane_key(frames[i])
            if sk_prev is not None and sk_cur is not None and int(sk_prev) != int(sk_cur):
                continue

        s = max(0, int(i) - rad)
        e = min(n - 1, int(i) + rad)
        if (e - s + 1) < 4:
            continue
        if any(not _frame_has_carla_pose(frames[j]) for j in range(s, e + 1)):
            continue

        # Preserve boundary anchors; smooth only interior points.
        pts = [(float(frames[j]["cx"]), float(frames[j]["cy"])) for j in range(s, e + 1)]
        cand = list(pts)
        for _ in range(max(1, int(passes))):
            nxt = list(cand)
            for k in range(1, len(cand) - 1):
                ax, ay = cand[k - 1]
                bx, by = cand[k]
                cx, cy = cand[k + 1]
                nx = 0.25 * ax + 0.50 * bx + 0.25 * cx
                ny = 0.25 * ay + 0.50 * by + 0.25 * cy
                nxt[k] = (float(nx), float(ny))
            cand = nxt

        old_cost = 0.0
        new_cost = 0.0
        feasible = True
        staged: List[Tuple[int, float, float, float]] = []
        for off, j in enumerate(range(s, e + 1)):
            if off == 0 or off == (len(cand) - 1):
                continue
            fr = frames[j]
            ox = float(fr["cx"])
            oy = float(fr["cy"])
            nx, ny = cand[off]
            shift = float(math.hypot(float(nx) - float(ox), float(ny) - float(oy)))
            if shift > float(max_shift_m):
                scale = float(max_shift_m) / max(1e-6, float(shift))
                nx = float(ox) + (float(nx) - float(ox)) * float(scale)
                ny = float(oy) + (float(ny) - float(oy)) * float(scale)
            qx, qy, _ = _frame_query_pose(fr)
            old_d = float(math.hypot(float(ox) - float(qx), float(oy) - float(qy)))
            new_d = float(math.hypot(float(nx) - float(qx), float(ny) - float(qy)))
            if new_d > old_d + float(max_query_cost_delta):
                feasible = False
                break
            old_cost += float(old_d)
            new_cost += float(new_d)
            staged.append((int(j), float(nx), float(ny), float(new_d)))

        if not feasible:
            continue
        if new_cost > old_cost + float(max_total_cost_delta):
            continue

        overrides = {int(j): (float(nx), float(ny)) for j, nx, ny, _ in staged}
        step_start = max(1, int(s))
        step_end = min(int(n) - 1, int(e))
        old_local_max = 0.0
        new_local_max = 0.0
        has_step = False
        creates_teleport = False
        for step_idx in range(int(step_start), int(step_end) + 1):
            old_step = _carla_step(int(step_idx), overrides={})
            new_step = _carla_step(int(step_idx), overrides=overrides)
            if old_step is None or new_step is None:
                continue
            has_step = True
            old_local_max = max(float(old_local_max), float(old_step))
            new_local_max = max(float(new_local_max), float(new_step))
            raw_step_local = float(_raw_step(int(step_idx)))
            if float(raw_step_local) < float(teleport_raw_step_m) and float(new_step) > float(teleport_snap_step_m):
                creates_teleport = True
                break
        if not bool(has_step):
            continue
        if bool(creates_teleport):
            continue
        if float(new_local_max) >= float(old_local_max) - 1e-6:
            continue

        for j, nx, ny, nd in staged:
            fr = frames[j]
            fr["cx"] = float(nx)
            fr["cy"] = float(ny)
            fr["cdist"] = float(nd)
            fr["csource"] = "transition_window_smooth"
            fr["transition_window_smooth"] = True
            changed_idx.add(int(j))

    if not changed_idx:
        return

    # Recompute local yaw from neighbors after positional smoothing.
    for j in sorted(changed_idx):
        if not _frame_has_carla_pose(frames[j]):
            continue
        prev_ok = j > 0 and _frame_has_carla_pose(frames[j - 1])
        next_ok = (j + 1) < n and _frame_has_carla_pose(frames[j + 1])
        if not prev_ok and not next_ok:
            continue
        if prev_ok and next_ok:
            dx = float(frames[j + 1]["cx"]) - float(frames[j - 1]["cx"])
            dy = float(frames[j + 1]["cy"]) - float(frames[j - 1]["cy"])
        elif prev_ok:
            dx = float(frames[j]["cx"]) - float(frames[j - 1]["cx"])
            dy = float(frames[j]["cy"]) - float(frames[j - 1]["cy"])
        else:
            dx = float(frames[j + 1]["cx"]) - float(frames[j]["cx"])
            dy = float(frames[j + 1]["cy"]) - float(frames[j]["cy"])
        if math.hypot(dx, dy) <= 1e-6:
            continue
        frames[j]["cyaw"] = float(_normalize_yaw_deg(math.degrees(math.atan2(dy, dx))))


def _consolidate_track_two_segment(
    frames: List[Dict[str, object]],
    carla_context: Optional[Dict[str, object]] = None,
    max_dist_threshold_m: float = 3.0,
    min_segment_frames: int = 6,
) -> bool:
    """Try to split the track into 2 contiguous segments where each is
    covered by a single CARLA lane. Then re-snap each segment to its line.

    Used when single-line consolidation fails: catches "go straight then turn"
    or "turn then go straight" tracks where one line covers half but not all.
    The split point is found via grid search.
    """
    if not carla_context or not bool(carla_context.get("enabled", False)):
        return False
    n = len(frames)
    if n < (2 * int(min_segment_frames) + 2):
        return False
    carla_feats = carla_context.get("carla_feats", {})
    if not isinstance(carla_feats, dict) or not carla_feats:
        return False

    raw_xy: List[Tuple[float, float, int]] = []
    for i, fr in enumerate(frames):
        x = _safe_float(fr.get("x"), float("nan"))
        y = _safe_float(fr.get("y"), float("nan"))
        if math.isfinite(x) and math.isfinite(y):
            raw_xy.append((float(x), float(y), int(i)))
    if len(raw_xy) < (2 * int(min_segment_frames) + 2):
        return False

    # Candidate lines: union of CCLIs already chosen + bbox-overlap.
    seen_clis: set = set()
    for fr in frames:
        cli = _safe_int(fr.get("ccli"), -1)
        if cli >= 0:
            seen_clis.add(int(cli))
    candidates: set = set(seen_clis)
    xs_all = [p[0] for p in raw_xy]
    ys_all = [p[1] for p in raw_xy]
    bx0, bx1 = min(xs_all) - 6.0, max(xs_all) + 6.0
    by0, by1 = min(ys_all) - 6.0, max(ys_all) + 6.0
    for li, cf in carla_feats.items():
        if not isinstance(cf, dict):
            continue
        poly = cf.get("poly_xy")
        if not isinstance(poly, np.ndarray) or poly.shape[0] < 2:
            continue
        if poly[:, 0].max() < bx0 or poly[:, 0].min() > bx1:
            continue
        if poly[:, 1].max() < by0 or poly[:, 1].min() > by1:
            continue
        candidates.add(int(li))
    if len(candidates) < 2:
        return False

    # Pre-compute distance from every raw point to every candidate line.
    dist_table: Dict[int, np.ndarray] = {}
    for li in candidates:
        cf = carla_feats.get(int(li))
        if not isinstance(cf, dict):
            continue
        poly = cf.get("poly_xy")
        if not isinstance(poly, np.ndarray) or poly.shape[0] < 2:
            continue
        seg_x0 = poly[:-1, 0]; seg_y0 = poly[:-1, 1]
        seg_x1 = poly[1:, 0];  seg_y1 = poly[1:, 1]
        sdx = seg_x1 - seg_x0
        sdy = seg_y1 - seg_y0
        L2 = sdx * sdx + sdy * sdy
        denom = np.where(L2 > 1e-9, L2, 1.0)
        dvec = np.zeros(len(raw_xy), dtype=np.float64)
        for i, (x, y, _idx) in enumerate(raw_xy):
            t = ((x - seg_x0) * sdx + (y - seg_y0) * sdy) / denom
            t = np.clip(t, 0.0, 1.0)
            cx_seg = seg_x0 + t * sdx
            cy_seg = seg_y0 + t * sdy
            dvec[i] = float(np.min(np.hypot(cx_seg - x, cy_seg - y)))
        dist_table[int(li)] = dvec

    if len(dist_table) < 2:
        return False

    # For each line, compute prefix-max distance and suffix-max distance.
    # Then for each split index k, find the best (line_left, line_right) pair.
    n_raw = len(raw_xy)
    prefix_max: Dict[int, np.ndarray] = {}
    suffix_max: Dict[int, np.ndarray] = {}
    for li, dvec in dist_table.items():
        pm = np.zeros(n_raw); sm = np.zeros(n_raw)
        cur = 0.0
        for i in range(n_raw):
            if dvec[i] > cur: cur = dvec[i]
            pm[i] = cur
        cur = 0.0
        for i in range(n_raw - 1, -1, -1):
            if dvec[i] > cur: cur = dvec[i]
            sm[i] = cur
        prefix_max[li] = pm
        suffix_max[li] = sm

    # Best split: minimize max(prefix_max[L][k-1], suffix_max[R][k]) over k, L≠R.
    best = (float("inf"), -1, -1, -1)  # (worst_max, k, L, R) where k is the split index in raw_xy
    li_list = list(dist_table.keys())
    min_seg_raw = max(2, int(min_segment_frames))
    for k in range(min_seg_raw, n_raw - min_seg_raw + 1):
        # Best line for left half = argmin prefix_max at k-1
        best_left_li = min(li_list, key=lambda li: float(prefix_max[li][k - 1]))
        best_left_d = float(prefix_max[best_left_li][k - 1])
        # Best line for right half = argmin suffix_max at k
        best_right_li = min(li_list, key=lambda li: float(suffix_max[li][k]))
        best_right_d = float(suffix_max[best_right_li][k])
        if best_left_li == best_right_li:
            # Same line → it's the single-lane case (already tried).
            continue
        worst = max(best_left_d, best_right_d)
        if worst < best[0]:
            best = (worst, k, best_left_li, best_right_li)

    if best[1] < 0 or best[0] > float(max_dist_threshold_m):
        return False
    _, split_k, left_li, right_li = best

    # Translate raw_xy index to frames index.
    left_frame_idx = set(raw_xy[i][2] for i in range(0, split_k))
    right_frame_idx = set(raw_xy[i][2] for i in range(split_k, n_raw))

    # Re-project each frame onto its assigned line.
    def _reproject(fi: int, target_li: int):
        cf_t = carla_feats.get(int(target_li))
        if not isinstance(cf_t, dict):
            return
        poly = cf_t.get("poly_xy")
        if not isinstance(poly, np.ndarray) or poly.shape[0] < 2:
            return
        fr = frames[int(fi)]
        x = _safe_float(fr.get("x"), float("nan"))
        y = _safe_float(fr.get("y"), float("nan"))
        if not (math.isfinite(x) and math.isfinite(y)):
            return
        seg_x0 = poly[:-1, 0]; seg_y0 = poly[:-1, 1]
        seg_x1 = poly[1:, 0];  seg_y1 = poly[1:, 1]
        sdx = seg_x1 - seg_x0
        sdy = seg_y1 - seg_y0
        L2 = sdx * sdx + sdy * sdy
        denom = np.where(L2 > 1e-9, L2, 1.0)
        t = ((x - seg_x0) * sdx + (y - seg_y0) * sdy) / denom
        t = np.clip(t, 0.0, 1.0)
        cx_seg = seg_x0 + t * sdx
        cy_seg = seg_y0 + t * sdy
        d_arr = np.hypot(cx_seg - x, cy_seg - y)
        k_idx = int(np.argmin(d_arr))
        npx = float(cx_seg[k_idx]); npy = float(cy_seg[k_idx])
        seg_dx = float(sdx[k_idx]); seg_dy = float(sdy[k_idx])
        if math.hypot(seg_dx, seg_dy) > 1e-6:
            new_yaw = math.degrees(math.atan2(seg_dy, seg_dx))
        else:
            new_yaw = _safe_float(fr.get("cyaw"), _safe_float(fr.get("yaw"), 0.0))
        fr["cx"] = npx
        fr["cy"] = npy
        fr["cyaw"] = float(_normalize_yaw_deg(new_yaw))
        fr["ccli"] = int(target_li)
        fr["csource"] = "two_lane_consolidate"
        fr["two_lane_consolidate"] = True

    for fi in range(len(frames)):
        if fi in left_frame_idx:
            _reproject(fi, left_li)
        elif fi in right_frame_idx:
            _reproject(fi, right_li)
    return True


def _consolidate_track_to_single_lane(
    frames: List[Dict[str, object]],
    carla_context: Optional[Dict[str, object]] = None,
    max_dist_threshold_m: float = 2.5,
    min_improvement_m: float = 0.5,
    consider_top_k: int = 16,
    min_frames: int = 8,
) -> Tuple[bool, int]:
    """If the entire track's RAW trajectory fits a single CARLA lane within
    `max_dist_threshold_m` AND that lane is at least `min_improvement_m` better
    than the worst per-frame snap fit, RE-SNAP every frame to that lane.

    Eliminates lane-bouncing during turns: a vehicle making a left turn
    might snap to several straight lanes consecutively when there's a single
    turn-lane polyline that perfectly covers the entire path. We detect that
    case and lock the track to the turn lane.

    Conservative: only fires when one lane covers the FULL track within the
    tight threshold (default 2.5 m, less than typical lane width). Returns
    (was_consolidated, line_index).
    """
    _dbg = os.environ.get("V2X_CARLA_SINGLE_LANE_CONSOLIDATE_VERBOSE_DEEP", "") == "1"
    if not carla_context or not bool(carla_context.get("enabled", False)):
        if _dbg: print(f"  [CONSOL-EARLY] no carla_context", flush=True)
        return (False, -1)
    n = len(frames)
    if n < int(min_frames):
        if _dbg: print(f"  [CONSOL-EARLY] n={n} < min_frames={min_frames}", flush=True)
        return (False, -1)
    carla_feats = carla_context.get("carla_feats", {})
    if not isinstance(carla_feats, dict) or not carla_feats:
        if _dbg: print(f"  [CONSOL-EARLY] no carla_feats", flush=True)
        return (False, -1)

    # Raw positions per frame (skip frames missing raw).
    raw_xy: List[Tuple[float, float, int]] = []
    for i, fr in enumerate(frames):
        x = _safe_float(fr.get("x"), float("nan"))
        y = _safe_float(fr.get("y"), float("nan"))
        if math.isfinite(x) and math.isfinite(y):
            raw_xy.append((float(x), float(y), int(i)))
    if len(raw_xy) < int(min_frames):
        if _dbg: print(f"  [CONSOL-EARLY] raw_xy={len(raw_xy)} < min_frames={min_frames}", flush=True)
        return (False, -1)

    # Candidate lines: union of CCLIs the snapper already chose, plus their
    # near-neighbors (we want to consider lines that the snapper *didn't*
    # pick but might still be a better single-line fit).
    seen_clis: set = set()
    for fr in frames:
        cli = _safe_int(fr.get("ccli"), -1)
        if cli >= 0:
            seen_clis.add(int(cli))
    candidates: set = set(seen_clis)
    # Also any line that has at least one waypoint in the trajectory's bbox.
    if raw_xy:
        xs = [p[0] for p in raw_xy]; ys = [p[1] for p in raw_xy]
        bx0, bx1 = min(xs) - 6.0, max(xs) + 6.0
        by0, by1 = min(ys) - 6.0, max(ys) + 6.0
        for li, cf in carla_feats.items():
            if not isinstance(cf, dict):
                continue
            poly = cf.get("poly_xy")
            if not isinstance(poly, np.ndarray) or poly.shape[0] < 2:
                continue
            # quick bbox check — any segment overlap?
            poly_xs = poly[:, 0]
            poly_ys = poly[:, 1]
            if poly_xs.max() < bx0 or poly_xs.min() > bx1:
                continue
            if poly_ys.max() < by0 or poly_ys.min() > by1:
                continue
            candidates.add(int(li))
    if not candidates:
        if _dbg: print(f"  [CONSOL-EARLY] no candidates", flush=True)
        return (False, -1)

    # For each candidate line, compute max + mean dist over all raw frames.
    def _line_fit(li: int) -> Optional[Tuple[float, float]]:
        cf = carla_feats.get(int(li))
        if not isinstance(cf, dict):
            return None
        poly = cf.get("poly_xy")
        if not isinstance(poly, np.ndarray) or poly.shape[0] < 2:
            return None
        max_d = 0.0
        sum_d = 0.0
        for x, y, _idx in raw_xy:
            # Cheap point-to-polyline min distance (numpy vectorized).
            seg_x0 = poly[:-1, 0]; seg_y0 = poly[:-1, 1]
            seg_x1 = poly[1:, 0];  seg_y1 = poly[1:, 1]
            dx = seg_x1 - seg_x0
            dy = seg_y1 - seg_y0
            L2 = dx * dx + dy * dy
            denom = np.where(L2 > 1e-9, L2, 1.0)
            t = ((x - seg_x0) * dx + (y - seg_y0) * dy) / denom
            t = np.clip(t, 0.0, 1.0)
            cx = seg_x0 + t * dx
            cy = seg_y0 + t * dy
            d = float(np.min(np.hypot(cx - x, cy - y)))
            if d > max_d:
                max_d = d
            sum_d += d
        return (max_d, sum_d / float(len(raw_xy)))

    best_li = -1
    best_max = float("inf")
    best_mean = float("inf")
    for li in candidates:
        fit = _line_fit(int(li))
        if fit is None:
            continue
        mx, mn = fit
        if mx < best_max - 1e-6:
            best_max = mx
            best_mean = mn
            best_li = int(li)

    if best_li < 0 or best_max > float(max_dist_threshold_m):
        if os.environ.get("V2X_CARLA_SINGLE_LANE_CONSOLIDATE_VERBOSE_DEEP", "") == "1":
            print(f"  [CONSOL-DBG] best_li={best_li} best_max={best_max:.2f}m thresh={max_dist_threshold_m:.2f}m candidates={len(candidates)} seen={sorted(seen_clis)}", flush=True)
        return (False, -1)

    # The per-frame snap is always close to whichever line the snapper picked
    # (that's how snapping works). The benefit of consolidation isn't lower
    # per-frame distance — it's eliminating discontinuous JUMPS between
    # adjacent lines. So instead of comparing distances, we require:
    #   (a) The track currently uses MULTIPLE distinct CCLIs (ccli runs > 1)
    #   (b) The single-line fit is tight (best_max < threshold) — already checked above
    # Skip consolidation for tracks already on a single CCLI: nothing to fix.
    distinct_clis = set()
    for fr in frames:
        cli = _safe_int(fr.get("ccli"), -1)
        if cli >= 0:
            distinct_clis.add(int(cli))
    if len(distinct_clis) <= 1:
        if _dbg: print(f"  [CONSOL-NOFIX] only {len(distinct_clis)} distinct CCLI(s) — nothing to consolidate", flush=True)
        return (False, -1)
    # Also skip if the chosen consolidate-line equals the only/dominant
    # current ccli (the snapper already locked onto it for the most part).
    if best_li in distinct_clis:
        # count how many frames are already on best_li
        on_best = sum(1 for fr in frames if _safe_int(fr.get("ccli"), -1) == int(best_li))
        if on_best >= int(0.9 * len(frames)):
            if _dbg: print(f"  [CONSOL-DOMINANT] {on_best}/{len(frames)} frames already on best_li={best_li}", flush=True)
            return (False, -1)
    if _dbg: print(f"  [CONSOL-FIRE] best_li={best_li} best_max={best_max:.2f}m runs_in={len(distinct_clis)} CCLIs", flush=True)

    # Re-project every frame onto best_li by direct geometric projection
    # onto the polyline. Don't go through `_best_projection_on_carla_line`
    # which has orientation/direction guards that can reject valid points
    # (e.g., when the line's heading at one end opposes the vehicle motion
    # in a turn lane). We trust the upstream max-fit check.
    cf_best = carla_feats.get(int(best_li))
    poly = cf_best.get("poly_xy") if isinstance(cf_best, dict) else None
    if not isinstance(poly, np.ndarray) or poly.shape[0] < 2:
        if _dbg: print(f"  [CONSOL-NOPOLY] best_li={best_li} has no poly", flush=True)
        return (False, int(best_li))
    seg_x0 = poly[:-1, 0]; seg_y0 = poly[:-1, 1]
    seg_x1 = poly[1:, 0];  seg_y1 = poly[1:, 1]
    sdx = seg_x1 - seg_x0
    sdy = seg_y1 - seg_y0
    L2 = sdx * sdx + sdy * sdy
    denom = np.where(L2 > 1e-9, L2, 1.0)
    n_modified = 0
    for fi, fr in enumerate(frames):
        x = _safe_float(fr.get("x"), float("nan"))
        y = _safe_float(fr.get("y"), float("nan"))
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        t_arr = ((x - seg_x0) * sdx + (y - seg_y0) * sdy) / denom
        t_arr = np.clip(t_arr, 0.0, 1.0)
        cx_arr = seg_x0 + t_arr * sdx
        cy_arr = seg_y0 + t_arr * sdy
        d_arr = np.hypot(cx_arr - x, cy_arr - y)
        k = int(np.argmin(d_arr))
        npx = float(cx_arr[k])
        npy = float(cy_arr[k])
        # Heading from this segment direction
        seg_dx = float(sdx[k])
        seg_dy = float(sdy[k])
        if math.hypot(seg_dx, seg_dy) > 1e-6:
            new_yaw = math.degrees(math.atan2(seg_dy, seg_dx))
        else:
            new_yaw = _safe_float(fr.get("cyaw"), _safe_float(fr.get("yaw"), 0.0))
        fr["cx"] = npx
        fr["cy"] = npy
        fr["cyaw"] = float(_normalize_yaw_deg(new_yaw))
        fr["ccli"] = int(best_li)
        fr["csource"] = "single_lane_consolidate"
        fr["single_lane_consolidate"] = True
        n_modified += 1
    if _dbg: print(f"  [CONSOL-DONE] modified {n_modified} frames to line={best_li}", flush=True)
    return (n_modified > 0, int(best_li))


def _shape_preserving_blend(
    frames: List[Dict[str, object]],
    s: int,
    e: int,
    source_label: str,
) -> int:
    """Replace (cx, cy) for interior frames in [s+1 .. e-1] with a blend that
    preserves raw trajectory shape.

    The snap correction `(cx - x, cy - y)` is smoothstep-interpolated between
    the anchor frames (s and e). Each interior frame's smoothed position is
    its RAW (x, y) plus the interpolated correction — so curving raw
    trajectories stay curving; only the lane-snap offset transitions
    smoothly. This avoids the bug where straight-line anchor-to-anchor
    blending replaces curving turns with straight lines.

    Returns number of frames modified. Mutates the frames in place.
    """
    if e <= s + 1:
        return 0
    n = len(frames)
    if s < 0 or e >= n:
        return 0
    fa = frames[s]
    fb = frames[e]
    if not (_frame_has_carla_pose(fa) and _frame_has_carla_pose(fb)):
        return 0
    cx_s = _safe_float(fa.get("cx"), float("nan"))
    cy_s = _safe_float(fa.get("cy"), float("nan"))
    cx_e = _safe_float(fb.get("cx"), float("nan"))
    cy_e = _safe_float(fb.get("cy"), float("nan"))
    if not all(math.isfinite(v) for v in (cx_s, cy_s, cx_e, cy_e)):
        return 0

    # Anchor's raw position. Fallback to snap if raw is missing — that turns
    # this back into the simple straight-line blend, which is at least
    # consistent at the anchor itself.
    x_s = _safe_float(fa.get("x"), cx_s)
    y_s = _safe_float(fa.get("y"), cy_s)
    x_e = _safe_float(fb.get("x"), cx_e)
    y_e = _safe_float(fb.get("y"), cy_e)
    corr_s_x = float(cx_s) - float(x_s)
    corr_s_y = float(cy_s) - float(y_s)
    corr_e_x = float(cx_e) - float(x_e)
    corr_e_y = float(cy_e) - float(y_e)

    span = e - s
    n_modified = 0
    for j in range(s + 1, e):
        fr = frames[j]
        x_j = _safe_float(fr.get("x"), float("nan"))
        y_j = _safe_float(fr.get("y"), float("nan"))
        if not (math.isfinite(x_j) and math.isfinite(y_j)):
            continue
        t = float(j - s) / float(span)
        # Smootherstep, C2-continuous.
        w = 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3
        corr_x = (1.0 - w) * corr_s_x + w * corr_e_x
        corr_y = (1.0 - w) * corr_s_y + w * corr_e_y
        # Smoothed snap = raw + smoothly-interpolated snap correction.
        # Raw curvature is preserved automatically.
        fr["cx"] = float(x_j) + float(corr_x)
        fr["cy"] = float(y_j) + float(corr_y)
        fr["csource"] = source_label
        fr[source_label] = True
        n_modified += 1
    return n_modified


def _recompute_yaw_from_neighbors(
    frames: List[Dict[str, object]],
    indices,
) -> None:
    """Recompute cyaw from neighboring (cx, cy) deltas after positional smoothing."""
    n = len(frames)
    for j in sorted(set(int(i) for i in indices)):
        if j < 0 or j >= n:
            continue
        if not _frame_has_carla_pose(frames[j]):
            continue
        prev_ok = j > 0 and _frame_has_carla_pose(frames[j - 1])
        next_ok = (j + 1) < n and _frame_has_carla_pose(frames[j + 1])
        if not prev_ok and not next_ok:
            continue
        if prev_ok and next_ok:
            dx = float(frames[j + 1]["cx"]) - float(frames[j - 1]["cx"])
            dy = float(frames[j + 1]["cy"]) - float(frames[j - 1]["cy"])
        elif prev_ok:
            dx = float(frames[j]["cx"]) - float(frames[j - 1]["cx"])
            dy = float(frames[j]["cy"]) - float(frames[j - 1]["cy"])
        else:
            dx = float(frames[j + 1]["cx"]) - float(frames[j]["cx"])
            dy = float(frames[j + 1]["cy"]) - float(frames[j]["cy"])
        if math.hypot(dx, dy) <= 1e-6:
            continue
        frames[j]["cyaw"] = float(_normalize_yaw_deg(math.degrees(math.atan2(dy, dx))))


def _smooth_fine_universal(
    frames: List[Dict[str, object]],
    sigma_frames: float = 0.7,
    radius_frames: int = 2,
    max_drift_m: float = 0.4,
    preserve_anchors: int = 1,
) -> int:
    """Final fine-grain Gaussian on (cx, cy) across ALL frames (no run gating).

    Catches the sub-meter / sub-frame jaggies that survive after the
    correction-jitter smoother, including the angular kinks at intersection
    turns where successive connector segments meet at slightly different
    headings. Sigma=0.7 frames is essentially a 3-tap symmetric kernel so
    legitimate motion (curves, lane changes, real turns) is preserved within
    the `max_drift_m` cap.

    `preserve_anchors` keeps the first/last N frames untouched so we don't
    rotate the trajectory at start/end where neighbours are missing.
    """
    n = len(frames)
    if n < (2 * radius_frames + 2 * preserve_anchors + 1):
        return 0

    # Gaussian kernel (1D, normalized).
    weights: List[float] = []
    for k in range(-int(radius_frames), int(radius_frames) + 1):
        weights.append(math.exp(-0.5 * (float(k) / float(sigma_frames)) ** 2))
    wsum = float(sum(weights))
    weights = [float(w) / wsum for w in weights]

    # Read sequence.
    cx_seq = [_safe_float(frames[i].get("cx"), float("nan")) for i in range(n)]
    cy_seq = [_safe_float(frames[i].get("cy"), float("nan")) for i in range(n)]

    n_modified = 0
    rad = int(radius_frames)
    pa = int(preserve_anchors)
    for j in range(pa, n - pa):
        # Center frame must have a valid pose.
        if not (math.isfinite(cx_seq[j]) and math.isfinite(cy_seq[j])):
            continue
        sx = sy = 0.0
        wnorm = 0.0
        for k_off, w_k in enumerate(weights):
            idx = j + k_off - rad
            if idx < 0 or idx >= n:
                continue
            cxk = cx_seq[idx]
            cyk = cy_seq[idx]
            if not (math.isfinite(cxk) and math.isfinite(cyk)):
                continue
            sx += float(w_k) * float(cxk)
            sy += float(w_k) * float(cyk)
            wnorm += float(w_k)
        if wnorm <= 1e-6:
            continue
        new_cx = sx / wnorm
        new_cy = sy / wnorm
        # Drift cap: don't move a point further than max_drift_m from its
        # original snap position. Protects legitimate turns/maneuvers.
        drift = math.hypot(new_cx - cx_seq[j], new_cy - cy_seq[j])
        if drift > float(max_drift_m):
            scale = float(max_drift_m) / max(1e-6, drift)
            new_cx = float(cx_seq[j]) + (new_cx - float(cx_seq[j])) * scale
            new_cy = float(cy_seq[j]) + (new_cy - float(cy_seq[j])) * scale
        frames[j]["cx"] = float(new_cx)
        frames[j]["cy"] = float(new_cy)
        # Don't overwrite csource here — it's a tiny per-frame nudge, not a
        # qualitative source change. Just leave whatever was there.
        n_modified += 1
    return n_modified


def _smooth_snap_correction_jitter(
    frames: List[Dict[str, object]],
    sigma_frames: float = 1.2,
    radius_frames: int = 3,
    min_run_frames: int = 6,
    max_correction_drift_m: float = 1.5,
) -> int:
    """Final-pass low-pass on the *snap correction* `(cx - x, cy - y)`.

    Why: CARLA polylines and V2XPNP lane geometry don't align perfectly, so
    even within a single stable CCLI run the per-frame snap correction
    wiggles by ~0.5-1 m. The visible (cx, cy) trajectory then has sub-2m
    jaggies that look like fake lane changes. Smoothing the *correction*
    (not the position itself) preserves raw curvature while killing
    snap-induced jitter.

    For each run of consecutive frames with the same CCLI, Gaussian-filter
    the (corr_x, corr_y) sequence and write back `(cx, cy) = (x, y) + corr_smooth`.

    `max_correction_drift_m` is a safety: if smoothed correction is far from
    raw original (i.e., we'd be moving the vehicle a lot toward raw), keep
    original. This avoids un-doing legitimate lane snapping in pathological cases.

    Returns number of frames modified.
    """
    if radius_frames < 1 or len(frames) < (2 * radius_frames + 2):
        return 0
    n = len(frames)

    # Build CCLI runs.
    runs: List[Tuple[int, int]] = []
    if n == 0:
        return 0
    cur_s = 0
    cur_li = _safe_int(frames[0].get("ccli"), -999)
    for i in range(1, n):
        li = _safe_int(frames[i].get("ccli"), -999)
        if li != cur_li:
            if (i - 1 - cur_s + 1) >= int(min_run_frames):
                runs.append((cur_s, i - 1))
            cur_s = i
            cur_li = li
    if (n - cur_s) >= int(min_run_frames):
        runs.append((cur_s, n - 1))
    if not runs:
        return 0

    # Gaussian kernel (1D, normalized).
    weights: List[float] = []
    for k in range(-int(radius_frames), int(radius_frames) + 1):
        weights.append(math.exp(-0.5 * (float(k) / float(sigma_frames)) ** 2))
    wsum = float(sum(weights))
    weights = [float(w) / wsum for w in weights]

    n_modified = 0
    for s, e in runs:
        L = e - s + 1
        if L < (2 * int(radius_frames) + 1):
            continue
        # Read corrections; bail if too many invalid frames.
        corr_xs = [float("nan")] * L
        corr_ys = [float("nan")] * L
        cx_vals = [float("nan")] * L
        cy_vals = [float("nan")] * L
        x_vals = [float("nan")] * L
        y_vals = [float("nan")] * L
        valid = 0
        for k_idx in range(L):
            fi = s + k_idx
            cx = _safe_float(frames[fi].get("cx"), float("nan"))
            cy = _safe_float(frames[fi].get("cy"), float("nan"))
            x_ = _safe_float(frames[fi].get("x"), float("nan"))
            y_ = _safe_float(frames[fi].get("y"), float("nan"))
            cx_vals[k_idx] = cx
            cy_vals[k_idx] = cy
            x_vals[k_idx] = x_
            y_vals[k_idx] = y_
            if all(math.isfinite(v) for v in (cx, cy, x_, y_)):
                corr_xs[k_idx] = cx - x_
                corr_ys[k_idx] = cy - y_
                valid += 1
        if valid < int(min_run_frames):
            continue

        # Convolve corrections with Gaussian (with edge handling).
        smoothed_x = list(corr_xs)
        smoothed_y = list(corr_ys)
        rad = int(radius_frames)
        for j in range(L):
            sx = sy = 0.0
            wnorm = 0.0
            for k_off, w_k in enumerate(weights):
                idx = j + k_off - rad
                if idx < 0 or idx >= L:
                    continue
                cxk = corr_xs[idx]
                cyk = corr_ys[idx]
                if not (math.isfinite(cxk) and math.isfinite(cyk)):
                    continue
                sx += float(w_k) * float(cxk)
                sy += float(w_k) * float(cyk)
                wnorm += float(w_k)
            if wnorm > 1e-6:
                smoothed_x[j] = sx / wnorm
                smoothed_y[j] = sy / wnorm

        # Write back. Skip frames where smoothing would move the position
        # by more than max_correction_drift_m relative to its original snap.
        for j in range(L):
            fi = s + j
            x_ = x_vals[j]
            y_ = y_vals[j]
            cx_orig = cx_vals[j]
            cy_orig = cy_vals[j]
            sx = smoothed_x[j]
            sy = smoothed_y[j]
            if not all(math.isfinite(v) for v in (x_, y_, cx_orig, cy_orig, sx, sy)):
                continue
            new_cx = float(x_) + float(sx)
            new_cy = float(y_) + float(sy)
            drift = math.hypot(new_cx - cx_orig, new_cy - cy_orig)
            if drift > float(max_correction_drift_m):
                continue
            frames[fi]["cx"] = float(new_cx)
            frames[fi]["cy"] = float(new_cy)
            frames[fi]["csource"] = "jitter_smooth"
            frames[fi]["jitter_smooth"] = True
            n_modified += 1
    return n_modified


def _smooth_lane_change_curves(
    frames: List[Dict[str, object]],
    window_radius: int = 4,
    skip_synthetic_turn: bool = True,
    require_semantic_change: bool = True,
    max_window_span_frames: int = 12,
) -> None:
    """Spread sharp lane-change steps in (cx, cy) into smooth S-curves.

    Per-frame lane snapping locks each frame's (cx, cy) to a single CARLA
    lane centerline. At a lane change, frame i-1 sits on lane A and frame i
    on lane B → the connecting segment is a near-perpendicular jump of
    ~one lane width. Closed-loop replay drives that jagged.

    For each detected lane change, anchor the window's first and last frames
    and shape-preserve-blend the interior frames: each interior frame keeps
    its RAW (x, y) curve and only the snap correction `(cx - x, cy - y)` is
    smoothstep-interpolated between the anchors. This preserves raw
    curvature (so a turn through a lane change stays curving), while still
    smoothing the lane-width step.
    """
    if window_radius < 1:
        return
    n = len(frames)
    if n < (2 * window_radius + 2):
        return

    # Find real lane-change indices first (CCLI changes at i-1 → i).
    changes: List[int] = []
    for i in range(1, n):
        if not (_frame_has_carla_pose(frames[i - 1]) and _frame_has_carla_pose(frames[i])):
            continue
        prev_li = _safe_int(frames[i - 1].get("ccli"), -1)
        cur_li = _safe_int(frames[i].get("ccli"), -1)
        if prev_li < 0 or cur_li < 0 or prev_li == cur_li:
            continue
        if skip_synthetic_turn and (
            bool(frames[i - 1].get("synthetic_turn", False))
            or bool(frames[i].get("synthetic_turn", False))
        ):
            continue
        if require_semantic_change:
            sk_prev = _frame_semantic_lane_key(frames[i - 1])
            sk_cur = _frame_semantic_lane_key(frames[i])
            # Both must be defined and DIFFERENT to count as a real change.
            if sk_prev is None or sk_cur is None or int(sk_prev) == int(sk_cur):
                continue
        changes.append(int(i))

    if not changes:
        return

    # Merge overlapping windows: if two changes are within 2*W frames of each
    # other (e.g., overtake = lane-out then lane-in), treat as one big window
    # so we don't double-smooth — but cap the merged span so we don't blend
    # across an entire long maneuver.
    merged: List[Tuple[int, int]] = []
    for ci in changes:
        s = max(0, ci - window_radius)
        e = min(n - 1, ci + window_radius - 1)
        if (
            merged
            and s <= merged[-1][1] + 1
            and (max(merged[-1][1], e) - merged[-1][0]) <= int(max_window_span_frames)
        ):
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    changed_idx: set = set()
    for s, e in merged:
        if (e - s) < 3:
            continue
        n_mod = _shape_preserving_blend(frames, int(s), int(e), "lane_change_blend")
        if n_mod > 0:
            for j in range(s + 1, e):
                changed_idx.add(int(j))

    if not changed_idx:
        return

    _recompute_yaw_from_neighbors(frames, changed_idx)


def _smooth_snap_artifact_jumps(
    frames: List[Dict[str, object]],
    min_jump_m: float = 2.0,
    raw_excess_m: float = 1.0,
    window_radius: int = 5,
    max_passes: int = 3,
    max_window_span_frames: int = 14,
) -> None:
    """Smooth abrupt (cx, cy) jumps introduced by lane snapping at intersections.

    Distinct from `_smooth_lane_change_curves` (which targets real semantic lane
    changes). This catches the case where the snapper bounces between
    connector lines mid-intersection: the raw query (x, y) moves smoothly but
    the snapped (cx, cy) makes a multi-meter jump. Detection: snap step
    exceeds the raw step by `raw_excess_m` *and* exceeds `min_jump_m` outright.

    Uses shape-preserving blend: interior frames keep their RAW shape and
    only the snap correction is smoothstep-interpolated. Critical for turns
    that span many frames — straight-line blending would replace the curve
    with a chord. Window merging is capped at `max_window_span_frames` so
    very long maneuvers aren't collapsed into one blend.

    Iterates up to `max_passes` times to handle cascades.
    """
    if window_radius < 1 or len(frames) < (2 * window_radius + 2):
        return
    n = len(frames)

    def _raw_step(i: int) -> float:
        if i <= 0 or i >= n:
            return 0.0
        a, b = frames[i - 1], frames[i]
        ax = _safe_float(a.get("x"), float("nan"))
        ay = _safe_float(a.get("y"), float("nan"))
        bx = _safe_float(b.get("x"), float("nan"))
        by = _safe_float(b.get("y"), float("nan"))
        if math.isfinite(ax) and math.isfinite(ay) and math.isfinite(bx) and math.isfinite(by):
            return float(math.hypot(bx - ax, by - ay))
        qax, qay, _ = _frame_query_pose(a)
        qbx, qby, _ = _frame_query_pose(b)
        return float(math.hypot(qbx - qax, qby - qay))

    def _snap_step(i: int) -> Optional[float]:
        if i <= 0 or i >= n:
            return None
        a, b = frames[i - 1], frames[i]
        if not (_frame_has_carla_pose(a) and _frame_has_carla_pose(b)):
            return None
        return float(math.hypot(
            float(b["cx"]) - float(a["cx"]),
            float(b["cy"]) - float(a["cy"]),
        ))

    for _pass in range(max(1, int(max_passes))):
        # Detect jumps for this pass.
        bad: List[int] = []
        for i in range(1, n):
            ss = _snap_step(i)
            if ss is None:
                continue
            rs = _raw_step(i)
            # Snap moved much more than raw → snapper introduced a discontinuity.
            if ss >= float(min_jump_m) and ss >= rs + float(raw_excess_m):
                bad.append(int(i))
        if not bad:
            return

        # Merge jumps that fall within 2*W frames into a single blend window.
        # Cap merged span so a long sequence of small jumps in a winding turn
        # doesn't get collapsed into one chord blend.
        merged: List[Tuple[int, int]] = []
        for ji in bad:
            s = max(0, ji - window_radius)
            e = min(n - 1, ji + window_radius - 1)
            if (
                merged
                and s <= merged[-1][1] + 1
                and (max(merged[-1][1], e) - merged[-1][0]) <= int(max_window_span_frames)
            ):
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        changed_idx: set = set()
        for s, e in merged:
            if (e - s) < 3:
                continue
            n_mod = _shape_preserving_blend(frames, int(s), int(e), "snap_jump_smooth")
            if n_mod > 0:
                for j in range(s + 1, e):
                    changed_idx.add(int(j))

        if not changed_idx:
            return

        _recompute_yaw_from_neighbors(frames, changed_idx)


def _frame_semantic_lane_key(frame: Dict[str, object]) -> Optional[int]:
    v = _safe_int(frame.get("assigned_lane_id"), 0)
    if v != 0:
        return int(v)
    v = _safe_int(frame.get("lane_id"), 0)
    if v != 0:
        return int(v)
    return None


def _stabilize_carla_line_ids(
    frames: List[Dict[str, object]],
    max_semantic_hold_run: int = 12,
    carla_context: Optional[Dict[str, object]] = None,
    opposite_reject_deg: float = 150.0,
) -> None:
    """
    Reduce CARLA line-id churn without changing XY geometry:
    - Keep a single line-id inside synthetic-turn spans.
    - Hold previous line-id across short runs when semantic lane id is stable.
    """
    if len(frames) < 2:
        return

    n = len(frames)
    hold_turn_runs = _env_int("V2X_CARLA_HOLD_TURN_LINE_IDS", 1, minimum=0, maximum=1) == 1
    hold_semantic = _env_int("V2X_CARLA_HOLD_SEMANTIC_LINE_IDS", 1, minimum=0, maximum=1) == 1
    if hold_turn_runs:
        turn_extend_frames = _env_int("V2X_CARLA_HOLD_TURN_LINE_IDS_EXTEND_FRAMES", 20, minimum=0, maximum=60)
        turn_hold_reject_wrong_way = _env_int("V2X_CARLA_HOLD_TURN_LINE_IDS_REJECT_WRONG_WAY", 0, minimum=0, maximum=1) == 1
        turn_hold_release_dist_gain_m = _env_float("V2X_CARLA_TURN_HOLD_RELEASE_DIST_GAIN_M", 0.85)
        turn_hold_release_anchor_max_dist_m = _env_float("V2X_CARLA_TURN_HOLD_RELEASE_ANCHOR_MAX_DIST_M", 2.8)
        turn_hold_release_lookahead = _env_int(
            "V2X_CARLA_TURN_HOLD_RELEASE_LOOKAHEAD_FRAMES",
            2,
            minimum=1,
            maximum=6,
        )

        def _raw_projection_dist(fr: Dict[str, object], line_idx: int) -> Optional[float]:
            if line_idx < 0 or not isinstance(carla_context, dict):
                return None
            qx, qy, qyaw = _frame_query_pose(fr)
            rx = _safe_float(fr.get("x"), float(qx))
            ry = _safe_float(fr.get("y"), float(qy))
            ryaw = _safe_float(fr.get("yaw"), _safe_float(fr.get("syaw"), float(qyaw)))
            cand = _best_projection_on_carla_line(
                carla_context=carla_context,
                line_index=int(line_idx),
                qx=float(rx),
                qy=float(ry),
                qyaw=float(ryaw),
                prefer_reversed=None,
                enforce_node_direction=False,
                opposite_reject_deg=float(opposite_reject_deg),
            )
            if not isinstance(cand, dict):
                return None
            proj = cand.get("projection", {})
            if not isinstance(proj, dict):
                return None
            d = _safe_float(proj.get("dist"), float("nan"))
            if not math.isfinite(d):
                return None
            return float(d)

        i = 0
        while i < n:
            if not bool(frames[i].get("synthetic_turn", False)):
                i += 1
                continue
            s = i
            while i + 1 < n and bool(frames[i + 1].get("synthetic_turn", False)):
                i += 1
            e = i
            anchor = -1
            if s > 0 and _frame_has_carla_pose(frames[s - 1]):
                anchor = _safe_int(frames[s - 1].get("ccli"), -1)
            if anchor < 0 and _frame_has_carla_pose(frames[s]):
                anchor = _safe_int(frames[s].get("ccli"), -1)
            if anchor < 0 and (e + 1) < n and _frame_has_carla_pose(frames[e + 1]):
                anchor = _safe_int(frames[e + 1].get("ccli"), -1)
            if anchor >= 0:
                base_sem_key = _frame_semantic_lane_key(frames[s])
                for fi in range(int(s), int(e) + 1):
                    fr = frames[fi]
                    if not _frame_has_carla_pose(fr):
                        continue
                    if base_sem_key is not None and _frame_semantic_lane_key(fr) != base_sem_key:
                        continue
                    alt_line = _safe_int(fr.get("cbcli"), -1)
                    if int(alt_line) >= 0 and int(alt_line) != int(anchor):
                        anchor_d = _raw_projection_dist(fr, int(anchor))
                        alt_d = _raw_projection_dist(fr, int(alt_line))
                        if (
                            anchor_d is not None
                            and alt_d is not None
                            and float(alt_d) + float(turn_hold_release_dist_gain_m) < float(anchor_d)
                            and (
                                float(anchor_d) >= float(turn_hold_release_anchor_max_dist_m)
                                or (float(anchor_d) - float(alt_d)) >= (1.35 * float(turn_hold_release_dist_gain_m))
                            )
                        ):
                            continue
                    if _safe_int(fr.get("ccli"), -1) == int(anchor):
                        continue
                    fr["ccli"] = int(anchor)
                    fr["csource"] = "turn_line_id_hold"
                    fr["turn_line_id_hold"] = True
                if int(turn_extend_frames) > 0:
                    for step_dir in (-1, 1):
                        steps = 0
                        j = int(s) - 1 if step_dir < 0 else int(e) + 1
                        while 0 <= j < n and steps < int(turn_extend_frames):
                            fr = frames[j]
                            if not _frame_has_carla_pose(fr):
                                break
                            if base_sem_key is not None and _frame_semantic_lane_key(fr) != base_sem_key:
                                break
                            # Release turn hold extension when the pre-turn anchor line
                            # is no longer the best fit in raw-space and an alternate
                            # stable line (typically cbcli) is clearly better.
                            alt_line = _safe_int(fr.get("cbcli"), -1)
                            if alt_line < 0 or int(alt_line) == int(anchor):
                                alt_line = _safe_int(fr.get("ccli"), -1)
                            if int(alt_line) >= 0 and int(alt_line) != int(anchor):
                                anchor_d0 = _raw_projection_dist(fr, int(anchor))
                                alt_d0 = _raw_projection_dist(fr, int(alt_line))
                                if (
                                    anchor_d0 is not None
                                    and alt_d0 is not None
                                    and float(alt_d0) + float(turn_hold_release_dist_gain_m) < float(anchor_d0)
                                ):
                                    look_hits = 0
                                    look_total = 0
                                    for off in range(max(1, int(turn_hold_release_lookahead))):
                                        jj = int(j) + int(step_dir) * int(off)
                                        if jj < 0 or jj >= n:
                                            break
                                        frj = frames[jj]
                                        if not _frame_has_carla_pose(frj):
                                            break
                                        if base_sem_key is not None and _frame_semantic_lane_key(frj) != base_sem_key:
                                            break
                                        alt_j = _safe_int(frj.get("cbcli"), -1)
                                        if alt_j < 0 or int(alt_j) == int(anchor):
                                            alt_j = _safe_int(frj.get("ccli"), -1)
                                        if int(alt_j) != int(alt_line):
                                            break
                                        ad = _raw_projection_dist(frj, int(anchor))
                                        bd = _raw_projection_dist(frj, int(alt_j))
                                        if ad is None or bd is None:
                                            break
                                        look_total += 1
                                        if float(bd) + float(turn_hold_release_dist_gain_m) < float(ad):
                                            look_hits += 1
                                        else:
                                            break
                                    need_hits = 1 if int(turn_hold_release_lookahead) <= 1 else 2
                                    if (
                                        int(look_hits) >= int(max(1, min(int(need_hits), int(look_total) if look_total > 0 else 1)))
                                        and (
                                            float(anchor_d0) >= float(turn_hold_release_anchor_max_dist_m)
                                            or (float(anchor_d0) - float(alt_d0)) >= (1.35 * float(turn_hold_release_dist_gain_m))
                                        )
                                    ):
                                        break
                            if bool(turn_hold_reject_wrong_way) and isinstance(carla_context, dict):
                                motion_yaw = _estimate_query_motion_yaw(frames, j, max_span=5, min_disp_m=0.12)
                                px = _safe_float(fr.get("cx"), float("nan"))
                                py = _safe_float(fr.get("cy"), float("nan"))
                                if motion_yaw is not None and math.isfinite(px) and math.isfinite(py):
                                    line_yaw = _carla_line_calibrated_yaw_at_point(
                                        carla_context=carla_context,
                                        line_index=int(anchor),
                                        x=float(px),
                                        y=float(py),
                                    )
                                    if (
                                        line_yaw is not None
                                        and float(_yaw_abs_diff_deg(float(line_yaw), float(motion_yaw))) >= float(opposite_reject_deg)
                                    ):
                                        break
                            if _safe_int(fr.get("ccli"), -1) != int(anchor):
                                fr["ccli"] = int(anchor)
                                fr["csource"] = "turn_line_id_hold_extend"
                                fr["turn_line_id_hold"] = True
                            j += int(step_dir)
                            steps += 1
            i += 1

    if not hold_semantic:
        return

    i = 1
    max_run = max(1, int(max_semantic_hold_run))
    hold_reject_wrong_way = _env_int("V2X_CARLA_SEMANTIC_LINE_HOLD_REJECT_WRONG_WAY", 1, minimum=0, maximum=1) == 1
    hold_max_prev_dist_m = _env_float("V2X_CARLA_SEMANTIC_LINE_HOLD_MAX_PREV_DIST_M", 8.5)
    hold_prev_dist_slack_m = _env_float("V2X_CARLA_SEMANTIC_LINE_HOLD_PREV_DIST_SLACK_M", 0.80)
    while i < n:
        prev = frames[i - 1]
        cur = frames[i]
        if not (_frame_has_carla_pose(prev) and _frame_has_carla_pose(cur)):
            i += 1
            continue
        prev_line = _safe_int(prev.get("ccli"), -1)
        cur_line = _safe_int(cur.get("ccli"), -1)
        if prev_line < 0 or cur_line < 0 or prev_line == cur_line:
            i += 1
            continue
        pk = _frame_semantic_lane_key(prev)
        ck = _frame_semantic_lane_key(cur)
        if pk is None or ck is None or int(pk) != int(ck):
            i += 1
            continue
        if bool(hold_reject_wrong_way) and isinstance(carla_context, dict):
            motion_yaw = _estimate_query_motion_yaw(frames, i, max_span=5, min_disp_m=0.12)
            px = _safe_float(cur.get("cx"), float("nan"))
            py = _safe_float(cur.get("cy"), float("nan"))
            if motion_yaw is not None and math.isfinite(px) and math.isfinite(py):
                line_yaw = _carla_line_calibrated_yaw_at_point(
                    carla_context=carla_context,
                    line_index=int(prev_line),
                    x=float(px),
                    y=float(py),
                )
                if (
                    line_yaw is not None
                    and float(_yaw_abs_diff_deg(float(line_yaw), float(motion_yaw))) >= float(opposite_reject_deg)
                ):
                    i += 1
                    continue

        j = i
        run = 0
        while j < n and run < max_run:
            fr = frames[j]
            if not _frame_has_carla_pose(fr):
                break
            if _frame_semantic_lane_key(fr) != pk:
                break
            li = _safe_int(fr.get("ccli"), -1)
            if li < 0:
                break
            if li == prev_line:
                break
            if isinstance(carla_context, dict):
                qx, qy, qyaw = _frame_query_pose(fr)
                prev_proj = _best_projection_on_carla_line(
                    carla_context=carla_context,
                    line_index=int(prev_line),
                    qx=float(qx),
                    qy=float(qy),
                    qyaw=float(qyaw),
                    prefer_reversed=None,
                    enforce_node_direction=True,
                    opposite_reject_deg=float(opposite_reject_deg),
                )
                if prev_proj is None:
                    prev_proj = _best_projection_on_carla_line(
                        carla_context=carla_context,
                        line_index=int(prev_line),
                        qx=float(qx),
                        qy=float(qy),
                        qyaw=float(qyaw),
                        prefer_reversed=None,
                        enforce_node_direction=False,
                        opposite_reject_deg=float(opposite_reject_deg),
                    )
                if prev_proj is None:
                    break
                prev_d = _safe_float(prev_proj.get("projection", {}).get("dist"), float("inf"))
                curr_d = _safe_float(fr.get("cdist"), float("inf"))
                if (not math.isfinite(prev_d)) or prev_d > float(hold_max_prev_dist_m):
                    break
                if math.isfinite(curr_d) and prev_d > float(curr_d) + float(hold_prev_dist_slack_m):
                    break
            fr["ccli"] = int(prev_line)
            fr["csource"] = "semantic_line_id_hold"
            fr["semantic_line_id_hold"] = True
            run += 1
            j += 1
        i = max(i + 1, j)


def _soften_semantic_boundary_jumps(
    frames: List[Dict[str, object]],
    jump_threshold_m: float = 2.6,
    jump_ratio_vs_query: float = 2.2,
    move_toward_mid: float = 0.55,
    max_shift_m: float = 1.35,
    max_query_cost_delta: float = 1.1,
) -> None:
    """
    Soften one-frame CARLA snap boundaries when semantic lane identity is stable.

    This targets connector transitions where lane IDs/segments differ but represent
    the same semantic lane family (e.g., junction connector between two road IDs).
    """
    if len(frames) < 2:
        return
    teleport_raw_step_m = _env_float("V2X_CARLA_SEMANTIC_BOUNDARY_TELEPORT_RAW_STEP_M", 0.6)
    teleport_snap_step_m = _env_float("V2X_CARLA_SEMANTIC_BOUNDARY_TELEPORT_SNAP_STEP_M", 1.8)

    def _propose_nudge(fi: int, tx: float, ty: float) -> Optional[Tuple[float, float, float]]:
        if fi < 0 or fi >= len(frames):
            return None
        fr = frames[fi]
        if not _frame_has_carla_pose(fr):
            return None
        old_x = _safe_float(fr.get("cx"), float("nan"))
        old_y = _safe_float(fr.get("cy"), float("nan"))
        if not math.isfinite(old_x) or not math.isfinite(old_y):
            return None

        dx = float(tx) - float(old_x)
        dy = float(ty) - float(old_y)
        shift = float(math.hypot(dx, dy))
        if shift <= 1e-6:
            return None
        if shift > float(max_shift_m):
            scale = float(max_shift_m) / max(1e-6, shift)
            tx = float(old_x) + float(dx) * scale
            ty = float(old_y) + float(dy) * scale

        qx, qy, _ = _frame_query_pose(fr)
        old_qdist = float(math.hypot(float(old_x) - float(qx), float(old_y) - float(qy)))
        new_qdist = float(math.hypot(float(tx) - float(qx), float(ty) - float(qy)))
        if new_qdist > old_qdist + float(max_query_cost_delta):
            return None

        return (float(tx), float(ty), float(new_qdist))

    def _raw_step(step_idx: int) -> float:
        if step_idx <= 0 or step_idx >= len(frames):
            return 0.0
        fa = frames[step_idx - 1]
        fb = frames[step_idx]
        ax = _safe_float(fa.get("x"), float("nan"))
        ay = _safe_float(fa.get("y"), float("nan"))
        bx = _safe_float(fb.get("x"), float("nan"))
        by = _safe_float(fb.get("y"), float("nan"))
        if math.isfinite(ax) and math.isfinite(ay) and math.isfinite(bx) and math.isfinite(by):
            return float(math.hypot(float(bx) - float(ax), float(by) - float(ay)))
        qax, qay, _ = _frame_query_pose(fa)
        qbx, qby, _ = _frame_query_pose(fb)
        return float(math.hypot(float(qbx) - float(qax), float(qby) - float(qay)))

    def _carla_step(
        step_idx: int,
        overrides: Dict[int, Tuple[float, float]],
    ) -> Optional[float]:
        if step_idx <= 0 or step_idx >= len(frames):
            return None
        fa = frames[step_idx - 1]
        fb = frames[step_idx]
        if not (_frame_has_carla_pose(fa) and _frame_has_carla_pose(fb)):
            return None
        ax, ay = overrides.get(
            int(step_idx - 1),
            (float(_safe_float(fa.get("cx"), 0.0)), float(_safe_float(fa.get("cy"), 0.0))),
        )
        bx, by = overrides.get(
            int(step_idx),
            (float(_safe_float(fb.get("cx"), 0.0)), float(_safe_float(fb.get("cy"), 0.0))),
        )
        return float(math.hypot(float(bx) - float(ax), float(by) - float(ay)))

    adjusted: set = set()
    for i in range(1, len(frames)):
        left = frames[i - 1]
        right = frames[i]
        if not (_frame_has_carla_pose(left) and _frame_has_carla_pose(right)):
            continue
        lli = _safe_int(left.get("ccli"), -1)
        rli = _safe_int(right.get("ccli"), -1)
        if lli < 0 or rli < 0 or lli == rli:
            continue
        lk = _frame_semantic_lane_key(left)
        rk = _frame_semantic_lane_key(right)
        if lk is None or rk is None or int(lk) != int(rk):
            continue

        lx = float(left["cx"])
        ly = float(left["cy"])
        rx = float(right["cx"])
        ry = float(right["cy"])
        carla_step = float(math.hypot(rx - lx, ry - ly))
        qlx, qly, _ = _frame_query_pose(left)
        qrx, qry, _ = _frame_query_pose(right)
        query_step = float(math.hypot(float(qrx) - float(qlx), float(qry) - float(qly)))
        if carla_step <= max(float(jump_threshold_m), float(jump_ratio_vs_query) * max(0.35, query_step)):
            continue

        mx = 0.5 * (lx + rx)
        my = 0.5 * (ly + ry)
        alpha = max(0.1, min(0.9, float(move_toward_mid)))

        ltx = lx + alpha * (mx - lx)
        lty = ly + alpha * (my - ly)
        rtx = rx + alpha * (mx - rx)
        rty = ry + alpha * (my - ry)

        left_prop = _propose_nudge(int(i - 1), float(ltx), float(lty))
        right_prop = _propose_nudge(int(i), float(rtx), float(rty))
        if left_prop is None and right_prop is None:
            continue

        overrides: Dict[int, Tuple[float, float]] = {}
        if left_prop is not None:
            overrides[int(i - 1)] = (float(left_prop[0]), float(left_prop[1]))
        if right_prop is not None:
            overrides[int(i)] = (float(right_prop[0]), float(right_prop[1]))
        affected_steps = [idx for idx in (i - 1, i, i + 1) if 0 < idx < len(frames)]
        old_local_max = 0.0
        new_local_max = 0.0
        has_step = False
        creates_teleport = False
        for step_idx in affected_steps:
            old_step = _carla_step(int(step_idx), overrides={})
            new_step = _carla_step(int(step_idx), overrides=overrides)
            if old_step is None or new_step is None:
                continue
            has_step = True
            old_local_max = max(float(old_local_max), float(old_step))
            new_local_max = max(float(new_local_max), float(new_step))
            raw_step_local = float(_raw_step(int(step_idx)))
            if float(raw_step_local) < float(teleport_raw_step_m) and float(new_step) > float(teleport_snap_step_m):
                creates_teleport = True
                break
        if not bool(has_step):
            continue
        if bool(creates_teleport):
            continue
        if float(new_local_max) >= float(old_local_max) - 1e-6:
            continue

        if left_prop is not None:
            frames[i - 1]["cx"] = float(left_prop[0])
            frames[i - 1]["cy"] = float(left_prop[1])
            frames[i - 1]["cdist"] = float(left_prop[2])
            frames[i - 1]["semantic_boundary_smooth"] = True
            frames[i - 1]["csource"] = "semantic_boundary_smooth"
            adjusted.add(int(i - 1))
        if right_prop is not None:
            frames[i]["cx"] = float(right_prop[0])
            frames[i]["cy"] = float(right_prop[1])
            frames[i]["cdist"] = float(right_prop[2])
            frames[i]["semantic_boundary_smooth"] = True
            frames[i]["csource"] = "semantic_boundary_smooth"
            adjusted.add(int(i))

    if not adjusted:
        return

    # Recompute yaw locally from neighbors after positional nudge.
    for fi in sorted(adjusted):
        fr = frames[fi]
        if not _frame_has_carla_pose(fr):
            continue
        prev_i = fi - 1
        next_i = fi + 1
        prev_ok = prev_i >= 0 and _frame_has_carla_pose(frames[prev_i])
        next_ok = next_i < len(frames) and _frame_has_carla_pose(frames[next_i])
        if not prev_ok and not next_ok:
            continue
        if prev_ok and next_ok:
            dx = float(frames[next_i]["cx"]) - float(frames[prev_i]["cx"])
            dy = float(frames[next_i]["cy"]) - float(frames[prev_i]["cy"])
        elif prev_ok:
            dx = float(fr["cx"]) - float(frames[prev_i]["cx"])
            dy = float(fr["cy"]) - float(frames[prev_i]["cy"])
        else:
            dx = float(frames[next_i]["cx"]) - float(fr["cx"])
            dy = float(frames[next_i]["cy"]) - float(fr["cy"])
        if math.hypot(dx, dy) <= 1e-6:
            continue
        fr["cyaw"] = float(_normalize_yaw_deg(math.degrees(math.atan2(dy, dx))))


def _hold_semantic_straight_switches(
    frames: List[Dict[str, object]],
    carla_context: Dict[str, object],
    enforce_node_direction: bool = False,
    opposite_reject_deg: float = 135.0,
) -> None:
    """
    Suppress teleport-like CARLA line switches when semantic lane identity
    remains unchanged and query motion is approximately straight.
    """
    if len(frames) < 2:
        return

    jump_threshold_m = _env_float("V2X_CARLA_SEMANTIC_HOLD_JUMP_M", 2.8)
    jump_ratio_vs_query = _env_float("V2X_CARLA_SEMANTIC_HOLD_JUMP_RATIO", 2.8)
    heading_max_deg = _env_float("V2X_CARLA_SEMANTIC_HOLD_HEADING_MAX_DEG", 12.0)
    prev_step_ratio_cap = _env_float("V2X_CARLA_SEMANTIC_HOLD_PREV_STEP_RATIO", 0.72)
    prev_step_abs_cap = _env_float("V2X_CARLA_SEMANTIC_HOLD_PREV_STEP_ABS_M", 2.6)
    prev_dist_delta_cap = _env_float("V2X_CARLA_SEMANTIC_HOLD_PREV_DIST_DELTA", 1.4)
    prev_dist_abs_cap = _env_float("V2X_CARLA_SEMANTIC_HOLD_PREV_DIST_ABS_M", 3.6)
    prefer_forward_line = _env_int("V2X_CARLA_SEMANTIC_HOLD_PREFER_FORWARD_LINE", 1, minimum=0, maximum=1) == 1
    forward_line_margin_m = _env_float("V2X_CARLA_SEMANTIC_HOLD_FORWARD_MARGIN_M", 0.8)
    forward_line_canonical_reject_deg = _env_float("V2X_CARLA_SEMANTIC_HOLD_FORWARD_CANON_REJECT_DEG", 170.0)

    n = len(frames)

    def _raw_motion_yaw(fi: int) -> Optional[float]:
        if fi < 0 or fi >= n:
            return None
        if n < 2:
            return None
        if 0 < fi < (n - 1):
            x0 = _safe_float(frames[fi - 1].get("x"), float("nan"))
            y0 = _safe_float(frames[fi - 1].get("y"), float("nan"))
            x1 = _safe_float(frames[fi + 1].get("x"), float("nan"))
            y1 = _safe_float(frames[fi + 1].get("y"), float("nan"))
            if math.isfinite(x0) and math.isfinite(y0) and math.isfinite(x1) and math.isfinite(y1):
                dx = float(x1) - float(x0)
                dy = float(y1) - float(y0)
                if math.hypot(dx, dy) >= 0.12:
                    return float(_normalize_yaw_deg(math.degrees(math.atan2(dy, dx))))
        if fi > 0:
            x0 = _safe_float(frames[fi - 1].get("x"), float("nan"))
            y0 = _safe_float(frames[fi - 1].get("y"), float("nan"))
            x1 = _safe_float(frames[fi].get("x"), float("nan"))
            y1 = _safe_float(frames[fi].get("y"), float("nan"))
            if math.isfinite(x0) and math.isfinite(y0) and math.isfinite(x1) and math.isfinite(y1):
                dx = float(x1) - float(x0)
                dy = float(y1) - float(y0)
                if math.hypot(dx, dy) >= 0.12:
                    return float(_normalize_yaw_deg(math.degrees(math.atan2(dy, dx))))
        if fi + 1 < n:
            x0 = _safe_float(frames[fi].get("x"), float("nan"))
            y0 = _safe_float(frames[fi].get("y"), float("nan"))
            x1 = _safe_float(frames[fi + 1].get("x"), float("nan"))
            y1 = _safe_float(frames[fi + 1].get("y"), float("nan"))
            if math.isfinite(x0) and math.isfinite(y0) and math.isfinite(x1) and math.isfinite(y1):
                dx = float(x1) - float(x0)
                dy = float(y1) - float(y0)
                if math.hypot(dx, dy) >= 0.12:
                    return float(_normalize_yaw_deg(math.degrees(math.atan2(dy, dx))))
        return None

    for i in range(1, n):
        left = frames[i - 1]
        right = frames[i]
        if not (_frame_has_carla_pose(left) and _frame_has_carla_pose(right)):
            continue
        if bool(left.get("synthetic_turn", False)) or bool(right.get("synthetic_turn", False)):
            continue

        left_line = _safe_int(left.get("ccli"), -1)
        right_line = _safe_int(right.get("ccli"), -1)
        if left_line < 0 or right_line < 0 or left_line == right_line:
            continue

        lk = _frame_semantic_lane_key(left)
        rk = _frame_semantic_lane_key(right)
        if lk is None or rk is None or int(lk) != int(rk):
            continue

        qlx, qly, _ = _frame_query_pose(left)
        qrx, qry, _ = _frame_query_pose(right)
        raw_prev_x = _safe_float(left.get("x"), float(qlx))
        raw_prev_y = _safe_float(left.get("y"), float(qly))
        raw_curr_x = _safe_float(right.get("x"), float(qrx))
        raw_curr_y = _safe_float(right.get("y"), float(qry))
        raw_step = float(math.hypot(float(raw_curr_x) - float(raw_prev_x), float(raw_curr_y) - float(raw_prev_y)))
        if raw_step < 0.05:
            raw_step = float(math.hypot(float(qrx) - float(qlx), float(qry) - float(qly)))

        lcx = float(left["cx"])
        lcy = float(left["cy"])
        rcx = float(right["cx"])
        rcy = float(right["cy"])
        carla_step = float(math.hypot(float(rcx) - float(lcx), float(rcy) - float(lcy)))
        if carla_step <= max(float(jump_threshold_m), float(jump_ratio_vs_query) * max(0.25, float(raw_step))):
            continue

        yaw_l = _raw_motion_yaw(i - 1)
        yaw_r = _raw_motion_yaw(i)
        if yaw_l is None:
            yaw_l = _estimate_query_motion_yaw(frames, i - 1, max_span=4, min_disp_m=0.12)
        if yaw_r is None:
            yaw_r = _estimate_query_motion_yaw(frames, i, max_span=4, min_disp_m=0.12)
        if yaw_l is not None and yaw_r is not None:
            if float(_yaw_abs_diff_deg(float(yaw_l), float(yaw_r))) > float(heading_max_deg):
                continue

        qx, qy, qyaw = _frame_query_pose(right)
        prev_proj = _best_projection_on_carla_line(
            carla_context=carla_context,
            line_index=int(left_line),
            qx=float(qx),
            qy=float(qy),
            qyaw=float(qyaw),
            prefer_reversed=None,
            enforce_node_direction=bool(enforce_node_direction),
            opposite_reject_deg=float(opposite_reject_deg),
        )
        if prev_proj is None:
            prev_proj = _best_projection_on_carla_line(
                carla_context=carla_context,
                line_index=int(left_line),
                qx=float(qx),
                qy=float(qy),
                qyaw=float(qyaw),
                prefer_reversed=None,
                enforce_node_direction=False,
                opposite_reject_deg=float(opposite_reject_deg),
            )
        if prev_proj is None:
            continue

        proj = prev_proj.get("projection", {})
        if not isinstance(proj, dict):
            continue
        px = _safe_float(proj.get("x"), float("nan"))
        py = _safe_float(proj.get("y"), float("nan"))
        if not (math.isfinite(px) and math.isfinite(py)):
            continue
        prev_step = float(math.hypot(float(px) - float(lcx), float(py) - float(lcy)))
        prev_step_limit = min(
            float(carla_step) * float(prev_step_ratio_cap),
            max(float(prev_step_abs_cap), 2.2 * max(0.25, float(raw_step))),
        )
        if prev_step > float(prev_step_limit):
            continue

        right_dist = _safe_float(right.get("cdist"), float("inf"))
        prev_dist = _safe_float(proj.get("dist"), float("inf"))
        if not math.isfinite(prev_dist):
            continue
        if prev_dist > float(prev_dist_abs_cap):
            continue
        if math.isfinite(right_dist) and prev_dist > right_dist + float(prev_dist_delta_cap):
            continue

        if bool(prefer_forward_line):
            motion_yaw = yaw_r if yaw_r is not None else _estimate_query_motion_yaw(frames, i, max_span=5, min_disp_m=0.12)
            if motion_yaw is not None:
                canonical_yaw = _carla_line_calibrated_yaw_at_point(
                    carla_context=carla_context,
                    line_index=int(left_line),
                    x=float(px),
                    y=float(py),
                )
                if (
                    canonical_yaw is not None
                    and float(_yaw_abs_diff_deg(float(canonical_yaw), float(motion_yaw))) >= float(forward_line_canonical_reject_deg)
                ):
                    alt = _nearest_projection_any_line(
                        carla_context=carla_context,
                        qx=float(qx),
                        qy=float(qy),
                        qyaw=float(motion_yaw),
                        enforce_node_direction=True,
                        opposite_reject_deg=float(opposite_reject_deg),
                        allow_reversed=False,
                    )
                    if alt is not None and isinstance(alt.get("projection"), dict):
                        alt_line = _safe_int(alt.get("line_index"), -1)
                        alt_proj = alt.get("projection", {})
                        alt_dist = _safe_float(alt_proj.get("dist"), float("inf"))
                        if (
                            alt_line >= 0
                            and alt_line != int(left_line)
                            and math.isfinite(alt_dist)
                            and float(alt_dist) <= float(prev_dist) + float(forward_line_margin_m)
                        ):
                            left_line = int(alt_line)
                            proj = alt_proj
                            px = _safe_float(proj.get("x"), float(px))
                            py = _safe_float(proj.get("y"), float(py))
                            prev_dist = float(alt_dist)

        _set_carla_pose(
            frame=right,
            line_index=int(left_line),
            x=float(px),
            y=float(py),
            yaw=float(_safe_float(proj.get("yaw"), _safe_float(right.get("cyaw"), qyaw))),
            dist=float(prev_dist),
            source="semantic_straight_hold_prev",
            quality=str(right.get("cquality", "none")),
        )


def _smooth_same_line_projection_spikes(
    frames: List[Dict[str, object]],
    jump_threshold_m: float = 2.4,
    jump_ratio_vs_query: float = 3.0,
    max_shift_m: float = 2.4,
    max_query_cost_delta: float = 0.7,
) -> None:
    """
    Repair large one-step CARLA spikes that are inconsistent with local query
    motion, especially when projection snaps along the same line.
    """
    if len(frames) < 3:
        return

    n = len(frames)
    for i in range(1, n - 1):
        a = frames[i - 1]
        b = frames[i]
        c = frames[i + 1]
        if not (_frame_has_carla_pose(a) and _frame_has_carla_pose(b) and _frame_has_carla_pose(c)):
            continue
        if bool(a.get("synthetic_turn", False)) or bool(b.get("synthetic_turn", False)) or bool(c.get("synthetic_turn", False)):
            continue

        cli_a = _safe_int(a.get("ccli"), -1)
        cli_b = _safe_int(b.get("ccli"), -1)
        cli_c = _safe_int(c.get("ccli"), -1)
        if cli_a < 0 or cli_b < 0:
            continue

        ax = float(a["cx"])
        ay = float(a["cy"])
        bx = float(b["cx"])
        by = float(b["cy"])
        cx = float(c["cx"])
        cy = float(c["cy"])

        carla_step = float(math.hypot(float(bx) - float(ax), float(by) - float(ay)))
        carla_next = float(math.hypot(float(cx) - float(bx), float(cy) - float(by)))
        qax, qay, _ = _frame_query_pose(a)
        qbx, qby, _ = _frame_query_pose(b)
        qcx, qcy, _ = _frame_query_pose(c)
        raw_step = float(math.hypot(float(qbx) - float(qax), float(qby) - float(qay)))
        if carla_step <= max(float(jump_threshold_m), float(jump_ratio_vs_query) * max(0.20, float(raw_step))):
            continue

        if cli_a == cli_b:
            # Same-line snap spike should be followed by much smaller motion.
            if carla_next > max(1.2, 0.70 * float(carla_step)):
                continue
        else:
            # Cross-line spike: only smooth near immediate return A->B->A pattern.
            if cli_c != cli_a:
                continue

        q_span = float(math.hypot(float(qcx) - float(qax), float(qcy) - float(qay)))
        if q_span > 0.20:
            alpha = float(math.hypot(float(qbx) - float(qax), float(qby) - float(qay))) / max(1e-6, q_span)
            alpha = max(0.0, min(1.0, float(alpha)))
        else:
            alpha = 0.5
        tx = float(ax) + float(alpha) * (float(cx) - float(ax))
        ty = float(ay) + float(alpha) * (float(cy) - float(ay))
        shift = float(math.hypot(float(tx) - float(bx), float(ty) - float(by)))
        if shift < 0.04:
            continue
        if shift > float(max_shift_m):
            scale = float(max_shift_m) / max(1e-6, float(shift))
            tx = float(bx) + (float(tx) - float(bx)) * float(scale)
            ty = float(by) + (float(ty) - float(by)) * float(scale)

        old_qdist = float(math.hypot(float(bx) - float(qbx), float(by) - float(qby)))
        new_qdist = float(math.hypot(float(tx) - float(qbx), float(ty) - float(qby)))
        if new_qdist > old_qdist + float(max_query_cost_delta):
            continue

        dy = float(cy) - float(ay)
        dx = float(cx) - float(ax)
        yaw = (
            float(_normalize_yaw_deg(math.degrees(math.atan2(dy, dx))))
            if math.hypot(dx, dy) > 1e-6
            else _safe_float(b.get("cyaw"), 0.0)
        )
        line_idx = int(cli_a) if int(cli_a) == int(cli_c) else int(cli_b)
        _set_carla_pose(
            frame=b,
            line_index=int(line_idx),
            x=float(tx),
            y=float(ty),
            yaw=float(yaw),
            dist=float(new_qdist),
            source="same_line_spike_smooth",
            quality=str(b.get("cquality", "none")),
        )
        b["same_line_spike_smooth"] = True


def _smooth_carla_micro_jitter(
    frames: List[Dict[str, object]],
    max_shift_m: float = 0.35,
    max_query_cost_delta: float = 0.25,
    passes: int = 2,
) -> None:
    """
    Conservative local smoother for high-frequency CARLA projection wiggles.
    Keeps frame timing unchanged and only nudges interior points when local
    second-difference spikes are likely projection artifacts.
    """
    if len(frames) < 3:
        return

    n = len(frames)
    for _ in range(max(1, int(passes))):
        changed = False
        for i in range(1, n - 1):
            a = frames[i - 1]
            b = frames[i]
            c = frames[i + 1]
            if not (_frame_has_carla_pose(a) and _frame_has_carla_pose(b) and _frame_has_carla_pose(c)):
                continue

            ta = _safe_float(a.get("t"), float(i - 1))
            tb = _safe_float(b.get("t"), float(i))
            tc = _safe_float(c.get("t"), float(i + 1))
            dt0 = float(tb - ta)
            dt1 = float(tc - tb)
            if dt0 <= 1e-4 or dt1 <= 1e-4 or dt0 > 0.35 or dt1 > 0.35:
                continue

            ax = float(a["cx"])
            ay = float(a["cy"])
            bx = float(b["cx"])
            by = float(b["cy"])
            cx = float(c["cx"])
            cy = float(c["cy"])

            s0 = float(math.hypot(bx - ax, by - ay))
            s1 = float(math.hypot(cx - bx, cy - by))
            if s0 < 0.20 or s1 < 0.20 or s0 > 2.60 or s1 > 2.60:
                continue

            h0 = math.degrees(math.atan2(by - ay, bx - ax))
            h1 = math.degrees(math.atan2(cy - by, cx - bx))
            turn_deg = float(abs(_normalize_yaw_deg(h1 - h0)))
            # Avoid smoothing through real turns / lane changes.
            if turn_deg > 55.0:
                continue

            # Local second-difference magnitude in XY.
            jx = float(cx - 2.0 * bx + ax)
            jy = float(cy - 2.0 * by + ay)
            jitter_mag = float(math.hypot(jx, jy))
            jitter_thr = max(0.16, 0.40 * min(s0, s1))
            if jitter_mag < jitter_thr:
                continue

            cand_x = 0.25 * ax + 0.50 * bx + 0.25 * cx
            cand_y = 0.25 * ay + 0.50 * by + 0.25 * cy
            shift = float(math.hypot(cand_x - bx, cand_y - by))
            if shift < 0.04:
                continue
            if shift > float(max_shift_m):
                scale = float(max_shift_m) / max(1e-6, shift)
                cand_x = bx + (cand_x - bx) * scale
                cand_y = by + (cand_y - by) * scale

            qx, qy, _ = _frame_query_pose(b)
            old_dist = float(math.hypot(float(bx) - float(qx), float(by) - float(qy)))
            new_dist = float(math.hypot(float(cand_x) - float(qx), float(cand_y) - float(qy)))
            if new_dist > old_dist + float(max_query_cost_delta):
                continue

            cli = _safe_int(b.get("ccli"), -1)
            acli = _safe_int(a.get("ccli"), -1)
            ccli = _safe_int(c.get("ccli"), -1)
            if acli >= 0 and acli == ccli and (cli < 0 or cli != acli):
                cli = int(acli)
            if cli < 0:
                cli = int(max(acli, ccli, 0))

            tan_dx = float(cx - ax)
            tan_dy = float(cy - ay)
            tan_yaw = (
                float(_normalize_yaw_deg(math.degrees(math.atan2(tan_dy, tan_dx))))
                if math.hypot(tan_dx, tan_dy) > 1e-6
                else _safe_float(b.get("cyaw"), 0.0)
            )
            old_yaw = _safe_float(b.get("cyaw"), tan_yaw)
            new_yaw = _interp_yaw_deg(old_yaw, tan_yaw, 0.65)

            _set_carla_pose(
                frame=b,
                line_index=int(cli),
                x=float(cand_x),
                y=float(cand_y),
                yaw=float(new_yaw),
                dist=float(new_dist),
                source="micro_jitter_smooth",
                quality=str(b.get("cquality", "none")),
            )
            b["micro_jitter_smooth"] = True
            changed = True

        if not changed:
            break


def _stabilize_carla_yaw_to_motion(frames: List[Dict[str, object]]) -> None:
    """
    Stabilize CARLA yaw against local motion direction to avoid tiny oscillations
    and near-opposite orientation artifacts that trigger control chatter.
    """
    if len(frames) < 2:
        return

    n = len(frames)
    for i, fr in enumerate(frames):
        if not _frame_has_carla_pose(fr):
            continue

        cyaw = _safe_float(fr.get("cyaw"), float("nan"))
        if not math.isfinite(cyaw):
            continue

        motion_yaw: Optional[float] = None
        if 0 < i < (n - 1) and _frame_has_carla_pose(frames[i - 1]) and _frame_has_carla_pose(frames[i + 1]):
            dx = float(frames[i + 1]["cx"]) - float(frames[i - 1]["cx"])
            dy = float(frames[i + 1]["cy"]) - float(frames[i - 1]["cy"])
            if math.hypot(dx, dy) >= 0.30:
                motion_yaw = float(_normalize_yaw_deg(math.degrees(math.atan2(dy, dx))))
        if motion_yaw is None and i > 0 and _frame_has_carla_pose(frames[i - 1]):
            dx = float(fr["cx"]) - float(frames[i - 1]["cx"])
            dy = float(fr["cy"]) - float(frames[i - 1]["cy"])
            if math.hypot(dx, dy) >= 0.30:
                motion_yaw = float(_normalize_yaw_deg(math.degrees(math.atan2(dy, dx))))
        if motion_yaw is None and i + 1 < n and _frame_has_carla_pose(frames[i + 1]):
            dx = float(frames[i + 1]["cx"]) - float(fr["cx"])
            dy = float(frames[i + 1]["cy"]) - float(fr["cy"])
            if math.hypot(dx, dy) >= 0.30:
                motion_yaw = float(_normalize_yaw_deg(math.degrees(math.atan2(dy, dx))))
        if motion_yaw is None:
            motion_yaw = _estimate_query_motion_yaw(frames, i)
        if motion_yaw is None:
            continue

        diff = float(_yaw_abs_diff_deg(float(cyaw), float(motion_yaw)))
        if diff < 10.0:
            continue
        alpha = 0.35 if diff < 40.0 else 0.65
        new_yaw = _interp_yaw_deg(float(cyaw), float(motion_yaw), float(alpha))

        _, _, qyaw = _frame_query_pose(fr)
        if _yaw_abs_diff_deg(float(new_yaw), float(qyaw)) >= 140.0 and _yaw_abs_diff_deg(float(cyaw), float(qyaw)) >= 140.0:
            new_yaw = _interp_yaw_deg(float(new_yaw), float(qyaw), 0.50)

        fr["cyaw"] = float(_normalize_yaw_deg(float(new_yaw)))
        fr["cyaw_smoothed"] = True


def _fallback_far_carla_samples(frames: List[Dict[str, object]]) -> None:
    """
    Final CARLA safety clamp: demote clearly-far projection samples to raw/V2 query
    pose fallback so visualization does not retain implausible snaps.
    """
    if not frames:
        return

    max_default = _env_float("V2X_CARLA_FAR_MAX_DEFAULT", 9.0)
    max_lane_corr = _env_float("V2X_CARLA_FAR_MAX_LANE_CORR", 10.0)
    max_nearest = _env_float("V2X_CARLA_FAR_MAX_NEAREST", 6.8)
    max_bridge = _env_float("V2X_CARLA_FAR_MAX_BRIDGE", 9.0)
    max_flicker = _env_float("V2X_CARLA_FAR_MAX_FLICKER", 8.5)
    max_guarded = _env_float("V2X_CARLA_FAR_MAX_GUARDED", 3.8)

    for fr in frames:
        if not _frame_has_carla_pose(fr):
            continue
        src = str(fr.get("csource", ""))
        if src in {"raw_fallback", "non_vehicle_raw", "raw_fallback_far"}:
            continue
        dist = _safe_float(fr.get("cdist"), float("inf"))
        if not math.isfinite(dist):
            continue

        src_l = src.strip().lower()
        max_allowed = float(max_default)
        if src_l.startswith("lane_corr"):
            max_allowed = float(max_lane_corr)
        elif src_l in {"nearest", "nearest_override", "nearest_gap"}:
            max_allowed = float(max_nearest)
        elif src_l in {"gap_bridge", "jump_stall_bridge"}:
            max_allowed = float(max_bridge)
        elif src_l in {"semantic_straight_hold_prev", "jump_guard_prev_line"}:
            max_allowed = float(max_guarded)
        elif "line_spike" in src_l or "flicker" in src_l:
            max_allowed = float(max_flicker)

        if dist <= float(max_allowed):
            continue

        qx, qy, qyaw = _frame_query_pose(fr)
        _set_carla_pose(
            frame=fr,
            line_index=-1,
            x=float(qx),
            y=float(qy),
            yaw=float(qyaw),
            dist=0.0,
            source="raw_fallback_far",
            quality="none",
        )


def _estimate_query_motion_yaw(
    frames: List[Dict[str, object]],
    index: int,
    max_span: int = 4,
    min_disp_m: float = 0.35,
) -> Optional[float]:
    n = len(frames)
    if n <= 1 or index < 0 or index >= n:
        return None

    i = int(index)
    qx, qy, _ = _frame_query_pose(frames[i])

    def _yaw_if_valid(x0: float, y0: float, x1: float, y1: float) -> Optional[float]:
        dx = float(x1) - float(x0)
        dy = float(y1) - float(y0)
        if math.hypot(dx, dy) < float(min_disp_m):
            return None
        return float(_normalize_yaw_deg(math.degrees(math.atan2(dy, dx))))

    # Prefer centered differences for robustness around noisy per-frame motion.
    for span in range(1, int(max_span) + 1):
        a = i - span
        b = i + span
        if a < 0 or b >= n:
            continue
        ax, ay, _ = _frame_query_pose(frames[a])
        bx, by, _ = _frame_query_pose(frames[b])
        yaw = _yaw_if_valid(ax, ay, bx, by)
        if yaw is not None:
            return yaw

    # Forward-only fallback.
    for span in range(1, int(max_span) + 1):
        b = i + span
        if b >= n:
            break
        bx, by, _ = _frame_query_pose(frames[b])
        yaw = _yaw_if_valid(qx, qy, bx, by)
        if yaw is not None:
            return yaw

    # Backward-only fallback.
    for span in range(1, int(max_span) + 1):
        a = i - span
        if a < 0:
            break
        ax, ay, _ = _frame_query_pose(frames[a])
        yaw = _yaw_if_valid(ax, ay, qx, qy)
        if yaw is not None:
            return yaw

    return None


def _clamp_turn_source_low_motion_steps(
    frames: List[Dict[str, object]],
    raw_low_step_m: float = 0.6,
    max_step_floor_m: float = 1.4,
    step_ratio_vs_raw: float = 2.4,
) -> None:
    """
    Clamp abrupt step sizes from turn-related CARLA sources during low-motion
    windows to avoid teleport-like spikes.
    """
    if len(frames) < 2:
        return

    turn_sources = {
        "turn_regen",
        "turn_line_id_hold_extend",
    }

    for i in range(1, len(frames)):
        prev = frames[i - 1]
        cur = frames[i]
        if not (_frame_has_carla_pose(prev) and _frame_has_carla_pose(cur)):
            continue
        src_prev = str(prev.get("csource", "")).strip().lower()
        src_cur = str(cur.get("csource", "")).strip().lower()
        if src_prev not in turn_sources and src_cur not in turn_sources:
            continue

        raw_prev_x = _safe_float(prev.get("x"), float("nan"))
        raw_prev_y = _safe_float(prev.get("y"), float("nan"))
        raw_cur_x = _safe_float(cur.get("x"), float("nan"))
        raw_cur_y = _safe_float(cur.get("y"), float("nan"))
        if math.isfinite(raw_prev_x) and math.isfinite(raw_prev_y) and math.isfinite(raw_cur_x) and math.isfinite(raw_cur_y):
            raw_step = float(
                math.hypot(
                    float(raw_cur_x) - float(raw_prev_x),
                    float(raw_cur_y) - float(raw_prev_y),
                )
            )
        else:
            qpx, qpy, _ = _frame_query_pose(prev)
            qcx, qcy, _ = _frame_query_pose(cur)
            raw_step = float(math.hypot(float(qcx) - float(qpx), float(qcy) - float(qpy)))
        if float(raw_step) >= float(raw_low_step_m):
            continue

        px = float(_safe_float(prev.get("cx"), 0.0))
        py = float(_safe_float(prev.get("cy"), 0.0))
        cx = float(_safe_float(cur.get("cx"), 0.0))
        cy = float(_safe_float(cur.get("cy"), 0.0))
        step = float(math.hypot(float(cx) - float(px), float(cy) - float(py)))
        max_allowed = max(float(max_step_floor_m), float(step_ratio_vs_raw) * float(raw_step))
        if float(step) <= float(max_allowed):
            continue
        if float(step) <= 1e-6:
            continue

        scale = float(max_allowed) / max(1e-6, float(step))
        nx = float(px) + (float(cx) - float(px)) * float(scale)
        ny = float(py) + (float(cy) - float(py)) * float(scale)
        qx, qy, _ = _frame_query_pose(cur)
        cur["cx"] = float(nx)
        cur["cy"] = float(ny)
        cur["cdist"] = float(math.hypot(float(nx) - float(qx), float(ny) - float(qy)))
        cur["csource"] = "turn_low_motion_clamp"
        cur["turn_low_motion_clamp"] = True

        dx = float(nx) - float(px)
        dy = float(ny) - float(py)
        if math.hypot(dx, dy) > 1e-6:
            yaw = float(_normalize_yaw_deg(math.degrees(math.atan2(dy, dx))))
            cur["cyaw"] = float(yaw)
            if math.hypot(dx, dy) > 0.05:
                prev["cyaw"] = float(yaw)


def _enforce_intersection_shape_consistency(
    frames: List[Dict[str, object]],
    carla_context: Optional[Dict[str, object]] = None,
    role: str = "vehicle",
) -> None:
    """
    Enforce one geometry mode per intersection window: either a smooth curve
    (cubic Bezier) or a straight line segment. This prevents mixed snap
    behavior (curve + straight) inside the same actor/window.
    """
    if len(frames) < 6:
        return
    if _env_int("V2X_CARLA_ENFORCE_INTERSECTION_SHAPE_MODE", 1, minimum=0, maximum=1) != 1:
        return

    role_norm = str(role or "").strip().lower()
    if role_norm not in {"ego", "vehicle"}:
        return

    summary = carla_context.get("summary", {}) if isinstance(carla_context, dict) else {}
    map_name = str(summary.get("map_name", "")).strip().lower()
    is_intersection_map = "intersection" in map_name

    n = len(frames)
    event_radius = _env_int("V2X_CARLA_INTERSECTION_SHAPE_EVENT_RADIUS", 5, minimum=2, maximum=12)
    synthetic_margin = _env_int("V2X_CARLA_INTERSECTION_SHAPE_SYN_MARGIN", 2, minimum=0, maximum=8)
    min_window_frames = _env_int("V2X_CARLA_INTERSECTION_SHAPE_MIN_WINDOW", 4, minimum=3, maximum=40)
    max_window_frames = _env_int("V2X_CARLA_INTERSECTION_SHAPE_MAX_WINDOW", 56, minimum=8, maximum=140)
    curve_tangent_scale = _env_float("V2X_CARLA_INTERSECTION_SHAPE_CURVE_TAN_SCALE", 0.42)
    curve_tangent_min_m = _env_float("V2X_CARLA_INTERSECTION_SHAPE_CURVE_TAN_MIN_M", 1.8)
    curve_tangent_max_m = _env_float("V2X_CARLA_INTERSECTION_SHAPE_CURVE_TAN_MAX_M", 18.0)
    curve_yaw_threshold_deg = _env_float("V2X_CARLA_INTERSECTION_SHAPE_CURVE_YAW_DEG", 12.0)
    curve_lateral_threshold_m = _env_float("V2X_CARLA_INTERSECTION_SHAPE_CURVE_LATERAL_M", 0.8)
    curve_prefer_slack = _env_float("V2X_CARLA_INTERSECTION_SHAPE_CURVE_SLACK", 1.2)
    straight_prefer_slack = _env_float("V2X_CARLA_INTERSECTION_SHAPE_STRAIGHT_SLACK", 0.4)
    max_error_worsen_per_frame = _env_float("V2X_CARLA_INTERSECTION_SHAPE_MAX_WORSEN_PER_FRAME_M", 0.55)
    # Hard guard: synthetic intersection shapes must remain near raw/query
    # trajectory so they cannot arc around the intersection.
    max_query_offset_m = _env_float("V2X_CARLA_INTERSECTION_SHAPE_MAX_QUERY_OFFSET_M", 3.0)
    max_raw_offset_m = _env_float("V2X_CARLA_INTERSECTION_SHAPE_MAX_RAW_OFFSET_M", 2.8)
    max_raw_worsen_per_frame = _env_float("V2X_CARLA_INTERSECTION_SHAPE_MAX_RAW_WORSEN_PER_FRAME_M", 0.35)
    max_raw_worsen_peak_m = _env_float("V2X_CARLA_INTERSECTION_SHAPE_MAX_RAW_WORSEN_PEAK_M", 0.8)

    def _semantic_transition(i: int) -> bool:
        if i <= 0 or i >= n:
            return False
        a = _frame_semantic_lane_key(frames[i - 1])
        b = _frame_semantic_lane_key(frames[i])
        if a is not None and b is not None:
            return bool(int(a) != int(b))
        ai = _safe_int(frames[i - 1].get("lane_index"), -1)
        bi = _safe_int(frames[i].get("lane_index"), -1)
        if ai >= 0 and bi >= 0:
            return bool(int(ai) != int(bi))
        return False

    def _source_is_intersectionish(fr: Dict[str, object]) -> bool:
        src = str(fr.get("csource", "")).strip().lower()
        if not src:
            return False
        if "turn" in src:
            return True
        if "transition" in src:
            return True
        if "semantic_boundary" in src:
            return True
        if "line_id_hold" in src:
            return True
        return False

    def _query_xy(fi: int) -> Tuple[float, float]:
        qx, qy, _ = _frame_query_pose(frames[fi])
        return (float(qx), float(qy))

    def _line_point_dist(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
        vx = float(bx) - float(ax)
        vy = float(by) - float(ay)
        den = float(vx * vx + vy * vy)
        if den <= 1e-9:
            return float(math.hypot(float(px) - float(ax), float(py) - float(ay)))
        wx = float(px) - float(ax)
        wy = float(py) - float(ay)
        t = (float(wx) * float(vx) + float(wy) * float(vy)) / float(den)
        t = max(0.0, min(1.0, float(t)))
        nx = float(ax) + float(t) * float(vx)
        ny = float(ay) + float(t) * float(vy)
        return float(math.hypot(float(px) - float(nx), float(py) - float(ny)))

    def _bezier_xy(
        p0: Tuple[float, float],
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        u: float,
    ) -> Tuple[float, float]:
        uu = max(0.0, min(1.0, float(u)))
        om = 1.0 - float(uu)
        b0 = float(om * om * om)
        b1 = float(3.0 * om * om * uu)
        b2 = float(3.0 * om * uu * uu)
        b3 = float(uu * uu * uu)
        x = float(b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0])
        y = float(b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1])
        return (float(x), float(y))

    def _bezier_deriv_xy(
        p0: Tuple[float, float],
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        u: float,
    ) -> Tuple[float, float]:
        uu = max(0.0, min(1.0, float(u)))
        om = 1.0 - float(uu)
        dx = (
            3.0 * float(om * om) * (float(p1[0]) - float(p0[0]))
            + 6.0 * float(om * uu) * (float(p2[0]) - float(p1[0]))
            + 3.0 * float(uu * uu) * (float(p3[0]) - float(p2[0]))
        )
        dy = (
            3.0 * float(om * om) * (float(p1[1]) - float(p0[1]))
            + 6.0 * float(om * uu) * (float(p2[1]) - float(p1[1]))
            + 3.0 * float(uu * uu) * (float(p3[1]) - float(p2[1]))
        )
        return (float(dx), float(dy))

    # Build candidate windows around line-id/semantic transitions and synthetic turns.
    windows: List[Tuple[int, int]] = []
    for i in range(1, n):
        prev = frames[i - 1]
        cur = frames[i]
        prev_cli = _safe_int(prev.get("ccli"), -1)
        cur_cli = _safe_int(cur.get("ccli"), -1)
        line_switch = prev_cli >= 0 and cur_cli >= 0 and int(prev_cli) != int(cur_cli)
        sem_switch = _semantic_transition(i)
        synth = bool(prev.get("synthetic_turn", False)) or bool(cur.get("synthetic_turn", False))
        src_hit = _source_is_intersectionish(prev) or _source_is_intersectionish(cur)
        if not (line_switch or sem_switch or synth or src_hit):
            continue
        if (not bool(is_intersection_map)) and (not bool(synth)) and (not bool(sem_switch)):
            continue
        s = max(0, int(i) - int(event_radius))
        e = min(n - 1, int(i) + int(event_radius))
        windows.append((int(s), int(e)))

    i = 0
    while i < n:
        if not bool(frames[i].get("synthetic_turn", False)):
            i += 1
            continue
        s = i
        while i + 1 < n and bool(frames[i + 1].get("synthetic_turn", False)):
            i += 1
        e = i
        windows.append((max(0, int(s) - int(synthetic_margin)), min(n - 1, int(e) + int(synthetic_margin))))
        i += 1

    if not windows:
        return

    windows.sort(key=lambda x: (int(x[0]), int(x[1])))
    merged: List[List[int]] = []
    for s, e in windows:
        if not merged or int(s) > int(merged[-1][1]) + 1:
            merged.append([int(s), int(e)])
        else:
            merged[-1][1] = max(int(merged[-1][1]), int(e))

    for pair in merged:
        s = int(pair[0])
        e = int(pair[1])
        if int(e) - int(s) + 1 < int(min_window_frames):
            continue
        if int(e) - int(s) + 1 > int(max_window_frames):
            continue
        valid_inside = [k for k in range(int(s), int(e) + 1) if _frame_has_carla_pose(frames[k])]
        if len(valid_inside) < int(min_window_frames):
            continue
        s0 = int(min(valid_inside))
        e0 = int(max(valid_inside))
        if int(e0) - int(s0) + 1 < int(min_window_frames):
            continue

        left = int(s0) - 1
        while left >= 0 and not _frame_has_carla_pose(frames[left]):
            left -= 1
        right = int(e0) + 1
        while right < n and not _frame_has_carla_pose(frames[right]):
            right += 1

        left_virtual = False
        right_virtual = False
        if left < 0:
            left = int(s0)
            left_virtual = True
        if right >= n:
            right = int(e0)
            right_virtual = True
        if int(right) - int(left) < 2:
            continue

        if not bool(left_virtual) and _frame_has_carla_pose(frames[left]):
            ax = float(_safe_float(frames[left].get("cx"), float("nan")))
            ay = float(_safe_float(frames[left].get("cy"), float("nan")))
            a_yaw = float(_safe_float(frames[left].get("cyaw"), float("nan")))
        else:
            ax, ay = _query_xy(int(s0))
            _, _, qyaw = _frame_query_pose(frames[int(s0)])
            a_yaw = float(qyaw)

        if not bool(right_virtual) and _frame_has_carla_pose(frames[right]):
            bx = float(_safe_float(frames[right].get("cx"), float("nan")))
            by = float(_safe_float(frames[right].get("cy"), float("nan")))
            b_yaw = float(_safe_float(frames[right].get("cyaw"), float("nan")))
        else:
            bx, by = _query_xy(int(e0))
            _, _, qyaw = _frame_query_pose(frames[int(e0)])
            b_yaw = float(qyaw)

        if not (math.isfinite(ax) and math.isfinite(ay) and math.isfinite(bx) and math.isfinite(by)):
            continue
        span = float(math.hypot(float(bx) - float(ax), float(by) - float(ay)))
        if span <= 0.8:
            continue

        # Parametrize by query cumulative displacement to preserve timing.
        idx_all = list(range(int(left), int(right) + 1))
        q_path: List[Tuple[float, float]] = [_query_xy(iq) for iq in idx_all]
        cum: List[float] = [0.0]
        for k in range(1, len(q_path)):
            cum.append(float(cum[-1]) + float(math.hypot(float(q_path[k][0]) - float(q_path[k - 1][0]), float(q_path[k][1]) - float(q_path[k - 1][1]))))
        total = float(cum[-1])
        u_by_idx: Dict[int, float] = {}
        if total <= 1e-6:
            den = max(1.0, float(len(idx_all) - 1))
            for off, fi in enumerate(idx_all):
                u_by_idx[int(fi)] = float(off) / float(den)
        else:
            for off, fi in enumerate(idx_all):
                u_by_idx[int(fi)] = float(cum[off]) / float(total)

        chord_yaw = float(_normalize_yaw_deg(math.degrees(math.atan2(float(by) - float(ay), float(bx) - float(ax)))))
        y0 = float(a_yaw)
        y1 = float(b_yaw)
        if not math.isfinite(y0):
            y0 = _safe_float(frames[s0].get("syaw"), _safe_float(frames[s0].get("yaw"), float(chord_yaw)))
        if not math.isfinite(y1):
            y1 = _safe_float(frames[e0].get("syaw"), _safe_float(frames[e0].get("yaw"), float(chord_yaw)))
        if not math.isfinite(y0):
            y0 = float(chord_yaw)
        if not math.isfinite(y1):
            y1 = float(chord_yaw)
        d = max(
            float(curve_tangent_min_m),
            min(float(curve_tangent_max_m), float(curve_tangent_scale) * float(span)),
        )
        y0r = math.radians(float(y0))
        y1r = math.radians(float(y1))
        p0 = (float(ax), float(ay))
        p3 = (float(bx), float(by))
        p1 = (float(ax) + float(d) * math.cos(float(y0r)), float(ay) + float(d) * math.sin(float(y0r)))
        p2 = (float(bx) - float(d) * math.cos(float(y1r)), float(by) - float(d) * math.sin(float(y1r)))

        cli_counts: Dict[int, int] = {}
        for fi in range(int(s0), int(e0) + 1):
            cli = _safe_int(frames[fi].get("ccli"), -1)
            if cli >= 0:
                cli_counts[int(cli)] = int(cli_counts.get(int(cli), 0) + 1)
        line_left = _safe_int(frames[left].get("ccli"), -1) if (not bool(left_virtual)) else -1
        line_right = _safe_int(frames[right].get("ccli"), -1) if (not bool(right_virtual)) else -1
        if cli_counts:
            dominant_cli = max(cli_counts.keys(), key=lambda k: int(cli_counts[k]))
        elif line_left >= 0:
            dominant_cli = int(line_left)
        else:
            dominant_cli = int(line_right)

        yaw_q0 = _estimate_query_motion_yaw(frames, int(s0), max_span=6, min_disp_m=0.20)
        yaw_q1 = _estimate_query_motion_yaw(frames, int(e0), max_span=6, min_disp_m=0.20)
        yaw_span = 0.0
        if yaw_q0 is not None and yaw_q1 is not None:
            yaw_span = float(_yaw_abs_diff_deg(float(yaw_q0), float(yaw_q1)))
        lateral_max = 0.0
        for fi in range(int(s0), int(e0) + 1):
            qx, qy = _query_xy(fi)
            lateral_max = max(float(lateral_max), float(_line_point_dist(float(qx), float(qy), float(ax), float(ay), float(bx), float(by))))
        curve_evidence = (
            float(yaw_span) >= float(curve_yaw_threshold_deg)
            or float(lateral_max) >= float(curve_lateral_threshold_m)
        )

        old_sum = 0.0
        old_max = 0.0
        old_raw_sum = 0.0
        old_raw_max = 0.0
        line_sum = 0.0
        line_max = 0.0
        line_raw_sum = 0.0
        line_raw_max = 0.0
        curve_sum = 0.0
        curve_max = 0.0
        curve_raw_sum = 0.0
        curve_raw_max = 0.0
        line_rows: List[Tuple[int, float, float, float, float, int]] = []
        curve_rows: List[Tuple[int, float, float, float, float, int]] = []
        for fi in range(int(s0), int(e0) + 1):
            fr = frames[fi]
            qx, qy = _query_xy(fi)
            rx = _safe_float(fr.get("x"), float(qx))
            ry = _safe_float(fr.get("y"), float(qy))
            ox = float(_safe_float(fr.get("cx"), float(qx)))
            oy = float(_safe_float(fr.get("cy"), float(qy)))
            old_d = float(math.hypot(float(ox) - float(qx), float(oy) - float(qy)))
            old_raw_d = float(math.hypot(float(ox) - float(rx), float(oy) - float(ry)))
            old_sum += float(old_d)
            old_max = max(float(old_max), float(old_d))
            old_raw_sum += float(old_raw_d)
            old_raw_max = max(float(old_raw_max), float(old_raw_d))

            u = float(u_by_idx.get(int(fi), 0.0))
            lx = float(ax) + float(u) * (float(bx) - float(ax))
            ly = float(ay) + float(u) * (float(by) - float(ay))
            lyaw = float(chord_yaw)
            ld = float(math.hypot(float(lx) - float(qx), float(ly) - float(qy)))
            ld_raw = float(math.hypot(float(lx) - float(rx), float(ly) - float(ry)))
            line_sum += float(ld)
            line_max = max(float(line_max), float(ld))
            line_raw_sum += float(ld_raw)
            line_raw_max = max(float(line_raw_max), float(ld_raw))
            line_cli = int(dominant_cli)
            if line_left >= 0 and line_right >= 0 and line_left != line_right:
                line_cli = int(line_left if float(u) < 0.5 else line_right)
            elif line_left >= 0:
                line_cli = int(line_left)
            elif line_right >= 0:
                line_cli = int(line_right)
            line_rows.append((int(fi), float(lx), float(ly), float(lyaw), float(ld), int(line_cli)))

            cx, cy = _bezier_xy(p0, p1, p2, p3, float(u))
            cdx, cdy = _bezier_deriv_xy(p0, p1, p2, p3, float(u))
            if math.hypot(float(cdx), float(cdy)) <= 1e-6:
                cyaw = float(chord_yaw)
            else:
                cyaw = float(_normalize_yaw_deg(math.degrees(math.atan2(float(cdy), float(cdx)))))
            cd = float(math.hypot(float(cx) - float(qx), float(cy) - float(qy)))
            cd_raw = float(math.hypot(float(cx) - float(rx), float(cy) - float(ry)))
            curve_sum += float(cd)
            curve_max = max(float(curve_max), float(cd))
            curve_raw_sum += float(cd_raw)
            curve_raw_max = max(float(curve_raw_max), float(cd_raw))
            curve_cli = int(dominant_cli)
            if line_left >= 0 and line_right >= 0 and line_left != line_right:
                curve_cli = int(line_left if float(u) < 0.5 else line_right)
            elif line_left >= 0:
                curve_cli = int(line_left)
            elif line_right >= 0:
                curve_cli = int(line_right)
            curve_rows.append((int(fi), float(cx), float(cy), float(cyaw), float(cd), int(curve_cli)))

        if not line_rows or not curve_rows:
            continue

        mode = "straight"
        if bool(curve_evidence):
            mode = "curve" if float(curve_sum) <= float(line_sum) + float(curve_prefer_slack) else "straight"
        else:
            mode = "straight" if float(line_sum) <= float(curve_sum) + float(straight_prefer_slack) else "curve"

        chosen_rows = curve_rows if str(mode) == "curve" else line_rows
        chosen_sum = float(curve_sum if str(mode) == "curve" else line_sum)
        chosen_max = float(curve_max if str(mode) == "curve" else line_max)
        chosen_raw_sum = float(curve_raw_sum if str(mode) == "curve" else line_raw_sum)
        chosen_raw_max = float(curve_raw_max if str(mode) == "curve" else line_raw_max)
        if float(chosen_max) > float(max_query_offset_m):
            # Reject synthetic window edits that drift too far from observed
            # trajectory; keep original CARLA projection for this window.
            continue
        if float(chosen_raw_max) > float(max_raw_offset_m):
            # Synthetic intersection windows must stay close to raw actor motion.
            continue
        local_len = max(1, int(e0) - int(s0) + 1)
        max_worsen = float(max_error_worsen_per_frame) * float(local_len)
        if float(chosen_sum) > float(old_sum) + float(max_worsen) and float(chosen_max) > float(old_max) + 1.8:
            # Keep original when both aggregate and worst-point errors would
            # degrade substantially.
            continue
        raw_max_worsen = float(max_raw_worsen_per_frame) * float(local_len)
        if (
            float(chosen_raw_sum) > float(old_raw_sum) + float(raw_max_worsen)
            and float(chosen_raw_max) > float(old_raw_max) + float(max_raw_worsen_peak_m)
        ):
            continue

        source = "intersection_shape_curve" if str(mode) == "curve" else "intersection_shape_straight"
        for fi, x, y, yaw, dist, cli in chosen_rows:
            fr = frames[int(fi)]
            _set_carla_pose(
                frame=fr,
                line_index=int(cli),
                x=float(x),
                y=float(y),
                yaw=float(yaw),
                dist=float(dist),
                source=str(source),
                quality=str(fr.get("cquality", "none")),
            )
            fr["intersection_shape_mode"] = str(mode)
            fr["intersection_shape_window_start"] = int(s0)
            fr["intersection_shape_window_end"] = int(e0)
            fr["intersection_shape_curve_evidence"] = bool(curve_evidence)


def _force_single_shape_mode_on_intersection_clusters(
    frames: List[Dict[str, object]],
    role: str = "vehicle",
) -> None:
    """
    Hard enforcement pass: every detected intersection cluster on a track is
    rewritten using exactly one model (curve or straight).
    """
    if len(frames) < 4:
        return
    if _env_int("V2X_CARLA_FORCE_SINGLE_INTERSECTION_MODE", 1, minimum=0, maximum=1) != 1:
        return

    role_norm = str(role or "").strip().lower()
    if role_norm not in {"ego", "vehicle"}:
        return

    n = len(frames)
    min_cluster_frames = _env_int("V2X_CARLA_INTERSECTION_CLUSTER_MIN_FRAMES", 3, minimum=2, maximum=50)
    curve_yaw_threshold_deg = _env_float("V2X_CARLA_INTERSECTION_CLUSTER_CURVE_YAW_DEG", 11.0)
    curve_lateral_threshold_m = _env_float("V2X_CARLA_INTERSECTION_CLUSTER_CURVE_LATERAL_M", 0.7)
    curve_tangent_scale = _env_float("V2X_CARLA_INTERSECTION_CLUSTER_CURVE_TAN_SCALE", 0.40)
    curve_tangent_min_m = _env_float("V2X_CARLA_INTERSECTION_CLUSTER_CURVE_TAN_MIN_M", 1.5)
    curve_tangent_max_m = _env_float("V2X_CARLA_INTERSECTION_CLUSTER_CURVE_TAN_MAX_M", 16.0)
    max_error_worsen_per_frame = _env_float("V2X_CARLA_INTERSECTION_CLUSTER_MAX_WORSEN_PER_FRAME_M", 0.75)

    def _query_xy(fi: int) -> Tuple[float, float]:
        qx, qy, _ = _frame_query_pose(frames[fi])
        return (float(qx), float(qy))

    def _point_seg_dist(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
        vx = float(bx) - float(ax)
        vy = float(by) - float(ay)
        den = float(vx * vx + vy * vy)
        if den <= 1e-9:
            return float(math.hypot(float(px) - float(ax), float(py) - float(ay)))
        wx = float(px) - float(ax)
        wy = float(py) - float(ay)
        t = (float(wx) * float(vx) + float(wy) * float(vy)) / float(den)
        t = max(0.0, min(1.0, float(t)))
        nx = float(ax) + float(t) * float(vx)
        ny = float(ay) + float(t) * float(vy)
        return float(math.hypot(float(px) - float(nx), float(py) - float(ny)))

    def _bezier_xy(
        p0: Tuple[float, float],
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        u: float,
    ) -> Tuple[float, float]:
        uu = max(0.0, min(1.0, float(u)))
        om = 1.0 - float(uu)
        b0 = float(om * om * om)
        b1 = float(3.0 * om * om * uu)
        b2 = float(3.0 * om * uu * uu)
        b3 = float(uu * uu * uu)
        x = float(b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0])
        y = float(b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1])
        return (float(x), float(y))

    def _bezier_deriv(
        p0: Tuple[float, float],
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        u: float,
    ) -> Tuple[float, float]:
        uu = max(0.0, min(1.0, float(u)))
        om = 1.0 - float(uu)
        dx = (
            3.0 * float(om * om) * (float(p1[0]) - float(p0[0]))
            + 6.0 * float(om * uu) * (float(p2[0]) - float(p1[0]))
            + 3.0 * float(uu * uu) * (float(p3[0]) - float(p2[0]))
        )
        dy = (
            3.0 * float(om * om) * (float(p1[1]) - float(p0[1]))
            + 6.0 * float(om * uu) * (float(p2[1]) - float(p1[1]))
            + 3.0 * float(uu * uu) * (float(p3[1]) - float(p2[1]))
        )
        return (float(dx), float(dy))

    def _is_intersectionish_source(fr: Dict[str, object]) -> bool:
        if bool(fr.get("synthetic_turn", False)):
            return True
        src = str(fr.get("csource", "")).strip().lower()
        if not src:
            return False
        keys = (
            "turn",
            "transition",
            "semantic_boundary",
            "line_id_hold",
            "intersection_shape_",
        )
        return any(k in src for k in keys)

    marks = [False] * n
    for i in range(n):
        if _is_intersectionish_source(frames[i]):
            marks[i] = True
    for i in range(1, n):
        a = frames[i - 1]
        b = frames[i]
        ka = _frame_semantic_lane_key(a)
        kb = _frame_semantic_lane_key(b)
        sem_switch = ka is not None and kb is not None and int(ka) != int(kb)
        ca = _safe_int(a.get("ccli"), -1)
        cb = _safe_int(b.get("ccli"), -1)
        line_switch = ca >= 0 and cb >= 0 and int(ca) != int(cb)
        if sem_switch or line_switch:
            marks[i - 1] = True
            marks[i] = True
    # Include immediate neighbors so boundary frames are classified with the
    # same intersection mode instead of being left as one-frame gaps.
    marks_expanded = list(marks)
    for i in range(n):
        if not bool(marks[i]):
            continue
        if i > 0:
            marks_expanded[i - 1] = True
        if i + 1 < n:
            marks_expanded[i + 1] = True
    marks = marks_expanded

    marked_idx = [i for i, v in enumerate(marks) if bool(v)]
    if not marked_idx:
        return
    # Global per-track forcing can over-constrain long trajectories and cause
    # large raw-fidelity drift on some actors; keep this opt-in.
    force_global_mode = (
        _env_int("V2X_CARLA_INTERSECTION_GLOBAL_MODE_PER_TRACK", 0, minimum=0, maximum=1) == 1
    )
    global_mode: Optional[str] = None
    if bool(force_global_mode):
        g_counts: Dict[str, int] = {}
        for fi in marked_idx:
            m = str(frames[fi].get("intersection_shape_mode", "")).strip().lower()
            if m in {"curve", "straight"}:
                g_counts[m] = int(g_counts.get(m, 0) + 1)
        if g_counts:
            global_mode = str(max(g_counts.keys(), key=lambda k: int(g_counts[k])))
        else:
            fs = int(marked_idx[0])
            fe = int(marked_idx[-1])
            yaw0 = _estimate_query_motion_yaw(frames, int(fs), max_span=6, min_disp_m=0.18)
            yaw1 = _estimate_query_motion_yaw(frames, int(fe), max_span=6, min_disp_m=0.18)
            qx0, qy0 = _query_xy(int(fs))
            qx1, qy1 = _query_xy(int(fe))
            lateral_max = 0.0
            for fi in marked_idx:
                qx, qy = _query_xy(int(fi))
                lateral_max = max(
                    float(lateral_max),
                    float(_point_seg_dist(float(qx), float(qy), float(qx0), float(qy0), float(qx1), float(qy1))),
                )
            yaw_span = 0.0
            if yaw0 is not None and yaw1 is not None:
                yaw_span = float(_yaw_abs_diff_deg(float(yaw0), float(yaw1)))
            global_mode = (
                "curve"
                if (
                    float(yaw_span) >= float(curve_yaw_threshold_deg)
                    or float(lateral_max) >= float(curve_lateral_threshold_m)
                )
                else "straight"
            )

    i = 0
    while i < n:
        if not marks[i]:
            i += 1
            continue
        s = i
        while i + 1 < n and marks[i + 1]:
            i += 1
        e = i
        if int(e) - int(s) + 1 < int(min_cluster_frames):
            i += 1
            continue

        left = int(s) - 1
        while left >= 0 and not _frame_has_carla_pose(frames[left]):
            left -= 1
        right = int(e) + 1
        while right < n and not _frame_has_carla_pose(frames[right]):
            right += 1
        left_virtual = False
        right_virtual = False
        if left < 0:
            left = int(s)
            left_virtual = True
        if right >= n:
            right = int(e)
            right_virtual = True
        if int(right) - int(left) < 2:
            i += 1
            continue

        if not bool(left_virtual) and _frame_has_carla_pose(frames[left]):
            ax = float(_safe_float(frames[left].get("cx"), float("nan")))
            ay = float(_safe_float(frames[left].get("cy"), float("nan")))
            ayaw = float(_safe_float(frames[left].get("cyaw"), float("nan")))
        else:
            ax, ay = _query_xy(int(s))
            _, _, qyaw = _frame_query_pose(frames[int(s)])
            ayaw = float(qyaw)
        if not bool(right_virtual) and _frame_has_carla_pose(frames[right]):
            bx = float(_safe_float(frames[right].get("cx"), float("nan")))
            by = float(_safe_float(frames[right].get("cy"), float("nan")))
            byaw = float(_safe_float(frames[right].get("cyaw"), float("nan")))
        else:
            bx, by = _query_xy(int(e))
            _, _, qyaw = _frame_query_pose(frames[int(e)])
            byaw = float(qyaw)
        if not (math.isfinite(ax) and math.isfinite(ay) and math.isfinite(bx) and math.isfinite(by)):
            i += 1
            continue

        span = float(math.hypot(float(bx) - float(ax), float(by) - float(ay)))
        if span <= 0.6:
            i += 1
            continue

        idx_all = list(range(int(left), int(right) + 1))
        q_path: List[Tuple[float, float]] = [_query_xy(k) for k in idx_all]
        cum: List[float] = [0.0]
        for k in range(1, len(q_path)):
            cum.append(float(cum[-1]) + float(math.hypot(float(q_path[k][0]) - float(q_path[k - 1][0]), float(q_path[k][1]) - float(q_path[k - 1][1]))))
        total = float(cum[-1])
        u_by_idx: Dict[int, float] = {}
        if total <= 1e-6:
            den = max(1.0, float(len(idx_all) - 1))
            for off, fi in enumerate(idx_all):
                u_by_idx[int(fi)] = float(off) / float(den)
        else:
            for off, fi in enumerate(idx_all):
                u_by_idx[int(fi)] = float(cum[off]) / float(total)

        mode_counts: Dict[str, int] = {}
        missing_mode = 0
        for fi in range(int(s), int(e) + 1):
            m = str(frames[fi].get("intersection_shape_mode", "")).strip().lower()
            if m in {"curve", "straight"}:
                mode_counts[m] = int(mode_counts.get(m, 0) + 1)
            else:
                missing_mode += 1

        chord_yaw = float(_normalize_yaw_deg(math.degrees(math.atan2(float(by) - float(ay), float(bx) - float(ax)))))
        yaw0 = _estimate_query_motion_yaw(frames, int(s), max_span=6, min_disp_m=0.18)
        yaw1 = _estimate_query_motion_yaw(frames, int(e), max_span=6, min_disp_m=0.18)
        yaw_span = 0.0
        if yaw0 is not None and yaw1 is not None:
            yaw_span = float(_yaw_abs_diff_deg(float(yaw0), float(yaw1)))
        lateral_max = 0.0
        for fi in range(int(s), int(e) + 1):
            qx, qy = _query_xy(fi)
            lateral_max = max(float(lateral_max), float(_point_seg_dist(float(qx), float(qy), float(ax), float(ay), float(bx), float(by))))

        if mode_counts:
            chosen_mode = max(mode_counts.keys(), key=lambda k: int(mode_counts[k]))
        else:
            chosen_mode = (
                "curve"
                if (
                    float(yaw_span) >= float(curve_yaw_threshold_deg)
                    or float(lateral_max) >= float(curve_lateral_threshold_m)
                )
                else "straight"
            )
        if str(global_mode or "") in {"curve", "straight"}:
            chosen_mode = str(global_mode)
        hard_enforce = bool(missing_mode > 0 or len(mode_counts) != 1)

        if not math.isfinite(ayaw):
            ayaw = _safe_float(frames[s].get("syaw"), _safe_float(frames[s].get("yaw"), float(chord_yaw)))
        if not math.isfinite(byaw):
            byaw = _safe_float(frames[e].get("syaw"), _safe_float(frames[e].get("yaw"), float(chord_yaw)))
        if not math.isfinite(ayaw):
            ayaw = float(chord_yaw)
        if not math.isfinite(byaw):
            byaw = float(chord_yaw)
        d = max(
            float(curve_tangent_min_m),
            min(float(curve_tangent_max_m), float(curve_tangent_scale) * float(span)),
        )
        ayaw_r = math.radians(float(ayaw))
        byaw_r = math.radians(float(byaw))
        p0 = (float(ax), float(ay))
        p3 = (float(bx), float(by))
        p1 = (float(ax) + float(d) * math.cos(float(ayaw_r)), float(ay) + float(d) * math.sin(float(ayaw_r)))
        p2 = (float(bx) - float(d) * math.cos(float(byaw_r)), float(by) - float(d) * math.sin(float(byaw_r)))

        ccounts: Dict[int, int] = {}
        for fi in range(int(s), int(e) + 1):
            cli = _safe_int(frames[fi].get("ccli"), -1)
            if cli >= 0:
                ccounts[int(cli)] = int(ccounts.get(int(cli), 0) + 1)
        left_cli = _safe_int(frames[left].get("ccli"), -1) if (not bool(left_virtual)) else -1
        right_cli = _safe_int(frames[right].get("ccli"), -1) if (not bool(right_virtual)) else -1
        if ccounts:
            dominant_cli = max(ccounts.keys(), key=lambda k: int(ccounts[k]))
        elif left_cli >= 0:
            dominant_cli = int(left_cli)
        else:
            dominant_cli = int(right_cli)

        old_sum = 0.0
        old_max = 0.0
        new_sum = 0.0
        new_max = 0.0
        rows: List[Tuple[int, float, float, float, float, int]] = []
        for fi in range(int(s), int(e) + 1):
            fr = frames[fi]
            qx, qy = _query_xy(fi)
            ox = float(_safe_float(fr.get("cx"), float(qx)))
            oy = float(_safe_float(fr.get("cy"), float(qy)))
            od = float(math.hypot(float(ox) - float(qx), float(oy) - float(qy)))
            old_sum += float(od)
            old_max = max(float(old_max), float(od))

            u = float(u_by_idx.get(int(fi), 0.0))
            if str(chosen_mode) == "curve":
                nx, ny = _bezier_xy(p0, p1, p2, p3, float(u))
                dx, dy = _bezier_deriv(p0, p1, p2, p3, float(u))
                if math.hypot(float(dx), float(dy)) <= 1e-6:
                    nyaw = float(chord_yaw)
                else:
                    nyaw = float(_normalize_yaw_deg(math.degrees(math.atan2(float(dy), float(dx)))))
                if left_cli >= 0 and right_cli >= 0 and left_cli != right_cli:
                    cli = int(left_cli if float(u) < 0.5 else right_cli)
                elif left_cli >= 0:
                    cli = int(left_cli)
                elif right_cli >= 0:
                    cli = int(right_cli)
                else:
                    cli = int(dominant_cli)
            else:
                nx = float(ax) + float(u) * (float(bx) - float(ax))
                ny = float(ay) + float(u) * (float(by) - float(ay))
                nyaw = float(chord_yaw)
                cli = int(dominant_cli)

            nd = float(math.hypot(float(nx) - float(qx), float(ny) - float(qy)))
            new_sum += float(nd)
            new_max = max(float(new_max), float(nd))
            rows.append((int(fi), float(nx), float(ny), float(nyaw), float(nd), int(cli)))

        if not rows:
            i += 1
            continue

        local_len = max(1, int(e) - int(s) + 1)
        max_worsen = float(max_error_worsen_per_frame) * float(local_len)
        if (
            (not bool(hard_enforce))
            and float(new_sum) > float(old_sum) + float(max_worsen)
            and float(new_max) > float(old_max) + 2.0
        ):
            i += 1
            continue

        source = (
            "intersection_cluster_curve"
            if str(chosen_mode) == "curve"
            else "intersection_cluster_straight"
        )
        for fi, nx, ny, nyaw, nd, cli in rows:
            fr = frames[int(fi)]
            _set_carla_pose(
                frame=fr,
                line_index=int(cli),
                x=float(nx),
                y=float(ny),
                yaw=float(nyaw),
                dist=float(nd),
                source=str(source),
                quality=str(fr.get("cquality", "none")),
            )
            fr["intersection_shape_mode"] = str(chosen_mode)
            fr["intersection_shape_window_start"] = int(s)
            fr["intersection_shape_window_end"] = int(e)
            fr["intersection_shape_cluster_force"] = True

        i += 1

    # Final harmonization on marked frames:
    # 1) remove one-frame mode flips between same-mode neighbors,
    # 2) fill unlabeled marked frames from nearest labeled neighbors.
    def _mode_at(fi: int) -> str:
        return str(frames[fi].get("intersection_shape_mode", "")).strip().lower()

    for fi in range(1, n - 1):
        if not bool(marks[fi]):
            continue
        mm = _mode_at(fi)
        pm = _mode_at(fi - 1) if bool(marks[fi - 1]) else ""
        nm = _mode_at(fi + 1) if bool(marks[fi + 1]) else ""
        if pm in {"curve", "straight"} and pm == nm and mm in {"curve", "straight"} and mm != pm:
            px = _safe_float(frames[fi - 1].get("cx"), float("nan"))
            py = _safe_float(frames[fi - 1].get("cy"), float("nan"))
            nx = _safe_float(frames[fi + 1].get("cx"), float("nan"))
            ny = _safe_float(frames[fi + 1].get("cy"), float("nan"))
            if math.isfinite(px) and math.isfinite(py) and math.isfinite(nx) and math.isfinite(ny):
                mx = 0.5 * (float(px) + float(nx))
                my = 0.5 * (float(py) + float(ny))
                dy = float(ny) - float(py)
                dx = float(nx) - float(px)
                myaw = float(
                    _normalize_yaw_deg(math.degrees(math.atan2(float(dy), float(dx))))
                    if math.hypot(float(dx), float(dy)) > 1e-6
                    else _safe_float(frames[fi].get("cyaw"), 0.0)
                )
                qx, qy, _ = _frame_query_pose(frames[fi])
                _set_carla_pose(
                    frame=frames[fi],
                    line_index=_safe_int(frames[fi].get("ccli"), -1),
                    x=float(mx),
                    y=float(my),
                    yaw=float(myaw),
                    dist=float(math.hypot(float(mx) - float(qx), float(my) - float(qy))),
                    source=f"intersection_cluster_{pm}_harmonize",
                    quality=str(frames[fi].get("cquality", "none")),
                )
            frames[fi]["intersection_shape_mode"] = str(pm)
            frames[fi]["intersection_shape_cluster_force"] = True

    for fi in range(n):
        if not bool(marks[fi]):
            continue
        if _mode_at(fi) in {"curve", "straight"}:
            continue
        left_mode = ""
        right_mode = ""
        li = int(fi) - 1
        while li >= 0:
            lm = _mode_at(li)
            if lm in {"curve", "straight"}:
                left_mode = lm
                break
            li -= 1
        ri = int(fi) + 1
        while ri < n:
            rm = _mode_at(ri)
            if rm in {"curve", "straight"}:
                right_mode = rm
                break
            ri += 1
        chosen = ""
        if left_mode and right_mode:
            chosen = left_mode if left_mode == right_mode else left_mode
        elif left_mode:
            chosen = left_mode
        elif right_mode:
            chosen = right_mode
        if chosen:
            frames[fi]["intersection_shape_mode"] = str(chosen)
            frames[fi]["intersection_shape_cluster_force"] = True

    # Snap boundary mode outliers to neighboring mode when one side is unlabeled.
    for fi in range(1, n - 1):
        if not bool(marks[fi]):
            continue
        cm = _mode_at(fi)
        if cm not in {"curve", "straight"}:
            continue
        pm = _mode_at(fi - 1) if bool(marks[fi - 1]) else ""
        nm = _mode_at(fi + 1) if bool(marks[fi + 1]) else ""
        target = ""
        if pm in {"curve", "straight"} and nm not in {"curve", "straight"}:
            target = pm
        elif nm in {"curve", "straight"} and pm not in {"curve", "straight"}:
            target = nm
        if target and target != cm:
            frames[fi]["intersection_shape_mode"] = str(target)
            frames[fi]["intersection_shape_cluster_force"] = True

    # Rebuild marks after local edits so the deterministic rewrite sees the
    # latest intersection-tagged frames.
    marks_final = [False] * n
    for fi in range(n):
        if _is_intersectionish_source(frames[fi]) or _mode_at(fi) in {"curve", "straight"}:
            marks_final[fi] = True
    for fi in range(1, n):
        a = frames[fi - 1]
        b = frames[fi]
        ka = _frame_semantic_lane_key(a)
        kb = _frame_semantic_lane_key(b)
        sem_switch = ka is not None and kb is not None and int(ka) != int(kb)
        ca = _safe_int(a.get("ccli"), -1)
        cb = _safe_int(b.get("ccli"), -1)
        line_switch = ca >= 0 and cb >= 0 and int(ca) != int(cb)
        if sem_switch or line_switch:
            marks_final[fi - 1] = True
            marks_final[fi] = True
    marks_final_expanded = list(marks_final)
    for fi in range(n):
        if not bool(marks_final[fi]):
            continue
        if fi > 0:
            marks_final_expanded[fi - 1] = True
        if fi + 1 < n:
            marks_final_expanded[fi + 1] = True
    marks_final = marks_final_expanded

    # Final deterministic rewrite: if a global mode is selected, apply it to
    # every marked cluster geometry so no residual mixed mode remains.
    if str(global_mode or "") in {"curve", "straight"}:
        gi = 0
        while gi < n:
            if not bool(marks_final[gi]):
                gi += 1
                continue
            gs = gi
            while gi + 1 < n and bool(marks_final[gi + 1]):
                gi += 1
            ge = gi
            if int(ge) - int(gs) + 1 < 2:
                gi += 1
                continue

            left = int(gs) - 1
            while left >= 0 and not _frame_has_carla_pose(frames[left]):
                left -= 1
            right = int(ge) + 1
            while right < n and not _frame_has_carla_pose(frames[right]):
                right += 1
            left_virtual = False
            right_virtual = False
            if left < 0:
                left = int(gs)
                left_virtual = True
            if right >= n:
                right = int(ge)
                right_virtual = True
            if int(right) - int(left) < 2:
                gi += 1
                continue

            if not bool(left_virtual) and _frame_has_carla_pose(frames[left]):
                ax = float(_safe_float(frames[left].get("cx"), float("nan")))
                ay = float(_safe_float(frames[left].get("cy"), float("nan")))
                ayaw = float(_safe_float(frames[left].get("cyaw"), float("nan")))
            else:
                ax, ay = _query_xy(int(gs))
                _, _, qyaw = _frame_query_pose(frames[int(gs)])
                ayaw = float(qyaw)
            if not bool(right_virtual) and _frame_has_carla_pose(frames[right]):
                bx = float(_safe_float(frames[right].get("cx"), float("nan")))
                by = float(_safe_float(frames[right].get("cy"), float("nan")))
                byaw = float(_safe_float(frames[right].get("cyaw"), float("nan")))
            else:
                bx, by = _query_xy(int(ge))
                _, _, qyaw = _frame_query_pose(frames[int(ge)])
                byaw = float(qyaw)
            if not (math.isfinite(ax) and math.isfinite(ay) and math.isfinite(bx) and math.isfinite(by)):
                gi += 1
                continue

            idx_all = list(range(int(left), int(right) + 1))
            q_path: List[Tuple[float, float]] = [_query_xy(k) for k in idx_all]
            cum: List[float] = [0.0]
            for k in range(1, len(q_path)):
                cum.append(float(cum[-1]) + float(math.hypot(float(q_path[k][0]) - float(q_path[k - 1][0]), float(q_path[k][1]) - float(q_path[k - 1][1]))))
            total = float(cum[-1])
            u_by_idx: Dict[int, float] = {}
            if total <= 1e-6:
                den = max(1.0, float(len(idx_all) - 1))
                for off, fi in enumerate(idx_all):
                    u_by_idx[int(fi)] = float(off) / float(den)
            else:
                for off, fi in enumerate(idx_all):
                    u_by_idx[int(fi)] = float(cum[off]) / float(total)

            chord_yaw = float(_normalize_yaw_deg(math.degrees(math.atan2(float(by) - float(ay), float(bx) - float(ax)))))
            if not math.isfinite(ayaw):
                ayaw = _safe_float(frames[gs].get("syaw"), _safe_float(frames[gs].get("yaw"), float(chord_yaw)))
            if not math.isfinite(byaw):
                byaw = _safe_float(frames[ge].get("syaw"), _safe_float(frames[ge].get("yaw"), float(chord_yaw)))
            if not math.isfinite(ayaw):
                ayaw = float(chord_yaw)
            if not math.isfinite(byaw):
                byaw = float(chord_yaw)
            span = float(math.hypot(float(bx) - float(ax), float(by) - float(ay)))
            d = max(
                float(curve_tangent_min_m),
                min(float(curve_tangent_max_m), float(curve_tangent_scale) * float(span)),
            )
            ayaw_r = math.radians(float(ayaw))
            byaw_r = math.radians(float(byaw))
            p0 = (float(ax), float(ay))
            p3 = (float(bx), float(by))
            p1 = (float(ax) + float(d) * math.cos(float(ayaw_r)), float(ay) + float(d) * math.sin(float(ayaw_r)))
            p2 = (float(bx) - float(d) * math.cos(float(byaw_r)), float(by) - float(d) * math.sin(float(byaw_r)))

            ccounts: Dict[int, int] = {}
            for fi in range(int(gs), int(ge) + 1):
                cli = _safe_int(frames[fi].get("ccli"), -1)
                if cli >= 0:
                    ccounts[int(cli)] = int(ccounts.get(int(cli), 0) + 1)
            left_cli = _safe_int(frames[left].get("ccli"), -1) if (not bool(left_virtual)) else -1
            right_cli = _safe_int(frames[right].get("ccli"), -1) if (not bool(right_virtual)) else -1
            if ccounts:
                dominant_cli = max(ccounts.keys(), key=lambda k: int(ccounts[k]))
            elif left_cli >= 0:
                dominant_cli = int(left_cli)
            else:
                dominant_cli = int(right_cli)

            for fi in range(int(gs), int(ge) + 1):
                fr = frames[fi]
                qx, qy = _query_xy(fi)
                u = float(u_by_idx.get(int(fi), 0.0))
                if str(global_mode) == "curve":
                    nx, ny = _bezier_xy(p0, p1, p2, p3, float(u))
                    dx, dy = _bezier_deriv(p0, p1, p2, p3, float(u))
                    if math.hypot(float(dx), float(dy)) <= 1e-6:
                        nyaw = float(chord_yaw)
                    else:
                        nyaw = float(_normalize_yaw_deg(math.degrees(math.atan2(float(dy), float(dx)))))
                    if left_cli >= 0 and right_cli >= 0 and left_cli != right_cli:
                        cli = int(left_cli if float(u) < 0.5 else right_cli)
                    elif left_cli >= 0:
                        cli = int(left_cli)
                    elif right_cli >= 0:
                        cli = int(right_cli)
                    else:
                        cli = int(dominant_cli)
                else:
                    nx = float(ax) + float(u) * (float(bx) - float(ax))
                    ny = float(ay) + float(u) * (float(by) - float(ay))
                    nyaw = float(chord_yaw)
                    cli = int(dominant_cli)
                nd = float(math.hypot(float(nx) - float(qx), float(ny) - float(qy)))
                _set_carla_pose(
                    frame=fr,
                    line_index=int(cli),
                    x=float(nx),
                    y=float(ny),
                    yaw=float(nyaw),
                    dist=float(nd),
                    source=(
                        "intersection_cluster_curve"
                        if str(global_mode) == "curve"
                        else "intersection_cluster_straight"
                    ),
                    quality=str(fr.get("cquality", "none")),
                )
                fr["intersection_shape_mode"] = str(global_mode)
                fr["intersection_shape_cluster_force"] = True
                fr["intersection_shape_window_start"] = int(gs)
                fr["intersection_shape_window_end"] = int(ge)

            gi += 1

        for fi in range(n):
            if not bool(marks_final[fi]):
                continue
            if _mode_at(fi) in {"curve", "straight"}:
                continue
            frames[fi]["intersection_shape_mode"] = str(global_mode)
            frames[fi]["intersection_shape_cluster_force"] = True
        for fi in range(n):
            if _mode_at(fi) in {"curve", "straight"}:
                continue
            frames[fi]["intersection_shape_mode"] = str(global_mode)
            frames[fi]["intersection_shape_cluster_force"] = True


# ---------------------------------------------------------------------------
# Deterministic intersection-connector commitment
# ---------------------------------------------------------------------------

def _build_junction_bundles(
    carla_feats: Dict[int, Dict[str, object]],
    orientation_sign_by_line: Optional[Dict[int, int]] = None,
    merge_radius_m: float = 3.0,
    min_heading_spread_deg: float = 25.0,
    max_connector_length_m: float = 55.0,
) -> List[Dict[str, object]]:
    """Build junction bundles: groups of CARLA lines sharing a start point.

    A bundle represents a fan of possible paths at a junction.  Only bundles
    whose members have >=min_heading_spread_deg angular spread are kept (to
    filter out parallel lanes on the same road segment).

    Lines longer than *max_connector_length_m* are excluded — they are
    regular road segments, not junction connectors.

    Returns list of dicts:
        {
            "center": (cx, cy),
            "lines": {line_idx: {"poly_xy": ..., "cumlen": ...,
                                  "start_heading": float, "end_heading": float,
                                  "length_m": float}},
        }
    """
    start_pts: Dict[int, Tuple[float, float]] = {}
    line_info: Dict[int, Dict[str, object]] = {}
    for li_raw, feat in carla_feats.items():
        li = int(li_raw) if isinstance(li_raw, str) else int(li_raw)
        poly = feat.get("poly_xy")
        if poly is None or not isinstance(poly, np.ndarray) or poly.shape[0] < 2:
            continue
        pxy = poly[:, :2].astype(np.float64) if poly.ndim == 2 else np.zeros((0, 2))
        if pxy.shape[0] < 2:
            continue
        sign = 0
        if isinstance(orientation_sign_by_line, dict):
            sign = _safe_int(
                orientation_sign_by_line.get(int(li), orientation_sign_by_line.get(str(int(li)), 0)),
                0,
            )
        if int(sign) < 0:
            pxy = pxy[::-1].copy()
        diffs = np.diff(pxy, axis=0)
        seg_lens = np.sqrt(np.sum(diffs * diffs, axis=1))
        cumlen = np.zeros(pxy.shape[0], dtype=np.float64)
        cumlen[1:] = np.cumsum(seg_lens)
        total_len = float(cumlen[-1])
        # Skip long road segments — they are not junction connectors.
        if total_len > float(max_connector_length_m):
            continue
        sh = math.degrees(math.atan2(float(pxy[1, 1] - pxy[0, 1]),
                                     float(pxy[1, 0] - pxy[0, 0])))
        eh = math.degrees(math.atan2(float(pxy[-1, 1] - pxy[-2, 1]),
                                     float(pxy[-1, 0] - pxy[-2, 0])))
        start_pts[li] = (float(pxy[0, 0]), float(pxy[0, 1]))
        line_info[li] = {
            "poly_xy": pxy, "cumlen": cumlen,
            "start_heading": float(sh), "end_heading": float(eh),
            "length_m": float(total_len),
            "orientation_sign": int(sign),
        }

    # Cluster by start-point proximity.
    assigned: set = set()
    raw_groups: List[List[int]] = []
    for li, (sx, sy) in sorted(start_pts.items()):
        if li in assigned:
            continue
        grp = [li]
        assigned.add(li)
        for lj, (sx2, sy2) in start_pts.items():
            if lj in assigned:
                continue
            if math.hypot(sx - sx2, sy - sy2) <= float(merge_radius_m):
                grp.append(lj)
                assigned.add(lj)
        raw_groups.append(grp)

    bundles: List[Dict[str, object]] = []
    for grp in raw_groups:
        if len(grp) < 2:
            continue
        # Check angular spread.
        headings = [float(line_info[li]["start_heading"]) for li in grp]
        max_spread = 0.0
        for ia in range(len(headings)):
            for ib in range(ia + 1, len(headings)):
                diff = abs(_normalize_yaw_deg(headings[ia] - headings[ib]))
                if diff > max_spread:
                    max_spread = diff
        if max_spread < float(min_heading_spread_deg):
            continue
        cx = float(np.mean([start_pts[li][0] for li in grp]))
        cy = float(np.mean([start_pts[li][1] for li in grp]))
        bundles.append({
            "center": (cx, cy),
            "lines": {int(li): line_info[li] for li in grp},
        })
    return bundles


def _select_best_connector(
    bundle: Dict[str, object],
    entry_heading_deg: float,
    exit_heading_deg: float,
    trajectory_xy: List[Tuple[float, float]],
    preferred_exit_lines: Optional[set] = None,
    preferred_exit_bonus: float = 0.0,
    opposite_reject_deg: float = 140.0,
) -> Optional[int]:
    """Pick the single connector from *bundle* that best fits the trajectory.

    Scoring (lower is better):
      1. Reject any line whose start heading is >opposite_reject_deg from
         entry_heading — prevents wrong-way selection.
      2. Primary: heading-match score = weighted combination of entry-heading
         match and exit-heading match.
      3. Secondary: average perpendicular distance of trajectory points to line.

    Returns the line index of the best connector, or None if all rejected.
    """
    entry_h = float(entry_heading_deg)
    exit_h = float(exit_heading_deg)
    lines = bundle.get("lines", {})
    if not lines:
        return None

    best_li: Optional[int] = None
    best_score = float("inf")
    preferred = preferred_exit_lines if isinstance(preferred_exit_lines, set) else set()

    for li, info in lines.items():
        poly_xy = info["poly_xy"]
        cumlen = info["cumlen"]
        sh = float(info["start_heading"])
        eh = float(info["end_heading"])

        # Reject wrong-way connectors.
        entry_diff = abs(_normalize_yaw_deg(entry_h - sh))
        if entry_diff > float(opposite_reject_deg):
            continue

        # Heading match: how well does the connector's arc match the
        # trajectory's entry→exit heading change?
        exit_diff = abs(_normalize_yaw_deg(exit_h - eh))
        heading_score = 0.6 * entry_diff + 0.4 * exit_diff

        # Geometric proximity: average distance of trajectory points to line.
        geo_score = 0.0
        if trajectory_xy and poly_xy.shape[0] >= 2:
            dists: List[float] = []
            for (tx, ty) in trajectory_xy:
                proj = _project_point_to_polyline_xy(poly_xy, cumlen,
                                                     float(tx), float(ty))
                if proj is not None:
                    dists.append(float(proj["dist"]))
            if dists:
                geo_score = float(np.mean(dists))

        # Combined score (heading dominates; geometry is tiebreaker).
        score = heading_score + 0.3 * geo_score
        if int(li) in preferred:
            score -= float(max(0.0, preferred_exit_bonus))
        if score < best_score:
            best_score = score
            best_li = int(li)

    return best_li


def _junction_zone_radii(
    bundles: List[Dict[str, object]],
    zone_radius_cap_m: float,
) -> List[float]:
    out: List[float] = []
    for bundle in bundles:
        lines = bundle.get("lines", {})
        lengths = [float(info.get("length_m", 0.0)) for info in lines.values()] if isinstance(lines, dict) else []
        if lengths:
            lengths_sorted = sorted(lengths)
            med = float(lengths_sorted[len(lengths_sorted) // 2])
            out.append(min(float(zone_radius_cap_m), max(8.0, med)))
        else:
            out.append(10.0)
    return out


def _detect_junction_episodes(
    frames: List[Dict[str, object]],
    bundles: List[Dict[str, object]],
    bundle_zone_radii: List[float],
    min_zone_frames: int = 3,
    approach_frames: int = 12,
    exit_frames: int = 12,
    episode_radius_scale: float = 1.8,
) -> List[Dict[str, int]]:
    n = len(frames)
    if n < 3 or not bundles:
        return []

    query_xy = [_frame_query_pose(fr)[:2] for fr in frames]

    def _dist_to_bundle(fi: int, bi: int) -> float:
        cx, cy = bundles[bi]["center"]
        qx, qy = query_xy[fi]
        return float(math.hypot(float(qx) - float(cx), float(qy) - float(cy)))

    zones: List[Tuple[int, int, int]] = []
    fi = 0
    while fi < n:
        bi_match = -1
        for bi in range(len(bundles)):
            if _dist_to_bundle(fi, bi) <= float(bundle_zone_radii[bi]):
                bi_match = int(bi)
                break
        if bi_match < 0:
            fi += 1
            continue
        s = int(fi)
        fi += 1
        while fi < n and _dist_to_bundle(fi, bi_match) <= float(bundle_zone_radii[bi_match]):
            fi += 1
        e = int(fi - 1)
        if (e - s + 1) >= int(min_zone_frames):
            zones.append((int(s), int(e), int(bi_match)))

    if not zones:
        return []

    episodes: List[Dict[str, int]] = []
    for (core_s, core_e, bi) in zones:
        radius = float(bundle_zone_radii[bi])
        episode_radius = max(radius + 2.0, float(episode_radius_scale) * radius)
        ep_s = max(0, int(core_s) - int(approach_frames))
        ep_e = min(n - 1, int(core_e) + int(exit_frames))
        while ep_s > 0 and _dist_to_bundle(ep_s - 1, bi) <= float(episode_radius):
            ep_s -= 1
        while ep_e + 1 < n and _dist_to_bundle(ep_e + 1, bi) <= float(episode_radius):
            ep_e += 1
        # Spawn-in-junction extension: if the episode starts close to frame 0,
        # pull ep_s to 0. This catches vehicles that spawn mid-turn — their
        # "approach" IS the connector arc from frame 0, not a separate straight
        # approach segment. Without this, frames [0, ep_s-1] land on the wrong
        # CCLI and downstream transition smoothers create multi-bend artifacts.
        # Safety net: the connector commit quality check (reject_commit / cand_med)
        # will reject the episode if those early frames don't fit the connector.
        _spawn_ep_s_thresh = _env_int("V2X_CARLA_SPAWN_IN_JUNCTION_MAX_EP_S", 40)
        if int(ep_s) > 0 and int(ep_s) <= int(_spawn_ep_s_thresh):
            ep_s = 0
        episodes.append(
            {
                "episode_start": int(ep_s),
                "core_start": int(core_s),
                "core_end": int(core_e),
                "episode_end": int(ep_e),
                "bundle_index": int(bi),
            }
        )

    episodes.sort(key=lambda row: int(row["episode_start"]))
    merged: List[Dict[str, int]] = []
    for row in episodes:
        if not merged:
            merged.append(dict(row))
            continue
        prev = merged[-1]
        # Never merge episodes from different bundles: temporal overlap between
        # nearby intersections is common and cross-bundle merging can collapse
        # many local episodes into an entire-track commit window.
        same_bundle = int(row.get("bundle_index", -1)) == int(prev.get("bundle_index", -2))
        if same_bundle and int(row["episode_start"]) <= int(prev["episode_end"]):
            prev["episode_end"] = max(int(prev["episode_end"]), int(row["episode_end"]))
            prev["core_end"] = max(int(prev["core_end"]), int(row["core_end"]))
            continue
        merged.append(dict(row))
    return merged


def _sample_polyline_at_s(
    poly_xy: np.ndarray,
    cumlen: np.ndarray,
    s: float,
) -> Optional[Tuple[float, float, float]]:
    if not isinstance(poly_xy, np.ndarray) or poly_xy.shape[0] < 2:
        return None
    if not isinstance(cumlen, np.ndarray) or cumlen.shape[0] != poly_xy.shape[0]:
        return None
    total = float(cumlen[-1]) if cumlen.size > 0 else 0.0
    if total <= 1e-9:
        return None
    ss = max(0.0, min(float(total), float(s)))
    idx = int(np.searchsorted(cumlen, ss, side="right") - 1)
    idx = max(0, min(idx, int(poly_xy.shape[0]) - 2))
    s0 = float(cumlen[idx])
    s1 = float(cumlen[idx + 1])
    seg = max(1e-9, s1 - s0)
    u = max(0.0, min(1.0, (ss - s0) / seg))
    x0, y0 = float(poly_xy[idx, 0]), float(poly_xy[idx, 1])
    x1, y1 = float(poly_xy[idx + 1, 0]), float(poly_xy[idx + 1, 1])
    x = x0 + u * (x1 - x0)
    y = y0 + u * (y1 - y0)
    yaw = _normalize_yaw_deg(math.degrees(math.atan2(y1 - y0, x1 - x0)))
    return (float(x), float(y), float(yaw))


def _build_extended_connector_polyline(
    base_poly_xy: np.ndarray,
    start_xy: Tuple[float, float],
    end_xy: Tuple[float, float],
    max_extend_m: float = 45.0,
) -> Optional[np.ndarray]:
    if not isinstance(base_poly_xy, np.ndarray) or base_poly_xy.shape[0] < 2:
        return None
    poly = base_poly_xy[:, :2].astype(np.float64)
    p0 = poly[0]
    p1 = poly[1]
    pn1 = poly[-2]
    pn = poly[-1]
    t0 = p1 - p0
    tn = pn - pn1
    n0 = float(np.linalg.norm(t0))
    nn = float(np.linalg.norm(tn))
    if n0 <= 1e-6 or nn <= 1e-6:
        return None
    t0 = t0 / n0
    tn = tn / nn
    sx, sy = float(start_xy[0]), float(start_xy[1])
    ex, ey = float(end_xy[0]), float(end_xy[1])
    back = min(float(max_extend_m), float(math.hypot(float(p0[0]) - sx, float(p0[1]) - sy)) + 4.0)
    fwd = min(float(max_extend_m), float(math.hypot(float(pn[0]) - ex, float(pn[1]) - ey)) + 4.0)
    pre = np.asarray([[float(p0[0] - back * t0[0]), float(p0[1] - back * t0[1])]], dtype=np.float64)
    post = np.asarray([[float(pn[0] + fwd * tn[0]), float(pn[1] + fwd * tn[1])]], dtype=np.float64)
    return np.vstack([pre, poly, post])


def _build_artificial_turn_polyline(
    start_xy: Tuple[float, float],
    end_xy: Tuple[float, float],
    entry_heading_deg: float,
    exit_heading_deg: float,
    samples: int = 36,
) -> np.ndarray:
    p0 = np.asarray([float(start_xy[0]), float(start_xy[1])], dtype=np.float64)
    p3 = np.asarray([float(end_xy[0]), float(end_xy[1])], dtype=np.float64)
    chord = float(np.linalg.norm(p3 - p0))
    if chord <= 1e-6:
        return np.asarray([p0, p3], dtype=np.float64)
    eh = math.radians(float(entry_heading_deg))
    xh = math.radians(float(exit_heading_deg))
    e_dir = np.asarray([math.cos(eh), math.sin(eh)], dtype=np.float64)
    x_dir = np.asarray([math.cos(xh), math.sin(xh)], dtype=np.float64)

    # Keep a short straight entry segment so approach does not "pre-bend"
    # before the turn starts, then connect with a single smooth cubic arc.
    entry_hold = max(1.5, min(10.0, 0.18 * chord))
    entry_hold = min(float(entry_hold), 0.45 * float(chord))
    p_hold = p0 + float(entry_hold) * e_dir
    rem = float(np.linalg.norm(p3 - p_hold))
    d_turn = max(2.0, min(12.0, 0.36 * rem))
    p1 = p_hold + d_turn * e_dir
    p2 = p3 - d_turn * x_dir

    n = max(8, int(samples))
    n_straight = max(3, min(n // 3, n - 6))
    n_curve = max(6, n - n_straight + 1)
    pts_straight = np.zeros((n_straight, 2), dtype=np.float64)
    for i in range(n_straight):
        u = float(i) / float(max(1, n_straight))
        pts_straight[i, :] = (1.0 - u) * p0 + u * p_hold

    pts_curve = np.zeros((n_curve, 2), dtype=np.float64)
    for i in range(n_curve):
        u = float(i) / float(max(1, n_curve - 1))
        om = 1.0 - u
        pts_curve[i, :] = (
            (om ** 3) * p_hold
            + 3.0 * (om ** 2) * u * p1
            + 3.0 * om * (u ** 2) * p2
            + (u ** 3) * p3
        )
    return np.vstack([pts_straight, pts_curve])


def _curvature_sign_flips_xy(
    poly_xy: np.ndarray,
    min_step_m: float = 0.2,
    min_curvature_abs: float = 0.015,
) -> int:
    if not isinstance(poly_xy, np.ndarray) or poly_xy.shape[0] < 3:
        return 0
    signs: List[int] = []
    for i in range(2, int(poly_xy.shape[0])):
        x0, y0 = float(poly_xy[i - 2, 0]), float(poly_xy[i - 2, 1])
        x1, y1 = float(poly_xy[i - 1, 0]), float(poly_xy[i - 1, 1])
        x2, y2 = float(poly_xy[i, 0]), float(poly_xy[i, 1])
        a = float(math.hypot(x1 - x0, y1 - y0))
        b = float(math.hypot(x2 - x1, y2 - y1))
        if a < float(min_step_m) or b < float(min_step_m):
            continue
        h0 = math.atan2(y1 - y0, x1 - x0)
        h1 = math.atan2(y2 - y1, x2 - x1)
        dh = math.atan2(math.sin(h1 - h0), math.cos(h1 - h0))
        k = float(dh) / max(1e-3, 0.5 * (a + b))
        if abs(k) < float(min_curvature_abs):
            continue
        signs.append(1 if k > 0.0 else -1)
    flips = 0
    for i in range(1, len(signs)):
        if int(signs[i]) != int(signs[i - 1]):
            flips += 1
    return int(flips)


def _polyline_total_abs_turn_deg(
    poly_xy: np.ndarray,
    min_step_m: float = 0.25,
) -> float:
    if not isinstance(poly_xy, np.ndarray) or poly_xy.shape[0] < 3:
        return 0.0
    total = 0.0
    for i in range(2, int(poly_xy.shape[0])):
        x0, y0 = float(poly_xy[i - 2, 0]), float(poly_xy[i - 2, 1])
        x1, y1 = float(poly_xy[i - 1, 0]), float(poly_xy[i - 1, 1])
        x2, y2 = float(poly_xy[i, 0]), float(poly_xy[i, 1])
        a = float(math.hypot(x1 - x0, y1 - y0))
        b = float(math.hypot(x2 - x1, y2 - y1))
        if a < float(min_step_m) or b < float(min_step_m):
            continue
        h0 = math.degrees(math.atan2(y1 - y0, x1 - x0))
        h1 = math.degrees(math.atan2(y2 - y1, x2 - x1))
        total += float(abs(_normalize_yaw_deg(float(h1) - float(h0))))
    return float(total)


def _polyline_major_bend_count(
    poly_xy: np.ndarray,
    min_step_m: float = 0.30,
    min_delta_deg: float = 3.5,
) -> int:
    if not isinstance(poly_xy, np.ndarray) or poly_xy.shape[0] < 3:
        return 0
    signs: List[int] = []
    for i in range(2, int(poly_xy.shape[0])):
        x0, y0 = float(poly_xy[i - 2, 0]), float(poly_xy[i - 2, 1])
        x1, y1 = float(poly_xy[i - 1, 0]), float(poly_xy[i - 1, 1])
        x2, y2 = float(poly_xy[i, 0]), float(poly_xy[i, 1])
        a = float(math.hypot(x1 - x0, y1 - y0))
        b = float(math.hypot(x2 - x1, y2 - y1))
        if a < float(min_step_m) or b < float(min_step_m):
            continue
        h0 = math.degrees(math.atan2(y1 - y0, x1 - x0))
        h1 = math.degrees(math.atan2(y2 - y1, x2 - x1))
        d = float(_normalize_yaw_deg(float(h1) - float(h0)))
        if abs(float(d)) < float(min_delta_deg):
            continue
        signs.append(1 if float(d) > 0.0 else -1)
    if not signs:
        return 0
    groups = 1
    for i in range(1, len(signs)):
        if int(signs[i]) != int(signs[i - 1]):
            groups += 1
    return int(groups)


def _bundle_exit_line_votes(
    frames: List[Dict[str, object]],
    lane_to_carla: Dict[int, Dict[str, object]],
    bundle_line_ids: set,
    core_end: int,
    episode_end: int,
    tail_frames: int = 18,
) -> Dict[int, float]:
    if not frames or not isinstance(lane_to_carla, dict) or not bundle_line_ids:
        return {}
    n = len(frames)
    s = max(0, min(int(core_end), n - 1))
    e = max(0, min(int(episode_end), n - 1))
    if e < s:
        s, e = e, s
    s = max(0, int(e) - max(4, int(tail_frames)) + 1)
    votes: Dict[int, float] = {}
    for fi in range(int(s), int(e) + 1):
        fr = frames[fi]
        lane_idx = _safe_int(fr.get("lane_index"), -1)
        if lane_idx >= 0:
            mapping = lane_to_carla.get(int(lane_idx))
            if isinstance(mapping, dict):
                ci = _safe_int(mapping.get("carla_line_index"), -1)
                if ci in bundle_line_ids:
                    q = str(mapping.get("quality", "poor")).strip().lower()
                    w = 2.0 if q in {"high", "good"} else (1.2 if q == "medium" else 0.7)
                    votes[int(ci)] = float(votes.get(int(ci), 0.0) + float(w))
        for key, w in (("cbcli", 0.35), ("ccli", 0.25)):
            ci2 = _safe_int(fr.get(key), -1)
            if ci2 in bundle_line_ids:
                votes[int(ci2)] = float(votes.get(int(ci2), 0.0) + float(w))
    return votes


def _commit_intersection_connectors(
    frames: List[Dict[str, object]],
    carla_context: Optional[Dict[str, object]] = None,
) -> None:
    """Commit one path for each approach+intersection+exit episode."""
    if not carla_context or not bool(carla_context.get("enabled")):
        return
    if _env_int("V2X_CARLA_COMMIT_INTERSECTION_CONNECTORS", 1,
                minimum=0, maximum=1) != 1:
        return
    n = len(frames)
    if n < 4:
        return

    carla_feats = carla_context.get("carla_feats", {})
    if not carla_feats:
        return
    orient_sign = _normalize_int_key_dict(carla_context.get("carla_line_orientation_sign", {}))

    bundles = _build_junction_bundles(
        carla_feats,
        orientation_sign_by_line={int(k): _safe_int(v, 0) for k, v in orient_sign.items()},
        merge_radius_m=_env_float(
            "V2X_CARLA_JUNCTION_MERGE_RADIUS_M", 8.0),
        min_heading_spread_deg=_env_float(
            "V2X_CARLA_JUNCTION_MIN_HEADING_SPREAD_DEG", 25.0),
    )
    if not bundles:
        return

    zone_radius_cap = _env_float("V2X_CARLA_JUNCTION_ZONE_RADIUS_CAP_M", 30.0)
    bundle_zone_radii = _junction_zone_radii(bundles, zone_radius_cap_m=float(zone_radius_cap))
    episodes = _detect_junction_episodes(
        frames,
        bundles,
        bundle_zone_radii,
        min_zone_frames=_env_int("V2X_CARLA_JUNCTION_MIN_ZONE_FRAMES", 3, minimum=2, maximum=20),
        approach_frames=_env_int("V2X_CARLA_JUNCTION_APPROACH_FRAMES", 12, minimum=2, maximum=80),
        exit_frames=_env_int("V2X_CARLA_JUNCTION_EXIT_FRAMES", 12, minimum=2, maximum=80),
        episode_radius_scale=_env_float("V2X_CARLA_JUNCTION_EPISODE_RADIUS_SCALE", 1.8),
    )
    if not episodes:
        return

    # Track-level query path lengths for episode-size guards.
    qpath_cum: List[float] = [0.0] * int(n)
    for qi in range(1, int(n)):
        q0x, q0y, _ = _frame_query_pose(frames[qi - 1])
        q1x, q1y, _ = _frame_query_pose(frames[qi])
        qpath_cum[qi] = float(qpath_cum[qi - 1]) + float(
            math.hypot(float(q1x) - float(q0x), float(q1y) - float(q0y))
        )
    total_track_query_path_m = float(qpath_cum[-1]) if qpath_cum else 0.0

    turn_heading_thresh = _env_float("V2X_CARLA_JUNCTION_TURN_HEADING_DEG", 28.0)
    turn_lateral_thresh = _env_float("V2X_CARLA_JUNCTION_TURN_LATERAL_M", 1.4)
    opposite_reject_deg = _env_float("V2X_CARLA_JUNCTION_OPPOSITE_REJECT_DEG", 140.0)
    connector_fit_max = _env_float("V2X_CARLA_CONNECTOR_LOCK_MAX_DIST_M", 5.0)
    max_yaw_step = _env_float("V2X_CARLA_JUNCTION_COMMIT_MAX_YAW_STEP_DEG", 34.0)
    s_adv_base = _env_float("V2X_CARLA_JUNCTION_COMMIT_S_ADV_BASE_M", 1.0)
    s_adv_scale = _env_float("V2X_CARLA_JUNCTION_COMMIT_S_ADV_SCALE", 3.8)
    lane_to_carla_raw = carla_context.get("lane_to_carla", {})
    lane_to_carla: Dict[int, Dict[str, object]] = {}
    if isinstance(lane_to_carla_raw, dict):
        for li, row in lane_to_carla_raw.items():
            if isinstance(row, dict):
                lii = _safe_int(li, -1)
                if lii >= 0:
                    lane_to_carla[int(lii)] = row
    exit_hint_tail_frames = _env_int("V2X_CARLA_JUNCTION_EXIT_HINT_TAIL_FRAMES", 18, minimum=6, maximum=40)
    exit_hint_bonus = _env_float("V2X_CARLA_JUNCTION_EXIT_HINT_BONUS", 0.9)
    spline_exit_anchor_max_dist = _env_float("V2X_CARLA_JUNCTION_SPLINE_EXIT_ANCHOR_MAX_DIST_M", 7.0)
    spline_start_anchor_max_dist = _env_float("V2X_CARLA_JUNCTION_SPLINE_START_ANCHOR_MAX_DIST_M", 4.0)
    connector_tail_offset_frames = _env_int(
        "V2X_CARLA_JUNCTION_CONNECTOR_TAIL_OFFSET_FRAMES",
        18,
        minimum=6,
        maximum=48,
    )
    connector_tail_offset_m = _env_float("V2X_CARLA_JUNCTION_CONNECTOR_TAIL_OFFSET_M", 2.5)
    connector_tail_offset_sign_ratio = _env_float("V2X_CARLA_JUNCTION_CONNECTOR_TAIL_OFFSET_SIGN_RATIO", 0.80)
    connector_tail_offset_growth = _env_float("V2X_CARLA_JUNCTION_CONNECTOR_TAIL_OFFSET_GROWTH", 1.9)
    commit_fit_max_raw_p95_m = _env_float("V2X_CARLA_JUNCTION_COMMIT_MAX_RAW_P95_M", 12.0)
    commit_fit_max_raw_median_m = _env_float("V2X_CARLA_JUNCTION_COMMIT_MAX_RAW_MEDIAN_M", 9.0)
    commit_fit_max_raw_median_gain_m = _env_float("V2X_CARLA_JUNCTION_COMMIT_MAX_RAW_MEDIAN_GAIN_M", 3.0)
    commit_fit_max_raw_median_ratio_vs_base = _env_float("V2X_CARLA_JUNCTION_COMMIT_MAX_RAW_MEDIAN_RATIO_VS_BASE", 1.8)
    commit_turn_mismatch_guard = _env_int("V2X_CARLA_JUNCTION_COMMIT_TURN_MISMATCH_GUARD", 1, minimum=0, maximum=1) == 1
    commit_turn_mismatch_query_min_deg = _env_float("V2X_CARLA_JUNCTION_COMMIT_TURN_QUERY_MIN_DEG", 35.0)
    commit_turn_mismatch_carla_max_deg = _env_float("V2X_CARLA_JUNCTION_COMMIT_TURN_CARLA_MAX_DEG", 15.0)
    max_episode_track_ratio = _env_float("V2X_CARLA_JUNCTION_COMMIT_MAX_EPISODE_TRACK_RATIO", 0.50)
    max_episode_min_track_path_m = _env_float("V2X_CARLA_JUNCTION_COMMIT_MAX_EPISODE_MIN_TRACK_PATH_M", 15.0)
    max_episode_query_path_m = _env_float("V2X_CARLA_JUNCTION_COMMIT_MAX_EPISODE_QUERY_PATH_M", 35.0)
    connector_start_max_query_dist = _env_float("V2X_CARLA_JUNCTION_CONNECTOR_START_MAX_QUERY_DIST_M", 3.0)
    connector_end_max_query_dist = _env_float("V2X_CARLA_JUNCTION_CONNECTOR_END_MAX_QUERY_DIST_M", 3.0)
    spline_start_max_query_dist = _env_float("V2X_CARLA_JUNCTION_SPLINE_START_MAX_QUERY_DIST_M", 2.5)
    spline_end_max_query_dist = _env_float("V2X_CARLA_JUNCTION_SPLINE_END_MAX_QUERY_DIST_M", 2.5)
    spline_peak_query_dist = _env_float("V2X_CARLA_JUNCTION_SPLINE_MAX_POINT_QUERY_DIST_M", 4.0)
    spline_end_lane_max_dist = _env_float("V2X_CARLA_JUNCTION_SPLINE_END_LANE_MAX_DIST_M", 2.8)
    spline_reject_if_turn_lane_available = (
        _env_int("V2X_CARLA_JUNCTION_SPLINE_REJECT_IF_TURN_LANE_AVAILABLE", 1, minimum=0, maximum=1) == 1
    )
    spline_max_major_bends = _env_int("V2X_CARLA_JUNCTION_SPLINE_MAX_MAJOR_BENDS", 1, minimum=0, maximum=4)
    spline_max_turn_extra_deg = _env_float("V2X_CARLA_JUNCTION_SPLINE_MAX_TURN_EXTRA_DEG", 40.0)
    commit_require_gain_when_base_good = (
        _env_int("V2X_CARLA_JUNCTION_COMMIT_REQUIRE_GAIN_WHEN_BASE_GOOD", 1, minimum=0, maximum=1) == 1
    )
    commit_base_good_median_m = _env_float("V2X_CARLA_JUNCTION_COMMIT_BASE_GOOD_MEDIAN_M", 1.25)
    commit_base_good_p95_m = _env_float("V2X_CARLA_JUNCTION_COMMIT_BASE_GOOD_P95_M", 2.5)
    commit_min_median_gain_m = _env_float("V2X_CARLA_JUNCTION_COMMIT_MIN_MEDIAN_GAIN_M", 0.55)

    for epi in episodes:
        ep_s = int(epi["episode_start"])
        core_s = int(epi["core_start"])
        core_e = int(epi["core_end"])
        ep_e = int(epi["episode_end"])
        bi = int(epi["bundle_index"])
        bundle = bundles[bi]
        if ep_e - ep_s + 1 < 3:
            continue
        ep_len = int(ep_e - ep_s + 1)
        core_len = int(core_e - core_s + 1)
        touches_boundary = bool(int(ep_s) <= 1 or int(ep_e) >= int(n) - 2)
        core_ratio = float(core_len) / float(max(1, ep_len))
        degenerate_core_ratio = _env_float("V2X_CARLA_JUNCTION_DEGENERATE_CORE_RATIO", 0.88)
        degenerate_track_ratio = _env_float("V2X_CARLA_JUNCTION_DEGENERATE_TRACK_RATIO", 0.80)

        commit_s = int(ep_s)
        commit_e = int(ep_e)
        ep_path_s = max(0, min(int(commit_s), int(n) - 1))
        ep_path_e = max(0, min(int(commit_e), int(n) - 1))
        if int(ep_path_e) < int(ep_path_s):
            ep_path_s, ep_path_e = ep_path_e, ep_path_s
        episode_query_path_m = float(qpath_cum[ep_path_e] - qpath_cum[ep_path_s]) if qpath_cum else 0.0
        if float(total_track_query_path_m) >= float(max_episode_min_track_path_m):
            episode_track_ratio = float(ep_len) / float(max(1, n))
            if float(episode_track_ratio) >= float(max_episode_track_ratio):
                continue
            if float(episode_query_path_m) >= float(max_episode_query_path_m):
                continue

        core_s_local = max(int(commit_s), min(int(core_s), int(commit_e)))
        core_e_local = min(int(commit_e), max(int(core_e), int(commit_s)))
        if int(core_e_local) - int(core_s_local) + 1 < 2:
            core_s_local = int(commit_s)
            core_e_local = int(commit_e)

        q0x, q0y, _ = _frame_query_pose(frames[commit_s])
        q1x, q1y, _ = _frame_query_pose(frames[commit_e])
        start_xy = (float(q0x), float(q0y))
        end_xy = (float(q1x), float(q1y))

        entry_h = _estimate_query_motion_yaw(frames, core_s_local, max_span=8, min_disp_m=0.35)
        exit_h = _estimate_query_motion_yaw(frames, core_e_local, max_span=8, min_disp_m=0.35)
        if entry_h is None or exit_h is None:
            ch = _normalize_yaw_deg(math.degrees(math.atan2(float(q1y) - float(q0y), float(q1x) - float(q0x))))
            if entry_h is None:
                entry_h = float(ch)
            if exit_h is None:
                exit_h = float(ch)

        core_xy: List[Tuple[float, float]] = []
        for fi2 in range(core_s_local, core_e_local + 1):
            qx, qy, _ = _frame_query_pose(frames[fi2])
            core_xy.append((float(qx), float(qy)))

        max_lat = 0.0
        chord_dx = float(q1x) - float(q0x)
        chord_dy = float(q1y) - float(q0y)
        chord_norm = math.hypot(chord_dx, chord_dy)
        if chord_norm > 1e-6:
            ux = chord_dx / chord_norm
            uy = chord_dy / chord_norm
            for (tx, ty) in core_xy:
                vx = float(tx) - float(q0x)
                vy = float(ty) - float(q0y)
                lat = abs(vx * (-uy) + vy * ux)
                max_lat = max(max_lat, float(lat))
        heading_delta = abs(_normalize_yaw_deg(float(exit_h) - float(entry_h)))
        turn_expected = bool(
            heading_delta >= float(turn_heading_thresh)
            or max_lat >= float(turn_lateral_thresh)
        )

        ccli_unique: set = set()
        ccli_switches = 0
        prev_cli = -1
        for fi2 in range(commit_s, commit_e + 1):
            cli = _safe_int(frames[fi2].get("ccli"), -1)
            if cli >= 0:
                ccli_unique.add(int(cli))
                if prev_cli >= 0 and int(cli) != int(prev_cli):
                    ccli_switches += 1
                prev_cli = int(cli)
        transition_evidence = bool(turn_expected or int(ccli_switches) >= 1 or int(len(ccli_unique)) >= 2)
        if bool(touches_boundary) and float(core_ratio) >= float(degenerate_core_ratio) and not bool(transition_evidence):
            continue
        if (
            float(ep_len) >= float(degenerate_track_ratio) * float(max(1, n))
            and float(core_ratio) >= 0.78
            and not bool(transition_evidence)
        ):
            continue

        bundle_lines = bundle.get("lines", {})
        bundle_line_ids: set = (
            {int(k) for k in bundle_lines.keys()}
            if isinstance(bundle_lines, dict)
            else set()
        )
        exit_votes = _bundle_exit_line_votes(
            frames=frames,
            lane_to_carla=lane_to_carla,
            bundle_line_ids=bundle_line_ids,
            core_end=int(core_e_local),
            episode_end=int(commit_e),
            tail_frames=int(exit_hint_tail_frames),
        )
        preferred_exit_lines: set = set()
        exit_hint_line = -1
        if exit_votes:
            exit_hint_line = int(max(exit_votes.keys(), key=lambda k: float(exit_votes.get(int(k), 0.0))))
            max_vote = float(max(exit_votes.values()))
            vote_floor = max(1.4, 0.60 * float(max_vote))
            preferred_exit_lines = {
                int(ci) for ci, vv in exit_votes.items() if float(vv) >= float(vote_floor)
            }

        chosen_li = _select_best_connector(
            bundle,
            float(entry_h),
            float(exit_h),
            core_xy,
            preferred_exit_lines=preferred_exit_lines,
            preferred_exit_bonus=float(exit_hint_bonus),
            opposite_reject_deg=float(opposite_reject_deg),
        )
        turn_lane_available = bool(
            (chosen_li is not None)
            or (int(exit_hint_line) >= 0 and int(exit_hint_line) in bundle_line_ids)
        )

        modal_line = -1
        freq: Dict[int, int] = {}
        for fi2 in range(commit_s, commit_e + 1):
            li = _safe_int(frames[fi2].get("ccli"), -1)
            if li >= 0:
                freq[int(li)] = int(freq.get(int(li), 0) + 1)
        if freq:
            modal_line = int(max(freq.keys(), key=lambda k: int(freq[k])))

        path_type = "straight"
        path_xy: Optional[np.ndarray] = None
        commit_line = int(chosen_li) if chosen_li is not None else int(modal_line)
        if chosen_li is not None and isinstance(bundle.get("lines"), dict):
            info = bundle["lines"].get(int(chosen_li))
            if isinstance(info, dict):
                path_xy = _build_extended_connector_polyline(
                    info["poly_xy"],
                    start_xy=start_xy,
                    end_xy=end_xy,
                    max_extend_m=_env_float("V2X_CARLA_JUNCTION_CONNECTOR_EXTEND_MAX_M", 45.0),
                )
                path_type = "connector"
                if path_xy is not None and core_xy:
                    c_cum = _polyline_cumlen_xy(path_xy)
                    c_d = []
                    c_signed = []
                    for tx, ty in core_xy:
                        proj = _project_point_to_polyline_xy(path_xy, c_cum, float(tx), float(ty))
                        if isinstance(proj, dict):
                            c_d.append(float(proj["dist"]))
                            s_sample = _sample_polyline_at_s(path_xy, c_cum, float(proj.get("s", 0.0)))
                            if isinstance(s_sample, tuple):
                                px, py, pyaw = s_sample
                                yaw_rad = math.radians(float(pyaw))
                                nx = -math.sin(yaw_rad)
                                ny = math.cos(yaw_rad)
                                c_signed.append(float((float(tx) - float(px)) * nx + (float(ty) - float(py)) * ny))
                    if c_d and float(np.mean(np.asarray(c_d, dtype=np.float64))) > float(connector_fit_max):
                        if turn_expected:
                            path_xy = None
                    if path_xy is not None and turn_expected and c_signed:
                        tail_n = max(6, min(int(connector_tail_offset_frames), int(len(c_signed))))
                        head_n = max(6, min(int(connector_tail_offset_frames), int(len(c_signed))))
                        tail = np.asarray(c_signed[-tail_n:], dtype=np.float64)
                        head = np.asarray(c_signed[:head_n], dtype=np.float64)
                        tail_abs = float(np.mean(np.abs(tail))) if tail.size > 0 else 0.0
                        head_abs = float(np.mean(np.abs(head))) if head.size > 0 else 0.0
                        pos = int(np.sum(tail >= 0.0))
                        neg = int(np.sum(tail <= 0.0))
                        sign_ratio = float(max(pos, neg)) / float(max(1, int(tail.size)))
                        growth = float(tail_abs) / float(max(0.25, head_abs))
                        if (
                            float(tail_abs) >= float(connector_tail_offset_m)
                            and float(sign_ratio) >= float(connector_tail_offset_sign_ratio)
                            and float(growth) >= float(connector_tail_offset_growth)
                        ):
                            path_xy = None

        if path_xy is None and turn_expected:
            if exit_hint_line >= 0 and isinstance(carla_feats, dict):
                qsx, qsy, _ = _frame_query_pose(frames[ep_s])
                start_anchor_set = False
                hint_rows = bundle.get("lines", {})
                hint_info = None
                if isinstance(hint_rows, dict):
                    hint_info = hint_rows.get(int(exit_hint_line))
                if isinstance(hint_info, dict):
                    hint_poly = _build_extended_connector_polyline(
                        hint_info.get("poly_xy"),
                        start_xy=start_xy,
                        end_xy=end_xy,
                        max_extend_m=_env_float("V2X_CARLA_JUNCTION_CONNECTOR_EXTEND_MAX_M", 45.0),
                    )
                    if isinstance(hint_poly, np.ndarray) and hint_poly.shape[0] >= 2:
                        hint_cum = _polyline_cumlen_xy(hint_poly)
                        hint_proj = _project_point_to_polyline_xy(
                            hint_poly,
                            hint_cum,
                            float(qsx),
                            float(qsy),
                        )
                        if isinstance(hint_proj, dict):
                            hdist = _safe_float(hint_proj.get("dist"), float("inf"))
                            hs = _safe_float(hint_proj.get("s"), 0.0)
                            hsamp = _sample_polyline_at_s(hint_poly, hint_cum, float(hs))
                            if (
                                isinstance(hsamp, tuple)
                                and math.isfinite(hdist)
                                and float(hdist) <= float(spline_start_anchor_max_dist)
                            ):
                                start_xy = (float(hsamp[0]), float(hsamp[1]))
                                entry_h = float(hsamp[2])
                                start_anchor_set = True

                if not bool(start_anchor_set):
                    start_proj = _best_projection_on_carla_line(
                        carla_context=carla_context,
                        line_index=int(exit_hint_line),
                        qx=float(qsx),
                        qy=float(qsy),
                        qyaw=float(entry_h),
                        prefer_reversed=None,
                        enforce_node_direction=False,
                        opposite_reject_deg=float(opposite_reject_deg),
                    )
                    if isinstance(start_proj, dict):
                        sproj = start_proj.get("projection", {})
                        if isinstance(sproj, dict):
                            sdist = _safe_float(sproj.get("dist"), float("inf"))
                            sx = _safe_float(sproj.get("x"), float(start_xy[0]))
                            sy = _safe_float(sproj.get("y"), float(start_xy[1]))
                            syaw = _safe_float(sproj.get("yaw"), float(entry_h))
                            if math.isfinite(sdist) and float(sdist) <= float(spline_start_anchor_max_dist):
                                start_xy = (float(sx), float(sy))
                                entry_h = float(syaw)

                qex, qey, _ = _frame_query_pose(frames[ep_e])
                hint_proj = _best_projection_on_carla_line(
                    carla_context=carla_context,
                    line_index=int(exit_hint_line),
                    qx=float(qex),
                    qy=float(qey),
                    qyaw=float(exit_h),
                    prefer_reversed=None,
                    enforce_node_direction=False,
                    opposite_reject_deg=float(opposite_reject_deg),
                )
                if isinstance(hint_proj, dict):
                    hproj = hint_proj.get("projection", {})
                    if isinstance(hproj, dict):
                        hdist = _safe_float(hproj.get("dist"), float("inf"))
                        hx = _safe_float(hproj.get("x"), float(end_xy[0]))
                        hy = _safe_float(hproj.get("y"), float(end_xy[1]))
                        hyaw = _safe_float(hproj.get("yaw"), float(exit_h))
                        if math.isfinite(hdist) and float(hdist) <= float(spline_exit_anchor_max_dist):
                            end_xy = (float(hx), float(hy))
                            exit_h = float(hyaw)
            path_xy = _build_artificial_turn_polyline(
                start_xy=start_xy,
                end_xy=end_xy,
                entry_heading_deg=float(entry_h),
                exit_heading_deg=float(exit_h),
                samples=max(24, commit_e - commit_s + 6),
            )
            path_type = "spline"
            if int(exit_hint_line) >= 0:
                commit_line = int(exit_hint_line)
            else:
                # Keep a stable lane id for spline episodes when possible so
                # downstream runtime never leaves non-transition frames detached.
                if int(commit_line) < 0:
                    qsx, qsy, qsyaw = _frame_query_pose(frames[commit_s])
                    qex, qey, qeyaw = _frame_query_pose(frames[commit_e])
                    near_start = _nearest_projection_any_line(
                        carla_context=carla_context,
                        qx=float(qsx),
                        qy=float(qsy),
                        qyaw=float(qsyaw),
                        enforce_node_direction=False,
                        opposite_reject_deg=179.0,
                    )
                    near_end = _nearest_projection_any_line(
                        carla_context=carla_context,
                        qx=float(qex),
                        qy=float(qey),
                        qyaw=float(qeyaw),
                        enforce_node_direction=False,
                        opposite_reject_deg=179.0,
                    )
                    pick = near_end if isinstance(near_end, dict) else near_start
                    if isinstance(pick, dict):
                        commit_line = int(_safe_int(pick.get("line_index"), -1))
                if int(commit_line) < 0 and int(modal_line) >= 0:
                    commit_line = int(modal_line)
        if path_xy is None:
            path_xy = np.asarray(
                [[float(start_xy[0]), float(start_xy[1])], [float(end_xy[0]), float(end_xy[1])]],
                dtype=np.float64,
            )
            path_type = "straight"
            if commit_line < 0:
                commit_line = int(modal_line)

        if not isinstance(path_xy, np.ndarray) or path_xy.shape[0] < 2:
            continue
        path_cum = _polyline_cumlen_xy(path_xy)
        if path_cum.size <= 0 or float(path_cum[-1]) <= 1e-6:
            continue

        head0 = _sample_polyline_at_s(path_xy, path_cum, 0.2)
        if head0 is not None:
            if abs(_normalize_yaw_deg(float(head0[2]) - float(entry_h))) > 100.0:
                path_xy = path_xy[::-1].copy()
                path_cum = _polyline_cumlen_xy(path_xy)

        if _curvature_sign_flips_xy(path_xy) > 1 and turn_expected:
            path_xy = _build_artificial_turn_polyline(
                start_xy=start_xy,
                end_xy=end_xy,
                entry_heading_deg=float(entry_h),
                exit_heading_deg=float(exit_h),
                samples=max(24, commit_e - commit_s + 6),
            )
            path_cum = _polyline_cumlen_xy(path_xy)
            path_type = "spline"

        def _project_rows_for_path(
            poly_xy: np.ndarray,
            poly_cum: np.ndarray,
        ) -> Tuple[List[Tuple[int, float, float, float, float]], List[float], List[float]]:
            prev_s_local: Optional[float] = None
            prev_qx_local: Optional[float] = None
            prev_qy_local: Optional[float] = None
            prev_yaw_local: Optional[float] = None
            rows_local: List[Tuple[int, float, float, float, float]] = []
            cand_raw_local: List[float] = []
            base_raw_local: List[float] = []
            for fi2 in range(commit_s, commit_e + 1):
                fr = frames[fi2]
                qx, qy, _ = _frame_query_pose(fr)
                proj = _project_point_to_polyline_xy(poly_xy, poly_cum, float(qx), float(qy))
                if not isinstance(proj, dict):
                    continue
                s_now = float(proj["s"])
                if prev_s_local is not None:
                    qstep = 0.0
                    if prev_qx_local is not None and prev_qy_local is not None:
                        qstep = float(math.hypot(float(qx) - float(prev_qx_local), float(qy) - float(prev_qy_local)))
                    max_adv = max(float(s_adv_base), float(s_adv_scale) * float(qstep) + 0.8)
                    s_now = max(float(prev_s_local), min(float(prev_s_local) + float(max_adv), float(s_now)))
                sample = _sample_polyline_at_s(poly_xy, poly_cum, s_now)
                if sample is None:
                    continue
                px, py, pyaw = sample
                if prev_yaw_local is not None:
                    dyaw = _normalize_yaw_deg(float(pyaw) - float(prev_yaw_local))
                    if abs(float(dyaw)) > float(max_yaw_step):
                        pyaw = _normalize_yaw_deg(float(prev_yaw_local) + math.copysign(float(max_yaw_step), float(dyaw)))
                pdist = float(math.hypot(float(px) - float(qx), float(py) - float(qy)))
                rows_local.append((int(fi2), float(px), float(py), float(pyaw), float(pdist)))
                rx = _safe_float(fr.get("x"), float(qx))
                ry = _safe_float(fr.get("y"), float(qy))
                cand_raw_local.append(float(math.hypot(float(px) - float(rx), float(py) - float(ry))))
                bcx = _safe_float(fr.get("cx"), float("nan"))
                bcy = _safe_float(fr.get("cy"), float("nan"))
                if math.isfinite(bcx) and math.isfinite(bcy):
                    base_raw_local.append(float(math.hypot(float(bcx) - float(rx), float(bcy) - float(ry))))
                prev_s_local = float(s_now)
                prev_qx_local = float(qx)
                prev_qy_local = float(qy)
                prev_yaw_local = float(pyaw)
            return rows_local, cand_raw_local, base_raw_local

        candidate_rows, cand_raw_dists, base_raw_dists = _project_rows_for_path(path_xy, path_cum)

        if not candidate_rows:
            continue
        if bool(commit_turn_mismatch_guard) and len(candidate_rows) >= 2:
            query_turn_abs = float(abs(_normalize_yaw_deg(float(exit_h) - float(entry_h))))
            cand_turn_abs = float(abs(_normalize_yaw_deg(float(candidate_rows[-1][3]) - float(candidate_rows[0][3]))))
            if (
                bool(turn_expected)
                and float(query_turn_abs) >= float(commit_turn_mismatch_query_min_deg)
                and float(cand_turn_abs) <= float(commit_turn_mismatch_carla_max_deg)
            ):
                path_xy_alt = _build_artificial_turn_polyline(
                    start_xy=start_xy,
                    end_xy=end_xy,
                    entry_heading_deg=float(entry_h),
                    exit_heading_deg=float(exit_h),
                    samples=max(24, commit_e - commit_s + 6),
                )
                path_cum_alt = _polyline_cumlen_xy(path_xy_alt)
                rows_alt, cand_raw_alt, base_raw_alt = _project_rows_for_path(path_xy_alt, path_cum_alt)
                if rows_alt:
                    path_xy = path_xy_alt
                    path_cum = path_cum_alt
                    path_type = "spline"
                    candidate_rows = rows_alt
                    cand_raw_dists = cand_raw_alt
                    base_raw_dists = base_raw_alt
        if not candidate_rows:
            continue
        q_start_x, q_start_y, _ = _frame_query_pose(frames[commit_s])
        q_end_x, q_end_y, _ = _frame_query_pose(frames[commit_e])
        start_qdist = float(
            math.hypot(float(candidate_rows[0][1]) - float(q_start_x), float(candidate_rows[0][2]) - float(q_start_y))
        )
        end_qdist = float(
            math.hypot(float(candidate_rows[-1][1]) - float(q_end_x), float(candidate_rows[-1][2]) - float(q_end_y))
        )
        peak_qdist = float(max((float(row[4]) for row in candidate_rows), default=0.0))
        if path_type == "connector":
            if float(start_qdist) > float(connector_start_max_query_dist):
                continue
            if float(end_qdist) > float(connector_end_max_query_dist):
                continue
        if path_type == "spline":
            if bool(turn_expected) and bool(turn_lane_available) and bool(spline_reject_if_turn_lane_available):
                continue
            if float(start_qdist) > float(spline_start_max_query_dist):
                continue
            if float(end_qdist) > float(spline_end_max_query_dist):
                continue
            if float(peak_qdist) > float(spline_peak_query_dist):
                continue
            bend_count = _polyline_major_bend_count(path_xy)
            if int(bend_count) > int(spline_max_major_bends):
                continue
            expected_turn = float(abs(_normalize_yaw_deg(float(exit_h) - float(entry_h))))
            total_turn = float(_polyline_total_abs_turn_deg(path_xy))
            if float(total_turn) > float(expected_turn) + float(spline_max_turn_extra_deg):
                continue
            end_lane_dist = float("inf")
            target_lines: List[int] = []
            if int(exit_hint_line) >= 0:
                target_lines.append(int(exit_hint_line))
            if int(commit_line) >= 0:
                target_lines.append(int(commit_line))
            if int(modal_line) >= 0:
                target_lines.append(int(modal_line))
            seen_lines: set = set()
            target_lines = [li for li in target_lines if not (li in seen_lines or seen_lines.add(li))]
            end_px = float(candidate_rows[-1][1])
            end_py = float(candidate_rows[-1][2])
            end_yaw = float(candidate_rows[-1][3])
            for li in target_lines:
                proj_line = _best_projection_on_carla_line(
                    carla_context=carla_context,
                    line_index=int(li),
                    qx=float(end_px),
                    qy=float(end_py),
                    qyaw=float(end_yaw),
                    prefer_reversed=None,
                    enforce_node_direction=False,
                    opposite_reject_deg=float(opposite_reject_deg),
                )
                if isinstance(proj_line, dict):
                    prow = proj_line.get("projection", {})
                    if isinstance(prow, dict):
                        dd = _safe_float(prow.get("dist"), float("inf"))
                        if math.isfinite(dd):
                            end_lane_dist = min(float(end_lane_dist), float(dd))
            if not math.isfinite(end_lane_dist):
                near_end = _nearest_projection_any_line(
                    carla_context=carla_context,
                    qx=float(end_px),
                    qy=float(end_py),
                    qyaw=float(end_yaw),
                    enforce_node_direction=False,
                    opposite_reject_deg=179.0,
                )
                if isinstance(near_end, dict):
                    prow = near_end.get("projection", {})
                    if isinstance(prow, dict):
                        dd = _safe_float(prow.get("dist"), float("inf"))
                        if math.isfinite(dd):
                            end_lane_dist = float(dd)
            if (not math.isfinite(end_lane_dist)) or float(end_lane_dist) > float(spline_end_lane_max_dist):
                continue
        cand_arr = np.asarray(cand_raw_dists, dtype=np.float64)
        cand_med = float(np.median(cand_arr))
        cand_p95 = float(np.percentile(cand_arr, 95.0))
        base_med = float("inf")
        base_p95 = float("inf")
        if base_raw_dists:
            base_arr = np.asarray(base_raw_dists, dtype=np.float64)
            base_med = float(np.median(base_arr))
            base_p95 = float(np.percentile(base_arr, 95.0))

        reject_commit = False
        if (
            bool(commit_require_gain_when_base_good)
            and math.isfinite(base_med)
            and math.isfinite(base_p95)
            and float(base_med) <= float(commit_base_good_median_m)
            and float(base_p95) <= float(commit_base_good_p95_m)
        ):
            # Preserve stable lane-following unless commit gives a clear gain.
            if float(cand_med) >= float(base_med) - float(commit_min_median_gain_m):
                reject_commit = True
        if float(cand_med) > float(commit_fit_max_raw_median_m) or float(cand_p95) > float(commit_fit_max_raw_p95_m):
            reject_commit = True
        if math.isfinite(base_med):
            if (
                float(cand_med) > float(base_med) + float(commit_fit_max_raw_median_gain_m)
                and float(cand_med) > float(commit_fit_max_raw_median_ratio_vs_base) * max(0.5, float(base_med))
                and (
                    (not math.isfinite(base_p95))
                    or float(cand_p95) > float(base_p95) + 1.0
                )
            ):
                reject_commit = True
        if bool(reject_commit):
            continue

        for fi2, px, py, pyaw, pdist in candidate_rows:
            fr = frames[int(fi2)]
            _set_carla_pose(
                frame=fr,
                line_index=int(commit_line),
                x=float(px),
                y=float(py),
                yaw=float(pyaw),
                dist=float(pdist),
                source=f"intersection_episode_commit_{path_type}",
                quality=str(fr.get("cquality", "none")),
            )
            fr["intersection_episode_committed"] = True
            fr["intersection_episode_start"] = int(commit_s)
            fr["intersection_episode_end"] = int(commit_e)
            fr["intersection_connector_locked"] = bool(path_type == "connector" and chosen_li is not None)
            if path_type == "connector" and chosen_li is not None:
                fr["intersection_connector_line"] = int(chosen_li)
            elif "intersection_connector_line" in fr:
                try:
                    del fr["intersection_connector_line"]
                except Exception:
                    fr["intersection_connector_line"] = -1


def _is_intersectionish_projection_frame(frame: Dict[str, object]) -> bool:
    if not isinstance(frame, dict):
        return False
    if bool(frame.get("intersection_episode_committed", False)):
        return True
    if bool(frame.get("intersection_connector_locked", False)):
        return True
    if bool(frame.get("synthetic_turn", False)):
        return True
    mode = str(frame.get("intersection_shape_mode", "")).strip().lower()
    if mode not in {"", "none", "unknown"}:
        return True
    cquality = str(frame.get("cquality", "")).strip().lower()
    if cquality == "intersection":
        return True
    csource = str(frame.get("csource", "")).strip().lower()
    if "intersection" in csource:
        return True
    return False


def _enforce_lane_attachment_outside_intersection(
    frames: List[Dict[str, object]],
    carla_context: Optional[Dict[str, object]] = None,
    role: str = "",
    opposite_reject_deg: float = 150.0,
) -> None:
    """
    Hard safety pass:
    Non-intersection, non-lane-change vehicle/ego frames must be attached to a
    valid CARLA line and pose (cx/cy/cyaw) projected onto that line.
    """
    if not isinstance(frames, list) or not frames:
        return
    role_norm = str(role or "").strip().lower()
    if role_norm not in {"vehicle", "ego"}:
        return
    if not isinstance(carla_context, dict) or not bool(carla_context.get("enabled", False)):
        return
    if _env_int("V2X_CARLA_ENFORCE_NON_INTERSECTION_LANE_ATTACHMENT", 1, minimum=0, maximum=1) != 1:
        return

    n = len(frames)
    lane_change_pre = _env_int(
        "V2X_CARLA_ENFORCE_LANE_ATTACH_LANE_CHANGE_PRE_FRAMES",
        2,
        minimum=0,
        maximum=12,
    )
    lane_change_post = _env_int(
        "V2X_CARLA_ENFORCE_LANE_ATTACH_LANE_CHANGE_POST_FRAMES",
        3,
        minimum=0,
        maximum=12,
    )
    lane_change_mask: List[bool] = [False] * int(n)
    sem_keys: List[Optional[int]] = [_frame_semantic_lane_key(fr) for fr in frames]
    for i in range(1, n):
        a = sem_keys[i - 1]
        b = sem_keys[i]
        if a is None or b is None:
            continue
        if int(a) == int(b):
            continue
        s = max(0, int(i) - int(lane_change_pre))
        e = min(int(n) - 1, int(i) + int(lane_change_post))
        for j in range(int(s), int(e) + 1):
            lane_change_mask[int(j)] = True
    for i, fr in enumerate(frames):
        if bool(fr.get("semantic_transition_sustained", False)):
            s = max(0, int(i) - int(lane_change_pre))
            e = min(int(n) - 1, int(i) + int(lane_change_post))
            for j in range(int(s), int(e) + 1):
                lane_change_mask[int(j)] = True

    unresolved = 0
    for i, fr in enumerate(frames):
        if _is_intersectionish_projection_frame(fr):
            continue
        if lane_change_mask[int(i)]:
            continue

        qx, qy, qyaw = _frame_query_pose(fr)
        cx = _safe_float(fr.get("cx"), float("nan"))
        cy = _safe_float(fr.get("cy"), float("nan"))
        cyaw = _safe_float(fr.get("cyaw"), float(qyaw))
        base_x = float(cx) if math.isfinite(cx) else float(qx)
        base_y = float(cy) if math.isfinite(cy) else float(qy)
        base_yaw = float(cyaw) if math.isfinite(cyaw) else float(qyaw)

        cand: Optional[Dict[str, object]] = None
        source = "lane_attach_nearest"
        line_idx = _safe_int(fr.get("ccli"), -1)
        if int(line_idx) >= 0:
            cand = _best_projection_on_carla_line(
                carla_context=carla_context,
                line_index=int(line_idx),
                qx=float(base_x),
                qy=float(base_y),
                qyaw=float(base_yaw),
                prefer_reversed=None,
                enforce_node_direction=True,
                opposite_reject_deg=float(opposite_reject_deg),
            )
            if cand is None:
                cand = _best_projection_on_carla_line(
                    carla_context=carla_context,
                    line_index=int(line_idx),
                    qx=float(base_x),
                    qy=float(base_y),
                    qyaw=float(base_yaw),
                    prefer_reversed=None,
                    enforce_node_direction=False,
                    opposite_reject_deg=float(opposite_reject_deg),
                )
            if cand is not None:
                source = "lane_attach_existing"

        if cand is None:
            cand = _nearest_projection_any_line(
                carla_context=carla_context,
                qx=float(base_x),
                qy=float(base_y),
                qyaw=float(base_yaw),
                enforce_node_direction=True,
                opposite_reject_deg=float(opposite_reject_deg),
            )
            if cand is None:
                cand = _nearest_projection_any_line(
                    carla_context=carla_context,
                    qx=float(base_x),
                    qy=float(base_y),
                    qyaw=float(base_yaw),
                    enforce_node_direction=False,
                    opposite_reject_deg=179.0,
                )
        if not isinstance(cand, dict):
            unresolved += 1
            continue
        cli = _safe_int(cand.get("line_index"), -1)
        proj = cand.get("projection", {})
        if int(cli) < 0 or not isinstance(proj, dict):
            unresolved += 1
            continue
        _set_carla_pose(
            frame=fr,
            line_index=int(cli),
            x=float(_safe_float(proj.get("x"), float(base_x))),
            y=float(_safe_float(proj.get("y"), float(base_y))),
            yaw=float(_safe_float(proj.get("yaw"), float(base_yaw))),
            dist=float(_safe_float(proj.get("dist"), 0.0)),
            source=str(source),
            quality=str(fr.get("cquality", "none")),
        )
        fr["strict_lane_attachment"] = True

    if unresolved > 0 and _env_int("V2X_CARLA_ENFORCE_NON_INTERSECTION_LANE_ATTACHMENT_DEBUG", 0, minimum=0, maximum=1) == 1:
        print(f"[LANE ATTACH] unresolved frames={int(unresolved)} role={role_norm}")


def _csource_origin_function(csource: str) -> str:
    src = str(csource or "").strip().lower()
    if src.startswith("intersection_episode_commit"):
        return "_commit_intersection_connectors"
    if src.startswith("intersection_connector_lock"):
        return "_commit_intersection_connectors"
    if src.startswith("intersection_cluster_"):
        return "_force_single_shape_mode_on_intersection_clusters"
    if src.startswith("intersection_shape_"):
        return "_enforce_intersection_shape_consistency"
    if src == "semantic_run_lock":
        return "_stabilize_ccli_within_semantic_runs"
    if src == "low_motion_teleport_clamp":
        return "_suppress_low_motion_teleports"
    if src.startswith("max_divergence_fallback"):
        return "_project_track_to_carla (max_divergence_fallback)"
    if "overlap" in src:
        return "_reduce_carla_overlap_with_parked_guidance"
    if src.startswith("lane_corr"):
        return "_project_track_to_carla (lane_corr projection)"
    return "_project_track_to_carla"


def _parse_actor_filter(actor_filter: str) -> Optional[set]:
    raw = str(actor_filter or "all").strip().lower()
    if raw in {"", "all", "*"}:
        return None
    out: set = set()
    for chunk in re.split(r"[,\s]+", raw):
        v = str(chunk).strip()
        if not v:
            continue
        out.add(v)
    return out if out else None


def report_intersection_episode_quality(
    tracks: List[Dict[str, object]],
    carla_context: Optional[Dict[str, object]],
    actor_filter: str = "all",
    scenario_name: str = "",
) -> List[Dict[str, object]]:
    """Detect and report bad intersection episodes."""
    if not isinstance(carla_context, dict) or not bool(carla_context.get("enabled", False)):
        print("[INTERSECTION DETECTOR] skipped: CARLA projection disabled.")
        return []
    if not isinstance(tracks, list) or not tracks:
        print("[INTERSECTION DETECTOR] skipped: no tracks.")
        return []

    carla_feats = carla_context.get("carla_feats", {})
    if not isinstance(carla_feats, dict) or not carla_feats:
        print("[INTERSECTION DETECTOR] skipped: missing carla_feats.")
        return []
    orient_sign = _normalize_int_key_dict(carla_context.get("carla_line_orientation_sign", {}))

    bundles = _build_junction_bundles(
        carla_feats,
        orientation_sign_by_line={int(k): _safe_int(v, 0) for k, v in orient_sign.items()},
        merge_radius_m=_env_float("V2X_CARLA_JUNCTION_MERGE_RADIUS_M", 8.0),
        min_heading_spread_deg=_env_float("V2X_CARLA_JUNCTION_MIN_HEADING_SPREAD_DEG", 25.0),
    )
    if not bundles:
        print("[INTERSECTION DETECTOR] skipped: no junction bundles.")
        return []

    bundle_zone_radii = _junction_zone_radii(
        bundles,
        zone_radius_cap_m=_env_float("V2X_CARLA_JUNCTION_ZONE_RADIUS_CAP_M", 30.0),
    )
    allow_ids = _parse_actor_filter(actor_filter)

    jump_thr = 2.2
    yaw_jump_thr = 70.0
    yaw_rate_thr = 260.0
    yaw_jerk_thr = 1800.0
    ccli_switch_thr = 2

    reports: List[Dict[str, object]] = []
    header = f"[INTERSECTION DETECTOR] scenario={scenario_name}" if str(scenario_name) else "[INTERSECTION DETECTOR]"
    print(header)

    for tr in tracks:
        if not isinstance(tr, dict):
            continue
        role = str(tr.get("role", "")).strip().lower()
        if role not in {"ego", "vehicle"}:
            continue
        actor_id = str(tr.get("id", "?")).strip()
        if allow_ids is not None and actor_id not in allow_ids:
            continue
        frames = tr.get("frames", [])
        if not isinstance(frames, list) or len(frames) < 4:
            continue

        episodes = _detect_junction_episodes(
            frames,
            bundles,
            bundle_zone_radii,
            min_zone_frames=_env_int("V2X_CARLA_JUNCTION_MIN_ZONE_FRAMES", 3, minimum=2, maximum=20),
            approach_frames=_env_int("V2X_CARLA_JUNCTION_APPROACH_FRAMES", 12, minimum=2, maximum=80),
            exit_frames=_env_int("V2X_CARLA_JUNCTION_EXIT_FRAMES", 12, minimum=2, maximum=80),
            episode_radius_scale=_env_float("V2X_CARLA_JUNCTION_EPISODE_RADIUS_SCALE", 1.8),
        )
        if not episodes:
            continue

        for ep_idx, epi in enumerate(episodes, start=1):
            s = int(epi["episode_start"])
            e = int(epi["episode_end"])
            if e - s < 2:
                continue

            ccli_vals: List[int] = []
            ccli_switch_frames: List[int] = []
            wrong_way_raw_frames: List[int] = []
            reversal_frames: List[int] = []
            max_jump = 0.0
            max_jump_frame = -1
            max_yaw_jump = 0.0
            max_yaw_jump_frame = -1
            max_yaw_rate = 0.0
            max_yaw_rate_frame = -1
            max_yaw_jerk = 0.0
            max_yaw_jerk_frame = -1
            prev_rate: Optional[float] = None
            path_xy: List[Tuple[float, float]] = []
            query_xy: List[Tuple[float, float]] = []
            teleport_frames: List[int] = []

            for fi in range(s, e + 1):
                fr = frames[fi]
                if _frame_has_carla_pose(fr):
                    x = _safe_float(fr.get("cx"), float("nan"))
                    y = _safe_float(fr.get("cy"), float("nan"))
                else:
                    x, y, _ = _frame_query_pose(fr)
                path_xy.append((float(x), float(y)))
                qx, qy, _ = _frame_query_pose(fr)
                query_xy.append((float(qx), float(qy)))
                ccli_vals.append(_safe_int(fr.get("ccli"), -1))

            for fi in range(s + 1, e + 1):
                fa = frames[fi - 1]
                fb = frames[fi]
                ax, ay = path_xy[fi - s - 1]
                bx, by = path_xy[fi - s]
                step = float(math.hypot(float(bx) - float(ax), float(by) - float(ay)))
                if step > max_jump:
                    max_jump = float(step)
                    max_jump_frame = int(fi)
                qax, qay = query_xy[fi - s - 1]
                qbx, qby = query_xy[fi - s]
                raw_step = float(math.hypot(float(qbx) - float(qax), float(qby) - float(qay)))
                if step > float(jump_thr) and step > float(raw_step) + 0.9:
                    teleport_frames.append(int(fi))

                ya = _safe_float(fa.get("cyaw"), _safe_float(fa.get("syaw"), _safe_float(fa.get("yaw"), 0.0)))
                yb = _safe_float(fb.get("cyaw"), _safe_float(fb.get("syaw"), _safe_float(fb.get("yaw"), 0.0)))
                yj = abs(_normalize_yaw_deg(float(yb) - float(ya)))
                if yj > max_yaw_jump:
                    max_yaw_jump = float(yj)
                    max_yaw_jump_frame = int(fi)
                dt = max(1e-3, _safe_float(fb.get("t"), float(fi) * 0.1) - _safe_float(fa.get("t"), float(fi - 1) * 0.1))
                rate = float(yj) / float(dt)
                if rate > max_yaw_rate:
                    max_yaw_rate = float(rate)
                    max_yaw_rate_frame = int(fi)
                if prev_rate is not None:
                    jerk = abs(float(rate) - float(prev_rate)) / float(dt)
                    if jerk > max_yaw_jerk:
                        max_yaw_jerk = float(jerk)
                        max_yaw_jerk_frame = int(fi)
                prev_rate = float(rate)

                cli_a = _safe_int(fa.get("ccli"), -1)
                cli_b = _safe_int(fb.get("ccli"), -1)
                if cli_a >= 0 and cli_b >= 0 and cli_a != cli_b:
                    ccli_switch_frames.append(int(fi))

                if fi >= s + 2:
                    px, py = path_xy[fi - s - 2]
                    ux, uy = float(ax) - float(px), float(ay) - float(py)
                    vx, vy = float(bx) - float(ax), float(by) - float(ay)
                    nu = float(math.hypot(ux, uy))
                    nv = float(math.hypot(vx, vy))
                    if nu > 0.12 and nv > 0.12:
                        dot = (ux * vx + uy * vy) / max(1e-9, nu * nv)
                        if float(dot) < -0.2:
                            reversal_frames.append(int(fi))

                motion_h = None
                if step > 0.12:
                    motion_h = _normalize_yaw_deg(math.degrees(math.atan2(float(by) - float(ay), float(bx) - float(ax))))
                else:
                    motion_h = _estimate_query_motion_yaw(frames, fi, max_span=4, min_disp_m=0.2)
                if motion_h is not None:
                    cli = _safe_int(fb.get("ccli"), -1)
                    if cli >= 0:
                        line_h = _carla_line_calibrated_yaw_at_point(
                            carla_context=carla_context,
                            line_index=int(cli),
                            x=float(bx),
                            y=float(by),
                        )
                        if line_h is not None:
                            if (
                                float(step) >= 0.6
                                and math.cos(
                                    math.radians(float(_normalize_yaw_deg(float(motion_h) - float(line_h))))
                                ) < 0.0
                            ):
                                wrong_way_raw_frames.append(int(fi))

            wrong_way_frames: List[int] = []
            if wrong_way_raw_frames:
                run: List[int] = [int(wrong_way_raw_frames[0])]
                for fi in wrong_way_raw_frames[1:]:
                    if int(fi) == int(run[-1]) + 1:
                        run.append(int(fi))
                        continue
                    if len(run) >= 3:
                        wrong_way_frames.extend(run)
                    run = [int(fi)]
                if len(run) >= 3:
                    wrong_way_frames.extend(run)

            path_arr = np.asarray(path_xy, dtype=np.float64)
            curvature_flips = _curvature_sign_flips_xy(path_arr)
            unique_ccli = sorted({int(v) for v in ccli_vals if int(v) >= 0})
            unique_ccli_count = int(len(unique_ccli))

            query_h0 = _estimate_query_motion_yaw(frames, s, max_span=6, min_disp_m=0.3)
            query_h1 = _estimate_query_motion_yaw(frames, e, max_span=6, min_disp_m=0.3)
            carla_h0 = _estimate_query_motion_yaw(
                [{"sx": p[0], "sy": p[1], "syaw": 0.0} for p in path_xy], 0, max_span=6, min_disp_m=0.3
            )
            carla_h1 = _estimate_query_motion_yaw(
                [{"sx": p[0], "sy": p[1], "syaw": 0.0} for p in path_xy], len(path_xy) - 1, max_span=6, min_disp_m=0.3
            )
            query_turn = abs(_normalize_yaw_deg(float(query_h1 or 0.0) - float(query_h0 or 0.0)))
            carla_turn = abs(_normalize_yaw_deg(float(carla_h1 or 0.0) - float(carla_h0 or 0.0)))

            failures: List[str] = []
            if teleport_frames:
                failures.append("position_jump")
            if (
                max_yaw_jump > float(yaw_jump_thr)
                or max_yaw_rate > float(yaw_rate_thr)
                or max_yaw_jerk > float(yaw_jerk_thr)
            ):
                failures.append("yaw_discontinuity")
            if reversal_frames:
                failures.append("direction_reversal")
            if wrong_way_frames:
                failures.append("wrong_way")
            if curvature_flips > 1:
                failures.append("multi_bend")
            if len(ccli_switch_frames) > int(ccli_switch_thr) or unique_ccli_count > 3:
                failures.append("ccli_oscillation")
            if query_turn >= 35.0 and carla_turn <= 15.0:
                failures.append("straight_instead_of_turn")
            if query_turn <= 15.0 and carla_turn >= 35.0:
                failures.append("turn_instead_of_straight")
            if not failures:
                failures = ["ok"]

            first_frame = -1
            first_reason = "ok"
            for fi in range(s + 1, e + 1):
                reasons: List[str] = []
                ax, ay = path_xy[fi - s - 1]
                bx, by = path_xy[fi - s]
                step = float(math.hypot(float(bx) - float(ax), float(by) - float(ay)))
                if fi in teleport_frames:
                    reasons.append("position_jump")
                fa = frames[fi - 1]
                fb = frames[fi]
                ya = _safe_float(fa.get("cyaw"), _safe_float(fa.get("syaw"), _safe_float(fa.get("yaw"), 0.0)))
                yb = _safe_float(fb.get("cyaw"), _safe_float(fb.get("syaw"), _safe_float(fb.get("yaw"), 0.0)))
                yj = abs(_normalize_yaw_deg(float(yb) - float(ya)))
                dt = max(1e-3, _safe_float(fb.get("t"), float(fi) * 0.1) - _safe_float(fa.get("t"), float(fi - 1) * 0.1))
                rate = float(yj) / float(dt)
                if yj > float(yaw_jump_thr) or rate > float(yaw_rate_thr):
                    reasons.append("yaw_discontinuity")
                if fi in wrong_way_frames and fi >= (s + 5):
                    reasons.append("wrong_way")
                if fi in reversal_frames and fi >= (s + 3):
                    reasons.append("direction_reversal")
                if reasons:
                    first_frame = int(fi)
                    first_reason = str(reasons[0])
                    break

            src = ""
            fn = ""
            if first_frame >= 0:
                src = str(frames[first_frame].get("csource", ""))
                fn = _csource_origin_function(src)

            report_row = {
                "actor_id": actor_id,
                "episode_index": int(ep_idx),
                "frame_range": (int(s), int(e)),
                "core_range": (int(epi["core_start"]), int(epi["core_end"])),
                "unique_ccli_count": int(unique_ccli_count),
                "ccli_switch_frames": list(ccli_switch_frames),
                "max_jump_m": float(max_jump),
                "max_jump_frame": int(max_jump_frame),
                "max_yaw_jump_deg": float(max_yaw_jump),
                "max_yaw_jump_frame": int(max_yaw_jump_frame),
                "max_yaw_rate_deg_s": float(max_yaw_rate),
                "max_yaw_rate_frame": int(max_yaw_rate_frame),
                "max_yaw_jerk_deg_s2": float(max_yaw_jerk),
                "max_yaw_jerk_frame": int(max_yaw_jerk_frame),
                "curvature_sign_flips": int(curvature_flips),
                "wrong_way": bool(len(wrong_way_frames) > 0),
                "wrong_way_frames": list(wrong_way_frames),
                "failure_types": list(failures),
                "first_violation_frame": int(first_frame),
                "first_violation_reason": str(first_reason),
                "first_violation_source": str(src),
                "first_violation_function": str(fn),
            }
            reports.append(report_row)

            print(
                f"  actor={actor_id} episode={ep_idx} frames={s}-{e} core={epi['core_start']}-{epi['core_end']} "
                f"ccli_unique={unique_ccli_count} switches={ccli_switch_frames}"
            )
            print(
                f"    max_jump={max_jump:.2f}@{max_jump_frame} "
                f"yaw_jump={max_yaw_jump:.1f}@{max_yaw_jump_frame} "
                f"yaw_rate={max_yaw_rate:.1f}@{max_yaw_rate_frame} "
                f"yaw_jerk={max_yaw_jerk:.1f}@{max_yaw_jerk_frame}"
            )
            print(
                f"    curvature_flips={curvature_flips} wrong_way={bool(wrong_way_frames)} "
                f"failure={','.join(failures)}"
            )
            if first_frame >= 0:
                print(
                    f"    first_violation=frame{first_frame} reason={first_reason} "
                    f"source={src} function={fn}"
                )

    if not reports:
        print("  no intersection episodes detected for selected actors.")
    return reports


def _stabilize_ccli_within_semantic_runs(
    frames: List[Dict[str, object]],
    carla_context: Optional[Dict[str, object]] = None,
    min_run_frames: int = 4,
    max_lock_dist_m: float = 5.0,
    opposite_reject_deg: float = 150.0,
) -> None:
    """
    Within each stable V2XPNP semantic lane run (constant assigned_lane_id +
    road_id), lock to the single most-common CARLA line index (modal ccli).
    This suppresses:
    - Intersection oscillation: vehicle snaps to multiple connector segments
    - Oscillating lane changes: vehicle switches between parallel lanes and back

    Only overrides if re-projection onto the modal line stays within
    max_lock_dist_m. Skips synthetic_turn frames (already handled by shape
    consistency functions).
    """
    if len(frames) < min_run_frames:
        return
    if not isinstance(carla_context, dict):
        return
    if _env_int("V2X_CARLA_STABILIZE_SEMANTIC_RUNS_ENABLED", 1, minimum=0, maximum=1) != 1:
        return

    max_lock_dist_m = float(_env_float("V2X_CARLA_SEMANTIC_RUN_LOCK_MAX_DIST_M", max_lock_dist_m))
    min_run_frames = int(_env_int("V2X_CARLA_SEMANTIC_RUN_LOCK_MIN_FRAMES", min_run_frames, minimum=2, maximum=40))
    max_dist_increase_m = float(
        _env_float("V2X_CARLA_SEMANTIC_RUN_LOCK_MAX_DIST_INCREASE_M", 0.65)
    )

    n = len(frames)

    # Build semantic runs: consecutive frames with same (road_id, assigned_lane_id)
    runs: List[Tuple[int, int]] = []  # (start, end) inclusive
    i = 0
    while i < n:
        fr = frames[i]
        if not _frame_has_carla_pose(fr):
            i += 1
            continue
        if bool(fr.get("intersection_episode_committed", False)):
            i += 1
            continue
        # Skip synthetic turn frames — shape functions own them
        if bool(fr.get("synthetic_turn", False)):
            i += 1
            continue
        sem_key = _frame_semantic_lane_key(fr)
        if sem_key is None:
            i += 1
            continue
        s = i
        while i + 1 < n:
            nxt = frames[i + 1]
            if not _frame_has_carla_pose(nxt):
                break
            if bool(nxt.get("intersection_episode_committed", False)):
                break
            if bool(nxt.get("synthetic_turn", False)):
                break
            nk = _frame_semantic_lane_key(nxt)
            if nk != sem_key:
                break
            i += 1
        e = i
        if (e - s + 1) >= int(min_run_frames):
            runs.append((int(s), int(e)))
        i += 1

    for s, e in runs:
        # Count frequency of each ccli in the run
        freq: Dict[int, int] = {}
        for fi in range(s, e + 1):
            li = _safe_int(frames[fi].get("ccli"), -1)
            if li >= 0:
                freq[li] = freq.get(li, 0) + 1
        if not freq:
            continue
        modal_li = max(freq, key=lambda k: freq[k])
        # If already uniform, nothing to do
        unique_lines = set(freq.keys())
        if len(unique_lines) <= 1:
            continue
        # Override frames with non-modal ccli by re-projecting onto modal line
        for fi in range(s, e + 1):
            fr = frames[fi]
            if not _frame_has_carla_pose(fr):
                continue
            if _safe_int(fr.get("ccli"), -1) == modal_li:
                continue
            qx, qy, qyaw = _frame_query_pose(fr)
            proj_cand = _best_projection_on_carla_line(
                carla_context=carla_context,
                line_index=int(modal_li),
                qx=float(qx),
                qy=float(qy),
                qyaw=float(qyaw),
                prefer_reversed=None,
                enforce_node_direction=True,
                opposite_reject_deg=float(opposite_reject_deg),
            )
            if proj_cand is None:
                # Try without direction enforcement as fallback
                proj_cand = _best_projection_on_carla_line(
                    carla_context=carla_context,
                    line_index=int(modal_li),
                    qx=float(qx),
                    qy=float(qy),
                    qyaw=float(qyaw),
                    prefer_reversed=None,
                    enforce_node_direction=False,
                    opposite_reject_deg=float(opposite_reject_deg),
                )
            if proj_cand is None:
                continue
            proj = proj_cand.get("projection", {})
            new_dist = float(_safe_float(proj.get("dist"), float("inf")))
            if not math.isfinite(new_dist) or new_dist > float(max_lock_dist_m):
                continue
            cur_dist = float(_safe_float(fr.get("cdist"), float("inf")))
            if math.isfinite(cur_dist) and new_dist > float(cur_dist) + float(max_dist_increase_m):
                # Run-lock should suppress oscillation, not force a materially
                # worse local fit that can create visible one-step jumps later.
                continue
            _set_carla_pose(
                frame=fr,
                line_index=int(modal_li),
                x=float(proj["x"]),
                y=float(proj["y"]),
                yaw=float(proj["yaw"]),
                dist=float(new_dist),
                source="semantic_run_lock",
                quality=str(fr.get("cquality", "none")),
            )


def _log_carla_ccli_changes(
    frames: List[Dict[str, object]],
    actor_label: str = "actor",
) -> None:
    """
    Print a compact summary of CARLA line (ccli) changes for debugging.
    Enabled by setting V2X_CARLA_DEBUG_ACTOR_IDS to a comma-separated list
    of actor IDs (or '*' for all actors).
    """
    import os as _os
    debug_ids_raw = str(_os.environ.get("V2X_CARLA_DEBUG_ACTOR_IDS", "")).strip()
    if not debug_ids_raw:
        return
    if debug_ids_raw != "*":
        allowed = set(v.strip() for v in debug_ids_raw.split(",") if v.strip())
        if str(actor_label) not in allowed:
            return

    changes: List[str] = []
    n = len(frames)
    for i in range(1, n):
        a = frames[i - 1]
        b = frames[i]
        if not _frame_has_carla_pose(a) or not _frame_has_carla_pose(b):
            continue
        ca = _safe_int(a.get("ccli"), -1)
        cb = _safe_int(b.get("ccli"), -1)
        if ca < 0 or cb < 0 or ca == cb:
            continue
        da = _safe_float(a.get("cdist"), float("nan"))
        db = _safe_float(b.get("cdist"), float("nan"))
        src_a = str(a.get("csource", ""))
        src_b = str(b.get("csource", ""))
        sem_a = _frame_semantic_lane_key(a)
        sem_b = _frame_semantic_lane_key(b)
        changes.append(
            f"  frame {i-1}→{i}: ccli {ca}→{cb}  "
            f"dist {da:.2f}→{db:.2f}  "
            f"sem {sem_a}→{sem_b}  "
            f"src [{src_a}]→[{src_b}]"
        )

    if not changes:
        print(f"[CCLI DEBUG] {actor_label}: no ccli changes")
    else:
        print(f"[CCLI DEBUG] {actor_label}: {len(changes)} ccli changes over {n} frames:")
        for line in changes:
            print(line)


def _suppress_low_motion_teleports(
    frames: List[Dict[str, object]],
    raw_step_threshold_m: float = 0.6,
    jump_threshold_m: float = 1.8,
    target_floor_m: float = 1.2,
    target_scale: float = 2.0,
    target_bias_m: float = 0.4,
) -> None:
    """
    Final safety pass: clamp one-step CARLA jumps during low-motion windows.
    """
    if len(frames) < 2:
        return

    def _raw_step(fi: int) -> float:
        if fi <= 0 or fi >= len(frames):
            return 0.0
        a = frames[fi - 1]
        b = frames[fi]
        ax = _safe_float(a.get("x"), float("nan"))
        ay = _safe_float(a.get("y"), float("nan"))
        bx = _safe_float(b.get("x"), float("nan"))
        by = _safe_float(b.get("y"), float("nan"))
        if math.isfinite(ax) and math.isfinite(ay) and math.isfinite(bx) and math.isfinite(by):
            return float(math.hypot(float(bx) - float(ax), float(by) - float(ay)))
        qax, qay, _ = _frame_query_pose(a)
        qbx, qby, _ = _frame_query_pose(b)
        return float(math.hypot(float(qbx) - float(qax), float(qby) - float(qay)))

    def _semantic_transition_sustained(fi: int, min_frames: int = 2) -> bool:
        if fi <= 0 or fi >= len(frames):
            return False
        prev_k = _frame_semantic_lane_key(frames[fi - 1])
        curr_k = _frame_semantic_lane_key(frames[fi])
        if prev_k is None or curr_k is None or int(prev_k) == int(curr_k):
            return False
        need = max(1, int(min_frames)) - 1
        if need <= 0:
            return True
        seen = 0
        j = int(fi) + 1
        while j < len(frames) and seen < int(need):
            a_k = _frame_semantic_lane_key(frames[j - 1])
            b_k = _frame_semantic_lane_key(frames[j])
            if a_k is None or b_k is None or int(a_k) == int(b_k):
                break
            seen += 1
            j += 1
        return bool(seen >= int(need))

    for i in range(1, len(frames)):
        prev = frames[i - 1]
        cur = frames[i]
        if not (_frame_has_carla_pose(prev) and _frame_has_carla_pose(cur)):
            continue
        if bool(_semantic_transition_sustained(int(i), min_frames=2)):
            continue
        raw_step = float(_raw_step(int(i)))
        if float(raw_step) >= float(raw_step_threshold_m):
            continue
        px = float(_safe_float(prev.get("cx"), 0.0))
        py = float(_safe_float(prev.get("cy"), 0.0))
        cx = float(_safe_float(cur.get("cx"), 0.0))
        cy = float(_safe_float(cur.get("cy"), 0.0))
        step = float(math.hypot(float(cx) - float(px), float(cy) - float(py)))
        if float(step) <= float(jump_threshold_m):
            continue
        if float(step) <= 1e-6:
            continue

        target = min(
            float(jump_threshold_m),
            max(float(target_floor_m), float(target_scale) * float(raw_step) + float(target_bias_m)),
        )
        scale = float(target) / max(1e-6, float(step))
        nx = float(px) + (float(cx) - float(px)) * float(scale)
        ny = float(py) + (float(cy) - float(py)) * float(scale)
        qx, qy, _ = _frame_query_pose(cur)
        cur["cx"] = float(nx)
        cur["cy"] = float(ny)
        cur["cdist"] = float(math.hypot(float(nx) - float(qx), float(ny) - float(qy)))
        prev_cli = _safe_int(prev.get("ccli"), -1)
        cur_cli = _safe_int(cur.get("ccli"), -1)
        if (
            prev_cli >= 0
            and cur_cli >= 0
            and int(prev_cli) != int(cur_cli)
            and (not bool(_semantic_transition_sustained(int(i), min_frames=2)))
        ):
            # Keep line-id continuity when we are explicitly clamping a
            # low-motion teleport without a sustained semantic transition.
            cur["ccli"] = int(prev_cli)
        cur["csource"] = "low_motion_teleport_clamp"
        cur["low_motion_teleport_clamp"] = True
        dx = float(nx) - float(px)
        dy = float(ny) - float(py)
        if math.hypot(dx, dy) > 1e-6:
            cur["cyaw"] = float(_normalize_yaw_deg(math.degrees(math.atan2(dy, dx))))


def _fix_tail_low_motion_line_spike(
    frames: List[Dict[str, object]],
    raw_step_threshold_m: float = 0.6,
    max_tail_run_frames: int = 2,
    min_prev_run_frames: int = 2,
) -> None:
    """
    Final line-id guard for short tail spikes introduced by late smoothing.
    Keeps geometry unchanged and only restores line-id continuity when the
    tail switch occurs in a low-motion window.
    """
    n = len(frames)
    if n < 2:
        return

    def _raw_step(fi: int) -> float:
        if fi <= 0 or fi >= n:
            return 0.0
        a = frames[fi - 1]
        b = frames[fi]
        ax = _safe_float(a.get("x"), float("nan"))
        ay = _safe_float(a.get("y"), float("nan"))
        bx = _safe_float(b.get("x"), float("nan"))
        by = _safe_float(b.get("y"), float("nan"))
        if math.isfinite(ax) and math.isfinite(ay) and math.isfinite(bx) and math.isfinite(by):
            return float(math.hypot(float(bx) - float(ax), float(by) - float(ay)))
        qax, qay, _ = _frame_query_pose(a)
        qbx, qby, _ = _frame_query_pose(b)
        return float(math.hypot(float(qbx) - float(qax), float(qby) - float(qay)))

    end = int(n - 1)
    while end > 0 and (not _frame_has_carla_pose(frames[end])):
        end -= 1
    if end <= 0:
        return
    tail_line = _safe_int(frames[end].get("ccli"), -1)
    if tail_line < 0:
        return

    start = int(end)
    while start - 1 >= 0 and _safe_int(frames[start - 1].get("ccli"), -1) == int(tail_line):
        start -= 1
    tail_len = int(end - start + 1)
    if tail_len <= 0 or tail_len > int(max(1, int(max_tail_run_frames))):
        return
    if start <= 0:
        return

    prev_idx = int(start - 1)
    while prev_idx >= 0 and _safe_int(frames[prev_idx].get("ccli"), -1) < 0:
        prev_idx -= 1
    if prev_idx < 0:
        return
    prev_line = _safe_int(frames[prev_idx].get("ccli"), -1)
    if prev_line < 0 or int(prev_line) == int(tail_line):
        return

    prev_start = int(prev_idx)
    while prev_start - 1 >= 0 and _safe_int(frames[prev_start - 1].get("ccli"), -1) == int(prev_line):
        prev_start -= 1
    prev_len = int(prev_idx - prev_start + 1)
    if prev_len < int(max(1, int(min_prev_run_frames))):
        return

    prev_sem = _frame_semantic_lane_key(frames[prev_idx])
    tail_sem = _frame_semantic_lane_key(frames[start])
    if prev_sem is not None and tail_sem is not None and int(prev_sem) != int(tail_sem):
        return

    boundary_step = float(_raw_step(int(start)))
    if boundary_step > float(raw_step_threshold_m):
        return
    tail_raw_max = 0.0
    for fi in range(int(start), int(end) + 1):
        tail_raw_max = max(float(tail_raw_max), float(_raw_step(int(fi))))
    if tail_raw_max > float(raw_step_threshold_m):
        return

    for fi in range(int(start), int(end) + 1):
        fr = frames[fi]
        fr["ccli"] = int(prev_line)
        fr["csource"] = "tail_low_motion_line_hold"
        fr["tail_low_motion_line_hold"] = True


def _fallback_wrong_way_carla_samples(
    frames: List[Dict[str, object]],
    opposite_reject_deg: float = 178.0,
    carla_context: Optional[Dict[str, object]] = None,
    canonical_reject_deg: float = 181.0,
    try_reproject: bool = True,
) -> None:
    """
    Final wrong-way guard: if CARLA snapped yaw is near-opposite to actor motion
    direction, demote to query-pose fallback for that frame.
    """
    if len(frames) < 2:
        return

    skip_sources = {
        "raw_fallback",
        "raw_fallback_far",
        "raw_fallback_wrong_way",
        "non_vehicle_raw",
    }
    eligible_sources = {
        "lane_corr",
        "lane_corr_split",
        "poor_corr_semantic_hold_prev",
        "nearest",
        "nearest_override",
        "nearest_gap",
        "nearest_continuity",
        "jump_guard_prev_line",
        "disconnect_guard_prev_line",
        "semantic_line_id_hold",
        "transition_window_smooth",
        "semantic_straight_hold_prev",
        "transition_spike_hold_prev",
    }
    guard_all_sources = _env_int("V2X_CARLA_WRONG_WAY_GUARD_ALL_SOURCES", 1, minimum=0, maximum=1) == 1
    require_persistence = _env_int("V2X_CARLA_WRONG_WAY_REQUIRE_PERSISTENCE", 0, minimum=0, maximum=1) == 1
    min_query_step_for_demote = _env_float("V2X_CARLA_WRONG_WAY_MIN_QUERY_STEP_M", 0.45)
    reject_deg = float(opposite_reject_deg)
    canonical_reject = float(canonical_reject_deg)
    has_carla_context = isinstance(carla_context, dict) and bool(carla_context)
    skip_parked_edge = _env_int("V2X_CARLA_WRONG_WAY_SKIP_PARKED_EDGE", 1, minimum=0, maximum=1) == 1

    diff_cache: List[Optional[float]] = [None] * len(frames)
    canonical_diff_cache: List[Optional[float]] = [None] * len(frames)
    motion_yaw_cache: List[Optional[float]] = [None] * len(frames)

    def _motion_yaw_with_neighbor_fallback(fi: int) -> Optional[float]:
        yaw = _estimate_query_motion_yaw(frames, int(fi))
        if yaw is not None:
            return float(yaw)
        # Tail/head low-motion duplicates can miss local yaw; borrow from the
        # nearest nearby frame with valid motion evidence.
        for span in range(1, 4):
            pj = int(fi) - int(span)
            if pj >= 0:
                py = _estimate_query_motion_yaw(frames, int(pj))
                if py is not None:
                    return float(py)
            nj = int(fi) + int(span)
            if nj < len(frames):
                ny = _estimate_query_motion_yaw(frames, int(nj))
                if ny is not None:
                    return float(ny)
        return None

    for i, fr in enumerate(frames):
        if not _frame_has_carla_pose(fr):
            continue
        src = str(fr.get("csource", "")).strip()
        if bool(skip_parked_edge) and src.startswith("parked_edge"):
            continue
        if src in skip_sources:
            continue
        if (not bool(guard_all_sources)) and src not in eligible_sources:
            continue
        line_idx = _safe_int(fr.get("ccli"), -1)
        cyaw = _safe_float(fr.get("cyaw"), float("nan"))
        motion_yaw = _motion_yaw_with_neighbor_fallback(int(i))
        if motion_yaw is None:
            continue
        motion_yaw_cache[i] = float(motion_yaw)
        if math.isfinite(cyaw):
            diff_cache[i] = float(_yaw_abs_diff_deg(float(cyaw), float(motion_yaw)))
        if has_carla_context and line_idx >= 0:
            cx = _safe_float(fr.get("cx"), float("nan"))
            cy = _safe_float(fr.get("cy"), float("nan"))
            if math.isfinite(cx) and math.isfinite(cy):
                line_yaw = _carla_line_calibrated_yaw_at_point(
                    carla_context=carla_context,
                    line_index=int(line_idx),
                    x=float(cx),
                    y=float(cy),
                )
                if line_yaw is not None:
                    canonical_diff_cache[i] = float(
                        _yaw_abs_diff_deg(float(line_yaw), float(motion_yaw))
                    )

    for i, fr in enumerate(frames):
        if not _frame_has_carla_pose(fr):
            continue
        src = str(fr.get("csource", "")).strip()
        if bool(skip_parked_edge) and src.startswith("parked_edge"):
            continue
        if src in skip_sources:
            continue
        if bool(fr.get("synthetic_turn", False)):
            continue
        diff = diff_cache[i]
        canonical_diff = canonical_diff_cache[i]
        is_wrong_by_cyaw = bool(diff is not None and float(diff) >= reject_deg)
        is_wrong_by_canonical = bool(
            canonical_diff is not None and float(canonical_diff) >= canonical_reject
        )
        if not (is_wrong_by_cyaw or is_wrong_by_canonical):
            continue
        if bool(require_persistence):
            # Optional persistence check to avoid one-frame wrong-way demotions.
            persistent = False
            for j in (i - 1, i + 1):
                if j < 0 or j >= len(frames):
                    continue
                dj = diff_cache[j]
                cdj = canonical_diff_cache[j]
                if (
                    (dj is not None and float(dj) >= reject_deg)
                    or (cdj is not None and float(cdj) >= canonical_reject)
                ):
                    persistent = True
                    break
            if not persistent:
                continue
        local_step = 0.0
        if i > 0:
            qx0, qy0, _ = _frame_query_pose(frames[i - 1])
            qx1, qy1, _ = _frame_query_pose(frames[i])
            local_step = max(local_step, float(math.hypot(float(qx1) - float(qx0), float(qy1) - float(qy0))))
        if i + 1 < len(frames):
            qx1, qy1, _ = _frame_query_pose(frames[i])
            qx2, qy2, _ = _frame_query_pose(frames[i + 1])
            local_step = max(local_step, float(math.hypot(float(qx2) - float(qx1), float(qy2) - float(qy1))))
        if float(local_step) < float(min_query_step_for_demote):
            continue
        qx, qy, qyaw = _frame_query_pose(fr)
        motion_yaw = motion_yaw_cache[i]
        if bool(try_reproject) and has_carla_context:
            qyaw_for_proj = float(motion_yaw if motion_yaw is not None else qyaw)
            cand = _nearest_projection_any_line(
                carla_context=carla_context,
                qx=float(qx),
                qy=float(qy),
                qyaw=float(qyaw_for_proj),
                enforce_node_direction=True,
                opposite_reject_deg=float(opposite_reject_deg),
                allow_reversed=False,
            )
            if cand is not None and isinstance(cand.get("projection"), dict):
                proj = cand.get("projection", {})
                cand_line = _safe_int(cand.get("line_index"), -1)
                cand_dist = _safe_float(proj.get("dist"), float("inf"))
                prev_dist = _safe_float(fr.get("cdist"), float("inf"))
                cand_line_yaw = None
                if cand_line >= 0:
                    cand_line_yaw = _carla_line_calibrated_yaw_at_point(
                        carla_context=carla_context,
                        line_index=int(cand_line),
                        x=_safe_float(proj.get("x"), float(qx)),
                        y=_safe_float(proj.get("y"), float(qy)),
                    )
                cand_ok = True
                if motion_yaw is not None and cand_line_yaw is not None:
                    if float(_yaw_abs_diff_deg(float(cand_line_yaw), float(motion_yaw))) >= canonical_reject:
                        cand_ok = False
                cand_proj_yaw = _safe_float(proj.get("yaw"), float("nan"))
                if motion_yaw is not None and math.isfinite(cand_proj_yaw):
                    if float(_yaw_abs_diff_deg(float(cand_proj_yaw), float(motion_yaw))) >= reject_deg:
                        cand_ok = False
                if math.isfinite(cand_dist) and math.isfinite(prev_dist):
                    if cand_dist > max(8.0, float(prev_dist) + 3.0):
                        cand_ok = False
                if cand_ok:
                    _set_carla_pose(
                        frame=fr,
                        line_index=int(cand_line),
                        x=float(_safe_float(proj.get("x"), float(qx))),
                        y=float(_safe_float(proj.get("y"), float(qy))),
                        yaw=float(_safe_float(proj.get("yaw"), qyaw_for_proj)),
                        dist=float(cand_dist if math.isfinite(cand_dist) else 0.0),
                        source="wrong_way_reproject",
                        quality="none",
                    )
                    continue
        # Hard fallback only when current CARLA yaw is opposite to observed
        # motion. Canonical-only mismatches can be map-orientation artifacts.
        if bool(is_wrong_by_cyaw):
            _set_carla_pose(
                frame=fr,
                line_index=-1,
                x=float(qx),
                y=float(qy),
                yaw=float(qyaw),
                dist=0.0,
                source="raw_fallback_wrong_way",
                quality="none",
            )


def _project_track_to_carla(
    frames: List[Dict[str, object]],
    role: str,
    carla_context: Dict[str, object],
    track_meta: Optional[Dict[str, object]] = None,
) -> None:
    if not frames:
        return

    role_norm = str(role or "").strip().lower()
    if role_norm not in {"ego", "vehicle"}:
        for fr in frames:
            qx, qy, qyaw = _frame_query_pose(fr)
            _set_carla_pose(
                frame=fr,
                line_index=-1,
                x=float(qx),
                y=float(qy),
                yaw=float(qyaw),
                dist=0.0,
                source="non_vehicle_raw",
                quality="none",
            )
        _snapshot_carla_pose_frames(frames, prefix="cb")
        return

    # Hard wrong-way gate for driving actors: never snap onto a CARLA line whose
    # node direction is near-opposite to actor travel direction.
    CARLA_OPPOSITE_DIRECTION_REJECT_DEG = _env_float("V2X_CARLA_OPPOSITE_REJECT_DEG", 165.0)
    if role_norm == "ego":
        CARLA_OPPOSITE_DIRECTION_REJECT_DEG = min(
            float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG),
            float(_env_float("V2X_CARLA_OPPOSITE_REJECT_EGO_DEG", 120.0)),
        )

    # Nearest-override guardrails for lane-correspondence projections.
    CORR_SOFT_DIST_GOOD = _env_float("V2X_CARLA_CORR_SOFT_DIST_GOOD", 3.2)
    CORR_SOFT_DIST_WEAK = _env_float("V2X_CARLA_CORR_SOFT_DIST_WEAK", 2.4)
    CORR_HARD_DIST_GOOD = _env_float("V2X_CARLA_CORR_HARD_DIST_GOOD", 7.0)
    CORR_HARD_DIST_WEAK = _env_float("V2X_CARLA_CORR_HARD_DIST_WEAK", 5.4)
    CORR_SCORE_MARGIN_GOOD = _env_float("V2X_CARLA_CORR_SCORE_MARGIN_GOOD", 0.85)
    CORR_SCORE_MARGIN_WEAK = _env_float("V2X_CARLA_CORR_SCORE_MARGIN_WEAK", 0.55)
    CORR_DIST_MARGIN_GOOD = _env_float("V2X_CARLA_CORR_DIST_MARGIN_GOOD", 0.90)
    CORR_DIST_MARGIN_WEAK = _env_float("V2X_CARLA_CORR_DIST_MARGIN_WEAK", 0.60)
    CORR_HARD_BAD_MARGIN = _env_float("V2X_CARLA_CORR_HARD_BAD_MARGIN", 0.35)
    CORR_POOR_DIST_GATE = _env_float("V2X_CARLA_CORR_POOR_DIST_GATE", 3.0)
    CORR_POOR_NEAR_MARGIN = _env_float("V2X_CARLA_CORR_POOR_NEAR_MARGIN", 0.35)
    CORR_RELAXED_OVERRIDE_MIN_GAIN = _env_float("V2X_CARLA_CORR_RELAXED_OVERRIDE_MIN_GAIN_M", 1.0)
    CORR_RELAXED_OVERRIDE_NEAR_MAX = _env_float("V2X_CARLA_CORR_RELAXED_OVERRIDE_NEAR_MAX_M", 2.4)
    CORR_RELAXED_OVERRIDE_SCORE_GAIN = _env_float("V2X_CARLA_CORR_RELAXED_OVERRIDE_SCORE_GAIN", 0.20)

    # Temporal continuity prior slack for nearest-based assignment.
    NEAREST_CONT_SCORE_SLACK = _env_float("V2X_CARLA_NEAREST_CONT_SCORE_SLACK", 0.60)
    NEAREST_CONT_DIST_SLACK = _env_float("V2X_CARLA_NEAREST_CONT_DIST_SLACK", 0.80)
    SPLIT_AMBIGUITY_PENALTY = _env_float("V2X_CARLA_SPLIT_AMBIGUITY_PENALTY", 1.20)
    DISCONNECTED_SWITCH_PENALTY = _env_float("V2X_CARLA_DISCONNECTED_SWITCH_PENALTY", 2.40)
    DISCONNECTED_OVERRIDE_DIST_GAIN = _env_float("V2X_CARLA_DISCONNECTED_OVERRIDE_DIST_GAIN", 1.40)
    DISCONNECTED_OVERRIDE_SCORE_GAIN = _env_float("V2X_CARLA_DISCONNECTED_OVERRIDE_SCORE_GAIN", 1.80)
    DISCONNECTED_HOLD_SCORE_SLACK = _env_float("V2X_CARLA_DISCONNECTED_HOLD_SCORE_SLACK", 1.15)
    DISCONNECTED_HOLD_DIST_SLACK = _env_float("V2X_CARLA_DISCONNECTED_HOLD_DIST_SLACK", 1.45)
    POOR_CORR_TRANSITION_HOLD_ENABLED = (
        _env_int("V2X_CARLA_POOR_CORR_TRANSITION_HOLD_ENABLED", 1, minimum=0, maximum=1) == 1
    )
    POOR_CORR_TRANSITION_HOLD_MAX_RAW_STEP = _env_float("V2X_CARLA_POOR_CORR_TRANSITION_HOLD_MAX_RAW_STEP_M", 1.4)
    POOR_CORR_TRANSITION_HOLD_MAX_YAW_DELTA = _env_float("V2X_CARLA_POOR_CORR_TRANSITION_HOLD_MAX_YAW_DELTA_DEG", 20.0)
    POOR_CORR_TRANSITION_HOLD_SCORE_SLACK = _env_float("V2X_CARLA_POOR_CORR_TRANSITION_HOLD_SCORE_SLACK", 1.8)
    POOR_CORR_TRANSITION_HOLD_DIST_SLACK = _env_float("V2X_CARLA_POOR_CORR_TRANSITION_HOLD_DIST_SLACK", 1.6)
    LOW_MOTION_RAW_STEP_MAX = _env_float("V2X_CARLA_LOW_MOTION_RAW_STEP_MAX_M", 0.6)
    LOW_MOTION_OVERRIDE_MIN_DIST_GAIN = _env_float("V2X_CARLA_LOW_MOTION_OVERRIDE_MIN_DIST_GAIN_M", 1.2)
    LOW_MOTION_OVERRIDE_MIN_SCORE_GAIN = _env_float("V2X_CARLA_LOW_MOTION_OVERRIDE_MIN_SCORE_GAIN", 0.85)
    LOW_MOTION_OVERRIDE_MAX_JUMP_FLOOR = _env_float("V2X_CARLA_LOW_MOTION_OVERRIDE_MAX_JUMP_FLOOR_M", 1.2)
    LOW_MOTION_OVERRIDE_MAX_JUMP_SCALE = _env_float("V2X_CARLA_LOW_MOTION_OVERRIDE_MAX_JUMP_SCALE", 2.0)
    LOW_MOTION_OVERRIDE_MAX_JUMP_BIAS = _env_float("V2X_CARLA_LOW_MOTION_OVERRIDE_MAX_JUMP_BIAS_M", 0.4)
    SEMANTIC_TRANSITION_SUSTAIN_FRAMES = _env_int("V2X_CARLA_SEMANTIC_TRANSITION_SUSTAIN_FRAMES", 2, minimum=1, maximum=4)
    NEAREST_CONT_JUMP_MARGIN = _env_float("V2X_CARLA_NEAREST_CONT_JUMP_MARGIN_M", 0.15)
    WEAK_SWITCH_GUARD_ENABLED = (
        _env_int("V2X_CARLA_WEAK_SWITCH_GUARD_ENABLED", 1, minimum=0, maximum=1) == 1
    )
    WEAK_SWITCH_GUARD_RAW_STEP_MAX_M = _env_float("V2X_CARLA_WEAK_SWITCH_GUARD_RAW_STEP_MAX_M", 0.9)
    WEAK_SWITCH_GUARD_MIN_DIST_GAIN_M = _env_float("V2X_CARLA_WEAK_SWITCH_GUARD_MIN_DIST_GAIN_M", 1.1)
    WEAK_SWITCH_GUARD_MIN_SCORE_GAIN = _env_float("V2X_CARLA_WEAK_SWITCH_GUARD_MIN_SCORE_GAIN", 0.9)
    WEAK_SWITCH_GUARD_MAX_PREV_DIST_M = _env_float("V2X_CARLA_WEAK_SWITCH_GUARD_MAX_PREV_DIST_M", 4.5)
    SKIP_POOR_REVERSED_CORR_WHEN_SEMANTIC_UNKNOWN = (
        _env_int("V2X_CARLA_SKIP_POOR_REVERSED_CORR_WHEN_SEMANTIC_UNKNOWN", 1, minimum=0, maximum=1) == 1
    )
    SEMANTIC_BRANCH_CONTINUITY_GUARD = (
        _env_int("V2X_CARLA_SEMANTIC_BRANCH_CONTINUITY_GUARD", 1, minimum=0, maximum=1) == 1
    )
    SEMANTIC_BRANCH_DIST_SLACK_M = _env_float("V2X_CARLA_SEMANTIC_BRANCH_DIST_SLACK_M", 3.0)
    SEMANTIC_BRANCH_SCORE_SLACK = _env_float("V2X_CARLA_SEMANTIC_BRANCH_SCORE_SLACK", 3.0)
    SEMANTIC_BRANCH_DEDICATED_PREF_M = _env_float("V2X_CARLA_SEMANTIC_BRANCH_DEDICATED_PREF_M", 0.8)

    lane_to_carla_raw = carla_context.get("lane_to_carla", {})
    lane_to_carla: Dict[int, Dict[str, object]] = {}
    if isinstance(lane_to_carla_raw, dict):
        for li, row in lane_to_carla_raw.items():
            if isinstance(row, dict):
                lane_to_carla[int(li)] = row
    carla_feats = carla_context.get("carla_feats", {})
    carla_successors = _normalize_int_set_map(carla_context.get("carla_successors", {}))

    def _is_carla_line_connected(ci_a: int, ci_b: int) -> bool:
        if ci_a < 0 or ci_b < 0:
            return False
        if ci_a == ci_b:
            return True
        if not isinstance(carla_feats, dict) or not carla_feats:
            return False
        return bool(
            ytm._carla_lines_connected(
                int(ci_a),
                int(ci_b),
                carla_feats,
                carla_successors,
                threshold_m=6.0,
            )
        )

    def _semantic_lane_key(fr: Dict[str, object]) -> Optional[Tuple[int, int]]:
        rid = _safe_int(fr.get("road_id"), -1)
        assigned_lid = _safe_int(fr.get("assigned_lane_id"), 0)
        lane_id = _safe_int(fr.get("lane_id"), 0)
        if rid >= 0:
            if assigned_lid != 0:
                return (int(rid), int(assigned_lid))
            if lane_id != 0:
                return (int(rid), int(lane_id))
        return None

    def _has_semantic_lane_transition(prev_fr: Dict[str, object], curr_fr: Dict[str, object]) -> bool:
        pk = _semantic_lane_key(prev_fr)
        ck = _semantic_lane_key(curr_fr)
        if pk is not None and ck is not None:
            return bool(pk != ck)
        prev_assigned = _safe_int(prev_fr.get("assigned_lane_id"), 0)
        curr_assigned = _safe_int(curr_fr.get("assigned_lane_id"), 0)
        if prev_assigned != 0 and curr_assigned != 0:
            return bool(prev_assigned != curr_assigned)
        prev_idx = _safe_int(prev_fr.get("lane_index"), -1)
        curr_idx = _safe_int(curr_fr.get("lane_index"), -1)
        if prev_idx >= 0 and curr_idx >= 0:
            return bool(prev_idx != curr_idx)
        return False

    def _semantic_transition_sustained(frame_idx: int, min_frames: int) -> bool:
        if frame_idx <= 0 or frame_idx >= len(frames):
            return False
        if not _has_semantic_lane_transition(frames[frame_idx - 1], frames[frame_idx]):
            return False
        need = max(1, int(min_frames)) - 1
        if need <= 0:
            return True
        seen = 0
        j = int(frame_idx) + 1
        while j < len(frames) and seen < int(need):
            if _has_semantic_lane_transition(frames[j - 1], frames[j]):
                seen += 1
                j += 1
                continue
            break
        return bool(seen >= int(need))

    # Metadata for CARLA lines used to discourage weak split snaps onto
    # geometrically ambiguous shared lines when a dedicated primary line exists.
    carla_line_unique_match_count: Dict[int, int] = {}
    carla_line_label_set: Dict[int, set] = {}
    carla_line_lane_id_by_index: Dict[int, int] = {}
    carla_line_road_id_by_index: Dict[int, int] = {}
    carla_line_outermost_by_index: Dict[int, bool] = {}
    road_side_max_abs_lane: Dict[Tuple[int, int], int] = {}
    carla_lines_data = carla_context.get("carla_lines_data", [])
    if isinstance(carla_lines_data, list):
        for row in carla_lines_data:
            if not isinstance(row, dict):
                continue
            ci = _safe_int(row.get("index"), -1)
            if ci < 0:
                continue
            labels = row.get("matched_v2_labels", [])
            uniq_count = 0
            label_set: set = set()
            if isinstance(labels, (list, tuple, set)):
                label_set = {str(v).strip() for v in labels if str(v).strip()}
                uniq_count = len(label_set)
            if uniq_count <= 0:
                uniq_count = max(1, _safe_int(row.get("matched_v2_count"), 0))
            carla_line_unique_match_count[int(ci)] = int(max(1, uniq_count))
            if label_set:
                carla_line_label_set[int(ci)] = label_set
            road_id = _safe_int(row.get("road_id"), 0)
            lane_id = _safe_int(row.get("lane_id"), 0)
            if road_id != 0 and lane_id != 0:
                carla_line_road_id_by_index[int(ci)] = int(road_id)
                carla_line_lane_id_by_index[int(ci)] = int(lane_id)
                side = 1 if int(lane_id) > 0 else -1
                key = (int(road_id), int(side))
                road_side_max_abs_lane[key] = max(
                    int(road_side_max_abs_lane.get(key, 0)),
                    int(abs(int(lane_id))),
                )
        for ci, road_id in carla_line_road_id_by_index.items():
            lane_id = int(carla_line_lane_id_by_index.get(int(ci), 0))
            if lane_id == 0:
                continue
            side = 1 if int(lane_id) > 0 else -1
            max_abs = int(road_side_max_abs_lane.get((int(road_id), int(side)), abs(int(lane_id))))
            carla_line_outermost_by_index[int(ci)] = bool(int(abs(int(lane_id))) >= int(max_abs))

    def _line_label_matches_prefix(line_idx: int, prefix: str) -> bool:
        if int(line_idx) < 0 or not prefix:
            return False
        labels = carla_line_label_set.get(int(line_idx), set())
        return any(str(lbl).startswith(str(prefix)) for lbl in labels)

    def _line_label_count(line_idx: int) -> int:
        if int(line_idx) < 0:
            return 0
        labels = carla_line_label_set.get(int(line_idx), set())
        if labels:
            return int(len(labels))
        return int(carla_line_unique_match_count.get(int(line_idx), 1))

    parked_raw_anchor_enabled = _env_int("V2X_CARLA_PARKED_RAW_ANCHOR_ENABLED", 1, minimum=0, maximum=1) == 1
    parked_raw_anchor_assign_line_max_dist_m = _env_float("V2X_CARLA_PARKED_RAW_ANCHOR_ASSIGN_LINE_MAX_DIST_M", 3.0)
    parked_raw_anchor_min_frames = _env_int("V2X_CARLA_PARKED_RAW_ANCHOR_MIN_FRAMES", 8, minimum=2, maximum=200)
    parked_raw_anchor_motion_yaw_span = _env_int("V2X_CARLA_PARKED_RAW_ANCHOR_YAW_SPAN", 5, minimum=1, maximum=12)
    parked_raw_anchor_motion_min_disp_m = _env_float("V2X_CARLA_PARKED_RAW_ANCHOR_YAW_MIN_DISP_M", 0.08)
    parked_edge_offset_target_m = _env_float("V2X_CARLA_PARKED_EDGE_OFFSET_TARGET_M", 1.55)
    parked_edge_offset_max_m = _env_float("V2X_CARLA_PARKED_EDGE_OFFSET_MAX_M", 2.4)
    parked_edge_offset_query_sign_min_m = _env_float("V2X_CARLA_PARKED_EDGE_OFFSET_QUERY_SIGN_MIN_M", 0.06)
    parked_edge_freeze_enabled = _env_int("V2X_CARLA_PARKED_EDGE_FREEZE_ENABLED", 1, minimum=0, maximum=1) == 1
    parked_edge_min_query_err_p95_m = _env_float("V2X_CARLA_PARKED_EDGE_MIN_QUERY_ERR_P95_M", 2.0)
    parked_edge_min_query_err_med_m = _env_float("V2X_CARLA_PARKED_EDGE_MIN_QUERY_ERR_MED_M", 0.9)
    parked_edge_require_outermost = _env_int("V2X_CARLA_PARKED_EDGE_REQUIRE_OUTERMOST", 1, minimum=0, maximum=1) == 1
    parked_edge_outermost_ratio_min = _env_float("V2X_CARLA_PARKED_EDGE_OUTERMOST_RATIO_MIN", 0.95)
    parked_edge_nearest_med_max_m = _env_float("V2X_CARLA_PARKED_EDGE_NEAREST_MED_MAX_M", 2.0)
    parked_low_motion_meta = bool(
        isinstance(track_meta, dict) and bool(track_meta.get("low_motion_vehicle", False))
    )
    parked_track_ref = track_meta if isinstance(track_meta, dict) else {"role": "vehicle", "frames": frames}
    parked_like_track = False
    if role_norm == "vehicle":
        # Lazy import: runtime_postprocess imports runtime_projection, so
        # module-level import would be circular.
        from v2xpnp.pipeline.runtime_postprocess import _is_parked_vehicle_track_for_overlap as _ipvtfo  # noqa: F811
        parked_like_track = bool(_ipvtfo(parked_track_ref))

    # Parked / near-static actors should sit near road edges, not lane centers.
    # Use CARLA lane geometry for the base pose and only use raw/query as a side
    # hint. This avoids dependence on absolute raw map offset.
    if bool(parked_raw_anchor_enabled) and bool(parked_like_track) and len(frames) >= int(parked_raw_anchor_min_frames):
        q_raw_errs: List[float] = []
        for fr in frames:
            qx, qy, _ = _frame_query_pose(fr)
            rx = _safe_float(fr.get("x"), float(qx))
            ry = _safe_float(fr.get("y"), float(qy))
            q_raw_errs.append(float(math.hypot(float(rx) - float(qx), float(ry) - float(qy))))
        if q_raw_errs:
            q_err_arr = np.asarray(q_raw_errs, dtype=np.float64)
            q_err_p95 = float(np.percentile(q_err_arr, 95.0))
            q_err_med = float(np.median(q_err_arr))
            if (
                float(q_err_p95) < float(parked_edge_min_query_err_p95_m)
                and float(q_err_med) < float(parked_edge_min_query_err_med_m)
                and (not bool(parked_low_motion_meta))
            ):
                # If query/raw are already close, parked edge forcing is usually
                # unnecessary and can introduce local overlap regressions.
                pass
            else:
                parked_rows: List[Tuple[float, float, float, int, float]] = []
                nearest_dist_vals: List[float] = []
                outer_hits = 0
                outer_total = 0
                for fi, fr in enumerate(frames):
                    qx, qy, qyaw = _frame_query_pose(fr)
                    yaw_est = _estimate_query_motion_yaw(
                        frames,
                        int(fi),
                        max_span=int(parked_raw_anchor_motion_yaw_span),
                        min_disp_m=float(parked_raw_anchor_motion_min_disp_m),
                    )
                    ryaw = float(yaw_est if yaw_est is not None else _safe_float(fr.get("yaw"), float(qyaw)))
                    nearest_raw = _nearest_projection_any_line(
                        carla_context=carla_context,
                        qx=float(qx),
                        qy=float(qy),
                        qyaw=float(ryaw),
                        enforce_node_direction=False,
                        opposite_reject_deg=float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG),
                    )
                    cli = -1
                    cdist = 0.0
                    px = float(qx)
                    py = float(qy)
                    pyaw = float(ryaw)
                    if isinstance(nearest_raw, dict):
                        proj = nearest_raw.get("projection", {})
                        d_raw = _safe_float(proj.get("dist"), float("inf"))
                        line_idx = int(_safe_int(nearest_raw.get("line_index"), -1))
                        is_outer = bool(carla_line_outermost_by_index.get(int(line_idx), False))
                        lane_id = int(carla_line_lane_id_by_index.get(int(line_idx), 0))
                        if math.isfinite(d_raw):
                            nearest_dist_vals.append(float(d_raw))
                        if bool(parked_edge_require_outermost):
                            outer_total += 1
                            if bool(is_outer):
                                outer_hits += 1
                        ppx = _safe_float(proj.get("x"), float(qx))
                        ppy = _safe_float(proj.get("y"), float(qy))
                        pyaw = _safe_float(proj.get("yaw"), float(ryaw))
                        nyaw = math.radians(float(pyaw))
                        tx = math.cos(float(nyaw))
                        ty = math.sin(float(nyaw))
                        nx = -float(ty)
                        ny = float(tx)
                        # For confirmed outermost lanes, force edge offset outward
                        # (away from inner lanes) instead of between-lane placement.
                        if bool(is_outer) and int(lane_id) != 0:
                            side = -1.0 if int(lane_id) > 0 else 1.0
                        else:
                            rx = _safe_float(fr.get("x"), float(qx))
                            ry = _safe_float(fr.get("y"), float(qy))
                            lat_hint = (float(rx) - float(ppx)) * float(nx) + (float(ry) - float(ppy)) * float(ny)
                            if abs(float(lat_hint)) < float(parked_edge_offset_query_sign_min_m):
                                lat_hint = (float(qx) - float(ppx)) * float(nx) + (float(qy) - float(ppy)) * float(ny)
                            side = 1.0 if float(lat_hint) >= 0.0 else -1.0
                        lat_query = (float(qx) - float(ppx)) * float(nx) + (float(qy) - float(ppy)) * float(ny)
                        desired = max(float(parked_edge_offset_target_m), abs(float(lat_query)))
                        desired = min(float(parked_edge_offset_max_m), float(desired))
                        px = float(ppx) + float(side) * float(desired) * float(nx)
                        py = float(ppy) + float(side) * float(desired) * float(ny)
                        if math.isfinite(d_raw) and float(d_raw) <= float(parked_raw_anchor_assign_line_max_dist_m):
                            cli = int(line_idx)
                            cdist = float(d_raw)
                    parked_rows.append((float(px), float(py), float(_normalize_yaw_deg(pyaw)), int(cli), float(cdist)))

                park_edge_ok = True
                if bool(parked_edge_require_outermost):
                    if int(outer_total) <= 0:
                        park_edge_ok = False
                    else:
                        outer_ratio = float(outer_hits) / float(max(1, int(outer_total)))
                        if float(outer_ratio) < float(parked_edge_outermost_ratio_min):
                            park_edge_ok = False
                if nearest_dist_vals:
                    nd_med = float(np.median(np.asarray(nearest_dist_vals, dtype=np.float64)))
                    if float(nd_med) > float(parked_edge_nearest_med_max_m):
                        park_edge_ok = False
                if not bool(park_edge_ok):
                    parked_rows = []

                if not parked_rows:
                    pass
                else:
                    if bool(parked_edge_freeze_enabled):
                        pxs = np.asarray([float(r[0]) for r in parked_rows], dtype=np.float64)
                        pys = np.asarray([float(r[1]) for r in parked_rows], dtype=np.float64)
                        yaws = np.asarray([math.radians(float(r[2])) for r in parked_rows], dtype=np.float64)
                        cli_vals = [int(r[3]) for r in parked_rows if int(r[3]) >= 0]
                        cdist_vals = [float(r[4]) for r in parked_rows if math.isfinite(float(r[4]))]
                        ax = float(np.median(pxs))
                        ay = float(np.median(pys))
                        ayaw = float(_normalize_yaw_deg(math.degrees(math.atan2(float(np.mean(np.sin(yaws))), float(np.mean(np.cos(yaws)))))))
                        acli = int(max(set(cli_vals), key=cli_vals.count)) if cli_vals else -1
                        acdist = float(np.median(np.asarray(cdist_vals, dtype=np.float64))) if cdist_vals else 0.0
                        for fr in frames:
                            _set_carla_pose(
                                frame=fr,
                                line_index=int(acli),
                                x=float(ax),
                                y=float(ay),
                                yaw=float(ayaw),
                                dist=float(acdist),
                                source="parked_edge_static",
                                quality="none",
                            )
                        return

                    for fr, (px, py, pyaw, cli, cdist) in zip(frames, parked_rows):
                        _set_carla_pose(
                            frame=fr,
                            line_index=int(cli),
                            x=float(px),
                            y=float(py),
                            yaw=float(_normalize_yaw_deg(pyaw)),
                            dist=float(cdist),
                            source="parked_edge_offset",
                            quality="none",
                        )
                    return

    for fi, fr in enumerate(frames):
        qx, qy, qyaw = _frame_query_pose(fr)
        qyaw_for_proj = float(qyaw)
        prev_fr = frames[fi - 1] if fi > 0 else None
        prev_line = _safe_int(prev_fr.get("ccli"), -1) if isinstance(prev_fr, dict) else -1
        lane_transition = (
            _has_semantic_lane_transition(prev_fr, fr)
            if isinstance(prev_fr, dict)
            else False
        )
        lane_transition_sustained = (
            _semantic_transition_sustained(int(fi), int(SEMANTIC_TRANSITION_SUSTAIN_FRAMES))
            if bool(lane_transition)
            else False
        )
        fr["semantic_transition_sustained"] = bool(lane_transition_sustained)
        motion_yaw = _estimate_query_motion_yaw(frames, int(fi), max_span=6, min_disp_m=0.25)
        if motion_yaw is not None:
            dy = float(_yaw_abs_diff_deg(float(qyaw), float(motion_yaw)))
            if dy > 20.0:
                qyaw_for_proj = float(motion_yaw)
            else:
                qyaw_for_proj = float(_interp_yaw_deg(float(qyaw), float(motion_yaw), 0.35))
            fr["projection_yaw_used"] = float(qyaw_for_proj)
        lane_idx = _safe_int(fr.get("lane_index"), -1)
        mapping = lane_to_carla.get(int(lane_idx))
        if bool(SKIP_POOR_REVERSED_CORR_WHEN_SEMANTIC_UNKNOWN) and isinstance(mapping, dict):
            map_quality = str(mapping.get("quality", "none")).strip().lower()
            map_reversed = bool(mapping.get("reversed", False))
            assigned_lane_id = _safe_int(fr.get("assigned_lane_id"), -1)
            if (
                role_norm == "ego"
                and int(assigned_lane_id) < 0
                and bool(map_reversed)
                and map_quality in {"poor", "low", "none"}
            ):
                mapping = None
        best: Optional[Dict[str, object]] = None
        source = "nearest"
        quality = "none"
        if mapping is not None:
            quality = str(mapping.get("quality", "none"))
            quality_norm = str(quality or "none").strip().lower()
            primary_ci = _safe_int(mapping.get("carla_line_index"), -1)
            split_extra_raw = mapping.get("split_extra_carla_lines", [])
            split_extra = [
                int(_safe_int(v, -1))
                for v in (split_extra_raw if isinstance(split_extra_raw, (list, tuple, set)) else [])
                if _safe_int(v, -1) >= 0
            ]
            line_candidates = []
            if primary_ci >= 0:
                line_candidates.append(int(primary_ci))
            for ci in split_extra:
                if ci not in line_candidates:
                    line_candidates.append(ci)
            best_adjusted_score = float("inf")
            for ci in line_candidates:
                prefer_reversed = bool(mapping.get("reversed", False)) if ci == primary_ci else None
                cand = _best_projection_on_carla_line(
                    carla_context=carla_context,
                    line_index=int(ci),
                    qx=float(qx),
                    qy=float(qy),
                    qyaw=float(qyaw_for_proj),
                    prefer_reversed=prefer_reversed,
                    enforce_node_direction=True,
                    opposite_reject_deg=float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG),
                )
                if cand is None:
                    continue
                cand_score = float(_safe_float(cand.get("score"), float("inf")))
                cand_adjusted = float(cand_score)
                # Weak correspondence may include split alternatives that collapse
                # onto a shared line serving multiple V2 lanes. Penalize those
                # choices slightly so dedicated primary lines are preferred unless
                # split geometry is materially better.
                if (
                    int(ci) != int(primary_ci)
                    and quality_norm in {"poor", "low", "none"}
                    and float(SPLIT_AMBIGUITY_PENALTY) > 0.0
                ):
                    split_labels = carla_line_label_set.get(int(ci), set())
                    primary_labels = carla_line_label_set.get(int(primary_ci), set()) if int(primary_ci) >= 0 else set()
                    uniq_ci = int(carla_line_unique_match_count.get(int(ci), 1))
                    uniq_primary = int(carla_line_unique_match_count.get(int(primary_ci), 1)) if int(primary_ci) >= 0 else 1
                    curr_road = _safe_int(fr.get("road_id"), -1)
                    curr_lane = _safe_int(fr.get("lane_id"), 0)
                    curr_prefix = f"r{int(curr_road)}_l{int(curr_lane)}_" if curr_road >= 0 and curr_lane != 0 else ""
                    split_has_curr_label = bool(curr_prefix) and any(str(lbl).startswith(curr_prefix) for lbl in split_labels)
                    # Apply only for true ambiguous-superset splits:
                    # dedicated primary label is contained within split labels,
                    # and split line carries additional semantic labels.
                    apply_penalty = (
                        len(primary_labels) == 1
                        and len(split_labels) > len(primary_labels)
                        and primary_labels.issubset(split_labels)
                        and split_has_curr_label
                        and uniq_ci > max(1, uniq_primary)
                    )
                    if apply_penalty:
                        extra_ambiguity = max(0, int(uniq_ci) - max(1, int(uniq_primary)))
                        if extra_ambiguity > 0:
                            cand_adjusted += float(SPLIT_AMBIGUITY_PENALTY) * float(extra_ambiguity)
                if (
                    prev_line >= 0
                    and not lane_transition_sustained
                    and int(ci) != int(prev_line)
                    and not _is_carla_line_connected(int(prev_line), int(ci))
                ):
                    cand_adjusted += float(DISCONNECTED_SWITCH_PENALTY)
                if best is None or cand_adjusted < best_adjusted_score:
                    best = cand
                    best_adjusted_score = float(cand_adjusted)
            if best is not None:
                best_ci = int(best["line_index"])
                source = "lane_corr" if best_ci == primary_ci else "lane_corr_split"
                # For weak correspondence rows, suppress tiny-motion semantic
                # lane transitions that would jump to a far CARLA line. In this
                # case keep continuity on the previous line when fit remains
                # comparable.
                if (
                    bool(POOR_CORR_TRANSITION_HOLD_ENABLED)
                    and fi > 0
                    and prev_line >= 0
                    and _frame_has_carla_pose(frames[fi - 1])
                    and lane_transition
                    and not bool(fr.get("synthetic_turn", False))
                    and str(quality_norm) in {"poor", "low", "none"}
                ):
                    qx_prev, qy_prev, qyaw_prev = _frame_query_pose(frames[fi - 1])
                    raw_prev_x = _safe_float(frames[fi - 1].get("x"), float(qx_prev))
                    raw_prev_y = _safe_float(frames[fi - 1].get("y"), float(qy_prev))
                    raw_curr_x = _safe_float(fr.get("x"), float(qx))
                    raw_curr_y = _safe_float(fr.get("y"), float(qy))
                    raw_step = float(
                        math.hypot(float(raw_curr_x) - float(raw_prev_x), float(raw_curr_y) - float(raw_prev_y))
                    )
                    if raw_step < 0.05:
                        raw_step = float(math.hypot(float(qx) - float(qx_prev), float(qy) - float(qy_prev)))
                    yaw_delta = float(_yaw_abs_diff_deg(float(qyaw_for_proj), float(qyaw_prev)))
                    if (
                        raw_step <= float(POOR_CORR_TRANSITION_HOLD_MAX_RAW_STEP)
                        and yaw_delta <= float(POOR_CORR_TRANSITION_HOLD_MAX_YAW_DELTA)
                    ):
                        prev_proj = _best_projection_on_carla_line(
                            carla_context=carla_context,
                            line_index=int(prev_line),
                            qx=float(qx),
                            qy=float(qy),
                            qyaw=float(qyaw_for_proj),
                            prefer_reversed=None,
                            enforce_node_direction=True,
                            opposite_reject_deg=float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG),
                        )
                        if prev_proj is None:
                            prev_proj = _best_projection_on_carla_line(
                                carla_context=carla_context,
                                line_index=int(prev_line),
                                qx=float(qx),
                                qy=float(qy),
                                qyaw=float(qyaw_for_proj),
                                prefer_reversed=None,
                                enforce_node_direction=False,
                                opposite_reject_deg=float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG),
                            )
                        if prev_proj is not None:
                            best_score = _safe_float(best.get("score"), float("inf"))
                            prev_score = _safe_float(prev_proj.get("score"), float("inf"))
                            best_dist = _safe_float(best.get("projection", {}).get("dist"), float("inf"))
                            prev_dist = _safe_float(prev_proj.get("projection", {}).get("dist"), float("inf"))
                            if (
                                prev_score <= best_score + float(POOR_CORR_TRANSITION_HOLD_SCORE_SLACK)
                                and prev_dist <= best_dist + float(POOR_CORR_TRANSITION_HOLD_DIST_SLACK)
                            ):
                                best = prev_proj
                                source = "poor_corr_semantic_hold_prev"
                                quality = "none"

        nearest_any = _nearest_projection_any_line(
            carla_context=carla_context,
            qx=float(qx),
            qy=float(qy),
            qyaw=float(qyaw_for_proj),
            enforce_node_direction=True,
            opposite_reject_deg=float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG),
        )
        if nearest_any is None:
            # Second chance near intersections: if strict node-direction gating finds
            # no line, allow geometric nearest and rely on continuity + final guards.
            nearest_any = _nearest_projection_any_line(
                carla_context=carla_context,
                qx=float(qx),
                qy=float(qy),
                qyaw=float(qyaw_for_proj),
                enforce_node_direction=False,
                opposite_reject_deg=float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG),
            )
        if best is None:
            best = nearest_any
            source = "nearest"
            quality = "none"
        elif nearest_any is not None:
            best_proj = best.get("projection", {})
            near_proj = nearest_any.get("projection", {})
            best_dist = _safe_float(best_proj.get("dist"), float("inf"))
            near_dist = _safe_float(near_proj.get("dist"), float("inf"))
            best_score = _safe_float(best.get("score"), float("inf"))
            near_score = _safe_float(nearest_any.get("score"), float("inf"))

            quality_norm = str(quality or "none").strip().lower()
            # Trust high-quality correspondence more, but still allow override if
            # nearest is clearly and consistently better.
            soft_dist_gate = float(CORR_SOFT_DIST_GOOD) if quality_norm in {"high", "good"} else float(CORR_SOFT_DIST_WEAK)
            hard_dist_gate = float(CORR_HARD_DIST_GOOD) if quality_norm in {"high", "good"} else float(CORR_HARD_DIST_WEAK)
            score_margin = float(CORR_SCORE_MARGIN_GOOD) if quality_norm in {"high", "good"} else float(CORR_SCORE_MARGIN_WEAK)
            dist_margin = float(CORR_DIST_MARGIN_GOOD) if quality_norm in {"high", "good"} else float(CORR_DIST_MARGIN_WEAK)

            strong_improvement = (
                near_dist + dist_margin < best_dist
                and near_score + score_margin < best_score
                and best_dist > soft_dist_gate
            )
            hard_bad_match = (
                best_dist > hard_dist_gate
                and near_dist + float(CORR_HARD_BAD_MARGIN) < best_dist
            )
            poor_quality_override = (
                quality_norm in {"poor", "low", "none"}
                and best_dist > float(CORR_POOR_DIST_GATE)
                and near_dist + float(CORR_POOR_NEAR_MARGIN) < best_dist
            )
            relaxed_override = (
                quality_norm in {"medium", "poor", "low", "none"}
                and near_dist <= float(CORR_RELAXED_OVERRIDE_NEAR_MAX)
                and near_dist + float(CORR_RELAXED_OVERRIDE_MIN_GAIN) < best_dist
                and near_score + float(CORR_RELAXED_OVERRIDE_SCORE_GAIN) < best_score
            )
            if source.startswith("lane_corr") and (
                strong_improvement or hard_bad_match or poor_quality_override or relaxed_override
            ):
                allow_override = True
                raw_step_for_override = float("inf")
                near_jump_for_override = float("inf")
                if fi > 0 and _frame_has_carla_pose(frames[fi - 1]) and not bool(fr.get("synthetic_turn", False)):
                    qx_prev, qy_prev, _ = _frame_query_pose(frames[fi - 1])
                    raw_prev_x = _safe_float(frames[fi - 1].get("x"), float(qx_prev))
                    raw_prev_y = _safe_float(frames[fi - 1].get("y"), float(qy_prev))
                    raw_curr_x = _safe_float(fr.get("x"), float(qx))
                    raw_curr_y = _safe_float(fr.get("y"), float(qy))
                    raw_step = float(math.hypot(float(raw_curr_x) - float(raw_prev_x), float(raw_curr_y) - float(raw_prev_y)))
                    if raw_step < 0.05:
                        raw_step = float(math.hypot(float(qx) - float(qx_prev), float(qy) - float(qy_prev)))
                    raw_step_for_override = float(raw_step)
                    near_px = _safe_float(near_proj.get("x"), float("nan"))
                    near_py = _safe_float(near_proj.get("y"), float("nan"))
                    prev_cx = _safe_float(frames[fi - 1].get("cx"), float("nan"))
                    prev_cy = _safe_float(frames[fi - 1].get("cy"), float("nan"))
                    if math.isfinite(near_px) and math.isfinite(near_py) and math.isfinite(prev_cx) and math.isfinite(prev_cy):
                        near_jump = float(math.hypot(float(near_px) - float(prev_cx), float(near_py) - float(prev_cy)))
                        near_jump_for_override = float(near_jump)
                        if near_jump > max(2.8, 2.9 * max(0.2, raw_step)):
                            allow_override = False
                if allow_override and math.isfinite(float(raw_step_for_override)) and float(raw_step_for_override) <= float(LOW_MOTION_RAW_STEP_MAX):
                    dist_gain = float(best_dist) - float(near_dist)
                    score_gain = float(best_score) - float(near_score)
                    score_gain_need = float(LOW_MOTION_OVERRIDE_MIN_SCORE_GAIN)
                    if quality_norm in {"medium", "poor", "low", "none"}:
                        score_gain_need = min(float(score_gain_need), 0.20)
                    low_motion_max_jump = max(
                        float(LOW_MOTION_OVERRIDE_MAX_JUMP_FLOOR),
                        float(LOW_MOTION_OVERRIDE_MAX_JUMP_SCALE) * float(raw_step_for_override) + float(LOW_MOTION_OVERRIDE_MAX_JUMP_BIAS),
                    )
                    if (
                        float(dist_gain) < float(LOW_MOTION_OVERRIDE_MIN_DIST_GAIN)
                        or float(score_gain) < float(score_gain_need)
                        or (not math.isfinite(float(near_jump_for_override)))
                        or float(near_jump_for_override) > float(low_motion_max_jump)
                    ):
                        allow_override = False
                if allow_override and prev_line >= 0 and not lane_transition_sustained:
                    near_line = _safe_int(nearest_any.get("line_index"), -1)
                    corr_line = _safe_int(best.get("line_index"), -1)
                    if (
                        near_line >= 0
                        and near_line != prev_line
                        and near_line != corr_line
                        and not _is_carla_line_connected(int(prev_line), int(near_line))
                    ):
                        very_strong = (
                            near_dist + float(DISCONNECTED_OVERRIDE_DIST_GAIN) < best_dist
                            and near_score + float(DISCONNECTED_OVERRIDE_SCORE_GAIN) < best_score
                        )
                        weak_relaxed = (
                            quality_norm in {"medium", "poor", "low", "none"}
                            and near_dist + 1.0 < best_dist
                            and near_score + 0.15 < best_score
                        )
                        if not (very_strong or weak_relaxed):
                            allow_override = False
                if allow_override:
                    best = nearest_any
                    source = "nearest_override"
                    quality = "none"

        # Prefer direct continuity over weak branch switches when semantic road
        # evidence points to the current corridor and nearby dedicated lines exist.
        if (
            bool(SEMANTIC_BRANCH_CONTINUITY_GUARD)
            and best is not None
            and fi > 0
            and prev_line >= 0
            and (not bool(lane_transition_sustained))
            and (not bool(fr.get("synthetic_turn", False)))
            and int(best.get("line_index", -1)) >= 0
            and int(best.get("line_index", -1)) != int(prev_line)
        ):
            sem_prev = _semantic_lane_key(frames[fi - 1])
            sem_curr = _semantic_lane_key(fr)
            sem_key = sem_curr if sem_curr is not None else sem_prev
            road_prefix = ""
            lane_prefix = ""
            if sem_key is not None:
                road_prefix = f"r{int(sem_key[0])}_"
                lane_prefix = f"r{int(sem_key[0])}_l{int(sem_key[1])}_"
            best_line = int(_safe_int(best.get("line_index"), -1))
            best_score = float(_safe_float(best.get("score"), float("inf")))
            best_dist = float(_safe_float(best.get("projection", {}).get("dist"), float("inf")))
            best_labels = carla_line_label_set.get(int(best_line), set())
            best_lane_match = bool(lane_prefix) and any(str(lbl).startswith(str(lane_prefix)) for lbl in best_labels)
            best_road_match = bool(road_prefix) and any(str(lbl).startswith(str(road_prefix)) for lbl in best_labels)
            need_branch_guard = bool(road_prefix) and ((not bool(best_lane_match)) or (not bool(best_road_match)))
            if bool(need_branch_guard):
                candidate_ids = _nearby_carla_lines(
                    carla_context=carla_context,
                    qx=float(qx),
                    qy=float(qy),
                    top_k=14,
                )
                if int(prev_line) not in candidate_ids:
                    candidate_ids = [int(prev_line)] + list(candidate_ids)
                branch_candidates: List[Tuple[float, float, int, int, Dict[str, object]]] = []
                for ci in candidate_ids:
                    if int(ci) < 0:
                        continue
                    if int(ci) != int(prev_line) and (not _is_carla_line_connected(int(prev_line), int(ci))):
                        continue
                    lane_match = bool(_line_label_matches_prefix(int(ci), lane_prefix)) if lane_prefix else False
                    road_match = bool(_line_label_matches_prefix(int(ci), road_prefix)) if road_prefix else False
                    if (not bool(lane_match)) and (not bool(road_match)):
                        continue
                    cand_proj = _best_projection_on_carla_line(
                        carla_context=carla_context,
                        line_index=int(ci),
                        qx=float(qx),
                        qy=float(qy),
                        qyaw=float(qyaw_for_proj),
                        prefer_reversed=None,
                        enforce_node_direction=True,
                        opposite_reject_deg=float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG),
                    )
                    if cand_proj is None:
                        cand_proj = _best_projection_on_carla_line(
                            carla_context=carla_context,
                            line_index=int(ci),
                            qx=float(qx),
                            qy=float(qy),
                            qyaw=float(qyaw_for_proj),
                            prefer_reversed=None,
                            enforce_node_direction=False,
                            opposite_reject_deg=float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG),
                        )
                    if cand_proj is None:
                        continue
                    cdist = float(_safe_float(cand_proj.get("projection", {}).get("dist"), float("inf")))
                    cscore = float(_safe_float(cand_proj.get("score"), float("inf")))
                    if not (math.isfinite(cdist) and math.isfinite(cscore)):
                        continue
                    label_count = int(_line_label_count(int(ci)))
                    match_rank = 0 if bool(lane_match) else 1
                    branch_candidates.append((float(cscore), float(cdist), int(label_count), int(match_rank), cand_proj))
                if branch_candidates:
                    branch_candidates.sort(key=lambda row: (float(row[3]), float(row[0]), float(row[1]), int(row[2])))
                    best_count = int(_line_label_count(int(best_line)))
                    prev_count = int(_line_label_count(int(prev_line)))
                    chosen_row = branch_candidates[0]
                    prev_row = None
                    for row in branch_candidates:
                        row_line = int(_safe_int(row[4].get("line_index"), -1))
                        if row_line == int(prev_line):
                            prev_row = row
                            break
                    if prev_row is not None and int(prev_count) > 1:
                        dedicated_prev_pool = [
                            row
                            for row in branch_candidates
                            if int(_safe_int(row[4].get("line_index"), -1)) != int(prev_line)
                            and int(row[2]) < int(prev_count)
                            and int(row[3]) <= 1
                            and float(row[1]) <= float(prev_row[1]) + float(SEMANTIC_BRANCH_DEDICATED_PREF_M)
                        ]
                        if dedicated_prev_pool:
                            dedicated_prev_pool.sort(key=lambda row: (int(row[3]), float(row[1]), float(row[0]), int(row[2])))
                            chosen_row = dedicated_prev_pool[0]
                    if int(best_count) > 1:
                        dedicated_pool = [
                            row
                            for row in branch_candidates
                            if int(row[2]) < int(best_count)
                            and float(row[1]) <= float(chosen_row[1]) + float(SEMANTIC_BRANCH_DEDICATED_PREF_M)
                        ]
                        if dedicated_pool:
                            dedicated_pool.sort(key=lambda row: (int(row[3]), float(row[1]), float(row[0]), int(row[2])))
                            chosen_row = dedicated_pool[0]
                    cscore, cdist, ccount, _, cand_best = chosen_row
                    prev_shared_dedicated = bool(int(prev_count) > 1 and int(ccount) < int(prev_count))
                    dedicated_ok = bool(
                        (int(ccount) < int(best_count) or bool(prev_shared_dedicated))
                        and float(cdist) <= float(best_dist) + float(SEMANTIC_BRANCH_DIST_SLACK_M) + float(SEMANTIC_BRANCH_DEDICATED_PREF_M)
                        and float(cscore) <= float(best_score) + float(SEMANTIC_BRANCH_SCORE_SLACK) + float(SEMANTIC_BRANCH_DEDICATED_PREF_M)
                    )
                    general_ok = bool(
                        float(cdist) <= float(best_dist) + min(float(SEMANTIC_BRANCH_DIST_SLACK_M), 1.9)
                        and float(cscore) <= float(best_score) + float(SEMANTIC_BRANCH_SCORE_SLACK)
                    )
                    if bool(general_ok) or bool(dedicated_ok):
                        best = cand_best
                        source = "semantic_branch_continuity"
                        quality = "none"

        # Temporal continuity prior for nearest-based assignment:
        # avoid one-frame jumps to random nearby segments unless the new line is
        # materially better or lane sequence indicates an actual transition.
        if (
            best is not None
            and fi > 0
            and source in {"nearest", "nearest_override"}
            and _frame_has_carla_pose(frames[fi - 1])
        ):
            if prev_line >= 0 and not lane_transition_sustained and int(best.get("line_index", -1)) != int(prev_line):
                prev_proj = _best_projection_on_carla_line(
                    carla_context=carla_context,
                    line_index=int(prev_line),
                    qx=float(qx),
                    qy=float(qy),
                    qyaw=float(qyaw_for_proj),
                    prefer_reversed=None,
                    enforce_node_direction=True,
                    opposite_reject_deg=float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG),
                )
                if prev_proj is None:
                    prev_proj = _best_projection_on_carla_line(
                        carla_context=carla_context,
                        line_index=int(prev_line),
                        qx=float(qx),
                        qy=float(qy),
                        qyaw=float(qyaw_for_proj),
                        prefer_reversed=None,
                        enforce_node_direction=False,
                        opposite_reject_deg=float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG),
                    )
                if prev_proj is not None:
                    best_score = _safe_float(best.get("score"), float("inf"))
                    prev_score = _safe_float(prev_proj.get("score"), float("inf"))
                    best_dist = _safe_float(best.get("projection", {}).get("dist"), float("inf"))
                    prev_dist = _safe_float(prev_proj.get("projection", {}).get("dist"), float("inf"))
                    if prev_score <= (best_score + float(NEAREST_CONT_SCORE_SLACK)) and prev_dist <= (best_dist + float(NEAREST_CONT_DIST_SLACK)):
                        best_jump = float("inf")
                        prev_jump = float("inf")
                        prev_cx = _safe_float(frames[fi - 1].get("cx"), float("nan"))
                        prev_cy = _safe_float(frames[fi - 1].get("cy"), float("nan"))
                        best_px = _safe_float(best.get("projection", {}).get("x"), float("nan"))
                        best_py = _safe_float(best.get("projection", {}).get("y"), float("nan"))
                        prev_px = _safe_float(prev_proj.get("projection", {}).get("x"), float("nan"))
                        prev_py = _safe_float(prev_proj.get("projection", {}).get("y"), float("nan"))
                        if math.isfinite(prev_cx) and math.isfinite(prev_cy):
                            if math.isfinite(best_px) and math.isfinite(best_py):
                                best_jump = float(math.hypot(float(best_px) - float(prev_cx), float(best_py) - float(prev_cy)))
                            if math.isfinite(prev_px) and math.isfinite(prev_py):
                                prev_jump = float(math.hypot(float(prev_px) - float(prev_cx), float(prev_py) - float(prev_cy)))
                        if (
                            (math.isfinite(prev_jump) and math.isfinite(best_jump) and prev_jump <= best_jump + float(NEAREST_CONT_JUMP_MARGIN))
                            or (not math.isfinite(prev_jump))
                            or (not math.isfinite(best_jump))
                        ):
                            best = prev_proj
                            source = "nearest_continuity"
                            quality = "none"

        # Jump guard for large one-step CARLA snaps: if a candidate introduces a
        # sudden jump far larger than raw motion, keep continuity on previous line
        # when that alternative has comparable geometric fit.
        if best is not None and fi > 0 and _frame_has_carla_pose(frames[fi - 1]) and not bool(fr.get("synthetic_turn", False)):
            if prev_line >= 0 and not lane_transition_sustained:
                qx_prev, qy_prev, _ = _frame_query_pose(frames[fi - 1])
                raw_prev_x = _safe_float(frames[fi - 1].get("x"), float(qx_prev))
                raw_prev_y = _safe_float(frames[fi - 1].get("y"), float(qy_prev))
                raw_curr_x = _safe_float(fr.get("x"), float(qx))
                raw_curr_y = _safe_float(fr.get("y"), float(qy))
                raw_step = float(math.hypot(float(raw_curr_x) - float(raw_prev_x), float(raw_curr_y) - float(raw_prev_y)))
                if raw_step < 0.05:
                    raw_step = float(math.hypot(float(qx) - float(qx_prev), float(qy) - float(qy_prev)))
                best_proj_now = best.get("projection", {}) if isinstance(best, dict) else {}
                bx = _safe_float(best_proj_now.get("x"), float("nan"))
                by = _safe_float(best_proj_now.get("y"), float("nan"))
                prev_cx = _safe_float(frames[fi - 1].get("cx"), float("nan"))
                prev_cy = _safe_float(frames[fi - 1].get("cy"), float("nan"))
                if math.isfinite(bx) and math.isfinite(by) and math.isfinite(prev_cx) and math.isfinite(prev_cy):
                    jump_step = float(math.hypot(float(bx) - float(prev_cx), float(by) - float(prev_cy)))
                    if jump_step > max(2.6, 2.7 * max(0.2, raw_step)):
                        prev_proj = _best_projection_on_carla_line(
                            carla_context=carla_context,
                            line_index=int(prev_line),
                            qx=float(qx),
                            qy=float(qy),
                            qyaw=float(qyaw_for_proj),
                            prefer_reversed=None,
                            enforce_node_direction=True,
                            opposite_reject_deg=float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG),
                        )
                        if prev_proj is None:
                            prev_proj = _best_projection_on_carla_line(
                                carla_context=carla_context,
                                line_index=int(prev_line),
                                qx=float(qx),
                                qy=float(qy),
                                qyaw=float(qyaw_for_proj),
                                prefer_reversed=None,
                                enforce_node_direction=False,
                                opposite_reject_deg=float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG),
                            )
                        if prev_proj is not None:
                            best_score = _safe_float(best.get("score"), float("inf"))
                            prev_score = _safe_float(prev_proj.get("score"), float("inf"))
                            best_dist = _safe_float(best.get("projection", {}).get("dist"), float("inf"))
                            prev_dist = _safe_float(prev_proj.get("projection", {}).get("dist"), float("inf"))
                            jump_guard_prev_dist_abs = _env_float("V2X_CARLA_JUMP_GUARD_PREV_DIST_ABS_M", 3.6)
                            jump_guard_prev_dist_delta = _env_float("V2X_CARLA_JUMP_GUARD_PREV_DIST_DELTA", 0.9)
                            if prev_dist > float(jump_guard_prev_dist_abs):
                                continue
                            if math.isfinite(best_dist) and prev_dist > best_dist + float(jump_guard_prev_dist_delta):
                                continue
                            if prev_score <= (best_score + 0.75) and prev_dist <= (best_dist + 0.95):
                                best = prev_proj
                                source = "jump_guard_prev_line"
                                quality = "none"

        # Additional guard for disconnected one-frame switches: unless a true
        # semantic lane transition is underway, prefer staying on a connected
        # line when fit quality is comparable.
        if (
            best is not None
            and prev_line >= 0
            and not lane_transition_sustained
            and int(best.get("line_index", -1)) >= 0
            and int(best.get("line_index", -1)) != int(prev_line)
            and not _is_carla_line_connected(int(prev_line), int(best.get("line_index", -1)))
        ):
            prev_proj = _best_projection_on_carla_line(
                carla_context=carla_context,
                line_index=int(prev_line),
                qx=float(qx),
                qy=float(qy),
                qyaw=float(qyaw_for_proj),
                prefer_reversed=None,
                enforce_node_direction=True,
                opposite_reject_deg=float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG),
            )
            if prev_proj is None:
                prev_proj = _best_projection_on_carla_line(
                    carla_context=carla_context,
                    line_index=int(prev_line),
                    qx=float(qx),
                    qy=float(qy),
                    qyaw=float(qyaw_for_proj),
                    prefer_reversed=None,
                    enforce_node_direction=False,
                    opposite_reject_deg=float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG),
                )
            if prev_proj is not None:
                best_score = _safe_float(best.get("score"), float("inf"))
                prev_score = _safe_float(prev_proj.get("score"), float("inf"))
                best_dist = _safe_float(best.get("projection", {}).get("dist"), float("inf"))
                prev_dist = _safe_float(prev_proj.get("projection", {}).get("dist"), float("inf"))
                if (
                    prev_score <= (best_score + float(DISCONNECTED_HOLD_SCORE_SLACK))
                    and prev_dist <= (best_dist + float(DISCONNECTED_HOLD_DIST_SLACK))
                ):
                    best = prev_proj
                    source = "disconnect_guard_prev_line"
                    quality = "none"

        # Conservative low-motion weak-switch guard:
        # when evidence is ambiguous, preserve previous consistent line to avoid
        # random one-frame lane/polylines snapping.
        if (
            bool(WEAK_SWITCH_GUARD_ENABLED)
            and best is not None
            and fi > 0
            and prev_line >= 0
            and not lane_transition_sustained
            and int(best.get("line_index", -1)) >= 0
            and int(best.get("line_index", -1)) != int(prev_line)
        ):
            qx_prev, qy_prev, _ = _frame_query_pose(frames[fi - 1])
            raw_prev_x = _safe_float(frames[fi - 1].get("x"), float(qx_prev))
            raw_prev_y = _safe_float(frames[fi - 1].get("y"), float(qy_prev))
            raw_curr_x = _safe_float(fr.get("x"), float(qx))
            raw_curr_y = _safe_float(fr.get("y"), float(qy))
            raw_step = float(
                math.hypot(
                    float(raw_curr_x) - float(raw_prev_x),
                    float(raw_curr_y) - float(raw_prev_y),
                )
            )
            if raw_step < 0.05:
                raw_step = float(math.hypot(float(qx) - float(qx_prev), float(qy) - float(qy_prev)))
            if float(raw_step) <= float(WEAK_SWITCH_GUARD_RAW_STEP_MAX_M):
                prev_proj = _best_projection_on_carla_line(
                    carla_context=carla_context,
                    line_index=int(prev_line),
                    qx=float(qx),
                    qy=float(qy),
                    qyaw=float(qyaw_for_proj),
                    prefer_reversed=None,
                    enforce_node_direction=True,
                    opposite_reject_deg=float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG),
                )
                if prev_proj is None:
                    prev_proj = _best_projection_on_carla_line(
                        carla_context=carla_context,
                        line_index=int(prev_line),
                        qx=float(qx),
                        qy=float(qy),
                        qyaw=float(qyaw_for_proj),
                        prefer_reversed=None,
                        enforce_node_direction=False,
                        opposite_reject_deg=float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG),
                    )
                if prev_proj is not None:
                    best_score = _safe_float(best.get("score"), float("inf"))
                    prev_score = _safe_float(prev_proj.get("score"), float("inf"))
                    best_dist = _safe_float(best.get("projection", {}).get("dist"), float("inf"))
                    prev_dist = _safe_float(prev_proj.get("projection", {}).get("dist"), float("inf"))
                    dist_gain = float(prev_dist) - float(best_dist)
                    score_gain = float(prev_score) - float(best_score)
                    weak_guard_max_prev_dist = float(WEAK_SWITCH_GUARD_MAX_PREV_DIST_M)
                    weak_guard_min_dist_gain = float(WEAK_SWITCH_GUARD_MIN_DIST_GAIN_M)
                    weak_guard_min_score_gain = float(WEAK_SWITCH_GUARD_MIN_SCORE_GAIN)
                    if role_norm == "ego":
                        weak_guard_max_prev_dist = min(
                            float(weak_guard_max_prev_dist),
                            float(_env_float("V2X_CARLA_WEAK_SWITCH_GUARD_MAX_PREV_DIST_EGO_M", 3.2)),
                        )
                        weak_guard_min_dist_gain = min(
                            float(weak_guard_min_dist_gain),
                            float(_env_float("V2X_CARLA_WEAK_SWITCH_GUARD_MIN_DIST_GAIN_EGO_M", 0.75)),
                        )
                        weak_guard_min_score_gain = min(
                            float(weak_guard_min_score_gain),
                            float(_env_float("V2X_CARLA_WEAK_SWITCH_GUARD_MIN_SCORE_GAIN_EGO", 0.75)),
                        )
                        assigned_lane_id_curr = _safe_int(fr.get("assigned_lane_id"), 0)
                        if int(assigned_lane_id_curr) < 0:
                            # When semantic lane id is unknown, prefer temporal
                            # continuity over weak nearest-line gains.
                            weak_guard_max_prev_dist = max(
                                float(weak_guard_max_prev_dist),
                                float(_env_float("V2X_CARLA_WEAK_SWITCH_GUARD_MAX_PREV_DIST_EGO_UNKNOWN_M", 4.8)),
                            )
                            weak_guard_min_dist_gain = max(
                                float(weak_guard_min_dist_gain),
                                float(_env_float("V2X_CARLA_WEAK_SWITCH_GUARD_MIN_DIST_GAIN_EGO_UNKNOWN_M", 1.0)),
                            )
                            weak_guard_min_score_gain = max(
                                float(weak_guard_min_score_gain),
                                float(_env_float("V2X_CARLA_WEAK_SWITCH_GUARD_MIN_SCORE_GAIN_EGO_UNKNOWN", 0.9)),
                            )
                    if (
                        float(prev_dist) <= float(weak_guard_max_prev_dist)
                        and float(dist_gain) < float(weak_guard_min_dist_gain)
                        and float(score_gain) < float(weak_guard_min_score_gain)
                    ):
                        best = prev_proj
                        source = "weak_switch_guard_prev_line"
                        quality = "none"

        if best is None and prev_line >= 0:
            prev_hold = _best_projection_on_carla_line(
                carla_context=carla_context,
                line_index=int(prev_line),
                qx=float(qx),
                qy=float(qy),
                qyaw=float(qyaw_for_proj),
                prefer_reversed=None,
                enforce_node_direction=True,
                opposite_reject_deg=float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG),
            )
            if prev_hold is None:
                prev_hold = _best_projection_on_carla_line(
                    carla_context=carla_context,
                    line_index=int(prev_line),
                    qx=float(qx),
                    qy=float(qy),
                    qyaw=float(qyaw_for_proj),
                    prefer_reversed=None,
                    enforce_node_direction=False,
                    opposite_reject_deg=float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG),
                )
            if isinstance(prev_hold, dict):
                prev_hold_dist = _safe_float(prev_hold.get("projection", {}).get("dist"), float("inf"))
                prev_hold_max_dist = _env_float("V2X_CARLA_PREV_LINE_HOLD_MAX_DIST_M", 5.8)
                if role_norm == "ego" and _safe_int(fr.get("assigned_lane_id"), 0) < 0:
                    prev_hold_max_dist = _env_float(
                        "V2X_CARLA_PREV_LINE_HOLD_MAX_DIST_EGO_UNKNOWN_M",
                        max(6.2, float(prev_hold_max_dist)),
                    )
                if math.isfinite(prev_hold_dist) and float(prev_hold_dist) <= float(prev_hold_max_dist):
                    best = prev_hold
                    source = "prev_line_hold_on_miss"
                    quality = "none"

        if best is None:
            continue
        proj = best["projection"]
        _set_carla_pose(
            frame=fr,
            line_index=int(best["line_index"]),
            x=float(proj["x"]),
            y=float(proj["y"]),
            yaw=float(proj["yaw"]),
            dist=float(proj["dist"]),
            source=source,
            quality=quality,
        )

    def _reanchor_spawn_head_transition() -> None:
        """
        If a track starts with an immediate disconnected lane hop, treat it as a
        spawn-in-turn case and backfill the first few frames onto the turn line.
        """
        if len(frames) < 3:
            return
        if _env_int("V2X_CARLA_SPAWN_TURN_REANCHOR_ENABLED", 1, minimum=0, maximum=1) != 1:
            return
        max_head_transition_frame = _env_int(
            "V2X_CARLA_SPAWN_TURN_REANCHOR_MAX_HEAD_TRANSITION_FRAME",
            6,
            minimum=1,
            maximum=10,
        )
        min_dist_gain_m = _env_float("V2X_CARLA_SPAWN_TURN_REANCHOR_MIN_DIST_GAIN_M", 0.6)
        target_max_dist_m = _env_float("V2X_CARLA_SPAWN_TURN_REANCHOR_TARGET_MAX_DIST_M", 4.5)

        trans_idx = -1
        from_line = -1
        to_line = -1
        for i in range(1, len(frames)):
            a = frames[i - 1]
            b = frames[i]
            if not (_frame_has_carla_pose(a) and _frame_has_carla_pose(b)):
                continue
            la = _safe_int(a.get("ccli"), -1)
            lb = _safe_int(b.get("ccli"), -1)
            if la < 0 or lb < 0 or la == lb:
                continue
            trans_idx = int(i)
            from_line = int(la)
            to_line = int(lb)
            break
        if trans_idx < 1 or trans_idx > int(max_head_transition_frame):
            return
        if int(from_line) < 0 or int(to_line) < 0 or int(from_line) == int(to_line):
            return
        from_road = int(carla_line_road_id_by_index.get(int(from_line), 0))
        to_road = int(carla_line_road_id_by_index.get(int(to_line), 0))
        from_lane = int(carla_line_lane_id_by_index.get(int(from_line), 0))
        to_lane = int(carla_line_lane_id_by_index.get(int(to_line), 0))
        same_road_continuation = bool(
            int(from_road) != 0
            and int(from_road) == int(to_road)
            and (
                int(from_lane) == 0
                or int(to_lane) == 0
                or int(from_lane) * int(to_lane) > 0
            )
        )
        if _is_carla_line_connected(int(from_line), int(to_line)) and bool(same_road_continuation):
            return

        base_dists: List[float] = []
        target_dists: List[float] = []
        target_rows: List[Tuple[int, Dict[str, object]]] = []
        for fi in range(0, int(trans_idx) + 1):
            fr = frames[fi]
            qx, qy, qyaw = _frame_query_pose(fr)
            cand_to = _best_projection_on_carla_line(
                carla_context=carla_context,
                line_index=int(to_line),
                qx=float(qx),
                qy=float(qy),
                qyaw=float(qyaw),
                prefer_reversed=None,
                enforce_node_direction=True,
                opposite_reject_deg=float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG),
            )
            if cand_to is None:
                cand_to = _best_projection_on_carla_line(
                    carla_context=carla_context,
                    line_index=int(to_line),
                    qx=float(qx),
                    qy=float(qy),
                    qyaw=float(qyaw),
                    prefer_reversed=None,
                    enforce_node_direction=False,
                    opposite_reject_deg=float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG),
                )
            cand_from = _best_projection_on_carla_line(
                carla_context=carla_context,
                line_index=int(from_line),
                qx=float(qx),
                qy=float(qy),
                qyaw=float(qyaw),
                prefer_reversed=None,
                enforce_node_direction=True,
                opposite_reject_deg=float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG),
            )
            if cand_from is None:
                cand_from = _best_projection_on_carla_line(
                    carla_context=carla_context,
                    line_index=int(from_line),
                    qx=float(qx),
                    qy=float(qy),
                    qyaw=float(qyaw),
                    prefer_reversed=None,
                    enforce_node_direction=False,
                    opposite_reject_deg=float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG),
                )
            if not isinstance(cand_to, dict) or not isinstance(cand_from, dict):
                return
            d_to = float(_safe_float(cand_to.get("projection", {}).get("dist"), float("inf")))
            d_from = float(_safe_float(cand_from.get("projection", {}).get("dist"), float("inf")))
            if not (math.isfinite(d_to) and math.isfinite(d_from)):
                return
            base_dists.append(float(d_from))
            target_dists.append(float(d_to))
            target_rows.append((int(fi), cand_to))
        if not base_dists or not target_dists:
            return
        base_med = float(np.median(np.asarray(base_dists, dtype=np.float64)))
        target_med = float(np.median(np.asarray(target_dists, dtype=np.float64)))
        if float(target_med) > float(target_max_dist_m):
            return
        if float(target_med) + float(min_dist_gain_m) > float(base_med):
            return

        for fi, cand in target_rows:
            proj = cand.get("projection", {})
            _set_carla_pose(
                frame=frames[int(fi)],
                line_index=int(to_line),
                x=float(_safe_float(proj.get("x"), float(frames[int(fi)].get("cx", 0.0)))),
                y=float(_safe_float(proj.get("y"), float(frames[int(fi)].get("cy", 0.0)))),
                yaw=float(_safe_float(proj.get("yaw"), float(frames[int(fi)].get("cyaw", 0.0)))),
                dist=float(_safe_float(proj.get("dist"), 0.0)),
                source="spawn_in_turn_line",
                quality=str(frames[int(fi)].get("cquality", "none")),
            )
            frames[int(fi)]["spawn_in_turn_line"] = True
            frames[int(fi)]["spawn_in_turn_target_line"] = int(to_line)

    _reanchor_spawn_head_transition()

    # Snapshot initial CARLA projection before post-processing stack so the
    # visualizer can compare where artifacts are introduced.
    _snapshot_carla_pose_frames(frames, prefix="cb")

    _regenerate_carla_turn_segments(frames)
    _bridge_missing_carla_gaps(
        frames,
        carla_context=carla_context,
        max_gap_frames=_env_int("V2X_CARLA_BRIDGE_MAX_GAP_FRAMES", 30, minimum=1, maximum=100),
        max_bridge_dist_m=_env_float("V2X_CARLA_BRIDGE_MAX_DIST_M", 42.0),
        enforce_node_direction=True,
        opposite_reject_deg=float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG),
    )
    _smooth_carla_jump_stall_artifacts(frames)
    _smooth_short_carla_line_runs(
        frames,
        carla_context=carla_context,
        track_label=str(track_meta.get("id", "")) if isinstance(track_meta, dict) else "",
        max_mid_run_frames=_env_int("V2X_CARLA_SMOOTH_MAX_MID_RUN", 14, minimum=1, maximum=40),
        max_edge_run_frames=_env_int("V2X_CARLA_SMOOTH_MAX_EDGE_RUN", 3, minimum=1, maximum=10),
        max_head_run_frames=_env_int("V2X_CARLA_SMOOTH_MAX_HEAD_RUN", 20, minimum=1, maximum=60),
        min_stable_neighbor_frames=_env_int("V2X_CARLA_SMOOTH_MIN_STABLE_NEIGHBOR", 20, minimum=2, maximum=80),
        transition_spike_max_frames=_env_int("V2X_CARLA_TRANSITION_SPIKE_MAX_FRAMES", 4, minimum=1, maximum=12),
        transition_min_neighbor_frames=_env_int("V2X_CARLA_TRANSITION_MIN_NEIGHBOR", 10, minimum=1, maximum=40),
        transition_cost_slack_base=_env_float("V2X_CARLA_TRANSITION_COST_SLACK_BASE", 1.2),
        transition_cost_slack_per_frame=_env_float("V2X_CARLA_TRANSITION_COST_SLACK_PER_FRAME", 0.45),
        transition_max_point_delta=_env_float("V2X_CARLA_TRANSITION_MAX_POINT_DELTA", 2.0),
        cost_slack_base=_env_float("V2X_CARLA_SMOOTH_COST_SLACK_BASE", 0.5),
        cost_slack_per_frame=_env_float("V2X_CARLA_SMOOTH_COST_SLACK_PER_FRAME", 0.35),
        max_point_delta=_env_float("V2X_CARLA_SMOOTH_MAX_POINT_DELTA", 1.5),
        max_point_dist=_env_float("V2X_CARLA_SMOOTH_MAX_POINT_DIST", 12.0),
        aba_cost_slack_base=_env_float("V2X_CARLA_ABA_COST_SLACK_BASE", 12.0),
        aba_max_point_dist=_env_float("V2X_CARLA_ABA_MAX_POINT_DIST", 20.0),
        aba_enforce_node_direction=_env_int("V2X_CARLA_ABA_ENFORCE_NODE_DIRECTION", 0) != 0,
        enforce_node_direction=True,
        opposite_reject_deg=float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG),
    )
    _repair_carla_transition_spikes(
        frames,
        carla_context=carla_context,
        enforce_node_direction=True,
        opposite_reject_deg=float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG),
    )
    _suppress_short_carla_switchbacks(
        frames,
        carla_context=carla_context,
        max_run_frames=_env_int("V2X_CARLA_SWITCHBACK_MAX_RUN", 20, minimum=1, maximum=60),
        cost_slack_base=_env_float("V2X_CARLA_SWITCHBACK_COST_SLACK_BASE", 1.0),
        cost_slack_per_frame=_env_float("V2X_CARLA_SWITCHBACK_COST_SLACK_PER_FRAME", 0.55),
        max_point_delta=_env_float("V2X_CARLA_SWITCHBACK_MAX_POINT_DELTA", 2.4),
        max_point_dist=_env_float("V2X_CARLA_SWITCHBACK_MAX_POINT_DIST", 10.0),
        enforce_node_direction=True,
        opposite_reject_deg=float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG),
    )
    _soften_small_step_transition_jumps(
        frames,
        jump_threshold_m=_env_float("V2X_CARLA_SMALL_TRANSITION_JUMP_M", 1.5),
        jump_ratio_vs_raw=_env_float("V2X_CARLA_SMALL_TRANSITION_JUMP_RATIO", 2.2),
        max_shift_m=_env_float("V2X_CARLA_SMALL_TRANSITION_MAX_SHIFT_M", 1.2),
        max_query_cost_delta=_env_float("V2X_CARLA_SMALL_TRANSITION_MAX_QUERY_DELTA", 1.0),
    )
    if not (
        role_norm == "ego"
        and _env_int("V2X_CARLA_SKIP_TRANSITION_WINDOW_SMOOTH_EGO", 1, minimum=0, maximum=1) == 1
    ):
        _smooth_carla_transition_windows(
            frames,
            window_radius=_env_int("V2X_CARLA_TRANSITION_WINDOW_RADIUS", 3, minimum=1, maximum=4),
            max_shift_m=_env_float("V2X_CARLA_TRANSITION_WINDOW_MAX_SHIFT_M", 1.2),
            max_query_cost_delta=_env_float("V2X_CARLA_TRANSITION_WINDOW_MAX_QUERY_DELTA", 1.0),
            max_total_cost_delta=_env_float("V2X_CARLA_TRANSITION_WINDOW_MAX_TOTAL_DELTA", 2.0),
            passes=_env_int("V2X_CARLA_TRANSITION_WINDOW_PASSES", 3, minimum=1, maximum=4),
        )
    _soften_semantic_boundary_jumps(
        frames,
        jump_threshold_m=_env_float("V2X_CARLA_SEMANTIC_BOUNDARY_JUMP_M", 2.6),
        jump_ratio_vs_query=_env_float("V2X_CARLA_SEMANTIC_BOUNDARY_QUERY_RATIO", 2.2),
        move_toward_mid=_env_float("V2X_CARLA_SEMANTIC_BOUNDARY_BLEND", 0.55),
        max_shift_m=_env_float("V2X_CARLA_SEMANTIC_BOUNDARY_MAX_SHIFT_M", 1.35),
        max_query_cost_delta=_env_float("V2X_CARLA_SEMANTIC_BOUNDARY_MAX_QUERY_DELTA", 1.1),
    )
    _hold_semantic_straight_switches(
        frames,
        carla_context=carla_context,
        enforce_node_direction=True,
        opposite_reject_deg=float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG),
    )
    _smooth_same_line_projection_spikes(
        frames,
        jump_threshold_m=_env_float("V2X_CARLA_SAME_LINE_SPIKE_JUMP_M", 2.4),
        jump_ratio_vs_query=_env_float("V2X_CARLA_SAME_LINE_SPIKE_RATIO", 3.0),
        max_shift_m=_env_float("V2X_CARLA_SAME_LINE_SPIKE_MAX_SHIFT_M", 2.4),
        max_query_cost_delta=_env_float("V2X_CARLA_SAME_LINE_SPIKE_MAX_QUERY_DELTA", 0.7),
    )
    _smooth_carla_jump_stall_artifacts(frames)
    _stabilize_carla_yaw_at_transitions(
        frames,
        max_low_step_yaw_jump_deg=_env_float("V2X_CARLA_TRANSITION_YAW_MAX_JUMP_DEG", 26.0),
    )
    # NOTE: these extra post-smoothers can over-steer intersection geometry.
    # Keep disabled by default to preserve raw/carla fidelity and turn behavior.
    if _env_int("V2X_CARLA_ENABLE_MICRO_JITTER_SMOOTH", 0, minimum=0, maximum=1) == 1:
        _smooth_carla_micro_jitter(frames)
    if _env_int("V2X_CARLA_ENABLE_YAW_STABILIZE", 0, minimum=0, maximum=1) == 1:
        _stabilize_carla_yaw_to_motion(frames)
    _smooth_carla_jump_stall_artifacts(frames)
    _stabilize_carla_line_ids(
        frames,
        max_semantic_hold_run=_env_int("V2X_CARLA_SEMANTIC_LINE_HOLD_MAX_RUN", 30, minimum=1, maximum=60),
        carla_context=carla_context,
        opposite_reject_deg=float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG),
    )
    _clamp_turn_source_low_motion_steps(
        frames,
        raw_low_step_m=_env_float("V2X_CARLA_TURN_LOW_MOTION_RAW_STEP_M", 0.6),
        max_step_floor_m=_env_float("V2X_CARLA_TURN_LOW_MOTION_MAX_STEP_FLOOR_M", 1.4),
        step_ratio_vs_raw=_env_float("V2X_CARLA_TURN_LOW_MOTION_STEP_RATIO", 2.4),
    )
    # Semantic-run lock: within each stable V2XPNP lane run, suppress ccli
    # oscillation by snapping all frames to the most-common CARLA line.
    # This directly fixes intersection connector confusion and oscillating
    # lane changes that are too long for _suppress_short_carla_switchbacks.
    _stabilize_ccli_within_semantic_runs(
        frames,
        carla_context=carla_context,
        opposite_reject_deg=float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG),
    )
    _fallback_far_carla_samples(frames)
    _fallback_wrong_way_carla_samples(
        frames,
        opposite_reject_deg=_env_float("V2X_CARLA_WRONG_WAY_REJECT_DEG", float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG)),
        carla_context=carla_context,
        canonical_reject_deg=_env_float("V2X_CARLA_WRONG_WAY_CANONICAL_REJECT_DEG", 181.0),
        try_reproject=bool(_env_int("V2X_CARLA_WRONG_WAY_TRY_REPROJECT", 1, minimum=0, maximum=1)),
    )
    _suppress_low_motion_teleports(
        frames,
        raw_step_threshold_m=_env_float("V2X_CARLA_FINAL_TELEPORT_RAW_STEP_M", 0.6),
        jump_threshold_m=_env_float("V2X_CARLA_FINAL_TELEPORT_JUMP_M", 1.8),
        target_floor_m=_env_float("V2X_CARLA_FINAL_TELEPORT_TARGET_FLOOR_M", 1.2),
        target_scale=_env_float("V2X_CARLA_FINAL_TELEPORT_TARGET_SCALE", 2.0),
        target_bias_m=_env_float("V2X_CARLA_FINAL_TELEPORT_TARGET_BIAS_M", 0.4),
    )
    _fallback_wrong_way_carla_samples(
        frames,
        opposite_reject_deg=_env_float("V2X_CARLA_WRONG_WAY_REJECT_DEG", float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG)),
        carla_context=carla_context,
        canonical_reject_deg=_env_float("V2X_CARLA_WRONG_WAY_CANONICAL_REJECT_DEG", 181.0),
        try_reproject=False,
    )
    # Final safety clamp: if any CARLA position ended up too far from the
    # query trajectory (V2XPNP/raw), fall back to the query position.
    # This catches ALL projection passes that produce bad results.
    _max_div = _env_float("V2X_CARLA_MAX_DIVERGENCE_M", 5.0)
    _use_raw_divergence = _env_int("V2X_CARLA_MAX_DIVERGENCE_USE_RAW", 1, minimum=0, maximum=1) == 1
    _use_raw_divergence_ego = (
        _env_int("V2X_CARLA_MAX_DIVERGENCE_USE_RAW_EGO", 0, minimum=0, maximum=1) == 1
    )
    _use_raw_for_role = bool(_use_raw_divergence)
    if role_norm == "ego":
        _use_raw_for_role = bool(_use_raw_divergence_ego)
    _sticky_enabled = (
        role_norm == "ego"
        and _env_int("V2X_CARLA_MAX_DIVERGENCE_STICKY_EGO_ENABLED", 1, minimum=0, maximum=1) == 1
    )
    _sticky_frames = _env_int("V2X_CARLA_MAX_DIVERGENCE_STICKY_FRAMES", 12, minimum=0, maximum=80)
    _sticky_dist = _env_float(
        "V2X_CARLA_MAX_DIVERGENCE_STICKY_DIST_M",
        max(2.8, 0.65 * float(_max_div)),
    )
    _release_jump_guard_enabled = (
        role_norm == "ego"
        and _env_int("V2X_CARLA_MAX_DIVERGENCE_RELEASE_JUMP_GUARD_EGO_ENABLED", 1, minimum=0, maximum=1) == 1
    )
    _release_jump_abs_m = _env_float("V2X_CARLA_MAX_DIVERGENCE_RELEASE_JUMP_M", 1.8)
    _release_jump_ratio = _env_float("V2X_CARLA_MAX_DIVERGENCE_RELEASE_JUMP_RAW_RATIO", 2.6)
    _fallback_clamp_enabled = (
        role_norm == "ego"
        and _env_int("V2X_CARLA_MAX_DIVERGENCE_FALLBACK_CLAMP_EGO_ENABLED", 1, minimum=0, maximum=1) == 1
    )
    _fallback_clamp_step_floor_m = _env_float("V2X_CARLA_MAX_DIVERGENCE_FALLBACK_CLAMP_STEP_FLOOR_M", 1.2)
    _fallback_clamp_step_ratio = _env_float("V2X_CARLA_MAX_DIVERGENCE_FALLBACK_CLAMP_STEP_RATIO", 1.8)
    _fallback_clamp_step_bias_m = _env_float("V2X_CARLA_MAX_DIVERGENCE_FALLBACK_CLAMP_STEP_BIAS_M", 0.35)
    _parked_divergence_guard_enabled = (
        bool(parked_like_track)
        and _env_int("V2X_CARLA_MAX_DIVERGENCE_PARKED_GUARD_ENABLED", 1, minimum=0, maximum=1) == 1
    )
    _parked_divergence_anchor: Optional[Tuple[float, float, float, int]] = None
    if bool(_parked_divergence_guard_enabled):
        ax_vals: List[float] = []
        ay_vals: List[float] = []
        ayaw_vals: List[float] = []
        acli_vals: List[int] = []
        for _fr_anchor in frames:
            _ax = _safe_float(_fr_anchor.get("cbx"), float("nan"))
            _ay = _safe_float(_fr_anchor.get("cby"), float("nan"))
            if not (math.isfinite(_ax) and math.isfinite(_ay)):
                _ax = _safe_float(_fr_anchor.get("cx"), float("nan"))
                _ay = _safe_float(_fr_anchor.get("cy"), float("nan"))
            if not (math.isfinite(_ax) and math.isfinite(_ay)):
                continue
            ax_vals.append(float(_ax))
            ay_vals.append(float(_ay))
            _ayaw = _safe_float(_fr_anchor.get("cbyaw"), float("nan"))
            if not math.isfinite(_ayaw):
                _ayaw = _safe_float(_fr_anchor.get("cyaw"), float("nan"))
            if math.isfinite(_ayaw):
                ayaw_vals.append(math.radians(float(_ayaw)))
            _acli = _safe_int(_fr_anchor.get("cbcli"), -1)
            if _acli < 0:
                _acli = _safe_int(_fr_anchor.get("ccli"), -1)
            if _acli >= 0:
                acli_vals.append(int(_acli))
        if ax_vals and ay_vals:
            anchor_x = float(np.median(np.asarray(ax_vals, dtype=np.float64)))
            anchor_y = float(np.median(np.asarray(ay_vals, dtype=np.float64)))
            anchor_yaw = float(_safe_float(frames[0].get("cyaw"), 0.0))
            if ayaw_vals:
                _sy = float(np.mean(np.sin(np.asarray(ayaw_vals, dtype=np.float64))))
                _cy = float(np.mean(np.cos(np.asarray(ayaw_vals, dtype=np.float64))))
                anchor_yaw = float(_normalize_yaw_deg(math.degrees(math.atan2(_sy, _cy))))
            anchor_cli = int(max(set(acli_vals), key=acli_vals.count)) if acli_vals else -1
            _parked_divergence_anchor = (float(anchor_x), float(anchor_y), float(anchor_yaw), int(anchor_cli))
    _sticky_left = 0
    for _idx, _fr in enumerate(frames):
        _cx = _safe_float(_fr.get("cx"), float("nan"))
        _cy = _safe_float(_fr.get("cy"), float("nan"))
        if not (math.isfinite(_cx) and math.isfinite(_cy)):
            continue
        _qx, _qy, _qyaw = _frame_query_pose(_fr)
        _d = math.hypot(_cx - _qx, _cy - _qy)
        # Also check against raw position — V2XPNP + CARLA offsets compound.
        _rx = _safe_float(_fr.get("x"), float("nan"))
        _ry = _safe_float(_fr.get("y"), float("nan"))
        _d_raw = (
            math.hypot(_cx - _rx, _cy - _ry)
            if bool(_use_raw_for_role) and math.isfinite(_rx) and math.isfinite(_ry)
            else float(_d)
        )
        needs_fallback = bool(_d > _max_div or (bool(_use_raw_for_role) and _d_raw > _max_div))
        if (
            (not bool(needs_fallback))
            and bool(_release_jump_guard_enabled)
            and int(_idx) > 0
        ):
            _prev = frames[int(_idx) - 1]
            _prev_src = str(_prev.get("csource", "")).strip().lower()
            if _prev_src.startswith("max_divergence_fallback"):
                _prev_cx = _safe_float(_prev.get("cx"), float("nan"))
                _prev_cy = _safe_float(_prev.get("cy"), float("nan"))
                if math.isfinite(_prev_cx) and math.isfinite(_prev_cy):
                    _proj_step = math.hypot(float(_cx) - float(_prev_cx), float(_cy) - float(_prev_cy))
                    _pqx, _pqy, _ = _frame_query_pose(_prev)
                    _raw_step = math.hypot(float(_qx) - float(_pqx), float(_qy) - float(_pqy))
                    _jump_allow = max(
                        float(_release_jump_abs_m),
                        float(_release_jump_ratio) * max(0.0, float(_raw_step)),
                    )
                    if float(_proj_step) > float(_jump_allow):
                        # Keep query fallback while projected geometry is still
                        # trying to "jump back" from a sticky fallback window.
                        needs_fallback = True
        if (not bool(needs_fallback)) and bool(_sticky_enabled) and int(_sticky_left) > 0:
            if _d > float(_sticky_dist) or (bool(_use_raw_for_role) and _d_raw > float(_sticky_dist)):
                needs_fallback = True
                _sticky_left = int(_sticky_left) - 1
            else:
                _sticky_left = 0
        if needs_fallback:
            if bool(_parked_divergence_guard_enabled) and _parked_divergence_anchor is not None:
                _ax, _ay, _ayaw, _acli = _parked_divergence_anchor
                _fr["cx"] = float(_ax)
                _fr["cy"] = float(_ay)
                _fr["cyaw"] = float(_ayaw)
                _fr["cdist"] = float(math.hypot(float(_ax) - float(_qx), float(_ay) - float(_qy)))
                if int(_acli) >= 0:
                    _fr["ccli"] = int(_acli)
                _fr["csource"] = "parked_static_divergence_guard"
                _fr["cquality"] = str(_fr.get("cquality", "none"))
                _sticky_left = 0
                continue
            fallback_source = (
                "max_divergence_fallback_sticky"
                if (bool(_sticky_enabled) and not (_d > _max_div or _d_raw > _max_div))
                else "max_divergence_fallback"
            )
            # Fall back toward query (V2XPNP/raw) position.
            target_x = float(_qx)
            target_y = float(_qy)
            target_yaw = float(_qyaw)
            clamped = False
            if bool(_fallback_clamp_enabled) and int(_idx) > 0:
                _prev = frames[int(_idx) - 1]
                _prev_cx = _safe_float(_prev.get("cx"), float("nan"))
                _prev_cy = _safe_float(_prev.get("cy"), float("nan"))
                if math.isfinite(_prev_cx) and math.isfinite(_prev_cy):
                    _pqx, _pqy, _ = _frame_query_pose(_prev)
                    _raw_step = math.hypot(float(_qx) - float(_pqx), float(_qy) - float(_pqy))
                    _step_allow = max(
                        float(_fallback_clamp_step_floor_m),
                        float(_fallback_clamp_step_ratio) * max(0.0, float(_raw_step)) + float(_fallback_clamp_step_bias_m),
                    )
                    _step_to_query = math.hypot(float(target_x) - float(_prev_cx), float(target_y) - float(_prev_cy))
                    if float(_step_to_query) > float(_step_allow) and float(_step_to_query) > 1e-6:
                        _scale = float(_step_allow) / float(_step_to_query)
                        target_x = float(_prev_cx) + (float(target_x) - float(_prev_cx)) * float(_scale)
                        target_y = float(_prev_cy) + (float(target_y) - float(_prev_cy)) * float(_scale)
                        target_yaw = float(_normalize_yaw_deg(math.degrees(math.atan2(
                            float(target_y) - float(_prev_cy),
                            float(target_x) - float(_prev_cx),
                        ))))
                        clamped = True
            _fr["cx"] = float(target_x)
            _fr["cy"] = float(target_y)
            if math.isfinite(target_yaw):
                _fr["cyaw"] = float(target_yaw)
            _fr["cdist"] = float(math.hypot(float(target_x) - float(_qx), float(target_y) - float(_qy)))
            if int(_idx) > 0:
                _prev_fb = frames[int(_idx) - 1]
                _prev_fb_src = str(_prev_fb.get("csource", "")).strip().lower()
                if _prev_fb_src.startswith("max_divergence_fallback"):
                    _prev_cli = _safe_int(_prev_fb.get("ccli"), -1)
                    if _prev_cli >= 0:
                        _fr["ccli"] = int(_prev_cli)
            _fr["csource"] = f"{fallback_source}_clamped" if bool(clamped) else str(fallback_source)
            if bool(_sticky_enabled) and (_d > _max_div or (bool(_use_raw_for_role) and _d_raw > _max_div)):
                _sticky_left = int(_sticky_frames)
    # Single committed intersection path runs last so later guards don't
    # re-introduce connector switching or turn-shape oscillation.
    _commit_intersection_connectors(
        frames,
        carla_context=carla_context,
    )
    def _enforce_spawn_turn_head_lock() -> None:
        if _env_int("V2X_CARLA_SPAWN_TURN_HEAD_LOCK_ENABLED", 1, minimum=0, maximum=1) != 1:
            return
        flagged_idx = [int(i) for i, fr in enumerate(frames) if bool(fr.get("spawn_in_turn_line", False))]
        if not flagged_idx:
            return
        target_vals = [
            int(_safe_int(frames[i].get("spawn_in_turn_target_line"), -1))
            for i in flagged_idx
            if int(_safe_int(frames[i].get("spawn_in_turn_target_line"), -1)) >= 0
        ]
        if not target_vals:
            return
        target_line = int(max(set(target_vals), key=target_vals.count))
        max_lock_frames = _env_int("V2X_CARLA_SPAWN_TURN_HEAD_LOCK_MAX_FRAMES", 8, minimum=1, maximum=24)
        lock_end = min(int(max(flagged_idx)), int(max_lock_frames) - 1, int(len(frames) - 1))
        if lock_end < 0:
            return
        for fi in range(0, int(lock_end) + 1):
            fr = frames[int(fi)]
            qx, qy, qyaw = _frame_query_pose(fr)
            cand = _best_projection_on_carla_line(
                carla_context=carla_context,
                line_index=int(target_line),
                qx=float(qx),
                qy=float(qy),
                qyaw=float(qyaw),
                prefer_reversed=None,
                enforce_node_direction=True,
                opposite_reject_deg=float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG),
            )
            if cand is None:
                cand = _best_projection_on_carla_line(
                    carla_context=carla_context,
                    line_index=int(target_line),
                    qx=float(qx),
                    qy=float(qy),
                    qyaw=float(qyaw),
                    prefer_reversed=None,
                    enforce_node_direction=False,
                    opposite_reject_deg=float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG),
                )
            if not isinstance(cand, dict):
                continue
            proj = cand.get("projection", {})
            _set_carla_pose(
                frame=fr,
                line_index=int(target_line),
                x=float(_safe_float(proj.get("x"), float(fr.get("cx", qx)))),
                y=float(_safe_float(proj.get("y"), float(fr.get("cy", qy)))),
                yaw=float(_safe_float(proj.get("yaw"), float(fr.get("cyaw", qyaw)))),
                dist=float(_safe_float(proj.get("dist"), 0.0)),
                source="spawn_in_turn_lock",
                quality=str(fr.get("cquality", "none")),
            )
            fr["spawn_in_turn_line"] = True
            fr["spawn_in_turn_target_line"] = int(target_line)

    _enforce_spawn_turn_head_lock()
    if (
        role_norm == "vehicle"
        and bool(parked_low_motion_meta)
        and _env_int("V2X_CARLA_FORCE_PARKED_STATIC_FOR_LOW_MOTION", 1, minimum=0, maximum=1) == 1
    ):
        pxs: List[float] = []
        pys: List[float] = []
        yaws: List[float] = []
        cli_vals: List[int] = []
        for _fr in frames:
            _x = _safe_float(_fr.get("cx"), float("nan"))
            _y = _safe_float(_fr.get("cy"), float("nan"))
            if not (math.isfinite(_x) and math.isfinite(_y)):
                _qx, _qy, _ = _frame_query_pose(_fr)
                _x = float(_qx)
                _y = float(_qy)
            if not (math.isfinite(_x) and math.isfinite(_y)):
                continue
            pxs.append(float(_x))
            pys.append(float(_y))
            _yaw = _safe_float(_fr.get("cyaw"), float("nan"))
            if not math.isfinite(_yaw):
                _yaw = _safe_float(_fr.get("yaw"), float("nan"))
            if math.isfinite(_yaw):
                yaws.append(math.radians(float(_yaw)))
            _cli = _safe_int(_fr.get("ccli"), -1)
            if _cli >= 0:
                cli_vals.append(int(_cli))
        if pxs and pys:
            ax = float(np.median(np.asarray(pxs, dtype=np.float64)))
            ay = float(np.median(np.asarray(pys, dtype=np.float64)))
            ayaw = 0.0
            if yaws:
                _sy = float(np.mean(np.sin(np.asarray(yaws, dtype=np.float64))))
                _cy = float(np.mean(np.cos(np.asarray(yaws, dtype=np.float64))))
                ayaw = float(_normalize_yaw_deg(math.degrees(math.atan2(_sy, _cy))))
            acli = int(max(set(cli_vals), key=cli_vals.count)) if cli_vals else -1
            for _fr in frames:
                qx, qy, _ = _frame_query_pose(_fr)
                _set_carla_pose(
                    frame=_fr,
                    line_index=int(acli),
                    x=float(ax),
                    y=float(ay),
                    yaw=float(ayaw),
                    dist=float(math.hypot(float(ax) - float(qx), float(ay) - float(qy))),
                    source="parked_static_low_motion",
                    quality="none",
                )
    _enforce_lane_attachment_outside_intersection(
        frames,
        carla_context=carla_context,
        role=str(role_norm),
        opposite_reject_deg=float(CARLA_OPPOSITE_DIRECTION_REJECT_DEG),
    )
    # Single-lane consolidation: if the entire track fits one CARLA lane within
    # 2.5 m, lock every frame to that lane. Eliminates the lane-bouncing where
    # a turn vehicle's snap flips between several straight lanes and the turn
    # lane (when the turn lane covers the full path on its own).
    if _env_int("V2X_CARLA_SINGLE_LANE_CONSOLIDATE_ENABLED", 1, minimum=0, maximum=1) == 1:
        _consol_ok, _consol_li = _consolidate_track_to_single_lane(
            frames,
            carla_context=carla_context,
            max_dist_threshold_m=_env_float("V2X_CARLA_SINGLE_LANE_MAX_DIST_M", 2.5),
            min_improvement_m=_env_float("V2X_CARLA_SINGLE_LANE_MIN_IMPROVEMENT_M", 0.5),
            min_frames=_env_int("V2X_CARLA_SINGLE_LANE_MIN_FRAMES", 8, minimum=2, maximum=200),
        )
        # Looser pass: for tracks the strict threshold missed but that have
        # heavy CCLI bouncing (many short runs), try a wider threshold.
        # The per-track-many-runs guard prevents legitimate multi-lane tracks
        # from being force-collapsed.
        if not _consol_ok:
            _bouncing_n_changes = 0
            for _i in range(1, len(frames)):
                if _safe_int(frames[_i].get("ccli"), -1) != _safe_int(frames[_i - 1].get("ccli"), -1):
                    _bouncing_n_changes += 1
            _bouncing_runs = _bouncing_n_changes + 1
            # Try increasingly looser thresholds. Each one only fires when
            # the previous tighter pass didn't and the bouncing rate justifies.
            for _bounce_min, _bounce_max_dist in (
                (4, _env_float("V2X_CARLA_SINGLE_LANE_BOUNCING_MAX_DIST_M", 4.0)),
                (6, _env_float("V2X_CARLA_SINGLE_LANE_BOUNCING2_MAX_DIST_M", 5.5)),
                (8, _env_float("V2X_CARLA_SINGLE_LANE_BOUNCING3_MAX_DIST_M", 7.0)),
                (10, _env_float("V2X_CARLA_SINGLE_LANE_BOUNCING4_MAX_DIST_M", 9.0)),
            ):
                if _consol_ok or _bouncing_runs < _bounce_min:
                    continue
                _consol_ok2, _consol_li2 = _consolidate_track_to_single_lane(
                    frames,
                    carla_context=carla_context,
                    max_dist_threshold_m=float(_bounce_max_dist),
                    min_improvement_m=_env_float("V2X_CARLA_SINGLE_LANE_BOUNCING_MIN_IMPROVEMENT_M", 0.3),
                    min_frames=_env_int("V2X_CARLA_SINGLE_LANE_MIN_FRAMES", 8, minimum=2, maximum=200),
                )
                if _consol_ok2:
                    _consol_ok = True
                    if _consol_li2 >= 0:
                        _consol_li = _consol_li2
                    break
        # Two-lane (segmented) consolidator: DISABLED by default — was
        # making turn-then-straight tracks worse by picking incompatible
        # line pairs. Only enable if you're sure your scenario benefits.
        if not _consol_ok and _env_int("V2X_CARLA_TWO_LANE_CONSOLIDATE_ENABLED", 0, minimum=0, maximum=1) == 1:
            _consolidate_track_two_segment(
                frames,
                carla_context=carla_context,
                max_dist_threshold_m=_env_float("V2X_CARLA_TWO_LANE_MAX_DIST_M", 3.0),
                min_segment_frames=_env_int("V2X_CARLA_TWO_LANE_MIN_SEG_FRAMES", 6, minimum=2, maximum=80),
            )
        if _env_int("V2X_CARLA_SINGLE_LANE_CONSOLIDATE_VERBOSE", 0, minimum=0, maximum=1) == 1:
            _track_label_dbg = str(track_meta.get("id", "?")) if isinstance(track_meta, dict) else "?"
            print(f"[CONSOL] track={_track_label_dbg} ok={_consol_ok} line={_consol_li}", flush=True)
    # Lane-change blend + snap-jump smooth must run LAST. Earlier smoothers
    # (e.g., _commit_intersection_connectors, _enforce_lane_attachment_outside_intersection)
    # hard-snap (cx, cy) to lane centerlines and would otherwise overwrite
    # any smoothing we did upstream.
    if _env_int("V2X_CARLA_LANE_CHANGE_BLEND_ENABLED", 1, minimum=0, maximum=1) == 1:
        # Default require_semantic_change=False so we also catch intersection
        # connector flips (CCLI changes within an intersection that don't
        # change the V2XPNP semantic lane id but still produce a visible kink).
        # Wider window (radius=6) gives smoother transitions across lane
        # changes and connector hops — shape-preserving blend means the raw
        # curvature is preserved so we don't oversmooth real maneuvers.
        _smooth_lane_change_curves(
            frames,
            window_radius=_env_int("V2X_CARLA_LANE_CHANGE_BLEND_RADIUS", 6, minimum=1, maximum=12),
            skip_synthetic_turn=_env_int("V2X_CARLA_LANE_CHANGE_BLEND_SKIP_SYNTHETIC_TURN", 1, minimum=0, maximum=1) == 1,
            require_semantic_change=_env_int("V2X_CARLA_LANE_CHANGE_BLEND_REQUIRE_SEMANTIC", 0, minimum=0, maximum=1) == 1,
        )
    if _env_int("V2X_CARLA_SNAP_JUMP_SMOOTH_ENABLED", 1, minimum=0, maximum=1) == 1:
        _smooth_snap_artifact_jumps(
            frames,
            min_jump_m=_env_float("V2X_CARLA_SNAP_JUMP_MIN_M", 1.2),
            raw_excess_m=_env_float("V2X_CARLA_SNAP_JUMP_RAW_EXCESS_M", 0.4),
            window_radius=_env_int("V2X_CARLA_SNAP_JUMP_WINDOW_RADIUS", 5, minimum=1, maximum=20),
            max_passes=_env_int("V2X_CARLA_SNAP_JUMP_MAX_PASSES", 3, minimum=1, maximum=8),
        )
    if _env_int("V2X_CARLA_JITTER_SMOOTH_ENABLED", 1, minimum=0, maximum=1) == 1:
        _smooth_snap_correction_jitter(
            frames,
            sigma_frames=_env_float("V2X_CARLA_JITTER_SIGMA_FRAMES", 1.2),
            radius_frames=_env_int("V2X_CARLA_JITTER_RADIUS_FRAMES", 3, minimum=1, maximum=8),
            min_run_frames=_env_int("V2X_CARLA_JITTER_MIN_RUN_FRAMES", 6, minimum=2, maximum=30),
            max_correction_drift_m=_env_float("V2X_CARLA_JITTER_MAX_DRIFT_M", 1.5),
        )
    # Final fine-grain pass: small-radius Gaussian over (cx, cy) across ALL
    # frames. Smooths the angular kinks at intersection-turn segment joins
    # and any remaining sub-frame jaggies. Drift cap protects real maneuvers
    # but is generous enough to round corner/connector kinks (~0.8 m).
    # Multiple passes amplify smoothing of high-frequency noise without
    # bleeding into low-frequency motion (Gaussian kernel iterated → wider).
    if _env_int("V2X_CARLA_FINE_SMOOTH_ENABLED", 1, minimum=0, maximum=1) == 1:
        for _pass in range(_env_int("V2X_CARLA_FINE_SMOOTH_PASSES", 5, minimum=1, maximum=12)):
            _smooth_fine_universal(
                frames,
                sigma_frames=_env_float("V2X_CARLA_FINE_SIGMA_FRAMES", 1.2),
                radius_frames=_env_int("V2X_CARLA_FINE_RADIUS_FRAMES", 3, minimum=1, maximum=6),
                max_drift_m=_env_float("V2X_CARLA_FINE_MAX_DRIFT_M", 1.2),
                preserve_anchors=_env_int("V2X_CARLA_FINE_PRESERVE_ANCHORS", 1, minimum=0, maximum=4),
            )
    # Detect tracks where CCLI is oscillating (e.g., snap alternating between
    # 2 parallel lanes). Apply a stronger Gaussian to kill the bouncing.
    # Trigger: > 10% of frames have CCLI change AND track has > 6 distinct CCLI runs.
    if _env_int("V2X_CARLA_OSCILLATION_SMOOTH_ENABLED", 1, minimum=0, maximum=1) == 1:
        n_frames_total = len(frames)
        if n_frames_total > 12:
            n_changes = 0
            for _i in range(1, n_frames_total):
                if _safe_int(frames[_i].get("ccli"), -1) != _safe_int(frames[_i - 1].get("ccli"), -1):
                    n_changes += 1
            change_rate = float(n_changes) / float(max(1, n_frames_total))
            if change_rate >= _env_float("V2X_CARLA_OSCILLATION_RATE_THRESH", 0.08) and n_changes >= 5:
                for _pass in range(_env_int("V2X_CARLA_OSCILLATION_PASSES", 4, minimum=1, maximum=10)):
                    _smooth_fine_universal(
                        frames,
                        sigma_frames=_env_float("V2X_CARLA_OSCILLATION_SIGMA_FRAMES", 2.0),
                        radius_frames=_env_int("V2X_CARLA_OSCILLATION_RADIUS_FRAMES", 5, minimum=1, maximum=10),
                        max_drift_m=_env_float("V2X_CARLA_OSCILLATION_MAX_DRIFT_M", 2.5),
                        preserve_anchors=_env_int("V2X_CARLA_OSCILLATION_PRESERVE_ANCHORS", 1, minimum=0, maximum=4),
                    )
    # Debug lane-change audit (set V2X_CARLA_DEBUG_ACTOR_IDS=<id>,<id> or * to enable)
    import os as _os_debug
    if str(_os_debug.environ.get("V2X_CARLA_DEBUG_ACTOR_IDS", "")).strip():
        _actor_label = str(track_meta.get("id", "?")) if isinstance(track_meta, dict) else "?"
        print(f"[CCLI DEBUG] _project_track_to_carla reached for {_actor_label} role={role_norm} frames={len(frames)}")
        _log_carla_ccli_changes(frames, actor_label=_actor_label)


def _collect_orientation_votes_from_tracks(
    tracks: List[Dict[str, object]],
    carla_context: Dict[str, object],
) -> Tuple[Dict[int, float], Dict[int, float], int]:
    carla_feats = carla_context.get("carla_feats", {})
    if not isinstance(carla_feats, dict) or not carla_feats:
        return {}, {}, 0

    stride = _env_int("V2X_CARLA_ORIENTATION_TRACK_SAMPLE_STRIDE", 4, minimum=1, maximum=20)
    nearby_top_k = _env_int("V2X_CARLA_ORIENTATION_TRACK_TOP_K", 10, minimum=1, maximum=32)
    max_proj_dist_m = _env_float("V2X_CARLA_ORIENTATION_TRACK_MAX_PROJ_DIST_M", 3.5)
    min_motion_disp_m = _env_float("V2X_CARLA_ORIENTATION_TRACK_MIN_DISP_M", 0.25)
    min_margin_deg = _env_float("V2X_CARLA_ORIENTATION_TRACK_MIN_MARGIN_DEG", 18.0)
    max_best_diff_deg = _env_float("V2X_CARLA_ORIENTATION_TRACK_MAX_BEST_DIFF_DEG", 95.0)
    vote_power = _env_float("V2X_CARLA_ORIENTATION_TRACK_VOTE_POWER", 1.2)

    rev_votes: Dict[int, float] = {}
    fwd_votes: Dict[int, float] = {}
    used_samples = 0

    for tr in tracks:
        if not isinstance(tr, dict):
            continue
        role = str(tr.get("role", tr.get("obj_type", ""))).strip().lower()
        if role not in {"ego", "vehicle"}:
            continue
        frames = tr.get("frames", [])
        if not isinstance(frames, list) or len(frames) < 3:
            continue
        for fi in range(0, len(frames), int(stride)):
            motion_yaw = _estimate_query_motion_yaw(
                frames,
                int(fi),
                max_span=6,
                min_disp_m=float(min_motion_disp_m),
            )
            if motion_yaw is None:
                continue
            qx, qy, _ = _frame_query_pose(frames[fi])
            near = _nearby_carla_lines(
                carla_context=carla_context,
                qx=float(qx),
                qy=float(qy),
                top_k=int(nearby_top_k),
            )
            if not near:
                continue
            used_samples += 1
            for ci in near:
                cf = carla_feats.get(int(ci))
                if not isinstance(cf, dict):
                    continue
                poly = cf.get("poly_xy")
                cum = cf.get("cum")
                if not isinstance(poly, np.ndarray) or poly.shape[0] < 2:
                    continue
                if not isinstance(cum, np.ndarray) or cum.shape[0] != poly.shape[0]:
                    continue
                proj = _project_point_to_polyline_xy(poly, cum, float(qx), float(qy))
                if not isinstance(proj, dict):
                    continue
                dist = _safe_float(proj.get("dist"), float("inf"))
                if (not math.isfinite(dist)) or dist > float(max_proj_dist_m):
                    continue
                fwd_yaw = _safe_float(proj.get("yaw"), float("nan"))
                if not math.isfinite(fwd_yaw):
                    continue
                rev_yaw = _normalize_yaw_deg(float(fwd_yaw) + 180.0)
                fwd_diff = float(_yaw_abs_diff_deg(float(motion_yaw), float(fwd_yaw)))
                rev_diff = float(_yaw_abs_diff_deg(float(motion_yaw), float(rev_yaw)))
                best_diff = min(float(fwd_diff), float(rev_diff))
                margin = abs(float(fwd_diff) - float(rev_diff))
                if best_diff > float(max_best_diff_deg) or margin < float(min_margin_deg):
                    continue
                w_dist = 1.0 / max(0.35, 0.6 + float(dist))
                w_margin = (float(margin) / 90.0) ** max(0.5, float(vote_power))
                w = float(max(0.05, w_dist * w_margin))
                if rev_diff + 1e-6 < fwd_diff:
                    rev_votes[int(ci)] = float(rev_votes.get(int(ci), 0.0) + float(w))
                else:
                    fwd_votes[int(ci)] = float(fwd_votes.get(int(ci), 0.0) + float(w))

    return rev_votes, fwd_votes, int(used_samples)


def _refresh_carla_orientation_from_tracks(
    tracks: List[Dict[str, object]],
    carla_context: Dict[str, object],
) -> None:
    if not isinstance(carla_context, dict):
        return
    base_rev_raw = carla_context.get("carla_line_orientation_rev_vote_base")
    base_fwd_raw = carla_context.get("carla_line_orientation_fwd_vote_base")
    if isinstance(base_rev_raw, dict):
        base_rev = {int(k): float(_safe_float(v, 0.0)) for k, v in _normalize_int_key_dict(base_rev_raw).items()}
    else:
        raw = carla_context.get("carla_line_orientation_rev_vote", {})
        base_rev = {int(k): float(_safe_float(v, 0.0)) for k, v in _normalize_int_key_dict(raw).items()}
    if isinstance(base_fwd_raw, dict):
        base_fwd = {int(k): float(_safe_float(v, 0.0)) for k, v in _normalize_int_key_dict(base_fwd_raw).items()}
    else:
        raw = carla_context.get("carla_line_orientation_fwd_vote", {})
        base_fwd = {int(k): float(_safe_float(v, 0.0)) for k, v in _normalize_int_key_dict(raw).items()}

    track_rev, track_fwd, used_samples = _collect_orientation_votes_from_tracks(
        tracks=tracks,
        carla_context=carla_context,
    )
    track_vote_weight = _env_float("V2X_CARLA_ORIENTATION_TRACK_VOTE_WEIGHT", 1.4)
    merged_rev: Dict[int, float] = {int(k): float(v) for k, v in base_rev.items()}
    merged_fwd: Dict[int, float] = {int(k): float(v) for k, v in base_fwd.items()}
    for ci, w in track_rev.items():
        merged_rev[int(ci)] = float(merged_rev.get(int(ci), 0.0) + float(track_vote_weight) * float(w))
    for ci, w in track_fwd.items():
        merged_fwd[int(ci)] = float(merged_fwd.get(int(ci), 0.0) + float(track_vote_weight) * float(w))

    line_indices: List[int] = []
    carla_feats = carla_context.get("carla_feats", {})
    if isinstance(carla_feats, dict):
        line_indices = [int(ci) for ci in carla_feats.keys()]
    if not line_indices:
        line_indices = sorted(set(list(merged_rev.keys()) + list(merged_fwd.keys())))

    orient_margin_ratio = _env_float("V2X_CARLA_ORIENTATION_HINT_MARGIN_RATIO", 1.15)
    orient_min_vote = _env_float("V2X_CARLA_ORIENTATION_HINT_MIN_VOTE", 0.8)
    merged_sign = _resolve_carla_orientation_signs(
        line_indices=line_indices,
        rev_votes=merged_rev,
        fwd_votes=merged_fwd,
        margin_ratio=float(orient_margin_ratio),
        min_vote=float(orient_min_vote),
    )
    metadata_hard_override = (
        _env_int("V2X_CARLA_ORIENTATION_METADATA_HARD_OVERRIDE", 1, minimum=0, maximum=1) == 1
    )
    if bool(metadata_hard_override):
        recs_raw = carla_context.get("carla_line_records", {})
        recs = _normalize_int_key_dict(recs_raw)
        for ci in line_indices:
            rec = recs.get(int(ci))
            if not isinstance(rec, dict):
                continue
            dsgn = rec.get("dir_sign")
            try:
                dsgn = int(dsgn) if dsgn is not None else None
            except Exception:
                dsgn = None
            if dsgn == -1:
                merged_sign[int(ci)] = -1
            elif dsgn == 1:
                merged_sign[int(ci)] = 1

    prev_sign_raw = carla_context.get("carla_line_orientation_sign", {})
    prev_sign = {int(k): _safe_int(v, 0) for k, v in _normalize_int_key_dict(prev_sign_raw).items()}
    changed = 0
    for ci in line_indices:
        if _safe_int(prev_sign.get(int(ci), 0), 0) != _safe_int(merged_sign.get(int(ci), 0), 0):
            changed += 1

    carla_context["carla_line_orientation_sign"] = merged_sign
    carla_context["carla_line_orientation_rev_vote"] = merged_rev
    carla_context["carla_line_orientation_fwd_vote"] = merged_fwd
    lines_data = carla_context.get("carla_lines_data", [])
    if isinstance(lines_data, list):
        _apply_orientation_to_lines_data(
            lines_data=lines_data,
            sign_by_line=merged_sign,
            rev_votes=merged_rev,
            fwd_votes=merged_fwd,
        )
    summary = carla_context.get("summary", {})
    if isinstance(summary, dict):
        hints = summary.get("orientation_hints", {})
        if isinstance(hints, dict):
            hints["lines_with_hint"] = int(sum(1 for v in merged_sign.values() if int(v) != 0))
            hints["reversed_hint_lines"] = int(sum(1 for v in merged_sign.values() if int(v) < 0))
            hints["forward_hint_lines"] = int(sum(1 for v in merged_sign.values() if int(v) > 0))
        summary["orientation_runtime"] = {
            "track_samples": int(used_samples),
            "changed_lines": int(changed),
        }


def apply_carla_projection_to_tracks(
    tracks: List[Dict[str, object]],
    carla_context: Optional[Dict[str, object]],
) -> None:
    if not carla_context or not bool(carla_context.get("enabled", False)):
        return
    t_start = time.perf_counter()
    total_tracks = int(len(tracks))
    total_frames = 0
    for tr in tracks:
        frames = tr.get("frames", []) if isinstance(tr, dict) else []
        if isinstance(frames, list):
            total_frames += int(len(frames))
    print(
        "[PERF] CARLA projection start: "
        f"tracks={total_tracks} frames={int(total_frames)}"
    )
    t_orientation = time.perf_counter()
    _refresh_carla_orientation_from_tracks(
        tracks=tracks,
        carla_context=carla_context,
    )
    print(
        "[PERF] CARLA projection orientation refresh: "
        f"elapsed={time.perf_counter() - t_orientation:.2f}s"
    )
    projected_tracks = 0
    projected_frames = 0
    slow_track_count = 0
    for ti, tr in enumerate(tracks, start=1):
        frames = tr.get("frames", [])
        if not isinstance(frames, list):
            continue
        role = str(tr.get("role", tr.get("obj_type", "")))
        track_id = str(tr.get("id", ""))
        print(
            "[PERF] CARLA projection track start: "
            f"idx={int(ti)}/{int(total_tracks)} id={track_id} role={role} frames={int(len(frames))}"
        )
        t_track = time.perf_counter()
        _project_track_to_carla(frames, role=role, carla_context=carla_context, track_meta=tr)
        elapsed = time.perf_counter() - t_track
        projected_tracks += 1
        projected_frames += int(len(frames))
        print(
            "[PERF] CARLA projection track done: "
            f"idx={int(ti)}/{int(total_tracks)} id={track_id} role={role} "
            f"frames={int(len(frames))} elapsed={elapsed:.2f}s"
        )
        if elapsed >= 0.50:
            slow_track_count += 1
            print(
                "[PERF] CARLA projection slow track: "
                f"id={tr.get('id', '')} role={role} frames={len(frames)} elapsed={elapsed:.2f}s"
            )
    print(
        "[PERF] CARLA projection done: "
        f"projected_tracks={int(projected_tracks)} projected_frames={int(projected_frames)} "
        f"slow_tracks_ge_0.5s={int(slow_track_count)} total_elapsed={time.perf_counter() - t_start:.2f}s"
    )
