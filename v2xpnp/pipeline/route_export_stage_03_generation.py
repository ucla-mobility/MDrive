"""Route export internals: scenario generation and actor synthesis."""

from __future__ import annotations

from v2xpnp.pipeline.route_export_stage_01_foundation import *  # noqa: F401,F403
from v2xpnp.pipeline.route_export_stage_02_alignment import *  # noqa: F401,F403



def _apply_lane_correspondence_to_payload(payload: Dict[str, object], correspondence: Dict[str, object]) -> None:
    if not bool(correspondence.get("enabled", False)):
        payload.setdefault("metadata", {})["lane_correspondence"] = {
            "enabled": False,
            "reason": str(correspondence.get("reason", "unknown")),
        }
        return

    v2_feats: Dict[int, Dict[str, object]] = dict(correspondence.get("v2_feats", {}))
    carla_feats: Dict[int, Dict[str, object]] = dict(correspondence.get("carla_feats", {}))
    lane_to_carla: Dict[int, Dict[str, object]] = dict(correspondence.get("lane_to_carla", {}))
    carla_to_lanes: Dict[int, List[int]] = dict(correspondence.get("carla_to_lanes", {}))

    lanes_raw = payload.get("map", {}).get("lanes", [])
    if isinstance(lanes_raw, list):
        for lane in lanes_raw:
            if not isinstance(lane, dict):
                continue
            li = _safe_int(lane.get("index"), -1)
            if li < 0:
                continue
            v2_label = f"r{_safe_int(lane.get('road_id'), 0)}_l{_safe_int(lane.get('lane_id'), 0)}_t{str(lane.get('lane_type', ''))}"
            lane["v2_label"] = v2_label
            m = lane_to_carla.get(int(li))
            if m is None:
                lane["carla_match"] = None
            else:
                q = str(m.get("quality", "poor"))
                split_extras = m.get("split_extra_carla_lines", [])
                lane["carla_match"] = {
                    "carla_line_index": int(m.get("carla_line_index", -1)),
                    "carla_line_label": f"c{int(m.get('carla_line_index', -1))}",
                    "reversed": bool(m.get("reversed", False)),
                    "quality": q,
                    "usable": bool(q != "poor"),
                    "score": float(m.get("score", float("inf"))),
                    "median_dist_m": float(m.get("median_dist_m", float("inf"))),
                    "p90_dist_m": float(m.get("p90_dist_m", float("inf"))),
                    "coverage_2m": float(m.get("coverage_2m", 0.0)),
                    "angle_median_deg": float(m.get("angle_median_deg", 180.0)),
                    "monotonic_ratio": float(m.get("monotonic_ratio", 0.0)),
                    "length_ratio": float(m.get("length_ratio", 0.0)),
                    "shared_carla_line": bool(m.get("shared_carla_line", False)),
                    "split_extra_carla_lines": [int(c) for c in split_extras] if split_extras else [],
                }

    carla_map = payload.get("carla_map")
    if isinstance(carla_map, dict):
        lines_raw = carla_map.get("lines", [])
        out_lines: List[Dict[str, object]] = []
        for i, ln in enumerate(lines_raw if isinstance(lines_raw, list) else []):
            poly_src = ln.get("polyline") if isinstance(ln, dict) else ln
            poly_xy = _polyline_xy_from_points(poly_src)
            if poly_xy.shape[0] < 2:
                continue
            mid = poly_xy[poly_xy.shape[0] // 2]
            matched_lanes = list(carla_to_lanes.get(int(i), []))
            matched_labels: List[str] = []
            for li in matched_lanes:
                lf = v2_feats.get(int(li))
                if lf is None:
                    matched_labels.append(f"lane_{int(li)}")
                else:
                    matched_labels.append(
                        f"r{int(lf.get('road_id', 0))}_l{int(lf.get('lane_id', 0))}_t{str(lf.get('lane_type', ''))}"
                    )
            best_quality = "none"
            for li in matched_lanes:
                m = lane_to_carla.get(int(li), {})
                q = str(m.get("quality", "poor"))
                if q == "high":
                    best_quality = "high"
                    break
                if q == "medium" and best_quality not in ("high",):
                    best_quality = "medium"
                if q == "low" and best_quality not in ("high", "medium"):
                    best_quality = "low"
                if q == "poor" and best_quality == "none":
                    best_quality = "poor"
            out_lines.append(
                {
                    "index": int(i),
                    "label": f"c{int(i)}",
                    "label_x": float(mid[0]),
                    "label_y": float(mid[1]),
                    "matched_v2_lane_indices": [int(v) for v in matched_lanes],
                    "matched_v2_labels": matched_labels,
                    "match_quality": best_quality,
                    "polyline": [[float(p[0]), float(p[1])] for p in poly_xy],
                }
            )
        carla_map["lines"] = out_lines
        carla_map["line_count"] = int(len(out_lines))

    all_tracks: List[Dict[str, object]] = []
    for key in ("ego_tracks", "actor_tracks"):
        tv = payload.get(key, [])
        if isinstance(tv, list):
            for tr in tv:
                if isinstance(tr, dict):
                    all_tracks.append(tr)

    print(f"[INFO] Applying lane correspondence projections to {len(all_tracks)} tracks...")
    _corr_apply_t0 = time.monotonic()

    # --- CARLA line connectivity for gap handling ---
    carla_succs_raw = correspondence.get("carla_successors", {})
    carla_succs: Dict[int, set] = {}
    if isinstance(carla_succs_raw, dict):
        for ci, succ in carla_succs_raw.items():
            ci_i = _safe_int(ci, -1)
            if ci_i < 0:
                continue
            if isinstance(succ, (set, list, tuple)):
                carla_succs[int(ci_i)] = {
                    int(_safe_int(v, -1))
                    for v in succ
                    if _safe_int(v, -1) >= 0
                }
    carla_preds: Dict[int, set] = {}
    for ci, succs in carla_succs.items():
        for cj in succs:
            carla_preds.setdefault(int(cj), set()).add(int(ci))
    carla_lat_adj = _carla_lateral_adjacency(carla_feats, threshold_m=5.0)

    # Build CARLA vertex index once for fast "nearby lines" retrieval.
    carla_vertex_xy: List[Tuple[float, float]] = []
    carla_vertex_line: List[int] = []
    for ci, cf in carla_feats.items():
        poly = cf.get("poly_xy")
        if not isinstance(poly, np.ndarray) or poly.shape[0] <= 0:
            continue
        for p in poly:
            carla_vertex_xy.append((float(p[0]), float(p[1])))
            carla_vertex_line.append(int(ci))
    if carla_vertex_xy:
        carla_vertex_arr = np.asarray(carla_vertex_xy, dtype=np.float64)
        carla_vertex_line_arr = np.asarray(carla_vertex_line, dtype=np.int32)
        carla_vertex_tree = cKDTree(carla_vertex_arr) if cKDTree is not None else None
    else:
        carla_vertex_arr = np.zeros((0, 2), dtype=np.float64)
        carla_vertex_line_arr = np.zeros((0,), dtype=np.int32)
        carla_vertex_tree = None

    def _nearby_carla_lines(qx: float, qy: float, top_k: int = 24) -> List[int]:
        if carla_vertex_arr.shape[0] <= 0:
            return []
        kk = max(1, min(int(top_k), int(carla_vertex_arr.shape[0])))
        if carla_vertex_tree is not None:
            _, idxs = carla_vertex_tree.query(np.asarray([float(qx), float(qy)], dtype=np.float64), k=kk)
            if np.isscalar(idxs):
                idx_list = [int(idxs)]
            else:
                idx_list = [int(v) for v in np.asarray(idxs).reshape(-1)]
        else:
            d2 = (carla_vertex_arr[:, 0] - float(qx)) ** 2 + (carla_vertex_arr[:, 1] - float(qy)) ** 2
            idx_list = [int(v) for v in np.argsort(d2)[:kk]]
        out: List[int] = []
        seen: set[int] = set()
        for vi in idx_list:
            if vi < 0 or vi >= int(carla_vertex_line_arr.shape[0]):
                continue
            ci = int(carla_vertex_line_arr[vi])
            if ci not in seen:
                seen.add(ci)
                out.append(ci)
        return out

    # Widen connectivity threshold for gap bridging (allow slightly larger gaps)
    _GAP_CONNECT_M = 6.0
    _DIR_OPPOSITE_REJECT_DEG = 95.0
    _DIR_HEADING_WEIGHT = 0.015
    _DIR_SAME_LANE_BONUS = 0.30
    _DIR_CONNECTED_BONUS = 0.12
    _DIR_DISCONNECTED_PENALTY = 0.40
    _DIR_SWITCH_PENALTY = 0.55
    _DIR_STEP_SOFT_M = 4.0
    _DIR_STEP_WEIGHT = 0.35
    _LEGAL_CONNECT_M = 3.5

    # Infer a preferred traversal orientation per CARLA line from lane matches.
    # This prevents the directional hard-filter from being bypassed by simply
    # reversing polyline order when a line has a stable mapped orientation.
    line_orient_votes: Dict[int, Dict[str, int]] = {}
    for m in lane_to_carla.values():
        if not isinstance(m, dict):
            continue
        ci = _safe_int(m.get("carla_line_index"), -1)
        if ci < 0:
            continue
        votes = line_orient_votes.setdefault(int(ci), {"fwd": 0, "rev": 0})
        if bool(m.get("reversed", False)):
            votes["rev"] = int(votes.get("rev", 0)) + 1
        else:
            votes["fwd"] = int(votes.get("fwd", 0)) + 1
    line_pref_reversed: Dict[int, bool] = {}
    for ci, votes in line_orient_votes.items():
        fwd = int(votes.get("fwd", 0))
        rev = int(votes.get("rev", 0))
        if fwd == rev:
            continue
        line_pref_reversed[int(ci)] = bool(rev > fwd)

    # Estimate intersection-like CARLA lines from endpoint density + heading diversity.
    intersection_like_lines: set[int] = set()
    _endpoint_radius_m = 9.0
    _endpoint_min_neighbors = 5
    _mid_radius_m = 16.0
    _mid_min_neighbors = 3
    _mid_heading_spread_deg = 55.0

    endpoint_xy: List[Tuple[float, float]] = []
    endpoint_owner: List[int] = []
    mid_xy: List[Tuple[float, float]] = []
    mid_owner: List[int] = []
    line_heading: Dict[int, float] = {}

    for ci, cf in carla_feats.items():
        poly = cf.get("poly_xy")
        if not isinstance(poly, np.ndarray) or poly.shape[0] < 2:
            continue
        ci_i = int(ci)
        sx, sy = float(poly[0, 0]), float(poly[0, 1])
        ex, ey = float(poly[-1, 0]), float(poly[-1, 1])
        mx, my = float(poly[poly.shape[0] // 2, 0]), float(poly[poly.shape[0] // 2, 1])
        hd = _normalize_yaw_deg(math.degrees(math.atan2(float(ey - sy), float(ex - sx))))
        line_heading[ci_i] = float(hd)
        endpoint_xy.extend([(sx, sy), (ex, ey)])
        endpoint_owner.extend([ci_i, ci_i])
        mid_xy.append((mx, my))
        mid_owner.append(ci_i)

    if endpoint_xy:
        ep_arr = np.asarray(endpoint_xy, dtype=np.float64)
        ep_tree = cKDTree(ep_arr) if cKDTree is not None else None
        for i, (x, y) in enumerate(endpoint_xy):
            if ep_tree is not None:
                near = ep_tree.query_ball_point(np.asarray([x, y], dtype=np.float64), r=float(_endpoint_radius_m))
                owners = {int(endpoint_owner[j]) for j in near if int(endpoint_owner[j]) != int(endpoint_owner[i])}
            else:
                owners = set()
                for j, (xj, yj) in enumerate(endpoint_xy):
                    if j == i:
                        continue
                    if math.hypot(float(xj - x), float(yj - y)) <= float(_endpoint_radius_m):
                        oj = int(endpoint_owner[j])
                        if oj != int(endpoint_owner[i]):
                            owners.add(oj)
            if len(owners) >= int(_endpoint_min_neighbors):
                intersection_like_lines.add(int(endpoint_owner[i]))

    if mid_xy:
        mid_arr = np.asarray(mid_xy, dtype=np.float64)
        mid_tree = cKDTree(mid_arr) if cKDTree is not None else None
        for i, (mx, my) in enumerate(mid_xy):
            if mid_tree is not None:
                near = mid_tree.query_ball_point(np.asarray([mx, my], dtype=np.float64), r=float(_mid_radius_m))
            else:
                near = [j for j, (nx, ny) in enumerate(mid_xy) if math.hypot(float(nx - mx), float(ny - my)) <= float(_mid_radius_m)]
            headings: List[float] = []
            for j in near:
                if j == i:
                    continue
                hj = line_heading.get(int(mid_owner[j]))
                if hj is not None:
                    headings.append(float(hj))
            if len(headings) < int(_mid_min_neighbors):
                continue
            spread = 0.0
            base_h = line_heading.get(int(mid_owner[i]), 0.0)
            for hj in headings:
                spread = max(spread, _yaw_abs_diff_deg(float(base_h), float(hj)))
            if spread >= float(_mid_heading_spread_deg):
                intersection_like_lines.add(int(mid_owner[i]))

    direction_opposite_frames = 0
    direction_fixed_frames = 0
    intersection_turn_runs = 0
    intersection_turn_polyline_snaps = 0
    intersection_turn_curve_only = 0
    lanechange_scurve_events = 0
    continuity_jump_repairs = 0
    illegal_lane_transition_detected = 0
    illegal_lane_transition_fixed = 0

    for tr in _progress_bar(
        all_tracks,
        total=len(all_tracks),
        desc="Lane corr projection",
        disable=len(all_tracks) < 8,
    ):
        frames = tr.get("frames", [])
        if not isinstance(frames, list):
            continue

        role_name = str(tr.get("role", tr.get("obj_type", ""))).strip().lower()
        enforce_direction = not _is_walker_like_role_name(role_name)
        motion_heading = (
            _compute_motion_heading_series_from_frames(frames, lookahead_frames=8, min_disp_m=0.35)
            if enforce_direction
            else [None] * len(frames)
        )

        def _frame_motion_yaw(fi: int) -> Optional[float]:
            if fi < 0 or fi >= len(motion_heading):
                return None
            mv = motion_heading[fi]
            if mv is None:
                return None
            mvf = float(mv)
            return mvf if math.isfinite(mvf) else None

        # Walkers/pedestrians should follow their corrected raw trajectory
        # (mx,my), not lane-centerline correspondence.
        if not enforce_direction:
            for fr in frames:
                if not isinstance(fr, dict):
                    continue
                fmx = _safe_float(fr.get("mx"), _safe_float(fr.get("x"), 0.0))
                fmy = _safe_float(fr.get("my"), _safe_float(fr.get("y"), 0.0))
                fmyaw = _safe_float(fr.get("myaw"), _safe_float(fr.get("yaw"), 0.0))
                fr["cx"] = float(fmx)
                fr["cy"] = float(fmy)
                fr["cyaw"] = float(fmyaw)
                fr["cli"] = -1
                fr["cld"] = 0.0
                fr["csource"] = "walker_raw"
                fr["cquality"] = "none"
                for tag in ("base", "cont", "turn"):
                    fr[f"cx_{tag}"] = float(fmx)
                    fr[f"cy_{tag}"] = float(fmy)
                    fr[f"cyaw_{tag}"] = float(fmyaw)
                    fr[f"cli_{tag}"] = -1
                    fr[f"cld_{tag}"] = 0.0
                    fr[f"csource_{tag}"] = "walker_raw"
                    fr[f"cquality_{tag}"] = "none"
            tr["carla_lane_changes"] = 0
            continue

        _LANECHANGE_SOURCES = {"lanechange_scurve", "lanechange_legal"}
        _LANECHANGE_MAX_STEP_M = 4.5
        _LANECHANGE_HEADING_MAX_DEG = 45.0

        def _transition_has_successor_connect(prev_cli: int, cur_cli: int) -> bool:
            return _carla_lines_connected(
                int(prev_cli),
                int(cur_cli),
                carla_feats,
                carla_succs,
                threshold_m=float(_LEGAL_CONNECT_M),
            )

        def _transition_is_lanechange_like(
            fi: int,
            prev_rm: Dict[str, object],
            cur_rm: Dict[str, object],
            prev_cli: int,
            cur_cli: int,
        ) -> bool:
            if int(prev_cli) < 0 or int(cur_cli) < 0 or int(prev_cli) == int(cur_cli):
                return False
            if int(cur_cli) not in carla_lat_adj.get(int(prev_cli), set()) and int(prev_cli) not in carla_lat_adj.get(int(cur_cli), set()):
                return False
            px = _safe_float(prev_rm.get("cx"), float("nan"))
            py = _safe_float(prev_rm.get("cy"), float("nan"))
            cxv = _safe_float(cur_rm.get("cx"), float("nan"))
            cyv = _safe_float(cur_rm.get("cy"), float("nan"))
            if not (math.isfinite(px) and math.isfinite(py) and math.isfinite(cxv) and math.isfinite(cyv)):
                return False
            step_d = math.hypot(float(cxv - px), float(cyv - py))
            if step_d > float(_LANECHANGE_MAX_STEP_M):
                return False

            prev_yaw = _safe_float(prev_rm.get("cyaw"), float("nan"))
            cur_yaw = _safe_float(cur_rm.get("cyaw"), float("nan"))
            if math.isfinite(prev_yaw) and math.isfinite(cur_yaw):
                if _yaw_abs_diff_deg(float(prev_yaw), float(cur_yaw)) > float(_LANECHANGE_HEADING_MAX_DEG):
                    return False

            mhy = _frame_motion_yaw(fi)
            if mhy is not None and math.isfinite(cur_yaw):
                if _yaw_abs_diff_deg(float(mhy), float(cur_yaw)) > float(_DIR_OPPOSITE_REJECT_DEG):
                    return False
            return True

        def _candidate_score(
            fi: int,
            cidx: int,
            proj: Dict[str, float],
            prev_cli: int,
            prev_xy: Optional[Tuple[float, float]] = None,
        ) -> float:
            score = float(proj.get("dist", float("inf")))
            if not math.isfinite(score):
                return float("inf")
            myaw = _frame_motion_yaw(fi)
            if enforce_direction and myaw is not None:
                yaw_diff = _yaw_abs_diff_deg(float(myaw), float(proj.get("yaw", 0.0)))
                if yaw_diff > float(_DIR_OPPOSITE_REJECT_DEG):
                    return float("inf")
                score += float(_DIR_HEADING_WEIGHT) * float(yaw_diff)
            if prev_cli >= 0 and cidx >= 0:
                if cidx == prev_cli:
                    score -= float(_DIR_SAME_LANE_BONUS)
                else:
                    score += float(_DIR_SWITCH_PENALTY)
                    succ_prev = carla_succs.get(int(prev_cli), set())
                    succ_cur = carla_succs.get(int(cidx), set())
                    if int(cidx) in succ_prev or int(prev_cli) in succ_cur:
                        score -= float(_DIR_CONNECTED_BONUS)
                    else:
                        score += float(_DIR_DISCONNECTED_PENALTY)
            if prev_xy is not None:
                step_d = math.hypot(float(proj.get("x", 0.0)) - float(prev_xy[0]), float(proj.get("y", 0.0)) - float(prev_xy[1]))
                if step_d > float(_DIR_STEP_SOFT_M):
                    score += float(_DIR_STEP_WEIGHT) * float(step_d - float(_DIR_STEP_SOFT_M))
            return float(score)

        def _iter_line_variants(
            cidx: int,
            mapped_reversed: Optional[bool] = None,
        ) -> List[Tuple[np.ndarray, np.ndarray, bool]]:
            cf = carla_feats.get(int(cidx))
            if cf is None:
                return []
            if mapped_reversed is not None:
                if bool(mapped_reversed):
                    return [(cf["poly_xy_rev"], cf["cum_rev"], True)]
                return [(cf["poly_xy"], cf["cum"], False)]
            pref = line_pref_reversed.get(int(cidx))
            if pref is None:
                return [
                    (cf["poly_xy"], cf["cum"], False),
                    (cf["poly_xy_rev"], cf["cum_rev"], True),
                ]
            if bool(pref):
                return [(cf["poly_xy_rev"], cf["cum_rev"], True)]
            return [(cf["poly_xy"], cf["cum"], False)]

        def _best_projection_on_line(
            fi: int,
            cidx: int,
            qx: float,
            qy: float,
            prev_cli: int,
            prev_xy: Optional[Tuple[float, float]] = None,
            mapped_reversed: Optional[bool] = None,
            force_bidir: bool = False,
        ) -> Optional[Dict[str, object]]:
            best: Optional[Dict[str, object]] = None
            if bool(force_bidir):
                cf_local = carla_feats.get(int(cidx))
                if cf_local is None:
                    return None
                variants = [
                    (cf_local["poly_xy"], cf_local["cum"], False),
                    (cf_local["poly_xy_rev"], cf_local["cum_rev"], True),
                ]
            else:
                variants = _iter_line_variants(int(cidx), mapped_reversed=mapped_reversed)
            for cxy, ccum, used_rev in variants:
                proj = _project_point_to_polyline_xy(cxy, ccum, float(qx), float(qy))
                if proj is None:
                    continue
                sc = _candidate_score(fi, int(cidx), proj, int(prev_cli), prev_xy=prev_xy)
                if not math.isfinite(sc):
                    continue
                cand = {
                    "ci": int(cidx),
                    "proj": proj,
                    "score": float(sc),
                    "reversed": bool(used_rev),
                }
                if best is None or float(cand["score"]) < float(best["score"]):
                    best = cand
            return best

        def _nearest_projection_on_line(
            cidx: int,
            qx: float,
            qy: float,
            mapped_reversed: Optional[bool] = None,
            force_bidir: bool = False,
        ) -> Optional[Dict[str, object]]:
            best: Optional[Dict[str, object]] = None
            if bool(force_bidir):
                cf_local = carla_feats.get(int(cidx))
                if cf_local is None:
                    return None
                variants = [
                    (cf_local["poly_xy"], cf_local["cum"], False),
                    (cf_local["poly_xy_rev"], cf_local["cum_rev"], True),
                ]
            else:
                variants = _iter_line_variants(int(cidx), mapped_reversed=mapped_reversed)
            for cxy, ccum, used_rev in variants:
                proj = _project_point_to_polyline_xy(cxy, ccum, float(qx), float(qy))
                if proj is None:
                    continue
                cand = {
                    "ci": int(cidx),
                    "proj": proj,
                    "reversed": bool(used_rev),
                }
                if best is None or float(proj.get("dist", float("inf"))) < float(best["proj"].get("dist", float("inf"))):
                    best = cand
            return best

        def _best_projection_any_line(
            fi: int,
            qx: float,
            qy: float,
            prev_cli: int,
            prev_xy: Optional[Tuple[float, float]] = None,
        ) -> Optional[Dict[str, object]]:
            best: Optional[Dict[str, object]] = None
            for cidx in carla_feats.keys():
                cand = _best_projection_on_line(
                    fi=fi,
                    cidx=int(cidx),
                    qx=float(qx),
                    qy=float(qy),
                    prev_cli=int(prev_cli),
                    prev_xy=prev_xy,
                    mapped_reversed=None,
                )
                if cand is None:
                    continue
                if best is None or float(cand["score"]) < float(best["score"]):
                    best = cand
            return best

        def _nearest_projection_any_line(
            qx: float,
            qy: float,
        ) -> Optional[Dict[str, object]]:
            best: Optional[Dict[str, object]] = None
            for cidx in carla_feats.keys():
                cand = _nearest_projection_on_line(
                    cidx=int(cidx),
                    qx=float(qx),
                    qy=float(qy),
                    mapped_reversed=None,
                )
                if cand is None:
                    continue
                dist = float(cand.get("proj", {}).get("dist", float("inf")))
                if not math.isfinite(dist):
                    continue
                cand_ex = {
                    "ci": int(cand.get("ci", -1)),
                    "proj": dict(cand.get("proj", {})),
                    "score": float(dist),
                }
                if best is None or float(cand_ex["score"]) < float(best["score"]):
                    best = cand_ex
            return best

        def _smoothstep01(v: float) -> float:
            vv = max(0.0, min(1.0, float(v)))
            return vv * vv * (3.0 - 2.0 * vv)

        def _cubic_bezier_pose(
            p0: Tuple[float, float],
            p1: Tuple[float, float],
            p2: Tuple[float, float],
            p3: Tuple[float, float],
            u: float,
        ) -> Tuple[float, float, float]:
            uu = max(0.0, min(1.0, float(u)))
            omt = 1.0 - uu
            omt2 = omt * omt
            uu2 = uu * uu
            x = (
                omt2 * omt * float(p0[0])
                + 3.0 * omt2 * uu * float(p1[0])
                + 3.0 * omt * uu2 * float(p2[0])
                + uu2 * uu * float(p3[0])
            )
            y = (
                omt2 * omt * float(p0[1])
                + 3.0 * omt2 * uu * float(p1[1])
                + 3.0 * omt * uu2 * float(p2[1])
                + uu2 * uu * float(p3[1])
            )
            dx = (
                3.0 * omt2 * (float(p1[0]) - float(p0[0]))
                + 6.0 * omt * uu * (float(p2[0]) - float(p1[0]))
                + 3.0 * uu2 * (float(p3[0]) - float(p2[0]))
            )
            dy = (
                3.0 * omt2 * (float(p1[1]) - float(p0[1]))
                + 6.0 * omt * uu * (float(p2[1]) - float(p1[1]))
                + 3.0 * uu2 * (float(p3[1]) - float(p2[1]))
            )
            yaw = _normalize_yaw_deg(math.degrees(math.atan2(dy, dx))) if (abs(dx) + abs(dy)) > 1e-9 else 0.0
            return float(x), float(y), float(yaw)

        # =====================================================================
        # Pass 1: Raw per-frame CARLA mapping
        # =====================================================================
        raw_mappings: List[Dict[str, object]] = []
        prev_cli_for_rank = -1
        prev_xy_for_rank: Optional[Tuple[float, float]] = None
        for fi, fr in enumerate(frames):
            if not isinstance(fr, dict):
                raw_mappings.append({})
                continue
            mx = _safe_float(fr.get("mx"), _safe_float(fr.get("x"), 0.0))
            my = _safe_float(fr.get("my"), _safe_float(fr.get("y"), 0.0))
            myaw = _safe_float(fr.get("myaw"), _safe_float(fr.get("yaw"), 0.0))
            li = _safe_int(fr.get("li"), -1)

            cx = float(mx)
            cy = float(my)
            cyaw = float(myaw)
            cli = -1
            cld = float("inf")
            mapping_quality = "none"
            source_mode = "fallback_raw"

            mapped = lane_to_carla.get(int(li))
            if mapped is not None:
                mapped_quality = str(mapped.get("quality", "poor"))
                # Poor global matches can still be locally correct for short turn
                # segments; accept only when local projection is close.
                poor_local_max_dist = 3.0
                cidx = _safe_int(mapped.get("carla_line_index"), -1)
                cf = carla_feats.get(cidx)
                lf = v2_feats.get(int(li))
                if cf is not None:
                    mapped_reversed = bool(mapped.get("reversed", False))
                    cxy = cf["poly_xy_rev"] if mapped_reversed else cf["poly_xy"]
                    ccum = cf["cum_rev"] if mapped_reversed else cf["cum"]
                    ratio_accept_dist = 5.0 if mapped_quality != "poor" else poor_local_max_dist
                    ratio = None
                    if lf is not None:
                        proj_v2 = _project_point_to_polyline_xy(lf["poly_xy"], lf["cum"], float(mx), float(my))
                        if proj_v2 is not None:
                            ratio = float(proj_v2.get("s_norm", 0.0))
                    if ratio is not None:
                        samp = _sample_polyline_at_ratio_xy(cxy, ccum, float(ratio))
                        if samp is not None:
                            ratio_d = float(math.hypot(float(samp["x"]) - mx, float(samp["y"]) - my))
                            if ratio_d <= ratio_accept_dist:
                                ratio_proj = {
                                    "x": float(samp["x"]),
                                    "y": float(samp["y"]),
                                    "yaw": float(samp["yaw"]),
                                    "dist": float(ratio_d),
                                }
                                sc_ratio = _candidate_score(
                                    fi,
                                    int(cidx),
                                    ratio_proj,
                                    int(prev_cli_for_rank),
                                    prev_xy=prev_xy_for_rank,
                                )
                                if math.isfinite(sc_ratio):
                                    cx = float(samp["x"])
                                    cy = float(samp["y"])
                                    cyaw = float(samp["yaw"])
                                    cli = int(cidx)
                                    cld = ratio_d
                                    mapping_quality = mapped_quality
                                    source_mode = "lane_correspondence"
                    # Fallback: direct projection onto the CARLA polyline
                    if cli < 0:
                        cand_mapped = _best_projection_on_line(
                            fi=fi,
                            cidx=int(cidx),
                            qx=float(mx),
                            qy=float(my),
                            prev_cli=int(prev_cli_for_rank),
                            prev_xy=prev_xy_for_rank,
                            mapped_reversed=bool(mapped_reversed),
                        )
                        if cand_mapped is not None:
                            proj_c = dict(cand_mapped.get("proj", {}))
                            proj_dist = float(proj_c.get("dist", float("inf")))
                            if mapped_quality != "poor" or proj_dist <= poor_local_max_dist:
                                cx = float(proj_c.get("x", mx))
                                cy = float(proj_c.get("y", my))
                                cyaw = float(proj_c.get("yaw", myaw))
                                cli = int(cidx)
                                cld = proj_dist
                                mapping_quality = mapped_quality
                                source_mode = "lane_projection"

            if cli < 0:
                cand_any = _best_projection_any_line(
                    fi=fi,
                    qx=float(mx),
                    qy=float(my),
                    prev_cli=int(prev_cli_for_rank),
                    prev_xy=prev_xy_for_rank,
                )
                if cand_any is not None:
                    best_proj = dict(cand_any.get("proj", {}))
                    best_ci = int(cand_any.get("ci", -1))
                    cx = float(best_proj.get("x", mx))
                    cy = float(best_proj.get("y", my))
                    cyaw = float(best_proj.get("yaw", myaw))
                    cli = int(best_ci)
                    cld = float(best_proj.get("dist", float("inf")))
                    mapping_quality = "nearest"
                    source_mode = "nearest_carla_line"
                elif enforce_direction:
                    cx = float(mx)
                    cy = float(my)
                    cyaw = float(myaw)
                    cli = -1
                    cld = 0.0
                    mapping_quality = "none"
                    source_mode = "direction_fallback_raw"

            raw_mappings.append({
                "cx": float(cx),
                "cy": float(cy),
                "cyaw": float(cyaw),
                "cli": int(cli),
                "cld": float(cld),
                "csource": str(source_mode),
                "cquality": str(mapping_quality),
            })
            if int(cli) >= 0:
                prev_cli_for_rank = int(cli)
                prev_xy_for_rank = (float(cx), float(cy))

        def _snapshot_stage(tag: str) -> None:
            suf = str(tag).strip().lower()
            if not suf:
                return
            for rm in raw_mappings:
                if not rm:
                    continue
                rm[f"cx_{suf}"] = float(rm.get("cx", 0.0))
                rm[f"cy_{suf}"] = float(rm.get("cy", 0.0))
                rm[f"cyaw_{suf}"] = float(rm.get("cyaw", 0.0))
                rm[f"cli_{suf}"] = int(rm.get("cli", -1))
                rm[f"cld_{suf}"] = float(rm.get("cld", float("inf")))
                rm[f"csource_{suf}"] = str(rm.get("csource", ""))
                rm[f"cquality_{suf}"] = str(rm.get("cquality", ""))

        if not raw_mappings:
            continue
        _snapshot_stage("base")

        # =====================================================================
        # Pass 2: Gap bridging — trajectory-level handling of unmatched regions
        # =====================================================================
        # Identify "anchored" runs (lane_correspondence / lane_projection)
        # and "gap" runs (nearest_carla_line / fallback_raw).
        # For gap runs:
        #   - If the raw cli sequence forms a CONNECTED CHAIN of CARLA segments,
        #     treat it as a natural road traversal (keep raw cli, use actor's
        #     own trajectory for cx,cy to avoid snapping to segment endpoints).
        #   - If the cli sequence is DISCONNECTED (intersection bouncing),
        #     use the actor's own (mx,my) and assign cli from nearest anchor.

        def _frame_is_anchored(fi: int) -> bool:
            if fi < 0 or fi >= len(raw_mappings):
                return False
            rm = raw_mappings[fi]
            return bool(rm) and str(rm.get("csource", "")) in ("lane_correspondence", "lane_projection")

        # Build runs of anchored / gap frames
        frame_runs: List[Tuple[int, int, bool]] = []  # (start, end, is_anchored)
        if raw_mappings:
            cur_anch = _frame_is_anchored(0)
            run_start = 0
            for fi in range(1, len(raw_mappings)):
                anch = _frame_is_anchored(fi)
                if anch != cur_anch:
                    frame_runs.append((run_start, fi - 1, cur_anch))
                    run_start = fi
                    cur_anch = anch
            frame_runs.append((run_start, len(raw_mappings) - 1, cur_anch))

        for run_idx, (rs, re, is_anch) in enumerate(frame_runs):
            if is_anch:
                continue  # already well-mapped
            gap_len = re - rs + 1
            if gap_len < 1:
                continue

            # --- Build RLE of raw cli values in this gap ---
            gap_cli_rle: List[Tuple[int, int, int]] = []  # (cli, start_fi, end_fi)
            prev_gap_cli = -999
            for fi in range(rs, re + 1):
                rm = raw_mappings[fi]
                if not rm:
                    continue
                cli_val = int(rm.get("cli", -1))
                if cli_val != prev_gap_cli:
                    gap_cli_rle.append((cli_val, fi, fi))
                    prev_gap_cli = cli_val
                else:
                    gap_cli_rle[-1] = (gap_cli_rle[-1][0], gap_cli_rle[-1][1], fi)

            # --- Check if consecutive cli values form a connected chain ---
            is_connected_chain = True
            if len(gap_cli_rle) > 1:
                for ri in range(1, len(gap_cli_rle)):
                    ci_prev = gap_cli_rle[ri - 1][0]
                    ci_curr = gap_cli_rle[ri][0]
                    if ci_prev < 0 or ci_curr < 0:
                        is_connected_chain = False
                        break
                    if not _carla_lines_connected(ci_prev, ci_curr, carla_feats, carla_succs, _GAP_CONNECT_M):
                        is_connected_chain = False
                        break
            elif len(gap_cli_rle) == 1:
                is_connected_chain = True  # single segment, trivially connected

            # --- Find nearest boundary anchors ---
            prev_anchor_fi = -1
            next_anchor_fi = -1
            for ri in range(run_idx - 1, -1, -1):
                if frame_runs[ri][2]:  # is_anchored
                    prev_anchor_fi = frame_runs[ri][1]  # last frame of prev anchor
                    break
            for ri in range(run_idx + 1, len(frame_runs)):
                if frame_runs[ri][2]:
                    next_anchor_fi = frame_runs[ri][0]  # first frame of next anchor
                    break

            prev_anchor_cli = int(raw_mappings[prev_anchor_fi].get("cli", -1)) if prev_anchor_fi >= 0 else -1
            next_anchor_cli = int(raw_mappings[next_anchor_fi].get("cli", -1)) if next_anchor_fi >= 0 else -1

            if is_connected_chain and len(gap_cli_rle) <= 8:
                # ----- CONNECTED CHAIN: natural road traversal -----
                # Keep the raw cli but use actor's own (mx,my) — no snap.
                for fi in range(rs, re + 1):
                    fr = frames[fi]
                    if not isinstance(fr, dict) or not raw_mappings[fi]:
                        continue
                    fmx = _safe_float(fr.get("mx"), _safe_float(fr.get("x"), 0.0))
                    fmy = _safe_float(fr.get("my"), _safe_float(fr.get("y"), 0.0))
                    fmyaw = _safe_float(fr.get("myaw"), _safe_float(fr.get("yaw"), 0.0))
                    raw_mappings[fi]["cx"] = float(fmx)
                    raw_mappings[fi]["cy"] = float(fmy)
                    raw_mappings[fi]["cyaw"] = float(fmyaw)
                    raw_mappings[fi]["cld"] = 0.0
                    raw_mappings[fi]["csource"] = "gap_chain"
                    raw_mappings[fi]["cquality"] = "chain"
            else:
                # ----- DISCONNECTED GAP: intersection or scattered segments -----
                # Use actor's own (mx,my) — no snap.  Assign cli from nearest
                # boundary anchor so the lane-id is reasonable but cx,cy tracks
                # the actor's actual position.
                anchor_cli = prev_anchor_cli if prev_anchor_cli >= 0 else next_anchor_cli
                for fi in range(rs, re + 1):
                    fr = frames[fi]
                    if not isinstance(fr, dict) or not raw_mappings[fi]:
                        continue
                    fmx = _safe_float(fr.get("mx"), _safe_float(fr.get("x"), 0.0))
                    fmy = _safe_float(fr.get("my"), _safe_float(fr.get("y"), 0.0))
                    fmyaw = _safe_float(fr.get("myaw"), _safe_float(fr.get("yaw"), 0.0))
                    raw_mappings[fi]["cx"] = float(fmx)
                    raw_mappings[fi]["cy"] = float(fmy)
                    raw_mappings[fi]["cyaw"] = float(fmyaw)
                    raw_mappings[fi]["cli"] = int(anchor_cli) if anchor_cli >= 0 else int(raw_mappings[fi].get("cli", -1))
                    raw_mappings[fi]["cld"] = 0.0
                    raw_mappings[fi]["csource"] = "gap_bridge"
                    raw_mappings[fi]["cquality"] = "bridge"

        # =====================================================================
        # Pass 2.5: Snap gap frames to nearest CARLA line
        # =====================================================================
        # Gap frames currently use raw (mx,my).  In many cases they are
        # inside an intersection where a CARLA *junction lane* (turn
        # polyline) exists nearby.  Projecting onto the nearest CARLA
        # line produces a much smoother visual result.
        _GAP_SNAP_DIST = 8.0  # max allowed projection distance for gap snap
        for fi in range(len(raw_mappings)):
            rm = raw_mappings[fi]
            if not rm or fi >= len(frames) or not isinstance(frames[fi], dict):
                continue
            src = str(rm.get("csource", ""))
            if src not in ("gap_chain", "gap_bridge"):
                continue
            fmx = _safe_float(frames[fi].get("mx"), _safe_float(frames[fi].get("x"), 0.0))
            fmy = _safe_float(frames[fi].get("my"), _safe_float(frames[fi].get("y"), 0.0))
            fmyaw = _safe_float(frames[fi].get("myaw"), _safe_float(frames[fi].get("yaw"), 0.0))
            prev_cli_gap = int(rm.get("cli", -1))
            prev_xy_gap: Optional[Tuple[float, float]] = None
            if fi > 0 and fi - 1 < len(raw_mappings) and raw_mappings[fi - 1]:
                prev_cli_gap = int(raw_mappings[fi - 1].get("cli", prev_cli_gap))
                prev_xy_gap = (
                    float(raw_mappings[fi - 1].get("cx", fmx)),
                    float(raw_mappings[fi - 1].get("cy", fmy)),
                )
            best_gap = _best_projection_any_line(
                fi=fi,
                qx=float(fmx),
                qy=float(fmy),
                prev_cli=int(prev_cli_gap),
                prev_xy=prev_xy_gap,
            )
            if best_gap is not None and float(best_gap["proj"].get("dist", float("inf"))) <= _GAP_SNAP_DIST:
                best_proj = dict(best_gap.get("proj", {}))
                rm["cx"] = float(best_proj.get("x", fmx))
                rm["cy"] = float(best_proj.get("y", fmy))
                rm["cyaw"] = float(best_proj.get("yaw", fmyaw))
                rm["cli"] = int(best_gap.get("ci", -1))
                rm["cld"] = float(best_proj.get("dist", 0.0))
                rm["csource"] = "gap_snap"
            # else: leave as raw (mx,my)

        # =====================================================================
        # Pass 3: Trajectory-level cli sequence smoothing
        # =====================================================================
        # Build run-length encoding of cli across ALL frames and remove very
        # short runs by merging into the adjacent longer run.
        MIN_CLI_RUN = 5  # minimum frames for a cli run to survive
        cli_seq = [int(rm.get("cli", -1)) if rm else -1 for rm in raw_mappings]
        cli_rle: List[List] = []  # [[cli, start, end]]
        for fi, cli_val in enumerate(cli_seq):
            if not cli_rle or cli_val != cli_rle[-1][0]:
                cli_rle.append([cli_val, fi, fi])
            else:
                cli_rle[-1][2] = fi

        # Remove short runs by merging into neighbours
        changed = True
        while changed:
            changed = False
            new_rle: List[List] = []
            for entry in cli_rle:
                cli_val, es, ee = entry
                run_len = ee - es + 1
                if run_len < MIN_CLI_RUN and len(new_rle) > 0:
                    # Merge into previous run
                    new_rle[-1][2] = ee
                    changed = True
                else:
                    new_rle.append(entry)
            cli_rle = new_rle
            # Also merge trailing short runs
            if len(cli_rle) >= 2:
                last = cli_rle[-1]
                if last[2] - last[1] + 1 < MIN_CLI_RUN:
                    cli_rle[-2][2] = last[2]
                    cli_rle.pop()
                    changed = True

        # Apply smoothed cli and re-project affected frames
        MAX_SNAP_DIST = 8.0  # safety cap: if projection > this, use mx,my
        for entry in cli_rle:
            cli_val, es, ee = entry
            for fi in range(es, ee + 1):
                rm = raw_mappings[fi]
                if not rm:
                    continue
                old_cli = int(rm.get("cli", -1))
                if old_cli != cli_val and cli_val >= 0:
                    # Skip re-projection for raw gap frames (they use mx,my)
                    src = str(rm.get("csource", ""))
                    if src in ("gap_chain", "gap_bridge"):
                        continue
                    if fi < len(frames) and isinstance(frames[fi], dict):
                        fmx = _safe_float(frames[fi].get("mx"), _safe_float(frames[fi].get("x"), 0.0))
                        fmy = _safe_float(frames[fi].get("my"), _safe_float(frames[fi].get("y"), 0.0))
                        prev_xy_sm = (
                            float(raw_mappings[fi - 1].get("cx", fmx)),
                            float(raw_mappings[fi - 1].get("cy", fmy)),
                        ) if fi > 0 and raw_mappings[fi - 1] else None
                        cand_cli = _best_projection_on_line(
                            fi=fi,
                            cidx=int(cli_val),
                            qx=float(fmx),
                            qy=float(fmy),
                            prev_cli=int(old_cli),
                            prev_xy=prev_xy_sm,
                            mapped_reversed=None,
                        )
                        if cand_cli is not None and float(cand_cli["proj"].get("dist", float("inf"))) <= MAX_SNAP_DIST:
                            proj = dict(cand_cli.get("proj", {}))
                            rm["cli"] = int(cand_cli.get("ci", int(cli_val)))
                            rm["cx"] = float(proj.get("x", fmx))
                            rm["cy"] = float(proj.get("y", fmy))
                            rm["cyaw"] = float(proj.get("yaw", _safe_float(frames[fi].get("myaw"), 0.0)))
                            rm["cld"] = float(proj.get("dist", 0.0))
                            rm["csource"] = "smoothed" if src != "gap_snap" else "gap_snap"
                        else:
                            cand_alt = _best_projection_any_line(
                                fi=fi,
                                qx=float(fmx),
                                qy=float(fmy),
                                prev_cli=int(old_cli),
                                prev_xy=prev_xy_sm,
                            )
                            if cand_alt is not None and float(cand_alt["proj"].get("dist", float("inf"))) <= MAX_SNAP_DIST:
                                proj = dict(cand_alt.get("proj", {}))
                                rm["cli"] = int(cand_alt.get("ci", -1))
                                rm["cx"] = float(proj.get("x", fmx))
                                rm["cy"] = float(proj.get("y", fmy))
                                rm["cyaw"] = float(proj.get("yaw", _safe_float(frames[fi].get("myaw"), 0.0)))
                                rm["cld"] = float(proj.get("dist", 0.0))
                                rm["csource"] = "smoothed_redirect" if src != "gap_snap" else "gap_snap"
                            else:
                                # Keep raw pose when no legal nearby projection exists.
                                fmyaw = _safe_float(frames[fi].get("myaw"), _safe_float(frames[fi].get("yaw"), 0.0))
                                rm["cli"] = -1
                                rm["cx"] = float(fmx)
                                rm["cy"] = float(fmy)
                                rm["cyaw"] = float(fmyaw)
                                rm["cld"] = 0.0
                                rm["csource"] = "smoothed_fallback"

        # =====================================================================
        # Pass 4: A→B→A lane-change suppression
        # =====================================================================
        # Detect patterns where cli goes A→B→A and B is short.  These are
        # false lane changes — collapse B into A.
        # Rebuild cli_rle from the (possibly smoothed) raw_mappings.
        cli_seq2 = [int(rm.get("cli", -1)) if rm else -1 for rm in raw_mappings]
        cli_rle2: List[List] = []
        for fi, cv in enumerate(cli_seq2):
            if not cli_rle2 or cv != cli_rle2[-1][0]:
                cli_rle2.append([cv, fi, fi])
            else:
                cli_rle2[-1][2] = fi

        MAX_ABA_FRAMES = 15  # B run must be <= this to be collapsed
        aba_changed = True
        while aba_changed:
            aba_changed = False
            for ri in range(1, len(cli_rle2) - 1):
                a_cli = cli_rle2[ri - 1][0]
                b_cli = cli_rle2[ri][0]
                c_cli = cli_rle2[ri + 1][0]
                b_len = cli_rle2[ri][2] - cli_rle2[ri][1] + 1
                if a_cli == c_cli and a_cli >= 0 and b_cli != a_cli and b_len <= MAX_ABA_FRAMES:
                    # Collapse B into A
                    cli_rle2[ri - 1][2] = cli_rle2[ri + 1][2]
                    cli_rle2.pop(ri + 1)
                    cli_rle2.pop(ri)
                    aba_changed = True
                    break  # restart scan after modification

        # Apply A→B→A suppression and re-project collapsed frames
        for entry in cli_rle2:
            cli_val, es, ee = entry
            for fi in range(es, ee + 1):
                rm = raw_mappings[fi]
                if not rm:
                    continue
                old_cli = int(rm.get("cli", -1))
                if old_cli != cli_val and cli_val >= 0:
                    src = str(rm.get("csource", ""))
                    if src in ("gap_chain", "gap_bridge"):
                        continue
                    if fi < len(frames) and isinstance(frames[fi], dict):
                        fmx = _safe_float(frames[fi].get("mx"), _safe_float(frames[fi].get("x"), 0.0))
                        fmy = _safe_float(frames[fi].get("my"), _safe_float(frames[fi].get("y"), 0.0))
                        prev_xy_aba = (
                            float(raw_mappings[fi - 1].get("cx", fmx)),
                            float(raw_mappings[fi - 1].get("cy", fmy)),
                        ) if fi > 0 and raw_mappings[fi - 1] else None
                        cand_cli = _best_projection_on_line(
                            fi=fi,
                            cidx=int(cli_val),
                            qx=float(fmx),
                            qy=float(fmy),
                            prev_cli=int(old_cli),
                            prev_xy=prev_xy_aba,
                            mapped_reversed=None,
                        )
                        if cand_cli is not None and float(cand_cli["proj"].get("dist", float("inf"))) <= MAX_SNAP_DIST:
                            proj = dict(cand_cli.get("proj", {}))
                            rm["cli"] = int(cand_cli.get("ci", int(cli_val)))
                            rm["cx"] = float(proj.get("x", fmx))
                            rm["cy"] = float(proj.get("y", fmy))
                            rm["cyaw"] = float(proj.get("yaw", _safe_float(frames[fi].get("myaw"), 0.0)))
                            rm["cld"] = float(proj.get("dist", 0.0))
                            rm["csource"] = "aba_suppress" if src != "gap_snap" else "gap_snap"
                        else:
                            cand_alt = _best_projection_any_line(
                                fi=fi,
                                qx=float(fmx),
                                qy=float(fmy),
                                prev_cli=int(old_cli),
                                prev_xy=prev_xy_aba,
                            )
                            if cand_alt is not None and float(cand_alt["proj"].get("dist", float("inf"))) <= MAX_SNAP_DIST:
                                proj = dict(cand_alt.get("proj", {}))
                                rm["cli"] = int(cand_alt.get("ci", -1))
                                rm["cx"] = float(proj.get("x", fmx))
                                rm["cy"] = float(proj.get("y", fmy))
                                rm["cyaw"] = float(proj.get("yaw", _safe_float(frames[fi].get("myaw"), 0.0)))
                                rm["cld"] = float(proj.get("dist", 0.0))
                                rm["csource"] = "aba_redirect" if src != "gap_snap" else "gap_snap"
                            else:
                                fmyaw = _safe_float(frames[fi].get("myaw"), _safe_float(frames[fi].get("yaw"), 0.0))
                                rm["cli"] = -1
                                rm["cx"] = float(fmx)
                                rm["cy"] = float(fmy)
                                rm["cyaw"] = float(fmyaw)
                                rm["cld"] = 0.0
                                rm["csource"] = "aba_fallback"

        # =====================================================================
        # Pass 5: Final safety cap — prevent any remaining huge gaps
        # =====================================================================
        for fi in range(len(raw_mappings)):
            rm = raw_mappings[fi]
            if not rm or fi >= len(frames) or not isinstance(frames[fi], dict):
                continue
            cld_val = float(rm.get("cld", 0.0))
            if cld_val > MAX_SNAP_DIST:
                fmx = _safe_float(frames[fi].get("mx"), _safe_float(frames[fi].get("x"), 0.0))
                fmy = _safe_float(frames[fi].get("my"), _safe_float(frames[fi].get("y"), 0.0))
                fmyaw = _safe_float(frames[fi].get("myaw"), _safe_float(frames[fi].get("yaw"), 0.0))
                prev_cli_safety = int(rm.get("cli", -1))
                prev_xy_safety: Optional[Tuple[float, float]] = None
                if fi > 0 and fi - 1 < len(raw_mappings) and raw_mappings[fi - 1]:
                    prev_cli_safety = int(raw_mappings[fi - 1].get("cli", prev_cli_safety))
                    prev_xy_safety = (
                        float(raw_mappings[fi - 1].get("cx", fmx)),
                        float(raw_mappings[fi - 1].get("cy", fmy)),
                    )
                best_safety = _best_projection_any_line(
                    fi=fi,
                    qx=float(fmx),
                    qy=float(fmy),
                    prev_cli=int(prev_cli_safety),
                    prev_xy=prev_xy_safety,
                )
                if best_safety is not None and float(best_safety["proj"].get("dist", float("inf"))) <= MAX_SNAP_DIST:
                    best_proj = dict(best_safety.get("proj", {}))
                    rm["cx"] = float(best_proj.get("x", fmx))
                    rm["cy"] = float(best_proj.get("y", fmy))
                    rm["cyaw"] = float(best_proj.get("yaw", fmyaw))
                    rm["cli"] = int(best_safety.get("ci", -1))
                    rm["cld"] = float(best_proj.get("dist", 0.0))
                    rm["csource"] = "safety_nearest"
                else:
                    rm["cx"] = float(fmx)
                    rm["cy"] = float(fmy)
                    rm["cyaw"] = float(fmyaw)
                    rm["cli"] = -1
                    rm["cld"] = 0.0
                    rm["csource"] = "safety_fallback"

        # =====================================================================
        # Pass 5.5: Hard wrong-way rejection for motorized tracks
        # =====================================================================
        if enforce_direction:
            for fi in range(len(raw_mappings)):
                rm = raw_mappings[fi]
                if not rm or fi >= len(frames) or not isinstance(frames[fi], dict):
                    continue
                myaw = _frame_motion_yaw(fi)
                if myaw is None:
                    continue
                cyaw = _safe_float(rm.get("cyaw"), float("nan"))
                if not math.isfinite(cyaw):
                    continue
                yaw_diff = _yaw_abs_diff_deg(float(myaw), float(cyaw))
                if yaw_diff <= float(_DIR_OPPOSITE_REJECT_DEG):
                    continue
                direction_opposite_frames += 1

                fmx = _safe_float(frames[fi].get("mx"), _safe_float(frames[fi].get("x"), 0.0))
                fmy = _safe_float(frames[fi].get("my"), _safe_float(frames[fi].get("y"), 0.0))
                fmyaw = _safe_float(frames[fi].get("myaw"), _safe_float(frames[fi].get("yaw"), 0.0))
                prev_cli_dir = int(rm.get("cli", -1))
                prev_xy_dir: Optional[Tuple[float, float]] = None
                if fi > 0 and fi - 1 < len(raw_mappings) and raw_mappings[fi - 1]:
                    prev_cli_dir = int(raw_mappings[fi - 1].get("cli", prev_cli_dir))
                    prev_xy_dir = (
                        float(raw_mappings[fi - 1].get("cx", fmx)),
                        float(raw_mappings[fi - 1].get("cy", fmy)),
                    )

                cand_dir = _best_projection_any_line(
                    fi=fi,
                    qx=float(fmx),
                    qy=float(fmy),
                    prev_cli=int(prev_cli_dir),
                    prev_xy=prev_xy_dir,
                )
                if cand_dir is not None and float(cand_dir["proj"].get("dist", float("inf"))) <= MAX_SNAP_DIST:
                    proj = dict(cand_dir.get("proj", {}))
                    rm["cx"] = float(proj.get("x", fmx))
                    rm["cy"] = float(proj.get("y", fmy))
                    rm["cyaw"] = float(proj.get("yaw", fmyaw))
                    rm["cli"] = int(cand_dir.get("ci", -1))
                    rm["cld"] = float(proj.get("dist", 0.0))
                    rm["csource"] = "direction_fix"
                    rm["cquality"] = "direction_legal"
                    direction_fixed_frames += 1
                else:
                    rm["cx"] = float(fmx)
                    rm["cy"] = float(fmy)
                    rm["cyaw"] = float(fmyaw)
                    rm["cli"] = -1
                    rm["cld"] = 0.0
                    rm["csource"] = "direction_fallback"
                    rm["cquality"] = "direction_none"
                    direction_fixed_frames += 1

        _snapshot_stage("cont")

        # =====================================================================
        # Pass 6: Intersection turn smoothing (single-polyline lock or curve)
        # =====================================================================
        if enforce_direction and len(raw_mappings) >= 5:
            inter_mask = [False] * len(raw_mappings)
            for fi, rm in enumerate(raw_mappings):
                if not rm:
                    continue
                ci = int(rm.get("cli", -1))
                if ci >= 0 and ci in intersection_like_lines:
                    inter_mask[fi] = True
                if fi > 0 and raw_mappings[fi - 1]:
                    ci_prev = int(raw_mappings[fi - 1].get("cli", -1))
                    if ci >= 0 and ci_prev >= 0 and ci != ci_prev:
                        for fj in range(max(0, fi - 2), min(len(raw_mappings), fi + 3)):
                            inter_mask[fj] = True

            raw_runs: List[Tuple[int, int]] = []
            run_s = -1
            for fi, flag in enumerate(inter_mask):
                if flag and run_s < 0:
                    run_s = fi
                elif (not flag) and run_s >= 0:
                    raw_runs.append((run_s, fi - 1))
                    run_s = -1
            if run_s >= 0:
                raw_runs.append((run_s, len(inter_mask) - 1))

            merged_runs: List[Tuple[int, int]] = []
            for rs, re in raw_runs:
                s = max(0, rs - 2)
                e = min(len(raw_mappings) - 1, re + 2)
                if merged_runs and s <= merged_runs[-1][1] + 1:
                    merged_runs[-1] = (merged_runs[-1][0], max(merged_runs[-1][1], e))
                else:
                    merged_runs.append((s, e))

            for rs, re in merged_runs:
                if re - rs + 1 < 3:
                    continue

                run_clis = [int(raw_mappings[fi].get("cli", -1)) for fi in range(rs, re + 1) if raw_mappings[fi]]
                run_clis = [ci for ci in run_clis if ci >= 0]
                if len(set(run_clis)) < 2:
                    continue
                has_intersection_lane = any(int(ci) in intersection_like_lines for ci in run_clis)

                a_idx = rs - 1
                while a_idx >= 0 and (not raw_mappings[a_idx]):
                    a_idx -= 1
                b_idx = re + 1
                while b_idx < len(raw_mappings) and (not raw_mappings[b_idx]):
                    b_idx += 1
                if a_idx < 0 or b_idx >= len(raw_mappings):
                    continue
                if not raw_mappings[a_idx] or not raw_mappings[b_idx]:
                    continue

                p0 = (
                    float(raw_mappings[a_idx].get("cx", 0.0)),
                    float(raw_mappings[a_idx].get("cy", 0.0)),
                )
                p3 = (
                    float(raw_mappings[b_idx].get("cx", 0.0)),
                    float(raw_mappings[b_idx].get("cy", 0.0)),
                )
                if math.hypot(float(p3[0] - p0[0]), float(p3[1] - p0[1])) < 2.0:
                    continue

                y0 = _frame_motion_yaw(a_idx)
                if y0 is None:
                    y0 = _safe_float(raw_mappings[a_idx].get("cyaw"), 0.0)
                y3 = _frame_motion_yaw(b_idx)
                if y3 is None:
                    y3 = _safe_float(raw_mappings[b_idx].get("cyaw"), 0.0)
                if y0 is None or y3 is None:
                    continue
                heading_delta = _yaw_abs_diff_deg(float(y0), float(y3))
                if (not has_intersection_lane) and heading_delta < 25.0 and len(set(run_clis)) <= 2:
                    # Straight lane switches are handled by dedicated lane-change S-curve pass.
                    continue

                chord = math.hypot(float(p3[0] - p0[0]), float(p3[1] - p0[1]))
                handle = max(1.5, min(10.0, 0.35 * float(chord)))
                p1 = (
                    float(p0[0] + handle * math.cos(math.radians(float(y0)))),
                    float(p0[1] + handle * math.sin(math.radians(float(y0)))),
                )
                p2 = (
                    float(p3[0] - handle * math.cos(math.radians(float(y3)))),
                    float(p3[1] - handle * math.sin(math.radians(float(y3)))),
                )

                curve_indices = list(range(rs, re + 1))
                m = len(curve_indices)
                curve_pts: List[Tuple[float, float, float]] = []
                for k in range(m):
                    u = float(k + 1) / float(m + 1)
                    curve_pts.append(_cubic_bezier_pose(p0, p1, p2, p3, u))

                candidate_lines: set[int] = set(int(ci) for ci in run_clis if int(ci) >= 0)
                for ci in list(candidate_lines):
                    for cj in carla_succs.get(int(ci), set()):
                        candidate_lines.add(int(cj))
                for ci in carla_feats.keys():
                    cf = carla_feats.get(int(ci))
                    if cf is None:
                        continue
                    proj_s = _project_point_to_polyline_xy(cf["poly_xy"], cf["cum"], float(p0[0]), float(p0[1]))
                    proj_e = _project_point_to_polyline_xy(cf["poly_xy"], cf["cum"], float(p3[0]), float(p3[1]))
                    ds = float(proj_s["dist"]) if proj_s is not None else float("inf")
                    de = float(proj_e["dist"]) if proj_e is not None else float("inf")
                    if min(ds, de) <= 12.0:
                        candidate_lines.add(int(ci))

                best_match: Optional[Dict[str, object]] = None
                for cidx in sorted(candidate_lines):
                    for cxy, ccum, _ in _iter_line_variants(int(cidx), mapped_reversed=None):
                        dists: List[float] = []
                        ydiffs: List[float] = []
                        projs: List[Dict[str, float]] = []
                        ok = True
                        for (cxv, cyv, cyawv) in curve_pts:
                            proj = _project_point_to_polyline_xy(cxy, ccum, float(cxv), float(cyv))
                            if proj is None:
                                ok = False
                                break
                            yd = _yaw_abs_diff_deg(float(cyawv), float(proj.get("yaw", 0.0)))
                            if yd > float(_DIR_OPPOSITE_REJECT_DEG):
                                ok = False
                                break
                            dists.append(float(proj["dist"]))
                            ydiffs.append(float(yd))
                            projs.append(proj)
                        if not ok or not dists:
                            continue
                        mean_d = float(np.mean(np.asarray(dists, dtype=np.float64)))
                        p90_d = float(np.quantile(np.asarray(dists, dtype=np.float64), 0.9))
                        mean_y = float(np.mean(np.asarray(ydiffs, dtype=np.float64)))
                        score = mean_d + 0.45 * p90_d + 0.012 * mean_y
                        if int(cidx) == int(raw_mappings[a_idx].get("cli", -1)):
                            score -= 0.25
                        if int(cidx) == int(raw_mappings[b_idx].get("cli", -1)):
                            score -= 0.25
                        cand = {
                            "ci": int(cidx),
                            "score": float(score),
                            "mean_d": float(mean_d),
                            "p90_d": float(p90_d),
                            "projs": projs,
                        }
                        if best_match is None or float(cand["score"]) < float(best_match["score"]):
                            best_match = cand

                use_polyline = bool(
                    best_match is not None
                    and float(best_match.get("mean_d", float("inf"))) <= 3.4
                    and float(best_match.get("p90_d", float("inf"))) <= 6.8
                )
                intersection_turn_runs += 1
                if use_polyline:
                    intersection_turn_polyline_snaps += 1
                    projs = list(best_match.get("projs", []))
                    for k, fi in enumerate(curve_indices):
                        if fi >= len(raw_mappings) or not raw_mappings[fi] or k >= len(projs):
                            continue
                        proj = projs[k]
                        raw_mappings[fi]["cx"] = float(proj.get("x", raw_mappings[fi].get("cx", 0.0)))
                        raw_mappings[fi]["cy"] = float(proj.get("y", raw_mappings[fi].get("cy", 0.0)))
                        raw_mappings[fi]["cyaw"] = float(proj.get("yaw", raw_mappings[fi].get("cyaw", 0.0)))
                        raw_mappings[fi]["cli"] = int(best_match.get("ci", -1))
                        raw_mappings[fi]["cld"] = float(proj.get("dist", 0.0))
                        raw_mappings[fi]["csource"] = "intersection_polyline"
                        raw_mappings[fi]["cquality"] = "intersection"
                else:
                    intersection_turn_curve_only += 1
                    for k, fi in enumerate(curve_indices):
                        if fi >= len(raw_mappings) or not raw_mappings[fi] or k >= len(curve_pts):
                            continue
                        cxv, cyv, cyawv = curve_pts[k]
                        prev_cli_curve = int(raw_mappings[fi - 1].get("cli", -1)) if fi > 0 and raw_mappings[fi - 1] else -1
                        prev_xy_curve = (
                            float(raw_mappings[fi - 1].get("cx", cxv)),
                            float(raw_mappings[fi - 1].get("cy", cyv)),
                        ) if fi > 0 and raw_mappings[fi - 1] else None
                        cand_curve = _best_projection_any_line(
                            fi=fi,
                            qx=float(cxv),
                            qy=float(cyv),
                            prev_cli=int(prev_cli_curve),
                            prev_xy=prev_xy_curve,
                        )
                        raw_mappings[fi]["cx"] = float(cxv)
                        raw_mappings[fi]["cy"] = float(cyv)
                        raw_mappings[fi]["cyaw"] = float(cyawv)
                        if cand_curve is not None and float(cand_curve["proj"].get("dist", float("inf"))) <= 8.0:
                            raw_mappings[fi]["cli"] = int(cand_curve.get("ci", -1))
                            raw_mappings[fi]["cld"] = float(cand_curve["proj"].get("dist", 0.0))
                        else:
                            raw_mappings[fi]["cli"] = -1
                            raw_mappings[fi]["cld"] = 0.0
                        raw_mappings[fi]["csource"] = "intersection_curve"
                        raw_mappings[fi]["cquality"] = "intersection"

        _snapshot_stage("turn")

        # =====================================================================
        # Pass 7: Smooth lane changes with an S-curve blend
        # =====================================================================
        if enforce_direction and len(raw_mappings) >= 6:
            cli_runs: List[Tuple[int, int, int]] = []
            r_cli = int(raw_mappings[0].get("cli", -1)) if raw_mappings and raw_mappings[0] else -1
            r_s = 0
            for fi in range(1, len(raw_mappings)):
                ci = int(raw_mappings[fi].get("cli", -1)) if raw_mappings[fi] else -1
                if ci != r_cli:
                    cli_runs.append((int(r_cli), int(r_s), int(fi - 1)))
                    r_cli = int(ci)
                    r_s = int(fi)
            cli_runs.append((int(r_cli), int(r_s), int(len(raw_mappings) - 1)))

            last_used_end = -1
            for ri in range(len(cli_runs) - 1):
                a_cli, a_s, a_e = cli_runs[ri]
                b_cli, b_s, b_e = cli_runs[ri + 1]
                if a_cli < 0 or b_cli < 0 or a_cli == b_cli:
                    continue
                len_a = a_e - a_s + 1
                len_b = b_e - b_s + 1
                if len_a < 3 or len_b < 3:
                    continue

                pre = min(10, max(4, len_a // 2))
                post = min(10, max(4, len_b // 2))
                ws = max(a_s, a_e - pre + 1)
                we = min(b_e, b_s + post - 1)
                if we - ws + 1 < 6:
                    continue
                if ws <= last_used_end:
                    continue

                proj_a: List[Dict[str, float]] = []
                proj_b: List[Dict[str, float]] = []
                valid = True
                for fi in range(ws, we + 1):
                    if fi >= len(frames) or not isinstance(frames[fi], dict):
                        valid = False
                        break
                    fmx = _safe_float(frames[fi].get("mx"), _safe_float(frames[fi].get("x"), 0.0))
                    fmy = _safe_float(frames[fi].get("my"), _safe_float(frames[fi].get("y"), 0.0))
                    pa = _best_projection_on_line(
                        fi=fi,
                        cidx=int(a_cli),
                        qx=float(fmx),
                        qy=float(fmy),
                        prev_cli=int(a_cli),
                        prev_xy=None,
                        mapped_reversed=None,
                    )
                    pb = _best_projection_on_line(
                        fi=fi,
                        cidx=int(b_cli),
                        qx=float(fmx),
                        qy=float(fmy),
                        prev_cli=int(b_cli),
                        prev_xy=None,
                        mapped_reversed=None,
                    )
                    if pa is None or pb is None:
                        valid = False
                        break
                    proj_a.append(dict(pa.get("proj", {})))
                    proj_b.append(dict(pb.get("proj", {})))
                if not valid or len(proj_a) != (we - ws + 1):
                    continue

                m = len(proj_a)
                for k, fi in enumerate(range(ws, we + 1)):
                    if fi >= len(raw_mappings) or not raw_mappings[fi]:
                        continue
                    t01 = float(k) / float(max(1, m - 1))
                    w = _smoothstep01(t01)
                    pa = proj_a[k]
                    pb = proj_b[k]
                    ax, ay = float(pa.get("x", 0.0)), float(pa.get("y", 0.0))
                    bx, by = float(pb.get("x", 0.0)), float(pb.get("y", 0.0))
                    syx = ax + (bx - ax) * w
                    syy = ay + (by - ay) * w
                    ayaw = float(pa.get("yaw", 0.0))
                    byaw = float(pb.get("yaw", ayaw))
                    syaw = _normalize_yaw_deg(ayaw + _normalize_yaw_deg(byaw - ayaw) * w)
                    mh = _frame_motion_yaw(fi)
                    if mh is not None and _yaw_abs_diff_deg(float(mh), float(syaw)) > float(_DIR_OPPOSITE_REJECT_DEG):
                        syaw = float(ayaw if _yaw_abs_diff_deg(float(mh), float(ayaw)) <= _yaw_abs_diff_deg(float(mh), float(byaw)) else byaw)
                    raw_mappings[fi]["cx"] = float(syx)
                    raw_mappings[fi]["cy"] = float(syy)
                    raw_mappings[fi]["cyaw"] = float(syaw)
                    raw_mappings[fi]["cli"] = int(a_cli if w < 0.5 else b_cli)
                    raw_mappings[fi]["cld"] = float((1.0 - w) * float(pa.get("dist", 0.0)) + w * float(pb.get("dist", 0.0)))
                    raw_mappings[fi]["csource"] = "lanechange_scurve"
                    raw_mappings[fi]["cquality"] = "scurve"

                lanechange_scurve_events += 1
                last_used_end = we

        # =====================================================================
        # Pass 8: Final anti-jump continuity clamp
        # =====================================================================
        if enforce_direction and len(raw_mappings) >= 2:
            _HARD_JUMP_M = 8.0
            for fi in range(1, len(raw_mappings)):
                prev_rm = raw_mappings[fi - 1]
                cur_rm = raw_mappings[fi]
                if not prev_rm or not cur_rm:
                    continue
                px, py = float(prev_rm.get("cx", 0.0)), float(prev_rm.get("cy", 0.0))
                cxv, cyv = float(cur_rm.get("cx", px)), float(cur_rm.get("cy", py))
                dpc = math.hypot(float(cxv - px), float(cyv - py))
                if dpc <= float(_HARD_JUMP_M):
                    continue
                continuity_jump_repairs += 1
                if fi + 1 < len(raw_mappings) and raw_mappings[fi + 1]:
                    nx = float(raw_mappings[fi + 1].get("cx", cxv))
                    ny = float(raw_mappings[fi + 1].get("cy", cyv))
                    if math.hypot(float(nx - px), float(ny - py)) <= 2.0 * float(_HARD_JUMP_M):
                        tx = 0.5 * (px + nx)
                        ty = 0.5 * (py + ny)
                    else:
                        r = float(_HARD_JUMP_M) / max(1e-6, dpc)
                        tx = px + (cxv - px) * r
                        ty = py + (cyv - py) * r
                else:
                    r = float(_HARD_JUMP_M) / max(1e-6, dpc)
                    tx = px + (cxv - px) * r
                    ty = py + (cyv - py) * r
                tyaw = _normalize_yaw_deg(math.degrees(math.atan2(float(ty - py), float(tx - px)))) if math.hypot(float(tx - px), float(ty - py)) > 1e-6 else float(cur_rm.get("cyaw", 0.0))
                prev_cli_j = int(prev_rm.get("cli", -1))
                cand_jump = _best_projection_any_line(
                    fi=fi,
                    qx=float(tx),
                    qy=float(ty),
                    prev_cli=int(prev_cli_j),
                    prev_xy=(float(px), float(py)),
                )
                cur_rm["cx"] = float(tx)
                cur_rm["cy"] = float(ty)
                cur_rm["cyaw"] = float(tyaw)
                if cand_jump is not None and float(cand_jump["proj"].get("dist", float("inf"))) <= 8.0:
                    cur_rm["cli"] = int(cand_jump.get("ci", -1))
                    cur_rm["cld"] = float(cand_jump["proj"].get("dist", 0.0))
                else:
                    cur_rm["cli"] = int(prev_cli_j)
                    cur_rm["cld"] = float(cur_rm.get("cld", 0.0))
                cur_rm["csource"] = "jump_repair"
                cur_rm["cquality"] = "continuity"

        # =====================================================================
        # Pass 8.25: Fill unknown cli gaps from the nearest legal predecessor
        # =====================================================================
        if enforce_direction and len(raw_mappings) >= 2:
            for fi in range(1, len(raw_mappings)):
                cur_rm = raw_mappings[fi]
                if not cur_rm:
                    continue
                cur_cli = int(cur_rm.get("cli", -1))
                if cur_cli >= 0:
                    continue
                if fi >= len(frames) or not isinstance(frames[fi], dict):
                    continue

                prev_idx = fi - 1
                while prev_idx >= 0:
                    prm = raw_mappings[prev_idx]
                    if prm and int(prm.get("cli", -1)) >= 0:
                        break
                    prev_idx -= 1
                if prev_idx < 0 or not raw_mappings[prev_idx]:
                    continue

                prev_rm = raw_mappings[prev_idx]
                prev_cli = int(prev_rm.get("cli", -1))
                if prev_cli < 0:
                    continue

                fmx = _safe_float(frames[fi].get("mx"), _safe_float(frames[fi].get("x"), 0.0))
                fmy = _safe_float(frames[fi].get("my"), _safe_float(frames[fi].get("y"), 0.0))
                fmyaw = _safe_float(frames[fi].get("myaw"), _safe_float(frames[fi].get("yaw"), 0.0))
                cand_gap = _nearest_projection_on_line(
                    cidx=int(prev_cli),
                    qx=float(fmx),
                    qy=float(fmy),
                    mapped_reversed=None,
                )
                if cand_gap is None:
                    continue
                proj_gap = dict(cand_gap.get("proj", {}))
                gap_dist = float(proj_gap.get("dist", float("inf")))
                if not math.isfinite(gap_dist) or gap_dist > 8.0:
                    continue
                cur_rm["cli"] = int(prev_cli)
                cur_rm["cx"] = float(proj_gap.get("x", fmx))
                cur_rm["cy"] = float(proj_gap.get("y", fmy))
                cur_rm["cyaw"] = float(fmyaw)
                cur_rm["cld"] = float(gap_dist)
                cur_rm["csource"] = "legal_gap_fill_prev"
                cur_rm["cquality"] = "legal"

        # =====================================================================
        # Pass 8.4: Divergence-based re-anchor
        # =====================================================================
        # If a track stays on one cli while the raw point drifts far away from
        # that centerline, allow recovery to this frame's mapped lane hint.
        if enforce_direction and len(raw_mappings) >= 2:
            _DIVERGE_TRIGGER_DIST_M = 3.0
            _DIVERGE_MIN_IMPROVE_M = 1.2
            _DIVERGE_POOR_HINT_TRIGGER_M = 7.0
            _DIVERGE_MAX_DIST_M = 8.0
            for fi in range(1, len(raw_mappings)):
                rm = raw_mappings[fi]
                if not rm:
                    continue
                cur_cli = int(rm.get("cli", -1))
                cur_cld = _safe_float(rm.get("cld"), float("inf"))
                if cur_cli < 0:
                    continue
                if fi >= len(frames) or not isinstance(frames[fi], dict):
                    continue

                fmx = _safe_float(frames[fi].get("mx"), _safe_float(frames[fi].get("x"), 0.0))
                fmy = _safe_float(frames[fi].get("my"), _safe_float(frames[fi].get("y"), 0.0))
                fmyaw = _safe_float(frames[fi].get("myaw"), _safe_float(frames[fi].get("yaw"), 0.0))

                # Use live distance-to-current-line (from raw point) instead of
                # stale cld, which may be inherited from earlier passes.
                cur_hold = _nearest_projection_on_line(
                    cidx=int(cur_cli),
                    qx=float(fmx),
                    qy=float(fmy),
                    mapped_reversed=None,
                    force_bidir=True,
                )
                cur_cld_eval = float(cur_cld)
                if cur_hold is not None:
                    cur_hold_dist = float(cur_hold.get("proj", {}).get("dist", float("inf")))
                    if math.isfinite(cur_hold_dist):
                        cur_cld_eval = float(cur_hold_dist)
                if not math.isfinite(cur_cld_eval) or cur_cld_eval < float(_DIVERGE_TRIGGER_DIST_M):
                    continue

                li_now = _safe_int(frames[fi].get("li"), -1)
                mapped_now = lane_to_carla.get(int(li_now))
                if not isinstance(mapped_now, dict):
                    continue
                map_ci = _safe_int(mapped_now.get("carla_line_index"), -1)
                map_q = str(mapped_now.get("quality", "poor")).strip().lower()
                map_rev = bool(mapped_now.get("reversed", False))
                if map_ci < 0 or int(map_ci) == int(cur_cli):
                    continue
                if map_q == "poor" and float(cur_cld_eval) < float(_DIVERGE_POOR_HINT_TRIGGER_M):
                    continue

                prev_rm = raw_mappings[fi - 1] if fi > 0 else None
                prev_cli = int(prev_rm.get("cli", -1)) if prev_rm else -1

                prev_xy = (
                    float(prev_rm.get("cx", fmx)),
                    float(prev_rm.get("cy", fmy)),
                ) if prev_rm else None

                cand = _best_projection_on_line(
                    fi=fi,
                    cidx=int(map_ci),
                    qx=float(fmx),
                    qy=float(fmy),
                    prev_cli=int(prev_cli if prev_cli >= 0 else cur_cli),
                    prev_xy=prev_xy,
                    mapped_reversed=bool(map_rev),
                    force_bidir=True,
                )
                if cand is None:
                    continue
                proj = dict(cand.get("proj", {}))
                p_dist = float(proj.get("dist", float("inf")))
                p_yaw = _safe_float(proj.get("yaw"), float("nan"))
                if not math.isfinite(p_dist) or p_dist > float(_DIVERGE_MAX_DIST_M):
                    continue
                if (p_dist + float(_DIVERGE_MIN_IMPROVE_M)) >= float(cur_cld_eval):
                    continue

                frame_mhy = _frame_motion_yaw(fi)
                if frame_mhy is not None and math.isfinite(p_yaw):
                    if _yaw_abs_diff_deg(float(frame_mhy), float(p_yaw)) > float(_DIR_OPPOSITE_REJECT_DEG):
                        continue

                legal_like = False
                if prev_cli >= 0:
                    legal_like = bool(
                        _transition_has_successor_connect(int(prev_cli), int(map_ci))
                        or _transition_has_successor_connect(int(cur_cli), int(map_ci))
                    )
                if (
                    not legal_like
                    and map_q != "poor"
                    and (
                        (float(cur_cld_eval) >= 3.0 and (p_dist + 0.5) < float(cur_cld_eval))
                        or float(cur_cld_eval) >= float(_DIVERGE_POOR_HINT_TRIGGER_M)
                    )
                ):
                    legal_like = True
                if not legal_like:
                    continue

                rm["cli"] = int(map_ci)
                rm["cx"] = float(proj.get("x", fmx))
                rm["cy"] = float(proj.get("y", fmy))
                rm["cyaw"] = float(proj.get("yaw", fmyaw))
                rm["cld"] = float(p_dist)
                rm["csource"] = "divergence_reanchor"
                rm["cquality"] = "legal"

        # =====================================================================
        # Pass 8.5: Strict legal transition clamp (successor chain only,
        # with explicit lane-change exception on lateral-adjacent lines)
        # =====================================================================
        if enforce_direction and len(raw_mappings) >= 2:
            _LEGAL_FIX_MAX_DIST_M = 8.0
            for fi in range(1, len(raw_mappings)):
                prev_rm = raw_mappings[fi - 1]
                cur_rm = raw_mappings[fi]
                if not prev_rm or not cur_rm:
                    continue

                prev_cli = int(prev_rm.get("cli", -1))
                cur_cli = int(cur_rm.get("cli", -1))
                if prev_cli >= 0 and cur_cli < 0:
                    illegal_lane_transition_detected += 1
                    if fi < len(frames) and isinstance(frames[fi], dict):
                        fmx = _safe_float(frames[fi].get("mx"), _safe_float(frames[fi].get("x"), 0.0))
                        fmy = _safe_float(frames[fi].get("my"), _safe_float(frames[fi].get("y"), 0.0))
                        fmyaw = _safe_float(frames[fi].get("myaw"), _safe_float(frames[fi].get("yaw"), 0.0))
                        hold = _nearest_projection_on_line(
                            cidx=int(prev_cli),
                            qx=float(fmx),
                            qy=float(fmy),
                            mapped_reversed=None,
                        )
                        if hold is not None and float(hold["proj"].get("dist", float("inf"))) <= 12.0:
                            hproj = dict(hold.get("proj", {}))
                            cur_rm["cx"] = float(hproj.get("x", fmx))
                            cur_rm["cy"] = float(hproj.get("y", fmy))
                            cur_rm["cyaw"] = float(fmyaw)
                            cur_rm["cld"] = float(hproj.get("dist", 0.0))
                        else:
                            cur_rm["cx"] = float(prev_rm.get("cx", fmx))
                            cur_rm["cy"] = float(prev_rm.get("cy", fmy))
                            cur_rm["cyaw"] = float(prev_rm.get("cyaw", fmyaw))
                            cur_rm["cld"] = float(cur_rm.get("cld", 0.0))
                    else:
                        cur_rm["cx"] = float(prev_rm.get("cx", cur_rm.get("cx", 0.0)))
                        cur_rm["cy"] = float(prev_rm.get("cy", cur_rm.get("cy", 0.0)))
                        cur_rm["cyaw"] = float(prev_rm.get("cyaw", cur_rm.get("cyaw", 0.0)))
                    cur_rm["cli"] = int(prev_cli)
                    cur_rm["csource"] = "legal_hold_missing"
                    cur_rm["cquality"] = "legal"
                    illegal_lane_transition_fixed += 1
                    continue

                if prev_cli < 0 or cur_cli < 0:
                    continue

                if prev_cli == cur_cli:
                    if fi >= len(frames) or not isinstance(frames[fi], dict):
                        continue

                    li_same = _safe_int(frames[fi].get("li"), -1)
                    mapped_same = lane_to_carla.get(int(li_same))
                    if not isinstance(mapped_same, dict):
                        continue
                    map_ci_same = _safe_int(mapped_same.get("carla_line_index"), -1)
                    map_q_same = str(mapped_same.get("quality", "poor")).strip().lower()
                    map_rev_same = bool(mapped_same.get("reversed", False))
                    if map_ci_same < 0 or int(map_ci_same) == int(cur_cli):
                        continue
                    fmx_same = _safe_float(frames[fi].get("mx"), _safe_float(frames[fi].get("x"), 0.0))
                    fmy_same = _safe_float(frames[fi].get("my"), _safe_float(frames[fi].get("y"), 0.0))
                    fmyaw_same = _safe_float(frames[fi].get("myaw"), _safe_float(frames[fi].get("yaw"), 0.0))
                    cur_cld_same = _safe_float(cur_rm.get("cld"), float("inf"))
                    same_hold = _nearest_projection_on_line(
                        cidx=int(cur_cli),
                        qx=float(fmx_same),
                        qy=float(fmy_same),
                        mapped_reversed=None,
                        force_bidir=True,
                    )
                    if same_hold is not None:
                        same_hold_dist = float(same_hold.get("proj", {}).get("dist", float("inf")))
                        if math.isfinite(same_hold_dist):
                            cur_cld_same = float(same_hold_dist)
                    if not math.isfinite(cur_cld_same):
                        continue
                    same_trigger = 7.0 if map_q_same == "poor" else 3.0
                    if float(cur_cld_same) < float(same_trigger):
                        continue

                    prev_xy_same = (
                        float(prev_rm.get("cx", fmx_same)),
                        float(prev_rm.get("cy", fmy_same)),
                    )
                    cand_same = _best_projection_on_line(
                        fi=fi,
                        cidx=int(map_ci_same),
                        qx=float(fmx_same),
                        qy=float(fmy_same),
                        prev_cli=int(prev_cli),
                        prev_xy=prev_xy_same,
                        mapped_reversed=bool(map_rev_same),
                        force_bidir=True,
                    )
                    if cand_same is None:
                        continue
                    proj_same = dict(cand_same.get("proj", {}))
                    p_dist_same = float(proj_same.get("dist", float("inf")))
                    p_yaw_same = _safe_float(proj_same.get("yaw"), float("nan"))
                    if not math.isfinite(p_dist_same) or p_dist_same > 8.0:
                        continue
                    if (p_dist_same + 1.2) >= float(cur_cld_same):
                        continue
                    mhy_same = _frame_motion_yaw(fi)
                    if mhy_same is not None and math.isfinite(p_yaw_same):
                        if _yaw_abs_diff_deg(float(mhy_same), float(p_yaw_same)) > float(_DIR_OPPOSITE_REJECT_DEG):
                            continue

                    same_legal_like = bool(_transition_has_successor_connect(int(prev_cli), int(map_ci_same)))
                    if (
                        not same_legal_like
                        and map_q_same != "poor"
                        and (
                            (float(cur_cld_same) >= 3.0 and (p_dist_same + 0.5) < float(cur_cld_same))
                            or float(cur_cld_same) >= 7.0
                        )
                    ):
                        same_legal_like = True
                    if not same_legal_like:
                        continue

                    cur_rm["cli"] = int(map_ci_same)
                    cur_rm["cx"] = float(proj_same.get("x", fmx_same))
                    cur_rm["cy"] = float(proj_same.get("y", fmy_same))
                    cur_rm["cyaw"] = float(proj_same.get("yaw", fmyaw_same))
                    cur_rm["cld"] = float(p_dist_same)
                    cur_rm["csource"] = "divergence_reanchor"
                    cur_rm["cquality"] = "legal"
                    continue

                src_now = str(cur_rm.get("csource", "")).strip().lower()
                lanechange_like_now = _transition_is_lanechange_like(
                    fi=fi,
                    prev_rm=prev_rm,
                    cur_rm=cur_rm,
                    prev_cli=int(prev_cli),
                    cur_cli=int(cur_cli),
                )
                allow_lanechange_exception = bool(src_now in _LANECHANGE_SOURCES or lanechange_like_now)

                legal_transition = _transition_has_successor_connect(int(prev_cli), int(cur_cli))
                if not legal_transition and allow_lanechange_exception and lanechange_like_now:
                    lanechange_ok = True
                    if fi < len(frames) and isinstance(frames[fi], dict):
                        fmx_lc = _safe_float(frames[fi].get("mx"), _safe_float(frames[fi].get("x"), 0.0))
                        fmy_lc = _safe_float(frames[fi].get("my"), _safe_float(frames[fi].get("y"), 0.0))
                        hold_prev_lc = _nearest_projection_on_line(
                            cidx=int(prev_cli),
                            qx=float(fmx_lc),
                            qy=float(fmy_lc),
                            mapped_reversed=None,
                            force_bidir=True,
                        )
                        hold_cur_lc = _nearest_projection_on_line(
                            cidx=int(cur_cli),
                            qx=float(fmx_lc),
                            qy=float(fmy_lc),
                            mapped_reversed=None,
                            force_bidir=True,
                        )
                        d_prev_lc = float("inf")
                        d_cur_lc = float("inf")
                        if hold_prev_lc is not None:
                            d_prev_lc = float(hold_prev_lc.get("proj", {}).get("dist", float("inf")))
                        if hold_cur_lc is not None:
                            d_cur_lc = float(hold_cur_lc.get("proj", {}).get("dist", float("inf")))
                        if math.isfinite(d_prev_lc) and math.isfinite(d_cur_lc):
                            if d_cur_lc > (d_prev_lc + 0.35):
                                lanechange_ok = False
                    if lanechange_ok:
                        legal_transition = True
                if legal_transition:
                    continue

                illegal_lane_transition_detected += 1

                if fi >= len(frames) or not isinstance(frames[fi], dict):
                    cur_rm["cli"] = int(prev_cli)
                    cur_rm["cx"] = float(prev_rm.get("cx", cur_rm.get("cx", 0.0)))
                    cur_rm["cy"] = float(prev_rm.get("cy", cur_rm.get("cy", 0.0)))
                    cur_rm["cyaw"] = float(prev_rm.get("cyaw", cur_rm.get("cyaw", 0.0)))
                    cur_rm["csource"] = "legal_hold"
                    cur_rm["cquality"] = "legal"
                    illegal_lane_transition_fixed += 1
                    continue

                fmx = _safe_float(frames[fi].get("mx"), _safe_float(frames[fi].get("x"), 0.0))
                fmy = _safe_float(frames[fi].get("my"), _safe_float(frames[fi].get("y"), 0.0))
                fmyaw = _safe_float(frames[fi].get("myaw"), _safe_float(frames[fi].get("yaw"), 0.0))
                prev_xy_legal = (
                    float(prev_rm.get("cx", fmx)),
                    float(prev_rm.get("cy", fmy)),
                )
                frame_mhy = _frame_motion_yaw(fi)
                hold_probe = _nearest_projection_on_line(
                    cidx=int(prev_cli),
                    qx=float(fmx),
                    qy=float(fmy),
                    mapped_reversed=None,
                )
                hold_probe_dist = float("inf")
                if hold_probe is not None:
                    hold_probe_dist = float(hold_probe["proj"].get("dist", float("inf")))
                _LEGAL_REANCHOR_TRIGGER_M = 2.5
                _LEGAL_POOR_HINT_TRIGGER_M = 5.5
                allow_map_reanchor = bool(
                    math.isfinite(hold_probe_dist) and hold_probe_dist >= float(_LEGAL_REANCHOR_TRIGGER_M)
                )

                mapped_hint_cis: List[int] = []
                mapped_hint_quality: Dict[int, str] = {}
                mapped_hint_reversed: Dict[int, bool] = {}
                li_now = _safe_int(frames[fi].get("li"), -1)
                mapped_now = lane_to_carla.get(int(li_now))
                if isinstance(mapped_now, dict):
                    mapped_ci = _safe_int(mapped_now.get("carla_line_index"), -1)
                    map_q = str(mapped_now.get("quality", "poor")).strip().lower()
                    map_rev = bool(mapped_now.get("reversed", False))
                    if mapped_ci >= 0:
                        mapped_hint_cis.append(int(mapped_ci))
                        mapped_hint_quality[int(mapped_ci)] = map_q
                        mapped_hint_reversed[int(mapped_ci)] = bool(map_rev)
                    extra_cis = mapped_now.get("split_extra_carla_lines")
                    if isinstance(extra_cis, (list, tuple)):
                        for ex in extra_cis:
                            exi = _safe_int(ex, -1)
                            if exi >= 0:
                                mapped_hint_cis.append(int(exi))
                                mapped_hint_quality[int(exi)] = map_q
                                mapped_hint_reversed[int(exi)] = bool(map_rev)
                mapped_hint_set = set(int(v) for v in mapped_hint_cis if int(v) >= 0)

                candidate_lines: List[int] = [int(prev_cli)]
                for ci in sorted(carla_succs.get(int(prev_cli), set())):
                    candidate_lines.append(int(ci))
                for ci in sorted(carla_preds.get(int(prev_cli), set())):
                    candidate_lines.append(int(ci))
                if allow_lanechange_exception:
                    for ci in sorted(carla_lat_adj.get(int(prev_cli), set())):
                        candidate_lines.append(int(ci))
                for ci in mapped_hint_cis:
                    candidate_lines.append(int(ci))

                seen_cis: set[int] = set()
                best_fix: Optional[Dict[str, object]] = None
                for ci in candidate_lines:
                    if int(ci) < 0 or int(ci) in seen_cis:
                        continue
                    seen_cis.add(int(ci))

                    is_successor_like = _transition_has_successor_connect(int(prev_cli), int(ci))
                    is_lateral_like = bool(
                        int(ci) in carla_lat_adj.get(int(prev_cli), set())
                        or int(prev_cli) in carla_lat_adj.get(int(ci), set())
                    )
                    is_map_hint = bool(int(ci) in mapped_hint_set)
                    map_q = str(mapped_hint_quality.get(int(ci), "poor")).strip().lower()
                    map_hint_allowed = bool(
                        map_q != "poor"
                        or (
                            math.isfinite(hold_probe_dist)
                            and hold_probe_dist >= float(_LEGAL_POOR_HINT_TRIGGER_M)
                        )
                    )
                    if int(ci) != int(prev_cli):
                        if is_successor_like:
                            pass
                        elif allow_lanechange_exception and is_lateral_like:
                            pass
                        elif allow_map_reanchor and is_map_hint and map_hint_allowed:
                            pass
                        else:
                            continue

                    cand = _best_projection_on_line(
                        fi=fi,
                        cidx=int(ci),
                        qx=float(fmx),
                        qy=float(fmy),
                        prev_cli=int(prev_cli),
                        prev_xy=prev_xy_legal,
                        mapped_reversed=mapped_hint_reversed.get(int(ci), None),
                        force_bidir=bool(int(ci) in mapped_hint_set),
                    )
                    if cand is None:
                        continue
                    proj = dict(cand.get("proj", {}))
                    p_dist = float(proj.get("dist", float("inf")))
                    p_yaw = _safe_float(proj.get("yaw"), float("nan"))
                    if not math.isfinite(p_dist) or p_dist > float(_LEGAL_FIX_MAX_DIST_M):
                        continue
                    if frame_mhy is not None and math.isfinite(p_yaw):
                        if _yaw_abs_diff_deg(float(frame_mhy), float(p_yaw)) > float(_DIR_OPPOSITE_REJECT_DEG):
                            continue

                    fix_score = float(cand.get("score", p_dist))
                    if int(ci) == int(prev_cli):
                        fix_score -= 0.30
                    elif is_successor_like:
                        fix_score -= 0.10
                    elif is_lateral_like:
                        fix_score += 0.10
                    elif is_map_hint:
                        # Only used when previous lane no longer explains the raw
                        # trajectory; keep slight penalty but allow recovery.
                        fix_score += 0.15
                        if (
                            allow_map_reanchor
                            and math.isfinite(hold_probe_dist)
                            and (hold_probe_dist - p_dist) >= 1.5
                        ):
                            fix_score -= 0.25

                    cand_fix = {
                        "ci": int(ci),
                        "proj": proj,
                        "score": float(fix_score),
                    }
                    if best_fix is None or float(cand_fix["score"]) < float(best_fix["score"]):
                        best_fix = cand_fix

                if best_fix is not None:
                    proj = dict(best_fix.get("proj", {}))
                    best_ci = int(best_fix.get("ci", int(prev_cli)))
                    cur_rm["cli"] = int(best_ci)
                    cur_rm["cx"] = float(proj.get("x", fmx))
                    cur_rm["cy"] = float(proj.get("y", fmy))
                    cur_rm["cyaw"] = float(proj.get("yaw", fmyaw))
                    cur_rm["cld"] = float(proj.get("dist", 0.0))
                    cur_rm["csource"] = "legal_clamp" if best_ci == int(prev_cli) else "legal_redirect"
                    cur_rm["cquality"] = "legal"
                    illegal_lane_transition_fixed += 1
                    continue

                hold_proj = hold_probe
                hold_dist = float(hold_proj["proj"].get("dist", float("inf"))) if hold_proj is not None else float("inf")
                if math.isfinite(hold_dist) and hold_dist > 6.0:
                    rec_any = _best_projection_any_line(
                        fi=fi,
                        qx=float(fmx),
                        qy=float(fmy),
                        prev_cli=int(prev_cli),
                        prev_xy=prev_xy_legal,
                    )
                    if rec_any is None:
                        rec_any = _nearest_projection_any_line(
                            qx=float(fmx),
                            qy=float(fmy),
                        )
                    if rec_any is not None:
                        rec_proj = dict(rec_any.get("proj", {}))
                        rec_dist = float(rec_proj.get("dist", float("inf")))
                        rec_yaw = _safe_float(rec_proj.get("yaw"), float("nan"))
                        rec_ci = int(rec_any.get("ci", -1))
                        rec_ok = bool(math.isfinite(rec_dist) and rec_dist <= 12.0)
                        if rec_ok and frame_mhy is not None and math.isfinite(rec_yaw):
                            rec_ok = bool(
                                _yaw_abs_diff_deg(float(frame_mhy), float(rec_yaw))
                                <= float(_DIR_OPPOSITE_REJECT_DEG)
                            )
                        if rec_ok and (int(rec_ci) != int(prev_cli) or (rec_dist + 0.5) < hold_dist):
                            cur_rm["cli"] = int(rec_ci)
                            cur_rm["cx"] = float(rec_proj.get("x", fmx))
                            cur_rm["cy"] = float(rec_proj.get("y", fmy))
                            cur_rm["cyaw"] = float(rec_proj.get("yaw", fmyaw))
                            cur_rm["cld"] = float(rec_dist)
                            cur_rm["csource"] = "legal_recover_any"
                            cur_rm["cquality"] = "recover"
                            illegal_lane_transition_fixed += 1
                            continue

                if hold_proj is not None and float(hold_proj["proj"].get("dist", float("inf"))) <= 12.0:
                    hproj = dict(hold_proj.get("proj", {}))
                    cur_rm["cli"] = int(prev_cli)
                    cur_rm["cx"] = float(hproj.get("x", fmx))
                    cur_rm["cy"] = float(hproj.get("y", fmy))
                    cur_rm["cyaw"] = float(hproj.get("yaw", fmyaw))
                    cur_rm["cld"] = float(hproj.get("dist", 0.0))
                    cur_rm["csource"] = "legal_hold_proj"
                    cur_rm["cquality"] = "legal"
                else:
                    cur_rm["cli"] = int(prev_cli)
                    cur_rm["cx"] = float(prev_rm.get("cx", fmx))
                    cur_rm["cy"] = float(prev_rm.get("cy", fmy))
                    cur_rm["cyaw"] = float(prev_rm.get("cyaw", fmyaw))
                    cur_rm["cld"] = float(cur_rm.get("cld", 0.0))
                    cur_rm["csource"] = "legal_hold"
                    cur_rm["cquality"] = "legal"
                illegal_lane_transition_fixed += 1

        # =====================================================================
        # Pass 8.75: Anti-freeze reproject
        # =====================================================================
        # When cx/cy is frozen but raw trajectory is still moving, reproject
        # current raw point so replay stays temporally faithful.
        if enforce_direction and len(raw_mappings) >= 2:
            _ANTI_FREEZE_EPS_M = 0.05
            _ANTI_FREEZE_RAW_STEP_M = 0.5
            for fi in range(1, len(raw_mappings)):
                prev_rm = raw_mappings[fi - 1]
                cur_rm = raw_mappings[fi]
                if not prev_rm or not cur_rm:
                    continue
                prev_cli = int(prev_rm.get("cli", -1))
                cur_cli = int(cur_rm.get("cli", -1))
                if prev_cli < 0 or cur_cli < 0 or prev_cli != cur_cli:
                    continue
                if fi >= len(frames) or not isinstance(frames[fi], dict):
                    continue
                if not isinstance(frames[fi - 1], dict):
                    continue

                prev_cx = _safe_float(prev_rm.get("cx"), float("nan"))
                prev_cy = _safe_float(prev_rm.get("cy"), float("nan"))
                cur_cx = _safe_float(cur_rm.get("cx"), float("nan"))
                cur_cy = _safe_float(cur_rm.get("cy"), float("nan"))
                if not (math.isfinite(prev_cx) and math.isfinite(prev_cy) and math.isfinite(cur_cx) and math.isfinite(cur_cy)):
                    continue
                car_step = math.hypot(float(cur_cx - prev_cx), float(cur_cy - prev_cy))
                if car_step > float(_ANTI_FREEZE_EPS_M):
                    continue

                pmx = _safe_float(frames[fi - 1].get("mx"), _safe_float(frames[fi - 1].get("x"), 0.0))
                pmy = _safe_float(frames[fi - 1].get("my"), _safe_float(frames[fi - 1].get("y"), 0.0))
                fmx = _safe_float(frames[fi].get("mx"), _safe_float(frames[fi].get("x"), 0.0))
                fmy = _safe_float(frames[fi].get("my"), _safe_float(frames[fi].get("y"), 0.0))
                fmyaw = _safe_float(frames[fi].get("myaw"), _safe_float(frames[fi].get("yaw"), 0.0))
                raw_step = math.hypot(float(fmx - pmx), float(fmy - pmy))
                if raw_step < float(_ANTI_FREEZE_RAW_STEP_M):
                    continue

                frame_mhy = _frame_motion_yaw(fi)
                moved = False

                reproj_same = _nearest_projection_on_line(
                    cidx=int(cur_cli),
                    qx=float(fmx),
                    qy=float(fmy),
                    mapped_reversed=None,
                )
                if reproj_same is not None:
                    rproj = dict(reproj_same.get("proj", {}))
                    rdist = float(rproj.get("dist", float("inf")))
                    ryaw = _safe_float(rproj.get("yaw"), float("nan"))
                    rstep = math.hypot(float(rproj.get("x", prev_cx)) - float(prev_cx), float(rproj.get("y", prev_cy)) - float(prev_cy))
                    ryaw_ok = True
                    if frame_mhy is not None and math.isfinite(ryaw):
                        ryaw_ok = bool(
                            _yaw_abs_diff_deg(float(frame_mhy), float(ryaw)) <= float(_DIR_OPPOSITE_REJECT_DEG)
                        )
                    if math.isfinite(rdist) and rdist <= 12.0 and rstep > float(_ANTI_FREEZE_EPS_M) and ryaw_ok:
                        cur_rm["cx"] = float(rproj.get("x", fmx))
                        cur_rm["cy"] = float(rproj.get("y", fmy))
                        cur_rm["cyaw"] = float(rproj.get("yaw", fmyaw))
                        cur_rm["cld"] = float(rdist)
                        cur_rm["csource"] = "anti_freeze"
                        cur_rm["cquality"] = "legal"
                        moved = True

                if moved:
                    continue

                li_now = _safe_int(frames[fi].get("li"), -1)
                mapped_now = lane_to_carla.get(int(li_now))
                if not isinstance(mapped_now, dict):
                    continue
                map_q = str(mapped_now.get("quality", "poor")).strip().lower()
                if map_q == "poor":
                    continue
                map_ci = _safe_int(mapped_now.get("carla_line_index"), -1)
                if map_ci < 0:
                    continue
                cand_map = _best_projection_on_line(
                    fi=fi,
                    cidx=int(map_ci),
                    qx=float(fmx),
                    qy=float(fmy),
                    prev_cli=int(prev_cli),
                    prev_xy=(float(prev_cx), float(prev_cy)),
                    mapped_reversed=bool(mapped_now.get("reversed", False)),
                    force_bidir=True,
                )
                if cand_map is None:
                    continue
                mproj = dict(cand_map.get("proj", {}))
                mdist = float(mproj.get("dist", float("inf")))
                myaw = _safe_float(mproj.get("yaw"), float("nan"))
                myaw_ok = True
                if frame_mhy is not None and math.isfinite(myaw):
                    myaw_ok = bool(
                        _yaw_abs_diff_deg(float(frame_mhy), float(myaw)) <= float(_DIR_OPPOSITE_REJECT_DEG)
                    )
                if not (math.isfinite(mdist) and mdist <= 8.0 and myaw_ok):
                    continue
                cur_rm["cli"] = int(map_ci)
                cur_rm["cx"] = float(mproj.get("x", fmx))
                cur_rm["cy"] = float(mproj.get("y", fmy))
                cur_rm["cyaw"] = float(mproj.get("yaw", fmyaw))
                cur_rm["cld"] = float(mdist)
                cur_rm["csource"] = "anti_freeze_reanchor"
                cur_rm["cquality"] = "legal"

        # =====================================================================
        # Pass 8.9: Fidelity-first continuity snap (alternative strategy)
        # =====================================================================
        # Build an alternate mapping that prioritizes:
        #  - closeness to raw trajectory
        #  - continuity (avoid abrupt segment hopping)
        #  - legal directionality (reject opposite-flow matches)
        # Then select whichever mapping has better objective.
        if enforce_direction and len(raw_mappings) >= 2:
            def _map_hints_for_frame(fi: int) -> Dict[int, Dict[str, object]]:
                out: Dict[int, Dict[str, object]] = {}
                if fi < 0 or fi >= len(frames) or not isinstance(frames[fi], dict):
                    return out
                li_now = _safe_int(frames[fi].get("li"), -1)
                mapped_now = lane_to_carla.get(int(li_now))
                if not isinstance(mapped_now, dict):
                    return out
                q = str(mapped_now.get("quality", "poor")).strip().lower()
                rev = bool(mapped_now.get("reversed", False))
                ci = _safe_int(mapped_now.get("carla_line_index"), -1)
                if ci >= 0:
                    out[int(ci)] = {"quality": q, "reversed": rev}
                extras = mapped_now.get("split_extra_carla_lines")
                if isinstance(extras, (list, tuple)):
                    for ex in extras:
                        exi = _safe_int(ex, -1)
                        if exi >= 0:
                            out[int(exi)] = {"quality": q, "reversed": rev}
                return out

            def _mapping_stats(maps: List[Dict[str, object]]) -> Dict[str, float]:
                if not maps:
                    return {
                        "objective": float("inf"),
                        "mean_dist": float("inf"),
                        "illegal_jumps": float("inf"),
                        "moving_stall": float("inf"),
                        "switches": float("inf"),
                    }
                dist_acc = 0.0
                dist_n = 0
                illegal_jumps = 0
                moving_stall = 0
                switches = 0
                for fi in range(len(maps)):
                    rm = maps[fi]
                    if not rm:
                        continue
                    cld = _safe_float(rm.get("cld"), float("inf"))
                    if math.isfinite(cld):
                        dist_acc += max(0.0, min(20.0, float(cld)))
                        dist_n += 1
                    if fi <= 0:
                        continue
                    prm = maps[fi - 1]
                    if not prm:
                        continue
                    prev_cli = int(prm.get("cli", -1))
                    cur_cli = int(rm.get("cli", -1))
                    if prev_cli >= 0 and cur_cli >= 0 and prev_cli != cur_cli:
                        switches += 1
                        lanechange_like_eval = _transition_is_lanechange_like(
                            fi=fi,
                            prev_rm=prm,
                            cur_rm=rm,
                            prev_cli=int(prev_cli),
                            cur_cli=int(cur_cli),
                        )
                        legal_eval = bool(
                            _transition_has_successor_connect(int(prev_cli), int(cur_cli))
                            or lanechange_like_eval
                        )
                        if not legal_eval:
                            illegal_jumps += 1

                    if fi >= len(frames):
                        continue
                    if not isinstance(frames[fi], dict) or not isinstance(frames[fi - 1], dict):
                        continue
                    car_step = math.hypot(
                        _safe_float(rm.get("cx"), 0.0) - _safe_float(prm.get("cx"), 0.0),
                        _safe_float(rm.get("cy"), 0.0) - _safe_float(prm.get("cy"), 0.0),
                    )
                    raw_step = math.hypot(
                        _safe_float(frames[fi].get("mx"), _safe_float(frames[fi].get("x"), 0.0))
                        - _safe_float(frames[fi - 1].get("mx"), _safe_float(frames[fi - 1].get("x"), 0.0)),
                        _safe_float(frames[fi].get("my"), _safe_float(frames[fi].get("y"), 0.0))
                        - _safe_float(frames[fi - 1].get("my"), _safe_float(frames[fi - 1].get("y"), 0.0)),
                    )
                    if car_step < 0.05 and raw_step > 0.5:
                        moving_stall += 1
                mean_dist = float(dist_acc / max(1, dist_n))
                objective = float(mean_dist + 1.6 * illegal_jumps + 0.42 * moving_stall + 0.06 * switches)
                return {
                    "objective": float(objective),
                    "mean_dist": float(mean_dist),
                    "illegal_jumps": float(illegal_jumps),
                    "moving_stall": float(moving_stall),
                    "switches": float(switches),
                }

            fidelity_maps: List[Dict[str, object]] = []
            for rm in raw_mappings:
                if rm:
                    fidelity_maps.append(dict(rm))
                else:
                    fidelity_maps.append({})

            prev_cli_fid = -1
            prev_xy_fid: Optional[Tuple[float, float]] = None
            for fi in range(len(fidelity_maps)):
                if fi >= len(frames) or not isinstance(frames[fi], dict):
                    continue
                rm_cur = fidelity_maps[fi]
                if not rm_cur:
                    rm_cur = {}
                    fidelity_maps[fi] = rm_cur

                fmx = _safe_float(frames[fi].get("mx"), _safe_float(frames[fi].get("x"), 0.0))
                fmy = _safe_float(frames[fi].get("my"), _safe_float(frames[fi].get("y"), 0.0))
                fmyaw = _safe_float(frames[fi].get("myaw"), _safe_float(frames[fi].get("yaw"), 0.0))
                frame_mhy = _frame_motion_yaw(fi)
                raw_step_fid = 0.0
                if fi > 0 and isinstance(frames[fi - 1], dict):
                    pmx_fid = _safe_float(frames[fi - 1].get("mx"), _safe_float(frames[fi - 1].get("x"), 0.0))
                    pmy_fid = _safe_float(frames[fi - 1].get("my"), _safe_float(frames[fi - 1].get("y"), 0.0))
                    raw_step_fid = math.hypot(float(fmx - pmx_fid), float(fmy - pmy_fid))

                map_hints = _map_hints_for_frame(fi)
                candidate_lines: List[int] = []
                if prev_cli_fid >= 0:
                    candidate_lines.append(int(prev_cli_fid))
                    for ci in sorted(carla_succs.get(int(prev_cli_fid), set())):
                        candidate_lines.append(int(ci))
                    for ci in sorted(carla_preds.get(int(prev_cli_fid), set())):
                        candidate_lines.append(int(ci))
                    for ci in sorted(carla_lat_adj.get(int(prev_cli_fid), set())):
                        candidate_lines.append(int(ci))
                for ci in map_hints.keys():
                    candidate_lines.append(int(ci))
                for ci in _nearby_carla_lines(float(fmx), float(fmy), top_k=28):
                    candidate_lines.append(int(ci))

                seen_cis: set[int] = set()
                best_cand: Optional[Dict[str, object]] = None
                same_cand: Optional[Dict[str, object]] = None
                for ci in candidate_lines:
                    if int(ci) < 0 or int(ci) in seen_cis:
                        continue
                    seen_cis.add(int(ci))
                    hint = map_hints.get(int(ci), {})
                    mapped_rev_hint = hint.get("reversed")
                    if mapped_rev_hint is not None:
                        mapped_rev_hint = bool(mapped_rev_hint)
                    cand = _nearest_projection_on_line(
                        cidx=int(ci),
                        qx=float(fmx),
                        qy=float(fmy),
                        mapped_reversed=mapped_rev_hint,
                        force_bidir=bool(int(ci) in map_hints),
                    )
                    if cand is None:
                        continue
                    proj = dict(cand.get("proj", {}))
                    p_dist = float(proj.get("dist", float("inf")))
                    p_yaw = _safe_float(proj.get("yaw"), float("nan"))
                    if not math.isfinite(p_dist) or p_dist > 12.0:
                        continue
                    if frame_mhy is not None and math.isfinite(p_yaw):
                        if _yaw_abs_diff_deg(float(frame_mhy), float(p_yaw)) > float(_DIR_OPPOSITE_REJECT_DEG):
                            continue

                    score = float(p_dist)
                    proj_x = float(proj.get("x", fmx))
                    proj_y = float(proj.get("y", fmy))
                    if prev_xy_fid is not None:
                        step_d = math.hypot(float(proj_x) - float(prev_xy_fid[0]), float(proj_y) - float(prev_xy_fid[1]))
                        if step_d > 3.0:
                            score += 0.18 * float(step_d - 3.0)
                        if prev_cli_fid >= 0 and int(ci) == int(prev_cli_fid):
                            # Avoid frozen projection on the same line when raw trajectory clearly moves.
                            if raw_step_fid > 0.5 and step_d < 0.05:
                                score += 1.1
                    if prev_cli_fid >= 0 and int(ci) != int(prev_cli_fid):
                        succ_like = _transition_has_successor_connect(int(prev_cli_fid), int(ci))
                        lat_like = bool(
                            int(ci) in carla_lat_adj.get(int(prev_cli_fid), set())
                            or int(prev_cli_fid) in carla_lat_adj.get(int(ci), set())
                        )
                        if succ_like:
                            score += 0.75
                        elif lat_like:
                            score += 1.05
                        else:
                            score += 2.6

                    q_hint = str(hint.get("quality", "none")).strip().lower()
                    if q_hint == "high":
                        score -= 0.15
                    elif q_hint == "medium":
                        score -= 0.08
                    elif q_hint == "poor":
                        score += 0.15

                    cand_ex = {
                        "ci": int(ci),
                        "proj": proj,
                        "score": float(score),
                    }
                    if prev_cli_fid >= 0 and int(ci) == int(prev_cli_fid):
                        same_cand = cand_ex
                    if best_cand is None or float(cand_ex["score"]) < float(best_cand["score"]):
                        best_cand = cand_ex

                chosen = best_cand
                if chosen is not None and same_cand is not None and prev_cli_fid >= 0:
                    if int(chosen.get("ci", -1)) != int(prev_cli_fid):
                        best_dist = float(chosen.get("proj", {}).get("dist", float("inf")))
                        same_dist = float(same_cand.get("proj", {}).get("dist", float("inf")))
                        same_step = float("inf")
                        if prev_xy_fid is not None:
                            same_step = math.hypot(
                                float(same_cand.get("proj", {}).get("x", prev_xy_fid[0])) - float(prev_xy_fid[0]),
                                float(same_cand.get("proj", {}).get("y", prev_xy_fid[1])) - float(prev_xy_fid[1]),
                            )
                        allow_same_hysteresis = not (raw_step_fid > 0.5 and math.isfinite(same_step) and same_step < 0.05)
                        if allow_same_hysteresis and math.isfinite(same_dist) and (same_dist <= (best_dist + 0.75)):
                            chosen = same_cand
                        elif allow_same_hysteresis and float(same_cand.get("score", float("inf"))) <= (float(chosen.get("score", float("inf"))) + 0.25):
                            chosen = same_cand

                if chosen is None:
                    if int(rm_cur.get("cli", -1)) >= 0:
                        prev_cli_fid = int(rm_cur.get("cli", -1))
                        prev_xy_fid = (
                            float(rm_cur.get("cx", fmx)),
                            float(rm_cur.get("cy", fmy)),
                        )
                    continue

                cproj = dict(chosen.get("proj", {}))
                cci = int(chosen.get("ci", -1))
                rm_cur["cli"] = int(cci)
                rm_cur["cx"] = float(cproj.get("x", fmx))
                rm_cur["cy"] = float(cproj.get("y", fmy))
                rm_cur["cyaw"] = float(cproj.get("yaw", fmyaw))
                rm_cur["cld"] = float(cproj.get("dist", 0.0))
                rm_cur["csource"] = "fidelity_snap"
                rm_cur["cquality"] = "fidelity"
                prev_cli_fid = int(cci)
                prev_xy_fid = (float(rm_cur["cx"]), float(rm_cur["cy"]))

            base_stats = _mapping_stats(raw_mappings)
            fid_stats = _mapping_stats(fidelity_maps)
            choose_fidelity = bool(float(fid_stats["objective"]) + 0.05 < float(base_stats["objective"]))
            if not choose_fidelity:
                if (
                    float(base_stats["moving_stall"]) >= 6.0
                    and float(fid_stats["moving_stall"]) <= float(base_stats["moving_stall"]) - 2.0
                    and float(fid_stats["illegal_jumps"]) <= float(base_stats["illegal_jumps"]) + 2.0
                ):
                    choose_fidelity = True
            if choose_fidelity:
                raw_mappings = fidelity_maps

            # Final anti-freeze fix after mapping selection.
            for fi in range(1, len(raw_mappings)):
                prev_rm = raw_mappings[fi - 1]
                cur_rm = raw_mappings[fi]
                if not prev_rm or not cur_rm:
                    continue
                if fi >= len(frames) or not isinstance(frames[fi], dict) or not isinstance(frames[fi - 1], dict):
                    continue
                prev_cx = _safe_float(prev_rm.get("cx"), float("nan"))
                prev_cy = _safe_float(prev_rm.get("cy"), float("nan"))
                cur_cx = _safe_float(cur_rm.get("cx"), float("nan"))
                cur_cy = _safe_float(cur_rm.get("cy"), float("nan"))
                if not (math.isfinite(prev_cx) and math.isfinite(prev_cy) and math.isfinite(cur_cx) and math.isfinite(cur_cy)):
                    continue
                car_step = math.hypot(float(cur_cx - prev_cx), float(cur_cy - prev_cy))
                pmx = _safe_float(frames[fi - 1].get("mx"), _safe_float(frames[fi - 1].get("x"), 0.0))
                pmy = _safe_float(frames[fi - 1].get("my"), _safe_float(frames[fi - 1].get("y"), 0.0))
                fmx = _safe_float(frames[fi].get("mx"), _safe_float(frames[fi].get("x"), 0.0))
                fmy = _safe_float(frames[fi].get("my"), _safe_float(frames[fi].get("y"), 0.0))
                fmyaw = _safe_float(frames[fi].get("myaw"), _safe_float(frames[fi].get("yaw"), 0.0))
                raw_step = math.hypot(float(fmx - pmx), float(fmy - pmy))
                if not (car_step < 0.05 and raw_step > 0.5):
                    continue

                prev_cli = int(prev_rm.get("cli", -1))
                map_hints_f = _map_hints_for_frame(fi)
                cand_lines_f: List[int] = []
                if prev_cli >= 0:
                    cand_lines_f.append(int(prev_cli))
                    for ci in sorted(carla_succs.get(int(prev_cli), set())):
                        cand_lines_f.append(int(ci))
                    for ci in sorted(carla_preds.get(int(prev_cli), set())):
                        cand_lines_f.append(int(ci))
                    for ci in sorted(carla_lat_adj.get(int(prev_cli), set())):
                        cand_lines_f.append(int(ci))
                for ci in map_hints_f.keys():
                    cand_lines_f.append(int(ci))
                cur_cld_f = _safe_float(cur_rm.get("cld"), float("inf"))

                frame_mhy_f = _frame_motion_yaw(fi)
                seen_f: set[int] = set()
                best_f: Optional[Dict[str, object]] = None
                for ci in cand_lines_f:
                    if int(ci) < 0 or int(ci) in seen_f:
                        continue
                    seen_f.add(int(ci))
                    hint_f = map_hints_f.get(int(ci), {})
                    rev_f = hint_f.get("reversed")
                    if rev_f is not None:
                        rev_f = bool(rev_f)
                    cand_f = _nearest_projection_on_line(
                        cidx=int(ci),
                        qx=float(fmx),
                        qy=float(fmy),
                        mapped_reversed=rev_f,
                        force_bidir=bool(int(ci) in map_hints_f),
                    )
                    if cand_f is None:
                        continue
                    proj_f = dict(cand_f.get("proj", {}))
                    p_dist_f = float(proj_f.get("dist", float("inf")))
                    p_yaw_f = _safe_float(proj_f.get("yaw"), float("nan"))
                    if not math.isfinite(p_dist_f) or p_dist_f > 12.0:
                        continue
                    if frame_mhy_f is not None and math.isfinite(p_yaw_f):
                        if _yaw_abs_diff_deg(float(frame_mhy_f), float(p_yaw_f)) > float(_DIR_OPPOSITE_REJECT_DEG):
                            continue
                    step_f = math.hypot(float(proj_f.get("x", prev_cx)) - float(prev_cx), float(proj_f.get("y", prev_cy)) - float(prev_cy))
                    if step_f < 0.05:
                        continue

                    qf = str(hint_f.get("quality", "none")).strip().lower()
                    is_map_hint = bool(int(ci) in map_hints_f)
                    if prev_cli >= 0 and int(ci) != int(prev_cli):
                        if not is_map_hint:
                            continue
                        succ_like = _transition_has_successor_connect(int(prev_cli), int(ci))
                        lat_like = bool(
                            int(ci) in carla_lat_adj.get(int(prev_cli), set())
                            or int(prev_cli) in carla_lat_adj.get(int(ci), set())
                        )
                        if not succ_like and not lat_like and not is_map_hint:
                            continue
                        if is_map_hint and qf == "poor" and math.isfinite(cur_cld_f):
                            if (p_dist_f + 0.8) >= float(cur_cld_f):
                                continue

                    score_f = float(p_dist_f)
                    if prev_cli >= 0 and int(ci) != int(prev_cli):
                        if _transition_has_successor_connect(int(prev_cli), int(ci)):
                            score_f += 0.65
                        elif int(ci) in carla_lat_adj.get(int(prev_cli), set()) or int(prev_cli) in carla_lat_adj.get(int(ci), set()):
                            score_f += 1.0
                        else:
                            score_f += 2.2
                    if qf == "high":
                        score_f -= 0.12
                    elif qf == "medium":
                        score_f -= 0.06

                    cand_scored = {
                        "ci": int(ci),
                        "proj": proj_f,
                        "score": float(score_f),
                    }
                    if best_f is None or float(cand_scored["score"]) < float(best_f["score"]):
                        best_f = cand_scored

                if best_f is None:
                    continue
                bproj = dict(best_f.get("proj", {}))
                cur_rm["cli"] = int(best_f.get("ci", cur_rm.get("cli", -1)))
                cur_rm["cx"] = float(bproj.get("x", fmx))
                cur_rm["cy"] = float(bproj.get("y", fmy))
                cur_rm["cyaw"] = float(bproj.get("yaw", fmyaw))
                cur_rm["cld"] = float(bproj.get("dist", cur_rm.get("cld", 0.0)))
                cur_rm["csource"] = "anti_freeze_final"
                cur_rm["cquality"] = "fidelity"

        # =====================================================================
        # Pass 8.95: Post-pass jitter collapse (A-B-A short oscillations)
        # =====================================================================
        # Later legal/reanchor passes can re-introduce short toggles between
        # nearby lines. Collapse short A-B-A runs unless B is clearly a much
        # better geometric fit or explicitly marked as lane-change motion.
        if enforce_direction and len(raw_mappings) >= 3:
            _POST_ABA_MAX_FRAMES = 6
            _POST_ABA_KEEP_IMPROVE_M = 0.45

            def _post_run_dist_to_line(fi: int, cidx: int) -> float:
                if fi < 0 or fi >= len(frames) or not isinstance(frames[fi], dict):
                    return float("inf")
                fmx_d = _safe_float(frames[fi].get("mx"), _safe_float(frames[fi].get("x"), 0.0))
                fmy_d = _safe_float(frames[fi].get("my"), _safe_float(frames[fi].get("y"), 0.0))
                cand_d = _nearest_projection_on_line(
                    cidx=int(cidx),
                    qx=float(fmx_d),
                    qy=float(fmy_d),
                    mapped_reversed=None,
                    force_bidir=True,
                )
                if cand_d is None:
                    return float("inf")
                return float(cand_d.get("proj", {}).get("dist", float("inf")))

            post_changed = True
            while post_changed:
                post_changed = False
                post_runs: List[List[int]] = []  # [cli, start, end]
                for fi, rm in enumerate(raw_mappings):
                    cli_val = int(rm.get("cli", -1)) if rm else -1
                    if not post_runs or cli_val != int(post_runs[-1][0]):
                        post_runs.append([int(cli_val), int(fi), int(fi)])
                    else:
                        post_runs[-1][2] = int(fi)

                for ri in range(1, len(post_runs) - 1):
                    a_cli, a_s, a_e = post_runs[ri - 1]
                    b_cli, b_s, b_e = post_runs[ri]
                    c_cli, c_s, c_e = post_runs[ri + 1]
                    if int(a_cli) < 0 or int(b_cli) < 0 or int(c_cli) < 0:
                        continue
                    if int(a_cli) != int(c_cli) or int(a_cli) == int(b_cli):
                        continue
                    b_len = int(b_e) - int(b_s) + 1
                    if b_len > int(_POST_ABA_MAX_FRAMES):
                        continue

                    has_lanechange_src = False
                    for fi in range(int(b_s), int(b_e) + 1):
                        rm_b = raw_mappings[fi]
                        if not rm_b:
                            continue
                        src_b = str(rm_b.get("csource", "")).strip().lower()
                        if src_b.startswith("lanechange"):
                            has_lanechange_src = True
                            break
                    if has_lanechange_src:
                        continue

                    dist_a_vals: List[float] = []
                    dist_b_vals: List[float] = []
                    for fi in range(int(b_s), int(b_e) + 1):
                        da = _post_run_dist_to_line(int(fi), int(a_cli))
                        db = _post_run_dist_to_line(int(fi), int(b_cli))
                        if math.isfinite(da):
                            dist_a_vals.append(float(da))
                        if math.isfinite(db):
                            dist_b_vals.append(float(db))
                    if not dist_a_vals or not dist_b_vals:
                        continue
                    mean_a = float(sum(dist_a_vals) / max(1, len(dist_a_vals)))
                    mean_b = float(sum(dist_b_vals) / max(1, len(dist_b_vals)))
                    if (mean_b + float(_POST_ABA_KEEP_IMPROVE_M)) < mean_a:
                        continue

                    prev_cli_post = int(a_cli)
                    prev_xy_post: Optional[Tuple[float, float]] = None
                    if int(b_s) > 0 and raw_mappings[int(b_s) - 1]:
                        prev_cli_post = int(raw_mappings[int(b_s) - 1].get("cli", int(a_cli)))
                        prev_xy_post = (
                            float(raw_mappings[int(b_s) - 1].get("cx", 0.0)),
                            float(raw_mappings[int(b_s) - 1].get("cy", 0.0)),
                        )

                    for fi in range(int(b_s), int(b_e) + 1):
                        if fi >= len(frames) or not isinstance(frames[fi], dict):
                            continue
                        fmx_post = _safe_float(frames[fi].get("mx"), _safe_float(frames[fi].get("x"), 0.0))
                        fmy_post = _safe_float(frames[fi].get("my"), _safe_float(frames[fi].get("y"), 0.0))
                        fmyaw_post = _safe_float(frames[fi].get("myaw"), _safe_float(frames[fi].get("yaw"), 0.0))
                        cand_post = _best_projection_on_line(
                            fi=fi,
                            cidx=int(a_cli),
                            qx=float(fmx_post),
                            qy=float(fmy_post),
                            prev_cli=int(prev_cli_post),
                            prev_xy=prev_xy_post,
                            mapped_reversed=None,
                            force_bidir=True,
                        )
                        if cand_post is None:
                            cand_post = _nearest_projection_on_line(
                                cidx=int(a_cli),
                                qx=float(fmx_post),
                                qy=float(fmy_post),
                                mapped_reversed=None,
                                force_bidir=True,
                            )
                            if cand_post is None:
                                continue
                        pproj = dict(cand_post.get("proj", {}))
                        p_yaw = _safe_float(pproj.get("yaw"), float("nan"))
                        mhy_post = _frame_motion_yaw(fi)
                        if mhy_post is not None and math.isfinite(p_yaw):
                            if _yaw_abs_diff_deg(float(mhy_post), float(p_yaw)) > float(_DIR_OPPOSITE_REJECT_DEG):
                                continue

                        rm_post = raw_mappings[fi]
                        rm_post["cli"] = int(a_cli)
                        rm_post["cx"] = float(pproj.get("x", fmx_post))
                        rm_post["cy"] = float(pproj.get("y", fmy_post))
                        rm_post["cyaw"] = float(pproj.get("yaw", fmyaw_post))
                        rm_post["cld"] = float(pproj.get("dist", rm_post.get("cld", 0.0)))
                        rm_post["csource"] = "post_aba_collapse"
                        rm_post["cquality"] = "continuity"

                        prev_cli_post = int(a_cli)
                        prev_xy_post = (float(rm_post["cx"]), float(rm_post["cy"]))

                    post_changed = True
                    break

            def _post_reproject_run_to_cli(run_s: int, run_e: int, target_cli: int, src_label: str) -> bool:
                changed_local = False
                prev_cli_post = int(target_cli)
                prev_xy_post: Optional[Tuple[float, float]] = None
                if int(run_s) > 0 and raw_mappings[int(run_s) - 1]:
                    prev_cli_post = int(raw_mappings[int(run_s) - 1].get("cli", int(target_cli)))
                    prev_xy_post = (
                        float(raw_mappings[int(run_s) - 1].get("cx", 0.0)),
                        float(raw_mappings[int(run_s) - 1].get("cy", 0.0)),
                    )

                for fi in range(int(run_s), int(run_e) + 1):
                    if fi >= len(frames) or not isinstance(frames[fi], dict):
                        continue
                    fmx_post = _safe_float(frames[fi].get("mx"), _safe_float(frames[fi].get("x"), 0.0))
                    fmy_post = _safe_float(frames[fi].get("my"), _safe_float(frames[fi].get("y"), 0.0))
                    fmyaw_post = _safe_float(frames[fi].get("myaw"), _safe_float(frames[fi].get("yaw"), 0.0))
                    cand_post = _best_projection_on_line(
                        fi=fi,
                        cidx=int(target_cli),
                        qx=float(fmx_post),
                        qy=float(fmy_post),
                        prev_cli=int(prev_cli_post),
                        prev_xy=prev_xy_post,
                        mapped_reversed=None,
                        force_bidir=True,
                    )
                    if cand_post is None:
                        cand_post = _nearest_projection_on_line(
                            cidx=int(target_cli),
                            qx=float(fmx_post),
                            qy=float(fmy_post),
                            mapped_reversed=None,
                            force_bidir=True,
                        )
                        if cand_post is None:
                            continue
                    pproj = dict(cand_post.get("proj", {}))
                    p_yaw = _safe_float(pproj.get("yaw"), float("nan"))
                    mhy_post = _frame_motion_yaw(fi)
                    if mhy_post is not None and math.isfinite(p_yaw):
                        if _yaw_abs_diff_deg(float(mhy_post), float(p_yaw)) > float(_DIR_OPPOSITE_REJECT_DEG):
                            continue

                    rm_post = raw_mappings[fi]
                    rm_post["cli"] = int(target_cli)
                    rm_post["cx"] = float(pproj.get("x", fmx_post))
                    rm_post["cy"] = float(pproj.get("y", fmy_post))
                    rm_post["cyaw"] = float(pproj.get("yaw", fmyaw_post))
                    rm_post["cld"] = float(pproj.get("dist", rm_post.get("cld", 0.0)))
                    rm_post["csource"] = str(src_label)
                    rm_post["cquality"] = "continuity"

                    prev_cli_post = int(target_cli)
                    prev_xy_post = (float(rm_post["cx"]), float(rm_post["cy"]))
                    changed_local = True

                return bool(changed_local)

            # Collapse tiny edge/intermediate runs when a neighbour line is
            # clearly closer; this removes last-frame ping-pong.
            _POST_SHORT_RUN_MAX_FRAMES = 2
            _POST_SHORT_RUN_IMPROVE_M = 0.35
            post_merge_changed = True
            while post_merge_changed:
                post_merge_changed = False
                post_runs2: List[List[int]] = []  # [cli, start, end]
                for fi, rm in enumerate(raw_mappings):
                    cli_val = int(rm.get("cli", -1)) if rm else -1
                    if not post_runs2 or cli_val != int(post_runs2[-1][0]):
                        post_runs2.append([int(cli_val), int(fi), int(fi)])
                    else:
                        post_runs2[-1][2] = int(fi)

                for ri, run in enumerate(post_runs2):
                    cur_cli, rs, re = int(run[0]), int(run[1]), int(run[2])
                    if cur_cli < 0:
                        continue
                    run_len = re - rs + 1
                    if run_len > int(_POST_SHORT_RUN_MAX_FRAMES):
                        continue
                    src_vals = {
                        str(raw_mappings[fi].get("csource", "")).strip().lower()
                        for fi in range(rs, re + 1)
                        if raw_mappings[fi]
                    }
                    if any(s.startswith("lanechange") for s in src_vals):
                        continue

                    candidate_cis: List[int] = []
                    if ri > 0:
                        prev_cli2 = int(post_runs2[ri - 1][0])
                        if prev_cli2 >= 0 and prev_cli2 != cur_cli:
                            candidate_cis.append(int(prev_cli2))
                    if ri + 1 < len(post_runs2):
                        next_cli2 = int(post_runs2[ri + 1][0])
                        if next_cli2 >= 0 and next_cli2 != cur_cli:
                            candidate_cis.append(int(next_cli2))
                    if not candidate_cis:
                        continue

                    cur_d_vals = [_post_run_dist_to_line(fi, int(cur_cli)) for fi in range(rs, re + 1)]
                    cur_d_vals = [float(v) for v in cur_d_vals if math.isfinite(v)]
                    if not cur_d_vals:
                        continue
                    cur_mean = float(sum(cur_d_vals) / max(1, len(cur_d_vals)))

                    best_alt_cli = -1
                    best_alt_mean = float("inf")
                    for alt_cli in candidate_cis:
                        alt_vals = [_post_run_dist_to_line(fi, int(alt_cli)) for fi in range(rs, re + 1)]
                        alt_vals = [float(v) for v in alt_vals if math.isfinite(v)]
                        if not alt_vals:
                            continue
                        alt_mean = float(sum(alt_vals) / max(1, len(alt_vals)))
                        if alt_mean < best_alt_mean:
                            best_alt_mean = float(alt_mean)
                            best_alt_cli = int(alt_cli)

                    if best_alt_cli < 0:
                        continue
                    if (float(best_alt_mean) + float(_POST_SHORT_RUN_IMPROVE_M)) >= float(cur_mean):
                        continue

                    if _post_reproject_run_to_cli(rs, re, int(best_alt_cli), "post_short_merge"):
                        post_merge_changed = True
                        break

            # Final de-freeze after post-collapse passes.
            for fi in range(1, len(raw_mappings)):
                prev_rm = raw_mappings[fi - 1]
                cur_rm = raw_mappings[fi]
                if not prev_rm or not cur_rm:
                    continue
                prev_cli = int(prev_rm.get("cli", -1))
                cur_cli = int(cur_rm.get("cli", -1))
                if prev_cli < 0 or cur_cli < 0 or prev_cli != cur_cli:
                    continue
                if fi >= len(frames) or not isinstance(frames[fi], dict) or not isinstance(frames[fi - 1], dict):
                    continue
                prev_cx = _safe_float(prev_rm.get("cx"), float("nan"))
                prev_cy = _safe_float(prev_rm.get("cy"), float("nan"))
                cur_cx = _safe_float(cur_rm.get("cx"), float("nan"))
                cur_cy = _safe_float(cur_rm.get("cy"), float("nan"))
                if not (math.isfinite(prev_cx) and math.isfinite(prev_cy) and math.isfinite(cur_cx) and math.isfinite(cur_cy)):
                    continue
                car_step = math.hypot(float(cur_cx - prev_cx), float(cur_cy - prev_cy))
                pmx_post = _safe_float(frames[fi - 1].get("mx"), _safe_float(frames[fi - 1].get("x"), 0.0))
                pmy_post = _safe_float(frames[fi - 1].get("my"), _safe_float(frames[fi - 1].get("y"), 0.0))
                fmx_post = _safe_float(frames[fi].get("mx"), _safe_float(frames[fi].get("x"), 0.0))
                fmy_post = _safe_float(frames[fi].get("my"), _safe_float(frames[fi].get("y"), 0.0))
                fmyaw_post = _safe_float(frames[fi].get("myaw"), _safe_float(frames[fi].get("yaw"), 0.0))
                raw_step = math.hypot(float(fmx_post - pmx_post), float(fmy_post - pmy_post))
                if not (car_step < 0.05 and raw_step > 0.5):
                    continue

                cand_post = _nearest_projection_on_line(
                    cidx=int(cur_cli),
                    qx=float(fmx_post),
                    qy=float(fmy_post),
                    mapped_reversed=None,
                    force_bidir=True,
                )
                if cand_post is None:
                    continue
                pproj = dict(cand_post.get("proj", {}))
                p_dist = float(pproj.get("dist", float("inf")))
                p_yaw = _safe_float(pproj.get("yaw"), float("nan"))
                step_post = math.hypot(
                    float(pproj.get("x", prev_cx)) - float(prev_cx),
                    float(pproj.get("y", prev_cy)) - float(prev_cy),
                )
                if math.isfinite(p_dist) and p_dist <= 12.0 and step_post <= 0.05:
                    # Nearest projection can stick to the same vertex when raw
                    # motion is mostly lateral; try projecting a motion-shifted
                    # target to recover forward progress on the same line.
                    tx_post = float(prev_cx + (fmx_post - pmx_post))
                    ty_post = float(prev_cy + (fmy_post - pmy_post))
                    cand_shift = _nearest_projection_on_line(
                        cidx=int(cur_cli),
                        qx=float(tx_post),
                        qy=float(ty_post),
                        mapped_reversed=None,
                        force_bidir=True,
                    )
                    if cand_shift is not None:
                        sproj = dict(cand_shift.get("proj", {}))
                        s_dist = float(sproj.get("dist", float("inf")))
                        s_yaw = _safe_float(sproj.get("yaw"), float("nan"))
                        s_step = math.hypot(
                            float(sproj.get("x", prev_cx)) - float(prev_cx),
                            float(sproj.get("y", prev_cy)) - float(prev_cy),
                        )
                        if math.isfinite(s_dist) and s_dist <= 12.0 and s_step > 0.05:
                            pproj = sproj
                            p_dist = float(s_dist)
                            p_yaw = float(s_yaw)
                            step_post = float(s_step)
                    if step_post <= 0.05:
                        # Last resort: advance along lane tangent by raw step.
                        prev_on_line = _nearest_projection_on_line(
                            cidx=int(cur_cli),
                            qx=float(prev_cx),
                            qy=float(prev_cy),
                            mapped_reversed=None,
                            force_bidir=True,
                        )
                        if prev_on_line is not None:
                            pl = dict(prev_on_line.get("proj", {}))
                            tan_yaw = _safe_float(pl.get("yaw"), float("nan"))
                            if math.isfinite(tan_yaw):
                                mhy_post = _frame_motion_yaw(fi)
                                if mhy_post is not None and _yaw_abs_diff_deg(float(mhy_post), float(tan_yaw)) > 90.0:
                                    tan_yaw = _normalize_yaw_deg(float(tan_yaw) + 180.0)
                                raw_step_mag = math.hypot(float(fmx_post - pmx_post), float(fmy_post - pmy_post))
                                tx_tan = float(prev_cx + raw_step_mag * math.cos(math.radians(float(tan_yaw))))
                                ty_tan = float(prev_cy + raw_step_mag * math.sin(math.radians(float(tan_yaw))))
                                cand_tan = _nearest_projection_on_line(
                                    cidx=int(cur_cli),
                                    qx=float(tx_tan),
                                    qy=float(ty_tan),
                                    mapped_reversed=None,
                                    force_bidir=True,
                                )
                                if cand_tan is not None:
                                    tproj = dict(cand_tan.get("proj", {}))
                                    t_dist = float(tproj.get("dist", float("inf")))
                                    t_yaw = _safe_float(tproj.get("yaw"), float("nan"))
                                    t_step = math.hypot(
                                        float(tproj.get("x", prev_cx)) - float(prev_cx),
                                        float(tproj.get("y", prev_cy)) - float(prev_cy),
                                    )
                                    if math.isfinite(t_dist) and t_dist <= 12.0 and t_step > 0.05:
                                        pproj = tproj
                                        p_dist = float(t_dist)
                                        p_yaw = float(t_yaw)
                                        step_post = float(t_step)

                if not (math.isfinite(p_dist) and p_dist <= 12.0 and step_post > 0.05):
                    tx_fb = float(prev_cx + (fmx_post - pmx_post))
                    ty_fb = float(prev_cy + (fmy_post - pmy_post))
                    fb_step = math.hypot(float(tx_fb - prev_cx), float(ty_fb - prev_cy))
                    if fb_step <= 0.05:
                        continue
                    fb_yaw = _normalize_yaw_deg(math.degrees(math.atan2(float(ty_fb - prev_cy), float(tx_fb - prev_cx))))
                    mhy_fb = _frame_motion_yaw(fi)
                    if mhy_fb is not None and _yaw_abs_diff_deg(float(mhy_fb), float(fb_yaw)) > float(_DIR_OPPOSITE_REJECT_DEG):
                        continue
                    cur_rm["cx"] = float(tx_fb)
                    cur_rm["cy"] = float(ty_fb)
                    cur_rm["cyaw"] = float(fb_yaw)
                    cur_rm["cld"] = float(cur_rm.get("cld", p_dist if math.isfinite(p_dist) else 0.0))
                    cur_rm["csource"] = "post_anti_freeze_shift"
                    cur_rm["cquality"] = "continuity"
                    continue
                mhy_post = _frame_motion_yaw(fi)
                if mhy_post is not None and math.isfinite(p_yaw):
                    if _yaw_abs_diff_deg(float(mhy_post), float(p_yaw)) > float(_DIR_OPPOSITE_REJECT_DEG):
                        continue
                cur_rm["cx"] = float(pproj.get("x", fmx_post))
                cur_rm["cy"] = float(pproj.get("y", fmy_post))
                cur_rm["cyaw"] = float(pproj.get("yaw", fmyaw_post))
                cur_rm["cld"] = float(p_dist)
                cur_rm["csource"] = "post_anti_freeze"
                cur_rm["cquality"] = "continuity"

        # =====================================================================
        # Write final mappings to frames and count lane changes
        # =====================================================================
        phantom_changes = 0
        prev_cli = -1
        for fi, fr in enumerate(frames):
            if not isinstance(fr, dict) or fi >= len(raw_mappings):
                continue
            rm = raw_mappings[fi]
            if not rm:
                continue
            fr["cx"] = float(rm.get("cx", 0.0))
            fr["cy"] = float(rm.get("cy", 0.0))
            fr["cyaw"] = float(rm.get("cyaw", 0.0))
            fr["cli"] = int(rm.get("cli", -1))
            fr["cld"] = float(rm.get("cld", float("inf")))
            fr["csource"] = str(rm.get("csource", "fallback_raw"))
            fr["cquality"] = str(rm.get("cquality", "none"))
            for tag in ("base", "cont", "turn"):
                fr[f"cx_{tag}"] = float(rm.get(f"cx_{tag}", rm.get("cx", 0.0)))
                fr[f"cy_{tag}"] = float(rm.get(f"cy_{tag}", rm.get("cy", 0.0)))
                fr[f"cyaw_{tag}"] = float(rm.get(f"cyaw_{tag}", rm.get("cyaw", 0.0)))
                fr[f"cli_{tag}"] = int(rm.get(f"cli_{tag}", rm.get("cli", -1)))
                fr[f"cld_{tag}"] = float(rm.get(f"cld_{tag}", rm.get("cld", float("inf"))))
                fr[f"csource_{tag}"] = str(rm.get(f"csource_{tag}", rm.get("csource", "")))
                fr[f"cquality_{tag}"] = str(rm.get(f"cquality_{tag}", rm.get("cquality", "")))
            cur_cli = int(rm.get("cli", -1))
            # Only count transitions between DIFFERENT cli values
            # in anchored regions as phantom changes
            src = str(rm.get("csource", ""))
            if prev_cli >= 0 and cur_cli >= 0 and cur_cli != prev_cli:
                if src in ("lane_correspondence", "lane_projection", "smoothed"):
                    phantom_changes += 1
            prev_cli = cur_cli

        tr["carla_lane_changes"] = int(phantom_changes)

    qual_counts: Dict[str, int] = {}
    for m in lane_to_carla.values():
        q = str(m.get("quality", "poor"))
        qual_counts[q] = qual_counts.get(q, 0) + 1
    usable_lane_count = int(sum(int(v) for q, v in qual_counts.items() if str(q) != "poor"))

    # Compute validation diagnostics
    total_phantom_changes = sum(
        int(tr.get("carla_lane_changes", 0))
        for tr in all_tracks
        if isinstance(tr, dict)
    )
    split_merge_count = len(correspondence.get("split_merges", {}))

    # Connectivity validation: count how many V2XPNP connectivity edges
    # are preserved (matched lanes → connected CARLA lines)
    v2_graph = correspondence.get("v2_graph", {})
    v2_succs: Dict[int, set] = v2_graph.get("successors", {}) if isinstance(v2_graph, dict) else {}
    conn_total = 0
    conn_preserved = 0
    for li_a, succs in v2_succs.items():
        for li_b in succs:
            ci_a_info = lane_to_carla.get(int(li_a))
            ci_b_info = lane_to_carla.get(int(li_b))
            if ci_a_info is None or ci_b_info is None:
                continue
            conn_total += 1
            ci_a = int(ci_a_info.get("carla_line_index", -1))
            ci_b = int(ci_b_info.get("carla_line_index", -1))
            if ci_a == ci_b:
                conn_preserved += 1
            elif ci_b in (correspondence.get("v2_graph", {}) or {}).get("successors", {}).get(ci_a, set()):
                conn_preserved += 1

    payload.setdefault("metadata", {})["lane_correspondence"] = {
        "enabled": True,
        "driving_lane_types": list(correspondence.get("driving_lane_types", [])),
        "mapped_lane_count": int(len(lane_to_carla)),
        "usable_lane_count": int(usable_lane_count),
        "mapped_carla_line_count": int(len(carla_to_lanes)),
        "quality_counts": qual_counts,
        "split_merge_count": int(split_merge_count),
        "total_phantom_lane_changes": int(total_phantom_changes),
        "connectivity_edges_total": int(conn_total),
        "connectivity_edges_preserved": int(conn_preserved),
        "wrong_way_frames_detected": int(direction_opposite_frames),
        "wrong_way_frames_fixed": int(direction_fixed_frames),
        "intersection_like_carla_lines": int(len(intersection_like_lines)),
        "intersection_turn_runs": int(intersection_turn_runs),
        "intersection_turn_polyline_snaps": int(intersection_turn_polyline_snaps),
        "intersection_turn_curve_only_runs": int(intersection_turn_curve_only),
        "lanechange_scurve_events": int(lanechange_scurve_events),
        "continuity_jump_repairs": int(continuity_jump_repairs),
        "illegal_lane_transitions_detected": int(illegal_lane_transition_detected),
        "illegal_lane_transitions_fixed": int(illegal_lane_transition_fixed),
        "continuity_scoring": {
            "switch_penalty": float(_DIR_SWITCH_PENALTY),
            "same_lane_bonus": float(_DIR_SAME_LANE_BONUS),
            "connected_bonus": float(_DIR_CONNECTED_BONUS),
            "disconnected_penalty": float(_DIR_DISCONNECTED_PENALTY),
            "step_soft_m": float(_DIR_STEP_SOFT_M),
            "step_weight": float(_DIR_STEP_WEIGHT),
        },
    }
    _corr_apply_dt = time.monotonic() - _corr_apply_t0
    print(f"[INFO] Lane correspondence projection pass complete in {_corr_apply_dt:.2f}s")

def _build_export_payload(
    scenario_dir: Path,
    selected_map: VectorMapData,
    selection_details: Sequence[Dict[str, object]],
    ego_trajs: Sequence[Sequence[Waypoint]],
    ego_times: Sequence[Sequence[float]],
    vehicles: Dict[int, Sequence[Waypoint]],
    vehicle_times: Dict[int, Sequence[float]],
    obj_info: Dict[int, Dict[str, object]],
    actor_source_subdir: Dict[int, str],
    actor_orig_vid: Dict[int, int],
    actor_alias_vids: Dict[int, List[int]],
    merge_stats: Dict[str, object],
    timing_optimization: Dict[str, object],
    matcher: LaneMatcher,
    snap_to_map: bool,
    map_max_points_per_line: int,
    lane_change_cfg: Optional[Dict[str, object]] = None,
    vehicle_lane_policy_cfg: Optional[Dict[str, object]] = None,
    parked_vehicle_cfg: Optional[Dict[str, float]] = None,
    carla_map_layer: Optional[Dict[str, object]] = None,
    default_dt: float = 0.1,
) -> Dict[str, object]:
    print(f"[INFO] Building ego track frames ({len(ego_trajs)} ego trajectories)...")
    ego_tracks: List[Dict[str, object]] = []
    for ego_idx, traj in enumerate(ego_trajs):
        times = ego_times[ego_idx] if ego_idx < len(ego_times) else []
        frames = _build_track_frames(
            traj,
            times,
            matcher,
            snap_to_map=snap_to_map,
            lane_change_cfg=lane_change_cfg,  # apply same map-lane stabilization to ego display
        )
        ego_tracks.append(
            {
                "id": f"ego_{ego_idx}",
                "name": f"ego_{ego_idx}",
                "role": "ego",
                "obj_type": "ego",
                "model": "ego",
                "source_subdir": str(ego_idx),
                "path_length_m": float(_trajectory_path_length(traj)),
                "frames": frames,
            }
        )

    print(f"[INFO] Building actor track frames ({len(vehicles)} actors)...")
    actor_tracks: List[Dict[str, object]] = []
    skipped_non_actors = 0
    map_lane_type_universe = sorted({str(l.lane_type) for l in selected_map.lanes})
    veh_policy_cfg = dict(vehicle_lane_policy_cfg or {})
    
    # Extract early spawn timing info for visualization
    early_spawn_info = timing_optimization.get("early_spawn", {})
    adjusted_spawn_times: Dict[str, float] = {}
    if early_spawn_info.get("applied"):
        raw_spawn_times = early_spawn_info.get("adjusted_spawn_times", {})
        for vid_str, spawn_t in raw_spawn_times.items():
            adjusted_spawn_times[str(vid_str)] = float(spawn_t)
    parked_cfg = dict(parked_vehicle_cfg or {})
    forbidden_lane_types = _parse_lane_type_set(veh_policy_cfg.get("forbidden_lane_types"), fallback=["2"])
    parked_only_lane_types = _parse_lane_type_set(veh_policy_cfg.get("parked_only_lane_types"), fallback=["3"])
    _actor_vid_list = sorted(vehicles.keys())
    for _actor_i, vid in enumerate(_progress_bar(_actor_vid_list, desc="Actor track frames", disable=len(_actor_vid_list) < 10)):
        traj = vehicles[vid]
        if not traj:
            continue
        meta = dict(obj_info.get(vid, {}))
        obj_type = str(meta.get("obj_type") or "npc")
        if not is_vehicle_type(obj_type):
            skipped_non_actors += 1
            continue

        model = str(meta.get("model") or map_obj_type(obj_type))
        role = _infer_actor_role(obj_type, traj)
        times = vehicle_times.get(vid, [])
        is_ped_or_cyclist = (str(role).lower() in {"walker", "cyclist"})
        is_motor_vehicle = str(role).lower() in {"vehicle", "npc", "static"}
        is_parked_vehicle = False
        parked_stats: Dict[str, float] = {}
        lane_policy: Optional[Dict[str, object]] = None
        if is_motor_vehicle:
            is_parked_vehicle, parked_stats = _is_parked_vehicle(
                traj=traj,
                times=times,
                default_dt=float(default_dt),
                cfg=parked_cfg,
            )
            allowed_types = set(map_lane_type_universe)
            allowed_types -= forbidden_lane_types
            if not is_parked_vehicle:
                allowed_types -= parked_only_lane_types
            stationary_types = parked_only_lane_types.intersection(allowed_types) if is_parked_vehicle else set()
            lane_policy = {
                "allowed_lane_types": sorted(str(v) for v in allowed_types),
                "stationary_when_lane_types": sorted(str(v) for v in stationary_types),
            }
        frames = _build_track_frames(
            traj,
            times,
            matcher,
            snap_to_map=snap_to_map,
            lane_change_cfg=lane_change_cfg,
            lane_policy=lane_policy,
            skip_snap=is_ped_or_cyclist,
        )
        # Compute spawn timing info
        first_observed_time = float(times[0]) if times else 0.0
        early_spawn_time = adjusted_spawn_times.get(str(vid), None)
        has_early_spawn = early_spawn_time is not None and early_spawn_time < first_observed_time - 1e-6
        
        actor_tracks.append(
            {
                "id": f"actor_{vid}",
                "vid": int(vid),
                "orig_vid": int(actor_orig_vid.get(vid, vid)),
                "merged_vids": [int(v) for v in actor_alias_vids.get(vid, [vid])],
                "name": f"actor_{vid}",
                "role": role,
                "obj_type": obj_type,
                "model": model,
                "length": _safe_float(meta.get("length"), 0.0),
                "width": _safe_float(meta.get("width"), 0.0),
                "source_subdir": str(actor_source_subdir.get(vid, "")),
                "path_length_m": float(_trajectory_path_length(traj)),
                "parked_vehicle": bool(is_parked_vehicle),
                "parked_motion_stats": parked_stats,
                "lane_snap_policy": lane_policy or {},
                "frames": frames,
                # Spawn timing visualization fields
                "first_observed_time": float(first_observed_time),
                "early_spawn_time": float(early_spawn_time) if early_spawn_time is not None else None,
                "has_early_spawn": bool(has_early_spawn),
            }
        )

    # --- Ego-vs-actor deduplication ---
    # Remove actor tracks that are duplicates of ego tracks (same physical
    # vehicle seen by both ego sensor and actor sensor).
    ego_actor_dedup_removed: List[str] = []
    for ego_tr in ego_tracks:
        ego_frames = ego_tr.get("frames", [])
        if not ego_frames:
            continue
        ego_xy = [(float(f.get("mx", f.get("x", 0))), float(f.get("my", f.get("y", 0)))) for f in ego_frames]
        remaining_actors: List[Dict[str, object]] = []
        for act_tr in actor_tracks:
            act_frames = act_tr.get("frames", [])
            if not act_frames:
                remaining_actors.append(act_tr)
                continue
            # Compare common frame range
            min_len = min(len(ego_xy), len(act_frames))
            if min_len < 5:
                remaining_actors.append(act_tr)
                continue
            dists: List[float] = []
            for fi in range(min_len):
                ex, ey = ego_xy[fi]
                ax = float(act_frames[fi].get("mx", act_frames[fi].get("x", 0)))
                ay = float(act_frames[fi].get("my", act_frames[fi].get("y", 0)))
                dists.append(math.hypot(ex - ax, ey - ay))
            dists.sort()
            median_d = dists[len(dists) // 2]
            p90_d = dists[int(len(dists) * 0.9)]
            if median_d < 2.0 and p90_d < 4.0:
                ego_actor_dedup_removed.append(str(act_tr.get("id", "")))
            else:
                remaining_actors.append(act_tr)
        actor_tracks = remaining_actors
    if ego_actor_dedup_removed:
        print(f"[INFO] Ego-actor dedup: removed {len(ego_actor_dedup_removed)} actor tracks: {ego_actor_dedup_removed}")

    print(f"[INFO] Preparing lane output ({len(selected_map.lanes)} lanes)...")
    lane_type_counts: Dict[str, int] = {}
    lanes_out: List[Dict[str, object]] = []
    for lane in selected_map.lanes:
        lane_type_counts[lane.lane_type] = lane_type_counts.get(lane.lane_type, 0) + 1
        lx, ly = lane.label_xy
        lanes_out.append(
            {
                "index": int(lane.index),
                "uid": lane.uid,
                "road_id": int(lane.road_id),
                "lane_id": int(lane.lane_id),
                "lane_type": lane.lane_type,
                "entry_lanes": list(lane.entry_lanes),
                "exit_lanes": list(lane.exit_lanes),
                "label_x": float(lx),
                "label_y": float(ly),
                "polyline": _downsample_line(lane.polyline, max_points=map_max_points_per_line),
                "boundary": _downsample_line(lane.boundary, max_points=map_max_points_per_line),
            }
        )

    print(f"[INFO] Preparing CARLA map layer output...")
    carla_map_out: Optional[Dict[str, object]] = None
    if isinstance(carla_map_layer, dict) and carla_map_layer:
        raw_lines = carla_map_layer.get("lines")
        out_lines: List[List[List[float]]] = []
        if isinstance(raw_lines, list):
            for ln in raw_lines:
                if not isinstance(ln, (list, tuple)):
                    continue
                pts: List[Tuple[float, float]] = []
                for p in ln:
                    if not isinstance(p, (list, tuple)) or len(p) < 2:
                        continue
                    x = _safe_float(p[0], float("nan"))
                    y = _safe_float(p[1], float("nan"))
                    if math.isfinite(x) and math.isfinite(y):
                        pts.append((float(x), float(y)))
                ds = _downsample_xy_line(pts, max_points=map_max_points_per_line)
                if len(ds) >= 2:
                    out_lines.append(ds)
        if out_lines:
            bbox_raw = carla_map_layer.get("bbox")
            bbox_dict = {"min_x": 0.0, "max_x": 1.0, "min_y": 0.0, "max_y": 1.0}
            if isinstance(bbox_raw, dict):
                bbox_dict = {
                    "min_x": _safe_float(bbox_raw.get("min_x"), 0.0),
                    "max_x": _safe_float(bbox_raw.get("max_x"), 1.0),
                    "min_y": _safe_float(bbox_raw.get("min_y"), 0.0),
                    "max_y": _safe_float(bbox_raw.get("max_y"), 1.0),
                }
            carla_map_out = {
                "name": str(carla_map_layer.get("name") or "carla_map"),
                "source_path": str(carla_map_layer.get("source_path") or ""),
                "bbox": bbox_dict,
                "line_count": int(len(out_lines)),
                "lines": out_lines,
                "transform": dict(carla_map_layer.get("transform") or {}),
            }
            # Attach the top-down image underlay if provided
            img_b64 = carla_map_layer.get("image_b64")
            img_bounds = carla_map_layer.get("image_bounds")
            if img_b64 and isinstance(img_bounds, dict):
                carla_map_out["image_b64"] = str(img_b64)
                carla_map_out["image_bounds"] = dict(img_bounds)

    print(f"[INFO] Collecting timeline & computing display bbox...")
    timeline = _collect_timeline(ego_tracks, actor_tracks)
    all_display_points: List[Tuple[float, float]] = []
    for lane in lanes_out:
        for p in lane["polyline"]:
            all_display_points.append((float(p[0]), float(p[1])))
    if carla_map_out is not None:
        for ln in carla_map_out.get("lines", []):
            for p in ln:
                all_display_points.append((float(p[0]), float(p[1])))
    for track in ego_tracks + actor_tracks:
        for fr in track["frames"]:
            all_display_points.append((float(fr["x"]), float(fr["y"])))
    points_arr = np.asarray(all_display_points, dtype=np.float64) if all_display_points else np.zeros((0, 2), dtype=np.float64)
    bbox = _compute_bbox_xy(points_arr)
    print(f"[INFO] Export payload assembled: ego_tracks={len(ego_tracks)} actor_tracks={len(actor_tracks)} lanes={len(lanes_out)} timeline_steps={len(timeline)}")

    return {
        "metadata": {
            "scenario_dir": str(scenario_dir),
            "selected_map": selected_map.name,
            "selected_map_path": selected_map.source_path,
            "map_selection_scores": list(selection_details),
            "snap_to_map": bool(snap_to_map),
            "num_ego_tracks": int(len(ego_tracks)),
            "num_actor_tracks": int(len(actor_tracks)),
            "skipped_non_actor_objects": int(skipped_non_actors),
            "timeline_steps": int(len(timeline)),
            "id_merge_stats": merge_stats,
            "timing_policy": dict(timing_optimization.get("timing_policy", {})),
            "timing_optimization": dict(timing_optimization),
            "lane_change_filter": dict(lane_change_cfg or {}),
            "vehicle_lane_policy": {
                "forbidden_lane_types": sorted(forbidden_lane_types),
                "parked_only_lane_types": sorted(parked_only_lane_types),
                "parked_vehicle_cfg": parked_cfg,
            },
            "map_layers": {
                "v2xpnp": True,
                "carla": bool(carla_map_out is not None),
            },
        },
        "map": {
            "name": selected_map.name,
            "source_path": selected_map.source_path,
            "bbox": {
                "min_x": float(selected_map.bbox[0]),
                "max_x": float(selected_map.bbox[1]),
                "min_y": float(selected_map.bbox[2]),
                "max_y": float(selected_map.bbox[3]),
            },
            "lane_type_counts": lane_type_counts,
            "lanes": lanes_out,
        },
        "carla_map": carla_map_out,
        "view_bbox": {
            "min_x": float(bbox[0]),
            "max_x": float(bbox[1]),
            "min_y": float(bbox[2]),
            "max_y": float(bbox[3]),
        },
        "timeline": [float(t) for t in timeline],
        "ego_tracks": ego_tracks,
        "actor_tracks": actor_tracks,
    }


def _build_html(dataset: Dict[str, object]) -> str:
    dataset_json = json.dumps(_sanitize_for_json(dataset), ensure_ascii=True, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>YAML to Map Replay</title>
  <style>
    :root {{
      --bg: #0d1a24;
      --panel: #142635;
      --panel2: #193346;
      --border: #2a4a60;
      --text: #e4edf3;
      --muted: #a8bccb;
      --accent: #4ec4ff;
      --ok: #74f4b3;
      --warn: #ffc26f;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; height: 100%; background: radial-gradient(circle at 15% 10%, #24455d 0%, #0f1f2a 42%, #09131b 100%); color: var(--text); font-family: "IBM Plex Sans", "Segoe UI", sans-serif; }}
    #app {{ height: 100%; display: grid; grid-template-columns: minmax(760px, 1fr) 390px; gap: 10px; padding: 10px; }}
    #main {{ position: relative; border: 1px solid var(--border); border-radius: 12px; overflow: hidden; background: #0b1822; }}
    #canvas {{ width: 100%; height: 100%; display: block; }}
    #hud {{ position: absolute; left: 12px; top: 12px; background: rgba(7, 16, 24, 0.78); border: 1px solid #365a72; border-radius: 10px; padding: 8px 10px; min-width: 290px; font-size: 12px; line-height: 1.35; }}
    #hud .line {{ margin: 2px 0; }}
    #sidebar {{ border: 1px solid var(--border); border-radius: 12px; background: linear-gradient(180deg, var(--panel) 0%, var(--panel2) 100%); display: flex; flex-direction: column; min-height: 0; }}
    #sidebarInner {{ padding: 12px; overflow: auto; }}
    h2 {{ margin: 0 0 10px 0; font-size: 18px; }}
    h3 {{ margin: 12px 0 8px 0; font-size: 13px; color: var(--ok); letter-spacing: 0.2px; }}
    .section {{ border: 1px solid #29465c; border-radius: 10px; padding: 10px; margin-bottom: 10px; background: rgba(9, 18, 26, 0.34); }}
    .small {{ font-size: 12px; color: var(--muted); }}
    .row {{ display: flex; gap: 8px; align-items: center; margin-bottom: 8px; }}
    .row.wrap {{ flex-wrap: wrap; }}
    button, input[type="range"], select {{ width: 100%; }}
    button {{ min-height: 34px; border: 1px solid #365d75; border-radius: 8px; background: linear-gradient(180deg, #2e5068 0%, #253f53 100%); color: var(--text); cursor: pointer; }}
    select {{ min-height: 34px; border: 1px solid #365d75; border-radius: 8px; background: #0f2433; color: var(--text); padding: 4px 8px; }}
    button:hover {{ border-color: #4e89a9; }}
    button:disabled {{ opacity: 0.45; cursor: default; }}
    .btnPrimary {{ background: linear-gradient(180deg, #3ca9df 0%, #2f83b0 100%); color: #08131a; font-weight: 700; }}
    .badge {{ display: inline-block; padding: 2px 6px; border-radius: 999px; border: 1px solid #50758d; font-size: 10px; margin-right: 6px; }}
    .legendItem {{ display: flex; align-items: center; gap: 8px; font-size: 12px; margin-bottom: 5px; }}
    .legendSwatch {{ width: 16px; height: 10px; border-radius: 3px; border: 1px solid #203645; }}
    .mono {{ font-family: "IBM Plex Mono", "Consolas", monospace; font-size: 11px; }}
    #actorList {{ max-height: 240px; overflow: auto; border: 1px solid #2f4e63; border-radius: 8px; padding: 6px; background: rgba(7, 14, 21, 0.45); }}
    .actorRow {{ display: grid; grid-template-columns: 16px 1fr; gap: 8px; align-items: center; margin-bottom: 5px; }}
    .actorDot {{ width: 10px; height: 10px; border-radius: 50%; }}
    #hoverInfo {{ white-space: pre-wrap; font-size: 12px; line-height: 1.35; }}
    label.cb {{ display: flex; align-items: center; gap: 6px; font-size: 12px; color: #d6e5ef; }}
  </style>
</head>
<body>
  <div id="app">
    <div id="main">
      <canvas id="canvas"></canvas>
      <div id="hud">
        <div class="line" id="hudScenario">Scenario: -</div>
        <div class="line" id="hudMap">Map: -</div>
        <div class="line" id="hudTime">Time: -</div>
        <div class="line" id="hudCounts">Actors: -</div>
        <div class="line" id="hudView">View: -</div>
      </div>
    </div>
    <aside id="sidebar">
      <div id="sidebarInner">
        <h2>YAML to Map Replay</h2>

        <div class="section">
          <h3>Timeline</h3>
          <div class="row">
            <button id="prevBtn">Prev</button>
            <button id="playBtn" class="btnPrimary">Play</button>
            <button id="nextBtn">Next</button>
          </div>
          <div class="row">
            <input id="timeSlider" type="range" min="0" max="0" value="0" step="1" />
          </div>
          <div class="small mono" id="timelineLabel">t=0.000</div>
        </div>

        <div class="section">
          <h3>Display</h3>
          <div class="row wrap">
            <label class="cb"><input id="snapToggle" type="checkbox" checked />Show CARLA replay/export path (when available)</label>
            <label class="cb"><input id="showLabelsToggle" type="checkbox" checked />Show lane labels</label>
            <label class="cb"><input id="showBoundaryToggle" type="checkbox" />Show lane boundaries</label>
            <label class="cb"><input id="showTrajToggle" type="checkbox" checked />Show full trajectories</label>
            <label class="cb"><input id="showImageToggle" type="checkbox" checked />Show map image underlay</label>
          </div>
          <div id="snapHint" class="small mono">
            Checked: replay actors use exported CARLA path; walkers always stay on raw logged path.
          </div>
          <div class="row wrap">
            <label class="cb"><input id="primContinuityToggle" type="checkbox" checked />Continuity primitives (gap/ABA/direction)</label>
            <label class="cb"><input id="primIntersectionToggle" type="checkbox" checked />Intersection turn smoothing</label>
            <label class="cb"><input id="primLanechangeToggle" type="checkbox" checked />Lane-change S-curve smoothing</label>
          </div>
          <div id="primitiveHint" class="small mono">
            Primitive toggles affect visualization path stage only; replay/export remains all-on by default.
          </div>
          <div class="row">
            <label class="small" style="min-width: 92px;">Image opacity</label>
            <input id="imageOpacitySlider" type="range" min="0" max="100" value="55" step="1" />
          </div>
          <div class="row">
            <label class="small" style="min-width: 92px;">Map Layer</label>
            <select id="mapLayerSelect">
              <option value="v2xpnp">V2XPNP Map</option>
              <option value="carla">CARLA Westwood Map</option>
            </select>
          </div>
          <div class="row">
            <button id="fitBtn">Fit View</button>
          </div>
        </div>

        <div class="section">
          <h3>Map Selection</h3>
          <div class="small mono" id="mapScoreText"></div>
        </div>

        <div class="section">
          <h3>Lane Types (V2XPNP)</h3>
          <div id="laneLegend"></div>
        </div>

        <div class="section">
          <h3>Actors</h3>
          <div id="actorLegend"></div>
          <div id="actorList"></div>
        </div>

        <div class="section">
          <h3>Hover</h3>
          <div id="hoverInfo" class="mono">Move cursor over an actor marker.</div>
        </div>
      </div>
    </aside>
  </div>

  <script id="dataset" type="application/json">{dataset_json}</script>
  <script>
  (() => {{
    'use strict';

    const DATA = JSON.parse(document.getElementById('dataset').textContent);

    const laneTypePalette = {{
      '1': '#4e79a7',
      '2': '#f28e2b',
      '3': '#59a14f',
      '4': '#e15759',
      'unknown': '#9aa6af',
    }};
    const actorRolePalette = {{
      ego: '#f8c65f',
      vehicle: '#6bc6ff',
      walker: '#7df0a8',
      cyclist: '#c3a4ff',
      static: '#c9ced2',
      npc: '#6bc6ff',
    }};

    const canvas = document.getElementById('canvas');
    const hudScenario = document.getElementById('hudScenario');
    const hudMap = document.getElementById('hudMap');
    const hudTime = document.getElementById('hudTime');
    const hudCounts = document.getElementById('hudCounts');
    const hudView = document.getElementById('hudView');
    const slider = document.getElementById('timeSlider');
    const playBtn = document.getElementById('playBtn');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const fitBtn = document.getElementById('fitBtn');
    const snapToggle = document.getElementById('snapToggle');
    const snapHint = document.getElementById('snapHint');
    const primContinuityToggle = document.getElementById('primContinuityToggle');
    const primIntersectionToggle = document.getElementById('primIntersectionToggle');
    const primLanechangeToggle = document.getElementById('primLanechangeToggle');
    const primitiveHint = document.getElementById('primitiveHint');
    const showLabelsToggle = document.getElementById('showLabelsToggle');
    const showBoundaryToggle = document.getElementById('showBoundaryToggle');
    const showTrajToggle = document.getElementById('showTrajToggle');
    const showImageToggle = document.getElementById('showImageToggle');
    const imageOpacitySlider = document.getElementById('imageOpacitySlider');
    const mapLayerSelect = document.getElementById('mapLayerSelect');
    const timelineLabel = document.getElementById('timelineLabel');
    const mapScoreText = document.getElementById('mapScoreText');
    const laneLegend = document.getElementById('laneLegend');
    const actorLegend = document.getElementById('actorLegend');
    const actorList = document.getElementById('actorList');
    const hoverInfo = document.getElementById('hoverInfo');

    const timeline = Array.isArray(DATA.timeline) ? DATA.timeline : [0.0];
    slider.max = String(Math.max(0, timeline.length - 1));
    slider.value = '0';
    const hasCarlaLayer = !!(DATA.carla_map && Array.isArray(DATA.carla_map.lines) && DATA.carla_map.lines.length > 0);
    if (!hasCarlaLayer && mapLayerSelect) {{
      mapLayerSelect.value = 'v2xpnp';
      mapLayerSelect.disabled = true;
    }}

    // --- Preload CARLA top-down image underlay ---
    let carlaMapImage = null;
    const hasCarlaImage = !!(DATA.carla_map && DATA.carla_map.image_b64 && DATA.carla_map.image_bounds);
    if (hasCarlaImage) {{
      carlaMapImage = new Image();
      carlaMapImage.src = 'data:image/jpeg;base64,' + DATA.carla_map.image_b64;
      carlaMapImage.onload = () => {{ render(); }};
    }}
    if (!hasCarlaImage && showImageToggle) {{
      showImageToggle.checked = false;
      showImageToggle.disabled = true;
    }}

    const state = {{
      tIndex: 0,
      playing: false,
      playHandle: null,
      lastLayer: 'v2xpnp',
      view: {{
        cx: 0.0,
        cy: 0.0,
        scale: 3.0,
        minScale: 0.1,
        maxScale: 260.0,
      }},
      drag: null,
      hovered: null,
    }};

    function clamp(v, lo, hi) {{
      return Math.max(lo, Math.min(hi, v));
    }}

    function normalizeDeg(x) {{
      let a = Number(x) || 0;
      while (a > 180) a -= 360;
      while (a <= -180) a += 360;
      return a;
    }}

    function lerpAngleDeg(a, b, t) {{
      const aa = normalizeDeg(a);
      const bb = normalizeDeg(b);
      const d = normalizeDeg(bb - aa);
      return normalizeDeg(aa + d * t);
    }}

    function getCanvasCssSize() {{
      const rect = canvas.getBoundingClientRect();
      return {{ width: rect.width, height: rect.height }};
    }}

    function resizeCanvas() {{
      const dpr = window.devicePixelRatio || 1;
      const size = getCanvasCssSize();
      const w = Math.max(2, Math.round(size.width * dpr));
      const h = Math.max(2, Math.round(size.height * dpr));
      if (canvas.width !== w || canvas.height !== h) {{
        canvas.width = w;
        canvas.height = h;
      }}
    }}

    function worldToScreen(x, y) {{
      const size = getCanvasCssSize();
      return {{
        x: size.width * 0.5 + (x - state.view.cx) * state.view.scale,
        y: size.height * 0.5 - (y - state.view.cy) * state.view.scale,
      }};
    }}

    function screenToWorld(sx, sy) {{
      const size = getCanvasCssSize();
      return {{
        x: state.view.cx + (sx - size.width * 0.5) / state.view.scale,
        y: state.view.cy - (sy - size.height * 0.5) / state.view.scale,
      }};
    }}

    function eventScreenPos(evt) {{
      const rect = canvas.getBoundingClientRect();
      return {{ x: evt.clientX - rect.left, y: evt.clientY - rect.top }};
    }}

    function timeNow() {{
      return timeline[clamp(state.tIndex, 0, timeline.length - 1)] || 0.0;
    }}

    function findFramePair(frames, t) {{
      if (!Array.isArray(frames) || frames.length === 0) return null;
      if (t < frames[0].t || t > frames[frames.length - 1].t) return null;
      let lo = 0;
      let hi = frames.length - 1;
      while (lo <= hi) {{
        const mid = (lo + hi) >> 1;
        const mt = frames[mid].t;
        if (mt < t) {{
          lo = mid + 1;
        }} else if (mt > t) {{
          hi = mid - 1;
        }} else {{
          return {{ i0: mid, i1: mid, alpha: 0.0 }};
        }}
      }}
      const i1 = clamp(lo, 0, frames.length - 1);
      const i0 = clamp(i1 - 1, 0, frames.length - 1);
      const t0 = frames[i0].t;
      const t1 = frames[i1].t;
      const alpha = (t1 > t0) ? ((t - t0) / (t1 - t0)) : 0.0;
      return {{ i0, i1, alpha }};
    }}

    function sampleTrackPose(track, t, useMapSnap, activeLayer, primitiveStage) {{
      const stage = String(primitiveStage || 'final');
      const useExec = shouldUseExecFrames(track, useMapSnap, stage);
      const frames = useExec ? (track.carla_exec_frames || []) : (track.frames || []);
      const pair = findFramePair(frames, t);
      if (!pair) return null;
      const f0 = frames[pair.i0];
      const f1 = frames[pair.i1];
      const a = pair.alpha;

      if (useExec) {{
        const ex0 = Number(f0.x);
        const ey0 = Number(f0.y);
        const ez0 = Number(f0.z);
        const eyaw0 = Number(f0.yaw);
        if (pair.i0 === pair.i1) {{
          return {{
            x: ex0, y: ey0, z: ez0, yaw: eyaw0,
            laneIndex: -1,
            laneDist: Number.NaN,
            laneSource: String(track.carla_path_source || 'carla_export'),
            laneQuality: 'exported',
            v2LaneIndex: Number(f0.li),
            carlaLineIndex: Number(f0.cli),
            frame: f0,
          }};
        }}
        const ex1 = Number(f1.x);
        const ey1 = Number(f1.y);
        const ez1 = Number(f1.z);
        const eyaw1 = Number(f1.yaw);
        return {{
          x: ex0 + (ex1 - ex0) * a,
          y: ey0 + (ey1 - ey0) * a,
          z: ez0 + (ez1 - ez0) * a,
          yaw: lerpAngleDeg(eyaw0, eyaw1, a),
          laneIndex: -1,
          laneDist: Number.NaN,
          laneSource: String(track.carla_path_source || 'carla_export'),
          laneQuality: 'exported',
          v2LaneIndex: Number.NaN,
          carlaLineIndex: Number.NaN,
          frame: (a < 0.5) ? f0 : f1,
        }};
      }}

      const forceRaw = isWalkerLikeTrack(track);
      const useMapPath = !!(useMapSnap && !forceRaw);
      const useCarlaSnap = !!(useMapPath && activeLayer === 'carla');
      const cx0 = carlaStageField(f0, 'cx', stage);
      const cy0 = carlaStageField(f0, 'cy', stage);
      const cyaw0 = carlaStageField(f0, 'cyaw', stage);
      const ccli0 = carlaStageField(f0, 'cli', stage);
      const ccld0 = carlaStageField(f0, 'cld', stage);
      const px0 = useCarlaSnap ? (Number.isFinite(cx0) ? cx0 : Number(f0.mx)) : (useMapPath ? Number(f0.mx) : Number(f0.rx));
      const py0 = useCarlaSnap ? (Number.isFinite(cy0) ? cy0 : Number(f0.my)) : (useMapPath ? Number(f0.my) : Number(f0.ry));
      const pz0 = useMapPath ? Number(f0.mz) : Number(f0.rz);
      const pyaw0 = useCarlaSnap ? (Number.isFinite(cyaw0) ? cyaw0 : Number(f0.myaw)) : (useMapPath ? Number(f0.myaw) : Number(f0.ryaw));
      if (pair.i0 === pair.i1) {{
        return {{
          x: px0, y: py0, z: pz0, yaw: pyaw0,
          laneIndex: useCarlaSnap ? (Number.isFinite(ccli0) ? ccli0 : Number(f0.cli)) : Number(f0.li),
          laneDist: useCarlaSnap ? (Number.isFinite(ccld0) ? ccld0 : Number(f0.cld)) : Number(f0.ld),
          laneSource: useCarlaSnap ? carlaStageText(f0, 'csource', stage, String(f0.csource || '')) : 'v2_lane',
          laneQuality: useCarlaSnap ? carlaStageText(f0, 'cquality', stage, String(f0.cquality || '')) : '',
          v2LaneIndex: Number(f0.li),
          carlaLineIndex: useCarlaSnap ? (Number.isFinite(ccli0) ? ccli0 : Number(f0.cli)) : Number(f0.cli),
          frame: f0,
        }};
      }}
      const cx1 = carlaStageField(f1, 'cx', stage);
      const cy1 = carlaStageField(f1, 'cy', stage);
      const cyaw1 = carlaStageField(f1, 'cyaw', stage);
      const px1 = useCarlaSnap ? (Number.isFinite(cx1) ? cx1 : Number(f1.mx)) : (useMapPath ? Number(f1.mx) : Number(f1.rx));
      const py1 = useCarlaSnap ? (Number.isFinite(cy1) ? cy1 : Number(f1.my)) : (useMapPath ? Number(f1.my) : Number(f1.ry));
      const pz1 = useMapPath ? Number(f1.mz) : Number(f1.rz);
      const pyaw1 = useCarlaSnap ? (Number.isFinite(cyaw1) ? cyaw1 : Number(f1.myaw)) : (useMapPath ? Number(f1.myaw) : Number(f1.ryaw));
      const laneFrame = (a < 0.5) ? f0 : f1;
      const laneCli = useCarlaSnap ? carlaStageField(laneFrame, 'cli', stage) : Number(laneFrame.li);
      const laneCld = useCarlaSnap ? carlaStageField(laneFrame, 'cld', stage) : Number(laneFrame.ld);
      return {{
        x: px0 + (px1 - px0) * a,
        y: py0 + (py1 - py0) * a,
        z: pz0 + (pz1 - pz0) * a,
        yaw: lerpAngleDeg(pyaw0, pyaw1, a),
        laneIndex: useCarlaSnap ? (Number.isFinite(laneCli) ? laneCli : Number(laneFrame.cli)) : Number(laneFrame.li),
        laneDist: useCarlaSnap ? (Number.isFinite(laneCld) ? laneCld : Number(laneFrame.cld)) : Number(laneFrame.ld),
        laneSource: useCarlaSnap ? carlaStageText(laneFrame, 'csource', stage, String(laneFrame.csource || '')) : 'v2_lane',
        laneQuality: useCarlaSnap ? carlaStageText(laneFrame, 'cquality', stage, String(laneFrame.cquality || '')) : '',
        v2LaneIndex: Number(laneFrame.li),
        carlaLineIndex: useCarlaSnap ? (Number.isFinite(laneCli) ? laneCli : Number(laneFrame.cli)) : Number(laneFrame.cli),
        frame: laneFrame,
      }};
    }}

    function laneColor(laneType) {{
      const key = String(laneType);
      return laneTypePalette[key] || '#8f9aa2';
    }}

    function actorColor(role) {{
      const key = String(role || 'npc').toLowerCase();
      return actorRolePalette[key] || actorRolePalette.npc;
    }}

    function isWalkerLikeTrack(track) {{
      const role = String(track?.role || track?.obj_type || '').toLowerCase();
      return role === 'walker' || role === 'pedestrian' || role === 'cyclist' || role === 'bicycle';
    }}

    function isReplayTrack(track) {{
      return String(track?.carla_control_mode || '').toLowerCase() === 'replay';
    }}

    function hasExecFrames(track) {{
      return Array.isArray(track?.carla_exec_frames) && track.carla_exec_frames.length > 0;
    }}

    function getPrimitiveStage() {{
      const useCont = !!(primContinuityToggle ? primContinuityToggle.checked : true);
      const useInter = !!(primIntersectionToggle ? primIntersectionToggle.checked : true);
      const useLc = !!(primLanechangeToggle ? primLanechangeToggle.checked : true);
      if (!useCont) return 'base';
      if (!useInter) return 'cont';
      if (!useLc) return 'turn';
      return 'final';
    }}

    function carlaStageField(fr, baseKey, stage) {{
      if (!fr || typeof fr !== 'object') return Number.NaN;
      if (stage === 'base') return Number(fr[`${{baseKey}}_base`]);
      if (stage === 'cont') return Number(fr[`${{baseKey}}_cont`]);
      if (stage === 'turn') return Number(fr[`${{baseKey}}_turn`]);
      return Number(fr[baseKey]);
    }}

    function carlaStageText(fr, baseKey, stage, fallback) {{
      if (!fr || typeof fr !== 'object') return String(fallback || '');
      if (stage === 'base') return String(fr[`${{baseKey}}_base`] ?? fallback ?? '');
      if (stage === 'cont') return String(fr[`${{baseKey}}_cont`] ?? fallback ?? '');
      if (stage === 'turn') return String(fr[`${{baseKey}}_turn`] ?? fallback ?? '');
      return String(fr[baseKey] ?? fallback ?? '');
    }}

    function shouldUseExecFrames(track, useMapSnap, stage) {{
      if (!useMapSnap) return false;
      if (String(stage || 'final') !== 'final') return false;
      if (!isReplayTrack(track)) return false;
      if (isWalkerLikeTrack(track)) return false;
      return hasExecFrames(track);
    }}

    function getActiveMapLayer() {{
      if (!mapLayerSelect) return 'v2xpnp';
      const v = String(mapLayerSelect.value || 'v2xpnp').toLowerCase();
      if (v === 'carla' && hasCarlaLayer) return 'carla';
      return 'v2xpnp';
    }}

    function getLayerBBox(layer) {{
      if (layer === 'carla') {{
        const b = DATA.carla_map?.bbox;
        if (b) return b;
      }}
      const b0 = DATA.map?.bbox;
      if (b0) return b0;
      return DATA.view_bbox || null;
    }}

    function getActiveLayerBBox() {{
      return getLayerBBox(getActiveMapLayer());
    }}

    function fitScaleForBBox(b) {{
      if (!b) return state.view.scale;
      const minX = Number(b.min_x);
      const maxX = Number(b.max_x);
      const minY = Number(b.min_y);
      const maxY = Number(b.max_y);
      if (!Number.isFinite(minX) || !Number.isFinite(maxX) || !Number.isFinite(minY) || !Number.isFinite(maxY)) {{
        return state.view.scale;
      }}
      const size = getCanvasCssSize();
      const rangeX = Math.max(1.0, maxX - minX);
      const rangeY = Math.max(1.0, maxY - minY);
      const pad = 0.90;
      const sx = (size.width * pad) / rangeX;
      const sy = (size.height * pad) / rangeY;
      return clamp(Math.min(sx, sy), state.view.minScale, state.view.maxScale);
    }}

    function fitView() {{
      const b = getActiveLayerBBox();
      if (!b) return;
      const minX = Number(b.min_x);
      const maxX = Number(b.max_x);
      const minY = Number(b.min_y);
      const maxY = Number(b.max_y);
      if (!Number.isFinite(minX) || !Number.isFinite(maxX) || !Number.isFinite(minY) || !Number.isFinite(maxY)) return;
      state.view.scale = fitScaleForBBox(b);
      state.view.cx = 0.5 * (minX + maxX);
      state.view.cy = 0.5 * (minY + maxY);
    }}

    function remapViewForLayerSwitch(prevLayer, nextLayer) {{
      const oldB = getLayerBBox(prevLayer);
      const newB = getLayerBBox(nextLayer);
      if (!oldB || !newB) return;
      const oldMinX = Number(oldB.min_x);
      const oldMaxX = Number(oldB.max_x);
      const oldMinY = Number(oldB.min_y);
      const oldMaxY = Number(oldB.max_y);
      const newMinX = Number(newB.min_x);
      const newMaxX = Number(newB.max_x);
      const newMinY = Number(newB.min_y);
      const newMaxY = Number(newB.max_y);
      if (
        !Number.isFinite(oldMinX) || !Number.isFinite(oldMaxX) || !Number.isFinite(oldMinY) || !Number.isFinite(oldMaxY) ||
        !Number.isFinite(newMinX) || !Number.isFinite(newMaxX) || !Number.isFinite(newMinY) || !Number.isFinite(newMaxY)
      ) {{
        return;
      }}
      const oldRangeX = Math.max(1.0, oldMaxX - oldMinX);
      const oldRangeY = Math.max(1.0, oldMaxY - oldMinY);
      const newRangeX = Math.max(1.0, newMaxX - newMinX);
      const newRangeY = Math.max(1.0, newMaxY - newMinY);

      const oldFit = fitScaleForBBox(oldB);
      const newFit = fitScaleForBBox(newB);
      const zoomFactor = (oldFit > 1e-9) ? (state.view.scale / oldFit) : 1.0;

      const nx = clamp((state.view.cx - oldMinX) / oldRangeX, 0.0, 1.0);
      const ny = clamp((state.view.cy - oldMinY) / oldRangeY, 0.0, 1.0);
      state.view.cx = newMinX + nx * newRangeX;
      state.view.cy = newMinY + ny * newRangeY;
      state.view.scale = clamp(newFit * zoomFactor, state.view.minScale, state.view.maxScale);
    }}

    function drawLaneLine(ctx, points, color, widthPx, alpha) {{
      if (!Array.isArray(points) || points.length < 2) return;
      ctx.beginPath();
      for (let i = 0; i < points.length; i += 1) {{
        const p = points[i];
        const px = Array.isArray(p) ? Number(p[0]) : Number(p?.x);
        const py = Array.isArray(p) ? Number(p[1]) : Number(p?.y);
        const s = worldToScreen(px, py);
        if (i === 0) ctx.moveTo(s.x, s.y);
        else ctx.lineTo(s.x, s.y);
      }}
      ctx.strokeStyle = color;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = widthPx;
      ctx.stroke();
      ctx.globalAlpha = 1.0;
    }}

    function drawActorShape(ctx, pose, role, color) {{
      const s = worldToScreen(pose.x, pose.y);
      const yaw = (Number(pose.yaw) || 0.0) * Math.PI / 180.0;
      const base = clamp(state.view.scale * 0.9, 4.0, 15.0);
      ctx.save();
      ctx.translate(s.x, s.y);
      ctx.rotate(-yaw);
      ctx.fillStyle = color;
      ctx.strokeStyle = '#09131a';
      ctx.lineWidth = 1.2;
      if (role === 'ego') {{
        ctx.beginPath();
        ctx.moveTo(base * 1.3, 0);
        ctx.lineTo(-base * 0.8, base * 0.85);
        ctx.lineTo(-base * 0.8, -base * 0.85);
        ctx.closePath();
      }} else if (role === 'vehicle' || role === 'npc') {{
        const w = base * 2.0;
        const h = base * 1.0;
        ctx.beginPath();
        ctx.rect(-w * 0.5, -h * 0.5, w, h);
      }} else if (role === 'cyclist') {{
        ctx.beginPath();
        ctx.moveTo(base * 1.0, 0);
        ctx.lineTo(0, base * 1.0);
        ctx.lineTo(-base * 1.0, 0);
        ctx.lineTo(0, -base * 1.0);
        ctx.closePath();
      }} else if (role === 'walker') {{
        ctx.beginPath();
        ctx.arc(0, 0, base * 0.75, 0, Math.PI * 2);
      }} else {{
        ctx.beginPath();
        ctx.rect(-base * 0.75, -base * 0.75, base * 1.5, base * 1.5);
      }}
      ctx.fill();
      ctx.stroke();
      ctx.restore();
      return s;
    }}

    function render() {{
      resizeCanvas();
      const size = getCanvasCssSize();
      const dpr = window.devicePixelRatio || 1;
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, size.width, size.height);
      ctx.fillStyle = '#0b1822';
      ctx.fillRect(0, 0, size.width, size.height);

      const useMapSnap = !!snapToggle.checked;
      const showLabels = !!showLabelsToggle.checked;
      const showBoundary = !!showBoundaryToggle.checked;
      const showTraj = !!showTrajToggle.checked;
      const primitiveStage = getPrimitiveStage();
      const t = timeNow();
      const activeLayer = getActiveMapLayer();
      const lanes = (DATA.map && Array.isArray(DATA.map.lanes)) ? DATA.map.lanes : [];
      // --- Draw CARLA map image underlay ---
      const showImage = hasCarlaImage && carlaMapImage && carlaMapImage.complete && carlaMapImage.naturalWidth > 0 && !!showImageToggle.checked;
      if (showImage && (activeLayer === 'carla')) {{
        const ib = DATA.carla_map.image_bounds;
        // image_bounds are in V2XPNP coords: min_x, max_x, min_y, max_y
        // Due to flip_y, the raw image top (small CARLA-Y) maps to max_y in V2XPNP
        // worldToScreen maps max_y to screen-top and min_y to screen-bottom
        const topLeft = worldToScreen(Number(ib.min_x), Number(ib.max_y));
        const bottomRight = worldToScreen(Number(ib.max_x), Number(ib.min_y));
        const imgOpacity = Number(imageOpacitySlider.value) / 100.0;
        ctx.globalAlpha = imgOpacity;
        ctx.drawImage(carlaMapImage, topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);
        ctx.globalAlpha = 1.0;
      }}
      if (activeLayer === 'carla' && hasCarlaLayer) {{
        const carlaLines = Array.isArray(DATA.carla_map?.lines) ? DATA.carla_map.lines : [];
        for (const ln of carlaLines) {{
          const poly = Array.isArray(ln?.polyline) ? ln.polyline : ln;
          drawLaneLine(ctx, poly, '#9fb2c4', 1.2, 0.78);
        }}
        if (showLabels) {{
          ctx.font = '10px "IBM Plex Mono", Consolas, monospace';
          ctx.fillStyle = '#d0dde8';
          ctx.globalAlpha = 0.86;
          for (const ln of carlaLines) {{
            const lx = Number(ln?.label_x);
            const ly = Number(ln?.label_y);
            if (!Number.isFinite(lx) || !Number.isFinite(ly)) continue;
            const s = worldToScreen(lx, ly);
            const lbl = String(ln?.label || `c${{Number(ln?.index ?? -1)}}`);
            let txt = lbl;
            if (Array.isArray(ln?.matched_v2_labels) && ln.matched_v2_labels.length > 0) {{
              txt += ` <- ${{ln.matched_v2_labels[0]}}`;
            }}
            ctx.fillText(txt, s.x + 3, s.y - 3);
          }}
          ctx.globalAlpha = 1.0;
        }}
      }} else {{
        for (const lane of lanes) {{
          drawLaneLine(ctx, lane.polyline, laneColor(lane.lane_type), 1.4, 0.78);
        }}
        if (showBoundary) {{
          ctx.setLineDash([4, 4]);
          for (const lane of lanes) {{
            drawLaneLine(ctx, lane.boundary, '#8a8a8a', 1.0, 0.48);
          }}
          ctx.setLineDash([]);
        }}
        if (showLabels) {{
          ctx.font = '10px "IBM Plex Mono", Consolas, monospace';
          ctx.fillStyle = '#d0dde8';
          ctx.globalAlpha = 0.85;
          for (const lane of lanes) {{
            const s = worldToScreen(Number(lane.label_x), Number(lane.label_y));
            let txt = `r${{lane.road_id}} l${{lane.lane_id}} t${{lane.lane_type}}`;
            const cm = lane.carla_match;
            if (cm && Number.isInteger(Number(cm.carla_line_index)) && Number(cm.carla_line_index) >= 0) {{
              txt += ` -> c${{Number(cm.carla_line_index)}} (${{String(cm.quality || '').slice(0,1)}})`;
            }}
            ctx.fillText(txt, s.x + 3, s.y - 3);
          }}
          ctx.globalAlpha = 1.0;
        }}
      }}

      const tracks = []
        .concat(Array.isArray(DATA.ego_tracks) ? DATA.ego_tracks : [])
        .concat(Array.isArray(DATA.actor_tracks) ? DATA.actor_tracks : []);
      if (showTraj) {{
        for (const tr of tracks) {{
          const useExec = shouldUseExecFrames(tr, useMapSnap, primitiveStage);
          const frames = useExec ? (tr.carla_exec_frames || []) : (tr.frames || []);
          if (frames.length < 2) continue;
          const forceRaw = isWalkerLikeTrack(tr);
          const useMapPath = !!(useMapSnap && !forceRaw);
          ctx.beginPath();
          for (let i = 0; i < frames.length; i += 1) {{
            const fr = frames[i];
            const x = useExec
              ? Number(fr.x)
              : ((useMapPath && activeLayer === 'carla')
                ? (() => {{
                    const v = carlaStageField(fr, 'cx', primitiveStage);
                    return Number.isFinite(v) ? v : Number(fr.mx);
                  }})()
                : (useMapPath ? Number(fr.mx) : Number(fr.rx)));
            const y = useExec
              ? Number(fr.y)
              : ((useMapPath && activeLayer === 'carla')
                ? (() => {{
                    const v = carlaStageField(fr, 'cy', primitiveStage);
                    return Number.isFinite(v) ? v : Number(fr.my);
                  }})()
                : (useMapPath ? Number(fr.my) : Number(fr.ry)));
            const s = worldToScreen(x, y);
            if (i === 0) ctx.moveTo(s.x, s.y);
            else ctx.lineTo(s.x, s.y);
          }}
          ctx.strokeStyle = actorColor(tr.role);
          ctx.globalAlpha = 0.24;
          ctx.lineWidth = (tr.role === 'ego') ? 2.2 : 1.0;
          if (useMapSnap && forceRaw) {{
            ctx.setLineDash([5, 4]);
          }}
          ctx.stroke();
          if (useMapSnap && forceRaw) {{
            ctx.setLineDash([]);
          }}
          ctx.globalAlpha = 1.0;
        }}
      }}

      const markers = [];
      for (const tr of tracks) {{
        const pose = sampleTrackPose(tr, t, useMapSnap, activeLayer, primitiveStage);
        if (!pose) continue;
        const color = actorColor(tr.role);
        const screen = drawActorShape(ctx, pose, String(tr.role || 'npc'), color);
        markers.push({{
          track: tr,
          pose,
          sx: screen.x,
          sy: screen.y,
        }});

        ctx.font = '10px "IBM Plex Sans", sans-serif';
        ctx.fillStyle = '#e6eef5';
        ctx.globalAlpha = 0.9;
        ctx.fillText(String(tr.name || tr.id), screen.x + 6, screen.y - 6);
        ctx.globalAlpha = 1.0;
      }}

      const hover = state.hovered;
      if (hover && hover.track && hover.pose) {{
        const s = worldToScreen(hover.pose.x, hover.pose.y);
        ctx.beginPath();
        ctx.arc(s.x, s.y, 9, 0, Math.PI * 2);
        ctx.strokeStyle = '#ffe08a';
        ctx.lineWidth = 2.0;
        ctx.stroke();
      }}

      hudScenario.textContent = `Scenario: ${{DATA.metadata?.scenario_dir || '-'}}`;
      const activeMapName = (activeLayer === 'carla' && hasCarlaLayer)
        ? (DATA.carla_map?.name || 'carla_map')
        : (DATA.map?.name || '-');
      hudMap.textContent = `Map: ${{activeMapName}} (layer=${{activeLayer}})`;
      state.lastLayer = activeLayer;
      hudTime.textContent = `Time: ${{t.toFixed(3)}} s (step ${{state.tIndex + 1}}/${{timeline.length}})`;
      hudCounts.textContent = `Actors: ego=${{(DATA.ego_tracks || []).length}} custom=${{(DATA.actor_tracks || []).length}}`;
      hudView.textContent = `View: scale=${{state.view.scale.toFixed(3)}} px/m center=(${{state.view.cx.toFixed(2)}}, ${{state.view.cy.toFixed(2)}})`;
      timelineLabel.textContent = `t=${{t.toFixed(3)}} sec`;
      if (snapHint) {{
        if (snapToggle.checked) {{
          snapHint.textContent =
            'Checked: replay actors use exported CARLA path; walkers always stay on raw logged path.';
        }} else {{
          snapHint.textContent = 'Unchecked: all actors are shown on raw logged trajectory (rx/ry).';
        }}
      }}
      if (primitiveHint) {{
        primitiveHint.textContent =
          `Primitive stage: ${{primitiveStage}} (visualization only; export/replay remains all-on by default).`;
      }}

      // hover nearest marker
      if (state.mouse) {{
        let best = null;
        let bestD2 = 14 * 14;
        for (const m of markers) {{
          const dx = m.sx - state.mouse.x;
          const dy = m.sy - state.mouse.y;
          const d2 = dx * dx + dy * dy;
          if (d2 < bestD2) {{
            bestD2 = d2;
            best = m;
          }}
        }}
        state.hovered = best;
      }} else {{
        state.hovered = null;
      }}

      if (state.hovered && state.hovered.track) {{
        const tr = state.hovered.track;
        const pose = state.hovered.pose;
        let laneTxt = 'lane: none';
        if (activeLayer === 'carla') {{
          const carlaLines = Array.isArray(DATA.carla_map?.lines) ? DATA.carla_map.lines : [];
          const ci = Number(pose.carlaLineIndex);
          if (Number.isInteger(ci) && ci >= 0) {{
            const cl = carlaLines.find((x) => Number(x?.index) === ci) || null;
            if (cl) {{
              const mappedLbl = (Array.isArray(cl.matched_v2_labels) && cl.matched_v2_labels.length > 0)
                ? cl.matched_v2_labels.join(', ')
                : '-';
              laneTxt =
                `carla_lane: ${{String(cl.label || `c${{ci}}`)}} dist=${{Number(pose.laneDist).toFixed(2)}}m src=${{String(pose.laneSource || '-')}} q=${{String(pose.laneQuality || '-')}}\\n` +
                `mapped_v2: ${{mappedLbl}}\\n` +
                `pose_v2_lane_index=${{Number.isInteger(pose.v2LaneIndex) ? pose.v2LaneIndex : -1}}`;
            }} else {{
              laneTxt = `carla_lane: c${{ci}} dist=${{Number(pose.laneDist).toFixed(2)}}m`;
            }}
          }}
        }} else if (Number.isInteger(pose.laneIndex) && pose.laneIndex >= 0) {{
          const lane = lanes.find((ln) => Number(ln?.index) === Number(pose.laneIndex)) || null;
          if (lane) {{
          const cm = lane.carla_match;
          const mapTxt = (cm && Number.isInteger(Number(cm.carla_line_index)) && Number(cm.carla_line_index) >= 0)
            ? ` -> c${{Number(cm.carla_line_index)}} q=${{String(cm.quality || '-')}}`
            : '';
          laneTxt = `lane: road=${{lane.road_id}} lane=${{lane.lane_id}} type=${{lane.lane_type}} dist=${{Number(pose.laneDist).toFixed(2)}}m${{mapTxt}}`;
          }}
        }}
        hoverInfo.textContent =
          `name: ${{tr.name}}\\n` +
          `id: ${{tr.id}}\\n` +
          `vid: ${{tr.vid ?? '-'}} orig_vid: ${{tr.orig_vid ?? '-'}}\\n` +
          `merged_vids: ${{Array.isArray(tr.merged_vids) ? tr.merged_vids.join(',') : '-'}}\\n` +
          `role: ${{tr.role}}\\n` +
          `obj_type: ${{tr.obj_type || '-'}}\\n` +
          `model: ${{tr.model || '-'}}\\n` +
          `control_mode: ${{tr.carla_control_mode || '-'}} source: ${{tr.carla_path_source || '-'}} walker_raw_forced=${{isWalkerLikeTrack(tr)}}\\n` +
          `x=${{Number(pose.x).toFixed(2)}} y=${{Number(pose.y).toFixed(2)}} z=${{Number(pose.z).toFixed(2)}} yaw=${{Number(pose.yaw).toFixed(1)}}\\n` +
          laneTxt;
      }} else {{
        hoverInfo.textContent = 'Move cursor over an actor marker.';
      }}
    }}

    function setIndex(i) {{
      state.tIndex = clamp(i, 0, timeline.length - 1);
      slider.value = String(state.tIndex);
      render();
    }}

    function togglePlay() {{
      state.playing = !state.playing;
      playBtn.textContent = state.playing ? 'Pause' : 'Play';
      if (state.playing) {{
        const tick = () => {{
          if (!state.playing) return;
          if (state.tIndex >= timeline.length - 1) {{
            state.playing = false;
            playBtn.textContent = 'Play';
            return;
          }}
          setIndex(state.tIndex + 1);
          state.playHandle = window.setTimeout(tick, 60);
        }};
        tick();
      }} else if (state.playHandle) {{
        window.clearTimeout(state.playHandle);
        state.playHandle = null;
      }}
    }}

    slider.addEventListener('input', (evt) => {{
      const next = Number.parseInt(evt.target.value, 10);
      setIndex(Number.isFinite(next) ? next : 0);
    }});
    prevBtn.addEventListener('click', () => setIndex(state.tIndex - 1));
    nextBtn.addEventListener('click', () => setIndex(state.tIndex + 1));
    playBtn.addEventListener('click', togglePlay);
    fitBtn.addEventListener('click', () => {{
      fitView();
      render();
    }});
    snapToggle.addEventListener('change', render);
    if (primContinuityToggle) primContinuityToggle.addEventListener('change', render);
    if (primIntersectionToggle) primIntersectionToggle.addEventListener('change', render);
    if (primLanechangeToggle) primLanechangeToggle.addEventListener('change', render);
    showLabelsToggle.addEventListener('change', render);
    showBoundaryToggle.addEventListener('change', render);
    showTrajToggle.addEventListener('change', render);
    if (showImageToggle) showImageToggle.addEventListener('change', render);
    if (imageOpacitySlider) imageOpacitySlider.addEventListener('input', render);
    if (mapLayerSelect) {{
      mapLayerSelect.addEventListener('change', () => {{
        const prevLayer = String(state.lastLayer || 'v2xpnp');
        const nextLayer = getActiveMapLayer();
        if (prevLayer !== nextLayer) {{
          remapViewForLayerSwitch(prevLayer, nextLayer);
        }}
        state.lastLayer = nextLayer;
        render();
      }});
    }}

    canvas.addEventListener('mousedown', (evt) => {{
      const p = eventScreenPos(evt);
      state.drag = {{ sx: p.x, sy: p.y, cx: state.view.cx, cy: state.view.cy }};
    }});
    window.addEventListener('mouseup', () => {{
      state.drag = null;
    }});
    window.addEventListener('mousemove', (evt) => {{
      const rect = canvas.getBoundingClientRect();
      if (evt.clientX < rect.left || evt.clientX > rect.right || evt.clientY < rect.top || evt.clientY > rect.bottom) {{
        state.mouse = null;
      }} else {{
        state.mouse = {{ x: evt.clientX - rect.left, y: evt.clientY - rect.top }};
      }}
      if (!state.drag) {{
        render();
        return;
      }}
      const p = eventScreenPos(evt);
      const dx = p.x - state.drag.sx;
      const dy = p.y - state.drag.sy;
      state.view.cx = state.drag.cx - dx / state.view.scale;
      state.view.cy = state.drag.cy + dy / state.view.scale;
      render();
    }});
    canvas.addEventListener('wheel', (evt) => {{
      evt.preventDefault();
      const p = eventScreenPos(evt);
      const before = screenToWorld(p.x, p.y);
      const zoom = Math.exp(-evt.deltaY * 0.0012);
      state.view.scale = clamp(state.view.scale * zoom, state.view.minScale, state.view.maxScale);
      const after = screenToWorld(p.x, p.y);
      state.view.cx += before.x - after.x;
      state.view.cy += before.y - after.y;
      render();
    }}, {{ passive: false }});

    window.addEventListener('keydown', (evt) => {{
      if (evt.key === 'ArrowRight') {{
        setIndex(state.tIndex + 1);
      }} else if (evt.key === 'ArrowLeft') {{
        setIndex(state.tIndex - 1);
      }} else if (evt.key === ' ') {{
        evt.preventDefault();
        togglePlay();
      }}
    }});

    function buildSidebar() {{
      const scores = DATA.metadata?.map_selection_scores || [];
      const scoreLines = [];
      for (const s of scores) {{
        scoreLines.push(
          `${{s.name}}\\n` +
          `  score=${{Number(s.score).toFixed(3)}} ` +
          `median=${{Number(s.median_nearest_m).toFixed(2)}}m ` +
          `p90=${{Number(s.p90_nearest_m).toFixed(2)}}m ` +
          `outside=${{(100 * Number(s.outside_bbox_ratio)).toFixed(1)}}%`
        );
      }}
      const mergeStats = DATA.metadata?.id_merge_stats || {{}};
      scoreLines.push(
        'ID merge stats\\n' +
        `  input=${{Number(mergeStats.input_tracks || 0)}} output=${{Number(mergeStats.output_tracks || 0)}} ` +
        `collisions=${{Number(mergeStats.ids_with_collisions || 0)}} ` +
        `merged=${{Number(mergeStats.merged_duplicates || 0)}} split=${{Number(mergeStats.split_tracks_created || 0)}} ` +
        `thr=${{Number(mergeStats.id_merge_distance_m || 0).toFixed(2)}}m`
      );
      scoreLines.push(
        'Cross-ID dedup\\n' +
        `  enabled=${{Boolean(mergeStats.cross_id_dedup_enabled)}} ` +
        `pairs_checked=${{Number(mergeStats.cross_id_pair_checks || 0)}} ` +
        `candidate_pairs=${{Number(mergeStats.cross_id_candidate_pairs || 0)}}\\n` +
        `  clusters=${{Number(mergeStats.cross_id_clusters || 0)}} removed=${{Number(mergeStats.cross_id_removed || 0)}}`
      );
      const tOpt = DATA.metadata?.timing_optimization || {{}};
      const tEarly = tOpt.early_spawn || {{}};
      const tLate = tOpt.late_despawn || {{}};
      scoreLines.push(
        'Timing optimization\\n' +
        `  early: enabled=${{Boolean(tEarly.enabled)}} applied=${{Boolean(tEarly.applied)}} ` +
        `adjusted=${{Array.isArray(tEarly.adjusted_actor_ids) ? tEarly.adjusted_actor_ids.length : 0}}\\n` +
        `  late: enabled=${{Boolean(tLate.enabled)}} applied=${{Boolean(tLate.applied)}} ` +
        `adjusted=${{Array.isArray(tLate.adjusted_actor_ids) ? tLate.adjusted_actor_ids.length : 0}} ` +
        `horizon=${{Number(tLate.hold_until_time || 0).toFixed(2)}}s`
      );
      const laneCfg = DATA.metadata?.lane_change_filter || {{}};
      scoreLines.push(
        'Lane-change filter\\n' +
        `  enabled=${{Boolean(laneCfg.enabled)}} top_k=${{Number(laneCfg.lane_top_k || 0)}} ` +
        `window=${{Number(laneCfg.confirm_window || 0)}} votes=${{Number(laneCfg.confirm_votes || 0)}}\\n` +
        `  cooldown=${{Number(laneCfg.cooldown_frames || 0)}} endpoint_guard=${{Number(laneCfg.endpoint_guard_frames || 0)}}`
      );
      const lc = DATA.metadata?.lane_correspondence || {{}};
      if (Boolean(lc.enabled)) {{
        scoreLines.push(
          'Lane correspondence\\n' +
          `  mapped_lanes=${{Number(lc.mapped_lane_count || 0)}} usable=${{Number(lc.usable_lane_count || 0)}} mapped_carla_lines=${{Number(lc.mapped_carla_line_count || 0)}}\\n` +
          `  quality=${{JSON.stringify(lc.quality_counts || {{}})}}\\n` +
          `  splits=${{Number(lc.split_merge_count || 0)}} phantom_changes=${{Number(lc.total_phantom_lane_changes || 0)}}\\n` +
          `  connectivity=${{Number(lc.connectivity_edges_preserved || 0)}}/${{Number(lc.connectivity_edges_total || 0)}}\\n` +
          `  wrong_way_fixed=${{Number(lc.wrong_way_frames_fixed || 0)}} jump_repairs=${{Number(lc.continuity_jump_repairs || 0)}}\\n` +
          `  legal_transition_fix=${{Number(lc.illegal_lane_transitions_fixed || 0)}}/${{Number(lc.illegal_lane_transitions_detected || 0)}}\\n` +
          `  intersection_runs=${{Number(lc.intersection_turn_runs || 0)}} polyline_locks=${{Number(lc.intersection_turn_polyline_snaps || 0)}} curve_only=${{Number(lc.intersection_turn_curve_only_runs || 0)}}\\n` +
          `  lanechange_scurve=${{Number(lc.lanechange_scurve_events || 0)}}`
        );
      }}
      const carlaExportMeta = DATA.metadata?.carla_route_export || {{}};
      if (Boolean(carlaExportMeta.enabled)) {{
        const srcMap = carlaExportMeta.actor_path_sources || {{}};
        const srcVals = Object.values(srcMap);
        const corrCount = srcVals.filter((x) => String(x?.source || '').toLowerCase().startsWith('corr')).length;
        const rawCount = srcVals.filter((x) => String(x?.source || '').toLowerCase().startsWith('raw')).length;
        scoreLines.push(
          'CARLA route export\\n' +
          `  actor_control=${{String(carlaExportMeta.actor_control_mode || '-')}} ` +
          `walker_control=${{String(carlaExportMeta.walker_control_mode || '-')}}\\n` +
          `  actor_sources: corr=${{corrCount}} raw=${{rawCount}} total=${{srcVals.length}}\\n` +
          '  snap toggle (checked): replay actors use exported route; walkers stay raw.'
        );
      }}
      if (DATA.carla_map && DATA.carla_map.transform) {{
        const ctf = DATA.carla_map.transform;
        scoreLines.push(
          'CARLA layer alignment\\n' +
          `  lines=${{Number(DATA.carla_map.line_count || 0)}} scale=${{Number(ctf.scale || 1).toFixed(3)}} ` +
          `theta=${{Number(ctf.theta_deg || 0).toFixed(3)}}deg\\n` +
          `  tx=${{Number(ctf.tx || 0).toFixed(3)}} ty=${{Number(ctf.ty || 0).toFixed(3)}} flip_y=${{Boolean(ctf.flip_y)}}`
        );
      }}
      mapScoreText.textContent = scoreLines.join('\\n\\n');

      laneLegend.innerHTML = '';
      const laneCounts = (DATA.map && DATA.map.lane_type_counts) ? DATA.map.lane_type_counts : {{}};
      Object.keys(laneCounts).sort((a, b) => Number(a) - Number(b)).forEach((k) => {{
        const row = document.createElement('div');
        row.className = 'legendItem';
        const sw = document.createElement('div');
        sw.className = 'legendSwatch';
        sw.style.background = laneColor(k);
        const txt = document.createElement('div');
        txt.textContent = `type=${{k}} lanes=${{laneCounts[k]}}`;
        row.appendChild(sw);
        row.appendChild(txt);
        laneLegend.appendChild(row);
      }});

      const actorCounts = {{}};
      const tracks = []
        .concat(Array.isArray(DATA.ego_tracks) ? DATA.ego_tracks : [])
        .concat(Array.isArray(DATA.actor_tracks) ? DATA.actor_tracks : []);
      for (const tr of tracks) {{
        const key = String(tr.role || 'npc');
        actorCounts[key] = (actorCounts[key] || 0) + 1;
      }}

      actorLegend.innerHTML = '';
      Object.keys(actorCounts).sort().forEach((k) => {{
        const row = document.createElement('div');
        row.className = 'legendItem';
        const sw = document.createElement('div');
        sw.className = 'legendSwatch';
        sw.style.background = actorColor(k);
        const txt = document.createElement('div');
        txt.textContent = `${{k}}: ${{actorCounts[k]}}`;
        row.appendChild(sw);
        row.appendChild(txt);
        actorLegend.appendChild(row);
      }});

      actorList.innerHTML = '';
      tracks.forEach((tr) => {{
        const row = document.createElement('div');
        row.className = 'actorRow';
        const dot = document.createElement('div');
        dot.className = 'actorDot';
        dot.style.background = actorColor(tr.role);
        const txt = document.createElement('div');
        const hasOrig = Number.isInteger(tr.orig_vid) && Number(tr.orig_vid) !== Number(tr.vid);
        const idPart = hasOrig ? `id=${{tr.vid}} (orig=${{tr.orig_vid}})` : `id=${{tr.vid || tr.id}}`;
        const mergedCount = Array.isArray(tr.merged_vids) ? tr.merged_vids.length : 1;
        const mergePart = mergedCount > 1 ? ` merged=${{mergedCount}}` : '';
        const ctrlMode = String(tr.carla_control_mode || '');
        const srcMode = String(tr.carla_path_source || '');
        const pathPart = ctrlMode ? ` ctrl=${{ctrlMode}} src=${{srcMode || '-'}}` : '';
        // Early spawn indicator
        let spawnPart = '';
        if (tr.has_early_spawn && tr.early_spawn_time != null) {{
          const advanceSec = (tr.first_observed_time - tr.early_spawn_time).toFixed(1);
          spawnPart = ` 🚀${{advanceSec}}s`;
        }}
        txt.textContent = `${{tr.name}} [${{tr.role}}] ${{idPart}} obj=${{tr.obj_type || '-'}}${{mergePart}}${{pathPart}}${{spawnPart}}`;
        // Add tooltip with timing details
        if (tr.first_observed_time != null) {{
          let tooltip = `First observed: ${{tr.first_observed_time.toFixed(2)}}s`;
          if (tr.early_spawn_time != null) {{
            tooltip += `\\nEarly spawn: ${{tr.early_spawn_time.toFixed(2)}}s`;
          }}
          row.title = tooltip;
        }}
        row.appendChild(dot);
        row.appendChild(txt);
        actorList.appendChild(row);
      }});
    }}

    window.addEventListener('resize', () => {{
      render();
    }});

    buildSidebar();
    fitView();
    render();
  }})();
  </script>
</body>
</html>
"""
