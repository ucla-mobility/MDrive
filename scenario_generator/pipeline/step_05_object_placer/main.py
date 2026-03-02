import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import numpy as np
except Exception as e:
    raise RuntimeError("This script requires numpy") from e

from .assets import get_asset_bbox, keyword_filter_assets, load_assets
from .csp import (
    _compute_merge_min_s_by_vehicle,
    build_stage2_constraints_prompt,
    CandidatePlacement,
    expand_group_to_actors,
    solve_weighted_csp_with_extension,
    validate_actor_specs,
)
from .constants import LATERAL_TO_M, LANE_WIDTH_M, SIDEWALK_OFFSET_M
from .filters import _contains_exact_quote, _should_drop_stage1_entity
from .guardrails import (
    apply_after_turn_segment_corrections,
    apply_in_intersection_segment_corrections,
    build_repair_prompt,
    validate_stage2_output,
)
from .geometry import heading_deg_from_vec, point_and_tangent_at_s, right_normal_world, wrap180
from .model import generate_with_model
from .nodes import _override_seg_points_with_picked, build_segments_from_nodes, load_nodes
from .parsing import parse_llm_json
from .prompts import build_stage1_prompt, build_stage2_prompt, build_vehicle_segment_summaries
from .spawn import build_motion_waypoints, compute_spawn_from_anchor, resolve_nodes_path
from .viz import visualize
import math


def run_object_placer(args, model=None, tokenizer=None):
    """
    Main pipeline body, optionally reusing a provided model/tokenizer.
    """
    t_obj_start = time.time()
    stats = getattr(args, "stats", None)
    scenario_spec_payload = getattr(args, "scenario_spec", None)
    allow_static_props = True
    if isinstance(scenario_spec_payload, dict):
        allow_static_props = bool(scenario_spec_payload.get("allow_static_props", True))
    actor_trace: List[Dict[str, Any]] = []

    def _log_actor_trace(step: str, actors: Optional[List[Dict[str, Any]]] = None, note: Optional[str] = None) -> None:
        """
        Append a lightweight trace entry capturing actor-related mutations.
        Stores counts and a small sample of actor ids/categories for debuggability.
        """
        sample = []
        try:
            if actors and isinstance(actors, list):
                for a in actors[:5]:
                    if isinstance(a, dict):
                        sample.append({
                            "id": a.get("id") or a.get("entity_id"),
                            "category": a.get("category") or a.get("actor_kind"),
                            "target_vehicle": (a.get("placement") or {}).get("target_vehicle"),
                            "seg": (a.get("placement") or {}).get("seg_id"),
                        })
        except Exception:
            pass
        actor_trace.append({
            "step": step,
            "elapsed_s": round(time.time() - t_obj_start, 3),
            "actor_count": len(actors) if isinstance(actors, list) else None,
            "note": note,
            "sample": sample,
        })

    def _adjust_triggers_by_paths(
        actors_world: List[Dict[str, Any]],
        picked: List[Dict[str, Any]],
        seg_by_id: Dict[int, np.ndarray],
        min_intersect_dist_m: float = 4.0,
        max_keep_dist_m: float = 25.0,
    ) -> List[Dict[str, Any]]:
        """
        Post-process moving NPCs (walkers/cyclists/vehicles) to align trigger.vehicle
        with the ego path they intersect or are closest to.
        """
        # Build ego polylines
        ego_paths: Dict[str, np.ndarray] = {}
        seg_ids_by_vehicle: Dict[str, List[int]] = {}
        for p in picked:
            vname = p.get("vehicle")
            seg_ids = (p.get("signature", {}) or {}).get("segment_ids", [])
            pts_list = []
            seg_ids_by_vehicle[vname] = []
            for sid in seg_ids or []:
                try:
                    sid_int = int(sid)
                except Exception:
                    continue
                pts = seg_by_id.get(sid_int)
                if pts is None or len(pts) == 0:
                    continue
                seg_ids_by_vehicle[vname].append(sid_int)
                pts_list.append(np.asarray(pts, dtype=float))
            if pts_list:
                ego_paths[vname] = np.vstack(pts_list)

        def _actor_points(a: Dict[str, Any]) -> np.ndarray:
            wps = a.get("world_waypoints")
            if isinstance(wps, list) and wps and isinstance(wps[0], dict):
                pts = [(float(w.get("x", 0.0)), float(w.get("y", 0.0))) for w in wps]
            else:
                sp = a.get("spawn", {}) or {}
                pts = [(float(sp.get("x", 0.0)), float(sp.get("y", 0.0)))]
            return np.asarray(pts, dtype=float)

        adjusted = []
        removed = 0
        for a in actors_world:
            cat = str(a.get("category", "")).lower()
            kind = str(a.get("actor_kind", "")).lower()
            motion = a.get("motion", {}) if isinstance(a.get("motion"), dict) else {}
            mtype = str(motion.get("type", "unknown")).lower()
            is_moving = mtype not in ("static", "unknown")
            if not (cat in ("walker", "cyclist", "vehicle") or kind in ("walker", "cyclist", "npc_vehicle")):
                adjusted.append(a)
                continue
            if not is_moving and cat == "static":
                adjusted.append(a)
                continue
            pts = _actor_points(a)
            if pts.size == 0:
                adjusted.append(a)
                continue

            best_vehicle = None
            best_dist = float("inf")
            reason = "closest"

            # Prefer direct segment overlap
            seg_id = (a.get("resolved") or {}).get("seg_id") or (a.get("placement") or {}).get("seg_id")
            if seg_id is not None:
                try:
                    seg_id = int(seg_id)
                except Exception:
                    seg_id = None
            if seg_id is not None:
                for vname, seg_ids in seg_ids_by_vehicle.items():
                    if seg_id in seg_ids:
                        best_vehicle = vname
                        best_dist = 0.0
                        reason = "same_segment"
                        break

            # Otherwise choose closest ego path
            if best_vehicle is None:
                for vname, path in ego_paths.items():
                    if path is None or len(path) == 0:
                        continue
                    # pairwise min distance
                    dists = np.linalg.norm(pts[:, None, :] - path[None, :, :], axis=2)
                    md = float(np.min(dists))
                    if md < best_dist:
                        best_dist = md
                        best_vehicle = vname

            if best_vehicle is None:
                best_dist = float("inf")

            # If too far, still pick closest but note it
            note = reason
            if best_dist > min_intersect_dist_m and reason != "same_segment":
                note = f"closest_{best_dist:.1f}m"

            if best_dist > max_keep_dist_m:
                removed += 1
                print(f"[INFO] Removing actor {a.get('id','actor')} (no nearby ego path; closest={best_dist:.1f}m)", flush=True)
                continue

            trig = a.get("trigger")
            if trig is None or not isinstance(trig, dict):
                trig = {"type": "distance_to_vehicle", "distance_m": 8.0, "vehicle": best_vehicle, "evidence": ""}
            else:
                trig = dict(trig)
                trig["vehicle"] = best_vehicle
            a_mod = dict(a)
            a_mod["trigger"] = trig
            adjusted.append(a_mod)
            print(f"[INFO] Adjusted trigger for {a.get('id','actor')}: vehicle={best_vehicle} ({note}) (prev trigger may differ)", flush=True)
        if removed:
            print(f"[INFO] Removed {removed} moving actor(s) with no nearby ego path", flush=True)
        return adjusted

    def _extract_ego_spawns_from_picked(picked_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen = set()
        for p in picked_items:
            if not isinstance(p, dict):
                continue
            vehicle_name = str(p.get("vehicle", "")).strip()
            if not vehicle_name or vehicle_name in seen:
                continue
            sig = p.get("signature", {}) if isinstance(p.get("signature"), dict) else {}
            entry = sig.get("entry", {}) if isinstance(sig.get("entry"), dict) else {}
            point = entry.get("point", {}) if isinstance(entry.get("point"), dict) else {}
            x = point.get("x")
            y = point.get("y")
            yaw = entry.get("heading_deg")
            if x is None or y is None:
                segs = sig.get("segments_detailed", [])
                if isinstance(segs, list) and segs:
                    first_seg = segs[0] if isinstance(segs[0], dict) else {}
                    start_obj = first_seg.get("start", {}) if isinstance(first_seg.get("start"), dict) else {}
                    start_point = start_obj.get("point", {}) if isinstance(start_obj.get("point"), dict) else {}
                    x = start_point.get("x")
                    y = start_point.get("y")
                    if yaw is None:
                        yaw = start_obj.get("heading_deg")
            if x is None or y is None:
                continue
            if yaw is None:
                yaw = 0.0
            out.append(
                {
                    "vehicle": vehicle_name,
                    "spawn": {
                        "x": float(x),
                        "y": float(y),
                        "yaw_deg": float(yaw),
                    },
                }
            )
            seen.add(vehicle_name)
        return out

    def _ego_path_point_cloud(picked_items: List[Dict[str, Any]]) -> np.ndarray:
        pts: List[Tuple[float, float]] = []
        for p in picked_items:
            if not isinstance(p, dict):
                continue
            sig = p.get("signature", {}) if isinstance(p.get("signature"), dict) else {}
            segs = sig.get("segments_detailed", [])
            if not isinstance(segs, list):
                continue
            for seg in segs:
                if not isinstance(seg, dict):
                    continue
                sample = seg.get("polyline_sample")
                if not isinstance(sample, list):
                    continue
                for pt in sample:
                    if not isinstance(pt, dict):
                        continue
                    x = pt.get("x")
                    y = pt.get("y")
                    if x is None or y is None:
                        continue
                    pts.append((float(x), float(y)))
        if not pts:
            return np.zeros((0, 2), dtype=float)
        return np.asarray(pts, dtype=float)

    def _ego_lane_samples_from_picked(picked_items: List[Dict[str, Any]]) -> np.ndarray:
        """
        Sample ego lane centerline points with local tangent vectors.
        Returned rows: [x, y, tx, ty].
        """
        rows: List[Tuple[float, float, float, float]] = []
        for p in picked_items:
            if not isinstance(p, dict):
                continue
            sig = p.get("signature", {}) if isinstance(p.get("signature"), dict) else {}
            segs = sig.get("segments_detailed", [])
            if not isinstance(segs, list):
                continue
            for seg in segs:
                if not isinstance(seg, dict):
                    continue
                poly = seg.get("polyline_sample")
                if not isinstance(poly, list) or len(poly) < 2:
                    continue
                pts: List[Tuple[float, float]] = []
                for pt in poly:
                    if not isinstance(pt, dict):
                        continue
                    x = pt.get("x")
                    y = pt.get("y")
                    if x is None or y is None:
                        continue
                    pts.append((float(x), float(y)))
                if len(pts) < 2:
                    continue
                arr = np.asarray(pts, dtype=float)
                for i in range(len(arr)):
                    if i == 0:
                        vec = arr[1] - arr[0]
                    elif i == len(arr) - 1:
                        vec = arr[-1] - arr[-2]
                    else:
                        vec = arr[i + 1] - arr[i - 1]
                    n = float(np.linalg.norm(vec))
                    if n < 1e-6:
                        continue
                    rows.append((float(arr[i, 0]), float(arr[i, 1]), float(vec[0] / n), float(vec[1] / n)))
        if not rows:
            return np.zeros((0, 4), dtype=float)
        return np.asarray(rows, dtype=float)

    def _lane_samples_from_segment_bank(
        segments: List[Dict[str, Any]],
        crop_bounds: Optional[Tuple[float, float, float, float]] = None,
        stride: int = 4,
    ) -> np.ndarray:
        """
        Sample map lane centerline points from full segment bank.
        Returned rows: [x, y, tx, ty].
        """
        rows: List[Tuple[float, float, float, float]] = []
        step = max(1, int(stride))
        xmin = xmax = ymin = ymax = None
        margin = 15.0
        if crop_bounds is not None:
            xmin, xmax, ymin, ymax = crop_bounds
        for seg in segments or []:
            if not isinstance(seg, dict):
                continue
            pts = seg.get("points")
            if pts is None:
                continue
            arr = np.asarray(pts, dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) < 2:
                continue
            if xmin is not None:
                sxmin = float(np.min(arr[:, 0]))
                sxmax = float(np.max(arr[:, 0]))
                symin = float(np.min(arr[:, 1]))
                symax = float(np.max(arr[:, 1]))
                if sxmax < xmin - margin or sxmin > xmax + margin or symax < ymin - margin or symin > ymax + margin:
                    continue
            for i in range(0, len(arr), step):
                x_i = float(arr[i, 0])
                y_i = float(arr[i, 1])
                if xmin is not None:
                    if x_i < xmin - margin or x_i > xmax + margin or y_i < ymin - margin or y_i > ymax + margin:
                        continue
                if i == 0:
                    vec = arr[1] - arr[0]
                elif i == len(arr) - 1:
                    vec = arr[-1] - arr[-2]
                else:
                    vec = arr[i + 1] - arr[i - 1]
                n = float(np.linalg.norm(vec))
                if n < 1e-6:
                    continue
                rows.append((x_i, y_i, float(vec[0] / n), float(vec[1] / n)))
        if not rows:
            return np.zeros((0, 4), dtype=float)
        return np.asarray(rows, dtype=float)

    def _estimate_road_lateral_bounds(
        seg_pts: np.ndarray,
        anchor_s: float,
        lane_samples: np.ndarray,
        along_window_m: float = 20.0,
        max_angle_diff_deg: float = 40.0,
        max_lateral_m: float = 35.0,
        clip_percentile: float = 0.0,
    ) -> Optional[Tuple[float, float]]:
        """
        Estimate local drivable-road lateral bounds around the crossing anchor.
        Returns (road_edge_min_lat, road_edge_max_lat) in anchor-local lateral coordinates,
        where 0 is anchor lane center, + is right of travel direction.
        """
        if lane_samples is None or lane_samples.size == 0:
            return None

        p_anchor, t_anchor = point_and_tangent_at_s(seg_pts, anchor_s)
        t_norm = np.asarray(t_anchor, dtype=float)
        t_n = float(np.linalg.norm(t_norm))
        if t_n < 1e-6:
            return None
        t_norm /= t_n
        n_right = right_normal_world(t_norm)

        cos_min = math.cos(math.radians(max_angle_diff_deg))
        kept_lats: List[float] = []

        for row in lane_samples:
            x, y, tx, ty = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            rel = np.asarray([x - float(p_anchor[0]), y - float(p_anchor[1])], dtype=float)
            along = float(rel[0] * t_norm[0] + rel[1] * t_norm[1])
            if abs(along) > along_window_m:
                continue
            cosang = abs(float(tx * t_norm[0] + ty * t_norm[1]))
            if cosang < cos_min:
                continue
            lat = float(rel[0] * n_right[0] + rel[1] * n_right[1])
            if abs(lat) > float(max_lateral_m):
                continue
            kept_lats.append(lat)

        if len(kept_lats) < 4:
            return None

        lat_arr = np.asarray(kept_lats, dtype=float)
        if lat_arr.size >= 12 and clip_percentile > 0.0:
            lo = float(np.percentile(lat_arr, clip_percentile))
            hi = float(np.percentile(lat_arr, 100.0 - clip_percentile))
        else:
            lo = float(np.min(lat_arr))
            hi = float(np.max(lat_arr))

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return None

        # Convert centerline extrema into lane-edge envelope.
        edge_min = float(lo - 0.5 * LANE_WIDTH_M)
        edge_max = float(hi + 0.5 * LANE_WIDTH_M)
        return edge_min, edge_max

    def _estimate_sidewalk_offset_from_lanes(
        seg_pts: np.ndarray,
        anchor_s: float,
        lane_samples: np.ndarray,
        along_window_m: float = 20.0,
        max_angle_diff_deg: float = 40.0,
        max_lateral_m: float = 35.0,
    ) -> float:
        """
        Estimate an off-road start offset from local road bounds.
        Fallbacks to SIDEWALK_OFFSET_M when bounds are not recoverable.
        """
        bounds = _estimate_road_lateral_bounds(
            seg_pts=seg_pts,
            anchor_s=anchor_s,
            lane_samples=lane_samples,
            along_window_m=along_window_m,
            max_angle_diff_deg=max_angle_diff_deg,
            max_lateral_m=max_lateral_m,
        )
        if bounds is None:
            return SIDEWALK_OFFSET_M
        edge_min, edge_max = bounds
        # Keep a small sidewalk clearance from the detected road edge.
        est = max(abs(edge_min), abs(edge_max)) + 0.5
        return max(SIDEWALK_OFFSET_M, float(est))

    def _min_dist_to_cloud(point_xy: Tuple[float, float], cloud: np.ndarray) -> float:
        if cloud is None or cloud.size == 0:
            return float("inf")
        px, py = point_xy
        d = np.linalg.norm(cloud - np.asarray([[float(px), float(py)]], dtype=float), axis=1)
        return float(np.min(d)) if d.size > 0 else float("inf")

    def _stabilize_walker_crossing(
        category: str,
        motion: Dict[str, Any],
        anchor_spawn: Dict[str, float],
        seg_pts: np.ndarray,
        waypoints: List[Dict[str, Any]],
        ego_cloud: np.ndarray,
        min_sidewalk_clearance_m: float = 3.5,
        lane_cloud: Optional[np.ndarray] = None,
        min_lane_edge_clearance_m: float = 0.5,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
        if category != "walker":
            return waypoints, motion, float("inf")
        if str((motion or {}).get("type", "")).lower() != "cross_perpendicular":
            return waypoints, motion, float("inf")
        if not waypoints or not isinstance(waypoints[0], dict):
            return waypoints, motion, float("inf")

        def _spawn_clearance(wps: List[Dict[str, Any]]) -> float:
            if not wps or not isinstance(wps[0], dict):
                return float("inf")
            x = float(wps[0].get("x", 0.0))
            y = float(wps[0].get("y", 0.0))
            return _min_dist_to_cloud((x, y), ego_cloud)

        lane_required = 0.5 * LANE_WIDTH_M + max(0.0, float(min_lane_edge_clearance_m))

        def _spawn_lane_clearance(wps: List[Dict[str, Any]]) -> float:
            if lane_cloud is None or lane_cloud.size == 0:
                return float("inf")
            if not wps or not isinstance(wps[0], dict):
                return float("inf")
            x = float(wps[0].get("x", 0.0))
            y = float(wps[0].get("y", 0.0))
            return _min_dist_to_cloud((x, y), lane_cloud)

        def _push_off_lanes(
            p: np.ndarray,
            direction: np.ndarray,
            required_clearance_m: float,
            max_push_m: float = 8.0,
        ) -> Tuple[np.ndarray, float]:
            if lane_cloud is None or lane_cloud.size == 0:
                return p, 0.0
            q = np.asarray(p, dtype=float)
            u = np.asarray(direction, dtype=float)
            n = float(np.linalg.norm(u))
            if n < 1e-9:
                return q, 0.0
            u /= n
            pushed = 0.0
            for _ in range(12):
                d = _min_dist_to_cloud((float(q[0]), float(q[1])), lane_cloud)
                if d >= required_clearance_m - 1e-3:
                    break
                step = max(0.25, required_clearance_m - d + 0.05)
                remaining = max_push_m - pushed
                if remaining <= 1e-6:
                    break
                step = min(step, remaining)
                q = q + step * u
                pushed += step
            return q, pushed

        def _enforce_lane_clearance(
            wps: List[Dict[str, Any]],
            m: Dict[str, Any],
        ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
            if lane_cloud is None or lane_cloud.size == 0:
                return wps, m, float("inf")
            if not isinstance(wps, list) or len(wps) < 2:
                return wps, m, _spawn_lane_clearance(wps)
            if not isinstance(wps[0], dict) or not isinstance(wps[-1], dict):
                return wps, m, _spawn_lane_clearance(wps)

            p0 = np.asarray([float(wps[0].get("x", 0.0)), float(wps[0].get("y", 0.0))], dtype=float)
            p1 = np.asarray([float(wps[-1].get("x", 0.0)), float(wps[-1].get("y", 0.0))], dtype=float)
            v = p1 - p0
            nv = float(np.linalg.norm(v))
            if nv < 1e-9:
                return wps, m, _spawn_lane_clearance(wps)
            u = v / nv

            q0, push0 = _push_off_lanes(p0, -u, lane_required)
            q1, push1 = _push_off_lanes(p1, +u, lane_required)
            if push0 <= 1e-6 and push1 <= 1e-6:
                return wps, m, _spawn_lane_clearance(wps)

            out_wps: List[Dict[str, Any]] = []
            yaw = wrap180(heading_deg_from_vec(q1 - q0))
            denom = max(1, len(wps) - 1)
            for idx, wp in enumerate(wps):
                alpha = float(idx) / float(denom)
                p = q0 + alpha * (q1 - q0)
                nw = dict(wp)
                nw["x"] = float(p[0])
                nw["y"] = float(p[1])
                nw["yaw_deg"] = float(yaw)
                out_wps.append(nw)

            out_motion = dict(m)
            new_dist = float(np.linalg.norm(q1 - q0))
            old_dist = float(out_motion.get("cross_distance_m", 0.0) or 0.0)
            out_motion["cross_distance_m"] = max(old_dist, new_dist)
            return out_wps, out_motion, _spawn_lane_clearance(out_wps)

        waypoints, motion, lane_spawn_clearance = _enforce_lane_clearance(waypoints, motion)
        ego_spawn_clearance = _spawn_clearance(waypoints)
        if lane_spawn_clearance >= lane_required and ego_spawn_clearance >= min_sidewalk_clearance_m:
            return waypoints, motion, ego_spawn_clearance

        current_clearance = ego_spawn_clearance
        best_wps = waypoints
        best_motion = dict(motion)
        best_clearance = current_clearance
        best_lane_clearance = lane_spawn_clearance
        if best_lane_clearance >= lane_required and current_clearance >= min_sidewalk_clearance_m:
            return best_wps, best_motion, best_clearance

        original_side = str(best_motion.get("cross_direction", "unknown") or "unknown").lower()
        side_candidates = [original_side] + [s for s in ("left", "right") if s != original_side]
        tried = set()
        base_offset = float(best_motion.get("start_offset_m", 0.0) or 0.0)
        if base_offset < 1e-6:
            base_offset = 5.25

        base_anchor_s = float(best_motion.get("anchor_s_along", 0.5) or 0.5)
        anchor_deltas = (0.0, -0.08, 0.08, -0.16, 0.16, -0.24, 0.24, -0.32, 0.32, -0.4, 0.4)

        for side in side_candidates:
            if side not in ("left", "right"):
                continue
            for extra in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
                for d_anchor in anchor_deltas:
                    trial = dict(best_motion)
                    trial["cross_direction"] = side
                    trial["start_offset_m"] = max(base_offset + extra, 5.25)
                    trial["anchor_s_along"] = min(0.95, max(0.05, base_anchor_s + d_anchor))
                    key = (
                        trial["cross_direction"],
                        round(float(trial["start_offset_m"]), 3),
                        round(float(trial["anchor_s_along"]), 3),
                    )
                    if key in tried:
                        continue
                    tried.add(key)
                    trial_anchor = compute_spawn_from_anchor(
                        seg_pts,
                        float(trial["anchor_s_along"]),
                        "center",
                    )
                    wps = build_motion_waypoints(trial, category, trial_anchor, seg_pts)
                    if not isinstance(wps, list) or not wps:
                        continue
                    wps, trial, lane_clr = _enforce_lane_clearance(wps, trial)
                    clr = _spawn_clearance(wps)
                    meets_lane = lane_clr >= lane_required
                    meets_ego = clr >= min_sidewalk_clearance_m
                    best_meets_lane = best_lane_clearance >= lane_required
                    best_meets_ego = best_clearance >= min_sidewalk_clearance_m
                    better = False
                    if meets_lane and not best_meets_lane:
                        better = True
                    elif meets_lane == best_meets_lane and meets_ego and not best_meets_ego:
                        better = True
                    elif meets_lane == best_meets_lane and meets_ego == best_meets_ego:
                        if lane_clr > best_lane_clearance + 1e-6:
                            better = True
                        elif abs(lane_clr - best_lane_clearance) <= 1e-6 and clr > best_clearance + 1e-6:
                            better = True
                    if better:
                        best_lane_clearance = lane_clr
                        best_clearance = clr
                        best_wps = wps
                        best_motion = trial
                    if best_lane_clearance >= lane_required and best_clearance >= min_sidewalk_clearance_m:
                        return best_wps, best_motion, best_clearance

        return best_wps, best_motion, best_clearance

    def _bump(key: str, amount: int = 1) -> None:
        if stats is None:
            return
        stats[key] = int(stats.get(key, 0)) + amount

    # Set default values for optional args that may not be provided by SimpleNamespace
    if not hasattr(args, 'placement_mode'):
        args.placement_mode = "csp"  # default to CSP-based placement
    if not hasattr(args, 'do_sample'):
        args.do_sample = False
    if not hasattr(args, 'temperature'):
        args.temperature = 0.2
    if not hasattr(args, 'top_p'):
        args.top_p = 0.95

    t0 = time.time()
    with open(args.picked_paths, "r", encoding="utf-8") as f:
        picked_payload = json.load(f)

    picked = picked_payload.get("picked", [])
    if not isinstance(picked, list) or not picked:
        raise SystemExit("[ERROR] picked_paths_detailed.json has no 'picked' list.")

    crop_region = picked_payload.get("crop_region")
    nodes_field = picked_payload.get("nodes")
    if not nodes_field:
        raise SystemExit("[ERROR] picked_paths_detailed.json missing 'nodes' field")

    resolved_nodes_path = resolve_nodes_path(args.picked_paths, str(nodes_field), args.nodes_root)
    if not os.path.exists(resolved_nodes_path):
        raise SystemExit(f"[ERROR] nodes path not found: {resolved_nodes_path}\n"
                         f"Tip: pass --nodes-root to resolve relative paths.")
    nodes: Optional[Dict[str, Any]] = None
    all_segments: List[Dict[str, Any]] = []
    seg_by_id: Dict[int, np.ndarray] = {}
    if args.placement_mode == "csp" or args.viz:
        nodes = load_nodes(resolved_nodes_path)
        all_segments = build_segments_from_nodes(nodes)
        seg_by_id = {int(s["seg_id"]): s["points"] for s in all_segments}
        seg_by_id = _override_seg_points_with_picked(picked, seg_by_id)

    all_assets = load_assets(args.carla_assets)

    # Build vehicle segment summaries for LLM
    vehicle_segments = build_vehicle_segment_summaries(picked)
    ego_path_cloud = _ego_path_point_cloud(picked)
    ego_lane_samples = _ego_lane_samples_from_picked(picked)
    crop_bounds: Optional[Tuple[float, float, float, float]] = None
    if isinstance(crop_region, dict):
        try:
            crop_bounds = (
                float(crop_region["xmin"]),
                float(crop_region["xmax"]),
                float(crop_region["ymin"]),
                float(crop_region["ymax"]),
            )
        except Exception:
            crop_bounds = None
    # Keep full-resolution lane samples so walker off-road checks are reliable near intersections.
    map_lane_samples = _lane_samples_from_segment_bank(all_segments, crop_bounds=crop_bounds, stride=1)
    if ego_lane_samples.size == 0:
        sidewalk_lane_samples = map_lane_samples
    elif map_lane_samples.size == 0:
        sidewalk_lane_samples = ego_lane_samples
    else:
        sidewalk_lane_samples = np.vstack([ego_lane_samples, map_lane_samples])
    lane_center_cloud = (
        sidewalk_lane_samples[:, :2].astype(float, copy=False)
        if isinstance(sidewalk_lane_samples, np.ndarray) and sidewalk_lane_samples.size > 0
        else np.zeros((0, 2), dtype=float)
    )
    print(f"[TIMING] object_placer setup (load paths, assets, summaries): {time.time() - t0:.2f}s", flush=True)

    # Load HF model if not provided
    if tokenizer is None or model is None:
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model.eval()
        print(f"[TIMING] object_placer model load: {time.time() - t0:.2f}s", flush=True)

    # --------------------------
    # Stage 1: extract entities (or use schema-provided actors)
    # --------------------------
    t_stage1_start = time.time()
    schema_entities = getattr(args, "schema_entities", None)
    print(f"[DEBUG] schema_entities provided: {schema_entities is not None}", flush=True)
    if schema_entities is not None:
        print(f"[DEBUG] schema_entities count: {len(schema_entities) if isinstance(schema_entities, list) else 'not a list'}", flush=True)
        for i, e in enumerate(schema_entities[:5] if isinstance(schema_entities, list) else []):
            print(f"[DEBUG]   entity[{i}]: entity_id={e.get('entity_id')}, kind={e.get('actor_kind')}, mention={e.get('mention')}", flush=True)
        print("[INFO] Using schema-provided actors; skipping Stage1 LLM.")
        stage1_obj = {"entities": schema_entities}
    else:
        t_stage1_start = time.time()
        stage1_prompt = build_stage1_prompt(args.description)
        t0 = time.time()
        stage1_text = generate_with_model(
            model=model,
            tokenizer=tokenizer,
            prompt=stage1_prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"[TIMING] Stage1 LLM generation: {time.time() - t0:.2f}s", flush=True)
        t0 = time.time()
        # Stage 1 parse (with repair if the model didn't output JSON)
        try:
            stage1_obj = parse_llm_json(stage1_text, required_top_keys=["entities"])
        except Exception:
            _bump("object_stage1_json_repair")
            repair_prompt = (
                "Return JSON ONLY with top-level key 'entities' (a list). No prose.\n"
                "If you previously wrote anything else, convert it into the required JSON now.\n\n"
                "RAW OUTPUT:\n" + stage1_text
            )
            repair_text = generate_with_model(
                model=model,
                tokenizer=tokenizer,
                prompt=repair_prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            stage1_obj = parse_llm_json(repair_text, required_top_keys=["entities"])
        print(f"[TIMING] Stage1 parse+repair: {time.time() - t0:.2f}s", flush=True)

    entities = stage1_obj.get("entities", [])
    if not isinstance(entities, list):
        raise SystemExit("[ERROR] Stage1: 'entities' must be a list.")
    _log_actor_trace("stage1_entities_extracted", entities)

    # Ensure each entity has a unique entity_id (normalize if LLM didn't provide one)
    valid_entity_ids = set()
    for idx, e in enumerate(entities):
        if not e.get("entity_id"):
            e["entity_id"] = f"entity_{idx + 1}"
        valid_entity_ids.add(e["entity_id"])

    
    # Post-filter Stage 1 entities to reduce hallucinations (Fix A + Fix D)
    t0 = time.time()
    dropped_stage1: List[Tuple[str, str, str]] = []
    filtered_entities: List[Dict[str, Any]] = []

    def _repair_evidence_with_llm(ent: Dict[str, Any]) -> None:
        """Best-effort: if Stage1 paraphrased, try to recover an EXACT supporting quote."""
        ev = str(ent.get("evidence") or "").strip()
        mention = str(ent.get("mention") or "").strip()
        if ev and _contains_exact_quote(args.description, ev):
            return
        if mention and _contains_exact_quote(args.description, mention):
            ent["evidence"] = mention
            return

        # One-shot repair: ask the model to point to an exact substring.
        try:
            _bump("object_stage1_evidence_repair")
            prompt = (
                "Return JSON ONLY: {\"evidence\": \"...\"}.\n"
                "The evidence MUST be an EXACT substring (<=20 words) copied from DESCRIPTION that explicitly mentions the actor (not just a location).\n"
                "If you cannot find any supporting substring, return {\"evidence\": \"\"}.\n\n"
                f"ACTOR_KIND: {ent.get('actor_kind','')}\n"
                f"MENTION (may be paraphrase): {mention}\n\n"
                f"DESCRIPTION:\n{args.description}\n"
            )
            txt = generate_with_model(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=80,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
            )
            obj = parse_llm_json(txt, required_top_keys=["evidence"])
            rep = str(obj.get("evidence") or "").strip()
            if rep and _contains_exact_quote(args.description, rep):
                ent["evidence"] = rep
        except Exception:
            return

    def _sanitize_trigger_action(ent: Dict[str, Any]) -> None:
        trigger = ent.get("trigger")
        action = ent.get("action")

        # Trigger sanitization
        if not isinstance(trigger, dict):
            ent["trigger"] = None
        else:
            ttype = str(trigger.get("type", "")).strip()
            if ttype != "distance_to_vehicle":
                ent["trigger"] = None
            else:
                vehicle = str(trigger.get("vehicle", "")).strip()
                m = re.match(r"^(?:Vehicle|vehicle)\s+(\d+)$", vehicle)
                if not m:
                    ent["trigger"] = None
                else:
                    vehicle = f"Vehicle {m.group(1)}"
                    try:
                        distance_m = float(trigger.get("distance_m", 8.0))
                    except Exception:
                        distance_m = 8.0
                    # Clamp to a sane range
                    distance_m = max(1.0, min(50.0, distance_m))
                    evidence = str(trigger.get("evidence", "")).strip()
                    if evidence and not _contains_exact_quote(args.description, evidence):
                        evidence = ""
                    ent["trigger"] = {
                        "type": "distance_to_vehicle",
                        "vehicle": vehicle,
                        "distance_m": distance_m,
                        "evidence": evidence,
                    }

        # Action sanitization
        if not isinstance(action, dict):
            ent["action"] = None
        else:
            atype = str(action.get("type", "")).strip()
            if atype not in ("start_motion", "hard_brake", "lane_change"):
                ent["action"] = None
            else:
                direction = action.get("direction")
                direction = str(direction).strip().lower() if direction is not None else ""
                if direction not in ("left", "right"):
                    direction = None
                target_vehicle = action.get("target_vehicle")
                target_vehicle = str(target_vehicle).strip() if target_vehicle else None
                if target_vehicle and not re.match(r"^Vehicle\s+\d+$", target_vehicle):
                    target_vehicle = None
                ent["action"] = {
                    "type": atype,
                    "direction": direction,
                    "target_vehicle": target_vehicle,
                }

    def _extract_trigger_candidates(description: str) -> List[Dict[str, Any]]:
        patterns = [
            re.compile(
                r"(?i)\b(?:when|once|if)\s+(Vehicle\s+\d+)\s+(?:gets|is)\s+within\s+(\d+(?:\.\d+)?)\s*(?:m|meter|meters|metres)\b"
            ),
            re.compile(
                r"(?i)\b(?:when|once|if)\s+(Vehicle\s+\d+)\s+gets\s+close\b"
            ),
            re.compile(
                r"(?i)\b(?:when|once|if)\s+(Vehicle\s+\d+)\s+approaches\b"
            ),
        ]
        candidates: List[Dict[str, Any]] = []
        for pat in patterns:
            for m in pat.finditer(description):
                vehicle = m.group(1)
                distance_m = 8.0
                if m.lastindex and m.lastindex >= 2:
                    try:
                        distance_m = float(m.group(2))
                    except Exception:
                        distance_m = 8.0
                candidates.append({
                    "vehicle": vehicle,
                    "distance_m": distance_m,
                    "evidence": m.group(0).strip(),
                    "span": m.span(),
                })
        return candidates

    def _find_substring_span(text: str, sub: str) -> Optional[Tuple[int, int]]:
        if not sub:
            return None
        idx = text.find(sub)
        if idx == -1:
            return None
        return idx, idx + len(sub)

    def _sentence_bounds(text: str, idx: int) -> Tuple[int, int]:
        seps = ".!?"
        start = -1
        for sep in seps:
            start = max(start, text.rfind(sep, 0, idx))
        start = 0 if start == -1 else start + 1
        end_positions = [text.find(sep, idx) for sep in seps if text.find(sep, idx) != -1]
        end = min(end_positions) if end_positions else len(text)
        return start, end

    def _apply_trigger_fallback(entities: List[Dict[str, Any]], description: str) -> None:
        candidates = _extract_trigger_candidates(description)
        if not candidates:
            return
        used = set()

        for ent in entities:
            if ent.get("trigger") is not None:
                continue
            if ent.get("action") is None:
                continue
            mention = str(ent.get("mention") or "")
            evidence = str(ent.get("evidence") or "")
            anchor_span = _find_substring_span(description, mention) or _find_substring_span(description, evidence)
            anchor_idx = anchor_span[0] if anchor_span else None

            available = [(i, c) for i, c in enumerate(candidates) if i not in used]
            if not available:
                continue

            picked = None
            picked_idx = None
            if anchor_idx is not None:
                sent_start, sent_end = _sentence_bounds(description, anchor_idx)
                in_sentence = []
                for idx, cand in available:
                    span = cand.get("span", (0, 0))
                    if span[0] >= sent_start and span[1] <= sent_end:
                        in_sentence.append((idx, cand))
                if in_sentence:
                    def _dist(c):
                        span = c.get("span", (0, 0))
                        mid = (span[0] + span[1]) / 2.0
                        return abs(mid - anchor_idx)
                    picked_idx, picked = min(in_sentence, key=lambda ic: _dist(ic[1]))
                    used.add(picked_idx)
                elif len(available) == 1:
                    picked_idx, picked = available[0]
                    used.add(picked_idx)
                else:
                    def _dist(c):
                        span = c.get("span", (0, 0))
                        mid = (span[0] + span[1]) / 2.0
                        return abs(mid - anchor_idx)
                    picked_idx, picked = min(available, key=lambda ic: _dist(ic[1]))
                    used.add(picked_idx)
            else:
                if len(available) == 1 and len(entities) == 1:
                    picked_idx, picked = available[0]
                    used.add(picked_idx)

            if picked:
                ent["trigger"] = {
                    "type": "distance_to_vehicle",
                    "vehicle": picked["vehicle"],
                    "distance_m": float(picked["distance_m"]),
                    "evidence": picked["evidence"],
                }
                ent_id = ent.get("entity_id", "entity")
                print(f"[INFO] Stage1 trigger fallback: {ent_id} -> {picked['vehicle']} @ {picked['distance_m']}m")

    def _is_moving_entity(ent: Dict[str, Any]) -> bool:
        kind = str(ent.get("actor_kind") or "").strip()
        motion_hint = str(ent.get("motion_hint") or "").strip().lower()
        speed_hint = str(ent.get("speed_hint") or "").strip().lower()
        action = ent.get("action")
        if isinstance(action, dict) and action.get("type") in ("start_motion", "hard_brake", "lane_change"):
            return True
        if motion_hint in ("crossing", "follow_lane"):
            return True
        if speed_hint in ("slow", "normal", "fast", "erratic"):
            return True
        if motion_hint == "static" or speed_hint == "stopped":
            return False
        return kind in ("walker", "cyclist", "npc_vehicle")

    def _default_trigger_vehicle(ent: Dict[str, Any]) -> str:
        vehicle = str(ent.get("affects_vehicle") or "").strip()
        if vehicle.lower() in ("ego", "ego vehicle", "ego_vehicle"):
            return "Vehicle 1"
        match = re.search(r"(\d+)", vehicle)
        if match:
            return f"Vehicle {match.group(1)}"
        return "Vehicle 1"

    def _apply_default_movement_triggers(entities: List[Dict[str, Any]]) -> None:
        for ent in entities:
            if not _is_moving_entity(ent):
                continue
            if ent.get("trigger") is None:
                ent["trigger"] = {
                    "type": "distance_to_vehicle",
                    "vehicle": _default_trigger_vehicle(ent),
                    "distance_m": 8.0,
                    "evidence": "",
                }
            if ent.get("action") is None:
                ent["action"] = {
                    "type": "start_motion",
                    "direction": None,
                    "target_vehicle": None,
                }
            _sanitize_trigger_action(ent)

    for e in entities:
        _repair_evidence_with_llm(e)
        _sanitize_trigger_action(e)

        drop, reason = _should_drop_stage1_entity(e, args.description)
        if drop:
            dropped_stage1.append((str(e.get("entity_id")), reason, str(e.get("mention") or "")))
            continue
        filtered_entities.append(e)
    _apply_trigger_fallback(filtered_entities, args.description)
    if dropped_stage1:
        for ent_id, reason, mention in dropped_stage1[:50]:
            print(f"[WARNING] Stage1 drop: {ent_id}: {reason}; mention='{mention[:60]}'")
    entities = filtered_entities
    _apply_default_movement_triggers(entities)
    valid_entity_ids = set(str(e.get("entity_id")) for e in entities if e.get("entity_id"))
    print(f"[TIMING] Stage1 entity filtering+evidence repair: {time.time() - t0:.2f}s", flush=True)
    print(f"[TIMING] Stage1 total: {time.time() - t_stage1_start:.2f}s", flush=True)

    print(f"[INFO] Stage1 extracted {len(entities)} entities: {list(valid_entity_ids)}")
    # Debug: show Stage 1 entity details including motion_hint and when (route phase)
    for e in entities:
        print(f"  - {e.get('entity_id')}: kind={e.get('actor_kind')}, motion_hint={e.get('motion_hint')}, when={e.get('when')}, mention='{e.get('mention', '')[:50]}'")

    # Keyword synonyms for better asset matching
    t0 = time.time()
    KEYWORD_SYNONYMS = {
        "cyclist": ["bike", "bicycle", "crossbike"],
        "bicyclist": ["bike", "bicycle", "crossbike"],
        "biker": ["bike", "bicycle", "crossbike", "motorcycle"],
        "motorcyclist": ["motorcycle", "harley", "yamaha", "kawasaki"],
        "pedestrian": ["pedestrian", "walker", "person"],
        "person": ["pedestrian", "walker"],
        "cone": ["cone", "trafficcone", "constructioncone"],
        "cones": ["cone", "trafficcone", "constructioncone"],
        "traffic cone": ["cone", "trafficcone", "constructioncone"],
        "barrier": ["barrier", "streetbarrier", "construction"],
        "truck": ["truck", "firetruck", "cybertruck", "pickup"],
        "police": ["police", "charger", "crown"],
        "ambulance": ["ambulance"],
        "firetruck": ["firetruck"],
    }
    OCCLUSION_HINTS = (
        "obstruct", "obstructs", "obstructing", "occlude", "occluding", "visibility",
        "view", "block", "blocking", "obscure", "obscuring", "blind", "line of sight",
    )
    LARGE_STATIC_TOKENS = (
        "streetbarrier", "barrier", "chainbarrier", "busstop", "advertisement",
        "container", "clothcontainer", "glasscontainer",
    )
    SMALL_STATIC_TOKENS = (
        "barrel", "cone", "trafficcone", "bin", "box", "trash", "garbage", "bag", "bottle", "can",
    )
    SPECIFIC_STATIC_TOKENS = LARGE_STATIC_TOKENS + SMALL_STATIC_TOKENS

    def _has_occlusion_hint(mention: str, evidence: str, description: str, static_count: int) -> bool:
        text = f"{mention} {evidence}".lower()
        if any(h in text for h in OCCLUSION_HINTS):
            return True
        if static_count == 1:
            dlow = str(description or "").lower()
            return any(h in dlow for h in OCCLUSION_HINTS)
        return False

    def _mentions_specific_static_prop(text: str) -> bool:
        low = str(text or "").lower()
        return any(t in low for t in SPECIFIC_STATIC_TOKENS)

    def _asset_matches_tokens(asset, tokens) -> bool:
        hay = " ".join([asset.asset_id.lower()] + asset.tags)
        return any(t in hay for t in tokens)

    def _asset_area(asset) -> float:
        if not asset.bbox:
            return 0.0
        return float(asset.bbox.length * asset.bbox.width)

    def _merge_assets(primary, secondary, limit: int = 12):
        seen = set()
        out = []
        for a in list(primary) + list(secondary):
            if a.asset_id in seen:
                continue
            seen.add(a.asset_id)
            out.append(a)
            if len(out) >= limit:
                break
        return out

    large_vehicle_assets = [a for a in all_assets if a.category == "vehicle" and a.bbox]
    large_vehicle_assets.sort(key=_asset_area, reverse=True)
    top_vehicle_occluders = large_vehicle_assets[:6]

    static_like_count = sum(
        1 for e in entities if str(e.get("actor_kind", "")) in ("static_prop", "parked_vehicle")
    )

    # For each entity, build small asset option list (keyed by entity_id)
    per_entity_options: Dict[str, List[Dict[str, Any]]] = {}
    for idx, e in enumerate(entities):
        entity_id = e.get("entity_id", f"entity_{idx+1}")
        mention = str(e.get("mention", f"entity_{idx+1}"))
        evidence = str(e.get("evidence", ""))
        kind = str(e.get("actor_kind", "static_prop"))
        low = mention.lower()
        occlusion_hint = False
        if kind in ("static_prop", "parked_vehicle"):
            occlusion_hint = _has_occlusion_hint(mention, evidence, args.description, static_like_count)

        # Extract keywords: split mention into words and check synonyms
        kws = set()
        words = [w.strip(".,!?") for w in low.split()]
        
        # Add all meaningful words from mention
        for word in words:
            if len(word) > 2:  # Skip tiny words
                kws.add(word)
            # Expand synonyms
            if word in KEYWORD_SYNONYMS:
                kws.update(KEYWORD_SYNONYMS[word])
        
        # Also check multi-word phrases
        for phrase, synonyms in KEYWORD_SYNONYMS.items():
            if phrase in low:
                kws.update(synonyms)

        # Add kind-based keywords (always, not just as fallback)
        if kind == "walker":
            categories = ["walker"]
            kws.update(["pedestrian", "walker"])
        elif kind == "cyclist":
            categories = ["vehicle"]
            kws.update(["bike", "bicycle", "crossbike"])
        elif kind in ("parked_vehicle", "npc_vehicle"):
            categories = ["vehicle"]
            # Keep mention-based keywords; add generic fallbacks only if empty
            if not any(w in kws for w in ["car", "truck", "bus", "van", "vehicle"]):
                kws.update(["car", "vehicle"])
        else:
            categories = ["static"]
            kws.update(["prop", "static"])
            if occlusion_hint:
                kws.update(["barrier", "streetbarrier", "chainbarrier", "busstop", "advertisement", "container"])

        kws_list = list(kws)
        options = keyword_filter_assets(all_assets, kws_list, categories=categories, k=12)

        # last-resort fallback options
        if not options:
            # choose a small default set by category
            options = keyword_filter_assets(all_assets, ["vehicle"], categories=categories, k=12) or all_assets[:12]

        if occlusion_hint and kind == "static_prop":
            text = f"{mention} {evidence}"
            if not _mentions_specific_static_prop(text):
                options = _merge_assets(top_vehicle_occluders, options)
            large = [a for a in options if _asset_matches_tokens(a, LARGE_STATIC_TOKENS)]
            if large:
                options = _merge_assets(large, options)
            else:
                options.sort(key=lambda a: (_asset_matches_tokens(a, SMALL_STATIC_TOKENS), a.asset_id))
        if occlusion_hint and kind == "parked_vehicle":
            options = _merge_assets(top_vehicle_occluders, options)
            options.sort(key=_asset_area, reverse=True)

        # Key by entity_id for Stage 2
        per_entity_options[entity_id] = [
            {"asset_id": a.asset_id, "category": a.category, "tags": a.tags[:6]} for a in options
        ]
    print(f"[TIMING] asset matching for {len(entities)} entities: {time.time() - t0:.2f}s", flush=True)

    ego_spawns: List[Dict[str, Any]] = _extract_ego_spawns_from_picked(picked)

    # Handle empty entities case early
    if not entities:
        print("[INFO] No entities extracted in Stage 1. Skipping Stage 2.")
        actors = []
        stage2_obj = {"actors": []}
    else:
        # --------------------------
        # Stage 2: resolve anchors
        # --------------------------
        t_stage2_start = time.time()

        if args.placement_mode == "llm_anchor":
            stage2_prompt = build_stage2_prompt(args.description, vehicle_segments, entities, per_entity_options)
            stage2_text = generate_with_model(
                model=model,
                tokenizer=tokenizer,
                prompt=stage2_prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print("\n[DEBUG] Stage2 raw output (full):\n" + stage2_text + "\n", flush=True)

            try:
                stage2_obj = parse_llm_json(stage2_text, required_top_keys=["actors"])
            except ValueError as exc:
                _bump("object_stage2_json_repair")
                repair_prompt = build_repair_prompt(stage2_text, [f"JSON parse failed: {exc}"])
                repair_text = generate_with_model(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=repair_prompt,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                stage2_obj = parse_llm_json(repair_text, required_top_keys=["actors"])
            actors = stage2_obj.get("actors", [])
            if not isinstance(actors, list):
                raise SystemExit("[ERROR] Stage2: 'actors' must be a list.")

            errs = validate_stage2_output(actors, vehicle_segments)
            if errs:
                _bump("object_stage2_validation_repair")
                repair_prompt = build_repair_prompt(stage2_text, errs)
                repair_text = generate_with_model(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=repair_prompt,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                stage2_obj = parse_llm_json(repair_text, required_top_keys=["actors"])
                actors = stage2_obj.get("actors", [])
                if not isinstance(actors, list):
                    raise SystemExit("[ERROR] Stage2 repair: 'actors' must be a list.")
            print(f"[TIMING] Stage2 (llm_anchor mode) total: {time.time() - t_stage2_start:.2f}s", flush=True)
        else:
            # CSP mode: LLM emits symbolic preferences; solver chooses anchors.
            # We need geometry (seg_by_id) to enumerate candidates; build it here.
            t0 = time.time()
            if nodes is None:
                nodes = load_nodes(resolved_nodes_path)
                all_segments = build_segments_from_nodes(nodes)
                seg_by_id = {int(s["seg_id"]): s["points"] for s in all_segments}
                seg_by_id = _override_seg_points_with_picked(picked, seg_by_id)
            merge_min_s_by_vehicle = _compute_merge_min_s_by_vehicle(picked_payload, picked, seg_by_id)
            print(f"[TIMING] Stage2 load nodes+build segments: {time.time() - t0:.2f}s", flush=True)

            t0 = time.time()
            stage2_prompt = build_stage2_constraints_prompt(args.description, vehicle_segments, entities, per_entity_options)
            stage2_text = generate_with_model(
                model=model,
                tokenizer=tokenizer,
                prompt=stage2_prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print(f"[TIMING] Stage2 LLM generation: {time.time() - t0:.2f}s", flush=True)
            print("\n[DEBUG] Stage2(CSP) raw output (full):\n" + stage2_text + "\n", flush=True)

            # Parse with a simple repair-on-failure
            t0 = time.time()
            try:
                stage2_obj = parse_llm_json(stage2_text, required_top_keys=["actor_specs"])
            except Exception:
                _bump("object_stage2_json_repair")
                repair_prompt = (
                    "Return JSON ONLY with top-level key 'actor_specs' (a list). No prose.\n"
                    "If needed, convert your previous output into the required JSON now.\n\n"
                    "RAW OUTPUT:\n" + stage2_text
                )
                repair_text = generate_with_model(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=repair_prompt,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                stage2_obj = parse_llm_json(repair_text, required_top_keys=["actor_specs"])
            print(f"[TIMING] Stage2 parse+repair: {time.time() - t0:.2f}s", flush=True)

            actor_specs_raw = stage2_obj.get("actor_specs", [])
            
            # Build debug log for file output
            debug_log = {
                "stage2_raw_output": stage2_text[:2000] if len(stage2_text) > 2000 else stage2_text,
                "actor_specs_raw_count": len(actor_specs_raw),
                "actor_specs_raw": actor_specs_raw,
                "entities_for_stage2": [{"entity_id": e.get("entity_id"), "actor_kind": e.get("actor_kind"), "mention": e.get("mention")} for e in entities],
                "per_entity_options": {eid: len(opts) for eid, opts in per_entity_options.items()},
            }
            
            print(f"[DEBUG] actor_specs_raw count: {len(actor_specs_raw)}", flush=True)
            print(f"[DEBUG] entities for Stage2: {[e.get('entity_id') for e in entities]}", flush=True)
            print(f"[DEBUG] per_entity_options keys: {list(per_entity_options.keys())}", flush=True)
            for eid, opts in per_entity_options.items():
                print(f"[DEBUG]   {eid}: {len(opts)} asset options", flush=True)
            actor_specs, warns = validate_actor_specs(actor_specs_raw, entities, per_entity_options, picked=picked)
            
            debug_log["actor_specs_after_validation_count"] = len(actor_specs)
            debug_log["actor_specs_after_validation"] = actor_specs
            debug_log["validation_warnings"] = warns
            debug_log["actor_trace"] = actor_trace
            _log_actor_trace("after_validation", actor_specs, note="stage2 actor_specs")
            
            # Save debug log to file
            debug_log_path = str(args.out).replace(".json", "_debug.json") if hasattr(args, 'out') and args.out else "/tmp/object_placer_debug.json"
            try:
                with open(debug_log_path, "w") as f:
                    json.dump(debug_log, f, indent=2, default=str)
                print(f"[DEBUG] Saved debug log to: {debug_log_path}", flush=True)
            except Exception as e:
                print(f"[DEBUG] Failed to save debug log: {e}", flush=True)
            
            print(f"[DEBUG] actor_specs after validation: {len(actor_specs)}", flush=True)
            for w in warns[:50]:
                print("[WARNING] Stage2(CSP): " + w)

            if not actor_specs:
                print("[DEBUG] actor_specs is EMPTY after validation - no actors will be placed!", flush=True)
                actors = []
                _log_actor_trace("actor_specs_empty", [])
            else:
                t0 = time.time()
                print(f"[TIMING] Starting CSP solve with {len(actor_specs)} actor specs...", flush=True)
                for i, spec in enumerate(actor_specs[:5]):
                    print(f"[DEBUG] actor_spec[{i}]: id={spec.get('id')}, asset_id={spec.get('asset_id')}, category={spec.get('category')}", flush=True)
                # Use iterative CSP with path extension - extends paths on-demand when distance constraints can't be met
                # Returns expanded crop_region to include extended path areas
                chosen, dbg, crop_region = solve_weighted_csp_with_extension(
                    actor_specs, picked, seg_by_id, crop_region,
                    all_segments=all_segments,
                    nodes=nodes,
                    merge_min_s_by_vehicle=merge_min_s_by_vehicle,
                    min_sep_scale=1.0,
                    max_extension_iterations=3,
                    ego_spawns=ego_spawns,
                )
                print(f"[TIMING] CSP solve (with extension): {time.time() - t0:.2f}s", flush=True)
                print("[INFO] CSP solve debug: " + json.dumps(dbg, indent=2))
                _log_actor_trace("after_csp_solve", actor_specs, note=f"chosen={list(chosen.keys())}")

                actors = []
                for spec in actor_specs:
                    sid = str(spec["id"])
                    cand = chosen.get(sid)
                    if cand is None:
                        continue

                    # Get bounding box info if available
                    asset_id = spec["asset_id"]
                    bbox = get_asset_bbox(asset_id)
                    bbox_info = None
                    if bbox:
                        bbox_info = {
                            "length": bbox.length,
                            "width": bbox.width,
                            "height": bbox.height,
                        }

                    base_actor = {
                        "id": sid,
                        "semantic": spec.get("semantic", sid),
                        "category": spec["category"],
                        "asset_id": asset_id,
                        "trigger": spec.get("trigger"),
                        "action": spec.get("action"),
                        "placement": {
                            "target_vehicle": f"Vehicle {cand.vehicle_num}" if cand.vehicle_num > 0 else None,
                            "segment_index": int(cand.segment_index),
                            "s_along": float(cand.s_along),
                            "lateral_relation": str(cand.lateral_relation),
                            "seg_id": int(cand.seg_id),  # Store seg_id directly for opposite-lane NPCs
                        },
                        "motion": spec.get("motion", {"type": "static", "speed_profile": "normal"}),
                        "confidence": float(spec.get("confidence", 0.6)),
                        "csp": {
                            "base_score": float(cand.base_score),
                            "path_s_m": float(cand.path_s_m),
                            "relations": spec.get("relations", []),
                        },
                        "bbox": bbox_info,  # Include bounding box if available
                    }

                    expanded = expand_group_to_actors(base_actor, spec, cand, seg_by_id)
                    # Validate expansion: check if quantity matches expected
                    expected_qty = int(spec.get("quantity", 1))
                    if len(expanded) < expected_qty:
                        print(f"[WARNING] {sid}: expected {expected_qty} actors but only got {len(expanded)} "
                              f"(group_pattern={spec.get('group_pattern')}, seg_id={cand.seg_id})", flush=True)
                    actors.extend(expanded)
                _log_actor_trace("after_expand_group", actors)
            print(f"[TIMING] Stage2 (CSP mode) total: {time.time() - t_stage2_start:.2f}s", flush=True)


    # --------------------------
    # Geometry reconstruction
    # --------------------------
    t0 = time.time()
    resolved_nodes_path = resolve_nodes_path(args.picked_paths, str(nodes_field), args.nodes_root)
    if not os.path.exists(resolved_nodes_path):
        raise SystemExit(f"[ERROR] nodes path not found: {resolved_nodes_path}\n"
                         f"Tip: pass --nodes-root to resolve relative paths.")

    nodes = load_nodes(resolved_nodes_path)
    all_segments = build_segments_from_nodes(nodes)
    seg_by_id: Dict[int, np.ndarray] = {int(s["seg_id"]): s["points"] for s in all_segments}
    # If paths were refined (start/end trimming or synthetic segments), prefer those polylines.
    seg_by_id = _override_seg_points_with_picked(picked, seg_by_id)
    print(f"[TIMING] geometry reconstruction: {time.time() - t0:.2f}s", flush=True)

    # --------------------------
    # Convert anchors -> world
    # --------------------------
    t0 = time.time()
    # Guardrail: if Stage1 said "after_turn" but Stage2 placed the actor on a turning connector,
    # shift it onto the inferred post-turn (exit) segment before spawning.
    apply_after_turn_segment_corrections(actors, stage1_obj.get("entities", []), picked, seg_by_id)
    # Guardrail: if Stage1 said "in_intersection", keep it on a turn-connector segment.
    apply_in_intersection_segment_corrections(actors, stage1_obj.get("entities", []), picked, seg_by_id)

    # Guardrail: re-anchor actors to Stage1 affects_vehicle when drifted.
    stage1_affects = {
        str(e.get("entity_id")): str(e.get("affects_vehicle", "") or "").strip()
        for e in stage1_obj.get("entities", [])
        if e.get("entity_id")
    }
    _log_actor_trace("before_reanchor", actors)
    filtered_actors: List[Dict[str, Any]] = []
    for actor in actors:
        ent_id = str(actor.get("entity_id") or actor.get("id") or "")
        intended = stage1_affects.get(ent_id, "")
        placement = actor.get("placement", {}) if isinstance(actor.get("placement", {}), dict) else {}
        placed_tv = str(placement.get("target_vehicle") or "").strip()
        if intended and intended.lower() not in ("", "none", "unknown") and placed_tv not in (intended, ""):
            # Try to re-anchor to intended vehicle's first segment midpoint
                picked_entry = next((p for p in picked if p.get("vehicle") == intended), None)
                if picked_entry:
                    seg_ids = (picked_entry.get("signature", {}) or {}).get("segment_ids", [])
                    if seg_ids:
                        placement = dict(placement)
                    placement["target_vehicle"] = intended
                    placement["segment_index"] = 1
                    placement["seg_id"] = seg_ids[0]
                    placement["s_along"] = 0.2
                    placement.setdefault("lateral_relation", "center")
                    actor = dict(actor)
                    actor["placement"] = placement
                    print(
                        f"[WARNING] Re-anchoring actor {ent_id} to {intended} (was {placed_tv or 'none'})",
                        flush=True,
                    )
                else:
                    print(
                        f"[WARNING] Could not re-anchor actor {ent_id}: no segments for {intended}",
                        flush=True,
                    )
                    continue
        filtered_actors.append(actor)
    actors = filtered_actors
    _log_actor_trace("after_reanchor", actors)

    # Expand quantity>1 entities for llm_anchor placement mode.
    if args.placement_mode == "llm_anchor":
        stage1_entities = stage1_obj.get("entities", [])
        entity_by_id = {
            str(e.get("entity_id")): e for e in stage1_entities if e.get("entity_id")
        }

        def _segment_id_for_actor(actor: Dict[str, Any]) -> Optional[int]:
            placement = actor.get("placement", {}) if isinstance(actor.get("placement", {}), dict) else {}
            seg_id = placement.get("seg_id")
            if seg_id is not None:
                return int(seg_id)
            tv = placement.get("target_vehicle")
            try:
                seg_idx = int(placement.get("segment_index", 0))
            except Exception:
                seg_idx = 0
            if not tv or seg_idx <= 0:
                return None
            picked_entry = next((p for p in picked if p.get("vehicle") == tv), None)
            if not picked_entry:
                return None
            seg_ids = (picked_entry.get("signature", {}) or {}).get("segment_ids", [])
            if not isinstance(seg_ids, list) or seg_idx > len(seg_ids):
                return None
            return int(seg_ids[seg_idx - 1])

        def _normalize_group_pattern(ent: Dict[str, Any], qty: int) -> Optional[Dict[str, Any]]:
            if qty <= 1:
                return None
            patt = str(ent.get("group_pattern", "unknown"))
            if patt == "diagonal":
                patt = "along_lane"
            if patt not in ("across_lane", "along_lane", "scatter", "unknown"):
                patt = "along_lane"
            sl = ent.get("start_lateral")
            el = ent.get("end_lateral")
            if sl is not None and str(sl) not in LATERAL_TO_M:
                sl = None
            if el is not None and str(el) not in LATERAL_TO_M:
                el = None
            return {"pattern": patt, "start_lateral": sl, "end_lateral": el, "spacing_bucket": "auto"}

        expanded: List[Dict[str, Any]] = []
        for actor in actors:
            entity_id = str(actor.get("entity_id") or actor.get("id") or "")
            if entity_id:
                actor.setdefault("id", entity_id)
            ent = entity_by_id.get(entity_id)
            if not ent:
                expanded.append(actor)
                continue

            if "trigger" not in actor and ent.get("trigger") is not None:
                actor["trigger"] = ent.get("trigger")
            if "action" not in actor and ent.get("action") is not None:
                actor["action"] = ent.get("action")

            try:
                qty = int(ent.get("quantity", 1))
            except Exception:
                qty = 1
            qty = max(1, qty)
            if qty <= 1:
                expanded.append(actor)
                continue

            seg_id = _segment_id_for_actor(actor)
            if seg_id is None or seg_id not in seg_by_id:
                print(
                    f"[WARNING] {entity_id}: missing seg_id for group expansion; "
                    f"expected {qty} actors, placing 1",
                    flush=True,
                )
                expanded.append(actor)
                continue

            placement = actor.get("placement", {})
            if isinstance(placement, dict):
                placement["seg_id"] = int(seg_id)
                actor["placement"] = placement

            spec = dict(ent)
            spec["id"] = entity_id
            spec["quantity"] = qty
            spec["group_pattern"] = _normalize_group_pattern(ent, qty)
            spec["category"] = actor.get("category", "static")
            spec["actor_kind"] = ent.get("actor_kind", "static_prop")
            spec["asset_id"] = actor.get("asset_id")

            chosen = CandidatePlacement(
                vehicle_num=0,
                segment_index=int(placement.get("segment_index", 1) or 1),
                seg_id=int(seg_id),
                s_along=float(placement.get("s_along", 0.5)),
                lateral_relation=str(placement.get("lateral_relation", "center") or "center"),
                x=0.0,
                y=0.0,
                yaw_deg=0.0,
                path_s_m=0.0,
                base_score=0.0,
            )
            expanded_children = expand_group_to_actors(actor, spec, chosen, seg_by_id)
            if len(expanded_children) < qty:
                print(
                    f"[WARNING] {entity_id}: expected {qty} actors but only got {len(expanded_children)} "
                    f"(group_pattern={spec.get('group_pattern')}, seg_id={seg_id})",
                    flush=True,
                )
            expanded.extend(expanded_children)
        actors = expanded
    _log_actor_trace("after_group_expansion_llm_anchor_mode", actors)

    actors_world: List[Dict[str, Any]] = []
    for a in actors:
        placement = a["placement"]
        tv = placement.get("target_vehicle")
        seg_idx = int(placement.get("segment_index", 1))  # 1-based
        s_along = float(placement["s_along"])
        lat_rel = placement["lateral_relation"]
        
        # Check if seg_id is directly specified (for opposite-lane NPCs)
        direct_seg_id = placement.get("seg_id")
        
        if direct_seg_id is not None:
            # Direct seg_id: use it directly without looking up picked paths
            seg_id = int(direct_seg_id)
            seg_pts = seg_by_id.get(seg_id)
        else:
            # Standard lookup via target_vehicle and picked paths
            # Find seg_id from the picked path signature order
            # vehicle_segments contains seg_id list in order via picked signature; easiest: pull from picked itself
            picked_entry = next((p for p in picked if p.get("vehicle") == tv), None)
            if not picked_entry:
                continue
            seg_ids = (picked_entry.get("signature", {}) or {}).get("segment_ids", [])
            if not isinstance(seg_ids, list) or seg_idx < 1 or seg_idx > len(seg_ids):
                continue
            seg_id = int(seg_ids[seg_idx - 1])

            seg_pts = seg_by_id.get(seg_id)
            if seg_pts is None:
                # fall back to polyline_sample if present
                segs_det = (picked_entry.get("signature", {}) or {}).get("segments_detailed", [])
                det = next((d for d in segs_det if int(d.get("seg_id", -1)) == seg_id), None)
                if det and isinstance(det.get("polyline_sample"), list) and det["polyline_sample"]:
                    seg_pts = np.array([[p["x"], p["y"]] for p in det["polyline_sample"]], dtype=float)

        if seg_pts is None:
            print(f"[WARNING] Missing segment geometry for seg_id={seg_id}; skipping actor {a.get('id')}")
            continue

        spawn = compute_spawn_from_anchor(seg_pts, s_along, lat_rel, placement.get("lateral_offset_m"))
        motion = a.get("motion", {}) if isinstance(a.get("motion", {}), dict) else {"type": "static"}
        # Let motion builder know anchor s for some types
        motion.setdefault("anchor_s_along", s_along)
        
        # Pass start_lateral for crossing direction inference
        motion.setdefault("start_lateral", lat_rel)

        # category normalization
        cat = str(a.get("category", "")).lower()
        if cat not in ("vehicle", "walker", "static", "cyclist"):
            # derive from asset category if needed
            # (we don't strictly enforce this)
            cat = "static"

        if cat == "walker" and str((motion or {}).get("type", "")).lower() == "cross_perpendicular":
            anchor_s = float(motion.get("anchor_s_along", s_along))
            road_bounds = _estimate_road_lateral_bounds(
                seg_pts=seg_pts,
                anchor_s=anchor_s,
                lane_samples=sidewalk_lane_samples,
            )
            if road_bounds is not None:
                # Spawn just outside outermost lane edge and cross through to opposite side.
                edge_min, edge_max = road_bounds
                edge_clearance_m = max(0.5, float(motion.get("edge_clearance_m", 0.5) or 0.5))
                motion["road_lat_min_m"] = float(edge_min)
                motion["road_lat_max_m"] = float(edge_max)
                motion["edge_clearance_m"] = float(edge_clearance_m)

                road_width = max(0.0, float(edge_max - edge_min))
                required_cross = road_width + 2.0 * edge_clearance_m
                prev_cross = float(motion.get("cross_distance_m", 0.0) or 0.0)
                motion["cross_distance_m"] = max(prev_cross, required_cross)

                prev_offset = float(motion.get("start_offset_m", 0.0) or 0.0)
                required_start = max(abs(edge_min), abs(edge_max)) + edge_clearance_m
                motion["start_offset_m"] = max(prev_offset, required_start)
            else:
                estimated_offset = _estimate_sidewalk_offset_from_lanes(
                    seg_pts=seg_pts,
                    anchor_s=anchor_s,
                    lane_samples=sidewalk_lane_samples,
                )
                prev_offset = float(motion.get("start_offset_m", 0.0) or 0.0)
                motion["start_offset_m"] = max(prev_offset, estimated_offset)

        wps = build_motion_waypoints(motion, cat, spawn, seg_pts)
        if cat == "walker" and str((motion or {}).get("type", "")).lower() == "cross_perpendicular":
            placement_obj = a.get("placement", {}) if isinstance(a.get("placement"), dict) else {}
            trigger_obj = a.get("trigger", {}) if isinstance(a.get("trigger"), dict) else {}
            target_vehicle = str(trigger_obj.get("vehicle", "")).strip() or str(placement_obj.get("target_vehicle", "")).strip()
            fixed_wps, fixed_motion, spawn_clearance = _stabilize_walker_crossing(
                category=cat,
                motion=motion,
                anchor_spawn=spawn,
                seg_pts=seg_pts,
                waypoints=wps,
                ego_cloud=ego_path_cloud,
                lane_cloud=lane_center_cloud,
            )
            if fixed_wps is not wps:
                motion = fixed_motion
                wps = fixed_wps
                print(
                    f"[INFO] Adjusted walker crossing for sidewalk clearance vs {target_vehicle or 'ego path'} (min dist: {spawn_clearance:.2f}m)",
                    flush=True,
                )
        if isinstance(wps, list) and wps and isinstance(wps[0], dict) and "x" in wps[0] and "y" in wps[0]:
            # Align spawn to the first waypoint so spawn == trajectory start.
            spawn = {
                "x": float(wps[0]["x"]),
                "y": float(wps[0]["y"]),
                "yaw_deg": float(wps[0].get("yaw_deg", spawn.get("yaw_deg", 0.0))),
            }

        actors_world.append({
            **a,
            "resolved": {
                "seg_id": seg_id,
                "nodes_path": resolved_nodes_path,
            },
            "spawn": spawn,
            "world_waypoints": wps,
        })
    _log_actor_trace("after_anchor_to_world", actors_world)
    print(f"[TIMING] anchor -> world conversion: {time.time() - t0:.2f}s", flush=True)

    # Enforce non-overlapping spawns using asset bounding boxes.
    def _approx_width(a: Dict[str, Any]) -> float:
        bbox = get_asset_bbox(str(a.get("asset_id", "")))
        if bbox and bbox.width and bbox.width > 0:
            return float(bbox.width)
        cat = str(a.get("category", "")).lower()
        # Fallback widths (meters)
        if cat == "vehicle":
            return 1.8
        if cat == "cyclist":
            return 0.6
        if cat == "walker":
            return 0.5
        return 0.4

    def _total_length(seg_pts: np.ndarray) -> float:
        if not isinstance(seg_pts, np.ndarray) or len(seg_pts) < 2:
            return 0.0
        diffs = seg_pts[1:] - seg_pts[:-1]
        return float(np.linalg.norm(diffs, axis=1).sum())

    def _reposition_forward(idx: int, meters: float) -> bool:
        """Shift actor forward along its segment by given meters; recompute spawn/waypoints."""
        try:
            actor = actors_world[idx]
            seg_id = int((actor.get("resolved") or {}).get("seg_id", -1))
            seg_pts = seg_by_id.get(seg_id)
            if seg_pts is None:
                return False
            total = _total_length(seg_pts)
            s_delta = meters / max(total, 1e-6)
            motion = actor.get("motion", {}) if isinstance(actor.get("motion", {}), dict) else {"type": "static"}
            # Increase anchor position modestly
            anchor_s = float(motion.get("anchor_s_along", 0.5))
            new_s = min(0.98, max(0.0, anchor_s + s_delta))
            motion["anchor_s_along"] = new_s
            lateral = str(motion.get("start_lateral", "center") or "center").lower()
            spawn = compute_spawn_from_anchor(seg_pts, new_s, lateral)
            cat = str(actor.get("category", "")).lower()
            wps = build_motion_waypoints(motion, cat, spawn, seg_pts)
            if isinstance(wps, list) and wps and isinstance(wps[0], dict):
                actor["spawn"] = {
                    "x": float(wps[0]["x"]),
                    "y": float(wps[0]["y"]),
                    "yaw_deg": float(wps[0].get("yaw_deg", spawn.get("yaw_deg", 0.0))),
                }
                actor["world_waypoints"] = wps
                actor["motion"] = motion
                return True
        except Exception:
            return False
        return False

    changed = True
    passes = 0
    while changed and passes < 5:
        changed = False
        passes += 1
        for i in range(len(actors_world)):
            ai = actors_world[i]
            xi, yi = float(ai.get("spawn", {}).get("x", 0.0)), float(ai.get("spawn", {}).get("y", 0.0))
            wi = _approx_width(ai)
            ri = wi * 0.5
            for j in range(i + 1, len(actors_world)):
                aj = actors_world[j]
                xj, yj = float(aj.get("spawn", {}).get("x", 0.0)), float(aj.get("spawn", {}).get("y", 0.0))
                wj = _approx_width(aj)
                rj = wj * 0.5
                d = math.hypot(xi - xj, yi - yj)
                threshold = ri + rj + 0.2  # small margin
                if d < threshold:
                    # Push the later actor forward along its segment by the overlap amount
                    push_m = max(0.5, threshold - d)
                    if _reposition_forward(j, push_m):
                        changed = True
                        # Update positions after move
                        xj, yj = float(actors_world[j]["spawn"]["x"]), float(actors_world[j]["spawn"]["y"])
    if passes > 0:
        print(f"[INFO] Non-overlap enforcement passes: {passes}")
    _log_actor_trace("after_non_overlap", actors_world, note=f"passes={passes}")

    # Post-process moving NPC triggers to align with intersecting/closest ego path
    actors_world = _adjust_triggers_by_paths(actors_world, picked, seg_by_id)
    _log_actor_trace("after_trigger_adjust", actors_world)

    # Optionally strip static props if category disallows them
    if not allow_static_props:
        before = len(actors_world)
        actors_world = [
            a for a in actors_world
            if str(a.get("category", "")).lower() not in ("static",)
            and str(a.get("actor_kind", "")).lower() not in ("static_prop",)
            and str((a.get("motion", {}) or {}).get("type", "unknown")).lower() not in ("static", "unknown")
        ]
        removed = before - len(actors_world)
        if removed > 0:
            print(f"[INFO] Removed {removed} static prop(s) because allow_static_props=False", flush=True)
        _log_actor_trace("after_strip_static_props", actors_world, note=f"removed={removed}")

    # Validation: Check if total placed actors matches expected from Stage 1
    stage1_entities = stage1_obj.get("entities", [])
    total_expected = sum(int(e.get("quantity", 1)) for e in stage1_entities)
    total_placed = len(actors_world)
    if total_placed < total_expected:
        print(f"[WARNING] VALIDATION FAILED: Stage 1 expected {total_expected} total actors "
              f"but only {total_placed} were placed. Check group expansion logs above.", flush=True)
        for e in stage1_entities:
            qty = int(e.get("quantity", 1))
            if qty > 1:
                eid = e.get("entity_id", "unknown")
                # Count how many actors have this entity as base
                placed_for_entity = sum(1 for a in actors_world if str(a.get("id", "")).startswith(eid))
                if placed_for_entity < qty:
                    print(f"  - {eid}: expected {qty}, placed {placed_for_entity}", flush=True)

    # Verify actors maintain minimum buffer from ego spawns (buffer enforced in CSP during placement)
    MIN_BUFFER_TO_ACTORS_M = 15.0  # Minimum buffer from ego spawn to any actor
    if ego_spawns:
        violations = []
        for actor in actors_world:
            actor_spawn = actor.get("spawn", {})
            if not isinstance(actor_spawn, dict):
                continue
            ax = actor_spawn.get("x")
            ay = actor_spawn.get("y")
            if ax is None or ay is None:
                continue
            
            for ego in ego_spawns:
                ego_spawn = ego.get("spawn", {})
                ex = ego_spawn.get("x")
                ey = ego_spawn.get("y")
                if ex is None or ey is None:
                    continue
                
                dist = math.hypot(float(ax) - float(ex), float(ay) - float(ey))
                if dist < MIN_BUFFER_TO_ACTORS_M:
                    actor_id = actor.get("id", "actor_?")
                    vehicle_id = ego.get("vehicle", "Vehicle?")
                    violations.append({
                        "actor": actor_id,
                        "vehicle": vehicle_id,
                        "distance": dist,
                        "required": MIN_BUFFER_TO_ACTORS_M
                    })
        
        if violations:
            print(f"[ERROR] {len(violations)} actor(s) violate ego spawn buffer constraint:", flush=True)
            for v in violations:
                print(f"  {v['actor']} only {v['distance']:.1f}m from {v['vehicle']} (min {v['required']}m)", flush=True)
    _log_actor_trace("final_before_write", actors_world)

    out_payload = {
        "source_picked_paths": args.picked_paths,
        "nodes": resolved_nodes_path,
        "crop_region": crop_region,
        "ego_picked": picked,
        "ego_spawns": ego_spawns,
        "actors": actors_world,
        "macro_plan": stage1_obj.get("entities", []),
        "actor_trace": actor_trace,
    }

    t0 = time.time()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"[INFO] Wrote scene objects to: {args.out} (actors={len(actors_world)})")

    if args.viz:
        t_viz = time.time()
        # Get scenario_spec from args if available (passed from pipeline_runner)
        scenario_spec = getattr(args, "scenario_spec", None)
        visualize(
            picked=picked,
            seg_by_id=seg_by_id,
            actors_world=actors_world,
            crop_region=crop_region if isinstance(crop_region, dict) else None,
            out_path=args.viz_out,
            description=args.description,
            show=args.viz_show,
            scenario_spec=scenario_spec,
            macro_plan=out_payload.get("macro_plan"),
        )
        print(f"[TIMING] visualization: {time.time() - t_viz:.2f}s", flush=True)
    print(f"[TIMING] object_placer total (internal): {time.time() - t_obj_start:.2f}s", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--picked-paths", required=True, help="picked_paths_detailed.json")
    ap.add_argument("--carla-assets", required=True, help="carla_assets.json")

    ap.add_argument("--description", required=True, help="Natural-language scene description")
    ap.add_argument("--out", default="scene_objects.json", help="Output IR + placements JSON")
    ap.add_argument("--viz-out", default="scene_objects.png", help="Output visualization image")
    ap.add_argument("--viz", action="store_true", help="Enable visualization")
    ap.add_argument("--viz-show", action="store_true", help="Show plot window (if supported)")

    ap.add_argument("--nodes-root", default=None, help="Optional root to resolve relative nodes path")
    ap.add_argument("--placement-mode", default="csp", choices=["csp","llm_anchor"], help="Placement stage: weighted CSP (solver) or legacy LLM anchors")

    # LLM gen controls
    ap.add_argument("--max-new-tokens", type=int, default=1200)
    ap.add_argument("--do-sample", action="store_true", default=True, help="Enable sampling (default: True)")
    ap.add_argument("--no-sample", dest="do_sample", action="store_false", help="Disable sampling")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--top-p", type=float, default=0.95)

    args = ap.parse_args()
    run_object_placer(args)


if __name__ == "__main__":
    main()
