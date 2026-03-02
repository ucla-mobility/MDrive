#!/usr/bin/env python3
"""
Diagnostic tool to measure the 4 failure class metrics.

Failure classes:
  FC1: Lane instability - A→B→A CCLI oscillations per actor
  FC2: Illegal intersection spawn - actors whose frame-0 is inside
       a junction zone but had an approach episode fabricated
  FC3: Duplicate actors - moving vehicle pairs with strong overlap
  FC4: Invalid outermost classification - actors marked edge-static
       but their CCLI maps to a non-outermost lane
"""
from __future__ import annotations

import sys
import os
import math
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── Helpers ────────────────────────────────────────────────────────────────

def _safe_int(v, default=0) -> int:
    try:
        if v is None: return int(default)
        return int(v)
    except Exception:
        return int(default)

def _safe_float(v, default=float('nan')) -> float:
    try:
        if v is None: return float(default)
        return float(v)
    except Exception:
        return float(default)

def _normalize_yaw_deg(d: float) -> float:
    d = float(d)
    while d > 180.0: d -= 360.0
    while d < -180.0: d += 360.0
    return d

# ─── FC1: Lane Instability ───────────────────────────────────────────────────

def _count_aba_oscillations(ccli_seq: List[int]) -> int:
    """Count A→B→A oscillations in a CCLI sequence."""
    count = 0
    for i in range(2, len(ccli_seq)):
        a, b, c = ccli_seq[i-2], ccli_seq[i-1], ccli_seq[i]
        if a >= 0 and b >= 0 and c >= 0 and a == c and a != b:
            count += 1
    return count

def _count_ccli_changes(ccli_seq: List[int]) -> int:
    """Count total CCLI changes (any transition)."""
    n = 0
    for i in range(1, len(ccli_seq)):
        if ccli_seq[i-1] >= 0 and ccli_seq[i] >= 0 and ccli_seq[i-1] != ccli_seq[i]:
            n += 1
    return n

def measure_fc1(tracks_data: List[Dict]) -> Dict:
    """Measure lane instability across all tracks."""
    total_aba = 0
    total_changes = 0
    unstable_actors = []
    for track in tracks_data:
        actor_id = str(track.get("id", "?"))
        frames = track.get("frames", [])
        ccli_seq = [_safe_int(f.get("ccli"), -1) for f in frames]
        aba = _count_aba_oscillations(ccli_seq)
        changes = _count_ccli_changes(ccli_seq)
        total_aba += aba
        total_changes += changes
        if aba > 0 or changes > 2:  # >2 changes is suspicious
            unstable_actors.append({
                "id": actor_id,
                "aba_oscillations": aba,
                "total_changes": changes,
                "frames": len(frames),
            })
    return {
        "total_aba_oscillations": total_aba,
        "total_ccli_changes": total_changes,
        "unstable_actors": unstable_actors,
    }

# ─── FC2: Illegal Intersection Spawn ────────────────────────────────────────

def _line_length_m(line_data: Dict) -> float:
    """Compute length of a CARLA polyline in meters."""
    pts = line_data.get("poly_xy")
    if pts is None:
        return 0.0
    if isinstance(pts, np.ndarray):
        arr = pts
    else:
        try:
            arr = np.array(pts, dtype=np.float64)
        except Exception:
            return 0.0
    if arr.ndim != 2 or arr.shape[0] < 2:
        return 0.0
    diffs = np.diff(arr, axis=0)
    return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))

def measure_fc2(
    tracks_data: List[Dict],
    carla_feats: Dict[int, Dict],
    connector_line_threshold_m: float = 55.0,
) -> Dict:
    """
    Measure illegal intersection spawns.

    An illegal spawn is when:
    - Track's first frame has ccli assigned to a short (connector) line
    - Track has an episode_committed or synthetic_turn later
    - AND episode_start == 0 (approach was fabricated from frame 0)
    """
    violations = []
    for track in tracks_data:
        actor_id = str(track.get("id", "?"))
        frames = track.get("frames", [])
        if not frames:
            continue
        f0 = frames[0]
        f0_ccli = _safe_int(f0.get("ccli"), -1)
        if f0_ccli < 0:
            continue

        # Check if first frame CCLI is on a connector (short line)
        line_data = carla_feats.get(f0_ccli, {})
        line_len = _line_length_m(line_data)
        is_connector = bool(line_len > 0 and line_len < connector_line_threshold_m)

        # Check for fabricated approach: episode_start == 0
        has_fabricated_approach = False
        has_illegal_cross = False
        episode_start_frame = -1

        # Look for episode markers in early frames
        for fi, f in enumerate(frames[:20]):  # check first 20 frames
            if bool(f.get("intersection_episode_committed", False)):
                ep_start = _safe_int(f.get("intersection_episode_start"), -1)
                if ep_start == 0:
                    has_fabricated_approach = True
                    episode_start_frame = fi
                break

        # Check for illegal lane cross: frame with synthetic_turn right at spawn start
        # AND predecessor frame on different road
        for fi in range(1, min(10, len(frames))):
            f_prev = frames[fi - 1]
            f_cur = frames[fi]
            prev_ccli = _safe_int(f_prev.get("ccli"), -1)
            cur_ccli = _safe_int(f_cur.get("ccli"), -1)
            if prev_ccli >= 0 and cur_ccli >= 0 and prev_ccli != cur_ccli:
                prev_line = carla_feats.get(prev_ccli, {})
                cur_line = carla_feats.get(cur_ccli, {})
                prev_road = _safe_int(prev_line.get("road_id"), -1)
                cur_road = _safe_int(cur_line.get("road_id"), -1)
                if prev_road >= 0 and cur_road >= 0 and prev_road != cur_road:
                    has_illegal_cross = True
                    break

        if is_connector and has_fabricated_approach:
            violations.append({
                "id": actor_id,
                "f0_ccli": f0_ccli,
                "f0_line_len_m": round(line_len, 2),
                "episode_start_frame": episode_start_frame,
                "has_illegal_cross": has_illegal_cross,
            })
        elif is_connector and has_illegal_cross:
            violations.append({
                "id": actor_id,
                "f0_ccli": f0_ccli,
                "f0_line_len_m": round(line_len, 2),
                "episode_start_frame": episode_start_frame,
                "has_illegal_cross": has_illegal_cross,
            })

    return {
        "illegal_spawn_count": len(violations),
        "violations": violations,
    }

# ─── FC3: Duplicate Actors ───────────────────────────────────────────────────

def measure_fc3(
    tracks_data: List[Dict],
    overlap_ratio_threshold: float = 0.30,
    dist_threshold_m: float = 2.5,
    min_common_frames: int = 8,
) -> Dict:
    """
    Measure duplicate actor pairs by spatiotemporal overlap.

    A pair is a duplicate when:
    - >= min_common_frames temporal overlap
    - >= overlap_ratio_threshold (each track)
    - median spatial distance < dist_threshold_m
    """
    # Build tick -> {actor_id: (x, y)} map from CARLA poses
    tick_to_poses: Dict[int, Dict[str, Tuple[float, float]]] = defaultdict(dict)
    actor_ticks: Dict[str, Set[int]] = {}

    for track in tracks_data:
        actor_id = str(track.get("id", "?"))
        role = str(track.get("role", "vehicle"))
        if not role.startswith("vehicle") and not role.startswith("bicycle"):
            continue
        frames = track.get("frames", [])
        ticks: Set[int] = set()
        for f in frames:
            tk = _safe_int(f.get("tick_key"), _safe_int(f.get("tick"), -1))
            if tk < 0:
                continue
            cx = _safe_float(f.get("cx"), float("nan"))
            cy = _safe_float(f.get("cy"), float("nan"))
            if math.isfinite(cx) and math.isfinite(cy):
                tick_to_poses[tk][actor_id] = (cx, cy)
                ticks.add(tk)
        if ticks:
            actor_ticks[actor_id] = ticks

    ids = list(actor_ticks.keys())
    dup_pairs = []

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            ia, ib = ids[i], ids[j]
            common = actor_ticks[ia] & actor_ticks[ib]
            if len(common) < min_common_frames:
                continue
            ra = len(common) / max(1, len(actor_ticks[ia]))
            rb = len(common) / max(1, len(actor_ticks[ib]))
            if min(ra, rb) < overlap_ratio_threshold:
                continue
            dists = []
            for tk in common:
                pa = tick_to_poses[tk].get(ia)
                pb = tick_to_poses[tk].get(ib)
                if pa and pb:
                    dists.append(math.hypot(pa[0] - pb[0], pa[1] - pb[1]))
            if not dists:
                continue
            med_dist = float(np.median(dists))
            if med_dist < dist_threshold_m:
                dup_pairs.append({
                    "actor_a": ia,
                    "actor_b": ib,
                    "common_frames": len(common),
                    "ratio_a": round(ra, 3),
                    "ratio_b": round(rb, 3),
                    "median_dist_m": round(med_dist, 3),
                })

    return {
        "duplicate_pairs_count": len(dup_pairs),
        "duplicate_pairs": dup_pairs,
    }

# ─── FC4: Invalid Outermost Classification ───────────────────────────────────

def _build_outermost_map(carla_feats: Dict[int, Dict]) -> Dict[int, bool]:
    """Build topology-based outermost map: is each CARLA line on the outermost lane of its road?"""
    road_side_max_abs_lane: Dict[Tuple[int, int], int] = {}
    line_road_id: Dict[int, int] = {}
    line_lane_id: Dict[int, int] = {}

    for ci, row in carla_feats.items():
        if not isinstance(row, dict):
            continue
        rid = _safe_int(row.get("road_id"), 0)
        lid = _safe_int(row.get("lane_id"), 0)
        line_road_id[ci] = rid
        line_lane_id[ci] = lid
        if lid == 0:
            continue
        side = 1 if lid > 0 else -1
        key = (rid, side)
        cur = road_side_max_abs_lane.get(key, 0)
        if abs(lid) > cur:
            road_side_max_abs_lane[key] = abs(lid)

    outermost: Dict[int, bool] = {}
    for ci, rid in line_road_id.items():
        lid = line_lane_id.get(ci, 0)
        if lid == 0:
            outermost[ci] = False
            continue
        side = 1 if lid > 0 else -1
        max_abs = road_side_max_abs_lane.get((rid, side), abs(lid))
        outermost[ci] = bool(abs(lid) >= max_abs)
    return outermost

def measure_fc4(
    tracks_data: List[Dict],
    carla_feats: Dict[int, Dict],
) -> Dict:
    """
    Measure invalid outermost classifications.

    A violation is when:
    - Actor has edge_static or parked placement (edge-like)
    - Modal CCLI is NOT outermost per topology
    """
    outermost_map = _build_outermost_map(carla_feats)
    violations = []

    for track in tracks_data:
        actor_id = str(track.get("id", "?"))
        frames = track.get("frames", [])
        if not frames:
            continue

        # Check if actor is parked/edge-static classified:
        # (a) any frame has a parked-invariant or parked-edge csource, OR
        # (b) track-level low_motion_vehicle flag is set (pipeline parked heuristic)
        _parked_csource_prefixes = (
            "parked_invariant_",
            "parked_edge_",
            "parked_static_",
            "parked_low_motion",
        )
        has_edge_marker = any(
            str(f.get("csource", "")).startswith(_parked_csource_prefixes)
            for f in frames
        )
        track_is_parked = bool(track.get("low_motion_vehicle", False))

        if not (has_edge_marker or track_is_parked):
            continue

        # Find modal CCLI
        ccli_counts: Dict[int, int] = defaultdict(int)
        for f in frames:
            cli = _safe_int(f.get("ccli"), -1)
            if cli >= 0:
                ccli_counts[cli] += 1
        if not ccli_counts:
            continue
        modal_cli = max(ccli_counts, key=lambda k: ccli_counts[k])

        is_outer = outermost_map.get(modal_cli, True)  # default True if unknown
        if not is_outer:
            line_data = carla_feats.get(modal_cli, {})
            violations.append({
                "id": actor_id,
                "modal_ccli": modal_cli,
                "road_id": _safe_int(line_data.get("road_id"), -1),
                "lane_id": _safe_int(line_data.get("lane_id"), 0),
                "outermost": is_outer,
            })

    return {
        "invalid_outermost_count": len(violations),
        "violations": violations,
    }

# ─── FC3 Handoff: terminal duplicate detection ───────────────────────────────

def measure_fc3_handoff(
    tracks_data: List[Dict],
    max_gap_s: float = 1.5,
    max_dist_m: float = 2.0,
) -> Dict:
    """
    Detect terminal-handoff duplicate pairs.

    A pair (A, B) is a handoff duplicate when:
    - Same role (both vehicle/bicycle)
    - One track's last timestamp is within max_gap_s of the other's first timestamp
    - Spatial distance between that endpoint and startpoint < max_dist_m
    """
    # Collect vehicle tracks with their time-ordered (t, x, y) tuples
    vehicle_tracks: List[Tuple[str, float, float, float, float, float, float]] = []
    # Each entry: (id, t_start, x_start, y_start, t_end, x_end, y_end)

    for track in tracks_data:
        actor_id = str(track.get("id", "?"))
        role = str(track.get("role", "vehicle"))
        if not role.startswith("vehicle") and not role.startswith("bicycle"):
            continue
        frames = track.get("frames", [])
        if not frames:
            continue
        # Use world position (x, y) - same coordinate space for all tracks
        t_vals = [_safe_float(f.get("t"), float("nan")) for f in frames]
        x_vals = [_safe_float(f.get("x"), float("nan")) for f in frames]
        y_vals = [_safe_float(f.get("y"), float("nan")) for f in frames]
        valid = [(t, x, y) for t, x, y in zip(t_vals, x_vals, y_vals)
                 if math.isfinite(t) and math.isfinite(x) and math.isfinite(y)]
        if not valid:
            continue
        valid.sort(key=lambda v: v[0])
        t_s, x_s, y_s = valid[0]
        t_e, x_e, y_e = valid[-1]
        vehicle_tracks.append((actor_id, t_s, x_s, y_s, t_e, x_e, y_e))

    handoff_pairs = []
    for i in range(len(vehicle_tracks)):
        id_a, ts_a, xs_a, ys_a, te_a, xe_a, ye_a = vehicle_tracks[i]
        for j in range(i + 1, len(vehicle_tracks)):
            id_b, ts_b, xs_b, ys_b, te_b, xe_b, ye_b = vehicle_tracks[j]
            # Check a ends → b starts
            gap_ab = float(ts_b) - float(te_a)
            if abs(gap_ab) <= float(max_gap_s):
                dist_ab = math.hypot(float(xe_a) - float(xs_b), float(ye_a) - float(ys_b))
                if dist_ab <= float(max_dist_m):
                    handoff_pairs.append({
                        "actor_a": id_a,
                        "actor_b": id_b,
                        "gap_s": round(gap_ab, 3),
                        "endpoint_dist_m": round(dist_ab, 3),
                        "direction": "a_end->b_start",
                    })
                    continue
            # Check b ends → a starts
            gap_ba = float(ts_a) - float(te_b)
            if abs(gap_ba) <= float(max_gap_s):
                dist_ba = math.hypot(float(xe_b) - float(xs_a), float(ye_b) - float(ys_a))
                if dist_ba <= float(max_dist_m):
                    handoff_pairs.append({
                        "actor_a": id_a,
                        "actor_b": id_b,
                        "gap_s": round(gap_ba, 3),
                        "endpoint_dist_m": round(dist_ba, 3),
                        "direction": "b_end->a_start",
                    })

    return {
        "handoff_pairs_count": len(handoff_pairs),
        "handoff_pairs": handoff_pairs,
    }


# ─── Pipeline Runner ─────────────────────────────────────────────────────────

def run_pipeline_and_extract(scenario_dir: Path, map_pkl: Path, carla_cache: Path, offset_json: Path) -> Optional[Dict]:
    """Run the pipeline on a scenario and extract track data."""
    import io, contextlib
    from v2xpnp.pipeline import pipeline_runtime

    # Capture output by calling the main pipeline function
    try:
        result = pipeline_runtime.run_scenario(
            scenario_dir=scenario_dir,
            map_pkl=map_pkl,
            carla_map_cache=carla_cache,
            carla_map_offset_json=offset_json,
            return_internal=True,
        )
        return result
    except (AttributeError, TypeError):
        pass

    # Fallback: use stage runner directly
    try:
        from v2xpnp.pipeline.stages import run_all_stages
        result = run_all_stages(
            scenario_dir=str(scenario_dir),
            map_pkl=str(map_pkl),
            carla_cache=str(carla_cache),
            offset_json=str(offset_json),
        )
        return result
    except Exception as e:
        print(f"[ERROR] Could not run pipeline internally: {e}")
        return None


def load_carla_feats(carla_cache: Path, offset_json: Path) -> Dict[int, Dict]:
    """Load CARLA polyline features from cache, returning {line_index: record_dict}."""
    from v2xpnp.pipeline.route_export_stage_01_foundation import _load_carla_map_cache
    try:
        # _load_carla_map_cache returns (lines, bounds, map_name, line_records)
        result = _load_carla_map_cache(Path(carla_cache))
        if isinstance(result, (list, tuple)) and len(result) == 4:
            line_records = result[3]
            if isinstance(line_records, list):
                return {i: dict(row) for i, row in enumerate(line_records) if isinstance(row, dict)}
    except Exception:
        pass

    # Direct pickle load fallback
    with open(carla_cache, "rb") as f:
        raw = pickle.load(f)
    if isinstance(raw, dict):
        # Top-level dict has keys like 'lines', 'line_records', etc.
        line_records = raw.get("line_records")
        if isinstance(line_records, list):
            return {i: dict(row) for i, row in enumerate(line_records) if isinstance(row, dict)}
        # Try interpreting direct keys as integer indices
        out: Dict[int, Dict] = {}
        for k, v in raw.items():
            try:
                out[int(k)] = v
            except (ValueError, TypeError):
                pass
        return out
    if isinstance(raw, list):
        return {i: row for i, row in enumerate(raw) if isinstance(row, dict)}
    return {}


def extract_tracks_from_html(html_path: Path) -> List[Dict]:
    """Extract track data from pipeline HTML output via embedded JSON."""
    import json
    import re
    text = html_path.read_text(encoding="utf-8")

    # Primary: parse <script id="dataset" type="application/json"> tag
    m = re.search(
        r'<script\s[^>]*id=["\']dataset["\'][^>]*>(.*?)</script>',
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        try:
            data = json.loads(m.group(1))
            if isinstance(data, list):
                return data
            return data.get("tracks", [])
        except Exception as e:
            print(f"[WARN] JSON parse (dataset tag): {e}")

    # Fallback: look for const DATA / window.PIPELINE_DATA assignments
    for pattern in (
        r'const DATA\s*=\s*(\{.*?\});',
        r'window\.PIPELINE_DATA\s*=\s*(\{.*?\});',
    ):
        m2 = re.search(pattern, text, re.DOTALL)
        if m2:
            try:
                data = json.loads(m2.group(1))
                return data.get("tracks", [])
            except Exception as e:
                print(f"[WARN] JSON parse (fallback): {e}")
    return []


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Measure 4 failure class metrics.")
    parser.add_argument("--scenarios", nargs="+", required=True, help="Scenario directories")
    parser.add_argument("--map-pkl", required=True, type=Path)
    parser.add_argument("--carla-map-cache", required=True, type=Path)
    parser.add_argument("--carla-map-offset-json", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/metrics_output"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load CARLA features once
    print("[INFO] Loading CARLA features...")
    carla_feats = load_carla_feats(args.carla_map_cache, args.carla_map_offset_json)
    print(f"[INFO] Loaded {len(carla_feats)} CARLA lines")
    outermost_map = _build_outermost_map(carla_feats)
    outer_count = sum(1 for v in outermost_map.values() if v)
    print(f"[INFO] Outermost lines: {outer_count}/{len(outermost_map)}")

    print("\n" + "="*80)
    print("FAILURE METRICS TABLE")
    print("="*80)
    print(f"{'Scenario':<40} {'FC1-ABA':>7} {'FC1-CHG':>7} {'FC2-SPW':>7} {'FC3-HND':>7} {'FC4-OTM':>7}")
    print("-"*80)

    total_fc1_aba = 0
    total_fc1_chg = 0
    total_fc2 = 0
    total_fc3_hnd = 0
    total_fc4 = 0

    for scenario_str in args.scenarios:
        scenario_dir = Path(scenario_str)
        name = scenario_dir.name[:39]

        # Run pipeline
        out_html = args.out_dir / f"{scenario_dir.name}.html"
        cmd = (
            f"python3 -m v2xpnp.pipeline.entrypoint "
            f"'{scenario_dir}' "
            f"--map-pkl '{args.map_pkl}' "
            f"--carla-map-cache '{args.carla_map_cache}' "
            f"--carla-map-offset-json '{args.carla_map_offset_json}' "
            f"--out '{out_html}' 2>/dev/null"
        )
        os.system(cmd)

        if not out_html.exists():
            print(f"{name:<40} {'ERR':>7} {'ERR':>7} {'ERR':>7} {'ERR':>7} {'ERR':>7}")
            continue

        tracks = extract_tracks_from_html(out_html)
        if not tracks:
            print(f"{name:<40} {'NO DATA':>7}")
            continue

        r1 = measure_fc1(tracks)
        r2 = measure_fc2(tracks, carla_feats)
        r3h = measure_fc3_handoff(tracks)
        r4 = measure_fc4(tracks, carla_feats)

        fc1_aba = r1["total_aba_oscillations"]
        fc1_chg = r1["total_ccli_changes"]
        fc2 = r2["illegal_spawn_count"]
        fc3_hnd = r3h["handoff_pairs_count"]
        fc4 = r4["invalid_outermost_count"]

        total_fc1_aba += fc1_aba
        total_fc1_chg += fc1_chg
        total_fc2 += fc2
        total_fc3_hnd += fc3_hnd
        total_fc4 += fc4

        print(f"{name:<40} {fc1_aba:>7} {fc1_chg:>7} {fc2:>7} {fc3_hnd:>7} {fc4:>7}")

        # Print detail for failing metrics
        if fc1_aba > 0:
            for a in r1["unstable_actors"]:
                if a["aba_oscillations"] > 0:
                    print(f"  FC1 ABA: actor={a['id']} aba={a['aba_oscillations']} changes={a['total_changes']}")
        if fc2 > 0:
            for v in r2["violations"]:
                print(f"  FC2 spawn: actor={v['id']} ccli={v['f0_ccli']} len={v['f0_line_len_m']}m")
        if fc3_hnd > 0:
            for p in r3h["handoff_pairs"]:
                print(f"  FC3 hnd: {p['actor_a']} -> {p['actor_b']} gap={p['gap_s']}s dist={p['endpoint_dist_m']}m")
        if fc4 > 0:
            for v in r4["violations"]:
                print(f"  FC4 outer: actor={v['id']} ccli={v['modal_ccli']} road={v['road_id']} lane={v['lane_id']}")

    print("-"*80)
    print(f"{'TOTAL':<40} {total_fc1_aba:>7} {total_fc1_chg:>7} {total_fc2:>7} {total_fc3_hnd:>7} {total_fc4:>7}")
    print("="*80)
    print()
    print("FC1-ABA: A→B→A lane oscillations (goal: 0)")
    print("FC1-CHG: Total CCLI changes (goal: minimize, not zero)")
    print("FC2-SPW: Illegal intersection spawn fabrications (goal: 0)")
    print("FC3-HND: Terminal handoff duplicate pairs (goal: 0)")
    print("FC4-OTM: Invalid outermost classifications (goal: 0)")


if __name__ == "__main__":
    main()
