#!/usr/bin/env python3
"""
Intersection trajectory judge for the real->sim pipeline.

This script evaluates intersection windows at actor level and reports
hard-fail patterns that matter for trajectory shape quality:
  - goes_around_intersection
  - mode_flip (curve + straight in one actor/window)
  - off_raw (projected path too far from raw)
  - jump_spike (low-motion raw, large projected step)

It is designed to be used as a fast regression gate and as the scoring backend
for parameter sweeps.
"""

from __future__ import annotations

import argparse
import contextlib
import html
import io
import json
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v2xpnp.pipeline import route_export as ytm
from v2xpnp.pipeline.pipeline_runtime import (
    PROCESSING_PROFILE_CHOICES,
    _apply_processing_profile,
    _is_scenario_directory,
    _normalize_processing_profile_name,
    _parse_lane_type_set,
    load_vector_map,
    process_single_scenario,
)


FAIL_KEYS: Tuple[str, ...] = (
    "goes_around_intersection",
    "mode_flip",
    "off_raw",
    "jump_spike",
)

FAIL_WEIGHTS: Dict[str, float] = {
    "goes_around_intersection": 3.0,
    "mode_flip": 2.0,
    "off_raw": 2.0,
    "jump_spike": 3.0,
}


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _xy_pair(frame: Dict[str, object], x_key: str, y_key: str) -> Optional[Tuple[float, float]]:
    x = frame.get(x_key, None)
    y = frame.get(y_key, None)
    if x is None or y is None:
        return None
    xf = _safe_float(x, float("nan"))
    yf = _safe_float(y, float("nan"))
    if not (math.isfinite(xf) and math.isfinite(yf)):
        return None
    return (float(xf), float(yf))


def _raw_xy(frame: Dict[str, object]) -> Optional[Tuple[float, float]]:
    return _xy_pair(frame, "x", "y")


def _final_xy(frame: Dict[str, object]) -> Optional[Tuple[float, float]]:
    for kx, ky in (("cx", "cy"), ("sx", "sy"), ("x", "y")):
        pt = _xy_pair(frame, kx, ky)
        if pt is not None:
            return pt
    return None


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return 0.0
    idx = int(math.floor(0.95 * float(len(vals) - 1)))
    return float(vals[max(0, min(len(vals) - 1, idx))])


def _median(values: Sequence[float]) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return float("nan")
    return float(vals[len(vals) // 2])


def _path_length(points: Sequence[Tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(points)):
        total += float(
            math.hypot(
                float(points[i][0]) - float(points[i - 1][0]),
                float(points[i][1]) - float(points[i - 1][1]),
            )
        )
    return float(total)


def _is_intersectionish_frame(frame: Dict[str, object]) -> bool:
    if any(k in frame for k in ("intersection_shape_mode", "intersection_shape_window_start", "intersection_shape_window_end")):
        return True
    if bool(frame.get("semantic_transition_sustained", False)):
        return True
    cquality = str(frame.get("cquality", "")).strip().lower()
    if cquality == "intersection":
        return True
    csource = str(frame.get("csource", "")).strip().lower()
    if not csource:
        return False
    tokens = (
        "intersection",
        "turn_",
        "transition_window",
        "semantic_boundary",
        "semantic_line_id_hold",
        "direct_turn",
    )
    return any(tok in csource for tok in tokens)


def _merge_windows(
    windows: Sequence[Tuple[int, int]],
    n_frames: int,
    merge_gap_frames: int,
) -> List[Tuple[int, int]]:
    if not windows:
        return []
    clamped: List[Tuple[int, int]] = []
    for s, e in windows:
        s0 = max(0, min(int(n_frames - 1), int(s)))
        e0 = max(0, min(int(n_frames - 1), int(e)))
        if e0 < s0:
            s0, e0 = e0, s0
        clamped.append((int(s0), int(e0)))
    clamped.sort(key=lambda w: (int(w[0]), int(w[1])))

    merged: List[List[int]] = []
    for s, e in clamped:
        if not merged:
            merged.append([int(s), int(e)])
            continue
        prev = merged[-1]
        if int(s) <= int(prev[1]) + int(merge_gap_frames) + 1:
            prev[1] = max(int(prev[1]), int(e))
        else:
            merged.append([int(s), int(e)])
    return [(int(w[0]), int(w[1])) for w in merged]


def _collect_windows(
    frames: Sequence[Dict[str, object]],
    min_window_frames: int,
    event_pad_frames: int,
    merge_gap_frames: int,
) -> List[Tuple[int, int]]:
    n = int(len(frames))
    if n <= 0:
        return []

    windows: List[Tuple[int, int]] = []

    for fr in frames:
        if not isinstance(fr, dict):
            continue
        if "intersection_shape_window_start" in fr and "intersection_shape_window_end" in fr:
            s = _safe_int(fr.get("intersection_shape_window_start"), -1)
            e = _safe_int(fr.get("intersection_shape_window_end"), -1)
            if s >= 0 and e >= 0:
                windows.append((int(s) - int(event_pad_frames), int(e) + int(event_pad_frames)))

    marks: List[bool] = [_is_intersectionish_frame(fr) if isinstance(fr, dict) else False for fr in frames]
    i = 0
    while i < n:
        if not marks[i]:
            i += 1
            continue
        s = int(i)
        e = int(i)
        i += 1
        while i < n and marks[i]:
            e = int(i)
            i += 1
        windows.append((int(s) - int(event_pad_frames), int(e) + int(event_pad_frames)))

    merged = _merge_windows(windows=windows, n_frames=n, merge_gap_frames=int(merge_gap_frames))
    out: List[Tuple[int, int]] = []
    for s, e in merged:
        if int(e) - int(s) + 1 >= int(min_window_frames):
            out.append((int(s), int(e)))
    return out


def _window_assessment(
    frames: Sequence[Dict[str, object]],
    start_idx: int,
    end_idx: int,
    args: argparse.Namespace,
) -> Dict[str, object]:
    idxs = list(range(int(start_idx), int(end_idx) + 1))

    raw_pts: List[Tuple[float, float]] = []
    final_pts: List[Tuple[float, float]] = []
    raw_steps: List[float] = []
    final_steps: List[float] = []
    off_dists: List[float] = []
    mode_set: set[str] = set()
    source_counter: Counter[str] = Counter()

    prev_raw: Optional[Tuple[float, float]] = None
    prev_final: Optional[Tuple[float, float]] = None
    for fi in idxs:
        fr = frames[fi]
        if not isinstance(fr, dict):
            prev_raw = None
            prev_final = None
            continue
        rp = _raw_xy(fr)
        fp = _final_xy(fr)
        if rp is not None:
            raw_pts.append(rp)
        if fp is not None:
            final_pts.append(fp)
        if rp is not None and fp is not None:
            off_dists.append(float(math.hypot(float(fp[0]) - float(rp[0]), float(fp[1]) - float(rp[1]))))

        mode = str(fr.get("intersection_shape_mode", "")).strip().lower()
        if mode in {"curve", "straight"}:
            mode_set.add(mode)
        src = str(fr.get("csource", "")).strip()
        if src:
            source_counter[src] += 1

        if prev_raw is not None and rp is not None:
            raw_steps.append(float(math.hypot(float(rp[0]) - float(prev_raw[0]), float(rp[1]) - float(prev_raw[1]))))
        if prev_final is not None and fp is not None:
            final_steps.append(float(math.hypot(float(fp[0]) - float(prev_final[0]), float(fp[1]) - float(prev_final[1]))))

        prev_raw = rp
        prev_final = fp

    raw_path = _path_length(raw_pts)
    final_path = _path_length(final_pts)
    p95_off = _p95(off_dists)
    max_off = max(off_dists) if off_dists else 0.0
    detour_ratio = (float(final_path) / float(max(0.1, raw_path))) if raw_path > 0.0 else float("inf")

    anchor_gap = 0.0
    if raw_pts and final_pts:
        cx = float(sum(p[0] for p in raw_pts) / float(len(raw_pts)))
        cy = float(sum(p[1] for p in raw_pts) / float(len(raw_pts)))
        raw_min = min(float(math.hypot(float(p[0]) - cx, float(p[1]) - cy)) for p in raw_pts)
        final_min = min(float(math.hypot(float(p[0]) - cx, float(p[1]) - cy)) for p in final_pts)
        anchor_gap = float(final_min - raw_min)

    jump_spike = False
    pair_n = min(len(raw_steps), len(final_steps))
    for i in range(pair_n):
        if float(raw_steps[i]) < float(args.raw_step_threshold) and float(final_steps[i]) > float(args.jump_threshold):
            jump_spike = True
            break

    mode_flip = ("curve" in mode_set) and ("straight" in mode_set)
    off_raw = (float(p95_off) > float(args.off_raw_p95_threshold)) or (float(max_off) > float(args.off_raw_max_threshold))
    goes_around = (
        float(detour_ratio) > float(args.detour_ratio_threshold)
        and float(anchor_gap) > float(args.anchor_gap_threshold)
        and float(p95_off) > float(args.goes_around_min_p95_off)
        and float(raw_path) >= float(args.goes_around_min_raw_path)
    )

    fail_flags = {
        "goes_around_intersection": bool(goes_around),
        "mode_flip": bool(mode_flip),
        "off_raw": bool(off_raw),
        "jump_spike": bool(jump_spike),
    }
    fail_count = int(sum(1 for k in FAIL_KEYS if bool(fail_flags.get(k, False))))
    score = 0.0
    for k in FAIL_KEYS:
        if bool(fail_flags.get(k, False)):
            score += float(FAIL_WEIGHTS.get(k, 1.0))
    score += 0.35 * max(0.0, float(p95_off) - float(args.off_raw_p95_threshold))

    return {
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "frames": int(end_idx - start_idx + 1),
        "mode_set": sorted(mode_set),
        "p95_off_raw_m": float(p95_off),
        "max_off_raw_m": float(max_off),
        "raw_path_m": float(raw_path),
        "final_path_m": float(final_path),
        "detour_ratio": float(detour_ratio),
        "anchor_gap_m": float(anchor_gap),
        "fail_flags": {k: bool(fail_flags[k]) for k in FAIL_KEYS},
        "fail_count": int(fail_count),
        "score": float(score),
        "top_sources": [[str(k), int(v)] for k, v in source_counter.most_common(6)],
    }


def _count_aba_flickers(seq: Sequence[int]) -> int:
    n = len(seq)
    if n < 3:
        return 0
    out = 0
    for i in range(2, n):
        a = int(seq[i - 2])
        b = int(seq[i - 1])
        c = int(seq[i])
        if a >= 0 and b >= 0 and c >= 0 and a == c and a != b:
            out += 1
    return int(out)


def _track_assessment(track: Dict[str, object], args: argparse.Namespace) -> Optional[Dict[str, object]]:
    frames = track.get("frames", [])
    if not isinstance(frames, list) or len(frames) < 2:
        return None
    windows = _collect_windows(
        frames=frames,
        min_window_frames=int(args.min_window_frames),
        event_pad_frames=int(args.event_pad_frames),
        merge_gap_frames=int(args.merge_gap_frames),
    )
    if not windows:
        return None

    window_rows: List[Dict[str, object]] = []
    fail_type_counter: Counter[str] = Counter()
    hard_fail_windows = 0
    track_score = 0.0
    for s, e in windows:
        row = _window_assessment(frames=frames, start_idx=int(s), end_idx=int(e), args=args)
        window_rows.append(row)
        if int(row.get("fail_count", 0)) > 0:
            hard_fail_windows += 1
        ff = row.get("fail_flags", {})
        if isinstance(ff, dict):
            for k in FAIL_KEYS:
                if bool(ff.get(k, False)):
                    fail_type_counter[k] += 1
        track_score += float(row.get("score", 0.0))

    window_rows.sort(key=lambda r: (int(r.get("fail_count", 0)), float(r.get("score", 0.0))), reverse=True)
    out = {
        "track_id": str(track.get("id", "unknown")),
        "role": str(track.get("role", "")),
        "window_count": int(len(window_rows)),
        "hard_fail_windows": int(hard_fail_windows),
        "hard_fail_type_counts": {k: int(fail_type_counter.get(k, 0)) for k in FAIL_KEYS},
        "score": float(track_score),
        "top_windows": window_rows[: int(max(1, args.top_windows_per_track))],
    }
    if bool(args.emit_all_windows):
        out["all_windows"] = window_rows
    return out


def _scenario_assessment(dataset: Dict[str, object], args: argparse.Namespace) -> Dict[str, object]:
    map_name = str(dataset.get("map_name", ""))
    is_intersection_map = ("intersection" in map_name.lower())
    tracks = dataset.get("tracks", [])
    if not isinstance(tracks, list):
        tracks = []

    actor_rows: List[Dict[str, object]] = []
    fail_type_counter: Counter[str] = Counter()
    hard_fail_windows = 0
    total_windows = 0
    scenario_score = 0.0
    off_vals: List[float] = []
    jump_spike_windows = 0
    mode_flip_windows = 0

    global_raw_final_dists: List[float] = []
    global_final_steps: List[float] = []
    global_step_ratios: List[float] = []
    nonmonotonic_time_pairs = 0
    duplicate_time_pairs = 0
    aba_line_flickers = 0
    frame_samples = 0

    for tr in tracks:
        if not isinstance(tr, dict):
            continue
        role = str(tr.get("role", "")).strip().lower()
        if role != "vehicle":
            continue

        tr_frames = tr.get("frames", [])
        if isinstance(tr_frames, list) and tr_frames:
            line_seq: List[int] = []
            prev_t: Optional[float] = None
            prev_raw: Optional[Tuple[float, float]] = None
            prev_final: Optional[Tuple[float, float]] = None
            for fr in tr_frames:
                if not isinstance(fr, dict):
                    prev_t = None
                    prev_raw = None
                    prev_final = None
                    continue
                frame_samples += 1
                t = _safe_float(fr.get("t"), float("nan"))
                if prev_t is not None and math.isfinite(float(t)):
                    if float(t) < float(prev_t):
                        nonmonotonic_time_pairs += 1
                    elif abs(float(t) - float(prev_t)) <= 1e-9:
                        duplicate_time_pairs += 1
                if math.isfinite(float(t)):
                    prev_t = float(t)
                else:
                    prev_t = None

                raw_pt = _raw_xy(fr)
                final_pt = _final_xy(fr)
                if raw_pt is not None and final_pt is not None:
                    global_raw_final_dists.append(
                        float(
                            math.hypot(
                                float(final_pt[0]) - float(raw_pt[0]),
                                float(final_pt[1]) - float(raw_pt[1]),
                            )
                        )
                    )
                if prev_final is not None and final_pt is not None:
                    final_step = float(
                        math.hypot(
                            float(final_pt[0]) - float(prev_final[0]),
                            float(final_pt[1]) - float(prev_final[1]),
                        )
                    )
                    global_final_steps.append(float(final_step))
                    if prev_raw is not None and raw_pt is not None:
                        raw_step = float(
                            math.hypot(
                                float(raw_pt[0]) - float(prev_raw[0]),
                                float(raw_pt[1]) - float(prev_raw[1]),
                            )
                        )
                        ratio = float(final_step) / float(max(0.05, raw_step))
                        global_step_ratios.append(float(ratio))
                prev_raw = raw_pt
                prev_final = final_pt

                cli = _safe_int(fr.get("ccli"), -1)
                line_seq.append(int(cli))
            aba_line_flickers += int(_count_aba_flickers(line_seq))

        row = _track_assessment(track=tr, args=args)
        if row is None:
            continue
        actor_rows.append(row)
        total_windows += int(row.get("window_count", 0))
        hard_fail_windows += int(row.get("hard_fail_windows", 0))
        scenario_score += float(row.get("score", 0.0))
        for k in FAIL_KEYS:
            v = int((row.get("hard_fail_type_counts", {}) or {}).get(k, 0))
            fail_type_counter[k] += v
        for wr in row.get("top_windows", []):
            if isinstance(wr, dict):
                off_vals.append(_safe_float(wr.get("p95_off_raw_m"), 0.0))
                ff = wr.get("fail_flags", {})
                if isinstance(ff, dict):
                    if bool(ff.get("jump_spike", False)):
                        jump_spike_windows += 1
                    if bool(ff.get("mode_flip", False)):
                        mode_flip_windows += 1

    actor_rows.sort(key=lambda r: (int(r.get("hard_fail_windows", 0)), float(r.get("score", 0.0))), reverse=True)
    top_actors = actor_rows[: int(max(1, args.top_actors_per_scenario))]

    return {
        "scenario": str(dataset.get("scenario_name", "unknown")),
        "map_name": map_name,
        "is_intersection_map": bool(is_intersection_map),
        "actor_count": int(len(actor_rows)),
        "intersection_windows": int(total_windows),
        "hard_fail_windows": int(hard_fail_windows),
        "hard_fail_type_counts": {k: int(fail_type_counter.get(k, 0)) for k in FAIL_KEYS},
        "tracks_with_hard_fails": int(sum(1 for r in actor_rows if int(r.get("hard_fail_windows", 0)) > 0)),
        "scenario_score": float(scenario_score),
        "median_window_p95_off_raw_m": float(_median(off_vals)),
        "median_raw_to_final_dist_m": float(_median(global_raw_final_dists)),
        "p95_raw_to_final_dist_m": float(_p95(global_raw_final_dists)),
        "p95_final_step_m": float(_p95(global_final_steps)),
        "p95_step_ratio_final_vs_raw": float(_p95(global_step_ratios)),
        "nonmonotonic_time_pairs": int(nonmonotonic_time_pairs),
        "duplicate_time_pairs": int(duplicate_time_pairs),
        "aba_line_flickers": int(aba_line_flickers),
        "frame_samples": int(frame_samples),
        "jump_spike_windows_in_top_actor_windows": int(jump_spike_windows),
        "mode_flip_windows_in_top_actor_windows": int(mode_flip_windows),
        "top_actors": top_actors,
    }


def _load_maps(paths: Iterable[str]) -> List[object]:
    out = []
    for p in paths:
        out.append(load_vector_map(Path(p).expanduser().resolve()))
    return out


def _load_carla_runtime(args: argparse.Namespace) -> Dict[str, object]:
    cache_path = Path(args.carla_map_cache).expanduser().resolve()
    align_path = Path(args.carla_map_offset_json).expanduser().resolve()
    raw_lines, raw_bounds, carla_map_name = ytm._load_carla_map_cache_lines(cache_path)
    align_cfg = ytm._load_carla_alignment_cfg(align_path)
    transformed_lines, transformed_bbox = ytm._transform_carla_lines(raw_lines, align_cfg)
    return {
        "lines_xy": transformed_lines,
        "bbox": transformed_bbox,
        "source_path": str(cache_path),
        "map_name": str(carla_map_name or "carla_map_cache"),
        "lane_corr_top_k": int(args.lane_correspondence_top_k),
        "lane_corr_cache_dir": Path(args.lane_correspondence_cache_dir).expanduser().resolve(),
        "lane_corr_driving_types": _parse_lane_type_set(args.lane_correspondence_driving_types),
        "raw_bounds": raw_bounds,
        "align_cfg": dict(align_cfg),
    }


def _iter_scenarios(root: Path) -> List[Path]:
    if _is_scenario_directory(root):
        return [root]
    if not root.is_dir():
        return []
    return [p for p in sorted(root.iterdir()) if _is_scenario_directory(p)]


def _select_scenarios(all_scenarios: Sequence[Path], requested_names: Sequence[str], max_scenarios: int) -> List[Path]:
    if not requested_names:
        out = list(all_scenarios)
    else:
        requested = [str(x).strip() for x in requested_names if str(x).strip()]
        by_name = {p.name: p for p in all_scenarios}
        out: List[Path] = []
        for name in requested:
            p = Path(name).expanduser()
            if p.is_absolute() and _is_scenario_directory(p):
                out.append(p.resolve())
                continue
            if name in by_name:
                out.append(by_name[name])
                continue
        dedup: List[Path] = []
        seen = set()
        for p in out:
            sp = str(p)
            if sp in seen:
                continue
            seen.add(sp)
            dedup.append(p)
        out = dedup
    if int(max_scenarios) > 0:
        out = out[: int(max_scenarios)]
    return out


def _parse_env_overrides(rows: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    pat = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    for item in rows:
        text = str(item).strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(f"Invalid --env-override '{text}', expected NAME=VALUE")
        name, value = text.split("=", 1)
        name = str(name).strip()
        value = str(value).strip()
        if not pat.match(name):
            raise ValueError(f"Invalid env var name in override: '{name}'")
        out[name] = value
    return out


def _build_html(report: Dict[str, object]) -> str:
    cfg = report.get("config", {}) if isinstance(report.get("config", {}), dict) else {}
    summ = report.get("summary", {}) if isinstance(report.get("summary", {}), dict) else {}
    scenarios = report.get("scenarios", [])
    if not isinstance(scenarios, list):
        scenarios = []

    def _h(v: object) -> str:
        return html.escape(str(v))

    rows = []
    for sc in scenarios:
        if not isinstance(sc, dict):
            continue
        counts = sc.get("hard_fail_type_counts", {})
        if not isinstance(counts, dict):
            counts = {}
        row = (
            "<tr>"
            f"<td>{_h(sc.get('scenario', '-'))}</td>"
            f"<td>{_h(sc.get('map_name', '-'))}</td>"
            f"<td>{int(sc.get('intersection_windows', 0))}</td>"
            f"<td>{int(sc.get('hard_fail_windows', 0))}</td>"
            f"<td>{int(sc.get('tracks_with_hard_fails', 0))}</td>"
            f"<td>{_safe_float(sc.get('scenario_score', 0.0), 0.0):.2f}</td>"
            f"<td>{int(counts.get('goes_around_intersection', 0))}</td>"
            f"<td>{int(counts.get('mode_flip', 0))}</td>"
            f"<td>{int(counts.get('off_raw', 0))}</td>"
            f"<td>{int(counts.get('jump_spike', 0))}</td>"
            "</tr>"
        )
        rows.append(row)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Intersection Judge Report</title>
  <style>
    body {{ font-family: "Segoe UI", sans-serif; margin: 18px; background:#0f1720; color:#e4edf3; }}
    h1,h2 {{ margin: 0 0 10px 0; }}
    .card {{ background:#142434; border:1px solid #27445f; border-radius:8px; padding:12px; margin-bottom:12px; }}
    table {{ width:100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border:1px solid #2f4f69; padding:6px 8px; text-align:left; }}
    th {{ background:#1b3348; }}
    .mono {{ font-family: monospace; }}
  </style>
</head>
<body>
  <h1>Intersection Judge Report</h1>
  <div class="card">
    <div><b>Generated:</b> {_h(report.get('generated_at_utc', '-'))}</div>
    <div><b>Scenario root:</b> <span class="mono">{_h(report.get('scenario_root', '-'))}</span></div>
    <div><b>Profile:</b> {_h(cfg.get('processing_profile', '-'))}</div>
    <div><b>Env overrides:</b> <span class="mono">{_h(cfg.get('env_overrides', {}))}</span></div>
  </div>
  <div class="card">
    <h2>Summary</h2>
    <div>Scenarios evaluated: {int(summ.get('scenarios_evaluated', 0))}</div>
    <div>Intersection windows: {int(summ.get('intersection_windows', 0))}</div>
    <div>Hard-fail windows: {int(summ.get('hard_fail_windows', 0))}</div>
    <div>Scenarios with hard fails: {int(summ.get('scenarios_with_hard_fails', 0))}</div>
    <div>Hard-fail counts: <span class="mono">{_h(summ.get('hard_fail_type_counts', {}))}</span></div>
    <div>Median(raw→final): {_safe_float(summ.get('median_of_scenario_median_raw_to_final_dist_m'), float('nan')):.3f} m</div>
    <div>Median(p95 raw→final): {_safe_float(summ.get('median_of_scenario_p95_raw_to_final_dist_m'), float('nan')):.3f} m</div>
    <div>Median(p95 final/raw step ratio): {_safe_float(summ.get('median_of_scenario_p95_step_ratio_final_vs_raw'), float('nan')):.3f}</div>
    <div>Time continuity violations: nonmono={int(summ.get('nonmonotonic_time_pairs_total', 0))}, duplicate={int(summ.get('duplicate_time_pairs_total', 0))}</div>
    <div>ABA line flickers total: {int(summ.get('aba_line_flickers_total', 0))}</div>
  </div>
  <div class="card">
    <h2>Scenario Ranking</h2>
    <table>
      <thead>
        <tr>
          <th>Scenario</th><th>Map</th><th>Windows</th><th>Hard Windows</th>
          <th>Tracks with Fails</th><th>Score</th><th>around</th><th>mode_flip</th>
          <th>off_raw</th><th>jump_spike</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </div>
</body>
</html>
"""


def _run(args: argparse.Namespace) -> int:
    scenario_root = Path(args.scenario_root).expanduser().resolve()
    all_scenarios = _iter_scenarios(scenario_root)
    selected = _select_scenarios(
        all_scenarios=all_scenarios,
        requested_names=args.scenario_name,
        max_scenarios=int(args.max_scenarios),
    )
    if not selected:
        print(f"[ERROR] No scenarios selected under: {scenario_root}", file=sys.stderr)
        return 2

    env_snapshot = dict(os.environ)
    try:
        profile = _normalize_processing_profile_name(args.processing_profile)
        if profile not in PROCESSING_PROFILE_CHOICES:
            allowed = ", ".join(PROCESSING_PROFILE_CHOICES)
            raise ValueError(f"Invalid --processing-profile '{args.processing_profile}'. Allowed: {allowed}")
        profile_overrides = _apply_processing_profile(profile=profile, verbose=bool(args.show_pipeline_logs))
        env_overrides = _parse_env_overrides(args.env_override)
        for k, v in env_overrides.items():
            os.environ[str(k)] = str(v)

        map_data_list = _load_maps(args.map_pkl)
        carla_runtime = _load_carla_runtime(args)
        carla_context_cache: Dict[str, Dict[str, object]] = {}

        scenario_rows: List[Dict[str, object]] = []
        fail_type_counter: Counter[str] = Counter()
        total_windows = 0
        total_hard_fail_windows = 0
        scenarios_with_hard_fails = 0
        scenario_errors = 0
        scenario_median_raw_final: List[float] = []
        scenario_p95_raw_final: List[float] = []
        scenario_p95_final_step: List[float] = []
        scenario_p95_step_ratio: List[float] = []
        total_nonmono_pairs = 0
        total_dup_pairs = 0
        total_aba_flickers = 0
        total_frame_samples = 0

        for idx, scenario in enumerate(selected, start=1):
            print(f"[{idx}/{len(selected)}] {scenario.name}")
            try:
                if args.show_pipeline_logs:
                    dataset = process_single_scenario(
                        scenario_dir=scenario,
                        map_data_list=map_data_list,
                        dt=float(args.dt),
                        tx=0.0,
                        ty=0.0,
                        tz=0.0,
                        yaw_deg=0.0,
                        flip_y=False,
                        subdir=None,
                        carla_runtime=carla_runtime,
                        enable_carla_projection=True,
                        timing_cfg={
                            "maximize_safe_early_spawn": True,
                            "maximize_safe_late_despawn": True,
                            "early_spawn_safety_margin": 0.25,
                            "late_despawn_safety_margin": 0.25,
                        },
                        walker_cfg={"enabled": False},
                        carla_context_cache=carla_context_cache,
                        verbose=False,
                    )
                else:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        dataset = process_single_scenario(
                            scenario_dir=scenario,
                            map_data_list=map_data_list,
                            dt=float(args.dt),
                            tx=0.0,
                            ty=0.0,
                            tz=0.0,
                            yaw_deg=0.0,
                            flip_y=False,
                            subdir=None,
                            carla_runtime=carla_runtime,
                            enable_carla_projection=True,
                            timing_cfg={
                                "maximize_safe_early_spawn": True,
                                "maximize_safe_late_despawn": True,
                                "early_spawn_safety_margin": 0.25,
                                "late_despawn_safety_margin": 0.25,
                            },
                            walker_cfg={"enabled": False},
                            carla_context_cache=carla_context_cache,
                            verbose=False,
                        )
                if dataset is None:
                    scenario_errors += 1
                    print(f"  [WARN] scenario processing returned None: {scenario.name}")
                    continue
                row = _scenario_assessment(dataset=dataset, args=args)
            except Exception as exc:
                scenario_errors += 1
                print(f"  [WARN] failed to evaluate scenario {scenario.name}: {exc}")
                continue

            if bool(args.only_intersection_map) and not bool(row.get("is_intersection_map", False)):
                print("  [SKIP] non-intersection map")
                continue

            scenario_rows.append(row)
            total_windows += int(row.get("intersection_windows", 0))
            total_hard_fail_windows += int(row.get("hard_fail_windows", 0))
            if int(row.get("hard_fail_windows", 0)) > 0:
                scenarios_with_hard_fails += 1
            m1 = _safe_float(row.get("median_raw_to_final_dist_m"), float("nan"))
            m2 = _safe_float(row.get("p95_raw_to_final_dist_m"), float("nan"))
            m3 = _safe_float(row.get("p95_final_step_m"), float("nan"))
            m4 = _safe_float(row.get("p95_step_ratio_final_vs_raw"), float("nan"))
            if math.isfinite(float(m1)):
                scenario_median_raw_final.append(float(m1))
            if math.isfinite(float(m2)):
                scenario_p95_raw_final.append(float(m2))
            if math.isfinite(float(m3)):
                scenario_p95_final_step.append(float(m3))
            if math.isfinite(float(m4)):
                scenario_p95_step_ratio.append(float(m4))
            total_nonmono_pairs += int(row.get("nonmonotonic_time_pairs", 0))
            total_dup_pairs += int(row.get("duplicate_time_pairs", 0))
            total_aba_flickers += int(row.get("aba_line_flickers", 0))
            total_frame_samples += int(row.get("frame_samples", 0))
            counts = row.get("hard_fail_type_counts", {})
            if isinstance(counts, dict):
                for k in FAIL_KEYS:
                    fail_type_counter[k] += int(counts.get(k, 0))

        scenario_rows.sort(
            key=lambda r: (int(r.get("hard_fail_windows", 0)), float(r.get("scenario_score", 0.0))),
            reverse=True,
        )
        summary = {
            "scenarios_selected": int(len(selected)),
            "scenarios_evaluated": int(len(scenario_rows)),
            "scenario_errors": int(scenario_errors),
            "scenarios_with_hard_fails": int(scenarios_with_hard_fails),
            "intersection_windows": int(total_windows),
            "hard_fail_windows": int(total_hard_fail_windows),
            "hard_fail_type_counts": {k: int(fail_type_counter.get(k, 0)) for k in FAIL_KEYS},
            "median_of_scenario_median_raw_to_final_dist_m": float(_median(scenario_median_raw_final)),
            "median_of_scenario_p95_raw_to_final_dist_m": float(_median(scenario_p95_raw_final)),
            "median_of_scenario_p95_final_step_m": float(_median(scenario_p95_final_step)),
            "median_of_scenario_p95_step_ratio_final_vs_raw": float(_median(scenario_p95_step_ratio)),
            "nonmonotonic_time_pairs_total": int(total_nonmono_pairs),
            "duplicate_time_pairs_total": int(total_dup_pairs),
            "aba_line_flickers_total": int(total_aba_flickers),
            "frame_samples_total": int(total_frame_samples),
        }
        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scenario_root": str(scenario_root),
            "config": {
                "processing_profile": str(profile),
                "profile_env_overrides": dict(profile_overrides),
                "env_overrides": dict(env_overrides),
                "only_intersection_map": bool(args.only_intersection_map),
                "thresholds": {
                    "raw_step_threshold_m": float(args.raw_step_threshold),
                    "jump_threshold_m": float(args.jump_threshold),
                    "off_raw_p95_threshold_m": float(args.off_raw_p95_threshold),
                    "off_raw_max_threshold_m": float(args.off_raw_max_threshold),
                    "detour_ratio_threshold": float(args.detour_ratio_threshold),
                    "anchor_gap_threshold_m": float(args.anchor_gap_threshold),
                },
            },
            "summary": summary,
            "scenarios": scenario_rows,
        }

        if args.output:
            out_path = Path(args.output).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"[INFO] Wrote JSON: {out_path}")
        if args.html_output:
            html_path = Path(args.html_output).expanduser().resolve()
            html_path.parent.mkdir(parents=True, exist_ok=True)
            html_path.write_text(_build_html(report), encoding="utf-8")
            print(f"[INFO] Wrote HTML: {html_path}")

        print(
            "[SUMMARY] scenarios={} hard_windows={} windows={} hard_types={} med_raw_final={:.3f} p95_raw_final={:.3f} p95_step_ratio={:.3f} nonmono={} dup={} aba={}".format(
                int(summary["scenarios_evaluated"]),
                int(summary["hard_fail_windows"]),
                int(summary["intersection_windows"]),
                summary["hard_fail_type_counts"],
                _safe_float(summary.get("median_of_scenario_median_raw_to_final_dist_m"), float("nan")),
                _safe_float(summary.get("median_of_scenario_p95_raw_to_final_dist_m"), float("nan")),
                _safe_float(summary.get("median_of_scenario_p95_step_ratio_final_vs_raw"), float("nan")),
                int(summary.get("nonmonotonic_time_pairs_total", 0)),
                int(summary.get("duplicate_time_pairs_total", 0)),
                int(summary.get("aba_line_flickers_total", 0)),
            )
        )
        if bool(args.fail_on_hard_fail) and int(summary["hard_fail_windows"]) > 0:
            return 1
        return 0
    finally:
        os.environ.clear()
        os.environ.update(env_snapshot)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Judge intersection trajectory quality for real->sim outputs.")
    parser.add_argument(
        "--scenario-root",
        type=str,
        default="/data2/marco/CoLMDriver/v2xpnp/dataset/train4",
        help="Scenario root directory (single scenario directory also supported).",
    )
    parser.add_argument(
        "--scenario-name",
        action="append",
        default=[],
        help="Scenario name (or absolute path) to evaluate. Repeatable.",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=0,
        help="Maximum number of scenarios to evaluate (0 = no limit).",
    )
    parser.add_argument(
        "--processing-profile",
        choices=PROCESSING_PROFILE_CHOICES,
        default="current",
        help="Processing profile applied before evaluation.",
    )
    parser.add_argument(
        "--env-override",
        action="append",
        default=[],
        help="Extra environment override NAME=VALUE (repeatable).",
    )
    parser.add_argument(
        "--map-pkl",
        nargs="+",
        default=[
            "/data2/marco/CoLMDriver/v2xpnp/map/v2v_corridors_vector_map.pkl",
            "/data2/marco/CoLMDriver/v2xpnp/map/v2x_intersection_vector_map.pkl",
        ],
        help="Vector map pickle paths.",
    )
    parser.add_argument(
        "--carla-map-cache",
        type=str,
        default="/data2/marco/CoLMDriver/v2xpnp/map/carla_map_cache.pkl",
        help="Path to CARLA map cache pickle.",
    )
    parser.add_argument(
        "--carla-map-offset-json",
        type=str,
        default="/data2/marco/CoLMDriver/v2xpnp/map/ucla_map_offset_carla.json",
        help="Path to CARLA alignment JSON.",
    )
    parser.add_argument(
        "--lane-correspondence-top-k",
        type=int,
        default=56,
        help="Candidate CARLA lines per V2 lane during correspondence.",
    )
    parser.add_argument(
        "--lane-correspondence-cache-dir",
        type=str,
        default="/tmp/v2x_stage_corr_cache",
        help="Cache directory for lane correspondence.",
    )
    parser.add_argument(
        "--lane-correspondence-driving-types",
        type=str,
        default="1",
        help="Comma/space-separated V2 lane types treated as driving.",
    )
    parser.add_argument("--dt", type=float, default=0.1, help="Frame dt.")

    parser.add_argument("--min-window-frames", type=int, default=4, help="Minimum intersection window frames.")
    parser.add_argument("--event-pad-frames", type=int, default=2, help="Pad around detected event windows.")
    parser.add_argument("--merge-gap-frames", type=int, default=1, help="Merge neighboring windows up to this gap.")
    parser.add_argument("--raw-step-threshold", type=float, default=0.6, help="Raw low-motion threshold for jump_spike.")
    parser.add_argument("--jump-threshold", type=float, default=1.8, help="Projected step threshold for jump_spike.")
    parser.add_argument("--off-raw-p95-threshold", type=float, default=2.5, help="P95 raw->projected offset threshold.")
    parser.add_argument("--off-raw-max-threshold", type=float, default=4.2, help="Max raw->projected offset threshold.")
    parser.add_argument("--detour-ratio-threshold", type=float, default=1.55, help="Final/raw path length ratio threshold.")
    parser.add_argument("--anchor-gap-threshold", type=float, default=1.8, help="Final miss distance from raw window centroid.")
    parser.add_argument("--goes-around-min-p95-off", type=float, default=1.8, help="Minimum p95 off-raw for goes_around.")
    parser.add_argument("--goes-around-min-raw-path", type=float, default=6.0, help="Minimum raw path length for goes_around.")

    parser.add_argument("--top-actors-per-scenario", type=int, default=12, help="Top actors to keep in scenario output.")
    parser.add_argument("--top-windows-per-track", type=int, default=6, help="Top windows to keep in track output.")
    parser.add_argument("--emit-all-windows", action="store_true", help="Emit all window rows per track.")
    parser.add_argument("--only-intersection-map", dest="only_intersection_map", action="store_true", default=True, help="Keep only intersection-map scenarios.")
    parser.add_argument("--include-all-maps", dest="only_intersection_map", action="store_false", help="Include scenarios mapped to non-intersection maps.")
    parser.add_argument("--show-pipeline-logs", action="store_true", help="Show pipeline logs for scenario processing.")
    parser.add_argument("--fail-on-hard-fail", action="store_true", help="Exit with code 1 when hard_fail_windows > 0.")
    parser.add_argument("--output", type=str, default="", help="JSON output path.")
    parser.add_argument("--html-output", type=str, default="", help="Optional HTML report output path.")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        return _run(args)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
