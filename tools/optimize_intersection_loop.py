#!/usr/bin/env python3
"""
Crash-resilient automated optimizer for intersection trajectory quality.

Loop behavior:
1) Evaluate current best configuration with tools/judge_intersections.py
2) Propose informed tweaks from current failure profile
3) Evaluate candidates, rank by strict multi-objective score
4) Persist best-so-far config + full state every iteration
5) Resume from state after interruption/crash

Design goals:
- Prioritize hard failures first (goes_around, mode_flip, jump_spike, off_raw)
- Then optimize fidelity/smoothness continuity metrics
- Keep profile fixed to current by default; tune env knobs directly
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


FAIL_KEYS: Tuple[str, ...] = (
    "goes_around_intersection",
    "mode_flip",
    "jump_spike",
    "off_raw",
)


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _default_knob_space() -> Dict[str, List[float]]:
    # Ordered values are low->high. "Strictness" direction depends on knob.
    return {
        # V2 alignment intersection/transition controls
        "V2X_ALIGN_INTERSECTION_BLEND_MAX_LOW_MOTION_MEAN_STEP_M": [0.25, 0.30, 0.35, 0.40, 0.45],
        "V2X_ALIGN_INTERSECTION_BLEND_MIN_YAW_EVIDENCE_DEG": [8.0, 10.0, 12.0, 14.0, 16.0],
        "V2X_ALIGN_INTERSECTION_BLEND_MIN_LATERAL_EVIDENCE_M": [0.50, 0.70, 0.80, 1.00, 1.20],
        "V2X_ALIGN_APPROACH_LOCK_STEP_GUARD_RATIO": [1.8, 2.0, 2.2, 2.4, 2.6],
        "V2X_ALIGN_APPROACH_LOCK_STEP_GUARD_FLOOR_M": [1.0, 1.2, 1.4, 1.6, 1.8],
        "V2X_ALIGN_RETIME_TRANSITION_MIN_IMPROVEMENT_M": [0.30, 0.45, 0.55, 0.70, 0.90],
        "V2X_ALIGN_SOFTEN_TRANSITION_JUMP_M": [1.2, 1.4, 1.6, 1.8, 2.0],
        "V2X_ALIGN_SOFTEN_TRANSITION_MAX_SHIFT_M": [0.8, 1.0, 1.2, 1.4, 1.6],
        # CARLA projection/override continuity controls
        "V2X_CARLA_LOW_MOTION_OVERRIDE_MIN_DIST_GAIN_M": [0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
        "V2X_CARLA_LOW_MOTION_OVERRIDE_MIN_SCORE_GAIN": [0.6, 0.8, 0.85, 1.0, 1.2],
        "V2X_CARLA_LOW_MOTION_OVERRIDE_MAX_JUMP_SCALE": [1.4, 1.6, 1.8, 2.0, 2.2],
        "V2X_CARLA_LOW_MOTION_OVERRIDE_MAX_JUMP_BIAS_M": [0.2, 0.3, 0.4, 0.5, 0.6],
        "V2X_CARLA_NEAREST_CONT_SCORE_SLACK": [0.4, 0.6, 0.8, 1.0, 1.2],
        "V2X_CARLA_NEAREST_CONT_DIST_SLACK": [0.5, 0.8, 1.0, 1.2, 1.5],
        "V2X_CARLA_DISCONNECTED_OVERRIDE_DIST_GAIN": [0.8, 1.0, 1.2, 1.4, 1.6],
        "V2X_CARLA_DISCONNECTED_OVERRIDE_SCORE_GAIN": [1.0, 1.4, 1.8, 2.2, 2.6],
        "V2X_CARLA_SEMANTIC_TRANSITION_SUSTAIN_FRAMES": [1.0, 2.0, 3.0, 4.0],
        # CARLA transition/postprocess smoothing
        "V2X_CARLA_TRANSITION_WINDOW_MAX_SHIFT_M": [0.6, 0.8, 1.0, 1.2, 1.4],
        "V2X_CARLA_TRANSITION_WINDOW_PASSES": [1.0, 2.0, 3.0, 4.0],
        "V2X_CARLA_SMALL_TRANSITION_JUMP_M": [1.1, 1.3, 1.5, 1.7, 1.9],
        "V2X_CARLA_SMALL_TRANSITION_MAX_SHIFT_M": [0.8, 1.0, 1.2, 1.4, 1.6],
        "V2X_CARLA_SEMANTIC_BOUNDARY_JUMP_M": [2.0, 2.3, 2.6, 2.9, 3.2],
        "V2X_CARLA_SEMANTIC_BOUNDARY_MAX_SHIFT_M": [0.9, 1.1, 1.35, 1.6, 1.9],
        "V2X_CARLA_HOLD_SEMANTIC_LINE_IDS": [0.0, 1.0],
        "V2X_CARLA_HOLD_TURN_LINE_IDS": [0.0, 1.0],
        # Intersection shape controls
        "V2X_CARLA_INTERSECTION_SHAPE_MAX_QUERY_OFFSET_M": [2.0, 2.2, 2.4, 2.6, 2.8, 3.0],
        "V2X_CARLA_INTERSECTION_SHAPE_MAX_RAW_OFFSET_M": [2.0, 2.2, 2.4, 2.6, 2.8, 3.0],
        "V2X_CARLA_INTERSECTION_SHAPE_MAX_RAW_WORSEN_PEAK_M": [0.4, 0.5, 0.6, 0.8, 1.0],
        "V2X_CARLA_INTERSECTION_SHAPE_MAX_RAW_WORSEN_PER_FRAME_M": [0.2, 0.25, 0.30, 0.35, 0.45],
        "V2X_CARLA_INTERSECTION_SHAPE_MAX_WORSEN_PER_FRAME_M": [0.35, 0.45, 0.55, 0.65, 0.80],
        "V2X_CARLA_INTERSECTION_SHAPE_CURVE_YAW_DEG": [8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
        "V2X_CARLA_INTERSECTION_SHAPE_CURVE_LATERAL_M": [0.5, 0.7, 0.8, 1.0, 1.2],
        "V2X_CARLA_INTERSECTION_SHAPE_CURVE_SLACK": [0.6, 0.9, 1.2, 1.5, 1.8],
        "V2X_CARLA_INTERSECTION_SHAPE_STRAIGHT_SLACK": [0.2, 0.3, 0.4, 0.6, 0.8],
        "V2X_CARLA_INTERSECTION_SHAPE_CURVE_TAN_SCALE": [0.25, 0.32, 0.40, 0.48, 0.56],
        "V2X_CARLA_INTERSECTION_CLUSTER_MAX_WORSEN_PER_FRAME_M": [0.45, 0.55, 0.65, 0.75, 0.90],
        "V2X_CARLA_INTERSECTION_CLUSTER_CURVE_YAW_DEG": [8.0, 10.0, 11.0, 13.0, 15.0],
        "V2X_CARLA_INTERSECTION_CLUSTER_CURVE_LATERAL_M": [0.5, 0.6, 0.7, 0.9, 1.1],
        "V2X_CARLA_INTERSECTION_GLOBAL_MODE_PER_TRACK": [0.0, 1.0],
        "V2X_CARLA_FORCE_SINGLE_INTERSECTION_MODE": [0.0, 1.0],
    }


def _default_values() -> Dict[str, float]:
    # Baselines from current script defaults where available.
    return {
        "V2X_ALIGN_INTERSECTION_BLEND_MAX_LOW_MOTION_MEAN_STEP_M": 0.40,
        "V2X_ALIGN_INTERSECTION_BLEND_MIN_YAW_EVIDENCE_DEG": 12.0,
        "V2X_ALIGN_INTERSECTION_BLEND_MIN_LATERAL_EVIDENCE_M": 0.8,
        "V2X_ALIGN_APPROACH_LOCK_STEP_GUARD_RATIO": 2.2,
        "V2X_ALIGN_APPROACH_LOCK_STEP_GUARD_FLOOR_M": 1.2,
        "V2X_ALIGN_RETIME_TRANSITION_MIN_IMPROVEMENT_M": 0.45,
        "V2X_ALIGN_SOFTEN_TRANSITION_JUMP_M": 1.6,
        "V2X_ALIGN_SOFTEN_TRANSITION_MAX_SHIFT_M": 1.2,
        "V2X_CARLA_LOW_MOTION_OVERRIDE_MIN_DIST_GAIN_M": 1.2,
        "V2X_CARLA_LOW_MOTION_OVERRIDE_MIN_SCORE_GAIN": 0.85,
        "V2X_CARLA_LOW_MOTION_OVERRIDE_MAX_JUMP_SCALE": 2.0,
        "V2X_CARLA_LOW_MOTION_OVERRIDE_MAX_JUMP_BIAS_M": 0.4,
        "V2X_CARLA_NEAREST_CONT_SCORE_SLACK": 0.6,
        "V2X_CARLA_NEAREST_CONT_DIST_SLACK": 0.8,
        "V2X_CARLA_DISCONNECTED_OVERRIDE_DIST_GAIN": 1.4,
        "V2X_CARLA_DISCONNECTED_OVERRIDE_SCORE_GAIN": 1.8,
        "V2X_CARLA_SEMANTIC_TRANSITION_SUSTAIN_FRAMES": 2.0,
        "V2X_CARLA_TRANSITION_WINDOW_MAX_SHIFT_M": 1.2,
        "V2X_CARLA_TRANSITION_WINDOW_PASSES": 3.0,
        "V2X_CARLA_SMALL_TRANSITION_JUMP_M": 1.5,
        "V2X_CARLA_SMALL_TRANSITION_MAX_SHIFT_M": 1.2,
        "V2X_CARLA_SEMANTIC_BOUNDARY_JUMP_M": 2.6,
        "V2X_CARLA_SEMANTIC_BOUNDARY_MAX_SHIFT_M": 1.35,
        "V2X_CARLA_HOLD_SEMANTIC_LINE_IDS": 1.0,
        "V2X_CARLA_HOLD_TURN_LINE_IDS": 1.0,
        "V2X_CARLA_INTERSECTION_SHAPE_MAX_QUERY_OFFSET_M": 3.0,
        "V2X_CARLA_INTERSECTION_SHAPE_MAX_RAW_OFFSET_M": 2.8,
        "V2X_CARLA_INTERSECTION_SHAPE_MAX_RAW_WORSEN_PEAK_M": 0.8,
        "V2X_CARLA_INTERSECTION_SHAPE_MAX_RAW_WORSEN_PER_FRAME_M": 0.35,
        "V2X_CARLA_INTERSECTION_SHAPE_MAX_WORSEN_PER_FRAME_M": 0.55,
        "V2X_CARLA_INTERSECTION_SHAPE_CURVE_YAW_DEG": 12.0,
        "V2X_CARLA_INTERSECTION_SHAPE_CURVE_LATERAL_M": 0.8,
        "V2X_CARLA_INTERSECTION_SHAPE_CURVE_SLACK": 1.2,
        "V2X_CARLA_INTERSECTION_SHAPE_STRAIGHT_SLACK": 0.4,
        "V2X_CARLA_INTERSECTION_SHAPE_CURVE_TAN_SCALE": 0.42,
        "V2X_CARLA_INTERSECTION_CLUSTER_MAX_WORSEN_PER_FRAME_M": 0.75,
        "V2X_CARLA_INTERSECTION_CLUSTER_CURVE_YAW_DEG": 11.0,
        "V2X_CARLA_INTERSECTION_CLUSTER_CURVE_LATERAL_M": 0.7,
        "V2X_CARLA_INTERSECTION_GLOBAL_MODE_PER_TRACK": 1.0,
        "V2X_CARLA_FORCE_SINGLE_INTERSECTION_MODE": 1.0,
    }


def _closest_idx(grid: Sequence[float], value: float) -> int:
    best_i = 0
    best_d = float("inf")
    for i, g in enumerate(grid):
        d = abs(float(g) - float(value))
        if d < best_d:
            best_d = d
            best_i = int(i)
    return int(best_i)


def _indices_from_values(space: Dict[str, List[float]], values: Dict[str, float]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for k, grid in space.items():
        out[k] = _closest_idx(grid, float(values.get(k, grid[len(grid) // 2])))
    return out


def _values_from_indices(space: Dict[str, List[float]], indices: Dict[str, int]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, grid in space.items():
        idx = int(indices.get(k, len(grid) // 2))
        idx = max(0, min(len(grid) - 1, idx))
        out[k] = float(grid[idx])
    return out


def _config_hash(config: Dict[str, float]) -> str:
    blob = json.dumps({k: config[k] for k in sorted(config.keys())}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _score_from_report(report: Dict[str, object]) -> Tuple[int, ...]:
    summary = report.get("summary", {})
    if not isinstance(summary, dict):
        # catastrophic parse failure
        return (10**9,) * 12
    counts = summary.get("hard_fail_type_counts", {})
    if not isinstance(counts, dict):
        counts = {}

    def _q(value: object, scale: float = 1000.0) -> int:
        v = _safe_float(value, float("inf"))
        if not math.isfinite(v):
            return 10**9
        return int(round(float(scale) * float(v)))

    return (
        int(summary.get("hard_fail_windows", 10**9)),
        int(summary.get("scenarios_with_hard_fails", 10**9)),
        int(counts.get("goes_around_intersection", 10**9)),
        int(counts.get("mode_flip", 10**9)),
        int(counts.get("jump_spike", 10**9)),
        int(counts.get("off_raw", 10**9)),
        _q(summary.get("median_of_scenario_p95_raw_to_final_dist_m", float("inf"))),
        _q(summary.get("median_of_scenario_p95_step_ratio_final_vs_raw", float("inf"))),
        _q(summary.get("median_of_scenario_p95_final_step_m", float("inf"))),
        int(summary.get("nonmonotonic_time_pairs_total", 10**9)),
        int(summary.get("duplicate_time_pairs_total", 10**9)),
        int(summary.get("aba_line_flickers_total", 10**9)),
    )


def _extract_source_hints(report: Dict[str, object], top_n_scenarios: int = 6) -> Dict[str, int]:
    out: Dict[str, int] = {}
    scenarios = report.get("scenarios", [])
    if not isinstance(scenarios, list):
        return out
    for sc in scenarios[: int(max(1, top_n_scenarios))]:
        if not isinstance(sc, dict):
            continue
        actors = sc.get("top_actors", [])
        if not isinstance(actors, list):
            continue
        for ar in actors[:8]:
            if not isinstance(ar, dict):
                continue
            wins = ar.get("top_windows", [])
            if not isinstance(wins, list):
                continue
            for wr in wins[:6]:
                if not isinstance(wr, dict):
                    continue
                ts = wr.get("top_sources", [])
                if not isinstance(ts, list):
                    continue
                for row in ts[:6]:
                    if isinstance(row, (list, tuple)) and len(row) >= 2:
                        src = str(row[0])
                        cnt = _safe_int(row[1], 0)
                        out[src] = out.get(src, 0) + int(max(0, cnt))
    return out


@dataclass
class Tweak:
    knob: str
    direction: int
    reason: str
    weight: float


def _dedupe_tweaks(tweaks: Sequence[Tweak]) -> List[Tweak]:
    out: List[Tweak] = []
    seen = set()
    for tw in tweaks:
        key = (tw.knob, int(tw.direction))
        if key in seen:
            continue
        seen.add(key)
        out.append(tw)
    out.sort(key=lambda t: float(t.weight), reverse=True)
    return out


def _propose_tweaks(report: Dict[str, object]) -> List[Tweak]:
    summary = report.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    counts = summary.get("hard_fail_type_counts", {})
    if not isinstance(counts, dict):
        counts = {}

    goes_around = _safe_int(counts.get("goes_around_intersection"), 0)
    mode_flip = _safe_int(counts.get("mode_flip"), 0)
    jump_spike = _safe_int(counts.get("jump_spike"), 0)
    off_raw = _safe_int(counts.get("off_raw"), 0)

    tweaks: List[Tweak] = []
    source_hints = _extract_source_hints(report)
    hint_top = sorted(source_hints.items(), key=lambda kv: kv[1], reverse=True)[:6]
    hint_keys = {k for k, _ in hint_top}

    # Hard-fail targeted tweaks
    if goes_around > 0 or off_raw > 0:
        tweaks.extend(
            [
                Tweak("V2X_CARLA_INTERSECTION_SHAPE_MAX_QUERY_OFFSET_M", -1, "Reduce long detours around intersection", 3.0),
                Tweak("V2X_CARLA_INTERSECTION_SHAPE_MAX_RAW_OFFSET_M", -1, "Keep projection closer to raw", 3.0),
                Tweak("V2X_CARLA_INTERSECTION_SHAPE_MAX_RAW_WORSEN_PEAK_M", -1, "Limit peak shape deviation from raw", 2.8),
                Tweak("V2X_CARLA_INTERSECTION_SHAPE_MAX_WORSEN_PER_FRAME_M", -1, "Limit per-frame shape correction drift", 2.6),
                Tweak("V2X_CARLA_INTERSECTION_CLUSTER_MAX_WORSEN_PER_FRAME_M", -1, "Tighten cluster-level curve drift", 2.4),
                Tweak("V2X_CARLA_NEAREST_CONT_SCORE_SLACK", -1, "Reduce permissive continuity switching", 2.1),
                Tweak("V2X_CARLA_NEAREST_CONT_DIST_SLACK", -1, "Reduce permissive continuity switching", 2.1),
            ]
        )
    if jump_spike > 0:
        tweaks.extend(
            [
                Tweak("V2X_CARLA_LOW_MOTION_OVERRIDE_MIN_DIST_GAIN_M", +1, "Require stronger gain before low-motion line switch", 3.2),
                Tweak("V2X_CARLA_LOW_MOTION_OVERRIDE_MIN_SCORE_GAIN", +1, "Require stronger score gain before low-motion switch", 3.2),
                Tweak("V2X_CARLA_LOW_MOTION_OVERRIDE_MAX_JUMP_SCALE", -1, "Clamp low-motion projected jumps", 3.1),
                Tweak("V2X_CARLA_LOW_MOTION_OVERRIDE_MAX_JUMP_BIAS_M", -1, "Clamp low-motion projected jumps", 3.1),
                Tweak("V2X_CARLA_SMALL_TRANSITION_JUMP_M", -1, "Repair small transition spikes more aggressively", 2.4),
                Tweak("V2X_CARLA_TRANSITION_WINDOW_MAX_SHIFT_M", -1, "Bound transition window over-corrections", 2.2),
            ]
        )
    if mode_flip > 0:
        tweaks.extend(
            [
                Tweak("V2X_CARLA_FORCE_SINGLE_INTERSECTION_MODE", +1, "Force one mode per intersection cluster", 3.0),
                Tweak("V2X_CARLA_INTERSECTION_GLOBAL_MODE_PER_TRACK", +1, "Harmonize mode across track-level intersection passes", 2.9),
                Tweak("V2X_CARLA_INTERSECTION_SHAPE_CURVE_YAW_DEG", +1, "Increase curve evidence threshold to avoid ambiguity", 2.1),
                Tweak("V2X_CARLA_INTERSECTION_SHAPE_CURVE_LATERAL_M", +1, "Increase curve evidence threshold to avoid ambiguity", 2.1),
                Tweak("V2X_CARLA_INTERSECTION_SHAPE_STRAIGHT_SLACK", -1, "Reduce mixed-mode ambiguity", 1.8),
            ]
        )

    if _safe_int(summary.get("aba_line_flickers_total"), 0) > 0:
        tweaks.extend(
            [
                Tweak("V2X_CARLA_HOLD_SEMANTIC_LINE_IDS", +1, "Reduce A-B-A line-id flicker", 2.2),
                Tweak("V2X_CARLA_HOLD_TURN_LINE_IDS", +1, "Reduce turn-line flicker", 2.1),
                Tweak("V2X_CARLA_SEMANTIC_TRANSITION_SUSTAIN_FRAMES", +1, "Require sustained semantic transition", 2.0),
            ]
        )

    # Source-informed augmentations
    if any("nearest_override" in k for k in hint_keys):
        tweaks.extend(
            [
                Tweak("V2X_CARLA_LOW_MOTION_OVERRIDE_MIN_DIST_GAIN_M", +1, "nearest_override dominant", 2.5),
                Tweak("V2X_CARLA_LOW_MOTION_OVERRIDE_MIN_SCORE_GAIN", +1, "nearest_override dominant", 2.5),
                Tweak("V2X_CARLA_NEAREST_CONT_SCORE_SLACK", -1, "nearest_override dominant", 2.0),
            ]
        )
    if any(("intersection_shape_" in k) or ("turn_regen" in k) for k in hint_keys):
        tweaks.extend(
            [
                Tweak("V2X_CARLA_INTERSECTION_SHAPE_MAX_RAW_OFFSET_M", -1, "shape/turn source dominant", 2.3),
                Tweak("V2X_CARLA_INTERSECTION_SHAPE_MAX_QUERY_OFFSET_M", -1, "shape/turn source dominant", 2.3),
                Tweak("V2X_CARLA_INTERSECTION_SHAPE_CURVE_TAN_SCALE", -1, "shape/turn source dominant", 1.9),
            ]
        )
    if any(("transition_window" in k) or ("semantic_boundary" in k) for k in hint_keys):
        tweaks.extend(
            [
                Tweak("V2X_CARLA_TRANSITION_WINDOW_MAX_SHIFT_M", -1, "transition source dominant", 2.3),
                Tweak("V2X_CARLA_TRANSITION_WINDOW_PASSES", +1, "transition source dominant", 1.8),
                Tweak("V2X_CARLA_SEMANTIC_BOUNDARY_MAX_SHIFT_M", -1, "transition source dominant", 2.1),
            ]
        )

    # Fallback if no clear signal
    if not tweaks:
        tweaks.extend(
            [
                Tweak("V2X_CARLA_INTERSECTION_SHAPE_MAX_RAW_OFFSET_M", -1, "fallback tighten raw fidelity", 1.0),
                Tweak("V2X_CARLA_LOW_MOTION_OVERRIDE_MIN_DIST_GAIN_M", +1, "fallback lower jump risk", 1.0),
                Tweak("V2X_CARLA_NEAREST_CONT_SCORE_SLACK", -1, "fallback reduce hopping", 1.0),
            ]
        )

    return _dedupe_tweaks(tweaks)


def _apply_tweaks(
    indices: Dict[str, int],
    tweaks: Sequence[Tweak],
    space: Dict[str, List[float]],
) -> Optional[Dict[str, int]]:
    out = dict(indices)
    for tw in tweaks:
        if tw.knob not in space:
            return None
        grid = space[tw.knob]
        cur = int(out.get(tw.knob, len(grid) // 2))
        nxt = int(cur + int(tw.direction))
        if nxt < 0 or nxt >= len(grid):
            return None
        out[tw.knob] = int(nxt)
    return out


def _env_from_config(config: Dict[str, float]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in config.items():
        fv = float(v)
        if abs(fv - round(fv)) <= 1e-9:
            out[k] = str(int(round(fv)))
        else:
            out[k] = f"{fv:.6f}".rstrip("0").rstrip(".")
    return out


def _run_judge(
    repo_root: Path,
    judge_args: Sequence[str],
    env_overrides: Dict[str, str],
    report_json: Path,
    report_html: Path,
    log_path: Path,
) -> Tuple[int, str]:
    cmd = [
        sys.executable,
        "tools/judge_intersections.py",
        *list(judge_args),
        "--output",
        str(report_json),
        "--html-output",
        str(report_html),
    ]
    run_env = dict(os.environ)
    run_env.update({str(k): str(v) for k, v in env_overrides.items()})
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=run_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    dt = time.time() - t0
    out = proc.stdout or ""
    _atomic_write_text(log_path, out)
    return int(proc.returncode), f"rc={proc.returncode} time_s={dt:.2f}"


def _load_report(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _state_init(
    scenario_root: str,
    judge_args: Sequence[str],
    space: Dict[str, List[float]],
    base_config: Dict[str, float],
) -> Dict[str, object]:
    return {
        "version": 1,
        "created_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "scenario_root": str(scenario_root),
        "judge_args": list(judge_args),
        "space": {k: list(v) for k, v in space.items()},
        "iterations_completed": 0,
        "no_improve_rounds": 0,
        "best": {
            "config": dict(base_config),
            "score": None,
            "report_json": "",
            "report_html": "",
            "log_path": "",
            "iteration": 0,
            "label": "baseline",
        },
        "history": [],
        "tried_hashes": [],
    }


def _append_history(state: Dict[str, object], row: Dict[str, object], max_keep: int = 20000) -> None:
    hist = state.get("history", [])
    if not isinstance(hist, list):
        hist = []
    hist.append(row)
    if len(hist) > int(max_keep):
        hist = hist[-int(max_keep) :]
    state["history"] = hist


def _as_score_list(score: Tuple[int, ...]) -> List[int]:
    return [int(x) for x in score]


def _score_better(a: Tuple[int, ...], b: Tuple[int, ...]) -> bool:
    return tuple(int(x) for x in a) < tuple(int(x) for x in b)


def _load_state(state_path: Path) -> Optional[Dict[str, object]]:
    if not state_path.exists():
        return None
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _candidate_label(iter_idx: int, cand_idx: int, suffix: str) -> str:
    return f"iter{iter_idx:04d}_cand{cand_idx:03d}_{suffix}"


def _build_judge_args(args: argparse.Namespace) -> List[str]:
    out = [
        "--scenario-root",
        str(args.scenario_root),
        "--processing-profile",
        str(args.processing_profile),
        "--max-scenarios",
        str(int(args.max_scenarios)),
        "--top-actors-per-scenario",
        str(int(args.top_actors_per_scenario)),
        "--top-windows-per-track",
        str(int(args.top_windows_per_track)),
        "--min-window-frames",
        str(int(args.min_window_frames)),
        "--event-pad-frames",
        str(int(args.event_pad_frames)),
        "--merge-gap-frames",
        str(int(args.merge_gap_frames)),
        "--raw-step-threshold",
        str(float(args.raw_step_threshold)),
        "--jump-threshold",
        str(float(args.jump_threshold)),
        "--off-raw-p95-threshold",
        str(float(args.off_raw_p95_threshold)),
        "--off-raw-max-threshold",
        str(float(args.off_raw_max_threshold)),
        "--detour-ratio-threshold",
        str(float(args.detour_ratio_threshold)),
        "--anchor-gap-threshold",
        str(float(args.anchor_gap_threshold)),
        "--goes-around-min-p95-off",
        str(float(args.goes_around_min_p95_off)),
        "--goes-around-min-raw-path",
        str(float(args.goes_around_min_raw_path)),
        "--lane-correspondence-cache-dir",
        str(args.lane_correspondence_cache_dir),
    ]
    if bool(args.only_intersection_map):
        out.append("--only-intersection-map")
    else:
        out.append("--include-all-maps")
    for p in args.scenario_name:
        out.extend(["--scenario-name", str(p)])
    if bool(args.show_pipeline_logs):
        out.append("--show-pipeline-logs")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Automated crash-resilient optimizer for intersection parameters.")
    parser.add_argument("--scenario-root", type=str, default="/data2/marco/CoLMDriver/v2xpnp/dataset/train4")
    parser.add_argument("--scenario-name", action="append", default=[], help="Scenario name/path filter (repeatable).")
    parser.add_argument("--processing-profile", type=str, default="current")
    parser.add_argument("--max-scenarios", type=int, default=0)
    parser.add_argument("--only-intersection-map", action="store_true", default=True)
    parser.add_argument("--include-all-maps", dest="only_intersection_map", action="store_false")
    parser.add_argument("--top-actors-per-scenario", type=int, default=10)
    parser.add_argument("--top-windows-per-track", type=int, default=5)
    parser.add_argument("--min-window-frames", type=int, default=4)
    parser.add_argument("--event-pad-frames", type=int, default=2)
    parser.add_argument("--merge-gap-frames", type=int, default=1)
    parser.add_argument("--raw-step-threshold", type=float, default=0.6)
    parser.add_argument("--jump-threshold", type=float, default=1.8)
    parser.add_argument("--off-raw-p95-threshold", type=float, default=2.5)
    parser.add_argument("--off-raw-max-threshold", type=float, default=4.2)
    parser.add_argument("--detour-ratio-threshold", type=float, default=1.55)
    parser.add_argument("--anchor-gap-threshold", type=float, default=1.8)
    parser.add_argument("--goes-around-min-p95-off", type=float, default=1.8)
    parser.add_argument("--goes-around-min-raw-path", type=float, default=6.0)
    parser.add_argument("--lane-correspondence-cache-dir", type=str, default="/tmp/v2x_stage_corr_cache")
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--state-path", type=str, default="")
    parser.add_argument("--best-env-path", type=str, default="")
    parser.add_argument("--best-json-path", type=str, default="")
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--candidates-per-iter", type=int, default=20)
    parser.add_argument("--random-candidates", type=int, default=6)
    parser.add_argument("--pair-candidates", type=int, default=6)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--reset-state", action="store_true", default=False)
    parser.add_argument("--show-pipeline-logs", action="store_true", default=False)
    args = parser.parse_args()

    rng = random.Random(int(args.seed))
    repo_root = REPO_ROOT
    scenario_root = Path(args.scenario_root).expanduser().resolve()
    if not scenario_root.exists():
        print(f"[ERROR] scenario root not found: {scenario_root}", file=sys.stderr)
        return 2

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if str(args.out_dir).strip() else (repo_root / "debug_runs" / f"intersection_opt_{timestamp}")
    reports_dir = out_dir / "reports"
    logs_dir = out_dir / "logs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    state_path = Path(args.state_path).expanduser().resolve() if str(args.state_path).strip() else (out_dir / "optimizer_state.json")
    best_env_path = Path(args.best_env_path).expanduser().resolve() if str(args.best_env_path).strip() else (out_dir / "best_config.env")
    best_json_path = Path(args.best_json_path).expanduser().resolve() if str(args.best_json_path).strip() else (out_dir / "best_config.json")

    space = _default_knob_space()
    defaults = _default_values()
    base_indices = _indices_from_values(space, defaults)
    base_config = _values_from_indices(space, base_indices)
    judge_args = _build_judge_args(args)

    if bool(args.reset_state) and state_path.exists():
        state_path.unlink()

    state = _load_state(state_path) if bool(args.resume) else None
    if state is None:
        state = _state_init(
            scenario_root=str(scenario_root),
            judge_args=judge_args,
            space=space,
            base_config=base_config,
        )
        _atomic_write_json(state_path, state)
        print(f"[INIT] created state: {state_path}")
    else:
        print(f"[RESUME] loaded state: {state_path}")

    # Normalize/recover state
    tried_hashes_raw = state.get("tried_hashes", [])
    tried_hashes = set(str(x) for x in tried_hashes_raw) if isinstance(tried_hashes_raw, list) else set()

    best = state.get("best", {})
    if not isinstance(best, dict):
        best = {}
        state["best"] = best
    best_config = best.get("config", {})
    if not isinstance(best_config, dict):
        best_config = dict(base_config)
    best_config = {k: float(v) for k, v in best_config.items() if k in space}
    for k in space.keys():
        if k not in best_config:
            best_config[k] = float(base_config[k])
    best_indices = _indices_from_values(space, best_config)
    best_config = _values_from_indices(space, best_indices)

    best_score: Optional[Tuple[int, ...]] = None
    raw_score = best.get("score", None)
    if isinstance(raw_score, list):
        try:
            best_score = tuple(int(x) for x in raw_score)
        except Exception:
            best_score = None

    best_report_json = Path(str(best.get("report_json", ""))).expanduser().resolve() if str(best.get("report_json", "")).strip() else None
    best_report: Optional[Dict[str, object]] = _load_report(best_report_json) if best_report_json is not None else None

    if best_score is None or best_report is None:
        label = "baseline_recover"
        cfg_hash = _config_hash(best_config)
        report_json = reports_dir / f"{label}_{cfg_hash}.json"
        report_html = reports_dir / f"{label}_{cfg_hash}.html"
        log_path = logs_dir / f"{label}_{cfg_hash}.log"
        rc, rc_msg = _run_judge(
            repo_root=repo_root,
            judge_args=judge_args,
            env_overrides=_env_from_config(best_config),
            report_json=report_json,
            report_html=report_html,
            log_path=log_path,
        )
        report = _load_report(report_json)
        if rc != 0 or report is None:
            print(f"[ERROR] baseline judge failed ({rc_msg}); see {log_path}", file=sys.stderr)
            return 2
        best_report = report
        best_score = _score_from_report(report)
        best = {
            "config": dict(best_config),
            "score": _as_score_list(best_score),
            "report_json": str(report_json),
            "report_html": str(report_html),
            "log_path": str(log_path),
            "iteration": int(state.get("iterations_completed", 0)),
            "label": label,
        }
        state["best"] = best
        tried_hashes.add(cfg_hash)
        state["tried_hashes"] = sorted(tried_hashes)
        state["updated_at_utc"] = _utc_now()
        _append_history(
            state,
            {
                "ts_utc": _utc_now(),
                "iteration": int(state.get("iterations_completed", 0)),
                "label": label,
                "config_hash": cfg_hash,
                "config": dict(best_config),
                "score": _as_score_list(best_score),
                "improved": True,
                "reason": "baseline",
                "judge_rc": int(rc),
                "report_json": str(report_json),
                "report_html": str(report_html),
                "log_path": str(log_path),
            },
        )
        _atomic_write_json(state_path, state)
        print(f"[BASE] score={best_score}")
    else:
        print(f"[BASE] resume score={best_score}")

    # Persist best snapshot outputs immediately.
    best_env_lines = [f"{k}={v}" for k, v in sorted(_env_from_config(best_config).items())]
    _atomic_write_text(best_env_path, "\n".join(best_env_lines) + "\n")
    _atomic_write_json(
        best_json_path,
        {
            "generated_at_utc": _utc_now(),
            "score": _as_score_list(best_score),
            "config": dict(best_config),
            "best_report_json": str(best.get("report_json", "")),
            "best_report_html": str(best.get("report_html", "")),
        },
    )

    iter_start = int(state.get("iterations_completed", 0)) + 1
    no_improve = int(state.get("no_improve_rounds", 0))
    max_iters = int(max(1, args.max_iters))
    cand_budget = int(max(1, args.candidates_per_iter))
    rand_budget = int(max(0, args.random_candidates))
    pair_budget = int(max(0, args.pair_candidates))
    patience = int(max(1, args.patience))

    for it in range(iter_start, max_iters + 1):
        assert best_report is not None
        assert best_score is not None
        tweak_pool = _propose_tweaks(best_report)
        if not tweak_pool:
            print(f"[ITER {it}] no tweaks proposed; stopping")
            break

        # Candidate generation: single tweaks + pair tweaks + random perturbations.
        best_indices = _indices_from_values(space, best_config)
        candidates: List[Tuple[str, Dict[str, int], str]] = []

        singles = tweak_pool[: max(1, cand_budget)]
        for idx, tw in enumerate(singles):
            cand_idx = _apply_tweaks(best_indices, [tw], space=space)
            if cand_idx is None:
                continue
            candidates.append((f"single_{idx}_{tw.knob}", cand_idx, tw.reason))

        if pair_budget > 0:
            pair_rules = tweak_pool[: min(len(tweak_pool), 10)]
            added = 0
            for a, b in itertools.combinations(pair_rules, 2):
                if a.knob == b.knob:
                    continue
                cand_idx = _apply_tweaks(best_indices, [a, b], space=space)
                if cand_idx is None:
                    continue
                candidates.append((f"pair_{a.knob}_{b.knob}", cand_idx, f"{a.reason}; {b.reason}"))
                added += 1
                if added >= int(pair_budget):
                    break

        if rand_budget > 0:
            knobs = list(space.keys())
            for ridx in range(rand_budget):
                cand_idx = dict(best_indices)
                mut_n = 1 if rng.random() < 0.65 else 2
                rng.shuffle(knobs)
                for k in knobs[:mut_n]:
                    grid = space[k]
                    cur = int(cand_idx[k])
                    step = rng.choice([-1, 1])
                    nxt = cur + int(step)
                    if nxt < 0 or nxt >= len(grid):
                        continue
                    cand_idx[k] = int(nxt)
                candidates.append((f"rand_{ridx}", cand_idx, "random_local_explore"))

        # Deduplicate + clip budget
        uniq: Dict[str, Tuple[str, Dict[str, int], str]] = {}
        for name, idxs, reason in candidates:
            cfg = _values_from_indices(space, idxs)
            h = _config_hash(cfg)
            if h in uniq:
                continue
            uniq[h] = (name, idxs, reason)
        deduped = list(uniq.items())
        rng.shuffle(deduped)
        deduped = deduped[: int(cand_budget)]

        print(f"[ITER {it}] evaluate {len(deduped)} candidates | best_score={best_score} | no_improve={no_improve}")

        iter_best_score: Optional[Tuple[int, ...]] = None
        iter_best_cfg: Optional[Dict[str, float]] = None
        iter_best_report: Optional[Dict[str, object]] = None
        iter_best_meta: Optional[Dict[str, object]] = None

        for cand_i, (cfg_hash, (label_suffix, idxs, reason)) in enumerate(deduped, start=1):
            cfg = _values_from_indices(space, idxs)
            cfg_hash = _config_hash(cfg)
            if cfg_hash in tried_hashes:
                continue
            label = _candidate_label(it, cand_i, label_suffix)
            report_json = reports_dir / f"{label}_{cfg_hash}.json"
            report_html = reports_dir / f"{label}_{cfg_hash}.html"
            log_path = logs_dir / f"{label}_{cfg_hash}.log"

            rc, rc_msg = _run_judge(
                repo_root=repo_root,
                judge_args=judge_args,
                env_overrides=_env_from_config(cfg),
                report_json=report_json,
                report_html=report_html,
                log_path=log_path,
            )
            rep = _load_report(report_json)
            if rc != 0 or rep is None:
                _append_history(
                    state,
                    {
                        "ts_utc": _utc_now(),
                        "iteration": it,
                        "label": label,
                        "config_hash": cfg_hash,
                        "config": cfg,
                        "improved": False,
                        "reason": f"judge_failed: {rc_msg}",
                        "judge_rc": rc,
                        "report_json": str(report_json),
                        "report_html": str(report_html),
                        "log_path": str(log_path),
                    },
                )
                tried_hashes.add(cfg_hash)
                continue

            score = _score_from_report(rep)
            improved_vs_global = _score_better(score, best_score)
            _append_history(
                state,
                {
                    "ts_utc": _utc_now(),
                    "iteration": it,
                    "label": label,
                    "config_hash": cfg_hash,
                    "config": cfg,
                    "score": _as_score_list(score),
                    "improved": bool(improved_vs_global),
                    "reason": str(reason),
                    "judge_rc": int(rc),
                    "report_json": str(report_json),
                    "report_html": str(report_html),
                    "log_path": str(log_path),
                },
            )
            tried_hashes.add(cfg_hash)

            if improved_vs_global:
                if iter_best_score is None or _score_better(score, iter_best_score):
                    iter_best_score = tuple(score)
                    iter_best_cfg = dict(cfg)
                    iter_best_report = rep
                    iter_best_meta = {
                        "report_json": str(report_json),
                        "report_html": str(report_html),
                        "log_path": str(log_path),
                        "label": label,
                        "reason": reason,
                    }

        improved = iter_best_score is not None and iter_best_cfg is not None and iter_best_report is not None and iter_best_meta is not None
        if improved:
            best_score = tuple(iter_best_score)  # type: ignore[arg-type]
            best_config = dict(iter_best_cfg)  # type: ignore[arg-type]
            best_report = iter_best_report  # type: ignore[assignment]
            best = {
                "config": dict(best_config),
                "score": _as_score_list(best_score),
                "report_json": str(iter_best_meta["report_json"]),
                "report_html": str(iter_best_meta["report_html"]),
                "log_path": str(iter_best_meta["log_path"]),
                "iteration": int(it),
                "label": str(iter_best_meta["label"]),
                "reason": str(iter_best_meta["reason"]),
            }
            state["best"] = best
            no_improve = 0

            best_env_lines = [f"{k}={v}" for k, v in sorted(_env_from_config(best_config).items())]
            _atomic_write_text(best_env_path, "\n".join(best_env_lines) + "\n")
            _atomic_write_json(
                best_json_path,
                {
                    "generated_at_utc": _utc_now(),
                    "iteration": int(it),
                    "score": _as_score_list(best_score),
                    "config": dict(best_config),
                    "best_report_json": str(best.get("report_json", "")),
                    "best_report_html": str(best.get("report_html", "")),
                },
            )
            print(f"[ITER {it}] accepted improvement: score={best_score}")
        else:
            no_improve += 1
            print(f"[ITER {it}] no improvement (no_improve={no_improve}/{patience})")

        state["tried_hashes"] = sorted(tried_hashes)
        state["iterations_completed"] = int(it)
        state["no_improve_rounds"] = int(no_improve)
        state["updated_at_utc"] = _utc_now()
        _atomic_write_json(state_path, state)

        if no_improve >= patience:
            print(f"[STOP] patience reached ({patience})")
            break

    print("[DONE] Optimization finished.")
    print(f"  state: {state_path}")
    print(f"  best env: {best_env_path}")
    print(f"  best json: {best_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
