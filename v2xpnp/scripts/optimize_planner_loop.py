#!/usr/bin/env python3
"""
Anytime whole-system optimizer for CARLA projection planner-likeness.

Core properties:
- Single unified loop (no duplicate frameworks)
- Adaptive parameter mutation guided by recent metric gains
- Crash-safe persistence and auto-resume
- Plateau detection + elite restarts for long overnight runs
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


KNOB_GRID: Dict[str, List[float]] = {
    # CARLA projection / postprocess knobs
    "V2X_CARLA_OPPOSITE_REJECT_DEG": [160.0, 165.0, 170.0, 175.0],
    "V2X_CARLA_WRONG_WAY_REJECT_DEG": [165.0, 170.0, 175.0, 178.0],
    "V2X_CARLA_NEAREST_CONT_SCORE_SLACK": [0.40, 0.60, 0.80, 1.00, 1.20],
    "V2X_CARLA_NEAREST_CONT_DIST_SLACK": [0.50, 0.80, 1.00, 1.20, 1.40],
    "V2X_CARLA_CORR_SCORE_MARGIN_GOOD": [0.65, 0.85, 1.05, 1.25],
    "V2X_CARLA_CORR_SCORE_MARGIN_WEAK": [0.40, 0.55, 0.75, 0.95],
    "V2X_CARLA_SMOOTH_MAX_MID_RUN": [8.0, 10.0, 12.0, 14.0, 16.0],
    "V2X_CARLA_TRANSITION_SPIKE_MAX_FRAMES": [2.0, 3.0, 4.0, 5.0, 6.0],
    "V2X_CARLA_SMOOTH_MIN_STABLE_NEIGHBOR": [16.0, 24.0, 30.0, 36.0, 44.0],
    "V2X_CARLA_FAR_MAX_NEAREST": [6.0, 6.8, 8.0, 9.0],
    "V2X_CARLA_ENABLE_MICRO_JITTER_SMOOTH": [0.0, 1.0],
    "V2X_CARLA_WEAK_SWITCH_GUARD_RAW_STEP_MAX_M": [0.6, 0.8, 0.9, 1.1],
    "V2X_CARLA_WEAK_SWITCH_GUARD_MIN_DIST_GAIN_M": [0.8, 1.1, 1.4, 1.8],
    "V2X_CARLA_WEAK_SWITCH_GUARD_MIN_SCORE_GAIN": [0.6, 0.9, 1.2, 1.5],
    "V2X_CARLA_INTERSECTION_SHAPE_MAX_QUERY_OFFSET_M": [2.4, 2.8, 3.2, 3.6],
    "V2X_CARLA_INTERSECTION_SHAPE_MAX_RAW_OFFSET_M": [2.2, 2.6, 2.8, 3.2],
    "V2X_CARLA_INTERSECTION_SHAPE_MAX_RAW_WORSEN_PEAK_M": [0.5, 0.8, 1.0, 1.2],
    "V2X_CARLA_TRANSITION_MIN_NEIGHBOR": [6.0, 8.0, 10.0, 12.0, 16.0],
    "V2X_CARLA_TRANSITION_COST_SLACK_BASE": [0.7, 1.0, 1.2, 1.5, 1.8],
    "V2X_CARLA_TRANSITION_COST_SLACK_PER_FRAME": [0.20, 0.35, 0.45, 0.60, 0.80],
    "V2X_CARLA_SMOOTH_COST_SLACK_BASE": [0.2, 0.35, 0.5, 0.7, 0.9],
    "V2X_CARLA_SMOOTH_COST_SLACK_PER_FRAME": [0.15, 0.25, 0.35, 0.50, 0.70],
    "V2X_CARLA_INTERSECTION_SHAPE_MAX_WORSEN_PER_FRAME_M": [0.35, 0.45, 0.55, 0.70, 0.90],
    "V2X_CARLA_INTERSECTION_SHAPE_MAX_RAW_WORSEN_PER_FRAME_M": [0.20, 0.28, 0.35, 0.45, 0.60],
    # V2 align-stage knobs (added for broader whole-pipeline tuning)
    "V2X_ALIGN_LANE_CHANGE_JUMP_RATIO": [1.8, 2.2, 2.4, 2.8, 3.2],
    "V2X_ALIGN_LANE_CHANGE_JUMP_ABS_M": [1.2, 1.5, 1.8, 2.2, 2.8],
    "V2X_ALIGN_LANE_CHANGE_WEAK_EVIDENCE_PENALTY": [60.0, 90.0, 120.0, 180.0, 260.0],
    "V2X_ALIGN_LANE_CHANGE_MIN_GAIN_M": [0.5, 0.7, 0.9, 1.2, 1.5],
    "V2X_ALIGN_EARLY_LANE_CHANGE_EXTRA_PENALTY": [60.0, 100.0, 140.0, 200.0, 280.0],
    "V2X_ALIGN_LANE_CHANGE_HORIZON_PENALTY": [60.0, 90.0, 120.0, 170.0, 240.0],
    "V2X_ALIGN_LANE_CHANGE_JUMP_GUARD_MIN_GAIN_M": [0.3, 0.4, 0.55, 0.8, 1.1],
    "V2X_ALIGN_LANE_CHANGE_JUMP_GUARD_PENALTY": [220.0, 320.0, 420.0, 560.0, 760.0],
    "V2X_ALIGN_OPPOSITE_SWITCH_MIN_GAIN_M": [0.7, 1.0, 1.2, 1.5, 1.9],
    "V2X_ALIGN_OPPOSITE_SWITCH_PENALTY": [350.0, 520.0, 700.0, 900.0, 1200.0],
    "V2X_ALIGN_SIGN_FLIP_STRICT_MIN_GAIN_M": [0.6, 0.8, 1.0, 1.3, 1.7],
    "V2X_ALIGN_SIGN_FLIP_STRICT_PENALTY": [260.0, 380.0, 520.0, 700.0, 920.0],
    "V2X_ALIGN_INTERSECTION_BLEND_MIN_YAW_EVIDENCE_DEG": [6.0, 9.0, 12.0, 16.0, 22.0],
    "V2X_ALIGN_INTERSECTION_BLEND_MIN_LATERAL_EVIDENCE_M": [0.4, 0.6, 0.8, 1.1, 1.5],
    "V2X_ALIGN_DIRECT_TURN_MIN_YAW_CHANGE_DEG": [10.0, 13.0, 16.0, 20.0, 26.0],
    "V2X_ALIGN_WEAK_JUMP_MIN_GAIN_M": [0.5, 0.7, 0.9, 1.2, 1.6],
    "V2X_ALIGN_WEAK_JUMP_RATIO": [1.8, 2.1, 2.4, 2.8, 3.3],
    "V2X_ALIGN_RETIME_TRANSITION_MIN_IMPROVEMENT_M": [0.20, 0.30, 0.45, 0.65, 0.90],
}

DEFAULTS: Dict[str, float] = {
    # CARLA projection / postprocess defaults
    "V2X_CARLA_OPPOSITE_REJECT_DEG": 170.0,
    "V2X_CARLA_WRONG_WAY_REJECT_DEG": 170.0,
    "V2X_CARLA_NEAREST_CONT_SCORE_SLACK": 0.60,
    "V2X_CARLA_NEAREST_CONT_DIST_SLACK": 0.80,
    "V2X_CARLA_CORR_SCORE_MARGIN_GOOD": 0.85,
    "V2X_CARLA_CORR_SCORE_MARGIN_WEAK": 0.55,
    "V2X_CARLA_SMOOTH_MAX_MID_RUN": 12.0,
    "V2X_CARLA_TRANSITION_SPIKE_MAX_FRAMES": 3.0,
    "V2X_CARLA_SMOOTH_MIN_STABLE_NEIGHBOR": 30.0,
    "V2X_CARLA_FAR_MAX_NEAREST": 8.0,
    "V2X_CARLA_ENABLE_MICRO_JITTER_SMOOTH": 0.0,
    "V2X_CARLA_WEAK_SWITCH_GUARD_RAW_STEP_MAX_M": 0.9,
    "V2X_CARLA_WEAK_SWITCH_GUARD_MIN_DIST_GAIN_M": 1.1,
    "V2X_CARLA_WEAK_SWITCH_GUARD_MIN_SCORE_GAIN": 0.9,
    "V2X_CARLA_INTERSECTION_SHAPE_MAX_QUERY_OFFSET_M": 3.0,
    "V2X_CARLA_INTERSECTION_SHAPE_MAX_RAW_OFFSET_M": 2.8,
    "V2X_CARLA_INTERSECTION_SHAPE_MAX_RAW_WORSEN_PEAK_M": 0.8,
    "V2X_CARLA_TRANSITION_MIN_NEIGHBOR": 10.0,
    "V2X_CARLA_TRANSITION_COST_SLACK_BASE": 1.2,
    "V2X_CARLA_TRANSITION_COST_SLACK_PER_FRAME": 0.45,
    "V2X_CARLA_SMOOTH_COST_SLACK_BASE": 0.5,
    "V2X_CARLA_SMOOTH_COST_SLACK_PER_FRAME": 0.35,
    "V2X_CARLA_INTERSECTION_SHAPE_MAX_WORSEN_PER_FRAME_M": 0.55,
    "V2X_CARLA_INTERSECTION_SHAPE_MAX_RAW_WORSEN_PER_FRAME_M": 0.35,
    # V2 align-stage defaults
    "V2X_ALIGN_LANE_CHANGE_JUMP_RATIO": 2.4,
    "V2X_ALIGN_LANE_CHANGE_JUMP_ABS_M": 1.8,
    "V2X_ALIGN_LANE_CHANGE_WEAK_EVIDENCE_PENALTY": 120.0,
    "V2X_ALIGN_LANE_CHANGE_MIN_GAIN_M": 0.9,
    "V2X_ALIGN_EARLY_LANE_CHANGE_EXTRA_PENALTY": 140.0,
    "V2X_ALIGN_LANE_CHANGE_HORIZON_PENALTY": 120.0,
    "V2X_ALIGN_LANE_CHANGE_JUMP_GUARD_MIN_GAIN_M": 0.55,
    "V2X_ALIGN_LANE_CHANGE_JUMP_GUARD_PENALTY": 420.0,
    "V2X_ALIGN_OPPOSITE_SWITCH_MIN_GAIN_M": 1.2,
    "V2X_ALIGN_OPPOSITE_SWITCH_PENALTY": 700.0,
    "V2X_ALIGN_SIGN_FLIP_STRICT_MIN_GAIN_M": 1.0,
    "V2X_ALIGN_SIGN_FLIP_STRICT_PENALTY": 520.0,
    "V2X_ALIGN_INTERSECTION_BLEND_MIN_YAW_EVIDENCE_DEG": 12.0,
    "V2X_ALIGN_INTERSECTION_BLEND_MIN_LATERAL_EVIDENCE_M": 0.8,
    "V2X_ALIGN_DIRECT_TURN_MIN_YAW_CHANGE_DEG": 16.0,
    "V2X_ALIGN_WEAK_JUMP_MIN_GAIN_M": 0.9,
    "V2X_ALIGN_WEAK_JUMP_RATIO": 2.4,
    "V2X_ALIGN_RETIME_TRANSITION_MIN_IMPROVEMENT_M": 0.45,
}


@dataclass
class Mutation:
    knob: str
    direction: int
    steps: int


@dataclass
class TrialResult:
    trial: int
    objective: float
    hard_ok: bool
    summary: Dict[str, float]
    report_path: str


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        out = float(v)
    except (TypeError, ValueError):
        return float(default)
    if not (out == out) or out in (float("inf"), float("-inf")):
        return float(default)
    return float(out)


def _safe_int(v: object, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return int(default)


def _closest_idx(grid: Sequence[float], value: float) -> int:
    best = 0
    best_d = float("inf")
    for i, g in enumerate(grid):
        d = abs(float(g) - float(value))
        if d < best_d:
            best_d = d
            best = int(i)
    return int(best)


def _config_from_indices(indices: Dict[str, int]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, grid in KNOB_GRID.items():
        idx = int(indices.get(k, _closest_idx(grid, DEFAULTS.get(k, float(grid[len(grid) // 2])))))
        idx = max(0, min(idx, len(grid) - 1))
        out[k] = float(grid[idx])
    return out


def _indices_from_config(config: Dict[str, float]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for k, grid in KNOB_GRID.items():
        out[k] = _closest_idx(grid, float(config.get(k, DEFAULTS.get(k, float(grid[len(grid) // 2])))))
    return out


def _indices_signature(indices: Dict[str, int]) -> str:
    parts = [f"{k}:{int(indices.get(k, 0))}" for k in sorted(KNOB_GRID.keys())]
    return "|".join(parts)


def _atomic_write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    tmp.replace(path)


def _summary_slice(report: Dict[str, object]) -> Dict[str, float]:
    summ = report.get("summary", {})
    if not isinstance(summ, dict):
        return {}
    ev = summ.get("event_totals", {})
    gm = summ.get("global_metrics", {})
    obj = summ.get("objective", {})
    sm = summ.get("stage_metrics", {})

    def _stage_val(stage: str, section: str, key: str, default: float = 0.0) -> float:
        if not isinstance(sm, dict):
            return float(default)
        row = sm.get(stage, {})
        if not isinstance(row, dict):
            return float(default)
        sec = row.get(section, {})
        if not isinstance(sec, dict):
            return float(default)
        return _safe_float(sec.get(key), float(default))

    return {
        "objective": _safe_float(obj.get("score") if isinstance(obj, dict) else 0.0, 0.0),
        "hard_pass_rate": _safe_float(summ.get("hard_pass_rate"), 0.0),
        "hard_ok_tracks": _safe_float(summ.get("hard_ok_tracks"), 0.0),
        "analyzed_tracks": _safe_float(summ.get("analyzed_tracks"), 0.0),
        "lane_jump_events": _safe_float(ev.get("lane_jump_events") if isinstance(ev, dict) else 0.0, 0.0),
        "low_motion_jump_events": _safe_float(ev.get("low_motion_jump_events") if isinstance(ev, dict) else 0.0, 0.0),
        "catastrophic_jump_events": _safe_float(
            ev.get("catastrophic_jump_events") if isinstance(ev, dict) else 0.0,
            0.0,
        ),
        "line_oscillation_events": _safe_float(
            ev.get("line_oscillation_events") if isinstance(ev, dict) else 0.0,
            0.0,
        ),
        "lane_oscillation_events": _safe_float(
            ev.get("lane_oscillation_events") if isinstance(ev, dict) else 0.0,
            0.0,
        ),
        "wrong_way_events": _safe_float(ev.get("wrong_way_events") if isinstance(ev, dict) else 0.0, 0.0),
        "facing_violation_events": _safe_float(
            ev.get("facing_violation_events") if isinstance(ev, dict) else 0.0,
            0.0,
        ),
        "intersection_bad_events": _safe_float(ev.get("intersection_bad_events") if isinstance(ev, dict) else 0.0, 0.0),
        "intersection_maneuver_inconsistent_events": _safe_float(
            ev.get("intersection_maneuver_inconsistent_events") if isinstance(ev, dict) else 0.0,
            0.0,
        ),
        "raw_carla_p95_mean_m": _safe_float(gm.get("raw_carla_p95_mean_m") if isinstance(gm, dict) else 0.0, 0.0),
        "frechet_p90_m": _safe_float(gm.get("frechet_p90_m") if isinstance(gm, dict) else 0.0, 0.0),
        "curvature_jitter_mean": _safe_float(gm.get("curvature_jitter_mean") if isinstance(gm, dict) else 0.0, 0.0),
        "yaw_step_p95_mean_deg": _safe_float(gm.get("yaw_step_p95_mean_deg") if isinstance(gm, dict) else 0.0, 0.0),
        "jerk_ratio_mean": _safe_float(gm.get("jerk_ratio_mean") if isinstance(gm, dict) else 0.0, 0.0),
        "stop_go_osc_rate_mean": _safe_float(gm.get("stop_go_osc_rate_mean") if isinstance(gm, dict) else 0.0, 0.0),
        "timestamp_nonmono_tracks": _safe_float(gm.get("timestamp_nonmono_tracks") if isinstance(gm, dict) else 0.0, 0.0),
        "timestamp_duplicate_tracks": _safe_float(gm.get("timestamp_duplicate_tracks") if isinstance(gm, dict) else 0.0, 0.0),
        "stage_v2_raw_p95_mean_m": _stage_val("v2", "raw_fidelity", "p95_mean_m", 0.0),
        "stage_carla_pre_raw_p95_mean_m": _stage_val("carla_pre", "raw_fidelity", "p95_mean_m", 0.0),
        "stage_carla_final_raw_p95_mean_m": _stage_val("carla_final", "raw_fidelity", "p95_mean_m", 0.0),
    }


def _hard_sanity_ok(
    summary: Dict[str, float],
    max_catastrophic_jumps: float,
    max_wrong_way_events: float,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    catastrophic = float(summary.get("catastrophic_jump_events", 0.0))
    wrong_way = float(summary.get("wrong_way_events", 0.0))
    nonmono = float(summary.get("timestamp_nonmono_tracks", 0.0))
    duplicate = float(summary.get("timestamp_duplicate_tracks", 0.0))
    if catastrophic > float(max_catastrophic_jumps):
        reasons.append(
            f"catastrophic_jump_events {catastrophic:.0f} > {float(max_catastrophic_jumps):.0f}"
        )
    if wrong_way > float(max_wrong_way_events):
        reasons.append(f"wrong_way_events {wrong_way:.0f} > {float(max_wrong_way_events):.0f}")
    if nonmono > 0.0:
        reasons.append(f"timestamp_nonmono_tracks {nonmono:.0f} > 0")
    if duplicate > 0.0:
        reasons.append(f"timestamp_duplicate_tracks {duplicate:.0f} > 0")
    return (len(reasons) == 0, reasons)


def _run_cmd(cmd: Sequence[str], env: Dict[str, str], cwd: Path, name: str) -> Tuple[int, str]:
    t0 = time.time()
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    dt = time.time() - t0
    tail = proc.stdout[-5000:] if proc.stdout else ""
    print(f"[{name}] exit={proc.returncode} time={dt:.1f}s")
    if tail:
        print(f"[{name}] tail:\n{tail}")
    return proc.returncode, proc.stdout or ""


def _run_iteration(
    repo_root: Path,
    dataset_root: Path,
    html_out: Path,
    metrics_out: Path,
    knobs: Dict[str, float],
    metrics_top_k: int,
) -> Dict[str, object]:
    env = dict(os.environ)
    for k, v in knobs.items():
        if float(v).is_integer():
            env[k] = str(int(v))
        else:
            env[k] = f"{float(v):.6f}".rstrip("0").rstrip(".")

    plot_cmd = [
        "python3",
        "-m",
        "v2xpnp.pipeline.entrypoint",
        str(dataset_root),
        "--multi",
        "--out",
        str(html_out),
    ]
    rc_plot, _ = _run_cmd(plot_cmd, env=env, cwd=repo_root, name="PLOT")
    if rc_plot != 0:
        raise RuntimeError("v2xpnp.pipeline.entrypoint failed")

    metrics_cmd = [
        "python3",
        "v2xpnp/scripts/planner_likeness_metrics.py",
        str(html_out),
        "--output",
        str(metrics_out),
        "--top-k",
        str(int(metrics_top_k)),
    ]
    rc_met, _ = _run_cmd(metrics_cmd, env=env, cwd=repo_root, name="METRICS")
    if rc_met != 0:
        raise RuntimeError("planner_likeness_metrics.py failed")

    return json.loads(metrics_out.read_text(encoding="utf-8"))


def _weighted_choice(rng: random.Random, items: List[str], weights: List[float]) -> str:
    total = float(sum(max(0.0, float(w)) for w in weights))
    if total <= 1e-9:
        return str(rng.choice(items))
    r = float(rng.random()) * total
    acc = 0.0
    for item, weight in zip(items, weights):
        acc += max(0.0, float(weight))
        if r <= acc:
            return str(item)
    return str(items[-1])


def _metric_driven_tweaks(summary: Dict[str, float]) -> Dict[str, int]:
    lane_jumps = float(summary.get("lane_jump_events", 0.0))
    line_osc = float(summary.get("line_oscillation_events", 0.0))
    lane_osc = float(summary.get("lane_oscillation_events", 0.0))
    wrong_way = float(summary.get("wrong_way_events", 0.0))
    facing = float(summary.get("facing_violation_events", 0.0))
    inter_bad = float(summary.get("intersection_bad_events", 0.0))
    inter_inconsistent = float(summary.get("intersection_maneuver_inconsistent_events", 0.0))
    cat_jump = float(summary.get("catastrophic_jump_events", 0.0))
    raw_p95 = float(summary.get("raw_carla_p95_mean_m", 0.0))

    out: Dict[str, int] = {}

    def _prefer(knob: str, direction: int) -> None:
        if knob not in KNOB_GRID:
            return
        if knob not in out:
            out[knob] = int(direction)

    if lane_jumps > 0.0 or line_osc > 0.0 or lane_osc > 0.0:
        _prefer("V2X_CARLA_NEAREST_CONT_SCORE_SLACK", +1)
        _prefer("V2X_CARLA_NEAREST_CONT_DIST_SLACK", +1)
        _prefer("V2X_CARLA_CORR_SCORE_MARGIN_GOOD", +1)
        _prefer("V2X_CARLA_CORR_SCORE_MARGIN_WEAK", +1)
        _prefer("V2X_CARLA_SMOOTH_MAX_MID_RUN", +1)
        _prefer("V2X_CARLA_TRANSITION_SPIKE_MAX_FRAMES", +1)
        _prefer("V2X_CARLA_WEAK_SWITCH_GUARD_MIN_DIST_GAIN_M", +1)
        _prefer("V2X_CARLA_WEAK_SWITCH_GUARD_MIN_SCORE_GAIN", +1)
        _prefer("V2X_ALIGN_LANE_CHANGE_JUMP_RATIO", -1)
        _prefer("V2X_ALIGN_LANE_CHANGE_JUMP_ABS_M", -1)
        _prefer("V2X_ALIGN_LANE_CHANGE_WEAK_EVIDENCE_PENALTY", +1)
        _prefer("V2X_ALIGN_LANE_CHANGE_MIN_GAIN_M", +1)
        _prefer("V2X_ALIGN_EARLY_LANE_CHANGE_EXTRA_PENALTY", +1)
        _prefer("V2X_ALIGN_LANE_CHANGE_HORIZON_PENALTY", +1)
        _prefer("V2X_ALIGN_LANE_CHANGE_JUMP_GUARD_MIN_GAIN_M", +1)
        _prefer("V2X_ALIGN_LANE_CHANGE_JUMP_GUARD_PENALTY", +1)
        _prefer("V2X_ALIGN_OPPOSITE_SWITCH_MIN_GAIN_M", +1)
        _prefer("V2X_ALIGN_OPPOSITE_SWITCH_PENALTY", +1)
        _prefer("V2X_ALIGN_SIGN_FLIP_STRICT_MIN_GAIN_M", +1)
        _prefer("V2X_ALIGN_SIGN_FLIP_STRICT_PENALTY", +1)

    if cat_jump > 0.0:
        _prefer("V2X_CARLA_WEAK_SWITCH_GUARD_RAW_STEP_MAX_M", -1)
        _prefer("V2X_CARLA_WEAK_SWITCH_GUARD_MIN_DIST_GAIN_M", +1)
        _prefer("V2X_CARLA_WEAK_SWITCH_GUARD_MIN_SCORE_GAIN", +1)
        _prefer("V2X_CARLA_SMOOTH_MAX_MID_RUN", +1)
        _prefer("V2X_ALIGN_WEAK_JUMP_MIN_GAIN_M", +1)
        _prefer("V2X_ALIGN_WEAK_JUMP_RATIO", -1)
        _prefer("V2X_ALIGN_RETIME_TRANSITION_MIN_IMPROVEMENT_M", +1)
        _prefer("V2X_CARLA_TRANSITION_COST_SLACK_BASE", +1)
        _prefer("V2X_CARLA_TRANSITION_COST_SLACK_PER_FRAME", +1)
        _prefer("V2X_CARLA_SMOOTH_COST_SLACK_BASE", +1)
        _prefer("V2X_CARLA_SMOOTH_COST_SLACK_PER_FRAME", +1)

    if wrong_way > 0.0 or facing > 0.0:
        _prefer("V2X_CARLA_OPPOSITE_REJECT_DEG", -1)
        _prefer("V2X_CARLA_WRONG_WAY_REJECT_DEG", -1)
        _prefer("V2X_ALIGN_OPPOSITE_SWITCH_MIN_GAIN_M", +1)
        _prefer("V2X_ALIGN_OPPOSITE_SWITCH_PENALTY", +1)
        _prefer("V2X_ALIGN_SIGN_FLIP_STRICT_MIN_GAIN_M", +1)
        _prefer("V2X_ALIGN_SIGN_FLIP_STRICT_PENALTY", +1)

    if inter_bad > 0.0 or inter_inconsistent > 0.0:
        _prefer("V2X_CARLA_TRANSITION_SPIKE_MAX_FRAMES", +1)
        _prefer("V2X_CARLA_SMOOTH_MIN_STABLE_NEIGHBOR", -1)
        _prefer("V2X_CARLA_INTERSECTION_SHAPE_MAX_QUERY_OFFSET_M", -1)
        _prefer("V2X_CARLA_INTERSECTION_SHAPE_MAX_RAW_OFFSET_M", -1)
        _prefer("V2X_CARLA_INTERSECTION_SHAPE_MAX_RAW_WORSEN_PEAK_M", -1)
        _prefer("V2X_CARLA_INTERSECTION_SHAPE_MAX_WORSEN_PER_FRAME_M", -1)
        _prefer("V2X_CARLA_INTERSECTION_SHAPE_MAX_RAW_WORSEN_PER_FRAME_M", -1)
        _prefer("V2X_ALIGN_INTERSECTION_BLEND_MIN_YAW_EVIDENCE_DEG", +1)
        _prefer("V2X_ALIGN_INTERSECTION_BLEND_MIN_LATERAL_EVIDENCE_M", +1)
        _prefer("V2X_ALIGN_DIRECT_TURN_MIN_YAW_CHANGE_DEG", +1)

    if raw_p95 > 2.0:
        _prefer("V2X_CARLA_NEAREST_CONT_SCORE_SLACK", -1)
        _prefer("V2X_CARLA_NEAREST_CONT_DIST_SLACK", -1)
        _prefer("V2X_CARLA_FAR_MAX_NEAREST", -1)
        _prefer("V2X_ALIGN_LANE_CHANGE_MIN_GAIN_M", -1)
        _prefer("V2X_ALIGN_OPPOSITE_SWITCH_MIN_GAIN_M", -1)
        _prefer("V2X_ALIGN_SIGN_FLIP_STRICT_MIN_GAIN_M", -1)
        _prefer("V2X_CARLA_TRANSITION_MIN_NEIGHBOR", -1)

    return out


def _mutate_indices(
    anchor: Dict[str, int],
    summary: Dict[str, float],
    knob_stats: Dict[str, Dict[str, float]],
    rng: random.Random,
    exploration: bool,
    min_mutations: int,
    max_mutations: int,
) -> Tuple[Dict[str, int], List[Mutation]]:
    targeted = _metric_driven_tweaks(summary)
    out = dict(anchor)
    mutations: List[Mutation] = []

    knobs = sorted(KNOB_GRID.keys())
    if not knobs:
        return out, mutations

    kmin = max(1, int(min_mutations))
    kmax = max(int(kmin), int(max_mutations))
    kcount = int(rng.randint(kmin, kmax))

    chosen: set = set()
    for _ in range(kcount):
        candidates = [k for k in knobs if k not in chosen]
        if not candidates:
            break
        weights: List[float] = []
        for k in candidates:
            st = knob_stats.get(k, {})
            attempts = max(0.0, float(_safe_float(st.get("attempts"), 0.0)))
            gain_sum = float(_safe_float(st.get("gain_sum"), 0.0))
            improv = max(0.0, float(_safe_float(st.get("improvements"), 0.0)))
            avg_gain = gain_sum / max(1.0, attempts)
            w = 1.0 + 2.0 * max(0.0, avg_gain) + 0.5 * improv
            if k in targeted:
                w += 2.5
            if exploration:
                w += 0.6
            weights.append(float(max(0.05, w)))
        knob = _weighted_choice(rng, candidates, weights)
        chosen.add(knob)

        direction = int(targeted.get(knob, 0))
        if direction == 0:
            direction = int(rng.choice([-1, +1]))
        elif exploration and rng.random() < 0.25:
            direction = -direction

        steps = 1
        if exploration and rng.random() < 0.35:
            steps = 2

        grid = KNOB_GRID[knob]
        cur = int(out.get(knob, 0))
        nxt = int(cur + direction * steps)
        if nxt < 0 or nxt >= len(grid):
            direction = -direction
            nxt = int(cur + direction * steps)
        if nxt < 0 or nxt >= len(grid):
            continue
        if nxt == cur:
            continue

        out[knob] = int(nxt)
        mutations.append(Mutation(knob=knob, direction=int(direction), steps=int(abs(nxt - cur))))

    return out, mutations


def _update_knob_stats(
    knob_stats: Dict[str, Dict[str, float]],
    before_indices: Dict[str, int],
    after_indices: Dict[str, int],
    objective_gain: float,
    improved: bool,
) -> None:
    changed = [k for k in KNOB_GRID.keys() if int(before_indices.get(k, 0)) != int(after_indices.get(k, 0))]
    for knob in changed:
        row = knob_stats.setdefault(knob, {"attempts": 0.0, "improvements": 0.0, "gain_sum": 0.0})
        row["attempts"] = float(_safe_float(row.get("attempts"), 0.0) + 1.0)
        if improved:
            row["improvements"] = float(_safe_float(row.get("improvements"), 0.0) + 1.0)
            row["gain_sum"] = float(_safe_float(row.get("gain_sum"), 0.0) + max(0.0, float(objective_gain)))


def _save_best_checkpoint(
    ckpt_dir: Path,
    trial: int,
    config: Dict[str, float],
    summary: Dict[str, float],
    report_path: Path,
    objective: float,
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_json = ckpt_dir / "best_checkpoint.json"
    best_report = ckpt_dir / "best_report.json"
    shutil.copyfile(str(report_path), str(best_report))
    payload = {
        "trial": int(trial),
        "objective": float(objective),
        "config": dict(config),
        "summary": dict(summary),
        "best_report": str(best_report),
        "updated_at_epoch_s": float(time.time()),
    }
    _atomic_write_json(best_json, payload)


def _load_resume_state(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _default_knob_stats() -> Dict[str, Dict[str, float]]:
    return {k: {"attempts": 0.0, "improvements": 0.0, "gain_sum": 0.0} for k in KNOB_GRID.keys()}


def _knob_catalog() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for k in sorted(KNOB_GRID.keys()):
        grid = [float(v) for v in KNOB_GRID[k]]
        rows.append(
            {
                "name": k,
                "default": float(DEFAULTS.get(k, grid[len(grid) // 2])),
                "grid": grid,
                "grid_size": len(grid),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Anytime whole-system planner-likeness optimizer.")
    parser.add_argument("dataset_root", type=Path, help="Dataset root passed to --multi (e.g. train4)")
    parser.add_argument("--repo-root", type=Path, default=Path("/data2/marco/CoLMDriver"), help="Repo root")
    parser.add_argument("--html-out", type=Path, default=Path("/tmp/trajectories_multi_opt.html"), help="Generated multi HTML path")
    parser.add_argument("--report-out", type=Path, default=Path("/tmp/planner_likeness_report.json"), help="Final/best report JSON path")
    parser.add_argument("--history-out", type=Path, default=Path("/tmp/planner_loop_history.json"), help="Optimizer state/history JSON path")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("/tmp/planner_loop_checkpoints"), help="Checkpoint directory")

    # Backward-compatible flags (kept):
    parser.add_argument("--max-iters", type=int, default=0, help="Max accepted improvements (0 = no limit)")
    parser.add_argument("--max-candidates", type=int, default=5, help="Candidates evaluated per round")
    parser.add_argument("--metrics-top-k", type=int, default=180, help="Top suspicious tracks stored in each report")

    # Anytime controls:
    parser.add_argument("--max-rounds", type=int, default=0, help="Max optimization rounds (0 = no limit)")
    parser.add_argument("--max-trials", type=int, default=0, help="Max total trials including baseline (0 = no limit)")
    parser.add_argument("--time-budget-s", type=float, default=0.0, help="Wall-clock budget in seconds (0 = no limit)")
    parser.add_argument("--resume", type=int, default=1, help="Resume from history state if available (1/0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for mutation sampling")

    parser.add_argument("--plateau-window", type=int, default=20, help="Window length for plateau detection")
    parser.add_argument("--plateau-min-improve", type=float, default=0.40, help="Min best-objective gain over window")
    parser.add_argument("--restart-burst-rounds", type=int, default=3, help="Exploration rounds after plateau restart")
    parser.add_argument("--explore-prob", type=float, default=0.12, help="Random exploration probability per candidate")
    parser.add_argument("--elite-size", type=int, default=8, help="Number of elite configs retained")
    parser.add_argument("--min-mutations", type=int, default=1, help="Min knobs mutated per candidate")
    parser.add_argument("--max-mutations", type=int, default=4, help="Max knobs mutated per candidate")
    parser.add_argument("--list-knobs", action="store_true", help="Print tunable knob catalog as JSON and exit")

    parser.add_argument("--max-catastrophic-jumps", type=float, default=0.0, help="Hard sanity: max catastrophic jump events")
    parser.add_argument("--max-wrong-way-events", type=float, default=0.0, help="Hard sanity: max wrong-way events")
    parser.add_argument("--improve-eps", type=float, default=1e-6, help="Minimum objective gain to count as improvement")

    args = parser.parse_args()

    if bool(args.list_knobs):
        payload = {
            "knob_count": len(KNOB_GRID),
            "knobs": _knob_catalog(),
        }
        print(json.dumps(payload, indent=2))
        return

    repo_root = args.repo_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    html_out = args.html_out.expanduser().resolve()
    report_out = args.report_out.expanduser().resolve()
    history_out = args.history_out.expanduser().resolve()
    ckpt_dir = args.checkpoint_dir.expanduser().resolve()

    html_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    history_out.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[CONFIG] knobs={} min_mutations={} max_mutations={} max_candidates={}".format(
            int(len(KNOB_GRID)),
            int(args.min_mutations),
            int(args.max_mutations),
            int(args.max_candidates),
        )
    )

    rng = random.Random(int(args.seed))

    trial_records: List[Dict[str, object]] = []
    knob_stats = _default_knob_stats()
    tried_signatures: set = set()
    elite: List[Dict[str, object]] = []
    best_obj = float("inf")
    best_indices = _indices_from_config(DEFAULTS)
    best_summary: Dict[str, float] = {}
    best_report_path: Optional[Path] = None
    accepted_improvements = 0
    round_idx = 0
    trial_idx = 0
    burst_rounds_remaining = 0
    best_obj_history: List[float] = []

    if int(args.resume) == 1:
        state = _load_resume_state(history_out)
        if isinstance(state, dict):
            try:
                rs_dataset = Path(str(state.get("dataset_root", ""))).expanduser().resolve()
                if str(rs_dataset) == str(dataset_root):
                    trial_records = list(state.get("history", [])) if isinstance(state.get("history"), list) else []
                    best_obj = float(_safe_float(state.get("best_objective"), float("inf")))
                    best_indices = {
                        k: int(v)
                        for k, v in dict(state.get("best_indices", {})).items()
                        if k in KNOB_GRID
                    }
                    if len(best_indices) != len(KNOB_GRID):
                        best_indices = _indices_from_config(DEFAULTS)
                    best_summary = {
                        k: float(_safe_float(v, 0.0))
                        for k, v in dict(state.get("best_summary", {})).items()
                    }
                    rp = str(state.get("best_report_path", "")).strip()
                    if rp:
                        best_report_path = Path(rp)
                    accepted_improvements = int(_safe_int(state.get("accepted_improvements"), 0))
                    round_idx = int(_safe_int(state.get("round_idx"), 0))
                    trial_idx = int(_safe_int(state.get("trial_idx"), len(trial_records)))
                    burst_rounds_remaining = int(_safe_int(state.get("burst_rounds_remaining"), 0))
                    tried_raw = state.get("tried_signatures", [])
                    if isinstance(tried_raw, list):
                        tried_signatures = {str(v) for v in tried_raw}
                    kb = state.get("knob_stats", {})
                    if isinstance(kb, dict):
                        for k in KNOB_GRID.keys():
                            row = kb.get(k)
                            if isinstance(row, dict):
                                knob_stats[k] = {
                                    "attempts": float(_safe_float(row.get("attempts"), 0.0)),
                                    "improvements": float(_safe_float(row.get("improvements"), 0.0)),
                                    "gain_sum": float(_safe_float(row.get("gain_sum"), 0.0)),
                                }
                    el = state.get("elite", [])
                    if isinstance(el, list):
                        elite = [e for e in el if isinstance(e, dict)]
                    bh = state.get("best_obj_history", [])
                    if isinstance(bh, list):
                        best_obj_history = [float(_safe_float(v, best_obj)) for v in bh]
                    print(
                        "[RESUME] loaded: trials={} accepted_improvements={} best_objective={:.6f}".format(
                            int(trial_idx),
                            int(accepted_improvements),
                            float(best_obj),
                        )
                    )
            except Exception:
                pass

    start_time = time.time()

    if not math.isfinite(best_obj) or not trial_records:
        base_indices = _indices_from_config(DEFAULTS)
        base_cfg = _config_from_indices(base_indices)
        base_report_path = ckpt_dir / f"trial_{0:06d}.json"
        print("[BASELINE] generating baseline report")
        base_report = _run_iteration(
            repo_root=repo_root,
            dataset_root=dataset_root,
            html_out=html_out,
            metrics_out=base_report_path,
            knobs=base_cfg,
            metrics_top_k=int(args.metrics_top_k),
        )
        base_summary = _summary_slice(base_report)
        base_obj = float(base_summary.get("objective", float("inf")))
        hard_ok, hard_reasons = _hard_sanity_ok(
            base_summary,
            max_catastrophic_jumps=float(args.max_catastrophic_jumps),
            max_wrong_way_events=float(args.max_wrong_way_events),
        )
        print(
            "[BASELINE] objective={:.6f} hard_ok={} reasons={}".format(
                float(base_obj),
                bool(hard_ok),
                hard_reasons,
            )
        )

        trial_record = {
            "trial": 0,
            "round": 0,
            "config": dict(base_cfg),
            "indices": dict(base_indices),
            "objective": float(base_obj),
            "hard_ok": bool(hard_ok),
            "hard_reasons": list(hard_reasons),
            "summary": dict(base_summary),
            "report_path": str(base_report_path),
            "mutations": [],
            "anchor": "baseline",
            "accepted": True,
            "timestamp": float(time.time()),
        }
        trial_records = [trial_record]
        trial_idx = 1
        round_idx = 1
        tried_signatures = {_indices_signature(base_indices)}

        best_obj = float(base_obj)
        best_indices = dict(base_indices)
        best_summary = dict(base_summary)
        best_report_path = Path(base_report_path)
        accepted_improvements = 0
        best_obj_history = [float(best_obj)]

        elite = [
            {
                "indices": dict(base_indices),
                "objective": float(base_obj),
                "summary": dict(base_summary),
                "trial": 0,
            }
        ]

        _save_best_checkpoint(
            ckpt_dir=ckpt_dir,
            trial=0,
            config=base_cfg,
            summary=base_summary,
            report_path=base_report_path,
            objective=float(base_obj),
        )

    while True:
        elapsed = float(time.time() - start_time)
        if float(args.time_budget_s) > 0.0 and elapsed >= float(args.time_budget_s):
            print(f"[STOP] time budget reached ({elapsed:.1f}s)")
            break
        if int(args.max_rounds) > 0 and int(round_idx) >= int(args.max_rounds):
            print(f"[STOP] max rounds reached ({round_idx})")
            break
        if int(args.max_trials) > 0 and int(trial_idx) >= int(args.max_trials):
            print(f"[STOP] max trials reached ({trial_idx})")
            break
        if int(args.max_iters) > 0 and int(accepted_improvements) >= int(args.max_iters):
            print(f"[STOP] max accepted improvements reached ({accepted_improvements})")
            break

        round_idx += 1
        round_improved = False
        round_best_obj = best_obj
        print(
            "[ROUND {:04d}] best_obj={:.6f} accepted_improvements={} trials={} burst_rounds={}".format(
                int(round_idx),
                float(best_obj),
                int(accepted_improvements),
                int(trial_idx),
                int(burst_rounds_remaining),
            )
        )

        exploration_round = bool(burst_rounds_remaining > 0)
        if burst_rounds_remaining > 0:
            burst_rounds_remaining -= 1

        for cand_idx in range(int(max(1, args.max_candidates))):
            if int(args.max_trials) > 0 and int(trial_idx) >= int(args.max_trials):
                break
            elapsed = float(time.time() - start_time)
            if float(args.time_budget_s) > 0.0 and elapsed >= float(args.time_budget_s):
                break

            explore = bool(exploration_round or (rng.random() < float(max(0.0, min(1.0, args.explore_prob)))))

            anchor_indices = dict(best_indices)
            anchor_label = "best"
            if elite and explore and rng.random() < 0.65:
                anchor = rng.choice(elite)
                if isinstance(anchor, dict):
                    idxs = anchor.get("indices", {})
                    if isinstance(idxs, dict):
                        anchor_indices = {k: int(v) for k, v in idxs.items() if k in KNOB_GRID}
                        if len(anchor_indices) != len(KNOB_GRID):
                            anchor_indices = dict(best_indices)
                        anchor_label = "elite"

            cand_indices, muts = _mutate_indices(
                anchor=anchor_indices,
                summary=best_summary,
                knob_stats=knob_stats,
                rng=rng,
                exploration=bool(explore),
                min_mutations=int(args.min_mutations),
                max_mutations=int(args.max_mutations),
            )

            sig = _indices_signature(cand_indices)
            retries = 0
            while sig in tried_signatures and retries < 20:
                cand_indices, muts = _mutate_indices(
                    anchor=anchor_indices,
                    summary=best_summary,
                    knob_stats=knob_stats,
                    rng=rng,
                    exploration=True,
                    min_mutations=max(1, int(args.min_mutations)),
                    max_mutations=max(2, int(args.max_mutations) + 1),
                )
                sig = _indices_signature(cand_indices)
                retries += 1
            if sig in tried_signatures:
                continue
            tried_signatures.add(sig)

            cand_cfg = _config_from_indices(cand_indices)
            cand_report_path = ckpt_dir / f"trial_{trial_idx:06d}.json"
            print(
                "[TRIAL {:06d}] cand#{:02d} anchor={} explore={} mutations={}".format(
                    int(trial_idx),
                    int(cand_idx + 1),
                    str(anchor_label),
                    bool(explore),
                    [f"{m.knob}:{m.direction:+d}x{m.steps}" for m in muts],
                )
            )

            try:
                cand_report = _run_iteration(
                    repo_root=repo_root,
                    dataset_root=dataset_root,
                    html_out=html_out,
                    metrics_out=cand_report_path,
                    knobs=cand_cfg,
                    metrics_top_k=int(args.metrics_top_k),
                )
            except Exception as exc:
                trial_records.append(
                    {
                        "trial": int(trial_idx),
                        "round": int(round_idx),
                        "config": dict(cand_cfg),
                        "indices": dict(cand_indices),
                        "objective": float("inf"),
                        "hard_ok": False,
                        "hard_reasons": [f"runtime_error: {exc}"],
                        "summary": {},
                        "report_path": str(cand_report_path),
                        "mutations": [m.__dict__ for m in muts],
                        "anchor": str(anchor_label),
                        "accepted": False,
                        "timestamp": float(time.time()),
                    }
                )
                trial_idx += 1
                continue

            cand_summary = _summary_slice(cand_report)
            cand_obj = float(cand_summary.get("objective", float("inf")))
            hard_ok, hard_reasons = _hard_sanity_ok(
                cand_summary,
                max_catastrophic_jumps=float(args.max_catastrophic_jumps),
                max_wrong_way_events=float(args.max_wrong_way_events),
            )
            improved = bool(hard_ok and (cand_obj + float(args.improve_eps) < float(best_obj)))
            obj_gain = float(best_obj - cand_obj) if math.isfinite(best_obj) and math.isfinite(cand_obj) else 0.0

            _update_knob_stats(
                knob_stats=knob_stats,
                before_indices=anchor_indices,
                after_indices=cand_indices,
                objective_gain=float(max(0.0, obj_gain)),
                improved=bool(improved),
            )

            trial_row = {
                "trial": int(trial_idx),
                "round": int(round_idx),
                "config": dict(cand_cfg),
                "indices": dict(cand_indices),
                "objective": float(cand_obj),
                "hard_ok": bool(hard_ok),
                "hard_reasons": list(hard_reasons),
                "summary": dict(cand_summary),
                "report_path": str(cand_report_path),
                "mutations": [m.__dict__ for m in muts],
                "anchor": str(anchor_label),
                "accepted": bool(improved),
                "timestamp": float(time.time()),
            }
            trial_records.append(trial_row)
            print(
                "[TRIAL {:06d}] objective={:.6f} hard_ok={} improved={} reasons={}".format(
                    int(trial_idx),
                    float(cand_obj),
                    bool(hard_ok),
                    bool(improved),
                    hard_reasons,
                )
            )

            if improved:
                best_obj = float(cand_obj)
                best_indices = dict(cand_indices)
                best_summary = dict(cand_summary)
                best_report_path = Path(cand_report_path)
                accepted_improvements += 1
                round_improved = True
                round_best_obj = min(float(round_best_obj), float(cand_obj))

                elite.append(
                    {
                        "indices": dict(cand_indices),
                        "objective": float(cand_obj),
                        "summary": dict(cand_summary),
                        "trial": int(trial_idx),
                    }
                )
                elite = sorted(elite, key=lambda r: float(_safe_float(r.get("objective"), float("inf"))))
                elite = elite[: max(1, int(args.elite_size))]

                _save_best_checkpoint(
                    ckpt_dir=ckpt_dir,
                    trial=int(trial_idx),
                    config=cand_cfg,
                    summary=cand_summary,
                    report_path=cand_report_path,
                    objective=float(cand_obj),
                )

            trial_idx += 1

        best_obj_history.append(float(best_obj))

        # Plateau detection based on moving best-objective window.
        plateau = False
        if len(best_obj_history) >= int(args.plateau_window) + 1:
            old = float(best_obj_history[-(int(args.plateau_window) + 1)])
            new = float(best_obj_history[-1])
            gain = float(old - new)
            plateau = bool(gain < float(args.plateau_min_improve))
            if plateau:
                print(
                    "[PLATEAU] window_gain={:.6f} < min_improve={:.6f}".format(
                        float(gain),
                        float(args.plateau_min_improve),
                    )
                )

        if plateau and not round_improved:
            burst_rounds_remaining = max(int(burst_rounds_remaining), int(args.restart_burst_rounds))
            if elite:
                pick = rng.choice(elite)
                idxs = pick.get("indices", {}) if isinstance(pick, dict) else {}
                if isinstance(idxs, dict):
                    best_indices = {k: int(v) for k, v in idxs.items() if k in KNOB_GRID}
                    if len(best_indices) != len(KNOB_GRID):
                        best_indices = _indices_from_config(DEFAULTS)
                    print(
                        "[RESTART] seeded from elite trial={} objective={:.6f} burst_rounds={}".format(
                            int(_safe_int(pick.get("trial"), -1)) if isinstance(pick, dict) else -1,
                            float(_safe_float(pick.get("objective"), best_obj)) if isinstance(pick, dict) else float(best_obj),
                            int(burst_rounds_remaining),
                        )
                    )

        state_payload = {
            "version": 2,
            "dataset_root": str(dataset_root),
            "repo_root": str(repo_root),
            "round_idx": int(round_idx),
            "trial_idx": int(trial_idx),
            "accepted_improvements": int(accepted_improvements),
            "best_objective": float(best_obj),
            "best_indices": dict(best_indices),
            "best_config": _config_from_indices(best_indices),
            "best_summary": dict(best_summary),
            "best_report_path": str(best_report_path) if best_report_path is not None else "",
            "burst_rounds_remaining": int(burst_rounds_remaining),
            "tried_signatures": sorted(list(tried_signatures)),
            "knob_stats": knob_stats,
            "elite": elite,
            "best_obj_history": best_obj_history[-400:],
            "history": trial_records,
            "updated_at_epoch_s": float(time.time()),
        }
        _atomic_write_json(history_out, state_payload)

    # Finalize outputs.
    if best_report_path is not None and best_report_path.exists():
        shutil.copyfile(str(best_report_path), str(report_out))
    final_config = _config_from_indices(best_indices)
    final_summary = dict(best_summary)
    final_state_payload = {
        "version": 2,
        "dataset_root": str(dataset_root),
        "repo_root": str(repo_root),
        "round_idx": int(round_idx),
        "trial_idx": int(trial_idx),
        "accepted_improvements": int(accepted_improvements),
        "best_objective": float(best_obj),
        "best_indices": dict(best_indices),
        "best_config": dict(final_config),
        "best_summary": dict(final_summary),
        "best_report_path": str(best_report_path) if best_report_path is not None else "",
        "burst_rounds_remaining": int(burst_rounds_remaining),
        "tried_signatures": sorted(list(tried_signatures)),
        "knob_stats": knob_stats,
        "elite": elite,
        "best_obj_history": best_obj_history[-400:],
        "history": trial_records,
        "updated_at_epoch_s": float(time.time()),
    }
    _atomic_write_json(history_out, final_state_payload)

    print("[FINAL] " + json.dumps(final_summary, sort_keys=True))
    print("[FINAL_CONFIG] " + json.dumps(final_config, sort_keys=True))
    print(f"[OK] Best report: {report_out}")
    print(f"[OK] History/state: {history_out}")
    print(f"[OK] Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    main()
