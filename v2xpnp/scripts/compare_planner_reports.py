#!/usr/bin/env python3
"""
Compare planner_likeness_metrics reports (checkpoint vs candidate).

Designed to answer: "Are metrics identical under the same scope?"
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _safe_float(v: object, default: float = float("nan")) -> float:
    try:
        out = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_keys(a: Dict[str, object], b: Dict[str, object]) -> Iterable[str]:
    return sorted(set(a.keys()) | set(b.keys()))


def _diff_scalar(a: object, b: object) -> Tuple[str, float]:
    af = _safe_float(a)
    bf = _safe_float(b)
    if math.isfinite(af) and math.isfinite(bf):
        return ("float", bf - af)
    return ("raw", 0.0 if a == b else 1.0)


def _compare_map(
    label: str,
    ref: Dict[str, object],
    cand: Dict[str, object],
    tol: float,
) -> Tuple[int, List[str]]:
    bad = 0
    lines: List[str] = []
    for k in _iter_keys(ref, cand):
        ra = ref.get(k)
        cb = cand.get(k)
        kind, d = _diff_scalar(ra, cb)
        if kind == "float":
            same = abs(float(d)) <= float(tol)
            if not same:
                bad += 1
                lines.append(
                    f"{label}.{k}: ref={_safe_float(ra, float('nan')):.12g} cand={_safe_float(cb, float('nan')):.12g} delta={float(d):+.12g}"
                )
        else:
            if ra != cb:
                bad += 1
                lines.append(f"{label}.{k}: ref={ra!r} cand={cb!r}")
    return bad, lines


def _summary_slice(report: Dict[str, object]) -> Dict[str, object]:
    summ = report.get("summary", {})
    if not isinstance(summ, dict):
        return {}
    out: Dict[str, object] = {}
    for k in ("hard_pass_rate", "hard_ok_tracks", "analyzed_tracks", "total_tracks"):
        if k in summ:
            out[k] = summ.get(k)
    ev = summ.get("event_totals", {})
    if isinstance(ev, dict):
        for k in (
            "lane_jump_events",
            "aba_flicker_events",
            "wrong_way_events",
            "intersection_bad_events",
            "reverse_motion_events",
            "overlap_events",
            "overlap_pair_events",
        ):
            if k in ev:
                out[f"event.{k}"] = ev.get(k)
    gm = summ.get("global_metrics", {})
    if isinstance(gm, dict):
        for k in (
            "raw_carla_mean_mean_m",
            "raw_carla_p95_mean_m",
            "frechet_mean_m",
            "frechet_p90_m",
            "hausdorff_mean_m",
            "curvature_jitter_mean",
            "yaw_step_p95_mean_deg",
            "jerk_ratio_mean",
            "stop_go_osc_rate_mean",
            "overlap_max_pen_mean_m",
            "overlap_max_pen_p95_m",
        ):
            if k in gm:
                out[f"global.{k}"] = gm.get(k)
    obj = summ.get("objective", {})
    if isinstance(obj, dict) and "score" in obj:
        out["objective.score"] = obj.get("score")
    return out


def _scenario_slice(report: Dict[str, object], scenario: str) -> Dict[str, object]:
    sc = report.get("scenario_summary", {})
    if not isinstance(sc, dict):
        return {}
    row = sc.get(str(scenario), {})
    if not isinstance(row, dict):
        return {}
    out: Dict[str, object] = {}
    for k in ("hard_pass_rate", "hard_ok_tracks", "analyzed_tracks", "total_tracks"):
        if k in row:
            out[k] = row.get(k)
    ev = row.get("event_totals", {})
    if isinstance(ev, dict):
        for k in (
            "lane_jump_events",
            "aba_flicker_events",
            "wrong_way_events",
            "intersection_bad_events",
            "reverse_motion_events",
            "overlap_events",
            "overlap_pair_events",
        ):
            if k in ev:
                out[f"event.{k}"] = ev.get(k)
    gm = row.get("global_metrics", {})
    if isinstance(gm, dict):
        for k in (
            "raw_carla_p95_mean_m",
            "frechet_p90_m",
            "hausdorff_mean_m",
            "curvature_jitter_mean",
            "yaw_step_p95_mean_deg",
            "overlap_max_pen_mean_m",
        ):
            if k in gm:
                out[f"global.{k}"] = gm.get(k)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare planner-likeness reports.")
    ap.add_argument("reference", type=Path, help="Reference report JSON")
    ap.add_argument("candidate", type=Path, help="Candidate report JSON")
    ap.add_argument(
        "--scenarios",
        type=str,
        default="",
        help="Comma-separated scenario names to compare in scenario_summary.",
    )
    ap.add_argument(
        "--ignore-summary",
        action="store_true",
        help="Skip top-level summary comparison (useful for multi-vs-single comparisons).",
    )
    ap.add_argument("--tol", type=float, default=1e-9, help="Absolute tolerance for numeric equality.")
    args = ap.parse_args()

    ref = _load_json(args.reference)
    cand = _load_json(args.candidate)

    total_bad = 0

    if not bool(args.ignore_summary):
        ref_sum = _summary_slice(ref)
        cand_sum = _summary_slice(cand)
        bad, lines = _compare_map("summary", ref_sum, cand_sum, tol=float(args.tol))
        total_bad += bad
        print(f"[COMPARE] summary mismatches: {bad}")
        for ln in lines[:200]:
            print("  " + ln)
        if len(lines) > 200:
            print(f"  ... ({len(lines) - 200} more)")

    scenarios = [s.strip() for s in str(args.scenarios).split(",") if s.strip()]
    for sc in scenarios:
        ref_sc = _scenario_slice(ref, sc)
        cand_sc = _scenario_slice(cand, sc)
        bad_sc, lines_sc = _compare_map(f"scenario[{sc}]", ref_sc, cand_sc, tol=float(args.tol))
        total_bad += bad_sc
        print(f"[COMPARE] {sc} mismatches: {bad_sc}")
        for ln in lines_sc[:200]:
            print("  " + ln)
        if len(lines_sc) > 200:
            print(f"  ... ({len(lines_sc) - 200} more)")

    if total_bad == 0:
        print("[COMPARE] EXACT_MATCH")
    else:
        print(f"[COMPARE] NOT_MATCH total_mismatches={total_bad}")


if __name__ == "__main__":
    main()
