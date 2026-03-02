#!/usr/bin/env python3
"""Pipeline audit helpers focused on blocker concentration.

This is intended for post-run analysis of optimization checkpoints, with
special visibility into intersection and wrong-way/catastrophic blockers.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


def _load_trials(checkpoint_dir: Path) -> List[Tuple[int, Dict[str, object]]]:
    rows: List[Tuple[int, Dict[str, object]]] = []
    for p in sorted(checkpoint_dir.glob("trial_*.json")):
        try:
            trial = int(p.stem.split("_")[1])
        except Exception:
            continue
        rows.append((trial, json.loads(p.read_text(encoding="utf-8"))))
    return rows


def _summary_row(trial: int, report: Dict[str, object]) -> Dict[str, object]:
    summary = report.get("summary", {})
    events = summary.get("event_totals", {}) if isinstance(summary, dict) else {}
    global_metrics = summary.get("global_metrics", {}) if isinstance(summary, dict) else {}
    objective = summary.get("objective", {}) if isinstance(summary, dict) else {}
    return {
        "trial": int(trial),
        "objective": float(objective.get("score", 0.0)) if isinstance(objective, dict) else 0.0,
        "hard_pass_rate": float(summary.get("hard_pass_rate", 0.0)) if isinstance(summary, dict) else 0.0,
        "lane_jump_events": int(events.get("lane_jump_events", 0)) if isinstance(events, dict) else 0,
        "catastrophic_jump_events": int(events.get("catastrophic_jump_events", 0)) if isinstance(events, dict) else 0,
        "wrong_way_events": int(events.get("wrong_way_events", 0)) if isinstance(events, dict) else 0,
        "intersection_bad_events": int(events.get("intersection_bad_events", 0)) if isinstance(events, dict) else 0,
        "intersection_maneuver_inconsistent_events": int(events.get("intersection_maneuver_inconsistent_events", 0))
        if isinstance(events, dict)
        else 0,
        "raw_carla_p95_mean_m": float(global_metrics.get("raw_carla_p95_mean_m", 0.0))
        if isinstance(global_metrics, dict)
        else 0.0,
    }


def _top_blocker_scenarios(report: Dict[str, object], key: str, top_k: int = 5) -> List[Tuple[str, int]]:
    scenario_summary = report.get("scenario_summary", {})
    if not isinstance(scenario_summary, dict):
        return []
    rows: List[Tuple[str, int]] = []
    for scenario, payload in scenario_summary.items():
        if not isinstance(payload, dict):
            continue
        events = payload.get("event_totals", {})
        if not isinstance(events, dict):
            continue
        v = int(events.get(key, 0))
        if v > 0:
            rows.append((str(scenario), int(v)))
    rows.sort(key=lambda x: (-x[1], x[0]))
    return rows[: max(1, int(top_k))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit planner loop trials for blocker concentration.")
    parser.add_argument("run_dir", type=Path, help="Run directory that contains checkpoints/")
    parser.add_argument("--top-k", type=int, default=10, help="Top objective runs to inspect")
    parser.add_argument("--scenario-top-k", type=int, default=5, help="Top scenarios per blocker key")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    ckpt_dir = run_dir / "checkpoints"
    trials = _load_trials(ckpt_dir)
    if not trials:
        raise SystemExit(f"No trial_*.json files found under {ckpt_dir}")

    rows = [_summary_row(t, r) for t, r in trials]
    rows_by_obj = sorted(rows, key=lambda x: float(x["objective"]))
    top = rows_by_obj[: max(1, int(args.top_k))]

    print(f"[AUDIT] run_dir={run_dir}")
    print(f"[AUDIT] trial_count={len(rows)}")

    print("[AUDIT] top objective rows")
    for row in top:
        print(
            " trial={trial:03d} obj={objective:.3f} hard={hard_pass_rate:.4f} "
            "lane={lane_jump_events} cat={catastrophic_jump_events} ww={wrong_way_events} "
            "inter_bad={intersection_bad_events} inter_inc={intersection_maneuver_inconsistent_events}".format(**row)
        )

    wrong_way_dist = Counter(int(r["wrong_way_events"]) for r in rows)
    cat_dist = Counter(int(r["catastrophic_jump_events"]) for r in rows)
    print(f"[AUDIT] wrong_way_distribution={dict(sorted(wrong_way_dist.items()))}")
    print(f"[AUDIT] catastrophic_distribution={dict(sorted(cat_dist.items()))}")

    best_trial_idx = int(top[0]["trial"])
    best_report = next(r for t, r in trials if int(t) == int(best_trial_idx))
    print("[AUDIT] top wrong-way scenarios (best-objective trial)")
    for sc, val in _top_blocker_scenarios(best_report, "wrong_way_events", top_k=int(args.scenario_top_k)):
        print(f"  {sc}: {val}")
    print("[AUDIT] top catastrophic scenarios (best-objective trial)")
    for sc, val in _top_blocker_scenarios(best_report, "catastrophic_jump_events", top_k=int(args.scenario_top_k)):
        print(f"  {sc}: {val}")
    print("[AUDIT] top intersection inconsistency scenarios (best-objective trial)")
    for sc, val in _top_blocker_scenarios(best_report, "intersection_maneuver_inconsistent_events", top_k=int(args.scenario_top_k)):
        print(f"  {sc}: {val}")


if __name__ == "__main__":
    main()

