#!/usr/bin/env python3
"""
Audit schema-stage outputs against category requirements.

Usage:
  python tools/audit_schema_outputs.py --glob "debug_runs/20260226_131*"
  python tools/audit_schema_outputs.py --glob "debug_runs/20260226_131*" --json-out debug_runs/schema_audit.json
"""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scenario_generator") not in sys.path:
    sys.path.insert(0, str(ROOT / "scenario_generator"))

from scenario_generator.capabilities import CATEGORY_DEFINITIONS  # noqa: E402


@dataclass
class Issue:
    severity: str  # "error" | "warning"
    category: str
    message: str


def _load_spec_from_run(run_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    out_path = run_dir / "01_schema" / "output.json"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    return payload, payload.get("spec", {})


def _vehicle_entries(spec: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for v in spec.get("ego_vehicles", []):
        if not isinstance(v, dict):
            continue
        vid = str(v.get("vehicle_id", "")).strip()
        if vid:
            out[vid] = str(v.get("entry_road", "unknown")).strip().lower()
    return out


def _constraint_pairs(spec: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    pairs: List[Tuple[str, str, str]] = []
    for c in spec.get("vehicle_constraints", []):
        if not isinstance(c, dict):
            continue
        ctype = str(c.get("type", "")).strip().lower()
        a = str(c.get("a", "")).strip()
        b = str(c.get("b", "")).strip()
        if ctype and a and b:
            pairs.append((ctype, a, b))
    return pairs


def _audit_spec(spec: Dict[str, Any]) -> List[Issue]:
    issues: List[Issue] = []
    category = str(spec.get("category", "")).strip()
    cat_info = CATEGORY_DEFINITIONS.get(category)
    if cat_info is None:
        return [Issue("error", "general", f"Unknown category '{category}'")]

    topology = str(spec.get("topology", "")).strip()
    if topology != cat_info.map.topology.value:
        issues.append(
            Issue(
                "error",
                "general",
                f"Topology mismatch: got '{topology}', expected '{cat_info.map.topology.value}'",
            )
        )

    for flag in ("needs_oncoming", "needs_multi_lane", "needs_on_ramp", "needs_merge"):
        required = bool(getattr(cat_info.map, flag))
        actual = bool(spec.get(flag, False))
        if required and not actual:
            issues.append(Issue("error", "general", f"Required flag missing: {flag}=true"))

    vehicles = spec.get("ego_vehicles", [])
    vehicle_ids = [str(v.get("vehicle_id", "")).strip() for v in vehicles if isinstance(v, dict)]
    if len(vehicle_ids) != len(set(vehicle_ids)):
        issues.append(Issue("error", "general", "Duplicate vehicle IDs"))

    known_ids = set(vehicle_ids)
    pairs = _constraint_pairs(spec)
    seen: Set[Tuple[str, str, str]] = set()
    for ctype, a, b in pairs:
        if a not in known_ids or b not in known_ids:
            issues.append(Issue("error", "general", f"Constraint references unknown vehicle: {ctype}({a}->{b})"))
        key = (ctype, a, b)
        if key in seen:
            issues.append(Issue("warning", "general", f"Duplicate constraint: {ctype}({a}->{b})"))
        seen.add(key)

    # Detect contradictory two-way merge constraints for the same pair.
    merge_pairs = {(a, b) for ctype, a, b in pairs if ctype == "merges_into_lane_of"}
    seen_bidirectional: Set[frozenset[str]] = set()
    for a, b in list(merge_pairs):
        if (b, a) not in merge_pairs:
            continue
        key = frozenset((a, b))
        if key in seen_bidirectional:
            continue
        seen_bidirectional.add(key)
        issues.append(Issue("warning", "general", f"Mutual merge detected between {a} and {b}"))

    # Interaction coverage.
    covered = set()
    for _ctype, a, b in pairs:
        covered.add(a)
        covered.add(b)
    for actor in spec.get("actors", []):
        if isinstance(actor, dict):
            aff = str(actor.get("affects_vehicle", "")).strip()
            if aff:
                covered.add(aff)
    uncovered = [vid for vid in vehicle_ids if vid not in covered]
    if uncovered:
        issues.append(Issue("warning", "general", f"Vehicles not participating in interactions: {', '.join(uncovered)}"))

    entries = _vehicle_entries(spec)
    maneuvers = {
        str(v.get("vehicle_id", "")).strip(): str(v.get("maneuver", "unknown")).strip().lower()
        for v in vehicles
        if isinstance(v, dict)
    }
    actors = [a for a in spec.get("actors", []) if isinstance(a, dict)]
    ctype_set = {t for t, _, _ in pairs}

    def has_merge_side_to_main() -> bool:
        for ctype, a, b in pairs:
            if ctype != "merges_into_lane_of":
                continue
            if entries.get(a) == "side" and entries.get(b) == "main":
                return True
        return False

    def has_merge_main_to_side() -> bool:
        for ctype, a, b in pairs:
            if ctype != "merges_into_lane_of":
                continue
            if entries.get(a) == "main" and entries.get(b) == "side":
                return True
        return False

    # Category-level checks.
    if category == "Highway On-Ramp Merge":
        if "side" not in set(entries.values()) or "main" not in set(entries.values()):
            issues.append(Issue("error", category, "Requires both side-road and main-road entry vehicles"))
        if not has_merge_side_to_main():
            issues.append(Issue("error", category, "Requires merges_into_lane_of from side -> main"))
        if has_merge_main_to_side():
            issues.append(Issue("error", category, "Invalid reverse merge detected: main -> side"))

    if category == "Interactive Lane Change":
        bad = [vid for vid, m in maneuvers.items() if m != "lane_change"]
        if bad:
            issues.append(Issue("error", category, f"All vehicles must lane_change (non-compliant: {', '.join(bad)})"))
        if actors:
            issues.append(Issue("warning", category, "Category typically avoids non-ego actors"))

    if category == "Blocked Lane (Obstacle)":
        has_obstacle = any(str(a.get("kind", "")).lower() in {"static_prop", "parked_vehicle"} for a in actors)
        if not has_obstacle:
            issues.append(Issue("error", category, "Requires a blocking static obstacle actor"))
        bad = [vid for vid, m in maneuvers.items() if m == "lane_change"]
        if bad:
            issues.append(Issue("error", category, f"Avoid lane_change maneuvers (found: {', '.join(bad)})"))
        if not ctype_set.intersection({"left_lane_of", "right_lane_of", "same_lane_as"}):
            issues.append(Issue("warning", category, "Expected adjacent-lane relation for blocked-lane context"))

    if category == "Lane Drop / Alternating Merge":
        if "merges_into_lane_of" not in ctype_set:
            issues.append(Issue("error", category, "Requires merge constraints"))
        has_obstacle = any(str(a.get("kind", "")).lower() in {"static_prop", "parked_vehicle"} for a in actors)
        if not has_obstacle:
            issues.append(Issue("error", category, "Requires an obstacle near the dropped-lane merge point"))

    if category == "Major/Minor Unsignalized Entry":
        if "side" not in set(entries.values()) or "main" not in set(entries.values()):
            issues.append(Issue("error", category, "Requires side-street + main-road vehicles"))
        if not has_merge_side_to_main():
            issues.append(Issue("error", category, "Requires side-street vehicle merging into main-road vehicle lane"))
        if len(vehicle_ids) >= 3 and "opposite_approach_of" not in ctype_set:
            issues.append(Issue("error", category, "For ego_count>=3, expected an oncoming main-road interaction"))
        if len(vehicle_ids) >= 4:
            has_main_to_side_turn = any(
                entries.get(vid) == "main" and str(v.get("exit_road", "")).strip().lower() == "side"
                for vid, v in [(str(v.get("vehicle_id", "")).strip(), v) for v in vehicles if isinstance(v, dict)]
            )
            if not has_main_to_side_turn:
                issues.append(Issue("error", category, "For ego_count>=4, expected a main-road vehicle turning into the side road"))

    if category == "Overtaking on Two-Lane Road":
        has_blocking_obstacle = any(
            str(a.get("kind", "")).lower() in {"static_prop", "parked_vehicle"}
            for a in actors
        )
        if not has_blocking_obstacle:
            issues.append(Issue("error", category, "Requires a blocking static obstacle"))
        bad = [vid for vid, m in maneuvers.items() if m == "lane_change"]
        if bad:
            issues.append(Issue("error", category, f"Avoid lane_change maneuvers (found: {', '.join(bad)})"))
        opposite_pairs = {(a, b) for ctype, a, b in pairs if ctype == "opposite_approach_of"}
        if ("Vehicle 1", "Vehicle 2") not in opposite_pairs and ("Vehicle 2", "Vehicle 1") not in opposite_pairs:
            issues.append(Issue("error", category, "Requires opposite_approach_of between Vehicle 1 and Vehicle 2"))
        illegal_opposites = [
            (a, b)
            for a, b in opposite_pairs
            if set((a, b)) != {"Vehicle 1", "Vehicle 2"}
        ]
        if illegal_opposites:
            issues.append(Issue("error", category, f"Unexpected opposite_approach_of pairs: {illegal_opposites}"))
        if ctype_set.intersection({"left_lane_of", "right_lane_of", "same_lane_as", "merges_into_lane_of"}):
            issues.append(Issue("error", category, "Two-lane overtaking should not include lane adjacency/merge constraints"))

    if category == "Pedestrian Crosswalk":
        has_crossing_walker = any(
            str(a.get("kind", "")).lower() == "walker" and str(a.get("motion", "")).lower() == "cross_perpendicular"
            for a in actors
        )
        if not has_crossing_walker:
            issues.append(Issue("error", category, "Requires at least one crossing walker actor"))

    if category == "Roundabout Navigation":
        bad = [vid for vid, m in maneuvers.items() if m == "lane_change"]
        if bad:
            issues.append(Issue("error", category, f"Avoid lane_change maneuvers in roundabout scenarios (found: {', '.join(bad)})"))
        if actors:
            issues.append(Issue("warning", category, "Roundabout category avoids props/pedestrians/cyclists; actor usage is suspicious"))

    if category == "Unprotected Left Turn":
        has_left = any(m == "left" for m in maneuvers.values())
        has_straight = any(m == "straight" for m in maneuvers.values())
        has_opposite = "opposite_approach_of" in ctype_set
        if not (has_left and has_straight and has_opposite):
            issues.append(Issue("error", category, "Requires left-turn + straight oncoming + opposite_approach_of"))
        if maneuvers.get("Vehicle 1") != "left":
            issues.append(Issue("warning", category, "Must-include text expects Vehicle 1 to be the left-turn vehicle"))

    if category == "Intersection Deadlock Resolution":
        if len(vehicle_ids) < 3:
            issues.append(Issue("error", category, "Expected at least 3 interacting vehicles"))
        if uncovered:
            issues.append(Issue("error", category, "All vehicles should be behaviorally interdependent"))

    if category == "Construction Zone":
        static_count = sum(1 for a in actors if str(a.get("kind", "")).lower() in {"static_prop", "parked_vehicle"})
        if static_count < 2:
            issues.append(Issue("warning", category, "Construction zone usually needs clustered/static prop groupings"))
        if any(str(a.get("kind", "")).lower() in {"walker", "cyclist"} for a in actors):
            issues.append(Issue("error", category, "Construction Zone should not include pedestrians/cyclists"))

    return issues


def _resolve_run_dirs(glob_patterns: List[str]) -> List[Path]:
    run_dirs: List[Path] = []
    for pattern in glob_patterns:
        for match in sorted(glob.glob(pattern)):
            p = Path(match)
            if p.is_file() and p.name == "output.json" and p.parent.name == "01_schema":
                run_dirs.append(p.parents[1])
                continue
            if p.is_dir() and (p / "01_schema" / "output.json").exists():
                run_dirs.append(p)
    # Stable de-dup.
    unique: List[Path] = []
    seen = set()
    for rd in run_dirs:
        key = str(rd.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(rd)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit schema stage outputs for category consistency.")
    parser.add_argument(
        "--glob",
        nargs="+",
        default=["debug_runs/*"],
        help="Glob(s) matching run directories or 01_schema/output.json files.",
    )
    parser.add_argument("--json-out", default="", help="Optional path to write JSON report.")
    args = parser.parse_args()

    run_dirs = _resolve_run_dirs(args.glob)
    if not run_dirs:
        print("No schema outputs found.")
        return 1

    report_runs: List[Dict[str, Any]] = []
    total_errors = 0
    total_warnings = 0

    for rd in run_dirs:
        payload, spec = _load_spec_from_run(rd)
        issues = _audit_spec(spec)
        err = sum(1 for i in issues if i.severity == "error")
        warn = sum(1 for i in issues if i.severity == "warning")
        total_errors += err
        total_warnings += warn

        report_runs.append(
            {
                "run_dir": str(rd),
                "category": spec.get("category", ""),
                "schema_errors_field": payload.get("errors", []),
                "schema_warnings_field": payload.get("warnings", []),
                "audit_issues": [asdict(i) for i in issues],
                "audit_error_count": err,
                "audit_warning_count": warn,
            }
        )

    print(f"Audited runs: {len(report_runs)}")
    print(f"Audit issues: errors={total_errors}, warnings={total_warnings}")
    for row in report_runs:
        if row["audit_error_count"] == 0 and row["audit_warning_count"] == 0:
            continue
        print(f"- {row['category']} ({Path(row['run_dir']).name}): "
              f"errors={row['audit_error_count']}, warnings={row['audit_warning_count']}")
        for issue in row["audit_issues"]:
            print(f"    [{issue['severity'].upper()}] {issue['category']}: {issue['message']}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": {
                "run_count": len(report_runs),
                "error_count": total_errors,
                "warning_count": total_warnings,
            },
            "runs": report_runs,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON report: {out_path}")

    return 0 if total_errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
