#!/usr/bin/env python3
"""
Scenario Pipeline Dashboard

Builds an interactive multi-run dashboard for schema + downstream stage outputs.

Features:
- Ingest many run folders (including repeated category redoes and multiple seeds).
- Deterministic natural-language summaries of what each schema encodes.
- Layered issue detection across schema, deterministic explicitization, and stages 02..10.
- Per-stage health tables and per-run end-to-end timeline + unified map layer viewer.

Usage:
  python tools/schema_dashboard.py --glob "debug_runs/20260226_133*"
  python tools/schema_dashboard.py --glob "debug_runs/20260226_13*" "debug_runs/20260225_*" --output debug_runs/schema_dashboard.html
"""

from __future__ import annotations

import argparse
import base64
import copy
import glob
import hashlib
import html
import importlib.util
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_GENERATOR_ROOT = REPO_ROOT / "scenario_generator"

import sys

if str(SCENARIO_GENERATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(SCENARIO_GENERATOR_ROOT))

from scenario_generator.capabilities import CATEGORY_DEFINITIONS  # noqa: E402
from scenario_generator.constraints import (  # noqa: E402
    ConstraintType,
    ScenarioSpec,
    spec_from_dict,
    spec_to_dict,
    validate_spec,
)
from scenario_generator.schema_generator import (  # noqa: E402
    _canonicalize_actors,
    _canonicalize_constraints,
    _conflict_findings,
    _ensure_direction_and_lane_constraints,
    _ensure_interaction_coverage,
)


@dataclass
class DashIssue:
    severity: str
    rule: str
    message: str
    suggestion: str = ""


@dataclass
class SchemaRun:
    run_dir: str
    run_name: str
    timestamp: str
    category: str
    seed: Optional[int]
    redo_index: int
    seed_redo_index: int
    spec: Dict[str, Any]
    natural_summary: str
    compact_summary: str
    issues: List[DashIssue]
    score: int
    status: str
    ego_count: int
    constraint_count: int
    actor_count: int
    schema_errors: List[str]
    schema_warnings: List[str]
    deterministic_adjustments: List[str]
    stage_trace: List[Dict[str, Any]]
    map_layers: Dict[str, Any]
    validation_score: Optional[float]
    carla_validation: Dict[str, Any]
    can_accept: bool
    fingerprint: str
    route_actor_fingerprint: str
    route_actor_tokens: List[str]
    role_counts: Dict[str, int]
    interaction_count: int
    interest_score: float
    interest_score_adjusted: float
    similarity_best: float
    similarity_peer: Optional[str]
    similarity_cluster_size: int
    duplicate_of: Optional[str]


PIPELINE_STAGES: List[Tuple[str, str]] = [
    ("schema", "01_schema"),
    ("geometry", "02_geometry"),
    ("crop", "03_crop"),
    ("legal_paths", "04_legal_paths"),
    ("pick_paths", "05_pick_paths"),
    ("refine_paths", "06_refine"),
    ("placement", "07_placement"),
    ("validation", "08_validation"),
    ("routes", "09_routes"),
    ("carla_validation", "10_carla_validation"),
]
PIPELINE_STAGE_INDEX = {name: idx for idx, (name, _dir) in enumerate(PIPELINE_STAGES)}
PIPELINE_STAGE_LABELS = {
    "schema": "Schema",
    "geometry": "Geometry",
    "crop": "Crop",
    "legal_paths": "Legal Paths",
    "pick_paths": "Pick Paths",
    "refine_paths": "Refine",
    "placement": "Placement",
    "validation": "Validation",
    "routes": "Routes",
    "carla_validation": "CARLA Validation",
}


def _load_category_audit_hook():
    """Load tools/audit_schema_outputs.py dynamically so we can reuse its checks."""
    path = REPO_ROOT / "tools" / "audit_schema_outputs.py"
    spec = importlib.util.spec_from_file_location("_schema_audit_module", str(path))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "_audit_spec"):
        return None
    return module._audit_spec


def _safe_int_vehicle_id(vehicle_id: str) -> Tuple[int, str]:
    m = re.search(r"Vehicle\s+(\d+)", vehicle_id or "")
    if not m:
        return (10**9, vehicle_id or "")
    return (int(m.group(1)), vehicle_id or "")


def _discover_run_dirs(patterns: Iterable[str]) -> List[Path]:
    run_dirs: List[Path] = []
    for pattern in patterns:
        for match in sorted(glob.glob(pattern)):
            p = Path(match)
            if p.is_file() and p.name == "output.json" and p.parent.name == "01_schema":
                run_dirs.append(p.parents[1])
                continue
            if p.is_dir() and (p / "01_schema" / "output.json").exists():
                run_dirs.append(p)
    unique: List[Path] = []
    seen = set()
    for rd in run_dirs:
        key = str(rd.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(rd)
    return unique


def discover_run_dirs(patterns: Iterable[str]) -> List[Path]:
    """Public wrapper for run discovery (used by direct integrations)."""
    return _discover_run_dirs(patterns)


def _parse_timestamp_from_run_name(run_name: str) -> str:
    m = re.match(r"^(\d{8}_\d{6})_", run_name)
    return m.group(1) if m else "unknown"


def _parse_seed_from_run_dir(run_dir: Path) -> Optional[int]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    seed_val = payload.get("seed")
    if isinstance(seed_val, int):
        return seed_val
    if isinstance(seed_val, str):
        seed_val = seed_val.strip()
        if seed_val and re.fullmatch(r"-?\d+", seed_val):
            return int(seed_val)
    return None


def _load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _to_float(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None
    return None


def _coord_from_obj(obj: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    if not isinstance(obj, dict):
        return None
    x = _to_float(obj.get("x"))
    y = _to_float(obj.get("y"))
    if x is None or y is None:
        x = _to_float(obj.get("world_x"))
        y = _to_float(obj.get("world_y"))
    if x is None or y is None:
        spawn = obj.get("spawn")
        if isinstance(spawn, dict):
            x = _to_float(spawn.get("x"))
            y = _to_float(spawn.get("y"))
    if x is None or y is None:
        point = obj.get("point")
        if isinstance(point, dict):
            x = _to_float(point.get("x"))
            y = _to_float(point.get("y"))
    if x is None or y is None:
        return None
    return (x, y)


def _path_segments_from_signature(sig: Any) -> List[Dict[str, Any]]:
    if not isinstance(sig, dict):
        return []
    segs = sig.get("segments_detailed")
    if not isinstance(segs, list):
        return []
    out: List[Dict[str, Any]] = []
    for s in segs:
        if isinstance(s, dict):
            out.append(s)
    return out


def _segment_heading_deg(seg: Dict[str, Any]) -> Optional[float]:
    if not isinstance(seg, dict):
        return None
    start = seg.get("start")
    end = seg.get("end")
    p0 = _coord_from_obj(start) if isinstance(start, dict) else None
    p1 = _coord_from_obj(end) if isinstance(end, dict) else None
    if p0 is None and isinstance(start, dict):
        p0 = _coord_from_obj(start.get("point", {}))
    if p1 is None and isinstance(end, dict):
        p1 = _coord_from_obj(end.get("point", {}))
    if p0 is None or p1 is None:
        return None
    dx = float(p1[0]) - float(p0[0])
    dy = float(p1[1]) - float(p0[1])
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None
    return math.degrees(math.atan2(dy, dx))


def _norm_angle_delta_deg(a: float, b: float) -> float:
    return abs(((b - a + 180.0) % 360.0) - 180.0)


def _max_segment_endpoint_gap_m(sig: Any) -> Optional[float]:
    segs = _path_segments_from_signature(sig)
    if len(segs) < 2:
        return None
    max_gap: Optional[float] = None
    for i in range(1, len(segs)):
        prev_end = _coord_from_obj(segs[i - 1].get("end", {}))
        curr_start = _coord_from_obj(segs[i].get("start", {}))
        if prev_end is None and isinstance(segs[i - 1].get("end"), dict):
            prev_end = _coord_from_obj(segs[i - 1]["end"].get("point", {}))
        if curr_start is None and isinstance(segs[i].get("start"), dict):
            curr_start = _coord_from_obj(segs[i]["start"].get("point", {}))
        if prev_end is None or curr_start is None:
            continue
        gap = math.hypot(float(curr_start[0]) - float(prev_end[0]), float(curr_start[1]) - float(prev_end[1]))
        if max_gap is None or gap > max_gap:
            max_gap = gap
    return max_gap


def _turn_class_from_signature_geometry(sig: Any) -> Optional[str]:
    segs = _path_segments_from_signature(sig)
    if not segs:
        return None
    h0 = _segment_heading_deg(segs[0])
    h1 = _segment_heading_deg(segs[-1])
    if h0 is None or h1 is None:
        return None
    d = _norm_angle_delta_deg(h0, h1)
    if d <= 30.0:
        return "straight"
    if d >= 150.0:
        return "u_turn"
    return "turn"


def _polyline_from_segments_detailed(segments: Any) -> List[List[float]]:
    if not isinstance(segments, list):
        return []
    points: List[List[float]] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        sample = seg.get("polyline_sample")
        if not isinstance(sample, list):
            continue
        for pt in sample:
            p = _coord_from_obj(pt) if isinstance(pt, dict) else None
            if p is None and isinstance(pt, (list, tuple)) and len(pt) >= 2:
                x = _to_float(pt[0])
                y = _to_float(pt[1])
                if x is not None and y is not None:
                    p = (x, y)
            if p is None:
                continue
            q = [round(float(p[0]), 3), round(float(p[1]), 3)]
            if not points or points[-1] != q:
                points.append(q)
    return points


def _polyline_from_signature(sig: Any) -> List[List[float]]:
    if not isinstance(sig, dict):
        return []
    return _polyline_from_segments_detailed(sig.get("segments_detailed"))


def _polyline_from_waypoints(waypoints: Any) -> List[List[float]]:
    if not isinstance(waypoints, list):
        return []
    out: List[List[float]] = []
    for wp in waypoints:
        p = _coord_from_obj(wp) if isinstance(wp, dict) else None
        if p is None and isinstance(wp, (list, tuple)) and len(wp) >= 2:
            x = _to_float(wp[0])
            y = _to_float(wp[1])
            if x is not None and y is not None:
                p = (x, y)
        if p is None:
            continue
        q = [round(float(p[0]), 3), round(float(p[1]), 3)]
        if not out or out[-1] != q:
            out.append(q)
    return out


def _path_point_min_distance(pt: Tuple[float, float], polyline: List[List[float]]) -> Optional[float]:
    if not polyline:
        return None
    x, y = pt
    best = None
    for p in polyline:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        px = _to_float(p[0])
        py = _to_float(p[1])
        if px is None or py is None:
            continue
        d = math.hypot(x - float(px), y - float(py))
        if best is None or d < best:
            best = d
    return best


def _polyline_pair_min_distance(poly1: List[List[float]], poly2: List[List[float]]) -> Optional[float]:
    if not poly1 or not poly2:
        return None
    best = None
    for p1 in poly1:
        if not isinstance(p1, (list, tuple)) or len(p1) < 2:
            continue
        x1 = _to_float(p1[0])
        y1 = _to_float(p1[1])
        if x1 is None or y1 is None:
            continue
        for p2 in poly2:
            if not isinstance(p2, (list, tuple)) or len(p2) < 2:
                continue
            x2 = _to_float(p2[0])
            y2 = _to_float(p2[1])
            if x2 is None or y2 is None:
                continue
            d = math.hypot(float(x1) - float(x2), float(y1) - float(y2))
            if best is None or d < best:
                best = d
    return best


def _path_heading_deg(poly: List[List[float]]) -> Optional[float]:
    if len(poly) < 2:
        return None
    p0 = poly[0]
    p1 = poly[min(3, len(poly) - 1)]
    if not isinstance(p0, (list, tuple)) or not isinstance(p1, (list, tuple)):
        return None
    x0 = _to_float(p0[0])
    y0 = _to_float(p0[1])
    x1 = _to_float(p1[0])
    y1 = _to_float(p1[1])
    if None in {x0, y0, x1, y1}:
        return None
    dx = float(x1) - float(x0)
    dy = float(y1) - float(y0)
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None
    return math.degrees(math.atan2(dy, dx))


def _extract_candidate_entries(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries = data.get("candidates")
    if isinstance(entries, list):
        return [e for e in entries if isinstance(e, dict)]
    entries = data.get("paths_named")
    if isinstance(entries, list):
        return [e for e in entries if isinstance(e, dict)]
    return []


def _extract_picked_entries(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries = data.get("picked")
    if isinstance(entries, list):
        return [e for e in entries if isinstance(e, dict)]
    entries = data.get("ego_picked")
    if isinstance(entries, list):
        return [e for e in entries if isinstance(e, dict)]
    return []


def _normalize_crop_region(crop_val: Any) -> Optional[Dict[str, float]]:
    if isinstance(crop_val, list) and len(crop_val) == 4:
        vals = [_to_float(x) for x in crop_val]
        if any(v is None for v in vals):
            return None
        xmin, xmax, ymin, ymax = vals
        return {"xmin": float(xmin), "xmax": float(xmax), "ymin": float(ymin), "ymax": float(ymax)}
    if isinstance(crop_val, dict):
        # Support both legacy and normalized field names.
        xmin = _to_float(crop_val.get("xmin", crop_val.get("x_min")))
        xmax = _to_float(crop_val.get("xmax", crop_val.get("x_max")))
        ymin = _to_float(crop_val.get("ymin", crop_val.get("y_min")))
        ymax = _to_float(crop_val.get("ymax", crop_val.get("y_max")))
        if None in {xmin, xmax, ymin, ymax}:
            return None
        return {"xmin": float(xmin), "xmax": float(xmax), "ymin": float(ymin), "ymax": float(ymax)}
    return None


def _crop_contains_point(crop: Optional[Dict[str, float]], x: float, y: float, margin: float = 2.0) -> bool:
    if not crop:
        return True
    return (
        crop["xmin"] - margin <= x <= crop["xmax"] + margin
        and crop["ymin"] - margin <= y <= crop["ymax"] + margin
    )


def _summary_stage_record_map(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    stages = summary.get("stages", [])
    if not isinstance(stages, list):
        return out
    for s in stages:
        if not isinstance(s, dict):
            continue
        stage = str(s.get("stage", "")).strip()
        if not stage:
            continue
        out[stage] = s
    return out


def _requested_stages_from_summary(summary: Dict[str, Any]) -> List[str]:
    stop_stage = summary.get("stopped_after") or summary.get("stopped_after_stage") or summary.get("stop_after_stage")
    if isinstance(stop_stage, str) and stop_stage in PIPELINE_STAGE_INDEX:
        stop_idx = PIPELINE_STAGE_INDEX[stop_stage]
        return [s for s, _ in PIPELINE_STAGES[: stop_idx + 1]]
    failed_stage = summary.get("failed_stage")
    if isinstance(failed_stage, str) and failed_stage in PIPELINE_STAGE_INDEX:
        fail_idx = PIPELINE_STAGE_INDEX[failed_stage]
        return [s for s, _ in PIPELINE_STAGES[: fail_idx + 1]]
    # If no stop stage is recorded, treat all stages as potentially requested.
    return [s for s, _ in PIPELINE_STAGES]


def _extract_pick_confidence(entry: Dict[str, Any]) -> Optional[float]:
    c = _to_float(entry.get("confidence"))
    if c is None:
        return None
    return max(0.0, min(1.0, c))


def _infer_stage_trace_and_pipeline_findings(
    run_dir: Path,
    spec: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[DashIssue], Optional[float]]:
    """
    Analyze stage artifacts (02..10) and return:
      - stage_trace for timeline/status UI,
      - map_layers for single-run map overlays,
      - pipeline findings (issues/warnings/anomalies),
      - validation score if available.
    """
    summary = _load_json_if_exists(run_dir / "summary.json") or {}
    stage_record_map = _summary_stage_record_map(summary)
    requested_stages = set(_requested_stages_from_summary(summary))
    ego_count = len([v for v in spec.get("ego_vehicles", []) if isinstance(v, dict)])
    actor_expectation = len([a for a in spec.get("actors", []) if isinstance(a, dict)])
    topology = str(spec.get("topology", "")).strip()

    issues: List[DashIssue] = []
    stage_trace: List[Dict[str, Any]] = []

    # Shared cross-stage metrics
    metrics: Dict[str, Any] = {
        "legal_num_paths": None,
        "picked_ego_count": None,
        "refined_ego_count": None,
        "placement_ego_count": None,
        "placement_actor_count": None,
        "validation_actual_vehicles": None,
        "validation_expected_vehicles": None,
        "validation_score": None,
        "route_file_count": None,
        "legal_candidate_names": set(),
        "picked_vehicle_names": [],
        "refined_vehicle_names": [],
    }

    map_layers: Dict[str, Any] = {
        "crop": None,
        "legal_paths": [],
        "picked_paths": [],
        "refined_paths": [],
        "actors": [],
        "ego_spawns": [],
    }

    for stage_name, stage_dir_name in PIPELINE_STAGES:
        stage_dir = run_dir / stage_dir_name
        output_path = stage_dir / "output.json"
        output = _load_json_if_exists(output_path) if output_path.exists() else None
        record = stage_record_map.get(stage_name)
        requested = stage_name in requested_stages
        present = stage_dir.exists()
        success = None if record is None else bool(record.get("success"))
        elapsed = None if record is None else _to_float(record.get("elapsed_s"))
        stage_metrics: Dict[str, Any] = {}
        status = "not_requested"
        if requested:
            if success is True:
                status = "ok"
            elif success is False:
                status = "failed"
            elif present:
                status = "present_untracked"
            else:
                status = "missing"
        if requested and success is True and output is None:
            issues.append(
                DashIssue(
                    "error",
                    f"stage_output_missing:{stage_name}",
                    f"Stage '{stage_name}' marked successful but output.json is missing or invalid.",
                )
            )
        if requested and success is False:
            err = ""
            if isinstance(record, dict):
                err = str(record.get("error", "")).strip()
            msg = f"Stage '{stage_name}' failed."
            if err:
                msg += f" Error: {err}"
            issues.append(DashIssue("error", f"stage_failed:{stage_name}", msg))
        if requested and not present and success is None:
            issues.append(
                DashIssue(
                    "warning",
                    f"stage_missing:{stage_name}",
                    f"Expected stage directory missing: {stage_dir_name}",
                )
            )

        # Stage-specific anomaly detection and map layer extraction.
        if stage_name == "geometry" and output:
            g_topology = str(output.get("topology", "")).strip()
            min_lane_count = _to_float(output.get("min_lane_count"))
            stage_metrics.update(
                {
                    "topology": g_topology,
                    "degree": output.get("degree"),
                    "min_lane_count": min_lane_count,
                    "needs_oncoming": bool(output.get("needs_oncoming")),
                    "needs_on_ramp": bool(output.get("needs_on_ramp")),
                    "needs_multi_lane": bool(output.get("needs_multi_lane")),
                    "needs_merge_onto_same_road": bool(output.get("needs_merge_onto_same_road")),
                }
            )
            if g_topology and topology and g_topology != topology:
                issues.append(
                    DashIssue(
                        "warning",
                        "geometry_topology_mismatch",
                        f"Geometry topology '{g_topology}' does not match schema topology '{topology}'.",
                    )
                )
            if min_lane_count is not None and min_lane_count < 1:
                issues.append(
                    DashIssue(
                        "error",
                        "geometry_lane_count_invalid",
                        f"Geometry min_lane_count={min_lane_count} is invalid.",
                    )
                )
            if bool(output.get("needs_on_ramp")) and not bool(output.get("needs_merge_onto_same_road")):
                issues.append(
                    DashIssue(
                        "warning",
                        "geometry_onramp_merge_incomplete",
                        "Geometry needs_on_ramp=true but needs_merge_onto_same_road=false.",
                    )
                )

        if stage_name == "crop" and output:
            crop_error = str(output.get("error", "") or "").strip()
            crop = _normalize_crop_region(output.get("crop"))
            map_layers["crop"] = crop
            stage_metrics.update(
                {
                    "method": output.get("method"),
                    "total_candidates": output.get("total_candidates"),
                    "crop": crop,
                    "error": crop_error or None,
                }
            )
            if crop_error:
                if "Image size of" in crop_error and "too large" in crop_error:
                    issues.append(
                        DashIssue(
                            "error",
                            "crop_viz_render_failure",
                            "Crop stage failed while rendering visualization (pixel bounds overflow).",
                        )
                    )
                elif "No crops satisfy geometry spec" in crop_error:
                    issues.append(
                        DashIssue(
                            "error",
                            "crop_no_satisfying_candidates",
                            "No candidate crop satisfied GeometrySpec constraints.",
                        )
                    )
                else:
                    issues.append(DashIssue("error", "crop_stage_error", crop_error))
            if crop is None:
                if not crop_error:
                    issues.append(DashIssue("error", "crop_invalid", "Crop output is missing/invalid bounds."))
            else:
                if crop["xmin"] >= crop["xmax"] or crop["ymin"] >= crop["ymax"]:
                    issues.append(DashIssue("error", "crop_bounds_invalid", f"Invalid crop bounds: {crop}"))
                area = (crop["xmax"] - crop["xmin"]) * (crop["ymax"] - crop["ymin"])
                stage_metrics["area"] = round(area, 2)
                if area < 100.0:
                    issues.append(
                        DashIssue("warning", "crop_area_small", f"Crop area {area:.2f} m^2 is unusually small.")
                    )
            crop_method = str(output.get("method", "")).lower()
            if crop_method == "fallback":
                issues.append(
                    DashIssue(
                        "warning",
                        "crop_fallback_used",
                        "Crop selection used fallback instead of CSP assignment.",
                    )
                )
            total_candidates = _to_float(output.get("total_candidates"))
            if total_candidates is not None and total_candidates <= 0:
                # Hardcoded/manual crops intentionally bypass candidate enumeration.
                if crop_method not in {"hardcoded", "manual", "fixed"}:
                    issues.append(DashIssue("error", "crop_no_candidates", "Crop stage reports zero candidates."))

        if stage_name == "legal_paths":
            num_paths = None
            if output:
                num_paths = int(output.get("num_paths", 0) or 0)
                metrics["legal_num_paths"] = num_paths
                stage_metrics.update(
                    {
                        "num_paths": num_paths,
                        "num_segments_cropped": output.get("num_segments_cropped"),
                    }
                )
                if num_paths <= 0:
                    issues.append(DashIssue("error", "legal_paths_empty", "No legal paths enumerated."))
                if ego_count > 0 and num_paths < ego_count:
                    issues.append(
                        DashIssue(
                            "warning",
                            "legal_paths_low_count",
                            f"Legal paths ({num_paths}) fewer than ego vehicles ({ego_count}).",
                        )
                    )
                nseg = _to_float(output.get("num_segments_cropped"))
                if nseg is not None and nseg <= 0:
                    issues.append(DashIssue("error", "legal_paths_no_segments", "No cropped segments for legal path generation."))
            detailed = _load_json_if_exists(stage_dir / "legal_paths_detailed.json")
            if detailed:
                candidates = _extract_candidate_entries(detailed)
                if output and num_paths is not None and len(candidates) != num_paths:
                    issues.append(
                        DashIssue(
                            "warning",
                            "legal_paths_count_mismatch",
                            f"legal_paths_detailed has {len(candidates)} candidates but output reports {num_paths}.",
                        )
                    )
                names = []
                continuity_alerts: List[Tuple[float, str]] = []
                for c in candidates:
                    name = str(c.get("name", "")).strip()
                    if name:
                        names.append(name)
                    sig = c.get("signature", {})
                    max_gap = _max_segment_endpoint_gap_m(sig)
                    if max_gap is not None and max_gap >= 2.0:
                        continuity_alerts.append((float(max_gap), name or "unnamed_path"))
                    poly = _polyline_from_signature(sig)
                    if poly:
                        map_layers["legal_paths"].append(
                            {
                                "name": name,
                                "polyline": poly,
                                "length_m": _to_float(sig.get("length_m")),
                                "turn": sig.get("entry_to_exit_turn"),
                            }
                        )
                metrics["legal_candidate_names"] = set(names)
                metrics["legal_paths_with_segment_gaps"] = len(continuity_alerts)
                if continuity_alerts:
                    worst_gap, worst_name = max(continuity_alerts, key=lambda x: x[0])
                    issues.append(
                        DashIssue(
                            "warning",
                            "legal_paths_segment_gap",
                            (
                                f"{len(continuity_alerts)} legal paths have segment endpoint gaps >= 2.0m "
                                f"(max {worst_gap:.2f}m at '{worst_name}')."
                            ),
                        )
                    )
                if len(names) != len(set(names)):
                    issues.append(DashIssue("warning", "legal_paths_duplicate_names", "Duplicate legal path names found."))
                if candidates and not map_layers["legal_paths"]:
                    issues.append(
                        DashIssue(
                            "warning",
                            "legal_paths_missing_polylines",
                            "Legal path candidates exist but no polyline samples were found.",
                        )
                    )

        if stage_name == "pick_paths":
            detailed = _load_json_if_exists(stage_dir / "picked_paths_detailed.json")
            picked = _extract_picked_entries(detailed) if detailed else []
            if output:
                pc = int(output.get("ego_picked_count", len(picked)) or 0)
                metrics["picked_ego_count"] = pc
                stage_metrics["ego_picked_count"] = pc
                if pc <= 0:
                    issues.append(DashIssue("error", "picked_paths_empty", "Path picker selected zero ego paths."))
                if ego_count > 0 and pc < ego_count:
                    issues.append(
                        DashIssue(
                            "error",
                            "picked_paths_missing_egos",
                            f"Path picker produced {pc} picks for {ego_count} ego vehicles.",
                        )
                    )
            if detailed:
                vehicles = []
                confidences: List[float] = []
                path_names: List[str] = []
                continuity_alerts: List[Tuple[float, str, str]] = []
                for p in picked:
                    vehicle = str(p.get("vehicle", "")).strip()
                    name = str(p.get("name", p.get("path_name", ""))).strip()
                    if vehicle:
                        vehicles.append(vehicle)
                    if name:
                        path_names.append(name)
                    conf = _extract_pick_confidence(p)
                    if conf is not None:
                        confidences.append(conf)
                    sig = p.get("signature", {})
                    max_gap = _max_segment_endpoint_gap_m(sig)
                    if max_gap is not None and max_gap >= 2.0:
                        continuity_alerts.append((float(max_gap), vehicle or "?", name or "unnamed_path"))
                    poly = _polyline_from_signature(sig)
                    if poly:
                        turn = sig.get("entry_to_exit_turn")
                        if not turn:
                            turn = _turn_class_from_signature_geometry(sig)
                        map_layers["picked_paths"].append(
                            {
                                "vehicle": vehicle,
                                "name": name,
                                "confidence": conf,
                                "polyline": poly,
                                "length_m": _to_float(sig.get("length_m")),
                                "turn": turn,
                                "start_heading_deg": _path_heading_deg(poly),
                            }
                        )
                    if name and metrics["legal_candidate_names"] and name not in metrics["legal_candidate_names"]:
                        issues.append(
                            DashIssue(
                                "warning",
                                "picked_path_name_unknown",
                                f"Picked path '{name}' not found in legal-path candidate names.",
                            )
                        )
                metrics["picked_vehicle_names"] = vehicles
                metrics["picked_unique_path_count"] = len(set(path_names))
                if len(vehicles) != len(set(vehicles)):
                    issues.append(DashIssue("error", "picked_duplicate_vehicle", "Path picker contains duplicate vehicle assignments."))
                if continuity_alerts:
                    worst_gap, worst_vehicle, worst_name = max(continuity_alerts, key=lambda x: x[0])
                    issues.append(
                        DashIssue(
                            "warning",
                            "picked_paths_segment_gap",
                            (
                                f"{len(continuity_alerts)} picked paths have segment endpoint gaps >= 2.0m "
                                f"(max {worst_gap:.2f}m at {worst_vehicle}:{worst_name})."
                            ),
                        )
                    )
                if confidences:
                    avg_conf = sum(confidences) / len(confidences)
                    stage_metrics["avg_confidence"] = round(avg_conf, 3)
                    if avg_conf < 0.5:
                        issues.append(
                            DashIssue(
                                "warning",
                                "picked_low_confidence",
                                f"Low average pick confidence: {avg_conf:.2f}.",
                            )
                        )

        if stage_name == "refine_paths":
            detailed = _load_json_if_exists(stage_dir / "picked_paths_refined.json")
            refined = _extract_picked_entries(detailed) if detailed else []
            if output:
                rc = int(output.get("ego_count", len(refined)) or 0)
                metrics["refined_ego_count"] = rc
                stage_metrics["ego_count"] = rc
                stage_metrics["roundabout_skip"] = bool(output.get("roundabout_skip"))
                if bool(output.get("roundabout_skip")) and topology != "roundabout":
                    issues.append(
                        DashIssue(
                            "warning",
                            "refine_roundabout_skip_mismatch",
                            "Refinement skipped as roundabout, but schema topology is not roundabout.",
                        )
                    )
            if detailed:
                vehicles = []
                continuity_alerts: List[Tuple[float, str, str]] = []
                for r in refined:
                    vehicle = str(r.get("vehicle", "")).strip()
                    name = str(r.get("name", r.get("path_name", ""))).strip()
                    if vehicle:
                        vehicles.append(vehicle)
                    sig = r.get("signature", {})
                    max_gap = _max_segment_endpoint_gap_m(sig)
                    if max_gap is not None and max_gap >= 2.0:
                        continuity_alerts.append((float(max_gap), vehicle or "?", name or "unnamed_path"))
                    poly = _polyline_from_signature(sig)
                    if poly:
                        turn = sig.get("entry_to_exit_turn")
                        if not turn:
                            turn = _turn_class_from_signature_geometry(sig)
                        map_layers["refined_paths"].append(
                            {
                                "vehicle": vehicle,
                                "name": name,
                                "polyline": poly,
                                "length_m": _to_float(sig.get("length_m")),
                                "turn": turn,
                                "start_heading_deg": _path_heading_deg(poly),
                            }
                        )
                    sig0 = r.get("signature_original", {})
                    if isinstance(sig, dict) and isinstance(sig0, dict):
                        l1 = _to_float(sig.get("length_m"))
                        l0 = _to_float(sig0.get("length_m"))
                        t1 = _turn_class_from_signature_geometry(sig)
                        t0 = _turn_class_from_signature_geometry(sig0)
                        if l0 is not None and l1 is not None and l0 > 1e-6:
                            ratio = abs(l1 - l0) / l0
                            if ratio > 0.6:
                                issues.append(
                                    DashIssue(
                                        "warning",
                                        "refine_length_delta_large",
                                        f"Vehicle {vehicle or '?'} path length changed by {ratio * 100:.1f}% during refinement.",
                                    )
                                )
                            elif ratio > 0.3:
                                issues.append(
                                    DashIssue(
                                        "warning",
                                        "refine_length_delta_moderate",
                                        f"Vehicle {vehicle or '?'} path length changed by {ratio * 100:.1f}% during refinement.",
                                    )
                                )
                        if t0 is not None and t1 is not None and t0 != t1:
                            issues.append(
                                DashIssue(
                                    "warning",
                                    "refine_turn_class_changed",
                                    f"Vehicle {vehicle or '?'} turn class changed from {t0} to {t1} during refinement.",
                                )
                            )
                metrics["refined_vehicle_names"] = vehicles
                if continuity_alerts:
                    worst_gap, worst_vehicle, worst_name = max(continuity_alerts, key=lambda x: x[0])
                    issues.append(
                        DashIssue(
                            "warning",
                            "refined_paths_segment_gap",
                            (
                                f"{len(continuity_alerts)} refined paths have segment endpoint gaps >= 2.0m "
                                f"(max {worst_gap:.2f}m at {worst_vehicle}:{worst_name})."
                            ),
                        )
                    )

        if stage_name == "placement":
            scene = _load_json_if_exists(stage_dir / "scene_objects.json")
            if output:
                metrics["placement_ego_count"] = int(output.get("ego_count", 0) or 0)
                metrics["placement_actor_count"] = int(output.get("npc_count", 0) or 0)
                stage_metrics["ego_count"] = metrics["placement_ego_count"]
                stage_metrics["npc_count"] = metrics["placement_actor_count"]
            if scene:
                ego_list = scene.get("ego_picked")
                if isinstance(ego_list, list):
                    metrics["placement_ego_count"] = len(ego_list)
                actors = scene.get("npc_objects")
                if not isinstance(actors, list):
                    actors = scene.get("actors")
                if not isinstance(actors, list):
                    actors = []
                metrics["placement_actor_count"] = len(actors)
                if actor_expectation > 0 and len(actors) == 0 and str(spec.get("category", "")) != "Unprotected Left Turn":
                    issues.append(
                        DashIssue(
                            "warning",
                            "placement_missing_actors",
                            f"Schema expects {actor_expectation} actor groups but placement produced zero actors.",
                        )
                    )
                crop = map_layers.get("crop")
                seen_points: List[Tuple[float, float, str]] = []
                ego_path_refs: List[List[List[float]]] = []
                ego_path_by_vehicle: Dict[str, List[List[float]]] = {}
                if map_layers["refined_paths"]:
                    for r in map_layers["refined_paths"]:
                        if not isinstance(r, dict):
                            continue
                        poly = r.get("polyline", [])
                        if isinstance(poly, list):
                            ego_path_refs.append(poly)
                            vehicle = str(r.get("vehicle", "")).strip()
                            if vehicle:
                                ego_path_by_vehicle[vehicle] = poly
                elif map_layers["picked_paths"]:
                    for r in map_layers["picked_paths"]:
                        if not isinstance(r, dict):
                            continue
                        poly = r.get("polyline", [])
                        if isinstance(poly, list):
                            ego_path_refs.append(poly)
                            vehicle = str(r.get("vehicle", "")).strip()
                            if vehicle:
                                ego_path_by_vehicle[vehicle] = poly
                for a in actors:
                    if not isinstance(a, dict):
                        continue
                    trajectory = _polyline_from_waypoints(a.get("world_waypoints"))
                    if not trajectory:
                        trajectory = _polyline_from_waypoints(a.get("trajectory"))
                    p = _coord_from_obj(a)
                    if p is None and trajectory:
                        p = (float(trajectory[0][0]), float(trajectory[0][1]))
                    aid = str(a.get("entity_id", a.get("actor_id", a.get("id", "actor"))))
                    kind = str(a.get("category", a.get("kind", a.get("type", a.get("actor_kind", "unknown")))))
                    semantic = str(a.get("semantic", ""))
                    motion = a.get("motion") if isinstance(a.get("motion"), dict) else {}
                    motion_type = str(motion.get("type", "unknown"))
                    trigger = a.get("trigger") if isinstance(a.get("trigger"), dict) else {}
                    if p is None:
                        issues.append(
                            DashIssue(
                                "warning",
                                "placement_actor_no_coords",
                                f"Actor '{aid}' has no coordinates.",
                            )
                        )
                        continue
                    x, y = p
                    yaw = None
                    spawn_obj = a.get("spawn")
                    if isinstance(spawn_obj, dict):
                        yaw = _to_float(spawn_obj.get("yaw_deg"))
                    if yaw is None and trajectory and len(trajectory) >= 2:
                        yaw = _path_heading_deg(trajectory)
                    map_layers["actors"].append(
                        {
                            "id": aid,
                            "kind": kind,
                            "semantic": semantic,
                            "motion_type": motion_type,
                            "x": round(x, 3),
                            "y": round(y, 3),
                            "yaw_deg": None if yaw is None else round(float(yaw), 2),
                            "trajectory": trajectory,
                            "target_vehicle": str((a.get("placement") or {}).get("target_vehicle", "")),
                            "trigger_type": str(trigger.get("type", "")),
                            "trigger_vehicle": str(trigger.get("vehicle", "")),
                            "trigger_distance_m": _to_float(trigger.get("distance_m")),
                        }
                    )
                    if not _crop_contains_point(crop, x, y, margin=5.0):
                        issues.append(
                            DashIssue(
                                "warning",
                                "placement_actor_outside_crop",
                                f"Actor '{aid}' is outside crop bounds.",
                            )
                        )
                    for ox, oy, oid in seen_points:
                        if math.hypot(x - ox, y - oy) < 0.3:
                            issues.append(
                                DashIssue(
                                    "warning",
                                    "placement_actor_overlap",
                                    f"Actors '{aid}' and '{oid}' appear overlapping (<0.3m).",
                                )
                            )
                    seen_points.append((x, y, aid))
                    if kind.lower() == "walker" and motion_type.lower() == "cross_perpendicular" and ego_path_refs:
                        dmin_any = None
                        for poly in ego_path_refs:
                            d = _path_point_min_distance((float(x), float(y)), poly if isinstance(poly, list) else [])
                            if d is None:
                                continue
                            if dmin_any is None or d < dmin_any:
                                dmin_any = d
                        target_vehicle = str(trigger.get("vehicle", "")).strip() or str((a.get("placement") or {}).get("target_vehicle", "")).strip()
                        dmin_target = None
                        if target_vehicle and target_vehicle in ego_path_by_vehicle:
                            dmin_target = _path_point_min_distance((float(x), float(y)), ego_path_by_vehicle[target_vehicle])
                        if dmin_any is not None and dmin_any < 0.8:
                            issues.append(
                                DashIssue(
                                    "warning",
                                    "pedestrian_spawn_too_close",
                                    f"Walker '{aid}' starts only {dmin_any:.1f}m from an ego path centerline.",
                                )
                            )
                        if dmin_target is not None and dmin_target < 2.2:
                            issues.append(
                                DashIssue(
                                    "warning",
                                    "pedestrian_spawn_too_close_target",
                                    f"Walker '{aid}' starts only {dmin_target:.1f}m from target vehicle path ({target_vehicle}).",
                                )
                            )
                ego_spawns = scene.get("ego_spawns")
                seen_ego = set()
                if isinstance(ego_spawns, list):
                    for e in ego_spawns:
                        if not isinstance(e, dict):
                            continue
                        p = _coord_from_obj(e)
                        if p is None:
                            continue
                        vehicle = str(e.get("vehicle", e.get("vehicle_id", "ego")))
                        if vehicle in seen_ego:
                            continue
                        yaw = _to_float(e.get("yaw_deg"))
                        if yaw is None:
                            spawn_obj = e.get("spawn")
                            if isinstance(spawn_obj, dict):
                                yaw = _to_float(spawn_obj.get("yaw_deg"))
                        map_layers["ego_spawns"].append(
                            {
                                "vehicle": vehicle,
                                "x": round(float(p[0]), 3),
                                "y": round(float(p[1]), 3),
                                "yaw_deg": None if yaw is None else round(float(yaw), 2),
                                "source": "scene_ego_spawns",
                            }
                        )
                        seen_ego.add(vehicle)
                if isinstance(ego_list, list):
                    for ego in ego_list:
                        if not isinstance(ego, dict):
                            continue
                        vehicle = str(ego.get("vehicle", "")).strip() or "ego"
                        if vehicle in seen_ego:
                            continue
                        sig = ego.get("signature", {}) if isinstance(ego.get("signature"), dict) else {}
                        entry = sig.get("entry", {}) if isinstance(sig.get("entry"), dict) else {}
                        point = entry.get("point", {}) if isinstance(entry.get("point"), dict) else {}
                        x = _to_float(point.get("x"))
                        y = _to_float(point.get("y"))
                        yaw = _to_float(entry.get("heading_deg"))
                        if x is None or y is None:
                            poly = _polyline_from_signature(sig)
                            if poly:
                                x = _to_float(poly[0][0])
                                y = _to_float(poly[0][1])
                                if yaw is None:
                                    yaw = _path_heading_deg(poly)
                        if x is None or y is None:
                            continue
                        map_layers["ego_spawns"].append(
                            {
                                "vehicle": vehicle,
                                "x": round(float(x), 3),
                                "y": round(float(y), 3),
                                "yaw_deg": None if yaw is None else round(float(yaw), 2),
                                "source": "scene_ego_picked_entry",
                            }
                        )
                        seen_ego.add(vehicle)
                if not map_layers["ego_spawns"]:
                    fallback_paths = map_layers["refined_paths"] if map_layers["refined_paths"] else map_layers["picked_paths"]
                    for pinfo in fallback_paths:
                        if not isinstance(pinfo, dict):
                            continue
                        poly = pinfo.get("polyline")
                        if not isinstance(poly, list) or not poly:
                            continue
                        vehicle = str(pinfo.get("vehicle", "ego"))
                        if vehicle in seen_ego:
                            continue
                        x = _to_float(poly[0][0]) if len(poly[0]) >= 2 else None
                        y = _to_float(poly[0][1]) if len(poly[0]) >= 2 else None
                        if x is None or y is None:
                            continue
                        map_layers["ego_spawns"].append(
                            {
                                "vehicle": vehicle,
                                "x": round(float(x), 3),
                                "y": round(float(y), 3),
                                "yaw_deg": _path_heading_deg(poly),
                                "source": "picked_or_refined_path_start",
                            }
                        )
                        seen_ego.add(vehicle)

        if stage_name == "validation" and output:
            score = _to_float(output.get("score"))
            metrics["validation_score"] = score
            metrics["validation_actual_vehicles"] = output.get("actual_vehicles")
            metrics["validation_expected_vehicles"] = output.get("expected_vehicles")
            stage_metrics.update(
                {
                    "is_valid": bool(output.get("is_valid")),
                    "score": score,
                    "errors_count": int(output.get("errors_count", 0) or 0),
                    "warnings_count": int(output.get("warnings_count", 0) or 0),
                    "issues_count": int(output.get("issues_count", 0) or 0),
                }
            )
            if score is None:
                issues.append(DashIssue("error", "validation_missing_score", "Validation output missing score."))
            else:
                if not (0.0 <= score <= 1.0):
                    issues.append(DashIssue("error", "validation_score_range", f"Validation score {score} outside [0,1]."))
                elif score < 0.7:
                    issues.append(DashIssue("error", "validation_score_low", f"Validation score is low ({score:.2f})."))
                elif score < 0.9:
                    issues.append(DashIssue("warning", "validation_score_warning", f"Validation score below target ({score:.2f})."))
            if not bool(output.get("is_valid", False)):
                issues.append(DashIssue("error", "validation_invalid_scene", "Validation marked scene as invalid."))
            if int(output.get("errors_count", 0) or 0) > 0:
                issues.append(
                    DashIssue(
                        "error",
                        "validation_errors_present",
                        f"Validation contains {int(output.get('errors_count', 0) or 0)} errors.",
                    )
                )
            ev = output.get("expected_vehicles")
            av = output.get("actual_vehicles")
            if isinstance(ev, int) and isinstance(av, int) and ev != av:
                issues.append(
                    DashIssue(
                        "error",
                        "validation_vehicle_count_mismatch",
                        f"Validation vehicle mismatch: expected={ev}, actual={av}.",
                    )
                )

        if stage_name == "routes":
            route_files_count = 0
            route_waypoint_counts: List[int] = []
            route_files_list: List[str] = []
            routes_out = _load_json_if_exists(stage_dir / "output.json")
            if routes_out:
                route_files = routes_out.get("route_files")
                if isinstance(route_files, list):
                    route_files_list = [str(x) for x in route_files if isinstance(x, str)]
                    route_files_count = len(route_files_list)
            routes_dir = stage_dir / "routes"
            if routes_dir.exists():
                xml_files = sorted(routes_dir.glob("*.xml"))
                if xml_files:
                    route_files_count = len(xml_files)
                    if not route_files_list:
                        route_files_list = [x.name for x in xml_files]
                for xmlf in xml_files:
                    try:
                        root = ET.parse(xmlf).getroot()
                        count = len(list(root.iter("waypoint")))
                    except Exception:
                        count = 0
                    route_waypoint_counts.append(count)
                    if count <= 0:
                        issues.append(DashIssue("error", "routes_empty_waypoints", f"Route {xmlf.name} has no waypoints."))
                    elif count < 5:
                        issues.append(
                            DashIssue(
                                "warning",
                                "routes_short_waypoints",
                                f"Route {xmlf.name} has only {count} waypoints.",
                            )
                        )
            metrics["route_file_count"] = route_files_count
            stage_metrics["route_file_count"] = route_files_count
            stage_metrics["route_files"] = route_files_list
            stage_metrics["routes_dir"] = str(routes_dir)
            if route_waypoint_counts:
                stage_metrics["avg_waypoints"] = round(sum(route_waypoint_counts) / len(route_waypoint_counts), 1)
            if requested and route_files_count <= 0 and status == "ok":
                issues.append(DashIssue("error", "routes_missing_files", "Routes stage succeeded but no XML routes were found."))

        if stage_name == "carla_validation" and output:
            checks = output.get("checks") if isinstance(output.get("checks"), dict) else {}
            metrics_obj = output.get("metrics") if isinstance(output.get("metrics"), dict) else {}
            repairs = output.get("repairs") if isinstance(output.get("repairs"), list) else []
            spawn_failed_entries = (
                output.get("spawn_failed_entries") if isinstance(output.get("spawn_failed_entries"), list) else []
            )
            spawn_repairs = output.get("spawn_repairs") if isinstance(output.get("spawn_repairs"), list) else []
            passed = bool(output.get("passed", False))
            failure_reason = str(output.get("failure_reason") or "").strip()

            stage_metrics.update(
                {
                    "passed": passed,
                    "gate_mode": str(output.get("gate_mode") or "hard"),
                    "failure_reason": failure_reason or None,
                    "repairs_count": len(repairs),
                    "spawn_expected": metrics_obj.get("spawn_expected"),
                    "spawn_actual": metrics_obj.get("spawn_actual"),
                    "min_ttc_s": metrics_obj.get("min_ttc_s"),
                    "near_miss": metrics_obj.get("near_miss"),
                    "route_completion_min": metrics_obj.get("route_completion_min"),
                    "driving_score_min": metrics_obj.get("driving_score_min"),
                    "checks": checks,
                    "repairs": repairs,
                    "spawn_failed_count": len(spawn_failed_entries),
                    "spawn_failed_entries": spawn_failed_entries,
                    "spawn_repairs": spawn_repairs,
                    "final_routes_dir": output.get("final_routes_dir"),
                    "can_accept": bool(passed and status == "ok"),
                }
            )

            if requested and status == "ok" and not passed:
                msg = "CARLA validation hard gate failed."
                if failure_reason:
                    msg += f" Reason: {failure_reason}"
                issues.append(DashIssue("error", "stage_carla_validation_gate_failed", msg))
            if requested and status == "ok":
                missing_checks = [k for k, v in checks.items() if not bool(v)]
                if missing_checks:
                    issues.append(
                        DashIssue(
                            "error",
                            "stage_carla_validation_checks_failed",
                            f"Failed CARLA checks: {', '.join(sorted(missing_checks))}.",
                        )
                    )
                if spawn_failed_entries:
                    failed_files = []
                    for item in spawn_failed_entries[:5]:
                        if isinstance(item, dict):
                            failed_files.append(str(item.get("file", "")).strip() or "unknown_file")
                    suffix = " ..." if len(spawn_failed_entries) > 5 else ""
                    issues.append(
                        DashIssue(
                            "warning",
                            "stage_carla_spawn_failed_entries",
                            f"CARLA spawn failed for {len(spawn_failed_entries)} actor entries: "
                            f"{', '.join(failed_files)}{suffix}",
                        )
                    )

        stage_trace.append(
            {
                "stage": stage_name,
                "label": PIPELINE_STAGE_LABELS.get(stage_name, stage_name),
                "dir": stage_dir_name,
                "requested": requested,
                "present": present,
                "status": status,
                "success": success,
                "elapsed_s": elapsed,
                "metrics": stage_metrics,
                "output_exists": output is not None,
            }
        )

    # Cross-stage consistency checks.
    lp = metrics.get("legal_num_paths")
    pc = metrics.get("picked_ego_count")
    upc = metrics.get("picked_unique_path_count")
    rc = metrics.get("refined_ego_count")
    plc = metrics.get("placement_ego_count")
    vv = metrics.get("validation_actual_vehicles")
    vf = metrics.get("route_file_count")

    if isinstance(lp, int) and isinstance(upc, int) and upc > lp:
        issues.append(
            DashIssue(
                "error",
                "cross_stage_pick_gt_legal",
                f"Picked unique path count ({upc}) exceeds legal path count ({lp}).",
            )
        )
    if isinstance(pc, int) and isinstance(rc, int) and rc != pc:
        issues.append(
            DashIssue(
                "warning",
                "cross_stage_refine_count_mismatch",
                f"Refined ego count ({rc}) does not match picked ego count ({pc}).",
            )
        )
    if isinstance(rc, int) and isinstance(plc, int) and plc != rc:
        issues.append(
            DashIssue(
                "warning",
                "cross_stage_placement_count_mismatch",
                f"Placement ego count ({plc}) does not match refined ego count ({rc}).",
            )
        )
    if isinstance(plc, int) and isinstance(vv, int) and vv != plc:
        issues.append(
            DashIssue(
                "error",
                "cross_stage_validation_count_mismatch",
                f"Validation actual vehicles ({vv}) do not match placement ego count ({plc}).",
            )
        )
    if isinstance(vv, int) and isinstance(vf, int) and vf != vv:
        issues.append(
            DashIssue(
                "warning",
                "cross_stage_routes_count_mismatch",
                f"Route file count ({vf}) does not match validation actual vehicles ({vv}).",
            )
        )
    if ego_count > 0 and len(map_layers.get("ego_spawns", [])) < ego_count:
        issues.append(
            DashIssue(
                "warning",
                "ego_spawn_metadata_missing",
                f"Ego spawn markers found for {len(map_layers.get('ego_spawns', []))}/{ego_count} vehicles.",
            )
        )

    category_name = str(spec.get("category", ""))

    if category_name == "Roundabout Navigation":
        pair_source = map_layers.get("refined_paths") or map_layers.get("picked_paths") or []
        if isinstance(pair_source, list) and len(pair_source) >= 2:
            def _entry_exit_from_name(name: str) -> Tuple[str, str]:
                m1 = re.search(r"__entry=([NSEW])\(", name or "")
                m2 = re.search(r"__exit=([NSEW])\(", name or "")
                return (
                    (m1.group(1).lower() if m1 else "unknown"),
                    (m2.group(1).lower() if m2 else "unknown"),
                )

            ok = False
            best_dist = None
            best_pair = "unknown"
            for i in range(len(pair_source)):
                for j in range(i + 1, len(pair_source)):
                    a = pair_source[i] if isinstance(pair_source[i], dict) else {}
                    b = pair_source[j] if isinstance(pair_source[j], dict) else {}
                    ap = a.get("polyline", []) if isinstance(a.get("polyline"), list) else []
                    bp = b.get("polyline", []) if isinstance(b.get("polyline"), list) else []
                    d = _polyline_pair_min_distance(ap, bp)
                    if d is not None and (best_dist is None or d < best_dist):
                        best_dist = d
                        best_pair = f"{a.get('vehicle', '?')} vs {b.get('vehicle', '?')}"
                    a_entry, a_exit = _entry_exit_from_name(str(a.get("name", "")))
                    b_entry, b_exit = _entry_exit_from_name(str(b.get("name", "")))
                    same_exit = a_exit != "unknown" and a_exit == b_exit
                    swap_merge = (
                        (a_exit != "unknown" and b_entry != "unknown" and a_exit == b_entry)
                        or (b_exit != "unknown" and a_entry != "unknown" and b_exit == a_entry)
                    )
                    close_enough = d is not None and d <= 6.0
                    if same_exit or swap_merge or close_enough:
                        ok = True
                        break
                if ok:
                    break
            if not ok:
                msg = "Roundabout ego paths appear disjoint."
                if best_dist is not None:
                    msg = f"Roundabout ego paths appear disjoint (closest pair {best_pair}: {best_dist:.1f}m)."
                issues.append(DashIssue("error", "roundabout_disjoint_paths", msg))

    # Clean up sets for JSON serialization.
    if isinstance(metrics.get("legal_candidate_names"), set):
        metrics["legal_candidate_names"] = sorted(metrics["legal_candidate_names"])

    map_layers["meta"] = {
        "legal_count": len(map_layers["legal_paths"]),
        "picked_count": len(map_layers["picked_paths"]),
        "refined_count": len(map_layers["refined_paths"]),
        "actors_count": len(map_layers["actors"]),
        "ego_spawn_count": len(map_layers["ego_spawns"]),
        "actor_traj_count": sum(1 for a in map_layers["actors"] if isinstance(a, dict) and a.get("trajectory")),
    }
    return stage_trace, map_layers, issues, metrics.get("validation_score")


def _constraint_phrase(ctype: str, a: str, b: str) -> str:
    mapping = {
        "same_approach_as": f"{a} approaches from the same direction as {b}.",
        "opposite_approach_of": f"{a} is oncoming relative to {b}.",
        "perpendicular_right_of": f"{a} approaches from the right-side perpendicular road of {b}.",
        "perpendicular_left_of": f"{a} approaches from the left-side perpendicular road of {b}.",
        "same_exit_as": f"{a} exits via the same road as {b}.",
        "same_road_as": f"{a} ends up on the same road as {b}.",
        "follow_route_of": f"{a} follows the route of {b}.",
        "left_lane_of": f"{a} is in the lane left of {b}.",
        "right_lane_of": f"{a} is in the lane right of {b}.",
        "merges_into_lane_of": f"{a} merges into the lane occupied by {b}.",
        "same_lane_as": f"{a} uses the same lane as {b}.",
    }
    return mapping.get(ctype, f"{ctype}({a}->{b}).")


def _vehicle_sentence(vehicle: Dict[str, Any]) -> str:
    vid = vehicle.get("vehicle_id", "Vehicle ?")
    maneuver = str(vehicle.get("maneuver", "unknown"))
    entry = str(vehicle.get("entry_road", "unknown"))
    exit_road = str(vehicle.get("exit_road", "unknown"))
    phase = str(vehicle.get("lane_change_phase", "unknown"))
    if maneuver == "lane_change":
        if phase in {"before_intersection", "after_intersection"}:
            return f"{vid} starts on {entry} and performs a lane change ({phase}), then exits on {exit_road}."
        return f"{vid} starts on {entry}, performs a lane change, and exits on {exit_road}."
    return f"{vid} starts on {entry}, drives {maneuver}, and exits on {exit_road}."


def _actor_sentence(actor: Dict[str, Any]) -> str:
    actor_id = str(actor.get("actor_id", "actor"))
    kind = str(actor.get("kind", "unknown"))
    motion = str(actor.get("motion", "unknown"))
    qty = int(actor.get("quantity", 1) or 1)
    target = actor.get("affects_vehicle")
    timing = str(actor.get("timing_phase", "unknown"))
    lateral = str(actor.get("lateral_position", "center"))
    qty_text = f"{qty}x " if qty > 1 else ""
    if target:
        return f"{qty_text}{actor_id} ({kind}) {motion} near {target} at {timing}, lateral={lateral}."
    return f"{qty_text}{actor_id} ({kind}) {motion} at {timing}, lateral={lateral}."


def _core_conflict_sentence(spec: Dict[str, Any]) -> str:
    flags = {
        "oncoming": bool(spec.get("needs_oncoming")),
        "multi_lane": bool(spec.get("needs_multi_lane")),
        "on_ramp": bool(spec.get("needs_on_ramp")),
        "merge": bool(spec.get("needs_merge")),
    }
    if flags["on_ramp"]:
        return "Core conflict: side-road on-ramp vehicle must merge into mainline traffic."
    if flags["merge"]:
        return "Core conflict: vehicles negotiate merge priority into overlapping lane space."
    if flags["oncoming"]:
        return "Core conflict: oncoming traffic creates crossing/gap-acceptance interactions."
    if flags["multi_lane"]:
        return "Core conflict: multi-lane occupancy creates adjacency and lane-priority interactions."
    return "Core conflict: interaction depends primarily on directional/perpendicular path crossings."


def _natural_language_summary(spec: Dict[str, Any]) -> Tuple[str, str]:
    category = str(spec.get("category", "unknown"))
    topology = str(spec.get("topology", "unknown"))
    vehicles = sorted(spec.get("ego_vehicles", []), key=lambda v: _safe_int_vehicle_id(str(v.get("vehicle_id", ""))))
    constraints = list(spec.get("vehicle_constraints", []))
    actors = list(spec.get("actors", []))

    lines: List[str] = []
    lines.append(
        f"This is a {category} schema on {topology} topology with "
        f"{len(vehicles)} ego vehicles, {len(constraints)} inter-vehicle constraints, and {len(actors)} non-ego actor groups."
    )
    lines.append(_core_conflict_sentence(spec))
    lines.append("Vehicle intents:")
    for v in vehicles:
        lines.append(f"- {_vehicle_sentence(v)}")
    lines.append("Interaction rules:")
    if constraints:
        for c in constraints:
            lines.append(f"- {_constraint_phrase(str(c.get('type', 'unknown')), str(c.get('a', '?')), str(c.get('b', '?')))}")
    else:
        lines.append("- No explicit inter-vehicle constraints are defined.")
    lines.append("Actor setup:")
    if actors:
        for a in actors:
            lines.append(f"- {_actor_sentence(a)}")
    else:
        lines.append("- No non-ego actors are present.")

    compact = (
        f"{category}: {len(vehicles)} egos on {topology}; "
        f"flags(oncoming={bool(spec.get('needs_oncoming'))}, "
        f"multi_lane={bool(spec.get('needs_multi_lane'))}, "
        f"on_ramp={bool(spec.get('needs_on_ramp'))}, "
        f"merge={bool(spec.get('needs_merge'))})."
    )
    return ("\n".join(lines), compact)


def _constraint_pairs(spec: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    for c in spec.get("vehicle_constraints", []):
        if not isinstance(c, dict):
            continue
        t = str(c.get("type", "")).strip()
        a = str(c.get("a", "")).strip()
        b = str(c.get("b", "")).strip()
        if t and a and b:
            out.append((t, a, b))
    return out


def _custom_consistency_checks(spec: Dict[str, Any]) -> List[DashIssue]:
    issues: List[DashIssue] = []
    vehicles = [str(v.get("vehicle_id", "")).strip() for v in spec.get("ego_vehicles", []) if isinstance(v, dict)]
    vehicle_set = set(vehicles)
    pairs = _constraint_pairs(spec)
    cset = set(pairs)

    # Contradictions and redundancy checks.
    for t, a, b in pairs:
        if a == b:
            issues.append(
                DashIssue("error", "self_relation", f"Self-referential constraint detected: {t}({a}->{b})", "Use two distinct vehicles.")
            )

    symmetric_types = {"opposite_approach_of", "same_approach_as", "same_exit_as", "same_road_as", "same_lane_as"}
    seen_sym_pairs = set()
    for t, a, b in pairs:
        if t in symmetric_types and (t, b, a) in cset:
            key = (t, tuple(sorted((a, b))))
            if key in seen_sym_pairs:
                continue
            seen_sym_pairs.add(key)
            issues.append(
                DashIssue("info", "symmetric_duplicate", f"Redundant bidirectional symmetric constraint: {t} between {a} and {b}")
            )

    if ("left_lane_of", "Vehicle 1", "Vehicle 1") in cset:
        issues.append(DashIssue("error", "invalid_lane_relation", "Vehicle cannot be left lane of itself."))

    # Pairwise contradiction families.
    by_pair: Dict[Tuple[str, str], set] = defaultdict(set)
    by_unordered: Dict[Tuple[str, str], set] = defaultdict(set)
    for t, a, b in pairs:
        by_pair[(a, b)].add(t)
        key = tuple(sorted((a, b)))
        by_unordered[key].add(t)

    for (a, b), types in by_pair.items():
        if "left_lane_of" in types and "right_lane_of" in types:
            issues.append(
                DashIssue("error", "lane_contradiction", f"Both left_lane_of and right_lane_of defined for {a}->{b}.")
            )
        if "same_approach_as" in types and "opposite_approach_of" in types:
            issues.append(
                DashIssue("error", "approach_contradiction", f"Both same_approach_as and opposite_approach_of defined for {a}->{b}.")
            )
        if "same_lane_as" in types and ("left_lane_of" in types or "right_lane_of" in types):
            issues.append(
                DashIssue("warning", "lane_overconstraint", f"same_lane_as overlaps with left/right lane relation for {a}->{b}.")
            )

    for (a, b), types in by_unordered.items():
        if "merges_into_lane_of" in types:
            if ("merges_into_lane_of", a, b) in cset and ("merges_into_lane_of", b, a) in cset:
                issues.append(
                    DashIssue("warning", "mutual_merge", f"Mutual merge detected between {a} and {b}.")
                )

    # Interaction graph connectivity (vehicle-only edges).
    graph: Dict[str, set] = {v: set() for v in vehicles}
    for _t, a, b in pairs:
        if a in vehicle_set and b in vehicle_set:
            graph[a].add(b)
            graph[b].add(a)
    # actor targets count as soft interaction edges to avoid false isolated warnings.
    targeted = set()
    for a in spec.get("actors", []):
        if not isinstance(a, dict):
            continue
        t = str(a.get("affects_vehicle", "")).strip()
        if t in vehicle_set:
            targeted.add(t)

    isolated = [v for v in vehicles if not graph[v] and v not in targeted]
    if isolated:
        sev = "error" if spec.get("category") == "Intersection Deadlock Resolution" else "warning"
        issues.append(
            DashIssue(sev, "interaction_coverage", f"Isolated vehicles without interactions: {', '.join(isolated)}")
        )

    # Topology vs actor timing sanity.
    topology = str(spec.get("topology", "")).strip()
    if topology in {"corridor", "highway", "two_lane_corridor"}:
        bad_timing = []
        for actor in spec.get("actors", []):
            if isinstance(actor, dict) and str(actor.get("timing_phase", "")).strip() == "in_intersection":
                bad_timing.append(str(actor.get("actor_id", "actor")))
        if bad_timing:
            issues.append(
                DashIssue(
                    "warning",
                    "actor_timing_topology",
                    f"Actors tagged as in_intersection on non-intersection topology: {', '.join(bad_timing)}",
                    "Use on_approach / after_exit / after_merge for corridor/highway contexts.",
                )
            )

    # Unknown reference checks.
    for _t, a, b in pairs:
        if a not in vehicle_set or b not in vehicle_set:
            issues.append(
                DashIssue("error", "unknown_vehicle_reference", f"Constraint references unknown vehicle(s): {_t}({a}->{b})")
            )

    return issues


def _dedupe_issues(issues: List[DashIssue]) -> List[DashIssue]:
    seen = set()
    out: List[DashIssue] = []
    severity_rank = {"error": 0, "warning": 1, "info": 2}
    for issue in sorted(issues, key=lambda x: (severity_rank.get(x.severity, 3), x.rule, x.message)):
        key = (issue.severity, issue.rule, issue.message)
        if key in seen:
            continue
        seen.add(key)
        out.append(issue)
    return out


def _compute_score(issues: List[DashIssue]) -> int:
    score = 100
    for i in issues:
        if i.severity == "error":
            score -= 25
        elif i.severity == "warning":
            score -= 8
        else:
            score -= 2
    if score < 0:
        score = 0
    return score


def _fingerprint(spec: Dict[str, Any]) -> str:
    vehicle_sig = ",".join(
        f"{v.get('vehicle_id')}:{v.get('maneuver')}:{v.get('entry_road')}->{v.get('exit_road')}"
        for v in sorted(spec.get("ego_vehicles", []), key=lambda x: _safe_int_vehicle_id(str(x.get("vehicle_id", ""))))
    )
    constraint_sig = ",".join(
        f"{c.get('type')}:{c.get('a')}->{c.get('b')}"
        for c in sorted(spec.get("vehicle_constraints", []), key=lambda x: (str(x.get("type")), str(x.get("a")), str(x.get("b"))))
    )
    actor_sig = ",".join(
        f"{a.get('kind')}:{a.get('actor_id')}:{a.get('quantity')}"
        for a in sorted(spec.get("actors", []), key=lambda x: (str(x.get("kind")), str(x.get("actor_id"))))
    )
    return f"{vehicle_sig}|{constraint_sig}|{actor_sig}"


def _read_route_xy_points(xml_path: Path) -> List[Tuple[float, float]]:
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return []
    pts: List[Tuple[float, float]] = []
    for wp in root.iter("waypoint"):
        try:
            x = float(wp.attrib.get("x", "nan"))
            y = float(wp.attrib.get("y", "nan"))
        except Exception:
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        pts.append((x, y))
    return pts


def _sample_polyline_points(points: List[Tuple[float, float]], max_points: int = 8) -> List[Tuple[float, float]]:
    if not points:
        return []
    if len(points) <= max_points:
        return list(points)
    if max_points <= 2:
        return [points[0], points[-1]]
    out: List[Tuple[float, float]] = [points[0]]
    interior = max_points - 2
    span = max(1, len(points) - 1)
    for i in range(1, interior + 1):
        idx = int(round(i * span / (interior + 1)))
        idx = max(1, min(len(points) - 2, idx))
        out.append(points[idx])
    out.append(points[-1])
    return out


def _route_actor_signature_for_run(run_dir: Path) -> Dict[str, Any]:
    routes_dir = run_dir / "09_routes" / "routes"
    manifest_path = routes_dir / "actors_manifest.json"
    manifest = _load_json_if_exists(manifest_path) if manifest_path.exists() else None
    if not isinstance(manifest, dict):
        return {
            "fingerprint": f"missing_manifest::{run_dir.name}",
            "tokens": [],
            "role_counts": {},
            "ego_route_count": 0,
        }

    def _q(v: float, step: float = 1.0) -> int:
        return int(round(round(float(v) / step) * step))

    tokens: List[str] = []
    role_counts: Dict[str, int] = {}
    ego_route_count = 0
    role_order = ["ego", "npc", "pedestrian", "bicycle", "static"]
    for role in role_order:
        entries = manifest.get(role)
        if not isinstance(entries, list):
            continue
        role_counts[role] = len(entries)
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            rel = str(entry.get("file", "")).strip()
            if not rel:
                continue
            route_path = routes_dir / rel
            points = _read_route_xy_points(route_path)
            if not points:
                continue
            if role == "static":
                sample = [points[0]]
            else:
                sample = _sample_polyline_points(points, max_points=8)
            if role == "ego":
                ego_route_count += 1
            qpts = [(int(_q(x, 1.0)), int(_q(y, 1.0))) for x, y in sample]
            model = str(entry.get("model", "")).strip()
            # Keep geometry/placement semantics for dedupe + similarity.
            # Avoid file/index identity so equivalent scenarios hash together.
            tokens.append(
                json.dumps(
                    {
                        "role": role,
                        "model": model,
                        "wp_count": int(len(points)),
                        "qpts": qpts,
                    },
                    sort_keys=True,
                    ensure_ascii=True,
                )
            )

    tokens.sort()
    payload = "\n".join(tokens).encode("utf-8")
    fingerprint = hashlib.sha1(payload).hexdigest() if tokens else f"no_tokens::{run_dir.name}"
    return {
        "fingerprint": fingerprint,
        "tokens": tokens,
        "role_counts": role_counts,
        "ego_route_count": int(ego_route_count),
    }


def _interaction_count_from_spec(spec: Dict[str, Any]) -> int:
    constraints = spec.get("vehicle_constraints")
    if not isinstance(constraints, list):
        return 0
    count = 0
    for c in constraints:
        if isinstance(c, dict) and str(c.get("type", "")).strip():
            count += 1
    return count


def _compute_interest_score(
    *,
    validation_score: Optional[float],
    interaction_count: int,
    ego_count: int,
    role_counts: Dict[str, int],
) -> float:
    vscore = 0.0
    if isinstance(validation_score, (int, float)):
        vscore = max(0.0, min(1.0, float(validation_score)))

    npc = int(role_counts.get("npc", 0) or 0)
    ped = int(role_counts.get("pedestrian", 0) or 0)
    bike = int(role_counts.get("bicycle", 0) or 0)
    static = int(role_counts.get("static", 0) or 0)
    dynamic_count = npc + ped + bike

    validation_term = 20.0 * vscore
    interaction_term = 35.0 * min(1.0, float(interaction_count) / 4.0)
    ego_term = 20.0 * min(1.0, float(max(0, ego_count)) / 3.0)
    dynamic_term = 20.0 * min(1.0, float(dynamic_count) / 4.0)
    static_context_term = 5.0 * min(1.0, float(static) / 8.0)
    score = validation_term + interaction_term + ego_term + dynamic_term + static_context_term
    return round(max(0.0, min(100.0, score)), 2)


def _jaccard_similarity(a: List[str], b: List[str]) -> float:
    aset = set(a)
    bset = set(b)
    if not aset and not bset:
        return 1.0
    if not aset or not bset:
        return 0.0
    inter = len(aset & bset)
    union = len(aset | bset)
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _jaccard_similarity_sets(aset: set, bset: set) -> float:
    if not aset and not bset:
        return 1.0
    if not aset or not bset:
        return 0.0
    inter = len(aset & bset)
    union = len(aset | bset)
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _stable_run_sort_key(run: SchemaRun) -> Tuple[Any, ...]:
    return (
        str(run.timestamp),
        str(run.category),
        int(run.seed) if isinstance(run.seed, int) else 10**9,
        int(run.seed_redo_index),
        str(run.run_name),
    )


def _annotate_similarity_and_duplicates(
    runs: List[SchemaRun],
    similarity_pair_limit: int = 200,
) -> List[Dict[str, Any]]:
    """
    Annotate per-run similarity/duplicate metadata and return top similar pairs.
    Duplicate policy is strict (exact route+actor fingerprint match).
    """
    if not runs:
        return []

    for r in runs:
        r.interest_score_adjusted = float(r.interest_score)
        r.similarity_best = 0.0
        r.similarity_peer = None
        r.similarity_cluster_size = 1
        r.duplicate_of = None

    token_sets: List[set] = [set(r.route_actor_tokens or []) for r in runs]
    pair_rows: List[Dict[str, Any]] = []
    n = len(runs)
    for i in range(n):
        for j in range(i + 1, n):
            sim = _jaccard_similarity_sets(token_sets[i], token_sets[j])
            if sim >= 0.55:
                pair_rows.append(
                    {
                        "similarity": round(float(sim), 4),
                        "run_a": runs[i].run_name,
                        "run_b": runs[j].run_name,
                        "category_a": runs[i].category,
                        "category_b": runs[j].category,
                        "seed_a": runs[i].seed,
                        "seed_b": runs[j].seed,
                        "run_dir_a": runs[i].run_dir,
                        "run_dir_b": runs[j].run_dir,
                    }
                )

            if sim > runs[i].similarity_best:
                runs[i].similarity_best = float(sim)
                runs[i].similarity_peer = runs[j].run_name
            if sim > runs[j].similarity_best:
                runs[j].similarity_best = float(sim)
                runs[j].similarity_peer = runs[i].run_name

    by_fp: Dict[str, List[int]] = defaultdict(list)
    for idx, run in enumerate(runs):
        fp = str(run.route_actor_fingerprint or "").strip()
        if fp:
            by_fp[fp].append(idx)

    for idxs in by_fp.values():
        if len(idxs) <= 1:
            continue
        idxs_sorted = sorted(idxs, key=lambda i: _stable_run_sort_key(runs[i]))
        cluster_size = len(idxs_sorted)
        keeper_idx = idxs_sorted[0]
        keeper = runs[keeper_idx]
        for i in idxs_sorted:
            runs[i].similarity_cluster_size = max(runs[i].similarity_cluster_size, cluster_size)
            runs[i].similarity_best = max(runs[i].similarity_best, 1.0)
            if runs[i].similarity_peer is None:
                runs[i].similarity_peer = keeper.run_name if i != keeper_idx else (
                    runs[idxs_sorted[1]].run_name if len(idxs_sorted) > 1 else None
                )
            if i == keeper_idx:
                continue
            runs[i].duplicate_of = keeper.run_dir
            runs[i].interest_score_adjusted = 0.0

    for r in runs:
        r.similarity_best = round(float(r.similarity_best), 4)
        r.interest_score_adjusted = round(float(r.interest_score_adjusted), 2)

    pair_rows.sort(
        key=lambda row: (
            -float(row.get("similarity", 0.0)),
            str(row.get("category_a", "")),
            str(row.get("category_b", "")),
            str(row.get("run_a", "")),
            str(row.get("run_b", "")),
        )
    )
    return pair_rows[: max(0, int(similarity_pair_limit))]


def _find_latest_normalized_attempt(run_dir: Path) -> Optional[Dict[str, Any]]:
    schema_dir = run_dir / "01_schema"
    files = sorted(schema_dir.glob("schema_normalized_attempt*.json"))
    if not files:
        return None
    indexed: List[Tuple[int, Path]] = []
    for f in files:
        m = re.search(r"attempt(\d+)\.json$", f.name)
        idx = int(m.group(1)) if m else -1
        indexed.append((idx, f))
    indexed.sort(key=lambda x: x[0])
    latest = indexed[-1][1]
    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(data, dict) and "scenario_spec" in data and isinstance(data.get("scenario_spec"), dict):
        return data["scenario_spec"]
    return data if isinstance(data, dict) else None


def _constraint_set(spec: Dict[str, Any]) -> set:
    return set(_constraint_pairs(spec))


def _constraint_set_from_spec_obj(spec_obj: ScenarioSpec) -> set:
    out = set()
    for c in spec_obj.vehicle_constraints:
        out.add((str(c.constraint_type.value), str(c.vehicle_a), str(c.vehicle_b)))
    return out


def _format_constraint_preview(items: List[Tuple[str, str, str]], max_items: int = 6) -> str:
    if not items:
        return ""
    body = ", ".join(f"{t}({a}->{b})" for t, a, b in items[:max_items])
    if len(items) > max_items:
        return f"{body} ..."
    return body


def _vehicle_map(spec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for v in spec.get("ego_vehicles", []):
        if not isinstance(v, dict):
            continue
        vid = str(v.get("vehicle_id", "")).strip()
        if not vid:
            continue
        out[vid] = {
            "maneuver": v.get("maneuver"),
            "lane_change_phase": v.get("lane_change_phase"),
            "entry_road": v.get("entry_road"),
            "exit_road": v.get("exit_road"),
        }
    return out


def _actor_map(spec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for a in spec.get("actors", []):
        if not isinstance(a, dict):
            continue
        key = str(a.get("actor_id", "")).strip()
        if not key:
            key = f"{a.get('kind','actor')}#{len(out)+1}"
        out[key] = {
            "kind": a.get("kind"),
            "motion": a.get("motion"),
            "timing_phase": a.get("timing_phase"),
            "lateral_position": a.get("lateral_position"),
            "affects_vehicle": a.get("affects_vehicle"),
            "quantity": a.get("quantity"),
        }
    return out


def _deterministic_projection_issues(run_dir: Path, final_spec: Dict[str, Any]) -> Tuple[List[DashIssue], List[str]]:
    """
    Compare pre-deterministic normalized schema to deterministic projection/final output.
    Flags implicit interactions that were not explicit in the LLM schema.
    """
    issues: List[DashIssue] = []
    adjustments: List[str] = []

    normalized = _find_latest_normalized_attempt(run_dir)
    if not normalized:
        adjustments.append("No schema_normalized_attempt*.json artifact found; pre/post deterministic diff unavailable.")
        return issues, adjustments

    try:
        raw_spec = spec_from_dict(copy.deepcopy(normalized))
    except Exception as exc:
        issues.append(
            DashIssue(
                "info",
                "deterministic_projection",
                f"Could not parse latest normalized attempt for deterministic projection diff: {exc}",
            )
        )
        adjustments.append(f"Could not parse normalized attempt for deterministic diff: {exc}")
        return issues, adjustments

    step_pipeline = [
        ("ensure_direction_and_lane_constraints", _ensure_direction_and_lane_constraints),
        ("canonicalize_constraints_pass1", _canonicalize_constraints),
        ("canonicalize_actors", _canonicalize_actors),
        ("ensure_interaction_coverage", _ensure_interaction_coverage),
        ("canonicalize_constraints_pass2", _canonicalize_constraints),
    ]

    projected = copy.deepcopy(raw_spec)
    for step_name, fn in step_pipeline:
        before = copy.deepcopy(projected)
        updated = fn(projected)
        if updated is not None:
            projected = updated
        before_constraints = _constraint_set_from_spec_obj(before)
        after_constraints = _constraint_set_from_spec_obj(projected)
        added_constraints = sorted(after_constraints - before_constraints)
        removed_constraints = sorted(before_constraints - after_constraints)

        if added_constraints:
            adjustments.append(
                f"[{step_name}] +{len(added_constraints)} constraints: "
                f"{_format_constraint_preview(added_constraints)}"
            )
        if removed_constraints:
            adjustments.append(
                f"[{step_name}] -{len(removed_constraints)} constraints: "
                f"{_format_constraint_preview(removed_constraints)}"
            )

        if step_name == "ensure_direction_and_lane_constraints" and added_constraints:
            issues.append(
                DashIssue(
                    "warning",
                    "implicit_interactions_direction_lane",
                    f"Deterministic direction/lane expansion added {len(added_constraints)} constraints: "
                    f"{_format_constraint_preview(added_constraints)}",
                    "Make required direction/lane/merge relations explicit in generated schema.",
                )
            )
        if step_name == "ensure_interaction_coverage" and added_constraints:
            issues.append(
                DashIssue(
                    "warning",
                    "implicit_interaction_coverage",
                    f"Deterministic coverage patch added {len(added_constraints)} constraints for uncovered ego vehicles: "
                    f"{_format_constraint_preview(added_constraints)}",
                    "Ensure every ego vehicle has explicit interaction edges in schema output.",
                )
            )
        if step_name.startswith("canonicalize_constraints") and removed_constraints:
            issues.append(
                DashIssue(
                    "warning",
                    "constraints_canonicalized",
                    f"Constraint canonicalization removed/rewrote {len(removed_constraints)} constraints: "
                    f"{_format_constraint_preview(removed_constraints)}",
                    "Remove duplicates and contradictory directional relations from schema generation.",
                )
            )

    projected_dict = spec_to_dict(projected)
    raw_dict = spec_to_dict(raw_spec)
    raw_constraints = _constraint_set(raw_dict)
    proj_constraints = _constraint_set(projected_dict)
    final_constraints = _constraint_set(final_spec)

    added_constraints = sorted(proj_constraints - raw_constraints)
    removed_constraints = sorted(raw_constraints - proj_constraints)
    if added_constraints:
        active_types = {
            ConstraintType.OPPOSITE_APPROACH_OF.value,
            ConstraintType.PERPENDICULAR_LEFT_OF.value,
            ConstraintType.PERPENDICULAR_RIGHT_OF.value,
            ConstraintType.LEFT_LANE_OF.value,
            ConstraintType.RIGHT_LANE_OF.value,
            ConstraintType.MERGES_INTO_LANE_OF.value,
            ConstraintType.SAME_LANE_AS.value,
            ConstraintType.FOLLOW_ROUTE_OF.value,
        }
        active_added = [c for c in added_constraints if c[0] in active_types]
        if active_added:
            issues.append(
                DashIssue(
                    "warning",
                    "implicit_active_interactions",
                    f"Deterministic logic had to synthesize {len(active_added)} active interactions not explicit in LLM schema: "
                    f"{_format_constraint_preview(active_added)}",
                    "Output these interactions directly so deterministic stage is validation-only, not behavior-defining.",
                )
            )
    if removed_constraints:
        issues.append(
            DashIssue(
                "warning",
                "constraints_rewritten",
                f"Final deterministic projection rewrote {len(removed_constraints)} raw constraints: "
                f"{_format_constraint_preview(removed_constraints)}",
                "Avoid duplicate inverse/symmetric constraints and ambiguous merge direction in generation output.",
            )
        )

    raw_vehicles = _vehicle_map(raw_dict)
    proj_vehicles = _vehicle_map(projected_dict)
    vehicle_changes: List[str] = []
    promoted_unknown = 0
    for vid in sorted(set(raw_vehicles) | set(proj_vehicles), key=_safe_int_vehicle_id):
        rv = raw_vehicles.get(vid, {})
        pv = proj_vehicles.get(vid, {})
        if not rv and pv:
            vehicle_changes.append(f"{vid}: added by deterministic logic")
            continue
        if rv and not pv:
            vehicle_changes.append(f"{vid}: removed by deterministic logic")
            continue
        changed_fields = [f for f in ("maneuver", "lane_change_phase", "entry_road", "exit_road") if rv.get(f) != pv.get(f)]
        if changed_fields:
            detail_parts = []
            for f in changed_fields:
                before_v = rv.get(f)
                after_v = pv.get(f)
                if before_v in {None, "unknown"} and after_v not in {None, "unknown"}:
                    promoted_unknown += 1
                detail_parts.append(f"{f}: {before_v} -> {after_v}")
            vehicle_changes.append(f"{vid}: {', '.join(detail_parts)}")

    if vehicle_changes:
        preview = "; ".join(vehicle_changes[:4])
        suffix = " ..." if len(vehicle_changes) > 4 else ""
        issues.append(
            DashIssue(
                "warning",
                "vehicle_fields_adjusted",
                f"Deterministic logic adjusted vehicle semantics ({len(vehicle_changes)} changes): {preview}{suffix}",
                "Emit complete vehicle metadata (entry/exit/maneuver/phase) directly in schema generation output.",
            )
        )
        if promoted_unknown > 0:
            issues.append(
                DashIssue(
                    "warning",
                    "unknown_vehicle_fields_promoted",
                    f"{promoted_unknown} vehicle fields were promoted from unknown/None to concrete values deterministically.",
                    "Unknown placeholders should be resolved by schema generation, not inferred post hoc.",
                )
            )
        adjustments.extend(vehicle_changes[:6])

    raw_actors = _actor_map(raw_dict)
    proj_actors = _actor_map(projected_dict)
    actor_changes: List[str] = []
    for aid in sorted(set(raw_actors) | set(proj_actors)):
        ra = raw_actors.get(aid, {})
        pa = proj_actors.get(aid, {})
        if not ra and pa:
            actor_changes.append(f"{aid}: added by deterministic logic")
            continue
        if ra and not pa:
            actor_changes.append(f"{aid}: removed by deterministic logic")
            continue
        changed_fields = [f for f in ("timing_phase", "lateral_position", "affects_vehicle", "motion", "kind", "quantity") if ra.get(f) != pa.get(f)]
        if changed_fields:
            detail = ", ".join(f"{f}: {ra.get(f)} -> {pa.get(f)}" for f in changed_fields)
            actor_changes.append(f"{aid}: {detail}")

    if actor_changes:
        preview = "; ".join(actor_changes[:4])
        suffix = " ..." if len(actor_changes) > 4 else ""
        issues.append(
            DashIssue(
                "info",
                "actor_fields_adjusted",
                f"Deterministic logic adjusted actor fields ({len(actor_changes)} changes): {preview}{suffix}",
            )
        )
        adjustments.extend(actor_changes[:6])

    raw_covered = set()
    raw_vehicle_ids = set(raw_vehicles.keys())
    for _t, a, b in raw_constraints:
        if a in raw_vehicle_ids:
            raw_covered.add(a)
        if b in raw_vehicle_ids:
            raw_covered.add(b)
    for actor in raw_dict.get("actors", []):
        if isinstance(actor, dict):
            t = str(actor.get("affects_vehicle", "")).strip()
            if t in raw_vehicle_ids:
                raw_covered.add(t)

    proj_covered = set()
    for _t, a, b in proj_constraints:
        if a in raw_vehicle_ids:
            proj_covered.add(a)
        if b in raw_vehicle_ids:
            proj_covered.add(b)
    for actor in projected_dict.get("actors", []):
        if isinstance(actor, dict):
            t = str(actor.get("affects_vehicle", "")).strip()
            if t in raw_vehicle_ids:
                proj_covered.add(t)

    newly_covered = sorted(proj_covered - raw_covered, key=_safe_int_vehicle_id)
    if newly_covered:
        issues.append(
            DashIssue(
                "warning",
                "implicit_vehicle_connectivity",
                f"Vehicles were disconnected in raw schema and became connected only after deterministic expansion: "
                f"{', '.join(newly_covered)}",
                "Define at least one explicit interaction (constraint or actor target) for every ego vehicle.",
            )
        )

    if proj_constraints != final_constraints:
        proj_only = sorted(proj_constraints - final_constraints)
        final_only = sorted(final_constraints - proj_constraints)
        detail_parts: List[str] = []
        if proj_only:
            detail_parts.append(f"projection-only: {_format_constraint_preview(proj_only, max_items=4)}")
        if final_only:
            detail_parts.append(f"final-only: {_format_constraint_preview(final_only, max_items=4)}")
        detail = " | ".join(detail_parts) if detail_parts else "constraint sets differ"
        issues.append(
            DashIssue(
                "info",
                "projection_final_mismatch",
                f"Deterministic projection from latest normalized attempt differs from final schema output ({detail}).",
                "Inspect attempt artifacts when repair-loop selected an earlier successful attempt.",
            )
        )

    if not adjustments:
        adjustments.append("No deterministic structural adjustments detected (normalized attempt already explicit).")
    return issues, adjustments


def _analyze_single_run(run_dir: Path, category_audit_hook) -> SchemaRun:
    out_path = run_dir / "01_schema" / "output.json"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    spec = payload.get("spec", {})
    summary = _load_json_if_exists(run_dir / "summary.json") or {}
    if not isinstance(summary, dict):
        summary = {}
    category = str(spec.get("category", "unknown"))
    seed = _parse_seed_from_run_dir(run_dir)

    issues: List[DashIssue] = []
    schema_errors = [str(x) for x in (payload.get("errors", []) or [])]
    schema_warnings = [str(x) for x in (payload.get("warnings", []) or [])]
    for msg in schema_errors:
        issues.append(DashIssue("error", "schema_stage", msg))
    for msg in schema_warnings:
        issues.append(DashIssue("warning", "schema_stage", msg))

    # Structural checks via ScenarioSpec parsing.
    spec_obj: Optional[ScenarioSpec] = None
    parse_failed = False
    try:
        spec_obj = spec_from_dict(spec)
        valid, struct_errors = validate_spec(spec_obj)
        if not valid:
            for msg in struct_errors:
                issues.append(DashIssue("error", "validate_spec", str(msg)))
        cat_info = CATEGORY_DEFINITIONS.get(category)
        if cat_info is not None:
            c_errors, c_warnings = _conflict_findings(spec_obj, cat_info)
            for msg in c_errors:
                issues.append(DashIssue("error", "conflict_findings", str(msg)))
            for msg in c_warnings:
                issues.append(DashIssue("warning", "conflict_findings", str(msg)))
    except Exception as exc:
        parse_failed = True
        issues.append(DashIssue("error", "spec_parse", f"spec_from_dict failed: {exc}"))

    # Category-specific deep checks from audit_schema_outputs.
    if category_audit_hook is not None:
        try:
            audit_issues = category_audit_hook(spec)
            for i in audit_issues:
                sev = str(getattr(i, "severity", "warning"))
                rule = str(getattr(i, "category", "category_rule"))
                msg = str(getattr(i, "message", ""))
                issues.append(DashIssue(sev, f"category_audit:{rule}", msg))
        except Exception as exc:
            issues.append(DashIssue("warning", "category_audit", f"category audit hook failed: {exc}"))

    # Custom consistency checks.
    if not parse_failed:
        issues.extend(_custom_consistency_checks(spec))

    deterministic_issues, deterministic_adjustments = _deterministic_projection_issues(run_dir, spec)
    issues.extend(deterministic_issues)

    stage_trace, map_layers, pipeline_issues, validation_score = _infer_stage_trace_and_pipeline_findings(run_dir, spec)
    issues.extend(pipeline_issues)

    carla_stage = next((s for s in stage_trace if s.get("stage") == "carla_validation"), None)
    carla_metrics = carla_stage.get("metrics", {}) if isinstance(carla_stage, dict) else {}
    if not isinstance(carla_metrics, dict):
        carla_metrics = {}
    carla_validation = {
        "present": isinstance(carla_stage, dict),
        "stage_status": carla_stage.get("status") if isinstance(carla_stage, dict) else "missing",
        "passed": bool(carla_metrics.get("passed", False)),
        "failure_reason": carla_metrics.get("failure_reason"),
        "gate_mode": carla_metrics.get("gate_mode"),
        "checks": carla_metrics.get("checks", {}),
        "metrics": {
            "spawn_expected": carla_metrics.get("spawn_expected"),
            "spawn_actual": carla_metrics.get("spawn_actual"),
            "min_ttc_s": carla_metrics.get("min_ttc_s"),
            "near_miss": carla_metrics.get("near_miss"),
            "route_completion_min": carla_metrics.get("route_completion_min"),
            "driving_score_min": carla_metrics.get("driving_score_min"),
        },
        "repairs_count": int(carla_metrics.get("repairs_count", 0) or 0),
        "repairs": carla_metrics.get("repairs", []),
        "final_routes_dir": carla_metrics.get("final_routes_dir"),
        "can_accept": bool(carla_metrics.get("can_accept", False)),
    }
    target_acceptance = summary.get("target_acceptance") if isinstance(summary, dict) else {}
    if not isinstance(target_acceptance, dict):
        target_acceptance = {}
    acceptance_level = str(target_acceptance.get("acceptance_level", "")).strip().lower()
    if acceptance_level not in {"high", "medium", "rejected"}:
        acceptance_level = "high" if bool(carla_validation.get("passed", False)) else "rejected"
    if acceptance_level == "medium":
        carla_validation["can_accept"] = True
        carla_validation["acceptance_level"] = "medium"
    elif acceptance_level == "high":
        carla_validation["acceptance_level"] = "high"
    else:
        carla_validation["acceptance_level"] = "rejected"
    if target_acceptance:
        carla_validation["target_acceptance"] = target_acceptance
    can_accept = bool(carla_validation.get("can_accept", False))

    issues = _dedupe_issues(issues)
    score = _compute_score(issues)
    status = "pass" if all(i.severity != "error" for i in issues) else "fail"
    natural, compact = _natural_language_summary(spec)
    route_actor_sig = _route_actor_signature_for_run(run_dir)
    interaction_count = _interaction_count_from_spec(spec)
    interest_score = _compute_interest_score(
        validation_score=validation_score,
        interaction_count=interaction_count,
        ego_count=len(spec.get("ego_vehicles", [])),
        role_counts=route_actor_sig.get("role_counts", {}),
    )

    run_name = run_dir.name
    return SchemaRun(
        run_dir=str(run_dir),
        run_name=run_name,
        timestamp=_parse_timestamp_from_run_name(run_name),
        category=category,
        seed=seed,
        redo_index=1,  # assigned later
        seed_redo_index=1,  # assigned later
        spec=spec,
        natural_summary=natural,
        compact_summary=compact,
        issues=issues,
        score=score,
        status=status,
        ego_count=len(spec.get("ego_vehicles", [])),
        constraint_count=len(spec.get("vehicle_constraints", [])),
        actor_count=len(spec.get("actors", [])),
        schema_errors=schema_errors,
        schema_warnings=schema_warnings,
        deterministic_adjustments=deterministic_adjustments,
        stage_trace=stage_trace,
        map_layers=map_layers,
        validation_score=validation_score,
        carla_validation=carla_validation,
        can_accept=can_accept,
        fingerprint=_fingerprint(spec),
        route_actor_fingerprint=str(route_actor_sig.get("fingerprint", "")),
        route_actor_tokens=list(route_actor_sig.get("tokens") or []),
        role_counts=dict(route_actor_sig.get("role_counts") or {}),
        interaction_count=int(interaction_count),
        interest_score=float(interest_score),
        interest_score_adjusted=float(interest_score),
        similarity_best=0.0,
        similarity_peer=None,
        similarity_cluster_size=1,
        duplicate_of=None,
    )


def _assign_redo_indices(runs: List[SchemaRun]) -> None:
    by_cat: Dict[str, List[SchemaRun]] = defaultdict(list)
    by_cat_seed: Dict[Tuple[str, Optional[int]], List[SchemaRun]] = defaultdict(list)
    for r in runs:
        by_cat[r.category].append(r)
        by_cat_seed[(r.category, r.seed)].append(r)
    for cat, entries in by_cat.items():
        entries.sort(key=lambda x: (x.timestamp, x.run_name, x.seed if x.seed is not None else 10**9))
        for idx, entry in enumerate(entries, start=1):
            entry.redo_index = idx
    for _key, entries in by_cat_seed.items():
        entries.sort(key=lambda x: (x.timestamp, x.run_name))
        for idx, entry in enumerate(entries, start=1):
            entry.seed_redo_index = idx


def _category_summary(runs: List[SchemaRun]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    by_cat: Dict[str, List[SchemaRun]] = defaultdict(list)
    for r in runs:
        by_cat[r.category].append(r)
    for cat, entries in sorted(by_cat.items()):
        total = len(entries)
        pass_count = sum(1 for x in entries if x.status == "pass")
        avg_score = round(sum(x.score for x in entries) / total, 1) if total else 0.0
        errors = sum(sum(1 for i in x.issues if i.severity == "error") for x in entries)
        warnings = sum(sum(1 for i in x.issues if i.severity == "warning") for x in entries)
        infos = sum(sum(1 for i in x.issues if i.severity == "info") for x in entries)
        unique_fps = len({x.fingerprint for x in entries})
        seed_values = sorted({x.seed for x in entries if x.seed is not None})
        out[cat] = {
            "total": total,
            "pass_count": pass_count,
            "pass_rate": round(100.0 * pass_count / total, 1) if total else 0.0,
            "avg_score": avg_score,
            "errors": errors,
            "warnings": warnings,
            "infos": infos,
            "unique_fingerprints": unique_fps,
            "seed_count": len(seed_values),
            "seed_values": seed_values,
        }
    return out


def _issue_taxonomy(runs: List[SchemaRun]) -> List[Tuple[str, int]]:
    c = Counter()
    for r in runs:
        for i in r.issues:
            c[f"{i.severity}:{i.rule}"] += 1
    return c.most_common(30)


def _seed_summary(runs: List[SchemaRun]) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[str, Optional[int]], List[SchemaRun]] = defaultdict(list)
    for r in runs:
        by_key[(r.category, r.seed)].append(r)
    rows: List[Dict[str, Any]] = []
    for (category, seed), entries in sorted(
        by_key.items(),
        key=lambda x: (x[0][0], x[0][1] if x[0][1] is not None else 10**9),
    ):
        total = len(entries)
        pass_count = sum(1 for e in entries if e.status == "pass")
        rows.append(
            {
                "category": category,
                "seed": seed,
                "runs": total,
                "pass_count": pass_count,
                "pass_rate": round(100.0 * pass_count / total, 1) if total else 0.0,
                "avg_score": round(sum(e.score for e in entries) / total, 1) if total else 0.0,
                "unique_fingerprints": len({e.fingerprint for e in entries}),
            }
        )
    return rows


def _pipeline_stage_summary(runs: List[SchemaRun]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for stage_name, _stage_dir in PIPELINE_STAGES[1:]:
        requested = 0
        ok = 0
        failed = 0
        missing = 0
        elapsed_vals: List[float] = []
        stage_issue_count = 0
        rule_tag = f":{stage_name}"
        for r in runs:
            st = next((s for s in r.stage_trace if s.get("stage") == stage_name), None)
            if st is None:
                continue
            if not st.get("requested"):
                continue
            requested += 1
            status = st.get("status")
            if status == "ok":
                ok += 1
            elif status == "failed":
                failed += 1
            elif status == "missing":
                missing += 1
            elapsed = _to_float(st.get("elapsed_s"))
            if elapsed is not None:
                elapsed_vals.append(elapsed)
            for i in r.issues:
                if i.rule.endswith(rule_tag) or i.rule.startswith(f"stage_{stage_name}") or f":{stage_name}" in i.rule:
                    stage_issue_count += 1
        rows.append(
            {
                "stage": stage_name,
                "label": PIPELINE_STAGE_LABELS.get(stage_name, stage_name),
                "requested": requested,
                "ok": ok,
                "failed": failed,
                "missing": missing,
                "success_rate": round(100.0 * ok / requested, 1) if requested else 0.0,
                "avg_elapsed_s": round(sum(elapsed_vals) / len(elapsed_vals), 2) if elapsed_vals else None,
                "issue_count": stage_issue_count,
            }
        )
    return rows


def _b64_json_payload(obj: Any) -> str:
    raw = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def _format_stage_metrics(metrics: Dict[str, Any], max_items: int = 6) -> str:
    if not isinstance(metrics, dict) or not metrics:
        return ""
    parts: List[str] = []
    for k in sorted(metrics.keys()):
        v = metrics[k]
        if v is None:
            continue
        if isinstance(v, dict):
            continue
        if isinstance(v, list):
            continue
        if isinstance(v, float):
            text = f"{v:.3f}" if abs(v) < 10 else f"{v:.2f}"
        else:
            text = str(v)
        parts.append(f"{k}={text}")
        if len(parts) >= max_items:
            break
    return "; ".join(parts)


def _escape_json(obj: Any) -> str:
    return html.escape(json.dumps(obj, indent=2, sort_keys=True))


def _badge_class(sev: str) -> str:
    return {"error": "badge-error", "warning": "badge-warning", "info": "badge-info"}.get(sev, "badge-info")


def _render_html(runs: List[SchemaRun], title: str, similarity_pairs: Optional[List[Dict[str, Any]]] = None) -> str:
    total = len(runs)
    categories = sorted({r.category for r in runs})
    seed_values = sorted({r.seed for r in runs if r.seed is not None})
    unknown_seed_count = sum(1 for r in runs if r.seed is None)
    pass_count = sum(1 for r in runs if r.status == "pass")
    fail_count = total - pass_count
    avg_score = round(sum(r.score for r in runs) / total, 1) if total else 0.0
    total_errors = sum(sum(1 for i in r.issues if i.severity == "error") for r in runs)
    total_warnings = sum(sum(1 for i in r.issues if i.severity == "warning") for r in runs)
    total_infos = sum(sum(1 for i in r.issues if i.severity == "info") for r in runs)
    category_summary = _category_summary(runs)
    seed_summary = _seed_summary(runs)
    stage_summary = _pipeline_stage_summary(runs)
    issue_tax = _issue_taxonomy(runs)
    similarity_pairs = list(similarity_pairs or [])
    full_run_count = 0
    validation_scores: List[float] = []
    can_accept_count = 0
    interest_values: List[float] = []
    interest_adjusted_values: List[float] = []
    duplicate_count = 0
    unique_route_actor_fp_count = len({str(r.route_actor_fingerprint or "") for r in runs if str(r.route_actor_fingerprint or "")})
    for r in runs:
        carla_stage = next((s for s in r.stage_trace if s.get("stage") == "carla_validation"), None)
        if carla_stage and carla_stage.get("requested") and carla_stage.get("status") == "ok":
            full_run_count += 1
        if r.can_accept:
            can_accept_count += 1
        if isinstance(r.validation_score, (int, float)):
            validation_scores.append(float(r.validation_score))
        interest_values.append(float(r.interest_score))
        interest_adjusted_values.append(float(r.interest_score_adjusted))
        if r.duplicate_of:
            duplicate_count += 1
    avg_validation = round(sum(validation_scores) / len(validation_scores), 3) if validation_scores else None
    avg_interest = round(sum(interest_values) / len(interest_values), 2) if interest_values else None
    avg_interest_adjusted = round(sum(interest_adjusted_values) / len(interest_adjusted_values), 2) if interest_adjusted_values else None

    category_rows = []
    for cat, s in category_summary.items():
        seed_text = ", ".join(str(x) for x in s.get("seed_values", [])) if s.get("seed_values") else "n/a"
        category_rows.append(
            f"""
            <tr>
              <td>{html.escape(cat)}</td>
              <td>{s['total']}</td>
              <td>{s['pass_count']} ({s['pass_rate']}%)</td>
              <td>{s['avg_score']}</td>
              <td>{s['errors']}</td>
              <td>{s['warnings']}</td>
              <td>{s['infos']}</td>
              <td>{s['unique_fingerprints']}</td>
              <td>{s.get('seed_count', 0)} ({html.escape(seed_text)})</td>
            </tr>
            """
        )

    seed_rows = []
    for row in seed_summary:
        seed_label = str(row["seed"]) if row["seed"] is not None else "unknown"
        seed_rows.append(
            f"""
            <tr>
              <td>{html.escape(row['category'])}</td>
              <td>{html.escape(seed_label)}</td>
              <td>{row['runs']}</td>
              <td>{row['pass_count']} ({row['pass_rate']}%)</td>
              <td>{row['avg_score']}</td>
              <td>{row['unique_fingerprints']}</td>
            </tr>
            """
        )

    stage_rows = []
    for row in stage_summary:
        avg_elapsed_text = f"{row['avg_elapsed_s']:.2f}s" if isinstance(row.get("avg_elapsed_s"), (int, float)) else "n/a"
        stage_rows.append(
            f"""
            <tr>
              <td>{html.escape(row['label'])}</td>
              <td>{row['requested']}</td>
              <td>{row['ok']} ({row['success_rate']}%)</td>
              <td>{row['failed']}</td>
              <td>{row['missing']}</td>
              <td>{avg_elapsed_text}</td>
              <td>{row['issue_count']}</td>
            </tr>
            """
        )

    issue_rows = []
    for key, count in issue_tax:
        sev, rule = key.split(":", 1)
        issue_rows.append(
            f"<tr><td><span class='badge {_badge_class(sev)}'>{sev.upper()}</span></td><td>{html.escape(rule)}</td><td>{count}</td></tr>"
        )

    ranking_rows = []
    ranked_runs = sorted(
        runs,
        key=lambda r: (
            -float(r.interest_score_adjusted),
            -float(r.interest_score),
            -float(r.validation_score) if isinstance(r.validation_score, (int, float)) else 0.0,
            str(r.category),
            int(r.seed) if isinstance(r.seed, int) else 10**9,
            str(r.run_name),
        ),
    )
    for idx, r in enumerate(ranked_runs, start=1):
        seed_label = str(r.seed) if r.seed is not None else "unknown"
        ranking_rows.append(
            f"""
            <tr>
              <td>{idx}</td>
              <td>{r.interest_score_adjusted:.2f}</td>
              <td>{r.interest_score:.2f}</td>
              <td>{r.similarity_best:.3f}</td>
              <td>{html.escape(r.category)}</td>
              <td>{html.escape(seed_label)}</td>
              <td>{html.escape(r.run_name)}</td>
              <td>{r.interaction_count}</td>
              <td>{'yes' if r.can_accept else 'no'}</td>
              <td>{'yes' if r.duplicate_of else 'no'}</td>
            </tr>
            """
        )

    similarity_rows = []
    for row in similarity_pairs[:100]:
        similarity_rows.append(
            f"""
            <tr>
              <td>{float(row.get('similarity', 0.0)):.3f}</td>
              <td>{html.escape(str(row.get('category_a', '')))} / {html.escape(str(row.get('seed_a', 'unknown')))}</td>
              <td>{html.escape(str(row.get('run_a', '')))}</td>
              <td>{html.escape(str(row.get('category_b', '')))} / {html.escape(str(row.get('seed_b', 'unknown')))}</td>
              <td>{html.escape(str(row.get('run_b', '')))}</td>
            </tr>
            """
        )

    run_cards = []
    runs_sorted = sorted(
        runs,
        key=lambda r: (
            r.category,
            r.seed if r.seed is not None else 10**9,
            r.seed_redo_index,
            r.timestamp,
            r.run_name,
        ),
    )
    for r in runs_sorted:
        err_count = sum(1 for i in r.issues if i.severity == "error")
        warn_count = sum(1 for i in r.issues if i.severity == "warning")
        info_count = sum(1 for i in r.issues if i.severity == "info")
        seed_label = str(r.seed) if r.seed is not None else "unknown"
        issue_lines = []
        if r.issues:
            for i in r.issues:
                sug = f" Suggestion: {i.suggestion}" if i.suggestion else ""
                issue_lines.append(
                    f"<li><span class='badge {_badge_class(i.severity)}'>{i.severity.upper()}</span> "
                    f"<code>{html.escape(i.rule)}</code> {html.escape(i.message)}{html.escape(sug)}</li>"
                )
        else:
            issue_lines.append("<li><span class='badge badge-pass'>PASS</span> No issues detected.</li>")

        vehicles = sorted(r.spec.get("ego_vehicles", []), key=lambda v: _safe_int_vehicle_id(str(v.get("vehicle_id", ""))))
        constraints = r.spec.get("vehicle_constraints", [])
        actors = r.spec.get("actors", [])
        vehicle_lines = "".join(f"<li>{html.escape(_vehicle_sentence(v))}</li>" for v in vehicles)
        constraint_lines = "".join(
            f"<li>{html.escape(_constraint_phrase(str(c.get('type', 'unknown')), str(c.get('a', '?')), str(c.get('b', '?'))))}</li>"
            for c in constraints
        ) or "<li>No constraints</li>"
        actor_lines = "".join(f"<li>{html.escape(_actor_sentence(a))}</li>" for a in actors) or "<li>No actors</li>"
        det_lines = "".join(f"<li>{html.escape(x)}</li>" for x in r.deterministic_adjustments) or "<li>None</li>"

        stage_chip_lines: List[str] = []
        stage_detail_rows: List[str] = []
        stage_status_counts = Counter()
        for st in r.stage_trace:
            status = str(st.get("status", "unknown"))
            stage_status_counts[status] += 1
            cls = {
                "ok": "stage-ok",
                "failed": "stage-failed",
                "missing": "stage-missing",
                "not_requested": "stage-skip",
                "present_untracked": "stage-warn",
            }.get(status, "stage-warn")
            elapsed = _to_float(st.get("elapsed_s"))
            elapsed_text = f"{elapsed:.1f}s" if elapsed is not None else "-"
            stage_chip_lines.append(
                f"<span class='stage-chip {cls}' title='{html.escape(status)}'>{html.escape(str(st.get('label', st.get('stage'))))} {elapsed_text}</span>"
            )
            stage_detail_rows.append(
                "<tr>"
                f"<td>{html.escape(str(st.get('label', st.get('stage'))))}</td>"
                f"<td>{html.escape(status)}</td>"
                f"<td>{elapsed_text}</td>"
                f"<td>{html.escape(_format_stage_metrics(st.get('metrics', {})) or '-')}</td>"
                "</tr>"
            )

        map_b64 = _b64_json_payload(r.map_layers)
        map_meta = r.map_layers.get("meta", {}) if isinstance(r.map_layers, dict) else {}
        map_meta_line = (
            f"legal={map_meta.get('legal_count', 0)} "
            f"picked={map_meta.get('picked_count', 0)} "
            f"refined={map_meta.get('refined_count', 0)} "
            f"actors={map_meta.get('actors_count', 0)} "
            f"actor_traj={map_meta.get('actor_traj_count', 0)} "
            f"ego_spawns={map_meta.get('ego_spawn_count', 0)}"
        )
        val_score_txt = f"{r.validation_score:.3f}" if isinstance(r.validation_score, (int, float)) else "n/a"
        interest_txt = f"{r.interest_score_adjusted:.2f}"
        interest_raw_txt = f"{r.interest_score:.2f}"
        similarity_txt = f"{r.similarity_best:.3f}"
        similarity_peer_txt = str(r.similarity_peer or "n/a")
        duplicate_of_txt = Path(str(r.duplicate_of)).name if r.duplicate_of else "n/a"
        timeline_search = " ".join(str(st.get("status", "")) for st in r.stage_trace)
        routes_stage = next((s for s in r.stage_trace if s.get("stage") == "routes"), None)
        route_metrics = routes_stage.get("metrics", {}) if isinstance(routes_stage, dict) else {}
        route_files = route_metrics.get("route_files", []) if isinstance(route_metrics, dict) else []
        if not isinstance(route_files, list):
            route_files = []
        route_files = [str(x) for x in route_files if isinstance(x, str)]
        route_files_b64 = _b64_json_payload(route_files)
        route_dir = str(route_metrics.get("routes_dir", str(Path(r.run_dir) / "09_routes" / "routes")))
        carla_stage = next((s for s in r.stage_trace if s.get("stage") == "carla_validation"), None)
        carla_metrics = carla_stage.get("metrics", {}) if isinstance(carla_stage, dict) else {}
        if not isinstance(carla_metrics, dict):
            carla_metrics = {}
        carla_pass = bool(r.carla_validation.get("passed", False))
        carla_reason = str(r.carla_validation.get("failure_reason") or "")
        carla_stage_status = str(carla_stage.get("status")) if isinstance(carla_stage, dict) else "missing"
        carla_gate_label = "PASS" if carla_pass else "FAIL"
        carla_gate_cls = "badge-pass" if carla_pass else "badge-error"
        can_accept_label = "yes" if r.can_accept else "no"
        carla_validation_b64 = _b64_json_payload(r.carla_validation)
        carla_metric_line = (
            f"spawn={carla_metrics.get('spawn_actual', 'n/a')}/{carla_metrics.get('spawn_expected', 'n/a')} "
            f"min_ttc={carla_metrics.get('min_ttc_s', 'n/a')} "
            f"rc_min={carla_metrics.get('route_completion_min', 'n/a')} "
            f"ds_min={carla_metrics.get('driving_score_min', 'n/a')} "
            f"repairs={carla_metrics.get('repairs_count', 0)}"
        )
        decision_search = " ".join(route_files + [carla_reason, carla_stage_status]).lower()

        run_cards.append(
            f"""
            <article class="run-card" data-category="{html.escape(r.category)}" data-run="{html.escape(r.run_name)}" data-status="{r.status}" data-seed="{html.escape(seed_label)}" data-run-dir="{html.escape(r.run_dir)}" data-route-dir="{html.escape(route_dir)}" data-route-files-b64="{route_files_b64}" data-search="{html.escape((r.category + ' ' + r.run_name + ' ' + r.compact_summary + ' seed ' + seed_label + ' ' + timeline_search + ' ' + decision_search).lower())}" data-can-accept="{can_accept_label}" data-carla-pass="{'1' if carla_pass else '0'}" data-carla-validation-b64="{carla_validation_b64}" data-interest-score="{interest_txt}" data-interest-score-raw="{interest_raw_txt}" data-similarity-best="{similarity_txt}">
              <header class="run-head">
                <div>
                  <h3>{html.escape(r.category)} <span class="redo">seed {html.escape(seed_label)} | redo #{r.redo_index} | seed-redo #{r.seed_redo_index}</span></h3>
                  <p class="meta">{html.escape(r.run_name)} | ts={html.escape(r.timestamp)} | score={r.score}</p>
                  <p class="meta">{html.escape(r.compact_summary)} | validation={val_score_txt} | interest={interest_txt} (raw={interest_raw_txt})</p>
                  <p class="meta">similarity_best={similarity_txt} peer={html.escape(similarity_peer_txt)} duplicate_of={html.escape(duplicate_of_txt)}</p>
                </div>
                <div class="metrics">
                  <span class="badge {'badge-pass' if r.status == 'pass' else 'badge-error'}">{r.status.upper()}</span>
                  <span class="chip">seed={html.escape(seed_label)}</span>
                  <span class="chip">E={err_count}</span>
                  <span class="chip">W={warn_count}</span>
                  <span class="chip">I={info_count}</span>
                  <span class="chip">V={r.ego_count}</span>
                  <span class="chip">C={r.constraint_count}</span>
                  <span class="chip">A={r.actor_count}</span>
                  <span class="chip">interactions={r.interaction_count}</span>
                  <span class="chip">interest={interest_txt}</span>
                  <span class="chip">sim={similarity_txt}</span>
                  <span class="chip">can_accept={can_accept_label}</span>
                  <span class="chip">stages ok={stage_status_counts.get('ok', 0)}</span>
                  <span class="chip decision-chip decision-pending" data-decision-chip>Review: pending</span>
                </div>
              </header>
              <section>
                <h4>Stage Timeline</h4>
                <div class="stage-line">{''.join(stage_chip_lines)}</div>
              </section>
              <section>
                <h4>Final CARLA Validation</h4>
                <p class="meta"><span class="badge {carla_gate_cls}">{carla_gate_label}</span> stage={html.escape(carla_stage_status)} can_accept={can_accept_label}</p>
                <p class="meta">{html.escape(carla_metric_line)}</p>
                <p class="meta">{html.escape(carla_reason) if carla_reason else "failure_reason=n/a"}</p>
              </section>
              <section class="map-block">
                <div class="map-head">
                  <h4>Unified Map Layers</h4>
                  <p class="meta">Toggle paths, actor trajectories, ego spawns, labels, and heading arrows ({html.escape(map_meta_line)}).</p>
                </div>
                <div class="map-controls">
                  <label><input type="checkbox" data-layer="legal" checked /> Legal</label>
                  <label><input type="checkbox" data-layer="picked" checked /> Picked</label>
                  <label><input type="checkbox" data-layer="refined" checked /> Refined</label>
                  <label><input type="checkbox" data-layer="actors" checked /> Actors</label>
                  <label><input type="checkbox" data-layer="traj" checked /> Actor Trajectory</label>
                  <label><input type="checkbox" data-layer="spawns" checked /> Ego Spawns</label>
                  <label><input type="checkbox" data-layer="labels" checked /> Labels</label>
                  <label><input type="checkbox" data-layer="arrows" checked /> Direction</label>
                </div>
                <svg class="map-svg" data-map-b64="{map_b64}" viewBox="0 0 1000 480" preserveAspectRatio="xMidYMid meet"></svg>
              </section>
              <section class="summary-block">
                <h4>Natural Language Scenario Summary</h4>
                <pre>{html.escape(r.natural_summary)}</pre>
              </section>
              <section class="cols">
                <div>
                  <h4>Vehicle Actions</h4>
                  <ul>{vehicle_lines}</ul>
                </div>
                <div>
                  <h4>Interaction Constraints</h4>
                  <ul>{constraint_lines}</ul>
                </div>
                <div>
                  <h4>Actors</h4>
                  <ul>{actor_lines}</ul>
                </div>
              </section>
              <section>
                <h4>Detected Issues</h4>
                <ul>{''.join(issue_lines)}</ul>
              </section>
              <section>
                <h4>Deterministic Explicitization Adjustments</h4>
                <ul>{det_lines}</ul>
              </section>
              <details>
                <summary>Stage Metrics</summary>
                <table>
                  <thead><tr><th>Stage</th><th>Status</th><th>Elapsed</th><th>Metrics</th></tr></thead>
                  <tbody>{''.join(stage_detail_rows)}</tbody>
                </table>
              </details>
              <details>
                <summary>Raw Schema JSON</summary>
                <pre>{_escape_json(r.spec)}</pre>
              </details>
              <details>
                <summary>Paths</summary>
                <pre>{html.escape(r.run_dir)}</pre>
              </details>
            </article>
            """
        )

    category_options = "".join(f"<option value='{html.escape(c)}'>{html.escape(c)}</option>" for c in categories)
    seed_options = "".join(f"<option value='{s}'>Seed {s}</option>" for s in seed_values)
    if unknown_seed_count:
        seed_options += "<option value='unknown'>Seed Unknown</option>"
    run_options = "".join(
        f"<option value='{html.escape(r.run_name)}'>{html.escape(r.run_name)}</option>"
        for r in runs_sorted
    )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f5f7f4;
      --panel: #ffffff;
      --ink: #1d2b24;
      --muted: #5f6f67;
      --line: #d5ddd8;
      --accent: #0f766e;
      --danger: #b91c1c;
      --warn: #b45309;
      --info: #0c4a6e;
      --ok: #166534;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: radial-gradient(circle at 20% 0%, #eef8f2 0, #f5f7f4 40%) fixed;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    }}
    header.page {{
      padding: 20px 24px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(90deg, #e9f3ef, #f8fbf9);
    }}
    h1 {{ margin: 0 0 8px 0; font-size: 24px; }}
    .sub {{ color: var(--muted); margin: 0; }}
    .wrap {{ padding: 18px 24px 40px; max-width: 1800px; margin: 0 auto; }}
    .cards {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      margin-bottom: 16px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 12px;
    }}
    .card .k {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .card .v {{ font-size: 24px; font-weight: 700; margin-top: 6px; }}
    .toolbar {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }}
    .toolbar input, .toolbar select {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      background: white;
      color: var(--ink);
    }}
    .toolbar button {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 12px;
      background: #eef7f3;
      color: #12463f;
      font-weight: 600;
      cursor: pointer;
    }}
    .toolbar .counter {{
      margin-left: auto;
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      background: #f5faf7;
      color: #33584c;
      font-weight: 600;
    }}
    .mode-tabs {{
      display: flex;
      gap: 8px;
      margin-bottom: 10px;
    }}
    .mode-tabs .tab-btn {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 12px;
      background: #f3f8f5;
      color: #183c31;
      font-weight: 700;
      cursor: pointer;
    }}
    .mode-tabs .tab-btn.active {{
      background: #dff1e8;
      border-color: #a9cfbc;
      color: #0f5132;
    }}
    .review-panel {{
      position: sticky;
      top: 0;
      z-index: 30;
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 10px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: rgba(250, 254, 252, 0.96);
      backdrop-filter: blur(4px);
    }}
    .review-panel button {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      background: #ffffff;
      color: #183c31;
      cursor: pointer;
      font-weight: 600;
    }}
    .review-panel .btn-accept {{
      background: #e6f6ea;
      border-color: #b6dfc0;
      color: #14532d;
    }}
    .review-panel .btn-reject {{
      background: #fdecec;
      border-color: #f4b4b4;
      color: #991b1b;
    }}
    .review-panel .btn-export {{
      background: #eaf3ff;
      border-color: #c8ddfb;
      color: #1e3a8a;
    }}
    .review-panel input {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      min-width: 300px;
      flex: 1 1 320px;
      background: #ffffff;
      color: var(--ink);
    }}
    .review-panel input.needs-input {{
      border-color: #b91c1c;
      box-shadow: 0 0 0 2px rgba(185, 28, 28, 0.15);
    }}
    .review-status {{
      font-size: 12px;
      color: #355047;
      font-weight: 600;
      padding: 4px 8px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #f7faf8;
    }}
    body.review-active #analyticsSection {{
      display: none;
    }}
    body.review-active .wrap {{
      max-width: 1400px;
    }}
    body.review-active {{
      overflow: hidden;
    }}
    body.review-active .run-list {{
      max-height: calc(100vh - 170px);
      overflow: hidden;
    }}
    body.review-active .run-list .run-card {{
      display: none;
    }}
    body.review-active .run-list .run-card.review-active-card {{
      display: block;
      border-width: 2px;
      border-color: #b5d7c7;
      box-shadow: 0 8px 24px rgba(21, 64, 52, 0.08);
      height: calc(100vh - 190px);
      overflow: auto;
      margin: 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
      margin-bottom: 16px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }}
    th {{ background: #eff4f1; font-size: 12px; text-transform: uppercase; color: #4a5b53; }}
    .run-list {{ display: grid; gap: 12px; }}
    .run-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 12px;
    }}
    .run-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      border-bottom: 1px dashed var(--line);
      padding-bottom: 8px;
      margin-bottom: 10px;
    }}
    .run-head h3 {{ margin: 0; font-size: 18px; }}
    .redo {{ color: var(--muted); font-size: 14px; font-weight: 500; }}
    .meta {{ margin: 2px 0; color: var(--muted); font-size: 13px; }}
    .metrics {{ display: flex; gap: 6px; align-items: center; flex-wrap: wrap; justify-content: flex-end; }}
    .stage-line {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }}
    .stage-chip {{
      display: inline-block;
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 11px;
      border: 1px solid var(--line);
      background: #f8faf9;
      color: #374b43;
      white-space: nowrap;
    }}
    .stage-ok {{ background: #e9f8ef; border-color: #b8e2c6; color: #14532d; }}
    .stage-failed {{ background: #fdecec; border-color: #f5b9b9; color: #991b1b; }}
    .stage-missing {{ background: #fff4e5; border-color: #f3d0a0; color: #92400e; }}
    .stage-skip {{ background: #f3f5f4; border-color: #dfe5e2; color: #66756f; }}
    .stage-warn {{ background: #eef6fb; border-color: #cfe3f0; color: #0c4a6e; }}
    .map-block {{
      margin: 10px 0;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fbfdfc;
    }}
    .map-head {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .map-controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      font-size: 12px;
      color: #355047;
      margin: 6px 0 8px;
    }}
    .map-controls label {{ user-select: none; }}
    .map-svg {{
      width: 100%;
      min-height: 320px;
      border: 1px solid #dce5e0;
      border-radius: 8px;
      background: linear-gradient(180deg, #f8fcfa, #f0f6f3);
    }}
    .chip {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      background: #f7faf8;
    }}
    .decision-chip {{
      font-weight: 700;
    }}
    .decision-pending {{
      background: #f3f4f6;
      color: #374151;
      border-color: #d1d5db;
    }}
    .decision-accepted {{
      background: #e6f6ea;
      color: #14532d;
      border-color: #b6dfc0;
    }}
    .decision-rejected {{
      background: #fdecec;
      color: #991b1b;
      border-color: #f4b4b4;
    }}
    .badge {{
      display: inline-block;
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 11px;
      color: white;
      font-weight: 700;
      letter-spacing: .02em;
      vertical-align: middle;
    }}
    .badge-error {{ background: var(--danger); }}
    .badge-warning {{ background: var(--warn); }}
    .badge-info {{ background: var(--info); }}
    .badge-pass {{ background: var(--ok); }}
    .summary-block pre {{
      white-space: pre-wrap;
      background: #f4f8f6;
      border: 1px solid #d9e3de;
      border-radius: 8px;
      padding: 10px;
      margin: 0;
      font-family: "IBM Plex Mono", "Consolas", monospace;
      font-size: 12px;
      line-height: 1.45;
    }}
    .cols {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 10px;
      margin-top: 10px;
      margin-bottom: 10px;
    }}
    h4 {{ margin: 8px 0; font-size: 14px; }}
    ul {{ margin: 0; padding-left: 18px; }}
    li {{ margin: 4px 0; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; }}
    details {{
      margin-top: 8px;
      background: #fafcfb;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
    }}
    summary {{ cursor: pointer; font-weight: 600; }}
    .section-title {{ margin: 16px 0 8px; }}
    @media (max-width: 900px) {{
      .run-head {{ flex-direction: column; }}
      .metrics {{ justify-content: flex-start; }}
      .review-panel {{ top: 0; }}
      .review-panel input {{ min-width: 100%; flex-basis: 100%; }}
      .toolbar .counter {{ width: 100%; margin-left: 0; }}
    }}
  </style>
</head>
<body>
  <header class="page">
    <h1>{html.escape(title)}</h1>
    <p class="sub">Multi-run pipeline dashboard with schema reasoning, stage-by-stage anomaly detection, unified map-layer exploration, and manual final-pass route review.</p>
  </header>
  <main class="wrap">
    <section id="analyticsSection">
    <section class="cards">
      <div class="card"><div class="k">Schemas</div><div class="v">{total}</div></div>
      <div class="card"><div class="k">Categories</div><div class="v">{len(categories)}</div></div>
      <div class="card"><div class="k">Distinct Seeds</div><div class="v">{len(seed_values)}{f" (+{unknown_seed_count} unknown)" if unknown_seed_count else ""}</div></div>
      <div class="card"><div class="k">Full 01→10 Runs</div><div class="v">{full_run_count}</div></div>
      <div class="card"><div class="k">CARLA Gate-Pass</div><div class="v">{can_accept_count}</div></div>
      <div class="card"><div class="k">Avg Validation</div><div class="v">{avg_validation if avg_validation is not None else "n/a"}</div></div>
      <div class="card"><div class="k">Avg Interest</div><div class="v">{avg_interest if avg_interest is not None else "n/a"}</div></div>
      <div class="card"><div class="k">Avg Interest (Deduped)</div><div class="v">{avg_interest_adjusted if avg_interest_adjusted is not None else "n/a"}</div></div>
      <div class="card"><div class="k">Unique Route+Actor Shapes</div><div class="v">{unique_route_actor_fp_count}</div></div>
      <div class="card"><div class="k">Exact Duplicates</div><div class="v">{duplicate_count}</div></div>
      <div class="card"><div class="k">Pass / Fail</div><div class="v">{pass_count} / {fail_count}</div></div>
      <div class="card"><div class="k">Average Score</div><div class="v">{avg_score}</div></div>
      <div class="card"><div class="k">Errors</div><div class="v">{total_errors}</div></div>
      <div class="card"><div class="k">Warnings</div><div class="v">{total_warnings}</div></div>
      <div class="card"><div class="k">Infos</div><div class="v">{total_infos}</div></div>
    </section>

    <h2 class="section-title">Category Health</h2>
    <table>
      <thead>
        <tr><th>Category</th><th>Runs</th><th>Pass Rate</th><th>Avg Score</th><th>Errors</th><th>Warnings</th><th>Info</th><th>Unique Schemas</th><th>Seeds</th></tr>
      </thead>
      <tbody>
        {''.join(category_rows)}
      </tbody>
    </table>

    <h2 class="section-title">Category × Seed Health</h2>
    <table>
      <thead>
        <tr><th>Category</th><th>Seed</th><th>Runs</th><th>Pass Rate</th><th>Avg Score</th><th>Unique Schemas</th></tr>
      </thead>
      <tbody>
        {''.join(seed_rows) if seed_rows else "<tr><td colspan='6'>No seed-tagged runs</td></tr>"}
      </tbody>
    </table>

    <h2 class="section-title">Stage 02→10 Health</h2>
    <table>
      <thead>
        <tr><th>Stage</th><th>Requested Runs</th><th>Success</th><th>Failed</th><th>Missing</th><th>Avg Elapsed</th><th>Related Findings</th></tr>
      </thead>
      <tbody>
        {''.join(stage_rows) if stage_rows else "<tr><td colspan='7'>No stage data</td></tr>"}
      </tbody>
    </table>

    <h2 class="section-title">Issue Taxonomy</h2>
    <table>
      <thead><tr><th>Severity</th><th>Rule</th><th>Count</th></tr></thead>
      <tbody>{''.join(issue_rows) if issue_rows else "<tr><td colspan='3'>No issues</td></tr>"}</tbody>
    </table>

    <h2 class="section-title">Interestingness Ranking</h2>
    <table>
      <thead>
        <tr><th>Rank</th><th>Interest (Adj)</th><th>Interest (Raw)</th><th>Best Similarity</th><th>Category</th><th>Seed</th><th>Run</th><th>Interactions</th><th>CARLA Gate</th><th>Duplicate</th></tr>
      </thead>
      <tbody>
        {''.join(ranking_rows) if ranking_rows else "<tr><td colspan='10'>No runs</td></tr>"}
      </tbody>
    </table>

    <h2 class="section-title">Similarity Pairs</h2>
    <table>
      <thead>
        <tr><th>Similarity</th><th>Scenario A</th><th>Run A</th><th>Scenario B</th><th>Run B</th></tr>
      </thead>
      <tbody>
        {''.join(similarity_rows) if similarity_rows else "<tr><td colspan='5'>No high-similarity pairs</td></tr>"}
      </tbody>
    </table>
    </section>

    <h2 class="section-title">Run Explorer</h2>
    <div class="mode-tabs">
      <button id="analyticsTabBtn" class="tab-btn active" type="button">Analytics Tab</button>
      <button id="reviewTabBtn" class="tab-btn" type="button">Accept/Reject Tab</button>
    </div>
    <div class="toolbar">
      <select id="categoryFilter">
        <option value="all">All Categories</option>
        {category_options}
      </select>
      <select id="seedFilter">
        <option value="all">All Seeds</option>
        {seed_options}
      </select>
      <select id="runFilter">
        <option value="all">All Runs</option>
        {run_options}
      </select>
      <select id="statusFilter">
        <option value="all">All Status</option>
        <option value="pass">Pass</option>
        <option value="fail">Fail</option>
      </select>
      <select id="decisionFilter">
        <option value="all">All Decisions</option>
        <option value="pending">Decision Pending</option>
        <option value="accepted">Accepted</option>
        <option value="rejected">Rejected</option>
      </select>
      <select id="carlaGateFilter">
        <option value="all">CARLA Gate: All</option>
        <option value="pass">CARLA Gate: Pass</option>
        <option value="fail">CARLA Gate: Fail</option>
        <option value="can_accept">Eligible To Accept</option>
      </select>
      <input id="searchBox" type="text" placeholder="Search category, run, summary..." />
      <span id="visibleCounter" class="counter">Visible: 0</span>
    </div>
    <div id="reviewPanel" class="review-panel">
      <button id="prevCardBtn" type="button">Prev (←)</button>
      <button id="nextCardBtn" type="button">Next (→)</button>
      <span id="reviewProgress" class="review-status">0 / 0</span>
      <button id="acceptBtn" class="btn-accept" type="button">Accept (A)</button>
      <input id="rejectReasonInput" type="text" placeholder="Optional rejection reason" />
      <button id="rejectBtn" class="btn-reject" type="button">Reject (R)</button>
      <button id="clearDecisionBtn" type="button">Clear (C)</button>
      <span id="acceptGateStatus" class="review-status">Gate: n/a</span>
      <span id="decisionSummary" class="review-status">Accepted: 0 | Rejected: 0 | Pending: 0</span>
      <button id="exportDecisionsBtn" class="btn-export" type="button">Download decisions JSON</button>
      <button id="exportFinalZipBtn" class="btn-export" type="button">Download final review ZIP</button>
    </div>
    <section id="runList" class="run-list">
      {''.join(run_cards)}
    </section>
  </main>
  <script>
    const categoryFilter = document.getElementById('categoryFilter');
    const seedFilter = document.getElementById('seedFilter');
    const runFilter = document.getElementById('runFilter');
    const statusFilter = document.getElementById('statusFilter');
    const decisionFilter = document.getElementById('decisionFilter');
    const carlaGateFilter = document.getElementById('carlaGateFilter');
    const searchBox = document.getElementById('searchBox');
    const analyticsTabBtn = document.getElementById('analyticsTabBtn');
    const reviewTabBtn = document.getElementById('reviewTabBtn');
    const visibleCounter = document.getElementById('visibleCounter');
    const reviewPanel = document.getElementById('reviewPanel');
    const reviewProgress = document.getElementById('reviewProgress');
    const acceptBtn = document.getElementById('acceptBtn');
    const rejectBtn = document.getElementById('rejectBtn');
    const clearDecisionBtn = document.getElementById('clearDecisionBtn');
    const prevCardBtn = document.getElementById('prevCardBtn');
    const nextCardBtn = document.getElementById('nextCardBtn');
    const rejectReasonInput = document.getElementById('rejectReasonInput');
    const acceptGateStatus = document.getElementById('acceptGateStatus');
    const decisionSummary = document.getElementById('decisionSummary');
    const exportDecisionsBtn = document.getElementById('exportDecisionsBtn');
    const exportFinalZipBtn = document.getElementById('exportFinalZipBtn');
    const runList = document.getElementById('runList');
    const cards = Array.from(document.querySelectorAll('.run-card'));

    const decisions = new Map();
    const decisionStorageKey = `schema_dashboard_review_decisions::${{location.pathname}}`;
    let reviewMode = false;
    let reviewIndex = 0;

    function decisionKey(card) {{
      return card.dataset.runDir || card.dataset.run || '';
    }}

    function saveDecisionsToStorage() {{
      try {{
        const obj = Object.fromEntries(decisions.entries());
        localStorage.setItem(decisionStorageKey, JSON.stringify(obj));
      }} catch (e) {{
        // non-fatal
      }}
    }}

    function loadDecisionsFromStorage() {{
      try {{
        const raw = localStorage.getItem(decisionStorageKey);
        if (!raw) return;
        const parsed = JSON.parse(raw);
        if (!parsed || typeof parsed !== 'object') return;
        for (const [k, v] of Object.entries(parsed)) {{
          if (!k || !v || typeof v !== 'object') continue;
          const decision = String(v.decision || '');
          if (decision !== 'accepted' && decision !== 'rejected') continue;
          decisions.set(k, {{
            decision,
            reason: String(v.reason || ''),
            reviewed_at: String(v.reviewed_at || ''),
          }});
        }}
      }} catch (e) {{
        // non-fatal
      }}
    }}

    function getDecision(card) {{
      return decisions.get(decisionKey(card)) || null;
    }}

    function parseRouteFiles(card) {{
      const b64 = card.dataset.routeFilesB64 || '';
      if (!b64) return [];
      try {{
        const parsed = JSON.parse(atob(b64));
        return Array.isArray(parsed) ? parsed.filter((x) => typeof x === 'string') : [];
      }} catch (e) {{
        return [];
      }}
    }}

    function parseCarlaValidation(card) {{
      const b64 = card.dataset.carlaValidationB64 || '';
      if (!b64) return {{}};
      try {{
        const parsed = JSON.parse(atob(b64));
        return parsed && typeof parsed === 'object' ? parsed : {{}};
      }} catch (e) {{
        return {{}};
      }}
    }}

    function canAcceptCard(card) {{
      return String(card.dataset.canAccept || 'no').toLowerCase() === 'yes';
    }}

    function applyCarlaGateFilter(card) {{
      const filterVal = carlaGateFilter.value || 'all';
      if (filterVal === 'all') return true;
      const pass = (card.dataset.carlaPass || '0') === '1';
      const eligible = canAcceptCard(card);
      if (filterVal === 'pass') return pass;
      if (filterVal === 'fail') return !pass;
      if (filterVal === 'can_accept') return eligible;
      return true;
    }}

    function applyDecisionFilter(card) {{
      const filterVal = decisionFilter.value || 'all';
      if (filterVal === 'all') return true;
      const d = getDecision(card);
      const state = d ? d.decision : 'pending';
      return state === filterVal;
    }}

    function updateDecisionChip(card) {{
      const chip = card.querySelector('[data-decision-chip]');
      if (!chip) return;
      chip.classList.remove('decision-pending', 'decision-accepted', 'decision-rejected');
      const d = getDecision(card);
      if (!d) {{
        chip.classList.add('decision-pending');
        chip.textContent = 'Review: pending';
        return;
      }}
      if (d.decision === 'accepted') {{
        chip.classList.add('decision-accepted');
        chip.textContent = 'Review: accepted';
        return;
      }}
      chip.classList.add('decision-rejected');
      chip.textContent = d.reason ? `Review: rejected (${{d.reason}})` : 'Review: rejected';
    }}

    function updateDecisionSummary() {{
      const filtered = cards.filter((c) => c.dataset.filterMatch === '1');
      let accepted = 0;
      let rejected = 0;
      for (const c of filtered) {{
        const d = getDecision(c);
        if (!d) continue;
        if (d.decision === 'accepted') accepted += 1;
        if (d.decision === 'rejected') rejected += 1;
      }}
      const pending = Math.max(0, filtered.length - accepted - rejected);
      decisionSummary.textContent = `Accepted: ${{accepted}} | Rejected: ${{rejected}} | Pending: ${{pending}}`;
    }}

    function filteredCards() {{
      return cards.filter((c) => c.dataset.filterMatch === '1');
    }}

    function autoRejectFailedValidation() {{
      let changed = false;
      const nowIso = new Date().toISOString();
      for (const card of cards) {{
        if (canAcceptCard(card)) continue;
        const key = decisionKey(card);
        if (!key) continue;
        const d = getDecision(card);
        if (d && d.decision === 'rejected') continue;
        decisions.set(key, {{
          decision: 'rejected',
          reason: d && d.reason ? String(d.reason) : 'failed validation (auto)',
          reviewed_at: d && d.reviewed_at ? String(d.reviewed_at) : nowIso,
        }});
        changed = true;
      }}
      if (changed) {{
        saveDecisionsToStorage();
      }}
      return changed;
    }}

    function currentReviewCard() {{
      const visible = filteredCards();
      if (!visible.length) return null;
      reviewIndex = Math.max(0, Math.min(reviewIndex, visible.length - 1));
      return visible[reviewIndex];
    }}

    function syncReviewCardView(scrollToCard) {{
      const visible = filteredCards();
      for (const c of cards) {{
        c.classList.remove('review-active-card');
        if (!reviewMode) {{
          c.style.display = c.dataset.filterMatch === '1' ? '' : 'none';
          continue;
        }}
        c.style.display = 'none';
      }}

      if (!reviewMode) {{
        reviewProgress.textContent = `${{visible.length}} visible`;
        rejectReasonInput.value = '';
        acceptBtn.disabled = false;
        acceptGateStatus.textContent = 'Gate: n/a';
        return;
      }}

      if (!visible.length) {{
        reviewProgress.textContent = '0 / 0';
        rejectReasonInput.value = '';
        acceptBtn.disabled = true;
        acceptGateStatus.textContent = 'Gate: n/a';
        return;
      }}

      const active = currentReviewCard();
      if (!active) {{
        reviewProgress.textContent = '0 / 0';
        acceptBtn.disabled = true;
        acceptGateStatus.textContent = 'Gate: n/a';
        return;
      }}
      active.style.display = '';
      active.classList.add('review-active-card');
      reviewProgress.textContent = `${{reviewIndex + 1}} / ${{visible.length}}`;

      const d = getDecision(active);
      rejectReasonInput.value = d && d.decision === 'rejected' ? (d.reason || '') : '';
      const cv = parseCarlaValidation(active);
      const eligible = canAcceptCard(active);
      const stageStatus = String(cv.stage_status || 'missing');
      const pass = !!cv.passed;
      const reason = String(cv.failure_reason || '').trim();
      const acceptanceLevel = String(cv.acceptance_level || '').toLowerCase();
      acceptBtn.disabled = !eligible;
      if (eligible) {{
        if (acceptanceLevel === 'medium') {{
          const why = String((cv.target_acceptance && cv.target_acceptance.reason) || reason || '').trim();
          acceptGateStatus.textContent = why
            ? `Gate: MEDIUM (manual review, reason=${{why}})`
            : `Gate: MEDIUM (manual review)`;
        }} else {{
          acceptGateStatus.textContent = `Gate: PASS (stage=${{stageStatus}})`;
        }}
      }} else {{
        acceptGateStatus.textContent = reason
          ? `Gate: BLOCKED (${{reason}})`
          : `Gate: BLOCKED (stage=${{stageStatus}}, pass=${{pass ? 'yes' : 'no'}})`;
      }}
      if (scrollToCard) {{
        // keep tinder-style single-card flow without list scrolling
      }}
    }}

    function applyFilters() {{
      const autoRejectedChanged = autoRejectFailedValidation();
      if (autoRejectedChanged) {{
        for (const c of cards) updateDecisionChip(c);
      }}
      const cat = categoryFilter.value;
      const seed = seedFilter.value;
      const runName = runFilter.value;
      const status = statusFilter.value;
      const q = searchBox.value.trim().toLowerCase();

      let visibleCount = 0;
      for (const c of cards) {{
        const catOk = cat === 'all' || c.dataset.category === cat;
        const seedOk = seed === 'all' || c.dataset.seed === seed;
        const runOk = runName === 'all' || c.dataset.run === runName;
        const statusOk = status === 'all' || c.dataset.status === status;
        const searchOk = q === '' || c.dataset.search.includes(q);
        const decisionOk = applyDecisionFilter(c);
        const carlaGateOk = applyCarlaGateFilter(c);
        const matched = catOk && seedOk && runOk && statusOk && searchOk && decisionOk && carlaGateOk;
        c.dataset.filterMatch = matched ? '1' : '0';
        if (matched) visibleCount += 1;
      }}

      visibleCounter.textContent = `Visible: ${{visibleCount}}`;
      updateDecisionSummary();
      if (!reviewMode) {{
        reviewIndex = 0;
      }} else {{
        const maxIdx = Math.max(0, visibleCount - 1);
        reviewIndex = Math.max(0, Math.min(reviewIndex, maxIdx));
      }}
      syncReviewCardView(false);
    }}

    function setReviewMode(enabled) {{
      reviewMode = !!enabled;
      document.body.classList.toggle('review-active', reviewMode);
      analyticsTabBtn.classList.toggle('active', !reviewMode);
      reviewTabBtn.classList.toggle('active', reviewMode);
      reviewPanel.style.display = reviewMode ? 'flex' : 'none';
      syncReviewCardView(reviewMode);
    }}

    function setDecision(card, decision, reason) {{
      if (!card) return false;
      if (decision === 'accepted' && !canAcceptCard(card)) {{
        const cv = parseCarlaValidation(card);
        const failReason = String(cv.failure_reason || '').trim();
        alert(failReason ? `Cannot accept: ${{failReason}}` : 'Cannot accept: CARLA hard gate is not passing.');
        return false;
      }}
      const key = decisionKey(card);
      if (!key) return false;
      decisions.set(key, {{
        decision,
        reason: String(reason || '').trim(),
        reviewed_at: new Date().toISOString(),
      }});
      saveDecisionsToStorage();
      updateDecisionChip(card);
      applyFilters();
      return true;
    }}

    function clearDecision(card) {{
      if (!card) return;
      const key = decisionKey(card);
      if (!key) return;
      decisions.delete(key);
      saveDecisionsToStorage();
      updateDecisionChip(card);
      applyFilters();
    }}

    function moveReview(delta) {{
      const visible = filteredCards();
      if (!visible.length) return;
      reviewIndex = Math.max(0, Math.min(reviewIndex + delta, visible.length - 1));
      syncReviewCardView(true);
    }}

    function buildDecisionRecord(card) {{
      const d = getDecision(card);
      const routeFiles = parseRouteFiles(card);
      const carlaValidation = parseCarlaValidation(card);
      return {{
        category: card.dataset.category || '',
        run_name: card.dataset.run || '',
        seed: card.dataset.seed || 'unknown',
        run_dir: card.dataset.runDir || '',
        route_dir: card.dataset.routeDir || '',
        route_files: routeFiles,
        pipeline_status: card.dataset.status || '',
        can_accept: canAcceptCard(card),
        carla_validation: carlaValidation,
        decision: d ? d.decision : 'pending',
        rejection_reason: d && d.decision === 'rejected' ? (d.reason || '') : '',
        reviewed_at: d ? (d.reviewed_at || null) : null,
      }};
    }}

    function exportDecisions() {{
      const visible = filteredCards();
      const accepted = [];
      const rejected = [];
      const pending = [];
      for (const card of visible) {{
        const rec = buildDecisionRecord(card);
        if (rec.decision === 'accepted') accepted.push(rec);
        else if (rec.decision === 'rejected') rejected.push(rec);
        else pending.push(rec);
      }}
      const payload = {{
        generated_at: new Date().toISOString(),
        title: document.title,
        filters: {{
          category: categoryFilter.value,
          seed: seedFilter.value,
          run: runFilter.value,
          status: statusFilter.value,
          decision: decisionFilter.value,
          carla_gate: carlaGateFilter.value,
          search: searchBox.value.trim(),
        }},
        counts: {{
          visible: visible.length,
          accepted: accepted.length,
          rejected: rejected.length,
          pending: pending.length,
        }},
        accepted_routes: accepted,
        rejected_routes: rejected,
        pending_routes: pending,
      }};
      const text = JSON.stringify(payload, null, 2);
      const blob = new Blob([text], {{ type: 'application/json' }});
      const url = URL.createObjectURL(blob);
      const ts = new Date().toISOString().replace(/[:.]/g, '-');
      const a = document.createElement('a');
      a.href = url;
      a.download = `manual_route_review_${{ts}}.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }}

    async function exportFinalReviewZip() {{
      const visible = filteredCards();
      const records = visible.map((card) => buildDecisionRecord(card));
      const accepted = records.filter((r) => r.decision === 'accepted');
      if (!accepted.length) {{
        alert('No accepted scenarios in current filtered view.');
        return;
      }}

      const payload = {{
        generated_at: new Date().toISOString(),
        title: document.title,
        filters: {{
          category: categoryFilter.value,
          seed: seedFilter.value,
          run: runFilter.value,
          status: statusFilter.value,
          decision: decisionFilter.value,
          carla_gate: carlaGateFilter.value,
          search: searchBox.value.trim(),
        }},
        records,
      }};

      try {{
        const resp = await fetch('http://127.0.0.1:8777/api/export-final-review', {{
          method: 'POST',
          headers: {{
            'Content-Type': 'application/json',
          }},
          body: JSON.stringify(payload),
        }});
        if (!resp.ok) {{
          let errText = `Export failed with HTTP ${{resp.status}}`;
          try {{
            const errJson = await resp.json();
            if (errJson && errJson.error) errText = String(errJson.error);
          }} catch (_e) {{
            // ignore parse error
          }}
          throw new Error(errText);
        }}
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const ts = new Date().toISOString().replace(/[:.]/g, '-');
        const a = document.createElement('a');
        a.href = url;
        a.download = `final_review_bundle_${{ts}}.zip`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      }} catch (err) {{
        const msg = err && err.message ? err.message : String(err);
        alert(
          `Final ZIP export failed: ${{msg}}\\n\\n` +
          `Start helper server:\\npython tools/final_review_export_server.py --host 127.0.0.1 --port 8777`
        );
      }}
    }}

    loadDecisionsFromStorage();
    for (const c of cards) {{
      updateDecisionChip(c);
    }}
    applyFilters();
    setReviewMode(false);

    categoryFilter.addEventListener('change', applyFilters);
    seedFilter.addEventListener('change', applyFilters);
    runFilter.addEventListener('change', applyFilters);
    statusFilter.addEventListener('change', applyFilters);
    decisionFilter.addEventListener('change', applyFilters);
    carlaGateFilter.addEventListener('change', applyFilters);
    searchBox.addEventListener('input', applyFilters);
    analyticsTabBtn.addEventListener('click', () => setReviewMode(false));
    reviewTabBtn.addEventListener('click', () => setReviewMode(true));
    prevCardBtn.addEventListener('click', () => moveReview(-1));
    nextCardBtn.addEventListener('click', () => moveReview(1));
    acceptBtn.addEventListener('click', () => {{
      const card = currentReviewCard();
      if (!card) return;
      const ok = setDecision(card, 'accepted', '');
      if (ok) moveReview(1);
    }});
    rejectBtn.addEventListener('click', () => {{
      const card = currentReviewCard();
      if (!card) return;
      const reason = rejectReasonInput.value.trim();
      const ok = setDecision(card, 'rejected', reason);
      if (ok) moveReview(1);
    }});
    clearDecisionBtn.addEventListener('click', () => {{
      const card = currentReviewCard();
      if (!card) return;
      clearDecision(card);
      rejectReasonInput.value = '';
    }});
    rejectReasonInput.addEventListener('input', () => {{
      const card = currentReviewCard();
      if (!card) return;
      const d = getDecision(card);
      if (d && d.decision === 'rejected') {{
        setDecision(card, 'rejected', rejectReasonInput.value.trim());
      }}
    }});
    exportDecisionsBtn.addEventListener('click', exportDecisions);
    exportFinalZipBtn.addEventListener('click', () => {{
      exportFinalReviewZip();
    }});

    document.addEventListener('keydown', (ev) => {{
      if (!reviewMode) return;
      const tag = String(document.activeElement && document.activeElement.tagName || '').toUpperCase();
      const typing = tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
      if (ev.key === 'Escape') {{
        setReviewMode(false);
        return;
      }}
      if (ev.key === 'ArrowRight' && !typing) {{
        ev.preventDefault();
        moveReview(1);
        return;
      }}
      if (ev.key === 'ArrowLeft' && !typing) {{
        ev.preventDefault();
        moveReview(-1);
        return;
      }}
      if ((ev.key === 'a' || ev.key === 'A') && !typing) {{
        ev.preventDefault();
        acceptBtn.click();
        return;
      }}
      if (ev.key === 'r' || ev.key === 'R') {{
        ev.preventDefault();
        rejectBtn.click();
        return;
      }}
      if ((ev.key === 'c' || ev.key === 'C') && !typing) {{
        ev.preventDefault();
        clearDecisionBtn.click();
      }}
    }});

    function parseMapData(svgEl) {{
      const b64 = svgEl.dataset.mapB64 || '';
      if (!b64) return null;
      try {{
        return JSON.parse(atob(b64));
      }} catch (e) {{
        return null;
      }}
    }}

    function collectAllPoints(data) {{
      const pts = [];
      const addPath = (arr) => {{
        if (!Array.isArray(arr)) return;
        for (const item of arr) {{
          const poly = item && item.polyline;
          if (!Array.isArray(poly)) continue;
          for (const p of poly) {{
            if (Array.isArray(p) && p.length >= 2) {{
              const x = Number(p[0]), y = Number(p[1]);
              if (Number.isFinite(x) && Number.isFinite(y)) pts.push([x, y]);
            }}
          }}
        }}
      }};
      addPath(data.legal_paths || []);
      addPath(data.picked_paths || []);
      addPath(data.refined_paths || []);
      for (const a of (data.actors || [])) {{
        const x = Number(a.x), y = Number(a.y);
        if (Number.isFinite(x) && Number.isFinite(y)) pts.push([x, y]);
        const traj = a && a.trajectory;
        if (Array.isArray(traj)) {{
          for (const p of traj) {{
            if (Array.isArray(p) && p.length >= 2) {{
              const tx = Number(p[0]), ty = Number(p[1]);
              if (Number.isFinite(tx) && Number.isFinite(ty)) pts.push([tx, ty]);
            }}
          }}
        }}
      }}
      for (const e of (data.ego_spawns || [])) {{
        const x = Number(e.x), y = Number(e.y);
        if (Number.isFinite(x) && Number.isFinite(y)) pts.push([x, y]);
      }}
      const c = data.crop;
      if (c && Number.isFinite(Number(c.xmin)) && Number.isFinite(Number(c.xmax)) && Number.isFinite(Number(c.ymin)) && Number.isFinite(Number(c.ymax))) {{
        pts.push([Number(c.xmin), Number(c.ymin)]);
        pts.push([Number(c.xmax), Number(c.ymax)]);
      }}
      return pts;
    }}

    function computeBounds(data) {{
      const pts = collectAllPoints(data);
      if (!pts.length) {{
        return {{xmin: 0, xmax: 100, ymin: 0, ymax: 100}};
      }}
      let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
      for (const [x, y] of pts) {{
        if (x < xmin) xmin = x;
        if (x > xmax) xmax = x;
        if (y < ymin) ymin = y;
        if (y > ymax) ymax = y;
      }}
      const dx = Math.max(1, xmax - xmin);
      const dy = Math.max(1, ymax - ymin);
      const padX = dx * 0.05;
      const padY = dy * 0.05;
      return {{xmin: xmin - padX, xmax: xmax + padX, ymin: ymin - padY, ymax: ymax + padY}};
    }}

    function projectPoint(x, y, bounds, width, height, pad) {{
      const availW = Math.max(1, width - pad * 2);
      const availH = Math.max(1, height - pad * 2);
      const sx = availW / Math.max(1e-6, (bounds.xmax - bounds.xmin));
      const sy = availH / Math.max(1e-6, (bounds.ymax - bounds.ymin));
      const s = Math.min(sx, sy);
      const usedW = (bounds.xmax - bounds.xmin) * s;
      const usedH = (bounds.ymax - bounds.ymin) * s;
      const ox = (width - usedW) / 2;
      const oy = (height - usedH) / 2;
      // Invert Y for map-like orientation
      const px = ox + (x - bounds.xmin) * s;
      const py = oy + (bounds.ymax - y) * s;
      return [px, py];
    }}

    function drawMap(svgEl, data, state) {{
      const width = 1000;
      const height = 480;
      const pad = 18;
      svgEl.innerHTML = '';
      if (!data) {{
        return;
      }}
      const bounds = computeBounds(data);
      const palette = ['#0f766e', '#b45309', '#7c3aed', '#0c4a6e', '#166534', '#9a3412', '#334155', '#be123c'];

      function addTextLabel(x, y, text, color, bg) {{
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', String(x + 4));
        label.setAttribute('y', String(y - 4));
        label.setAttribute('fill', color || '#0f172a');
        label.setAttribute('font-size', '10');
        label.setAttribute('font-family', 'ui-monospace, SFMono-Regular, Menlo, monospace');
        label.setAttribute('paint-order', 'stroke');
        label.setAttribute('stroke', bg || 'rgba(255,255,255,0.86)');
        label.setAttribute('stroke-width', '2');
        label.textContent = text;
        svgEl.appendChild(label);
      }}

      function addArrowAtEnd(polyPoints, color) {{
        if (!state.arrows) return;
        if (!Array.isArray(polyPoints) || polyPoints.length < 2) return;
        const p1 = polyPoints[polyPoints.length - 1];
        const p0 = polyPoints[polyPoints.length - 2];
        const ang = Math.atan2(p1[1] - p0[1], p1[0] - p0[0]);
        const size = 5.5;
        const left = [p1[0] - size * Math.cos(ang - 0.5), p1[1] - size * Math.sin(ang - 0.5)];
        const right = [p1[0] - size * Math.cos(ang + 0.5), p1[1] - size * Math.sin(ang + 0.5)];
        const tri = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        tri.setAttribute('points', `${{p1[0].toFixed(2)}},${{p1[1].toFixed(2)}} ${{left[0].toFixed(2)}},${{left[1].toFixed(2)}} ${{right[0].toFixed(2)}},${{right[1].toFixed(2)}}`);
        tri.setAttribute('fill', color || '#0f172a');
        tri.setAttribute('opacity', '0.92');
        svgEl.appendChild(tri);
      }}

      function addPathLayer(items, opts) {{
        let idx = 0;
        for (const item of items || []) {{
          const poly = item && item.polyline;
          if (!Array.isArray(poly) || poly.length < 2) continue;
          const pts = [];
          const screenPts = [];
          for (const p of poly) {{
            if (!Array.isArray(p) || p.length < 2) continue;
            const [x, y] = projectPoint(Number(p[0]), Number(p[1]), bounds, width, height, pad);
            screenPts.push([x, y]);
            pts.push(`${{x.toFixed(2)}},${{y.toFixed(2)}}`);
          }}
          if (pts.length < 2) continue;
          const pl = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
          pl.setAttribute('fill', 'none');
          const color = opts.colorByIndex ? palette[idx % palette.length] : opts.color;
          pl.setAttribute('stroke', color);
          pl.setAttribute('stroke-width', String(opts.width || 2));
          if (opts.dash) pl.setAttribute('stroke-dasharray', opts.dash);
          pl.setAttribute('stroke-linecap', 'round');
          pl.setAttribute('stroke-linejoin', 'round');
          pl.setAttribute('points', pts.join(' '));
          const label = [
            item.vehicle ? `vehicle=${{item.vehicle}}` : '',
            item.name ? `path=${{item.name}}` : '',
            Number.isFinite(Number(item.confidence)) ? `conf=${{Number(item.confidence).toFixed(2)}}` : '',
            Number.isFinite(Number(item.length_m)) ? `len=${{Number(item.length_m).toFixed(1)}}m` : '',
          ].filter(Boolean).join(' | ');
          if (label) {{
            const t = document.createElementNS('http://www.w3.org/2000/svg', 'title');
            t.textContent = label;
            pl.appendChild(t);
          }}
          svgEl.appendChild(pl);
          if (state.labels && screenPts.length >= 1) {{
            const shortName = item.vehicle ? `${{item.vehicle}}` : (item.name ? `${{item.name}}` : '');
            const turn = item.turn ? ` • ${{item.turn}}` : '';
            if (shortName) addTextLabel(screenPts[0][0], screenPts[0][1], `${{shortName}}${{turn}}`, color, 'rgba(255,255,255,0.85)');
          }}
          addArrowAtEnd(screenPts, color);
          idx += 1;
        }}
      }}

      if (data.crop) {{
        const c = data.crop;
        const [x1, y1] = projectPoint(Number(c.xmin), Number(c.ymin), bounds, width, height, pad);
        const [x2, y2] = projectPoint(Number(c.xmax), Number(c.ymax), bounds, width, height, pad);
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', String(Math.min(x1, x2)));
        rect.setAttribute('y', String(Math.min(y1, y2)));
        rect.setAttribute('width', String(Math.abs(x2 - x1)));
        rect.setAttribute('height', String(Math.abs(y2 - y1)));
        rect.setAttribute('fill', 'rgba(15,118,110,0.05)');
        rect.setAttribute('stroke', '#0f766e');
        rect.setAttribute('stroke-width', '1.2');
        rect.setAttribute('stroke-dasharray', '5 4');
        svgEl.appendChild(rect);
      }}

      if (state.legal) addPathLayer(data.legal_paths || [], {{color: '#9aa6a1', width: 1.3}});
      if (state.picked) addPathLayer(data.picked_paths || [], {{colorByIndex: true, width: 2.6}});
      if (state.refined) addPathLayer(data.refined_paths || [], {{colorByIndex: true, width: 3.0, dash: '7 4'}});

      function actorColor(kind) {{
        const k = String(kind || '').toLowerCase();
        if (k.includes('walker')) return '#dc2626';
        if (k.includes('cyclist')) return '#ea580c';
        if (k.includes('vehicle')) return '#7c3aed';
        if (k.includes('static')) return '#64748b';
        return '#ea580c';
      }}

      if (state.actors) {{
        for (const a of (data.actors || [])) {{
          const x = Number(a.x), y = Number(a.y);
          if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
          const [px, py] = projectPoint(x, y, bounds, width, height, pad);
          const color = actorColor(a.kind || a.motion_type);
          const dot = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
          dot.setAttribute('x', String(px - 3));
          dot.setAttribute('y', String(py - 3));
          dot.setAttribute('width', '6');
          dot.setAttribute('height', '6');
          dot.setAttribute('rx', '1.2');
          dot.setAttribute('fill', color);
          dot.setAttribute('stroke', '#0f172a');
          dot.setAttribute('stroke-width', '0.7');
          const t = document.createElementNS('http://www.w3.org/2000/svg', 'title');
          const distTxt = Number.isFinite(Number(a.trigger_distance_m)) ? ('(' + Number(a.trigger_distance_m).toFixed(1) + 'm)') : '';
          const triggerTxt = (a.trigger_type || a.trigger_vehicle) ? (' trigger=' + (a.trigger_type || 'unknown') + ':' + (a.trigger_vehicle || 'n/a') + distTxt) : '';
          t.textContent = `${{a.id || 'actor'}} (${{a.kind || 'unknown'}}) motion=${{a.motion_type || 'unknown'}}${{triggerTxt}}`;
          dot.appendChild(t);
          svgEl.appendChild(dot);
          if (state.labels) {{
            const actorLbl = `${{a.id || 'actor'}} • ${{a.kind || 'unknown'}}`;
            addTextLabel(px, py, actorLbl, color, 'rgba(255,255,255,0.90)');
          }}
          if (state.traj && Array.isArray(a.trajectory) && a.trajectory.length >= 2) {{
            const trajPts = [];
            const raw = [];
            for (const p of a.trajectory) {{
              if (!Array.isArray(p) || p.length < 2) continue;
              const [tx, ty] = projectPoint(Number(p[0]), Number(p[1]), bounds, width, height, pad);
              raw.push([tx, ty]);
              trajPts.push(`${{tx.toFixed(2)}},${{ty.toFixed(2)}}`);
            }}
            if (trajPts.length >= 2) {{
              const traj = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
              traj.setAttribute('fill', 'none');
              traj.setAttribute('stroke', color);
              traj.setAttribute('stroke-opacity', '0.9');
              traj.setAttribute('stroke-width', '2.2');
              traj.setAttribute('stroke-dasharray', '4 3');
              traj.setAttribute('points', trajPts.join(' '));
              svgEl.appendChild(traj);
              addArrowAtEnd(raw, color);
            }}
          }}
        }}
      }}

      if (state.spawns) {{
        for (const e of (data.ego_spawns || [])) {{
          const x = Number(e.x), y = Number(e.y);
          if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
          const [px, py] = projectPoint(x, y, bounds, width, height, pad);
          const yaw = Number(e.yaw_deg);
          const ang = Number.isFinite(yaw) ? (-yaw * Math.PI / 180.0) : (-Math.PI / 2);
          const r = 5.0;
          const p1 = [px + r * Math.cos(ang), py + r * Math.sin(ang)];
          const p2 = [px + r * Math.cos(ang + 2.55), py + r * Math.sin(ang + 2.55)];
          const p3 = [px + r * Math.cos(ang - 2.55), py + r * Math.sin(ang - 2.55)];
          const tri = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
          tri.setAttribute('points', `${{p1[0].toFixed(2)}},${{p1[1].toFixed(2)}} ${{p2[0].toFixed(2)}},${{p2[1].toFixed(2)}} ${{p3[0].toFixed(2)}},${{p3[1].toFixed(2)}}`);
          tri.setAttribute('fill', '#1d4ed8');
          tri.setAttribute('stroke', '#1e3a8a');
          tri.setAttribute('stroke-width', '0.8');
          const t = document.createElementNS('http://www.w3.org/2000/svg', 'title');
          t.textContent = `${{e.vehicle || 'ego'}} spawn${{e.source ? ` (${{e.source}})` : ''}}`;
          tri.appendChild(t);
          svgEl.appendChild(tri);
          if (state.labels) addTextLabel(px, py, `${{e.vehicle || 'ego'}}`, '#1e3a8a', 'rgba(255,255,255,0.95)');
        }}
      }}
    }}

    const mapBlocks = Array.from(document.querySelectorAll('.map-block'));
    for (const block of mapBlocks) {{
      const svg = block.querySelector('.map-svg');
      if (!svg) continue;
      const data = parseMapData(svg);
      const state = {{legal: true, picked: true, refined: true, actors: true, traj: true, spawns: true, labels: true, arrows: true}};
      const boxes = Array.from(block.querySelectorAll('input[data-layer]'));
      for (const cb of boxes) {{
        cb.addEventListener('change', () => {{
          const layer = cb.dataset.layer;
          state[layer] = !!cb.checked;
          drawMap(svg, data, state);
        }});
      }}
      drawMap(svg, data, state);
    }}
  </script>
</body>
</html>
"""
    return html_doc


def _to_json_payload(runs: List[SchemaRun], similarity_pairs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    stage_summary = _pipeline_stage_summary(runs)
    similarity_pairs = list(similarity_pairs or [])
    full_run_count = 0
    validation_scores: List[float] = []
    can_accept_count = 0
    interest_scores: List[float] = []
    interest_adjusted_scores: List[float] = []
    duplicate_count = 0
    unique_route_actor_fp_count = len({str(r.route_actor_fingerprint or "") for r in runs if str(r.route_actor_fingerprint or "")})
    for r in runs:
        carla_stage = next((s for s in r.stage_trace if s.get("stage") == "carla_validation"), None)
        if carla_stage and carla_stage.get("requested") and carla_stage.get("status") == "ok":
            full_run_count += 1
        if r.can_accept:
            can_accept_count += 1
        if isinstance(r.validation_score, (int, float)):
            validation_scores.append(float(r.validation_score))
        interest_scores.append(float(r.interest_score))
        interest_adjusted_scores.append(float(r.interest_score_adjusted))
        if r.duplicate_of:
            duplicate_count += 1
    return {
        "summary": {
            "total_runs": len(runs),
            "categories": sorted({r.category for r in runs}),
            "seed_values": sorted({r.seed for r in runs if r.seed is not None}),
            "seed_unknown_count": sum(1 for r in runs if r.seed is None),
            "pass_count": sum(1 for r in runs if r.status == "pass"),
            "fail_count": sum(1 for r in runs if r.status != "pass"),
            "avg_score": round(sum(r.score for r in runs) / len(runs), 1) if runs else 0.0,
            "full_run_count": full_run_count,
            "can_accept_count": can_accept_count,
            "avg_validation_score": round(sum(validation_scores) / len(validation_scores), 3) if validation_scores else None,
            "avg_interest_score": round(sum(interest_scores) / len(interest_scores), 3) if interest_scores else None,
            "avg_interest_score_adjusted": round(sum(interest_adjusted_scores) / len(interest_adjusted_scores), 3) if interest_adjusted_scores else None,
            "duplicate_count": int(duplicate_count),
            "unique_route_actor_fingerprint_count": int(unique_route_actor_fp_count),
        },
        "category_summary": _category_summary(runs),
        "seed_summary": _seed_summary(runs),
        "stage_summary": stage_summary,
        "issue_taxonomy": _issue_taxonomy(runs),
        "similarity_pairs": similarity_pairs,
        "runs": [
            {
                "run_dir": r.run_dir,
                "run_name": r.run_name,
                "timestamp": r.timestamp,
                "category": r.category,
                "seed": r.seed,
                "redo_index": r.redo_index,
                "seed_redo_index": r.seed_redo_index,
                "score": r.score,
                "status": r.status,
                "ego_count": r.ego_count,
                "constraint_count": r.constraint_count,
                "actor_count": r.actor_count,
                "schema_errors": r.schema_errors,
                "schema_warnings": r.schema_warnings,
                "deterministic_adjustments": r.deterministic_adjustments,
                "stage_trace": r.stage_trace,
                "map_layers": r.map_layers,
                "validation_score": r.validation_score,
                "carla_validation": r.carla_validation,
                "can_accept": r.can_accept,
                "compact_summary": r.compact_summary,
                "natural_summary": r.natural_summary,
                "issues": [asdict(i) for i in r.issues],
                "spec": r.spec,
                "fingerprint": r.fingerprint,
                "route_actor_fingerprint": r.route_actor_fingerprint,
                "role_counts": r.role_counts,
                "interaction_count": r.interaction_count,
                "interest_score": r.interest_score,
                "interest_score_adjusted": r.interest_score_adjusted,
                "similarity_best": r.similarity_best,
                "similarity_peer": r.similarity_peer,
                "similarity_cluster_size": r.similarity_cluster_size,
                "duplicate_of": r.duplicate_of,
            }
            for r in runs
        ],
    }


def build_dashboard_from_run_dirs(
    run_dirs: Iterable[Path],
    output_path: Path,
    json_out: Optional[Path] = None,
    title: str = "Scenario Pipeline Dashboard",
) -> Dict[str, Any]:
    """
    Build dashboard from concrete run directories that contain 01_schema/output.json.

    Returns the generated JSON payload.
    """
    unique: List[Path] = []
    seen = set()
    for rd in run_dirs:
        p = Path(rd)
        if not (p / "01_schema" / "output.json").exists():
            continue
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)

    if not unique:
        raise ValueError("No valid schema run directories provided.")

    category_audit_hook = _load_category_audit_hook()
    runs: List[SchemaRun] = []
    for rd in unique:
        try:
            runs.append(_analyze_single_run(rd, category_audit_hook))
        except Exception as exc:
            print(f"[WARN] Failed to analyze {rd}: {exc}")

    if not runs:
        raise RuntimeError("No analyzable schema runs found.")

    _assign_redo_indices(runs)
    similarity_pairs = _annotate_similarity_and_duplicates(runs)

    payload = _to_json_payload(runs, similarity_pairs=similarity_pairs)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_html(runs, title, similarity_pairs=similarity_pairs), encoding="utf-8")

    json_path = Path(json_out) if json_out else output_path.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload


def build_dashboard_from_globs(
    patterns: Iterable[str],
    output_path: Path,
    json_out: Optional[Path] = None,
    title: str = "Scenario Pipeline Dashboard",
) -> Dict[str, Any]:
    """Build dashboard from glob patterns."""
    run_dirs = _discover_run_dirs(patterns)
    return build_dashboard_from_run_dirs(run_dirs, output_path=output_path, json_out=json_out, title=title)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an interactive scenario-pipeline dashboard from many runs.")
    parser.add_argument(
        "--glob",
        nargs="+",
        default=["debug_runs/202*"],
        help="Glob(s) for run dirs or 01_schema/output.json files.",
    )
    parser.add_argument(
        "--output",
        default="debug_runs/schema_dashboard.html",
        help="Output HTML path.",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional JSON dump path (default: <output>.json).",
    )
    parser.add_argument(
        "--title",
        default="Scenario Pipeline Dashboard",
        help="Dashboard title.",
    )
    args = parser.parse_args()

    try:
        output_path = Path(args.output)
        json_out = Path(args.json_out) if args.json_out else output_path.with_suffix(".json")
        payload = build_dashboard_from_globs(
            args.glob,
            output_path=output_path,
            json_out=json_out,
            title=args.title,
        )
    except ValueError:
        print("No schema runs found for patterns:", args.glob)
        return 1
    except RuntimeError:
        print("No analyzable schema runs found.")
        return 2

    pass_count = int(payload["summary"]["pass_count"])
    fail_count = int(payload["summary"]["fail_count"])
    total = int(payload["summary"]["total_runs"])
    print(f"Dashboard written: {output_path}")
    print(f"JSON written: {json_out}")
    print(f"Runs analyzed: {total} (pass={pass_count}, fail={fail_count})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
