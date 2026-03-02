#!/usr/bin/env python3
"""
Debug Pipeline Wrapper — Explicit stage boundaries with rich visualizations.

This wrapper orchestrates the EXISTING pipeline logic without modifying any
internal algorithms, prompts, or CSP logic.  Every stage:

  1. Takes explicit input objects.
  2. Returns explicit output objects.
  3. Writes input.json, output.json, debug.txt, and visualization.png
     inside its numbered sub-folder.
  4. Asserts critical invariants before proceeding.
  5. Can be stopped via ``stop_after_stage``.

Usage
-----
    python scenario_generator/start_pipeline.py \\
        --category "Highway On-Ramp Merge" \\
        --stop-after placement \\
        --seed 42

Programmatic:
    from start_pipeline import run_pipeline_debug, Stage
    run_pipeline_debug("Construction Zone", stop_after_stage=Stage.PLACEMENT)

Stage Names (pass as string to --stop-after)
---------------------------------------------
    schema        — Generate structured scenario spec via LLM
    geometry      — Extract GeometrySpec from spec (deterministic)
    crop          — Select map crop region via CSP
    legal_paths   — Enumerate legal paths in the crop
    pick_paths    — LLM picks vehicle→path assignments
    refine_paths  — LLM + CSP refine spawn/speed parameters
    placement     — LLM + CSP place non-ego actors
    validation    — Rule-based scene validation & scoring
    routes        — Convert scene to per-vehicle CARLA route XMLs
    carla_validation — Final CARLA validation + repair loop (hard infra/spawn gate + soft quality signals)
"""

from __future__ import annotations

import argparse
import enum
import fcntl
import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue as queue_mod
import random
import re
import signal
import socket
import subprocess
import sys
import textwrap
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup (mirrors run_audit_benchmark.py / pipeline_runner.py)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scenario_generator"))

# Also add the *outer* scenario_generator/ dir so deferred imports of
# generate_legal_paths, run_path_picker, run_crop_region_picker, etc. work
# (they live as top-level scripts there, not under scenario_generator.scenario_generator).
_SCENGEN_DIR = str(REPO_ROOT / "scenario_generator")
if _SCENGEN_DIR not in sys.path:
    sys.path.insert(0, _SCENGEN_DIR)

# Imports from the existing codebase — NO modifications to these modules.
from scenario_generator.schema_generator import (
    SchemaScenarioGenerator,
    SchemaGenerationConfig,
)
from scenario_generator.schema_utils import geometry_spec_from_scenario_spec
from scenario_generator.constraints import spec_to_dict
from scenario_generator.scene_validator import SceneValidator, SceneValidationResult
from scenario_generator.pipeline_runner import (
    _constraints_from_schema,
    _entities_from_schema,
)
from scenario_generator.capabilities import (
    CATEGORY_DEFINITIONS,
    TopologyType,
    get_available_categories,
)
# NOTE: convert_scene_to_routes is imported inside stage_generate_routes()
# because it requires the outer scenario_generator/ on sys.path.

# Matplotlib — non-interactive backend for server environments
import matplotlib

if hasattr(matplotlib, "use"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Optional: networkx for schema topology graph
try:
    import networkx as nx

    HAS_NX = True
except ImportError:
    HAS_NX = False

# ###########################################################################
#  Stage Enum
# ###########################################################################


class Stage(enum.Enum):
    """
    Pipeline stages in execution order.

    Pass the ``value`` string (e.g. ``"schema"``) to ``stop_after_stage``.
    """

    SCHEMA = "schema"
    GEOMETRY = "geometry"
    CROP = "crop"
    LEGAL_PATHS = "legal_paths"
    PICK_PATHS = "pick_paths"
    REFINE_PATHS = "refine_paths"
    PLACEMENT = "placement"
    VALIDATION = "validation"
    ROUTES = "routes"
    CARLA_VALIDATION = "carla_validation"


# Ordered list for iteration / index lookup
STAGE_ORDER: List[Stage] = list(Stage)

STAGE_DESCRIPTIONS: Dict[Stage, str] = {
    Stage.SCHEMA: "Generate schema from category and rules.",
    Stage.GEOMETRY: "Extract geometry requirements from schema.",
    Stage.CROP: "Select map crop region with CSP (+ fallback).",
    Stage.LEGAL_PATHS: "Enumerate legal path candidates in crop.",
    Stage.PICK_PATHS: "Pick per-vehicle paths with path picker model.",
    Stage.REFINE_PATHS: "Refine path spawn/speed details.",
    Stage.PLACEMENT: "Place non-ego objects into the scene.",
    Stage.VALIDATION: "Run deterministic scene validation.",
    Stage.ROUTES: "Generate per-vehicle route XMLs.",
    Stage.CARLA_VALIDATION: "Run final CARLA validation + repair loop.",
}

# Human alias required by user request:
# stop_after_stage="paths" == stop after legal path generation stage.
STOP_STAGE_ALIASES: Dict[str, str] = {
    "paths": Stage.LEGAL_PATHS.value,
}

STAGE_DIR_PREFIX = {
    Stage.SCHEMA: "01_schema",
    Stage.GEOMETRY: "02_geometry",
    Stage.CROP: "03_crop",
    Stage.LEGAL_PATHS: "04_legal_paths",
    Stage.PICK_PATHS: "05_pick_paths",
    Stage.REFINE_PATHS: "06_refine",
    Stage.PLACEMENT: "07_placement",
    Stage.VALIDATION: "08_validation",
    Stage.ROUTES: "09_routes",
    Stage.CARLA_VALIDATION: "10_carla_validation",
}

# ###########################################################################
#  Lightweight data containers
# ###########################################################################


@dataclass
class StageResult:
    stage: str
    success: bool
    elapsed_s: float
    error: Optional[str] = None
    traceback: Optional[str] = None


@dataclass
class PipelineContext:
    """Mutable bag passed across stages — all explicit, no hidden globals."""

    category: str
    seed: Optional[int]
    debug_root: Path

    # Populated by successive stages:
    spec: Optional[Any] = None
    spec_dict: Optional[Dict[str, Any]] = None
    schema_text: Optional[str] = None
    geometry_spec: Optional[Any] = None
    town: str = "Town05"
    cat_info: Optional[Any] = None

    # Crop
    crops: Optional[List[Any]] = None
    crop: Optional[Any] = None  # CropBox
    crop_vals: Optional[List[float]] = None
    crop_assignment_method: str = "csp"  # "csp" | "fallback"

    # Legal paths
    cropped_segments: Optional[List[Any]] = None
    legal_paths: Optional[List[Any]] = None
    legal_json_path: Optional[Path] = None
    legal_prompt_path: Optional[Path] = None
    candidates: Optional[List[Dict[str, Any]]] = None

    # Picked / refined paths
    picked_paths_path: Optional[Path] = None
    refined_paths_path: Optional[Path] = None

    # Placement
    scene_json_path: Optional[Path] = None
    scene_png_path: Optional[Path] = None

    # Validation
    validation: Optional[SceneValidationResult] = None

    # Routes
    routes_dir: Optional[Path] = None
    carla_validation: Optional[Dict[str, Any]] = None

    # Model handles
    model: Optional[Any] = None
    tokenizer: Optional[Any] = None

    # CARLA validation controls
    carla_validate: bool = True
    carla_host: str = "127.0.0.1"
    carla_port: int = 3000
    carla_repair_max_attempts: int = 2
    carla_repair_xy_offsets: str = "0.0,0.25,-0.25,0.5,-0.5,1.0,-1.0"
    carla_repair_z_offsets: str = "0.0,0.2,-0.2,0.5,-0.5,1.0"
    carla_align_before_validate: bool = False
    carla_require_risk: bool = True
    carla_validation_timeout: float = 180.0

    # Timing
    stage_results: List[StageResult] = field(default_factory=list)


# ###########################################################################
#  Utility helpers
# ###########################################################################

def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _now_tag_us() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _safe(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")


def _make_unique_run_dir(debug_root: Path, category: str, seed: Optional[int]) -> Path:
    """
    Build a collision-safe run directory path.
    Important for parallel workers that may start in the same second.
    """
    safe_cat = _safe(category)
    seed_suffix = f"_s{seed}" if seed is not None else ""
    base = f"{_now_tag_us()}_{safe_cat}{seed_suffix}"
    root = Path(debug_root)

    for idx in range(1000):
        suffix = "" if idx == 0 else f"_r{idx:03d}"
        run_dir = root / f"{base}{suffix}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue

    raise RuntimeError(f"Failed to allocate unique run directory under {debug_root} for category={category}, seed={seed}")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o: Any) -> Any:
        if hasattr(o, "__dict__"):
            return {k: v for k, v in o.__dict__.items() if not k.startswith("_")}
        if isinstance(o, enum.Enum):
            return o.value
        if isinstance(o, set):
            return sorted(o)
        if isinstance(o, Path):
            return str(o)
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_default, ensure_ascii=False)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_fig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _is_cuda_oom_message(msg: Optional[str]) -> bool:
    """Robust check for CUDA OOM-like failures across torch/transformers variants."""
    text = str(msg or "").lower()
    if not text:
        return False
    return (
        ("cuda" in text and "out of memory" in text)
        or "torch.outofmemoryerror" in text
        or "cublas_status_alloc_failed" in text
        or "cuda_error_out_of_memory" in text
    )


def _is_carla_connection_error(msg: Optional[str]) -> bool:
    """Detect CARLA connectivity/startup failures that should trigger server restart."""
    text = str(msg or "").lower()
    if not text:
        return False
    return (
        "carla_connect_failed" in text
        or ("time-out" in text and "simulator" in text)
        or "did not open within" in text
        or "failed to start" in text
    )


def _extract_failed_stage_error_from_run(run_dir: Path, summary: Dict[str, Any]) -> Optional[str]:
    """Try to recover the exact stage error text from <stage>/output.json."""
    failed_stage = summary.get("failed_stage")
    if not failed_stage:
        return summary.get("error_message")
    try:
        stage_enum = Stage(str(failed_stage))
    except Exception:
        return summary.get("error_message")
    stage_output = Path(run_dir) / STAGE_DIR_PREFIX[stage_enum] / "output.json"
    if stage_output.exists():
        try:
            out_obj = _read_json(stage_output)
            if isinstance(out_obj, dict) and out_obj.get("error"):
                return str(out_obj.get("error"))
        except Exception:
            pass
    return summary.get("error_message")


def _should_defer_carla_validation(stop_after_stage: Optional[str], carla_validate: bool) -> bool:
    """
    Multi-run policy:
    1) Generate scenarios through routes first.
    2) Run CARLA validation afterward in a single sequential lane.
    """
    if not bool(carla_validate):
        return False
    resolved = _resolve_stop_stage(stop_after_stage)
    return resolved is None or STAGE_ORDER.index(resolved) >= STAGE_ORDER.index(Stage.CARLA_VALIDATION)


def _refresh_result_record_from_run(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Refresh task result fields from run_dir/summary.json if available."""
    run_dir_raw = rec.get("run_dir")
    if not run_dir_raw:
        rec["carla_infra_warning"] = False
        return rec
    run_dir = Path(str(run_dir_raw))
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        rec["all_passed"] = False
        rec["error"] = str(rec.get("error") or "missing summary.json")
        rec["carla_infra_warning"] = False
        return rec
    try:
        summary = _read_json(summary_path)
    except Exception as exc:
        rec["all_passed"] = False
        rec["error"] = f"failed to read summary.json: {exc}"
        rec["carla_infra_warning"] = False
        return rec
    if rec.get("category") in (None, ""):
        rec["category"] = summary.get("category")
    if rec.get("seed") is None:
        rec["seed"] = summary.get("seed")
    rec["all_passed"] = bool(summary.get("all_stages_passed", False))
    rec["failed_stage"] = summary.get("failed_stage")
    rec["score"] = summary.get("validation_score")
    rec["error"] = _extract_failed_stage_error_from_run(run_dir, summary)
    rec["is_cuda_oom"] = _is_cuda_oom_message(rec.get("error"))
    rec["carla_infra_warning"] = bool(summary.get("carla_validation_infra_error", False))
    return rec


def _upsert_carla_stage_record(summary: Dict[str, Any], stage_record: Dict[str, Any]) -> None:
    """Insert or replace the carla_validation stage record in-order."""
    stages_raw = summary.get("stages")
    stages: List[Dict[str, Any]] = list(stages_raw) if isinstance(stages_raw, list) else []
    stages = [s for s in stages if str((s or {}).get("stage")) != Stage.CARLA_VALIDATION.value]
    stages.append(stage_record)

    def _stage_sort_key(item: Dict[str, Any]) -> int:
        name = str((item or {}).get("stage"))
        try:
            return STAGE_ORDER.index(Stage(name))
        except Exception:
            return 10**6

    stages.sort(key=_stage_sort_key)
    summary["stages"] = stages


def _read_route_xy_points(xml_path: Path) -> List[Tuple[float, float]]:
    try:
        import xml.etree.ElementTree as ET

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


def _scenario_geometry_fingerprint(run_dir: Path) -> str:
    """
    Build a stable geometry fingerprint from final route shapes and actor placements.
    Used for target-mode duplicate filtering.
    """
    run_dir = Path(run_dir)
    routes_dir = run_dir / STAGE_DIR_PREFIX[Stage.ROUTES] / "routes"
    manifest_path = routes_dir / "actors_manifest.json"
    if not manifest_path.exists():
        return f"missing_routes::{run_dir.name}"
    try:
        manifest = _read_json(manifest_path)
    except Exception:
        return f"bad_manifest::{run_dir.name}"

    def _q(v: float, step: float = 1.0) -> float:
        return round(float(v) / step) * step

    token_rows: List[str] = []
    role_order = ["ego", "npc", "pedestrian", "bicycle", "static"]
    for role in role_order:
        entries = manifest.get(role)
        if not isinstance(entries, list):
            continue
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
            quantized = [(int(round(_q(x, 1.0))), int(round(_q(y, 1.0)))) for x, y in sample]
            model = str(entry.get("model", "")).strip()
            token_rows.append(
                json.dumps(
                    {
                        "role": role,
                        "model": model,
                        "wp_count": int(len(points)),
                        "qpts": quantized,
                    },
                    sort_keys=True,
                    ensure_ascii=True,
                )
            )

    if not token_rows:
        return f"no_tokens::{run_dir.name}"
    token_rows.sort()
    payload = "\n".join(token_rows).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _annotate_target_acceptance(
    run_dir: Path,
    *,
    accepted: bool,
    reason: str,
    acceptance_level: Optional[str] = None,
    duplicate_of: Optional[str] = None,
    geometry_fingerprint: Optional[str] = None,
    target_cycle: Optional[int] = None,
) -> None:
    """
    Persist target-mode acceptance bookkeeping in summary.json without mutating
    stage pass/fail semantics.
    """
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return
    try:
        summary = _read_json(summary_path)
    except Exception:
        return
    if not isinstance(summary, dict):
        return
    target_meta = summary.get("target_acceptance")
    if not isinstance(target_meta, dict):
        target_meta = {}
    level = str(acceptance_level or ("high" if accepted else "rejected")).strip().lower()
    if level not in {"high", "medium", "rejected"}:
        level = "high" if accepted else "rejected"
    target_meta.update(
        {
            "accepted": bool(accepted),
            "reason": str(reason),
            "acceptance_level": level,
            "duplicate_of": str(duplicate_of) if duplicate_of else None,
            "geometry_fingerprint": str(geometry_fingerprint) if geometry_fingerprint else None,
            "target_cycle": int(target_cycle) if target_cycle is not None else None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    summary["target_acceptance"] = target_meta
    _write_json(summary_path, summary)


def _is_fully_passed_record(rec: Dict[str, Any]) -> bool:
    return bool(rec.get("all_passed", False)) and not rec.get("failed_stage")


def _carla_gate_status_from_run(run_dir: Path) -> Tuple[bool, str]:
    """
    Return whether final CARLA gate passed for a run plus a short reason token.
    Target mode uses this to assign acceptance level:
      - high: CARLA gate pass
      - medium: scenario otherwise valid but CARLA gate failed
    """
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    summary: Dict[str, Any] = {}
    if summary_path.exists():
        try:
            loaded = _read_json(summary_path)
            if isinstance(loaded, dict):
                summary = loaded
        except Exception:
            summary = {}

    if bool(summary.get("carla_validation_pass", False)):
        return True, "carla_gate_passed"
    if bool(summary.get("carla_validation_infra_error", False)):
        return False, "carla_infra_warning"

    output_path = run_dir / STAGE_DIR_PREFIX[Stage.CARLA_VALIDATION] / "output.json"
    if output_path.exists():
        try:
            out = _read_json(output_path)
            if isinstance(out, dict):
                if bool(out.get("passed", False)):
                    return True, "carla_gate_passed"
                reason = str(out.get("failure_reason") or "").strip()
                if reason:
                    return False, f"carla_gate_failed:{reason}"
        except Exception:
            pass
    reason = str(summary.get("carla_validation_reason") or "").strip()
    if reason:
        return False, f"carla_gate_failed:{reason}"
    return False, "carla_gate_not_passed"


def _downgrade_connect_failure_to_infra_warning(
    run_dir: Path,
    *,
    error: Optional[str],
    attempts: int,
) -> Dict[str, Any]:
    """
    Convert persistent CARLA connect failures into a non-blocking infra warning.
    Scenario generation remains successful; CARLA validation is marked as infra-limited.
    """
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    summary: Dict[str, Any] = {}
    if summary_path.exists():
        try:
            loaded = _read_json(summary_path)
            if isinstance(loaded, dict):
                summary = dict(loaded)
        except Exception:
            summary = {}

    reason = str(
        error
        or summary.get("carla_validation_reason")
        or summary.get("error_message")
        or "carla_connect_failed"
    )

    existing_stage = None
    for s in list(summary.get("stages") or []):
        if str((s or {}).get("stage")) == Stage.CARLA_VALIDATION.value:
            existing_stage = dict(s)
            break
    if not isinstance(existing_stage, dict):
        existing_stage = {"stage": Stage.CARLA_VALIDATION.value, "elapsed_s": 0.0}
    existing_stage["stage"] = Stage.CARLA_VALIDATION.value
    existing_stage["success"] = True
    existing_stage["warning"] = "carla_connect_failed_non_blocking"
    existing_stage["infra_error"] = reason
    _upsert_carla_stage_record(summary, existing_stage)

    if str(summary.get("failed_stage")) == Stage.CARLA_VALIDATION.value:
        summary.pop("failed_stage", None)
        summary.pop("error_message", None)
    summary.pop("stopped_after", None)
    summary.pop("stopped_after_stage", None)
    summary.pop("stop_after_stage", None)

    summary["carla_validation_pass"] = False
    summary["carla_validation_reason"] = reason
    summary["carla_validation_infra_error"] = True
    summary["carla_validation_soft_fail"] = True
    summary["carla_validation_attempts"] = max(1, int(attempts))
    summary.setdefault("carla_validation_metrics", {})
    summary.setdefault("repairs_applied", [])
    summary["completed_at"] = datetime.now(timezone.utc).isoformat()

    stages_raw = summary.get("stages")
    if isinstance(stages_raw, list):
        summary["all_stages_passed"] = all(bool((s or {}).get("success", False)) for s in stages_raw)

    _write_json(summary_path, summary)

    output_path = run_dir / STAGE_DIR_PREFIX[Stage.CARLA_VALIDATION] / "output.json"
    if output_path.exists():
        try:
            output_obj = _read_json(output_path)
            if isinstance(output_obj, dict):
                output_obj["infra_warning"] = True
                output_obj["soft_fail_non_blocking"] = True
                output_obj["infra_failure_reason"] = reason
                output_obj["infra_attempts"] = max(1, int(attempts))
                _write_json(output_path, output_obj)
        except Exception:
            pass

    refreshed = {
        "category": summary.get("category"),
        "seed": summary.get("seed"),
        "run_dir": str(run_dir),
        "all_passed": bool(summary.get("all_stages_passed", False)),
        "failed_stage": summary.get("failed_stage"),
        "score": summary.get("validation_score"),
        "error": None,
        "is_cuda_oom": False,
        "carla_infra_warning": True,
    }
    return _refresh_result_record_from_run(refreshed)


def _run_deferred_carla_validation_for_run(
    run_dir: Path,
    *,
    carla_host: str,
    carla_port: int,
    carla_repair_max_attempts: int,
    carla_repair_xy_offsets: str,
    carla_repair_z_offsets: str,
    carla_align_before_validate: bool,
    carla_require_risk: bool,
    carla_validation_timeout: float,
    carla_process_started: bool,
) -> Dict[str, Any]:
    """
    Execute only Stage 10 (carla_validation) for an existing run directory and
    merge the stage result into run_dir/summary.json.
    """
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    existing_summary: Dict[str, Any] = {}
    if summary_path.exists():
        try:
            loaded = _read_json(summary_path)
            if isinstance(loaded, dict):
                existing_summary = dict(loaded)
        except Exception:
            existing_summary = {}

    category = str(existing_summary.get("category") or run_dir.name)
    seed = existing_summary.get("seed")
    town = str(existing_summary.get("town") or _resolve_town(category))
    routes_dir = run_dir / STAGE_DIR_PREFIX[Stage.ROUTES] / "routes"

    ctx = PipelineContext(
        category=category,
        seed=seed,
        debug_root=run_dir,
        town=town,
        routes_dir=routes_dir,
        carla_validate=True,
        carla_host=str(carla_host),
        carla_port=int(carla_port),
        carla_repair_max_attempts=max(0, int(carla_repair_max_attempts)),
        carla_repair_xy_offsets=str(carla_repair_xy_offsets),
        carla_repair_z_offsets=str(carla_repair_z_offsets),
        carla_align_before_validate=bool(carla_align_before_validate),
        carla_require_risk=bool(carla_require_risk),
        carla_validation_timeout=float(carla_validation_timeout),
    )

    stage_dir = run_dir / STAGE_DIR_PREFIX[Stage.CARLA_VALIDATION]
    ok, _ = _run_stage(Stage.CARLA_VALIDATION, stage_dir, stage_carla_validation, ctx)
    stage_result = ctx.stage_results[-1]
    stage_record: Dict[str, Any] = {
        "stage": Stage.CARLA_VALIDATION.value,
        "success": bool(stage_result.success),
        "elapsed_s": float(stage_result.elapsed_s),
    }
    if not stage_result.success:
        stage_record["error"] = stage_result.error
        stage_record["traceback"] = stage_result.traceback

    summary = dict(existing_summary)
    summary.setdefault("category", category)
    summary.setdefault("seed", seed)
    summary.setdefault("town", town)
    summary.setdefault("run_dir", str(run_dir))
    summary["carla_process_started"] = bool(carla_process_started)
    summary["carla_host"] = str(carla_host)
    summary["carla_port"] = int(carla_port)
    _upsert_carla_stage_record(summary, stage_record)
    summary.pop("stopped_after", None)
    summary.pop("stopped_after_stage", None)
    summary.pop("stop_after_stage", None)

    if not ok:
        summary["failed_stage"] = Stage.CARLA_VALIDATION.value
        summary["error_message"] = stage_result.error
    elif str(summary.get("failed_stage")) == Stage.CARLA_VALIDATION.value:
        summary.pop("failed_stage", None)
        summary.pop("error_message", None)

    if isinstance(ctx.carla_validation, dict):
        summary["carla_validation_pass"] = bool(ctx.carla_validation.get("passed", False))
        summary["carla_validation_reason"] = ctx.carla_validation.get("failure_reason")
        summary["carla_validation_metrics"] = ctx.carla_validation.get("metrics", {})
        summary["repairs_applied"] = ctx.carla_validation.get("repairs", [])
        # Clear infra error flags from previous connection failures if validation now succeeded
        if bool(ctx.carla_validation.get("passed", False)):
            summary.pop("carla_validation_infra_error", None)
            summary.pop("carla_validation_soft_fail", None)
    else:
        summary.pop("carla_validation_pass", None)
        summary.pop("carla_validation_reason", None)
        summary.pop("carla_validation_metrics", None)
        summary.pop("repairs_applied", None)

    stages_raw = summary.get("stages")
    if isinstance(stages_raw, list):
        summary["all_stages_passed"] = all(bool((s or {}).get("success", False)) for s in stages_raw)
    summary["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(summary_path, summary)

    rec = {
        "category": category,
        "seed": seed,
        "run_dir": str(run_dir),
        "all_passed": bool(summary.get("all_stages_passed", False)),
        "failed_stage": summary.get("failed_stage"),
        "score": summary.get("validation_score"),
        "error": _extract_failed_stage_error_from_run(run_dir, summary),
        "is_cuda_oom": False,
    }
    return rec


def _run_deferred_carla_validation_batch(
    *,
    results_summary: List[Dict[str, Any]],
    carla_host: str,
    carla_port: int,
    carla_repair_max_attempts: int,
    carla_repair_xy_offsets: str,
    carla_repair_z_offsets: str,
    carla_align_before_validate: bool,
    carla_require_risk: bool,
    carla_validation_timeout: float,
    start_carla: bool,
    carla_root: Optional[str],
    carla_args: Optional[List[str]],
    carla_startup_timeout: float,
    carla_shutdown_timeout: float,
    carla_port_tries: int,
    carla_port_step: int,
    preferred_gpu_ids: Optional[List[int]],
    connect_retries: int = 3,
) -> None:
    """
    Run final CARLA validation sequentially for runs that already passed through routes.
    Keeps one CARLA process alive across runs when start_carla=True.
    """
    candidates: List[Dict[str, Any]] = []
    for rec in results_summary:
        run_dir_raw = rec.get("run_dir")
        if not run_dir_raw:
            continue
        if not bool(rec.get("all_passed", False)):
            continue
        if rec.get("failed_stage"):
            continue
        candidates.append(rec)

    if not candidates:
        print("[INFO] Deferred CARLA validation: no eligible runs (all failed before routes or no run_dir).")
        return

    print("\n" + "#" * 70)
    print(
        "  DEFERRED CARLA VALIDATION — "
        f"{len(candidates)} run(s), mode={'single auto-started instance' if start_carla else 'external CARLA'}"
    )
    print("#" * 70)

    global _active_carla_manager
    previous_active_carla = _active_carla_manager
    carla_manager: Optional[CarlaProcessManager] = None
    active_carla_port = int(carla_port)
    should_own_carla = bool(start_carla)

    resolved_carla_root: Optional[Path] = None
    resolved_carla_args: List[str] = []
    user_has_graphics_arg = False
    gpu_id_candidates: List[int] = []
    current_gpu_index = 0

    def _set_launch_gpu(gpu_id: Optional[int]) -> None:
        nonlocal resolved_carla_args
        if gpu_id is None:
            return
        resolved_carla_args = [
            arg
            for arg in resolved_carla_args
            if not str(arg).strip().lower().startswith("-graphicsadapter")
        ]
        resolved_carla_args.append(f"-graphicsadapter={int(gpu_id)}")

    def _advance_port_for_retry(reason: str) -> None:
        nonlocal active_carla_port
        old_port = int(active_carla_port)
        retry_step = max(CARLA_TM_PORT_OFFSET + 1, int(carla_port_step))
        candidate_start = old_port + retry_step
        try:
            active_carla_port = int(
                _find_available_carla_port(
                    carla_host,
                    candidate_start,
                    tries=max(2, int(carla_port_tries)),
                    step=max(1, int(carla_port_step)),
                )
            )
        except Exception:
            active_carla_port = int(candidate_start)
        if int(active_carla_port) != old_port:
            print(f"  [CARLA-PORT] {old_port} -> {active_carla_port} ({reason})")

    if should_own_carla:
        _install_carla_signal_handlers()
        resolved_carla_root = (
            Path(carla_root).expanduser().resolve()
            if carla_root
            else Path(os.environ.get("CARLA_ROOT", str(REPO_ROOT / "carla912"))).expanduser().resolve()
        )
        resolved_carla_args = list(carla_args or [])
        user_has_graphics_arg = _has_graphics_adapter_arg(resolved_carla_args)
        seen = set()
        for gid in list(preferred_gpu_ids or []):
            try:
                v = int(gid)
            except Exception:
                continue
            if v < 0 or v in seen:
                continue
            seen.add(v)
            gpu_id_candidates.append(v)
        if not gpu_id_candidates and not user_has_graphics_arg:
            inferred = _infer_primary_visible_gpu_id()
            if inferred is not None:
                gpu_id_candidates.append(int(inferred))

        if gpu_id_candidates:
            _set_launch_gpu(gpu_id_candidates[current_gpu_index])
            print(
                "[INFO] Deferred CARLA launch pinned to GPU "
                f"{gpu_id_candidates[current_gpu_index]} (candidate set: {gpu_id_candidates})."
            )
        elif user_has_graphics_arg:
            print("[INFO] Deferred CARLA launch using user-provided -graphicsadapter setting.")

    def _start_or_restart_carla(reason: str) -> None:
        nonlocal carla_manager, active_carla_port
        global _active_carla_manager
        if not should_own_carla:
            return
        if resolved_carla_root is None:
            raise RuntimeError("Deferred CARLA root not initialized")
        if carla_manager is not None:
            try:
                carla_manager.stop()
            except Exception:
                pass
        mgr = CarlaProcessManager(
            carla_root=resolved_carla_root,
            host=carla_host,
            port=int(active_carla_port),
            extra_args=resolved_carla_args,
            startup_timeout_s=float(carla_startup_timeout),
            shutdown_timeout_s=float(carla_shutdown_timeout),
            port_tries=int(carla_port_tries),
            port_step=int(carla_port_step),
        )
        _active_carla_manager = mgr
        active_carla_port = int(mgr.start())
        carla_manager = mgr
        print(
            f"  [CARLA] using {active_carla_port}/{active_carla_port + CARLA_STREAM_PORT_OFFSET}/"
            f"{active_carla_port + CARLA_TM_PORT_OFFSET} ({reason})"
        )

    try:
        if should_own_carla:
            launch_ok = False
            last_launch_error: Optional[str] = None
            initial_launch_tries = max(1, int(connect_retries) + 1)
            for launch_try in range(initial_launch_tries):
                reason = "initial" if launch_try == 0 else f"initial_retry_{launch_try + 1}"
                if launch_try > 0:
                    _advance_port_for_retry(reason)
                    if len(gpu_id_candidates) > 1:
                        current_gpu_index = (current_gpu_index + 1) % len(gpu_id_candidates)
                        _set_launch_gpu(gpu_id_candidates[current_gpu_index])
                        print(
                            f"  [CARLA-GPU] rotating to GPU {gpu_id_candidates[current_gpu_index]} "
                            f"({reason})"
                        )
                try:
                    _start_or_restart_carla(reason)
                    launch_ok = True
                    break
                except Exception as exc:
                    last_launch_error = str(exc)
                    print(
                        "[WARN] Initial CARLA launch attempt failed "
                        f"({launch_try + 1}/{initial_launch_tries}): {exc}"
                    )
            if not launch_ok:
                err_text = str(last_launch_error or "initial_carla_launch_failed")
                print(f"[WARN] Initial CARLA launch failed after retries: {err_text}")
                total = len(candidates)
                for idx, rec in enumerate(candidates, 1):
                    run_dir = Path(str(rec.get("run_dir")))
                    category = str(rec.get("category", ""))
                    seed = rec.get("seed")
                    rec.update(
                        _downgrade_connect_failure_to_infra_warning(
                            run_dir,
                            error=f"carla_connect_failed:{err_text}",
                            attempts=initial_launch_tries,
                        )
                    )
                    print(
                        f"  [CARLA {idx}/{total}] ✓ {category} (seed={seed}) "
                        "warning=carla_connect_failed_non_blocking"
                    )
                return

        total = len(candidates)
        for idx, rec in enumerate(candidates, 1):
            run_dir = Path(str(rec.get("run_dir")))
            category = str(rec.get("category", ""))
            seed = rec.get("seed")

            max_connect_retries = max(0, int(connect_retries)) if should_own_carla else 0
            final_connect_failure = False
            attempts_used = 0
            for infra_try in range(max_connect_retries + 1):
                attempts_used = infra_try + 1
                if should_own_carla:
                    needs_restart = (
                        carla_manager is None
                        or (not carla_manager.is_running())
                        or (not _is_port_open(carla_host, int(active_carla_port), timeout=1.0))
                    )
                    if needs_restart:
                        reason = "healthcheck_failed_before_validation"
                        if infra_try > 0:
                            reason = "retry_healthcheck_failed_before_validation"
                            _advance_port_for_retry(reason)
                            if len(gpu_id_candidates) > 1:
                                current_gpu_index = (current_gpu_index + 1) % len(gpu_id_candidates)
                                _set_launch_gpu(gpu_id_candidates[current_gpu_index])
                                print(
                                    f"  [CARLA-GPU] rotating to GPU {gpu_id_candidates[current_gpu_index]} "
                                    f"({reason})"
                                )
                        print(f"  [CARLA-RESTART] {reason}")
                        try:
                            _start_or_restart_carla(reason)
                        except Exception as exc:
                            rec.update(
                                {
                                    "failed_stage": Stage.CARLA_VALIDATION.value,
                                    "error": f"carla_connect_failed:{exc}",
                                }
                            )
                            final_connect_failure = True
                            if infra_try < max_connect_retries:
                                _advance_port_for_retry("restart_launch_failure")
                                if len(gpu_id_candidates) > 1:
                                    current_gpu_index = (current_gpu_index + 1) % len(gpu_id_candidates)
                                    _set_launch_gpu(gpu_id_candidates[current_gpu_index])
                                    print(
                                        f"  [CARLA-GPU] rotating to GPU {gpu_id_candidates[current_gpu_index]} "
                                        "(restart_launch_failure)"
                                    )
                                continue
                            break

                updated = _run_deferred_carla_validation_for_run(
                    run_dir,
                    carla_host=carla_host,
                    carla_port=int(active_carla_port),
                    carla_repair_max_attempts=carla_repair_max_attempts,
                    carla_repair_xy_offsets=carla_repair_xy_offsets,
                    carla_repair_z_offsets=carla_repair_z_offsets,
                    carla_align_before_validate=carla_align_before_validate,
                    carla_require_risk=carla_require_risk,
                    carla_validation_timeout=carla_validation_timeout,
                    carla_process_started=bool(should_own_carla),
                )
                rec.update(updated)

                is_connect_failure = bool(
                    rec.get("failed_stage") == Stage.CARLA_VALIDATION.value
                    and _is_carla_connection_error(rec.get("error"))
                )
                final_connect_failure = bool(is_connect_failure)
                if is_connect_failure and should_own_carla and infra_try < max_connect_retries:
                    print("  [CARLA-RESTART] post_validation_connect_failure")
                    _advance_port_for_retry("post_validation_connect_failure")
                    if len(gpu_id_candidates) > 1:
                        current_gpu_index = (current_gpu_index + 1) % len(gpu_id_candidates)
                        _set_launch_gpu(gpu_id_candidates[current_gpu_index])
                        print(
                            f"  [CARLA-GPU] rotating to GPU {gpu_id_candidates[current_gpu_index]} "
                            "(post_validation_connect_failure)"
                        )
                    try:
                        _start_or_restart_carla("post_validation_connect_failure")
                    except Exception as exc:
                        rec.update(
                            {
                                "failed_stage": Stage.CARLA_VALIDATION.value,
                                "error": f"carla_connect_failed:{exc}",
                            }
                        )
                        final_connect_failure = True
                        if infra_try < max_connect_retries:
                            _advance_port_for_retry("post_validation_restart_launch_failure")
                            if len(gpu_id_candidates) > 1:
                                current_gpu_index = (current_gpu_index + 1) % len(gpu_id_candidates)
                                _set_launch_gpu(gpu_id_candidates[current_gpu_index])
                                print(
                                    f"  [CARLA-GPU] rotating to GPU {gpu_id_candidates[current_gpu_index]} "
                                    "(post_validation_restart_launch_failure)"
                                )
                            continue
                    continue
                break

            if final_connect_failure:
                print(
                    "  [CARLA-WARN] connectivity remained unstable after "
                    f"{attempts_used} attempt(s); marking as infra warning (non-blocking)."
                )
                rec.update(
                    _downgrade_connect_failure_to_infra_warning(
                        run_dir,
                        error=rec.get("error"),
                        attempts=attempts_used,
                    )
                )

            status = "✓" if rec.get("all_passed") else "✗"
            fail = f" failed={rec.get('failed_stage')}" if rec.get("failed_stage") else ""
            err = f" error={rec.get('error')}" if rec.get("error") else ""
            warn = " warning=carla_connect_failed_non_blocking" if rec.get("carla_infra_warning") else ""
            print(f"  [CARLA {idx}/{total}] {status} {category} (seed={seed}){fail}{err}{warn}")
    finally:
        if carla_manager is not None:
            try:
                carla_manager.stop()
            except Exception:
                pass
        _active_carla_manager = previous_active_carla


def _query_gpu_inventory_mib(gpu_ids: List[int]) -> Optional[Dict[int, Dict[str, int]]]:
    """
    Query GPU memory totals/usage via nvidia-smi.
    Returns {gpu_id: {"total": MiB, "used": MiB, "free": MiB}} or None.
    """
    if not gpu_ids:
        return None
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    inventory: Dict[int, Dict[str, int]] = {}
    for line in proc.stdout.splitlines():
        raw = [t.strip() for t in line.split(",")]
        if len(raw) != 3:
            continue
        try:
            idx = int(raw[0])
            total = int(raw[1])
            used = int(raw[2])
        except Exception:
            continue
        free = max(0, total - used)
        inventory[idx] = {"total": total, "used": used, "free": free}

    selected = {gid: inventory[gid] for gid in gpu_ids if gid in inventory}
    return selected or None


def _cap_workers_per_gpu_by_memory(
    gpu_ids: List[int],
    requested_workers_per_gpu: int,
    model_reserve_mib: int = 22000,
    generation_headroom_mib: int = 2500,
) -> Tuple[int, Optional[Dict[int, Dict[str, int]]]]:
    """
    Cap workers/GPU using available memory to avoid guaranteed OOM.
    Uses a conservative per-worker budget: model footprint + runtime headroom.
    """
    if requested_workers_per_gpu <= 1 or not gpu_ids:
        return max(1, requested_workers_per_gpu), None
    inv = _query_gpu_inventory_mib(gpu_ids)
    if not inv:
        return requested_workers_per_gpu, None

    per_worker_need = model_reserve_mib + generation_headroom_mib
    per_gpu_caps: List[int] = []
    for gid in gpu_ids:
        info = inv.get(gid)
        if not info:
            continue
        free = int(info["free"])
        cap = max(1, free // max(1, per_worker_need))
        per_gpu_caps.append(cap)
    if not per_gpu_caps:
        return requested_workers_per_gpu, inv

    capped = max(1, min(requested_workers_per_gpu, min(per_gpu_caps)))
    return capped, inv


def _resolve_town(category: str, default: str = "Town05") -> str:
    """Pick the right town for a category (mirrors _resolve_town_for_category)."""
    info = CATEGORY_DEFINITIONS.get(category)
    if info:
        topo = info.map.topology
        if topo == TopologyType.ROUNDABOUT:
            return "Town03"
        # Town05 has no valid highway/on-ramp crops in current map features.
        if topo == TopologyType.HIGHWAY:
            return "Town06"
        # Keep T-junction categories on a town with known feasible true T-junction crops.
        if topo == TopologyType.T_JUNCTION:
            return "Town02"
        # Keep two-lane overtaking on genuinely two-lane roads.
        if topo == TopologyType.TWO_LANE_CORRIDOR:
            return "Town01"
    return default


CARLA_STARTUP_TIMEOUT_S = 120.0
CARLA_SHUTDOWN_TIMEOUT_S = 15.0
CARLA_PORT_TRIES = 8
CARLA_PORT_STEP = 1
CARLA_TM_PORT_OFFSET = 5
CARLA_STREAM_PORT_OFFSET = 1
CARLA_PARALLEL_PORT_MIN_STRIDE = 10
CARLA_START_RETRIES = 3
try:
    CARLA_START_STABILITY_S = float(os.environ.get("COLMDRIVER_CARLA_START_STABILITY_S", "70.0"))
except Exception:
    CARLA_START_STABILITY_S = 70.0
if CARLA_START_STABILITY_S < 0.5:
    CARLA_START_STABILITY_S = 0.5
CARLA_START_LOCK_PATH = "/tmp/colmdriver_carla_start.lock"
CARLA_RUNTIME_ROOT = "/tmp/colmdriver_carla_runtime"
_CARLA_USER_CLEANUP_DONE = False


def _is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for_port(host: str, port: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + max(0.1, float(timeout_s))
    while time.monotonic() < deadline:
        if _is_port_open(host, port, timeout=1.0):
            return True
        time.sleep(0.5)
    return False


def _wait_for_port_or_process_exit(
    host: str,
    port: int,
    proc: subprocess.Popen,
    timeout_s: float,
) -> Tuple[bool, Optional[int]]:
    """Wait until CARLA port opens or the launched process exits."""
    deadline = time.monotonic() + max(0.1, float(timeout_s))
    while time.monotonic() < deadline:
        if _is_port_open(host, port, timeout=1.0):
            return True, None
        rc = proc.poll()
        if rc is not None:
            return False, int(rc)
        time.sleep(0.5)
    rc = proc.poll()
    if rc is not None:
        return False, int(rc)
    return False, None


def _wait_for_port_close(host: str, port: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + max(0.1, float(timeout_s))
    while time.monotonic() < deadline:
        if not _is_port_open(host, port, timeout=1.0):
            return True
        time.sleep(0.5)
    return False


def _is_carla_port_pair_free(
    host: str,
    world_port: int,
    tm_port_offset: int = CARLA_TM_PORT_OFFSET,
    timeout: float = 1.0,
) -> bool:
    world = int(world_port)
    stream_port = world + CARLA_STREAM_PORT_OFFSET
    tm_port = world + int(tm_port_offset)
    if world <= 0 or world > 65535 or stream_port > 65535 or tm_port > 65535:
        return False
    if _is_port_open(host, world, timeout=timeout):
        return False
    if _is_port_open(host, stream_port, timeout=timeout):
        return False
    if _is_port_open(host, tm_port, timeout=timeout):
        return False
    return True


def _find_available_carla_port(host: str, preferred_port: int, tries: int, step: int) -> int:
    attempts = max(1, int(tries))
    # Never probe immediately adjacent world ports because CARLA reserves
    # companion ports (+1 stream, +5 traffic manager).
    step_val = max(CARLA_TM_PORT_OFFSET + 1, int(step))
    # Legacy default 2000 tends to be crowded on shared hosts.
    # Prefer searching from 3000 first, then fall back to the requested range.
    if int(preferred_port) == 2000:
        starts = [3000, int(preferred_port)]
    else:
        starts = [int(preferred_port)]
    for start in starts:
        for idx in range(attempts):
            port = int(start) + idx * step_val
            if port > 65535:
                break
            if _is_carla_port_pair_free(host, port, tm_port_offset=CARLA_TM_PORT_OFFSET, timeout=1.0):
                return port
    raise RuntimeError(
        f"No free CARLA port set found starting at {preferred_port} "
        f"(tries={attempts}, step={step_val}, required_offsets=0,{CARLA_STREAM_PORT_OFFSET},{CARLA_TM_PORT_OFFSET})."
    )


@contextmanager
def _carla_startup_lock(lock_path: str = CARLA_START_LOCK_PATH):
    """
    Serialize CARLA launch across processes to reduce startup storms and
    transient bind/startup failures.
    """
    lock_file = Path(lock_path)
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with lock_file.open("a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _compute_parallel_carla_port_stride(carla_port_tries: int, carla_port_step: int) -> int:
    tries = max(1, int(carla_port_tries))
    step = max(1, int(carla_port_step))
    search_span = tries * step
    return max(CARLA_PARALLEL_PORT_MIN_STRIDE, search_span + CARLA_TM_PORT_OFFSET + 1)


def _parallel_worker_carla_port(
    base_carla_port: int,
    worker_index: int,
    carla_port_tries: int,
    carla_port_step: int,
) -> int:
    stride = _compute_parallel_carla_port_stride(carla_port_tries, carla_port_step)
    port = int(base_carla_port) + int(worker_index) * int(stride)
    tm_port = port + CARLA_TM_PORT_OFFSET
    if port > 65535 or tm_port > 65535:
        raise ValueError(
            f"Computed CARLA worker port out of range: world={port}, tm={tm_port}. "
            f"base={base_carla_port}, worker_index={worker_index}, stride={stride}."
        )
    return port


def _find_parallel_carla_base_port(
    host: str,
    preferred_port: int,
    worker_count: int,
    carla_port_tries: int,
    carla_port_step: int,
) -> int:
    """
    Pick one base port whose full worker block is free:
      worker_i uses base + i*stride (plus CARLA stream/TM companion ports).
    """
    count = max(1, int(worker_count))
    stride = _compute_parallel_carla_port_stride(carla_port_tries, carla_port_step)
    # The 2000-range is commonly occupied by manually launched CARLA instances.
    if int(preferred_port) == 2000:
        starts = [24000, int(preferred_port)]
    else:
        starts = [int(preferred_port)]

    probe_rows = max(16, max(1, int(carla_port_tries)) * 8)
    max_world_port = 65535 - max(CARLA_STREAM_PORT_OFFSET, CARLA_TM_PORT_OFFSET)
    for start in starts:
        for row in range(probe_rows):
            candidate = int(start) + row * stride
            highest_world = candidate + (count - 1) * stride
            if highest_world > max_world_port:
                break
            ok = True
            for w in range(count):
                world = int(candidate) + w * stride
                if not _is_carla_port_pair_free(host, world, tm_port_offset=CARLA_TM_PORT_OFFSET, timeout=0.5):
                    ok = False
                    break
            if ok:
                return int(candidate)

    raise RuntimeError(
        "Unable to allocate a collision-free CARLA port block for parallel workers: "
        f"preferred={preferred_port}, workers={count}, stride={stride}, host={host}, "
        f"required_offsets=0,{CARLA_STREAM_PORT_OFFSET},{CARLA_TM_PORT_OFFSET}."
    )


def _has_graphics_adapter_arg(extra_args: List[str]) -> bool:
    for arg in extra_args:
        a = str(arg).strip().lower()
        if a.startswith("-graphicsadapter"):
            return True
    return False


def _infer_primary_visible_gpu_id() -> Optional[int]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not raw:
        return None
    token = str(raw).split(",")[0].strip()
    if not token:
        return None
    if not token.lstrip("-").isdigit():
        return None
    gpu_id = int(token)
    if gpu_id < 0:
        return None
    return gpu_id


def _effective_carla_args(extra_args: Optional[List[str]]) -> List[str]:
    args = list(extra_args or [])
    if _has_graphics_adapter_arg(args):
        return args
    gpu_id = _infer_primary_visible_gpu_id()
    if gpu_id is None:
        return args
    args.append(f"-graphicsadapter={gpu_id}")
    return args


def _is_pid_alive(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def _list_user_owned_carla_processes() -> List[Dict[str, Any]]:
    uid = int(os.getuid())
    matches: List[Dict[str, Any]] = []
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pid=,uid=,cmd="],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except Exception as exc:
        print(f"[WARN] Unable to enumerate user CARLA processes: {exc}")
        return matches

    for raw_line in str(proc.stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid_raw, uid_raw, cmd = parts
        try:
            pid = int(pid_raw)
            row_uid = int(uid_raw)
        except Exception:
            continue
        if row_uid != uid:
            continue
        lowered = str(cmd).strip().lower()
        if "carlaue4-linux-shipping" not in lowered and "carlaue4.sh" not in lowered:
            continue
        matches.append({"pid": int(pid), "uid": row_uid, "cmd": str(cmd).strip()})
    return matches


def _kill_user_owned_carla_processes(reason: str, term_timeout_s: float = 5.0) -> Dict[str, int]:
    processes = _list_user_owned_carla_processes()
    if not processes:
        print("[INFO] User-scoped CARLA cleanup: no existing CARLA processes found.")
        return {
            "matched": 0,
            "term_sent": 0,
            "kill_sent": 0,
            "failed": 0,
            "survivors": 0,
        }

    uid = int(os.getuid())
    print(
        "[INFO] User-scoped CARLA cleanup: terminating "
        f"{len(processes)} process(es) for uid={uid} before launch ({reason})."
    )

    term_sent = 0
    failed = 0
    for rec in processes:
        pid = int(rec["pid"])
        try:
            os.kill(pid, signal.SIGTERM)
            term_sent += 1
        except ProcessLookupError:
            pass
        except Exception:
            failed += 1

    deadline = time.monotonic() + max(0.5, float(term_timeout_s))
    survivors = [rec for rec in processes if _is_pid_alive(int(rec["pid"]))]
    while survivors and time.monotonic() < deadline:
        time.sleep(0.2)
        survivors = [rec for rec in survivors if _is_pid_alive(int(rec["pid"]))]

    kill_sent = 0
    if survivors:
        for rec in survivors:
            pid = int(rec["pid"])
            try:
                os.kill(pid, signal.SIGKILL)
                kill_sent += 1
            except ProcessLookupError:
                pass
            except Exception:
                failed += 1
        time.sleep(0.2)
        survivors = [rec for rec in survivors if _is_pid_alive(int(rec["pid"]))]

    summary = {
        "matched": len(processes),
        "term_sent": int(term_sent),
        "kill_sent": int(kill_sent),
        "failed": int(failed),
        "survivors": int(len(survivors)),
    }
    if int(summary["survivors"]) > 0:
        survivor_pids = [str(int(rec["pid"])) for rec in survivors[:8]]
        suffix = "..." if len(survivors) > 8 else ""
        print(
            "[WARN] User-scoped CARLA cleanup left surviving process(es): "
            f"{','.join(survivor_pids)}{suffix}"
        )
    print(
        "[INFO] User-scoped CARLA cleanup result: "
        f"matched={summary['matched']} term={summary['term_sent']} "
        f"kill={summary['kill_sent']} failed={summary['failed']} "
        f"survivors={summary['survivors']}"
    )
    return summary


def _maybe_cleanup_user_carla_processes(enable: bool, reason: str) -> None:
    global _CARLA_USER_CLEANUP_DONE
    if not bool(enable):
        return
    if _CARLA_USER_CLEANUP_DONE:
        return
    _CARLA_USER_CLEANUP_DONE = True
    _kill_user_owned_carla_processes(reason=reason)


def _should_add_offscreen(extra_args: List[str]) -> bool:
    if os.environ.get("DISPLAY"):
        return False
    lowered = {str(arg).lower() for arg in extra_args}
    if any("renderoffscreen" in arg for arg in lowered):
        return False
    if any("windowed" in arg for arg in lowered):
        return False
    return True


def _should_isolate_carla_runtime() -> bool:
    raw = str(os.environ.get("COLMDRIVER_CARLA_ISOLATE_RUNTIME", "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _has_any_renderer_arg(extra_args: List[str]) -> bool:
    lowered = {str(arg).strip().lower() for arg in extra_args}
    return any(arg in {"-opengl", "-vulkan", "-nullrhi"} for arg in lowered)


def _prefer_headless_opengl() -> bool:
    raw = str(os.environ.get("COLMDRIVER_CARLA_HEADLESS_OPENGL", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _wait_for_process_and_port_stable(
    host: str,
    port: int,
    proc: subprocess.Popen,
    settle_s: float,
) -> Tuple[bool, Optional[int]]:
    """
    After the world port first opens, ensure CARLA stays alive/healthy for a
    short settle window before declaring startup success.
    """
    deadline = time.monotonic() + max(0.5, float(settle_s))
    while time.monotonic() < deadline:
        rc = proc.poll()
        if rc is not None:
            return False, int(rc)
        if not _is_port_open(host, int(port), timeout=1.0):
            return False, None
        time.sleep(0.5)
    return True, None


class CarlaProcessManager:
    """Owns a local CARLA server process and guarantees teardown."""

    def __init__(
        self,
        carla_root: Path,
        host: str,
        port: int,
        extra_args: List[str],
        startup_timeout_s: float,
        shutdown_timeout_s: float,
        port_tries: int,
        port_step: int,
    ) -> None:
        self.carla_root = Path(carla_root)
        self.host = str(host)
        self.port = int(port)
        self.extra_args = list(extra_args)
        self.startup_timeout_s = float(startup_timeout_s)
        self.shutdown_timeout_s = float(shutdown_timeout_s)
        self.port_tries = int(port_tries)
        self.port_step = int(port_step)
        self.process: Optional[subprocess.Popen] = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self) -> int:
        if self.is_running():
            return int(self.port)

        carla_script = self.carla_root / "CarlaUE4.sh"
        if not carla_script.exists():
            raise FileNotFoundError(f"CARLA launcher not found: {carla_script}")

        max_start_attempts = max(1, int(CARLA_START_RETRIES))
        last_error: Optional[str] = None

        # Startup storms from multiple workers can make CARLA flaky.
        # Serialize launch, but keep instances running concurrently afterward.
        with _carla_startup_lock():
            for start_attempt in range(1, max_start_attempts + 1):
                if not _is_carla_port_pair_free(self.host, self.port, tm_port_offset=CARLA_TM_PORT_OFFSET, timeout=1.0):
                    new_port = _find_available_carla_port(
                        self.host,
                        self.port,
                        tries=self.port_tries,
                        step=self.port_step,
                    )
                    if new_port != self.port:
                        old_stream = int(self.port) + CARLA_STREAM_PORT_OFFSET
                        old_tm = int(self.port) + CARLA_TM_PORT_OFFSET
                        new_stream = int(new_port) + CARLA_STREAM_PORT_OFFSET
                        new_tm = int(new_port) + CARLA_TM_PORT_OFFSET
                        print(
                            "[INFO] CARLA port set "
                            f"{self.port}/{old_stream}/{old_tm} busy; switching to "
                            f"{new_port}/{new_stream}/{new_tm}."
                        )
                        self.port = int(new_port)

                launch_args = list(self.extra_args)
                if (
                    _prefer_headless_opengl()
                    and _should_add_offscreen(launch_args)
                    and not _has_any_renderer_arg(launch_args)
                ):
                    launch_args.append("-opengl")

                renderer_fallback = start_attempt >= 2 and not _has_any_renderer_arg(launch_args)
                if renderer_fallback:
                    launch_args.append("-opengl")
                if start_attempt >= 3 and not any(str(a).strip().lower() == "-nosound" for a in launch_args):
                    launch_args.append("-nosound")

                cmd = [str(carla_script), f"--world-port={self.port}"]
                if _should_add_offscreen(launch_args):
                    cmd.append("-RenderOffScreen")
                cmd.extend(launch_args)
                env = os.environ.copy()
                if _should_isolate_carla_runtime():
                    # Optional isolation mode for debugging shared-cache issues.
                    runtime_root = Path(CARLA_RUNTIME_ROOT) / f"port_{int(self.port)}"
                    cfg_root = runtime_root / "config"
                    cache_root = runtime_root / "cache"
                    data_root = runtime_root / "data"
                    cfg_root.mkdir(parents=True, exist_ok=True)
                    cache_root.mkdir(parents=True, exist_ok=True)
                    data_root.mkdir(parents=True, exist_ok=True)
                    env["XDG_CONFIG_HOME"] = str(cfg_root)
                    env["XDG_CACHE_HOME"] = str(cache_root)
                    env["XDG_DATA_HOME"] = str(data_root)

                print(
                    f"[INFO] Starting CARLA (attempt {start_attempt}/{max_start_attempts}): "
                    f"{' '.join(cmd)}"
                )
                if start_attempt == 1 and any(str(a).strip().lower() == "-opengl" for a in launch_args):
                    print("[INFO] CARLA launch policy: headless renderer set to -opengl.")
                if renderer_fallback:
                    print("[INFO] CARLA startup fallback: enabling -opengl renderer.")
                self.process = subprocess.Popen(
                    cmd,
                    cwd=str(self.carla_root),
                    env=env,
                    start_new_session=True,
                )

                opened, exit_code = _wait_for_port_or_process_exit(
                    self.host,
                    self.port,
                    self.process,
                    self.startup_timeout_s,
                )
                if opened:
                    if float(CARLA_START_STABILITY_S) >= 10.0:
                        print(
                            "[INFO] CARLA startup warmup: monitoring process/port health for "
                            f"{CARLA_START_STABILITY_S:.1f}s."
                        )
                    stable, stable_exit = _wait_for_process_and_port_stable(
                        self.host,
                        self.port,
                        self.process,
                        settle_s=float(CARLA_START_STABILITY_S),
                    )
                    if stable:
                        return int(self.port)
                    if stable_exit is not None:
                        last_error = (
                            f"CARLA exited with code {stable_exit} during startup stabilization "
                            f"(port={self.port}, settle={CARLA_START_STABILITY_S:.1f}s)."
                        )
                    else:
                        last_error = (
                            f"CARLA became unhealthy during startup stabilization "
                            f"(port={self.port}, settle={CARLA_START_STABILITY_S:.1f}s)."
                        )
                    self.stop()
                    if start_attempt < max_start_attempts:
                        retry_step = max(CARLA_TM_PORT_OFFSET + 1, int(self.port_step))
                        self.port = int(self.port) + retry_step
                        time.sleep(1.0)
                    continue

                if exit_code is not None:
                    last_error = (
                        f"CARLA exited early with code {exit_code} before opening port {self.port} "
                        f"(startup timeout={self.startup_timeout_s:.1f}s)."
                    )
                else:
                    last_error = (
                        f"CARLA port {self.port} did not open within {self.startup_timeout_s:.1f}s."
                    )
                self.stop()

                if start_attempt < max_start_attempts:
                    # Move forward to a fresh candidate set before retrying.
                    retry_step = max(CARLA_TM_PORT_OFFSET + 1, int(self.port_step))
                    self.port = int(self.port) + retry_step
                    time.sleep(1.0)

        raise RuntimeError(last_error or "CARLA failed to start.")

    def stop(self) -> None:
        proc = self.process
        if proc is None:
            return
        if proc.poll() is None:
            print("[INFO] Stopping CARLA...")
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
            try:
                proc.wait(timeout=max(1.0, self.shutdown_timeout_s))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                try:
                    proc.wait(timeout=max(1.0, self.shutdown_timeout_s))
                except Exception:
                    pass

        _wait_for_port_close(self.host, self.port, timeout_s=5.0)
        self.process = None


_active_carla_manager: Optional[CarlaProcessManager] = None
_carla_signal_handlers_installed = False


def _install_carla_signal_handlers() -> None:
    global _carla_signal_handlers_installed
    if _carla_signal_handlers_installed:
        return

    def _handler(signum: int, _frame: Any) -> None:
        mgr = _active_carla_manager
        if mgr is not None:
            try:
                mgr.stop()
            except Exception:
                pass
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    _carla_signal_handlers_installed = True


def _hardcoded_crop_override(category: str, town: str) -> Optional[List[float]]:
    """
    Deterministic crop overrides for categories with known robust map regions.
    """
    key = (str(category).strip(), str(town).strip())
    hardcoded: Dict[Tuple[str, str], List[float]] = {
        # Known T-junction in Town02 (x:[100,180], y:[200,260]).
        ("Major/Minor Unsignalized Entry", "Town02"): [100.0, 180.0, 200.0, 260.0],
    }
    vals = hardcoded.get(key)
    return list(vals) if vals is not None else None


def _resolve_stop_stage(stop_after_stage: Optional[str]) -> Optional[Stage]:
    """Resolve user-provided stop stage with alias support."""
    if stop_after_stage is None:
        return None
    normalized = STOP_STAGE_ALIASES.get(stop_after_stage, stop_after_stage)
    try:
        return Stage(normalized)
    except ValueError as exc:
        valid = [s.value for s in Stage] + sorted(STOP_STAGE_ALIASES.keys())
        raise ValueError(
            f"Invalid stage '{stop_after_stage}'. Valid stages: {sorted(set(valid))}"
        ) from exc


def _build_integrated_schema_dashboard(
    run_dirs: List[Path],
    debug_root: Path,
    output_path: Optional[str] = None,
    json_path: Optional[str] = None,
    title: Optional[str] = None,
    extra_globs: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Build integrated pipeline dashboard directly from debug run directories.
    Returns metadata dict on success, None on failure.
    """
    if not run_dirs:
        return None

    try:
        import schema_dashboard as schema_dash
    except Exception as exc:
        try:
            import importlib.util
            dash_path = REPO_ROOT / "scenario_generator" / "schema_dashboard.py"
            spec = importlib.util.spec_from_file_location("_integrated_schema_dashboard", str(dash_path))
            if spec is None or spec.loader is None:
                raise RuntimeError("Unable to resolve schema_dashboard.py spec")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            schema_dash = module
        except Exception as exc2:
            print(f"[WARN] Failed to import integrated schema dashboard: {exc}; fallback failed: {exc2}")
            return None

    selected: List[Path] = []
    seen = set()
    for rd in run_dirs:
        key = str(Path(rd).resolve())
        if key in seen:
            continue
        seen.add(key)
        selected.append(Path(rd))

    if extra_globs:
        try:
            extra_dirs = schema_dash.discover_run_dirs(extra_globs)
            for rd in extra_dirs:
                key = str(Path(rd).resolve())
                if key in seen:
                    continue
                seen.add(key)
                selected.append(Path(rd))
        except Exception as exc:
            print(f"[WARN] Failed to include extra dashboard globs {extra_globs}: {exc}")

    out_html = Path(output_path) if output_path else Path(debug_root) / f"pipeline_dashboard_{_now_tag()}.html"
    out_json = Path(json_path) if json_path else out_html.with_suffix(".json")
    dash_title = title or f"Scenario Pipeline Dashboard ({len(selected)} runs)"

    try:
        payload = schema_dash.build_dashboard_from_run_dirs(
            selected,
            output_path=out_html,
            json_out=out_json,
            title=dash_title,
        )
    except Exception as exc:
        print(f"[WARN] Integrated schema dashboard build failed: {exc}")
        return None

    summary = payload.get("summary", {})
    print(f"  Pipeline dashboard: {out_html}")
    print(f"  Pipeline dashboard JSON: {out_json}")
    print(
        "  Dashboard summary: "
        f"runs={summary.get('total_runs', 0)}, "
        f"pass={summary.get('pass_count', 0)}, "
        f"fail={summary.get('fail_count', 0)}, "
        f"avg_score={summary.get('avg_score', 0.0)}"
    )
    return {
        "html_path": str(out_html),
        "json_path": str(out_json),
        "summary": summary,
        "included_run_count": len(selected),
    }


def _required_relations_to_dicts(cat_info: Any) -> List[Dict[str, Any]]:
    rels = []
    if not cat_info or not getattr(cat_info.rules, "required_relations", None):
        return rels
    for rel in cat_info.rules.required_relations:
        rels.append(
            {
                "entry_relation": getattr(rel, "entry_relation", "any"),
                "first_maneuver": getattr(getattr(rel, "first_maneuver", "unknown"), "value", getattr(rel, "first_maneuver", "unknown")),
                "second_maneuver": getattr(getattr(rel, "second_maneuver", "unknown"), "value", getattr(rel, "second_maneuver", "unknown")),
                "exit_relation": getattr(rel, "exit_relation", "any"),
                "entry_lane_relation": getattr(rel, "entry_lane_relation", "any"),
                "exit_lane_relation": getattr(rel, "exit_lane_relation", "any"),
            }
        )
    return rels


def _stage_input_fallback(ctx: PipelineContext, stage: Stage) -> Dict[str, Any]:
    """Minimal context snapshot if a stage failed before writing input.json."""
    return {
        "stage": stage.value,
        "category": ctx.category,
        "seed": ctx.seed,
        "town": ctx.town,
        "schema_available": ctx.spec_dict is not None,
        "geometry_available": ctx.geometry_spec is not None,
        "crop_available": ctx.crop_vals is not None,
        "legal_paths_available": bool(ctx.legal_paths),
        "picked_paths_path": str(ctx.picked_paths_path) if ctx.picked_paths_path else None,
        "refined_paths_path": str(ctx.refined_paths_path) if ctx.refined_paths_path else None,
        "scene_json_path": str(ctx.scene_json_path) if ctx.scene_json_path else None,
    }


def _viz_stage_failure(stage_dir: Path, stage: Stage, error: str) -> None:
    """Always generate a visualization for failed stages."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.set_facecolor("#fff3f3")
    ax.text(
        0.5,
        0.70,
        f"STAGE FAILED: {stage.value}",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#b71c1c",
    )
    ax.text(
        0.5,
        0.35,
        textwrap.shorten(error or "Unknown error", width=140, placeholder=" ..."),
        ha="center",
        va="center",
        fontsize=10,
        color="#4e342e",
    )
    ax.axis("off")
    _save_fig(fig, stage_dir / "visualization.png")


def _ensure_stage_artifacts(
    stage: Stage,
    stage_dir: Path,
    ctx: PipelineContext,
    result: Optional[Any] = None,
    error: Optional[str] = None,
) -> None:
    """Guarantee per-stage artifact contract."""
    input_path = stage_dir / "input.json"
    output_path = stage_dir / "output.json"
    debug_path = stage_dir / "debug.txt"
    viz_path = stage_dir / "visualization.png"

    if not input_path.exists():
        _write_json(input_path, _stage_input_fallback(ctx, stage))
    if not output_path.exists():
        _write_json(
            output_path,
            {"stage": stage.value, "result": result, "error": error},
        )
    if not debug_path.exists():
        if error:
            _write_text(debug_path, f"Stage {stage.value} failed:\n{error}\n")
        else:
            _write_text(debug_path, f"Stage {stage.value} completed.\n")
    if not viz_path.exists():
        if error:
            _viz_stage_failure(stage_dir, stage, error)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(8, 2))
            ax.text(
                0.5,
                0.5,
                f"Stage {stage.value}: visualization auto-generated placeholder",
                ha="center",
                va="center",
                fontsize=10,
            )
            ax.axis("off")
            _save_fig(fig, viz_path)


def _xy_from_point(point: Any) -> Optional[Tuple[float, float]]:
    if isinstance(point, dict) and "x" in point and "y" in point:
        try:
            return float(point["x"]), float(point["y"])
        except Exception:
            return None
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        try:
            return float(point[0]), float(point[1])
        except Exception:
            return None
    return None


def _extract_xy_from_signature(sig: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for seg in sig.get("segments_detailed", []) or []:
        poly = (
            seg.get("polyline_sampled")
            or seg.get("polyline_sample")
            or seg.get("polyline")
            or []
        )
        for p in poly:
            xy = _xy_from_point(p)
            if xy is not None:
                xs.append(xy[0])
                ys.append(xy[1])
    return xs, ys


def _path_length_m(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return sum(math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]) for i in range(len(xs) - 1))


def _polyline_overlap_points(
    polylines: List[Tuple[str, List[float], List[float]]],
    threshold_m: float = 1.0,
) -> List[Tuple[float, float, str]]:
    overlaps: List[Tuple[float, float, str]] = []
    for i in range(len(polylines)):
        name_i, x_i, y_i = polylines[i]
        for j in range(i + 1, len(polylines)):
            name_j, x_j, y_j = polylines[j]
            min_d = 1e9
            best = None
            for a in range(len(x_i)):
                for b in range(len(x_j)):
                    d = math.hypot(x_i[a] - x_j[b], y_i[a] - y_j[b])
                    if d < min_d:
                        min_d = d
                        best = (x_i[a], y_i[a])
            if best is not None and min_d <= threshold_m:
                overlaps.append((best[0], best[1], f"{name_i}<->{name_j} ({min_d:.2f}m)"))
    return overlaps


# ###########################################################################
#  VISUALIZATION HELPERS
# ###########################################################################

# ----- constraint type → color map for schema viz -----
_CONSTRAINT_COLORS: Dict[str, str] = {
    "same_approach_as": "#1565C0",      # blue
    "opposite_approach": "#6A1B9A",     # purple
    "adjacent_lane": "#2E7D32",         # green
    "same_lane": "#00838F",             # teal
    "merge_into_lane_of": "#E65100",    # orange
    "yields_to": "#C62828",             # red
    "faster_than": "#AD1457",           # pink
    "slower_than": "#4527A0",           # deep purple
    "on_ramp_entry": "#EF6C00",         # amber
}


def _constraint_color(ctype: str) -> str:
    ctype_lower = ctype.lower()
    for key, color in _CONSTRAINT_COLORS.items():
        if key in ctype_lower:
            return color
    return "#555555"  # grey default


def _viz_schema(stage_dir: Path, spec_dict: Dict[str, Any], cat_info: Any) -> None:
    """
    STAGE 1 — Schema topology graph with pipeline-context reminders.

    Layout: 3 panels
      Left   — Topology graph (vehicles as nodes, constraints as color-coded edges)
      Middle — Vehicle detail cards + actor list
      Right  — Pipeline reminder / what-to-check diagnostics
    """
    vehicles = spec_dict.get("ego_vehicles", [])
    constraints = spec_dict.get("vehicle_constraints", [])
    actors = spec_dict.get("actors", [])
    required_relations = _required_relations_to_dicts(cat_info)

    topology = spec_dict.get("topology", "?")
    category = spec_dict.get("category", "?")
    n_vehicles = len(vehicles)
    n_constraints = len(constraints)
    n_actors = len(actors)

    fig = plt.figure(figsize=(22, 11))
    gs = fig.add_gridspec(2, 3, width_ratios=[3, 2, 2], height_ratios=[4, 1],
                          hspace=0.15, wspace=0.18)
    ax_graph = fig.add_subplot(gs[0, 0])
    ax_details = fig.add_subplot(gs[0, 1])
    ax_checks = fig.add_subplot(gs[0, 2])
    ax_legend = fig.add_subplot(gs[1, :])

    # ====================== LEFT: Topology Graph ======================
    if HAS_NX and vehicles:
        G = nx.DiGraph()
        for v in vehicles:
            vid = v.get("vehicle_id", "?")
            maneuver = v.get("maneuver", "?")
            lc_phase = v.get("lane_change_phase", "")
            label = f"{vid}\n{maneuver}"
            if lc_phase and lc_phase != "unknown":
                label += f"\nlc={lc_phase}"
            G.add_node(vid, label=label)

        # Color-coded edges per constraint type
        edge_colors_list = []
        for c in constraints:
            a, b = c.get("a", "?"), c.get("b", "?")
            ctype = c.get("type", "?")
            if a in G and b in G:
                G.add_edge(a, b, ctype=ctype)
                edge_colors_list.append(_constraint_color(ctype))

        pos = nx.spring_layout(G, seed=42, k=2.8)
        nx.draw_networkx_nodes(
            G, pos, ax=ax_graph,
            node_color="#4fc3f7", node_size=2800, edgecolors="black", linewidths=2,
        )
        nx.draw_networkx_labels(
            G, pos, ax=ax_graph,
            labels={n: G.nodes[n]["label"] for n in G.nodes},
            font_size=9, font_weight="bold",
        )
        edges = list(G.edges())
        if edges:
            nx.draw_networkx_edges(
                G, pos, ax=ax_graph,
                edgelist=edges,
                edge_color=edge_colors_list if edge_colors_list else "#444",
                arrowstyle="->", arrowsize=18,
                connectionstyle="arc3,rad=0.15", width=2.2,
            )
            # Edge labels
            edge_labels = {}
            for c in constraints:
                a, b = c.get("a", "?"), c.get("b", "?")
                ctype = c.get("type", "?")
                if (a, b) in [(e[0], e[1]) for e in edges]:
                    short = ctype.replace("_", " ")
                    edge_labels[(a, b)] = short
            nx.draw_networkx_edge_labels(
                G, pos, ax=ax_graph,
                edge_labels=edge_labels, font_size=7,
                font_color="#333",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=0.3),
            )

        # Overlay required-relation arrows (dashed red)
        vehicle_ids = [v.get("vehicle_id", "?") for v in vehicles if v.get("vehicle_id")]
        for i, rel in enumerate(required_relations):
            if len(vehicle_ids) < 2:
                break
            a = vehicle_ids[i % len(vehicle_ids)]
            b = vehicle_ids[(i + 1) % len(vehicle_ids)]
            if a not in pos or b not in pos:
                continue
            start, end = pos[a], pos[b]
            rad = 0.35 + 0.06 * i
            ax_graph.annotate(
                "", xy=end, xytext=start,
                arrowprops={"arrowstyle": "->", "lw": 1.5, "linestyle": "--",
                            "color": "#d32f2f",
                            "connectionstyle": f"arc3,rad={rad}"},
                zorder=8,
            )
            mx = (start[0] + end[0]) * 0.5
            my = (start[1] + end[1]) * 0.5
            ax_graph.text(
                mx, my + 0.12 + i * 0.04,
                f"REQ{i+1}: {rel.get('first_maneuver', '?')}/{rel.get('second_maneuver', '?')}",
                fontsize=7, color="#d32f2f", fontweight="bold",
                bbox={"facecolor": "#fff8e1", "alpha": 0.9, "edgecolor": "#d32f2f", "pad": 2},
                zorder=9,
            )

        ax_graph.set_title(
            f"Topology: {topology}  |  {n_vehicles} vehicles, {n_constraints} constraints",
            fontsize=11, fontweight="bold",
        )
    else:
        # Fallback: text
        lines = []
        for v in vehicles:
            lines.append(f"{v.get('vehicle_id')}: man={v.get('maneuver')}, "
                         f"entry={v.get('entry_road')}, exit={v.get('exit_road')}")
        for c in constraints:
            lines.append(f"{c.get('a')} -> {c.get('b')} ({c.get('type')})")
        for i, rel in enumerate(required_relations, start=1):
            lines.append(f"REQ{i}: {rel}")
        ax_graph.text(0.05, 0.95, "Vehicle Topology\n" + "\n".join(lines),
                      transform=ax_graph.transAxes, va="top", fontsize=9, family="monospace")
        ax_graph.set_title(f"Topology: {topology} (text fallback)", fontsize=11)
    ax_graph.axis("off")

    # ====================== MIDDLE: Vehicle Cards + Actors ======================
    detail_lines = []
    detail_lines.append(f"CATEGORY: {category}")
    detail_lines.append(f"TOPOLOGY: {topology}")
    detail_lines.append("")

    # Flags summary with visual markers
    flags = [
        ("needs_on_ramp", spec_dict.get("needs_on_ramp", False)),
        ("needs_merge", spec_dict.get("needs_merge", False)),
        ("needs_multi_lane", spec_dict.get("needs_multi_lane", False)),
        ("needs_oncoming", spec_dict.get("needs_oncoming", False)),
    ]
    detail_lines.append("FLAGS:")
    for fname, fval in flags:
        marker = "■" if fval else "□"
        detail_lines.append(f"  {marker} {fname}")
    detail_lines.append("")

    # Per-vehicle detail cards
    detail_lines.append(f"─── VEHICLES ({n_vehicles}) ───")
    for v in vehicles:
        vid = v.get("vehicle_id", "?")
        detail_lines.append(f"  ┌─ {vid}")
        detail_lines.append(f"  │  maneuver  : {v.get('maneuver', '?')}")
        detail_lines.append(f"  │  entry_road: {v.get('entry_road', '?')}")
        detail_lines.append(f"  │  exit_road : {v.get('exit_road', '?')}")
        lc = v.get("lane_change_phase", "")
        if lc and lc != "unknown":
            detail_lines.append(f"  │  lc_phase  : {lc}")
        # Which constraints involve this vehicle?
        involved = [c for c in constraints if c.get("a") == vid or c.get("b") == vid]
        if involved:
            for ic in involved:
                other = ic.get("b") if ic.get("a") == vid else ic.get("a")
                detail_lines.append(f"  │  constraint: {ic.get('type')} ↔ {other}")
        detail_lines.append(f"  └────")
    detail_lines.append("")

    # Constraints summary
    detail_lines.append(f"─── CONSTRAINTS ({n_constraints}) ───")
    for c in constraints:
        detail_lines.append(f"  {c.get('a')} → {c.get('b')}: {c.get('type')}")
    if not constraints:
        detail_lines.append("  (none)")
    detail_lines.append("")

    # Actors
    detail_lines.append(f"─── ACTORS ({n_actors}) ───")
    for a in actors:
        aid = a.get("actor_id", "?")
        kind = a.get("kind", "?")
        motion = a.get("motion", "?")
        affects = a.get("affects_vehicle", "—")
        qty = a.get("quantity", 1)
        detail_lines.append(f"  {aid} ({kind} ×{qty})")
        detail_lines.append(f"    motion={motion}  affects={affects}")
        lat = a.get("lateral_position", "")
        timing = a.get("timing_phase", "")
        if lat:
            detail_lines.append(f"    lateral={lat}  timing={timing}")
    if not actors:
        detail_lines.append("  (none)")

    ax_details.text(
        0.03, 0.97, "\n".join(detail_lines),
        transform=ax_details.transAxes, va="top",
        fontsize=7.5, family="monospace",
        bbox=dict(facecolor="#fafafa", edgecolor="#ddd", pad=5),
    )
    ax_details.set_title("Vehicle & Actor Details", fontsize=11, fontweight="bold")
    ax_details.axis("off")

    # ====================== RIGHT: Pipeline Reminders + Diagnostics ======================
    check_lines = []
    check_lines.append("── PIPELINE REMINDER ──")
    check_lines.append("This schema drives ALL later stages:")
    check_lines.append("  1. Schema → 2. Geometry → 3. Crop")
    check_lines.append("  → 4. Legal Paths → 5. Pick Paths")
    check_lines.append("  → 6. Refine → 7. Place → 8. Validate")
    check_lines.append("  → 9. Routes")
    check_lines.append("")
    check_lines.append("Schema errors propagate everywhere.")
    check_lines.append("Bad constraints → wrong paths picked.")
    check_lines.append("Wrong topology flag → wrong crop region.")
    check_lines.append("")

    # Required relations block
    check_lines.append("── REQUIRED RELATIONS ──")
    if required_relations:
        for i, rel in enumerate(required_relations, start=1):
            check_lines.append(f"  REQ{i}:")
            check_lines.append(f"    entry_rel     : {rel.get('entry_relation', '?')}")
            check_lines.append(f"    first_maneuver: {rel.get('first_maneuver', '?')}")
            check_lines.append(f"    second_maneuv : {rel.get('second_maneuver', '?')}")
            check_lines.append(f"    exit_relation : {rel.get('exit_relation', '?')}")
            check_lines.append(f"    entry_lane_rel: {rel.get('entry_lane_relation', '?')}")
            check_lines.append(f"    exit_lane_rel : {rel.get('exit_lane_relation', '?')}")
    else:
        check_lines.append("  (no category-level required relations)")
    check_lines.append("")

    # Diagnostics — things to eyeball
    check_lines.append("── WHAT TO CHECK ──")
    issues_found = 0

    # Check: duplicate vehicle IDs?
    vids = [v.get("vehicle_id") for v in vehicles]
    if len(vids) != len(set(vids)):
        check_lines.append("  ✗ DUPLICATE vehicle IDs!")
        issues_found += 1
    else:
        check_lines.append(f"  ✓ {n_vehicles} unique vehicle IDs")

    # Check: all constraints reference valid vehicles?
    vid_set = set(vids)
    bad_refs = [c for c in constraints if c.get("a") not in vid_set or c.get("b") not in vid_set]
    if bad_refs:
        check_lines.append(f"  ✗ {len(bad_refs)} constraint(s) ref unknown vehicles")
        for br in bad_refs:
            check_lines.append(f"    {br.get('a')} → {br.get('b')}: {br.get('type')}")
        issues_found += 1
    else:
        check_lines.append("  ✓ All constraint refs valid")

    # Check: maneuvers consistent with topology?
    maneuvers = [v.get("maneuver", "") for v in vehicles]
    if topology in ("corridor", "highway", "two_lane_corridor"):
        non_straight = [m for m in maneuvers if m not in ("straight", "lane_change", "merge", "on_ramp_merge")]
        if non_straight:
            check_lines.append(f"  ⚠ Non-straight maneuvers in {topology}: {non_straight}")
            issues_found += 1
        else:
            check_lines.append(f"  ✓ Maneuvers consistent with {topology}")
    elif topology in ("intersection", "t_junction"):
        check_lines.append(f"  ℹ Junction topology — left/right/straight all valid")
    elif topology == "roundabout":
        check_lines.append(f"  ℹ Roundabout — straight = pass-through")
    else:
        check_lines.append(f"  ℹ Topology '{topology}' — maneuver check N/A")

    # Check: actors reference valid vehicles?
    for a in actors:
        av = a.get("affects_vehicle")
        if av and av not in vid_set:
            check_lines.append(f"  ✗ Actor '{a.get('actor_id')}' affects unknown '{av}'")
            issues_found += 1

    # Check: on-ramp flag consistency
    has_merge_maneuver = any(m in ("merge", "on_ramp_merge") for m in maneuvers)
    needs_ramp = spec_dict.get("needs_on_ramp", False)
    if needs_ramp and not has_merge_maneuver:
        check_lines.append("  ⚠ needs_on_ramp=True but no merge maneuver")
        issues_found += 1
    if has_merge_maneuver and not needs_ramp:
        check_lines.append("  ⚠ Merge maneuver but needs_on_ramp=False")
        issues_found += 1

    check_lines.append("")
    if issues_found == 0:
        check_lines.append("  ══ NO ISSUES DETECTED ══")
    else:
        check_lines.append(f"  ══ {issues_found} ISSUE(S) FOUND ══")

    bg_color = "#fff8e1" if issues_found > 0 else "#e8f5e9"
    ax_checks.set_facecolor(bg_color)
    ax_checks.text(
        0.03, 0.97, "\n".join(check_lines),
        transform=ax_checks.transAxes, va="top",
        fontsize=7.5, family="monospace",
    )
    ax_checks.set_title("Pipeline Reminders & Checks", fontsize=11, fontweight="bold")
    ax_checks.axis("off")

    # ====================== BOTTOM: Color Legend ======================
    ax_legend.axis("off")
    legend_items = []
    # Constraint type legend
    seen_types = set()
    for c in constraints:
        ct = c.get("type", "?")
        if ct not in seen_types:
            seen_types.add(ct)
            legend_items.append(
                Line2D([0], [0], color=_constraint_color(ct), lw=3,
                       label=ct.replace("_", " "))
            )
    # Special items
    legend_items.append(
        Line2D([0], [0], color="#d32f2f", lw=2, linestyle="--",
               label="Required relation (category rule)")
    )
    if legend_items:
        ax_legend.legend(
            handles=legend_items, loc="center", ncol=min(len(legend_items), 5),
            fontsize=9, frameon=True, title="Constraint Legend", title_fontsize=10,
        )

    fig.suptitle(
        f"Stage 1 — Schema Spec: {category}",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    _save_fig(fig, stage_dir / "visualization.png")


def _viz_geometry(stage_dir: Path, geometry_spec: Any, spec_dict: Dict[str, Any]) -> None:
    """
    STAGE 2 — Geometry extraction summary table + simple shape diagram.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_table, ax_diagram = axes

    # Table
    gs = geometry_spec
    rows = [
        ["topology", str(getattr(gs, "topology", "?"))],
        ["degree", str(getattr(gs, "degree", "?"))],
        ["needs_oncoming", str(getattr(gs, "needs_oncoming", False))],
        ["needs_on_ramp", str(getattr(gs, "needs_on_ramp", False))],
        ["needs_merge", str(getattr(gs, "needs_merge_onto_same_road", False))],
        ["needs_multi_lane", str(getattr(gs, "needs_multi_lane", False))],
        ["min_lane_count", str(getattr(gs, "min_lane_count", 1))],
        ["min_entry_runup_m", str(getattr(gs, "min_entry_runup_m", "?"))],
        ["min_exit_runout_m", str(getattr(gs, "min_exit_runout_m", "?"))],
    ]
    maneuvers = getattr(gs, "required_maneuvers", {})
    for k, v in maneuvers.items():
        rows.append([f"maneuver_{k}", str(v)])

    ax_table.axis("off")
    tbl = ax_table.table(
        cellText=rows,
        colLabels=["Property", "Value"],
        cellLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.4)
    ax_table.set_title("Geometry Spec", fontsize=12, fontweight="bold")

    # Shape diagram
    topo = str(getattr(gs, "topology", "unknown"))
    ax_diagram.set_xlim(-5, 5)
    ax_diagram.set_ylim(-5, 5)
    ax_diagram.set_aspect("equal")
    if "intersection" in topo:
        # Draw a + shape
        ax_diagram.plot([0, 0], [-4, 4], "k-", linewidth=4)
        ax_diagram.plot([-4, 4], [0, 0], "k-", linewidth=4)
        ax_diagram.annotate("N", (0, 4.3), ha="center", fontsize=10, fontweight="bold")
        ax_diagram.annotate("S", (0, -4.5), ha="center", fontsize=10, fontweight="bold")
        ax_diagram.annotate("E", (4.3, 0), ha="center", fontsize=10, fontweight="bold")
        ax_diagram.annotate("W", (-4.5, 0), ha="center", fontsize=10, fontweight="bold")
    elif "t_junction" in topo:
        ax_diagram.plot([-4, 4], [0, 0], "k-", linewidth=4)
        ax_diagram.plot([0, 0], [-4, 0], "k-", linewidth=4)
        ax_diagram.annotate("T-junction", (0, 2), ha="center", fontsize=11, fontweight="bold")
    elif "corridor" in topo or "highway" in topo:
        n_lanes = getattr(gs, "min_lane_count", 2)
        for i in range(n_lanes + 1):
            y = -2 + i * (4 / max(n_lanes, 1))
            style = "k--" if 0 < i < n_lanes else "k-"
            ax_diagram.plot([-4, 4], [y, y], style, linewidth=2)
        ax_diagram.annotate(f"{n_lanes}-lane", (0, 3), ha="center", fontsize=11, fontweight="bold")
    elif "roundabout" in topo:
        circle = plt.Circle((0, 0), 2, fill=False, edgecolor="black", linewidth=3)
        ax_diagram.add_patch(circle)
        for angle in [0, 90, 180, 270]:
            import math as _m
            dx = 3.5 * _m.cos(_m.radians(angle))
            dy = 3.5 * _m.sin(_m.radians(angle))
            ax_diagram.annotate(
                "", xy=(2.2 * _m.cos(_m.radians(angle)), 2.2 * _m.sin(_m.radians(angle))),
                xytext=(dx, dy),
                arrowprops=dict(arrowstyle="->", lw=2),
            )
    else:
        ax_diagram.text(0, 0, topo, ha="center", va="center", fontsize=14)
    ax_diagram.set_title(f"Expected Shape: {topo}", fontsize=12, fontweight="bold")
    ax_diagram.axis("off")

    fig.suptitle("Stage 2 — Geometry Extraction", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, stage_dir / "visualization.png")


def _viz_crop(
    stage_dir: Path,
    crops: List[Any],
    selected_crop: Any,
    crop_method: str,
    geometry_spec: Any,
    nodes_path: Path,
) -> None:
    """
    STAGE 3 — Town map with candidate crops, selected crop highlighted.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    def _crop_bounds(c: Any) -> Optional[Tuple[float, float, float, float]]:
        if c is None:
            return None
        crop_obj = getattr(c, "crop", c)
        try:
            if isinstance(crop_obj, (list, tuple)) and len(crop_obj) >= 4:
                xmin, xmax, ymin, ymax = (
                    float(crop_obj[0]),
                    float(crop_obj[1]),
                    float(crop_obj[2]),
                    float(crop_obj[3]),
                )
            else:
                xmin = float(getattr(crop_obj, "xmin"))
                xmax = float(getattr(crop_obj, "xmax"))
                ymin = float(getattr(crop_obj, "ymin"))
                ymax = float(getattr(crop_obj, "ymax"))
        except Exception:
            return None
        if not all(math.isfinite(v) for v in (xmin, xmax, ymin, ymax)):
            return None
        if xmin >= xmax or ymin >= ymax:
            return None
        return (xmin, xmax, ymin, ymax)

    sel_bounds = _crop_bounds(selected_crop)
    if sel_bounds is not None:
        focus_bounds = sel_bounds
    else:
        valid_bounds = [b for b in (_crop_bounds(c) for c in crops[:200]) if b is not None]
        if valid_bounds:
            focus_bounds = (
                min(b[0] for b in valid_bounds),
                max(b[1] for b in valid_bounds),
                min(b[2] for b in valid_bounds),
                max(b[3] for b in valid_bounds),
            )
        else:
            focus_bounds = None

    view_bounds: Optional[Tuple[float, float, float, float]] = None
    if focus_bounds is not None:
        xmin_f, xmax_f, ymin_f, ymax_f = focus_bounds
        span_x = max(1.0, xmax_f - xmin_f)
        span_y = max(1.0, ymax_f - ymin_f)
        margin_x = max(12.0, 0.20 * span_x)
        margin_y = max(12.0, 0.20 * span_y)
        vxmin = xmin_f - margin_x
        vxmax = xmax_f + margin_x
        vymin = ymin_f - margin_y
        vymax = ymax_f + margin_y
        # Hard cap view spans to avoid pathological huge render extents.
        max_span = 2500.0
        if (vxmax - vxmin) > max_span:
            cx = 0.5 * (vxmin + vxmax)
            vxmin, vxmax = cx - max_span / 2.0, cx + max_span / 2.0
        if (vymax - vymin) > max_span:
            cy = 0.5 * (vymin + vymax)
            vymin, vymax = cy - max_span / 2.0, cy + max_span / 2.0
        view_bounds = (vxmin, vxmax, vymin, vymax)

    # Load town nodes for background
    try:
        with open(nodes_path, "r") as f:
            town_data = json.load(f)
        # Plot road network lightly
        for seg_key, seg_data in town_data.items():
            if isinstance(seg_data, dict):
                points = seg_data.get("points", seg_data.get("polyline", []))
                if points and len(points) > 1:
                    xs = []
                    ys = []
                    for p in points:
                        if isinstance(p, (list, tuple)) and len(p) >= 2:
                            x, y = p[0], p[1]
                        elif isinstance(p, dict):
                            x, y = p.get("x"), p.get("y")
                        elif hasattr(p, "x") and hasattr(p, "y"):
                            x, y = p.x, p.y
                        else:
                            continue
                        try:
                            xf = float(x)
                            yf = float(y)
                        except Exception:
                            continue
                        if not (math.isfinite(xf) and math.isfinite(yf)):
                            continue
                        # Discard extreme outliers that can explode render bounds.
                        if abs(xf) > 1e5 or abs(yf) > 1e5:
                            continue
                        xs.append(xf)
                        ys.append(yf)
                    if len(xs) <= 1:
                        continue
                    if view_bounds is not None:
                        vxmin, vxmax, vymin, vymax = view_bounds
                        pad_x = 0.15 * (vxmax - vxmin)
                        pad_y = 0.15 * (vymax - vymin)
                        if max(xs) < (vxmin - pad_x) or min(xs) > (vxmax + pad_x):
                            continue
                        if max(ys) < (vymin - pad_y) or min(ys) > (vymax + pad_y):
                            continue
                    ax.plot(xs, ys, color="#ddd", linewidth=0.5, zorder=1)
    except Exception:
        pass  # town data format varies; best effort

    # Draw candidate crops as light rectangles
    for i, c in enumerate(crops[:200]):  # limit for performance
        crop_obj = getattr(c, "crop", c)
        xmin = getattr(crop_obj, "xmin", 0)
        xmax = getattr(crop_obj, "xmax", 0)
        ymin = getattr(crop_obj, "ymin", 0)
        ymax = getattr(crop_obj, "ymax", 0)
        rect = mpatches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=0.3, edgecolor="#aaa", facecolor="#eef", alpha=0.15, zorder=2,
        )
        ax.add_patch(rect)

    # Highlight selected crop
    sel = selected_crop
    sel_crop = getattr(sel, "crop", sel) if sel is not None else None
    if sel_crop is not None:
        xmin = getattr(sel_crop, "xmin", sel_crop[0] if isinstance(sel_crop, (list, tuple)) else 0)
        xmax = getattr(sel_crop, "xmax", sel_crop[1] if isinstance(sel_crop, (list, tuple)) else 0)
        ymin = getattr(sel_crop, "ymin", sel_crop[2] if isinstance(sel_crop, (list, tuple)) else 0)
        ymax = getattr(sel_crop, "ymax", sel_crop[3] if isinstance(sel_crop, (list, tuple)) else 0)
        rect = mpatches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=3, edgecolor="green", facecolor="#cfc", alpha=0.4, zorder=10,
        )
        ax.add_patch(rect)
        ax.text(
            (xmin + xmax) / 2, ymax + 2,
            f"SELECTED ({crop_method})",
            ha="center", fontsize=10, fontweight="bold", color="green", zorder=11,
        )

    # CSP fallback warning
    if crop_method == "fallback":
        ax.text(
            0.5, 0.02,
            "CSP FAILED: using smallest satisfying crop (FALLBACK)",
            transform=ax.transAxes, ha="center", fontsize=12, fontweight="bold",
            color="red", bbox=dict(facecolor="yellow", alpha=0.9),
        )

    # Annotation: why it satisfies
    gs = geometry_spec
    notes = []
    if getattr(gs, "needs_on_ramp", False):
        notes.append("needs_on_ramp")
    if getattr(gs, "needs_multi_lane", False):
        notes.append("needs_multi_lane")
    if getattr(gs, "needs_oncoming", False):
        notes.append("needs_oncoming")
    if getattr(gs, "needs_merge_onto_same_road", False):
        notes.append("needs_merge")
    if notes:
        ax.text(
            0.02, 0.98, "Spec requires: " + ", ".join(notes),
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8),
        )

    ax.set_title(f"Stage 3 — Crop Selection  ({len(crops)} candidates)", fontsize=14, fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    if view_bounds is not None:
        ax.set_xlim(view_bounds[0], view_bounds[1])
        ax.set_ylim(view_bounds[2], view_bounds[3])
        ax.set_autoscale_on(False)
    fig.tight_layout()
    _save_fig(fig, stage_dir / "visualization.png")


def _extract_path_xy(path: Any) -> Tuple[List[float], List[float]]:
    """Extract (xs, ys) lists from a LegalPath or list-of-segments.

    LegalPath has .segments (list of LaneSegment), each with .points (np.ndarray (N,2)).
    """
    xs: List[float] = []
    ys: List[float] = []
    segments = getattr(path, "segments", path)  # LegalPath.segments or bare list
    if not isinstance(segments, (list, tuple)):
        segments = [segments]
    for seg in segments:
        pts = getattr(seg, "points", None)
        if pts is None:
            continue
        # pts is np.ndarray (N,2) or list of tuples
        try:
            import numpy as np
            if isinstance(pts, np.ndarray):
                xs.extend(pts[:, 0].tolist())
                ys.extend(pts[:, 1].tolist())
                continue
        except Exception:
            pass
        for p in pts:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                xs.append(float(p[0]))
                ys.append(float(p[1]))
            elif hasattr(p, "x"):
                xs.append(float(p.x))
                ys.append(float(p.y))
    return xs, ys


def _viz_legal_paths(stage_dir: Path, legal_paths: List[Any], crop: Any) -> None:
    """
    STAGE 4 — All legal paths in the crop, color-coded with start/end markers.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    cmap = plt.cm.get_cmap("tab20", max(len(legal_paths), 1))

    for idx, path in enumerate(legal_paths):
        color = cmap(idx % 20)
        xs, ys = _extract_path_xy(path)
        if xs:
            ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.7, zorder=3)
            ax.plot(xs[0], ys[0], "o", color=color, markersize=5, zorder=4)
            ax.plot(xs[-1], ys[-1], "s", color=color, markersize=5, zorder=4)
            length = sum(
                math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
                for i in range(len(xs) - 1)
            )
            mid = len(xs) // 2
            ax.annotate(
                f"{length:.0f}m",
                (xs[mid], ys[mid]),
                fontsize=5, color=color, alpha=0.8,
            )

    # Draw crop boundary
    if crop is not None:
        xmin = getattr(crop, "xmin", 0)
        xmax = getattr(crop, "xmax", 0)
        ymin = getattr(crop, "ymin", 0)
        ymax = getattr(crop, "ymax", 0)
        rect = mpatches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor="black", facecolor="none", linestyle="--", zorder=1,
        )
        ax.add_patch(rect)

    if not legal_paths:
        ax.text(
            0.5, 0.5, "NO LEGAL PATHS GENERATED\nCheck connectivity",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=16, color="red", fontweight="bold",
        )

    ax.set_title(f"Stage 4 — Legal Paths ({len(legal_paths)} paths)", fontsize=14, fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    fig.tight_layout()
    _save_fig(fig, stage_dir / "visualization.png")


def _viz_picked_paths(
    stage_dir: Path,
    picked_json: Dict,
    legal_paths: List[Any],
    candidates: List[Dict],
    crop: Any,
    violations: Optional[List[str]] = None,
) -> None:
    """
    STAGE 5 — Selected paths bolded, unselected faded, vehicle IDs labeled.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    picked_names = set()
    picked_vehicles = {}  # name → vehicle_id
    picked_entries, _ = _extract_picked_entries_with_source(picked_json)
    for entry in picked_entries:
        pname = entry.get("path_name") or entry.get("name") or ""
        picked_names.add(pname)
        picked_vehicles[pname] = entry.get("vehicle", "?")

    # Draw all paths: faded unless picked
    cmap = plt.cm.get_cmap("Set1", max(len(picked_entries), 1))
    vehicle_color = {}
    color_idx = 0
    for entry in picked_entries:
        vid = entry.get("vehicle", "?")
        if vid not in vehicle_color:
            vehicle_color[vid] = cmap(color_idx % 9)
            color_idx += 1

    for idx, path in enumerate(legal_paths):
        xs, ys = _extract_path_xy(path)
        if not xs:
            continue

        # Check if this path is picked
        cand_name = candidates[idx]["name"] if idx < len(candidates) else ""
        if cand_name in picked_names:
            vid = picked_vehicles.get(cand_name, "?")
            color = vehicle_color.get(vid, "blue")
            ax.plot(xs, ys, color=color, linewidth=3, alpha=1.0, zorder=5)
            ax.plot(xs[0], ys[0], "o", color=color, markersize=8, zorder=6)
            ax.plot(xs[-1], ys[-1], "s", color=color, markersize=8, zorder=6)
            mid = len(xs) // 2
            ax.annotate(
                vid, (xs[mid], ys[mid]),
                fontsize=9, fontweight="bold", color=color,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor=color),
                zorder=7,
            )
        else:
            ax.plot(xs, ys, color="#ccc", linewidth=0.5, alpha=0.4, zorder=2)

    # Crop boundary
    if crop is not None:
        xmin = getattr(crop, "xmin", 0)
        xmax = getattr(crop, "xmax", 0)
        ymin = getattr(crop, "ymin", 0)
        ymax = getattr(crop, "ymax", 0)
        rect = mpatches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor="black", facecolor="none", linestyle="--", zorder=1,
        )
        ax.add_patch(rect)

    # Legend
    handles = [Line2D([0], [0], color=c, lw=3, label=v) for v, c in vehicle_color.items()]
    handles.append(Line2D([0], [0], color="#ccc", lw=1, label="(unselected)"))
    ax.legend(handles=handles, loc="upper right", fontsize=9)

    if violations:
        ax.text(
            0.01,
            0.99,
            "Constraint issues:\n" + "\n".join(f" - {v}" for v in violations[:8]),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="#b71c1c",
            bbox={"facecolor": "#fff8e1", "alpha": 0.95, "edgecolor": "#d32f2f"},
            zorder=20,
        )

    ax.set_title(f"Stage 5 — Picked Paths ({len(picked_entries)} vehicles)", fontsize=14, fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    fig.tight_layout()
    _save_fig(fig, stage_dir / "visualization.png")


def _viz_refined_paths(stage_dir: Path, refined_json: Dict, crop: Any) -> None:
    """
    STAGE 6 — Refined paths with spawn positions, speeds, lane-change emphasis.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    picked_entries, _ = _extract_picked_entries_with_source(refined_json)
    cmap = plt.cm.get_cmap("Set1", max(len(picked_entries), 1))
    polylines: List[Tuple[str, List[float], List[float]]] = []

    for idx, entry in enumerate(picked_entries):
        vid = entry.get("vehicle", f"V{idx + 1}")
        color = cmap(idx % 9)
        refined = entry.get("refined", {})
        speed = refined.get("speed_mps", "?")
        spawn_s = refined.get("spawn_s", refined.get("start_s", "?"))

        sig = entry.get("signature", {})
        xs, ys = _extract_xy_from_signature(sig)
        if not xs:
            for pp in entry.get("path_points", []):
                xy = _xy_from_point(pp)
                if xy is not None:
                    xs.append(xy[0])
                    ys.append(xy[1])
        if not xs:
            continue
        polylines.append((vid, xs, ys))

        # Detect lane-change-like segments via local turn sharpness.
        lc_segments = []
        for i in range(2, len(xs)):
            dx0 = xs[i - 1] - xs[i - 2]
            dy0 = ys[i - 1] - ys[i - 2]
            dx1 = xs[i] - xs[i - 1]
            dy1 = ys[i] - ys[i - 1]
            denom = (math.hypot(dx0, dy0) * math.hypot(dx1, dy1)) + 1e-6
            cosang = max(-1.0, min(1.0, (dx0 * dx1 + dy0 * dy1) / denom))
            turn_deg = abs(math.degrees(math.acos(cosang)))
            if turn_deg > 12.0:
                lc_segments.append(i)

        ax.plot(xs, ys, color=color, linewidth=2.5, alpha=0.9, zorder=5)
        for li in lc_segments:
            ax.plot(
                xs[max(0, li - 1) : li + 1],
                ys[max(0, li - 1) : li + 1],
                color=color,
                linewidth=5,
                alpha=0.65,
                zorder=6,
            )

        ax.plot(xs[0], ys[0], "*", color=color, markersize=14, zorder=7)
        ax.annotate(
            f"{vid}\nspd={speed} m/s\nspawn_s={spawn_s}",
            (xs[0], ys[0]),
            fontsize=8,
            fontweight="bold",
            color=color,
            xytext=(10, 10),
            textcoords="offset points",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": color},
            zorder=8,
        )

    # Overlap hotspots between refined trajectories.
    overlap_hits = _polyline_overlap_points(polylines, threshold_m=1.2)
    for ox, oy, label in overlap_hits:
        ax.plot(ox, oy, "o", color="#d50000", markersize=9, zorder=10)
        ax.annotate(
            label,
            (ox, oy),
            fontsize=7,
            color="#d50000",
            xytext=(6, -12),
            textcoords="offset points",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#d50000"},
            zorder=11,
        )

    # Crop boundary
    if crop is not None:
        xmin = getattr(crop, "xmin", 0)
        xmax = getattr(crop, "xmax", 0)
        ymin = getattr(crop, "ymin", 0)
        ymax = getattr(crop, "ymax", 0)
        rect = mpatches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor="black", facecolor="none", linestyle="--", zorder=1,
        )
        ax.add_patch(rect)

    ax.set_title("Stage 6 — Refined Paths (spawn + speed)", fontsize=14, fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    fig.tight_layout()
    _save_fig(fig, stage_dir / "visualization.png")


def _viz_placement(stage_dir: Path, scene_data: Dict) -> None:
    """
    STAGE 7 — Object placement with bounding boxes and role annotations.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))

    # Draw ego paths
    ego_picked = scene_data.get("ego_picked", [])
    cmap = plt.cm.get_cmap("Set1", max(len(ego_picked), 1))
    for idx, entry in enumerate(ego_picked):
        vid = entry.get("vehicle", f"V{idx + 1}")
        color = cmap(idx % 9)
        xs, ys = _extract_xy_from_signature(entry.get("signature", {}))
        if xs:
            ax.plot(xs, ys, color=color, linewidth=2, alpha=0.7, zorder=3)
            ax.annotate(
                vid,
                (xs[0], ys[0]),
                fontsize=8,
                fontweight="bold",
                color=color,
                zorder=6,
            )

    # Draw placed actors (scene format currently uses "actors")
    actors = scene_data.get("actors", []) or scene_data.get("npc_objects", [])
    actor_points: List[Tuple[str, float, float, float, str, str]] = []
    for obj in actors:
        spawn = obj.get("spawn", {}) if isinstance(obj.get("spawn"), dict) else {}
        x = obj.get("x", obj.get("world_x", spawn.get("x", 0.0)))
        y = obj.get("y", obj.get("world_y", spawn.get("y", 0.0)))
        yaw = obj.get("yaw", obj.get("yaw_deg", spawn.get("yaw_deg", 0.0)))
        entity_id = obj.get("id", obj.get("entity_id", "?"))
        kind = obj.get("semantic", obj.get("actor_kind", obj.get("kind", "?")))
        role = obj.get("category", obj.get("role", "npc"))
        actor_points.append((entity_id, float(x), float(y), float(yaw), str(kind), str(role)))

    # Bounding boxes + collision zones
    collision_warnings = []
    for entity_id, x, y, yaw, kind, role in actor_points:
        half_w = 1.0
        half_l = 2.5
        kind_l = kind.lower()
        if any(tok in kind_l for tok in ("cone", "barrel", "prop", "debris")):
            half_w, half_l = 0.4, 0.4
        elif any(tok in kind_l for tok in ("walker", "pedestrian", "cyclist")):
            half_w, half_l = 0.5, 0.8

        cos_y = math.cos(math.radians(yaw))
        sin_y = math.sin(math.radians(yaw))
        corners = [
            (x + half_l * cos_y - half_w * sin_y, y + half_l * sin_y + half_w * cos_y),
            (x + half_l * cos_y + half_w * sin_y, y + half_l * sin_y - half_w * cos_y),
            (x - half_l * cos_y + half_w * sin_y, y - half_l * sin_y - half_w * cos_y),
            (x - half_l * cos_y - half_w * sin_y, y - half_l * sin_y + half_w * cos_y),
        ]
        bbox_poly = plt.Polygon(
            corners,
            fill=True,
            facecolor="#ffb74d",
            edgecolor="black",
            alpha=0.65,
            zorder=5,
        )
        ax.add_patch(bbox_poly)

        # Collision-zone overlay (radius proxy)
        zone_radius = max(1.0, half_l + 0.8)
        zone = plt.Circle(
            (x, y),
            zone_radius,
            fill=False,
            edgecolor="#e53935",
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
            zorder=4,
        )
        ax.add_patch(zone)

        ax.annotate(
            f"{entity_id}\n{kind}\nrole={role}",
            (x, y),
            fontsize=6,
            ha="center",
            color="black",
            bbox={"facecolor": "white", "alpha": 0.8, "pad": 1},
            zorder=7,
        )

    # Pairwise collision-risk labels
    for i in range(len(actor_points)):
        id_i, x_i, y_i, _, _, _ = actor_points[i]
        for j in range(i + 1, len(actor_points)):
            id_j, x_j, y_j, _, _, _ = actor_points[j]
            d = math.hypot(x_i - x_j, y_i - y_j)
            if d < 2.0:
                collision_warnings.append(f"{id_i}<->{id_j} ({d:.2f}m)")
                ax.plot(
                    [x_i, x_j],
                    [y_i, y_j],
                    color="#d32f2f",
                    linewidth=1.4,
                    zorder=8,
                )

    # Failed placement attempts from actor_trace deltas
    actor_trace = scene_data.get("actor_trace", [])
    failed_attempt_note = ""
    if actor_trace:
        counts = [int(t.get("actor_count", 0)) for t in actor_trace if isinstance(t, dict)]
        if counts:
            dropped = sum(1 for i in range(1, len(counts)) if counts[i] < counts[i - 1])
            if dropped > 0:
                failed_attempt_note = f"failed_placement_steps={dropped}"

    if collision_warnings or failed_attempt_note:
        lines = []
        if collision_warnings:
            lines.extend([f"collision-risk: {w}" for w in collision_warnings[:8]])
        if failed_attempt_note:
            lines.append(failed_attempt_note)
        ax.text(
            0.01,
            0.99,
            "\n".join(lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="#b71c1c",
            bbox={"facecolor": "#fff3e0", "alpha": 0.95, "edgecolor": "#d32f2f"},
            zorder=20,
        )

    ax.set_title(f"Stage 7 — Object Placement ({len(actors)} objects)", fontsize=14, fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    fig.tight_layout()
    _save_fig(fig, stage_dir / "visualization.png")


def _viz_validation(stage_dir: Path, validation: SceneValidationResult) -> None:
    """
    STAGE 8 — Validation score + issue table.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [1, 2]})
    ax_score, ax_issues = axes

    # Score gauge
    score = validation.score
    color = "green" if score >= 0.7 else ("orange" if score >= 0.3 else "red")
    ax_score.barh(["Score"], [score], color=color, height=0.5, edgecolor="black")
    ax_score.set_xlim(0, 1)
    ax_score.text(
        score + 0.02, 0, f"{score:.2f}",
        va="center", fontsize=18, fontweight="bold", color=color,
    )
    result_txt = "PASS" if validation.is_valid else "FAIL"
    ax_score.set_title(f"Validation: {result_txt}", fontsize=16, fontweight="bold", color=color)

    # Issues table
    issues = validation.issues
    if issues:
        rows = []
        for iss in issues[:30]:  # cap display
            rows.append([
                iss.severity,
                str(iss.issue_type.name if hasattr(iss.issue_type, "name") else iss.issue_type),
                textwrap.shorten(iss.message, width=60),
            ])
        ax_issues.axis("off")
        tbl = ax_issues.table(
            cellText=rows,
            colLabels=["Severity", "Type", "Message"],
            cellLoc="left",
            loc="upper center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.scale(1.0, 1.3)
        # Color severity cells
        for i, row in enumerate(rows, start=1):
            cell = tbl[i, 0]
            if row[0] == "error":
                cell.set_facecolor("#ffcdd2")
            elif row[0] == "warning":
                cell.set_facecolor("#fff9c4")
    else:
        ax_issues.text(
            0.5, 0.5, "No issues found ✓",
            transform=ax_issues.transAxes, ha="center", va="center",
            fontsize=16, color="green", fontweight="bold",
        )
        ax_issues.axis("off")

    ax_issues.set_title(f"Issues ({len(issues)})", fontsize=12, fontweight="bold")
    fig.suptitle("Stage 8 — Scene Validation", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, stage_dir / "visualization.png")


def _viz_routes(stage_dir: Path, routes_dir: Path, scene_data: Optional[Dict] = None) -> None:
    """
    STAGE 9 — Route polylines from XML files plotted on top of scene paths.
    """
    import xml.etree.ElementTree as ET

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # Draw ego paths from scene as background ("before alignment")
    if scene_data:
        for idx, entry in enumerate(scene_data.get("ego_picked", [])):
            xs, ys = _extract_xy_from_signature(entry.get("signature", {}))
            if xs:
                ax.plot(xs, ys, color="#bdbdbd", linewidth=1, alpha=0.55, zorder=2)

    # Parse XML route files
    route_files = sorted(routes_dir.glob("*.xml"))
    cmap = plt.cm.get_cmap("Set1", max(len(route_files), 1))
    for ri, rf in enumerate(route_files):
        color = cmap(ri % 9)
        try:
            tree = ET.parse(rf)
            root = tree.getroot()
            xs, ys = [], []
            for wp in root.iter("waypoint"):
                x = float(wp.get("x", 0))
                y = float(wp.get("y", 0))
                xs.append(x)
                ys.append(y)
            if xs:
                ax.plot(xs, ys, "o-", color=color, linewidth=2, markersize=4, zorder=5, label=rf.stem)
                ax.plot(xs[0], ys[0], "^", color=color, markersize=10, zorder=6)
                ax.plot(xs[-1], ys[-1], "v", color=color, markersize=10, zorder=6)
        except Exception:
            pass

    if not route_files:
        ax.text(
            0.5, 0.5, "No route XML files found",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=14, color="red",
        )

    if route_files:
        ax.legend(loc="upper right", fontsize=8)
    ax.text(
        0.01,
        0.99,
        "Gray: scene path (before alignment)\nColor: XML route waypoints",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#9e9e9e"},
    )
    ax.set_title(f"Stage 9 — Routes ({len(route_files)} files)", fontsize=14, fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    fig.tight_layout()
    _save_fig(fig, stage_dir / "visualization.png")


# ###########################################################################
#  INVARIANT CHECKS
# ###########################################################################

class InvariantError(Exception):
    """Raised when a post-stage invariant fails."""
    pass


def _assert_schema(spec_dict: Dict[str, Any]) -> None:
    vehicles = spec_dict.get("ego_vehicles", [])
    vids = [v.get("vehicle_id") for v in vehicles]
    if len(vids) != len(set(vids)):
        raise InvariantError(f"Duplicate vehicle IDs: {vids}")
    if not vehicles:
        raise InvariantError("No ego vehicles defined")
    constraints = spec_dict.get("vehicle_constraints", [])
    for c in constraints:
        if not c.get("type") or not c.get("a") or not c.get("b"):
            raise InvariantError(f"Incomplete constraint: {c}")


def _assert_geometry(geometry_spec: Any) -> None:
    mlc = getattr(geometry_spec, "min_lane_count", None)
    if mlc is not None and mlc < 1:
        raise InvariantError(f"min_lane_count={mlc} < 1")
    maneuvers = getattr(geometry_spec, "required_maneuvers", {})
    for k, v in maneuvers.items():
        if v < 0:
            raise InvariantError(f"Negative maneuver count: {k}={v}")


def _assert_legal_paths(legal_paths: List[Any]) -> None:
    if not legal_paths:
        raise InvariantError("No legal paths generated")
    for i, path in enumerate(legal_paths):
        segs = getattr(path, "segments", path)  # LegalPath.segments or bare list
        if not segs:
            raise InvariantError(f"Path {i} is empty (no segments)")
        total_pts = sum(len(getattr(seg, "points", [])) for seg in segs)
        if total_pts < 2:
            raise InvariantError(f"Path {i} has only {total_pts} points (need ≥2)")


def _extract_picked_entries_with_source(picked_json: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    """
    Normalize stage-5/6 payload variants:
      - legacy: {"ego_picked": [{"vehicle": ..., "path_name": ...}, ...]}
      - current: {"picked": [{"vehicle": ..., "name": ...}, ...]}
    Returns (normalized_entries, source_key).
    """
    if not isinstance(picked_json, dict):
        return [], "none"

    source_key = "none"
    raw_entries: List[Any] = []
    for key in ("ego_picked", "picked"):
        val = picked_json.get(key)
        if isinstance(val, list) and val:
            source_key = key
            raw_entries = val
            break
    if source_key == "none":
        for key in ("ego_picked", "picked"):
            val = picked_json.get(key)
            if isinstance(val, list):
                source_key = key
                raw_entries = val
                break

    entries: List[Dict[str, Any]] = []
    for item in raw_entries:
        if not isinstance(item, dict):
            continue
        entry = dict(item)
        pname = entry.get("path_name") or entry.get("name")
        if pname:
            entry.setdefault("path_name", pname)
            entry.setdefault("name", pname)
        entries.append(entry)
    return entries, source_key


def _assert_picked_paths(picked_json: Dict) -> None:
    entries, _ = _extract_picked_entries_with_source(picked_json)
    if not entries:
        raise InvariantError("No paths were picked for any vehicle")
    for entry in entries:
        if not (entry.get("path_name") or entry.get("name")):
            raise InvariantError(f"Vehicle {entry.get('vehicle')} has no path_name")


def _assert_placement(scene_data: Dict, crop: Any) -> None:
    ego = scene_data.get("ego_picked", [])
    if not ego:
        raise InvariantError("scene_objects has no ego_picked")
    # Check NPC bounding box overlaps (simplified: center-distance)
    npcs = scene_data.get("npc_objects", [])
    for i, a in enumerate(npcs):
        ax_val = a.get("x", a.get("world_x", 0))
        ay_val = a.get("y", a.get("world_y", 0))
        for j, b in enumerate(npcs):
            if j <= i:
                continue
            bx_val = b.get("x", b.get("world_x", 0))
            by_val = b.get("y", b.get("world_y", 0))
            dist = math.hypot(ax_val - bx_val, ay_val - by_val)
            if dist < 0.3:
                raise InvariantError(
                    f"NPC objects {a.get('entity_id')} and {b.get('entity_id')} "
                    f"overlap (dist={dist:.2f}m)"
                )


def _assert_validation(validation: SceneValidationResult) -> None:
    if not (0.0 <= validation.score <= 1.0):
        raise InvariantError(f"Validation score {validation.score} not in [0, 1]")
    if math.isnan(validation.score):
        raise InvariantError("Validation score is NaN")


def _assert_routes(routes_dir: Path) -> None:
    xml_files = list(routes_dir.glob("*.xml"))
    if not xml_files:
        raise InvariantError("No route XML files generated")
    import xml.etree.ElementTree as ET
    for rf in xml_files:
        tree = ET.parse(rf)
        root = tree.getroot()
        waypoints = list(root.iter("waypoint"))
        if not waypoints:
            raise InvariantError(f"Route {rf.name} has no waypoints")


# ###########################################################################
#  STAGE FUNCTIONS
# ###########################################################################

def _run_stage(
    stage: Stage,
    stage_dir: Path,
    func,
    ctx: PipelineContext,
    *args, **kwargs,
) -> Tuple[bool, Any]:
    """
    Generic stage wrapper:
      1. Save input state
      2. Execute stage function
      3. Save output state
      4. Generate visualization
      5. On error, save traceback + abort

    Returns (success, result).
    """
    stage_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    try:
        result = func(ctx, stage_dir, *args, **kwargs)
        elapsed = time.time() - t0
        ctx.stage_results.append(StageResult(stage=stage.value, success=True, elapsed_s=elapsed))
        return True, result
    except Exception as exc:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        ctx.stage_results.append(StageResult(
            stage=stage.value, success=False, elapsed_s=elapsed,
            error=str(exc), traceback=tb,
        ))
        _write_text(stage_dir / "debug.txt", f"STAGE FAILED: {stage.value}\n\n{tb}")
        output_path = stage_dir / "output.json"
        if not output_path.exists():
            _write_json(output_path, {"error": str(exc), "stage": stage.value})
        else:
            _write_json(stage_dir / "error.json", {"error": str(exc), "stage": stage.value})
        return False, None


# ---- Stage 1: Schema ----

def stage_generate_schema(ctx: PipelineContext, stage_dir: Path) -> Dict[str, Any]:
    """Generate structured scenario spec via LLM."""
    _write_json(stage_dir / "input.json", {"category": ctx.category, "seed": ctx.seed})

    config = SchemaGenerationConfig()
    generator = SchemaScenarioGenerator(config=config, model=ctx.model, tokenizer=ctx.tokenizer)

    spec, errors, warnings = generator.generate_spec(
        ctx.category,
        stats={},
        debug_dir=stage_dir,
    )
    if spec is None:
        raise RuntimeError(f"Schema generation failed: {'; '.join(errors)}")

    ctx.spec = spec
    ctx.spec_dict = spec_to_dict(spec)
    # Category policy is authoritative for placement constraints.
    if ctx.cat_info is not None:
        try:
            ctx.spec_dict["allow_static_props"] = bool(ctx.cat_info.allow_static_props)
        except Exception:
            pass
    # If static props are disallowed, drop static-style actors at schema level so
    # downstream stages do not carry contradictory expectations.
    if not bool(ctx.spec_dict.get("allow_static_props", True)):
        actors = ctx.spec_dict.get("actors", [])
        if isinstance(actors, list):
            filtered: List[Any] = []
            removed = 0
            static_kind_tokens = (
                "static",
                "parked",
                "barrier",
                "cone",
                "debris",
                "obstacle",
                "prop",
            )
            static_kind_exact = {
                "parked_vehicle",
                "static_prop",
                "construction_cone",
                "construction_barrier",
                "traffic_cone",
            }
            for actor in actors:
                if not isinstance(actor, dict):
                    filtered.append(actor)
                    continue
                kind = str(actor.get("kind", "")).strip().lower()
                motion = str(actor.get("motion", "")).strip().lower()
                is_static_kind = kind in static_kind_exact or any(tok in kind for tok in static_kind_tokens)
                is_static_motion = motion in {"static", "stopped"}
                if is_static_kind or is_static_motion:
                    removed += 1
                    continue
                filtered.append(actor)
            if removed > 0:
                ctx.spec_dict["actors"] = filtered
    ctx.schema_text = json.dumps(ctx.spec_dict, indent=2, sort_keys=True)

    output = {
        "spec": ctx.spec_dict,
        "errors": errors,
        "warnings": warnings,
    }
    _write_json(stage_dir / "output.json", output)
    _write_text(stage_dir / "debug.txt", f"Schema generated for {ctx.category}\n"
                f"Vehicles: {len(ctx.spec_dict.get('ego_vehicles', []))}\n"
                f"Constraints: {len(ctx.spec_dict.get('vehicle_constraints', []))}\n"
                f"Actors: {len(ctx.spec_dict.get('actors', []))}\n"
                f"Errors: {errors}\nWarnings: {warnings}")

    # Invariant check
    _assert_schema(ctx.spec_dict)

    # Visualization
    _viz_schema(stage_dir, ctx.spec_dict, ctx.cat_info)

    return output


# ---- Stage 2: Geometry ----

def stage_extract_geometry(ctx: PipelineContext, stage_dir: Path) -> Dict[str, Any]:
    """Extract GeometrySpec from ScenarioSpec (deterministic)."""
    _write_json(stage_dir / "input.json", {"spec": ctx.spec_dict})

    ctx.geometry_spec = geometry_spec_from_scenario_spec(ctx.spec)

    gs = ctx.geometry_spec
    output = {
        "topology": getattr(gs, "topology", "?"),
        "degree": getattr(gs, "degree", 0),
        "needs_oncoming": getattr(gs, "needs_oncoming", False),
        "needs_on_ramp": getattr(gs, "needs_on_ramp", False),
        "needs_merge_onto_same_road": getattr(gs, "needs_merge_onto_same_road", False),
        "needs_multi_lane": getattr(gs, "needs_multi_lane", False),
        "min_lane_count": getattr(gs, "min_lane_count", 1),
        "required_maneuvers": getattr(gs, "required_maneuvers", {}),
    }
    _write_json(stage_dir / "output.json", output)
    _write_text(stage_dir / "debug.txt", json.dumps(output, indent=2))

    _assert_geometry(ctx.geometry_spec)
    _viz_geometry(stage_dir, ctx.geometry_spec, ctx.spec_dict)

    return output


# ---- Stage 3: Crop ----

def stage_crop_selection(ctx: PipelineContext, stage_dir: Path) -> Dict[str, Any]:
    """Select map crop region via CSP (or fallback)."""
    from generate_legal_paths import CropBox
    import run_crop_region_picker as crop_picker
    from pipeline.step_01_crop.scoring import crop_satisfies_spec

    spec = ctx.geometry_spec
    town = ctx.town
    nodes_path = REPO_ROOT / "scenario_generator" / "town_nodes" / f"{town}.json"
    if not nodes_path.exists():
        raise FileNotFoundError(f"Town nodes not found: {nodes_path}")

    _write_json(stage_dir / "input.json", {
        "town": town,
        "nodes_path": str(nodes_path),
        "geometry_spec": {
            "topology": getattr(spec, "topology", "?"),
            "needs_on_ramp": getattr(spec, "needs_on_ramp", False),
            "needs_multi_lane": getattr(spec, "needs_multi_lane", False),
        },
    })

    forced_crop_vals = _hardcoded_crop_override(ctx.category, town)
    if forced_crop_vals is not None:
        ctx.crops = []
        ctx.crop = CropBox(
            xmin=float(forced_crop_vals[0]),
            xmax=float(forced_crop_vals[1]),
            ymin=float(forced_crop_vals[2]),
            ymax=float(forced_crop_vals[3]),
        )
        ctx.crop_vals = list(forced_crop_vals)
        ctx.crop_assignment_method = "hardcoded"

        output = {
            "crop": list(forced_crop_vals),
            "method": "hardcoded",
            "total_candidates": 0,
            "hardcoded_reason": f"{ctx.category} fixed T-junction crop in {town}",
        }
        _write_json(stage_dir / "output.json", output)
        _write_text(
            stage_dir / "debug.txt",
            f"Crop selected via hardcoded override\nTown: {town}\nCrop: {forced_crop_vals}\n"
            f"Reason: {ctx.category} fixed T-junction crop",
        )
        try:
            _viz_crop(stage_dir, [], forced_crop_vals, "hardcoded", ctx.geometry_spec, nodes_path)
        except Exception as exc:
            output["viz_warning"] = f"Crop visualization failed (non-fatal): {exc}"
            _write_json(stage_dir / "output.json", output)
        return output

    is_roundabout = (
        getattr(spec, "topology", "") == "roundabout"
        or (ctx.cat_info and ctx.cat_info.map.topology == TopologyType.ROUNDABOUT)
    )
    is_two_lane_corridor = (
        getattr(spec, "topology", "") == "two_lane_corridor"
        or (ctx.cat_info and ctx.cat_info.map.topology == TopologyType.TWO_LANE_CORRIDOR)
    )

    if is_roundabout:
        roundabout_crop = crop_picker.build_roundabout_crop_for_town(
            town_name=town, town_json_path=str(nodes_path),
            min_path_len=20.0, max_paths=100, max_depth=8,
        )
        if roundabout_crop is None:
            raise RuntimeError("Failed to build roundabout crop for Town03")
        crops = [roundabout_crop]
    elif is_two_lane_corridor:
        crops = crop_picker.build_corridor_candidate_crops_for_town(
            town_name=town, town_json_path=str(nodes_path),
            min_corridor_length=80.0, crop_width=20.0,
            min_path_len=15.0, max_paths=100, max_depth=5,
        )
        if not crops:
            crops = crop_picker.build_candidate_crops_for_town(
                town_name=town, town_json_path=str(nodes_path),
                radii=[45.0, 55.0, 65.0], min_path_len=22.0, max_paths=80, max_depth=8,
            )
    else:
        crops = crop_picker.build_candidate_crops_for_town(
            town_name=town, town_json_path=str(nodes_path),
            radii=[45.0, 55.0, 65.0], min_path_len=22.0, max_paths=80, max_depth=8,
        )

    if not crops:
        raise RuntimeError(f"No candidate crops found for {town}")

    ctx.crops = crops

    # CSP assignment
    scenario = crop_picker.Scenario(sid="debug_run", text=ctx.schema_text or "")
    res = crop_picker.solve_assignment(
        scenarios=[scenario],
        specs={"debug_run": spec},
        crops=crops,
        domain_k=50, capacity_per_crop=10,
        reuse_weight=4000.0, junction_penalty=25000.0, log_every=0,
    )
    assignments = res.detailed.get("assignments", {})

    crop_method = "csp"
    selected_crop_obj = None
    if "debug_run" not in assignments:
        # Fallback
        crop_method = "fallback"
        satisfying = [c for c in crops if crop_satisfies_spec(spec, c)]
        if not satisfying:
            raise RuntimeError("No crops satisfy geometry spec (CSP + fallback both failed)")
        selected_crop_obj = min(satisfying, key=lambda c: c.area)
        crop_vals = [
            selected_crop_obj.crop.xmin, selected_crop_obj.crop.xmax,
            selected_crop_obj.crop.ymin, selected_crop_obj.crop.ymax,
        ]
    else:
        crop_vals = assignments["debug_run"]["crop"]
        # Find matching crop object for visualization
        for c in crops:
            if (abs(c.crop.xmin - crop_vals[0]) < 0.1 and abs(c.crop.xmax - crop_vals[1]) < 0.1 and
                abs(c.crop.ymin - crop_vals[2]) < 0.1 and abs(c.crop.ymax - crop_vals[3]) < 0.1):
                selected_crop_obj = c
                break

    ctx.crop = CropBox(xmin=crop_vals[0], xmax=crop_vals[1], ymin=crop_vals[2], ymax=crop_vals[3])
    ctx.crop_vals = crop_vals
    ctx.crop_assignment_method = crop_method

    output = {
        "crop": crop_vals,
        "method": crop_method,
        "total_candidates": len(crops),
    }
    _write_json(stage_dir / "output.json", output)
    _write_text(stage_dir / "debug.txt",
                f"Crop selected via {crop_method}\n"
                f"Total candidates: {len(crops)}\n"
                f"Crop: {crop_vals}")

    try:
        _viz_crop(stage_dir, crops, selected_crop_obj, crop_method, ctx.geometry_spec, nodes_path)
    except Exception as exc:
        viz_warning = f"Crop visualization failed (non-fatal): {exc}"
        output["viz_warning"] = viz_warning
        _write_json(stage_dir / "output.json", output)
        with open(stage_dir / "debug.txt", "a", encoding="utf-8") as f:
            f.write(f"\n{viz_warning}\n")

    return output


# ---- Stage 4: Legal Paths ----

def stage_generate_legal_paths(ctx: PipelineContext, stage_dir: Path) -> Dict[str, Any]:
    """Enumerate legal paths in the selected crop."""
    from generate_legal_paths import (
        load_nodes, build_segments, crop_segments, build_connectivity,
        generate_legal_paths, build_path_signature, make_path_name,
        save_aggregated_signatures_json, save_prompt_file,
        build_segments_detailed_for_path,
    )
    from pipeline.step_02_legal_paths.segments import crop_segments_t_junction

    _write_json(stage_dir / "input.json", {"crop": ctx.crop_vals, "town": ctx.town})

    spec = ctx.geometry_spec
    nodes_path = REPO_ROOT / "scenario_generator" / "town_nodes" / f"{ctx.town}.json"

    is_roundabout = getattr(spec, "topology", "") == "roundabout"
    is_two_lane_corridor = getattr(spec, "topology", "") == "two_lane_corridor"
    is_t_junction = getattr(spec, "topology", "") == "t_junction"

    data = load_nodes(str(nodes_path))
    if is_roundabout:
        all_segments = build_segments(data, min_points=2)
    else:
        all_segments = build_segments(data)

    if is_t_junction:
        # Use T-junction specific cropping
        cropped = crop_segments_t_junction(
            all_segments, ctx.crop,
            junction_center=None,
            junction_radius=30.0,
        )
    else:
        cropped = crop_segments(all_segments, ctx.crop)

    if not cropped:
        raise RuntimeError("No segments found in crop region")

    ctx.cropped_segments = cropped

    adj = build_connectivity(
        cropped,
        connect_radius_m=6.0,
        connect_yaw_tol_deg=60.0,
        allow_cross_lane=is_roundabout,
        strict_endpoint_dist_m=4.5,
    )

    max_paths = 200 if getattr(spec, "needs_on_ramp", False) else 100
    max_depth = 8 if getattr(spec, "needs_on_ramp", False) else 5

    legal = generate_legal_paths(
        cropped, adj, ctx.crop,
        min_path_length=15.0 if is_two_lane_corridor else 20.0,
        max_paths=max_paths,
        max_depth=10 if is_roundabout else max_depth,
        allow_within_region_fallback=is_roundabout,
        corridor_mode=is_two_lane_corridor,
        roundabout_mode=is_roundabout,
        t_junction_mode=is_t_junction,
    )

    ctx.legal_paths = legal
    ctx.legal_prompt_path = stage_dir / "legal_paths_prompt.txt"
    ctx.legal_json_path = stage_dir / "legal_paths_detailed.json"

    # Build candidate list
    candidates = []
    for i, path in enumerate(legal):
        sig = build_path_signature(path)
        name = make_path_name(i, sig)
        sig["segments_detailed"] = build_segments_detailed_for_path(path, polyline_sample_n=10)
        candidates.append({"name": name, "signature": sig})
    ctx.candidates = candidates

    # Compute lane counts for prompt
    from collections import defaultdict
    lane_ids_by_road: Dict[int, set] = defaultdict(set)
    for s in cropped:
        try:
            lane_ids_by_road[int(s.road_id)].add(int(s.lane_id))
        except Exception:
            pass
    lane_counts_by_road = {rid: len(lanes) for rid, lanes in lane_ids_by_road.items()}

    params = {
        "max_yaw_diff_deg": 60.0,
        "connect_radius_m": 6.0,
        "min_path_length_m": 20.0,
        "max_paths": max_paths,
        "max_depth": max_depth,
        "turn_frame": "WORLD_FRAME",
    }

    save_prompt_file(
        str(ctx.legal_prompt_path), crop=ctx.crop,
        nodes_path=str(nodes_path), params=params, paths_named=candidates,
    )
    save_aggregated_signatures_json(
        str(ctx.legal_json_path), crop=ctx.crop,
        nodes_path=str(nodes_path), params=params, paths_named=candidates,
        lane_counts_by_road=lane_counts_by_road,
    )

    output = {
        "num_paths": len(legal),
        "num_segments_cropped": len(cropped),
        "path_names": [c["name"] for c in candidates[:20]],
    }
    _write_json(stage_dir / "output.json", output)
    _write_text(stage_dir / "debug.txt",
                f"Legal paths: {len(legal)}\n"
                f"Cropped segments: {len(cropped)}\n"
                f"Roundabout={is_roundabout}, Corridor={is_two_lane_corridor}, TJunction={is_t_junction}")

    _assert_legal_paths(legal)
    _viz_legal_paths(stage_dir, legal, ctx.crop)

    return output


# ---- Stage 5: Pick Paths ----

def stage_pick_paths(ctx: PipelineContext, stage_dir: Path) -> Dict[str, Any]:
    """LLM picks vehicle→path assignments."""
    from run_path_picker import pick_paths_with_model

    _write_json(stage_dir / "input.json", {
        "legal_paths_count": len(ctx.legal_paths),
        "category": ctx.category,
    })

    # Build prompt (same as pipeline_runner)
    prompt_text = ctx.legal_prompt_path.read_text(encoding="utf-8").strip()
    prompt_text += (
        "\n\nUSER SCENARIO SCHEMA (JSON):\n"
        + (ctx.schema_text or "")
        + "\nUse this structured schema (not prose) to choose ego paths and constraints.\n"
        + "(Only assign paths to moving vehicles; ignore static/parked props.)\n"
    )

    schema_constraints = _constraints_from_schema(ctx.spec_dict)

    # Required relations from category definition
    required_relations = []
    if ctx.cat_info and getattr(ctx.cat_info.rules, "required_relations", None):
        for rel in ctx.cat_info.rules.required_relations:
            required_relations.append({
                "entry_relation": rel.entry_relation,
                "first_maneuver": getattr(rel.first_maneuver, "value", rel.first_maneuver),
                "second_maneuver": getattr(rel.second_maneuver, "value", rel.second_maneuver),
                "exit_relation": rel.exit_relation,
                "entry_lane_relation": rel.entry_lane_relation,
                "exit_lane_relation": rel.exit_lane_relation,
            })

    require_straight = False
    if ctx.cat_info and ctx.cat_info.map.topology in {
        TopologyType.CORRIDOR, TopologyType.TWO_LANE_CORRIDOR, TopologyType.HIGHWAY
    }:
        require_straight = True

    require_on_ramp = bool(getattr(ctx.geometry_spec, "needs_on_ramp", False))

    # Per-scenario seed for this pick
    try:
        import torch
        torch.manual_seed(abs(hash("debug_run")) % (2 ** 31))
    except Exception:
        pass

    ctx.picked_paths_path = stage_dir / "picked_paths_detailed.json"
    pick_paths_with_model(
        prompt=prompt_text,
        aggregated_json=str(ctx.legal_json_path),
        out_picked_json=str(ctx.picked_paths_path),
        model=ctx.model,
        tokenizer=ctx.tokenizer,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.2,
        top_p=0.95,
        require_straight=require_straight,
        require_on_ramp=require_on_ramp,
        schema_constraints=schema_constraints,
        required_relations=required_relations,
    )

    picked_json = _read_json(ctx.picked_paths_path)
    picked_entries, picked_source = _extract_picked_entries_with_source(picked_json)
    output = {
        "ego_picked_count": len(picked_entries),
        "picked_count": len(picked_entries),
        "picked_source": picked_source,
        "vehicles": [e.get("vehicle") for e in picked_entries],
    }
    _write_json(stage_dir / "output.json", output)
    _write_text(stage_dir / "debug.txt",
                f"Picked {len(picked_entries)} vehicle→path assignments "
                f"(source={picked_source})")

    _assert_picked_paths(picked_json)
    _viz_picked_paths(stage_dir, picked_json, ctx.legal_paths, ctx.candidates, ctx.crop)

    return output


# ---- Stage 6: Refine Paths ----

def stage_refine_paths(ctx: PipelineContext, stage_dir: Path) -> Dict[str, Any]:
    """LLM + CSP refine spawn/speed parameters."""
    from run_path_refiner import refine_picked_paths_with_model

    _write_json(stage_dir / "input.json", {"picked_paths": str(ctx.picked_paths_path)})

    is_roundabout = (
        getattr(ctx.geometry_spec, "topology", "") == "roundabout"
        or (ctx.cat_info and ctx.cat_info.map.topology == TopologyType.ROUNDABOUT)
    )

    ctx.refined_paths_path = stage_dir / "picked_paths_refined.json"

    if is_roundabout:
        # Skip refiner for roundabouts — use picked paths directly
        import shutil
        shutil.copy2(str(ctx.picked_paths_path), str(ctx.refined_paths_path))
        _write_text(stage_dir / "debug.txt", "Roundabout — skipped refinement, using picked paths directly")
    else:
        parent_dir = REPO_ROOT / "scenario_generator"
        refine_picked_paths_with_model(
            picked_paths_json=str(ctx.picked_paths_path),
            description=ctx.schema_text or "",
            out_json=str(ctx.refined_paths_path),
            model=ctx.model,
            tokenizer=ctx.tokenizer,
            max_new_tokens=2048,
            carla_assets=str(parent_dir / "carla_assets.json"),
            schema_payload=ctx.spec_dict,
        )
        _write_text(stage_dir / "debug.txt", "Refinement complete")

    refined_json = _read_json(ctx.refined_paths_path)
    refined_entries, refined_source = _extract_picked_entries_with_source(refined_json)
    output = {
        "ego_count": len(refined_entries),
        "picked_source": refined_source,
        "roundabout_skip": is_roundabout,
    }
    _write_json(stage_dir / "output.json", output)

    _viz_refined_paths(stage_dir, refined_json, ctx.crop)

    return output


# ---- Stage 7: Object Placement ----

def stage_place_objects(ctx: PipelineContext, stage_dir: Path) -> Dict[str, Any]:
    """LLM + CSP place non-ego actors into the scene."""
    from run_object_placer import run_object_placer
    from types import SimpleNamespace

    refined_path = ctx.refined_paths_path or ctx.picked_paths_path
    _write_json(stage_dir / "input.json", {"refined_paths": str(refined_path)})

    parent_dir = REPO_ROOT / "scenario_generator"
    ctx.scene_json_path = stage_dir / "scene_objects.json"
    ctx.scene_png_path = stage_dir / "scene_objects.png"

    schema_entities = _entities_from_schema(ctx.spec_dict)
    nodes_path = REPO_ROOT / "scenario_generator" / "town_nodes"

    placer_args = SimpleNamespace(
        model="",
        picked_paths=str(refined_path),
        carla_assets=str(parent_dir / "carla_assets.json"),
        description=ctx.schema_text or "",
        out=str(ctx.scene_json_path),
        viz_out=str(ctx.scene_png_path),
        viz=True,
        viz_show=False,
        nodes_root=str(nodes_path),
        max_new_tokens=2048,
        do_sample=False,
        temperature=0.2,
        top_p=0.95,
        placement_mode="csp",
        schema_entities=schema_entities,
        scenario_spec=ctx.spec_dict,
    )

    stats_dict = {}
    placer_args.stats = stats_dict
    run_object_placer(placer_args, model=ctx.model, tokenizer=ctx.tokenizer)

    if not ctx.scene_json_path.exists():
        raise RuntimeError("scene_objects.json was not created by object placer")

    scene_data = _read_json(ctx.scene_json_path)
    output = {
        "ego_count": len(scene_data.get("ego_picked", [])),
        "npc_count": len(scene_data.get("npc_objects", [])),
        "repair_stats": stats_dict,
    }
    _write_json(stage_dir / "output.json", output)
    _write_text(stage_dir / "debug.txt",
                f"Placement complete\n"
                f"Ego vehicles: {output['ego_count']}\n"
                f"NPC objects: {output['npc_count']}\n"
                f"Repair stats: {json.dumps(stats_dict, indent=2)}")

    _assert_placement(scene_data, ctx.crop)
    _viz_placement(stage_dir, scene_data)

    return output


# ---- Stage 8: Validation ----

def stage_validate_scene(ctx: PipelineContext, stage_dir: Path) -> Dict[str, Any]:
    """Rule-based scene validation and scoring."""
    _write_json(stage_dir / "input.json", {"scene_path": str(ctx.scene_json_path)})

    validator = SceneValidator()
    validation = validator.validate_scene(
        scene_path=str(ctx.scene_json_path),
        scenario_text=ctx.schema_text or "",
        category=ctx.category,
        scenario_spec=ctx.spec_dict,
    )
    ctx.validation = validation

    issues_list = []
    for iss in validation.issues:
        issues_list.append({
            "severity": iss.severity,
            "type": str(iss.issue_type.name if hasattr(iss.issue_type, "name") else iss.issue_type),
            "message": iss.message,
            "expected": iss.expected,
            "actual": iss.actual,
            "suggestion": iss.suggestion,
        })

    output = {
        "is_valid": validation.is_valid,
        "score": validation.score,
        "expected_vehicles": validation.expected_vehicles,
        "actual_vehicles": validation.actual_vehicles,
        "paths_intersect": validation.paths_intersect,
        "issues_count": len(validation.issues),
        "errors_count": len(validation.get_errors()),
        "warnings_count": len(validation.get_warnings()),
        "issues": issues_list,
    }
    _write_json(stage_dir / "output.json", output)
    _write_text(stage_dir / "debug.txt", validation.summary())

    _assert_validation(validation)
    _viz_validation(stage_dir, validation)

    return output


# ---- Stage 9: Routes ----

def stage_generate_routes(ctx: PipelineContext, stage_dir: Path) -> Dict[str, Any]:
    """Convert scene to per-vehicle CARLA route XMLs (no CARLA server needed if align=False)."""
    from convert_scene_to_routes import convert_scene_to_routes

    _write_json(stage_dir / "input.json", {"scene_path": str(ctx.scene_json_path)})

    ctx.routes_dir = stage_dir / "routes"
    ctx.routes_dir.mkdir(parents=True, exist_ok=True)

    convert_scene_to_routes(
        scene_json_path=str(ctx.scene_json_path),
        output_dir=str(ctx.routes_dir),
        ego_num=None,
        align_routes=False,  # No CARLA server needed for debug runs
    )

    xml_files = sorted(ctx.routes_dir.glob("*.xml"))
    manifest_path = ctx.routes_dir / "actors_manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}

    output = {
        "route_files": [f.name for f in xml_files],
        "manifest": manifest,
    }
    _write_json(stage_dir / "output.json", output)
    _write_text(stage_dir / "debug.txt",
                f"Generated {len(xml_files)} route files\n"
                + "\n".join(f"  {f.name}" for f in xml_files))

    _assert_routes(ctx.routes_dir)

    scene_data = _read_json(ctx.scene_json_path) if ctx.scene_json_path.exists() else None
    _viz_routes(stage_dir, ctx.routes_dir, scene_data)

    return output


# ---- Stage 10: Final CARLA Validation ----

def stage_carla_validation(ctx: PipelineContext, stage_dir: Path) -> Dict[str, Any]:
    """Run final CARLA validation/repair checks on generated route bundle."""
    from carla_validation import (
        CarlaValidationConfig,
        parse_offset_csv,
        run_final_carla_validation,
        write_carla_validation_report,
    )

    if ctx.routes_dir is None:
        raise RuntimeError("Routes directory is missing; cannot run CARLA validation.")

    _write_json(
        stage_dir / "input.json",
        {
            "routes_dir": str(ctx.routes_dir),
            "carla_validate": bool(ctx.carla_validate),
            "carla_host": ctx.carla_host,
            "carla_port": int(ctx.carla_port),
            "carla_repair_max_attempts": int(ctx.carla_repair_max_attempts),
            "carla_repair_xy_offsets": str(ctx.carla_repair_xy_offsets),
            "carla_repair_z_offsets": str(ctx.carla_repair_z_offsets),
            "carla_align_before_validate": bool(ctx.carla_align_before_validate),
            "carla_require_risk": bool(ctx.carla_require_risk),
            "carla_validation_timeout": float(ctx.carla_validation_timeout),
        },
    )

    if not bool(ctx.carla_validate):
        output = {
            "passed": False,
            "gate_mode": "hard",
            "checks": {
                "xml_manifest_contract": False,
                "route_feasibility_grp": False,
                "spawn_all_actors": False,
                "constant_trajectory_risk_check": False,
                "baseline_route_follow": False,
            },
            "metrics": {
                "spawn_expected": 0,
                "spawn_actual": 0,
                "min_ttc_s": None,
                "near_miss": False,
                "route_completion_min": 0.0,
                "driving_score_min": 0.0,
            },
            "repairs": [],
            "final_routes_dir": str(ctx.routes_dir),
            "failure_reason": "carla_validation_disabled",
            "skipped": True,
        }
        ctx.carla_validation = output
        _write_json(stage_dir / "output.json", output)
        _write_text(stage_dir / "debug.txt", "CARLA validation disabled via --no-carla-validate.")
        return output

    cfg = CarlaValidationConfig(
        routes_dir=ctx.routes_dir,
        carla_host=str(ctx.carla_host),
        carla_port=int(ctx.carla_port),
        carla_validation_timeout=float(ctx.carla_validation_timeout),
        carla_require_risk=bool(ctx.carla_require_risk),
        carla_align_before_validate=bool(ctx.carla_align_before_validate),
        carla_repair_max_attempts=max(0, int(ctx.carla_repair_max_attempts)),
        carla_repair_xy_offsets=parse_offset_csv(
            ctx.carla_repair_xy_offsets,
            default=(0.0, 0.25, -0.25, 0.5, -0.5, 1.0, -1.0),
        ),
        carla_repair_z_offsets=parse_offset_csv(
            ctx.carla_repair_z_offsets,
            default=(0.0, 0.2, -0.2, 0.5, -0.5, 1.0),
        ),
    )
    output = run_final_carla_validation(cfg)
    ctx.carla_validation = output
    write_carla_validation_report(stage_dir / "output.json", output)
    _write_text(
        stage_dir / "debug.txt",
        json.dumps(
            {
                "passed": bool(output.get("passed", False)),
                "failure_reason": output.get("failure_reason"),
                "checks": output.get("checks", {}),
                "metrics": output.get("metrics", {}),
                "repairs": output.get("repairs", []),
            },
            indent=2,
            ensure_ascii=False,
        ),
    )

    if not bool(output.get("passed", False)):
        reason = str(output.get("failure_reason") or "carla_validation_failed")
        raise RuntimeError(f"Final CARLA validation failed (hard gate): {reason}")

    return output


# ###########################################################################
#  MAIN ORCHESTRATOR
# ###########################################################################

def run_pipeline_debug(
    category: str,
    stop_after_stage: Optional[str] = None,
    seed: Optional[int] = None,
    debug_root: Path = Path("debug_runs"),
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    town: Optional[str] = None,
    model_id: str = "Qwen/Qwen2.5-32B-Instruct-AWQ",
    carla_validate: bool = True,
    carla_host: str = "127.0.0.1",
    carla_port: int = 3000,
    carla_repair_max_attempts: int = 2,
    carla_repair_xy_offsets: str = "0.0,0.25,-0.25,0.5,-0.5,1.0,-1.0",
    carla_repair_z_offsets: str = "0.0,0.2,-0.2,0.5,-0.5,1.0",
    carla_align_before_validate: bool = False,
    carla_require_risk: bool = True,
    carla_validation_timeout: float = 180.0,
    start_carla: bool = False,
    carla_root: Optional[str] = None,
    carla_args: Optional[List[str]] = None,
    carla_startup_timeout: float = CARLA_STARTUP_TIMEOUT_S,
    carla_shutdown_timeout: float = CARLA_SHUTDOWN_TIMEOUT_S,
    carla_port_tries: int = CARLA_PORT_TRIES,
    carla_port_step: int = CARLA_PORT_STEP,
) -> Path:
    """
    Run the scenario pipeline with explicit stage boundaries and rich debug output.

    Parameters
    ----------
    category : str
        Scenario category name (e.g. "Highway On-Ramp Merge").
    stop_after_stage : str, optional
        Stage name to stop after (see Stage enum values).
        If None, runs all stages.
    seed : int, optional
        RNG seed for reproducibility.
    debug_root : Path
        Root directory for debug output.
    model : optional
        Pre-loaded HuggingFace model. If None, will load from model_id.
    tokenizer : optional
        Pre-loaded tokenizer. If None, will load from model_id.
    town : str, optional
        CARLA town name override. Auto-detected from category if None.
    model_id : str
        HuggingFace model ID (used only if model/tokenizer not provided).

    Returns
    -------
    Path
        The debug run directory.
    """
    global _active_carla_manager

    # Resolve stop stage (with aliases)
    stop_stage = _resolve_stop_stage(stop_after_stage)

    # Validate category
    available = get_available_categories()
    if category not in available:
        raise ValueError(f"Unknown category '{category}'. Available: {sorted(available)}")

    # Seed
    _set_seed(seed)

    # Create run directory (collision-safe for parallel workers)
    run_dir = _make_unique_run_dir(Path(debug_root), category, seed)

    print("=" * 70)
    print(f"  DEBUG PIPELINE — {category}")
    print(f"  Run dir: {run_dir}")
    print(f"  Seed: {seed}")
    print(f"  Stop after: {stop_after_stage or '(all stages)'}")
    print(f"  CARLA validate: {'on' if carla_validate else 'off'}")
    print(f"  Auto-start CARLA: {'on' if start_carla else 'off'}")
    if carla_validate:
        print(
            "  CARLA config: "
            f"{carla_host}:{carla_port} "
            f"repair_attempts={carla_repair_max_attempts} "
            f"align_before={carla_align_before_validate} "
            f"require_risk={carla_require_risk} "
            f"timeout={carla_validation_timeout:.1f}s"
        )
    print("=" * 70)

    # Build context
    cat_info = CATEGORY_DEFINITIONS.get(category)
    resolved_town = town or _resolve_town(category)

    # Load model if not provided
    loaded_local_model = False
    if model is None or tokenizer is None:
        print(f"[INIT] Loading model: {model_id}")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model.eval()
        loaded_local_model = True
        print("[INIT] Model loaded.")

    carla_manager: Optional[CarlaProcessManager] = None
    previous_active_carla = _active_carla_manager
    should_start_carla = bool(
        start_carla
        and carla_validate
        and (
            stop_stage is None
            or STAGE_ORDER.index(stop_stage) >= STAGE_ORDER.index(Stage.CARLA_VALIDATION)
        )
    )
    if start_carla and not should_start_carla:
        print(
            "[INFO] --start-carla requested but CARLA validation stage will not run "
            "(disabled or stop-after before carla_validation); skipping launch."
        )
    if should_start_carla:
        _install_carla_signal_handlers()
        resolved_carla_root = (
            Path(carla_root).expanduser().resolve()
            if carla_root
            else Path(os.environ.get("CARLA_ROOT", str(REPO_ROOT / "carla912"))).expanduser().resolve()
        )
        resolved_carla_args = _effective_carla_args(carla_args)
        if resolved_carla_args != list(carla_args or []):
            print(f"[INFO] Auto-added CARLA arg(s): {resolved_carla_args[len(list(carla_args or [])):]}")
        carla_manager = CarlaProcessManager(
            carla_root=resolved_carla_root,
            host=carla_host,
            port=int(carla_port),
            extra_args=resolved_carla_args,
            startup_timeout_s=float(carla_startup_timeout),
            shutdown_timeout_s=float(carla_shutdown_timeout),
            port_tries=int(carla_port_tries),
            port_step=int(carla_port_step),
        )
        _active_carla_manager = carla_manager
        resolved_port = carla_manager.start()
        if int(resolved_port) != int(carla_port):
            print(f"[INFO] CARLA validation port adjusted: {carla_port} -> {resolved_port}")
            carla_port = int(resolved_port)

    ctx = PipelineContext(
        category=category,
        seed=seed,
        debug_root=run_dir,
        town=resolved_town,
        cat_info=cat_info,
        model=model,
        tokenizer=tokenizer,
        carla_validate=bool(carla_validate),
        carla_host=str(carla_host),
        carla_port=int(carla_port),
        carla_repair_max_attempts=max(0, int(carla_repair_max_attempts)),
        carla_repair_xy_offsets=str(carla_repair_xy_offsets),
        carla_repair_z_offsets=str(carla_repair_z_offsets),
        carla_align_before_validate=bool(carla_align_before_validate),
        carla_require_risk=bool(carla_require_risk),
        carla_validation_timeout=float(carla_validation_timeout),
    )

    # Stage execution table
    stage_funcs = [
        (Stage.SCHEMA, stage_generate_schema),
        (Stage.GEOMETRY, stage_extract_geometry),
        (Stage.CROP, stage_crop_selection),
        (Stage.LEGAL_PATHS, stage_generate_legal_paths),
        (Stage.PICK_PATHS, stage_pick_paths),
        (Stage.REFINE_PATHS, stage_refine_paths),
        (Stage.PLACEMENT, stage_place_objects),
        (Stage.VALIDATION, stage_validate_scene),
        (Stage.ROUTES, stage_generate_routes),
        (Stage.CARLA_VALIDATION, stage_carla_validation),
    ]

    summary = {
        "category": category,
        "seed": seed,
        "town": resolved_town,
        "stop_after_stage": stop_after_stage,
        "run_dir": str(run_dir),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stages": [],
        "carla_process_started": bool(carla_manager is not None),
        "carla_host": str(carla_host),
        "carla_port": int(carla_port),
    }
    try:
        for stage, func in stage_funcs:
            stage_dir = run_dir / STAGE_DIR_PREFIX[stage]
            print(f"\n{'='*60}")
            print(f"  STAGE: {stage.value}  ({STAGE_DIR_PREFIX[stage]})")
            print(f"{'='*60}")

            ok, _result = _run_stage(stage, stage_dir, func, ctx)

            stage_record = {
                "stage": stage.value,
                "success": ok,
                "elapsed_s": ctx.stage_results[-1].elapsed_s,
            }
            if not ok:
                stage_record["error"] = ctx.stage_results[-1].error
                stage_record["traceback"] = ctx.stage_results[-1].traceback
                summary["stages"].append(stage_record)
                summary["failed_stage"] = stage.value
                summary["error_message"] = ctx.stage_results[-1].error
                if isinstance(ctx.carla_validation, dict):
                    summary["carla_validation_pass"] = bool(ctx.carla_validation.get("passed", False))
                    summary["carla_validation_reason"] = ctx.carla_validation.get("failure_reason")
                    summary["carla_validation_metrics"] = ctx.carla_validation.get("metrics", {})
                    summary["repairs_applied"] = ctx.carla_validation.get("repairs", [])
                summary["completed_at"] = datetime.now(timezone.utc).isoformat()
                _write_json(run_dir / "summary.json", summary)
                print(f"\n  ✗ STAGE {stage.value} FAILED: {ctx.stage_results[-1].error}")
                print(f"  See: {stage_dir / 'debug.txt'}")
                return run_dir

            summary["stages"].append(stage_record)
            elapsed = ctx.stage_results[-1].elapsed_s
            print(f"  ✓ {stage.value} completed in {elapsed:.1f}s")

            if stop_stage is not None and stage == stop_stage:
                print(f"\n  ■ Stopping after stage '{stage.value}' (as requested)")
                summary["stopped_after"] = stage.value
                break

        summary["completed_at"] = datetime.now(timezone.utc).isoformat()
        summary["all_stages_passed"] = all(s["success"] for s in summary["stages"])
        if ctx.validation:
            summary["validation_score"] = ctx.validation.score
            summary["validation_is_valid"] = ctx.validation.is_valid
        if isinstance(ctx.carla_validation, dict):
            summary["carla_validation_pass"] = bool(ctx.carla_validation.get("passed", False))
            summary["carla_validation_reason"] = ctx.carla_validation.get("failure_reason")
            summary["carla_validation_metrics"] = ctx.carla_validation.get("metrics", {})
            summary["repairs_applied"] = ctx.carla_validation.get("repairs", [])
        _write_json(run_dir / "summary.json", summary)

        print(f"\n{'='*70}")
        print(f"  PIPELINE {'COMPLETE' if summary.get('all_stages_passed') else 'INCOMPLETE'}")
        print(f"  Run dir: {run_dir}")
        print(f"  Summary: {run_dir / 'summary.json'}")
        if ctx.validation:
            print(f"  Validation score: {ctx.validation.score:.2f}")
        if isinstance(ctx.carla_validation, dict):
            cv_pass = bool(ctx.carla_validation.get("passed", False))
            cv_reason = ctx.carla_validation.get("failure_reason")
            print(f"  CARLA validation: {'PASS' if cv_pass else 'FAIL'}")
            if cv_reason:
                print(f"  CARLA failure reason: {cv_reason}")
        print(f"{'='*70}")

        return run_dir
    finally:
        if carla_manager is not None:
            try:
                carla_manager.stop()
            except Exception as exc:
                print(f"[WARN] Failed to stop CARLA cleanly: {exc}")
        _active_carla_manager = previous_active_carla
        if loaded_local_model:
            _release_worker_gpu_resources(model, tokenizer)


# ###########################################################################
#  MULTI-CATEGORY ORCHESTRATOR
# ###########################################################################


def _release_worker_gpu_resources(model: Optional[Any], tokenizer: Optional[Any]) -> None:
    """Best-effort GPU cleanup for worker exits."""
    try:
        del model
    except Exception:
        pass
    try:
        del tokenizer
    except Exception:
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


def _parallel_gpu_worker(
    worker_name: str,
    gpu_id: int,
    task_queue: Any,
    result_queue: Any,
    stop_after_stage: Optional[str],
    debug_root: str,
    town: Optional[str],
    model_id: str,
    carla_validate: bool,
    carla_host: str,
    carla_port: int,
    carla_repair_max_attempts: int,
    carla_repair_xy_offsets: str,
    carla_repair_z_offsets: str,
    carla_align_before_validate: bool,
    carla_require_risk: bool,
    carla_validation_timeout: float,
    start_carla: bool,
    carla_root: Optional[str],
    carla_args: Optional[List[str]],
    carla_startup_timeout: float,
    carla_shutdown_timeout: float,
    carla_port_tries: int,
    carla_port_step: int,
) -> None:
    """
    Persistent worker bound to one GPU.
    Loads model once, processes many tasks, and releases GPU memory on exit.
    """
    global _active_carla_manager

    model = None
    tokenizer = None
    carla_manager: Optional[CarlaProcessManager] = None
    previous_active_carla = _active_carla_manager
    worker_carla_port = int(carla_port)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        resolved_stop_stage = _resolve_stop_stage(stop_after_stage)
        should_own_carla = bool(
            start_carla
            and carla_validate
            and (
                resolved_stop_stage is None
                or STAGE_ORDER.index(resolved_stop_stage) >= STAGE_ORDER.index(Stage.CARLA_VALIDATION)
            )
        )
        resolved_carla_root: Optional[Path] = None
        resolved_carla_args: List[str] = []

        def _start_or_restart_worker_carla(reason: str) -> None:
            global _active_carla_manager
            nonlocal carla_manager, worker_carla_port
            if not should_own_carla:
                return
            if resolved_carla_root is None:
                raise RuntimeError("worker CARLA root not initialized")
            if carla_manager is not None:
                try:
                    carla_manager.stop()
                except Exception:
                    pass
            mgr = CarlaProcessManager(
                carla_root=resolved_carla_root,
                host=carla_host,
                port=int(worker_carla_port),
                extra_args=resolved_carla_args,
                startup_timeout_s=float(carla_startup_timeout),
                shutdown_timeout_s=float(carla_shutdown_timeout),
                port_tries=int(carla_port_tries),
                port_step=int(carla_port_step),
            )
            _active_carla_manager = mgr
            worker_carla_port = int(mgr.start())
            carla_manager = mgr
            result_queue.put(
                {
                    "kind": "worker_carla_ready",
                    "worker": worker_name,
                    "gpu": gpu_id,
                    "carla_port": int(worker_carla_port),
                    "reason": str(reason),
                }
            )

        if should_own_carla:
            _install_carla_signal_handlers()
            resolved_carla_root = (
                Path(carla_root).expanduser().resolve()
                if carla_root
                else Path(os.environ.get("CARLA_ROOT", str(REPO_ROOT / "carla912"))).expanduser().resolve()
            )
            resolved_carla_args = _effective_carla_args(carla_args)
            _start_or_restart_worker_carla("initial")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model.eval()
        result_queue.put({"kind": "worker_ready", "worker": worker_name, "gpu": gpu_id})

        while True:
            task = task_queue.get()
            if task is None:
                break

            seq = int(task.get("seq", -1))
            category = str(task.get("category", ""))
            seed = task.get("seed")
            attempt = int(task.get("attempt", 0) or 0)
            max_infra_retries = 1 if should_own_carla else 0
            for infra_try in range(max_infra_retries + 1):
                if should_own_carla:
                    needs_restart = (
                        carla_manager is None
                        or (not carla_manager.is_running())
                        or (not _is_port_open(carla_host, int(worker_carla_port), timeout=1.0))
                    )
                    if needs_restart:
                        reason = "healthcheck_failed_before_task"
                        if infra_try > 0:
                            reason = "retry_healthcheck_failed"
                        result_queue.put(
                            {
                                "kind": "worker_carla_restart",
                                "worker": worker_name,
                                "gpu": gpu_id,
                                "reason": reason,
                            }
                        )
                        _start_or_restart_worker_carla(reason)

                try:
                    run_dir = run_pipeline_debug(
                        category=category,
                        stop_after_stage=stop_after_stage,
                        seed=seed,
                        debug_root=Path(debug_root),
                        model=model,
                        tokenizer=tokenizer,
                        town=town,
                        model_id=model_id,
                        carla_validate=carla_validate,
                        carla_host=carla_host,
                        carla_port=worker_carla_port,
                        carla_repair_max_attempts=carla_repair_max_attempts,
                        carla_repair_xy_offsets=carla_repair_xy_offsets,
                        carla_repair_z_offsets=carla_repair_z_offsets,
                        carla_align_before_validate=carla_align_before_validate,
                        carla_require_risk=carla_require_risk,
                        carla_validation_timeout=carla_validation_timeout,
                        start_carla=False if should_own_carla else start_carla,
                        carla_root=carla_root,
                        carla_args=carla_args,
                        carla_startup_timeout=carla_startup_timeout,
                        carla_shutdown_timeout=carla_shutdown_timeout,
                        carla_port_tries=carla_port_tries,
                        carla_port_step=carla_port_step,
                    )
                    summary_path = run_dir / "summary.json"
                    if summary_path.exists():
                        summary = _read_json(summary_path)
                    else:
                        summary = {}
                    failed_stage_error = _extract_failed_stage_error_from_run(run_dir, summary)
                    is_cuda_oom = _is_cuda_oom_message(failed_stage_error)
                    is_carla_conn_fail = bool(
                        summary.get("failed_stage") == Stage.CARLA_VALIDATION.value
                        and _is_carla_connection_error(failed_stage_error)
                    )
                    if is_carla_conn_fail and should_own_carla and infra_try < max_infra_retries:
                        result_queue.put(
                            {
                                "kind": "worker_carla_restart",
                                "worker": worker_name,
                                "gpu": gpu_id,
                                "reason": "post_validation_connect_failure",
                            }
                        )
                        _start_or_restart_worker_carla("post_validation_connect_failure")
                        continue
                    result_queue.put(
                        {
                            "kind": "task_result",
                            "seq": seq,
                            "attempt": attempt,
                            "category": category,
                            "seed": seed,
                            "run_dir": str(run_dir),
                            "all_passed": bool(summary.get("all_stages_passed", False)),
                            "failed_stage": summary.get("failed_stage"),
                            "score": summary.get("validation_score"),
                            "error": failed_stage_error,
                            "is_cuda_oom": is_cuda_oom,
                        }
                    )
                    if is_cuda_oom:
                        # Free transient allocations aggressively before next task.
                        _release_worker_gpu_resources(None, None)
                    break
                except Exception as exc:
                    err_text = str(exc)
                    if should_own_carla and _is_carla_connection_error(err_text) and infra_try < max_infra_retries:
                        result_queue.put(
                            {
                                "kind": "worker_carla_restart",
                                "worker": worker_name,
                                "gpu": gpu_id,
                                "reason": "exception_connect_failure",
                            }
                        )
                        _start_or_restart_worker_carla("exception_connect_failure")
                        continue
                    result_queue.put(
                        {
                            "kind": "task_result",
                            "seq": seq,
                            "attempt": attempt,
                            "category": category,
                            "seed": seed,
                            "all_passed": False,
                            "error": err_text,
                            "is_cuda_oom": _is_cuda_oom_message(err_text),
                        }
                    )
                    break
    except Exception as exc:
        result_queue.put(
            {
                "kind": "worker_fatal",
                "worker": worker_name,
                "gpu": gpu_id,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if carla_manager is not None:
            try:
                carla_manager.stop()
            except Exception:
                pass
        _active_carla_manager = previous_active_carla
        _release_worker_gpu_resources(model, tokenizer)
        result_queue.put({"kind": "worker_exit", "worker": worker_name, "gpu": gpu_id})


def _run_pipeline_debug_multi_parallel_subprocess(
    resolved_categories: List[str],
    stop_after_stage: Optional[str],
    seed_schedule: List[Optional[int]],
    debug_root: Path,
    town: Optional[str],
    model_id: str,
    schema_dashboard: bool,
    schema_dashboard_output: Optional[str],
    schema_dashboard_json: Optional[str],
    schema_dashboard_title: Optional[str],
    schema_dashboard_globs: Optional[List[str]],
    parallel_gpu_ids: List[int],
    workers_per_gpu: int,
    max_parallel_workers: Optional[int],
    max_oom_retries: int,
    carla_validate: bool,
    carla_host: str,
    carla_port: int,
    carla_repair_max_attempts: int,
    carla_repair_xy_offsets: str,
    carla_repair_z_offsets: str,
    carla_align_before_validate: bool,
    carla_require_risk: bool,
    carla_validation_timeout: float,
    start_carla: bool,
    carla_root: Optional[str],
    carla_args: Optional[List[str]],
    carla_startup_timeout: float,
    carla_shutdown_timeout: float,
    carla_port_tries: int,
    carla_port_step: int,
) -> List[Path]:
    """
    Fallback parallel engine that avoids multiprocessing semaphores.
    It launches subprocesses per task and schedules them across GPU slots.
    """
    defer_carla_validation = _should_defer_carla_validation(stop_after_stage, carla_validate)
    phase1_stop_after_stage = Stage.ROUTES.value if defer_carla_validation else stop_after_stage
    phase1_carla_validate = bool(carla_validate and not defer_carla_validation)
    phase1_start_carla = bool(start_carla and not defer_carla_validation)
    if defer_carla_validation:
        print(
            "[INFO] Deferred CARLA mode enabled: generating through routes first, "
            "then validating sequentially with a single CARLA instance."
        )

    worker_specs: List[Tuple[int, int]] = []
    for gpu_id in parallel_gpu_ids:
        for slot_idx in range(workers_per_gpu):
            worker_specs.append((gpu_id, slot_idx))
    if max_parallel_workers is not None and max_parallel_workers > 0:
        worker_specs = worker_specs[:max_parallel_workers]

    task_list: List[Dict[str, Any]] = []
    seq = 0
    for seed_value in seed_schedule:
        for cat in resolved_categories:
            task_list.append({"seq": seq, "category": cat, "seed": seed_value, "attempt": 0})
            seq += 1
    total_runs = len(task_list)
    if not worker_specs:
        raise ValueError("No available worker slots for subprocess fallback mode.")
    slot_count = min(len(worker_specs), total_runs)
    worker_specs = worker_specs[:slot_count]
    if phase1_start_carla:
        slot_worker_ports: Dict[int, int] = {
            slot_idx: _parallel_worker_carla_port(
                base_carla_port=carla_port,
                worker_index=slot_idx,
                carla_port_tries=carla_port_tries,
                carla_port_step=carla_port_step,
            )
            for slot_idx in range(slot_count)
        }
    else:
        slot_worker_ports = {slot_idx: int(carla_port) for slot_idx in range(slot_count)}

    print("\n[WARN] Falling back to subprocess parallel mode (multiprocessing queue unavailable).")
    print(
        f"       Jobs={total_runs}, slots={slot_count}, GPUs={parallel_gpu_ids}, "
        f"workers/GPU={workers_per_gpu}, oom_retries={max_oom_retries}, "
        f"auto_start_carla={'on' if phase1_start_carla else 'off'}"
    )

    pending = list(task_list)
    active: List[Dict[str, Any]] = []
    results_summary: List[Dict[str, Any]] = []
    run_dirs: List[Path] = []
    free_slots: List[int] = list(range(slot_count))

    def _spawn_task(task: Dict[str, Any], slot_idx: int) -> Dict[str, Any]:
        gpu_id, slot_id = worker_specs[slot_idx]
        selected_carla_port = int(slot_worker_ports[slot_idx]) if phase1_start_carla else int(carla_port)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--category",
            str(task["category"]),
            "--debug-root",
            str(debug_root),
            "--model",
            model_id,
            "--no-schema-dashboard",
        ]
        if task.get("seed") is not None:
            cmd.extend(["--seed", str(task["seed"])])
        if phase1_stop_after_stage:
            cmd.extend(["--stop-after", str(phase1_stop_after_stage)])
        if town:
            cmd.extend(["--town", str(town)])
        if not phase1_carla_validate:
            cmd.append("--no-carla-validate")
        cmd.extend(["--carla-host", str(carla_host)])
        cmd.extend(["--carla-port", str(selected_carla_port)])
        cmd.extend(["--carla-repair-max-attempts", str(int(carla_repair_max_attempts))])
        cmd.extend(["--carla-repair-xy-offsets", str(carla_repair_xy_offsets)])
        cmd.extend(["--carla-repair-z-offsets", str(carla_repair_z_offsets)])
        if carla_align_before_validate:
            cmd.append("--carla-align-before-validate")
        if not carla_require_risk:
            cmd.append("--no-carla-require-risk")
        cmd.extend(["--carla-validation-timeout", str(float(carla_validation_timeout))])
        if phase1_start_carla:
            launch_carla_args = list(carla_args or [])
            if not _has_graphics_adapter_arg(launch_carla_args):
                launch_carla_args.append(f"-graphicsadapter={gpu_id}")
            cmd.append("--start-carla")
            if carla_root:
                cmd.extend(["--carla-root", str(carla_root)])
            for extra in launch_carla_args:
                cmd.append(f"--carla-arg={str(extra)}")
            cmd.extend(["--carla-startup-timeout", str(float(carla_startup_timeout))])
            cmd.extend(["--carla-shutdown-timeout", str(float(carla_shutdown_timeout))])
            cmd.extend(["--carla-port-tries", str(int(carla_port_tries))])
            cmd.extend(["--carla-port-step", str(int(carla_port_step))])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(
            f"  [LAUNCH] job#{task['seq']} {task['category']} (seed={task.get('seed')}, "
            f"attempt={int(task.get('attempt', 0)) + 1}) on gpu={gpu_id} slot={slot_id + 1} "
            f"carla={selected_carla_port}/{selected_carla_port + CARLA_STREAM_PORT_OFFSET}/"
            f"{selected_carla_port + CARLA_TM_PORT_OFFSET}"
        )
        return {
            "task": task,
            "proc": proc,
            "gpu_id": gpu_id,
            "slot_id": slot_id,
            "slot_idx": slot_idx,
        }

    try:
        while pending or active:
            while pending and free_slots:
                slot_idx = free_slots.pop(0)
                active.append(_spawn_task(pending.pop(0), slot_idx))

            if not active:
                break

            time.sleep(0.2)
            finished_indices: List[int] = []
            for idx, item in enumerate(active):
                proc: subprocess.Popen = item["proc"]
                rc = proc.poll()
                if rc is None:
                    continue

                out_text = ""
                try:
                    stdout_text, _ = proc.communicate(timeout=1.0)
                    out_text = stdout_text or ""
                except Exception:
                    pass

                task = item["task"]
                run_dir = None
                m = re.search(r"Run dir:\s*(.+)", out_text)
                if m:
                    run_dir = m.group(1).strip()

                rec = {
                    "seq": int(task["seq"]),
                    "category": task["category"],
                    "seed": task.get("seed"),
                    "attempt": int(task.get("attempt", 0)),
                    "run_dir": run_dir,
                    "all_passed": False,
                    "failed_stage": None,
                    "score": None,
                    "error": None,
                    "is_cuda_oom": False,
                }

                if run_dir:
                    rd = Path(run_dir)
                    summary_path = rd / "summary.json"
                    if summary_path.exists():
                        summary = _read_json(summary_path)
                        rec["all_passed"] = bool(summary.get("all_stages_passed", False))
                        rec["failed_stage"] = summary.get("failed_stage")
                        rec["score"] = summary.get("validation_score")
                        rec["error"] = _extract_failed_stage_error_from_run(rd, summary)
                        rec["is_cuda_oom"] = _is_cuda_oom_message(rec.get("error"))

                if rc != 0:
                    lines = [ln for ln in out_text.strip().splitlines() if ln.strip()]
                    tail = "\n".join(lines[-8:]) if lines else ""
                    rec["error"] = f"subprocess exit {rc}" + (f"; tail:\n{tail}" if tail else "")
                    if _is_cuda_oom_message(out_text):
                        rec["is_cuda_oom"] = True

                if bool(rec.get("is_cuda_oom")) and int(rec.get("attempt", 0)) < max_oom_retries:
                    retry_task = {
                        "seq": int(task["seq"]),
                        "category": task["category"],
                        "seed": task.get("seed"),
                        "attempt": int(task.get("attempt", 0)) + 1,
                    }
                    pending.append(retry_task)
                    print(
                        f"  [RETRY] {task['category']} (seed={task.get('seed')}) "
                        f"CUDA OOM on attempt {int(task.get('attempt', 0)) + 1}; retrying..."
                    )
                else:
                    if rec.get("run_dir"):
                        run_dirs.append(Path(str(rec["run_dir"])))
                    results_summary.append(rec)
                status = "✓" if rec.get("all_passed") else "✗"
                score = f" score={rec['score']:.2f}" if rec.get("score") is not None else ""
                failed = f" failed={rec['failed_stage']}" if rec.get("failed_stage") else ""
                err = f" error={rec.get('error')}" if rec.get("error") else ""
                if not (bool(rec.get("is_cuda_oom")) and int(rec.get("attempt", 0)) < max_oom_retries):
                    print(
                        f"  [{len(results_summary)}/{total_runs}] {status} {rec['category']} "
                        f"(seed={rec.get('seed')}, attempt={int(rec.get('attempt', 0)) + 1}){score}{failed}{err}"
                    )
                finished_indices.append(idx)

            for idx in reversed(finished_indices):
                finished = active.pop(idx)
                free_slots.append(int(finished["slot_idx"]))
                free_slots.sort()

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Terminating subprocess GPU workers safely...")
        for item in active:
            proc = item["proc"]
            if proc.poll() is None:
                proc.terminate()
        for item in active:
            proc = item["proc"]
            try:
                proc.wait(timeout=10.0)
            except Exception:
                if proc.poll() is None:
                    proc.kill()
        raise
    finally:
        for item in active:
            proc = item["proc"]
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except Exception:
                    if proc.poll() is None:
                        proc.kill()

    results_summary.sort(key=lambda x: int(x.get("seq", 10**9)))

    if defer_carla_validation:
        _run_deferred_carla_validation_batch(
            results_summary=results_summary,
            carla_host=carla_host,
            carla_port=carla_port,
            carla_repair_max_attempts=carla_repair_max_attempts,
            carla_repair_xy_offsets=carla_repair_xy_offsets,
            carla_repair_z_offsets=carla_repair_z_offsets,
            carla_align_before_validate=carla_align_before_validate,
            carla_require_risk=carla_require_risk,
            carla_validation_timeout=carla_validation_timeout,
            start_carla=start_carla,
            carla_root=carla_root,
            carla_args=carla_args,
            carla_startup_timeout=carla_startup_timeout,
            carla_shutdown_timeout=carla_shutdown_timeout,
            carla_port_tries=carla_port_tries,
            carla_port_step=carla_port_step,
            preferred_gpu_ids=list(parallel_gpu_ids),
        )
        for rec in results_summary:
            _refresh_result_record_from_run(rec)

    print("\n" + "#" * 70)
    print("  MULTI-CATEGORY RESULTS")
    print("#" * 70)
    for r in results_summary:
        status = "✓" if r.get("all_passed") else "✗"
        seed_str = f"  seed={r.get('seed')}" if "seed" in r else ""
        score = f"  score={r['score']:.2f}" if r.get("score") is not None else ""
        failed = f"  failed={r['failed_stage']}" if r.get("failed_stage") else ""
        err = f"  error={r.get('error', '')}" if r.get("error") else ""
        warn = "  warning=carla_connect_failed_non_blocking" if r.get("carla_infra_warning") else ""
        print(f"  {status} {r['category']}{seed_str}{score}{failed}{err}{warn}")
    print()

    dashboard_info = None
    if schema_dashboard:
        dashboard_info = _build_integrated_schema_dashboard(
            run_dirs=run_dirs,
            debug_root=Path(debug_root),
            output_path=schema_dashboard_output,
            json_path=schema_dashboard_json,
            title=schema_dashboard_title,
            extra_globs=schema_dashboard_globs,
        )

    multi_summary_path = Path(debug_root) / f"multi_run_{_now_tag()}.json"
    _write_json(
        multi_summary_path,
        {
            "categories": resolved_categories,
            "seeds": seed_schedule,
            "results": [{k: v for k, v in r.items() if k != "seq"} for r in results_summary],
            "seed": seed_schedule[0] if seed_schedule else None,
            "stop_after_stage": stop_after_stage,
            "schema_dashboard": dashboard_info,
            "parallel": {
                "enabled": True,
                "mode": "subprocess_fallback",
                "gpu_ids": parallel_gpu_ids,
                "workers_per_gpu": workers_per_gpu,
                "max_parallel_workers": max_parallel_workers,
                "worker_count": slot_count,
                "max_oom_retries": max_oom_retries,
            },
            "deferred_carla_validation": {
                "enabled": bool(defer_carla_validation),
                "single_instance": bool(defer_carla_validation),
                "start_carla": bool(start_carla),
            },
        },
    )
    print(f"  Multi-run summary: {multi_summary_path}")
    return run_dirs


def _run_pipeline_debug_multi_parallel(
    resolved_categories: List[str],
    stop_after_stage: Optional[str],
    seed_schedule: List[Optional[int]],
    debug_root: Path,
    town: Optional[str],
    model_id: str,
    schema_dashboard: bool,
    schema_dashboard_output: Optional[str],
    schema_dashboard_json: Optional[str],
    schema_dashboard_title: Optional[str],
    schema_dashboard_globs: Optional[List[str]],
    parallel_gpu_ids: List[int],
    workers_per_gpu: int,
    max_parallel_workers: Optional[int],
    max_oom_retries: int,
    carla_validate: bool,
    carla_host: str,
    carla_port: int,
    carla_repair_max_attempts: int,
    carla_repair_xy_offsets: str,
    carla_repair_z_offsets: str,
    carla_align_before_validate: bool,
    carla_require_risk: bool,
    carla_validation_timeout: float,
    start_carla: bool,
    carla_root: Optional[str],
    carla_args: Optional[List[str]],
    carla_startup_timeout: float,
    carla_shutdown_timeout: float,
    carla_port_tries: int,
    carla_port_step: int,
) -> List[Path]:
    if workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be >= 1")

    defer_carla_validation = _should_defer_carla_validation(stop_after_stage, carla_validate)
    phase1_stop_after_stage = Stage.ROUTES.value if defer_carla_validation else stop_after_stage
    phase1_carla_validate = bool(carla_validate and not defer_carla_validation)
    phase1_start_carla = bool(start_carla and not defer_carla_validation)
    if defer_carla_validation:
        print(
            "[INFO] Deferred CARLA mode enabled: generating through routes first, "
            "then validating sequentially with a single CARLA instance."
        )

    task_list: List[Dict[str, Any]] = []
    seq = 0
    for seed_value in seed_schedule:
        for cat in resolved_categories:
            task_list.append({"seq": seq, "category": cat, "seed": seed_value})
            seq += 1
    total_runs = len(task_list)
    if total_runs == 0:
        return []

    worker_specs: List[Tuple[int, int]] = []
    for gpu_id in parallel_gpu_ids:
        for slot_idx in range(workers_per_gpu):
            worker_specs.append((gpu_id, slot_idx))

    if max_parallel_workers is not None and max_parallel_workers > 0:
        worker_specs = worker_specs[:max_parallel_workers]
    if not worker_specs:
        raise ValueError("Parallel mode requested but no workers could be created. Check GPU IDs and worker settings.")

    worker_count = min(len(worker_specs), total_runs)
    worker_specs = worker_specs[:worker_count]
    effective_carla_port = int(carla_port)
    if phase1_start_carla:
        selected_base = _find_parallel_carla_base_port(
            host=carla_host,
            preferred_port=int(carla_port),
            worker_count=worker_count,
            carla_port_tries=carla_port_tries,
            carla_port_step=carla_port_step,
        )
        if selected_base != int(carla_port):
            print(
                f"[INFO] Parallel CARLA base port adjusted: {carla_port} -> {selected_base} "
                f"(workers={worker_count})"
            )
        effective_carla_port = int(selected_base)

    print("\n" + "#" * 70)
    print(
        "  PARALLEL MULTI-CATEGORY DEBUG RUN — "
        f"{len(resolved_categories)} categories × {len(seed_schedule)} seed(s) "
        f"({total_runs} jobs, {worker_count} workers on GPUs {parallel_gpu_ids}, "
        f"{workers_per_gpu}/GPU, oom_retries={max_oom_retries}, "
        f"auto_start_carla={'on' if phase1_start_carla else 'off'})"
    )
    print("#" * 70)

    ctx = mp.get_context("spawn")
    try:
        task_queue = ctx.Queue()
        result_queue = ctx.Queue()
    except Exception as exc:
        print(f"[WARN] multiprocessing queue init failed: {exc}")
        return _run_pipeline_debug_multi_parallel_subprocess(
            resolved_categories=resolved_categories,
            stop_after_stage=stop_after_stage,
            seed_schedule=seed_schedule,
            debug_root=debug_root,
            town=town,
            model_id=model_id,
            schema_dashboard=schema_dashboard,
            schema_dashboard_output=schema_dashboard_output,
            schema_dashboard_json=schema_dashboard_json,
            schema_dashboard_title=schema_dashboard_title,
            schema_dashboard_globs=schema_dashboard_globs,
            parallel_gpu_ids=parallel_gpu_ids,
            workers_per_gpu=workers_per_gpu,
            max_parallel_workers=max_parallel_workers,
            max_oom_retries=max_oom_retries,
            carla_validate=carla_validate,
            carla_host=carla_host,
            carla_port=effective_carla_port,
            carla_repair_max_attempts=carla_repair_max_attempts,
            carla_repair_xy_offsets=carla_repair_xy_offsets,
            carla_repair_z_offsets=carla_repair_z_offsets,
            carla_align_before_validate=carla_align_before_validate,
            carla_require_risk=carla_require_risk,
            carla_validation_timeout=carla_validation_timeout,
            start_carla=start_carla,
            carla_root=carla_root,
            carla_args=carla_args,
            carla_startup_timeout=carla_startup_timeout,
            carla_shutdown_timeout=carla_shutdown_timeout,
            carla_port_tries=carla_port_tries,
            carla_port_step=carla_port_step,
        )
    for task in task_list:
        t = dict(task)
        t["attempt"] = 0
        task_queue.put(t)

    workers: List[Tuple[str, mp.Process]] = []
    for idx, (gpu_id, slot_idx) in enumerate(worker_specs):
        worker_carla_port = (
            _parallel_worker_carla_port(
                base_carla_port=effective_carla_port,
                worker_index=idx,
                carla_port_tries=carla_port_tries,
                carla_port_step=carla_port_step,
            )
            if phase1_start_carla
            else int(effective_carla_port)
        )
        worker_name = f"worker{idx + 1}/gpu{gpu_id}/slot{slot_idx + 1}"
        if phase1_start_carla:
            print(
                f"  [PORT] {worker_name} -> "
                f"{worker_carla_port}/{worker_carla_port + CARLA_STREAM_PORT_OFFSET}/"
                f"{worker_carla_port + CARLA_TM_PORT_OFFSET}"
            )
        proc = ctx.Process(
            target=_parallel_gpu_worker,
            args=(
                worker_name,
                gpu_id,
                task_queue,
                result_queue,
                phase1_stop_after_stage,
                str(debug_root),
                town,
                model_id,
                phase1_carla_validate,
                carla_host,
                worker_carla_port,
                carla_repair_max_attempts,
                carla_repair_xy_offsets,
                carla_repair_z_offsets,
                carla_align_before_validate,
                carla_require_risk,
                carla_validation_timeout,
                phase1_start_carla,
                carla_root,
                carla_args,
                carla_startup_timeout,
                carla_shutdown_timeout,
                carla_port_tries,
                carla_port_step,
            ),
            daemon=False,
        )
        proc.start()
        workers.append((worker_name, proc))

    run_dirs: List[Path] = []
    results_by_seq: Dict[int, Dict[str, Any]] = {}
    completed = 0
    ready_workers = set()

    try:
        while completed < total_runs:
            try:
                msg = result_queue.get(timeout=5.0)
            except queue_mod.Empty:
                alive = [p for _, p in workers if p.is_alive()]
                if not alive:
                    raise RuntimeError(f"All workers exited early (completed {completed}/{total_runs} jobs).")
                continue

            kind = str(msg.get("kind", ""))
            if kind == "worker_ready":
                worker_name = str(msg.get("worker"))
                ready_workers.add(worker_name)
                print(f"  [READY] {worker_name}")
                continue

            if kind == "worker_carla_ready":
                worker_name = str(msg.get("worker"))
                port_raw = msg.get("carla_port")
                reason = str(msg.get("reason", "")).strip()
                if port_raw is not None:
                    try:
                        world_port = int(port_raw)
                        print(
                            f"  [CARLA] {worker_name} using "
                            f"{world_port}/{world_port + CARLA_STREAM_PORT_OFFSET}/"
                            f"{world_port + CARLA_TM_PORT_OFFSET}"
                            + (f" ({reason})" if reason else "")
                        )
                    except Exception:
                        print(
                            f"  [CARLA] {worker_name} using port={port_raw}"
                            + (f" ({reason})" if reason else "")
                        )
                else:
                    print(f"  [CARLA] {worker_name} ready")
                continue

            if kind == "worker_carla_restart":
                worker_name = str(msg.get("worker"))
                reason = str(msg.get("reason", "unknown"))
                print(f"  [CARLA-RESTART] {worker_name}: {reason}")
                continue

            if kind == "worker_fatal":
                worker_name = str(msg.get("worker"))
                err = str(msg.get("error", "unknown error"))
                print(f"  [FATAL] {worker_name}: {err}")
                tb = str(msg.get("traceback", "")).strip()
                if tb:
                    print(tb)
                continue

            if kind == "worker_exit":
                worker_name = str(msg.get("worker"))
                if worker_name in ready_workers:
                    print(f"  [EXIT] {worker_name}")
                continue

            if kind != "task_result":
                continue

            rec = {
                "seq": int(msg.get("seq", -1)),
                "attempt": int(msg.get("attempt", 0) or 0),
                "category": msg.get("category"),
                "seed": msg.get("seed"),
                "run_dir": msg.get("run_dir"),
                "all_passed": bool(msg.get("all_passed", False)),
                "failed_stage": msg.get("failed_stage"),
                "score": msg.get("score"),
                "error": msg.get("error"),
                "is_cuda_oom": bool(msg.get("is_cuda_oom", False)),
            }
            if rec.get("is_cuda_oom") and rec["attempt"] < max_oom_retries:
                retry_task = {
                    "seq": rec["seq"],
                    "category": rec.get("category"),
                    "seed": rec.get("seed"),
                    "attempt": rec["attempt"] + 1,
                }
                task_queue.put(retry_task)
                print(
                    f"  [RETRY] {rec.get('category')} (seed={rec.get('seed')}) "
                    f"CUDA OOM on attempt {rec['attempt'] + 1}; re-queued."
                )
                continue

            if rec.get("run_dir"):
                run_dirs.append(Path(str(rec["run_dir"])))
            results_by_seq[int(rec["seq"])] = rec
            completed = len(results_by_seq)

            status = "✓" if rec.get("all_passed") else "✗"
            score = f" score={rec['score']:.2f}" if rec.get("score") is not None else ""
            fail = f" failed={rec['failed_stage']}" if rec.get("failed_stage") else ""
            err = f" error={rec['error']}" if rec.get("error") else ""
            print(
                f"  [{completed}/{total_runs}] {status} {rec.get('category')} "
                f"(seed={rec.get('seed')}, attempt={rec['attempt'] + 1}){score}{fail}{err}"
            )

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Terminating GPU workers safely...")
        for _, proc in workers:
            if proc.is_alive():
                proc.terminate()
        for _, proc in workers:
            proc.join(timeout=10.0)
        raise
    finally:
        for _ in range(worker_count):
            try:
                task_queue.put(None)
            except Exception:
                break
        for _, proc in workers:
            if proc.is_alive():
                proc.join(timeout=1.0)
            if proc.is_alive():
                proc.terminate()
        for _, proc in workers:
            proc.join(timeout=10.0)

    results_summary = list(results_by_seq.values())
    results_summary.sort(key=lambda x: int(x.get("seq", 10**9)))

    if defer_carla_validation:
        _run_deferred_carla_validation_batch(
            results_summary=results_summary,
            carla_host=carla_host,
            carla_port=carla_port,
            carla_repair_max_attempts=carla_repair_max_attempts,
            carla_repair_xy_offsets=carla_repair_xy_offsets,
            carla_repair_z_offsets=carla_repair_z_offsets,
            carla_align_before_validate=carla_align_before_validate,
            carla_require_risk=carla_require_risk,
            carla_validation_timeout=carla_validation_timeout,
            start_carla=start_carla,
            carla_root=carla_root,
            carla_args=carla_args,
            carla_startup_timeout=carla_startup_timeout,
            carla_shutdown_timeout=carla_shutdown_timeout,
            carla_port_tries=carla_port_tries,
            carla_port_step=carla_port_step,
            preferred_gpu_ids=list(parallel_gpu_ids),
        )
        for rec in results_summary:
            _refresh_result_record_from_run(rec)

    print("\n" + "#" * 70)
    print("  MULTI-CATEGORY RESULTS")
    print("#" * 70)
    for r in results_summary:
        status = "✓" if r.get("all_passed") else "✗"
        seed_str = f"  seed={r.get('seed')}" if "seed" in r else ""
        score = f"  score={r['score']:.2f}" if r.get("score") is not None else ""
        failed = f"  failed={r['failed_stage']}" if r.get("failed_stage") else ""
        err = f"  error={r.get('error', '')}" if r.get("error") else ""
        warn = "  warning=carla_connect_failed_non_blocking" if r.get("carla_infra_warning") else ""
        print(f"  {status} {r['category']}{seed_str}{score}{failed}{err}{warn}")
    print()

    dashboard_info = None
    if schema_dashboard:
        dashboard_info = _build_integrated_schema_dashboard(
            run_dirs=run_dirs,
            debug_root=Path(debug_root),
            output_path=schema_dashboard_output,
            json_path=schema_dashboard_json,
            title=schema_dashboard_title,
            extra_globs=schema_dashboard_globs,
        )

    multi_summary_path = Path(debug_root) / f"multi_run_{_now_tag()}.json"
    _write_json(
        multi_summary_path,
        {
            "categories": resolved_categories,
            "seeds": seed_schedule,
            "results": [{k: v for k, v in r.items() if k != "seq"} for r in results_summary],
            "seed": seed_schedule[0] if seed_schedule else None,
            "stop_after_stage": stop_after_stage,
            "schema_dashboard": dashboard_info,
            "parallel": {
                "enabled": True,
                "gpu_ids": parallel_gpu_ids,
                "workers_per_gpu": workers_per_gpu,
                "max_parallel_workers": max_parallel_workers,
                "worker_count": worker_count,
                "max_oom_retries": max_oom_retries,
            },
            "deferred_carla_validation": {
                "enabled": bool(defer_carla_validation),
                "single_instance": bool(defer_carla_validation),
                "start_carla": bool(start_carla),
            },
        },
    )
    print(f"  Multi-run summary: {multi_summary_path}")

    return run_dirs


def run_pipeline_debug_multi_target(
    categories: List[str],
    target_scenarios: int,
    stop_after_stage: Optional[str] = None,
    seed: Optional[int] = None,
    seeds: Optional[List[Optional[int]]] = None,
    debug_root: Path = Path("debug_runs"),
    town: Optional[str] = None,
    model_id: str = "Qwen/Qwen2.5-32B-Instruct-AWQ",
    schema_dashboard: bool = True,
    schema_dashboard_output: Optional[str] = None,
    schema_dashboard_json: Optional[str] = None,
    schema_dashboard_title: Optional[str] = None,
    schema_dashboard_globs: Optional[List[str]] = None,
    parallel_gpu_ids: Optional[List[int]] = None,
    workers_per_gpu: int = 2,
    max_parallel_workers: Optional[int] = None,
    max_oom_retries: int = 1,
    auto_workers_per_gpu: bool = True,
    carla_host: str = "127.0.0.1",
    carla_port: int = 3000,
    carla_repair_max_attempts: int = 2,
    carla_repair_xy_offsets: str = "0.0,0.25,-0.25,0.5,-0.5,1.0,-1.0",
    carla_repair_z_offsets: str = "0.0,0.2,-0.2,0.5,-0.5,1.0",
    carla_align_before_validate: bool = False,
    carla_require_risk: bool = True,
    carla_validation_timeout: float = 180.0,
    start_carla: bool = False,
    carla_root: Optional[str] = None,
    carla_args: Optional[List[str]] = None,
    carla_startup_timeout: float = CARLA_STARTUP_TIMEOUT_S,
    carla_shutdown_timeout: float = CARLA_SHUTDOWN_TIMEOUT_S,
    carla_port_tries: int = CARLA_PORT_TRIES,
    carla_port_step: int = CARLA_PORT_STEP,
    target_seed_round_batch: int = 4,
    target_max_generated: Optional[int] = None,
) -> List[Path]:
    """
    Target mode:
      1) Keep generating through routes until enough pre-CARLA valid runs exist.
      2) Run deferred CARLA validation on that pool.
      3) Count non-duplicate scenarios toward target:
           - high acceptance: CARLA gate pass
           - medium acceptance: CARLA gate failed (manual review bucket)
      4) Repeat until target reached.
    """
    if target_scenarios < 1:
        raise ValueError("target_scenarios must be >= 1")

    available = sorted(get_available_categories())
    resolved: List[str] = []
    for cat in categories:
        if str(cat).lower() == "all":
            resolved = available
            break
        resolved.append(str(cat))
    if not resolved:
        resolved = available

    for cat in resolved:
        if cat not in available:
            raise ValueError(f"Unknown category '{cat}'. Available: {available}")

    if stop_after_stage is not None:
        print("[WARN] --stop-after is ignored in target mode; generation is forced through 'routes'.")

    explicit_seed_queue: List[int] = []
    if seeds:
        for s in seeds:
            if s is None:
                continue
            explicit_seed_queue.append(int(s))
    elif seed is not None:
        explicit_seed_queue.append(int(seed))
    else:
        explicit_seed_queue.append(0)

    next_seed = int(max(explicit_seed_queue) + 1) if explicit_seed_queue else 0

    def _consume_seed_rounds(round_count: int) -> List[int]:
        nonlocal next_seed
        out: List[int] = []
        for _ in range(max(1, int(round_count))):
            if explicit_seed_queue:
                out.append(int(explicit_seed_queue.pop(0)))
            else:
                out.append(int(next_seed))
                next_seed += 1
        return out

    target_seed_round_batch = max(1, int(target_seed_round_batch))
    max_generated = (
        int(target_max_generated)
        if target_max_generated is not None and int(target_max_generated) > 0
        else None
    )

    generated_run_dirs: List[Path] = []
    generated_records: List[Dict[str, Any]] = []
    candidate_pool: List[Dict[str, Any]] = []
    candidate_seen = set()
    carla_evaluated_records: List[Dict[str, Any]] = []
    accepted_records: List[Dict[str, Any]] = []
    accepted_high_records: List[Dict[str, Any]] = []
    accepted_medium_records: List[Dict[str, Any]] = []
    accepted_run_dirs: List[Path] = []
    accepted_fingerprints: Dict[str, str] = {}
    cycle_index = 0

    print("\n" + "#" * 70)
    print(
        "  TARGET MODE — "
        f"goal={target_scenarios} accepted scenarios (high+medium), categories={resolved}, "
        f"seed_round_batch={target_seed_round_batch}"
    )
    print("#" * 70)

    while len(accepted_run_dirs) < target_scenarios:
        cycle_index += 1
        remaining = target_scenarios - len(accepted_run_dirs)
        print("\n" + "=" * 70)
        print(
            f"  TARGET CYCLE {cycle_index} — accepted={len(accepted_run_dirs)}/{target_scenarios}, "
            f"need_this_cycle={remaining}, pool={len(candidate_pool)}"
        )
        print("=" * 70)

        generation_loops = 0
        while len(candidate_pool) < remaining:
            generation_loops += 1
            if max_generated is not None and len(generated_run_dirs) >= max_generated:
                raise RuntimeError(
                    "Target mode stopped early: reached --target-max-generated limit "
                    f"({max_generated}) before collecting {target_scenarios} accepted scenarios."
                )

            rounds_needed = int(math.ceil((remaining - len(candidate_pool)) / max(1, len(resolved))))
            seed_rounds = max(1, min(target_seed_round_batch, rounds_needed))
            batch_seeds = _consume_seed_rounds(seed_rounds)
            print(
                f"[TARGET] Generation batch {generation_loops}: "
                f"{len(resolved)} category(ies) × {len(batch_seeds)} seed round(s) "
                f"(seeds {batch_seeds[0]}..{batch_seeds[-1]})"
            )

            batch_run_dirs = run_pipeline_debug_multi(
                categories=resolved,
                stop_after_stage=Stage.ROUTES.value,
                seed=batch_seeds[0] if batch_seeds else None,
                seeds=batch_seeds,
                debug_root=Path(debug_root),
                town=town,
                model_id=model_id,
                schema_dashboard=False,
                parallel_gpu_ids=parallel_gpu_ids,
                workers_per_gpu=workers_per_gpu,
                max_parallel_workers=max_parallel_workers,
                max_oom_retries=max_oom_retries,
                auto_workers_per_gpu=auto_workers_per_gpu,
                carla_validate=False,
                carla_host=carla_host,
                carla_port=carla_port,
                carla_repair_max_attempts=carla_repair_max_attempts,
                carla_repair_xy_offsets=carla_repair_xy_offsets,
                carla_repair_z_offsets=carla_repair_z_offsets,
                carla_align_before_validate=carla_align_before_validate,
                carla_require_risk=carla_require_risk,
                carla_validation_timeout=carla_validation_timeout,
                start_carla=False,
                carla_root=carla_root,
                carla_args=carla_args,
                carla_startup_timeout=carla_startup_timeout,
                carla_shutdown_timeout=carla_shutdown_timeout,
                carla_port_tries=carla_port_tries,
                carla_port_step=carla_port_step,
            )

            batch_prevalidated = 0
            for rd in batch_run_dirs:
                rd_path = Path(rd)
                generated_run_dirs.append(rd_path)
                rec = _refresh_result_record_from_run(
                    {
                        "seq": len(generated_records),
                        "attempt": 0,
                        "category": None,
                        "seed": None,
                        "run_dir": str(rd_path),
                        "all_passed": False,
                        "failed_stage": None,
                        "score": None,
                        "error": None,
                        "is_cuda_oom": False,
                    }
                )
                generated_records.append(dict(rec))
                if _is_fully_passed_record(rec):
                    key = str(Path(str(rec.get("run_dir"))).resolve())
                    if key not in candidate_seen:
                        candidate_seen.add(key)
                        candidate_pool.append(dict(rec))
                        batch_prevalidated += 1

            print(
                f"[TARGET] Generation batch result: prevalidated_added={batch_prevalidated}, "
                f"pool_now={len(candidate_pool)}, generated_total={len(generated_run_dirs)}"
            )

            if generation_loops >= 2000 and len(candidate_pool) < remaining:
                raise RuntimeError(
                    "Target mode generation did not accumulate enough prevalidated runs. "
                    "Aborting after 2000 generation loops to avoid an endless overnight run."
                )

        cycle_candidates = [candidate_pool.pop(0) for _ in range(min(remaining, len(candidate_pool)))]
        if not cycle_candidates:
            raise RuntimeError("Target mode internal error: no candidates available for CARLA validation.")

        preferred_gpu_ids = list(parallel_gpu_ids) if parallel_gpu_ids else None
        if preferred_gpu_ids is None:
            inferred = _infer_primary_visible_gpu_id()
            preferred_gpu_ids = [int(inferred)] if inferred is not None else None

        _run_deferred_carla_validation_batch(
            results_summary=cycle_candidates,
            carla_host=carla_host,
            carla_port=carla_port,
            carla_repair_max_attempts=carla_repair_max_attempts,
            carla_repair_xy_offsets=carla_repair_xy_offsets,
            carla_repair_z_offsets=carla_repair_z_offsets,
            carla_align_before_validate=carla_align_before_validate,
            carla_require_risk=carla_require_risk,
            carla_validation_timeout=carla_validation_timeout,
            start_carla=start_carla,
            carla_root=carla_root,
            carla_args=carla_args,
            carla_startup_timeout=carla_startup_timeout,
            carla_shutdown_timeout=carla_shutdown_timeout,
            carla_port_tries=carla_port_tries,
            carla_port_step=carla_port_step,
            preferred_gpu_ids=preferred_gpu_ids,
        )

        accepted_this_cycle = 0
        accepted_high_this_cycle = 0
        accepted_medium_this_cycle = 0
        for rec in cycle_candidates:
            _refresh_result_record_from_run(rec)
            carla_evaluated_records.append(dict(rec))

            rd_raw = rec.get("run_dir")
            if not rd_raw:
                continue
            rd = Path(str(rd_raw))

            failed_stage = str(rec.get("failed_stage") or "").strip()
            if failed_stage and failed_stage != Stage.CARLA_VALIDATION.value:
                reason = str(rec.get("failed_stage") or rec.get("error") or "validation_failed")
                _annotate_target_acceptance(
                    rd,
                    accepted=False,
                    reason=reason,
                    acceptance_level="rejected",
                    target_cycle=cycle_index,
                )
                continue

            carla_ok, carla_reason = _carla_gate_status_from_run(rd)
            rec["target_carla_passed"] = bool(carla_ok)
            rec["target_acceptance_level"] = "high" if carla_ok else "medium"

            geom_fp = _scenario_geometry_fingerprint(rd)
            duplicate_of = accepted_fingerprints.get(geom_fp)
            if duplicate_of:
                rec["target_duplicate"] = True
                rec["target_duplicate_of"] = duplicate_of
                print(
                    f"  [TARGET-REJECT] duplicate scenario: {rec.get('category')} seed={rec.get('seed')} "
                    f"run={rd.name} duplicate_of={Path(duplicate_of).name}"
                )
                _annotate_target_acceptance(
                    rd,
                    accepted=False,
                    reason="duplicate_scenario",
                    acceptance_level="rejected",
                    duplicate_of=duplicate_of,
                    geometry_fingerprint=geom_fp,
                    target_cycle=cycle_index,
                )
                continue

            accepted_fingerprints[geom_fp] = str(rd)
            accepted_records.append(dict(rec))
            accepted_run_dirs.append(rd)
            accepted_this_cycle += 1
            if carla_ok:
                accepted_high_records.append(dict(rec))
                accepted_high_this_cycle += 1
                acceptance_level = "high"
                acceptance_reason = "accepted"
            else:
                accepted_medium_records.append(dict(rec))
                accepted_medium_this_cycle += 1
                acceptance_level = "medium"
                acceptance_reason = str(carla_reason or "accepted_medium_carla_failed")
                print(
                    f"  [TARGET-MEDIUM] {rec.get('category')} seed={rec.get('seed')} run={rd.name} "
                    f"carla={acceptance_reason}"
                )
            _annotate_target_acceptance(
                rd,
                accepted=True,
                reason=acceptance_reason,
                acceptance_level=acceptance_level,
                geometry_fingerprint=geom_fp,
                target_cycle=cycle_index,
            )

        print(
            f"[TARGET] Cycle {cycle_index} complete: +{accepted_this_cycle} accepted "
            f"(high={accepted_high_this_cycle}, medium={accepted_medium_this_cycle}), "
            f"accepted_total={len(accepted_run_dirs)}/{target_scenarios}, pool_remaining={len(candidate_pool)}"
        )

    accepted_run_dirs = accepted_run_dirs[:target_scenarios]
    accepted_records = accepted_records[:target_scenarios]
    accepted_high_records = accepted_high_records[:target_scenarios]
    accepted_medium_records = accepted_medium_records[:target_scenarios]

    dashboard_info = None
    if schema_dashboard:
        dashboard_info = _build_integrated_schema_dashboard(
            run_dirs=generated_run_dirs,
            debug_root=Path(debug_root),
            output_path=schema_dashboard_output,
            json_path=schema_dashboard_json,
            title=schema_dashboard_title,
            extra_globs=schema_dashboard_globs,
        )

    target_summary_path = Path(debug_root) / f"target_run_{_now_tag()}.json"
    _write_json(
        target_summary_path,
        {
            "mode": "target_scenarios",
            "target_scenarios": int(target_scenarios),
            "accepted_count": len(accepted_run_dirs),
            "accepted_high_count": len(accepted_high_records),
            "accepted_medium_count": len(accepted_medium_records),
            "generated_count": len(generated_run_dirs),
            "evaluated_in_carla_count": len(carla_evaluated_records),
            "categories": resolved,
            "parallel_gpu_ids": list(parallel_gpu_ids) if parallel_gpu_ids else None,
            "target_seed_round_batch": int(target_seed_round_batch),
            "next_seed": int(next_seed),
            "accepted_run_dirs": [str(p) for p in accepted_run_dirs],
            "accepted_records": accepted_records,
            "accepted_high_records": accepted_high_records,
            "accepted_medium_records": accepted_medium_records,
            "carla_evaluated_records": carla_evaluated_records,
            "generated_records": generated_records,
            "schema_dashboard": dashboard_info,
        },
    )

    print("\n" + "#" * 70)
    print("  TARGET MODE COMPLETE")
    print("#" * 70)
    print(f"  Accepted scenarios: {len(accepted_run_dirs)} / {target_scenarios}")
    print(
        "  Acceptance mix: "
        f"high={len(accepted_high_records)} medium={len(accepted_medium_records)}"
    )
    print(f"  Generated scenarios: {len(generated_run_dirs)}")
    print(f"  Evaluated in CARLA: {len(carla_evaluated_records)}")
    print(f"  Target summary: {target_summary_path}")

    return accepted_run_dirs


def run_pipeline_debug_multi(
    categories: List[str],
    stop_after_stage: Optional[str] = None,
    seed: Optional[int] = None,
    seeds: Optional[List[Optional[int]]] = None,
    debug_root: Path = Path("debug_runs"),
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    town: Optional[str] = None,
    model_id: str = "Qwen/Qwen2.5-32B-Instruct-AWQ",
    schema_dashboard: bool = True,
    schema_dashboard_output: Optional[str] = None,
    schema_dashboard_json: Optional[str] = None,
    schema_dashboard_title: Optional[str] = None,
    schema_dashboard_globs: Optional[List[str]] = None,
    parallel_gpu_ids: Optional[List[int]] = None,
    workers_per_gpu: int = 2,
    max_parallel_workers: Optional[int] = None,
    max_oom_retries: int = 1,
    auto_workers_per_gpu: bool = True,
    carla_validate: bool = True,
    carla_host: str = "127.0.0.1",
    carla_port: int = 3000,
    carla_repair_max_attempts: int = 2,
    carla_repair_xy_offsets: str = "0.0,0.25,-0.25,0.5,-0.5,1.0,-1.0",
    carla_repair_z_offsets: str = "0.0,0.2,-0.2,0.5,-0.5,1.0",
    carla_align_before_validate: bool = False,
    carla_require_risk: bool = True,
    carla_validation_timeout: float = 180.0,
    start_carla: bool = False,
    carla_root: Optional[str] = None,
    carla_args: Optional[List[str]] = None,
    carla_startup_timeout: float = CARLA_STARTUP_TIMEOUT_S,
    carla_shutdown_timeout: float = CARLA_SHUTDOWN_TIMEOUT_S,
    carla_port_tries: int = CARLA_PORT_TRIES,
    carla_port_step: int = CARLA_PORT_STEP,
) -> List[Path]:
    """
    Run the debug pipeline for multiple categories/seeds.
    In sequential mode the model loads once; in parallel mode each worker loads once.

    Parameters
    ----------
    categories : list of str
        Category names.  Pass ``["all"]`` or an empty list to run every category.
    stop_after_stage, seed, seeds, debug_root, model, tokenizer, town, model_id
        Forwarded to :func:`run_pipeline_debug` per category.
    schema_dashboard
        If True, generate integrated pipeline dashboard after run completion.
    schema_dashboard_output, schema_dashboard_json, schema_dashboard_title, schema_dashboard_globs
        Optional dashboard configuration.
    parallel_gpu_ids, workers_per_gpu, max_parallel_workers
        Optional multi-process GPU parallelism controls.
    max_oom_retries
        Automatic retries for CUDA OOM task attempts in parallel mode.
    auto_workers_per_gpu
        If True, cap requested workers/GPU using live nvidia-smi memory to avoid oversubscription.

    Returns
    -------
    list of Path
        Per-category run directories.
    """
    available = sorted(get_available_categories())

    # Resolve "all"
    resolved: List[str] = []
    for cat in categories:
        if cat.lower() == "all":
            resolved = available
            break
        resolved.append(cat)
    if not resolved:
        resolved = available

    # Validate
    for cat in resolved:
        if cat not in available:
            raise ValueError(f"Unknown category '{cat}'. Available: {available}")

    seed_schedule = list(seeds) if seeds else [seed]
    if not seed_schedule:
        seed_schedule = [seed]

    if parallel_gpu_ids:
        if model is not None or tokenizer is not None:
            raise ValueError("parallel_gpu_ids cannot be used together with preloaded model/tokenizer handles.")
        effective_workers_per_gpu = workers_per_gpu
        gpu_inventory = None
        if auto_workers_per_gpu:
            effective_workers_per_gpu, gpu_inventory = _cap_workers_per_gpu_by_memory(
                parallel_gpu_ids,
                requested_workers_per_gpu=workers_per_gpu,
            )
            if gpu_inventory is None and workers_per_gpu > 1:
                # Conservative fallback when telemetry is unavailable.
                # Prevents repeated OOM cascades from oversubscribing large AWQ models.
                effective_workers_per_gpu = 1
                print(
                    "[INFO] Unable to query nvidia-smi memory telemetry; "
                    "falling back to 1 worker/GPU for OOM safety."
                )
            if effective_workers_per_gpu < workers_per_gpu:
                print(
                    "[INFO] Capping workers per GPU from "
                    f"{workers_per_gpu} to {effective_workers_per_gpu} based on live GPU memory."
                )
                if gpu_inventory:
                    for gid in parallel_gpu_ids:
                        info = gpu_inventory.get(gid)
                        if not info:
                            continue
                        print(
                            f"       gpu{gid}: total={info['total']}MiB used={info['used']}MiB free={info['free']}MiB"
                        )
        return _run_pipeline_debug_multi_parallel(
            resolved_categories=resolved,
            stop_after_stage=stop_after_stage,
            seed_schedule=seed_schedule,
            debug_root=Path(debug_root),
            town=town,
            model_id=model_id,
            schema_dashboard=schema_dashboard,
            schema_dashboard_output=schema_dashboard_output,
            schema_dashboard_json=schema_dashboard_json,
            schema_dashboard_title=schema_dashboard_title,
            schema_dashboard_globs=schema_dashboard_globs,
            parallel_gpu_ids=parallel_gpu_ids,
            workers_per_gpu=effective_workers_per_gpu,
            max_parallel_workers=max_parallel_workers,
            max_oom_retries=max_oom_retries,
            carla_validate=carla_validate,
            carla_host=carla_host,
            carla_port=carla_port,
            carla_repair_max_attempts=carla_repair_max_attempts,
            carla_repair_xy_offsets=carla_repair_xy_offsets,
            carla_repair_z_offsets=carla_repair_z_offsets,
            carla_align_before_validate=carla_align_before_validate,
            carla_require_risk=carla_require_risk,
            carla_validation_timeout=carla_validation_timeout,
            start_carla=start_carla,
            carla_root=carla_root,
            carla_args=carla_args,
            carla_startup_timeout=carla_startup_timeout,
            carla_shutdown_timeout=carla_shutdown_timeout,
            carla_port_tries=carla_port_tries,
            carla_port_step=carla_port_step,
        )

    defer_carla_validation = _should_defer_carla_validation(stop_after_stage, carla_validate)
    phase1_stop_after_stage = Stage.ROUTES.value if defer_carla_validation else stop_after_stage
    phase1_carla_validate = bool(carla_validate and not defer_carla_validation)
    phase1_start_carla = bool(start_carla and not defer_carla_validation)
    if defer_carla_validation:
        print(
            "[INFO] Deferred CARLA mode enabled: generating through routes first, "
            "then validating sequentially with a single CARLA instance."
        )

    # Load model once
    loaded_local_model = False
    if model is None or tokenizer is None:
        print(f"[INIT] Loading model once for {len(resolved)} categories: {model_id}")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model.eval()
        loaded_local_model = True
        print("[INIT] Model loaded.")

    run_dirs: List[Path] = []
    results_summary: List[Dict[str, Any]] = []
    released_model_before_carla = False

    print("\n" + "#" * 70)
    print(f"  MULTI-CATEGORY DEBUG RUN — {len(resolved)} categories × {len(seed_schedule)} seed(s)")
    print("#" * 70)
    total_runs = len(resolved) * len(seed_schedule)
    run_idx = 0
    for seed_idx, seed_value in enumerate(seed_schedule, 1):
        if len(seed_schedule) > 1:
            print(f"\n{'=' * 70}")
            print(f"  SEED BATCH [{seed_idx}/{len(seed_schedule)}] — seed={seed_value}")
            print(f"{'=' * 70}")
        for cat in resolved:
            run_idx += 1
            print(f"\n{'━' * 70}")
            print(f"  [{run_idx}/{total_runs}]  {cat}  (seed={seed_value})")
            print(f"{'━' * 70}")
            try:
                rd = run_pipeline_debug(
                    category=cat,
                    stop_after_stage=phase1_stop_after_stage,
                    seed=seed_value,
                    debug_root=debug_root,
                    model=model,
                    tokenizer=tokenizer,
                    town=town,
                    model_id=model_id,
                    carla_validate=phase1_carla_validate,
                    carla_host=carla_host,
                    carla_port=carla_port,
                    carla_repair_max_attempts=carla_repair_max_attempts,
                    carla_repair_xy_offsets=carla_repair_xy_offsets,
                    carla_repair_z_offsets=carla_repair_z_offsets,
                    carla_align_before_validate=carla_align_before_validate,
                    carla_require_risk=carla_require_risk,
                    carla_validation_timeout=carla_validation_timeout,
                    start_carla=phase1_start_carla,
                    carla_root=carla_root,
                    carla_args=carla_args,
                    carla_startup_timeout=carla_startup_timeout,
                    carla_shutdown_timeout=carla_shutdown_timeout,
                    carla_port_tries=carla_port_tries,
                    carla_port_step=carla_port_step,
                )
                run_dirs.append(rd)
                rec = {
                    "seq": run_idx - 1,
                    "attempt": 0,
                    "category": cat,
                    "seed": seed_value,
                    "run_dir": str(rd),
                    "all_passed": False,
                    "failed_stage": None,
                    "score": None,
                    "error": None,
                    "is_cuda_oom": False,
                }
                results_summary.append(_refresh_result_record_from_run(rec))
            except Exception as exc:
                print(f"  ✗ CATEGORY FAILED: {cat} (seed={seed_value}): {exc}")
                results_summary.append(
                    {
                        "seq": run_idx - 1,
                        "attempt": 0,
                        "category": cat,
                        "seed": seed_value,
                        "error": str(exc),
                        "all_passed": False,
                    }
                )

    results_summary.sort(key=lambda x: int(x.get("seq", 10**9)))

    if defer_carla_validation:
        if loaded_local_model:
            _release_worker_gpu_resources(model, tokenizer)
            released_model_before_carla = True
            model = None
            tokenizer = None
        inferred_gpu_id = _infer_primary_visible_gpu_id()
        deferred_gpu_candidates = [int(inferred_gpu_id)] if inferred_gpu_id is not None else None
        _run_deferred_carla_validation_batch(
            results_summary=results_summary,
            carla_host=carla_host,
            carla_port=carla_port,
            carla_repair_max_attempts=carla_repair_max_attempts,
            carla_repair_xy_offsets=carla_repair_xy_offsets,
            carla_repair_z_offsets=carla_repair_z_offsets,
            carla_align_before_validate=carla_align_before_validate,
            carla_require_risk=carla_require_risk,
            carla_validation_timeout=carla_validation_timeout,
            start_carla=start_carla,
            carla_root=carla_root,
            carla_args=carla_args,
            carla_startup_timeout=carla_startup_timeout,
            carla_shutdown_timeout=carla_shutdown_timeout,
            carla_port_tries=carla_port_tries,
            carla_port_step=carla_port_step,
            preferred_gpu_ids=deferred_gpu_candidates,
        )
        for rec in results_summary:
            _refresh_result_record_from_run(rec)

    # Print summary table
    print("\n" + "#" * 70)
    print("  MULTI-CATEGORY RESULTS")
    print("#" * 70)
    for r in results_summary:
        status = "✓" if r.get("all_passed") else "✗"
        seed_str = f"  seed={r.get('seed')}" if "seed" in r else ""
        score = f"  score={r['score']:.2f}" if r.get("score") is not None else ""
        failed = f"  failed={r['failed_stage']}" if r.get("failed_stage") else ""
        err = f"  error={r.get('error', '')}" if r.get("error") else ""
        warn = "  warning=carla_connect_failed_non_blocking" if r.get("carla_infra_warning") else ""
        print(f"  {status} {r['category']}{seed_str}{score}{failed}{err}{warn}")
    print()

    dashboard_info = None
    if schema_dashboard:
        dashboard_info = _build_integrated_schema_dashboard(
            run_dirs=run_dirs,
            debug_root=Path(debug_root),
            output_path=schema_dashboard_output,
            json_path=schema_dashboard_json,
            title=schema_dashboard_title,
            extra_globs=schema_dashboard_globs,
        )

    # Write multi-run summary
    multi_summary_path = Path(debug_root) / f"multi_run_{_now_tag()}.json"
    _write_json(multi_summary_path, {
        "categories": resolved,
        "seeds": seed_schedule,
        "results": [{k: v for k, v in r.items() if k != "seq"} for r in results_summary],
        "seed": seed,
        "stop_after_stage": stop_after_stage,
        "schema_dashboard": dashboard_info,
        "parallel": {"enabled": False},
        "deferred_carla_validation": {
            "enabled": bool(defer_carla_validation),
            "single_instance": bool(defer_carla_validation),
            "start_carla": bool(start_carla),
        },
    })
    print(f"  Multi-run summary: {multi_summary_path}")

    if loaded_local_model and not released_model_before_carla:
        _release_worker_gpu_resources(model, tokenizer)

    return run_dirs


# ###########################################################################
#  CLI
# ###########################################################################

def _parse_categories(raw: str) -> List[str]:
    """Parse comma-separated category string. 'all' is a special keyword."""
    if raw.strip().lower() == "all":
        return ["all"]
    # Split on comma, strip whitespace
    return [c.strip() for c in raw.split(",") if c.strip()]


def _resolve_seed_schedule(
    seed: Optional[int],
    seeds: Optional[List[int]],
    seed_count: int,
) -> List[Optional[int]]:
    if seeds:
        # Preserve caller order and duplicates intentionally:
        # duplicated seed entries can be used as explicit redoes.
        return [int(s) for s in seeds]

    if seed_count > 1:
        if seed is None:
            raise ValueError("--seed is required when --seed-count > 1")
        return [seed + i for i in range(seed_count)]

    return [seed]


def _parse_parallel_gpu_ids(raw_values: Optional[List[str]]) -> Optional[List[int]]:
    """
    Parse GPU IDs from CLI values.
    Supports both space-separated and comma-separated forms:
      --parallel-gpus 0 1
      --parallel-gpus 0,1
    """
    if not raw_values:
        return None

    parsed: List[int] = []
    seen = set()
    for raw in raw_values:
        if raw is None:
            continue
        for token in str(raw).split(","):
            token = token.strip()
            if not token:
                continue
            if not token.lstrip("-").isdigit():
                raise ValueError(f"Invalid GPU id '{token}' in --parallel-gpus")
            gpu_id = int(token)
            if gpu_id < 0:
                raise ValueError(f"GPU id must be >= 0, got {gpu_id}")
            if gpu_id in seen:
                continue
            seen.add(gpu_id)
            parsed.append(gpu_id)

    return parsed or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug pipeline wrapper with per-stage visualizations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(f"""\
            Valid stage names for --stop-after:
              {', '.join(s.value for s in Stage)}

            Examples:
              # Run ONE category, stop at schema:
              python scenario_generator/start_pipeline.py --category "Construction Zone" --stop-after schema

              # Run ONE category through path picking:
              python scenario_generator/start_pipeline.py --category "Highway On-Ramp Merge" --stop-after pick_paths --seed 42

              # Run MULTIPLE categories (comma-separated):
              python scenario_generator/start_pipeline.py --category "Highway On-Ramp Merge, Construction Zone"

              # Run ALL categories:
              python scenario_generator/start_pipeline.py --category all

              # Run all stages for a single category:
              python scenario_generator/start_pipeline.py --category "Interactive Lane Change"
        """),
    )
    parser.add_argument(
        "--category", default=None,
        help='Category name(s). Comma-separated for multiple, or "all" for every category.',
    )
    parser.add_argument(
        "--stop-after",
        dest="stop_after",
        default=None,
        choices=[s.value for s in Stage],
        help="Stage name to stop after (applies to every category)",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Explicit list of seeds (overrides --seed and --seed-count). Example: --seeds 41 42 43. "
             "Duplicate entries are allowed and treated as explicit redoes.",
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=1,
        help="If >1, run a contiguous seed block starting from --seed (e.g., --seed 42 --seed-count 3 => 42,43,44).",
    )
    parser.add_argument(
        "--target-scenarios",
        type=int,
        default=0,
        help=(
            "Target mode: keep generating/validating until this many non-duplicate scenarios are accepted "
            "(high=CARLA pass, medium=CARLA failed but reviewable)."
        ),
    )
    parser.add_argument(
        "--target-seed-round-batch",
        type=int,
        default=4,
        help="Target mode: seed rounds per generation batch (each round runs all selected categories once).",
    )
    parser.add_argument(
        "--target-max-generated",
        type=int,
        default=0,
        help="Target mode safety cap on total generated scenarios (0 = unlimited).",
    )
    parser.add_argument(
        "--parallel-gpus",
        nargs="+",
        default=None,
        help="Enable GPU-parallel execution on these GPU IDs. Supports space or comma syntax: "
             "--parallel-gpus 0 1  OR  --parallel-gpus 0,1",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=2,
        help="Requested max worker processes per GPU in parallel mode (subject to auto-cap unless disabled).",
    )
    parser.add_argument(
        "--max-parallel-workers",
        type=int,
        default=None,
        help="Optional cap on total workers across all GPUs.",
    )
    parser.add_argument(
        "--oom-retries",
        type=int,
        default=1,
        help="Retries per task when a CUDA OOM is detected in parallel mode (default: 1).",
    )
    parser.add_argument(
        "--auto-workers-per-gpu",
        dest="auto_workers_per_gpu",
        action="store_true",
        default=True,
        help="Auto-cap workers per GPU using live memory (default: enabled).",
    )
    parser.add_argument(
        "--no-auto-workers-per-gpu",
        dest="auto_workers_per_gpu",
        action="store_false",
        help="Disable memory-based worker capping.",
    )
    parser.add_argument("--debug-root", default="debug_runs", help="Root directory for debug output")
    parser.add_argument("--town", default=None, help="Override CARLA town name")
    parser.add_argument(
        "--carla-validate",
        dest="carla_validate",
        action="store_true",
        default=True,
        help="Run final CARLA validation stage (default: enabled).",
    )
    parser.add_argument(
        "--no-carla-validate",
        dest="carla_validate",
        action="store_false",
        help="Disable final CARLA validation stage.",
    )
    parser.add_argument("--carla-host", default="127.0.0.1", help="CARLA host for final validation stage.")
    parser.add_argument("--carla-port", type=int, default=3000, help="CARLA port for final validation stage.")
    parser.add_argument(
        "--start-carla",
        action="store_true",
        help="Launch a local CARLA process for this debug run and stop it at the end.",
    )
    parser.add_argument(
        "--carla-kill-user-processes",
        action="store_true",
        help=(
            "Before any auto-started CARLA launch, terminate existing CARLA processes owned by the current user "
            "(shared-server safe: does not touch other users)."
        ),
    )
    parser.add_argument(
        "--carla-root",
        default=None,
        help="CARLA root dir containing CarlaUE4.sh (default: $CARLA_ROOT or <repo>/carla912).",
    )
    parser.add_argument(
        "--carla-arg",
        action="append",
        default=[],
        help="Extra args passed to CarlaUE4.sh (repeatable).",
    )
    parser.add_argument(
        "--carla-startup-timeout",
        type=float,
        default=CARLA_STARTUP_TIMEOUT_S,
        help="Seconds to wait for auto-started CARLA port to open.",
    )
    parser.add_argument(
        "--carla-shutdown-timeout",
        type=float,
        default=CARLA_SHUTDOWN_TIMEOUT_S,
        help="Seconds to wait before force-killing CARLA on shutdown.",
    )
    parser.add_argument(
        "--carla-port-tries",
        type=int,
        default=CARLA_PORT_TRIES,
        help="When --start-carla and port is busy, number of candidate ports to try.",
    )
    parser.add_argument(
        "--carla-port-step",
        type=int,
        default=CARLA_PORT_STEP,
        help="Step size between candidate ports when probing for a free CARLA port.",
    )
    parser.add_argument(
        "--carla-repair-max-attempts",
        type=int,
        default=2,
        help="Maximum persistent CARLA repair attempts after validation failures.",
    )
    parser.add_argument(
        "--carla-repair-xy-offsets",
        default="0.0,0.25,-0.25,0.5,-0.5,1.0,-1.0",
        help="Comma-separated XY offset candidates for CARLA spawn repair.",
    )
    parser.add_argument(
        "--carla-repair-z-offsets",
        default="0.0,0.2,-0.2,0.5,-0.5,1.0",
        help="Comma-separated Z offset candidates for CARLA spawn repair.",
    )
    parser.add_argument(
        "--carla-align-before-validate",
        action="store_true",
        help="Run CARLA GRP route alignment before final CARLA validation.",
    )
    parser.add_argument(
        "--carla-require-risk",
        dest="carla_require_risk",
        action="store_true",
        default=True,
        help="Require constant-trajectory near-miss/collision-risk signal to pass CARLA validation.",
    )
    parser.add_argument(
        "--no-carla-require-risk",
        dest="carla_require_risk",
        action="store_false",
        help="Disable risk-signal requirement in CARLA validation.",
    )
    parser.add_argument(
        "--carla-validation-timeout",
        type=float,
        default=180.0,
        help="Timeout budget in seconds for final CARLA validation simulation.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-32B-Instruct-AWQ",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="List valid stage names and exit",
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List available categories and exit",
    )
    parser.add_argument(
        "--schema-dashboard",
        dest="schema_dashboard",
        action="store_true",
        default=None,
        help="Generate integrated pipeline dashboard (schema + downstream stages) after run(s). "
             "Default: enabled for all runs.",
    )
    parser.add_argument(
        "--no-schema-dashboard",
        dest="schema_dashboard",
        action="store_false",
        help="Disable integrated pipeline dashboard generation.",
    )
    parser.add_argument(
        "--schema-dashboard-output",
        default=None,
        help="Output HTML path for integrated dashboard.",
    )
    parser.add_argument(
        "--schema-dashboard-json",
        default=None,
        help="Output JSON path for integrated dashboard payload.",
    )
    parser.add_argument(
        "--schema-dashboard-title",
        default=None,
        help="Dashboard title override.",
    )
    parser.add_argument(
        "--schema-dashboard-glob",
        nargs="+",
        default=None,
        help="Optional EXTRA run globs to include in integrated dashboard (for history/redo comparisons). "
             "Not required: current command's generated runs are always included automatically.",
    )

    args = parser.parse_args()

    if args.list_stages:
        print("Valid stages:")
        for s in Stage:
            print(f"  {s.value:15s}  ({STAGE_DIR_PREFIX[s]})")
        return

    if args.list_categories:
        print("Available categories:")
        for c in sorted(get_available_categories()):
            print(f"  {c}")
        return

    if not args.category:
        parser.error(
            '--category is required (unless using --list-stages or --list-categories).\n'
            'Use --category "all" for every category, or comma-separate multiple.\n'
            'Run --list-categories to see available options.'
        )

    categories = _parse_categories(args.category)
    try:
        seed_schedule = _resolve_seed_schedule(args.seed, args.seeds, args.seed_count)
    except ValueError as exc:
        parser.error(str(exc))
        return
    try:
        parallel_gpu_ids = _parse_parallel_gpu_ids(args.parallel_gpus)
    except ValueError as exc:
        parser.error(str(exc))
        return
    if args.workers_per_gpu < 1:
        parser.error("--workers-per-gpu must be >= 1")
        return
    if args.max_parallel_workers is not None and args.max_parallel_workers < 1:
        parser.error("--max-parallel-workers must be >= 1")
        return
    if args.oom_retries < 0:
        parser.error("--oom-retries must be >= 0")
        return
    if args.target_scenarios < 0:
        parser.error("--target-scenarios must be >= 0")
        return
    if args.target_seed_round_batch < 1:
        parser.error("--target-seed-round-batch must be >= 1")
        return
    if args.target_max_generated < 0:
        parser.error("--target-max-generated must be >= 0")
        return
    if args.carla_port <= 0:
        parser.error("--carla-port must be > 0")
        return
    if args.carla_repair_max_attempts < 0:
        parser.error("--carla-repair-max-attempts must be >= 0")
        return
    if args.carla_validation_timeout <= 0:
        parser.error("--carla-validation-timeout must be > 0")
        return
    if args.carla_startup_timeout <= 0:
        parser.error("--carla-startup-timeout must be > 0")
        return
    if args.carla_shutdown_timeout <= 0:
        parser.error("--carla-shutdown-timeout must be > 0")
        return
    if args.carla_port_tries < 1:
        parser.error("--carla-port-tries must be >= 1")
        return
    if args.carla_port_step < 1:
        parser.error("--carla-port-step must be >= 1")
        return

    if args.carla_kill_user_processes and not args.start_carla:
        print("[INFO] --carla-kill-user-processes ignored because --start-carla is not enabled.")
    if args.carla_kill_user_processes and args.start_carla:
        _maybe_cleanup_user_carla_processes(True, reason="cli_prelaunch")

    auto_dashboard = True
    enable_schema_dashboard = args.schema_dashboard if args.schema_dashboard is not None else auto_dashboard

    if args.target_scenarios > 0:
        if not args.carla_validate:
            parser.error("--target-scenarios requires CARLA validation enabled (remove --no-carla-validate).")
            return
        run_pipeline_debug_multi_target(
            categories=categories,
            target_scenarios=args.target_scenarios,
            stop_after_stage=args.stop_after,
            seed=seed_schedule[0] if seed_schedule else args.seed,
            seeds=seed_schedule,
            debug_root=Path(args.debug_root),
            town=args.town,
            model_id=args.model,
            schema_dashboard=enable_schema_dashboard,
            schema_dashboard_output=args.schema_dashboard_output,
            schema_dashboard_json=args.schema_dashboard_json,
            schema_dashboard_title=args.schema_dashboard_title,
            schema_dashboard_globs=args.schema_dashboard_glob,
            parallel_gpu_ids=parallel_gpu_ids,
            workers_per_gpu=args.workers_per_gpu,
            max_parallel_workers=args.max_parallel_workers,
            max_oom_retries=args.oom_retries,
            auto_workers_per_gpu=args.auto_workers_per_gpu,
            carla_host=args.carla_host,
            carla_port=args.carla_port,
            carla_repair_max_attempts=args.carla_repair_max_attempts,
            carla_repair_xy_offsets=args.carla_repair_xy_offsets,
            carla_repair_z_offsets=args.carla_repair_z_offsets,
            carla_align_before_validate=args.carla_align_before_validate,
            carla_require_risk=args.carla_require_risk,
            carla_validation_timeout=args.carla_validation_timeout,
            start_carla=args.start_carla,
            carla_root=args.carla_root,
            carla_args=args.carla_arg,
            carla_startup_timeout=args.carla_startup_timeout,
            carla_shutdown_timeout=args.carla_shutdown_timeout,
            carla_port_tries=args.carla_port_tries,
            carla_port_step=args.carla_port_step,
            target_seed_round_batch=args.target_seed_round_batch,
            target_max_generated=args.target_max_generated if args.target_max_generated > 0 else None,
        )
        return

    if len(categories) == 1 and categories[0].lower() != "all" and len(seed_schedule) == 1 and not parallel_gpu_ids:
        # Single category — use direct function for simpler output
        run_dir = run_pipeline_debug(
            category=categories[0],
            stop_after_stage=args.stop_after,
            seed=seed_schedule[0],
            debug_root=Path(args.debug_root),
            town=args.town,
            model_id=args.model,
            carla_validate=args.carla_validate,
            carla_host=args.carla_host,
            carla_port=args.carla_port,
            carla_repair_max_attempts=args.carla_repair_max_attempts,
            carla_repair_xy_offsets=args.carla_repair_xy_offsets,
            carla_repair_z_offsets=args.carla_repair_z_offsets,
            carla_align_before_validate=args.carla_align_before_validate,
            carla_require_risk=args.carla_require_risk,
            carla_validation_timeout=args.carla_validation_timeout,
            start_carla=args.start_carla,
            carla_root=args.carla_root,
            carla_args=args.carla_arg,
            carla_startup_timeout=args.carla_startup_timeout,
            carla_shutdown_timeout=args.carla_shutdown_timeout,
            carla_port_tries=args.carla_port_tries,
            carla_port_step=args.carla_port_step,
        )
        if enable_schema_dashboard:
            _build_integrated_schema_dashboard(
                run_dirs=[run_dir],
                debug_root=Path(args.debug_root),
                output_path=args.schema_dashboard_output,
                json_path=args.schema_dashboard_json,
                title=args.schema_dashboard_title,
                extra_globs=args.schema_dashboard_glob,
            )
    else:
        # Multiple categories and/or multiple seeds (or explicit parallel mode).
        run_pipeline_debug_multi(
            categories=categories,
            stop_after_stage=args.stop_after,
            seed=seed_schedule[0] if seed_schedule else args.seed,
            seeds=seed_schedule,
            debug_root=Path(args.debug_root),
            town=args.town,
            model_id=args.model,
            schema_dashboard=enable_schema_dashboard,
            schema_dashboard_output=args.schema_dashboard_output,
            schema_dashboard_json=args.schema_dashboard_json,
            schema_dashboard_title=args.schema_dashboard_title,
            schema_dashboard_globs=args.schema_dashboard_glob,
            parallel_gpu_ids=parallel_gpu_ids,
            workers_per_gpu=args.workers_per_gpu,
            max_parallel_workers=args.max_parallel_workers,
            max_oom_retries=args.oom_retries,
            auto_workers_per_gpu=args.auto_workers_per_gpu,
            carla_validate=args.carla_validate,
            carla_host=args.carla_host,
            carla_port=args.carla_port,
            carla_repair_max_attempts=args.carla_repair_max_attempts,
            carla_repair_xy_offsets=args.carla_repair_xy_offsets,
            carla_repair_z_offsets=args.carla_repair_z_offsets,
            carla_align_before_validate=args.carla_align_before_validate,
            carla_require_risk=args.carla_require_risk,
            carla_validation_timeout=args.carla_validation_timeout,
            start_carla=args.start_carla,
            carla_root=args.carla_root,
            carla_args=args.carla_arg,
            carla_startup_timeout=args.carla_startup_timeout,
            carla_shutdown_timeout=args.carla_shutdown_timeout,
            carla_port_tries=args.carla_port_tries,
            carla_port_step=args.carla_port_step,
        )


if __name__ == "__main__":
    main()
