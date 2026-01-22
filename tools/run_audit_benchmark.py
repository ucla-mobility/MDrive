#!/usr/bin/env python3
"""
Audit runner for schema-driven scenario generation + CARLA evaluation.

Runs each category, samples best-of candidates, ranks the top K, records each
run in a single CSV, and is restartable without duplicating completed work.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scenario_generator"))

from scenario_generator.schema_generator import (  # noqa: E402
    SchemaScenarioGenerator,
    SchemaGenerationConfig,
)
from scenario_generator.schema_utils import (  # noqa: E402
    geometry_spec_from_scenario_spec,
)
from scenario_generator.constraints import spec_to_dict  # noqa: E402
from scenario_generator.scene_validator import SceneValidator  # noqa: E402
from scenario_generator.pipeline_runner import PipelineRunner  # noqa: E402
from scenario_generator.capabilities import CATEGORY_DEFINITIONS, TopologyType, get_available_categories  # noqa: E402
from convert_scene_to_routes import convert_scene_to_routes  # noqa: E402


STATES = [
    "scenario_generation",
    "repair_loops",
    "route_generation",
    "carla_simulation",
    "video_generation",
    "csv_commit",
]

CSV_FIELDS = [
    "row_type",
    "run_id",
    "run_key",
    "category",
    "category_description",
    "run_tag",
    "variant_index",
    "scenario_id",
    "scenario_description",
    "scenario_spec_path",
    "scene_objects_path",
    "scenario_objects_json",
    "scenario_generation_success",
    "validation_score",
    "failure_state",
    "failure_reason",
    "pipeline_attempts",
    "pipeline_attempts_used",
    "repair_attempts_total",
    "repair_outer_attempts",
    "repair_schema_attempts",
    "schema_template_fallback",
    "repair_object_stage1_json",
    "repair_object_stage1_evidence",
    "repair_object_stage2_json",
    "repair_object_stage2_validation",
    "routes_dir",
    "route_files",
    "route_generation_success",
    "scene_png_path",
    "carla_results_tag",
    "carla_results_dir",
    "carla_image_dir",
    "carla_simulation_success",
    "video_path",
    "video_generation_success",
    "state_history",
    "best_rank",
    "kept_best",
    "start_time",
    "end_time",
    "elapsed_s",
    "csv_committed_at",
]


# =============================================================================
# Cross-Difficulty Output Hash Tracking
# =============================================================================

def _compute_scene_output_hash(scene_path: str) -> Optional[str]:
    """
    Compute a hash of vehicle spawn positions from scene_objects.json.
    This is used to detect when different specs produce identical outputs.
    
    Returns None if file cannot be read or has no vehicles.
    """
    import hashlib
    try:
        with open(scene_path, 'r') as f:
            scene_data = json.load(f)
        
        # Extract vehicle spawn positions and sort for deterministic hash
        ego_picked = scene_data.get('ego_picked', [])
        if not ego_picked:
            return None
        
        spawn_positions = []
        for ego in ego_picked:
            # Get spawn position from signature.entry.point
            sig = ego.get('signature', {})
            entry = sig.get('entry', {})
            point = entry.get('point', {})
            
            if point:
                spawn_positions.append((
                    round(point.get('x', 0), 1),
                    round(point.get('y', 0), 1),
                ))
            else:
                # Fallback to first point of first segment
                segments = sig.get('segments_detailed', [])
                if segments:
                    start = segments[0].get('start', {}).get('point', {})
                    if start:
                        spawn_positions.append((
                            round(start.get('x', 0), 1),
                            round(start.get('y', 0), 1),
                        ))
        
        if not spawn_positions:
            return None
        
        # Sort and hash
        spawn_positions.sort()
        hash_input = json.dumps(spawn_positions, sort_keys=True)
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    except Exception as e:
        print(f"[HASH] Error computing hash: {e}")
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(text: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return name or "run"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(path)


def _atomic_write_json(path: Path, obj: Any) -> None:
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True)
    _atomic_write_text(path, text)


def _write_schema_validation_report(
    report_path: Path,
    category: str,
    attempt: int,
    errors: List[str],
    warnings: Optional[List[str]] = None,
) -> None:
    issues = []
    for msg in errors:
        issues.append({"severity": "error", "message": msg})
    for msg in warnings or []:
        issues.append({"severity": "warning", "message": msg})

    report = {
        "timestamp": _now_iso(),
        "category": category,
        "attempt": attempt,
        "is_valid": False,
        "summary": {
            "error_count": len(errors),
            "warning_count": len(warnings or []),
            "failure_reason": errors[0] if errors else "spec_generation_failed",
        },
        "issues": issues,
    }
    _atomic_write_json(report_path, report)
    print(f"[Schema Validation Report] Saved to {report_path}")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _serialize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    return str(value)


def _normalize_row(row: Dict[str, Any]) -> Dict[str, str]:
    out = {field: "" for field in CSV_FIELDS}
    for key, value in row.items():
        if key in out:
            out[key] = _serialize_value(value)
    return out


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _write_csv_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(_normalize_row(row))
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(path)


def _ensure_category_row(
    csv_path: Path,
    run_id: str,
    category: str,
    category_description: str,
) -> None:
    rows = _read_csv_rows(csv_path)
    for row in rows:
        if row.get("row_type") == "category_info" and row.get("category") == category:
            return

    category_row = {
        "row_type": "category_info",
        "run_id": run_id,
        "category": category,
        "category_description": category_description,
        "start_time": _now_iso(),
    }

    insert_at = None
    for idx, row in enumerate(rows):
        if row.get("row_type") == "scenario_run" and row.get("category") == category:
            insert_at = idx
            break
    if insert_at is None:
        rows.append(category_row)
    else:
        rows.insert(insert_at, category_row)
    _write_csv_rows(csv_path, rows)


def _append_scenario_row(csv_path: Path, scenario_row: Dict[str, Any]) -> None:
    rows = _read_csv_rows(csv_path)
    for row in rows:
        if row.get("row_type") == "scenario_run" and row.get("run_key") == scenario_row.get("run_key"):
            return
    rows.append(scenario_row)
    _write_csv_rows(csv_path, rows)


def _set_state(status: Dict[str, Any], state: str) -> None:
    status["current_state"] = state
    status.setdefault("state_history", []).append({"state": state, "timestamp": _now_iso()})
    status.setdefault("timestamps", {})[f"{state}_start"] = _now_iso()


def _complete_state(status: Dict[str, Any], state: str) -> None:
    completed = status.setdefault("completed_states", [])
    if state not in completed:
        completed.append(state)
    status.setdefault("timestamps", {})[f"{state}_end"] = _now_iso()


def _fail_state(status: Dict[str, Any], state: str, reason: str) -> None:
    status["failure_state"] = state
    status["failure_reason"] = reason
    status["success"] = False


def _build_category_description(category: str) -> str:
    info = CATEGORY_DEFINITIONS.get(category)
    if not info:
        return ""
    parts = [info.summary.strip()] if info.summary else []
    if info.intent:
        parts.append(info.intent.strip())
    if info.must_include:
        parts.append("must_include: " + "; ".join(info.must_include))
    if info.avoid:
        parts.append("avoid: " + "; ".join(info.avoid))
    if info.vary:
        axis_text = "; ".join(f"{ax.name}: {', '.join(ax.options)}" for ax in info.vary)
        parts.append("variation_axes: " + axis_text)
    return " | ".join(p for p in parts if p)


def _resolve_town_for_category(
    category: str,
    default_town: str,
    highway_town: Optional[str],
    t_junction_town: Optional[str] = None,
    two_lane_corridor_town: Optional[str] = None,
) -> str:
    info = CATEGORY_DEFINITIONS.get(category)
    if info:
        if highway_town and info.map.topology == TopologyType.HIGHWAY:
            return highway_town
        if t_junction_town and info.map.topology == TopologyType.T_JUNCTION:
            return t_junction_town
        if two_lane_corridor_town and info.map.topology == TopologyType.TWO_LANE_CORRIDOR:
            return two_lane_corridor_town
    return default_town


def _load_scene_objects(scene_path: Optional[str]) -> Dict[str, Any]:
    if not scene_path:
        return {}
    path = Path(scene_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_object_manifest(spec: Dict[str, Any], scene: Dict[str, Any]) -> Dict[str, Any]:
    actors = scene.get("actors") or []
    npc_actors = [a for a in actors if str(a.get("category", "")).lower() in {"vehicle", "walker", "cyclist"}]
    props = [a for a in actors if str(a.get("category", "")).lower() == "static"]
    return {
        "ego_vehicles": spec.get("ego_vehicles", []),
        "vehicle_constraints": spec.get("vehicle_constraints", []),
        "ego_paths": scene.get("ego_picked", []),
        "npc_actors": npc_actors,
        "props": props,
        "all_actors": actors,
        "crop_region": scene.get("crop_region"),
    }


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np  # noqa: F401
    except Exception:
        np = None
    if np is not None:
        np.random.seed(seed)
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _load_shared_model(model_id: str):
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
    return model, tokenizer


def _run_cmd(cmd: Sequence[str], log_path: Path, env: Optional[Dict[str, str]] = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n[CMD] " + " ".join(cmd) + "\n")
        f.flush()
        result = subprocess.run(cmd, stdout=f, stderr=f, env=env)
        return result.returncode


def _find_latest_image_dir(image_root: Path) -> Optional[Path]:
    if not image_root.exists():
        return None
    candidates = [p for p in image_root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _has_route_files(routes_dir: Path) -> bool:
    if not routes_dir.exists():
        return False
    for _ in routes_dir.rglob("*.xml"):
        return True
    return False


def _collect_route_files(routes_dir: Path) -> List[str]:
    if not routes_dir.exists():
        return []
    files = [str(p) for p in routes_dir.rglob("*.xml")]
    files.sort()
    return files


def _init_status(run_id: str, run_key: str, category: str, run_tag: str, variant_index: int) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "run_key": run_key,
        "category": category,
        "run_tag": run_tag,
        "variant_index": variant_index,
        "state_history": [],
        "completed_states": [],
        "repair_counts": {},
        "metrics": {},
        "attempt_history": [],
        "success": False,
        "row_written": False,
        "start_time": _now_iso(),
    }


def _create_success_symlinks(run_root: Path, run_key: str, status: Dict[str, Any]) -> None:
    """
    Create symlinks in a 'successful_runs' directory for runs that completed successfully.
    Structure: successful_runs/<run_key>/routes/ (routes xml files)
                                        /scene_objects.png
                                        /video.mp4
    """
    success_dir = run_root / "successful_runs" / run_key
    success_dir.mkdir(parents=True, exist_ok=True)
    
    # Symlink routes directory
    routes_dir = status.get("routes_dir")
    if routes_dir and Path(routes_dir).exists():
        routes_link = success_dir / "routes"
        if routes_link.exists() or routes_link.is_symlink():
            routes_link.unlink()
        routes_link.symlink_to(Path(routes_dir), target_is_directory=True)
    
    # Symlink scene PNG
    scene_png = status.get("scene_png_path")
    if scene_png and Path(scene_png).exists():
        png_link = success_dir / "scene_objects.png"
        if png_link.exists() or png_link.is_symlink():
            png_link.unlink()
        png_link.symlink_to(Path(scene_png))
    
    # Symlink video
    video_path = status.get("video_path")
    if video_path and Path(video_path).exists():
        video_link = success_dir / "video.mp4"
        if video_link.exists() or video_link.is_symlink():
            video_link.unlink()
        video_link.symlink_to(Path(video_path))


def _rank_and_mark_best(statuses: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """
    Rank a batch of run statuses by validation_score and mark the top_k unique outputs.
    Deduplicates within the batch using the scene output hash.
    """
    candidates: List[Dict[str, Any]] = []
    for status in statuses:
        if not status:
            continue
        scene_path = status.get("scene_objects_path")
        output_hash = _compute_scene_output_hash(scene_path) if scene_path else None
        status["output_hash"] = output_hash
        status["kept_best"] = False
        status["best_rank"] = ""
        candidates.append(status)

    candidates.sort(
        key=lambda s: (
            bool(s.get("scenario_generation_success")),
            float(s.get("validation_score") or 0.0),
        ),
        reverse=True,
    )

    seen_hashes: Set[str] = set()
    kept_count = 0
    for status in candidates:
        if status.get("output_hash") and status["output_hash"] in seen_hashes:
            continue
        if kept_count < top_k and status.get("scenario_generation_success"):
            status["kept_best"] = True
            kept_count += 1
            status["best_rank"] = kept_count
            if status.get("output_hash"):
                seen_hashes.add(status["output_hash"])

    return candidates


def _update_best_marks(csv_path: Path, statuses: List[Dict[str, Any]]) -> None:
    """Update existing CSV rows with best-of annotations."""
    status_by_run_key = {
        s.get("run_key"): s for s in statuses if s and s.get("run_key")
    }
    if not status_by_run_key:
        return

    rows = _read_csv_rows(csv_path)
    for row in rows:
        run_key = row.get("run_key")
        if run_key in status_by_run_key:
            s = status_by_run_key[run_key]
            row["best_rank"] = s.get("best_rank", "")
            row["kept_best"] = s.get("kept_best", False)
    _write_csv_rows(csv_path, rows)


def _log_kept_best(category: str, statuses: List[Dict[str, Any]], top_k: int) -> None:
    """Print a concise summary of which samples were kept as best."""
    kept = [s for s in statuses if s and s.get("kept_best")]
    kept.sort(key=lambda s: (int(s.get("best_rank") or 999), s.get("run_key", "")))
    if not kept:
        print(f"[BEST] Category {category}: no kept samples (top_k={top_k}).")
        return
    print(f"[BEST] Category {category}: kept {len(kept)}/{top_k}")
    for s in kept:
        score = s.get("validation_score")
        score_str = f"{float(score):.3f}" if score is not None else "n/a"
        print(
            f"  #{s.get('best_rank')}: {s.get('run_key')} "
            f"score={score_str} hash={s.get('output_hash', '')}"
        )


def _create_kept_best_symlinks(run_root: Path, statuses: List[Dict[str, Any]]) -> None:
    """
    Create symlinks for kept-best scenarios only.
    Structure: kept_best/<run_key>/{scenario_spec.json, scene_objects.json, scene_objects.png, routes/}
    """
    kept = [s for s in statuses if s and s.get("kept_best")]
    if not kept:
        return

    kept_root = run_root / "kept_best"
    kept_root.mkdir(parents=True, exist_ok=True)

    for s in kept:
        run_key = s.get("run_key")
        if not run_key:
            continue
        dest = kept_root / run_key
        dest.mkdir(parents=True, exist_ok=True)

        def link(src: Optional[str], name: str, is_dir: bool = False) -> None:
            if not src:
                return
            src_path = Path(src)
            if not src_path.exists():
                return
            tgt = dest / name
            if tgt.exists() or tgt.is_symlink():
                tgt.unlink()
            tgt.symlink_to(src_path, target_is_directory=is_dir)

        link(s.get("scenario_spec_path"), "scenario_spec.json")
        link(s.get("scene_objects_path"), "scene_objects.json")
        link(s.get("scene_png_path"), "scene_objects.png")
        link(s.get("routes_dir"), "routes", is_dir=True)


def _run_single(
    args: argparse.Namespace,
    run_id: str,
    run_root: Path,
    csv_path: Path,
    generator: SchemaScenarioGenerator,
    model,
    tokenizer,
    scene_validator: SceneValidator,
    category: str,
    run_tag: str,
    variant_index: int,
) -> Dict[str, Any]:
    safe_category = _safe_name(category)
    safe_tag = _safe_name(run_tag)
    run_key = f"{safe_category}_{safe_tag}"
    if variant_index > 1:
        run_key = f"{run_key}_v{variant_index}"

    run_dir = run_root / "runs" / run_key
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    status = _read_json(status_path)
    if not status:
        status = _init_status(run_id, run_key, category, run_tag, variant_index)

    if status.get("success") and status.get("row_written"):
        return

    log_path = run_dir / "run.log"

    def log(msg: str) -> None:
        timestamp = _now_iso()
        line = f"[{timestamp}] {msg}"
        print(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def save_status() -> None:
        status["updated_at"] = _now_iso()
        _atomic_write_json(status_path, status)

    repair_counts = status.setdefault("repair_counts", {})
    metrics = status.setdefault("metrics", {})
    attempt_history = status.setdefault("attempt_history", [])

    last_error = ""
    attempt_used = status.get("pipeline_attempts_used")
    if attempt_used is None and status.get("scenario_id"):
        m = re.search(r"_attempt(\d+)$", str(status.get("scenario_id")))
        if m:
            attempt_used = int(m.group(1))
            status["pipeline_attempts_used"] = attempt_used

    if "scenario_generation" not in status.get("completed_states", []):
        _set_state(status, "scenario_generation")
        save_status()

        start_attempt = len(attempt_history) + 1
        schema_attempts_total = int(metrics.get("schema_generation_attempts_total", 0))
        schema_repairs_total = int(metrics.get("schema_generation_repair_attempts_total", 0))
        template_fallback_used = int(metrics.get("schema_template_fallback", 0))
        failed_spec_signatures: Set[str] = set()
        last_validation_feedback: Optional[Dict[str, Any]] = None
        max_attempts = args.max_attempts if args.max_attempts and args.max_attempts > 0 else None
        attempt = start_attempt
        while True:
            if attempt > 1:
                _set_state(status, "repair_loops")
                save_status()

            scenario_id = f"{run_key}_attempt{attempt}"
            scenario_id_safe = _safe_name(scenario_id)
            pipeline_root = run_dir / "pipeline"
            scenario_dir = pipeline_root / scenario_id_safe
            scenario_dir.mkdir(parents=True, exist_ok=True)

            attempt_record = {"attempt": attempt, "scenario_id": scenario_id, "started_at": _now_iso()}

            spec_stats: Dict[str, Any] = {}
            spec, errors, warnings = generator.generate_spec(
                category, 
                stats=spec_stats,
                exclude_signatures=failed_spec_signatures if attempt > 1 else None,
                previous_validation_feedback=last_validation_feedback if attempt > 1 else None,
                debug_dir=scenario_dir,
            )
            schema_attempts_total += int(spec_stats.get("schema_generation_attempts", 0))
            schema_repairs_total += int(spec_stats.get("schema_generation_repair_attempts", 0))
            template_fallback_used = max(template_fallback_used, int(spec_stats.get("schema_template_fallback", 0)))
            metrics["schema_generation_attempts_total"] = schema_attempts_total
            metrics["schema_generation_repair_attempts_total"] = schema_repairs_total
            metrics["schema_template_fallback"] = template_fallback_used
            repair_counts["schema_generation_repair_attempts"] = schema_repairs_total

            if spec is None:
                attempt_record["spec_errors"] = errors
                if warnings:
                    attempt_record["spec_warnings"] = warnings
                _write_schema_validation_report(
                    scenario_dir / "schema_validation_report.json",
                    category=category,
                    attempt=attempt,
                    errors=errors,
                    warnings=warnings,
                )
                # Feed schema validation errors into next attempt to avoid repeating the same spec
                issue_entries = []
                for msg in errors:
                    issue_entries.append({
                        "severity": "error",
                        "category": "SCHEMA_VALIDATION",
                        "message": msg,
                        "expected": "",
                        "actual": "",
                        "suggestion": msg,
                    })
                for msg in warnings:
                    issue_entries.append({
                        "severity": "warning",
                        "category": "SCHEMA_VALIDATION",
                        "message": msg,
                        "expected": "",
                        "actual": "",
                        "suggestion": msg,
                    })
                last_validation_feedback = {
                    "score": 0.0,
                    "issues": issue_entries,
                }
                attempt_record["status"] = "spec_failed"
                last_error = "; ".join(errors) if errors else "spec_generation_failed"
                attempt_history.append(attempt_record)
                save_status()
                attempt += 1
                if max_attempts and attempt > max_attempts:
                    break
                continue

            geometry_spec = geometry_spec_from_scenario_spec(spec)
            spec_dict = spec_to_dict(spec)
            schema_text = json.dumps(spec_dict, indent=2, sort_keys=True)
            _atomic_write_text(
                scenario_dir / "scenario_description.txt",
                schema_text,
            )
            _atomic_write_json(scenario_dir / "scenario_spec.json", spec_dict)

            pipeline_runner = PipelineRunner(
                output_base_dir=str(pipeline_root),
                model=model,
                tokenizer=tokenizer,
                viz_objects=True,
                routes_out_dir="",
                routes_ego_num=args.routes_ego_num,
            )

            town = _resolve_town_for_category(
                category,
                args.town,
                args.highway_town,
                args.t_junction_town,
                args.two_lane_corridor_town,
            )
            success, scene_path, error_msg = pipeline_runner.run_full_pipeline(
                scenario_text=schema_text,
                scenario_schema=spec_dict,
                category=category,
                scenario_id=scenario_id_safe,
                town=town,
                mute=not args.show_pipeline,
                model_id=args.model,
                geometry_spec=geometry_spec,
                stats=repair_counts,
            )

            if not success:
                attempt_record["pipeline_error"] = error_msg
                attempt_record["status"] = "pipeline_failed"
                last_error = error_msg or "pipeline_failed"
                attempt_history.append(attempt_record)
                save_status()
                attempt += 1
                if max_attempts and attempt > max_attempts:
                    break
                continue

            validation = scene_validator.validate_scene(
                scene_path,
                schema_text,
                category=category,
                scenario_spec=spec_dict,
            )
            
            # Generate and save validation report
            try:
                validation_report = scene_validator.generate_validation_report(
                    validation=validation,
                    scene_path=scene_path,
                    scenario_text=schema_text,
                    category=category,
                    repair_history=[{
                        "attempt": attempt,
                        "pipeline_success": success,
                        "validation_score": validation.score,
                        "is_valid": validation.is_valid,
                    }],
                )
                report_path = str(Path(scene_path).parent / "validation_report.json")
                scene_validator.save_validation_report(validation_report, report_path)
            except Exception as e:
                print(f"[WARNING] Failed to generate validation report: {e}")
            
            attempt_record["validation_score"] = validation.score

            if validation.score < args.min_score:
                attempt_record["status"] = "validation_failed"
                last_error = f"validation_score={validation.score:.2f}"
                # Track the failed spec signature to avoid regenerating it
                spec_sig = generator._signature(spec)
                failed_spec_signatures.add(spec_sig)
                # Temporarily remove from generated set to allow retry with different spec
                if spec_sig in generator.generated_signatures.get(category, set()):
                    generator.generated_signatures[category].discard(spec_sig)
                # Store validation feedback for next attempt
                issues_list = [
                    {
                        "severity": issue.severity,
                        "category": issue.issue_type.name if hasattr(issue.issue_type, 'name') else str(issue.issue_type),
                        "message": issue.message,
                        "expected": issue.expected if issue.expected else 'N/A',
                        "actual": issue.actual if issue.actual else 'N/A',
                        "suggestion": issue.suggestion if issue.suggestion else 'N/A',
                    }
                    for issue in validation.issues
                ]
                
                last_validation_feedback = {
                    "score": validation.score,
                    "issues": issues_list,
                }
                attempt_history.append(attempt_record)
                save_status()
                attempt += 1
                if max_attempts and attempt > max_attempts:
                    break
                continue

            attempt_record["status"] = "success"
            attempt_record["scene_path"] = scene_path
            attempt_history.append(attempt_record)
            attempt_used = attempt

            status["scenario_id"] = scenario_id_safe
            status["scenario_description"] = schema_text
            status["scenario_spec_path"] = str(scenario_dir / "scenario_spec.json")
            status["scene_objects_path"] = scene_path
            status["scene_png_path"] = str(Path(scene_path).parent / "scene_objects.png")
            status["validation_score"] = validation.score
            status["scenario_generation_success"] = True
            status["pipeline_attempts_used"] = attempt_used
            break

        if attempt_used is None:
            failure_state = "repair_loops" if (max_attempts and max_attempts > 1) or start_attempt > 1 else "scenario_generation"
            _fail_state(status, failure_state, last_error or "scenario_generation_failed")
            status["scenario_generation_success"] = False
            status["pipeline_attempts_used"] = len(attempt_history)
            save_status()
        else:
            _complete_state(status, "scenario_generation")
            if attempt_used > 1:
                _complete_state(status, "repair_loops")
            else:
                _set_state(status, "repair_loops")
                _complete_state(status, "repair_loops")
            save_status()

    if status.get("scenario_generation_success") and "route_generation" not in status.get("completed_states", []):
        _set_state(status, "route_generation")
        save_status()

        routes_dir = run_dir / "routes"
        routes_dir.mkdir(parents=True, exist_ok=True)
        if not _has_route_files(routes_dir):
            scene_path = status.get("scene_objects_path")
            if not scene_path or not Path(scene_path).exists():
                _fail_state(status, "route_generation", "scene_objects.json missing for route generation")
                save_status()
            else:
                try:
                    convert_scene_to_routes(
                        scene_path,
                        str(routes_dir),
                        ego_num=args.routes_ego_num,
                        align_routes=not args.no_align_routes,
                        carla_host=args.carla_host,
                        carla_port=args.carla_port,
                    )
                    status["routes_dir"] = str(routes_dir)
                    status["route_files"] = _collect_route_files(routes_dir)
                    status["route_generation_success"] = True
                    _complete_state(status, "route_generation")
                    save_status()
                except Exception as exc:
                    _fail_state(status, "route_generation", f"{type(exc).__name__}: {exc}")
                    save_status()
        else:
            status["routes_dir"] = str(routes_dir)
            status["route_files"] = _collect_route_files(routes_dir)
            status["route_generation_success"] = True
            _complete_state(status, "route_generation")
            save_status()

    if (
        status.get("route_generation_success")
        and not args.skip_carla
        and "carla_simulation" not in status.get("completed_states", [])
    ):
        _set_state(status, "carla_simulation")
        save_status()

        routes_dir = Path(status.get("routes_dir", ""))
        if not routes_dir.exists():
            _fail_state(status, "carla_simulation", "routes directory missing for CARLA")
            save_status()
        else:
            tag_parts = []
            if args.results_tag_prefix:
                tag_parts.append(args.results_tag_prefix)
            tag_parts.append(run_id)
            tag_parts.append(run_key)
            results_tag = _safe_name("__".join(tag_parts))
            results_root = REPO_ROOT / "results" / "results_driving_custom" / results_tag
            status["carla_results_tag"] = results_tag
            status["carla_results_dir"] = str(results_root)

            image_root = results_root / "image"
            image_dir = _find_latest_image_dir(image_root)
            if image_dir and image_dir.exists() and not args.force_carla:
                status["carla_image_dir"] = str(image_dir)
                status["carla_simulation_success"] = True
                _complete_state(status, "carla_simulation")
                save_status()
            else:
                cmd = [
                    args.carla_python or sys.executable,
                    str(REPO_ROOT / "tools" / "run_custom_eval.py"),
                    "--routes-dir",
                    str(routes_dir),
                    "--port",
                    str(args.carla_port),
                    "--planner",
                    args.planner,
                    "--results-tag",
                    results_tag,
                    "--repetitions",
                    str(args.repetitions),
                    "--track",
                    args.track,
                    "--timeout",
                    str(args.timeout),
                    "--carla-seed",
                    str(args.carla_seed),
                    "--traffic-seed",
                    str(args.traffic_seed),
                    "--scenario-parameter",
                    args.scenario_parameter,
                    "--scenarios",
                    args.scenarios,
                ]
                if args.agent:
                    cmd.extend(["--agent", args.agent])
                if args.agent_config:
                    cmd.extend(["--agent-config", args.agent_config])
                if args.traffic_manager_port is not None:
                    cmd.extend(["--traffic-manager-port", str(args.traffic_manager_port)])
                if args.resume_carla:
                    cmd.append("--resume")
                if args.no_skip_existed:
                    cmd.append("--no-skip-existed")
                if args.dry_run_carla:
                    cmd.append("--dry-run")

                env = os.environ.copy()
                if args.seed is not None:
                    env["PYTHONHASHSEED"] = str(args.seed)

                rc = _run_cmd(cmd, log_path, env=env)
                if rc != 0:
                    _fail_state(status, "carla_simulation", f"run_custom_eval exit code {rc}")
                    save_status()
                else:
                    image_dir = _find_latest_image_dir(image_root)
                    if image_dir is None:
                        _fail_state(status, "carla_simulation", "No image directory created by CARLA")
                        save_status()
                    else:
                        status["carla_image_dir"] = str(image_dir)
                        status["carla_simulation_success"] = True
                        _complete_state(status, "carla_simulation")
                        save_status()

    if (
        status.get("carla_simulation_success")
        and not args.skip_video
        and "video_generation" not in status.get("completed_states", [])
    ):
        _set_state(status, "video_generation")
        save_status()

        image_dir = status.get("carla_image_dir")
        if not image_dir or not Path(image_dir).exists():
            _fail_state(status, "video_generation", "CARLA image directory missing for video generation")
            save_status()
        else:
            video_path = run_dir / "video.mp4"
            if video_path.exists() and not args.force_video:
                status["video_path"] = str(video_path)
                status["video_generation_success"] = True
                _complete_state(status, "video_generation")
                save_status()
            else:
                cmd = [
                    args.video_python or sys.executable,
                    str(REPO_ROOT / "visualization" / "gen_video.py"),
                    str(image_dir),
                    "--output",
                    str(video_path),
                    "--fps",
                    str(args.video_fps),
                    "--resize-factor",
                    str(args.video_resize_factor),
                ]
                rc = _run_cmd(cmd, log_path)
                if rc != 0 or not video_path.exists():
                    _fail_state(status, "video_generation", f"video generation failed (exit {rc})")
                    save_status()
                else:
                    status["video_path"] = str(video_path)
                    status["video_generation_success"] = True
                    _complete_state(status, "video_generation")
                    save_status()

    status["end_time"] = _now_iso()
    if status.get("start_time") and status.get("end_time"):
        try:
            start_ts = datetime.fromisoformat(status["start_time"])
            end_ts = datetime.fromisoformat(status["end_time"])
            status["elapsed_s"] = (end_ts - start_ts).total_seconds()
        except Exception:
            status["elapsed_s"] = ""

    if not status.get("failure_state") and status.get("scenario_generation_success"):
        status["success"] = True
        # Create symlinks for successful runs
        try:
            _create_success_symlinks(run_root, run_key, status)
        except Exception as exc:
            log(f"Warning: Failed to create success symlinks: {exc}")

    if not status.get("row_written"):
        _set_state(status, "csv_commit")
        save_status()

        spec = {}
        if status.get("scenario_spec_path"):
            spec = _read_json(Path(status["scenario_spec_path"]))
        scene = _load_scene_objects(status.get("scene_objects_path"))
        manifest = _build_object_manifest(spec, scene)

        run_tag_value = status.get("run_tag", args.run_tag)
        repair_outer_attempts = 0
        if attempt_used:
            repair_outer_attempts = max(0, int(attempt_used) - 1)

        repair_schema_attempts = int(repair_counts.get("schema_generation_repair_attempts", 0))
        repair_object_stage1_json = int(repair_counts.get("object_stage1_json_repair", 0))
        repair_object_stage1_evidence = int(repair_counts.get("object_stage1_evidence_repair", 0))
        repair_object_stage2_json = int(repair_counts.get("object_stage2_json_repair", 0))
        repair_object_stage2_validation = int(repair_counts.get("object_stage2_validation_repair", 0))
        repair_total = (
            repair_outer_attempts
            + repair_schema_attempts
            + repair_object_stage1_json
            + repair_object_stage1_evidence
            + repair_object_stage2_json
            + repair_object_stage2_validation
        )

        scenario_row = {
            "row_type": "scenario_run",
            "run_id": run_id,
            "run_key": run_key,
            "category": category,
            "category_description": _build_category_description(category),
            "run_tag": run_tag_value,
            "variant_index": variant_index,
            "scenario_id": status.get("scenario_id"),
            "scenario_description": status.get("scenario_description"),
            "scenario_spec_path": status.get("scenario_spec_path"),
            "scene_objects_path": status.get("scene_objects_path"),
            "scenario_objects_json": manifest,
            "scenario_generation_success": status.get("scenario_generation_success"),
            "validation_score": status.get("validation_score"),
            "best_rank": status.get("best_rank", ""),
            "kept_best": status.get("kept_best", False),
            "failure_state": status.get("failure_state"),
            "failure_reason": status.get("failure_reason"),
            "pipeline_attempts": args.max_attempts if args.max_attempts > 0 else "unbounded",
            "pipeline_attempts_used": attempt_used or "",
            "repair_attempts_total": repair_total,
            "repair_outer_attempts": repair_outer_attempts,
            "repair_schema_attempts": repair_schema_attempts,
            "schema_template_fallback": metrics.get("schema_template_fallback", 0),
            "repair_object_stage1_json": repair_object_stage1_json,
            "repair_object_stage1_evidence": repair_object_stage1_evidence,
            "repair_object_stage2_json": repair_object_stage2_json,
            "repair_object_stage2_validation": repair_object_stage2_validation,
            "routes_dir": status.get("routes_dir"),
            "route_files": status.get("route_files"),
            "route_generation_success": status.get("route_generation_success"),
            "scene_png_path": status.get("scene_png_path"),
            "carla_results_tag": status.get("carla_results_tag"),
            "carla_results_dir": status.get("carla_results_dir"),
            "carla_image_dir": status.get("carla_image_dir"),
            "carla_simulation_success": status.get("carla_simulation_success"),
            "video_path": status.get("video_path"),
            "video_generation_success": status.get("video_generation_success"),
            "state_history": status.get("state_history"),
            "start_time": status.get("start_time"),
            "end_time": status.get("end_time"),
            "elapsed_s": status.get("elapsed_s"),
            "csv_committed_at": _now_iso(),
        }

        try:
            _append_scenario_row(csv_path, scenario_row)
            status["row_written"] = True
            _complete_state(status, "csv_commit")
        except Exception as exc:
            status["csv_commit_error"] = f"{type(exc).__name__}: {exc}"
            status["prior_failure_state"] = status.get("failure_state")
            _fail_state(status, "csv_commit", status["csv_commit_error"])
        save_status()

    log(f"Completed run {run_key} (success={status.get('success')}, failure_state={status.get('failure_state')})")
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="", help="Run identifier (defaults to timestamp).")
    parser.add_argument(
        "--output-root",
        default="benchmark_runs",
        help="Root directory for audit outputs (relative to repo root unless absolute).",
    )
    parser.add_argument("--categories", nargs="+", default=None, help="Categories to run (default: all).")
    parser.add_argument("--run-tag", type=str, default="base", help="Tag to differentiate runs (included in run_key and CSV).")
    parser.add_argument("--count-per-combination", type=int, default=2, help="How many top scenarios to keep per category (alias: --keep-top).")
    parser.add_argument("--keep-top", type=int, default=None, help="Alias for --count-per-combination.")
    parser.add_argument("--best-of", type=int, default=5, help="Target number of successful samples per category (alias: --samples-per-category). Failed samples do not count toward this target.")
    parser.add_argument("--samples-per-category", type=int, default=None, help="Alias for --best-of (target successful samples per category).")
    parser.add_argument("--max-sample-attempts", type=int, default=None, help="Safety cap on total sample attempts per category (successes + failures). Default: 3x samples-per-category.")
    parser.add_argument("--town", default="Town05")
    parser.add_argument(
        "--highway-town",
        default="Town06",
        help="Town to use for highway categories (default: Town06)",
    )
    parser.add_argument(
        "--t-junction-town",
        default="Town02",
        help="Town to use for T-junction categories (default: Town02)",
    )
    parser.add_argument(
        "--two-lane-corridor-town",
        default="Town02",
        help="Town to use for two-lane corridor categories (default: Town02)",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-AWQ")
    parser.add_argument("--schema-max-new-tokens", type=int, default=1024)
    parser.add_argument("--schema-temperature", type=float, default=0.6)
    parser.add_argument("--schema-top-p", type=float, default=0.9)
    parser.add_argument("--schema-repetition-penalty", type=float, default=1.1)
    parser.add_argument("--schema-max-retries", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=1, help="Per-sample retries before moving to a new sample (set 0 for unlimited).")
    parser.add_argument("--min-score", type=float, default=0.3)
    parser.add_argument("--show-pipeline", action="store_true")
    parser.add_argument("--routes-ego-num", type=int, default=None)
    parser.add_argument("--no-align-routes", action="store_true")
    parser.add_argument("--carla-host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--planner", default="tcp")
    parser.add_argument("--agent", default=None)
    parser.add_argument("--agent-config", default=None)
    parser.add_argument("--track", default="SENSORS")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--carla-seed", type=int, default=2000)
    parser.add_argument("--traffic-seed", type=int, default=2000)
    parser.add_argument(
        "--scenario-parameter",
        default="simulation/leaderboard/leaderboard/scenarios/scenario_parameter_Interdrive_no_npc.yaml",
    )
    parser.add_argument(
        "--scenarios",
        default="simulation/leaderboard/data/scenarios/no_scenarios.json",
    )
    parser.add_argument("--traffic-manager-port", type=int, default=None)
    parser.add_argument("--results-tag-prefix", default="audit")
    parser.add_argument("--resume-carla", action="store_true")
    parser.add_argument("--no-skip-existed", action="store_true")
    parser.add_argument("--dry-run-carla", action="store_true")
    parser.add_argument(
        "--carla-python",
        default="/data/miniconda3/envs/colmdrivermarco2/bin/python3",
        help="Python interpreter to use for CARLA evaluation (defaults to colmdrivermarco2 env).",
    )
    parser.add_argument("--skip-carla", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--force-carla", action="store_true")
    parser.add_argument("--force-video", action="store_true")
    parser.add_argument("--video-fps", type=float, default=5.0)
    parser.add_argument("--video-resize-factor", type=int, default=2)
    parser.add_argument(
        "--video-python",
        default=None,
        help="Python interpreter to use for video generation (defaults to current).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed RNGs for deterministic generation (default: unset for non-deterministic).",
    )
    parser.add_argument("--resume", action="store_true", help="Resume an existing run directory.")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing run directory first.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve aliases and enforce intuitive naming
    if args.keep_top is not None:
        args.count_per_combination = args.keep_top
    if args.samples_per_category is not None:
        args.best_of = args.samples_per_category
    # Ensure we aim for at least as many successes as we plan to keep
    args.best_of = max(args.best_of, args.count_per_combination)
    if args.max_sample_attempts is None:
        args.max_sample_attempts = args.best_of * 3
    
    # If only one category specified and no custom run_id, use category name
    if not args.run_id and args.categories and len(args.categories) == 1:
        category_safe = _safe_name(args.categories[0])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{category_safe}_{timestamp}"
    else:
        run_id = args.run_id or datetime.now().strftime("audit_%Y%m%d_%H%M%S")
    
    run_id = _safe_name(run_id)

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    run_root = output_root / run_id

    if run_root.exists() and not args.resume:
        if args.overwrite:
            shutil.rmtree(run_root)
        else:
            raise SystemExit(f"Run directory already exists: {run_root} (use --resume or --overwrite)")

    run_root.mkdir(parents=True, exist_ok=True)
    csv_path = run_root / "audit.csv"
    config_path = run_root / "config.json"

    _set_seed(args.seed)

    if not config_path.exists() or args.overwrite:
        _atomic_write_json(
            config_path,
            {
                "run_id": run_id,
                "created_at": _now_iso(),
                "args": vars(args),
                "categories": args.categories,
                "run_tag": args.run_tag,
                "states": STATES,
                "csv_fields": CSV_FIELDS,
            },
        )

    categories = args.categories or list(get_available_categories())
    available = set(get_available_categories())
    invalid = [c for c in categories if c not in available]
    if invalid:
        raise SystemExit(f"Invalid categories: {invalid}")

    schema_config = SchemaGenerationConfig(
        model_id=args.model,
        max_new_tokens=args.schema_max_new_tokens,
        temperature=args.schema_temperature,
        top_p=args.schema_top_p,
        repetition_penalty=args.schema_repetition_penalty,
        max_retries=args.schema_max_retries,
        do_sample=True,
        allow_template_fallback=False,
    )

    model, tokenizer = _load_shared_model(args.model)
    generator = SchemaScenarioGenerator(
        schema_config,
        model=model,
        tokenizer=tokenizer,
    )
    scene_validator = SceneValidator()

    for category in categories:
        category_description = _build_category_description(category)
        _ensure_category_row(csv_path, run_id, category, category_description)
        
        statuses: List[Dict[str, Any]] = []
        sample_attempts = 0
        success_count = 0
        target_successes = args.best_of
        while (
            sample_attempts < args.max_sample_attempts
            and success_count < target_successes
        ):
            sample_attempts += 1
            sample_idx = sample_attempts
            status = _run_single(
                args=args,
                run_id=run_id,
                run_root=run_root,
                csv_path=csv_path,
                generator=generator,
                model=model,
                tokenizer=tokenizer,
                scene_validator=scene_validator,
                category=category,
                run_tag=args.run_tag,
                variant_index=sample_idx,
            )
            statuses.append(status)
            if status and status.get("scenario_generation_success") and (status.get("validation_score") or 0) >= args.min_score:
                success_count += 1
                if success_count >= args.count_per_combination and success_count >= target_successes:
                    break
        
        # Rank best runs and update CSV rows
        ranked = _rank_and_mark_best(statuses, args.count_per_combination)
        _update_best_marks(csv_path, ranked)
        _log_kept_best(category, ranked, args.count_per_combination)
        _create_kept_best_symlinks(run_root, ranked)


if __name__ == "__main__":
    main()
