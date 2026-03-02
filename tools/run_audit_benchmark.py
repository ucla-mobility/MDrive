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
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    HAS_IMAGE_SUPPORT = True
except ImportError:
    HAS_IMAGE_SUPPORT = False


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
from tools.carla_validation import (  # noqa: E402
    compute_near_miss as shared_compute_near_miss,
    validate_spawn as shared_validate_spawn,
)

# Debug suite import (deferred to avoid circular imports)
def _generate_debug_report(run_root: Path) -> Optional[Path]:
    """Generate debug visualization report for the run."""
    try:
        from audit_debug_suite import load_run_analysis, generate_html_report
        print("[INFO] Generating debug visualization report...")
        analysis = load_run_analysis(run_root)
        output_path = run_root / "debug_report.html"
        generate_html_report(analysis, output_path)
        print(f"[OK] Debug report saved to: {output_path}")
        return output_path
    except Exception as exc:
        print(f"[WARN] Failed to generate debug report: {exc}")
        return None


STATES = [
    "scenario_generation",
    "repair_loops",
    "route_generation",
    "baseline_validation",
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
        # Optional debug artifacts
        "csp_debug_path",
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


def _run_csp_debug(run_dir: Path, out_path: Optional[Path] = None) -> Optional[Path]:
    """
    Optional helper: rerun CSP solver on saved attempt artifacts and emit rich debug JSON.
    Safe to call even if debug script is missing; failures are logged but non-fatal.
    """
    script = REPO_ROOT / "scripts" / "rerun_csp_debug.py"
    if not script.exists():
        print(f"[DEBUG] CSP debug script not found at {script}, skipping rerun.")
        return None
    out_path = out_path or (run_dir / "scene_objects_csp_rerun_debug.json")
    cmd = [
        sys.executable,
        str(script),
        "--run-dir",
        str(run_dir),
        "--out",
        str(out_path),
    ]
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[DEBUG] Failed to run CSP debug rerun: {exc}")
        return None
    if res.returncode != 0:
        print(f"[DEBUG] CSP debug rerun failed (code {res.returncode}): {res.stderr.strip()}")
        return None
    if res.stdout.strip():
        print(res.stdout.strip())
    return out_path


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
        # Roundabout is only available in Town03
        if info.map.topology == TopologyType.ROUNDABOUT:
            return "Town03"
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


# =============================================================================
# CARLA baseline validation helpers
# =============================================================================


def _load_carla_module(carla_root: Path):
    """
    Best-effort import of the CARLA egg bundled under repo_root/carla.
    Returns the imported module or raises the original ImportError.
    """
    try:
        import carla  # type: ignore
        return carla
    except Exception:
        pass

    dist_dir = carla_root / "PythonAPI" / "carla" / "dist"
    eggs = sorted(dist_dir.glob("carla-*.egg"))
    if not eggs:
        raise FileNotFoundError(f"No CARLA egg found under {dist_dir}")
    py3_eggs = [egg for egg in eggs if "-py3" in egg.name]
    egg_path = str(py3_eggs[0] if py3_eggs else eggs[0])
    if egg_path not in sys.path:
        sys.path.append(egg_path)
    import carla  # type: ignore  # noqa: E401
    return carla


def _load_actor_manifest(routes_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    manifest_path = routes_dir / "actors_manifest.json"
    if not manifest_path.exists():
        return {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_route_file(route_path: Path, carla_module) -> List[Any]:
    """
    Parse a simple route XML into a list of carla.Transform.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(route_path)
    root = tree.getroot()
    route_nodes = list(root.iter("route"))
    if not route_nodes:
        return []
    waypoints = []
    for wp in route_nodes[0].iter("waypoint"):
        x = float(wp.get("x", "0"))
        y = float(wp.get("y", "0"))
        z = float(wp.get("z", "0"))
        yaw = float(wp.get("yaw", "0"))
        transform = carla_module.Transform(
            carla_module.Location(x=x, y=y, z=z),
            carla_module.Rotation(yaw=yaw),
        )
        waypoints.append(transform)
    return waypoints


def _destroy_actors(actors: List[Any]) -> None:
    for actor in actors:
        try:
            actor.destroy()
        except Exception:
            pass


def validate_spawn(world, expected_actor_ids: List[int]) -> Tuple[bool, str]:
    """
    Check that all expected actor ids resolve to live actors in the world.
    """
    return shared_validate_spawn(world, expected_actor_ids)


def _bounding_sphere_radius(actor) -> float:
    try:
        bb = actor.bounding_box
        # Use the smaller footprint dimension to avoid over-flagging tailgating as collision.
        return max(min(bb.extent.x, bb.extent.y), 0.8)
    except Exception:
        return 1.5


def _actor_velocity(actor):
    try:
        vel = actor.get_velocity()
        return vel
    except Exception:
        return None


def _project_location(loc, vel, dt, carla_module):
    if vel is None:
        return loc
    return carla_module.Location(
        x=loc.x + vel.x * dt,
        y=loc.y + vel.y * dt,
        z=loc.z + vel.z * dt,
    )


def _unit_vector(vec) -> Tuple[float, float, float]:
    mag = math.sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z)
    if mag < 1e-6:
        return (0.0, 0.0, 0.0)
    return (vec.x / mag, vec.y / mag, vec.z / mag)


def _actor_radius(actor, margin: float = 0.2) -> float:
    try:
        bb = actor.bounding_box
        r = max(bb.extent.x, bb.extent.y) + margin
        return max(r, 0.4)  # light floor for walkers/bikes
    except Exception:
        return 1.5


def _closing_speed(ego_loc, ego_vel, other_loc, other_vel) -> float:
    if ego_vel is None or other_vel is None:
        return 0.0
    dx = ego_loc.x - other_loc.x
    dy = ego_loc.y - other_loc.y
    dz = ego_loc.z - other_loc.z
    mag = math.sqrt(dx * dx + dy * dy + dz * dz)
    if mag < 1e-6:
        return 0.0
    ux, uy, uz = dx / mag, dy / mag, dz / mag
    relx = other_vel.x - ego_vel.x
    rely = other_vel.y - ego_vel.y
    relz = other_vel.z - ego_vel.z
    return relx * ux + rely * uy + relz * uz


def _compute_near_miss(
    ego_actor,
    other_actors,
    carla_module,
    horizon_s: float,
    step_s: float,
    ttc_thresh: float,
    ttc_severe: float,
    closing_min: float,
    role_lookup: Optional[Dict[int, str]] = None,
    debug_hits: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[bool, bool, float]:
    return shared_compute_near_miss(
        ego_actor=ego_actor,
        other_actors=other_actors,
        carla_module=carla_module,
        horizon_s=horizon_s,
        step_s=step_s,
        ttc_thresh=ttc_thresh,
        ttc_severe=ttc_severe,
        closing_min=closing_min,
        role_lookup=role_lookup,
        debug_hits=debug_hits,
    )


def _run_baseline_constant_velocity(
    carla_module,
    client,
    world,
    routes_dir: Path,
    args: argparse.Namespace,
    log_fn,
) -> Dict[str, Any]:
    """
    Spawn actors from routes_dir and run a constant-velocity baseline controller.
    Returns a result dict with keys: status, reason, rc, ds, near_miss, min_ttc, tags.
    """
    manifest = _load_actor_manifest(routes_dir)
    if not manifest:
        return {"status": "reject", "reason": "missing_manifest"}

    blueprint_library = world.get_blueprint_library()
    spawned: List[Any] = []
    sensors: List[Any] = []
    expected_ids: List[int] = []
    ego_actors: List[Any] = []
    role_lookup: Dict[int, str] = {}
    event_log: List[Dict[str, Any]] = []

    def pick_blueprint(role: str, model: Optional[str] = None):
        """
        Choose a sensible blueprint per role with fallbacks.
        """
        if model:
            try:
                return blueprint_library.find(model)
            except Exception:
                pass
        if role == "pedestrian":
            walkers = [bp for bp in blueprint_library.filter("walker.pedestrian.*")]
            if walkers:
                return random.choice(walkers)
        if role == "static":
            props = [bp for bp in blueprint_library.filter("static.prop.*")]
            if props:
                return random.choice(props)
        # default vehicle
        vehicles = [bp for bp in blueprint_library.filter("vehicle.*model3*")] or blueprint_library.filter("vehicle.*")
        return random.choice(vehicles)

    def _spawn_with_retries(bp, transform, role_name, retries=5):
        last_actor = None
        for i in range(retries):
            tf = carla_module.Transform(
                carla_module.Location(
                    x=transform.location.x,
                    y=transform.location.y,
                    z=transform.location.z + 0.2 * i,
                ),
                transform.rotation,
            )
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", role_name)
            try:
                actor = world.try_spawn_actor(bp, tf)
            except Exception:
                actor = None
            if actor:
                return actor
            time.sleep(0.05)
        return last_actor

    try:
        # Cleanup any leftover heroes from previous attempts to avoid collisions
        leftovers = [a for a in world.get_actors().filter("vehicle.*") if a.attributes.get("role_name", "").startswith("hero")]
        if leftovers:
            log_fn(f"[BASELINE] destroying {len(leftovers)} leftover hero vehicles before spawn")
        _destroy_actors(leftovers)

        # Spawn order: statics, pedestrians, npc, ego
        spawn_plan = [
            ("static", manifest.get("static", [])),
            ("pedestrian", manifest.get("pedestrian", [])),
            ("npc", manifest.get("npc", [])),
            ("ego", manifest.get("ego", [])),
        ]

        def _spawn_candidates_from_map(start_tf: Any, offsets: List[float], carla_map) -> List[Any]:
            """
            Use CARLA map to walk forward along the lane from the first aligned pose.
            Offsets are distances in meters.
            """
            if start_tf is None or carla_map is None:
                return []
            try:
                wp0 = carla_map.get_waypoint(start_tf.location, project_to_road=True, lane_type=carla_module.LaneType.Driving)
            except Exception:
                return [start_tf]

            candidates: List[Any] = []
            for offset in offsets:
                if offset <= 0:
                    candidates.append(wp0.transform)
                    continue
                try:
                    nxt = wp0.next(offset)
                    if nxt:
                        candidates.append(nxt[0].transform)
                    else:
                        candidates.append(wp0.transform)
                except Exception:
                    candidates.append(wp0.transform)

            # de-duplicate while preserving order
            uniq = []
            seen = set()
            for tf in candidates:
                key = (round(tf.location.x, 3), round(tf.location.y, 3), round(tf.location.z, 3), round(tf.rotation.yaw, 3))
                if key not in seen:
                    seen.add(key)
                    uniq.append(tf)
            return uniq

        spawn_offsets = [0.0, 5.0, 10.0]

        for role_group, entries in spawn_plan:
            for idx, entry in enumerate(entries):
                route_rel = entry.get("file")
                if not route_rel:
                    continue
                route_path = (routes_dir / route_rel).resolve()
                waypoints = _parse_route_file(route_path, carla_module)
                if not waypoints:
                    return {"status": "reject", "reason": f"empty_route_{route_rel}"}
                spawn_tf_candidates = _spawn_candidates_from_map(waypoints[0], spawn_offsets, world.get_map()) if waypoints else []
                model = entry.get("model")
                bp = pick_blueprint(role_group, model)
                role_name = entry.get("kind") or role_group
                if role_group == "ego":
                    role_name = f"hero_{len(ego_actors)}"
                actor = None
                for cand_tf in spawn_tf_candidates:
                    log_fn(f"[BASELINE] spawning {role_group} idx={idx} model={model} role={role_name} at {cand_tf.location}")
                    actor = _spawn_with_retries(bp, cand_tf, role_name)
                    if actor:
                        break
                    else:
                        log_fn(f"[BASELINE] spawn attempt failed for {role_group} idx={idx} at {cand_tf.location}")
                if actor is None:
                    log_fn(f"[BASELINE] spawn failed for {role_group} idx={idx} model={model} role={role_name} after {len(spawn_tf_candidates)} candidates")
                    return {"status": "reject", "reason": f"spawn_failed_{role_group}_{idx}"}
                spawned.append(actor)
                expected_ids.append(actor.id)
                try:
                    role_lookup[actor.id] = actor.attributes.get("role_name", role_name) if hasattr(actor, "attributes") else role_name
                except Exception:
                    role_lookup[actor.id] = role_name
                if role_group == "ego":
                    ego_actors.append(actor)

        ok, reason = validate_spawn(world, expected_ids)
        if not ok:
            return {"status": "reject", "reason": f"spawn_validation_{reason}"}

        # Warmup ticks: ensure motion for egos
        warmup_ticks = max(1, int(args.baseline_warmup_ticks))
        ego_initial = [actor.get_transform() for actor in ego_actors]
        moved_flags = [False for _ in ego_actors]
        for _ in range(warmup_ticks):
            for ego in ego_actors:
                ego.apply_control(carla_module.VehicleControl(throttle=0.2, steer=0.0, brake=0.0))
            world.tick()
            for i, actor in enumerate(ego_actors):
                now_tf = actor.get_transform()
                vel = _actor_velocity(actor)
                speed = 0.0
                if vel:
                    speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
                if now_tf.location.distance(ego_initial[i].location) > args.baseline_min_motion or speed > args.baseline_min_motion:
                    moved_flags[i] = True
        if ego_actors and not all(moved_flags):
            bad = [i for i, m in enumerate(moved_flags) if not m]
            return {"status": "reject", "reason": f"ego_stuck_{bad}"}

        # Baseline control loop
        waypoint_cache: Dict[int, List[Any]] = {}
        route_lengths: Dict[int, float] = {}
        wp_reached: Dict[int, int] = {}
        collisions: Dict[int, bool] = {}
        near_miss_triggered = False
        min_ttc = float("inf")
        severe_trigger = False
        consecutive_hits: Dict[int, int] = {}
        hit_streak_needed = 3
        ttc_severe = min(args.baseline_ttc_thresh, 0.65)
        closing_min = 1.0

        # Preload waypoints for egos
        for idx, entry in enumerate(manifest.get("ego", [])):
            route_rel = entry.get("file")
            route_path = (routes_dir / route_rel).resolve()
            waypoints = _parse_route_file(route_path, carla_module)
            waypoint_cache[idx] = waypoints
            wp_reached[idx] = 0
            route_lengths[idx] = max(1.0, float(len(waypoints)))
            collisions[idx] = False
            consecutive_hits[idx] = 0
        near_miss_log: List[Dict[str, Any]] = []

        # Attach collision sensors to egos
        for ego_idx, ego in enumerate(ego_actors):
            col_bp = blueprint_library.find("sensor.other.collision")
            sensor = world.spawn_actor(col_bp, carla_module.Transform(), attach_to=ego)
            sensors.append(sensor)

            def _make_cb(ei):
                def _on_col(event):
                    collisions[ei] = True
                    try:
                        other_id = event.other_actor.id if getattr(event, "other_actor", None) else None
                        impulse = None
                        if hasattr(event, "normal_impulse"):
                            ni = event.normal_impulse
                            try:
                                impulse = math.sqrt(ni.x ** 2 + ni.y ** 2 + ni.z ** 2)
                            except Exception:
                                impulse = None
                        event_log.append({
                            "type": "collision",
                            "ego_index": ei,
                            "ego_id": ego_actors[ei].id if ei < len(ego_actors) else None,
                            "other_id": other_id,
                            "other_role": role_lookup.get(other_id, ""),
                            "frame": getattr(event, "frame", None),
                            "impulse": impulse,
                        })
                    except Exception:
                        pass
                return _on_col

            sensor.listen(_make_cb(ego_idx))

        # Setup traffic manager for autopilot
        use_autopilot = getattr(args, "baseline_use_autopilot", True)
        if use_autopilot:
            try:
                tm_port = getattr(args, "traffic_manager_port", None)
                tm_port = int(tm_port) if tm_port is not None else 8000
                traffic_manager = client.get_trafficmanager(tm_port)
                traffic_manager.set_synchronous_mode(True)
                traffic_manager.set_random_device_seed(0)
                log_fn(f"[BASELINE] using CARLA traffic manager autopilot on port {tm_port}")

                def tm_call(name, *call_args):
                    fn = getattr(traffic_manager, name, None)
                    if not fn:
                        return
                    try:
                        fn(*call_args)
                    except TypeError:
                        # API signature mismatch (e.g., version differences) – skip quietly
                        return
                
                for ego_idx, ego in enumerate(ego_actors):
                    # Enable autopilot for this vehicle
                    ego.set_autopilot(True, tm_port)
                    # Configure traffic manager for "ghost mode" - ignore everything, follow route exactly
                    tm_call("ignore_lights_percentage", ego, 100.0)
                    tm_call("ignore_walkers_percentage", ego, 100.0)
                    tm_call("ignore_vehicles_percentage", ego, 100.0)
                    tm_call("ignore_signs_percentage", ego, 100.0)
                    tm_call("distance_to_leading_vehicle", ego, 0.0)
                    # Disable collision avoidance: 0.9.12 expects (actor, other_actor, bool); we skip to avoid signature issues.
                    # tm_call("collision_detection", ego, ego, False)
                    # Set constant speed (convert from m/s to % over speed limit)
                    target_speed_ms = args.baseline_target_speed  # e.g., 8.0 m/s
                    speed_diff = -20.0  # Negative = slower than speed limit (placeholder, overridden by lane_change settings)
                    tm_call("vehicle_percentage_speed_difference", ego, speed_diff)
                    tm_call("auto_lane_change", ego, False)
                    # In 0.9.12+ the API is set_percentage_keep_right_rule
                    if hasattr(traffic_manager, "set_percentage_keep_right_rule"):
                        tm_call("set_percentage_keep_right_rule", ego, 100.0)
                    else:
                        tm_call("keep_right_rule_percentage", ego, 100.0)
                    log_fn(f"[BASELINE] enabled ghost-mode autopilot for ego {ego_idx}")
            except Exception as exc:
                log_fn(f"[BASELINE] failed to setup traffic manager autopilot: {exc}")
                use_autopilot = False

        # Attach camera sensors if baseline image saving is enabled
        camera_data: Dict[int, Any] = {}
        baseline_image_dir = getattr(args, "baseline_image_dir", None)
        if baseline_image_dir and HAS_IMAGE_SUPPORT:
            baseline_image_dir = Path(baseline_image_dir)
            baseline_image_dir.mkdir(parents=True, exist_ok=True)
            log_fn(f"[BASELINE] saving images to {baseline_image_dir}")
            
            for ego_idx, ego in enumerate(ego_actors):
                cam_bp = blueprint_library.find("sensor.camera.rgb")
                cam_bp.set_attribute("image_size_x", "800")
                cam_bp.set_attribute("image_size_y", "600")
                cam_bp.set_attribute("fov", "90")
                cam_transform = carla_module.Transform(
                    carla_module.Location(x=1.5, z=2.4),
                    carla_module.Rotation(pitch=-15)
                )
                camera = world.spawn_actor(cam_bp, cam_transform, attach_to=ego)
                sensors.append(camera)
                camera_data[ego_idx] = {"latest_image": None, "frame_count": 0}

                def _make_cam_cb(ei):
                    def _on_image(image):
                        camera_data[ei]["latest_image"] = image
                    return _on_image

                camera.listen(_make_cam_cb(ego_idx))

        def compute_control(actor, ego_idx):
            waypoints = waypoint_cache[ego_idx]
            if not waypoints:
                return None
            idx = wp_reached[ego_idx]
            idx = min(idx, len(waypoints) - 1)
            target_idx = min(len(waypoints) - 1, idx + args.baseline_waypoint_lookahead)
            target = waypoints[target_idx]
            loc = actor.get_transform().location
            vec = target.location - loc
            desired_yaw = math.atan2(vec.y, vec.x)
            current_yaw = math.radians(actor.get_transform().rotation.yaw)
            yaw_err = desired_yaw - current_yaw
            while yaw_err > math.pi:
                yaw_err -= 2 * math.pi
            while yaw_err < -math.pi:
                yaw_err += 2 * math.pi
            steer = max(-1.0, min(1.0, args.baseline_steer_kp * yaw_err))

            vel = _actor_velocity(actor)
            speed = 0.0
            if vel:
                speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
            desired = min(args.baseline_target_speed, args.baseline_speed_cap)
            if speed < desired:
                throttle = max(0.0, min(1.0, args.baseline_speed_kp * (desired - speed)))
                brake = 0.0
            else:
                throttle = 0.0
                brake = max(0.0, min(1.0, (speed - desired) * 0.2))

            ctrl = carla_module.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=brake,
                hand_brake=False,
                reverse=False,
            )
            return ctrl

        max_ticks = max(1, int(args.baseline_max_ticks))
        waypoint_thresh = args.baseline_waypoint_thresh
        save_interval = getattr(args, "baseline_save_interval", 10)  # Save every N frames

        for tick_idx in range(max_ticks):
            # Only apply manual controls if not using autopilot
            if not use_autopilot:
                for ego_idx, ego in enumerate(ego_actors):
                    if hasattr(ego, "is_alive") and not ego.is_alive:
                        continue
                    ctrl = compute_control(ego, ego_idx)
                    if ctrl:
                        ego.apply_control(ctrl)

            try:
                world.tick()
            except RuntimeError as exc:
                if "destroyed actor" in str(exc).lower():
                    log_fn("[BASELINE] world.tick failed due to destroyed actor; aborting baseline early.")
                    return {"status": "reject", "reason": "destroyed_actor_tick"}
                raise
            
            # Save camera frames with event labels
            if baseline_image_dir and HAS_IMAGE_SUPPORT and (tick_idx % save_interval == 0 or tick_idx < 5):
                for ego_idx, ego in enumerate(ego_actors):
                    if hasattr(ego, "is_alive") and not ego.is_alive:
                        continue
                    if ego_idx not in camera_data:
                        continue
                    img = camera_data[ego_idx]["latest_image"]
                    if img is None:
                        continue
                    
                    # Convert CARLA image to numpy array
                    array = np.frombuffer(img.raw_data, dtype=np.uint8)
                    array = array.reshape((img.height, img.width, 4))[:, :, :3]
                    pil_img = Image.fromarray(array)
                    
                    # Add event labels
                    draw = ImageDraw.Draw(pil_img)
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
                    except Exception:
                        font = ImageFont.load_default()
                    
                    labels = []
                    if collisions.get(ego_idx, False):
                        labels.append("COLLISION")
                    if consecutive_hits.get(ego_idx, 0) >= 2:
                        labels.append(f"NEAR-MISS (streak={consecutive_hits[ego_idx]})")
                    
                    # Draw labels on image
                    y_offset = 10
                    for label in labels:
                        bbox = draw.textbbox((0, 0), label, font=font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                        x = img.width - text_width - 20
                        draw.rectangle([x - 5, y_offset - 5, x + text_width + 5, y_offset + text_height + 5], 
                                       fill=(255, 0, 0, 200))
                        draw.text((x, y_offset), label, fill=(255, 255, 255), font=font)
                        y_offset += text_height + 15
                    
                    # Add frame info
                    info_text = f"Ego {ego_idx} | Tick {tick_idx} | RC {wp_reached[ego_idx]}/{len(waypoint_cache[ego_idx])}"
                    draw.text((10, 10), info_text, fill=(255, 255, 255), font=font)
                    
                    # Save frame
                    frame_path = baseline_image_dir / f"ego{ego_idx}_frame{tick_idx:06d}.jpg"
                    pil_img.save(frame_path, quality=85)
                    camera_data[ego_idx]["frame_count"] += 1

            # update progress + near miss
            for ego_idx, ego in enumerate(ego_actors):
                if hasattr(ego, "is_alive") and not ego.is_alive:
                    continue
                others = []
                for a in spawned:
                    try:
                        if a.id != ego.id:
                            others.append(a)
                    except Exception:
                        continue
                waypoints = waypoint_cache[ego_idx]
                if not waypoints:
                    continue
                next_idx = wp_reached[ego_idx]
                next_idx = min(next_idx, len(waypoints) - 1)
                try:
                    if ego.get_location().distance(waypoints[next_idx].location) < waypoint_thresh:
                        wp_reached[ego_idx] = min(len(waypoints) - 1, next_idx + 1)
                except Exception:
                    continue

                hit, severe, ttc = _compute_near_miss(
                    ego,
                    others,
                    carla_module,
                    args.baseline_ttc_horizon,
                    args.baseline_ttc_step,
                    args.baseline_ttc_thresh,
                    ttc_severe,
                    closing_min,
                    role_lookup=role_lookup,
                    debug_hits=near_miss_log,
                )
                if hit:
                    consecutive_hits[ego_idx] += 1
                else:
                    consecutive_hits[ego_idx] = 0

                if severe:
                    severe_trigger = True

                min_ttc = min(min_ttc, ttc)

            # early exit if all egos reached goal
            if all(wp_reached[i] >= len(waypoint_cache[i]) - 1 for i in wp_reached):
                break

        # near miss decision with persistence/severity
        hit_streak = max(consecutive_hits.values()) if consecutive_hits else 0
        if severe_trigger or hit_streak >= hit_streak_needed:
            near_miss_triggered = True
        # Metrics
        rc_vals = []
        ds_vals = []
        for ego_idx in waypoint_cache.keys():
            rc = wp_reached[ego_idx] / route_lengths[ego_idx]
            collided = collisions.get(ego_idx, False)
            ds = rc if not collided else rc * 0.5
            rc_vals.append(rc)
            ds_vals.append(ds)

        rc_min = min(rc_vals) if rc_vals else 0.0
        ds_min = min(ds_vals) if ds_vals else 0.0

        easy = rc_min >= args.baseline_easy_rc and ds_min >= args.baseline_easy_ds
        tags: List[str] = []
        status = "accept"
        reason = ""
        verbose_baseline = getattr(args, "show_pipeline", False)
        if verbose_baseline:
            log_fn(
                f"[BASELINE][TTC] min_ttc={min_ttc:.3f} severe={severe_trigger} "
                f"hit_streak={hit_streak if 'hit_streak' in locals() else 0} "
                f"needed={hit_streak_needed}"
            )
        if easy:
            if near_miss_triggered:
                tags.append("accepted_due_to_ttc_nearmiss")
                reason = "accepted_due_to_ttc_nearmiss"
            else:
                status = "reject"
                reason = "rejected_easy_no_nearmiss"
                tags.append("rejected_easy_no_nearmiss")
        else:
            tags.append("baseline_ok")

        result = {
            "status": status,
            "reason": reason or ("accepted_due_to_ttc_nearmiss" if near_miss_triggered and easy else ""),
            "rc": rc_min,
            "ds": ds_min,
            "near_miss": near_miss_triggered,
            "min_ttc": min_ttc if near_miss_triggered else float("inf"),
            "tags": tags,
        }
        # Persist detailed per-tick/debug info for postmortem analysis.
        try:
            log_dir = Path(getattr(args, "baseline_image_dir", "") or routes_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            with open(log_dir / "baseline_events.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "collisions": event_log,
                        "near_miss_hits": near_miss_log,
                    },
                    f,
                    indent=2,
                )
        except Exception as exc:
            log_fn(f"[BASELINE] failed to write baseline_events.json: {exc}")

        return result
    finally:
        _destroy_actors(sensors + spawned)


def _run_baseline_validation(args: argparse.Namespace, routes_dir: Path, log_fn) -> Dict[str, Any]:
    """
    Connect to CARLA, enforce synchronous mode, spawn actors, run baseline.
    """
    carla_root = REPO_ROOT / "carla912"
    try:
        carla_module = _load_carla_module(carla_root)
    except Exception as exc:  # noqa: BLE001
        log_fn(f"[BASELINE] Failed to import CARLA: {exc}")
        return {"status": "reject", "reason": f"carla_import_failed_{exc}"}

    client = carla_module.Client(args.carla_host, args.carla_port)
    client.set_timeout(5.0)

    desired_town = None
    manifest = _load_actor_manifest(routes_dir)
    for role_group in ("ego", "npc", "pedestrian", "static"):
        entries = manifest.get(role_group, [])
        if entries:
            desired_town = entries[0].get("town")
            break
    if not desired_town:
        desired_town = args.town

    try:
        world = client.get_world()
        current_map = world.get_map().name
        log_fn(f"[BASELINE] current map: {current_map}, desired: {desired_town}")
        if desired_town and desired_town.lower() not in current_map.lower():
            world = client.load_world(desired_town)
            log_fn(f"[BASELINE] loaded map: {world.get_map().name}")
    except Exception as exc:  # noqa: BLE001
        log_fn(f"[BASELINE] Failed to get/load world: {exc}")
        return {"status": "reject", "reason": f"carla_world_failed_{exc}"}

    original_settings = world.get_settings()
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        if args.baseline_delta_seconds:
            settings.fixed_delta_seconds = args.baseline_delta_seconds
        world.apply_settings(settings)
        log_fn(f"[BASELINE] world settings -> sync={settings.synchronous_mode} delta={settings.fixed_delta_seconds}")

        result = _run_baseline_constant_velocity(
            carla_module=carla_module,
            client=client,
            world=world,
            routes_dir=routes_dir,
            args=args,
            log_fn=log_fn,
        )
        return result
    finally:
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass


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
        if status.get("failure_state"):
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

            # Optional: rerun CSP locally and emit rich debug artifact
            if args.csp_debug and scene_path:
                attempt_dir = Path(scene_path).parent
                csp_debug_path = _run_csp_debug(attempt_dir)
                if csp_debug_path:
                    attempt_record["csp_debug_path"] = str(csp_debug_path)
                    status["csp_debug_path"] = str(csp_debug_path)

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

    # ------------------------------------------------------------------
    # Baseline validation gate (constant-velocity + TTC near-miss + spawn checks)
    # ------------------------------------------------------------------
    if (
        status.get("route_generation_success")
        and not args.skip_carla
        and "baseline_validation" not in status.get("completed_states", [])
    ):
        _set_state(status, "baseline_validation")
        save_status()

        routes_dir = Path(status.get("routes_dir", ""))
        if not routes_dir.exists():
            _fail_state(status, "baseline_validation", "routes directory missing for baseline")
            save_status()
        else:
            log("[BASELINE] Starting baseline validation (constant velocity).")
            if args.baseline_save_images:
                baseline_image_dir = run_root / run_key / "baseline_images"
                args.baseline_image_dir = str(baseline_image_dir)
                status["baseline_image_dir"] = str(baseline_image_dir)
                # Add a symlink inside the current pipeline attempt for quick access
                attempt_link = scenario_dir / "baseline_images"
                try:
                    if attempt_link.exists() or attempt_link.is_symlink():
                        attempt_link.unlink()
                    attempt_link.symlink_to(baseline_image_dir, target_is_directory=True)
                except Exception:
                    pass
            baseline_result = _run_baseline_validation(args, routes_dir, log)
            status.setdefault("metrics", {})["baseline"] = baseline_result
            try:
                log(
                    "[BASELINE] rc={:.3f} ds={:.3f} near_miss={} status={} reason={}".format(
                        float(baseline_result.get("rc", 0.0)),
                        float(baseline_result.get("ds", 0.0)),
                        baseline_result.get("near_miss"),
                        baseline_result.get("status"),
                        baseline_result.get("reason", ""),
                    )
                )
            except Exception:
                log(f"[BASELINE] summary {baseline_result}")
            if baseline_result.get("status") == "reject":
                reason = baseline_result.get("reason", "baseline_rejected")
                _fail_state(status, "baseline_validation", reason)
                save_status()
            else:
                _complete_state(status, "baseline_validation")
                save_status()

    if (
        status.get("route_generation_success")
        and not args.skip_carla
        and not args.baseline_only
        and not status.get("failure_state")
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
            log(f"[CARLA] results dir: {results_root}")

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
                        log(f"[CARLA] image dir: {image_dir}")
                        status["carla_simulation_success"] = True
                        _complete_state(status, "carla_simulation")
                        save_status()

    if (
        status.get("carla_simulation_success")
        and not args.skip_video
        and not args.baseline_only
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
                    log(f"[VIDEO] generation failed (exit {rc}); marking video_generation_success=false but not failing run")
                    status["video_generation_success"] = False
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
            "csp_debug_path": status.get("csp_debug_path"),
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
    parser.add_argument("--schema-temperature", type=float, default=0.3)
    parser.add_argument("--schema-top-p", type=float, default=0.9)
    parser.add_argument("--schema-repetition-penalty", type=float, default=1.1)
    parser.add_argument("--schema-max-retries", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=1, help="Per-sample retries before moving to a new sample (set 0 for unlimited).")
    parser.add_argument("--min-score", type=float, default=0.3)
    parser.add_argument("--show-pipeline", action="store_true")
    parser.add_argument("--routes-ego-num", type=int, default=None)
    parser.add_argument("--csp-debug", action="store_true", help="Rerun CSP solver for each pipeline attempt and emit scene_objects_csp_rerun_debug.json")
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
    # Baseline validation knobs
    parser.add_argument("--baseline-delta-seconds", type=float, default=0.05)
    parser.add_argument("--baseline-warmup-ticks", type=int, default=15)
    parser.add_argument("--baseline-min-motion", type=float, default=0.05)
    parser.add_argument("--baseline-use-manual-control", dest="baseline_use_autopilot", action="store_false", default=True, help="Use simple waypoint-following controller instead of autopilot (default: use autopilot)")
    parser.add_argument("--baseline-target-speed", type=float, default=8.0)
    parser.add_argument("--baseline-speed-cap", type=float, default=12.0)
    parser.add_argument("--baseline-speed-kp", type=float, default=0.4)
    parser.add_argument("--baseline-steer-kp", type=float, default=0.6)
    parser.add_argument("--baseline-waypoint-lookahead", type=int, default=4)
    parser.add_argument("--baseline-waypoint-thresh", type=float, default=2.5)
    parser.add_argument("--baseline-max-ticks", type=int, default=600)
    parser.add_argument("--baseline-easy-ds", type=float, default=0.95)
    parser.add_argument("--baseline-easy-rc", type=float, default=0.95)
    parser.add_argument("--baseline-ttc-thresh", type=float, default=0.9)
    parser.add_argument("--baseline-ttc-horizon", type=float, default=1.8)
    parser.add_argument("--baseline-ttc-step", type=float, default=0.15)
    parser.add_argument("--baseline-save-images", action="store_true", help="Save camera frames from baseline validation")
    parser.add_argument("--baseline-save-interval", type=int, default=10, help="Save every Nth frame (lower = more frames)")
    parser.add_argument(
        "--carla-python",
        default="/data/miniconda3/envs/colmdrivermarco2/bin/python3",
        help="Python interpreter to use for CARLA evaluation (defaults to colmdrivermarco2 env).",
    )
    parser.add_argument("--skip-carla", action="store_true")
    parser.add_argument("--baseline-only", action="store_true", help="Run only baseline validation, skip main planner simulation")
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
    # Debug suite options
    parser.add_argument(
        "--debug-report",
        action="store_true",
        default=True,
        help="Generate HTML debug visualization report after run (default: enabled)",
    )
    parser.add_argument(
        "--no-debug-report",
        action="store_true",
        help="Disable automatic debug report generation",
    )
    parser.add_argument(
        "--open-debug-report",
        action="store_true",
        help="Open debug report in browser after generation",
    )
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

    # Skip local model loading for OpenAI API models
    is_openai_model = args.model.startswith("gpt-") or args.model.startswith("o1-") or args.model.startswith("o3-") or args.model.startswith("o5-")
    if is_openai_model:
        model, tokenizer = None, None
        print(f"[INFO] Using OpenAI API model: {args.model}")
    else:
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
    
    # Generate debug visualization report
    if args.debug_report and not args.no_debug_report:
        report_path = _generate_debug_report(run_root)
        if report_path and args.open_debug_report:
            import webbrowser
            webbrowser.open(f"file://{report_path.resolve()}")
    
    print(f"\n{'='*60}")
    print(f"Audit benchmark complete: {run_root}")
    print(f"CSV log: {csv_path}")
    if args.debug_report and not args.no_debug_report:
        print(f"Debug report: {run_root / 'debug_report.html'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
