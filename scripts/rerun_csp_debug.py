#!/usr/bin/env python3
"""
Rerun the CSP solver on a saved pipeline attempt and dump rich debug JSON.

Inputs:
  --run-dir: directory containing scene_objects_debug.json and picked_paths_detailed.json
  --out: optional output path (defaults to <run-dir>/scene_objects_csp_rerun_debug.json)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
from typing import Any, Dict, List

import numpy as np

# Ensure project root is on sys.path for package imports
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_step_modules():
    """
    Load step_05_object_placer modules without importing the package __init__
    (avoids transformers dependency).
    """
    import types

    base_pkg = "scenario_generator"
    sg = types.ModuleType(base_pkg)
    sg.__path__ = [str(ROOT / "scenario_generator")]
    sg_pipeline = types.ModuleType(f"{base_pkg}.pipeline")
    sg_pipeline.__path__ = [str(ROOT / "scenario_generator" / "pipeline")]
    sg_step = types.ModuleType(f"{base_pkg}.pipeline.step_05_object_placer")
    sg_step.__path__ = [
        str(ROOT / "scenario_generator" / "pipeline" / "step_05_object_placer")
    ]
    sys.modules[base_pkg] = sg
    sys.modules[f"{base_pkg}.pipeline"] = sg_pipeline
    sys.modules[f"{base_pkg}.pipeline.step_05_object_placer"] = sg_step

    def _load(mod_name: str):
        path = (
            ROOT
            / "scenario_generator"
            / "pipeline"
            / "step_05_object_placer"
            / f"{mod_name}.py"
        )
        full = f"{base_pkg}.pipeline.step_05_object_placer.{mod_name}"
        spec = importlib.util.spec_from_file_location(
            full, path, submodule_search_locations=sg_step.__path__
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
        return mod

    csp = _load("csp")
    nodes = _load("nodes")
    spawn = _load("spawn")
    return csp, nodes, spawn


# Lazy-load modules to avoid importing transformers-heavy main
CSP_MOD, NODES_MOD, SPAWN_MOD = _load_step_modules()
CandidatePlacement = CSP_MOD.CandidatePlacement
_compute_merge_min_s_by_vehicle = CSP_MOD._compute_merge_min_s_by_vehicle
solve_weighted_csp_with_extension = CSP_MOD.solve_weighted_csp_with_extension
_override_seg_points_with_picked = NODES_MOD._override_seg_points_with_picked
build_segments_from_nodes = NODES_MOD.build_segments_from_nodes
load_nodes = NODES_MOD.load_nodes
compute_spawn_from_anchor = SPAWN_MOD.compute_spawn_from_anchor


def _load_run(run_dir: pathlib.Path) -> Dict[str, Any]:
    debug_path = run_dir / "scene_objects_debug.json"
    picked_path = run_dir / "picked_paths_detailed.json"
    if not debug_path.exists():
        raise SystemExit(f"Missing scene_objects_debug.json in {run_dir}")
    if not picked_path.exists():
        raise SystemExit(f"Missing picked_paths_detailed.json in {run_dir}")
    debug = json.load(open(debug_path, "r"))
    picked_payload = json.load(open(picked_path, "r"))
    return {"debug": debug, "picked": picked_payload}


def _build_segments(picked_payload: Dict[str, Any]):
    nodes_path = picked_payload.get("nodes")
    if not nodes_path:
        raise SystemExit("picked_paths_detailed.json missing 'nodes' field")
    nodes = load_nodes(nodes_path)
    all_segments = build_segments_from_nodes(nodes)
    seg_by_id = {int(s["seg_id"]): s["points"] for s in all_segments}
    seg_by_id = _override_seg_points_with_picked(picked_payload.get("picked", []), seg_by_id)
    return nodes, all_segments, seg_by_id


def _build_ego_spawns(picked: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for p in picked:
        sig = p.get("signature", {}) if isinstance(p.get("signature"), dict) else {}
        entry = sig.get("entry", {}) if isinstance(sig.get("entry"), dict) else {}
        pt = entry.get("point", {}) if isinstance(entry.get("point"), dict) else {}
        if pt.get("x") is None or pt.get("y") is None:
            continue
        out.append(
            {
                "vehicle": p.get("vehicle"),
                "spawn": {
                    "x": float(pt["x"]),
                    "y": float(pt["y"]),
                    "yaw_deg": float(entry.get("heading_deg", 0.0)),
                },
            }
        )
    return out


def _placement_to_world(
    cand: CandidatePlacement, seg_by_id: Dict[int, np.ndarray]
) -> Dict[str, float]:
    pts = seg_by_id.get(int(cand.seg_id))
    if pts is None:
        return {}
    spawn = compute_spawn_from_anchor(
        pts, float(cand.s_along), str(cand.lateral_relation), None
    )
    return {
        "x": float(spawn.get("x", 0.0)),
        "y": float(spawn.get("y", 0.0)),
        "yaw_deg": float(spawn.get("yaw_deg", 0.0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir",
        required=True,
        help="Path to pipeline attempt dir containing scene_objects_debug.json",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: <run-dir>/scene_objects_csp_rerun_debug.json)",
    )
    ap.add_argument(
        "--max-backtrack",
        type=int,
        default=50000,
        help="Max backtrack nodes for CSP",
    )
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir).resolve()
    out_path = (
        pathlib.Path(args.out).resolve()
        if args.out
        else run_dir / "scene_objects_csp_rerun_debug.json"
    )

    payload = _load_run(run_dir)
    actor_specs = payload["debug"].get("actor_specs_after_validation", [])
    picked_payload = payload["picked"]
    picked = picked_payload.get("picked", [])
    crop_region = picked_payload.get("crop_region")

    nodes, all_segments, seg_by_id = _build_segments(picked_payload)
    merge_min_s_by_vehicle = _compute_merge_min_s_by_vehicle(
        picked_payload, picked, seg_by_id
    )
    ego_spawns = _build_ego_spawns(picked)

    chosen, dbg, new_crop = solve_weighted_csp_with_extension(
        actor_specs,
        picked,
        seg_by_id,
        crop_region,
        all_segments=all_segments,
        merge_min_s_by_vehicle=merge_min_s_by_vehicle,
        ego_spawns=ego_spawns,
        max_backtrack=args.max_backtrack,
    )

    placements = {}
    for sid, cand in chosen.items():
        placements[sid] = {
            "vehicle_num": cand.vehicle_num,
            "segment_index": cand.segment_index,
            "seg_id": cand.seg_id,
            "s_along": cand.s_along,
            "lateral_relation": cand.lateral_relation,
            "path_s_m": cand.path_s_m,
            "base_score": cand.base_score,
            "world_spawn": _placement_to_world(cand, seg_by_id),
        }

    out = {
        "run_dir": str(run_dir),
        "input_files": {
            "scene_objects_debug": str(run_dir / "scene_objects_debug.json"),
            "picked_paths_detailed": str(run_dir / "picked_paths_detailed.json"),
        },
        "actor_specs_count": len(actor_specs),
        "merge_min_s_by_vehicle": merge_min_s_by_vehicle,
        "ego_spawns": ego_spawns,
        "crop_region_in": crop_region,
        "crop_region_out": new_crop,
        "domain_sizes": dbg.get("domain_sizes"),
        "candidate_debug": dbg.get("candidate_debug"),
        "fallback_used": dbg.get("fallback_used"),
        "missing_entities": dbg.get("missing_entities"),
        "nodes_searched": dbg.get("nodes_searched"),
        "best_score": dbg.get("best_score"),
        "placements": placements,
    }

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[INFO] Wrote CSP rerun debug to {out_path}")


if __name__ == "__main__":
    main()
