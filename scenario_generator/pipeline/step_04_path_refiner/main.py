import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:  # pragma: no cover
    AutoTokenizer = None
    AutoModelForCausalLM = None

from .csp import refine_spawn_and_speeds_soft_csp
from .geometry import CropBox, _segments_to_polyline_with_map, _slice_segments_detailed, find_conflict_between_polylines, split_corridor_path_segments
from .lane_change import apply_merge_into_lane_of
from .llm import extract_refinement_constraints
from .viz import visualize_refinement

# Optional: reuse asset bbox parsing from object placer
try:
    from ..step_05_object_placer.assets import load_assets, get_asset_bbox
except Exception:
    load_assets = None
    get_asset_bbox = None

# For loading road network nodes
try:
    from run_object_placer import (
        load_nodes,
        build_segments_from_nodes,
        resolve_nodes_path,
    )
except Exception:
    load_nodes = None
    build_segments_from_nodes = None
    resolve_nodes_path = None


def _resolve_carla_assets_path(picked_paths_json: str, carla_assets: Optional[str]) -> Optional[str]:
    if carla_assets:
        return carla_assets
    try:
        candidate = Path(picked_paths_json).resolve().parent / "carla_assets.json"
        if candidate.exists():
            return str(candidate)
    except Exception:
        pass
    try:
        candidate = Path(__file__).resolve().parents[2] / "carla_assets.json"
        if candidate.exists():
            return str(candidate)
    except Exception:
        pass
    return None


def _compute_min_spawn_sep_from_assets(carla_assets_path: Optional[str], vehicle_id: str) -> Optional[float]:
    if not carla_assets_path or not load_assets or not get_asset_bbox:
        return None
    try:
        load_assets(carla_assets_path)
        bbox = get_asset_bbox(vehicle_id)
    except Exception as e:
        print(f"[WARN] Failed to load carla assets for spawn separation: {e}", flush=True)
        return None
    if bbox is None:
        print(f"[WARN] No bbox found for {vehicle_id}; using default spawn separation", flush=True)
        return None
    sep_m = max(float(bbox.length), float(bbox.width))
    return sep_m if sep_m > 0.0 else None


def refine_picked_paths_with_model(
    picked_paths_json: str,
    description: str,
    out_json: str,
    model=None,
    tokenizer=None,
    max_new_tokens: int = 512,
    viz: bool = False,
    viz_out: str = "picked_paths_refined_viz.png",
    viz_show: bool = False,
    prompt_out: Optional[str] = None,
    nodes_root: Optional[str] = None,
    carla_assets: Optional[str] = None,
    schema_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    t_refiner_start = time.time()
    t0 = time.time()
    with open(picked_paths_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    crop_region = data.get("crop_region") or {}
    crop = CropBox(
        xmin=float(crop_region.get("xmin", -1e9)),
        xmax=float(crop_region.get("xmax", 1e9)),
        ymin=float(crop_region.get("ymin", -1e9)),
        ymax=float(crop_region.get("ymax", 1e9)),
    )

    # Load nodes for road network visualization (if helpers available)
    seg_by_id: Optional[Dict[int, Any]] = None
    nodes_field = data.get("nodes")
    if viz and nodes_field and load_nodes and build_segments_from_nodes and resolve_nodes_path:
        try:
            resolved = resolve_nodes_path(picked_paths_json, str(nodes_field), nodes_root)
            if os.path.exists(resolved):
                nodes_data = load_nodes(resolved)
                all_segs = build_segments_from_nodes(nodes_data)
                seg_by_id = {int(s["seg_id"]): s["points"] for s in all_segs}
        except Exception as e:
            print(f"[WARNING] Could not load nodes for viz: {e}")

    picked = data.get("picked", [])
    if not isinstance(picked, list) or not picked:
        raise SystemExit("[ERROR] picked_paths JSON missing 'picked' list.")

    vehicles = [p.get("vehicle") for p in picked if isinstance(p, dict) and isinstance(p.get("vehicle"), str)]
    vehicles = [v for v in vehicles if v]
    if not vehicles:
        raise SystemExit("[ERROR] No vehicles found in picked paths.")
    print(f"[TIMING] refiner setup (load data, crop): {time.time() - t0:.2f}s", flush=True)

    t0 = time.time()
    constraints = extract_refinement_constraints(
        description=description,
        vehicles=vehicles,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        schema_payload=schema_payload,
    )
    print(f"[TIMING] refiner LLM constraint extraction: {time.time() - t0:.2f}s", flush=True)

    # If many vehicles share a corridor, allow starts slightly outside the crop
    # to improve feasible spawn separation without changing semantics.
    try:
        if len(vehicles) >= 4:
            opts = constraints.setdefault("options", {})
            current = float(opts.get("start_outside_crop_margin_m", 0.0) or 0.0)
            opts["start_outside_crop_margin_m"] = max(current, 20.0)
    except Exception as e:
        print(f"[WARN] refiner could not set start_outside_crop_margin_m: {e}", flush=True)

    # For t-junctions with perpendicular relations, allow spawns a bit outside the crop
    # so the side-road vehicle can start before merging onto the main road.
    try:
        topo = str((schema_payload or {}).get("topology", "")).lower()
        vehicle_constraints = (schema_payload or {}).get("vehicle_constraints", [])
        has_perp = any(str(c.get("type", "")).lower() in ("perpendicular_left_of", "perpendicular_right_of") for c in vehicle_constraints)
        if topo == "t_junction" and has_perp:
            opts = constraints.setdefault("options", {})
            current = float(opts.get("start_outside_crop_margin_m", 0.0) or 0.0)
            # 12m covers the clipped side-road entry in Town02 without impacting other crops
            opts["start_outside_crop_margin_m"] = max(current, 12.0)
            # If any ego explicitly enters from the side road, widen further to pull spawn points outside the crop.
            ego_entries = {
                str(v.get("vehicle_id")): str(v.get("entry_road", "")).lower()
                for v in (schema_payload or {}).get("ego_vehicles", [])
                if isinstance(v, dict) and v.get("vehicle_id")
            }
            if any(val == "side" for val in ego_entries.values()):
                opts["start_outside_crop_margin_m"] = max(opts["start_outside_crop_margin_m"], 50.0)
            # Remove lane-change macros for vehicles on the side road so their perpendicular approach is preserved.
            side_road_vehicles = {
                str(v.get("vehicle_id"))
                for v in (schema_payload or {}).get("ego_vehicles", [])
                if str(v.get("entry_road", "")).lower() == "side"
            }
            if side_road_vehicles:
                before = len(constraints.get("lane_changes", []) or [])
                constraints["lane_changes"] = [
                    lc
                    for lc in (constraints.get("lane_changes", []) or [])
                    if lc.get("vehicle") not in side_road_vehicles and lc.get("target") not in side_road_vehicles
                ]
                after = len(constraints.get("lane_changes", []) or [])
                if before != after:
                    print(
                        f"[INFO] refiner: dropped {before - after} lane_change macros for side-road vehicles (t-junction perpendicular)",
                        flush=True,
                    )
    except Exception as e:
        print(f"[WARN] refiner could not set t-junction spawn margin: {e}", flush=True)

    if prompt_out:
        try:
            Path(prompt_out).write_text(
                json.dumps({"description": description, "vehicles": vehicles, "constraints": constraints}, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    # Apply lane-change macros first (best effort)
    t0 = time.time()
    picked2 = [dict(p) for p in picked]
    lc_debug = []
    synthetic_id = 900000
    v_to_idx = {p.get("vehicle"): i for i, p in enumerate(picked2)}
    for lc in constraints.get("lane_changes", []):
        v = lc["vehicle"]
        tgt = lc["target"]
        if v not in v_to_idx or tgt not in v_to_idx:
            continue
        mover = picked2[v_to_idx[v]]
        target = picked2[v_to_idx[tgt]]
        updated, dbg = apply_merge_into_lane_of(
            mover=mover,
            target=target,
            crop=crop,
            style=lc.get("style", "polite"),
            timing=lc.get("timing", "near_conflict"),
            synthetic_seg_id_base=synthetic_id,
        )
        synthetic_id += 10
        picked2[v_to_idx[v]] = updated
        lc_debug.append({"lane_change": lc, "debug": dbg})
    print(f"[TIMING] refiner lane-change macros: {time.time() - t0:.2f}s", flush=True)

    # Build per_vehicle segments for solver
    t0 = time.time()
    per_vehicle = {}
    for p in picked2:
        v = p.get("vehicle")
        sig = (p.get("signature") or {}) if isinstance(p.get("signature"), dict) else {}
        segs = sig.get("segments_detailed", []) if isinstance(sig.get("segments_detailed", []), list) else []
        per_vehicle[v] = {"segments_detailed": segs}

    assets_path = _resolve_carla_assets_path(picked_paths_json, carla_assets)
    min_spawn_sep_m = _compute_min_spawn_sep_from_assets(assets_path, "vehicle.lincoln.mkz2017")
    if min_spawn_sep_m is not None:
        print(f"[INFO] refiner spawn separation set to {min_spawn_sep_m:.2f} m", flush=True)

    solution = refine_spawn_and_speeds_soft_csp(
        per_vehicle,
        crop,
        constraints,
        min_spawn_sep_m=min_spawn_sep_m,
    )
    print(f"[TIMING] refiner CSP solve: {time.time() - t0:.2f}s", flush=True)

    # Apply start/end slicing to segments_detailed
    t0 = time.time()
    refined_picked = []
    for p in picked2:
        v = p.get("vehicle")
        sig = (p.get("signature") or {}) if isinstance(p.get("signature"), dict) else {}
        segs = sig.get("segments_detailed", []) if isinstance(sig.get("segments_detailed", []), list) else []
        pts, _ = _segments_to_polyline_with_map(segs)
        if not pts:
            refined_picked.append(p)
            continue

        s_idx = int(solution["start_idx"].get(v, 0))
        e_idx = int(solution["end_idx"].get(v, len(pts) - 1))
        new_segs = _slice_segments_detailed(segs, s_idx, e_idx)

        sig2 = dict(sig)
        sig2["segments_detailed"] = new_segs
        sig2["segment_ids"] = [int(s.get("seg_id", 0)) for s in new_segs]
        sig2["num_segments"] = int(len(new_segs))
        sig2["length_m"] = float(sum(float(s.get("length_m", 0.0)) for s in new_segs))
        if new_segs:
            sig2["entry"] = dict(sig2.get("entry", {}))
            sig2["entry"]["point"] = {"x": float(new_segs[0]["start"]["point"]["x"]), "y": float(new_segs[0]["start"]["point"]["y"])}
            sig2["exit"] = dict(sig2.get("exit", {}))
            sig2["exit"]["point"] = {"x": float(new_segs[-1]["end"]["point"]["x"]), "y": float(new_segs[-1]["end"]["point"]["y"])}

        p2 = dict(p)
        if "signature_original" not in p2:
            p2["signature_original"] = p.get("signature")
        p2["signature"] = sig2
        p2["refined"] = {
            "start_idx_global": s_idx,
            "end_idx_global": e_idx,
            "speed_mps": float(solution["speed_mps"].get(v, 8.0)),
        }
        refined_picked.append(p2)
    print(f"[TIMING] refiner apply slicing: {time.time() - t0:.2f}s", flush=True)

    # Apply corridor segment splitting for TWO_LANE_CORRIDOR topologies
    # This splits long single-segment paths into multiple virtual sub-segments
    # so that the object placer has multiple segment_index values to work with.
    t0 = time.time()
    topo = str((schema_payload or {}).get("topology", "")).lower()
    if topo in ("two_lane_corridor", "corridor"):
        print(f"[INFO] Applying corridor segment splitting for topology: {topo}")
        split_picked = []
        for p in refined_picked:
            split_p = split_corridor_path_segments(p, target_length_m=25.0, min_splits=3)
            split_picked.append(split_p)
            if split_p.get("corridor_split_applied"):
                n_segs = len((split_p.get("signature") or {}).get("segments_detailed", []))
                print(f"  {p.get('vehicle')}: split into {n_segs} sub-segments")
        refined_picked = split_picked
        print(f"[TIMING] refiner corridor splitting: {time.time() - t0:.2f}s", flush=True)

    out_payload = dict(data)
    out_payload["source_picked_paths"] = picked_paths_json
    out_payload["picked"] = refined_picked
    out_payload["refinement"] = {
        "constraints": constraints,
        "lane_change_debug": lc_debug,
        "solution": solution,
        "notes": {
            "semantics": "All LLM constraints are treated as soft preferences; defaults used if infeasible.",
            "kinematics": "Constant speed on polyline; time = distance/speed.",
            "conflict_detection": "Closest approach between polylines with a distance threshold.",
        },
    }

    t0 = time.time()
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"[INFO] Wrote refined picked paths to: {out_json}")
    print(f"[TIMING] refiner total (internal): {time.time() - t_refiner_start:.2f}s", flush=True)

    if viz:
        # For visualization, always show geometric conflict points (even if sync is disabled).
        # Also compute per-vehicle arrival times at each conflict using the chosen speeds.
        speed_mps = solution.get("speed_mps", {}) or {}
        conflicts_geom: List[Dict[str, Any]] = []
        conflict_times: List[Dict[str, Any]] = []

        # Build (vehicle -> polyline) from the final refined segments
        v_to_pts: Dict[str, List[Tuple[float, float]]] = {}
        for pe in refined_picked:
            v = pe.get("vehicle")
            sig = (pe.get("signature") or {}) if isinstance(pe.get("signature"), dict) else {}
            segs = sig.get("segments_detailed", []) if isinstance(sig.get("segments_detailed", []), list) else []
            pts, _ = _segments_to_polyline_with_map(segs)
            if isinstance(v, str) and pts:
                v_to_pts[v] = pts
        vehicles_for_viz = sorted(v_to_pts.keys())
        for i in range(len(vehicles_for_viz)):
            for j in range(i + 1, len(vehicles_for_viz)):
                va, vb = vehicles_for_viz[i], vehicles_for_viz[j]
                conf = find_conflict_between_polylines(v_to_pts[va], v_to_pts[vb], dist_thresh_m=3.0)
                if conf is None:
                    continue
                conflicts_geom.append({"a": va, "b": vb, "point": conf.get("point", {})})
                sa = float((conf.get("s_along") or {}).get("p1_m", 0.0))
                sb = float((conf.get("s_along") or {}).get("p2_m", 0.0))
                ta = sa / max(0.5, float(speed_mps.get(va, 8.0)))
                tb = sb / max(0.5, float(speed_mps.get(vb, 8.0)))
                conflict_times.append({"a": va, "b": vb, "point": conf.get("point", {}), "t": {va: ta, vb: tb}})

        visualize_refinement(
            viz_out,
            crop,
            refined_picked,
            conflicts=conflicts_geom,
            lane_change_debug=lc_debug,
            conflict_times=conflict_times,
            seg_by_id=seg_by_id,
            show=viz_show,
        )
        print(f"[INFO] Wrote refinement viz to: {viz_out}")

    return out_payload


def load_model(model_id: str):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--picked-paths", required=True, help="picked_paths_detailed.json")
    ap.add_argument("--description", required=True, help="Scene description")
    ap.add_argument("--out", default="picked_paths_refined.json", help="Output refined picked paths JSON")
    ap.add_argument("--prompt-out", default="", help="Write debug prompt/constraints JSON here (optional)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--viz", action="store_true", help="Write visualization PNG")
    ap.add_argument("--viz-out", default="picked_paths_refined_viz.png")
    ap.add_argument("--viz-show", action="store_true")
    ap.add_argument("--nodes-root", default=None, help="Directory to search for nodes JSON files")
    ap.add_argument("--carla-assets", default="", help="carla_assets.json path (optional)")

    args = ap.parse_args()

    model, tokenizer = load_model(args.model)
    refine_picked_paths_with_model(
        picked_paths_json=args.picked_paths,
        description=args.description,
        out_json=args.out,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        viz=args.viz,
        viz_out=args.viz_out,
        viz_show=args.viz_show,
        prompt_out=(args.prompt_out if args.prompt_out else None),
        nodes_root=args.nodes_root,
        carla_assets=(args.carla_assets if args.carla_assets else None),
    )


if __name__ == "__main__":
    main()
