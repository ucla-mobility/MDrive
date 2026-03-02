#!/usr/bin/env python3
"""
End-to-end scenario pipeline (batch or single):
1) Pick crop regions for scenarios (shared LLM).
2) Generate legal paths per crop.
3) Use one shared HF model to pick paths from the prompt + scenario description.
4) Reuse the same model to place objects for the scene description.

Outputs per scenario:
- legal_paths_prompt.txt           (compact candidate list for path picking)
- legal_paths_detailed.json       (aggregated candidates for machine use)
- path_picker_prompt.txt          (prompt + user description fed to path picker)
- picked_paths_detailed.json      (chosen ego paths)
- scene_objects.json / .png       (final placements)
- routes output (optional, when --routes-out-dir is set)

NOTE:
This is a legacy free-text runner kept for compatibility. The canonical
category-based pipeline entrypoint
is `scenario_generator/start_pipeline.py`.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from convert_scene_to_routes import convert_scene_to_routes
from generate_legal_paths import (
    CropBox,
    build_connectivity,
    build_path_signature,
    build_segments,
    build_segments_detailed_for_path,
    crop_segments,
    generate_legal_paths,
    load_nodes,
    make_path_name,
    save_aggregated_signatures_json,
    save_prompt_file,
    visualize_individual_paths,
    visualize_legal_paths,
)
import run_crop_region_picker as crop_picker
from run_object_placer import run_object_placer
from run_path_picker import pick_paths_with_model
from run_path_refiner import refine_picked_paths_with_model


def build_candidates(
    legal_paths,
    out_json_path: str,
    prompt_path: str,
    nodes_path: str,
    crop: CropBox,
    params,
    lane_counts_by_road: Optional[Dict[int, int]] = None,
):
    candidates = []
    for i, p in enumerate(legal_paths):
        sig = build_path_signature(p)
        name = make_path_name(i, sig)
        sig["segments_detailed"] = build_segments_detailed_for_path(p, polyline_sample_n=10)
        candidates.append({"name": name, "signature": sig})

    # Save prompt + aggregated JSON
    save_prompt_file(
        prompt_path,
        crop=crop,
        nodes_path=nodes_path,
        params=params,
        paths_named=candidates,
    )
    save_aggregated_signatures_json(
        out_json_path,
        crop=crop,
        nodes_path=nodes_path,
        params=params,
        paths_named=candidates,
        lane_counts_by_road=lane_counts_by_road,
    )


def load_shared_model(model_id: str):
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


def _safe_name(text: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return name or "scenario"


def _slice_scenarios(scenarios, start_index: int, end_index: int):
    if start_index < 1:
        raise SystemExit("--start-index must be >= 1")
    if end_index and end_index < start_index:
        raise SystemExit("--end-index must be >= --start-index")
    start = start_index - 1
    end = end_index if end_index else len(scenarios)
    return scenarios[start:end]

def _parse_indices(indices_str: str):
    if not indices_str:
        return []
    parts = re.split(r"[,\s]+", indices_str.strip())
    indices = []
    for part in parts:
        if not part:
            continue
        try:
            idx = int(part)
        except ValueError as exc:
            raise SystemExit("--indices must be a comma/space-separated list of integers") from exc
        if idx < 1:
            raise SystemExit("--indices values must be >= 1")
        indices.append(idx)
    return indices


def _slice_scenarios_by_indices(scenarios, indices):
    if not indices:
        return []
    total = len(scenarios)
    selected = []
    for idx in indices:
        if idx > total:
            raise SystemExit(f"--indices value {idx} exceeds scenario count {total}")
        selected.append(scenarios[idx - 1])
    return selected


def _load_scenarios(args):
    scenarios = crop_picker.load_scenarios(args.scenarios_file, args.scenarios_dir)
    if not scenarios:
        raise SystemExit("No scenarios loaded. Provide --scenarios-file or --scenarios-dir")
    indices = _parse_indices(args.indices)
    if indices:
        scenarios = _slice_scenarios_by_indices(scenarios, indices)
    else:
        scenarios = _slice_scenarios(scenarios, args.start_index, args.end_index)
    if not scenarios:
        raise SystemExit("No scenarios selected after applying index filters")
    return scenarios


def _resolve_model_device(model):
    try:
        return model.device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_crop_region_picker(
    scenarios,
    town_nodes_dir: str,
    model,
    tokenizer,
    args,
    out_dir: Path,
):
    town_files = []
    for fn in sorted(os.listdir(town_nodes_dir)):
        if fn.lower().endswith(".json") and fn.lower().startswith("town"):
            town_files.append(os.path.join(town_nodes_dir, fn))
    if not town_files:
        raise SystemExit(f"No Town*.json found in {town_nodes_dir}")

    radii = [float(x) for x in args.crop_radii.split(",") if x.strip()]

    all_crops = []
    for p in town_files:
        town = os.path.splitext(os.path.basename(p))[0]
        print(f"[INFO] indexing {town} ...")
        crops = crop_picker.build_candidate_crops_for_town(
            town_name=town,
            town_json_path=p,
            radii=radii,
            min_path_len=args.crop_min_path_length,
            max_paths=args.crop_max_paths,
            max_depth=args.crop_max_depth,
        )
        print(f"[INFO]  {town}: {len(crops)} candidate crops")
        all_crops.extend(crops)

    if not all_crops:
        raise SystemExit("No candidate crops built. Try larger radii or smaller --crop-min-path-length.")

    specs = {}
    extractor = crop_picker.LLMGeometryExtractor(model_name=args.model, device=str(_resolve_model_device(model)))
    if args.crop_extractor == "llm":
        extractor._tok = tokenizer
        extractor._mdl = model
        total = len(scenarios)
        for i, sc in enumerate(scenarios, start=1):
            specs[sc.sid] = extractor.extract(sc.text)
            if args.crop_log_every and (i % args.crop_log_every == 0 or i == total):
                print(f"[INFO] extracted {i}/{total}")
    else:
        total = len(scenarios)
        for i, sc in enumerate(scenarios, start=1):
            specs[sc.sid] = extractor._fallback_spec(sc.text, notes="rules_mode")
            if args.crop_log_every and (i % args.crop_log_every == 0 or i == total):
                print(f"[INFO] extracted {i}/{total}")

    viz_out_dir = ""
    if args.crop_viz:
        if crop_picker.plt is None:
            print("[WARN] matplotlib not available; skipping crop visualizations")
        else:
            viz_out_dir = str(out_dir / "crop_viz")

    res = crop_picker.solve_assignment(
        scenarios=scenarios,
        specs=specs,
        crops=all_crops,
        domain_k=args.crop_domain_k,
        capacity_per_crop=args.crop_capacity_per_crop,
        reuse_weight=args.crop_reuse_weight,
        junction_penalty=args.crop_junction_penalty,
        log_every=args.crop_log_every,
        viz_out_dir=viz_out_dir,
        viz_invert_x=args.crop_viz_invert_x,
        viz_dpi=args.crop_viz_dpi,
        viz_max=args.crop_viz_max,
    )

    crop_map_path = out_dir / "crop_map.json"
    crop_map_detailed_path = out_dir / "crop_map_detailed.json"
    with open(crop_map_path, "w", encoding="utf-8") as f:
        json.dump(res.mapping, f, indent=2, sort_keys=True)
    with open(crop_map_detailed_path, "w", encoding="utf-8") as f:
        json.dump(res.detailed, f, indent=2, sort_keys=True)
    print(f"[OK] wrote crop mapping to {crop_map_path}")
    print(f"[OK] wrote crop detailed to {crop_map_detailed_path}")
    if viz_out_dir:
        print(f"[OK] wrote crop visualizations to {viz_out_dir}")
    return res


def _run_single_scenario(
    scenario_id: str,
    description: str,
    town: str,
    crop_vals,
    args,
    model,
    tokenizer,
    out_dir: Path,
    routes_out_dir: Optional[Path] = None,
):
    nodes_path = Path(args.town_nodes_root) / f"{town}.json"
    if not nodes_path.exists():
        raise SystemExit(f"[ERROR] nodes file not found for town '{town}': {nodes_path}")

    crop = CropBox(xmin=crop_vals[0], xmax=crop_vals[1], ymin=crop_vals[2], ymax=crop_vals[3])

    legal_prompt_path = out_dir / "legal_paths_prompt.txt"
    legal_json_path = out_dir / "legal_paths_detailed.json"
    picker_prompt_path = out_dir / "path_picker_prompt.txt"
    picked_paths_path = out_dir / "picked_paths_detailed.json"
    refined_paths_path = out_dir / "picked_paths_refined.json"
    refiner_prompt_path = out_dir / "path_refiner_prompt.json"
    refined_viz_path = out_dir / "picked_paths_refined_viz.png"
    scene_json_path = out_dir / "scene_objects.json"
    scene_png_path = out_dir / "scene_objects.png"

    print(f"[INFO] Scenario {scenario_id}: loading nodes {nodes_path}")
    t_start_scenario = time.time()
    t0 = time.time()
    data = load_nodes(str(nodes_path))
    print(f"[TIMING] load_nodes: {time.time() - t0:.2f}s")

    print("[INFO] Building segments and connectivity...")
    t0 = time.time()
    all_segments = build_segments(data)
    cropped_segments = crop_segments(all_segments, crop)
    if not cropped_segments:
        raise SystemExit("[ERROR] No segments found in crop region.")
    lane_ids_by_road: Dict[int, set] = {}
    for s in cropped_segments:
        try:
            rid = int(s.road_id)
            lid = int(s.lane_id)
        except Exception:
            continue
        lane_ids_by_road.setdefault(rid, set()).add(lid)
    lane_counts_by_road = {rid: len(lanes) for rid, lanes in lane_ids_by_road.items()}
    adj = build_connectivity(
        cropped_segments,
        connect_radius_m=6.0,
        connect_yaw_tol_deg=60.0,
    )
    print(f"[TIMING] build_segments+crop+connectivity: {time.time() - t0:.2f}s")

    print("[INFO] Generating legal paths...")
    t0 = time.time()
    legal_paths = generate_legal_paths(
        cropped_segments,
        adj,
        crop,
        min_path_length=20.0,
        max_paths=100,
        max_depth=5,
        allow_within_region_fallback=False,
    )
    print(f"[TIMING] generate_legal_paths: {time.time() - t0:.2f}s")

    if not legal_paths:
        raise SystemExit(
            "[ERROR] No legal paths found for this crop. "
            "This usually means the crop does not have boundary-crossing lanes. "
            "Pick a different crop (or regenerate crop_map with the updated crop picker)."
        )

    print(f"[INFO] Legal paths found: {len(legal_paths)}")

    params = {
        "max_yaw_diff_deg": 60.0,
        "connect_radius_m": 6.0,
        "min_path_length_m": 20.0,
        "max_paths": 100,
        "max_depth": 5,
        "turn_frame": "WORLD_FRAME",
    }
    t0 = time.time()
    build_candidates(
        legal_paths=legal_paths,
        out_json_path=str(legal_json_path),
        prompt_path=str(legal_prompt_path),
        nodes_path=str(nodes_path),
        crop=crop,
        params=params,
        lane_counts_by_road=lane_counts_by_road,
    )
    print(f"[TIMING] build_candidates: {time.time() - t0:.2f}s")

    if args.viz_legal_all and legal_paths:
        legal_all_path = out_dir / "legal_paths_all.png"
        visualize_legal_paths(cropped_segments, legal_paths, crop, str(legal_all_path))
        print(f"[INFO] Legal paths viz:    {legal_all_path}")

    prompt_text = Path(legal_prompt_path).read_text(encoding="utf-8").strip()
    prompt_text += (
        "\n\nUSER SCENARIO DESCRIPTION:\n"
        + description
        + "\n(Only assign paths to moving vehicles; ignore static/parked props.)\n"
    )
    picker_prompt_path.write_text(prompt_text, encoding="utf-8")

    print("[INFO] Running path picker with shared model...")
    t0 = time.time()
    pick_paths_with_model(
        prompt=prompt_text,
        aggregated_json=str(legal_json_path),
        out_picked_json=str(picked_paths_path),
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.picker_max_new_tokens,
        do_sample=args.picker_do_sample,
        temperature=args.picker_temperature,
        top_p=args.picker_top_p,
        allow_fuzzy_match=args.allow_fuzzy_match,
    )
    print(f"[TIMING] path_picker total: {time.time() - t0:.2f}s")

    picked_paths_for_placer = picked_paths_path
    if args.refine_paths:
        print('[INFO] Running path refiner (post-pick)...')
        t0 = time.time()
        refine_picked_paths_with_model(
            picked_paths_json=str(picked_paths_path),
            description=description,
            out_json=str(refined_paths_path),
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.refiner_max_new_tokens,
            viz=args.viz_refined_paths,
            viz_out=str(refined_viz_path),
            viz_show=args.viz_show,
            prompt_out=str(refiner_prompt_path),
            nodes_root=args.town_nodes_root,
            carla_assets=args.carla_assets,
        )
        print(f"[TIMING] path_refiner total: {time.time() - t0:.2f}s")
        picked_paths_for_placer = refined_paths_path

    print("[INFO] Running object placer with shared model...")
    t0 = time.time()
    placer_args = SimpleNamespace(
        model=args.model,
        picked_paths=str(picked_paths_for_placer),
        carla_assets=args.carla_assets,
        description=description,
        out=str(scene_json_path),
        viz_out=str(scene_png_path),
        viz=args.viz_objects,
        viz_show=args.viz_show,
        nodes_root=args.town_nodes_root,
        max_new_tokens=args.placer_max_new_tokens,
        do_sample=args.placer_do_sample,
        temperature=args.placer_temperature,
        top_p=args.placer_top_p,
        placement_mode="csp",  # Use CSP mode with group expansion support
    )
    run_object_placer(placer_args, model=model, tokenizer=tokenizer)
    print(f"[TIMING] object_placer total: {time.time() - t0:.2f}s")

    if routes_out_dir:
        align_routes = not getattr(args, 'no_align_routes', False)
        print(f"[INFO] Converting scene to routes (align={align_routes})...")
        t0 = time.time()
        convert_scene_to_routes(
            str(scene_json_path),
            str(routes_out_dir),
            ego_num=args.routes_ego_num,
            align_routes=align_routes,
            carla_host=args.carla_host,
            carla_port=args.carla_port,
        )
        print(f"[TIMING] scene_to_routes total: {time.time() - t0:.2f}s")
        print(f"[OK] Routes output in {routes_out_dir}")

    if args.individual and legal_paths:
        indiv_dir = Path(args.individual_dir)
        if not indiv_dir.is_absolute():
            indiv_dir = out_dir / indiv_dir
        indiv_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Generating individual legal path visualizations...")
        visualize_individual_paths(cropped_segments, legal_paths, crop, str(indiv_dir))
        print(f"  Individual path images: {indiv_dir}")

    print(f"[TIMING] scenario total: {time.time() - t_start_scenario:.2f}s")
    print(f"[INFO] Scenario {scenario_id} outputs in {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="End-to-end scenario pipeline (crop -> picker -> object placer)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-AWQ", help="HF model id or local path")
    ap.add_argument("--scenarios-file", default="", help="Scenario file (.txt/.json/.jsonl)")
    ap.add_argument("--scenarios-dir", default="", help="Directory of scenario files")
    ap.add_argument("--start-index", type=int, default=1, help="1-based start index into loaded scenarios")
    ap.add_argument("--end-index", type=int, default=0, help="1-based end index (0 = end of list)")
    ap.add_argument(
        "--indices",
        default="",
        help="Comma/space-separated 1-based indices to run (overrides --start-index/--end-index)",
    )

    ap.add_argument("--town", default="", help="Town name (single-scenario mode)")
    ap.add_argument("--town-nodes-dir", dest="town_nodes_root", default="town_nodes",
                    help="Root dir containing town node JSONs")
    ap.add_argument("--town-nodes-root", dest="town_nodes_root", help=argparse.SUPPRESS)
    ap.add_argument(
        "--crop",
        type=float,
        nargs=4,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help="Crop region as: xmin xmax ymin ymax (single-scenario mode)",
    )
    ap.add_argument("--description", default="", help="Natural-language scene description (single-scenario mode)")
    ap.add_argument("--carla-assets", default="carla_assets.json", help="carla_assets.json path")
    ap.add_argument("--out-dir", default="log", help="Directory to write pipeline artifacts")

    # Crop region picker controls
    ap.add_argument("--crop-extractor", choices=["llm", "rules"], default="llm")
    ap.add_argument("--crop-radii", default="45,55,65",
                    help="Comma-separated crop half-width radii (meters)")
    ap.add_argument("--crop-min-path-length", type=float, default=22.0)
    ap.add_argument("--crop-max-paths", type=int, default=80)
    ap.add_argument("--crop-max-depth", type=int, default=8)
    ap.add_argument("--crop-domain-k", type=int, default=50)
    ap.add_argument("--crop-capacity-per-crop", type=int, default=10)
    ap.add_argument("--crop-reuse-weight", type=float, default=4000.0)
    ap.add_argument("--crop-junction-penalty", type=float, default=25000.0)
    ap.add_argument("--crop-log-every", type=int, default=1)
    ap.add_argument("--crop-viz", action="store_true", help="Write crop visualizations while assigning")
    ap.add_argument("--crop-viz-invert-x", action="store_true")
    ap.add_argument("--crop-viz-dpi", type=int, default=150)
    ap.add_argument("--crop-viz-max", type=int, default=0)

    # Picker generation controls
    ap.add_argument("--picker-max-new-tokens", type=int, default=2048)
    ap.add_argument("--picker-do-sample", action="store_true", default=True, help="Enable sampling (default: True)")
    ap.add_argument("--picker-no-sample", dest="picker_do_sample", action="store_false", help="Disable sampling")
    ap.add_argument("--picker-temperature", type=float, default=0.2)
    ap.add_argument("--picker-top-p", type=float, default=0.95)
    ap.add_argument("--allow-fuzzy-match", action="store_true", help="Permit fuzzy path name matching")

    # Refiner controls (post-pick spawn/lane-change/timing)
    ap.add_argument('--refine-paths', action='store_true', default=True, help='Run post-picker path refinement (spawn points, lane changes, conflict timing)')
    ap.add_argument('--refiner-max-new-tokens', type=int, default=2048)
    ap.add_argument('--viz-refined-paths', action='store_true', help='Write refined picked-paths visualization PNG')

    # Object placer generation controls
    ap.add_argument("--placer-max-new-tokens", type=int, default=2048)
    ap.add_argument("--placer-do-sample", action="store_true", default=True, help="Enable sampling (default: True)")
    ap.add_argument("--placer-no-sample", dest="placer_do_sample", action="store_false", help="Disable sampling")
    ap.add_argument("--placer-temperature", type=float, default=0.2)
    ap.add_argument("--placer-top-p", type=float, default=0.95)

    ap.add_argument("--viz-objects", action="store_true", help="Enable visualization in object placer")
    ap.add_argument("--viz-show", action="store_true", help="Show matplotlib window (if available)")
    ap.add_argument("--individual", action="store_true", help="Generate separate visualization for each legal path")
    ap.add_argument(
        "--individual-dir",
        type=str,
        default="legal_paths_individual",
        help="Directory for individual path visualizations (relative to --out-dir unless absolute)",
    )
    ap.add_argument("--viz-legal-all", action="store_true", help="Save a single image of all legal paths in the crop")
    ap.add_argument(
        "--routes-out-dir",
        default="routes",
        help=(
            "Convert scene_objects.json to routes after placement (default: 'routes'). "
            "Use --no-routes to disable route generation. "
            "Relative paths are resolved from the current working directory; "
            "batch mode writes per-scenario subfolders."
        ),
    )
    ap.add_argument(
        "--no-routes",
        action="store_true",
        help="Disable route file generation (overrides --routes-out-dir).",
    )
    ap.add_argument(
        "--routes-ego-num",
        type=int,
        default=None,
        help="Override ego vehicle count for route conversion (default: auto-detect).",
    )
    ap.add_argument(
        "--carla-port",
        type=int,
        default=2000,
        help="CARLA server port for route alignment (default: 2000).",
    )
    ap.add_argument(
        "--carla-host",
        type=str,
        default="127.0.0.1",
        help="CARLA server host for route alignment (default: 127.0.0.1).",
    )
    ap.add_argument(
        "--no-align-routes",
        action="store_true",
        help="Skip route alignment with CARLA (faster, but routes may not match road network exactly).",
    )

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine routes output directory (None if disabled via --no-routes)
    if getattr(args, 'no_routes', False) or not args.routes_out_dir:
        routes_root = None
    else:
        routes_root = Path(args.routes_out_dir)

    batch_mode = bool(args.scenarios_file or args.scenarios_dir)
    if batch_mode:
        scenarios = _load_scenarios(args)
        print(f"[INFO] Loaded {len(scenarios)} scenarios")

        print("[INFO] Loading shared model once...")
        model, tokenizer = load_shared_model(args.model)

        res = _run_crop_region_picker(
            scenarios=scenarios,
            town_nodes_dir=args.town_nodes_root,
            model=model,
            tokenizer=tokenizer,
            args=args,
            out_dir=out_dir,
        )

        assignments = res.detailed.get("assignments", {})
        total = len(scenarios)
        processed = 0
        for idx, sc in enumerate(scenarios, start=1):
            info = assignments.get(sc.sid)
            if not info:
                print(f"[WARN] scenario {sc.sid} unassigned; skipping")
                continue
            scenario_dir_name = _safe_name(sc.sid)
            scenario_dir = out_dir / scenario_dir_name
            scenario_dir.mkdir(parents=True, exist_ok=True)
            routes_out_dir = routes_root / scenario_dir_name if routes_root else None

            crop_vals = info["crop"]
            _run_single_scenario(
                scenario_id=sc.sid,
                description=sc.text,
                town=info["town"],
                crop_vals=crop_vals,
                args=args,
                model=model,
                tokenizer=tokenizer,
                out_dir=scenario_dir,
                routes_out_dir=routes_out_dir,
            )
            processed += 1
            print(f"[INFO] Completed {processed}/{total}: {sc.sid}")

        print("[INFO] Pipeline complete.")
    else:
        if not (args.description and args.town and args.crop):
            raise SystemExit(
                "Provide --scenarios-file/--scenarios-dir for batch mode, "
                "or --description/--town/--crop for single-scenario mode."
            )

        print("[INFO] Loading shared model once...")
        model, tokenizer = load_shared_model(args.model)

        # Create a scenario-specific subfolder based on description
        scenario_id = _safe_name(args.description)[:32]
        scenario_dir = out_dir / scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        routes_out_dir = routes_root / scenario_id if routes_root else None
        if routes_out_dir:
            routes_out_dir.mkdir(parents=True, exist_ok=True)

        _run_single_scenario(
            scenario_id=scenario_id,
            description=args.description,
            town=args.town,
            crop_vals=args.crop,
            args=args,
            model=model,
            tokenizer=tokenizer,
            out_dir=scenario_dir,
            routes_out_dir=routes_out_dir,
        )

        print("[INFO] Pipeline complete.")


if __name__ == "__main__":
    main()
