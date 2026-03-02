import argparse
import json
import os
from typing import Dict, List, Tuple

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except Exception:
    plt = None
    mpatches = None

from .candidates import build_candidate_crops_for_town
from .csp import solve_assignment
from .llm_extractor import LLMGeometryExtractor
from .models import CropFeatures, CropKey, GeometrySpec
from .scenario_io import load_scenarios
from .viz import save_viz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--town-nodes-dir", required=True, help="Directory containing Town*.json")
    ap.add_argument("--scenarios-file", default="", help="One scenario per line")
    ap.add_argument("--scenarios-dir", default="", help="Directory of scenario files (.txt/.json/.jsonl)")

    ap.add_argument("--out", required=True, help="Output mapping JSON")
    ap.add_argument("--out-detailed", default="", help="Output detailed JSON")

    ap.add_argument("--extractor", choices=["llm", "rules"], default="llm",
                    help="Use LLM-centered extraction (default) or a conservative fallback")
    ap.add_argument("--llm-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--radii", default="45,55,65",
                    help="Comma-separated crop half-width radii (meters). Include at least one >=55 for run-up.")
    ap.add_argument("--min-path-length", type=float, default=22.0)
    ap.add_argument("--max-paths", type=int, default=80)
    ap.add_argument("--max-depth", type=int, default=8)

    ap.add_argument("--domain-k", type=int, default=50)
    ap.add_argument("--capacity-per-crop", type=int, default=10)
    ap.add_argument("--reuse-weight", type=float, default=4000.0)
    ap.add_argument("--junction-penalty", type=float, default=25000.0)
    ap.add_argument("--log-every", type=int, default=25,
                    help="Print progress every N scenarios (0 = off). With --viz-out-dir, writes viz in batches.")

    ap.add_argument("--viz-out-dir", default="")
    ap.add_argument("--viz-invert-x", action="store_true")
    ap.add_argument("--viz-dpi", type=int, default=150)
    ap.add_argument("--viz-max", type=int, default=0)

    args = ap.parse_args()

    scenarios = load_scenarios(args.scenarios_file, args.scenarios_dir)
    if not scenarios:
        raise SystemExit("No scenarios loaded. Provide --scenarios-file or --scenarios-dir")

    town_files = []
    for fn in sorted(os.listdir(args.town_nodes_dir)):
        if fn.lower().endswith(".json") and fn.lower().startswith("town"):
            town_files.append(os.path.join(args.town_nodes_dir, fn))
    if not town_files:
        raise SystemExit(f"No Town*.json found in {args.town_nodes_dir}")

    radii = [float(x) for x in args.radii.split(",") if x.strip()]

    all_crops: List[CropFeatures] = []
    for p in town_files:
        town = os.path.splitext(os.path.basename(p))[0]
        print(f"[INFO] indexing {town} ...")
        crops = build_candidate_crops_for_town(
            town_name=town,
            town_json_path=p,
            radii=radii,
            min_path_len=args.min_path_length,
            max_paths=args.max_paths,
            max_depth=args.max_depth,
        )
        print(f"[INFO]  {town}: {len(crops)} candidate crops")
        all_crops.extend(crops)

    if not all_crops:
        raise SystemExit("No candidate crops built. Try larger radii or smaller --min-path-length.")

    specs: Dict[str, GeometrySpec] = {}
    extractor = LLMGeometryExtractor(model_name=args.llm_model, device=args.device)
    total = len(scenarios)
    viz_streaming = False
    viz_out_dir = args.viz_out_dir
    if viz_out_dir:
        if plt is None or mpatches is None:
            print("[WARN] matplotlib not available; skipping visualization")
            viz_out_dir = ""
        elif args.log_every and args.log_every > 0:
            viz_streaming = True

    if args.extractor == "llm":
        for i, sc in enumerate(scenarios, start=1):
            specs[sc.sid] = extractor.extract(sc.text)
            if args.log_every and (i % args.log_every == 0 or i == total):
                print(f"[INFO] extracted {i}/{total}")
    else:
        for i, sc in enumerate(scenarios, start=1):
            specs[sc.sid] = extractor._fallback_spec(sc.text, notes="rules_mode")
            if args.log_every and (i % args.log_every == 0 or i == total):
                print(f"[INFO] extracted {i}/{total}")

    res = solve_assignment(
        scenarios=scenarios,
        specs=specs,
        crops=all_crops,
        domain_k=args.domain_k,
        capacity_per_crop=args.capacity_per_crop,
        reuse_weight=args.reuse_weight,
        junction_penalty=args.junction_penalty,
        log_every=args.log_every,
        viz_out_dir=viz_out_dir if viz_streaming else "",
        viz_invert_x=args.viz_invert_x,
        viz_dpi=args.viz_dpi,
        viz_max=args.viz_max,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res.mapping, f, indent=2, sort_keys=True)
    print(f"[OK] wrote mapping to {args.out}")

    if args.out_detailed:
        with open(args.out_detailed, "w", encoding="utf-8") as f:
            json.dump(res.detailed, f, indent=2, sort_keys=True)
        print(f"[OK] wrote detailed to {args.out_detailed}")

    if args.viz_out_dir and not viz_streaming and viz_out_dir:
        items = list(res.detailed.get("assignments", {}).items())
        if args.viz_max and args.viz_max > 0:
            items = items[: args.viz_max]

        scen_map = {s.sid: s.text for s in scenarios}

        crop_lookup: Dict[Tuple[str, str], CropFeatures] = {}
        for c in all_crops:
            crop_lookup[(c.town, c.crop.to_str())] = c

        for sid, info in items:
            town = info["town"]
            crop_vals = info["crop"]
            ck = CropKey(float(crop_vals[0]), float(crop_vals[1]), float(crop_vals[2]), float(crop_vals[3]))
            crop_str = ck.to_str()
            feat = crop_lookup.get((town, crop_str))
            if feat is None:
                continue
            out_png = os.path.join(args.viz_out_dir, f"{sid}__{town}__{ck.to_str()}.png")
            try:
                save_viz(
                    out_png=out_png,
                    scenario_id=sid,
                    scenario_text=scen_map.get(sid, info.get("scenario", "")),
                    crop=ck,
                    crop_feat=feat,
                    invert_x=args.viz_invert_x,
                    dpi=args.viz_dpi,
                )
            except Exception as e:
                print(f"[WARN] viz failed for {sid}: {e}")
        print(f"[OK] wrote visualizations to {args.viz_out_dir}")
    elif viz_streaming:
        print(f"[OK] wrote visualizations to {args.viz_out_dir}")

    n_assigned = len(res.detailed.get("assignments", {}))
    n_total = len(scenarios)
    print(f"[SUMMARY] assigned {n_assigned}/{n_total} scenarios")
    if res.detailed.get("unassigned"):
        print(f"[WARN] {len(res.detailed['unassigned'])} scenarios unassigned (see --out-detailed)")


if __name__ == "__main__":
    main()
