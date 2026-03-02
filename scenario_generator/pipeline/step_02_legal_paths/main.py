import argparse
import os
from typing import Dict

import numpy as np

from .connectivity import build_connectivity, generate_legal_paths
from .models import CropBox
from .prompt import save_aggregated_signatures_json, save_prompt_file
from .segments import build_segments, crop_segments, load_nodes
from .signatures import build_path_signature, build_segments_detailed_for_path, make_path_name
from .viz import visualize_individual_paths, visualize_legal_paths


def main():
    parser = argparse.ArgumentParser(
        description="Visualize all legal path segments in a cropped region"
    )
    parser.add_argument("--nodes", type=str, required=True, help="Path to town nodes JSON file")
    parser.add_argument(
        "--crop",
        type=float,
        nargs=4,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        required=True,
        help="Crop region as: xmin xmax ymin ymax",
    )
    parser.add_argument(
        "--max-yaw-diff",
        type=float,
        default=60.0,
        help="Maximum yaw difference for connecting segments (degrees)",
    )
    parser.add_argument(
        "--connect-radius",
        type=float,
        default=6.0,
        help="Maximum distance to connect segment endpoints (meters)",
    )
    parser.add_argument(
        "--min-path-length",
        type=float,
        default=20.0,
        help="Minimum total path length to display (meters)",
    )
    parser.add_argument("--max-paths", type=int, default=100, help="Maximum number of paths to generate")
    parser.add_argument("--max-depth", type=int, default=5, help="Maximum number of segments per path")

    parser.add_argument("--viz", action="store_true", help="Display the visualization")
    parser.add_argument("--out", type=str, default="legal_paths_viz.png", help="Output image file path")

    # NEW:
    parser.add_argument(
        "--out-prompt",
        type=str,
        default="legal_paths_prompt.txt",
        help="Prompt-ready combined file containing all candidate path signatures",
    )
    parser.add_argument(
        "--out-json-detailed",
        type=str,
        default=None,
        help="Optional: also save a pure JSON file containing all candidate path signatures",
    )

    parser.add_argument("--individual", action="store_true", help="Generate separate visualization for each path")
    parser.add_argument(
        "--individual-dir",
        type=str,
        default="legal_paths_individual",
        help="Directory for individual path visualizations (default: legal_paths_individual)",
    )

    args = parser.parse_args()

    crop = CropBox(xmin=args.crop[0], xmax=args.crop[1], ymin=args.crop[2], ymax=args.crop[3])

    print(f"[INFO] Loading nodes from: {args.nodes}")
    data = load_nodes(args.nodes)

    print("[INFO] Building lane segments...")
    all_segments = build_segments(data)
    print(f"[INFO] Total segments in map: {len(all_segments)}")

    print(f"[INFO] Cropping to region: [{crop.xmin}, {crop.xmax}] x [{crop.ymin}, {crop.ymax}]")
    cropped_segments = crop_segments(all_segments, crop)
    print(f"[INFO] Segments in crop region: {len(cropped_segments)}")

    if len(cropped_segments) == 0:
        print("[ERROR] No segments found in crop region!")
        return

    print("[INFO] Building connectivity graph...")
    adj = build_connectivity(
        cropped_segments,
        connect_radius_m=args.connect_radius,
        connect_yaw_tol_deg=args.max_yaw_diff,
    )
    total_connections = sum(len(neighbors) for neighbors in adj)
    print(f"[INFO] Total connections: {total_connections}")

    print("[INFO] Generating legal paths (from outside crop to outside)...")
    legal_paths = generate_legal_paths(
        cropped_segments,
        adj,
        crop,
        min_path_length=args.min_path_length,
        max_paths=args.max_paths,
        max_depth=args.max_depth,
    )

    print(f"[INFO] Found {len(legal_paths)} legal paths")

    if len(legal_paths) == 0:
        print("[WARNING] No legal paths found! Try adjusting parameters:")
        print("  - Increase --max-yaw-diff")
        print("  - Increase --connect-radius")
        print("  - Decrease --min-path-length")
        print("  - Increase --max-paths or --max-depth")
    else:
        lengths = [p.total_length for p in legal_paths]
        seg_counts = [len(p.segments) for p in legal_paths]
        print("[INFO] Path length statistics:")
        print(f"  Min: {min(lengths):.2f} m")
        print(f"  Max: {max(lengths):.2f} m")
        print(f"  Mean: {np.mean(lengths):.2f} m")
        print(f"  Median: {np.median(lengths):.2f} m")
        print("[INFO] Segments per path:")
        print(f"  Min: {min(seg_counts)}")
        print(f"  Max: {max(seg_counts)}")
        print(f"  Mean: {np.mean(seg_counts):.2f}")

    # Build candidate list: name + ideal signature
    params = {
        "max_yaw_diff_deg": args.max_yaw_diff,
        "connect_radius_m": args.connect_radius,
        "min_path_length_m": args.min_path_length,
        "max_paths": args.max_paths,
        "max_depth": args.max_depth,
        "turn_frame": "WORLD_FRAME",
    }

    candidates = []
    for i, p in enumerate(legal_paths):
        sig = build_path_signature(p)
        name = make_path_name(i, sig)
        sig["segments_detailed"] = build_segments_detailed_for_path(p, polyline_sample_n=10)
        candidates.append({"name": name, "signature": sig})

    lane_ids_by_road: Dict[int, set] = {}
    for s in cropped_segments:
        try:
            rid = int(s.road_id)
            lid = int(s.lane_id)
        except Exception:
            continue
        lane_ids_by_road.setdefault(rid, set()).add(lid)
    lane_counts_by_road = {rid: len(lanes) for rid, lanes in lane_ids_by_road.items()}

    save_prompt_file(
        args.out_prompt,
        crop=crop,
        nodes_path=args.nodes,
        params=params,
        paths_named=candidates,
    )

    if args.out_json_detailed:
        save_aggregated_signatures_json(
            args.out_json_detailed,
            crop=crop,
            nodes_path=args.nodes,
            params=params,
            paths_named=candidates,
            lane_counts_by_road=lane_counts_by_road,
        )

    if args.viz:
        visualize_legal_paths(cropped_segments, legal_paths, crop, args.out)

    if args.individual:
        os.makedirs(args.individual_dir, exist_ok=True)
        visualize_individual_paths(cropped_segments, legal_paths, crop, args.individual_dir)


if __name__ == "__main__":
    main()
