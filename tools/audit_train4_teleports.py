#!/usr/bin/env python3
"""
Audit low-motion teleport events for the real->sim trajectory pipeline.

Gate condition:
  - Fail when any scenario has final CARLA teleport events, where:
      raw_step < raw_step_threshold and carla_final_step > jump_threshold
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v2xpnp.pipeline import route_export as ytm
from v2xpnp.pipeline.pipeline_runtime import (
    _is_scenario_directory,
    _parse_lane_type_set,
    load_vector_map,
    process_single_scenario,
)


@dataclass
class ScenarioMetrics:
    name: str
    map_name: str
    teleport_events_v2: int
    teleport_events_pre: int
    teleport_events_final: int
    line_change_count: int
    frame_pairs: int
    median_carla_raw_dist_m: float
    top_sources: List[Tuple[str, int]]
    top_tracks: List[Tuple[str, int]]

    def to_json(self) -> Dict[str, object]:
        return {
            "scenario": self.name,
            "map_name": self.map_name,
            "teleport_events_v2": int(self.teleport_events_v2),
            "teleport_events_pre": int(self.teleport_events_pre),
            "teleport_events_final": int(self.teleport_events_final),
            "line_change_count": int(self.line_change_count),
            "frame_pairs": int(self.frame_pairs),
            "median_carla_raw_dist_m": float(self.median_carla_raw_dist_m),
            "top_sources": [[str(k), int(v)] for k, v in self.top_sources],
            "top_tracks": [[str(k), int(v)] for k, v in self.top_tracks],
        }


def _step_xy(fr: Dict[str, object], key_x: str, key_y: str) -> Tuple[float, float]:
    return (
        float(fr.get(key_x, fr.get("x", 0.0))),
        float(fr.get(key_y, fr.get("y", 0.0))),
    )


def _track_line_changes(frames: List[Dict[str, object]]) -> int:
    prev: Optional[int] = None
    changes = 0
    for fr in frames:
        cli = int(fr.get("ccli", -1))
        if cli < 0:
            continue
        if prev is not None and int(cli) != int(prev):
            changes += 1
        prev = int(cli)
    return int(changes)


def _scenario_metrics(
    dataset: Dict[str, object],
    raw_step_threshold: float,
    jump_threshold: float,
) -> ScenarioMetrics:
    stage_counts = {
        "v2": 0,
        "pre": 0,
        "final": 0,
    }
    line_changes = 0
    frame_pairs = 0
    source_counter: Counter[str] = Counter()
    track_counter: Counter[str] = Counter()
    carla_raw_dists: List[float] = []

    tracks = dataset.get("tracks", [])
    if not isinstance(tracks, list):
        tracks = []

    for tr in tracks:
        if not isinstance(tr, dict):
            continue
        if str(tr.get("role", "")).strip().lower() != "vehicle":
            continue
        tr_id = str(tr.get("id", "unknown"))
        frames = tr.get("frames", [])
        if not isinstance(frames, list) or len(frames) < 2:
            continue
        line_changes += int(_track_line_changes(frames))

        for fr in frames:
            if (
                "cx" in fr
                and "cy" in fr
                and "x" in fr
                and "y" in fr
            ):
                try:
                    dx = float(fr["cx"]) - float(fr["x"])
                    dy = float(fr["cy"]) - float(fr["y"])
                    carla_raw_dists.append(float(math.hypot(dx, dy)))
                except Exception:
                    pass

        for i in range(1, len(frames)):
            a = frames[i - 1]
            b = frames[i]
            raw_dx = float(b.get("x", 0.0)) - float(a.get("x", 0.0))
            raw_dy = float(b.get("y", 0.0)) - float(a.get("y", 0.0))
            raw_step = float(math.hypot(raw_dx, raw_dy))
            frame_pairs += 1
            if raw_step >= float(raw_step_threshold):
                continue

            ax, ay = _step_xy(a, "sx", "sy")
            bx, by = _step_xy(b, "sx", "sy")
            if float(math.hypot(float(bx) - float(ax), float(by) - float(ay))) > float(jump_threshold):
                stage_counts["v2"] += 1

            ax, ay = _step_xy(a, "cbx", "cby")
            bx, by = _step_xy(b, "cbx", "cby")
            if float(math.hypot(float(bx) - float(ax), float(by) - float(ay))) > float(jump_threshold):
                stage_counts["pre"] += 1

            ax, ay = _step_xy(a, "cx", "cy")
            bx, by = _step_xy(b, "cx", "cy")
            final_step = float(math.hypot(float(bx) - float(ax), float(by) - float(ay)))
            if final_step > float(jump_threshold):
                stage_counts["final"] += 1
                source_counter[str(b.get("csource", "unknown"))] += 1
                track_counter[str(tr_id)] += 1

    if carla_raw_dists:
        sorted_d = sorted(float(v) for v in carla_raw_dists if math.isfinite(float(v)))
        if sorted_d:
            med = sorted_d[len(sorted_d) // 2]
        else:
            med = float("nan")
    else:
        med = float("nan")

    return ScenarioMetrics(
        name=str(dataset.get("scenario_name", "unknown")),
        map_name=str(dataset.get("map_name", "")),
        teleport_events_v2=int(stage_counts["v2"]),
        teleport_events_pre=int(stage_counts["pre"]),
        teleport_events_final=int(stage_counts["final"]),
        line_change_count=int(line_changes),
        frame_pairs=int(frame_pairs),
        median_carla_raw_dist_m=float(med),
        top_sources=source_counter.most_common(10),
        top_tracks=track_counter.most_common(10),
    )


def _load_carla_runtime(args: argparse.Namespace) -> Dict[str, object]:
    cache_path = Path(args.carla_map_cache).expanduser().resolve()
    align_path = Path(args.carla_map_offset_json).expanduser().resolve()
    raw_lines, raw_bounds, carla_map_name = ytm._load_carla_map_cache_lines(cache_path)
    align_cfg = ytm._load_carla_alignment_cfg(align_path)
    transformed_lines, transformed_bbox = ytm._transform_carla_lines(raw_lines, align_cfg)
    return {
        "lines_xy": transformed_lines,
        "bbox": transformed_bbox,
        "source_path": str(cache_path),
        "map_name": str(carla_map_name or "carla_map_cache"),
        "lane_corr_top_k": int(args.lane_correspondence_top_k),
        "lane_corr_cache_dir": Path(args.lane_correspondence_cache_dir).expanduser().resolve(),
        "lane_corr_driving_types": _parse_lane_type_set(args.lane_correspondence_driving_types),
        "raw_bounds": raw_bounds,
        "align_cfg": dict(align_cfg),
    }


def _load_maps(paths: Iterable[str]) -> List[object]:
    out = []
    for p in paths:
        out.append(load_vector_map(Path(p).expanduser().resolve()))
    return out


def _iter_scenarios(root: Path) -> List[Path]:
    if _is_scenario_directory(root):
        return [root]
    if not root.is_dir():
        return []
    out = [p for p in sorted(root.iterdir()) if _is_scenario_directory(p)]
    return out


def _run(args: argparse.Namespace) -> int:
    scenario_root = Path(args.scenario_root).expanduser().resolve()
    scenarios = _iter_scenarios(scenario_root)
    if not scenarios:
        print(f"[ERROR] No scenarios found under: {scenario_root}", file=sys.stderr)
        return 2

    map_data_list = _load_maps(args.map_pkl)
    carla_runtime = _load_carla_runtime(args)
    carla_context_cache: Dict[str, Dict[str, object]] = {}

    metrics: List[ScenarioMetrics] = []
    totals = defaultdict(int)
    medians: List[float] = []

    for scenario in scenarios:
        if args.show_pipeline_logs:
            dataset = process_single_scenario(
                scenario_dir=scenario,
                map_data_list=map_data_list,
                dt=float(args.dt),
                tx=0.0,
                ty=0.0,
                tz=0.0,
                yaw_deg=0.0,
                flip_y=False,
                subdir=None,
                carla_runtime=carla_runtime,
                enable_carla_projection=True,
                timing_cfg={
                    "maximize_safe_early_spawn": True,
                    "maximize_safe_late_despawn": True,
                    "early_spawn_safety_margin": 0.25,
                    "late_despawn_safety_margin": 0.25,
                },
                walker_cfg={"enabled": False},
                carla_context_cache=carla_context_cache,
                verbose=False,
            )
        else:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                dataset = process_single_scenario(
                    scenario_dir=scenario,
                    map_data_list=map_data_list,
                    dt=float(args.dt),
                    tx=0.0,
                    ty=0.0,
                    tz=0.0,
                    yaw_deg=0.0,
                    flip_y=False,
                    subdir=None,
                    carla_runtime=carla_runtime,
                    enable_carla_projection=True,
                    timing_cfg={
                        "maximize_safe_early_spawn": True,
                        "maximize_safe_late_despawn": True,
                        "early_spawn_safety_margin": 0.25,
                        "late_despawn_safety_margin": 0.25,
                    },
                    walker_cfg={"enabled": False},
                    carla_context_cache=carla_context_cache,
                    verbose=False,
                )
        if dataset is None:
            continue
        sm = _scenario_metrics(
            dataset=dataset,
            raw_step_threshold=float(args.raw_step_threshold),
            jump_threshold=float(args.jump_threshold),
        )
        metrics.append(sm)
        totals["teleport_events_v2"] += int(sm.teleport_events_v2)
        totals["teleport_events_pre"] += int(sm.teleport_events_pre)
        totals["teleport_events_final"] += int(sm.teleport_events_final)
        totals["line_change_count"] += int(sm.line_change_count)
        totals["frame_pairs"] += int(sm.frame_pairs)
        if math.isfinite(float(sm.median_carla_raw_dist_m)):
            medians.append(float(sm.median_carla_raw_dist_m))

    failed = [m.name for m in metrics if int(m.teleport_events_final) > 0]
    global_median = float("nan")
    if medians:
        medians_sorted = sorted(medians)
        global_median = float(medians_sorted[len(medians_sorted) // 2])

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_root": str(scenario_root),
        "scenario_count": int(len(metrics)),
        "thresholds": {
            "raw_step_threshold_m": float(args.raw_step_threshold),
            "jump_threshold_m": float(args.jump_threshold),
        },
        "totals": {
            "teleport_events_v2": int(totals["teleport_events_v2"]),
            "teleport_events_pre": int(totals["teleport_events_pre"]),
            "teleport_events_final": int(totals["teleport_events_final"]),
            "line_change_count": int(totals["line_change_count"]),
            "frame_pairs": int(totals["frame_pairs"]),
            "median_carla_raw_dist_m": float(global_median),
        },
        "failed_scenarios": failed,
        "scenarios": [m.to_json() for m in metrics],
    }

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[INFO] Wrote report: {out_path}")

    print(
        "[SUMMARY] scenarios={} final_teleports={} pre_teleports={} v2_teleports={} line_changes={} median_carla_raw_dist_m={:.3f}".format(
            int(len(metrics)),
            int(totals["teleport_events_final"]),
            int(totals["teleport_events_pre"]),
            int(totals["teleport_events_v2"]),
            int(totals["line_change_count"]),
            float(global_median) if math.isfinite(float(global_median)) else float("nan"),
        )
    )
    if failed:
        print("[FAIL] final teleport events present in scenarios:")
        for name in failed:
            print(f"  - {name}")
        return 1
    print("[PASS] final teleport events = 0 for all scenarios")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit train4 (or a scenario tree) for low-motion teleport events."
    )
    parser.add_argument(
        "--scenario-root",
        type=str,
        default="/data2/marco/CoLMDriver/v2xpnp/dataset/train4",
        help="Scenario root directory (single scenario dir also supported).",
    )
    parser.add_argument(
        "--map-pkl",
        nargs="+",
        default=[
            "/data2/marco/CoLMDriver/v2xpnp/map/v2v_corridors_vector_map.pkl",
            "/data2/marco/CoLMDriver/v2xpnp/map/v2x_intersection_vector_map.pkl",
        ],
        help="Vector map pickle paths.",
    )
    parser.add_argument(
        "--carla-map-cache",
        type=str,
        default="/data2/marco/CoLMDriver/v2xpnp/map/carla_map_cache.pkl",
        help="Path to CARLA map cache pickle.",
    )
    parser.add_argument(
        "--carla-map-offset-json",
        type=str,
        default="/data2/marco/CoLMDriver/v2xpnp/map/ucla_map_offset_carla.json",
        help="Path to CARLA alignment JSON.",
    )
    parser.add_argument(
        "--lane-correspondence-top-k",
        type=int,
        default=56,
        help="Candidate CARLA lines per V2 lane during correspondence.",
    )
    parser.add_argument(
        "--lane-correspondence-cache-dir",
        type=str,
        default="/tmp/v2x_stage_corr_cache",
        help="Cache directory for lane correspondence.",
    )
    parser.add_argument(
        "--lane-correspondence-driving-types",
        type=str,
        default="1",
        help="Comma/space-separated V2 lane types treated as driving.",
    )
    parser.add_argument(
        "--raw-step-threshold",
        type=float,
        default=0.6,
        help="Raw step threshold (m) for teleport event definition.",
    )
    parser.add_argument(
        "--jump-threshold",
        type=float,
        default=1.8,
        help="Projected step threshold (m) for teleport event definition.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Frame dt for scenario processing.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--show-pipeline-logs",
        action="store_true",
        help="Show pipeline logs while processing scenarios.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
