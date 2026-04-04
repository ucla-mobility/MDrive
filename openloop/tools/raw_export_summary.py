#!/usr/bin/env python3
"""Summarize Stage 1 raw trajectory export artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO_ROOT / "results" / "openloop" / "raw_exports"


def _iter_records(raw_dir: Path):
    if not raw_dir.exists():
        return
    for planner_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        for jsonl_path in sorted(planner_dir.glob("*.jsonl")):
            with jsonl_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        yield planner_dir.name, jsonl_path, payload


def _fmt_counter(counter: Counter[Any]) -> str:
    if not counter:
        return "{}"
    items = ", ".join(f"{key}: {value}" for key, value in sorted(counter.items(), key=lambda item: str(item[0])))
    return "{" + items + "}"


def run(raw_dir: Path) -> None:
    planner_stats: dict[str, dict[str, Any]] = {}
    for planner_name, _jsonl_path, record in _iter_records(raw_dir):
        stats = planner_stats.setdefault(
            planner_name,
            {
                "records": 0,
                "fresh": 0,
                "reused": 0,
                "shapes": Counter(),
                "native_dt": Counter(),
                "point0_mode": Counter(),
            },
        )
        stats["records"] += 1
        shape = record.get("raw_shape")
        stats["shapes"][tuple(shape) if isinstance(shape, list) else shape] += 1
        dt = record.get("native_dt")
        stats["native_dt"][dt] += 1
        stats["point0_mode"][record.get("point0_mode", "unknown")] += 1
        if record.get("is_fresh_plan") is True:
            stats["fresh"] += 1
        if record.get("is_fresh_plan") is False and record.get("raw_positions") is not None:
            stats["reused"] += 1

    if not planner_stats:
        print(f"No raw export records found under {raw_dir}")
        return

    for planner_name in sorted(planner_stats):
        stats = planner_stats[planner_name]
        total = max(1, stats["records"])
        fresh_fraction = stats["fresh"] / total
        reused_fraction = stats["reused"] / total
        print(f"[{planner_name}]")
        print(f"  records: {stats['records']}")
        print(f"  shapes observed: {_fmt_counter(stats['shapes'])}")
        print(f"  fresh fraction: {fresh_fraction:.3f}")
        print(f"  reused fraction: {reused_fraction:.3f}")
        print(f"  native_dt values seen: {_fmt_counter(stats['native_dt'])}")
        print(f"  point0_mode counts: {_fmt_counter(stats['point0_mode'])}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="Directory containing raw export JSONL files")
    args = parser.parse_args()
    run(Path(args.raw_dir))
