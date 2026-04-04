#!/usr/bin/env python3
"""Plot Stage 2 canonical trajectory artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANONICAL_DIR = REPO_ROOT / "results" / "openloop" / "canonical_exports"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "openloop" / "canonical_plots"


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _xy_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) == 0:
        return None
    if not np.isfinite(arr).all():
        return None
    return arr


def _mask_array(value: Any, expected: int) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=bool).reshape(-1)
    except Exception:
        return np.zeros(expected, dtype=bool)
    if len(arr) != expected:
        return np.zeros(expected, dtype=bool)
    return arr


def _pick_record(records: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    fresh_valid = [
        r for r in records
        if bool(r.get("is_fresh_plan"))
        and _xy_array(r.get("canonical_positions")) is not None
        and _mask_array(r.get("valid_mask"), 6).any()
    ]
    if fresh_valid:
        return fresh_valid[len(fresh_valid) // 2]
    any_valid = [
        r for r in records
        if _xy_array(r.get("canonical_positions")) is not None
        and _mask_array(r.get("valid_mask"), 6).any()
    ]
    if any_valid:
        return any_valid[len(any_valid) // 2]
    return None


def _bounds(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    union = np.vstack(arrays)
    mins = union.min(axis=0)
    maxs = union.max(axis=0)
    if np.allclose(mins, maxs):
        mins = mins - 1.0
        maxs = maxs + 1.0
    return mins, maxs


def _timestamp_error_lines(record: dict[str, Any], canonical: np.ndarray, valid_mask: np.ndarray) -> list[str]:
    gt = _xy_array(record.get("gt_canonical_positions"))
    gt_mask = _mask_array(record.get("gt_valid_mask"), len(canonical))
    try:
        timestamps = np.asarray(record.get("canonical_timestamps"), dtype=np.float64).reshape(-1)
    except Exception:
        timestamps = np.arange(1, len(canonical) + 1, dtype=np.float64)
    if len(timestamps) != len(canonical):
        timestamps = np.arange(1, len(canonical) + 1, dtype=np.float64)

    lines = ["per-timestamp error vs GT:"]
    for idx, ts in enumerate(timestamps):
        if gt is None or len(gt) != len(canonical) or not valid_mask[idx] or not gt_mask[idx]:
            lines.append(f"t={float(ts):>3.1f}s  err=n/a")
            continue
        err = float(np.linalg.norm(canonical[idx] - gt[idx]))
        lines.append(f"t={float(ts):>3.1f}s  err={err:>5.2f} m")
    return lines


def _plot_record(record: dict[str, Any], out_path: Path) -> None:
    canonical = _xy_array(record.get("canonical_positions"))
    if canonical is None:
        return
    valid_mask = _mask_array(record.get("valid_mask"), len(canonical))
    gt = _xy_array(record.get("gt_canonical_positions"))
    gt_mask = _mask_array(record.get("gt_valid_mask"), len(canonical))
    raw_world = _xy_array(record.get("raw_world_positions"))

    arrays = [canonical[valid_mask]] if np.any(valid_mask) else [canonical]
    if raw_world is not None:
        arrays.append(raw_world)
    if gt is not None and np.any(gt_mask):
        arrays.append(gt[gt_mask])
    mins, maxs = _bounds([arr for arr in arrays if len(arr) > 0])

    width = 1260
    height = 900
    margin = 90
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)
    span = np.maximum(maxs - mins, 1e-6)
    plot_right = 860
    scale = min((plot_right - margin) / float(span[0]), (height - 2 * margin) / float(span[1]))
    center = (mins + maxs) / 2.0

    def _project(point: np.ndarray) -> tuple[float, float]:
        x = (float(point[0]) - float(center[0])) * scale + ((plot_right + margin) / 2.0)
        y = (height / 2.0) - (float(point[1]) - float(center[1])) * scale
        return x, y

    for gx in range(margin, plot_right + 1, 100):
        draw.line([(gx, margin), (gx, height - margin)], fill=(225, 225, 225), width=1)
    for gy in range(margin, height - margin + 1, 100):
        draw.line([(margin, gy), (plot_right, gy)], fill=(225, 225, 225), width=1)

    if raw_world is not None and len(raw_world) > 0:
        raw_px = [_project(pt) for pt in raw_world]
        if len(raw_px) >= 2:
            draw.line(raw_px, fill=(255, 140, 0), width=2)
        for idx, point in enumerate(raw_px):
            x, y = point
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 165, 0), outline=(160, 90, 0))
            draw.text((x + 5, y + 3), f"r{idx}", fill="black")

    canonical_px = [_project(pt) for pt in canonical]
    for idx in range(1, len(canonical_px)):
        if valid_mask[idx - 1] and valid_mask[idx]:
            draw.line([canonical_px[idx - 1], canonical_px[idx]], fill=(0, 90, 220), width=3)
    for idx, point in enumerate(canonical_px):
        x, y = point
        color = (0, 90, 220) if valid_mask[idx] else (180, 180, 180)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline=(0, 0, 0))
        draw.text((x + 6, y + 2), f"c{idx}", fill="black")

    if gt is not None and len(gt) == len(canonical):
        gt_px = [_project(pt) for pt in gt]
        for idx in range(1, len(gt_px)):
            if gt_mask[idx - 1] and gt_mask[idx]:
                draw.line([gt_px[idx - 1], gt_px[idx]], fill=(0, 170, 120), width=2)
        for idx, point in enumerate(gt_px):
            if not gt_mask[idx]:
                continue
            x, y = point
            draw.rectangle((x - 4, y - 4, x + 4, y + 4), fill=(0, 170, 120), outline=(0, 0, 0))
            draw.text((x + 6, y + 2), f"g{idx}", fill="black")

    title_lines = [
        f"{record.get('planner_name', 'unknown')} | {record.get('scenario_id', 'unknown')}",
        (
            f"frame={record.get('frame_id')}  fresh={record.get('is_fresh_plan')}  "
            f"timestamp_source={record.get('timestamp_source', 'unknown')}"
        ),
        (
            f"interp={record.get('interpolation_method', 'unknown')}  "
            f"valid={int(valid_mask.sum())}/{len(valid_mask)}"
        ),
        "raw=orange  canonical=blue  gt@canonical=green",
    ]
    text_y = 16
    for line in title_lines:
        draw.text((16, text_y), line, fill="black")
        text_y += 18

    info_x = 900
    info_y = 36
    for line in _timestamp_error_lines(record, canonical, valid_mask):
        draw.text((info_x, info_y), line, fill="black")
        info_y += 18

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def run(canonical_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    planner_dirs = sorted(p for p in canonical_dir.iterdir() if p.is_dir()) if canonical_dir.exists() else []
    if not planner_dirs:
        print(f"No planner subdirectories found under {canonical_dir}")
        return

    generated = 0
    for planner_dir in planner_dirs:
        for jsonl_path in sorted(planner_dir.glob("*.jsonl")):
            record = _pick_record(_load_records(jsonl_path))
            if record is None:
                print(f"  [skip] {planner_dir.name}/{jsonl_path.name}: no plottable canonical record")
                continue
            out_path = out_dir / f"{planner_dir.name}_{jsonl_path.stem}.png"
            _plot_record(record, out_path)
            print(f"  saved {out_path.name}")
            generated += 1
    print(f"\nGenerated {generated} canonical debug plot(s) in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", default=str(DEFAULT_CANONICAL_DIR), help="Directory containing canonical JSONL files")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Directory for PNG output")
    args = parser.parse_args()
    run(Path(args.canonical_dir), Path(args.out_dir))
