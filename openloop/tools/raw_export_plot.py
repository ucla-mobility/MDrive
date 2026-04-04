#!/usr/bin/env python3
"""
Stage 1 + Stage 2 multi-frame trajectory plotter.

Loads canonical-export records (which contain raw world positions, canonical
predictions AND ground-truth) and generates a per-planner grid of frames
so you can see:
  - Orange dashed  : raw planner waypoints in world frame (Stage 1 output)
  - Blue solid     : canonical interpolated predictions (Stage 2 output)
  - Green squares  : ground-truth trajectory at canonical timestamps

When --canonical-dir is not given, the script auto-detects it as the
'canonical_exports' sibling of --raw-dir.  If only raw records are available
it falls back to plotting raw-only.

Outputs (per planner):
  {planner}_frames_grid.png        – grid of up to MAX_GRID frames
  {planner}_frames_grid_stale.png  – same but stale frames
comparison_grid.png                – side-by-side tcp vs vad on same frames
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO_ROOT / "results" / "openloop" / "raw_exports"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "openloop" / "raw_plots"

MAX_GRID = 12   # max frames shown in each grid figure
COLS = 4        # columns per grid


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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _to_arr(v: Any) -> Optional[np.ndarray]:
    if v is None:
        return None
    try:
        a = np.asarray(v, dtype=np.float64)
    except Exception:
        return None
    if a.ndim != 2 or a.shape[1] != 2 or len(a) == 0:
        return None
    return a if np.isfinite(a).all() else None


def _load_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


def _sample_evenly(records: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """Return n evenly-spaced records from the list."""
    if len(records) <= n:
        return records
    indices = [round(i * (len(records) - 1) / (n - 1)) for i in range(n)]
    return [records[i] for i in indices]


# ---------------------------------------------------------------------------
# Per-cell subplot renderer
# ---------------------------------------------------------------------------

def _draw_cell(ax: plt.Axes, rec: Dict[str, Any], show_legend: bool = False) -> None:
    """
    Draw one frame cell onto *ax*.
    Uses canonical record fields so all three curves share world coords.
    Falls back gracefully when fields are absent.
    """
    planner   = rec.get("planner_name", "?")
    frame_id  = rec.get("frame_id", "?")
    is_fresh  = rec.get("is_fresh_plan", False)
    reused    = rec.get("reused_from_step")

    raw_world = _to_arr(rec.get("raw_world_positions"))
    canonical = _to_arr(rec.get("canonical_positions"))
    gt        = _to_arr(rec.get("gt_canonical_positions"))
    timestamps= rec.get("canonical_timestamps") or []
    valid_mask= rec.get("valid_mask") or []

    # ---- Gather all points to compute shared axis limits ----
    all_pts = []
    for arr in (raw_world, canonical, gt):
        if arr is not None:
            all_pts.append(arr)
    if not all_pts:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"frame {frame_id}", fontsize=8)
        return

    combined = np.vstack(all_pts)
    # Apply valid_mask to canonical before computing limits
    if canonical is not None and valid_mask:
        n = min(len(canonical), len(valid_mask))
        canonical_valid = canonical[:n][np.array(valid_mask[:n], dtype=bool)]
        if len(canonical_valid):
            combined = np.vstack([p for p in (raw_world, canonical_valid, gt) if p is not None and len(p)])

    cx, cy  = combined[:, 0].mean(), combined[:, 1].mean()
    half    = max(combined[:, 0].ptp(), combined[:, 1].ptp()) / 2.0 * 1.35
    half    = max(half, 3.0)  # minimum 6 m window

    # ---- Ego start marker (origin of prediction = first raw world point) ----
    if raw_world is not None and len(raw_world):
        ax.plot(raw_world[0, 0], raw_world[0, 1],
                "k^", ms=7, zorder=10, label="ego pos" if show_legend else "")

    # ---- Raw world waypoints (Stage 1) ----
    if raw_world is not None and len(raw_world):
        ax.plot(raw_world[:, 0], raw_world[:, 1],
                "o--", color="darkorange", lw=1.5, ms=4, zorder=4,
                label="raw (S1)" if show_legend else "")

    # ---- Canonical prediction (Stage 2), only valid points ----
    if canonical is not None and len(canonical):
        if valid_mask:
            n = min(len(canonical), len(valid_mask))
            mask = np.array(valid_mask[:n], dtype=bool)
            c_pts = canonical[:n]
            valid_c = c_pts[mask]
            invalid_c = c_pts[~mask]
        else:
            valid_c = canonical
            invalid_c = np.empty((0, 2))

        if len(valid_c):
            ax.plot(valid_c[:, 0], valid_c[:, 1],
                    "s-", color="royalblue", lw=2, ms=5, zorder=5,
                    label="canonical (S2)" if show_legend else "")
            # timestamp labels at each point
            ts_list = list(timestamps) if timestamps else []
            for i, pt in enumerate(valid_c):
                t = ts_list[i] if i < len(ts_list) else None
                if t is not None:
                    ax.annotate(f"{t:.1f}s", (pt[0], pt[1]),
                                fontsize=5, color="royalblue",
                                xytext=(3, 3), textcoords="offset points")
        if len(invalid_c):
            ax.scatter(invalid_c[:, 0], invalid_c[:, 1],
                       marker="x", c="lightblue", s=20, zorder=5)

    # ---- Ground-truth trajectory (GT) ----
    if gt is not None and len(gt):
        gt_valid = gt
        if valid_mask:
            n = min(len(gt), len(valid_mask))
            gt_mask = np.array(valid_mask[:n], dtype=bool)
            gt_valid_pts = gt[:n][gt_mask]
            if len(gt_valid_pts):
                gt_valid = gt_valid_pts
        ax.plot(gt_valid[:, 0], gt_valid[:, 1],
                "D-", color="forestgreen", lw=1.5, ms=5, zorder=6,
                label="GT" if show_legend else "")

    # ---- Error lines between canonical and GT ----
    if canonical is not None and gt is not None:
        n = min(len(canonical), len(gt), len(valid_mask) if valid_mask else 9999)
        for i in range(n):
            vm = valid_mask[i] if i < len(valid_mask) else True
            if vm:
                ax.plot([canonical[i, 0], gt[i, 0]],
                        [canonical[i, 1], gt[i, 1]],
                        "-", color="red", lw=0.8, alpha=0.5, zorder=3)

    # ---- Axis limits & cosmetics ----
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=5)
    ax.grid(True, lw=0.4, alpha=0.4)

    # ADE for this frame
    ade_str = ""
    if canonical is not None and gt is not None and valid_mask:
        n = min(len(canonical), len(gt), len(valid_mask))
        mask = np.array(valid_mask[:n], dtype=bool)
        if mask.any():
            errs = np.linalg.norm(canonical[:n][mask] - gt[:n][mask], axis=1)
            ade_str = f"  ADE={errs.mean():.2f}m"

    freshness = "fresh" if is_fresh else f"stale(←{reused})"
    ax.set_title(f"frame {frame_id}  [{freshness}]{ade_str}", fontsize=7, pad=2)


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------

def _make_grid(
    records: List[Dict[str, Any]],
    planner: str,
    scenario_id: str,
    out_path: Path,
    n_frames: int = 8,
    title_suffix: str = "",
) -> None:
    """Sample n_frames evenly and plot them in a COLS-wide grid."""
    sampled = _sample_evenly(records, n_frames)
    n = len(sampled)
    if n == 0:
        return
    cols = min(COLS, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.8, rows * 3.8))
    axes = np.array(axes).reshape(-1) if n > 1 else np.array([axes])

    for idx, rec in enumerate(sampled):
        _draw_cell(axes[idx], rec, show_legend=(idx == 0))

    # Hide unused cells
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    # Shared legend from first cell
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        extra = [
            mpatches.Patch(color="darkorange", label="raw world wpts (Stage 1)"),
            mpatches.Patch(color="royalblue",  label="canonical prediction (Stage 2)"),
            mpatches.Patch(color="forestgreen",label="ground truth"),
            mpatches.Patch(color="red",        label="prediction error"),
        ]
        fig.legend(handles=extra, loc="lower center", ncol=4,
                   fontsize=8, framealpha=0.9,
                   bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        f"{planner}  |  {scenario_id}{title_suffix}\n"
        f"Showing {n} evenly-spaced {'fresh' if 'fresh' in title_suffix else ''} frames  "
        f"(orange=raw S1  blue=canonical S2  green=GT  red=error)",
        fontsize=9, y=1.01,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


# ---------------------------------------------------------------------------
# Comparison: two planners side-by-side on matched frames
# ---------------------------------------------------------------------------

def _make_comparison_grid(
    planners_records: Dict[str, List[Dict[str, Any]]],
    scenario_id: str,
    out_path: Path,
    n_frames: int = 6,
) -> None:
    """
    Rows = sampled frames  |  Columns = one per planner.
    Frames are chosen by frame_id intersection so both planners show the same world state.
    """
    if len(planners_records) < 2:
        return

    planner_names = list(planners_records.keys())
    # Build frame_id → record maps
    by_fid: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for pname, recs in planners_records.items():
        by_fid[pname] = {r["frame_id"]: r for r in recs if "frame_id" in r}

    # Try exact frame_id intersection first; fall back to nearest-match
    common_exact = sorted(
        set.intersection(*[set(d.keys()) for d in by_fid.values()])
    )

    if common_exact:
        common_fids = common_exact
        # pair: each planner uses the record at that frame_id exactly
        def _get(pname: str, fid: int) -> Optional[Dict[str, Any]]:
            return by_fid[pname].get(fid)
    else:
        # No exact overlap: use the union of all fresh frame IDs and match
        # each planner to its nearest available frame.
        all_fids = sorted(set().union(*[set(d.keys()) for d in by_fid.values()]))
        common_fids = all_fids

        def _get(pname: str, fid: int) -> Optional[Dict[str, Any]]:  # type: ignore[misc]
            d = by_fid[pname]
            if fid in d:
                return d[fid]
            if not d:
                return None
            nearest = min(d.keys(), key=lambda k: abs(k - fid))
            return d[nearest]

    # Evenly sample
    if len(common_fids) > n_frames:
        idx = [round(i * (len(common_fids) - 1) / (n_frames - 1)) for i in range(n_frames)]
        common_fids = [common_fids[i] for i in idx]

    n_cols = len(planner_names)
    n_rows = len(common_fids)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.8, n_rows * 3.8),
                              squeeze=False)

    for row, fid in enumerate(common_fids):
        for col, pname in enumerate(planner_names):
            rec = _get(pname, fid)
            ax = axes[row][col]
            if rec is not None:
                _draw_cell(ax, rec, show_legend=False)
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes)
            if row == 0:
                ax.set_title(f"{pname}\nframe {fid}", fontsize=8)
            else:
                ax.set_title(f"frame {fid}", fontsize=7, pad=2)

    extra = [
        mpatches.Patch(color="darkorange", label="raw world wpts (Stage 1)"),
        mpatches.Patch(color="royalblue",  label="canonical prediction (Stage 2)"),
        mpatches.Patch(color="forestgreen",label="ground truth"),
        mpatches.Patch(color="red",        label="prediction error"),
    ]
    fig.legend(handles=extra, loc="lower center", ncol=4, fontsize=8,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        f"Planner comparison  |  {scenario_id}  |  {n_rows} matched frames\n"
        "(orange=raw S1  blue=canonical S2  green=GT  red=error)",
        fontsize=9, y=1.01,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(canonical_dir: Path, out_dir: Path, n_frames: int = 8) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    planner_dirs = sorted(p for p in canonical_dir.iterdir() if p.is_dir()) \
        if canonical_dir.exists() else []
    if not planner_dirs:
        print(f"No planner subdirectories found under {canonical_dir}")
        return

    all_fresh: Dict[str, List[Dict[str, Any]]] = {}

    for planner_dir in planner_dirs:
        pname = planner_dir.name
        records: List[Dict[str, Any]] = []
        for jsonl_path in sorted(planner_dir.glob("*.jsonl")):
            records.extend(_load_records(jsonl_path))

        if not records:
            print(f"  [skip] {pname}: no records")
            continue

        scenario_id = records[0].get("scenario_id", "unknown")

        fresh   = [r for r in records if r.get("is_fresh_plan") and _to_arr(r.get("raw_world_positions")) is not None]
        stale   = [r for r in records if not r.get("is_fresh_plan") and _to_arr(r.get("raw_world_positions")) is not None]

        print(f"  {pname}: {len(fresh)} fresh, {len(stale)} stale records")

        if fresh:
            _make_grid(fresh, pname, scenario_id,
                       out_dir / f"{pname}_fresh_grid.png",
                       n_frames=n_frames, title_suffix="  [fresh frames only]")
            all_fresh[pname] = fresh

        if stale:
            # Show fewer stale frames — just 4 to confirm reuse looks sane
            _make_grid(stale, pname, scenario_id,
                       out_dir / f"{pname}_stale_sample.png",
                       n_frames=min(4, len(stale)), title_suffix="  [stale/reused sample]")

    # Cross-planner comparison on fresh frames
    if len(all_fresh) >= 2:
        scenario_id = next(iter(all_fresh.values()))[0].get("scenario_id", "unknown")
        _make_comparison_grid(all_fresh, scenario_id,
                              out_dir / "comparison_fresh_grid.png",
                              n_frames=n_frames)

    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir",
                        default=str(DEFAULT_RAW_DIR),
                        help="Stage 1 raw_exports dir (used to auto-detect canonical dir)")
    parser.add_argument("--canonical-dir",
                        default=None,
                        help="Stage 2 canonical_exports dir (default: sibling of --raw-dir)")
    parser.add_argument("--out-dir",
                        default=str(DEFAULT_OUT_DIR),
                        help="Output directory for PNGs")
    parser.add_argument("--n-frames",
                        type=int, default=8,
                        help="Number of evenly-spaced frames per grid (default: 8)")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if args.canonical_dir:
        canonical_dir = Path(args.canonical_dir)
    else:
        # auto-detect: canonical_exports sits next to raw_exports
        canonical_dir = raw_dir.parent / "canonical_exports"

    if not canonical_dir.exists():
        # fallback: try raw records only (limited info)
        print(f"[warn] canonical_dir not found ({canonical_dir}), trying raw_dir")
        canonical_dir = raw_dir

    run(canonical_dir, Path(args.out_dir), n_frames=args.n_frames)
