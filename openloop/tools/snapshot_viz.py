#!/usr/bin/env python3
"""
Snapshot visualization: one plot per planner × scenario × ego.

For the middle valid (non-stale, has raw predictions, has ADE) frame:
  • Black line  – full GT trajectory (all logged ego positions)
  • Red dots    – GT positions at the planner's native dt intervals
                  (what the planner would need to predict for 0 ADE)
  • Orange dots – planner's raw predicted waypoints (no resampling)
  • Green star  – ego position at the selected frame

Usage:
    python snapshot_viz.py
    python snapshot_viz.py --debug-dir /path/to/debug_frames --out-dir /path/to/viz
"""
import argparse
import json
from pathlib import Path
from typing import Optional

import os
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ── Defaults ──────────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[2]
DEFAULT_DEBUG_DIR = _REPO / "results" / "openloop" / "debug_frames"
DEFAULT_OUT_DIR   = _REPO / "results" / "openloop" / "viz"

RUNTIME_DT = 0.1  # scenario playback rate (10 Hz → 0.1 s per frame)

# Planner native dt: seconds between consecutive raw predicted waypoints.
# CoDriving is inferred per-frame (depends on n_wps; see infer_native_dt).
NATIVE_DT: dict = {
    "codriving":           None,   # inferred: 0.05 s if ≥10 wps, else 0.2 s
    "colmdriver":          0.1,
    "colmdriver_rulebase": 0.1,
    "vad":                 0.5,
    "uniad":               0.5,
    "tcp":                 0.5,
    "lmdrive":             0.5,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def infer_native_dt(planner: str, n_wps: int) -> float:
    dt = NATIVE_DT.get(planner.lower())
    if dt is not None:
        return dt
    # CoDriving special case
    return 0.05 if n_wps >= 10 else 0.2


def pick_middle_frame(frames: list) -> Optional[dict]:
    """Middle frame that has raw predictions, a valid ADE, and is not stale."""
    def _valid(f):
        return (
            f.get("planner_future_world_raw") is not None
            and f.get("metrics") is not None
            and f["metrics"].get("plan_ade") is not None
            and not f.get("stale_control", True)
        )

    valid = [f for f in frames if _valid(f)]
    if not valid:
        # Relax stale constraint as fallback
        valid = [
            f for f in frames
            if (f.get("planner_future_world_raw") is not None
                and f.get("metrics") is not None
                and f["metrics"].get("plan_ade") is not None)
        ]
    return valid[len(valid) // 2] if valid else None


def compute_native_gt(gt_traj: np.ndarray, frame_idx: int,
                      native_dt: float, n_wps: int) -> np.ndarray:
    """
    Sample gt_traj at frame_idx + k * native_dt_frames for k = 1..n_wps.
    Returns shape [M, 2] where M ≤ n_wps (truncated at end of trajectory).
    """
    dt_frames = max(1, round(native_dt / RUNTIME_DT))
    indices = [frame_idx + dt_frames * k for k in range(1, n_wps + 1)]
    valid_idx = [i for i in indices if i < len(gt_traj)]
    if not valid_idx:
        return np.zeros((0, 2))
    return gt_traj[valid_idx]


def make_snap(ax, planner: str, gt_traj: np.ndarray,
              frame: dict, title_label: str) -> None:
    pred_raw  = np.array(frame["planner_future_world_raw"])    # [N, 2]
    ego_xy    = np.array(frame["ego_world_xy"])
    frame_idx = int(frame["frame_idx"])
    ade       = frame["metrics"]["plan_ade"]
    native_dt = infer_native_dt(planner, len(pred_raw))

    gt_native = compute_native_gt(gt_traj, frame_idx, native_dt, len(pred_raw))

    # ── Full GT trajectory (thin black) ──
    ax.plot(gt_traj[:, 0], gt_traj[:, 1],
            color="black", linewidth=1.2, alpha=0.35,
            label="GT trajectory (full)", zorder=1)

    # ── Red: what the planner should predict for 0 ADE ──
    if len(gt_native) > 0:
        ax.scatter(gt_native[:, 0], gt_native[:, 1],
                   color="red", s=75, zorder=4,
                   edgecolors="darkred", linewidths=0.8,
                   label=f"GT target  ({len(gt_native)} pts, Δt={native_dt}s)")
        for k, pt in enumerate(gt_native):
            ax.annotate(f"t+{(k + 1) * native_dt:.2f}s", pt,
                        xytext=(4, 3), textcoords="offset points",
                        fontsize=5.5, color="darkred", zorder=5)

    # ── Orange: raw planner predictions ──
    ax.scatter(pred_raw[:, 0], pred_raw[:, 1],
               color="orange", s=75, zorder=5,
               edgecolors="darkorange", linewidths=0.8,
               label=f"Predicted raw  ({len(pred_raw)} pts)")
    ax.plot(pred_raw[:, 0], pred_raw[:, 1],
            color="orange", alpha=0.55, linewidth=1.0, zorder=3)
    for k, pt in enumerate(pred_raw):
        ax.annotate(str(k), pt,
                    xytext=(4, 3), textcoords="offset points",
                    fontsize=5.5, color="darkorange", zorder=6)

    # ── Green star: ego ──
    ax.scatter(*ego_xy, color="limegreen", s=220, marker="*",
               zorder=7, edgecolors="green", linewidths=0.8,
               label="Ego (current)")

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_ylabel("y (m)", fontsize=8)
    stale_str = "stale" if frame.get("stale_control") else "fresh"
    ax.set_title(
        f"{planner}  |  {title_label}\n"
        f"frame={frame_idx}  ADE={ade:.3f} m  "
        f"native_dt={native_dt}s  [{stale_str}]",
        fontsize=8.5,
    )
    ax.legend(fontsize=7.5, loc="best")
    ax.grid(True, alpha=0.25)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(debug_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    planners = sorted(p.name for p in debug_dir.iterdir() if p.is_dir())
    if not planners:
        print(f"No planner subdirectories found under {debug_dir}")
        return

    generated = 0
    for planner in planners:
        planner_dir = debug_dir / planner
        for jsonl_path in sorted(planner_dir.glob("*.jsonl")):
            stem = jsonl_path.stem.replace("_frames", "")

            frames = []
            with open(jsonl_path) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            frames.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

            if not frames:
                print(f"  [skip] {planner}/{stem}: empty file")
                continue

            # Full GT trajectory from per-frame ego positions
            gt_traj = np.array([f["ego_world_xy"] for f in frames],
                                dtype=np.float64)

            frame = pick_middle_frame(frames)
            if frame is None:
                print(f"  [skip] {planner}/{stem}: no valid frame found")
                continue

            fig, ax = plt.subplots(figsize=(7, 7))
            make_snap(ax, planner, gt_traj, frame, stem)
            fig.tight_layout()

            out_path = out_dir / f"snap_{planner}_{stem}.png"
            fig.savefig(out_path, dpi=130, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved  {out_path.name}")
            generated += 1

    print(f"\nDone — {generated} snapshot(s) saved to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--debug-dir", default=str(DEFAULT_DEBUG_DIR),
                   help="Path to debug_frames directory")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                   help="Directory to write PNG snapshots")
    args = p.parse_args()
    run(Path(args.debug_dir), Path(args.out_dir))
