#!/usr/bin/env python3
"""Generate thorough high-ADE/FDE diagnostics visualizations for Stage 4 outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "matplotlib is required for high_fde_visualize.py. "
        f"Import error: {exc}"
    )


def _parse_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _as_xy_array(value: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) == 0:
        return None
    if not np.isfinite(arr).all():
        return None
    return arr


def _local_to_world(local_wps: np.ndarray, ego_world_xy: np.ndarray, compass_rad: float) -> np.ndarray:
    # local_wps convention used in adapters: x=right, y=forward
    # convert to CARLA BEV forward/left then rotate to world.
    bev = np.column_stack([local_wps[:, 1], -local_wps[:, 0]])
    theta = float(compass_rad) + math.pi / 2.0
    c = math.cos(theta)
    s = math.sin(theta)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return np.asarray(ego_world_xy, dtype=np.float64).reshape(1, 2) + (rot @ bev.T).T


def _load_hypothesis_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _fit_similarity_transform(pred: np.ndarray, gt: np.ndarray) -> tuple[Optional[np.ndarray], Optional[float], Optional[float]]:
    if pred.shape != gt.shape or pred.ndim != 2 or pred.shape[1] != 2 or len(pred) < 2:
        return None, None, None
    p_mu = np.mean(pred, axis=0)
    g_mu = np.mean(gt, axis=0)
    p_center = pred - p_mu
    g_center = gt - g_mu
    denom = float(np.sum(p_center ** 2))
    if denom <= 1e-12:
        return None, None, None
    h = p_center.T @ g_center
    try:
        u, s, vt = np.linalg.svd(h)
    except np.linalg.LinAlgError:
        return None, None, None
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1
        rot = vt.T @ u.T
    scale = float(np.sum(s) / denom)
    trans = g_mu - scale * (p_mu @ rot.T)
    pred_sim = scale * (pred @ rot.T) + trans.reshape(1, 2)
    rot_deg = float(math.degrees(math.atan2(rot[1, 0], rot[0, 0])))
    return pred_sim, scale, rot_deg


def _apply_affine(points: np.ndarray, affine_matrix_3x2: Any) -> Optional[np.ndarray]:
    if affine_matrix_3x2 is None:
        return None
    try:
        affine = np.asarray(affine_matrix_3x2, dtype=np.float64)
    except Exception:
        return None
    if affine.shape != (3, 2):
        return None
    return np.column_stack([points, np.ones((len(points), 1), dtype=np.float64)]) @ affine


def _timestamp_error_profile(scored_rows: list[dict[str, Any]], affine_matrix_3x2: Any = None) -> Optional[np.ndarray]:
    sums = np.zeros(6, dtype=np.float64)
    counts = np.zeros(6, dtype=np.float64)
    for row in scored_rows:
        pred = _as_xy_array(row.get("predicted_canonical_points"))
        gt = _as_xy_array(row.get("gt_canonical_points"))
        if pred is None or gt is None or len(pred) != 6 or len(gt) != 6:
            continue
        pred_eval = _apply_affine(pred, affine_matrix_3x2) if affine_matrix_3x2 is not None else pred
        if pred_eval is None:
            continue
        valid = np.asarray(row.get("valid_mask", []), dtype=bool)
        gt_valid = np.asarray(row.get("gt_valid_mask", []), dtype=bool)
        if valid.shape != (6,) or gt_valid.shape != (6,):
            continue
        errs = np.linalg.norm(pred_eval - gt, axis=1)
        mask = valid & gt_valid & np.isfinite(errs)
        if not np.any(mask):
            continue
        sums[mask] += errs[mask]
        counts[mask] += 1.0
    if not np.any(counts > 0):
        return None
    out = np.full(6, np.nan, dtype=np.float64)
    nz = counts > 0
    out[nz] = sums[nz] / counts[nz]
    return out


def _pose_lag_profile(planner_hyp: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose_lag = planner_hyp.get("pose_lag_hypothesis", {})
    per_shift = pose_lag.get("per_shift", {}) if isinstance(pose_lag, dict) else {}
    x_vals: list[int] = []
    ade_vals: list[float] = []
    fde_vals: list[float] = []
    for shift_key, shift_payload in per_shift.items():
        if not isinstance(shift_payload, dict):
            continue
        try:
            shift = int(shift_key)
        except Exception:
            continue
        ade = shift_payload.get("mean_ade")
        fde = shift_payload.get("mean_fde")
        if ade is None or fde is None:
            continue
        x_vals.append(shift)
        ade_vals.append(float(ade))
        fde_vals.append(float(fde))
    if not x_vals:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    order = np.argsort(np.asarray(x_vals, dtype=np.int64))
    return (
        np.asarray(x_vals, dtype=np.int64)[order],
        np.asarray(ade_vals, dtype=np.float64)[order],
        np.asarray(fde_vals, dtype=np.float64)[order],
    )


def _error_matrix(scored_rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    if not scored_rows:
        return np.asarray([], dtype=np.int64), np.zeros((0, 6), dtype=np.float64)
    ordered = sorted(scored_rows, key=lambda r: int(r.get("frame_id")))
    frame_ids: list[int] = []
    rows: list[np.ndarray] = []
    for row in ordered:
        fid = int(row.get("frame_id"))
        breakdown = row.get("ade_breakdown_m")
        if not isinstance(breakdown, list) or len(breakdown) != 6:
            continue
        vals = []
        for v in breakdown:
            if v is None:
                vals.append(np.nan)
            else:
                try:
                    vals.append(float(v))
                except Exception:
                    vals.append(np.nan)
        frame_ids.append(fid)
        rows.append(np.asarray(vals, dtype=np.float64))
    if not rows:
        return np.asarray([], dtype=np.int64), np.zeros((0, 6), dtype=np.float64)
    return np.asarray(frame_ids, dtype=np.int64), np.vstack(rows)


def _extra_plots(
    *,
    planner: str,
    scored_rows: list[dict[str, Any]],
    planner_dir: Path,
) -> None:
    planner_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: frame x timestamp error heatmap.
    frame_ids, err_m = _error_matrix(scored_rows)
    if err_m.shape[0] > 0:
        fig, ax = plt.subplots(figsize=(12, 5))
        img = np.ma.masked_invalid(err_m)
        im = ax.imshow(img, aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_title(f"{planner.upper()} canonical error heatmap")
        ax.set_xlabel("canonical timestamp index")
        ax.set_ylabel("scored frame order")
        ax.set_xticks(np.arange(6))
        ax.set_xticklabels(["0.5", "1.0", "1.5", "2.0", "2.5", "3.0"])
        yticks = np.linspace(0, len(frame_ids) - 1, min(8, len(frame_ids)), dtype=int)
        if len(yticks) > 0:
            ax.set_yticks(yticks)
            ax.set_yticklabels([str(frame_ids[i]) for i in yticks])
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
        cbar.set_label("error (m)")
        fig.tight_layout()
        fig.savefig(planner_dir / f"{planner}_error_heatmap.png", dpi=170)
        plt.close(fig)

    # Plot 2: all scored trajectories overlay (corridor-style).
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    n_pred = 0
    n_gt = 0
    for row in scored_rows:
        pred = _as_xy_array(row.get("predicted_canonical_points"))
        gt = _as_xy_array(row.get("gt_canonical_points"))
        if pred is not None and len(pred) == 6:
            ax.plot(pred[:, 0], pred[:, 1], color="#1f77b4", alpha=0.18, linewidth=1.0)
            n_pred += 1
        if gt is not None and len(gt) == 6:
            ax.plot(gt[:, 0], gt[:, 1], color="#2ca02c", alpha=0.18, linewidth=1.0)
            n_gt += 1
    ax.set_title(f"{planner.upper()} scored trajectory overlay (pred/gt)")
    ax.set_xlabel("world x")
    ax.set_ylabel("world y")
    ax.grid(True, alpha=0.25)
    ax.axis("equal")
    ax.text(
        0.01,
        0.01,
        f"pred_trajs={n_pred}  gt_trajs={n_gt}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        family="monospace",
    )
    fig.tight_layout()
    fig.savefig(planner_dir / f"{planner}_corridor_overlay.png", dpi=170)
    plt.close(fig)


def _summary_plot(
    *,
    planner: str,
    scored_rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    planner_hyp: dict[str, Any],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame_ids = np.asarray([int(r.get("frame_id")) for r in scored_rows], dtype=np.int64) if scored_rows else np.asarray([], dtype=np.int64)
    ades = np.asarray([float(r.get("plan_ade")) for r in scored_rows], dtype=np.float64) if scored_rows else np.asarray([], dtype=np.float64)
    fdes = np.asarray([float(r.get("plan_fde")) for r in scored_rows], dtype=np.float64) if scored_rows else np.asarray([], dtype=np.float64)

    reason_counts: dict[str, int] = {}
    for row in all_rows:
        reason = str(row.get("inclusion_reason"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    fig.suptitle(
        f"{planner.upper()} high-FDE diagnostics | variants: baseline + similarity-ub(diagnostic)",
        fontsize=13,
    )

    ax = axes[0, 0]
    if len(frame_ids) > 0:
        ax.plot(frame_ids, ades, marker="o", label="ADE")
        ax.plot(frame_ids, fdes, marker="x", label="FDE")
    ax.set_title("Scored frame ADE/FDE (baseline scoring output)")
    ax.set_xlabel("frame_id")
    ax.set_ylabel("meters")
    ax.grid(True, alpha=0.3)
    if len(frame_ids) > 0:
        ax.legend(loc="best")

    ax = axes[0, 1]
    if reason_counts:
        labels = list(reason_counts.keys())
        values = [reason_counts[k] for k in labels]
        xpos = np.arange(len(labels))
        ax.bar(xpos, values, color="#4472c4")
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("Inclusion/exclusion reasons")
    ax.set_ylabel("count")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 0]
    if len(scored_rows) > 0:
        pred_disp = []
        gt_disp = []
        for row in scored_rows:
            pred = _as_xy_array(row.get("predicted_canonical_points"))
            gt = _as_xy_array(row.get("gt_canonical_points"))
            if pred is None or gt is None or len(pred) != 6 or len(gt) != 6:
                continue
            pred_disp.append(float(np.linalg.norm(pred[-1] - pred[0])))
            gt_disp.append(float(np.linalg.norm(gt[-1] - gt[0])))
        if pred_disp:
            ax.scatter(gt_disp, pred_disp, c="#2ca02c", alpha=0.8)
            mmin = min(min(gt_disp), min(pred_disp))
            mmax = max(max(gt_disp), max(pred_disp))
            ax.plot([mmin, mmax], [mmin, mmax], "k--", alpha=0.5)
    ax.set_title("3.0s displacement bias (pred vs GT)")
    ax.set_xlabel("GT displacement (m)")
    ax.set_ylabel("Pred displacement (m)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    baseline_fde = planner_hyp.get("baseline_mean_fde")
    shift_h = planner_hyp.get("timestamp_shift_hypotheses", {})
    scale_h = planner_hyp.get("displacement_scale_hypothesis", {})
    temporal_h = planner_hyp.get("temporal_scale_hypothesis", {})
    anchor_h = planner_hyp.get("current_anchor_hypothesis", {})
    trans_h = planner_hyp.get("translation_bias_hypothesis", {})
    affine_h = planner_hyp.get("global_affine_hypothesis", {})
    simub_h = planner_hyp.get("similarity_upper_bound_hypothesis", {})
    pose_lag_h = planner_hyp.get("pose_lag_hypothesis", {})
    oracle_h = planner_hyp.get("branch_oracle_hypothesis", {})
    names = [
        "baseline",
        "shift+1",
        "shift-1",
        "disp-scale",
        "time-scale",
        "anchor@t0",
        "translation",
        "global-affine",
        "similarity-ub",
        "pose-lag(best)",
        "branch-oracle",
    ]
    vals = [
        baseline_fde,
        (shift_h.get("timestamp_shift_forward_1step") or {}).get("mean_fde"),
        (shift_h.get("timestamp_shift_backward_1step") or {}).get("mean_fde"),
        scale_h.get("mean_fde"),
        temporal_h.get("best_mean_fde"),
        anchor_h.get("mean_fde"),
        trans_h.get("mean_fde"),
        affine_h.get("mean_fde"),
        simub_h.get("mean_fde"),
        pose_lag_h.get("best_mean_fde"),
        oracle_h.get("mean_oracle_fde"),
    ]
    plot_names = []
    plot_vals = []
    for n, v in zip(names, vals):
        if v is None:
            continue
        plot_names.append(n)
        plot_vals.append(float(v))
    if plot_names:
        xpos = np.arange(len(plot_names))
        ax.bar(xpos, plot_vals, color="#d62728")
        ax.set_xticks(xpos)
        ax.set_xticklabels(plot_names, rotation=30, ha="right")
    ax.set_title("Hypothesis FDE comparison")
    ax.set_ylabel("mean FDE (m)")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[2, 0]
    t = np.asarray([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float64)
    baseline_profile = _timestamp_error_profile(scored_rows, affine_matrix_3x2=None)
    affine_profile = _timestamp_error_profile(scored_rows, affine_matrix_3x2=affine_h.get("affine_matrix_3x2"))
    similarity_profile = None
    # Similarity upper-bound is per-frame, so there is no single global transform.
    # Show it as a flat reference line from the reported mean ADE.
    try:
        simub_mean = simub_h.get("mean_ade")
        if simub_mean is not None:
            similarity_profile = np.full(6, float(simub_mean), dtype=np.float64)
    except Exception:
        similarity_profile = None
    if baseline_profile is not None:
        ax.plot(t, baseline_profile, marker="o", color="#1f77b4", label="baseline")
    if affine_profile is not None:
        ax.plot(t, affine_profile, marker="s", color="#d62728", label="global affine")
    if similarity_profile is not None:
        ax.plot(t, similarity_profile, linestyle="--", color="#9467bd", label="similarity upper-bound (mean)")
    ax.set_title("Per-timestamp mean error (baseline vs variants)")
    ax.set_xlabel("timestamp (s)")
    ax.set_ylabel("meters")
    ax.grid(True, alpha=0.3)
    if baseline_profile is not None or affine_profile is not None or similarity_profile is not None:
        ax.legend(loc="best")

    ax = axes[2, 1]
    lag_x, lag_ade, lag_fde = _pose_lag_profile(planner_hyp)
    if len(lag_x) > 0:
        ax.plot(lag_x, lag_ade, marker="o", color="#2ca02c", label="ADE")
        ax.plot(lag_x, lag_fde, marker="x", color="#ff7f0e", label="FDE")
    ax.axvline(0, color="k", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.set_title("Pose-lag sweep (diagnostic variant)")
    ax.set_xlabel("ego pose frame shift")
    ax.set_ylabel("meters")
    ax.grid(True, alpha=0.3)
    if len(lag_x) > 0:
        ax.legend(loc="best")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _worst_frame_plot(
    *,
    planner: str,
    scored_row: dict[str, Any],
    frame_row: Optional[dict[str, Any]],
    out_path: Path,
) -> None:
    pred = _as_xy_array(scored_row.get("predicted_canonical_points"))
    gt = _as_xy_array(scored_row.get("gt_canonical_points"))
    if pred is None or gt is None or len(pred) != 6 or len(gt) != 6:
        return

    pred_sim, sim_scale, sim_rot_deg = _fit_similarity_transform(pred, gt)
    sim_ade = None
    sim_fde = None
    if pred_sim is not None:
        sim_err = np.linalg.norm(pred_sim - gt, axis=1)
        sim_ade = float(np.mean(sim_err))
        sim_fde = float(sim_err[-1])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 5))
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(pred[:, 0], pred[:, 1], "-o", label="pred canonical", color="#1f77b4")
    ax.plot(gt[:, 0], gt[:, 1], "-s", label="gt canonical", color="#2ca02c")
    for i in range(6):
        ax.plot([pred[i, 0], gt[i, 0]], [pred[i, 1], gt[i, 1]], color="#ff7f0e", alpha=0.5, linewidth=1.0)
    ax.set_title(
        f"{planner.upper()} frame {int(scored_row.get('frame_id'))} | variant=baseline (scored)"
    )
    ax.set_xlabel("world x")
    ax.set_ylabel("world y")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.axis("equal")

    ax = fig.add_subplot(1, 3, 2)
    ax.plot(gt[:, 0], gt[:, 1], "-s", label="gt canonical", color="#2ca02c")
    if pred_sim is not None:
        ax.plot(pred_sim[:, 0], pred_sim[:, 1], "-o", label="similarity-aligned pred", color="#9467bd")
        for i in range(6):
            ax.plot([pred_sim[i, 0], gt[i, 0]], [pred_sim[i, 1], gt[i, 1]], color="#d62728", alpha=0.5, linewidth=1.0)
    ax.set_title("variant=similarity-ub (non-causal diagnostic)")
    ax.set_xlabel("world x")
    ax.set_ylabel("world y")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.axis("equal")

    ax = fig.add_subplot(1, 3, 3)
    ax.axis("off")
    lines = [
        f"frame_id: {int(scored_row.get('frame_id'))}",
        f"ADE: {float(scored_row.get('plan_ade')):.4f} m",
        f"FDE: {float(scored_row.get('plan_fde')):.4f} m",
        f"timestamp_source: {scored_row.get('timestamp_source')}",
        f"inclusion_reason: {scored_row.get('inclusion_reason')}",
    ]
    if sim_ade is not None and sim_fde is not None:
        lines.append(f"sim_upper_ADE: {sim_ade:.4f} m")
        lines.append(f"sim_upper_FDE: {sim_fde:.4f} m")
    if sim_scale is not None:
        lines.append(f"sim_scale: {sim_scale:.5f}")
    if sim_rot_deg is not None:
        lines.append(f"sim_rot_deg: {sim_rot_deg:.3f}")
    lines.append("sim_upper note: per-frame fit; diagnostic only")
    if frame_row is not None and isinstance(frame_row.get("planner_adapter_debug"), dict):
        dbg = frame_row.get("planner_adapter_debug") or {}
        if "selected_command_index" in dbg:
            lines.append(f"selected_command_index: {dbg.get('selected_command_index')}")
        candidates = dbg.get("candidate_local_wps_by_command")
        try:
            c_arr = np.asarray(candidates, dtype=np.float64)
            if c_arr.ndim == 3 and c_arr.shape[-1] == 2:
                lines.append(f"candidate_branches: {int(c_arr.shape[0])}")
                ego = _as_xy_array([frame_row.get("ego_world_xy")])
                compass = frame_row.get("resolved_compass_rad")
                gt_future = _as_xy_array(frame_row.get("gt_future_world"))
                if ego is not None and gt_future is not None and compass is not None:
                    branch_ades = []
                    for b in c_arr:
                        if len(b) != len(gt_future):
                            continue
                        pred_b = _local_to_world(b, ego[0], float(compass))
                        branch_ades.append(float(np.mean(np.linalg.norm(pred_b - gt_future, axis=1))))
                    if branch_ades:
                        lines.append(f"best_branch_ADE: {min(branch_ades):.4f} m")
                        lines.append(f"worst_branch_ADE: {max(branch_ades):.4f} m")
        except Exception:
            pass
    ax.text(0.01, 0.99, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace")

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def run(out_dir: Path, planners: list[str], top_k: int) -> None:
    hypothesis = _load_hypothesis_report(out_dir / "high_fde_hypotheses.json")
    vis_root = out_dir / "high_fde_visuals"
    vis_root.mkdir(parents=True, exist_ok=True)

    for planner in planners:
        scoring_candidates = sorted((out_dir / "scoring_debug" / planner).glob("*_scoring.jsonl"))
        frame_candidates = sorted((out_dir / "debug_frames" / planner).glob("*_frames.jsonl"))
        scoring_file = scoring_candidates[0] if scoring_candidates else None
        frame_file = frame_candidates[0] if frame_candidates else None
        if scoring_file is None:
            continue
        rows = _parse_jsonl(scoring_file)
        frames = _parse_jsonl(frame_file) if frame_file is not None else []
        scored = [r for r in rows if bool(r.get("scored"))]
        scored.sort(key=lambda r: float(r.get("plan_fde", float("-inf"))), reverse=True)
        frame_by_id = {
            int(r.get("frame_idx")): r
            for r in frames
            if r.get("frame_idx") is not None
        }

        planner_dir = vis_root / planner
        planner_dir.mkdir(parents=True, exist_ok=True)
        _summary_plot(
            planner=planner,
            scored_rows=sorted(scored, key=lambda r: int(r.get("frame_id"))),
            all_rows=rows,
            planner_hyp=(hypothesis.get("planners", {}).get(planner, {}) if isinstance(hypothesis, dict) else {}),
            out_path=planner_dir / f"{planner}_summary.png",
        )
        _extra_plots(
            planner=planner,
            scored_rows=sorted(scored, key=lambda r: int(r.get("frame_id"))),
            planner_dir=planner_dir,
        )

        for row in scored[:max(0, int(top_k))]:
            frame_id = int(row.get("frame_id"))
            _worst_frame_plot(
                planner=planner,
                scored_row=row,
                frame_row=frame_by_id.get(frame_id),
                out_path=planner_dir / f"{planner}_frame_{frame_id:03d}_worst.png",
            )
    print(f"Generated high-FDE visual diagnostics in {vis_root}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, help="Stage4 output directory.")
    parser.add_argument("--planners", default="tcp,vad", help="Comma-separated planners.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of worst-FDE frames per planner to plot.")
    args = parser.parse_args()
    planners = [p.strip() for p in str(args.planners).split(",") if p.strip()]
    run(Path(args.out_dir).resolve(), planners=planners, top_k=int(args.top_k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
