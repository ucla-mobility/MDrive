"""Open-loop post-run metrics + visualization.

Consumes a `results/<tag>/<scene>/ego_vehicle_<i>/` directory populated by the
agent in --openloop mode (per-tick `<step>.json` + `<step>_pred.npz`) and emits:

    openloop_metrics.json — ADE / FDE / AP@0.3/0.5/0.7 / CR / counts
    openloop_overview.png — top-down matplotlib summary

GT comes entirely from the scenarioset (REPLAY.xml for ego trajectory; per-actor
XMLs for surrounding-actor positions). No real-world dataset ingest at any step.

Metric definitions (paper-defensible)
-------------------------------------
ADE (anchor-shifted, time-aligned)
    Mean L2 distance between the predicted ego trajectory and the ground-truth
    future ego trajectory at MATCHED TIMESTAMPS, after a one-time anchor shift:
        gt_shifted[k] = gt[k] - gt[0] + pred[0]
        ade_loss[k]   = sqrt((pred_x[k] - gt_shifted_x[k])^2
                             + (pred_y[k] - gt_shifted_y[k])^2)
        ADE = mean(ade_loss[k])
    The shift forces err[0] = 0 by construction, so the metric is invariant
    to the +0.1 s sampling-offset between pred[0] and gt[0] (which has no
    physical meaning here — both are samples one step into the future from
    the same ego pose). All other error sources are preserved: shape
    divergence AND speed-profile mismatch both still register. RiskM
    (RiskM/opencood/utils/eval_utils.py:300-308) uses raw L2 with no shift;
    we deviate because we work in world coords where step-0 offset is
    discretization noise rather than meaningful frame-relative position.

ADE_path (speed-corrected, arc-length-aligned)
    For each predicted waypoint at cumulative arc length s_pred[k], find the
    GT waypoint at the SAME arc length s along the GT trajectory (interpolated)
    and compute L2 distance. Decouples temporal pacing from spatial path
    quality — answers "is the planner driving the right path even if the
    speed is wrong?". This is the right metric for a perception-vs-planner
    ablation where the planner cruises at a fixed speed and the GT replay
    has a bursty real-driver speed profile.

FDE / FDE_path
    L2 distance at the final predicted horizon step under the corresponding
    matching rule.

AP@IoU∈{0.3, 0.5, 0.7}
    BEV-polygon AP via OpenCOOD's calculate_ap, accumulated frame-by-frame by
    the agent's CarlaGTApCollector. We just read the JSON it dumped in /tmp.

CR (Collision Rate, predicted)
    Per-tick: does any predicted ego waypoint, expanded by half-Lincoln-MKZ
    extents, intersect any actor (NPC / walker / cyclist / static) at the same
    future timestamp, expanded by that actor's class extents? CR = fraction of
    eval frames with at least one such intersection.

CLI
---
    python tools/openloop_post_metrics.py \\
        --run-dir   results/.../perception_swap_fcooper/<scene>/image/<runtag>/ego_vehicle_0 \\
        --scenarioset-dir scenarioset/v2xpnp/<scene> \\
        --ap-json   /tmp/perception_swap_ap_fcooper_ego0.json \\
        [--out-dir  <where to write summary + figure>]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# Half-extents in meters (BEV approx).
_HALF_EXTENTS_BY_KIND: Dict[str, Tuple[float, float]] = {
    "ego":     (2.45, 1.00),
    "npc":     (2.30, 0.95),
    "cyclist": (0.80, 0.30),
    "walker":  (0.30, 0.30),
    "static":  (2.30, 0.95),
}

# Codriving / agent JSON convention: ego-local waypoints with
#   forward_carla = -local_y,   left_carla = -local_x
# We invert to get world-frame predicted waypoints.
SCENARIO_DT_S = 0.1
DEFAULT_PRED_DT_S = 0.2


# ───── Parsing ──────────────────────────────────────────────────────────────

@dataclass
class ActorTrack:
    kind: str          # "npc" | "walker" | "cyclist" | "static"
    model: str
    name: str
    times: np.ndarray  # (T,)
    xy:    np.ndarray  # (T, 2)
    yaw:   np.ndarray  # (T,)  degrees


def _parse_route_xml(xml_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (times, xy, yaw_deg) arrays from an actor / ego REPLAY XML."""
    root = ET.parse(str(xml_path)).getroot()
    ts, xs, ys, yaws = [], [], [], []
    for wp in root.iter("waypoint"):
        try:
            ts.append(float(wp.attrib["time"]))
            xs.append(float(wp.attrib["x"]))
            ys.append(float(wp.attrib["y"]))
            yaws.append(float(wp.attrib.get("yaw", 0.0)))
        except (KeyError, ValueError):
            continue
    if not ts:
        raise ValueError(f"no waypoints in {xml_path}")
    return np.asarray(ts), np.asarray(list(zip(xs, ys))), np.asarray(yaws)


def load_scenarioset(scenarioset_dir: Path, ego_slot: int = 0
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[ActorTrack]]:
    """Load ego REPLAY + all actor tracks for a scenarioset scene.

    Returns (ego_times, ego_xy, ego_yaw_deg, actor_tracks).
    """
    ego_xml = scenarioset_dir / f"ucla_v2_custom_ego_vehicle_{ego_slot}_REPLAY.xml"
    if not ego_xml.is_file():
        raise FileNotFoundError(
            f"No REPLAY XML for ego_slot={ego_slot} at {ego_xml}. "
            f"Single-ego scenarioset? Try --ego-slot {1 - ego_slot}."
        )
    ego_times, ego_xy, ego_yaw = _parse_route_xml(ego_xml)

    manifest = json.loads((scenarioset_dir / "actors_manifest.json").read_text())
    actors: List[ActorTrack] = []
    for kind in ("npc", "walker", "cyclist", "static"):
        for entry in manifest.get(kind, []) or []:
            ax = scenarioset_dir / entry["file"]
            if not ax.is_file():
                continue
            try:
                t, xy, yaw = _parse_route_xml(ax)
            except Exception:
                continue
            actors.append(ActorTrack(
                kind=kind, model=str(entry.get("model", "?")),
                name=str(entry.get("name", "?")), times=t, xy=xy, yaw=yaw,
            ))
    return ego_times, ego_xy, ego_yaw, actors


def _interp_xy_yaw(times: np.ndarray, xy: np.ndarray, yaw: np.ndarray, t: float
                   ) -> Optional[Tuple[float, float, float]]:
    """Linear interpolation of (x, y, yaw_deg) at time t.
    Clamps t > times[-1] to times[-1] (returns last GT pose) so a stationary
    REPLAY (ego parked at goal) doesn't make every future step "out of range".
    Returns None only when times is empty or t < times[0]."""
    if len(times) == 0 or t < times[0]:
        return None
    if t > times[-1]:
        return float(xy[-1, 0]), float(xy[-1, 1]), float(yaw[-1])
    i = int(np.searchsorted(times, t))
    if i >= len(times):
        return float(xy[-1, 0]), float(xy[-1, 1]), float(yaw[-1])
    if i == 0 or times[i] == t:
        return float(xy[i, 0]), float(xy[i, 1]), float(yaw[i])
    t0, t1 = times[i - 1], times[i]
    a = (t - t0) / max(1e-9, t1 - t0)
    x = (1 - a) * xy[i - 1, 0] + a * xy[i, 0]
    y = (1 - a) * xy[i - 1, 1] + a * xy[i, 1]
    # yaw circular interp not strictly needed here (used only for footprint
    # orientation); plain linear is fine for the < 0.1 s deltas we use.
    yh = (1 - a) * yaw[i - 1] + a * yaw[i]
    return float(x), float(y), float(yh)


# ───── Geometry helpers ────────────────────────────────────────────────────

def _bev_polygon(cx: float, cy: float, yaw_rad: float, hl: float, hw: float):
    from shapely.geometry import Polygon
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    corners = [
        (cx + c * hl - s * hw, cy + s * hl + c * hw),
        (cx + c * hl + s * hw, cy + s * hl - c * hw),
        (cx - c * hl + s * hw, cy - s * hl - c * hw),
        (cx - c * hl - s * hw, cy - s * hl + c * hw),
    ]
    return Polygon(corners)


def _local_waypoints_to_world(local_wps: np.ndarray, lx: float, ly: float, yaw_rad: float
                              ) -> np.ndarray:
    """Codriving / agent convention: forward = -local_y, left = -local_x."""
    forward = -local_wps[:, 1]
    left    = -local_wps[:, 0]
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    dx = c * forward - s * left
    dy = s * forward + c * left
    return np.column_stack([dx + lx, dy + ly])


def _arc_length(xy: np.ndarray) -> np.ndarray:
    """Cumulative arc length of an (N, 2) polyline. Returns (N,)."""
    if len(xy) < 2:
        return np.zeros(len(xy), dtype=np.float64)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _pathwise_match(pred_xy: np.ndarray, gt_xy: np.ndarray) -> np.ndarray:
    """For each predicted waypoint at cumulative arc length s_p[k], return
    the GT waypoint at the same arc length s along gt_xy (linearly interpolated).

    Anchor-shifts GT to start at pred[0] before matching, so this measures
    only path *shape* divergence, not absolute localisation offset.

    If pred's arc length exceeds GT's total arc length (GT is shorter — e.g.
    GT is decelerating into a stop while pred is cruising), the excess pred
    points are matched to GT's last point. That's the conservative choice:
    the planner is wrong to keep moving when the route is "out of rope".
    """
    if len(pred_xy) < 2 or len(gt_xy) < 2:
        return np.full_like(pred_xy, fill_value=np.nan)
    s_p = _arc_length(pred_xy)
    s_g = _arc_length(gt_xy)
    shifted_gt = gt_xy - gt_xy[0] + pred_xy[0]
    matched = np.zeros_like(pred_xy)
    L_g = float(s_g[-1])
    for k in range(len(pred_xy)):
        s = min(float(s_p[k]), L_g)
        i = int(np.searchsorted(s_g, s, side="right"))
        if i <= 0:
            matched[k] = shifted_gt[0]
        elif i >= len(s_g):
            matched[k] = shifted_gt[-1]
        else:
            denom = max(1e-9, s_g[i] - s_g[i - 1])
            a = (s - s_g[i - 1]) / denom
            matched[k] = (1.0 - a) * shifted_gt[i - 1] + a * shifted_gt[i]
    return matched


def _gt_horizon_polyline(ego_times: np.ndarray, ego_xy: np.ndarray, ego_yaw: np.ndarray,
                         t_anchor: float, horizon_s: float, dt: float = 0.05
                         ) -> np.ndarray:
    """Sample the GT ego trajectory between t_anchor and t_anchor+horizon_s
    at fixed temporal cadence `dt` (smaller than agent's control dt for a
    smooth arc-length parameterisation). Returns (M, 2) world-frame xy.
    Out-of-range samples are clipped to the available range."""
    t_end = min(t_anchor + horizon_s, float(ego_times[-1]))
    if t_end <= t_anchor:
        return ego_xy[-1:].copy()
    t_grid = np.arange(t_anchor, t_end + 1e-9, dt)
    out = np.zeros((len(t_grid), 2), dtype=np.float64)
    for i, t in enumerate(t_grid):
        samp = _interp_xy_yaw(ego_times, ego_xy, ego_yaw, float(t))
        if samp is None:
            out[i] = ego_xy[-1] if t > ego_times[-1] else ego_xy[0]
        else:
            out[i] = (samp[0], samp[1])
    return out


# ───── Metrics ─────────────────────────────────────────────────────────────

def compute_open_loop_metrics(
    *,
    run_dir: Path,
    scenarioset_dir: Path,
    ego_slot: int = 0,
    pred_dt_s: float = DEFAULT_PRED_DT_S,
    ap_json_path: Optional[Path] = None,
) -> Dict:
    """Compute ADE / FDE / CR + read AP/IoU for one open-loop ego run."""
    json_paths = sorted(run_dir.glob("*.json"))
    json_paths = [p for p in json_paths if p.stem.isdigit()]
    if not json_paths:
        return {"warning": f"no <step>.json prediction files in {run_dir}"}

    ego_times, ego_xy, ego_yaw, actors = load_scenarioset(scenarioset_dir, ego_slot=ego_slot)
    print(f"[openloop] loaded ego REPLAY ({len(ego_times)} pts, "
          f"{ego_times[-1]:.1f}s) + {len(actors)} actor tracks "
          f"({sum(1 for a in actors if a.kind=='npc')} npc, "
          f"{sum(1 for a in actors if a.kind=='walker')} walker, "
          f"{sum(1 for a in actors if a.kind=='cyclist')} cyclist, "
          f"{sum(1 for a in actors if a.kind=='static')} static)")

    ego_hl, ego_hw = _HALF_EXTENTS_BY_KIND["ego"]
    n_ade = 0
    sum_ade = 0.0
    sum_fde = 0.0
    n_ade_path = 0
    sum_ade_path = 0.0
    sum_fde_path = 0.0
    n_cr_eval = 0
    n_cr_hit = 0
    cr_breakdown = {k: 0 for k in ("npc", "walker", "cyclist", "static")}
    per_step_rows: List[Dict] = []

    for jp in json_paths:
        try:
            d = json.loads(jp.read_text())
        except Exception:
            continue
        # Prefer pre-transformed world-frame waypoints when the agent
        # provides them (colmdriver / colmdriver_rulebase write
        # 'waypoints_world' from their closed-loop planning_bank because
        # their local-frame convention disagrees with the codriving
        # forward=-y, left=-x, yaw=theta-π/2 transform used below).
        wps_world_raw = d.get("waypoints_world")
        if wps_world_raw is not None:
            try:
                pred_world = np.asarray(wps_world_raw, dtype=np.float64).reshape(-1, 2)
            except Exception:
                pred_world = None
        else:
            pred_world = None
        if pred_world is None or pred_world.size == 0:
            local_wps = np.asarray(d.get("waypoints", []), dtype=np.float64).reshape(-1, 2)
            if local_wps.size == 0:
                continue
            lx = float(d["lidar_pose_x"])
            ly = float(d["lidar_pose_y"])
            yaw_rad = float(d.get("yaw_rad", float(d.get("theta", 0.0)) - math.pi / 2.0))
            pred_world = _local_waypoints_to_world(local_wps, lx, ly, yaw_rad)
        lx = float(d["lidar_pose_x"])
        ly = float(d["lidar_pose_y"])
        n_pred = len(pred_world)

        # Match prediction to a sim-time using the JSON's own ego pose.
        di = np.argmin(np.linalg.norm(ego_xy - np.array([lx, ly]), axis=1))
        t_anchor = float(ego_times[di])

        # GT future at +pred_dt_s, +2*pred_dt_s, …, +n_pred*pred_dt_s.
        gt_xy = []
        valid = []
        for k in range(n_pred):
            t_k = t_anchor + (k + 1) * pred_dt_s
            samp = _interp_xy_yaw(ego_times, ego_xy, ego_yaw, t_k)
            if samp is None:
                gt_xy.append((np.nan, np.nan))
                valid.append(False)
            else:
                gt_xy.append((samp[0], samp[1]))
                valid.append(True)
        gt_xy = np.asarray(gt_xy)
        valid_mask = np.asarray(valid)

        # Anchor-shift GT to pred[0] before measuring. This nullifies the
        # start-offset (a sampling artifact in open-loop: pred and GT are
        # both sampled +pred_dt_s into the future from the same ego pose,
        # so any offset at step 0 has no physical meaning). All other
        # error sources — shape divergence, speed-profile mismatch — are
        # preserved:
        #   gt_shifted[k] = gt[k] - gt[0] + pred[0]
        #   err[k]        = ||pred[k] - gt_shifted[k]||
        #                 = ||(pred[k]-pred[0]) - (gt[k]-gt[0])||
        # NOTE: this differs from RiskM/opencood/utils/eval_utils.py:300
        # which uses raw L2 (no shift). RiskM works in an ego-relative
        # frame where step-0 offset is meaningful; we work in world
        # coordinates where step-0 offset is just discretization noise.
        if not np.any(valid_mask):
            continue
        gt_for_metric = gt_xy.copy()
        if valid_mask[0]:
            gt_for_metric -= gt_for_metric[0] - pred_world[0]
        d2 = np.linalg.norm(pred_world - gt_for_metric, axis=1)
        d2 = d2[valid_mask]
        if d2.size == 0:
            continue
        ade = float(d2.mean())
        fde = float(d2[-1])
        sum_ade += ade
        sum_fde += fde
        n_ade += 1

        # ── Speed-corrected (arc-length-aligned) ADE / FDE ──
        # Build a smooth GT polyline over the prediction horizon and match each
        # predicted waypoint to the GT point at the SAME cumulative arc length.
        # This decouples temporal pacing — the metric measures only whether the
        # planner is following the right *path*, regardless of how fast it
        # cruises along that path.
        horizon_s = (n_pred + 1) * pred_dt_s   # +1 to allow a little overshoot
        gt_poly = _gt_horizon_polyline(ego_times, ego_xy, ego_yaw,
                                       t_anchor=t_anchor, horizon_s=horizon_s,
                                       dt=0.05)
        ade_path = float("nan"); fde_path = float("nan")
        if len(gt_poly) >= 2:
            matched = _pathwise_match(pred_world, gt_poly)
            d_path = np.linalg.norm(pred_world - matched, axis=1)
            if np.all(np.isfinite(d_path)):
                ade_path = float(d_path.mean())
                fde_path = float(d_path[-1])
                sum_ade_path += ade_path
                sum_fde_path += fde_path
                n_ade_path += 1

        # CR: predicted ego trajectory vs surrounding actors over the horizon.
        # Evaluate per-frame; first hit-class wins.
        from shapely.geometry import Polygon  # noqa: F401  (used in helpers)
        pred_yaw = np.arctan2(
            np.diff(pred_world[:, 1], prepend=ly),
            np.diff(pred_world[:, 0], prepend=lx),
        )
        cr_hit_kind = None
        for k in range(n_pred):
            if not valid_mask[k]:
                continue
            t_k = t_anchor + (k + 1) * pred_dt_s
            ego_poly = _bev_polygon(
                pred_world[k, 0], pred_world[k, 1],
                float(pred_yaw[k]), ego_hl, ego_hw,
            )
            for a in actors:
                samp = _interp_xy_yaw(a.times, a.xy, a.yaw, t_k)
                if samp is None:
                    continue
                hl, hw = _HALF_EXTENTS_BY_KIND[a.kind]
                a_poly = _bev_polygon(samp[0], samp[1], math.radians(samp[2]), hl, hw)
                if ego_poly.intersects(a_poly):
                    cr_hit_kind = a.kind
                    break
            if cr_hit_kind is not None:
                break
        n_cr_eval += 1
        if cr_hit_kind is not None:
            n_cr_hit += 1
            cr_breakdown[cr_hit_kind] += 1
        per_step_rows.append({
            "step": int(jp.stem),
            "t_anchor_s": t_anchor,
            "ade_m": ade, "fde_m": fde,
            "ade_path_m": ade_path, "fde_path_m": fde_path,
            "cr_hit": cr_hit_kind,
        })
        # Standardize cross-planner comparisons: only the first tick that
        # produces a valid eval frame counts. Without this, codriving (which
        # saves every ~4 ticks) gets averaged across many ticks while
        # perception_swap (saves only tick 0) is one-shot, making the two
        # families incomparable.
        break

    # Read AP/IoU summary if present.
    ap_summary: Dict = {}
    if ap_json_path is not None and ap_json_path.is_file():
        try:
            ap_d = json.loads(ap_json_path.read_text())
            ap_summary = {
                "n_frames":     ap_d.get("n_frames", 0),
                "cum_n_pred":   ap_d.get("cum_n_pred", 0),
                "cum_n_gt":     ap_d.get("cum_n_gt", 0),
                "cum_n_self_dets": ap_d.get("cum_n_self_dets", 0),
                "ap_per_iou":   {
                    f"{thr:.2f}": (ap_d.get("ap", {}).get(f"ap_{int(thr*100)}", {}) or {}).get("ap")
                    for thr in (0.3, 0.5, 0.7)
                },
                "tp_per_iou":   {
                    f"{thr:.2f}": (ap_d.get("ap", {}).get(f"ap_{int(thr*100)}", {}) or {}).get("n_tp")
                    for thr in (0.3, 0.5, 0.7)
                },
                "fp_per_iou":   {
                    f"{thr:.2f}": (ap_d.get("ap", {}).get(f"ap_{int(thr*100)}", {}) or {}).get("n_fp")
                    for thr in (0.3, 0.5, 0.7)
                },
            }
        except Exception as exc:
            ap_summary = {"warning": f"failed to parse AP file: {exc}"}

    summary = {
        "scenarioset":   str(scenarioset_dir),
        "run_dir":       str(run_dir),
        "ego_slot":      int(ego_slot),
        "n_eval_frames": int(n_ade),
        "ade_m":         (sum_ade / n_ade) if n_ade else float("nan"),
        "fde_m":         (sum_fde / n_ade) if n_ade else float("nan"),
        "ade_path_m":    (sum_ade_path / n_ade_path) if n_ade_path else float("nan"),
        "fde_path_m":    (sum_fde_path / n_ade_path) if n_ade_path else float("nan"),
        "n_eval_frames_path": int(n_ade_path),
        "collision_rate": (n_cr_hit / n_cr_eval) if n_cr_eval else float("nan"),
        "n_cr_eval":     int(n_cr_eval),
        "n_cr_hit":      int(n_cr_hit),
        "cr_by_kind":    cr_breakdown,
        "ap_summary":    ap_summary,
        "per_step":      per_step_rows[:50],   # truncate to keep file small
    }
    return summary


# ───── Visualization ───────────────────────────────────────────────────────

def render_overview(summary: Dict, run_dir: Path, scenarioset_dir: Path,
                    out_path: Path, ego_slot: int = 0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ego_times, ego_xy, _, actors = load_scenarioset(scenarioset_dir, ego_slot=ego_slot)
    fig, ax = plt.subplots(figsize=(11, 9))
    # Ego GT trajectory.
    ax.plot(ego_xy[:, 0], ego_xy[:, 1], color="#888", lw=1.4, alpha=0.6,
            label=f"ego REPLAY  ({ego_times[-1]:.1f} s)")
    ax.scatter([ego_xy[0, 0]], [ego_xy[0, 1]], marker="o", color="#444",
               s=70, zorder=5, label="ego start")
    ax.scatter([ego_xy[-1, 0]], [ego_xy[-1, 1]], marker="*", color="#444",
               s=140, zorder=5, label="ego goal")

    # Actor tracks.
    color_by_kind = {"npc": "#3498db", "walker": "#e67e22",
                     "cyclist": "#9b59b6", "static": "#95a5a6"}
    for a in actors:
        ax.plot(a.xy[:, 0], a.xy[:, 1], color=color_by_kind.get(a.kind, "#888"),
                lw=0.8, alpha=0.5)

    # Per-tick ADE/FDE markers.
    for row in summary.get("per_step", []):
        if row.get("cr_hit") is not None:
            i = int(row["step"]) % len(ego_xy) if ego_xy.size else 0
            ax.scatter([ego_xy[i, 0]], [ego_xy[i, 1]],
                       marker="x", s=70, color="red", zorder=6)

    # Title with the headline numbers.
    ade = summary.get("ade_m", float("nan"))
    fde = summary.get("fde_m", float("nan"))
    ade_p = summary.get("ade_path_m", float("nan"))
    fde_p = summary.get("fde_path_m", float("nan"))
    cr  = summary.get("collision_rate", float("nan"))
    ap_per = summary.get("ap_summary", {}).get("ap_per_iou", {})
    ax.set_title(
        f"OPEN-LOOP overview — {scenarioset_dir.name}\n"
        f"ADE={ade:.2f}m (path {ade_p:.2f})  FDE={fde:.2f}m (path {fde_p:.2f})  CR={cr:.2%}  "
        f"AP@.30={ap_per.get('0.30','-')}  AP@.50={ap_per.get('0.50','-')}  AP@.70={ap_per.get('0.70','-')}  "
        f"n_eval={summary.get('n_eval_frames',0)}",
        fontsize=10,
    )
    ax.set_xlabel("CARLA world x (m)")
    ax.set_ylabel("CARLA world y (m)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ───── CLI ─────────────────────────────────────────────────────────────────

def _maybe(p):
    return Path(p) if p else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="ego_vehicle_<i>/ directory with *.json + *_pred.npz")
    ap.add_argument("--scenarioset-dir", required=True, type=Path,
                    help="scenarioset/v2xpnp/<scene>/  (has REPLAY.xml + actors/)")
    ap.add_argument("--ap-json", type=_maybe, default=None,
                    help="Optional /tmp/perception_swap_ap_<tag>_ego{N}.json to fold AP/IoU into the summary")
    ap.add_argument("--ego-slot", type=int, default=0)
    ap.add_argument("--pred-dt-s", type=float, default=DEFAULT_PRED_DT_S,
                    help="Spacing of predicted waypoints (default 0.2 s, codriving convention)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Where to write summary + figure (default = run_dir)")
    args = ap.parse_args()

    out_dir = args.out_dir or args.run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Single-ego scenariosets only have REPLAY.xml for one slot but the
    # leaderboard still produces stub ego_vehicle_<other> dirs. Exit clean
    # (rc=0) with a one-line skip so the parent walker isn't noisy.
    _replay = (
        args.scenarioset_dir
        / f"ucla_v2_custom_ego_vehicle_{args.ego_slot}_REPLAY.xml"
    )
    if not _replay.is_file():
        print(
            f"[openloop] {args.scenarioset_dir.name}/ego{args.ego_slot}: "
            f"no REPLAY XML ({_replay.name}); skipping (single-ego scenarioset?)"
        )
        return

    summary = compute_open_loop_metrics(
        run_dir=args.run_dir,
        scenarioset_dir=args.scenarioset_dir,
        ego_slot=args.ego_slot,
        pred_dt_s=args.pred_dt_s,
        ap_json_path=args.ap_json,
    )

    summary_path = out_dir / "openloop_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[openloop] wrote {summary_path}")

    if "ade_m" in summary:
        fig_path = out_dir / "openloop_overview.png"
        try:
            render_overview(summary, args.run_dir, args.scenarioset_dir,
                            fig_path, ego_slot=args.ego_slot)
            print(f"[openloop] wrote {fig_path}")
        except Exception as exc:
            print(f"[openloop] WARN: figure render failed: {exc}")

        ap = summary.get("ap_summary", {}).get("ap_per_iou", {})
        print(
            f"[openloop] {args.run_dir.name}  "
            f"ADE={summary.get('ade_m', float('nan')):.2f}m "
            f"(path {summary.get('ade_path_m', float('nan')):.2f}m)  "
            f"FDE={summary.get('fde_m', float('nan')):.2f}m "
            f"(path {summary.get('fde_path_m', float('nan')):.2f}m)  "
            f"CR={summary.get('collision_rate', float('nan')):.3f}  "
            f"AP@.30={ap.get('0.30','-')}  AP@.50={ap.get('0.50','-')}  AP@.70={ap.get('0.70','-')}  "
            f"n_eval={summary.get('n_eval_frames', 0)}"
        )


if __name__ == "__main__":
    main()
