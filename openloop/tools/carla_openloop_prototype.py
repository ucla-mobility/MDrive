"""Open-loop visual debugging prototype: align real V2X-PnP future trajectory
with a CARLA closed-loop scenario and overlay a codriving prediction.

Scope (prototype):
  * One scenario, one frame, one planner (codriving)
  * Strong top-down matplotlib visualization (primary)
  * Top-down CARLA RGB camera capture (secondary; requires live CARLA server)
  * Optional live codriving inference via run_custom_eval (--live-codriving)

Run with the `run_custom_eval_baseline` conda env so carla / yaml / numpy are available:
  /data/miniconda3/envs/run_custom_eval_baseline/bin/python \
      openloop/tools/carla_openloop_prototype.py

For live codriving inference (runs the full leaderboard evaluation fresh):
  ... carla_openloop_prototype.py --live-codriving [--carla-port 2000]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import yaml

# Make the stock CARLA 0.9.12 egg importable from the conda env (per repo README).
_CARLA_EGG_CANDIDATES = [
    "/data2/marco/CoLMDriver/carla912/PythonAPI/carla/dist/carla-0.9.12-py3.7-linux-x86_64.egg",
]
for _egg in _CARLA_EGG_CANDIDATES:
    if os.path.isfile(_egg) and _egg not in sys.path:
        sys.path.append(_egg)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REAL_DATASET_ROOTS = [
    Path("/data/dataset/v2x-real-pnp/train"),
    Path("/data/dataset/v2x-real-pnp/val"),
    Path("/data/dataset/v2x-real-pnp/test"),
    Path("/data/dataset/v2x-real-pnp/v2v_test"),
]
CARLA_SCENARIOSET_ROOT = Path("/data2/marco/CoLMDriver/scenarioset/v2xpnp")
CARLA_ALIGN_JSON = Path("/data2/marco/CoLMDriver/v2xpnp/map/map_offset_carla.json")
CODRIVING_RESULTS_ROOT = Path(
    "/data2/marco/CoLMDriver/results/results_driving_custom/v2xpnp/codriving"
)


# Planner registry — extending the prototype to launch arbitrary leaderboard
# planner aliases (the same names registered in run_custom_eval's PLANNER_SPECS).
# Each entry tells the prototype where to find that planner's per-step prediction
# JSONs after a live-inference run. Keeping codriving as the reference entry
# preserves backward compatibility for every existing call site.
PLANNER_RESULTS_ROOTS = {
    "codriving": CODRIVING_RESULTS_ROOT,
    "perception_swap_fcooper": Path("/data2/marco/CoLMDriver/results/results_driving_custom/v2xpnp/perception_swap_fcooper"),
    "perception_swap_attfuse": Path("/data2/marco/CoLMDriver/results/results_driving_custom/v2xpnp/perception_swap_attfuse"),
    "perception_swap_disco":   Path("/data2/marco/CoLMDriver/results/results_driving_custom/v2xpnp/perception_swap_disco"),
    "perception_swap_cobevt":  Path("/data2/marco/CoLMDriver/results/results_driving_custom/v2xpnp/perception_swap_cobevt"),
}

# Real-ego id under /data/dataset/v2x-real-pnp/.../{ego_id}/ ↔ CARLA ego_N index.
# Confirmed by numerical cross-check (see investigation).
REAL_TO_CARLA_EGO = {1: 0, 2: 1}

SCENARIO_DT_S = 0.1  # 10 Hz real + CARLA replay cadence
CODRIVING_NATIVE_DT_S = 0.2  # adapter constant
DEFAULT_HORIZON_S = 2.0
DEFAULT_SNAP_RADIUS_M = 3.0


# ------------------------------------------------------------------ #
# Coordinate transforms                                              #
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class AlignCfg:
    tx: float
    ty: float
    theta_deg: float
    flip_y: bool

    @classmethod
    def load(cls, path: Path) -> "AlignCfg":
        data = json.loads(path.read_text())
        return cls(
            tx=float(data["tx"]),
            ty=float(data["ty"]),
            theta_deg=float(data.get("theta_deg", 0.0)),
            flip_y=bool(data.get("flip_y", False)),
        )


def real_to_carla_xy(real_xy: np.ndarray, cfg: AlignCfg) -> np.ndarray:
    """Apply the inverse SE(2) used by the v2xpnp pipeline: real(x,y) -> CARLA(x,y).

    Mirrors `invert_se2` in v2xpnp.pipeline.trajectory_ingest_stage_01_types_io.
    """
    real_xy = np.asarray(real_xy, dtype=np.float64).reshape(-1, 2)
    x = real_xy[:, 0] - cfg.tx
    y = real_xy[:, 1] - cfg.ty
    rad = math.radians(-cfg.theta_deg)
    c, s = math.cos(rad), math.sin(rad)
    xr = c * x - s * y
    yr = s * x + c * y
    if cfg.flip_y:
        yr = -yr
    return np.column_stack([xr, yr])


def real_yaw_to_carla_yaw(real_yaw_deg: float, cfg: AlignCfg) -> float:
    """CARLA yaw = -real_yaw when flip_y and theta=0 (standard case)."""
    yaw = real_yaw_deg - cfg.theta_deg
    if cfg.flip_y:
        yaw = -yaw
    return yaw


# ------------------------------------------------------------------ #
# Real dataset loading                                               #
# ------------------------------------------------------------------ #


def find_real_scenario_dir(scenario: str) -> Path:
    for root in REAL_DATASET_ROOTS:
        p = root / scenario
        if p.is_dir():
            return p
    raise FileNotFoundError(f"Real scenario not found under any split: {scenario}")


def list_real_frames(real_scenario_dir: Path, real_ego: int) -> List[int]:
    actor_dir = real_scenario_dir / str(real_ego)
    frames = []
    for p in actor_dir.glob("*.yaml"):
        try:
            frames.append(int(p.stem))
        except ValueError:
            pass
    return sorted(frames)


def load_real_frame_yaml(real_scenario_dir: Path, real_ego: int, frame_idx: int) -> dict:
    p = real_scenario_dir / str(real_ego) / f"{frame_idx:06d}.yaml"
    return yaml.safe_load(p.read_text())


def load_real_surrounding_trajectories(
    real_scenario_dir: Path,
    real_ego: int,
    frame_indices: Sequence[int],
    align_cfg: "AlignCfg",
) -> Tuple[List[int], List[np.ndarray]]:
    """For every surrounding actor observed at the query frames, return its future
    CARLA-frame trajectory.

    Returns (ids, trajectories) where trajectories[k] is an [N_valid, 2] array of
    frames at which that actor was observed (variable length per actor).
    """
    per_actor: dict = {}  # actor_id -> list[(frame_idx, carla_xy)]
    for fi in frame_indices:
        p = real_scenario_dir / str(real_ego) / f"{fi:06d}.yaml"
        if not p.exists():
            continue
        try:
            d = yaml.safe_load(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        vehicles = d.get("vehicles") or {}
        for aid, meta in vehicles.items():
            loc = meta.get("location")
            if not loc or len(loc) < 2:
                continue
            try:
                aid_i = int(aid)
            except (TypeError, ValueError):
                continue
            xy_real = np.array([float(loc[0]), float(loc[1])], dtype=np.float64)
            xy_carla = real_to_carla_xy(xy_real[None, :], align_cfg).squeeze()
            per_actor.setdefault(aid_i, []).append((fi, xy_carla))
    ids: List[int] = []
    trajs: List[np.ndarray] = []
    for aid, pairs in per_actor.items():
        if len(pairs) < 2:
            continue
        pairs.sort(key=lambda p: p[0])
        ids.append(aid)
        trajs.append(np.array([xy for _, xy in pairs], dtype=np.float64))
    return ids, trajs


def load_real_ego_trajectory(
    real_scenario_dir: Path,
    real_ego: int,
    frame_indices: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (xy [N,2], mask [N]) of lidar_pose[:2] for the requested frame indices."""
    out = np.full((len(frame_indices), 2), np.nan, dtype=np.float64)
    mask = np.zeros(len(frame_indices), dtype=bool)
    for i, fi in enumerate(frame_indices):
        p = real_scenario_dir / str(real_ego) / f"{fi:06d}.yaml"
        if not p.exists():
            continue
        try:
            d = yaml.safe_load(p.read_text())
            pose = d.get("lidar_pose")
            if pose and len(pose) >= 2:
                out[i, 0] = float(pose[0])
                out[i, 1] = float(pose[1])
                mask[i] = True
        except Exception:  # noqa: BLE001
            pass
    return out, mask


def pick_start_frame_by_motion(
    real_scenario_dir: Path,
    real_ego: int,
    min_speed_mps: float = 2.0,
) -> int:
    """Pick the first frame where the ego is actually moving (prototype convenience).

    The dataset's ego_speed field is always 0 (data-recording bug), so speed is
    derived from the pose delta between consecutive frames instead.
    """
    frames = list_real_frames(real_scenario_dir, real_ego)
    if not frames:
        raise FileNotFoundError(f"No frames for ego {real_ego}")
    prev_pose: Optional[list] = None
    for fi in frames[:-5]:
        try:
            d = yaml.safe_load(
                (real_scenario_dir / str(real_ego) / f"{fi:06d}.yaml").read_text()
            )
            pose = d["lidar_pose"]
            if prev_pose is not None:
                dx = pose[0] - prev_pose[0]
                dy = pose[1] - prev_pose[1]
                speed = math.sqrt(dx * dx + dy * dy) / SCENARIO_DT_S
                if speed >= min_speed_mps:
                    return fi
            prev_pose = pose
        except Exception:  # noqa: BLE001
            prev_pose = None
            continue
    return frames[len(frames) // 3]  # fallback: one-third in


def joint_pick_frame_and_codriving_json(
    real_scenario_dir: Path,
    real_ego: int,
    run_dir: Path,
    align_cfg: "AlignCfg",
    min_speed_mps: float = 2.0,
    horizon_steps: int = 30,
) -> Tuple[int, Path, float]:
    """Find the (real_frame_idx, codriving_json) pair whose ego poses coincide best.

    Closed-loop codriving drove its own trajectory, so the ego position in a saved JSON
    generally does not match frame_idx * 0.1s on the replay timeline. For open-loop
    visualization we want both anchors at the *same geometric pose*, so iterate all
    JSONs and, for each, find the real frame minimising ||codriving_lidar - real_lidar||
    (with the real ego actually moving and with enough future frames remaining).

    Returns (best_frame_idx, best_json_path, best_distance_m).
    """
    real_frames = list_real_frames(real_scenario_dir, real_ego)
    if not real_frames:
        raise FileNotFoundError(f"No frames for ego {real_ego}")
    # Pre-load all real ego poses, transformed to CARLA.
    all_real_xy_real, all_mask = load_real_ego_trajectory(
        real_scenario_dir, real_ego, real_frames
    )
    all_real_xy_carla = real_to_carla_xy(all_real_xy_real, align_cfg)
    # Ego_speed in this dataset is not reliably populated, so derive motion directly
    # from position deltas: a frame is "moving" if mean displacement over a 1 s window
    # exceeds min_speed_mps * dt.
    window = max(2, int(round(1.0 / SCENARIO_DT_S)))
    disp = np.full(len(real_frames), 0.0, dtype=np.float64)
    for i in range(window, len(real_frames)):
        if all_mask[i] and all_mask[i - window]:
            disp[i] = float(np.linalg.norm(all_real_xy_carla[i] - all_real_xy_carla[i - window]))
    speeds = disp / (window * SCENARIO_DT_S)

    best: Optional[Tuple[int, Path, float]] = None
    for jp in sorted(run_dir.glob("*.json")):
        try:
            d = json.loads(jp.read_text())
        except Exception:  # noqa: BLE001
            continue
        if "lidar_pose_x" not in d or "waypoints" not in d:
            continue
        q = np.array([float(d["lidar_pose_x"]), float(d["lidar_pose_y"])])
        dists = np.linalg.norm(all_real_xy_carla - q[None, :], axis=1)
        # Constrain to moving frames with enough future horizon left.
        valid = (speeds >= min_speed_mps) & all_mask
        max_idx = len(real_frames) - horizon_steps - 1
        if max_idx > 0:
            valid[max_idx:] = False
        if not np.any(valid):
            continue
        dists_masked = np.where(valid, dists, np.inf)
        i_best = int(np.argmin(dists_masked))
        best_d = float(dists_masked[i_best])
        if best is None or best_d < best[2]:
            best = (real_frames[i_best], jp, best_d)
    if best is None:
        raise RuntimeError("Could not find any matching codriving JSON / real frame pair.")
    return best


# ------------------------------------------------------------------ #
# CARLA scenario XML parsing                                         #
# ------------------------------------------------------------------ #


@dataclass
class RoutePolyline:
    xy: np.ndarray  # [N,2] CARLA world
    times: Optional[np.ndarray] = None  # [N] seconds from start
    yaw_deg: Optional[np.ndarray] = None  # [N]


def parse_carla_route_xml(xml_path: Path) -> RoutePolyline:
    root = ET.parse(str(xml_path)).getroot()
    xs, ys, ts, yaws = [], [], [], []
    for wp in root.iter("waypoint"):
        try:
            xs.append(float(wp.attrib["x"]))
            ys.append(float(wp.attrib["y"]))
            ts.append(float(wp.attrib["time"]) if "time" in wp.attrib else math.nan)
            yaws.append(float(wp.attrib.get("yaw", 0.0)))
        except (KeyError, ValueError):
            continue
    if not xs:
        raise ValueError(f"No waypoints in {xml_path}")
    xy = np.column_stack([xs, ys])
    t_arr = np.array(ts) if not any(math.isnan(t) for t in ts) else None
    return RoutePolyline(xy=xy, times=t_arr, yaw_deg=np.array(yaws))


# ------------------------------------------------------------------ #
# CoDriving prediction loading                                       #
# ------------------------------------------------------------------ #


@dataclass
class CodrivingPrediction:
    source_json: Path
    lidar_xy: np.ndarray  # [2] world
    theta: float  # compass radians
    waypoints_local: np.ndarray  # [10, 2] ego-local as saved
    waypoints_world: np.ndarray  # [10, 2] converted to CARLA world
    step_counter: int  # filename index


def find_planner_run_dir(planner: str, scenario: str, ego_slot: int = 0) -> Optional[Path]:
    """Find a planner's ``image/<runtag>/ego_vehicle_<slot>/`` directory with predictions.

    Generalized form of ``find_codriving_run_dir``. Looks up the planner's results
    root from PLANNER_RESULTS_ROOTS so we can support codriving, perception_swap_*
    and any future leaderboard alias with a single call site.
    """
    root = PLANNER_RESULTS_ROOTS.get(planner)
    if root is None:
        # Fall back to a sensible default path so an unregistered planner can
        # still be probed without a code change (the same convention the rest
        # of the repo uses for results layout).
        root = Path(f"/data2/marco/CoLMDriver/results/results_driving_custom/v2xpnp/{planner}")
    base = root / scenario / "image"
    if not base.is_dir():
        return None
    candidates = sorted(
        (p for p in base.iterdir() if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    for run in candidates:
        ego_dir = run / f"ego_vehicle_{ego_slot}"
        if ego_dir.is_dir() and any(ego_dir.glob("*.json")):
            return ego_dir
    return None


def find_codriving_run_dir(scenario: str, ego_slot: int = 0) -> Optional[Path]:
    """Backward-compatible wrapper for legacy codriving call sites."""
    return find_planner_run_dir("codriving", scenario, ego_slot)


def local_waypoints_to_world(
    local_wps: np.ndarray,
    ego_xy: Tuple[float, float],
    yaw_rad: float,
) -> np.ndarray:
    """Convert codriving ego-local waypoints to CARLA world XY.

    CoDriving's saved convention (per ``_adapt_codriving``):
        carla-body forward = -local_y
        carla-body left    = -local_x

    We rotate (forward, left) by the ego's CARLA world yaw to get a world offset,
    then translate by the ego's world XY. We take yaw directly rather than trusting
    the json's ``theta`` field, which is not a plain compass angle in this dataset
    (empirically ``theta ≈ yaw_rad + π/2``, so ``theta + π/2`` used by the adapter
    points 180° backward).
    """
    local_wps = np.asarray(local_wps, dtype=np.float64).reshape(-1, 2)
    forward = -local_wps[:, 1]
    left = -local_wps[:, 0]
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    # Rotation: world = R(yaw) · (forward, left)^T
    #   where R(yaw) = [[cos, -sin], [sin, cos]] when mapping (forward, left) -> (x, y).
    world_dx = c * forward - s * left
    world_dy = s * forward + c * left
    world_offsets = np.column_stack([world_dx, world_dy])
    return world_offsets + np.array([ego_xy[0], ego_xy[1]], dtype=np.float64)


def _load_single_codriving_json(jp: Path) -> CodrivingPrediction:
    d = json.loads(jp.read_text())
    lx = float(d["lidar_pose_x"])
    ly = float(d["lidar_pose_y"])
    theta = float(d.get("theta", 0.0))
    local_wps = np.array(d["waypoints"], dtype=np.float64).reshape(-1, 2)
    yaw_rad = theta - math.pi / 2.0  # empirical: json theta ≈ yaw_rad + pi/2
    world_wps = local_waypoints_to_world(local_wps, (lx, ly), yaw_rad)
    try:
        step = int(jp.stem)
    except ValueError:
        step = -1
    return CodrivingPrediction(
        source_json=jp,
        lidar_xy=np.array([lx, ly]),
        theta=theta,
        waypoints_local=local_wps,
        waypoints_world=world_wps,
        step_counter=step,
    )


def load_codriving_prediction_closest_to(
    run_dir: Path,
    target_xy: np.ndarray,
) -> CodrivingPrediction:
    """Load the codriving json whose recorded lidar_pose is closest to target_xy."""
    best: Optional[CodrivingPrediction] = None
    best_d2 = math.inf
    for jp in sorted(run_dir.glob("*.json")):
        try:
            d = json.loads(jp.read_text())
        except Exception:  # noqa: BLE001
            continue
        if "waypoints" not in d or "lidar_pose_x" not in d:
            continue
        lx = float(d["lidar_pose_x"])
        ly = float(d["lidar_pose_y"])
        d2 = (lx - float(target_xy[0])) ** 2 + (ly - float(target_xy[1])) ** 2
        if d2 >= best_d2:
            continue
        local_wps = np.array(d["waypoints"], dtype=np.float64).reshape(-1, 2)
        theta = float(d.get("theta", 0.0))
        yaw_rad = theta - math.pi / 2.0  # empirical: json theta ≈ yaw_rad + pi/2
        world_wps = local_waypoints_to_world(local_wps, (lx, ly), yaw_rad)
        try:
            step = int(jp.stem)
        except ValueError:
            step = -1
        best = CodrivingPrediction(
            source_json=jp,
            lidar_xy=np.array([lx, ly]),
            theta=theta,
            waypoints_local=local_wps,
            waypoints_world=world_wps,
            step_counter=step,
        )
        best_d2 = d2
    if best is None:
        raise FileNotFoundError(f"No usable codriving predictions in {run_dir}")
    return best


# ------------------------------------------------------------------ #
# Alignment algorithm: soft project onto CARLA route                 #
# ------------------------------------------------------------------ #


def _project_point_to_polyline(
    p: np.ndarray,
    polyline: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Return (closest_point_on_polyline, distance)."""
    best_q = polyline[0]
    best_d = math.inf
    for i in range(len(polyline) - 1):
        a = polyline[i]
        b = polyline[i + 1]
        ab = b - a
        denom = float(ab @ ab)
        if denom < 1e-9:
            q = a
        else:
            t = float((p - a) @ ab / denom)
            t = max(0.0, min(1.0, t))
            q = a + t * ab
        d = float(np.hypot(p[0] - q[0], p[1] - q[1]))
        if d < best_d:
            best_d = d
            best_q = q
    return best_q, best_d


def align_gt_to_route(
    gt_carla: np.ndarray,
    route: np.ndarray,
    snap_radius_m: float = DEFAULT_SNAP_RADIUS_M,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Soft-project each GT point onto the CARLA route polyline.

    Returns
    -------
    aligned : [T, 2] blended trajectory
    projections : [T, 2] closest points on route
    blend_weights : [T] in [0, 1] (1 = fully snapped, 0 = unchanged)
    """
    T = gt_carla.shape[0]
    aligned = np.empty_like(gt_carla)
    projections = np.empty_like(gt_carla)
    weights = np.zeros(T, dtype=np.float64)
    for i, p in enumerate(gt_carla):
        q, d = _project_point_to_polyline(p, route)
        projections[i] = q
        w = max(0.0, min(1.0, 1.0 - d / snap_radius_m))
        weights[i] = w
        aligned[i] = (1.0 - w) * p + w * q
    return aligned, projections, weights


def ade(pred: np.ndarray, gt: np.ndarray) -> float:
    n = min(len(pred), len(gt))
    if n == 0:
        return float("nan")
    d = np.sqrt(np.sum((pred[:n] - gt[:n]) ** 2, axis=1))
    return float(d.mean())


# ------------------------------------------------------------------ #
# Matplotlib rendering                                               #
# ------------------------------------------------------------------ #


def render_alignment_detail(
    out_path: Path,
    *,
    scenario: str,
    frame_idx: int,
    ego_xy: np.ndarray,
    route_full: np.ndarray,
    gt_raw: np.ndarray,
    gt_aligned: np.ndarray,
    projections: np.ndarray,
    blend_weights: np.ndarray,
    zoom_radius_m: float = 25.0,
) -> None:
    """Dedicated figure showing how the raw GT is soft-projected onto the global route."""
    import matplotlib.cm as _cm  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(
        route_full[:, 0], route_full[:, 1],
        color="#aaaaaa", linewidth=1.5, alpha=0.7, label="Global route",
    )
    # Projection lines coloured by blend weight (RdYlGn: red=not snapped, green=fully snapped).
    cmap = _cm.RdYlGn
    for i in range(len(gt_raw)):
        c = cmap(float(blend_weights[i]))
        ax.plot(
            [gt_raw[i, 0], projections[i, 0]],
            [gt_raw[i, 1], projections[i, 1]],
            color=c, linewidth=1.0, alpha=0.7,
        )
    sc = ax.scatter(
        gt_raw[:, 0], gt_raw[:, 1],
        c=blend_weights, cmap="RdYlGn", vmin=0.0, vmax=1.0,
        s=55, zorder=5, label="Raw GT (colour = blend weight)",
    )
    ax.plot(
        gt_aligned[:, 0], gt_aligned[:, 1],
        color="#f1c40f", linewidth=2.0, marker="o", markersize=4,
        label="Aligned GT", zorder=6,
    )
    ax.scatter([ego_xy[0]], [ego_xy[1]], color="#27ae60", marker="*", s=260, zorder=10, label="Ego")
    plt.colorbar(sc, ax=ax, label="Blend weight  (1 = fully snapped to route,  0 = unchanged)")
    ax.set_xlim(ego_xy[0] - zoom_radius_m, ego_xy[0] + zoom_radius_m)
    ax.set_ylim(ego_xy[1] - zoom_radius_m, ego_xy[1] + zoom_radius_m)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("CARLA world x (m)")
    ax.set_ylabel("CARLA world y (m)")
    ax.set_title(f"GT alignment to global route — {scenario}  frame={frame_idx}")
    ax.legend(loc="lower right", fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def render_topdown(
    out_path: Path,
    *,
    scenario: str,
    frame_idx: int,
    horizon_s: float,
    ego_xy: np.ndarray,
    route_full: np.ndarray,
    route_goal: np.ndarray,
    gt_raw: np.ndarray,
    gt_aligned: np.ndarray,
    projections: np.ndarray,
    codriving_world: np.ndarray,
    codriving_lidar_xy: np.ndarray,
    ade_raw: float,
    ade_aligned: float,
    ade_codriving: float,
    zoom_radius_m: Optional[float] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.plot(
        route_full[:, 0], route_full[:, 1],
        color="#888888", linewidth=1.2, alpha=0.6, label="CARLA ego route (REPLAY.xml)",
    )
    ax.scatter(
        route_goal[:, 0], route_goal[:, 1],
        color="#555555", marker="s", s=55, label="CARLA goal plan",
    )
    ax.plot(
        gt_raw[:, 0], gt_raw[:, 1],
        color="#e67e22", linewidth=2.2, marker="o", markersize=4,
        label=f"Real GT transformed  (ADE vs route={ade_raw:.2f}m)",
    )
    ax.plot(
        gt_aligned[:, 0], gt_aligned[:, 1],
        color="#f1c40f", linewidth=2.2, marker="o", markersize=4,
        label=f"GT aligned to route   (ADE vs route={ade_aligned:.2f}m)",
    )
    # Thin lines showing the soft-projection offsets
    for p, q in zip(gt_raw, projections):
        ax.plot([p[0], q[0]], [p[1], q[1]], color="#f1c40f", linewidth=0.5, alpha=0.5)
    ax.plot(
        codriving_world[:, 0], codriving_world[:, 1],
        color="#e74c3c", linewidth=2.2, marker="^", markersize=5,
        label=f"codriving prediction  (ADE vs aligned GT={ade_codriving:.2f}m)",
    )
    ax.scatter([ego_xy[0]], [ego_xy[1]], color="#27ae60", marker="*", s=260, zorder=10, label="Ego (this frame)")
    ax.scatter(
        [codriving_lidar_xy[0]], [codriving_lidar_xy[1]],
        facecolor="none", edgecolor="#c0392b", marker="o", s=220, linewidth=2.0,
        label="Ego (codriving json)",
    )

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("CARLA world x (m)")
    ax.set_ylabel("CARLA world y (m)")
    title = (
        f"Open-loop prototype: {scenario}\n"
        f"frame={frame_idx}  horizon={horizon_s:.1f}s  "
        f"(real→CARLA transform verified)"
    )
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)

    if zoom_radius_m is not None:
        ax.set_xlim(ego_xy[0] - zoom_radius_m, ego_xy[0] + zoom_radius_m)
        ax.set_ylim(ego_xy[1] - zoom_radius_m, ego_xy[1] + zoom_radius_m)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ #
# Live CARLA: spawn the scene at t=target, capture top-down PNGs     #
# ------------------------------------------------------------------ #


@dataclass
class ActorSpawnSpec:
    role: str  # "ego" | "npc" | "walker" | "static"
    name: str
    model: str
    loc_xyz: Tuple[float, float, float]
    rot_prp: Tuple[float, float, float]  # pitch, roll, yaw (degrees)


def _pose_at_time(xml_path: Path, t_target_s: float) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """Return the (loc, rot) whose ``time`` attribute is closest to ``t_target_s``.

    For routes without ``time`` (sparse goal plans), returns the first waypoint.
    """
    try:
        root = ET.parse(str(xml_path)).getroot()
    except Exception:  # noqa: BLE001
        return None
    best_wp = None
    best_dt = math.inf
    first_wp = None
    for wp in root.iter("waypoint"):
        if first_wp is None:
            first_wp = wp
        if "time" not in wp.attrib:
            continue
        try:
            dt = abs(float(wp.attrib["time"]) - t_target_s)
        except ValueError:
            continue
        if dt < best_dt:
            best_dt = dt
            best_wp = wp
    wp = best_wp if best_wp is not None else first_wp
    if wp is None:
        return None
    try:
        loc = (
            float(wp.attrib["x"]),
            float(wp.attrib["y"]),
            float(wp.attrib.get("z", 0.0)),
        )
        rot = (
            float(wp.attrib.get("pitch", 0.0)),
            float(wp.attrib.get("roll", 0.0)),
            float(wp.attrib.get("yaw", 0.0)),
        )
    except (KeyError, ValueError):
        return None
    return loc, rot


def build_actor_spawn_list(
    carla_scenario_dir: Path,
    t_target_s: float,
) -> List[ActorSpawnSpec]:
    """Read actors_manifest.json and resolve every actor's pose at ``t_target_s``."""
    manifest = json.loads((carla_scenario_dir / "actors_manifest.json").read_text())
    out: List[ActorSpawnSpec] = []
    for role in ("ego", "npc", "walker", "static"):
        for entry in manifest.get(role, []):
            # Egos: prefer the REPLAY.xml (has timing); fallback to goal plan.
            rel = entry["file"]
            xml_path = carla_scenario_dir / rel
            if role == "ego":
                replay_path = xml_path.with_name(xml_path.stem + "_REPLAY.xml")
                if replay_path.exists():
                    xml_path = replay_path
            pose = _pose_at_time(xml_path, t_target_s)
            if pose is None:
                print(f"[WARN] no pose found in {xml_path.name}; skipping")
                continue
            loc, rot = pose
            out.append(
                ActorSpawnSpec(
                    role=role,
                    name=str(entry.get("name", "?")),
                    model=str(entry.get("model", "vehicle.lincoln.mkz_2020")),
                    loc_xyz=loc,
                    rot_prp=rot,
                )
            )
    return out


class CarlaSceneSession:
    """Context manager that spawns a scenario in live CARLA at t=target, captures
    top-down images, pushes debug draws, then cleans up.
    """

    EXPECTED_TOWN = "ucla_v2"

    def __init__(self, host: str = "localhost", port: int = 2000, timeout_s: float = 20.0):
        import carla  # type: ignore  # noqa: PLC0415
        self._carla = carla
        self._host = host
        self._port = port
        self._timeout = timeout_s
        self.client = None
        self.world = None
        self.bp_lib = None
        self._spawned: List[object] = []  # actors + sensors
        self._original_settings = None
        self._camera_frame: Optional[np.ndarray] = None
        self._front_camera_frame: Optional[np.ndarray] = None

    def __enter__(self) -> "CarlaSceneSession":
        self.client = self._carla.Client(self._host, self._port)
        self.client.set_timeout(max(self._timeout, 60.0))
        # Always (re)load ucla_v2 so we start from a clean world: wipes stale debug
        # primitives from previous runs (0.9.12 has no debug-clear API) and removes
        # lingering vehicle/walker actors from earlier scenarios.
        print(f"[CARLA] loading '{self.EXPECTED_TOWN}' for clean session...")
        self.world = self.client.load_world(self.EXPECTED_TOWN)
        town = self.world.get_map().name.lower()
        if self.EXPECTED_TOWN not in town:
            raise RuntimeError(
                f"failed to load {self.EXPECTED_TOWN}; current map = '{town}'"
            )
        print(f"[CARLA] loaded map: {self.world.get_map().name}")
        self.bp_lib = self.world.get_blueprint_library()
        # Freeze weather to clear noon so the sun position doesn't drift and bake
        # lens flare into successive frames while we tick.
        try:
            self.world.set_weather(self._carla.WeatherParameters.ClearNoon)
        except Exception:  # noqa: BLE001
            pass
        # Switch to sync mode for deterministic ticks / camera capture.
        self._original_settings = self.world.get_settings()
        new_settings = self.world.get_settings()
        new_settings.synchronous_mode = True
        new_settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(new_settings)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        carla = self._carla
        # Destroy sensors first (children), then actors.
        for act in list(self._spawned)[::-1]:
            try:
                if hasattr(act, "stop"):
                    try:
                        act.stop()
                    except Exception:  # noqa: BLE001
                        pass
                act.destroy()
            except Exception:  # noqa: BLE001
                pass
        self._spawned.clear()
        if self._original_settings is not None and self.world is not None:
            try:
                self.world.apply_settings(self._original_settings)
            except Exception:  # noqa: BLE001
                pass

    # -- scene population ---------------------------------------------------- #

    def destroy_existing_world_actors(self) -> int:
        """Destroy every vehicle + walker currently in the world (not ones we own)."""
        carla = self._carla
        victims = [
            a for a in self.world.get_actors()
            if a.type_id.startswith("vehicle.") or a.type_id.startswith("walker.")
        ]
        batch = [carla.command.DestroyActor(a.id) for a in victims]
        if batch:
            self.client.apply_batch_sync(batch, True)
        self.world.tick()
        return len(victims)

    def _resolve_blueprint(self, model_id: str, role: str):
        matches = self.bp_lib.filter(model_id)
        if matches:
            return matches[0]
        # Fallbacks per role.
        if role == "walker":
            fb = self.bp_lib.filter("walker.pedestrian*")
        else:
            fb = self.bp_lib.filter("vehicle.lincoln.mkz_2020")
        return fb[0] if fb else None

    def spawn_scene(self, specs: Sequence[ActorSpawnSpec]) -> Tuple[int, int]:
        """Spawn every actor at its resolved pose; return (ok, skipped)."""
        carla = self._carla
        ok = 0
        skip = 0
        for s in specs:
            bp = self._resolve_blueprint(s.model, s.role)
            if bp is None:
                skip += 1
                continue
            if bp.has_attribute("role_name"):
                try:
                    bp.set_attribute("role_name", f"{s.role}_{s.name}")
                except Exception:  # noqa: BLE001
                    pass
            tf = carla.Transform(
                carla.Location(x=s.loc_xyz[0], y=s.loc_xyz[1], z=s.loc_xyz[2] + 0.10),
                carla.Rotation(pitch=s.rot_prp[0], roll=s.rot_prp[1], yaw=s.rot_prp[2]),
            )
            try:
                actor = self.world.try_spawn_actor(bp, tf)
            except Exception:  # noqa: BLE001
                actor = None
            if actor is None:
                # Retry with a small z bump in case of ground collision.
                tf.location.z += 0.5
                try:
                    actor = self.world.try_spawn_actor(bp, tf)
                except Exception:  # noqa: BLE001
                    actor = None
            if actor is None:
                skip += 1
                continue
            try:
                actor.set_simulate_physics(False)
            except Exception:  # noqa: BLE001
                pass
            self._spawned.append(actor)
            ok += 1
        # Tick once to settle.
        self.world.tick()
        return ok, skip

    # -- camera + capture ---------------------------------------------------- #

    def spawn_topdown_camera(
        self,
        focus_xy: Tuple[float, float],
        altitude_m: float = 40.0,
        width: int = 1280,
        height: int = 960,
        fov_deg: float = 90.0,
    ):
        carla = self._carla
        bp = self.bp_lib.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(int(width)))
        bp.set_attribute("image_size_y", str(int(height)))
        bp.set_attribute("fov", str(float(fov_deg)))
        bp.set_attribute("sensor_tick", "0.0")
        # Disable post-process FX (lens flare, bloom, tone mapping drift) so the
        # image looks the same regardless of how many ticks we've run.
        # Keep tone mapping (colorful output) but disable auto-exposure/bloom/motion
        # blur so the image doesn't drift brighter as we tick.
        for attr, val in (
            ("motion_blur_intensity", "0.0"),
            ("motion_blur_max_distortion", "0.0"),
            ("motion_blur_min_object_screen_size", "0.0"),
            ("lens_flare_intensity", "0.0"),
            ("bloom_intensity", "0.0"),
            ("exposure_mode", "manual"),
            ("exposure_compensation", "0.0"),
            ("shutter_speed", "200.0"),
            ("iso", "100.0"),
            ("gamma", "2.2"),
        ):
            if bp.has_attribute(attr):
                try:
                    bp.set_attribute(attr, val)
                except Exception:  # noqa: BLE001
                    pass
        tf = carla.Transform(
            carla.Location(x=float(focus_xy[0]), y=float(focus_xy[1]), z=float(altitude_m)),
            carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
        )
        cam = self.world.spawn_actor(bp, tf)
        self._spawned.append(cam)

        def _cb(img):
            arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
            # BGRA -> RGB
            self._camera_frame = arr[:, :, [2, 1, 0]].copy()

        cam.listen(_cb)
        return cam

    def move_spectator_topdown(self, focus_xy: Tuple[float, float], altitude_m: float = 40.0) -> None:
        carla = self._carla
        self.world.get_spectator().set_transform(
            carla.Transform(
                carla.Location(x=float(focus_xy[0]), y=float(focus_xy[1]), z=float(altitude_m)),
                carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
            )
        )

    def tick_until_frame_ready(self, max_ticks: int = 20, settle_s: float = 0.05) -> bool:
        import time as _t  # noqa: PLC0415
        for _ in range(int(max_ticks)):
            self.world.tick()
            # In CARLA 0.9.12 sync mode the sensor listen callback may fire a hair
            # after tick() returns; poll briefly.
            for _wait in range(10):
                if self._camera_frame is not None:
                    return True
                _t.sleep(settle_s)
        return self._camera_frame is not None

    def capture_png(
        self,
        out_path: Path,
        legend: Optional[Sequence[Tuple[str, Tuple[int, int, int]]]] = None,
        title: Optional[str] = None,
        overlays: Optional[Sequence[Tuple[np.ndarray, Tuple[int, int, int], int, str]]] = None,
        cam_xy: Optional[Tuple[float, float]] = None,
        cam_alt_m: Optional[float] = None,
        cam_fov_deg: float = 90.0,
    ) -> Optional[Path]:
        """Write the latest camera frame; optionally overlay a color legend + title."""
        if self._camera_frame is None:
            return None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image, ImageDraw, ImageFont  # type: ignore  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            plt.imsave(str(out_path), self._camera_frame)
            return out_path
        img = Image.fromarray(self._camera_frame).convert("RGB")

        # Project and draw trajectories directly onto the image (CARLA debug.draw_line
        # does not render reliably into sensor camera frames at high altitudes).
        if overlays and cam_xy is not None and cam_alt_m is not None:
            draw = ImageDraw.Draw(img)
            w, h = img.width, img.height
            # For a top-down pinhole camera (pitch=-90, yaw=0) with fov in degrees:
            # meters-per-pixel at ground level = (2 * altitude * tan(fov/2)) / width
            half_fov = math.radians(float(cam_fov_deg) / 2.0)
            world_half_w = float(cam_alt_m) * math.tan(half_fov)
            m_per_px = (2.0 * world_half_w) / float(w)

            def _w2p(xy: np.ndarray) -> Tuple[int, int]:
                # Image top = +x world, right = +y world (see CARLA axis notes).
                dx = float(xy[0]) - float(cam_xy[0])
                dy = float(xy[1]) - float(cam_xy[1])
                u = int(round(w / 2.0 + dy / m_per_px))
                v = int(round(h / 2.0 - dx / m_per_px))
                return u, v

            for pts, rgb, width_px, style in overlays:
                arr = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
                if len(arr) < 1:
                    continue
                color = tuple(int(c) for c in rgb)
                pixels = [_w2p(arr[i]) for i in range(len(arr))]
                # Clip to only pixels within image bounds (still draw lines to edges).
                if len(pixels) >= 2:
                    draw.line(pixels, fill=color, width=int(width_px))
                if style in ("markers", "both"):
                    r = max(3, int(width_px))
                    for u, v in pixels:
                        if 0 <= u < w and 0 <= v < h:
                            draw.ellipse([(u - r, v - r), (u + r, v + r)], fill=color)
        if legend or title:
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
                )
                small = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14
                )
            except Exception:  # noqa: BLE001
                font = ImageFont.load_default()
                small = font
            if title:
                draw.rectangle([(0, 0), (img.width, 28)], fill=(0, 0, 0))
                draw.text((10, 4), title, fill=(255, 255, 255), font=font)
            if legend:
                pad = 8
                row_h = 20
                width = 320
                box_h = row_h * len(legend) + pad * 2
                top = img.height - box_h - 10
                left = 10
                draw.rectangle(
                    [(left, top), (left + width, top + box_h)], fill=(0, 0, 0, 180)
                )
                for i, (label, rgb) in enumerate(legend):
                    y = top + pad + i * row_h
                    draw.rectangle(
                        [(left + pad, y + 3), (left + pad + 26, y + row_h - 3)],
                        fill=tuple(int(c) for c in rgb),
                    )
                    draw.text(
                        (left + pad + 34, y), label, fill=(240, 240, 240), font=small
                    )
        img.save(str(out_path))
        return out_path

    def reset_capture(self) -> None:
        self._camera_frame = None

    # -- front camera -------------------------------------------------------- #

    def spawn_front_camera(
        self,
        ego_xy: Tuple[float, float],
        ego_yaw_deg: float,
        height_m: float = 2.3,
        fwd_offset_m: float = 1.3,
        width: int = 1280,
        height: int = 720,
        fov_deg: float = 90.0,
    ) -> Tuple[float, float, float, float]:
        """Spawn a forward-looking RGB camera near the ego's hood position.

        Returns (cam_wx, cam_wy, cam_wz, cam_yaw_deg) for projection.
        """
        carla = self._carla
        bp = self.bp_lib.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(int(width)))
        bp.set_attribute("image_size_y", str(int(height)))
        bp.set_attribute("fov", str(float(fov_deg)))
        bp.set_attribute("sensor_tick", "0.0")
        for attr, val in (
            ("motion_blur_intensity", "0.0"),
            ("motion_blur_max_distortion", "0.0"),
            ("motion_blur_min_object_screen_size", "0.0"),
            ("lens_flare_intensity", "0.0"),
            ("bloom_intensity", "0.0"),
            ("exposure_mode", "manual"),
            ("exposure_compensation", "0.0"),
            ("shutter_speed", "200.0"),
            ("iso", "100.0"),
            ("gamma", "2.2"),
        ):
            if bp.has_attribute(attr):
                try:
                    bp.set_attribute(attr, val)
                except Exception:  # noqa: BLE001
                    pass
        yaw_rad = math.radians(ego_yaw_deg)
        cam_wx = float(ego_xy[0]) + fwd_offset_m * math.cos(yaw_rad)
        cam_wy = float(ego_xy[1]) + fwd_offset_m * math.sin(yaw_rad)
        cam_wz = float(height_m)
        tf = carla.Transform(
            carla.Location(x=cam_wx, y=cam_wy, z=cam_wz),
            carla.Rotation(pitch=0.0, yaw=float(ego_yaw_deg), roll=0.0),
        )
        cam = self.world.spawn_actor(bp, tf)
        self._spawned.append(cam)

        def _cb(img):
            arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
            self._front_camera_frame = arr[:, :, [2, 1, 0]].copy()

        cam.listen(_cb)
        return cam_wx, cam_wy, cam_wz, float(ego_yaw_deg)

    def reset_front_capture(self) -> None:
        self._front_camera_frame = None

    def capture_front_png(
        self,
        out_path: Path,
        gt_raw_world: Optional[np.ndarray] = None,
        codriving_world: Optional[np.ndarray] = None,
        cam_wx: float = 0.0,
        cam_wy: float = 0.0,
        cam_wz: float = 2.3,
        cam_yaw_deg: float = 0.0,
        cam_fov_deg: float = 90.0,
        title: Optional[str] = None,
    ) -> Optional[Path]:
        """Render a front-camera perspective PNG with expert GT + codriving overlays."""
        if self._front_camera_frame is None:
            return None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image, ImageDraw, ImageFont  # type: ignore  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            plt.imsave(str(out_path), self._front_camera_frame)
            return out_path
        img = Image.fromarray(self._front_camera_frame).convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.width, img.height
        cam_yaw_rad = math.radians(cam_yaw_deg)

        def _draw_traj(pts_world: np.ndarray, rgb: Tuple[int, int, int], lw: int) -> None:
            px = _project_world_to_front_cam(
                pts_world, cam_wx, cam_wy, cam_wz, cam_yaw_rad, cam_fov_deg, w, h
            )
            visible = [
                (int(round(float(px[i, 0]))), int(round(float(px[i, 1]))))
                for i in range(len(px))
                if not (math.isnan(px[i, 0]) or math.isnan(px[i, 1]))
                and 0 <= px[i, 0] < w and 0 <= px[i, 1] < h
            ]
            if len(visible) >= 2:
                draw.line(visible, fill=rgb, width=lw)
            r = max(3, lw)
            for uv in visible:
                draw.ellipse([(uv[0] - r, uv[1] - r), (uv[0] + r, uv[1] + r)], fill=rgb)

        if gt_raw_world is not None and len(gt_raw_world) > 0:
            _draw_traj(gt_raw_world, (235, 130, 40), 4)    # orange = expert GT
        if codriving_world is not None and len(codriving_world) > 0:
            _draw_traj(codriving_world, (225, 50, 50), 5)  # red = codriving
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except Exception:  # noqa: BLE001
            font = ImageFont.load_default()
            small = font
        if title:
            draw.rectangle([(0, 0), (img.width, 28)], fill=(0, 0, 0))
            draw.text((10, 4), title, fill=(255, 255, 255), font=font)
        legend = [("Expert GT (raw)", (235, 130, 40)), ("CoDriving prediction", (225, 50, 50))]
        pad, row_h, box_w = 8, 20, 270
        box_h = row_h * len(legend) + pad * 2
        top, left = img.height - box_h - 10, 10
        draw.rectangle([(left, top), (left + box_w, top + box_h)], fill=(0, 0, 0, 180))
        for i, (lbl, rgb) in enumerate(legend):
            y = top + pad + i * row_h
            draw.rectangle(
                [(left + pad, y + 3), (left + pad + 26, y + row_h - 3)], fill=tuple(rgb)
            )
            draw.text((left + pad + 34, y), lbl, fill=(240, 240, 240), font=small)
        img.save(str(out_path))
        return out_path

    # -- debug draws --------------------------------------------------------- #

    def draw_overlays(
        self,
        *,
        ego_xy: np.ndarray,
        route_full: np.ndarray,
        gt_raw: np.ndarray,
        gt_aligned: np.ndarray,
        codriving_world: np.ndarray,
        surrounding_gt: Optional[List[np.ndarray]] = None,
        z_base: float = 10.5,
        life_time: float = 15.0,
    ) -> None:
        carla = self._carla
        debug = self.world.debug

        def _loc(x: float, y: float, dz: float = 0.0):
            return carla.Location(x=float(x), y=float(y), z=float(z_base + dz))

        def _poly(pts: np.ndarray, color, thickness: float, dz: float) -> None:
            for i in range(len(pts) - 1):
                debug.draw_line(
                    _loc(pts[i][0], pts[i][1], dz),
                    _loc(pts[i + 1][0], pts[i + 1][1], dz),
                    thickness=thickness, color=color, life_time=life_time, persistent_lines=True,
                )

        def _pts(pts: np.ndarray, color, size: float, dz: float) -> None:
            for p in pts:
                debug.draw_point(
                    _loc(p[0], p[1], dz), size=size, color=color,
                    life_time=life_time, persistent_lines=False,
                )

        # Thickness and z-offsets are scaled for a top-down camera at tens of meters
        # altitude: from up there, 0.1m-thick lines are sub-pixel. We draw at roof
        # height (~3m above ground) so the lines sit above vehicles and remain visible.
        # Global route (gray).
        _poly(route_full, carla.Color(110, 110, 110), thickness=0.8, dz=2.0)
        # Raw real GT (orange).
        _poly(gt_raw, carla.Color(235, 130, 40), thickness=1.2, dz=3.0)
        _pts(gt_raw, carla.Color(235, 130, 40), size=0.35, dz=3.0)
        # Aligned GT (yellow).
        _poly(gt_aligned, carla.Color(245, 200, 30), thickness=1.2, dz=3.5)
        _pts(gt_aligned, carla.Color(245, 200, 30), size=0.35, dz=3.5)
        # CoDriving prediction (red).
        if len(codriving_world) > 1:
            _poly(codriving_world, carla.Color(225, 50, 50), thickness=1.4, dz=4.0)
            _pts(codriving_world, carla.Color(225, 50, 50), size=0.45, dz=4.0)
        # Surrounding actor GTs (teal).
        if surrounding_gt:
            for tr in surrounding_gt:
                if len(tr) > 1:
                    _poly(tr, carla.Color(30, 200, 200), thickness=0.7, dz=2.5)
        # Ego marker.
        debug.draw_point(
            _loc(float(ego_xy[0]), float(ego_xy[1]), dz=5.0),
            size=0.9, color=carla.Color(40, 210, 90),
            life_time=life_time, persistent_lines=False,
        )



# ------------------------------------------------------------------ #
# Front-camera perspective projection                                #
# ------------------------------------------------------------------ #


def _project_world_to_front_cam(
    world_xy: np.ndarray,
    cam_wx: float,
    cam_wy: float,
    cam_wz: float,
    cam_yaw_rad: float,
    cam_fov_deg: float,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """Project world XY ground points (z=0) into a front-facing pinhole camera.

    Camera coordinate frame: Z=forward, X=right, Y=down (standard pinhole).
    Returns pixel coordinates [N, 2] (u, v) as float; NaN for points behind camera.
    """
    pts = np.asarray(world_xy, dtype=np.float64).reshape(-1, 2)
    dx = pts[:, 0] - cam_wx
    dy = pts[:, 1] - cam_wy
    c, s = math.cos(cam_yaw_rad), math.sin(cam_yaw_rad)
    cam_z = dx * c + dy * s          # depth (forward)
    cam_x = -dx * s + dy * c         # rightward
    cam_y_val = float(cam_wz)        # all ground points at same elevation below camera
    f = img_w / (2.0 * math.tan(math.radians(cam_fov_deg / 2.0)))
    valid = cam_z > 0.1
    safe_z = np.where(valid, cam_z, 1.0)
    u = np.where(valid, f * cam_x / safe_z + img_w / 2.0, np.nan)
    v = np.where(valid, f * cam_y_val / safe_z + img_h / 2.0, np.nan)
    return np.column_stack([u, v])


# ------------------------------------------------------------------ #
# Front-camera image from codriving agent's saved composite JPG     #
# ------------------------------------------------------------------ #


def _extract_front_view_from_run_jpg(
    run_dir: Path,
    codriving_json: Path,
    out_path: Path,
) -> Optional[Path]:
    """Extract the front-camera view from the composite JPG saved by pnp_infer_action_e2e_v2.

    The composite is 6750 wide x 1500 tall: front RGB (first 3000 px), lidar 3D, lidar BEV.
    Returns out_path on success, None if the JPG does not exist.
    """
    jpg_path = codriving_json.with_suffix(".jpg")
    if not jpg_path.exists():
        return None
    try:
        from PIL import Image as _Image  # noqa: PLC0415
        composite = _Image.open(jpg_path).convert("RGB")
        w, _ = composite.size
        front_w = min(3000, w)
        front_img = composite.crop((0, 0, front_w, composite.height))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        front_img.save(str(out_path))
        return out_path
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] could not extract front view from {jpg_path}: {exc}")
        return None


# ------------------------------------------------------------------ #
# Live CARLA driver                                                  #
# ------------------------------------------------------------------ #


def live_carla_capture(
    *,
    carla_dir: Path,
    target_t_s: float,
    ego_xy: np.ndarray,
    ego_yaw_deg: float,
    route_full: np.ndarray,
    gt_raw: np.ndarray,
    gt_aligned: np.ndarray,
    codriving_world: np.ndarray,
    altitude_m: float,
    out_stem: Path,
    carla_port: int = 2000,
) -> Tuple[Optional[Path], Optional[Path], Optional[Path], str]:
    """End-to-end live CARLA capture: spawn the scene at ``target_t_s``, snapshot
    from a top-down camera, push debug draws, then produce three annotated snapshots.

    Returns (scene_png, gt_png, overlay_png, status_message):
      scene_png   — clean scene, no annotations
      gt_png      — GT expert vs aligned GT only
      overlay_png — aligned GT + codriving prediction only
    """
    try:
        import carla  # type: ignore  # noqa: PLC0415,F401
    except Exception as exc:  # noqa: BLE001
        return None, None, None, f"carla import failed: {exc}"

    try:
        specs = build_actor_spawn_list(carla_dir, target_t_s)
    except Exception as exc:  # noqa: BLE001
        return None, None, None, f"failed to parse actor manifest: {exc}"
    print(f"[CARLA] actor specs at t={target_t_s:.2f}s: {len(specs)} total")

    scene_png = gt_png = overlay_png = None
    try:
        with CarlaSceneSession(port=carla_port) as sess:
            removed = sess.destroy_existing_world_actors()
            print(f"[CARLA] destroyed {removed} pre-existing world actors")
            ok, skip = sess.spawn_scene(specs)
            print(f"[CARLA] spawned {ok} / skipped {skip} actors from scenario manifest")
            sess.move_spectator_topdown((float(ego_xy[0]), float(ego_xy[1])), altitude_m=altitude_m)
            sess.spawn_topdown_camera(
                (float(ego_xy[0]), float(ego_xy[1])),
                altitude_m=altitude_m,
            )
            # Pre-warm: tick long enough for map streaming + lighting to settle after
            # load_world. Without this the first camera frame can show missing
            # road textures / markings.
            for _ in range(30):
                sess.world.tick()
            sess.reset_capture()
            if not sess.tick_until_frame_ready(max_ticks=30):
                return None, None, None, "camera did not produce a frame after 30 ticks"

            # 1. Clean scene — no title, no overlays.
            scene_target = out_stem.with_name(out_stem.name + "_carla_scene.png")
            scene_png = sess.capture_png(scene_target)

            # Also push in-world debug lines (for CARLA spectator window only).
            sess.draw_overlays(
                ego_xy=ego_xy,
                route_full=route_full,
                gt_raw=gt_raw,
                gt_aligned=gt_aligned,
                codriving_world=codriving_world,
                z_base=10.3,
                life_time=30.0,
            )

            # 2. GT expert vs aligned GT (no prediction, no surrounding actors).
            gt_target = out_stem.with_name(out_stem.name + "_carla_gt.png")
            gt_png = sess.capture_png(
                gt_target,
                title=f"GT expert vs aligned  t={target_t_s:.2f}s  {out_stem.name}",
                legend=[
                    ("CARLA global route", (200, 200, 200)),
                    ("real GT (transformed)", (235, 130, 40)),
                    ("GT aligned to route", (245, 200, 30)),
                    ("ego @ this frame", (40, 210, 90)),
                ],
                overlays=[
                    (route_full, (200, 200, 200), 2, "line"),
                    (gt_raw, (235, 130, 40), 4, "both"),
                    (gt_aligned, (245, 200, 30), 4, "both"),
                    (np.array([ego_xy]), (40, 210, 90), 8, "markers"),
                ],
                cam_xy=(float(ego_xy[0]), float(ego_xy[1])),
                cam_alt_m=altitude_m,
                cam_fov_deg=90.0,
            )

            # 3. Aligned GT + prediction only.
            pred_overlays: List[Tuple[np.ndarray, Tuple[int, int, int], int, str]] = [
                (route_full, (200, 200, 200), 2, "line"),
                (gt_aligned, (245, 200, 30), 4, "both"),
            ]
            if len(codriving_world) > 0:
                pred_overlays.append((codriving_world, (225, 50, 50), 5, "both"))
            pred_overlays.append((np.array([ego_xy]), (40, 210, 90), 8, "markers"))
            overlay_target = out_stem.with_name(out_stem.name + "_carla_overlay.png")
            overlay_png = sess.capture_png(
                overlay_target,
                title=f"CARLA scene + aligned GT + prediction  t={target_t_s:.2f}s  {out_stem.name}",
                legend=[
                    ("CARLA global route", (200, 200, 200)),
                    ("GT aligned to route", (245, 200, 30)),
                    ("codriving prediction", (225, 50, 50)),
                    ("ego @ this frame", (40, 210, 90)),
                ],
                overlays=pred_overlays,
                cam_xy=(float(ego_xy[0]), float(ego_xy[1])),
                cam_alt_m=altitude_m,
                cam_fov_deg=90.0,
            )

    except RuntimeError as exc:
        return None, None, None, str(exc)
    except Exception as exc:  # noqa: BLE001
        return None, None, None, f"CARLA session failed: {exc}"

    return scene_png, gt_png, overlay_png, "captured live CARLA scene + overlays"


# ------------------------------------------------------------------ #
# CARLA server lifecycle                                             #
# ------------------------------------------------------------------ #


def start_carla_server(
    port: int = 2000,
    extra_args: Sequence[str] = (),
    ready_timeout_s: float = 120.0,
    graphics_adapter: Optional[int] = 0,
) -> "subprocess.Popen[bytes]":
    """Launch CarlaUE4.sh in the background and block until it accepts TCP connections.

    Returns the Popen handle so the caller can terminate it when done.
    Raises RuntimeError if CARLA isn't reachable within *ready_timeout_s* seconds.
    """
    import socket as _socket  # noqa: PLC0415
    import tempfile as _tmp  # noqa: PLC0415
    import time as _t  # noqa: PLC0415

    repo_root = Path(__file__).resolve().parents[2]
    carla_sh = repo_root / "carla912" / "CarlaUE4.sh"
    if not carla_sh.exists():
        raise FileNotFoundError(f"CarlaUE4.sh not found at {carla_sh}")

    log_path = Path(_tmp.mktemp(suffix="_carla.log", prefix="carla_"))
    adapter_args = [f"-graphicsadapter={graphics_adapter}"] if graphics_adapter is not None else []
    cmd = [str(carla_sh), f"--world-port={port}", "-RenderOffScreen"] + adapter_args + list(extra_args)
    env = os.environ.copy()
    # Disable the custom strace/diagnostic wrapper — same as run_custom_eval.py's
    # _build_runtime_env() does when --carla-forensics is off.  Without this the
    # wrapper launches CARLA under strace and the shell-script PID exits immediately
    # with code 127 before CARLA finishes initialising.
    env["CARLA_SERVER_DIAG"] = "0"
    env["CARLA_DIAG_ENABLE_STRACE"] = "0"
    env.setdefault("SDL_VIDEODRIVER", "offscreen")
    print(f"[CARLA] launching: {' '.join(cmd)}")
    print(f"[CARLA] stdout/stderr -> {log_path}")
    log_fh = open(log_path, "wb")  # noqa: WPS515
    proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=log_fh)

    def _port_open() -> bool:
        try:
            s = _socket.create_connection(("localhost", port), timeout=2.0)
            s.close()
            return True
        except OSError:
            return False

    deadline = _t.time() + ready_timeout_s
    attempt = 0
    while _t.time() < deadline:
        # Bail immediately if the process already died.
        if proc.poll() is not None:
            log_fh.close()
            tail = log_path.read_text(errors="replace")[-2000:]
            raise RuntimeError(
                f"CarlaUE4.sh exited with code {proc.returncode} before becoming ready.\n"
                f"Last log lines:\n{tail}"
            )
        if _port_open():
            log_fh.close()
            print(f"[CARLA] server ready on port {port} (attempt {attempt + 1})")
            return proc
        attempt += 1
        _t.sleep(3.0)

    proc.terminate()
    log_fh.close()
    tail = log_path.read_text(errors="replace")[-2000:]
    raise RuntimeError(
        f"CARLA server on port {port} did not become ready within {ready_timeout_s:.0f}s.\n"
        f"Log tail ({log_path}):\n{tail}"
    )


# ------------------------------------------------------------------ #
# Codriving evaluation via run_custom_eval (1-step, in live CARLA)   #
# ------------------------------------------------------------------ #


def _inject_ego_speed_thread(
    carla_port: int,
    speed_mps: float,
    yaw_deg: float,
    poll_interval_s: float = 0.05,
    timeout_s: float = 60.0,
) -> None:
    """Background thread: hold ego vehicle at the given speed by re-injecting every poll cycle.

    The CARLA scenario always starts the ego at rest (speed=0). If the real dataset
    ego was already moving at the target frame, the codriving model receives speed=0
    and predicts very short waypoints. This thread injects the real ego speed via
    set_target_velocity on every poll cycle for the full timeout duration so the
    vehicle stays at the target speed through multiple inference steps.
    set_target_velocity is a one-shot impulse that CARLA physics overrides on the next
    tick — continuous re-injection is required to maintain the speed.
    """
    import time as _time  # noqa: PLC0415

    try:
        import carla as _carla  # type: ignore  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        print("[speed-inject] carla not importable; skipping velocity injection")
        return

    yaw_rad = math.radians(yaw_deg)
    vx = speed_mps * math.cos(yaw_rad)
    vy = speed_mps * math.sin(yaw_rad)

    deadline = _time.time() + timeout_s
    client: Optional[object] = None
    ego_actor: Optional[object] = None
    inject_count = 0

    while _time.time() < deadline:
        try:
            if client is None:
                client = _carla.Client("localhost", carla_port)
                client.set_timeout(3.0)  # type: ignore[union-attr]
            # Re-resolve ego actor periodically in case world reloaded
            if ego_actor is None:
                world = client.get_world()  # type: ignore[union-attr]
                for actor in world.get_actors():
                    if not actor.type_id.startswith("vehicle."):
                        continue
                    role = (getattr(actor, "attributes", {}) or {}).get("role_name", "")
                    if "ego" not in role.lower() and "hero" not in role.lower():
                        continue
                    ego_actor = actor
                    print(
                        f"[speed-inject] found ego '{role}' — "
                        f"holding at {speed_mps:.2f} m/s @ {yaw_deg:.1f}° "
                        f"(vx={vx:.2f}, vy={vy:.2f}) for {timeout_s:.0f}s"
                    )
                    break
            if ego_actor is not None:
                ego_actor.set_target_velocity(  # type: ignore[union-attr]
                    _carla.Vector3D(float(vx), float(vy), 0.0)
                )
                inject_count += 1
        except Exception as _exc:  # noqa: BLE001
            # CARLA world reloading or actor gone — reset and retry
            client = None
            ego_actor = None
            _ = _exc
        _time.sleep(poll_interval_s)

    if inject_count == 0:
        print(f"[speed-inject] WARN: could not inject ego speed within {timeout_s:.0f}s")
    else:
        print(f"[speed-inject] done — injected velocity {inject_count} times over {timeout_s:.0f}s")


def run_planner_evaluation(
    scenario: str,
    *,
    planner: str = "codriving",
    carla_port: int = 2000,
    pred_poll_timeout_s: float = 120.0,
    initial_ego_speed_mps: float = 0.0,
    initial_ego_yaw_deg: float = 0.0,
    timeout_s: int = 10,
) -> Optional[Path]:
    """Run any leaderboard planner for one prediction step using an already-running CARLA.

    Generalized form of ``run_codriving_evaluation``. The planner alias must
    be registered in ``run_custom_eval``'s PLANNER_SPECS (e.g. ``codriving``,
    ``perception_swap_fcooper``). Per-step prediction JSONs end up under
    ``PLANNER_RESULTS_ROOTS[planner] / scenario / image/<runtag>/ego_vehicle_<i>/``.

    We launch in the background, poll for the first *new* prediction JSON, then
    wait for the process to exit naturally within the ``timeout_s`` window.
    CARLA stays alive on ``carla_port`` afterwards so live_carla_capture / metric
    queries can reuse the same server. Connects via ``--no-start-carla``.

    If ``initial_ego_speed_mps > 0``, a background thread polls CARLA for the
    ego vehicle and calls set_target_velocity as soon as it appears, giving
    the model a non-zero speed on its first inference step.
    """
    import threading as _threading  # noqa: PLC0415
    import time as _time  # noqa: PLC0415

    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "tools" / "run_custom_eval.py"
    routes_dir = repo_root / "scenarioset" / "v2xpnp" / scenario
    results_tag = f"v2xpnp/{planner}/{scenario}"
    results_root = PLANNER_RESULTS_ROOTS.get(
        planner,
        Path(f"/data2/marco/CoLMDriver/results/results_driving_custom/v2xpnp/{planner}"),
    )

    # Snapshot the set of prediction dirs that already exist so we can detect new ones.
    _image_base = results_root / scenario / "image"
    _pre_existing: set = set()
    if _image_base.is_dir():
        for _ego_dir in _image_base.glob("*/ego_vehicle_*/"):
            if any(_ego_dir.glob("*.json")):
                _pre_existing.add(str(_ego_dir))

    # Find a free port for the traffic manager to avoid collisions with orphaned TM
    # processes from previous sessions (the default carla_port+5 may be occupied).
    import socket as _socket  # noqa: PLC0415
    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _s:
        _s.bind(("", 0))
        _s.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        _free_tm_port = _s.getsockname()[1]
    print(f"[{planner}-eval] using free TM port {_free_tm_port}")

    cmd = [
        sys.executable, str(script),
        "--routes-dir", str(routes_dir),
        "--planner", planner,
        "--results-tag", results_tag,
        "--no-start-carla",            # don't manage CARLA; connect to existing server
        "--port", str(carla_port),
        "--traffic-manager-port", str(_free_tm_port),
        "--no-skip-existed",           # always use a fresh runtag; avoids stale heartbeat watchdog
        "--timeout", str(int(timeout_s)),
    ]
    print(f"[{planner}-eval] launching: {' '.join(cmd)}")
    # For perception_swap_* aliases, ask the agent to dump per-tick *.json +
    # *_pred.npz prediction files compatible with compute_metrics_from_run_dir.
    # Codriving and other legacy planners write their own native format already,
    # so don't touch their env.
    _child_env = dict(os.environ)
    if planner.startswith("perception_swap_"):
        _child_env["PERCEPTION_SWAP_WRITE_PRED_FILES"] = "1"
        # Match codriving's 0.2 s prediction cadence (5 Hz @ 20 Hz tick rate).
        _child_env.setdefault("PERCEPTION_SWAP_PRED_FILE_STRIDE", "4")
    proc = subprocess.Popen(cmd, cwd=str(repo_root), env=_child_env)

    # Inject real ego speed in a background thread so the model doesn't start from 0.
    if initial_ego_speed_mps > 0.05:
        print(
            f"[{planner}-eval] spawning speed-inject thread: "
            f"{initial_ego_speed_mps:.2f} m/s @ {initial_ego_yaw_deg:.1f}°"
        )
        _threading.Thread(
            target=_inject_ego_speed_thread,
            args=(carla_port, initial_ego_speed_mps, initial_ego_yaw_deg),
            daemon=True,
        ).start()

    new_run_dir: Optional[Path] = None
    deadline = _time.time() + pred_poll_timeout_s
    while _time.time() < deadline:
        # Check for a new prediction dir (not in pre_existing snapshot).
        _cur = find_planner_run_dir(planner, scenario)
        if _cur and str(_cur) not in _pre_existing and any(_cur.glob("*.json")):
            new_run_dir = _cur
            print(f"[{planner}-eval] new prediction at {new_run_dir}")
            break
        if proc.poll() is not None:
            print(f"[{planner}-eval] process exited (code {proc.returncode})")
            break
        _time.sleep(2.0)

    # Wait for the process to exit fully so CARLA is idle before scene capture.
    try:
        proc.wait(timeout=30.0)
    except Exception:  # noqa: BLE001
        proc.terminate()

    if new_run_dir is None:
        # Process exited first; check one more time for any new predictions.
        _cur = find_planner_run_dir(planner, scenario)
        if _cur and str(_cur) not in _pre_existing and any(_cur.glob("*.json")):
            new_run_dir = _cur
            print(f"[{planner}-eval] prediction found after proc exit: {new_run_dir}")

    if new_run_dir is not None:
        return new_run_dir
    print(f"[{planner}-eval] WARN: no new prediction JSONs found")
    return None


def run_codriving_evaluation(
    scenario: str,
    carla_port: int = 2000,
    pred_poll_timeout_s: float = 120.0,
    initial_ego_speed_mps: float = 0.0,
    initial_ego_yaw_deg: float = 0.0,
) -> Optional[Path]:
    """Backward-compatible wrapper for legacy ``--live-codriving`` callers."""
    return run_planner_evaluation(
        scenario,
        planner="codriving",
        carla_port=carla_port,
        pred_poll_timeout_s=pred_poll_timeout_s,
        initial_ego_speed_mps=initial_ego_speed_mps,
        initial_ego_yaw_deg=initial_ego_yaw_deg,
    )


# ------------------------------------------------------------------ #
# AP@50 evaluation: per-frame world-frame predictions vs live CARLA  #
# actor ground-truth bounding boxes.                                 #
# ------------------------------------------------------------------ #


def _carla_actor_world_corners(actor) -> np.ndarray:
    """Return an (8, 3) world-frame corner array for one CARLA vehicle/walker actor.

    Prefers ``BoundingBox.get_world_vertices`` (CARLA >=0.9.10); falls back to
    a manual yaw-rotate + translate using ``bounding_box.extent`` and the actor
    transform.
    """
    bb = actor.bounding_box
    tf = actor.get_transform()
    try:
        verts = bb.get_world_vertices(tf)
        return np.array([[v.x, v.y, v.z] for v in verts], dtype=np.float64)
    except Exception:  # noqa: BLE001
        ext = bb.extent
        body = np.array([
            [+ext.x, +ext.y, +ext.z], [+ext.x, -ext.y, +ext.z],
            [-ext.x, -ext.y, +ext.z], [-ext.x, +ext.y, +ext.z],
            [+ext.x, +ext.y, -ext.z], [+ext.x, -ext.y, -ext.z],
            [-ext.x, -ext.y, -ext.z], [-ext.x, +ext.y, -ext.z],
        ], dtype=np.float64)
        yaw_rad = math.radians(float(tf.rotation.yaw))
        c, s = math.cos(yaw_rad), math.sin(yaw_rad)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        rotated = body @ R.T
        loc = np.array([
            float(tf.location.x) + float(bb.location.x),
            float(tf.location.y) + float(bb.location.y),
            float(tf.location.z) + float(bb.location.z),
        ])
        return rotated + loc


def extract_carla_gt_corners(
    sess: "CarlaSceneSession",
    exclude_actor_ids: Sequence[int] = (),
) -> np.ndarray:
    """Return (N, 8, 3) world-frame GT corners for every vehicle in ``sess.world``,
    skipping any actor whose id is in ``exclude_actor_ids`` (typically the ego).
    """
    out: List[np.ndarray] = []
    for a in sess.world.get_actors().filter("vehicle.*"):
        if a.id in exclude_actor_ids:
            continue
        out.append(_carla_actor_world_corners(a))
    if not out:
        return np.zeros((0, 8, 3), dtype=np.float32)
    return np.stack(out, axis=0).astype(np.float32)


def _identify_ego_actor_ids(sess: "CarlaSceneSession") -> set:
    """Return the set of actor ids whose role_name matches the spawn-time ego tag.

    ``CarlaSceneSession.spawn_scene`` sets ``role_name = f"{role}_{name}"`` (e.g.
    ``ego_vehicle_0``). We treat any role_name starting with ``ego`` or containing
    ``hero`` as the ego/agent vehicle to exclude from GT.
    """
    egos: set = set()
    for a in sess.world.get_actors().filter("vehicle.*"):
        role = (getattr(a, "attributes", {}) or {}).get("role_name", "") or ""
        rl = role.lower()
        if rl.startswith("ego") or "hero" in rl:
            egos.add(a.id)
    return egos


def _yaml_actors_to_carla_xy(
    yaml_path: Path,
    align_cfg: "AlignCfg",
) -> List[Tuple[int, np.ndarray]]:
    """Read a v2xpnp per-frame YAML and return [(actor_id, [x,y]_carla), ...]."""
    if not yaml_path.exists():
        return []
    try:
        d = yaml.safe_load(yaml_path.read_text())
    except Exception:  # noqa: BLE001
        return []
    out: List[Tuple[int, np.ndarray]] = []
    vehicles = (d or {}).get("vehicles") or {}
    for aid, meta in vehicles.items():
        loc = (meta or {}).get("location")
        if not loc or len(loc) < 2:
            continue
        try:
            aid_i = int(aid)
        except (TypeError, ValueError):
            continue
        xy_real = np.array([float(loc[0]), float(loc[1])], dtype=np.float64)
        xy_carla = real_to_carla_xy(xy_real[None, :], align_cfg).squeeze()
        out.append((aid_i, xy_carla))
    return out


# ---------------------------------------------------------------- #
# Collision-rate primitives: BEV polygons for ego, actors, static  #
# map geometry (buildings, trees, poles, signs, fences).            #
# ---------------------------------------------------------------- #


def _yaw_from_trajectory(pred_wps: np.ndarray, anchor_xy: np.ndarray) -> np.ndarray:
    """Return per-waypoint yaw (radians) derived from trajectory tangents.

    For waypoint k>0: yaw = atan2(wps[k] - wps[k-1]).
    For waypoint 0:   yaw = atan2(wps[0] - anchor_xy).
    Holds the previous yaw on near-zero deltas.
    """
    pred_wps = np.asarray(pred_wps, dtype=np.float64)
    T = len(pred_wps)
    yaws = np.zeros(T, dtype=np.float64)
    for k in range(T):
        d = pred_wps[k] - (pred_wps[k - 1] if k > 0 else np.asarray(anchor_xy, dtype=np.float64))
        if np.linalg.norm(d) < 1e-3:
            yaws[k] = yaws[k - 1] if k > 0 else 0.0
        else:
            yaws[k] = math.atan2(d[1], d[0])
    return yaws


def _bev_polygon(cx: float, cy: float, yaw_rad: float, half_length: float, half_width: float):
    """Build a shapely.Polygon for an oriented rectangle in BEV (top-down)."""
    from shapely.geometry import Polygon  # noqa: PLC0415
    body = np.array([
        [+half_length, +half_width],
        [+half_length, -half_width],
        [-half_length, -half_width],
        [-half_length, +half_width],
    ], dtype=np.float64)
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    R = np.array([[c, -s], [s, c]])
    rotated = body @ R.T
    return Polygon(rotated + np.array([cx, cy], dtype=np.float64))


def _yaml_vehicles_bev_polygons(
    yaml_path: Path,
    align_cfg: "AlignCfg",
):
    """Return list of (centroid_xy, shapely_polygon) for vehicles in a v2xpnp YAML.

    Each vehicle has location [x,y,z], angle [roll, yaw_deg, pitch], extent
    [half_x, half_y, half_z]. We transform location and yaw to the CARLA frame
    and build an oriented BEV rectangle from extent[:2].
    """
    if not yaml_path.exists():
        return []
    try:
        d = yaml.safe_load(yaml_path.read_text())
    except Exception:  # noqa: BLE001
        return []
    out: List[Tuple[np.ndarray, "Polygon"]] = []
    vehicles = (d or {}).get("vehicles") or {}
    for _aid, meta in vehicles.items():
        loc = (meta or {}).get("location")
        ang = (meta or {}).get("angle")
        ext = (meta or {}).get("extent")
        if not loc or not ang or not ext:
            continue
        if len(loc) < 2 or len(ang) < 2 or len(ext) < 2:
            continue
        xy_real = np.array([float(loc[0]), float(loc[1])], dtype=np.float64)
        xy_carla = real_to_carla_xy(xy_real[None, :], align_cfg).squeeze()
        yaw_real_deg = float(ang[1])
        yaw_carla_rad = math.radians(real_yaw_to_carla_yaw(yaw_real_deg, align_cfg))
        hl, hw = float(ext[0]), float(ext[1])
        try:
            poly = _bev_polygon(float(xy_carla[0]), float(xy_carla[1]),
                                yaw_carla_rad, hl, hw)
        except Exception:  # noqa: BLE001
            continue
        if poly.is_valid and not poly.is_empty:
            out.append((np.array([xy_carla[0], xy_carla[1]]), poly))
    return out


_STATIC_OBSTACLE_LABELS = (
    "Buildings", "Vegetation", "Walls", "Fences", "Poles",
    "TrafficSigns", "TrafficLight", "GuardRail", "Bridge", "Static",
)


def _carla_static_bev_polygons(
    world,
    carla_module,
    center_xy: Optional[Tuple[float, float]] = None,
    radius_m: float = 80.0,
):
    """Query CARLA's static map bounding boxes for the labels in
    ``_STATIC_OBSTACLE_LABELS`` and return a list of BEV ``shapely.Polygon``s.

    If ``center_xy`` is given, BBs whose centre is farther than ``radius_m``
    are skipped — keeps the polygon list small enough for tight loops.
    """
    polys = []
    cx0 = cy0 = None
    if center_xy is not None:
        cx0, cy0 = float(center_xy[0]), float(center_xy[1])
    for name in _STATIC_OBSTACLE_LABELS:
        label = getattr(carla_module.CityObjectLabel, name, None)
        if label is None:
            continue
        try:
            bbs = world.get_level_bbs(label)
        except Exception:  # noqa: BLE001
            continue
        for bb in bbs:
            try:
                cx = float(bb.location.x)
                cy = float(bb.location.y)
                if cx0 is not None:
                    if (cx - cx0) ** 2 + (cy - cy0) ** 2 > radius_m * radius_m:
                        continue
                ex = float(bb.extent.x)
                ey = float(bb.extent.y)
                if ex <= 0.0 or ey <= 0.0:
                    continue
                yaw_rad = math.radians(float(bb.rotation.yaw))
                p = _bev_polygon(cx, cy, yaw_rad, ex, ey)
                if p.is_valid and not p.is_empty:
                    polys.append(p)
            except Exception:  # noqa: BLE001
                continue
    return polys


def _build_static_obstacle_index(static_polys):
    """Wrap a list of static BEV polygons in an STRtree if available; otherwise
    return the raw list. Caller handles both shapes via ``_query_obstacles``.
    """
    if not static_polys:
        return None
    try:
        from shapely.strtree import STRtree  # noqa: PLC0415
        return STRtree(static_polys)
    except Exception:  # noqa: BLE001
        return static_polys


def _query_static_candidates(index, query_poly):
    """Return candidate static polygons whose envelope overlaps ``query_poly``.

    Falls back to the full list when STRtree is unavailable.
    """
    if index is None:
        return []
    if isinstance(index, list):
        return index
    try:
        cands = index.query(query_poly)
    except Exception:  # noqa: BLE001
        return []
    # shapely 2.x returns ndarray of indices; older returns list of geoms.
    if isinstance(cands, np.ndarray) and cands.dtype.kind in ("i", "u"):
        try:
            geoms = list(getattr(index, "geometries", []))
            return [geoms[i] for i in cands.tolist() if 0 <= i < len(geoms)]
        except Exception:  # noqa: BLE001
            return []
    return list(cands)


def compute_metrics_from_run_dir(
    *,
    run_dir: Path,
    real_dir: Path,
    real_ego: int,
    align_cfg: "AlignCfg",
    carla_dir: Path,
    carla_port: int,
    scenario_dt_s: float = SCENARIO_DT_S,
    pred_dt_s: float = 0.2,
    pred_horizon_steps: int = 10,
    iou_thresh: float = 0.5,
    iou_thresholds: Optional[List[float]] = None,
    save_path: Optional[Path] = None,
    sess: Optional["CarlaSceneSession"] = None,
    skip_carla_ap: bool = False,
    gt_source: str = "replay",
) -> dict:
    """Compute AP@iou_thresh, ADE, FDE, and Collision Rate over every ``*_pred.npz``
    saved by PnP_infer in ``run_dir``.

    Per-frame anchoring strategy
    ----------------------------
    The codriving step counter (npz/JSON filename) advances at CARLA tick rate,
    not at the v2xpnp recording cadence — and codriving drove its own trajectory,
    so step×dt does not give scenario time. We use the JSON's ``lidar_pose_x/y``
    to find the v2xpnp frame_idx whose ego pose is closest. From that frame_idx,
    everything follows: t_s = frame_idx * scenario_dt_s for AP@50 actor respawn,
    frame_idx + k*stride for ADE/FDE GT, surrounding actors at those frames for CR.

    Metrics
    -------
    - **AP@iou_thresh** : detection AP via the OpenCOOD/RiskM pipeline. GT comes
      from live CARLA actor bounding boxes after re-spawning the scene at t_s.
    - **ADE / FDE**     : prediction waypoints (in world frame) vs v2xpnp future
      ego trajectory at the matched frame's stride.
    - **Collision Rate**: RiskMM-style L2<6m + |ΔY|<3.5m gate over the prediction
      horizon, surrounding actor positions read from v2xpnp YAMLs.

    AP@50 needs CARLA; ADE/FDE/CR don't. If ``skip_carla_ap=True`` or the CARLA
    session can't be opened, AP@50 is reported as NaN and the rest still run.

    GT trajectory source
    --------------------
    ``gt_source`` selects where the ego ground-truth trajectory comes from:
      * ``"replay"`` (default): parse the scenarioset's REPLAY.xml and use that
        as the GT. Fully CARLA-frame, zero real-world data ingest at runtime.
        ADE/FDE are computed with an anchor-shift (``gt = gt - gt[0] + pred[0]``)
        so the 0th waypoint of predictions matches the 0th waypoint of GT — this
        isolates trajectory-shape error from per-frame ego-localization error.
        Collision Rate is reported as NaN in this mode (NPC future poses are
        unavailable without re-spawning each future frame).
      * ``"real"``: legacy behavior — load v2xpnp YAMLs as GT, soft-snap onto
        the REPLAY route. Kept for backward-compatibility reproduction of
        codriving baselines that were originally measured this way.

    AP IoU thresholds
    -----------------
    Pass ``iou_thresholds=[0.3, 0.5, 0.7]`` to get AP at all three IoUs. The
    legacy ``iou_thresh`` scalar is honoured when ``iou_thresholds`` is None.
    """
    try:
        from opencood.utils import eval_utils  # noqa: PLC0415
        import torch  # noqa: PLC0415
        _have_eval_utils = True
    except Exception as exc:  # noqa: BLE001
        print(f"[METRICS] could not import opencood eval utils ({exc}); AP@50 disabled")
        eval_utils = None
        torch = None
        _have_eval_utils = False
        skip_carla_ap = True

    npz_paths = sorted(run_dir.glob("*_pred.npz"))
    json_paths = sorted(run_dir.glob("*.json"))
    if not json_paths:
        print(f"[METRICS] no *.json files in {run_dir}; nothing to evaluate")
        return {}
    # If no NPZ files (older runs), skip AP@50 entirely.
    if not npz_paths:
        print(f"[METRICS] no *_pred.npz files in {run_dir}; AP@50 disabled "
              f"(re-run with the patched PnP_infer to enable)")
        skip_carla_ap = True

    # --- Pre-load ego trajectory for pose matching. ---
    # gt_source="replay": parse the scenarioset's REPLAY.xml — pure CARLA frame,
    #                     no real-world dataset ingest. The xml is at scenario_dt_s
    #                     spacing (typically 0.1s) and contains x,y,yaw,time per row.
    # gt_source="real":   legacy path; reads v2xpnp YAMLs and soft-snaps to route.
    if gt_source == "replay":
        carla_ego_slot = REAL_TO_CARLA_EGO.get(real_ego, real_ego - 1)
        replay_xml = carla_dir / f"ucla_v2_custom_ego_vehicle_{carla_ego_slot}_REPLAY.xml"
        if not replay_xml.is_file():
            print(f"[METRICS] missing REPLAY.xml ({replay_xml.name}); falling back to gt_source='real'")
            gt_source = "real"
        else:
            replay_route = parse_carla_route_xml(replay_xml)
            real_xy_carla = replay_route.xy.astype(np.float64)             # (T, 2)
            real_mask = np.ones(len(real_xy_carla), dtype=bool)
            real_frames_all = list(range(len(real_xy_carla)))
            print(
                f"[METRICS] gt_source=replay: loaded {len(real_xy_carla)} REPLAY.xml waypoints "
                f"({replay_route.times[0]:.1f}s..{replay_route.times[-1]:.1f}s)"
            )

    if gt_source != "replay":
        # Legacy real-data path.
        real_frames_all = list_real_frames(real_dir, real_ego)
        if not real_frames_all:
            print(f"[METRICS] no v2xpnp frames for ego {real_ego}")
            return {}
        real_xy_real, real_mask = load_real_ego_trajectory(real_dir, real_ego, real_frames_all)
        real_xy_carla = real_to_carla_xy(real_xy_real, align_cfg)

    # --- Accumulators ---
    if iou_thresholds is None:
        iou_thresholds = [float(iou_thresh)]
    iou_thresholds = sorted(set(float(t) for t in iou_thresholds))
    result_stat = {t: {"tp": [], "fp": [], "score": [], "gt": 0} for t in iou_thresholds}
    ade_sum = 0.0
    fde_sum = 0.0
    n_traj = 0
    n_collisions = 0
    n_collisions_actor = 0
    n_collisions_static = 0
    n_cr_frames = 0
    n_ap_frames = 0
    stride = max(1, int(round(pred_dt_s / scenario_dt_s)))

    # --- Open CARLA session for AP@50 + static obstacle queries. ---
    own_sess_ctx = None
    if not skip_carla_ap and sess is None:
        try:
            own_sess_ctx = CarlaSceneSession(port=carla_port).__enter__()
            sess = own_sess_ctx
        except Exception as exc:  # noqa: BLE001
            print(f"[METRICS] CARLA session failed ({exc}); AP@50 + static-obstacle CR disabled")
            sess = None
            skip_carla_ap = True

    # --- Cache static map BBs (buildings, trees, poles, fences, signs ...) once. ---
    # These don't change between frames, so we query around the run's mean ego
    # position once and reuse for every waypoint collision check.
    static_obstacles_index = None
    n_static_obstacles = 0
    if sess is not None:
        try:
            valid_xy = real_xy_carla[real_mask]
            cxy = (
                (float(np.mean(valid_xy[:, 0])), float(np.mean(valid_xy[:, 1])))
                if valid_xy.shape[0] > 0 else None
            )
            static_polys = _carla_static_bev_polygons(
                sess.world, sess._carla, center_xy=cxy, radius_m=120.0,
            )
            n_static_obstacles = len(static_polys)
            static_obstacles_index = _build_static_obstacle_index(static_polys)
            print(f"[METRICS] cached {n_static_obstacles} static obstacle BBs "
                  f"(buildings/trees/poles/...) within 120m of run centroid")
        except Exception as exc:  # noqa: BLE001
            print(f"[METRICS] static-obstacle query failed ({exc}); "
                  f"infrastructure collisions disabled")

    # Ego dimensions for CR polygon: prefer live CARLA ego BB after first respawn;
    # fall back to Lincoln MKZ defaults (matches PnP_infer's hardcoded box=[2.45, 1.0]).
    ego_half_length = 2.45
    ego_half_width = 1.0
    ego_dims_resolved = False

    CLOSE_ACTOR_THRESH = 2.6  # drop the v2xpnp ego entry from the actor list (V2X partner)

    try:
        for json_path in json_paths:
            try:
                step = int(json_path.stem)
            except ValueError:
                continue
            try:
                jd = json.loads(json_path.read_text())
            except Exception:  # noqa: BLE001
                continue
            if "lidar_pose_x" not in jd or "waypoints" not in jd:
                continue

            json_lidar_xy = np.array(
                [float(jd["lidar_pose_x"]), float(jd["lidar_pose_y"])],
                dtype=np.float64,
            )

            # --- Match to closest v2xpnp frame, with enough future horizon left ---
            need_future = pred_horizon_steps * stride
            valid = real_mask.copy()
            max_i = len(real_frames_all) - need_future - 1
            if max_i > 0:
                valid[max_i:] = False
            if not np.any(valid):
                continue
            dists_match = np.linalg.norm(real_xy_carla - json_lidar_xy[None, :], axis=1)
            dists_masked = np.where(valid, dists_match, np.inf)
            i_best = int(np.argmin(dists_masked))
            frame_idx = real_frames_all[i_best]
            ego_curr_xy = real_xy_carla[i_best] if real_mask[i_best] else None

            # --- Predicted waypoints in world frame ---
            pred_local_wps = np.array(jd["waypoints"], dtype=np.float64).reshape(-1, 2)
            theta = float(jd.get("theta", 0.0))
            yaw_rad = theta - math.pi / 2.0
            pred_world_wps = local_waypoints_to_world(
                pred_local_wps,
                (float(json_lidar_xy[0]), float(json_lidar_xy[1])),
                yaw_rad,
            )

            # --- ADE / FDE: pred vs future ego positions. ---
            # gt_source="replay" path:
            #   * Future GT comes from REPLAY.xml at frame_idx + (k+1)*stride.
            #   * Apply anchor-shift `gt - gt[0] + pred[0]` so the 0th waypoints
            #     coincide. ADE then measures trajectory-shape error, decoupled
            #     from per-step ego-localization mismatch.
            # gt_source="real" path: original behavior (real YAMLs, no shift).
            future_frames = [
                frame_idx + (k + 1) * stride for k in range(pred_horizon_steps)
            ]
            if gt_source == "replay":
                future_xy_carla = np.full((pred_horizon_steps, 2), np.nan, dtype=np.float64)
                future_mask = np.zeros(pred_horizon_steps, dtype=np.float32)
                for k, fk in enumerate(future_frames):
                    if 0 <= fk < len(real_xy_carla):
                        future_xy_carla[k] = real_xy_carla[fk]
                        future_mask[k] = 1.0
            else:
                future_xy_real, future_mask = load_real_ego_trajectory(
                    real_dir, real_ego, future_frames
                )
                future_xy_carla = real_to_carla_xy(future_xy_real, align_cfg)
            T_min = min(len(pred_world_wps), len(future_xy_carla))
            if T_min == 0:
                continue
            mask = future_mask[:T_min]
            valid_T = mask > 0
            if not np.any(valid_T):
                continue

            # Anchor-shift in replay mode: align GT[0] to Pred[0] before ADE.
            gt_for_metric = future_xy_carla[:T_min].copy()
            if gt_source == "replay" and valid_T[0]:
                gt_for_metric = gt_for_metric - gt_for_metric[0] + pred_world_wps[0]

            dists_pf = np.linalg.norm(
                pred_world_wps[:T_min] - gt_for_metric, axis=1
            )
            dists_valid = dists_pf[valid_T]
            ade = float(dists_valid.mean())
            fde = float(dists_valid[-1])
            ade_sum += ade
            fde_sum += fde
            n_traj += 1

            # --- Collision Rate: BEV polygon intersection over horizon. ---
            # Obstacle list at each future timestep:
            #   - Vehicles from the v2xpnp YAML at frame_idx + (k+1)*stride, with
            #     proper extents + yaw, transformed into the CARLA frame. The ego's
            #     own entry (V2X partner self-collision artefact) is dropped by
            #     centroid proximity to the current ego pose.
            #   - Static map geometry (buildings, trees, poles, signs, walls, fences,
            #     traffic lights, guardrails) cached once from CARLA. Skipped when
            #     CARLA is unavailable.
            # Collision Rate: needs per-frame surrounding-actor poses.
            # In gt_source="replay" mode those poses live only in the real-world
            # yamls (the scenarioset doesn't record per-frame NPC tracks), so CR
            # is skipped and reported as NaN. Closed-loop infraction counts
            # (collisions_vehicle / pedestrian / layout) cover the same concept
            # from the actual CARLA drive — see compute_closed_loop_metrics.
            collision = 0
            collision_kind = ""
            if gt_source != "replay":
                ego_yaws_rad = _yaw_from_trajectory(pred_world_wps[:T_min], json_lidar_xy)
                for k in range(T_min):
                    if not valid_T[k]:
                        continue
                    ego_poly = _bev_polygon(
                        float(pred_world_wps[k, 0]), float(pred_world_wps[k, 1]),
                        float(ego_yaws_rad[k]), ego_half_length, ego_half_width,
                    )

                    # 1) Actors at this future frame.
                    fk = future_frames[k]
                    yaml_p = real_dir / str(real_ego) / f"{fk:06d}.yaml"
                    actor_polys = _yaml_vehicles_bev_polygons(yaml_p, align_cfg)
                    if ego_curr_xy is not None and actor_polys:
                        actor_polys = [
                            (cxy, p) for (cxy, p) in actor_polys
                            if math.hypot(float(cxy[0]) - float(ego_curr_xy[0]),
                                          float(cxy[1]) - float(ego_curr_xy[1])) >= CLOSE_ACTOR_THRESH
                        ]
                    hit_actor = any(ego_poly.intersects(p) for _cxy, p in actor_polys)
                    if hit_actor:
                        collision = 1
                        collision_kind = "actor"
                        break

                    # 2) Static map geometry (buildings/trees/poles/signs/...) — only
                    #    when we have a CARLA session that gave us a static BB index.
                    if static_obstacles_index is not None:
                        cands = _query_static_candidates(static_obstacles_index, ego_poly)
                        hit_static = any(ego_poly.intersects(p) for p in cands)
                        if hit_static:
                            collision = 1
                            collision_kind = "static"
                            break
                n_collisions += collision
                if collision_kind == "actor":
                    n_collisions_actor += 1
                elif collision_kind == "static":
                    n_collisions_static += 1
                n_cr_frames += 1

            # --- AP@iou_thresh: respawn at t_s, read CARLA GT corners ---
            ap_frame_used = False
            if (sess is not None) and _have_eval_utils:
                npz_p = run_dir / f"{step:04d}_pred.npz"
                if npz_p.exists():
                    t_s = float(frame_idx) * scenario_dt_s
                    try:
                        specs = build_actor_spawn_list(carla_dir, t_s)
                        sess.destroy_existing_world_actors()
                        ok, skip = sess.spawn_scene(specs)
                        for _ in range(3):
                            sess.world.tick()
                        # First time we see the ego, snap its real BB extents
                        # so subsequent CR polygon checks use the actual size.
                        if not ego_dims_resolved:
                            for _a in sess.world.get_actors().filter("vehicle.*"):
                                _role = (getattr(_a, "attributes", {}) or {}).get("role_name", "") or ""
                                if _role.lower().startswith("ego") or "hero" in _role.lower():
                                    ext = _a.bounding_box.extent
                                    if ext.x > 0.1 and ext.y > 0.1:
                                        ego_half_length = float(ext.x)
                                        ego_half_width = float(ext.y)
                                        ego_dims_resolved = True
                                        print(f"[METRICS] resolved ego BB: "
                                              f"length={2*ego_half_length:.2f}m  "
                                              f"width={2*ego_half_width:.2f}m  "
                                              f"(role='{_role}')")
                                    break
                        ego_ids = _identify_ego_actor_ids(sess)
                        gt_corners = extract_carla_gt_corners(
                            sess, exclude_actor_ids=ego_ids
                        )
                        data = np.load(str(npz_p))
                        pred_world = np.asarray(data["pred_box_world"], dtype=np.float32)
                        pred_score_arr = np.asarray(data["pred_score"], dtype=np.float32)
                        if pred_world.shape[0] > 0 or gt_corners.shape[0] > 0:
                            gt_t = (
                                torch.from_numpy(gt_corners)
                                if gt_corners.shape[0] > 0
                                else torch.zeros((0, 8, 3), dtype=torch.float32)
                            )
                            pred_t = (
                                torch.from_numpy(pred_world)
                                if pred_world.shape[0] > 0 else None
                            )
                            score_t = (
                                torch.from_numpy(pred_score_arr)
                                if pred_score_arr.shape[0] > 0 else None
                            )
                            for _t in iou_thresholds:
                                eval_utils.caluclate_tp_fp(
                                    pred_t, score_t, gt_t, result_stat, _t
                                )
                            n_ap_frames += 1
                            ap_frame_used = True
                    except Exception as exc:  # noqa: BLE001
                        print(f"[METRICS] AP frame step={step}: {exc}")

            print(
                f"[METRICS] step={step:04d} v2xpnp_frame={frame_idx:06d}  "
                f"ADE={ade:.2f}m  FDE={fde:.2f}m  "
                f"collision={collision}{('('+collision_kind+')') if collision_kind else ''}  "
                f"AP_used={'Y' if ap_frame_used else 'N'}"
            )
    finally:
        if own_sess_ctx is not None:
            try:
                own_sess_ctx.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass

    summary = {
        "gt_source": gt_source,
        "n_eval_frames": int(n_traj),
        "ade_m": (ade_sum / n_traj) if n_traj > 0 else float("nan"),
        "fde_m": (fde_sum / n_traj) if n_traj > 0 else float("nan"),
        "collision_rate": (
            (n_collisions / n_cr_frames) if n_cr_frames > 0 else float("nan")
        ),
        "n_collisions": int(n_collisions),
        "n_collisions_actor": int(n_collisions_actor),
        "n_collisions_static": int(n_collisions_static),
        "n_cr_frames": int(n_cr_frames),
        "n_static_obstacles_cached": int(n_static_obstacles),
        "ego_half_length_m": float(ego_half_length),
        "ego_half_width_m": float(ego_half_width),
        "ego_dims_from_carla": bool(ego_dims_resolved),
        "ap_iou_thresholds": list(iou_thresholds),
        "ap_per_iou": {},                # filled below: {"0.30": 0.123, ...}
        "ap_n_frames": int(n_ap_frames),
        "ap_n_predictions_per_iou": {},
        "ap_n_gt_total": 0,
    }
    if n_ap_frames > 0 and _have_eval_utils:
        for _t in iou_thresholds:
            ap, _mrec, _mpre = eval_utils.calculate_ap(result_stat, _t)
            summary["ap_per_iou"][f"{_t:.2f}"] = float(ap)
            summary["ap_n_predictions_per_iou"][f"{_t:.2f}"] = int(len(result_stat[_t]["tp"]))
        summary["ap_n_gt_total"] = int(result_stat[iou_thresholds[0]]["gt"])
    else:
        for _t in iou_thresholds:
            summary["ap_per_iou"][f"{_t:.2f}"] = float("nan")
            summary["ap_n_predictions_per_iou"][f"{_t:.2f}"] = 0
    # Backward-compat: keep ap50/ap_iou_thresh keys populated against 0.5 if present.
    if 0.5 in iou_thresholds:
        summary["ap50"] = summary["ap_per_iou"]["0.50"]
        summary["ap_iou_thresh"] = 0.5

    cr_str = (
        f"{summary['collision_rate']:.3f}"
        if not (isinstance(summary["collision_rate"], float)
                and math.isnan(summary["collision_rate"]))
        else "NaN(skipped — gt_source=replay)"
    )
    ap_str = "  ".join(f"AP@{t:.2f}={summary['ap_per_iou'][f'{t:.2f}']:.3f}"
                       for t in iou_thresholds)
    print(
        f"[METRICS] gt_source={gt_source}  mean ADE={summary['ade_m']:.3f}m  "
        f"mean FDE={summary['fde_m']:.3f}m  CR={cr_str}  "
        f"({ap_str})  n_eval={n_traj}  ap_n={n_ap_frames}"
    )
    if save_path is not None:
        save_path.write_text(json.dumps(summary, indent=2))
        print(f"[METRICS] wrote {save_path}")
    return summary


# ------------------------------------------------------------------ #
# Main                                                               #
# ------------------------------------------------------------------ #


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", default="2023-03-17-16-03-02_11_0")
    p.add_argument("--real-ego", type=int, default=1, choices=(1, 2))
    p.add_argument(
        "--frame-idx",
        type=int,
        default=-1,
        help="-1 = auto-pick first moving frame",
    )
    p.add_argument("--horizon-s", type=float, default=DEFAULT_HORIZON_S)
    p.add_argument("--snap-radius-m", type=float, default=DEFAULT_SNAP_RADIUS_M)
    p.add_argument(
        "--out-dir",
        default="/data2/marco/CoLMDriver/openloop/runresults/carla_prototype",
    )
    p.add_argument(
        "--no-carla", action="store_true",
        help="Skip the live-CARLA scene spawning + camera capture step.",
    )
    p.add_argument(
        "--topdown-altitude-m", type=float, default=40.0,
        help="Height of the top-down RGB camera above the ego (meters).",
    )
    p.add_argument(
        "--live-codriving", action="store_true",
        help=(
            "Run the codriving agent in CARLA for one prediction step (--no-start-carla "
            "--timeout 10). Requires an already-running CARLA on --carla-port (e.g. from "
            "the scenario pool). After the eval exits, the same CARLA is reused for scene capture."
        ),
    )
    p.add_argument(
        "--carla-port", type=int, default=2000,
        help="Port of a running CARLA server (used for both --live-codriving and scene capture).",
    )
    p.add_argument(
        "--use-cached-prediction", action="store_true",
        help=(
            "Load the most recent cached codriving prediction for this scenario instead of "
            "running a fresh eval. By default no prediction is used unless --live-codriving "
            "is passed."
        ),
    )
    p.add_argument(
        "--start-carla", action="store_true",
        help=(
            "Launch carla912/CarlaUE4.sh for the scene capture step (--no-carla not set). "
            "Not needed for --live-codriving — run_custom_eval handles that CARLA itself."
        ),
    )
    p.add_argument(
        "--carla-adapter", type=int, default=0,
        help=(
            "Vulkan -graphicsadapter index passed to CarlaUE4.sh when --start-carla is used. "
            "Required on multi-GPU machines — CARLA exits immediately without it. "
            "Check existing CARLA processes for which indices are in use."
        ),
    )
    p.add_argument(
        "--no-metrics", action="store_true",
        help=(
            "Skip the multi-frame AP@50 + ADE + FDE + Collision Rate sweep that "
            "runs by default whenever a codriving run_dir is available. Useful when "
            "CARLA is offline (AP@50 needs CARLA; the rest don't, but if --no-carla "
            "is set AP@50 is auto-skipped anyway)."
        ),
    )
    p.add_argument(
        "--planner", default="codriving",
        help=(
            "Leaderboard planner alias to live-infer when --live-planner is set. "
            "Must match an entry in run_custom_eval's PLANNER_SPECS "
            "(e.g. codriving, perception_swap_fcooper, perception_swap_attfuse). "
            "Default 'codriving' matches legacy --live-codriving behaviour."
        ),
    )
    p.add_argument(
        "--live-planner", action="store_true",
        help=(
            "Run the configured --planner in CARLA for one prediction step "
            "(generalisation of --live-codriving for any registered alias). "
            "Mutually exclusive with --live-codriving."
        ),
    )
    p.add_argument(
        "--gt-source", choices=["replay", "real"], default="replay",
        help=(
            "Source of the ego ground-truth trajectory used for ADE/FDE. "
            "'replay' (default, paper-defensible): parse REPLAY.xml from the "
            "scenarioset — fully CARLA-frame, no real-world dataset ingest. "
            "Anchor-shifts so prediction[0] coincides with GT[0]. "
            "'real' (legacy): read v2xpnp YAMLs and soft-snap onto the route."
        ),
    )
    p.add_argument(
        "--iou-thresholds", default="0.3,0.5,0.7",
        help=(
            "Comma-separated BEV-IoU thresholds for AP. Defaults to 0.3,0.5,0.7 "
            "(OPV2V/HEAL convention). Pass '0.5' for legacy AP@50-only behaviour."
        ),
    )
    args = p.parse_args()
    iou_thresholds = sorted({float(x) for x in args.iou_thresholds.split(",") if x.strip()})

    scenario = args.scenario
    real_ego = int(args.real_ego)
    carla_ego_slot = REAL_TO_CARLA_EGO[real_ego]
    horizon_s = float(args.horizon_s)
    horizon_steps = int(round(horizon_s / SCENARIO_DT_S))

    # Paths
    real_dir = find_real_scenario_dir(scenario)
    carla_dir = CARLA_SCENARIOSET_ROOT / scenario
    if not carla_dir.is_dir():
        raise FileNotFoundError(f"CARLA scenario dir not found: {carla_dir}")

    align_cfg = AlignCfg.load(CARLA_ALIGN_JSON)
    print(f"[INFO] align cfg: {align_cfg}")

    # Optionally launch CARLA in the background before everything else.
    _carla_proc = None
    if args.start_carla:
        print(f"[INFO] --start-carla: launching CARLA on port {args.carla_port} ...")
        try:
            _carla_proc = start_carla_server(port=args.carla_port, graphics_adapter=args.carla_adapter)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"[ERROR] Failed to start CARLA: {exc}") from exc
        import atexit as _atexit  # noqa: PLC0415
        _atexit.register(lambda: _carla_proc.terminate())

    # Pre-compute real ego speed for the target frame so we can inject it into CARLA
    # before the first codriving inference step.  The dataset's ego_speed field is
    # always 0 (data-recording bug), so derive speed from the pose delta instead.
    _prelim_frame = args.frame_idx if args.frame_idx >= 0 else pick_start_frame_by_motion(real_dir, real_ego)
    _prelim_yaml = load_real_frame_yaml(real_dir, real_ego, _prelim_frame)
    _prelim_real_xy = np.array(_prelim_yaml["lidar_pose"][:2], dtype=np.float64)
    _prelim_yaw = float(_prelim_yaml["lidar_pose"][4])
    _prelim_yaw_carla = real_yaw_to_carla_yaw(_prelim_yaw, align_cfg)
    _initial_ego_speed_mps = 0.0
    if _prelim_frame > 0:
        try:
            _prev_yaml = load_real_frame_yaml(real_dir, real_ego, _prelim_frame - 1)
            _prev_xy = np.array(_prev_yaml["lidar_pose"][:2], dtype=np.float64)
            _dx, _dy = _prelim_real_xy[0] - _prev_xy[0], _prelim_real_xy[1] - _prev_xy[1]
            _initial_ego_speed_mps = math.sqrt(_dx * _dx + _dy * _dy) / SCENARIO_DT_S
        except Exception:  # noqa: BLE001
            pass
    print(
        f"[INFO] real ego speed at frame {_prelim_frame}: "
        f"{_initial_ego_speed_mps:.2f} m/s ({_initial_ego_speed_mps * 2.237:.1f} mph) "
        f"— will inject into CARLA for live-codriving"
    )

    # Determine which codriving run_dir to use for predictions.
    # --live-codriving: run a fresh eval; only that new dir is used (never cached).
    # --use-cached-prediction: load the most recent cached dir.
    # Neither flag: proceed without any prediction.
    run_dir: Optional[Path] = None
    # --live-planner is the generalised form; --live-codriving stays as a
    # back-compat alias that pins planner=codriving regardless of --planner.
    if args.live_codriving and args.live_planner and args.planner != "codriving":
        raise SystemExit(
            "--live-codriving and --live-planner with non-codriving --planner are "
            "mutually exclusive; pass only one"
        )
    _live_planner_alias = (
        "codriving" if args.live_codriving else (args.planner if args.live_planner else None)
    )
    if _live_planner_alias is not None:
        print(f"[INFO] live planner inference: planner={_live_planner_alias}")
        run_dir = run_planner_evaluation(
            scenario,
            planner=_live_planner_alias,
            carla_port=args.carla_port,
            initial_ego_speed_mps=_initial_ego_speed_mps,
            initial_ego_yaw_deg=_prelim_yaw_carla,
        )
        if run_dir is None:
            print(f"[WARN] live planner ({_live_planner_alias}): eval produced no new prediction; skipping CARLA capture too")
            args.no_carla = True  # CARLA is clearly unreachable; don't waste 60s timing out
        else:
            print(f"[INFO] fresh {_live_planner_alias} run dir: {run_dir}")
    elif args.use_cached_prediction:
        # Use --planner (default codriving) to pick which planner's results to load.
        run_dir = find_planner_run_dir(args.planner, scenario, ego_slot=carla_ego_slot)
        if run_dir is not None:
            print(f"[INFO] --use-cached-prediction: using {run_dir}")
            # Warn if cached predictions have speed=0 (data-recording bug + CARLA
            # starts at rest means old predictions will show too-short waypoints).
            _sample_json = next(iter(sorted(run_dir.glob("*.json"))), None)
            if _sample_json is not None:
                try:
                    _spd = float(json.loads(_sample_json.read_text()).get("speed", -1))
                    if abs(_spd) < 0.01:
                        print(
                            f"[WARN] cached prediction has speed={_spd:.4f} m/s — "
                            f"model received speed≈0 and will predict short waypoints. "
                            f"Re-run with --live-codriving to get speed-corrected predictions."
                        )
                except Exception:  # noqa: BLE001
                    pass
        else:
            print("[WARN] --use-cached-prediction: no cached run dir found; proceeding without prediction")
    else:
        print("[INFO] no prediction source (use --live-codriving or --use-cached-prediction)")

    # Pick frame: jointly pick (real_frame, codriving_json) so ego anchors coincide.
    frame_idx = args.frame_idx
    preselected_codriving_json: Optional[Path] = None
    if frame_idx < 0 and run_dir is not None:
        frame_idx, preselected_codriving_json, pair_d = joint_pick_frame_and_codriving_json(
            real_dir, real_ego, run_dir, align_cfg,
            horizon_steps=int(round(horizon_s / SCENARIO_DT_S)),
        )
        print(
            f"[INFO] jointly picked frame_idx={frame_idx}  "
            f"codriving_json={preselected_codriving_json.name}  Δego={pair_d:.2f}m"
        )
    elif frame_idx < 0:
        frame_idx = pick_start_frame_by_motion(real_dir, real_ego)
        print(f"[INFO] auto-picked frame_idx={frame_idx} (first moving frame)")

    # --- current ego pose from real YAML, transformed to CARLA --- #
    curr_yaml = load_real_frame_yaml(real_dir, real_ego, frame_idx)
    curr_real_xy = np.array(curr_yaml["lidar_pose"][:2], dtype=np.float64)
    curr_real_yaw = float(curr_yaml["lidar_pose"][4])
    ego_xy = real_to_carla_xy(curr_real_xy, align_cfg).squeeze()
    ego_yaw_deg = real_yaw_to_carla_yaw(curr_real_yaw, align_cfg)
    print(
        f"[INFO] ego real=({curr_real_xy[0]:.2f},{curr_real_xy[1]:.2f}) "
        f"yaw={curr_real_yaw:.2f} -> CARLA=({ego_xy[0]:.2f},{ego_xy[1]:.2f}) yaw={ego_yaw_deg:.2f}"
    )

    # --- real GT future trajectory --- #
    future_frames = list(range(frame_idx + 1, frame_idx + 1 + horizon_steps))
    gt_real, gt_mask = load_real_ego_trajectory(real_dir, real_ego, future_frames)
    if gt_mask.sum() == 0:
        raise SystemExit(f"No valid future frames after {frame_idx}")
    gt_real = gt_real[gt_mask]
    gt_carla = real_to_carla_xy(gt_real, align_cfg)
    print(f"[INFO] loaded {len(gt_carla)} real future frames -> CARLA coords")

    # --- CARLA route plan --- #
    replay_xml = carla_dir / f"ucla_v2_custom_ego_vehicle_{carla_ego_slot}_REPLAY.xml"
    goal_xml = carla_dir / f"ucla_v2_custom_ego_vehicle_{carla_ego_slot}.xml"
    route_full = parse_carla_route_xml(replay_xml)
    route_goal = parse_carla_route_xml(goal_xml)
    print(
        f"[INFO] CARLA route: {len(route_full.xy)} replay wps "
        f"({route_full.times[0]:.2f}s..{route_full.times[-1]:.2f}s) "
        f"+ {len(route_goal.xy)} sparse goals"
    )

    # --- surrounding actor GT futures (for overlay + context) --- #
    surr_ids, surrounding_trajs = load_real_surrounding_trajectories(
        real_dir, real_ego, future_frames, align_cfg
    )
    if surr_ids:
        print(f"[INFO] loaded {len(surr_ids)} surrounding actor GT tracks")

    # --- alignment --- #
    gt_aligned, projections, blend_w = align_gt_to_route(
        gt_carla, route_full.xy, snap_radius_m=args.snap_radius_m
    )
    ade_raw_vs_route = ade(gt_carla, projections)
    ade_aligned_vs_route = ade(gt_aligned, projections)
    print(
        f"[INFO] alignment: mean raw→route dist={ade_raw_vs_route:.2f}m  "
        f"aligned→route dist={ade_aligned_vs_route:.2f}m  "
        f"mean blend weight={blend_w.mean():.2f}"
    )

    # --- codriving prediction --- #
    codriving_world = np.zeros((0, 2))
    codriving_pred: Optional[CodrivingPrediction] = None
    ade_codriving = float("nan")

    if run_dir is None:
        print("[WARN] no codriving run dir; proceeding without prediction")
    else:
        try:
            if preselected_codriving_json is not None:
                codriving_pred = _load_single_codriving_json(preselected_codriving_json)
            else:
                codriving_pred = load_codriving_prediction_closest_to(run_dir, ego_xy)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] could not load codriving prediction: {exc}")

    if codriving_pred is not None:
        codriving_world = codriving_pred.waypoints_world
        delta_m = float(np.linalg.norm(codriving_pred.lidar_xy - ego_xy))
        print(
            f"[INFO] codriving prediction: step={codriving_pred.step_counter}  "
            f"pred_ego=({codriving_pred.lidar_xy[0]:.2f},{codriving_pred.lidar_xy[1]:.2f})  "
            f"Δego={delta_m:.2f}m  T={len(codriving_world)} waypoints"
        )
        n_pred = codriving_world.shape[0]
        # GT is at 0.1s; codriving at 0.2s → every 2nd GT point, up to n_pred
        gt_resampled = gt_aligned[1::2][:n_pred]
        ade_codriving = ade(codriving_world, gt_resampled)
    else:
        print("[WARN] no codriving prediction available; proceeding without it")

    # Visual alignment offset: shift gt_aligned so waypoint[0] coincides with
    # prediction waypoint[0], correcting the slight ego-position mismatch between
    # the real dataset frame and codriving's LiDAR pose. Only used for visualization;
    # ADE metrics above are computed with the original gt_aligned.
    gt_aligned_viz = gt_aligned.copy()
    if codriving_pred is not None and len(codriving_world) > 0:
        gt_aligned_viz = gt_aligned + (codriving_world[0] - gt_aligned[0])

    # --- render matplotlib --- #
    out_dir = Path(args.out_dir)
    overview_path = out_dir / f"prototype_{scenario}_ego{real_ego}_f{frame_idx:06d}.png"
    detail_path = out_dir / f"prototype_{scenario}_ego{real_ego}_f{frame_idx:06d}_detail.png"
    info_path = out_dir / f"prototype_{scenario}_ego{real_ego}_f{frame_idx:06d}.json"

    render_topdown(
        overview_path,
        scenario=scenario,
        frame_idx=frame_idx,
        horizon_s=horizon_s,
        ego_xy=ego_xy,
        route_full=route_full.xy,
        route_goal=route_goal.xy,
        gt_raw=gt_carla,
        gt_aligned=gt_aligned_viz,
        projections=projections,
        codriving_world=codriving_world,
        codriving_lidar_xy=(codriving_pred.lidar_xy if codriving_pred is not None else ego_xy),
        ade_raw=ade_raw_vs_route,
        ade_aligned=ade_aligned_vs_route,
        ade_codriving=ade_codriving,
    )
    render_topdown(
        detail_path,
        scenario=scenario,
        frame_idx=frame_idx,
        horizon_s=horizon_s,
        ego_xy=ego_xy,
        route_full=route_full.xy,
        route_goal=route_goal.xy,
        gt_raw=gt_carla,
        gt_aligned=gt_aligned_viz,
        projections=projections,
        codriving_world=codriving_world,
        codriving_lidar_xy=(codriving_pred.lidar_xy if codriving_pred is not None else ego_xy),
        ade_raw=ade_raw_vs_route,
        ade_aligned=ade_aligned_vs_route,
        ade_codriving=ade_codriving,
        zoom_radius_m=max(15.0, horizon_s * 10.0),
    )
    alignment_path = out_dir / f"prototype_{scenario}_ego{real_ego}_f{frame_idx:06d}_alignment.png"
    render_alignment_detail(
        alignment_path,
        scenario=scenario,
        frame_idx=frame_idx,
        ego_xy=ego_xy,
        route_full=route_full.xy,
        gt_raw=gt_carla,
        gt_aligned=gt_aligned,
        projections=projections,
        blend_weights=blend_w,
    )
    print(f"[OK] wrote {overview_path}")
    print(f"[OK] wrote {detail_path}")
    print(f"[OK] wrote {alignment_path}")

    # --- info sidecar --- #
    info = {
        "scenario": scenario,
        "real_ego": real_ego,
        "carla_ego_slot": carla_ego_slot,
        "frame_idx": frame_idx,
        "horizon_s": horizon_s,
        "horizon_steps": horizon_steps,
        "scenario_dt_s": SCENARIO_DT_S,
        "align_cfg": {
            "tx": align_cfg.tx,
            "ty": align_cfg.ty,
            "theta_deg": align_cfg.theta_deg,
            "flip_y": align_cfg.flip_y,
        },
        "ego_real_xy": curr_real_xy.tolist(),
        "ego_carla_xy": ego_xy.tolist(),
        "ego_carla_yaw_deg": ego_yaw_deg,
        "gt_n_future_points": int(gt_carla.shape[0]),
        "ade_gt_raw_vs_route_m": ade_raw_vs_route,
        "ade_gt_aligned_vs_route_m": ade_aligned_vs_route,
        "mean_blend_weight": float(blend_w.mean()),
        "snap_radius_m": args.snap_radius_m,
        "codriving": (
            {
                "step_counter": codriving_pred.step_counter if codriving_pred is not None else None,
                "source_json": str(codriving_pred.source_json) if codriving_pred is not None else None,
                "pred_ego_carla_xy": codriving_pred.lidar_xy.tolist() if codriving_pred is not None else None,
                "n_waypoints": int(codriving_world.shape[0]),
                "ade_vs_aligned_gt_m": ade_codriving,
                "delta_ego_m": float(np.linalg.norm(codriving_pred.lidar_xy - ego_xy)) if codriving_pred is not None else None,
            }
            if codriving_pred is not None
            else None
        ),
        "output_png": str(overview_path),
        "output_detail_png": str(detail_path),
    }
    info_path.write_text(json.dumps(info, indent=2))
    print(f"[OK] wrote {info_path}")

    # --- front-camera image from codriving agent's own RGB sensor --- #
    # The codriving agent saves a composite JPG (front view + lidar visualizations)
    # alongside each prediction JSON.  We crop out the front-view portion directly.
    if preselected_codriving_json is not None:
        front_stem = out_dir / f"prototype_{scenario}_ego{real_ego}_f{frame_idx:06d}"
        front_target = front_stem.with_name(front_stem.name + "_carla_front.png")
        front_png = _extract_front_view_from_run_jpg(
            run_dir,
            preselected_codriving_json,
            front_target,
        )
        if front_png is not None:
            print(f"[OK] wrote {front_png}  (front camera from codriving agent)")
        else:
            print("[WARN] front camera JPG not found alongside codriving prediction JSON")

    # --- Metrics sweep (AP@50, ADE, FDE, Collision Rate) over all saved frames --- #
    if not args.no_metrics:
        if run_dir is None:
            print(
                "[METRICS] no codriving run_dir; pass --live-codriving or "
                "--use-cached-prediction to enable AP@50/ADE/FDE/CR"
            )
        else:
            metrics_path = out_dir / f"prototype_{scenario}_ego{real_ego}_f{frame_idx:06d}_metrics.json"
            try:
                compute_metrics_from_run_dir(
                    run_dir=run_dir,
                    real_dir=real_dir,
                    real_ego=real_ego,
                    align_cfg=align_cfg,
                    carla_dir=carla_dir,
                    carla_port=args.carla_port,
                    scenario_dt_s=SCENARIO_DT_S,
                    save_path=metrics_path,
                    skip_carla_ap=args.no_carla,
                    gt_source=args.gt_source,
                    iou_thresholds=iou_thresholds,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[METRICS] failed: {exc}")

    # --- live CARLA scene + top-down capture --- #
    if not args.no_carla:
        carla_scene_png, carla_gt_png, carla_overlay_png, carla_status = live_carla_capture(
            carla_dir=carla_dir,
            target_t_s=float(frame_idx) * SCENARIO_DT_S,
            ego_xy=ego_xy,
            ego_yaw_deg=ego_yaw_deg,
            route_full=route_full.xy,
            gt_raw=gt_carla,
            gt_aligned=gt_aligned_viz,
            codriving_world=codriving_world,
            altitude_m=args.topdown_altitude_m,
            out_stem=out_dir / f"prototype_{scenario}_ego{real_ego}_f{frame_idx:06d}",
            carla_port=args.carla_port,
        )
        print(f"[CARLA] {carla_status}")
        if carla_scene_png and Path(carla_scene_png).exists():
            print(f"[OK] wrote {carla_scene_png}")
        if carla_gt_png and Path(carla_gt_png).exists():
            print(f"[OK] wrote {carla_gt_png}")
        if carla_overlay_png and Path(carla_overlay_png).exists():
            print(f"[OK] wrote {carla_overlay_png}")


if __name__ == "__main__":
    main()
