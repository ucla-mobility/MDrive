"""
planner_trajectory_adapter.py
=============================
Planner-agnostic trajectory extraction for open-loop evaluation.

This module extracts the planner's *intended future trajectory* from
accessible agent attributes **without modifying any planner code**.

The evaluator previously relied on kinematic control rollout, which
conflates control-to-trajectory conversion errors with the actual
planning quality.  This adapter reads the planner's own future
trajectory (when available) and converts it to a standardised
``PlannerOutput`` in the world frame.

Extraction priority per planner
-------------------------------
1. ``agent.last_planned_waypoints_world`` — world-frame, used directly.
2. ``agent.pid_metadata['plan']`` — ego-local trajectory (VAD, UniAD).
3. ``agent.pid_metadata['all_plan']`` — command-indexed trajectories (VAD).
4. ``agent.pid_metadata`` wp_1…wp_N — individual waypoints (TCP).
5. Fallback → ``None``  (evaluator uses control rollout).

Coordinate convention
---------------------
Ego-local frames follow the CARLA BEV convention used in all
Bench2Drive agents:  x = forward, y = left.  World conversion uses::

    theta = compass + π/2
    R     = [[cos θ, −sin θ], [sin θ, cos θ]]
    world = ego_xy + R @ local_xy

Time alignment
--------------
The evaluator compares at 2 Hz (0.5 s steps) for 3 s → 6 future
steps.  Adapters resample the planner's native resolution to exactly
6 world-frame waypoints at t+0.5, t+1.0, … t+3.0 s.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

# ── Standard output ──────────────────────────────────────────

@dataclass
class PlannerOutput:
    """Standardised trajectory output from a planner adapter.

    Attributes
    ----------
    future_trajectory_world : optional [T, 2] array
        Planned ego positions in world (x, y) at 0.5 s intervals,
        corresponding to t+1 … t+T (T = ``EVAL_HORIZON_STEPS``).
        ``None`` when the planner does not expose a trajectory.
    traj_source : str
        Human-readable label for provenance logging.
    raw_length : int
        Number of waypoints available *before* resampling.
    raw_world_wps : optional [N, 2] array
        The planner's waypoints in world frame **before** any resampling or
        interpolation.  N equals the planner's native prediction count and the
        time-step matches the planner's native rate.  ``None`` when the adapter
        cannot produce a trajectory.
    native_dt : float
        Seconds between consecutive ``raw_world_wps`` waypoints (planner's
        own time step, e.g. 0.1 s for ColMDriver, 0.05 s for CoDriving,
        0.5 s for VAD/UniAD/TCP/LMDrive).  Defaults to EVAL_DT (0.5 s).
    raw_export : optional RawTrajectoryExport
        Native-space pre-resampling trajectory payload for Stage 1 debugging.
    debug : optional dict
        Adapter-specific diagnostic payload to help trace extraction issues.
    """
    future_trajectory_world: Optional[np.ndarray] = None
    traj_source: str = "none"
    raw_length: int = 0
    raw_world_wps: Optional[np.ndarray] = None
    native_dt: float = 0.5   # seconds between consecutive raw_world_wps; set per-adapter
    raw_export: Optional["RawTrajectoryExport"] = None
    debug: Optional[Dict[str, Any]] = None


@dataclass
class RawTrajectoryExport:
    """Planner-native trajectory payload written before resampling/scoring."""

    raw_positions: Optional[np.ndarray] = None
    source_frame_description: str = "unknown"
    axis_convention_description: str = "unknown"
    is_cumulative_positions: Optional[bool] = None
    point0_mode: str = "unknown"
    native_timestamps: Optional[np.ndarray] = None
    native_dt: Optional[float] = None
    adapter_debug_notes: list[str] = field(default_factory=list)
    freshness_token: Optional[Any] = None


_AUTO_NATIVE_TIMESTAMPS = object()


def _native_timestamps(
    n_points: int,
    native_dt: Optional[float],
    point0_mode: str,
) -> Optional[np.ndarray]:
    """Return best-effort native timestamps without fabricating unknown semantics."""
    if native_dt is None or n_points <= 0:
        return None
    if point0_mode == "future":
        return np.asarray(
            [float(native_dt) * float(i) for i in range(1, n_points + 1)],
            dtype=np.float64,
        )
    if point0_mode == "current":
        return np.asarray(
            [float(native_dt) * float(i) for i in range(n_points)],
            dtype=np.float64,
        )
    return None


def _build_raw_export(
    *,
    raw_positions: Optional[np.ndarray],
    source_frame_description: str,
    axis_convention_description: str,
    is_cumulative_positions: Optional[bool],
    point0_mode: str,
    native_dt: Optional[float],
    native_timestamps: Any = _AUTO_NATIVE_TIMESTAMPS,
    adapter_debug_notes: Optional[list[str]] = None,
    freshness_token: Optional[Any] = None,
) -> RawTrajectoryExport:
    arr = None if raw_positions is None else np.asarray(raw_positions, dtype=np.float64)
    n_points = int(len(arr)) if arr is not None and arr.ndim >= 1 else 0
    if native_timestamps is _AUTO_NATIVE_TIMESTAMPS:
        timestamps = _native_timestamps(n_points, native_dt, point0_mode)
    else:
        timestamps = None if native_timestamps is None else np.asarray(native_timestamps, dtype=np.float64)
    return RawTrajectoryExport(
        raw_positions=arr,
        source_frame_description=str(source_frame_description),
        axis_convention_description=str(axis_convention_description),
        is_cumulative_positions=is_cumulative_positions,
        point0_mode=str(point0_mode),
        native_timestamps=timestamps,
        native_dt=(None if native_dt is None else float(native_dt)),
        adapter_debug_notes=list(adapter_debug_notes or []),
        freshness_token=freshness_token,
    )


# ── Constants ────────────────────────────────────────────────

EVAL_HORIZON_STEPS = 6       # 3 s at 2 Hz
EVAL_DT = 0.5                # seconds between evaluation waypoints
MAX_REASONABLE_FIRST_WP_OFFSET_M = 50.0
DEGENERATE_TRAJ_EPS_M = 1e-4
TRAJECTORY_EXTRAPOLATION_ENABLED = True


def set_trajectory_extrapolation(enabled: bool) -> None:
    """Globally enable/disable linear extrapolation beyond native planner horizon."""
    global TRAJECTORY_EXTRAPOLATION_ENABLED
    TRAJECTORY_EXTRAPOLATION_ENABLED = bool(enabled)


# ── Resampling / alignment helpers ───────────────────────────

def _resample_trajectory(
    traj: np.ndarray,
    src_dt: float,
    dst_dt: float = EVAL_DT,
    n_steps: int = EVAL_HORIZON_STEPS,
) -> np.ndarray:
    """Resample a trajectory from *src_dt* spacing to *dst_dt* spacing.

    Returns exactly *n_steps* waypoints corresponding to
    t+dst_dt, t+2*dst_dt, …, t+n_steps*dst_dt.

    For source waypoints indexed from 1 (i.e. the first waypoint is at
    t+src_dt), the target time ``t_k = (k+1)*dst_dt`` maps to source
    fractional index ``idx = t_k / src_dt − 1``.

    * If *idx* falls between two source points → linear interpolation.
    * If *idx* exceeds the source length:
      - extrapolation enabled  → linear extrapolation from the last two
        source points (constant-velocity assumption).
      - extrapolation disabled → hold the last waypoint constant.
    """
    if traj.ndim != 2 or traj.shape[1] < 2:
        raise ValueError(f"Expected [N, >=2] trajectory, got {traj.shape}")

    n_src = len(traj)
    out = np.empty((n_steps, 2), dtype=np.float64)

    for k in range(n_steps):
        t_target = (k + 1) * dst_dt          # future time in seconds
        # Fractional 0-based index into source array whose first entry
        # represents t=src_dt.
        frac_idx = t_target / src_dt - 1.0

        if frac_idx < 0:
            # Target time is before the first source waypoint.
            if TRAJECTORY_EXTRAPOLATION_ENABLED and n_src >= 2:
                # Backward extrapolation from first two source points.
                direction = traj[0] - traj[1]
                out[k] = traj[0] + direction * (-frac_idx)
            else:
                # No extrapolation: clamp to earliest waypoint.
                out[k] = traj[0]
        elif frac_idx >= n_src - 1:
            # Beyond source horizon.
            if TRAJECTORY_EXTRAPOLATION_ENABLED and n_src >= 2:
                # Linear extrapolation.
                vel = traj[-1] - traj[-2]
                over = frac_idx - (n_src - 1)
                out[k] = traj[-1] + vel * over
            else:
                # No extrapolation: hold last known waypoint.
                out[k] = traj[-1]
        else:
            lo = int(math.floor(frac_idx))
            hi = lo + 1
            alpha = frac_idx - lo
            out[k] = traj[lo] * (1 - alpha) + traj[hi] * alpha

    return out


def _ego_local_to_world(
    local_wps: np.ndarray,
    ego_world_xy: np.ndarray,
    compass_rad: float,
) -> np.ndarray:
    """Convert ego-local (forward-x, left-y) waypoints to world frame.

    Uses the CARLA BEV convention:  theta = compass + π/2.
    """
    theta = compass_rad + math.pi / 2.0
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    R = np.array([[cos_t, -sin_t],
                   [sin_t,  cos_t]], dtype=np.float64)
    return ego_world_xy + (R @ local_wps[:, :2].T).T


def _has_finite_xy(arr: np.ndarray) -> bool:
    """Return True when all first-two-dimension waypoint values are finite."""
    return np.isfinite(arr[:, :2]).all()


def _is_degenerate_trajectory(arr: np.ndarray, eps: float = DEGENERATE_TRAJ_EPS_M) -> bool:
    """Return True when all waypoints are effectively identical."""
    if len(arr) <= 1:
        return True
    step_norm = np.linalg.norm(np.diff(arr[:, :2], axis=0), axis=1)
    return bool(np.max(step_norm) <= eps)


def _first_wp_offset_m(world_wps: np.ndarray, ego_world_xy: np.ndarray) -> float:
    """Distance from ego position to first predicted waypoint in world frame."""
    if len(world_wps) == 0:
        return float("inf")
    return float(np.linalg.norm(world_wps[0, :2] - ego_world_xy[:2]))


def _latest_json_record(
    directory: Optional[Path],
    pattern: str,
) -> Optional[Dict[str, Any]]:
    """Load the most recent JSON record matching *pattern* from *directory*."""
    if directory is None:
        return None
    try:
        path = Path(directory)
    except Exception:
        return None
    if not path.exists() or not path.is_dir():
        return None
    try:
        candidates = list(path.glob(pattern))
    except Exception:
        return None
    if not candidates:
        return None
    try:
        latest = max(
            candidates,
            key=lambda p: (p.stat().st_mtime_ns, p.name),
        )
    except Exception:
        latest = sorted(candidates)[-1]
    try:
        with latest.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def _coerce_waypoints(value: Any) -> Optional[np.ndarray]:
    """Best-effort conversion to a finite [N,2] waypoint array."""
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1, 2)
    except Exception:
        return None
    if len(arr) < 2:
        return None
    if not _has_finite_xy(arr):
        return None
    return arr


def _pick_first_waypoint_array(value: Any) -> Optional[np.ndarray]:
    """Extract first usable waypoint array from scalar/list/dict containers."""
    arr = _coerce_waypoints(value)
    if arr is not None:
        return arr
    if isinstance(value, dict):
        for key in (0, "0", 1, "1"):
            if key in value:
                arr = _coerce_waypoints(value.get(key))
                if arr is not None:
                    return arr
        for candidate in value.values():
            arr = _coerce_waypoints(candidate)
            if arr is not None:
                return arr
    if isinstance(value, (list, tuple)):
        for candidate in value:
            arr = _coerce_waypoints(candidate)
            if arr is not None:
                return arr
    return None


# ── Per-planner adapter functions ────────────────────────────
#
# Signature:  adapter(agent, ego_world_xy, compass_rad, **kw) → PlannerOutput
#
# Each reads *only* from public/accessible agent attributes.
# NO planner code is modified.

def _adapt_colmdriver(
    agent: Any,
    ego_world_xy: np.ndarray,
    compass_rad: float,
    **kw,
) -> PlannerOutput:
    """CoLMDriver exposes ``last_planned_waypoints_world`` [N, 2] at 0.1 s.

    Coordinate correction
    ---------------------
    CoLMDriver builds its planning_bank waypoints using the agent's internal
    GPS-mapped coordinates where the ego origin is
        (measurements['x'], measurements['y']) = (world_y, -world_x)
    rather than the true lidar_pose world frame (world_x, world_y).

    The stored waypoints are absolute positions in that *agent* frame:
        colmd_wp = [world_y + delta_x, -world_x + delta_y]

    To recover the true world-frame position (world_x + delta_x,
    world_y + delta_y) we add the per-frame offset:
        true_wp[:, 0] = colmd_wp[:, 0] + (world_x - world_y)
        true_wp[:, 1] = colmd_wp[:, 1] + (world_x + world_y)

    ego_world_xy = [world_x, world_y] is passed in by the evaluator from
    pose[:2] (lidar_pose), so this correction is exact.
    """
    planner_label = str(kw.get("planner_name") or "colmdriver").lower()
    planner_family_note = (
        "CoLMDriver_Rulebase shares the same planning_bank extraction path under the "
        "rulebase agent config"
        if "rulebase" in planner_label
        else "CoLMDriver planner config uses the same planning_bank extraction path"
    )

    wps = getattr(agent, "last_planned_waypoints_world", None)
    if wps is None or len(wps) < 2:
        return PlannerOutput(traj_source="colmdriver_none")

    raw_wps = np.asarray(wps, dtype=np.float64)[:, :2]
    if not _has_finite_xy(raw_wps):
        return PlannerOutput(traj_source="colmdriver_nonfinite")

    # Correct the agent-frame origin offset to true lidar_pose world frame.
    world_x = float(ego_world_xy[0])
    world_y = float(ego_world_xy[1])
    offset = np.array([world_x - world_y, world_x + world_y], dtype=np.float64)
    wps = raw_wps + offset
    if not _has_finite_xy(wps):
        return PlannerOutput(traj_source="colmdriver_nonfinite_corrected")

    # Guard against stale/uninitialized planning_bank values.
    #
    # CoLMDriver initialises planning_bank with constant placeholders; when the
    # inference branch is skipped near route end those placeholders can leak
    # into last_planned_waypoints_world. After offset correction this creates
    # huge absolute positions and dominates ADE/FDE.
    first_wp_offset = _first_wp_offset_m(wps, ego_world_xy)
    raw_scale = float(np.median(np.linalg.norm(raw_wps, axis=1)))
    if first_wp_offset > MAX_REASONABLE_FIRST_WP_OFFSET_M:
        # Always reject far-away first waypoints for CoLMDriver extraction:
        # at 0.1 s horizon the first planned point should be close to ego.
        # This catches stale/default planning_bank snapshots that can be
        # hundreds of meters away in the wrong origin frame.
        if _is_degenerate_trajectory(wps) or raw_scale < 50.0:
            return PlannerOutput(
                traj_source="colmdriver_invalid_fallback",
                debug={
                    "first_wp_offset_m": first_wp_offset,
                    "raw_scale_median": raw_scale,
                    "raw_wps": raw_wps,
                    "corrected_world_wps": wps,
                },
            )
        return PlannerOutput(
            traj_source="colmdriver_far_first_wp",
            debug={
                "first_wp_offset_m": first_wp_offset,
                "raw_scale_median": raw_scale,
                "raw_wps": raw_wps,
                "corrected_world_wps": wps,
            },
        )

    # Native dt = 0.1 s → resample to 0.5 s evaluation grid.
    resampled = _resample_trajectory(wps, src_dt=0.1)
    return PlannerOutput(
        future_trajectory_world=resampled,
        traj_source="colmdriver_world_wps",
        raw_length=len(wps),
        raw_world_wps=wps,
        native_dt=0.1,
        raw_export=_build_raw_export(
            raw_positions=raw_wps,
            source_frame_description=(
                "CoLMDriver internal GPS-mapped absolute frame before lidar-pose "
                "offset correction"
            ),
            axis_convention_description=(
                "absolute [x, y] positions in the agent's mapped world-like frame; "
                "ego origin differs from lidar_pose world frame"
            ),
            is_cumulative_positions=False,
            point0_mode="future",
            native_dt=0.1,
            native_timestamps=None,
            adapter_debug_notes=[
                "raw_positions are exported before lidar-pose offset correction",
                "world-frame raw_world_wps remains the evaluator-facing pre-resample path",
                (
                    "planning_bank stores absolute future positions in CoLMDriver's "
                    "mapped world-like frame; the adapter adds the per-frame lidar-pose "
                    "offset before canonical/world evaluation"
                ),
                (
                    "point0 is treated as the first future waypoint from planning_bank; "
                    "current ego pose is not included in the exported waypoint list"
                ),
                (
                    "per-point timestamps are not emitted by planning_bank; native_dt=0.1 s "
                    "is treated as an inferred sampling assumption from the 20-point planner "
                    "horizon, so native_timestamps is intentionally left null"
                ),
                (
                    "observed freshness/reuse: CoLMDriver-family agents reuse prior control "
                    "between planner updates; non-realtime path skips inference when "
                    "step % skip_frames != 0"
                ),
                planner_family_note,
            ],
        ),
        debug={
            "first_wp_offset_m": first_wp_offset,
            "raw_scale_median": raw_scale,
            "raw_wps": raw_wps,
            "corrected_world_wps": wps,
            "resampled_world_wps": resampled,
        },
    )


def _adapt_vad(
    agent: Any,
    ego_world_xy: np.ndarray,
    compass_rad: float,
    **kw,
) -> PlannerOutput:
    """VAD stores ``pid_metadata['plan']`` as ego-local future positions."""
    md = getattr(agent, "pid_metadata", None)
    if not isinstance(md, dict):
        return PlannerOutput(traj_source="vad_no_metadata")

    plan = md.get("plan")
    if plan is None:
        return PlannerOutput(traj_source="vad_no_plan")

    local_wps = np.asarray(plan, dtype=np.float64).reshape(-1, 2)
    if len(local_wps) == 0:
        return PlannerOutput(traj_source="vad_empty_plan")
    if not _has_finite_xy(local_wps):
        return PlannerOutput(traj_source="vad_nonfinite_plan")

    # VAD trajectory/PID frame is (right=+x, forward=+y).
    # Convert to CARLA BEV (forward=+x, left=+y).
    carla_bev = np.column_stack([local_wps[:, 1], -local_wps[:, 0]])

    world_wps = _ego_local_to_world(carla_bev, ego_world_xy, compass_rad)
    if _first_wp_offset_m(world_wps, ego_world_xy) > MAX_REASONABLE_FIRST_WP_OFFSET_M:
        return PlannerOutput(
            traj_source="vad_far_first_wp",
            debug={
                "local_wps": local_wps,
                "carla_bev_wps": carla_bev,
                "world_wps": world_wps,
                "first_wp_offset_m": _first_wp_offset_m(world_wps, ego_world_xy),
            },
        )
    # VAD outputs exactly 6 waypoints at 0.5 s — matches eval grid.
    resampled = _resample_trajectory(world_wps, src_dt=EVAL_DT)
    vad_debug: Dict[str, Any] = {
        "local_wps": local_wps,
        "carla_bev_wps": carla_bev,
        "world_wps": world_wps,
        "resampled_world_wps": resampled,
    }
    try:
        all_plan = md.get("all_plan")
        all_plan_arr = np.asarray(all_plan, dtype=np.float64)
        if all_plan_arr.ndim == 3 and all_plan_arr.shape[-1] == 2 and all_plan_arr.shape[0] >= 2:
            vad_debug["candidate_local_wps_by_command"] = all_plan_arr
            cmd = md.get("command")
            if cmd is not None:
                vad_debug["selected_command_index"] = int(cmd)
    except Exception:
        # Branch diagnostics are best-effort and must never affect planner behavior.
        pass
    return PlannerOutput(
        future_trajectory_world=resampled,
        traj_source="vad_plan",
        raw_length=len(local_wps),
        raw_world_wps=world_wps,
        native_dt=EVAL_DT,
        raw_export=_build_raw_export(
            raw_positions=local_wps,
            source_frame_description="VAD pid_metadata['plan'] ego-local controller frame",
            axis_convention_description="x=right, y=forward",
            is_cumulative_positions=False,
            point0_mode="future",
            native_dt=EVAL_DT,
            native_timestamps=None,
            adapter_debug_notes=[
                "raw_positions are exported before CARLA-BEV axis remap",
                (
                    "VAD planner code applies np.cumsum to ego_fut_preds before "
                    "storing pid_metadata['plan']; exported points are absolute "
                    "future positions in the ego-local PID/controller frame"
                ),
                (
                    "point0 is the first future target, not the current ego pose; "
                    "the current anchor is omitted from pid_metadata['plan']"
                ),
                (
                    "per-point timestamps are not emitted by VAD; native_dt=0.5 s "
                    "is carried as an inferred timing assumption from the 6-step "
                    "future horizon used by the adapter/evaluator, so "
                    "native_timestamps is left null on purpose"
                ),
                (
                    "observed freshness/reuse: VAD commonly reuses the previous plan "
                    "between fresh updates; planner source gates recomputation with "
                    "step % 4 == 1 in non-realtime mode"
                ),
            ],
        ),
        debug=vad_debug,
    )


def _adapt_uniad(
    agent: Any,
    ego_world_xy: np.ndarray,
    compass_rad: float,
    **kw,
) -> PlannerOutput:
    """Adapt UniAD ``pid_metadata['plan']`` into the shared planner output.

    UniAD writes the model planning output ``sdc_traj`` into
    ``pid_metadata['plan']`` after calling ``control_pid``. The controller
    consumes the trajectory as ego-local positions (not cumulative deltas)
    using the (right=+x, forward=+y) convention.
    """
    md = getattr(agent, "pid_metadata", None)
    if not isinstance(md, dict):
        return PlannerOutput(traj_source="uniad_no_metadata")

    plan = md.get("plan")
    if plan is None:
        return PlannerOutput(traj_source="uniad_no_plan")

    local_wps = np.asarray(plan, dtype=np.float64).reshape(-1, 2)
    if len(local_wps) == 0:
        return PlannerOutput(traj_source="uniad_empty_plan")
    if not _has_finite_xy(local_wps):
        return PlannerOutput(traj_source="uniad_nonfinite_plan")

    # UniAD trajectory/PID frame is (right=+x, forward=+y).
    # Convert to CARLA BEV (forward=+x, left=+y).
    carla_bev = np.column_stack([local_wps[:, 1], -local_wps[:, 0]])
    world_wps = _ego_local_to_world(carla_bev, ego_world_xy, compass_rad)
    if _first_wp_offset_m(world_wps, ego_world_xy) > MAX_REASONABLE_FIRST_WP_OFFSET_M:
        return PlannerOutput(
            traj_source="uniad_far_first_wp",
            debug={
                "local_wps": local_wps,
                "carla_bev_wps": carla_bev,
                "world_wps": world_wps,
                "first_wp_offset_m": _first_wp_offset_m(world_wps, ego_world_xy),
            },
        )
    resampled = _resample_trajectory(world_wps, src_dt=EVAL_DT)
    source_step_token = md.get("step") if isinstance(md, dict) else None
    adapter_notes = [
        "raw_positions are exported before CARLA-BEV axis remap",
        (
            "UniAD assigns output_data_batch['planning']['result_planning']['sdc_traj'][0] "
            "to pid_metadata['plan']; exported points are interpreted as ego-local "
            "absolute future positions used directly by PID control"
        ),
        (
            "point0 is treated as the first future target waypoint; current ego "
            "anchor is not included in pid_metadata['plan']"
        ),
        (
            "per-point timestamps are not emitted by UniAD; native_dt=0.5 s is "
            "carried as an inferred timing assumption from the 6-step future "
            "horizon, so native_timestamps is intentionally left null"
        ),
        (
            "observed freshness/reuse: UniAD commonly reuses the previous control "
            "between fresh updates; non-realtime path gates recomputation with "
            "step % 4 == 1"
        ),
    ]
    if source_step_token is None:
        adapter_notes.append(
            "pid_metadata does not currently expose a planner source-step token; "
            "freshness falls back to content-based reuse detection"
        )
    else:
        adapter_notes.append("freshness token uses pid_metadata['step'] when available")
    return PlannerOutput(
        future_trajectory_world=resampled,
        traj_source="uniad_plan",
        raw_length=len(local_wps),
        raw_world_wps=world_wps,
        native_dt=EVAL_DT,
        raw_export=_build_raw_export(
            raw_positions=local_wps,
            source_frame_description="UniAD pid_metadata['plan'] ego-local controller frame",
            axis_convention_description="x=right, y=forward",
            is_cumulative_positions=False,
            point0_mode="future",
            native_dt=EVAL_DT,
            native_timestamps=None,
            adapter_debug_notes=adapter_notes,
            freshness_token=source_step_token,
        ),
        debug={
            "local_wps": local_wps,
            "carla_bev_wps": carla_bev,
            "world_wps": world_wps,
            "resampled_world_wps": resampled,
        },
    )


def _adapt_tcp(
    agent: Any,
    ego_world_xy: np.ndarray,
    compass_rad: float,
    **kw,
) -> PlannerOutput:
    """TCP stores ``pid_metadata['wp_1']`` … ``wp_4`` as ego-local tuples.

    TCP metadata uses the post-`control_pid` convention from TCP model:
      right = +x, forward = +y.
    The adapter converts this to CARLA BEV:
      forward = +x, left = +y.

    TCP waypoints span ~2.0 s (4 points at ~0.5 s each).  We
    extrapolate to fill the 3.0 s evaluation horizon.
    """
    md = getattr(agent, "pid_metadata", None)
    if not isinstance(md, dict):
        return PlannerOutput(traj_source="tcp_no_metadata")

    wps: list = []
    for i in range(1, 20):  # wp_1 .. wp_N (usually 4)
        wp = md.get(f"wp_{i}")
        if wp is None:
            break
        try:
            wps.append([float(wp[0]), float(wp[1])])
        except (TypeError, IndexError):
            break

    cache_attr = "_openloop_cached_tcp_wps_local"
    source_label = "tcp_wp"

    if len(wps) >= 2:
        tcp_wps = np.array(wps, dtype=np.float64)
        if not _has_finite_xy(tcp_wps):
            return PlannerOutput(traj_source="tcp_nonfinite_wp")
        try:
            setattr(agent, cache_attr, tcp_wps.copy())
        except Exception:
            pass
    else:
        cached_wps = _pick_first_waypoint_array(getattr(agent, cache_attr, None))
        if cached_wps is None:
            return PlannerOutput(traj_source="tcp_insufficient_wps")
        tcp_wps = cached_wps
        source_label = "tcp_wp_cached"

    # TCP convention after y-flip in model.py:
    #   x = lateral-right (+), y = forward (+)
    # i.e. (right=+x, forward=+y) — nuScenes-style.
    # Convert to CARLA BEV (forward=+x, left=+y):
    #   CARLA_x = TCP_y,  CARLA_y = -TCP_x
    carla_bev = np.column_stack([tcp_wps[:, 1], -tcp_wps[:, 0]])
    world_wps = _ego_local_to_world(carla_bev, ego_world_xy, compass_rad)
    if _first_wp_offset_m(world_wps, ego_world_xy) > MAX_REASONABLE_FIRST_WP_OFFSET_M:
        return PlannerOutput(
            traj_source="tcp_far_first_wp",
            debug={
                "tcp_wps_raw": tcp_wps,
                "carla_bev_wps": carla_bev,
                "world_wps": world_wps,
                "first_wp_offset_m": _first_wp_offset_m(world_wps, ego_world_xy),
            },
        )
    # 4 waypoints at ~0.5 s each.
    resampled = _resample_trajectory(world_wps, src_dt=0.5)
    return PlannerOutput(
        future_trajectory_world=resampled,
        traj_source=source_label,
        raw_length=len(tcp_wps),
        raw_world_wps=world_wps,
        native_dt=0.5,
        raw_export=_build_raw_export(
            raw_positions=tcp_wps,
            source_frame_description="TCP pid_metadata waypoint tuples in ego-local controller frame",
            axis_convention_description="x=right, y=forward",
            is_cumulative_positions=False,
            point0_mode="future",
            native_dt=0.5,
            native_timestamps=None,
            adapter_debug_notes=[
                "raw_positions are exported before CARLA-BEV axis remap",
                f"source_label={source_label}",
                (
                    "TCP control_pid flips the model waypoint y-axis before storing "
                    "wp_1..wp_4 in pid_metadata; exported points are absolute future "
                    "ego-local positions in the controller frame"
                ),
                (
                    "point0 is the first future target waypoint; TCP does not include "
                    "the current ego anchor in wp_1..wp_4"
                ),
                (
                    "per-point timestamps are not emitted by TCP; native_dt=0.5 s is "
                    "treated as an inferred timing assumption from the 4-point controller "
                    "trajectory, so native_timestamps is intentionally left null"
                ),
                (
                    "observed freshness/reuse: TCP commonly reuses the previous control "
                    "between fresh updates; non-realtime path gates recomputation with "
                    "step % 4 == 0"
                ),
            ],
        ),
        debug={
            "tcp_wps_raw": tcp_wps,
            "carla_bev_wps": carla_bev,
            "world_wps": world_wps,
            "resampled_world_wps": resampled,
        },
    )


def _adapt_codriving(
    agent: Any,
    ego_world_xy: np.ndarray,
    compass_rad: float,
    **kw,
) -> PlannerOutput:
    """CoDriving exports predicted waypoints to infer.save_path/ego_vehicle_0/*.json.

    The planner emits 10-step local waypoints per inference frame. Empirically and
    by the planner-side visualizer convention, these are best aligned to CARLA BEV as:
      forward = -local_y, left = -local_x
    with native sampling inferred at 5 Hz (dt ~= 0.2 s).
    """
    infer = getattr(agent, "infer", None)
    local_wps = None
    source_label = "codriving_plan_live"
    source_debug: Dict[str, Any] = {}

    if infer is not None:
        local_wps = _pick_first_waypoint_array(
            getattr(infer, "last_predicted_waypoints_local", None)
        )
        if local_wps is not None:
            source_debug["live_step"] = getattr(infer, "last_predicted_waypoints_step", None)

    if local_wps is None:
        local_wps = _pick_first_waypoint_array(
            getattr(agent, "last_predicted_waypoints_local", None)
        )
        if local_wps is not None:
            source_debug["live_step"] = getattr(agent, "last_predicted_waypoints_step", None)

    if local_wps is None:
        save_path = getattr(infer, "save_path", None)
        if save_path is None:
            return PlannerOutput(traj_source="codriving_no_save_path")

        record = _latest_json_record(Path(save_path) / "ego_vehicle_0", "*.json")
        if not isinstance(record, dict):
            return PlannerOutput(traj_source="codriving_no_plan_file")

        waypoints = record.get("waypoints")
        if waypoints is None:
            return PlannerOutput(traj_source="codriving_no_waypoints")

        local_wps = _coerce_waypoints(waypoints)
        if local_wps is None:
            return PlannerOutput(traj_source="codriving_insufficient_wps")
        source_label = "codriving_plan_file"
        source_debug["plan_file"] = record.get("step")

    # CoDriving visualizer maps waypoints as x=-wp_y, y=-wp_x before plotting in
    # ego-centric lidar frame; use the same sign convention when converting to CARLA BEV.
    carla_bev = np.column_stack([-local_wps[:, 1], -local_wps[:, 0]])
    world_wps = _ego_local_to_world(carla_bev, ego_world_xy, compass_rad)
    if _first_wp_offset_m(world_wps, ego_world_xy) > MAX_REASONABLE_FIRST_WP_OFFSET_M:
        return PlannerOutput(
            traj_source="codriving_far_first_wp",
            debug={
                "local_wps": local_wps,
                "carla_bev_wps": carla_bev,
                "world_wps": world_wps,
            },
        )

    # CoDriving controller derives desired speed with a *5 gain over waypoint
    # displacements, implying 0.2 s spacing for the 10 predicted future points.
    src_dt = 0.2
    resampled = _resample_trajectory(world_wps, src_dt=src_dt)
    return PlannerOutput(
        future_trajectory_world=resampled,
        traj_source=source_label,
        raw_length=len(local_wps),
        raw_world_wps=world_wps,
        native_dt=src_dt,
        raw_export=_build_raw_export(
            raw_positions=local_wps,
            source_frame_description=(
                "CoDriving predicted ego-local waypoint array from live inference or saved plan file"
            ),
            axis_convention_description=(
                "planner-local coordinates; adapter uses visualizer/controller-aligned "
                "mapping carla_forward=-local_y, carla_left=-local_x while original "
                "axis naming remains unresolved"
            ),
            is_cumulative_positions=False,
            point0_mode="future",
            native_dt=src_dt,
            native_timestamps=None,
            adapter_debug_notes=[
                "raw_positions are exported before CARLA-BEV sign remap",
                "visualizer-aligned conversion remains in adapter debug metadata",
                f"source_label={source_label}",
                (
                    "planning_model['future_waypoints'] are consumed directly by V2X_Controller "
                    "as absolute future local positions, not cumulative deltas"
                ),
                (
                    "point0 is treated as the first future target waypoint; current ego "
                    "anchor is not included in the 10 predicted points"
                ),
                (
                    "per-point timestamps are not emitted by CoDriving; native_dt=0.2 s is "
                    "treated as an inferred timing assumption from the 10-point horizon and "
                    "controller desired-speed scaling, so native_timestamps is intentionally left null"
                ),
                (
                    "observed freshness/reuse: CoDriving commonly reuses the previous control "
                    "between fresh updates; non-realtime path gates recomputation with "
                    "step % 4 == 0"
                ),
            ],
            freshness_token=source_debug.get("live_step", source_debug.get("plan_file")),
        ),
        debug={
            "local_wps": local_wps,
            "carla_bev_wps": carla_bev,
            "world_wps": world_wps,
            "resampled_world_wps": resampled,
            "src_dt": float(src_dt),
            "coord_map": "carla_forward=-local_y,carla_left=-local_x",
            **source_debug,
        },
    )


def _adapt_lmdrive(
    agent: Any,
    ego_world_xy: np.ndarray,
    compass_rad: float,
    **kw,
) -> PlannerOutput:
    """LMDrive saves route snapshots with predicted_waypoints_local per inference step.

    Source path: save_path/meta_0/*_route_debug.json
    ``predicted_waypoints_local`` uses controller frame (right=+x, forward=+y).
    """
    source_label = "lmdrive_live_waypoints"
    source_debug: Dict[str, Any] = {}
    local_wps = _pick_first_waypoint_array(
        getattr(agent, "last_predicted_waypoints_local", None)
    )
    if local_wps is not None:
        source_debug["live_step"] = getattr(agent, "last_predicted_waypoints_step", None)

    if local_wps is None:
        save_path = getattr(agent, "save_path", None)
        if save_path is None:
            return PlannerOutput(traj_source="lmdrive_no_save_path")

        record = _latest_json_record(Path(save_path) / "meta_0", "*_route_debug.json")
        if not isinstance(record, dict):
            return PlannerOutput(traj_source="lmdrive_no_route_debug")

        local = record.get("predicted_waypoints_local")
        if local is None:
            return PlannerOutput(traj_source="lmdrive_no_predicted_waypoints")

        local_wps = _coerce_waypoints(local)
        if local_wps is None:
            return PlannerOutput(traj_source="lmdrive_insufficient_wps")
        source_label = "lmdrive_route_debug"
        source_debug["frame"] = record.get("frame")

    # LMDrive snapshot local frame: right=+x, forward=+y.
    carla_bev = np.column_stack([local_wps[:, 1], -local_wps[:, 0]])
    world_wps = _ego_local_to_world(carla_bev, ego_world_xy, compass_rad)
    if _first_wp_offset_m(world_wps, ego_world_xy) > MAX_REASONABLE_FIRST_WP_OFFSET_M:
        return PlannerOutput(
            traj_source="lmdrive_far_first_wp",
            debug={
                "local_wps": local_wps,
                "carla_bev_wps": carla_bev,
                "world_wps": world_wps,
            },
        )

    # LMDrive predicts 5 waypoints with controller scaling by 2.0 (dt ~= 0.5 s).
    resampled = _resample_trajectory(world_wps, src_dt=0.5)
    return PlannerOutput(
        future_trajectory_world=resampled,
        traj_source=source_label,
        raw_length=len(local_wps),
        raw_world_wps=world_wps,
        native_dt=0.5,
        raw_export=_build_raw_export(
            raw_positions=local_wps,
            source_frame_description=(
                "LMDrive predicted ego-local waypoint array from live cache or route_debug snapshot"
            ),
            axis_convention_description="x=right, y=forward",
            is_cumulative_positions=False,
            point0_mode="future",
            native_dt=0.5,
            native_timestamps=None,
            adapter_debug_notes=[
                "raw_positions are exported before CARLA-BEV axis remap",
                f"source_label={source_label}",
                (
                    "LMDrive flips model waypoint y before control_pid metadata/adapter export; "
                    "exported points are absolute future positions in the ego-local controller frame"
                ),
                (
                    "point0 is treated as the first future target waypoint; LMDrive does not "
                    "include the current ego anchor in predicted_waypoints_local"
                ),
                (
                    "per-point timestamps are not emitted by LMDrive; native_dt=0.5 s is "
                    "treated as an inferred timing assumption from the 5-point waypoint horizon "
                    "and controller speed scaling, so native_timestamps is intentionally left null"
                ),
                (
                    "observed freshness/reuse: LMDrive commonly reuses the previous control "
                    "between fresh updates; non-realtime path gates recomputation with "
                    "step % 5 == 0"
                ),
            ],
            freshness_token=source_debug.get("live_step", source_debug.get("frame")),
        ),
        debug={
            "local_wps": local_wps,
            "carla_bev_wps": carla_bev,
            "world_wps": world_wps,
            "resampled_world_wps": resampled,
            **source_debug,
        },
    )


def _adapt_generic(
    agent: Any,
    ego_world_xy: np.ndarray,
    compass_rad: float,
    **kw,
) -> PlannerOutput:
    """Generic / fallback adapter.

    Attempts all known attribute patterns in priority order:
    1. ``last_planned_waypoints_world`` (world-frame, 0.1 s)
    2. ``pid_metadata['plan']`` (ego-local, 0.5 s)
    3. ``pid_metadata`` wp_1…wp_N (ego-local, ~0.5 s)
    """
    # Priority 1: world-frame planned waypoints stored by CoLMDriver-style agents.
    # These use the agent's GPS-mapped origin (world_y, -world_x) rather than
    # the true lidar_pose origin (world_x, world_y).  Apply the same offset
    # correction as _adapt_colmdriver.
    wps = getattr(agent, "last_planned_waypoints_world", None)
    if wps is not None and hasattr(wps, "__len__") and len(wps) >= 2:
        wps = np.asarray(wps, dtype=np.float64)[:, :2]
        if not _has_finite_xy(wps):
            return PlannerOutput(traj_source="generic_world_nonfinite")
        world_x = float(ego_world_xy[0])
        world_y = float(ego_world_xy[1])
        offset = np.array([world_x - world_y, world_x + world_y], dtype=np.float64)
        wps = wps + offset
        if not _has_finite_xy(wps):
            return PlannerOutput(traj_source="generic_world_nonfinite_corrected")
        if _first_wp_offset_m(wps, ego_world_xy) > MAX_REASONABLE_FIRST_WP_OFFSET_M:
            return PlannerOutput(
                traj_source="generic_world_far_first_wp",
                debug={"world_wps_corrected": wps},
            )
        resampled = _resample_trajectory(wps, src_dt=0.1)
        return PlannerOutput(
            future_trajectory_world=resampled,
            traj_source="generic_world_wps",
            raw_length=len(wps),
            raw_export=_build_raw_export(
                raw_positions=wps - offset,
                source_frame_description=(
                    "generic world-like absolute frame before lidar-pose offset correction"
                ),
                axis_convention_description="absolute [x, y] positions in planner-provided source frame",
                is_cumulative_positions=False,
                point0_mode="unknown",
                native_dt=0.1,
                adapter_debug_notes=[
                    "generic fallback could not prove point-0 semantics",
                    "raw_positions are exported before evaluator offset correction",
                ],
            ),
            debug={
                "world_wps_corrected": wps,
                "resampled_world_wps": resampled,
            },
        )

    # Priority 2: pid_metadata['plan'] (ego-local)
    md = getattr(agent, "pid_metadata", None)
    if isinstance(md, dict):
        plan = md.get("plan")
        if plan is not None:
            local_wps = np.asarray(plan, dtype=np.float64).reshape(-1, 2)
            if len(local_wps) >= 2:
                if not _has_finite_xy(local_wps):
                    return PlannerOutput(traj_source="generic_pid_plan_nonfinite")
                world_wps = _ego_local_to_world(local_wps, ego_world_xy, compass_rad)
                if _first_wp_offset_m(world_wps, ego_world_xy) > MAX_REASONABLE_FIRST_WP_OFFSET_M:
                    return PlannerOutput(
                        traj_source="generic_pid_plan_far_first_wp",
                        debug={"local_wps": local_wps, "world_wps": world_wps},
                    )
                resampled = _resample_trajectory(world_wps, src_dt=EVAL_DT)
                return PlannerOutput(
                    future_trajectory_world=resampled,
                    traj_source="generic_pid_plan",
                    raw_length=len(local_wps),
                    raw_export=_build_raw_export(
                        raw_positions=local_wps,
                        source_frame_description="generic pid_metadata['plan'] ego-local frame",
                        axis_convention_description="unknown ego-local axes",
                        is_cumulative_positions=None,
                        point0_mode="unknown",
                        native_dt=EVAL_DT,
                        adapter_debug_notes=[
                            "generic fallback keeps axis semantics unresolved",
                        ],
                    ),
                    debug={
                        "local_wps": local_wps,
                        "world_wps": world_wps,
                        "resampled_world_wps": resampled,
                    },
                )

        # Priority 3: wp_1 … wp_N
        wp_list: list = []
        for i in range(1, 20):
            wp = md.get(f"wp_{i}")
            if wp is None:
                break
            try:
                wp_list.append([float(wp[0]), float(wp[1])])
            except (TypeError, IndexError):
                break
        if len(wp_list) >= 2:
            local_arr = np.array(wp_list, dtype=np.float64)
            if not _has_finite_xy(local_arr):
                return PlannerOutput(traj_source="generic_pid_wp_nonfinite")
            world_wps = _ego_local_to_world(local_arr, ego_world_xy, compass_rad)
            if _first_wp_offset_m(world_wps, ego_world_xy) > MAX_REASONABLE_FIRST_WP_OFFSET_M:
                return PlannerOutput(
                    traj_source="generic_pid_wp_far_first_wp",
                    debug={"local_wps": local_arr, "world_wps": world_wps},
                )
            resampled = _resample_trajectory(world_wps, src_dt=0.5)
            return PlannerOutput(
                future_trajectory_world=resampled,
                traj_source="generic_pid_wp",
                raw_length=len(wp_list),
                raw_export=_build_raw_export(
                    raw_positions=local_arr,
                    source_frame_description="generic pid_metadata wp_i ego-local frame",
                    axis_convention_description="unknown ego-local axes",
                    is_cumulative_positions=None,
                    point0_mode="unknown",
                    native_dt=0.5,
                    adapter_debug_notes=[
                        "generic fallback keeps axis semantics unresolved",
                    ],
                ),
                debug={
                    "local_wps": local_arr,
                    "world_wps": world_wps,
                    "resampled_world_wps": resampled,
                },
            )

    return PlannerOutput(traj_source="generic_fallback_none")


# ── Adapter registry ─────────────────────────────────────────

# Maps planner preset name → adapter function.
# The generic adapter is used for any planner not in this dict.
AdapterFn = Callable[..., PlannerOutput]

ADAPTER_REGISTRY: Dict[str, AdapterFn] = {
    "colmdriver":          _adapt_colmdriver,
    "colmdriver_rulebase": _adapt_colmdriver,
    "vad":                 _adapt_vad,
    "uniad":               _adapt_uniad,
    "tcp":                 _adapt_tcp,
    "codriving":           _adapt_codriving,
    "lmdrive":             _adapt_lmdrive,
    "autopilot":           _adapt_generic,
}


def get_adapter(planner_name: Optional[str] = None) -> AdapterFn:
    """Return the adapter function for a planner by preset name.

    Falls back to the generic adapter for unknown planners.
    """
    if planner_name is None:
        return _adapt_generic
    name = str(planner_name).lower()
    if name in ADAPTER_REGISTRY:
        return ADAPTER_REGISTRY[name]
    # Planner labels may include suffixes/run IDs (for example,
    # "colmdriver__g0"), so we also support substring matching.
    for key, fn in ADAPTER_REGISTRY.items():
        if key in name:
            return fn
    return _adapt_generic


def extract_planner_trajectory(
    agent: Any,
    ego_world_xy: np.ndarray,
    compass_rad: float,
    planner_name: Optional[str] = None,
) -> PlannerOutput:
    """Top-level entry point used by the evaluator.

    Selects the appropriate adapter and extracts + aligns the trajectory.
    """
    adapter = get_adapter(planner_name)
    return adapter(agent, ego_world_xy, compass_rad, planner_name=planner_name)


# ── Stale-control detector (debug-only, non-breaking) ────────

class StaleControlDetector:
    """Detects when a planner reuses the same control across frames.

    This is a *diagnostic* tool — it does NOT alter evaluation.
    """

    def __init__(self):
        self._prev_ctrl: Optional[Dict[str, float]] = None
        self.n_total: int = 0
        self.n_stale: int = 0

    def check(self, ctrl: Optional[Dict[str, float]]) -> bool:
        """Return ``True`` if control is identical to the previous frame."""
        self.n_total += 1
        if ctrl is None or self._prev_ctrl is None:
            self._prev_ctrl = ctrl
            return False
        is_stale = (
            abs(ctrl.get("steer", 0) - self._prev_ctrl.get("steer", 0)) < 1e-8
            and abs(ctrl.get("throttle", 0) - self._prev_ctrl.get("throttle", 0)) < 1e-8
            and abs(ctrl.get("brake", 0) - self._prev_ctrl.get("brake", 0)) < 1e-8
        )
        if is_stale:
            self.n_stale += 1
        self._prev_ctrl = dict(ctrl)
        return is_stale

    def summary(self) -> Dict[str, Any]:
        return {
            "n_total": self.n_total,
            "n_stale": self.n_stale,
            "stale_fraction": self.n_stale / max(1, self.n_total),
        }
