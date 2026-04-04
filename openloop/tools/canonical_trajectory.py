"""Stage 2 canonical trajectory construction for open-loop evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Optional

import numpy as np


CANONICAL_TIMESTAMPS = np.asarray([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float64)
_INFERRED_TIMESTAMP_PLANNERS = frozenset(
    {"vad", "uniad", "tcp", "lmdrive", "codriving", "colmdriver", "colmdriver_rulebase"}
)
_ENABLE_LINEAR_EXTRAPOLATION = os.environ.get(
    "OPENLOOP_CANONICAL_EXTRAPOLATE", "1"
).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class CanonicalTrajectory:
    planner_name: str
    scenario_id: str
    frame_id: int
    canonical_timestamps: np.ndarray
    canonical_positions: np.ndarray
    valid_mask: np.ndarray
    canonical_frame_description: str
    timestamp_source: str
    interpolation_method: str
    is_fresh_plan: bool
    provenance_debug_notes: list[str] = field(default_factory=list)


def _as_xy_array(value: Any) -> Optional[np.ndarray]:
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


def _as_time_array(value: Any, expected_len: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if len(arr) != expected_len or not np.isfinite(arr).all():
        return None
    return arr


def _infer_timestamps(
    n_points: int,
    *,
    native_timestamps: Any,
    native_dt: Any,
    point0_mode: Any,
) -> tuple[Optional[np.ndarray], str, list[str]]:
    notes: list[str] = []
    explicit = _as_time_array(native_timestamps, expected_len=n_points)
    if explicit is not None:
        return explicit, "explicit", notes

    try:
        dt_value = None if native_dt is None else float(native_dt)
    except Exception:
        dt_value = None
    if dt_value is None or not np.isfinite(dt_value) or dt_value <= 0.0:
        notes.append("native_dt missing or invalid; timestamp semantics unresolved")
        return None, "unresolved", notes

    mode = str(point0_mode or "unknown")
    if mode == "future":
        return dt_value * np.arange(1, n_points + 1, dtype=np.float64), "inferred", notes
    if mode == "current":
        return dt_value * np.arange(0, n_points, dtype=np.float64), "inferred", notes

    notes.append("point0_mode unresolved; timestamps not inferred")
    return None, "unresolved", notes


def _downgrade_adapter_derived_timing_to_inferred(
    planner_name: str,
    *,
    n_points: int,
    timestamps: Optional[np.ndarray],
    timestamp_source: str,
    native_dt: Any,
    point0_mode: Any,
) -> tuple[Optional[np.ndarray], str, list[str]]:
    notes: list[str] = []
    planner = str(planner_name).lower()
    if planner not in _INFERRED_TIMESTAMP_PLANNERS:
        return timestamps, timestamp_source, notes
    if timestamps is None or timestamp_source != "explicit":
        return timestamps, timestamp_source, notes

    try:
        dt_value = None if native_dt is None else float(native_dt)
    except Exception:
        dt_value = None
    if dt_value is None or not np.isfinite(dt_value) or dt_value <= 0.0:
        notes.append(
            f"{planner.upper()} native_timestamps kept explicit because native_dt is unavailable"
        )
        return timestamps, timestamp_source, notes

    mode = str(point0_mode or "unknown")
    if mode == "future":
        expected = dt_value * np.arange(1, n_points + 1, dtype=np.float64)
    elif mode == "current":
        expected = dt_value * np.arange(0, n_points, dtype=np.float64)
    else:
        notes.append(
            f"{planner.upper()} point0_mode unresolved; explicit timestamps left unchanged"
        )
        return timestamps, timestamp_source, notes

    if np.allclose(np.asarray(timestamps, dtype=np.float64), expected, atol=1e-9, rtol=0.0):
        notes.append(
            f"{planner.upper()} native_timestamps match the adapter-derived point0_mode/native_dt "
            "grid; timestamp semantics are treated as inferred because the planner "
            "does not expose per-point timestamps"
        )
        return expected, "inferred", notes

    notes.append(
        f"{planner.upper()} native_timestamps differ from the adapter-derived grid; kept explicit"
    )
    return timestamps, timestamp_source, notes


def _document_inferred_adapter_timing(
    planner_name: str,
    *,
    timestamp_source: str,
    native_timestamps: Any,
    native_dt: Any,
    point0_mode: Any,
) -> list[str]:
    planner = str(planner_name).lower()
    if planner not in _INFERRED_TIMESTAMP_PLANNERS:
        return []
    if timestamp_source != "inferred" or native_timestamps is not None:
        return []

    try:
        dt_value = None if native_dt is None else float(native_dt)
    except Exception:
        dt_value = None
    if dt_value is None or not np.isfinite(dt_value) or dt_value <= 0.0:
        return []

    mode = str(point0_mode or "unknown")
    if mode not in {"future", "current"}:
        return []

    return [
        (
            f"{planner.upper()} timestamp semantics are treated as inferred from "
            f"point0_mode={mode} and native_dt={dt_value:.3f}s because the planner "
            "does not expose per-point timestamps"
        )
    ]


def _reconstruct_absolute_positions(
    source_world_positions: np.ndarray,
    *,
    is_cumulative_positions: Any,
) -> tuple[np.ndarray, list[str]]:
    notes: list[str] = []
    is_cumulative = bool(is_cumulative_positions) if is_cumulative_positions is not None else False
    if not is_cumulative:
        return np.asarray(source_world_positions, dtype=np.float64), notes
    notes.append("reconstructed cumulative displacements into absolute positions with np.cumsum")
    return np.cumsum(np.asarray(source_world_positions, dtype=np.float64), axis=0), notes


def _resample_to_canonical(
    source_positions: np.ndarray,
    source_timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str, list[str]]:
    notes: list[str] = []
    canonical_positions = np.zeros((len(CANONICAL_TIMESTAMPS), 2), dtype=np.float64)
    valid_mask = np.zeros(len(CANONICAL_TIMESTAMPS), dtype=bool)

    if len(source_positions) == 1:
        for idx, ts in enumerate(CANONICAL_TIMESTAMPS):
            if abs(float(source_timestamps[0]) - float(ts)) <= 1e-9:
                canonical_positions[idx] = source_positions[0]
                valid_mask[idx] = True
        notes.append("single-point source; only exact timestamp matches are marked valid")
        return canonical_positions, valid_mask, "exact_match_only", notes

    order = np.argsort(source_timestamps)
    ordered_t = np.asarray(source_timestamps[order], dtype=np.float64)
    ordered_p = np.asarray(source_positions[order], dtype=np.float64)

    dedup_t: list[float] = []
    dedup_p: list[np.ndarray] = []
    for ts, pos in zip(ordered_t, ordered_p):
        if dedup_t and abs(float(ts) - float(dedup_t[-1])) <= 1e-9:
            dedup_p[-1] = np.asarray(pos, dtype=np.float64)
        else:
            dedup_t.append(float(ts))
            dedup_p.append(np.asarray(pos, dtype=np.float64))
    ordered_t = np.asarray(dedup_t, dtype=np.float64)
    ordered_p = np.asarray(dedup_p, dtype=np.float64)

    if len(ordered_t) == 1:
        notes.append("duplicate timestamps collapsed to one point; only exact matches are valid")
        return _resample_to_canonical(ordered_p, ordered_t)

    t_min = float(ordered_t[0])
    t_max = float(ordered_t[-1])

    if not _ENABLE_LINEAR_EXTRAPOLATION:
        valid_mask = (CANONICAL_TIMESTAMPS >= t_min - 1e-9) & (CANONICAL_TIMESTAMPS <= t_max + 1e-9)
        if np.any(valid_mask):
            canonical_positions[valid_mask, 0] = np.interp(
                CANONICAL_TIMESTAMPS[valid_mask], ordered_t, ordered_p[:, 0]
            )
            canonical_positions[valid_mask, 1] = np.interp(
                CANONICAL_TIMESTAMPS[valid_mask], ordered_t, ordered_p[:, 1]
            )
        if not np.all(valid_mask):
            notes.append("canonical valid_mask excludes timestamps beyond native horizon coverage")
        return canonical_positions, valid_mask, "linear_world_interp_no_extrapolation", notes

    # Stage-4 optional mode: linearly extrapolate beyond native horizon so
    # short-horizon planners can still be compared on the canonical 3.0 s grid.
    dt_head = float(ordered_t[1] - ordered_t[0])
    dt_tail = float(ordered_t[-1] - ordered_t[-2])
    if abs(dt_head) <= 1e-12 or abs(dt_tail) <= 1e-12:
        valid_mask = (CANONICAL_TIMESTAMPS >= t_min - 1e-9) & (CANONICAL_TIMESTAMPS <= t_max + 1e-9)
        if np.any(valid_mask):
            canonical_positions[valid_mask, 0] = np.interp(
                CANONICAL_TIMESTAMPS[valid_mask], ordered_t, ordered_p[:, 0]
            )
            canonical_positions[valid_mask, 1] = np.interp(
                CANONICAL_TIMESTAMPS[valid_mask], ordered_t, ordered_p[:, 1]
            )
        notes.append("degenerate timestamp spacing; fell back to no-extrapolation")
        if not np.all(valid_mask):
            notes.append("canonical valid_mask excludes timestamps beyond native horizon coverage")
        return canonical_positions, valid_mask, "linear_world_interp_no_extrapolation", notes

    head_vel = (ordered_p[1] - ordered_p[0]) / dt_head
    tail_vel = (ordered_p[-1] - ordered_p[-2]) / dt_tail
    n_extrapolated = 0
    for idx, ts in enumerate(CANONICAL_TIMESTAMPS):
        ts_f = float(ts)
        if t_min - 1e-9 <= ts_f <= t_max + 1e-9:
            canonical_positions[idx, 0] = float(np.interp(ts_f, ordered_t, ordered_p[:, 0]))
            canonical_positions[idx, 1] = float(np.interp(ts_f, ordered_t, ordered_p[:, 1]))
            valid_mask[idx] = True
            continue
        if ts_f < t_min:
            canonical_positions[idx] = ordered_p[0] + head_vel * (ts_f - t_min)
            valid_mask[idx] = True
            n_extrapolated += 1
            continue
        if ts_f > t_max:
            canonical_positions[idx] = ordered_p[-1] + tail_vel * (ts_f - t_max)
            valid_mask[idx] = True
            n_extrapolated += 1

    if n_extrapolated > 0:
        notes.append(
            f"linearly extrapolated {n_extrapolated} canonical timestamp(s) beyond native horizon"
        )
    method = (
        "linear_world_interp_with_linear_extrapolation"
        if n_extrapolated > 0
        else "linear_world_interp_no_extrapolation"
    )
    return canonical_positions, valid_mask, method, notes


def build_canonical_trajectory(
    *,
    planner_name: str,
    scenario_id: str,
    frame_id: int,
    raw_export_record: dict[str, Any],
    source_world_positions: Any,
) -> CanonicalTrajectory:
    notes = list(raw_export_record.get("adapter_debug_notes", []) or [])
    world_positions = _as_xy_array(source_world_positions)
    if world_positions is None:
        notes.append("adapter did not expose source world positions; canonical positions left invalid")
        return CanonicalTrajectory(
            planner_name=str(planner_name),
            scenario_id=str(scenario_id),
            frame_id=int(frame_id),
            canonical_timestamps=CANONICAL_TIMESTAMPS.copy(),
            canonical_positions=np.zeros((len(CANONICAL_TIMESTAMPS), 2), dtype=np.float64),
            valid_mask=np.zeros(len(CANONICAL_TIMESTAMPS), dtype=bool),
            canonical_frame_description="world frame x=world_x, y=world_y (unavailable)",
            timestamp_source="unresolved",
            interpolation_method="unresolved",
            is_fresh_plan=bool(raw_export_record.get("is_fresh_plan", False)),
            provenance_debug_notes=notes,
        )

    world_positions, cumulative_notes = _reconstruct_absolute_positions(
        world_positions,
        is_cumulative_positions=raw_export_record.get("is_cumulative_positions"),
    )
    notes.extend(cumulative_notes)

    timestamps, timestamp_source, timing_notes = _infer_timestamps(
        len(world_positions),
        native_timestamps=raw_export_record.get("native_timestamps"),
        native_dt=raw_export_record.get("native_dt"),
        point0_mode=raw_export_record.get("point0_mode"),
    )
    notes.extend(timing_notes)
    timestamps, timestamp_source, downgraded_timing_notes = _downgrade_adapter_derived_timing_to_inferred(
        str(planner_name),
        n_points=len(world_positions),
        timestamps=timestamps,
        timestamp_source=timestamp_source,
        native_dt=raw_export_record.get("native_dt"),
        point0_mode=raw_export_record.get("point0_mode"),
    )
    notes.extend(downgraded_timing_notes)
    notes.extend(
        _document_inferred_adapter_timing(
            str(planner_name),
            timestamp_source=timestamp_source,
            native_timestamps=raw_export_record.get("native_timestamps"),
            native_dt=raw_export_record.get("native_dt"),
            point0_mode=raw_export_record.get("point0_mode"),
        )
    )
    if timestamps is None:
        return CanonicalTrajectory(
            planner_name=str(planner_name),
            scenario_id=str(scenario_id),
            frame_id=int(frame_id),
            canonical_timestamps=CANONICAL_TIMESTAMPS.copy(),
            canonical_positions=np.zeros((len(CANONICAL_TIMESTAMPS), 2), dtype=np.float64),
            valid_mask=np.zeros(len(CANONICAL_TIMESTAMPS), dtype=bool),
            canonical_frame_description="world frame x=world_x, y=world_y (timing unresolved)",
            timestamp_source="unresolved",
            interpolation_method="unresolved",
            is_fresh_plan=bool(raw_export_record.get("is_fresh_plan", False)),
            provenance_debug_notes=notes,
        )

    canonical_positions, valid_mask, interpolation_method, interp_notes = _resample_to_canonical(
        world_positions, timestamps
    )
    notes.extend(interp_notes)
    return CanonicalTrajectory(
        planner_name=str(planner_name),
        scenario_id=str(scenario_id),
        frame_id=int(frame_id),
        canonical_timestamps=CANONICAL_TIMESTAMPS.copy(),
        canonical_positions=canonical_positions,
        valid_mask=valid_mask,
        canonical_frame_description=(
            "world frame x=world_x, y=world_y derived from Stage 1 adapter-converted source positions"
        ),
        timestamp_source=timestamp_source,
        interpolation_method=interpolation_method,
        is_fresh_plan=bool(raw_export_record.get("is_fresh_plan", False)),
        provenance_debug_notes=notes,
    )


def compute_gt_canonical_future(
    gt_traj_all: np.ndarray,
    *,
    frame_idx: int,
    runtime_dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    gt = np.asarray(gt_traj_all, dtype=np.float64)
    target_times = CANONICAL_TIMESTAMPS
    gt_positions = np.zeros((len(target_times), 2), dtype=np.float64)
    gt_valid_mask = np.zeros(len(target_times), dtype=bool)
    if len(gt) == 0 or runtime_dt <= 0.0:
        return gt_positions, gt_valid_mask

    for idx, offset_s in enumerate(target_times):
        frame_offset = float(offset_s) / float(runtime_dt)
        src = float(frame_idx) + frame_offset
        lo = int(np.floor(src))
        hi = int(np.ceil(src))
        if lo < 0 or hi >= len(gt):
            continue
        if lo == hi:
            gt_positions[idx] = gt[lo, :2]
            gt_valid_mask[idx] = True
            continue
        alpha = src - float(lo)
        gt_positions[idx] = gt[lo, :2] * (1.0 - alpha) + gt[hi, :2] * alpha
        gt_valid_mask[idx] = True
    return gt_positions, gt_valid_mask


def canonical_to_record(
    canonical: CanonicalTrajectory,
    *,
    raw_export_record: dict[str, Any],
    source_world_positions: Any,
    gt_canonical_positions: Optional[np.ndarray] = None,
    gt_valid_mask: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    source_world = _as_xy_array(source_world_positions)
    return {
        "planner_name": canonical.planner_name,
        "scenario_id": canonical.scenario_id,
        "frame_id": canonical.frame_id,
        "canonical_timestamps": canonical.canonical_timestamps,
        "canonical_positions": canonical.canonical_positions,
        "valid_mask": canonical.valid_mask,
        "canonical_frame_description": canonical.canonical_frame_description,
        "timestamp_source": canonical.timestamp_source,
        "interpolation_method": canonical.interpolation_method,
        "is_fresh_plan": canonical.is_fresh_plan,
        "provenance_debug_notes": canonical.provenance_debug_notes,
        "raw_positions": raw_export_record.get("raw_positions"),
        "raw_shape": raw_export_record.get("raw_shape"),
        "raw_world_positions": source_world,
        "raw_source_frame_description": raw_export_record.get("source_frame_description"),
        "raw_axis_convention_description": raw_export_record.get("axis_convention_description"),
        "raw_point0_mode": raw_export_record.get("point0_mode"),
        "raw_native_dt": raw_export_record.get("native_dt"),
        "gt_canonical_positions": gt_canonical_positions,
        "gt_valid_mask": gt_valid_mask,
    }
