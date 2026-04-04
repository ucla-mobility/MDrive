#!/usr/bin/env python

"""
In-process CARLA crash recovery orchestration for leaderboard evaluator.
"""

from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import carla
except Exception:  # pragma: no cover
    carla = None

from leaderboard.autoagents.agent_wrapper import AgentError

from .checkpoint_manager import CheckpointManager
from .evaluation_state_manager import EvaluationStateManager
from .planner_continuity import PlannerContinuityManager
from .provenance_logger import FrameProvenanceLogger, RecoveryJournal


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return int(default)
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None:
        return str(default)
    return str(value)


class RecoveryManager:
    ALLOWED_RECOVERY_MODES = ("off", "engineering", "publication_safe")

    def __init__(self, *, evaluator: Any, args: Any, run_root: str | Path) -> None:
        self.evaluator = evaluator
        self.args = args
        self.run_root = Path(run_root)
        self.recovery_dir = self.run_root / "recovery"
        self.recovery_dir.mkdir(parents=True, exist_ok=True)

        raw_mode = _env_str("CARLA_RECOVERY_MODE", "off").strip().lower()
        self.recovery_mode = raw_mode if raw_mode in self.ALLOWED_RECOVERY_MODES else "off"
        self.publication_safe = self.recovery_mode == "publication_safe"
        enabled_flag = _env_flag("CARLA_CHECKPOINT_RECOVERY_ENABLE", False)
        self.enabled = bool(enabled_flag and self.recovery_mode != "off")
        self.max_recoveries = max(0, _env_int("CARLA_CHECKPOINT_MAX_RECOVERIES", 5))
        self.restart_wait_timeout_s = max(5.0, _env_float("CARLA_RECOVERY_WAIT_TIMEOUT_S", 180.0))
        self.restart_cmd = os.environ.get("CARLA_RECOVERY_RESTART_CMD", "").strip()
        self.checkpoint_interval_frames = max(1, _env_int("CARLA_CHECKPOINT_INTERVAL_FRAMES", 20))
        self.max_checkpoints = max(0, _env_int("CARLA_CHECKPOINT_MAX_COUNT", 0))
        self.replay_window = max(1, _env_int("CARLA_PLANNER_REPLAY_WINDOW", 40))
        self.checkpoint_queue_size = max(1, _env_int("CARLA_CHECKPOINT_QUEUE_SIZE", 1))
        self.checkpoint_async_writes = _env_flag("CARLA_CHECKPOINT_ASYNC_WRITES", True)
        self.unsafe_recovery_radius_m = max(0.0, _env_float("CARLA_RECOVERY_UNSAFE_RADIUS_M", 12.0))

        self.checkpoints = CheckpointManager(
            self.run_root,
            checkpoint_interval_frames=self.checkpoint_interval_frames,
            max_checkpoints=self.max_checkpoints,
            async_writes=self.checkpoint_async_writes,
            queue_size=self.checkpoint_queue_size,
        )
        self.state_manager = EvaluationStateManager(
            nearby_actor_radius=max(5.0, _env_float("CARLA_CHECKPOINT_NEARBY_RADIUS_M", 90.0)),
            traffic_light_radius=max(5.0, _env_float("CARLA_CHECKPOINT_TL_RADIUS_M", 120.0)),
            actor_match_distance=max(1.0, _env_float("CARLA_CHECKPOINT_ACTOR_MATCH_M", 6.0)),
        )
        self.planner = PlannerContinuityManager(
            replay_window=self.replay_window,
            recovery_mode=self.recovery_mode,
            max_publication_state_bytes=max(
                4096,
                _env_int("CARLA_PUBLICATION_SAFE_PLANNER_STATE_MAX_BYTES", 1_000_000),
            ),
        )
        self.journal = RecoveryJournal(self.recovery_dir)
        self.provenance = FrameProvenanceLogger(self.recovery_dir)
        self.recovered_manifest_path = self.recovery_dir / "recovered_routes_manifest.jsonl"

        self.crash_generation = 0
        self.segment_id = self.provenance.start_segment(checkpoint_id=None, crash_generation=0)
        self.current_route_context: dict[str, Any] = {}
        self.current_manager = None
        self.current_scenario = None
        self.current_checkpoint_id: int | None = None
        self.pending_checkpoint_id: int | None = None
        self.route_recovered = False
        self.route_partial_restore = False
        self.route_crash_count = 0
        self.planner_modes: list[str] = []
        self.actor_restore_notes: list[str] = []
        self.last_planner_mode = "preserved_alive"
        self.last_recovery_reason = ""
        self.last_recovery_failure_kind = ""
        self.last_recovery_failure_reason = ""
        self.route_invalid_for_benchmark = False
        self.route_benchmark_eligible_candidate = True

    def start_route(self, *, config: Any) -> None:
        self.current_route_context = {
            "route_name": str(getattr(config, "name", "")),
            "route_index": int(getattr(config, "index", -1)),
            "route_repetition": int(getattr(config, "repetition_index", 0)),
        }
        self._drain_checkpoint_events()
        self.current_checkpoint_id = None
        self.pending_checkpoint_id = None
        self.route_recovered = False
        self.route_partial_restore = False
        self.route_crash_count = 0
        self.planner_modes = []
        self.actor_restore_notes = []
        self.last_planner_mode = "preserved_alive"
        self.last_recovery_reason = ""
        self.last_recovery_failure_kind = ""
        self.last_recovery_failure_reason = ""
        self.route_invalid_for_benchmark = False
        self.route_benchmark_eligible_candidate = True
        self.segment_id = self.provenance.start_segment(checkpoint_id=None, crash_generation=self.crash_generation)
        self.journal.append(
            "route_start",
            **self.current_route_context,
            crash_generation=int(self.crash_generation),
            recovery_enabled=int(bool(self.enabled)),
            recovery_mode=str(self.recovery_mode),
            checkpoint_interval_frames=int(self.checkpoint_interval_frames),
        )

    def bind_runtime(self, *, manager: Any, scenario: Any) -> None:
        self.current_manager = manager
        self.current_scenario = scenario
        if manager is not None and hasattr(manager, "set_external_tick_callback"):
            manager.set_external_tick_callback(self._on_tick)

    def route_metadata(self) -> dict[str, Any]:
        planner_mode = self.last_planner_mode
        if self.planner_modes:
            planner_mode = str(self.planner_modes[-1])
        checkpoint_id = self.current_checkpoint_id if self.current_checkpoint_id is not None else self.pending_checkpoint_id
        recovery_reason = self.last_recovery_failure_reason or self.last_recovery_reason
        benchmark_eligible = bool(self.route_benchmark_eligible_candidate and not self.route_invalid_for_benchmark)
        return {
            "route_recovered": bool(self.route_recovered),
            "recovery_attempted": bool(self.route_crash_count > 0),
            "recovery_count": int(self.route_crash_count),
            "planner_continuity_mode": str(planner_mode or ""),
            "recovered_approximate": bool(self.route_recovered),
            "checkpoint_id": checkpoint_id,
            "recovery_reason": str(recovery_reason or ""),
            "recovery_mode": str(self.recovery_mode),
            "recovery_failed": bool(self.last_recovery_failure_kind),
            "recovery_failure_kind": str(self.last_recovery_failure_kind or ""),
            "invalid_for_benchmark": bool(self.route_invalid_for_benchmark),
            "benchmark_eligible_candidate": bool(benchmark_eligible),
            "publication_safe_mode": bool(self.publication_safe),
            "crash_generation": int(self.route_crash_count),
            "crash_generation_total": int(self.crash_generation),
            "planner_modes": list(self.planner_modes),
            "partial_restore": bool(self.route_partial_restore),
            "actor_restore_notes": list(self.actor_restore_notes),
            "current_checkpoint_id": checkpoint_id,
        }

    @staticmethod
    def _is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    @staticmethod
    def _speed_from_velocity_dict(payload: dict[str, Any] | None) -> float | None:
        if not isinstance(payload, dict):
            return None
        try:
            vx = float(payload.get("x", 0.0))
            vy = float(payload.get("y", 0.0))
            vz = float(payload.get("z", 0.0))
            return (vx * vx + vy * vy + vz * vz) ** 0.5
        except Exception:
            return None

    def _wait_for_carla_ready(self) -> bool:
        host = str(getattr(self.args, "host", "127.0.0.1"))
        port = int(getattr(self.args, "port", 2000))
        deadline = time.monotonic() + self.restart_wait_timeout_s
        while time.monotonic() < deadline:
            if not self._is_port_open(host, port, timeout=0.5):
                time.sleep(0.5)
                continue
            if carla is None:
                return True
            try:
                client = carla.Client(host, port)
                client.set_timeout(3.0)
                _ = client.get_world()
                return True
            except Exception:
                time.sleep(0.5)
        return False

    def _restart_carla(self, failure_reason: str) -> bool:
        if self.restart_cmd:
            try:
                if os.environ.get("CARLA_RECOVERY_RESTART_SHELL", "1").strip() in ("0", "false", "False"):
                    cmd = shlex.split(self.restart_cmd)
                    subprocess.run(cmd, check=False, timeout=max(30.0, self.restart_wait_timeout_s))
                else:
                    subprocess.run(
                        self.restart_cmd,
                        shell=True,
                        check=False,
                        timeout=max(30.0, self.restart_wait_timeout_s),
                    )
                self.journal.append(
                    "carla_restart_invoked",
                    reason=failure_reason,
                    command=self.restart_cmd,
                )
            except Exception as exc:
                self.journal.append(
                    "carla_restart_failed",
                    reason=failure_reason,
                    command=self.restart_cmd,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                return False
        return self._wait_for_carla_ready()

    @staticmethod
    def _is_infrastructure_failure(exc: BaseException) -> bool:
        if isinstance(exc, AgentError):
            return False
        text = "{} {}".format(type(exc).__name__, str(exc)).lower()
        markers = (
            "sensor took too long",
            "sensorreceivednodata",
            "rpc",
            "connection",
            "carla",
            "timeout",
            "stream lost",
            "world snapshot",
            "failed to connect",
            "socket",
        )
        return any(marker in text for marker in markers)

    def _drain_checkpoint_events(self) -> None:
        for result in self.checkpoints.drain_completed():
            if result.ok and result.ref is not None:
                self.current_checkpoint_id = int(result.ref.index)
                if self.pending_checkpoint_id == self.current_checkpoint_id:
                    self.pending_checkpoint_id = None
                self.journal.append(
                    "checkpoint_committed",
                    checkpoint_id=int(result.ref.index),
                    logical_frame_id=int(result.ref.meta.get("logical_frame_id", -1)),
                    route_name=str(result.ref.meta.get("route_name", "")),
                    route_index=int(result.ref.meta.get("route_index", -1)),
                    crash_generation=int(result.ref.meta.get("crash_generation", self.crash_generation)),
                    segment_id=int(result.ref.meta.get("segment_id", self.segment_id)),
                    reason=str(result.ref.meta.get("reason", "periodic")),
                    snapshot_prepare_ms=float(result.snapshot_prepare_ms),
                    checkpoint_write_ms=float(result.checkpoint_write_ms),
                    checkpoint_payload_bytes=int(result.checkpoint_payload_bytes),
                    checkpoint_queue_depth=int(result.checkpoint_queue_depth),
                    skipped_checkpoints_due_to_backpressure=int(
                        result.skipped_checkpoints_due_to_backpressure
                    ),
                )
                continue
            self.journal.append(
                "checkpoint_commit_failed",
                checkpoint_id=result.checkpoint_id,
                error_type=str(result.error_type or ""),
                error=str(result.error or ""),
                snapshot_prepare_ms=float(result.snapshot_prepare_ms),
                checkpoint_write_ms=float(result.checkpoint_write_ms),
                checkpoint_queue_depth=int(result.checkpoint_queue_depth),
                skipped_checkpoints_due_to_backpressure=int(
                    result.skipped_checkpoints_due_to_backpressure
                ),
            )

    def _capture_checkpoint(self, tick_info: dict[str, Any], reason: str = "periodic") -> None:
        if self.current_manager is None:
            return
        route_name = self.current_route_context.get("route_name", "")
        route_index = int(self.current_route_context.get("route_index", -1))
        logical_frame_id = int(tick_info.get("logical_frame_id", 0))
        snapshot_start = time.perf_counter()
        planner_snapshot = self.planner.snapshot(getattr(self.evaluator, "agent_instance", None))
        cfg_view = SimpleNamespace(
            name=str(route_name),
            index=int(route_index),
            repetition_index=int(self.current_route_context.get("route_repetition", 0)),
        )
        payload = self.state_manager.capture(
            evaluator=self.evaluator,
            manager=self.current_manager,
            config=cfg_view,
            scenario=self.current_scenario,
            planner_snapshot=planner_snapshot,
            crash_generation=self.crash_generation,
            segment_id=self.segment_id,
        )
        snapshot_prepare_ms = max(0.0, (time.perf_counter() - snapshot_start) * 1000.0)
        schedule = self.checkpoints.enqueue_commit(
            payload,
            logical_frame_id=logical_frame_id,
            route_name=str(route_name),
            route_index=int(route_index),
            crash_generation=int(self.crash_generation),
            segment_id=int(self.segment_id),
            reason=reason,
            snapshot_prepare_ms=float(snapshot_prepare_ms),
        )
        if not bool(schedule.get("scheduled", False)):
            self.journal.append(
                "checkpoint_skipped_backpressure",
                logical_frame_id=int(logical_frame_id),
                route_name=str(route_name),
                route_index=int(route_index),
                crash_generation=int(self.crash_generation),
                segment_id=int(self.segment_id),
                reason=str(reason),
                snapshot_prepare_ms=float(snapshot_prepare_ms),
                checkpoint_queue_depth=int(schedule.get("queue_depth", 0) or 0),
                skipped_checkpoints_due_to_backpressure=int(
                    schedule.get("skipped_checkpoints_due_to_backpressure", 0) or 0
                ),
            )
            return
        checkpoint_id = schedule.get("checkpoint_id")
        if checkpoint_id is not None:
            self.pending_checkpoint_id = int(checkpoint_id)
        self.journal.append(
            "checkpoint_enqueued",
            checkpoint_id=checkpoint_id,
            logical_frame_id=int(logical_frame_id),
            route_name=str(route_name),
            route_index=int(route_index),
            crash_generation=int(self.crash_generation),
            segment_id=int(self.segment_id),
            reason=str(reason),
            snapshot_prepare_ms=float(snapshot_prepare_ms),
            checkpoint_queue_depth=int(schedule.get("queue_depth", 0) or 0),
            skipped_checkpoints_due_to_backpressure=int(
                schedule.get("skipped_checkpoints_due_to_backpressure", 0) or 0
            ),
        )
        return

    def _on_tick(self, tick_info: dict[str, Any]) -> None:
        self._drain_checkpoint_events()
        logical_frame_id = int(tick_info.get("logical_frame_id", 0))
        carla_frame = tick_info.get("carla_frame", None)
        checkpoint_for_frame = self.current_checkpoint_id
        if checkpoint_for_frame is None:
            checkpoint_for_frame = self.pending_checkpoint_id
        self.provenance.log_frame(
            logical_frame_id=logical_frame_id,
            checkpoint_id=checkpoint_for_frame,
            crash_generation=self.crash_generation,
            carla_episode_frame=int(carla_frame) if carla_frame is not None else None,
            frame_origin="recovered" if self.route_recovered else "original",
        )
        self.planner.observe_runtime(getattr(self.evaluator, "agent_instance", None))
        if self.enabled and self.checkpoints.should_checkpoint(logical_frame_id):
            self._capture_checkpoint(tick_info, reason="periodic")

    def try_recover(
        self,
        *,
        exc: BaseException,
        args: Any,
        config: Any,
        scenario_parameter: dict[str, Any],
        log_dir: str,
    ) -> tuple[bool, Any]:
        self._drain_checkpoint_events()
        self.last_recovery_reason = type(exc).__name__
        self.last_recovery_failure_kind = ""
        self.last_recovery_failure_reason = ""
        if not self.enabled:
            return False, None
        if not self._is_infrastructure_failure(exc):
            return False, None
        if self.crash_generation >= self.max_recoveries:
            self.last_recovery_failure_kind = "recovery_exhausted"
            self.last_recovery_failure_reason = "max_recoveries_exhausted"
            if self.publication_safe:
                self.route_invalid_for_benchmark = True
                self.route_benchmark_eligible_candidate = False
            self.journal.append(
                "recovery_exhausted",
                crash_generation=int(self.crash_generation),
                max_recoveries=int(self.max_recoveries),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return False, None
        ref, payload = self.checkpoints.load_latest_payload()
        if ref is None or payload is None:
            self.last_recovery_failure_kind = "recovery_failed_no_checkpoint"
            self.last_recovery_failure_reason = "no_valid_checkpoint"
            if self.publication_safe:
                self.route_invalid_for_benchmark = True
                self.route_benchmark_eligible_candidate = False
            self.journal.append(
                "recovery_failed_no_checkpoint",
                crash_generation=int(self.crash_generation),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return False, None

        self.crash_generation += 1
        self.route_crash_count += 1
        checkpoint_ego_states = payload.get("ego_states", []) or []
        pre_recovery_speed = None
        if checkpoint_ego_states and isinstance(checkpoint_ego_states[0], dict):
            pre_recovery_speed = self._speed_from_velocity_dict(
                checkpoint_ego_states[0].get("velocity")
            )
        self.journal.append(
            "crash_detected",
            crash_generation=int(self.crash_generation),
            checkpoint_id=int(ref.index),
            route_name=str(getattr(config, "name", "")),
            route_index=int(getattr(config, "index", -1)),
            error_type=type(exc).__name__,
            error=str(exc),
            pre_recovery_speed_mps=pre_recovery_speed,
        )

        if not self._restart_carla(failure_reason=type(exc).__name__):
            self.last_recovery_failure_kind = "recovery_failed_restart"
            self.last_recovery_failure_reason = "carla_restart_failed"
            if self.publication_safe:
                self.route_invalid_for_benchmark = True
                self.route_benchmark_eligible_candidate = False
            self.journal.append(
                "recovery_failed_restart",
                crash_generation=int(self.crash_generation),
                checkpoint_id=int(ref.index),
            )
            return False, None

        scenario = self.evaluator._rebuild_route_runtime_for_recovery(
            args=args,
            config=config,
            scenario_parameter=scenario_parameter,
            log_dir=log_dir,
        )
        if scenario is None:
            self.last_recovery_failure_kind = "recovery_failed_rebuild"
            self.last_recovery_failure_reason = "runtime_rebuild_failed"
            if self.publication_safe:
                self.route_invalid_for_benchmark = True
                self.route_benchmark_eligible_candidate = False
            self.journal.append(
                "recovery_failed_rebuild",
                crash_generation=int(self.crash_generation),
                checkpoint_id=int(ref.index),
            )
            return False, None
        self.bind_runtime(manager=self.evaluator.manager, scenario=scenario)

        restore_report = self.state_manager.restore(
            evaluator=self.evaluator,
            manager=self.evaluator.manager,
            scenario=scenario,
            checkpoint_payload=payload,
        )
        post_recovery_speed = None
        ego_vehicles = list(getattr(self.evaluator, "ego_vehicles", []) or [])
        if ego_vehicles and ego_vehicles[0] is not None and hasattr(ego_vehicles[0], "get_velocity"):
            try:
                vec = ego_vehicles[0].get_velocity()
                post_recovery_speed = self._speed_from_velocity_dict(
                    {"x": getattr(vec, "x", 0.0), "y": getattr(vec, "y", 0.0), "z": getattr(vec, "z", 0.0)}
                )
            except Exception:
                post_recovery_speed = None
        self.route_partial_restore = bool(restore_report.get("partial_metric_restore", False))
        self.actor_restore_notes.extend(restore_report.get("actor_restore_mismatches", []) or [])
        planner_mode, planner_notes = self.planner.restore(
            getattr(self.evaluator, "agent_instance", None),
            payload.get("planner_snapshot", {}),
            world=getattr(self.evaluator, "world", None),
            client=getattr(self.evaluator, "client", None),
            ego_vehicles=list(getattr(self.evaluator, "ego_vehicles", []) or []),
        )
        self.last_planner_mode = str(planner_mode or "preserved_alive")
        self.planner_modes.append(planner_mode)
        if self.publication_safe and planner_mode not in PlannerContinuityManager.PUBLICATION_SAFE_ALLOWED_MODES:
            self.last_recovery_failure_kind = "failed_recovery_planner_continuity"
            self.last_recovery_failure_reason = "planner_continuity_not_publication_safe"
            self.route_invalid_for_benchmark = True
            self.route_benchmark_eligible_candidate = False
            self.journal.append(
                "recovery_failed_continuity",
                crash_generation=int(self.crash_generation),
                checkpoint_id=int(ref.index),
                planner_mode=str(planner_mode),
                planner_notes=planner_notes,
            )
            return False, None

        if self.publication_safe and self.route_partial_restore:
            self.route_invalid_for_benchmark = True
            self.route_benchmark_eligible_candidate = False

        self.segment_id = self.provenance.start_segment(
            checkpoint_id=ref.index,
            crash_generation=self.crash_generation,
            planner_mode=planner_mode,
            notes=";".join(planner_notes + (restore_report.get("actor_restore_mismatches", []) or [])),
        )
        self.route_recovered = True
        self.current_checkpoint_id = ref.index
        self.journal.append(
            "recovery_succeeded",
            crash_generation=int(self.crash_generation),
            checkpoint_id=int(ref.index),
            resumed_segment_id=int(self.segment_id),
            planner_mode=planner_mode,
            planner_notes=planner_notes,
            partial_metric_restore=int(bool(self.route_partial_restore)),
            actor_restore_mismatches=restore_report.get("actor_restore_mismatches", []),
            traffic_light_restore_notes=restore_report.get("traffic_light_restore_notes", []),
            restored_logical_frame_id=restore_report.get("restored_logical_frame_id"),
            pre_recovery_speed_mps=pre_recovery_speed,
            post_recovery_speed_mps=post_recovery_speed,
        )
        return True, scenario

    def record_route_outcome(
        self,
        *,
        ego_index: int,
        route_status: str,
        route_meta: dict[str, Any] | None,
    ) -> None:
        meta = dict(route_meta or {})
        recovered = bool(meta.get("route_recovered", False))
        recovery_failed = bool(meta.get("recovery_failed", False))
        if not recovered and not recovery_failed:
            return
        payload = {
            "timestamp_unix": time.time(),
            "route_name": str(self.current_route_context.get("route_name", "")),
            "route_index": int(self.current_route_context.get("route_index", -1)),
            "route_repetition": int(self.current_route_context.get("route_repetition", 0)),
            "ego_index": int(ego_index),
            "route_status": str(route_status or ""),
            "route_recovered": int(recovered),
            "recovery_count": int(meta.get("recovery_count", self.route_crash_count) or 0),
            "planner_continuity_mode": str(meta.get("planner_continuity_mode", self.last_planner_mode) or ""),
            "recovered_approximate": int(bool(meta.get("recovered_approximate", recovered))),
            "checkpoint_id": meta.get("checkpoint_id", self.current_checkpoint_id),
            "recovery_reason": str(
                meta.get("recovery_reason", self.last_recovery_failure_reason or self.last_recovery_reason) or ""
            ),
            "recovery_failed": int(recovery_failed),
            "recovery_failure_kind": str(meta.get("recovery_failure_kind", self.last_recovery_failure_kind) or ""),
            "invalid_for_benchmark": int(bool(meta.get("invalid_for_benchmark", self.route_invalid_for_benchmark))),
            "benchmark_eligible_candidate": int(
                bool(
                    meta.get(
                        "benchmark_eligible_candidate",
                        self.route_benchmark_eligible_candidate and not self.route_invalid_for_benchmark,
                    )
                )
            ),
            "recovery_mode": str(meta.get("recovery_mode", self.recovery_mode)),
        }
        line = "{}\n".format(json.dumps(payload, sort_keys=True))
        with self.recovered_manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    def close(self) -> None:
        self._drain_checkpoint_events()
        self.checkpoints.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
