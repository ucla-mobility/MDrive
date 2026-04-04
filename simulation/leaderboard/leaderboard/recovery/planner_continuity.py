#!/usr/bin/env python

"""
Planner continuity utilities for evaluator-side CARLA recovery.
"""

from __future__ import annotations

import copy
import json
from collections import deque
from typing import Any

from .serialization import serialize_value, deserialize_value


class PlannerContinuityManager:
    PUBLICATION_SAFE_ALLOWED_MODES = {"preserved_alive", "preserved_with_rebind"}

    def __init__(
        self,
        replay_window: int = 30,
        *,
        recovery_mode: str = "engineering",
        max_publication_state_bytes: int = 1_000_000,
    ) -> None:
        self.replay_window = max(1, int(replay_window))
        mode = str(recovery_mode or "engineering").strip().lower()
        if mode not in ("off", "engineering", "publication_safe"):
            mode = "engineering"
        self.recovery_mode = mode
        self.publication_safe = mode == "publication_safe"
        self.max_publication_state_bytes = max(4096, int(max_publication_state_bytes))
        self._replay_buffer: deque[dict[str, Any]] = deque(maxlen=self.replay_window)

    def observe_runtime(self, agent_instance: Any) -> None:
        if self.publication_safe:
            # Publication-safe mode never stores generic raw observations for replay.
            return
        if agent_instance is None:
            return
        timestamp = getattr(agent_instance, "_checkpoint_last_timestamp", None)
        input_data = getattr(agent_instance, "_checkpoint_last_input_data", None)
        output_data = getattr(agent_instance, "_checkpoint_last_output", None)
        if timestamp is None and input_data is None and output_data is None:
            return
        record = {
            "timestamp": serialize_value(timestamp),
            "input_data": serialize_value(input_data),
            "output_data": serialize_value(output_data),
        }
        self._replay_buffer.append(record)

    @staticmethod
    def _estimate_serialized_size(payload: Any) -> int:
        try:
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            return int(len(encoded))
        except Exception:
            try:
                return int(len(repr(payload).encode("utf-8")))
            except Exception:
                return 0

    def snapshot(self, agent_instance: Any) -> dict[str, Any]:
        planner_state = None
        planner_state_supported = False
        planner_state_bytes = 0
        planner_state_dropped = False
        planner_state_drop_reason = ""
        if agent_instance is not None and hasattr(agent_instance, "get_checkpoint_state"):
            try:
                planner_state = serialize_value(agent_instance.get_checkpoint_state())
                planner_state_supported = True
                planner_state_bytes = self._estimate_serialized_size(planner_state)
                if self.publication_safe and planner_state_bytes > self.max_publication_state_bytes:
                    planner_state = None
                    planner_state_dropped = True
                    planner_state_drop_reason = "planner_state_too_large"
            except Exception:
                planner_state = None
                planner_state_supported = False
                planner_state_bytes = 0
                planner_state_dropped = False
                planner_state_drop_reason = ""
        replay_buffer = [] if self.publication_safe else list(self._replay_buffer)
        return {
            "planner_state_supported": bool(planner_state_supported),
            "planner_state": planner_state,
            "planner_state_bytes": int(planner_state_bytes),
            "planner_state_dropped": bool(planner_state_dropped),
            "planner_state_drop_reason": str(planner_state_drop_reason or ""),
            "replay_buffer": replay_buffer,
        }

    @staticmethod
    def _best_effort_rebind(
        agent_instance: Any,
        *,
        world: Any,
        client: Any,
        ego_vehicles: list[Any],
    ) -> bool:
        if agent_instance is None:
            return False
        changed = False
        ego0 = ego_vehicles[0] if ego_vehicles else None
        fields = (
            ("_world", world),
            ("world", world),
            ("_client", client),
            ("client", client),
            ("_map", world.get_map() if world is not None else None),
            ("_carla_map", world.get_map() if world is not None else None),
            ("_vehicle", ego0),
            ("vehicle", ego0),
            ("_vehicles", list(ego_vehicles)),
        )
        for field_name, field_value in fields:
            if hasattr(agent_instance, field_name):
                try:
                    setattr(agent_instance, field_name, field_value)
                    changed = True
                except Exception:
                    continue
        return changed

    def restore(
        self,
        agent_instance: Any,
        planner_snapshot: dict[str, Any] | None,
        *,
        world: Any,
        client: Any,
        ego_vehicles: list[Any],
    ) -> tuple[str, list[str]]:
        if agent_instance is None:
            return "stateless_restart", ["agent_instance_missing"]

        notes: list[str] = []
        mode = "preserved_alive"

        rebind_done = False
        if hasattr(agent_instance, "rebind_after_recovery"):
            try:
                context = {
                    "world": world,
                    "client": client,
                    "ego_vehicles": ego_vehicles,
                }
                rebind_done = bool(agent_instance.rebind_after_recovery(context))
            except Exception as exc:
                notes.append(f"planner_rebind_hook_error:{type(exc).__name__}")
        if not rebind_done:
            rebind_done = self._best_effort_rebind(
                agent_instance,
                world=world,
                client=client,
                ego_vehicles=ego_vehicles,
            )
        if rebind_done:
            mode = "preserved_with_rebind"
        else:
            notes.append("planner_rebind_not_supported")

        snapshot = planner_snapshot or {}
        planner_state_supported = bool(snapshot.get("planner_state_supported"))
        planner_state = snapshot.get("planner_state")
        if planner_state_supported and planner_state is not None and hasattr(agent_instance, "set_checkpoint_state"):
            try:
                agent_instance.set_checkpoint_state(deserialize_value(planner_state))
            except Exception as exc:
                notes.append(f"planner_state_restore_error:{type(exc).__name__}")
        elif planner_state_supported and planner_state is not None:
            notes.append("planner_state_restore_hook_missing")

        replay_buffer = snapshot.get("replay_buffer", []) if isinstance(snapshot, dict) else []
        if self.publication_safe:
            if snapshot.get("planner_state_dropped", False):
                notes.append(f"planner_state_dropped:{snapshot.get('planner_state_drop_reason', 'unknown')}")
            # Publication-safe mode disallows generic replay fallback.
            return mode, notes

        if replay_buffer:
            replay_mode_used = False
            if hasattr(agent_instance, "replay_from_checkpoint"):
                try:
                    agent_instance.replay_from_checkpoint(copy.deepcopy(replay_buffer))
                    replay_mode_used = True
                except Exception as exc:
                    notes.append(f"planner_replay_hook_error:{type(exc).__name__}")
            elif hasattr(agent_instance, "checkpoint_replay_step"):
                try:
                    for record in replay_buffer:
                        agent_instance.checkpoint_replay_step(copy.deepcopy(record))
                    replay_mode_used = True
                except Exception as exc:
                    notes.append(f"planner_replay_step_error:{type(exc).__name__}")
            if replay_mode_used:
                mode = "restored_via_replay"
            else:
                notes.append("planner_replay_not_supported")
        return mode, notes
