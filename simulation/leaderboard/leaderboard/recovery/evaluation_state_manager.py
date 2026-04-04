#!/usr/bin/env python

"""
Capture/restore evaluator-side runtime state for checkpoint recovery.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Any

try:
    import carla
except Exception:  # pragma: no cover - test environments may not provide CARLA
    carla = None

try:
    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
    from srunner.scenariomanager.timer import GameTime
except Exception:  # pragma: no cover - unit tests without full CARLA stack
    class CarlaDataProvider:  # type: ignore[no-redef]
        @staticmethod
        def get_world():
            return None

    class GameTime:  # type: ignore[no-redef]
        @staticmethod
        def get_time():
            return 0.0

        @staticmethod
        def get_carla_time():
            return 0.0

        @staticmethod
        def get_frame():
            return 0

from .serialization import (
    deserialize_transform,
    deserialize_value,
    deserialize_vector,
    is_carla_actor,
    serialize_location,
    serialize_transform,
    serialize_value,
    serialize_vector,
)


CRITERION_SKIP_KEYS = {
    "logger",
    "blackbox_level",
    "parent",
    "children",
    "qualified_name",
    "feedback_message",
    "iterator",
}


class EvaluationStateManager:
    def __init__(
        self,
        *,
        nearby_actor_radius: float = 90.0,
        traffic_light_radius: float = 120.0,
        actor_match_distance: float = 6.0,
    ) -> None:
        self.nearby_actor_radius = max(5.0, float(nearby_actor_radius))
        self.traffic_light_radius = max(5.0, float(traffic_light_radius))
        self.actor_match_distance = max(1.0, float(actor_match_distance))

    @staticmethod
    def _distance_xy(a: dict[str, float], b: dict[str, float]) -> float:
        dx = float(a.get("x", 0.0)) - float(b.get("x", 0.0))
        dy = float(a.get("y", 0.0)) - float(b.get("y", 0.0))
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def _capture_vehicle_control(actor: Any) -> dict[str, Any] | None:
        if actor is None or not hasattr(actor, "get_control"):
            return None
        try:
            control = actor.get_control()
            return {
                "throttle": float(getattr(control, "throttle", 0.0)),
                "steer": float(getattr(control, "steer", 0.0)),
                "brake": float(getattr(control, "brake", 0.0)),
                "hand_brake": bool(getattr(control, "hand_brake", False)),
                "reverse": bool(getattr(control, "reverse", False)),
                "manual_gear_shift": bool(getattr(control, "manual_gear_shift", False)),
                "gear": int(getattr(control, "gear", 0)),
            }
        except Exception:
            return None

    @staticmethod
    def _capture_physics_hint(actor: Any) -> dict[str, Any] | None:
        if actor is None or not hasattr(actor, "get_physics_control"):
            return None
        try:
            physics = actor.get_physics_control()
            return {
                "mass": float(getattr(physics, "mass", 0.0)),
                "drag_coefficient": float(getattr(physics, "drag_coefficient", 0.0)),
                "use_gear_autobox": bool(getattr(physics, "use_gear_autobox", True)),
            }
        except Exception:
            return None

    def capture_actor_state(self, actor: Any, *, include_control: bool = False) -> dict[str, Any] | None:
        if actor is None:
            return None
        try:
            transform = actor.get_transform()
            velocity = actor.get_velocity()
            angular_velocity = actor.get_angular_velocity()
            acceleration = actor.get_acceleration()
        except Exception:
            return None
        role_name = ""
        blueprint_id = str(getattr(actor, "type_id", ""))
        try:
            role_name = str(actor.attributes.get("role_name", ""))
        except Exception:
            role_name = ""
        state: dict[str, Any] = {
            "actor_id": int(getattr(actor, "id", -1)),
            "blueprint_id": blueprint_id,
            "role_name": role_name,
            "transform": serialize_transform(transform),
            "velocity": serialize_vector(velocity),
            "angular_velocity": serialize_vector(angular_velocity),
            "acceleration": serialize_vector(acceleration),
            "is_alive": bool(getattr(actor, "is_alive", False)),
            "captured_unix_time": time.time(),
            "vehicle_light_state": None,
            "physics_hint": self._capture_physics_hint(actor),
        }
        if include_control:
            state["last_control"] = self._capture_vehicle_control(actor)
        if hasattr(actor, "get_light_state"):
            try:
                state["vehicle_light_state"] = int(actor.get_light_state())
            except Exception:
                state["vehicle_light_state"] = None
        return state

    def _collect_ego_anchor_locations(self, ego_states: list[dict[str, Any]]) -> list[dict[str, float]]:
        anchors: list[dict[str, float]] = []
        for ego_state in ego_states:
            transform = ego_state.get("transform", {}) if isinstance(ego_state, dict) else {}
            location = transform.get("location", {}) if isinstance(transform, dict) else {}
            if isinstance(location, dict):
                anchors.append(
                    {
                        "x": float(location.get("x", 0.0)),
                        "y": float(location.get("y", 0.0)),
                        "z": float(location.get("z", 0.0)),
                    }
                )
        return anchors

    def _near_any_anchor(self, actor_location: dict[str, float], anchors: list[dict[str, float]], radius: float) -> bool:
        for anchor in anchors:
            if self._distance_xy(actor_location, anchor) <= radius:
                return True
        return False

    def capture_nearby_actor_states(self, world: Any, ego_states: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if world is None:
            return []
        anchors = self._collect_ego_anchor_locations(ego_states)
        if not anchors:
            return []

        nearby_states: list[dict[str, Any]] = []
        try:
            actors = list(world.get_actors())
        except Exception:
            return nearby_states

        for actor in actors:
            if actor is None:
                continue
            type_id = str(getattr(actor, "type_id", ""))
            if not (type_id.startswith("vehicle.") or type_id.startswith("walker.")):
                continue
            try:
                role_name = str(actor.attributes.get("role_name", ""))
            except Exception:
                role_name = ""
            if role_name.startswith("hero_"):
                continue
            actor_state = self.capture_actor_state(actor, include_control=False)
            if actor_state is None:
                continue
            loc = actor_state.get("transform", {}).get("location", {})
            if not isinstance(loc, dict):
                continue
            if not self._near_any_anchor(loc, anchors, self.nearby_actor_radius):
                continue
            nearby_states.append(actor_state)
        nearby_states.sort(key=lambda item: (str(item.get("role_name", "")), str(item.get("blueprint_id", ""))))
        return nearby_states

    def capture_traffic_light_states(self, world: Any, ego_states: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if world is None:
            return []
        anchors = self._collect_ego_anchor_locations(ego_states)
        if not anchors:
            return []
        lights: list[dict[str, Any]] = []
        try:
            actors = list(world.get_actors().filter("*traffic_light*"))
        except Exception:
            return lights
        for light in actors:
            if light is None:
                continue
            try:
                transform = light.get_transform()
                location = serialize_location(transform.location)
                if location is None:
                    continue
                if not self._near_any_anchor(location, anchors, self.traffic_light_radius):
                    continue
                freeze_state = None
                if hasattr(light, "is_frozen"):
                    try:
                        freeze_state = bool(light.is_frozen())
                    except Exception:
                        freeze_state = None
                lights.append(
                    {
                        "transform": serialize_transform(transform),
                        "state": int(getattr(light, "state", 0)),
                        "frozen": freeze_state,
                    }
                )
            except Exception:
                continue
        return lights

    def _capture_criteria_state(self, manager: Any) -> list[dict[str, Any]]:
        snapshot: list[dict[str, Any]] = []
        scenario_list = getattr(manager, "scenario", []) or []
        for scenario_idx, scenario_instance in enumerate(scenario_list):
            criteria_entries: list[dict[str, Any]] = []
            if scenario_instance is not None and hasattr(scenario_instance, "get_criteria"):
                try:
                    criteria = scenario_instance.get_criteria() or []
                except Exception:
                    criteria = []
                for criterion in criteria:
                    state: dict[str, Any] = {}
                    for key, value in getattr(criterion, "__dict__", {}).items():
                        if key in CRITERION_SKIP_KEYS or key.startswith("_Behaviour__"):
                            continue
                        if key in ("actor", "_actor") or is_carla_actor(value):
                            continue
                        state[key] = serialize_value(value)
                    criteria_entries.append(
                        {
                            "name": str(getattr(criterion, "name", "")),
                            "class_name": criterion.__class__.__name__,
                            "module_name": criterion.__class__.__module__,
                            "state": state,
                        }
                    )
            timeout_state = {}
            timeout_node = getattr(scenario_instance, "timeout_node", None) if scenario_instance is not None else None
            if timeout_node is not None:
                timeout_state = {
                    "timeout": bool(getattr(timeout_node, "timeout", False)),
                    "start_time": serialize_value(getattr(timeout_node, "_start_time", 0.0)),
                }
            snapshot.append(
                {
                    "scenario_index": int(scenario_idx),
                    "criteria": criteria_entries,
                    "timeout_node": timeout_state,
                }
            )
        return snapshot

    def _restore_criteria_state(self, manager: Any, snapshot: list[dict[str, Any]]) -> bool:
        scenario_list = getattr(manager, "scenario", []) or []
        if not isinstance(snapshot, list):
            return False
        partial = False
        for scenario_entry in snapshot:
            if not isinstance(scenario_entry, dict):
                continue
            scenario_index = int(scenario_entry.get("scenario_index", -1))
            if scenario_index < 0 or scenario_index >= len(scenario_list):
                partial = True
                continue
            scenario_instance = scenario_list[scenario_index]
            if scenario_instance is None or not hasattr(scenario_instance, "get_criteria"):
                partial = True
                continue
            try:
                current_criteria = list(scenario_instance.get_criteria() or [])
            except Exception:
                current_criteria = []
            criteria_snapshot = scenario_entry.get("criteria", []) or []
            for idx, criterion_state in enumerate(criteria_snapshot):
                if idx >= len(current_criteria):
                    partial = True
                    break
                if not isinstance(criterion_state, dict):
                    partial = True
                    continue
                criterion = current_criteria[idx]
                expected_name = str(criterion_state.get("name", ""))
                expected_class = str(criterion_state.get("class_name", ""))
                if expected_name and expected_name != str(getattr(criterion, "name", "")):
                    partial = True
                if expected_class and expected_class != criterion.__class__.__name__:
                    partial = True
                state = criterion_state.get("state", {})
                if not isinstance(state, dict):
                    partial = True
                    continue
                for key, raw_value in state.items():
                    if key in ("actor", "_actor"):
                        continue
                    try:
                        setattr(criterion, key, deserialize_value(raw_value))
                    except Exception:
                        partial = True
            timeout_snapshot = scenario_entry.get("timeout_node", {}) or {}
            timeout_node = getattr(scenario_instance, "timeout_node", None)
            if timeout_node is not None and isinstance(timeout_snapshot, dict):
                try:
                    timeout_node.timeout = bool(timeout_snapshot.get("timeout", False))
                except Exception:
                    partial = True
                try:
                    timeout_node._start_time = deserialize_value(timeout_snapshot.get("start_time"))
                except Exception:
                    partial = True
        return partial

    def capture(
        self,
        *,
        evaluator: Any,
        manager: Any,
        config: Any,
        scenario: Any,
        planner_snapshot: dict[str, Any],
        crash_generation: int,
        segment_id: int,
    ) -> dict[str, Any]:
        world = getattr(evaluator, "world", None) or CarlaDataProvider.get_world()
        logical_frame = (
            manager.get_logical_frame_id()
            if hasattr(manager, "get_logical_frame_id")
            else int(getattr(manager, "_logical_frame_id", 0))
        )
        ego_states = []
        for ego_actor in list(getattr(evaluator, "ego_vehicles", []) or []):
            state = self.capture_actor_state(ego_actor, include_control=True)
            if state is not None:
                ego_states.append(state)

        manager_state = {}
        if hasattr(manager, "export_checkpoint_state"):
            try:
                manager_state = manager.export_checkpoint_state()
            except Exception:
                manager_state = {}

        stats_state = []
        for stat_mgr in list(getattr(evaluator, "statistics_manager", []) or []):
            if hasattr(stat_mgr, "snapshot_state"):
                try:
                    stats_state.append(stat_mgr.snapshot_state())
                except Exception:
                    stats_state.append(None)
            else:
                stats_state.append(None)

        checkpoint = {
            "format_version": 1,
            "created_unix_time": time.time(),
            "route_name": str(getattr(config, "name", "")),
            "route_index": int(getattr(config, "index", -1)),
            "route_repetition": int(getattr(config, "repetition_index", 0)),
            "logical_frame_id": int(logical_frame),
            "game_time": float(GameTime.get_time()),
            "carla_time": float(GameTime.get_carla_time()),
            "carla_frame": int(GameTime.get_frame()),
            "crash_generation": int(crash_generation),
            "segment_id": int(segment_id),
            "ego_states": ego_states,
            "nearby_actor_states": self.capture_nearby_actor_states(world, ego_states),
            "traffic_light_states": self.capture_traffic_light_states(world, ego_states),
            "manager_state": manager_state,
            "criteria_state": self._capture_criteria_state(manager),
            "statistics_state": stats_state,
            "planner_snapshot": copy.deepcopy(planner_snapshot),
            "route_scenario_debug": {
                "has_scenario": bool(scenario is not None),
                "sampled_scenarios_count": len(getattr(scenario, "sampled_scenarios_definitions", []) or [])
                if scenario is not None
                else 0,
            },
        }
        return checkpoint

    @staticmethod
    def _apply_vehicle_control(actor: Any, control_payload: dict[str, Any] | None) -> None:
        if carla is None or actor is None or control_payload is None:
            return
        if not hasattr(actor, "apply_control"):
            return
        try:
            control = carla.VehicleControl()
            control.throttle = float(control_payload.get("throttle", 0.0))
            control.steer = float(control_payload.get("steer", 0.0))
            control.brake = float(control_payload.get("brake", 0.0))
            control.hand_brake = bool(control_payload.get("hand_brake", False))
            control.reverse = bool(control_payload.get("reverse", False))
            control.manual_gear_shift = bool(control_payload.get("manual_gear_shift", False))
            control.gear = int(control_payload.get("gear", 0))
            actor.apply_control(control)
        except Exception:
            return

    def _apply_actor_state(
        self,
        actor: Any,
        state: dict[str, Any],
        *,
        apply_control: bool = False,
    ) -> bool:
        if actor is None or not isinstance(state, dict):
            return False
        success = True
        transform = deserialize_transform(state.get("transform"))
        velocity = deserialize_vector(state.get("velocity"))
        angular_velocity = deserialize_vector(state.get("angular_velocity"))
        if transform is not None and hasattr(actor, "set_transform"):
            try:
                actor.set_transform(transform)
            except Exception:
                success = False
        if velocity is not None and hasattr(actor, "set_target_velocity"):
            try:
                actor.set_target_velocity(velocity)
            except Exception:
                success = False
        if angular_velocity is not None and hasattr(actor, "set_target_angular_velocity"):
            try:
                actor.set_target_angular_velocity(angular_velocity)
            except Exception:
                success = False
        if carla is not None and hasattr(actor, "set_light_state"):
            light_state = state.get("vehicle_light_state", None)
            if light_state is not None:
                try:
                    actor.set_light_state(carla.VehicleLightState(int(light_state)))
                except Exception:
                    success = False
        if apply_control:
            self._apply_vehicle_control(actor, state.get("last_control"))
        return success

    @staticmethod
    def _actor_location_dict(actor: Any) -> dict[str, float]:
        try:
            tf = actor.get_transform()
            return {
                "x": float(tf.location.x),
                "y": float(tf.location.y),
                "z": float(tf.location.z),
            }
        except Exception:
            return {"x": 0.0, "y": 0.0, "z": 0.0}

    def _match_or_spawn_actor(self, world: Any, state: dict[str, Any], existing: list[Any]) -> Any:
        role_name = str(state.get("role_name", "") or "")
        blueprint_id = str(state.get("blueprint_id", "") or "")
        target_loc = state.get("transform", {}).get("location", {}) if isinstance(state.get("transform"), dict) else {}

        if role_name:
            for actor in existing:
                try:
                    if str(actor.attributes.get("role_name", "")) == role_name:
                        return actor
                except Exception:
                    continue

        best_actor = None
        best_distance = float("inf")
        for actor in existing:
            if str(getattr(actor, "type_id", "")) != blueprint_id:
                continue
            dist = self._distance_xy(self._actor_location_dict(actor), target_loc if isinstance(target_loc, dict) else {})
            if dist < best_distance:
                best_distance = dist
                best_actor = actor
        if best_actor is not None and best_distance <= self.actor_match_distance:
            return best_actor

        transform = deserialize_transform(state.get("transform"))
        if world is None or transform is None or not blueprint_id:
            return None
        try:
            bp_library = world.get_blueprint_library()
            blueprint = bp_library.find(blueprint_id)
            if role_name and blueprint.has_attribute("role_name"):
                blueprint.set_attribute("role_name", role_name)
            return world.try_spawn_actor(blueprint, transform)
        except Exception:
            return None

    def _restore_nearby_actors(
        self,
        world: Any,
        nearby_states: list[dict[str, Any]],
    ) -> list[str]:
        mismatches: list[str] = []
        if world is None or not isinstance(nearby_states, list):
            return mismatches
        try:
            existing = [
                actor
                for actor in list(world.get_actors())
                if actor is not None
                and (
                    str(getattr(actor, "type_id", "")).startswith("vehicle.")
                    or str(getattr(actor, "type_id", "")).startswith("walker.")
                )
            ]
        except Exception:
            existing = []

        for state in nearby_states:
            if not isinstance(state, dict):
                continue
            actor = self._match_or_spawn_actor(world, state, existing)
            if actor is None:
                mismatches.append(
                    "missing_actor role={} blueprint={}".format(
                        state.get("role_name", ""),
                        state.get("blueprint_id", ""),
                    )
                )
                continue
            ok = self._apply_actor_state(actor, state, apply_control=False)
            if not ok:
                mismatches.append(
                    "apply_failed role={} blueprint={}".format(
                        state.get("role_name", ""),
                        state.get("blueprint_id", ""),
                    )
                )
        return mismatches

    def _restore_traffic_lights(self, world: Any, tl_states: list[dict[str, Any]]) -> list[str]:
        notes: list[str] = []
        if world is None or carla is None or not isinstance(tl_states, list):
            return notes
        try:
            lights = list(world.get_actors().filter("*traffic_light*"))
        except Exception:
            lights = []
        for saved in tl_states:
            if not isinstance(saved, dict):
                continue
            target_loc = saved.get("transform", {}).get("location", {})
            if not isinstance(target_loc, dict):
                continue
            match = None
            best_distance = float("inf")
            for light in lights:
                try:
                    light_loc = serialize_location(light.get_transform().location) or {}
                except Exception:
                    continue
                dist = self._distance_xy(light_loc, target_loc)
                if dist < best_distance:
                    best_distance = dist
                    match = light
            if match is None or best_distance > 12.0:
                notes.append("traffic_light_match_failed")
                continue
            try:
                match.set_state(carla.TrafficLightState(int(saved.get("state", 0))))
            except Exception:
                notes.append("traffic_light_state_apply_failed")
            frozen = saved.get("frozen", None)
            if frozen is not None and hasattr(match, "freeze"):
                try:
                    match.freeze(bool(frozen))
                except Exception:
                    notes.append("traffic_light_freeze_apply_failed")
        return notes

    def restore(
        self,
        *,
        evaluator: Any,
        manager: Any,
        scenario: Any,
        checkpoint_payload: dict[str, Any],
    ) -> dict[str, Any]:
        report = {
            "partial_metric_restore": False,
            "actor_restore_mismatches": [],
            "traffic_light_restore_notes": [],
            "restored_logical_frame_id": None,
        }
        if not isinstance(checkpoint_payload, dict):
            report["partial_metric_restore"] = True
            return report

        for idx, stat_mgr in enumerate(list(getattr(evaluator, "statistics_manager", []) or [])):
            snapshot_list = checkpoint_payload.get("statistics_state", [])
            state = snapshot_list[idx] if isinstance(snapshot_list, list) and idx < len(snapshot_list) else None
            if state is None:
                report["partial_metric_restore"] = True
                continue
            if hasattr(stat_mgr, "restore_state"):
                try:
                    stat_mgr.restore_state(state)
                except Exception:
                    report["partial_metric_restore"] = True
            else:
                report["partial_metric_restore"] = True

        manager_state = checkpoint_payload.get("manager_state", {})
        if hasattr(manager, "import_checkpoint_state"):
            try:
                manager.import_checkpoint_state(manager_state)
            except Exception:
                report["partial_metric_restore"] = True
        elif hasattr(manager, "set_logical_frame_id"):
            try:
                manager.set_logical_frame_id(int(checkpoint_payload.get("logical_frame_id", 0)))
            except Exception:
                report["partial_metric_restore"] = True

        partial_criteria = self._restore_criteria_state(manager, checkpoint_payload.get("criteria_state", []))
        report["partial_metric_restore"] = bool(report["partial_metric_restore"] or partial_criteria)
        report["restored_logical_frame_id"] = int(checkpoint_payload.get("logical_frame_id", 0))

        ego_states = checkpoint_payload.get("ego_states", []) or []
        for idx, ego_state in enumerate(ego_states):
            if idx >= len(getattr(evaluator, "ego_vehicles", []) or []):
                report["actor_restore_mismatches"].append(f"ego_index_missing:{idx}")
                continue
            ego_actor = evaluator.ego_vehicles[idx]
            if ego_actor is None:
                report["actor_restore_mismatches"].append(f"ego_actor_none:{idx}")
                continue
            ok = self._apply_actor_state(ego_actor, ego_state, apply_control=True)
            if not ok:
                report["actor_restore_mismatches"].append(f"ego_apply_failed:{idx}")

        world = getattr(evaluator, "world", None) or CarlaDataProvider.get_world()
        mismatches = self._restore_nearby_actors(world, checkpoint_payload.get("nearby_actor_states", []) or [])
        report["actor_restore_mismatches"].extend(mismatches)
        tl_notes = self._restore_traffic_lights(world, checkpoint_payload.get("traffic_light_states", []) or [])
        report["traffic_light_restore_notes"].extend(tl_notes)

        agent_instance = getattr(evaluator, "agent_instance", None)
        if agent_instance is not None and hasattr(agent_instance, "sensor_interface"):
            try:
                agent_instance.sensor_interface.reset_after_recovery()
            except Exception:
                pass

        return report
