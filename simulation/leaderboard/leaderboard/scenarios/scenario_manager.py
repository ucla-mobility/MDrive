#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the ScenarioManager implementations.
It must not be modified and is for reference only!
"""

from __future__ import print_function
import copy
import json
from pathlib import Path
import queue
import signal
import sys
import time
import math
import re
from typing import Optional

import py_trees
import carla
import os
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog
from srunner.scenariomanager.scenarioatomics.atomic_criteria import RouteCompletionTest
from leaderboard.scenarios.scenarioatomics.atomic_criteria import ActorSpeedAboveThresholdTest

from leaderboard.autoagents.agent_wrapper import AgentWrapper, AgentError
from leaderboard.scenarios.tick_forensics import make_forensics
from leaderboard.envs.sensor_interface import SensorReceivedNoData
from leaderboard.utils.result_writer import ResultOutputProvider
from leaderboard.scenarios.openloop_runtime import (
    OpenLoopRuntime,
    OpenLoopIntegrityError,
)


DASHBOARD_STATUS_ENV = "RUN_CUSTOM_EVAL_STATUS_FILE"
DASHBOARD_PROGRESS_DELTA_PCT = 1.0


class ScenarioManager(object):

    """
    Basic scenario manager class. This class holds all functionality
    required to start, run and stop a scenario.

    The user must not modify this class.

    To use the ScenarioManager:
    1. Create an object via manager = ScenarioManager()
    2. Load a scenario via manager.load_scenario()
    3. Trigger the execution of the scenario manager.run_scenario()
       This function is designed to explicitly control start and end of
       the scenario execution
    4. If needed, cleanup with manager.stop_scenario()
    """


    def __init__(self, timeout, debug_mode=False):
        """
        Setups up the parameters, which will be filled at load_scenario()
        """
        self.scenario = []
        self.scenario_tree = []
        self.scenario_class = None
        self.ego_vehicles = None
        self.other_actors = None
        
        self._debug_mode = debug_mode
        self._agent = None
        self._running = False
        self._timestamp_last_run = 0.0
        self._timeout = float(timeout)

        # Used to detect if the simulation is down
        watchdog_timeout = max(5, self._timeout - 2)
        self._watchdog = Watchdog(watchdog_timeout)

        # Avoid the agent from freezing the simulation
        agent_timeout = watchdog_timeout - 1
        self._agent_watchdog = Watchdog(agent_timeout)

        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self._recovery_duration_offset_system = 0.0
        self._recovery_duration_offset_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        self.end_game_time = None

        # Register the scenario tick as callback for the CARLA world
        # Use the callback_id inside the signal handler to allow external interrupts
        signal.signal(signal.SIGINT, self.signal_handler)

        self.prev_ego_trans = None
        self.first_entry = []
        self.set_flag = True

        # Position-based stall detection (conservative fallback)
        # Tracks position history to detect when ALL vehicles are stuck.
        #
        # DEFAULT NOW DISABLED: forensics showed scenarios were repeatedly killed
        # by this 90s/0.5m heuristic during legitimately stationary phases (egos
        # waiting at red lights, blocked behind a stopped lead vehicle, queuing
        # at intersections).  False-positive rate was high.  Re-enable with
        # CUSTOM_ENABLE_STALL_DETECTION=1 if a specific debug session needs it.
        self._stall_position_history = {}  # {ego_id: [(time, x, y, z), ...]}
        self._stall_check_interval = 5.0   # Check every 5 seconds
        self._stall_last_check_time = 0.0
        self._stall_threshold_time = 600.0  # 10 min of minimal movement (was 90s)
        self._stall_min_distance = 0.5      # Must move at least 0.5m in threshold_time
        # Default: OFF.  Opt-in via CUSTOM_ENABLE_STALL_DETECTION=1.
        # CUSTOM_DISABLE_STALL_DETECTION still recognized for backward compat.
        _opt_in = self._env_flag("CUSTOM_ENABLE_STALL_DETECTION", False)
        _opt_out = self._env_flag("CUSTOM_DISABLE_STALL_DETECTION", False)
        self._stall_detection_enabled = _opt_in and not _opt_out
        if not self._stall_detection_enabled:
            print(
                "[ScenarioManager] Position-based stall detection disabled "
                "(CUSTOM_DISABLE_STALL_DETECTION=1)."
            )

        # PDM trace logging (for navsim-style metrics)
        self.pdm_traces = []
        self._pdm_prev_ang_vel = []
        self._pdm_prev_time = []
        self.pdm_world_trace = []
        self._pdm_last_world_time = None
        self.pdm_tl_polygons = {}

        # ----- Open-loop replay enforcement (single source of truth) -----
        # Originally: when CUSTOM_EGO_LOG_REPLAY=1 (set by --openloop) the
        # manager force-teleports every ego from the parsed REPLAY XML each
        # tick. We've moved to a frame-0-only metrics model: the agent dumps
        # its tick-0 perception + planner waypoints, and ADE / AP / CR are
        # computed offline from that snapshot vs the REPLAY XML. Closed-loop
        # driving runs normally for DS/RC scoring. Teleport stays disabled
        # by default; opt back in for legacy full-trajectory open-loop with
        # CUSTOM_OPENLOOP_TELEPORT=1.
        self._openloop_mode: bool = (
            self._env_flag("CUSTOM_EGO_LOG_REPLAY", False)
            and self._env_flag("CUSTOM_OPENLOOP_TELEPORT", False)
        )
        self._openloop_runtime: Optional[OpenLoopRuntime] = None
        self._openloop_terminated: bool = False
        # CARLA fixed delta_seconds (read on first tick, used to compute the
        # next-tick sim time when force-teleporting at end of current tick).
        self._openloop_dt: float = 0.05

        # record simulation time cost
        self.time_record = []
        self.c_time_record = []
        self.a_time_record = []
        self.sc_time_record = []
        self._reset_event_idx = 0
        self._scenario_name = None
        self._scenario_town = None
        self._scenario_route = None
        self._overhead_capture_enabled = False
        self._overhead_draw_boxes = True
        self._overhead_save_every_n = 1
        self._overhead_output_subdir = "overhead_tick"
        self._overhead_width = 2048
        self._overhead_height = 2048
        self._overhead_fov = 90.0
        self._overhead_margin_m = 30.0
        self._overhead_z_padding = 20.0
        self._overhead_min_camera_z = 80.0
        self._overhead_box_size_xy = 0.8
        self._overhead_box_height = 0.6
        self._overhead_box_z_offset = 0.2
        self._overhead_box_thickness = 0.08
        self._overhead_box_life_time = 600.0
        self._overhead_color_regular = carla.Color(255, 40, 40)
        self._overhead_color_corrected = carla.Color(255, 255, 0)
        self._overhead_color_sanitized = carla.Color(40, 255, 40)
        self._overhead_output_dir = None
        self._overhead_camera = None
        self._overhead_queue = None
        self._overhead_last_saved_frame = None
        self._overhead_tick_idx = 0
        self._overhead_node_count = 0
        self._dashboard_status_path = self._get_dashboard_status_path()
        self._dashboard_last_progress_scores = {}
        # Track dead-ego runtime cleanup attempts to avoid repeated destroy churn
        # while the evaluator loop is still ticking.
        self._runtime_dead_ego_cleanup_done = set()
        # Track egos that already reached route completion and were despawned.
        self._runtime_completed_ego_cleanup_done = set()
        # Tier 2 early-termination: per-ego "softly completed" tracking.
        # Maps ego_idx -> first sim_time at which (RC>=threshold AND speed below cutoff).
        # If the condition holds continuously for SOFT_COMPLETE_DWELL_S sim seconds the
        # ego is treated as "done" without waiting for AgentBlockedTest's full timeout.
        # Saves wall time on the common pattern: ego reaches near-end-of-route, stops,
        # and would otherwise block the scenario for the full blocked-timer duration.
        self._soft_complete_first_seen: dict[int, float] = {}
        self._soft_complete_done: set[int] = set()
        # Per-ego termination-path metadata, surfaced into results so we can
        # validate Tier 1/2 didn't change scores.  Values: "route_complete",
        # "blocked", "soft_complete", "stationary_timeout", or absent.
        self._termination_path: dict[int, str] = {}
        # Position-based stuck detector.  Backstops AgentBlockedTest, which
        # has speed-spike-reset bugs.  Tracks (sim_t, x, y) history per ego;
        # if cumulative distance moved over the last STATIONARY_WINDOW_S
        # sim seconds is below STATIONARY_DISTANCE_M, the ego is marked
        # terminal.  Position-based instead of speed-based, so micro-spikes
        # (slip/rebound off walls) don't reset the detection — what matters
        # is whether the ego actually went somewhere.
        self._ego_position_history: dict[int, list[tuple[float, float, float]]] = {}
        self._stationary_terminal: set[int] = set()
        self._external_tick_callback = None
        self._logical_frame_id = 0
        self._last_ego_action = None
        self._resume_from_checkpoint = False
        self._recovery_duration_offset_system = 0.0
        self._recovery_duration_offset_game = 0.0
        self._stall_debug_logging = self._env_flag("CUSTOM_STALL_DEBUG", False)
        self._enable_test_hooks = self._env_flag("CARLA_ENABLE_TEST_HOOKS", False)
        self._forced_infra_crash_frames = set()
        forced_frames_env = os.environ.get("CARLA_FORCE_INFRA_CRASH_FRAMES", "").strip()
        if forced_frames_env and not self._enable_test_hooks:
            print(
                "[ScenarioManager] Ignoring CARLA_FORCE_INFRA_CRASH_FRAMES because "
                "CARLA_ENABLE_TEST_HOOKS is not enabled."
            )
        if forced_frames_env and self._enable_test_hooks:
            for token in forced_frames_env.split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    frame_id = int(float(token))
                except Exception:
                    continue
                if frame_id > 0:
                    self._forced_infra_crash_frames.add(frame_id)

    def set_external_tick_callback(self, callback):
        """Register a per-tick callback used by checkpoint/recovery instrumentation."""
        self._external_tick_callback = callback

    def get_logical_frame_id(self):
        return int(self._logical_frame_id)

    def set_logical_frame_id(self, logical_frame_id):
        try:
            self._logical_frame_id = max(0, int(logical_frame_id))
        except Exception:
            self._logical_frame_id = 0

    def export_checkpoint_state(self):
        """
        Export manager-side runtime state required for best-effort mid-route recovery.
        """
        return {
            "logical_frame_id": int(self._logical_frame_id),
            "timestamp_last_run": float(self._timestamp_last_run),
            "scenario_duration_system": float(self.scenario_duration_system),
            "scenario_duration_game": float(self.scenario_duration_game),
            "start_system_time": self.start_system_time,
            "start_game_time": getattr(self, "start_game_time", None),
            "stall_position_history": copy.deepcopy(self._stall_position_history),
            "stall_last_check_time": float(self._stall_last_check_time),
            "runtime_dead_ego_cleanup_done": sorted(int(v) for v in self._runtime_dead_ego_cleanup_done),
            "runtime_completed_ego_cleanup_done": sorted(
                int(v) for v in self._runtime_completed_ego_cleanup_done
            ),
            "pdm_traces": copy.deepcopy(self.pdm_traces),
            "pdm_prev_ang_vel": copy.deepcopy(self._pdm_prev_ang_vel),
            "pdm_prev_time": copy.deepcopy(self._pdm_prev_time),
            "pdm_world_trace": copy.deepcopy(self.pdm_world_trace),
            "pdm_last_world_time": self._pdm_last_world_time,
            "pdm_tl_polygons": copy.deepcopy(self.pdm_tl_polygons),
            "time_record": copy.deepcopy(self.time_record),
            "c_time_record": copy.deepcopy(self.c_time_record),
            "a_time_record": copy.deepcopy(self.a_time_record),
            "sc_time_record": copy.deepcopy(self.sc_time_record),
        }

    def import_checkpoint_state(self, state):
        """
        Restore manager runtime state from a checkpoint payload.
        """
        if not isinstance(state, dict):
            return
        self._logical_frame_id = int(state.get("logical_frame_id", self._logical_frame_id) or 0)
        self._restored_timestamp_last_run = float(
            state.get("timestamp_last_run", self._timestamp_last_run) or 0.0
        )
        # CARLA episode frame/time reset after restart. Keep timestamp gate open.
        self._timestamp_last_run = -1.0
        self.scenario_duration_system = float(
            state.get("scenario_duration_system", self.scenario_duration_system) or 0.0
        )
        self.scenario_duration_game = float(
            state.get("scenario_duration_game", self.scenario_duration_game) or 0.0
        )
        self._recovery_duration_offset_system = float(self.scenario_duration_system)
        self._recovery_duration_offset_game = float(self.scenario_duration_game)
        self.start_system_time = state.get("start_system_time", self.start_system_time)
        self.start_game_time = state.get("start_game_time", getattr(self, "start_game_time", None))
        self._stall_position_history = copy.deepcopy(state.get("stall_position_history", {}))
        self._stall_last_check_time = float(
            state.get("stall_last_check_time", self._stall_last_check_time) or 0.0
        )
        self._runtime_dead_ego_cleanup_done = set(
            int(v) for v in state.get("runtime_dead_ego_cleanup_done", []) or []
        )
        self._runtime_completed_ego_cleanup_done = set(
            int(v) for v in state.get("runtime_completed_ego_cleanup_done", []) or []
        )
        self.pdm_traces = copy.deepcopy(state.get("pdm_traces", self.pdm_traces))
        self._pdm_prev_ang_vel = copy.deepcopy(state.get("pdm_prev_ang_vel", self._pdm_prev_ang_vel))
        self._pdm_prev_time = copy.deepcopy(state.get("pdm_prev_time", self._pdm_prev_time))
        self.pdm_world_trace = copy.deepcopy(state.get("pdm_world_trace", self.pdm_world_trace))
        self._pdm_last_world_time = state.get("pdm_last_world_time", self._pdm_last_world_time)
        self.pdm_tl_polygons = copy.deepcopy(state.get("pdm_tl_polygons", self.pdm_tl_polygons))
        self.time_record = copy.deepcopy(state.get("time_record", self.time_record))
        self.c_time_record = copy.deepcopy(state.get("c_time_record", self.c_time_record))
        self.a_time_record = copy.deepcopy(state.get("a_time_record", self.a_time_record))
        self.sc_time_record = copy.deepcopy(state.get("sc_time_record", self.sc_time_record))
        self._resume_from_checkpoint = True
        
    def signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        self._debug_reset(
            "signal_interrupt",
            details=f"signum={signum}",
        )
        self._running = False

    def _debug_reset(self, reason: str, details: Optional[str] = None) -> None:
        if os.environ.get("CUSTOM_LOG_REPLAY_DEBUG", "").lower() not in ("1", "true", "yes"):
            return
        self._reset_event_idx += 1
        try:
            sim_time = float(GameTime.get_time())
        except Exception:
            sim_time = None
        try:
            carla_time = float(GameTime.get_carla_time())
        except Exception:
            carla_time = None
        parts = [f"[LOG_REPLAY_DEBUG] RESET#{self._reset_event_idx} reason={reason}"]
        if self._scenario_name:
            parts.append(f"scenario={self._scenario_name}")
        if self._scenario_town:
            parts.append(f"town={self._scenario_town}")
        if self._scenario_route is not None:
            parts.append(f"route={self._scenario_route}")
        if self.repetition_number is not None:
            parts.append(f"rep={self.repetition_number}")
        if sim_time is not None:
            parts.append(f"sim_t={sim_time:.3f}")
        if carla_time is not None:
            parts.append(f"carla_t={carla_time:.3f}")
        if details:
            parts.append(details)
        print(" ".join(parts))

    @staticmethod
    def _env_flag(name, default=False):
        value = os.environ.get(name)
        if value is None:
            return bool(default)
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _env_int(name, default):
        value = os.environ.get(name)
        if value is None:
            return int(default)
        try:
            return int(float(value))
        except Exception:
            return int(default)

    @staticmethod
    def _env_float(name, default):
        value = os.environ.get(name)
        if value is None:
            return float(default)
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _parse_rgb_color(value, fallback):
        try:
            parts = [int(str(p).strip()) for p in str(value).split(",")]
            if len(parts) != 3:
                raise ValueError("expected 3 components")
            return carla.Color(
                r=max(0, min(255, parts[0])),
                g=max(0, min(255, parts[1])),
                b=max(0, min(255, parts[2])),
            )
        except Exception:
            return carla.Color(
                r=int(fallback[0]),
                g=int(fallback[1]),
                b=int(fallback[2]),
            )

    @staticmethod
    def _slugify(value):
        text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or ""))
        text = text.strip("._")
        return text or "scenario"

    @staticmethod
    def _get_dashboard_status_path():
        value = os.environ.get(DASHBOARD_STATUS_ENV, "").strip()
        if not value:
            return None
        try:
            return Path(value).expanduser()
        except Exception:
            return None

    @staticmethod
    def _load_dashboard_status_payload(status_path):
        try:
            with status_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _write_dashboard_status_payload(status_path, payload):
        try:
            status_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = status_path.with_name(".{}.tmp".format(status_path.name))
            temp_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temp_path.replace(status_path)
        except Exception:
            return

    def _collect_live_ego_route_scores(self):
        live_scores = {}
        for ego_idx, scenario_instance in enumerate(self.scenario):
            if scenario_instance is None:
                continue
            try:
                criteria = scenario_instance.get_criteria()
            except Exception:
                criteria = None
            if not criteria:
                continue
            for criterion in criteria:
                if not isinstance(criterion, RouteCompletionTest):
                    continue
                try:
                    route_score = float(
                        getattr(criterion, "_percentage_route_completed", 0.0) or 0.0
                    )
                except Exception:
                    route_score = 0.0
                live_scores[int(ego_idx)] = max(0.0, min(100.0, route_score))
                break
        return live_scores

    def _maybe_update_dashboard_progress(self):
        status_path = self._dashboard_status_path
        if status_path is None:
            return

        live_scores = self._collect_live_ego_route_scores()
        if not live_scores:
            return

        should_write = False
        if not self._dashboard_last_progress_scores:
            should_write = True
        elif set(live_scores.keys()) != set(self._dashboard_last_progress_scores.keys()):
            should_write = True
        else:
            for ego_idx, route_score in live_scores.items():
                previous_score = float(self._dashboard_last_progress_scores.get(ego_idx, 0.0))
                if abs(route_score - previous_score) >= DASHBOARD_PROGRESS_DELTA_PCT:
                    should_write = True
                    break
        if not should_write:
            return

        payload = self._load_dashboard_status_payload(status_path)
        payload["ego_route_scores"] = {
            str(ego_idx): round(route_score, 2)
            for ego_idx, route_score in sorted(live_scores.items())
        }
        if live_scores:
            payload["route_score"] = round(
                sum(live_scores.values()) / float(len(live_scores)),
                2,
            )
        payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._write_dashboard_status_payload(status_path, payload)
        self._dashboard_last_progress_scores = dict(live_scores)

    def _configure_overhead_capture_from_env(self):
        # Deprecated: per-tick overhead capture has been retired from runtime.
        # Keep this hook for CLI/env compatibility, but force-disable the feature.
        self._overhead_capture_enabled = False
        if (
            self._env_flag("CUSTOM_OVERHEAD_CAPTURE_PER_TICK", False)
            and not getattr(self, "_overhead_capture_deprecation_warned", False)
        ):
            print(
                "[ScenarioManager] CUSTOM_OVERHEAD_CAPTURE_PER_TICK is set, "
                "but overhead capture has been disabled."
            )
            self._overhead_capture_deprecation_warned = True

    def _extract_route_nodes_for_overhead(self):
        nodes = []
        route = getattr(self.scenario_class, "route", None)
        route_debug = getattr(self.scenario_class, "route_debug", None)
        if not isinstance(route, list):
            return nodes

        for ego_idx, ego_route in enumerate(route):
            if not isinstance(ego_route, list):
                continue

            debug_entry = {}
            if isinstance(route_debug, list) and ego_idx < len(route_debug) and isinstance(route_debug[ego_idx], dict):
                debug_entry = route_debug[ego_idx]
            postprocess_meta = dict(debug_entry.get("postprocess_meta", {}) or {})
            corrected_set = {
                int(idx) for idx in postprocess_meta.get("corrected_indices", [])
                if isinstance(idx, (int, float))
            }
            sanitized_set = {
                int(idx) for idx in postprocess_meta.get("sanitized_indices", [])
                if isinstance(idx, (int, float))
            }

            for point_idx, route_entry in enumerate(ego_route):
                target = route_entry[0] if isinstance(route_entry, tuple) and route_entry else route_entry
                transform = None
                if hasattr(target, "location") and hasattr(target, "rotation"):
                    transform = target
                elif hasattr(target, "transform"):
                    transform = target.transform
                if transform is None:
                    continue

                node_kind = "regular"
                if point_idx in corrected_set:
                    node_kind = "corrected"
                if point_idx in sanitized_set:
                    node_kind = "sanitized"

                nodes.append(
                    {
                        "ego_index": int(ego_idx),
                        "point_index": int(point_idx),
                        "x": float(transform.location.x),
                        "y": float(transform.location.y),
                        "z": float(transform.location.z),
                        "kind": node_kind,
                    }
                )
        return nodes

    def _build_overhead_output_dir(self):
        save_path = os.environ.get("SAVE_PATH", "").strip()
        if not save_path:
            return None
        try:
            base = Path(save_path) / str(self._overhead_output_subdir)
            route_token = self._slugify(self._scenario_route if self._scenario_route is not None else "route")
            scenario_token = self._slugify(self._scenario_name if self._scenario_name is not None else "scenario")
            rep_token = "rep{}".format(int(self.repetition_number) if self.repetition_number is not None else 0)
            out_dir = base / "{}__{}__{}".format(scenario_token, route_token, rep_token)
            out_dir.mkdir(parents=True, exist_ok=True)
            return out_dir
        except Exception as exc:
            print("[ScenarioManager] Overhead capture disabled (output dir error): {}".format(exc))
            return None

    def _draw_overhead_grp_boxes(self, world, nodes):
        if not nodes:
            return
        extent = carla.Vector3D(
            x=0.5 * float(self._overhead_box_size_xy),
            y=0.5 * float(self._overhead_box_size_xy),
            z=0.5 * float(self._overhead_box_height),
        )
        for node in nodes:
            node_kind = str(node.get("kind", "regular")).strip().lower()
            color = self._overhead_color_regular
            if node_kind == "corrected":
                color = self._overhead_color_corrected
            elif node_kind == "sanitized":
                color = self._overhead_color_sanitized

            loc = carla.Location(
                x=float(node["x"]),
                y=float(node["y"]),
                z=float(node["z"]) + float(self._overhead_box_z_offset),
            )
            bbox = carla.BoundingBox(loc, extent)
            world.debug.draw_box(
                bbox,
                carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
                thickness=float(self._overhead_box_thickness),
                color=color,
                life_time=float(self._overhead_box_life_time),
            )

    def _initialize_overhead_capture(self):
        self._teardown_overhead_capture()
        self._configure_overhead_capture_from_env()
        if not self._overhead_capture_enabled:
            return

        world = CarlaDataProvider.get_world()
        if world is None:
            print("[ScenarioManager] Overhead capture disabled (world unavailable).")
            return

        nodes = self._extract_route_nodes_for_overhead()
        if not nodes:
            print("[ScenarioManager] Overhead capture disabled (no GRP nodes available).")
            return

        out_dir = self._build_overhead_output_dir()
        if out_dir is None:
            return

        min_x = min(float(node["x"]) for node in nodes)
        max_x = max(float(node["x"]) for node in nodes)
        min_y = min(float(node["y"]) for node in nodes)
        max_y = max(float(node["y"]) for node in nodes)
        max_z = max(float(node["z"]) for node in nodes)
        center_x = 0.5 * (min_x + max_x)
        center_y = 0.5 * (min_y + max_y)
        half_extent = 0.5 * max(max_x - min_x, max_y - min_y) + float(self._overhead_margin_m)
        half_extent = max(5.0, float(half_extent))

        tan_half = math.tan(math.radians(float(self._overhead_fov)) * 0.5)
        camera_z_rel = max(float(self._overhead_min_camera_z), half_extent / max(1e-6, tan_half) + float(self._overhead_z_padding))
        camera_z = float(max_z) + camera_z_rel

        bp_library = world.get_blueprint_library()
        cam_bp = bp_library.find("sensor.camera.rgb")
        if cam_bp is None:
            print("[ScenarioManager] Overhead capture disabled (camera blueprint unavailable).")
            return
        cam_bp.set_attribute("image_size_x", str(int(self._overhead_width)))
        cam_bp.set_attribute("image_size_y", str(int(self._overhead_height)))
        cam_bp.set_attribute("fov", "{:.2f}".format(float(self._overhead_fov)))
        cam_bp.set_attribute("sensor_tick", "0.0")

        cam_tf = carla.Transform(
            carla.Location(x=float(center_x), y=float(center_y), z=float(camera_z)),
            carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
        )

        image_queue = queue.Queue(maxsize=32)
        try:
            camera = world.spawn_actor(cam_bp, cam_tf)
            camera.listen(lambda image: self._queue_overhead_image(image_queue, image))
        except Exception as exc:
            print("[ScenarioManager] Overhead capture disabled (camera spawn failed): {}".format(exc))
            return

        if self._overhead_draw_boxes:
            try:
                self._draw_overhead_grp_boxes(world, nodes)
            except Exception as exc:
                print("[ScenarioManager] Failed to draw overhead GRP boxes: {}".format(exc))

        self._overhead_queue = image_queue
        self._overhead_camera = camera
        self._overhead_output_dir = out_dir
        self._overhead_last_saved_frame = None
        self._overhead_tick_idx = 0
        self._overhead_node_count = len(nodes)

        metadata = {
            "scenario_name": self._scenario_name,
            "town": self._scenario_town,
            "route_id": self._scenario_route,
            "repetition": int(self.repetition_number) if self.repetition_number is not None else None,
            "node_count": int(self._overhead_node_count),
            "draw_boxes": bool(self._overhead_draw_boxes),
            "save_every_n": int(self._overhead_save_every_n),
            "camera": {
                "center_x": float(center_x),
                "center_y": float(center_y),
                "z": float(camera_z),
                "fov": float(self._overhead_fov),
                "width": int(self._overhead_width),
                "height": int(self._overhead_height),
            },
            "route_bounds": {
                "min_x": float(min_x),
                "max_x": float(max_x),
                "min_y": float(min_y),
                "max_y": float(max_y),
                "max_z": float(max_z),
            },
        }
        try:
            (out_dir / "capture_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        except Exception:
            pass
        print("[ScenarioManager] Overhead per-tick capture enabled: {}".format(out_dir))

    @staticmethod
    def _queue_overhead_image(image_queue, image):
        try:
            image_queue.put_nowait(image)
        except queue.Full:
            try:
                image_queue.get_nowait()
            except Exception:
                pass
            try:
                image_queue.put_nowait(image)
            except Exception:
                pass

    def _save_overhead_tick_image(self, timestamp):
        if not self._overhead_capture_enabled:
            return
        if self._overhead_queue is None or self._overhead_output_dir is None:
            return

        self._overhead_tick_idx += 1
        latest = None
        while True:
            try:
                latest = self._overhead_queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break
        if latest is None:
            return

        if self._overhead_tick_idx % max(1, int(self._overhead_save_every_n)) != 0:
            return

        frame_id = int(getattr(latest, "frame", -1))
        if frame_id >= 0 and self._overhead_last_saved_frame == frame_id:
            return

        tick_frame = int(getattr(timestamp, "frame", self._overhead_tick_idx))
        if frame_id >= 0:
            filename = "tick_{:08d}_frame_{:08d}.png".format(tick_frame, frame_id)
        else:
            filename = "tick_{:08d}.png".format(tick_frame)
        try:
            latest.save_to_disk(str(self._overhead_output_dir / filename))
            if frame_id >= 0:
                self._overhead_last_saved_frame = frame_id
        except Exception as exc:
            print("[ScenarioManager] Failed to save overhead tick image: {}".format(exc))

    def _teardown_overhead_capture(self):
        self._overhead_queue = None
        if self._overhead_camera is not None:
            try:
                self._overhead_camera.stop()
            except Exception:
                pass
            try:
                self._overhead_camera.destroy()
            except Exception:
                pass
        self._overhead_camera = None
        self._overhead_output_dir = None
        self._overhead_last_saved_frame = None
        self._overhead_tick_idx = 0
        self._overhead_node_count = 0

    def cleanup(self):
        """
        Reset all parameters
        """
        self._teardown_overhead_capture()
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        self.end_game_time = None

    def load_scenario(self, scenario, agent, rep_number, ego_vehicles_num, save_root=None, sensor_tf_list=None, is_crazy=False):
        """
        Load scenario instance and agent instance into manager
        Args:
            scenario: RouteScenario, scenario instance
            agent: agent instance
            rep_number: number of repetition
            ego_vehicles_num: number of ego vehicles
            save_root: root directory to save sensor data
        """

        self._scenario_name = getattr(getattr(scenario, "config", None), "name", None)
        self._scenario_town = getattr(getattr(scenario, "config", None), "town", None)
        self._scenario_route = getattr(getattr(scenario, "config", None), "route_id", None)
        self.repetition_number = rep_number
        self._debug_reset(
            "load_scenario",
            details=f"ego_num={ego_vehicles_num}",
        )
        GameTime.restart()
        self._logical_frame_id = 0
        self._last_ego_action = None
        forced_frames_env = os.environ.get("CARLA_FORCE_INFRA_CRASH_FRAMES", "").strip()
        if forced_frames_env and self._enable_test_hooks:
            self._forced_infra_crash_frames = set()
            for token in forced_frames_env.split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    frame_id = int(float(token))
                except Exception:
                    continue
                if frame_id > 0:
                    self._forced_infra_crash_frames.add(frame_id)

        agent.town_id = scenario.config.town # pass town id to agent
        agent.sampled_scenarios = scenario.sampled_scenarios_definitions
        agent.scenario_cofing_name = scenario.config.name

        self.is_crazy=is_crazy
        self._agent = AgentWrapper(agent)
        self.sensor_tf_list = sensor_tf_list
        self.scenario_class = scenario
        self.ego_vehicles = scenario.ego_vehicles
        self.other_actors = scenario.other_actors
        self.repetition_number = rep_number
        self.scenario=[] # important!!!!
        self.scenario_tree=[] # important!!!!

        self.ego_vehicles_num = ego_vehicles_num
        self._runtime_dead_ego_cleanup_done = set()
        self._runtime_completed_ego_cleanup_done = set()
        if self.ego_vehicles_num == 0:
            scenario_obj = getattr(scenario, "scenario", None)
            if isinstance(scenario_obj, list):
                for item in scenario_obj:
                    if item is not None:
                        self.scenario.append(item)
            elif scenario_obj is not None:
                self.scenario.append(scenario_obj)
            for scenario_item in self.scenario:
                try:
                    self.scenario_tree.append(scenario_item.scenario_tree)
                except Exception:
                    pass
        elif self.ego_vehicles_num != 1 :
            for ego_vehicle_id in range(ego_vehicles_num):
                self.scenario.append(scenario.scenario[ego_vehicle_id])
            for ego_vehicle_id in range(ego_vehicles_num):
                self.scenario_tree.append(self.scenario[ego_vehicle_id].scenario_tree)
        else:
            self.scenario.append(scenario.scenario)
            self.scenario_tree.append(self.scenario[0].scenario_tree)

        # reset PDM traces for each ego
        self.pdm_traces = [[] for _ in range(self.ego_vehicles_num)]
        self._pdm_prev_ang_vel = [None for _ in range(self.ego_vehicles_num)]
        self._pdm_prev_time = [None for _ in range(self.ego_vehicles_num)]
        self.pdm_world_trace = []
        self._pdm_last_world_time = None
        self.pdm_tl_polygons = {}

        # ----- Open-loop runtime (only when --openloop / CUSTOM_EGO_LOG_REPLAY=1) -----
        # We rebuild per scenario because each scenario has its own dense
        # REPLAY XML attached as `config.multi_replay_*` by
        # leaderboard_evaluator_parameter.py.
        self._openloop_runtime = None
        self._openloop_terminated = False
        if self._openloop_mode:
            try:
                self._openloop_runtime = OpenLoopRuntime.from_scenario_config(
                    scenario.config
                )
            except Exception as exc:
                print(f"[OPENLOOP] failed to build runtime: {exc}")
                self._openloop_runtime = None
            if self._openloop_runtime is None:
                print(
                    "[OPENLOOP] WARN: CUSTOM_EGO_LOG_REPLAY=1 set but no usable "
                    "_REPLAY.xml trajectory was found in scenario config "
                    f"({scenario.config.name}). Ego will run closed-loop — "
                    "this is the bug --openloop is supposed to prevent."
                )
            else:
                stats = self._openloop_runtime.stats()
                print(
                    f"[OPENLOOP] runtime ready: "
                    f"egos_with_plan={stats['n_egos_with_plan']}/{self.ego_vehicles_num}"
                )
            try:
                world = CarlaDataProvider.get_world()
                if world is not None:
                    settings = world.get_settings()
                    if getattr(settings, "fixed_delta_seconds", None):
                        self._openloop_dt = float(settings.fixed_delta_seconds)
            except Exception:
                pass

        # To print the scenario tree uncomment the next line
        # py_trees.display.render_dot_tree(self.scenario_tree)

        for vehicle_num in range(self.ego_vehicles_num):
            print("set ip sensor for ego vehicle {}".format(vehicle_num))
            self._agent.setup_sensors(self.ego_vehicles[vehicle_num], vehicle_num, save_root, self._debug_mode)
            self.first_entry.append(True)

        self._initialize_overhead_capture()

    def run_scenario(self):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        if self._resume_from_checkpoint:
            self._resume_from_checkpoint = False
            self.start_system_time = time.time()
            self.start_game_time = GameTime.get_time()
        else:
            self.start_system_time = time.time()
            self.start_game_time = GameTime.get_time()

        self._watchdog.start()
        self._running = True

        # ------------------------------------------------------------------
        # Tick forensics: comprehensive instrumentation for tick rate,
        # phase timing, RPC liveness, and stall detection.  Off unless
        # CARLA_TICK_FORENSICS=1 is set.  See tick_forensics.py for full
        # documentation of emitted events.
        # ------------------------------------------------------------------
        try:
            self._forensics = make_forensics(CarlaDataProvider.get_world())
            self._forensics.start()
        except Exception as _e:
            print(f"[TICK_FORENSICS] INIT_FAILED err={type(_e).__name__}: {_e}", flush=True)
            self._forensics = make_forensics(None)  # falls back to no-op

        # ------------------------------------------------------------------
        # Sensor pre-warm: tick the world a few times BEFORE entering the
        # main agent() loop so all CARLA sensor streaming sessions get a
        # chance to establish their first packet delivery.
        #
        # Why this is needed
        # ------------------
        # CARLA's streaming layer opens a TCP session per sensor lazily.
        # For multi-ego scenarios, late-spawned IMU/GPS/LiDAR sensors lose
        # the race against the setup_sensors world.tick() and don't deliver
        # their first packet during setup.  Once run_scenario starts and
        # agent()->get_data() blocks waiting for those sensors, the world
        # cannot tick (tick is at the END of _tick_scenario, after agent()).
        # That deadlocks until the leaderboard timeout fires.
        #
        # Pre-warming with explicit ticks here -- BEFORE any agent() call --
        # gives the streaming server time to deliver first packets while
        # the world can still tick freely.
        #
        # Configurable via env (default 10 ticks ~= 0.5s sim time, ~1s wall):
        #   CARLA_SENSOR_WARMUP_TICKS     int, default 10
        #   CARLA_SENSOR_WARMUP_DISABLE   set to 1 to skip entirely
        # ------------------------------------------------------------------
        if not os.environ.get("CARLA_SENSOR_WARMUP_DISABLE", "").lower() in ("1", "true", "yes"):
            try:
                _warmup_n = max(0, int(os.environ.get("CARLA_SENSOR_WARMUP_TICKS", "10")))
            except (TypeError, ValueError):
                _warmup_n = 10
            if _warmup_n > 0:
                _world = CarlaDataProvider.get_world()
                if _world is not None:
                    _warmup_t0 = time.time()
                    _warmup_completed = 0
                    _warmup_failed_at = None
                    for _i in range(_warmup_n):
                        try:
                            _world.tick(self._timeout)
                            _warmup_completed += 1
                        except Exception as _e:
                            _warmup_failed_at = (_i, str(_e))
                            break
                    print(
                        f"[SENSOR_WARMUP] completed={_warmup_completed}/{_warmup_n} "
                        f"wall_s={time.time() - _warmup_t0:.2f} "
                        f"failed_at={_warmup_failed_at}",
                        flush=True,
                    )

        while self._running:
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                if self.is_crazy:
                    # turn off traffic light at every frame to prevent from turning on in accident
                    [tf.set_state(carla.libcarla.TrafficLightState.Green) for tf in world.get_actors().filter("*traffic_light*") if hasattr(tf,"set_state")]
                    [tf.freeze(True) for tf in world.get_actors().filter("*traffic_light*") if hasattr(tf,"freeze")]
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if timestamp:
                self._tick_scenario(timestamp)

    def _tick_scenario(self, timestamp):
        """
        Run next tick of scenario and the agent and tick the world.
        """

        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds
            _tick_t0 = time.time()
            # Forensics: begin tick + first phase. self._forensics is a no-op
            # stub when CARLA_TICK_FORENSICS is unset, so this is free.
            try:
                self._forensics.begin_tick(self._logical_frame_id)
                self._forensics.begin_phase("world_state_setup")
            except Exception:
                pass
            self._watchdog.update()
            # Heartbeat for external freeze watchdog (see run_carla_rootcause_capture.sh)
            _hb_path = os.environ.get("CARLA_TICK_HEARTBEAT_FILE", "")
            if _hb_path:
                _now = time.time()
                if not hasattr(self, "_hb_last") or _now - self._hb_last >= 1.0:
                    self._hb_last = _now
                    try:
                        Path(_hb_path).touch()
                    except OSError:
                        pass
            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()
            self._save_overhead_tick_image(timestamp)
            # ----- Open-loop correctness check (top of tick, post-physics) -----
            # Verify that the actor pose CARLA just integrated matches what
            # the REPLAY XML says it should be at this sim_t. Catches any
            # leak (planner control that snuck through, broken set_transform,
            # behavior-tree ordering bug) within ONE tick instead of letting
            # the trajectory silently drift. Tunable thresholds:
            #   OPENLOOP_POSE_WARN_M   (default 0.05 m)
            #   OPENLOOP_POSE_ABORT_M  (default 1.00 m)  -> raises
            if self._openloop_mode and self._openloop_runtime is not None:
                try:
                    self._openloop_runtime.check_pose(
                        self.ego_vehicles, float(timestamp.elapsed_seconds)
                    )
                except OpenLoopIntegrityError as exc:
                    print(f"[OPENLOOP] FATAL: {exc}")
                    self._openloop_terminated = True
                    self._running = False
                    raise
            _tick_t1 = time.time()  # end of world-tick phase (GameTime + CarlaDataProvider + overhead)

            if os.environ.get("CUSTOM_LOG_REPLAY_DEBUG", "").lower() in ("1", "true", "yes"):
                try:
                    interval = float(os.environ.get("CUSTOM_LOG_REPLAY_DEBUG_INTERVAL", "2.0"))
                except Exception:
                    interval = 2.0
                if not hasattr(self, "_log_replay_debug_last"):
                    self._log_replay_debug_last = -1.0
                    self._log_replay_prev_time = None
                    self._log_replay_prev_locs = {}
                if (
                    self._log_replay_prev_time is not None
                    and timestamp.elapsed_seconds + 1e-3 < self._log_replay_prev_time
                ):
                    print(
                        f"[LOG_REPLAY_DEBUG] scenario time went backwards "
                        f"({self._log_replay_prev_time:.3f} -> {timestamp.elapsed_seconds:.3f})"
                    )
                if (
                    self._log_replay_debug_last < 0
                    or timestamp.elapsed_seconds - self._log_replay_debug_last >= interval
                ):
                    for vehicle_num in range(self.ego_vehicles_num):
                        ego = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)
                        if ego is None:
                            print(f"[LOG_REPLAY_DEBUG] ego{vehicle_num}: missing")
                            continue
                        try:
                            loc = ego.get_location()
                            vel = ego.get_velocity()
                            speed = math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)
                        except Exception:
                            loc = None
                            speed = 0.0
                        prev = self._log_replay_prev_locs.get(vehicle_num)
                        jump = None
                        if loc is not None and prev is not None:
                            try:
                                jump = loc.distance(prev)
                            except Exception:
                                jump = None
                        status = None
                        try:
                            status = self.scenario_tree[vehicle_num].status
                        except Exception:
                            status = None
                        if loc is None:
                            loc_str = "n/a"
                        else:
                            loc_str = f"({loc.x:.2f},{loc.y:.2f},{loc.z:.2f})"
                        print(
                            f"[LOG_REPLAY_DEBUG] t={timestamp.elapsed_seconds:.3f} ego{vehicle_num} "
                            f"loc={loc_str} spd={speed:.2f} "
                            f"jump={jump if jump is not None else 'n/a'} status={status}"
                        )
                        if loc is not None:
                            self._log_replay_prev_locs[vehicle_num] = loc
                    if self._openloop_mode:
                        for idx in range(self.ego_vehicles_num):
                            try:
                                done_key = f"log_replay_done_ego_{idx}"
                                replay_done = bool(py_trees.blackboard.Blackboard().get(done_key))
                            except Exception:
                                replay_done = False
                            print(f"[LOG_REPLAY_DEBUG] ego{idx} replay_done={replay_done}")
                    self._log_replay_debug_last = timestamp.elapsed_seconds
                self._log_replay_prev_time = timestamp.elapsed_seconds

            # destroy ego if it is not alive
            for vehicle_num in range(self.ego_vehicles_num):
                if vehicle_num in self._runtime_completed_ego_cleanup_done:
                    continue
                ego_actor = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)
                if ego_actor and ego_actor.is_alive:
                    self._runtime_dead_ego_cleanup_done.discard(vehicle_num)
                if ego_actor and not ego_actor.is_alive:
                    if vehicle_num in self._runtime_dead_ego_cleanup_done:
                        continue
                    self._agent.cleanup_single(vehicle_num)
                    self._agent.cleanup_rsu(vehicle_num)
                    print("destroy ego type 0 : {}".format(vehicle_num))
                    ego_id = getattr(ego_actor, "id", None)
                    if ego_id is not None:
                        CarlaDataProvider.remove_actor_by_id(
                            int(ego_id),
                            max_retries=0,
                            timeout_s=0.5,
                            poll_s=0.02,
                            direct_fallback=False,
                            reason="scenario_manager_runtime_ego_cleanup",
                            phase="destroy_ego_type_0",
                        )
                    self._runtime_dead_ego_cleanup_done.add(vehicle_num)

            # Agent take action (eg. save data/produce control signal)
            try:
                self._forensics.end_phase("world_state_setup")
                self._forensics.begin_phase("agent_call")
            except Exception:
                pass
            try:
                ego_action = self._agent()
                self._last_ego_action = ego_action

            # Special exception inside the agent that isn't caused by the agent
            except SensorReceivedNoData as e:
                raise RuntimeError(e)

            except Exception as e:
                raise AgentError(e)
            try:
                self._forensics.end_phase("agent_call")
                self._forensics.begin_phase("apply_control_and_scenario")
            except Exception:
                pass

            _tick_t2 = time.time()  # end of agent call phase

            # destroy ego if it is not alive
            for vehicle_num in range(self.ego_vehicles_num):
                if vehicle_num in self._runtime_completed_ego_cleanup_done:
                    continue
                ego_actor = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)
                if ego_actor and ego_actor.is_alive:
                    self._runtime_dead_ego_cleanup_done.discard(vehicle_num)
                if ego_actor and not ego_actor.is_alive:
                    if vehicle_num in self._runtime_dead_ego_cleanup_done:
                        continue
                    self._agent.cleanup_single(vehicle_num)
                    self._agent.cleanup_rsu(vehicle_num)
                    print("destroy ego type 1 : {}".format(vehicle_num))
                    ego_id = getattr(ego_actor, "id", None)
                    if ego_id is not None:
                        CarlaDataProvider.remove_actor_by_id(
                            int(ego_id),
                            max_retries=0,
                            timeout_s=0.5,
                            poll_s=0.02,
                            direct_fallback=False,
                            reason="scenario_manager_runtime_ego_cleanup",
                            phase="destroy_ego_type_1",
                        )
                    self._runtime_dead_ego_cleanup_done.add(vehicle_num)

            # ----- Open-loop control suppression -----
            # In --openloop the planner's (steer, throttle, brake) MUST NOT
            # reach the actuator. We replace each ego_action with a neutral
            # control (0/0/0) when an OpenLoopRuntime plan exists for the
            # ego. The agent's run_step still ran above, so per-tick
            # predictions / *.json / *_pred.npz are still emitted.
            if self._openloop_mode and self._openloop_runtime is not None:
                neutral = carla.VehicleControl()
                neutral.steer = 0.0
                neutral.throttle = 0.0
                neutral.brake = 0.0
                neutral.hand_brake = False
                neutral.manual_gear_shift = False
                for vehicle_num in range(self.ego_vehicles_num):
                    if self._openloop_runtime.has_plan_for(vehicle_num):
                        ego_action[vehicle_num] = neutral

            # Execute driving control signal
            for vehicle_num in range(self.ego_vehicles_num):
                try:
                    if vehicle_num in self._runtime_completed_ego_cleanup_done:
                        continue
                    ego = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)
                    if ego:
                        if ego.is_alive:
                            if os.environ.get('DEBUG_SCENARIOMGR', '').lower() in ('1', 'true', 'yes'):
                                print(f"[DEBUG SCENARIO_MGR] Applying control to ego {vehicle_num}: throttle={ego_action[vehicle_num].throttle:.3f}, brake={ego_action[vehicle_num].brake:.3f}, steer={ego_action[vehicle_num].steer:.3f}")
                            self.ego_vehicles[vehicle_num].apply_control(ego_action[vehicle_num])
                            # record trace for PDM metrics
                            self._record_pdm_sample(vehicle_num, ego, timestamp)
                except:
                    pass

            _tick_t3 = time.time()  # end of apply_control phase
            # Tick scenario
            for vehicle_num in range(self.ego_vehicles_num):
                try:
                    if vehicle_num in self._runtime_completed_ego_cleanup_done:
                        continue
                    ego = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)
                    if ego and ego.is_alive:
                        self.scenario_tree[vehicle_num].tick_once()
                except:
                    pass
            if self.ego_vehicles_num == 0 and self.scenario_tree:
                try:
                    self.scenario_tree[0].tick_once()
                except Exception:
                    pass

            # ----- Open-loop force-teleport (end of tick) -----
            # AFTER apply_control + scenario_tree.tick_once: pin every ego to
            # its replay pose for the NEXT physics frame. This makes the
            # planner's controls (already queued via apply_control above) a
            # no-op because we overwrite the pose and zero the velocity that
            # the next world.tick() would integrate against. With the control
            # also forced to neutral upstream, this is belt-and-braces.
            #
            # We target sim_t_next = current sim_t + dt because that is the
            # pose the actor will be observed at by the agent on the next
            # tick. (The next tick's pose check verifies the actor actually
            # ended up there.)
            if self._openloop_mode and self._openloop_runtime is not None:
                try:
                    sim_t_next = float(timestamp.elapsed_seconds) + self._openloop_dt
                    self._openloop_runtime.ensure_pose(self.ego_vehicles, sim_t_next)
                    if self._openloop_runtime.all_exhausted(sim_t_next):
                        if not self._openloop_runtime.consume_tail_tick():
                            print(
                                f"[OPENLOOP] replay exhausted at sim_t_next="
                                f"{sim_t_next:.3f}s (all egos past end of REPLAY.xml). "
                                f"Terminating cleanly."
                            )
                            self._openloop_terminated = True
                            self._running = False
                            for ego_idx in range(self.ego_vehicles_num):
                                self._termination_path[ego_idx] = "openloop_replay_complete"
                except OpenLoopIntegrityError as exc:
                    # Bubble out of the tick loop. The launcher catches this
                    # and writes _run_status.json with failure_kind so the
                    # regression is loudly visible instead of silent drift.
                    print(f"[OPENLOOP] FATAL: {exc}")
                    self._openloop_terminated = True
                    self._running = False
                    raise
                except Exception as exc:
                    print(f"[OPENLOOP] non-fatal teleport error: {exc}")

            _tick_t4 = time.time()  # end of scenario_tree phase
            # Accumulate per-tick timing
            self.c_time_record.append(_tick_t1 - _tick_t0)   # world tick (GameTime + CarlaDataProvider)
            self.a_time_record.append(_tick_t2 - _tick_t1)   # agent call (sensor read + inference)
            self.time_record.append(_tick_t3 - _tick_t2)     # apply_control + PDM sample
            self.sc_time_record.append(_tick_t4 - _tick_t3)  # scenario_tree tick_once

            # ----- Early-exit hook (frame-0 openloop debug) -----
            # CUSTOM_EARLY_EXIT_TICKS=N terminates the scenario cleanly after
            # N ticks. Use case: --openloop runs only need tick 0 for frame-0
            # metrics; the rest of the closed-loop drive is wasted compute.
            # Clean termination = scenario-pool sees success returncode=0 =
            # NO retry. The openloop post-process script still fires after
            # scenario end and processes whatever pred files were written.
            _early_n = int(os.environ.get("CUSTOM_EARLY_EXIT_TICKS", "0") or "0")
            if _early_n > 0 and len(self.a_time_record) >= _early_n:
                print(f"[EARLY-EXIT] ticks={len(self.a_time_record)} reached "
                      f"CUSTOM_EARLY_EXIT_TICKS={_early_n}; terminating cleanly.")
                self._running = False
                for ego_idx in range(self.ego_vehicles_num):
                    self._termination_path.setdefault(ego_idx, "early_exit_first_frame")

            self._maybe_update_dashboard_progress()

            if self._debug_mode:
                print("\n")
                for vehicle_num in range(self.ego_vehicles_num):
                    # if self.scenario_tree[vehicle_num].status == py_trees.common.Status.RUNNING \
                    #    or self.scenario_tree[vehicle_num].status == py_trees.common.Status.INVALID:
                    try:
                        ego = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)
                        if ego and ego.is_alive:
                        # if CarlaDataProvider.get_hero_actor(hero_id=vehicle_num).is_alive:
                            py_trees.display.print_ascii_tree(
                                self.scenario_tree[vehicle_num], show_status=True)
                            sys.stdout.flush()
                    except:
                        pass
                if self.ego_vehicles_num == 0 and self.scenario_tree:
                    try:
                        py_trees.display.print_ascii_tree(self.scenario_tree[0], show_status=True)
                        sys.stdout.flush()
                    except Exception:
                        pass

            # destroy ego if it is not in RUNNING status or not alive
            stop_flag = 0
            # Use the unified openloop gate (both CUSTOM_EGO_LOG_REPLAY=1 AND
            # CUSTOM_OPENLOOP_TELEPORT=1). The legacy single-env check broke
            # the new "frame-0 metrics + closed-loop drive" pattern: with
            # CUSTOM_EGO_LOG_REPLAY=1 alone, this code waited forever for
            # the LogReplayFollower's blackboard key, but no LogReplayFollower
            # is created in that mode, so even after timeout/criteria fired,
            # `continue` at L~1389 prevented stop_flag from incrementing
            # and the scenario never ended.
            log_replay_ego = self._openloop_mode
            missing_egos = []
            nonrunning_egos = []
            dead_egos = []
            for vehicle_num in range(self.ego_vehicles_num):
                ego_actor = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)
                if ego_actor is None:
                    missing_egos.append(vehicle_num)
                    stop_flag += 1
                    if stop_flag == self.ego_vehicles_num:
                        self._debug_reset(
                            "all_egos_missing",
                            details=f"missing={missing_egos}",
                        )
                        self._running = False
                
                else:
                    if ego_actor.is_alive:
                        self._runtime_dead_ego_cleanup_done.discard(vehicle_num)
                    status_not_running = (
                        self.scenario_tree[vehicle_num].status != py_trees.common.Status.RUNNING
                    )
                    ego_dead = not ego_actor.is_alive
                    if not (status_not_running or ego_dead):
                        continue
                    nonrunning_egos.append(vehicle_num)
                    if ego_dead:
                        dead_egos.append(vehicle_num)
                    if log_replay_ego:
                        try:
                            done_key = f"log_replay_done_ego_{vehicle_num}"
                            replay_done = bool(py_trees.blackboard.Blackboard().get(done_key))
                        except Exception:
                            replay_done = False
                        # In log replay mode, keep ego alive until replay has reached the end.
                        if not replay_done and ego_actor.is_alive:
                            continue
                    stop_flag += 1
                    if ego_dead and vehicle_num not in self._runtime_dead_ego_cleanup_done:
                        self._agent.cleanup_single(vehicle_num)
                        self._agent.cleanup_rsu(vehicle_num)
                        print("destroy ego type 3 {}".format(vehicle_num))
                        print('flag1:', status_not_running)
                        print('flag2:', ego_dead)
                        ego_id = getattr(ego_actor, "id", None)
                        if ego_id is not None:
                            CarlaDataProvider.remove_actor_by_id(
                                int(ego_id),
                                max_retries=0,
                                timeout_s=0.5,
                                poll_s=0.02,
                                direct_fallback=False,
                                reason="scenario_manager_runtime_ego_cleanup",
                                phase="destroy_ego_type_3",
                            )
                        self._runtime_dead_ego_cleanup_done.add(vehicle_num)
                    if stop_flag == self.ego_vehicles_num:
                        status_list = []
                        for idx in range(self.ego_vehicles_num):
                            try:
                                status_list.append(str(self.scenario_tree[idx].status))
                            except Exception:
                                status_list.append("unknown")
                        self._debug_reset(
                            "all_egos_nonrunning_or_dead",
                            details=f"nonrunning={nonrunning_egos} dead={dead_egos} statuses={status_list}",
                        )
                        try:
                            _now_sim_outer = float(GameTime.get_time())
                        except Exception:
                            _now_sim_outer = -1.0
                        print(
                            f"[SCENARIO_TERMINATED] sim_t={_now_sim_outer:.1f}s "
                            f"trigger=outer_all_nonrunning_or_dead "
                            f"n_egos={self.ego_vehicles_num} "
                            f"missing={missing_egos} dead={dead_egos} nonrunning={nonrunning_egos} "
                            f"tree_statuses={status_list}",
                            flush=True,
                        )
                        self._running = False

            # set spectator
            spectator = CarlaDataProvider.get_world().get_spectator()
            if self.ego_vehicles_num > 0:
                if CarlaDataProvider.get_hero_actor(hero_id=0):
                    ego_trans = CarlaDataProvider.get_hero_actor(hero_id=0).get_transform()
                    self.prev_ego_trans = ego_trans
                else:
                    for vehicle_num in range(1, self.ego_vehicles_num):
                        if CarlaDataProvider.get_hero_actor(hero_id=vehicle_num):
                            ego_trans = self.ego_vehicles[vehicle_num].get_transform()
                            self.prev_ego_trans = ego_trans
                            break
                    # if none of the ego vehicle is alive
                    ego_trans = self.prev_ego_trans
                if ego_trans is not None:
                    spectator.set_transform(
                        carla.Transform(
                            ego_trans.location + carla.Location(z=50),
                            carla.Rotation(pitch=-90),
                        )
                    )
            else:
                target_tf = None
                world = CarlaDataProvider.get_world()
                fake_names = [
                    n.strip()
                    for n in os.environ.get("CUSTOM_FAKE_EGO_CAMERA_NAMES", "").split(",")
                    if n.strip()
                ]
                try:
                    vehicle_actors = list(world.get_actors().filter("vehicle.*")) if world else []
                except Exception:
                    vehicle_actors = []
                if fake_names and vehicle_actors:
                    for fake_name in fake_names:
                        for actor in vehicle_actors:
                            try:
                                if actor.attributes.get("role_name", "") == fake_name:
                                    target_tf = actor.get_transform()
                                    break
                            except Exception:
                                continue
                        if target_tf is not None:
                            break
                if target_tf is None and vehicle_actors:
                    try:
                        target_tf = vehicle_actors[0].get_transform()
                    except Exception:
                        target_tf = None
                if target_tf is not None:
                    spectator.set_transform(
                        carla.Transform(
                            target_tf.location + carla.Location(z=50),
                            carla.Rotation(pitch=-90),
                        )
                    )

            # terminate route scenarios once all egos are either completed OR blocked.
            # Same unified gate as the destroy block above.
            log_replay_ego = self._openloop_mode
            all_egos_done = True
            done_details = None
            if self.ego_vehicles_num == 0:
                fake_names = [
                    n.strip()
                    for n in os.environ.get("CUSTOM_FAKE_EGO_CAMERA_NAMES", "").split(",")
                    if n.strip()
                ]
                fake_done_flags = []
                fake_done_known_count = 0
                for fake_name in fake_names:
                    key_name = re.sub(r"[^A-Za-z0-9_]+", "_", str(fake_name)).strip("_")
                    done_key = f"log_replay_done_actor_{key_name}"
                    try:
                        done_val = py_trees.blackboard.Blackboard().get(done_key)
                    except Exception:
                        done_val = None
                    if done_val is not None:
                        fake_done_known_count += 1
                    fake_done_flags.append(bool(done_val))

                # Debug: Log fake_ego done status periodically
                if not hasattr(self, '_last_fake_ego_debug'):
                    self._last_fake_ego_debug = 0
                if time.time() - self._last_fake_ego_debug > 5:
                    print(f"[ScenarioManager] fake_ego done check: names={fake_names}, "
                          f"known={fake_done_known_count}/{len(fake_names)}, flags={fake_done_flags}")
                    self._last_fake_ego_debug = time.time()

                if fake_names and fake_done_known_count == len(fake_names):
                    all_egos_done = all(fake_done_flags)
                    done_details = f"fake_ego_replay_done={dict(zip(fake_names, fake_done_flags))}"
                elif self.scenario_tree:
                    try:
                        tree_status = self.scenario_tree[0].status
                    except Exception:
                        tree_status = py_trees.common.Status.RUNNING
                    all_egos_done = tree_status != py_trees.common.Status.RUNNING
                    done_details = f"scenario_tree_status={tree_status}"
                else:
                    all_egos_done = True
                    done_details = "scenario_tree=empty"
            elif log_replay_ego:
                for idx in range(self.ego_vehicles_num):
                    try:
                        done_key = f"log_replay_done_ego_{idx}"
                        replay_done = bool(py_trees.blackboard.Blackboard().get(done_key))
                    except Exception:
                        replay_done = False
                    if not replay_done:
                        all_egos_done = False
                        break
            else:
                for idx, scenario_instance in enumerate(self.scenario):
                    if idx in self._runtime_completed_ego_cleanup_done:
                        continue
                    if scenario_instance is None:
                        continue
                    # If the ego actor itself is gone or dead (despawned by
                    # the runtime cleanup, or destroyed by a hard collision),
                    # treat the ego as terminal so the inner all_egos_done
                    # rollup doesn't get blocked waiting on a vehicle that
                    # no longer exists.  Without this, scenarios with one
                    # despawned-early ego + N stuck egos kept ticking forever
                    # because the despawned ego's criteria never satisfied
                    # any of (route_complete, blocked, soft_complete).
                    try:
                        _ego_actor_check = CarlaDataProvider.get_hero_actor(hero_id=idx)
                    except Exception:
                        _ego_actor_check = None
                    if _ego_actor_check is None or not _ego_actor_check.is_alive:
                        was_first = idx not in self._termination_path
                        self._termination_path.setdefault(idx, "ego_despawned_or_dead")
                        if was_first:
                            try:
                                _now_sim = float(GameTime.get_time())
                            except Exception:
                                _now_sim = -1.0
                            print(
                                f"[TERM_DECISION] ego={idx} sim_t={_now_sim:.1f}s "
                                f"path=ego_despawned_or_dead "
                                f"actor_is_none={_ego_actor_check is None} "
                                f"is_alive={getattr(_ego_actor_check, 'is_alive', 'n/a')}",
                                flush=True,
                            )
                        # Don't gate all_egos_done on this ego — it's already gone.
                        continue
                    criteria = scenario_instance.get_criteria()
                    if not criteria:
                        # Empty criteria means this ego's scenario_instance is a
                        # no-test placeholder. In multi-ego scenarios the criteria
                        # (RouteCompletionTest, ActorSpeedAboveThresholdTest, etc.)
                        # attach only to the primary ego's instance; instances 1..N
                        # for additional egos legitimately have no criteria.
                        # Previous behaviour was to set all_egos_done=False and
                        # break, which made the scenario never terminate via the
                        # all_egos_done path: the loop saw ego_0 as terminal
                        # (e.g. stationary), then bailed at ego_1 with empty
                        # criteria, leaving the run ticking forever even when all
                        # egos were physically parked. Symmetrise with the
                        # despawned-ego handling above: treat empty-criteria as
                        # "nothing to gate on" and continue.
                        if os.environ.get('DEBUG_FINISH', '').lower() in ('1', 'true', 'yes'):
                            print(f"[DEBUG FINISH] Ego {idx}: No criteria found -- skipping (not blocking all_egos_done)")
                        was_first = idx not in self._termination_path
                        self._termination_path.setdefault(idx, "no_criteria_placeholder")
                        if was_first:
                            try:
                                _now_sim = float(GameTime.get_time())
                            except Exception:
                                _now_sim = -1.0
                            print(
                                f"[TERM_DECISION] ego={idx} sim_t={_now_sim:.1f}s "
                                f"path=no_criteria_placeholder "
                                f"reason=multi-ego_secondary_with_empty_criteria",
                                flush=True,
                            )
                        continue
                    # Check if this ego is either completed (SUCCESS) or blocked (FAILURE)
                    ego_completed = False
                    ego_blocked = False
                    ego_route_completion_100 = False
                    ego_deviated = False    # InRouteTest FAILURE — ego went off-route
                    ego_collided_terminal = False  # CollisionTest with terminate_on_failure
                    # Debug: Show criterion types (every 10 seconds)
                    if not hasattr(self, '_last_debug_time'):
                        self._last_debug_time = 0
                    if time.time() - self._last_debug_time > 10:
                        if os.environ.get('DEBUG_FINISH', '').lower() in ('1', 'true', 'yes'):
                            print(f"[DEBUG FINISH] Ego {idx}: Found {len(criteria)} criteria: {[type(c).__name__ for c in criteria]}")
                            for c in criteria:
                                print(f"[DEBUG FINISH]   - {type(c).__name__}: status={getattr(c, 'test_status', 'N/A')}")
                        self._last_debug_time = time.time()
                    route_completion_pct_for_ego = 0.0
                    for criterion in criteria:
                        if isinstance(criterion, RouteCompletionTest):
                            try:
                                route_completion_pct = float(
                                    getattr(criterion, "_percentage_route_completed", 0.0) or 0.0
                                )
                            except Exception:
                                route_completion_pct = 0.0
                            route_completion_pct_for_ego = route_completion_pct
                            if route_completion_pct >= 100.0 - 1e-3:
                                # Despawn immediately once route completion reaches 100%,
                                # even if RouteCompletionTest has not switched to SUCCESS yet
                                # (SUCCESS also depends on distance-to-target < threshold).
                                ego_route_completion_100 = True
                            if criterion.test_status == "SUCCESS":
                                ego_completed = True
                                if os.environ.get('DEBUG_FINISH', '').lower() in ('1', 'true', 'yes'):
                                    print(f"[DEBUG FINISH] Ego {idx}: RouteCompletionTest SUCCESS")
                        elif isinstance(criterion, ActorSpeedAboveThresholdTest):
                            if criterion.test_status == "FAILURE":
                                ego_blocked = True
                                if os.environ.get('DEBUG_FINISH', '').lower() in ('1', 'true', 'yes'):
                                    print(f"[DEBUG FINISH] Ego {idx}: AgentBlockedTest FAILURE")
                        else:
                            # Catch other terminal-on-failure criteria that
                            # py_trees might not propagate up to scenario_tree.
                            # InRouteTest fires FAILURE when ego goes >30m off
                            # route (terminate_on_failure=True).  CollisionTest
                            # may fire on hard impact.  Both should mark the
                            # ego terminal even if scenario_tree.status didn't
                            # change.  Match by class name (avoid extra imports).
                            cls_name = type(criterion).__name__
                            ts = getattr(criterion, "test_status", None)
                            if cls_name == "InRouteTest" and ts == "FAILURE":
                                ego_deviated = True
                            elif cls_name == "CollisionTest" and ts == "FAILURE":
                                ego_collided_terminal = True

                    # Tier 2 early-termination check: ego is at high RC AND has
                    # been stopped for >SOFT_COMPLETE_DWELL_S sim seconds.  This
                    # is the "near-goal stop" pattern — the agent has essentially
                    # reached the destination but didn't quite tick over to 100%.
                    # We treat it as done without waiting for AgentBlockedTest's
                    # full timer.  Disabled when CARLA_TIER2_SOFT_COMPLETE_ENABLED=0.
                    ego_soft_complete = False
                    if (
                        not ego_completed
                        and not ego_route_completion_100
                        and not ego_blocked
                        and os.environ.get("CARLA_TIER2_SOFT_COMPLETE_ENABLED", "1").lower() not in ("0", "false", "no")
                    ):
                        try:
                            _rc_threshold = float(os.environ.get("CARLA_TIER2_SOFT_COMPLETE_RC_PCT", "95.0"))
                            _dwell_s = float(os.environ.get("CARLA_TIER2_SOFT_COMPLETE_DWELL_S", "5.0"))
                            _speed_cutoff = float(os.environ.get("CARLA_TIER2_SOFT_COMPLETE_SPEED_MPS", "0.1"))
                        except Exception:
                            _rc_threshold, _dwell_s, _speed_cutoff = 95.0, 5.0, 0.1
                        if route_completion_pct_for_ego >= _rc_threshold:
                            try:
                                _ego_actor = CarlaDataProvider.get_hero_actor(hero_id=idx)
                                _ego_speed = CarlaDataProvider.get_velocity(_ego_actor) if _ego_actor else None
                            except Exception:
                                _ego_speed = None
                            try:
                                _now_sim = float(GameTime.get_time())
                            except Exception:
                                _now_sim = None
                            if _ego_speed is not None and _now_sim is not None:
                                if _ego_speed < _speed_cutoff:
                                    if idx not in self._soft_complete_first_seen:
                                        self._soft_complete_first_seen[idx] = _now_sim
                                    elif (_now_sim - self._soft_complete_first_seen[idx]) >= _dwell_s:
                                        ego_soft_complete = True
                                        self._soft_complete_done.add(idx)
                                        self._termination_path.setdefault(idx, "soft_complete")
                                        try:
                                            _stop_dur = _now_sim - self._soft_complete_first_seen[idx]
                                            print(
                                                f"[TERM_DECISION] ego={idx} sim_t={_now_sim:.1f}s "
                                                f"path=soft_complete (Tier 2) "
                                                f"rc={route_completion_pct_for_ego:.1f}% "
                                                f"stopped_for={_stop_dur:.1f}s speed={_ego_speed:.3f}m/s "
                                                f"(saved up to ~{30.0 - _stop_dur:.1f}s of blocked-timer wait)",
                                                flush=True,
                                            )
                                        except Exception:
                                            pass
                                else:
                                    # Speed went above cutoff again — reset the
                                    # dwell timer so we only fire on continuous stop.
                                    self._soft_complete_first_seen.pop(idx, None)

                    # Record which path triggered "done" for this ego (first one wins).
                    if idx not in self._termination_path:
                        try:
                            _now_sim_log = float(GameTime.get_time())
                        except Exception:
                            _now_sim_log = -1.0
                        if ego_completed or ego_route_completion_100:
                            path = "route_complete"
                            self._termination_path[idx] = path
                            print(
                                f"[TERM_DECISION] ego={idx} sim_t={_now_sim_log:.1f}s "
                                f"path={path} rc={route_completion_pct_for_ego:.1f}% "
                                f"completed={ego_completed} rc100={ego_route_completion_100}",
                                flush=True,
                            )
                        elif ego_blocked:
                            path = "blocked"
                            self._termination_path[idx] = path
                            print(
                                f"[TERM_DECISION] ego={idx} sim_t={_now_sim_log:.1f}s "
                                f"path={path} (AgentBlockedTest fired) "
                                f"rc={route_completion_pct_for_ego:.1f}%",
                                flush=True,
                            )
                        elif ego_deviated:
                            path = "deviated"
                            self._termination_path[idx] = path
                            print(
                                f"[TERM_DECISION] ego={idx} sim_t={_now_sim_log:.1f}s "
                                f"path={path} (InRouteTest FAILURE) "
                                f"rc={route_completion_pct_for_ego:.1f}%",
                                flush=True,
                            )
                        elif ego_collided_terminal:
                            path = "collided_terminal"
                            self._termination_path[idx] = path
                            print(
                                f"[TERM_DECISION] ego={idx} sim_t={_now_sim_log:.1f}s "
                                f"path={path} (CollisionTest FAILURE) "
                                f"rc={route_completion_pct_for_ego:.1f}%",
                                flush=True,
                            )
                    if (ego_completed or ego_route_completion_100) and idx not in self._runtime_completed_ego_cleanup_done:
                        # Once route completion reaches terminal state (SUCCESS or 100% RC),
                        # retire this ego immediately so it cannot keep receiving controls.
                        try:
                            self._agent.cleanup_single(idx)
                        except Exception:
                            pass
                        try:
                            self._agent.cleanup_rsu(idx)
                        except Exception:
                            pass
                        completed_actor = CarlaDataProvider.get_hero_actor(hero_id=idx)
                        completed_actor_id = getattr(completed_actor, "id", None)
                        if completed_actor_id is not None:
                            try:
                                CarlaDataProvider.remove_actor_by_id(
                                    int(completed_actor_id),
                                    max_retries=0,
                                    timeout_s=0.5,
                                    poll_s=0.02,
                                    direct_fallback=False,
                                    reason="scenario_manager_completed_ego_cleanup",
                                    phase="route_completion_success",
                                )
                            except Exception:
                                pass
                        self._runtime_completed_ego_cleanup_done.add(idx)
                        self._runtime_dead_ego_cleanup_done.add(idx)
                        if os.environ.get('DEBUG_FINISH', '').lower() in ('1', 'true', 'yes'):
                            print(
                                f"[DEBUG FINISH] Ego {idx}: completed ego despawn triggered "
                                f"(success={ego_completed}, rc100={ego_route_completion_100})"
                            )
                    # Position-based stuck detector.  Independent backstop
                    # for AgentBlockedTest — uses cumulative DISTANCE over a
                    # rolling window, so micro-spikes (ego slipping against
                    # a wall, or rebound from collision) don't reset the
                    # detection.  An ego that hasn't moved more than
                    # STATIONARY_DISTANCE_M (default 2 m) over the past
                    # STATIONARY_WINDOW_S (default 30 sim sec) is marked
                    # terminal.  Resistant to noise that breaks
                    # AgentBlockedTest's instant-reset semantics.
                    # Disable: CARLA_STATIONARY_DETECTOR_DISABLE=1.
                    ego_stationary_terminal = False
                    if (
                        not ego_completed
                        and not ego_route_completion_100
                        and not ego_blocked
                        and not ego_soft_complete
                        and os.environ.get("CARLA_STATIONARY_DETECTOR_DISABLE", "0").lower() not in ("1", "true", "yes")
                    ):
                        try:
                            _stat_window_s = float(os.environ.get("CARLA_STATIONARY_WINDOW_S", "30.0"))
                            _stat_dist_m = float(os.environ.get("CARLA_STATIONARY_DISTANCE_M", "2.0"))
                        except Exception:
                            _stat_window_s, _stat_dist_m = 30.0, 2.0
                        try:
                            _ego_actor = CarlaDataProvider.get_hero_actor(hero_id=idx)
                            _ego_loc = CarlaDataProvider.get_location(_ego_actor) if _ego_actor else None
                            _now_sim = float(GameTime.get_time())
                        except Exception:
                            _ego_loc = None
                            _now_sim = None
                        if _ego_loc is not None and _now_sim is not None:
                            # Maintain a rolling deque of (sim_t, x, y) per ego.
                            if not hasattr(self, "_ego_position_history"):
                                self._ego_position_history = {}
                            hist = self._ego_position_history.setdefault(idx, [])
                            hist.append((_now_sim, float(_ego_loc.x), float(_ego_loc.y)))
                            # Trim history older than 2x window.
                            cutoff = _now_sim - 2 * _stat_window_s
                            while hist and hist[0][0] < cutoff:
                                hist.pop(0)
                            # Find the position from window_s ago.
                            target_t = _now_sim - _stat_window_s
                            past_pos = None
                            if hist[0][0] <= target_t:
                                # Binary search would be nicer; linear is fine for short list.
                                for h in hist:
                                    if h[0] <= target_t:
                                        past_pos = h
                                    else:
                                        break
                            if past_pos is not None:
                                dist_moved = ((_ego_loc.x - past_pos[1]) ** 2
                                              + (_ego_loc.y - past_pos[2]) ** 2) ** 0.5
                                if dist_moved < _stat_dist_m:
                                    ego_stationary_terminal = True
                                    self._stationary_terminal.add(idx)
                                    self._termination_path.setdefault(idx, "stationary_position")
                                    try:
                                        print(
                                            f"[TERM_DECISION] ego={idx} sim_t={_now_sim:.1f}s "
                                            f"path=stationary_position "
                                            f"rc={route_completion_pct_for_ego:.1f}% "
                                            f"moved_in_last_{_stat_window_s:.0f}s={dist_moved:.2f}m "
                                            f"(threshold={_stat_dist_m:.1f}m, position-based, "
                                            f"micro-spike-immune)",
                                            flush=True,
                                        )
                                    except Exception:
                                        pass

                    # Ego is "done" if it satisfies ANY of:
                    #   1. ego_completed              (RouteCompletionTest SUCCESS)
                    #   2. ego_route_completion_100   (RC% >= 100, faster than #1)
                    #   3. ego_blocked                (AgentBlockedTest FAILURE)
                    #   4. ego_soft_complete          (Tier 2: high RC + stopped + dwell)
                    #   5. ego_stationary_terminal    (position-based: didn't move in window)
                    #   6. ego_deviated               (InRouteTest FAILURE — off-route)
                    #   7. ego_collided_terminal      (CollisionTest FAILURE — destroyed)
                    # The despawned-ego case is handled earlier (continue at top of loop).
                    if not (ego_completed or ego_route_completion_100 or ego_blocked
                            or ego_soft_complete or ego_stationary_terminal
                            or ego_deviated or ego_collided_terminal):
                        all_egos_done = False
                        break

            if all_egos_done and self._running:
                if log_replay_ego and self.ego_vehicles_num > 0:
                    flags = []
                    for idx in range(self.ego_vehicles_num):
                        try:
                            done_key = f"log_replay_done_ego_{idx}"
                            flags.append(bool(py_trees.blackboard.Blackboard().get(done_key)))
                        except Exception:
                            flags.append(False)
                    done_details = f"replay_done={flags}"
                self._debug_reset("all_egos_done", details=done_details)
                if os.environ.get('DEBUG_FINISH', '').lower() in ('1', 'true', 'yes'):
                    print(f"[DEBUG FINISH] ALL EGOS DONE - setting _running = False")
                # Single-line termination summary so an offline analyzer can
                # confirm WHY the scenario ended without parsing per-ego logs.
                # Each ego appears with its resolved termination_path; absent
                # entries mean "skipped via _runtime_completed_ego_cleanup_done"
                # (route-completion despawn).
                try:
                    _now_sim_end = float(GameTime.get_time())
                except Exception:
                    _now_sim_end = -1.0
                paths_summary = ",".join(
                    f"ego_{i}={self._termination_path.get(i, 'cleanup_completed') if i in self._runtime_completed_ego_cleanup_done else self._termination_path.get(i, 'unknown')}"
                    for i in range(self.ego_vehicles_num)
                )
                print(
                    f"[SCENARIO_TERMINATED] sim_t={_now_sim_end:.1f}s "
                    f"trigger=all_egos_done n_egos={self.ego_vehicles_num} paths=[{paths_summary}]",
                    flush=True,
                )
                self._running = False

            # Position-based stall detection (conservative fallback)
            # Only check if still running after criteria checks
            if self._running and self._stall_detection_enabled:
                stall_detected = self._check_position_stall()
                if stall_detected:
                    self._debug_reset(
                        "stall_detection",
                        details=f"threshold_time={self._stall_threshold_time} min_dist={self._stall_min_distance}",
                    )
                    print(f"\n[STALL DETECTION] All vehicles have been stationary for {self._stall_threshold_time}s - terminating scenario")
                    self._running = False

            self._logical_frame_id += 1
            if callable(self._external_tick_callback):
                try:
                    self._external_tick_callback(
                        {
                            "logical_frame_id": int(self._logical_frame_id),
                            "carla_frame": int(getattr(timestamp, "frame", -1)),
                            "elapsed_seconds": float(getattr(timestamp, "elapsed_seconds", 0.0)),
                            "ego_action": self._last_ego_action,
                            "running": bool(self._running),
                        }
                    )
                except Exception as exc:
                    print(f"[ScenarioManager] Tick callback error: {exc}")
            if (
                self._enable_test_hooks
                and self._forced_infra_crash_frames
                and self._logical_frame_id in self._forced_infra_crash_frames
            ):
                self._forced_infra_crash_frames.discard(self._logical_frame_id)
                raise RuntimeError(
                    "CARLA forced infrastructure crash injection at logical_frame={}".format(
                        self._logical_frame_id
                    )
                )

        if self._running and self.get_running_status():
            try:
                self._forensics.end_phase("apply_control_and_scenario")
                self._forensics.begin_phase("world_tick")
            except Exception:
                pass
            CarlaDataProvider.get_world().tick(self._timeout)
            try:
                self._forensics.end_phase("world_tick")
                self._forensics.end_tick()
            except Exception:
                pass
        else:
            # Tick was skipped (running=False or watchdog tripped). Still
            # close out the tick record so forensics totals stay accurate.
            try:
                self._forensics.end_phase("apply_control_and_scenario")
                self._forensics.end_tick()
            except Exception:
                pass

    def _record_pdm_sample(self, vehicle_num, ego, timestamp):
        """
        Record per-tick kinematics for PDM-style metrics.
        """
        if vehicle_num >= len(self.pdm_traces):
            return
        try:
            self._record_pdm_world(timestamp)
            transform = ego.get_transform()
            vel = ego.get_velocity()
            acc = ego.get_acceleration()
            ang_vel = ego.get_angular_velocity()
            extent = ego.bounding_box.extent

            current_time = timestamp.elapsed_seconds
            prev_time = self._pdm_prev_time[vehicle_num]
            dt = (
                timestamp.delta_seconds
                if hasattr(timestamp, "delta_seconds") and timestamp.delta_seconds
                else (current_time - prev_time if prev_time is not None else 0.05)
            )

            prev_ang = self._pdm_prev_ang_vel[vehicle_num]
            ang_acc = 0.0
            if prev_ang is not None and dt and dt > 0:
                ang_acc = (ang_vel.z - prev_ang) / dt

            self._pdm_prev_ang_vel[vehicle_num] = ang_vel.z
            self._pdm_prev_time[vehicle_num] = current_time

            self.pdm_traces[vehicle_num].append(
                {
                    "t": current_time,
                    "actor_id": ego.id,
                    "x": transform.location.x,
                    "y": transform.location.y,
                    "yaw": math.radians(transform.rotation.yaw),
                    "vel_x": vel.x,
                    "vel_y": vel.y,
                    "accel_x": acc.x,
                    "accel_y": acc.y,
                    "ang_vel": ang_vel.z,
                    "ang_acc": ang_acc,
                    "extent_x": extent.x,
                    "extent_y": extent.y,
                }
            )
        except Exception:
            pass

    def _record_pdm_world(self, timestamp):
        """
        Record world actors and traffic light states once per tick.
        For each ego vehicle, we need separate world traces that exclude that specific ego
        but include other egos as potential collision targets.
        """
        current_time = timestamp.elapsed_seconds
        if self._pdm_last_world_time is not None and abs(current_time - self._pdm_last_world_time) < 1e-6:
            return
        self._pdm_last_world_time = current_time

        world = CarlaDataProvider.get_world()
        if world is None:
            return

        # Get all hero IDs
        hero_actors = {}
        for ego_id in range(self.ego_vehicles_num):
            hero = CarlaDataProvider.get_hero_actor(hero_id=ego_id)
            if hero is not None:
                hero_actors[ego_id] = hero

        all_actors = world.get_actors()
        
        if os.environ.get('DEBUG_PDM', '').lower() in ('1', 'true', 'yes'):
            if len(self.pdm_world_trace) == 0:
                print(f"[DEBUG PDM] _record_pdm_world first call: total world actors={len(all_actors)}, num_egos={len(hero_actors)}")
                # Sample some non-vehicle/pedestrian actors to see what they are
                other_types = {}
                for actor in all_actors:
                    if actor.id not in [h.id for h in hero_actors.values()] and not actor.type_id.startswith("vehicle.") and not actor.type_id.startswith("walker.pedestrian"):
                        actor_type = actor.type_id.split('.')[0] if '.' in actor.type_id else actor.type_id
                        other_types[actor_type] = other_types.get(actor_type, 0) + 1
                if other_types:
                    print(f"[DEBUG PDM] Non-vehicle/pedestrian actor types: {other_types}")
        
        # Collect all vehicles and pedestrians (including OTHER egos)
        actors = []
        vehicle_count = 0
        pedestrian_count = 0
        
        for actor in all_actors:
            try:
                if actor.type_id.startswith("vehicle."):
                    vehicle_count += 1
                elif actor.type_id.startswith("walker.pedestrian"):
                    pedestrian_count += 1
                    
                if actor.type_id.startswith("vehicle.") or actor.type_id.startswith("walker.pedestrian"):
                    transform = actor.get_transform()
                    vel = actor.get_velocity()
                    extent = actor.bounding_box.extent
                    actors.append(
                        {
                            "id": actor.id,
                            "type": actor.type_id,
                            "x": transform.location.x,
                            "y": transform.location.y,
                            "yaw": math.radians(transform.rotation.yaw),
                            "vel_x": vel.x,
                            "vel_y": vel.y,
                            "extent_x": extent.x,
                            "extent_y": extent.y,
                        }
                    )
            except Exception as e:
                if os.environ.get('DEBUG_PDM', '').lower() in ('1', 'true', 'yes') and len(self.pdm_world_trace) == 0:
                    print(f"[DEBUG PDM] Exception processing actor: {e}")
                continue
        
        if os.environ.get('DEBUG_PDM', '').lower() in ('1', 'true', 'yes'):
            if len(self.pdm_world_trace) == 0:
                print(f"[DEBUG PDM] After filtering: vehicles={vehicle_count} (including {len(hero_actors)} egos), pedestrians={pedestrian_count}, final_actors={len(actors)}")

        # cache traffic light polygons (static)
        if not self.pdm_tl_polygons:
            for tl in world.get_actors().filter("*traffic_light*"):
                try:
                    trig = tl.trigger_volume
                    tf = tl.get_transform()
                    # trigger volume center in local space
                    center = trig.location
                    extent = trig.extent
                    # build 4 corners in local frame
                    corners = [
                        carla.Location(x=center.x + extent.x, y=center.y + extent.y),
                        carla.Location(x=center.x + extent.x, y=center.y - extent.y),
                        carla.Location(x=center.x - extent.x, y=center.y - extent.y),
                        carla.Location(x=center.x - extent.x, y=center.y + extent.y),
                    ]
                    world_corners = [tf.transform(loc) for loc in corners]
                    self.pdm_tl_polygons[tl.id] = [(c.x, c.y) for c in world_corners]
                except Exception:
                    continue

        tl_states = {}
        for tl in world.get_actors().filter("*traffic_light*"):
            try:
                tl_states[tl.id] = int(tl.state)
            except Exception:
                continue

        self.pdm_world_trace.append(
            {
                "t": current_time,
                "actors": actors,
                "traffic_lights": tl_states,
            }
        )
        
        if os.environ.get('DEBUG_PDM', '').lower() in ('1', 'true', 'yes'):
            if len(self.pdm_world_trace) == 1 or len(self.pdm_world_trace) % 50 == 0:
                print(f"[DEBUG PDM] pdm_world_trace length: {len(self.pdm_world_trace)}, actors in this frame: {len(actors)}")

    def get_running_status(self):
        """
        returns:
           bool: False if watchdog exception occured, True otherwise
        """
        return self._watchdog.get_status()

    def _check_position_stall(self):
        """
        Conservative position-based stall detection.
        Returns True if ALL ego vehicles have been stationary (moved < threshold) 
        for the stall threshold time.
        
        This is a fallback mechanism in case velocity-based detection fails.
        """
        import math
        
        current_time = time.time()
        
        # Only check periodically to reduce overhead
        if current_time - self._stall_last_check_time < self._stall_check_interval:
            return False
        self._stall_last_check_time = current_time
        
        # Record current positions for all active egos
        active_ego_count = 0
        for vehicle_num in range(self.ego_vehicles_num):
            ego = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)
            if ego and ego.is_alive:
                active_ego_count += 1
                loc = ego.get_location()
                if vehicle_num not in self._stall_position_history:
                    self._stall_position_history[vehicle_num] = []
                self._stall_position_history[vehicle_num].append((current_time, loc.x, loc.y, loc.z))
                
                # Keep only positions within the threshold window (plus some buffer)
                cutoff_time = current_time - self._stall_threshold_time - 10.0
                self._stall_position_history[vehicle_num] = [
                    p for p in self._stall_position_history[vehicle_num] if p[0] >= cutoff_time
                ]
        
        # If no active egos, don't report stall (let other logic handle termination)
        if active_ego_count == 0:
            return False
        
        # Check if ALL active egos have been stationary for the threshold time
        all_stalled = True
        for vehicle_num in range(self.ego_vehicles_num):
            ego = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)
            if not ego or not ego.is_alive:
                continue  # Skip inactive egos
                
            history = self._stall_position_history.get(vehicle_num, [])
            if not history:
                all_stalled = False
                break
            
            # Need at least threshold_time worth of history
            oldest_time = history[0][0]
            if current_time - oldest_time < self._stall_threshold_time:
                all_stalled = False
                break
            
            # Calculate max displacement from oldest position in threshold window
            oldest_in_window = None
            for t, x, y, z in history:
                if current_time - t <= self._stall_threshold_time:
                    if oldest_in_window is None:
                        oldest_in_window = (x, y, z)
                    break
            
            if oldest_in_window is None:
                all_stalled = False
                break
            
            # Calculate displacement from oldest to current
            current_pos = history[-1]
            dx = current_pos[1] - oldest_in_window[0]
            dy = current_pos[2] - oldest_in_window[1]
            dz = current_pos[3] - oldest_in_window[2]
            distance = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            # Also check current velocity - if moving at all, not stalled
            vel = ego.get_velocity()
            speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            
            if distance >= self._stall_min_distance or speed > 0.1:
                # This ego has moved enough OR is currently moving, not stalled
                all_stalled = False
                if self._stall_debug_logging:
                    print(
                        f"[STALL DEBUG] Ego {vehicle_num} NOT stalled: "
                        f"distance={distance:.2f}m, speed={speed:.2f}m/s"
                    )
                break
        
        # Debug: if all stalled, print warning before triggering
        if all_stalled:
            if self._stall_debug_logging:
                print(f"[STALL DEBUG] All {active_ego_count} egos appear stalled. Checking one more time...")
            # Double-check: verify ALL velocities are near zero right now
            for vehicle_num in range(self.ego_vehicles_num):
                ego = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)
                if ego and ego.is_alive:
                    vel = ego.get_velocity()
                    speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                    if speed > 0.1:
                        if self._stall_debug_logging:
                            print(f"[STALL DEBUG] Wait - Ego {vehicle_num} has speed={speed:.2f}m/s, NOT stalled!")
                        all_stalled = False
                        break
        
        return all_stalled

    def stop_scenario(self):
        """
        This function triggers a proper termination of a scenario
        """
        self._debug_reset("stop_scenario_called")
        self._watchdog.stop()
        # Stop tick forensics so daemon threads exit and final snapshot fires.
        try:
            if hasattr(self, "_forensics") and self._forensics is not None:
                self._forensics.stop()
        except Exception as _e:
            print(f"[TICK_FORENSICS] STOP_FAILED err={type(_e).__name__}: {_e}", flush=True)

        # Print and optionally save per-tick timing summary
        _tick_count = len(self.a_time_record)
        if _tick_count > 0:
            def _ms_stats(records):
                if not records:
                    return "n/a"
                _n = len(records)
                _mean_ms = sum(records) / _n * 1000
                _max_ms  = max(records) * 1000
                _min_ms  = min(records) * 1000
                return f"mean={_mean_ms:.2f}ms max={_max_ms:.2f}ms min={_min_ms:.2f}ms n={_n}"
            _lines = [
                f"[ScenarioManager tick timing summary] ticks={_tick_count}",
                f"  world_tick (CarlaDataProvider): {_ms_stats(self.c_time_record)}",
                f"  agent_call (sensor+inference):  {_ms_stats(self.a_time_record)}",
                f"  apply_control (+PDM sample):    {_ms_stats(self.time_record)}",
                f"  scenario_tree (tick_once):      {_ms_stats(self.sc_time_record)}",
            ]
            _summary = '\n'.join(_lines)
            print(_summary)
            _save_path = os.environ.get("SAVE_PATH", "").strip()
            if _save_path:
                _timing_log = os.path.join(_save_path, "tick_timing.log")
                try:
                    with open(_timing_log, 'a') as _f:
                        _f.write(_summary + '\n')
                except Exception:
                    pass

        self.end_system_time = time.time()
        self.end_game_time = GameTime.get_time()

        segment_duration_system = self.end_system_time - self.start_system_time
        segment_duration_game = self.end_game_time - self.start_game_time
        self.scenario_duration_system = float(self._recovery_duration_offset_system) + float(segment_duration_system)
        self.scenario_duration_game = float(self._recovery_duration_offset_game) + float(segment_duration_game)
        self._recovery_duration_offset_system = 0.0
        self._recovery_duration_offset_game = 0.0

        watchdog_ok = self.get_running_status()
        try:
            # terminate scenario trees regardless of watchdog status (best effort)
            if len(self.ego_vehicles) == 0:
                for scenario_item in self.scenario:
                    if scenario_item is not None:
                        scenario_item.terminate()
            else:
                for ego_vehicle_id in range(len(self.ego_vehicles)):
                    if self.scenario[ego_vehicle_id] is not None:
                        self.scenario[ego_vehicle_id].terminate()

            if watchdog_ok:
                self.analyze_scenario()
        finally:
            # Resource cleanup must be unconditional, even if watchdog has failed.
            self._teardown_overhead_capture()
            if self._agent is not None:
                try:
                    self._agent.cleanup()
                except Exception:
                    pass
                self._agent = None

            if self.sensor_tf_list is not None:
                for _sensor in self.sensor_tf_list:
                    try:
                        _sensor.cleanup()
                    except Exception:
                        pass
                self.sensor_tf_list = None

    def analyze_scenario(self):
        """
        Analyzes and prints the results of the route
        """
        global_result = '\033[92m'+'SUCCESS'+'\033[0m'
        for ego_vehicle_id in range(len(self.ego_vehicles)):
            for criterion in self.scenario[ego_vehicle_id].get_criteria():
                if criterion.test_status != "SUCCESS":
                    global_result = '\033[91m'+'FAILURE'+'\033[0m'

            if self.scenario[ego_vehicle_id].timeout_node.timeout:
                global_result = '\033[91m'+'FAILURE'+'\033[0m'

            ResultOutputProvider(self, global_result, ego_vehicle_id)
