#!/usr/bin/env python
# Copyright (c) 2018-2019 Intel Corporation.
# authors: German Ros (german.ros@intel.com), Felipe Codevilla (felipe.alcm@gmail.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA Challenge Evaluator Routes

Provisional code to evaluate Autonomous Agents for the CARLA Autonomous Driving challenge
"""
from __future__ import print_function

import traceback
import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
import importlib
import os
import sys
import gc
import sys
import carla
import copy
import signal
import torch
import time
import threading
import json
import yaml
import numpy as np
# Patch for numpy 1.24+ compatibility with older libraries (like networkx < 2.6)
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'bool'):
    np.bool = bool
import random
import shutil

from srunner.scenariomanager.carla_data_provider import *
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from leaderboard.scenarios.scenario_manager import ScenarioManager
from leaderboard.scenarios.route_scenario import RouteScenario
from leaderboard.envs.sensor_interface import SensorInterface, SensorConfigurationInvalid
from leaderboard.autoagents.agent_wrapper import  AgentWrapper, AgentError
from leaderboard.utils.statistics_manager import StatisticsManager
from leaderboard.utils.route_indexer import RouteIndexer
from leaderboard.utils.route_parser import RouteParser
from leaderboard.recovery.recovery_manager import RecoveryManager

try:
    from common.carla_connection_events import (
        create_logged_client,
        install_process_lifecycle_logging,
        log_carla_event,
        log_process_exception,
    )
except Exception:  # pragma: no cover - keep evaluator runnable without repo root on PYTHONPATH
    def create_logged_client(carla_module, host, port, *, timeout_s=None, context="", attempt=None, process_name=None):
        del context, attempt, process_name
        client = carla_module.Client(host, int(port))
        if timeout_s is not None:
            client.set_timeout(float(timeout_s))
        return client

    def install_process_lifecycle_logging(process_name, *, env_keys=None):
        del process_name, env_keys

    def log_carla_event(event_type, *, process_name=None, **fields):
        del event_type, process_name, fields

    def log_process_exception(exc, *, process_name, where=""):
        del exc, process_name, where


EVALUATOR_PROCESS_NAME = "leaderboard_evaluator_parameter"

def check_log_file(file_path, target_string):
    try:
        # 打开文件
        with open(file_path, 'r') as file:
            # 读取文件内容
            file_content = file.read()

            # 判断文件内容中是否包含特定字符
            if target_string in file_content:
                # print(f"文件中包含目标字符串 '{target_string}'")
                return True
            else:
                # print(f"文件中不包含目标字符串 '{target_string}'")
                return False

    except FileNotFoundError:
        print(f"文件 '{file_path}' 不存在")

class Logger(object):
    def __init__(self, file_name = 'temp.log', stream = sys.stdout) -> None:
        self.terminal = stream
        self.file_name = file_name
        self.log = None
        self._closed = False
        try:
            # Line-buffered logging file to avoid descriptor churn and flush on newlines.
            self.log = open(self.file_name, "a", buffering=1)
        except Exception:
            self.log = None

    def write(self, message):
        try:
            self.terminal.write(message)
        except Exception:
            pass
        try:
            if not self._closed and self.log:
                self.log.write(message)
        except Exception:
            pass

    def flush(self):
        try:
            self.terminal.flush()
        except Exception:
            pass
        try:
            if not self._closed and self.log:
                self.log.flush()
        except Exception:
            pass

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.log:
                self.log.flush()
                self.log.close()
        except Exception:
            pass
        self.log = None


def _append_traceback_to_log(log_file_path):
    if not log_file_path:
        return
    try:
        with open(log_file_path, "a") as log_file:
            traceback.print_exc(file=log_file)
    except Exception:
        pass

def backup_script(full_path, folders_to_save=["simulation/leaderboard/leaderboard","simulation/leaderboard/team_code", "simulation/scenario_runner"]):
    target_folder = os.path.join(full_path, 'scripts')
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    else:
        return
    
    current_path = os.path.dirname(__file__)  # __file__ refer to this file, then the dirname is "?/tools"

    for folder_name in folders_to_save:
        ttarget_folder = os.path.join(target_folder, folder_name)
        source_folder = os.path.join(current_path, f'../../../{folder_name}')
        shutil.copytree(source_folder, ttarget_folder)

sensors_to_icons = {
    'sensor.camera.rgb':        'carla_camera',
    'sensor.camera.semantic_segmentation': 'carla_camera',
    'sensor.camera.depth':      'carla_camera',
    'sensor.lidar.ray_cast':    'carla_lidar',
    'sensor.lidar.ray_cast_semantic':    'carla_lidar',
    'sensor.other.radar':       'carla_radar',
    'sensor.other.collision':   'carla_collision',
    'sensor.other.gnss':        'carla_gnss',
    'sensor.other.imu':         'carla_imu',
    'sensor.opendrive_map':     'carla_opendrive_map',
    'sensor.speedometer':       'carla_speedometer'
}


class LeaderboardEvaluator(object):

    """
    TODO: document me!
    """

    ego_vehicles = []

    # Tunable parameters
    client_timeout = 10.0  # in seconds
    wait_for_world = 20.0  # in seconds
    frame_rate = 20.0      # in Hz

    def __init__(self, args, statistics_manager):
        """
        Setup CARLA client and world
        Setup ScenarioManager
        """
        # Pre-init teardown bookkeeping to its zero state BEFORE any code path
        # that can call _cleanup() — including the SIGINT handler installed
        # later in __init__.  Without this, a signal arriving between the
        # signal.signal() call and the legacy "self._teardown_seq = 0" line
        # below trips an AttributeError in _cleanup, which the failure
        # classifier then maps to no_route_execution.
        self._teardown_seq = 0
        self._teardown_in_progress = False
        self._last_teardown_completed_seq = 0
        self._last_teardown_duration_s = 0.0
        self.ego_vehicles = []

        self.statistics_manager = statistics_manager
        self._args = args
        self.sensors = None
        self.sensor_icons = []
        self._vehicle_lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
        self._route_plots_only = bool(getattr(args, "route_plots_only", False))

        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        if args.timeout:
            self.client_timeout = float(args.timeout)
        self.client = create_logged_client(
            carla,
            args.host,
            int(args.port),
            timeout_s=self.client_timeout,
            context="leaderboard_evaluator_init",
            process_name=EVALUATOR_PROCESS_NAME,
        )
        log_carla_event(
            "EVALUATOR_INIT",
            process_name=EVALUATOR_PROCESS_NAME,
            host=args.host,
            port=int(args.port),
            tm_port=int(args.trafficManagerPort),
            timeout_s=float(self.client_timeout),
            route_plots_only=int(self._route_plots_only),
            ego_vehicles=int(args.ego_num),
        )

        self.traffic_manager = self.client.get_trafficmanager(int(args.trafficManagerPort))

        # Load agent module unless we only need route plotting artifacts
        self.module_agent = None
        if not self._route_plots_only:
            module_name = os.path.basename(args.agent).split('.')[0]
            sys.path.insert(0, os.path.dirname(args.agent))
            self.module_agent = importlib.import_module(module_name)

        # Create the ScenarioManager
        self.manager = ScenarioManager(args.timeout, args.debug > 1)

        # Time control for summary purposes
        self._start_time = GameTime.get_time()
        self._end_time = None

        # Create the agent timer
        self._agent_watchdog = Watchdog(int(float(args.timeout)))
        signal.signal(signal.SIGINT, self._signal_handler)

        self.ego_vehicles_num = args.ego_num
        self._teardown_seq = 0
        self._teardown_in_progress = False
        self._last_teardown_completed_seq = 0
        self._last_teardown_duration_s = 0.0
        self.agent_instance = None
        self._current_route_recovery_metadata = {}
        recovery_mode = str(os.environ.get("CARLA_RECOVERY_MODE", "off")).strip().lower()
        recovery_flag = str(os.environ.get("CARLA_CHECKPOINT_RECOVERY_ENABLE", "0")).strip().lower()
        recovery_enabled = recovery_mode != "off" and recovery_flag in ("1", "true", "yes", "on")
        self._recovery_hard_off = not bool(recovery_enabled)
        self.recovery_manager = None
        if recovery_enabled:
            self.recovery_manager = RecoveryManager(
                evaluator=self,
                args=args,
                run_root=os.path.dirname(args.checkpoint),
            )
        self._reset_route_runtime_ego_mapping()

    def _reset_route_runtime_ego_mapping(self):
        self._current_route_runtime_ego_count = int(self.ego_vehicles_num)
        self._current_route_active_to_original_ego_index = list(
            range(int(self.ego_vehicles_num))
        )
        self._current_route_original_to_active_ego_index = {
            int(idx): int(idx) for idx in range(int(self.ego_vehicles_num))
        }
        self._current_route_spawn_metadata = {}

    def _capture_route_runtime_ego_mapping(self, scenario):
        runtime_ego_count = int(
            getattr(
                scenario,
                "runtime_ego_vehicle_count",
                getattr(scenario, "ego_vehicles_num", self.ego_vehicles_num),
            )
            or 0
        )
        active_to_original = list(
            getattr(
                scenario,
                "active_to_original_ego_index",
                list(range(runtime_ego_count)),
            )
            or []
        )
        if len(active_to_original) != runtime_ego_count:
            active_to_original = list(range(runtime_ego_count))
        original_to_active = {
            int(original_idx): int(active_idx)
            for active_idx, original_idx in enumerate(active_to_original)
        }
        requested_ego_count = int(
            getattr(scenario, "requested_ego_vehicle_count", self.ego_vehicles_num)
            or self.ego_vehicles_num
        )
        skipped_ego_indices = [
            idx
            for idx in range(requested_ego_count)
            if idx not in original_to_active
        ]
        spawn_failures = list(getattr(scenario, "ego_spawn_failures", []) or [])
        self._current_route_runtime_ego_count = runtime_ego_count
        self._current_route_active_to_original_ego_index = active_to_original
        self._current_route_original_to_active_ego_index = original_to_active
        self._current_route_spawn_metadata = {
            "partial_spawn_accepted": bool(
                getattr(scenario, "partial_ego_spawn_accepted", False)
            ),
            "requested_ego_count": requested_ego_count,
            "runtime_ego_count": runtime_ego_count,
            "active_to_original_ego_index": active_to_original,
            "original_to_active_ego_index": {
                str(key): value for key, value in original_to_active.items()
            },
            "skipped_ego_indices": skipped_ego_indices,
            "ego_spawn_failures": spawn_failures,
        }
        return runtime_ego_count

    def _count_manager_tracked_sensors(self):
        manager_agent = getattr(getattr(self, "manager", None), "_agent", None)
        sensor_lists = getattr(manager_agent, "_sensors_list", None)
        if not isinstance(sensor_lists, list):
            return 0
        total = 0
        for sensors in sensor_lists:
            if not isinstance(sensors, list):
                continue
            for sensor in sensors:
                if sensor is not None:
                    total += 1
        return int(total)

    def _collect_world_sensor_actor_ids(self):
        world = getattr(self, "world", None)
        if world is None:
            return []
        sensor_ids = []
        try:
            for sensor in world.get_actors().filter("*sensor*"):
                if sensor is None:
                    continue
                sensor_id = getattr(sensor, "id", None)
                if sensor_id is None:
                    continue
                sensor_ids.append(int(sensor_id))
        except Exception:
            return []
        sensor_ids = sorted(set(sensor_ids))
        return sensor_ids

    def _stop_world_sensor_streams(self, *, stage):
        world = getattr(self, "world", None)
        summary = {
            "sensor_ids": [],
            "stopped": 0,
            "failed": 0,
        }
        if world is None:
            return summary

        try:
            sensors = list(world.get_actors().filter("*sensor*"))
        except Exception:
            return summary
        summary["sensor_ids"] = sorted(
            {
                int(getattr(sensor, "id"))
                for sensor in sensors
                if sensor is not None and getattr(sensor, "id", None) is not None
            }
        )
        log_carla_event(
            "EVALUATOR_SENSOR_STOP_BEGIN",
            process_name=EVALUATOR_PROCESS_NAME,
            stage=stage,
            sensor_count=len(summary["sensor_ids"]),
        )
        for sensor in sensors:
            if sensor is None:
                continue
            sensor_id = getattr(sensor, "id", None)
            sensor_type = getattr(sensor, "type_id", "")
            stop_status = "ok"
            try:
                sensor.stop()
                summary["stopped"] += 1
            except Exception:
                stop_status = "fail"
                summary["failed"] += 1
            log_carla_event(
                "EVALUATOR_SENSOR_STOP",
                process_name=EVALUATOR_PROCESS_NAME,
                stage=stage,
                sensor_id=sensor_id,
                sensor_type=sensor_type,
                status=stop_status,
            )
        log_carla_event(
            "EVALUATOR_SENSOR_STOP_END",
            process_name=EVALUATOR_PROCESS_NAME,
            stage=stage,
            sensor_count=len(summary["sensor_ids"]),
            stopped=summary["stopped"],
            failed=summary["failed"],
        )
        return summary

    def _collect_teardown_counts(self):
        tracked_actor_pool = getattr(CarlaDataProvider, "_carla_actor_pool", {})
        return {
            "tracked_sensors": int(self._count_manager_tracked_sensors()),
            "tracked_actor_pool": int(len(tracked_actor_pool) if isinstance(tracked_actor_pool, dict) else 0),
            "world_sensor_actors": int(len(self._collect_world_sensor_actor_ids())),
            "ego_refs": int(len(self.ego_vehicles)),
        }

    def _collect_investigative_actor_counts(self):
        counts = dict(self._collect_teardown_counts())
        world_actor_total = None
        if hasattr(self, "world") and self.world is not None:
            try:
                world_actor_total = int(len(self.world.get_actors()))
            except Exception:
                world_actor_total = None
        hero_alive = 0
        for ego_id in range(int(getattr(self, "ego_vehicles_num", 0) or 0)):
            try:
                hero = CarlaDataProvider.get_hero_actor(hero_id=ego_id)
            except Exception:
                hero = None
            if hero is not None and getattr(hero, "is_alive", True):
                hero_alive += 1
        counts["world_actor_total"] = world_actor_total
        counts["hero_alive"] = int(hero_alive)
        return counts

    def _emit_actor_snapshot(self, label, **extra_fields):
        payload = {"label": str(label)}
        payload.update(self._collect_investigative_actor_counts())
        for key, value in extra_fields.items():
            if value is not None:
                payload[str(key)] = value
        print("[ACTOR SNAPSHOT] " + json.dumps(payload, sort_keys=True))

    def _wait_for_teardown_ready(self, *, stage, timeout_s=10.0):
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        last_counts = self._collect_teardown_counts()
        while time.monotonic() < deadline:
            counts = self._collect_teardown_counts()
            last_counts = counts
            teardown_clean = (
                counts["tracked_sensors"] == 0
                and counts["tracked_actor_pool"] == 0
                and counts["world_sensor_actors"] == 0
                and counts["ego_refs"] == 0
            )
            if (not self._teardown_in_progress) and teardown_clean:
                log_carla_event(
                    "EVALUATOR_START_ALLOWED",
                    process_name=EVALUATOR_PROCESS_NAME,
                    stage=stage,
                    teardown_seq=int(self._last_teardown_completed_seq),
                    teardown_duration_s=float(self._last_teardown_duration_s),
                )
                return True
            time.sleep(0.1)

        log_carla_event(
            "EVALUATOR_START_GATE_TIMEOUT",
            process_name=EVALUATOR_PROCESS_NAME,
            stage=stage,
            teardown_in_progress=int(bool(self._teardown_in_progress)),
            tracked_sensors=last_counts["tracked_sensors"],
            tracked_actor_pool=last_counts["tracked_actor_pool"],
            world_sensor_actors=last_counts["world_sensor_actors"],
            ego_refs=last_counts["ego_refs"],
        )
        return False

    def _wait_for_actor_ids_gone(self, actor_ids, timeout_s=5.0, poll_s=0.1):
        pending = {int(actor_id) for actor_id in actor_ids if actor_id is not None}
        if not pending:
            return set()
        world = getattr(self, "world", None)
        if world is None:
            return set(pending)
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while pending and time.monotonic() < deadline:
            try:
                alive = world.get_actors(list(pending))
                pending = {
                    int(actor.id)
                    for actor in alive
                    if actor is not None and getattr(actor, "is_alive", False)
                }
            except Exception:
                return set(pending)
            if not pending:
                return set()
            time.sleep(max(0.01, float(poll_s)))
        return set(pending)

    def _destroy_actor_ids_with_verification(
        self,
        actor_ids,
        *,
        reason,
        stage,
        stop_sensors=False,
    ):
        actor_ids = sorted({int(actor_id) for actor_id in actor_ids if actor_id is not None})
        if not actor_ids:
            return {"requested_ids": [], "failed_ids": [], "remaining_alive_ids": []}
        if hasattr(CarlaDataProvider, "destroy_actor_ids"):
            try:
                return CarlaDataProvider.destroy_actor_ids(
                    actor_ids,
                    stop_sensors=bool(stop_sensors),
                    reason=reason,
                    phase=stage,
                    max_retries=2,
                    timeout_s=5.0,
                    poll_s=0.1,
                )
            except Exception:
                pass

        # Fallback path if running with an older CarlaDataProvider implementation.
        failed = set(actor_ids)
        try:
            if bool(stop_sensors) and hasattr(self, "world") and self.world is not None:
                try:
                    for sensor in self.world.get_actors(actor_ids):
                        if sensor is None:
                            continue
                        if not str(getattr(sensor, "type_id", "")).startswith("sensor."):
                            continue
                        try:
                            sensor.stop()
                        except Exception:
                            pass
                except Exception:
                    pass
            if self.client is not None:
                destroy_cmds = [carla.command.DestroyActor(int(actor_id)) for actor_id in actor_ids]
                responses = self.client.apply_batch_sync(destroy_cmds, False)
                failed = set()
                for actor_id, response in zip(actor_ids, responses or []):
                    err = str(getattr(response, "error", "") or "")
                    if err:
                        failed.add(actor_id)
                if responses is None or len(responses) < len(actor_ids):
                    failed.update(actor_ids[len(responses or []):])
        except Exception:
            failed = set(actor_ids)
        remaining = self._wait_for_actor_ids_gone(actor_ids, timeout_s=5.0, poll_s=0.1)
        failed.update(remaining)
        return {
            "requested_ids": actor_ids,
            "failed_ids": sorted(failed),
            "remaining_alive_ids": sorted(remaining),
        }

    def _destroy_ego_vehicles(self, *, stage):
        ego_ids = []
        for ego in self.ego_vehicles:
            if ego is None:
                continue
            ego_id = getattr(ego, "id", None)
            if ego_id is None:
                continue
            ego_ids.append(int(ego_id))
        remaining_ego_ids = sorted(
            self._wait_for_actor_ids_gone(
                ego_ids,
                timeout_s=0.5,
                poll_s=0.05,
            )
        )
        destroy_summary = {
            "requested_ids": ego_ids,
            "failed_ids": [],
            "remaining_alive_ids": [],
        }
        if remaining_ego_ids:
            destroy_summary = self._destroy_actor_ids_with_verification(
                remaining_ego_ids,
                reason="leaderboard_evaluator_ego_cleanup",
                stage=stage,
                stop_sensors=False,
            )
        for i, _ in enumerate(self.ego_vehicles):
            self.ego_vehicles[i] = None
        self.ego_vehicles = []
        return destroy_summary

    def _teardown_barrier(self, *, stage, preserve_agent_instance=False):
        """
        Enforce deterministic teardown ordering before the next route can start.
        """
        if self._teardown_in_progress:
            log_carla_event(
                "EVALUATOR_TEARDOWN_REENTRANT",
                process_name=EVALUATOR_PROCESS_NAME,
                stage=stage,
                seq=int(self._teardown_seq),
            )
            return

        self._teardown_seq += 1
        seq = self._teardown_seq
        self._teardown_in_progress = True
        start_monotonic = time.monotonic()
        counts_before = self._collect_teardown_counts()
        self._emit_actor_snapshot("teardown_begin", stage=stage, seq=int(seq))
        log_carla_event(
            "EVALUATOR_TEARDOWN_BEGIN",
            process_name=EVALUATOR_PROCESS_NAME,
            stage=stage,
            seq=int(seq),
            tracked_sensors=counts_before["tracked_sensors"],
            tracked_actor_pool=counts_before["tracked_actor_pool"],
            world_sensor_actors=counts_before["world_sensor_actors"],
            ego_refs=counts_before["ego_refs"],
        )

        # Lower the carla.Client RPC timeout for teardown only.  During the
        # scenario itself the timeout is large because individual ticks can
        # legitimately take many seconds under load (median 3.85s, p99 545s,
        # max 557s observed in this repo).  Teardown RPCs are queries, not
        # ticks: destroy_actor / get_actors / apply_settings should each
        # complete in <1s on a healthy server.  A 10s deadline here lets
        # the known-stuck CARLA teardown paths (those Fix 3-4 don't cover)
        # fail fast instead of blocking select() for minutes.  We don't
        # restore — by the time _teardown_barrier returns the process is
        # heading for exit anyway.
        try:
            client_for_timeout = getattr(self, "client", None)
            if client_for_timeout is not None:
                try:
                    teardown_rpc_timeout_s = float(
                        os.environ.get("LEADERBOARD_TEARDOWN_RPC_TIMEOUT_S", "10")
                    )
                except Exception:
                    teardown_rpc_timeout_s = 10.0
                if teardown_rpc_timeout_s > 0:
                    client_for_timeout.set_timeout(teardown_rpc_timeout_s)
                    print(
                        f"[TEARDOWN] carla.Client.set_timeout({teardown_rpc_timeout_s:.1f}s) "
                        f"for stage={stage}",
                        flush=True,
                    )
        except Exception:
            # If the client is gone or set_timeout fails, just continue —
            # the post-results watchdog is the second-line safety net.
            pass

        try:
            # Keep simulator async during teardown to avoid blocking shutdown calls.
            if self.manager and self.manager.get_running_status() \
                    and hasattr(self, 'world') and self.world:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
                self.traffic_manager.set_synchronous_mode(False)

            # 1) stop/detach sensor listeners first
            manager_agent = getattr(self.manager, "_agent", None) if self.manager else None
            if manager_agent is not None:
                try:
                    manager_agent.cleanup(finalize_capture=(not preserve_agent_instance))
                except Exception:
                    pass
                try:
                    self.manager._agent = None
                except Exception:
                    pass

            # 2) explicitly stop any world sensor stream before destroy RPC.
            sensor_stop_summary = self._stop_world_sensor_streams(stage=stage)

            # 3) destroy remaining sensor actors before other actor cleanup
            sensor_actor_ids = sensor_stop_summary.get("sensor_ids", []) or self._collect_world_sensor_actor_ids()
            sensor_destroy_summary = self._destroy_actor_ids_with_verification(
                sensor_actor_ids,
                reason=f"leaderboard_evaluator_sensor_cleanup_{stage}",
                stage=stage,
                stop_sensors=True,
            )

            # 4) destroy tracked scenario actors
            CarlaDataProvider.cleanup()

            # 5) destroy egos in a controlled, verified batch
            ego_destroy_summary = self._destroy_ego_vehicles(stage=stage)

            # 6) manager and watchdog teardown
            if self.manager:
                try:
                    self.manager.cleanup()
                except Exception:
                    pass

            if self._agent_watchdog._timer:
                self._agent_watchdog.stop()

            if hasattr(self, 'agent_instance') and self.agent_instance and not preserve_agent_instance:
                try:
                    self.agent_instance.destroy()
                except Exception:
                    pass
                self.agent_instance = None

            if hasattr(self, 'statistics_manager') and self.statistics_manager:
                for j in range(self.ego_vehicles_num):
                    self.statistics_manager[j].scenario = None

            counts_after = self._collect_teardown_counts()
            duration_s = max(0.0, time.monotonic() - start_monotonic)
            self._last_teardown_duration_s = duration_s
            self._last_teardown_completed_seq = seq
            self._emit_actor_snapshot(
                "teardown_end",
                stage=stage,
                seq=int(seq),
                duration_s=round(float(duration_s), 4),
            )
            log_carla_event(
                "EVALUATOR_TEARDOWN_END",
                process_name=EVALUATOR_PROCESS_NAME,
                stage=stage,
                seq=int(seq),
                duration_s=duration_s,
                sensor_stop_total=len(sensor_actor_ids),
                sensor_stop_failed=sensor_stop_summary.get("failed", 0),
                sensor_destroy_failed=len(sensor_destroy_summary.get("failed_ids", [])),
                sensor_destroy_remaining=len(sensor_destroy_summary.get("remaining_alive_ids", [])),
                ego_destroy_failed=len(ego_destroy_summary.get("failed_ids", [])),
                ego_destroy_remaining=len(ego_destroy_summary.get("remaining_alive_ids", [])),
                tracked_sensors=counts_after["tracked_sensors"],
                tracked_actor_pool=counts_after["tracked_actor_pool"],
                world_sensor_actors=counts_after["world_sensor_actors"],
                ego_refs=counts_after["ego_refs"],
            )
        finally:
            self._teardown_in_progress = False

    def _dispose_client_session(self, *, stage):
        has_client = hasattr(self, "client") and self.client is not None
        has_world = hasattr(self, "world") and self.world is not None
        log_carla_event(
            "EVALUATOR_CLIENT_CLOSE_BEGIN",
            process_name=EVALUATOR_PROCESS_NAME,
            stage=stage,
            has_client=int(bool(has_client)),
            has_world=int(bool(has_world)),
        )
        try:
            if hasattr(self, "traffic_manager"):
                self.traffic_manager = None
            if hasattr(self, "world"):
                self.world = None
            if hasattr(self, "client"):
                self.client = None
            gc.collect()
        finally:
            log_carla_event(
                "EVALUATOR_CLIENT_CLOSE_END",
                process_name=EVALUATOR_PROCESS_NAME,
                stage=stage,
            )

    def _signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        if self._agent_watchdog and not self._agent_watchdog.get_status():
            raise RuntimeError("Timeout: Agent took too long to setup")
        elif self.manager:
            self.manager.signal_handler(signum, frame)

    def __del__(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """

        self._cleanup()
        self._dispose_client_session(stage="destructor")
        try:
            if hasattr(self, "recovery_manager") and self.recovery_manager is not None:
                self.recovery_manager.close()
        except Exception:
            pass
        if hasattr(self, 'manager') and self.manager:
            del self.manager
        if hasattr(self, 'world') and self.world:
            del self.world

    def _cleanup(self, *, preserve_agent_instance=False):
        """
        Remove and destroy all actors
        """
        # Defensive: if a SIGINT or other early failure invokes _cleanup
        # before __init__ finishes initialising teardown state, the
        # AttributeError compounds the failure into no_route_execution
        # in the wrapper's classifier.  Fall back to safe defaults instead.
        log_carla_event(
            "EVALUATOR_CLEANUP_BEGIN",
            process_name=EVALUATOR_PROCESS_NAME,
            ego_vehicle_count=len(getattr(self, "ego_vehicles", []) or []),
            teardown_seq=int(getattr(self, "_teardown_seq", 0) or 0),
            teardown_in_progress=int(bool(getattr(self, "_teardown_in_progress", False))),
        )
        self._teardown_barrier(stage="cleanup", preserve_agent_instance=preserve_agent_instance)
        log_carla_event(
            "EVALUATOR_CLEANUP_END",
            process_name=EVALUATOR_PROCESS_NAME,
            teardown_seq=int(self._last_teardown_completed_seq),
            teardown_duration_s=self._last_teardown_duration_s,
        )

    def _prepare_ego_vehicles(self, ego_vehicles, wait_for_ego_vehicles=False):
        """
        Spawn or update the ego vehicles
        """

        if not wait_for_ego_vehicles:
            for vehicle in ego_vehicles:
                self.ego_vehicles.append(CarlaDataProvider.request_new_actor(vehicle.model,
                                                                             vehicle.transform,
                                                                             vehicle.rolename,
                                                                             color=vehicle.color,
                                                                             vehicle_category=vehicle.category))

        else:
            ego_vehicle_missing = True
            while ego_vehicle_missing:
                self.ego_vehicles = []
                ego_vehicle_missing = False
                for ego_vehicle in ego_vehicles:
                    ego_vehicle_found = False
                    carla_vehicles = CarlaDataProvider.get_world().get_actors().filter('vehicle.*')
                    for carla_vehicle in carla_vehicles:
                        if carla_vehicle.attributes['role_name'] == ego_vehicle.rolename:
                            ego_vehicle_found = True
                            self.ego_vehicles.append(carla_vehicle)
                            break
                    if not ego_vehicle_found:
                        ego_vehicle_missing = True
                        break

            for i, _ in enumerate(self.ego_vehicles):
                self.ego_vehicles[i].set_transform(ego_vehicles[i].transform)

        # sync state
        CarlaDataProvider.get_world().tick()

    def _load_and_wait_for_world(self, args, town, ego_vehicles=None):
        """
        Load a new CARLA world and provide data to CarlaDataProvider
        """
        log_carla_event(
            "EVALUATOR_WORLD_LOAD_BEGIN",
            process_name=EVALUATOR_PROCESS_NAME,
            requested_town=town,
            tm_port=int(args.trafficManagerPort),
            provider_seed=int(args.carlaProviderSeed),
            tm_seed=int(args.trafficManagerSeed),
        )

        self.world = self.client.load_world(town)
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1.0 / self.frame_rate
        settings.synchronous_mode = True
        settings.deterministic_ragdolls = True
        self.world.apply_settings(settings)

        self.world.reset_all_traffic_lights()

        # self.world.set_pedestrians_seed(int(args.trafficManagerSeed))

        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(int(args.trafficManagerPort))
        CarlaDataProvider.set_random_seed(int(args.carlaProviderSeed))

        np.random.seed(int(args.carlaProviderSeed))
        random.seed(int(args.carlaProviderSeed))
        torch.manual_seed(int(args.carlaProviderSeed))
        torch.cuda.manual_seed_all(int(args.carlaProviderSeed))

        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(int(args.trafficManagerSeed))

        # Wait for the world to be ready
        if CarlaDataProvider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # CARLA 9.12 compatibility: map name might include full path like "/Game/Carla/Maps/Town05"
        current_map_name = CarlaDataProvider.get_map().name
        if town not in current_map_name:
            if os.environ.get('DEBUG_FINISH', '').lower() in ('1', 'true', 'yes'):
                print(f"[DEBUG FINISH] Current map: {current_map_name}, Required: {town}")
            raise Exception("The CARLA server uses the wrong map!"
                            "This scenario requires to use map {}".format(town))
        log_carla_event(
            "EVALUATOR_WORLD_LOAD_READY",
            process_name=EVALUATOR_PROCESS_NAME,
            requested_town=town,
            loaded_map=current_map_name,
            sync_mode=int(bool(settings.synchronous_mode)),
            fixed_delta_s=settings.fixed_delta_seconds,
        )

    def _reconnect_carla_client(self, args, *, stage):
        self.client = create_logged_client(
            carla,
            args.host,
            int(args.port),
            timeout_s=self.client_timeout,
            context=f"leaderboard_reconnect_{stage}",
            process_name=EVALUATOR_PROCESS_NAME,
        )
        self.traffic_manager = self.client.get_trafficmanager(int(args.trafficManagerPort))
        log_carla_event(
            "EVALUATOR_CLIENT_RECONNECTED",
            process_name=EVALUATOR_PROCESS_NAME,
            stage=stage,
            host=args.host,
            port=int(args.port),
            tm_port=int(args.trafficManagerPort),
        )

    def _apply_route_light_overrides(self, scenario_parameter):
        background = {}
        if isinstance(scenario_parameter, dict):
            background = scenario_parameter.get('Background', {}) or {}
        if background.get('turn_off_light', False):
            print("[INFO] Applying turn_off_light override: forcing all traffic lights to green/frozen.")
            [
                tf.set_state(carla.libcarla.TrafficLightState.Green)
                for tf in self.world.get_actors().filter("*traffic_light*")
                if hasattr(tf, "set_state")
            ]
            [
                tf.freeze(True)
                for tf in self.world.get_actors().filter("*traffic_light*")
                if hasattr(tf, "freeze")
            ]

    def _initialize_route_runtime(self, args, config, scenario_parameter, log_dir, *, recreate_manager=False):
        scenario_parameter = scenario_parameter if isinstance(scenario_parameter, dict) else {}
        if recreate_manager or self.manager is None:
            self.manager = ScenarioManager(args.timeout, args.debug > 1)

        self._load_and_wait_for_world(args, config.town, config.ego_vehicles)
        if self._route_plots_only:
            print("\033[1m> Route-plots-only mode: skipping ego actor spawn\033[0m")
        else:
            self._prepare_ego_vehicles(config.ego_vehicles, False)

        self._apply_route_light_overrides(scenario_parameter)
        scenario = RouteScenario(
            world=self.world,
            config=config,
            debug_mode=args.debug,
            ego_vehicles_num=self.ego_vehicles_num,
            log_dir=log_dir,
            scenario_parameter=scenario_parameter,
            route_plots_only=self._route_plots_only,
        )
        config.trajectory = scenario.get_new_config_trajectory()
        runtime_ego_count = self._capture_route_runtime_ego_mapping(scenario)
        if runtime_ego_count < int(self.ego_vehicles_num):
            skipped = self._current_route_spawn_metadata.get("skipped_ego_indices", [])
            print(
                "[WARN] Proceeding with partial ego spawn during evaluation: "
                f"runtime_egos={runtime_ego_count}/{int(self.ego_vehicles_num)} "
                f"skipped_original_egos={skipped}"
            )

        if not self._route_plots_only:
            if runtime_ego_count != 1:
                for active_idx, original_idx in enumerate(
                    self._current_route_active_to_original_ego_index
                ):
                    self.statistics_manager[int(original_idx)].set_scenario(
                        scenario.scenario[active_idx]
                    )
            else:
                original_idx = (
                    self._current_route_active_to_original_ego_index[0]
                    if self._current_route_active_to_original_ego_index
                    else 0
                )
                scenario_obj = (
                    scenario.scenario[0]
                    if isinstance(scenario.scenario, list)
                    else scenario.scenario
                )
                self.statistics_manager[int(original_idx)].set_scenario(scenario_obj)

            if config.weather.sun_altitude_angle < 0.0:
                for vehicle in scenario.ego_vehicles:
                    vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))

            self.manager.load_scenario(
                scenario,
                self.agent_instance,
                config.repetition_index,
                runtime_ego_count,
                save_root=config.save_path_root,
                sensor_tf_list=scenario.get_sensor_tf(),
                is_crazy=(scenario_parameter.get('Background', {}) or {}).get('turn_off_light', False),
            )

        return scenario

    def _rebuild_route_runtime_for_recovery(self, *, args, config, scenario_parameter, log_dir):
        try:
            self._cleanup(preserve_agent_instance=True)
        except Exception:
            traceback.print_exc()
        self._dispose_client_session(stage="recovery_pre_reconnect")
        self._reconnect_carla_client(args, stage="recovery")
        scenario = self._initialize_route_runtime(
            args,
            config,
            scenario_parameter,
            log_dir,
            recreate_manager=True,
        )
        if args.record:
            try:
                record_name = "{}_rep{}_recovery{}.log".format(
                    config.name,
                    config.repetition_index,
                    int(getattr(self.recovery_manager, "crash_generation", 0)),
                )
                self.client.start_recorder("{}/{}".format(args.record, record_name))
            except Exception:
                traceback.print_exc()
        return scenario

    # ── Post-valid-result hard-exit watchdog ────────────────────────────────
    # Once per-ego results.json files are committed to disk we have everything
    # we care about for this scenario. CARLA 0.9.12 has known RPC paths that
    # can wedge teardown indefinitely (Fix 3 covers one; others remain), and
    # without this watchdog a stuck destroy_actor / world.get_actors() call
    # in `_teardown_barrier` blocks the subprocess for the whole 1500s
    # `timeout` wrapper or until the parent-side POOL_STUCK_KILLER fires
    # (default 600s).  This watchdog caps that wasted wait at ~90s post-save.
    #
    # Cancel hooks: if a NEW scenario starts (`_load_and_run_scenario` runs
    # again), we cancel the previous arming so multi-scenario runs are not
    # killed mid-route. For the LAST scenario (no `peek()` follow-up) the
    # watchdog is what actually unblocks the process exit.
    def _arm_post_results_watchdog(self):
        try:
            budget_s = float(os.environ.get("LEADERBOARD_POST_RESULTS_KILL_S", "90"))
        except Exception:
            budget_s = 90.0
        if budget_s <= 0:
            return  # disabled
        self._disarm_post_results_watchdog()
        cancel = threading.Event()
        def _wait_and_kill():
            if cancel.wait(budget_s):
                return  # disarmed before timeout — normal teardown completed
            print(
                f"[POST_RESULTS_KILL] hard-exiting after {budget_s:.0f}s of "
                f"post-result teardown — per-ego results.json already on disk",
                flush=True,
            )
            os._exit(0)
        t = threading.Thread(target=_wait_and_kill, name="post_results_kill", daemon=True)
        self._post_results_kill_cancel = cancel
        self._post_results_kill_thread = t
        t.start()
        print(
            f"[POST_RESULTS_KILL] armed (budget={budget_s:.0f}s); cancel on next "
            f"scenario start, fire if teardown stalls past budget",
            flush=True,
        )

    def _disarm_post_results_watchdog(self):
        cancel = getattr(self, "_post_results_kill_cancel", None)
        if cancel is not None:
            cancel.set()
        self._post_results_kill_cancel = None
        self._post_results_kill_thread = None

    def _register_statistics(self, config, ego_car_num, checkpoint, entry_status, crash_message="",):
        """
        Computes and saved the simulation statistics
        """
        # register statistics

        current_stats_record = []
        current_stats_record.extend([[] for _ in range(0,ego_car_num)])
        spawn_meta = dict(self._current_route_spawn_metadata or {})
        original_to_active = dict(self._current_route_original_to_active_ego_index or {})
        runtime_ego_count = int(
            spawn_meta.get("runtime_ego_count", self._current_route_runtime_ego_count)
            or self._current_route_runtime_ego_count
            or ego_car_num
        )
        spawn_failures_by_original = {}
        for failure in spawn_meta.get("ego_spawn_failures", []) or []:
            try:
                original_idx = int(failure.get("original_ego_index"))
            except Exception:
                continue
            spawn_failures_by_original[original_idx] = dict(failure)
        for i in range(ego_car_num):
            active_idx = original_to_active.get(i)
            route_failure = crash_message
            if active_idx is None and bool(spawn_meta.get("partial_spawn_accepted", False)):
                route_failure = "Ego spawn skipped"
            current_stats_record[i] = self.statistics_manager[i].compute_route_statistics(
                config,
                self.manager.scenario_duration_system,
                self.manager.scenario_duration_game,
                route_failure,
                pdm_trace=(
                    self.manager.pdm_traces[active_idx]
                    if hasattr(self.manager, "pdm_traces")
                    and isinstance(self.manager.pdm_traces, (list, tuple))
                    and active_idx is not None
                    and active_idx < len(self.manager.pdm_traces)
                    else None
                ),
                pdm_world_trace=getattr(self.manager, "pdm_world_trace", None),
                pdm_tl_polygons=getattr(self.manager, "pdm_tl_polygons", None),
            )
            recovery_meta = dict(self._current_route_recovery_metadata or {})
            if recovery_meta:
                if recovery_meta.get("route_recovered", False):
                    recovery_meta.setdefault("recovered_approximate", True)
                if not isinstance(current_stats_record[i].meta, dict):
                    current_stats_record[i].meta = {}
                current_stats_record[i].meta.update(recovery_meta)
            if not isinstance(current_stats_record[i].meta, dict):
                current_stats_record[i].meta = {}
            if spawn_meta:
                current_stats_record[i].meta["requested_ego_count"] = int(
                    spawn_meta.get("requested_ego_count", ego_car_num) or ego_car_num
                )
                current_stats_record[i].meta["runtime_ego_count"] = runtime_ego_count
                current_stats_record[i].meta["skipped_ego_indices"] = list(
                    spawn_meta.get("skipped_ego_indices", []) or []
                )
                current_stats_record[i].meta["active_to_original_ego_index"] = list(
                    spawn_meta.get("active_to_original_ego_index", []) or []
                )
                current_stats_record[i].meta["original_to_active_ego_index"] = dict(
                    spawn_meta.get("original_to_active_ego_index", {}) or {}
                )
                current_stats_record[i].meta["partial_spawn_accepted"] = bool(
                    spawn_meta.get("partial_spawn_accepted", False)
                )
                if active_idx is None:
                    current_stats_record[i].meta["ego_spawn_status"] = "skipped"
                    failure_detail = spawn_failures_by_original.get(i)
                    if failure_detail is not None:
                        current_stats_record[i].meta["ego_spawn_failure"] = failure_detail
                else:
                    current_stats_record[i].meta["ego_spawn_status"] = "spawned"
                    current_stats_record[i].meta["runtime_ego_index"] = int(active_idx)
                    current_stats_record[i].meta["original_ego_index"] = int(i)
            route_status = str(getattr(current_stats_record[i], "status", "") or "")
            lower_status = route_status.lower()
            runtime_markers = (
                "crash",
                "sensor",
                "runtime",
                "recovery",
                "simulator",
                "timeout",
                "couldn't be set up",
                "invalid",
            )
            is_runtime_fail = any(marker in lower_status for marker in runtime_markers)
            if is_runtime_fail:
                terminal_state = (
                    "failed_recovery"
                    if bool(recovery_meta.get("recovery_failed", False))
                    else "failed_runtime"
                )
            elif bool(recovery_meta.get("route_recovered", False)):
                terminal_state = "completed_recovered"
            else:
                terminal_state = "completed_clean"
            current_stats_record[i].meta["route_terminal_status"] = terminal_state
            # Tier 1/2 metadata: which exit path triggered "done" for this ego.
            # Values: "route_complete" (RC=100% or RouteCompletionTest SUCCESS),
            # "blocked" (AgentBlockedTest fired), "soft_complete" (Tier 2:
            # near-goal stop detected before blocked-timer fires), or absent if
            # the scenario terminated for another reason (timeout, signal, etc.).
            try:
                _path_dict = getattr(self.manager, "_termination_path", None) or {}
                _path = _path_dict.get(int(i))
                if _path:
                    current_stats_record[i].meta["termination_path"] = str(_path)
                _soft_seen = getattr(self.manager, "_soft_complete_first_seen", None) or {}
                if int(i) in _soft_seen:
                    current_stats_record[i].meta["soft_complete_first_seen_sim_s"] = float(
                        _soft_seen[int(i)]
                    )
                if int(i) in (getattr(self.manager, "_soft_complete_done", None) or set()):
                    current_stats_record[i].meta["soft_complete_done"] = True
            except Exception:
                pass

            print("\033[1m> Registering the route statistics\033[0m")
            path_tmp = os.path.join(os.path.dirname(checkpoint), "ego_vehicle_{}".format(i), os.path.basename(checkpoint))
            folder_path = os.path.join(os.path.dirname(checkpoint), "ego_vehicle_{}".format(i))
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            self.statistics_manager[i].save_record(current_stats_record[i], config.index, path_tmp)
            self.statistics_manager[i].save_entry_status(entry_status, False, path_tmp)
            if hasattr(self.recovery_manager, "record_route_outcome"):
                try:
                    self.recovery_manager.record_route_outcome(
                        ego_index=int(i),
                        route_status=route_status,
                        route_meta=dict(current_stats_record[i].meta or {}),
                    )
                except Exception:
                    pass

    def _load_and_run_scenario(self, args, config):
        """
        Load and run the scenario given by config.

        Depending on what code fails, the simulation will either stop the route and
        continue from the next one, or report a crash and stop.

        Args:
            args: argparse.Namespace, global config
            config: srunner.scenarioconfigs.route_scenario_configuration.RouteScenarioConfiguration, config for route scenarios

        """
        # New scenario starting → cancel any post-results watchdog left over
        # from the previous scenario's teardown (it completed, so we don't
        # need the safety-net kill).
        self._disarm_post_results_watchdog()

        crash_message = ""
        entry_status = "Started"
        scenario = None
        scenario_loaded = False
        skip_route_execution = False
        recorder_started = False
        should_register_stats = False
        fatal_exit_code = None
        early_return = False
        log_file_dir = None
        original_stdout = sys.stdout
        logger = None
        _phase_times = {}   # phase_name -> elapsed seconds; printed in finally block
        scenario_parameter = {}
        self._current_route_recovery_metadata = {}
        self._reset_route_runtime_ego_mapping()

        self._wait_for_teardown_ready(stage="pre_route_start", timeout_s=10.0)

        print("\n\033[1m========= Preparing {} (repetition {}) =========".format(config.name, config.repetition_index))
        print("> Setting up the agent\033[0m")
        _t_agent_setup = time.time()
        log_carla_event(
            "ROUTE_PREPARE_BEGIN",
            process_name=EVALUATOR_PROCESS_NAME,
            route_name=config.name,
            route_index=config.index,
            repetition=config.repetition_index,
            town=getattr(config, "town", ""),
        )

        # Prepare the statistics of the route
        for j in range(self.ego_vehicles_num):
            self.statistics_manager[j].set_route(config.name, config.index)

        # Hard teardown gate: never begin a new route until prior teardown is fully quiesced.
        gate_counts_before = self._collect_teardown_counts()
        self._emit_actor_snapshot(
            "route_start_gate_begin",
            route_name=config.name,
            route_index=config.index,
            repetition=config.repetition_index,
        )
        log_carla_event(
            "ROUTE_START_GATE_BEGIN",
            process_name=EVALUATOR_PROCESS_NAME,
            route_name=config.name,
            route_index=config.index,
            repetition=config.repetition_index,
            tracked_sensors=gate_counts_before["tracked_sensors"],
            tracked_actor_pool=gate_counts_before["tracked_actor_pool"],
            world_sensor_actors=gate_counts_before["world_sensor_actors"],
            ego_refs=gate_counts_before["ego_refs"],
        )
        self._teardown_barrier(stage="route_start_gate")
        gate_counts_after = self._collect_teardown_counts()
        self._emit_actor_snapshot(
            "route_start_gate_end",
            route_name=config.name,
            route_index=config.index,
            repetition=config.repetition_index,
            teardown_seq=int(self._last_teardown_completed_seq),
        )
        log_carla_event(
            "ROUTE_START_GATE_END",
            process_name=EVALUATOR_PROCESS_NAME,
            route_name=config.name,
            route_index=config.index,
            repetition=config.repetition_index,
            tracked_sensors=gate_counts_after["tracked_sensors"],
            tracked_actor_pool=gate_counts_after["tracked_actor_pool"],
            world_sensor_actors=gate_counts_after["world_sensor_actors"],
            ego_refs=gate_counts_after["ego_refs"],
            teardown_seq=int(self._last_teardown_completed_seq),
        )

        try:
            if self._route_plots_only:
                class _RoutePlotsOnlyAgent:
                    track = "SENSORS"

                    def set_global_plan(self, *_args, **_kwargs):
                        return

                    def sensors(self):
                        return []

                    def destroy(self):
                        return

                self.agent_instance = _RoutePlotsOnlyAgent()
                config.agent = self.agent_instance
                config.save_path_root = None
                log_root_dir = os.path.dirname(os.environ["CHECKPOINT_ENDPOINT"])
            else:
                # Set up the user's agent, and the timer self._agent_watchdog to avoid freezing the simulation
                try:
                    self._agent_watchdog.start()
                    # agent_class_name for example 'AutoPilot', 'PnP_Agent' .etc
                    agent_class_name = getattr(self.module_agent, 'get_entry_point')()
                    self.agent_instance = getattr(self.module_agent, agent_class_name)(
                        args.agent_config,
                        self.ego_vehicles_num,
                    )
                    config.agent = self.agent_instance
                    if hasattr(self.agent_instance, "get_save_path"):
                        print("Data Generation Confirmed!")
                        config.save_path_root = self.agent_instance.get_save_path()
                        log_root_dir = config.save_path_root
                    else:
                        print("Evaluation Process!")
                        config.save_path_root = None
                        log_root_dir = os.path.dirname(os.environ["CHECKPOINT_ENDPOINT"])
                        try:
                            log_root_dir = self.agent_instance.get_save_path()
                        except Exception:
                            print('load save path failed')

                    # save source code
                    backup_script(os.environ["RESULT_ROOT"])

                    # Check and store the sensors
                    if not self.sensors:
                        self.sensors = self.agent_instance.sensors()
                        track = self.agent_instance.track
                        AgentWrapper.validate_sensor_configuration(self.sensors, track, args.track)
                        self.sensor_icons = [sensors_to_icons[sensor['type']] for sensor in self.sensors]
                        for j in range(self.ego_vehicles_num):
                            self.statistics_manager[j].save_sensors(self.sensor_icons, args.checkpoint)
                    self._agent_watchdog.stop()

                except SensorConfigurationInvalid as e:
                    # The sensors are invalid -> set the execution to rejected and stop
                    print("\n\033[91mThe sensor's configuration used is invalid:")
                    print("> {}\033[0m\n".format(e))
                    traceback.print_exc()
                    _append_traceback_to_log(log_file_dir)
                    crash_message = "Agent's sensors were invalid"
                    entry_status = "Rejected"
                    fatal_exit_code = -1
                    log_carla_event(
                        "ROUTE_AGENT_SENSOR_INVALID",
                        process_name=EVALUATOR_PROCESS_NAME,
                        route_name=config.name,
                        route_index=config.index,
                        repetition=config.repetition_index,
                        error=str(e),
                    )

                except Exception as e:
                    # The agent setup has failed -> start the next route
                    print("\n\033[91mCould not set up the required agent:")
                    print("> {}\033[0m\n".format(e))
                    traceback.print_exc()
                    _append_traceback_to_log(log_file_dir)
                    crash_message = "Agent couldn't be set up"
                    early_return = True
                    log_carla_event(
                        "ROUTE_AGENT_SETUP_FAIL",
                        process_name=EVALUATOR_PROCESS_NAME,
                        route_name=config.name,
                        route_index=config.index,
                        repetition=config.repetition_index,
                        error_type=type(e).__name__,
                        error=str(e),
                    )

            if not early_return and fatal_exit_code is None:
                _phase_times['agent_setup'] = time.time() - _t_agent_setup
                print(f"[Timing] agent_setup={_phase_times['agent_setup']:.2f}s")
                # Log simulation information(or error), every printed string will be logged in file log.log
                log_dir = os.path.join(log_root_dir, 'log')
                self.log_dir = log_dir
                if not os.path.exists(log_dir):
                    os.mkdir(log_dir)
                log_file_dir = os.path.join(log_dir, 'log.log')
                self.log_file_dir = log_file_dir
                logger = Logger(log_file_dir, stream=original_stdout)
                sys.stdout = logger
                args_file_dir = os.path.join(log_dir, 'args.json')
                args_dict = vars(args)
                json_str = json.dumps(args_dict, indent=2)
                with open(args_file_dir, 'w') as json_file:
                    json_file.write(json_str)
                print("{} (repetition {}) Log file initialized!".format(config.name, config.repetition_index))

                # Load the world and the scenario
                print("\033[1m> Loading the world\033[0m")
                try:
                    with open(args.scenario_parameter, 'r', encoding='utf-8') as f:
                        scenario_parameter = yaml.load(f.read(), Loader=yaml.FullLoader)
                    _t_runtime = time.time()
                    scenario = self._initialize_route_runtime(
                        args,
                        config,
                        scenario_parameter,
                        log_dir,
                        recreate_manager=False,
                    )
                    scenario_loaded = True
                    _phase_times['route_runtime_init'] = time.time() - _t_runtime
                    print(f"[Timing] route_runtime_init={_phase_times['route_runtime_init']:.2f}s")
                    log_carla_event(
                        "ROUTE_SCENARIO_LOADED",
                        process_name=EVALUATOR_PROCESS_NAME,
                        route_name=config.name,
                        route_index=config.index,
                        repetition=config.repetition_index,
                        town=getattr(config, "town", ""),
                    )

                    if getattr(args, "route_plots_only", False):
                        print("\033[1m> Route-plots-only mode: skipping scenario execution\033[0m")
                        skip_route_execution = True
                    else:
                        if args.record:
                            self.client.start_recorder("{}/{}_rep{}.log".format(args.record, config.name, config.repetition_index))
                            recorder_started = True
                        should_register_stats = True
                        if self.recovery_manager is not None:
                            self.recovery_manager.start_route(config=config)
                            self.recovery_manager.bind_runtime(manager=self.manager, scenario=scenario)

                except Exception as e:
                    # The scenario is wrong -> set the execution to crashed and stop
                    print("\n\033[91mThe scenario could not be loaded:")
                    print("> {}\033[0m\n".format(e))
                    traceback.print_exc()
                    _append_traceback_to_log(log_file_dir)
                    crash_message = "Simulation crashed"
                    entry_status = "Crashed"
                    should_register_stats = not self._route_plots_only
                    fatal_exit_code = -1
                    log_carla_event(
                        "ROUTE_SCENARIO_LOAD_FAIL",
                        process_name=EVALUATOR_PROCESS_NAME,
                        route_name=config.name,
                        route_index=config.index,
                        repetition=config.repetition_index,
                        error_type=type(e).__name__,
                        error=str(e),
                    )

                if scenario_loaded and not skip_route_execution and fatal_exit_code is None:
                    print("\033[1m> Running the route\033[0m")
                    log_carla_event(
                        "ROUTE_RUN_BEGIN",
                        process_name=EVALUATOR_PROCESS_NAME,
                        route_name=config.name,
                        route_index=config.index,
                        repetition=config.repetition_index,
                    )
                    _t_run = time.time()
                    scenario_completed = False
                    while not scenario_completed:
                        try:
                            if self.recovery_manager is not None:
                                self.recovery_manager.bind_runtime(manager=self.manager, scenario=scenario)
                            self.manager.run_scenario()
                            scenario_completed = True
                            recovered_flag = False
                            crash_generation = 0
                            if self.recovery_manager is not None:
                                recovered_flag = bool(self.recovery_manager.route_recovered)
                                crash_generation = int(self.recovery_manager.crash_generation)
                            log_carla_event(
                                "ROUTE_RUN_END",
                                process_name=EVALUATOR_PROCESS_NAME,
                                route_name=config.name,
                                route_index=config.index,
                                repetition=config.repetition_index,
                                status="ok",
                                recovered=int(recovered_flag),
                                crash_generation=int(crash_generation),
                            )
                        except AgentError as e:
                            # The agent has failed -> stop the route
                            print("\n\033[91mStopping the route, the agent has crashed:")
                            print("> {}\033[0m\n".format(e))
                            traceback.print_exc()
                            _append_traceback_to_log(log_file_dir)
                            crash_message = "Agent crashed"
                            scenario_completed = True
                            log_carla_event(
                                "ROUTE_RUN_FAIL",
                                process_name=EVALUATOR_PROCESS_NAME,
                                route_name=config.name,
                                route_index=config.index,
                                repetition=config.repetition_index,
                                status="agent_error",
                                error=str(e),
                            )
                        except Exception as e:
                            recovered = False
                            rebuilt_scenario = None
                            if self.recovery_manager is not None:
                                recovered, rebuilt_scenario = self.recovery_manager.try_recover(
                                    exc=e,
                                    args=args,
                                    config=config,
                                    scenario_parameter=scenario_parameter,
                                    log_dir=log_dir,
                                )
                            if recovered:
                                scenario = rebuilt_scenario
                                scenario_loaded = True
                                continue
                            if self.recovery_manager is not None:
                                self._current_route_recovery_metadata = self.recovery_manager.route_metadata()
                            else:
                                self._current_route_recovery_metadata = {}
                            print("\n\033[91mError during the simulation:")
                            print("> {}\033[0m\n".format(e))
                            traceback.print_exc()
                            _append_traceback_to_log(log_file_dir)
                            if bool(self._current_route_recovery_metadata.get("recovery_failed", False)):
                                crash_message = "Recovery failed"
                                entry_status = "Failed - Recovery"
                            else:
                                crash_message = "Run scenario crashed"
                                entry_status = "Crashed"
                            scenario_completed = True
                            log_carla_event(
                                "ROUTE_RUN_FAIL",
                                process_name=EVALUATOR_PROCESS_NAME,
                                route_name=config.name,
                                route_index=config.index,
                                repetition=config.repetition_index,
                                status="exception",
                                error_type=type(e).__name__,
                                error=str(e),
                            )
                    should_register_stats = True
                    _phase_times['run_scenario'] = time.time() - _t_run
                    print(f"[Timing] run_scenario={_phase_times['run_scenario']:.2f}s")
                    if self.recovery_manager is not None:
                        self._current_route_recovery_metadata = self.recovery_manager.route_metadata()
                    else:
                        self._current_route_recovery_metadata = {}

        finally:
            if scenario_loaded and not skip_route_execution:
                try:
                    print("\033[1m> Stopping the route\033[0m")
                    self.manager.stop_scenario()
                    log_carla_event(
                        "ROUTE_STOP",
                        process_name=EVALUATOR_PROCESS_NAME,
                        route_name=config.name,
                        route_index=config.index,
                        repetition=config.repetition_index,
                        status="ok",
                    )
                except Exception as e:
                    print("\n\033[91mFailed to stop the scenario cleanly:")
                    print("> {}\033[0m\n".format(e))
                    traceback.print_exc()
                    _append_traceback_to_log(log_file_dir)
                    log_carla_event(
                        "ROUTE_STOP",
                        process_name=EVALUATOR_PROCESS_NAME,
                        route_name=config.name,
                        route_index=config.index,
                        repetition=config.repetition_index,
                        status="fail",
                        error_type=type(e).__name__,
                        error=str(e),
                    )

            if should_register_stats and not skip_route_execution and entry_status != "Rejected":
                try:
                    self._register_statistics(
                        config,
                        args.ego_num,
                        args.checkpoint,
                        entry_status,
                        crash_message,
                    )
                except Exception:
                    traceback.print_exc()
                    _append_traceback_to_log(log_file_dir)

            if recorder_started:
                try:
                    self.client.stop_recorder()
                except Exception:
                    traceback.print_exc()
                    _append_traceback_to_log(log_file_dir)

            # Arm the post-results hard-exit watchdog NOW, immediately before
            # scenario.remove_all_actors() and the post-scenario _cleanup().
            # All meaningful work for this route is done at this point:
            #   • per-ego results.json is on disk (if _register_statistics ran)
            #   • or the scenario crashed and there are no results to protect
            # Either way, anything beyond here is teardown — and CARLA RPC
            # paths in remove_all_actors() and _teardown_barrier can wedge
            # select() for minutes. The watchdog gives us a hard ceiling on
            # that wait.
            #
            # Placement chosen carefully:
            #   • _cleanup() is also called from the recovery path
            #     (_rebuild_route_runtime_for_recovery), where we DO want the
            #     RPCs to complete, not be cut short. By arming here, only at
            #     the per-route end-of-life teardown, we avoid arming during
            #     recovery.
            #   • Armed BEFORE remove_all_actors so the watchdog covers actor
            #     destroy too — basic_scenario.remove_all_actors has its own
            #     30s hard-bail (CARLA_REMOVE_ACTORS_HARD_TIMEOUT_S) for the
            #     batched cleanup itself; this 90s budget catches the wider
            #     teardown including _teardown_barrier.
            #   • The next _load_and_run_scenario() disarms at entry, so
            #     multi-scenario / --scenario-pool runs won't trip it.
            try:
                self._arm_post_results_watchdog()
            except Exception:
                traceback.print_exc()

            if scenario is not None:
                try:
                    scenario.remove_all_actors()
                except Exception:
                    traceback.print_exc()
                    _append_traceback_to_log(log_file_dir)

            try:
                self._cleanup()
            except Exception:
                traceback.print_exc()
                _append_traceback_to_log(log_file_dir)

            if logger is not None and sys.stdout is logger:
                # Print phase timing summary while stdout still goes to the log file
                if _phase_times:
                    _timing_lines = [f"[Evaluator phase timing summary] route={config.name} rep={config.repetition_index}"]
                    _ordered = ['agent_setup', 'world_load', 'ego_spawn', 'route_scenario_init',
                                'load_scenario_sensors', 'run_scenario']
                    for _k in _ordered:
                        if _k in _phase_times:
                            _timing_lines.append(f"  {_k}: {_phase_times[_k]:.2f}s")
                    for _k, _v in _phase_times.items():
                        if _k not in _ordered:
                            _timing_lines.append(f"  {_k}: {_v:.2f}s")
                    print('\n'.join(_timing_lines))
                sys.stdout = original_stdout
            elif sys.stdout is not original_stdout and logger is None:
                sys.stdout = original_stdout
            if logger is not None:
                try:
                    logger.flush()
                except Exception:
                    pass
                logger.close()

        if fatal_exit_code is not None:
            log_carla_event(
                "ROUTE_EXIT_FATAL",
                process_name=EVALUATOR_PROCESS_NAME,
                route_name=config.name,
                route_index=config.index,
                repetition=config.repetition_index,
                exit_code=int(fatal_exit_code),
                entry_status=entry_status,
                crash_message=crash_message,
            )
            sys.exit(fatal_exit_code)
        if early_return:
            log_carla_event(
                "ROUTE_EARLY_RETURN",
                process_name=EVALUATOR_PROCESS_NAME,
                route_name=config.name,
                route_index=config.index,
                repetition=config.repetition_index,
                entry_status=entry_status,
                crash_message=crash_message,
            )
            return
        log_carla_event(
            "ROUTE_COMPLETE",
            process_name=EVALUATOR_PROCESS_NAME,
            route_name=config.name,
            route_index=config.index,
            repetition=config.repetition_index,
            entry_status=entry_status,
            crash_message=crash_message,
            scenario_loaded=int(bool(scenario_loaded)),
            skip_route_execution=int(bool(skip_route_execution)),
            recovered=int(bool(self._current_route_recovery_metadata.get("route_recovered", False))),
            crash_generation=int(self._current_route_recovery_metadata.get("crash_generation", 0) or 0),
            partial_restore=int(bool(self._current_route_recovery_metadata.get("partial_restore", False))),
        )

    def run(self, args: argparse.Namespace):
        """
        Run the challenge mode
        """

        print(f'''run with {args.ego_num} cars\n''')
        log_carla_event(
            "ROUTE_INPUT_VALIDATION_BEGIN",
            process_name=EVALUATOR_PROCESS_NAME,
            routes_dir=getattr(args, "routes_dir", None),
            ego_num=int(args.ego_num),
            scenarios=getattr(args, "scenarios", None),
            repetitions=int(args.repetitions),
        )
        route_indexer_dict = {}
        route_path_dict = {}
        executed_route_count = 0

        if not args.routes_dir or not os.path.isdir(args.routes_dir):
            message = f"Routes directory does not exist: {args.routes_dir}"
            log_carla_event(
                "ROUTE_INPUT_VALIDATION_FAIL",
                process_name=EVALUATOR_PROCESS_NAME,
                stage="routes_dir",
                routes_dir=getattr(args, "routes_dir", None),
                ego_num=int(args.ego_num),
                error=message,
            )
            raise RuntimeError(message)
        
        # Search recursively for XML files in the routes directory
        # This handles nested scenario directories (e.g., routes/Scenario_Name/vehicle_*.xml)
        import glob
        xml_files = glob.glob(os.path.join(args.routes_dir, '**', '*.xml'), recursive=True)

        # Ignore custom actor XMLs (they live under actors/); only keep ego route files
        xml_files = [
            x for x in xml_files
            if "actors" not in os.path.normpath(x).split(os.sep)
        ]
        
        # Separate the dense REPLAY XMLs (used for open-loop ego teleport) from
        # the sparse route XMLs (used by the planner as its target route).
        #
        # Filename conventions used by the v2xpnp dataset:
        #   ucla_v2_custom_ego_vehicle_<N>.xml         -> sparse route (4-8 waypoints, no `time`)
        #   ucla_v2_custom_ego_vehicle_<N>_REPLAY.xml  -> dense replay (~123 waypoints, `time` per waypoint)
        #
        # Previously the suffix-stripping logic mapped "..._0_REPLAY.xml" to
        # ego_id_str="REPLAY" and the dense trajectory was silently filtered
        # out at the usable_keys step below. That broke the LogReplayEgo
        # follower (it only fires when `trajectory_times` is fully populated)
        # and made --openloop effectively closed-loop for the ego.
        replay_path_dict = {}  # ego_id_str -> path/to/<...>_REPLAY.xml
        for xml_path in xml_files:
            filename = os.path.basename(xml_path)
            stem = filename.split('.')[0]
            is_replay = stem.endswith("_REPLAY")
            base_stem = stem[: -len("_REPLAY")] if is_replay else stem
            ego_id_str = base_stem.split('_')[-1]

            if is_replay:
                if ego_id_str not in replay_path_dict:
                    replay_path_dict[ego_id_str] = xml_path
            else:
                if ego_id_str not in route_path_dict:
                    route_path_dict[ego_id_str] = xml_path

        if not route_path_dict:
            message = f"No route XML files found under {args.routes_dir}"
            log_carla_event(
                "ROUTE_INPUT_VALIDATION_FAIL",
                process_name=EVALUATOR_PROCESS_NAME,
                stage="route_xml_discovery",
                routes_dir=getattr(args, "routes_dir", None),
                ego_num=int(args.ego_num),
                error=message,
            )
            raise RuntimeError(message)

        route_indexer = None
        if args.ego_num > 0:
            # Build the ego_id -> XML mapping the scheduler will use.  Some
            # scenarios are authored with ego XMLs starting at index 1 (e.g.
            # the v2xpnp set, where the only ego file is "..._1.xml").  Ask
            # for ego_num=1 against such a scenario and the strict
            # ``str(0) not in route_path_dict`` check below would raise.
            # Instead, fall back to the lowest-numbered keys present and
            # rebind them to the contiguous ``range(args.ego_num)`` slot
            # space the rest of this function expects.
            def _ego_key_order(key: str):
                try:
                    return (0, int(key))
                except (TypeError, ValueError):
                    return (1, str(key))

            usable_keys = [k for k in route_path_dict.keys() if k != "REPLAY"]
            usable_keys.sort(key=_ego_key_order)
            need_remap = any(
                str(eid) not in route_path_dict for eid in range(args.ego_num)
            )
            if need_remap and len(usable_keys) >= args.ego_num:
                remapped_keys = usable_keys[: args.ego_num]
                if any(remapped_keys[i] != str(i) for i in range(args.ego_num)):
                    log_carla_event(
                        "ROUTE_INPUT_VALIDATION_REMAP",
                        process_name=EVALUATOR_PROCESS_NAME,
                        stage="ego_route_mapping",
                        routes_dir=getattr(args, "routes_dir", None),
                        ego_num=int(args.ego_num),
                        discovered_keys=sorted(route_path_dict.keys()),
                        remapped_keys=remapped_keys,
                    )
                    print(
                        "[INFO] Remapping ego route XML keys "
                        f"{remapped_keys} -> {[str(i) for i in range(args.ego_num)]} "
                        f"(discovered_keys={sorted(route_path_dict.keys())})"
                    )
                    route_path_dict = {
                        str(i): route_path_dict[remapped_keys[i]]
                        for i in range(args.ego_num)
                    }
                    # Mirror the same remap onto replay_path_dict so the
                    # openloop teleport path can find the dense REPLAY XML by
                    # the new (contiguous) ego_id keys. Without this, v2xpnp
                    # scenarios where the only ego file is "..._1.xml" would
                    # have route_path_dict['0'] (after remap) but
                    # replay_path_dict['1'] (no remap), causing
                    # `replay_path_dict.get(str(0))` at the multi_replay_*
                    # build site to return None and OpenLoopRuntime to never
                    # construct → openloop runs would silently degrade to
                    # closed-loop. Closed-loop runs are unaffected because
                    # replay_path_dict is only consumed in the openloop branch.
                    replay_remapped = {
                        str(i): replay_path_dict[remapped_keys[i]]
                        for i in range(args.ego_num)
                        if remapped_keys[i] in replay_path_dict
                    }
                    if replay_remapped:
                        replay_path_dict = replay_remapped
            for ego_id in range(args.ego_num):
                if str(ego_id) not in route_path_dict:
                    message = (
                        f"Missing ego route XML for ego_id={ego_id}. "
                        f"Discovered keys: {sorted(route_path_dict.keys())}"
                    )
                    log_carla_event(
                        "ROUTE_INPUT_VALIDATION_FAIL",
                        process_name=EVALUATOR_PROCESS_NAME,
                        stage="ego_route_mapping",
                        routes_dir=getattr(args, "routes_dir", None),
                        ego_num=int(args.ego_num),
                        missing_ego_id=int(ego_id),
                        discovered_keys=sorted(route_path_dict.keys()),
                        error=message,
                    )
                    raise RuntimeError(message)
                route_indexer_dict[ego_id] = RouteIndexer(
                    route_path_dict[str(ego_id)], args.scenarios, args.repetitions
                )
                if ego_id == 0:
                    route_indexer = route_indexer_dict[ego_id]
        else:
            # No-ego mode still needs a primary route indexer to iterate scenario configs.
            def _route_key_order(key: str):
                try:
                    return (0, int(key))
                except (TypeError, ValueError):
                    return (1, str(key))

            first_key = sorted(route_path_dict.keys(), key=_route_key_order)[0]
            route_indexer = RouteIndexer(
                route_path_dict[first_key], args.scenarios, args.repetitions
            )
        log_carla_event(
            "ROUTE_INPUT_VALIDATION_OK",
            process_name=EVALUATOR_PROCESS_NAME,
            routes_dir=getattr(args, "routes_dir", None),
            ego_num=int(args.ego_num),
            discovered_keys=sorted(route_path_dict.keys()),
            route_indexer_count=len(route_indexer_dict) if route_indexer_dict else 1,
            route_indexer_route=getattr(route_indexer, "route_file", None),
        )

        # if args.routes_0 is not None and args.routes_1 is not None and args.routes_2 is not None:
        #     route_indexer_0 = RouteIndexer(args.routes_0, args.scenarios, args.repetitions)
        #     route_indexer_1 = RouteIndexer(args.routes_1, args.scenarios, args.repetitions)
        #     route_indexer_2 = RouteIndexer(args.routes_2, args.scenarios, args.repetitions)
        #     route_indexer = route_indexer_0
        #     print('run with 3 cars\n')
        # elif args.routes_0 is not None and args.routes_1 is not None:
        #     route_indexer_0 = RouteIndexer(args.routes_0, args.scenarios, args.repetitions)
        #     route_indexer_1 = RouteIndexer(args.routes_1, args.scenarios, args.repetitions)
        #     route_indexer_2 = None
        #     route_indexer = route_indexer_0
        #     print('run with 2 cars\n')
        # else:
        #     route_indexer_0 = None
        #     route_indexer_1 = None
        #     route_indexer_2 = None
        #     route_indexer = RouteIndexer(args.routes, args.scenarios, args.repetitions)
        #     print('run with 1 car\n')

        if args.resume:
            route_indexer.resume(args.checkpoint)
            for i in range(self.ego_vehicles_num):
                self.statistics_manager[i].resume(args.checkpoint)
        else:
            for i in range(self.ego_vehicles_num):              
                self.statistics_manager[i].clear_record(args.checkpoint)
            route_indexer.save_state(args.checkpoint)

        # Parse a REPLAY XML once and return a single RouteScenarioConfiguration
        # exposing the dense trajectory + per-waypoint times. Returns None if the
        # path is missing or unparseable so callers can fall back gracefully.
        def _load_replay_config(replay_xml_path):
            if not replay_xml_path or not os.path.isfile(replay_xml_path):
                return None
            try:
                # parse_routes_file expects (routes_file, scenarios_file, ...)
                # — a non-empty scenarios_file is required even though we don't
                # consume scenario triggers from the REPLAY XML.
                replay_configs = RouteParser.parse_routes_file(
                    replay_xml_path, args.scenarios, False
                )
            except Exception as exc:
                print(f"[OPENLOOP] Failed to parse replay XML {replay_xml_path}: {exc}")
                return None
            if not replay_configs:
                return None
            return replay_configs[0]

        print("Start Running!")
        while route_indexer.peek():
            try:
                # setup, load config of the next route
                config = route_indexer.next()
                # ----- Open-loop replay attachments (all egos) -----
                # Always populate `multi_replay_*` even with ego_num == 1, so
                # downstream code (scenario_manager openloop teleport, viz, etc)
                # has a single source of truth regardless of ego count.
                config.multi_replay_traj = []
                config.multi_replay_yaws = []
                config.multi_replay_pitches = []
                config.multi_replay_rolls = []
                config.multi_replay_times = []
                config.multi_replay_xml_path = []
                for ego_id in range(max(1, int(args.ego_num))):
                    rxml = replay_path_dict.get(str(ego_id))
                    rconfig = _load_replay_config(rxml)
                    if rconfig is None:
                        config.multi_replay_traj.append(None)
                        config.multi_replay_yaws.append(None)
                        config.multi_replay_pitches.append(None)
                        config.multi_replay_rolls.append(None)
                        config.multi_replay_times.append(None)
                        config.multi_replay_xml_path.append(None)
                        continue
                    config.multi_replay_traj.append(getattr(rconfig, "trajectory", None))
                    config.multi_replay_yaws.append(getattr(rconfig, "trajectory_yaws", None))
                    config.multi_replay_pitches.append(getattr(rconfig, "trajectory_pitches", None))
                    config.multi_replay_rolls.append(getattr(rconfig, "trajectory_rolls", None))
                    config.multi_replay_times.append(getattr(rconfig, "trajectory_times", None))
                    config.multi_replay_xml_path.append(rxml)
                if args.ego_num > 1:
                    config.multi_traj = [config.trajectory]
                    config.multi_traj_yaws = [getattr(config, "trajectory_yaws", None)]
                    config.multi_traj_pitches = [getattr(config, "trajectory_pitches", None)]
                    config.multi_traj_rolls = [getattr(config, "trajectory_rolls", None)]
                    config.multi_traj_times = [getattr(config, "trajectory_times", None)]
                    for i in range(1,args.ego_num):
                        route_indexer_other = route_indexer_dict[i]
                        route_indexer_dict[i].peek()
                        config_other = route_indexer_dict[i].next()
                        config.multi_traj.append(config_other.trajectory)
                        config.multi_traj_yaws.append(getattr(config_other, "trajectory_yaws", None))
                        config.multi_traj_pitches.append(getattr(config_other, "trajectory_pitches", None))
                        config.multi_traj_rolls.append(getattr(config_other, "trajectory_rolls", None))
                        config.multi_traj_times.append(getattr(config_other, "trajectory_times", None))

                # if route_indexer_1 is not None:
                #     route_indexer_1.peek()
                #     config_1 = route_indexer_1.next()
                #     config.multi_traj = [config.trajectory, config_1.trajectory]
                # if route_indexer_2 is not None:
                #     route_indexer_2.peek()
                #     config_2 = route_indexer_2.next()
                #     config.multi_traj = [config.trajectory, config_1.trajectory, config_2.trajectory]

                # reinitialize random seed after each route
                np.random.seed(int(args.carlaProviderSeed))
                random.seed(int(args.carlaProviderSeed))
                torch.manual_seed(int(args.carlaProviderSeed))
                torch.cuda.manual_seed_all(int(args.carlaProviderSeed))
                # run
                self._load_and_run_scenario(args, config)

                for i in range(args.ego_num):
                    folder_path = os.path.join(os.path.dirname(args.checkpoint), "ego_vehicle_{}".format(i))
                    if not os.path.exists(folder_path):
                        os.mkdir(folder_path)
                    path_tmp = os.path.join(os.path.dirname(args.checkpoint), "ego_vehicle_{}".format(i), os.path.basename(args.checkpoint))
                    route_indexer.save_state(path_tmp)
                executed_route_count += 1

            except Exception as e:
                print('route error:',e)
                log_carla_event(
                    "ROUTE_EXECUTION_FAIL",
                    process_name=EVALUATOR_PROCESS_NAME,
                    error_type=type(e).__name__,
                    error=str(e),
                    routes_dir=getattr(args, "routes_dir", None),
                    route_name=getattr(config, "name", None),
                    route_index=getattr(config, "index", None),
                    repetition=getattr(config, "repetition_index", None),
                )
                raise

        if executed_route_count == 0:
            message = (
                "No routes were executed. "
                f"routes_dir={getattr(args, 'routes_dir', None)} "
                f"ego_num={int(args.ego_num)}"
            )
            log_carla_event(
                "ROUTE_NO_ROUTE_EXECUTED",
                process_name=EVALUATOR_PROCESS_NAME,
                routes_dir=getattr(args, "routes_dir", None),
                ego_num=int(args.ego_num),
                route_total=getattr(route_indexer, "total", None),
                checkpoint=getattr(args, "checkpoint", None),
                resume=int(bool(getattr(args, "resume", 0))),
            )
            raise RuntimeError(message)

        # save global statistics
        if args.route_plots_only:
            print("\033[1m> Route-plots-only mode: skipping global statistics export\033[0m")
            self._dispose_client_session(stage="run_route_plots_only_exit")
            return

        print("\033[1m> Registering the global statistics\033[0m")
        # TODO: save global records for every statistics manager.
        try:
            for i in range(self.ego_vehicles_num):
                global_stats_record = self.statistics_manager[i].compute_global_statistics(route_indexer.total)
                # print("------------ego_{}---------".format(i))
                # print(global_stats_record)
                path_tmp = os.path.join(os.path.dirname(args.checkpoint), "ego_vehicle_{}".format(i), os.path.basename(args.checkpoint)) 
                self.statistics_manager[i].save_global_record(global_stats_record, self.sensor_icons, route_indexer.total, path_tmp)
        except Exception as e:
            print('route error:',e)
            traceback.print_exc()
            _append_traceback_to_log(getattr(self, "log_file_dir", None))
            raise
        self._dispose_client_session(stage="run_complete")

def main():
    description = "CARLA AD Leaderboard Evaluation: evaluate your Agent in CARLA scenarios\n"

    # general parameters
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default='50010', help='TCP port to listen to (default: 2000)')
    parser.add_argument('--trafficManagerPort', default='50050',
                        help='Port to use for the TrafficManager (default: 8000)')
    parser.add_argument('--trafficManagerSeed', default='1',
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--carlaProviderSeed', default='2000',
                        help='Seed used by the CarlaProvider (default: 2000)')
    parser.add_argument('--debug', type=int, help='Run with debug output', default=0)
    parser.add_argument('--record', type=str, default='',
                        help='Use CARLA recording feature to create a recording of the scenario')
    parser.add_argument('--timeout', default="600.0",
                        help='Set the CARLA client timeout value in seconds')

    # simulation setup
    parser.add_argument('--routes',
                        default=None,
                        help='Name of the route to be executed. Point to the route_xml_file to be executed.')
    parser.add_argument('--routes_0',
                        default=None,
                        help='Name of the route to be executed by ego vehicle 0.')
    parser.add_argument('--routes_1',
                        default=None,
                        help='Name of the route to be executed by ego vehicle 1.')
    parser.add_argument('--routes_2',
                        default=None,
                        help='Name of the route to be executed by ego vehicle 2.')
    parser.add_argument('--routes_dir',
                        default=None,
                        help='the directory that contains all routes for multiple vehicles')
    parser.add_argument('--scenarios',
                        default='simulation/leaderboard/data/scenarios/town05_all_scenarios_2.json',
                        help='Name of the scenario annotation file to be mixed with the route.')
    parser.add_argument('--scenario_parameter', 
                        default='simulation/leaderboard/leaderboard/scenarios/scenario_parameter_demo_0.yaml',
                        help='Defination of the scenario parameters.')
    parser.add_argument('--repetitions',
                        type=int,
                        default=1,
                        help='Number of repetitions per route.')

    # agent-related options
    parser.add_argument("-a", "--agent", type=str, default='simulation/leaderboard/team_code/pnp_agent_e2e_demo.py', help="Path to Agent's py file to evaluate")
    parser.add_argument("--agent-config", type=str, default='simulation/leaderboard/team_code/agent_config_e2e/pnp_config_codriving_5_10.yaml', help="Path to Agent's configuration file")

    parser.add_argument("--track", type=str, default='SENSORS', help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=int, default=0, help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str,
                        default='/GPFS/data/gjliu-1/Auto-driving/V2Xverse/out_dir/demo/results.json',
                        help="Path to checkpoint used for saving statistics and resuming")
    parser.add_argument('--ego-num', type=int, default=1, help='The number of ego vehicles')
    parser.add_argument('--skip_existed', type=int, default=1, help='If the result exist, return')
    parser.add_argument(
        '--route-plots-only',
        action='store_true',
        help='Build RouteScenario and export route plots only; skip running the driving episode.',
    )
    # crazy level: 0-5, the probability of ignoring front car.
    # crazy proportion: the probability of a car is crazy 
    
    arguments = parser.parse_args()
    install_process_lifecycle_logging(
        EVALUATOR_PROCESS_NAME,
        env_keys=(
            "CARLA_ROOTCAUSE_LOGDIR",
            "CARLA_CONNECTION_EVENTS_LOG",
            "SAVE_PATH",
            "CHECKPOINT_ENDPOINT",
        ),
    )
    print("[INFO] leaderboard_evaluator_parameter starting.")
    check_result = False

    if arguments.skip_existed:
        expected_checkpoint = os.path.join(
            os.path.dirname(arguments.checkpoint),
            "ego_vehicle_{}".format(0),
            os.path.basename(arguments.checkpoint),
        )
        # Safeguard the rerun logic: Only inspect existing runs when the expected checkpoint is present.
        if os.path.exists(expected_checkpoint):
            image_save_path = os.environ['SAVE_PATH']
            if os.path.isdir(image_save_path):
                route_list = [
                    entry for entry in os.listdir(image_save_path)
                    if os.path.isdir(os.path.join(image_save_path, entry))
                ]
                route_list.sort()

                rerun_required = False
                for log_dir in reversed(route_list):
                    log_file_path = "{}/{}/log/log.log".format(image_save_path, log_dir)
                    if not os.path.isfile(log_file_path):
                        continue

                    for invalid_str in ['Traceback', 'No space']:
                        if check_log_file(log_file_path, invalid_str):
                            rerun_required = True
                            break

                    if not rerun_required and not check_log_file(log_file_path, 'RouteCompletionTest'):
                        rerun_required = True

                    if rerun_required:
                        print('Invalid results in {}, rerun!'.format(log_dir))
                        break
                    else:
                        print('Valid results in {}, skip!'.format(log_dir))
                        return

                if not route_list:
                    print('No previous results found, rerunning!')
                    rerun_required = True
                elif not rerun_required:
                    print('No valid logs found to verify, rerunning!')
                if rerun_required:
                    print('Proceeding with rerun.')
                else:
                    print('No rerun required, continuing with evaluation.')
                    return

            else:
                print('{} is empty, rerunning!'.format(image_save_path))
        else:
            print('{} do not exists, continue!'.format(os.path.join(os.path.dirname(arguments.checkpoint), "ego_vehicle_{}".format(0), os.path.basename(arguments.checkpoint))))

    if not os.path.exists(os.environ["SAVE_PATH"]):
        os.makedirs(os.environ["SAVE_PATH"])
    if not os.path.exists(os.path.dirname(os.environ["CHECKPOINT_ENDPOINT"])):
        os.makedirs(os.path.dirname(os.environ["CHECKPOINT_ENDPOINT"]))
        
    if not "RESULT_ROOT" in os.environ:
        os.environ["RESULT_ROOT"] = os.environ["DATA_ROOT"]

    statistics_manager_all = []
    for i in range(arguments.ego_num):
        statistics_manager = StatisticsManager(ego_car_id=i)
        statistics_manager_all.append(statistics_manager)

    leaderboard_evaluator = None
    try:
        leaderboard_evaluator = LeaderboardEvaluator(arguments, statistics_manager_all)
        leaderboard_evaluator.run(arguments)

    except Exception as e:
        log_process_exception(e, process_name=EVALUATOR_PROCESS_NAME, where="main")
        traceback.print_exc()
        sys.exit(1)
    finally:
        if leaderboard_evaluator is not None:
            del leaderboard_evaluator


if __name__ == '__main__':
    main()
