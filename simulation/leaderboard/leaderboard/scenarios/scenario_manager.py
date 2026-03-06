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
import signal
import sys
import time
import math
import re
from pathlib import Path
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
from leaderboard.envs.sensor_interface import SensorReceivedNoData
from leaderboard.utils.result_writer import ResultOutputProvider
try:
    from leaderboard.utils.veer_audit import VeerAuditor
except Exception:
    VeerAuditor = None


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
        # Tracks position history to detect when ALL vehicles are stuck
        def _env_float(name, default):
            raw = os.environ.get(name, "")
            if not raw:
                return default
            try:
                return float(raw)
            except (TypeError, ValueError):
                print(f"[STALL CONFIG] Invalid {name}={raw!r}; using default {default}")
                return default

        self._stall_position_history = {}  # {ego_id: [(time, x, y, z), ...]}
        self._stall_check_interval = max(0.5, _env_float("CUSTOM_STALL_CHECK_INTERVAL", 5.0))
        self._stall_last_check_time = 0.0
        # Use relaxed defaults to reduce premature stall termination.
        self._stall_threshold_time = max(
            self._stall_check_interval,
            _env_float("CUSTOM_STALL_THRESHOLD_TIME", 240.0),
        )
        self._stall_min_distance = max(0.0, _env_float("CUSTOM_STALL_MIN_DISTANCE", 0.2))
        self._stall_min_speed = max(0.0, _env_float("CUSTOM_STALL_MIN_SPEED", 0.03))

        # PDM trace logging (for navsim-style metrics)
        self.pdm_traces = []
        self._pdm_prev_ang_vel = []
        self._pdm_prev_time = []
        self.pdm_world_trace = []
        self._pdm_last_world_time = None
        self.pdm_tl_polygons = {}

        # record simulation time cost
        self.time_record = []
        self.c_time_record = []
        self.a_time_record = []
        self.sc_time_record = []
        self._reset_event_idx = 0
        self._scenario_name = None
        self._scenario_town = None
        self._scenario_route = None
        self._veer_auditor = None
        
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

    def _init_veer_auditor(self, scenario, rep_number) -> None:
        self._veer_auditor = None
        enabled = os.environ.get("CUSTOM_VEER_AUDIT", "").lower() in ("1", "true", "yes")
        if not enabled:
            return
        if VeerAuditor is None:
            print("[VEER_AUDIT] disabled: missing module leaderboard.utils.veer_audit")
            return
        try:
            result_root = os.environ.get("RESULT_ROOT", "").strip()
            if not result_root:
                checkpoint = os.environ.get("CHECKPOINT_ENDPOINT", "").strip()
                if checkpoint:
                    result_root = str(Path(checkpoint).parent)
                else:
                    result_root = "."

            scenario_name = getattr(getattr(scenario, "config", None), "name", None) or "scenario"
            scenario_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(scenario_name)).strip("_")
            if not scenario_slug:
                scenario_slug = "scenario"
            out_dir = Path(result_root) / "veer_audit" / f"{scenario_slug}__rep{int(rep_number)}"
            save_ticks = os.environ.get("CUSTOM_VEER_AUDIT_SAVE_TICKS", "").lower() in (
                "1",
                "true",
                "yes",
            )
            self._veer_auditor = VeerAuditor(
                output_dir=out_dir,
                scenario_name=str(scenario_name),
                town=getattr(getattr(scenario, "config", None), "town", None),
                route_id=getattr(getattr(scenario, "config", None), "route_id", None),
                repetition=int(rep_number),
                save_ticks=bool(save_ticks),
            )
            self._veer_auditor.configure_from_scenario(scenario, int(self.ego_vehicles_num))
            print(f"[VEER_AUDIT] enabled output={out_dir} save_ticks={int(bool(save_ticks))}")
        except Exception as exc:
            self._veer_auditor = None
            print(f"[VEER_AUDIT] init failed: {exc}")

    def cleanup(self):
        """
        Reset all parameters
        """
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

        # To print the scenario tree uncomment the next line
        # py_trees.display.render_dot_tree(self.scenario_tree)

        for vehicle_num in range(self.ego_vehicles_num):
            print("set ip sensor for ego vehicle {}".format(vehicle_num))
            self._agent.setup_sensors(self.ego_vehicles[vehicle_num], vehicle_num, save_root, self._debug_mode)
            self.first_entry.append(True)

        self._init_veer_auditor(scenario, rep_number)

    def run_scenario(self):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        self._watchdog.start()
        self._running = True
        
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

            self._watchdog.update()
            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()

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
                    if os.environ.get("CUSTOM_EGO_LOG_REPLAY", "").lower() in ("1", "true", "yes"):
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
                if  CarlaDataProvider.get_hero_actor(hero_id=vehicle_num) and not CarlaDataProvider.get_hero_actor(hero_id=vehicle_num).is_alive:
                    self._agent.del_ego_sensor(vehicle_num)
                    self._agent.cleanup_single(vehicle_num)
                    self._agent.cleanup_rsu(vehicle_num)
                    print("destroy ego type 0 : {}".format(vehicle_num))
                    CarlaDataProvider.remove_actor_by_id(CarlaDataProvider.get_hero_actor(hero_id=vehicle_num).id)

            # Agent take action (eg. save data/produce control signal)
            try:
                ego_action = self._agent()

            # Special exception inside the agent that isn't caused by the agent
            except SensorReceivedNoData as e:
                raise RuntimeError(e)

            except Exception as e:
                raise AgentError(e)

            # destroy ego if it is not alive
            for vehicle_num in range(self.ego_vehicles_num):
                if  CarlaDataProvider.get_hero_actor(hero_id=vehicle_num) and not CarlaDataProvider.get_hero_actor(hero_id=vehicle_num).is_alive:
                    self._agent.del_ego_sensor(vehicle_num)
                    self._agent.cleanup_single(vehicle_num)
                    self._agent.cleanup_rsu(vehicle_num)
                    print("destroy ego type 1 : {}".format(vehicle_num))
                    CarlaDataProvider.remove_actor_by_id(CarlaDataProvider.get_hero_actor(hero_id=vehicle_num).id)

            # Execute driving control signal
            for vehicle_num in range(self.ego_vehicles_num):
                try:
                    ego = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)
                    if ego:
                        if ego.is_alive:
                            if os.environ.get('DEBUG_SCENARIOMGR', '').lower() in ('1', 'true', 'yes'):
                                print(f"[DEBUG SCENARIO_MGR] Applying control to ego {vehicle_num}: throttle={ego_action[vehicle_num].throttle:.3f}, brake={ego_action[vehicle_num].brake:.3f}, steer={ego_action[vehicle_num].steer:.3f}")
                            self.ego_vehicles[vehicle_num].apply_control(ego_action[vehicle_num])
                            # record trace for PDM metrics
                            self._record_pdm_sample(vehicle_num, ego, timestamp)
                            self._record_veer_sample(vehicle_num, ego, timestamp)
                except:
                    pass

            # Tick scenario
            for vehicle_num in range(self.ego_vehicles_num):
                try:
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
            log_replay_ego = os.environ.get("CUSTOM_EGO_LOG_REPLAY", "").lower() in ("1", "true", "yes")
            missing_egos = []
            nonrunning_egos = []
            dead_egos = []
            for vehicle_num in range(self.ego_vehicles_num):
                if CarlaDataProvider.get_hero_actor(hero_id=vehicle_num) is None:
                    missing_egos.append(vehicle_num)
                    stop_flag += 1
                    if CarlaDataProvider.get_hero_actor(hero_id=vehicle_num):
                        self._agent.del_ego_sensor(vehicle_num)
                        self._agent.cleanup_single(vehicle_num)
                        self._agent.cleanup_rsu(vehicle_num)
                        print("destroy ego type 2 {}".format(vehicle_num))
                        CarlaDataProvider.remove_actor_by_id(CarlaDataProvider.get_hero_actor(hero_id=vehicle_num).id)
                    if stop_flag == self.ego_vehicles_num:
                        self._debug_reset(
                            "all_egos_missing",
                            details=f"missing={missing_egos}",
                        )
                        self._running = False
                
                elif self.scenario_tree[vehicle_num].status != py_trees.common.Status.RUNNING or not CarlaDataProvider.get_hero_actor(hero_id=vehicle_num).is_alive:
                    nonrunning_egos.append(vehicle_num)
                    if not CarlaDataProvider.get_hero_actor(hero_id=vehicle_num).is_alive:
                        dead_egos.append(vehicle_num)
                    if log_replay_ego:
                        try:
                            done_key = f"log_replay_done_ego_{vehicle_num}"
                            replay_done = bool(py_trees.blackboard.Blackboard().get(done_key))
                        except Exception:
                            replay_done = False
                        # In log replay mode, keep ego alive until replay has reached the end.
                        if not replay_done and CarlaDataProvider.get_hero_actor(hero_id=vehicle_num) and CarlaDataProvider.get_hero_actor(hero_id=vehicle_num).is_alive:
                            continue
                    stop_flag += 1
                    if CarlaDataProvider.get_hero_actor(hero_id=vehicle_num):
                        self._agent.del_ego_sensor(vehicle_num)
                        self._agent.cleanup_single(vehicle_num)
                        self._agent.cleanup_rsu(vehicle_num)
                        print("destroy ego type 3 {}".format(vehicle_num))
                        print('flag1:', self.scenario_tree[vehicle_num].status != py_trees.common.Status.RUNNING)
                        print('flag2:', not CarlaDataProvider.get_hero_actor(hero_id=vehicle_num).is_alive)
                        CarlaDataProvider.remove_actor_by_id(CarlaDataProvider.get_hero_actor(hero_id=vehicle_num).id)
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

            # terminate route scenarios once all egos are either completed OR blocked
            log_replay_ego = os.environ.get("CUSTOM_EGO_LOG_REPLAY", "").lower() in ("1", "true", "yes")
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
                import time
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
                    if scenario_instance is None:
                        continue
                    criteria = scenario_instance.get_criteria()
                    if not criteria:
                        if os.environ.get('DEBUG_FINISH', '').lower() in ('1', 'true', 'yes'):
                            print(f"[DEBUG FINISH] Ego {idx}: No criteria found")
                        all_egos_done = False
                        break
                    # Check if this ego is either completed (SUCCESS) or blocked (FAILURE)
                    ego_completed = False
                    ego_blocked = False
                    # Debug: Show criterion types (every 10 seconds)
                    import time
                    if not hasattr(self, '_last_debug_time'):
                        self._last_debug_time = 0
                    if time.time() - self._last_debug_time > 10:
                        if os.environ.get('DEBUG_FINISH', '').lower() in ('1', 'true', 'yes'):
                            print(f"[DEBUG FINISH] Ego {idx}: Found {len(criteria)} criteria: {[type(c).__name__ for c in criteria]}")
                            for c in criteria:
                                print(f"[DEBUG FINISH]   - {type(c).__name__}: status={getattr(c, 'test_status', 'N/A')}")
                        self._last_debug_time = time.time()
                    for criterion in criteria:
                        if isinstance(criterion, RouteCompletionTest):
                            if criterion.test_status == "SUCCESS":
                                ego_completed = True
                                if os.environ.get('DEBUG_FINISH', '').lower() in ('1', 'true', 'yes'):
                                    print(f"[DEBUG FINISH] Ego {idx}: RouteCompletionTest SUCCESS")
                        elif isinstance(criterion, ActorSpeedAboveThresholdTest):
                            if criterion.test_status == "FAILURE":
                                ego_blocked = True
                                if os.environ.get('DEBUG_FINISH', '').lower() in ('1', 'true', 'yes'):
                                    print(f"[DEBUG FINISH] Ego {idx}: AgentBlockedTest FAILURE")
                    # Ego is "done" if it completed OR if it's blocked
                    if not (ego_completed or ego_blocked):
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
                self._running = False

            # Position-based stall detection (conservative fallback)
            # Only check if still running after criteria checks
            if self._running:
                stall_detected = self._check_position_stall()
                if stall_detected:
                    self._debug_reset(
                        "stall_detection",
                        details=f"threshold_time={self._stall_threshold_time} min_dist={self._stall_min_distance}",
                    )
                    print(f"\n[STALL DETECTION] All vehicles have been stationary for {self._stall_threshold_time}s - terminating scenario")
                    self._running = False

        if self._running and self.get_running_status():
            CarlaDataProvider.get_world().tick(self._timeout)

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

    def _record_veer_sample(self, vehicle_num, ego, timestamp):
        if self._veer_auditor is None:
            return
        try:
            transform = ego.get_transform()
            velocity = ego.get_velocity()
            self._veer_auditor.log_tick(
                ego_idx=int(vehicle_num),
                sim_time_s=float(timestamp.elapsed_seconds),
                transform=transform,
                velocity=velocity,
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
        
        # Use simulation time so stall logic is aligned with CARLA progress.
        try:
            current_time = float(GameTime.get_time())
        except Exception:
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
            
            if distance >= self._stall_min_distance or speed > self._stall_min_speed:
                # This ego has moved enough OR is currently moving, not stalled
                all_stalled = False
                # Debug: print which vehicle is NOT stalled
                print(f"[STALL DEBUG] Ego {vehicle_num} NOT stalled: distance={distance:.2f}m, speed={speed:.2f}m/s")
                break
        
        # Debug: if all stalled, print warning before triggering
        if all_stalled:
            print(f"[STALL DEBUG] All {active_ego_count} egos appear stalled. Checking one more time...")
            # Double-check: verify ALL velocities are near zero right now
            for vehicle_num in range(self.ego_vehicles_num):
                ego = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)
                if ego and ego.is_alive:
                    vel = ego.get_velocity()
                    speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                    if speed > self._stall_min_speed:
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

        self.end_system_time = time.time()
        self.end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - self.start_system_time
        self.scenario_duration_game = self.end_game_time - self.start_game_time

        if self._veer_auditor is not None:
            try:
                self._veer_auditor.finalize()
                print("[VEER_AUDIT] summary written")
            except Exception as exc:
                print(f"[VEER_AUDIT] finalize failed: {exc}")
            self._veer_auditor = None

        if self.get_running_status():
            # print("terminate ego vehicle in the first step {}".format(ego_vehicle_id))
            if len(self.ego_vehicles) == 0:
                for scenario_item in self.scenario:
                    if scenario_item is not None:
                        scenario_item.terminate()
            else:
                for ego_vehicle_id in range(len(self.ego_vehicles)):
                    if self.scenario[ego_vehicle_id] is not None:
                        # print("terminate ego vehicle {}".format(ego_vehicle_id))
                        self.scenario[ego_vehicle_id].terminate()

            if self._agent is not None:
                self._agent.cleanup()
                self._agent = None

            if self.sensor_tf_list is not None:
                [_sensor.cleanup() for _sensor in self.sensor_tf_list]
                self.sensor_tf_list = None

            self.analyze_scenario()

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
