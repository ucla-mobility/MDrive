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
from distutils.version import LooseVersion
import importlib
import os
import sys
import gc
import pkg_resources
import sys
import carla
import copy
import signal
import torch
import time
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

    def write(self, message):
        local_time = time.localtime(time.time())
        if message=='\n':
            time_str = ''
        else:
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
        try:
            self.terminal.write(message)
            self.log = open(self.file_name, "a")
            # self.log.write(time_str+'  '+message)
            self.log.write(message)
            # self.log.close()
        except Exception as e:
            pass

    def flush(self):
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
        self.statistics_manager = statistics_manager
        self.sensors = None
        self.sensor_icons = []
        self._vehicle_lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam

        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        self.client = carla.Client(args.host, int(args.port))
        if args.timeout:
            self.client_timeout = float(args.timeout)
        self.client.set_timeout(self.client_timeout)

        self.traffic_manager = self.client.get_trafficmanager(int(args.trafficManagerPort))

        dist = pkg_resources.get_distribution("carla")
        # if dist.version != 'leaderboard':
        #     if LooseVersion(dist.version) < LooseVersion('0.9.10'):
        #         raise ImportError("CARLA version 0.9.10.1 or newer required. CARLA version found: {}".format(dist))

        # Load agent
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
        if hasattr(self, 'manager') and self.manager:
            del self.manager
        if hasattr(self, 'world') and self.world:
            del self.world

    def _cleanup(self):
        """
        Remove and destroy all actors
        """

        # Simulation still running and in synchronous mode?
        if self.manager and self.manager.get_running_status() \
                and hasattr(self, 'world') and self.world:
            # Reset to asynchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            self.traffic_manager.set_synchronous_mode(False)

        if self.manager:
            self.manager.cleanup()

        CarlaDataProvider.cleanup()

        for i, _ in enumerate(self.ego_vehicles):
            if self.ego_vehicles[i]:
                self.ego_vehicles[i].destroy()
                self.ego_vehicles[i] = None
        self.ego_vehicles = []

        if self._agent_watchdog._timer:
            self._agent_watchdog.stop()

        if hasattr(self, 'agent_instance') and self.agent_instance:
            self.agent_instance.destroy()
            self.agent_instance = None

        if hasattr(self, 'statistics_manager') and self.statistics_manager:
            for j in range(self.ego_vehicles_num):
                self.statistics_manager[j].scenario = None

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

    def _register_statistics(self, config, ego_car_num, checkpoint, entry_status, crash_message="",):
        """
        Computes and saved the simulation statistics
        """
        # register statistics
        
        current_stats_record = []
        current_stats_record.extend([[] for _ in range(0,ego_car_num)])
        for i in range(ego_car_num):
            
            current_stats_record[i] = self.statistics_manager[i].compute_route_statistics(
                config,
                self.manager.scenario_duration_system,
                self.manager.scenario_duration_game,
                crash_message,
                pdm_trace=self.manager.pdm_traces[i] if hasattr(self.manager, "pdm_traces") else None,
                pdm_world_trace=getattr(self.manager, "pdm_world_trace", None),
                pdm_tl_polygons=getattr(self.manager, "pdm_tl_polygons", None),
            )

            print("\033[1m> Registering the route statistics\033[0m")
            path_tmp = os.path.join(os.path.dirname(checkpoint), "ego_vehicle_{}".format(i), os.path.basename(checkpoint))
            folder_path = os.path.join(os.path.dirname(checkpoint), "ego_vehicle_{}".format(i))
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            self.statistics_manager[i].save_record(current_stats_record[i], config.index, path_tmp)
            self.statistics_manager[i].save_entry_status(entry_status, False, path_tmp)

    def _load_and_run_scenario(self, args, config):
        """
        Load and run the scenario given by config.

        Depending on what code fails, the simulation will either stop the route and
        continue from the next one, or report a crash and stop.

        Args:
            args: argparse.Namespace, global config
            config: srunner.scenarioconfigs.route_scenario_configuration.RouteScenarioConfiguration, config for route scenarios

        """
        crash_message = ""
        entry_status = "Started"

        print("\n\033[1m========= Preparing {} (repetition {}) =========".format(config.name, config.repetition_index))
        print("> Setting up the agent\033[0m")

        # Prepare the statistics of the route
        for j in range(self.ego_vehicles_num):
            self.statistics_manager[j].set_route(config.name, config.index)

        # Set up the user's agent, and the timer self._agent_watchdog to avoid freezing the simulation
        try:
            self._agent_watchdog.start()
            # agent_class_name for example 'AutoPilot', 'PnP_Agent' .etc
            agent_class_name = getattr(self.module_agent, 'get_entry_point')()
            self.agent_instance = getattr(self.module_agent, agent_class_name)(args.agent_config, 
                                                                               self.ego_vehicles_num)
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
                except:
                    print('load save path failed')

            # Log simulation information(or error), every printed string will be logged in file log.log
            log_dir = None
            log_dir = os.path.join(log_root_dir,'log')
            self.log_dir = log_dir
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            log_file_dir = os.path.join(log_dir,'log.log')
            self.log_file_dir = log_file_dir
            sys.stdout = Logger(log_file_dir)
            args_file_dir = os.path.join(log_dir,'args.json')
            args_dict = vars(args)
            json_str = json.dumps(args_dict, indent=2)
            with open(args_file_dir, 'w') as json_file:
                json_file.write(json_str)
            print("{} (repetition {}) Log file initialized!".format(config.name, config.repetition_index))

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
            # The sensors are invalid -> set the ejecution to rejected and stop
            print("\n\033[91mThe sensor's configuration used is invalid:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()
            traceback.print_exc(file=open(log_file_dir,'a'))
            crash_message = "Agent's sensors were invalid"
            entry_status = "Rejected"
            # self._register_statistics(config, args.ego_num, args.checkpoint, entry_status, crash_message)
            self._cleanup()
            sys.exit(-1)

        except Exception as e:
            # The agent setup has failed -> start the next route
            print("\n\033[91mCould not set up the required agent:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()
            traceback.print_exc(file=open(log_file_dir,'a'))
            crash_message = "Agent couldn't be set up"
            # self._register_statistics(config,  args.ego_num, args.checkpoint, entry_status, crash_message)
            self._cleanup()
            return

        # Load the world and the scenario
        print("\033[1m> Loading the world\033[0m")    
        try:
            # create world
            self._load_and_wait_for_world(args, config.town, config.ego_vehicles)
            self._prepare_ego_vehicles(config.ego_vehicles, False)
            with open(args.scenario_parameter, 'r', encoding='utf-8') as f:
                scenario_parameter = yaml.load(f.read(), Loader=yaml.FullLoader)
            # NOTE(GJH)：change the arg of RouteScenario - add the arg of scenario_config
            # print(args.scenario_parameter)
            if scenario_parameter['Background']['turn_off_light']:
                # crazy traffic light
                print("crazy traffic lights")
                [tf.set_state(carla.libcarla.TrafficLightState.Green) for tf in self.world.get_actors().filter("*traffic_light*") if hasattr(tf,"set_state")]
                [tf.freeze(True) for tf in self.world.get_actors().filter("*traffic_light*") if hasattr(tf,"freeze")]

            scenario = RouteScenario(world=self.world, config=config, debug_mode=args.debug, \
                    ego_vehicles_num=self.ego_vehicles_num, log_dir=log_dir,scenario_parameter=scenario_parameter)
            config.trajectory=scenario.get_new_config_trajectory()
            if self.ego_vehicles_num != 1 :
                for j in range(self.ego_vehicles_num):
                    self.statistics_manager[j].set_scenario(scenario.scenario[j])
            else:
                self.statistics_manager[0].set_scenario(scenario.scenario)


            # self.agent_instance._init()
            # self.agent_instance.sensor_interface = SensorInterface()

            # Night mode
            if config.weather.sun_altitude_angle < 0.0:
                for vehicle in scenario.ego_vehicles:
                    vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))

            # Load scenario and agent into manager then prepare to run it
            if args.record:
                self.client.start_recorder("{}/{}_rep{}.log".format(args.record, config.name, config.repetition_index))
            self.manager.load_scenario(scenario, self.agent_instance, config.repetition_index, self.ego_vehicles_num, save_root=config.save_path_root, sensor_tf_list=scenario.get_sensor_tf(), is_crazy=scenario_parameter['Background']['turn_off_light'])

        except Exception as e:
            # The scenario is wrong -> set the ejecution to crashed and stop
            print("\n\033[91mThe scenario could not be loaded:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()
            traceback.print_exc(file=open(log_file_dir,'a'))

            crash_message = "Simulation crashed"
            entry_status = "Crashed"

            self._register_statistics(config,  args.ego_num, args.checkpoint, entry_status, crash_message)

            if args.record:
                self.client.stop_recorder()

            self._cleanup()
            sys.exit(-1)

        print("\033[1m> Running the route\033[0m")

        # Run the scenario
        try:
            self.manager.run_scenario()
        except AgentError as e:
            # The agent has failed -> stop the route
            print("\n\033[91mStopping the route, the agent has crashed:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()
            traceback.print_exc(file=open(log_file_dir,'a'))

            crash_message = "Agent crashed"
        except Exception as e:
            print("\n\033[91mError during the simulation:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()
            traceback.print_exc(file=open(log_file_dir,'a'))

            crash_message = "Run scenario crashed"
            entry_status = "Crashed"

        # Stop the scenario
        try:
            print("\033[1m> Stopping the route\033[0m")
            self.manager.stop_scenario()
            #GXK111
            self._register_statistics(config,  args.ego_num, args.checkpoint, entry_status, crash_message)

            if args.record:
                self.client.stop_recorder()

            # Remove all actors
            scenario.remove_all_actors()

            self._cleanup()
            # Final clean up!
            # print(self.world.get_actors())
            # print(self.world.get_actors().filter("*sensor*"))
            for zombie in self.world.get_actors().filter("*sensor*"):
                zombie.stop()
                zombie.destroy()
                zombie = None            

        except Exception as e:
            print("\n\033[91mFailed to stop the scenario, the statistics might be empty:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()
            traceback.print_exc(file=open(log_file_dir,'a'))

            crash_message = "Simulation crashed"

        # if crash_message == "Simulation crashed":
        #     sys.exit(-1)

    def run(self, args: argparse.Namespace):
        """
        Run the challenge mode
        """

        print(f'''run with {args.ego_num} cars\n''')
        route_indexer_dict = {}
        route_path_dict = {}
        
        # Search recursively for XML files in the routes directory
        # This handles nested scenario directories (e.g., routes/Scenario_Name/vehicle_*.xml)
        import glob
        xml_files = glob.glob(os.path.join(args.routes_dir, '**', '*.xml'), recursive=True)

        # Ignore custom actor XMLs (they live under actors/); only keep ego route files
        xml_files = [
            x for x in xml_files
            if "actors" not in os.path.normpath(x).split(os.sep)
        ]
        
        for xml_path in xml_files:
            # Extract the ego_id from the filename (last part before .xml)
            # e.g., "town05_vehicle_1_0.xml" -> "0"
            filename = os.path.basename(xml_path)
            ego_id_str = filename.split('.')[0].split('_')[-1]
            
            # Only add if not already present (in case of duplicates across scenarios)
            if ego_id_str not in route_path_dict:
                route_path_dict[ego_id_str] = xml_path

        if not route_path_dict:
            raise RuntimeError(
                f"No route XML files found under {args.routes_dir}"
            )

        route_indexer = None
        if args.ego_num > 0:
            for ego_id in range(args.ego_num):
                if str(ego_id) not in route_path_dict:
                    raise RuntimeError(
                        f"Missing ego route XML for ego_id={ego_id}. "
                        f"Discovered keys: {sorted(route_path_dict.keys())}"
                    )
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

        print("Start Running!")
        while route_indexer.peek():
            try:
                # setup, load config of the next route
                config = route_indexer.next()
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

            except Exception as e:
                print('route error:',e)

        # save global statistics
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
            traceback.print_exc(file=open(self.log_file_dir,'a'))

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
    # crazy level: 0-5, the probability of ignoring front car.
    # crazy proportion: the probability of a car is crazy 
    
    arguments = parser.parse_args()
    print("start")
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

    try:
        leaderboard_evaluator = LeaderboardEvaluator(arguments, statistics_manager_all)
        leaderboard_evaluator.run(arguments)

    except Exception as e:
        traceback.print_exc()
    finally:
        del leaderboard_evaluator


if __name__ == '__main__':
    main()
