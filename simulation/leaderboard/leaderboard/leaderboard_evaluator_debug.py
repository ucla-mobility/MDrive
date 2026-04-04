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

sys.path.append('/GPFS/data/changxingliu/V2Xverse/simulation/scenario_runner')
sys.path.append('/GPFS/data/changxingliu/V2Xverse/simulation/leaderboard')
sys.path.append('/GPFS/data/changxingliu/V2Xverse')
sys.path.append('/GPFS/data/changxingliu/V2Xverse/carla/PythonAPI/carla')

os.environ["CARLA_ROOT"] = '/GPFS/data/changxingliu/V2Xverse/carla'
os.environ["DATA_ROOT"] = 'dataset_cop3'
os.environ["YAML_ROOT"] = 'simulation/hypes_yaml'
os.environ["ROUTES"]='/GPFS/data/changxingliu/V2Xverse/simulation/leaderboard/data/new_routes/town05_short_r1_ins_c_0.xml'
os.environ["LEADERBOARD_ROOT"]='simulation/leaderboard'
os.environ["CHALLENGE_TRACK_CODENAME"]='SENSORS'

os.environ["PORT"]='1415' # same as the carla server port

os.environ["TEAM_AGENT"]='simulation/leaderboard/team_code/pnp_vlm_intention_agent_v2v.py'   #'simulation/leaderboard/team_code/auto_pilot.py' , 'simulation/leaderboard/team_code/pnp_agent.py' 
os.environ["TEAM_CONFIG"]='simulation/leaderboard/team_code/agent_config/pnp_config_codriving_vlm_8_10.yaml' # 'simulation/hypes_yaml/weather-0.yaml' , 'simulation/leaderboard/team_code/pnp_config.py'
os.environ["CHECKPOINT_ENDPOINT"]='/GPFS/data/changxingliu/V2Xverse/results/results_driving_test/v2x_final/town05_short_r1_ins_c/results.json'
os.environ["SCENARIOS"]='simulation/leaderboard/data/scenarios/town05_all_scenarios_2.json'
os.environ["SAVE_PATH"]='/GPFS/data/changxingliu/V2Xverse/results/results_driving_test/image/v2x_final/town05_short_r1_ins_c'
os.environ["RESULT_ROOT"]='/GPFS/data/changxingliu/V2Xverse/results/results_driving_test'

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
from leaderboard_evaluator_parameter import main


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

if __name__ == '__main__':
    main()
