#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Module used to parse all the route and scenario configuration parameters.
"""
from collections import OrderedDict
from pathlib import Path
import json
import math
import xml.etree.ElementTree as ET
from typing import Dict, List
import os

import carla
from agents.navigation.local_planner import RoadOption
from srunner.scenarioconfigs.route_scenario_configuration import RouteScenarioConfiguration

# TODO  check this threshold, it could be a bit larger but not so large that we cluster scenarios.
TRIGGER_THRESHOLD = 2.0  # Threshold to say if a trigger position is new or repeated, works for matching positions
TRIGGER_ANGLE_THRESHOLD = 10  # Threshold to say if two angles can be considering matching when matching transforms.

_CUSTOM_ACTOR_MANIFEST_CACHE = None
_CUSTOM_ACTOR_BEHAVIOR_CACHE = None
_EGO_VEHICLE_MODELS_CACHE = None  # Cache for ego vehicle models from manifest
ROLE_DEFAULTS: Dict[str, Dict[str, object]] = {
    "npc": {"model": "vehicle.tesla.model3", "speed": 8.0},
    "pedestrian": {"model": "walker.pedestrian.0001", "speed": 1.5},
    "walker": {"model": "walker.pedestrian.0001", "speed": 1.5},  # Alias for pedestrian
    "bicycle": {"model": "vehicle.diamondback.century", "speed": 4.0},
    "bike": {"model": "vehicle.diamondback.century", "speed": 4.0},
    "cyclist": {"model": "vehicle.diamondback.century", "speed": 4.0},  # Alias for bicycle
    "static": {"model": "static.prop.trafficcone01", "speed": 0.0},
    "static_prop": {"model": "static.prop.trafficcone01", "speed": 0.0},
}


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_routes_dir(manifest_path: Path) -> Path:
    """
    Determine the base directory for custom actor files.
    """
    env_route_dir = os.environ.get("ROUTES_DIR")
    if env_route_dir:
        return Path(env_route_dir).expanduser().resolve()
    return manifest_path.parent.resolve()


def _load_custom_actor_manifest() -> Dict[str, List[Dict[str, object]]]:
    """
    Load and cache the custom actor manifest if present.
    Checks environment variable first, then looks for actors_manifest.json in standard locations.
    """
    global _CUSTOM_ACTOR_MANIFEST_CACHE  # pylint: disable=global-statement

    if _CUSTOM_ACTOR_MANIFEST_CACHE is not None:
        return _CUSTOM_ACTOR_MANIFEST_CACHE

    manifest_path = None
    
    # Try environment variable first
    manifest_env = os.environ.get("CUSTOM_ACTOR_MANIFEST")
    if manifest_env:
        manifest_path = Path(manifest_env).expanduser().resolve()
    
    # If not found in env or doesn't exist, look for it in route directories
    if not manifest_path or not manifest_path.exists():
        # Look for actors_manifest.json in scenario route subdirectories
        # Try to find any actors_manifest.json in routes-like directories
        candidates = [
            Path.cwd() / "routes" / "actors_manifest.json",
            Path.cwd() / "scenario_builder_api" / "routes" / "actors_manifest.json",
        ]
        
        # Also check for manifests in route subdirectories (e.g., routes/Scenario_1_attempt1/actors_manifest.json)
        for routes_dir in [Path.cwd() / "routes", Path.cwd() / "scenario_builder_api" / "routes"]:
            if routes_dir.exists():
                for subdir_manifest in routes_dir.glob("*/actors_manifest.json"):
                    candidates.append(subdir_manifest)
        
        for candidate in candidates:
            if candidate.exists():
                manifest_path = candidate
                break
    
    if not manifest_path or not manifest_path.exists():
        _CUSTOM_ACTOR_MANIFEST_CACHE = {}
        return _CUSTOM_ACTOR_MANIFEST_CACHE

    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _CUSTOM_ACTOR_MANIFEST_CACHE = {}
        return _CUSTOM_ACTOR_MANIFEST_CACHE

    base_dir = _resolve_routes_dir(manifest_path)
    actor_entries: Dict[str, List[Dict[str, object]]] = {}

    for role, entries in manifest_data.items():
        if role == "ego":
            continue
        if not isinstance(entries, list):
            continue
        for entry in entries:
            route_id = entry.get("route_id")
            rel_path = entry.get("file")
            town = entry.get("town")
            if not route_id or not rel_path:
                continue
            rel_path_obj = Path(rel_path)
            actor_path = (base_dir / rel_path_obj).resolve()
            actor_entries.setdefault(str(route_id), []).append(
                {
                    "role": role,
                    "town": town,
                    "path": actor_path,
                    "name": entry.get("name") or rel_path_obj.stem,
                    "speed": entry.get("speed"),
                    "model": entry.get("model"),
                    "control_mode": entry.get("control_mode"),
                }
            )

    _CUSTOM_ACTOR_MANIFEST_CACHE = actor_entries
    return _CUSTOM_ACTOR_MANIFEST_CACHE


def _load_custom_actor_behaviors() -> Dict[str, List[Dict[str, object]]]:
    """
    Load and cache optional behavior specs for custom actors.
    Returns a mapping of route_id -> list of behavior entries.
    """
    global _CUSTOM_ACTOR_BEHAVIOR_CACHE  # pylint: disable=global-statement

    if _CUSTOM_ACTOR_BEHAVIOR_CACHE is not None:
        return _CUSTOM_ACTOR_BEHAVIOR_CACHE

    behavior_path = None
    env_path = os.environ.get("CUSTOM_ACTOR_BEHAVIORS")
    if env_path:
        behavior_path = Path(env_path).expanduser().resolve()

    if not behavior_path or not behavior_path.exists():
        manifest_env = os.environ.get("CUSTOM_ACTOR_MANIFEST")
        if manifest_env:
            cand = Path(manifest_env).expanduser().resolve().parent / "actors_behavior.json"
            if cand.exists():
                behavior_path = cand

    if not behavior_path or not behavior_path.exists():
        candidates = [
            Path.cwd() / "routes" / "actors_behavior.json",
            Path.cwd() / "scenario_builder_api" / "routes" / "actors_behavior.json",
        ]
        for routes_dir in [Path.cwd() / "routes", Path.cwd() / "scenario_builder_api" / "routes"]:
            if routes_dir.exists():
                for subdir_behavior in routes_dir.glob("*/actors_behavior.json"):
                    candidates.append(subdir_behavior)
        for candidate in candidates:
            if candidate.exists():
                behavior_path = candidate
                break

    if not behavior_path or not behavior_path.exists():
        _CUSTOM_ACTOR_BEHAVIOR_CACHE = {}
        return _CUSTOM_ACTOR_BEHAVIOR_CACHE

    try:
        data = json.loads(behavior_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _CUSTOM_ACTOR_BEHAVIOR_CACHE = {}
        return _CUSTOM_ACTOR_BEHAVIOR_CACHE

    if isinstance(data, dict) and isinstance(data.get("behaviors"), list):
        entries = data["behaviors"]
    elif isinstance(data, list):
        entries = data
    else:
        entries = []

    by_route: Dict[str, List[Dict[str, object]]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        route_id = entry.get("route_id") or entry.get("route") or "*"
        by_route.setdefault(str(route_id), []).append(entry)

    _CUSTOM_ACTOR_BEHAVIOR_CACHE = by_route
    return _CUSTOM_ACTOR_BEHAVIOR_CACHE


def _load_ego_vehicle_models() -> Dict[int, str]:
    """
    Load ego vehicle models from the manifest.
    Returns a dict mapping ego index (0, 1, 2, ...) to vehicle model string.
    """
    global _EGO_VEHICLE_MODELS_CACHE  # pylint: disable=global-statement

    if _EGO_VEHICLE_MODELS_CACHE is not None:
        return _EGO_VEHICLE_MODELS_CACHE

    manifest_env = os.environ.get("CUSTOM_ACTOR_MANIFEST")
    if not manifest_env:
        _EGO_VEHICLE_MODELS_CACHE = {}
        return _EGO_VEHICLE_MODELS_CACHE

    manifest_path = Path(manifest_env).expanduser().resolve()
    if not manifest_path.exists():
        _EGO_VEHICLE_MODELS_CACHE = {}
        return _EGO_VEHICLE_MODELS_CACHE

    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _EGO_VEHICLE_MODELS_CACHE = {}
        return _EGO_VEHICLE_MODELS_CACHE

    ego_models: Dict[int, str] = {}
    ego_entries = manifest_data.get("ego", [])
    if isinstance(ego_entries, list):
        for entry in ego_entries:
            # Extract ego index from filename pattern like "town01_custom_ego_actor_1.xml"
            file_path = entry.get("file", "")
            model = entry.get("model")
            if model:
                # Try to extract index from filename
                import re
                match = re.search(r'_actor_(\d+)\.xml$', file_path)
                if match:
                    ego_idx = int(match.group(1))
                    ego_models[ego_idx] = model

    _EGO_VEHICLE_MODELS_CACHE = ego_models
    return _EGO_VEHICLE_MODELS_CACHE


def get_ego_vehicle_model(ego_index: int, default: str = "vehicle.lincoln.mkz2017") -> str:
    """
    Get the vehicle model for a specific ego vehicle index.
    
    Args:
        ego_index: The ego vehicle index (0, 1, 2, ...)
        default: Default model if not specified in manifest
        
    Returns:
        The CARLA vehicle blueprint ID (e.g., 'vehicle.kawasaki.ninja')
    """
    ego_models = _load_ego_vehicle_models()
    return ego_models.get(ego_index, default)


def _get_default_actor_speed() -> float:
    """
    Obtain the default speed for custom actors from the environment, fallback to 8.0 m/s.
    """
    default_speed = os.environ.get("CUSTOM_ACTOR_DEFAULT_SPEED", "8.0")
    try:
        return max(0.0, float(default_speed))
    except ValueError:
        return 8.0


def _build_custom_actor_configs(route_id: str, town: str) -> List[Dict[str, object]]:
    """
    Convert manifest entries for a route into configuration dictionaries.
    """
    manifest = _load_custom_actor_manifest()
    if not manifest:
        return []

    entries = manifest.get(str(route_id), [])
    if not entries:
        return []

    actor_configs: List[Dict[str, object]] = []
    global_default_speed = _get_default_actor_speed()

    for entry in entries:
        town_filter = entry.get("town")
        if town_filter and town_filter != town:
            continue

        xml_path: Path = entry["path"]
        if not xml_path.exists():
            continue

        try:
            xml_root = ET.parse(str(xml_path)).getroot()
        except ET.ParseError:
            continue

        route_node = xml_root.find("route")
        if route_node is None:
            continue

        plan_locations: List[carla.Location] = []
        plan_transforms: List[carla.Transform] = []
        plan_time_candidates: List[float | None] = []
        plan_speed_candidates: List[float | None] = []
        spawn_transform = None
        # Optional opt-out: route attribute snap_to_road="false" disables road snapping for this actor
        snap_attr = str(route_node.attrib.get("snap_to_road", "true")).lower()
        snap_to_road = snap_attr not in ("false", "0", "no", "off")
        # Optional spawn-only override: allows lane-snapped spawn while keeping authored plan handling.
        snap_spawn_attr = route_node.attrib.get("snap_spawn_to_road")
        if snap_spawn_attr is None:
            snap_spawn_to_road = snap_to_road
        else:
            snap_spawn_to_road = str(snap_spawn_attr).lower() not in ("false", "0", "no", "off")

        for index, waypoint in enumerate(route_node.iter("waypoint")):
            try:
                loc = carla.Location(
                    x=float(waypoint.attrib.get("x", 0.0)),
                    y=float(waypoint.attrib.get("y", 0.0)),
                    z=float(waypoint.attrib.get("z", 0.0)),
                )
            except (TypeError, ValueError):
                continue

            plan_locations.append(loc)
            yaw = _safe_float(waypoint.attrib.get("yaw")) or 0.0
            pitch = _safe_float(waypoint.attrib.get("pitch")) or 0.0
            roll = _safe_float(waypoint.attrib.get("roll")) or 0.0
            plan_transforms.append(carla.Transform(loc, carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)))
            plan_time_candidates.append(_safe_float(waypoint.attrib.get("time") or waypoint.attrib.get("t")))
            plan_speed_candidates.append(_safe_float(waypoint.attrib.get("speed")))

            if index == 0:
                spawn_transform = plan_transforms[-1]

        if not plan_locations or spawn_transform is None:
            continue

        plan_times = None
        if plan_time_candidates and all(t is not None for t in plan_time_candidates):
            plan_times = [float(t) for t in plan_time_candidates]

        plan_speeds = None
        if plan_speed_candidates and all(s is not None for s in plan_speed_candidates):
            plan_speeds = [float(s) for s in plan_speed_candidates]

        role = (entry.get("kind") or entry.get("role") or "npc").lower()
        role_defaults = ROLE_DEFAULTS.get(role, {})

        route_control_mode = str(route_node.attrib.get("control_mode", "")).strip().lower()
        manifest_control_mode = str(entry.get("control_mode", "")).strip().lower()
        if route_control_mode in ("policy", "replay"):
            control_mode = route_control_mode
        elif manifest_control_mode in ("policy", "replay"):
            control_mode = manifest_control_mode
        elif role in ("static", "static_prop"):
            control_mode = "replay"
        else:
            control_mode = "policy"

        default_model = entry.get("model") or role_defaults.get("model") or "vehicle.*"
        try:
            target_speed = float(entry.get("speed", role_defaults.get("speed", global_default_speed)))
        except (TypeError, ValueError):
            target_speed = float(role_defaults.get("speed", global_default_speed))

        actor_configs.append(
            {
                "name": entry["name"],
                "rolename": entry["name"],
                "role": role,
                "model": default_model,
                "spawn_transform": spawn_transform,
                "plan": plan_locations,
                "plan_transforms": plan_transforms,
                "plan_times": plan_times,
                "plan_speeds": plan_speeds,
                "target_speed": target_speed,
                "avoid_collision": entry.get("avoid_collision", False),
                "snap_to_road": snap_to_road,
                "snap_spawn_to_road": snap_spawn_to_road,
                "control_mode": control_mode,
            }
        )

    return actor_configs

# for loading predefined weathers while parsing routes
WEATHERS = {
        '1': carla.WeatherParameters.ClearNoon,
        '2': carla.WeatherParameters.ClearSunset,
        '3': carla.WeatherParameters.CloudyNoon,
        '4': carla.WeatherParameters.CloudySunset,
        '5': carla.WeatherParameters.WetNoon,
        '6': carla.WeatherParameters.WetSunset,
        '7': carla.WeatherParameters.MidRainyNoon,
        '8': carla.WeatherParameters.MidRainSunset,
        '9': carla.WeatherParameters.WetCloudyNoon,
        '10': carla.WeatherParameters.WetCloudySunset,
        '11': carla.WeatherParameters.HardRainNoon,
        '12': carla.WeatherParameters.HardRainSunset,
        '13': carla.WeatherParameters.SoftRainNoon,
        '14': carla.WeatherParameters.SoftRainSunset,
}


class RouteParser(object):

    """
    Pure static class used to parse all the route and scenario configuration parameters.
    """

    @staticmethod
    def parse_annotations_file(annotation_filename):
        """
        Return the annotations of which positions where the scenarios are going to happen.
        :param annotation_filename: the filename for the anotations file
        :return:
        """
        with open(annotation_filename, 'r') as f:
            annotation_dict = json.loads(f.read(), object_pairs_hook=OrderedDict)

        final_dict = OrderedDict()

        for town_dict in annotation_dict['available_scenarios']:
            final_dict.update(town_dict)

        return final_dict  # the file has a current maps name that is an one element vec

    @staticmethod
    def parse_routes_file(route_filename, scenario_file, single_route=None):
        """
        Returns a list of route elements.
        :param route_filename: the path to a set of routes.
        :param single_route: If set, only this route shall be returned
        :return: List of dicts containing the waypoints, id and town of the routes
        """

        list_route_descriptions = []
        tree = ET.parse(route_filename)
        for route in tree.iter("route"):

            route_id = route.attrib['id']
            if single_route and route_id != single_route:
                continue

            new_config = RouteScenarioConfiguration()
            new_config.town = route.attrib['town']
            new_config.name = "RouteScenario_{}".format(route_id)
            new_config.weather = RouteParser.parse_weather(route) # default: parse_weather(route)
            new_config.scenario_file = scenario_file

            waypoint_list = []  # the list of waypoints that can be found on this route
            waypoint_yaws: List[float | None] = []
            waypoint_pitches: List[float | None] = []
            waypoint_rolls: List[float | None] = []
            waypoint_times: List[float | None] = []
            for waypoint in route.iter('waypoint'):
                waypoint_list.append(carla.Location(x=float(waypoint.attrib['x']),
                                                    y=float(waypoint.attrib['y']),
                                                    z=float(waypoint.attrib['z'])))
                waypoint_yaws.append(_safe_float(waypoint.attrib.get('yaw')))
                waypoint_pitches.append(_safe_float(waypoint.attrib.get('pitch')))
                waypoint_rolls.append(_safe_float(waypoint.attrib.get('roll')))
                waypoint_times.append(_safe_float(waypoint.attrib.get('time') or waypoint.attrib.get('t')))

            new_config.trajectory = waypoint_list
            new_config.trajectory_yaws = waypoint_yaws
            new_config.trajectory_pitches = waypoint_pitches
            new_config.trajectory_rolls = waypoint_rolls
            new_config.trajectory_times = waypoint_times
            new_config.custom_actors = _build_custom_actor_configs(route_id, new_config.town)
            behaviors_map = _load_custom_actor_behaviors()
            route_behaviors = list(behaviors_map.get(str(route_id), []))
            route_behaviors += list(behaviors_map.get("*", []))
            new_config.custom_actor_behaviors = route_behaviors

            list_route_descriptions.append(new_config)

        return list_route_descriptions

    @staticmethod
    def parse_weather(route):
        """
        Returns a carla.WeatherParameters with the corresponding weather for that route. If the route
        has no weather attribute, the default one is triggered.
        """

        route_weather = route.find("weather")

        if route_weather is None:

            weather = carla.WeatherParameters(sun_altitude_angle=70, cloudiness=30)

        else:
            weather = carla.WeatherParameters()
            for weather_attrib in route.iter("weather"):

                if 'cloudiness' in weather_attrib.attrib:
                    weather.cloudiness = float(weather_attrib.attrib['cloudiness']) 
                if 'precipitation' in weather_attrib.attrib:
                    weather.precipitation = float(weather_attrib.attrib['precipitation'])
                if 'precipitation_deposits' in weather_attrib.attrib:
                    weather.precipitation_deposits =float(weather_attrib.attrib['precipitation_deposits'])
                if 'wind_intensity' in weather_attrib.attrib:
                    weather.wind_intensity = float(weather_attrib.attrib['wind_intensity'])
                if 'sun_azimuth_angle' in weather_attrib.attrib:
                    weather.sun_azimuth_angle = float(weather_attrib.attrib['sun_azimuth_angle'])
                if 'sun_altitude_angle' in weather_attrib.attrib:
                    weather.sun_altitude_angle = float(weather_attrib.attrib['sun_altitude_angle'])
                if 'wetness' in weather_attrib.attrib:
                    weather.wetness = float(weather_attrib.attrib['wetness'])
                if 'fog_distance' in weather_attrib.attrib:
                    weather.fog_distance = float(weather_attrib.attrib['fog_distance'])
                if 'fog_density' in weather_attrib.attrib:
                    weather.fog_density = float(weather_attrib.attrib['fog_density'])
                if 'fog_falloff' in weather_attrib.attrib:
                    weather.fog_falloff = float(weather_attrib.attrib['fog_falloff'])

        return weather

    @staticmethod
    def parse_preset_weather(route):
        """
        Returns one of the 14 preset weather condition. If the route
        has no weather attribute, the default one is triggered.
        """

        if 'weather' not in route.attrib:
            weather = carla.WeatherParameters(sun_altitude_angle=70, cloudiness=30)
        else:
            weather = WEATHERS[route.attrib['weather']]

        return weather

    @staticmethod
    def check_trigger_position(new_trigger: OrderedDict, existing_triggers: OrderedDict):
        """
        Check if this trigger position already exists or if it is a new one.
        :param new_trigger:
        :param existing_triggers:
        :return:
        """

        for trigger_id in existing_triggers.keys():
            trigger = existing_triggers[trigger_id]
            dx = trigger['x'] - new_trigger['x']
            dy = trigger['y'] - new_trigger['y']
            distance = math.sqrt(dx * dx + dy * dy)

            dyaw = (trigger['yaw'] - new_trigger['yaw']) % 360
            if distance < TRIGGER_THRESHOLD \
                and (dyaw < TRIGGER_ANGLE_THRESHOLD or dyaw > (360 - TRIGGER_ANGLE_THRESHOLD)):
                return trigger_id

        return None

    @staticmethod
    def convert_waypoint_float(waypoint):
        """
        Convert waypoint values to float
        """
        waypoint['x'] = float(waypoint['x'])
        waypoint['y'] = float(waypoint['y'])
        waypoint['z'] = float(waypoint['z'])
        waypoint['yaw'] = float(waypoint['yaw'])

    @staticmethod
    def match_world_location_to_route(world_location: OrderedDict, route_description: List):
        """
        We match this location to a given route.
            world_location: trigger point
            route_description: list of waypoints
        Return:
            The first waypoint that is close enough to the trigger point or None
        """
        def match_waypoints(waypoint1, wtransform):
            """
            Check if waypoint1 and wtransform are similar
            """
            dx = float(waypoint1['x']) - wtransform.location.x
            dy = float(waypoint1['y']) - wtransform.location.y
            dz = float(waypoint1['z']) - wtransform.location.z
            dpos = math.sqrt(dx * dx + dy * dy + dz * dz)

            dyaw = (float(waypoint1['yaw']) - wtransform.rotation.yaw) % 360

            return dpos < TRIGGER_THRESHOLD \
                and (dyaw < TRIGGER_ANGLE_THRESHOLD or dyaw > (360 - TRIGGER_ANGLE_THRESHOLD))

        match_position = 0
        # TODO this function can be optimized to run on Log(N) time
        for route_waypoint in route_description:
            if match_waypoints(world_location, route_waypoint[0]):
                return match_position
            match_position += 1

        return None

    @staticmethod
    def get_scenario_type(scenario, match_position, trajectory):
        """
        Some scenarios have different types depending on the route.
        :param scenario: the scenario name
        :param match_position: the matching position for the scenarion
        :param trajectory: the route trajectory the ego is following
        :return: tag representing this subtype

        Also used to check which are not viable (Such as an scenario
        that triggers when turning but the route doesnt')
        WARNING: These tags are used at:
            - VehicleTurningRoute
            - SignalJunctionCrossingRoute
        and changes to these tags will affect them
        """

        def check_this_waypoint(tuple_wp_turn):
            """
            Decides whether or not the waypoint will define the scenario behavior
            """
            if RoadOption.LANEFOLLOW == tuple_wp_turn[1]:
                return False
            elif RoadOption.CHANGELANELEFT == tuple_wp_turn[1]:
                return False
            elif RoadOption.CHANGELANERIGHT == tuple_wp_turn[1]:
                return False
            return True

        # Unused tag for the rest of scenarios,
        # can't be None as they are still valid scenarios
        subtype = 'valid'

        if scenario == 'Scenario4':
            for tuple_wp_turn in trajectory[match_position:]:
                if check_this_waypoint(tuple_wp_turn):
                    if RoadOption.LEFT == tuple_wp_turn[1]:
                        subtype = 'S4left'
                    elif RoadOption.RIGHT == tuple_wp_turn[1]:
                        subtype = 'S4right'
                    else:
                        subtype = None
                    break  # Avoid checking all of them
                subtype = None

        if scenario == 'Scenario7':
            for tuple_wp_turn in trajectory[match_position:]:
                if check_this_waypoint(tuple_wp_turn):
                    if RoadOption.LEFT == tuple_wp_turn[1]:
                        subtype = 'S7left'
                    elif RoadOption.RIGHT == tuple_wp_turn[1]:
                        subtype = 'S7right'
                    elif RoadOption.STRAIGHT == tuple_wp_turn[1]:
                        subtype = 'S7opposite'
                    else:
                        subtype = None
                    break  # Avoid checking all of them
                subtype = None

        if scenario == 'Scenario8':
            for tuple_wp_turn in trajectory[match_position:]:
                if check_this_waypoint(tuple_wp_turn):
                    if RoadOption.LEFT == tuple_wp_turn[1]:
                        subtype = 'S8left'
                    else:
                        subtype = None
                    break  # Avoid checking all of them
                subtype = None

        if scenario == 'Scenario9':
            for tuple_wp_turn in trajectory[match_position:]:
                if check_this_waypoint(tuple_wp_turn):
                    if RoadOption.RIGHT == tuple_wp_turn[1]:
                        subtype = 'S9right'
                    else:
                        subtype = None
                    break  # Avoid checking all of them
                subtype = None

        return subtype

    @staticmethod
    def scan_route_for_scenarios(route_name, trajectory, world_annotations):
        """
        Just returns a plain list of possible scenarios that can happen in this route by matching
        the locations from the scenario into the route description
        Args:
            route_name: the town that route belongs to
            trajectory: list of fine grained waypoints location
            world_annotations: every possible scenario trigger point
        Return:  
            A list of scenario definitions with their correspondent parameters
            possible_scenarios: OrderedDict, len(possible_scenarios) is the number of position that is possible to trigger scenarios along this route, 
                                and possible_scenarios[i] is a list of scenarios that is possible to be triggered at the ith position
        """

        # the triggers dictionaries:
        existent_triggers = OrderedDict()
        # We have a table of IDs and trigger positions associated
        possible_scenarios = OrderedDict()

        # Keep track of the trigger ids being added
        latest_trigger_id = 0

        for town_name in world_annotations.keys():
            if town_name != route_name:
                continue

            scenarios = world_annotations[town_name]
            for scenario in scenarios:  # For each existent scenario
                scenario_name = scenario["scenario_type"]
                print('load all the {}'.format(scenario_name))
                for event in scenario["available_event_configurations"]:
                    waypoint = event['transform']  # trigger point of this scenario
                    RouteParser.convert_waypoint_float(waypoint)
                    # We match trigger point to the  route, now we need to check if the route affects
                    match_position = RouteParser.match_world_location_to_route(
                        waypoint, trajectory)
                    if match_position is not None:
                        # We match a location for this scenario, create a scenario object so this scenario
                        # can be instantiated later

                        if 'other_actors' in event:
                            other_vehicles = event['other_actors']
                        else:
                            other_vehicles = None
                        scenario_subtype = RouteParser.get_scenario_type(scenario_name, match_position,
                                                                         trajectory)
                        if scenario_subtype is None:
                            continue
                        scenario_description = {
                            'name': scenario_name,
                            'other_actors': other_vehicles,
                            'trigger_position': waypoint,
                            'scenario_type': scenario_subtype, # some scenarios have route dependent configurations
                        }

                        trigger_id = RouteParser.check_trigger_position(waypoint, existent_triggers)
                        if trigger_id is None:
                            # This trigger does not exist create a new reference on existent triggers
                            existent_triggers.update({latest_trigger_id: waypoint})
                            # Update a reference for this trigger on the possible scenarios
                            possible_scenarios.update({latest_trigger_id: []})
                            trigger_id = latest_trigger_id
                            # Increment the latest trigger
                            latest_trigger_id += 1

                        possible_scenarios[trigger_id].append(scenario_description)

        return possible_scenarios, existent_triggers
