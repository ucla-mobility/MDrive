#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Wrapper for autonomous agents required for tracking and checking of used sensors
"""

from __future__ import print_function
import math
import os
import time

import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from leaderboard.envs.sensor_interface import CallBack, OpenDriveMapReader, SpeedometerReader, SensorConfigurationInvalid
from leaderboard.autoagents.autonomous_agent import Track
import json
import numpy as np

try:
    from common.carla_connection_events import log_sensor_event as _log_sensor_event
except Exception:  # pragma: no cover - keep runnable without repo root on PYTHONPATH
    def _log_sensor_event(event, **kwargs):  # type: ignore[misc]
        pass

MAX_ALLOWED_RADIUS_SENSOR = 1000.0  # increased for topdown map generation

SENSORS_LIMITS = {
    'sensor.camera.rgb': 10,
    'sensor.camera.semantic_segmentation': 10,
    'sensor.camera.depth': 10,
    'sensor.lidar.ray_cast': 1,
    'sensor.lidar.ray_cast_semantic': 1,
    'sensor.other.radar': 2,
    'sensor.other.collision': 1,
    'sensor.other.gnss': 1,
    'sensor.other.imu': 1,
    'sensor.opendrive_map': 1,
    'sensor.speedometer': 1
}

def get_camera_intrinsic(sensor):
    """
    Retrieve the camera intrinsic matrix.
    Parameters
    ----------
    sensor : carla.sensor
        Carla rgb camera object.
    Returns
    -------
    matrix_x : list
        The 2d intrinsic matrix.
    """
    VIEW_WIDTH = int(sensor.attributes['image_size_x'])
    VIEW_HEIGHT = int(sensor.attributes['image_size_y'])
    VIEW_FOV = int(float(sensor.attributes['fov']))

    matrix_k = np.identity(3)
    matrix_k[0, 2] = VIEW_WIDTH / 2.0
    matrix_k[1, 2] = VIEW_HEIGHT / 2.0
    matrix_k[0, 0] = matrix_k[1, 1] = VIEW_WIDTH / \
        (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))

    return matrix_k.tolist()


def _safe_set_bp_attribute(bp, key, value):
    """Set blueprint attribute only when available on this CARLA build."""
    try:
        if bp.has_attribute(key):
            bp.set_attribute(key, str(value))
    except Exception:  # pylint: disable=broad-except
        pass

class AgentError(Exception):
    """
    Exceptions thrown when the agent returns an error during the simulation
    """

    def __init__(self, message):
        super(AgentError, self).__init__(message)


class AgentWrapper(object):

    """
    Wrapper for autonomous agents required for tracking and checking of used sensors
    This class defines the params of each type of sensor
    """

    allowed_sensors = [
        'sensor.opendrive_map',
        'sensor.speedometer',
        'sensor.camera.rgb',
        'sensor.camera.semantic_segmentation',
        'sensor.camera.depth',
        'sensor.camera',
        'sensor.lidar.ray_cast',
        'sensor.lidar.ray_cast_semantic',
        'sensor.other.radar',
        'sensor.other.collision',
        'sensor.other.gnss',
        'sensor.other.imu'
    ]

    _agent = None

    def __init__(self, agent):
        """
        Set the autonomous agent
        """
        self._agent = agent
        self.ego_vehicles_num = agent.ego_vehicles_num
        self._sensors_list = [[] for _ in range(self.ego_vehicles_num)]

    def __call__(self):
        """
        Pass the call directly to the agent
        """
        return self._agent()

    def setup_sensors(self, vehicle, vehicle_num = 0, save_root=None, debug_mode=False):
        """
        Create the sensors defined by the user and attach them to the ego-vehicle
        Args:
            vehicle: ego vehicle, Carla Vehicle instance
            vehicle_num: id of vehicle
        
        """
        sensor_param={}
        bp_library = CarlaDataProvider.get_world().get_blueprint_library()
        dense_capture = os.environ.get("TCP_CAPTURE_SENSOR_FRAMES", "").lower() in ("1", "true", "yes")
        for sensor_spec in self._agent.sensors():
            # These are the pseudosensors (not spawned)
            if sensor_spec['type'].startswith('sensor.opendrive_map'):
                # The HDMap pseudo sensor is created directly here
                sensor = OpenDriveMapReader(vehicle, sensor_spec['reading_frequency'])
            elif sensor_spec['type'].startswith('sensor.speedometer'):
                delta_time = CarlaDataProvider.get_world().get_settings().fixed_delta_seconds
                frame_rate = 1 / delta_time
                sensor = SpeedometerReader(vehicle, frame_rate)
            # These are the sensors spawned on the carla world
            else:
                bp = bp_library.find(str(sensor_spec['type']))
                if sensor_spec['type'].startswith('sensor.camera.semantic_segmentation'):
                    bp.set_attribute('image_size_x', str(sensor_spec['width']))
                    bp.set_attribute('image_size_y', str(sensor_spec['height']))
                    bp.set_attribute('fov', str(sensor_spec['fov']))

                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                elif sensor_spec['type'].startswith('sensor.camera.depth'):
                    bp.set_attribute('image_size_x', str(sensor_spec['width']))
                    bp.set_attribute('image_size_y', str(sensor_spec['height']))
                    bp.set_attribute('fov', str(sensor_spec['fov']))

                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                elif sensor_spec['type'].startswith('sensor.camera'):
                    bp.set_attribute('image_size_x', str(sensor_spec['width']))
                    bp.set_attribute('image_size_y', str(sensor_spec['height']))
                    bp.set_attribute('fov', str(sensor_spec['fov']))
                    is_logreplay_rgb = sensor_spec.get('id') == 'logreplay_rgb'
                    if is_logreplay_rgb:
                        # Keep log-replay image captures visually clean while preserving
                        # existing post-processing for planner-facing cameras.
                        _safe_set_bp_attribute(bp, 'chromatic_aberration_intensity', 0.0)
                        _safe_set_bp_attribute(bp, 'chromatic_aberration_offset', 0.0)
                        _safe_set_bp_attribute(bp, 'lens_circle_multiplier', 0.0)
                        _safe_set_bp_attribute(bp, 'lens_circle_falloff', 0.0)
                        _safe_set_bp_attribute(bp, 'lens_k', 0.0)
                        _safe_set_bp_attribute(bp, 'lens_kcube', 0.0)
                    else:
                        bp.set_attribute('chromatic_aberration_intensity', str(0.5))
                        bp.set_attribute('chromatic_aberration_offset', str(0))

                        if self._agent.agent_name in ['TCP','transfuser','LAV','interfuser','WOR']:
                            bp.set_attribute('lens_circle_multiplier', str(3.0))
                            bp.set_attribute('lens_circle_falloff', str(3.0))    
                        else:
                            bp.set_attribute('lens_circle_multiplier', str(1.0))
                            bp.set_attribute('lens_circle_falloff', str(1.0))                                        

                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                elif sensor_spec['type'].startswith('sensor.lidar.ray_cast_semantic'):
                    # Honor sensor_spec values if the agent provided them so the
                    # visibility-filter ray pattern can be matched to the main
                    # perception LiDAR (openloop AP protocol). Falls back to the
                    # legacy hardcoded defaults when the spec is silent.
                    bp.set_attribute('range', str(sensor_spec.get('range', 85)))
                    bp.set_attribute('rotation_frequency', str(sensor_spec.get('rotation_frequency', 20)))
                    bp.set_attribute('channels', str(sensor_spec.get('channels', 32)))
                    bp.set_attribute('upper_fov', str(sensor_spec.get('upper_fov', 10)))
                    bp.set_attribute('lower_fov', str(sensor_spec.get('lower_fov', -30)))
                    bp.set_attribute('points_per_second', str(sensor_spec.get('points_per_second', 250000)))
                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                elif sensor_spec['type'].startswith('sensor.lidar'):
                    bp.set_attribute('range', str(85))
                    bp.set_attribute('rotation_frequency', str(20)) # default: 10, change to 20 to generate 360 degree LiDAR point cloud
                    bp.set_attribute('channels', str(64)) # 64
                    bp.set_attribute('upper_fov', str(10))
                    bp.set_attribute('lower_fov', str(-30))
                    bp.set_attribute('points_per_second', str(500000)) # 500000
                    bp.set_attribute('atmosphere_attenuation_rate', str(0.004))
                    bp.set_attribute('dropoff_general_rate', str(0)) ##### NOTE: changed, 0.45 -> 0.0
                    bp.set_attribute('dropoff_intensity_limit', str(0.8))
                    bp.set_attribute('dropoff_zero_intensity', str(0.4))

                    if self._agent.agent_name in ['TCP','LAV','transfuser','interfuser','WOR']:
                        bp.set_attribute('rotation_frequency', str(10))
                        bp.set_attribute('points_per_second', str(600000))
                        bp.set_attribute('dropoff_general_rate', str(0.45))

                    if self._agent.agent_name in ['lmdrive']:
                        bp.set_attribute('range', str(85))
                        bp.set_attribute('rotation_frequency', str(10)) # default: 10, change to 20 to generate 360 degree LiDAR point cloud
                        bp.set_attribute('channels', str(64))
                        bp.set_attribute('upper_fov', str(10))
                        bp.set_attribute('lower_fov', str(-30))
                        bp.set_attribute('points_per_second', str(600000))
                        bp.set_attribute('atmosphere_attenuation_rate', str(0.004))
                        bp.set_attribute('dropoff_general_rate', str(0.45))
                        bp.set_attribute('dropoff_intensity_limit', str(0.8))
                        bp.set_attribute('dropoff_zero_intensity', str(0.4))                        

                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                elif sensor_spec['type'].startswith('sensor.other.radar'):
                    bp.set_attribute('horizontal_fov', str(sensor_spec['fov']))  # degrees
                    bp.set_attribute('vertical_fov', str(sensor_spec['fov']))  # degrees
                    bp.set_attribute('points_per_second', '1500')
                    bp.set_attribute('range', '100')  # meters

                    sensor_location = carla.Location(x=sensor_spec['x'],
                                                     y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])

                elif sensor_spec['type'].startswith('sensor.other.collision'):
                    sensor_location = carla.Location(x=sensor_spec['x'],
                                                     y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])

                elif sensor_spec['type'].startswith('sensor.other.gnss'):
                    # bp.set_attribute('noise_alt_stddev', str(0.000005))
                    # bp.set_attribute('noise_lat_stddev', str(0.000005))
                    # bp.set_attribute('noise_lon_stddev', str(0.000005))
                    bp.set_attribute('noise_alt_bias', str(0.0))
                    bp.set_attribute('noise_lat_bias', str(0.0))
                    bp.set_attribute('noise_lon_bias', str(0.0))

                    sensor_location = carla.Location(x=sensor_spec['x'],
                                                     y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation()

                elif sensor_spec['type'].startswith('sensor.other.imu'):
                    bp.set_attribute('noise_accel_stddev_x', str(0.001))
                    bp.set_attribute('noise_accel_stddev_y', str(0.001))
                    bp.set_attribute('noise_accel_stddev_z', str(0.015))
                    bp.set_attribute('noise_gyro_stddev_x', str(0.001))
                    bp.set_attribute('noise_gyro_stddev_y', str(0.001))
                    bp.set_attribute('noise_gyro_stddev_z', str(0.001))

                    sensor_location = carla.Location(x=sensor_spec['x'],
                                                     y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                # create sensor
                sensor_transform = carla.Transform(sensor_location, sensor_rotation)
                sensor = CarlaDataProvider.get_world().spawn_actor(bp, sensor_transform, vehicle)
            # log sensor spawn for forensic lifecycle tracking
            try:
                _log_sensor_event(
                    "SENSOR_SPAWN",
                    sensor=sensor,
                    sensor_type=sensor_spec.get('type', ''),
                    sensor_tag=sensor_spec.get('id', '') + "_{}".format(vehicle_num),
                )
            except Exception:
                pass
            # setup callback
            sensor_tag = sensor_spec['id']+"_{}".format(vehicle_num)
            sensor.listen(CallBack(sensor_tag, sensor_spec['type'], sensor, self._agent.sensor_interface))
            if (
                dense_capture
                and sensor_spec['type'].startswith('sensor.camera')
                and hasattr(self._agent.sensor_interface, "configure_image_capture")
                and hasattr(self._agent, "save_path")
                and self._agent.save_path is not None
            ):
                capture_root = getattr(self._agent, "logreplayimages_path", None)
                if capture_root is None:
                    capture_root = getattr(self._agent, "calibration_path", None)
                if capture_root is None:
                    capture_root = self._agent.save_path
                out_dir = os.path.join(str(capture_root), f"{sensor_spec['id']}_{vehicle_num}")
                self._agent.sensor_interface.configure_image_capture(sensor_tag, out_dir)
            self._sensors_list[vehicle_num].append(sensor)

        # Hook for stationary roadside / infrastructure sensors that aren't
        # attached to a vehicle. Called BEFORE the priming world.tick() below
        # so the new sensors' first-frame data is published at the same tick
        # as the ego sensors. Idempotent — agents implement
        # setup_infra_sensors() to be a no-op on repeat calls (one per ego).
        try:
            _hook = getattr(self._agent, "setup_infra_sensors", None)
            if callable(_hook):
                _hook()
        except Exception as _exc:
            print(f"[agent_wrapper] setup_infra_sensors hook failed: {_exc}")

        # Tick once to spawn the sensors
        CarlaDataProvider.get_world().tick()


    @staticmethod
    def validate_sensor_configuration(sensors, agent_track, selected_track):
        """
        Ensure that the sensor configuration is valid, in case the challenge mode is used
        Returns true on valid configuration, false otherwise
        """
        if Track(selected_track) != agent_track:
            raise SensorConfigurationInvalid("You are submitting to the wrong track [{}]!".format(Track(selected_track)))

        sensor_count = {}
        sensor_ids = []

        for sensor in sensors:

            # Check if the is has been already used
            sensor_id = sensor['id']
            if sensor_id in sensor_ids:
                raise SensorConfigurationInvalid("Duplicated sensor tag [{}]".format(sensor_id))
            else:
                sensor_ids.append(sensor_id)

            # Check if the sensor is valid
            if agent_track == Track.SENSORS:
                if sensor['type'].startswith('sensor.opendrive_map'):
                    raise SensorConfigurationInvalid("Illegal sensor used for Track [{}]!".format(agent_track))

            # Check the sensors validity
            if sensor['type'] not in AgentWrapper.allowed_sensors:
                raise SensorConfigurationInvalid("Illegal sensor used. {} are not allowed!".format(sensor['type']))

            # Check the extrinsics of the sensor
            if 'x' in sensor and 'y' in sensor and 'z' in sensor:
                if math.sqrt(sensor['x']**2 + sensor['y']**2 + sensor['z']**2) > MAX_ALLOWED_RADIUS_SENSOR:
                    raise SensorConfigurationInvalid(
                        "Illegal sensor extrinsics used for Track [{}]!".format(agent_track))

            # Check the amount of sensors
            if sensor['type'] in sensor_count:
                sensor_count[sensor['type']] += 1
            else:
                sensor_count[sensor['type']] = 1


        for sensor_type, max_instances_allowed in SENSORS_LIMITS.items():
            if sensor_type in sensor_count and sensor_count[sensor_type] > max_instances_allowed:
                raise SensorConfigurationInvalid(
                    "Too many {} used! "
                    "Maximum number allowed is {}, but {} were requested.".format(sensor_type,
                                                                                  max_instances_allowed,
                                                                                  sensor_count[sensor_type]))
    
    def get_save_frame(self):
        if hasattr(self._agent, "get_save_frame"):
            return self._agent.get_save_frame()
        return None
    
    def del_ego_sensor(self, vehicle_num = 0):
        """
        Remove sensor tags of one ego vehicle
        """
        sensor_interface = getattr(self._agent, "sensor_interface", None)
        if sensor_interface is None:
            return
        for sensor_spec in self._agent.sensors():
            sensor_tag = sensor_spec['id']+"_{}".format(vehicle_num)
            try:
                _log_sensor_event(
                    "SENSOR_LISTENER_DETACH",
                    sensor_tag=sensor_tag,
                    status="begin",
                )
            except Exception:
                pass
            if hasattr(sensor_interface, "unregister_image_capture"):
                sensor_interface.unregister_image_capture(sensor_tag)
            if hasattr(sensor_interface, "unregister_sensor"):
                sensor_interface.unregister_sensor(sensor_tag)
            else:
                sensor_interface._sensors_objects.pop(sensor_tag, None)
                if hasattr(sensor_interface, "_last_returned_timestamps"):
                    sensor_interface._last_returned_timestamps.pop(sensor_tag, None)
                if hasattr(sensor_interface, "_latest_data"):
                    sensor_interface._latest_data.pop(sensor_tag, None)
            try:
                _log_sensor_event(
                    "SENSOR_LISTENER_DETACH",
                    sensor_tag=sensor_tag,
                    status="end",
                )
            except Exception:
                pass

    @staticmethod
    def _sensor_actor_id(sensor):
        if sensor is None:
            return None
        try:
            sensor_id = getattr(sensor, "id", None)
            if sensor_id is None:
                return None
            return int(sensor_id)
        except Exception:
            return None

    def _cleanup_vehicle_sensors(self, vehicle_num, *, fast=False):
        if vehicle_num < 0 or vehicle_num >= len(self._sensors_list):
            return

        sensors_ego = list(self._sensors_list[vehicle_num] or [])
        if not sensors_ego:
            self.del_ego_sensor(vehicle_num)
            self._sensors_list[vehicle_num] = []
            return

        # Phase 1: unregister callbacks/listeners so late packets are dropped.
        self.del_ego_sensor(vehicle_num)

        # Phase 2: stop sensor streams before any actor destruction RPC.
        for idx, sensor in enumerate(sensors_ego):
            if sensor is None:
                continue
            try:
                _log_sensor_event(
                    "SENSOR_STOP_BEGIN",
                    sensor=sensor,
                    sensor_tag=f"ego{vehicle_num}:{idx}",
                )
            except Exception:
                pass
            stop_status = "ok"
            try:
                sensor.stop()
            except Exception:
                stop_status = "fail"
            try:
                _log_sensor_event(
                    "SENSOR_STOP_END",
                    sensor=sensor,
                    sensor_tag=f"ego{vehicle_num}:{idx}",
                    status=stop_status,
                )
            except Exception:
                pass

        # Phase 3: destroy CARLA-backed sensors with verified synchronous batch RPC.
        sensor_actor_ids = []
        sensor_actor_id_set = set()
        for sensor in sensors_ego:
            sensor_id = self._sensor_actor_id(sensor)
            if sensor_id is None:
                continue
            if sensor_id in sensor_actor_id_set:
                continue
            sensor_actor_id_set.add(sensor_id)
            sensor_actor_ids.append(sensor_id)

        if sensor_actor_ids:
            try:
                _log_sensor_event(
                    "SENSOR_DESTROY_BATCH_BEGIN",
                    sensor_tag=f"ego_{vehicle_num}",
                    actor_count=len(sensor_actor_ids),
                    actor_ids=",".join(str(i) for i in sensor_actor_ids[:20]),
                )
            except Exception:
                pass
        destroy_summary = CarlaDataProvider.destroy_actor_ids(
            sensor_actor_ids,
            reason=f"agent_wrapper_sensor_cleanup_vehicle_{vehicle_num}",
            timeout_s=0.5 if fast else 4.0,
            poll_s=0.02 if fast else 0.05,
            max_retries=0 if fast else 2,
            direct_fallback=not fast,
        )
        if sensor_actor_ids:
            try:
                _log_sensor_event(
                    "SENSOR_DESTROY_BATCH_RESULT",
                    sensor_tag=f"ego_{vehicle_num}",
                    actor_count=len(sensor_actor_ids),
                    destroyed_count=len(destroy_summary.get("destroyed_ids", [])),
                    already_gone_count=len(destroy_summary.get("already_gone_ids", [])),
                    failed_count=len(destroy_summary.get("failed_ids", [])),
                    remaining_count=len(destroy_summary.get("remaining_alive_ids", [])),
                    failed_ids=",".join(
                        str(i) for i in destroy_summary.get("failed_ids", [])[:20]
                    ),
                )
            except Exception:
                pass
        failed_actor_ids = set(destroy_summary.get("failed_ids", []))
        failed_actor_ids.update(destroy_summary.get("remaining_alive_ids", []))

        # Phase 4: direct destroy fallback for pseudo sensors; CARLA-backed failures
        # are deferred in fast mode to avoid aggressive mid-route RPC churn.
        for idx, sensor in enumerate(sensors_ego):
            if sensor is None:
                continue
            sensor_id = self._sensor_actor_id(sensor)
            destroyed_via_batch = sensor_id is not None and sensor_id not in failed_actor_ids
            is_carla_sensor = sensor_id is not None
            use_direct_destroy = not destroyed_via_batch and (
                (not is_carla_sensor) or (not fast)
            )
            try:
                _log_sensor_event(
                    "SENSOR_DESTROY",
                    sensor=sensor,
                    sensor_tag=f"ego{vehicle_num}:{idx}",
                    teardown_path=(
                        "batch_rpc"
                        if destroyed_via_batch
                        else ("direct" if use_direct_destroy else "deferred")
                    ),
                )
            except Exception:
                pass

            if use_direct_destroy:
                try:
                    sensor.destroy()
                except Exception:
                    pass
            self._sensors_list[vehicle_num][idx] = None

        self._sensors_list[vehicle_num] = []

    def cleanup(self, *, finalize_capture=True):
        """
        Remove and destroy all sensors
        """
        for vehicle_num, _ in enumerate(self._sensors_list):
            self._cleanup_vehicle_sensors(vehicle_num, fast=False)
        sensor_interface = getattr(self._agent, "sensor_interface", None)
        if finalize_capture and sensor_interface is not None and hasattr(sensor_interface, "finalize_image_capture"):
            sensor_interface.finalize_image_capture()
        self._sensors_list = [[] for _ in range(self.ego_vehicles_num)]

    def cleanup_single(self, vehicle_num = 0):
        """
        Remove and destroy all sensors
        """
        self._cleanup_vehicle_sensors(vehicle_num, fast=True)

    def cleanup_rsu(self, vehicle_num):
        if hasattr(self._agent,"clean_rsu_single"):
            self._agent.clean_rsu_single(vehicle_num)
