import copy
import logging
import numpy as np
import os
import time
from pathlib import Path
from threading import Lock, Thread

from queue import Queue
from queue import Empty
from queue import Full

import carla
import cv2
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime


def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.setDaemon(True)
        thread.start()

        return thread
    return wrapper


class SensorConfigurationInvalid(Exception):
    """
    Exceptions thrown when the sensors used by the agent are not allowed for that specific submissions
    """

    def __init__(self, message):
        super(SensorConfigurationInvalid, self).__init__(message)


class SensorReceivedNoData(Exception):
    """
    Exceptions thrown when the sensors used by the agent take too long to receive data
    """

    def __init__(self, message):
        super(SensorReceivedNoData, self).__init__(message)


class GenericMeasurement(object):
    def __init__(self, data, frame):
        self.data = data
        self.frame = frame


class BaseReader(object):
    def __init__(self, vehicle, reading_frequency=1.0):
        self._vehicle = vehicle
        self._reading_frequency = reading_frequency
        self._callback = None
        self._run_ps = True
        self.run()

    def __call__(self):
        pass

    @threaded
    def run(self):
        first_time = True
        latest_time = GameTime.get_time()
        while self._run_ps:
            if self._callback is not None:
                current_time = GameTime.get_time()

                # Second part forces the sensors to send data at the first tick, regardless of frequency
                if current_time - latest_time > (1 / self._reading_frequency) \
                        or (first_time and GameTime.get_frame() != 0):
                    self._callback(GenericMeasurement(self.__call__(), GameTime.get_frame()))
                    latest_time = GameTime.get_time()
                    first_time = False

                else:
                    time.sleep(0.001)

    def listen(self, callback):
        # Tell that this function receives what the producer does.
        self._callback = callback

    def stop(self):
        self._run_ps = False

    def destroy(self):
        self._run_ps = False


class SpeedometerReader(BaseReader):
    """
    Sensor to measure the speed of the vehicle.
    """
    MAX_CONNECTION_ATTEMPTS = 10

    def _get_forward_speed(self, transform=None, velocity=None, acc=None, vel_angle=None):
        """ Convert the vehicle transform directly to forward speed """
        if not velocity:
            velocity = self._vehicle.get_velocity()
        if not transform:
            transform = self._vehicle.get_transform()
        if not acc:
            acc=self._vehicle.get_acceleration()
        if not vel_angle:
            vel_angle = self._vehicle.get_angular_velocity()

        vel_list = [velocity.x, velocity.y, velocity.z]
        vel_np = np.array(vel_list)
        acc_np = [acc.x, acc.y, acc.z]
        vel_angle_np = [vel_angle.x, vel_angle.y, vel_angle.z]
        transform_dict = {'x':transform.location.x,
                          'y':transform.location.y,
                          'z':transform.location.z,
                          'roll':transform.rotation.roll,
                          'yaw':transform.rotation.yaw,
                          'pitch':transform.rotation.pitch}
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        speed = np.dot(vel_np, orientation)
        return speed, vel_list, acc_np, vel_angle_np, transform_dict

    def __call__(self):
        """ We convert the vehicle physics information into a convenient dictionary """

        # protect this access against timeout
        attempts = 0
        while attempts < self.MAX_CONNECTION_ATTEMPTS:
            try:
                velocity = self._vehicle.get_velocity()
                transform = self._vehicle.get_transform()
                acc=self._vehicle.get_acceleration()
                vel_angle = self._vehicle.get_angular_velocity()
                break
            except Exception:
                attempts += 1
                time.sleep(0.2)
                continue

        speed, vel_np, acc_np, vel_angle_np, transform_dict = self._get_forward_speed(transform=transform, velocity=velocity, acc=acc, vel_angle=vel_angle)

        return {'speed': speed,
                'move_state':{'speed': speed,
                    'speed_xyz': vel_np,
                    'acc_xyz': acc_np,
                    'speed_angle_xyz': vel_angle_np,
                    'transform_xyz_rollyawpitch':transform_dict}}


class OpenDriveMapReader(BaseReader):
    def __call__(self):
        return {'opendrive': CarlaDataProvider.get_map().to_opendrive()}


class CallBack(object):
    def __init__(self, tag, sensor_type, sensor, data_provider):
        self._tag = tag
        self._data_provider = data_provider

        self._data_provider.register_sensor(tag, sensor_type, sensor)

    def __call__(self, data):
        if isinstance(data, carla.libcarla.Image):
            self._parse_image_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.LidarMeasurement):
            self._parse_lidar_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.SemanticLidarMeasurement):
            self._parse_semantic_lidar_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.RadarMeasurement):
            self._parse_radar_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.GnssMeasurement):
            self._parse_gnss_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.IMUMeasurement):
            self._parse_imu_cb(data, self._tag)
        elif isinstance(data, GenericMeasurement):
            self._parse_pseudosensor(data, self._tag)
        else:
            logging.error('No callback method for this sensor.')

    # Parsing CARLA physical Sensors
    def _parse_image_cb(self, image, tag):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = copy.deepcopy(array)
        array = np.reshape(array, (image.height, image.width, 4))
        self._data_provider.maybe_save_image_frame(tag, array, image.frame)
        self._data_provider.update_sensor(tag, array, image.frame)

    def _parse_lidar_cb(self, lidar_data, tag):
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        self._data_provider.update_sensor(tag, points, lidar_data.frame)

    def _parse_semantic_lidar_cb(self, semantic_lidar_data, tag):
        points = np.frombuffer(semantic_lidar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 6), 6))
        self._data_provider.update_sensor(tag, points, semantic_lidar_data.frame)

    def _parse_radar_cb(self, radar_data, tag):
        # [depth, azimuth, altitute, velocity]
        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        points = np.flip(points, 1)
        self._data_provider.update_sensor(tag, points, radar_data.frame)

    def _parse_gnss_cb(self, gnss_data, tag):
        array = np.array([gnss_data.latitude,
                          gnss_data.longitude,
                          gnss_data.altitude], dtype=np.float64)
        self._data_provider.update_sensor(tag, array, gnss_data.frame)

    def _parse_imu_cb(self, imu_data, tag):
        array = np.array([imu_data.accelerometer.x,
                          imu_data.accelerometer.y,
                          imu_data.accelerometer.z,
                          imu_data.gyroscope.x,
                          imu_data.gyroscope.y,
                          imu_data.gyroscope.z,
                          imu_data.compass,
                         ], dtype=np.float64)
        self._data_provider.update_sensor(tag, array, imu_data.frame)

    def _parse_pseudosensor(self, package, tag):
        self._data_provider.update_sensor(tag, package.data, package.frame)


class SensorInterface(object):
    def __init__(self):
        self._sensors_objects = {}
        self._data_buffers = {}
        self._new_data_buffers = Queue()
        self._queue_timeout = 100 # default: 10
        self._last_returned_timestamps = {}

        # Only sensor that doesn't get the data on tick, needs special treatment
        self._opendrive_tag = None
        self._dense_capture_enabled = os.environ.get("TCP_CAPTURE_SENSOR_FRAMES", "").lower() in ("1", "true", "yes")
        self._dense_capture_targets = {}
        self._dense_capture_lock = Lock()
        self._dense_capture_queue = None
        self._dense_capture_writer_thread = None
        self._dense_capture_queue_max = 0
        self._dense_capture_png_compression = 1
        self._dense_capture_dropped = 0
        if self._dense_capture_enabled:
            try:
                self._dense_capture_queue_max = max(16, int(os.environ.get("TCP_CAPTURE_QUEUE_MAX", "256")))
            except (TypeError, ValueError):
                self._dense_capture_queue_max = 256
            try:
                self._dense_capture_png_compression = min(
                    9, max(0, int(os.environ.get("TCP_CAPTURE_PNG_COMPRESSION", "1")))
                )
            except (TypeError, ValueError):
                self._dense_capture_png_compression = 1
            self._dense_capture_queue = Queue(maxsize=self._dense_capture_queue_max)
            self._dense_capture_writer_thread = Thread(
                target=self._dense_capture_writer_loop,
                name="dense_image_writer",
                daemon=True,
            )
            self._dense_capture_writer_thread.start()

    def configure_image_capture(self, tag, output_dir):
        """
        Register a per-sensor image sink. When dense capture is enabled, every incoming
        frame for this sensor will be written to disk from the sensor callback thread.
        """
        if not self._dense_capture_enabled:
            return
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with self._dense_capture_lock:
            self._dense_capture_targets[tag] = {
                "dir": output_path,
                "counter": 0,
                "last_frame": None,
            }

    def unregister_image_capture(self, tag):
        if not self._dense_capture_enabled:
            return
        with self._dense_capture_lock:
            self._dense_capture_targets.pop(tag, None)

    def _dense_capture_writer_loop(self):
        while True:
            item = self._dense_capture_queue.get()
            if item is None:
                self._dense_capture_queue.task_done()
                break
            out_path, bgr = item
            try:
                cv2.imwrite(
                    str(out_path),
                    bgr,
                    [int(cv2.IMWRITE_PNG_COMPRESSION), self._dense_capture_png_compression],
                )
            except Exception:
                pass
            finally:
                self._dense_capture_queue.task_done()

    def finalize_image_capture(self):
        if not self._dense_capture_enabled or self._dense_capture_queue is None:
            return
        # Flush pending writes before cleanup.
        try:
            self._dense_capture_queue.join()
        except Exception:
            pass
        try:
            self._dense_capture_queue.put_nowait(None)
        except Exception:
            pass
        if self._dense_capture_writer_thread is not None:
            self._dense_capture_writer_thread.join(timeout=5.0)
        if self._dense_capture_dropped > 0:
            print(
                "[SensorInterface] Dense capture dropped {} frames (queue max={}).".format(
                    self._dense_capture_dropped, self._dense_capture_queue_max
                )
            )

    def maybe_save_image_frame(self, tag, image_bgra, frame):
        if not self._dense_capture_enabled:
            return
        if self._dense_capture_queue is None:
            return
        target = None
        with self._dense_capture_lock:
            target = self._dense_capture_targets.get(tag)
            if target is None:
                return
            if target["last_frame"] == frame:
                return
            out_path = target["dir"] / f"{target['counter']:06d}.png"
            target["counter"] += 1
            target["last_frame"] = frame
        try:
            # Callback payload is BGRA. Keep BGR and offload PNG encoding to writer thread.
            bgr = np.ascontiguousarray(image_bgra[:, :, :3])
            self._dense_capture_queue.put_nowait((out_path, bgr))
        except Full:
            self._dense_capture_dropped += 1
        except Exception:
            pass


    def register_sensor(self, tag, sensor_type, sensor):
        if tag in self._sensors_objects:
            raise SensorConfigurationInvalid("Duplicated sensor tag [{}]".format(tag))

        self._sensors_objects[tag] = sensor
        self._last_returned_timestamps[tag] = None

        if sensor_type == 'sensor.opendrive_map': 
            self._opendrive_tag = tag

    def update_sensor(self, tag, data, timestamp):
        # print("Updating {} - {}".format(tag, timestamp))
        if tag not in self._sensors_objects:
            # sensor may have been removed (e.g., ego destroyed); ignore late callbacks
            return

        self._new_data_buffers.put((tag, timestamp, data))

    def _store_latest_sample(self, data_dict, tag, timestamp, payload):
        """
        Keep the newest available packet for each tag.
        Reject packets older than the last packet we already returned for this tag
        to avoid one-frame rollbacks from out-of-order callback delivery.
        """
        if tag not in self._sensors_objects:
            return False

        last_returned = self._last_returned_timestamps.get(tag)
        if last_returned is not None and timestamp < last_returned:
            return False

        current = data_dict.get(tag)
        if current is not None and timestamp < current[0]:
            return False

        data_dict[tag] = (timestamp, payload)
        return True

    def get_data(self):
        try: 
            data_dict = {}
            while len(data_dict.keys()) < len(self._sensors_objects.keys()):

                # Don't wait for the opendrive sensor
                if self._opendrive_tag and self._opendrive_tag not in data_dict.keys() \
                        and len(self._sensors_objects.keys()) == len(data_dict.keys()) + 1:
                    # print("Ignoring opendrive sensor")
                    break

                sensor_data = self._new_data_buffers.get(True, self._queue_timeout)

                # Drop stale data that belongs to sensors already removed (e.g., after an ego vehicle is destroyed)
                tag = sensor_data[0]
                self._store_latest_sample(data_dict, tag, sensor_data[1], sensor_data[2])

            # Drain any already-queued packets and keep the newest sample per tag.
            # Without this, we can return stale camera frames when newer packets are
            # already in the queue for tags we have seen earlier in this call.
            while True:
                try:
                    sensor_data = self._new_data_buffers.get_nowait()
                except Empty:
                    break

                tag = sensor_data[0]
                self._store_latest_sample(data_dict, tag, sensor_data[1], sensor_data[2])

        except Empty:
            raise SensorReceivedNoData("A sensor took too long to send their data")

        for tag, (timestamp, _) in data_dict.items():
            self._last_returned_timestamps[tag] = timestamp

        return data_dict
