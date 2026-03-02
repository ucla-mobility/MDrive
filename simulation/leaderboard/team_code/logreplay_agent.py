"""
Minimal agent for log-replay mode.

This agent does nothing except return neutral controls. When CUSTOM_EGO_LOG_REPLAY=1
is set, the ego vehicle's actual movement is controlled by LogReplayFollower in
route_scenario.py, which uses set_transform() to follow the logged trajectory.

This agent exists solely to satisfy the leaderboard's agent requirement without
performing any complex initialization that could fail on custom maps.
"""

import datetime
import os
import pathlib
from typing import Dict, List

import carla
import numpy as np
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def get_entry_point():
    return "LogReplayAgent"


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    """Read an integer environment variable with default and minimum."""
    try:
        val = int(os.environ.get(name, default))
        return max(minimum, val)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable with fallback."""
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


class LogReplayAgent(AutonomousAgent):
    """Minimal agent that returns neutral controls for log-replay mode."""

    def setup(self, path_to_conf_file, ego_vehicles_num=1):
        """Set up log-replay agent with image capture support."""
        self.track = Track.SENSORS
        self._ego_vehicles_num = ego_vehicles_num
        self.ego_vehicles_num = ego_vehicles_num  # Required by AgentWrapper

        # Image capture settings (same pattern as tcp_agent.py)
        self.save_path = None
        self.logreplayimages_path = None
        self.calibration_path = None
        self.save_interval = _env_int("TCP_SAVE_INTERVAL", 1, minimum=1)  # Save every frame by default
        self.capture_sensor_frames = os.environ.get("TCP_CAPTURE_SENSOR_FRAMES", "").lower() in ("1", "true", "yes")
        self.capture_logreplay_images = os.environ.get("TCP_CAPTURE_LOGREPLAY_IMAGES", "").lower() in ("1", "true", "yes")
        self._frame_count = 0
        self._saved_frames = 0
        self._fake_ego_sensor_by_slot: Dict[int, object] = {}
        self._fake_ego_actor_id_by_slot: Dict[int, int] = {}
        self._fake_ego_frame_counter: Dict[int, int] = {}
        self._fake_ego_saved_counter: Dict[int, int] = {}
        self._fake_ego_last_warn_missing: Dict[int, int] = {}

        names_env = os.environ.get("CUSTOM_FAKE_EGO_CAMERA_NAMES", "")
        self._fake_ego_camera_names: List[str] = [n.strip() for n in names_env.split(",") if n.strip()]
        ids_env = os.environ.get("CUSTOM_FAKE_EGO_CAMERA_IDS", "")
        parsed_ids: List[int] = []
        for token in ids_env.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                parsed_ids.append(int(token))
            except (TypeError, ValueError):
                continue
        if len(parsed_ids) == len(self._fake_ego_camera_names):
            self._fake_ego_camera_ids = parsed_ids
        else:
            self._fake_ego_camera_ids = list(range(len(self._fake_ego_camera_names)))

        self._fake_cam_cfg = {
            "x": _env_float("TCP_LOGREPLAY_RGB_X", 1.3),
            "y": _env_float("TCP_LOGREPLAY_RGB_Y", 0.0),
            "z": _env_float("TCP_LOGREPLAY_RGB_Z", 2.3),
            "roll": _env_float("TCP_LOGREPLAY_RGB_ROLL", 0.0),
            "pitch": _env_float("TCP_LOGREPLAY_RGB_PITCH", 0.0),
            "yaw": _env_float("TCP_LOGREPLAY_RGB_YAW", 0.0),
            "width": _env_int("TCP_LOGREPLAY_RGB_WIDTH", 1280, minimum=16),
            "height": _env_int("TCP_LOGREPLAY_RGB_HEIGHT", 720, minimum=16),
            "fov": _env_float("TCP_LOGREPLAY_RGB_FOV", 100.0),
        }

        # Set up save paths if SAVE_PATH is defined
        save_path_env = os.environ.get("SAVE_PATH")
        if save_path_env:
            now = datetime.datetime.now()
            routes_env = os.environ.get("ROUTES", "logreplay")
            string = pathlib.Path(routes_env).stem + '_'
            string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
            
            self.save_path = pathlib.Path(save_path_env) / string
            try:
                self.save_path.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                print(f"[LogReplayAgent] Warning: Could not create save_path: {exc}")
                self.save_path = None
            
            if self.save_path:
                self.logreplayimages_path = self.save_path / "logreplayimages"
                if self.capture_logreplay_images or self.capture_sensor_frames:
                    try:
                        self.logreplayimages_path.mkdir(parents=True, exist_ok=True)
                    except Exception as exc:
                        print(f"[LogReplayAgent] Warning: Could not create logreplayimages_path: {exc}")
                # Backward-compatible alias
                self.calibration_path = self.logreplayimages_path

                # Create directories for each ego vehicle (only rgb_front)
                for ego_id in range(self.ego_vehicles_num):
                    try:
                        (self.save_path / f'rgb_{ego_id}').mkdir(parents=True, exist_ok=True)
                        (self.save_path / f'meta_{ego_id}').mkdir(parents=True, exist_ok=True)
                    except Exception as exc:
                        print(f"[LogReplayAgent] Warning: Could not create dirs for ego {ego_id}: {exc}")

                if (self.capture_logreplay_images or self.capture_sensor_frames) and self.logreplayimages_path:
                    rgb_front_ids = set(range(self.ego_vehicles_num))
                    rgb_front_ids.update(self._fake_ego_camera_ids)
                    for cam_id in sorted(rgb_front_ids):
                        try:
                            (self.logreplayimages_path / f'rgb_front_{cam_id}').mkdir(
                                parents=True, exist_ok=True
                            )
                        except Exception as exc:
                            print(
                                f"[LogReplayAgent] Warning: Could not create logreplay dir for camera {cam_id}: {exc}"
                            )

                print(f"[LogReplayAgent] Image capture enabled:")
                print(f"  save_path: {self.save_path}")
                print(f"  logreplayimages_path: {self.logreplayimages_path}")
                print(f"  capture_logreplay_images: {self.capture_logreplay_images}")
                print(f"  capture_sensor_frames: {self.capture_sensor_frames}")
                print(f"  save_interval: {self.save_interval}")
                if self._fake_ego_camera_names:
                    fake_info = ", ".join(
                        f"{name}->rgb_front_{idx}"
                        for name, idx in zip(self._fake_ego_camera_names, self._fake_ego_camera_ids)
                    )
                    print(f"  fake_ego_views: {fake_info}")
                if not HAS_CV2:
                    print(f"  WARNING: cv2 not available, images will not be saved!")

    def sensors(self):
        """Return sensor setup - only front RGB camera for each ego vehicle."""
        if self.ego_vehicles_num <= 0:
            return []
        return [
            # Front view camera (rgb_front_0, rgb_front_1, etc. for each ego)
            {
                "type": "sensor.camera.rgb",
                "id": "rgb_front",
                "x": 1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": 1280,
                "height": 720,
                "fov": 100,
            },
        ]

    def _destroy_fake_camera(self, slot: int):
        camera = self._fake_ego_sensor_by_slot.pop(slot, None)
        self._fake_ego_actor_id_by_slot.pop(slot, None)
        if camera is None:
            return
        try:
            camera.stop()
        except Exception:
            pass
        try:
            camera.destroy()
        except Exception:
            pass

    @staticmethod
    def _find_actor_by_role_name(world, role_name: str):
        if world is None or not role_name:
            return None
        try:
            for actor in world.get_actors().filter("vehicle.*"):
                if actor.attributes.get("role_name", "") == role_name:
                    return actor
        except Exception:
            return None
        return None

    def _on_fake_ego_image(self, slot: int, cam_id: int, image):
        if self.logreplayimages_path is None or not HAS_CV2:
            return
        if not (self.capture_logreplay_images or self.capture_sensor_frames):
            return

        frame_count = int(self._fake_ego_frame_counter.get(slot, 0)) + 1
        self._fake_ego_frame_counter[slot] = frame_count
        if not self.capture_sensor_frames and (frame_count % self.save_interval != 0):
            return

        out_dir = self.logreplayimages_path / f"rgb_front_{cam_id}"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return
        save_idx = int(self._fake_ego_saved_counter.get(slot, 0))
        self._fake_ego_saved_counter[slot] = save_idx + 1

        try:
            bgra = np.frombuffer(image.raw_data, dtype=np.uint8)
            bgra = bgra.reshape((image.height, image.width, 4))
            bgr = bgra[:, :, :3]
            cv2.imwrite(str(out_dir / f"{save_idx:06d}.jpg"), bgr)
        except Exception:
            if save_idx == 0:
                print(f"[LogReplayAgent] Warning: Failed to save fake-ego frame slot={slot}")

    def _spawn_fake_ego_camera(self, slot: int, actor, cam_id: int):
        world = actor.get_world() if actor else None
        if world is None:
            return
        try:
            bp_lib = world.get_blueprint_library()
            cam_bp = bp_lib.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", str(self._fake_cam_cfg["width"]))
            cam_bp.set_attribute("image_size_y", str(self._fake_cam_cfg["height"]))
            cam_bp.set_attribute("fov", str(self._fake_cam_cfg["fov"]))
            rel_tf = carla.Transform(
                carla.Location(
                    x=float(self._fake_cam_cfg["x"]),
                    y=float(self._fake_cam_cfg["y"]),
                    z=float(self._fake_cam_cfg["z"]),
                ),
                carla.Rotation(
                    roll=float(self._fake_cam_cfg["roll"]),
                    pitch=float(self._fake_cam_cfg["pitch"]),
                    yaw=float(self._fake_cam_cfg["yaw"]),
                ),
            )
            camera = world.spawn_actor(
                cam_bp,
                rel_tf,
                attach_to=actor,
                attachment_type=carla.AttachmentType.Rigid,
            )
            camera.listen(lambda image: self._on_fake_ego_image(slot, cam_id, image))
            self._fake_ego_sensor_by_slot[slot] = camera
            self._fake_ego_actor_id_by_slot[slot] = int(actor.id)
            self._fake_ego_frame_counter.setdefault(slot, 0)
            self._fake_ego_saved_counter.setdefault(slot, 0)
        except Exception as exc:
            print(f"[LogReplayAgent] Warning: failed to spawn fake-ego camera slot={slot}: {exc}")

    def _ensure_fake_ego_cameras(self):
        if not self._fake_ego_camera_names:
            return
        if self.logreplayimages_path is None or not HAS_CV2:
            return

        world = None
        try:
            world = CarlaDataProvider.get_world()
        except Exception:
            world = None
        if world is None:
            return

        for slot, role_name in enumerate(self._fake_ego_camera_names):
            camera_id = (
                self._fake_ego_camera_ids[slot]
                if slot < len(self._fake_ego_camera_ids)
                else slot
            )
            actor = self._find_actor_by_role_name(world, role_name)
            if actor is None:
                warn_count = int(self._fake_ego_last_warn_missing.get(slot, 0))
                if warn_count < 5:
                    print(f"[LogReplayAgent] Waiting for fake-ego actor '{role_name}'")
                    self._fake_ego_last_warn_missing[slot] = warn_count + 1
                self._destroy_fake_camera(slot)
                continue

            self._fake_ego_last_warn_missing[slot] = 0
            prev_actor_id = self._fake_ego_actor_id_by_slot.get(slot)
            if prev_actor_id is not None and prev_actor_id != int(actor.id):
                self._destroy_fake_camera(slot)

            if slot not in self._fake_ego_sensor_by_slot:
                self._spawn_fake_ego_camera(slot, actor, camera_id)

    def run_step(self, input_data, timestamp):
        """Return neutral controls for all ego vehicles; save images if enabled."""
        self._frame_count += 1

        if self.capture_logreplay_images or self.capture_sensor_frames:
            self._ensure_fake_ego_cameras()
        
        # Save images if capture is enabled and this is a save frame
        should_save = (
            self.capture_logreplay_images 
            and self.logreplayimages_path is not None 
            and HAS_CV2
            and (self._frame_count % self.save_interval == 0)
        )
        
        if should_save:
            self._save_sensor_images(input_data, timestamp)
        
        # Must return a list of controls, one per ego vehicle
        control_all = []
        for _ in range(self.ego_vehicles_num):
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0
            control.hand_brake = False
            control.manual_gear_shift = False
            control_all.append(control)
        return control_all

    def _save_sensor_images(self, input_data, timestamp):
        """Save rgb_front sensor images from input_data to disk."""
        if not self.logreplayimages_path or not HAS_CV2:
            return
        
        frame_id = self._saved_frames
        self._saved_frames += 1
        
        # Only save rgb_front for each ego vehicle
        for ego_id in range(self.ego_vehicles_num):
            # Sensor tag format: rgb_front_<ego_id>
            sensor_tag = f"rgb_front_{ego_id}"
            
            if sensor_tag in input_data:
                try:
                    # input_data[sensor_tag] is a tuple: (frame_number, image_data)
                    frame_num, image_data = input_data[sensor_tag]
                    
                    # Convert to numpy array if needed
                    if hasattr(image_data, 'shape'):
                        img = image_data
                    else:
                        # It's raw bytes, need to decode
                        continue
                    
                    # Handle different image formats
                    if len(img.shape) == 3:
                        if img.shape[2] == 4:
                            # BGRA -> BGR
                            img = img[:, :, :3]
                        elif img.shape[2] == 3:
                            # Already RGB or BGR
                            pass
                    
                    # Save image
                    out_dir = self.logreplayimages_path / f'rgb_front_{ego_id}'
                    out_path = out_dir / f'{frame_id:06d}.jpg'
                    cv2.imwrite(str(out_path), img)
                except Exception as exc:
                    # Don't spam errors, just log occasionally
                    if frame_id == 0:
                        print(f"[LogReplayAgent] Warning: Failed to save {sensor_tag}: {exc}")

    def destroy(self):
        """Cleanup."""
        for slot in list(self._fake_ego_sensor_by_slot.keys()):
            self._destroy_fake_camera(slot)
        if self._saved_frames > 0:
            print(f"[LogReplayAgent] Saved {self._saved_frames} frames to {self.logreplayimages_path}")
        fake_saved_total = int(sum(self._fake_ego_saved_counter.values()))
        if fake_saved_total > 0 and self.logreplayimages_path is not None:
            print(
                f"[LogReplayAgent] Saved {fake_saved_total} fake-ego frames to {self.logreplayimages_path}"
            )
