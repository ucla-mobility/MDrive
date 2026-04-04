import os
import json
import random
import datetime
import pathlib
import time
import imp
from collections import deque
import math
import traceback
import textwrap

import yaml
import cv2
import torch
import carla
import numpy as np
from PIL import Image
# from easydict import EasyDict
from torchvision import transforms

from leaderboard.autoagents import autonomous_agent
from leaderboard.utils.route_manipulation import _get_latlon_ref
from team_code.lm_planner import RoutePlanner, InstructionPlanner
from team_code.lm_pid_controller import PIDController
from team_code.route_debugger_runtime import RouteRuntimeDebugger
from timm.models import create_model
import sys
sys.path.insert(0, 'simulation/assets/LMDrive/LAVIS')
from lavis.common.registry import registry
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

try:
	import pygame
except ImportError:
	raise RuntimeError("cannot import pygame, make sure pygame package is installed")


SAVE_PATH = os.environ.get("SAVE_PATH", 'eval')
LMDRIVE_ROUTE_DEBUG_FLAG = str(os.environ.get("LMDRIVE_ROUTE_DEBUG", "0")).lower() in ("1", "true", "yes")
LMDRIVE_ROUTE_DEBUG_DIR = os.environ.get("LMDRIVE_ROUTE_DEBUG_DIR", "results/lmdrive_route_debug")
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def rotate_lidar(lidar, angle):
	radian = np.deg2rad(angle)
	return lidar @ [
		[ np.cos(radian), np.sin(radian), 0, 0],
		[-np.sin(radian), np.cos(radian), 0, 0],
		[0,0,1,0],
		[0,0,0,1]
	]

def lidar_to_raw_features(lidar):
	def preprocess(lidar_xyzr, lidar_painted=None):

		idx = (lidar_xyzr[:,0] > -1.2)&(lidar_xyzr[:,0] < 1.2)&(lidar_xyzr[:,1]>-1.2)&(lidar_xyzr[:,1]<1.2)

		idx = np.argwhere(idx)

		if lidar_painted is None:
			return np.delete(lidar_xyzr, idx, axis=0)
		else:
			return np.delete(lidar_xyzr, idx, axis=0), np.delete(lidar_painted, idx, axis=0)

	lidar_xyzr = preprocess(lidar)

	idxs = np.arange(len(lidar_xyzr))
	np.random.shuffle(idxs)
	lidar_xyzr = lidar_xyzr[idxs]

	lidar = np.zeros((40000, 4), dtype=np.float32)
	num_points = min(40000, len(lidar_xyzr))
	lidar[:num_points,:4] = lidar_xyzr[:num_points]
	lidar[np.isinf(lidar)] = 0
	lidar[np.isnan(lidar)] = 0
	lidar = rotate_lidar(lidar, -90).astype(np.float32)
	return lidar, num_points

class DisplayInterface(object):
	def __init__(self):
		self._width = 1200
		self._height = 900
		self._surface = None

		pygame.init()
		pygame.font.init()
		self._clock = pygame.time.Clock()
		self._display = pygame.display.set_mode(
			(self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF
		)
		pygame.display.set_caption("LMDrive Agent")

	def run_interface(self, input_data):
		rgb = input_data['rgb_front']
		rgb_left = input_data['rgb_left']
		rgb_right = input_data['rgb_right']
		rgb_focus = input_data['rgb_center']
		surface = np.zeros((900, 1200, 3),np.uint8)
		surface[:, :1200] = rgb
		surface[:210,:280] = input_data['rgb_left']
		surface[:210, 920:1200] = input_data['rgb_right']
		surface[:210, 495:705] = input_data['rgb_center']
		surface = cv2.putText(surface, input_data['time'], (20,710), cv2.FONT_HERSHEY_TRIPLEX,0.75,(0,0,255), 1)
		surface = cv2.putText(surface, input_data['meta_control'], (20,740), cv2.FONT_HERSHEY_TRIPLEX,0.75,(0,0,255), 1)
		surface = cv2.putText(surface, input_data['waypoints'], (20,770), cv2.FONT_HERSHEY_TRIPLEX,0.75,(0,0,255), 1)
		surface = cv2.putText(surface, input_data['instruction'], (20,800), cv2.FONT_HERSHEY_TRIPLEX,0.75,(0,0,255), 1)
		surface = cv2.putText(surface, input_data['notice'], (20,830), cv2.FONT_HERSHEY_TRIPLEX,0.75,(0,0,255), 1)

		surface = cv2.putText(surface, 'Left  View', (60,245), cv2.FONT_HERSHEY_TRIPLEX,0.75,(139,69,19), 2)
		surface = cv2.putText(surface, 'Focus View', (535,245), cv2.FONT_HERSHEY_TRIPLEX,0.75,(139,69,19), 2)
		surface = cv2.putText(surface, 'Right View', (980,245), cv2.FONT_HERSHEY_TRIPLEX,0.75,(139,69,19), 2)

		surface[:210, 278:282] = [139,69,19]
		surface[:210, 493:497] = [139,69,19]
		surface[:210, 703:707] = [139,69,19]
		surface[:210, 918:922] = [139,69,19]
		surface[208:212, :280] = [139,69,19]
		surface[208:212, 920:1200] = [139,69,19]
		surface[208:212, 495:705] = [139,69,19]

		self._surface = pygame.surfarray.make_surface(surface.swapaxes(0,1))

		if self._surface is not None:
			self._display.blit(self._surface, (0, 0))

		pygame.display.flip()
		pygame.event.get()
		return surface

	def _quit(self):
		pygame.quit()


def get_entry_point():
	return "LMDriveAgent"


class Resize2FixedSize:
	def __init__(self, size):
		self.size = size

	def __call__(self, pil_img):
		pil_img = pil_img.resize(self.size)
		return pil_img


def create_carla_rgb_transform(
	input_size, need_scale=True, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
):

	if isinstance(input_size, (tuple, list)):
		img_size = input_size[-2:]
	else:
		img_size = input_size
	tfl = []

	if isinstance(input_size, (tuple, list)):
		input_size_num = input_size[-1]
	else:
		input_size_num = input_size

	if need_scale:
		if input_size_num == 112:
			tfl.append(Resize2FixedSize((170, 128)))
		elif input_size_num == 128:
			tfl.append(Resize2FixedSize((195, 146)))
		elif input_size_num == 224:
			tfl.append(Resize2FixedSize((341, 256)))
		elif input_size_num == 256:
			tfl.append(Resize2FixedSize((288, 288)))
		else:
			raise ValueError("Can't find proper crop size")
	tfl.append(transforms.CenterCrop(img_size))
	tfl.append(transforms.ToTensor())
	tfl.append(transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)))

	return transforms.Compose(tfl)


class LMDriveAgent(autonomous_agent.AutonomousAgent):
	def setup(self, path_to_conf_file, ego_vehicles_num):
		self.agent_name = 'lmdrive'
		self._hic = DisplayInterface()
		self.track = autonomous_agent.Track.SENSORS
		self.ego_vehicles_num = ego_vehicles_num #1
		self._state_slots = max(10, int(self.ego_vehicles_num))
		self.step = [-1] * self._state_slots  # Per-vehicle step counters
		self.wall_start = time.time()
		self.initialized = False
		self.rgb_front_transform = create_carla_rgb_transform(224)
		self.rgb_left_transform = create_carla_rgb_transform(128)
		self.rgb_right_transform = create_carla_rgb_transform(128)
		self.rgb_center_transform = create_carla_rgb_transform(128, need_scale=False)

		self.active_misleading_instruction = [False] * self._state_slots
		self.remaining_misleading_frames = [0] * self._state_slots

		self.visual_feature_buffer = [[] for _ in range(self._state_slots)]

		# self.lm_config = imp.load_source("MainModel", '/GPFS/data/changxingliu/V2Xverse/simulation/leaderboard/team_code/lmdriver_config.py').GlobalConfig()
		self.lm_config = imp.load_source("MainModel", path_to_conf_file).GlobalConfig()
		self.config = self.lm_config
		self.turn_controller = [
			PIDController(
				K_P=self.lm_config.turn_KP,
				K_I=self.lm_config.turn_KI,
				K_D=self.lm_config.turn_KD,
				n=self.lm_config.turn_n,
			)
			for _ in range(self._state_slots)
		]
		self.speed_controller = [
			PIDController(
				K_P=self.lm_config.speed_KP,
				K_I=self.lm_config.speed_KI,
				K_D=self.lm_config.speed_KD,
				n=self.lm_config.speed_n,
			)
			for _ in range(self._state_slots)
		]

		model_cls = registry.get_model_class('vicuna_drive')
		self.agent_use_notice = self.lm_config.agent_use_notice
		self.traffic_light_notice = [''] * self._state_slots
		self.curr_notice = [''] * self._state_slots
		self.curr_notice_frame_id = [-1] * self._state_slots
		self.sample_rate = self.lm_config.sample_rate * 2 # The frequency of CARLA simulation is 20Hz

		self._debug("[SETUP] build_model=1")
		model = model_cls(preception_model=self.lm_config.preception_model,
						  preception_model_ckpt=self.lm_config.preception_model_ckpt,
						  llm_model=self.lm_config.llm_model,
						  max_txt_len=64,
						  use_notice_prompt=self.lm_config.agent_use_notice,
						  )
		self.net = model
		self._debug("[SETUP] load_model_state=1")
		self.net.load_state_dict(torch.load(self.lm_config.lmdrive_ckpt)["model"], strict=False)
		self.net.cuda()
		self.net.eval()
		self.softmax = torch.nn.Softmax(dim=1)
		self.prev_lidar = [None] * self._state_slots
		self.prev_control = [None] * self._state_slots
		self.last_predicted_waypoints_local = {}
		self.last_predicted_waypoints_step = {}
		self.curr_instruction = ['Drive Safely'] * self._state_slots
		self.sampled_scenarios = None
		self.instruction = ''
		self.lat_ref = 0.0
		self.lon_ref = 0.0
		self._carla_map = None
		self.route_debug_enabled = LMDRIVE_ROUTE_DEBUG_FLAG
		self.route_debug_dir = pathlib.Path(LMDRIVE_ROUTE_DEBUG_DIR)
		self.debug_log_every = max(1, int(os.environ.get("LMDRIVE_DEBUG_LOG_EVERY", "5")))
		self.debug_divergence_threshold = float(os.environ.get("LMDRIVE_DEBUG_DIVERGENCE_TARGET_NORM", "8.0"))
		self.debug_route_centerline_threshold = float(os.environ.get("LMDRIVE_DEBUG_ROUTE_CENTERLINE_THRESHOLD", "3.0"))
		self.debug_collapse_waypoint_norm = float(os.environ.get("LMDRIVE_DEBUG_COLLAPSE_WAYPOINT_NORM", "0.35"))
		self.debug_weak_recovery_steer = float(os.environ.get("LMDRIVE_DEBUG_WEAK_RECOVERY_STEER", "0.10"))
		self.debug_nearby_actor_radius = float(os.environ.get("LMDRIVE_DEBUG_NEARBY_RADIUS", "35.0"))
		self.debug_force_fresh_control = str(
			os.environ.get("LMDRIVE_DEBUG_FORCE_FRESH_CONTROL", "0")
		).lower() in ("1", "true", "yes")
		self.enable_misleading_instruction = str(
			os.environ.get("LMDRIVE_ENABLE_MISLEADING_INSTRUCTION", "0")
		).lower() in ("1", "true", "yes")
		self.debug_event_paths = [None] * self._state_slots
		self.route_debuggers = [None] * self._state_slots
		self.reused_control_steps = [0] * self._state_slots
		self.last_inference_step = [-1] * self._state_slots
		self.last_collision_event_frame = [None] * self._state_slots
		self.debug_runtime_state = [
			{
				"diverging": False,
				"collapse": False,
				"weak_recovery": False,
				"last_target_norm": None,
				"last_distance_to_waypoint": None,
				"last_route_centerline_distance": None,
			}
			for _ in range(self._state_slots)
		]
		self.debug_stats = [
			{
				"inference_calls": 0,
				"reuse_steps": 0,
				"divergence_events": 0,
				"collapse_events": 0,
				"weak_recovery_events": 0,
				"misleading_activations": 0,
				"collision_events": 0,
				"max_target_norm": 0.0,
				"max_distance_to_waypoint": 0.0,
				"max_route_centerline_distance": 0.0,
				"max_inference_s": 0.0,
				"max_abs_roll_deg": 0.0,
				"max_abs_pitch_deg": 0.0,
				"max_collision_impulse": 0.0,
				"min_other_ego_distance": None,
				"min_nearby_actor_distance": None,
			}
			for _ in range(self._state_slots)
		]

		self.save_path = None
		if SAVE_PATH is not None:
			now = datetime.datetime.now()
			string = pathlib.Path(os.environ["ROUTES"]).stem + "_"
			string += "_".join(
				map(
					lambda x: "%02d" % x,
					(now.month, now.day, now.hour, now.minute, now.second),
				)
			)

			self._debug(f"[SETUP] data_save_path={string}")

			self.save_path = pathlib.Path(SAVE_PATH) / string
			self.save_path.mkdir(parents=True, exist_ok=False)
			for ego_id in range(ego_vehicles_num):
				(self.save_path / f"meta_{ego_id}").mkdir(parents=True, exist_ok=False)
			self.debug_base_dir = None
			if self.route_debug_enabled:
				self.debug_base_dir = self.save_path / "debug_lmdrive" if self.save_path is not None else self.route_debug_dir
				self.debug_base_dir.mkdir(parents=True, exist_ok=True)
				for ego_id in range(self.ego_vehicles_num):
					self.debug_event_paths[ego_id] = self.debug_base_dir / f"ego_{ego_id}_events.jsonl"
			self._debug(
				f"[SETUP] ego_vehicles_num={self.ego_vehicles_num} state_slots={self._state_slots} "
				f"route_debug_enabled={int(self.route_debug_enabled)} debug_log_every={self.debug_log_every} "
			f"debug_force_fresh_control={int(self.debug_force_fresh_control)} "
			f"misleading_instruction_enabled={int(self.enable_misleading_instruction)} "
			f"debug_base_dir={self.debug_base_dir}"
		)

	def _init(self):
		try:
			self.lat_ref, self.lon_ref = _get_latlon_ref(CarlaDataProvider.get_world())
		except Exception as exc:
			self.lat_ref, self.lon_ref = 0.0, 0.0
			self._debug(f"[INIT_WARN] failed_to_get_latlon_ref error={exc}")
		try:
			world = CarlaDataProvider.get_world()
			self._carla_map = world.get_map() if world is not None else None
		except Exception as exc:
			self._carla_map = None
			self._debug(f"[INIT_WARN] failed_to_get_carla_map error={exc}")
		self._route_planner = RoutePlanner(5, 50.0)
		self._route_planner.set_route(self._global_plan, True, self._global_plan_world_coord)
		# Keep planner state isolated per ego to avoid cross-ego instruction leakage.
		self._instruction_planners = [
			InstructionPlanner(notice_light_switch=True)
			for _ in range(self.ego_vehicles_num)
		]
		self.next_action_step = [-1] * self._state_slots
		self.initialized = True
		random.seed(''.join([str(x[0]) for x in self._global_plan]))
		route_tag = pathlib.Path(os.environ.get("ROUTES", "custom_route")).stem
		session_tag = self.save_path.name if self.save_path is not None else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
		if self.route_debug_enabled:
			route_debug_root = self.debug_base_dir / "route_runtime"
			for ego_id in range(self.ego_vehicles_num):
				debugger = RouteRuntimeDebugger(
					base_dir=str(route_debug_root),
					route_tag=f"{route_tag}_ego{ego_id}",
					session_tag=session_tag,
				)
				debugger.divergence_threshold = float(
					os.environ.get(
						"LMDRIVE_ROUTE_DEBUG_DISTANCE_THRESHOLD",
						str(self.debug_route_centerline_threshold),
					)
				)
				debugger.set_reference(self.lat_ref, self.lon_ref)
				debugger.capture_plan(
					self._global_plan[ego_id],
					self._global_plan_world_coord[ego_id],
					list(self._route_planner.route[ego_id]),
				)
				self.route_debuggers[ego_id] = debugger
				self._write_debug_event(
					ego_id,
					"route_debug_init",
					{
						"route_tag": route_tag,
						"session_tag": session_tag,
						"route_points": len(self._global_plan[ego_id]),
						"lat_ref": self.lat_ref,
						"lon_ref": self.lon_ref,
						"debug_dir": debugger.base_dir,
					},
				)
		self._write_meta_route_overviews()
		self._debug(
			f"[INIT] lat_ref={self.lat_ref:.6f} lon_ref={self.lon_ref:.6f} "
			f"route_lengths={[len(route) for route in self._global_plan]}"
		)

		self.debug_dict = {}
		for id in range(self._state_slots):
			self.debug_dict[id] = {
				"num_misleading": 0,
				"num_instruction": 0,
    			"num_self.curr_instruction != last_instruction":0,
       			"num_last_misleading_instruction_exist": 0
			}


	def _get_position(self, tick_data):
		gps = tick_data["gps"]
		gps = (gps - self._route_planner.mean) * self._route_planner.scale
		return gps

		def _debug(self, message):
			debug_stdout = str(os.environ.get("LMDRIVE_DEBUG_STDOUT", "0")).lower() in ("1", "true", "yes")
			if not getattr(self, "route_debug_enabled", LMDRIVE_ROUTE_DEBUG_FLAG) and not debug_stdout:
				return
			print(f"[DEBUG LMDRIVE] {message}", flush=True)

	def _json_safe(self, value):
		if isinstance(value, dict):
			return {str(k): self._json_safe(v) for k, v in value.items()}
		if isinstance(value, (list, tuple)):
			return [self._json_safe(v) for v in value]
		if isinstance(value, np.ndarray):
			return value.tolist()
		if isinstance(value, np.generic):
			return value.item()
		if isinstance(value, pathlib.Path):
			return str(value)
		return value

	def _write_debug_event(self, ego_id, category, payload):
		path = self.debug_event_paths[ego_id] if ego_id < len(self.debug_event_paths) else None
		if path is None:
			return
		record = {
			"category": str(category),
			"ego_id": int(ego_id),
			"timestamp_unix": float(time.time()),
			"step": int(self.step[ego_id]) if ego_id < len(self.step) else None,
			"payload": self._json_safe(payload),
		}
		with path.open("a", encoding="utf-8") as handle:
			handle.write(json.dumps(record, sort_keys=True) + "\n")

	def _command_name(self, command):
		return getattr(command, "name", str(command))

	def _format_xy(self, value):
		if value is None:
			return "(None,None)"
		if hasattr(value, "tolist"):
			value = value.tolist()
		if isinstance(value, (list, tuple)) and len(value) >= 2:
			return f"({float(value[0]):.2f},{float(value[1]):.2f})"
		return f"({value})"

	def _format_control(self, control):
		if control is None:
			return "None"
		return (
			f"(steer={float(control.steer):.3f},throttle={float(control.throttle):.3f},"
			f"brake={float(control.brake):.3f})"
		)

	def _default_collision_event(self):
		return {
			"has_collision": False,
			"event_frame": None,
			"other_actor_id": None,
			"other_actor_type": "",
			"other_actor_role_name": "",
			"normal_impulse": [0.0, 0.0, 0.0],
			"normal_impulse_magnitude": 0.0,
		}

	def _normalize_collision_event(self, payload):
		result = self._default_collision_event()
		if not isinstance(payload, dict):
			return result
		result["has_collision"] = bool(payload.get("has_collision", False))
		result["event_frame"] = payload.get("event_frame")
		result["other_actor_id"] = payload.get("other_actor_id")
		result["other_actor_type"] = str(payload.get("other_actor_type", ""))
		result["other_actor_role_name"] = str(payload.get("other_actor_role_name", ""))
		impulse = payload.get("normal_impulse", [0.0, 0.0, 0.0])
		if not isinstance(impulse, (list, tuple)) or len(impulse) < 3:
			impulse = [0.0, 0.0, 0.0]
		result["normal_impulse"] = [float(impulse[0]), float(impulse[1]), float(impulse[2])]
		result["normal_impulse_magnitude"] = float(payload.get("normal_impulse_magnitude", 0.0))
		return result

	def _format_collision(self, collision_event):
		if not collision_event or not collision_event.get("has_collision"):
			return "none"
		return (
			f"frame={collision_event.get('event_frame')} other={collision_event.get('other_actor_type') or 'unknown'}"
			f"#{collision_event.get('other_actor_id')} impulse={float(collision_event.get('normal_impulse_magnitude', 0.0)):.2f}"
		)

	def _get_wheel_steer_angles(self, actor):
		if actor is None or not hasattr(actor, "get_wheel_steer_angle") or not hasattr(carla, "VehicleWheelLocation"):
			return None
		locations = {
			"front_left": ("FL_Wheel", "Front_Left"),
			"front_right": ("FR_Wheel", "Front_Right"),
			"rear_left": ("RL_Wheel", "Rear_Left"),
			"rear_right": ("RR_Wheel", "Rear_Right"),
		}
		result = {}
		for key, candidates in locations.items():
			location = None
			for candidate in candidates:
				location = getattr(carla.VehicleWheelLocation, candidate, None)
				if location is not None:
					break
			if location is None:
				continue
			try:
				result[key] = float(actor.get_wheel_steer_angle(location))
			except Exception:
				continue
		return result or None

	def _get_actor_control_state(self, actor):
		if actor is None or not hasattr(actor, "get_control"):
			return None
		try:
			control = actor.get_control()
		except Exception:
			return None
		return {
			"steer": float(control.steer),
			"throttle": float(control.throttle),
			"brake": float(control.brake),
			"hand_brake": bool(control.hand_brake),
			"reverse": bool(control.reverse),
			"manual_gear_shift": bool(control.manual_gear_shift),
			"gear": int(control.gear),
		}

	def _get_actor_state(self, actor):
		if actor is None:
			return None
		try:
			transform = actor.get_transform()
			velocity = actor.get_velocity()
			acceleration = actor.get_acceleration()
			angular_velocity = actor.get_angular_velocity()
		except Exception:
			return None
		speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
		return {
			"x": float(transform.location.x),
			"y": float(transform.location.y),
			"z": float(transform.location.z),
			"roll": float(transform.rotation.roll),
			"pitch": float(transform.rotation.pitch),
			"yaw": float(transform.rotation.yaw),
			"speed": float(speed),
			"velocity": [float(velocity.x), float(velocity.y), float(velocity.z)],
			"acceleration": [float(acceleration.x), float(acceleration.y), float(acceleration.z)],
			"angular_velocity": [float(angular_velocity.x), float(angular_velocity.y), float(angular_velocity.z)],
			"control": self._get_actor_control_state(actor),
			"wheel_steer_angles_deg": self._get_wheel_steer_angles(actor),
		}

	def _get_map_waypoint_descriptor(self, *, transform=None, location=None):
		if self._carla_map is None:
			return None
		if transform is not None:
			location = transform.location
		elif isinstance(location, dict):
			if location.get("x") is None or location.get("y") is None:
				return None
			location = carla.Location(
				x=float(location.get("x", 0.0)),
				y=float(location.get("y", 0.0)),
				z=float(location.get("z", 0.0)),
			)
		if location is None:
			return None
		try:
			waypoint = self._carla_map.get_waypoint(location)
		except TypeError:
			waypoint = self._carla_map.get_waypoint(location, project_to_road=True)
		except Exception:
			return None
		if waypoint is None:
			return None
		junction_id = None
		if waypoint.is_junction:
			try:
				junction = waypoint.get_junction()
				if junction is not None:
					junction_id = int(junction.id)
			except Exception:
				junction_id = None
		waypoint_transform = waypoint.transform
		return {
			"road_id": int(waypoint.road_id),
			"section_id": int(waypoint.section_id),
			"lane_id": int(waypoint.lane_id),
			"lane_type": str(waypoint.lane_type).split(".")[-1],
			"lane_change": str(waypoint.lane_change).split(".")[-1],
			"is_junction": bool(waypoint.is_junction),
			"junction_id": junction_id,
			"lane_width": float(waypoint.lane_width),
			"x": float(waypoint_transform.location.x),
			"y": float(waypoint_transform.location.y),
			"z": float(waypoint_transform.location.z),
			"yaw": float(waypoint_transform.rotation.yaw),
		}

	def _format_waypoint_brief(self, waypoint_info):
		if not waypoint_info:
			return "n/a"
		junction = int(bool(waypoint_info.get("is_junction")))
		return (
			f"road={waypoint_info.get('road_id')} lane={waypoint_info.get('lane_id')} "
			f"sec={waypoint_info.get('section_id')} jct={junction}"
		)

	def _wrap_debug_text(self, text, width=42):
		text = "" if text is None else str(text)
		wrapped = textwrap.wrap(text, width=max(8, int(width)))
		return wrapped or [""]

	def _draw_text_lines(
		self,
		canvas,
		lines,
		*,
		x,
		y,
		line_height=22,
		font_scale=0.55,
		color=(235, 235, 235),
		thickness=1,
	):
		for line in lines:
			cv2.putText(
				canvas,
				str(line),
				(int(x), int(y)),
				cv2.FONT_HERSHEY_SIMPLEX,
				font_scale,
				color,
				thickness,
				lineType=cv2.LINE_AA,
			)
			y += line_height
		return y

	def _command_color(self, command_name):
		name = str(command_name or "UNKNOWN").upper()
		if "CHANGE" in name and "LEFT" in name:
			return (0, 220, 220)
		if "CHANGE" in name and "RIGHT" in name:
			return (255, 90, 180)
		if "LEFT" in name:
			return (82, 156, 255)
		if "RIGHT" in name:
			return (255, 170, 70)
		if "STRAIGHT" in name:
			return (72, 214, 112)
		if "FOLLOW" in name:
			return (220, 220, 220)
		if "VOID" in name:
			return (255, 80, 80)
		return (255, 220, 0)

	def _route_entries_to_local(self, ego_position, compass, route_entries, max_points=40):
		ego_position = np.asarray(ego_position, dtype=np.float64)
		theta = float(compass) + np.pi / 2
		rotation = np.array(
			[[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
			dtype=np.float64,
		)
		local_points = []
		for idx, entry in enumerate(list(route_entries)[:max_points]):
			route_pos = np.asarray(entry[0], dtype=np.float64)
			local_xy = rotation.T.dot(route_pos - ego_position)
			transform = entry[2] if len(entry) > 2 else None
			local_points.append(
				{
					"index": int(idx),
					"local_xy": [float(local_xy[0]), float(local_xy[1])],
					"command_name": self._command_name(entry[1] if len(entry) > 1 else None),
					"world_waypoint": self._get_map_waypoint_descriptor(transform=transform),
				}
			)
		return local_points

	def _build_route_debug_snapshot(
		self,
		ego_id,
		tick_data,
		planner,
		last_instruction,
		last_notice,
		last_traffic_light_notice,
		last_misleading_instruction,
		instruction_update_reason,
		waypoints_debug,
		control,
		metadata,
	):
		route_entries = list(self._route_planner.route[ego_id])
		instruction_state = planner.get_debug_state(ego_id) if hasattr(planner, "get_debug_state") else {}
		local_route_points = self._route_entries_to_local(
			tick_data["gps"],
			tick_data["compass"],
			route_entries,
			max_points=40,
		)
		predicted_waypoints = np.asarray(waypoints_debug, dtype=np.float64).copy()
		if predicted_waypoints.size:
			predicted_waypoints[:, 1] *= -1.0
		target_point = np.asarray(tick_data["target_point"], dtype=np.float64)
		next_waypoint = np.asarray(tick_data["target_point"], dtype=np.float64)
		route_projection_local = None
		route_projection = tick_data.get("nearest_route_centerline_projection")
		if route_projection is not None:
			ego_position = np.asarray(tick_data["gps"], dtype=np.float64)
			theta = float(tick_data["compass"]) + np.pi / 2
			rotation = np.array(
				[[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
				dtype=np.float64,
			)
			route_projection_local = rotation.T.dot(np.asarray(route_projection, dtype=np.float64) - ego_position)
			route_projection_local = [float(route_projection_local[0]), float(route_projection_local[1])]

		nearest_route_map = None
		nearest_route_index = tick_data.get("nearest_route_point_index")
		if nearest_route_index is not None and 0 <= int(nearest_route_index) < len(route_entries):
			transform = route_entries[int(nearest_route_index)][2] if len(route_entries[int(nearest_route_index)]) > 2 else None
			nearest_route_map = self._get_map_waypoint_descriptor(transform=transform)

		return {
			"ego_id": int(ego_id),
			"step": int(self.step[ego_id]),
			"town_id": instruction_state.get("town_id", self.town_id),
			"next_command_name": tick_data.get("next_command_name"),
			"next_command_value": int(tick_data.get("next_command", -1)),
			"route_remaining": int(tick_data.get("route_remaining", 0)),
			"distance_to_waypoint": float(tick_data.get("distance_to_waypoint", 0.0)),
			"target_norm": float(np.linalg.norm(target_point)),
			"target_point_local": [float(target_point[0]), float(target_point[1])],
			"next_waypoint_local": [float(next_waypoint[0]), float(next_waypoint[1])],
			"predicted_waypoints_local": predicted_waypoints.tolist(),
			"route_projection_local": route_projection_local,
			"nearest_route_point_index": None if nearest_route_index is None else int(nearest_route_index),
			"nearest_route_segment_index": tick_data.get("nearest_route_segment_index"),
			"nearest_route_segment_t": tick_data.get("nearest_route_segment_t"),
			"nearest_route_centerline_distance": tick_data.get("nearest_route_centerline_distance"),
			"next_waypoint_world": tick_data.get("next_waypoint_world"),
			"ego_world": tick_data.get("ego_world"),
			"ego_waypoint": self._get_map_waypoint_descriptor(location=tick_data.get("ego_world")),
			"next_waypoint_waypoint": self._get_map_waypoint_descriptor(location=tick_data.get("next_waypoint_world")),
			"nearest_route_waypoint": nearest_route_map,
			"route_local_points": local_route_points,
			"instruction": {
				"route_instruction_text": last_instruction,
				"chosen_prompt_text": self.curr_instruction[ego_id],
				"pending_instruction_text": instruction_state.get("pending_instruction_text"),
				"route_instruction_id": instruction_state.get("returned_instruction_id"),
				"pending_instruction_id": instruction_state.get("pending_instruction_id"),
				"misleading_candidate_text": last_misleading_instruction,
				"misleading_active": bool(self.active_misleading_instruction[ego_id]),
				"misleading_enabled": bool(self.enable_misleading_instruction),
				"remaining_misleading_frames": int(self.remaining_misleading_frames[ego_id]),
				"update_reason": instruction_update_reason,
				"destination_distance_m": instruction_state.get("destination_distance_m"),
				"planner_current_command_value": instruction_state.get("current_command_value"),
				"planner_last_command_value": instruction_state.get("last_command_value"),
				"left_lane_num": instruction_state.get("left_lane_num"),
				"right_lane_num": instruction_state.get("right_lane_num"),
			},
			"notice": {
				"notice_text": last_notice,
				"traffic_notice_text": last_traffic_light_notice,
			},
			"control": {
				"steer": float(control.steer),
				"throttle": float(control.throttle),
				"brake": float(control.brake),
				"desired_speed": float(metadata.get("desired_speed", 0.0)),
				"speed_ratio": float(metadata.get("speed_ratio", 0.0)),
				"angle": float(metadata.get("angle", 0.0)),
				"aim": self._json_safe(metadata.get("aim")),
				"launch_assist": bool(metadata.get("launch_assist", False)),
			},
		}

	def _render_route_debug_panel(self, snapshot):
		canvas = np.zeros((900, 1200, 3), dtype=np.uint8)
		canvas[:] = (20, 24, 32)
		plot_x0, plot_y0, plot_x1, plot_y1 = 36, 56, 760, 860
		panel_x0 = 790
		cv2.rectangle(canvas, (plot_x0, plot_y0), (plot_x1, plot_y1), (45, 50, 62), 1)
		cv2.rectangle(canvas, (panel_x0, 30), (1160, 860), (45, 50, 62), 1)
		cv2.putText(
			canvas,
			"LMDrive Route / Instruction Debug",
			(38, 34),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.9,
			(240, 240, 240),
			2,
			lineType=cv2.LINE_AA,
		)

		route_points = snapshot.get("route_local_points", [])
		target_point = snapshot.get("target_point_local") or [0.0, 0.0]
		next_waypoint = snapshot.get("next_waypoint_local") or [0.0, 0.0]
		predicted_waypoints = snapshot.get("predicted_waypoints_local") or []
		route_projection = snapshot.get("route_projection_local")

		all_x = [0.0, float(target_point[0]), float(next_waypoint[0])]
		all_y = [0.0, float(target_point[1]), float(next_waypoint[1])]
		for point in route_points:
			local_xy = point.get("local_xy") or [0.0, 0.0]
			all_x.append(float(local_xy[0]))
			all_y.append(float(local_xy[1]))
		for point in predicted_waypoints:
			all_x.append(float(point[0]))
			all_y.append(float(point[1]))
		if route_projection is not None:
			all_x.append(float(route_projection[0]))
			all_y.append(float(route_projection[1]))

		left_right_extent = max(14.0, min(45.0, max(abs(x) for x in all_x) + 6.0))
		forward_extent = max(28.0, min(85.0, max(all_y) + 10.0))
		rear_extent = max(8.0, min(25.0, abs(min(all_y)) + 6.0))
		scale_x = (plot_x1 - plot_x0 - 80) / max(2.0 * left_right_extent, 1e-6)
		scale_y = (plot_y1 - plot_y0 - 110) / max(forward_extent + rear_extent, 1e-6)
		scale = float(min(scale_x, scale_y))
		origin_x = int((plot_x0 + plot_x1) / 2)
		origin_y = int(plot_y1 - 38 - rear_extent * scale)

		def _to_px(local_xy):
			return (
				int(round(origin_x + float(local_xy[0]) * scale)),
				int(round(origin_y - float(local_xy[1]) * scale)),
			)

		x_tick = int(max(10, round(left_right_extent / 4.0 / 5.0) * 5))
		y_tick = int(max(10, round(forward_extent / 5.0 / 5.0) * 5))
		for meter in range(-int(left_right_extent), int(left_right_extent) + 1, x_tick):
			x_px, _ = _to_px((meter, 0.0))
			cv2.line(canvas, (x_px, plot_y0 + 16), (x_px, plot_y1 - 16), (32, 36, 46), 1)
		for meter in range(-int(rear_extent), int(forward_extent) + 1, y_tick):
			_, y_px = _to_px((0.0, meter))
			cv2.line(canvas, (plot_x0 + 16, y_px), (plot_x1 - 16, y_px), (32, 36, 46), 1)

		cv2.line(canvas, (plot_x0 + 16, origin_y), (plot_x1 - 16, origin_y), (80, 85, 96), 1)
		cv2.line(canvas, (origin_x, plot_y0 + 16), (origin_x, plot_y1 - 16), (80, 85, 96), 1)
		cv2.putText(canvas, "left / right", (plot_x0 + 18, plot_y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1, lineType=cv2.LINE_AA)
		cv2.putText(canvas, "forward", (origin_x + 8, plot_y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1, lineType=cv2.LINE_AA)

		for idx in range(len(route_points) - 1):
			start = route_points[idx]
			end = route_points[idx + 1]
			color = self._command_color(end.get("command_name"))
			cv2.line(canvas, _to_px(start["local_xy"]), _to_px(end["local_xy"]), color, 3, lineType=cv2.LINE_AA)
		prev_command = None
		for point in route_points:
			local_xy = point["local_xy"]
			color = self._command_color(point.get("command_name"))
			radius = 5 if point["index"] == 1 else 3
			cv2.circle(canvas, _to_px(local_xy), radius, color, -1, lineType=cv2.LINE_AA)
			command_name = point.get("command_name")
			if point["index"] < 4 or command_name != prev_command:
				label_xy = _to_px(local_xy)
				cv2.putText(
					canvas,
					f"{point['index']}:{command_name[:3]}",
					(label_xy[0] + 6, label_xy[1] - 6),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.4,
					(225, 225, 225),
					1,
					lineType=cv2.LINE_AA,
				)
			prev_command = command_name

		cv2.circle(canvas, _to_px((0.0, 0.0)), 8, (255, 255, 255), -1, lineType=cv2.LINE_AA)
		cv2.arrowedLine(canvas, _to_px((0.0, 0.0)), _to_px((0.0, 5.5)), (255, 255, 255), 2, line_type=cv2.LINE_AA, tipLength=0.25)
		cv2.arrowedLine(canvas, _to_px((0.0, 0.0)), _to_px(target_point), (0, 220, 255), 2, line_type=cv2.LINE_AA, tipLength=0.22)
		cv2.circle(canvas, _to_px(target_point), 6, (0, 220, 255), -1, lineType=cv2.LINE_AA)
		cv2.circle(canvas, _to_px(next_waypoint), 7, (255, 215, 0), 2, lineType=cv2.LINE_AA)
		if route_projection is not None:
			proj_px = _to_px(route_projection)
			cv2.drawMarker(canvas, proj_px, (180, 110, 255), markerType=cv2.MARKER_SQUARE, markerSize=13, thickness=2)
			cv2.line(canvas, _to_px((0.0, 0.0)), proj_px, (110, 75, 170), 1, lineType=cv2.LINE_AA)
		if predicted_waypoints:
			pred_px = [_to_px(point) for point in predicted_waypoints]
			for idx in range(len(pred_px) - 1):
				cv2.line(canvas, pred_px[idx], pred_px[idx + 1], (255, 90, 90), 3, lineType=cv2.LINE_AA)
			for idx, point in enumerate(pred_px):
				cv2.circle(canvas, point, 4 if idx == 0 else 3, (255, 90, 90), -1, lineType=cv2.LINE_AA)

		legend_y = plot_y1 - 8
		legend_lines = [
			"route=command color   pred=red   target=cyan   next wp=gold   proj=purple",
		]
		self._draw_text_lines(canvas, legend_lines, x=plot_x0 + 14, y=legend_y, line_height=18, font_scale=0.42, color=(190, 190, 190))

		panel_y = 66
		accent = (120, 205, 255)
		panel_y = self._draw_text_lines(canvas, ["Instruction"], x=panel_x0 + 14, y=panel_y, line_height=22, font_scale=0.7, color=accent, thickness=2)
		instruction = snapshot.get("instruction", {})
		panel_y = self._draw_text_lines(
			canvas,
			[
				f"next_command={snapshot.get('next_command_name')}/{snapshot.get('next_command_value')}",
				f"route_remaining={snapshot.get('route_remaining')}",
				f"returned_id={instruction.get('route_instruction_id')} pending_id={instruction.get('pending_instruction_id')}",
				f"update_reason={instruction.get('update_reason')}",
			],
			x=panel_x0 + 14,
			y=panel_y,
		)
		for title, value in [
			("Returned", instruction.get("route_instruction_text")),
			("Chosen", instruction.get("chosen_prompt_text")),
			("Pending", instruction.get("pending_instruction_text")),
			("Mislead Candidate", instruction.get("misleading_candidate_text")),
		]:
			panel_y = self._draw_text_lines(canvas, [f"{title}:"], x=panel_x0 + 14, y=panel_y + 4, line_height=20, font_scale=0.52, color=(210, 210, 210), thickness=1)
			panel_y = self._draw_text_lines(
				canvas,
				self._wrap_debug_text(value, width=44),
				x=panel_x0 + 28,
				y=panel_y,
				line_height=20,
				font_scale=0.5,
				color=(240, 240, 240),
			)
		panel_y += 6
		panel_y = self._draw_text_lines(canvas, ["Plan Geometry"], x=panel_x0 + 14, y=panel_y, line_height=22, font_scale=0.7, color=accent, thickness=2)
		panel_y = self._draw_text_lines(
			canvas,
			[
				f"target_local={self._format_xy(snapshot.get('target_point_local'))}",
				f"target_norm={float(snapshot.get('target_norm') or 0.0):.2f}m",
				f"d_next_wp={float(snapshot.get('distance_to_waypoint') or 0.0):.2f}m",
				f"d_centerline={float(snapshot.get('nearest_route_centerline_distance') or 0.0):.2f}m",
				f"nearest_idx={snapshot.get('nearest_route_point_index')} seg={snapshot.get('nearest_route_segment_index')} t={float(snapshot.get('nearest_route_segment_t') or 0.0):.2f}",
				f"planner_cmd={instruction.get('planner_current_command_value')} last_cmd={instruction.get('planner_last_command_value')}",
				f"dest_distance={float(instruction.get('destination_distance_m') or 0.0):.1f}m",
				f"left_lanes={instruction.get('left_lane_num')} right_lanes={instruction.get('right_lane_num')}",
			],
			x=panel_x0 + 14,
			y=panel_y,
		)
		panel_y += 4
		panel_y = self._draw_text_lines(canvas, ["Road IDs"], x=panel_x0 + 14, y=panel_y, line_height=22, font_scale=0.7, color=accent, thickness=2)
		panel_y = self._draw_text_lines(
			canvas,
			[
				f"ego: {self._format_waypoint_brief(snapshot.get('ego_waypoint'))}",
				f"next: {self._format_waypoint_brief(snapshot.get('next_waypoint_waypoint'))}",
				f"nearest route: {self._format_waypoint_brief(snapshot.get('nearest_route_waypoint'))}",
			],
			x=panel_x0 + 14,
			y=panel_y,
		)
		panel_y += 4
		panel_y = self._draw_text_lines(canvas, ["Control"], x=panel_x0 + 14, y=panel_y, line_height=22, font_scale=0.7, color=accent, thickness=2)
		control = snapshot.get("control", {})
		panel_y = self._draw_text_lines(
			canvas,
			[
				f"steer={float(control.get('steer') or 0.0):.3f}",
				f"throttle={float(control.get('throttle') or 0.0):.3f}",
				f"brake={float(control.get('brake') or 0.0):.3f}",
				f"desired_speed={float(control.get('desired_speed') or 0.0):.2f}m/s",
				f"speed_ratio={float(control.get('speed_ratio') or 0.0):.2f}",
				f"launch_assist={int(bool(control.get('launch_assist')))}",
			],
			x=panel_x0 + 14,
			y=panel_y,
		)
		panel_y += 4
		panel_y = self._draw_text_lines(canvas, ["Notice"], x=panel_x0 + 14, y=panel_y, line_height=22, font_scale=0.7, color=accent, thickness=2)
		for title, value in [
			("route_notice", snapshot.get("notice", {}).get("notice_text")),
			("traffic_notice", snapshot.get("notice", {}).get("traffic_notice_text")),
		]:
			panel_y = self._draw_text_lines(canvas, [f"{title}:"], x=panel_x0 + 14, y=panel_y, line_height=20, font_scale=0.5, color=(210, 210, 210))
			panel_y = self._draw_text_lines(
				canvas,
				self._wrap_debug_text(value, width=44),
				x=panel_x0 + 28,
				y=panel_y,
				line_height=19,
				font_scale=0.48,
				color=(240, 240, 240),
			)
		return canvas

	def _render_route_overview(self, ego_id):
		if ego_id >= len(self._global_plan_world_coord):
			return None
		route_world = self._global_plan_world_coord[ego_id]
		if not route_world:
			return None
		route_points = []
		for idx, entry in enumerate(route_world):
			transform = entry[0]
			command = entry[1] if len(entry) > 1 else None
			waypoint_info = self._get_map_waypoint_descriptor(transform=transform)
			route_points.append(
				{
					"index": int(idx),
					"x": float(transform.location.x),
					"y": float(transform.location.y),
					"command_name": self._command_name(command),
					"waypoint": waypoint_info,
				}
			)
		canvas = np.zeros((900, 1200, 3), dtype=np.uint8)
		canvas[:] = (20, 24, 32)
		plot_x0, plot_y0, plot_x1, plot_y1 = 42, 60, 760, 860
		panel_x0 = 790
		cv2.putText(canvas, f"LMDrive Route Overview Ego {ego_id}", (44, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, lineType=cv2.LINE_AA)
		cv2.rectangle(canvas, (plot_x0, plot_y0), (plot_x1, plot_y1), (45, 50, 62), 1)
		cv2.rectangle(canvas, (panel_x0, 30), (1160, 860), (45, 50, 62), 1)

		x_values = [point["x"] for point in route_points]
		y_values = [point["y"] for point in route_points]
		x_min, x_max = min(x_values), max(x_values)
		y_min, y_max = min(y_values), max(y_values)
		x_span = max(1.0, x_max - x_min)
		y_span = max(1.0, y_max - y_min)
		scale = min((plot_x1 - plot_x0 - 60) / x_span, (plot_y1 - plot_y0 - 60) / y_span)
		margin_x = ((plot_x1 - plot_x0) - x_span * scale) / 2.0
		margin_y = ((plot_y1 - plot_y0) - y_span * scale) / 2.0

		def _to_px(point):
			x_px = int(round(plot_x0 + margin_x + (point["x"] - x_min) * scale))
			y_px = int(round(plot_y1 - margin_y - (point["y"] - y_min) * scale))
			return (x_px, y_px)

		change_points = []
		prev_command = None
		prev_road_lane = None
		for idx in range(len(route_points) - 1):
			start = route_points[idx]
			end = route_points[idx + 1]
			color = self._command_color(end["command_name"])
			cv2.line(canvas, _to_px(start), _to_px(end), color, 4, lineType=cv2.LINE_AA)
		for point in route_points:
			cv2.circle(canvas, _to_px(point), 4, self._command_color(point["command_name"]), -1, lineType=cv2.LINE_AA)
			waypoint = point.get("waypoint") or {}
			road_lane = (waypoint.get("road_id"), waypoint.get("lane_id"))
			if point["index"] == 0 or point["command_name"] != prev_command or road_lane != prev_road_lane:
				change_points.append(point)
			prev_command = point["command_name"]
			prev_road_lane = road_lane

		start_px = _to_px(route_points[0])
		end_px = _to_px(route_points[-1])
		cv2.circle(canvas, start_px, 7, (72, 214, 112), -1, lineType=cv2.LINE_AA)
		cv2.circle(canvas, end_px, 8, (255, 90, 90), 2, lineType=cv2.LINE_AA)
		cv2.putText(canvas, "start", (start_px[0] + 8, start_px[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (225, 225, 225), 1, lineType=cv2.LINE_AA)
		cv2.putText(canvas, "end", (end_px[0] + 8, end_px[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (225, 225, 225), 1, lineType=cv2.LINE_AA)

		for label_idx, point in enumerate(change_points[:12], start=1):
			point_px = _to_px(point)
			cv2.putText(
				canvas,
				str(label_idx),
				(point_px[0] + 6, point_px[1] - 6),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.45,
				(240, 240, 240),
				1,
				lineType=cv2.LINE_AA,
			)

		panel_y = 64
		accent = (120, 205, 255)
		panel_y = self._draw_text_lines(canvas, ["Command Changes / Road IDs"], x=panel_x0 + 14, y=panel_y, line_height=22, font_scale=0.68, color=accent, thickness=2)
		panel_y = self._draw_text_lines(
			canvas,
			[
				f"route_points={len(route_points)}",
				f"world_span=({x_span:.1f}m x {y_span:.1f}m)",
			],
			x=panel_x0 + 14,
			y=panel_y,
		)
		for label_idx, point in enumerate(change_points[:16], start=1):
			waypoint = point.get("waypoint")
			lines = [
				f"{label_idx}. idx={point['index']} cmd={point['command_name']}",
				f"   {self._format_waypoint_brief(waypoint)}",
			]
			panel_y = self._draw_text_lines(canvas, lines, x=panel_x0 + 14, y=panel_y + 2, line_height=20, font_scale=0.48, color=(240, 240, 240))
		panel_y += 8
		panel_y = self._draw_text_lines(canvas, ["Interpretation"], x=panel_x0 + 14, y=panel_y, line_height=22, font_scale=0.68, color=accent, thickness=2)
		panel_y = self._draw_text_lines(
			canvas,
			self._wrap_debug_text(
				"LMDrive follows the global route points and commands here. The language instruction planner does not derive named roads from this plot; it mainly uses next_command plus hard-coded town-specific spatial zones, lane counts, and traffic lights.",
				width=45,
			),
			x=panel_x0 + 14,
			y=panel_y,
			line_height=20,
			font_scale=0.48,
			color=(230, 230, 230),
		)
		return canvas

	def _write_meta_route_overviews(self):
		if self.save_path is None:
			return
		for ego_id in range(self.ego_vehicles_num):
			canvas = self._render_route_overview(ego_id)
			if canvas is None:
				continue
			meta_dir = self.save_path / f"meta_{ego_id}"
			Image.fromarray(canvas).save(meta_dir / "_route_overview.png")

	def _collect_ego_interactions(self, ego_id):
		ego_actor = CarlaDataProvider.get_hero_actor(hero_id=ego_id)
		ego_state = self._get_actor_state(ego_actor)
		if ego_state is None:
			return []
		interactions = []
		ego_yaw_rad = math.radians(ego_state["yaw"])
		ego_speed = float(ego_state["speed"])
		for other_id in range(self.ego_vehicles_num):
			if other_id == ego_id:
				continue
			other_actor = CarlaDataProvider.get_hero_actor(hero_id=other_id)
			other_state = self._get_actor_state(other_actor)
			if other_state is None:
				continue
			dx = float(other_state["x"] - ego_state["x"])
			dy = float(other_state["y"] - ego_state["y"])
			distance = math.sqrt(dx ** 2 + dy ** 2)
			world_bearing = math.degrees(math.atan2(dy, dx))
			relative_bearing = ((world_bearing - ego_state["yaw"] + 180.0) % 360.0) - 180.0
			unit_x = dx / max(distance, 1e-6)
			unit_y = dy / max(distance, 1e-6)
			ego_closing = ego_state["velocity"][0] * unit_x + ego_state["velocity"][1] * unit_y
			other_closing = other_state["velocity"][0] * unit_x + other_state["velocity"][1] * unit_y
			closing_speed = float(ego_closing - other_closing)
			interactions.append(
				{
					"other_ego_id": int(other_id),
					"distance_m": float(distance),
					"relative_bearing_deg": float(relative_bearing),
					"relative_speed_mps": float(other_state["speed"] - ego_speed),
					"closing_speed_mps": closing_speed,
					"other_world": other_state,
				}
			)
			stats = self.debug_stats[ego_id]
			min_distance = stats["min_other_ego_distance"]
			if min_distance is None or distance < min_distance:
				stats["min_other_ego_distance"] = float(distance)
		interactions.sort(key=lambda item: item["distance_m"])
		return interactions

	def _collect_nearby_actors(self, ego_id, radius=None, max_count=3):
		radius = float(radius or self.debug_nearby_actor_radius)
		world = CarlaDataProvider.get_world()
		ego_actor = CarlaDataProvider.get_hero_actor(hero_id=ego_id)
		if world is None or ego_actor is None:
			return []
		hero_ids = set()
		for other_id in range(self.ego_vehicles_num):
			hero_actor = CarlaDataProvider.get_hero_actor(hero_id=other_id)
			if hero_actor is not None:
				hero_ids.add(hero_actor.id)
		ego_loc = ego_actor.get_location()
		neighbors = []
		for actor in world.get_actors():
			if actor.id in hero_ids or actor.id == ego_actor.id:
				continue
			type_id = str(getattr(actor, "type_id", ""))
			if type_id.startswith("sensor.") or type_id.startswith("traffic.") or type_id.startswith("spectator"):
				continue
			try:
				loc = actor.get_location()
			except RuntimeError:
				continue
			distance = ego_loc.distance(loc)
			if distance > radius:
				continue
			try:
				velocity = actor.get_velocity()
				speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
			except RuntimeError:
				speed = 0.0
			role_name = ""
			if hasattr(actor, "attributes"):
				role_name = actor.attributes.get("role_name", "")
			neighbors.append(
				{
					"actor_id": int(actor.id),
					"type_id": type_id,
					"role_name": role_name,
					"distance_m": float(distance),
					"location": [float(loc.x), float(loc.y), float(loc.z)],
					"speed_mps": float(speed),
				}
			)
		neighbors.sort(key=lambda item: item["distance_m"])
		if neighbors:
			stats = self.debug_stats[ego_id]
			min_distance = stats["min_nearby_actor_distance"]
			if min_distance is None or neighbors[0]["distance_m"] < min_distance:
				stats["min_nearby_actor_distance"] = float(neighbors[0]["distance_m"])
		return neighbors[:max_count]

	def _format_interactions(self, interactions):
		if not interactions:
			return "none"
		parts = []
		for item in interactions[:2]:
			parts.append(
				f"ego{item['other_ego_id']}:d={item['distance_m']:.2f}m,bearing={item['relative_bearing_deg']:.1f}deg,"
				f"rel_speed={item['relative_speed_mps']:.2f}mps"
			)
		return "; ".join(parts)

	def _format_nearby_actors(self, neighbors):
		if not neighbors:
			return "none"
		parts = []
		for item in neighbors[:3]:
			type_name = item["type_id"].split(".")[-1]
			role = f",role={item['role_name']}" if item["role_name"] else ""
			parts.append(
				f"{type_name}#{item['actor_id']}:d={item['distance_m']:.2f}m,speed={item['speed_mps']:.2f}mps{role}"
			)
		return "; ".join(parts)

	def _update_debug_stats(
		self,
		ego_id,
		*,
		target_norm=None,
		distance_to_waypoint=None,
		route_centerline_distance=None,
		inference_s=None,
		ego_world=None,
		collision_event=None,
	):
		stats = self.debug_stats[ego_id]
		if target_norm is not None:
			stats["max_target_norm"] = max(float(stats["max_target_norm"]), float(target_norm))
		if distance_to_waypoint is not None:
			stats["max_distance_to_waypoint"] = max(
				float(stats["max_distance_to_waypoint"]),
				float(distance_to_waypoint),
			)
		if route_centerline_distance is not None:
			stats["max_route_centerline_distance"] = max(
				float(stats["max_route_centerline_distance"]),
				float(route_centerline_distance),
			)
		if inference_s is not None:
			stats["max_inference_s"] = max(float(stats["max_inference_s"]), float(inference_s))
		if ego_world is not None:
			stats["max_abs_roll_deg"] = max(
				float(stats["max_abs_roll_deg"]),
				abs(float(ego_world.get("roll", 0.0))),
			)
			stats["max_abs_pitch_deg"] = max(
				float(stats["max_abs_pitch_deg"]),
				abs(float(ego_world.get("pitch", 0.0))),
			)
		if collision_event is not None and collision_event.get("has_collision"):
			stats["max_collision_impulse"] = max(
				float(stats["max_collision_impulse"]),
				float(collision_event.get("normal_impulse_magnitude", 0.0)),
			)

	def _emit_divergence_debug(self, ego_id, tick_data, waypoints, control, metadata, interactions, neighbors):
		target_norm = float(np.linalg.norm(tick_data["target_point"]))
		waypoint0_norm = float(np.linalg.norm(waypoints[0]))
		route_centerline_distance = tick_data.get("nearest_route_centerline_distance")
		if route_centerline_distance is not None:
			route_centerline_distance = float(route_centerline_distance)
		state = self.debug_runtime_state[ego_id]
		stats = self.debug_stats[ego_id]
		if route_centerline_distance is not None:
			diverging = route_centerline_distance >= self.debug_route_centerline_threshold
		else:
			diverging = target_norm >= self.debug_divergence_threshold
		collapse = diverging and waypoint0_norm <= self.debug_collapse_waypoint_norm
		weak_recovery = diverging and waypoint0_norm < 3.0 and abs(float(control.steer)) <= self.debug_weak_recovery_steer
		if diverging and not state["diverging"]:
			stats["divergence_events"] += 1
			self._debug(
				f"[DIVERGENCE_ENTER] ego={ego_id} step={self.step[ego_id]} target_norm={target_norm:.2f} "
				f"route_centerline={route_centerline_distance if route_centerline_distance is not None else -1.0:.2f} "
				f"target_local={self._format_xy(tick_data['target_point'])} world={tick_data.get('ego_world')} "
				f"next_world={tick_data.get('next_waypoint_world')} interactions={self._format_interactions(interactions)} "
				f"nearby={self._format_nearby_actors(neighbors)} collision={self._format_collision(tick_data.get('collision_event'))}"
			)
			self._write_debug_event(
				ego_id,
				"divergence_enter",
				{
					"target_norm": target_norm,
					"nearest_route_centerline_distance": route_centerline_distance,
					"nearest_route_centerline_projection": tick_data.get("nearest_route_centerline_projection"),
					"target_point": tick_data["target_point"],
					"distance_to_waypoint": tick_data.get("distance_to_waypoint"),
					"ego_world": tick_data.get("ego_world"),
					"next_waypoint_world": tick_data.get("next_waypoint_world"),
					"interactions": interactions,
					"nearby_actors": neighbors,
					"collision_event": tick_data.get("collision_event"),
				},
			)
		elif (not diverging) and state["diverging"]:
			self._debug(
				f"[DIVERGENCE_EXIT] ego={ego_id} step={self.step[ego_id]} target_norm={target_norm:.2f} "
				f"route_centerline={route_centerline_distance if route_centerline_distance is not None else -1.0:.2f} "
				f"distance_to_waypoint={float(tick_data.get('distance_to_waypoint', target_norm)):.2f}"
			)
		if collapse and not state["collapse"]:
			stats["collapse_events"] += 1
			self._debug(
				f"[COLLAPSE] ego={ego_id} step={self.step[ego_id]} target_norm={target_norm:.2f} "
				f"route_centerline={route_centerline_distance if route_centerline_distance is not None else -1.0:.2f} "
				f"wp0_norm={waypoint0_norm:.3f} desired_speed={float(metadata['desired_speed']):.3f} "
				f"control={self._format_control(control)}"
			)
		if weak_recovery and not state["weak_recovery"]:
			stats["weak_recovery_events"] += 1
			self._debug(
				f"[WEAK_RECOVERY] ego={ego_id} step={self.step[ego_id]} target_norm={target_norm:.2f} "
				f"route_centerline={route_centerline_distance if route_centerline_distance is not None else -1.0:.2f} "
				f"wp0={self._format_xy(waypoints[0])} steer={float(control.steer):.3f} "
				f"desired_speed={float(metadata['desired_speed']):.3f}"
			)
		state["diverging"] = diverging
		state["collapse"] = collapse
		state["weak_recovery"] = weak_recovery
		state["last_target_norm"] = target_norm
		state["last_distance_to_waypoint"] = float(tick_data.get("distance_to_waypoint", target_norm))
		state["last_route_centerline_distance"] = route_centerline_distance

	def sensors(self):
		return [
			{
				"type": "sensor.camera.rgb",
				"x": 1.3,
				"y": 0.0,
				"z": 2.3,
				"roll": 0.0,
				"pitch": 0.0,
				"yaw": 0.0,
				"width": 1200,
				"height": 900,
				"fov": 100,
				"id": "rgb_front",
			},
			{
				"type": "sensor.camera.rgb",
				"x": 1.3,
				"y": 0.0,
				"z": 2.3,
				"roll": 0.0,
				"pitch": 0.0,
				"yaw": -60.0,
				"width": 400,
				"height": 300,
				"fov": 100,
				"id": "rgb_left",
			},
			{
				"type": "sensor.camera.rgb",
				"x": 1.3,
				"y": 0.0,
				"z": 2.3,
				"roll": 0.0,
				"pitch": 0.0,
				"yaw": 60.0,
				"width": 400,
				"height": 300,
				"fov": 100,
				"id": "rgb_right",
			},
			{
				"type": "sensor.camera.rgb",
				"x": -1.3,
				"y": 0.0,
				"z": 2.3,
				"roll": 0.0,
				"pitch": 0.0,
				"yaw": 180.0,
				"width": 400,
				"height": 300,
				"fov": 100,
				"id": "rgb_rear",
			},
			{
				"type": "sensor.lidar.ray_cast",
				"x": 1.3,
				"y": 0.0,
				"z": 2.5,
				"roll": 0.0,
				"pitch": 0.0,
				"yaw": -90.0,
				"id": "lidar",
			},
			{
				"type": "sensor.other.imu",
				"x": 0.0,
				"y": 0.0,
				"z": 0.0,
				"roll": 0.0,
				"pitch": 0.0,
				"yaw": 0.0,
				"sensor_tick": 0.05,
				"id": "imu",
			},
			{
				"type": "sensor.other.gnss",
				"x": 0.0,
				"y": 0.0,
				"z": 0.0,
				"roll": 0.0,
				"pitch": 0.0,
				"yaw": 0.0,
				"sensor_tick": 0.01,
				"id": "gps",
			},
			{
				"type": "sensor.other.collision",
				"x": 0.0,
				"y": 0.0,
				"z": 0.0,
				"roll": 0.0,
				"pitch": 0.0,
				"yaw": 0.0,
				"id": "collision",
			},
			{"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"},
		]

	def tick(self, ego_id, input_data):

		rgb_front = cv2.cvtColor(input_data[f"rgb_front_{ego_id}"][1][:, :, :3], cv2.COLOR_BGR2RGB)
		rgb_left = cv2.cvtColor(input_data[f"rgb_left_{ego_id}"][1][:, :, :3], cv2.COLOR_BGR2RGB)
		rgb_right = cv2.cvtColor(
			input_data[f"rgb_right_{ego_id}"][1][:, :, :3], cv2.COLOR_BGR2RGB
		)
		rgb_rear = cv2.cvtColor(
			input_data[f"rgb_rear_{ego_id}"][1][:, :, :3], cv2.COLOR_BGR2RGB
		)
		gps = input_data[f"gps_{ego_id}"][1][:2]
		speed = input_data[f"speed_{ego_id}"][1]["speed"]
		compass = input_data[f"imu_{ego_id}"][1][-1]
		if (
			math.isnan(compass) == True
		):  # It can happen that the compass sends nan for a few frames
			compass = 0.0

		result = {
			"rgb_front": rgb_front,
			"rgb_left": rgb_left,
			"rgb_right": rgb_right,
			'rgb_rear': rgb_rear,
			"gps": gps,
			"speed": speed,
			"compass": compass,
		}

		pos = self._get_position(result)

		lidar_data = input_data[f'lidar_{ego_id}'][1]
		result['raw_lidar'] = lidar_data

		lidar_unprocessed = lidar_data[..., :4]
		if self.prev_lidar[ego_id] is not None:
			lidar_unprocessed_full = np.concatenate([lidar_unprocessed, self.prev_lidar[ego_id]])
		else:
			lidar_unprocessed_full = lidar_unprocessed
		self.prev_lidar[ego_id] = lidar_unprocessed

		lidar_processed, num_points= lidar_to_raw_features(lidar_unprocessed_full)
		result['lidar'] = lidar_processed
		result['num_points'] = num_points

		result["gps"] = pos
		route_out = self._route_planner.run_step(pos, ego_id)
		next_wp, next_cmd = route_out[0], route_out[1]
		next_wp_world = route_out[2] if len(route_out) > 2 else None
		result["next_waypoint"] = next_wp
		result["next_command"] = next_cmd.value
		result["next_command_name"] = self._command_name(next_cmd)
		result["next_command_enum"] = next_cmd
		result['measurements'] = [pos[0], pos[1], compass, speed]
		result['speed'] = speed

		theta = compass + np.pi / 2
		R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

		local_command_point = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])
		local_command_point = R.T.dot(local_command_point)
		result["target_point"] = local_command_point
		result["next_waypoint_world"] = None
		if next_wp_world is not None:
			result["next_waypoint_world"] = {
				"x": float(next_wp_world.location.x),
				"y": float(next_wp_world.location.y),
				"z": float(next_wp_world.location.z),
				"yaw": float(next_wp_world.rotation.yaw),
			}
		result["route_remaining"] = len(self._route_planner.route[ego_id])
		result["distance_to_waypoint"] = float(np.linalg.norm(next_wp - pos))
		ego_actor = CarlaDataProvider.get_hero_actor(hero_id=ego_id)
		result["ego_world"] = self._get_actor_state(ego_actor)
		route_tracking = self._route_planner.get_route_tracking_state(pos, ego_id)
		result.update(route_tracking)
		collision_payload = input_data.get(f"collision_{ego_id}", (None, None))[1]
		result["collision_event"] = self._normalize_collision_event(collision_payload)
		if result["collision_event"]["has_collision"]:
			event_frame = result["collision_event"].get("event_frame")
			if event_frame != self.last_collision_event_frame[ego_id]:
				self.last_collision_event_frame[ego_id] = event_frame
				self.debug_stats[ego_id]["collision_events"] += 1
				self._debug(
					f"[COLLISION] ego={ego_id} step={self.step[ego_id]} {self._format_collision(result['collision_event'])} "
					f"ego_world={result['ego_world']}"
				)
				self._write_debug_event(
					ego_id,
					"collision",
					{
						"collision_event": result["collision_event"],
						"ego_world": result["ego_world"],
						"nearest_route_centerline_distance": result.get("nearest_route_centerline_distance"),
						"nearest_route_centerline_projection": result.get("nearest_route_centerline_projection"),
						"distance_to_waypoint": result.get("distance_to_waypoint"),
					},
				)
		if self.route_debuggers[ego_id] is not None:
			self.route_debuggers[ego_id].log_tick(
				ego_id=ego_id,
				tick_index=self.step[ego_id],
				timestamp=time.time(),
				gps=gps,
				local_pos=pos,
				speed=speed,
				compass=compass,
				near_node=next_wp,
				near_command=next_cmd,
				near_world=next_wp_world,
				extra_fields={
					"target_point_local": self._json_safe(local_command_point),
					"route_remaining": int(result["route_remaining"]),
					"ego_world": result["ego_world"],
					"next_command_value": int(next_cmd.value),
					"nearest_route_point_index": result.get("nearest_route_point_index"),
					"nearest_route_point_distance": result.get("nearest_route_point_distance"),
					"nearest_route_centerline_distance": result.get("nearest_route_centerline_distance"),
					"nearest_route_centerline_projection": result.get("nearest_route_centerline_projection"),
					"nearest_route_segment_index": result.get("nearest_route_segment_index"),
					"nearest_route_segment_t": result.get("nearest_route_segment_t"),
					"nearest_route_heading_deg": result.get("nearest_route_heading_deg"),
					"collision_event": result.get("collision_event"),
				},
			)
		if (
			self.step[ego_id] < 25
			or self.step[ego_id] % self.debug_log_every == 0
			or float(result.get("nearest_route_centerline_distance") or 0.0) >= self.debug_route_centerline_threshold
		):
			self._debug(
				f"[TICK] ego={ego_id} step={self.step[ego_id]} cmd={result['next_command_name']}/{result['next_command']} "
				f"speed={float(speed):.2f} pos={self._format_xy(pos)} target_local={self._format_xy(local_command_point)} "
				f"target_norm={float(np.linalg.norm(local_command_point)):.2f} "
				f"route_centerline={float(result.get('nearest_route_centerline_distance') or 0.0):.2f} "
				f"route_remaining={result['route_remaining']} next_world={result['next_waypoint_world']} "
				f"collision={self._format_collision(result.get('collision_event'))} ego_world={result['ego_world']}"
			)
		self._update_debug_stats(
			ego_id,
			target_norm=float(np.linalg.norm(local_command_point)),
			distance_to_waypoint=result["distance_to_waypoint"],
			route_centerline_distance=result.get("nearest_route_centerline_distance"),
			ego_world=result.get("ego_world"),
			collision_event=result.get("collision_event"),
		)

		return result


	def update_and_collect(self, image_embeds, ego_id):
		if 'lane' in self.curr_instruction[ego_id]: # change lane
			sample_rate = 1
		else:
			sample_rate = 2
		sample_rate = 2
		ego_feature_buffer = self.visual_feature_buffer[ego_id]
		ego_feature_buffer.append(image_embeds)
		result = ego_feature_buffer[::self.sample_rate]
		if (len(ego_feature_buffer) -1) % self.sample_rate != 0:
			result.append(ego_feature_buffer[-1])
		return torch.stack(result, 1)

	@torch.no_grad()
	def run_step(self, input_data, timestamp):
		control_all = []
		for ego_id in range(self.ego_vehicles_num):
			if CarlaDataProvider.get_hero_actor(hero_id=ego_id) is not None:
				control = self.run_step_single_vehicle(ego_id, input_data, timestamp)
				control_all.append(control)
			else:
				control_all.append(None)
		return control_all


	@torch.no_grad()
	def run_step_single_vehicle(self, ego_id, input_data, timestamp):
		if not self.initialized:
			self._init()

		self.step[ego_id] += 1
		self._debug(f"[STEP] ego={ego_id} step={self.step[ego_id]}")

		try:
			tick_data = self.tick(ego_id, input_data)
		except KeyError as exc:
			self._debug(f"[SENSOR_MISSING] ego={ego_id} missing_sensor={exc} action=full_brake")
			self._write_debug_event(
				ego_id,
				"sensor_missing",
				{
					"missing_sensor": str(exc),
					"traceback": traceback.format_exc(),
				},
			)
			control = carla.VehicleControl()
			control.steer = 0.0
			control.throttle = 0.0
			control.brake = 1.0
			return control

		if self.step[ego_id] < 20:
			control = carla.VehicleControl()
			control.steer = float(0)
			control.throttle = float(0)
			control.brake = float(1)
			self._debug(f"[WARMUP] ego={ego_id} step={self.step[ego_id]} action=full_brake")
			return control

		realtime_mode = os.environ.get('REALTIME_MODE', '0')
		reuse_reason = None
		if realtime_mode == '1':
			if self.step[ego_id] < self.next_action_step[ego_id] and self.step[ego_id]:
				reuse_reason = "realtime_wait"
		else:
			if self.debug_force_fresh_control:
				reuse_reason = None
			elif self.step[ego_id] % 5 != 0 and self.step[ego_id]:
				reuse_reason = "non_realtime_skip"
		if reuse_reason is not None:
			prev_control = self.prev_control[ego_id]
			if prev_control is not None:
				self.reused_control_steps[ego_id] += 1
				self.debug_stats[ego_id]["reuse_steps"] += 1
				if self.reused_control_steps[ego_id] == 1 or self.reused_control_steps[ego_id] % self.debug_log_every == 0:
					self._debug(
						f"[REUSE] ego={ego_id} step={self.step[ego_id]} reason={reuse_reason} "
						f"reuse_count={self.reused_control_steps[ego_id]} prev_control={self._format_control(prev_control)} "
						f"next_action_step={self.next_action_step[ego_id]}"
					)
				return prev_control
			self._debug(
				f"[REUSE_MISS] ego={ego_id} step={self.step[ego_id]} reason={reuse_reason} "
				f"prev_control=None forcing_inference=1"
			)


		start_time = time.time()
		reused_span = self.reused_control_steps[ego_id]
		self.reused_control_steps[ego_id] = 0
		self.last_inference_step[ego_id] = self.step[ego_id]
		self.debug_stats[ego_id]["inference_calls"] += 1
		velocity = tick_data["speed"]
		command = tick_data["next_command"]
		command_name = tick_data["next_command_name"]
		target_norm = float(np.linalg.norm(tick_data["target_point"]))
		interactions = self._collect_ego_interactions(ego_id)
		neighbors = self._collect_nearby_actors(ego_id)
		self._debug(
			f"[INFER_START] ego={ego_id} step={self.step[ego_id]} cmd={command_name}/{command} "
			f"speed={float(velocity):.2f} target_local={self._format_xy(tick_data['target_point'])} "
			f"target_norm={target_norm:.2f} distance_to_waypoint={float(tick_data['distance_to_waypoint']):.2f} "
			f"route_centerline={float(tick_data.get('nearest_route_centerline_distance') or 0.0):.2f} "
			f"route_remaining={tick_data['route_remaining']} reuse_span={reused_span} "
			f"fresh_control={int(self.debug_force_fresh_control and realtime_mode != '1')} "
			f"interactions={self._format_interactions(interactions)} nearby={self._format_nearby_actors(neighbors)} "
			f"collision={self._format_collision(tick_data.get('collision_event'))}"
		)

		rgb_front = (
			self.rgb_front_transform(Image.fromarray(tick_data["rgb_front"]))
			.unsqueeze(0)
			.cuda()
			.float()
		)
		rgb_left = (
			self.rgb_left_transform(Image.fromarray(tick_data["rgb_left"]))
			.unsqueeze(0)
			.cuda()
			.float()
		)
		rgb_right = (
			self.rgb_right_transform(Image.fromarray(tick_data["rgb_right"]))
			.unsqueeze(0)
			.cuda()
			.float()
		)
		rgb_rear = (
			self.rgb_right_transform(Image.fromarray(tick_data["rgb_rear"]))
			.unsqueeze(0)
			.cuda()
			.float()
		)
		rgb_center = (
			self.rgb_center_transform(Image.fromarray(cv2.resize(tick_data["rgb_front"], (800, 600))))
			.unsqueeze(0)
			.cuda()
			.float()
		)

		planner = self._instruction_planners[ego_id]
		last_instruction = planner.command2instruct(self.town_id, tick_data, self._route_planner.route, ego_id)
		sampled_scenarios = self.sampled_scenarios[0] if self.sampled_scenarios else []
		last_notice = planner.pos2notice(sampled_scenarios, tick_data)
		last_traffic_light_notice = planner.traffic_notice(tick_data, ego_id)
		last_misleading_instruction = planner.command2mislead(self.town_id, tick_data, ego_id)

		if last_notice == '':
			last_notice = last_traffic_light_notice

		self.debug_dict[ego_id]["num_last_misleading_instruction_exist"] += bool(last_misleading_instruction)          #TODO debug

		instruction_update_reason = "hold"
		if self.curr_instruction[ego_id] != last_instruction or len(self.visual_feature_buffer[ego_id]) > 400:
			if self.enable_misleading_instruction and self.remaining_misleading_frames[ego_id] > 0:
				self.remaining_misleading_frames[ego_id] = self.remaining_misleading_frames[ego_id] - 1
				instruction_update_reason = "misleading_hold"
			else:
				self.active_misleading_instruction[ego_id] = False
				self.remaining_misleading_frames[ego_id] = 0
				if self.enable_misleading_instruction and last_misleading_instruction != '' and random.random() < 0.2:
					self.curr_instruction[ego_id] = last_misleading_instruction
					self.active_misleading_instruction[ego_id] = True
					self.remaining_misleading_frames[ego_id] = 4
					self.debug_stats[ego_id]["misleading_activations"] += 1
					instruction_update_reason = "misleading_new"
				else:
					self.curr_instruction[ego_id] = last_instruction
					if last_misleading_instruction != '' and not self.enable_misleading_instruction:
						instruction_update_reason = "misleading_disabled_route_refresh"
					else:
						instruction_update_reason = "route_refresh"
				self.visual_feature_buffer[ego_id] = []
				self.curr_notice[ego_id] = ''
				self.curr_notice_frame_id[ego_id] = -1
		self._debug(
			f"[PROMPT] ego={ego_id} step={self.step[ego_id]} cmd={command_name}/{command} "
			f"route_instruction={last_instruction!r} chosen_instruction={self.curr_instruction[ego_id]!r} "
			f"reason={instruction_update_reason} notice={last_notice!r} traffic_notice={last_traffic_light_notice!r} "
			f"misleading_enabled={int(self.enable_misleading_instruction)} "
			f"misleading_candidate={last_misleading_instruction!r} active={int(self.active_misleading_instruction[ego_id])} "
			f"remaining_misleading_frames={self.remaining_misleading_frames[ego_id]}"
		)

		input_data = {}
		input_data["rgb_front"] = rgb_front
		input_data["rgb_left"] = rgb_left
		input_data["rgb_right"] = rgb_right
		input_data["rgb_center"] = rgb_center
		input_data["rgb_rear"] = rgb_rear
		input_data['target_point'] = torch.tensor(tick_data['target_point']).cuda().view(1,2).float()
		input_data["lidar"] = (
			torch.from_numpy(tick_data["lidar"]).float().cuda().unsqueeze(0)
		)
		input_data['num_points'] = torch.tensor([tick_data['num_points']]).cuda().unsqueeze(0)
		input_data['velocity'] = torch.tensor([tick_data['speed']]).cuda().view(1, 1).float()

		input_data['text_input'] = [self.curr_instruction[ego_id]]
		image_embeds = self.net.visual_encoder(input_data)
		image_embeds = self.update_and_collect(image_embeds, ego_id)
		input_data['valid_frames'] = [image_embeds.size(1)]

		self.debug_dict[ego_id]["num_instruction"] +=1             #TODO debug

		if last_notice != '' and last_notice != self.curr_notice[ego_id]:
			new_notice_flag = True
			self.curr_notice[ego_id] = last_notice
			self.curr_notice_frame_id[ego_id] = image_embeds.size(1) - 1
			self._debug(
				f"[NOTICE] ego={ego_id} step={self.step[ego_id]} notice={last_notice!r} "
				f"notice_frame_id={self.curr_notice_frame_id[ego_id]}"
			)
		else:
			new_notice_flag = False

		if self.agent_use_notice:
			input_data['notice_text'] = [self.curr_notice[ego_id]]
			input_data['notice_frame_id'] = [self.curr_notice_frame_id[ego_id]]

		with torch.cuda.amp.autocast(enabled=True):
			waypoints, is_end = self.net(input_data, inference_mode=True, image_embeds=image_embeds)

		waypoints = waypoints[-1]
		waypoints = waypoints.view(5, 2)
		waypoints_debug = waypoints.detach().float().cpu().numpy()
		waypoints_local_for_adapter = np.asarray(waypoints_debug, dtype=np.float64).copy()
		if waypoints_local_for_adapter.size:
			waypoints_local_for_adapter[:, 1] *= -1.0
		self.last_predicted_waypoints_local[ego_id] = waypoints_local_for_adapter
		self.last_predicted_waypoints_step[ego_id] = int(self.step[ego_id])
		end_prob = self.softmax(is_end)[-1][1] # is_end[1] means the prob of the frame is the last frame

		steer, throttle, brake, metadata = self.control_pid(waypoints, velocity, ego_id)

		# If the predictor collapses to tiny waypoints at standstill while the route target
		# is clearly ahead, allow a gentle launch nudge to escape deadlock.
		if (
			self.step[ego_id] < 220
			and float(velocity) < 0.15
			and float(metadata["desired_speed"]) < float(self.lm_config.brake_speed)
			and float(brake) > 0.5
			and target_norm > 1.0
		):
			throttle = max(float(throttle), min(0.35, float(self.lm_config.max_throttle)))
			brake = 0.0
			metadata["launch_assist"] = True
		else:
			metadata["launch_assist"] = False

		self._debug(
			f"[MODEL] ego={ego_id} step={self.step[ego_id]} cmd={command_name}/{command} "
			f"waypoints[0]=[{waypoints_debug[0,0]:.2f}, {waypoints_debug[0,1]:.2f}] "
			f"target_norm={target_norm:.2f}, route_centerline={float(tick_data.get('nearest_route_centerline_distance') or 0.0):.2f}, "
			f"desired_speed={metadata['desired_speed']:.3f}, "
			f"speed_ratio={metadata['speed_ratio']:.3f}, velocity={float(velocity):.2f}, "
			f"raw_control=(steer={steer:.3f},throttle={float(throttle):.3f},brake={float(brake):.3f}), "
			f"launch_assist={int(metadata['launch_assist'])} end_prob={float(end_prob):.3f}"
		)

		if end_prob > 0.75:
			self.visual_feature_buffer[ego_id] = []
			self.curr_notice[ego_id] = ''
			self.curr_notice_frame_id[ego_id] = -1
			self._debug(f"[SEQUENCE_RESET] ego={ego_id} step={self.step[ego_id]} end_prob={float(end_prob):.3f}")

		if brake < 0.05:
			brake = 0.0
		if brake > 0.1:
			throttle = 0.0

		control = carla.VehicleControl()
		control.steer = float(steer) * 0.8
		control.throttle = float(throttle)
		control.brake = float(brake)
		self._emit_divergence_debug(ego_id, tick_data, waypoints_debug, control, metadata, interactions, neighbors)

		end_time = time.time()
		inference_s = end_time - start_time

		self.next_action_step[ego_id] = self.step[ego_id] + int(inference_s * 20)
		self._update_debug_stats(
			ego_id,
			target_norm=target_norm,
			distance_to_waypoint=tick_data["distance_to_waypoint"],
			route_centerline_distance=tick_data.get("nearest_route_centerline_distance"),
			inference_s=inference_s,
			ego_world=tick_data.get("ego_world"),
			collision_event=tick_data.get("collision_event"),
		)
		self._debug(
			f"[CONTROL] ego={ego_id} step={self.step[ego_id]} control={self._format_control(control)} "
			f"next_action_step={self.next_action_step[ego_id]} inference_s={inference_s:.3f} "
			f"reuse_span={reused_span} valid_frames={image_embeds.size(1)} new_notice={int(new_notice_flag)} "
			f"route_centerline={float(tick_data.get('nearest_route_centerline_distance') or 0.0):.2f}"
		)
		self._write_debug_event(
			ego_id,
			"inference",
			{
				"command_value": int(command),
				"command_name": command_name,
				"target_norm": target_norm,
				"target_point_local": tick_data["target_point"],
				"distance_to_waypoint": tick_data["distance_to_waypoint"],
				"nearest_route_point_index": tick_data.get("nearest_route_point_index"),
				"nearest_route_point_distance": tick_data.get("nearest_route_point_distance"),
				"nearest_route_centerline_distance": tick_data.get("nearest_route_centerline_distance"),
				"nearest_route_centerline_projection": tick_data.get("nearest_route_centerline_projection"),
				"nearest_route_segment_index": tick_data.get("nearest_route_segment_index"),
				"nearest_route_segment_t": tick_data.get("nearest_route_segment_t"),
				"nearest_route_heading_deg": tick_data.get("nearest_route_heading_deg"),
				"route_remaining": tick_data["route_remaining"],
				"ego_world": tick_data.get("ego_world"),
				"next_waypoint_local": tick_data["next_waypoint"],
				"next_waypoint_world": tick_data.get("next_waypoint_world"),
				"collision_event": tick_data.get("collision_event"),
				"interactions": interactions,
				"nearby_actors": neighbors,
				"route_instruction": last_instruction,
				"chosen_instruction": self.curr_instruction[ego_id],
				"instruction_update_reason": instruction_update_reason,
				"misleading_candidate": last_misleading_instruction,
				"misleading_enabled": bool(self.enable_misleading_instruction),
				"misleading_active": bool(self.active_misleading_instruction[ego_id]),
				"remaining_misleading_frames": int(self.remaining_misleading_frames[ego_id]),
				"notice": last_notice,
				"traffic_notice": last_traffic_light_notice,
				"new_notice": bool(new_notice_flag),
				"valid_frames": int(image_embeds.size(1)),
				"waypoints": waypoints_debug,
				"metadata": metadata,
				"control": {
					"steer": float(control.steer),
					"throttle": float(control.throttle),
					"brake": float(control.brake),
				},
				"inference_s": float(inference_s),
				"reuse_span": int(reused_span),
				"end_prob": float(end_prob),
			},
		)

		display_data = {}
		display_data['rgb_front'] = cv2.resize(tick_data['rgb_front'], (1200, 900))
		display_data['rgb_left'] = cv2.resize(tick_data['rgb_left'], (280, 210))
		display_data['rgb_right'] = cv2.resize(tick_data['rgb_right'], (280, 210))
		display_data['rgb_center'] = cv2.resize(tick_data['rgb_front'][330:570, 480:720], (210, 210))

		if self.active_misleading_instruction[ego_id]:
			self.debug_dict[ego_id]["num_misleading"] +=1             #TODO debug
			display_data['instruction'] = "Instruction: [Misleading] %s" % input_data['text_input'][0]
		else:
			display_data['instruction'] = "Instruction: %s" % input_data['text_input'][0]
		display_data['time'] = 'Time: %.3f. Frames: %d. End prob: %.2f' % (timestamp, len(self.visual_feature_buffer[ego_id]), end_prob)
		display_data['meta_control'] = 'Throttle: %.2f. Steer: %.2f. Brake: %.2f' %(
			control.steer, control.throttle, control.brake
		)
		display_data['waypoints'] = 'Waypoints: (%.1f, %.1f), (%.1f, %.1f)' % (waypoints[0,0], -waypoints[0,1], waypoints[1,0], -waypoints[1,1])
		display_data['notice'] = "Notice: %s" % last_notice
		if self.save_path is not None:
			route_debug_snapshot = self._build_route_debug_snapshot(
				ego_id,
				tick_data,
				planner,
				last_instruction,
				last_notice,
				last_traffic_light_notice,
				last_misleading_instruction,
				instruction_update_reason,
				waypoints_debug,
				control,
				metadata,
			)
			tick_data['route_debug_snapshot'] = route_debug_snapshot
			tick_data['route_debug_surface'] = self._render_route_debug_panel(route_debug_snapshot)
		surface = self._hic.run_interface(display_data)
		tick_data['surface'] = surface
		self.prev_control[ego_id] = control
		if SAVE_PATH is not None:
			self.save(ego_id, tick_data)
		return control

	def save(self, ego_id, tick_data):
		frame = (self.step[ego_id] - 20)
		meta_dir = self.save_path / f"meta_{ego_id}"
		Image.fromarray(tick_data["surface"]).save(meta_dir / ("%04d.jpg" % frame))
		route_debug_surface = tick_data.get("route_debug_surface")
		if route_debug_surface is not None:
			Image.fromarray(route_debug_surface).save(meta_dir / ("%04d_route_debug.jpg" % frame))
		route_debug_snapshot = tick_data.get("route_debug_snapshot")
		if route_debug_snapshot is not None:
			(meta_dir / ("%04d_route_debug.json" % frame)).write_text(
				json.dumps(self._json_safe(route_debug_snapshot), indent=2),
				encoding="utf-8",
			)
		return

	def destroy(self):
		for ego_id in range(self.ego_vehicles_num):
			if self.route_debuggers[ego_id] is not None:
				self.route_debuggers[ego_id].finalize()
				self.route_debuggers[ego_id] = None
		summary = {
			"lat_ref": float(self.lat_ref),
			"lon_ref": float(self.lon_ref),
			"route_debug_enabled": bool(self.route_debug_enabled),
			"debug_log_every": int(self.debug_log_every),
			"debug_force_fresh_control": bool(self.debug_force_fresh_control),
			"misleading_instruction_enabled": bool(self.enable_misleading_instruction),
			"debug_route_centerline_threshold": float(self.debug_route_centerline_threshold),
			"ego_stats": {f"ego_{ego_id}": self.debug_stats[ego_id] for ego_id in range(self.ego_vehicles_num)},
		}
		if self.debug_base_dir is not None:
			summary_path = self.debug_base_dir / "summary.json"
			summary_path.write_text(json.dumps(self._json_safe(summary), indent=2), encoding="utf-8")
			self._debug(f"[SUMMARY] wrote_debug_summary={summary_path}")
		del self.net

	def control_pid(self, waypoints, velocity, ego_id):
		'''
		Predicts vehicle control with a PID controller.
		Args:
			waypoints (tensor): predicted waypoints
			velocity (tensor): speedometer input
		'''
		assert(waypoints.size(0)==5)
		waypoints = waypoints.data.cpu().numpy()

		# flip y is (forward is negative in our waypoints)
		waypoints[:,1] *= -1
		speed = float(velocity)

		# Use the full predicted trajectory instead of only wp0->wp1. This is less jittery
		# and reduces false brake toggles from tiny first-segment predictions.
		if waypoints.shape[0] > 1:
			segment_speeds = np.linalg.norm(np.diff(waypoints, axis=0), axis=1) * 2.0
			desired_speed = float(np.mean(segment_speeds))
		else:
			desired_speed = 0.0

		if not np.isfinite(desired_speed):
			desired_speed = 0.0
		desired_speed = max(0.0, desired_speed)
		speed_ratio = float(speed / max(desired_speed, 1e-3))
		brake = desired_speed < float(self.lm_config.brake_speed) or speed_ratio > float(self.lm_config.brake_ratio)

		aim = (waypoints[1] + waypoints[0]) / 2.0
		angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
		if(speed < 0.01):
			angle = np.array(0.0) # When we don't move we don't want the angle error to accumulate in the integral
		steer = self.turn_controller[ego_id].step(angle)
		steer = np.clip(steer, -1.0, 1.0)

		delta = float(np.clip(desired_speed - speed, 0.0, self.lm_config.clip_delta))
		throttle = self.speed_controller[ego_id].step(delta)
		throttle = np.clip(throttle, 0.0, self.lm_config.max_throttle)
		throttle = throttle if not brake else 0.0

		metadata = {
			'speed': float(speed),
			'steer': float(steer),
			'throttle': float(throttle),
			'brake': float(brake),
			'wp_2': tuple(waypoints[1].astype(np.float64)),
			'wp_1': tuple(waypoints[0].astype(np.float64)),
			'desired_speed': float(desired_speed),
			'speed_ratio': float(speed_ratio),
			'angle': float(angle.astype(np.float64)),
			'aim': tuple(aim.astype(np.float64)),
			'delta': float(delta),
		}

		return steer, throttle, brake, metadata
