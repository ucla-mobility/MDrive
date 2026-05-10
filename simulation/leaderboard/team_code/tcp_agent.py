import os
import json
import datetime
import pathlib
import time
import cv2
import carla
from collections import deque
import math
from collections import OrderedDict
import yaml

import torch
torch.backends.cudnn.benchmark = True
import carla
import numpy as np
from PIL import Image
from torchvision import transforms as T

from leaderboard.autoagents import autonomous_agent

from leaderboard.utils.route_manipulation import downsample_route
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
import sys
sys.path.append('simulation/assets/TCP')

from TCP.model import TCP
from TCP.config import GlobalConfig
from team_code.planner import RoutePlanner


SAVE_PATH = os.environ.get('SAVE_PATH', None)


def _env_int(name: str, default: int, minimum: int = 1) -> int:
	try:
		value = int(os.environ.get(name, default))
	except (TypeError, ValueError):
		value = default
	return max(minimum, value)


def _env_float(name: str, default: float) -> float:
	try:
		return float(os.environ.get(name, default))
	except (TypeError, ValueError):
		return default


def get_entry_point():
	return 'TCPAgent'


class TCPAgent(autonomous_agent.AutonomousAgent):
	def setup(self, path_to_conf_file, ego_vehicles_num=1):
		self.agent_name='TCP'
		self.ego_vehicles_num= ego_vehicles_num
		self._timing_debug_stdout = os.environ.get("TCP_TIMING_DEBUG", "").lower() in ("1", "true", "yes")

		self.track = autonomous_agent.Track.SENSORS
		self.alpha = 0.3
		self.status = [0] * 10
		self.steer_step = [0] * 10
		self.last_moving_status = [0] * 10
		self.last_moving_step = [-1] * 10
		self.last_steers = [deque()] * 10

		self.config_path = path_to_conf_file
		self.step = -1
		self.wall_start = time.time()
		self.initialized = False

		self.config_TCP = GlobalConfig()
		self.net = TCP(self.config_TCP)

		if self._timing_debug_stdout:
			print('path_to_conf_file:', path_to_conf_file)
		self.config = yaml.load(open(path_to_conf_file),Loader=yaml.Loader)
		ckpt = torch.load(self.config['ckpt_path'])
		# ckpt = torch.load(path_to_conf_file)
		ckpt = ckpt["state_dict"]

		# self.config['target_point_distance'] = model_config.get('target_distance',50)
		# self.target_distance = model_config.get('target_distance',50)

		new_state_dict = OrderedDict()
		for key, value in ckpt.items():
			new_key = key.replace("model.","")
			new_state_dict[new_key] = value
		self.net.load_state_dict(new_state_dict, strict = False)
		self.net.cuda()
		self.net.eval()

		if self._timing_debug_stdout:
			print(
				f"\033[0;31;40m[PERCEPTION]\033[0m Model param count:{sum([m.numel() for m in self.net.parameters()])}\n"
			)

		self.takeover = False
		self.stop_time = 0
		self.takeover_time = 0

		self.save_path = None
		self.logreplayimages_path = None
		self.calibration_path = None
		self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
		self.pid_metadata = {}
		# Per-step timing accumulators (only populated on inference steps that are not skipped)
		self._step_timings = {
			'tick_preprocess': [],   # tick() sensor read + route planner + optional cv2.resize
			'tensor_build':    [],   # CPU->GPU tensor construction
			'net_forward':     [],   # TCP model forward pass (ResNet34 + GRU loops)
			'process_action':  [],   # process_action() + control_pid() combined
		}
		self._timing_log_path = None
		self._saved_first_tick = [False] * self.ego_vehicles_num
		self.save_interval = _env_int("TCP_SAVE_INTERVAL", 4, minimum=1)
		self.model_rgb_width = _env_int("TCP_MODEL_RGB_WIDTH", 900, minimum=1)
		self.model_rgb_height = _env_int("TCP_MODEL_RGB_HEIGHT", 256, minimum=1)
		self.capture_sensor_frames = os.environ.get("TCP_CAPTURE_SENSOR_FRAMES", "").lower() in ("1", "true", "yes")
		self.capture_logreplay_images = (
			os.environ.get("TCP_CAPTURE_LOGREPLAY_IMAGES", "").lower() in ("1", "true", "yes")
		)
		# TCP-Vid mode: same TCP driving policy, but also captures additional
		# cinematic CARLA RGB cameras every tick and auto-compiles each into
		# an mp4. The agent_wrapper detects the 'cine_' id prefix and zeroes
		# out lens distortion + chromatic aberration + motion blur on every
		# cine_* sensor (planner-facing rgb/bev keep the heavy distortion the
		# TCP model was trained on, so driving behavior is unchanged).
		self.video_mode = os.environ.get("TCP_VIDEO_MODE", "").lower() in ("1", "true", "yes")
		# Cinematic CARLA RGB cameras (per-ego). Saved to <save_path>/<id>_<ego>/.
		# All real photographic captures — no schematics.
		self.cine_cam_ids = [
			'cine_front',     # forward camera over the hood
			'cine_chase',     # third-person chase, behind & above
			'cine_side',      # side tracking shot (driver-side profile)
			'cine_top_close', # real top-down photo, zoomed in (orbital ~30 m)
			'cine_top_wide',  # real top-down photo, wider context (~120 m)
		]
		if self.video_mode:
			# Save every agent tick — cinematic-quality is the whole point.
			self.save_interval = 1
		self.video_fps = _env_int("TCP_VIDEO_FPS", 20, minimum=1)

		if SAVE_PATH is not None:
			now = datetime.datetime.now()
			string = pathlib.Path(os.environ['ROUTES']).stem + '_'
			string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))

			self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
			self.save_path.mkdir(parents=True, exist_ok=False)
			self._timing_log_path = self.save_path / 'timing_tcp.log'
			self.logreplayimages_path = self.save_path / "logreplayimages"
			if self.capture_logreplay_images or self.capture_sensor_frames:
				self.logreplayimages_path.mkdir(parents=True, exist_ok=True)
			# Backward-compatible alias for older capture paths.
			self.calibration_path = self.logreplayimages_path

			for ego_id in range(self.ego_vehicles_num):
				(self.save_path / 'meta_{}'.format(ego_id)).mkdir()
				if not self.video_mode:
					(self.save_path / 'rgb_{}'.format(ego_id)).mkdir()
					(self.save_path / 'bev_{}'.format(ego_id)).mkdir()
				if self.capture_logreplay_images and self.logreplayimages_path is not None:
					(self.logreplayimages_path / 'logreplay_rgb_{}'.format(ego_id)).mkdir(parents=True, exist_ok=True)
				if self.video_mode:
					for cam in self.cine_cam_ids:
						(self.save_path / f'{cam}_{ego_id}').mkdir(parents=True, exist_ok=True)
			if self.video_mode:
				(self.save_path / 'videos').mkdir(parents=True, exist_ok=True)

	def _init(self):
		self._route_planner = RoutePlanner(4.0, 10.0) # (4.0, 50.0)
		self._route_planner.set_route(self._global_plan, True)
		self.next_action_step = [-1] * 10
		self.prev_control = [0] * 10

		self.initialized = True

	def _get_position(self, tick_data):
		gps = tick_data['gps']
		gps = (gps - self._route_planner.mean) * self._route_planner.scale

		return gps

	def _append_route_diag_file(self, payload):
		if self.save_path is None:
			return
		try:
			diag_path = self.save_path / "tcp_route_diag.jsonl"
			with open(diag_path, "a") as handle:
				handle.write(json.dumps(payload, sort_keys=True) + "\n")
		except Exception:
			pass

	def _emit_route_command_diagnostic(self, ego_id, pos, next_wp, next_cmd):
		route_debug = {}
		try:
			route_debug = self._route_planner.get_last_debug_snapshot(ego_id)
		except Exception:
			route_debug = {}

		hero = None
		try:
			hero = CarlaDataProvider.get_hero_actor(hero_id=ego_id)
		except Exception:
			hero = None

		world_plan_len = None
		if hasattr(self, "_global_plan_world_coord_all") and self._global_plan_world_coord_all is not None:
			try:
				world_plan_len = len(self._global_plan_world_coord_all[ego_id])
			except Exception:
				world_plan_len = None

		downsampled_plan_len = None
		if hasattr(self, "_global_plan_world_coord") and self._global_plan_world_coord is not None:
			try:
				downsampled_plan_len = len(self._global_plan_world_coord[ego_id])
			except Exception:
				downsampled_plan_len = None

		payload = {
			"event": "tcp_route_command_none",
			"ego_id": int(ego_id),
			"step": int(self.step),
			"sim_time_s": round(float(time.time() - self.wall_start), 4),
			"gps_x": round(float(pos[0]), 4),
			"gps_y": round(float(pos[1]), 4),
			"next_wp_x": round(float(next_wp[0]), 4),
			"next_wp_y": round(float(next_wp[1]), 4),
			"next_cmd_is_none": next_cmd is None,
			"hero_actor_id": None if hero is None else int(hero.id),
			"global_plan_len": downsampled_plan_len,
			"global_plan_world_coord_all_len": world_plan_len,
			"route_debug": route_debug,
		}
		rendered = json.dumps(payload, sort_keys=True)
		print("[TCP_ROUTE_DIAG] " + rendered)
		self._append_route_diag_file(payload)

	def sensors(self):
		rgb_w = _env_int("TCP_RGB_WIDTH", 900, minimum=1)
		rgb_h = _env_int("TCP_RGB_HEIGHT", 256, minimum=1)
		rgb_fov = _env_float("TCP_RGB_FOV", 100.0)
		rgb_x = _env_float("TCP_RGB_X", -1.5)
		rgb_y = _env_float("TCP_RGB_Y", 0.0)
		rgb_z = _env_float("TCP_RGB_Z", 2.0)
		rgb_roll = _env_float("TCP_RGB_ROLL", 0.0)
		rgb_pitch = _env_float("TCP_RGB_PITCH", 0.0)
		rgb_yaw = _env_float("TCP_RGB_YAW", 0.0)
		bev_w = _env_int("TCP_BEV_WIDTH", 256, minimum=1)
		bev_h = _env_int("TCP_BEV_HEIGHT", 256, minimum=1)
		bev_fov = _env_float("TCP_BEV_FOV", 120.0)
		# Keep log-replay capture independent from model-facing RGB defaults (900x256),
		# so saved comparison videos are not squeezed.
		logreplay_rgb_w = _env_int("TCP_LOGREPLAY_RGB_WIDTH", 1600, minimum=1)
		logreplay_rgb_h = _env_int("TCP_LOGREPLAY_RGB_HEIGHT", 900, minimum=1)
		logreplay_rgb_fov = _env_float("TCP_LOGREPLAY_RGB_FOV", 58.0)
		logreplay_rgb_x = _env_float("TCP_LOGREPLAY_RGB_X", rgb_x + 0.90)
		logreplay_rgb_y = _env_float("TCP_LOGREPLAY_RGB_Y", rgb_y)
		logreplay_rgb_z = _env_float("TCP_LOGREPLAY_RGB_Z", rgb_z + 0.05)
		logreplay_rgb_roll = _env_float("TCP_LOGREPLAY_RGB_ROLL", rgb_roll)
		logreplay_rgb_pitch = _env_float("TCP_LOGREPLAY_RGB_PITCH", rgb_pitch - 1.2)
		logreplay_rgb_yaw = _env_float("TCP_LOGREPLAY_RGB_YAW", rgb_yaw)
		sensors = [
		{
			'type': 'sensor.camera.rgb',
			'x': rgb_x, 'y': rgb_y, 'z': rgb_z,
			'roll': rgb_roll, 'pitch': rgb_pitch, 'yaw': rgb_yaw,
			'width': rgb_w, 'height': rgb_h, 'fov': rgb_fov,
			'id': 'rgb'
			},
		{
			'type': 'sensor.camera.rgb',
			'x': 0.0, 'y': 0.0, 'z': 120.0,
			'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
			'width': bev_w, 'height': bev_h, 'fov': bev_fov, 
			# 4096*4096 for visualization; 256*256 default
			'id': 'bev'
			},
		{
			'type': 'sensor.other.imu',
			'x': 0.0, 'y': 0.0, 'z': 0.0,
			'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
			'sensor_tick': 0.05,
			'id': 'imu'
			},
		{
			'type': 'sensor.other.gnss',
			'x': 0.0, 'y': 0.0, 'z': 0.0,
			'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
			'sensor_tick': 0.01,
			'id': 'gps'
			},
		{
			'type': 'sensor.speedometer',
			'reading_frequency': 20,
			'id': 'speed'
			}
		]
		if self.capture_logreplay_images:
			sensors.append(
				{
					'type': 'sensor.camera.rgb',
					'x': logreplay_rgb_x, 'y': logreplay_rgb_y, 'z': logreplay_rgb_z,
					'roll': logreplay_rgb_roll, 'pitch': logreplay_rgb_pitch, 'yaw': logreplay_rgb_yaw,
					'width': logreplay_rgb_w, 'height': logreplay_rgb_h, 'fov': logreplay_rgb_fov,
					'id': 'logreplay_rgb'
				}
			)
		if self.video_mode:
			sensors.extend(self._cinematic_sensors())
		return sensors

	def _cinematic_sensors(self):
		# CARLA vehicle frame: +X forward, +Y right, +Z up.
		# Per-ego suffixes (_0, _1, ...) are added by the leaderboard wrapper,
		# so here we just use the bare 'id'. Every sensor here uses the
		# 'cine_' id prefix; agent_wrapper.py detects that and zeroes out
		# lens distortion + chromatic aberration + motion blur (vs. the
		# heavy planner-facing defaults).
		cine_w = _env_int("TCP_VID_CINE_WIDTH", 1920, minimum=1)
		cine_h = _env_int("TCP_VID_CINE_HEIGHT", 1080, minimum=1)
		# 1) Front camera over the hood
		front_x = _env_float("TCP_VID_FRONT_X", -1.5)
		front_z = _env_float("TCP_VID_FRONT_Z", 2.0)
		front_fov = _env_float("TCP_VID_FRONT_FOV", 50.0)
		# 2) Chase camera — far enough back & high enough that the full
		# back of the ego car (including trunk) is visible at frame bottom.
		chase_x = _env_float("TCP_VID_CHASE_X", -14.0)
		chase_z = _env_float("TCP_VID_CHASE_Z", 6.0)
		chase_pitch = _env_float("TCP_VID_CHASE_PITCH", -15.0)
		chase_fov = _env_float("TCP_VID_CHASE_FOV", 50.0)
		# 3) Side tracking shot (driver-side profile of the ego). Camera x
		# is aligned with the vehicle origin so the body sits horizontally
		# centered (yaw=90 → image-LEFT = vehicle forward; offsetting x in
		# either direction shifts the car off-center in that direction).
		side_x = _env_float("TCP_VID_SIDE_X", 0.0)
		side_y = _env_float("TCP_VID_SIDE_Y", -9.0)
		side_z = _env_float("TCP_VID_SIDE_Z", 1.5)
		side_yaw = _env_float("TCP_VID_SIDE_YAW", 90.0)   # face the car
		side_pitch = _env_float("TCP_VID_SIDE_PITCH", 0.0)
		side_fov = _env_float("TCP_VID_SIDE_FOV", 60.0)
		# 4) Real top-down photo, zoomed in on the ego (orbital ~30 m above).
		# Narrow FOV from this height keeps it close to orthographic.
		top_close_z = _env_float("TCP_VID_TOP_CLOSE_Z", 30.0)
		top_close_fov = _env_float("TCP_VID_TOP_CLOSE_FOV", 50.0)
		# 5) Real top-down photo, wider context (~120 m high).
		top_wide_z = _env_float("TCP_VID_TOP_WIDE_Z", 120.0)
		top_wide_fov = _env_float("TCP_VID_TOP_WIDE_FOV", 50.0)
		return [
			{
				'type': 'sensor.camera.rgb',
				'x': front_x, 'y': 0.0, 'z': front_z,
				'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
				'width': cine_w, 'height': cine_h, 'fov': front_fov,
				'id': 'cine_front',
			},
			{
				'type': 'sensor.camera.rgb',
				'x': chase_x, 'y': 0.0, 'z': chase_z,
				'roll': 0.0, 'pitch': chase_pitch, 'yaw': 0.0,
				'width': cine_w, 'height': cine_h, 'fov': chase_fov,
				'id': 'cine_chase',
			},
			{
				'type': 'sensor.camera.rgb',
				'x': side_x, 'y': side_y, 'z': side_z,
				'roll': 0.0, 'pitch': side_pitch, 'yaw': side_yaw,
				'width': cine_w, 'height': cine_h, 'fov': side_fov,
				'id': 'cine_side',
			},
			{
				'type': 'sensor.camera.rgb',
				'x': 0.0, 'y': 0.0, 'z': top_close_z,
				'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
				'width': cine_w, 'height': cine_h, 'fov': top_close_fov,
				'id': 'cine_top_close',
			},
			{
				'type': 'sensor.camera.rgb',
				'x': 0.0, 'y': 0.0, 'z': top_wide_z,
				'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
				'width': cine_w, 'height': cine_h, 'fov': top_wide_fov,
				'id': 'cine_top_wide',
			},
		]

	def tick(self, ego_id, input_data):
		# self.step += 1

		rgb = cv2.cvtColor(input_data['rgb_{}'.format(ego_id)][1][:, :, :3], cv2.COLOR_BGR2RGB)
		bev = cv2.cvtColor(input_data['bev_{}'.format(ego_id)][1][:, :, :3], cv2.COLOR_BGR2RGB)
		logreplay_rgb = None
		logreplay_key = 'logreplay_rgb_{}'.format(ego_id)
		if logreplay_key in input_data:
			logreplay_rgb = cv2.cvtColor(input_data[logreplay_key][1][:, :, :3], cv2.COLOR_BGR2RGB)
		gps = input_data['gps_{}'.format(ego_id)][1][:2]
		speed = input_data['speed_{}'.format(ego_id)][1]['speed']
		compass = input_data['imu_{}'.format(ego_id)][1][-1]

		if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
			compass = 0.0

		result = {
				'rgb': rgb,
				'gps': gps,
				'speed': speed,
				'compass': compass,
				'bev': bev,
				'logreplay_rgb': logreplay_rgb
				}
		if self.video_mode:
			for cam in self.cine_cam_ids:
				cam_key = f'{cam}_{ego_id}'
				if cam_key in input_data:
					result[cam] = cv2.cvtColor(
						input_data[cam_key][1][:, :, :3], cv2.COLOR_BGR2RGB
					)
				else:
					result[cam] = None
		
		pos = self._get_position(result)
		result['gps'] = pos
		next_wp, next_cmd = self._route_planner.run_step(pos, vehicle_num=ego_id)
		if next_cmd is None:
			self._emit_route_command_diagnostic(ego_id, pos, next_wp, next_cmd)
			from agents.navigation.local_planner import RoadOption
			next_cmd = RoadOption.LANEFOLLOW
		result['next_command'] = next_cmd.value


		theta = compass + np.pi/2
		R = np.array([
			[np.cos(theta), -np.sin(theta)],
			[np.sin(theta), np.cos(theta)]
			])

		local_command_point = np.array([next_wp[0]-pos[0], next_wp[1]-pos[1]])
		local_command_point = R.T.dot(local_command_point)
		result['target_point'] = tuple(local_command_point)

		return result

	@torch.no_grad()
	def run_step(self, input_data, timestamp):
		control_all = []
		self.step += 1
		for ego_id in range(self.ego_vehicles_num):
			hero = CarlaDataProvider.get_hero_actor(hero_id=ego_id)
			if hero is None or not getattr(hero, "is_alive", True):
				control_all.append(None)
				continue
			if f'rgb_{ego_id}' not in input_data:
				control_all.append(None)
				continue
			control = self.run_step_single_vehicle(ego_id, input_data, timestamp)
			control_all.append(control)
		return control_all

	@torch.no_grad()
	def run_step_single_vehicle(self, ego_id, input_data, timestamp):
		if not self.initialized:
			self._init()
		# print('input_data:', input_data.keys())
		_t_tick_start = time.time()
		tick_data = self.tick(ego_id, input_data)
		if SAVE_PATH is not None and not self.capture_sensor_frames:
			if not self._saved_first_tick[ego_id]:
				self.save(ego_id, tick_data)
				self._saved_first_tick[ego_id] = True
			elif self.step % self.save_interval == 0:
				self.save(ego_id, tick_data)
		rgb_input = tick_data['rgb']
		if rgb_input is not None and (
			rgb_input.shape[1] != self.model_rgb_width
			or rgb_input.shape[0] != self.model_rgb_height
		):
			rgb_input = cv2.resize(
				rgb_input,
				(self.model_rgb_width, self.model_rgb_height),
				interpolation=cv2.INTER_AREA,
			)
		_t_tick_end = time.time()
		if self.step < self.config_TCP.seq_len:
			rgb = self._im_transform(rgb_input).unsqueeze(0)

			control = carla.VehicleControl()
			control.steer = 0.0
			control.throttle = 0.0
			control.brake = 0.0
			
			return control

		import os
		realtime_mode = os.environ.get('REALTIME_MODE', '0')
		force_fresh_every_frame = os.environ.get('OPENLOOP_FORCE_FRESH_INFERENCE', '0') == '1'
		if realtime_mode == '1':
			# REAL-TIME
			if self.step < self.next_action_step[ego_id] and self.step > 0:
				# return the previous control signal.
				return self.prev_control[ego_id]
		else:
			# NON-REAL-TIME
			if not force_fresh_every_frame and self.step % 4 != 0 and self.step > 0:
				# return the previous control signal.
				return self.prev_control[ego_id]

		_t_tensor_start = time.time()
		gt_velocity = torch.FloatTensor([tick_data['speed']]).to('cuda', dtype=torch.float32)
		command = tick_data['next_command']
		if command < 0:
			command = 4
		command -= 1
		assert command in [0, 1, 2, 3, 4, 5]
		cmd_one_hot = [0] * 6
		cmd_one_hot[command] = 1
		cmd_one_hot = torch.tensor(cmd_one_hot).view(1, 6).to('cuda', dtype=torch.float32)
		speed = torch.FloatTensor([float(tick_data['speed'])]).view(1,1).to('cuda', dtype=torch.float32)
		speed = speed / 12
		rgb = self._im_transform(rgb_input).unsqueeze(0).to('cuda', dtype=torch.float32)

		tick_data['target_point'] = [torch.FloatTensor([tick_data['target_point'][0]]),
										torch.FloatTensor([tick_data['target_point'][1]])]
		target_point = torch.stack(tick_data['target_point'], dim=1).to('cuda', dtype=torch.float32)
		state = torch.cat([speed, target_point, cmd_one_hot], 1)
		_t_tensor_end = time.time()

		_t_net_start = time.time()
		pred= self.net(rgb, state, target_point)
		_t_net_end = time.time()

		_t_action_start = time.time()
		steer_ctrl, throttle_ctrl, brake_ctrl, metadata = self.net.process_action(pred, tick_data['next_command'], gt_velocity, target_point)

		steer_traj, throttle_traj, brake_traj, metadata_traj = self.net.control_pid(pred['pred_wp'], gt_velocity, target_point)
		_t_action_end = time.time()
		if brake_traj < 0.05: brake_traj = 0.0
		if throttle_traj > brake_traj: brake_traj = 0.0

		self.pid_metadata = metadata_traj
		control = carla.VehicleControl()

		#### Note: the following two lines are added by Weibo to reach the unstable high-speed TCP model.
		# steer_traj *= 4
		# steer_ctrl *= 4
		if self.status[ego_id] == 0:
			self.alpha = 0.3
			self.pid_metadata['agent'] = 'traj'
			control.steer = np.clip(self.alpha*steer_ctrl + (1-self.alpha)*steer_traj, -1, 1)
			control.throttle = np.clip(self.alpha*throttle_ctrl + (1-self.alpha)*throttle_traj, 0, 0.75)
			control.brake = np.clip(self.alpha*brake_ctrl + (1-self.alpha)*brake_traj, 0, 1)
		else:
			self.alpha = 0.3
			self.pid_metadata['agent'] = 'ctrl'
			control.steer = np.clip(self.alpha*steer_traj + (1-self.alpha)*steer_ctrl, -1, 1)
			control.throttle = np.clip(self.alpha*throttle_traj + (1-self.alpha)*throttle_ctrl, 0, 0.75)
			control.brake = np.clip(self.alpha*brake_traj + (1-self.alpha)*brake_ctrl, 0, 1)


		self.pid_metadata['theta'] = tick_data['compass'] + np.pi/2
		self.pid_metadata['gps_x'] = tick_data['gps'][0]
		self.pid_metadata['gps_y'] = tick_data['gps'][1]

		self.pid_metadata['steer_ctrl'] = float(steer_ctrl)
		self.pid_metadata['steer_traj'] = float(steer_traj)
		self.pid_metadata['throttle_ctrl'] = float(throttle_ctrl)
		self.pid_metadata['throttle_traj'] = float(throttle_traj)
		self.pid_metadata['brake_ctrl'] = float(brake_ctrl)
		self.pid_metadata['brake_traj'] = float(brake_traj)

		if control.brake > 0.5:
			control.throttle = float(0)

		if len(self.last_steers[ego_id]) >= 20:
			self.last_steers[ego_id].popleft()
		self.last_steers[ego_id].append(abs(float(control.steer)))
		#chech whether ego is turning
		# num of steers larger than 0.1
		num = 0
		for s in self.last_steers[ego_id]:
			if s > 0.10:
				num += 1
		if num > 10:
			self.status[ego_id] = 1
			self.steer_step[ego_id] += 1

		else:
			self.status[ego_id] = 0

		self.pid_metadata['status'] = self.status[ego_id]

		self.prev_control[ego_id] = control
		_dt_tick   = _t_tick_end    - _t_tick_start
		_dt_tensor = _t_tensor_end  - _t_tensor_start
		_dt_net    = _t_net_end     - _t_net_start
		_dt_action = _t_action_end  - _t_action_start
		_dt_total  = _t_action_end  - _t_tensor_start
		self._step_timings['tick_preprocess'].append(_dt_tick)
		self._step_timings['tensor_build'].append(_dt_tensor)
		self._step_timings['net_forward'].append(_dt_net)
		self._step_timings['process_action'].append(_dt_action)
		_timing_msg = (
			f"[TCP timing] step={self.step} ego={ego_id} "
			f"tick={_dt_tick*1000:.1f}ms "
			f"tensor={_dt_tensor*1000:.1f}ms "
			f"net={_dt_net*1000:.1f}ms "
			f"action={_dt_action*1000:.1f}ms "
			f"total={_dt_total*1000:.1f}ms"
		)
		if self._timing_debug_stdout:
			print(_timing_msg)
		if self._timing_log_path is not None:
			try:
				with open(self._timing_log_path, 'a') as _f:
					_f.write(_timing_msg + '\n')
			except Exception:
				pass
		self.next_action_step[ego_id] = self.step + _dt_total * 2
		return control

	def save(self, ego_id, tick_data):
		frame = self.step

		if self.save_path is None:
			return

		frame_name = '%04d.png' % frame
		# In video mode, suppress the stretched 900x256 RGB and 256x256 BEV
		# PNG dumps — those are TCP's model inputs, not viewing artifacts.
		# The cinematic cameras below are the only on-disk frames.
		if not self.video_mode:
			rgb_image = Image.fromarray(tick_data['rgb'])
			bev_image = Image.fromarray(tick_data['bev'])
			rgb_image.save(self.save_path / 'rgb_{}'.format(ego_id) / frame_name)
			bev_image.save(self.save_path / 'bev_{}'.format(ego_id) / frame_name)
		if self.capture_logreplay_images and self.logreplayimages_path is not None:
			logreplay_rgb = tick_data.get('logreplay_rgb')
			if logreplay_rgb is not None:
				Image.fromarray(logreplay_rgb).save(
					self.logreplayimages_path / 'logreplay_rgb_{}'.format(ego_id) / frame_name
				)
		if self.video_mode:
			for cam in self.cine_cam_ids:
				img = tick_data.get(cam)
				if img is not None:
					Image.fromarray(img).save(
						self.save_path / f'{cam}_{ego_id}' / frame_name
					)

		outfile = open(self.save_path / 'meta_{}'.format(ego_id) / ('%04d.json' % frame), 'w')
		json.dump(self.pid_metadata, outfile, indent=4)
		outfile.close()

	def _compile_videos(self):
		"""Walk each cinematic camera output dir and emit an mp4 per (id, ego)."""
		if self.save_path is None:
			return
		videos_dir = self.save_path / 'videos'
		videos_dir.mkdir(parents=True, exist_ok=True)
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		for ego_id in range(self.ego_vehicles_num):
			for cam in self.cine_cam_ids:
				frames_dir = self.save_path / f'{cam}_{ego_id}'
				if not frames_dir.is_dir():
					continue
				png_paths = sorted(frames_dir.glob('*.png'))
				if not png_paths:
					continue
				first = cv2.imread(str(png_paths[0]))
				if first is None:
					continue
				h, w = first.shape[:2]
				out_path = videos_dir / f'{cam}_{ego_id}.mp4'
				writer = cv2.VideoWriter(str(out_path), fourcc, float(self.video_fps), (w, h))
				if not writer.isOpened():
					print(f"[TCP-Vid] failed to open VideoWriter for {out_path}")
					continue
				try:
					writer.write(first)
					for png_path in png_paths[1:]:
						img = cv2.imread(str(png_path))
						if img is None:
							continue
						if img.shape[:2] != (h, w):
							img = cv2.resize(img, (w, h))
						writer.write(img)
				finally:
					writer.release()
				print(f"[TCP-Vid] wrote {out_path} ({len(png_paths)} frames @ {self.video_fps} fps)")

	def destroy(self):
		# Print and save per-step timing summary before cleanup
		_summary_lines = ["[TCP timing summary]"]
		for _phase, _durations in self._step_timings.items():
			if _durations:
				_n = len(_durations)
				_mean = sum(_durations) / _n * 1000
				_max  = max(_durations) * 1000
				_min  = min(_durations) * 1000
				_summary_lines.append(
					f"  {_phase}: n={_n} mean={_mean:.1f}ms max={_max:.1f}ms min={_min:.1f}ms"
				)
		_summary = '\n'.join(_summary_lines)
		if self._timing_debug_stdout:
			print(_summary)
		if self._timing_log_path is not None:
			try:
				with open(self._timing_log_path, 'a') as _f:
					_f.write(_summary + '\n')
			except Exception:
				pass
		if self.video_mode:
			try:
				self._compile_videos()
			except Exception as exc:
				print(f"[TCP-Vid] video compilation failed: {exc}")
		del self.net
		torch.cuda.empty_cache()

	# def set_global_plan(self, global_plan_gps, global_plan_world_coord, wp=None):
	# 	"""
	# 	Set the plan (route) for the agent
	# 	"""
	# 	ds_ids = downsample_route(global_plan_world_coord, 50)
	# 	self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]
	# 	self._global_plan = [global_plan_gps[x] for x in ds_ids]
