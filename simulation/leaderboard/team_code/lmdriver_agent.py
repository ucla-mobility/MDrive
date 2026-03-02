import os
import json
import random
import datetime
import pathlib
import time
import imp
from collections import deque
import math

import yaml
import cv2
import torch
import carla
import numpy as np
from PIL import Image
# from easydict import EasyDict
from torchvision import transforms

from leaderboard.autoagents import autonomous_agent
from team_code.lm_planner import RoutePlanner, InstructionPlanner
from team_code.lm_pid_controller import PIDController
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
	lidar[:num_points,:4] = lidar_xyzr
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
		self.step = [-1] * 10  # Per-vehicle step counters
		self.wall_start = time.time()
		self.initialized = False
		self.rgb_front_transform = create_carla_rgb_transform(224)
		self.rgb_left_transform = create_carla_rgb_transform(128)
		self.rgb_right_transform = create_carla_rgb_transform(128)
		self.rgb_center_transform = create_carla_rgb_transform(128, need_scale=False)

		self.active_misleading_instruction = [False] * 10
		self.remaining_misleading_frames = [0] * 10

		self.visual_feature_buffer = []

		# self.lm_config = imp.load_source("MainModel", '/GPFS/data/changxingliu/V2Xverse/simulation/leaderboard/team_code/lmdriver_config.py').GlobalConfig()
		self.lm_config = imp.load_source("MainModel", path_to_conf_file).GlobalConfig()
		self.config = self.lm_config
		self.ego_vehicles_num = ego_vehicles_num #1
		self.turn_controller = PIDController(K_P=self.lm_config.turn_KP, K_I=self.lm_config.turn_KI, K_D=self.lm_config.turn_KD, n=self.lm_config.turn_n)
		self.speed_controller = PIDController(K_P=self.lm_config.speed_KP, K_I=self.lm_config.speed_KI, K_D=self.lm_config.speed_KD, n=self.lm_config.speed_n)

		model_cls = registry.get_model_class('vicuna_drive')
		self.agent_use_notice = self.lm_config.agent_use_notice
		self.traffic_light_notice = [''] * 10
		self.curr_notice = [''] * 10
		self.curr_notice_frame_id = [-1] * 10
		self.sample_rate = self.lm_config.sample_rate * 2 # The frequency of CARLA simulation is 20Hz

		print('build model...')
		model = model_cls(preception_model=self.lm_config.preception_model,
						  preception_model_ckpt=self.lm_config.preception_model_ckpt,
						  llm_model=self.lm_config.llm_model,
						  max_txt_len=64,
						  use_notice_prompt=self.lm_config.agent_use_notice,
						  )
		self.net = model
		print('load model...')
		self.net.load_state_dict(torch.load(self.lm_config.lmdrive_ckpt)["model"], strict=False)
		self.net.cuda()
		self.net.eval()
		self.softmax = torch.nn.Softmax(dim=1)
		self.prev_lidar = None
		self.prev_control = [None] * 10
		self.curr_instruction = ['Drive Safely'] * 10
		self.sampled_scenarios = None
		self.instruction = ''

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

			print("Data save path:", string)

			self.save_path = pathlib.Path(SAVE_PATH) / string
			self.save_path.mkdir(parents=True, exist_ok=False)
			for ego_id in range(ego_vehicles_num):
				(self.save_path / f"meta_{ego_id}").mkdir(parents=True, exist_ok=False)

	def _init(self):
		self._route_planner = RoutePlanner(5, 50.0)
		self._route_planner.set_route(self._global_plan, True)
		self._instruction_planner = InstructionPlanner(notice_light_switch = True) # self.scenario_cofing_name, 
		self.next_action_step = [-1] * 10
		self.initialized = True
		random.seed(''.join([str(x[0]) for x in self._global_plan]))
  
		self.debug_dict = {}
		for id in range(10):
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
		if self.prev_lidar is not None:
			lidar_unprocessed_full = np.concatenate([lidar_unprocessed, self.prev_lidar])
		else:
			lidar_unprocessed_full = lidar_unprocessed
		self.prev_lidar = lidar_unprocessed

		lidar_processed, num_points= lidar_to_raw_features(lidar_unprocessed_full)
		result['lidar'] = lidar_processed
		result['num_points'] = num_points

		result["gps"] = pos
		next_wp, next_cmd = self._route_planner.run_step(pos, ego_id)
		result["next_waypoint"] = next_wp
		result["next_command"] = next_cmd.value
		result['measurements'] = [pos[0], pos[1], compass, speed]
		result['speed'] = speed

		theta = compass + np.pi / 2
		R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

		local_command_point = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])
		local_command_point = R.T.dot(local_command_point)
		result["target_point"] = local_command_point

		return result


	def update_and_collect(self, image_embeds, ego_id):
		if 'lane' in self.curr_instruction[ego_id]: # change lane
			sample_rate = 1
		else:
			sample_rate = 2
		sample_rate = 2
		self.visual_feature_buffer.append(image_embeds)
		result = self.visual_feature_buffer[::self.sample_rate]
		if (len(self.visual_feature_buffer) -1) % self.sample_rate != 0:
			result.append(self.visual_feature_buffer[-1])
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
		print(f"[DEBUG LMDRIVE] ego={ego_id}, step={self.step[ego_id]}")
		
		try:
			tick_data = self.tick(ego_id, input_data)
		except KeyError as e:
			print(f"[WARN LMDRIVE] Missing sensor data {e} for ego {ego_id}, sending full brake.")
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
			print(f"[DEBUG LMDRIVE] ego={ego_id}, step={self.step[ego_id]} < 20, returning BRAKE control")
			return control

		realtime_mode = os.environ.get('REALTIME_MODE', '0')
		if realtime_mode == '1':
			# REAL-TIME
			if self.step[ego_id] < self.next_action_step[ego_id] and self.step[ego_id]:
				return self.prev_control[ego_id]
		else:
			# NON-REAL-TIME
			if self.step[ego_id] % 5 != 0 and self.step[ego_id]:
				return self.prev_control[ego_id]

		
		start_time = time.time()
		velocity = tick_data["speed"]
		command = tick_data["next_command"]
		
		print(ego_id, command)

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
		
		# self.town_id = "Town07"
		last_instruction = self._instruction_planner.command2instruct(self.town_id, tick_data, self._route_planner.route, ego_id)
		if ego_id == 1:
			print(last_instruction)
		last_notice = self._instruction_planner.pos2notice(self.sampled_scenarios[0], tick_data)
		last_traffic_light_notice = self._instruction_planner.traffic_notice(tick_data, ego_id)
		last_misleading_instruction = self._instruction_planner.command2mislead(self.town_id, tick_data, ego_id)

		if last_notice == '':
			last_notice = last_traffic_light_notice

		self.debug_dict[ego_id]["num_last_misleading_instruction_exist"] += bool(last_misleading_instruction)          #TODO debug

		
		if self.curr_instruction[ego_id] != last_instruction or len(self.visual_feature_buffer) > 400:
			#self.debug_dict[ego_id]["num_self.curr_instruction != last_instruction"] +=1             #TODO debug
			
			if self.remaining_misleading_frames[ego_id] > 0:  
				self.remaining_misleading_frames[ego_id] = self.remaining_misleading_frames[ego_id] - 1
			else:
				self.active_misleading_instruction[ego_id] = False
				if last_misleading_instruction!= '' and random.random() < 0.2:
					self.curr_instruction[ego_id] = last_misleading_instruction
					self.active_misleading_instruction[ego_id] = True 
					self.remaining_misleading_frames[ego_id] = 4
				else:
					self.curr_instruction[ego_id] = last_instruction 
				self.visual_feature_buffer = []
				self.curr_notice[ego_id] = ''
				self.curr_notice_frame_id[ego_id] = -1
		
		print(self.curr_instruction[ego_id])
		
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
		else:
			new_notice_flag = False

		if self.agent_use_notice:
			input_data['notice_text'] = [self.curr_notice[ego_id]]
			input_data['notice_frame_id'] = [self.curr_notice_frame_id[ego_id]]

		with torch.cuda.amp.autocast(enabled=True):
			waypoints, is_end = self.net(input_data, inference_mode=True, image_embeds=image_embeds)

		waypoints = waypoints[-1]
		waypoints = waypoints.view(5, 2)
		end_prob = self.softmax(is_end)[-1][1] # is_end[1] means the prob of the frame is the last frame

		steer, throttle, brake, metadata = self.control_pid(waypoints, velocity)
		print(f"[DEBUG LMDRIVE] ego={ego_id}, waypoints[0]=[{waypoints[0,0]:.2f}, {waypoints[0,1]:.2f}], velocity={velocity:.2f}, raw_control: steer={steer:.3f}, throttle={throttle:.3f}, brake={brake:.3f}")

		if end_prob > 0.75:
			self.visual_feature_buffer = []
			self.curr_notice[ego_id] = ''
			self.curr_notice_frame_id[ego_id] = -1

		if brake < 0.05:
			brake = 0.0
		if brake > 0.1:
			throttle = 0.0

		control = carla.VehicleControl()
		control.steer = float(steer) * 0.8
		control.throttle = float(throttle)
		control.brake = float(brake)
		print(f"[DEBUG LMDRIVE] ego={ego_id}, step={self.step[ego_id]}, control: steer={control.steer:.3f}, throttle={control.throttle:.3f}, brake={control.brake:.3f}")

		end_time = time.time()
		print(end_time-start_time)
		
		self.next_action_step[ego_id] = self.step[ego_id] + int((end_time - start_time) * 20)

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
		display_data['time'] = 'Time: %.3f. Frames: %d. End prob: %.2f' % (timestamp, len(self.visual_feature_buffer), end_prob)  
		display_data['meta_control'] = 'Throttle: %.2f. Steer: %.2f. Brake: %.2f' %(
			control.steer, control.throttle, control.brake
		)
		display_data['waypoints'] = 'Waypoints: (%.1f, %.1f), (%.1f, %.1f)' % (waypoints[0,0], -waypoints[0,1], waypoints[1,0], -waypoints[1,1])
		display_data['notice'] = "Notice: %s" % last_notice
		surface = self._hic.run_interface(display_data)
		tick_data['surface'] = surface
		self.prev_control[ego_id] = control
		if SAVE_PATH is not None:
			self.save(ego_id, tick_data)
		# print(self.debug_dict)
		return control

	def save(self, ego_id, tick_data):
		frame = (self.step[ego_id] - 20)
		Image.fromarray(tick_data["surface"]).save(
			self.save_path / f"meta_{ego_id}" / ("%04d.jpg" % frame)
		)
		return

	def destroy(self):
		del self.net

	def control_pid(self, waypoints, velocity):
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
		speed = velocity

		desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0
		brake = desired_speed < self.lm_config.brake_speed or (speed / desired_speed) > self.lm_config.brake_ratio

		aim = (waypoints[1] + waypoints[0]) / 2.0
		angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
		if(speed < 0.01):
			angle = np.array(0.0) # When we don't move we don't want the angle error to accumulate in the integral
		steer = self.turn_controller.step(angle)
		steer = np.clip(steer, -1.0, 1.0)

		delta = np.clip(desired_speed - speed, 0.0, self.lm_config.clip_delta)
		throttle = self.speed_controller.step(delta)
		throttle = np.clip(throttle, 0.0, self.lm_config.max_throttle)
		throttle = throttle if not brake else 0.0

		metadata = {
			'speed': float(speed.astype(np.float64)),
			'steer': float(steer),
			'throttle': float(throttle),
			'brake': float(brake),
			'wp_2': tuple(waypoints[1].astype(np.float64)),
			'wp_1': tuple(waypoints[0].astype(np.float64)),
			'desired_speed': float(desired_speed.astype(np.float64)),
			'angle': float(angle.astype(np.float64)),
			'aim': tuple(aim.astype(np.float64)),
			'delta': float(delta.astype(np.float64)),
		}

		return steer, throttle, brake, metadata
