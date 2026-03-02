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
import torch
import carla
import numpy as np
from PIL import Image
from torchvision import transforms as T
from team_code.pid_controller_b2d import PIDController
from team_code.planner_b2d import RoutePlanner
from leaderboard.autoagents import autonomous_agent
# import sys
# sys.path.insert(0, '/GPFS/rhome/zijunwang/WorkSpace/Bench2DriveZoo')
from mmcv import Config
from mmcv.models import build_model
from mmcv.utils import (get_dist_info, init_dist, load_checkpoint,wrap_fp16_model)
from mmcv.datasets.pipelines import Compose
from mmcv.parallel.collate import collate as  mm_collate_to_batch_form
from mmcv.core.bbox import get_box_type
from pyquaternion import Quaternion
from scipy.optimize import fsolve
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.utils.route_manipulation import _get_latlon_ref

SAVE_PATH = os.environ.get('SAVE_PATH', None)
IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)

def get_entry_point():
	return 'UniadAgent'

class UniadAgent(autonomous_agent.AutonomousAgent):
	def setup(self, path_to_conf_file, ego_vehicles_num=1):
		self.ego_vehicles_num = ego_vehicles_num
		self.track = autonomous_agent.Track.SENSORS
		self.steer_step = 0
		self.last_moving_status = 0
		self.last_moving_step = -1
		self.last_steers = deque()
		# Per-ego PID controllers to avoid cross-vehicle state sharing
		self.pidcontrollers = [PIDController() for _ in range(ego_vehicles_num)]
		self.config_path = 'Bench2DriveZoo/adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py'
		self.ckpt_path = 'ckpt/UniAD/uniad_base_b2d.pth'
		now = datetime.datetime.now()
		if IS_BENCH2DRIVE:
			self.save_name = path_to_conf_file.split('+')[-1] if '+' in path_to_conf_file else 'uniad_b2d'
		else:
			self.save_name = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
		self.step = -1
		self.wall_start = time.time()
		self.initialized = False
		# import ipdb; ipdb.set_trace()
		cfg = Config.fromfile(self.config_path)
		cfg.model['motion_head']['anchor_info_path'] = os.path.join('Bench2DriveZoo',cfg.model['motion_head']['anchor_info_path'])
		if hasattr(cfg, 'plugin'):
			if cfg.plugin:
				import importlib
				if hasattr(cfg, 'plugin_dir'):
					plugin_dir = cfg.plugin_dir
					plugin_dir = os.path.join("Bench2DriveZoo", plugin_dir)
					_module_dir = os.path.dirname(plugin_dir)
					_module_dir = _module_dir.split('/')
					_module_path = _module_dir[0]
					for m in _module_dir[1:]:
						_module_path = _module_path + '.' + m
					print(_module_path)
					plg_lib = importlib.import_module(_module_path)  
  
		self.model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
		checkpoint = load_checkpoint(self.model, self.ckpt_path, map_location='cpu', strict=True)
		self.model.cuda()
		self.model.eval()
		self.inference_only_pipeline = []
		for inference_only_pipeline in cfg.inference_only_pipeline:
			if inference_only_pipeline["type"] not in ['LoadMultiViewImageFromFilesInCeph']:
				self.inference_only_pipeline.append(inference_only_pipeline)
		self.inference_only_pipeline = Compose(self.inference_only_pipeline)
		self.takeover = False
		self.stop_time = 0
		self.takeover_time = 0
		self.save_path = None
		self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
		# Debug logging controls; set DRIVE_DEBUG=0 to disable or DRIVE_DEBUG_EVERY to change frequency
		self.debug_log = os.environ.get("DRIVE_DEBUG", "1") != "0"
		self.log_every = int(os.environ.get("DRIVE_DEBUG_EVERY", "1"))
		self.last_steers = deque()
		self.lat_ref, self.lon_ref = 42.0, 2.0
		# Independent previous-control state per ego to avoid aliasing
		base_control = carla.VehicleControl()
		base_control.steer = 0.0
		base_control.throttle = 0.0
		base_control.brake = 0.0	
		self.prev_control = [carla.VehicleControl(steer=base_control.steer,
												throttle=base_control.throttle,
												brake=base_control.brake)
								for _ in range(ego_vehicles_num)]
		if SAVE_PATH is not None:
			now = datetime.datetime.now()
			# string = pathlib.Path(os.environ['ROUTES']).stem + '_'
			string = self.save_name
			self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
			self.save_path.mkdir(parents=True, exist_ok=False)
			for ego_id in range(self.ego_vehicles_num):
				(self.save_path / 'rgb_front_{}'.format(ego_id)).mkdir()
				(self.save_path / 'rgb_front_right_{}'.format(ego_id)).mkdir()
				(self.save_path / 'rgb_front_left_{}'.format(ego_id)).mkdir()
				(self.save_path / 'rgb_back_{}'.format(ego_id)).mkdir()
				(self.save_path / 'rgb_back_right_{}'.format(ego_id)).mkdir()
				(self.save_path / 'rgb_back_left_{}'.format(ego_id)).mkdir()
				(self.save_path / 'meta_{}'.format(ego_id)).mkdir()
				(self.save_path / 'bev_{}'.format(ego_id)).mkdir()
   
		# write extrinsics directly
		self.lidar2img = {
		'CAM_FRONT':np.array([[ 1.14251841e+03,  8.00000000e+02,  0.00000000e+00, -9.52000000e+02],
								  [ 0.00000000e+00,  4.50000000e+02, -1.14251841e+03, -8.09704417e+02],
								  [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -1.19000000e+00],
								 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
		  'CAM_FRONT_LEFT':np.array([[ 6.03961325e-14,  1.39475744e+03,  0.00000000e+00, -9.20539908e+02],
								   [-3.68618420e+02,  2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
								   [-8.19152044e-01,  5.73576436e-01,  0.00000000e+00, -8.29094072e-01],
								   [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
		  'CAM_FRONT_RIGHT':np.array([[ 1.31064327e+03, -4.77035138e+02,  0.00000000e+00,-4.06010608e+02],
									   [ 3.68618420e+02,  2.58109396e+02, -1.14251841e+03,-6.47296750e+02],
									[ 8.19152044e-01,  5.73576436e-01,  0.00000000e+00,-8.29094072e-01],
									[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]),
		'CAM_BACK':np.array([[-1.00000000e+00, -1.22464680e-16,  0.00000000e+00, -1.97168135e-16],
							 [ 0.00000000e+00,  0.00000000e+00, -1.00000000e+00, -2.40000000e-01],
							 [ 1.22464680e-16, -1.00000000e+00,  0.00000000e+00, -1.61000000e+00],
							 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
		'CAM_BACK_LEFT':np.array([[-1.14251841e+03,  8.00000000e+02,  0.00000000e+00, -6.84385123e+02],
								  [-4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
								  [-9.39692621e-01, -3.42020143e-01,  0.00000000e+00, -4.92889531e-01],
								  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
  
		'CAM_BACK_RIGHT': np.array([[ 3.60989788e+02, -1.34723223e+03,  0.00000000e+00, -1.04238127e+02],
									[ 4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
									[ 9.39692621e-01, -3.42020143e-01,  0.00000000e+00, -4.92889531e-01],
									[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
		}
		self.lidar2cam = {
		'CAM_FRONT':np.array([[ 1.  ,  0.  ,  0.  ,  0.  ],
								 [ 0.  ,  0.  , -1.  , -0.24],
								 [ 0.  ,  1.  ,  0.  , -1.19],
							  [ 0.  ,  0.  ,  0.  ,  1.  ]]),
		'CAM_FRONT_LEFT':np.array([[ 0.57357644,  0.81915204,  0.  , -0.22517331],
									  [ 0.        ,  0.        , -1.  , -0.24      ],
								   [-0.81915204,  0.57357644,  0.  , -0.82909407],
								   [ 0.        ,  0.        ,  0.  ,  1.        ]]),
		  'CAM_FRONT_RIGHT':np.array([[ 0.57357644, -0.81915204, 0.  ,  0.22517331],
								   [ 0.        ,  0.        , -1.  , -0.24      ],
								   [ 0.81915204,  0.57357644,  0.  , -0.82909407],
								   [ 0.        ,  0.        ,  0.  ,  1.        ]]),
		'CAM_BACK':np.array([[-1. ,  0.,  0.,  0.  ],
							 [ 0. ,  0., -1., -0.24],
							 [ 0. , -1.,  0., -1.61],
							 [ 0. ,  0.,  0.,  1.  ]]),
	 
		'CAM_BACK_LEFT':np.array([[-0.34202014,  0.93969262,  0.  , -0.25388956],
								  [ 0.        ,  0.        , -1.  , -0.24      ],
								  [-0.93969262, -0.34202014,  0.  , -0.49288953],
								  [ 0.        ,  0.        ,  0.  ,  1.        ]]),
  
		'CAM_BACK_RIGHT':np.array([[-0.34202014, -0.93969262,  0.  ,  0.25388956],
								  [ 0.        ,  0.         , -1.  , -0.24      ],
								  [ 0.93969262, -0.34202014 ,  0.  , -0.49288953],
								  [ 0.        ,  0.         ,  0.  ,  1.        ]])
		}
		self.lidar2ego = np.array([[ 0. ,  1. ,  0. , -0.39],
								   [-1. ,  0. ,  0. ,  0.  ],
								   [ 0. ,  0. ,  1. ,  1.84],
								   [ 0. ,  0. ,  0. ,  1.  ]])
		
		topdown_extrinsics =  np.array([[0.0, -0.0, -1.0, 50.0], [0.0, 1.0, -0.0, 0.0], [1.0, -0.0, 0.0, -0.0], [0.0, 0.0, 0.0, 1.0]])
		unreal2cam = np.array([[0,1,0,0], [0,0,-1,0], [1,0,0,0], [0,0,0,1]])
		self.coor2topdown = unreal2cam @ topdown_extrinsics
		topdown_intrinsics = np.array([[548.993771650447, 0.0, 256.0, 0], [0.0, 548.993771650447, 256.0, 0], [0.0, 0.0, 1.0, 0], [0, 0, 0, 1.0]])
		self.coor2topdown = topdown_intrinsics @ self.coor2topdown

	def _init(self):
		# Inspect route inputs to understand shapes/types
		if self.debug_log:
			try:
				gpwc0 = self._global_plan_world_coord[0][0]
				gpgps0 = self._global_plan[0][0]
				self._log(f"[INIT_DEBUG] _global_plan_world_coord[0][0] type={type(gpwc0)} content={gpwc0}")
				self._log(f"[INIT_DEBUG] _global_plan[0][0] type={type(gpgps0)} content={gpgps0}")
			except Exception as e:
				self._log(f"[INIT_DEBUG] failed to inspect plan entries: {e}")
		try:
			# Prefer the map's geo-reference to avoid tuple/format issues
			self.lat_ref, self.lon_ref = _get_latlon_ref(CarlaDataProvider.get_world())
		except Exception as e:
			print(e, flush=True)
			self.lat_ref, self.lon_ref = 0, 0        
		self._route_planners = []
		for ego_id in range(self.ego_vehicles_num):
			route_planner = RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
			route_planner.set_route([self._global_plan[ego_id]], True, [self._global_plan_world_coord[ego_id]])
			self._route_planners.append(route_planner)
		self.next_action_step = [0] * self.ego_vehicles_num
		self.initialized = True
		self._log(f"[INIT] lat_ref={self.lat_ref:.6f}, lon_ref={self.lon_ref:.6f}, route_len={len(self._global_plan[0]) if self._global_plan else 0}")
  
  

	def sensors(self):
		sensors =[
				# camera rgb
				{
					'type': 'sensor.camera.rgb',
					'x': 0.80, 'y': 0.0, 'z': 1.60,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'width': 1600, 'height': 900, 'fov': 70,
					'id': 'CAM_FRONT'
				},
				{
					'type': 'sensor.camera.rgb',
					'x': 0.27, 'y': -0.55, 'z': 1.60,
					'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
					'width': 1600, 'height': 900, 'fov': 70,
					'id': 'CAM_FRONT_LEFT'
				},
				{
					'type': 'sensor.camera.rgb',
					'x': 0.27, 'y': 0.55, 'z': 1.60,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
					'width': 1600, 'height': 900, 'fov': 70,
					'id': 'CAM_FRONT_RIGHT'
				},
				{
					'type': 'sensor.camera.rgb',
					'x': -2.0, 'y': 0.0, 'z': 1.60,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
					'width': 1600, 'height': 900, 'fov': 110,
					'id': 'CAM_BACK'
				},
				{
					'type': 'sensor.camera.rgb',
					'x': -0.32, 'y': -0.55, 'z': 1.60,
					'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
					'width': 1600, 'height': 900, 'fov': 70,
					'id': 'CAM_BACK_LEFT'
				},
				{
					'type': 'sensor.camera.rgb',
					'x': -0.32, 'y': 0.55, 'z': 1.60,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
					'width': 1600, 'height': 900, 'fov': 70,
					'id': 'CAM_BACK_RIGHT'
				},
				# imu
				{
					'type': 'sensor.other.imu',
					'x': -1.4, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.05,
					'id': 'IMU'
				},
				# gps
				{
					'type': 'sensor.other.gnss',
					'x': -1.4, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.01,
					'id': 'GPS'
				},
				# speed
				{
					'type': 'sensor.speedometer',
					'reading_frequency': 20,
					'id': 'SPEED'
				},
    			{	
					'type': 'sensor.camera.rgb',
					'x': 0.0, 'y': 0.0, 'z': 50.0,
					'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
					'width': 512, 'height': 512, 'fov': 5 * 10.0,
					'id': 'bev'
				}
				
			]
		
		if IS_BENCH2DRIVE:
			sensors += [
					{	
						'type': 'sensor.camera.rgb',
						'x': 0.0, 'y': 0.0, 'z': 50.0,
						'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
						'width': 512, 'height': 512, 'fov': 5 * 10.0,
						'id': 'bev'
					}]
		return sensors

	def tick(self, ego_id, input_data):
		# import ipdb; ipdb.set_trace()
		# self.step += 1
		encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
		imgs = {}
		for cam in [f'CAM_FRONT_{ego_id}',f'CAM_FRONT_LEFT_{ego_id}',f'CAM_FRONT_RIGHT_{ego_id}',f'CAM_BACK_{ego_id}',f'CAM_BACK_LEFT_{ego_id}',f'CAM_BACK_RIGHT_{ego_id}']:
			img = cv2.cvtColor(input_data[cam][1][:, :, :3], cv2.COLOR_BGR2RGB)
			_, img = cv2.imencode('.jpg', img, encode_param)
			img = cv2.imdecode(img, cv2.IMREAD_COLOR)
			imgs[cam] = img
		bev = cv2.cvtColor(input_data[f'bev_{ego_id}'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		gps = input_data[f'GPS_{ego_id}'][1][:2]
		speed = input_data[f'SPEED_{ego_id}'][1]['speed']
		compass = input_data[f'IMU_{ego_id}'][1][-1]
		acceleration = input_data[f'IMU_{ego_id}'][1][:3]
		angular_velocity = input_data[f'IMU_{ego_id}'][1][3:6]
		pos = self.gps_to_location(gps)
		route_out = self._route_planners[ego_id].run_step(pos, 0)
		# Allow for (pos, cmd, pos_world) tuples
		near_node, near_command = route_out[0], route_out[1]
		near_world = route_out[2] if len(route_out) > 2 else None

		if self.debug_log and (self.step % self.log_every == 0):
			ego_actor = CarlaDataProvider.get_hero_actor(hero_id=ego_id)
			ego_tf = ego_actor.get_transform() if ego_actor else None
			ego_loc = ego_tf.location if ego_tf else None
			ego_yaw = ego_tf.rotation.yaw if ego_tf else None
			world_target = (near_world.location.x, near_world.location.y) if near_world else None
			ego_x = getattr(ego_loc, "x", None)
			ego_y = getattr(ego_loc, "y", None)
			self._log(
				f"[TICK_WORLD][ego={ego_id}] gps_latlon={gps.tolist()} pos_local={pos.tolist()} "
				f"lat_ref={self.lat_ref:.6f} lon_ref={self.lon_ref:.6f} "
				f"ego_world=({ego_x},{ego_y}, yaw={ego_yaw}) "
				f"near_node_local=({float(near_node[0]):.3f},{float(near_node[1]):.3f}) "
				f"near_node_world={world_target}"
			)
		if self.debug_log and (self.step % self.log_every == 0):
			self._log(f"[TICK][ego={ego_id}] gps={gps.tolist()} pos={pos.tolist()} lat_ref={self.lat_ref:.6f} lon_ref={self.lon_ref:.6f} "
					  f"speed={speed:.3f} compass={compass:.3f} near_node={(float(near_node[0]), float(near_node[1]))} near_cmd={near_command}")
		if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
			compass = 0.0
			acceleration = np.zeros(3)
			angular_velocity = np.zeros(3)

		result = {
				'imgs': imgs,
				'gps': gps,
				'pos':pos,
				'speed': speed,
				'compass': compass,
				'bev': bev,
				'acceleration':acceleration,
				'angular_velocity':angular_velocity,
				'command_near':near_command,
				'command_near_xy':near_node
	
				}
		
		return result
	
	@torch.no_grad()
	def run_step(self, input_data, timestamp):
		control_all = []
		self.step += 1
		#print(self.step)
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
		tick_data = self.tick(ego_id, input_data)
		
		# Frame skipping logic based on environment variable
		realtime_mode = os.environ.get('REALTIME_MODE', '0')
		if realtime_mode == '1':
			# REAL-TIME simulation
			if self.step < self.next_action_step[ego_id] and self.step:
				return self.prev_control[ego_id]
		else:
			# NON-REAL-TIME simulation
			if self.step % 4 != 1 and self.step:
				return self.prev_control[ego_id]

		
		results = {}
		results['lidar2img'] = []
		results['lidar2cam'] = []
		results['img'] = []
		results['folder'] = ' '
		results['scene_token'] = ' '  
		results['frame_idx'] = 0
		results['timestamp'] = self.step / 20
		results['box_type_3d'], _ = get_box_type('LiDAR')

		start_time = time.time()
		# for cam in [f'CAM_FRONT_{ego_id}',f'CAM_FRONT_LEFT_{ego_id}',f'CAM_FRONT_RIGHT_{ego_id}',f'CAM_BACK_{ego_id}',f'CAM_BACK_LEFT_{ego_id}',f'CAM_BACK_RIGHT_{ego_id}']:
		for cam in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']:
			results['lidar2img'].append(self.lidar2img[cam])
			results['lidar2cam'].append(self.lidar2cam[cam])
			results['img'].append(tick_data['imgs'][cam+f"_{ego_id}"])
		results['lidar2img'] = np.stack(results['lidar2img'],axis=0)
		results['lidar2cam'] = np.stack(results['lidar2cam'],axis=0)
  
		raw_theta = tick_data['compass']   if not np.isnan(tick_data['compass']) else 0
		ego_theta = -raw_theta + np.pi/2
		rotation = list(Quaternion(axis=[0, 0, 1], radians=ego_theta))
  
		can_bus = np.zeros(18)
		can_bus[0] = tick_data['pos'][0]
		can_bus[1] = -tick_data['pos'][1]
		can_bus[3:7] = rotation
		can_bus[7] = tick_data['speed']
		can_bus[10:13] = tick_data['acceleration']
		can_bus[11] *= -1
		can_bus[13:16] = -tick_data['angular_velocity']
		can_bus[16] = ego_theta
		can_bus[17] = ego_theta / np.pi * 180 
		results['can_bus'] = can_bus
		command = tick_data['command_near'].value
		if command < 0:
			command = 4
		command -= 1
		results['command'] = command

		
		theta_to_lidar = raw_theta
		command_near_xy = np.array([tick_data['command_near_xy'][0]-can_bus[0],-tick_data['command_near_xy'][1]-can_bus[1]])
		rotation_matrix = np.array([[np.cos(theta_to_lidar),-np.sin(theta_to_lidar)],[np.sin(theta_to_lidar),np.cos(theta_to_lidar)]])
		local_command_xy = rotation_matrix @ command_near_xy
  
		ego2world = np.eye(4)
		ego2world[0:3,0:3] = Quaternion(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
		# ego2world[0:3,0:2] = can_bus[0:2]
		ego2world[0:2,3] = can_bus[0:2]
		lidar2global = ego2world @ self.lidar2ego
		results['l2g_r_mat'] = lidar2global[0:3,0:3]
		results['l2g_t'] = lidar2global[0:3,3]
		stacked_imgs = np.stack(results['img'],axis=-1)
		results['img_shape'] = stacked_imgs.shape
		results['ori_shape'] = stacked_imgs.shape
		results['pad_shape'] = stacked_imgs.shape
		results = self.inference_only_pipeline(results)
		self.device="cuda"
		input_data_batch = mm_collate_to_batch_form([results], samples_per_gpu=1)
		for key, data in input_data_batch.items():
			if key != 'img_metas':
				if torch.is_tensor(data[0]):
					data[0] = data[0].to(self.device)
		inference_start = time.time()
		output_data_batch = self.model(input_data_batch, return_loss=False, rescale=True)
		inference_end = time.time()
		print('inference time:', inference_end - inference_start)
		out_truck =  output_data_batch[0]['planning']['result_planning']['sdc_traj'][0].cpu().numpy()
		controller = self.pidcontrollers[ego_id]
		steer_traj, throttle_traj, brake_traj, metadata_traj = controller.control_pid(out_truck, tick_data['speed'], local_command_xy)
		
		end_time = time.time()
		print('time:', end_time - start_time)
		
		if brake_traj < 0.05: brake_traj = 0.0
		if throttle_traj > brake_traj: brake_traj = 0.0
		if tick_data['speed']>5:
			throttle_traj = 0
		control = carla.VehicleControl()
		self.pid_metadata = metadata_traj
		self.pid_metadata['agent'] = 'only_traj'
		control.steer = np.clip(float(steer_traj), -1, 1)
		control.throttle = np.clip(float(throttle_traj), 0, 0.75)
		control.brake = np.clip(float(brake_traj), 0, 1)
		self.pid_metadata['steer'] = control.steer
		self.pid_metadata['throttle'] = control.throttle
		self.pid_metadata['brake'] = control.brake
		self.pid_metadata['steer_traj'] = float(steer_traj)
		self.pid_metadata['throttle_traj'] = float(throttle_traj)
		self.pid_metadata['brake_traj'] = float(brake_traj)
		self.pid_metadata['plan'] = out_truck.tolist()
		if self.debug_log and (self.step % self.log_every == 0):
			self._log(f"[CTRL][ego={ego_id}] can_pos=({can_bus[0]:.2f},{can_bus[1]:.2f}) "
					  f"theta={ego_theta:.3f} speed={tick_data['speed']:.3f} "
					  f"target_local=({local_command_xy[0]:.2f},{local_command_xy[1]:.2f}) "
					  f"wp0=({out_truck[0][0]:.2f},{out_truck[0][1]:.2f}) steer={control.steer:.3f} "
					  f"throttle={control.throttle:.3f} brake={control.brake:.3f}")
		self.next_action_step[ego_id] = self.step + int((end_time - start_time) * 20)
		print(self.next_action_step)

		if SAVE_PATH is not None and self.step % 5 == 0:
			self.save(ego_id, tick_data)
		self.prev_control[ego_id] = control
		return control

	def save(self, ego_id, tick_data):
		frame = self.step // 10
		Image.fromarray(tick_data['imgs'][f'CAM_FRONT_{ego_id}']).save(self.save_path / 'rgb_front_{}'.format(ego_id) / ('%04d.png' % frame))
		Image.fromarray(tick_data['imgs'][f'CAM_FRONT_LEFT_{ego_id}']).save(self.save_path / 'rgb_front_left_{}'.format(ego_id) / ('%04d.png' % frame))
		Image.fromarray(tick_data['imgs'][f'CAM_FRONT_RIGHT_{ego_id}']).save(self.save_path / 'rgb_front_right_{}'.format(ego_id) / ('%04d.png' % frame))
		Image.fromarray(tick_data['imgs'][f'CAM_BACK_{ego_id}']).save(self.save_path / 'rgb_back_{}'.format(ego_id) / ('%04d.png' % frame))
		Image.fromarray(tick_data['imgs'][f'CAM_BACK_LEFT_{ego_id}']).save(self.save_path / 'rgb_back_left_{}'.format(ego_id) / ('%04d.png' % frame))
		Image.fromarray(tick_data['imgs'][f'CAM_BACK_RIGHT_{ego_id}']).save(self.save_path / 'rgb_back_right_{}'.format(ego_id) / ('%04d.png' % frame))
		Image.fromarray(tick_data['bev']).save(self.save_path / 'bev_{}'.format(ego_id) / ('%04d.png' % frame))
		outfile = open(self.save_path / 'meta_{}'.format(ego_id) / ('%04d.json' % frame), 'w')
		json.dump(self.pid_metadata, outfile, indent=4)
		outfile.close()

	def destroy(self):
		del self.model
		torch.cuda.empty_cache()

	def gps_to_location(self, gps):
		EARTH_RADIUS_EQUA = 6378137.0
		# gps content: numpy array: [lat, lon, alt]
		lat, lon = gps
		scale = math.cos(self.lat_ref * math.pi / 180.0)
		my = math.log(math.tan((lat+90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
		mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
		y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
		x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
		return np.array([x, y])

	def _log(self, msg):
		"""Lightweight debug logger."""
		print(msg, flush=True)
