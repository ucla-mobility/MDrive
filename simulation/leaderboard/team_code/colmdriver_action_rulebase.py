import os
import copy
import re
import io
import json
import numpy as np
import torch
import carla
import cv2
import math
import datetime
import pathlib
from PIL import Image
from skimage.measure import block_reduce
import time
from typing import OrderedDict
import shapely
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pygame

from openai import OpenAI
import base64

from agents.navigation.local_planner import RoadOption

from team_code.v2x_controller import V2X_Controller
from team_code.render_v2x import render_self_car
from team_code.v2x_utils import get_yaw_angle, boxes_to_corners_3d, get_points_in_rotated_box_3d

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from opencood.tools import train_utils
from opencood.tools import train_utils, inference_utils
from opencood.visualization import vis_utils, my_vis, simple_vis_multiclass

####### Input: raw_data, N(actor)+M(RSU)
####### Output: actors action, N(actor)
####### Generate the action with the trained model.

SAVE_PATH = os.environ.get("SAVE_PATH", 'eval')
os.environ["SDL_VIDEODRIVER"] = "dummy"

commands_dict = {"-1": "void",
				 "1": "turn left",
				 "2": "turn right",
				 "3": "straight",
				 "4": "lane follow",
				 "5": "lane change left",
				 "6": "lane change right"}

def read_txt_file(file_path):
	with open(file_path, 'r', encoding='utf-8') as file:
		content = file.read()
	return content

# Function to encode the image
def encode_image(image_path):
	with open(image_path, "rb") as image_file:
		return base64.b64encode(image_file.read()).decode('utf-8')

def _numpy(carla_vector, normalize=False):
	result = np.float32([carla_vector.x, carla_vector.y])
	if normalize:
		return result / (np.linalg.norm(result) + 1e-4)
	return result

def _location(x, y, z):
	return carla.Location(x=float(x), y=float(y), z=float(z))

def _orientation(yaw):
	return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])

def get_collision(p1, v1, p2, v2):
	A = np.stack([v1, -v2], 1)
	b = p2 - p1
	if abs(np.linalg.det(A)) < 1e-3:
		return False, None
	x = np.linalg.solve(A, b)
	collides = all(x >= 0) and all(x <= 1)  # how many seconds until collision
	return collides, p1 + x[0] * v1

class DisplayInterface(object):
	def __init__(self):
		self._width = 6000
		self._height = 1500
		self._surface = None
		pygame.init()
		pygame.font.init()
		self._clock = pygame.time.Clock()
		self._display = pygame.display.set_mode(
			(self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF
		)
		pygame.display.set_caption("V2X Agent")
	def run_interface(self, input_data):
		rgb = input_data['rgb']
		map = input_data['map']
		lidar = input_data['lidar']
		surface = np.zeros((1500, 6750, 3),np.uint8)
		surface[:, :3000] = rgb
		surface[:, 3000:3000+2250] = input_data['lidar_3d']
		surface[:, 10500//2:] = input_data['lidar_bev']
		surface = cv2.putText(surface, input_data['comm'], (10*5,290*5), cv2.FONT_HERSHEY_SIMPLEX,0.5*3,(0,0,255), 1*3)
		surface = cv2.putText(surface, input_data['control'], (10*5,280*5), cv2.FONT_HERSHEY_SIMPLEX,0.5*3,(0,0,255), 1*3)
		surface = cv2.putText(surface, input_data['meta_infos'][1], (10*5,270*5), cv2.FONT_HERSHEY_SIMPLEX,0.5*3,(0,0,255), 1*3)
		surface = cv2.putText(surface, input_data['meta_infos'][2], (10*5,260*5), cv2.FONT_HERSHEY_SIMPLEX,0.5*3,(0,0,255), 1*3)
		surface = cv2.putText(surface, input_data['time'], (10*5,250*5), cv2.FONT_HERSHEY_SIMPLEX,0.5*3,(0,0,255), 1*3)
		surface = cv2.putText(surface, 'Front view', (2400,20*5), cv2.FONT_HERSHEY_SIMPLEX,0.5*3*2,(0,0,0), 8)
		surface = cv2.putText(surface, 'Lidar 3D', (2500+2250,20*5), cv2.FONT_HERSHEY_SIMPLEX,0.5*3*2,(255,255,255), 8)
		surface = cv2.putText(surface, 'Lidar BEV', (2500+2250+1500,20*5), cv2.FONT_HERSHEY_SIMPLEX,0.5*3*2,(255,255,255), 8)
		surface[:, 5990//2:6010//2] = 255
		surface[:, 10490//2:10510//2] = 255

		# display image
		self._surface = pygame.surfarray.make_surface(surface.swapaxes(0, 1))
		if self._surface is not None:
			self._display.blit(self._surface, (0, 0))

		pygame.display.flip()
		pygame.event.get()
		return surface

	def _quit(self):
		pygame.quit()


def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
	"""
	Build a rotation matrix and take the dot product.
	"""
	# z value to 1 for rotation
	xy1 = xyz.copy()
	xy1[:, 2] = 1

	c, s = np.cos(r1), np.sin(r1)
	r1_to_world = np.matrix([[c, -s, t1_x], [s, c, t1_y], [0, 0, 1]])

	# np.dot converts to a matrix, so we explicitly change it back to an array
	world = np.asarray(r1_to_world @ xy1.T)

	c, s = np.cos(r2), np.sin(r2)
	r2_to_world = np.matrix([[c, -s, t2_x], [s, c, t2_y], [0, 0, 1]])
	# world frame -> r2 frame
	world_to_r2 = np.linalg.inv(r2_to_world)

	out = np.asarray(world_to_r2 @ world).T
	# reset z-coordinate
	out[:, 2] = xyz[:, 2]

	return out

def turn_back_into_theta(input):
	output = torch.cat([input[:,:,:2],torch.atan2(input[:,:,2:3], input[:,:,-1:]),input[:,:,3:]],dim=2)
	assert output.shape[2] == input.shape[2]
	return output

def turn_traffic_into_map(all_bbox, det_range):
	data_total = []
	if len(all_bbox) == 0:
		all_bbox = np.zeros((1,4,2))

	fig = plt.figure(figsize=(6, 12), dpi=16)
	plt.gca().xaxis.set_major_locator(plt.NullLocator())
	plt.gca().yaxis.set_major_locator(plt.NullLocator())
	plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
	plt.margins(0,0)
	ax = plt.gca()
	ax.set_facecolor("black")

	plt.xlim((-det_range[2], det_range[3]))
	plt.ylim((-det_range[1], det_range[0]))

	for i in range(len(all_bbox)):
		plt.fill(all_bbox[i,:,0], all_bbox[i,:,1], color = 'white')

	# If we haven't already shown or saved the plot, then we need to draw the figure first...
	fig.canvas.draw()

	# Now we can save it to a numpy array.
	data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
	data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
	# H=192, W=96, 3
	data_total.append(data[:, :, 0])
	# plt.savefig('/GPFS/public/InterFuser/results/cop3/pnp/multiclass_finetune_fusion_none/test.png')
	plt.close()

	occ_map = np.stack(data_total, axis=0) # B * T_p, H, W
	return occ_map


def x_to_world(pose):
	"""
	The transformation matrix from x-coordinate system to carla world system
	Also is the pose in world coordinate: T_world_x

	Parameters
	----------
	pose : list
		[x, y, z, roll, yaw, pitch], degree
		[x, y, roll], radians
	Returns
	-------
	matrix : np.ndarray
		The transformation matrix.
	"""
	x, y, roll= pose[:]
	z = 0
	c_r = np.cos(roll)
	s_r = np.sin(roll)

	matrix = np.identity(4)

	# translation matrix
	matrix[0, 3] = x
	matrix[1, 3] = y
	matrix[2, 3] = z

	# rotation matrix
	matrix[0,0] = c_r
	matrix[0,1] = -s_r
	matrix[1,0] = s_r
	matrix[1,1] = c_r

	return matrix

def get_pairwise_transformation(pose, max_cav):
	"""
	Get pair-wise transformation matrix accross different agents.

	Parameters
	----------
	base_data_dict : dict
		Key : cav id, item: transformation matrix to ego, lidar points.

	max_cav : int
		The maximum number of cav, default 5

	Return
	------
	pairwise_t_matrix : np.array
		The pairwise transformation matrix across each cav.
		shape: (L, L, 4, 4), L is the max cav number in a scene
		pairwise_t_matrix[i, j] is Tji, i_to_j
	"""
	pairwise_t_matrix = np.tile(np.eye(4), (max_cav, max_cav, 1, 1)) # (L, L, 4, 4)
	t_list = []

	# save all transformation matrix in a list in order first.
	for i in range(max_cav):
		lidar_pose = pose[i]
		t_list.append(x_to_world(lidar_pose))  # Twx

	for i in range(len(t_list)):
		for j in range(len(t_list)):
			# identity matrix to self
			if i != j:
				# i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
				t_matrix = np.linalg.solve(t_list[j], t_list[i])  # Tjw*Twi = Tji
				pairwise_t_matrix[i, j] = t_matrix

	return pairwise_t_matrix

def warp_affine_simple(src, M, dsize, align_corners=False):
	B, C, H, W = src.size()
	grid = F.affine_grid(M,
						[B, C, dsize[0], dsize[1]],
						align_corners=align_corners).to(src)
	return F.grid_sample(src, grid, align_corners=align_corners)

def warp_image(det_pose, occ_map):
	'''
	det_pose: B, T_p, 3, torch.Tensor
	occ_map: B, T_p, C, H, W, torch.Tensor
	'''
	B, T, C, H, W = occ_map.shape
	occ_fused = []
	for b in range(B):
		pairwise_t_matrix = \
			get_pairwise_transformation(det_pose[b].cpu(), T)
		# t_matrix[i, j]-> from i to j
		pairwise_t_matrix = pairwise_t_matrix[:,:,[0, 1],:][:,:,:,[0, 1, 3]] # [N, N, 2, 3]
		pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
		pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
		pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (12)  #(downsample_rate * discrete_ratio * W) * 2
		pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (24)

		t_matrix = torch.from_numpy(pairwise_t_matrix[:T, :T, :, :])
		
		neighbor_feature = warp_affine_simple(occ_map[b], t_matrix[-1, :, :, :], (H, W))
		occ_fused.append(neighbor_feature)
	
	return torch.stack(occ_fused)

command_dict = {-1: "no command", 
				 1:'turn left at intersection',
				 2:'turn right at intersection',
				 3:'go straight at intersection',
				 4:'follow the lane',
				 5:'left lane change',
				 6:'right lane change'}

class VLM_Client:
	def __init__(self, url):
		self.client = OpenAI(
			api_key='EMPTY',
			base_url=url
		)
		self.model_name = self.client.models.list().data[0].id
	
	def edit_prompt(self, prompt, measurements, comm, perception, det_type):
		speed = measurements['speed']
		command = int(measurements['command'])
		if command in command_dict.keys():
			command_str = command_dict[command]
		else:
			command_str = 'no command'

		if comm != None:
			prompt = prompt.replace("No interaction now", comm['speed'])
			print('comm:',comm['nav'] + ', ' + comm['speed'])
		print('speed:',"{:.2f}".format(speed))
		
		perc = [[], [], []]
		for actor, actor_loc in perception.items():
			for loc in actor_loc:
				if loc[0] < -12 or loc[0] > 12 or loc[1] < -12 or loc[1] > 36:
					continue
				perc[actor].append([-loc[0], loc[1]])
				
		perception_str = ''
		if det_type=='no_det':
			print('no detection for VLM!')
			pass
		else:
			for k in range(3):
				veh_type = 'Car' if k==0 else 'Pedestrian' if k==1 else 'Bike'
				for i, p in enumerate(perc[k]):
					if p[0]>0: lr = 'left'
					else: lr = 'right'
					if p[1]>0: fb = 'forward'
					else: fb = 'backward'
					perception_str = perception_str + f'{veh_type} {i+1}, {abs(int(p[1]))} meters {fb} and {abs(int(p[0]))} meters to the {lr};\n'

		prompt_dic = {
			"nvins": command_str,
			"speed": "{:.2f}".format(speed),
			"pr": perception_str
		}
		prompt = prompt.format(**prompt_dic)

		return prompt

	def infer(self, front_img, system_prompt):
		pil_image = Image.fromarray(front_img)
		buffered = io.BytesIO()
		pil_image.save(buffered, format="JPEG")
		image_bytes = base64.b64encode(buffered.getvalue()).decode('utf-8')

		t = time.time()
		response = self.client.chat.completions.create(
			model=self.model_name,
			messages=[
				{
				"role": "user",
				"content": [
					{
					"type": "text",
					"text": system_prompt
					},
					{
					"type": "image_url",
					"image_url": {
						"url": f"data:image/jpeg;base64,{image_bytes}"
					}
					}
				]
				}
			],
			max_tokens=4,
			)
		content = response.choices[0].message.content
		return content, time.time()-t

class Comm_Client:
	def __init__(self, url, api_key='', api_base=''):
		if api_key!='':
			self.client = OpenAI(
				api_key=api_key,
				base_url=api_base
			)
			self.model_name = 'gpt-4o-mini'
		else:
			self.client = OpenAI(
				api_key='EMPTY',
				base_url=url
			)
			self.model_name = self.client.models.list().data[0].id
		print("Comm_Client", self.model_name)
		self.nav_int_list = ['follow the lane', 'left lane change', 'right lane change', 'go straight at intersection', 'turn left at intersection', 'turn right at intersection']
		self.speed_int_list = ['STOP', 'SLOWER', 'FASTER', 'KEEP']
		self.cons_anly = 'None'
	
	def send_message(self, prompt, max_tokens=256, return_time=False):
		t = time.time()
		while True:
			try:
				response = self.client.chat.completions.create(
					model=self.model_name,
					messages=[
						{
						"role": "user",
						"content": [
							{
							"type": "text",
							"text": prompt
							}
						]
						}
					],
					max_tokens=max_tokens,
					)
				break
			except Exception as e:
				print('error: ', e)
		resp = response.choices[0].message.content
		llm_infer_time = time.time()-t
		# print('time comsume: ', llm_infer_time, '\n')
		if return_time:
			return resp, llm_infer_time
		else:
			return resp

	def judge_direction(self, x0, y0, theta0, x1, y1, theta1):
		'''
		Given (x0, y0, theta0) and (x1, y1, theta1), output the relative pose from 0 to 1

		Parameters
		----------
		(x0, y0, theta0): pose of vehicle 0
		(x1, y1, theta1): pose of vehicle 1
	
		Returns
		-------
		direction, heading: the direction and heading of vehicle 0 in the perspective of vehicle 1
		'''		

		# cal relative position
		theta0 = self.normalize_angle(-theta0)
		theta1 = self.normalize_angle(-theta1)
		dx = x1 - x0
		dy = y0 - y1
		# trans to local coordinate system
		dx_local = dx * math.cos(theta1-math.pi) + dy * math.sin(theta1-math.pi)
		dy_local = -dx * math.sin(theta1-math.pi) + dy * math.cos(theta1-math.pi)
		
		# cal relative angle
		relative_angle = self.normalize_angle(math.atan2(dx_local, dy_local))
		if -math.pi / 8 <= relative_angle <= math.pi / 8:
			direction = 'front'
		elif math.pi / 8 < relative_angle <= 3 * math.pi / 8:
			direction = 'right front'
		elif 3 * math.pi / 8 < relative_angle <= 5 * math.pi / 8:
			direction = 'right'
		elif 5 * math.pi / 8 < relative_angle <= 7 * math.pi / 8:
			direction = 'right rear'
		elif 7 * math.pi / 8 < relative_angle <= math.pi:
			direction = 'rear'
		elif -math.pi <= relative_angle < -7 * math.pi / 8:
			direction = 'rear'
		elif -7 * math.pi / 8 <= relative_angle < -5 * math.pi / 8:
			direction = 'left rear'
		elif -5 * math.pi / 8 <= relative_angle < -3 * math.pi / 8:
			direction = 'left'
		elif -3 * math.pi / 8 <= relative_angle < - math.pi / 8:
			direction = 'left front'

		# cal heading
		angle_diff = self.normalize_angle(theta1 - theta0)
		if -math.pi / 4 <= angle_diff <= math.pi / 4:
			heading = 'forward'
		elif math.pi / 4 < angle_diff <= 3 * math.pi / 4:
			heading = 'right'
		elif -3 * math.pi / 4 <= angle_diff < -math.pi / 4:
			heading = 'left'
		else:
			heading = 'backward'

		return direction, heading

	@staticmethod
	def normalize_angle(angle):
		while angle > math.pi:
			angle -= 2 * math.pi
		while angle < -math.pi:
			angle += 2 * math.pi
		return angle

	def form_nego_order(self, group, car_data_raw, dir_inten_veh, speed_inten_veh, step):
		dis_metric = []
		comm_info_list = []
		for car_id in group:
			intention_dict = {}
			for i, dir_inten in enumerate(dir_inten_veh):
				intention = dir_inten

				if car_id==i:
					ego_intention = intention
				elif i in group:
					intention_dict[i] = intention
				else: pass

			dir_dict, heading_dict, dis_dict, speed_dict = {}, {}, {}, {}
			for i in group:
				if car_id==i: continue
				try:
					direction, heading = self.judge_direction(car_data_raw[i]['measurements']['x'], car_data_raw[i]['measurements']['y'], car_data_raw[i]['measurements']['theta'], car_data_raw[car_id]['measurements']['x'], car_data_raw[car_id]['measurements']['y'], car_data_raw[car_id]['measurements']['theta'])
				except Exception as e:
					print('\n\n\nerror in judge direction:', e)
					print(car_data_raw[i]['measurements'])
					print(car_data_raw, '\n\n\n')
				dir_dict[i] = direction
				heading_dict[i] = heading
				speed_dict[i] = car_data_raw[i]['measurements']['speed']
				dis_dict[i] = math.sqrt((car_data_raw[car_id]['measurements']['x'] - car_data_raw[i]['measurements']['x']) ** 2 + (car_data_raw[car_id]['measurements']['y'] - car_data_raw[i]['measurements']['y']) ** 2)

			comm_info_list.append({'ego_id': car_id, 'ego_speed': car_data_raw[car_id]['measurements']['speed'], 'ego_intention': ego_intention, 'other_direction': dir_dict, 'other_heading': heading_dict, 'other_distance': dis_dict, 'other_speed': speed_dict, 'other_intention': intention_dict})

			dis_metric.append([])
			for i in group:
				dis_metric[-1].append(math.sqrt((car_data_raw[car_id]['measurements']['x'] - car_data_raw[i]['measurements']['x']) ** 2 + (car_data_raw[car_id]['measurements']['y'] - car_data_raw[i]['measurements']['y']) ** 2))

		dis = []
		for row in dis_metric:
			dis.append(sum(row))
		dis_order = sorted(range(len(dis)), key=lambda k: dis[k])

		return comm_info_list, dis_order

	def comm(self, comm_info_list, car_id, local_comm_content, suggestion):
		for comm_dict in comm_info_list:
			if comm_dict['ego_id'] == car_id:
				info = comm_dict
				break

		veh_string = ''
		for i in range(len(info['other_direction'])):
			veh_string += f'  - Vehicle (ID: {list(info["other_direction"].keys())[i]}): {round(list(info["other_distance"].values())[i], 1)} meters at your {list(info["other_direction"].values())[i]}, heading {list(info["other_heading"].values())[i]}, at a speed of {round(list(info["other_speed"].values())[i])} m/s, and its driving intention is {list(info["other_intention"].values())[i]}.\n'

		previous_conv = ''
		for item in local_comm_content: # {'id': car_id, 'type': 'propose', 'message': proposal}
			previous_conv += f"- Vehicle {item['id']}: '{item['message']}'\n"
		if previous_conv=='':
			previous_conv = 'None'

		if suggestion=='':
			sug_str = ''
		else:
			sug_str = f"\n\n**Critic Feedback for Negotiation**\n{suggestion}"

		prompt = f'''**Role**
You are a driving assistant of a car. Given a scenario where multiple vehicles are in conflict, you need to negotiate with other vehicles to reach a consensus and ensure the safety and efficiency of all vehicles involved.

**Scenario**
- Ego Vehicle (ID: {info['ego_id']}): Intention = {info['ego_intention']}, Speed = {round(info['ego_speed'], 1)}m/s
- Surrounding Vehicles:
{veh_string}

**Traffic Rules**
0. In emergency situations, allow vehicles with special circumstances to pass through first.
1. Merging cars slow down to yield to straight car.
2. Left-turn cars slow down to yield to straight/right-turn car.
3. The car being yielded to go faster.
4. Cars behind decrease speed during emergency braking.

**Task**
Based on the information, analyze the situation considering the speed, direction, distance, and intention of each vehicle. Identify any potential conflicts and propose actions that ensure the safety and efficiency of all vehicles involved. You can either request others to yeild or you can yeild to others. Only output the message you want to negotiate with other cars as short as possible. Message may contain the action you will take and requests for others if not mentioned. If decide to yeild, you should stop; if others yeild to you, you should go faster.

**Conversation History**
{previous_conv}{sug_str}

**Only output the message!**
You are Vehicle {info['ego_id']}, the message you want to negotiate with other cars is:'''
		message, llm_infer_time = self.send_message(prompt, return_time=True)

		return message, llm_infer_time

	def sum_action(self, comm_content):
		prompt = '''**Task**
Given a conversation of multiple cars negotiating to reach consensus, classify each vehicle's speed change into [STOP, SLOWER, KEEP, FASTER] and output the result as a string in format: {'id': car_id, 'speed': category}.

**Classification rules**
- STOP: Come to a complete stop.
- SLOWER: Decrease speed.
- KEEP: Maintain current speed.
- FASTER: Increase speed.

Additional rules:
- If a car request others to yield, it should go faster
- If a car yield to others, it should stop

Output example:
{"0": {"speed": "STOP"}, "2":{"speed": "SLOWER"}}

Input proposal:
ppp
Your task is to analyze the given conversations for each vehicle and output the classification as a string in the specified format. DO NOT output other content other than the required actions. Ensure the output matches the required structure exactly.'''
		previous_conv = ''
		for item in comm_content: # {'id': car_id, 'type': 'propose', 'message': proposal}
			previous_conv += f"- Vehicle {item['id']}: '{item['message']}'\n"
		prompt = prompt.replace('ppp', previous_conv)
		message, sum_action_time = self.send_message(prompt, return_time=True)
		action_list = json.loads(message)
	
		return action_list, sum_action_time


class Nego_Client:
	def __init__(self, url, api_key, api_base):
		if api_key!='':
			self.client = OpenAI(
				api_key=api_key,
				base_url=api_base
			)
			self.model_name = 'gpt-4o-mini'
		else:
			self.client = OpenAI(
				api_key='EMPTY',
				base_url=url
			)
			self.model_name = self.client.models.list().data[0].id
		print("Nego_Client:", self.model_name)
		self.nego_content = {}
		self.nego_content['action'] = {}
		self.current_step = -1

	def send_message(self, prompt, max_tokens=256):
		t = time.time()
		while True:
			try:
				response = self.client.chat.completions.create(
					model=self.model_name,
					messages=[
						{
						"role": "user",
						"content": [
							{
							"type": "text",
							"text": prompt
							}
						]
						}
					],
					max_tokens=max_tokens,
					)
				break
			except Exception as e:
				print('error: ', e)
		resp = response.choices[0].message.content
		print(resp)
		# print('time comsume: ', time.time()-t, '\n')

		return resp

	def llm_cons_score(self, conversation):
		conv_str = ''
		for message in conversation:
			conv_str += f"Vehicle {message['id']}: '{message['message']}'\n"

		prompt = '''Task Description:
Please analyze the following conversation and determine whether the characters have reached a consensus in the given scenario. Your response should include two parts: the first part is a brief explanation of whether a consensus was reached; the second part is a score indicating the degree of consensus, ranging from 0 to 100, where 0 means no consensus at all, and 100 means complete consensus.

Scoring Criteria:
0-20: There are significant disagreements with almost no common ground.
21-40: While there are some disagreements, there are one or two points where both parties can accept each other's views.
41-60: There is a moderate level of compromise and understanding on most discussed topics, but important disagreements remain unresolved.
61-80: Consensus has been reached on most issues, with only minor differences of opinion on a few details.
81-100: Almost all issues have been agreed upon by all parties, with only negligible objections remaining.

Scenario: On the road, multiple cars may have driving conflicts now. They negotiate with each other to avoid conflict.
Conversation:
-conv-

Your output format:
Short analysis: very short sentence to sum the consensus situation of the conversation.
Consensus score: int'''
		prompt = prompt.replace('-conv-', conv_str)
		resp = self.send_message(prompt)

		pattern = r'Consensus score: (\d+)'
		match = re.search(pattern, resp)
		if match:
			score = float(match.group(1))/100
		else:
			score = 0
		return score

	def safety_score(self, waypoints1, waypoints2):
		min_distance = 1000
		min_point = 0
		num_points = len(waypoints1)
		for i in range(num_points):
			distance = np.sqrt((waypoints2[i][0] - waypoints1[i][0]) ** 2 + (waypoints2[i][1] - waypoints1[i][1]) ** 2)
			if distance < min_distance:
				min_distance = distance
				min_point = i
		print('min_distance:', min_distance, 'min_point:', min_point)
		# print('waypoints:', waypoints1[0], waypoints1[-1], waypoints2[0], waypoints2[-1])
		return min_distance, 1 / (1 + np.exp(-2 * (min_distance - 3))), min_point*4

	def safety_score_with_distance(self, waypoints1, waypoints2, distance1, distance2):
		min_distance = 1000
		min_point = 0
		num_points = len(waypoints1)
		for i in range(num_points):
			distance = np.sqrt((waypoints2[i][0] - waypoints1[i][0]) ** 2 + (waypoints2[i][1] - waypoints1[i][1]) ** 2)
			if distance < min_distance:
				min_distance = distance
				min_point = i
		this_distance1 = np.sqrt((waypoints1[min_point][0] - waypoints1[0][0]) ** 2 + (waypoints1[min_point][1] - waypoints1[0][1]) ** 2)
		distance1 = this_distance1 if this_distance1 < distance1 else distance1
		this_distance2 = np.sqrt((waypoints2[min_point][0] - waypoints2[0][0]) ** 2 + (waypoints2[min_point][1] - waypoints2[0][1]) ** 2)
		distance2 = this_distance2 if this_distance2 < distance2 else distance2
		print('min_distance:', min_distance, 'min_point:', min_point)
		# print('waypoints:', waypoints1[0], waypoints1[-1], waypoints2[0], waypoints2[-1])
		return min_distance, 1 / (1 + np.exp(-2 * (min_distance - 3))), min_point*4, distance1, distance2

	def efficiency_score(self, waypoints):
		average_speed = []
		for k in range(len(waypoints)):
			total_distance = 0
			for i in range(len(waypoints[k]) - 1):
				distance = np.sqrt((waypoints[k][i][0] - waypoints[k][i+1][0])**2 + (waypoints[k][i][1] - waypoints[k][i+1][1])**2)
				total_distance += distance
			average_speed.append(total_distance / ((len(waypoints[k])-1)*0.2))
		print('average_speed: ', [round(s,2) for s in average_speed])
		avg_speed = sum(average_speed) / len(average_speed)
		if avg_speed > 8:
			score = 1
		else:
			score = avg_speed / 8
		return score

	def evaluator(self, comm_results, local_comm_content, waypoints, main_path, step):
		'''
		comm_results: {car_id: {'nav': 'follow the lane', 'speed': 'KEEP'}, ...}
		conversation: {step: [[sender_id, receiver_id, message], ...], ...}
		'''
		cons_score = self.llm_cons_score(local_comm_content)
		efficiency_score = self.efficiency_score(waypoints)
		print('efficiency_score:', round(efficiency_score, 2))

		safety_score_list = []
		safety_id_list = []
		for i in range(len(waypoints)):
			for j in range(len(waypoints)):
				if i>=j: continue
				min_distance, safety_score, dur = self.safety_score(waypoints[i], waypoints[j])
				print('safety_score:', safety_score, ' min_distance:', min_distance)
				safety_score_list.append(safety_score)
				safety_id_list.append([i, j])

		total_score = round(3*cons_score + 5*min(safety_score_list) + 2*efficiency_score, 2)
		print('total_score:', total_score, '\n')

		# suggestion
		suggestion = ''
		for i in range(len(safety_score_list)):
			if safety_score_list[i] < 0.7:
				suggestion+=f'Safety issue: Vehicle {safety_id_list[i][0]} and vehicle {safety_id_list[i][1]} may have conflict. Suggest one of them to yeild, and another go pass faster.\n'
		if efficiency_score < 0.4:
			suggestion+='Efficiency issue: The efficiency is not high. Some of the cars can be faster.\n'
		print('suggestion:', suggestion)

		chat_ids = list(comm_results.keys())
		id_str = ''
		for i in chat_ids:
			id_str += str(i)
		self.nego_content[step] = {}
		self.nego_content[step][id_str] = {}
		self.nego_content[step][id_str]['cons_score'] = cons_score
		self.nego_content[step][id_str]['safety_score'] = safety_score
		self.nego_content[step][id_str]['efficiency_score'] = efficiency_score
		self.nego_content[step][id_str]['total_score'] = total_score
		self.nego_content[step][id_str]['content'] = local_comm_content
		self.nego_content[step][id_str]['suggestion'] = suggestion
		self.nego_content[step][id_str]['min_distance'] = min_distance

		self.plot_and_save_waypoints(waypoints, main_path / pathlib.Path(f'waypoints/{step}_{total_score}.png'))

		return total_score, suggestion, [cons_score, min(safety_score_list), efficiency_score]
	
	def plot_and_save_waypoints(self, pred_waypoints, save_path):		
		x_coords = []
		y_coords = []
		for waypoints in pred_waypoints:
			x_values = [point[0] for point in waypoints]
			y_values = [point[1] for point in waypoints]
			x_coords.append(x_values)
			y_coords.append(y_values)

		plt.figure(figsize=(10, 10))
		for i, (x, y) in enumerate(zip(x_coords, y_coords), start=1):
			plt.plot(x, y, marker='o', label=f'Path {i}')

		plt.title('Waypoints Trajectories')
		plt.xlabel('X-axis')
		plt.ylabel('Y-axis')

		all_x = [item for sublist in x_coords for item in sublist]
		all_y = [item for sublist in y_coords for item in sublist]
		x_min, x_max = min(all_x), max(all_x)
		y_min, y_max = min(all_y), max(all_y)
		
		max_range = max(x_max - x_min, y_max - y_min)
		x_center = (x_max + x_min) / 2
		y_center = (y_max + y_min) / 2
		
		plt.xlim(x_center - max_range * 1.1 / 2, x_center + max_range * 1.1 / 2)
		plt.ylim(y_center - max_range * 1.1 / 2, y_center + max_range * 1.1 / 2)

		plt.legend()
		plt.grid(True)
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		plt.savefig(save_path)
		plt.close()


class PnP_infer():
	def __init__(self, config=None, ego_vehicles_num=1, perception_model=None, planning_model=None, perception_dataloader=None, device=None) -> None:
		self.config = config
		self._hic = DisplayInterface()
		self.ego_vehicles_num = ego_vehicles_num

		self.det_range = [36, 12, 12, 12, 0.25]
		self.max_distance = 36
		self.distance_to_map_center = (self.det_range[0]+self.det_range[1])/2-self.det_range[1]

		self.perception_model = perception_model
		self.planning_model = planning_model
		self.perception_dataloader = perception_dataloader
		self.device=device

		# Optional perception backbone swap & BEV-feature ablation. Both flags
		# default OFF, so the regular `colmdriver_rulebase` planner is
		# bit-identical to before. Enabled via the variant YAML configs
		# (`colmdriver_rulebase_{attfuse,fcooper,disco,cobevt}.yaml`) — see
		# `perception.backbone` and `perception.zero_bev_feature` keys.
		# Mirrors the pattern in pnp_infer_action_e2e_v2v.py.
		_perc_cfg = (self.config or {}).get('perception', {}) or {}
		self._perception_backbone = str(_perc_cfg.get('backbone', 'codriving')).strip().lower()
		self._zero_bev_feature = bool(_perc_cfg.get('zero_bev_feature', False))
		self._heal_perception = None
		_VALID_BACKBONES = ('codriving', 'attfuse', 'fcooper', 'disco', 'cobevt')
		if self._perception_backbone not in _VALID_BACKBONES:
			raise ValueError(
				"perception.backbone={!r} not recognized; expected one of: {}".format(
					self._perception_backbone, ', '.join(_VALID_BACKBONES)))
		if self._perception_backbone != 'codriving':
			# Lazy import: only paid when a HEAL backbone is selected.
			from tools.perception_runner import PerceptionRunner
			self._heal_perception = PerceptionRunner(detector=self._perception_backbone)
			print('[PnP_infer:rulebase] HEAL perception backbone active: {}'.format(self._perception_backbone))
		if self._zero_bev_feature:
			print('[PnP_infer:rulebase] BEV feature path zeroed (perception.zero_bev_feature=true)')

		self.perception_memory_bank = [[{}] for _ in range(self.ego_vehicles_num)]

		self.controller = [V2X_Controller(self.config['control']) for _ in range(self.ego_vehicles_num)]

		self.input_lidar_size = 224
		self.lidar_range = [36, 36, 36, 36]

		self.softmax = torch.nn.Softmax(dim=0)
		self.traffic_meta_moving_avg = np.zeros((ego_vehicles_num, 400, 7))
		self.momentum = self.config['control']['momentum']
		self.prev_lidar = []
		self.prev_control = {}
		self.prev_surround_map = {}
		self.pre_raw_data_bank = {}

		############
		###### multi-agent related components
		############

		### generate the save files for images
		self.skip_frames = self.config['simulation']['skip_frames']
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
			print(string)
			self.save_path = pathlib.Path(SAVE_PATH) / string
			self.save_path.mkdir(parents=True, exist_ok=False)
			(self.save_path / "meta").mkdir(parents=True, exist_ok=False)
			self.log_path = self.save_path / pathlib.Path("log")
			self.log_path.mkdir(parents=True, exist_ok=True)
		else:
			self.log_path = None

		self.total_time_cost = 0
		self.total_infer_step = 0

		self.comm_client = Comm_Client(url=self.config['comm_server']['comm_client'], api_key=self.config['comm_server']['api_key'], api_base=self.config['comm_server']['api_base'])
		self.vlm_client = VLM_Client(url=self.config['comm_server']['vlm_client'])
		self.nego_assistant = Nego_Client(url=self.config['comm_server']['comm_client'], api_key=self.config['comm_server']['api_key'], api_base=self.config['comm_server']['api_base'])

		self.dir_inten_veh = ['STOP' for _ in range(self.ego_vehicles_num)]
		self.speed_inten_veh = ['STOP' for _ in range(self.ego_vehicles_num)]
		self.print_comm = ''

		self.dir_inten_list = ['turn left at intersection', 'turn right at intersection', 'go straight at intersection', 'follow the lane', 'left lane change', 'right lane change']
		self.speed_inten_list = ['STOP', 'SLOWER', 'FASTER', 'KEEP']
  
		self.vlm_next_infer_time = [0 for _ in range(self.ego_vehicles_num)]
		self.vlm_prev_content = [None for _ in range(self.ego_vehicles_num)]
		self.vlm_current_content = [None for _ in range(self.ego_vehicles_num)]
		self.next_comm_time = [0 for _ in range(self.ego_vehicles_num)]
		self.prev_comm = [None for _ in range(self.ego_vehicles_num)]
		self.round_comm_complete_time = {}
		self.round_comm_bank = {}
		for id in range(self.ego_vehicles_num):
			self.round_comm_bank[id] = []
			self.round_comm_complete_time[id] = []
		
		self.comm_bank = {}
		for id in range(self.ego_vehicles_num):
			self.comm_bank[id] = {'speed': 'KEEP', 'nav': 'follow the lane'}

		self.distance_to_conflict = {}
		for id in range(self.ego_vehicles_num):
			self.distance_to_conflict[id] = 1000

		self.cons_anly = ''
		self.message_ego = ''
		self.response_message = ''
		self.comm_dict_ego = ''

		self.planning_bank = {}
		for id_1 in range(self.ego_vehicles_num):
			self.planning_bank[id_1] = np.ones((20,2)) * id_1 * 100

		self.waypoints_num = self.config['planning']['waypoints_num']

		self.nego_dur_bank = {}
		for id_1 in range(self.ego_vehicles_num):
			self.nego_dur_bank[id_1] = 0
		self.nego_group_bank = []
		
		realtime_mode = os.environ.get('REALTIME_MODE', '0')
		self.realtime = True if realtime_mode=='1' else False
		self.negotiation_enabled = os.environ.get("COLMDRIVER_DISABLE_NEGOTIATION", "0") != "1"
		if not self.negotiation_enabled:
			print("[INFO] Negotiation disabled via COLMDRIVER_DISABLE_NEGOTIATION.")

	def judge_speed(self, i):
		# our trained planning model set a reversed FASTER and KEEP, so need to align back the index.
		# you can set your own mappings
		if i==0:return 0
		if i==1:return 1
		if i==2:return 3
		if i==3:return 2

	def group_negotiation(self, group, comm_info_list, default_order, comm_results, total_car_data_raw, total_rsu_data_raw, step, timestamp):
		'''
		comm_info_list = [{'ego_id': car_id, 'ego_speed': car_data_raw[car_id]['measurements']['speed'], 'ego_intention': intention, 'other_direction': dir_list, 'other_heading': heading_list, 'other_distance': dis_list, 'other_speed': speed_list, 'other_intention': intention_list}, ...]
		'''
		# version 1: human like communication
		self.local_max_round = 3
		local_comm_content = []

		suggestion = ''
		comm_time_all = 0
		sum_action_time_all = 0
		for k in range(self.local_max_round):
			# actor
			for order, o in enumerate(default_order):
				car_id = comm_info_list[o]['ego_id']
				print('actor\n local round: ', k, ' order: ', order, ' car_id: ', car_id)
				message, comm_time = self.comm_client.comm(comm_info_list, car_id, local_comm_content, suggestion)
				print('message: ', message)
				if self.realtime:
					print(f"round {k} comm time {comm_time} car id {car_id}")
					comm_time_all += comm_time
				local_comm_content.append({'id': car_id, 'message': message})
			# critic
			print('\ncritic')
			#Fixes bug with numbers being printed after "critic", causing crash
			retry_attempt = 0
			max_retries = 3
			fallback_used = False
			while True:
				try:
					action_list, sum_action_time = self.comm_client.sum_action(local_comm_content[-len(group):])
					expected_ids = {str(item['ego_id']) for item in comm_info_list}
					missing_ids = [ego_id for ego_id in expected_ids if ego_id not in action_list]
					if missing_ids:
						fallback_used = True
						print(f"[WARN] Missing actions for vehicles {missing_ids}. Applying KEEP fallback.")
						for ego_id in missing_ids:
							action_list[ego_id] = {'speed': 'KEEP'}
					for item in comm_info_list:
						action_list[str(item['ego_id'])]['nav'] = item['ego_intention']
						if action_list[str(item['ego_id'])]['speed'] == 'KEEP' and item['ego_speed'] < 4:
							action_list[str(item['ego_id'])]['speed'] = 'FASTER'
					# check key 'nav' in action_list[value], if nav not in action_list[value], remove action_list[value]
					keys_to_remove = [key for key, value in action_list.items() if 'nav' not in value]
					for key in keys_to_remove:
						del action_list[key]
					break
				except Exception as e:
					retry_attempt += 1
					print(f"[WARN] sum_action attempt {retry_attempt}/{max_retries} failed with error: {e}")
					if retry_attempt >= max_retries:
						expected_ids = {str(item['ego_id']) for item in comm_info_list}
						action_list = {ego_id: {'speed': 'KEEP'} for ego_id in expected_ids}
						sum_action_time = 0
						fallback_used = True
						print("[WARN] sum_action retries exhausted. Using KEEP fallback for all vehicles.")
						break
					time.sleep(0.1)
			if fallback_used:
				print(f"[INFO] Fallback action list used: {action_list}")
			print('action_list: ', action_list)
			if self.realtime:
				print(f"round {k} sum action time {sum_action_time}")
				sum_action_time_all += sum_action_time

			for key, value in action_list.items():
				comm_results[int(key)] = value

			pred_waypoints = []
			for _id in group:
				waypoints = self.get_action_from_list_inter(_id, total_car_data_raw[_id], total_rsu_data_raw[_id], step, timestamp, comm_info=comm_results[_id], nego=True)
				pred_waypoints.append(waypoints)

			score, suggestion, detailed_score = self.nego_assistant.evaluator(comm_results, local_comm_content, pred_waypoints, self.log_path, step)

			for id in group:
				self.round_comm_complete_time[id].append(step*0.05+ comm_time_all + sum_action_time_all)
			for key, value in comm_results.items():
				self.round_comm_bank[key].append(value)

			# check agreement
			if suggestion=='':
				print('Consensus reached!\n')
				break
			print('\n\n')

		print("step: ", step, "round_comm_complete_time: ", self.round_comm_complete_time)
		self.print_comm = f'comm results: {self.comm_bank}, current time: {step*0.05}, nego_dur_time: {self.nego_dur_bank}, next_comm_time: {self.next_comm_time}, vlm_next_infer_time: {self.vlm_next_infer_time}'

		# The comm_bank stores the comm result to be used, and it should be used after the comm ends (after next_comm_time)	
		self.comm_bank = self.prev_comm
		for key, value in comm_results.items():
			self.prev_comm[key] = value

		# If the path starts comm, it can be used directly because there is no negotiation time reserved, which does not match the actual situation
		for id, comm_info in enumerate(self.comm_bank):
			if comm_info is None:
				self.comm_bank[id] = self.prev_comm[id]
		print(f"comm_time_all: {comm_time_all}, sum_action_time_all: {sum_action_time_all}")

		for car_id in group:
			self.next_comm_time[car_id] = step*0.05 + comm_time_all + sum_action_time_all
		print("current_time:", step*0.05, "next_comm_time:", self.next_comm_time)
		# print("current comm results:", self.comm_bank)

		return comm_results

	def form_comm_group(self, car_data_raw, step):
		'''construct communication graph, agent i and j will communicate if their safety score is below a threshold'''
		if not self.negotiation_enabled:
			return [], []

		def find_connected_vehicles(i):
			'''Use DFS to find all connected vehicles'''
			stack = [i]
			connected_vehicles = []
			max_dur_in_group = 0

			while stack:
				vehicle = stack.pop()
				if not visited[vehicle]:
					visited[vehicle] = True
					connected_vehicles.append(vehicle)
					for j in range(self.ego_vehicles_num):
						if not visited[j] and car_data_raw[j] is not None:
							min_distance, safety_score, dur, dis1, dis2 = self.nego_assistant.safety_score_with_distance(self.planning_bank[vehicle], self.planning_bank[j], self.distance_to_conflict[vehicle], self.distance_to_conflict[j])
							self.distance_to_conflict[vehicle] = dis1
							self.distance_to_conflict[j] = dis2

							self.nego_assistant.plot_and_save_waypoints([self.planning_bank[vehicle], self.planning_bank[j]], self.log_path / pathlib.Path(f'waypoints/{step}_{vehicle}_{j}_{round(min_distance, 2)}.png'))
							if safety_score < 0.75:
								stack.append(j)
								print('safety_score: ', safety_score)
								if dur > max_dur_in_group:
									max_dur_in_group = dur
			return connected_vehicles, max_dur_in_group

		def merge_groups(nego_groups, nego_group_bank):
			i = 0
			while i < len(nego_groups):
				current_group = nego_groups[i]
				j = 0
				while j < len(nego_group_bank):
					bank_group = nego_group_bank[j]['group']
					if set(current_group) & set(bank_group):
						merged_group = list(set(current_group) | set(bank_group))
						nego_groups[i] = merged_group
						del nego_group_bank[j]
						break
					j += 1
				else:
					i += 1
			return nego_groups, nego_group_bank

		nego_groups = []
		group_max_durs = []
		visited = [False] * self.ego_vehicles_num
		# Traverse all vehicles and find all indirectly affected vehicle groups
		print('\nFind connected vehicles...')
		for i in range(self.ego_vehicles_num):
			if not visited[i] and car_data_raw[i] is not None:
				group, max_dur = find_connected_vehicles(i)
				if len(group) > 1:
					comm_flag = True
					for car_id in group:
						if step * 0.05 < self.next_comm_time[car_id]:
							comm_flag = False
							break
					if comm_flag:
						nego_groups.append(group)
						group_max_durs.append(max_dur)
		# print('Connected vehicles:', nego_groups)
		# print('group_max_durs:', group_max_durs)
		# print('self.nego_group_bank:', self.nego_group_bank)

		group_bank = self.nego_group_bank.copy()
		for group in group_bank:
			if group['group'] in nego_groups:
				# still conflict at end of duration, give longer execution time
				if step - group['nego_start'] >= self.nego_dur_bank[group['group'][0]]-8 and step - group['nego_start'] <= self.nego_dur_bank[group['group'][0]]+8:
					self.nego_dur_bank[group['group'][0]] = step + group_max_durs[nego_groups.index(group['group'])] + 20
					print('Extend negotiation duration!')
				elif step - group['nego_start'] > self.nego_dur_bank[group['group'][0]]+8:
					self.nego_group_bank.remove(group)
					print('Clear bank, negotiation done!')

				# cancel negotiation if just negotiated for 1 s
				if step - group['nego_start'] < 20:
					group_max_durs.remove(group_max_durs[nego_groups.index(group['group'])])
					nego_groups.remove(group['group'])
					print('Remove group:', group['group'], ', just negotiated!')

		nego_groups, self.nego_group_bank = merge_groups(nego_groups, self.nego_group_bank)
		
		print('Connected vehicles:', nego_groups)
		print('group_max_durs:', group_max_durs)
		print('self.nego_group_bank:', self.nego_group_bank)
		print('self.nego_dur_bank:', self.nego_dur_bank)
		for group in nego_groups:
			group[:] = [i for i in group if car_data_raw[i] is not None]
		return nego_groups, group_max_durs

	def _run_heal_perception_single_agent(self, car_data_raw):
		"""Run a HEAL HeterBaseline detector on the ego LiDAR.

		Returns ``(corners, scores)`` where ``corners`` is a ``(P, 8, 3)``
		torch tensor on ``self.device`` and ``scores`` is a ``(P,)`` torch
		tensor — same convention as codriving's own ``pred_box_tensor`` /
		``pred_score`` BEFORE the in-place x↔y swap. Single-agent path
		(degenerate vs codriving's cooperative perception, but enough for
		the detector-quality ablation). Mirrors the v2v helper so the
		downstream LiDAR→ego transform applies unchanged.
		"""
		_pcd = car_data_raw[0]['lidar']
		if not isinstance(_pcd, np.ndarray):
			_pcd = np.asarray(_pcd)
		_pcd = _pcd.astype(np.float32, copy=False)
		if _pcd.ndim != 2 or _pcd.shape[-1] < 3:
			_empty_corners = torch.zeros((0, 8, 3), dtype=torch.float32, device=self.device)
			_empty_scores = torch.zeros((0,), dtype=torch.float32, device=self.device)
			return _empty_corners, _empty_scores
		if _pcd.shape[-1] == 3:
			_pad = np.zeros((_pcd.shape[0], 1), dtype=np.float32)
			_pcd = np.concatenate([_pcd, _pad], axis=-1)
		elif _pcd.shape[-1] > 4:
			_pcd = _pcd[:, :4]
		_det = self._heal_perception.detect(_pcd)
		if _det.P == 0:
			_empty_corners = torch.zeros((0, 8, 3), dtype=torch.float32, device=self.device)
			_empty_scores = torch.zeros((0,), dtype=torch.float32, device=self.device)
			return _empty_corners, _empty_scores
		_corners = torch.from_numpy(np.ascontiguousarray(_det.corners)).float().to(self.device)
		_scores = torch.from_numpy(np.ascontiguousarray(_det.scores)).float().to(self.device)
		return _corners, _scores

	def get_action_from_list_multi_vehicle(self, car_data_raw, rsu_data_raw, step, timestamp):
		# keys(['gps_x', 'gps_y', 'x', 'y', 'theta', 'speed', 'compass', 'command', 'target_point'])
		print('step infer time: ', (step+1)*0.05, self.vlm_next_infer_time, self.next_comm_time)
		## re-pack driving information
		self.step = step
		self.nego_assistant.nego_content['action'][step] = []
		control_all = []
		total_car_data_raw = []
		total_rsu_data_raw = []
		for ego_id in range(self.ego_vehicles_num):
			new_car_data_raw = []
			new_rsu_data_raw = []
			if car_data_raw[ego_id] is not None:
				new_car_data_raw.append(car_data_raw[ego_id])
				if len(rsu_data_raw) > 0:
					if rsu_data_raw[ego_id] is not None:
						new_rsu_data_raw.append(rsu_data_raw[ego_id])
					else:
						pass
			else:
				new_car_data_raw.append(None)
			total_car_data_raw.append(new_car_data_raw)
			total_rsu_data_raw.append(new_rsu_data_raw)

		## form communication group
		nego_groups, group_max_durs = self.form_comm_group(car_data_raw, step)

		# reset self.distance_to_conflict
		dis_to_conflict = {id: self.distance_to_conflict[id] for id in range(self.ego_vehicles_num)}
		self.distance_to_conflict = {id: 1000 for id in range(self.ego_vehicles_num)}
		print('dis_to_conflict:', dis_to_conflict)

		## start negotiation for each pair of cooperator
		self.print_comm = ''
		if len(nego_groups)>0:
			for gi, group in enumerate(nego_groups):
				# priority: straight = follow lane > turn right > turn left > lane change right > lane change left
				rule_based_results = {}
				min_distance_to_conflict = 1000
				highest_priority_car_id = 0
				for car_id in group:
					flag = False
					if dis_to_conflict[car_id] < min_distance_to_conflict:
						flag = True
					if flag:
						rule_based_results[highest_priority_car_id] = {'nav': self.dir_inten_veh[highest_priority_car_id], 'speed': 'STOP'}
						rule_based_results[car_id] = {'nav': self.dir_inten_veh[car_id], 'speed': self.speed_inten_veh[car_id]}
						# before highest priority id modification
						min_distance_to_conflict = dis_to_conflict[car_id]
						highest_priority_car_id = car_id
					else: 
						rule_based_results[car_id] = {'nav': self.dir_inten_veh[car_id], 'speed': 'STOP'}
					self.nego_dur_bank[car_id] = step + group_max_durs[gi] + 20
				if (rule_based_results[highest_priority_car_id]['speed'] == 'KEEP' and car_data_raw[highest_priority_car_id]['measurements']['speed'] < 4) or rule_based_results[highest_priority_car_id]['speed'] == 'STOP':
					rule_based_results[highest_priority_car_id]['speed'] = 'FASTER'
				print('rule_based_results in complex!!!: ', rule_based_results)
				for key, value in rule_based_results.items():
					self.comm_bank[key] = value

		## start ego inference
		for ego_id in range(self.ego_vehicles_num):
			if total_car_data_raw[ego_id][0] is not None:
				# check negotiation duration
				if step < self.nego_dur_bank[ego_id]:
					comm_info = self.comm_bank[ego_id]
					# print('ego inference time: ', step*0.05, "round_comm_complete_time: ", self.round_comm_complete_time[ego_id], "round_comm_bank: ", self.round_comm_bank[ego_id])
					while len(self.round_comm_complete_time[ego_id]) > 0 and step * 0.05 > self.round_comm_complete_time[ego_id][0]:
						self.round_comm_complete_time[ego_id].pop(0)
						comm_info = self.round_comm_bank[ego_id].pop(0)	
					print('\nVLM using negotiation results!')
					print('step:', step, 'comm_info:', comm_info)
				else:
					comm_info = None
					print('\nVLM infer independently!')
				control_all.append(self.get_action_from_list_inter(ego_id, total_car_data_raw[ego_id], total_rsu_data_raw[ego_id], step, timestamp, comm_info=comm_info)[0])
			else:
				control_all.append(self.prev_control[ego_id])

		if self.log_path is not None:
			with open(self.log_path / pathlib.Path("nego.json"), 'w') as f:
				json.dump(self.nego_assistant.nego_content, f, indent=4)

		return control_all

	def get_action_from_list_inter(self, ego_id, car_data_raw, rsu_data_raw, step, timestamp, comm_info=None, nego=False):
		'''
		generate the action for N cars from the record data.

		Parameters
		----------
		car_data : list[ dict{}, dict{}, None, ...],
		rsu_data : list[ dict{}, dict{}, None, ...],
		model : trained model, probably we can store it in the initialization.
		step : int, frame in the game, 20hz.
		timestamp : float, time in the game.
	
		Returns
		-------
		controll_all: list, detailed actions for N cars.
		'''
		
		## communication latency, get data from 6 frames past (6*50ms = 300ms)
		if 'comm_latency' in self.config['simulation']:
			raw_data_dict = {'car_data': car_data_raw, 
							'rsu_data': rsu_data_raw}
			self.pre_raw_data_bank.update({step: raw_data_dict})
			latency_step = self.config['simulation']['comm_latency']
			sorted_keys = sorted(list(self.pre_raw_data_bank.keys()))
			if step > latency_step:
				if step-sorted_keys[0] > latency_step:   
					self.pre_raw_data_bank.pop(sorted_keys[0])
				if step-latency_step in self.pre_raw_data_bank:
					raw_data_used = self.pre_raw_data_bank[step-latency_step]
					for i in range(len(car_data_raw)):
						if i > 0:
							car_data_raw[i] = raw_data_used['car_data'][i]
					rsu_data_raw = raw_data_used['rsu_data']
				else:
					print('latency data not found!')

		# initalize data folder
		folder_path = self.save_path / pathlib.Path("ego_vehicle_{}".format(0))
		if not os.path.exists(folder_path):
			os.mkdir(folder_path)

		### load data for visualization and planning
		car_data, car_mask = self.check_data(car_data_raw)
		rsu_data, _ = self.check_data(rsu_data_raw, car=False)
		batch_data = self.collate_batch_infer_perception(car_data, rsu_data)  # batch_size: N*(N+M)

		### load data for perception
		extra_source = {}
		actors_data = self.collect_actor_data(ego_id)
		for data in car_data_raw + rsu_data_raw:
			data['actors_data'] = actors_data
		extra_source['car_data'] = car_data_raw
		car_lidar_len = extra_source['car_data'][0]['lidar'].shape[0]
		extra_source['rsu_data'] = rsu_data_raw
		if len(rsu_data_raw)>0:
			rsu_lidar_len = extra_source['rsu_data'][0]['lidar'].shape[0]
		else:
			# print('No RSU!')
			rsu_lidar_len = 0
		lidar_len = [car_lidar_len,rsu_lidar_len]
		data = self.perception_dataloader.__getitem__(idx=None, extra_source=extra_source)
		batch_data_perception = [data]
		batch_data_perception = self.perception_dataloader.collate_batch_test(batch_data_perception, online_eval_only=False)
		batch_data_perception = train_utils.to_device(batch_data_perception, self.device)
		
		output_dict = OrderedDict()
		for cav_id, cav_content in batch_data_perception.items():
			output_dict[cav_id] = self.perception_model(cav_content)
		pred_box_tensor, pred_score, gt_box_tensor = \
			self.perception_dataloader.post_process_multiclass(batch_data_perception,
								output_dict, online_eval_only=False)
		infer_result = {"pred_box_tensor" : pred_box_tensor, \
						"pred_score" : pred_score, \
						"gt_box_tensor" : gt_box_tensor}
		if "comm_rate" in output_dict['ego']:
			infer_result.update({"comm_rate" : output_dict['ego']['comm_rate']})

		# record perception info
		perception_num = []
		self.perception_info = {0: [], 1: [], 2: []} # vehicle, cyclist, pedestrain
		try:
			for pt, perc_type in enumerate(infer_result['pred_box_tensor']):
				if perc_type is not None:
					perception_num.extend([pt for p in range(len(perc_type[0]))])
		except: pass

		attrib_list = ['pred_box_tensor', 'pred_score', 'gt_box_tensor']
		for attrib in attrib_list:
			if isinstance(infer_result[attrib], list):
				infer_result_tensor = []
				for i in range(len(infer_result[attrib])):
					if infer_result[attrib][i] is not None:
						infer_result_tensor.append(infer_result[attrib][i])
				if len(infer_result_tensor)>0:
					infer_result[attrib] = torch.cat(infer_result_tensor, dim=0)
				else:
					infer_result[attrib] = None

		# Optional: replace codriving's perception output with HEAL detector
		# boxes. Single-agent inference path. `output_dict['ego']['fused_feature']`
		# is left untouched here so the feature-stash step can either keep it
		# (if zero_bev_feature is False) or zero it (default for the variant
		# configs). HEAL is single-class (vehicle), so we also reset
		# perception_num to all zeros — it's used downstream to bucket
		# detections into self.perception_info[0/1/2] for the LLM context.
		if self._heal_perception is not None:
			heal_corners, heal_scores = self._run_heal_perception_single_agent(car_data_raw)
			infer_result['pred_box_tensor'] = heal_corners
			infer_result['pred_score'] = heal_scores
			# gt_box_tensor is GT (used for vis only); leave as codriving's GT.
			perception_num = [0] * (heal_corners.shape[0] if heal_corners is not None else 0)

		infer_result_clone = copy.deepcopy(infer_result)

		# Stash world-frame perception predictions for the open-loop
		# AP@IoU evaluator. Pure observation: the agent's OpenLoopAPHook
		# reads this dict (when --openloop is active) and ignores it
		# otherwise. Mirrors the pnp_infer_action_e2e_v2v.py / colmdriver_action.py
		# pattern so all three planners share AP plumbing. Closed-loop
		# behavior is unaffected — the dict is written, never read here.
		try:
			_meas_full = car_data_raw[0]['measurements']
			_pred_world_corners_list = []
			if (infer_result['pred_box_tensor'] is not None
					and len(infer_result['pred_box_tensor']) > 0):
				for _i in range(infer_result['pred_box_tensor'].shape[0]):
					_world_corners = transform_2d_points(
						infer_result['pred_box_tensor'][_i].cpu().numpy(),
						np.pi / 2 - _meas_full["theta"],
						_meas_full["lidar_pose_y"],
						_meas_full["lidar_pose_x"],
						0.0, 0.0, 0.0,  # car2 = (0, 0, 0) → output stays in world frame
					)
					_pred_world_corners_list.append(_world_corners)
			_pred_world_arr = (
				np.stack(_pred_world_corners_list, axis=0).astype(np.float32)
				if _pred_world_corners_list
				else np.zeros((0, 8, 3), dtype=np.float32)
			)
			_pred_score_arr = (
				infer_result['pred_score'].cpu().numpy().astype(np.float32)
				if infer_result.get('pred_score') is not None
				else np.zeros((0,), dtype=np.float32)
			)
			# Raw opencood lidar-frame predictions (untransformed). The
			# AP hook prefers this over pred_box_world because it matches
			# perception_swap_agent's pred_frame='ego' convention. The
			# in-place x↔y swap below hasn't happened yet at this point.
			_pred_lidar_arr = np.zeros((0, 8, 3), dtype=np.float32)
			try:
				_pbt = infer_result.get('pred_box_tensor')
				if _pbt is not None and len(_pbt) > 0:
					_pred_lidar_arr = _pbt.cpu().numpy().astype(np.float32)
			except Exception:
				pass
			if not hasattr(self, '_last_perception_for_save'):
				self._last_perception_for_save = {}
			self._last_perception_for_save[ego_id] = {
				'pred_box_world': _pred_world_arr,
				'pred_box_lidar': _pred_lidar_arr,
				'pred_score':     _pred_score_arr,
				'lidar_pose_x':   float(_meas_full["lidar_pose_x"]),
				'lidar_pose_y':   float(_meas_full["lidar_pose_y"]),
				'lidar_pose_z':   float(_meas_full.get("lidar_pose_z", 0.0)),
				'theta':          float(_meas_full["theta"]),
				'ego_x':          float(_meas_full.get("x", 0.0)),
				'ego_y':          float(_meas_full.get("y", 0.0)),
			}
		except Exception:
			# Never let observation-stashing break the inference path.
			pass

		### filte out ego box
		if not infer_result['pred_box_tensor'] is None:
			if len(infer_result['pred_box_tensor']) > 0:
				tmp = infer_result['pred_box_tensor'][:,:,0].clone()
				infer_result['pred_box_tensor'][:,:,0]=infer_result['pred_box_tensor'][:,:,1]
				infer_result['pred_box_tensor'][:,:,1] = tmp
			measurements = car_data_raw[0]['measurements']
			num_object = infer_result['pred_box_tensor'].shape[0]
			object_list = []
			# transform from lidar pose to ego pose
			for i in range(num_object):
				transformed_box = transform_2d_points(
						infer_result['pred_box_tensor'][i].cpu().numpy(),
						np.pi/2 - measurements["theta"], # car1_to_world parameters
						measurements["lidar_pose_y"],
						measurements["lidar_pose_x"],
						np.pi/2 - measurements["theta"], # car2_to_world parameters, note that not world_to_car2
						measurements["y"],
						measurements["x"],
					)
				location_box = np.mean(transformed_box[:4,:2], 0)
				if np.linalg.norm(location_box) < 1.4:
					continue
				else:
					try:
						self.perception_info[perception_num[i]].append(location_box)
					except:
						pass
				object_list.append(torch.from_numpy(transformed_box))
			if len(object_list) > 0:
				processed_pred_box = torch.stack(object_list, dim=0)
			else:
				processed_pred_box = infer_result['pred_box_tensor'][:0]
		else:
			processed_pred_box = [] # infer_result['pred_box_tensor']

		### turn boxes into occupancy map
		if len(processed_pred_box) > 0:
			occ_map = turn_traffic_into_map(processed_pred_box[:,:4,:2].cpu(), self.det_range)
		else:
			occ_map = turn_traffic_into_map(processed_pred_box, self.det_range)

		# # N, K, H, W, C=7
		occ_map_shape = occ_map.shape
		occ_map = torch.from_numpy(occ_map).cuda().contiguous().view((-1, 1) + occ_map_shape[1:]) 
		# N, 1, H, W
		
		da = []
		for i in range(1): # len(car_data_raw)
			da.append(torch.from_numpy(car_data_raw[i]['drivable_area']).cuda().float().unsqueeze(0))
		
		# load feature
		perception_results = output_dict['ego']
		fused_feature_2 = perception_results['fused_feature'].permute(0,1,3,2) #  ([2, 128, 96, 288]) -> ([2, 128, 288, 96])
		fused_feature_3 = torch.flip(fused_feature_2, dims=[2])
		feature = fused_feature_3[:,:,:192,:]
		# Optional ablation: zero the BEV feature path so the planner runs on
		# rasterized detections only. Default for the colmdriver_rulebase_<heal>
		# variants (zero influence from codriving's perception backbone).
		if self._zero_bev_feature:
			feature = torch.zeros_like(feature)

		self.perception_memory_bank[ego_id].pop(0)
		if len(self.perception_memory_bank[ego_id])<5:
			for _ in range(5 - len(self.perception_memory_bank[ego_id])):
				self.perception_memory_bank[ego_id].append({
					'occ_map': occ_map, # N, 1, H, W
					'drivable_area': torch.stack(da), # N, 1, H, W
					'detmap_pose': batch_data['detmap_pose'][:1], # N, 3 :len(car_data_raw)
					'target': batch_data['target'][:1], # N, 2 :len(car_data_raw)
					'feature': feature, # N, 128, H, W
				})
		
		ego_llm_id = 0
		ego_data = car_data_raw[ego_llm_id] # raw data for ego vehicle
		front_img = ego_data['rgb_front']
		measurements = car_data_raw[0]['measurements']
		self.cmd_direction = torch.zeros(6)
		self.cmd_speed = torch.zeros(4)

		### input prompt and images for llms
		# update vlm output: execute vlm inference, and the result should be adopted when next inference starts (end of this inference)
		if (step+1)*0.05 > self.vlm_next_infer_time[ego_id]:
			print(f'\nVLM infering for car {ego_id}...')
			system_prompt = read_txt_file(self.config['llm']['system_prompt_file'])
			det_type = self.config['llm'].get('det_type','w_det')
			system_prompt = self.vlm_client.edit_prompt(system_prompt, measurements, comm_info, self.perception_info, det_type)

			output_content, infer_time = self.vlm_client.infer(front_img, system_prompt)
			self.vlm_prev_content[ego_id] = self.vlm_current_content[ego_id]
			self.vlm_current_content[ego_id] = output_content
			# if not have stored output, use current output to init
			if self.vlm_prev_content[ego_id] is None:
				self.vlm_prev_content[ego_id] = output_content
			
			if not self.realtime:
				infer_time = 0
				self.vlm_prev_content[ego_id] = output_content
			self.vlm_next_infer_time[ego_id] += infer_time
			print('vlm output:',output_content)

		# adopt vlm output
		try:
			if comm_info != None:
				for i, speed_inten in enumerate(self.speed_inten_list):
					if speed_inten == comm_info['speed']:
						cmd_index = self.judge_speed(i)
						self.cmd_speed[cmd_index] = 1
						self.speed_inten_veh[ego_id] = speed_inten
						break
				for i, dir_inten in enumerate(self.dir_inten_list):
					if dir_inten == comm_info['nav']:
						self.cmd_direction[i] = 1
						self.dir_inten_veh[ego_id] = dir_inten
						break
				print('using nego result: ', comm_info['nav'], comm_info['speed'])
			else:
				# nav info is steady in vlm output, so we omit the vlm output of nav to save time
				command = int(measurements['command'])
				if command in command_dict.keys():
					command_str = command_dict[command]
				else:
					command_str = 'no command'
				output_content = self.vlm_prev_content[ego_id] + ' ' + command_str
				print(f"using vlm output: {self.vlm_prev_content[ego_id]}")

				for i, dir_inten in enumerate(self.dir_inten_list):
					if dir_inten in output_content:
						self.cmd_direction[i] = 1
						self.dir_inten_veh[ego_id] = dir_inten
						break
				for i, speed_inten in enumerate(self.speed_inten_list):
					if speed_inten in output_content:
						cmd_index = self.judge_speed(i)
						self.cmd_speed[cmd_index] = 1
						self.speed_inten_veh[ego_id] = speed_inten
						break
		except Exception as e:
			print('vlm inference failed because ', e)

		### Turn the memoried perception output into planning input
		planning_input = self.generate_planning_input(ego_id) # planning_input['occupancy'] [1, 5, 6, 192, 96] planning_input['target'] [1,2]

		predicted_waypoints = self.planning_model(planning_input) # [1, 10, 2]
		predicted_waypoints = predicted_waypoints['future_waypoints']
		# Stash raw local-frame waypoints for the open-loop BEV viz path
		# (see colmdriver_action.py for rationale).
		try:
			if not hasattr(self, "last_predicted_waypoints_local"):
				self.last_predicted_waypoints_local = {}
				self.last_predicted_waypoints_step = {}
			self.last_predicted_waypoints_local[ego_id] = (
				np.asarray(predicted_waypoints[0].detach().cpu().numpy(), dtype=np.float64)
				if hasattr(predicted_waypoints[0], "detach")
				else np.asarray(predicted_waypoints[0], dtype=np.float64)
			).copy()
			self.last_predicted_waypoints_step[ego_id] = int(step)
		except Exception:
			pass
		# predicted_waypoints: N, T_f=10, 2
		# transform waypoints for negotiation
		x_world = measurements['x']
		y_world = measurements['y']
		theta = measurements['theta']

		theta_prime = theta
		rotation_matrix = np.array([
			[np.cos(theta_prime), -np.sin(theta_prime)],
			[np.sin(theta_prime), np.cos(theta_prime)]
		])
		
		transformed_waypoints = []
		for waypoint in predicted_waypoints[0]:
			x_car, y_car = waypoint
			transformed_point = np.dot(rotation_matrix, np.array([x_car.item(), y_car.item()]))
			transformed_point += np.array([x_world, y_world])
			transformed_waypoints.append(transformed_point)
		waypoints = np.array(transformed_waypoints)
		# print('waypoints:',waypoints[0], waypoints[9], waypoints[-1])

		if nego:
			return waypoints
		else:
			self.planning_bank[ego_id] = waypoints
			if comm_info != None:
				self.nego_assistant.nego_content['action'][step].append(comm_info['speed'])
			else:
				pass

		# visualization
		if step % self.skip_frames == 0:
			vis_save_path = os.path.join(folder_path, '3d_%04d.png' % step)
			lidar_3d = visualize(infer_result_clone,
							batch_data_perception['ego'][
								'origin_lidar'][0],
							self.config['perception']['perception_hypes']['postprocess']['gt_range'],
							vis_save_path,
							method='first_person',
							left_hand=False,
							measurements=car_data_raw[0]['measurements'],
							lidar_len=lidar_len,
							predicted_waypoints=predicted_waypoints,
							target_point = batch_data['target'][:len(car_data_raw)],
							waypoints_num = self.waypoints_num)
			vis_save_path = os.path.join(folder_path, 'bev_%04d.png' % step)
			lidar_bev = visualize(infer_result_clone,
							batch_data_perception['ego'][
								'origin_lidar'][0],
							self.config['perception']['perception_hypes']['postprocess']['gt_range'],
							vis_save_path,
							method='bev',
							left_hand=False,
							measurements=car_data_raw[0]['measurements'],
							lidar_len=lidar_len,
							predicted_waypoints=predicted_waypoints,
							target_point = batch_data['target'][:len(car_data_raw)],
							waypoints_num = self.waypoints_num)
		if step % self.skip_frames == 0:
			batch_data['lidar_3d'] = lidar_3d
			batch_data['lidar_bev'] = lidar_bev

		### output postprocess to generate the action, list of actions for N agents
		control_all = self.generate_action_from_model_output(ego_id, predicted_waypoints, car_data_raw, rsu_data_raw, car_data, rsu_data, batch_data, planning_input, car_mask, step, timestamp)
		return control_all
	
	def generate_planning_input(self, ego_id):
		
		occ_final = torch.zeros(1, 5, 6, 192, 96).cuda().float() # self.ego_vehicles_num, 5, 6, 192, 96
		# N, T, C, H, W

		self_car_map = render_self_car( 
			loc=np.array([0, 0]),
			ori=np.array([0, -1]),
			box=np.array([2.45, 1.0]),
			color=[1, 1, 0], 
			pixels_per_meter=8,
			max_distance=self.max_distance
		)[:, :, 0]
		self_car_map = block_reduce(self_car_map, block_size=(2, 2), func=np.mean)
		self_car_map = np.clip(self_car_map, 0.0, 255.0)
		self_car_map = self_car_map[:48*4, 48*2:48*4]  # H, W
		occ_ego_temp = torch.from_numpy(self_car_map).cuda().float()[None, None, None, :, :].repeat(1, 5, 1, 1, 1) # self.ego_vehicles_num, 5, 1, 1, 1


		coordinate_map = torch.ones((5, 2, 192, 96))
		for h in range(192):
			coordinate_map[:, 0, h, :] *= h*self.det_range[-1]-self.det_range[0]
		for w in range(96):
			coordinate_map[:, 1, :, w] *= w*self.det_range[-1]-self.det_range[2]
		coordinate_map = coordinate_map.cuda().float()

		occ_to_warp = torch.zeros(1, 5, 2, 192, 96).cuda().float() # self.ego_vehicles_num, 5, 2, 192, 96
		# B, T, 2, H, W
		occ_to_warp[:, :, 1:2] = occ_ego_temp
		det_map_pose = torch.zeros(1, 5, 3).cuda().float() # self.ego_vehicles_num, 5, 3

		##################  end2end feature #####################
		feature_dim = self.perception_memory_bank[ego_id][0]['feature'].shape[1] # 128,256
		feature_to_warp = torch.zeros(1, 5, feature_dim, 192, 96).cuda().float() # self.ego_vehicles_num, 5, feature_dim, 192, 96

		feature_warped_list = []

		for agent_i in range(1): # self.ego_vehicles_num
			for t in range(5):
				occ_others = self.perception_memory_bank[ego_id][t]['occ_map'][agent_i]  # 1, H, W
				occ_to_warp[agent_i, t, 0:1] = occ_others
				feature_to_warp[agent_i , t, :] = self.perception_memory_bank[ego_id][t]['feature'][agent_i]
				det_map_pose[:, t] = self.perception_memory_bank[ego_id][t]['detmap_pose'] # N, 3
			
			local_command_map = render_self_car( 
				loc=self.perception_memory_bank[ego_id][-1]['target'][agent_i].cpu().numpy(),
				ori=np.array([0, 1]),
				box=np.array([1.0, 1.0]),
				color=[1, 1, 0], 
				pixels_per_meter=8,
				max_distance=self.max_distance
			)[:, :, 0]	
			local_command_map = block_reduce(local_command_map, block_size=(2, 2), func=np.mean)
			local_command_map = np.clip(local_command_map, 0.0, 255.0)
			local_command_map = torch.from_numpy(local_command_map[:48*4, 48*2:48*4]).cuda().float()[None, None, :, :].repeat(5, 1, 1, 1)

			da = self.perception_memory_bank[ego_id][-1]['drivable_area'][agent_i][None, :, :, :].repeat(5, 1, 1, 1) # 5, 1, H, W

			occ_final[agent_i, :, 2:3] = local_command_map
			occ_final[agent_i, :, 3:5] = coordinate_map
			occ_final[agent_i, :, 5:6] = da
		
			feature_warped = warp_image(det_map_pose, feature_to_warp)
			feature_warped_list.append(feature_warped)

		occ_warped = warp_image(det_map_pose, occ_to_warp)
		occ_final[:, :, :2] = occ_warped

		return {
			"occupancy": occ_final, # N, T=5, C=6, H=192, W=96
			"target": self.perception_memory_bank[ego_id][-1]['target'],  # N, 2
			"feature_warpped_list": feature_warped_list,
			"cmd_direction": torch.stack([self.cmd_direction], dim=0).cuda().float(),
			"cmd_speed": torch.stack([self.cmd_speed], dim=0).cuda().float()
		}

	def generate_action_from_model_output(self, ego_id, pred_waypoints_total, car_data_raw, rsu_data_raw, car_data, rsu_data, batch_data, planning_input, car_mask, step, timestamp):
		control_all = []
		tick_data = []
		ego_i = -1
		for count_i in range(1): # self.ego_vehicles_num
			if not car_mask[count_i]:
				control_all.append(None)
				tick_data.append(None)
				continue

			# store the data for visualization
			tick_data.append({})
			ego_i += 1
			# get the data for current vehicle
			pred_waypoints = np.around(pred_waypoints_total[ego_i].detach().cpu().numpy(), decimals=2)

			route_info = {
				'speed': car_data_raw[ego_i]['measurements']["speed"],
				'waypoints': pred_waypoints,
				'target': car_data_raw[ego_i]['measurements']["target_point"],
				'route_length': 0,
				'route_time': 0,
				'drive_length': 0,
				'drive_time': 0
			}

			steer, throttle, brake, meta_infos = self.controller[ego_id].run_step(
				route_info
			)

			if brake < 0.05:
				brake = 0.0
			if brake > 0.1:
				throttle = 0.0

			control = carla.VehicleControl()
			control.steer = float(steer)
			control.throttle = float(throttle)
			control.brake = float(brake)
			self.prev_control[ego_id] = control
			control_all.append(control)

			route_info['speed'] = route_info['speed'].tolist()
			route_info['target'] = route_info['target'].tolist()
			route_info['steer'] = float(steer)
			route_info['throttle'] = float(throttle)
			route_info['brake'] = float(brake)
			route_info['lidar_pose_x'] = car_data_raw[ego_i]['measurements']["lidar_pose_x"]
			route_info['lidar_pose_y'] = car_data_raw[ego_i]['measurements']["lidar_pose_y"]
			route_info['x'] = car_data_raw[ego_i]['measurements']["x"]
			route_info['y'] = car_data_raw[ego_i]['measurements']["y"]
			route_info['theta'] = float(car_data_raw[ego_i]['measurements']["theta"])
			route_info['waypoints'] = route_info['waypoints'].tolist()
			# See colmdriver_action.py for rationale.
			_pb = getattr(self, "planning_bank", {}) or {}
			_pb_wps = _pb.get(ego_id)
			if _pb_wps is not None:
				try:
					_pb_arr = np.asarray(_pb_wps, dtype=np.float64).reshape(-1, 2)
					_rel = _pb_arr - _pb_arr[0:1]
					_lx = float(car_data_raw[ego_i]['measurements']["lidar_pose_x"])
					_ly = float(car_data_raw[ego_i]['measurements']["lidar_pose_y"])
					_world = _rel + np.array([[_lx, _ly]], dtype=np.float64)
					route_info['waypoints_world'] = _world.tolist()
				except Exception:
					pass

			tick_data[ego_i]["planning"] = route_info

			cur_actors = planning_input["occupancy"][ego_i][-1][:3].cpu().permute(1, 2, 0).contiguous().numpy()
			cur_bev = (planning_input["occupancy"][ego_i][-1][-1:].cpu().permute(1, 2, 0).repeat(1, 1, 3)*120).contiguous().numpy()
			tick_data[ego_i]["map"] = np.where(cur_actors.sum(axis=2, keepdims=True)>5, cur_actors, cur_bev)
			tick_data[ego_i]["map"] = (tick_data[ego_i]["map"]/tick_data[ego_i]["map"].max()*255).astype(np.uint8)
			# 192, 96, 3
			cur_actors = planning_input["occupancy"][ego_i][-1][:3].cpu().permute(1, 2, 0).contiguous().numpy()
			cur_bev = (planning_input["occupancy"][ego_i][-1][-1:].cpu().permute(1, 2, 0).repeat(1, 1, 3)*120).contiguous().numpy()
			tick_data[ego_i]["map_gt"] = np.where(cur_actors.sum(axis=2, keepdims=True)>5, cur_actors, cur_bev)
			tick_data[ego_i]["map_gt"] = (tick_data[ego_i]["map_gt"]/tick_data[ego_i]["map_gt"].max()*255).astype(np.uint8)
			# 192, 96, 3
			tick_data[ego_i]["map_t1"] = planning_input["occupancy"][ego_i][-2][:3].cpu().permute(1, 2, 0).numpy()

			tick_data[ego_i]["rgb_raw"] = car_data_raw[ego_i]["rgb_front"]
			tick_data[ego_i]["lidar"] = np.rot90((np.transpose(car_data[ego_i]["lidar_original"], (1, 2, 0))*127).astype(np.uint8), k=1, axes=(1,0))
			try:
				tick_data[ego_i]["lidar_rsu"] = np.rot90((np.transpose(rsu_data[ego_i]["lidar_original"], (1, 2, 0))*127).astype(np.uint8), k=1, axes=(1,0))
			except:
				tick_data[ego_i]["lidar_rsu"] = np.ones_like(tick_data[ego_i]["lidar"])
			tick_data[ego_i]["rgb_left_raw"] = car_data_raw[ego_i]["rgb_left"]
			tick_data[ego_i]["rgb_right_raw"] = car_data_raw[ego_i]["rgb_right"]

			for t_i in range(self.waypoints_num):
				try:
					tick_data[ego_i]["map"][int(pred_waypoints[t_i][1]*4+144), int(pred_waypoints[t_i][0]*4+48)] = np.array([255, 0, 0])
				except Exception as e:
					print(e)
					print('ego_i', ego_i, int(pred_waypoints[t_i][1]*4+144), int(pred_waypoints[t_i][0]*4+48), pred_waypoints[t_i])
			tick_data[ego_i]["map"] = cv2.resize(tick_data[ego_i]["map"], (300, 600))
			tick_data[ego_i]["map_t1"] = cv2.resize(tick_data[ego_i]["map_t1"], (300, 600))
			tick_data[ego_i]["map_gt"] = cv2.resize(tick_data[ego_i]["map_gt"], (300, 600))
			tick_data[ego_i]["rgb"] = cv2.resize(tick_data[ego_i]["rgb_raw"], (3000, 1500))
			tick_data[ego_i]["lidar"] = cv2.resize(tick_data[ego_i]["lidar"], (600, 600))
			tick_data[ego_i]["lidar_rsu"] = cv2.resize(tick_data[ego_i]["lidar_rsu"], (600, 600))
			tick_data[ego_i]["rgb_left"] = cv2.resize(tick_data[ego_i]["rgb_left_raw"], (200, 150))
			tick_data[ego_i]["rgb_right"] = cv2.resize(tick_data[ego_i]["rgb_right_raw"], (200, 150))
			tick_data[ego_i]["rgb_focus"] = cv2.resize(tick_data[ego_i]["rgb_raw"][244:356, 344:456], (150, 150))
			if len(rsu_data_raw)>0:
				tick_data[ego_i]["control"] = "throttle: %.2f, steer: %.2f, brake: %.2f, ego: %.2f, %.2f/rsu: %.2f, %.2f, vlm: %s, %s" % (
					control.throttle,
					control.steer,
					control.brake,
					car_data_raw[ego_i]['measurements']["lidar_pose_x"],
					car_data_raw[ego_i]['measurements']["lidar_pose_y"],
					rsu_data_raw[ego_i]['measurements']["lidar_pose_x"],
					rsu_data_raw[ego_i]['measurements']["lidar_pose_y"],
					self.dir_inten_veh[ego_id],
					self.speed_inten_veh[ego_id],
				)
			else:
				tick_data[ego_i]["control"] = "throttle: %.2f, steer: %.2f, brake: %.2f, ego: %.2f, %.2f/rsu: None, vlm: %s, %s" % (
					control.throttle,
					control.steer,
					control.brake,
					car_data_raw[ego_i]['measurements']["lidar_pose_x"],
					car_data_raw[ego_i]['measurements']["lidar_pose_y"],
					self.dir_inten_veh[ego_id],
					self.speed_inten_veh[ego_id],
				)
			tick_data[ego_i]["comm"] = self.print_comm
			meta_infos[2] += ", target point: %.2f, %.2f" % (batch_data['target'][ego_i][0], batch_data['target'][ego_i][1])
			if str(car_data_raw[ego_i]['measurements']["command"]) in commands_dict:
				meta_infos[2] += ", command: %s" % (commands_dict[str(car_data_raw[ego_i]['measurements']["command"])])
			else:
				meta_infos[2] += ", command: %s" % ("undefined")
			tick_data[ego_i]["meta_infos"] = meta_infos
			tick_data[ego_i]["mes"] = "speed: %.2f" % car_data_raw[ego_i]['measurements']["speed"]
			tick_data[ego_i]["time"] = "time: %.3f" % timestamp

			if step % self.skip_frames == 0:
				tick_data[ego_i]["lidar_3d"] = batch_data["lidar_3d"]
				tick_data[ego_i]["lidar_bev"] = batch_data["lidar_bev"]
				surface = self._hic.run_interface(tick_data[ego_i])
				tick_data[ego_i]["surface"] = surface
		
		if SAVE_PATH is not None:
			self.save(tick_data, step, ego_id)
		
		return control_all


	def save(self, tick_data, frame, ego_id):
		if frame % self.skip_frames != 0:
			return
		for ego_i in range(1): # self.ego_vehicles_num
			folder_path = self.save_path / pathlib.Path("ego_vehicle_{}".format(ego_id))
			if not os.path.exists(folder_path):
				os.mkdir(folder_path)
			Image.fromarray(tick_data[ego_i]["surface"]).save(
				folder_path / ("%04d.jpg" % frame)
			)
			results_to_write = {}
			results_to_write.update(tick_data[ego_i]['planning'])
			with open(folder_path / ("%04d.json" % frame), 'w') as f:
				json.dump(results_to_write, f, indent=4)
			# Per-frame world-frame perception predictions, parity with
			# codriving's pnp_infer_action_e2e_v2v.py:1003. Lets
			# carla_openloop_prototype recompute AP@50 against live CARLA
			# actor GT without re-running inference.
			_per = getattr(self, '_last_perception_for_save', {}).get(ego_id)
			if _per is not None:
				try:
					np.savez(
						str(folder_path / ("%04d_pred.npz" % frame)),
						pred_box_world=_per['pred_box_world'],
						pred_score=_per['pred_score'],
						lidar_pose_x=_per['lidar_pose_x'],
						lidar_pose_y=_per['lidar_pose_y'],
						theta=_per['theta'],
						ego_x=_per['ego_x'],
						ego_y=_per['ego_y'],
					)
				except Exception:
					pass
		return


	def generate_last_info(self, measurements_last):
		# print(measurements_last.keys())
		ego_theta = measurements_last["theta"]
		ego_x = measurements_last["gps_x"]
		ego_y = measurements_last["gps_y"]

		egp_pos_last = {
			"x": ego_y,
			"y": -ego_x,
			"theta": ego_theta
		}

		R = np.array(
			[
				[np.cos(ego_theta), -np.sin(ego_theta)],
				[np.sin(ego_theta), np.cos(ego_theta)],
			]
		)

		ego_last = {
			'egp_pos_last': egp_pos_last,
			'ego_x': ego_x,
			'ego_y': ego_y,
			'ego_theta': ego_theta,
			'R': R,
			'local_command_point': measurements_last['target_point']
		}
		return ego_last
	

	def reduce_image(self, img, pixel_per_meter=1):
		img_after = block_reduce(img, block_size=(pixel_per_meter, pixel_per_meter), func=np.mean)
		# occ_map: 75, 75
		
		img_after = np.clip(img_after, 0.0, 255.0)
		# TODO: change it into auto calculation
		img_after = torch.from_numpy(img_after[:48*8, 48*4:48*8])

		return img_after

	def check_data(self, raw_data, car=True):
		mask = []
		data = [] # without None
		for i in raw_data:
			if i is not None:
				mask.append(1) # filter the data!
				data.append(self.preprocess_data(copy.deepcopy(i), car=car))
			else:
				mask.append(0)
				data.append(0)
		return data, mask

	def preprocess_data(self, data, car=True):
		output_record = {}
		measurements = data['measurements']
		cmd_one_hot = [0, 0, 0, 0, 0, 0]
		if not car:
			measurements['command'] = -1
			measurements["speed"] = 0
			measurements['target_point'] = np.array([0, 0])
		cmd = measurements['command'] - 1
		if cmd < 0:
			cmd = 3
		cmd_one_hot[cmd] = 1
		cmd_one_hot.append(measurements["speed"])
		mes = np.array(cmd_one_hot)
		mes = torch.from_numpy(mes).cuda().float()

		output_record["measurements"] = mes
		output_record['command'] = cmd

		lidar_pose_x = measurements["lidar_pose_x"]
		lidar_pose_y = measurements["lidar_pose_y"]
		lidar_theta = measurements["theta"] + np.pi
		
		output_record['lidar_pose'] = np.array([-lidar_pose_y, lidar_pose_x, lidar_theta])

		## Calculate the world coordinates of the center point of density map. 
		# The current density map prediction range is 10m left and right, 18m first and 2m after
		detmap_pose_x = measurements['lidar_pose_x'] + self.distance_to_map_center*np.cos(measurements["theta"]-np.pi/2)
		detmap_pose_y = measurements['lidar_pose_y'] + self.distance_to_map_center*np.sin(measurements["theta"]-np.pi/2)
		detmap_theta = measurements["theta"] + np.pi/2
		output_record['detmap_pose'] = np.array([-detmap_pose_y, detmap_pose_x, detmap_theta])
		output_record["target_point"] = torch.from_numpy(measurements['target_point']).cuda().float()
		
		##########
		## load and pre-process LiDAR from 3D point cloud to 2D map
		##########
		lidar_unprocessed = data['lidar'][:, :3]
		lidar_unprocessed[:, 1] *= -1
		if not car:
			lidar_unprocessed[:, 2] = lidar_unprocessed[:, 2] + np.array([measurements["lidar_pose_z"]])[np.newaxis, :] - np.array([2.1])[np.newaxis, :] 
		
		lidar_processed = self.lidar_to_histogram_features(
			lidar_unprocessed, crop=self.input_lidar_size, lidar_range=self.lidar_range
		)		
		output_record["lidar_original"] = lidar_processed

		return output_record


	def collect_actor_data_with_visibility(self, measurements, lidar_data):
		lidar_data = lidar_data[:, :3]
		lidar_data[:, 1] *= -1
		actors_data = self.collect_actor_data()
		original_actors_data = copy.deepcopy(actors_data)
		
		ego_x = measurements["lidar_pose_x"]
		ego_y = measurements["lidar_pose_y"]
		ego_z = measurements["lidar_pose_z"]
		ego_theta = measurements["theta"] + np.pi # !note, plus pi in extra.
		# rotate counterclockwise by ego_theta
		R = np.array(
			[
				[np.cos(ego_theta), -np.sin(ego_theta)],
				[np.sin(ego_theta), np.cos(ego_theta)],
			]
		)

		for _id in actors_data.keys():
			raw_loc = actors_data[_id]['loc'][:2]
			new_loc = R.T.dot(np.array([raw_loc[0] - ego_x , raw_loc[1] - ego_y]))
			new_loc[1] = -new_loc[1]
			actors_data[_id]['loc'][:2] = np.array(new_loc)
			actors_data[_id]['loc'][2] -= (actors_data[_id]['box'][2] + ego_z)
			raw_ori = actors_data[_id]['ori'][:2]
			new_ori = R.T.dot(np.array([raw_ori[0], raw_ori[1]]))
			actors_data[_id]['ori'][:2] = np.array(new_ori)
		
		boxes_corner = [] # pose and orientation of the box,
				# (x, y, z, scale_x, scale_y, scale_z, yaw)
		id_map = {}
		count = 0
		for _id in actors_data.keys():
			cur_data = actors_data[_id]
			yaw = get_yaw_angle(cur_data['ori'][:2])
			cur_data['loc'][2] += cur_data['box'][2]
			boxes_corner.append(cur_data['loc']+ [i*2 for i in cur_data['box']] + [yaw])
			id_map[count] = _id
			count += 1
		boxes_corner = np.array(boxes_corner)   

		corners = boxes_to_corners_3d(boxes_corner, order='lwh')

		lidar_visible = []
		for N in range(boxes_corner.shape[0]):
			if actors_data[id_map[N]]['tpe']==2:
				original_actors_data[id_map[N]]['lidar_visible'] = 0
				original_actors_data[id_map[N]]['camera_visible'] = 0
				continue
			num_lidar_points = get_points_in_rotated_box_3d(lidar_data, corners[N])
			if len(num_lidar_points)>8:
				original_actors_data[id_map[N]]['lidar_visible'] = 1
				original_actors_data[id_map[N]]['camera_visible'] = 0
				lidar_visible += [1]
			else:
				original_actors_data[id_map[N]]['lidar_visible'] = 0
				original_actors_data[id_map[N]]['camera_visible'] = 0
				lidar_visible += [0]
		return original_actors_data

	def collect_actor_data(self, ego_id):
		data = {}
		vehicles = CarlaDataProvider.get_world().get_actors().filter("*vehicle*")
		for actor in vehicles:
			loc = actor.get_location()
			if loc.z<-1:
				continue
			_id = actor.id
			_vehicle = CarlaDataProvider.get_hero_actor(hero_id=ego_id)
			if _id == _vehicle.id:
				continue
			data[_id] = {}
			data[_id]["loc"] = [loc.x, loc.y, loc.z]
			ori = actor.get_transform().rotation.get_forward_vector()
			data[_id]["ori"] = [ori.x, ori.y, ori.z]
			box = actor.bounding_box.extent
			data[_id]["box"] = [box.x, box.y, box.z]
			vel = actor.get_velocity()
			data[_id]["vel"] = [vel.x, vel.y, vel.z]
			if actor.type_id=="vehicle.diamondback.century":
				data[_id]["tpe"] = 3
			else:
				data[_id]["tpe"] = 0
		
		walkers = CarlaDataProvider.get_world().get_actors().filter("*walker*")
		for actor in walkers:
			loc = actor.get_location()
			if loc.z<-1:
				continue
			_id = actor.id
			data[_id] = {}
			data[_id]["loc"] = [loc.x, loc.y, loc.z]
			ori = actor.get_transform().rotation.get_forward_vector()
			data[_id]["ori"] = [ori.x, ori.y, ori.z]
			try:
				box = actor.bounding_box.extent
				data[_id]["box"] = [box.x, box.y, box.z]
			except:
				del data[_id]
				continue
			try:
				vel = actor.get_velocity()
				data[_id]["vel"] = [vel.x, vel.y, vel.z]
			except:
				data[_id]["vel"] = [0, 0, 0]
			data[_id]["tpe"] = 1

		
		return data

	def _should_brake(self, command=None):
		actors = CarlaDataProvider.get_world().get_actors()
		self._map = CarlaDataProvider.get_world().get_map()

		vehicle_list = actors.filter("*vehicle*")
		vehicle_list = list(vehicle_list)
		walker_list = actors.filter("*walker*")
		walker_list = list(walker_list)

		vehicle = self._is_vehicle_hazard(vehicle_list, command)  # actors.filter("*vehicle*")
		walker = self._is_walker_hazard(walker_list) # actors.filter("*walker*")
		bike = self._is_bike_hazard(vehicle_list)
		# record the reason for braking
		self.is_vehicle_present = [x.id for x in vehicle]
		self.is_pedestrian_present = [x.id for x in walker]
		self.is_bike_present = [x.id for x in bike]

		self.is_junction = self._map.get_waypoint(
			self._vehicle.get_location()
		).is_junction

		return any(
			len(x) > 0
			for x in [
				vehicle,
				bike,
				walker,
			]
		)

	def _is_actor_affected_by_stop(self, actor, stop, multi_step=20):
		"""
		Check if the given actor is affected by the stop
		"""
		affected = False
		# first we run a fast coarse test
		current_location = actor.get_location()
		stop_location = stop.get_transform().location
		if stop_location.distance(current_location) > self.PROXIMITY_THRESHOLD:
			return affected

		stop_t = stop.get_transform()
		transformed_tv = stop_t.transform(stop.trigger_volume.location)

		# slower and accurate test based on waypoint's horizon and geometric test
		list_locations = [current_location]
		waypoint = self._map.get_waypoint(current_location)
		for _ in range(multi_step):
			if waypoint:
				waypoint = waypoint.next(self.WAYPOINT_STEP)[0]
				if not waypoint:
					break
				list_locations.append(waypoint.transform.location)

		for actor_location in list_locations:
			if self._point_inside_boundingbox(
				actor_location, transformed_tv, stop.trigger_volume.extent
			):
				affected = True

		return affected

	def _is_junction_vehicle_hazard(self, vehicle_list, command):
		res = []
		o1 = _orientation(self._vehicle.get_transform().rotation.yaw)
		x1 = self._vehicle.bounding_box.extent.x
		p1 = (
			self._vehicle.get_location()
			+ x1 * self._vehicle.get_transform().get_forward_vector()
		)
		w1 = self._map.get_waypoint(p1)
		s1 = np.linalg.norm(_numpy(self._vehicle.get_velocity()))
		if command == RoadOption.RIGHT:
			shift_angle = 25
		elif command == RoadOption.LEFT:
			shift_angle = -25
		else:
			shift_angle = 0
		v1 = (4 * s1 + 5) * _orientation(
			self._vehicle.get_transform().rotation.yaw + shift_angle
		)

		for target_vehicle in vehicle_list:
			if target_vehicle.id == self._vehicle.id:
				continue

			o2 = _orientation(target_vehicle.get_transform().rotation.yaw)
			o2_left = _orientation(target_vehicle.get_transform().rotation.yaw - 15)
			o2_right = _orientation(target_vehicle.get_transform().rotation.yaw + 15)
			x2 = target_vehicle.bounding_box.extent.x

			p2 = target_vehicle.get_location()
			p2_hat = p2 - (x2 + 2) * target_vehicle.get_transform().get_forward_vector()
			w2 = self._map.get_waypoint(p2)
			s2 = np.linalg.norm(_numpy(target_vehicle.get_velocity()))

			v2 = (4 * s2 + 2 * x2 + 6) * o2
			v2_left = (4 * s2 + 2 * x2 + 6) * o2_left
			v2_right = (4 * s2 + 2 * x2 + 6) * o2_right

			angle_between_heading = np.degrees(np.arccos(np.clip(o1.dot(o2), -1, 1)))

			if self._vehicle.get_location().distance(p2) > 20:
				continue
			if w1.is_junction == False and w2.is_junction == False:
				continue
			if angle_between_heading < 15.0 or angle_between_heading > 165:
				continue
			collides, collision_point = get_collision(
				_numpy(p1), v1, _numpy(p2_hat), v2
			)
			if collides is None:
				collides, collision_point = get_collision(
					_numpy(p1), v1, _numpy(p2_hat), v2_left
				)
			if collides is None:
				collides, collision_point = get_collision(
					_numpy(p1), v1, _numpy(p2_hat), v2_right
				)

			light = self._find_closest_valid_traffic_light(
				target_vehicle.get_location(), min_dis=10
			)
			if (
				light is not None
				and (self._vehicle.get_traffic_light_state()
				== carla.libcarla.TrafficLightState.Yellow or self._vehicle.get_traffic_light_state()
				== carla.libcarla.TrafficLightState.Red)
			):
				continue
			if collides:
				res.append(target_vehicle)
		return res

	def _is_lane_vehicle_hazard(self, vehicle_list, command):
		res = []
		if (
			command != RoadOption.CHANGELANELEFT
			and command != RoadOption.CHANGELANERIGHT
		):
			return []

		z = self._vehicle.get_location().z
		w1 = self._map.get_waypoint(self._vehicle.get_location())
		o1 = _orientation(self._vehicle.get_transform().rotation.yaw)
		p1 = self._vehicle.get_location()

		yaw_w1 = w1.transform.rotation.yaw
		lane_width = w1.lane_width
		location_w1 = w1.transform.location

		lft_shift = 0.5
		rgt_shift = 0.5
		if command == RoadOption.CHANGELANELEFT:
			rgt_shift += 1
		else:
			lft_shift += 1

		lft_lane_wp = self.rotate_point(
			carla.Vector3D(lft_shift * lane_width, 0.0, location_w1.z), yaw_w1 + 90
		)
		lft_lane_wp = location_w1 + carla.Location(lft_lane_wp)
		rgt_lane_wp = self.rotate_point(
			carla.Vector3D(rgt_shift * lane_width, 0.0, location_w1.z), yaw_w1 - 90
		)
		rgt_lane_wp = location_w1 + carla.Location(rgt_lane_wp)

		for target_vehicle in vehicle_list:
			if target_vehicle.id == self._vehicle.id:
				continue

			w2 = self._map.get_waypoint(target_vehicle.get_location())
			o2 = _orientation(target_vehicle.get_transform().rotation.yaw)
			p2 = target_vehicle.get_location()
			x2 = target_vehicle.bounding_box.extent.x
			p2_hat = p2 - target_vehicle.get_transform().get_forward_vector() * x2 * 2
			s2 = (
				target_vehicle.get_velocity()
				+ target_vehicle.get_transform().get_forward_vector() * x2
			)
			s2_value = max(
				12,
				2
				+ 2 * x2
				+ 3.0 * np.linalg.norm(_numpy(target_vehicle.get_velocity())),
			)

			distance = p1.distance(p2)

			if distance > s2_value:
				continue
			if w1.road_id != w2.road_id or w1.lane_id * w2.lane_id < 0:
				continue
			if command == RoadOption.CHANGELANELEFT:
				if w1.lane_id > 0:
					if w2.lane_id != w1.lane_id - 1:
						continue
				if w1.lane_id < 0:
					if w2.lane_id != w1.lane_id + 1:
						continue
			if command == RoadOption.CHANGELANERIGHT:
				if w1.lane_id > 0:
					if w2.lane_id != w1.lane_id + 1:
						continue
				if w1.lane_id < 0:
					if w2.lane_id != w1.lane_id - 1:
						continue

			if self._are_vehicles_crossing_future(p2_hat, s2, lft_lane_wp, rgt_lane_wp):
				res.append(target_vehicle)
		return res

	def _are_vehicles_crossing_future(self, p1, s1, lft_lane, rgt_lane):
		p1_hat = carla.Location(x=p1.x + 3 * s1.x, y=p1.y + 3 * s1.y)
		line1 = shapely.geometry.LineString([(p1.x, p1.y), (p1_hat.x, p1_hat.y)])
		line2 = shapely.geometry.LineString(
			[(lft_lane.x, lft_lane.y), (rgt_lane.x, rgt_lane.y)]
		)
		inter = line1.intersection(line2)
		return not inter.is_empty

	def _is_stop_sign_hazard(self, stop_sign_list):
		res = []
		if self._affected_by_stop[self.vehicle_num]:
			if not self._stop_completed[self.vehicle_num]:
				current_speed = self._get_forward_speed()
				if current_speed < self.SPEED_THRESHOLD:
					self._stop_completed[self.vehicle_num] = True
					return res
				else:
					return [self._target_stop_sign[self.vehicle_num]]
			else:
				# reset if the ego vehicle is outside the influence of the current stop sign
				if not self._is_actor_affected_by_stop(
					self._vehicle, self._target_stop_sign[self.vehicle_num]
				):
					self._affected_by_stop[self.vehicle_num] = False
					self._stop_completed[self.vehicle_num] = False
					self._target_stop_sign[self.vehicle_num] = None
				return res

		ve_tra = self._vehicle.get_transform()
		ve_dir = ve_tra.get_forward_vector()

		wp = self._map.get_waypoint(ve_tra.location)
		wp_dir = wp.transform.get_forward_vector()

		dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

		if dot_ve_wp > 0:  # Ignore all when going in a wrong lane
			for stop_sign in stop_sign_list:
				if self._is_actor_affected_by_stop(self._vehicle, stop_sign):
					# this stop sign is affecting the vehicle
					self._affected_by_stop[self.vehicle_num] = True
					self._target_stop_sign[self.vehicle_num] = stop_sign
					res.append(self._target_stop_sign[self.vehicle_num])

		return res

	def _is_light_red(self, lights_list):
		if (
			self._vehicle.get_traffic_light_state()
			== carla.libcarla.TrafficLightState.Yellow or self._vehicle.get_traffic_light_state()
			== carla.libcarla.TrafficLightState.Red
		):
			affecting = self._vehicle.get_traffic_light()

			for light in self._traffic_lights:
				if light.id == affecting.id:
					return [light]

		light = self._find_closest_valid_traffic_light(
			self._vehicle.get_location(), min_dis=8
		)
		if light is not None and (self._vehicle.get_traffic_light_state()
			== carla.libcarla.TrafficLightState.Yellow or self._vehicle.get_traffic_light_state()
			== carla.libcarla.TrafficLightState.Red):
			return [light]
		return []

	def _find_closest_valid_traffic_light(self, loc, min_dis):
		wp = self._map.get_waypoint(loc)
		min_wp = None
		min_distance = min_dis
		for waypoint in self._list_traffic_waypoints:
			if waypoint.road_id != wp.road_id or waypoint.lane_id * wp.lane_id < 0:
				continue
			dis = loc.distance(waypoint.transform.location)
			if dis <= min_distance:
				min_distance = dis
				min_wp = waypoint
		if min_wp is None:
			return None
		else:
			return self._dict_traffic_lights[min_wp][0]


	def _is_walker_hazard(self, walkers_list):
		res = []
		p1 = _numpy(self._vehicle.get_location())
		v1 = 10.0 * _orientation(self._vehicle.get_transform().rotation.yaw)

		for walker in walkers_list:
			v2_hat = _orientation(walker.get_transform().rotation.yaw)
			s2 = np.linalg.norm(_numpy(walker.get_velocity()))

			if s2 < 0.05:
				v2_hat *= s2

			p2 = -3.0 * v2_hat + _numpy(walker.get_location())
			v2 = 8.0 * v2_hat

			collides, collision_point = get_collision(p1, v1, p2, v2)

			if collides:
				res.append(walker)

		return res

	def _is_bike_hazard(self, bikes_list):
		res = []
		o1 = _orientation(self._vehicle.get_transform().rotation.yaw)
		v1_hat = o1
		p1 = _numpy(self._vehicle.get_location())
		v1 = 10.0 * o1

		for bike in bikes_list:
			o2 = _orientation(bike.get_transform().rotation.yaw)
			s2 = np.linalg.norm(_numpy(bike.get_velocity()))
			v2_hat = o2
			p2 = _numpy(bike.get_location())

			p2_p1 = p2 - p1
			distance = np.linalg.norm(p2_p1)
			p2_p1_hat = p2_p1 / (distance + 1e-4)

			angle_to_car = np.degrees(np.arccos(np.clip(v1_hat.dot(p2_p1_hat), -1, 1)))
			angle_between_heading = np.degrees(np.arccos(np.clip(o1.dot(o2), -1, 1)))

			# to consider -ve angles too
			angle_to_car = min(angle_to_car, 360.0 - angle_to_car)
			angle_between_heading = min(
				angle_between_heading, 360.0 - angle_between_heading
			)
			if distance > 20:
				continue
			if angle_to_car > 30:
				continue
			if angle_between_heading < 80 and angle_between_heading > 100:
				continue

			p2_hat = -2.0 * v2_hat + _numpy(bike.get_location())
			v2 = 7.0 * v2_hat

			collides, collision_point = get_collision(p1, v1, p2_hat, v2)

			if collides:
				res.append(bike)

		return res

	def _is_vehicle_hazard(self, vehicle_list, command=None):
		res = []
		z = self._vehicle.get_location().z

		o1 = _orientation(self._vehicle.get_transform().rotation.yaw)
		p1 = _numpy(self._vehicle.get_location())
		s1 = max(
			10, 3.0 * np.linalg.norm(_numpy(self._vehicle.get_velocity()))
		)  # increases the threshold distance
		s1a = np.linalg.norm(_numpy(self._vehicle.get_velocity()))
		w1 = self._map.get_waypoint(self._vehicle.get_location())
		v1_hat = o1
		v1 = s1 * v1_hat

		for target_vehicle in vehicle_list:
			if target_vehicle.id == self._vehicle.id:
				continue
			if not target_vehicle.is_alive:
				continue

			o2 = _orientation(target_vehicle.get_transform().rotation.yaw)
			p2 = _numpy(target_vehicle.get_location())
			s2 = max(5.0, 2.0 * np.linalg.norm(_numpy(target_vehicle.get_velocity())))
			s2a = np.linalg.norm(_numpy(target_vehicle.get_velocity()))
			w2 = self._map.get_waypoint(target_vehicle.get_location())
			v2_hat = o2
			v2 = s2 * v2_hat

			p2_p1 = p2 - p1
			distance = np.linalg.norm(p2_p1)
			p2_p1_hat = p2_p1 / (distance + 1e-4)

			angle_to_car = np.degrees(np.arccos(np.clip(v1_hat.dot(p2_p1_hat), -1, 1)))
			angle_between_heading = np.degrees(np.arccos(np.clip(o1.dot(o2), -1, 1)))

			# to consider -ve angles too
			angle_to_car = min(angle_to_car, 360.0 - angle_to_car)
			angle_between_heading = min(
				angle_between_heading, 360.0 - angle_between_heading
			)

			if (
				not w2.is_junction
				and angle_between_heading > 45.0
				and s2a < 0.5
				and distance > 4
			):
				if w1.road_id != w2.road_id:
					continue
			if (angle_between_heading < 15
				and w1.road_id == w2.road_id
				and w1.lane_id != w2.lane_id
				and command != RoadOption.CHANGELANELEFT
				and command != RoadOption.CHANGELANERIGHT
			):
				continue

			if angle_between_heading > 60.0 and not (
				angle_to_car < 15 and distance < s1
			):
				continue
			elif angle_to_car > 30.0:
				continue
			elif distance > s1:
				continue

			res.append(target_vehicle)

		return res


	# load data
	def _load_image(self, path):
		try:
			img = Image.open(self.root_path + path)
		except Exception as e:
			print('[Error] Can not find the IMAGE path.')
			n = path[-8:-4]
			new_path = path[:-8] + "%04d.jpg" % (int(n) - 1)
			img = Image.open(self.root_path + new_path)
		return img
	
	def _load_json(self, path):
		try:
			json_value = json.load(open(self.root_path + path))
		except Exception as e:
			print('[Error] Can not find the JSON path.')
			n = path[-9:-5]
			new_path = path[:-9] + "%04d.json" % (int(n) - 1)
			json_value = json.load(open(self.root_path + new_path))
		return json_value

	def _load_npy(self, path):
		try:
			array = np.load(self.root_path + path, allow_pickle=True)
		except Exception as e:
			print('[Error] Can not find the NPY path.')
			n = path[-8:-4]
			new_path = path[:-8] + "%04d.npy" % (int(n) - 1)
			array = np.load(self.root_path + new_path, allow_pickle=True)
		return array
	

	def lidar_to_histogram_features(self, lidar, crop=256, lidar_range=[28,28,28,28]):
		"""
		Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
		"""

		def splat_points(point_cloud):
			# 256 x 256 grid
			pixels_per_meter = 4
			hist_max_per_pixel = 5
			# x_meters_max = 28
			# y_meters_max = 28
			xbins = np.linspace(
				- lidar_range[3],
				lidar_range[2],
				(lidar_range[2]+lidar_range[3])* pixels_per_meter + 1,
			)
			ybins = np.linspace(-lidar_range[0], lidar_range[1], (lidar_range[0]+lidar_range[1]) * pixels_per_meter + 1)
			hist = np.histogramdd(point_cloud[..., :2], bins=(xbins, ybins))[0]
			hist[hist > hist_max_per_pixel] = hist_max_per_pixel
			overhead_splat = hist / hist_max_per_pixel
			return overhead_splat

		below = lidar[lidar[..., 2] <= -1.45]
		above = lidar[lidar[..., 2] > -1.45]
		below_features = splat_points(below)
		above_features = splat_points(above)
		total_features = below_features + above_features
		features = np.stack([below_features, above_features, total_features], axis=-1)
		features = np.transpose(features, (2, 0, 1)).astype(np.float32)
		return features

	def collate_batch_infer_perception(self, car_data: list, rsu_data: list) -> dict:
		'''
		Re-collate a batch
		'''

		output_dict = {
			"lidar_pose": [],
			"voxel_features": [],
			"voxel_num_points": [],
			"voxel_coords": [],

			"lidar_original": [],

			"detmap_pose": [],

			"record_len": [],
		
			"target": [],
		}
		
		count = 0
		for j in range(len(car_data)):
			output_dict["record_len"].append(len(car_data)+len(rsu_data))
			output_dict["target"].append(car_data[j]['target_point'].unsqueeze(0).float())

			# Set j-th car as the ego-car.
			output_dict["lidar_original"].append(torch.from_numpy(car_data[j]['lidar_original']).unsqueeze(0))
					
			output_dict["lidar_pose"].append(torch.from_numpy(car_data[j]['lidar_pose']).unsqueeze(0).cuda().float())
			output_dict["detmap_pose"].append(torch.from_numpy(car_data[j]['detmap_pose']).unsqueeze(0).cuda().float())
			count += 1
			for i in range(len(car_data)):
				if i==j:
					continue
				output_dict["lidar_original"].append(torch.from_numpy(car_data[i]['lidar_original']).unsqueeze(0))						
				output_dict["lidar_pose"].append(torch.from_numpy(car_data[i]['lidar_pose']).unsqueeze(0).cuda().float())
				output_dict["detmap_pose"].append(torch.from_numpy(car_data[i]['detmap_pose']).unsqueeze(0).cuda().float())
				count += 1
			for i in range(len(rsu_data)):
				output_dict["lidar_original"].append(torch.from_numpy(rsu_data[i]['lidar_original']).unsqueeze(0))						
				output_dict["lidar_pose"].append(torch.from_numpy(rsu_data[i]['lidar_pose']).unsqueeze(0).cuda().float())
				output_dict["detmap_pose"].append(torch.from_numpy(rsu_data[i]['detmap_pose']).unsqueeze(0).cuda().float())
				count += 1
		for key in ["target", "lidar_pose", "detmap_pose" , "lidar_original"]:  # 
			output_dict[key] = torch.cat(output_dict[key], dim=0)
		
		output_dict["record_len"] = torch.from_numpy(np.array(output_dict["record_len"]))

		return output_dict

from matplotlib import pyplot as plt
import numpy as np
import copy
import torch

from opencood.tools.inference_utils import get_cav_box
import opencood.visualization.simple_plot3d.canvas_3d as canvas_3d
import opencood.visualization.simple_plot3d.canvas_bev as canvas_bev

def visualize(infer_result, pcd, pc_range, save_path, method='3d', left_hand=False, measurements=None, lidar_len=None, predicted_waypoints=None, target_point=None, waypoints_num=10):
		"""
		Visualize the prediction, ground truth with point cloud together.
		They may be flipped in y axis. Since carla is left hand coordinate, while kitti is right hand.

		Parameters
		----------
		infer_result:
			pred_box_tensor : torch.Tensor
				(N, 8, 3) prediction.

			gt_tensor : torch.Tensor
				(N, 8, 3) groundtruth bbx
			
			uncertainty_tensor : optional, torch.Tensor
				(N, ?)

			lidar_agent_record: optional, torch.Tensor
				(N_agnet, )


		pcd : torch.Tensor
			PointCloud, (N, 4).

		pc_range : list
			[xmin, ymin, zmin, xmax, ymax, zmax]

		save_path : str
			Save the visualization results to given path.

		dataset : BaseDataset
			opencood dataset object.

		method: str, 'bev' or '3d'

		"""

		if method in ["first_person","bev"]:
			plt.figure()
		else:
			plt.figure(figsize=[(pc_range[3]-pc_range[0])/10, (pc_range[4]-pc_range[1])/10])

		pc_range = [int(i) for i in pc_range]

		if isinstance(pcd, list):
			pcd_np = pcd[0].cpu().numpy()
		else:
			pcd_np = pcd.cpu().numpy()

		attrib_list = ['pred_box_tensor', 'pred_score', 'gt_box_tensor', 'score_tensor']
		for attrib in attrib_list:
			if attrib in infer_result:
				if isinstance(infer_result[attrib], list):
					infer_result_tensor = []
					for i in range(len(infer_result[attrib])):
						if infer_result[attrib][i] is not None:
							infer_result_tensor.append(infer_result[attrib][i])
					if len(infer_result_tensor)>0:
						infer_result[attrib] = torch.cat(infer_result_tensor, dim=0)
					else:
						infer_result[attrib] = None

		pred_box_tensor = infer_result.get("pred_box_tensor", None)
		gt_box_tensor = infer_result.get("gt_box_tensor", None)

		if pred_box_tensor is not None:
			pred_box_np = pred_box_tensor.cpu().numpy()

			score = infer_result.get("score_tensor", None)
			if score is not None:
				score_np = score.cpu().numpy()

			uncertainty = infer_result.get("uncertainty_tensor", None)
			if uncertainty is not None:
				uncertainty_np = uncertainty.cpu().numpy()
				uncertainty_np = np.exp(uncertainty_np)
				d_a_square = 1.6**2 + 3.9**2
				
				if uncertainty_np.shape[1] == 3:
					uncertainty_np[:,:2] *= d_a_square
					uncertainty_np = np.sqrt(uncertainty_np) 
					# yaw angle is in radian, it's the same in g2o SE2's setting.

				elif uncertainty_np.shape[1] == 2:
					uncertainty_np[:,:2] *= d_a_square
					uncertainty_np = np.sqrt(uncertainty_np) # yaw angle is in radian

				elif uncertainty_np.shape[1] == 7:
					uncertainty_np[:,:2] *= d_a_square
					uncertainty_np = np.sqrt(uncertainty_np) # yaw angle is in radian     

		if gt_box_tensor is not None:
			gt_box_np = gt_box_tensor.cpu().numpy()

		mode_list = ['gt','pred','gt+pred']

		filte_array = None
		if pred_box_tensor is not None:
			if len(pred_box_tensor) > 0:
				pred_box_clone = pred_box_tensor.clone()
				tmp = pred_box_clone[:,:,0].clone()
				pred_box_clone[:,:,0]=pred_box_clone[:,:,1]
				pred_box_clone[:,:,1] = tmp
				filte_array = pred_box_clone.cpu().numpy()
		
		ego_list = []

		if filte_array is not None:
			for i in range(filte_array.shape[0]):
				transformed_box = transform_2d_points(
						filte_array[i],
						np.pi/2 - measurements["theta"], # car1_to_world parameters
						measurements["lidar_pose_y"],
						measurements["lidar_pose_x"],
						np.pi/2 - measurements["theta"], # car2_to_world parameters, note that not world_to_car2
						measurements["y"],
						measurements["x"],
					)
				location_box = np.mean(transformed_box[:4,:2], 0)
				if np.linalg.norm(location_box) < 1.4:
					ego_list.append(i)
		if pred_box_tensor is not None:
			for idx in ego_list:
				pred_box_np[idx] = 0


		canvas_list = []
		
		if method == 'bev':
			for mode in ['gt+pred']:
				canvas = canvas_bev.Canvas_BEV_heading_right(canvas_shape=(3000//2,3000//2),
												canvas_x_range=(pc_range[1], pc_range[3]), 
												canvas_y_range=(pc_range[1]-12, pc_range[4]+12),
												left_hand=left_hand) 

				pcd_np[:,1] *= -1
				canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np) # Get Canvas Coords

				colors = np.zeros((pcd_np.shape[0],3))
				green = np.array([215,255,215])
				purple = np.array([200,200,255])
				colors[:,:] = green
				colors[:lidar_len[0],:] = purple

				canvas.draw_canvas_points(canvas_xy[valid_mask],radius=1,colors=colors[valid_mask]) # Only draw valid points		

				# plot waypoints
				wp_3d = torch.zeros((waypoints_num,3))
				wp_3d[:,0] = -predicted_waypoints[0][:,1]
				wp_3d[:,1] = -predicted_waypoints[0][:,0]
				wp_3d[:,2] = 0.3
				canvas_xy_wp, valid_mask_wp = canvas.get_canvas_coords(wp_3d)
				canvas.draw_canvas_points(canvas_xy_wp[valid_mask_wp],radius=6,colors=(255,200,255))

				# plot target points
				tp_3d = torch.zeros((1,3))
				tp_3d[:,0] = -target_point[0][1]
				tp_3d[:,1] = -target_point[0][0]
				tp_3d[:,2] = 0.3
				canvas_xy_tp, valid_mask_tp = canvas.get_canvas_coords(tp_3d)
				canvas.draw_canvas_points(canvas_xy_tp[valid_mask_tp],radius=16,colors=(200,255,255))				

				# plot ego car area
				ego_car_box = np.array([[[2.2-1.3,1,0],
										[2.2-1.3,-1,0],
										[-2.2-1.3,-1,0],
										[-2.2-1.3,1,0],
										[2.2-1.3,1,0],
										[2.2-1.3,-1,0],
										[-2.2-1.3,-1,0],
										[-2.2-1.3,1,0]]])
				canvas.draw_boxes(ego_car_box,colors=(255,255,255),box_line_thickness=6)

				if gt_box_tensor is not None and (mode=='gt' or mode=='gt+pred'):
					gt_box_np[:,:,1] *= -1
					canvas.draw_boxes(gt_box_np,colors=(255,0,0),box_line_thickness=6)  #  , texts=gt_name
				if pred_box_tensor is not None and (mode=='pred' or mode=='gt+pred'):
					pred_box_np[:,:,1] *= -1
					canvas.draw_boxes(pred_box_np, colors=(0,255,0),box_line_thickness=6)  #  , texts=pred_name

				# heterogeneous
				lidar_agent_record = infer_result.get("lidar_agent_record", None)
				cav_box_np = infer_result.get("cav_box_np", None)
				if lidar_agent_record is not None:
					cav_box_np = copy.deepcopy(cav_box_np)
					for i, islidar in enumerate(lidar_agent_record):
						text = ['lidar'] if islidar else ['camera']
						color = (0,191,255) if islidar else (255,185,15)
						canvas.draw_boxes(cav_box_np[i:i+1], colors=color, texts=text,box_line_thickness=6)

				canvas_list.append(canvas)
				
				image_bev = np.transpose(canvas.canvas, axes=(1, 0, 2))
				image_bev = np.flip(image_bev, axis=0)

				return image_bev


		elif method == '3d':
			for mode in mode_list:
				canvas = canvas_3d.Canvas_3D(left_hand=left_hand)
				canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)
				canvas.draw_canvas_points(canvas_xy[valid_mask],radius=5)
				if gt_box_tensor is not None and (mode=='gt' or mode=='gt+pred'):
					canvas.draw_boxes(gt_box_np,colors=(0,255,0)) #  , texts=gt_name
				if pred_box_tensor is not None and (mode=='pred' or mode=='gt+pred'):
					canvas.draw_boxes(pred_box_np, colors=(255,0,0)) #  , texts=pred_name

				# heterogeneous
				lidar_agent_record = infer_result.get("lidar_agent_record", None)
				cav_box_np = infer_result.get("cav_box_np", None)
				if lidar_agent_record is not None:
					cav_box_np = copy.deepcopy(cav_box_np)
					for i, islidar in enumerate(lidar_agent_record):
						text = ['lidar'] if islidar else ['camera']
						color = (0,191,255) if islidar else (255,185,15)
						canvas.draw_boxes(cav_box_np[i:i+1], colors=color, texts=text,box_line_thickness=6)

				canvas_list.append(canvas)

		elif method == 'first_person':
			canvas = canvas_3d.Canvas_3D(left_hand=left_hand,canvas_shape=(3000//2, 4500//2)) # camera_center_coords=(0, 0, 2.3 + 1.9396926), camera_focus_coords=(0 + 4.9396926, 0, 2.3),
			pcd_np[:,1] *= -1

			colors = np.zeros((pcd_np.shape[0],3))
			green = np.array([215,255,215])
			purple = np.array([200,200,255])
			colors[:,:] = green
			colors[:lidar_len[0],:] = purple

			canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)
			canvas.draw_canvas_points(canvas_xy[valid_mask],radius=1,colors=colors[valid_mask])

			# plot waypoints
			wp_3d = torch.zeros((waypoints_num,3))
			wp_3d[:,0] = -predicted_waypoints[0][:,1]
			wp_3d[:,1] = -predicted_waypoints[0][:,0]
			wp_3d[:,2] = 0.3
			canvas_xy_wp, valid_mask_wp = canvas.get_canvas_coords(wp_3d)
			canvas.draw_canvas_points(canvas_xy_wp[valid_mask_wp],radius=6,colors=(255,200,255))

			# plot target points
			tp_3d = torch.zeros((1,3))
			tp_3d[:,0] = -target_point[0][1]
			tp_3d[:,1] = -target_point[0][0]
			tp_3d[:,2] = 0.3
			canvas_xy_tp, valid_mask_tp = canvas.get_canvas_coords(tp_3d)
			canvas.draw_canvas_points(canvas_xy_tp[valid_mask_tp],radius=16,colors=(200,255,255))		

			# plot ego car area
			ego_car_box = np.array([[2.2-1.3,1,0],
									[2.2-1.3,-1,0],
									[-2.2-1.3,-1,0],
									[-2.2-1.3,1,0],
									[2.2-1.3,1,0],
									[2.2-1.3,-1,0],
									[-2.2-1.3,-1,0],
									[-2.2-1.3,1,0]])
			canvas.draw_boxes(ego_car_box,colors=(255,255,255),box_line_thickness=6)

			if gt_box_tensor is not None :
				gt_box_np[:,:,1] *= -1
				canvas.draw_boxes(gt_box_np,colors=(255,0,0),box_line_thickness=6) #  , texts=gt_name
			if pred_box_tensor is not None :
				pred_box_np[:,:,1] *= -1
				canvas.draw_boxes(pred_box_np, colors=(0,255,0),box_line_thickness=6) #  , texts=pred_name

			# heterogeneous
			lidar_agent_record = infer_result.get("lidar_agent_record", None)
			cav_box_np = infer_result.get("cav_box_np", None)
			if lidar_agent_record is not None:
				cav_box_np = copy.deepcopy(cav_box_np)
				for i, islidar in enumerate(lidar_agent_record):
					text = ['lidar'] if islidar else ['camera']
					color = (0,191,255) if islidar else (255,185,15)
					canvas.draw_boxes(cav_box_np[i:i+1], colors=color, texts=text,box_line_thickness=6)

			return canvas.canvas
		else:
			raise(f"Not Completed for f{method} visualization.")


		ax1 = plt.subplot(221)
		ax1.imshow(canvas_list[0].canvas)
		ax1.set_title('GT', fontsize= 15)
		ax1.axis('off')  

		ax2 = plt.subplot(222)
		ax2.imshow(canvas_list[1].canvas)
		ax2.set_title('Preds', fontsize= 15)
		ax2.axis('off')  

		ax3 = plt.subplot(212)
		ax3.imshow(canvas_list[2].canvas)
		ax3.set_title('GT with pred', fontsize= 15)
		ax3.axis('off')  

		plt.tight_layout()
		plt.savefig(save_path, transparent=False, dpi=500)
		plt.clf()
		plt.close()
