import imp
import os
import pathlib
import time
import torch
torch.backends.cudnn.benchmark = True
import math
import cv2
import carla
import numpy as np
from PIL import Image
import yaml
from typing import OrderedDict, Optional, List

from codriving import CODRIVING_REGISTRY
from codriving.models.model_decoration import decorate_model

from team_code.utils.carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions
from team_code.planner_pnp import RoutePlanner
import team_code.utils.yaml_utils as yaml_utils


class CoLMRoutePlanner(RoutePlanner):
    """RoutePlanner variant scoped to the CoLMDriver model.

    Replaces the parent's single-segment safe_ahead projection with a
    polyline-based cumulative lookahead. When the safe_ahead guard
    fires, walks the route polyline forward from route[1] accumulating
    segment distances until reaching `lookahead_meters` ahead of the
    ego, returning that point. Handles dense waypoint clusters
    (e.g., on-ramp merge transitions with 0.5m spacing) and curves
    cleanly because the target stays on the actual route polyline
    rather than projected through a tiny local segment direction.

    Default `lookahead_meters` is 6 m (in distribution for the model).
    Default trigger threshold is 5 m. Tunable per-process via env
    `COLM_POLYLINE_LOOKAHEAD` and `COLM_SAFE_AHEAD_TRIGGER`.
    """

    def __init__(self, *args, lookahead_meters=6.0,
                 trigger_threshold=5.0, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            env_v = os.environ.get('COLM_POLYLINE_LOOKAHEAD')
            if env_v is not None:
                lookahead_meters = float(env_v)
        except Exception:
            pass
        try:
            env_t = os.environ.get('COLM_SAFE_AHEAD_TRIGGER')
            if env_t is not None:
                trigger_threshold = float(env_t)
        except Exception:
            pass
        self._lookahead_meters = float(lookahead_meters)
        self._trigger_threshold = float(trigger_threshold)

    def _safe_ahead_trigger_threshold(self):
        return self._trigger_threshold

    def _compute_safe_ahead_target(self, vehicle_num, gps, seg_dir, safe_ahead):
        if self._lookahead_meters <= safe_ahead:
            return super()._compute_safe_ahead_target(
                vehicle_num, gps, seg_dir, safe_ahead)
        route = self.route[vehicle_num]
        if len(route) < 2:
            return super()._compute_safe_ahead_target(
                vehicle_num, gps, seg_dir, safe_ahead)
        gps_arr = np.array(gps, dtype=float)
        route1 = np.array(route[1][0], dtype=float)
        dist_to_route1 = float(np.dot(route1 - gps_arr, seg_dir))
        if self._lookahead_meters <= dist_to_route1:
            return gps_arr + seg_dir * self._lookahead_meters
        distance_to_walk = self._lookahead_meters - dist_to_route1
        prev_pt = route1
        cum = 0.0
        for i in range(2, len(route)):
            wp = np.array(route[i][0], dtype=float)
            seg_v = wp - prev_pt
            seg_l = float(np.linalg.norm(seg_v))
            if seg_l < 1e-6:
                continue
            if cum + seg_l >= distance_to_walk:
                seg_unit = seg_v / seg_l
                return prev_pt + seg_unit * (distance_to_walk - cum)
            cum += seg_l
            prev_pt = wp
        last_seg_dir = None
        if len(route) >= 2:
            last_a = np.array(route[-2][0], dtype=float)
            last_b = np.array(route[-1][0], dtype=float)
            seg_v = last_b - last_a
            seg_l = float(np.linalg.norm(seg_v))
            if seg_l > 1e-6:
                last_seg_dir = seg_v / seg_l
        if last_seg_dir is None:
            last_seg_dir = seg_dir
        remaining_walk = distance_to_walk - cum
        return np.array(route[-1][0], dtype=float) + last_seg_dir * remaining_walk


from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from leaderboard.sensors.fixed_sensors import RoadSideUnit, get_rsu_point, InfraLiDARUnit
from leaderboard.autoagents import autonomous_agent
from team_code.infra_lidar_config import INFRA_LIDAR_POSES, is_infra_collab_enabled, write_diag_line as _infra_diag

from opencood.tools import train_utils, inference_utils
from opencood.tools.train_utils import create_model as create_perception_model
from opencood.tools.train_utils import load_saved_model as load_perception_model
from opencood.data_utils.datasets import build_dataset

from common.torch_helper import load_checkpoint as load_planning_model_checkpoint
from common.registry import build_object_within_registry_from_config as build_planning_model
from common.io import load_config_from_yaml


def get_entry_point():
    return "PnP_Agent"

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
    VIEW_WIDTH = int(sensor['width'])
    VIEW_HEIGHT = int(sensor['height'])
    VIEW_FOV = int(float(sensor['fov']))
    # VIEW_WIDTH = int(sensor.attributes['image_size_x'])
    # VIEW_HEIGHT = int(sensor.attributes['image_size_y'])
    # VIEW_FOV = int(float(sensor.attributes['fov']))

    matrix_k = np.identity(3)
    matrix_k[0, 2] = VIEW_WIDTH / 2.0
    matrix_k[1, 2] = VIEW_HEIGHT / 2.0
    matrix_k[0, 0] = matrix_k[1, 1] = VIEW_WIDTH / \
        (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))

    return matrix_k.tolist()

def get_camera_extrinsic(cur_extrin, ref_extrin):
    """
    Args:
        cur_extrin (carla.Transform): current extrinsic
        ref_extrin (carla.Transform): reference extrinsic
    Returns:
        extrin (list): 4x4 extrinsic matrix with respect to reference coordinate system 
    """
    extrin = np.array(ref_extrin.get_inverse_matrix()) @ np.array(cur_extrin.get_matrix())
    return extrin.tolist()


#### agents
class PnP_Agent(autonomous_agent.AutonomousAgent):
    """
    Navigated by a pipline with perception then prediction, from sensor data to control signal
    """
    def setup(self, path_to_conf_file, ego_vehicles_num):
    
        self.agent_name = "pnp"
        self.wall_start = time.time()
        self.initialized = False
        self._rgb_sensor_data = {"width": 3000, "height": 1500, "fov": 100}
        self._sensor_data = self._rgb_sensor_data.copy()
        self.ego_vehicles_num = ego_vehicles_num   

        # load agent config
        self.config = yaml.load(open(path_to_conf_file),Loader=yaml.Loader)
        
        # Override LLM/VLM ports from environment variables if set
        import os
        if 'COLMDRIVER_LLM_PORT' in os.environ:
            llm_port = os.environ['COLMDRIVER_LLM_PORT']
            self.config['comm_server']['comm_client'] = f'http://127.0.0.1:{llm_port}/v1'
            print(f'[CoLMDriver] Using LLM port {llm_port} from environment')
        if 'COLMDRIVER_VLM_PORT' in os.environ:
            vlm_port = os.environ['COLMDRIVER_VLM_PORT']
            self.config['comm_server']['vlm_client'] = f'http://127.0.0.1:{vlm_port}/v1'
            print(f'[CoLMDriver] Using VLM port {vlm_port} from environment')
        
        self.use_semantic = self.config['perception'].get('use_semantic', False)
        print('if use semantic?', self.use_semantic)
        control_cfg = self.config.get('control', {})
        self.force_global_route = control_cfg.get('force_global_route', True)

        import importlib
        module_name = self.config['action_name']
        module = importlib.import_module(module_name)
        PnP_infer = getattr(module, 'PnP_infer')
        
        # load perception model and dataloader
        perception_hypes = yaml_utils.load_yaml(self.config['perception']['perception_model_dir'])
        self.config['perception']['perception_hypes'] = perception_hypes
        # self.config.perception_hypes = perception_hypes
        
        # pose noise
        # np.random.seed(30330)
        # torch.manual_seed(10000)
        if 'pose_error' in self.config['perception']:
            noise_setting = OrderedDict()
            noise_args = {'pos_std': self.config['perception']['pose_error']['pos_std'],
                            'rot_std': self.config['perception']['pose_error']['rot_std'],
                            'pos_mean': self.config['perception']['pose_error']['pos_mean'],
                            'rot_mean': self.config['perception']['pose_error']['rot_mean']}

            noise_setting['add_noise'] = True
            noise_setting['args'] = noise_args

            if self.config['perception']['pose_error'].get('use_laplace',False):
                noise_setting['args']['laplace'] = True

            print(f"Noise Added: {noise_args['pos_std']}/{noise_args['rot_std']}/{noise_args['pos_mean']}/{noise_args['rot_mean']}.")
            perception_hypes.update({"noise_setting": noise_setting})

        perception_dataloader = build_dataset(perception_hypes, visualize=True, train=False)
        print('Creating perception Model')
        self.perception_model = create_perception_model(perception_hypes)
        print('Loading perception Model from checkpoint')
        resume_epoch, self.perception_model = load_perception_model(self.config['perception']['perception_model_dir'], self.perception_model)
        print(f"resume from {resume_epoch} epoch.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.perception_model.to(device)
        self.perception_model.eval()

        # load planning model
        self.waypoints_num = self.config.get("planning", {}).get("waypoints_num", 20)
        planner_config = load_config_from_yaml(self.config['planning']['planner_config'])
        planning_model_config = planner_config['model']
        print('Creating planning Model')
        planning_model = build_planning_model(
            CODRIVING_REGISTRY,
            planning_model_config,
        )
        print('Loading planning Model from checkpoint')
        load_planning_model_checkpoint(self.config['planning']['planner_model_checkpoint'], device, planning_model)
        model_decoration_config = planner_config['model_decoration']
        decorate_model(planning_model, **model_decoration_config)
        planning_model.to(device)
        planning_model.eval()
        self.planning_model = planning_model

        # core module, infer the action from sensor data
        self.infer = PnP_infer(config=self.config,
                            ego_vehicles_num=self.ego_vehicles_num,
                            perception_model=self.perception_model,
                            planning_model=planning_model,
                            perception_dataloader=perception_dataloader,
                            device=device)

        # Open-loop AP / visible-actor / GT-replay hook. Inert in closed-loop
        # runs (auto-detects --openloop env signal); when active it
        # accumulates per-tick TP/FP against live CARLA actors and writes
        # ``perception_swap_ap_ego{N}.json`` next to SAVE_PATH at destroy().
        # Uses colmdriver's perception_hypes lidar range so AP's GT filter
        # matches what the perception model actually saw.
        try:
            from team_code.openloop_helpers import (
                OpenLoopAPHook, BEVRenderer, resolve_openloop_ap_eval_range,
            )
            cav_lidar_range = perception_hypes.get(
                "preprocess", {}
            ).get("cav_lidar_range")
            if cav_lidar_range is None:
                def _find_cav_lidar(d):
                    if isinstance(d, dict):
                        if "cav_lidar_range" in d:
                            return d["cav_lidar_range"]
                        for v in d.values():
                            r = _find_cav_lidar(v)
                            if r is not None:
                                return r
                    return None
                cav_lidar_range = _find_cav_lidar(perception_hypes)
            # Detector label: distinguish colmdriver vs colmdriver_rulebase
            # by the action_name (the importable module that was loaded for
            # PnP_infer). This shows up in the AP json so a sweep across
            # both variants doesn't conflate them.
            _det_label = "colmdriver"
            _action_name = str(self.config.get("action_name", ""))
            if "rulebase" in _action_name:
                _det_label = "colmdriver_rulebase"
            ap_eval_range = resolve_openloop_ap_eval_range(cav_lidar_range)
            self._oloop = OpenLoopAPHook(
                ego_vehicles_num=self.ego_vehicles_num,
                detector_label=_det_label,
                lidar_range=ap_eval_range,
                iou_thresholds=(0.3, 0.5, 0.7),
            )
            if self._oloop.enabled:
                print(
                    f"[{_det_label}] open-loop AP hook enabled "
                    f"(model_lidar_range={cav_lidar_range}, "
                    f"ap_eval_range={ap_eval_range}, "
                    f"save_root={self._oloop.ap_save_root})"
                )
            self._bev = BEVRenderer(detector_label=_det_label) if self._oloop.enabled else None
        except Exception as exc:  # pragma: no cover — never break setup
            print(f"[colmdriver] open-loop AP hook init failed: {exc}")
            self._oloop = None
            self._bev = None

        # Pre-init the infra-LiDAR list state so _ensure_infra_units() can
        # safely fire from setup() before _init() runs.
        self.infra_units = []
        self._infra_spawn_attempted = False
        self._infra_spawn_step = None
        # Spawn infra LiDARs NOW (before agent_wrapper.setup_sensors → world.tick)
        # so they're included in the first tick that primes ego sensors.
        # Idempotent — _init()/spawn_rsu() may also call this and it's a no-op.
        try:
            self._ensure_infra_units()
        except Exception as _exc:
            print(f"[infra-collab] setup-time spawn skipped: {_exc}")


    def _init(self):
        """
        initialization before the first step of simulation
        """
        self._route_planner = CoLMRoutePlanner(2.0, 10.0)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True
        self.first_generate_rsu = True
        self.vehicle_num = 0
        self.step = -1
        # NOTE: infra_units / _infra_spawn_attempted / _infra_spawn_step are
        # initialised in setup() (so _ensure_infra_units can fire there before
        # the leaderboard's first world.tick). Don't reset them here — setup()
        # may have already populated them.

        print('Set BEV producers')
        self.birdview_producer = BirdViewProducer(
            CarlaDataProvider.get_client(),  # carla.Client
            target_size=PixelDimensions(width=400, height=400),
            pixels_per_meter=5,
            crop_type=BirdViewCropType.FRONT_AND_REAR_AREA,
        )

    def _get_position(self, tick_data):
        # GPS coordinate!
        gps = tick_data["gps"]
        gps = (gps - self._route_planner.mean) * self._route_planner.scale
        return gps
    
    def get_save_path(self):
        return  self.infer.save_path   
    
    def pose_def(self):
        """
        location of sensor related to vehicle
        """
        self.lidar_pose = carla.Transform(carla.Location(x=1.3,y=0.0,z=1.85),
                                        carla.Rotation(roll=0.0,pitch=0.0,yaw=-90.0))
        self.camera_front_pose = carla.Transform(carla.Location(x=1.3,y=0.0,z=2.3),
                                        carla.Rotation(roll=0.0,pitch=0.0,yaw=0.0))
        self.camera_rear_pose = carla.Transform(carla.Location(x=-1.3,y=0.0,z=2.3),
                                        carla.Rotation(roll=0.0,pitch=0.0,yaw=180.0))
        self.camera_left_pose = carla.Transform(carla.Location(x=1.3,y=0.0,z=2.3),
                                        carla.Rotation(roll=0.0,pitch=0.0,yaw=-60.0))
        self.camera_right_pose = carla.Transform(carla.Location(x=1.3,y=0.0,z=2.3),
                                        carla.Rotation(roll=0.0,pitch=0.0,yaw=60.0))
        return

    def sensors(self):
        """
        Return the sensor list.
        """
        self.pose_def()
        sensors_list = [
            {
                "type": "sensor.camera.rgb",
                "x": self.camera_front_pose.location.x,
                "y": self.camera_front_pose.location.y,
                "z": self.camera_front_pose.location.z,
                "roll": self.camera_front_pose.rotation.roll,
                "pitch": self.camera_front_pose.rotation.pitch,
                "yaw": self.camera_front_pose.rotation.yaw,
                "width": self._rgb_sensor_data["width"],
                "height": self._rgb_sensor_data["height"],
                "fov": self._rgb_sensor_data["fov"],
                "id": "rgb_front",
            },
            {
                "type": "sensor.camera.rgb",
                "x": self.camera_rear_pose.location.x,
                "y": self.camera_rear_pose.location.y,
                "z": self.camera_rear_pose.location.z,
                "roll": self.camera_rear_pose.rotation.roll,
                "pitch": self.camera_rear_pose.rotation.pitch,
                "yaw": self.camera_rear_pose.rotation.yaw,
                "width": self._rgb_sensor_data["width"],
                "height": self._rgb_sensor_data["height"],
                "fov": self._rgb_sensor_data["fov"],
                "id": "rgb_rear",
            },
            {
                "type": "sensor.camera.rgb",
                "x": self.camera_left_pose.location.x,
                "y": self.camera_left_pose.location.y,
                "z": self.camera_left_pose.location.z,
                "roll": self.camera_left_pose.rotation.roll,
                "pitch": self.camera_left_pose.rotation.pitch,
                "yaw": self.camera_left_pose.rotation.yaw,
                "width": self._rgb_sensor_data["width"],
                "height": self._rgb_sensor_data["height"],
                "fov": self._rgb_sensor_data["fov"],
                "id": "rgb_left",
            },
            {
                "type": "sensor.camera.rgb",
                "x": self.camera_right_pose.location.x,
                "y": self.camera_right_pose.location.y,
                "z": self.camera_right_pose.location.z,
                "roll": self.camera_right_pose.rotation.roll,
                "pitch": self.camera_right_pose.rotation.pitch,
                "yaw": self.camera_right_pose.rotation.yaw,
                "width": self._rgb_sensor_data["width"],
                "height": self._rgb_sensor_data["height"],
                "fov": self._rgb_sensor_data["fov"],
                "id": "rgb_right",
            },
            {
                "type": "sensor.lidar.ray_cast",
                "x": self.lidar_pose.location.x,
                "y": self.lidar_pose.location.y,
                "z": self.lidar_pose.location.z,
                "roll": self.lidar_pose.rotation.roll,
                "pitch": self.lidar_pose.rotation.pitch,
                "yaw": self.lidar_pose.rotation.yaw,
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
                # "sensor_tick": 0.05,
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
                # "sensor_tick": 0.01,
                "id": "gps",
            },
            {"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"},
        ]
        return sensors_list

    def tick(self, input_data: dict, vehicle_num: int):
        """
        Pre-load raw sensor data
        """
        _vehicle = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)

        # bev drivable area image
        bev_image = Image.fromarray(BirdViewProducer.as_rgb(
            self.birdview_producer.produce(agent_vehicle=_vehicle, actor_exist=False)
        )) #  (400, 400, 3), Image
        img_c = bev_image.crop([140, 20, 260, 260])
        img_r = np.array(img_c.resize((96, 192))) # (96, 192, 3)
        drivable_area = np.where(img_r.sum(axis=2)>200, 1, 0) # size (192, 96), only 0/1.
        
        # rgb camera data
        rgb_front = cv2.cvtColor(
            input_data["rgb_front_{}".format(vehicle_num)][1][:, :, :3], cv2.COLOR_BGR2RGB
        )
        rgb_left = cv2.cvtColor(
            input_data["rgb_left_{}".format(vehicle_num)][1][:, :, :3], cv2.COLOR_BGR2RGB
        )
        rgb_right = cv2.cvtColor(
            input_data["rgb_right_{}".format(vehicle_num)][1][:, :, :3], cv2.COLOR_BGR2RGB
        )
        rgb_rear = cv2.cvtColor(
            input_data["rgb_rear_{}".format(vehicle_num)][1][:, :, :3], cv2.COLOR_BGR2RGB
        )

        # lidar sensor data
        lidar = input_data["lidar_{}".format(vehicle_num)][1]

        # measurements
        gps = input_data["gps_{}".format(vehicle_num)][1][:2]
        speed = input_data["speed_{}".format(vehicle_num)][1]["move_state"]["speed"]
        compass = input_data["imu_{}".format(vehicle_num)][1][-1]
        if (
            math.isnan(compass) == True
        ):  # It can happen that the compass sends nan for a few frames
            compass = 0.0

        # compute lidar pose in world coordinate
        cur_lidar_pose = carla.Location(x=self.lidar_pose.location.x,
                                            y=self.lidar_pose.location.y,
                                            z=self.lidar_pose.location.z)
        _vehicle.get_transform().transform(cur_lidar_pose)

        # compute camera intrinsic and extrinsic params
        self.intrinsics = get_camera_intrinsic(self._rgb_sensor_data)
        self.lidar2front = get_camera_extrinsic(self.camera_front_pose, self.lidar_pose)
        self.lidar2left = get_camera_extrinsic(self.camera_left_pose, self.lidar_pose)
        self.lidar2right = get_camera_extrinsic(self.camera_right_pose, self.lidar_pose)
        self.lidar2rear = get_camera_extrinsic(self.camera_rear_pose, self.lidar_pose)

        # target waypoint produced by global planner
        pos = self._get_position({"gps": gps})
        next_wp, next_cmd = self._route_planner.run_step(pos, vehicle_num)
        theta = compass + np.pi / 2
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        local_command_point = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])
        local_command_point = R.T.dot(local_command_point)
        local_command_point = np.clip(local_command_point, a_min=[-self.config['perception']['detection_range'][1], -self.config['perception']['detection_range'][0]],
                                                                     a_max=[self.config['perception']['detection_range'][2], self.config['perception']['detection_range'][3]])
        route_local = self._project_route_to_local(vehicle_num, pos, compass)

        # record measurements
        mes = {
            "gps_x": pos[0],
            "gps_y": pos[1],
            "x": pos[1],
            "y": -pos[0],
            "theta": compass,
            "lidar_pose_x": cur_lidar_pose.x,
            "lidar_pose_y": cur_lidar_pose.y,
            "lidar_pose_z": cur_lidar_pose.z,
            "lidar_pose_gps_x": -cur_lidar_pose.y,
            "lidar_pose_gps_y": cur_lidar_pose.x,
            "camera_front_intrinsics": self.intrinsics,
            "camera_front_extrinsics": self.lidar2front,
            "camera_left_intrinsics": self.intrinsics,
            "camera_left_extrinsics": self.lidar2left,
            "camera_right_intrinsics": self.intrinsics,
            "camera_right_extrinsics": self.lidar2right,
            "camera_rear_intrinsics": self.intrinsics,
            "camera_rear_extrinsics": self.lidar2rear,
            "speed": speed,
            "compass": compass,
            "command": next_cmd.value,
            "target_point": local_command_point,
            "route_local": route_local,
        }

        # return pre-loaded sensor data
        result = {
            "rgb_front": rgb_front,
            "rgb_left": rgb_left,
            "rgb_right": rgb_right,
            "rgb_rear": rgb_rear,
            "lidar": lidar,
            "measurements": mes,
            "bev": bev_image,
            "drivable_area": drivable_area
        }
        return result

    def _project_route_to_local(self, vehicle_num: int, pos: np.ndarray, compass: float) -> Optional[List[List[float]]]:
        if not hasattr(self, "_route_planner"):
            return None
        future = self._route_planner.get_future_waypoints(vehicle_num, num=self.config['planning']['waypoints_num'] + 5)
        if not future:
            return None
        theta = compass + np.pi / 2.0
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        local_points: list[list[float]] = []
        for waypoint in future:
            wp = np.array(waypoint[:2], dtype=float)
            local = rot.T.dot(wp - pos)
            local_points.append(local.tolist())
        if not local_points:
            return None
        if len(local_points) < self.waypoints_num:
            local_points.extend([local_points[-1]] * (self.waypoints_num - len(local_points)))
        else:
            local_points = local_points[: self.waypoints_num]
        return local_points

    def setup_infra_sensors(self):
        """Public hook called by agent_wrapper.setup_sensors() right before
        its priming world.tick() — guarantees infra LiDARs are spawned and
        their first frame is published at the same tick that primes ego
        sensors, so step 0's read sees infra data (no spawn-step gate needed).
        """
        self._ensure_infra_units()

    def _ensure_infra_units(self):
        """Lazy one-shot spawn of stationary infrastructure LiDARs for v2xpnp.

        No-op unless COLMDRIVER_INFRA_COLLAB=1 (set by run_custom_eval's
        --infra-collab) AND the loaded CARLA map is ucla_v2 (the only map
        these world-frame poses are valid for). Idempotent.

        Called twice: first from setup() (so the spawn happens BEFORE the
        leaderboard's first world.tick() that primes ego sensors — and the
        infra LiDAR queue therefore gets its first frame at the same tick),
        and again from spawn_rsu() in case setup() couldn't reach the world.
        """
        if self._infra_spawn_attempted:
            return
        if not is_infra_collab_enabled():
            self._infra_spawn_attempted = True  # known no-op, never retry
            return
        try:
            world = CarlaDataProvider.get_world()
            if world is None:
                return  # world not set yet — retry on next call
            map_name = world.get_map().name
        except Exception:
            return  # transient — retry on next call
        if "ucla_v2" not in map_name.lower():
            print(f"[infra-collab] map '{map_name}' is not ucla_v2 — skipping infra LiDAR spawn")
            self._infra_spawn_attempted = True
            return
        self._infra_spawn_attempted = True
        import carla as _carla
        carla_map = world.get_map()
        for i, p in enumerate(INFRA_LIDAR_POSES):
            # See pnp_agent_e2e_v2v.py for rationale: configured z is height
            # above ground; CARLA ucla_v2 ground is ~10.7m, so spawn at
            # ground_z + p.z (avoids 6-8m underground sensors).
            ground_z = float(p.z)
            try:
                wp = carla_map.get_waypoint(
                    _carla.Location(x=p.x, y=p.y, z=0.0),
                    project_to_road=True,
                    lane_type=_carla.LaneType.Any,
                )
                if wp is not None:
                    ground_z = float(wp.transform.location.z) + float(p.z)
            except Exception:
                pass
            tf = _carla.Transform(
                _carla.Location(x=p.x, y=p.y, z=ground_z),
                _carla.Rotation(roll=p.roll, pitch=p.pitch, yaw=p.yaw),
            )
            unit = InfraLiDARUnit(
                world_transform=tf,
                save_path=None,
                id=900000 + i,
                use_semantic=False,
            )
            unit.setup_sensors()
            self.infra_units.append(unit)
            print(f"[infra-collab] spawned {p.name} at "
                  f"({p.x:.2f}, {p.y:.2f}, {ground_z:.2f}) yaw={p.yaw:.2f}° "
                  f"(config_z_above_ground={p.z:.2f}m)")
        # If spawn happened in setup() (before _init set self.step), record -1
        # so the read-gate in _collect_infra_data lets the very first run_step
        # read through. The leaderboard's agent_wrapper.setup_sensors → world.tick
        # has by that point already fired the infra LiDAR callback, so the
        # SensorInterface queue has data.
        self._infra_spawn_step = int(getattr(self, "step", -1))

    def _collect_infra_data(self):
        """Return list of infra-LiDAR rsu_data dicts for this tick (or [])."""
        if not self.infra_units:
            return []
        # Don't read on the same step we just spawned the units — the LiDAR
        # callback can't have fired yet (CARLA only generates sensor data
        # during world.tick(), which runs BETWEEN run_step calls).
        # Reading now would block 100s on SensorInterface.get_data().
        if self._infra_spawn_step is not None and int(self.step) <= self._infra_spawn_step:
            return []
        out = []
        for unit in self.infra_units:
            try:
                d = unit.process(unit.tick(), is_train=True)
            except Exception as exc:
                print(f"[infra-collab] process() failed for {unit.name}_{unit.id}: {exc}")
                d = None
            if d is not None:
                out.append(d)
        return out

    def spawn_rsu(self):
        """
        spawn rsu and delete old ones
        """
        # Spawn stationary infra-LiDAR units once, regardless of rotation cadence.
        self._ensure_infra_units()
        if self.step % self.config['RSU']['change_rsu_frame'] == 0:
            if self.first_generate_rsu:
                self.first_generate_rsu = False
            else:
                # destroy previous rsu sensor except the first entry
                [self.rsu[vehicle_num].cleanup() for vehicle_num in range(self.ego_vehicles_num)]
                self.rsu=[]
            # handle the initialization issue
            for vehicle_num in range(self.ego_vehicles_num):
                # generate rsu
                vehicle = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)
                # spawn rsu to capture data without saving
                self.rsu.append(RoadSideUnit(save_path=None,
                                            id=int((self.step/self.config['RSU']['change_rsu_frame']+1)*1000+vehicle_num),
                                            is_None=(vehicle is None),
                                            use_semantic=self.use_semantic))
                if vehicle is not None:
                    spawn_loc = get_rsu_point(vehicle, height=self.config['RSU']['rsu_height'], lane_side=self.config['RSU']['rsu_lane_side'], distance=self.config['RSU']['rsu_distance'])
                    self.rsu[vehicle_num].setup_sensors(parent=None, spawn_point=spawn_loc)
        return
    
    @torch.no_grad()
    def run_step(self, input_data: dict, timestamp):
        """
        Execute one step of navigation.
        Args:
            input_data: raw sensor data from ego vehicle
        Returns: 
            control_all: control_all[i] represents control signal of the ith vehicle
        """

        # initialization before starting
        if not self.initialized:
            self._init()
        self.step += 1

        # ── First-tick stub save ──────────────────────────────────────────
        # Guarantees a minimal 0000.json + 0000_pred.npz exists for every
        # ego BEFORE inference runs. Required for short / failed / zero-route
        # scenarios where inference never reaches the proper save() call in
        # colmdriver_action.py:save(). The proper save overwrites these stubs
        # on later ticks, so successful runs are unchanged.
        if self.step == 0:
            try:
                self._write_tick_stubs(timestamp)
            except Exception as _e:
                print(f"[colmdriver-stub] tick0 stub save failed: {_e}", flush=True)

        # spawn the rsu.
        self.spawn_rsu()
        rsu_data = []

        # If the frame is skipped.
        force_fresh_every_frame = os.environ.get('OPENLOOP_FORCE_FRESH_INFERENCE', '0') == '1'
        if (
            (not force_fresh_every_frame)
            and self.step % self.config['simulation']['skip_frames'] != 0
            and self.step > self.config['simulation']['skip_frames']
        ):
            # return the previous control signal.   
            return self.infer.prev_control

        # capture a list of sensor data from rsu
        if self.step % self.config['RSU']['change_rsu_frame'] != 0:
            for vehicle_num in range(self.ego_vehicles_num):
                if CarlaDataProvider.get_hero_actor(hero_id=vehicle_num) is not None:
                    rsu_data.append(
                        self.rsu[vehicle_num].process(self.rsu[vehicle_num].tick(),is_train=True)
                    )
                else:
                    rsu_data.append(None)

        # Append stationary infrastructure LiDAR streams (v2xpnp ablation).
        # When --infra-collab is off or the map isn't ucla_v2, infra_units is
        # empty and this is a no-op. The fusion path treats these identically
        # to regular RSUs (same dict shape, same voxelizer, same model batch).
        rsu_data.extend(self._collect_infra_data())

        # capture a list of sensor data from ego vehicles
        # Sensor-key presence guard mirrors pnp_agent_e2e_v2v.run_step: after
        # TERM_DECISION rc=100% fires, ScenarioManager still issues one more
        # run_step before stopping, by which time the per-ego cameras have
        # been destroyed. A bare input_data["rgb_front_N"] then KeyErrors and
        # the leaderboard prints "agent has crashed: 'rgb_front_N'", which
        # flips the run from SUCCESS to FAILURE (RouteCompletionTest=0.01%).
        # Treat "sensors gone" the same as "actor gone": yield None.
        ego_data = [
            self.tick(input_data, vehicle_num)
            if (
                CarlaDataProvider.get_hero_actor(hero_id=vehicle_num) is not None
                and "rgb_front_{}".format(vehicle_num) in input_data
            )
            else None
            for vehicle_num in range(self.ego_vehicles_num)
        ]
        if all(e is None for e in ego_data):
            prev = getattr(getattr(self, "infer", None), "prev_control", None)
            if prev is not None:
                return prev

        # Per-tick fusion topology diag log: one JSONL record per planner-running
        # tick proves whether infra LiDAR streams actually entered this tick's
        # cooperative batch. Written even when --infra-collab is off so
        # ablation pairs (on vs off) can be diff-compared after the fact.
        _infra_count = len(getattr(self, "infra_units", []) or [])
        _n_rsus_total = len(rsu_data)
        _planner_label = (
            "colmdriver_rulebase"
            if "rulebase" in str(self.config.get("action_name", "")).lower()
            else "colmdriver"
        )
        _infra_diag(
            log_dir=getattr(self.infer, "log_path", None),
            step=int(self.step),
            sim_t=float(timestamp) if timestamp is not None else None,
            planner_label=_planner_label,
            n_egos_total=len(ego_data),
            n_egos_present=sum(1 for e in ego_data if e is not None),
            n_regular_rsus=max(0, _n_rsus_total - _infra_count),
            n_infra_lidars=_infra_count,
            n_cooperators_in_fusion=(sum(1 for e in ego_data if e is not None)
                                     + sum(1 for r in rsu_data if r is not None)),
        )

        control_all = []

        # Tolerate per-tick inference exceptions ("too many indices for
        # tensor of dimension 1" etc) that can happen on scenarios with
        # unusual ego/RSU counts. Without this the AgentError propagates
        # to the leaderboard and terminates the route after 1 tick.
        try:
            control_all = self.infer.get_action_from_list_multi_vehicle(car_data_raw=ego_data,
                                                            rsu_data_raw=rsu_data,
                                                            step=self.step,
                                                            timestamp=timestamp)
        except (IndexError, RuntimeError) as _exc:
            print(f"[colmdriver_agent] inference exception ({type(_exc).__name__}: "
                  f"{_exc}); emitting zero control for this tick")
            import carla as _carla
            control_all = [_carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
                           for _ in range(max(1, self.ego_vehicles_num))]

        # Expose planning model's world-frame waypoints so the open-loop
        # evaluator can use them directly instead of rolling out controls.
        # planning_bank[0]: [N_wps, 2] world-frame (x, y) at 0.1 s intervals.
        self.last_planned_waypoints_world = self.infer.planning_bank.get(0)

        # Open-loop AP collection. No-op in closed-loop runs.
        oloop = getattr(self, "_oloop", None)
        if oloop is not None and oloop.enabled:
            try:
                import math as _math
                per_save = getattr(self.infer, "_last_perception_for_save", {}) or {}
                ego_actor_ids = set()
                for vehicle_num in range(self.ego_vehicles_num):
                    a = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)
                    if a is not None:
                        ego_actor_ids.add(a.id)
                for ego_id, _per in per_save.items():
                    # Prefer pred_box_lidar (raw opencood lidar frame)
                    # over the codriving-specific axis-swapped pred_box_world.
                    # See pnp_agent_e2e_v2v.py for full rationale.
                    pred_lidar = _per.get("pred_box_lidar")
                    pred_score = _per.get("pred_score")
                    if pred_lidar is None or pred_score is None:
                        continue
                    actor = CarlaDataProvider.get_hero_actor(hero_id=int(ego_id))
                    if actor is None:
                        continue
                    tf = actor.get_transform()
                    # Pass LIDAR pose (not actor pose) since pred_box_lidar
                    # is in LIDAR frame. On v2xpnp/ucla_v2 the actor is at
                    # world Z≈10m and the LiDAR mount is +2.5m above, so
                    # using actor pose offsets GT corners by ~12m in Z and
                    # zero-out IoU matches → AP=0. The infer cached the real
                    # CARLA-world LiDAR pose in _last_perception_for_save.
                    _lz = float(_per.get("lidar_pose_z", float(tf.location.z) + 2.5))
                    primary_pose_world = [
                        float(_per.get("lidar_pose_x", tf.location.x)),
                        float(_per.get("lidar_pose_y", tf.location.y)),
                        _lz,
                        float(tf.rotation.roll),
                        float(tf.rotation.yaw),
                        float(tf.rotation.pitch),
                    ]
                    oloop.update_ego(
                        ego_idx=int(ego_id),
                        primary_pose_world=primary_pose_world,
                        pred_corners_ego=pred_lidar,
                        pred_scores=pred_score,
                        ego_actor_ids=ego_actor_ids,
                        sim_t=float(timestamp) if timestamp is not None else 0.0,
                    )
                # Periodic AP snapshot — write next to ``self.infer.save_path``
                # so the post-metrics sweep pairs it via ``run_dir.parent``.
                # Fire on EVERY planner-running tick (cheap, idempotent) so
                # short scenarios that crash before tick 10 still produce
                # an AP file. Without this, --openloop runs that hit
                # collided_terminal at tick ~1 leave AP=- everywhere.
                _infer_save = getattr(self.infer, "save_path", None)
                if _infer_save is not None:
                    try:
                        oloop.save_all(ap_save_root=str(_infer_save))
                    except Exception as _exc:
                        if getattr(self, "_oloop_save_err_count", 0) < 3:
                            self._oloop_save_err_count = getattr(self, "_oloop_save_err_count", 0) + 1
                            print(f"[colmdriver] AP save fail: {_exc}")
            except Exception as exc:
                _ec = getattr(self, "_oloop_err_count", 0) + 1
                self._oloop_err_count = _ec
                if _ec <= 3 or _ec % 100 == 0:
                    print(f"[colmdriver] open-loop AP update fail #{_ec}: {exc}")

        # BEV viz mirrors codriving — same shared BEVRenderer.
        bev = getattr(self, "_bev", None)
        if (oloop is not None and oloop.enabled
                and bev is not None and bev.viz_dir):
            try:
                import math as _math
                import numpy as _np
                per_save = getattr(self.infer, "_last_perception_for_save", {}) or {}
                ego_actor_map = {}
                for vehicle_num in range(self.ego_vehicles_num):
                    a = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)
                    if a is not None:
                        ego_actor_map[vehicle_num] = a
                for ego_id, _per in per_save.items():
                    if int(ego_id) not in ego_actor_map:
                        continue
                    actor = ego_actor_map[int(ego_id)]
                    tf = actor.get_transform()
                    vel = actor.get_velocity()
                    ego_xy_world = (float(tf.location.x), float(tf.location.y))
                    ego_yaw_rad = _math.radians(float(tf.rotation.yaw))
                    ego_speed = float(_math.hypot(vel.x, vel.y))
                    # Build the BEV pred-trajectory in CARLA world frame from
                    # the RAW local-frame model output, matching codriving's
                    # axis-swap convention (forward=-y_local, left=-x_local
                    # rotated by ego yaw). The closed-loop planning_bank uses
                    # a different in-place rotation (compass instead of yaw,
                    # no axis swap) that produces world coords offset from
                    # the actual predicted motion — fine for the closed-loop
                    # planner internals but wrong for the BEV viz / open-loop
                    # ADE evaluation. We don't touch the closed-loop path.
                    last_local = getattr(self.infer, "last_predicted_waypoints_local", {}) or {}
                    local_wps = last_local.get(int(ego_id))
                    traj_world = None
                    if local_wps is not None and len(local_wps) > 0:
                        wp = _np.asarray(local_wps, dtype=_np.float64).reshape(-1, 2)
                        fwd = -wp[:, 1]
                        lft = -wp[:, 0]
                        cy_, sy_ = _math.cos(ego_yaw_rad), _math.sin(ego_yaw_rad)
                        wx = ego_xy_world[0] + cy_ * fwd - sy_ * lft
                        wy = ego_xy_world[1] + sy_ * fwd + cy_ * lft
                        traj_world = _np.stack([wx, wy], axis=1)
                    gt_future, ade_t, fde_t = oloop.gt_future_for_render(
                        int(ego_id),
                        float(timestamp) if timestamp is not None else 0.0,
                        traj_world,
                    )
                    coll = (
                        oloop.collectors[int(ego_id)]
                        if 0 <= int(ego_id) < len(oloop.collectors)
                        else None
                    )
                    running_ap = None
                    if coll is not None and coll.n_frames > 0:
                        try:
                            from team_code.openloop_helpers import _lazy_eval_imports as _lei
                            _, _, calculate_ap, _, _ = _lei()
                            ap_value, _, _ = calculate_ap(coll.result_stat, 0.5)
                            running_ap = float(ap_value)
                        except Exception:
                            running_ap = None
                    # Transform raw opencood-lidar preds → CARLA world for
                    # the BEV pane. See pnp_agent_e2e_v2v.py for rationale.
                    pred_lidar_for_bev = _per.get("pred_box_lidar")
                    pred_world_carla = None
                    if pred_lidar_for_bev is not None and len(pred_lidar_for_bev) > 0:
                        cy_, sy_ = _math.cos(ego_yaw_rad), _math.sin(ego_yaw_rad)
                        pl = _np.asarray(pred_lidar_for_bev, dtype=_np.float64)
                        x_l = pl[..., 0]
                        y_l = pl[..., 1]
                        x_w = ego_xy_world[0] + cy_ * x_l - sy_ * y_l
                        y_w = ego_xy_world[1] + sy_ * x_l + cy_ * y_l
                        pred_world_carla = _np.stack([x_w, y_w, pl[..., 2]], axis=-1)
                    # Reuse the AP collector's last-frame snapshot — same
                    # rationale as codriving / pnp_agent_e2e_v2v: the BEV
                    # pane shows exactly what was scored, no double query.
                    last = oloop.last_frame_state(int(ego_id), iou=0.5)
                    bev.render(
                        tick=int(self.step),
                        ego_idx=int(ego_id),
                        ego_xy=ego_xy_world,
                        ego_yaw_rad=ego_yaw_rad,
                        ego_speed_mps=ego_speed,
                        traj_world=traj_world,
                        gt_future_world=gt_future,
                        pred_corners_world=pred_world_carla,
                        pred_scores=_per.get("pred_score"),
                        gt_corners_world=(last.get("gt_corners_world")
                                          if last is not None else None),
                        ade=ade_t, fde=fde_t,
                        running_ap=running_ap,
                        ap_iou=0.5,
                        ap_eval_range=oloop.lidar_range,
                        n_tp_frame=(last.get("n_tp_frame", 0)
                                     if last is not None else 0),
                        n_fp_frame=(last.get("n_fp_frame", 0)
                                     if last is not None else 0),
                        n_fn_frame=(last.get("n_fn_frame", 0)
                                     if last is not None else 0),
                        pred_status=(last.get("pred_status")
                                      if last is not None else None),
                        gt_matched=(last.get("gt_matched")
                                     if last is not None else None),
                    )
            except Exception as exc:
                _ec = getattr(self, "_bev_err_count", 0) + 1
                self._bev_err_count = _ec
                if _ec <= 3 or _ec % 100 == 0:
                    print(f"[colmdriver] open-loop BEV render fail #{_ec}: {exc}")

        ### return the control signal in list format.
        return control_all

    def _write_tick_stubs(self, timestamp):
        """Write a minimal 0000.json + 0000_pred.npz per ego using the current
        spawn pose + zero waypoints. Lets openloop_post_metrics.py compute ADE
        for short / failed / zero-route scenarios where the proper save call
        in colmdriver_action.py:save() never executes. Real saves on later
        ticks (or successful runs) overwrite these stubs."""
        import json as _json
        import math as _math
        import numpy as _np
        save_path = getattr(self.infer, "save_path", None)
        if save_path is None:
            return
        save_path = pathlib.Path(str(save_path))
        for ego_id in range(self.ego_vehicles_num):
            ego_dir = save_path / f"ego_vehicle_{ego_id}"
            tick_json = ego_dir / "0000.json"
            tick_pred = ego_dir / "0000_pred.npz"
            if tick_json.is_file() and tick_pred.is_file():
                continue  # already saved by a prior pass
            try:
                actor = CarlaDataProvider.get_hero_actor(hero_id=ego_id)
                if actor is None:
                    continue
                tf = actor.get_transform()
                lx = float(tf.location.x); ly = float(tf.location.y)
                theta = _math.radians(float(tf.rotation.yaw))
                yaw_rad = theta - _math.pi / 2.0
            except Exception:
                continue
            ego_dir.mkdir(parents=True, exist_ok=True)
            if not tick_json.is_file():
                payload = {
                    "speed": 0.0,
                    "waypoints": [[0.0, 0.0]] * 10,
                    "lidar_pose_x": lx,
                    "lidar_pose_y": ly,
                    "theta": theta,
                    "yaw_rad": yaw_rad,
                    "_stub": True,
                }
                with open(tick_json, "w") as f:
                    _json.dump(payload, f)
            if not tick_pred.is_file():
                _np.savez(
                    tick_pred,
                    pred_box_world=_np.zeros((0, 8, 3), dtype=_np.float32),
                    pred_score=_np.zeros((0,), dtype=_np.float32),
                    lidar_pose_x=_np.float64(lx),
                    lidar_pose_y=_np.float64(ly),
                    theta=_np.float64(theta),
                    ego_x=_np.float64(lx),
                    ego_y=_np.float64(ly),
                )

    def destroy(self):
        # Final fallback: ensure every ego has at least a stub tick0 saved.
        # Covers zero-route scenarios where run_step never executed (the
        # in-run_step call at step==0 didn't fire) — e.g., route_terminal
        # fired during scenario init.
        try:
            self._write_tick_stubs(None)
        except Exception:
            pass
        # Flush AP json BEFORE deleting the planning models. Write next to
        # ``self.infer.save_path`` for the post-metrics pairing.
        oloop = getattr(self, "_oloop", None)
        if oloop is not None and oloop.enabled:
            try:
                infer_save_path = getattr(self.infer, "save_path", None)
                ap_root = (
                    str(infer_save_path)
                    if infer_save_path is not None else None
                )
                oloop.save_all(ap_save_root=ap_root)
            except Exception as exc:
                print(f"[colmdriver] open-loop AP final save failed: {exc}")
        # Tear down infrastructure LiDAR units before model deletion so they
        # don't outlive the scenario in CARLA. clean_rsu_single() is per-ego
        # only; clean_rsu() isn't externally called, so we must clean infra
        # explicitly here.
        for unit in getattr(self, "infra_units", []) or []:
            try:
                unit.cleanup()
            except Exception:
                pass
        self.infra_units = []
        try:
            del self.perception_model
            del self.planning_model
        except:
            print('delete model failed')

    def clean_rsu_single(self,vehicle_num):
        self.rsu[vehicle_num].cleanup()
        return
      
    def clean_rsu(self):
        [self.clean_rsu_single(vehicle_num) for vehicle_num in range(self.ego_vehicles_num)]
        for unit in getattr(self, "infra_units", []) or []:
            try:
                unit.cleanup()
            except Exception:
                pass
        self.infra_units = []
        return
