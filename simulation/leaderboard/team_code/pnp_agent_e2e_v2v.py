import time
import torch
torch.backends.cudnn.benchmark = True
import math
import cv2
import carla
import numpy as np
from PIL import Image
import pdb
import sys
import yaml
import os

from typing import OrderedDict

from codriving.utils.torch_helper import \
        move_dict_data_to_device, build_dataloader
from common.torch_helper import load_checkpoint as load_planning_model_checkpoint

from team_code.utils.carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions
from team_code.pnp_infer_action_e2e_v2v import PnP_infer
from team_code.planner_pnp import RoutePlanner
import team_code.utils.yaml_utils as yaml_utils

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from leaderboard.sensors.fixed_sensors import RoadSideUnit, get_rsu_point, InfraLiDARUnit
from leaderboard.autoagents import autonomous_agent
from team_code.infra_lidar_config import INFRA_LIDAR_POSES, is_infra_collab_enabled, write_diag_line as _infra_diag

from opencood.tools import train_utils, inference_utils
from opencood.tools.train_utils import create_model as create_perception_model
from opencood.tools.train_utils import load_saved_model as load_perception_model
from opencood.data_utils.datasets import build_dataset

from common.registry import build_object_within_registry_from_config as build_planning_model
from common.io import load_config_from_yaml

from codriving import CODRIVING_REGISTRY
from codriving.models.model_decoration import decorate_model

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
        # These RGB cameras are used for logging/visualization only in the CoDriving stack.
        # Keep them lighter by default to avoid render corruption in multi-ego runs.
        self._rgb_sensor_data = {
            "width": max(1, int(os.environ.get("CODRIVING_RGB_WIDTH", "1600"))),
            "height": max(1, int(os.environ.get("CODRIVING_RGB_HEIGHT", "900"))),
            "fov": 100,
        }
        self._sensor_data = self._rgb_sensor_data.copy()
        self.ego_vehicles_num = ego_vehicles_num   

        # load agent config
        self.config = yaml.load(open(path_to_conf_file),Loader=yaml.Loader)
        self.use_semantic = self.config['perception'].get('use_semantic', False)
        print('if use semantic?', self.use_semantic)
        
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
        if self.config['planning']['core_method'] == 'MotionNet':
            self.infer = PnP_infer(config=self.config,
                                ego_vehicles_num=self.ego_vehicles_num,
                                perception_model=self.perception_model,
                                planning_model=planning_model,
                                perception_dataloader=perception_dataloader,
                                device=device)
        elif self.config['planning']['core_method'] == 'rule_based':
            self.infer = rule_infer(config=self.config,
                                ego_vehicles_num=self.ego_vehicles_num,
                                perception_model=self.perception_model,
                                planning_model=planning_model,
                                perception_dataloader=perception_dataloader,
                                device=device)

        # Open-loop AP / visible-actor / GT-replay hook. Inert in closed-loop
        # runs (auto-detects the --openloop env signal); when active it
        # accumulates per-tick TP/FP against live CARLA actors and writes
        # ``perception_swap_ap_ego{N}.json`` next to SAVE_PATH at destroy().
        # Uses codriving's perception_hypes lidar range so AP's GT filter
        # matches what the perception model actually saw.
        try:
            from team_code.openloop_helpers import (
                OpenLoopAPHook, BEVRenderer, resolve_openloop_ap_eval_range,
            )
            cav_lidar_range = perception_hypes.get(
                "preprocess", {}
            ).get("cav_lidar_range")
            if cav_lidar_range is None:
                # Older configs nest it deeper; walk to find it.
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
            # AP scoring uses a fixed ``140 × 80 m`` crop in openloop mode
            # (env-tunable) — same crop as perception_swap_agent — so detector
            # comparisons aren't biased by each model's native cav_lidar_range.
            ap_eval_range = resolve_openloop_ap_eval_range(cav_lidar_range)
            self._oloop = OpenLoopAPHook(
                ego_vehicles_num=self.ego_vehicles_num,
                detector_label="codriving",
                lidar_range=ap_eval_range,
                iou_thresholds=(0.3, 0.5, 0.7),
            )
            if self._oloop.enabled:
                print(
                    f"[codriving] open-loop AP hook enabled "
                    f"(model_lidar_range={cav_lidar_range}, "
                    f"ap_eval_range={ap_eval_range}, "
                    f"save_root={self._oloop.ap_save_root})"
                )
            # BEV renderer — emits side-by-side front-cam + top-down BEV
            # PNGs every N ticks, only when openloop viz is on. Inert
            # when ``viz_dir`` resolves to empty (closed-loop runs don't
            # have SAVE_PATH set up the same way).
            self._bev = BEVRenderer(detector_label="codriving") if self._oloop.enabled else None
        except Exception as exc:  # pragma: no cover — never break setup
            print(f"[codriving] open-loop AP hook init failed: {exc}")
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
        self._route_planner = RoutePlanner(2.0, 10.0)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True
        self.first_generate_rsu = True
        self.vehicle_num = 0
        self.step = -1
        self.next_action_step=-1
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
        # LiDAR mount (xyz + yaw) is configurable so HEAL-detector variants
        # can match perception_swap's mount exactly — same physical sensor
        # position + orientation → same rays → same point cloud → fair
        # AP comparison. Codriving's own perception model expects the
        # legacy mount (1.3, 0, 1.85, yaw=-90) — its training distribution
        # — so vanilla codriving and codriving_zerofeat use the defaults.
        # HEAL variants (codriving_attfuse/fcooper/disco/cobevt) override
        # via YAML to (0, 0, 2.5, yaw=0). Codriving's perception still
        # runs forward but its output is discarded (BEV feature zeroed,
        # detections replaced with HEAL's), so the off-distribution input
        # is harmless for that path.
        _perc_cfg = (self.config or {}).get('perception', {}) or {}
        _lidar_mount_x = float(_perc_cfg.get('lidar_mount_x', 1.3))
        _lidar_mount_y = float(_perc_cfg.get('lidar_mount_y', 0.0))
        _lidar_mount_z = float(_perc_cfg.get('lidar_mount_z', 1.85))
        _lidar_mount_yaw = float(_perc_cfg.get('lidar_mount_yaw', -90.0))
        self.lidar_pose = carla.Transform(
            carla.Location(x=_lidar_mount_x, y=_lidar_mount_y, z=_lidar_mount_z),
            carla.Rotation(roll=0.0, pitch=0.0, yaw=_lidar_mount_yaw),
        )
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
            "target_point": local_command_point
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

    def setup_infra_sensors(self):
        """Public hook called by agent_wrapper.setup_sensors() before its
        priming world.tick() — guarantees infra has data at step 0.
        """
        self._ensure_infra_units()

    def _ensure_infra_units(self):
        """Lazy one-shot spawn of stationary infrastructure LiDARs for v2xpnp.

        No-op unless COLMDRIVER_INFRA_COLLAB=1 (set by run_custom_eval's
        --infra-collab) AND the loaded CARLA map is ucla_v2 (the only map
        these world-frame poses are valid for). Idempotent — safe to call
        any number of times.

        Called twice: first from setup() (so the spawn happens BEFORE the
        leaderboard's first world.tick() that primes ego sensors — and the
        infra LiDAR queue therefore gets its first frame at the same tick),
        and again from spawn_rsu() in case setup() couldn't reach the world
        (the second call is a no-op if the first succeeded).
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
            # The configured z (4.08 / 2.50 m) is the LiDAR's height above
            # ground in v2x-real-pnp world coords (where ground ≈ 0). At UCLA
            # Westwood in CARLA, ground sits at ~10.7m absolute world Z, so
            # spawning at the literal config Z drops the sensor 6-8m
            # underground — invalid raycasts that corrupt cooperative fusion
            # and confuse the BEV viz. Query CARLA's road waypoint Z at the
            # XY and add the configured Z as a height-above-ground offset.
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
                # id offset chosen to not collide with regular RSU ids
                # (which use (step/change_rsu_frame+1)*1000 + vehicle_num).
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
        """Update virtual RSU positions every change_rsu_frame frames.

        Old behaviour: destroy + respawn the RSU and its 5 sensors per ego
        every rotation. Over a long scenario this fires hundreds of CARLA
        spawn_actor calls, each with a small probability of hitting the
        sensor "streaming_open_race" — empirically the dominant cause of
        sensor_failure (~50% of failed compute on codriving).

        New behaviour: keep RSU sensors alive across rotations and just
        teleport them with set_transform(). Per-tick effective sensor
        poses are identical to the old implementation. Sensors stream
        continuously instead of stopping/starting every 0.25 s.

        Lifecycle correctness — three transitions to handle safely:
          1. First rotation (vehicle present): create RoadSideUnit, call
             setup_sensors. Sensors registered in its SensorInterface.
          2. Subsequent rotation, RSU still healthy: call move_to().
             Sensors stay open, tags stay stable.
          3. Subsequent rotation, RSU's sensors were destroyed externally
             (e.g. leaderboard called cleanup_rsu(idx) when the ego hit
             route_complete mid-scenario): _sensors_list is empty AND the
             SensorInterface may still hold stale tags from the original
             registration. Calling setup_sensors() on the same object
             would re-register existing tags — boom, "Duplicated sensor
             tag". So we REPLACE the RoadSideUnit with a fresh instance
             (which has its own SensorInterface) — matches the original
             code's per-rotation freshness for this edge case.

        The id formula uses a step-dependent component (matching the
        original code) so that even when we re-create a RoadSideUnit, the
        new tag namespace doesn't collide with any orphaned registrations
        in the agent's main SensorInterface (defensive — should not be
        needed since each RoadSideUnit owns its own SensorInterface, but
        keeps tags grep-friendly across rotations).
        """
        # Spawn stationary infra-LiDAR units once, regardless of rotation cadence.
        self._ensure_infra_units()

        if self.step % self.config['RSU']['change_rsu_frame'] != 0:
            return

        cfg = self.config['RSU']
        if self.first_generate_rsu:
            # Transition #1: first rotation — fresh per-ego RSU spawn.
            self.first_generate_rsu = False
            for vehicle_num in range(self.ego_vehicles_num):
                vehicle = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)
                self.rsu.append(RoadSideUnit(
                    save_path=None,
                    id=int((self.step / cfg['change_rsu_frame'] + 1) * 1000 + vehicle_num),
                    is_None=(vehicle is None),
                    use_semantic=self.use_semantic,
                ))
                if vehicle is not None:
                    spawn_loc = get_rsu_point(
                        vehicle,
                        height=cfg['rsu_height'],
                        lane_side=cfg['rsu_lane_side'],
                        distance=cfg['rsu_distance'],
                    )
                    self.rsu[vehicle_num].setup_sensors(parent=None, spawn_point=spawn_loc)
            return

        # Subsequent rotations.
        for vehicle_num in range(self.ego_vehicles_num):
            vehicle = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)
            if vehicle is None:
                # Ego despawned. Do not touch its RSU; preserve any state
                # for if/when the ego comes back. If the ego stays
                # despawned, the unused RSU just sits idle at its last
                # location until scenario teardown.
                continue
            new_loc = get_rsu_point(
                vehicle,
                height=cfg['rsu_height'],
                lane_side=cfg['rsu_lane_side'],
                distance=cfg['rsu_distance'],
            )
            rsu = self.rsu[vehicle_num]
            if rsu._sensors_list:
                # Transition #2: healthy RSU — just move it.
                rsu.move_to(new_loc)
            else:
                # Transition #3: sensors gone (placeholder OR
                # cleanup_rsu fired externally). Replace the RoadSideUnit
                # with a fresh instance. The old RoadSideUnit (and its
                # stale SensorInterface entries, if any) become
                # garbage-collectable — equivalent to the original code's
                # per-rotation respawn for this single ego.
                fresh = RoadSideUnit(
                    save_path=None,
                    id=int((self.step / cfg['change_rsu_frame'] + 1) * 1000 + vehicle_num),
                    is_None=False,
                    use_semantic=self.use_semantic,
                )
                fresh.setup_sensors(parent=None, spawn_point=new_loc)
                self.rsu[vehicle_num] = fresh
    
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

        # spawn the rsu.
        self.spawn_rsu()
        rsu_data = []

        # If the frame is skipped.
        realtime_mode = os.environ.get('REALTIME_MODE', '0')
        force_fresh_every_frame = os.environ.get('OPENLOOP_FORCE_FRESH_INFERENCE', '0') == '1'
        if realtime_mode == '1':
            # REAL-TIME
            if self.step and self.step < self.next_action_step:
                # return the previous control signal.
                return self.infer.prev_control
        else:
            # NON_REAL_TIME
            if self.step and (not force_fresh_every_frame) and self.step % 4 != 0:
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
        # Two guards required to avoid KeyError mid-scenario:
        #   1. CarlaDataProvider.get_hero_actor()           — actor still in pool
        #   2. "rgb_front_<N>" in input_data                — sensors still publishing
        # When an ego completes its route (RC=100%) or hits collided_terminal,
        # the scenario_manager despawns it: cleanup_single() then
        # remove_actor_by_id() (which unregisters its sensors). There is a
        # tick window where the actor reference may still be cached but the
        # sensor stream has stopped — without guard #2 we hit
        # `KeyError: 'rgb_front_<N>'` at line ~331 and the whole agent
        # crashes via AgentError, killing the scenario for the surviving
        # egos too. Treat "sensors gone" the same as "actor gone": yield
        # None so the planner skips this ego.
        ego_data = [
            self.tick(input_data, vehicle_num)
            if (
                CarlaDataProvider.get_hero_actor(hero_id=vehicle_num) is not None
                and "rgb_front_{}".format(vehicle_num) in input_data
            )
            else None
            for vehicle_num in range(self.ego_vehicles_num)
        ]

        # Per-tick fusion topology diag log: one JSONL record per planner-running
        # tick proves whether infra LiDAR streams actually entered this tick's
        # cooperative batch. Written even when --infra-collab is off so
        # ablation pairs (on vs off) can be diff-compared after the fact.
        _infra_count = len(getattr(self, "infra_units", []) or [])
        _n_rsus_total = len(rsu_data)
        _infra_diag(
            log_dir=getattr(self.infer, "log_path", None),
            step=int(self.step),
            sim_t=float(timestamp) if timestamp is not None else None,
            planner_label="codriving",
            core_method=self.config['planning']['core_method'],
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
            if self.config['planning']['core_method'] == 'MotionNet':
                control_all, self.next_action_step = self.infer.get_action_from_list_multi_vehicle(car_data_raw=ego_data,
                                                                rsu_data_raw=rsu_data,
                                                                step=self.step,
                                                                timestamp=timestamp)
            elif self.config['planning']['core_method'] == 'rule_based':
                control_all, self.next_action_step = self.infer.get_action_from_list_inter(car_data_raw=ego_data,
                                                                rsu_data_raw=rsu_data,
                                                                step=self.step,
                                                                timestamp=timestamp)
        except (IndexError, RuntimeError) as _exc:
            print(f"[pnp_agent] inference exception ({type(_exc).__name__}: "
                  f"{_exc}); emitting zero control for this tick")
            import carla as _carla
            control_all = [_carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
                           for _ in range(max(1, self.ego_vehicles_num))]
            self.next_action_step = self.step + 1

        # Open-loop AP collection: read the perception predictions the infer
        # already stashed for the npz dump and feed them to the AP hook. No-op
        # in closed-loop runs (hook.enabled is False there). Try/except so a
        # perception-side glitch can never kill the drive.
        oloop = getattr(self, "_oloop", None)
        if oloop is not None and oloop.enabled:
            try:
                per_save = getattr(self.infer, "_last_perception_for_save", {}) or {}
                ego_actor_ids = set()
                for vehicle_num in range(self.ego_vehicles_num):
                    a = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)
                    if a is not None:
                        ego_actor_ids.add(a.id)
                for ego_id, _per in per_save.items():
                    # Prefer pred_box_lidar (raw opencood lidar frame, same
                    # convention perception_swap_agent feeds successfully)
                    # over the legacy pred_box_world (which lives in a
                    # codriving-specific axis-swapped frame and won't IoU
                    # against CARLA-world GT).
                    pred_lidar = _per.get("pred_box_lidar")
                    pred_score = _per.get("pred_score")
                    if pred_lidar is None or pred_score is None:
                        continue
                    # Use the actual hero actor's CARLA pose for the AP
                    # collector — that's what the GT projection (x_to_world
                    # in opencood) expects. Codriving's measurements dict
                    # carries lidar_pose_x/_y/theta but lidar may be offset
                    # from the actor center; using the actor pose makes the
                    # GT-from-CARLA query frame-consistent with how the
                    # perception model was trained.
                    actor = CarlaDataProvider.get_hero_actor(hero_id=int(ego_id))
                    if actor is None:
                        continue
                    tf = actor.get_transform()
                    # Pass LIDAR pose (not actor pose) — on v2xpnp/ucla_v2
                    # the actor is at world Z≈10m and the LiDAR mount is
                    # +2.5m above, so using actor pose offsets GT corners
                    # by ~12m in Z → no IoU matches → AP=0.
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

                    # Unified per-tick prediction + GT cache — same schema
                    # as perception_swap_agent so a single post-hoc analysis
                    # script can consume both pipelines. Overwrites the
                    # legacy 0000_pred.npz that pnp_infer_action_e2e_v2v.py
                    # writes (which used a swapped-frame pred_box_world)
                    # and adds a companion 0000_gt.npz with every CARLA
                    # vehicle in true world frame, no range filter. Lets
                    # post-hoc tools change AP eval range / IoU / score
                    # threshold without re-running CARLA.
                    _agent_save_path = getattr(self.infer, "save_path", None)
                    if (os.environ.get("PERCEPTION_SWAP_WRITE_PRED_FILES", "0") == "1"
                            and _agent_save_path is not None):
                        try:
                            from opencood.utils.transformation_utils import x_to_world
                            import pathlib as _pathlib
                            ego_dir = _pathlib.Path(_agent_save_path) / f"ego_vehicle_{int(ego_id)}"
                            ego_dir.mkdir(parents=True, exist_ok=True)
                            stem = f"{int(self.step):04d}"
                            # pred_box_lidar is the codriving model's raw
                            # output (pre X↔Y swap). For HEAL variants it's
                            # already in actor body frame (rotation applied
                            # in _run_heal_perception_single_agent). For
                            # vanilla codriving it's in the yaw=-90 lidar
                            # sensor frame and the X↔Y swap downstream
                            # converts to actor body. Apply that swap here
                            # only when not running a HEAL backbone.
                            heal_active = getattr(self.infer, "_heal_perception", None) is not None
                            pred_actor = np.asarray(pred_lidar, dtype=np.float64)
                            if pred_actor.size and not heal_active:
                                pred_actor = pred_actor[..., [1, 0, 2]]
                            # Apply x_to_world(primary_pose_world) →
                            # true CARLA world frame, same convention
                            # perception_swap saves in.
                            if pred_actor.size:
                                T_world_ego = x_to_world(primary_pose_world)
                                P, _, _ = pred_actor.shape
                                hom = np.concatenate(
                                    [pred_actor.reshape(-1, 3),
                                     np.ones((P * 8, 1), dtype=np.float64)],
                                    axis=1,
                                )
                                pred_world_arr = (T_world_ego @ hom.T).T[:, :3].reshape(P, 8, 3).astype(np.float32)
                            else:
                                pred_world_arr = np.zeros((0, 8, 3), dtype=np.float32)
                            # Resolve the per-ego AP collector to reuse its
                            # GT builder (queries CARLA, applies the
                            # bottom-CCW corner ordering perception_swap
                            # also uses).
                            coll = (oloop.collectors[int(ego_id)]
                                    if 0 <= int(ego_id) < len(oloop.collectors)
                                    else None)
                            gt_w = np.zeros((0, 8, 3), dtype=np.float32)
                            gt_ids = np.zeros((0,), dtype=np.int64)
                            gt_roles: list = []
                            gt_types: list = []
                            if coll is not None:
                                vis_ids = None
                                if not getattr(oloop, "_force_no_filter_for_save", True):
                                    vis_ids = coll._visible_ids_for_frame(
                                        oloop.frame_idx_at(
                                            float(timestamp) if timestamp is not None else 0.0
                                        )
                                    )
                                gt_w, gt_ids, gt_roles, gt_types = coll._build_gt_world_with_meta(
                                    set(ego_actor_ids), visible_actor_ids=vis_ids,
                                )
                            _yaw_deg = float(tf.rotation.yaw)
                            _yaw_rad = math.radians(_yaw_deg)
                            np.savez(
                                str(ego_dir / f"{stem}_pred.npz"),
                                pred_box_world=pred_world_arr,
                                pred_score=np.asarray(pred_score, dtype=np.float32),
                                lidar_pose_x=np.float64(primary_pose_world[0]),
                                lidar_pose_y=np.float64(primary_pose_world[1]),
                                theta=np.float64(_per.get("theta", 0.0)),
                                ego_x=np.float64(tf.location.x),
                                ego_y=np.float64(tf.location.y),
                                yaw_deg=np.float64(_yaw_deg),
                                yaw_rad=np.float64(_yaw_rad),
                            )
                            np.savez(
                                str(ego_dir / f"{stem}_gt.npz"),
                                gt_box_world=gt_w,
                                gt_actor_ids=gt_ids,
                                gt_role_names=np.array(gt_roles, dtype=object),
                                gt_type_ids=np.array(gt_types, dtype=object),
                                ego_x=np.float64(tf.location.x),
                                ego_y=np.float64(tf.location.y),
                                theta=np.float64(_per.get("theta", 0.0)),
                                visibility_filter_hits=np.int64(0),
                                visibility_filter_applied=bool(False),
                            )
                        except Exception as _exc:
                            if getattr(self, "_unified_save_err_count", 0) < 3:
                                self._unified_save_err_count = getattr(self, "_unified_save_err_count", 0) + 1
                                print(f"[codriving] unified pred/gt cache save fail: {_exc}")
                # AP snapshot — fire on EVERY planner-running tick so that
                # short scenarios (e.g. v2xpnp + collided_terminal at tick 1)
                # still produce an AP file before destroy() runs. Cheap +
                # idempotent.
                _infer_save = getattr(self.infer, "save_path", None)
                if _infer_save is not None:
                    try:
                        oloop.save_all(ap_save_root=str(_infer_save))
                    except Exception as _exc:
                        if getattr(self, "_oloop_save_err_count", 0) < 3:
                            self._oloop_save_err_count = getattr(self, "_oloop_save_err_count", 0) + 1
                            print(f"[codriving] AP save fail: {_exc}")
            except Exception as exc:
                _ec = getattr(self, "_oloop_err_count", 0) + 1
                self._oloop_err_count = _ec
                if _ec <= 3 or _ec % 100 == 0:
                    print(f"[codriving] open-loop AP update fail #{_ec}: {exc}")

        # Open-loop BEV viz — emits side-by-side cam+BEV PNG every viz_every
        # ticks. Driven by the same _last_perception_for_save dict as the AP
        # hook plus this tick's planner waypoints. No-op in closed-loop.
        bev = getattr(self, "_bev", None)
        if (oloop is not None and oloop.enabled
                and bev is not None and bev.viz_dir):
            try:
                per_save = getattr(self.infer, "_last_perception_for_save", {}) or {}
                # Hero actors gathered above for AP; reuse rather than re-query.
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
                    ego_yaw_rad = math.radians(float(tf.rotation.yaw))
                    ego_speed = float(math.hypot(vel.x, vel.y))
                    # Predicted ego trajectory: codriving stores waypoints
                    # in a local frame on ``last_predicted_waypoints_local``.
                    # Convention (per openloop/tools/carla_openloop_prototype.py:402):
                    #   forward = -waypoint[1]
                    #   left    = -waypoint[0]
                    # then rotate by ego yaw and translate by ego world xy.
                    last_local = getattr(
                        self.infer, "last_predicted_waypoints_local", {}
                    ) or {}
                    local_wps = last_local.get(int(ego_id))
                    traj_world = None
                    if local_wps is not None and len(local_wps) > 0:
                        wp = np.asarray(local_wps, dtype=np.float64).reshape(-1, 2)
                        fwd = -wp[:, 1]
                        lft = -wp[:, 0]
                        cy_, sy_ = math.cos(ego_yaw_rad), math.sin(ego_yaw_rad)
                        wx = ego_xy_world[0] + cy_ * fwd - sy_ * lft
                        wy = ego_xy_world[1] + sy_ * fwd + cy_ * lft
                        traj_world = np.stack([wx, wy], axis=1)
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
                            from openloop_helpers import _lazy_eval_imports as _lei
                            _, _, calculate_ap, _, _ = _lei()
                            ap_value, _, _ = calculate_ap(coll.result_stat, 0.5)
                            running_ap = float(ap_value)
                        except Exception:
                            running_ap = None
                    # BEV pred-corners path. Two branches:
                    #
                    # Vanilla codriving (default): keep the legacy lidar→world
                    # transform here — it's been validated against historical
                    # codriving openloop runs and we don't want to change
                    # vanilla baseline numbers.
                    #
                    # New variants (codriving_zerofeat / codriving_<heal>):
                    # use pred_box_world directly. That field is built in
                    # pnp_infer_action_e2e_v2v.py via transform_2d_points
                    # (X↔Y-swapped tensor + compass-corrected rotation +
                    # swapped translation), which lands in true CARLA world.
                    # The legacy lidar→world re-derivation here had a yaw-
                    # dependent rotation offset vs. gt_corners_world that
                    # was visible on the new variants' BEV PNGs — using
                    # pred_box_world keeps pred and GT in the same world
                    # frame for those runs.
                    _is_codriving_variant = (
                        getattr(self.infer, "_zero_bev_feature", False)
                        or getattr(self.infer, "_perception_backbone", "codriving") != "codriving"
                    )
                    pred_world_carla = None
                    if _is_codriving_variant:
                        _pred_world_arr = _per.get("pred_box_world")
                        if _pred_world_arr is not None and len(_pred_world_arr) > 0:
                            pred_world_carla = _pred_world_arr
                    else:
                        # Transform raw opencood-lidar predictions into CARLA
                        # world for the BEV pane (which expects world coords).
                        # The legacy pred_box_world stored in _last_perception_for_save
                        # is in a codriving-specific axis-swapped frame and won't
                        # render at the right place on the BEV.
                        pred_lidar_for_bev = _per.get("pred_box_lidar")
                        if pred_lidar_for_bev is not None and len(pred_lidar_for_bev) > 0:
                            cy_, sy_ = math.cos(ego_yaw_rad), math.sin(ego_yaw_rad)
                            # (P, 8, 3). Rotate xy by yaw, translate by ego_xy_world.
                            pl = np.asarray(pred_lidar_for_bev, dtype=np.float64)
                            x_l = pl[..., 0]
                            y_l = pl[..., 1]
                            x_w = ego_xy_world[0] + cy_ * x_l - sy_ * y_l
                            y_w = ego_xy_world[1] + sy_ * x_l + cy_ * y_l
                            pred_world_carla = np.stack([x_w, y_w, pl[..., 2]], axis=-1)
                    # Reuse the AP collector's last-frame snapshot — it
                    # already queried CARLA this tick AND filtered to the
                    # 140×80m AP eval crop, so the BEV pane shows exactly
                    # what was scored (no double CARLA query).
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
                    print(f"[codriving] open-loop BEV render fail #{_ec}: {exc}")

        ### return the control signal in list format.
        return control_all

    def destroy(self):
        # Flush AP json BEFORE deleting the planning models, in case the
        # subprocess hard-exits during model teardown. Write next to the
        # per-tick pred dirs (``self.infer.save_path``) so the
        # ``run_custom_eval`` post-metrics sweep finds it via
        # ``run_dir.parent / perception_swap_ap_ego{N}.json``.
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
                print(f"[codriving] open-loop AP final save failed: {exc}")
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
