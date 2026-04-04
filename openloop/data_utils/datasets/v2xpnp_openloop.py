"""
v2xpnp_openloop.py
==================
Dataset loader for v2xpnp recorded scenarios, producing the ``input_data``
dict format expected by leaderboard agents and the ``car_data_raw`` format
expected by ``PnP_infer``, without any CARLA server dependency.

Scenario directory structure assumed::

    <scenario_dir>/
        <actor_id>/           # integer subfolder, one per ego vehicle
            000000.yaml
            000000.bin        # float32 LiDAR [N, 4] in sensor-local frame
            000000_cam1_left.jpg   (legacy export)
            000000_cam2_left.jpg
            000000_cam3_left.jpg
            000000_cam4_left.jpg
            000000_cam1.jpeg       (new export)
            000000_cam2.jpeg
            000000_cam3.jpeg
            000000_cam4.jpeg
            000001.yaml ...
            ...

Each YAML contains:
  lidar_pose:     [x, y, z, roll_deg, yaw_deg, pitch_deg]   (world frame, angles in DEGREES)
  true_ego_pose:  [x, y, z, roll_deg, yaw_deg, pitch_deg]   (identical to lidar_pose)
  ego_speed:      float (m/s)
  infra:          bool
  cam1_left .. cam4_left  (legacy)  OR  cam1 .. cam4 (new):
      cords:      [x, y, z, roll, pitch, yaw]
      extrinsic:  4×4 list (camera→world)
      intrinsic:  3×3 list
  vehicles:       { id: {location, angle, extent, obj_type, attribute} }

Camera assignment (front / left / right / rear) is computed dynamically each
frame from ego-relative camera positions (``cam*.cords``). This is more robust
for v2xpnp exports where extrinsic yaw can be inconsistent with image naming.
For the new ``cam1..cam4`` schema we use fixed mapping:
    cam1=front, cam2=right, cam3=left, cam4=rear.
If needed, we fall back to a yaw-based heuristic.

GPS encoding: world (x, y) coordinates are converted to a fake lat/lon pair so
that the RoutePlanner (which multiplies by 111 324 m/deg) reproduces the
original metric coordinates:
    fake_lat = world_x / GPS_SCALE_LAT
    fake_lon = world_y / GPS_SCALE_LON
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml


# GPS scale factors used by RoutePlanner (planner.py / planner_pnp.py)
GPS_SCALE = np.array([111324.60662786, 111319.490945])   # metres per degree [lat, lon]

# Nominal camera FOV and resolution used by the agents (from base_agent defaults)
_CAM_WIDTH  = 800
_CAM_HEIGHT = 600
_CAM_FOV    = 100

# How many future steps to include as ground-truth trajectory.
# Matches RiskM config: prediction_len=6 at prediction_hz=2.
MAX_GT_STEPS = 6

# Downsampling rate: data is at 10 Hz; predictions are evaluated at 2 Hz.
# Every PREDICTION_DOWNSAMPLING_RATE-th frame = 0.5 s per step → 3 s horizon.
PREDICTION_DOWNSAMPLING_RATE = 5


# ---------------------------------------------------------------------------
# Utility: pose / transform helpers
# ---------------------------------------------------------------------------

def _yaw_from_extrinsic(extrinsic: List[List[float]]) -> float:
    """Extract the world-frame yaw (radians) from a 4×4 camera→world matrix."""
    R = np.array(extrinsic)[:3, :3]
    return math.atan2(R[1, 0], R[0, 0])


def _detect_camera_keys(meta: Dict[str, Any]) -> List[str]:
    """
    Detect camera keys present in one frame's YAML.

    Preference order:
      1) new schema:    cam1..cam4
      2) legacy schema: cam1_left..cam4_left
      3) fallback: any cam* dicts with intrinsic/extrinsic fields
    """
    if all(k in meta for k in ("cam1", "cam2", "cam3", "cam4")):
        return ["cam1", "cam2", "cam3", "cam4"]
    if all(k in meta for k in ("cam1_left", "cam2_left", "cam3_left", "cam4_left")):
        return ["cam1_left", "cam2_left", "cam3_left", "cam4_left"]

    keys: List[str] = []
    for k, v in meta.items():
        if not isinstance(v, dict):
            continue
        if "intrinsic" in v and "extrinsic" in v and re.fullmatch(r"cam\d+(?:_left)?", str(k)):
            keys.append(str(k))

    def _sort_key(name: str) -> Tuple[int, str]:
        m = re.match(r"cam(\d+)", name)
        idx = int(m.group(1)) if m else 999
        return idx, name

    return sorted(keys, key=_sort_key)


def _assign_camera_directions_from_position(
    meta: Dict[str, Any],
    cam_keys: List[str],
) -> Dict[str, str]:
    """
    Return a mapping {cam_key → direction} where direction ∈
    {front, left, right, rear}.

    Preferred method: infer direction from each camera's ego-relative
    translation in world XY (using ``cam*.cords``).

    This avoids relying on extrinsic yaw conventions that differ across
    dataset exporters.
    """
    ego_yaw = math.radians(meta["lidar_pose"][4])    # index 4 = yaw_deg → rad
    ego_x, ego_y = float(meta["lidar_pose"][0]), float(meta["lidar_pose"][1])
    fwd = np.array([math.cos(ego_yaw), math.sin(ego_yaw)], dtype=np.float64)
    left = np.array([-math.sin(ego_yaw), math.cos(ego_yaw)], dtype=np.float64)

    # (cam_key, front_score, left_score)
    entries = []
    for k in cam_keys:
        if k not in meta or "cords" not in meta[k]:
            continue
        cords = meta[k]["cords"]
        if not isinstance(cords, (list, tuple)) or len(cords) < 2:
            continue
        rel = np.array([float(cords[0]) - ego_x, float(cords[1]) - ego_y],
                       dtype=np.float64)
        entries.append((k, float(rel @ fwd), float(rel @ left)))

    if not entries:
        return {}

    assignment: Dict[str, str] = {}
    remaining = list(entries)

    # Greedy unique picks per axis extremum.
    # front/rear from forward axis, left/right from left axis.
    picks = [
        ("front", lambda e: e[1], True),
        ("rear", lambda e: e[1], False),
        ("left", lambda e: e[2], True),
        ("right", lambda e: e[2], False),
    ]
    for direction, score_fn, reverse in picks:
        if not remaining:
            break
        best = sorted(remaining, key=score_fn, reverse=reverse)[0]
        assignment[best[0]] = direction
        remaining = [e for e in remaining if e[0] != best[0]]

    return assignment


def _assign_camera_directions_from_extrinsic_yaw(
    meta: Dict[str, Any],
    cam_keys: List[str],
) -> Dict[str, str]:
    """
    Fallback method: assign cameras from relative yaw extracted from camera
    extrinsic matrices.
    """
    ego_yaw = math.radians(meta["lidar_pose"][4])    # index 4 = yaw_deg → rad
    # (cam_key, relative_yaw_deg)
    entries = []
    for k in cam_keys:
        if k not in meta or "extrinsic" not in meta[k]:
            continue
        world_yaw = _yaw_from_extrinsic(meta[k]["extrinsic"])
        rel_deg = math.degrees(
            (world_yaw - ego_yaw + math.pi) % (2 * math.pi) - math.pi
        )
        entries.append((k, rel_deg))

    assignment: Dict[str, str] = {}
    remaining = list(entries)
    for target_dir, target_deg in [
        ("front", 0.0), ("right", -90.0), ("left", 90.0), ("rear", 180.0)
    ]:
        if not remaining:
            break
        best_key = min(
            remaining,
            key=lambda kd: abs(
                (kd[1] - target_deg + 180) % 360 - 180
            ),
        )[0]
        assignment[best_key] = target_dir
        remaining = [kd for kd in remaining if kd[0] != best_key]

    return assignment


def _assign_camera_directions(
    meta: Dict[str, Any]
) -> Dict[str, str]:
    """
    Assign camera directions with robust position-based inference first,
    falling back to extrinsic-yaw matching when needed.
    """
    cam_keys = _detect_camera_keys(meta)
    if not cam_keys:
        return {}

    # New schema has explicit semantic ordering.
    if cam_keys == ["cam1", "cam2", "cam3", "cam4"]:
        return {
            "cam1": "front",
            "cam2": "right",
            "cam3": "left",
            "cam4": "rear",
        }

    assign = _assign_camera_directions_from_position(meta, cam_keys)
    if assign:
        return assign
    return _assign_camera_directions_from_extrinsic_yaw(meta, cam_keys)


def _compute_camera_intrinsics(width: int = _CAM_WIDTH,
                                height: int = _CAM_HEIGHT,
                                fov: int = _CAM_FOV) -> List[List[float]]:
    """
    Compute pinhole intrinsics matching CARLA's projection convention.
    This mirrors ``get_camera_intrinsic`` in ``pnp_agent_e2e.py``.
    """
    cx = width  / 2.0
    cy = height / 2.0
    f  = width  / (2.0 * math.tan(math.radians(fov / 2.0)))
    return [[f, 0.0, cx],
            [0.0, f, cy],
            [0.0, 0.0, 1.0]]


def _pose_to_matrix4(pose: List[float]) -> np.ndarray:
    """Convert [x, y, z, roll_rad, pitch_rad, yaw_rad] (radians) to a 4×4 matrix."""
    x, y, z, roll, pitch, yaw = pose
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    R = np.array([
        [cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr],
        [sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr],
        [-sp,      cp * sr,                  cp * cr               ],
    ])
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3]  = [x, y, z]
    return M


def _dataset_pose_to_matrix4(pose: List[float]) -> np.ndarray:
    """Convert dataset lidar_pose [x, y, z, roll_deg, yaw_deg, pitch_deg] to 4×4."""
    x, y, z = pose[0], pose[1], pose[2]
    roll_r  = math.radians(pose[3])
    yaw_r   = math.radians(pose[4])
    pitch_r = math.radians(pose[5])
    return _pose_to_matrix4([x, y, z, roll_r, pitch_r, yaw_r])


def _wrap_angle_rad(angle: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _compute_lidar2camera(
    lidar_pose: List[float],
    cam_extrinsic: List[List[float]],
) -> List[List[float]]:
    """
    Compute lidar→camera extrinsic (lidar-frame → camera-frame).

    Matches the logic in ``pnp_agent_e2e.get_camera_extrinsic``:
        extrin = inv(ref_extrin.get_matrix()) @ cur_extrin.get_matrix()
    where ref_extrin = lidar transform and cur_extrin = camera transform.

    Both are expressed as camera/lidar → world matrices.
    """
    lidar_world  = _dataset_pose_to_matrix4(lidar_pose)
    camera_world = np.array(cam_extrinsic)         # 4×4 camera→world from YAML
    lidar2cam    = np.linalg.inv(lidar_world) @ camera_world
    return lidar2cam.tolist()


def _sparsify_route_waypoints(gt_traj: np.ndarray) -> np.ndarray:
    """
    Heavily downsample route anchors to reduce future-trajectory leakage while
    keeping CARLA-style route conditioning alive.

    Policy:
      - Always include start and end.
      - Include 0/1/2 intermediate anchors based on total path length.
    """
    gt = np.asarray(gt_traj, dtype=np.float64)
    if gt.ndim != 2 or gt.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if gt.shape[0] == 1:
        return gt.copy()

    seg = np.linalg.norm(np.diff(gt, axis=0), axis=1)
    total_len = float(np.sum(seg))
    if total_len < 40.0:
        n_mid = 0
    elif total_len < 120.0:
        n_mid = 1
    else:
        n_mid = 2

    cum = np.concatenate([[0.0], np.cumsum(seg)])
    frac_targets = [0.0]
    if n_mid == 1:
        frac_targets.append(0.5)
    elif n_mid == 2:
        frac_targets.extend([1.0 / 3.0, 2.0 / 3.0])
    frac_targets.append(1.0)

    picked_idx: List[int] = []
    for frac in frac_targets:
        d = frac * total_len
        idx = int(np.argmin(np.abs(cum - d)))
        picked_idx.append(idx)

    # Preserve order and uniqueness.
    uniq_idx: List[int] = []
    for i in picked_idx:
        if i not in uniq_idx:
            uniq_idx.append(i)
    if uniq_idx[0] != 0:
        uniq_idx.insert(0, 0)
    if uniq_idx[-1] != (len(gt) - 1):
        uniq_idx.append(len(gt) - 1)

    sparse = gt[uniq_idx]
    if sparse.shape[0] == 1:
        sparse = np.vstack([sparse, gt[-1:]])
    return sparse


# ---------------------------------------------------------------------------
# ScenarioFrame — single-frame data container
# ---------------------------------------------------------------------------

class ScenarioFrame:
    """
    All data for one actor at one time-step, in the format expected by the
    agents:

    ``input_data`` dict::

        "rgb_front_{vid}" : (frame_idx, np.ndarray[H, W, 4] BGRA)
        "rgb_left_{vid}"  : (frame_idx, np.ndarray[H, W, 4] BGRA)
        "rgb_right_{vid}" : (frame_idx, np.ndarray[H, W, 4] BGRA)
        "rgb_rear_{vid}"  : (frame_idx, np.ndarray[H, W, 4] BGRA)
        "lidar_{vid}"     : (frame_idx, np.ndarray[N, 4]  x,y,z,intensity)
        "gps_{vid}"       : (frame_idx, np.ndarray[2]  fake_lat/lon)
        "imu_{vid}"       : (frame_idx, np.ndarray[7]  [..., compass_rad])
        "speed_{vid}"     : (frame_idx, {"move_state": {"speed": float}})

    ``car_data_raw`` entry::

        {
          "rgb_front", "rgb_left", "rgb_right", "rgb_rear"  : np.ndarray[H,W,3] RGB
          "lidar"       : np.ndarray[N, 4]
          "measurements": {
              "gps_x", "gps_y": float             (local metres from RoutePlanner)
              "x", "y"        : float             (same, axes flipped per agent)
              "theta"         : float             (compass heading, radians)
              "lidar_pose_x/y/z"                  (world frame)
              "lidar_pose_gps_x/y"                (same, axes swapped per agent)
              "camera_front/left/right/rear_intrinsics" : list 3×3
              "camera_front/left/right/rear_extrinsics" : list 4×4 (lidar→cam)
              "speed"         : float
              "compass"       : float  (radians)
              "command"       : int    (RoadOption.LANEFOLLOW = 4)
              "target_point"  : np.ndarray[2] (local next-waypoint, metres)
          }
          "bev"          : PIL.Image (placeholder; never used by planner model)
          "drivable_area": np.ndarray[192, 96] binary
        }
    """

    def __init__(self, actor_id: int, frame_idx: int,
                 input_data: Dict, car_data_raw_entry: Dict,
                 world_pose: np.ndarray):
        self.actor_id         = actor_id
        self.frame_idx        = frame_idx
        self.input_data       = input_data
        self.car_data_raw_entry = car_data_raw_entry
        self.world_pose       = world_pose   # [x, y, z, roll, pitch, yaw]


# ---------------------------------------------------------------------------
# V2XPnPScenario — loads one scenario directory
# ---------------------------------------------------------------------------

class V2XPnPScenario:
    """
    Loads a v2xpnp recorded scenario and exposes it frame-by-frame.

    Parameters
    ----------
    scenario_dir : str or Path
        Path to the scenario directory (e.g. ``…/2023-04-07-15-02-15_1_1``).
    fps : float
        Assumed recording frame rate (default: 10 Hz).
    """

    def __init__(self, scenario_dir, fps: float = 10.0):
        self.scenario_dir = Path(scenario_dir)
        self.fps          = fps
        self._actor_ids   = self._discover_actors()
        self._frame_counts: Dict[int, int] = {}
        for aid in self._actor_ids:
            self._frame_counts[aid] = self._count_frames(aid)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def actor_ids(self) -> List[int]:
        return self._actor_ids

    def num_frames(self, actor_id: int) -> int:
        return self._frame_counts.get(actor_id, 0)

    # ------------------------------------------------------------------
    # Core data loading
    # ------------------------------------------------------------------

    def load_frame(self, actor_id: int, frame_idx: int,
                   next_waypoints: Optional[np.ndarray] = None) -> ScenarioFrame:
        """
        Load one frame and return a ``ScenarioFrame``.

        Parameters
        ----------
        actor_id : int
            Which ego vehicle.
        frame_idx : int
            Zero-based frame index.
        next_waypoints : np.ndarray[K, 2] or None
            Future world-frame (x, y) positions used to compute the
            ``target_point`` local waypoint.  If None, an ahead-pointing
            placeholder is used.
        """
        base = self.scenario_dir / str(actor_id) / f"{frame_idx:06d}"

        # ---- Load YAML metadata ----
        with open(f"{base}.yaml") as fh:
            meta = yaml.safe_load(fh)

        pose       = meta["lidar_pose"]          # [x, y, z, roll_deg, yaw_deg, pitch_deg]
        ego_speed  = float(meta.get("ego_speed", 0.0))

        # If recorded speed is zero (common data-recording bug where the
        # field was never populated), derive it from position deltas.
        if ego_speed == 0.0 and frame_idx > 0:
            prev_yaml = self.scenario_dir / str(actor_id) / f"{frame_idx - 1:06d}.yaml"
            try:
                with open(prev_yaml) as _fh:
                    prev_meta = yaml.safe_load(_fh)
                prev_pose = prev_meta["lidar_pose"]
                dx = pose[0] - prev_pose[0]
                dy = pose[1] - prev_pose[1]
                ego_speed = math.sqrt(dx * dx + dy * dy) * self.fps
            except Exception:
                pass  # keep ego_speed = 0 if prev frame unavailable

        # Keep compass aligned with dataset yaw for open-loop replay.
        # This matches the CoDriving/PnP preprocessing path used by the
        # leaderboard agents when consuming v2xpnp recordings.
        yaw_rad    = math.radians(float(pose[4]))   # dataset yaw_deg -> rad
        compass    = _wrap_angle_rad(yaw_rad)

        world_x, world_y, world_z = pose[0], pose[1], pose[2]

        # ---- Camera assignment ----
        cam_assign = _assign_camera_directions(meta)   # {cam_key → direction}
        dir_to_cam = {v: k for k, v in cam_assign.items()}

        # ---- Load camera images → BGRA ----
        def load_bgra(cam_key: str) -> np.ndarray:
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
                path = Path(f"{base}_{cam_key}{ext}")
                if not path.exists():
                    continue
                img = cv2.imread(str(path))
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            return np.zeros((_CAM_HEIGHT, _CAM_WIDTH, 4), dtype=np.uint8)

        cameras_bgra = {
            direction: load_bgra(cam_key)
            for cam_key, direction in cam_assign.items()
        }
        # Ensure all four directions exist
        for d in ("front", "left", "right", "rear"):
            if d not in cameras_bgra:
                cameras_bgra[d] = np.zeros((_CAM_HEIGHT, _CAM_WIDTH, 4),
                                            dtype=np.uint8)

        # ---- Load LiDAR ----
        lidar = np.fromfile(f"{base}.bin", dtype=np.float32).reshape(-1, 4)

        # ---- GPS (fake lat/lon so RoutePlanner reproduces world xy) ----
        fake_gps = np.array([world_x / GPS_SCALE[0],
                              world_y / GPS_SCALE[1]], dtype=np.float64)

        # ---- IMU (axes 0-5 zeroed, compass at index 6) ----
        imu      = np.zeros(7, dtype=np.float64)
        imu[6]   = compass

        # ---- Speed ----
        # Keep as numpy scalar so planner-side logging code using `.tolist()`
        # remains compatible without planner code changes.
        ego_speed_np = np.float32(ego_speed)
        speed_dict = {"move_state": {"speed": ego_speed_np}}

        # ---- Intrinsics / Extrinsics (per direction) ----
        #  Agent computes ONE intrinsic for all cameras (same sensor spec).
        #  The YAML stores per-camera intrinsics; we use them directly.
        intrinsics: Dict[str, List] = {}
        extrinsics: Dict[str, List] = {}
        for cam_key, direction in cam_assign.items():
            cam_data = meta[cam_key]
            intrinsics[direction] = cam_data["intrinsic"]   # 3×3 list
            extrinsics[direction] = _compute_lidar2camera(
                pose, cam_data["extrinsic"]
            )
        for d in ("front", "left", "right", "rear"):
            if d not in intrinsics:
                intrinsics[d] = _compute_camera_intrinsics()
                extrinsics[d] = np.eye(4).tolist()

        # ---- Target waypoint in ego-local frame ----
        theta      = compass + math.pi / 2
        R          = np.array([[math.cos(theta), -math.sin(theta)],
                                [math.sin(theta),  math.cos(theta)]])
        pos        = fake_gps * GPS_SCALE    # ≈ world (x, y) in metres

        if next_waypoints is not None and len(next_waypoints) > 0:
            # Use the nearest future waypoint as the local navigation target
            nwp_world = next_waypoints[min(1, len(next_waypoints) - 1)]
            world_diff = nwp_world - pos
        else:
            # Placeholder: 5 m ahead in vehicle-forward heading.
            world_diff = np.array([math.cos(theta), math.sin(theta)]) * 5.0

        local_target = R.T @ world_diff

        # Clip to detection range [front=36, rear=12, left=12, right=12]
        DET_RANGE = [36.0, 12.0, 12.0, 12.0]
        local_target = np.clip(local_target,
                               a_min=[-DET_RANGE[2], -DET_RANGE[0]],
                               a_max=[ DET_RANGE[3],  DET_RANGE[1]])

        # ---- Drivable area: 192×96 binary (filled by MockBirdViewProducer) ----
        # The agent's tick() constructs this from BirdViewProducer.  In the
        # open-loop path it is constructed by our adapter when run_step calls
        # tick(). Here we also expose a pre-computed version for planners that
        # bypass tick() (e.g. when car_data_raw is built directly).
        drivable_area = np.ones((192, 96), dtype=np.float32)  # conservative fallback

        # ---- Assemble input_data (keyed by actor_id as vehicle_num) ----
        vid = actor_id
        input_data: Dict[str, Any] = {
            f"rgb_front_{vid}":  (frame_idx, cameras_bgra["front"]),
            f"rgb_left_{vid}":   (frame_idx, cameras_bgra["left"]),
            f"rgb_right_{vid}":  (frame_idx, cameras_bgra["right"]),
            f"rgb_rear_{vid}":   (frame_idx, cameras_bgra["rear"]),
            f"lidar_{vid}":      (frame_idx, lidar),
            f"gps_{vid}":        (frame_idx, fake_gps),
            f"imu_{vid}":        (frame_idx, imu),
            f"speed_{vid}":      (frame_idx, speed_dict),
        }

        # ---- Assemble car_data_raw entry ----
        measurements = {
            "gps_x":                    pos[0],
            "gps_y":                    pos[1],
            "x":                        pos[1],    # agent convention: x = gps_y
            "y":                        -pos[0],   # agent convention: y = -gps_x
            "theta":                    compass,
            "lidar_pose_x":             world_x,
            "lidar_pose_y":             world_y,
            "lidar_pose_z":             world_z,
            "lidar_pose_gps_x":         -world_y,  # agent convention
            "lidar_pose_gps_y":          world_x,  # agent convention
            "speed":                    ego_speed_np,
            "compass":                  compass,
            "command":                  4,         # RoadOption.LANEFOLLOW
            "target_point":             local_target,
            "camera_front_intrinsics":  intrinsics["front"],
            "camera_front_extrinsics":  extrinsics["front"],
            "camera_left_intrinsics":   intrinsics["left"],
            "camera_left_extrinsics":   extrinsics["left"],
            "camera_right_intrinsics":  intrinsics["right"],
            "camera_right_extrinsics":  extrinsics["right"],
            "camera_rear_intrinsics":   intrinsics["rear"],
            "camera_rear_extrinsics":   extrinsics["rear"],
        }

        # RGB in HWC uint8 (agent expects BGR→RGB conversion already done)
        def bgra_to_rgb(arr: np.ndarray) -> np.ndarray:
            return cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2RGB)

        from PIL import Image as _PILImage
        car_data_raw_entry = {
            "rgb_front":    bgra_to_rgb(cameras_bgra["front"]),
            "rgb_left":     bgra_to_rgb(cameras_bgra["left"]),
            "rgb_right":    bgra_to_rgb(cameras_bgra["right"]),
            "rgb_rear":     bgra_to_rgb(cameras_bgra["rear"]),
            "lidar":        lidar,
            "measurements": measurements,
            "bev":          _PILImage.new("RGB", (400, 400), (0, 0, 0)),
            "drivable_area": drivable_area,
        }

        return ScenarioFrame(
            actor_id         = actor_id,
            frame_idx        = frame_idx,
            input_data       = input_data,
            car_data_raw_entry = car_data_raw_entry,
            world_pose       = np.array(pose),
        )

    # ------------------------------------------------------------------
    # Ground-truth trajectory
    # ------------------------------------------------------------------

    def get_gt_trajectory(self, actor_id: int) -> np.ndarray:
        """
        Return all world-frame ego positions for the given actor.

        Returns
        -------
        np.ndarray  shape [N_frames, 2]   columns: [world_x, world_y]
        """
        poses = []
        for fi in range(self.num_frames(actor_id)):
            yaml_path = (self.scenario_dir / str(actor_id) /
                         f"{fi:06d}.yaml")
            with open(yaml_path) as fh:
                m = yaml.safe_load(fh)
            poses.append(m["lidar_pose"][:2])
        return np.array(poses, dtype=np.float64)   # [N, 2]

    def get_surrounding_trajectories(
            self, actor_id: int, frame_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return future world-frame positions of surrounding vehicles at a given
        frame, extracted from the YAML ``vehicles`` field.

        Because each YAML only contains the CURRENT positions of surrounding
        actors, we build trajectories by loading consecutive frames.

        Returns
        -------
        traj  : np.ndarray  [N_actors, T_future, 2]
        mask  : np.ndarray  [N_actors, T_future]  1 = valid
        """
        n_future = MAX_GT_STEPS
        n_frames = self.num_frames(actor_id)

        # Collect actor ids present at current frame
        yaml_path = (self.scenario_dir / str(actor_id) /
                     f"{frame_idx:06d}.yaml")
        with open(yaml_path) as fh:
            m0 = yaml.safe_load(fh)
        # Exclude the ego vehicle itself (actor_id) from the surrounding set.
        # This mirrors RiskM's closest-actor filter and prevents false self-collisions
        # when the ego actor ID appears in its own YAML vehicles dict.
        actor_ids_at_t0 = sorted(
            aid for aid in m0.get("vehicles", {}).keys() if aid != actor_id
        )

        if not actor_ids_at_t0:
            return np.zeros((0, n_future, 2)), np.zeros((0, n_future))

        traj = np.zeros((len(actor_ids_at_t0), n_future, 2), dtype=np.float64)
        mask = np.zeros((len(actor_ids_at_t0), n_future),    dtype=np.float64)

        # Sample at PREDICTION_DOWNSAMPLING_RATE-frame intervals (0.5 s per step)
        # to match the 2 Hz prediction frequency used in RiskM.
        for t_step in range(1, n_future + 1):
            fi = frame_idx + t_step * PREDICTION_DOWNSAMPLING_RATE
            if fi >= n_frames:
                break
            yp = (self.scenario_dir / str(actor_id) / f"{fi:06d}.yaml")
            with open(yp) as fh:
                mt = yaml.safe_load(fh)
            veh_data = mt.get("vehicles", {})
            for ai, aid in enumerate(actor_ids_at_t0):
                if aid in veh_data:
                    loc = veh_data[aid]["location"]
                    traj[ai, t_step - 1, 0] = loc[0]
                    traj[ai, t_step - 1, 1] = loc[1]
                    mask[ai, t_step - 1]    = 1.0

        return traj, mask

    # ------------------------------------------------------------------
    # Global plan (route) for RoutePlanner
    # ------------------------------------------------------------------

    def build_global_plan(
            self, actor_id: int, sparsify: bool = False
    ) -> List[List[Tuple[Dict, Any]]]:
        """
        Build a ``global_plan`` consumable by
        ``RoutePlanner.set_route(plan, gps=True)``.

        Each entry is  ``({"lat": fake_lat, "lon": fake_lon}, command)``,
        where fake lat/lon encode world (x, y) so that
        ``(latlon - mean) * scale = world_xy``.

        By default this returns the full per-frame route anchors (no downsampling),
        which is useful for orientation / calibration sweeps.
        Set ``sparsify=True`` to restore the fairness-focused sparse anchors.

        Returns
        -------
        list of length ``ego_vehicles_num``, each element being a list of
        (gps_dict, RoadOption) tuples — one per trajectory waypoint.
        """
        gt_traj = self.get_gt_trajectory(actor_id)   # [N, 2]
        route_xy = _sparsify_route_waypoints(gt_traj) if sparsify else gt_traj

        # Import the stub RoadOption set up by install_openloop_stubs
        try:
            from agents.navigation.local_planner import RoadOption
        except ImportError:
            # Fallback: make a minimal local stub
            class RoadOption:
                def __init__(self, v=4): self.value = v

        route: List[Tuple[Dict, Any]] = []
        for xy in route_xy:
            gps = {
                "lat": float(xy[0] / GPS_SCALE[0]),
                "lon": float(xy[1] / GPS_SCALE[1]),
            }
            route.append((gps, RoadOption(4)))  # LANEFOLLOW = 4

        # RoutePlanner expects list-of-lists (one per ego vehicle); return for
        # actor_id mapped to vehicle slot 0 (single-vehicle evaluation) or
        # whatever slot matches the actor ordering.
        return [route]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_actors(self) -> List[int]:
        """Find integer-named subdirectories (actor IDs)."""
        actors = []
        for p in self.scenario_dir.iterdir():
            if p.is_dir() and re.fullmatch(r"\d+", p.name):
                actors.append(int(p.name))
        return sorted(actors)

    def _count_frames(self, actor_id: int) -> int:
        actor_dir = self.scenario_dir / str(actor_id)
        # Count contiguous frame indices from 000000 upward where both YAML and
        # LiDAR files exist. This is robust to partially downloaded tails.
        n = 0
        while True:
            base = actor_dir / f"{n:06d}"
            if not (base.with_suffix(".yaml").exists() and base.with_suffix(".bin").exists()):
                break
            n += 1
        return n


# ---------------------------------------------------------------------------
# Multi-scenario collection
# ---------------------------------------------------------------------------

class V2XPnPOpenLoopDataset:
    """
    Discovers and iterates over multiple v2xpnp scenario directories.

    Parameters
    ----------
    root_dir : str or Path
        Parent directory containing scenario subdirs.  Each subdir that
        contains at least one integer-named sub-subdirectory is treated as a
        scenario.
    scenario_names : list[str] or None
        Explicit list of scenario names to include.  If None, all valid
        subdirectories are included.
    """

    def __init__(self, root_dir, scenario_names: Optional[List[str]] = None):
        self.root_dir = Path(root_dir)
        if scenario_names is not None:
            self._scenario_dirs = [self.root_dir / n for n in scenario_names]
        else:
            self._scenario_dirs = sorted([
                p for p in self.root_dir.iterdir()
                if p.is_dir() and _has_actor_subdirs(p)
            ])

    def __len__(self) -> int:
        return len(self._scenario_dirs)

    def __getitem__(self, idx: int) -> V2XPnPScenario:
        return V2XPnPScenario(self._scenario_dirs[idx])

    def iter_scenarios(self):
        for sd in self._scenario_dirs:
            yield V2XPnPScenario(sd)


def _has_actor_subdirs(path: Path) -> bool:
    return any(
        p.is_dir() and re.fullmatch(r"\d+", p.name)
        for p in path.iterdir()
    )
