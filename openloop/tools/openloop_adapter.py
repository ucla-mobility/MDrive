"""
openloop_adapter.py
===================
CARLA-free adapter for open-loop evaluation of leaderboard planners.

Usage (must be called BEFORE any planner imports):

    from openloop.tools.openloop_adapter import install_openloop_stubs
    install_openloop_stubs(vector_map_path)  # installs the CARLA/leaderboard stubs

After this call, any planner agent can be imported and instantiated normally.
The adapter:
  - Provides all carla.* data-structure classes without a server connection
  - Installs BirdViewProducer / BirdViewCropType / PixelDimensions stubs
    up front so planners can import their normal CARLA-facing modules without
    any evaluator-side monkeypatching
  - Replaces CarlaDataProvider with a lightweight registry whose "hero actors"
    are MockVehicle objects whose poses are updated each frame by the eval loop
  - Stubs out RoadSideUnit / get_rsu_point so RSU collection is a no-op

No planner file is modified.
"""
from __future__ import annotations

import math
import os
import sys
import types
import fnmatch
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Build 3x3 rotation matrix from Tait-Bryan angles (radians, ZYX order).

    CARLA's convention: yaw = rotation around Z, pitch around Y, roll around X,
    applied in that order (Z then Y then X).
    """
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    return np.array([
        [cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr],
        [sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr],
        [-sp,      cp * sr,                  cp * cr               ],
    ])


def _pose_to_matrix4(x: float, y: float, z: float,
                     roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return a 4x4 homogeneous transform matrix from a 6-DOF pose.

    Angles are in radians.  Matches carla.Transform.get_matrix().
    """
    M = np.eye(4)
    M[:3, :3] = _euler_to_rotation_matrix(roll, pitch, yaw)
    M[:3, 3]  = [x, y, z]
    return M


# ---------------------------------------------------------------------------
# Minimal CARLA data-structure stubs
# ---------------------------------------------------------------------------

class _Location:
    """Mirrors carla.Location."""
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        # Support CARLA-like overloaded construction:
        #   Location(), Location(x,y,z), Location(Vector3D/Location)
        if hasattr(x, "x") and hasattr(x, "y"):
            z_val = getattr(x, "z", z)
            self.x = float(getattr(x, "x", 0.0))
            self.y = float(getattr(x, "y", 0.0))
            self.z = float(z_val)
        else:
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)

    def distance(self, other: "_Location") -> float:
        return math.sqrt((self.x - other.x) ** 2 +
                         (self.y - other.y) ** 2 +
                         (self.z - other.z) ** 2)

    def __repr__(self):
        return f"Location(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f})"

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __add__(self, other):
        ox = getattr(other, "x", 0.0)
        oy = getattr(other, "y", 0.0)
        oz = getattr(other, "z", 0.0)
        return _Location(self.x + ox, self.y + oy, self.z + oz)

    def __sub__(self, other):
        ox = getattr(other, "x", 0.0)
        oy = getattr(other, "y", 0.0)
        oz = getattr(other, "z", 0.0)
        return _Location(self.x - ox, self.y - oy, self.z - oz)

    def __mul__(self, scalar: float):
        return _Location(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float):
        return self.__mul__(scalar)


class _Rotation:
    """Mirrors carla.Rotation.  All angles stored in DEGREES (CARLA convention)."""
    __slots__ = ("roll", "pitch", "yaw")

    def __init__(self, roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0):
        if hasattr(roll, "roll") and hasattr(roll, "pitch") and hasattr(roll, "yaw"):
            self.roll = float(getattr(roll, "roll", 0.0))
            self.pitch = float(getattr(roll, "pitch", 0.0))
            self.yaw = float(getattr(roll, "yaw", 0.0))
        else:
            self.roll  = float(roll)
            self.pitch = float(pitch)
            self.yaw   = float(yaw)

    def __repr__(self):
        return f"Rotation(roll={self.roll:.2f}, pitch={self.pitch:.2f}, yaw={self.yaw:.2f})"

    def get_forward_vector(self):
        yaw = math.radians(self.yaw)
        pitch = math.radians(self.pitch)
        return _Vector3D(
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            math.sin(pitch),
        )

    def get_right_vector(self):
        yaw = math.radians(self.yaw + 90.0)
        pitch = math.radians(self.pitch)
        return _Vector3D(
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            math.sin(pitch),
        )

    def get_up_vector(self):
        # Up vector from full rotation matrix.
        R = _euler_to_rotation_matrix(
            roll=math.radians(self.roll),
            pitch=math.radians(self.pitch),
            yaw=math.radians(self.yaw),
        )
        v = R[:, 2]
        return _Vector3D(v[0], v[1], v[2])


class _Transform:
    """Mirrors carla.Transform."""
    __slots__ = ("location", "rotation")

    def __init__(self, location: Optional[_Location] = None,
                 rotation: Optional[_Rotation] = None):
        self.location = _Location(location) if location is not None else _Location()
        self.rotation = _Rotation(rotation) if rotation is not None else _Rotation()

    # carla uses degrees; convert when building the matrix
    def _yaw_rad(self) -> float:
        return math.radians(self.rotation.yaw)

    def _pitch_rad(self) -> float:
        return math.radians(self.rotation.pitch)

    def _roll_rad(self) -> float:
        return math.radians(self.rotation.roll)

    def get_matrix(self) -> List[List[float]]:
        M = _pose_to_matrix4(
            self.location.x, self.location.y, self.location.z,
            self._roll_rad(), self._pitch_rad(), self._yaw_rad(),
        )
        return M.tolist()

    def get_inverse_matrix(self) -> List[List[float]]:
        return np.linalg.inv(np.array(self.get_matrix())).tolist()

    def transform(self, location: _Location) -> None:
        """Transform *location* from local to world frame (in-place)."""
        M = np.array(self.get_matrix())
        p = M @ np.array([location.x, location.y, location.z, 1.0])
        location.x, location.y, location.z = float(p[0]), float(p[1]), float(p[2])

    def get_right_vector(self):
        return self.rotation.get_right_vector()

    def get_up_vector(self):
        return self.rotation.get_up_vector()

    def __repr__(self):
        return f"Transform(loc={self.location}, rot={self.rotation})"

    def get_forward_vector(self):
        return self.rotation.get_forward_vector()


class _VehicleControl:
    """Mirrors carla.VehicleControl."""
    __slots__ = ("throttle", "steer", "brake",
                 "hand_brake", "reverse", "manual_gear_shift", "gear")

    def __init__(self, throttle: float = 0.0, steer: float = 0.0,
                 brake: float = 0.0, hand_brake: bool = False,
                 reverse: bool = False, manual_gear_shift: bool = False,
                 gear: int = 1):
        self.throttle          = float(throttle)
        self.steer             = float(steer)
        self.brake             = float(brake)
        self.hand_brake        = bool(hand_brake)
        self.reverse           = bool(reverse)
        self.manual_gear_shift = bool(manual_gear_shift)
        self.gear              = int(gear)


class _Vector2D:
    __slots__ = ("x", "y")
    def __init__(self, x=0.0, y=0.0):
        self.x, self.y = float(x), float(y)


class _Vector3D:
    __slots__ = ("x", "y", "z")
    def __init__(self, x=0.0, y=0.0, z=0.0):
        if hasattr(x, "x") and hasattr(x, "y"):
            self.x = float(getattr(x, "x", 0.0))
            self.y = float(getattr(x, "y", 0.0))
            self.z = float(getattr(x, "z", z))
        else:
            self.x, self.y, self.z = float(x), float(y), float(z)

    def __add__(self, other):
        ox = getattr(other, "x", 0.0)
        oy = getattr(other, "y", 0.0)
        oz = getattr(other, "z", 0.0)
        return _Vector3D(self.x + ox, self.y + oy, self.z + oz)

    def __sub__(self, other):
        ox = getattr(other, "x", 0.0)
        oy = getattr(other, "y", 0.0)
        oz = getattr(other, "z", 0.0)
        return _Vector3D(self.x - ox, self.y - oy, self.z - oz)

    def __mul__(self, scalar: float):
        return _Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float):
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float):
        s = float(scalar)
        if abs(s) < 1e-12:
            return _Vector3D(self.x, self.y, self.z)
        return _Vector3D(self.x / s, self.y / s, self.z / s)

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z


class _BoundingBox:
    __slots__ = ("location", "extent", "rotation")
    def __init__(self, location=None, extent=None):
        self.location = location or _Location()
        self.extent   = extent   or _Vector3D()
        self.rotation = _Rotation()


class _WalkerControl:
    __slots__ = ("direction", "speed", "jump")
    def __init__(self, direction=None, speed: float = 0.0, jump: bool = False):
        self.direction = direction or _Vector3D(1.0, 0.0, 0.0)
        self.speed = float(speed)
        self.jump = bool(jump)


class _WeatherParameters:
    _FIELDS = (
        "cloudiness", "precipitation", "precipitation_deposits",
        "wind_intensity", "sun_azimuth_angle", "sun_altitude_angle",
        "fog_density", "fog_distance", "wetness", "fog_falloff",
    )

    def __init__(self, *args, **kwargs):
        defaults = {
            "cloudiness": 0.0,
            "precipitation": 0.0,
            "precipitation_deposits": 0.0,
            "wind_intensity": 0.0,
            "sun_azimuth_angle": 0.0,
            "sun_altitude_angle": 70.0,
            "fog_density": 0.0,
            "fog_distance": 75.0,
            "wetness": 0.0,
            "fog_falloff": 1.0,
        }
        for idx, val in enumerate(args):
            if idx >= len(self._FIELDS):
                break
            defaults[self._FIELDS[idx]] = float(val)
        for k, v in kwargs.items():
            if k in defaults:
                defaults[k] = float(v)
        for k, v in defaults.items():
            setattr(self, k, float(v))

    def __repr__(self):
        return (
            f"WeatherParameters(cloudiness={self.cloudiness}, "
            f"precipitation={self.precipitation}, "
            f"sun_altitude_angle={self.sun_altitude_angle})"
        )


class _CommandBase:
    def then(self, _next):
        return self


class _DestroyActorCommand(_CommandBase):
    def __init__(self, actor):
        self.actor = actor
        self.actor_id = getattr(actor, "id", None)


class _SpawnActorCommand(_CommandBase):
    def __init__(self, blueprint, transform):
        self.blueprint = blueprint
        self.transform = transform
        self.actor_id = None


class _SetAutopilotCommand(_CommandBase):
    def __init__(self, actor, enabled=True):
        self.actor = actor
        self.enabled = enabled


class _SetSimulatePhysicsCommand(_CommandBase):
    def __init__(self, actor, enabled=True):
        self.actor = actor
        self.enabled = enabled


class _ApplyTransformCommand(_CommandBase):
    def __init__(self, actor, transform):
        self.actor = actor
        self.transform = transform


class _OpenLoopGameTime:
    _t = 0.0
    _frame = 0
    _wallclock_t0 = None

    @classmethod
    def restart(cls):
        cls._t = 0.0
        cls._frame = 0
        cls._wallclock_t0 = None

    @classmethod
    def set_time(cls, timestamp_s: float, frame_idx: int, delta_seconds: float | None = None):
        cls._t = float(timestamp_s)
        cls._frame = int(frame_idx)
        if cls._wallclock_t0 is None:
            import datetime as _dt
            cls._wallclock_t0 = _dt.datetime.now()

    @classmethod
    def on_carla_tick(cls, timestamp):
        cls._frame += 1
        try:
            cls._t = float(getattr(timestamp, "elapsed_seconds"))
        except Exception:
            try:
                cls._t += float(getattr(timestamp, "delta_seconds"))
            except Exception:
                cls._t += 0.05

    @classmethod
    def get_time(cls):
        return float(cls._t)

    @classmethod
    def get_carla_time(cls):
        return float(cls._t)

    @classmethod
    def get_wallclocktime(cls):
        import datetime as _dt
        if cls._wallclock_t0 is None:
            cls._wallclock_t0 = _dt.datetime.now()
        return _dt.datetime.now()

    @classmethod
    def get_frame(cls):
        return int(cls._frame)


def _ensure_module(name: str, package: bool = False) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    if package and not hasattr(mod, "__path__"):
        mod.__path__ = []  # type: ignore[attr-defined]
    return mod


def _normalize_yaw_delta_deg(delta_deg: float) -> float:
    return (float(delta_deg) + 180.0) % 360.0 - 180.0


def _resolve_runtime_delta_seconds(
    timestamp_s: float | None,
    delta_seconds: float | None,
    previous_timestamp_s: float | None,
) -> float | None:
    if delta_seconds is not None:
        try:
            dt = float(delta_seconds)
            return dt if dt > 0.0 else None
        except Exception:
            return None
    if timestamp_s is not None and previous_timestamp_s is not None:
        try:
            dt = float(timestamp_s) - float(previous_timestamp_s)
            return dt if dt > 0.0 else None
        except Exception:
            return None
    return None


def _set_actor_runtime_state(
    actor,
    transform: _Transform,
    *,
    timestamp_s: float | None = None,
    delta_seconds: float | None = None,
    speed_mps: float | None = None,
    acceleration_mps2: float | None = None,
    yaw_rate_rad_s: float | None = None,
) -> None:
    prev_transform = getattr(actor, "_last_transform", None)
    prev_velocity = getattr(actor, "_velocity", _Vector3D())
    prev_timestamp = getattr(actor, "_last_timestamp_s", None)
    dt = _resolve_runtime_delta_seconds(timestamp_s, delta_seconds, prev_timestamp)

    if speed_mps is not None:
        forward = transform.get_forward_vector()
        velocity = _Vector3D(forward.x, forward.y, forward.z) * float(speed_mps)
    elif prev_transform is not None and dt is not None:
        velocity = _Vector3D(
            (transform.location.x - prev_transform.location.x) / dt,
            (transform.location.y - prev_transform.location.y) / dt,
            (transform.location.z - prev_transform.location.z) / dt,
        )
    else:
        velocity = _Vector3D()

    if acceleration_mps2 is not None:
        forward = transform.get_forward_vector()
        acceleration = _Vector3D(forward.x, forward.y, forward.z) * float(acceleration_mps2)
    elif prev_transform is not None and dt is not None:
        acceleration = (velocity - prev_velocity) / dt
    else:
        acceleration = _Vector3D()

    if yaw_rate_rad_s is not None:
        angular_velocity = _Vector3D(0.0, 0.0, float(yaw_rate_rad_s))
    elif prev_transform is not None and dt is not None:
        delta_yaw_deg = _normalize_yaw_delta_deg(
            float(transform.rotation.yaw) - float(prev_transform.rotation.yaw)
        )
        angular_velocity = _Vector3D(0.0, 0.0, math.radians(delta_yaw_deg) / dt)
    else:
        angular_velocity = _Vector3D()

    actor._transform = transform
    actor._last_transform = transform
    actor._velocity = velocity
    actor._acceleration = acceleration
    actor._angular_velocity = angular_velocity
    if timestamp_s is not None:
        actor._last_timestamp_s = float(timestamp_s)
    elif prev_timestamp is not None and dt is not None:
        actor._last_timestamp_s = float(prev_timestamp) + float(dt)
    else:
        actor._last_timestamp_s = prev_timestamp


def _set_actor_transform(
    actor,
    location: _Location,
    rotation: _Rotation,
    *,
    timestamp_s: float | None = None,
    delta_seconds: float | None = None,
    speed_mps: float | None = None,
    acceleration_mps2: float | None = None,
    yaw_rate_rad_s: float | None = None,
) -> None:
    transform = _Transform(location=location, rotation=rotation)
    _set_actor_runtime_state(
        actor,
        transform,
        timestamp_s=timestamp_s,
        delta_seconds=delta_seconds,
        speed_mps=speed_mps,
        acceleration_mps2=acceleration_mps2,
        yaw_rate_rad_s=yaw_rate_rad_s,
    )


def _ensure_birdview_stub_modules() -> None:
    birdview_mod = _ensure_module("carla_birdeye_view", package=True)
    mask_mod = _ensure_module("carla_birdeye_view.mask")
    import importlib
    try:
        team_utils_mod = importlib.import_module("team_code.utils")
    except Exception:
        team_utils_mod = _ensure_module("team_code.utils", package=True)
    team_birdview_mod = _ensure_module("team_code.utils.carla_birdeye_view")

    birdview_mod.BirdViewProducer = _MockBirdViewProducer
    birdview_mod.BirdViewCropType = _BirdViewCropType
    birdview_mod.PixelDimensions = _PixelDimensions
    birdview_mod.DEFAULT_WIDTH = 400
    birdview_mod.DEFAULT_HEIGHT = 400
    birdview_mod.BirdView = np.ndarray
    birdview_mod.RgbCanvas = np.ndarray
    birdview_mod.__all__ = [
        "BirdViewProducer",
        "BirdViewCropType",
        "PixelDimensions",
        "DEFAULT_WIDTH",
        "DEFAULT_HEIGHT",
    ]

    mask_mod.PixelDimensions = _PixelDimensions
    mask_mod.__all__ = ["PixelDimensions"]

    team_birdview_mod.BirdViewProducer = _MockBirdViewProducer
    team_birdview_mod.BirdViewCropType = _BirdViewCropType
    team_birdview_mod.PixelDimensions = _PixelDimensions
    team_birdview_mod.DEFAULT_WIDTH = 400
    team_birdview_mod.DEFAULT_HEIGHT = 400
    team_birdview_mod.BirdView = np.ndarray
    team_birdview_mod.RgbCanvas = np.ndarray
    team_birdview_mod.__all__ = list(birdview_mod.__all__)

    birdview_mod.mask = mask_mod
    team_birdview_mod.mask = mask_mod
    team_utils_mod.carla_birdeye_view = team_birdview_mod

    sys.modules["carla_birdeye_view"] = birdview_mod
    sys.modules["carla_birdeye_view.mask"] = mask_mod
    sys.modules["team_code.utils.carla_birdeye_view"] = team_birdview_mod


def _sync_runtime_clock(frame_idx: int, timestamp_s: float, delta_seconds: float) -> None:
    _OpenLoopGameTime.set_time(timestamp_s, frame_idx, delta_seconds)
    world = _MockCarlaDataProvider._world
    if world is not None:
        try:
            world._tick_frame = int(frame_idx)
            if hasattr(world, "_settings") and world._settings is not None:
                world._settings.fixed_delta_seconds = float(delta_seconds)
        except Exception:
            pass
    _MockCarlaDataProvider._runtime_frame_idx = int(frame_idx)
    _MockCarlaDataProvider._runtime_timestamp_s = float(timestamp_s)
    _MockCarlaDataProvider._runtime_delta_seconds = float(delta_seconds)
    try:
        _MockCarlaDataProvider.on_carla_tick()
    except Exception:
        pass


def _build_carla_module() -> types.ModuleType:
    """Build a synthetic `carla` module containing the stub classes."""
    mod = types.ModuleType("carla")
    mod.__file__       = "<openloop_carla_stub>"
    mod.Location       = _Location
    mod.Rotation       = _Rotation
    mod.Transform      = _Transform
    mod.VehicleControl = _VehicleControl
    mod.Vector2D       = _Vector2D
    mod.Vector3D       = _Vector3D
    mod.BoundingBox    = _BoundingBox
    mod.WalkerControl  = _WalkerControl
    # Type placeholders used by annotations in planner utility modules.
    mod.Actor          = object
    mod.Vehicle        = object
    mod.Waypoint       = object
    mod.TrafficLight   = object
    mod.WorldSnapshot  = object
    # Colour / misc stubs
    mod.Color          = lambda r=0, g=0, b=0, a=255: (r, g, b, a)
    mod.Map            = object  # never constructed in open-loop path
    mod.World          = object
    mod.Client         = object
    # Enum-like placeholders occasionally referenced by helper modules.
    mod.TrafficLightState = types.SimpleNamespace(
        Red=0, Yellow=1, Green=2, Off=3, Unknown=4
    )
    mod.LaneType = types.SimpleNamespace(
        NONE=0,
        Driving=1,
        Shoulder=2,
        Sidewalk=4,
        Parking=8,
        Bidirectional=16,
        Any=255,
    )
    mod.LaneChange = types.SimpleNamespace(NONE=0, Left=1, Right=2, Both=3)
    mod.LaneMarkingType   = types.SimpleNamespace(
        NONE=0, Broken=1, Solid=2,
        SolidBroken=3, BrokenSolid=4, BrokenBroken=5, SolidSolid=6,
    )
    mod.LaneMarkingColor  = types.SimpleNamespace(Other=0)
    mod.AttachmentType = types.SimpleNamespace(Rigid=0, SpringArm=1)
    mod.VehicleLightState = types.SimpleNamespace(
        NONE=0,
        Position=1,
        LowBeam=2,
        HighBeam=4,
        Brake=8,
        RightBlinker=16,
        LeftBlinker=32,
        Reverse=64,
        Fog=128,
        Interior=256,
        Special1=512,
        Special2=1024,
        All=2047,
    )
    mod.ColorConverter = types.SimpleNamespace(Raw=0, CityScapesPalette=1, LogarithmicDepth=2)
    mod.CityObjectLabel = types.SimpleNamespace(
        None_=0,
        Buildings=1,
        Fences=2,
        Other=3,
        Pedestrians=4,
        Poles=5,
        RoadLines=6,
        Roads=7,
        Sidewalks=8,
        TrafficSigns=9,
        Vegetation=10,
        TrafficLight=11,
        Static=12,
        Dynamic=13,
        Walls=14,
        Terrain=15,
    )
    setattr(mod.CityObjectLabel, "None", 0)
    mod.WeatherParameters = _WeatherParameters
    _weather_presets = {
        "ClearNoon": dict(cloudiness=10, precipitation=0, wetness=0, sun_altitude_angle=75),
        "ClearSunset": dict(cloudiness=15, precipitation=0, wetness=0, sun_altitude_angle=15),
        "CloudyNoon": dict(cloudiness=80, precipitation=0, wetness=0, sun_altitude_angle=75),
        "CloudySunset": dict(cloudiness=80, precipitation=0, wetness=0, sun_altitude_angle=15),
        "WetNoon": dict(cloudiness=20, precipitation=0, wetness=60, sun_altitude_angle=75),
        "WetSunset": dict(cloudiness=20, precipitation=0, wetness=60, sun_altitude_angle=15),
        "MidRainyNoon": dict(cloudiness=80, precipitation=60, wetness=80, sun_altitude_angle=75),
        "MidRainSunset": dict(cloudiness=80, precipitation=60, wetness=80, sun_altitude_angle=15),
        "WetCloudyNoon": dict(cloudiness=90, precipitation=0, wetness=85, sun_altitude_angle=75),
        "WetCloudySunset": dict(cloudiness=90, precipitation=0, wetness=85, sun_altitude_angle=15),
        "HardRainNoon": dict(cloudiness=100, precipitation=100, wetness=100, sun_altitude_angle=75),
        "HardRainSunset": dict(cloudiness=100, precipitation=100, wetness=100, sun_altitude_angle=15),
        "SoftRainNoon": dict(cloudiness=70, precipitation=35, wetness=70, sun_altitude_angle=75),
        "SoftRainSunset": dict(cloudiness=70, precipitation=35, wetness=70, sun_altitude_angle=15),
    }
    for _name, _kwargs in _weather_presets.items():
        setattr(mod.WeatherParameters, _name, _WeatherParameters(**_kwargs))
    # Some planners check carla.libcarla.TrafficLightState.*
    mod.libcarla = types.SimpleNamespace(TrafficLightState=mod.TrafficLightState)
    mod.command = types.SimpleNamespace(
        SpawnActor=_SpawnActorCommand,
        SetAutopilot=_SetAutopilotCommand,
        SetSimulatePhysics=_SetSimulatePhysicsCommand,
        FutureActor=object(),
        ApplyTransform=_ApplyTransformCommand,
        DestroyActor=_DestroyActorCommand,
    )
    return mod


# ---------------------------------------------------------------------------
# MockVehicle — per-frame pose is injected by the eval loop
# ---------------------------------------------------------------------------

class MockVehicle:
    """
    Lightweight stand-in for a carla.Actor / carla.Vehicle.

    The eval loop calls ``update_pose()`` before every ``agent.run_step()``,
    so that any code inside the agent that calls ``get_transform()`` receives
    the correct world-frame pose for the current dataset frame.
    """

    def __init__(self, hero_id: int = 0):
        self.id      = hero_id
        self.is_alive = True
        self.type_id = "vehicle.ego"
        self.attributes = {"role_name": "hero" if int(hero_id) == 0 else f"hero_{int(hero_id)}"}
        self.bounding_box = _BoundingBox(extent=_Vector3D(2.3, 1.0, 0.8))
        self._transform = _Transform()
        self._velocity = _Vector3D()
        self._acceleration = _Vector3D()
        self._angular_velocity = _Vector3D()
        self._last_transform = None
        self._last_timestamp_s = None
        self._control = _VehicleControl()

    def update_runtime_state(
        self,
        *,
        x: float,
        y: float,
        z: float,
        roll_rad: float,
        pitch_rad: float,
        yaw_rad: float,
        timestamp_s: float | None = None,
        delta_seconds: float | None = None,
        speed_mps: float | None = None,
        acceleration_mps2: float | None = None,
        yaw_rate_rad_s: float | None = None,
    ) -> None:
        """Update pose and optional kinematics before agent.run_step()."""
        _set_actor_transform(
            self,
            _Location(x, y, z),
            _Rotation(
                roll=math.degrees(roll_rad),
                pitch=math.degrees(pitch_rad),
                yaw=math.degrees(yaw_rad),
            ),
            timestamp_s=timestamp_s,
            delta_seconds=delta_seconds,
            speed_mps=speed_mps,
            acceleration_mps2=acceleration_mps2,
            yaw_rate_rad_s=yaw_rate_rad_s,
        )

    def update_pose(
        self,
        x: float,
        y: float,
        z: float,
        roll_rad: float,
        pitch_rad: float,
        yaw_rad: float,
        timestamp_s: float | None = None,
        delta_seconds: float | None = None,
        speed_mps: float | None = None,
        acceleration_mps2: float | None = None,
        yaw_rate_rad_s: float | None = None,
    ):
        """Call this each frame before agent.run_step()."""
        self.update_runtime_state(
            x=x,
            y=y,
            z=z,
            roll_rad=roll_rad,
            pitch_rad=pitch_rad,
            yaw_rad=yaw_rad,
            timestamp_s=timestamp_s,
            delta_seconds=delta_seconds,
            speed_mps=speed_mps,
            acceleration_mps2=acceleration_mps2,
            yaw_rate_rad_s=yaw_rate_rad_s,
        )

    def get_transform(self) -> _Transform:
        return self._transform

    def get_location(self) -> _Location:
        return self._transform.location

    def get_velocity(self) -> _Vector3D:
        return self._velocity

    def get_acceleration(self) -> _Vector3D:
        return self._acceleration

    def get_angular_velocity(self) -> _Vector3D:
        return self._angular_velocity

    def get_bounding_box(self) -> _BoundingBox:
        return self.bounding_box

    def get_world(self):
        return _MockCarlaDataProvider.get_world()

    def apply_control(self, control) -> None:
        if control is not None:
            self._control = control

    def get_control(self):
        return self._control

    def is_junction(self) -> bool:
        return False

    def get_traffic_light_state(self):
        return 2

    def get_traffic_light(self):
        return None

    def get_speed_limit(self) -> float:
        return 30.0

    def set_autopilot(self, enabled=True):
        return None

    def destroy(self):
        self.is_alive = False

    def set_autopilot(self, enabled=True):
        return None


class _MockActor:
    """Generic surrounding actor stub (vehicle / walker)."""
    def __init__(self, actor_id: int, type_id: str,
                 location: _Location, rotation: _Rotation,
                 extent_xyz: List[float],
                 attributes: Optional[Dict[str, str]] = None,
                 source_tag: str = "spawned"):
        self.id = int(actor_id)
        self.type_id = str(type_id)
        self.is_alive = True
        self.attributes = dict(attributes or {})
        self._transform = _Transform(location=location, rotation=rotation)
        self._velocity = _Vector3D()
        self._acceleration = _Vector3D()
        self._angular_velocity = _Vector3D()
        self._last_transform = None
        self._last_timestamp_s = None
        self._source_tag = str(source_tag)
        ex = float(extent_xyz[0]) if len(extent_xyz) > 0 else 1.0
        ey = float(extent_xyz[1]) if len(extent_xyz) > 1 else 0.5
        ez = float(extent_xyz[2]) if len(extent_xyz) > 2 else 0.8
        self.bounding_box = _BoundingBox(extent=_Vector3D(ex, ey, ez))
        self._control = _VehicleControl()

    def update_runtime_state(
        self,
        *,
        location: _Location,
        rotation: _Rotation,
        timestamp_s: float | None = None,
        delta_seconds: float | None = None,
        speed_mps: float | None = None,
        acceleration_mps2: float | None = None,
        yaw_rate_rad_s: float | None = None,
    ) -> None:
        _set_actor_transform(
            self,
            location,
            rotation,
            timestamp_s=timestamp_s,
            delta_seconds=delta_seconds,
            speed_mps=speed_mps,
            acceleration_mps2=acceleration_mps2,
            yaw_rate_rad_s=yaw_rate_rad_s,
        )

    def update_pose(
        self,
        x: float,
        y: float,
        z: float,
        roll_rad: float,
        pitch_rad: float,
        yaw_rad: float,
        timestamp_s: float | None = None,
        delta_seconds: float | None = None,
        speed_mps: float | None = None,
        acceleration_mps2: float | None = None,
        yaw_rate_rad_s: float | None = None,
    ) -> None:
        self.update_runtime_state(
            location=_Location(x, y, z),
            rotation=_Rotation(
                roll=math.degrees(roll_rad),
                pitch=math.degrees(pitch_rad),
                yaw=math.degrees(yaw_rad),
            ),
            timestamp_s=timestamp_s,
            delta_seconds=delta_seconds,
            speed_mps=speed_mps,
            acceleration_mps2=acceleration_mps2,
            yaw_rate_rad_s=yaw_rate_rad_s,
        )

    def get_location(self):
        return self._transform.location

    def get_transform(self):
        return self._transform

    def get_velocity(self):
        return self._velocity

    def get_acceleration(self):
        return self._acceleration

    def get_angular_velocity(self):
        return self._angular_velocity

    def get_world(self):
        return _MockCarlaDataProvider.get_world()

    def apply_control(self, control):
        if control is not None:
            self._control = control

    def get_control(self):
        return self._control

    def get_speed_limit(self) -> float:
        return 30.0

    def destroy(self):
        self.is_alive = False
        try:
            _MockCarlaDataProvider._dynamic_actors = [
                a for a in _MockCarlaDataProvider._dynamic_actors
                if getattr(a, "id", None) != self.id
            ]
        except Exception:
            pass


class _MockActorCollection(list):
    """List-like CARLA actor container with wildcard filtering."""
    def filter(self, pattern: str):
        return _MockActorCollection(
            [a for a in self if fnmatch.fnmatch(getattr(a, "type_id", ""), pattern)]
        )

    def find(self, actor_id: int):
        for a in self:
            if int(getattr(a, "id", -1)) == int(actor_id):
                return a
        return None


class _MockAttributeValue:
    __slots__ = ("_value", "recommended_values")
    def __init__(self, value):
        self._value = str(value)
        self.recommended_values = []

    def as_str(self):
        return str(self._value)

    def as_int(self):
        try:
            return int(float(self._value))
        except Exception:
            return 0

    def as_float(self):
        try:
            return float(self._value)
        except Exception:
            return 0.0

    def as_bool(self):
        return str(self._value).lower() in ("1", "true", "yes", "on")

    def as_color(self):
        txt = str(self._value).strip().replace("(", "").replace(")", "")
        parts = [p.strip() for p in txt.split(",") if p.strip()]
        try:
            r = int(float(parts[0])) if len(parts) > 0 else 0
            g = int(float(parts[1])) if len(parts) > 1 else 0
            b = int(float(parts[2])) if len(parts) > 2 else 0
        except Exception:
            r, g, b = 0, 0, 0
        return types.SimpleNamespace(r=r, g=g, b=b)

    def __str__(self):
        return str(self._value)


class _MockBlueprint:
    __slots__ = ("id", "attributes")
    def __init__(self, bp_id: str):
        self.id = str(bp_id)
        self.attributes: Dict[str, str] = {}

    def set_attribute(self, key: str, value: str) -> None:
        self.attributes[str(key)] = str(value)

    def get_attribute(self, key: str):
        return _MockAttributeValue(self.attributes.get(str(key), ""))

    def has_attribute(self, key: str) -> bool:
        return str(key) in self.attributes


class _MockBlueprintLibrary(list):
    def __init__(self):
        super().__init__()
        for bid in [
            "sensor.camera.rgb",
            "sensor.lidar.ray_cast",
            "vehicle.tesla.model3",
            "vehicle.audi.tt",
            "walker.pedestrian.0001",
        ]:
            self.append(_MockBlueprint(bid))

    def find(self, bp_id: str) -> _MockBlueprint:
        bp = _MockBlueprint(bp_id)
        if bp_id == "sensor.camera.rgb":
            bp.set_attribute("image_size_x", "1600")
            bp.set_attribute("image_size_y", "900")
            bp.set_attribute("fov", "90")
        return bp

    def filter(self, pattern: str):
        return [bp for bp in self if fnmatch.fnmatch(bp.id, pattern)]


class _MockImage:
    __slots__ = ("width", "height", "raw_data")
    def __init__(self, width: int = 800, height: int = 600):
        self.width = int(max(1, width))
        self.height = int(max(1, height))
        self.raw_data = bytes(self.width * self.height * 4)


class _MockSensorActor:
    _next_id = 900000

    def __init__(self, blueprint: _MockBlueprint, transform: _Transform,
                 parent, world):
        self.id = _MockSensorActor._next_id
        _MockSensorActor._next_id += 1
        self.type_id = blueprint.id
        self.attributes = dict(getattr(blueprint, "attributes", {}))
        self.attributes.setdefault("role_name", "sensor")
        self._transform = transform or _Transform()
        self._parent = parent
        self._world = world
        self._callbacks = []
        self.is_alive = True

    def get_world(self):
        return self._world

    def get_transform(self):
        return self._transform

    def get_location(self):
        return self._transform.location

    def set_transform(self, transform):
        self._transform = _Transform(
            location=getattr(transform, "location", _Location()),
            rotation=getattr(transform, "rotation", _Rotation()),
        )

    def listen(self, callback):
        if callback is None:
            return
        self._callbacks.append(callback)
        # Emit one frame immediately so callback-dependent pipelines do not block.
        self._emit_once()

    def stop(self):
        self._callbacks = []

    def destroy(self):
        self.stop()
        self.is_alive = False
        try:
            self._world._spawned_sensors = [
                s for s in self._world._spawned_sensors if getattr(s, "id", None) != self.id
            ]
        except Exception:
            pass

    def _emit_once(self):
        if not self._callbacks:
            return
        w = int(float(self.attributes.get("image_size_x", 800)))
        h = int(float(self.attributes.get("image_size_y", 600)))
        img = _MockImage(width=w, height=h)
        for cb in list(self._callbacks):
            try:
                cb(img)
            except Exception:
                pass


class _MockWaypoint:
    def __init__(self, location: _Location, yaw_deg: float = 0.0,
                 road_id: int = 1, lane_id: int = 1, s: float = 0.0):
        self.transform = _Transform(location=_Location(location.x, location.y, location.z),
                                    rotation=_Rotation(yaw=float(yaw_deg)))
        self.is_junction = False
        self.is_intersection = False
        self.road_id = int(road_id)
        self.lane_id = int(lane_id)
        self.lane_width = 3.5
        self.lane_type = 1  # carla.LaneType.Driving
        self.lane_change = 0
        self.left_lane_marking = types.SimpleNamespace(type=1, color=0)
        self.right_lane_marking = types.SimpleNamespace(type=1, color=0)
        self.s = float(s)
        self.section_id = 0

    def next(self, _distance: float):
        d = float(_distance) if _distance is not None else 1.0
        if d <= 0:
            d = 1.0
        # finite horizon avoids infinite loops in map traversal utilities
        if self.s > 1000.0:
            return []
        yaw = math.radians(self.transform.rotation.yaw)
        nx = self.transform.location.x + d * math.cos(yaw)
        ny = self.transform.location.y + d * math.sin(yaw)
        return [_MockWaypoint(
            location=_Location(nx, ny, self.transform.location.z),
            yaw_deg=self.transform.rotation.yaw,
            road_id=self.road_id,
            lane_id=self.lane_id,
            s=self.s + d,
        )]

    def previous(self, _distance: float):
        d = float(_distance) if _distance is not None else 1.0
        if d <= 0:
            d = 1.0
        yaw = math.radians(self.transform.rotation.yaw)
        px = self.transform.location.x - d * math.cos(yaw)
        py = self.transform.location.y - d * math.sin(yaw)
        return [_MockWaypoint(
            location=_Location(px, py, self.transform.location.z),
            yaw_deg=self.transform.rotation.yaw,
            road_id=self.road_id,
            lane_id=self.lane_id,
            s=max(0.0, self.s - d),
        )]

    def get_left_lane(self):
        if abs(self.lane_id) >= 5:
            return None
        sign = 1 if self.lane_id >= 0 else -1
        yaw = math.radians(self.transform.rotation.yaw + 90.0)
        shift = self.lane_width
        lx = self.transform.location.x + sign * shift * math.cos(yaw)
        ly = self.transform.location.y + sign * shift * math.sin(yaw)
        return _MockWaypoint(
            location=_Location(lx, ly, self.transform.location.z),
            yaw_deg=self.transform.rotation.yaw,
            road_id=self.road_id,
            lane_id=self.lane_id + sign,
            s=self.s,
        )

    def get_right_lane(self):
        if abs(self.lane_id) >= 5:
            return None
        sign = 1 if self.lane_id >= 0 else -1
        yaw = math.radians(self.transform.rotation.yaw - 90.0)
        shift = self.lane_width
        rx = self.transform.location.x + sign * shift * math.cos(yaw)
        ry = self.transform.location.y + sign * shift * math.sin(yaw)
        return _MockWaypoint(
            location=_Location(rx, ry, self.transform.location.z),
            yaw_deg=self.transform.rotation.yaw,
            road_id=self.road_id,
            lane_id=self.lane_id - sign,
            s=self.s,
        )

    def get_junction(self):
        return None


class _MockMap:
    name = "openloop_map"

    def get_waypoint(self, location, *args, **kwargs):
        loc = _Location(location) if location is not None else _Location()
        return _MockWaypoint(loc)

    def get_waypoint_xodr(self, road_id, lane_id, s):
        try:
            road_id = int(road_id)
        except Exception:
            road_id = 1
        try:
            lane_id = int(lane_id)
        except Exception:
            lane_id = 1
        try:
            s = float(s)
        except Exception:
            s = 0.0
        return _MockWaypoint(_Location(s, lane_id * 3.5, 0.0),
                             yaw_deg=0.0, road_id=road_id, lane_id=lane_id, s=s)

    def to_opendrive(self):
        return (
            "<OpenDRIVE>"
            "<header revMajor='1' revMinor='4' name='openloop_map' version='1.00'>"
            "<geoReference>+proj=tmerc +lat_0=42 +lon_0=2 +ellps=WGS84 +datum=WGS84 +units=m +no_defs</geoReference>"
            "</header>"
            "</OpenDRIVE>"
        )

    def get_topology(self):
        w0 = _MockWaypoint(_Location(0.0, 0.0, 0.0), road_id=1, lane_id=1, s=0.0)
        w1 = _MockWaypoint(_Location(50.0, 0.0, 0.0), road_id=1, lane_id=1, s=50.0)
        return [(w0, w1)]

    def generate_waypoints(self, _precision):
        precision = float(_precision) if _precision else 2.0
        n = max(10, int(100 / max(precision, 0.5)))
        return [
            _MockWaypoint(_Location(i * precision, 0.0, 0.0), road_id=1, lane_id=1, s=i * precision)
            for i in range(n)
        ]

    def get_spawn_points(self):
        return [_Transform(location=_Location(0.0, 0.0, 0.0), rotation=_Rotation())]


class _MockWorld:
    def __init__(self):
        self._map = _MockMap()
        self._blueprints = _MockBlueprintLibrary()
        self._spawned_sensors: List[_MockSensorActor] = []
        self._tick_callbacks = []
        self._tick_frame = 0
        self._weather = _WeatherParameters()
        self.debug = types.SimpleNamespace(
            draw_point=lambda *a, **kw: None,
            draw_line=lambda *a, **kw: None,
            draw_string=lambda *a, **kw: None,
            draw_box=lambda *a, **kw: None,
            draw_arrow=lambda *a, **kw: None,
        )
        self._settings = types.SimpleNamespace(
            fixed_delta_seconds=0.05,
            synchronous_mode=True,
            no_rendering_mode=True,
        )
        self._spectator = types.SimpleNamespace(
            set_transform=lambda *a, **kw: None,
            get_transform=lambda: _Transform(),
        )

    def get_actors(self, actor_ids=None):
        actors = list(_MockCarlaDataProvider.get_actor_list()) + list(self._spawned_sensors)
        if actor_ids is not None:
            wanted = set(int(x) for x in actor_ids)
            actors = [a for a in actors if int(getattr(a, "id", -1)) in wanted]
        return _MockActorCollection(actors)

    def get_map(self):
        return self._map

    def get_blueprint_library(self):
        return self._blueprints

    def spawn_actor(self, blueprint, transform, attach_to=None, attachment_type=None):
        bp = blueprint if isinstance(blueprint, _MockBlueprint) else _MockBlueprint(str(blueprint))
        tf = transform if isinstance(transform, _Transform) else _Transform()
        if bp.id.startswith("sensor."):
            sensor = _MockSensorActor(bp, tf, attach_to, self)
            self._spawned_sensors.append(sensor)
            return sensor

        base_type = bp.id if "." in bp.id else "vehicle.unknown"
        actor = _MockActor(
            actor_id=1000000 + len(_MockCarlaDataProvider._dynamic_actors),
            type_id=base_type,
            location=tf.location,
            rotation=tf.rotation,
            extent_xyz=[1.0, 0.5, 0.8],
            attributes=dict(getattr(bp, "attributes", {})),
        )
        actor.attributes.setdefault("role_name", "scenario")
        _MockCarlaDataProvider._dynamic_actors.append(actor)
        return actor

    def try_spawn_actor(self, blueprint, transform, attach_to=None, attachment_type=None):
        try:
            return self.spawn_actor(blueprint, transform, attach_to=attach_to,
                                    attachment_type=attachment_type)
        except Exception:
            return None

    def tick(self, timeout: float = 0.0):
        self._tick_frame += 1
        ts = types.SimpleNamespace(
            frame=self._tick_frame,
            elapsed_seconds=self._tick_frame * float(self._settings.fixed_delta_seconds),
            delta_seconds=float(self._settings.fixed_delta_seconds),
        )
        for sensor in list(self._spawned_sensors):
            try:
                sensor._emit_once()
            except Exception:
                pass
        for cb in list(self._tick_callbacks):
            try:
                cb(ts)
            except Exception:
                pass
        try:
            _sync_runtime_clock(
                frame_idx=self._tick_frame,
                timestamp_s=float(ts.elapsed_seconds),
                delta_seconds=float(ts.delta_seconds),
            )
        except Exception:
            pass
        return self._tick_frame

    def wait_for_tick(self, timeout: float = 0.0):
        self.tick(timeout=timeout)
        return types.SimpleNamespace(
            frame=self._tick_frame,
            elapsed_seconds=self._tick_frame * float(self._settings.fixed_delta_seconds),
        )

    def on_tick(self, callback):
        if callback is None:
            return
        self._tick_callbacks.append(callback)

    def get_spectator(self):
        return self._spectator

    def set_weather(self, weather):
        self._weather = weather if weather is not None else _WeatherParameters()

    def get_weather(self):
        return self._weather

    def get_snapshot(self):
        actors = self.get_actors()
        snapshot_map = {
            int(a.id): types.SimpleNamespace(
                id=int(a.id),
                get_transform=(lambda _a=a: _a.get_transform()),
            )
            for a in actors
        }
        class _Snapshot:
            def __init__(self, frame, delta):
                self.timestamp = types.SimpleNamespace(
                    frame=frame,
                    elapsed_seconds=frame * delta,
                    delta_seconds=delta,
                )
                self._m = snapshot_map

            def find(self, actor_id):
                return self._m.get(int(actor_id))

            def __iter__(self):
                return iter(self._m.values())
        return _Snapshot(self._tick_frame, float(self._settings.fixed_delta_seconds))

    def get_actor(self, actor_id: int):
        for actor in self.get_actors():
            if actor.id == actor_id:
                return actor
        return None

    def get_settings(self):
        return self._settings

    def apply_settings(self, settings):
        if settings is not None:
            self._settings = settings
        return self._settings

    def get_random_location_from_navigation(self):
        # Deterministic pseudo-random location near origin.
        return _Location(
            float(_MockCarlaDataProvider._rng.uniform(-50.0, 50.0)),
            float(_MockCarlaDataProvider._rng.uniform(-50.0, 50.0)),
            0.0,
        )

    def get_level_bbs(self, actor_type=None):
        actors = self.get_actors()
        out = []
        label = actor_type
        for a in actors:
            tid = str(getattr(a, "type_id", "")).lower()
            if label is None:
                pass
            elif label in (4,) and "walker" not in tid:   # Pedestrians
                continue
            elif label in (11,) and "traffic_light" not in tid:
                continue
            elif label in (9,) and "traffic" not in tid:
                continue
            bb = getattr(a, "bounding_box", None)
            if bb is not None:
                out.append(bb)
        return out

    def get_traffic_lights_from_waypoint(self, waypoint, distance=30):
        return []

    def get_traffic_lights(self):
        return []

    def get_environment_objects(self, object_type):
        # Expose lightweight objects with transform + bounding_box.
        objs = []
        for a in self.get_actors():
            bb = getattr(a, "bounding_box", None)
            tf = getattr(a, "get_transform", lambda: _Transform())()
            objs.append(types.SimpleNamespace(
                id=int(getattr(a, "id", -1)),
                transform=tf,
                location=tf.location,
                bounding_box=bb if bb is not None else _BoundingBox(location=tf.location),
                type=object_type,
            ))
        return objs


class _MockClient:
    def __init__(self, world: _MockWorld):
        self._world = world
        self._traffic_manager = _MockTrafficManager()

    def get_world(self):
        return self._world

    def set_timeout(self, seconds: float):
        return None

    def get_trafficmanager(self, port: int = 8000):
        self._traffic_manager._port = int(port)
        return self._traffic_manager

    def apply_batch_sync(self, batch, do_tick=False):
        # Minimal support for DestroyActor commands.
        responses = []
        for cmd in (batch or []):
            actor_id = getattr(cmd, "actor_id", None)
            if actor_id is None and hasattr(cmd, "actor"):
                actor_id = getattr(cmd.actor, "id", None)
            if actor_id is not None:
                a = self._world.get_actor(int(actor_id))
                if a is not None and hasattr(a, "destroy"):
                    try:
                        a.destroy()
                        responses.append(types.SimpleNamespace(error=None, actor_id=int(actor_id)))
                    except Exception as exc:
                        responses.append(types.SimpleNamespace(error=str(exc), actor_id=int(actor_id)))
                else:
                    responses.append(types.SimpleNamespace(error="actor_not_found", actor_id=int(actor_id)))
            else:
                responses.append(types.SimpleNamespace(error=None, actor_id=-1))
        if do_tick:
            try:
                self._world.tick()
            except Exception:
                pass
        return responses

    def apply_batch(self, batch):
        return self.apply_batch_sync(batch, do_tick=False)

    def load_world(self, map_name: str):
        return self._world

    def get_available_maps(self):
        return ["/Game/Carla/Maps/openloop_map"]


class _MockTrafficManager:
    def __init__(self):
        self._port = 8000

    def __getattr__(self, name):
        # expose any traffic-manager API as a no-op callable
        return lambda *args, **kwargs: None


# ---------------------------------------------------------------------------
# MockCarlaDataProvider — registry of hero actors
# ---------------------------------------------------------------------------

class _MockCarlaDataProvider:
    """
    Drop-in for ``srunner.scenariomanager.carla_data_provider.CarlaDataProvider``.

    The eval loop registers MockVehicle objects here; agents look them up via
    ``get_hero_actor(hero_id=N)``.
    """

    _hero_actors: Dict[int, MockVehicle] = {}
    _dynamic_actors: List[_MockActor] = []
    _world = None
    _client = None
    _traffic_manager_port = 8000
    _hazard_num = 0
    _actor_velocity_map: Dict[int, _Vector3D] = {}
    _actor_location_map: Dict[int, _Location] = {}
    _actor_transform_map: Dict[int, _Transform] = {}
    _rng = np.random.RandomState(0)
    _ego_vehicle_route = []
    _traffic_lights = []
    _surrounding_actor_cache: Dict[str, _MockActor] = {}
    _runtime_frame_idx = 0
    _runtime_timestamp_s = 0.0
    _runtime_delta_seconds = 0.05

    @classmethod
    def register_hero(cls, hero_id: int, vehicle: MockVehicle):
        if vehicle is not None and hasattr(vehicle, "attributes"):
            if int(hero_id) == 0:
                vehicle.attributes["role_name"] = vehicle.attributes.get("role_name", "hero")
            vehicle.attributes.setdefault("hero_id", str(int(hero_id)))
        cls._hero_actors[hero_id] = vehicle

    @classmethod
    def register_actor(cls, actor):
        if actor is None:
            return
        aid = int(getattr(actor, "id", -1))
        if aid < 0:
            return
        if aid in [int(k) for k in cls._hero_actors.keys()]:
            return
        if getattr(actor, "_source_tag", None) == "surrounding":
            cls._surrounding_actor_cache[str(aid)] = actor
        if all(int(getattr(a, "id", -1)) != aid for a in cls._dynamic_actors):
            cls._dynamic_actors.append(actor)

    @classmethod
    def register_actors(cls, actors):
        for actor in (actors or []):
            cls.register_actor(actor)

    @classmethod
    def on_carla_tick(cls):
        # Refresh lightweight caches for compatibility with scenario-runner API.
        cls._actor_velocity_map.clear()
        cls._actor_location_map.clear()
        cls._actor_transform_map.clear()
        for actor in cls.get_actor_list():
            aid = int(getattr(actor, "id", -1))
            if aid < 0:
                continue
            try:
                cls._actor_velocity_map[aid] = actor.get_velocity()
                cls._actor_location_map[aid] = actor.get_location()
                cls._actor_transform_map[aid] = actor.get_transform()
            except Exception:
                continue

    @classmethod
    def configure_world_timing(cls, delta_seconds: float) -> None:
        world = cls.get_world()
        try:
            settings = world.get_settings()
            settings.fixed_delta_seconds = float(delta_seconds)
            if hasattr(settings, "synchronous_mode"):
                settings.synchronous_mode = True
            if hasattr(settings, "no_rendering_mode"):
                settings.no_rendering_mode = True
        except Exception:
            pass
        try:
            world._tick_frame = 0
        except Exception:
            pass
        cls._runtime_frame_idx = 0
        cls._runtime_timestamp_s = 0.0
        cls._runtime_delta_seconds = float(delta_seconds)
        try:
            _OpenLoopGameTime.restart()
        except Exception:
            pass

    @classmethod
    def advance_runtime(
        cls,
        frame_idx: int,
        timestamp_s: float,
        delta_seconds: float,
    ) -> None:
        cls._runtime_frame_idx = int(frame_idx)
        cls._runtime_timestamp_s = float(timestamp_s)
        cls._runtime_delta_seconds = float(delta_seconds)
        _sync_runtime_clock(
            frame_idx=int(frame_idx),
            timestamp_s=float(timestamp_s),
            delta_seconds=float(delta_seconds),
        )

    @classmethod
    def get_velocity(cls, actor):
        if actor is None:
            return _Vector3D()
        aid = int(getattr(actor, "id", -1))
        if aid in cls._actor_velocity_map:
            return cls._actor_velocity_map[aid]
        try:
            return actor.get_velocity()
        except Exception:
            return _Vector3D()

    @classmethod
    def get_location(cls, actor):
        if actor is None:
            return _Location()
        aid = int(getattr(actor, "id", -1))
        if aid in cls._actor_location_map:
            return cls._actor_location_map[aid]
        try:
            return actor.get_location()
        except Exception:
            return _Location()

    @classmethod
    def get_transform(cls, actor):
        if actor is None:
            return _Transform()
        aid = int(getattr(actor, "id", -1))
        if aid in cls._actor_transform_map:
            return cls._actor_transform_map[aid]
        try:
            return actor.get_transform()
        except Exception:
            return _Transform()

    @classmethod
    def set_client(cls, client):
        cls._client = client

    @classmethod
    def get_hero_actor(cls, hero_id: int = 0) -> Optional[MockVehicle]:
        actor = cls._hero_actors.get(hero_id)
        if actor is not None:
            return actor
        target = f"hero_{int(hero_id)}"
        for a in cls._hero_actors.values():
            attrs = getattr(a, "attributes", {})
            if attrs.get("role_name", "") in (target, "hero"):
                return a
        return None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            cls._client = _MockClient(cls.get_world())
        return cls._client

    @classmethod
    def get_traffic_manager(cls):
        return cls.get_client().get_trafficmanager(cls._traffic_manager_port)

    @classmethod
    def get_actor_list(cls):
        return list(cls._hero_actors.values()) + [a for a in cls._dynamic_actors if a is not None]

    @classmethod
    def get_actors(cls):
        # scenario-runner style: iterable of (id, actor)
        return [(int(a.id), a) for a in cls.get_actor_list()]

    @classmethod
    def get_actor_pool(cls):
        return {int(a.id): a for a in cls.get_actor_list()}

    @classmethod
    def actor_id_exists(cls, actor_id: int) -> bool:
        return any(int(getattr(a, "id", -1)) == int(actor_id) for a in cls.get_actor_list())

    @classmethod
    def get_actor_by_id(cls, actor_id: int):
        for a in cls.get_actor_list():
            if int(getattr(a, "id", -1)) == int(actor_id):
                return a
        return None

    @classmethod
    def get_map(cls, world=None):
        if world is not None and hasattr(world, "get_map"):
            try:
                return world.get_map()
            except Exception:
                pass
        return cls._world.get_map() if cls._world is not None else _MockMap()

    @classmethod
    def get_world(cls):
        if cls._world is None:
            cls._world = _MockWorld()
        return cls._world

    @classmethod
    def set_world(cls, world):
        cls._world = world if world is not None else _MockWorld()
        if cls._client is None:
            cls._client = _MockClient(cls._world)

    @classmethod
    def get_traffic_manager_port(cls):
        return int(cls._traffic_manager_port)

    @classmethod
    def set_traffic_manager_port(cls, tm_port):
        try:
            cls._traffic_manager_port = int(tm_port)
        except Exception:
            cls._traffic_manager_port = 8000

    @classmethod
    def set_random_seed(cls, seed):
        try:
            cls._rng = np.random.RandomState(int(seed))
        except Exception:
            cls._rng = np.random.RandomState(0)

    @classmethod
    def is_sync_mode(cls):
        try:
            return bool(getattr(cls.get_world().get_settings(), "synchronous_mode", True))
        except Exception:
            return True

    @classmethod
    def find_weather_presets(cls):
        carla_mod = sys.modules.get("carla")
        wp = getattr(carla_mod, "WeatherParameters", None)
        if wp is None:
            return []
        presets = []
        for k in dir(wp):
            if k and k[0].isupper():
                v = getattr(wp, k)
                if isinstance(v, _WeatherParameters):
                    presets.append((v, k))
        return presets

    @classmethod
    def prepare_map(cls):
        # Open-loop map is static and already available via get_map().
        return None

    @classmethod
    def annotate_trafficlight_in_group(cls, traffic_light):
        return {"ref": [traffic_light], "opposite": [], "left": [], "right": []}

    @classmethod
    def get_next_traffic_light(cls, actor, use_cached_location=True):
        return None

    @classmethod
    def set_ego_vehicle_route(cls, route):
        cls._ego_vehicle_route = route if route is not None else []

    @classmethod
    def get_ego_vehicle_route(cls):
        return cls._ego_vehicle_route

    @classmethod
    def register_hazard_actor(cls):
        cls._hazard_num += 1
        return cls._hazard_num

    @classmethod
    def get_hazard_actor(cls, hazard_id):
        rid = f"hazard_{hazard_id}"
        for a in cls.get_actor_list():
            attrs = getattr(a, "attributes", {})
            if attrs.get("role_name", "") == rid:
                return a
        return None

    @classmethod
    def remove_actor_by_id(cls, actor_id: int):
        actor = cls.get_actor_by_id(actor_id)
        if actor is None:
            return
        try:
            actor.destroy()
        except Exception:
            pass
        # Hero actors
        for hid in list(cls._hero_actors.keys()):
            if int(getattr(cls._hero_actors[hid], "id", -1)) == int(actor_id):
                cls._hero_actors.pop(hid, None)
        # Dynamic actors
        cls._dynamic_actors = [
            a for a in cls._dynamic_actors
            if int(getattr(a, "id", -1)) != int(actor_id)
        ]
        cls._surrounding_actor_cache = {
            key: value
            for key, value in cls._surrounding_actor_cache.items()
            if int(getattr(value, "id", -1)) != int(actor_id)
        }

    @classmethod
    def remove_actors_in_surrounding(cls, location, distance):
        if location is None:
            return
        try:
            dist = float(distance)
        except Exception:
            dist = 0.0
        keep = []
        for a in cls._dynamic_actors:
            try:
                if a.get_location().distance(location) < dist:
                    a.destroy()
                else:
                    keep.append(a)
            except Exception:
                keep.append(a)
        cls._dynamic_actors = keep

    @classmethod
    def request_new_actor(cls, model, spawn_point, rolename='scenario',
                          autopilot=False, random_location=False,
                          color=None, actor_category="car", hero=False):
        world = cls.get_world()
        bp = world.get_blueprint_library().find(str(model) if model else "vehicle.tesla.model3")
        bp.set_attribute("role_name", "hero" if hero else str(rolename))
        actor = world.try_spawn_actor(bp, spawn_point if spawn_point is not None else _Transform())
        if actor is not None:
            try:
                actor.set_autopilot(autopilot)
            except Exception:
                pass
            cls.register_actor(actor)
        return actor

    @classmethod
    def request_new_actors(cls, actor_list):
        created = []
        for actor_desc in (actor_list or []):
            model = getattr(actor_desc, "model", "vehicle.tesla.model3")
            transform = getattr(actor_desc, "transform", _Transform())
            role = getattr(actor_desc, "rolename", "scenario")
            actor = cls.request_new_actor(model=model, spawn_point=transform, rolename=role)
            if actor is not None:
                created.append(actor)
        return created

    @classmethod
    def request_new_batch_actors(cls, model, amount, spawn_points,
                                 autopilot=False, random_location=False,
                                 rolename='scenario'):
        actors = []
        spawn_points = list(spawn_points or [])
        for i in range(int(max(0, amount))):
            tf = spawn_points[i] if i < len(spawn_points) else _Transform()
            a = cls.request_new_actor(model=model, spawn_point=tf,
                                      rolename=rolename, autopilot=autopilot)
            if a is not None:
                actors.append(a)
        return actors

    @classmethod
    def cleanup(cls):
        for a in list(cls.get_actor_list()):
            try:
                a.destroy()
            except Exception:
                pass
        cls._hero_actors = {}
        cls._dynamic_actors = []
        cls._surrounding_actor_cache = {}
        cls._actor_velocity_map.clear()
        cls._actor_location_map.clear()
        cls._actor_transform_map.clear()
        cls._hazard_num = 0
        cls._runtime_frame_idx = 0
        cls._runtime_timestamp_s = 0.0
        cls._runtime_delta_seconds = 0.05

    @classmethod
    def set_surrounding_from_v2xpnp(
        cls,
        vehicles_dict: Optional[Dict],
        delta_seconds: float | None = None,
        previous_vehicles_dict: Optional[Dict] = None,
        timestamp_s: float | None = None,
    ):
        """Update non-ego world actors from one v2xpnp frame YAML."""
        preserved = [
            a for a in cls._dynamic_actors
            if getattr(a, "_source_tag", "spawned") != "surrounding"
        ]
        cls._actor_velocity_map.clear()
        cls._actor_location_map.clear()
        cls._actor_transform_map.clear()
        if not vehicles_dict:
            cls._dynamic_actors = preserved
            return

        current: List[_MockActor] = []
        prev_vehicles_dict = previous_vehicles_dict or {}
        for raw_id, item in vehicles_dict.items():
            if not isinstance(item, dict):
                continue
            loc = item.get("location", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0]
            ang = item.get("angle", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0]
            ext = item.get("extent", [1.0, 0.5, 0.8]) or [1.0, 0.5, 0.8]
            obj = str(item.get("obj_type", "Car")).lower()

            if "walker" in obj or "pedestrian" in obj or "person" in obj:
                type_id = "walker.pedestrian"
            elif "cyclist" in obj or "bike" in obj or "bicycle" in obj:
                type_id = "vehicle.diamondback.century"
            elif "bus" in obj:
                type_id = "vehicle.bus"
            else:
                type_id = "vehicle.car"

            try:
                rid = int(raw_id)
            except Exception:
                rid = abs(hash(str(raw_id))) % 100000
            actor_id = 100000 + rid

            yaw_deg = float(ang[1]) if len(ang) > 1 else 0.0
            roll_deg = float(ang[0]) if len(ang) > 0 else 0.0
            pitch_deg = float(ang[2]) if len(ang) > 2 else 0.0

            cache_key = str(raw_id)
            actor = cls._surrounding_actor_cache.get(cache_key)
            if actor is None:
                actor = _MockActor(
                    actor_id=actor_id,
                    type_id=type_id,
                    location=_Location(
                        float(loc[0]) if len(loc) > 0 else 0.0,
                        float(loc[1]) if len(loc) > 1 else 0.0,
                        float(loc[2]) if len(loc) > 2 else 0.0,
                    ),
                    rotation=_Rotation(roll=roll_deg, pitch=pitch_deg, yaw=yaw_deg),
                    extent_xyz=[
                        float(ext[0]) if len(ext) > 0 else 1.0,
                        float(ext[1]) if len(ext) > 1 else 0.5,
                        float(ext[2]) if len(ext) > 2 else 0.8,
                    ],
                    attributes={"role_name": f"npc_{rid}"},
                    source_tag="surrounding",
                )
                cls._surrounding_actor_cache[cache_key] = actor
            else:
                actor.type_id = type_id
                actor.attributes.setdefault("role_name", f"npc_{rid}")
                actor.bounding_box = _BoundingBox(
                    extent=_Vector3D(
                        float(ext[0]) if len(ext) > 0 else 1.0,
                        float(ext[1]) if len(ext) > 1 else 0.5,
                        float(ext[2]) if len(ext) > 2 else 0.8,
                    )
                )

            prev_item = prev_vehicles_dict.get(raw_id)
            if prev_item is None:
                prev_item = prev_vehicles_dict.get(str(raw_id))
            if isinstance(prev_item, dict) and actor._last_transform is None:
                prev_loc = prev_item.get("location", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0]
                prev_ang = prev_item.get("angle", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0]
                _set_actor_runtime_state(
                    actor,
                    _Transform(
                        location=_Location(
                            float(prev_loc[0]) if len(prev_loc) > 0 else 0.0,
                            float(prev_loc[1]) if len(prev_loc) > 1 else 0.0,
                            float(prev_loc[2]) if len(prev_loc) > 2 else 0.0,
                        ),
                        rotation=_Rotation(
                            roll=float(prev_ang[0]) if len(prev_ang) > 0 else 0.0,
                            yaw=float(prev_ang[1]) if len(prev_ang) > 1 else 0.0,
                            pitch=float(prev_ang[2]) if len(prev_ang) > 2 else 0.0,
                        ),
                    ),
                    delta_seconds=delta_seconds,
                )

            actor.update_runtime_state(
                location=_Location(
                    float(loc[0]) if len(loc) > 0 else 0.0,
                    float(loc[1]) if len(loc) > 1 else 0.0,
                    float(loc[2]) if len(loc) > 2 else 0.0,
                ),
                rotation=_Rotation(roll=roll_deg, pitch=pitch_deg, yaw=yaw_deg),
                timestamp_s=timestamp_s,
                delta_seconds=delta_seconds,
            )
            current.append(actor)

        cls._dynamic_actors = preserved + current


# ---------------------------------------------------------------------------
# DrivableAreaRenderer — rasterises the v2xpnp vector map into a BEV image
# ---------------------------------------------------------------------------

class _DrivableAreaRenderer:
    """
    Renders a 400×400 pixel BEV image (pixels_per_meter=5, 80m×80m) of the
    drivable lane area around the current ego position, using the v2xpnp
    vector-map Lane boundary polygons.

    The rendered image is used as a drop-in replacement for the BEV produced by
    BirdViewProducer; the downstream code crops and thresholds it to obtain the
    192×96 binary drivable-area mask fed into the planning model.
    """

    PIXELS_PER_METER: int = 5
    IMAGE_SIZE:       int = 400   # 80 m × 80 m

    def __init__(self, driving_lanes):
        """
        Parameters
        ----------
        driving_lanes : list of Lane
            All LaneType.Driving lanes from the v2xpnp vector map.
        """
        self._lanes = driving_lanes

    def render(self, ego_x: float, ego_y: float, ego_yaw_rad: float) -> np.ndarray:
        """
        Render a 400×400 RGB ndarray.

        Coordinate conventions (matching BirdViewCropType.FRONT_AND_REAR_AREA):
          - Ego at pixel (200, 200)
          - Ego forward direction points toward the TOP of the image (−v)
          - Ego right direction points toward the RIGHT (+u)
          - White pixels (R+G+B > 200) = drivable

        Parameters
        ----------
        ego_x, ego_y : float
            Ego world-frame position (metres).
        ego_yaw_rad : float
            Ego heading in radians (CARLA / v2xpnp convention: yaw around Z).
        """
        sz  = self.IMAGE_SIZE
        ppm = self.PIXELS_PER_METER
        cx  = cy = sz // 2   # ego pixel position

        # Rotation: world → ego frame.  Ego forward = +X, left = +Y.
        # BEV pixel:  u = cx + ego_left * ppm,  v = cy − ego_forward * ppm
        cos_yaw = math.cos(-ego_yaw_rad)
        sin_yaw = math.sin(-ego_yaw_rad)

        img  = Image.new("RGB", (sz, sz), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        for lane in self._lanes:
            # Prefer boundary polygon; fall back to polyline buffered ±1.5 m.
            raw_pts = lane.boundary if len(lane.boundary) >= 3 else lane.polyline
            if len(raw_pts) < 2:
                continue

            # World → BEV pixel
            pixels = []
            for mp in raw_pts:
                dx = mp.x - ego_x
                dy = mp.y - ego_y
                # Rotate to ego frame
                fwd  =  cos_yaw * dx - sin_yaw * dy   # forward (+X in ego)
                # Keep lateral sign consistent with planner local-frame convention.
                left =  sin_yaw * dx + cos_yaw * dy   # left    (+Y in ego)
                u = int(cx - left * ppm)   # ego-left → image-left
                v = int(cy - fwd  * ppm)   # ego-fwd  → image-up
                pixels.append((u, v))

            # Only draw if at least one vertex is within the image
            if any(0 <= u < sz and 0 <= v < sz for u, v in pixels):
                if len(pixels) >= 3:
                    draw.polygon(pixels, fill=(255, 255, 255))
                else:
                    draw.line(pixels, fill=(255, 255, 255), width=ppm * 3)

        return np.array(img)   # (400, 400, 3) uint8


# ---------------------------------------------------------------------------
# MockBirdViewProducer — drop-in replacement (same public API)
# ---------------------------------------------------------------------------

class _MockBirdViewProducer:
    """
    Replaces ``team_code.utils.carla_birdeye_view.BirdViewProducer``.

    Accepts the same constructor arguments as the real class (carla_client,
    target_size, pixels_per_meter, crop_type) but ignores them; instead uses
    ``_DrivableAreaRenderer`` backed by the v2xpnp vector map.
    """

    def __init__(self, carla_client=None, target_size=None,
                 pixels_per_meter=None, crop_type=None):
        self._renderer: Optional[_DrivableAreaRenderer] = None
        self._target_size = target_size
        self._crop_type = crop_type
        self._pixels_per_meter = pixels_per_meter

    def _set_renderer(self, renderer: _DrivableAreaRenderer):
        """Called by ``install_openloop_stubs`` after loading the map."""
        self._renderer = renderer

    def produce(self, agent_vehicle, actor_exist: bool = False) -> np.ndarray:
        """
        Returns a 400×400 RGB ndarray representing the drivable area BEV.
        The downstream code does:
            BirdViewProducer.as_rgb(producer.produce(vehicle))
        Since we already return an ndarray, ``as_rgb`` is identity here.
        """
        if self._renderer is None:
            # Fallback: all-drivable (white) image
            bird_view = np.full((400, 400, 3), 255, dtype=np.uint8)
        else:
            t = agent_vehicle.get_transform()
            yaw_rad = math.radians(t.rotation.yaw)   # carla stores degrees
            bird_view = self._renderer.render(t.location.x, t.location.y, yaw_rad)

        if self._target_size is not None:
            try:
                width = int(getattr(self._target_size, "width", bird_view.shape[1]))
                height = int(getattr(self._target_size, "height", bird_view.shape[0]))
                if bird_view.shape[1] != width or bird_view.shape[0] != height:
                    bird_view = np.array(
                        Image.fromarray(bird_view).resize((width, height), Image.NEAREST)
                    )
            except Exception:
                pass
        return bird_view

    @staticmethod
    def as_rgb(bird_view: np.ndarray) -> np.ndarray:
        """Already an ndarray — pass through."""
        return bird_view


class _PixelDimensions:
    def __init__(self, width: int = 400, height: int = 400):
        self.width  = width
        self.height = height


class _BirdViewCropType:
    FRONT_AND_REAR_AREA = "FRONT_AND_REAR_AREA"
    FRONT_AREA_ONLY     = "FRONT_AREA_ONLY"


# ---------------------------------------------------------------------------
# MockRoadSideUnit — no-op RSU for open-loop (no live RSU sensors)
# ---------------------------------------------------------------------------

class _MockRoadSideUnit:
    def __init__(self, save_path=None, id=0, is_None=False, use_semantic=False):
        self._is_none = is_None

    def setup_sensors(self, parent=None, spawn_point=None):
        pass

    def tick(self):
        return None

    def process(self, data, is_train=False):
        return None

    def cleanup(self):
        pass


def _mock_get_rsu_point(vehicle, height=7.5, lane_side="right", distance=12):
    """Return a placeholder Transform; MockRoadSideUnit ignores spawn_point."""
    if vehicle is None:
        return None
    t = vehicle.get_transform()
    return _Transform(location=_Location(t.location.x, t.location.y,
                                         t.location.z + height))


# ---------------------------------------------------------------------------
# RoadOption stub (agents.navigation.local_planner)
# ---------------------------------------------------------------------------

class _RoadOption:
    VOID            = -1
    LEFT            = 1
    RIGHT           = 2
    STRAIGHT        = 3
    LANEFOLLOW      = 4
    CHANGELANELEFT  = 5
    CHANGELANERIGHT = 6

    def __init__(self, value: int = 4):
        self.value = value

    def __repr__(self):
        return f"RoadOption({self.value})"


# ---------------------------------------------------------------------------
# Waypoint interceptor — captures planned BEV waypoints without modifying
# any planner file.
# ---------------------------------------------------------------------------

class WaypointInterceptor:
    """
    Wraps a planner's planning_model so that every forward pass captures the
    output ``future_waypoints`` tensor into ``self.last_waypoints``.

    Usage::

        interceptor = WaypointInterceptor(pnp_infer_obj)
        # ... run agent.run_step() ...
        bev_wps = interceptor.last_waypoints  # np.ndarray [T, 2], may be None

    The planner model forward() is called normally; we only save the output.
    No planner source file is modified.
    """

    def __init__(self, infer_obj):
        self.last_waypoints: Optional[np.ndarray] = None
        interceptor = self  # closure reference

        orig_model = infer_obj.planning_model

        class _Wrapper:
            def __call__(self_, planning_input):
                output = orig_model(planning_input)
                wps = output.get("future_waypoints")
                if wps is not None:
                    interceptor.last_waypoints = wps.detach().cpu().numpy()
                return output

            # Forward any attribute access (e.g., .parameters(), .eval(), etc.)
            def __getattr__(self_, name):
                return getattr(orig_model, name)

        infer_obj.planning_model = _Wrapper()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# Singleton renderer / producer shared across the process
_global_renderer: Optional[_DrivableAreaRenderer] = None
_global_producer: Optional[_MockBirdViewProducer] = None
_hero_actors: Dict[int, MockVehicle] = {}


def install_openloop_stubs(vector_map_path: Optional[str] = None,
                           num_ego_vehicles: int = 2) -> Dict[int, MockVehicle]:
    """
    Install the CARLA / leaderboard shims so agents can be imported and run
    without a live CARLA server.

    Must be called BEFORE importing any agent or planner module.

    Parameters
    ----------
    vector_map_path : str or None
        Path to ``v2v_corridors_vector_map.pkl``.  If None, the drivable-area
        BEV will be all-white (fully drivable), which is a conservative fallback.
    num_ego_vehicles : int
        Number of MockVehicle objects to pre-register.

    Returns
    -------
    dict[int, MockVehicle]
        The registered hero vehicles, keyed by hero_id.  The eval loop should
        call ``vehicle.update_pose(...)`` before each ``agent.run_step()``.
    """
    global _global_renderer, _global_producer, _hero_actors
    _hero_actors = {}
    _MockCarlaDataProvider._hero_actors = {}

    # ---- 1. Set env vars required by planner internals ----
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("ROUTES",          "openloop_dummy_route")

    # ---- 2. Inject synthetic `carla` module ----
    if "carla" not in sys.modules:
        sys.modules["carla"] = _build_carla_module()

    # ---- 2.5. Create synthetic world/client for CarlaDataProvider ----
    _MockCarlaDataProvider._world = _MockWorld()
    _MockCarlaDataProvider._client = _MockClient(_MockCarlaDataProvider._world)
    _MockCarlaDataProvider._dynamic_actors = []
    _MockCarlaDataProvider._ego_vehicle_route = []
    _MockCarlaDataProvider._traffic_lights = []
    _MockCarlaDataProvider._surrounding_actor_cache = {}
    _MockCarlaDataProvider.configure_world_timing(
        float(_MockCarlaDataProvider.get_world().get_settings().fixed_delta_seconds)
    )

    # ---- 3. Load vector map and build drivable-area renderer ----
    if vector_map_path is not None:
        _global_renderer = _load_renderer_from_map(vector_map_path)
    else:
        _global_renderer = None

    _global_producer = _MockBirdViewProducer()
    if _global_renderer is not None:
        _global_producer._set_renderer(_global_renderer)

    # Install the BirdView API at module import time so planner code can import
    # its normal CARLA-facing modules without any agent-instance monkeypatching.
    _ensure_birdview_stub_modules()

    # ---- 4. Stub srunner / CarlaDataProvider ----
    _install_srunner_stubs()

    # ---- 5. Stub leaderboard fixed_sensors (RSU) ----
    _install_leaderboard_sensor_stubs()

    # ---- 6. Stub agents.navigation.local_planner ----
    _install_navigation_stubs()

    # ---- 7. Stub pygame ----
    _install_pygame_stub()

    # ---- 9. Create and register MockVehicles ----
    for vid in range(num_ego_vehicles):
        mv = MockVehicle(hero_id=vid)
        _MockCarlaDataProvider.register_hero(vid, mv)
        _hero_actors[vid] = mv

    return _hero_actors


def patch_agent_birdview(agent_instance) -> None:
    """
    Compatibility shim kept for older evaluator paths.

    BirdViewProducer is now installed at module import time, so no instance
    monkeypatching is required here.
    """
    if _global_producer is None:
        raise RuntimeError("install_openloop_stubs() must be called first")
    _ensure_birdview_stub_modules()
    return None


def configure_openloop_world(delta_seconds: float) -> None:
    """Configure the open-loop runtime clock and world step size."""
    _MockCarlaDataProvider.configure_world_timing(float(delta_seconds))


def advance_openloop_runtime(frame_idx: int, timestamp_s: float, delta_seconds: float) -> None:
    """Advance GameTime, world timing, and CarlaDataProvider caches for one frame."""
    _MockCarlaDataProvider.advance_runtime(
        frame_idx=int(frame_idx),
        timestamp_s=float(timestamp_s),
        delta_seconds=float(delta_seconds),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_renderer_from_map(path: str) -> _DrivableAreaRenderer:
    """Load v2xpnp vector map and return a renderer for driving lanes."""
    import importlib.util as _ilu

    # Load local map_types directly to avoid package __init__ chain triggers
    _openloop_root = Path(__file__).resolve().parents[1]
    _mt_path = str(_openloop_root / "data_utils" / "datasets" / "map" / "map_types.py")
    if not Path(_mt_path).exists():
        raise FileNotFoundError(f"openloop map_types.py not found at {_mt_path}")

    spec = _ilu.spec_from_file_location("_map_types_tmp", _mt_path)
    mt   = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mt)

    # Ensure pickle can find the classes under their canonical name
    for _name in ("Lane", "Map", "MapPoint", "LaneType",
                  "Crosswalk", "RoadLine", "WalkButton",
                  "TrafficSignalLaneState", "DynamicState"):
        canonical = f"opencood.data_utils.datasets.map.map_types"
        pkg = sys.modules.setdefault(canonical, types.ModuleType(canonical))
        setattr(pkg, _name, getattr(mt, _name))

    import pickle
    with open(path, "rb") as fh:
        vmap = pickle.load(fh)

    driving_lanes = [
        f for f in vmap.map_features
        if isinstance(f, mt.Lane) and f.type == mt.LaneType.Driving
    ]
    return _DrivableAreaRenderer(driving_lanes)


def _install_srunner_stubs():
    """Inject a minimal srunner stub so agents can import CarlaDataProvider."""
    import importlib

    def _make(name):
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    # Prefer real package if available; only create missing modules.
    try:
        importlib.import_module("srunner")
    except Exception:
        if "srunner" not in sys.modules:
            _make("srunner")

    try:
        importlib.import_module("srunner.scenariomanager")
    except Exception:
        if "srunner.scenariomanager" not in sys.modules:
            _make("srunner.scenariomanager")

    # Provide a deterministic runtime clock used by leaderboard.autoagents.
    try:
        importlib.import_module("srunner.scenariomanager.timer")
    except Exception:
        tmod = sys.modules.get("srunner.scenariomanager.timer")
        if tmod is None:
            tmod = _make("srunner.scenariomanager.timer")
    else:
        tmod = sys.modules.get("srunner.scenariomanager.timer")
        if tmod is None:
            tmod = _make("srunner.scenariomanager.timer")

    tmod.GameTime = _OpenLoopGameTime

    try:
        cdp_mod = importlib.import_module("srunner.scenariomanager.carla_data_provider")
    except Exception:
        cdp_mod = sys.modules.get("srunner.scenariomanager.carla_data_provider")
        if cdp_mod is None:
            cdp_mod = _make("srunner.scenariomanager.carla_data_provider")
    cdp_mod.CarlaDataProvider = _MockCarlaDataProvider
    if not hasattr(cdp_mod, "CarlaActorPool"):
        class _CarlaActorPool:
            @staticmethod
            def set_client(client):
                _MockCarlaDataProvider.set_client(client)

            @staticmethod
            def get_client():
                return _MockCarlaDataProvider.get_client()

            @staticmethod
            def set_world(world):
                _MockCarlaDataProvider.set_world(world)

            @staticmethod
            def get_actor_by_id(actor_id):
                return _MockCarlaDataProvider.get_actor_by_id(actor_id)

            @staticmethod
            def get_hero_actor(hero_id=0):
                return _MockCarlaDataProvider.get_hero_actor(hero_id)

            @staticmethod
            def remove_actor_by_id(actor_id):
                _MockCarlaDataProvider.remove_actor_by_id(actor_id)

            @staticmethod
            def cleanup():
                _MockCarlaDataProvider.cleanup()

        cdp_mod.CarlaActorPool = _CarlaActorPool


def _install_leaderboard_sensor_stubs():
    """Stub leaderboard.sensors.fixed_sensors (RSU)."""
    def _make(name):
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    # Do not shadow the real `leaderboard` package: planners also import
    # `leaderboard.autoagents`. Only patch/create the fixed_sensors module.
    try:
        import importlib
        importlib.import_module("leaderboard")
    except Exception:
        if "leaderboard" not in sys.modules:
            _make("leaderboard")

    if "leaderboard.sensors" not in sys.modules:
        _make("leaderboard.sensors")
    if "leaderboard.sensors.fixed_sensors" not in sys.modules:
        _make("leaderboard.sensors.fixed_sensors")

    fmod = sys.modules.get("leaderboard.sensors.fixed_sensors")
    fmod.RoadSideUnit   = _MockRoadSideUnit
    fmod.get_rsu_point  = _mock_get_rsu_point


def _install_navigation_stubs():
    """Stub agents.navigation.local_planner (RoadOption)."""
    import importlib

    def _make(name):
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    # Keep real `agents` package if available; create only missing pieces.
    try:
        importlib.import_module("agents")
    except Exception:
        if "agents" not in sys.modules:
            _make("agents")

    if "agents.navigation" not in sys.modules:
        _make("agents.navigation")
    if "agents.navigation.local_planner" not in sys.modules:
        _make("agents.navigation.local_planner")
    if "agents.navigation.global_route_planner" not in sys.modules:
        _make("agents.navigation.global_route_planner")
    if "agents.navigation.global_route_planner_dao" not in sys.modules:
        _make("agents.navigation.global_route_planner_dao")

    sys.modules["agents.navigation.local_planner"].RoadOption = _RoadOption

    class _GlobalRoutePlanner:
        def __init__(self, *args, **kwargs):
            pass

        def setup(self):
            pass

        def trace_route(self, *args, **kwargs):
            return []

    class _GlobalRoutePlannerDAO:
        def __init__(self, *args, **kwargs):
            pass

    sys.modules["agents.navigation.global_route_planner"].GlobalRoutePlanner = _GlobalRoutePlanner
    sys.modules["agents.navigation.global_route_planner_dao"].GlobalRoutePlannerDAO = _GlobalRoutePlannerDAO


def _install_pygame_stub():
    """
    Stub pygame with just enough surface so DisplayInterface.__init__()
    doesn't crash.  Requires SDL_VIDEODRIVER=dummy already set.
    """
    try:
        import pygame as pg
    except Exception:
        pg = types.ModuleType("pygame")
        sys.modules["pygame"] = pg

    class _DummyDisplay:
        def blit(self, *args, **kwargs):
            return None

    if not hasattr(pg, "init"):
        pg.init = lambda: None
    if not hasattr(pg, "quit"):
        pg.quit = lambda: None

    if not hasattr(pg, "display"):
        pg.display = types.SimpleNamespace()
    if not hasattr(pg.display, "set_mode"):
        pg.display.set_mode = lambda *a, **kw: _DummyDisplay()
    if not hasattr(pg.display, "set_caption"):
        pg.display.set_caption = lambda *a, **kw: None
    if not hasattr(pg.display, "flip"):
        pg.display.flip = lambda: None

    if not hasattr(pg, "font"):
        pg.font = types.SimpleNamespace()
    if not hasattr(pg.font, "init"):
        pg.font.init = lambda: None

    if not hasattr(pg, "Surface"):
        pg.Surface = lambda *a, **kw: None
    if not hasattr(pg, "HWSURFACE"):
        pg.HWSURFACE = 0
    if not hasattr(pg, "DOUBLEBUF"):
        pg.DOUBLEBUF = 0

    if not hasattr(pg, "time"):
        pg.time = types.SimpleNamespace()
    if not hasattr(pg.time, "Clock"):
        pg.time.Clock = lambda: types.SimpleNamespace(tick=lambda fps: None)

    if not hasattr(pg, "event"):
        pg.event = types.SimpleNamespace()
    if not hasattr(pg.event, "get"):
        pg.event.get = lambda: []

    # Some minimal pygame builds expose pygame but not surfarray.
    if not hasattr(pg, "surfarray"):
        pg.surfarray = types.SimpleNamespace()
    if not hasattr(pg.surfarray, "make_surface"):
        pg.surfarray.make_surface = lambda arr: arr

    sys.modules["pygame"] = pg
