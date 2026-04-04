#!/usr/bin/env python

"""
Serialization helpers for evaluator-side recovery checkpoints.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency in tests
    np = None

try:
    import carla
except Exception:  # pragma: no cover - unit tests may run without CARLA
    carla = None

try:
    from srunner.scenariomanager.traffic_events import TrafficEvent, TrafficEventType
except Exception:  # pragma: no cover
    TrafficEvent = None
    TrafficEventType = None


def is_carla_actor(value: Any) -> bool:
    if value is None:
        return False
    return hasattr(value, "id") and hasattr(value, "type_id") and hasattr(value, "get_transform")


def is_carla_location(value: Any) -> bool:
    return carla is not None and isinstance(value, carla.Location)


def is_carla_rotation(value: Any) -> bool:
    return carla is not None and isinstance(value, carla.Rotation)


def is_carla_vector(value: Any) -> bool:
    return carla is not None and isinstance(value, carla.Vector3D)


def is_carla_transform(value: Any) -> bool:
    return carla is not None and isinstance(value, carla.Transform)


def serialize_location(location: Any) -> dict[str, float] | None:
    if location is None:
        return None
    try:
        return {
            "x": float(location.x),
            "y": float(location.y),
            "z": float(location.z),
        }
    except Exception:
        return None


def deserialize_location(payload: Any):
    if carla is None:
        return payload
    if not isinstance(payload, dict):
        return None
    try:
        return carla.Location(
            x=float(payload.get("x", 0.0)),
            y=float(payload.get("y", 0.0)),
            z=float(payload.get("z", 0.0)),
        )
    except Exception:
        return None


def serialize_rotation(rotation: Any) -> dict[str, float] | None:
    if rotation is None:
        return None
    try:
        return {
            "pitch": float(rotation.pitch),
            "yaw": float(rotation.yaw),
            "roll": float(rotation.roll),
        }
    except Exception:
        return None


def deserialize_rotation(payload: Any):
    if carla is None:
        return payload
    if not isinstance(payload, dict):
        return None
    try:
        return carla.Rotation(
            pitch=float(payload.get("pitch", 0.0)),
            yaw=float(payload.get("yaw", 0.0)),
            roll=float(payload.get("roll", 0.0)),
        )
    except Exception:
        return None


def serialize_vector(vector: Any) -> dict[str, float] | None:
    if vector is None:
        return None
    try:
        return {
            "x": float(vector.x),
            "y": float(vector.y),
            "z": float(vector.z),
        }
    except Exception:
        return None


def deserialize_vector(payload: Any):
    if carla is None:
        return payload
    if not isinstance(payload, dict):
        return None
    try:
        return carla.Vector3D(
            x=float(payload.get("x", 0.0)),
            y=float(payload.get("y", 0.0)),
            z=float(payload.get("z", 0.0)),
        )
    except Exception:
        return None


def serialize_transform(transform: Any) -> dict[str, Any] | None:
    if transform is None:
        return None
    try:
        return {
            "location": serialize_location(transform.location),
            "rotation": serialize_rotation(transform.rotation),
        }
    except Exception:
        return None


def deserialize_transform(payload: Any):
    if carla is None:
        return payload
    if not isinstance(payload, dict):
        return None
    location = deserialize_location(payload.get("location", {}))
    rotation = deserialize_rotation(payload.get("rotation", {}))
    if location is None or rotation is None:
        return None
    try:
        return carla.Transform(location, rotation)
    except Exception:
        return None


def serialize_traffic_event(event: Any) -> Any:
    if TrafficEvent is None or not isinstance(event, TrafficEvent):
        return serialize_value(event)
    event_type = event.get_type()
    event_name = event_type.name if hasattr(event_type, "name") else str(event_type)
    return {
        "__traffic_event__": True,
        "event_type": event_name,
        "message": event.get_message(),
        "payload": serialize_value(event.get_dict()),
    }


def deserialize_traffic_event(payload: Any) -> Any:
    if not isinstance(payload, dict) or not payload.get("__traffic_event__"):
        return deserialize_value(payload)
    if TrafficEvent is None or TrafficEventType is None:
        return payload
    try:
        event_type_name = str(payload.get("event_type", "NORMAL_DRIVING"))
        event_type = getattr(TrafficEventType, event_type_name, TrafficEventType.NORMAL_DRIVING)
        event = TrafficEvent(event_type)
        event.set_message(payload.get("message"))
        event.set_dict(deserialize_value(payload.get("payload")))
        return event
    except Exception:
        return payload


def serialize_value(value: Any, depth: int = 0, max_depth: int = 6) -> Any:
    if depth > max_depth:
        return repr(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    if isinstance(value, Enum):
        return {
            "__enum__": f"{value.__class__.__module__}:{value.__class__.__name__}:{value.name}",
        }
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if np is not None and isinstance(value, np.ndarray):
        return {"__ndarray__": value.tolist()}
    if is_carla_location(value):
        return {"__carla_type__": "Location", "data": serialize_location(value)}
    if is_carla_rotation(value):
        return {"__carla_type__": "Rotation", "data": serialize_rotation(value)}
    if is_carla_vector(value):
        return {"__carla_type__": "Vector3D", "data": serialize_vector(value)}
    if is_carla_transform(value):
        return {"__carla_type__": "Transform", "data": serialize_transform(value)}
    if TrafficEvent is not None and isinstance(value, TrafficEvent):
        return serialize_traffic_event(value)
    if isinstance(value, (list, tuple, set)):
        return [serialize_value(item, depth=depth + 1, max_depth=max_depth) for item in value]
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out[str(key)] = serialize_value(item, depth=depth + 1, max_depth=max_depth)
        return out
    if is_carla_actor(value):
        return {"__carla_actor_ref__": int(getattr(value, "id", -1))}
    if hasattr(value, "__dict__"):
        return {
            "__object__": f"{value.__class__.__module__}:{value.__class__.__name__}",
            "state": serialize_value(dict(value.__dict__), depth=depth + 1, max_depth=max_depth),
        }
    return repr(value)


def deserialize_value(value: Any) -> Any:
    if isinstance(value, list):
        return [deserialize_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "__traffic_event__" in value:
        return deserialize_traffic_event(value)
    carla_type = value.get("__carla_type__")
    if carla_type == "Location":
        return deserialize_location(value.get("data"))
    if carla_type == "Rotation":
        return deserialize_rotation(value.get("data"))
    if carla_type == "Vector3D":
        return deserialize_vector(value.get("data"))
    if carla_type == "Transform":
        return deserialize_transform(value.get("data"))
    if "__ndarray__" in value:
        return value.get("__ndarray__")
    if "__enum__" in value:
        return value.get("__enum__")
    if "__object__" in value:
        return deserialize_value(value.get("state", {}))
    out = {}
    for key, item in value.items():
        out[key] = deserialize_value(item)
    return out

