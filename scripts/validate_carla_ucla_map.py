#!/usr/bin/env python3
"""Minimal validation client for UCLA map loading in CARLA 0.9.12."""

import argparse
import sys
import time
from pathlib import Path


def _load_carla_module(carla_root: Path):
    egg = carla_root / "PythonAPI" / "carla" / "dist" / "carla-0.9.12-py3.7-linux-x86_64.egg"
    if not egg.exists():
        raise FileNotFoundError(f"CARLA egg not found: {egg}")
    sys.path.append(str(egg))
    import carla  # pylint: disable=import-error

    return carla


def _resolve_ucla_map_name(client, requested: str):
    candidates = [
        f"/Game/Carla/Maps/{requested}/{requested}",
        f"/Game/Carla/Maps/{requested}",
        requested,
    ]
    try:
        available = client.get_available_maps()
    except RuntimeError:
        available = []
    lower_req = requested.lower()
    for m in available:
        if lower_req in m.lower():
            candidates.insert(0, m)
            break
    seen = set()
    ordered = []
    for c in candidates:
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered


def validate(host: str, port: int, carla_root: Path, map_name: str) -> None:
    carla = _load_carla_module(carla_root)

    client = carla.Client(host, port)
    client.set_timeout(20.0)

    last_error = None
    world = None
    for candidate in _resolve_ucla_map_name(client, map_name):
        try:
            world = client.load_world(candidate)
            current = world.get_map().name
            if map_name.lower() in current.lower():
                break
        except RuntimeError as exc:
            last_error = exc
            continue
    else:
        raise RuntimeError(f"Unable to load UCLA map '{map_name}'") from last_error

    time.sleep(2.0)

    # Weather call is part of the validation because this is one known failure mode.
    world.set_weather(carla.WeatherParameters.ClearNoon)
    try:
        world.wait_for_tick(5.0)
    except RuntimeError:
        time.sleep(1.0)

    current_map = world.get_map().name
    xodr = world.get_map().to_opendrive()
    if not xodr or len(xodr) < 1024:
        raise RuntimeError("Unexpected OpenDRIVE payload size from loaded UCLA map")

    print(f"Loaded map: {current_map}")
    print(f"OpenDRIVE bytes: {len(xodr)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate UCLA map loading in CARLA")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2140)
    parser.add_argument("--carla-root", type=Path, required=True)
    parser.add_argument("--map", default="ucla_v2")
    args = parser.parse_args()

    validate(args.host, args.port, args.carla_root.resolve(), args.map)


if __name__ == "__main__":
    main()
