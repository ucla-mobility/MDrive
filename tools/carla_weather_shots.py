#!/usr/bin/env python3
"""
Capture one screenshot per CARLA weather preset.

Usage:
  python tools/carla_weather_shots.py --host 127.0.0.1 --port 2000 --output ./weather_shots

Requires a running CARLA server. Uses synchronous mode, spawns a single vehicle
with a front-facing RGB camera, cycles weather presets, and saves PNGs named
<preset>.png in the output directory.
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path
from queue import Queue, Empty
import inspect
import numpy as np


def _add_carla_to_path(carla_root: Path) -> None:
    """Append CARLA Python API egg to sys.path."""
    egg_pattern = str(carla_root / "PythonAPI" / "carla" / "dist" / "carla-*py3*.egg")
    eggs = glob.glob(egg_pattern)
    if not eggs:
        raise SystemExit(f"Could not find CARLA egg at {egg_pattern}")
    sys.path.append(eggs[0])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1", help="CARLA host (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=2000, help="CARLA port (default: 2000)")
    ap.add_argument("--timeout", type=float, default=5.0, help="Client timeout seconds")
    ap.add_argument(
        "--carla-root",
        default="/data2/marco/CoLMDriver/carla",
        help="Path to CARLA root containing PythonAPI (default: /data2/marco/CoLMDriver/carla)",
    )
    ap.add_argument(
        "--output",
        default="weather_shots",
        help="Directory to save screenshots (created if missing)",
    )
    ap.add_argument(
        "--settle-ticks",
        type=int,
        default=24,
        help="World ticks after weather change before capturing (default: 24).",
    )
    ap.add_argument(
        "--frames-per-preset",
        type=int,
        default=40,
        help="Number of frames to capture per preset for GIF (default: 40).",
    )
    ap.add_argument(
        "--gif-fps",
        type=int,
        default=10,
        help="GIF frame rate (default: 10 fps).",
    )
    ap.add_argument(
        "--list-presets",
        action="store_true",
        help="List available presets and exit.",
    )
    return ap.parse_args()


def _discover_presets(carla):
    """Return list of (name, WeatherParameters) discovered from the CARLA module."""
    presets = []
    for name, val in inspect.getmembers(carla.WeatherParameters):
        if isinstance(val, carla.WeatherParameters):
            presets.append((name, val))
    # Deduplicate by name and sort for stability
    seen = set()
    unique = []
    for name, val in sorted(presets, key=lambda x: x[0].lower()):
        if name in seen:
            continue
        seen.add(name)
        unique.append((name, val))
    return unique


def main() -> None:
    args = parse_args()
    _add_carla_to_path(Path(args.carla_root))

    import carla  # noqa: WPS433

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    world = client.get_world()
    original_settings = world.get_settings()

    presets = _discover_presets(carla)
    if not presets:
        raise SystemExit("No WeatherParameters presets discovered from CARLA API.")
    if args.list_presets:
        print("Available presets:")
        for name, _ in presets:
            print(f"- {name}")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    vehicle = None
    camera = None
    actors = []
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("vehicle.*model3*")
        vehicle_bp = vehicle_bp[0] if vehicle_bp else blueprint_library.filter("vehicle.*")[0]

        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available")
        vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
        actors.append(vehicle)

        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "1280")
        camera_bp.set_attribute("image_size_y", "720")
        camera_bp.set_attribute("fov", "90")
        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, cam_transform, attach_to=vehicle)
        actors.append(camera)

        q: "Queue[carla.Image]" = Queue()
        camera.listen(q.put)

        # Warm up
        world.tick()

        for name, weather in presets:
            print(f"[INFO] Capturing weather: {name}")
            world.set_weather(weather)
            # Flush old frames
            while not q.empty():
                try:
                    q.get_nowait()
                except Empty:
                    break
            # Settle
            for _ in range(max(1, args.settle_ticks)):
                world.tick()

            frames = []
            for i in range(max(1, args.frames_per_preset)):
                world.tick()
                try:
                    img = q.get(timeout=5.0)
                except Empty:
                    print(f"[WARN] No image received for {name} at frame {i}, stopping capture")
                    break
                if i == 0:
                    # Save first frame as PNG for convenience
                    img.save_to_disk(str(output_dir / f"{name}.png"))
                arr = np.frombuffer(img.raw_data, dtype=np.uint8)
                arr = arr.reshape((img.height, img.width, 4))[:, :, :3]
                frames.append(arr)

            if not frames:
                continue

            # Try to save GIF; fall back to per-frame PNGs if imageio unavailable
            gif_path = output_dir / f"{name}.gif"
            try:
                import imageio.v2 as imageio  # type: ignore

                imageio.mimsave(gif_path, frames, fps=args.gif_fps)
            except Exception as exc:
                print(f"[WARN] Could not write GIF for {name} ({exc}); writing frame sequence instead")
                frame_dir = output_dir / f"{name}_frames"
                frame_dir.mkdir(parents=True, exist_ok=True)
                for idx, frame in enumerate(frames):
                    img_path = frame_dir / f"{idx:03d}.png"
                    try:
                        from PIL import Image  # type: ignore

                        Image.fromarray(frame).save(img_path)
                    except Exception:
                        np.save(str(img_path.with_suffix(".npy")), frame)
    finally:
        if camera:
            camera.stop()
        for actor in actors:
            try:
                actor.destroy()
            except Exception:
                pass
        world.apply_settings(original_settings)


if __name__ == "__main__":
    main()
