#!/usr/bin/env python3
"""Convenience launcher for running custom CARLA leaderboard evaluations."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from setup_scenario_from_zip import prepare_routes_from_zip, parse_route_metadata


NIGHT_WEATHER = {
    "cloudiness": 20.0,
    "precipitation": 0.0,
    "precipitation_deposits": 0.0,
    "wind_intensity": 5.0,
    "sun_azimuth_angle": 300.0,
    "sun_altitude_angle": -15.0,
    "wetness": 10.0,
    "fog_distance": 80.0,
    "fog_density": 5.0,
    "fog_falloff": 2.0,
}

CLOUDY_WEATHER = {
    "cloudiness": 100.0,
    "precipitation": 15.0,
    "precipitation_deposits": 25.0,
    "wind_intensity": 45.0,
    "sun_azimuth_angle": 210.0,
    "sun_altitude_angle": 20.0,
    "wetness": 35.0,
    "fog_distance": 70.0,
    "fog_density": 18.0,
    "fog_falloff": 1.5,
}

OVERCAST_WEATHER = {
    "cloudiness": 100.0,
    "precipitation": 0.0,
    "precipitation_deposits": 0.0,
    "wind_intensity": 25.0,
    "sun_azimuth_angle": 210.0,
    "sun_altitude_angle": 45.0,
    "wetness": 20.0,
    "fog_distance": 60.0,
    "fog_density": 20.0,
    "fog_falloff": 1.5,
}

DEFAULT_NPC_PARAMETER = (
    "simulation/leaderboard/leaderboard/scenarios/scenario_parameter_Interdrive_npc.yaml"
)

CARLA_STARTUP_TIMEOUT = 60.0
CARLA_RESTART_WAIT = 2.0
CARLA_CRASH_MULTIPLIER = 2.0
CARLA_PORT_TRIES = 8
CARLA_PORT_STEP = 1


@dataclass
class VariantSpec:
    label: str
    suffix: str | None
    description: str
    weather: dict[str, float] | None = None
    scenario_parameter: str | None = None


@dataclass
class NegotiationMode:
    label: str
    suffix: str | None
    description: str
    disable_comm: bool


@dataclass(frozen=True)
class PlannerSpec:
    agent: str
    agent_config: str


PLANNER_SPECS: dict[str, PlannerSpec] = {
    "autopilot": PlannerSpec(
        agent="simulation/leaderboard/team_code/auto_pilot.py",
        agent_config="simulation/leaderboard/team_code/agent_config/colmdriver_config.yaml",
    ),
    "colmdriver": PlannerSpec(
        agent="simulation/leaderboard/team_code/colmdriver_agent.py",
        agent_config="simulation/leaderboard/team_code/agent_config/colmdriver_config.yaml",
    ),
    "colmdriver_rulebase": PlannerSpec(
        agent="simulation/leaderboard/team_code/colmdriver_agent.py",
        agent_config="simulation/leaderboard/team_code/agent_config/colmdriver_rulebase_config.yaml",
    ),
    "codriving": PlannerSpec(
        agent="simulation/leaderboard/team_code/pnp_agent_e2e_v2v.py",
        agent_config="simulation/leaderboard/team_code/agent_config/pnp_config_codriving_5_10.yaml",
    ),
    "tcp": PlannerSpec(
        agent="simulation/leaderboard/team_code/tcp_agent.py",
        agent_config="simulation/leaderboard/team_code/agent_config/tcp_5_10_config.yaml",
    ),
    "log-replay": PlannerSpec(
        agent="simulation/leaderboard/team_code/tcp_agent.py",
        agent_config="simulation/leaderboard/team_code/agent_config/tcp_5_10_config.yaml",
    ),
    "vad": PlannerSpec(
        agent="simulation/leaderboard/team_code/vad_b2d_agent.py",
        agent_config="simulation/leaderboard/team_code/agent_config/pnp_config_vad.yaml",
    ),
    "uniad": PlannerSpec(
        agent="simulation/leaderboard/team_code/uniad_b2d_agent.py",
        agent_config="simulation/leaderboard/team_code/agent_config/uniad.yaml",
    ),
    "lmdrive": PlannerSpec(
        agent="simulation/leaderboard/team_code/lmdriver_agent.py",
        agent_config="simulation/leaderboard/team_code/agent_config/lmdriver_config_8_10.py",
    ),
}

DEFAULT_PLANNER = "colmdriver"
DEFAULT_AGENT = PLANNER_SPECS[DEFAULT_PLANNER].agent
DEFAULT_AGENT_CONFIG = PLANNER_SPECS[DEFAULT_PLANNER].agent_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--zip", type=Path, help="ZIP produced by scenario_builder.")
    group.add_argument(
        "--routes-dir",
        type=Path,
        help="Existing routes directory prepared for the leaderboard.",
    )

    parser.add_argument(
        "--scenario-name",
        help="Name for the scenario (defaults to ZIP stem or routes-dir name).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("simulation/leaderboard/data/CustomRoutes"),
        help="Where to place extracted routes when using --zip.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing scenario directory if it already exists.",
    )
    parser.add_argument(
        "--results-tag",
        help="Folder tag under results/. Defaults to scenario name.",
    )
    parser.add_argument(
        "--results-subdir",
        help=(
            "Optional subfolder appended under results tag. Useful for multi-scenario runs "
            "to keep a shared results tag with per-scenario subfolders."
        ),
    )
    parser.add_argument(
        "--routes-leaf",
        default=None,
        help=(
            "Optional leaf subfolder name to locate routes inside each scenario folder. "
            "If set, will look for <routes-dir>/*/<routes-leaf> before falling back to <routes-dir>/*."
        ),
    )
    parser.add_argument(
        "--ego-num",
        type=int,
        help="Override the number of ego vehicles (auto-detected otherwise).",
    )
    parser.add_argument("--port", type=int, default=2014, help="CARLA server port.")
    parser.add_argument(
        "--traffic-manager-port",
        type=int,
        dest="tm_port",
        help="Traffic Manager port (defaults to port + 5).",
    )
    parser.add_argument(
        "--start-carla",
        action="store_true",
        help="Launch a local CARLA server from CARLA_ROOT/carla912 before evaluation.",
    )
    parser.add_argument(
        "--carla-arg",
        action="append",
        default=[],
        help="Extra arguments to pass to CarlaUE4.sh (repeatable).",
    )
    parser.add_argument(
        "--carla-port-tries",
        type=int,
        default=CARLA_PORT_TRIES,
        help="How many ports to try if the desired CARLA port is already in use.",
    )
    parser.add_argument(
        "--carla-port-step",
        type=int,
        default=CARLA_PORT_STEP,
        help="Port increment to use when searching for a free CARLA port.",
    )
    parser.add_argument(
        "--carla-crash-multiplier",
        type=float,
        default=CARLA_CRASH_MULTIPLIER,
        help=(
            "If a scenario runs longer than timeout * multiplier (and other factors), "
            "assume CARLA has crashed and restart it (multi-scenario runs only)."
        ),
    )
    parser.add_argument(
        "--carla-restart-wait",
        type=float,
        default=CARLA_RESTART_WAIT,
        help="Seconds to wait after restarting CARLA before starting the next scenario.",
    )
    parser.add_argument(
        "--scenario-parameter",
        default="simulation/leaderboard/leaderboard/scenarios/scenario_parameter_Interdrive_no_npc.yaml",
        help="Scenario parameter YAML.",
    )
    parser.add_argument(
        "--scenarios",
        default="simulation/leaderboard/data/scenarios/no_scenarios.json",
        help="Scenario JSON to load alongside routes.",
    )
    parser.add_argument(
        "--planner",
        choices=sorted(PLANNER_SPECS.keys()),
        help=(
            "Select a predefined planner preset (sets --agent/--agent-config defaults). "
            f"Defaults to {DEFAULT_PLANNER} if not set."
        ),
    )
    parser.add_argument(
        "--agent",
        default=None,
        help="Path to the evaluation agent Python file (overrides --planner default).",
    )
    parser.add_argument(
        "--agent-config",
        default=None,
        help="Agent configuration file (overrides --planner default).",
    )
    parser.add_argument("--repetitions", type=int, default=1, help="Route repetitions.")
    parser.add_argument("--track", default="SENSORS", help="Leaderboard track name.")
    parser.add_argument("--timeout", type=float, default=600.0, help="Route timeout.")
    parser.add_argument("--debug", action="store_true", help="Enable debug output.")
    parser.add_argument(
        "--capture-actor-images",
        action="store_true",
        help="Save a snapshot image for each spawned custom actor.",
    )
    parser.add_argument(
        "--normalize-actor-z",
        action="store_true",
        help="Adjust custom actor Z to nearest ground height without changing x/y or rotation.",
    )
    parser.add_argument(
        "--spawn-retry-xy-offsets",
        default="0.0,0.25,-0.25,0.5,-0.5,1.0,-1.0",
        help=(
            "Comma-separated XY offsets (meters) used for custom-actor spawn retries. "
            "All pairwise (dx,dy) combinations are attempted in increasing offset cost."
        ),
    )
    parser.add_argument(
        "--spawn-retry-z-offsets",
        default="0.0,0.2,-0.2,0.5,-0.5,1.0",
        help=(
            "Comma-separated Z offsets (meters) used with XY retry combinations for "
            "custom-actor spawn retries."
        ),
    )
    parser.add_argument(
        "--spawn-retry-max-attempts",
        type=int,
        default=180,
        help="Maximum number of custom-actor spawn retry offset candidates per actor.",
    )
    parser.add_argument(
        "--image-save-interval",
        type=int,
        default=None,
        help="Frames between saved images for TCP/log-replay (env: TCP_SAVE_INTERVAL).",
    )
    parser.add_argument(
        "--capture-every-sensor-frame",
        action="store_true",
        help=(
            "Save RGB/BEV images directly from camera callbacks for every sensor frame "
            "(decoupled from agent-step save cadence)."
        ),
    )
    parser.add_argument(
        "--capture-logreplay-images",
        dest="capture_logreplay_images",
        action="store_true",
        help=(
            "Save extra log-replay camera images into logreplayimages/ at the same cadence "
            "as normal agent image saves (controlled by --image-save-interval)."
        ),
    )
    parser.add_argument(
        "--capture-calibration-at-save-interval",
        dest="capture_logreplay_images",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--rgb-width",
        type=int,
        default=None,
        help="Override TCP front RGB width (env: TCP_RGB_WIDTH).",
    )
    parser.add_argument(
        "--rgb-height",
        type=int,
        default=None,
        help="Override TCP front RGB height (env: TCP_RGB_HEIGHT).",
    )
    parser.add_argument(
        "--rgb-fov",
        type=float,
        default=None,
        help="Override TCP front RGB FOV (env: TCP_RGB_FOV).",
    )
    parser.add_argument(
        "--rgb-x",
        type=float,
        default=None,
        help="Override TCP front RGB camera X (env: TCP_RGB_X).",
    )
    parser.add_argument(
        "--rgb-y",
        type=float,
        default=None,
        help="Override TCP front RGB camera Y (env: TCP_RGB_Y).",
    )
    parser.add_argument(
        "--rgb-z",
        type=float,
        default=None,
        help="Override TCP front RGB camera Z (env: TCP_RGB_Z).",
    )
    parser.add_argument(
        "--rgb-roll",
        type=float,
        default=None,
        help="Override TCP front RGB camera roll (env: TCP_RGB_ROLL).",
    )
    parser.add_argument(
        "--rgb-pitch",
        type=float,
        default=None,
        help="Override TCP front RGB camera pitch (env: TCP_RGB_PITCH).",
    )
    parser.add_argument(
        "--rgb-yaw",
        type=float,
        default=None,
        help="Override TCP front RGB camera yaw (env: TCP_RGB_YAW).",
    )
    parser.add_argument(
        "--logreplay-rgb-width",
        type=int,
        default=None,
        help="Override logreplayimages RGB width (env: TCP_LOGREPLAY_RGB_WIDTH).",
    )
    parser.add_argument(
        "--logreplay-rgb-height",
        type=int,
        default=None,
        help="Override logreplayimages RGB height (env: TCP_LOGREPLAY_RGB_HEIGHT).",
    )
    parser.add_argument(
        "--logreplay-rgb-fov",
        type=float,
        default=None,
        help="Override logreplayimages RGB FOV (env: TCP_LOGREPLAY_RGB_FOV).",
    )
    parser.add_argument(
        "--logreplay-rgb-x",
        type=float,
        default=None,
        help="Override logreplayimages RGB camera X (env: TCP_LOGREPLAY_RGB_X).",
    )
    parser.add_argument(
        "--logreplay-rgb-y",
        type=float,
        default=None,
        help="Override logreplayimages RGB camera Y (env: TCP_LOGREPLAY_RGB_Y).",
    )
    parser.add_argument(
        "--logreplay-rgb-z",
        type=float,
        default=None,
        help="Override logreplayimages RGB camera Z (env: TCP_LOGREPLAY_RGB_Z).",
    )
    parser.add_argument(
        "--logreplay-rgb-roll",
        type=float,
        default=None,
        help="Override logreplayimages RGB camera roll (env: TCP_LOGREPLAY_RGB_ROLL).",
    )
    parser.add_argument(
        "--logreplay-rgb-pitch",
        type=float,
        default=None,
        help="Override logreplayimages RGB camera pitch (env: TCP_LOGREPLAY_RGB_PITCH).",
    )
    parser.add_argument(
        "--logreplay-rgb-yaw",
        type=float,
        default=None,
        help="Override logreplayimages RGB camera yaw (env: TCP_LOGREPLAY_RGB_YAW).",
    )
    parser.add_argument(
        "--bev-width",
        type=int,
        default=None,
        help="Override TCP BEV camera width (env: TCP_BEV_WIDTH).",
    )
    parser.add_argument(
        "--bev-height",
        type=int,
        default=None,
        help="Override TCP BEV camera height (env: TCP_BEV_HEIGHT).",
    )
    parser.add_argument(
        "--bev-fov",
        type=float,
        default=None,
        help="Override TCP BEV camera FOV (env: TCP_BEV_FOV).",
    )
    parser.add_argument(
        "--log-replay-actors",
        action="store_true",
        help="Replay custom actors using per-waypoint timing if present in XML.",
    )
    parser.add_argument(
        "--smooth-log-replay-vehicles",
        dest="smooth_log_replay_vehicles",
        action="store_true",
        default=None,
        help="Enable adaptive smoothing for logged NPC vehicle transforms.",
    )
    parser.add_argument(
        "--no-smooth-log-replay-vehicles",
        dest="smooth_log_replay_vehicles",
        action="store_false",
        help="Disable adaptive smoothing for logged NPC vehicle transforms.",
    )
    parser.add_argument(
        "--log-replay-smooth-min-cutoff",
        type=float,
        default=None,
        help="One-Euro min cutoff for vehicle x/y smoothing (env: CUSTOM_LOG_REPLAY_SMOOTH_MIN_CUTOFF).",
    )
    parser.add_argument(
        "--log-replay-smooth-beta",
        type=float,
        default=None,
        help="One-Euro beta for vehicle x/y smoothing (env: CUSTOM_LOG_REPLAY_SMOOTH_BETA).",
    )
    parser.add_argument(
        "--log-replay-smooth-yaw-min-cutoff",
        type=float,
        default=None,
        help="One-Euro min cutoff for vehicle yaw smoothing (env: CUSTOM_LOG_REPLAY_SMOOTH_YAW_MIN_CUTOFF).",
    )
    parser.add_argument(
        "--log-replay-smooth-yaw-beta",
        type=float,
        default=None,
        help="One-Euro beta for vehicle yaw smoothing (env: CUSTOM_LOG_REPLAY_SMOOTH_YAW_BETA).",
    )
    parser.add_argument(
        "--log-replay-smooth-z-min-cutoff",
        type=float,
        default=None,
        help="One-Euro min cutoff for vehicle z smoothing (env: CUSTOM_LOG_REPLAY_SMOOTH_Z_MIN_CUTOFF).",
    )
    parser.add_argument(
        "--log-replay-smooth-z-beta",
        type=float,
        default=None,
        help="One-Euro beta for vehicle z smoothing (env: CUSTOM_LOG_REPLAY_SMOOTH_Z_BETA).",
    )
    parser.add_argument(
        "--log-replay-smooth-d-cutoff",
        type=float,
        default=None,
        help="One-Euro derivative cutoff for vehicle smoothing (env: CUSTOM_LOG_REPLAY_SMOOTH_D_CUTOFF).",
    )
    parser.add_argument(
        "--log-replay-lateral-damping",
        type=float,
        default=None,
        help="Lateral jitter damping strength in [0,1] (env: CUSTOM_LOG_REPLAY_LATERAL_DAMPING).",
    )
    parser.add_argument(
        "--log-replay-lateral-passes",
        type=int,
        default=None,
        help="Number of lateral jitter suppression passes (env: CUSTOM_LOG_REPLAY_LATERAL_PASSES).",
    )
    parser.add_argument(
        "--log-replay-lateral-max-correction",
        type=float,
        default=None,
        help="Max per-pass lateral correction in meters (env: CUSTOM_LOG_REPLAY_LATERAL_MAX_CORRECTION).",
    )
    parser.add_argument(
        "--log-replay-lateral-turn-keep",
        type=float,
        default=None,
        help="Keep factor in turns in [0,1] (env: CUSTOM_LOG_REPLAY_LATERAL_TURN_KEEP).",
    )
    parser.add_argument(
        "--log-replay-lateral-turn-angle-deg",
        type=float,
        default=None,
        help="Turn-angle threshold in degrees (env: CUSTOM_LOG_REPLAY_LATERAL_TURN_ANGLE_DEG).",
    )
    parser.add_argument(
        "--log-replay-time-lead",
        type=float,
        default=None,
        help=(
            "Replay time lead in seconds to compensate one-tick sensor/update lag "
            "(env: CUSTOM_LOG_REPLAY_TIME_LEAD)."
        ),
    )
    parser.add_argument(
        "--stabilize-static-log-replay-vehicles",
        dest="stabilize_static_log_replay_vehicles",
        action="store_true",
        default=None,
        help="Freeze near-static replay vehicle segments to remove parked-car jitter.",
    )
    parser.add_argument(
        "--no-stabilize-static-log-replay-vehicles",
        dest="stabilize_static_log_replay_vehicles",
        action="store_false",
        help="Disable near-static replay vehicle segment stabilization.",
    )
    parser.add_argument(
        "--static-log-replay-total-disp",
        type=float,
        default=None,
        help="Global static hold threshold for start-end XY displacement in meters.",
    )
    parser.add_argument(
        "--static-log-replay-total-path",
        type=float,
        default=None,
        help="Global static hold threshold for cumulative XY path length in meters.",
    )
    parser.add_argument(
        "--static-log-replay-window-min-duration",
        type=float,
        default=None,
        help="Minimum near-static segment duration in seconds before freezing.",
    )
    parser.add_argument(
        "--static-log-replay-window-max-disp",
        type=float,
        default=None,
        help="Maximum net XY displacement (meters) for a near-static segment window.",
    )
    parser.add_argument(
        "--static-log-replay-window-max-speed",
        type=float,
        default=None,
        help="Maximum XY speed (m/s) for near-static segment detection.",
    )
    parser.add_argument(
        "--static-log-replay-window-max-yaw-delta",
        type=float,
        default=None,
        help="Maximum per-step yaw delta (deg) for near-static segment detection.",
    )
    parser.add_argument(
        "--animate-walkers",
        dest="animate_walkers",
        action="store_true",
        default=None,
        help="Animate walker replay via WalkerControl instead of teleport-only transforms.",
    )
    parser.add_argument(
        "--no-animate-walkers",
        dest="animate_walkers",
        action="store_false",
        help="Disable walker replay animation and use transform replay only.",
    )
    parser.add_argument(
        "--walker-max-speed",
        type=float,
        default=None,
        help="Max walker replay speed in m/s (env: CUSTOM_LOG_REPLAY_WALKER_MAX_SPEED).",
    )
    parser.add_argument(
        "--walker-teleport-distance",
        type=float,
        default=None,
        help=(
            "Distance threshold in meters where walker replay snaps to target instead of control "
            "(env: CUSTOM_LOG_REPLAY_WALKER_TELEPORT_DIST)."
        ),
    )
    parser.add_argument(
        "--walker-ground-lift",
        type=float,
        default=None,
        help="Extra z offset applied after ground snap for walkers (env: CUSTOM_WALKER_GROUND_LIFT).",
    )
    parser.add_argument(
        "--normalize-ego-z",
        default=True,
        action="store_true",
        help="Snap ego vehicle Z to nearest driving road height before spawning.",
    )
    parser.add_argument(
        "--align-ego-routes",
        action="store_true",
        help=(
            "Align ego route waypoints to the CARLA map using the scenario_generator "
            "route alignment module (uses CARLA GRP to snap headings/lanes)."
        ),
    )
    parser.add_argument(
        "--align-ego-host",
        default="127.0.0.1",
        help="CARLA host for ego route alignment (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--align-ego-port",
        type=int,
        default=None,
        help="CARLA port for ego route alignment (default: --port).",
    )
    parser.add_argument(
        "--align-ego-sampling-resolution",
        type=float,
        default=2.0,
        help="GRP sampling resolution (meters) used for ego route alignment.",
    )
    parser.add_argument(
        "--llm-port",
        type=int,
        default=8888,
        help="Port for LLM server (CoLMDriver only, default: 8888).",
    )
    parser.add_argument(
        "--vlm-port",
        type=int,
        default=1111,
        help="Port for VLM server (CoLMDriver only, default: 1111).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous checkpoint if available.",
    )
    parser.add_argument(
        "--skip-existed",
        action="store_true",
        default=True,
        help="Skip routes that already have results (default: on).",
    )
    parser.add_argument(
        "--no-skip-existed",
        dest="skip_existed",
        action="store_false",
        help="Force rerun even if results exist.",
    )
    parser.add_argument(
        "--carla-seed", type=int, default=2000, help="Seed for CARLA world randomisation."
    )
    parser.add_argument(
        "--traffic-seed",
        type=int,
        default=2000,
        help="Seed for CARLA Traffic Manager.",
    )
    parser.add_argument(
        "--record",
        default="",
        help="Optional CARLA recording file path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command and exit without running the evaluator.",
    )
    parser.add_argument(
        "--add-night",
        action="store_true",
        help="Also evaluate the same scenario with a forced night-time weather override.",
    )
    parser.add_argument(
        "--add-cloudy",
        action="store_true",
        help="Also evaluate the same scenario under heavy-cloud conditions.",
    )
    parser.add_argument(
        "--weather",
        choices=["clear", "cloudy", "overcast", "night"],
        default=None,
        help="Run a single weather variant and tag results with the weather name.",
    )
    parser.add_argument(
        "--add-npcs",
        action="store_true",
        help="Also evaluate with CARLA's random traffic/pedestrians enabled.",
    )
    parser.add_argument(
        "--npc-scenario-parameter",
        default=DEFAULT_NPC_PARAMETER,
        help=(
            "Scenario parameter YAML to use whenever --add-npcs is supplied "
            f"(default: {DEFAULT_NPC_PARAMETER})."
        ),
    )
    parser.add_argument(
        "--add-no-negotiation",
        action="store_true",
        help=(
            "For every scenario variant, also run a copy with inter-ego negotiation disabled."
        ),
    )
    return parser.parse_args()


def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_for_port(host: str, port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_port_open(host, port, timeout=1.0):
            return True
        time.sleep(0.5)
    return False


def wait_for_port_close(host: str, port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_port_open(host, port, timeout=1.0):
            return True
        time.sleep(0.5)
    return False


def find_available_port(host: str, preferred_port: int, tries: int, step: int) -> int:
    attempts = max(1, tries)
    step = max(1, step)
    for idx in range(attempts):
        port = preferred_port + idx * step
        if port > 65535:
            break
        if not is_port_open(host, port, timeout=1.0):
            return port
    raise RuntimeError(
        f"No free CARLA port found starting at {preferred_port} "
        f"(tries={attempts}, step={step})."
    )


def should_add_offscreen(extra_args: List[str]) -> bool:
    if os.environ.get("DISPLAY"):
        return False
    lowered = {arg.lower() for arg in extra_args}
    if any("renderoffscreen" in arg for arg in lowered):
        return False
    if any("windowed" in arg for arg in lowered):
        return False
    return True


class CarlaProcessManager:
    def __init__(
        self,
        carla_root: Path,
        host: str,
        port: int,
        extra_args: List[str],
        env: Dict[str, str],
        port_tries: int,
        port_step: int,
    ) -> None:
        self.carla_root = carla_root
        self.host = host
        self.port = port
        self.extra_args = list(extra_args)
        self.env = env
        self.port_tries = port_tries
        self.port_step = port_step
        self.process: subprocess.Popen | None = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self) -> int:
        if self.is_running():
            return self.port
        carla_script = self.carla_root / "CarlaUE4.sh"
        if not carla_script.exists():
            raise FileNotFoundError(carla_script)
        if is_port_open(self.host, self.port):
            new_port = find_available_port(
                self.host,
                self.port,
                tries=self.port_tries,
                step=self.port_step,
            )
            if new_port != self.port:
                print(
                    f"[INFO] Port {self.port} is busy. Switching CARLA to port {new_port}."
                )
                self.port = new_port
        cmd = [str(carla_script), f"--world-port={self.port}"]
        if should_add_offscreen(self.extra_args):
            cmd.append("-RenderOffScreen")
        cmd.extend(self.extra_args)
        print(f"[INFO] Starting CARLA: {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd,
            cwd=str(self.carla_root),
            env=self.env,
        )
        if not wait_for_port(self.host, self.port, CARLA_STARTUP_TIMEOUT):
            print(
                f"[WARN] CARLA port {self.port} did not open within {CARLA_STARTUP_TIMEOUT:.0f}s."
            )
        return self.port

    def stop(self, timeout: float = 15.0) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            print("[INFO] Stopping CARLA...")
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=timeout)
        self.process = None

    def restart(self, wait_seconds: float) -> int:
        self.stop()
        wait_for_port_close(self.host, self.port, timeout=10.0)
        time.sleep(max(0.0, wait_seconds))
        return self.start()


_active_carla_manager: CarlaProcessManager | None = None
_signal_handlers_installed = False


def _install_signal_handlers() -> None:
    global _signal_handlers_installed
    if _signal_handlers_installed:
        return

    def _handler(signum: int, frame: object | None) -> None:
        if _active_carla_manager is not None:
            _active_carla_manager.stop()
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        sys.exit(0)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    _signal_handlers_installed = True


def count_variants(args: argparse.Namespace) -> int:
    if args.weather:
        return 1
    count = 1
    if args.add_night:
        count += 1
    if args.add_cloudy:
        count += 1
    if args.add_npcs:
        count += 1
    return count


def count_negotiation_modes(args: argparse.Namespace) -> int:
    return 2 if args.add_no_negotiation else 1


def estimate_scenario_timeout(args: argparse.Namespace, routes_dir: Path) -> float:
    ego_routes = detect_ego_routes(routes_dir)
    route_count = max(1, ego_routes)
    expected_runs = max(1, args.repetitions) * count_variants(args) * count_negotiation_modes(args)
    base = float(args.timeout) * route_count * expected_runs
    return base * float(args.carla_crash_multiplier)


def append_pythonpath(env: Dict[str, str], path: Path) -> None:
    path_str = str(path)
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{path_str}:{current}" if current else path_str


def find_carla_egg(carla_root: Path) -> Path:
    """
    Locate a CARLA Python client build compatible with the current interpreter.

    Priority:
    1) Wheels/eggs under PythonAPI/carla/dist whose filename matches our py3{minor} or cp3{minor}.
    2) Same search in PythonAPI_docker/carla/dist (useful when only docker-built eggs exist).
    3) Any py3*/cp3* egg/wheel under those dirs.
    4) Fallback to the first available artifact.
    """
    search_dirs = [
        carla_root / "PythonAPI" / "carla" / "dist",
        carla_root / "PythonAPI_docker" / "carla" / "dist",
    ]
    artifacts: list[Path] = []
    for dist_dir in search_dirs:
        artifacts.extend(sorted(dist_dir.glob("carla-*.egg")))
        artifacts.extend(sorted(dist_dir.glob("carla-*.whl")))

    if not artifacts:
        raise FileNotFoundError(
            f"No CARLA egg/wheel found under {search_dirs[0]} or {search_dirs[1]}"
        )

    minor = sys.version_info.minor
    def matches_minor(art: Path) -> bool:
        name = art.name
        return (
            f"cp3{minor}" in name
            or f"py3{minor}" in name
            or f"py3.{minor}" in name
        )

    preferred = [art for art in artifacts if matches_minor(art)]
    if preferred:
        return preferred[0]

    py3 = [art for art in artifacts if "py3" in art.name or "cp3" in art.name]
    if py3:
        return py3[0]

    return artifacts[0]


def detect_ego_routes(routes_dir: Path) -> int:
    count = 0
    for xml_path in routes_dir.rglob("*.xml"):
        try:
            _, _, role = parse_route_metadata(xml_path.read_bytes())
        except Exception:
            continue
        if role == "ego":
            count += 1
    return count


def detect_scenario_subfolders(routes_dir: Path, routes_leaf: str | None = None) -> List[tuple[Path, str]]:
    """
    Check if routes_dir contains multiple scenario subfolders (each with their own ego routes).
    
    Returns a list of subfolders that each contain at least one ego route.
    If routes_dir itself contains ego routes directly (not in subfolders), returns empty list.
    """
    # First check if there are ego routes directly in routes_dir (not in subfolders)
    for xml_path in routes_dir.glob("*.xml"):
        try:
            _, _, role = parse_route_metadata(xml_path.read_bytes())
            if role == "ego":
                # routes_dir itself is a scenario directory
                return []
        except Exception:
            continue
    
    # Check subfolders for ego routes
    scenario_folders: list[tuple[Path, str]] = []
    for subdir in sorted(routes_dir.iterdir()):
        if not subdir.is_dir():
            continue
        candidates = []
        if routes_leaf:
            leaf = subdir / routes_leaf
            if leaf.is_dir():
                candidates.append(leaf)
        candidates.append(subdir)

        selected = None
        selected_name = None
        for candidate in candidates:
            has_ego = False
            for xml_path in candidate.rglob("*.xml"):
                try:
                    _, _, role = parse_route_metadata(xml_path.read_bytes())
                    if role == "ego":
                        has_ego = True
                        break
                except Exception:
                    continue
            if has_ego:
                selected = candidate
                selected_name = subdir.name
                break

        if selected is not None and selected_name is not None:
            scenario_folders.append((selected, selected_name))
    
    return scenario_folders


def read_manifest(routes_dir: Path) -> tuple[Path | None, Dict[str, List[Dict[str, str]]] | None]:
    manifest_path = routes_dir / "actors_manifest.json"
    if not manifest_path.exists():
        return None, None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return manifest_path, None
    return manifest_path, data


def apply_weather_override(routes_dir: Path, weather: dict[str, float]) -> None:
    """Inject/replace weather blocks in every route XML under routes_dir."""
    for xml_path in routes_dir.rglob("*.xml"):
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            continue
        root = tree.getroot()
        route_nodes = list(root.iter("route"))
        if not route_nodes:
            continue
        for route in route_nodes:
            for weather_node in list(route.findall("weather")):
                route.remove(weather_node)
            weather_node = ET.SubElement(route, "weather")
            for key, value in weather.items():
                weather_node.set(key, f"{value}")
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)


def clone_routes_with_weather_override(
    src: Path, weather: dict[str, float] | None
) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    """Create a temporary copy of routes with an optional weather override applied."""
    if weather is None:
        return src, None
    temp_dir = tempfile.TemporaryDirectory(prefix=f"{src.name}_")
    clone_root = Path(temp_dir.name) / src.name
    shutil.copytree(src, clone_root)
    apply_weather_override(clone_root, weather)
    return clone_root, temp_dir


def align_ego_routes_in_directory(
    routes_dir: Path,
    town: str,
    carla_host: str,
    carla_port: int,
    sampling_resolution: float,
) -> None:
    """
    Align only ego route XMLs in routes_dir using the scenario_generator alignment module.
    """
    try:
        from scenario_generator.pipeline.step_07_route_alignment import align_route_file
        from scenario_generator.pipeline.step_07_route_alignment import main as align_mod
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Ego route alignment unavailable (import failed): {exc}")
        return

    try:
        carla, GlobalRoutePlanner, GlobalRoutePlannerDAO = align_mod._ensure_carla()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Ego route alignment unavailable (CARLA import failed): {exc}")
        return

    try:
        print(f"[INFO] Aligning ego routes with CARLA at {carla_host}:{carla_port} (town={town})...")
        client = carla.Client(carla_host, carla_port)
        client.set_timeout(30.0)
        world = client.get_world()
        current_map = world.get_map().name
        current_map_base = current_map.split("/")[-1] if "/" in current_map else current_map
        if current_map_base != town:
            world = client.load_world(town)
        carla_map = world.get_map()

        try:
            grp = GlobalRoutePlanner(carla_map, sampling_resolution)
        except TypeError:
            grp = GlobalRoutePlanner(GlobalRoutePlannerDAO(carla_map, sampling_resolution))
        if hasattr(grp, "setup"):
            grp.setup()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Ego route alignment failed to initialize CARLA/GRP: {exc}")
        return

    ego_xmls: List[Path] = []
    for xml_path in routes_dir.glob("*.xml"):
        try:
            _, _, role = parse_route_metadata(xml_path.read_bytes())
        except Exception:
            continue
        if role == "ego":
            ego_xmls.append(xml_path)

    if not ego_xmls:
        print(f"[WARN] No ego route XMLs found for alignment in {routes_dir}.")
        return

    for xml_path in ego_xmls:
        try:
            orig, aligned = align_route_file(
                xml_path,
                carla_map,
                grp,
                backup=False,
            )
            if orig != aligned:
                print(f"[INFO] Aligned {xml_path.name}: {orig} -> {aligned} waypoints")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WARN] Failed to align {xml_path.name}: {exc}")


def main() -> None:
    global _active_carla_manager
    args = parse_args()
    if args.planner:
        planner = PLANNER_SPECS[args.planner]
        if args.agent is None:
            args.agent = planner.agent
        if args.agent_config is None:
            args.agent_config = planner.agent_config
    else:
        if args.agent is None:
            args.agent = DEFAULT_AGENT
        if args.agent_config is None:
            args.agent_config = DEFAULT_AGENT_CONFIG
    repo_root = Path(__file__).resolve().parents[1]
    # Allow overriding CARLA install via environment; fallback to bundled carla912.
    carla_root = Path(os.environ.get("CARLA_ROOT", repo_root / "carla912")).expanduser().resolve()
    if not carla_root.exists():
        raise FileNotFoundError(
            f"CARLA_ROOT '{carla_root}' does not exist. "
            "Set CARLA_ROOT to a valid CARLA installation (with PythonAPI/carla/dist)."
        )

    scenario_summary = None
    routes_dir: Path
    scenario_name = args.scenario_name
    manifest_path: Path | None = None
    actors_manifest: Dict[str, List[Dict[str, str]]] | None = None

    if args.zip is not None:
        scenario_summary = prepare_routes_from_zip(
            zip_path=args.zip,
            scenario_name=args.scenario_name,
            output_root=args.output_root,
            overwrite=args.overwrite,
        )
        routes_dir = scenario_summary["output_dir"]
        scenario_name = scenario_summary["scenario_name"]
        auto_ego_num = scenario_summary["ego_count"]
        manifest_path = scenario_summary.get("manifest_path")
        actors_manifest = scenario_summary.get("actors_manifest")
    else:
        routes_dir = args.routes_dir.expanduser().resolve()
        if not routes_dir.exists():
            raise FileNotFoundError(routes_dir)
        
        # Check if routes_dir contains multiple scenario subfolders
        scenario_subfolders = detect_scenario_subfolders(routes_dir, args.routes_leaf)
        if scenario_subfolders:
            base_results_tag = args.results_tag or args.scenario_name or routes_dir.name
            print(f"Detected {len(scenario_subfolders)} scenario subfolders in {routes_dir}:")
            for sf_path, sf_name in scenario_subfolders:
                print(f"  - {sf_name} ({sf_path})")
            print("\nRunning each scenario sequentially...\n")

            carla_manager = None
            if args.start_carla and not args.dry_run:
                carla_manager = CarlaProcessManager(
                    carla_root=carla_root,
                    host="127.0.0.1",
                    port=args.port,
                    extra_args=args.carla_arg,
                    env=os.environ.copy(),
                    port_tries=args.carla_port_tries,
                    port_step=args.carla_port_step,
                )
                _active_carla_manager = carla_manager
                _install_signal_handlers()
                selected_port = carla_manager.start()
                if selected_port != args.port:
                    args.port = selected_port

            try:
                # Run each subfolder as a separate scenario
                for i, (subfolder, sub_name) in enumerate(scenario_subfolders, 1):
                    print(f"\n{'='*60}")
                    print(f"SCENARIO {i}/{len(scenario_subfolders)}: {sub_name}")
                    print(f"{'='*60}")

                    if carla_manager and not carla_manager.is_running():
                        print("[WARN] CARLA process is not running. Restarting...")
                        selected_port = carla_manager.restart(args.carla_restart_wait)
                        if selected_port != args.port:
                            args.port = selected_port

                    # Build a new command to run this subfolder
                    new_cmd = [sys.executable, __file__]
                    new_cmd.extend(["--routes-dir", str(subfolder)])
                    new_cmd.extend(["--results-tag", base_results_tag])
                    new_cmd.extend(["--results-subdir", sub_name])
                    new_cmd.extend(["--scenario-name", sub_name])
                    new_cmd.extend(["--port", str(args.port)])
                    if args.align_ego_routes:
                        new_cmd.append("--align-ego-routes")
                        new_cmd.extend(["--align-ego-host", str(args.align_ego_host)])
                        if args.align_ego_port is not None:
                            new_cmd.extend(["--align-ego-port", str(args.align_ego_port)])
                        new_cmd.extend(
                            [
                                "--align-ego-sampling-resolution",
                                str(args.align_ego_sampling_resolution),
                            ]
                        )
                    if args.planner:
                        new_cmd.extend(["--planner", args.planner])
                    if args.agent:
                        new_cmd.extend(["--agent", args.agent])
                    if args.agent_config:
                        new_cmd.extend(["--agent-config", args.agent_config])
                    new_cmd.extend(["--repetitions", str(args.repetitions)])
                    new_cmd.extend(["--track", args.track])
                    new_cmd.extend(["--timeout", str(args.timeout)])
                    new_cmd.extend(["--carla-seed", str(args.carla_seed)])
                    new_cmd.extend(["--traffic-seed", str(args.traffic_seed)])
                    if args.debug:
                        new_cmd.append("--debug")
                    if args.resume:
                        new_cmd.append("--resume")
                    if not args.skip_existed:
                        new_cmd.append("--no-skip-existed")
                    if args.add_night:
                        new_cmd.append("--add-night")
                    if args.add_cloudy:
                        new_cmd.append("--add-cloudy")
                    if args.weather:
                        new_cmd.extend(["--weather", args.weather])
                    if args.add_npcs:
                        new_cmd.append("--add-npcs")
                    if args.add_no_negotiation:
                        new_cmd.append("--add-no-negotiation")
                    if args.dry_run:
                        new_cmd.append("--dry-run")
                    if args.capture_actor_images:
                        new_cmd.append("--capture-actor-images")
                    if args.normalize_actor_z:
                        new_cmd.append("--normalize-actor-z")
                    if args.normalize_ego_z:
                        new_cmd.append("--normalize-ego-z")

                    timeout_seconds = None
                    if not args.dry_run and args.carla_crash_multiplier > 0:
                        timeout_seconds = estimate_scenario_timeout(args, subfolder)
                        print(
                            f"[INFO] Scenario timeout watchdog: {timeout_seconds:.1f}s"
                        )

                    try:
                        subprocess.run(
                            new_cmd,
                            check=True,
                            env=os.environ.copy(),
                            timeout=timeout_seconds,
                        )
                    except subprocess.TimeoutExpired:
                        print(
                            f"[WARNING] Scenario {subfolder.name} exceeded the watchdog timeout. "
                            "Assuming CARLA crashed."
                        )
                        if carla_manager:
                            print("[INFO] Restarting CARLA before continuing.")
                            selected_port = carla_manager.restart(args.carla_restart_wait)
                            if selected_port != args.port:
                                args.port = selected_port
                        else:
                            print("[WARNING] CARLA auto-restart is disabled (use --start-carla).")
                        continue
                    except subprocess.CalledProcessError as e:
                        print(
                            f"[WARNING] Scenario {subfolder.name} failed with exit code {e.returncode}"
                        )
                        continue
            finally:
                if carla_manager:
                    carla_manager.stop()
                _active_carla_manager = None

            print(f"\n{'='*60}")
            print(f"Completed all {len(scenario_subfolders)} scenarios")
            print(f"{'='*60}")
            return
        
        auto_ego_num = detect_ego_routes(routes_dir)
        if auto_ego_num == 0:
            raise RuntimeError(
                f"No ego routes located under {routes_dir}. "
                "Ensure your XML files include role=\"ego\"."
            )
        manifest_path, actors_manifest = read_manifest(routes_dir)

    ego_num = args.ego_num or auto_ego_num
    scenario_name = scenario_name or routes_dir.name
    results_tag = args.results_tag or scenario_name
    if args.results_subdir:
        results_tag = f"{results_tag}/{args.results_subdir}"

    variants: list[VariantSpec] = []
    if args.weather:
        weather_key = args.weather.lower()
        if weather_key == "night":
            weather_override = NIGHT_WEATHER
            desc = "Night-time lighting (sun altitude override)."
        elif weather_key == "overcast":
            weather_override = OVERCAST_WEATHER
            desc = "Overcast daylight (reduced sun + heavy cloud cover)."
        elif weather_key == "cloudy":
            weather_override = CLOUDY_WEATHER
            desc = "Daylight with heavy cloud cover."
        elif weather_key == "clear":
            weather_override = None
            desc = "Default daylight / clear run."
        else:
            weather_override = None
            desc = "Default daylight / clear run."
        variants.append(
            VariantSpec(
                label=weather_key,
                suffix=weather_key,
                description=desc,
                scenario_parameter=args.scenario_parameter,
                weather=weather_override,
            )
        )
    else:
        variants.append(
            VariantSpec(
                label="baseline",
                suffix=None,
                description="Default daylight / clear run.",
                scenario_parameter=args.scenario_parameter,
                weather=None,
            )
        )
        if args.add_night:
            variants.append(
                VariantSpec(
                    label="night",
                    suffix="night",
                    description="Night-time lighting (sun altitude override).",
                    weather=NIGHT_WEATHER,
                    scenario_parameter=args.scenario_parameter,
                )
            )
        if args.add_cloudy:
            variants.append(
                VariantSpec(
                    label="cloudy",
                    suffix="cloudy",
                    description="Daylight with heavy cloud cover.",
                    weather=CLOUDY_WEATHER,
                    scenario_parameter=args.scenario_parameter,
                )
            )
        if args.add_npcs:
            variants.append(
                VariantSpec(
                    label="npcs",
                    suffix="npcs",
                    description="Interdrive NPC parameter file (random traffic/pedestrians).",
                    scenario_parameter=args.npc_scenario_parameter,
                    weather=None,
                )
            )

    negotiation_modes: list[NegotiationMode] = [
        NegotiationMode(
            label="negotiation",
            suffix=None,
            description="Negotiation enabled (default).",
            disable_comm=False,
        )
    ]
    if args.add_no_negotiation:
        negotiation_modes.append(
            NegotiationMode(
                label="no-negotiation",
                suffix="nonego",
                description="Negotiation disabled; egos act independently.",
                disable_comm=True,
            )
        )

    leaderboard_root = repo_root / "simulation" / "leaderboard"
    scenario_runner_root = repo_root / "simulation" / "scenario_runner"
    data_root = repo_root / "simulation" / "assets" / "v2xverse_debug"

    env_template = os.environ.copy()
    env_template["CARLA_ROOT"] = str(carla_root)
    env_template["LEADERBOARD_ROOT"] = str(leaderboard_root)
    env_template["SCENARIO_RUNNER_ROOT"] = str(scenario_runner_root)
    env_template["DATA_ROOT"] = str(data_root)
    env_template.setdefault("COLMDRIVER_OFFLINE", "0")

    append_pythonpath(env_template, repo_root)
    append_pythonpath(env_template, scenario_runner_root)
    append_pythonpath(env_template, leaderboard_root)
    append_pythonpath(env_template, leaderboard_root / "team_code")
    append_pythonpath(env_template, carla_root / "PythonAPI")
    append_pythonpath(env_template, carla_root / "PythonAPI" / "carla")
    append_pythonpath(env_template, find_carla_egg(carla_root))

    tm_port = args.tm_port or (args.port + 5)

    evaluator = (
        leaderboard_root
        / "leaderboard"
        / "leaderboard_evaluator_parameter.py"
    )
    agent_path = (repo_root / args.agent).resolve()
    agent_config_path = (repo_root / args.agent_config).resolve()
    scenarios_json = (repo_root / args.scenarios).resolve()

    print("Scenario directory:", routes_dir)
    print("Ego vehicles:", ego_num)
    if scenario_summary is not None:
        print("Actors discovered:")
        for path, role, route_id, town in scenario_summary["routes"]:
            rel = path.relative_to(routes_dir)
            print(f"  - {role:11s} route_id={route_id:>4s} town={town} file={rel}")
    elif actors_manifest:
        print("Actors discovered:")
        for role, entries in actors_manifest.items():
            for entry in entries:
                rel = entry.get("file", "?")
                route_id = entry.get("route_id", "?")
                town = entry.get("town", "?")
                print(f"  - {role:11s} route_id={route_id:>4s} town={town} file={rel}")
    if actors_manifest:
        non_ego = {
            role: len(entries)
            for role, entries in actors_manifest.items()
            if role != "ego"
        }
        if non_ego:
            print(
                "Non-ego actor counts:",
                ", ".join(f"{role}={count}" for role, count in sorted(non_ego.items())),
            )
    if manifest_path:
        print("Actor manifest:", manifest_path)

    run_matrix = [(spec, nego) for spec in variants for nego in negotiation_modes]
    if run_matrix:
        print("\nConfigured evaluation variants:")
        for spec, nego in run_matrix:
            base_name = spec.suffix or "baseline"
            mode_note = " (no negotiation)" if nego.disable_comm else ""
            print(f"  - {base_name}{mode_note}: {spec.description} [{nego.description}]")

    carla_manager = None
    if args.start_carla and not args.dry_run:
        carla_manager = CarlaProcessManager(
            carla_root=carla_root,
            host="127.0.0.1",
            port=args.port,
            extra_args=args.carla_arg,
            env=os.environ.copy(),
            port_tries=args.carla_port_tries,
            port_step=args.carla_port_step,
        )
        _active_carla_manager = carla_manager
        _install_signal_handlers()
        selected_port = carla_manager.start()
        if selected_port != args.port:
            args.port = selected_port

    try:
        for spec, negotiation in run_matrix:
            if carla_manager and not carla_manager.is_running():
                print("[WARN] CARLA process is not running. Restarting...")
                carla_manager.restart(args.carla_restart_wait)

            variant_routes_dir, temp_dir = clone_routes_with_weather_override(routes_dir, spec.weather)
            try:
                if args.align_ego_routes:
                    if temp_dir is None:
                        temp_dir = tempfile.TemporaryDirectory(prefix=f"{routes_dir.name}_align_")
                        clone_root = Path(temp_dir.name) / routes_dir.name
                        shutil.copytree(routes_dir, clone_root)
                        variant_routes_dir = clone_root

                    align_port = args.align_ego_port or args.port
                    town = None
                    for xml_path in variant_routes_dir.glob("*.xml"):
                        try:
                            _, xml_town, role = parse_route_metadata(xml_path.read_bytes())
                        except Exception:
                            continue
                        if role == "ego":
                            town = xml_town
                            break
                    if town:
                        align_ego_routes_in_directory(
                            routes_dir=variant_routes_dir,
                            town=town,
                            carla_host=str(args.align_ego_host),
                            carla_port=int(align_port),
                            sampling_resolution=float(args.align_ego_sampling_resolution),
                        )
                    else:
                        print("[WARN] Ego route alignment requested, but no ego XMLs found to infer town.")

                env = env_template.copy()
                env["ROUTES"] = str(variant_routes_dir)
                env["ROUTES_DIR"] = str(variant_routes_dir)

                variant_manifest = variant_routes_dir / "actors_manifest.json"
                if variant_manifest.exists():
                    env["CUSTOM_ACTOR_MANIFEST"] = str(variant_manifest)
                    actors_root = variant_manifest.parent / "actors"
                    if actors_root.exists():
                        env["CUSTOM_ACTOR_ROOT"] = str(actors_root)
                    else:
                        env.pop("CUSTOM_ACTOR_ROOT", None)
                    behavior_path = variant_manifest.parent / "actors_behavior.json"
                    if behavior_path.exists():
                        env["CUSTOM_ACTOR_BEHAVIORS"] = str(behavior_path)
                    else:
                        env.pop("CUSTOM_ACTOR_BEHAVIORS", None)
                else:
                    env.pop("CUSTOM_ACTOR_MANIFEST", None)
                    env.pop("CUSTOM_ACTOR_ROOT", None)
                    env.pop("CUSTOM_ACTOR_BEHAVIORS", None)

                if negotiation.disable_comm:
                    env["COLMDRIVER_DISABLE_NEGOTIATION"] = "1"
                else:
                    env.pop("COLMDRIVER_DISABLE_NEGOTIATION", None)

                # Set LLM/VLM ports for CoLMDriver
                env["COLMDRIVER_LLM_PORT"] = str(args.llm_port)
                env["COLMDRIVER_VLM_PORT"] = str(args.vlm_port)
                if args.debug:
                    env.setdefault("CUSTOM_LOG_REPLAY_DEBUG", "1")
                    env.setdefault("CUSTOM_LOG_REPLAY_DEBUG_INTERVAL", "2.0")

                suffix_parts: list[str] = []
                if spec.suffix:
                    suffix_parts.append(spec.suffix)
                if negotiation.suffix:
                    suffix_parts.append(negotiation.suffix)
                variant_suffix = "_".join(suffix_parts) if suffix_parts else None
                variant_results_tag = results_tag if not variant_suffix else f"{results_tag}_{variant_suffix}"
                result_root = repo_root / "results" / "results_driving_custom" / variant_results_tag
                save_path = result_root / "image"
                checkpoint_endpoint = result_root / "results.json"
                save_path.mkdir(parents=True, exist_ok=True)
                checkpoint_endpoint.parent.mkdir(parents=True, exist_ok=True)

                env["RESULT_ROOT"] = str(result_root)
                env["SAVE_PATH"] = str(save_path)
                env["CHECKPOINT_ENDPOINT"] = str(checkpoint_endpoint)
                if args.capture_actor_images:
                    actor_image_dir = result_root / "actor_images"
                    actor_image_dir.mkdir(parents=True, exist_ok=True)
                    env["CUSTOM_ACTOR_IMAGE_DIR"] = str(actor_image_dir)
                else:
                    env.pop("CUSTOM_ACTOR_IMAGE_DIR", None)
                if args.normalize_actor_z:
                    env["CUSTOM_ACTOR_NORMALIZE_Z"] = "1"
                else:
                    env.pop("CUSTOM_ACTOR_NORMALIZE_Z", None)
                env["CUSTOM_ACTOR_SPAWN_XY_OFFSETS"] = str(args.spawn_retry_xy_offsets)
                env["CUSTOM_ACTOR_SPAWN_Z_OFFSETS"] = str(args.spawn_retry_z_offsets)
                env["CUSTOM_ACTOR_SPAWN_MAX_ATTEMPTS"] = str(max(0, int(args.spawn_retry_max_attempts)))
                if args.image_save_interval is not None:
                    env["TCP_SAVE_INTERVAL"] = str(args.image_save_interval)
                if args.capture_every_sensor_frame:
                    env["TCP_CAPTURE_SENSOR_FRAMES"] = "1"
                else:
                    env.pop("TCP_CAPTURE_SENSOR_FRAMES", None)
                if args.capture_logreplay_images:
                    env["TCP_CAPTURE_LOGREPLAY_IMAGES"] = "1"
                else:
                    env.pop("TCP_CAPTURE_LOGREPLAY_IMAGES", None)
                if args.rgb_width is not None:
                    env["TCP_RGB_WIDTH"] = str(args.rgb_width)
                if args.rgb_height is not None:
                    env["TCP_RGB_HEIGHT"] = str(args.rgb_height)
                if args.rgb_fov is not None:
                    env["TCP_RGB_FOV"] = str(args.rgb_fov)
                if args.rgb_x is not None:
                    env["TCP_RGB_X"] = str(args.rgb_x)
                if args.rgb_y is not None:
                    env["TCP_RGB_Y"] = str(args.rgb_y)
                if args.rgb_z is not None:
                    env["TCP_RGB_Z"] = str(args.rgb_z)
                if args.rgb_roll is not None:
                    env["TCP_RGB_ROLL"] = str(args.rgb_roll)
                if args.rgb_pitch is not None:
                    env["TCP_RGB_PITCH"] = str(args.rgb_pitch)
                if args.rgb_yaw is not None:
                    env["TCP_RGB_YAW"] = str(args.rgb_yaw)
                if args.logreplay_rgb_width is not None:
                    env["TCP_LOGREPLAY_RGB_WIDTH"] = str(args.logreplay_rgb_width)
                if args.logreplay_rgb_height is not None:
                    env["TCP_LOGREPLAY_RGB_HEIGHT"] = str(args.logreplay_rgb_height)
                if args.logreplay_rgb_fov is not None:
                    env["TCP_LOGREPLAY_RGB_FOV"] = str(args.logreplay_rgb_fov)
                if args.logreplay_rgb_x is not None:
                    env["TCP_LOGREPLAY_RGB_X"] = str(args.logreplay_rgb_x)
                if args.logreplay_rgb_y is not None:
                    env["TCP_LOGREPLAY_RGB_Y"] = str(args.logreplay_rgb_y)
                if args.logreplay_rgb_z is not None:
                    env["TCP_LOGREPLAY_RGB_Z"] = str(args.logreplay_rgb_z)
                if args.logreplay_rgb_roll is not None:
                    env["TCP_LOGREPLAY_RGB_ROLL"] = str(args.logreplay_rgb_roll)
                if args.logreplay_rgb_pitch is not None:
                    env["TCP_LOGREPLAY_RGB_PITCH"] = str(args.logreplay_rgb_pitch)
                if args.logreplay_rgb_yaw is not None:
                    env["TCP_LOGREPLAY_RGB_YAW"] = str(args.logreplay_rgb_yaw)
                if args.bev_width is not None:
                    env["TCP_BEV_WIDTH"] = str(args.bev_width)
                if args.bev_height is not None:
                    env["TCP_BEV_HEIGHT"] = str(args.bev_height)
                if args.bev_fov is not None:
                    env["TCP_BEV_FOV"] = str(args.bev_fov)
                if args.log_replay_actors:
                    env["CUSTOM_ACTOR_LOG_REPLAY"] = "1"
                else:
                    env.pop("CUSTOM_ACTOR_LOG_REPLAY", None)
                if args.planner == "log-replay":
                    env["CUSTOM_EGO_LOG_REPLAY"] = "1"
                    env["CUSTOM_ACTOR_LOG_REPLAY"] = "1"
                else:
                    env.pop("CUSTOM_EGO_LOG_REPLAY", None)
                if args.smooth_log_replay_vehicles is not None:
                    env["CUSTOM_LOG_REPLAY_SMOOTH_VEHICLES"] = (
                        "1" if args.smooth_log_replay_vehicles else "0"
                    )
                if args.log_replay_smooth_min_cutoff is not None:
                    env["CUSTOM_LOG_REPLAY_SMOOTH_MIN_CUTOFF"] = str(
                        float(args.log_replay_smooth_min_cutoff)
                    )
                if args.log_replay_smooth_beta is not None:
                    env["CUSTOM_LOG_REPLAY_SMOOTH_BETA"] = str(float(args.log_replay_smooth_beta))
                if args.log_replay_smooth_yaw_min_cutoff is not None:
                    env["CUSTOM_LOG_REPLAY_SMOOTH_YAW_MIN_CUTOFF"] = str(
                        float(args.log_replay_smooth_yaw_min_cutoff)
                    )
                if args.log_replay_smooth_yaw_beta is not None:
                    env["CUSTOM_LOG_REPLAY_SMOOTH_YAW_BETA"] = str(float(args.log_replay_smooth_yaw_beta))
                if args.log_replay_smooth_z_min_cutoff is not None:
                    env["CUSTOM_LOG_REPLAY_SMOOTH_Z_MIN_CUTOFF"] = str(
                        float(args.log_replay_smooth_z_min_cutoff)
                    )
                if args.log_replay_smooth_z_beta is not None:
                    env["CUSTOM_LOG_REPLAY_SMOOTH_Z_BETA"] = str(float(args.log_replay_smooth_z_beta))
                if args.log_replay_smooth_d_cutoff is not None:
                    env["CUSTOM_LOG_REPLAY_SMOOTH_D_CUTOFF"] = str(float(args.log_replay_smooth_d_cutoff))
                if args.log_replay_lateral_damping is not None:
                    env["CUSTOM_LOG_REPLAY_LATERAL_DAMPING"] = str(
                        min(1.0, max(0.0, float(args.log_replay_lateral_damping)))
                    )
                if args.log_replay_lateral_passes is not None:
                    env["CUSTOM_LOG_REPLAY_LATERAL_PASSES"] = str(
                        max(1, int(args.log_replay_lateral_passes))
                    )
                if args.log_replay_lateral_max_correction is not None:
                    env["CUSTOM_LOG_REPLAY_LATERAL_MAX_CORRECTION"] = str(
                        max(0.0, float(args.log_replay_lateral_max_correction))
                    )
                if args.log_replay_lateral_turn_keep is not None:
                    env["CUSTOM_LOG_REPLAY_LATERAL_TURN_KEEP"] = str(
                        min(1.0, max(0.0, float(args.log_replay_lateral_turn_keep)))
                    )
                if args.log_replay_lateral_turn_angle_deg is not None:
                    env["CUSTOM_LOG_REPLAY_LATERAL_TURN_ANGLE_DEG"] = str(
                        max(0.0, float(args.log_replay_lateral_turn_angle_deg))
                    )
                if args.log_replay_time_lead is not None:
                    env["CUSTOM_LOG_REPLAY_TIME_LEAD"] = str(
                        min(0.25, max(0.0, float(args.log_replay_time_lead)))
                    )
                if args.stabilize_static_log_replay_vehicles is not None:
                    env["CUSTOM_LOG_REPLAY_STABILIZE_STATIC_VEHICLES"] = (
                        "1" if args.stabilize_static_log_replay_vehicles else "0"
                    )
                if args.static_log_replay_total_disp is not None:
                    env["CUSTOM_LOG_REPLAY_STATIC_TOTAL_DISP"] = str(
                        max(0.0, float(args.static_log_replay_total_disp))
                    )
                if args.static_log_replay_total_path is not None:
                    env["CUSTOM_LOG_REPLAY_STATIC_TOTAL_PATH"] = str(
                        max(0.0, float(args.static_log_replay_total_path))
                    )
                if args.static_log_replay_window_min_duration is not None:
                    env["CUSTOM_LOG_REPLAY_STATIC_WINDOW_MIN_DURATION"] = str(
                        max(0.0, float(args.static_log_replay_window_min_duration))
                    )
                if args.static_log_replay_window_max_disp is not None:
                    env["CUSTOM_LOG_REPLAY_STATIC_WINDOW_MAX_DISP"] = str(
                        max(0.0, float(args.static_log_replay_window_max_disp))
                    )
                if args.static_log_replay_window_max_speed is not None:
                    env["CUSTOM_LOG_REPLAY_STATIC_WINDOW_MAX_SPEED"] = str(
                        max(0.0, float(args.static_log_replay_window_max_speed))
                    )
                if args.static_log_replay_window_max_yaw_delta is not None:
                    env["CUSTOM_LOG_REPLAY_STATIC_WINDOW_MAX_YAW_DELTA"] = str(
                        max(0.0, float(args.static_log_replay_window_max_yaw_delta))
                    )
                if args.animate_walkers is not None:
                    env["CUSTOM_LOG_REPLAY_ANIMATE_WALKERS"] = "1" if args.animate_walkers else "0"
                if args.walker_max_speed is not None:
                    env["CUSTOM_LOG_REPLAY_WALKER_MAX_SPEED"] = str(max(0.1, float(args.walker_max_speed)))
                if args.walker_teleport_distance is not None:
                    env["CUSTOM_LOG_REPLAY_WALKER_TELEPORT_DIST"] = str(
                        max(0.1, float(args.walker_teleport_distance))
                    )
                if args.walker_ground_lift is not None:
                    env["CUSTOM_WALKER_GROUND_LIFT"] = str(float(args.walker_ground_lift))
                if args.normalize_ego_z:
                    env["CUSTOM_EGO_NORMALIZE_Z"] = "1"
                else:
                    env.pop("CUSTOM_EGO_NORMALIZE_Z", None)

                scenario_parameter_rel = spec.scenario_parameter or args.scenario_parameter
                scenario_parameter_path = (repo_root / scenario_parameter_rel).resolve()

                cmd: List[str] = [
                    sys.executable,
                    str(evaluator),
                    "--routes_dir",
                    str(variant_routes_dir),
                    "--ego-num",
                    str(ego_num),
                    "--scenario_parameter",
                    str(scenario_parameter_path),
                    "--agent",
                    str(agent_path),
                    "--agent-config",
                    str(agent_config_path),
                    "--port",
                    str(args.port),
                    "--trafficManagerPort",
                    str(tm_port),
                    "--scenarios",
                    str(scenarios_json),
                    "--repetitions",
                    str(args.repetitions),
                    "--track",
                    args.track,
                    "--checkpoint",
                    str(checkpoint_endpoint),
                    "--debug",
                    "1" if args.debug else "0",
                    "--record",
                    args.record,
                    "--resume",
                    "1" if args.resume else "0",
                    "--carlaProviderSeed",
                    str(args.carla_seed),
                    "--trafficManagerSeed",
                    str(args.traffic_seed),
                    "--skip_existed",
                    "1" if args.skip_existed else "0",
                    "--timeout",
                    str(args.timeout),
                ]

                variant_name = spec.suffix or "baseline"
                if negotiation.disable_comm:
                    variant_name = f"{variant_name} (no negotiation)"
                print(f"\n=== Running variant: {variant_name} ===")
                print("Scenario parameter:", scenario_parameter_path)
                if spec.weather:
                    print("Weather override applied via:", variant_routes_dir)
                print("Results will be stored under:", result_root)
                print("Command:")
                print("  " + " ".join(cmd))

                if args.dry_run:
                    continue

                subprocess.run(cmd, check=True, env=env)
            finally:
                if temp_dir is not None:
                    temp_dir.cleanup()
    finally:
        if carla_manager:
            carla_manager.stop()
        _active_carla_manager = None


if __name__ == "__main__":
    main()
