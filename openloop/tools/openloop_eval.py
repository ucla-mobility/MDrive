#!/usr/bin/env python3
"""
openloop_eval.py
================
Open-loop evaluation of leaderboard planners on v2xpnp recorded scenarios.

Usage
-----
Run from the CoLMDriver root::

    python openloop/tools/openloop_eval.py \\
        --planner   codriving \\
        --data-dir  v2xpnp/dataset \\
        --map-pkl   v2xpnp/map/v2v_corridors_vector_map.pkl \\
        --output    results/openloop

Alternatively, evaluate multiple planners at once::

    python openloop/tools/openloop_eval.py \\
        --planner   '[colmdriver,colmdrivermarco2]' \\
        --planner   '[colmdriver_rulebase,colmdrivermarco2]' \\
        --planner   '[tcp,colmdrivermarco2]' \\
        --planner   '[vad,b2d_zoo]' \\
        --gpus      0,1,2,3 \\
        --start-colmdriver-vllm \\
        --data-dir  v2xpnp/dataset \\
        --map-pkl   v2xpnp/map/v2v_corridors_vector_map.pkl \\
        --output    results/openloop

Or run a direct leaderboard agent without any planner preset::

    python openloop/tools/openloop_eval.py \\
        --agent-path   simulation/leaderboard/team_code/lmdriver_agent.py \\
        --agent-config simulation/leaderboard/team_code/agent_config/lmdriver_config_8_10.py \\
        --data-dir     v2xpnp/dataset \\
        --map-pkl      v2xpnp/map/v2v_corridors_vector_map.pkl \\
        --output       results/openloop_lmdrive

Design principles
-----------------
* **No planner files are modified.**  Agents are loaded exactly as
  ``run_custom_eval.py`` does; the adapter layer monkeypatches CARLA
  dependencies before any planner import.
* **Scientifically robust metrics** – Plan ADE, FDE, and Collision Rate are
  computed using the same formula as ``opencood/utils/eval_utils.py``
  ``calculate_end2end_pred_metrics``.
* **Correct sensor simulation** – camera images, LiDAR point clouds,
  intrinsics, and extrinsics are read directly from the recorded dataset.
  Camera assignment (front/left/right/rear) is computed per-frame from
  ego-relative camera geometry, with a yaw-based fallback.
* **Planner-agnostic execution** – the core evaluator drives the public
  leaderboard/CARLA agent contract: ``setup → set_global_plan → run_step``.
  Legacy ``--planner`` entries are only launcher aliases to agent paths/configs.

Metric definitions
------------------
Plan ADE
    Mean L2 distance (metres) between predicted ego future trajectory and
    the ground-truth future ego positions from the dataset.

Plan FDE
    L2 distance at the final predicted horizon step.

Collision Rate
    Fraction of frames where at least one predicted point is within
    6 m (lateral distance < 3.5 m) of any ground-truth surrounding actor
    position — matching the formula in ``eval_utils.calculate_end2end_pred_metrics``.
"""
from __future__ import annotations

# ============================================================
# STEP 0: Install CARLA stubs BEFORE any planner imports
# ============================================================
import atexit
import os
import sys
import argparse
import datetime as dt
import importlib.util
import json
import logging
import math
import re
import signal
import socket
import subprocess
import threading
import time
import traceback
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

np = None
matplotlib = None
plt = None
mpatches = None
cv2 = None
_MPL_AVAILABLE = False
_CV2_AVAILABLE = False

# Resolve roots from this file's location
_THIS_FILE = Path(__file__).resolve()
_OPENLOOP_ROOT = _THIS_FILE.parents[1]        # openloop/
_PROJECT_ROOT = _OPENLOOP_ROOT.parent          # CoLMDriver/

# Add paths that planners need
# NOTE:
# Keep local project paths at the FRONT of sys.path so namespace packages
# (e.g. Bench2DriveZoo) do not resolve to stale copies from other checkouts.
_LOCAL_IMPORT_PATHS = [
    str(_PROJECT_ROOT),
    str(_PROJECT_ROOT / "simulation" / "leaderboard"),
    str(_PROJECT_ROOT / "simulation" / "leaderboard" / "leaderboard"),
    str(_PROJECT_ROOT / "simulation" / "leaderboard" / "scenario_runner"),
    str(_PROJECT_ROOT / "simulation" / "leaderboard" / "team_code"),
    str(_OPENLOOP_ROOT),
]
for _p in reversed(_LOCAL_IMPORT_PATHS):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

install_openloop_stubs = None
configure_openloop_world = None
advance_openloop_runtime = None
MockVehicle = None
_MockCarlaDataProvider = None
_ds_mod = None
V2XPnPScenario = Any
V2XPnPOpenLoopDataset = Any
GPS_SCALE = (1.0, 1.0)
MAX_GT_STEPS = 0
PREDICTION_DOWNSAMPLING_RATE = 5  # set from _ds_mod at runtime
_OPENLOOP_RUNTIME_READY = False

# ============================================================
# Planner preset registry (legacy launcher convenience only).
# ============================================================
from dataclasses import dataclass

@dataclass(frozen=True)
class PlannerSpec:
    agent:        str   # path relative to CoLMDriver/
    agent_config: str   # path relative to CoLMDriver/
    entry_point: Optional[str] = None


@dataclass
class PlannerRequest:
    name: str
    agent: str
    agent_config: str
    conda_env: Optional[str] = None
    entry_point: Optional[str] = None
    preset_name: Optional[str] = None
    run_id: Optional[str] = None


PLANNER_SPECS: Dict[str, PlannerSpec] = {
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
    "perception_swap_fcooper": PlannerSpec(
        agent="simulation/leaderboard/team_code/perception_swap_agent.py",
        agent_config="simulation/leaderboard/team_code/agent_config/perception_swap_fcooper.yaml",
    ),
    "perception_swap_attfuse": PlannerSpec(
        agent="simulation/leaderboard/team_code/perception_swap_agent.py",
        agent_config="simulation/leaderboard/team_code/agent_config/perception_swap_attfuse.yaml",
    ),
    "perception_swap_disco": PlannerSpec(
        agent="simulation/leaderboard/team_code/perception_swap_agent.py",
        agent_config="simulation/leaderboard/team_code/agent_config/perception_swap_disco.yaml",
    ),
    "perception_swap_cobevt": PlannerSpec(
        agent="simulation/leaderboard/team_code/perception_swap_agent.py",
        agent_config="simulation/leaderboard/team_code/agent_config/perception_swap_cobevt.yaml",
    ),
    "perception_swap_attfuse_behavior": PlannerSpec(
        agent="simulation/leaderboard/team_code/perception_swap_agent.py",
        agent_config="simulation/leaderboard/team_code/agent_config/perception_swap_attfuse_behavior.yaml",
    ),
    "perception_swap_fcooper_behavior": PlannerSpec(
        agent="simulation/leaderboard/team_code/perception_swap_agent.py",
        agent_config="simulation/leaderboard/team_code/agent_config/perception_swap_fcooper_behavior.yaml",
    ),
    "perception_swap_disco_behavior": PlannerSpec(
        agent="simulation/leaderboard/team_code/perception_swap_agent.py",
        agent_config="simulation/leaderboard/team_code/agent_config/perception_swap_disco_behavior.yaml",
    ),
    "perception_swap_cobevt_behavior": PlannerSpec(
        agent="simulation/leaderboard/team_code/perception_swap_agent.py",
        agent_config="simulation/leaderboard/team_code/agent_config/perception_swap_cobevt_behavior.yaml",
    ),
}

COLMDRIVER_VLLM_ENV_DEFAULT = "vllm"
COLMDRIVER_VLM_MODEL_DEFAULT = "ckpt/colmdriver/VLM"
COLMDRIVER_LLM_MODEL_DEFAULT = "ckpt/colmdriver/LLM"
COLMDRIVER_VLM_MAX_MODEL_LEN_DEFAULT = 8192
COLMDRIVER_LLM_MAX_MODEL_LEN_DEFAULT = 4096
COLMDRIVER_VLLM_STARTUP_TIMEOUT_DEFAULT = 300.0
COLMDRIVER_VLLM_PLANNERS = frozenset({"colmdriver", "colmdriver_rulebase"})
COLMDRIVER_VLLM_MIN_FREE_RATIO_DEFAULT = 0.90
COLMDRIVER_VLLM_MIN_FREE_MARGIN_MIB_DEFAULT = 256
VLLM_PROCESS_MATCH_TOKENS = ("vllm", "api_server", "openai.api_server")
AUTO_WORKERS_PER_GPU_SETTLE_TIMEOUT_DEFAULT = 20.0
AUTO_WORKERS_PER_GPU_MIN_FREE_MIB_DEFAULT = 6144
AUTO_WORKERS_PER_GPU_STABLE_SAMPLES_DEFAULT = 2
AUTO_WORKERS_PER_GPU_POLL_SECONDS_DEFAULT = 2.0
AUTO_WORKERS_PER_GPU_STABLE_TOLERANCE_MIB_DEFAULT = 512

_active_runtime_lock = threading.Lock()
_active_planner_children: set["PlannerChildProcess"] = set()
_active_local_vllm_managers: set["ColmDriverVLLMManager"] = set()
_signal_handlers_installed = False


@dataclass(frozen=True)
class ColmDriverVLLMReservation:
    scheduler_gpus: Tuple[str, ...]
    vlm_gpu: str
    llm_gpu: str
    selection_source: str


@dataclass(frozen=True)
class GPUMemoryStats:
    index: str
    total_mib: int
    used_mib: int
    free_mib: int


@dataclass
class GPUPlannerState:
    gpu: Optional[str]
    active_count: int = 0
    launch_capacity: int = 1
    telemetry_available: bool = False
    last_stats: Optional[GPUMemoryStats] = None
    last_probe_source: str = "static"


def _register_runtime_child(child: "PlannerChildProcess") -> None:
    with _active_runtime_lock:
        _active_planner_children.add(child)


def _unregister_runtime_child(child: "PlannerChildProcess") -> None:
    with _active_runtime_lock:
        _active_planner_children.discard(child)


def _register_local_vllm_manager(manager: "ColmDriverVLLMManager") -> None:
    with _active_runtime_lock:
        _active_local_vllm_managers.add(manager)


def _unregister_local_vllm_manager(manager: "ColmDriverVLLMManager") -> None:
    with _active_runtime_lock:
        _active_local_vllm_managers.discard(manager)


def _cleanup_runtime_resources() -> None:
    with _active_runtime_lock:
        children = list(_active_planner_children)
        managers = list(_active_local_vllm_managers)
    for child in children:
        try:
            child.terminate(reason="[INFO] Stopping planner child during launcher shutdown.")
        except Exception:
            pass
    for manager in managers:
        try:
            manager.stop()
        except Exception:
            pass
    try:
        _MockCarlaDataProvider.cleanup()
    except Exception:
        pass


atexit.register(_cleanup_runtime_resources)


def _install_signal_handlers() -> None:
    global _signal_handlers_installed
    if _signal_handlers_installed:
        return

    def _handler(signum: int, frame: object | None) -> None:
        _cleanup_runtime_resources()
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        sys.exit(0)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    _signal_handlers_installed = True


def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((str(host), int(port)), timeout=max(0.1, float(timeout))):
            return True
    except OSError:
        return False


def wait_for_process_port(
    proc: subprocess.Popen[Any],
    host: str,
    port: int,
    timeout: float,
) -> bool:
    deadline = time.monotonic() + max(0.1, float(timeout))
    while time.monotonic() < deadline:
        if is_port_open(host, port, timeout=1.0):
            return True
        if proc.poll() is not None:
            return False
        time.sleep(0.5)
    return False


def _is_local_host(host: str) -> bool:
    normalized = str(host).strip().lower()
    return normalized in {"127.0.0.1", "localhost", "0.0.0.0", "::1", "::"}


def read_process_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
        if raw:
            return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    except Exception:
        pass
    try:
        proc = subprocess.run(
            ["ps", "-p", str(int(pid)), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip()
    except Exception:
        return ""


def find_listening_pids(port: int) -> List[int]:
    try:
        proc = subprocess.run(
            [
                "lsof",
                "-nP",
                "-t",
                f"-iTCP:{int(port)}",
                "-sTCP:LISTEN",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []
    pids: List[int] = []
    for line in proc.stdout.splitlines():
        token = line.strip()
        if not token:
            continue
        try:
            pid = int(token)
        except ValueError:
            continue
        if pid not in pids:
            pids.append(pid)
    return pids


def process_matches_tokens(pid: int, match_tokens: Tuple[str, ...]) -> bool:
    if not match_tokens:
        return True
    cmdline = read_process_cmdline(pid).lower()
    return any(token.lower() in cmdline for token in match_tokens)


def terminate_pid_or_group(pid: int, grace_s: float = 15.0) -> None:
    try:
        pgid = os.getpgid(pid)
    except Exception:
        pgid = None

    terminated = False
    if pgid is not None and pgid > 0:
        try:
            os.killpg(pgid, signal.SIGTERM)
            terminated = True
        except Exception:
            terminated = False
    if not terminated:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception:
            return

    deadline = time.monotonic() + max(0.0, grace_s)
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except Exception:
            break
        time.sleep(0.2)

    if pgid is not None and pgid > 0:
        try:
            os.killpg(pgid, signal.SIGKILL)
            return
        except Exception:
            pass
    try:
        os.kill(pid, signal.SIGKILL)
    except Exception:
        return


def cleanup_listening_processes(
    *,
    host: str,
    ports: Tuple[int, ...],
    service_name: str,
    match_tokens: Tuple[str, ...],
    grace_s: float = 15.0,
) -> Tuple[bool, List[str]]:
    if not _is_local_host(host):
        return False, []

    all_pids: List[int] = []
    unmatched_details: List[str] = []
    for port in sorted(set(int(port) for port in ports)):
        for pid in find_listening_pids(port):
            if pid in all_pids:
                continue
            if process_matches_tokens(pid, match_tokens):
                all_pids.append(pid)
            else:
                cmdline = read_process_cmdline(pid) or "<unknown>"
                unmatched_details.append(f"port={port} pid={pid} cmd={cmdline}")

    for pid in all_pids:
        terminate_pid_or_group(pid, grace_s=grace_s)

    deadline = time.monotonic() + max(1.0, grace_s)
    target_ports = tuple(sorted(set(int(port) for port in ports)))
    while time.monotonic() < deadline:
        if all(not is_port_open(host, port, timeout=0.2) for port in target_ports):
            return True, unmatched_details
        time.sleep(0.2)
    closed = all(not is_port_open(host, port, timeout=0.2) for port in target_ports)
    if not closed:
        print(
            f"[WARN] {service_name} port cleanup timed out for ports {target_ports}. "
            "Some processes may still be alive."
        )
    return closed, unmatched_details


def terminate_process_tree(proc: subprocess.Popen[Any], grace_s: float = 15.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            return
    deadline = time.monotonic() + max(0.0, grace_s)
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.2)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            return


def find_available_port(
    host: str,
    preferred_port: int,
    *,
    reserved_ports: Optional[set[int]] = None,
    tries: int = 200,
    step: int = 1,
) -> int:
    reserved = reserved_ports or set()
    for idx in range(max(1, int(tries))):
        port = int(preferred_port) + idx * max(1, int(step))
        if port in reserved or port > 65535:
            continue
        if not is_port_open(host, port, timeout=0.2):
            return port
    raise RuntimeError(f"No free port found starting at {preferred_port} on host {host}.")


def _parse_gpu_tokens(raw_values: Optional[List[str]]) -> List[str]:
    gpus: List[str] = []
    for raw in raw_values or []:
        for token in str(raw).split(","):
            gpu = token.strip()
            if gpu:
                gpus.append(gpu)
    return gpus


def _detect_visible_gpu_tokens() -> List[str]:
    env_tokens = _parse_gpu_tokens([os.environ.get("CUDA_VISIBLE_DEVICES", "")])
    if env_tokens:
        return env_tokens
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except Exception:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def sanitize_run_id(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    sanitized = sanitized.strip("._-")
    return sanitized or "run"


def _finalize_request_run_ids(requests: List[PlannerRequest]) -> List[PlannerRequest]:
    if not requests:
        return []

    name_counts: Dict[str, int] = {}
    for req in requests:
        name_counts[req.name] = name_counts.get(req.name, 0) + 1

    run_id_counts: Dict[str, int] = {}
    finalized: List[PlannerRequest] = []
    for req in requests:
        base_id = req.name
        if req.conda_env and name_counts.get(req.name, 0) > 1:
            base_id = f"{req.name}_{req.conda_env}"
        base_id = sanitize_run_id(base_id)
        run_id_counts[base_id] = run_id_counts.get(base_id, 0) + 1
        run_id = base_id if run_id_counts[base_id] == 1 else f"{base_id}_{run_id_counts[base_id]}"
        finalized.append(
            PlannerRequest(
                name=req.name,
                agent=req.agent,
                agent_config=req.agent_config,
                conda_env=req.conda_env,
                entry_point=req.entry_point,
                preset_name=req.preset_name,
                run_id=run_id,
            )
        )
    return finalized


def parse_planner_requests(raw_values: Optional[List[str]]) -> List[PlannerRequest]:
    requests: List[PlannerRequest] = []
    for raw in raw_values or []:
        token = str(raw).strip()
        if not token:
            continue
        if token.startswith("[") and token.endswith("]"):
            parts = [part.strip() for part in token[1:-1].split(",")]
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(
                    "Planner list entries must be formatted as [planner,conda_env]."
                )
            planner_name, conda_env = parts
        else:
            planner_name, conda_env = token, None
        if planner_name not in PLANNER_SPECS:
            raise ValueError(
                f"Unknown planner '{planner_name}'. "
                f"Available planners: {', '.join(sorted(PLANNER_SPECS))}"
            )
        preset = PLANNER_SPECS[planner_name]
        requests.append(
            PlannerRequest(
                name=planner_name,
                agent=preset.agent,
                agent_config=preset.agent_config,
                conda_env=conda_env,
                entry_point=preset.entry_point,
                preset_name=planner_name,
            )
        )
    return _finalize_request_run_ids(requests)


def parse_agent_requests(raw_values: Optional[List[str]]) -> List[PlannerRequest]:
    requests: List[PlannerRequest] = []
    for raw in raw_values or []:
        token = str(raw).strip()
        if not token:
            continue
        if not (token.startswith("[") and token.endswith("]")):
            raise ValueError(
                "Generic agent entries must be formatted as "
                "[label,agent_path,agent_config], "
                "[label,agent_path,agent_config,conda_env], or "
                "[label,agent_path,agent_config,entry_point,conda_env]."
            )
        parts = [part.strip() for part in token[1:-1].split(",")]
        if len(parts) not in {3, 4, 5} or any(not part for part in parts[:3]):
            raise ValueError(
                "Generic agent entries must be formatted as "
                "[label,agent_path,agent_config], "
                "[label,agent_path,agent_config,conda_env], or "
                "[label,agent_path,agent_config,entry_point,conda_env]."
            )
        label, agent_path, agent_config = parts[:3]
        entry_point = None
        conda_env = None
        if len(parts) == 4:
            conda_env = parts[3]
        elif len(parts) == 5:
            entry_point = parts[3]
            conda_env = parts[4]
        requests.append(
            PlannerRequest(
                name=label,
                agent=agent_path,
                agent_config=agent_config,
                conda_env=conda_env,
                entry_point=entry_point,
                preset_name=None,
            )
        )
    return _finalize_request_run_ids(requests)


def _strip_path_entry(path_value: str, entry: str) -> str:
    target = os.path.abspath(entry)
    kept = [item for item in path_value.split(os.pathsep) if os.path.abspath(item) != target]
    return os.pathsep.join(item for item in kept if item)


def _build_conda_launch_env(base_env: Dict[str, str]) -> Dict[str, str]:
    env = dict(base_env)
    virtual_env = env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONNOUSERSITE", None)
    env.pop("PYTHONUSERBASE", None)
    env.pop("PYTHONEXECUTABLE", None)
    env.pop("PYTHONSTARTUP", None)
    env.pop("__PYVENV_LAUNCHER__", None)
    env.pop("CONDA_DEFAULT_ENV", None)
    env.pop("CONDA_PREFIX", None)
    env.pop("CONDA_PROMPT_MODIFIER", None)
    env.pop("CONDA_SHLVL", None)
    for key in list(env.keys()):
        if key.startswith("CONDA_PREFIX_"):
            env.pop(key, None)
    if virtual_env:
        venv_bin = Path(virtual_env) / ("Scripts" if os.name == "nt" else "bin")
        env["PATH"] = _strip_path_entry(env.get("PATH", ""), str(venv_bin))
    return env


def query_gpu_memory_stats(requested_gpus: List[str]) -> Dict[str, GPUMemoryStats]:
    gpu_tokens = [str(gpu).strip() for gpu in requested_gpus if str(gpu).strip()]
    if not gpu_tokens:
        return {}
    cmd = [
        "nvidia-smi",
        f"--id={','.join(gpu_tokens)}",
        "--query-gpu=index,uuid,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except Exception:
        return {}

    rows: List[Tuple[str, str, GPUMemoryStats]] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        index, uuid, total_mib, used_mib, free_mib = parts[:5]
        try:
            rows.append(
                (
                    index,
                    uuid,
                    GPUMemoryStats(
                        index=index,
                        total_mib=int(total_mib),
                        used_mib=int(used_mib),
                        free_mib=int(free_mib),
                    ),
                )
            )
        except ValueError:
            continue

    stats_by_requested: Dict[str, GPUMemoryStats] = {}
    unmatched_rows = list(rows)
    for requested_gpu in gpu_tokens:
        match_index = next(
            (
                row_index
                for row_index, (gpu_index, gpu_uuid, _stats) in enumerate(unmatched_rows)
                if requested_gpu == gpu_index or requested_gpu == gpu_uuid
            ),
            None,
        )
        if match_index is None:
            continue
        _gpu_index, _gpu_uuid, stats = unmatched_rows.pop(match_index)
        stats_by_requested[requested_gpu] = stats

    if not stats_by_requested and len(rows) == len(gpu_tokens):
        for requested_gpu, (_gpu_index, _gpu_uuid, stats) in zip(gpu_tokens, rows):
            stats_by_requested[requested_gpu] = stats

    return stats_by_requested


def wait_for_gpu_memory_settle(
    gpu: str,
    *,
    settle_timeout_s: float,
    poll_seconds: float,
    stable_samples: int,
    stable_tolerance_mib: int,
) -> Optional[GPUMemoryStats]:
    stable_samples = max(1, int(stable_samples))
    settle_timeout_s = max(0.0, float(settle_timeout_s))
    poll_seconds = max(0.1, float(poll_seconds))
    stable_tolerance_mib = max(0, int(stable_tolerance_mib))

    deadline = time.monotonic() + settle_timeout_s
    stable_count = 0
    last_free_mib: Optional[int] = None
    latest_stats: Optional[GPUMemoryStats] = None

    while True:
        stats = query_gpu_memory_stats([gpu]).get(gpu)
        if stats is None:
            return latest_stats
        latest_stats = stats

        if last_free_mib is None:
            stable_count = 1
        elif abs(stats.free_mib - last_free_mib) <= stable_tolerance_mib:
            stable_count += 1
        else:
            stable_count = 1
        last_free_mib = stats.free_mib

        if stable_count >= stable_samples:
            return latest_stats
        if time.monotonic() >= deadline:
            return latest_stats
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))


def refresh_gpu_launch_capacity(
    gpu_state: GPUPlannerState,
    *,
    min_free_mib: int,
    settle_timeout_s: float,
    poll_seconds: float,
    stable_samples: int,
    stable_tolerance_mib: int,
) -> GPUPlannerState:
    if gpu_state.gpu is None:
        gpu_state.telemetry_available = False
        gpu_state.last_stats = None
        gpu_state.last_probe_source = "no_gpu"
        gpu_state.launch_capacity = 1
        return gpu_state

    stats = wait_for_gpu_memory_settle(
        gpu_state.gpu,
        settle_timeout_s=settle_timeout_s,
        poll_seconds=poll_seconds,
        stable_samples=stable_samples,
        stable_tolerance_mib=stable_tolerance_mib,
    )
    if stats is None:
        gpu_state.telemetry_available = False
        gpu_state.last_stats = None
        gpu_state.last_probe_source = "nvidia_smi_unavailable"
        gpu_state.launch_capacity = max(1, gpu_state.active_count)
        return gpu_state

    gpu_state.telemetry_available = True
    gpu_state.last_stats = stats
    gpu_state.last_probe_source = "live"
    gpu_state.launch_capacity = gpu_state.active_count + (
        1 if stats.free_mib >= max(0, int(min_free_mib)) else 0
    )
    return gpu_state


def choose_force_launch_gpu(
    gpu_states: Dict[Optional[str], GPUPlannerState],
) -> GPUPlannerState:
    return max(
        gpu_states.values(),
        key=lambda state: (
            state.last_stats.free_mib if state.last_stats is not None else -1,
            -(state.active_count),
        ),
    )


def _colmdriver_vllm_required_free_mib(stats: GPUMemoryStats) -> int:
    total_based = int(math.ceil(
        float(stats.total_mib) * float(COLMDRIVER_VLLM_MIN_FREE_RATIO_DEFAULT)
    ))
    return max(
        int(AUTO_WORKERS_PER_GPU_MIN_FREE_MIB_DEFAULT),
        total_based + int(COLMDRIVER_VLLM_MIN_FREE_MARGIN_MIB_DEFAULT),
    )


def _gpu_has_colmdriver_vllm_capacity(stats: GPUMemoryStats) -> bool:
    return int(stats.free_mib) >= _colmdriver_vllm_required_free_mib(stats)


def list_colmdriver_vllm_gpu_reservations(
    requested_gpus: List[str],
    *,
    exclude_gpus: Optional[List[str]] = None,
) -> List[ColmDriverVLLMReservation]:
    scheduler_gpus = tuple(str(gpu).strip() for gpu in requested_gpus if str(gpu).strip())
    if len(scheduler_gpus) < 2:
        raise RuntimeError(
            "--start-colmdriver-vllm requires at least 2 distinct GPUs so VLM and LLM "
            "can be placed separately."
        )
    if len(set(scheduler_gpus)) != len(scheduler_gpus):
        raise RuntimeError(
            "--gpus must contain distinct GPU ids/UUIDs."
        )

    excluded = {
        str(gpu).strip()
        for gpu in (exclude_gpus or [])
        if str(gpu).strip()
    }
    candidate_gpus = [gpu for gpu in scheduler_gpus if gpu not in excluded]
    if len(candidate_gpus) < 2:
        return []

    live_stats = query_gpu_memory_stats(list(candidate_gpus))
    if live_stats:
        requested_order = {gpu: idx for idx, gpu in enumerate(scheduler_gpus)}
        eligible = {
            gpu: stats
            for gpu, stats in live_stats.items()
            if _gpu_has_colmdriver_vllm_capacity(stats)
        }
        if len(eligible) >= 2:
            scored_pairs: List[Tuple[Tuple[float, float, float, float], ColmDriverVLLMReservation]] = []
            for vlm_gpu, vlm_stats in eligible.items():
                for llm_gpu, llm_stats in eligible.items():
                    if llm_gpu == vlm_gpu:
                        continue
                    score = (
                        float(min(vlm_stats.free_mib, llm_stats.free_mib)),
                        float(vlm_stats.free_mib + llm_stats.free_mib),
                        float(-requested_order.get(vlm_gpu, 0)),
                        float(-requested_order.get(llm_gpu, 0)),
                    )
                    scored_pairs.append(
                        (
                            score,
                            ColmDriverVLLMReservation(
                                scheduler_gpus=scheduler_gpus,
                                vlm_gpu=vlm_gpu,
                                llm_gpu=llm_gpu,
                                selection_source="live_free_vram_threshold",
                            ),
                        )
                    )
            scored_pairs.sort(key=lambda item: item[0], reverse=True)
            return [reservation for _score, reservation in scored_pairs]
        return []

    fallback: List[ColmDriverVLLMReservation] = []
    for vlm_gpu in candidate_gpus:
        for llm_gpu in candidate_gpus:
            if llm_gpu == vlm_gpu:
                continue
            fallback.append(
                ColmDriverVLLMReservation(
                    scheduler_gpus=scheduler_gpus,
                    vlm_gpu=vlm_gpu,
                    llm_gpu=llm_gpu,
                    selection_source="requested_order_fallback",
                )
            )
    return fallback


def select_colmdriver_vllm_gpus(
    requested_gpus: List[str],
    *,
    exclude_gpus: Optional[List[str]] = None,
) -> ColmDriverVLLMReservation:
    reservations = list_colmdriver_vllm_gpu_reservations(
        requested_gpus,
        exclude_gpus=exclude_gpus,
    )
    if not reservations:
        raise RuntimeError(
            "No CoLMDriver vLLM GPU pair currently has enough free VRAM. "
            "Wait for GPUs to free up or provide a larger --gpus pool."
        )
    return reservations[0]


class BackgroundServiceProcess:
    def __init__(
        self,
        *,
        name: str,
        cmd: List[str],
        env: Dict[str, str],
        cwd: Path,
        host: str,
        port: int,
        startup_timeout_s: float,
        log_path: Path,
        match_tokens: Tuple[str, ...],
    ) -> None:
        self.name = str(name)
        self.cmd = list(cmd)
        self.env = dict(env)
        self.cwd = str(cwd)
        self.host = str(host)
        self.port = int(port)
        self.startup_timeout_s = max(1.0, float(startup_timeout_s))
        self.log_path = Path(log_path)
        self.match_tokens = tuple(str(token) for token in match_tokens)
        self.process: subprocess.Popen[Any] | None = None
        self._log_handle = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self) -> None:
        if self.is_running():
            return
        if is_port_open(self.host, self.port, timeout=1.0):
            cleaned, unmatched = cleanup_listening_processes(
                host=self.host,
                ports=(self.port,),
                service_name=self.name,
                match_tokens=self.match_tokens,
                grace_s=10.0,
            )
            if not cleaned or is_port_open(self.host, self.port, timeout=1.0):
                detail = ""
                if unmatched:
                    detail = " Conflicting listener(s): " + "; ".join(unmatched)
                raise RuntimeError(
                    f"{self.name} port {self.port} is already in use. "
                    "Choose a different port or stop the existing service."
                    + detail
                )

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("a", encoding="utf-8")
        self._log_handle.write(
            f"\n=== {self.name} START {dt.datetime.now().isoformat(timespec='seconds')} "
            f"port={self.port} ===\n"
        )
        self._log_handle.write("COMMAND: " + " ".join(self.cmd) + "\n")
        self._log_handle.flush()

        print(f"[INFO] Starting {self.name} on port {self.port}: {' '.join(self.cmd)}")
        try:
            self.process = subprocess.Popen(
                self.cmd,
                cwd=self.cwd,
                env=self.env,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception:
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None
            raise

        if not wait_for_process_port(
            self.process,
            self.host,
            self.port,
            timeout=self.startup_timeout_s,
        ):
            returncode = self.process.poll()
            self.stop(timeout=5.0)
            detail = (
                f"process exited with code {returncode}"
                if returncode is not None
                else f"port did not open within {self.startup_timeout_s:.0f}s"
            )
            raise RuntimeError(
                f"{self.name} failed to start: {detail}. See {self.log_path}."
            )

    def stop(self, timeout: float = 15.0) -> None:
        if self.process is not None and self.process.poll() is None:
            print(f"[INFO] Stopping {self.name}...")
            terminate_process_tree(self.process, grace_s=timeout)
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
        if self._log_handle is not None:
            self._log_handle.write(
                f"=== {self.name} STOP {dt.datetime.now().isoformat(timespec='seconds')} ===\n"
            )
            self._log_handle.flush()
            self._log_handle.close()
            self._log_handle = None
        cleanup_listening_processes(
            host=self.host,
            ports=(self.port,),
            service_name=self.name,
            match_tokens=self.match_tokens,
            grace_s=max(5.0, timeout),
        )
        self.process = None


class ColmDriverVLLMManager:
    def __init__(
        self,
        *,
        repo_root: Path,
        base_env: Dict[str, str],
        conda_env: str,
        reservation: ColmDriverVLLMReservation,
        llm_port: int,
        vlm_port: int,
        startup_timeout_s: float,
        log_dir: Path,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.base_env = _build_conda_launch_env(base_env)
        self.conda_env = str(conda_env)
        self.reservation = reservation
        self.llm_port = int(llm_port)
        self.vlm_port = int(vlm_port)
        self.startup_timeout_s = max(1.0, float(startup_timeout_s))
        self.log_dir = Path(log_dir)
        self.host = "127.0.0.1"
        self.vlm_service = self._build_service(
            name="CoLMDriver VLM vLLM",
            model_rel=COLMDRIVER_VLM_MODEL_DEFAULT,
            max_model_len=COLMDRIVER_VLM_MAX_MODEL_LEN_DEFAULT,
            gpu=self.reservation.vlm_gpu,
            port=self.vlm_port,
            log_name="colmdriver_vlm_vllm.log",
        )
        self.llm_service = self._build_service(
            name="CoLMDriver LLM vLLM",
            model_rel=COLMDRIVER_LLM_MODEL_DEFAULT,
            max_model_len=COLMDRIVER_LLM_MAX_MODEL_LEN_DEFAULT,
            gpu=self.reservation.llm_gpu,
            port=self.llm_port,
            log_name="colmdriver_llm_vllm.log",
        )

    def _build_service(
        self,
        *,
        name: str,
        model_rel: str,
        max_model_len: int,
        gpu: str,
        port: int,
        log_name: str,
    ) -> BackgroundServiceProcess:
        model_path = (self.repo_root / model_rel).resolve()
        if not model_path.exists():
            raise FileNotFoundError(
                f"Required CoLMDriver model checkpoint not found: {model_path}"
            )
        env = dict(self.base_env)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            self.conda_env,
            "vllm",
            "serve",
            str(model_path),
            "--port",
            str(port),
            "--max-model-len",
            str(int(max_model_len)),
            "--trust-remote-code",
            "--enable-prefix-caching",
        ]
        return BackgroundServiceProcess(
            name=name,
            cmd=cmd,
            env=env,
            cwd=self.repo_root,
            host=self.host,
            port=int(port),
            startup_timeout_s=self.startup_timeout_s,
            log_path=self.log_dir / log_name,
            match_tokens=VLLM_PROCESS_MATCH_TOKENS,
        )

    def start(self) -> None:
        if self.llm_port == self.vlm_port:
            raise ValueError("--llm-port and --vlm-port must be different.")
        try:
            self.vlm_service.start()
            self.llm_service.start()
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        self.llm_service.stop()
        self.vlm_service.stop()

# ============================================================
# Data loading — loaded via importlib to bypass the opencood __init__.py
# chain (which pulls in matplotlib, icecream, pyquaternion, etc. that may
# not be installed in all environments).
# ============================================================
def _bootstrap_dataset_module():
    """Load v2xpnp_openloop + map_types directly without triggering __init__."""
    import importlib.util as _ilu, types as _types, sys as _sys

    # 1. map_types — needed by pickle when loading the vector map
    _mt_path = str(_OPENLOOP_ROOT / "data_utils" / "datasets" / "map" / "map_types.py")
    _canon   = "opencood.data_utils.datasets.map.map_types"
    if _canon not in _sys.modules:
        _spec = _ilu.spec_from_file_location(_canon, _mt_path)
        _mod  = _ilu.module_from_spec(_spec)
        _sys.modules[_canon] = _mod
        _spec.loader.exec_module(_mod)

    # 2. v2xpnp_openloop — loaded as a top-level module
    _ol_path = str(_OPENLOOP_ROOT / "data_utils" / "datasets" / "v2xpnp_openloop.py")
    _ol_name = "v2xpnp_openloop_direct"
    _spec2 = _ilu.spec_from_file_location(_ol_name, _ol_path)
    _mod2  = _ilu.module_from_spec(_spec2)
    _sys.modules[_ol_name] = _mod2
    _spec2.loader.exec_module(_mod2)
    return _mod2


def _ensure_openloop_runtime() -> None:
    global np, matplotlib, plt, mpatches, cv2
    global _MPL_AVAILABLE, _CV2_AVAILABLE
    global install_openloop_stubs, configure_openloop_world, advance_openloop_runtime
    global MockVehicle, _MockCarlaDataProvider
    global _ds_mod, V2XPnPScenario, V2XPnPOpenLoopDataset, GPS_SCALE, MAX_GT_STEPS, PREDICTION_DOWNSAMPLING_RATE
    global _OPENLOOP_RUNTIME_READY

    if _OPENLOOP_RUNTIME_READY:
        return

    try:
        import numpy as _np
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "openloop_eval runtime dependencies are unavailable in this process. "
            "Use [planner,conda_env] entries so each child runs inside a planner environment, "
            "or install numpy in the current environment."
        ) from exc
    np = _np

    try:
        import matplotlib as _matplotlib
        _matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        import matplotlib.patches as _mpatches
        matplotlib = _matplotlib
        plt = _plt
        mpatches = _mpatches
        _MPL_AVAILABLE = True
    except (ImportError, AttributeError):
        matplotlib = None
        plt = None
        mpatches = None
        _MPL_AVAILABLE = False

    try:
        import cv2 as _cv2
        cv2 = _cv2
        _CV2_AVAILABLE = True
    except ImportError:
        cv2 = None
        _CV2_AVAILABLE = False

    adapter_path = _OPENLOOP_ROOT / "tools" / "openloop_adapter.py"
    adapter_spec = importlib.util.spec_from_file_location(
        "openloop_adapter_runtime", str(adapter_path)
    )
    if adapter_spec is None or adapter_spec.loader is None:
        raise RuntimeError(f"Failed to load openloop adapter from {adapter_path}")
    openloop_adapter = importlib.util.module_from_spec(adapter_spec)
    adapter_spec.loader.exec_module(openloop_adapter)

    install_openloop_stubs = openloop_adapter.install_openloop_stubs
    configure_openloop_world = getattr(openloop_adapter, "configure_openloop_world", None)
    advance_openloop_runtime = getattr(openloop_adapter, "advance_openloop_runtime", None)
    MockVehicle = openloop_adapter.MockVehicle
    _MockCarlaDataProvider = openloop_adapter._MockCarlaDataProvider

    _ds_mod = _bootstrap_dataset_module()
    V2XPnPScenario = _ds_mod.V2XPnPScenario
    V2XPnPOpenLoopDataset = _ds_mod.V2XPnPOpenLoopDataset
    GPS_SCALE = _ds_mod.GPS_SCALE
    MAX_GT_STEPS = _ds_mod.MAX_GT_STEPS
    PREDICTION_DOWNSAMPLING_RATE = _ds_mod.PREDICTION_DOWNSAMPLING_RATE

    _OPENLOOP_RUNTIME_READY = True


# ============================================================
# Agent loading
# ============================================================

def _resolve_agent_module_path(agent_path: str) -> Path:
    path = Path(agent_path)
    if path.is_absolute():
        return path
    return (_PROJECT_ROOT / path).resolve()


def _load_agent_class(agent_path: str, entry_class_name: Optional[str] = None):
    """
    Dynamically load the agent module and return its entry-point class.
    Mirrors how the leaderboard evaluator loads agents.
    """
    module_path = _resolve_agent_module_path(agent_path)
    module_name = module_path.stem + "_openloop"
    mod_spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    module      = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(module)

    if entry_class_name is None:
        get_entry_point = getattr(module, "get_entry_point", None)
        if not callable(get_entry_point):
            raise AttributeError(
                f"Agent module '{module_path}' does not define get_entry_point(); "
                "provide --agent-entry-point."
            )
        entry_class_name = get_entry_point()
    return getattr(module, entry_class_name)


def _build_agent(
    *,
    agent_path: str,
    agent_config: str,
    num_ego_vehicles: int,
    entry_class_name: Optional[str] = None,
):
    """
    Instantiate, setup, and return a fully-initialised agent.
    """
    AgentClass = _load_agent_class(agent_path, entry_class_name=entry_class_name)
    config_path = _resolve_abs_path(agent_config) or str(agent_config)
    # Support both leaderboard agent styles:
    # 1) no-arg ctor + explicit setup(config_path, ego_vehicles_num=...)
    # 2) ctor(path_to_conf_file, ego_vehicles_num) that calls setup internally
    try:
        agent = AgentClass()
        agent.setup(config_path, ego_vehicles_num=num_ego_vehicles)
    except TypeError as exc:
        msg = str(exc)
        if "path_to_conf_file" not in msg:
            raise
        agent = AgentClass(config_path, ego_vehicles_num=num_ego_vehicles)
    return agent


def _build_scenario_global_plan(
    scenario: V2XPnPScenario,
    actor_ids: List[int],
) -> Tuple[List[List[Tuple[Dict[str, float], Any]]], List[List[Tuple[Any, Any]]]]:
    """
    Build the public leaderboard route payload for ``agent.set_global_plan(...)``.
    """
    plans = []
    world_plans = []
    for actor_id in actor_ids:
        actor_plans = scenario.build_global_plan(actor_id)
        plans.extend(actor_plans)

        for route in actor_plans:
            if not route:
                world_plans.append([])
                continue
            xy = []
            cmds = []
            for gps_dict, cmd in route:
                try:
                    x = float(gps_dict.get("lat", 0.0)) * float(GPS_SCALE[0])
                    y = float(gps_dict.get("lon", 0.0)) * float(GPS_SCALE[1])
                except Exception:
                    x, y = 0.0, 0.0
                xy.append((x, y))
                cmds.append(cmd)

            yaws = []
            for i in range(len(xy)):
                if i + 1 < len(xy):
                    dx = xy[i + 1][0] - xy[i][0]
                    dy = xy[i + 1][1] - xy[i][1]
                elif i > 0:
                    dx = xy[i][0] - xy[i - 1][0]
                    dy = xy[i][1] - xy[i - 1][1]
                else:
                    dx, dy = 1.0, 0.0
                yaw_deg = math.degrees(math.atan2(dy, dx)) if (abs(dx) + abs(dy)) > 1e-6 else 0.0
                yaws.append(yaw_deg)

            try:
                import carla
                world_route = [
                    (
                        carla.Transform(
                            carla.Location(x=float(x), y=float(y), z=0.0),
                            carla.Rotation(roll=0.0, pitch=0.0, yaw=float(yaw_deg)),
                        ),
                        cmd,
                    )
                    for (x, y), yaw_deg, cmd in zip(xy, yaws, cmds)
                ]
            except Exception as exc:
                raise RuntimeError("Failed to build CARLA route transforms.") from exc
            world_plans.append(world_route)
    return plans, world_plans


def _set_agent_global_plan(agent, scenario: V2XPnPScenario, actor_ids: List[int]) -> None:
    set_global_plan = getattr(agent, "set_global_plan", None)
    if not callable(set_global_plan):
        raise AttributeError(
            f"Agent '{agent.__class__.__name__}' does not expose set_global_plan(...)."
        )
    gps_plans, world_plans = _build_scenario_global_plan(scenario, actor_ids)
    set_global_plan(gps_plans, world_plans)
    if _MockCarlaDataProvider is not None:
        try:
            route_locations = [
                [(transform.location, cmd) for transform, cmd in route]
                for route in world_plans
            ]
            _MockCarlaDataProvider.set_ego_vehicle_route(route_locations)
        except Exception:
            pass


def _apply_agent_runtime_context(
    agent,
    *,
    scenario: V2XPnPScenario,
    town_id: Optional[str],
    sampled_scenarios: Optional[List[Any]] = None,
) -> None:
    resolved_town = str(town_id) if town_id else None
    if not resolved_town:
        try:
            resolved_town = str(_MockCarlaDataProvider.get_map().name)
        except Exception:
            resolved_town = "openloop_map"
    setattr(agent, "town_id", resolved_town)
    setattr(agent, "sampled_scenarios", sampled_scenarios if sampled_scenarios is not None else [])
    setattr(agent, "scenario_cofing_name", str(scenario.scenario_dir.name))


def _safe_sensor_specs(agent) -> List[Dict[str, Any]]:
    """
    Return planner-declared sensor specs (`agent.sensors()`), or [] if unavailable.
    """
    sensors_fn = getattr(agent, "sensors", None)
    if sensors_fn is None:
        return []
    try:
        specs = sensors_fn()
    except Exception:
        return []
    if not isinstance(specs, list):
        return []
    clean: List[Dict[str, Any]] = []
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        if "id" not in spec:
            continue
        clean.append(spec)
    return clean


def _normalize_deg(x: float) -> float:
    return (float(x) + 180.0) % 360.0 - 180.0


def _ang_dist_deg(a: float, b: float) -> float:
    return abs(_normalize_deg(a - b))


def _direction_from_sensor_id(sensor_id: str) -> Optional[str]:
    sid = str(sensor_id).strip().lower().replace("-", "_")
    sid = re.sub(r"_+", "_", sid)

    if "front_left" in sid or "frontleft" in sid or "left_front" in sid:
        return "left"
    if "front_right" in sid or "frontright" in sid or "right_front" in sid:
        return "right"
    if "back_left" in sid or "rear_left" in sid or "left_back" in sid:
        return "left"
    if "back_right" in sid or "rear_right" in sid or "right_back" in sid:
        return "right"
    if "rear" in sid or "back" in sid:
        return "rear"
    if "left" in sid:
        return "left"
    if "right" in sid:
        return "right"
    if ("front" in sid or sid in {"rgb", "camera", "image", "tel_rgb", "logreplay_rgb"}):
        return "front"
    return None


def _direction_from_sensor_spec(spec: Dict[str, Any]) -> str:
    sid = str(spec.get("id", ""))
    by_id = _direction_from_sensor_id(sid)
    if by_id is not None:
        return by_id

    try:
        yaw = float(spec.get("yaw", 0.0))
    except Exception:
        yaw = 0.0

    candidates = {
        "front": 0.0,
        "left": -90.0,
        "right": 90.0,
        "rear": 180.0,
    }
    return min(candidates.keys(), key=lambda k: _ang_dist_deg(yaw, candidates[k]))


def _resize_bgra(img_bgra: np.ndarray, width: int, height: int) -> np.ndarray:
    if img_bgra is None:
        return np.zeros((height, width, 4), dtype=np.uint8)
    h, w = img_bgra.shape[:2]
    if h == height and w == width:
        return img_bgra
    if _CV2_AVAILABLE:
        interp = cv2.INTER_AREA if (width < w or height < h) else cv2.INTER_LINEAR
        return cv2.resize(img_bgra, (width, height), interpolation=interp)
    # Fallback nearest-neighbor resize without OpenCV.
    ys = (np.linspace(0, max(h - 1, 0), num=height)).astype(np.int64)
    xs = (np.linspace(0, max(w - 1, 0), num=width)).astype(np.int64)
    return img_bgra[ys][:, xs]


def _extract_speed_scalar(speed_payload: Any) -> np.float32:
    if isinstance(speed_payload, dict):
        ms = speed_payload.get("move_state", {})
        if isinstance(ms, dict) and "speed" in ms:
            return np.float32(ms["speed"])
        if "speed" in speed_payload:
            return np.float32(speed_payload["speed"])
    try:
        return np.float32(speed_payload)
    except Exception:
        return np.float32(0.0)


def _build_speed_payload(speed_scalar: np.float32) -> Dict[str, Any]:
    # Expose both schemas used by different baselines.
    return {
        "speed": np.float32(speed_scalar),
        "move_state": {"speed": np.float32(speed_scalar)},
    }


def _bgra_from_rgb(rgb: np.ndarray) -> np.ndarray:
    if rgb is None:
        return np.zeros((600, 800, 4), dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return np.zeros((600, 800, 4), dtype=np.uint8)
    if _CV2_AVAILABLE:
        return cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGRA)
    out = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    out[:, :, 0] = rgb[:, :, 2]
    out[:, :, 1] = rgb[:, :, 1]
    out[:, :, 2] = rgb[:, :, 0]
    out[:, :, 3] = 255
    return out


def _get_live_bev_image(hero_vehicle: MockVehicle) -> Optional[np.ndarray]:
    try:
        producer = _OPENLOOP_ADAPTER._global_producer
        if producer is None:
            return None
        return producer.produce(hero_vehicle)
    except Exception:
        return None


def _bev_rgb(bev_img: Optional[np.ndarray]) -> np.ndarray:
    if bev_img is None:
        return np.zeros((400, 400, 3), dtype=np.uint8)
    arr = np.asarray(bev_img)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[2] >= 4:
        arr = arr[:, :, :3]
    return arr.astype(np.uint8, copy=False)


def _bev_to_bgra(bev_img: Optional[np.ndarray], width: int, height: int) -> np.ndarray:
    rgb = _bev_rgb(bev_img)
    if _CV2_AVAILABLE:
        bgra = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGRA)
    else:
        bgra = _bgra_from_rgb(rgb)
    return _resize_bgra(bgra, width, height)


def _bev_semantic_bgra(bev_img: Optional[np.ndarray], width: int, height: int) -> np.ndarray:
    rgb = _bev_rgb(bev_img)
    if _CV2_AVAILABLE:
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_NEAREST)
    else:
        ys = (np.linspace(0, max(rgb.shape[0] - 1, 0), num=height)).astype(np.int64)
        xs = (np.linspace(0, max(rgb.shape[1] - 1, 0), num=width)).astype(np.int64)
        rgb = rgb[ys][:, xs]
    mask = np.mean(rgb.astype(np.float32), axis=2) > 32.0
    out = np.zeros((height, width, 4), dtype=np.uint8)
    # map_agent consumes channel 2 as CARLA semantic class id (6/7/10 = drivable)
    out[:, :, 2] = np.where(mask, 7, 0).astype(np.uint8)
    out[:, :, 3] = 255
    return out


def _zero_depth_bgra(width: int, height: int) -> np.ndarray:
    out = np.zeros((height, width, 4), dtype=np.uint8)
    out[:, :, 3] = 255
    return out


def _semantic_lidar_stub(lidar_xyz_i: np.ndarray) -> np.ndarray:
    if lidar_xyz_i is None:
        return np.zeros((0, 6), dtype=np.float32)
    lidar = np.asarray(lidar_xyz_i, dtype=np.float32)
    if lidar.ndim != 2:
        return np.zeros((0, 6), dtype=np.float32)
    n = lidar.shape[0]
    out = np.zeros((n, 6), dtype=np.float32)
    cols = min(lidar.shape[1], 4)
    out[:, :cols] = lidar[:, :cols]
    return out


def _build_agent_input_data(agent,
                            sensor_specs: List[Dict[str, Any]],
                            frame_data,
                            actor_id: int,
                            slot_id: int,
                            frame_idx: int,
                            bev_img: Optional[np.ndarray]) -> Dict[str, Tuple[int, Any]]:
    """
    Build CARLA-style `input_data` keyed by planner sensor IDs (plus ego slot).
    This keeps the pipeline planner-agnostic and driven by `agent.sensors()`.
    """
    source_input = frame_data.input_data
    source_raw = frame_data.car_data_raw_entry

    cam_sources: Dict[str, np.ndarray] = {}
    for direction in ("front", "left", "right", "rear"):
        key = f"rgb_{direction}_{actor_id}"
        if key in source_input:
            cam_sources[direction] = np.asarray(source_input[key][1])
            continue
        rgb = source_raw.get(f"rgb_{direction}")
        if rgb is not None:
            cam_sources[direction] = _bgra_from_rgb(np.asarray(rgb))

    if not cam_sources:
        cam_sources["front"] = np.zeros((600, 800, 4), dtype=np.uint8)

    default_cam = cam_sources.get("front", next(iter(cam_sources.values())))

    lidar = np.asarray(source_input.get(f"lidar_{actor_id}", (frame_idx, np.zeros((0, 4), dtype=np.float32)))[1])
    gps = np.asarray(source_input.get(f"gps_{actor_id}", (frame_idx, np.zeros(2, dtype=np.float64)))[1], dtype=np.float64)
    imu = np.asarray(source_input.get(f"imu_{actor_id}", (frame_idx, np.zeros(7, dtype=np.float64)))[1], dtype=np.float64)
    speed_src = source_input.get(f"speed_{actor_id}", (frame_idx, {"speed": np.float32(0.0)}))[1]
    speed_scalar = _extract_speed_scalar(speed_src)
    speed_payload = _build_speed_payload(speed_scalar)

    out: Dict[str, Tuple[int, Any]] = {}

    for spec in sensor_specs:
        sensor_id = str(spec.get("id", "")).strip()
        if not sensor_id:
            continue
        key = sensor_id if sensor_id.endswith(f"_{slot_id}") else f"{sensor_id}_{slot_id}"
        stype = str(spec.get("type", "")).strip().lower()
        sid_lower = sensor_id.lower()

        if "sensor.camera" in stype:
            try:
                width = int(spec.get("width", default_cam.shape[1]))
                height = int(spec.get("height", default_cam.shape[0]))
            except Exception:
                width, height = int(default_cam.shape[1]), int(default_cam.shape[0])
            width = max(1, width)
            height = max(1, height)

            if "semantic_segmentation" in stype:
                cam = _bev_semantic_bgra(bev_img, width, height)
            elif "depth" in stype:
                cam = _zero_depth_bgra(width, height)
            elif "bev" in sid_lower or sid_lower == "map" or "topdown" in sid_lower:
                cam = _bev_to_bgra(bev_img, width, height)
            else:
                direction = _direction_from_sensor_spec(spec)
                src = cam_sources.get(direction, cam_sources.get("front", default_cam))
                cam = _resize_bgra(src, width, height)
            out[key] = (frame_idx, cam)
            continue

        if stype == "sensor.lidar.ray_cast_semantic":
            out[key] = (frame_idx, _semantic_lidar_stub(lidar))
            continue
        if stype.startswith("sensor.lidar"):
            out[key] = (frame_idx, lidar)
            continue
        if stype == "sensor.other.gnss":
            out[key] = (frame_idx, gps)
            continue
        if stype == "sensor.other.imu":
            out[key] = (frame_idx, imu)
            continue
        if stype == "sensor.speedometer":
            out[key] = (frame_idx, _build_speed_payload(speed_scalar))
            continue
        if stype == "sensor.opendrive_map":
            out[key] = (frame_idx, _MockCarlaDataProvider.get_map().to_opendrive())
            continue
        if stype == "sensor.other.radar":
            out[key] = (frame_idx, np.zeros((0, 4), dtype=np.float32))
            continue
        if stype == "sensor.other.collision":
            out[key] = (frame_idx, [])
            continue
        if stype == "sensor.other.lane_invasion":
            out[key] = (frame_idx, [])
            continue
        if stype == "sensor.other.obstacle":
            out[key] = (frame_idx, [])
            continue

        # Generic fallback for uncommon sensors:
        # prefer raw keyed payload if present, otherwise provide None.
        src_key_with_actor = f"{sensor_id}_{actor_id}"
        if src_key_with_actor in source_input:
            out[key] = source_input[src_key_with_actor]
        elif sensor_id in source_input:
            out[key] = source_input[sensor_id]
        else:
            out[key] = (frame_idx, None)

    # Fallback for planners that don't declare sensors() cleanly.
    if not out:
        for base in ("rgb_front", "rgb_left", "rgb_right", "rgb_rear", "lidar", "gps", "imu", "speed"):
            src_key = f"{base}_{actor_id}"
            if src_key not in source_input:
                continue
            if base == "speed":
                out[f"speed_{slot_id}"] = (frame_idx, _build_speed_payload(speed_scalar))
            else:
                out[f"{base}_{slot_id}"] = source_input[src_key]

    return out

def _extract_primary_control(run_step_ret: Any, slot_id: int = 0) -> Optional[Dict[str, float]]:
    """
    Extract a CARLA-style control command from agent.run_step(...) return.

    Returns dict {'steer','throttle','brake'} in [-1,1]/[0,1]/[0,1], or None.
    """
    ctrl_obj = None

    if isinstance(run_step_ret, (list, tuple)):
        if len(run_step_ret) == 0:
            return None
        if slot_id < len(run_step_ret):
            ctrl_obj = run_step_ret[slot_id]
        else:
            ctrl_obj = run_step_ret[0]
    else:
        ctrl_obj = run_step_ret

    if ctrl_obj is None:
        return None

    try:
        if hasattr(ctrl_obj, "steer") and hasattr(ctrl_obj, "throttle") and hasattr(ctrl_obj, "brake"):
            steer = float(getattr(ctrl_obj, "steer"))
            throttle = float(getattr(ctrl_obj, "throttle"))
            brake = float(getattr(ctrl_obj, "brake"))
            return {
                "steer": float(np.clip(steer, -1.0, 1.0)),
                "throttle": float(np.clip(throttle, 0.0, 1.0)),
                "brake": float(np.clip(brake, 0.0, 1.0)),
            }
    except Exception:
        pass

    if isinstance(ctrl_obj, dict):
        try:
            steer = float(ctrl_obj.get("steer", 0.0))
            throttle = float(ctrl_obj.get("throttle", 0.0))
            brake = float(ctrl_obj.get("brake", 0.0))
            return {
                "steer": float(np.clip(steer, -1.0, 1.0)),
                "throttle": float(np.clip(throttle, 0.0, 1.0)),
                "brake": float(np.clip(brake, 0.0, 1.0)),
            }
        except Exception:
            return None

    return None


def _rollout_control_trajectory(
    pose: np.ndarray,
    speed_mps: float,
    control: Dict[str, float],
    horizon_steps: int,
    dt: float,
    wheelbase_m: float = 2.9,
    max_steer_angle_rad: float = 0.6,
    max_accel_mps2: float = 2.5,
    max_brake_mps2: float = 6.0,
    linear_drag: float = 0.12,
) -> np.ndarray:
    """
    Roll out a planner-agnostic kinematic trajectory from CARLA controls.

    This avoids planner-specific waypoint conventions and yields a unified
    world-frame trajectory from (steer, throttle, brake).
    """
    if horizon_steps <= 0:
        return np.zeros((0, 2), dtype=np.float64)

    x = float(pose[0])
    y = float(pose[1])
    yaw = math.radians(float(pose[4]))
    v = max(0.0, float(speed_mps))

    steer_norm = float(np.clip(control.get("steer", 0.0), -1.0, 1.0))
    throttle = float(np.clip(control.get("throttle", 0.0), 0.0, 1.0))
    brake = float(np.clip(control.get("brake", 0.0), 0.0, 1.0))
    steer_rad = steer_norm * max_steer_angle_rad

    out = np.zeros((horizon_steps, 2), dtype=np.float64)
    for t in range(horizon_steps):
        a = throttle * max_accel_mps2 - brake * max_brake_mps2 - linear_drag * v
        v = max(0.0, v + a * dt)
        yaw = _wrap_angle_rad(yaw + (v / max(wheelbase_m, 1e-3)) * math.tan(steer_rad) * dt)
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        out[t, 0] = x
        out[t, 1] = y
    return out


def _wrap_angle_rad(angle: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _angular_error_rad(a: float, b: float) -> float:
    """Absolute wrapped angular error |a-b| in radians."""
    return abs(_wrap_angle_rad(float(a) - float(b)))


def _carla_compass_from_pose(pose: np.ndarray) -> float:
    """
    Derive CARLA-style compass from dataset yaw pose.

    CARLA-centric planners typically assume:
      theta_world = compass + pi/2
    where theta_world is ego heading in world XY.

    Given dataset yaw is world heading, corresponding CARLA compass is
    yaw - pi/2.
    """
    yaw_rad = math.radians(float(pose[4]))
    return _wrap_angle_rad(yaw_rad - math.pi / 2.0)


def _normalize_compass_for_bev(raw_compass: float, pose: np.ndarray) -> float:
    """
    Normalize measured compass to CARLA-style compass used by this evaluator.

    Some exports provide `compass` as raw yaw, while others already provide
    CARLA compass. We choose between:
      c0 = raw_compass
      c1 = raw_compass - pi/2
    by whichever is closer to pose-derived CARLA compass.
    """
    c0 = _wrap_angle_rad(float(raw_compass))
    c1 = _wrap_angle_rad(c0 - math.pi / 2.0)
    target = _carla_compass_from_pose(pose)
    return c0 if _angular_error_rad(c0, target) <= _angular_error_rad(c1, target) else c1


def _fallback_compass_from_pose(pose: np.ndarray) -> float:
    """
    Derive planner compass from dataset yaw pose.

    Returns CARLA-style compass so that downstream conversion:
      theta = compass + pi/2
    recovers dataset world yaw.
    """
    return _carla_compass_from_pose(pose)


def _resolve_compass_rad(pose: np.ndarray,
                         measurements: Optional[Dict[str, Any]] = None) -> float:
    """Prefer measurement compass; fallback to pose-derived compass."""
    if isinstance(measurements, dict):
        try:
            c = float(measurements.get("compass"))
            if np.isfinite(c):
                return _normalize_compass_for_bev(c, pose)
        except Exception:
            pass
    return _fallback_compass_from_pose(pose)


def _bev_to_world(bev_wps: np.ndarray, ego_world_xy: np.ndarray,
                  compass_rad: float) -> np.ndarray:
    """
    Convert ego-local BEV waypoints to world-frame positions.

    The BEV frame rotation in this pipeline follows:
        theta = compass + π/2

    Conversion:
        theta = compass + π/2
        R     = [[cos θ, -sin θ], [sin θ, cos θ]]
        world = ego_world_xy + R @ bev_wp

    Parameters
    ----------
    bev_wps       : np.ndarray  [T, 2]   (forward, lateral) in metres
    ego_world_xy  : np.ndarray  [2]       world (x, y) of ego at this frame
    compass_rad   : float                 ego IMU compass (radians)

    Returns
    -------
    np.ndarray  [T, 2]  world-frame (x, y) for each waypoint
    """
    theta = compass_rad + math.pi / 2.0
    R     = np.array([[math.cos(theta), -math.sin(theta)],
                       [math.sin(theta),  math.cos(theta)]])
    return ego_world_xy + (R @ bev_wps.T).T


# ============================================================
# Metric accumulators
# ============================================================

class MetricAccumulator:
    """Accumulates Plan ADE / FDE / Collision Rate across all frames."""

    def __init__(self):
        self.plan_ade_sum      = 0.0
        self.plan_fde_sum      = 0.0
        self.collision_count   = 0
        self.n_frames          = 0

    def update(self,
               pred_world_wps: np.ndarray,
               gt_future_xy:   np.ndarray,
               gt_future_mask: np.ndarray,
               surround_traj:  np.ndarray,
               surround_mask:  np.ndarray,
               ego_pose:        Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute and accumulate metrics for one frame.

        Parameters
        ----------
        pred_world_wps  : [T_pred, 2]  predicted ego positions in world frame
        gt_future_xy    : [T_gt, 2]    GT future ego world positions
        gt_future_mask  : [T_gt]       1 = valid GT step
        surround_traj   : [N_actors, T_gt, 2]  surrounding actor world positions
        surround_mask   : [N_actors, T_gt]      1 = valid

        Returns
        -------
        dict with per-frame metrics.
        """
        pred_world_wps = np.asarray(pred_world_wps, dtype=np.float64)
        gt_future_xy = np.asarray(gt_future_xy, dtype=np.float64)
        gt_future_mask = np.asarray(gt_future_mask)

        if pred_world_wps.ndim != 2 or pred_world_wps.shape[1] < 2:
            raise ValueError(
                f"pred_world_wps must be [T,2+] but got {pred_world_wps.shape}"
            )
        if gt_future_xy.ndim != 2 or gt_future_xy.shape[1] < 2:
            raise ValueError(
                f"gt_future_xy must be [T,2+] but got {gt_future_xy.shape}"
            )
        if gt_future_mask.ndim != 1:
            raise ValueError(
                f"gt_future_mask must be [T] but got {gt_future_mask.shape}"
            )

        T_pred = len(pred_world_wps)
        T_gt   = len(gt_future_xy)
        T_min  = min(T_pred, T_gt)

        if T_min == 0:
            return {}

        pred   = pred_world_wps[:T_min, :2]
        gt     = gt_future_xy  [:T_min, :2]
        mask   = gt_future_mask[:T_min]

        # ---- Plan ADE (mean L2 over valid steps) ----
        dists   = np.sqrt(np.sum((pred - gt) ** 2, axis=1))
        valid   = mask > 0
        n_valid = valid.sum()
        if n_valid == 0:
            return {}

        plan_ade = float((dists * valid).sum() / n_valid)
        plan_fde = float(dists[valid][-1]) if valid.any() else 0.0

        # ---- Collision rate (RiskMM formula: L2 < 6 m gate, then |ΔY| ≤ 3.5 m) ----
        # Faithfully matches eval_utils.py in the RiskMM codebase:
        #   1. Remove the closest actor if within 2.6 m of ego (handles V2X partner ego).
        #   2. Flag collision if any (t, actor) pair has L2 < 6 m AND |ΔY| ≤ 3.5 m
        #      over all predicted timesteps (no TTC threshold).
        # World-frame Y-difference (axis-aligned) used for lateral, matching RiskMM.
        _EUCLIDEAN_GATE    = 6.0   # metres — pre-filter before lateral check
        _LAT_THRESH        = 3.5   # metres — world-frame |ΔY|
        _CLOSE_ACTOR_THRESH = 2.6  # metres — remove physically co-located actor
        collision = 0
        if surround_traj.shape[0] > 0:
            N_actors = surround_traj.shape[0]
            B  = surround_traj[:, :T_min, :].transpose(1, 0, 2)  # [T, N, 2]
            sm = surround_mask[:, :T_min].T                       # [T, N]

            # Step 1: remove the closest actor if it is virtually co-located
            # (e.g. the V2X partner ego appearing in its own actor list).
            actor_keep = np.ones(N_actors, dtype=bool)
            if ego_pose is not None:
                ego_xy = np.asarray(ego_pose[:2], dtype=np.float64)
                first_pos = surround_traj[:, 0, :]  # [N, 2] — proxy for t=0
                cur_dists = np.linalg.norm(first_pos - ego_xy, axis=1)
                closest_idx = int(np.argmin(cur_dists))
                if cur_dists[closest_idx] < _CLOSE_ACTOR_THRESH:
                    actor_keep[closest_idx] = False

            # Step 2: L2 distances over all timesteps [T, N]
            A = pred[:T_min, :2]           # [T, 2]
            diffs_all = A[:, None, :] - B  # [T, N, 2]
            l2_all    = np.linalg.norm(diffs_all, axis=2)  # [T, N]
            valid_sm  = sm > 0             # [T, N]
            l2_gated  = np.where(valid_sm, l2_all, np.inf)
            l2_gated[:, ~actor_keep] = np.inf

            # Step 3: lateral check only where L2 < gate
            close_pairs = l2_gated < _EUCLIDEAN_GATE
            if np.any(close_pairs):
                t_idx, n_idx = np.where(close_pairs)
                delta_y = np.abs(A[t_idx, 1] - B[t_idx, n_idx, 1])
                if np.any(delta_y <= _LAT_THRESH):
                    collision = 1

        self.plan_ade_sum    += plan_ade
        self.plan_fde_sum    += plan_fde
        self.collision_count += collision
        self.n_frames        += 1

        return {
            "plan_ade":  plan_ade,
            "plan_fde":  plan_fde,
            "collision": collision,
        }

    def summary(self) -> Dict[str, float]:
        if self.n_frames == 0:
            return {"plan_ade": float("nan"), "plan_fde": float("nan"),
                    "collision_rate": float("nan"), "n_frames": 0}
        return {
            "plan_ade":       self.plan_ade_sum    / self.n_frames,
            "plan_fde":       self.plan_fde_sum    / self.n_frames,
            "collision_rate": self.collision_count / self.n_frames,
            "n_frames":       self.n_frames,
        }


def _compute_plan_metrics_core(
    *,
    pred_world_wps: np.ndarray,
    gt_future_xy: np.ndarray,
    gt_future_mask: np.ndarray,
    surround_traj: np.ndarray,
    surround_mask: np.ndarray,
    surround_actor_ids: Optional[List[int]] = None,
    ego_pose: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    pred_world_wps = np.asarray(pred_world_wps, dtype=np.float64)
    gt_future_xy = np.asarray(gt_future_xy, dtype=np.float64)
    gt_future_mask = np.asarray(gt_future_mask)

    if pred_world_wps.ndim != 2 or pred_world_wps.shape[1] < 2:
        raise ValueError(
            f"pred_world_wps must be [T,2+] but got {pred_world_wps.shape}"
        )
    if gt_future_xy.ndim != 2 or gt_future_xy.shape[1] < 2:
        raise ValueError(
            f"gt_future_xy must be [T,2+] but got {gt_future_xy.shape}"
        )
    if gt_future_mask.ndim != 1:
        raise ValueError(
            f"gt_future_mask must be [T] but got {gt_future_mask.shape}"
        )

    T_pred = len(pred_world_wps)
    T_gt = len(gt_future_xy)
    T_min = min(T_pred, T_gt)
    if T_min == 0:
        return None

    pred = pred_world_wps[:T_min, :2]
    gt = gt_future_xy[:T_min, :2]
    mask = gt_future_mask[:T_min] > 0
    dists = np.sqrt(np.sum((pred - gt) ** 2, axis=1))
    n_valid = int(mask.sum())
    if n_valid == 0:
        return None

    plan_ade = float((dists * mask).sum() / n_valid)
    plan_fde = float(dists[mask][-1]) if np.any(mask) else 0.0

    # ---- Collision rate (RiskMM formula: L2 < 6 m gate, then |ΔY| ≤ 3.5 m) ----
    # Faithfully matches eval_utils.py in the RiskMM codebase:
    #   1. Remove the closest actor if within 2.6 m of ego (handles V2X partner ego).
    #   2. Flag collision if any (t, actor) pair has L2 < 6 m AND |ΔY| ≤ 3.5 m
    #      over all predicted timesteps (no TTC threshold).
    # World-frame Y-difference (axis-aligned) used for lateral, matching RiskMM.
    _EUCLIDEAN_GATE     = 6.0   # metres
    _LAT_THRESH         = 3.5   # metres — world-frame |ΔY|
    _CLOSE_ACTOR_THRESH = 2.6   # metres — remove virtually co-located actor
    collision = 0
    collision_debug: Dict[str, Any] = {
        "collision": False,
        "actor_index": None,
        "actor_id": None,
        "step_index": None,
        "t_seconds": None,
        "delta_y_m": None,
        "euclidean_m": None,
        "pred_world_xy": None,
        "actor_world_xy": None,
    }
    if surround_traj.shape[0] > 0:
        N_actors = surround_traj.shape[0]
        B  = surround_traj[:, :T_min, :].transpose(1, 0, 2)  # [T, N, 2]
        sm = surround_mask[:, :T_min].T                       # [T, N]

        # Step 1: remove closest actor if virtually co-located (V2X partner ego).
        actor_keep = np.ones(N_actors, dtype=bool)
        if ego_pose is not None:
            ego_xy = np.asarray(ego_pose[:2], dtype=np.float64)
            first_pos = surround_traj[:, 0, :]  # [N, 2] — proxy for current pos
            cur_dists = np.linalg.norm(first_pos - ego_xy, axis=1)
            closest_idx = int(np.argmin(cur_dists))
            if cur_dists[closest_idx] < _CLOSE_ACTOR_THRESH:
                actor_keep[closest_idx] = False

        # Step 2: L2 distances over all timesteps.
        A = pred[:T_min, :2]           # [T, 2]
        diffs_all = A[:, None, :] - B  # [T, N, 2]
        l2_all    = np.linalg.norm(diffs_all, axis=2)  # [T, N]
        valid_sm  = sm > 0
        l2_gated  = np.where(valid_sm, l2_all, np.inf)
        l2_gated[:, ~actor_keep] = np.inf

        # Step 3: lateral check only where L2 < gate.
        close_pairs = l2_gated < _EUCLIDEAN_GATE
        if np.any(close_pairs):
            t_indices, n_indices = np.where(close_pairs)
            delta_y = np.abs(A[t_indices, 1] - B[t_indices, n_indices, 1])
            lat_ok  = delta_y <= _LAT_THRESH
            if np.any(lat_ok):
                collision = 1
                first     = int(np.where(lat_ok)[0][0])
                t_hit     = int(t_indices[first])
                n_hit     = int(n_indices[first])
                _dt_step  = float(PREDICTION_DOWNSAMPLING_RATE) * 0.1
                actor_id_hit: Optional[int] = None
                if (
                    surround_actor_ids is not None
                    and isinstance(surround_actor_ids, list)
                    and 0 <= n_hit < len(surround_actor_ids)
                ):
                    try:
                        actor_id_hit = int(surround_actor_ids[n_hit])
                    except Exception:
                        pass
                collision_debug = {
                    "collision": True,
                    "actor_index": n_hit,
                    "actor_id": actor_id_hit,
                    "step_index": t_hit,
                    "t_seconds": float((t_hit + 1) * _dt_step),
                    "delta_y_m": float(delta_y[first]),
                    "euclidean_m": float(l2_all[t_hit, n_hit]),
                    "pred_world_xy": [float(A[t_hit, 0]), float(A[t_hit, 1])],
                    "actor_world_xy": [
                        float(B[t_hit, n_hit, 0]),
                        float(B[t_hit, n_hit, 1]),
                    ],
                }

    per_step_errors = [None] * T_min
    for idx in np.where(mask)[0]:
        per_step_errors[int(idx)] = float(dists[idx])

    return {
        "plan_ade": plan_ade,
        "plan_fde": plan_fde,
        "collision": collision,
        "collision_debug": collision_debug,
        "n_valid": n_valid,
        "per_step_errors": per_step_errors,
        "valid_mask": [bool(v) for v in mask.tolist()],
    }


class CanonicalMetricAccumulator:
    """Accumulates Stage-4 canonical fresh-only metrics and coverage."""

    def __init__(self):
        self.plan_ade_sum = 0.0
        self.plan_fde_sum = 0.0
        self.collision_count = 0
        self.n_frames = 0
        self.total_frames = 0
        self.scored_frames = 0
        self.skipped_stale = 0
        self.skipped_unresolved_semantics = 0
        self.skipped_insufficient_horizon = 0
        self.skipped_missing_canonical = 0
        self.skipped_run_step_exception = 0
        self.skipped_other = 0
        from canonical_trajectory import CANONICAL_TIMESTAMPS

        n_steps = len(CANONICAL_TIMESTAMPS)
        # Per-horizon ADE tracking on the configured canonical grid.
        self._per_step_ade_sum = [0.0] * n_steps
        self._per_step_count = [0] * n_steps

    def record_skip(self, reason: str) -> None:
        self.total_frames += 1
        reason = str(reason or "other")
        if reason == "stale":
            self.skipped_stale += 1
        elif reason == "unresolved_semantics":
            self.skipped_unresolved_semantics += 1
        elif reason == "insufficient_horizon":
            self.skipped_insufficient_horizon += 1
        elif reason == "missing_canonical":
            self.skipped_missing_canonical += 1
        elif reason == "run_step_exception":
            self.skipped_run_step_exception += 1
        else:
            self.skipped_other += 1

    def update_from_debug_record(self, debug_record: Dict[str, Any]) -> None:
        scored = bool(debug_record.get("scored", False))
        if not scored:
            self.record_skip(str(debug_record.get("inclusion_reason", "other")))
            return
        self.total_frames += 1
        self.scored_frames += 1
        self.plan_ade_sum += float(debug_record.get("plan_ade", 0.0))
        self.plan_fde_sum += float(debug_record.get("plan_fde", 0.0))
        self.collision_count += int(bool(debug_record.get("collision", 0)))
        self.n_frames += 1
        # Per-horizon ADE tracking.
        breakdown = debug_record.get("ade_breakdown_m")
        if isinstance(breakdown, (list, tuple, np.ndarray)) and len(breakdown) > 0:
            for idx in range(min(len(self._per_step_ade_sum), len(breakdown))):
                val = breakdown[idx]
                if val is not None and isinstance(val, (int, float)):
                    self._per_step_ade_sum[idx] += float(val)
                    self._per_step_count[idx] += 1

    def summary(self) -> Dict[str, float]:
        if self.n_frames == 0:
            return {
                "plan_ade": float("nan"),
                "plan_fde": float("nan"),
                "collision_rate": float("nan"),
                "n_frames": 0,
            }
        return {
            "plan_ade": self.plan_ade_sum / self.n_frames,
            "plan_fde": self.plan_fde_sum / self.n_frames,
            "collision_rate": self.collision_count / self.n_frames,
            "n_frames": self.n_frames,
        }

    def per_horizon_ade(self) -> Dict[str, Any]:
        """Per-canonical-timestep mean ADE keyed by timestamp label (e.g. '0.5s')."""
        from canonical_trajectory import CANONICAL_TIMESTAMPS
        result: Dict[str, Any] = {}
        for idx, ts in enumerate(CANONICAL_TIMESTAMPS):
            key = f"{float(ts):.1f}s"
            if idx < len(self._per_step_count) and self._per_step_count[idx] > 0:
                result[key] = round(self._per_step_ade_sum[idx] / self._per_step_count[idx], 4)
            else:
                result[key] = None
        return result

    def coverage_summary(self) -> Dict[str, int]:
        return {
            "total_frames": int(self.total_frames),
            "scored_frames": int(self.scored_frames),
            "skipped_stale": int(self.skipped_stale),
            "skipped_unresolved_semantics": int(self.skipped_unresolved_semantics),
            "skipped_insufficient_horizon": int(self.skipped_insufficient_horizon),
            "skipped_missing_canonical": int(self.skipped_missing_canonical),
            "skipped_run_step_exception": int(self.skipped_run_step_exception),
            "skipped_other": int(self.skipped_other),
        }


# ============================================================
# Visualization & logging helper
# ============================================================

class _VisLogger:
    """
    Handles console output and optional visualizations.

    Basic visualizations (matplotlib):
      <viz_dir>/<scenario>_<planner>_trajectory.png
      <viz_dir>/<scenario>_<planner>_metrics.png
      <viz_dir>/<scenario>_<planner>_bev_<frame>.png

    Rich visualizations (OpenCV):
      <viz_dir>/<scenario>_<planner>_rich_<frame>.png
      <viz_dir>/<scenario>_<planner>_rich.mp4
    """

    _W  = 72    # console width
    _HR = "─" * 72

    def __init__(self,
                 planner_name: str,
                 viz_dir: Optional[str] = None,
                 viz_every: int = 5,
                 verbose: bool = True,
                 viz_style: str = "basic",
                 viz_video: Optional[bool] = None,
                 video_fps: Optional[float] = None,
                 video_codec: str = "mp4v",
                 panel_width: int = 1920):
        self.planner     = planner_name
        self.viz_dir     = Path(viz_dir) if viz_dir else None
        self.viz_every   = max(1, viz_every)
        self.verbose     = verbose
        self.viz_style   = viz_style
        self.video_fps   = video_fps
        self.video_codec = video_codec
        self.panel_width = max(960, int(panel_width))
        self.viz_video   = (viz_style == "rich") if viz_video is None else bool(viz_video)
        if viz_style != "rich":
            self.viz_video = False

        if self.viz_dir:
            self.viz_dir.mkdir(parents=True, exist_ok=True)

        # Per-scenario buffers (reset at scenario start)
        self._gt_traj:   Optional[np.ndarray] = None
        self._pred_list: List[np.ndarray]     = []   # [T,2] per frame
        self._raw_pred_list: List[Optional[np.ndarray]] = []   # raw (pre-resample) per frame
        self._frame_ades: List[float]         = []
        self._frame_fdes: List[float]         = []
        self._frame_cols: List[int]           = []
        self._frame_latency_ms: List[float]   = []
        self._scenario_name: str             = ""
        self._scenario_fps: float             = 10.0
        self._video_writer = None
        self._video_path: Optional[Path]      = None
        self._video_enabled: bool             = False
        self._video_frames: int               = 0

    # ------------------------------------------------------------------
    # Scenario-level hooks
    # ------------------------------------------------------------------

    def begin_scenario(self, scenario_name: str, actor_ids: List[int],
                       primary_actor_id: int, n_frames: int,
                       gt_traj: np.ndarray, fps: float = 10.0) -> None:
        self._scenario_name = scenario_name
        self._gt_traj       = gt_traj
        self._scenario_fps  = float(fps) if fps else 10.0
        self._pred_list.clear()
        self._raw_pred_list.clear()
        self._frame_ades.clear()
        self._frame_fdes.clear()
        self._frame_cols.clear()
        self._frame_latency_ms.clear()
        self._close_video_writer()
        self._video_frames  = 0
        self._video_enabled = bool(
            self.viz_dir and self.viz_style == "rich" and self.viz_video and _CV2_AVAILABLE
        )
        if self.viz_style == "rich" and not _CV2_AVAILABLE:
            print("  [viz] rich mode requested but OpenCV is unavailable — falling back to basic mode")
        if self._video_enabled:
            self._video_path = self.viz_dir / f"{scenario_name}_{self.planner}_rich.mp4"
        else:
            self._video_path = None

        path_len = float(np.sum(np.linalg.norm(np.diff(gt_traj, axis=0), axis=1))) \
                   if len(gt_traj) > 1 else 0.0

        x_range = f"[{gt_traj[:,0].min():.1f}, {gt_traj[:,0].max():.1f}]"
        y_range = f"[{gt_traj[:,1].min():.1f}, {gt_traj[:,1].max():.1f}]"

        print()
        print("┌" + "─" * (self._W - 2) + "┐")
        print(f"│  Scenario : {scenario_name:<{self._W - 16}}│")
        print(f"│  Planner  : {self.planner:<{self._W - 16}}│")
        print(f"│  Actors   : {str(actor_ids):<{self._W - 16}}│")
        print(f"│  Ego      : actor {primary_actor_id}  │  Frames: {n_frames}  │  GT path: {path_len:.2f} m{'':{max(0,self._W-16-len(str(actor_ids))-45)}}│")
        print(f"│  GT x     : {x_range:<{self._W - 16}}│")
        print(f"│  GT y     : {y_range:<{self._W - 16}}│")
        print(f"│  Metrics  : Plan ADE / FDE / Collision Rate — formula from{'':{self._W-63}}│")
        print(f"│             opencood/utils/eval_utils.py{'':{self._W-43}}│")
        print("└" + "─" * (self._W - 2) + "┘")
        print(self._HR)

    def end_scenario(self, metrics: Dict[str, Any], per_frame: List[Dict],
                     scenario_name: str, primary_actor_id: int) -> None:
        self._close_video_writer()
        n = metrics["n_frames"]
        ade = metrics["plan_ade"]
        fde = metrics["plan_fde"]
        col = metrics["collision_rate"]

        print(self._HR)
        print(f"  Scenario result  [{scenario_name}]  ego={primary_actor_id}")
        print(f"  ┌────────────────────────────────────────┐")
        print(f"  │  Plan ADE        : {ade:>8.4f} m          │")
        print(f"  │  Plan FDE        : {fde:>8.4f} m          │")
        print(f"  │  Collision Rate  : {col:>8.4f}             │")
        print(f"  │  Measured frames : {n:>8d}             │")
        if self._frame_ades:
            print(f"  │  Per-frame ADE   : min={min(self._frame_ades):.3f}  max={max(self._frame_ades):.3f}   │")
        print(f"  └────────────────────────────────────────┘")

        # Save visualizations
        if self.viz_style == "rich" and self.viz_dir and self._video_frames > 0:
            if self._video_path is not None:
                print(f"  [viz] saved {self._video_path.name}")
        if _MPL_AVAILABLE and self.viz_dir and n > 0:
            self._save_trajectory_plot(scenario_name, per_frame)
            self._save_metrics_plot(scenario_name)
        elif not _MPL_AVAILABLE and self.viz_dir:
            print("  [viz] matplotlib not available — skipping plots")

    # ------------------------------------------------------------------
    # Frame-level hook
    # ------------------------------------------------------------------

    def log_frame(self, frame_idx: int, n_frames: int,
                  pose: np.ndarray, frame_data,
                  bev_img: Optional[np.ndarray],
                  pred_world: Optional[np.ndarray],
                  gt_future_xy: Optional[np.ndarray],
                  frame_metrics: Dict[str, float],
                  latency_s: float,
                  cam_assign: Optional[Dict],
                  run_step_exc: Optional[Exception] = None,
                  pred_world_raw: Optional[np.ndarray] = None) -> None:
        """Log one frame. Always called; detail level controlled by self.verbose."""
        has_prediction = pred_world is not None

        # Cache predictions for trajectory plot
        if pred_world is not None:
            self._pred_list.append(pred_world)
        self._raw_pred_list.append(pred_world_raw)  # None if unavailable
        if "plan_ade" in frame_metrics:
            self._frame_ades.append(frame_metrics["plan_ade"])
            # Keep ADE/FDE arrays aligned so plotting cannot crash on length mismatch.
            self._frame_fdes.append(frame_metrics.get("plan_fde", float("nan")))
            self._frame_cols.append(frame_metrics.get("collision", 0))
        self._frame_latency_ms.append(float(latency_s * 1000.0))

        if not self.verbose:
            # Minimal one-liner
            ade_s = f"{frame_metrics.get('plan_ade', float('nan')):.3f}" \
                    if "plan_ade" in frame_metrics else "  n/a "
            print(f"  [{frame_idx:03d}/{n_frames-1:03d}]  "
                  f"ADE={ade_s} m  "
                  f"col={frame_metrics.get('collision', '-')}  "
                  f"t={latency_s*1000:.0f}ms"
                  + ("  ⚠ run_step FAILED" if run_step_exc else ""))
            # Visualization output even in non-verbose mode.
            if self.viz_style == "rich" and has_prediction:
                self._save_rich_frame(
                    frame_idx=frame_idx,
                    n_frames=n_frames,
                    pose=pose,
                    frame_data=frame_data,
                    bev_img=bev_img,
                    pred_world=pred_world,
                    gt_future_xy=gt_future_xy,
                    frame_metrics=frame_metrics,
                    latency_s=latency_s,
                    run_step_exc=run_step_exc,
                    pred_world_raw=pred_world_raw,
                )
            elif (_MPL_AVAILABLE and self.viz_dir and bev_img is not None
                    and pred_world is not None
                    and frame_idx % self.viz_every == 0):
                self._save_bev_frame(frame_idx, bev_img, pred_world,
                                     gt_future_xy, pose, pred_world_raw=pred_world_raw)
            return

        # ── Verbose output ──────────────────────────────────────────────
        lidar = frame_data.car_data_raw_entry["lidar"]
        meas  = frame_data.car_data_raw_entry["measurements"]

        yaw_deg = float(pose[4])
        gps_x   = float(meas.get("gps_x", 0.0))
        gps_y   = float(meas.get("gps_y", 0.0))
        speed   = float(meas.get("speed", 0.0))

        lidar_n = lidar.shape[0]
        lidar_z = (lidar[:, 2].min(), lidar[:, 2].max()) if lidar_n > 0 else (0, 0)

        if bev_img is not None:
            white_frac = float((np.sum(bev_img > 200, axis=2) > 2).sum()) / \
                         (bev_img.shape[0] * bev_img.shape[1])
        else:
            white_frac = float("nan")

        cam_str = ""
        if cam_assign:
            cam_str = "  ".join(f"{d}={k}" for k, d in cam_assign.items())

        print(f"\n  ━━━ Frame {frame_idx:03d}/{n_frames-1:03d} "
              f"{'━' * max(0, self._W - 18)}")
        print(f"  Ego pos   : ({gps_x:.2f}, {gps_y:.2f} m)  "
              f"yaw={yaw_deg:.1f}°  speed={speed:.2f} m/s")
        if cam_str:
            print(f"  Cameras   : {cam_str}")
        print(f"  LiDAR     : {lidar_n:,} pts  "
              f"z ∈ [{lidar_z[0]:.2f}, {lidar_z[1]:.2f}] m")
        print(f"  BEV cov   : {white_frac*100:.1f}%  (drivable pixels / total)")

        if run_step_exc is not None:
            print(f"  ⚠ run_step EXCEPTION: {run_step_exc}")
        elif pred_world is None:
            print(f"  ⚠ no waypoints extracted from planner")
        else:
            T = len(pred_world)
            p0 = pred_world[0]  if T > 0 else np.zeros(2)
            p_end = pred_world[-1] if T > 0 else np.zeros(2)
            print(f"  Predicted : {T} waypoints (BEV→world)")
            print(f"    first=({p0[0]:.2f},{p0[1]:.2f})  "
                  f"last=({p_end[0]:.2f},{p_end[1]:.2f})")
            if gt_future_xy is not None and len(gt_future_xy) > 0:
                g0 = gt_future_xy[0]
                g_end = gt_future_xy[min(len(gt_future_xy)-1, T-1)]
                print(f"  GT future : {len(gt_future_xy)} steps  "
                      f"({g0[0]:.2f},{g0[1]:.2f}) … ({g_end[0]:.2f},{g_end[1]:.2f})")

        if "plan_ade" in frame_metrics:
            col_flag = "⚠ COLLISION" if frame_metrics.get("collision") else "✓"
            print(f"  Metrics   : ADE={frame_metrics['plan_ade']:.4f} m  "
                  f"FDE={frame_metrics['plan_fde']:.4f} m  {col_flag}")
        print(f"  Latency   : {latency_s*1000:.1f} ms")

        # Visualization output
        if self.viz_style == "rich" and has_prediction:
            self._save_rich_frame(
                frame_idx=frame_idx,
                n_frames=n_frames,
                pose=pose,
                frame_data=frame_data,
                bev_img=bev_img,
                pred_world=pred_world,
                gt_future_xy=gt_future_xy,
                frame_metrics=frame_metrics,
                latency_s=latency_s,
                run_step_exc=run_step_exc,
                pred_world_raw=pred_world_raw,
            )
        elif (_MPL_AVAILABLE and self.viz_dir and bev_img is not None
                and pred_world is not None
                and frame_idx % self.viz_every == 0):
            self._save_bev_frame(frame_idx, bev_img, pred_world,
                                 gt_future_xy, pose, pred_world_raw=pred_world_raw)

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def _save_bev_frame(self, frame_idx: int, bev_img: np.ndarray,
                        pred_world: np.ndarray,
                        gt_future_xy: Optional[np.ndarray],
                        pose: np.ndarray,
                        pred_world_raw: Optional[np.ndarray] = None) -> None:
        """Save BEV image + predicted waypoints for one frame."""
        if not _MPL_AVAILABLE:
            return
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            f"{self._scenario_name}  |  {self.planner}  |  frame {frame_idx:03d}",
            fontsize=11, fontweight="bold"
        )

        # ── Left: BEV image ──────────────────────────────────────────
        ax = axes[0]
        ax.imshow(bev_img, origin="upper")
        ax.set_title("BEV (drivable area = white)", fontsize=9)
        ax.axis("off")

        # Overlay BEV-frame predicted waypoints
        # BEV pixel: cx=200, cy=200, ppm=5, fwd=up, left=left
        ppm  = 5
        cx = cy = 200
        # Recover BEV coords from world coords (approximate — using
        # _bev_to_world inverse for display only)
        compass_rad = _fallback_compass_from_pose(pose)
        theta = compass_rad + math.pi / 2
        R = np.array([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta),  math.cos(theta)]])
        ego_xy = np.array([pose[0], pose[1]])
        bev_coords = (R.T @ (pred_world - ego_xy).T).T  # [T, 2] (fwd, left)
        u_coords = cx - bev_coords[:, 1] * ppm   # left → left pixel
        v_coords = cy - bev_coords[:, 0] * ppm   # fwd  → up pixel
        ax.plot(u_coords, v_coords, "b.-", markersize=5, linewidth=1.5,
                label=f"resampled ({len(pred_world)} pts)")
        if pred_world_raw is not None and len(pred_world_raw) > 0:
            bev_raw = (R.T @ (np.asarray(pred_world_raw) - ego_xy).T).T
            u_raw = cx - bev_raw[:, 1] * ppm
            v_raw = cy - bev_raw[:, 0] * ppm
            ax.plot(u_raw, v_raw, "o", color="orange", markersize=6, linewidth=0,
                    label=f"raw ({len(pred_world_raw)} pts)")
        ax.plot(cx, cy, "g*", markersize=12, label="ego")
        ax.legend(loc="lower right", fontsize=8)

        # ── Right: world-frame view ───────────────────────────────────
        ax2 = axes[1]
        ax2.set_aspect("equal")
        ax2.set_title("World frame (top-down)", fontsize=9)
        ax2.set_xlabel("x (m)")
        ax2.set_ylabel("y (m)")

        if self._gt_traj is not None:
            ax2.plot(self._gt_traj[:, 0], self._gt_traj[:, 1],
                     "k--", linewidth=1, alpha=0.4, label="GT full")

        ax2.plot(pred_world[:, 0], pred_world[:, 1],
                 "b.-", markersize=4, linewidth=1.5,
                 label=f"resampled ({len(pred_world)} pts)")
        if pred_world_raw is not None and len(pred_world_raw) > 0:
            raw = np.asarray(pred_world_raw)
            ax2.plot(raw[:, 0], raw[:, 1], "o", color="orange",
                     markersize=6, linewidth=0, label=f"raw ({len(raw)} pts)")
        ax2.plot(ego_xy[0], ego_xy[1], "g*", markersize=12, label="ego now")

        if gt_future_xy is not None and len(gt_future_xy) > 0:
            valid = gt_future_xy[gt_future_xy[:, 0] != 0]
            if len(valid):
                ax2.plot(valid[:, 0], valid[:, 1],
                         "c.-", markersize=4, linewidth=1.5, label="GT future")

        ax2.legend(loc="best", fontsize=8)
        ax2.grid(True, alpha=0.3)

        fname = (self.viz_dir /
                 f"{self._scenario_name}_{self.planner}_bev_{frame_idx:04d}.png")
        fig.savefig(fname, dpi=100, bbox_inches="tight")
        plt.close(fig)
        if self.verbose:
            print(f"  [viz] saved {fname.name}")

    def _close_video_writer(self) -> None:
        if self._video_writer is not None:
            try:
                self._video_writer.release()
            except Exception:
                pass
            self._video_writer = None

    def _init_video_writer_if_needed(self, frame_bgr: np.ndarray) -> None:
        if not self._video_enabled or self._video_writer is not None:
            return
        if not _CV2_AVAILABLE or self._video_path is None:
            self._video_enabled = False
            return
        h, w = frame_bgr.shape[:2]
        fps = float(self.video_fps) if self.video_fps else float(self._scenario_fps)
        if not np.isfinite(fps) or fps <= 0:
            fps = 10.0
        try:
            fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
            writer = cv2.VideoWriter(str(self._video_path), fourcc, fps, (w, h))
            if not writer.isOpened():
                print("  [viz] warning: failed to initialize video writer — continuing with PNG only")
                self._video_enabled = False
                return
            self._video_writer = writer
        except Exception as exc:
            print(f"  [viz] warning: failed to initialize video writer ({exc}) — continuing with PNG only")
            self._video_enabled = False

    @staticmethod
    def _draw_polyline_with_mask(image: np.ndarray, pts: np.ndarray, mask: np.ndarray,
                                 color: Tuple[int, int, int], thickness: int = 2,
                                 radius: int = 3) -> None:
        if pts is None or len(pts) == 0:
            return
        if mask is None:
            mask = np.ones(len(pts), dtype=bool)
        mask = np.asarray(mask, dtype=bool)
        for i in range(1, len(pts)):
            if mask[i] and mask[i - 1]:
                p0 = tuple(np.round(pts[i - 1]).astype(int))
                p1 = tuple(np.round(pts[i]).astype(int))
                cv2.line(image, p0, p1, color, thickness, lineType=cv2.LINE_AA)
        for i in np.where(mask)[0]:
            p = tuple(np.round(pts[i]).astype(int))
            cv2.circle(image, p, radius, color, -1, lineType=cv2.LINE_AA)

    @staticmethod
    def _world_to_local(world_xy: np.ndarray, ego_xy: np.ndarray, compass_rad: float) -> np.ndarray:
        theta = compass_rad + math.pi / 2.0
        R = np.array([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta),  math.cos(theta)]], dtype=np.float64)
        return (R.T @ (world_xy - ego_xy).T).T  # [N,2] = [forward, left]

    def _world_to_bev_pixels(self, world_xy: np.ndarray, pose: np.ndarray,
                              ppm: float = 5.0, cx: float = 200.0, cy: float = 200.0,
                              compass_rad: Optional[float] = None) -> np.ndarray:
        if world_xy is None or len(world_xy) == 0:
            return np.zeros((0, 2), dtype=np.float64)
        ego_xy = np.array([float(pose[0]), float(pose[1])], dtype=np.float64)
        if compass_rad is None:
            compass_rad = _fallback_compass_from_pose(pose)
        local = self._world_to_local(np.asarray(world_xy, dtype=np.float64), ego_xy, compass_rad)
        u = cx - local[:, 1] * ppm
        v = cy - local[:, 0] * ppm
        return np.stack([u, v], axis=1)

    def _project_world_xy_to_front(self, world_xy: np.ndarray, pose: np.ndarray,
                                   measurements: Dict[str, Any],
                                   image_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        if world_xy is None or len(world_xy) == 0:
            return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=bool)
        K = np.array(measurements.get("camera_front_intrinsics", []), dtype=np.float64)
        ext = np.array(measurements.get("camera_front_extrinsics", []), dtype=np.float64)
        if K.shape != (3, 3) or ext.shape != (4, 4):
            n = len(world_xy)
            return np.zeros((n, 2), dtype=np.float64), np.zeros((n,), dtype=bool)

        h, w = image_shape[:2]
        world_xy = np.asarray(world_xy, dtype=np.float64)
        ego_xy = np.array([float(pose[0]), float(pose[1])], dtype=np.float64)
        compass = _resolve_compass_rad(pose, measurements)
        local = self._world_to_local(world_xy, ego_xy, compass)

        # Ground-plane assumption in lidar frame: x=fwd, y=-left, z=0.
        pts_lidar = np.stack([local[:, 0], -local[:, 1], np.zeros(len(local))], axis=1)
        pts_h = np.concatenate([pts_lidar, np.ones((len(pts_lidar), 1), dtype=np.float64)], axis=1)

        def _project(transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
            cam = (transform @ pts_h.T).T[:, :3]
            z = cam[:, 2]
            valid = np.isfinite(z) & (z > 1e-5)
            uv = np.full((len(cam), 2), np.nan, dtype=np.float64)
            if np.any(valid):
                proj = (K @ cam[valid].T).T
                uv_valid = proj[:, :2] / proj[:, 2:3]
                uv[valid] = uv_valid
            valid &= np.isfinite(uv).all(axis=1)
            valid &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            return uv, valid, int(valid.sum())

        candidates: List[np.ndarray] = [ext]
        try:
            ext_inv = np.linalg.inv(ext)
            candidates.append(ext_inv)
        except np.linalg.LinAlgError:
            pass

        best_uv = np.zeros((len(world_xy), 2), dtype=np.float64)
        best_mask = np.zeros((len(world_xy),), dtype=bool)
        best_score = -1
        for T in candidates:
            uv, mask, score = _project(T)
            if score > best_score:
                best_uv, best_mask, best_score = uv, mask, score
        return best_uv, best_mask

    def _draw_front_overlay(self, front_rgb: np.ndarray, pose: np.ndarray,
                            frame_data, pred_world: Optional[np.ndarray],
                            gt_future_xy: Optional[np.ndarray],
                            run_step_exc: Optional[Exception]) -> np.ndarray:
        if front_rgb is None:
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
        else:
            img = np.asarray(front_rgb)
            if img.ndim != 3 or img.shape[2] != 3:
                img = np.zeros((720, 1280, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)

        meas = frame_data.car_data_raw_entry.get("measurements", {})

        if pred_world is not None and len(pred_world) > 0:
            pred_uv, pred_mask = self._project_world_xy_to_front(pred_world, pose, meas, img.shape)
            self._draw_polyline_with_mask(img, pred_uv, pred_mask, color=(40, 40, 255), thickness=3, radius=4)

        if gt_future_xy is not None and len(gt_future_xy) > 0:
            gt = np.asarray(gt_future_xy, dtype=np.float64)
            gt_mask = np.any(np.abs(gt) > 1e-9, axis=1)
            gt_uv, gt_proj_mask = self._project_world_xy_to_front(gt, pose, meas, img.shape)
            self._draw_polyline_with_mask(
                img, gt_uv, gt_mask & gt_proj_mask, color=(255, 255, 0), thickness=2, radius=3
            )

        # Subtle ego marker near lower center
        h, w = img.shape[:2]
        cx, cy = int(w * 0.5), int(h * 0.90)
        cv2.line(img, (cx - 12, cy), (cx + 12, cy), (235, 235, 235), 2, cv2.LINE_AA)
        cv2.line(img, (cx, cy - 12), (cx, cy + 12), (235, 235, 235), 2, cv2.LINE_AA)

        cv2.putText(img, "Pred", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "GT",   (120, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

        if run_step_exc is not None:
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (w, 56), (0, 0, 160), -1)
            img = cv2.addWeighted(overlay, 0.38, img, 0.62, 0)
            msg = str(run_step_exc).replace("\n", " ")
            if len(msg) > 130:
                msg = msg[:127] + "..."
            cv2.putText(img, f"run_step EXCEPTION: {msg}", (12, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2, cv2.LINE_AA)

        return img

    @staticmethod
    def _valid_world_points(world_xy: Optional[np.ndarray]) -> np.ndarray:
        """Return finite, non-zero world XY points as [N,2]."""
        if world_xy is None:
            return np.zeros((0, 2), dtype=np.float64)
        arr = np.asarray(world_xy, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) == 0:
            return np.zeros((0, 2), dtype=np.float64)
        finite = np.isfinite(arr).all(axis=1)
        nonzero = np.any(np.abs(arr) > 1e-9, axis=1)
        mask = finite & nonzero
        return arr[mask]

    def _render_bev_panel(self, bev_img: Optional[np.ndarray], pose: np.ndarray,
                          pred_world: Optional[np.ndarray],
                          gt_future_xy: Optional[np.ndarray],
                          width: int, height: int,
                          compass_rad: Optional[float] = None,
                          pred_world_raw: Optional[np.ndarray] = None) -> np.ndarray:
        if bev_img is None:
            bev = np.zeros((400, 400, 3), dtype=np.uint8)
        else:
            bev = np.asarray(bev_img).copy()
            if bev.ndim != 3 or bev.shape[2] != 3:
                bev = np.zeros((400, 400, 3), dtype=np.uint8)
            else:
                bev = cv2.cvtColor(bev.astype(np.uint8), cv2.COLOR_RGB2BGR)

        h0, w0 = bev.shape[:2]
        ppm = float(w0) / 80.0  # 80m span for 400px default birdview
        cx = float(w0) / 2.0
        cy = float(h0) / 2.0

        if pred_world is not None and len(pred_world) > 0:
            pred_px = self._world_to_bev_pixels(
                pred_world, pose, ppm=ppm, cx=cx, cy=cy, compass_rad=compass_rad
            )
            pred_mask = np.ones(len(pred_px), dtype=bool)
            self._draw_polyline_with_mask(bev, pred_px, pred_mask, color=(200, 80, 80), thickness=2, radius=2)

        if pred_world_raw is not None and len(pred_world_raw) > 0:
            raw_px = self._world_to_bev_pixels(
                np.asarray(pred_world_raw, dtype=np.float64),
                pose, ppm=ppm, cx=cx, cy=cy, compass_rad=compass_rad
            )
            raw_mask = np.ones(len(raw_px), dtype=bool)
            # Orange circles (no connecting line) for exact raw predictions
            for i in np.where(raw_mask)[0]:
                p = tuple(np.round(raw_px[i]).astype(int))
                cv2.circle(bev, p, 5, (0, 165, 255), -1, lineType=cv2.LINE_AA)

        if gt_future_xy is not None and len(gt_future_xy) > 0:
            gt = np.asarray(gt_future_xy, dtype=np.float64)
            gt_mask = np.any(np.abs(gt) > 1e-9, axis=1)
            gt_px = self._world_to_bev_pixels(
                gt, pose, ppm=ppm, cx=cx, cy=cy, compass_rad=compass_rad
            )
            self._draw_polyline_with_mask(bev, gt_px, gt_mask, color=(255, 255, 0), thickness=2, radius=2)

        cv2.circle(bev, (int(cx), int(cy)), 6, (60, 255, 60), -1, lineType=cv2.LINE_AA)
        bev = cv2.resize(bev, (width, height), interpolation=cv2.INTER_LINEAR)
        cv2.putText(bev, "BEV", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(bev, "Raw=orange  Resampled=red  GT=cyan  Ego=green",
                    (10, max(42, height - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (225, 225, 225), 1, cv2.LINE_AA)
        return bev

    def _render_world_panel(self, frame_idx: int, pose: np.ndarray,
                            pred_world: Optional[np.ndarray],
                            gt_future_xy: Optional[np.ndarray],
                            width: int, height: int,
                            compass_rad: Optional[float] = None,
                            pred_world_raw: Optional[np.ndarray] = None) -> np.ndarray:
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (18, 18, 18)

        # Dynamic zoom: keep trajectories readable even in very short-horizon
        # near-stationary segments where fixed 30 m zoom collapses into a dot.
        radius_m = 30.0
        extent_samples = []
        pred_valid = self._valid_world_points(pred_world)
        if len(pred_valid) > 0:
            extent_samples.append(pred_valid)
        if pred_world_raw is not None:
            raw_valid = self._valid_world_points(pred_world_raw)
            if len(raw_valid) > 0:
                extent_samples.append(raw_valid)
        gt_valid = self._valid_world_points(gt_future_xy)
        if len(gt_valid) > 0:
            extent_samples.append(gt_valid)
        if self._gt_traj is not None and len(self._gt_traj) > 0:
            hist = self._gt_traj[max(0, frame_idx - 25):frame_idx + 1]
            if len(hist) > 0:
                extent_samples.append(hist)
        if extent_samples:
            ego_xy_tmp = np.array([float(pose[0]), float(pose[1])], dtype=np.float64)
            compass_tmp = _fallback_compass_from_pose(pose) if compass_rad is None else float(compass_rad)
            union = np.vstack(extent_samples)
            local_union = self._world_to_local(union, ego_xy_tmp, compass_tmp)
            max_abs = float(np.max(np.abs(local_union))) if len(local_union) else 0.0
            radius_m = float(np.clip(max(8.0, max_abs * 1.35), 8.0, 30.0))

        ppm = min((width * 0.45) / radius_m, (height * 0.70) / radius_m)
        origin = np.array([width * 0.50, height * 0.82], dtype=np.float64)

        # Grid
        for d in range(-30, 31, 5):
            c = (55, 55, 55) if d % 10 == 0 else (36, 36, 36)
            u = int(origin[0] + d * ppm)
            v = int(origin[1] - d * ppm)
            cv2.line(panel, (u, 0), (u, height), c, 1, cv2.LINE_AA)
            cv2.line(panel, (0, v), (width, v), c, 1, cv2.LINE_AA)

        ego_xy = np.array([float(pose[0]), float(pose[1])], dtype=np.float64)
        compass = _fallback_compass_from_pose(pose) if compass_rad is None else float(compass_rad)

        def _to_panel(world_xy: np.ndarray) -> np.ndarray:
            local = self._world_to_local(world_xy, ego_xy, compass)
            u = origin[0] - local[:, 1] * ppm
            v = origin[1] - local[:, 0] * ppm
            return np.stack([u, v], axis=1)

        # GT history
        if self._gt_traj is not None and len(self._gt_traj) > 0:
            hist = self._gt_traj[:frame_idx + 1]
            if len(hist) > 1:
                hist_px = _to_panel(hist)
                hist_mask = np.ones(len(hist_px), dtype=bool)
                self._draw_polyline_with_mask(panel, hist_px, hist_mask, color=(120, 120, 120), thickness=2, radius=0)

        # GT future
        if gt_future_xy is not None and len(gt_future_xy) > 0:
            gt = np.asarray(gt_future_xy, dtype=np.float64)
            gt_mask = np.any(np.abs(gt) > 1e-9, axis=1)
            gt_px = _to_panel(gt)
            self._draw_polyline_with_mask(panel, gt_px, gt_mask, color=(255, 255, 0), thickness=2, radius=2)

        # Prediction (resampled — blue line with dots)
        if pred_world is not None and len(pred_world) > 0:
            pred_px = _to_panel(np.asarray(pred_world, dtype=np.float64))
            pred_mask = np.ones(len(pred_px), dtype=bool)
            self._draw_polyline_with_mask(panel, pred_px, pred_mask, color=(200, 80, 80), thickness=2, radius=2)

        # Raw prediction (uninterpolated — orange circles only)
        if pred_world_raw is not None and len(pred_world_raw) > 0:
            raw_px = _to_panel(np.asarray(pred_world_raw, dtype=np.float64))
            for px in raw_px:
                p = tuple(np.round(px).astype(int))
                cv2.circle(panel, p, 5, (0, 165, 255), -1, lineType=cv2.LINE_AA)

        cv2.circle(panel, tuple(np.round(origin).astype(int)), 6, (60, 255, 60), -1, lineType=cv2.LINE_AA)
        cv2.putText(panel, "World Local", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(panel, f"zoom={radius_m:.1f}m  Raw=orange  Resampled=red  GT=cyan  Hist=gray",
                    (10, max(42, height - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (225, 225, 225), 1, cv2.LINE_AA)
        return panel

    def _render_status_strip(self, frame_idx: int, n_frames: int,
                             frame_metrics: Dict[str, float], latency_s: float,
                             run_step_exc: Optional[Exception],
                             pred_world: Optional[np.ndarray],
                             gt_future_xy: Optional[np.ndarray],
                             width: int, height: int) -> np.ndarray:
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (24, 24, 24)
        cur_ade = frame_metrics.get("plan_ade", float("nan"))
        cur_fde = frame_metrics.get("plan_fde", float("nan"))
        roll_ade = float(np.nanmean(self._frame_ades)) if self._frame_ades else float("nan")
        roll_fde = float(np.nanmean(self._frame_fdes)) if self._frame_fdes else float("nan")
        lat_ms = float(latency_s * 1000.0)
        roll_lat = float(np.nanmean(self._frame_latency_ms[-20:])) if self._frame_latency_ms else float("nan")
        status = "FAILED" if run_step_exc is not None else "OK"
        c = (90, 90, 255) if run_step_exc is not None else (90, 220, 90)

        pred_valid = self._valid_world_points(pred_world)
        gt_valid = self._valid_world_points(gt_future_xy)
        traj_cmp = "Traj compare: n/a"
        if len(pred_valid) > 0 and len(gt_valid) > 0:
            t = min(len(pred_valid), len(gt_valid))
            end_err = float(np.linalg.norm(pred_valid[t - 1] - gt_valid[t - 1]))
            first_err = float(np.linalg.norm(pred_valid[0] - gt_valid[0]))
            traj_cmp = (
                f"Traj pred/gt pts: {len(pred_valid)}/{len(gt_valid)}    "
                f"err first/end: {first_err:.3f} / {end_err:.3f} m"
            )
        elif len(pred_valid) > 0:
            traj_cmp = f"Traj pred/gt pts: {len(pred_valid)}/0 (no GT future window)"
        elif len(gt_valid) > 0:
            traj_cmp = f"Traj pred/gt pts: 0/{len(gt_valid)} (no predicted path)"

        lines = [
            f"Frame {frame_idx:03d}/{max(0, n_frames-1):03d}   Status: {status}",
            f"ADE cur/mean: {cur_ade:.3f} / {roll_ade:.3f} m    FDE cur/mean: {cur_fde:.3f} / {roll_fde:.3f} m",
            traj_cmp,
            f"Latency cur/mean: {lat_ms:.1f} / {roll_lat:.1f} ms",
            f"Planner: {self.planner}    Scenario: {self._scenario_name}",
        ]
        y = 28
        for i, text in enumerate(lines):
            color = c if i == 0 else (225, 225, 225)
            cv2.putText(panel, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
            y += 26
        return panel

    def _render_rich_composite(self, frame_idx: int, n_frames: int,
                               pose: np.ndarray, frame_data,
                               bev_img: Optional[np.ndarray],
                               pred_world: Optional[np.ndarray],
                               gt_future_xy: Optional[np.ndarray],
                               frame_metrics: Dict[str, float],
                               latency_s: float,
                               run_step_exc: Optional[Exception],
                               pred_world_raw: Optional[np.ndarray] = None) -> np.ndarray:
        measurements = frame_data.car_data_raw_entry.get("measurements", {})
        compass_rad = _resolve_compass_rad(pose, measurements)
        front_rgb = frame_data.car_data_raw_entry.get("rgb_front")
        front = self._draw_front_overlay(
            front_rgb=front_rgb,
            pose=pose,
            frame_data=frame_data,
            pred_world=pred_world,
            gt_future_xy=gt_future_xy,
            run_step_exc=run_step_exc,
        )

        canvas_w = int(self.panel_width)
        left_w = int(round(canvas_w * 0.65))
        right_w = canvas_w - left_w
        if front is None or front.size == 0:
            front = np.zeros((720, 1280, 3), dtype=np.uint8)
        front_h = int(round(left_w * front.shape[0] / max(front.shape[1], 1)))
        total_h = max(640, min(1400, front_h))
        top_h = int(total_h * 0.40)
        mid_h = int(total_h * 0.37)
        bot_h = total_h - top_h - mid_h

        canvas = np.zeros((total_h, canvas_w, 3), dtype=np.uint8)
        front_rs = cv2.resize(front, (left_w, total_h), interpolation=cv2.INTER_LINEAR)
        canvas[:, :left_w] = front_rs

        bev_panel = self._render_bev_panel(
            bev_img, pose, pred_world, gt_future_xy, right_w, top_h,
            compass_rad=compass_rad, pred_world_raw=pred_world_raw
        )
        world_panel = self._render_world_panel(
            frame_idx, pose, pred_world, gt_future_xy, right_w, mid_h,
            compass_rad=compass_rad, pred_world_raw=pred_world_raw
        )
        status_panel = self._render_status_strip(
            frame_idx, n_frames, frame_metrics, latency_s, run_step_exc,
            pred_world, gt_future_xy, right_w, bot_h
        )
        canvas[:top_h, left_w:] = bev_panel
        canvas[top_h:top_h + mid_h, left_w:] = world_panel
        canvas[top_h + mid_h:, left_w:] = status_panel

        # Borders
        cv2.line(canvas, (left_w, 0), (left_w, total_h - 1), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(canvas, (left_w, top_h), (canvas_w - 1, top_h), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (left_w, top_h + mid_h), (canvas_w - 1, top_h + mid_h), (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def _save_rich_frame(self, frame_idx: int, n_frames: int,
                         pose: np.ndarray, frame_data,
                         bev_img: Optional[np.ndarray],
                         pred_world: Optional[np.ndarray],
                         gt_future_xy: Optional[np.ndarray],
                         frame_metrics: Dict[str, float],
                         latency_s: float,
                         run_step_exc: Optional[Exception],
                         pred_world_raw: Optional[np.ndarray] = None) -> None:
        if (not _CV2_AVAILABLE or self.viz_dir is None
                or pred_world is None):
            return
        frame = self._render_rich_composite(
            frame_idx=frame_idx,
            n_frames=n_frames,
            pose=pose,
            frame_data=frame_data,
            bev_img=bev_img,
            pred_world=pred_world,
            gt_future_xy=gt_future_xy,
            frame_metrics=frame_metrics,
            latency_s=latency_s,
            run_step_exc=run_step_exc,
            pred_world_raw=pred_world_raw,
        )
        if self._video_enabled:
            self._init_video_writer_if_needed(frame)
            if self._video_writer is not None:
                self._video_writer.write(frame)
                self._video_frames += 1

        if frame_idx % self.viz_every == 0:
            fname = self.viz_dir / f"{self._scenario_name}_{self.planner}_rich_{frame_idx:04d}.png"
            cv2.imwrite(str(fname), frame)
            if self.verbose:
                print(f"  [viz] saved {fname.name}")

    def _save_trajectory_plot(self, scenario_name: str,
                               per_frame: List[Dict]) -> None:
        """Top-down trajectory overview for the full scenario."""
        if not _MPL_AVAILABLE or not self._gt_traj is not None:
            return
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title(
            f"Trajectory overview\n{scenario_name}  |  {self.planner}",
            fontsize=11, fontweight="bold"
        )
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        # GT path
        gt = self._gt_traj
        ax.plot(gt[:, 0], gt[:, 1], "k-", linewidth=2.5,
                label="GT trajectory", zorder=5)
        ax.plot(gt[0, 0], gt[0, 1], "go", markersize=10,
                label="start", zorder=6)
        ax.plot(gt[-1, 0], gt[-1, 1], "rs", markersize=10,
                label="end", zorder=6)

        # Predicted waypoints (all frames, colour-faded by frame)
        n_pred = len(self._pred_list)
        for i, pred in enumerate(self._pred_list):
            alpha = 0.2 + 0.6 * (i / max(n_pred - 1, 1))
            ax.plot(pred[:, 0], pred[:, 1], ".-",
                    color=(0.5, 0.5, 0.9, alpha), markersize=3,
                    linewidth=0.8)

        # Raw (pre-resample) waypoints — orange dots, no connecting line
        n_raw = len(self._raw_pred_list)
        has_any_raw = any(r is not None for r in self._raw_pred_list)
        if has_any_raw:
            for i, raw in enumerate(self._raw_pred_list):
                if raw is None or len(raw) == 0:
                    continue
                raw = np.asarray(raw)
                alpha = 0.2 + 0.6 * (i / max(n_raw - 1, 1))
                ax.plot(raw[:, 0], raw[:, 1], "o",
                        color=(1.0, 0.55, 0.0, alpha), markersize=4,
                        linewidth=0)

        # Legend proxies
        if n_pred:
            ax.plot([], [], "b.-", label=f"resampled ({n_pred} frames)")
        if has_any_raw:
            ax.plot([], [], "o", color="orange", label="raw predictions")

        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

        fname = self.viz_dir / f"{scenario_name}_{self.planner}_trajectory.png"
        fig.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [viz] saved {fname.name}")

    def _save_metrics_plot(self, scenario_name: str) -> None:
        """Per-frame ADE / FDE / collision time-series."""
        if not _MPL_AVAILABLE or not self._frame_ades:
            return
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        fig.suptitle(
            f"Per-frame metrics\n{scenario_name}  |  {self.planner}",
            fontsize=11, fontweight="bold"
        )
        n = min(len(self._frame_ades), len(self._frame_fdes))
        if n <= 0:
            return
        xs = list(range(n))

        ax = axes[0]
        ade_vals = self._frame_ades[:n]
        fde_vals = self._frame_fdes[:n]
        ax.plot(xs, ade_vals, "b.-", label="ADE")
        ax.plot(xs, fde_vals, "r.-", label="FDE")
        ax.axhline(np.nanmean(ade_vals), color="b",
                   linestyle="--", alpha=0.5, label=f"mean ADE={np.nanmean(ade_vals):.3f}")
        ax.set_ylabel("metres")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title("Plan ADE / FDE per frame", fontsize=9)

        ax2 = axes[1]
        col_vals = self._frame_cols[:n]
        ax2.bar(xs[:len(col_vals)], col_vals, color="red", alpha=0.6, label="collision")
        ax2.set_ylabel("collision (0/1)")
        ax2.set_xlabel("frame index")
        ax2.set_ylim(-0.05, 1.3)
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Collision flags per frame", fontsize=9)

        fig.tight_layout()
        fname = self.viz_dir / f"{scenario_name}_{self.planner}_metrics.png"
        fig.savefig(fname, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  [viz] saved {fname.name}")

    def log_sensor_sanity(self, frame_data, scenario_name: str,
                           frame_idx: int) -> None:
        """
        Print a one-time sensor sanity report at frame 0 to confirm
        everything is wired up correctly.
        """
        if frame_idx != 0:
            return
        meas  = frame_data.car_data_raw_entry["measurements"]
        lidar = frame_data.car_data_raw_entry["lidar"]
        print()
        print(f"  ── Sensor sanity check (frame 0, scenario {scenario_name}) ──")
        print(f"  GPS (x,y)         : ({meas.get('gps_x',0):.3f}, {meas.get('gps_y',0):.3f}) m")
        print(f"  Compass           : {math.degrees(meas.get('compass',0)):.2f}°")
        print(f"  Speed             : {meas.get('speed',0):.3f} m/s")
        print(f"  LiDAR pts         : {lidar.shape[0]:,}  dtype={lidar.dtype}  shape={lidar.shape}")
        lidar_r = np.linalg.norm(lidar[:, :2], axis=1)
        print(f"  LiDAR range (xy)  : min={lidar_r.min():.2f}  max={lidar_r.max():.2f} m")
        print(f"  LiDAR z           : min={lidar[:,2].min():.2f}  max={lidar[:,2].max():.2f} m")
        print(f"  LiDAR intensity   : min={lidar[:,3].min():.3f}  max={lidar[:,3].max():.3f}")
        for cam in ("front", "left", "right", "rear"):
            img = frame_data.car_data_raw_entry.get(f"rgb_{cam}")
            if img is not None:
                print(f"  Camera {cam:<6}    : shape={img.shape}  "
                      f"mean={img.mean():.1f}  nonzero={np.count_nonzero(img):,}")
        print(f"  Intrinsics (front): {meas.get('camera_front_intrinsics','N/A')}")
        ext = np.array(meas.get('camera_front_extrinsics', np.eye(4)))
        print(f"  Extrinsic (l→cam) : det={np.linalg.det(ext[:3,:3]):.4f} (should be ≈1.0)")
        print(f"  target_point      : {meas.get('target_point','N/A')}")
        print(f"  ─────────────────────────────────────────────────────────────")


# ============================================================
# Per-scenario evaluation
# ============================================================

def _jsonify_debug_value(value: Any) -> Any:
    """Convert numpy/Path/exception-rich values to JSON-safe primitives."""
    if value is None:
        return None
    if np is not None:
        if isinstance(value, np.ndarray):
            return _jsonify_debug_value(value.tolist())
        if isinstance(value, (np.floating, np.integer, np.bool_)):
            return _jsonify_debug_value(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonify_debug_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify_debug_value(v) for v in value]
    if isinstance(value, Exception):
        return f"{type(value).__name__}: {value}"
    return str(value)


class _RawExportFreshnessTracker:
    """Best-effort freshness tracker for Stage 1 raw export artifacts."""

    def __init__(self) -> None:
        self._prev_positions: Optional[np.ndarray] = None
        self._prev_frame_id: Optional[int] = None
        self._prev_token: Optional[str] = None

    def update(
        self,
        *,
        frame_id: int,
        raw_positions: Optional[np.ndarray],
        freshness_token: Optional[Any],
    ) -> Dict[str, Any]:
        if raw_positions is None:
            return {
                "is_fresh_plan": False,
                "reused_from_step": None,
                "freshness_note": "no raw trajectory was exported for this frame",
            }

        arr = np.asarray(raw_positions, dtype=np.float64)
        if arr.size == 0:
            return {
                "is_fresh_plan": False,
                "reused_from_step": None,
                "freshness_note": "empty raw trajectory export",
            }

        token = None if freshness_token is None else str(freshness_token)
        if token is not None:
            is_fresh = (self._prev_token is None) or (token != self._prev_token)
            reused_from_step = None if is_fresh else self._prev_token
            freshness_note = "token-based freshness using adapter-provided source step"
        else:
            same_as_prev = (
                self._prev_positions is not None
                and arr.shape == self._prev_positions.shape
                and np.allclose(arr, self._prev_positions, equal_nan=True)
            )
            is_fresh = not same_as_prev
            reused_from_step = None if is_fresh else self._prev_frame_id
            freshness_note = "content-based freshness using exact raw-position reuse"

        self._prev_positions = arr.copy()
        self._prev_frame_id = int(frame_id)
        self._prev_token = token
        return {
            "is_fresh_plan": bool(is_fresh),
            "reused_from_step": reused_from_step,
            "freshness_note": freshness_note,
        }


def _build_raw_export_record(
    *,
    planner_name: Optional[str],
    scenario_id: str,
    frame_id: int,
    planner_out: Any,
    adapter_source: str,
    traj_source_selected: Optional[str],
    run_step_exception: Optional[Exception],
    freshness_tracker: _RawExportFreshnessTracker,
) -> Dict[str, Any]:
    raw_export = getattr(planner_out, "raw_export", None) if planner_out is not None else None
    raw_positions = getattr(raw_export, "raw_positions", None)
    if raw_positions is None and planner_out is not None and getattr(planner_out, "raw_world_wps", None) is not None:
        raw_positions = np.asarray(planner_out.raw_world_wps, dtype=np.float64)
        source_frame_description = "world-frame pre-resample fallback (native source unavailable)"
        axis_convention_description = "x=world_x, y=world_y"
        is_cumulative_positions = None
        point0_mode = "unknown"
        native_timestamps = None
        native_dt = getattr(planner_out, "native_dt", None)
        adapter_notes = [
            "native raw export metadata was unavailable; fell back to raw_world_wps",
        ]
        freshness_token = None
    else:
        source_frame_description = (
            getattr(raw_export, "source_frame_description", "unknown")
            if raw_export is not None else "unknown"
        )
        axis_convention_description = (
            getattr(raw_export, "axis_convention_description", "unknown")
            if raw_export is not None else "unknown"
        )
        is_cumulative_positions = (
            getattr(raw_export, "is_cumulative_positions", None)
            if raw_export is not None else None
        )
        point0_mode = (
            getattr(raw_export, "point0_mode", "unknown")
            if raw_export is not None else "unknown"
        )
        native_timestamps = (
            getattr(raw_export, "native_timestamps", None)
            if raw_export is not None else None
        )
        native_dt = (
            getattr(raw_export, "native_dt", None)
            if raw_export is not None else getattr(planner_out, "native_dt", None)
        )
        adapter_notes = list(getattr(raw_export, "adapter_debug_notes", []) or [])
        freshness_token = getattr(raw_export, "freshness_token", None) if raw_export is not None else None

    freshness = freshness_tracker.update(
        frame_id=frame_id,
        raw_positions=raw_positions,
        freshness_token=freshness_token,
    )
    adapter_notes.extend([
        f"adapter_source={adapter_source}",
        f"traj_source_selected={traj_source_selected or 'none'}",
        freshness["freshness_note"],
    ])
    if run_step_exception is not None:
        adapter_notes.append(f"run_step_exception={type(run_step_exception).__name__}: {run_step_exception}")

    raw_shape = None
    if raw_positions is not None:
        raw_shape = list(np.asarray(raw_positions, dtype=np.float64).shape)

    return {
        "planner_name": str(planner_name or "generic"),
        "scenario_id": str(scenario_id),
        "frame_id": int(frame_id),
        "raw_positions": raw_positions,
        "raw_shape": raw_shape,
        "source_frame_description": str(source_frame_description),
        "axis_convention_description": str(axis_convention_description),
        "is_cumulative_positions": is_cumulative_positions,
        "point0_mode": (
            point0_mode
            if point0_mode in {"current", "future", "unknown"}
            else "unknown"
        ),
        "native_timestamps": native_timestamps,
        "native_dt": (None if native_dt is None else float(native_dt)),
        "is_fresh_plan": bool(freshness["is_fresh_plan"]),
        "reused_from_step": freshness["reused_from_step"],
        "adapter_debug_notes": adapter_notes,
    }


def _build_canonical_export_record(
    *,
    raw_export_record: Dict[str, Any],
    planner_out: Any,
    gt_traj_all: np.ndarray,
    runtime_dt: float,
) -> Dict[str, Any]:
    from canonical_trajectory import (
        build_canonical_trajectory,
        canonical_to_record,
        compute_gt_canonical_future,
    )

    source_world_positions = (
        getattr(planner_out, "raw_world_wps", None) if planner_out is not None else None
    )
    canonical = build_canonical_trajectory(
        planner_name=str(raw_export_record.get("planner_name", "generic")),
        scenario_id=str(raw_export_record.get("scenario_id", "unknown")),
        frame_id=int(raw_export_record.get("frame_id", 0)),
        raw_export_record=raw_export_record,
        source_world_positions=source_world_positions,
    )
    gt_canonical_positions, gt_valid_mask = compute_gt_canonical_future(
        gt_traj_all,
        frame_idx=int(raw_export_record.get("frame_id", 0)),
        runtime_dt=float(runtime_dt),
    )
    return canonical_to_record(
        canonical,
        raw_export_record=raw_export_record,
        source_world_positions=source_world_positions,
        gt_canonical_positions=gt_canonical_positions,
        gt_valid_mask=gt_valid_mask,
    )


def _build_stage4_scoring_debug_record(
    *,
    planner_name: str,
    scenario_id: str,
    frame_id: int,
    canonical_record: Optional[Dict[str, Any]],
    surround_traj: Optional[np.ndarray],
    surround_mask: Optional[np.ndarray],
    surround_actor_ids: Optional[List[int]],
    ego_pose: Optional[np.ndarray],
    run_step_exception: Optional[Exception],
) -> Dict[str, Any]:
    record = canonical_record or {}
    pred = np.asarray(record.get("canonical_positions", np.zeros((0, 2))), dtype=np.float64)
    gt = np.asarray(record.get("gt_canonical_positions", np.zeros((0, 2))), dtype=np.float64)
    valid_mask = np.asarray(record.get("valid_mask", []), dtype=bool).reshape(-1)
    gt_valid_mask = np.asarray(record.get("gt_valid_mask", []), dtype=bool).reshape(-1)

    inclusion_reason = "other"
    scored = False
    metrics_payload: Optional[Dict[str, Any]] = None

    if run_step_exception is not None:
        inclusion_reason = "run_step_exception"
    elif not record:
        inclusion_reason = "missing_canonical"
    elif not bool(record.get("is_fresh_plan", False)):
        inclusion_reason = "stale"
    elif str(record.get("timestamp_source", "unresolved")) not in {"explicit", "inferred"}:
        inclusion_reason = "unresolved_semantics"
    elif (
        pred.ndim != 2
        or gt.ndim != 2
        or pred.shape[1] != 2
        or gt.shape[1] != 2
        or pred.shape[0] != gt.shape[0]
        or len(valid_mask) != pred.shape[0]
        or len(gt_valid_mask) != gt.shape[0]
    ):
        inclusion_reason = "missing_canonical"
    else:
        score_mask = valid_mask & gt_valid_mask
        if not np.any(score_mask):
            inclusion_reason = "insufficient_horizon"
        else:
            metrics_payload = _compute_plan_metrics_core(
                pred_world_wps=pred,
                gt_future_xy=gt,
                gt_future_mask=score_mask,
                surround_traj=(
                    np.asarray(surround_traj, dtype=np.float64)
                    if surround_traj is not None else np.zeros((0, len(gt), 2), dtype=np.float64)
                ),
                surround_mask=(
                    np.asarray(surround_mask)
                    if surround_mask is not None else np.zeros((0, len(gt)), dtype=np.float64)
                ),
                surround_actor_ids=(
                    list(surround_actor_ids) if surround_actor_ids is not None else None
                ),
                ego_pose=ego_pose,
            )
            if metrics_payload is None:
                inclusion_reason = "missing_canonical"
            else:
                inclusion_reason = "scored"
                scored = True

    n_steps = min(
        len(valid_mask),
        len(gt_valid_mask),
        pred.shape[0] if pred.ndim == 2 else 0,
        gt.shape[0] if gt.ndim == 2 else 0,
    )
    per_step_errors = [None] * n_steps
    if n_steps > 0 and pred.ndim == 2 and gt.ndim == 2:
        mask = valid_mask[:n_steps] & gt_valid_mask[:n_steps]
        dists = np.sqrt(np.sum((pred[:n_steps] - gt[:n_steps]) ** 2, axis=1))
        for idx in range(n_steps):
            if bool(mask[idx]):
                per_step_errors[idx] = float(dists[idx])

    return {
        "planner_name": str(planner_name or "generic"),
        "scenario_id": str(scenario_id),
        "frame_id": int(frame_id),
        "scoring_path": "canonical_fresh_only",
        "scored": bool(scored),
        "inclusion_reason": inclusion_reason,
        "timestamp_source": record.get("timestamp_source"),
        "is_fresh_plan": bool(record.get("is_fresh_plan", False)),
        "canonical_timestamps": record.get("canonical_timestamps"),
        "predicted_canonical_points": record.get("canonical_positions"),
        "gt_canonical_points": record.get("gt_canonical_positions"),
        "valid_mask": [bool(v) for v in valid_mask.tolist()] if len(valid_mask) else [],
        "gt_valid_mask": [bool(v) for v in gt_valid_mask.tolist()] if len(gt_valid_mask) else [],
        "ade_breakdown_m": per_step_errors,
        "plan_ade": (
            float(metrics_payload["plan_ade"])
            if metrics_payload is not None else float("nan")
        ),
        "plan_fde": (
            float(metrics_payload["plan_fde"])
            if metrics_payload is not None else float("nan")
        ),
        "collision": (
            int(metrics_payload["collision"])
            if metrics_payload is not None else 0
        ),
        "collision_debug": (
            metrics_payload.get("collision_debug")
            if isinstance(metrics_payload, dict)
            else None
        ),
        "provenance_debug_notes": list(record.get("provenance_debug_notes", []) or []),
        "run_step_exception": (
            f"{type(run_step_exception).__name__}: {run_step_exception}"
            if run_step_exception is not None else None
        ),
    }


def _summarize_input_payload(value: Any) -> Dict[str, Any]:
    """Compact JSON-safe summary of one run_step input payload."""
    if value is None:
        return {"type": "none"}
    if np is not None and isinstance(value, np.ndarray):
        arr = np.asarray(value)
        summary: Dict[str, Any] = {
            "type": "ndarray",
            "shape": [int(v) for v in arr.shape],
            "dtype": str(arr.dtype),
        }
        if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
            finite = np.isfinite(arr)
            summary["finite_fraction"] = float(np.mean(finite))
            if np.any(finite):
                vals = arr[finite]
                summary["min"] = float(np.min(vals))
                summary["max"] = float(np.max(vals))
                summary["mean"] = float(np.mean(vals))
        return summary
    if isinstance(value, dict):
        keys = sorted(str(k) for k in value.keys())
        preview: Dict[str, Any] = {}
        for k, v in value.items():
            ks = str(k)
            if isinstance(v, (str, int, bool)):
                preview[ks] = v
                continue
            if isinstance(v, float):
                preview[ks] = float(v) if math.isfinite(v) else None
                continue
            if np is not None and isinstance(v, (np.floating, np.integer, np.bool_)):
                try:
                    vv = v.item()
                    preview[ks] = float(vv) if isinstance(vv, float) else vv
                except Exception:
                    preview[ks] = str(type(v).__name__)
                continue
            if np is not None and isinstance(v, np.ndarray):
                preview[ks] = _summarize_input_payload(v)
                continue
            if isinstance(v, dict):
                preview[ks] = {
                    "type": "dict",
                    "keys": sorted(str(x) for x in v.keys()),
                }
                continue
            preview[ks] = str(type(v).__name__)
        return {
            "type": "dict",
            "keys": keys,
            "preview": preview,
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "length": len(value),
        }
    if isinstance(value, (str, int, bool)):
        return {"type": type(value).__name__, "value": value}
    if isinstance(value, float):
        return {"type": "float", "value": (float(value) if math.isfinite(value) else None)}
    return {"type": str(type(value).__name__)}


def _summarize_input_data(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Summarise planner-facing input_data keys and payload shapes."""
    out: Dict[str, Any] = {}
    for key in sorted(input_data.keys(), key=lambda x: str(x)):
        val = input_data[key]
        rec: Dict[str, Any] = {}
        if isinstance(val, (list, tuple)) and len(val) == 2:
            ts = val[0]
            try:
                rec["frame_token"] = int(ts)
            except Exception:
                rec["frame_token"] = ts
            rec["payload"] = _summarize_input_payload(val[1])
        else:
            rec["payload"] = _summarize_input_payload(val)
        out[str(key)] = rec
    return out


def _summarize_frame_sensors(frame_data: Any) -> Dict[str, Any]:
    """Compute compact per-frame sensor statistics for debug JSONL."""
    stats: Dict[str, Any] = {"cameras": {}, "lidar": {}}
    if frame_data is None:
        return stats
    entry = getattr(frame_data, "car_data_raw_entry", None) or {}

    lidar = entry.get("lidar")
    if lidar is not None:
        arr = np.asarray(lidar)
        lidar_stats: Dict[str, Any] = {
            "shape": [int(v) for v in arr.shape],
            "dtype": str(arr.dtype),
            "num_points": int(arr.shape[0]) if arr.ndim >= 1 else 0,
        }
        if arr.ndim == 2 and arr.shape[1] >= 3 and arr.size > 0:
            xyz = arr[:, :3]
            finite_xyz = np.isfinite(xyz).all(axis=1)
            lidar_stats["finite_xyz_fraction"] = float(np.mean(finite_xyz)) if len(finite_xyz) else 0.0
            if np.any(finite_xyz):
                xyzf = xyz[finite_xyz]
                lidar_stats["xyz_min"] = [float(v) for v in np.min(xyzf, axis=0)]
                lidar_stats["xyz_max"] = [float(v) for v in np.max(xyzf, axis=0)]
                xy_norm = np.linalg.norm(xyzf[:, :2], axis=1)
                if len(xy_norm):
                    lidar_stats["xy_range_min"] = float(np.min(xy_norm))
                    lidar_stats["xy_range_max"] = float(np.max(xy_norm))
        if arr.ndim == 2 and arr.shape[1] >= 4 and arr.size > 0:
            intensity = arr[:, 3]
            finite_i = np.isfinite(intensity)
            if np.any(finite_i):
                iv = intensity[finite_i]
                lidar_stats["intensity_min"] = float(np.min(iv))
                lidar_stats["intensity_max"] = float(np.max(iv))
                lidar_stats["intensity_mean"] = float(np.mean(iv))
        stats["lidar"] = lidar_stats

    for direction in ("front", "left", "right", "rear"):
        img = entry.get(f"rgb_{direction}")
        if img is None:
            stats["cameras"][direction] = {"available": False}
            continue
        arr = np.asarray(img)
        cam_stats: Dict[str, Any] = {
            "available": True,
            "shape": [int(v) for v in arr.shape],
            "dtype": str(arr.dtype),
        }
        if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
            finite = np.isfinite(arr)
            cam_stats["finite_fraction"] = float(np.mean(finite))
            cam_stats["nonzero_fraction"] = float(np.count_nonzero(arr) / arr.size)
            if np.any(finite):
                vals = arr[finite]
                cam_stats["min"] = float(np.min(vals))
                cam_stats["max"] = float(np.max(vals))
                cam_stats["mean"] = float(np.mean(vals))
                cam_stats["std"] = float(np.std(vals))
        stats["cameras"][direction] = cam_stats

    return stats


def _find_frame_camera_image(base: Path, cam_key: str) -> Optional[str]:
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = Path(f"{base}_{cam_key}{ext}")
        if p.exists():
            return str(p)
    return None


def _build_frame_input_files_manifest(
    *,
    scenario_dir: Path,
    actor_id: int,
    frame_idx: int,
    camera_assignment: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    """Return source file paths used to build one open-loop frame."""
    base = scenario_dir / str(actor_id) / f"{frame_idx:06d}"
    camera_images: Dict[str, Optional[str]] = {
        "front": None,
        "left": None,
        "right": None,
        "rear": None,
    }
    camera_keys: Dict[str, str] = {}

    if isinstance(camera_assignment, dict):
        for cam_key, direction in camera_assignment.items():
            d = str(direction)
            if d not in camera_images:
                continue
            camera_keys[d] = str(cam_key)
            camera_images[d] = _find_frame_camera_image(base, str(cam_key))

    return {
        "scenario_dir": str(scenario_dir),
        "actor_id": int(actor_id),
        "frame_base": str(base),
        "yaml": str(base.with_suffix(".yaml")),
        "lidar_bin": str(base.with_suffix(".bin")),
        "camera_keys": camera_keys,
        "camera_images": camera_images,
    }


def _build_measurements_debug_payload(
    measurements: Dict[str, Any],
    *,
    speed_mps: float,
) -> Dict[str, Any]:
    """Persist high-value measurement fields used by planners and projection code."""
    payload: Dict[str, Any] = {
        "speed_mps": float(speed_mps),
    }
    keys = [
        "speed",
        "gps_x",
        "gps_y",
        "x",
        "y",
        "theta",
        "compass",
        "command",
        "target_point",
        "lidar_pose_x",
        "lidar_pose_y",
        "lidar_pose_z",
        "lidar_pose_gps_x",
        "lidar_pose_gps_y",
        "camera_front_intrinsics",
        "camera_front_extrinsics",
        "camera_left_intrinsics",
        "camera_left_extrinsics",
        "camera_right_intrinsics",
        "camera_right_extrinsics",
        "camera_rear_intrinsics",
        "camera_rear_extrinsics",
    ]
    for k in keys:
        if isinstance(measurements, dict) and k in measurements:
            payload[k] = measurements.get(k)
    return payload


_ROADOPTION_NAME_BY_VALUE: Dict[int, str] = {
    1: "LEFT",
    2: "RIGHT",
    3: "STRAIGHT",
    4: "LANEFOLLOW",
    5: "CHANGELANELEFT",
    6: "CHANGELANERIGHT",
}
_VAD_BRANCH_NAME_BY_INDEX: List[str] = [
    "LEFT",
    "RIGHT",
    "STRAIGHT",
    "LANEFOLLOW",
    "CHANGELANELEFT",
    "CHANGELANERIGHT",
]


def _roadoption_name(value: Optional[int]) -> Optional[str]:
    if value is None:
        return None
    return _ROADOPTION_NAME_BY_VALUE.get(int(value), f"UNKNOWN_{int(value)}")


def _world_to_carla_local(
    world_xy: np.ndarray,
    ego_world_xy: np.ndarray,
    compass_rad: float,
) -> np.ndarray:
    theta = float(compass_rad) + math.pi / 2.0
    rot = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=np.float64,
    )
    return (rot.T @ (world_xy[:, :2] - ego_world_xy[:2]).T).T


def _carla_local_to_world(
    local_xy: np.ndarray,
    ego_world_xy: np.ndarray,
    compass_rad: float,
) -> np.ndarray:
    theta = float(compass_rad) + math.pi / 2.0
    rot = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=np.float64,
    )
    return np.asarray(ego_world_xy, dtype=np.float64)[:2] + (rot @ local_xy[:, :2].T).T


def _path_length_m(xy: np.ndarray) -> Optional[float]:
    arr = np.asarray(xy, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) < 2:
        return None
    seg = arr[1:, :2] - arr[:-1, :2]
    if not np.isfinite(seg).all():
        return None
    return float(np.sum(np.linalg.norm(seg, axis=1)))


def _heading_change_rad(xy: np.ndarray) -> Optional[float]:
    arr = np.asarray(xy, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) < 3:
        return None
    seg = arr[1:, :2] - arr[:-1, :2]
    norms = np.linalg.norm(seg, axis=1)
    valid = np.where(norms > 1e-6)[0]
    if len(valid) < 2:
        return None
    h0 = math.atan2(seg[valid[0], 1], seg[valid[0], 0])
    h1 = math.atan2(seg[valid[-1], 1], seg[valid[-1], 0])
    dh = (h1 - h0 + math.pi) % (2.0 * math.pi) - math.pi
    return float(dh)


def _to_list_or_none(values: np.ndarray, mask: np.ndarray) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    n = min(len(values), len(mask))
    for i in range(n):
        if bool(mask[i]) and math.isfinite(float(values[i])):
            out.append(float(values[i]))
        else:
            out.append(None)
    return out


def _trajectory_error_components_local(
    pred_world_xy: np.ndarray,
    gt_world_xy: np.ndarray,
    *,
    ego_world_xy: np.ndarray,
    compass_rad: float,
    valid_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    pred = np.asarray(pred_world_xy, dtype=np.float64)
    gt = np.asarray(gt_world_xy, dtype=np.float64)
    n = min(len(pred), len(gt))
    if n <= 0:
        return {
            "longitudinal_error_m_by_step": [],
            "lateral_error_m_by_step": [],
            "abs_longitudinal_error_m_by_step": [],
            "abs_lateral_error_m_by_step": [],
            "mean_abs_longitudinal_error_m": None,
            "mean_abs_lateral_error_m": None,
        }
    pred = pred[:n, :2]
    gt = gt[:n, :2]
    pred_local = _world_to_carla_local(pred, np.asarray(ego_world_xy, dtype=np.float64), float(compass_rad))
    gt_local = _world_to_carla_local(gt, np.asarray(ego_world_xy, dtype=np.float64), float(compass_rad))
    err_local = pred_local - gt_local

    if valid_mask is None:
        mask = np.ones(n, dtype=bool)
    else:
        mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if len(mask) < n:
            padded = np.zeros(n, dtype=bool)
            padded[: len(mask)] = mask
            mask = padded
        else:
            mask = mask[:n]

    lon = err_local[:, 0]
    lat = err_local[:, 1]
    abs_lon = np.abs(lon)
    abs_lat = np.abs(lat)
    return {
        "longitudinal_error_m_by_step": _to_list_or_none(lon, mask),
        "lateral_error_m_by_step": _to_list_or_none(lat, mask),
        "abs_longitudinal_error_m_by_step": _to_list_or_none(abs_lon, mask),
        "abs_lateral_error_m_by_step": _to_list_or_none(abs_lat, mask),
        "mean_abs_longitudinal_error_m": (
            float(np.mean(abs_lon[mask])) if np.any(mask) else None
        ),
        "mean_abs_lateral_error_m": (
            float(np.mean(abs_lat[mask])) if np.any(mask) else None
        ),
    }


def _select_vad_branch_world_from_index(
    *,
    planner_debug: Optional[Dict[str, Any]],
    branch_index: Optional[int],
    ego_world_xy: np.ndarray,
    compass_rad: Optional[float],
) -> Optional[Dict[str, Any]]:
    """Convert a VAD branch index from adapter debug payload into world waypoints."""
    if not isinstance(planner_debug, dict):
        return None
    if compass_rad is None or branch_index is None:
        return None
    try:
        idx = int(branch_index)
    except Exception:
        return None

    branches_raw = planner_debug.get("candidate_local_wps_by_command")
    if branches_raw is None:
        return None
    try:
        branches = np.asarray(branches_raw, dtype=np.float64)
    except Exception:
        return None
    if branches.ndim != 3 or branches.shape[-1] < 2 or branches.shape[0] <= 0:
        return None
    if idx < 0 or idx >= int(branches.shape[0]):
        return None

    vad_local = np.asarray(branches[idx, :, :2], dtype=np.float64)
    if vad_local.ndim != 2 or vad_local.shape[0] <= 0:
        return None
    if not np.isfinite(vad_local).all():
        return None

    # VAD local frame: x=right, y=forward -> CARLA local: x=forward, y=left.
    carla_local = np.column_stack([vad_local[:, 1], -vad_local[:, 0]])
    world_wps = _carla_local_to_world(
        carla_local,
        np.asarray(ego_world_xy, dtype=np.float64),
        float(compass_rad),
    )
    if not np.isfinite(world_wps).all():
        return None
    return {
        "branch_index": int(idx),
        "local_wps_vad_frame": vad_local,
        "world_wps": world_wps,
    }


def _compute_vad_branch_diagnostics(
    *,
    planner_name: Optional[str],
    planner_debug: Optional[Dict[str, Any]],
    measurements: Optional[Dict[str, Any]],
    ego_world_xy: np.ndarray,
    compass_rad: Optional[float],
    gt_future_world: Optional[np.ndarray],
    gt_valid_mask: Optional[np.ndarray],
) -> Optional[Dict[str, Any]]:
    """Best-effort VAD branch-selection diagnostics for per-frame debug JSONL."""
    planner_tag = str(planner_name or "").lower()
    if planner_tag != "vad" and "vad" not in planner_tag:
        return None
    if not isinstance(planner_debug, dict):
        return None

    branches_raw = planner_debug.get("candidate_local_wps_by_command")
    if branches_raw is None:
        return None
    try:
        branches = np.asarray(branches_raw, dtype=np.float64)
    except Exception:
        return None
    if branches.ndim != 3 or branches.shape[-1] < 2 or branches.shape[0] <= 0:
        return None

    selected_idx: Optional[int]
    try:
        selected_idx = int(planner_debug.get("selected_command_index"))
    except Exception:
        selected_idx = None

    route_cmd_value: Optional[int]
    try:
        route_cmd_value = int((measurements or {}).get("command"))
    except Exception:
        route_cmd_value = None

    selected_cmd_raw_value: Optional[int]
    try:
        selected_cmd_raw_value = int(planner_debug.get("selected_command_raw_value"))
    except Exception:
        selected_cmd_raw_value = None
    selected_cmd_fallback_to_lanefollow: Optional[bool]
    raw_fallback_val = planner_debug.get("selected_command_fallback_to_lanefollow")
    if isinstance(raw_fallback_val, bool):
        selected_cmd_fallback_to_lanefollow = bool(raw_fallback_val)
    else:
        selected_cmd_fallback_to_lanefollow = None

    selected_cmd_value = (
        int(selected_idx + 1)
        if selected_idx is not None and 0 <= selected_idx < len(_VAD_BRANCH_NAME_BY_INDEX)
        else None
    )

    summary: Dict[str, Any] = {
        "route_command_value": route_cmd_value,
        "route_command_name": _roadoption_name(route_cmd_value),
        "selected_branch_index": selected_idx,
        "selected_command_value": selected_cmd_value,
        "selected_command_name": _roadoption_name(selected_cmd_value),
        "selected_command_raw_value": selected_cmd_raw_value,
        "selected_command_fallback_to_lanefollow": selected_cmd_fallback_to_lanefollow,
        "selected_matches_route_command": (
            bool(selected_cmd_value == route_cmd_value)
            if selected_cmd_value is not None and route_cmd_value is not None
            else None
        ),
    }

    if compass_rad is None or gt_future_world is None:
        return summary

    gt_world = np.asarray(gt_future_world, dtype=np.float64).reshape(-1, 2)
    if len(gt_world) == 0:
        return summary

    if gt_valid_mask is None:
        valid_mask = np.ones(len(gt_world), dtype=bool)
    else:
        valid_mask = np.asarray(gt_valid_mask, dtype=bool).reshape(-1)
        if len(valid_mask) < len(gt_world):
            padded = np.zeros(len(gt_world), dtype=bool)
            padded[: len(valid_mask)] = valid_mask
            valid_mask = padded
        else:
            valid_mask = valid_mask[: len(gt_world)]

    gt_local_carla = _world_to_carla_local(gt_world, np.asarray(ego_world_xy, dtype=np.float64), float(compass_rad))
    gt_local_finite = np.isfinite(gt_local_carla).all(axis=1)

    # Branches from VAD metadata use (right=+x, forward=+y). Convert to
    # CARLA-local (forward=+x, left=+y) for direct GT-local comparison.
    branches_local_carla = np.stack([branches[:, :, 1], -branches[:, :, 0]], axis=-1)
    ade_by_branch: List[Optional[float]] = []
    fde_by_branch: List[Optional[float]] = []
    world_ade_by_branch: List[Optional[float]] = []
    world_fde_by_branch: List[Optional[float]] = []
    for branch in branches_local_carla:
        n = min(len(branch), len(gt_local_carla))
        if n <= 0:
            ade_by_branch.append(None)
            fde_by_branch.append(None)
            world_ade_by_branch.append(None)
            world_fde_by_branch.append(None)
            continue
        branch_n = branch[:n, :2]
        gt_n = gt_local_carla[:n, :2]
        valid = (
            valid_mask[:n]
            & gt_local_finite[:n]
            & np.isfinite(branch_n).all(axis=1)
        )
        if not np.any(valid):
            ade_by_branch.append(None)
            fde_by_branch.append(None)
            world_ade_by_branch.append(None)
            world_fde_by_branch.append(None)
            continue
        d_local = np.linalg.norm(branch_n[valid] - gt_n[valid], axis=1)
        ade_by_branch.append(float(np.mean(d_local)))
        fde_by_branch.append(float(d_local[-1]))

        branch_world = _carla_local_to_world(
            branch_n,
            np.asarray(ego_world_xy, dtype=np.float64),
            float(compass_rad),
        )
        gt_world_n = gt_world[:n, :2]
        valid_world = (
            valid
            & np.isfinite(branch_world).all(axis=1)
            & np.isfinite(gt_world_n).all(axis=1)
        )
        if not np.any(valid_world):
            world_ade_by_branch.append(None)
            world_fde_by_branch.append(None)
        else:
            d_world = np.linalg.norm(
                branch_world[valid_world] - gt_world_n[valid_world],
                axis=1,
            )
            world_ade_by_branch.append(float(np.mean(d_world)))
            world_fde_by_branch.append(float(d_world[-1]))

    finite = [
        (idx, val) for idx, val in enumerate(ade_by_branch)
        if val is not None and math.isfinite(float(val))
    ]
    finite_world = [
        (idx, val) for idx, val in enumerate(world_ade_by_branch)
        if val is not None and math.isfinite(float(val))
    ]
    if not finite:
        summary["branch_local_ade_m_by_index"] = ade_by_branch
        summary["branch_local_fde_m_by_index"] = fde_by_branch
        summary["branch_world_ade_m_by_index"] = world_ade_by_branch
        summary["branch_world_fde_m_by_index"] = world_fde_by_branch
        return summary

    best_idx, best_ade = min(finite, key=lambda iv: float(iv[1]))
    best_world_idx: Optional[int] = None
    best_world_ade: Optional[float] = None
    if finite_world:
        best_world_idx, best_world_ade = min(finite_world, key=lambda iv: float(iv[1]))
    selected_ade: Optional[float] = None
    selected_fde: Optional[float] = None
    selected_world_ade: Optional[float] = None
    selected_world_fde: Optional[float] = None
    if selected_idx is not None and 0 <= selected_idx < len(ade_by_branch):
        selected_ade = ade_by_branch[selected_idx]
        selected_fde = fde_by_branch[selected_idx]
        selected_world_ade = world_ade_by_branch[selected_idx]
        selected_world_fde = world_fde_by_branch[selected_idx]

    summary.update({
        "branch_local_ade_m_by_index": ade_by_branch,
        "branch_local_fde_m_by_index": fde_by_branch,
        "branch_world_ade_m_by_index": world_ade_by_branch,
        "branch_world_fde_m_by_index": world_fde_by_branch,
        "best_branch_index_by_local_ade": int(best_idx),
        "best_command_value_by_local_ade": int(best_idx + 1),
        "best_command_name_by_local_ade": _roadoption_name(int(best_idx + 1)),
        "best_branch_local_ade_m": float(best_ade),
        "best_branch_local_fde_m": (
            float(fde_by_branch[best_idx])
            if fde_by_branch[best_idx] is not None and math.isfinite(float(fde_by_branch[best_idx]))
            else None
        ),
        "best_branch_index_by_world_ade": (
            int(best_world_idx) if best_world_idx is not None else None
        ),
        "best_command_value_by_world_ade": (
            int(best_world_idx + 1) if best_world_idx is not None else None
        ),
        "best_command_name_by_world_ade": (
            _roadoption_name(int(best_world_idx + 1))
            if best_world_idx is not None else None
        ),
        "best_branch_world_ade_m": (
            float(best_world_ade)
            if best_world_ade is not None and math.isfinite(float(best_world_ade))
            else None
        ),
        "best_branch_world_fde_m": (
            float(world_fde_by_branch[best_world_idx])
            if best_world_idx is not None
            and world_fde_by_branch[best_world_idx] is not None
            and math.isfinite(float(world_fde_by_branch[best_world_idx]))
            else None
        ),
        "selected_branch_local_ade_m": (
            float(selected_ade)
            if selected_ade is not None and math.isfinite(float(selected_ade))
            else None
        ),
        "selected_branch_local_fde_m": (
            float(selected_fde)
            if selected_fde is not None and math.isfinite(float(selected_fde))
            else None
        ),
        "selected_branch_world_ade_m": (
            float(selected_world_ade)
            if selected_world_ade is not None and math.isfinite(float(selected_world_ade))
            else None
        ),
        "selected_branch_world_fde_m": (
            float(selected_world_fde)
            if selected_world_fde is not None and math.isfinite(float(selected_world_fde))
            else None
        ),
        "selected_minus_best_local_ade_m": (
            float(selected_ade - best_ade)
            if selected_ade is not None and math.isfinite(float(selected_ade))
            else None
        ),
        "selected_minus_best_world_ade_m": (
            float(selected_world_ade - best_world_ade)
            if selected_world_ade is not None
            and best_world_ade is not None
            and math.isfinite(float(selected_world_ade))
            and math.isfinite(float(best_world_ade))
            else None
        ),
        "selected_is_best_by_local_ade": (
            bool(float(selected_ade) <= float(best_ade) + 1e-6)
            if selected_ade is not None and math.isfinite(float(selected_ade))
            else None
        ),
        "selected_is_best_by_world_ade": (
            bool(float(selected_world_ade) <= float(best_world_ade) + 1e-6)
            if selected_world_ade is not None
            and best_world_ade is not None
            and math.isfinite(float(selected_world_ade))
            and math.isfinite(float(best_world_ade))
            else None
        ),
    })
    return summary


def evaluate_scenario(agent,
                      scenario: V2XPnPScenario,
                      hero_vehicles: Dict[int, MockVehicle],
                      eval_actor_id: Optional[int] = None,
                      verbose: bool = False,
                      vis_logger: Optional[_VisLogger] = None,
                      max_frames: Optional[int] = None,
                      traj_source: str = "auto",
                      trajectory_extrapolation: bool = True,
                      town_id: Optional[str] = None,
                      planner_name: Optional[str] = None,
                      frame_debug_dir: Optional[Path] = None,
                      raw_export_dir: Optional[Path] = None,
                      canonical_export_dir: Optional[Path] = None,
                      scoring_debug_dir: Optional[Path] = None,
                      native_horizon: bool = True,
                      vad_oracle_branch_by_local_ade: bool = True,
                      skip_stale_frames: bool = True,
                      scoring_path: str = "canonical_fresh_only") -> Dict[str, Any]:
    """
    Run one scenario through the agent and return per-frame and aggregate metrics.

    For each frame:
      1. Dataset YAML + images + LiDAR are loaded from disk (no CARLA server).
      2. MockVehicle pose is updated so agent.get_transform() is correct.
      3. agent.run_step(input_data, timestamp) is called identically to CARLA.
      4. Future trajectory source is selected by ``traj_source``:
         - ``control_rollout``: always rollout from returned control.
         - ``planner_waypoints``: adapter-only planner waypoints.
         - ``auto``: adapter first, fallback to control rollout.
      5. Plan ADE / FDE / Collision Rate via
         opencood/utils/eval_utils.py formula.
    """
    from planner_trajectory_adapter import (
        extract_planner_trajectory,
        StaleControlDetector,
        EVAL_HORIZON_STEPS,
        set_trajectory_extrapolation,
    )
    _log = logging.getLogger("openloop_eval")
    legacy_accum = MetricAccumulator()
    canonical_accum = CanonicalMetricAccumulator()
    debug_rollout_accum = MetricAccumulator()   # debug-only: control-rollout metrics
    branch_selected_accum = MetricAccumulator()  # command-selected branch (non-oracle)
    branch_oracle_accum = MetricAccumulator()    # best branch by GT-local ADE (oracle)
    branch_selected_scored_accum = MetricAccumulator()  # canonical-scored subset only
    branch_oracle_scored_accum = MetricAccumulator()    # canonical-scored subset only
    stale_detector = StaleControlDetector()
    traj_source_counts: Dict[str, int] = {}     # provenance counter
    adapter_source_counts: Dict[str, int] = {}  # adapter raw source before fallback
    traj_mode = str(traj_source or "auto").strip().lower()
    if traj_mode not in {"control_rollout", "planner_waypoints", "auto"}:
        traj_mode = "auto"
    set_trajectory_extrapolation(bool(trajectory_extrapolation))
    planner_waypoint_miss_count = 0
    sanity_checks = {
        "pred_eq_gt_ade": None,
        "gt_shifted_ade": None,
    }

    def _compute_ade_only(pred_xy: np.ndarray, gt_xy: np.ndarray, mask: np.ndarray) -> float:
        pred_xy = np.asarray(pred_xy, dtype=np.float64)
        gt_xy = np.asarray(gt_xy, dtype=np.float64)
        mask = np.asarray(mask) > 0
        T = min(len(pred_xy), len(gt_xy), len(mask))
        if T <= 0:
            return float("nan")
        pred_xy = pred_xy[:T, :2]
        gt_xy = gt_xy[:T, :2]
        valid = mask[:T]
        if not np.any(valid):
            return float("nan")
        d = np.sqrt(np.sum((pred_xy - gt_xy) ** 2, axis=1))
        return float(np.mean(d[valid]))
    per_frame  = []

    actor_ids  = scenario.actor_ids
    if not actor_ids:
        return {
            "metrics": legacy_accum.summary(),
            "legacy_metrics": legacy_accum.summary(),
            "coverage_stats": canonical_accum.coverage_summary(),
            "per_frame": [],
        }

    primary_actor_id = actor_ids[0] if eval_actor_id is None else int(eval_actor_id)
    if primary_actor_id not in actor_ids:
        raise ValueError(
            f"Actor id {primary_actor_id} not found in scenario {scenario.scenario_dir.name}. "
            f"Available actors: {actor_ids}"
        )
    n_frames         = scenario.num_frames(primary_actor_id)
    sname            = str(scenario.scenario_dir.name)
    sname_eval       = f"{sname}_ego{primary_actor_id}"
    frame_debug_path: Optional[Path] = None
    frame_debug_fh = None
    raw_export_path: Optional[Path] = None
    raw_export_fh = None
    canonical_export_path: Optional[Path] = None
    canonical_export_fh = None
    scoring_debug_path: Optional[Path] = None
    scoring_debug_fh = None
    if frame_debug_dir is not None:
        frame_debug_dir.mkdir(parents=True, exist_ok=True)
        frame_debug_path = frame_debug_dir / f"{sanitize_run_id(sname_eval)}_frames.jsonl"
        frame_debug_fh = frame_debug_path.open("w", encoding="utf-8")
    if raw_export_dir is not None:
        raw_export_dir.mkdir(parents=True, exist_ok=True)
        raw_export_path = raw_export_dir / f"{sanitize_run_id(sname_eval)}_raw.jsonl"
        raw_export_fh = raw_export_path.open("w", encoding="utf-8")
    if canonical_export_dir is not None:
        canonical_export_dir.mkdir(parents=True, exist_ok=True)
        canonical_export_path = (
            canonical_export_dir / f"{sanitize_run_id(sname_eval)}_canonical.jsonl"
        )
        canonical_export_fh = canonical_export_path.open("w", encoding="utf-8")
    if scoring_debug_dir is not None:
        scoring_debug_dir.mkdir(parents=True, exist_ok=True)
        scoring_debug_path = (
            scoring_debug_dir / f"{sanitize_run_id(sname_eval)}_scoring.jsonl"
        )
        scoring_debug_fh = scoring_debug_path.open("w", encoding="utf-8")
    raw_export_freshness = _RawExportFreshnessTracker()

    runtime_dt = (1.0 / float(scenario.fps)) if scenario.fps else 0.1
    _apply_agent_runtime_context(agent, scenario=scenario, town_id=town_id)
    _set_agent_global_plan(agent, scenario, [primary_actor_id])
    if callable(configure_openloop_world):
        try:
            configure_openloop_world(runtime_dt)
        except Exception:
            pass

    sensor_specs = _safe_sensor_specs(agent)
    if verbose and sensor_specs:
        print(f"  [openloop] using {len(sensor_specs)} sensor specs from agent.sensors()")
    adapter_name = planner_name or "generic"
    if traj_mode == "control_rollout":
        print(
            f"  [openloop] trajectory mode=control_rollout "
            f"(planner adapter='{adapter_name}' collected for debug only)."
        )
    elif traj_mode == "planner_waypoints":
        print(
            f"  [openloop] trajectory mode=planner_waypoints "
            f"(strict adapter-only, planner='{adapter_name}')."
        )
    else:
        print(
            f"  [openloop] trajectory mode=auto "
            f"(planner adapter='{adapter_name}', fallback=control_rollout)."
        )
    print(
        f"  [openloop] trajectory extrapolation="
        f"{'enabled' if trajectory_extrapolation else 'disabled'}"
    )
    print(
        f"  [openloop] native_horizon={'enabled' if native_horizon else 'disabled (0.5s grid)'}  "
        f"skip_stale_frames={'enabled' if skip_stale_frames else 'disabled'}"
    )
    try:
        from canonical_trajectory import CANONICAL_TIMESTAMPS

        if len(CANONICAL_TIMESTAMPS) > 0:
            print(
                "  [openloop] canonical_horizon="
                f"{float(CANONICAL_TIMESTAMPS[-1]):.1f}s "
                f"({len(CANONICAL_TIMESTAMPS)} step(s) at 0.5 s)"
            )
    except Exception:
        pass
    print(
        f"  [openloop] vad_oracle_branch_by_local_ade="
        f"{'enabled' if vad_oracle_branch_by_local_ade else 'disabled'}"
    )
    print(f"  [openloop] scoring_path={scoring_path}")

    gt_traj_all = scenario.get_gt_trajectory(primary_actor_id)  # [N_frames, 2]

    # Optionally limit number of frames evaluated (for fast sweeps)
    effective_n_frames = min(n_frames, max_frames) if max_frames is not None else n_frames

    if vis_logger is not None:
        vis_logger.begin_scenario(sname_eval, actor_ids, primary_actor_id,
                                  effective_n_frames, gt_traj_all, fps=scenario.fps)

    last_run_exc_signature: Optional[str] = None
    for frame_idx in range(effective_n_frames):
        t0 = time.perf_counter()

        next_wps = (gt_traj_all[frame_idx + 1:frame_idx + MAX_GT_STEPS + 1]
                    if frame_idx + 1 < n_frames else None)

        frame_data = scenario.load_frame(primary_actor_id, frame_idx,
                                          next_waypoints=next_wps)

        timestamp = frame_idx * runtime_dt
        meta = {}
        prev_meta = {}
        cam_assign: Optional[Dict[str, str]] = None
        try:
            import yaml as _yaml
            _yp = (scenario.scenario_dir / str(primary_actor_id)
                   / f"{frame_idx:06d}.yaml")
            with open(_yp) as _fh:
                meta = _yaml.safe_load(_fh) or {}
            if frame_idx > 0:
                _prev_yp = (scenario.scenario_dir / str(primary_actor_id)
                            / f"{frame_idx - 1:06d}.yaml")
                with open(_prev_yp) as _fh:
                    prev_meta = _yaml.safe_load(_fh) or {}
            _assign_camera_directions = _ds_mod._assign_camera_directions
            cam_assign = _assign_camera_directions(meta)
        except Exception:
            meta = {}
            prev_meta = {}
            cam_assign = None

        # Update synthetic world actors from recorded frame metadata so agent
        # calls to CarlaDataProvider.get_world().get_actors() stay open-loop.
        try:
            _MockCarlaDataProvider.set_surrounding_from_v2xpnp(
                meta.get("vehicles", {}),
                delta_seconds=runtime_dt,
                previous_vehicles_dict=prev_meta.get("vehicles", {}),
                timestamp_s=timestamp,
            )
        except TypeError:
            _MockCarlaDataProvider.set_surrounding_from_v2xpnp(
                meta.get("vehicles", {})
            )

        if vis_logger is not None:
            vis_logger.log_sensor_sanity(frame_data, sname_eval, frame_idx)

        # ---- Update MockVehicle pose ----
        # Dataset format: [x, y, z, roll_deg, yaw_deg, pitch_deg]
        pose = frame_data.world_pose
        measurements = frame_data.car_data_raw_entry.get("measurements", {})
        try:
            speed_mps = float(measurements.get("speed", 0.0))
        except Exception:
            speed_mps = 0.0
        if not np.isfinite(speed_mps):
            speed_mps = 0.0
        hero_update_kwargs = dict(
            x=pose[0],
            y=pose[1],
            z=pose[2],
            roll_rad=math.radians(pose[3]),
            pitch_rad=math.radians(pose[5]),
            yaw_rad=math.radians(pose[4]),
            timestamp_s=timestamp,
            speed_mps=speed_mps,
            delta_seconds=runtime_dt,
        )
        try:
            hero_vehicles[0].update_runtime_state(**hero_update_kwargs)
        except AttributeError:
            try:
                hero_vehicles[0].update_pose(**hero_update_kwargs)
            except TypeError:
                hero_vehicles[0].update_pose(
                    pose[0], pose[1], pose[2],
                    math.radians(pose[3]),
                    math.radians(pose[5]),
                    math.radians(pose[4]),
                )
        except TypeError:
            hero_vehicles[0].update_pose(
                pose[0], pose[1], pose[2],
                math.radians(pose[3]),
                math.radians(pose[5]),
                math.radians(pose[4]),
            )

        if callable(advance_openloop_runtime):
            try:
                advance_openloop_runtime(
                    frame_idx=frame_idx,
                    timestamp_s=timestamp,
                    delta_seconds=runtime_dt,
                )
            except TypeError:
                advance_openloop_runtime(frame_idx, timestamp, runtime_dt)

        # Build planner-facing input data by declared sensor schema.
        bev_img = _get_live_bev_image(hero_vehicles[0])
        input_data = _build_agent_input_data(
            agent=agent,
            sensor_specs=sensor_specs,
            frame_data=frame_data,
            actor_id=primary_actor_id,
            slot_id=0,
            frame_idx=frame_idx,
            bev_img=bev_img,
        )
        # Normalize all speed payloads so both ['speed'] and ['move_state']['speed']
        # exist as numpy scalars (planner logging often calls .tolist()).
        for _speed_key, _speed_pair in list(input_data.items()):
            if not str(_speed_key).startswith("speed_"):
                continue
            try:
                _t, _sd = _speed_pair
                _sp = _extract_speed_scalar(_sd)
                input_data[_speed_key] = (_t, _build_speed_payload(_sp))
            except Exception:
                pass
        try:
            _m = measurements
            _sp = _m.get("speed", None)
            if _sp is not None and not hasattr(_sp, "tolist"):
                _m["speed"] = np.float32(_sp)
        except Exception:
            pass
        input_data_overview = _summarize_input_data(input_data)
        sensor_stats = _summarize_frame_sensors(frame_data)
        input_files = _build_frame_input_files_manifest(
            scenario_dir=scenario.scenario_dir,
            actor_id=int(primary_actor_id),
            frame_idx=int(frame_idx),
            camera_assignment=cam_assign,
        )
        measurements_debug = _build_measurements_debug_payload(
            measurements,
            speed_mps=speed_mps,
        )

        run_exc = None
        run_output = None
        try:
            run_output = agent.run_step(input_data, timestamp)
        except Exception as exc:
            run_exc = exc
            exc_signature = f"{type(exc).__name__}: {exc}"
            if verbose or exc_signature != last_run_exc_signature:
                print(f"  [frame {frame_idx}] run_step exception: {exc_signature}")
                last_run_exc_signature = exc_signature
            if verbose:
                traceback.print_exc()

        dt = time.perf_counter() - t0

        # BEV image was generated before run_step for sensor synthesis; keep one
        # post-run fallback in case producer was not ready at pre-run time.
        if bev_img is None:
            bev_img = _get_live_bev_image(hero_vehicles[0])

        pred_world    = None
        rollout_pred  = None
        gt_future_xy  = None
        frame_metrics: Dict[str, float] = {}
        legacy_frame_metrics: Dict[str, Any] = {}
        canonical_frame_metrics: Dict[str, Any] = {}
        pred_source = ""
        ctrl = None
        planner_out = None
        adapter_source = "none"
        is_stale = False
        _compass = None
        _gt_indices: List[int] = []
        surround_traj = np.zeros((0, 0, 2), dtype=np.float64)
        surround_mask = np.zeros((0, 0), dtype=np.float64)
        surround_actor_ids: List[int] = []
        primary_pred_world_vis = pred_world
        primary_gt_world_vis = gt_future_xy
        primary_pred_raw_vis = None
        vad_branch_diagnostics: Optional[Dict[str, Any]] = None
        branch_selected_frame_metrics: Dict[str, Any] = {}
        branch_oracle_frame_metrics: Dict[str, Any] = {}
        frame_eval_diagnostics: Dict[str, Any] = {}
        selected_eval_wps_ref = np.zeros((0, 2), dtype=np.float64)
        selected_gt_future_ref = np.zeros((0, 2), dtype=np.float64)
        selected_gt_mask_ref = np.zeros((0,), dtype=np.float64)
        selected_eval_dt_frames_ref = int(PREDICTION_DOWNSAMPLING_RATE)
        oracle_branch_world_candidate: Optional[np.ndarray] = None
        oracle_eval_wps_ref = np.zeros((0, 2), dtype=np.float64)
        oracle_gt_future_ref = np.zeros((0, 2), dtype=np.float64)
        oracle_gt_mask_ref = np.zeros((0,), dtype=np.float64)

        if run_exc is None:
            ctrl = _extract_primary_control(run_output, slot_id=0)
            if ctrl is not None:
                try:
                    import carla
                    hero_vehicles[0].apply_control(
                        carla.VehicleControl(
                            throttle=float(ctrl.get("throttle", 0.0)),
                            steer=float(ctrl.get("steer", 0.0)),
                            brake=float(ctrl.get("brake", 0.0)),
                        )
                    )
                except Exception:
                    try:
                        hero_vehicles[0].apply_control(ctrl)
                    except Exception:
                        pass
                # ── Stale-control detection (diagnostic only) ──
                is_stale = stale_detector.check(ctrl)
                if is_stale and verbose:
                    _log.debug("frame %d: control unchanged from previous frame", frame_idx)

                # ── Compute control rollout (kept for debug metric) ──
                rollout_pred = _rollout_control_trajectory(
                    pose=pose,
                    speed_mps=speed_mps,
                    control=ctrl,
                    horizon_steps=int(EVAL_HORIZON_STEPS),
                    dt=PREDICTION_DOWNSAMPLING_RATE * runtime_dt,
                )

                # ── Extract planner's own trajectory via adapter ──
                # Resolve ego compass for ego-local → world conversion.
                _compass = _resolve_compass_rad(pose, measurements)
                _ego_xy  = np.array([pose[0], pose[1]], dtype=np.float64)
                planner_out = extract_planner_trajectory(
                    agent, _ego_xy, _compass, planner_name=planner_name,
                    measurements=measurements,
                )
                adapter_source = str(planner_out.traj_source or "none")
                adapter_source_counts[adapter_source] = (
                    adapter_source_counts.get(adapter_source, 0) + 1
                )

                planner_future = planner_out.future_trajectory_world
                if traj_mode == "control_rollout":
                    pred_world = rollout_pred
                    pred_source = "control_rollout"
                elif traj_mode == "planner_waypoints":
                    if planner_future is not None:
                        pred_world = planner_future
                        pred_source = str(planner_out.traj_source or "planner_waypoints")
                    else:
                        pred_world = None
                        pred_source = "planner_waypoints_missing"
                        planner_waypoint_miss_count += 1
                else:
                    if planner_future is not None:
                        pred_world = planner_future
                        pred_source = str(planner_out.traj_source or "planner_waypoints")
                    else:
                        # Fallback in auto mode when planner does not
                        # expose a trajectory for this frame.
                        pred_world = rollout_pred
                        pred_source = "control_rollout"

                traj_source_counts[pred_source] = (
                    traj_source_counts.get(pred_source, 0) + 1
                )

        if run_exc is None and pred_world is not None:
            # ── Clip resampled pred for visualisation / rollout ──
            T_pred = min(int(EVAL_HORIZON_STEPS), len(pred_world))
            pred_world = np.asarray(pred_world, dtype=np.float64)[:T_pred, :2]
            if rollout_pred is not None:
                rollout_pred = np.asarray(rollout_pred, dtype=np.float64)[:T_pred, :2]

            # ── Select evaluation waypoints + GT time grid ──────────
            def _build_eval_grid(
                pred_world_now: np.ndarray,
            ) -> Tuple[bool, np.ndarray, int, List[int], np.ndarray, np.ndarray]:
                # native_horizon=True: use planner raw (pre-resample) waypoints
                # and match GT at planner native dt.
                raw_avail_now = (
                    native_horizon
                    and planner_out is not None
                    and planner_out.raw_world_wps is not None
                    and pred_source != "control_rollout"
                )
                if raw_avail_now:
                    eval_now = np.asarray(planner_out.raw_world_wps, dtype=np.float64)[:, :2]
                    eval_dt_frames_now = max(1, round(planner_out.native_dt / runtime_dt))
                else:
                    eval_now = np.asarray(pred_world_now, dtype=np.float64)[:, :2]
                    eval_dt_frames_now = int(PREDICTION_DOWNSAMPLING_RATE)
                gt_idx_now = [
                    frame_idx + eval_dt_frames_now * t
                    for t in range(1, len(eval_now) + 1)
                    if frame_idx + eval_dt_frames_now * t < n_frames
                ]
                gt_now = (
                    gt_traj_all[gt_idx_now]
                    if gt_idx_now
                    else np.zeros((0, 2), dtype=np.float64)
                )
                gt_mask_now = np.ones(len(gt_now), dtype=np.float64)
                return raw_avail_now, eval_now, eval_dt_frames_now, gt_idx_now, gt_now, gt_mask_now

            _raw_avail, eval_wps, _eval_dt_frames, _gt_indices, gt_future_xy, gt_future_mask = (
                _build_eval_grid(pred_world)
            )
            T_eval = len(eval_wps)
            selected_eval_wps_ref = np.asarray(eval_wps, dtype=np.float64).copy()
            selected_gt_future_ref = np.asarray(gt_future_xy, dtype=np.float64).copy()
            selected_gt_mask_ref = np.asarray(gt_future_mask, dtype=np.float64).copy()
            selected_eval_dt_frames_ref = int(_eval_dt_frames)

            # Standard 0.5-s GT grid kept for the debug rollout accumulator
            _dsr = PREDICTION_DOWNSAMPLING_RATE
            _gt_indices_std = [
                frame_idx + _dsr * t
                for t in range(1, T_pred + 1)
                if frame_idx + _dsr * t < n_frames
            ]
            gt_std_xy = (
                gt_traj_all[_gt_indices_std]
                if _gt_indices_std
                else np.zeros((0, 2), dtype=np.float64)
            )
            gt_std_mask = np.ones(len(gt_std_xy))

            vad_branch_diagnostics = _compute_vad_branch_diagnostics(
                planner_name=planner_name,
                planner_debug=(planner_out.debug if planner_out is not None else None),
                measurements=measurements,
                ego_world_xy=np.asarray([pose[0], pose[1]], dtype=np.float64),
                compass_rad=_compass,
                gt_future_world=gt_future_xy,
                gt_valid_mask=gt_future_mask,
            )

            best_idx = (
                vad_branch_diagnostics.get("best_branch_index_by_local_ade")
                if isinstance(vad_branch_diagnostics, dict)
                else None
            )
            branch_override = _select_vad_branch_world_from_index(
                planner_debug=(planner_out.debug if planner_out is not None else None),
                branch_index=best_idx,
                ego_world_xy=np.asarray([pose[0], pose[1]], dtype=np.float64),
                compass_rad=_compass,
            )
            if branch_override is not None:
                oracle_branch_world_candidate = np.asarray(
                    branch_override["world_wps"], dtype=np.float64
                )

            if (
                vad_oracle_branch_by_local_ade
                and planner_out is not None
                and pred_source != "control_rollout"
                and branch_override is not None
            ):
                oracle_idx = int(branch_override["branch_index"])
                oracle_world = np.asarray(branch_override["world_wps"], dtype=np.float64)
                oracle_local = np.asarray(
                    branch_override["local_wps_vad_frame"], dtype=np.float64
                )

                prev_source = str(pred_source or "")
                if prev_source:
                    traj_source_counts[prev_source] = max(
                        0, int(traj_source_counts.get(prev_source, 0)) - 1
                    )
                    pred_source = f"{prev_source}|vad_oracle_local_ade"
                else:
                    pred_source = "vad_oracle_local_ade"
                traj_source_counts[pred_source] = (
                    int(traj_source_counts.get(pred_source, 0)) + 1
                )

                pred_world = oracle_world
                planner_out.future_trajectory_world = oracle_world
                planner_out.raw_world_wps = oracle_world
                planner_out.raw_length = int(len(oracle_world))

                if not isinstance(planner_out.debug, dict):
                    planner_out.debug = {}
                planner_out.debug["selected_command_index_original"] = (
                    planner_out.debug.get("selected_command_index")
                )
                planner_out.debug["selected_command_index"] = int(oracle_idx)
                planner_out.debug["oracle_selected_by_local_ade"] = True
                planner_out.debug["oracle_selected_branch_index"] = int(oracle_idx)
                planner_out.debug["oracle_selection_metric"] = (
                    "local_ade_to_gt_eval_grid"
                )

                raw_export_obj = getattr(planner_out, "raw_export", None)
                if raw_export_obj is not None:
                    try:
                        raw_export_obj.raw_positions = oracle_local
                        notes = list(getattr(raw_export_obj, "adapter_debug_notes", []) or [])
                        notes.append(
                            "oracle branch override enabled: selected VAD branch "
                            "with minimum local ADE to GT on the current eval grid"
                        )
                        raw_export_obj.adapter_debug_notes = notes
                    except Exception:
                        pass

                _raw_avail, eval_wps, _eval_dt_frames, _gt_indices, gt_future_xy, gt_future_mask = (
                    _build_eval_grid(pred_world)
                )
                T_eval = len(eval_wps)

                if vad_branch_diagnostics is None:
                    vad_branch_diagnostics = {}
                vad_branch_diagnostics["oracle_branch_selection_applied"] = True
                vad_branch_diagnostics["oracle_branch_selection_mode"] = (
                    "min_local_ade_to_gt"
                )
                vad_branch_diagnostics["oracle_selected_branch_index"] = int(oracle_idx)
                vad_branch_diagnostics["oracle_selected_command_value"] = int(oracle_idx + 1)
                vad_branch_diagnostics["oracle_selected_command_name"] = _roadoption_name(
                    int(oracle_idx + 1)
                )
            elif vad_branch_diagnostics is not None:
                vad_branch_diagnostics["oracle_branch_selection_applied"] = False

            if len(gt_future_xy) > 0:
                if len(gt_future_xy) < T_eval:
                    pad = T_eval - len(gt_future_xy)
                    gt_future_xy   = np.vstack([gt_future_xy,
                                                np.zeros((pad, 2))])
                    gt_future_mask = np.concatenate([gt_future_mask,
                                                     np.zeros(pad)])
                if len(gt_std_xy) < T_pred:
                    pad = T_pred - len(gt_std_xy)
                    gt_std_xy   = np.vstack([gt_std_xy,
                                             np.zeros((pad, 2))])
                    gt_std_mask = np.concatenate([gt_std_mask,
                                                  np.zeros(pad)])

                try:
                    _surround_out = scenario.get_surrounding_trajectories(
                        primary_actor_id,
                        frame_idx,
                        return_actor_ids=True,
                    )
                    if (
                        isinstance(_surround_out, tuple)
                        and len(_surround_out) == 3
                    ):
                        surround_traj, surround_mask, surround_actor_ids = _surround_out
                    else:
                        surround_traj, surround_mask = _surround_out  # type: ignore[misc]
                        surround_actor_ids = []
                except TypeError:
                    surround_traj, surround_mask = scenario.get_surrounding_trajectories(
                        primary_actor_id, frame_idx
                    )
                    surround_actor_ids = []

                oracle_eval_wps_ref = np.zeros((0, 2), dtype=np.float64)
                oracle_gt_future_ref = np.zeros((0, 2), dtype=np.float64)
                oracle_gt_mask_ref = np.zeros((0,), dtype=np.float64)
                if oracle_branch_world_candidate is not None and selected_eval_dt_frames_ref > 0:
                    oracle_eval_wps_ref = np.asarray(
                        oracle_branch_world_candidate, dtype=np.float64
                    )[:, :2]
                    oracle_gt_indices = [
                        frame_idx + selected_eval_dt_frames_ref * t
                        for t in range(1, len(oracle_eval_wps_ref) + 1)
                        if frame_idx + selected_eval_dt_frames_ref * t < n_frames
                    ]
                    oracle_gt_future_ref = (
                        gt_traj_all[oracle_gt_indices]
                        if oracle_gt_indices
                        else np.zeros((0, 2), dtype=np.float64)
                    )
                    oracle_gt_mask_ref = np.ones(len(oracle_gt_future_ref), dtype=np.float64)

                # ── Skip stale frames from metric accumulation ───────
                if skip_stale_frames and is_stale:
                    legacy_frame_metrics = {
                        "frame_idx": frame_idx,
                        "latency_s": round(dt, 4),
                        "stale": True,
                        "inclusion_reason": "stale",
                    }
                    branch_selected_frame_metrics = {
                        "inclusion_reason": "stale",
                        "scored": False,
                    }
                    branch_oracle_frame_metrics = {
                        "inclusion_reason": "stale",
                        "scored": False,
                    }
                else:
                    if len(selected_eval_wps_ref) > 0 and len(selected_gt_future_ref) > 0:
                        _selected_branch_m = branch_selected_accum.update(
                            pred_world_wps=selected_eval_wps_ref,
                            gt_future_xy=selected_gt_future_ref,
                            gt_future_mask=selected_gt_mask_ref,
                            surround_traj=surround_traj,
                            surround_mask=surround_mask,
                            ego_pose=pose,
                        )
                        if _selected_branch_m:
                            branch_selected_frame_metrics = {
                                **_selected_branch_m,
                                "inclusion_reason": "scored",
                                "scored": True,
                            }
                    if len(oracle_eval_wps_ref) > 0 and len(oracle_gt_future_ref) > 0:
                        _oracle_branch_m = branch_oracle_accum.update(
                            pred_world_wps=oracle_eval_wps_ref,
                            gt_future_xy=oracle_gt_future_ref,
                            gt_future_mask=oracle_gt_mask_ref,
                            surround_traj=surround_traj,
                            surround_mask=surround_mask,
                            ego_pose=pose,
                        )
                        if _oracle_branch_m:
                            branch_oracle_frame_metrics = {
                                **_oracle_branch_m,
                                "inclusion_reason": "scored",
                                "scored": True,
                            }

                    legacy_frame_metrics = legacy_accum.update(
                        pred_world_wps = eval_wps,
                        gt_future_xy   = gt_future_xy,
                        gt_future_mask = gt_future_mask,
                        surround_traj  = surround_traj,
                        surround_mask  = surround_mask,
                        ego_pose       = pose,
                    )
                    legacy_frame_metrics.update({
                        "frame_idx": frame_idx,
                        "latency_s": round(dt, 4),
                        "inclusion_reason": "scored",
                    })
                if pred_source:
                    legacy_frame_metrics["traj_source"] = pred_source
                legacy_frame_metrics["adapter_source"] = adapter_source
                legacy_frame_metrics["stale"] = bool(is_stale)
                legacy_frame_metrics["traj_len_pred"] = int(T_eval)
                legacy_frame_metrics["traj_len_gt"] = int(len(gt_future_xy))
                legacy_frame_metrics["traj_len_raw"] = int(
                    planner_out.raw_length
                    if run_exc is None and ctrl is not None and planner_out is not None
                    else 0
                )
                legacy_frame_metrics["native_dt_used"] = float(
                    planner_out.native_dt
                    if _raw_avail and planner_out is not None
                    else PREDICTION_DOWNSAMPLING_RATE * runtime_dt
                )

                # ── Debug: also compute rollout-based ADE/FDE ────────
                if rollout_pred is not None and len(gt_std_xy) > 0:
                    _debug_m = debug_rollout_accum.update(
                        pred_world_wps = rollout_pred,
                        gt_future_xy   = gt_std_xy,
                        gt_future_mask = gt_std_mask,
                        surround_traj  = surround_traj,
                        surround_mask  = surround_mask,
                        ego_pose       = pose,
                    )
                    if _debug_m:
                        legacy_frame_metrics["debug_ade_rollout"] = _debug_m.get(
                            "plan_ade", float("nan"))
                        legacy_frame_metrics["debug_fde_rollout"] = _debug_m.get(
                            "plan_fde", float("nan"))

                # Sanity check 1: pred == gt should yield ADE ~= 0.
                if sanity_checks["pred_eq_gt_ade"] is None:
                    sanity_checks["pred_eq_gt_ade"] = _compute_ade_only(
                        pred_xy=gt_future_xy,
                        gt_xy=gt_future_xy,
                        mask=gt_future_mask,
                    )

                # Sanity check 2: shifting GT by one step should worsen ADE.
                if sanity_checks["gt_shifted_ade"] is None and len(gt_future_xy) >= 2:
                    _shifted = np.vstack([gt_future_xy[1:], gt_future_xy[-1:]])
                    sanity_checks["gt_shifted_ade"] = _compute_ade_only(
                        pred_xy=gt_future_xy,
                        gt_xy=_shifted,
                        mask=gt_future_mask,
                    )
        elif verbose and run_exc is None:
            print(f"  [frame {frame_idx}] no control extracted (skipping metrics)")

        raw_export_record = _build_raw_export_record(
            planner_name=planner_name,
            scenario_id=sname_eval,
            frame_id=frame_idx,
            planner_out=planner_out,
            adapter_source=adapter_source,
            traj_source_selected=pred_source or None,
            run_step_exception=run_exc,
            freshness_tracker=raw_export_freshness,
        )
        if raw_export_fh is not None:
            try:
                raw_export_fh.write(
                    json.dumps(_jsonify_debug_value(raw_export_record), ensure_ascii=False)
                    + "\n"
                )
            except Exception as exc:
                _log.warning(
                    "[%s] failed to write raw export record at frame %d: %s",
                    sname_eval,
                    frame_idx,
                    exc,
                )
        canonical_export_record = None
        if (
            canonical_export_fh is not None
            or scoring_debug_fh is not None
            or scoring_path == "canonical_fresh_only"
            or vis_logger is not None
        ):
            canonical_export_record = _build_canonical_export_record(
                raw_export_record=raw_export_record,
                planner_out=planner_out,
                gt_traj_all=gt_traj_all,
                runtime_dt=runtime_dt,
            )
        if canonical_export_fh is not None and canonical_export_record is not None:
            try:
                canonical_export_fh.write(
                    json.dumps(_jsonify_debug_value(canonical_export_record), ensure_ascii=False)
                    + "\n"
                )
            except Exception as exc:
                _log.warning(
                    "[%s] failed to write canonical export record at frame %d: %s",
                    sname_eval,
                    frame_idx,
                    exc,
                )

        if canonical_export_record is not None:
            canonical_frame_metrics = _build_stage4_scoring_debug_record(
                planner_name=str(planner_name or "generic"),
                scenario_id=sname_eval,
                frame_id=frame_idx,
                canonical_record=canonical_export_record,
                surround_traj=surround_traj,
                surround_mask=surround_mask,
                surround_actor_ids=surround_actor_ids,
                ego_pose=pose,
                run_step_exception=run_exc,
            )
        elif run_exc is not None:
            canonical_frame_metrics = _build_stage4_scoring_debug_record(
                planner_name=str(planner_name or "generic"),
                scenario_id=sname_eval,
                frame_id=frame_idx,
                canonical_record=None,
                surround_traj=surround_traj,
                surround_mask=surround_mask,
                surround_actor_ids=surround_actor_ids,
                ego_pose=pose,
                run_step_exception=run_exc,
            )

        if scoring_debug_fh is not None and canonical_frame_metrics:
            try:
                scoring_debug_fh.write(
                    json.dumps(_jsonify_debug_value(canonical_frame_metrics), ensure_ascii=False)
                    + "\n"
                )
            except Exception as exc:
                _log.warning(
                    "[%s] failed to write scoring debug record at frame %d: %s",
                    sname_eval,
                    frame_idx,
                    exc,
                )

        selected_mask_bool = np.asarray(selected_gt_mask_ref, dtype=np.float64) > 0
        oracle_mask_bool = np.asarray(oracle_gt_mask_ref, dtype=np.float64) > 0
        selected_error_components: Optional[Dict[str, Any]] = None
        oracle_error_components: Optional[Dict[str, Any]] = None
        if _compass is not None and len(selected_eval_wps_ref) > 0 and len(selected_gt_future_ref) > 0:
            selected_error_components = _trajectory_error_components_local(
                selected_eval_wps_ref,
                selected_gt_future_ref,
                ego_world_xy=np.asarray([pose[0], pose[1]], dtype=np.float64),
                compass_rad=float(_compass),
                valid_mask=selected_mask_bool,
            )
        if _compass is not None and len(oracle_eval_wps_ref) > 0 and len(oracle_gt_future_ref) > 0:
            oracle_error_components = _trajectory_error_components_local(
                oracle_eval_wps_ref,
                oracle_gt_future_ref,
                ego_world_xy=np.asarray([pose[0], pose[1]], dtype=np.float64),
                compass_rad=float(_compass),
                valid_mask=oracle_mask_bool,
            )

        selected_path_len = _path_length_m(selected_eval_wps_ref)
        gt_path_len = _path_length_m(selected_gt_future_ref)
        oracle_path_len = _path_length_m(oracle_eval_wps_ref)
        selected_headchg = _heading_change_rad(selected_eval_wps_ref)
        gt_headchg = _heading_change_rad(selected_gt_future_ref)
        oracle_headchg = _heading_change_rad(oracle_eval_wps_ref)
        frame_eval_diagnostics = {
            "speed_mps": float(speed_mps),
            "startup_frame_idx_threshold": 1,
            "low_speed_threshold_mps": 1.0,
            "is_startup_frame": bool(int(frame_idx) <= 1),
            "is_low_speed_frame": bool(float(speed_mps) <= 1.0),
            "route_command_value": (
                vad_branch_diagnostics.get("route_command_value")
                if isinstance(vad_branch_diagnostics, dict) else None
            ),
            "route_command_name": (
                vad_branch_diagnostics.get("route_command_name")
                if isinstance(vad_branch_diagnostics, dict) else None
            ),
            "selected_command_raw_value": (
                vad_branch_diagnostics.get("selected_command_raw_value")
                if isinstance(vad_branch_diagnostics, dict) else None
            ),
            "selected_command_fallback_to_lanefollow": (
                vad_branch_diagnostics.get("selected_command_fallback_to_lanefollow")
                if isinstance(vad_branch_diagnostics, dict) else None
            ),
            "selected_branch_index": (
                vad_branch_diagnostics.get("selected_branch_index")
                if isinstance(vad_branch_diagnostics, dict) else None
            ),
            "best_branch_index_by_local_ade": (
                vad_branch_diagnostics.get("best_branch_index_by_local_ade")
                if isinstance(vad_branch_diagnostics, dict) else None
            ),
            "best_branch_index_by_world_ade": (
                vad_branch_diagnostics.get("best_branch_index_by_world_ade")
                if isinstance(vad_branch_diagnostics, dict) else None
            ),
            "selected_ade_m": (
                branch_selected_frame_metrics.get("plan_ade")
                if isinstance(branch_selected_frame_metrics, dict) else None
            ),
            "selected_fde_m": (
                branch_selected_frame_metrics.get("plan_fde")
                if isinstance(branch_selected_frame_metrics, dict) else None
            ),
            "oracle_ade_m": (
                branch_oracle_frame_metrics.get("plan_ade")
                if isinstance(branch_oracle_frame_metrics, dict) else None
            ),
            "oracle_fde_m": (
                branch_oracle_frame_metrics.get("plan_fde")
                if isinstance(branch_oracle_frame_metrics, dict) else None
            ),
            "selected_minus_oracle_ade_m": (
                float(branch_selected_frame_metrics.get("plan_ade") - branch_oracle_frame_metrics.get("plan_ade"))
                if isinstance(branch_selected_frame_metrics.get("plan_ade"), (int, float))
                and isinstance(branch_oracle_frame_metrics.get("plan_ade"), (int, float))
                and math.isfinite(float(branch_selected_frame_metrics.get("plan_ade")))
                and math.isfinite(float(branch_oracle_frame_metrics.get("plan_ade")))
                else None
            ),
            "selected_path_length_m": selected_path_len,
            "gt_path_length_m": gt_path_len,
            "oracle_path_length_m": oracle_path_len,
            "selected_path_length_minus_gt_m": (
                float(selected_path_len - gt_path_len)
                if selected_path_len is not None and gt_path_len is not None
                else None
            ),
            "oracle_path_length_minus_gt_m": (
                float(oracle_path_len - gt_path_len)
                if oracle_path_len is not None and gt_path_len is not None
                else None
            ),
            "selected_heading_change_rad": selected_headchg,
            "gt_heading_change_rad": gt_headchg,
            "oracle_heading_change_rad": oracle_headchg,
            "selected_heading_change_minus_gt_rad": (
                float(selected_headchg - gt_headchg)
                if selected_headchg is not None and gt_headchg is not None
                else None
            ),
            "oracle_heading_change_minus_gt_rad": (
                float(oracle_headchg - gt_headchg)
                if oracle_headchg is not None and gt_headchg is not None
                else None
            ),
            "selected_error_components_ego_local": selected_error_components,
            "oracle_error_components_ego_local": oracle_error_components,
            "is_fresh_plan": bool(raw_export_record.get("is_fresh_plan", False)),
            "reused_from_step": raw_export_record.get("reused_from_step"),
            "timestamp_source": (
                canonical_frame_metrics.get("timestamp_source")
                if isinstance(canonical_frame_metrics, dict) else None
            ),
            "interpolation_method": (
                canonical_export_record.get("interpolation_method")
                if isinstance(canonical_export_record, dict) else None
            ),
            "trajectory_extrapolated": (
                "extrapolation" in str(canonical_export_record.get("interpolation_method", ""))
                if isinstance(canonical_export_record, dict) else None
            ),
            "scored": (
                bool(canonical_frame_metrics.get("scored", False))
                if isinstance(canonical_frame_metrics, dict) else False
            ),
            "inclusion_reason": (
                canonical_frame_metrics.get("inclusion_reason")
                if isinstance(canonical_frame_metrics, dict) else None
            ),
        }

        if scoring_path == "canonical_fresh_only":
            if not canonical_frame_metrics:
                canonical_frame_metrics = _build_stage4_scoring_debug_record(
                    planner_name=str(planner_name or "generic"),
                    scenario_id=sname_eval,
                    frame_id=frame_idx,
                    canonical_record=None,
                    surround_traj=surround_traj,
                    surround_mask=surround_mask,
                    surround_actor_ids=surround_actor_ids,
                    ego_pose=pose,
                    run_step_exception=run_exc,
                )
            canonical_accum.update_from_debug_record(canonical_frame_metrics)
            canonical_timestamps = canonical_frame_metrics.get("canonical_timestamps")
            canonical_traj_len = (
                int(len(canonical_timestamps))
                if canonical_timestamps is not None
                else 0
            )
            frame_metrics = {
                "frame_idx": int(frame_idx),
                "latency_s": round(dt, 4),
                "traj_source": pred_source or None,
                "adapter_source": adapter_source,
                "stale": bool(is_stale),
                "traj_len_pred": canonical_traj_len,
                "traj_len_gt": canonical_traj_len,
                "traj_len_raw": int(
                    planner_out.raw_length
                    if run_exc is None and ctrl is not None and planner_out is not None
                    else 0
                ),
                "native_dt_used": float(
                    raw_export_record.get("native_dt")
                    if raw_export_record.get("native_dt") is not None
                    else PREDICTION_DOWNSAMPLING_RATE * runtime_dt
                ),
                "scoring_path": "canonical_fresh_only",
                "scored": bool(canonical_frame_metrics.get("scored", False)),
                "inclusion_reason": canonical_frame_metrics.get("inclusion_reason"),
                "timestamp_source": canonical_frame_metrics.get("timestamp_source"),
                "collision": int(canonical_frame_metrics.get("collision", 0)),
                "plan_ade": canonical_frame_metrics.get("plan_ade", float("nan")),
                "plan_fde": canonical_frame_metrics.get("plan_fde", float("nan")),
                "ade_breakdown_m": canonical_frame_metrics.get("ade_breakdown_m"),
                "legacy_plan_ade": legacy_frame_metrics.get("plan_ade", float("nan")),
                "legacy_plan_fde": legacy_frame_metrics.get("plan_fde", float("nan")),
                "legacy_collision": legacy_frame_metrics.get("collision", 0),
                "branch_selected_plan_ade": branch_selected_frame_metrics.get(
                    "plan_ade", float("nan")
                ),
                "branch_selected_plan_fde": branch_selected_frame_metrics.get(
                    "plan_fde", float("nan")
                ),
                "branch_oracle_plan_ade": branch_oracle_frame_metrics.get(
                    "plan_ade", float("nan")
                ),
                "branch_oracle_plan_fde": branch_oracle_frame_metrics.get(
                    "plan_fde", float("nan")
                ),
                "selected_minus_oracle_ade_m": frame_eval_diagnostics.get(
                    "selected_minus_oracle_ade_m", float("nan")
                ),
            }
            if "debug_ade_rollout" in legacy_frame_metrics:
                frame_metrics["debug_ade_rollout"] = legacy_frame_metrics["debug_ade_rollout"]
                frame_metrics["debug_fde_rollout"] = legacy_frame_metrics.get(
                    "debug_fde_rollout", float("nan")
                )
            primary_pred_world_vis = (
                np.asarray(canonical_export_record.get("canonical_positions"), dtype=np.float64)
                if canonical_export_record is not None else None
            )
            primary_gt_world_vis = (
                np.asarray(canonical_export_record.get("gt_canonical_positions"), dtype=np.float64)
                if canonical_export_record is not None else None
            )
            primary_pred_raw_vis = (
                planner_out.raw_world_wps
                if planner_out is not None and pred_source != "control_rollout"
                else None
            )

            # Branch diagnostics are also tracked on the canonical scored subset
            # so selected-vs-oracle numbers are directly comparable to overall ADE/FDE.
            if bool(canonical_frame_metrics.get("scored", False)):
                if len(selected_eval_wps_ref) > 0 and len(selected_gt_future_ref) > 0:
                    branch_selected_scored_accum.update(
                        pred_world_wps=selected_eval_wps_ref,
                        gt_future_xy=selected_gt_future_ref,
                        gt_future_mask=selected_gt_mask_ref,
                        surround_traj=surround_traj,
                        surround_mask=surround_mask,
                        ego_pose=pose,
                    )
                if len(oracle_eval_wps_ref) > 0 and len(oracle_gt_future_ref) > 0:
                    branch_oracle_scored_accum.update(
                        pred_world_wps=oracle_eval_wps_ref,
                        gt_future_xy=oracle_gt_future_ref,
                        gt_future_mask=oracle_gt_mask_ref,
                        surround_traj=surround_traj,
                        surround_mask=surround_mask,
                        ego_pose=pose,
                    )
        else:
            frame_metrics = dict(legacy_frame_metrics)
            frame_metrics["scoring_path"] = "legacy"
            frame_metrics["branch_selected_plan_ade"] = branch_selected_frame_metrics.get(
                "plan_ade", float("nan")
            )
            frame_metrics["branch_selected_plan_fde"] = branch_selected_frame_metrics.get(
                "plan_fde", float("nan")
            )
            frame_metrics["branch_oracle_plan_ade"] = branch_oracle_frame_metrics.get(
                "plan_ade", float("nan")
            )
            frame_metrics["branch_oracle_plan_fde"] = branch_oracle_frame_metrics.get(
                "plan_fde", float("nan")
            )
            frame_metrics["selected_minus_oracle_ade_m"] = frame_eval_diagnostics.get(
                "selected_minus_oracle_ade_m", float("nan")
            )
            primary_pred_world_vis = pred_world
            primary_gt_world_vis = gt_future_xy
            primary_pred_raw_vis = (
                planner_out.raw_world_wps
                if planner_out is not None and pred_source != "control_rollout"
                else None
            )

        if frame_debug_fh is not None:
            frame_debug_record = {
                "scenario": sname,
                "scenario_eval": sname_eval,
                "ego_actor_id": int(primary_actor_id),
                "frame_idx": int(frame_idx),
                "timestamp_s": float(timestamp),
                "runtime_dt_s": float(runtime_dt),
                "pose_world": np.asarray(pose, dtype=np.float64),
                "ego_world_xy": [float(pose[0]), float(pose[1])],
                "measurements": measurements_debug,
                "sensor_stats": sensor_stats,
                "input_files": input_files,
                "camera_assignment": cam_assign,
                "input_data_overview": input_data_overview,
                "run_step_exception": run_exc,
                "run_output_type": (
                    type(run_output).__name__ if run_output is not None else None
                ),
                "control": ctrl,
                "stale_control": bool(is_stale),
                "traj_source_selected": pred_source or None,
                "adapter_source": adapter_source,
                "trajectory_extrapolation": bool(trajectory_extrapolation),
                "planner_raw_length": (
                    planner_out.raw_length if planner_out is not None else 0
                ),
                "planner_future_available": bool(
                    planner_out is not None and planner_out.future_trajectory_world is not None
                ),
                "planner_future_world_raw": (
                    planner_out.future_trajectory_world if planner_out is not None else None
                ),
                "planner_adapter_debug": (
                    planner_out.debug if planner_out is not None else None
                ),
                "vad_branch_diagnostics": vad_branch_diagnostics,
                "frame_eval_diagnostics": frame_eval_diagnostics,
                "pred_world_used": primary_pred_world_vis,
                "control_rollout_world": rollout_pred,
                "gt_future_world": primary_gt_world_vis,
                "gt_future_indices": _gt_indices,
                "metrics": frame_metrics if frame_metrics else None,
                "legacy_metrics": legacy_frame_metrics if legacy_frame_metrics else None,
                "canonical_scoring_metrics": (
                    canonical_frame_metrics if canonical_frame_metrics else None
                ),
                "latency_s": float(dt),
                "resolved_compass_rad": _compass,
            }
            try:
                frame_debug_fh.write(
                    json.dumps(_jsonify_debug_value(frame_debug_record), ensure_ascii=False)
                    + "\n"
                )
            except Exception as exc:
                _log.warning(
                    "[%s] failed to write frame debug record at frame %d: %s",
                    sname_eval,
                    frame_idx,
                    exc,
                )

        per_frame.append(frame_metrics)

        if vis_logger is not None:
            vis_logger.log_frame(
                frame_idx    = frame_idx,
                n_frames     = n_frames,
                pose         = pose,
                frame_data   = frame_data,
                bev_img      = bev_img,
                pred_world   = primary_pred_world_vis,
                gt_future_xy = primary_gt_world_vis,
                frame_metrics= frame_metrics,
                latency_s    = dt,
                cam_assign   = cam_assign,
                run_step_exc = run_exc,
                pred_world_raw = primary_pred_raw_vis,
            )
        elif verbose and frame_metrics:
            print(
                f"  frame {frame_idx:3d}  "
                f"plan_ade={frame_metrics.get('plan_ade', float('nan')):.3f} m  "
                f"col={frame_metrics.get('collision', 0)}  "
                f"t={dt*1000:.1f} ms"
            )

    if frame_debug_fh is not None:
        try:
            frame_debug_fh.flush()
            frame_debug_fh.close()
        except Exception:
            pass
    if raw_export_fh is not None:
        try:
            raw_export_fh.flush()
            raw_export_fh.close()
        except Exception:
            pass
    if canonical_export_fh is not None:
        try:
            canonical_export_fh.flush()
            canonical_export_fh.close()
        except Exception:
            pass
    if scoring_debug_fh is not None:
        try:
            scoring_debug_fh.flush()
            scoring_debug_fh.close()
        except Exception:
            pass

    # ── End-of-scenario logging / validation ──
    _legacy_metrics = legacy_accum.summary()
    _canonical_metrics = canonical_accum.summary()
    _primary_metrics = (
        _canonical_metrics
        if scoring_path == "canonical_fresh_only"
        else _legacy_metrics
    )
    _debug_metrics   = debug_rollout_accum.summary()
    _branch_selected_metrics = branch_selected_accum.summary()
    _branch_oracle_metrics = branch_oracle_accum.summary()
    _branch_selected_scored_metrics = branch_selected_scored_accum.summary()
    _branch_oracle_scored_metrics = branch_oracle_scored_accum.summary()
    _stale_info      = stale_detector.summary()
    _coverage_stats  = canonical_accum.coverage_summary()
    _per_horizon_ade = canonical_accum.per_horizon_ade()

    # Warn if most frames fell back to control rollout.
    n_rollout = traj_source_counts.get("control_rollout", 0)
    n_total_src = sum(traj_source_counts.values())
    if traj_mode in {"auto", "control_rollout"} and n_total_src > 0 and n_rollout > 0.5 * n_total_src:
        _log.warning(
            "[%s] %.0f%% of frames used control_rollout fallback "
            "(%d / %d). Planner may not expose trajectory data.",
            sname_eval,
            100.0 * n_rollout / n_total_src,
            n_rollout,
            n_total_src,
        )
    if traj_mode == "planner_waypoints" and planner_waypoint_miss_count > 0:
        _log.warning(
            "[%s] planner_waypoints strict mode skipped %d frame(s) due to missing planner trajectory.",
            sname_eval,
            planner_waypoint_miss_count,
        )

    if _stale_info["n_stale"] > 0 and verbose:
        _log.info(
            "[%s] Stale controls detected: %d / %d frames (%.1f%%)",
            sname_eval,
            _stale_info["n_stale"],
            _stale_info["n_total"],
            100.0 * _stale_info["stale_fraction"],
        )

    if sanity_checks["pred_eq_gt_ade"] is not None and sanity_checks["pred_eq_gt_ade"] > 1e-6:
        _log.warning(
            "[%s] Sanity check failed: pred==gt ADE is %.6f (expected near 0)",
            sname_eval,
            sanity_checks["pred_eq_gt_ade"],
        )
    if (
        sanity_checks["pred_eq_gt_ade"] is not None
        and sanity_checks["gt_shifted_ade"] is not None
        and sanity_checks["gt_shifted_ade"] <= sanity_checks["pred_eq_gt_ade"]
    ):
        _log.warning(
            "[%s] Sanity check failed: shifted GT ADE (%.6f) did not increase over baseline (%.6f)",
            sname_eval,
            sanity_checks["gt_shifted_ade"],
            sanity_checks["pred_eq_gt_ade"],
        )

    # Summary print
    if verbose or True:  # always print per-scenario summary
        print(
            f"  [{sname_eval}]  "
            f"ADE={_primary_metrics.get('plan_ade', float('nan')):.4f}  "
            f"FDE={_primary_metrics.get('plan_fde', float('nan')):.4f}  "
            f"(debug rollout ADE={_debug_metrics.get('plan_ade', float('nan')):.4f})  "
            f"scoring={scoring_path}  "
            f"mode={traj_mode}  "
            f"sources={dict(traj_source_counts)}  "
            f"adapter_sources={dict(adapter_source_counts)}"
            )
        if scoring_path == "canonical_fresh_only":
            print(
                f"  [{sname_eval}] coverage: total={_coverage_stats['total_frames']}  "
                f"scored={_coverage_stats['scored_frames']}  "
                f"stale={_coverage_stats['skipped_stale']}  "
                f"unresolved={_coverage_stats['skipped_unresolved_semantics']}  "
                f"insufficient={_coverage_stats['skipped_insufficient_horizon']}"
            )
            _horizon_parts = [
                f"{k}={v:.2f}" if v is not None else f"{k}=n/a"
                for k, v in _per_horizon_ade.items()
            ]
            print(f"  [{sname_eval}] per-horizon ADE: {', '.join(_horizon_parts)}")
        print(
            f"  [{sname_eval}] branch-score(selected) ADE="
            f"{_branch_selected_metrics.get('plan_ade', float('nan')):.4f}  "
            f"FDE={_branch_selected_metrics.get('plan_fde', float('nan')):.4f}  "
            f"n={_branch_selected_metrics.get('n_frames', 0)}"
        )
        print(
            f"  [{sname_eval}] branch-score(oracle best-branch) ADE="
            f"{_branch_oracle_metrics.get('plan_ade', float('nan')):.4f}  "
            f"FDE={_branch_oracle_metrics.get('plan_fde', float('nan')):.4f}  "
            f"n={_branch_oracle_metrics.get('n_frames', 0)}"
        )
        if scoring_path == "canonical_fresh_only":
            print(
                f"  [{sname_eval}] branch-score(selected, canonical-scored) ADE="
                f"{_branch_selected_scored_metrics.get('plan_ade', float('nan')):.4f}  "
                f"FDE={_branch_selected_scored_metrics.get('plan_fde', float('nan')):.4f}  "
                f"n={_branch_selected_scored_metrics.get('n_frames', 0)}"
            )
            print(
                f"  [{sname_eval}] branch-score(oracle, canonical-scored) ADE="
                f"{_branch_oracle_scored_metrics.get('plan_ade', float('nan')):.4f}  "
                f"FDE={_branch_oracle_scored_metrics.get('plan_fde', float('nan')):.4f}  "
                f"n={_branch_oracle_scored_metrics.get('n_frames', 0)}"
            )

    result = {
        "scenario":  sname,
        "scenario_eval": sname_eval,
        "ego_actor_id": primary_actor_id,
        "metrics":   _primary_metrics,
        "legacy_metrics": _legacy_metrics,
        "canonical_metrics": _canonical_metrics,
        "debug_rollout_metrics": _debug_metrics,
        "branch_selected_metrics": _branch_selected_metrics,
        "branch_oracle_metrics": _branch_oracle_metrics,
        "branch_selected_scored_metrics": _branch_selected_scored_metrics,
        "branch_oracle_scored_metrics": _branch_oracle_scored_metrics,
        "traj_source_counts":   dict(traj_source_counts),
        "adapter_source_counts": dict(adapter_source_counts),
        "traj_mode": traj_mode,
        "scoring_path": scoring_path,
        "trajectory_extrapolation": bool(trajectory_extrapolation),
        "vad_oracle_branch_by_local_ade": bool(vad_oracle_branch_by_local_ade),
        "planner_waypoint_missing_frames": int(planner_waypoint_miss_count),
        "stale_control_info":   _stale_info,
        "coverage_stats": _coverage_stats,
        "per_horizon_ade": _per_horizon_ade,
        "sanity_checks":        sanity_checks,
        "frame_debug_file": (str(frame_debug_path) if frame_debug_path is not None else None),
        "raw_export_file": (str(raw_export_path) if raw_export_path is not None else None),
        "canonical_export_file": (
            str(canonical_export_path) if canonical_export_path is not None else None
        ),
        "scoring_debug_file": (
            str(scoring_debug_path) if scoring_debug_path is not None else None
        ),
        "per_frame": per_frame,
    }
    if vis_logger is not None:
        vis_logger.end_scenario(_primary_metrics, per_frame, sname_eval, primary_actor_id)
    return result


# ============================================================
# Main evaluation loop
# ============================================================

def evaluate_agent(
    *,
    agent_path:      str,
    agent_config:    str,
    data_dir:        str,
    map_pkl:         str,
    output_dir:      str,
    agent_label:     Optional[str] = None,
    agent_conda_env: Optional[str] = None,
    agent_entry_point: Optional[str] = None,
    scenario_names:  Optional[List[str]] = None,
    actor_id:        Optional[int] = None,
    num_ego:         int = 1,
    verbose:         bool = False,
    viz_dir:         Optional[str] = None,
    viz_every:       int = 1,
    viz_style:       str = "basic",
    viz_video:       Optional[bool] = None,
    video_fps:       Optional[float] = None,
    video_codec:     str = "mp4v",
    panel_width:     int = 1920,
    max_frames:      Optional[int] = None,
    traj_source:     str = "auto",
    trajectory_extrapolation: bool = True,
    town_id:         Optional[str] = None,
    dump_frame_debug: bool = True,
    richviz:         bool = False,
    richviz_dir:     Optional[str] = None,
    richviz_all_frames: bool = False,
    richviz_frame_stride: Optional[int] = None,
    richviz_max_frames: int = 40,
    planner_name_hint: Optional[str] = None,
    native_horizon:  bool = True,
    vad_oracle_branch_by_local_ade: bool = True,
    vad_force_fresh_inference: bool = True,
    skip_stale_frames: bool = True,
    scoring_path: str = "canonical_fresh_only",
    canonical_horizon_s: float = 3.0,
) -> Dict[str, Any]:
    """
    Evaluate a single leaderboard agent on the v2xpnp dataset.

    Parameters
    ----------
    agent_path      : module path implementing the leaderboard agent
    agent_config    : config path passed to agent.setup(...)
    data_dir        : root directory of v2xpnp scenario dataset
    map_pkl         : path to v2v_corridors_vector_map.pkl
    output_dir      : where to save JSON results
    scenario_names  : restrict to these scenario directory names (or None = all)
    actor_id        : actor id to evaluate within each scenario (default: all actors)
    num_ego         : number of ego vehicles to register
    verbose         : print detailed per-frame sensor + prediction diagnostics
    viz_dir         : directory to save PNG visualizations (None = no viz)
    viz_every       : save BEV frame image every this many frames
    viz_style       : visualization style ("basic" or "rich")
    viz_video       : in rich mode, whether to export mp4 (None => enabled)
    video_fps       : output video fps (None => scenario fps)
    video_codec     : OpenCV fourcc codec, e.g. "mp4v"
    panel_width     : rich composite frame width in pixels
    max_frames      : limit frames per scenario (None = all frames)
    traj_source     : trajectory source policy:
                      - control_rollout: always control rollout.
                      - planner_waypoints: strict adapter-only.
                      - auto: adapter with rollout fallback.
    trajectory_extrapolation:
                      when False, adapter resampling will not extrapolate
                      beyond native planner horizon (holds last waypoint).
    richviz         : run planner_inspector_viz automatically from dumped
                      frame-debug JSONL files.
    vad_oracle_branch_by_local_ade:
                      VAD-only debug mode that uses GT to select the minimum
                      local-ADE branch each frame (oracle branch selection).
    vad_force_fresh_inference:
                      Open-loop override: force fresh planner inference on
                      every fed frame at the evaluator's 10 Hz cadence.
                      This only applies inside openloop_eval.

    Returns
    -------
    dict with "agent", "overall" metrics, and per-scenario "scenarios" list.
    """
    _ensure_openloop_runtime()
    agent_label = agent_label or Path(agent_path).stem
    result_stem = sanitize_run_id(agent_label)
    output_path = Path(output_dir)
    frame_debug_dir = (
        output_path / "debug_frames" / result_stem
        if dump_frame_debug
        else None
    )
    raw_export_dir = (
        output_path / "raw_exports" / result_stem
        if dump_frame_debug
        else None
    )
    canonical_export_dir = (
        output_path / "canonical_exports" / result_stem
        if dump_frame_debug
        else None
    )
    scoring_debug_dir = (
        output_path / "scoring_debug" / result_stem
        if dump_frame_debug
        else None
    )

    print(f"\n{'='*60}")
    print(f"  Open-loop evaluation  (uses opencood metric formula)")
    print(f"  Agent   : {agent_label}")
    print(f"  Module  : {agent_path}")
    print(f"  Config  : {agent_config}")
    if agent_conda_env:
        print(f"  Env     : {agent_conda_env}")
    print(f"  Data    : {data_dir}")
    print(f"  Output  : {output_dir}")
    print(f"  Viz dir : {viz_dir or '(disabled)'}")
    print(f"  Viz mode: {viz_style}")
    _traj_mode = str(traj_source or "auto").strip().lower()
    print(f"  Traj src mode: {_traj_mode}")
    print(
        f"  Traj extrapolation: "
        f"{'enabled' if trajectory_extrapolation else 'disabled'}"
    )
    print(
        f"  VAD oracle branch (min local ADE): "
        f"{'enabled' if vad_oracle_branch_by_local_ade else 'disabled'}"
    )
    print(
        "  Planner fresh inference override (open-loop, all planners): "
        f"{'enabled' if vad_force_fresh_inference else 'disabled'}"
    )
    print(f"  Frame debug dump: {str(frame_debug_dir) if frame_debug_dir else '(disabled)'}")
    if richviz:
        planned_richviz_dir = richviz_dir or str((Path(output_dir) / "richviz").resolve())
        print(f"  Integrated richviz: enabled  -> {planned_richviz_dir}")
    else:
        print("  Integrated richviz: disabled")
    print(f"  Raw export dump : {str(raw_export_dir) if raw_export_dir else '(disabled)'}")
    print(f"  Canonical dump  : {str(canonical_export_dir) if canonical_export_dir else '(disabled)'}")
    print(f"  Scoring debug   : {str(scoring_debug_dir) if scoring_debug_dir else '(disabled)'}")
    print(f"  Scoring path    : {scoring_path}")
    print(f"  matplotlib: {'available' if _MPL_AVAILABLE else 'NOT available — no plots'}")
    print(f"{'='*60}")

    # ---- Load dataset ----
    dataset = V2XPnPOpenLoopDataset(data_dir, scenario_names=scenario_names)
    if len(dataset) == 0:
        raise RuntimeError(f"No valid scenarios found in {data_dir}")
    print(f"Found {len(dataset)} scenario(s).")

    # ---- Create vis logger ----
    vis_logger = _VisLogger(
        planner_name = agent_label,
        viz_dir      = viz_dir,
        viz_every    = viz_every,
        verbose      = verbose,
        viz_style    = viz_style,
        viz_video    = viz_video,
        video_fps    = video_fps,
        video_codec  = video_codec,
        panel_width  = panel_width,
    )
    effective_town_id = town_id or (Path(map_pkl).stem if map_pkl else None)

    try:
        # ---- Per-scenario evaluation ----
        global_accum = MetricAccumulator()
        global_legacy_accum = MetricAccumulator()
        global_branch_selected_accum = MetricAccumulator()
        global_branch_oracle_accum = MetricAccumulator()
        global_branch_selected_scored_accum = MetricAccumulator()
        global_branch_oracle_scored_accum = MetricAccumulator()
        global_coverage = CanonicalMetricAccumulator()
        scenario_results: List[Dict] = []

        for idx, scenario in enumerate(dataset.iter_scenarios()):
            sname = scenario.scenario_dir.name
            scenario_actor_ids = scenario.actor_ids
            if actor_id is None:
                eval_actor_ids = list(scenario_actor_ids)
            else:
                eval_actor_ids = [int(actor_id)]
            missing = [aid for aid in eval_actor_ids if aid not in scenario_actor_ids]
            if missing:
                raise ValueError(
                    f"Actor id(s) {missing} not found in scenario {sname}. "
                    f"Available actors: {scenario_actor_ids}"
                )
            for eval_actor in eval_actor_ids:
                print(f"\n[{idx+1}/{len(dataset)}] {sname}  "
                      f"actors={scenario_actor_ids}  "
                      f"eval_actor={eval_actor}  "
                      f"frames={scenario.num_frames(eval_actor)}")

                prior_routes_env = os.environ.get("ROUTES")
                prior_vad_force_env = os.environ.get("VAD_FORCE_FRESH_INFERENCE")
                prior_uniad_force_env = os.environ.get("UNIAD_FORCE_FRESH_INFERENCE")
                prior_openloop_force_env = os.environ.get("OPENLOOP_FORCE_FRESH_INFERENCE")
                os.environ["ROUTES"] = (
                    f"openloop_{sanitize_run_id(sname)}_ego{int(eval_actor)}.xml"
                )
                if vad_force_fresh_inference:
                    os.environ["VAD_FORCE_FRESH_INFERENCE"] = "1"
                    os.environ["UNIAD_FORCE_FRESH_INFERENCE"] = "1"
                    os.environ["OPENLOOP_FORCE_FRESH_INFERENCE"] = "1"
                else:
                    os.environ.pop("VAD_FORCE_FRESH_INFERENCE", None)
                    os.environ.pop("UNIAD_FORCE_FRESH_INFERENCE", None)
                    os.environ.pop("OPENLOOP_FORCE_FRESH_INFERENCE", None)
                try:
                    print(f"Loading agent '{agent_label}'...")
                    try:
                        agent = _build_agent(
                            agent_path=agent_path,
                            agent_config=agent_config,
                            num_ego_vehicles=num_ego,
                            entry_class_name=agent_entry_point,
                        )
                    except Exception as exc:
                        print(f"ERROR: could not load agent: {exc}")
                        traceback.print_exc()
                        raise
                    print("Agent loaded.")

                    try:
                        result = evaluate_scenario(
                            agent        = agent,
                            scenario     = scenario,
                            hero_vehicles = _MockCarlaDataProvider._hero_actors,
                            eval_actor_id = eval_actor,
                            verbose      = verbose,
                            vis_logger   = vis_logger,
                            max_frames   = max_frames,
                            traj_source  = traj_source,
                            trajectory_extrapolation = trajectory_extrapolation,
                            town_id      = effective_town_id,
                            planner_name = agent_label,
                            frame_debug_dir = frame_debug_dir,
                            raw_export_dir = raw_export_dir,
                            canonical_export_dir = canonical_export_dir,
                            scoring_debug_dir = scoring_debug_dir,
                            native_horizon = native_horizon,
                            vad_oracle_branch_by_local_ade = vad_oracle_branch_by_local_ade,
                            skip_stale_frames = skip_stale_frames,
                            scoring_path = scoring_path,
                        )
                    finally:
                        destroy_fn = getattr(agent, "destroy", None)
                        if callable(destroy_fn):
                            try:
                                destroy_fn()
                            except Exception as exc:
                                print(f"[WARN] Agent '{agent_label}' destroy() failed: {exc}")
                finally:
                    if prior_vad_force_env is None:
                        os.environ.pop("VAD_FORCE_FRESH_INFERENCE", None)
                    else:
                        os.environ["VAD_FORCE_FRESH_INFERENCE"] = prior_vad_force_env
                    if prior_uniad_force_env is None:
                        os.environ.pop("UNIAD_FORCE_FRESH_INFERENCE", None)
                    else:
                        os.environ["UNIAD_FORCE_FRESH_INFERENCE"] = prior_uniad_force_env
                    if prior_openloop_force_env is None:
                        os.environ.pop("OPENLOOP_FORCE_FRESH_INFERENCE", None)
                    else:
                        os.environ["OPENLOOP_FORCE_FRESH_INFERENCE"] = prior_openloop_force_env
                    if prior_routes_env is None:
                        os.environ.pop("ROUTES", None)
                    else:
                        os.environ["ROUTES"] = prior_routes_env

                scenario_results.append(result)

                # Accumulate into global stats
                m = result["metrics"]
                for _ in range(m["n_frames"]):
                    global_accum.plan_ade_sum    += m["plan_ade"]
                    global_accum.plan_fde_sum    += m["plan_fde"]
                    global_accum.collision_count += m["collision_rate"]
                    global_accum.n_frames        += 1
                lm = result.get("legacy_metrics", {})
                for _ in range(lm.get("n_frames", 0)):
                    global_legacy_accum.plan_ade_sum += lm["plan_ade"]
                    global_legacy_accum.plan_fde_sum += lm["plan_fde"]
                    global_legacy_accum.collision_count += lm["collision_rate"]
                    global_legacy_accum.n_frames += 1
                bsm = result.get("branch_selected_metrics", {})
                for _ in range(bsm.get("n_frames", 0)):
                    global_branch_selected_accum.plan_ade_sum += bsm["plan_ade"]
                    global_branch_selected_accum.plan_fde_sum += bsm["plan_fde"]
                    global_branch_selected_accum.collision_count += bsm["collision_rate"]
                    global_branch_selected_accum.n_frames += 1
                bom = result.get("branch_oracle_metrics", {})
                for _ in range(bom.get("n_frames", 0)):
                    global_branch_oracle_accum.plan_ade_sum += bom["plan_ade"]
                    global_branch_oracle_accum.plan_fde_sum += bom["plan_fde"]
                    global_branch_oracle_accum.collision_count += bom["collision_rate"]
                    global_branch_oracle_accum.n_frames += 1
                bssm = result.get("branch_selected_scored_metrics", {})
                for _ in range(bssm.get("n_frames", 0)):
                    global_branch_selected_scored_accum.plan_ade_sum += bssm["plan_ade"]
                    global_branch_selected_scored_accum.plan_fde_sum += bssm["plan_fde"]
                    global_branch_selected_scored_accum.collision_count += bssm["collision_rate"]
                    global_branch_selected_scored_accum.n_frames += 1
                bosm = result.get("branch_oracle_scored_metrics", {})
                for _ in range(bosm.get("n_frames", 0)):
                    global_branch_oracle_scored_accum.plan_ade_sum += bosm["plan_ade"]
                    global_branch_oracle_scored_accum.plan_fde_sum += bosm["plan_fde"]
                    global_branch_oracle_scored_accum.collision_count += bosm["collision_rate"]
                    global_branch_oracle_scored_accum.n_frames += 1
                cov = result.get("coverage_stats", {})
                global_coverage.total_frames += int(cov.get("total_frames", 0))
                global_coverage.scored_frames += int(cov.get("scored_frames", 0))
                global_coverage.skipped_stale += int(cov.get("skipped_stale", 0))
                global_coverage.skipped_unresolved_semantics += int(cov.get("skipped_unresolved_semantics", 0))
                global_coverage.skipped_insufficient_horizon += int(cov.get("skipped_insufficient_horizon", 0))
                global_coverage.skipped_missing_canonical += int(cov.get("skipped_missing_canonical", 0))
                global_coverage.skipped_run_step_exception += int(cov.get("skipped_run_step_exception", 0))
                global_coverage.skipped_other += int(cov.get("skipped_other", 0))
                global_coverage.n_frames += int(cov.get("scored_frames", 0))

        # Aggregate debug rollout metrics across all scenarios.
        global_debug_rollout = MetricAccumulator()
        global_traj_source_counts: Dict[str, int] = {}
        global_adapter_source_counts: Dict[str, int] = {}
        global_stale_total = 0
        global_stale_count = 0
        for sr in scenario_results:
            dm = sr.get("debug_rollout_metrics", {})
            nf = dm.get("n_frames", 0)
            if nf > 0:
                for _ in range(nf):
                    global_debug_rollout.plan_ade_sum += dm.get("plan_ade", 0)
                    global_debug_rollout.plan_fde_sum += dm.get("plan_fde", 0)
                    global_debug_rollout.n_frames += 1
            for src, cnt in sr.get("traj_source_counts", {}).items():
                global_traj_source_counts[src] = (
                    global_traj_source_counts.get(src, 0) + cnt
                )
            for src, cnt in sr.get("adapter_source_counts", {}).items():
                global_adapter_source_counts[src] = (
                    global_adapter_source_counts.get(src, 0) + cnt
                )
            si = sr.get("stale_control_info", {})
            global_stale_total += si.get("n_total", 0)
            global_stale_count += si.get("n_stale", 0)

        overall = global_accum.summary()
        legacy_overall = global_legacy_accum.summary()
        branch_selected_overall = global_branch_selected_accum.summary()
        branch_oracle_overall = global_branch_oracle_accum.summary()
        branch_selected_scored_overall = global_branch_selected_scored_accum.summary()
        branch_oracle_scored_overall = global_branch_oracle_scored_accum.summary()
        debug_rollout_overall = global_debug_rollout.summary()
        coverage_summary = global_coverage.coverage_summary()

        # Aggregate per-horizon ADE across all scenarios from per-frame data.
        from canonical_trajectory import CANONICAL_TIMESTAMPS

        global_per_step_sum = [0.0] * len(CANONICAL_TIMESTAMPS)
        global_per_step_count = [0] * len(CANONICAL_TIMESTAMPS)
        for sr in scenario_results:
            for fm in sr.get("per_frame", []):
                breakdown = fm.get("ade_breakdown_m")
                if not isinstance(breakdown, (list, tuple, np.ndarray)):
                    continue
                if len(breakdown) == 0:
                    continue
                for idx in range(min(len(global_per_step_sum), len(breakdown))):
                    val = breakdown[idx]
                    if val is not None and isinstance(val, (int, float)):
                        global_per_step_sum[idx] += float(val)
                        global_per_step_count[idx] += 1
        global_per_horizon_ade: Dict[str, Any] = {}
        for idx, ts in enumerate(CANONICAL_TIMESTAMPS):
            key = f"{float(ts):.1f}s"
            if global_per_step_count[idx] > 0:
                global_per_horizon_ade[key] = round(
                    global_per_step_sum[idx] / global_per_step_count[idx], 4
                )
            else:
                global_per_horizon_ade[key] = None

        print(f"\n{'='*60}")
        print(f"  OVERALL  [{agent_label}]")
        print(f"  Scoring path    = {scoring_path}")
        print(f"  Plan ADE        = {overall['plan_ade']:.4f} m")
        print(f"  Plan FDE        = {overall['plan_fde']:.4f} m")
        print(f"  Collision Rate  = {overall['collision_rate']:.4f}")
        print(f"  Total frames    = {overall['n_frames']}")
        if scoring_path == "canonical_fresh_only":
            print(
                f"  Coverage        = total={coverage_summary['total_frames']}  "
                f"scored={coverage_summary['scored_frames']}  "
                f"stale={coverage_summary['skipped_stale']}  "
                f"unresolved={coverage_summary['skipped_unresolved_semantics']}  "
                f"insufficient={coverage_summary['skipped_insufficient_horizon']}"
            )
            print(f"  Legacy ADE      = {legacy_overall['plan_ade']:.4f} m")
            print(f"  Legacy FDE      = {legacy_overall['plan_fde']:.4f} m")
        _horizon_parts_global = [
            f"{k}={v:.2f}" if v is not None else f"{k}=n/a"
            for k, v in global_per_horizon_ade.items()
        ]
        print(f"  Per-horizon ADE = {', '.join(_horizon_parts_global)}")
        print(
            f"  Branch Selected ADE/FDE = "
            f"{branch_selected_overall['plan_ade']:.4f} / {branch_selected_overall['plan_fde']:.4f} m "
            f"(n={branch_selected_overall['n_frames']})"
        )
        print(
            f"  Branch Oracle ADE/FDE   = "
            f"{branch_oracle_overall['plan_ade']:.4f} / {branch_oracle_overall['plan_fde']:.4f} m "
            f"(n={branch_oracle_overall['n_frames']})"
        )
        if scoring_path == "canonical_fresh_only":
            print(
                f"  Branch Selected ADE/FDE (canonical-scored) = "
                f"{branch_selected_scored_overall['plan_ade']:.4f} / "
                f"{branch_selected_scored_overall['plan_fde']:.4f} m "
                f"(n={branch_selected_scored_overall['n_frames']})"
            )
            print(
                f"  Branch Oracle ADE/FDE   (canonical-scored) = "
                f"{branch_oracle_scored_overall['plan_ade']:.4f} / "
                f"{branch_oracle_scored_overall['plan_fde']:.4f} m "
                f"(n={branch_oracle_scored_overall['n_frames']})"
            )
        print(f"  Traj sources    = {dict(global_traj_source_counts)}")
        print(f"  Adapter sources = {dict(global_adapter_source_counts)}")
        print(f"  Debug rollout ADE = {debug_rollout_overall.get('plan_ade', float('nan')):.4f} m")
        print(f"  Debug rollout FDE = {debug_rollout_overall.get('plan_fde', float('nan')):.4f} m")
        if global_stale_total > 0:
            print(f"  Stale controls  = {global_stale_count}/{global_stale_total} "
                  f"({100.0 * global_stale_count / global_stale_total:.1f}%)")
        print(f"  Metric formula  : opencood/utils/eval_utils.py")
        print(f"{'='*60}")

        out = {
            "agent": agent_path,
            "agent_config": agent_config,
            "agent_entry_point": agent_entry_point,
            "agent_label": agent_label,
            "agent_conda_env": agent_conda_env,
            # Preserve legacy keys for existing tooling.
            "planner": agent_path,
            "planner_label": agent_label,
            "planner_conda_env": agent_conda_env,
            "overall":   overall,
            "per_horizon_ade": global_per_horizon_ade,
            "legacy_overall": legacy_overall,
            "branch_selected_overall": branch_selected_overall,
            "branch_oracle_overall": branch_oracle_overall,
            "branch_selected_scored_overall": branch_selected_scored_overall,
            "branch_oracle_scored_overall": branch_oracle_scored_overall,
            "debug_rollout_overall": debug_rollout_overall,
            "traj_source_counts": dict(global_traj_source_counts),
            "adapter_source_counts": dict(global_adapter_source_counts),
            "scoring_path": scoring_path,
            "trajectory_extrapolation": bool(trajectory_extrapolation),
            "canonical_horizon_s": float(canonical_horizon_s),
            "native_horizon": bool(native_horizon),
            "vad_oracle_branch_by_local_ade": bool(vad_oracle_branch_by_local_ade),
            "vad_force_fresh_inference": bool(vad_force_fresh_inference),
            "skip_stale_frames": bool(skip_stale_frames),
            "frame_debug_dir": (str(frame_debug_dir) if frame_debug_dir is not None else None),
            "raw_export_dir": (str(raw_export_dir) if raw_export_dir is not None else None),
            "canonical_export_dir": (
                str(canonical_export_dir) if canonical_export_dir is not None else None
            ),
            "scoring_debug_dir": (
                str(scoring_debug_dir) if scoring_debug_dir is not None else None
            ),
            "stale_control_summary": {
                "n_total": global_stale_total,
                "n_stale": global_stale_count,
                "stale_fraction": global_stale_count / max(1, global_stale_total),
            },
            "coverage_summary": coverage_summary,
            "scenarios": scenario_results,
        }

        output_path = Path(output_dir)
        richviz_outputs: List[Dict[str, Any]] = []
        resolved_richviz_dir = None
        if richviz:
            if not dump_frame_debug:
                print("  [richviz] skipped: --richviz requires --dump-frame-debug.")
            else:
                resolved_richviz_dir = richviz_dir or str((output_path / "richviz").resolve())
                richviz_outputs = _run_integrated_richviz_generation(
                    planner_name=(planner_name_hint or agent_label),
                    scenario_results=scenario_results,
                    richviz_dir=resolved_richviz_dir,
                    richviz_all_frames=bool(richviz_all_frames),
                    richviz_frame_stride=richviz_frame_stride,
                    richviz_max_frames=int(richviz_max_frames),
                )

        # ---- Save results ----
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"{result_stem}_openloop.json"
        out["integrated_richviz"] = {
            "enabled": bool(richviz),
            "richviz_dir": (str(resolved_richviz_dir) if resolved_richviz_dir else None),
            "all_frames": bool(richviz_all_frames),
            "frame_stride": (None if richviz_frame_stride is None else int(richviz_frame_stride)),
            "max_frames": int(richviz_max_frames),
            "outputs": richviz_outputs,
        }
        with open(result_file, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"Results saved → {result_file}")

        return out
    finally:
        try:
            vis_logger._close_video_writer()
        except Exception:
            pass


# ============================================================
# CLI
# ============================================================

def _planner_uses_colmdriver_vllm(planner_name: str) -> bool:
    return planner_name in COLMDRIVER_VLLM_PLANNERS


def _planner_request_uses_colmdriver_vllm(planner_request: PlannerRequest) -> bool:
    preset_name = str(planner_request.preset_name or "")
    return _planner_uses_colmdriver_vllm(preset_name)


def _resolve_abs_path(path_value: Optional[str]) -> Optional[str]:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((_PROJECT_ROOT / path).resolve())


def _resolve_output_dir(args: argparse.Namespace, planner_label: str) -> str:
    if args.output:
        return _resolve_abs_path(args.output) or str(_PROJECT_ROOT)
    return str((_PROJECT_ROOT / "results" / f"openloop_{sanitize_run_id(planner_label)}").resolve())


def _resolve_viz_dir(args: argparse.Namespace, output_dir: str) -> Optional[str]:
    if args.no_viz:
        return None
    if args.viz_dir:
        return _resolve_abs_path(args.viz_dir)
    return str((Path(output_dir) / "viz").resolve())


def _resolve_richviz_dir(args: argparse.Namespace, output_dir: str) -> str:
    if args.richviz_dir:
        return _resolve_abs_path(args.richviz_dir) or str(Path(args.richviz_dir).resolve())
    return str((Path(output_dir) / "richviz").resolve())


def _normalize_inspector_planner_name(planner_name: Optional[str]) -> Optional[str]:
    tag = str(planner_name or "").strip().lower()
    if not tag:
        return "generic"
    if "colmdriver_rulebase" in tag:
        return "colmdriver_rulebase"
    if "colmdriver" in tag:
        return "colmdriver"
    if "vad" in tag:
        return "vad"
    if "tcp" in tag:
        return "tcp"
    # planner_inspector_viz supports generic rendering for unknown planners.
    return tag


def _run_integrated_richviz_generation(
    *,
    planner_name: Optional[str],
    scenario_results: List[Dict[str, Any]],
    richviz_dir: str,
    richviz_all_frames: bool,
    richviz_frame_stride: Optional[int],
    richviz_max_frames: int,
) -> List[Dict[str, Any]]:
    """Generate planner_inspector_viz outputs from dumped frame-debug JSONL files."""
    planner_mode = _normalize_inspector_planner_name(planner_name)
    outputs: List[Dict[str, Any]] = []

    inspector_script = (_OPENLOOP_ROOT / "tools" / "planner_inspector_viz.py").resolve()
    if not inspector_script.exists():
        print(f"  [richviz] skipped: inspector script not found at {inspector_script}")
        return outputs

    base_out = Path(richviz_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    print(f"  [richviz] generating integrated inspector plots → {base_out}")
    for sr in scenario_results:
        debug_frames_file = sr.get("frame_debug_file")
        if not debug_frames_file:
            continue
        debug_path = Path(str(debug_frames_file))
        if not debug_path.exists():
            outputs.append({
                "debug_frames_file": str(debug_path),
                "status": "missing_debug_frames_file",
            })
            continue

        scenario_eval = str(sr.get("scenario_eval") or debug_path.stem)
        out_dir = base_out / sanitize_run_id(scenario_eval)
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(inspector_script),
            "--debug-frames",
            str(debug_path),
            "--planner",
            str(planner_mode),
            "--out-dir",
            str(out_dir),
            "--max-frames",
            str(max(1, int(richviz_max_frames))),
        ]
        if richviz_all_frames:
            cmd.append("--all-frames")
        elif richviz_frame_stride is not None:
            cmd.extend(["--frame-stride", str(max(1, int(richviz_frame_stride)))])

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(_PROJECT_ROOT),
                capture_output=True,
                text=True,
            )
            status = "ok" if proc.returncode == 0 else "failed"
            if proc.returncode == 0:
                print(f"  [richviz] OK  {scenario_eval} -> {out_dir}")
            else:
                tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or [""]
                print(f"  [richviz] FAIL {scenario_eval}: {tail[0]}")
            outputs.append({
                "scenario_eval": scenario_eval,
                "debug_frames_file": str(debug_path),
                "out_dir": str(out_dir),
                "status": status,
                "returncode": int(proc.returncode),
            })
        except Exception as exc:
            print(f"  [richviz] FAIL {scenario_eval}: {exc}")
            outputs.append({
                "scenario_eval": scenario_eval,
                "debug_frames_file": str(debug_path),
                "out_dir": str(out_dir),
                "status": "failed_exception",
                "error": str(exc),
            })
    return outputs


def _push_colmdriver_port_env(planner_name: str, llm_port: int, vlm_port: int) -> Dict[str, Optional[str]]:
    saved = {
        "COLMDRIVER_LLM_PORT": os.environ.get("COLMDRIVER_LLM_PORT"),
        "COLMDRIVER_VLM_PORT": os.environ.get("COLMDRIVER_VLM_PORT"),
    }
    if _planner_uses_colmdriver_vllm(planner_name):
        os.environ["COLMDRIVER_LLM_PORT"] = str(int(llm_port))
        os.environ["COLMDRIVER_VLM_PORT"] = str(int(vlm_port))
    return saved


def _restore_colmdriver_port_env(saved: Dict[str, Optional[str]]) -> None:
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _resolve_requested_scheduler_gpus(args: argparse.Namespace) -> List[str]:
    requested = _parse_gpu_tokens(args.gpus)
    if not requested:
        requested = _detect_visible_gpu_tokens()
    deduped: List[str] = []
    for gpu in requested:
        if gpu not in deduped:
            deduped.append(gpu)
    if len(deduped) != len(requested):
        raise RuntimeError("--gpus must contain distinct GPU ids/UUIDs.")
    return deduped


def _refresh_scheduler_gpu_states(
    gpu_states: Dict[Optional[str], GPUPlannerState],
) -> Dict[Optional[str], GPUPlannerState]:
    has_real_gpu = any(state.gpu is not None for state in gpu_states.values())
    if not has_real_gpu:
        for state in gpu_states.values():
            state.launch_capacity = 1
            state.telemetry_available = False
            state.last_stats = None
            state.last_probe_source = "no_gpu"
        return gpu_states

    for state in gpu_states.values():
        if state.gpu is None:
            state.launch_capacity = 1
            state.telemetry_available = False
            state.last_stats = None
            state.last_probe_source = "no_gpu"
            continue
        refresh_gpu_launch_capacity(
            state,
            min_free_mib=AUTO_WORKERS_PER_GPU_MIN_FREE_MIB_DEFAULT,
            settle_timeout_s=AUTO_WORKERS_PER_GPU_SETTLE_TIMEOUT_DEFAULT,
            poll_seconds=AUTO_WORKERS_PER_GPU_POLL_SECONDS_DEFAULT,
            stable_samples=AUTO_WORKERS_PER_GPU_STABLE_SAMPLES_DEFAULT,
            stable_tolerance_mib=AUTO_WORKERS_PER_GPU_STABLE_TOLERANCE_MIB_DEFAULT,
        )

    if not any(
        state.launch_capacity > state.active_count
        for state in gpu_states.values()
        if state.gpu is not None
    ):
        forced_state = choose_force_launch_gpu(gpu_states)
        forced_state.launch_capacity = max(
            forced_state.launch_capacity,
            forced_state.active_count + 1,
        )
    return gpu_states


def _select_launch_gpu_state(
    gpu_states: Dict[Optional[str], GPUPlannerState],
) -> Optional[GPUPlannerState]:
    candidates = [
        state
        for state in gpu_states.values()
        if state.active_count < state.launch_capacity
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda state: (
            state.last_stats.free_mib if state.last_stats is not None else -1,
            -(state.active_count),
        ),
    )


def _start_local_colmdriver_vllm_if_needed(
    args: argparse.Namespace,
    planner_name: str,
    output_dir: str,
) -> Optional[ColmDriverVLLMManager]:
    if not args.start_colmdriver_vllm or not _planner_uses_colmdriver_vllm(planner_name):
        return None
    requested_gpus = _resolve_requested_scheduler_gpus(args)
    if not requested_gpus:
        raise RuntimeError(
            "Could not determine GPUs for CoLMDriver vLLM startup. Provide "
            "--gpus or set CUDA_VISIBLE_DEVICES."
        )
    reservation = select_colmdriver_vllm_gpus(requested_gpus)
    manager = ColmDriverVLLMManager(
        repo_root=_PROJECT_ROOT,
        base_env=os.environ,
        conda_env=str(args.colmdriver_vllm_env),
        reservation=reservation,
        llm_port=int(args.llm_port),
        vlm_port=int(args.vlm_port),
        startup_timeout_s=float(args.colmdriver_vllm_startup_timeout),
        log_dir=Path(output_dir) / "services",
    )
    manager.start()
    _register_local_vllm_manager(manager)
    print(
        "[INFO] CoLMDriver vLLM started with "
        f"VLM gpu={reservation.vlm_gpu} port={args.vlm_port} and "
        f"LLM gpu={reservation.llm_gpu} port={args.llm_port}."
    )
    return manager


class PlannerChildProcess:
    def __init__(
        self,
        *,
        planner_name: str,
        cmd: List[str],
        env: Dict[str, str],
        cwd: Path,
        log_path: Path,
        result_file: Path,
        vllm_manager: Optional[ColmDriverVLLMManager] = None,
        reservation: Optional[ColmDriverVLLMReservation] = None,
        llm_port: Optional[int] = None,
        vlm_port: Optional[int] = None,
    ) -> None:
        self.planner_name = str(planner_name)
        self.cmd = list(cmd)
        self.env = dict(env)
        self.cwd = str(cwd)
        self.log_path = Path(log_path)
        self.result_file = Path(result_file)
        self.vllm_manager = vllm_manager
        self.reservation = reservation
        self.llm_port = None if llm_port is None else int(llm_port)
        self.vlm_port = None if vlm_port is None else int(vlm_port)
        self.process: Optional[subprocess.Popen[str]] = None
        self.returncode: Optional[int] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._log_handle = None
        self._finalized = False

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("a", encoding="utf-8")
        self._log_handle.write(
            f"\n=== PLANNER START {dt.datetime.now().isoformat(timespec='seconds')} "
            f"planner={self.planner_name} ===\n"
        )
        self._log_handle.write("COMMAND: " + " ".join(self.cmd) + "\n")
        self._log_handle.flush()
        _register_runtime_child(self)
        try:
            if self.vllm_manager is not None:
                self.vllm_manager.start()
            self.process = subprocess.Popen(
                self.cmd,
                cwd=self.cwd,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except Exception:
            self._finalize()
            raise

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        print(
            f"[INFO] Started planner child '{self.planner_name}'"
            + (
                f" (pid={self.process.pid})"
                if self.process is not None
                else ""
            )
        )

    def _reader_loop(self) -> None:
        proc = self.process
        if proc is None or proc.stdout is None:
            return
        try:
            for raw_line in iter(proc.stdout.readline, ""):
                if raw_line == "":
                    break
                line = raw_line.rstrip("\n")
                if self._log_handle is not None:
                    self._log_handle.write(raw_line)
                    self._log_handle.flush()
                sys.stdout.write(f"[{self.planner_name}] {line}\n")
                sys.stdout.flush()
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass

    def poll(self) -> Optional[int]:
        if self.process is None:
            return self.returncode
        return self.process.poll()

    def wait(self) -> int:
        if self.process is None:
            self._finalize()
            return 1 if self.returncode is None else int(self.returncode)
        self.returncode = self.process.wait()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=5.0)
        self._finalize()
        return int(self.returncode)

    def terminate(self, reason: Optional[str] = None) -> None:
        if reason:
            print(reason)
            if self._log_handle is not None:
                self._log_handle.write(reason.rstrip("\n") + "\n")
                self._log_handle.flush()
        if self.process is not None and self.process.poll() is None:
            terminate_process_tree(self.process, grace_s=15.0)
            try:
                self.returncode = self.process.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                pass
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=5.0)
        self._finalize()

    def _finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        if self.vllm_manager is not None:
            try:
                self.vllm_manager.stop()
            except Exception as exc:
                print(f"[WARN] Failed to stop vLLM manager for {self.planner_name}: {exc}")
            finally:
                self.vllm_manager = None
        if self._log_handle is not None:
            self._log_handle.write(
                f"=== PLANNER END {dt.datetime.now().isoformat(timespec='seconds')} "
                f"planner={self.planner_name} returncode={self.returncode} ===\n"
            )
            self._log_handle.flush()
            self._log_handle.close()
            self._log_handle = None
        _unregister_runtime_child(self)


def _build_child_command(
    args: argparse.Namespace,
    *,
    planner_request: PlannerRequest,
    output_dir: str,
    viz_dir: Optional[str],
    llm_port: int,
    vlm_port: int,
    richviz_dir_override: Optional[str] = None,
) -> List[str]:
    child_argv = [
        str(_THIS_FILE),
        "--child-runner",
        "--agent-path",
        str(planner_request.agent),
        "--agent-config",
        str(planner_request.agent_config),
        "--agent-label",
        str(planner_request.run_id or planner_request.name),
        "--data-dir",
        _resolve_abs_path(args.data_dir) or args.data_dir,
        "--output",
        output_dir,
        "--ego-num",
        str(int(args.ego_num)),
        "--viz-every",
        str(int(args.viz_every)),
        "--viz-style",
        str(args.viz_style),
        "--video-codec",
        str(args.video_codec),
        "--panel-width",
        str(int(args.panel_width)),
        "--traj-source",
        str(args.traj_source),
        "--scoring-path",
        str(args.scoring_path),
        "--llm-port",
        str(int(llm_port)),
        "--vlm-port",
        str(int(vlm_port)),
        "--colmdriver-vllm-env",
        str(args.colmdriver_vllm_env),
        "--colmdriver-vllm-startup-timeout",
        str(float(args.colmdriver_vllm_startup_timeout)),
        "--planner-run-id",
        str(planner_request.run_id or planner_request.name),
    ]
    if planner_request.entry_point:
        child_argv.extend(["--agent-entry-point", str(planner_request.entry_point)])
    if planner_request.conda_env:
        child_argv.extend(["--agent-conda-env-name", str(planner_request.conda_env)])
    map_pkl = _resolve_abs_path(args.map_pkl)
    if map_pkl:
        child_argv.extend(["--map-pkl", map_pkl])
    if args.scenarios:
        child_argv.extend(["--scenarios", *list(args.scenarios)])
    if args.actor_id is not None:
        child_argv.extend(["--actor-id", str(int(args.actor_id))])
    if args.verbose:
        child_argv.append("--verbose")
    if viz_dir is None:
        child_argv.append("--no-viz")
    else:
        child_argv.extend(["--viz-dir", viz_dir])
    if args.viz_video is True:
        child_argv.append("--viz-video")
    elif args.viz_video is False:
        child_argv.append("--no-viz-video")
    if args.video_fps is not None:
        child_argv.extend(["--video-fps", str(float(args.video_fps))])
    if bool(getattr(args, "richviz", False)):
        child_argv.append("--richviz")
    if richviz_dir_override:
        child_argv.extend(["--richviz-dir", str(richviz_dir_override)])
    elif getattr(args, "richviz_dir", None):
        child_argv.extend(["--richviz-dir", str(args.richviz_dir)])
    if bool(getattr(args, "richviz_all_frames", False)):
        child_argv.append("--richviz-all-frames")
    if getattr(args, "richviz_frame_stride", None) is not None:
        child_argv.extend(["--richviz-frame-stride", str(int(args.richviz_frame_stride))])
    if getattr(args, "richviz_max_frames", None) is not None:
        child_argv.extend(["--richviz-max-frames", str(int(args.richviz_max_frames))])
    if bool(getattr(args, "vad_oracle_branch_by_local_ade", True)):
        child_argv.append("--vad-oracle-branch-by-local-ade")
    else:
        child_argv.append("--no-vad-oracle-branch-by-local-ade")
    if bool(getattr(args, "vad_force_fresh_inference", False)):
        child_argv.append("--vad-force-fresh-inference")
    if getattr(args, "canonical_horizon_s", None) is not None:
        child_argv.extend(["--canonical-horizon-s", str(float(args.canonical_horizon_s))])
    if not bool(getattr(args, "native_horizon", True)):
        child_argv.append("--no-native-horizon")
    if not bool(getattr(args, "skip_stale_frames", True)):
        child_argv.append("--no-skip-stale-frames")
    if args.town_id:
        child_argv.extend(["--town-id", str(args.town_id)])
    if not bool(getattr(args, "dump_frame_debug", True)):
        child_argv.append("--no-dump-frame-debug")
    if bool(getattr(args, "trajectory_extrapolation", False)):
        child_argv.append("--trajectory-extrapolation")
    if bool(getattr(args, "canonical_extrapolation", False)):
        child_argv.append("--canonical-extrapolation")

    if planner_request.conda_env:
        return [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            str(planner_request.conda_env),
            "python",
            *child_argv,
        ]
    return [sys.executable, *child_argv]


def _comparison_sort_key(value: Any) -> Tuple[int, float]:
    try:
        numeric = float(value)
    except Exception:
        return (1, float("inf"))
    if not math.isfinite(numeric):
        return (1, float("inf"))
    return (0, numeric)


def _write_comparison_summary(
    *,
    comparison_path: Path,
    planner_payloads: Dict[str, Dict[str, Any]],
    failures: Dict[str, Dict[str, Any]],
    scheduler_gpus: List[str],
    max_parallel_children: int,
) -> None:
    overall = {
        name: payload.get("overall", {})
        for name, payload in planner_payloads.items()
    }
    per_horizon = {
        name: payload.get("per_horizon_ade", {})
        for name, payload in planner_payloads.items()
    }
    ranking = {
        "plan_ade": sorted(
            overall.keys(),
            key=lambda name: _comparison_sort_key(overall[name].get("plan_ade")),
        ),
        "plan_fde": sorted(
            overall.keys(),
            key=lambda name: _comparison_sort_key(overall[name].get("plan_fde")),
        ),
        "collision_rate": sorted(
            overall.keys(),
            key=lambda name: _comparison_sort_key(overall[name].get("collision_rate")),
        ),
    }
    payload = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "scheduler": {
            "gpus": list(scheduler_gpus),
            "max_parallel_children": int(max_parallel_children),
            "gpu_min_free_mib": int(AUTO_WORKERS_PER_GPU_MIN_FREE_MIB_DEFAULT),
        },
        "results": overall,
        "per_horizon_ade": per_horizon,
        "ranking": ranking,
        "failures": failures,
    }
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with comparison_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[INFO] Comparison summary saved → {comparison_path}")


def _comparison_output_path(args: argparse.Namespace) -> Path:
    if args.output:
        base = Path(_resolve_abs_path(args.output) or args.output)
        return base / "comparison.json"
    return (_PROJECT_ROOT / "results" / "openloop_comparison.json").resolve()


def _direct_agent_request_from_args(args: argparse.Namespace) -> Optional[PlannerRequest]:
    agent_path = getattr(args, "agent_path", None)
    agent_config = getattr(args, "agent_config", None)
    if not agent_path and not agent_config:
        return None
    if not agent_path or not agent_config:
        raise ValueError("Direct agent execution requires both --agent-path and --agent-config.")
    label = str(getattr(args, "agent_label", None) or Path(str(agent_path)).stem)
    conda_env = getattr(args, "agent_conda_env_name", None) or getattr(
        args, "planner_conda_env_name", None
    )
    return PlannerRequest(
        name=label,
        agent=str(agent_path),
        agent_config=str(agent_config),
        conda_env=conda_env,
        entry_point=getattr(args, "agent_entry_point", None),
        preset_name=None,
        run_id=sanitize_run_id(str(getattr(args, "planner_run_id", None) or label)),
    )


def _resolve_agent_requests_from_args(args: argparse.Namespace) -> List[PlannerRequest]:
    direct_request = _direct_agent_request_from_args(args)
    generic_requests = parse_agent_requests(getattr(args, "agent", None))
    preset_requests = parse_planner_requests(args.planner) if args.planner else []
    if direct_request is not None and (generic_requests or preset_requests):
        raise ValueError(
            "Use either direct --agent-path/--agent-config or repeated --agent/--planner "
            "entries, not both."
        )
    if direct_request is not None:
        return [direct_request]
    combined = generic_requests + preset_requests
    if combined:
        return _finalize_request_run_ids(combined)
    raise ValueError(
        "Provide at least one --planner or --agent entry, or pass --agent-path with --agent-config."
    )


def _run_single_planner_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    # Set canonical extrapolation env var before canonical_trajectory is imported.
    os.environ["OPENLOOP_CANONICAL_HORIZON_S"] = (
        f"{float(getattr(args, 'canonical_horizon_s', 3.0)):.12g}"
    )
    if not bool(getattr(args, "canonical_extrapolation", False)):
        os.environ["OPENLOOP_CANONICAL_EXTRAPOLATE"] = "0"
    planner_requests = _resolve_agent_requests_from_args(args)
    if len(planner_requests) != 1:
        raise ValueError("Single-agent execution requires exactly one request.")

    planner_request = planner_requests[0]
    planner_name = str(planner_request.preset_name or planner_request.name)
    planner_label = str(args.planner_run_id or planner_request.run_id or planner_request.name)
    planner_conda_env = (
        getattr(args, "agent_conda_env_name", None)
        or args.planner_conda_env_name
        or planner_request.conda_env
    )
    output_dir = _resolve_output_dir(args, planner_label)
    viz_dir = _resolve_viz_dir(args, output_dir)
    map_pkl = _resolve_abs_path(args.map_pkl)
    data_dir = _resolve_abs_path(args.data_dir) or args.data_dir

    _ensure_openloop_runtime()
    _install_signal_handlers()
    env_saved = _push_colmdriver_port_env(planner_name, int(args.llm_port), int(args.vlm_port))
    local_manager = None
    try:
        local_manager = _start_local_colmdriver_vllm_if_needed(args, planner_name, output_dir)
        hero_vehicles = install_openloop_stubs(
            vector_map_path=map_pkl,
            num_ego_vehicles=args.ego_num,
        )
        return evaluate_agent(
            agent_path=planner_request.agent,
            agent_config=planner_request.agent_config,
            data_dir=data_dir,
            map_pkl=map_pkl,
            output_dir=output_dir,
            agent_label=planner_label,
            agent_conda_env=planner_conda_env,
            agent_entry_point=planner_request.entry_point,
            scenario_names=args.scenarios,
            actor_id=args.actor_id,
            num_ego=args.ego_num,
            verbose=args.verbose,
            viz_dir=viz_dir,
            viz_every=args.viz_every,
            viz_style=args.viz_style,
            viz_video=(False if args.no_viz else args.viz_video),
            video_fps=args.video_fps,
            video_codec=args.video_codec,
            panel_width=args.panel_width,
            max_frames=args.max_frames,
            traj_source=args.traj_source,
            trajectory_extrapolation=args.trajectory_extrapolation,
            town_id=args.town_id,
            dump_frame_debug=args.dump_frame_debug,
            richviz=args.richviz,
            richviz_dir=args.richviz_dir,
            richviz_all_frames=args.richviz_all_frames,
            richviz_frame_stride=args.richviz_frame_stride,
            richviz_max_frames=args.richviz_max_frames,
            planner_name_hint=planner_name,
            native_horizon=args.native_horizon,
            vad_oracle_branch_by_local_ade=args.vad_oracle_branch_by_local_ade,
            vad_force_fresh_inference=args.vad_force_fresh_inference,
            skip_stale_frames=args.skip_stale_frames,
            scoring_path=args.scoring_path,
            canonical_horizon_s=float(getattr(args, "canonical_horizon_s", 3.0)),
        )
    finally:
        if local_manager is not None:
            try:
                local_manager.stop()
            except Exception as exc:
                print(f"[WARN] Failed to stop local CoLMDriver vLLM manager: {exc}")
            finally:
                _unregister_local_vllm_manager(local_manager)
        try:
            _MockCarlaDataProvider.cleanup()
        except Exception:
            pass
        _restore_colmdriver_port_env(env_saved)


def _run_multi_planner_launcher(args: argparse.Namespace) -> int:
    planner_requests = _resolve_agent_requests_from_args(args)
    multi_planner_run = len(planner_requests) > 1
    planner_labels = [str(req.run_id or req.name) for req in planner_requests]
    planner_request_by_label = {
        str(req.run_id or req.name): req
        for req in planner_requests
    }
    _install_signal_handlers()
    scheduler_gpus = _resolve_requested_scheduler_gpus(args)
    if args.start_colmdriver_vllm and any(
        _planner_request_uses_colmdriver_vllm(req) for req in planner_requests
    ):
        if len(scheduler_gpus) < 2:
            raise RuntimeError(
                "--start-colmdriver-vllm requires at least 2 distinct GPUs in --gpus."
            )

    scheduler_gpu_ids: List[Optional[str]] = scheduler_gpus[:] if scheduler_gpus else [None]
    gpu_states: Dict[Optional[str], GPUPlannerState] = {
        gpu: GPUPlannerState(gpu=gpu, launch_capacity=1)
        for gpu in scheduler_gpu_ids
    }
    _refresh_scheduler_gpu_states(gpu_states)

    used_ports: set[int] = set()
    reserved_vllm_gpu_counts: Dict[str, int] = {}
    pending = list(planner_requests)
    running: List[PlannerChildProcess] = []
    planner_payloads: Dict[str, Dict[str, Any]] = {}
    failures: Dict[str, Dict[str, Any]] = {}
    max_parallel_children = 0
    shared_colmdriver_manager: Optional[ColmDriverVLLMManager] = None
    shared_colmdriver_reservation: Optional[ColmDriverVLLMReservation] = None
    shared_colmdriver_llm_port: Optional[int] = None
    shared_colmdriver_vlm_port: Optional[int] = None

    def _stop_shared_colmdriver_manager() -> None:
        nonlocal shared_colmdriver_manager
        nonlocal shared_colmdriver_reservation
        nonlocal shared_colmdriver_llm_port
        nonlocal shared_colmdriver_vlm_port
        if shared_colmdriver_manager is None:
            return
        try:
            shared_colmdriver_manager.stop()
        except Exception as exc:
            print(f"[WARN] Failed to stop shared CoLMDriver vLLM manager: {exc}")
        finally:
            _unregister_local_vllm_manager(shared_colmdriver_manager)
        if shared_colmdriver_reservation is not None:
            for reserved_gpu in (
                shared_colmdriver_reservation.vlm_gpu,
                shared_colmdriver_reservation.llm_gpu,
            ):
                next_count = reserved_vllm_gpu_counts.get(reserved_gpu, 0) - 1
                if next_count > 0:
                    reserved_vllm_gpu_counts[reserved_gpu] = next_count
                else:
                    reserved_vllm_gpu_counts.pop(reserved_gpu, None)
        if shared_colmdriver_llm_port is not None:
            used_ports.discard(shared_colmdriver_llm_port)
        if shared_colmdriver_vlm_port is not None:
            used_ports.discard(shared_colmdriver_vlm_port)
        shared_colmdriver_manager = None
        shared_colmdriver_reservation = None
        shared_colmdriver_llm_port = None
        shared_colmdriver_vlm_port = None

    def _has_future_colmdriver_work() -> bool:
        if any(_planner_request_uses_colmdriver_vllm(req) for req in pending):
            return True
        for child in running:
            planner_request = planner_request_by_label.get(child.planner_name)
            planner_name = (
                str(planner_request.preset_name or planner_request.name)
                if planner_request is not None
                else str(child.planner_name)
            )
            if _planner_uses_colmdriver_vllm(planner_name):
                return True
        return False

    while pending or running:
        launched_any = False
        _refresh_scheduler_gpu_states(gpu_states)
        while pending:
            launch_gpu_state = _select_launch_gpu_state(gpu_states)
            if launch_gpu_state is None:
                break

            selected_index: Optional[int] = None
            selected_request: Optional[PlannerRequest] = None
            reservation_candidates: Optional[List[ColmDriverVLLMReservation]] = None
            for pending_index, pending_request in enumerate(pending):
                pending_name = str(pending_request.preset_name or pending_request.name)
                pending_needs_vllm = _planner_request_uses_colmdriver_vllm(pending_request)
                if not (args.start_colmdriver_vllm and pending_needs_vllm):
                    selected_index = pending_index
                    selected_request = pending_request
                    reservation_candidates = None
                    break
                if shared_colmdriver_manager is not None:
                    selected_index = pending_index
                    selected_request = pending_request
                    reservation_candidates = None
                    break

                excluded_vllm_gpus = list(reserved_vllm_gpu_counts.keys())
                if launch_gpu_state.gpu is not None:
                    excluded_vllm_gpus.append(str(launch_gpu_state.gpu))
                candidate_reservations = list_colmdriver_vllm_gpu_reservations(
                    scheduler_gpus,
                    exclude_gpus=excluded_vllm_gpus,
                )
                if candidate_reservations:
                    selected_index = pending_index
                    selected_request = pending_request
                    reservation_candidates = candidate_reservations
                    break

            if selected_index is None or selected_request is None:
                break

            planner_request = pending.pop(selected_index)
            planner_name = str(planner_request.preset_name or planner_request.name)
            planner_label = str(planner_request.run_id or planner_request.name)
            planner_slug = sanitize_run_id(planner_label)
            needs_vllm = _planner_request_uses_colmdriver_vllm(planner_request)
            wants_local_manager = bool(args.start_colmdriver_vllm and needs_vllm)
            if args.output and multi_planner_run:
                output_root = Path(_resolve_abs_path(args.output) or args.output)
                output_dir = str((output_root / planner_slug).resolve())
            else:
                output_dir = _resolve_output_dir(args, planner_label)
            viz_dir = _resolve_viz_dir(args, output_dir)
            llm_port = int(args.llm_port)
            vlm_port = int(args.vlm_port)
            manager: Optional[ColmDriverVLLMManager] = None
            richviz_dir_override: Optional[str] = None
            if multi_planner_run and getattr(args, "richviz_dir", None):
                richviz_root = Path(_resolve_abs_path(args.richviz_dir) or str(args.richviz_dir))
                richviz_dir_override = str((richviz_root / planner_slug).resolve())

            if wants_local_manager:
                if shared_colmdriver_manager is None:
                    llm_port = find_available_port("127.0.0.1", llm_port, reserved_ports=used_ports)
                    used_ports.add(llm_port)
                    vlm_port = find_available_port("127.0.0.1", vlm_port, reserved_ports=used_ports)
                    used_ports.add(vlm_port)
                else:
                    llm_port = int(shared_colmdriver_llm_port or llm_port)
                    vlm_port = int(shared_colmdriver_vlm_port or vlm_port)
                reservation = None
            else:
                reservation = None

            cmd = _build_child_command(
                args,
                planner_request=planner_request,
                output_dir=output_dir,
                viz_dir=viz_dir,
                llm_port=llm_port,
                vlm_port=vlm_port,
                richviz_dir_override=richviz_dir_override,
            )
            child_env = dict(os.environ)
            if planner_request.conda_env:
                child_env = _build_conda_launch_env(child_env)
            if launch_gpu_state.gpu is not None:
                child_env["CUDA_VISIBLE_DEVICES"] = str(launch_gpu_state.gpu)
            child = PlannerChildProcess(
                planner_name=planner_label,
                cmd=cmd,
                env=child_env,
                cwd=_PROJECT_ROOT,
                log_path=Path(output_dir) / "logs" / f"{sanitize_run_id(planner_label)}_child.log",
                result_file=Path(output_dir) / f"{sanitize_run_id(planner_label)}_openloop.json",
                vllm_manager=manager,
                reservation=reservation,
                llm_port=None,
                vlm_port=None,
            )
            launch_exc: Optional[Exception] = None
            vllm_retry_errors: List[str] = []
            if wants_local_manager:
                if shared_colmdriver_manager is None:
                    for reservation_index, reservation in enumerate(reservation_candidates or []):
                        manager = ColmDriverVLLMManager(
                            repo_root=_PROJECT_ROOT,
                            base_env=os.environ,
                            conda_env=str(args.colmdriver_vllm_env),
                            reservation=reservation,
                            llm_port=llm_port,
                            vlm_port=vlm_port,
                            startup_timeout_s=float(args.colmdriver_vllm_startup_timeout),
                            log_dir=Path(output_dir) / "services",
                        )
                        try:
                            manager.start()
                            _register_local_vllm_manager(manager)
                            shared_colmdriver_manager = manager
                            shared_colmdriver_reservation = reservation
                            shared_colmdriver_llm_port = llm_port
                            shared_colmdriver_vlm_port = vlm_port
                            reserved_vllm_gpu_counts[reservation.vlm_gpu] = (
                                reserved_vllm_gpu_counts.get(reservation.vlm_gpu, 0) + 1
                            )
                            reserved_vllm_gpu_counts[reservation.llm_gpu] = (
                                reserved_vllm_gpu_counts.get(reservation.llm_gpu, 0) + 1
                            )
                            print(
                                "[INFO] Shared CoLMDriver vLLM started with "
                                f"VLM gpu={reservation.vlm_gpu} port={vlm_port} and "
                                f"LLM gpu={reservation.llm_gpu} port={llm_port}."
                            )
                            break
                        except Exception as exc:
                            launch_exc = exc
                            vllm_retry_errors.append(
                                f"{reservation.vlm_gpu}/{reservation.llm_gpu}: {exc}"
                            )
                            try:
                                manager.stop()
                            except Exception:
                                pass
                            if reservation_index + 1 < len(reservation_candidates or []):
                                print(
                                    f"[WARN] Planner '{planner_label}' CoLMDriver vLLM startup "
                                    f"failed on gpu pair {reservation.vlm_gpu}/{reservation.llm_gpu}; "
                                    "retrying another pair."
                                )
                    if shared_colmdriver_manager is None:
                        used_ports.discard(llm_port)
                        used_ports.discard(vlm_port)
                if shared_colmdriver_manager is not None:
                    launch_exc = None
                    try:
                        child.start()
                    except Exception as exc:
                        launch_exc = exc
            else:
                try:
                    child.start()
                    launch_exc = None
                except Exception as exc:
                    launch_exc = exc

            if launch_exc is not None:
                failures[planner_label] = {
                    "stage": "startup",
                    "error": str(launch_exc),
                    "log_path": str(child.log_path),
                    "planner": planner_name,
                    "planner_label": planner_label,
                    "planner_conda_env": planner_request.conda_env,
                }
                if vllm_retry_errors:
                    failures[planner_label]["vllm_attempts"] = vllm_retry_errors
                print(f"[ERROR] Failed to start planner '{planner_label}': {launch_exc}")
                continue

            launch_gpu_state.active_count += 1
            running.append(child)
            launched_any = True
            max_parallel_children = max(max_parallel_children, len(running))
            _refresh_scheduler_gpu_states(gpu_states)

        if not running:
            if pending and not launched_any:
                blocked_vllm_requests = []
                if args.start_colmdriver_vllm and shared_colmdriver_manager is None:
                    for pending_request in pending:
                        if not _planner_request_uses_colmdriver_vllm(pending_request):
                            continue
                        if list_colmdriver_vllm_gpu_reservations(
                            scheduler_gpus,
                            exclude_gpus=list(reserved_vllm_gpu_counts.keys()),
                        ):
                            continue
                        blocked_vllm_requests.append(pending_request)
                if blocked_vllm_requests and len(blocked_vllm_requests) == len(pending):
                    for blocked_request in blocked_vllm_requests:
                        blocked_label = str(blocked_request.run_id or blocked_request.name)
                        failures[blocked_label] = {
                            "stage": "startup",
                            "error": (
                                "No CoLMDriver vLLM GPU pair currently has enough free VRAM "
                                "across --gpus. Wait for GPUs to free up or provide a larger "
                                "GPU pool."
                            ),
                            "planner": str(blocked_request.agent),
                            "planner_label": blocked_label,
                            "planner_conda_env": blocked_request.conda_env,
                        }
                        print(
                            f"[ERROR] Planner '{blocked_label}' could not start because no "
                            "eligible CoLMDriver vLLM GPU pair was available."
                        )
                    pending.clear()
                    break
                forced_state = choose_force_launch_gpu(gpu_states)
                forced_state.launch_capacity = max(
                    forced_state.launch_capacity,
                    forced_state.active_count + 1,
                )
                time.sleep(0.5)
                continue
            break

        finished: List[PlannerChildProcess] = []
        for child in list(running):
            if child.poll() is None:
                continue
            child.wait()
            finished.append(child)

        if not finished:
            time.sleep(1.0)
            continue

        for child in finished:
            running.remove(child)
            if child.reservation is not None:
                for reserved_gpu in (child.reservation.vlm_gpu, child.reservation.llm_gpu):
                    next_count = reserved_vllm_gpu_counts.get(reserved_gpu, 0) - 1
                    if next_count > 0:
                        reserved_vllm_gpu_counts[reserved_gpu] = next_count
                    else:
                        reserved_vllm_gpu_counts.pop(reserved_gpu, None)
            assigned_gpu = child.env.get("CUDA_VISIBLE_DEVICES")
            if assigned_gpu in gpu_states:
                gpu_states[assigned_gpu].active_count = max(
                    0,
                    gpu_states[assigned_gpu].active_count - 1,
                )
            elif None in gpu_states:
                gpu_states[None].active_count = max(0, gpu_states[None].active_count - 1)
            _refresh_scheduler_gpu_states(gpu_states)

            if child.returncode == 0 and child.result_file.exists():
                try:
                    with child.result_file.open("r", encoding="utf-8") as fh:
                        planner_payloads[child.planner_name] = json.load(fh)
                except Exception as exc:
                    planner_request = planner_request_by_label.get(child.planner_name)
                    failures[child.planner_name] = {
                        "stage": "result_read",
                        "error": str(exc),
                        "result_file": str(child.result_file),
                        "log_path": str(child.log_path),
                        "planner": (
                            planner_request.agent
                            if planner_request is not None
                            else child.planner_name
                        ),
                        "planner_label": child.planner_name,
                        "planner_conda_env": (
                            planner_request.conda_env
                            if planner_request is not None
                            else None
                        ),
                    }
            else:
                planner_request = planner_request_by_label.get(child.planner_name)
                failures[child.planner_name] = {
                    "stage": "runtime",
                    "returncode": child.returncode,
                    "result_file": str(child.result_file),
                    "log_path": str(child.log_path),
                    "planner": (
                        planner_request.agent
                        if planner_request is not None
                        else child.planner_name
                    ),
                    "planner_label": child.planner_name,
                    "planner_conda_env": (
                        planner_request.conda_env
                        if planner_request is not None
                        else None
                    ),
                }
                print(
                    f"[ERROR] Planner '{child.planner_name}' failed "
                    f"(returncode={child.returncode}). See {child.log_path}."
                )

        if shared_colmdriver_manager is not None and not _has_future_colmdriver_work():
            _stop_shared_colmdriver_manager()

    overall_results = {
        name: payload.get("overall", {})
        for name, payload in planner_payloads.items()
    }

    if len(planner_payloads) > 1:
        print("\n" + "=" * 70)
        print(f"  {'Agent':<20} {'Plan ADE':>10} {'Plan FDE':>10} {'Coll. Rate':>12}")
        print("  " + "-" * 56)
        for name in planner_labels:
            metrics = overall_results.get(name)
            if metrics is None:
                continue
            print(
                f"  {name:<20} "
                f"{metrics.get('plan_ade', float('nan')):>10.4f} "
                f"{metrics.get('plan_fde', float('nan')):>10.4f} "
                f"{metrics.get('collision_rate', float('nan')):>12.4f}"
            )
        print("=" * 70)

    comparison_path = _comparison_output_path(args)
    _write_comparison_summary(
        comparison_path=comparison_path,
        planner_payloads=planner_payloads,
        failures=failures,
        scheduler_gpus=scheduler_gpus,
        max_parallel_children=max_parallel_children,
    )

    _stop_shared_colmdriver_manager()

    if failures:
        print("[WARN] Some planners failed:")
        for planner_name in planner_labels:
            failure = failures.get(planner_name)
            if failure is None:
                continue
            print(f"  - {planner_name}: {failure}")
        return 1
    return 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--planner",
        action="append",
        default=[],
        help=(
            "Legacy planner preset alias. Repeat to launch multiple preset-backed runs. "
            "Each value can be <planner> or [<planner>,<conda_env>]. "
            f"Available planners: {', '.join(sorted(PLANNER_SPECS.keys()))}."
        ),
    )
    p.add_argument(
        "--agent-path",
        default=None,
        help=(
            "Path to a leaderboard agent module. Use with --agent-config for "
            "planner-agnostic single-agent evaluation."
        ),
    )
    p.add_argument(
        "--agent-config",
        default=None,
        help="Path passed to agent.setup(...). Use with --agent-path.",
    )
    p.add_argument(
        "--agent-label",
        default=None,
        help="Display label for direct --agent-path runs (default: module stem).",
    )
    p.add_argument(
        "--agent",
        action="append",
        default=[],
        help=(
            "Generic multi-agent request. Repeat to compare arbitrary agents using "
            "[label,agent_path,agent_config], "
            "[label,agent_path,agent_config,conda_env], or "
            "[label,agent_path,agent_config,entry_point,conda_env]."
        ),
    )
    p.add_argument(
        "--agent-entry-point",
        default=None,
        help="Optional agent class name if the module does not define get_entry_point().",
    )
    p.add_argument(
        "--gpu",
        "--gpus",
        dest="gpus",
        action="append",
        default=[],
        help=(
            "GPU ids or UUIDs available to the scheduler. Repeat the flag or pass a "
            "comma-separated list. Agent placement, CoLMDriver vLLM placement, and "
            "concurrency are derived automatically from this pool."
        ),
    )
    p.add_argument(
        "--data-dir", required=True,
        help="Root directory of the v2xpnp scenario dataset.",
    )
    p.add_argument(
        "--map-pkl", default=None,
        help="Path to v2v_corridors_vector_map.pkl (enables accurate "
             "drivable-area BEV rendering; falls back to all-drivable if omitted).",
    )
    p.add_argument(
        "--output", default=None,
        help="Directory to write JSON result files. "
             "When multiple agents are evaluated, a shared comparison.json is also written. "
             "Default: results/openloop_<label>.",
    )
    p.add_argument(
        "--scenarios", nargs="*", default=None,
        help="Restrict evaluation to these scenario directory names.",
    )
    p.add_argument(
        "--actor-id", type=int, default=None,
        help="Evaluate this actor id from each scenario (default: evaluate all actor ids).",
    )
    p.add_argument(
        "--ego-num", type=int, default=1,
        help="Number of ego vehicles per scenario.",
    )
    p.add_argument(
        "--town-id",
        default=None,
        help=(
            "Generic leaderboard runtime town id exposed as agent.town_id. "
            "Defaults to the mock map name when omitted."
        ),
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print detailed per-frame sensor, prediction, and metric diagnostics.",
    )
    p.add_argument(
        "--viz-dir", default=None,
        help="Directory to save PNG visualizations (BEV frames, trajectory plots, "
             "metrics time-series). Requires matplotlib. Default: <output>/viz/",
    )
    p.add_argument(
        "--viz-every", type=int, default=1,
        help="Save a BEV+world frame visualization every N frames (default: every frame).",
    )
    p.add_argument(
        "--viz-style", choices=["basic", "rich"], default="basic",
        help="Visualization style. 'basic' keeps matplotlib plots; 'rich' creates cinematic composites.",
    )
    p.add_argument(
        "--viz-video", dest="viz_video", action="store_true", default=None,
        help="Enable MP4 export for rich visualization mode (default: enabled in rich mode).",
    )
    p.add_argument(
        "--no-viz-video", dest="viz_video", action="store_false",
        help="Disable MP4 export in rich visualization mode.",
    )
    p.add_argument(
        "--video-fps", type=float, default=None,
        help="Output video FPS (default: scenario FPS).",
    )
    p.add_argument(
        "--video-codec", default="mp4v",
        help="FourCC video codec for OpenCV writer (default: mp4v).",
    )
    p.add_argument(
        "--panel-width", type=int, default=1920,
        help="Rich visualization composite width in pixels (default: 1920).",
    )
    p.add_argument(
        "--llm-port", type=int, default=8888,
        help="Base LLM server port for CoLMDriver auto-start (default: 8888).",
    )
    p.add_argument(
        "--vlm-port", type=int, default=1111,
        help="Base VLM server port for CoLMDriver auto-start (default: 1111).",
    )
    p.add_argument(
        "--start-colmdriver-vllm", action="store_true",
        help=(
            "Automatically start shared CoLMDriver VLM/LLM vLLM services and reuse "
            "them across CoLMDriver-family planner evaluations."
        ),
    )
    p.add_argument(
        "--colmdriver-vllm-env",
        default=COLMDRIVER_VLLM_ENV_DEFAULT,
        help=(
            "Conda environment containing the `vllm` CLI for CoLMDriver server startup "
            f"(default: {COLMDRIVER_VLLM_ENV_DEFAULT})."
        ),
    )
    p.add_argument(
        "--colmdriver-vllm-startup-timeout",
        type=float,
        default=COLMDRIVER_VLLM_STARTUP_TIMEOUT_DEFAULT,
        help=(
            "Seconds to wait for the auto-started CoLMDriver VLM/LLM ports to come up "
            f"(default: {COLMDRIVER_VLLM_STARTUP_TIMEOUT_DEFAULT:.0f})."
        ),
    )
    p.add_argument(
        "--traj-source",
        choices=["control_rollout", "planner_waypoints", "auto"],
        default="auto",
        help=(
            "Trajectory source policy: control_rollout (always rollout), "
            "planner_waypoints (strict adapter-only), or auto "
            "(adapter with control_rollout fallback; default)."
        ),
    )
    p.add_argument(
        "--trajectory-extrapolation",
        dest="trajectory_extrapolation",
        action="store_true",
        default=False,
        help=(
            "Enable linear extrapolation beyond each planner's native horizon. "
            "(default: disabled — no extrapolation beyond native horizon)."
        ),
    )
    p.add_argument(
        "--no-trajectory-extrapolation",
        dest="trajectory_extrapolation",
        action="store_false",
        help=(
            "Disable extrapolation in adapter resampling (default). Future points "
            "beyond native horizon hold the last available waypoint."
        ),
    )
    p.add_argument(
        "--canonical-horizon-s",
        type=float,
        default=3.0,
        help=(
            "Canonical evaluation horizon in seconds on the shared 0.5 s grid "
            "(default: 3.0). Use 5.0 for a uniform 5 s benchmark."
        ),
    )
    p.add_argument(
        "--canonical-extrapolation",
        dest="canonical_extrapolation",
        action="store_true",
        default=False,
        help=(
            "Enable linear extrapolation in canonical trajectory construction. "
            "Short-horizon planners are extrapolated to the configured canonical horizon. "
            "(default: disabled — planners are scored only within their native horizon)."
        ),
    )
    p.add_argument(
        "--no-canonical-extrapolation",
        dest="canonical_extrapolation",
        action="store_false",
        help=(
            "Disable canonical extrapolation (default): each planner is scored only "
            "at canonical timesteps within its native prediction horizon. "
            "This is the scientifically rigorous default."
        ),
    )
    p.add_argument(
        "--native-horizon",
        dest="native_horizon",
        action="store_true",
        default=True,
        help=(
            "Evaluate each planner at its own native time step (no resampling). "
            "pred_wp[k] is compared to GT at t + k * native_dt. "
            "(default: enabled)"
        ),
    )
    p.add_argument(
        "--no-native-horizon",
        dest="native_horizon",
        action="store_false",
        help=(
            "Revert to fixed 0.5 s evaluation grid (original behaviour). "
            "All planners are resampled and compared on the shared 0.5 s grid "
            "up to --canonical-horizon-s."
        ),
    )
    p.add_argument(
        "--vad-oracle-branch-by-local-ade",
        dest="vad_oracle_branch_by_local_ade",
        action="store_true",
        default=True,
        help=(
            "VAD-only debug mode (default enabled): replace command-selected branch "
            "with the branch "
            "that has the minimum GT local ADE on the current frame's evaluation grid "
            "(oracle; uses GT and is not deployable)."
        ),
    )
    p.add_argument(
        "--no-vad-oracle-branch-by-local-ade",
        dest="vad_oracle_branch_by_local_ade",
        action="store_false",
        help="Disable VAD oracle best-branch selection and use command-selected branch.",
    )
    p.add_argument(
        "--vad-force-fresh-inference",
        dest="vad_force_fresh_inference",
        action="store_true",
        default=True,
        help=(
            "Planner option (default enabled): in open-loop evaluation, force fresh "
            "planner inference on every fed frame (10 Hz) and bypass planner-side "
            "non-realtime frame skipping."
        ),
    )
    p.add_argument(
        "--no-vad-force-fresh-inference",
        dest="vad_force_fresh_inference",
        action="store_false",
        help=(
            "Disable open-loop fresh-inference override and use each planner's native "
            "non-realtime inference cadence."
        ),
    )
    p.add_argument(
        "--skip-stale-frames",
        dest="skip_stale_frames",
        action="store_true",
        default=True,
        help=(
            "Exclude frames with recycled (stale) control outputs from ADE/FDE "
            "computation; only fresh-inference frames contribute to metrics. "
            "(default: enabled)"
        ),
    )
    p.add_argument(
        "--no-skip-stale-frames",
        dest="skip_stale_frames",
        action="store_false",
        help="Include stale-control frames in ADE/FDE computation.",
    )
    p.add_argument(
        "--scoring-path",
        choices=["canonical_fresh_only", "legacy"],
        default="canonical_fresh_only",
        help=(
            "Primary scoring path. canonical_fresh_only uses CanonicalTrajectory, "
            "fresh-only filtering, and full configured canonical-horizon coverage. "
            "legacy preserves the pre-Stage-4 evaluator as the active scorer."
        ),
    )
    p.add_argument(
        "--no-viz", action="store_true",
        help="Disable all visualizations even if matplotlib is available.",
    )
    p.add_argument(
        "--max-frames", type=int, default=None,
        help="Optional limit on frames per scenario for faster smoke runs.",
    )
    p.add_argument(
        "--dump-frame-debug",
        dest="dump_frame_debug",
        action="store_true",
        default=True,
        help=(
            "Write per-frame JSONL debug snapshots (predictions, GT, controls, "
            "adapter internals) under <output>/debug_frames/<planner_label>/."
        ),
    )
    p.add_argument(
        "--no-dump-frame-debug",
        dest="dump_frame_debug",
        action="store_false",
        help="Disable per-frame JSONL debug snapshot dumping.",
    )
    p.add_argument(
        "--richviz",
        action="store_true",
        help=(
            "Automatically run planner_inspector_viz after evaluation using dumped "
            "debug JSONL files. Supports all planners (generic fallback panels "
            "for unknown adapter families)."
        ),
    )
    p.add_argument(
        "--richviz-dir",
        default=None,
        help="Output directory for integrated planner_inspector_viz figures (default: <output>/richviz).",
    )
    p.add_argument(
        "--richviz-all-frames",
        action="store_true",
        help="When --richviz is enabled, render all frames instead of sampled frames.",
    )
    p.add_argument(
        "--richviz-frame-stride",
        type=int,
        default=None,
        help="When --richviz is enabled, sample every N-th frame for per-frame pages.",
    )
    p.add_argument(
        "--richviz-max-frames",
        type=int,
        default=40,
        help="When --richviz is enabled, max sampled per-frame pages (default: 40).",
    )
    p.add_argument(
        "--child-runner", action="store_true", help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--planner-run-id", default=None, help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--planner-conda-env-name", default=None, help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--agent-conda-env-name", default=None, help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--planner-jobs", type=int, default=None, help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--colmdriver-vllm-gpus", action="append", dest="gpus", help=argparse.SUPPRESS,
    )
    return p.parse_args()


def _configure_canonical_horizon_from_args(args: argparse.Namespace) -> None:
    horizon_s = float(getattr(args, "canonical_horizon_s", 3.0))
    if not math.isfinite(horizon_s) or horizon_s <= 0.0:
        raise ValueError(f"--canonical-horizon-s must be positive, got {horizon_s!r}")
    n_steps = int(round(horizon_s / 0.5))
    if n_steps <= 0 or abs(n_steps * 0.5 - horizon_s) > 1e-9:
        raise ValueError(
            "--canonical-horizon-s must be a positive multiple of 0.5 s; "
            f"got {horizon_s!r}"
        )
    os.environ["OPENLOOP_CANONICAL_HORIZON_S"] = f"{horizon_s:.12g}"


def main():
    args = parse_args()
    _configure_canonical_horizon_from_args(args)
    # Set canonical extrapolation env var so child processes inherit it.
    if not bool(getattr(args, "canonical_extrapolation", False)):
        os.environ["OPENLOOP_CANONICAL_EXTRAPOLATE"] = "0"
    if args.child_runner:
        _run_single_planner_from_args(args)
        return

    if args.planner_jobs is not None:
        print("[WARN] --planner-jobs is deprecated and ignored. Concurrency is GPU/VRAM-driven.")

    raise SystemExit(_run_multi_planner_launcher(args))


if __name__ == "__main__":
    main()
