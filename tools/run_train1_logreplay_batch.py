#!/usr/bin/env python3
"""
Batch pipeline for train1 scenarios:
1) yaml_to_carla_log
2) run_custom_eval in log-replay mode
3) gen_video side-by-side + CARLA-only + real-only per ego
"""

from __future__ import annotations

import argparse
import os
import select
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set


def _format_exit_code(returncode: int) -> str:
    code = int(returncode)
    if code == 0:
        return "exit 0"
    if code < 0:
        sig_num = -code
        try:
            sig_name = signal.Signals(sig_num).name
        except ValueError:
            sig_name = f"signal {sig_num}"
        return f"signal {sig_num} ({sig_name})"
    if code >= 128:
        sig = code - 128
        try:
            sig_name = signal.Signals(sig).name
            return f"exit {code} (likely signal {sig}: {sig_name})"
        except ValueError:
            return f"exit {code} (likely signal {sig})"
    return f"exit {code}"


def _run(
    cmd: Sequence[str],
    *,
    dry_run: bool,
    log_path: Optional[Path] = None,
    failure_tail_lines: int = 80,
    timeout_seconds: Optional[float] = None,
) -> None:
    print("[CMD]", " ".join(cmd))
    if dry_run:
        return
    env = os.environ.copy()
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")

    if timeout_seconds is not None and timeout_seconds <= 0:
        timeout_seconds = None

    if log_path is None:
        proc = subprocess.Popen(
            list(cmd),
            env=env,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            _terminate_process(proc)
            raise subprocess.TimeoutExpired(list(cmd), timeout_seconds) from exc
        except KeyboardInterrupt:
            _terminate_process(proc)
            raise
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, list(cmd))
        return

    _ensure_dir(log_path.parent)
    tail_cache: List[str] = []
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(
            "\n\n===== {} =====\n$ {}\n".format(
                datetime.now().isoformat(timespec="seconds"),
                " ".join(cmd),
            )
        )
        logf.flush()
        proc = subprocess.Popen(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=True,
        )
        assert proc.stdout is not None
        start_ts = time.monotonic()
        try:
            while True:
                if timeout_seconds is not None and (time.monotonic() - start_ts) > timeout_seconds:
                    raise subprocess.TimeoutExpired(list(cmd), timeout_seconds)

                ready, _, _ = select.select([proc.stdout], [], [], 0.5)
                if ready:
                    line = proc.stdout.readline()
                    if line:
                        print(line, end="")
                        logf.write(line)
                        tail_cache.append(line.rstrip("\n"))
                        if len(tail_cache) > max(10, int(failure_tail_lines)):
                            tail_cache.pop(0)

                if proc.poll() is not None:
                    remainder = proc.stdout.read()
                    if remainder:
                        for line in remainder.splitlines(keepends=True):
                            print(line, end="")
                            logf.write(line)
                            tail_cache.append(line.rstrip("\n"))
                            if len(tail_cache) > max(10, int(failure_tail_lines)):
                                tail_cache.pop(0)
                    break
        except subprocess.TimeoutExpired as exc:
            _terminate_process(proc)
            logf.write(f"\n[COMMAND_TIMEOUT] {timeout_seconds}\n")
            logf.flush()
            print(
                f"[ERROR] Command timed out after {timeout_seconds:.1f}s. "
                f"Scenario log: {log_path}"
            )
            raise subprocess.TimeoutExpired(list(cmd), timeout_seconds) from exc
        except KeyboardInterrupt:
            _terminate_process(proc)
            raise

        returncode = proc.returncode
        logf.write(f"\n[COMMAND_EXIT] {returncode}\n")
        logf.flush()

    if returncode != 0:
        print(
            f"[ERROR] Command failed with {_format_exit_code(returncode)}. "
            f"Scenario log: {log_path}"
        )
        if returncode == -11 or returncode == 139:
            print(
                "[ERROR] Detected segmentation fault (SIGSEGV). "
                "This is usually a native crash in CARLA/OpenCV/extension code."
            )
        if tail_cache:
            print(f"[ERROR] Last {len(tail_cache)} log lines:")
            for line in tail_cache[-max(10, int(failure_tail_lines)) :]:
                print(line)
        raise subprocess.CalledProcessError(returncode, list(cmd))


def _terminate_process(proc: subprocess.Popen, *, grace_seconds: float = 10.0) -> None:
    if proc.poll() is not None:
        try:
            proc.wait(timeout=0.1)
        except Exception:
            pass
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        proc.terminate()
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.2)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except PermissionError:
            proc.kill()
    try:
        proc.wait(timeout=5.0)
    except Exception:
        pass


def _is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for_port_state(host: str, port: int, *, should_be_open: bool, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        current = _is_port_open(host, port, timeout=0.5)
        if current == should_be_open:
            return True
        time.sleep(0.5)
    return _is_port_open(host, port, timeout=0.5) == should_be_open


def _find_available_carla_port(
    *,
    host: str,
    preferred_port: int,
    tries: int,
    step: int,
    tm_offset: int,
) -> int:
    max_tries = max(1, int(tries))
    port_step = max(1, int(step))
    for i in range(max_tries):
        port = int(preferred_port) + i * port_step
        tm_port = port + int(tm_offset)
        if port > 65535 or tm_port > 65535:
            break
        if _is_port_open(host, port, timeout=0.5):
            continue
        if _is_port_open(host, tm_port, timeout=0.5):
            continue
        return port
    raise RuntimeError(
        f"Unable to find free CARLA/TM ports from base {preferred_port} "
        f"(tries={max_tries}, step={port_step}, tm_offset={tm_offset})."
    )


def _resolve_carla_script(path_arg: Optional[Path]) -> Path:
    candidates: List[Path] = []
    if path_arg is not None:
        candidates.append(path_arg.expanduser())
    which_path = shutil.which("CarlaUE4.sh")
    if which_path:
        candidates.append(Path(which_path))
    env_path = os.environ.get("CARLA_SCRIPT")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    env_root = os.environ.get("CARLA_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser() / "CarlaUE4.sh")
    candidates.extend(
        [
            Path("/opt/carla-simulator/CarlaUE4.sh"),
            Path("/opt/carla/CarlaUE4.sh"),
        ]
    )
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    searched = ", ".join(str(c) for c in candidates) if candidates else "<none>"
    raise FileNotFoundError(
        "Unable to locate CarlaUE4.sh. Provide --carla-script or set CARLA_SCRIPT/CARLA_ROOT. "
        f"Checked: {searched}"
    )


class CarlaServer:
    def __init__(
        self,
        *,
        script_path: Path,
        host: str,
        gpu: Optional[str],
        startup_timeout: float,
        extra_args: Sequence[str],
    ) -> None:
        self.script_path = script_path
        self.host = host
        self.gpu = gpu
        self.startup_timeout = float(startup_timeout)
        self.extra_args = list(extra_args)
        self.proc: Optional[subprocess.Popen] = None
        self.port: Optional[int] = None

    def start(self, port: int) -> None:
        self.stop()
        cmd = [str(self.script_path), f"-carla-rpc-port={int(port)}", "-RenderOffScreen"]
        cmd.extend(self.extra_args)
        env = os.environ.copy()
        if self.gpu is not None and str(self.gpu).strip():
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu).strip()

        print("[CMD]", " ".join(cmd))
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(self.script_path.parent),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.port = int(port)

        if not _wait_for_port_state(
            self.host,
            self.port,
            should_be_open=True,
            timeout_seconds=self.startup_timeout,
        ):
            self.stop()
            raise RuntimeError(
                f"CARLA did not open {self.host}:{self.port} within {self.startup_timeout:.1f}s"
            )
        print(f"[INFO] CARLA ready on {self.host}:{self.port}")

    def stop(self) -> None:
        if self.proc is None:
            self.port = None
            return
        proc = self.proc
        port = self.port
        self.proc = None
        self.port = None
        _terminate_process(proc, grace_seconds=8.0)
        if port is not None:
            _wait_for_port_state(self.host, int(port), should_be_open=False, timeout_seconds=8.0)
            print(f"[INFO] CARLA stopped on {self.host}:{int(port)}")


def _normalized_env_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    value = str(name).strip()
    return value or None


def _python_cmd(
    *,
    script_path: Path,
    script_args: Sequence[str],
    conda_env: Optional[str],
    fallback_python: str,
) -> List[str]:
    env_name = _normalized_env_name(conda_env)
    cmd: List[str] = []
    if env_name:
        cmd.extend(["conda", "run", "-n", env_name, "python", "-X", "faulthandler", "-u", str(script_path)])
    else:
        cmd.extend([fallback_python, "-X", "faulthandler", "-u", str(script_path)])
    cmd.extend(list(script_args))
    return cmd


def _scenario_dirs(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])


def _ego_indices(scenario_dir: Path) -> List[int]:
    indices: List[int] = []
    for child in scenario_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name.strip()
        if not name.isdigit():
            continue
        idx = int(name)
        if idx > 0:
            indices.append(idx)
    return sorted(set(indices))


def _scenario_video_outputs(
    scenario_name: str,
    ego_ids: Sequence[int],
    fullvideos_dir: Path,
) -> List[Path]:
    outputs: List[Path] = []
    for ego_id in ego_ids:
        outputs.append(fullvideos_dir / f"{scenario_name}_sidebyside_{ego_id}.mp4")
        outputs.append(fullvideos_dir / f"{scenario_name}_carla_{ego_id}.mp4")
        outputs.append(fullvideos_dir / f"{scenario_name}_real_{ego_id}.mp4")
    return outputs


def _is_completed_scenario(
    scenario_name: str,
    scenario_dir: Path,
    fullvideos_dir: Path,
) -> bool:
    ego_ids = _ego_indices(scenario_dir)
    if not ego_ids:
        return False
    expected_outputs = _scenario_video_outputs(scenario_name, ego_ids, fullvideos_dir)
    return all(path.is_file() and path.stat().st_size > 0 for path in expected_outputs)


def _dir_names(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    return {p.name for p in path.iterdir() if p.is_dir()}


def _pick_run_image_dir(image_root: Path, before_names: Iterable[str]) -> Path:
    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")

    dirs = [p for p in image_root.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No image run folders found in: {image_root}")

    before_set = set(before_names)
    new_dirs = [p for p in dirs if p.name not in before_set]
    if len(new_dirs) == 1:
        return new_dirs[0]
    if new_dirs:
        return max(new_dirs, key=lambda p: p.stat().st_mtime)
    return max(dirs, key=lambda p: p.stat().st_mtime)


def _pick_run_image_dir_soft(image_root: Path, before_names: Iterable[str]) -> Optional[Path]:
    if not image_root.exists():
        return None

    dirs = [p for p in image_root.iterdir() if p.is_dir()]
    if not dirs:
        return None

    before_set = set(before_names)
    new_dirs = [p for p in dirs if p.name not in before_set]
    if len(new_dirs) == 1:
        return new_dirs[0]
    if new_dirs:
        return max(new_dirs, key=lambda p: p.stat().st_mtime)
    return max(dirs, key=lambda p: p.stat().st_mtime)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run yaml_to_carla_log + run_custom_eval + gen_video for all scenarios in train1."
    )
    parser.add_argument(
        "--train1-root",
        type=Path,
        default=Path("/data2/marco/CoLMDriver/v2xpnp/dataset/train4"),
        help="Root folder containing scenario subfolders.",
    )
    parser.add_argument(
        "--coord-json",
        type=Path,
        default=Path("/data2/marco/CoLMDriver/v2xpnp/map/ucla_map_offset_carla.json"),
        help="Path to map offset json used by yaml_to_carla_log.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/data2/marco/CoLMDriver/results/results_driving_custom"),
        help="Root results folder used by run_custom_eval.",
    )
    parser.add_argument(
        "--fullvideos-dir",
        type=Path,
        default=Path("/data2/marco/CoLMDriver/results/results_driving_custom/fullvideos"),
        help="Output directory for generated mp4 videos.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=2005,
        help="Preferred base CARLA port (actual port is selected dynamically).",
    )
    parser.add_argument(
        "--yaml-to-log-carla-port",
        type=int,
        default=2025,
        help="Deprecated fixed CARLA port. Dynamic per-scenario CARLA port is used instead.",
    )
    parser.add_argument(
        "--carla-script",
        type=Path,
        default=None,
        help="Path to CarlaUE4.sh. If omitted, tries CARLA_SCRIPT/CARLA_ROOT.",
    )
    parser.add_argument(
        "--carla-host",
        type=str,
        default="127.0.0.1",
        help="CARLA host used for health checks and client connections.",
    )
    parser.add_argument(
        "--carla-cuda-visible-devices",
        type=str,
        default="4",
        help="Value for CUDA_VISIBLE_DEVICES when launching CARLA (empty to keep current env).",
    )
    parser.add_argument(
        "--carla-port-tries",
        type=int,
        default=30,
        help="How many candidate ports to probe from --port.",
    )
    parser.add_argument(
        "--carla-port-step",
        type=int,
        default=1,
        help="Port increment while probing for a free CARLA port.",
    )
    parser.add_argument(
        "--tm-port-offset",
        type=int,
        default=5,
        help="Traffic manager port offset from selected CARLA port.",
    )
    parser.add_argument(
        "--carla-startup-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for CARLA RPC port to become available.",
    )
    parser.add_argument(
        "--carla-arg",
        action="append",
        default=[],
        help="Additional argument forwarded to CarlaUE4.sh (repeatable).",
    )
    parser.add_argument(
        "--eval-timeout-seconds",
        type=float,
        default=1800.0,
        help=(
            "Timeout for run_custom_eval. If exceeded, continue by generating videos from whatever "
            "frames exist and move to next scenario."
        ),
    )
    parser.add_argument("--fps", type=float, default=10.0, help="FPS for gen_video.")
    parser.add_argument(
        "--resize-factor",
        type=int,
        default=1,
        help="Resize factor for gen_video.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Scenario folder name to run (repeatable). If omitted, run all subfolders.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Fallback Python executable when a command is not run via conda.",
    )
    parser.add_argument(
        "--yaml-conda-env",
        type=str,
        default="b2d_zoo",
        help="Conda env for yaml_to_carla_log stage (default: b2d_zoo).",
    )
    parser.add_argument(
        "--eval-conda-env",
        type=str,
        default="colmdrivermarco2",
        help="Conda env for run_custom_eval stage (default: colmdrivermarco2).",
    )
    parser.add_argument(
        "--video-conda-env",
        type=str,
        default="colmdrivermarco2",
        help="Conda env for gen_video stage (default: colmdrivermarco2).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to next scenario when one fails.",
    )
    parser.add_argument(
        "--skip-existing-videos",
        action="store_true",
        help="Skip gen_video if the output mp4 already exists.",
    )
    parser.add_argument(
        "--failure-tail-lines",
        type=int,
        default=80,
        help="How many trailing log lines to print automatically when a command fails.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    train1_root: Path = args.train1_root.expanduser().resolve()
    coord_json: Path = args.coord_json.expanduser().resolve()
    results_root: Path = args.results_root.expanduser().resolve()
    fullvideos_dir: Path = args.fullvideos_dir.expanduser().resolve()
    python_bin: str = args.python
    repo_root = Path(__file__).resolve().parents[1]
    yaml_to_log_script = repo_root / "v2xpnp" / "scripts" / "yaml_to_carla_log.py"
    eval_script = repo_root / "tools" / "run_custom_eval.py"
    gen_video_script = repo_root / "visualization" / "gen_video.py"

    if not train1_root.exists():
        raise FileNotFoundError(f"train1 root not found: {train1_root}")
    if not coord_json.exists():
        raise FileNotFoundError(f"coord json not found: {coord_json}")
    if not yaml_to_log_script.exists():
        raise FileNotFoundError(f"yaml_to_carla_log script not found: {yaml_to_log_script}")
    if not eval_script.exists():
        raise FileNotFoundError(f"run_custom_eval script not found: {eval_script}")
    if not gen_video_script.exists():
        raise FileNotFoundError(f"gen_video script not found: {gen_video_script}")

    carla_server: Optional[CarlaServer] = None
    prev_sigterm = signal.getsignal(signal.SIGTERM)

    def _handle_sigterm(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_sigterm)

    _ensure_dir(fullvideos_dir)

    scenarios = _scenario_dirs(train1_root)
    if args.scenario:
        wanted = set(args.scenario)
        scenarios = [s for s in scenarios if s.name in wanted]

    if not scenarios:
        print("No scenarios found to process.")
        return 0

    print(f"Found {len(scenarios)} scenario(s) under {train1_root}")

    try:
        for scenario_dir in scenarios:
            scenario_name = scenario_dir.name
            if _is_completed_scenario(scenario_name, scenario_dir, fullvideos_dir):
                print(f"\n=== Scenario: {scenario_name} ===")
                print("[SKIP] Scenario already completed (all expected videos exist).")
                continue

            scenario_log_dir = results_root / scenario_name / "batch_logs"
            scenario_log_path = scenario_log_dir / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            print(f"\n=== Scenario: {scenario_name} ===")
            print(f"[INFO] scenario log: {scenario_log_path}")

            try:
                selected_port = _find_available_carla_port(
                    host=str(args.carla_host),
                    preferred_port=int(args.port),
                    tries=int(args.carla_port_tries),
                    step=int(args.carla_port_step),
                    tm_offset=int(args.tm_port_offset),
                )
                tm_port = selected_port + int(args.tm_port_offset)
                if args.dry_run:
                    print(
                        f"[INFO] [dry-run] would start CARLA at {args.carla_host}:{selected_port} "
                        f"(tm_port={tm_port})"
                    )
                else:
                    if carla_server is None:
                        carla_script = _resolve_carla_script(args.carla_script)
                        carla_server = CarlaServer(
                            script_path=carla_script,
                            host=str(args.carla_host),
                            gpu=args.carla_cuda_visible_devices,
                            startup_timeout=float(args.carla_startup_timeout),
                            extra_args=args.carla_arg,
                        )
                    carla_server.start(selected_port)

                # Step 1: yaml_to_carla_log
                yaml_args = [
                    "--scenario-dir",
                    str(scenario_dir),
                    "--coord-json",
                    str(coord_json),
                    "--use-carla-map",
                    "--encode-timing",
                    "--carla-host",
                    str(args.carla_host),
                    "--carla-port",
                    str(selected_port),
                ]
                cmd_yaml = _python_cmd(
                    script_path=yaml_to_log_script,
                    script_args=yaml_args,
                    conda_env=args.yaml_conda_env,
                    fallback_python=python_bin,
                )
                _run(
                    cmd_yaml,
                    dry_run=args.dry_run,
                    log_path=scenario_log_path,
                    failure_tail_lines=args.failure_tail_lines,
                )

                routes_dir = scenario_dir / "carla_log_export"
                if not args.dry_run and not routes_dir.exists():
                    raise FileNotFoundError(f"Expected routes dir not found: {routes_dir}")

                # Track image run folders before eval.
                image_root = results_root / scenario_name / "image"
                before_names = _dir_names(image_root)

                # Step 2: run_custom_eval
                eval_args = [
                    "--planner",
                    "log-replay",
                    "--port",
                    str(selected_port),
                    "--traffic-manager-port",
                    str(tm_port),
                    "--routes-dir",
                    str(routes_dir),
                    "--image-save-interval",
                    "1",
                    "--capture-logreplay-images",
                    "--results-tag",
                    scenario_name,
                    "--normalize-actor-z",
                ]
                cmd_eval = _python_cmd(
                    script_path=eval_script,
                    script_args=eval_args,
                    conda_env=args.eval_conda_env,
                    fallback_python=python_bin,
                )

                eval_failed = False
                try:
                    _run(
                        cmd_eval,
                        dry_run=args.dry_run,
                        log_path=scenario_log_path,
                        failure_tail_lines=args.failure_tail_lines,
                        timeout_seconds=args.eval_timeout_seconds,
                    )
                except subprocess.TimeoutExpired:
                    eval_failed = True
                    print(
                        "[WARN] run_custom_eval timed out; assuming CARLA/Eval got stuck. "
                        "Will generate videos from any available frames and continue."
                    )
                except subprocess.CalledProcessError as exc:
                    eval_failed = True
                    print(
                        f"[WARN] run_custom_eval exited with {_format_exit_code(exc.returncode)}. "
                        "Will generate videos from any available frames and continue."
                    )

                # Discover this run's image folder.
                if args.dry_run:
                    run_image_dir: Optional[Path] = image_root
                elif eval_failed:
                    run_image_dir = _pick_run_image_dir_soft(image_root, before_names)
                else:
                    run_image_dir = _pick_run_image_dir(image_root, before_names)

                if run_image_dir is None:
                    print(f"[WARN] No image run directory found under {image_root}.")
                else:
                    print(f"[INFO] image run dir: {run_image_dir}")

                # Step 3: per-ego videos (side-by-side, carla-only, real-only).
                ego_ids = _ego_indices(scenario_dir)
                if not ego_ids:
                    print(f"[WARN] No positive ego-index folders found in {scenario_dir}; skipping video step.")
                    continue

                for ego_id in ego_ids:
                    real_cam_dir = scenario_dir / str(ego_id)
                    side_dir = (
                        run_image_dir / "logreplayimages" / f"logreplay_rgb_{ego_id - 1}"
                        if run_image_dir is not None
                        else None
                    )
                    out_side = fullvideos_dir / f"{scenario_name}_sidebyside_{ego_id}.mp4"
                    out_carla = fullvideos_dir / f"{scenario_name}_carla_{ego_id}.mp4"
                    out_real = fullvideos_dir / f"{scenario_name}_real_{ego_id}.mp4"

                    real_exists = bool(args.dry_run or real_cam_dir.exists())
                    side_exists = bool(side_dir is not None and (args.dry_run or side_dir.exists()))

                    if not real_exists:
                        print(f"[WARN] Missing real cam folder for ego {ego_id}: {real_cam_dir}")
                    if side_dir is not None and not side_exists:
                        print(f"[WARN] Missing logreplay folder for ego {ego_id}: {side_dir}")

                    jobs = []
                    if real_exists and side_exists and side_dir is not None:
                        jobs.append(
                            (
                                "sidebyside",
                                out_side,
                                _python_cmd(
                                    script_path=gen_video_script,
                                    script_args=[
                                        str(real_cam_dir),
                                        "--only-suffix",
                                        "cam1",
                                        "--side-by-side-dir",
                                        str(side_dir),
                                        "--fps",
                                        str(args.fps),
                                        "--resize-factor",
                                        str(args.resize_factor),
                                        "--output",
                                        str(out_side),
                                    ],
                                    conda_env=args.video_conda_env,
                                    fallback_python=python_bin,
                                ),
                            )
                        )
                    if side_exists and side_dir is not None:
                        jobs.append(
                            (
                                "carla",
                                out_carla,
                                _python_cmd(
                                    script_path=gen_video_script,
                                    script_args=[
                                        str(side_dir),
                                        "--fps",
                                        str(args.fps),
                                        "--resize-factor",
                                        str(args.resize_factor),
                                        "--output",
                                        str(out_carla),
                                    ],
                                    conda_env=args.video_conda_env,
                                    fallback_python=python_bin,
                                ),
                            )
                        )
                    if real_exists:
                        jobs.append(
                            (
                                "real",
                                out_real,
                                _python_cmd(
                                    script_path=gen_video_script,
                                    script_args=[
                                        str(real_cam_dir),
                                        "--only-suffix",
                                        "cam1",
                                        "--fps",
                                        str(args.fps),
                                        "--resize-factor",
                                        str(args.resize_factor),
                                        "--output",
                                        str(out_real),
                                    ],
                                    conda_env=args.video_conda_env,
                                    fallback_python=python_bin,
                                ),
                            )
                        )

                    if not jobs:
                        print(f"[WARN] No video inputs available for ego {ego_id}; skipping.")
                        continue

                    for tag, out_mp4, cmd_vid in jobs:
                        if args.skip_existing_videos and out_mp4.exists():
                            print(f"[SKIP] existing {tag} video: {out_mp4}")
                            continue
                        _run(
                            cmd_vid,
                            dry_run=args.dry_run,
                            log_path=scenario_log_path,
                            failure_tail_lines=args.failure_tail_lines,
                        )

            except Exception as exc:  # pylint: disable=broad-except
                print(f"[ERROR] Scenario failed: {scenario_name} -> {exc}")
                print(f"[ERROR] See scenario log: {scenario_log_path}")
                if not args.continue_on_error:
                    return 1
            finally:
                if carla_server is not None:
                    carla_server.stop()
    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user. Shutting down CARLA and exiting.")
        return 130
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)
        if carla_server is not None:
            carla_server.stop()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
