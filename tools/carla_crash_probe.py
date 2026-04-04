#!/usr/bin/env python3
"""Probe CARLA crash timing across startup, map load, weather, and route phases."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from common.carla_connection_events import (
        create_logged_client,
        install_process_lifecycle_logging,
        log_carla_event,
        log_process_exception,
    )
except Exception:  # pragma: no cover - probe should still run standalone
    def create_logged_client(carla_module, host, port, *, timeout_s=None, context="", attempt=None, process_name=None):
        del context, attempt, process_name
        client = carla_module.Client(host, int(port))
        if timeout_s is not None:
            client.set_timeout(float(timeout_s))
        return client

    def install_process_lifecycle_logging(process_name, *, env_keys=None):
        del process_name, env_keys

    def log_carla_event(event_type, *, process_name=None, **fields):
        del event_type, process_name, fields

    def log_process_exception(exc, *, process_name, where=""):
        del exc, process_name, where


CRASH_PATTERNS = [
    "LowLevelFatalError",
    "Failed to find parameter collection buffer with GUID",
    "Signal 11",
    "Segmentation fault",
    "Unhandled Exception",
    "Assertion failed",
]
CARLA_CRASH_PROBE_PROCESS_NAME = "carla_crash_probe"


@dataclass
class ProbeResult:
    case: str
    phase: str
    ok: bool
    detail: str
    elapsed_s: float
    server_alive: bool
    crash_pattern: str = ""
    log_tail: str = ""


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            return sock.connect_ex((host, port)) == 0
        finally:
            sock.close()
    except PermissionError:
        # Some constrained sandboxes block socket syscalls entirely.
        return False


def _wait_for_port(host: str, port: int, should_be_open: bool, timeout_s: float) -> bool:
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if _is_port_open(host, port, timeout=0.5) == should_be_open:
            return True
        time.sleep(0.4)
    return _is_port_open(host, port, timeout=0.5) == should_be_open


def _tail(path: Path, lines: int = 120) -> str:
    if not path.exists():
        return ""
    try:
        data = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    return "\n".join(data[-lines:])


def _find_crash_pattern(text: str) -> str:
    lower = text.lower()
    for pattern in CRASH_PATTERNS:
        if pattern.lower() in lower:
            return pattern
    return ""


def _terminate_process(proc: subprocess.Popen[Any], grace_s: float = 8.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    deadline = time.monotonic() + grace_s
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
            pass


def _load_carla_module(carla_root: Path):
    try:
        import carla  # type: ignore

        return carla
    except ModuleNotFoundError:
        dist = carla_root / "PythonAPI" / "carla" / "dist"
        if dist.exists():
            for path in sorted(dist.glob("carla-*.egg")) + sorted(dist.glob("carla-*.whl")):
                sys.path.append(str(path))
        import carla  # type: ignore

        return carla


def _candidate_map_names(client: Any, requested: str) -> List[str]:
    candidates = [
        requested,
        f"/Game/Carla/Maps/{requested}",
        f"/Game/Carla/Maps/{requested}/{requested}",
    ]
    try:
        available = client.get_available_maps()
    except Exception:
        available = []
    target = requested.lower()
    for m in available:
        if target in str(m).lower():
            candidates.insert(0, str(m))
    ordered: List[str] = []
    seen = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        ordered.append(c)
    return ordered


def _parse_route(route_xml: Path, route_id: str = "") -> Dict[str, Any]:
    tree = ET.parse(route_xml)
    root = tree.getroot()
    routes = root.findall(".//route")
    if not routes:
        raise RuntimeError(f"No <route> found in {route_xml}")

    chosen = None
    if route_id:
        for r in routes:
            if str(r.attrib.get("id", "")).strip() == route_id.strip():
                chosen = r
                break
        if chosen is None:
            raise RuntimeError(f"Route id '{route_id}' not found in {route_xml}")
    else:
        chosen = routes[0]

    waypoints: List[Tuple[float, float, float]] = []
    for wp in chosen.findall("waypoint"):
        x = float(wp.attrib.get("x", "0.0"))
        y = float(wp.attrib.get("y", "0.0"))
        z = float(wp.attrib.get("z", "0.0"))
        waypoints.append((x, y, z))

    weather_attrs: Dict[str, str] = {}
    weather_node = chosen.find("weather")
    if weather_node is not None:
        weather_attrs = dict(weather_node.attrib)

    return {
        "id": str(chosen.attrib.get("id", "")),
        "town": str(chosen.attrib.get("town", "")),
        "waypoints": waypoints,
        "weather": weather_attrs,
    }


class CarlaServer:
    def __init__(
        self,
        *,
        carla_script: Path,
        host: str,
        port: int,
        gpu: str,
        startup_timeout_s: float,
        extra_args: Sequence[str],
        log_path: Path,
    ) -> None:
        self.carla_script = carla_script
        self.host = host
        self.port = int(port)
        self.gpu = gpu.strip()
        self.startup_timeout_s = float(startup_timeout_s)
        self.extra_args = list(extra_args)
        self.log_path = log_path
        self.proc: Optional[subprocess.Popen[Any]] = None
        self.log_handle = None

    def start(self) -> Tuple[bool, str]:
        self.stop()
        log_carla_event(
            "CRASH_PROBE_SERVER_START_BEGIN",
            process_name=CARLA_CRASH_PROBE_PROCESS_NAME,
            host=self.host,
            port=self.port,
            script=str(self.carla_script),
            gpu=self.gpu,
            startup_timeout_s=self.startup_timeout_s,
            cmd_extra=" ".join(self.extra_args),
            log_path=str(self.log_path),
        )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.log_path.open("w", encoding="utf-8")
        cmd = [
            str(self.carla_script),
            f"-carla-rpc-port={self.port}",
            "-RenderOffScreen",
        ]
        cmd.extend(self.extra_args)

        env = os.environ.copy()
        if self.gpu:
            env["CUDA_VISIBLE_DEVICES"] = self.gpu

        self.proc = subprocess.Popen(
            cmd,
            cwd=str(self.carla_script.parent),
            env=env,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        ok = _wait_for_port(self.host, self.port, should_be_open=True, timeout_s=self.startup_timeout_s)
        if ok:
            log_carla_event(
                "CRASH_PROBE_SERVER_START_READY",
                process_name=CARLA_CRASH_PROBE_PROCESS_NAME,
                host=self.host,
                port=self.port,
                pid=None if self.proc is None else self.proc.pid,
            )
            return True, "CARLA RPC port opened"
        log_carla_event(
            "CRASH_PROBE_SERVER_START_FAIL",
            process_name=CARLA_CRASH_PROBE_PROCESS_NAME,
            host=self.host,
            port=self.port,
            pid=None if self.proc is None else self.proc.pid,
        )
        return False, "CARLA RPC port did not open before timeout"

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        if self.proc is not None:
            log_carla_event(
                "CRASH_PROBE_SERVER_STOP_BEGIN",
                process_name=CARLA_CRASH_PROBE_PROCESS_NAME,
                host=self.host,
                port=self.port,
                pid=self.proc.pid,
            )
            _terminate_process(self.proc, grace_s=8.0)
            _wait_for_port(self.host, self.port, should_be_open=False, timeout_s=8.0)
            self.proc = None
            log_carla_event(
                "CRASH_PROBE_SERVER_STOP_END",
                process_name=CARLA_CRASH_PROBE_PROCESS_NAME,
                host=self.host,
                port=self.port,
            )
        if self.log_handle is not None:
            self.log_handle.flush()
            self.log_handle.close()
            self.log_handle = None


def _mk_result(
    *,
    case: str,
    phase: str,
    ok: bool,
    detail: str,
    elapsed_s: float,
    server_alive: bool,
    log_path: Path,
) -> ProbeResult:
    tail = _tail(log_path, lines=120)
    crash = _find_crash_pattern(tail)
    return ProbeResult(
        case=case,
        phase=phase,
        ok=ok,
        detail=detail,
        elapsed_s=round(float(elapsed_s), 3),
        server_alive=bool(server_alive),
        crash_pattern=crash,
        log_tail=tail,
    )


def _run_case(
    *,
    case_name: str,
    map_name: str,
    route_info: Optional[Dict[str, Any]],
    carla: Any,
    server: CarlaServer,
    rpc_timeout_s: float,
    stable_s: float,
    weather_presets: Sequence[str],
    waypoint_limit: int,
) -> List[ProbeResult]:
    results: List[ProbeResult] = []
    log_carla_event(
        "CRASH_PROBE_CASE_BEGIN",
        process_name=CARLA_CRASH_PROBE_PROCESS_NAME,
        case=case_name,
        map_name=map_name,
        has_route=int(bool(route_info)),
        rpc_timeout_s=float(rpc_timeout_s),
        stable_s=float(stable_s),
        weather_presets=",".join(str(p) for p in weather_presets),
        waypoint_limit=int(waypoint_limit),
    )
    t0 = time.monotonic()
    started, msg = server.start()
    results.append(
        _mk_result(
            case=case_name,
            phase="server_start",
            ok=started,
            detail=msg,
            elapsed_s=time.monotonic() - t0,
            server_alive=server.alive(),
            log_path=server.log_path,
        )
    )
    if not started:
        return results

    client = create_logged_client(
        carla,
        server.host,
        server.port,
        timeout_s=float(rpc_timeout_s),
        context=f"crash_probe:{case_name}",
        process_name=CARLA_CRASH_PROBE_PROCESS_NAME,
    )

    t1 = time.monotonic()
    try:
        world = client.get_world()
        current_map = world.get_map().name
        detail = f"Connected to server; current map={current_map}"
        ok = True
    except Exception as exc:
        world = None
        detail = f"client.get_world failed: {exc}"
        ok = False
    results.append(
        _mk_result(
            case=case_name,
            phase="client_connect",
            ok=ok,
            detail=detail,
            elapsed_s=time.monotonic() - t1,
            server_alive=server.alive(),
            log_path=server.log_path,
        )
    )
    if not ok or world is None:
        return results

    if map_name:
        t2 = time.monotonic()
        loaded = None
        last_exc: Optional[Exception] = None
        for cand in _candidate_map_names(client, map_name):
            try:
                loaded = client.load_world(cand)
                detail = f"load_world succeeded with '{cand}'"
                ok = True
                break
            except Exception as exc:
                last_exc = exc
                continue
        if loaded is None:
            ok = False
            detail = f"load_world failed for '{map_name}': {last_exc}"
            world = None
        else:
            world = loaded
        results.append(
            _mk_result(
                case=case_name,
                phase="load_world",
                ok=ok,
                detail=detail,
                elapsed_s=time.monotonic() - t2,
                server_alive=server.alive(),
                log_path=server.log_path,
            )
        )
        if not ok or world is None:
            return results

    for preset in weather_presets:
        t3 = time.monotonic()
        try:
            weather = getattr(carla.WeatherParameters, preset)
            world.set_weather(weather)
            try:
                world.wait_for_tick(5.0)
            except Exception:
                time.sleep(1.0)
            ok = True
            detail = f"set_weather({preset}) succeeded"
        except Exception as exc:
            ok = False
            detail = f"set_weather({preset}) failed: {exc}"
        results.append(
            _mk_result(
                case=case_name,
                phase=f"set_weather:{preset}",
                ok=ok,
                detail=detail,
                elapsed_s=time.monotonic() - t3,
                server_alive=server.alive(),
                log_path=server.log_path,
            )
        )
        if not ok or not server.alive():
            return results

    if route_info:
        town = str(route_info.get("town", "")).strip()
        if town and map_name and town.lower() not in map_name.lower():
            results.append(
                _mk_result(
                    case=case_name,
                    phase="route_check",
                    ok=True,
                    detail=f"Skipped route waypoint probe (route town={town}, map={map_name})",
                    elapsed_s=0.0,
                    server_alive=server.alive(),
                    log_path=server.log_path,
                )
            )
        else:
            t4 = time.monotonic()
            try:
                carla_map = world.get_map()
                checked = 0
                for x, y, z in route_info.get("waypoints", [])[: max(1, int(waypoint_limit))]:
                    loc = carla.Location(x=float(x), y=float(y), z=float(z))
                    _ = carla_map.get_waypoint(
                        loc,
                        project_to_road=True,
                        lane_type=carla.LaneType.Driving,
                    )
                    checked += 1
                ok = True
                detail = f"Projected {checked} route waypoints to map lanes"
            except Exception as exc:
                ok = False
                detail = f"Route waypoint projection failed: {exc}"
            results.append(
                _mk_result(
                    case=case_name,
                    phase="route_waypoints",
                    ok=ok,
                    detail=detail,
                    elapsed_s=time.monotonic() - t4,
                    server_alive=server.alive(),
                    log_path=server.log_path,
                )
            )
            if not ok or not server.alive():
                return results

    if stable_s > 0:
        t5 = time.monotonic()
        time.sleep(float(stable_s))
        ok = server.alive()
        detail = f"Server remained alive for stability window ({stable_s:.1f}s)"
        if not ok:
            detail = "Server died during post-phase stability window"
        results.append(
            _mk_result(
                case=case_name,
                phase="stability_wait",
                ok=ok,
                detail=detail,
                elapsed_s=time.monotonic() - t5,
                server_alive=server.alive(),
                log_path=server.log_path,
            )
        )

    log_carla_event(
        "CRASH_PROBE_CASE_END",
        process_name=CARLA_CRASH_PROBE_PROCESS_NAME,
        case=case_name,
        phases=len(results),
        failed_phases=sum(1 for item in results if not item.ok),
        server_alive=int(bool(server.alive())),
    )
    return results


def _write_summary(results: List[ProbeResult], out_dir: Path) -> None:
    summary_path = out_dir / "summary.md"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# CARLA Crash Probe Summary\n\n")
        f.write("| case | phase | ok | server_alive | crash_pattern | detail |\n")
        f.write("|---|---|---:|---:|---|---|\n")
        for r in results:
            cp = r.crash_pattern or "-"
            detail = r.detail.replace("|", "/")
            f.write(
                f"| {r.case} | {r.phase} | {int(r.ok)} | {int(r.server_alive)} | {cp} | {detail} |\n"
            )


def main() -> int:
    install_process_lifecycle_logging(
        CARLA_CRASH_PROBE_PROCESS_NAME,
        env_keys=("CARLA_ROOTCAUSE_LOGDIR", "CARLA_CONNECTION_EVENTS_LOG", "CUDA_VISIBLE_DEVICES"),
    )
    parser = argparse.ArgumentParser(description="Probe CARLA crash phase across maps/routes.")
    parser.add_argument("--carla-root", type=Path, default=Path("/data2/marco/CoLMDriver/carla912"))
    parser.add_argument("--carla-script", type=Path, default=None, help="Path to CarlaUE4.sh")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2066, help="CARLA RPC port")
    parser.add_argument("--gpu", default="", help="CUDA_VISIBLE_DEVICES value (empty = unchanged)")
    parser.add_argument("--maps", default="Town01,Town05,Town06,ucla_v2")
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--rpc-timeout", type=float, default=20.0)
    parser.add_argument("--stable-seconds", type=float, default=4.0)
    parser.add_argument("--weather-presets", default="ClearNoon")
    parser.add_argument("--extra-arg", action="append", default=[], help="Extra arg passed to CarlaUE4.sh")
    parser.add_argument("--route-xml", type=Path, default=None)
    parser.add_argument("--route-id", default="")
    parser.add_argument("--route-waypoint-limit", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    carla_root = args.carla_root.resolve()
    carla_script = args.carla_script.resolve() if args.carla_script else (carla_root / "CarlaUE4.sh")
    if not carla_script.exists():
        print(f"[ERROR] CarlaUE4.sh not found: {carla_script}", file=sys.stderr)
        return 2

    if args.output_dir is not None:
        out_dir = args.output_dir.resolve()
    else:
        out_dir = (Path.cwd() / "debug_runs" / f"carla_crash_probe_{_timestamp()}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    route_info = None
    if args.route_xml is not None:
        route_info = _parse_route(args.route_xml.resolve(), args.route_id)

    maps = [m.strip() for m in str(args.maps).split(",") if m.strip()]
    weather_presets = [w.strip() for w in str(args.weather_presets).split(",") if w.strip()]
    if not weather_presets:
        weather_presets = ["ClearNoon"]

    try:
        carla = _load_carla_module(carla_root)
    except Exception as exc:
        print(f"[ERROR] Unable to import carla Python module: {exc}", file=sys.stderr)
        return 2

    config = {
        "carla_script": str(carla_script),
        "host": args.host,
        "port": int(args.port),
        "gpu": str(args.gpu),
        "maps": maps,
        "weather_presets": weather_presets,
        "route_xml": str(args.route_xml.resolve()) if args.route_xml else "",
        "route_id": str(args.route_id),
        "route_waypoint_limit": int(args.route_waypoint_limit),
        "startup_timeout": float(args.startup_timeout),
        "rpc_timeout": float(args.rpc_timeout),
        "stable_seconds": float(args.stable_seconds),
        "extra_args": list(args.extra_arg),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    if route_info:
        (out_dir / "route_info.json").write_text(json.dumps(route_info, indent=2) + "\n", encoding="utf-8")

    all_results: List[ProbeResult] = []
    cases: List[Tuple[str, str]] = [("startup_only", "")]
    cases.extend((f"map:{m}", m) for m in maps)
    if route_info:
        route_town = str(route_info.get("town", "")).strip()
        if route_town:
            cases.append((f"route:{route_town}", route_town))

    for idx, (case_name, map_name) in enumerate(cases, start=1):
        case_log = out_dir / f"{idx:02d}_{case_name.replace(':', '_')}.log"
        server = CarlaServer(
            carla_script=carla_script,
            host=args.host,
            port=int(args.port),
            gpu=str(args.gpu),
            startup_timeout_s=float(args.startup_timeout),
            extra_args=list(args.extra_arg),
            log_path=case_log,
        )
        try:
            results = _run_case(
                case_name=case_name,
                map_name=map_name,
                route_info=route_info if case_name.startswith("route:") else None,
                carla=carla,
                server=server,
                rpc_timeout_s=float(args.rpc_timeout),
                stable_s=float(args.stable_seconds),
                weather_presets=weather_presets,
                waypoint_limit=int(args.route_waypoint_limit),
            )
            all_results.extend(results)
        finally:
            server.stop()

    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps([asdict(r) for r in all_results], indent=2) + "\n", encoding="utf-8")
    _write_summary(all_results, out_dir)

    failures = [r for r in all_results if not r.ok]
    print(f"[INFO] Output dir: {out_dir}")
    print(f"[INFO] Results: {json_path}")
    print(f"[INFO] Summary: {out_dir / 'summary.md'}")
    print(f"[INFO] Failures: {len(failures)} / {len(all_results)}")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log_process_exception(exc, process_name=CARLA_CRASH_PROBE_PROCESS_NAME, where="__main__")
        raise
