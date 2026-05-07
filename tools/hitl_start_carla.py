#!/usr/bin/env python3
"""Start CARLA for HITL experiments and verify it is ready.

This is a thin wrapper around `tools.run_custom_eval.CarlaProcessManager`
(the same launcher the canonical eval uses). It exposes:

  * `start_carla(...)`         — launches CARLA, returns (manager, bundle).
  * `wait_for_carla_ready(...)` — explicit RPC round-trip probe with retries.
  * `stop_carla(manager)`      — graceful teardown (idempotent).

and a CLI for manual smoke testing:

    python -m tools.hitl_start_carla --port 4010
    python -m tools.hitl_start_carla --port 4010 --probe-only
    python -m tools.hitl_start_carla --port 4010 --probe-existing
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOLS_DIR.parent
# Both paths are needed: repo root for `tools.run_custom_eval`, tools dir for
# the sibling-style imports it does internally (e.g. `from
# setup_scenario_from_zip import ...`).
for _p in (_REPO_ROOT, _TOOLS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _resolve_carla_root(override: str | None) -> Path:
    if override:
        return Path(override).resolve()
    env = os.environ.get("CARLA_ROOT")
    if env:
        return Path(env).resolve()
    return (_REPO_ROOT / "carla912").resolve()


def _import_carla(carla_root: Path):
    """Import the carla python client matching this interpreter version."""
    from tools.run_custom_eval import _ensure_carla_import_for_overlay
    return _ensure_carla_import_for_overlay(carla_root)


@dataclass
class CarlaHandle:
    """What `start_carla` returns. Hold onto this and call `stop()` when done."""
    manager: object  # CarlaProcessManager
    port: int
    tm_port: int
    host: str
    carla_root: Path

    def stop(self, timeout: float = 15.0) -> None:
        try:
            self.manager.stop(timeout=timeout)
        except Exception as exc:  # teardown is best-effort
            print(f"[hitl_start_carla] stop() raised: {exc!r}", file=sys.stderr)


def start_carla(
    *,
    host: str = "127.0.0.1",
    port: int = 4010,
    tm_port_offset: int = 5,
    extra_args: Sequence[str] = ("-RenderOffScreen",),
    carla_root: str | None = None,
    port_tries: int = 4,
    port_step: int = 10,
    quiet: bool = False,
    server_log_path: str | None = None,
) -> CarlaHandle:
    """Launch CARLA and wait until its RPC port accepts connections.

    Returns a `CarlaHandle` with the actual selected port (which may differ
    from the requested one if `port_tries` > 1 and the requested port was busy).

    Does NOT do the explicit `get_world()` round-trip — call
    `wait_for_carla_ready` after this if you want that guarantee.
    """
    from tools.run_custom_eval import CarlaProcessManager

    root = _resolve_carla_root(carla_root)
    if not (root / "CarlaUE4.sh").exists():
        raise FileNotFoundError(
            f"CARLA install not found at {root}. Set CARLA_ROOT or pass --carla-root."
        )

    env = os.environ.copy()
    log_path_obj: Path | None = None
    if server_log_path:
        log_path_obj = Path(server_log_path).expanduser().resolve()
        log_path_obj.parent.mkdir(parents=True, exist_ok=True)
    manager = CarlaProcessManager(
        carla_root=root,
        host=host,
        port=port,
        tm_port_offset=tm_port_offset,
        extra_args=list(extra_args),
        env=env,
        port_tries=port_tries,
        port_step=port_step,
        quiet=quiet,
        server_log_path=log_path_obj,
    )
    bundle = manager.start()  # blocks until RPC port is open (internal probe)
    return CarlaHandle(
        manager=manager,
        port=bundle.port,
        tm_port=bundle.tm_port,
        host=host,
        carla_root=root,
    )


def wait_for_carla_ready(
    *,
    host: str,
    port: int,
    timeout_s: float = 60.0,
    retry_interval_s: float = 1.0,
    client_timeout_s: float = 10.0,
    carla_root: str | None = None,
) -> None:
    """Block until `carla.Client(host, port).get_world()` succeeds.

    Raises `TimeoutError` with the underlying error if the budget is exhausted.
    """
    root = _resolve_carla_root(carla_root)
    carla = _import_carla(root)

    deadline = time.monotonic() + max(timeout_s, 1.0)
    last_exc: Exception | None = None
    attempts = 0
    while time.monotonic() < deadline:
        attempts += 1
        try:
            client = carla.Client(host, int(port))
            client.set_timeout(client_timeout_s)
            world = client.get_world()
            map_name = world.get_map().name
            print(
                f"[hitl_start_carla] CARLA RPC ready at {host}:{port} "
                f"(map={map_name}, attempt={attempts})"
            )
            return
        except Exception as exc:  # carla raises RuntimeError on RPC failure
            last_exc = exc
            time.sleep(retry_interval_s)

    raise TimeoutError(
        f"CARLA at {host}:{port} not ready after {timeout_s:.1f}s "
        f"({attempts} attempts). Last error: {last_exc!r}"
    )


def stop_carla(handle: CarlaHandle, timeout: float = 15.0) -> None:
    handle.stop(timeout=timeout)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="hitl_start_carla",
        description="Start CARLA for HITL experiments and verify readiness.",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=4010, help="CARLA RPC port (default 4010)")
    p.add_argument(
        "--tm-port-offset",
        type=int,
        default=5,
        help="Traffic manager port = port + offset (default 5)",
    )
    p.add_argument(
        "--carla-root",
        default=None,
        help="Path to CARLA install. Defaults to $CARLA_ROOT or <repo>/carla912.",
    )
    p.add_argument(
        "--carla-arg",
        action="append",
        default=None,
        help=(
            "Extra arg passed to CarlaUE4.sh. May be repeated. "
            "Defaults to ['-RenderOffScreen']."
        ),
    )
    p.add_argument(
        "--ready-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for the RPC round-trip probe (default 60).",
    )
    p.add_argument(
        "--probe-only",
        action="store_true",
        help="Start, probe ready, then immediately stop. Useful for CI/smoke.",
    )
    p.add_argument(
        "--probe-existing",
        action="store_true",
        help=(
            "Do NOT launch CARLA — only run the readiness probe against an "
            "already-running server."
        ),
    )
    p.add_argument(
        "--port-tries",
        type=int,
        default=1,
        help=(
            "How many port bundles to try before giving up. Default 1 means "
            "fail fast if the requested port is busy. Increase to scan."
        ),
    )
    p.add_argument(
        "--port-step",
        type=int,
        default=10,
        help="Spacing between candidate world ports when --port-tries > 1.",
    )
    return p.parse_args(argv)


def _run_cli(args: argparse.Namespace) -> int:
    extra_args: List[str] = list(args.carla_arg) if args.carla_arg is not None else ["-RenderOffScreen"]

    if args.probe_existing:
        try:
            wait_for_carla_ready(
                host=args.host,
                port=args.port,
                timeout_s=args.ready_timeout,
                carla_root=args.carla_root,
            )
        except TimeoutError as exc:
            print(f"[hitl_start_carla] FAILED: {exc}", file=sys.stderr)
            return 2
        print("[hitl_start_carla] OK (existing server passed readiness probe).")
        return 0

    print(
        f"[hitl_start_carla] launching CARLA at {args.host}:{args.port} "
        f"(tm offset={args.tm_port_offset}, extra={extra_args})"
    )
    handle: CarlaHandle | None = None
    try:
        handle = start_carla(
            host=args.host,
            port=args.port,
            tm_port_offset=args.tm_port_offset,
            extra_args=extra_args,
            carla_root=args.carla_root,
            port_tries=args.port_tries,
            port_step=args.port_step,
        )
        print(
            f"[hitl_start_carla] launched. selected port={handle.port}, "
            f"tm_port={handle.tm_port}"
        )

        try:
            wait_for_carla_ready(
                host=handle.host,
                port=handle.port,
                timeout_s=args.ready_timeout,
                carla_root=str(handle.carla_root),
            )
        except TimeoutError as exc:
            print(f"[hitl_start_carla] readiness probe FAILED: {exc}", file=sys.stderr)
            return 3

        if args.probe_only:
            print("[hitl_start_carla] --probe-only: stopping CARLA.")
            return 0

        print(
            "[hitl_start_carla] CARLA is up. Press Ctrl-C to stop. "
            f"(world={handle.host}:{handle.port}, tm={handle.host}:{handle.tm_port})"
        )
        # Hold open until SIGINT/SIGTERM. Sleep in short increments so signals
        # are responsive on Linux.
        stop_requested = {"flag": False}

        def _on_signal(signum, frame):  # noqa: ARG001
            stop_requested["flag"] = True

        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)
        while not stop_requested["flag"]:
            time.sleep(0.5)
        print("[hitl_start_carla] received stop signal.")
        return 0

    finally:
        if handle is not None:
            handle.stop()
            print("[hitl_start_carla] stopped.")


def main(argv: Sequence[str] | None = None) -> int:
    return _run_cli(_parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
