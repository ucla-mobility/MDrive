#!/usr/bin/env python3
"""Helpers for the CARLA root-cause capture wrapper."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import run_custom_eval as run_eval  # noqa: E402


RUN_CUSTOM_EVAL_MARKERS = ("tools/run_custom_eval.py", "run_custom_eval.py")


def _print_kv(key: str, value: object | None) -> None:
    if value is None:
        print(f"{key}=")
        return
    print(f"{key}={value}")


def inspect_eval_cmd(eval_cmd: str) -> int:
    text = eval_cmd or ""
    uses_run_custom_eval = any(marker in text for marker in RUN_CUSTOM_EVAL_MARKERS)
    has_start_carla = bool(re.search(r"(?<!\S)--start-carla(?=\s|$)", text))

    port_matches = [int(match.group(1)) for match in re.finditer(r"(?<!\S)--port\s+(\d+)(?=\s|$)", text)]
    tm_matches = [
        int(match.group(1))
        for match in re.finditer(r"(?<!\S)--traffic-manager-port\s+(\d+)(?=\s|$)", text)
    ]

    eval_port = port_matches[-1] if port_matches else None
    eval_tm_port = tm_matches[-1] if tm_matches else None
    tm_port_offset = None
    if eval_port is not None and eval_tm_port is not None and eval_tm_port != eval_port:
        tm_port_offset = eval_tm_port - eval_port

    _print_kv("uses_run_custom_eval", int(uses_run_custom_eval))
    _print_kv("has_start_carla", int(has_start_carla))
    _print_kv("eval_port", eval_port)
    _print_kv("eval_tm_port", eval_tm_port)
    _print_kv("eval_tm_port_offset", tm_port_offset)
    return 0


def select_port_bundle(
    *,
    host: str,
    preferred_port: int,
    port_tries: int,
    port_step: int,
    tm_port_offset: int,
    stream_port_offset: int,
    cleanup_stale: bool,
    cleanup_grace_s: float,
) -> int:
    desired_ports = run_eval.carla_service_ports(
        preferred_port,
        tm_port_offset,
        stream_port_offset=stream_port_offset,
    )
    desired_available = run_eval.are_ports_available(host, desired_ports)

    cleanup_closed = False
    unmatched_count = 0
    selection_reason = "requested"

    if not desired_available and cleanup_stale:
        cleanup_closed, unmatched = run_eval.cleanup_listening_processes(
            host=host,
            ports=desired_ports,
            service_name="CARLA",
            match_tokens=run_eval.CARLA_PROCESS_MATCH_TOKENS,
            grace_s=cleanup_grace_s,
        )
        unmatched_count = len(unmatched)
        desired_available = run_eval.are_ports_available(host, desired_ports)
        if cleanup_closed and desired_available:
            selection_reason = "requested_after_cleanup"

    if desired_available:
        selected_port = preferred_port
        selected_tm_port = preferred_port + tm_port_offset
        selected_service_ports = run_eval.carla_service_ports(
            selected_port,
            tm_port_offset,
            stream_port_offset=stream_port_offset,
        )
    else:
        bundle = run_eval.find_available_port_bundle(
            host,
            preferred_port,
            port_tries,
            port_step,
            stream_port_offset=stream_port_offset,
            tm_port_offset=tm_port_offset,
        )
        selected_port = int(bundle.port)
        selected_tm_port = int(bundle.tm_port)
        selected_service_ports = run_eval.carla_service_ports(
            selected_port,
            tm_port_offset,
            stream_port_offset=stream_port_offset,
        )
        selection_reason = "next_free"

    _print_kv("selected_port", selected_port)
    _print_kv("selected_tm_port", selected_tm_port)
    _print_kv("selected_service_ports", ",".join(str(port) for port in selected_service_ports))
    _print_kv("selection_reason", selection_reason)
    _print_kv("cleanup_closed", int(cleanup_closed))
    _print_kv("unmatched_count", unmatched_count)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-eval-cmd")
    inspect_parser.add_argument("--eval-cmd", default="", help="Shell command passed to --eval-cmd.")

    select_parser = subparsers.add_parser("select-port-bundle")
    select_parser.add_argument("--host", default="127.0.0.1")
    select_parser.add_argument("--preferred-port", type=int, required=True)
    select_parser.add_argument("--port-tries", type=int, default=8)
    select_parser.add_argument("--port-step", type=int, default=1)
    select_parser.add_argument("--tm-port-offset", type=int, default=run_eval.CARLA_TM_PORT_OFFSET_DEFAULT)
    select_parser.add_argument("--stream-port-offset", type=int, default=run_eval.CARLA_STREAM_PORT_OFFSET_DEFAULT)
    select_parser.add_argument("--cleanup-stale", action="store_true")
    select_parser.add_argument("--cleanup-grace-s", type=float, default=10.0)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect-eval-cmd":
        return inspect_eval_cmd(args.eval_cmd)
    if args.command == "select-port-bundle":
        return select_port_bundle(
            host=args.host,
            preferred_port=args.preferred_port,
            port_tries=args.port_tries,
            port_step=args.port_step,
            tm_port_offset=args.tm_port_offset,
            stream_port_offset=args.stream_port_offset,
            cleanup_stale=args.cleanup_stale,
            cleanup_grace_s=args.cleanup_grace_s,
        )
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
