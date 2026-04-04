#!/usr/bin/env python3
"""Shared CARLA client/process connection event logging helpers."""

from __future__ import annotations

import atexit
import datetime as dt
import os
import signal
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Iterable

import time as _time
import uuid as _uuid

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover - non-POSIX fallback
    fcntl = None

CARLA_CONNECTION_EVENTS_LOG_ENV = "CARLA_CONNECTION_EVENTS_LOG"
CARLA_ROOTCAUSE_LOGDIR_ENV = "CARLA_ROOTCAUSE_LOGDIR"
# Alternate spelling accepted from command-line / env helpers.
CARLA_ROOTCAUSE_DIR_ENV = "CARLA_ROOTCAUSE_DIR"

# Per-run planner / port context — set by the launcher, inherited by children.
CARLA_PLANNER_NAME_ENV = "CARLA_PLANNER_NAME"
CARLA_WORLD_PORT_ENV   = "CARLA_WORLD_PORT"
CARLA_STREAM_PORT_ENV  = "CARLA_STREAM_PORT"   # world_port + 1 by default
CARLA_TM_PORT_ENV      = "CARLA_TM_PORT"

_INSTALL_LOCK = threading.Lock()
_INSTALLED_PROCESSES: set[str] = set()
_WRITE_FAIL_LOCK = threading.Lock()
_WRITE_FAIL_COUNTER = 0

# ---------------------------------------------------------------------------
# Multi-file routing: specialty log files written to the rootcause directory.
# Maps event_type (UPPERCASE) → list of specialty filenames (within rootcause
# dir).  Events not listed here go only to the primary carla_connection_events
# catch-all log.
# ---------------------------------------------------------------------------
_EVENT_SPECIALTY_FILES: "dict[str, list[str]]" = {
    "CLIENT_CREATE":                ["connection_events.log", "rpc_activity.log"],
    "CLIENT_CONNECTED":             ["connection_events.log", "rpc_activity.log"],
    "CLIENT_CONNECT_FAIL":          ["connection_events.log", "rpc_activity.log"],
    "CLIENT_LIFETIME_START":        ["connection_events.log"],
    "CLIENT_LIFETIME_END":          ["connection_events.log"],
    "EVALUATOR_INIT":               ["connection_events.log", "rpc_activity.log"],
    "CLIENT_RPC_READY":             ["connection_events.log", "rpc_activity.log"],
    "CLIENT_RETRY":                 ["connection_events.log", "rpc_activity.log"],
    "PROCESS_START":                ["process_lifecycle.log"],
    "PROCESS_EXIT":                 ["process_lifecycle.log"],
    "PROCESS_ENV":                  ["process_lifecycle.log"],
    "PROCESS_SIGNAL":               ["process_lifecycle.log"],
    "PROCESS_EXCEPTION":            ["process_lifecycle.log"],
    "PROCESS_UNHANDLED_EXCEPTION":  ["process_lifecycle.log"],
    "THREAD_UNHANDLED_EXCEPTION":   ["process_lifecycle.log"],
    "EVALUATOR_CLEANUP_BEGIN":      ["process_lifecycle.log"],
    "EVALUATOR_CLEANUP_END":        ["process_lifecycle.log"],
    "EVALUATOR_TEARDOWN_BEGIN":     ["process_lifecycle.log", "actor_lifecycle.log"],
    "EVALUATOR_TEARDOWN_END":       ["process_lifecycle.log", "actor_lifecycle.log"],
    "EVALUATOR_TEARDOWN_REENTRANT": ["process_lifecycle.log", "actor_lifecycle.log"],
    "EVALUATOR_SENSOR_STOP_BEGIN":  ["actor_lifecycle.log"],
    "EVALUATOR_SENSOR_STOP":        ["actor_lifecycle.log"],
    "EVALUATOR_SENSOR_STOP_END":    ["actor_lifecycle.log"],
    "EVALUATOR_CLIENT_CLOSE_BEGIN": ["connection_events.log", "process_lifecycle.log"],
    "EVALUATOR_CLIENT_CLOSE_END":   ["connection_events.log", "process_lifecycle.log"],
    "EVALUATOR_START_ALLOWED":      ["connection_events.log", "process_lifecycle.log"],
    "EVALUATOR_START_GATE_TIMEOUT": ["connection_events.log", "process_lifecycle.log"],
    "ROUTE_START_GATE_BEGIN":       ["connection_events.log", "process_lifecycle.log"],
    "ROUTE_START_GATE_END":         ["connection_events.log", "process_lifecycle.log"],
    "ROUTE_SENSOR_SWEEP_BEGIN":     ["actor_lifecycle.log"],
    "ROUTE_SENSOR_SWEEP_END":       ["actor_lifecycle.log"],
    "SOCKET_LOGGER_START":          ["process_lifecycle.log"],
    "ACTOR_SPAWN":                  ["actor_lifecycle.log"],
    "ACTOR_DESTROY":                ["actor_lifecycle.log"],
    "ACTOR_DESTROY_BATCH":          ["actor_lifecycle.log"],
    "ACTOR_DESTROY_BATCH_BEGIN":    ["actor_lifecycle.log"],
    "ACTOR_DESTROY_BATCH_RESULT":   ["actor_lifecycle.log"],
    "SENSOR_SPAWN":                 ["actor_lifecycle.log"],
    "SENSOR_DESTROY":               ["actor_lifecycle.log"],
    "SENSOR_LISTENER_DETACH":       ["actor_lifecycle.log"],
    "SENSOR_STOP_BEGIN":            ["actor_lifecycle.log"],
    "SENSOR_STOP_END":              ["actor_lifecycle.log"],
    "SENSOR_DESTROY_BATCH_RESULT":  ["actor_lifecycle.log"],
    "EVALUATOR_START_GATE_BEGIN":   ["connection_events.log"],
    "EVALUATOR_START_GATE_BASELINE_SET":    ["connection_events.log"],
    "EVALUATOR_START_GATE_BASELINE_UPDATE": ["connection_events.log"],
    "EVALUATOR_START_GATE_SAMPLE":  ["connection_events.log"],
    "EVALUATOR_START_GATE_STATE":   ["connection_events.log"],
    "EVALUATOR_START_GATE_END":     ["connection_events.log"],
    "EVALUATOR_START_GATE_SKIP":    ["connection_events.log"],
    # Planner / port context
    "PLANNER_START":                ["planner_info.log", "connection_events.log"],
    "STREAM_PORT":                  ["planner_info.log"],
    # Socket lifecycle deltas — go to both snapshot log and connection log for causality
    "SOCKET_OPEN":                  ["socket_snapshot.log", "connection_events.log"],
    "SOCKET_CLOSE":                 ["socket_snapshot.log", "connection_events.log"],
    "SHORT_LIVED_CONNECTION":       ["socket_snapshot.log", "connection_events.log"],
    # Summary / verbose — snapshot log only (don't flood the catch-all)
    "SOCKET_SUMMARY":               ["socket_snapshot.log"],
}

# Events written ONLY to specialty files, never to the primary catch-all log.
# Individual SOCKET snapshot lines are written directly (not via log_carla_event)
# so they never appear in the catch-all at all.
# SOCKET_SUMMARY is 1 Hz noise — skip the primary log.
_PRIMARY_LOG_SKIP: "frozenset[str]" = frozenset({"SOCKET_SNAPSHOT", "SOCKET_SUMMARY"})

_SOCKET_SNAPSHOT_GUARD = threading.Lock()
_SOCKET_SNAPSHOT_INSTALLED = False

_CLIENT_LIFETIME_LOCK = threading.Lock()
_CLIENT_LIFETIMES: "dict[str, float]" = {}  # client_id → start monotonic


def _sanitize(value: Any) -> str:
    text = str(value)
    text = text.replace("\n", "\\n").replace("\r", "\\r")
    text = text.replace(" ", "_")
    return text


def _candidate_log_paths() -> list[Path]:
    candidates: list[Path] = []
    raw_log = os.environ.get(CARLA_CONNECTION_EVENTS_LOG_ENV, "").strip()
    if raw_log:
        try:
            candidates.append(Path(raw_log).expanduser().resolve())
        except Exception:
            pass
    raw_root = os.environ.get(CARLA_ROOTCAUSE_LOGDIR_ENV, "").strip()
    if raw_root:
        try:
            candidates.append(Path(raw_root).expanduser().resolve() / "carla_connection_events.log")
        except Exception:
            pass
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _write_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, line.encode("utf-8", errors="replace"))
        os.fsync(fd)
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _report_write_failure(path: Path, exc: Exception) -> None:
    global _WRITE_FAIL_COUNTER
    with _WRITE_FAIL_LOCK:
        _WRITE_FAIL_COUNTER += 1
        count = _WRITE_FAIL_COUNTER
    if count <= 5 or count % 100 == 0:
        try:
            sys.stderr.write(
                "[CARLA_EVENT_WRITE_FAIL] "
                f"path={path} count={count} error={type(exc).__name__}: {exc}\n"
            )
            sys.stderr.flush()
        except Exception:
            pass


def log_carla_event(event_type: str, *, process_name: str | None = None, **fields: Any) -> None:
    """Append one CARLA connection/process lifecycle event line, if enabled."""
    ev_upper = event_type.upper()
    paths = _candidate_log_paths()
    has_rootcause = bool(
        os.environ.get(CARLA_ROOTCAUSE_LOGDIR_ENV, "").strip()
        or os.environ.get(CARLA_ROOTCAUSE_DIR_ENV, "").strip()
    )
    if not paths and not has_rootcause:
        return

    timestamp = dt.datetime.now().isoformat(timespec="milliseconds")
    monotonic_s = f"{os.times().elapsed:.3f}"
    pid = os.getpid()
    tid = threading.get_ident()
    proc = process_name or os.environ.get("CARLA_EVENT_PROCESS_NAME", "unknown")
    parts = [
        f"[{timestamp}]",
        f"[pid={pid}]",
        f"[tid={tid}]",
        f"[mono_s={monotonic_s}]",
        f"[process={_sanitize(proc)}]",
        f"[event={_sanitize(event_type)}]",
    ]
    # Auto-inject planner name if the env var is set (set by launcher, inherited by children).
    # Skip if the caller already passed planner= explicitly in fields.
    _planner = os.environ.get(CARLA_PLANNER_NAME_ENV, "").strip()
    if _planner and "planner" not in fields:
        parts.append(f"planner={_sanitize(_planner)}")
    for key in sorted(fields):
        value = fields[key]
        if value is None:
            continue
        parts.append(f"{_sanitize(key)}={_sanitize(value)}")
    line = " ".join(parts) + "\n"
    # Write to primary catch-all log (unless this event is specialty-only).
    if ev_upper not in _PRIMARY_LOG_SKIP:
        for path in paths:
            try:
                _write_line(path, line)
                break
            except Exception as exc:
                _report_write_failure(path, exc)
    # Always write to specialty files in the rootcause directory.
    _write_to_specialty_files(ev_upper, line)


def set_connection_events_logdir(logdir: str | Path, *, process_name: str | None = None) -> Path:
    """Point event logging to <logdir>/carla_connection_events.log and export env vars."""
    root = Path(logdir).expanduser().resolve()
    log_path = root / "carla_connection_events.log"
    # Set both env var spellings so every calling convention works.
    os.environ[CARLA_ROOTCAUSE_LOGDIR_ENV] = str(root)
    os.environ[CARLA_ROOTCAUSE_DIR_ENV]    = str(root)
    os.environ[CARLA_CONNECTION_EVENTS_LOG_ENV] = str(log_path)
    log_carla_event(
        "PROCESS_ENV",
        process_name=process_name,
        rootcause_logdir=str(root),
        events_log=str(log_path),
    )
    return log_path


def install_process_lifecycle_logging(
    process_name: str,
    *,
    env_keys: Iterable[str] | None = None,
) -> None:
    """Install one-time PROCESS_START/PROCESS_ENV/PROCESS_EXIT hooks for this process."""
    with _INSTALL_LOCK:
        if process_name in _INSTALLED_PROCESSES:
            return
        _INSTALLED_PROCESSES.add(process_name)

    os.environ.setdefault("CARLA_EVENT_PROCESS_NAME", process_name)
    log_carla_event("PROCESS_START", process_name=process_name, argv=" ".join(os.sys.argv))

    # Unconditional python/path snapshot so every process is self-describing.
    log_carla_event(
        "PROCESS_ENV",
        process_name=process_name,
        pid=os.getpid(),
        python=sys.executable,
        file=os.path.abspath(sys.argv[0]) if sys.argv else "unknown",
        path_prefix=":".join(os.environ.get("PATH", "").split(":")[:3]),
    )

    if env_keys:
        env_payload = {key: os.environ.get(key, "") for key in env_keys}
        compact = ";".join(f"{key}={_sanitize(value)}" for key, value in sorted(env_payload.items()))
        log_carla_event("PROCESS_ENV", process_name=process_name, env=compact)

    def _on_exit() -> None:
        log_carla_event("PROCESS_EXIT", process_name=process_name)

    atexit.register(_on_exit)

    # Auto-start 1 Hz socket snapshot logger for any instrumented process.
    install_socket_snapshot_logger()

    original_excepthook = sys.excepthook

    def _event_excepthook(exc_type, exc_value, exc_tb):
        tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        log_carla_event(
            "PROCESS_UNHANDLED_EXCEPTION",
            process_name=process_name,
            error_type=getattr(exc_type, "__name__", str(exc_type)),
            error=str(exc_value),
            traceback=tb_text,
        )
        original_excepthook(exc_type, exc_value, exc_tb)

    sys.excepthook = _event_excepthook

    if hasattr(threading, "excepthook"):
        original_threading_excepthook = threading.excepthook

        def _threading_excepthook(args):
            tb_text = "".join(
                traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)
            )
            log_carla_event(
                "THREAD_UNHANDLED_EXCEPTION",
                process_name=process_name,
                thread_name=getattr(args.thread, "name", "unknown"),
                error_type=getattr(args.exc_type, "__name__", str(args.exc_type)),
                error=str(args.exc_value),
                traceback=tb_text,
            )
            original_threading_excepthook(args)

        threading.excepthook = _threading_excepthook

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handler = signal.getsignal(sig)

        def _make_handler(sig_value: int, prev_handler: Any):
            def _handler(signum, frame):
                log_carla_event(
                    "PROCESS_SIGNAL",
                    process_name=process_name,
                    signal=signum,
                    signal_name=getattr(signal.Signals(signum), "name", str(signum)),
                )
                if callable(prev_handler):
                    return prev_handler(signum, frame)
                if prev_handler == signal.SIG_DFL:
                    raise SystemExit(128 + int(sig_value))
                return None

            return _handler

        try:
            signal.signal(sig, _make_handler(int(sig), previous_handler))
        except Exception:
            continue


def log_process_exception(exc: BaseException, *, process_name: str, where: str = "") -> None:
    tb_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    log_carla_event(
        "PROCESS_EXCEPTION",
        process_name=process_name,
        where=where,
        error_type=type(exc).__name__,
        error=str(exc),
        traceback=tb_text,
    )


def create_logged_client(
    carla_module: Any,
    host: str,
    port: int,
    *,
    timeout_s: float | None = None,
    context: str = "",
    attempt: int | None = None,
    process_name: str | None = None,
) -> Any:
    log_carla_event(
        "CLIENT_CREATE",
        process_name=process_name,
        host=host,
        port=int(port),
        context=context,
        attempt=attempt,
    )
    try:
        client = carla_module.Client(host, int(port))
        if timeout_s is not None:
            client.set_timeout(float(timeout_s))
        log_carla_event(
            "CLIENT_CONNECTED",
            process_name=process_name,
            host=host,
            port=int(port),
            context=context,
            attempt=attempt,
            timeout_s=timeout_s,
        )
        return client
    except Exception as exc:
        log_carla_event(
            "CLIENT_CONNECT_FAIL",
            process_name=process_name,
            host=host,
            port=int(port),
            context=context,
            attempt=attempt,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise


def log_client_rpc_ready(
    *,
    host: str,
    port: int,
    context: str = "",
    attempt: int | None = None,
    process_name: str | None = None,
) -> None:
    log_carla_event(
        "CLIENT_RPC_READY",
        process_name=process_name,
        host=host,
        port=int(port),
        context=context,
        attempt=attempt,
    )


def log_client_retry(
    *,
    host: str,
    port: int,
    context: str = "",
    attempt: int | None = None,
    sleep_s: float | None = None,
    reason: str = "",
    process_name: str | None = None,
) -> None:
    log_carla_event(
        "CLIENT_RETRY",
        process_name=process_name,
        host=host,
        port=int(port),
        context=context,
        attempt=attempt,
        sleep_s=sleep_s,
        reason=reason,
    )


# =============================================================================
# Routing helpers (used internally by log_carla_event)
# =============================================================================

def _get_rootcause_dir() -> "Path | None":
    """Return the rootcause directory Path — checks both env var spellings."""
    for env_var in (CARLA_ROOTCAUSE_LOGDIR_ENV, CARLA_ROOTCAUSE_DIR_ENV):
        raw = os.environ.get(env_var, "").strip()
        if raw:
            try:
                return Path(raw).expanduser().resolve()
            except Exception:
                pass
    return None


def _write_to_specialty_files(ev_upper: str, line: str) -> None:
    """Write *line* to every specialty log file associated with *ev_upper*."""
    rootcause_dir = _get_rootcause_dir()
    if rootcause_dir is None:
        return
    for fname in _EVENT_SPECIALTY_FILES.get(ev_upper, []):
        try:
            _write_line(rootcause_dir / fname, line)
        except Exception as exc:
            _report_write_failure(rootcause_dir / fname, exc)


# =============================================================================
# Actor / Sensor lifecycle logging
# =============================================================================

def log_actor_event(
    event: str,
    *,
    actor: "Any | None" = None,
    actor_id: "int | None" = None,
    actor_type: "str | None" = None,
    process_name: "str | None" = None,
    **extra: Any,
) -> None:
    """Log ACTOR_SPAWN, ACTOR_DESTROY, ACTOR_DESTROY_BATCH, etc. to actor_lifecycle.log."""
    aid = actor_id
    atype = actor_type
    if actor is not None:
        if aid is None:
            try:
                aid = actor.id
            except Exception:
                pass
        if atype is None:
            try:
                atype = actor.type_id
            except Exception:
                pass
    log_carla_event(event, process_name=process_name, actor_id=aid, actor_type=atype, **extra)


def log_sensor_event(
    event: str,
    *,
    sensor: "Any | None" = None,
    sensor_id: "int | None" = None,
    sensor_type: "str | None" = None,
    sensor_tag: "str | None" = None,
    process_name: "str | None" = None,
    **extra: Any,
) -> None:
    """Log SENSOR_SPAWN, SENSOR_DESTROY, etc. to actor_lifecycle.log."""
    sid = sensor_id
    stype = sensor_type
    if sensor is not None:
        if sid is None:
            try:
                sid = sensor.id
            except Exception:
                pass
        if stype is None:
            try:
                stype = sensor.type_id
            except Exception:
                pass
    log_carla_event(
        event,
        process_name=process_name,
        sensor_id=sid,
        sensor_type=stype,
        sensor_tag=sensor_tag,
        **extra,
    )


# =============================================================================
# Client lifetime tracking (UUID-based duration measurement)
# =============================================================================

def log_client_lifetime_start(
    *,
    host: str,
    port: int,
    context: str = "",
    process_name: "str | None" = None,
) -> str:
    """
    Record the start of a CARLA client session.
    Returns a short client_id UUID string to pass to log_client_lifetime_end().
    Also writes CLIENT_LIFETIME_START to connection_events.log.
    """
    client_id = str(_uuid.uuid4())[:8]
    with _CLIENT_LIFETIME_LOCK:
        _CLIENT_LIFETIMES[client_id] = _time.monotonic()
    log_carla_event(
        "CLIENT_LIFETIME_START",
        process_name=process_name,
        host=host,
        port=int(port),
        context=context,
        client_id=client_id,
    )
    return client_id


def log_client_lifetime_end(
    client_id: str,
    *,
    process_name: "str | None" = None,
) -> None:
    """
    Record the end of a CARLA client session.
    Computes duration from the matching log_client_lifetime_start() call and
    writes CLIENT_LIFETIME_END to connection_events.log.
    """
    with _CLIENT_LIFETIME_LOCK:
        start_mono = _CLIENT_LIFETIMES.pop(client_id, None)
    duration_s = None
    if start_mono is not None:
        duration_s = round(_time.monotonic() - start_mono, 3)
    log_carla_event(
        "CLIENT_LIFETIME_END",
        process_name=process_name,
        client_id=client_id,
        duration_s=duration_s,
    )


# =============================================================================
# Socket snapshot logger (1 Hz background daemon thread)
# =============================================================================

# =============================================================================
# Socket snapshot logger — structured, delta-aware, 1 Hz
# =============================================================================

def _safe_int(s: str, default: int = 0) -> int:
    try:
        return int(s)
    except (ValueError, TypeError):
        return default


def _split_ss_addr_port(addr_port: str) -> "tuple[str, int | None]":
    """Split 'addr:port' or '[addr]:port' into (ip_str, port_int)."""
    if addr_port.startswith('['):
        bracket_end = addr_port.rfind(']')
        if bracket_end < 0:
            return addr_port, None
        ip = addr_port[1:bracket_end]
        port_str = addr_port[bracket_end + 2:] if len(addr_port) > bracket_end + 2 else ""
    else:
        last_colon = addr_port.rfind(':')
        if last_colon < 0:
            return addr_port, None
        ip = addr_port[:last_colon]
        port_str = addr_port[last_colon + 1:]
    try:
        return ip, int(port_str)
    except ValueError:
        return ip, None


def _parse_ss_line(line: str, relevant_ports: "set[int]") -> "dict | None":
    """
    Parse one line of 'ss -tanp' output into a connection record dict.
    Returns None if the line is a header, irrelevant, or malformed.
    """
    import re as _re
    parts = line.split()
    if not parts or parts[0] in ('Netid', 'State'):
        return None
    # Some ss versions include Netid column (tcp / tcp6 / …), others skip it.
    if parts[0].lower() in ('tcp', 'tcp6', 'udp', 'udp6', 'raw', 'raw6', 'u_str'):
        offset = 1
    else:
        offset = 0
    if len(parts) < offset + 5:
        return None
    state      = parts[offset]
    local_str  = parts[offset + 3]
    peer_str   = parts[offset + 4]
    proc_str   = " ".join(parts[offset + 5:]) if len(parts) > offset + 5 else ""

    local_ip, local_port = _split_ss_addr_port(local_str)
    remote_ip, remote_port = _split_ss_addr_port(peer_str)

    if local_port is None or remote_port is None:
        return None

    # Filter: keep only connections whose local OR remote port is in the set.
    if relevant_ports and local_port not in relevant_ports and remote_port not in relevant_ports:
        return None

    pid: "int | None" = None
    proc_name = ""
    if proc_str:
        m = _re.search(r'"([^"]+)",pid=(\d+)', proc_str)
        if m:
            proc_name = m.group(1)
            pid = int(m.group(2))

    return {
        'state':       state,
        'local_ip':    local_ip,
        'local_port':  local_port,
        'remote_ip':   remote_ip,
        'remote_port': remote_port,
        'pid':         pid,
        'process_name': proc_name,
    }


def _parse_ss_connections(
    ss_output: str,
    relevant_ports: "set[int]",
) -> "dict[tuple, dict]":
    """
    Parse full 'ss -tanp' output.
    Returns dict keyed by (local_ip, local_port, remote_ip, remote_port).
    """
    result = {}
    for line in ss_output.splitlines():
        rec = _parse_ss_line(line, relevant_ports)
        if rec is None:
            continue
        key = (rec['local_ip'], rec['local_port'], rec['remote_ip'], rec['remote_port'])
        result[key] = rec
    return result


def _format_socket_line(
    ts: str,
    pid: int,
    proc_name: str,
    planner: str,
    rec: dict,
) -> str:
    """Format a single SOCKET snapshot line (written directly to socket_snapshot.log)."""
    planner_part = f" planner={_sanitize(planner)}" if planner else ""
    sock_pid = rec['pid'] if rec['pid'] is not None else "unknown"
    return (
        f"[{ts}] [pid={pid}] [{proc_name}]{planner_part} SOCKET"
        f" local_ip={rec['local_ip']}"
        f" local_port={rec['local_port']}"
        f" remote_ip={rec['remote_ip']}"
        f" remote_port={rec['remote_port']}"
        f" state={rec['state']}"
        f" pid={sock_pid}"
        f" process_name={rec['process_name'] or 'unknown'}"
        "\n"
    )


def install_socket_snapshot_logger(interval_s: float = 1.0) -> None:
    """
    Start a 1 Hz daemon thread that provides full forensic socket attribution:

    Per snapshot (written to socket_snapshot.log):
      • One SOCKET line per active connection on world/stream/TM ports.
      • SOCKET_OPEN  — new connection appeared since last snapshot.
      • SOCKET_CLOSE — connection disappeared; includes duration_s.
      • SHORT_LIVED_CONNECTION — connection gone in < 2 s (also → connection_events.log).
      • SOCKET_SUMMARY — connection counts per port category.

    Ports monitored are read from env vars at each tick so they update
    automatically after set_planner_context() is called:
      CARLA_WORLD_PORT, CARLA_STREAM_PORT, CARLA_TM_PORT

    Falls back to the ":20xx" range if no ports are configured yet.
    Safe to call multiple times — installs at most one thread per process.
    """
    global _SOCKET_SNAPSHOT_INSTALLED
    with _SOCKET_SNAPSHOT_GUARD:
        if _SOCKET_SNAPSHOT_INSTALLED:
            return
        _SOCKET_SNAPSHOT_INSTALLED = True

    stop_event = threading.Event()
    atexit.register(stop_event.set)
    self_pid = os.getpid()
    proc_name_at_start = os.environ.get("CARLA_EVENT_PROCESS_NAME", "unknown")

    def _loop() -> None:
        import subprocess as _sp

        prev_connections: "dict[tuple, dict]" = {}
        conn_open_times:  "dict[tuple, float]" = {}  # key -> monotonic time first seen

        while not stop_event.wait(timeout=interval_s):
            rootcause_dir = _get_rootcause_dir()
            if rootcause_dir is None:
                continue

            # Re-read ports from env each tick so they reflect set_planner_context() calls.
            world_port  = _safe_int(os.environ.get(CARLA_WORLD_PORT_ENV,  "0"))
            stream_port = _safe_int(os.environ.get(CARLA_STREAM_PORT_ENV, "0"))
            tm_port     = _safe_int(os.environ.get(CARLA_TM_PORT_ENV,     "0"))
            relevant_ports: "set[int]" = {p for p in (world_port, stream_port, tm_port) if p > 0}

            planner = os.environ.get(CARLA_PLANNER_NAME_ENV, "")
            proc_name = os.environ.get("CARLA_EVENT_PROCESS_NAME", proc_name_at_start)

            try:
                result = _sp.run(
                    ["ss", "-tanp"],
                    capture_output=True,
                    text=True,
                    timeout=3.0,
                )
                now_mono = _time.monotonic()
                ts = dt.datetime.now().isoformat(timespec="milliseconds")

                # If no specific ports configured yet, fall back to broad :20xx range.
                effective_ports = relevant_ports if relevant_ports else set(range(2000, 2100))
                curr_connections = _parse_ss_connections(result.stdout, effective_ports)

                socket_log = rootcause_dir / "socket_snapshot.log"

                # ── Per-connection SOCKET lines ──────────────────────────────
                for rec in curr_connections.values():
                    line = _format_socket_line(ts, self_pid, proc_name, planner, rec)
                    try:
                        _write_line(socket_log, line)
                    except Exception as exc:
                        _report_write_failure(socket_log, exc)

                # ── Delta: SOCKET_OPEN ───────────────────────────────────────
                for key, rec in curr_connections.items():
                    if key not in prev_connections:
                        conn_open_times[key] = now_mono
                        log_carla_event(
                            "SOCKET_OPEN",
                            process_name=proc_name,
                            local_ip=rec['local_ip'],
                            local_port=rec['local_port'],
                            remote_ip=rec['remote_ip'],
                            remote_port=rec['remote_port'],
                            state=rec['state'],
                            socket_pid=rec['pid'],
                            socket_process=rec['process_name'],
                        )

                # ── Delta: SOCKET_CLOSE ──────────────────────────────────────
                for key, rec in prev_connections.items():
                    if key not in curr_connections:
                        open_time = conn_open_times.pop(key, None)
                        duration_s = round(now_mono - open_time, 3) if open_time is not None else None
                        log_carla_event(
                            "SOCKET_CLOSE",
                            process_name=proc_name,
                            local_ip=rec['local_ip'],
                            local_port=rec['local_port'],
                            remote_ip=rec['remote_ip'],
                            remote_port=rec['remote_port'],
                            duration_s=duration_s,
                            socket_pid=rec['pid'],
                            socket_process=rec['process_name'],
                        )
                        if duration_s is not None and duration_s < 2.0:
                            log_carla_event(
                                "SHORT_LIVED_CONNECTION",
                                process_name=proc_name,
                                local_port=rec['local_port'],
                                remote_port=rec['remote_port'],
                                duration_s=duration_s,
                                socket_pid=rec['pid'],
                                socket_process=rec['process_name'],
                            )

                # ── SOCKET_SUMMARY ───────────────────────────────────────────
                def _port_count(port: int) -> int:
                    return sum(
                        1 for k in curr_connections
                        if port and (k[1] == port or k[3] == port)
                    )

                log_carla_event(
                    "SOCKET_SUMMARY",
                    process_name=proc_name,
                    total=len(curr_connections),
                    world_port_conns=_port_count(world_port),
                    stream_port_conns=_port_count(stream_port),
                    tm_port_conns=_port_count(tm_port),
                )

                prev_connections = curr_connections

            except Exception:
                pass

    t = threading.Thread(target=_loop, name="forensic_socket_snapshot", daemon=True)
    t.start()
    log_carla_event(
        "SOCKET_LOGGER_START",
        process_name=os.environ.get("CARLA_EVENT_PROCESS_NAME", "unknown"),
        interval_s=interval_s,
    )


# =============================================================================
# Planner context — set once per run, inherited by all children via env vars
# =============================================================================

def set_planner_context(
    planner_name: str,
    *,
    world_port: int,
    tm_port: int,
    stream_port: "int | None" = None,
    route_id: str = "",
    gpu_id: str = "",
    process_name: "str | None" = None,
) -> None:
    """
    Record the current planner / port context and propagate it to all child
    processes via environment variables.

    Must be called in the launcher process (run_custom_eval.py) before spawning
    evaluator subprocesses so every log line includes planner= automatically.

    Side-effects:
      • Sets CARLA_PLANNER_NAME, CARLA_WORLD_PORT, CARLA_STREAM_PORT,
        CARLA_TM_PORT in os.environ.
      • Logs PLANNER_START  → planner_info.log + connection_events.log
      • Logs STREAM_PORT    → planner_info.log
    """
    if stream_port is None:
        stream_port = world_port + 1

    os.environ[CARLA_PLANNER_NAME_ENV] = str(planner_name)
    os.environ[CARLA_WORLD_PORT_ENV]   = str(world_port)
    os.environ[CARLA_STREAM_PORT_ENV]  = str(stream_port)
    os.environ[CARLA_TM_PORT_ENV]      = str(tm_port)

    log_carla_event(
        "PLANNER_START",
        process_name=process_name,
        planner=planner_name,
        port=world_port,
        stream_port=stream_port,
        tm_port=tm_port,
        route_id=route_id,
        gpu_id=gpu_id,
        pid=os.getpid(),
    )
    log_carla_event(
        "STREAM_PORT",
        process_name=process_name,
        stream_port=stream_port,
        world_port=world_port,
    )
